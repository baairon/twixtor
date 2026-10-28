//! Goldens of the fast path on hand built clips of tiny frames at 8, 16 and 32 bpc.
//!
//! The expected frames come from `reference`, a transcription of the premultiply, blend and
//! unpremultiply formulas that shares no code with the engine. `retime/tests/retime.rs`
//! checks the same scenarios against literal values produced by an independent numpy float32
//! script.
//!
//! Clip frames are numbered by their index; `Case::at` gives the transparent frame outside the
//! clip. Output frame `n` of a 30 fps composition has `current_time = 100 * n`.

use super::*;
use crate::host::test_support::settings;
use crate::host::{EnvHost, LayerIds};
use crate::image::Fmt;
use crate::timeline::test_support::{fixed, Mock};
use crate::timeline::RemapTimeMap;
use fx_core::{Depth, Env, Frame, Layer, LayerSet, MemLayer, ParamCurves, Pixels, Rational, TimeBase};
use std::sync::Arc;

mod reference {
    use fx_core::{Frame, Pixels};

    /// `q(x) = trunc((x as f64) + 0.5) as i64`, the rounding of the conversions below.
    fn q(x: f32) -> i64 {
        ((x as f64) + 0.5) as i64
    }

    /// Wire `[r, g, b, a]` to engine `[A, R, G, B]`, 8 bpc.
    fn pre8([r, g, b, a]: [u8; 4]) -> [u8; 4] {
        let af = a as f32 * 0.003921568859368563f32;
        [a, q(r as f32 * af) as u8, q(g as f32 * af) as u8, q(b as f32 * af) as u8]
    }

    /// Engine `[A, R, G, B]` to wire `[r, g, b, a]`, 8 bpc.
    fn unpre8([a, r, g, b]: [u8; 4]) -> [u8; 4] {
        if a == 0 {
            return [r, g, b, a];
        }
        let s = 255.0f32 / a as f32;
        let f = |c: u8| {
            let mut v = c as f32 * s;
            if 255.0f32 < v {
                v = 255.0;
            }
            q(v) as u8
        };
        [f(r), f(g), f(b), a]
    }

    fn blend8(a: [u8; 4], b: [u8; 4], w: f32) -> [u8; 4] {
        std::array::from_fn(|i| ((((b[i] as f32 - a[i] as f32) * w) + a[i] as f32) + 0.5f32) as i64 as u8)
    }

    /// Wire range 0..=65535 to the range 0..=32768 (round half up).
    fn ae(v: u16) -> u16 {
        ((v as u32 * 32768 + 32767) / 65535) as u16
    }

    /// The range 0..=32768 back to the wire (round half up, saturating at white).
    fn wire(v: u16) -> u16 {
        ((v.min(32768) as u32 * 65535 + 16384) / 32768) as u16
    }

    /// 16 bpc: wire to engine range 0..=65535.
    fn pre16(px: [u16; 4]) -> [u16; 4] {
        let [r, g, b, a] = px.map(ae);
        let af = a as f32 * 6.1034224927425385e-05f32;
        [q(a as f32 * 1.999969482421875f32) as u16, q(r as f32 * af) as u16, q(g as f32 * af) as u16, q(b as f32 * af) as u16]
    }

    /// 16 bpc: engine to the range 0..=32768, then to the wire.
    fn unpre16([a, r, g, b]: [u16; 4]) -> [u16; 4] {
        let out = if a == 0 {
            [a, r, g, b]
        } else {
            let s = 32768.0f32 / a as f32;
            [q(a as f32 * 0.5000076293945312f32) as u16, q(r as f32 * s) as u16, q(g as f32 * s) as u16, q(b as f32 * s) as u16]
        };
        let [a, r, g, b] = out.map(wire);
        [r, g, b, a]
    }

    fn blend16(a: [u16; 4], b: [u16; 4], w: f32) -> [u16; 4] {
        std::array::from_fn(|i| ((((b[i] as f32 - a[i] as f32) * w) + a[i] as f32) + 0.5f32) as i64 as u16)
    }

    /// 32 bpc: wire `[r, g, b, a]` to engine `[A, R, G, B]`, colour times alpha.
    fn pre32([r, g, b, a]: [f32; 4]) -> [f32; 4] {
        [a, a * r, a * g, a * b]
    }

    /// 32 bpc: engine `[A, R, G, B]` to wire `[r, g, b, a]`; all zero when alpha is below 1e-12.
    fn unpre32([a, r, g, b]: [f32; 4]) -> [f32; 4] {
        if (a as f64) >= 1e-12 {
            let inv = 1.0f32 / a;
            [inv * r, inv * g, inv * b, a]
        } else {
            [0.0; 4]
        }
    }

    fn blend32(a: [f32; 4], b: [f32; 4], w: f32) -> [f32; 4] {
        std::array::from_fn(|i| (((b[i] - a[i]) * w) + a[i]) + 0.0f32)
    }

    fn rebuild(f: &Frame, pixels: Pixels) -> Frame {
        Frame { width: f.width, height: f.height, pixels }
    }

    /// The premultiply / unpremultiply round trip of one frame.
    pub fn round_trip(f: &Frame) -> Frame {
        rebuild(
            f,
            match &f.pixels {
                Pixels::U8(px) => Pixels::U8(px.iter().map(|&p| unpre8(pre8(p))).collect()),
                Pixels::U16(px) => Pixels::U16(px.iter().map(|&p| unpre16(pre16(p))).collect()),
                Pixels::F32(px) => Pixels::F32(px.iter().map(|&p| unpre32(pre32(p))).collect()),
            },
        )
    }

    /// `blend(A, B, w)` between the premultiplied frames, then the finish.
    pub fn blend(a: &Frame, b: &Frame, w: f32) -> Frame {
        rebuild(
            a,
            match (&a.pixels, &b.pixels) {
                (Pixels::U8(x), Pixels::U8(y)) => {
                    Pixels::U8(x.iter().zip(y).map(|(&p, &q)| unpre8(blend8(pre8(p), pre8(q), w))).collect())
                }
                (Pixels::U16(x), Pixels::U16(y)) => {
                    Pixels::U16(x.iter().zip(y).map(|(&p, &q)| unpre16(blend16(pre16(p), pre16(q), w))).collect())
                }
                (Pixels::F32(x), Pixels::F32(y)) => {
                    Pixels::F32(x.iter().zip(y).map(|(&p, &q)| unpre32(blend32(pre32(p), pre32(q), w))).collect())
                }
                _ => panic!("mixed depths"),
            },
        )
    }
}

fn clock() -> TimeBase {
    TimeBase::default_for_rate(Rational::new(30, 1).unwrap())
}

/// A clip of four frames at one depth.
struct Case {
    depth: Depth,
    frames: Vec<Frame>,
}

fn f8(px: &[[u8; 4]]) -> Frame {
    Frame { width: 3, height: 2, pixels: Pixels::U8(px.to_vec()) }
}

fn f16(px: &[[u16; 4]]) -> Frame {
    Frame { width: 2, height: 2, pixels: Pixels::U16(px.to_vec()) }
}

fn f32s(px: &[[f32; 4]]) -> Frame {
    Frame { width: 2, height: 2, pixels: Pixels::F32(px.to_vec()) }
}

/// Opaque, translucent, alpha 1 and alpha 0 pixels (with colour that the engine loses) at every depth.
fn cases() -> Vec<Case> {
    vec![
        Case {
            depth: Depth::U8,
            frames: vec![
                f8(&[[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 128], [200, 100, 50, 64], [90, 180, 255, 0], [17, 34, 51, 255]]),
                f8(&[[0, 255, 255, 255], [255, 0, 255, 200], [10, 20, 30, 1], [255, 255, 255, 254], [1, 2, 3, 128], [100, 100, 100, 100]]),
                f8(&[[128, 64, 32, 255], [33, 66, 99, 33], [250, 5, 130, 177], [0, 0, 0, 0], [64, 128, 192, 255], [255, 0, 0, 3]]),
                f8(&[[12, 200, 77, 255], [0, 0, 0, 255], [255, 255, 255, 255], [170, 85, 10, 90], [5, 5, 5, 5], [240, 120, 60, 128]]),
            ],
        },
        Case {
            depth: Depth::U16,
            frames: vec![
                f16(&[[65535, 0, 0, 65535], [0, 65535, 0, 40000], [12345, 23456, 34567, 1], [50000, 40000, 30000, 0]]),
                f16(&[[1000, 2000, 3000, 65535], [65535, 65535, 65535, 32768], [0, 0, 65535, 65534], [777, 888, 999, 12345]]),
                f16(&[[32768, 16384, 8192, 65535], [40000, 50000, 60000, 50000], [1, 2, 3, 4], [65535, 0, 65535, 500]]),
                f16(&[[0, 0, 0, 65535], [21845, 43690, 65535, 60000], [65535, 65535, 65535, 65535], [9, 99, 999, 9999]]),
            ],
        },
        Case {
            depth: Depth::F32,
            frames: vec![
                f32s(&[[1.0, 0.5, 0.25, 1.0], [0.2, 0.4, 0.6, 0.5], [0.9, 0.8, 0.7, 0.0], [0.3, 0.6, 0.9, 1e-13]]),
                f32s(&[[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.25], [-0.25, 0.5, 1.5, 0.75], [0.125, 0.375, 0.625, 0.875]]),
                f32s(&[[0.5, 0.5, 0.5, 0.5], [0.0, 1.0, 0.0, 1.0], [2.0, 3.0, 4.0, 1.0], [0.1, 0.2, 0.3, 0.4]]),
                f32s(&[[0.75, 0.25, 0.125, 1.0], [0.6, 0.7, 0.8, 0.9], [1.0, 1.0, 1.0, 1.0], [0.01, 0.02, 0.03, 0.04]]),
            ],
        },
    ]
}

impl Case {
    fn size(&self) -> (usize, usize) {
        (self.frames[0].width, self.frames[0].height)
    }

    fn layer(&self) -> MemLayer {
        MemLayer::new(clock(), self.frames.clone()).unwrap()
    }

    fn settings(&self) -> EngineSettings {
        let (w, h) = self.size();
        settings(w, h, Fmt::from_depth(self.depth))
    }

    /// Clip frame `i`, the transparent frame before and after the clip.
    fn at(&self, i: i64) -> Frame {
        usize::try_from(i).ok().and_then(|i| self.frames.get(i)).cloned().unwrap_or_else(|| {
            let (w, h) = self.size();
            Frame::new(w, h, self.depth)
        })
    }

    /// The expected output when frame `i` is passed through the engine untouched.
    fn rt(&self, i: i64) -> Frame {
        reference::round_trip(&self.at(i))
    }

    /// The expected output of `blend(i, j, w)`.
    fn mix(&self, i: i64, j: i64, w: f32) -> Frame {
        reference::blend(&self.at(i), &self.at(j), w)
    }

    fn transparent(&self) -> Frame {
        self.rt(-1)
    }
}

fn run(input: &dyn Layer, layers: &LayerSet, s: &EngineSettings, mock: &Mock) -> Result<Rendered> {
    let params = ParamCurves::new(Vec::new());
    let env = Env { params: &params, input, layers };
    let map = RemapTimeMap::new(mock, s.rates, s.current_time);
    let host = EnvHost::new(&env, s, &map, LayerIds::default())?;
    render(s, &host)
}

/// Output frame `n` of the case with the given time parameters, as a wire frame.
fn out_with(case: &Case, mock: Mock, n: i32, tweak: impl Fn(&mut EngineSettings)) -> Result<Frame> {
    let mut s = case.settings();
    s.current_time = 100 * n;
    tweak(&mut s);
    let layer = case.layer();
    run(&layer, &LayerSet::new(), &s, &mock).map(Rendered::into_frame)
}

fn out(case: &Case, mock: Mock, n: i32) -> Frame {
    out_with(case, mock, n, |_| {}).unwrap()
}

fn out_interp(case: &Case, mock: Mock, n: i32, interp: i32) -> Frame {
    out_with(case, mock, n, |s| s.interp = interp).unwrap()
}

/// The texture of the vector path checks: strong gradients in x and y that are not parallel, a
/// different phase per colour channel, so a 3 x 2 or 2 x 2 frame solves to a non zero flow at
/// Sensitivity 100 and the vector path gives other bytes than the frame blend.
fn texture(c: usize, x: f64, y: f64) -> f64 {
    let p = 2.3 * c as f64;
    0.5 + 0.45 * (2.1 * x + 0.4 + p).sin() * (1.7 * y - 0.2 + 0.5 * p).cos()
}

/// Clips of four opaque frames of [`texture`] moving by (0.6, 0.35) pixels per frame (a sub-pixel
/// shift between consecutive frames), at 3 x 2 and 2 x 2 for every depth. The golden cases above
/// solve to a zero flow (one pyramid level at Sensitivity 70), whose warps give the bytes of the
/// frame blend, so they cannot tell the vector path from a fall back to the fast path.
fn moving_cases() -> Vec<Case> {
    let mut out = Vec::new();
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        for (w, h) in [(3usize, 2usize), (2, 2)] {
            let frames = (0..4)
                .map(|k| {
                    let mut f = Frame::new(w, h, depth);
                    for y in 0..h {
                        for x in 0..w {
                            let (px, py) = (x as f64 - 0.6 * k as f64, y as f64 - 0.35 * k as f64);
                            let [r, g, b] = [texture(0, px, py), texture(1, px, py), texture(2, px, py)];
                            let q8 = |v: f64| (v * 255.0 + 0.5) as u8;
                            let q16 = |v: f64| (v * 65535.0 + 0.5) as u16;
                            match &mut f.pixels {
                                Pixels::U8(p) => p[y * w + x] = [q8(r), q8(g), q8(b), 255],
                                Pixels::U16(p) => p[y * w + x] = [q16(r), q16(g), q16(b), 65535],
                                Pixels::F32(p) => p[y * w + x] = [r as f32, g as f32, b as f32, 1.0],
                            }
                        }
                    }
                    f
                })
                .collect();
            out.push(Case { depth, frames });
        }
    }
    out
}

/// Sensitivity 100: a one level pyramid keeps its flow (at the default 70 it solves to zero).
fn sensitive(s: &mut EngineSettings) {
    s.sensitivity = 1.0;
}

/// The selection of output frame 1 at half speed: between frames 0 and 1, `wA = 0.5`.
fn half(vectors_quality: i32, interp: i32) -> Selection {
    Selection { s0: 0.5, s1: 1.0, a: 0, b: 1, wa: 0.5, exact: false, vectors_quality, interp }
}

/// The selection the core makes for `s` (only to check the literal selections of the tests).
fn core_selection(case: &Case, mock: Mock, s: &EngineSettings) -> Selection {
    let layer = case.layer();
    let params = ParamCurves::new(Vec::new());
    let layers = LayerSet::new();
    let env = Env { params: &params, input: &layer, layers: &layers };
    let map = RemapTimeMap::new(&mock, s.rates, s.current_time);
    let host = EnvHost::new(&env, s, &map, LayerIds::default()).unwrap();
    select_frames(s, &host).unwrap()
}

/// The vector path computed from explicit inputs: frames `sel.a` and `sel.b` of the case
/// (`Image::from_frame`), the selection `sel`, [`solve_flows`] and [`compose_vectors`], then the
/// finish. It shares no routing with [`retime`].
fn direct(case: &Case, s: &EngineSettings, sel: Selection) -> Frame {
    let a = Image::from_frame(&case.at(i64::from(sel.a)));
    let b = Image::from_frame(&case.at(i64::from(sel.b)));
    let flows = solve_flows(s, &sel, &a, &b).unwrap();
    Rendered::Image(compose_vectors(s, &sel, &a, &b, &flows).unwrap()).into_frame()
}

/// Output frame `n` with `tweak` is the vector path: the selection of the core is `sel`, the frame is
/// [`direct`] for `sel`, and it is not the No Vectors render of the same input (the moving cases make
/// the two differ), so a silent fall back to the fast path fails here.
fn assert_vector_path(case: &Case, mock: impl Fn() -> Mock, n: i32, tweak: impl Fn(&mut EngineSettings), sel: Selection) {
    let what = format!("{:?} {:?} frame {n}", case.depth, case.size());
    let mut s = case.settings();
    s.current_time = 100 * n;
    tweak(&mut s);
    assert_eq!(core_selection(case, mock(), &s), sel, "{what}: the selection");
    let got = out_with(case, mock(), n, &tweak).unwrap_or_else(|e| panic!("{what}: {e}"));
    assert_eq!(got, direct(case, &s, sel), "{what}: the vector path");
    let plain = out_with(case, mock(), n, |t| {
        tweak(t);
        t.vectors_quality = 0;
    })
    .unwrap();
    assert_ne!(got, plain, "{what}: the No Vectors render");
}

#[test]
fn speed_100_percent_on_exact_frames_is_the_documented_round_trip() {
    for case in cases() {
        for n in 0..6 {
            assert_eq!(out(&case, Mock::speed(100.0), n), case.rt(n as i64), "{:?} frame {n}", case.depth);
        }
        // frames of the clip beyond its end are transparent
        assert_eq!(out(&case, Mock::speed(100.0), 4), case.transparent());
    }
}

#[test]
fn opaque_pixels_survive_the_round_trip_exactly_and_the_others_do_not() {
    for case in cases() {
        let got = out(&case, Mock::speed(100.0), 0);
        assert_eq!(got, case.rt(0));
        // pixel 0 of frame 0 is opaque at every depth and comes back bit for bit
        match (&got.pixels, &case.frames[0].pixels) {
            (Pixels::U8(g), Pixels::U8(f)) => assert_eq!(g[0], f[0]),
            (Pixels::U16(g), Pixels::U16(f)) => assert_eq!(g[0], f[0]),
            (Pixels::F32(g), Pixels::F32(f)) => assert_eq!(g[0], f[0]),
            _ => unreachable!(),
        }
        // hand computed (numpy float32 script): the lossy pixels
        match &got.pixels {
            // (200, 100, 50, 64) premultiplies to (50, 25, 13) and comes back as (199, 100, 52); alpha 0 loses its colour
            Pixels::U8(g) => {
                assert_eq!(g[3], [199, 100, 52, 64]);
                assert_eq!(g[4], [0, 0, 0, 0]);
            }
            // alpha 40000 -> 39999; alpha 1 is destroyed: (12345, 23456, 34567, 1) -> (0, 32768, 32768, 2)
            Pixels::U16(g) => {
                assert_eq!(g[1], [0, 65535, 0, 39999]);
                assert_eq!(g[2], [0, 32768, 32768, 2]);
                assert_eq!(g[3], [0, 0, 0, 0]);
            }
            // alpha 0 and alpha below 1e-12 become all zero
            Pixels::F32(g) => assert_eq!((g[2], g[3]), ([0.0; 4], [0.0; 4])),
        }
    }
}

#[test]
fn half_speed_by_hand_at_8_bpc() {
    // Output frame 1 at half speed is the middle of clip frames 0 and 1 (w = 0.5), worked out by
    // hand with f32 arithmetic and q(x) = trunc(x + 0.5):
    //   pixel 0, F0 (255, 0, 0, 255) and F1 (0, 255, 255, 255) are opaque, so premultiply is the
    //     identity: R (0 - 255) * 0.5 + 255 + 0.5 = 128.0, G (255 - 0) * 0.5 + 0 + 0.5 = 128.0, B
    //     the same, A 255.5 -> 255: (128, 128, 128, 255), unpremultiply by 255 / 255 = 1 keeps
    //     it.
    //   pixel 5, F0 (17, 34, 51, 255) and F1 (100, 100, 100, 100): F1 premultiplies with 100 *
    //     0.003921569 = 0.39215687 to (39, 39, 39) at alpha 100; the blend is A (100 - 255) * 0.5
    //     + 255 + 0.5 = 178.0, R (39 - 17) * 0.5 + 17 + 0.5 = 28.5 -> 28, G (39 - 34) * 0.5 + 34
    //     + 0.5 = 37.0, B (39 - 51) * 0.5 + 51 + 0.5 = 45.5 -> 45; the finish scales by 255 / 178
    //     = 1.4325843 and gives 40.6 -> 40, 53.5 -> 53, 64.9 -> 64: (40, 53, 64, 178).
    let case = &cases()[0];
    for interp in [1, 2] {
        let got = out_interp(case, Mock::speed(50.0), 1, interp);
        let Pixels::U8(px) = &got.pixels else { unreachable!() };
        assert_eq!(px[0], [128, 128, 128, 255]);
        assert_eq!(px[5], [40, 53, 64, 178]);
    }
    // Nearest takes frame 0 at the tie: (17, 34, 51, 255) comes back unchanged
    let near = out_interp(case, Mock::speed(50.0), 1, 0);
    let Pixels::U8(px) = &near.pixels else { unreachable!() };
    assert_eq!((px[0], px[5]), ([255, 0, 0, 255], [17, 34, 51, 255]));
}

#[test]
fn half_speed_nearest_blend_and_weighted_blend_on_odd_frames() {
    for case in cases() {
        for n in 0..10i32 {
            let k = (n / 2) as i64;
            let on_frame = n % 2 == 0;
            // Nearest: wa == 0.5 on the odd frames picks A
            assert_eq!(out_interp(&case, Mock::speed(50.0), n, 0), case.rt(k), "{:?} nearest {n}", case.depth);
            for interp in [1, 2] {
                let want = if on_frame { case.rt(k) } else { case.mix(k, k + 1, 0.5) };
                assert_eq!(out_interp(&case, Mock::speed(50.0), n, interp), want, "{:?} interp {interp} frame {n}", case.depth);
            }
        }
    }
}

#[test]
fn quarter_speed_uses_the_fraction_as_the_weight_and_nearest_switches_at_one_half() {
    for case in cases() {
        // x = n / 4: n = 1 has wa = 0.75 (A), n = 2 is the tie (A), n = 3 has wa = 0.25 (B)
        let nearest = |n| out_interp(&case, Mock::speed(25.0), n, 0);
        assert_eq!((nearest(1), nearest(2), nearest(3)), (case.rt(0), case.rt(0), case.rt(1)));
        let blend = |n| out_interp(&case, Mock::speed(25.0), n, 1);
        assert_eq!(blend(1), case.mix(0, 1, 0.25));
        assert_eq!(blend(2), case.mix(0, 1, 0.5));
        assert_eq!(blend(3), case.mix(0, 1, 0.75));
        assert_eq!(blend(4), case.rt(1));
        let weighted = |n| out_interp(&case, Mock::speed(25.0), n, 2);
        assert_eq!((weighted(1), weighted(2), weighted(3)), (blend(1), blend(2), blend(3)));
    }
}

#[test]
fn a_fraction_that_is_not_a_power_of_two_gives_the_rounded_single_precision_weight() {
    // 30 percent: the first half step adds 0.15f32 twice, so x = 0.3f32, wa = 1 - 0.3f32 = 0.7f32 (exactly 11744051 * 2^-24)
    // and the Blend weight (1.0 - wa as f64) as f32 is 0.3f32 again.
    for case in cases() {
        assert_eq!(out_interp(&case, Mock::speed(30.0), 1, 1), case.mix(0, 1, 0.3f32));
    }
}

#[test]
fn frame_number_mode_shows_the_chosen_frame_for_every_output_frame() {
    for case in cases() {
        for n in [0, 3, 9] {
            assert_eq!(out_interp(&case, Mock::frame_number(2.5), n, 1), case.mix(2, 3, 0.5));
            assert_eq!(out_interp(&case, Mock::frame_number(2.5), n, 0), case.rt(2), "the tie picks A");
            assert_eq!(out_interp(&case, Mock::frame_number(2.5), n, 2), case.mix(2, 3, 0.5));
            assert_eq!(out(&case, Mock::frame_number(2.0), n), case.rt(2));
            assert_eq!(out(&case, Mock::frame_number(9.0), n), case.transparent(), "beyond the clip");
        }
        // Src Time Offset is not read in Frame Number mode: base stays 0
        let mock = Mock::frame_number(1.0).with_offset(|_| fixed(5.0));
        assert_eq!(out(&case, mock, 0), case.rt(1));
    }
}

#[test]
fn a_negative_position_extrapolates_between_frames_zero_and_one() {
    // Frame Num -0.5: a = trunc(-0.5) = 0, wa = 1 - (-0.5) = 1.5, so the Blend weight is -0.5 and
    // `blend` takes its extrapolation branch on frames 0 and 1. The expected image is built with
    // the blend primitive itself.
    for case in cases() {
        let layer = case.layer();
        let (a, b) = (crate::Image::from_frame(&layer.frame_at(0).unwrap().unwrap()), crate::Image::from_frame(&layer.frame_at(100).unwrap().unwrap()));
        let mut want = crate::Image::new(a.width, a.height, a.fmt());
        crate::blend::blend(&mut want, &a, &b, -0.5);
        assert_eq!(out(&case, Mock::frame_number(-0.5), 0), want.to_frame(), "{:?}", case.depth);
        assert_eq!(out_interp(&case, Mock::frame_number(-0.5), 0, 0), case.rt(0), "Nearest picks A");
    }
}

#[test]
fn a_positive_src_time_offset_looks_ahead_and_a_negative_one_delays_the_clip() {
    for case in cases() {
        // Speed 100 percent, offset +2 frames: output n shows clip frame n + 2
        let ahead = |n| out(&case, Mock::speed(100.0).with_offset(|_| fixed(2.0)), n);
        assert_eq!((ahead(0), ahead(1), ahead(2)), (case.rt(2), case.rt(3), case.transparent()));
        // offset -2 frames: base = 2, frame_time(f) = (f - 2) * 100, the clip starts at output frame 2
        let late = |n| out(&case, Mock::speed(100.0).with_offset(|_| fixed(-2.0)), n);
        assert_eq!(late(0), case.transparent());
        assert_eq!(late(1), case.transparent());
        assert_eq!(late(2), case.rt(0));
        assert_eq!(late(3), case.rt(1));
        assert_eq!(late(5), case.rt(3));
        assert_eq!(late(6), case.transparent());
        // the whole number part is truncated toward zero: -1.9 frames delays by one
        let trunc = |n| out(&case, Mock::speed(100.0).with_offset(|_| fixed(-1.9)), n);
        assert_eq!((trunc(0), trunc(1), trunc(2)), (case.transparent(), case.rt(0), case.rt(1)));
        // half speed with offset -1: output 1 is x = 0.5, frames 0 (time -100, missing) and 1 (time 0, clip frame 0)
        let half = out(&case, Mock::speed(50.0).with_offset(|_| fixed(-1.0)), 1);
        assert_eq!(half, reference::blend(&case.transparent(), &case.at(0), 0.5));
    }
}

#[test]
fn frames_beyond_the_clip_end_are_transparent() {
    for case in cases() {
        for n in 4..8 {
            assert_eq!(out(&case, Mock::speed(100.0), n), case.transparent());
            assert_eq!(out_interp(&case, Mock::speed(100.0), n, 0), case.transparent());
        }
        // between the last frame and the end: blend with the transparent frame
        assert_eq!(out(&case, Mock::speed(50.0), 7), case.mix(3, 4, 0.5));
    }
}

#[test]
fn negative_speed_plays_backwards_from_the_offset() {
    for case in cases() {
        let mock = || Mock::speed(-100.0).with_offset(|_| fixed(3.0));
        let got: Vec<Frame> = (0..5).map(|n| out(&case, mock(), n)).collect();
        assert_eq!(got, vec![case.rt(3), case.rt(2), case.rt(1), case.rt(0), case.transparent()]);
        // reverse half speed: x = -n / 2, the odd frames lie between two frames (negative fractions
        // use frames trunc(x) and trunc(x) + 1 with a negative frac, the quirk is exercised by the
        // extrapolation test); the even frames are exact
        let half = |n| out(&case, Mock::speed(-50.0).with_offset(|_| fixed(3.0)), n);
        assert_eq!((half(0), half(2), half(4), half(6)), (case.rt(3), case.rt(2), case.rt(1), case.rt(0)));
    }
}

#[test]
fn display_source_is_the_round_trip_of_the_frame_at_the_offset() {
    for case in cases() {
        let source = |n, offset: f64| {
            out_with(&case, Mock::speed(50.0), n, |s| {
                s.display_source = true;
                s.src_time_offset_fixed = fixed(offset);
            })
            .unwrap()
        };
        // current_time + trunc(offset) * time_step: no retiming, the speed is ignored
        assert_eq!((source(0, 0.0), source(1, 0.0), source(3, 0.0)), (case.rt(0), case.rt(1), case.rt(3)));
        assert_eq!((source(0, 2.0), source(1, 2.0), source(2, 2.0)), (case.rt(2), case.rt(3), case.transparent()));
        assert_eq!((source(0, 2.9), source(1, -1.0)), (case.rt(2), case.rt(0)));
        assert_eq!(source(0, -1.0), case.transparent());
    }
}

/// The error fill as a wire frame: size mismatch is half alpha red, a missing layer half alpha
/// blue (plugin 01). In the range 0..=32768, 0x4000 is 32768 on the wire and 0x8000 is 65535.
fn fill(depth: Depth, red: bool, width: usize, height: usize) -> Frame {
    let n = width * height;
    let pixels = match (depth, red) {
        (Depth::U8, true) => Pixels::U8(vec![[255, 0, 0, 128]; n]),
        (Depth::U8, false) => Pixels::U8(vec![[0, 0, 255, 128]; n]),
        (Depth::U16, true) => Pixels::U16(vec![[65535, 0, 0, 32768]; n]),
        (Depth::U16, false) => Pixels::U16(vec![[0, 0, 65535, 32768]; n]),
        (Depth::F32, true) => Pixels::F32(vec![[1.0, 0.0, 0.0, 0.5]; n]),
        (Depth::F32, false) => Pixels::F32(vec![[0.0, 0.0, 1.0, 0.5]; n]),
    };
    Frame { width, height, pixels }
}

#[test]
fn a_layer_of_the_wrong_size_gives_the_red_fill_and_a_zero_size_layer_the_blue_one() {
    for case in cases() {
        // the clip is 3x2 (8 bpc) or 2x2, the output is declared with another width
        let (w, h) = case.size();
        let layer = case.layer();
        let mut s = case.settings();
        s.width = if w == 3 { 2 } else { 3 };
        let got = run(&layer, &LayerSet::new(), &s, &Mock::speed(100.0)).unwrap().into_frame();
        assert_eq!(got, fill(case.depth, true, s.width, h), "{:?}", case.depth);

        // a zero size input: blue, at the output size of the settings
        let empty = MemLayer::new(clock(), vec![Frame::new(0, 0, case.depth)]).unwrap();
        let s = case.settings();
        let got = run(&empty, &LayerSet::new(), &s, &Mock::speed(100.0)).unwrap().into_frame();
        assert_eq!(got, fill(case.depth, false, w, h), "{:?}", case.depth);
    }
}

/// A layer that declares the output size but hands out a frame of another size from time 100 on.
struct Odd {
    good: MemLayer,
    bad: fn(Depth) -> Frame,
}

impl Layer for Odd {
    fn width(&self) -> usize {
        self.good.width()
    }
    fn height(&self) -> usize {
        self.good.height()
    }
    fn depth(&self) -> Depth {
        self.good.depth()
    }
    fn frame_at(&self, time: i64) -> fx_core::Result<Option<Arc<Frame>>> {
        if time >= 100 {
            Ok(Some(Arc::new((self.bad)(self.good.depth()))))
        } else {
            self.good.frame_at(time)
        }
    }
}

#[test]
fn a_fetched_frame_of_the_wrong_size_shows_the_fill_too() {
    for case in cases() {
        let (w, h) = case.size();
        let s = case.settings();
        // frame b (time 100) of output frame 0 has another size: red, although frame a is fine
        let odd = Odd { good: case.layer(), bad: |d| Frame::new(5, 1, d) };
        let got = run(&odd, &LayerSet::new(), &s, &Mock::speed(100.0)).unwrap().into_frame();
        assert_eq!(got, fill(case.depth, true, w, h));
        // a zero sized frame is a missing layer: blue
        let odd = Odd { good: case.layer(), bad: |d| Frame::new(0, 4, d) };
        let got = run(&odd, &LayerSet::new(), &s, &Mock::speed(100.0)).unwrap().into_frame();
        assert_eq!(got, fill(case.depth, false, w, h));
        // Display == Source fetches through the same checks
        let mut s = case.settings();
        s.display_source = true;
        s.current_time = 100;
        let odd = Odd { good: case.layer(), bad: |d| Frame::new(5, 1, d) };
        let got = run(&odd, &LayerSet::new(), &s, &Mock::speed(100.0)).unwrap().into_frame();
        assert_eq!(got, fill(case.depth, true, w, h));
    }
}

/// The frames of these cases are 3 x 2 and 2 x 2. The motion search never halves below 2 x 2 (the tiny
/// frames rule, parity gap G7), so every quality searches them at their own size. A position between
/// frames with Vectors on takes the vector path (`retime` calls `vector::vector_path`): its output is
/// the solve and the composition computed directly ([`direct`]), never the frame blend. The vector
/// path itself is tested in `vector/tests.rs` and, with real motion, in `retime/tests/vector.rs`.
#[test]
fn the_vector_path_runs_for_positions_between_frames_and_a_failure_is_never_a_silent_fallback() {
    for case in moving_cases() {
        // half speed, odd frame, Vectors = Best (vq 4): the position is between frames
        let best = |s: &mut EngineSettings| {
            sensitive(s);
            s.vectors_quality = 4;
        };
        assert_vector_path(&case, || Mock::speed(50.0), 1, best, half(4, 1));
        // every quality above No Vectors and every interpolation mode asks for it
        for vq in 1..=4 {
            for interp in 0..=2 {
                let tweak = |s: &mut EngineSettings| {
                    sensitive(s);
                    s.vectors_quality = vq;
                    s.interp = interp;
                };
                assert_vector_path(&case, || Mock::speed(50.0), 1, tweak, half(vq, interp));
            }
        }
        // every warping code runs the same solver and its own composition; a bad code is refused by name
        for interp in 0..=2 {
            for code in 0..=3 {
                let tweak = |s: &mut EngineSettings| {
                    sensitive(s);
                    s.vectors_quality = 4;
                    s.interp = interp;
                    s.smart_blend = code == 1;
                    s.warp_mode = code;
                };
                assert_vector_path(&case, || Mock::speed(50.0), 1, tweak, half(4, interp));
            }
            let r = out_with(&case, Mock::speed(50.0), 1, |s| {
                s.vectors_quality = 4;
                s.interp = interp;
                s.warp_mode = 5;
            });
            assert!(matches!(r, Err(FxError::BadParam { .. })), "{r:?}");
        }
        // ... and an exact position with Smart Blend on is the fast path (the vector path does not run)
        let ok = out_with(&case, Mock::speed(100.0), 1, |s| {
            sensitive(s);
            s.vectors_quality = 4;
            s.smart_blend = true;
            s.warp_mode = 1;
        })
        .unwrap();
        assert_eq!(ok, case.rt(1));
    }
}

#[test]
fn exact_positions_with_vectors_on_use_the_fast_path() {
    for case in cases() {
        for vq in 1..=4 {
            for n in 0..6 {
                let got = out_with(&case, Mock::speed(100.0), n, |s| s.vectors_quality = vq).unwrap();
                assert_eq!(got, case.rt(n as i64), "vq {vq} frame {n}");
            }
        }
        // even output frames of half speed are on a frame as well
        for n in [0, 2, 4, 6] {
            let got = out_with(&case, Mock::speed(50.0), n, |s| {
                s.vectors_quality = 4;
                s.interp = 2;
            })
            .unwrap();
            assert_eq!(got, case.rt((n / 2) as i64));
        }
    }
}

#[test]
fn the_exactness_tolerance_is_five_thousandths_of_a_frame() {
    for case in cases() {
        // Frame Num 2.004 is within 0.005 of frame 2: exact, fast path, and the blend weight 1 - wa
        // (0.004 in f32) is below the 0.005 snap of `blend`, so the output is frame 2 itself.
        let near = out_with(&case, Mock::frame_number(2.004), 0, |s| s.vectors_quality = 4).unwrap();
        assert_eq!(near, case.rt(2));
        let below = out_with(&case, Mock::frame_number(1.996), 0, |s| s.vectors_quality = 4).unwrap();
        assert_eq!(below, case.rt(2), "1.996: a = 1, wa = 0.004 + 0 -> weight 0.996 snaps to B");
    }
    // Frame Num 2.006 is not: the vector path, between frames 2 and 3
    let far = Selection { s0: 2.006, s1: 2.006, a: 2, b: 3, wa: 1.0f32 - (2.006f32 - 2.0f32), exact: false, vectors_quality: 4, interp: 1 };
    for case in moving_cases() {
        let tweak = |s: &mut EngineSettings| {
            sensitive(s);
            s.vectors_quality = 4;
        };
        assert_vector_path(&case, || Mock::frame_number(2.006), 0, tweak, far);
    }
}

#[test]
fn blur_compensation_makes_a_moving_position_inexact() {
    // speed 100 percent: s0 and s1 differ by 1, blur above 0.005 turns the exact position into a vector path frame.
    // Its bytes are frame 1 (wA = 1, and the blur amount (|s0 - s1| - 1) * blur is 0), so the selection shows it;
    // at speed 200 the same holds and the blur tail runs along the flow with amount 1, which changes the bytes
    let tweak = |s: &mut EngineSettings| {
        sensitive(s);
        s.vectors_quality = 4;
        s.blur = 1.0;
    };
    let at_100 = Selection { s0: 1.0, s1: 2.0, a: 1, b: 2, wa: 1.0, exact: false, vectors_quality: 4, interp: 1 };
    let at_200 = Selection { s0: 2.0, s1: 4.0, a: 2, b: 3, wa: 1.0, exact: false, vectors_quality: 4, interp: 1 };
    for case in moving_cases() {
        let mut s = case.settings();
        s.current_time = 100;
        tweak(&mut s);
        assert_eq!(core_selection(&case, Mock::speed(100.0), &s), at_100, "{:?}", case.depth);
        assert_vector_path(&case, || Mock::speed(200.0), 1, tweak, at_200);
    }
    for case in cases() {
        // blur at or below 0.005 does not
        let ok = out_with(&case, Mock::speed(100.0), 1, |s| {
            s.vectors_quality = 4;
            s.blur = 0.005;
        })
        .unwrap();
        assert_eq!(ok, case.rt(1));
        // No Vectors never runs the vector path
        let ok = out_with(&case, Mock::speed(100.0), 1, |s| {
            s.vectors_quality = 0;
            s.blur = 1.0;
        })
        .unwrap();
        assert_eq!(ok, case.rt(1));
        // a standing position (speed 0) has s0 == s1: exact whatever the blur
        let ok = out_with(&case, Mock::speed(0.0), 3, |s| {
            s.vectors_quality = 4;
            s.blur = 5.0;
        })
        .unwrap();
        assert_eq!(ok, case.rt(0));
    }
}

#[test]
fn segment_markers_override_the_vector_path_and_the_interpolation_mode() {
    for case in cases() {
        // "No MV, Use Nearest" (popup value 4): no vectors, Nearest, whatever the user chose; output 1 at
        // half speed is the tie and takes A
        let nearest = out_with(&case, Mock::speed(50.0).with_marks(|_| 4), 1, |s| {
            s.vectors_quality = 4;
            s.interp = 1;
        })
        .unwrap();
        assert_eq!(nearest, case.rt(0));
        // "No MV, Frame Blend" (popup value 5): no vectors, Blend, even when Nearest was chosen
        let blended = out_with(&case, Mock::speed(50.0).with_marks(|_| 5), 1, |s| {
            s.vectors_quality = 4;
            s.interp = 0;
        })
        .unwrap();
        assert_eq!(blended, case.mix(0, 1, 0.5));
        // a cut between the two frames (Cut A at layer time 0, Cut B from layer time 100): Nearest
        let cut = out_with(&case, Mock::speed(50.0).with_marks(|t| if t >= 100 { 2 } else { 1 }), 1, |s| {
            s.vectors_quality = 4;
            s.interp = 1;
        })
        .unwrap();
        assert_eq!(cut, case.rt(0));
    }
    // the same marker on both sides keeps the user's choice (and so the vector path)
    for case in moving_cases() {
        let tweak = |s: &mut EngineSettings| {
            sensitive(s);
            s.vectors_quality = 4;
        };
        assert_vector_path(&case, || Mock::speed(50.0).with_marks(|_| 2), 1, tweak, half(4, 1));
    }
}

#[test]
fn an_unknown_interpolation_mode_is_an_error() {
    let case = &cases()[0];
    assert!(matches!(out_with(case, Mock::speed(50.0), 1, |s| s.interp = 3), Err(FxError::BadParam { .. })));
}

#[test]
fn the_output_is_cleared_first_so_no_stale_pixels_leak_through() {
    // an all transparent source gives an all zero output at every depth
    for case in cases() {
        let (w, h) = case.size();
        let blank = MemLayer::new(clock(), vec![Frame::new(w, h, case.depth)]).unwrap();
        let s = case.settings();
        for mock in [Mock::speed(100.0), Mock::speed(50.0)] {
            let got = run(&blank, &LayerSet::new(), &s, &mock).unwrap().into_frame();
            assert_eq!(got, Frame::new(w, h, case.depth));
        }
    }
}

mod sweep {
    use super::*;

    fn next(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *seed >> 33
    }

    fn random_frame(depth: Depth, seed: &mut u64) -> Frame {
        let (w, h) = (3usize, 2usize);
        let mut f = Frame::new(w, h, depth);
        match &mut f.pixels {
            Pixels::U8(px) => {
                for p in px.iter_mut() {
                    let a = [0u8, 1, 254, 255, (next(seed) & 255) as u8][(next(seed) % 5) as usize];
                    *p = [(next(seed) & 255) as u8, (next(seed) & 255) as u8, (next(seed) & 255) as u8, a];
                }
            }
            Pixels::U16(px) => {
                for p in px.iter_mut() {
                    let a = [0u16, 1, 65534, 65535, (next(seed) & 0xffff) as u16][(next(seed) % 5) as usize];
                    *p = [(next(seed) & 0xffff) as u16, (next(seed) & 0xffff) as u16, (next(seed) & 0xffff) as u16, a];
                }
            }
            Pixels::F32(px) => {
                for p in px.iter_mut() {
                    let u = |s: &mut u64| (next(s) & 0xffffff) as f32 / 16777216.0;
                    let a = [0.0f32, 1e-13, 1.0, u(seed), u(seed)][(next(seed) % 5) as usize];
                    *p = [u(seed), u(seed) * 2.0 - 0.5, u(seed), a];
                }
            }
        }
        f
    }

    #[test]
    fn random_clips_match_the_reference_at_every_depth_and_mode() {
        let mut seed = 0x5eed_u64;
        for depth in [Depth::U8, Depth::U16, Depth::F32] {
            for round in 0..40 {
                let frames: Vec<Frame> = (0..4).map(|_| random_frame(depth, &mut seed)).collect();
                let case = Case { depth, frames };
                for (percent, weights) in [(50.0, vec![0.5f32]), (25.0, vec![0.25, 0.5, 0.75]), (75.0, vec![0.75, 0.5, 0.25])] {
                    let step = (percent / 100.0) as f32;
                    for n in 0..8i32 {
                        let x = step * n as f32;
                        let k = x.floor() as i64;
                        let frac = x - k as f32;
                        for interp in 0..=2 {
                            let got = out_interp(&case, Mock::speed(percent), n, interp);
                            let want = if frac == 0.0 {
                                case.rt(k)
                            } else {
                                let w = frac; // the blend weight 1 - wa equals the fraction for these dyadic positions
                                assert!(weights.contains(&w));
                                match interp {
                                    0 => case.rt(if 1.0 - frac >= 0.5 { k } else { k + 1 }),
                                    _ => case.mix(k, k + 1, w),
                                }
                            };
                            assert_eq!(got, want, "{depth:?} round {round} speed {percent} frame {n} interp {interp}");
                        }
                    }
                }
            }
        }
    }
}
