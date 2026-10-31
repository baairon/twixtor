//! Tests of the n-layer retime path: the groups, the byte identity of the single layer
//! cases, the identity of a black FG matte, the sign of a tracking point at its pixel, the
//! analytic two layer scene against its exact in-between frame, the wrap of the layered
//! composition and the thread count independence.

use super::*;
use crate::forward::forward_warp_coverage;
use crate::compose::{retime, retime_layers, retime_layers_with_flows, retime_with_flows};
use crate::flow::FlowSettings;
use crate::host::test_support::settings;
use crate::host::{EnvHost, LayerIds};
use crate::image::{Fmt, Samples};
use crate::timeline::test_support::Mock;
use crate::timeline::RemapTimeMap;
use fx_core::{Depth, Env, Frame, Layer, LayerSet, MemLayer, ParamCurves, Pixels, Rational, TimeBase};
use std::sync::Arc;

fn base() -> TimeBase {
    TimeBase::default_for_rate(Rational::new(30, 1).unwrap())
}

/// A smooth texture (a sum of plane waves, phases per channel and `seed`), in 0..1.
fn tex(x: f64, y: f64, seed: f64) -> [f64; 3] {
    const WAVES: [(f64, f64, f64); 4] = [(26.0, 20.0, 0.30), (19.0, 110.0, 0.28), (13.0, 60.0, 0.22), (34.0, 160.0, 0.20)];
    let mut out = [0.0; 3];
    for (c, o) in out.iter_mut().enumerate() {
        let mut s = 0.0;
        for (i, &(lambda, deg, amp)) in WAVES.iter().enumerate() {
            let th = (deg + 37.0 * seed).to_radians();
            let phase = 1.7 * i as f64 + 2.3 * c as f64 + seed;
            s += amp * (std::f64::consts::TAU * ((x * th.cos() + y * th.sin()) / lambda) + phase).sin();
        }
        *o = 0.5 + 0.5 * s;
    }
    out
}

/// An opaque wire frame of `w x h` at `depth`, colours from `f`.
fn frame(depth: Depth, w: usize, h: usize, f: impl Fn(usize, usize) -> [f64; 3]) -> Frame {
    let mut fr = Frame::new(w, h, depth);
    for y in 0..h {
        for x in 0..w {
            let c = f(x, y);
            let i = y * w + x;
            match &mut fr.pixels {
                Pixels::U8(px) => px[i] = [(c[0] * 255.0 + 0.5) as u8, (c[1] * 255.0 + 0.5) as u8, (c[2] * 255.0 + 0.5) as u8, 255],
                Pixels::U16(px) => px[i] = [(c[0] * 65535.0 + 0.5) as u16, (c[1] * 65535.0 + 0.5) as u16, (c[2] * 65535.0 + 0.5) as u16, 65535],
                Pixels::F32(px) => px[i] = [c[0] as f32, c[1] as f32, c[2] as f32, 1.0],
            }
        }
    }
    fr
}

/// The two layer scene: a FG texture moving by `vf` per frame inside a disc of radius `r` that moves with
/// it (centre `c0 + k * vf` at frame `k`), over a BG texture moving by `vb`.
#[derive(Clone, Copy)]
struct Scene {
    w: usize,
    h: usize,
    c0: (f64, f64),
    r: f64,
    vf: (f64, f64),
    vb: (f64, f64),
}

impl Scene {
    fn inside(&self, x: f64, y: f64, s: f64) -> bool {
        let (cx, cy) = (self.c0.0 + s * self.vf.0, self.c0.1 + s * self.vf.1);
        (x - cx) * (x - cx) + (y - cy) * (y - cy) <= self.r * self.r
    }
    fn distance(&self, x: f64, y: f64, s: f64) -> f64 {
        let (cx, cy) = (self.c0.0 + s * self.vf.0, self.c0.1 + s * self.vf.1);
        ((x - cx) * (x - cx) + (y - cy) * (y - cy)).sqrt()
    }
    fn rgb(&self, x: usize, y: usize, s: f64) -> [f64; 3] {
        let (xf, yf) = (x as f64, y as f64);
        if self.inside(xf, yf, s) {
            tex(xf - s * self.vf.0, yf - s * self.vf.1, 0.0)
        } else {
            tex(xf - s * self.vb.0, yf - s * self.vb.1, 1.3)
        }
    }
    fn frame(&self, depth: Depth, s: f64) -> Frame {
        frame(depth, self.w, self.h, |x, y| self.rgb(x, y, s))
    }
    fn matte(&self, depth: Depth, s: f64) -> Frame {
        frame(depth, self.w, self.h, |x, y| if self.inside(x as f64, y as f64, s) { [1.0, 1.0, 1.0] } else { [0.0, 0.0, 0.0] })
    }
}

/// A rig over `EnvHost`: the clip, the matte clips (FG1 first) and the settings of output frame `n` at
/// `speed` percent.
struct Rig {
    clip: MemLayer,
    layers: LayerSet,
    ids: Vec<&'static str>,
    params: ParamCurves,
}

const FG: [&str; 3] = ["fg1_matte", "fg2_matte", "fg3_matte"];

impl Rig {
    fn new(clip: Vec<Frame>, mattes: Vec<Vec<Frame>>) -> Rig {
        let mut layers = LayerSet::new();
        let mut ids = Vec::new();
        for (k, m) in mattes.into_iter().enumerate() {
            layers.insert(FG[k], Arc::new(MemLayer::new(base(), m).unwrap()));
            ids.push(FG[k]);
        }
        Rig { clip: MemLayer::new(base(), clip).unwrap(), layers, ids, params: ParamCurves::new(Vec::new()) }
    }

    fn env(&self) -> Env<'_> {
        Env { params: &self.params, input: &self.clip, layers: &self.layers }
    }

    fn settings(&self, n: i32) -> EngineSettings {
        let f = self.clip.frame_at(0).unwrap().unwrap();
        let mut s = settings(f.width, f.height, Fmt::from_depth(f.depth()));
        s.vectors_quality = 4;
        s.current_time = 100 * n;
        s
    }

    /// `retime_layers` with `job` (its `nfg` mattes attached) and `points` for every frame.
    fn layered(&self, s: &EngineSettings, speed: f64, job: &LayerJob, points: &[Vec<UserPoint>]) -> Result<Rendered> {
        let env = self.env();
        let mock = Mock::speed(speed);
        let map = RemapTimeMap::new(&mock, s.rates, s.current_time);
        let host = EnvHost::new(&env, s, &map, LayerIds::default())?.with_mattes(&env, &self.ids[..job.nfg])?;
        retime_layers(s, &host, job, |_| Ok(points.to_vec()))
    }

    /// Today's single layer core.
    fn single(&self, s: &EngineSettings, speed: f64) -> Result<Rendered> {
        let env = self.env();
        let mock = Mock::speed(speed);
        let map = RemapTimeMap::new(&mock, s.rates, s.current_time);
        let host = EnvHost::new(&env, s, &map, LayerIds::default())?;
        retime(s, &host)
    }
}

fn layer(sensitivity: f32) -> LayerParams {
    LayerParams { matte_chan: 1, sensitivity, shrink: 0.0, track: true }
}

/// The job of `nfg` FG layers (Red matte channel) and Main_BG, all at `sensitivity`, showing `display`.
fn job(nfg: usize, sensitivity: f32, display: i32) -> LayerJob {
    LayerJob::for_display(&vec![layer(sensitivity); nfg], layer(sensitivity), display)
}

fn image(r: Rendered) -> Image {
    match r {
        Rendered::Image(i) => i,
        Rendered::Error(_) => panic!("an error fill"),
    }
}

#[test]
fn the_groups_of_the_display_and_layers_output_popups() {
    // display 1 all, 3 to 5 the single group min(display - 3, nfg), 6 the background; no matte: one group
    for nfg in 0..=3usize {
        assert_eq!(LayerJob::display_group(1, nfg), (0, nfg));
        assert_eq!(LayerJob::display_group(6, nfg), (nfg, nfg));
        for display in 3..=5 {
            let g = ((display - 3) as usize).min(nfg);
            assert_eq!(LayerJob::display_group(display, nfg), (g, g), "display {display} nfg {nfg}");
        }
        for lo in 2..=5 {
            let g = ((lo - 2) as usize).min(nfg);
            assert_eq!(LayerJob::output_group(lo, nfg), (g, g), "layers output {lo} nfg {nfg}");
        }
        assert_eq!(LayerJob::output_group(1, nfg), (0, nfg));
    }
    for display in 1..=6 {
        assert_eq!(LayerJob::display_group(display, 0), (0, 0), "without mattes every display is the one group");
    }
    assert_eq!(LayerJob::display_group(4, 1), (1, 1), "FG2 of one matte is the background");
    assert_eq!(LayerJob::output_group(5, 2), (2, 2), "MAIN_BG");
    // a job's checks
    assert!(job(2, 0.7, 1).check().is_ok());
    assert!(LayerJob { nfg: 1, layers: vec![layer(0.7)], first: 0, last: 1, rebuild_group: false, full_group_reads_no_matte: false, pl_kernels: false, plates: false }.check().is_err());
    assert!(LayerJob { nfg: 1, layers: vec![layer(0.7); 2], first: 1, last: 0, rebuild_group: false, full_group_reads_no_matte: false, pl_kernels: false, plates: false }.check().is_err());
    assert!(LayerJob { nfg: 4, layers: vec![layer(0.7); 5], first: 0, last: 4, rebuild_group: false, full_group_reads_no_matte: false, pl_kernels: false, plates: false }.check().is_err());
}

/// A textured clip of 32 x 24 moving by (2.6, -1.3) per frame.
fn moving_clip(depth: Depth) -> Vec<Frame> {
    (0..6).map(|k| frame(depth, 32, 24, |x, y| tex(x as f64 - 2.6 * k as f64, y as f64 + 1.3 * k as f64, 0.4))).collect()
}

#[test]
fn without_mattes_and_points_the_layered_core_is_todays_core_bit_for_bit() {
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let rig = Rig::new(moving_clip(depth), Vec::new());
        for (code, interp, blur, wrap, speed, n) in [
            (0, 1, 0.0f32, false, 50.0, 1),
            (0, 0, 0.0, false, 50.0, 3),
            (0, 2, 1.0, false, 75.0, 2),
            (1, 1, 0.0, false, 50.0, 1),
            (1, 0, 0.0, true, 50.0, 1),
            (2, 1, 0.5, true, 150.0, 1),
            (2, 2, 0.0, false, 50.0, 3),
            (3, 1, 0.0, false, 50.0, 1),
            (3, 0, 1.0, true, 75.0, 2),
            (1, 2, 0.0, true, 100.0, 2),
        ] {
            let mut s = rig.settings(n);
            s.warp_mode = code;
            s.interp = interp;
            s.blur = blur;
            s.wrap_360 = wrap;
            for display in [1, 3, 6] {
                let got = rig.layered(&s, speed, &job(0, s.sensitivity, display), &[Vec::new()]).unwrap();
                assert_eq!(got, rig.single(&s, speed).unwrap(), "{depth:?} code {code} interp {interp} wrap {wrap} speed {speed}");
            }
        }
    }
}

#[test]
fn the_vectors_in_branch_without_mattes_is_todays_branch_bit_for_bit() {
    let rig = Rig::new(moving_clip(Depth::U8), Vec::new());
    let flow = |dx: f32, dy: f32| FlowField { width: 32, height: 24, data: (0..32 * 24).flat_map(|_| [dx, dy]).collect() };
    for (code, interp, both) in [(0, 1, true), (2, 1, true), (0, 0, false), (2, 2, false)] {
        let mut s = rig.settings(1);
        s.warp_mode = code;
        s.interp = interp;
        let env = rig.env();
        let mock = Mock::speed(50.0);
        let map = RemapTimeMap::new(&mock, s.rates, s.current_time);
        let host = EnvHost::new(&env, &s, &map, LayerIds::default()).unwrap();
        let flows = || Flows { f_ab: Some(flow(2.6, -1.3)), f_ba: both.then(|| flow(-2.6, 1.3)) };
        let want = retime_with_flows(&s, &host, |_| Ok(flows())).unwrap();
        let mut calls = Vec::new();
        let got = retime_layers_with_flows(&s, &host, &job(0, 0.7, 1), |_, k, m| {
            calls.push((k, m.at_a.data.iter().all(|&v| v == 1.0), m.at_b.data.len()));
            Ok(flows())
        })
        .unwrap();
        assert_eq!(got, want, "code {code} interp {interp}");
        assert_eq!(calls, vec![(0, true, 32 * 24)], "layer 0 once, with the constant matte 1");
    }
}

#[test]
fn a_black_fg_matte_gives_the_single_layer_render_for_the_warping_codes_without_a_fill() {
    // FG1's matte is 0 everywhere: its layer image is transparent (nothing to OVER) and Main_BG's tracking matte is
    // the product of the inverses, 1 everywhere, whose mask pyramid is all ones, so Main_BG's solves are the single
    // layer solves bit for bit; codes 0 and 3 then compose exactly as the single layer path. Codes 1 and 2 cannot
    // match: the single layer path fills the holes of the combined coverage after the composite, the layered path
    // fills each running result by its alpha before it, so the fill moves.
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let clip = moving_clip(depth);
        let black: Vec<Frame> = (0..6).map(|_| frame(depth, 32, 24, |_, _| [0.0; 3])).collect();
        let rig = Rig::new(clip, vec![black]);
        for code in [0, 3] {
            for interp in [0, 1, 2] {
                for (blur, speed, n) in [(0.0f32, 50.0, 1), (1.0, 50.0, 3), (0.5, 150.0, 1), (0.0, 100.0, 2)] {
                    let mut s = rig.settings(n);
                    s.warp_mode = code;
                    s.interp = interp;
                    s.blur = blur;
                    let got = rig.layered(&s, speed, &job(1, s.sensitivity, 1), &[Vec::new(), Vec::new()]).unwrap();
                    assert_eq!(got, rig.single(&s, speed).unwrap(), "{depth:?} code {code} interp {interp} blur {blur} speed {speed}");
                }
            }
        }
        for code in [1, 2] {
            let mut s = rig.settings(1);
            s.warp_mode = code;
            assert!(rig.layered(&s, 50.0, &job(1, s.sensitivity, 1), &[Vec::new(), Vec::new()]).is_ok());
        }
    }
}

#[test]
fn one_moving_point_sets_the_flow_at_its_pixel_in_the_solvers_output_convention() {
    // Sensitivity 100 has no damping: the user displacement consumer writes d / w at the point's pixel at the end of level 0
    // (one point: w = 1 and d = -(to * k_0 - from * k_0), points::user_displacement_with), and the solver negates its
    // internal displacement, so F_AB at the pixel of the point's position in A is exactly to * k_0 - from * k_0 and F_BA
    // at its pixel in B is from * k_0 - to * k_0, k_0 = (w_0 - 1) / (W - 1) per axis
    let clip = moving_clip(Depth::U8);
    let (a, b) = (Image::from_frame(&clip[1]), Image::from_frame(&clip[2]));
    let mut s = settings(32, 24, Fmt::U8);
    s.vectors_quality = 4;
    s.interp = 1;
    let sel = Selection { s0: 1.5, s1: 2.0, a: 1, b: 2, wa: 0.5, exact: false, vectors_quality: 4, interp: 1 };
    let fs = FlowSettings { sensitivity: 1.0, ..flow_settings(&s, &sel) };
    let (pa, pb) = pyramids(&a, &b, &fs).unwrap();
    let geo = pa.geometry().clone();
    let pt = UserPoint { at_a: [12.0, 9.0], at_b: [17.0, 6.0] };
    let flows = solve_layer(&s, &sel, &fs, true, &pa, &pb, None, &[pt]).unwrap();
    let (kx, ky) = ((geo.width - 1) as f32 / 31.0, (geo.height - 1) as f32 / 23.0);
    let px = |p: [f32; 2]| ((p[0] * kx + 0.5) as usize, (p[1] * ky + 0.5) as usize);
    let (ax, ay) = px(pt.at_a);
    let (bx, by) = px(pt.at_b);
    let f_ab = flows.f_ab.unwrap();
    let f_ba = flows.f_ba.unwrap();
    assert_eq!(f_ab.at(ax, ay), [17.0f32 * kx - 12.0f32 * kx, 6.0f32 * ky - 9.0f32 * ky]);
    assert_eq!(f_ba.at(bx, by), [12.0f32 * kx - 17.0f32 * kx, 9.0f32 * ky - 6.0f32 * ky]);
    // the motion the point names, about 5 px to the right and 3 up, in working pixels
    assert!(f_ab.at(ax, ay)[0] > 2.0 && f_ab.at(ax, ay)[1] < -1.0);
    // no point is the plain solve
    let none = solve_layer(&s, &sel, &fs, true, &pa, &pb, None, &[]).unwrap();
    assert_eq!(none.f_ab.unwrap(), crate::flow::solve(&a, &b, &fs).unwrap());
    // through the core: the point changes the in between frame, with no matte as with one
    let rig = Rig::new(clip.clone(), Vec::new());
    let s = rig.settings(3);
    let plain = rig.single(&s, 50.0).unwrap();
    let pointed = rig.layered(&s, 50.0, &job(0, s.sensitivity, 1), &[vec![pt]]).unwrap();
    assert_ne!(pointed, plain);
    assert_eq!(rig.layered(&rig.settings(2), 50.0, &job(0, s.sensitivity, 1), &[vec![pt]]).unwrap(), rig.single(&rig.settings(2), 50.0).unwrap(), "an exact frame");
}

/// Mean absolute colour error of `got` (engine image, opaque or premultiplied) against the exact frame of `scene`
/// at `s` over the pixels `keep` selects, as fractions of white.
fn region_error(got: &Image, scene: &Scene, s: f64, keep: impl Fn(f64, f64) -> bool) -> f64 {
    let f = got.to_frame();
    let (mut sum, mut n) = (0.0, 0usize);
    for y in 0..scene.h {
        for x in 0..scene.w {
            if !keep(x as f64, y as f64) {
                continue;
            }
            let want = scene.rgb(x, y, s);
            let i = y * scene.w + x;
            let px = match &f.pixels {
                Pixels::U8(p) => p[i],
                _ => unreachable!(),
            };
            for c in 0..3 {
                sum += (px[c] as f64 / 255.0 - want[c]).abs();
                n += 1;
            }
        }
    }
    sum / n as f64
}

#[test]
fn two_layers_under_complementary_disc_mattes_render_the_exact_in_between_frame() {
    // FG at (+3, 0) per frame inside a disc of radius 60, BG at (-2, +1) outside, 256 x 192, 50 percent speed: output
    // frame 3 is position 1.5. On the FG core and on the BG far from the disc the layered render is the exact frame
    // at 1.5 within the sampling floor of the warps (the single layer solver recovers both motions there too: its
    // render is at the same floor, so the layers cannot halve it; where they do halve the error is the flow next to
    // the disc, see the test below)
    let scene = Scene { w: 256, h: 192, c0: (120.0, 96.0), r: 60.0, vf: (3.0, 0.0), vb: (-2.0, 1.0) };
    let clip: Vec<Frame> = (0..5).map(|k| scene.frame(Depth::U8, k as f64)).collect();
    let mattes: Vec<Frame> = (0..5).map(|k| scene.matte(Depth::U8, k as f64)).collect();
    let rig = Rig::new(clip, vec![mattes]);
    let core = |x: f64, y: f64| scene.distance(x, y, 1.5) < scene.r - 3.0;
    let far = |x: f64, y: f64| scene.distance(x, y, 1.5) > scene.r + 4.0 && x >= 12.0 && y >= 12.0 && x < 244.0 && y < 180.0;
    for interp in [1, 0] {
        let mut s = rig.settings(3);
        s.interp = interp;
        let layered = image(rig.layered(&s, 50.0, &job(1, s.sensitivity, 1), &[Vec::new(), Vec::new()]).unwrap());
        let single = image(rig.single(&s, 50.0).unwrap());
        let (lc, sc) = (region_error(&layered, &scene, 1.5, core), region_error(&single, &scene, 1.5, core));
        let (lf, sf) = (region_error(&layered, &scene, 1.5, far), region_error(&single, &scene, 1.5, far));
        // measured (Blend): FG core 0.0018 against 0.0018, BG far 0.0024 against 0.0025; the frame blend is about 0.05
        assert!(lc < 0.003 && lc <= sc * 1.05, "interp {interp}: FG core error {lc:.5} (single layer {sc:.5})");
        assert!(lf < 0.004 && lf <= sf * 1.05, "interp {interp}: BG far error {lf:.5} (single layer {sf:.5})");
    }
}

#[test]
fn each_layer_solves_its_own_motion_and_halves_the_flow_error_next_to_the_disc() {
    // the scene above at frames 1 and 2: F_AB of the single layer solve against the solves of the two layers with their
    // mask pyramids (FG1 the disc, Main_BG its complement), each measured on its own region against the true motion
    // in working pixels. Next to the disc the single flow smears the FG motion into the BG; Main_BG's masked solve
    // does not (measured 0.24 against 0.66 px in the 12 px band outside the disc), and on the FG core the FG layer is
    // at least as close (0.007 against 0.011 px)
    let scene = Scene { w: 256, h: 192, c0: (120.0, 96.0), r: 60.0, vf: (3.0, 0.0), vb: (-2.0, 1.0) };
    let (a, b) = (Image::from_frame(&scene.frame(Depth::U8, 1.0)), Image::from_frame(&scene.frame(Depth::U8, 2.0)));
    let (ma, mb) = (matte::extract(&Image::from_frame(&scene.matte(Depth::U8, 1.0)), 1), matte::extract(&Image::from_frame(&scene.matte(Depth::U8, 2.0)), 1));
    let (ea, eb) = (matte::effective(&[ma], &[0.0], scene.w, scene.h).unwrap(), matte::effective(&[mb], &[0.0], scene.w, scene.h).unwrap());
    let mut s = settings(256, 192, Fmt::U8);
    s.vectors_quality = 4;
    let sel = Selection { s0: 1.5, s1: 2.0, a: 1, b: 2, wa: 0.5, exact: false, vectors_quality: 4, interp: 1 };
    let fs = flow_settings(&s, &sel);
    let (pa, pb) = pyramids(&a, &b, &fs).unwrap();
    let geo = pa.geometry().clone();
    let single = solve_layer(&s, &sel, &fs, true, &pa, &pb, None, &[]).unwrap().f_ab.unwrap();
    let mut layer_flows = Vec::new();
    for k in 0..2 {
        let (m0, m1) = (MaskPyramid::build(&ea[k], &fs).unwrap(), MaskPyramid::build(&eb[k], &fs).unwrap());
        layer_flows.push(solve_layer(&s, &sel, &fs, true, &pa, &pb, Some((&m0, &m1)), &[]).unwrap().f_ab.unwrap());
    }
    let (kx, ky) = ((geo.width - 1) as f64 / 255.0, (geo.height - 1) as f64 / 191.0);
    let band = |lo: f64, hi: f64| {
        let (mut e_s, mut e_l, mut n) = (0.0, 0.0, 0usize);
        for y in 0..geo.height {
            for x in 0..geo.width {
                let (fx, fy) = (x as f64 / kx, y as f64 / ky);
                let d = scene.distance(fx, fy, 1.0);
                if d < lo || d >= hi || fx < 12.0 || fy < 12.0 || fx > 244.0 || fy > 180.0 {
                    continue;
                }
                let fg = d <= scene.r;
                let truth = if fg { (3.0 * kx, 0.0) } else { (-2.0 * kx, ky) };
                let err = |f: [f32; 2]| ((f[0] as f64 - truth.0).powi(2) + (f[1] as f64 - truth.1).powi(2)).sqrt();
                e_s += err(single.at(x, y));
                e_l += err(layer_flows[if fg { 0 } else { 1 }].at(x, y));
                n += 1;
            }
        }
        (e_l / n as f64, e_s / n as f64)
    };
    let (l, s1) = band(60.0, 72.0);
    assert!(l <= 0.5 * s1, "BG next to the disc: Main_BG's flow error {l:.4} against {s1:.4} px of the single flow");
    let (l, s1) = band(0.0, 50.0);
    assert!(l <= s1 && l < 0.05, "FG core: FG1's flow error {l:.4} against {s1:.4} px of the single flow");
    let (l, s1) = band(72.0, 400.0);
    assert!(l <= s1 * 1.05 && l < 0.1, "BG far: Main_BG's flow error {l:.4} against {s1:.4} px");
}

#[test]
fn the_layered_render_is_the_same_for_any_thread_count() {
    let scene = Scene { w: 64, h: 48, c0: (30.0, 24.0), r: 14.0, vf: (2.0, 0.0), vb: (-1.0, 1.0) };
    let clip: Vec<Frame> = (0..5).map(|k| scene.frame(Depth::U16, k as f64)).collect();
    let m1: Vec<Frame> = (0..5).map(|k| scene.matte(Depth::U16, k as f64)).collect();
    let m2: Vec<Frame> = (0..5).map(|k| frame(Depth::U16, 64, 48, |x, _| if x > 40 + k { [1.0; 3] } else { [0.0; 3] })).collect();
    let rig = Rig::new(clip, vec![m1, m2]);
    let pts = vec![vec![UserPoint { at_a: [30.0, 24.0], at_b: [32.0, 24.0] }], Vec::new(), vec![UserPoint { at_a: [5.0, 5.0], at_b: [4.0, 6.0] }]];
    for (code, interp, wrap) in [(0, 1, false), (1, 1, true), (2, 2, false), (3, 0, true)] {
        let mut s = rig.settings(3);
        s.warp_mode = code;
        s.interp = interp;
        s.wrap_360 = wrap;
        s.blur = 0.5;
        let mut j = job(2, 0.8, 1);
        j.layers[0].shrink = 1.5;
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| rig.layered(&s, 75.0, &j, &pts).unwrap())
        };
        let one = run(1);
        assert_eq!(run(4), one, "code {code}");
        assert_eq!(run(16), one, "code {code}");
    }
}

/// An engine image of `w x h` (8 bpc, opaque) periodic in x with period `w`, moved `shift` whole pixels to the
/// right: one quantised period indexed modulo `w`, so every shift is exact.
fn periodic(w: usize, h: usize, shift: i64) -> Image {
    let period = frame(Depth::U8, w, h, |x, y| {
        let t = std::f64::consts::TAU * x as f64 / w as f64;
        [0.5 + 0.4 * t.sin(), 0.5 + 0.3 * (2.0 * t).cos(), 0.5 + 0.2 * (y as f64 * 0.3).sin()]
    });
    let mut f = period.clone();
    if let (Pixels::U8(dst), Pixels::U8(src)) = (&mut f.pixels, &period.pixels) {
        for y in 0..h {
            for x in 0..w {
                dst[y * w + x] = src[y * w + (x as i64 - shift).rem_euclid(w as i64) as usize];
            }
        }
    }
    Image::from_frame(&f)
}

#[test]
fn the_layered_composition_wraps_its_warps_fills_and_blur() {
    // hand built uniform flows over a texture periodic in x: frame B is frame A moved 4 px to the right, so with the
    // wrap flag the inverse warps read across the seam and the in between frame is the texture moved 2 px, exactly
    let (w, h) = (32usize, 16usize);
    let (a, b) = (periodic(w, h, 0), periodic(w, h, 4));
    let uniform = |dx: f32| FlowField { width: w, height: h, data: (0..w * h).flat_map(|_| [dx, 0.0]).collect() };
    let flows = vec![Flows { f_ab: Some(uniform(4.0)), f_ba: Some(uniform(-4.0)) }];
    let mattes = (vec![Matte::filled(w, h, 1.0)], vec![Matte::filled(w, h, 1.0)]);
    let sel = Selection { s0: 1.5, s1: 1.5, a: 1, b: 2, wa: 0.5, exact: false, vectors_quality: 4, interp: 1 };
    let want = periodic(w, h, 2);
    let j = job(1, 0.7, 6);
    for code in [0, 1] {
        let mut s = settings(w, h, Fmt::U8);
        s.warp_mode = code;
        s.wrap_360 = true;
        let got = compose_layers(&s, &sel, &j, &a, &b, &mattes, &flows).unwrap();
        assert_eq!(got, want, "code {code}: the wrapped warps are exact shifts across the seam");
        s.wrap_360 = false;
        let plain = compose_layers(&s, &sel, &j, &a, &b, &mattes, &flows).unwrap();
        assert_ne!(plain, want, "code {code}: without the wrap the border columns are held or faded");
    }
    // the fill of a running result wraps: a hole column at the seam is filled from both sides with the flag
    let mut r = a.clone();
    if let Samples::U8(v) = &mut r.samples {
        for y in 0..h {
            v[(y * w) * 4..(y * w) * 4 + 4].copy_from_slice(&[0, 0, 0, 0]);
        }
    }
    let (mut wrapped, mut plain) = (r.clone(), r.clone());
    fill_alpha_holes(&mut wrapped, true).unwrap();
    fill_alpha_holes(&mut plain, false).unwrap();
    assert_ne!(wrapped, plain, "the wrapped fill reads the last column");
    // the blur tail wraps: a fast pan with Blur Compensation differs at the seam with the flag
    let mut s = settings(w, h, Fmt::U8);
    s.blur = 1.0;
    let fast = Selection { s0: 1.0, s1: 3.0, ..sel };
    s.wrap_360 = true;
    let blurred = compose_layers(&s, &fast, &j, &a, &b, &mattes, &flows).unwrap();
    s.wrap_360 = false;
    let held = compose_layers(&s, &fast, &j, &a, &b, &mattes, &flows).unwrap();
    assert_ne!(blurred, held);
}

/// A wire frame of `w x h` at `depth` with straight colours and alpha from `f` (`[r, g, b, a]` in 0..1).
fn rgba_frame(depth: Depth, w: usize, h: usize, f: impl Fn(usize, usize) -> [f64; 4]) -> Frame {
    let mut fr = Frame::new(w, h, depth);
    for y in 0..h {
        for x in 0..w {
            let c = f(x, y);
            let i = y * w + x;
            match &mut fr.pixels {
                Pixels::U8(px) => px[i] = c.map(|v| (v * 255.0 + 0.5) as u8),
                Pixels::U16(px) => px[i] = c.map(|v| (v * 65535.0 + 0.5) as u16),
                Pixels::F32(px) => px[i] = c.map(|v| v as f32),
            }
        }
    }
    fr
}

/// The samples of an engine image as f32, `[alpha, r, g, b]` per pixel.
fn samples_f32(img: &Image) -> Vec<f32> {
    match &img.samples {
        Samples::U8(v) => v.iter().map(|&x| f32::from(x)).collect(),
        Samples::U16(v) => v.iter().map(|&x| f32::from(x)).collect(),
        Samples::F32(v) => v.clone(),
    }
}

/// The composite of layer images (`[alpha, r, g, b]` pixels, premultiplied, in units of `white`) in the order given,
/// bottom first, by the OVER written out here: `src` with alpha 0 keeps the result, an opaque `src` replaces
/// it, else `ia = 1 - a / white`, `alpha = (white - dest.alpha) * (a / white) + dest.alpha`, `rgb = dest.rgb * ia +
/// src.rgb`; every result is stored as the format stores (`x + 0.5` truncated and saturated at 8 and 16 bpc,
/// `x + 0.0` in float) before the next layer.
fn over_by_hand(layers: &[[f32; 4]], white: f32) -> [f32; 4] {
    let store = |x: f32| {
        if white == 1.0 {
            x + 0.0
        } else {
            let y = x + 0.5;
            if 0.0 > y {
                0.0
            } else if y > white {
                white
            } else {
                y.trunc()
            }
        }
    };
    let mut r = [0.0f32; 4];
    for src in layers {
        let v = if src[0] == 0.0 {
            r
        } else if src[0] == white {
            *src
        } else {
            let ia = 1.0f32 - src[0] / white;
            let alpha = (white - r[0]) * (src[0] / white) + r[0];
            [alpha, r[1] * ia + src[1], r[2] * ia + src[2], r[3] * ia + src[3]]
        };
        r = v.map(store);
    }
    r
}

#[test]
fn the_fast_path_composes_the_group_bottom_up_by_the_over_of_a5() {
    // an exact frame takes the fast path, which composes the displayed group of the unwarped frame from
    // layer `last` down to `first` (Main_BG first, FG1 last, so FG1 ends on top). Every layer image of one frame is
    // the frame scaled by a matte, so the order shows only in the rounding of each stored OVER: soft mattes over a
    // partly transparent frame give many pixels where the other order stores another value. The render is the
    // composite worked out here, bit for bit, and the top down composite is not
    let (w, h) = (32usize, 24usize);
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let clip: Vec<Frame> = (0..4)
            .map(|k| {
                rgba_frame(depth, w, h, |x, y| {
                    let c = tex(x as f64 + k as f64, y as f64, 0.2);
                    [c[0], c[1], c[2], 0.25 + 0.7 * ((x * 7 + y * 3) % 11) as f64 / 10.0]
                })
            })
            .collect();
        let m1: Vec<Frame> = (0..4).map(|_| frame(depth, w, h, |x, y| [(x as f64 / (w - 1) as f64 * 0.9 + 0.003 * y as f64).min(1.0); 3])).collect();
        let m2: Vec<Frame> = (0..4).map(|_| frame(depth, w, h, |x, y| [0.15 + 0.8 * y as f64 / (h - 1) as f64 - 0.002 * x as f64; 3])).collect();
        let rig = Rig::new(clip.clone(), vec![m1.clone(), m2.clone()]);
        let mut s = rig.settings(2);
        s.interp = 0;
        let run = |threads: usize| in_pool(threads, || image(rig.layered(&s, 100.0, &job(2, s.sensitivity, 1), &[Vec::new(), Vec::new(), Vec::new()]).unwrap()));
        let got = run(1);
        assert_eq!(run(4), got, "{depth:?}: 1 and 4 threads");
        // the frame and the raw mattes of source frame 2, as the host prepares them
        let a = Image::from_frame(&clip[2]);
        let mattes = [matte::extract(&Image::from_frame(&m1[2]), 1), matte::extract(&Image::from_frame(&m2[2]), 1)];
        let white = match depth {
            Depth::U8 => 255.0f32,
            Depth::U16 => 65535.0,
            Depth::F32 => 1.0,
        };
        // the premultiply `q(c * m)` (float: `c * m`)
        let premul = |c: f32, m: f32| if white == 1.0 { c * m } else { ((c * m) as f64 + 0.5).trunc() as f32 };
        let px = samples_f32(&a);
        let (mut want, mut top_down) = (Vec::with_capacity(px.len()), Vec::with_capacity(px.len()));
        for (i, p) in px.chunks(4).enumerate() {
            let bg = [p[0], p[1], p[2], p[3]];
            let fg = |k: usize| [p[0], p[1], p[2], p[3]].map(|c| premul(c, mattes[k].data[i]));
            want.extend(over_by_hand(&[bg, fg(1), fg(0)], white));
            top_down.extend(over_by_hand(&[fg(0), fg(1), bg], white));
        }
        assert_eq!(bits_of(&samples_f32(&got)), bits_of(&want), "{depth:?}: the bottom up composite");
        let differ = want.iter().zip(&top_down).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        eprintln!("{depth:?}: {differ} values differ between the two orders");
        assert!(differ >= 20, "{depth:?}: the data shows the order ({differ} values)");
    }
}

fn bits_of(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(f)
}

#[test]
fn each_side_is_premultiplied_by_the_matte_of_its_own_frame() {
    // side A's layer image is frame A times the matte AT A, side B's frame B times the matte AT B. A still
    // texture under a disc matte that moves between A and B, zero flows and the FG1 group alone (display 3): the
    // in-between frame is the blend of the two premultiplied frames, so the part of the disc inside at A only comes
    // from side A (weight wA) and the part inside at B only from side B (weight 1 - wA)
    let (w, h) = (32usize, 16usize);
    let still = Image::from_frame(&frame(Depth::U8, w, h, |x, y| tex(x as f64, y as f64, 0.9)));
    let disc = |cx: f64| Matte {
        width: w,
        height: h,
        data: (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f64 - cx, (i / w) as f64 - 8.0);
                if x * x + y * y <= 25.0 { 1.0 } else { 0.0 }
            })
            .collect(),
    };
    let (ma, mb) = (disc(10.0), disc(20.0));
    let zero = FlowField::new(w, h);
    let flows = vec![Flows { f_ab: Some(zero.clone()), f_ba: Some(zero) }];
    let mattes = (vec![ma.clone()], vec![mb.clone()]);
    let sel = Selection { s0: 1.75, s1: 1.75, a: 1, b: 2, wa: 0.25, exact: false, vectors_quality: 4, interp: 1 };
    let frac = 1.0f32 - sel.wa;
    let (pa, pb) = (matte::premultiply(&still, &ma).unwrap(), matte::premultiply(&still, &mb).unwrap());
    let mut want = Image::new(w, h, Fmt::U8);
    blend(&mut want, &pa, &pb, frac);
    let alpha = |img: &Image, x: usize, y: usize| match &img.samples {
        Samples::U8(v) => v[(y * w + x) * 4],
        _ => unreachable!(),
    };
    for code in [0, 3] {
        let mut s = settings(w, h, Fmt::U8);
        s.warp_mode = code;
        let run = |threads: usize| in_pool(threads, || compose_layers(&s, &sel, &job(1, 0.7, 3), &still, &still, &mattes, &flows).unwrap());
        let got = run(1);
        assert_eq!(run(4), got, "code {code}: 1 and 4 threads");
        // inside at A only: wA of white; inside at B only: 1 - wA of white; inside both: opaque; outside: transparent
        for (x, y, lo, hi) in [(6usize, 8usize, 63u8, 64u8), (24, 8, 191, 192), (15, 8, 255, 255), (2, 2, 0, 0)] {
            let a = alpha(&got, x, y);
            assert!(a >= lo && a <= hi, "code {code} ({x}, {y}): alpha {a}");
        }
        assert_eq!(got, want, "code {code}: the blend of A times its matte and B times its matte");
    }
}

#[test]
fn the_fill_of_a_forward_warped_layer_wraps_its_seam_holes() {
    // G6: code 2 pushes each layer image along its own flow and the forward warp itself never wraps, so a pan
    // leaves holes in the columns next to the seam (A's push to the right opens columns 0 and 1, B's push to the left
    // the last two); the fill of each running result is the circular one under `wrap_360`, and stays plain under
    // Weighted Blend. The composite is the one of the running results built here by the same calls and filled with
    // the flag, and the plain fill gives another one
    let (w, h) = (32usize, 16usize);
    let (a, b) = (periodic(w, h, 0), periodic(w, h, 4));
    let uniform = |dx: f32| FlowField { width: w, height: h, data: (0..w * h).flat_map(|_| [dx, 0.0]).collect() };
    let flows = vec![Flows { f_ab: Some(uniform(4.0)), f_ba: Some(uniform(-4.0)) }];
    let mattes = (vec![Matte::filled(w, h, 1.0)], vec![Matte::filled(w, h, 1.0)]);
    let j = job(1, 0.7, 6);
    let sel = Selection { s0: 1.5, s1: 1.5, a: 1, b: 2, wa: 0.5, exact: false, vectors_quality: 4, interp: 1 };
    let frac = 1.0f32 - sel.wa;
    // the running result of the one layer of the group (Main_BG, the frame itself) pushed by its own flow
    let running = |img: &Image, flow: &FlowField, amount: f32| {
        let mut t = Image::new(w, h, Fmt::U8);
        let _ = forward_warp_coverage(&mut t, img, flow, amount).unwrap();
        let mut r = Image::new(w, h, Fmt::U8);
        matte::over(&mut r, &t).unwrap();
        r
    };
    let (ra, rb) = (running(&a, flows[0].f_ab.as_ref().unwrap(), frac), running(&b, flows[0].f_ba.as_ref().unwrap(), sel.wa));
    let alpha = |img: &Image, x: usize, y: usize| match &img.samples {
        Samples::U8(v) => v[(y * w + x) * 4],
        _ => unreachable!(),
    };
    for y in 0..h {
        assert!(alpha(&ra, 0, y) < 255 && alpha(&rb, w - 1, y) < 255, "row {y}: the pushes open the seam columns");
    }
    let composite = |is360: bool, weighted: bool| {
        let (mut x, mut y) = (ra.clone(), rb.clone());
        fill_alpha_holes(&mut x, is360).unwrap();
        fill_alpha_holes(&mut y, is360).unwrap();
        if weighted {
            sharpen_step(&mut x, &a, frac);
            sharpen_step(&mut y, &b, sel.wa);
        }
        let mut out = Image::new(w, h, Fmt::U8);
        smart_blend(&mut out, &x, &y, frac).unwrap();
        out
    };
    let mut s = settings(w, h, Fmt::U8);
    s.warp_mode = 2;
    s.wrap_360 = true;
    let got = in_pool(1, || compose_layers(&s, &sel, &j, &a, &b, &mattes, &flows).unwrap());
    assert_eq!(in_pool(4, || compose_layers(&s, &sel, &j, &a, &b, &mattes, &flows).unwrap()), got, "1 and 4 threads");
    assert_eq!(got, composite(true, false), "Blend: the wrapped fill of each running result");
    assert_ne!(got, composite(false, false), "the plain fill of the seam holes differs");
    let weighted = Selection { interp: 2, ..sel };
    assert_eq!(compose_layers(&s, &weighted, &j, &a, &b, &mattes, &flows).unwrap(), composite(false, true), "Weighted Blend fills plain");
    // Nearest has one running result (wA = 0.5 takes A), filled with the flag
    let nearest = Selection { interp: 0, ..sel };
    let mut x = ra.clone();
    fill_alpha_holes(&mut x, true).unwrap();
    assert_eq!(compose_layers(&s, &nearest, &j, &a, &b, &mattes, &flows).unwrap(), x, "Nearest: the wrapped fill");
    // without the flag the same render fills plain
    s.wrap_360 = false;
    assert_eq!(compose_layers(&s, &sel, &j, &a, &b, &mattes, &flows).unwrap(), composite(false, false));
}

/// The same wire frame with its rows in reverse order.
fn mirrored(f: &Frame) -> Frame {
    let (w, h) = (f.width, f.height);
    let mut out = f.clone();
    fn rev<T: Copy>(dst: &mut [T], src: &[T], w: usize, h: usize) {
        for y in 0..h {
            dst[y * w..(y + 1) * w].copy_from_slice(&src[(h - 1 - y) * w..(h - y) * w]);
        }
    }
    match (&mut out.pixels, &f.pixels) {
        (Pixels::U8(d), Pixels::U8(s)) => rev(d, s, w, h),
        (Pixels::U16(d), Pixels::U16(s)) => rev(d, s, w, h),
        (Pixels::F32(d), Pixels::F32(s)) => rev(d, s, w, h),
        _ => unreachable!(),
    }
    out
}

#[test]
fn the_layered_vector_path_runs_on_bottom_up_rows_when_the_wiring_says_so() {
    // `motion_rows_reversed` (TwixtorPro_64 runs its motion path on bottom-up rows): the layered render of the frames
    // is the vertical mirror of the unreversed render of the mirrored frames and mattes
    let scene = Scene { w: 64, h: 48, c0: (30.0, 24.0), r: 14.0, vf: (2.0, 1.0), vb: (-1.0, 1.0) };
    let clip: Vec<Frame> = (0..5).map(|k| scene.frame(Depth::U8, k as f64)).collect();
    let m1: Vec<Frame> = (0..5).map(|k| scene.matte(Depth::U8, k as f64)).collect();
    let rig = Rig::new(clip.clone(), vec![m1.clone()]);
    let flipped = Rig::new(clip.iter().map(mirrored).collect(), vec![m1.iter().map(mirrored).collect()]);
    let pts = vec![Vec::new(); 2];
    for (code, interp) in [(0, 1), (1, 1), (3, 0)] {
        let mut s = rig.settings(3);
        s.warp_mode = code;
        s.interp = interp;
        let j = job(1, 0.8, 1);
        let mut s_rev = s.clone();
        s_rev.motion_rows_reversed = true;
        let (Rendered::Image(a), Rendered::Image(b)) = (rig.layered(&s_rev, 75.0, &j, &pts).unwrap(), flipped.layered(&s, 75.0, &j, &pts).unwrap()) else {
            panic!("an image is expected");
        };
        assert_eq!(a, b.rows_reversed(), "code {code}");
    }
}

#[test]
fn the_single_layer_point_path_runs_on_bottom_up_rows_when_the_wiring_says_so() {
    // no FG layer, one tracking point: with `motion_rows_reversed` the render is the vertical mirror of the unreversed
    // render of the mirrored frames with the point at H - 1 - y, as the vector path without points
    let scene = Scene { w: 64, h: 48, c0: (30.0, 24.0), r: 14.0, vf: (2.0, 1.0), vb: (-1.0, 1.0) };
    let clip: Vec<Frame> = (0..5).map(|k| scene.frame(Depth::U8, k as f64)).collect();
    let rig = Rig::new(clip.clone(), Vec::new());
    let flipped = Rig::new(clip.iter().map(mirrored).collect(), Vec::new());
    let pt = UserPoint { at_a: [30.0, 20.0], at_b: [33.0, 22.5] };
    let up = UserPoint { at_a: [30.0, 47.0 - 20.0], at_b: [33.0, 47.0 - 22.5] };
    for (code, interp) in [(0, 1), (1, 1), (3, 0)] {
        let mut s = rig.settings(3);
        s.warp_mode = code;
        s.interp = interp;
        let j = job(0, 0.8, 1);
        let mut s_rev = s.clone();
        s_rev.motion_rows_reversed = true;
        let a = image(rig.layered(&s_rev, 75.0, &j, &[vec![pt]]).unwrap());
        let b = image(flipped.layered(&s, 75.0, &j, &[vec![up]]).unwrap());
        assert_eq!(a, b.rows_reversed(), "code {code}");
        // and the point is used: without it the render differs
        assert_ne!(image(rig.layered(&s_rev, 75.0, &j, &[Vec::new()]).unwrap()), a, "code {code}");
    }
}

/// The fill of the single group rebuild runs in the frame of the motion path: with the rows reversed and the tap above
/// first it is the fill of the reversed image turned back (the frame of TwixtorPro_64's single channel fill, which
/// reproduces the plugin's filled plates bit for bit); the known pixels
/// keep their values, and the frame changes the last bits of the filled ones.
#[test]
fn the_rebuild_fill_runs_on_the_rows_of_the_motion_path() {
    let (w, h) = (24usize, 16usize);
    let mut img = Image::new(w, h, Fmt::F32);
    if let Samples::F32(s) = &mut img.samples {
        for (i, p) in s.chunks_exact_mut(CHANNELS).enumerate() {
            let c = tex((i % w) as f64, (i / w) as f64, 0.37);
            p.copy_from_slice(&[1.0, c[0] as f32, c[1] as f32, c[2] as f32]);
        }
    }
    let boxed = |x: usize, y: usize| ((3..10).contains(&x) && (1..9).contains(&y)) || ((12..20).contains(&x) && (5..13).contains(&y));
    let holes: Vec<f32> = (0..w * h).map(|i| if boxed(i % w, i / w) { 1.0 } else { 0.0 }).collect();
    let mut host = img.clone();
    FillFrame { is360: false, taps: VerticalTaps::BelowFirst, reversed: false }.fill(&mut host, &holes).unwrap();
    let mut plain = img.clone();
    fill_holes_masked(&mut plain, HoleMaskRef::F32(&holes), false).unwrap();
    assert_eq!(host, plain, "host rows and the engine's order are the plain fill");
    let mut want = img.rows_reversed();
    let rev: Vec<f32> = holes.chunks(w).rev().flatten().copied().collect();
    fill_holes_masked_with(&mut want, HoleMaskRef::F32(&rev), false, VerticalTaps::AboveFirst).unwrap();
    let want = want.rows_reversed();
    let mut got = img.clone();
    FillFrame { is360: false, taps: VerticalTaps::AboveFirst, reversed: true }.fill(&mut got, &holes).unwrap();
    assert_eq!(got, want);
    let (Samples::F32(g), Samples::F32(o), Samples::F32(p)) = (&got.samples, &img.samples, &plain.samples) else { unreachable!() };
    let mut moved = 0;
    for i in 0..w * h {
        for c in 0..CHANNELS {
            if holes[i] == 0.0 {
                assert_eq!(g[i * CHANNELS + c].to_bits(), o[i * CHANNELS + c].to_bits(), "known pixel {i} channel {c}");
            } else {
                moved += usize::from(g[i * CHANNELS + c].to_bits() != p[i * CHANNELS + c].to_bits());
            }
        }
    }
    assert!(moved > 0, "the bottom-up frame changes filled values");
}

/// The plate rebuild of the fast path ([`rebuilt_plate`], a render with plates): on an opaque frame with hard mattes it
/// gives the old rebuild's bytes (FG2 and Background displays); a frame pixel that is not opaque is a hole of the plate
/// (filled, its alpha the frame's on the Background display), where the old rebuild kept it as it was.
#[test]
fn the_plate_rebuild_of_the_fast_path_keeps_hard_mattes_and_fills_partial_alpha() {
    let (w, h) = (12usize, 8usize);
    let mut img = Image::from_frame(&frame(Depth::U8, w, h, |x, y| tex(x as f64, y as f64, 0.2)));
    let band = |x0: usize, x1: usize| Matte {
        width: w,
        height: h,
        data: (0..w * h).map(|i| if (x0..x1).contains(&(i % w)) { 1.0 } else { 0.0 }).collect(),
    };
    let mattes = vec![band(2, 6), band(4, 9)];
    let fill = FillFrame { is360: false, taps: VerticalTaps::BelowFirst, reversed: false };
    let eff = matte::effective(&mattes, &[0.0, 0.0], w, h).unwrap();
    for d in [4, 6] {
        let mut j = job(2, 0.7, d);
        j.plates = true;
        let k = j.first;
        let old = reconstructed_layer(&img, &mattes, &eff[k], k, 2, fill).unwrap();
        assert_eq!(rebuilt_plate(&img, &mattes, &j, fill).unwrap(), old, "display {d}");
    }
    let at = (10usize, 3usize);
    if let Samples::U8(s) = &mut img.samples {
        s[(at.1 * w + at.0) * CHANNELS + A] = 128;
    }
    let mut j = job(2, 0.7, 6);
    j.plates = true;
    let old = reconstructed_layer(&img, &mattes, &eff[2], 2, 2, fill).unwrap();
    let new = rebuilt_plate(&img, &mattes, &j, fill).unwrap();
    let px = |i: &Image| match &i.samples {
        Samples::U8(s) => s[(at.1 * w + at.0) * CHANNELS..(at.1 * w + at.0 + 1) * CHANNELS].to_vec(),
        _ => unreachable!(),
    };
    assert_eq!(px(&new)[A], 128, "the frame's alpha");
    assert_ne!(px(&new), px(&old), "filled and premultiplied, not kept");
}
