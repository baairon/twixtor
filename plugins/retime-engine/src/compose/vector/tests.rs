//! Goldens of the composition of the vector path with HAND BUILT flows (no solver in the loop
//! unless a test says so): uniform flow fields over textures that are exact integer shifts of
//! each other, so every warp is an exact pixel shift that this file computes by plain indexing,
//! and the blend weights are checked against formulas worked out here.
//!
//! The scenario of the uniform tests: frame `B` is frame `A` moved by `v = (8, 4)` pixels
//! (`B(p) = A(p - v)`), so the flow `F_AB` (`A(x) ~ B(x + F_AB(x))`) is `+v` and `F_BA` is `-v`.
//! With `frac = 1 - wA` the exact frame between them is `A(p - frac * v)`. The amounts used
//! (0.25, 0.5, 0.75) make `frac * v` whole pixels, where the bilinear warp is an exact shift.

use super::*;
use crate::filters::sharpen_step;
use crate::compose::{render, select_frames, Rendered};
use crate::flow::solve_pair;
use crate::host::test_support::settings;
use crate::host::{EnvHost, LayerIds};
use crate::image::{Fmt, Samples};
use crate::timeline::test_support::Mock;
use crate::timeline::RemapTimeMap;
use fx_core::{Depth, Env, Frame, LayerSet, MemLayer, ParamCurves, Pixels, Rational, TimeBase};
use std::sync::Arc;

const W: usize = 48;
const H: usize = 36;
/// The motion of the uniform scenario, in pixels from A to B.
const V: (i32, i32) = (8, 4);

fn hash(x: i64, y: i64, seed: u64) -> u64 {
    let mut z = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ seed.wrapping_mul(0x1656_67B1_9E37_79F9);
    z ^= z >> 29;
    z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^= z >> 32;
    z = z.wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Noise colour of the infinite plane at `(x, y)`, three channels in `0..1`.
fn noise(x: i64, y: i64, seed: u64) -> [f64; 3] {
    let h = hash(x, y, seed);
    [(h & 0xffff) as f64 / 65535.0, ((h >> 16) & 0xffff) as f64 / 65535.0, ((h >> 32) & 0xffff) as f64 / 65535.0]
}

/// An opaque engine image in the format, colours from `f`.
fn make(fmt: Fmt, w: usize, h: usize, f: impl Fn(usize, usize) -> [f64; 3]) -> Image {
    let mut img = Image::new(w, h, fmt);
    for y in 0..h {
        for x in 0..w {
            let c = f(x, y);
            let i = (y * w + x) * 4;
            match &mut img.samples {
                Samples::U8(v) => {
                    v[i] = 255;
                    for k in 0..3 {
                        v[i + 1 + k] = (c[k] * 255.0 + 0.5) as u8;
                    }
                }
                Samples::U16(v) => {
                    v[i] = 65535;
                    for k in 0..3 {
                        v[i + 1 + k] = (c[k] * 65535.0 + 0.5) as u16;
                    }
                }
                Samples::F32(v) => {
                    v[i] = 1.0;
                    for k in 0..3 {
                        v[i + 1 + k] = c[k] as f32;
                    }
                }
            }
        }
    }
    img
}

/// A sample of an image as f64 (raw engine value).
fn sample(img: &Image, x: usize, y: usize, c: usize) -> f64 {
    let i = (y * img.width + x) * 4 + c;
    match &img.samples {
        Samples::U8(v) => f64::from(v[i]),
        Samples::U16(v) => f64::from(v[i]),
        Samples::F32(v) => f64::from(v[i]),
    }
}

/// The frames of the uniform scenario: `A` is the noise plane, `B(p) = A(p - V)`, both of the size `W x H`
/// but cut from one larger plane so that the moved content is real data at the borders too.
fn pair(fmt: Fmt) -> (Image, Image) {
    let a = make(fmt, W, H, |x, y| noise(x as i64, y as i64, 7));
    let b = make(fmt, W, H, |x, y| noise(x as i64 - i64::from(V.0), y as i64 - i64::from(V.1), 7));
    (a, b)
}

fn uniform(w: usize, h: usize, v: (f32, f32)) -> FlowField {
    FlowField { width: w, height: h, data: (0..w * h).flat_map(|_| [v.0, v.1]).collect() }
}

/// The true flows of the scenario: `F_AB = +V`, `F_BA = -V`, on the grid of the images.
fn true_flows() -> Flows {
    Flows {
        f_ab: Some(uniform(W, H, (V.0 as f32, V.1 as f32))),
        f_ba: Some(uniform(W, H, (-V.0 as f32, -V.1 as f32))),
    }
}

/// The selection of a frame pair `A = 3`, `B = 4` with weight `wa` and mode `interp`.
fn sel(interp: i32, wa: f32) -> Selection {
    let s0 = 3.0f32 + (1.0f32 - wa);
    Selection { s0, s1: s0, a: 3, b: 4, wa, exact: false, vectors_quality: 4, interp }
}

fn cfg(fmt: Fmt) -> EngineSettings {
    let mut s = settings(W, H, fmt);
    s.vectors_quality = 4;
    s
}

/// The raw values of `got` and `want` agree at every pixel at least `margin` away from the edges
/// (floats within 2e-6, integers exactly); the message names the first difference.
fn same_inside(got: &Image, want: &Image, margin: usize) -> std::result::Result<(), String> {
    assert!(got.same_shape(want));
    for y in margin..got.height - margin {
        for x in margin..got.width - margin {
            for c in 0..4 {
                let (g, w) = (sample(got, x, y, c), sample(want, x, y, c));
                let tol = if got.fmt() == Fmt::F32 { 2e-6 } else { 0.0 };
                if (g - w).abs() > tol {
                    return Err(format!("pixel ({x}, {y}) channel {c}: got {g}, want {w}"));
                }
            }
        }
    }
    Ok(())
}

/// `A` moved by whole pixels: `out(p) = A(p - d)`, computed by indexing; pixels whose source lies outside
/// the picture stay zero (callers compare the interior only).
fn moved(a: &Image, d: (i32, i32)) -> Image {
    let mut out = Image::new(a.width, a.height, a.fmt());
    for y in 0..a.height as i32 {
        for x in 0..a.width as i32 {
            let (sx, sy) = (x - d.0, y - d.1);
            if sx < 0 || sy < 0 || sx >= a.width as i32 || sy >= a.height as i32 {
                continue;
            }
            for c in 0..4 {
                let v = sample(a, sx as usize, sy as usize, c);
                let i = (y as usize * a.width + x as usize) * 4 + c;
                match &mut out.samples {
                    Samples::U8(s) => s[i] = v as u8,
                    Samples::U16(s) => s[i] = v as u16,
                    Samples::F32(s) => s[i] = v as f32,
                }
            }
        }
    }
    out
}

const MARGIN: usize = 10;

#[test]
fn the_scenario_is_what_it_says() {
    // B(p) = A(p - V): moving A by V gives B in the interior
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let (a, b) = pair(fmt);
        same_inside(&moved(&a, V), &b, MARGIN).unwrap();
    }
}

#[test]
fn nearest_warps_the_nearer_frame_by_its_distance_with_the_flow_the_rule_picks() {
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let (a, b) = pair(fmt);
        let s = cfg(fmt);
        // wA >= 0.5: frame A, flow F_BA, amount 1 - wA. wA = 0.75: A moved back by a quarter of V = (2, 1)
        // along -F_BA... A(p + 0.25 * F_BA(p)) = A(p - (2, 1)), the frame a quarter of the way to B
        for (wa, shift) in [(0.75f32, (2, 1)), (0.5, (4, 2)), (1.0, (0, 0))] {
            // only F_BA is available: the rule must not touch F_AB
            let flows = Flows { f_ab: None, f_ba: true_flows().f_ba };
            let got = compose_vectors(&s, &sel(0, wa), &a, &b, &flows).unwrap();
            same_inside(&got, &moved(&a, shift), MARGIN).unwrap_or_else(|e| panic!("{fmt:?} wA {wa}: {e}"));
        }
        // wA < 0.5: frame B, flow F_AB, amount wA. wA = 0.25: B(p + 0.25 * F_AB) = B(p + (2, 1)) = A(p - (6, 3))
        for (wa, shift) in [(0.25f32, (6, 3)), (0.0, (8, 4))] {
            let flows = Flows { f_ab: true_flows().f_ab, f_ba: None };
            let got = compose_vectors(&s, &sel(0, wa), &a, &b, &flows).unwrap();
            same_inside(&got, &moved(&a, shift), MARGIN).unwrap_or_else(|e| panic!("{fmt:?} wA {wa}: {e}"));
        }
        // the flow that is not the rule's is an error rather than a guess
        let only_ab = Flows { f_ab: true_flows().f_ab, f_ba: None };
        assert!(matches!(compose_vectors(&s, &sel(0, 0.75), &a, &b, &only_ab), Err(FxError::Mismatch(_))));
        let only_ba = Flows { f_ab: None, f_ba: true_flows().f_ba };
        assert!(matches!(compose_vectors(&s, &sel(0, 0.25), &a, &b, &only_ba), Err(FxError::Mismatch(_))));
    }
}

#[test]
fn the_nearest_rule_is_the_one_of_section_14_1() {
    // (wA < 0.5 && code < 2) || (wA >= 0.5 && code > 1): X = A, Y = B, flow F_AB
    for code in 0..4 {
        for wa in [0.0f32, 0.25, 0.499_999_97, 0.5, 0.75, 1.0] {
            let want = (wa < 0.5 && code < 2) || (wa >= 0.5 && code > 1);
            assert_eq!(nearest_uses_f_ab(wa, code), want, "wA {wa} code {code}");
        }
    }
    // for the inverse warps (codes 0 and 1) the flow is F_BA while the warp moves A, F_AB while it moves B
    assert!(!nearest_uses_f_ab(0.5, 0) && nearest_uses_f_ab(0.4999, 0));
}

/// The blend formula for one raw sample, worked out here: u8 and u16
/// `trunc(((b - a) * w + a) + 0.5)` in single precision, float `((b - a) * w + a) + 0.0`.
fn mix(fmt: Fmt, a: f64, b: f64, w: f32) -> f64 {
    let (a, b) = (a as f32, b as f32);
    let v = (b - a) * w + a;
    match fmt {
        Fmt::F32 => f64::from(v + 0.0),
        _ => f64::from((v + 0.5) as i64 as u32),
    }
}

#[test]
fn blend_warps_a_with_f_ba_by_frac_and_b_with_f_ab_by_wa_then_mixes_with_frac() {
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let (a, b) = pair(fmt);
        let s = cfg(fmt);
        let zero = uniform(W, H, (0.0, 0.0));
        let truth = true_flows();
        for wa in [0.25f32, 0.5, 0.75] {
            let frac = 1.0f32 - wa;
            // (1) the exact flows: both warped frames are the exact in-between frame A(p - frac * V), so the
            // mix is that frame whatever the weight is
            let got = compose_vectors(&s, &sel(1, wa), &a, &b, &truth).unwrap();
            let want = moved(&a, ((frac * V.0 as f32) as i32, (frac * V.1 as f32) as i32));
            same_inside(&got, &want, MARGIN).unwrap_or_else(|e| panic!("{fmt:?} wA {wa}: {e}"));
            // (2) F_BA = 0: tA is A itself; tB = B(p + wA * V): the weight frac is on tB
            let flows = Flows { f_ab: truth.f_ab.clone(), f_ba: Some(zero.clone()) };
            let got = compose_vectors(&s, &sel(1, wa), &a, &b, &flows).unwrap();
            let tb = moved(&b, ((-wa * V.0 as f32) as i32, (-wa * V.1 as f32) as i32));
            for y in MARGIN..H - MARGIN {
                for x in MARGIN..W - MARGIN {
                    for c in 0..4 {
                        let want = mix(fmt, sample(&a, x, y, c), sample(&tb, x, y, c), frac);
                        assert_eq!(sample(&got, x, y, c), want, "{fmt:?} wA {wa} F_BA = 0, ({x}, {y}) {c}");
                    }
                }
            }
            // (3) F_AB = 0: tB is B itself; tA = A(p + frac * F_BA) = A(p - frac * V)
            let flows = Flows { f_ab: Some(zero.clone()), f_ba: truth.f_ba.clone() };
            let got = compose_vectors(&s, &sel(1, wa), &a, &b, &flows).unwrap();
            let ta = moved(&a, ((frac * V.0 as f32) as i32, (frac * V.1 as f32) as i32));
            for y in MARGIN..H - MARGIN {
                for x in MARGIN..W - MARGIN {
                    for c in 0..4 {
                        let want = mix(fmt, sample(&ta, x, y, c), sample(&b, x, y, c), frac);
                        assert_eq!(sample(&got, x, y, c), want, "{fmt:?} wA {wa} F_AB = 0, ({x}, {y}) {c}");
                    }
                }
            }
        }
    }
}

#[test]
fn the_blend_weight_snaps_at_the_ends_of_the_interval() {
    // frac below 0.005: the blend returns tA, and tA with amount < 0.005 is A itself (a copy), whatever the flows
    for fmt in [Fmt::U8, Fmt::F32] {
        let (a, b) = pair(fmt);
        let s = cfg(fmt);
        let got = compose_vectors(&s, &sel(1, 0.998), &a, &b, &true_flows()).unwrap();
        assert_eq!(got, a, "{fmt:?}");
        // frac above 0.995: tB is B moved by wA * V which is < 0.005 px: a copy of B
        let got = compose_vectors(&s, &sel(1, 0.002), &a, &b, &true_flows()).unwrap();
        assert_eq!(got, b, "{fmt:?}");
    }
}

#[test]
fn weighted_blend_sharpens_each_warped_frame_against_its_own_unwarped_frame_then_blends_with_frac() {
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let (a, b) = pair(fmt);
        let s = cfg(fmt);
        // lopsided flows so that the two warped frames differ from the unwarped frames and from
        // each other
        let flows = Flows { f_ab: Some(uniform(W, H, (6.0, 2.0))), f_ba: Some(uniform(W, H, (-5.0, -3.0))) };
        for wa in [0.25f32, 0.5, 0.8] {
            let frac = 1.0f32 - wa;
            let mut ta = Image::new(W, H, fmt);
            let mut tb = Image::new(W, H, fmt);
            inv_warp(&mut ta, &a, flows.f_ba.as_ref().unwrap(), frac, WarpOpts::default()).unwrap();
            inv_warp(&mut tb, &b, flows.f_ab.as_ref().unwrap(), wa, WarpOpts::default()).unwrap();
            let (pa, pb) = (ta.clone(), tb.clone());
            sharpen_step(&mut ta, &a, frac);
            sharpen_step(&mut tb, &b, wa);
            assert_ne!(ta, pa, "the sharpening changes the warped frame ({fmt:?}, wA {wa})");
            assert_ne!(tb, pb);
            let mut want = Image::new(W, H, fmt);
            blend(&mut want, &ta, &tb, frac);
            let got = compose_vectors(&s, &sel(2, wa), &a, &b, &flows).unwrap();
            assert_eq!(got, want, "{fmt:?} wA {wa}");
            // and it differs from the plain Blend of the same warps
            let blended = compose_vectors(&s, &sel(1, wa), &a, &b, &flows).unwrap();
            assert_ne!(got, blended);
        }
    }
}

#[test]
fn weighted_blend_equals_blend_where_the_warp_changes_nothing() {
    // constant frames and any flow: the warped frame equals the unwarped frame, the change mask
    // is empty and the sharpened copy is never used
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let a = make(fmt, W, H, |_, _| [0.25, 0.5, 0.75]);
        let b = make(fmt, W, H, |_, _| [0.75, 0.25, 0.5]);
        let s = cfg(fmt);
        let flows = Flows { f_ab: Some(uniform(W / 2, H / 2, (3.0, -2.0))), f_ba: Some(uniform(W / 2, H / 2, (-1.0, 2.0))) };
        for wa in [0.1f32, 0.25, 0.5, 0.75, 0.9] {
            let blended = compose_vectors(&s, &sel(1, wa), &a, &b, &flows).unwrap();
            let weighted = compose_vectors(&s, &sel(2, wa), &a, &b, &flows).unwrap();
            assert_eq!(blended, weighted, "{fmt:?} wA {wa}");
        }
    }
}

#[test]
fn blur_compensation_runs_last_with_f_ab_and_only_when_the_source_moves_and_the_slider_is_up() {
    // wA = 1: frac = 0, so the composition is a copy of A (the blend snaps), and what is left is the tail.
    // Speed 200 percent: s1 - s0 = 2, amount = (2 - 1) * blur = 1. The flow F_AB = (4, 0) is whole pixels
    // along x: kx = (0.5 / rx) * 1 = 0.5, v = (2, 0), n = trunc(2 + 1) = 3, step (1, 0): taps x - 2 ..= x + 2
    // of the source, a box of 5 pixels along the row (bilinear taps on whole pixels are exact).
    let fmt = Fmt::U8;
    let a = make(fmt, W, H, |x, y| noise(x as i64, y as i64, 3));
    let b = make(fmt, W, H, |x, y| noise(x as i64, y as i64, 4));
    let mut s = cfg(fmt);
    s.blur = 1.0;
    let mut sl = sel(1, 1.0);
    sl.s0 = 3.0;
    sl.s1 = 5.0;
    let flows = Flows { f_ab: Some(uniform(W, H, (4.0, 0.0))), f_ba: Some(uniform(W, H, (9.0, 9.0))) };
    let got = compose_vectors(&s, &sl, &a, &b, &flows).unwrap();
    for y in 0..H {
        for x in 2..W - 2 {
            for c in 0..4 {
                let sum: f32 = (x - 2..=x + 2).map(|i| sample(&a, i, y, c) as f32).fold(0.0, |acc, v| acc + v);
                // single precision: (1 / count) * sum, then trunc(v + 0.5)
                let want = (((1.0f32 / 5.0f32) * sum) + 0.5f32) as i64 as f64;
                let alpha_ok = c != 0 || want == 255.0;
                assert!(alpha_ok);
                assert_eq!(sample(&got, x, y, c), want, "({x}, {y}) channel {c}");
            }
        }
    }
    // the same render without the slider, or with a static source position, is the plain composition
    let mut off = s.clone();
    off.blur = 0.0;
    assert_eq!(compose_vectors(&off, &sl, &a, &b, &flows).unwrap(), a);
    let mut still = sl;
    still.s1 = still.s0;
    assert_eq!(compose_vectors(&s, &still, &a, &b, &flows).unwrap(), a);
    // Nearest takes the tail's flow from the single flow it has
    let only_ba = Flows { f_ab: None, f_ba: Some(uniform(W, H, (4.0, 0.0))) };
    let mut near = sel(0, 1.0);
    near.s0 = 3.0;
    near.s1 = 5.0;
    let got_nearest = compose_vectors(&s, &near, &a, &b, &only_ba).unwrap();
    for y in 0..H {
        for x in 2..W - 2 {
            let sum: f32 = (x - 2..=x + 2).map(|i| sample(&a, i, y, 1) as f32).fold(0.0, |acc, v| acc + v);
            assert_eq!(sample(&got_nearest, x, y, 1), (((1.0f32 / 5.0f32) * sum) + 0.5f32) as i64 as f64);
        }
    }
}

#[test]
fn slow_motion_blur_compensation_sharpens_along_the_flow() {
    // speed below 100 percent: amount = (|s1 - s0| - 1) * blur < 0, the flow directed unsharp
    // mask `lerp(blurred, image, amount)`: the result is not the frame and not a plain blur
    let fmt = Fmt::U8;
    let a = make(fmt, W, H, |x, y| noise(x as i64, y as i64, 3));
    let b = make(fmt, W, H, |x, y| noise(x as i64, y as i64, 4));
    let mut s = cfg(fmt);
    s.blur = 1.0;
    let mut sl = sel(1, 1.0);
    sl.s0 = 3.0;
    sl.s1 = 3.5;
    let flows = Flows { f_ab: Some(uniform(W, H, (4.0, 0.0))), f_ba: Some(uniform(W, H, (4.0, 0.0))) };
    let got = compose_vectors(&s, &sl, &a, &b, &flows).unwrap();
    assert_ne!(got, a);
    let mut want = a.clone();
    crate::warp::blur_compensation(&mut want, flows.f_ab.as_ref().unwrap(), 3.0, 3.5, 1.0, true).unwrap();
    assert_eq!(got, want);
}

#[test]
fn bad_codes_and_modes_are_bad_parameters_and_every_code_from_0_to_3_is_served() {
    let (a, b) = pair(Fmt::U8);
    let flows = true_flows();
    for interp in 0..=2 {
        for code in 0..=3 {
            let mut s = cfg(Fmt::U8);
            s.warp_mode = code;
            compose_vectors(&s, &sel(interp, 0.75), &a, &b, &flows).unwrap_or_else(|e| panic!("code {code} interp {interp}: {e}"));
        }
        let mut s = cfg(Fmt::U8);
        s.warp_mode = 7;
        assert!(matches!(compose_vectors(&s, &sel(interp, 0.75), &a, &b, &flows), Err(FxError::BadParam { .. })));
        s.warp_mode = -1;
        assert!(matches!(compose_vectors(&s, &sel(interp, 0.75), &a, &b, &flows), Err(FxError::BadParam { .. })));
    }
    // the solver entry refuses a bad code before it solves anything
    let mut s = cfg(Fmt::U8);
    s.warp_mode = 4;
    assert!(matches!(solve_flows(&s, &sel(1, 0.5), &a, &b), Err(FxError::BadParam { .. })));
    // an unknown interpolation mode is a bad parameter
    let s = cfg(Fmt::U8);
    assert!(matches!(compose_vectors(&s, &sel(3, 0.5), &a, &b, &flows), Err(FxError::BadParam { .. })));
    assert!(matches!(solve_flows(&s, &sel(-1, 0.5), &a, &b), Err(FxError::BadParam { .. })));
}

/// The forward warp codes through `compose_vectors`: exactly the compositions of `forward.rs` with the flow pairing of
/// 14.2 (the pushed frame uses its OWN flow: A with F_AB, B with F_BA), code 2 with the options of the "360 Video"
/// checkbox (Blend only), code 3 without, for every mode, depth and weight.
#[test]
fn the_forward_codes_compose_with_the_compositions_of_forward_rs() {
    use crate::forward::{
        blend_forward, blend_forward_fill, nearest_forward, nearest_forward_fill, weighted_forward, weighted_forward_fill,
    };
    use crate::smartblend::SmartOpts;
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let a = translucent(fmt, W, H, 21);
        let b = translucent(fmt, W, H, 22);
        let flows = Flows { f_ab: Some(uniform(W / 2, H / 2, (3.0, 1.0))), f_ba: Some(uniform(W / 2, H / 2, (-2.5, -1.5))) };
        let (f_ab, f_ba) = (flows.f_ab.clone().unwrap(), flows.f_ba.clone().unwrap());
        for wa in [0.25f32, 0.5, 0.75] {
            for binary in [false, true] {
                let opts = SmartOpts { binary_alpha: binary, is360: false, ..SmartOpts::default() };
                let mut s = cfg(fmt);
                s.video_360 = binary;
                let run = |code: i32, interp: i32| {
                    let mut s = s.clone();
                    s.warp_mode = code;
                    compose_vectors(&s, &sel(interp, wa), &a, &b, &flows).unwrap()
                };
                let mut want = Image::new(W, H, fmt);
                // Nearest: the single flow is F_AB for wA >= 0.5 (the opposite of the inverse warps), F_BA below
                let single = if wa >= 0.5 { &f_ab } else { &f_ba };
                nearest_forward(&mut want, &a, &b, single, wa).unwrap();
                assert_eq!(run(3, 0), want, "{fmt:?} code 3 nearest wA {wa}");
                let mut want = Image::new(W, H, fmt);
                nearest_forward_fill(&mut want, &a, &b, single, wa, opts).unwrap();
                assert_eq!(run(2, 0), want, "{fmt:?} code 2 nearest wA {wa} binary {binary}");
                let mut want = Image::new(W, H, fmt);
                blend_forward(&mut want, &a, &f_ab, &b, &f_ba, wa).unwrap();
                assert_eq!(run(3, 1), want, "{fmt:?} code 3 blend wA {wa}");
                let mut want = Image::new(W, H, fmt);
                blend_forward_fill(&mut want, &a, &f_ab, &b, &f_ba, wa, opts).unwrap();
                assert_eq!(run(2, 1), want, "{fmt:?} code 2 blend wA {wa} binary {binary}");
                let mut want = Image::new(W, H, fmt);
                weighted_forward(&mut want, &a, &f_ab, &b, &f_ba, wa).unwrap();
                assert_eq!(run(3, 2), want, "{fmt:?} code 3 weighted wA {wa}");
                let mut want = Image::new(W, H, fmt);
                weighted_forward_fill(&mut want, &a, &f_ab, &b, &f_ba, wa, opts).unwrap();
                assert_eq!(run(2, 2), want, "{fmt:?} code 2 weighted wA {wa} binary {binary}");
            }
        }
    }
}

#[test]
fn forward_blend_and_nearest_push_each_frame_along_its_own_flow_to_the_in_between_frame() {
    // B is A moved by (8, 4): the forward warps push A by frac * (8, 4) along F_AB and B back by wA * (-8, -4) along
    // F_BA, both onto the exact in between frame; whole pixel amounts make the warps exact shifts
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let (a, b) = pair(fmt);
        for code in [2, 3] {
            let mut s = cfg(fmt);
            s.warp_mode = code;
            for interp in 0..=1 {
                for wa in [0.25f32, 0.5, 0.75] {
                    let frac = 1.0f32 - wa;
                    let got = compose_vectors(&s, &sel(interp, wa), &a, &b, &true_flows()).unwrap();
                    let want = moved(&a, ((frac * V.0 as f32) as i32, (frac * V.1 as f32) as i32));
                    // the textured warp reads a mip pyramid: a whole pixel shift keeps level 0 for 8 bit data (exact), but the
                    // 16 bit and float runs differ from the plain shift by a few units of the last place of a tiny level blend
                    let white = match fmt {
                        Fmt::U8 => 255.0,
                        Fmt::U16 => 65535.0,
                        Fmt::F32 => 1.0,
                    };
                    let tol = if fmt == Fmt::U8 { 0.0 } else { 1e-3 * white };
                    for y in MARGIN..H - MARGIN {
                        for x in MARGIN..W - MARGIN {
                            for c in 0..4 {
                                let (g, w) = (sample(&got, x, y, c), sample(&want, x, y, c));
                                assert!((g - w).abs() <= tol, "{fmt:?} code {code} interp {interp} wA {wa} ({x}, {y}) channel {c}: {g} against {w}");
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn code_2_fills_what_the_forward_warp_leaves_empty_and_code_3_leaves_it_transparent() {
    // Nearest at wA = 0.75 pushes A by (2, 1): the two left columns and the top row receive nothing
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let (a, b) = pair(fmt);
        let white = match fmt {
            Fmt::U8 => 255.0,
            Fmt::U16 => 65535.0,
            Fmt::F32 => 1.0,
        };
        let mut s3 = cfg(fmt);
        s3.warp_mode = 3;
        let got3 = compose_vectors(&s3, &sel(0, 0.75), &a, &b, &true_flows()).unwrap();
        assert_eq!(sample(&got3, 0, 10, 0), 0.0, "{fmt:?} code 3: a hole is transparent black");
        assert_eq!(sample(&got3, 1, 10, 0), 0.0);
        assert_eq!(sample(&got3, 10, 0, 0), 0.0);
        assert_eq!(sample(&got3, 20, 20, 0), white);
        let mut s2 = cfg(fmt);
        s2.warp_mode = 2;
        let got2 = compose_vectors(&s2, &sel(0, 0.75), &a, &b, &true_flows()).unwrap();
        for y in 0..H {
            for x in 0..W {
                assert_eq!(sample(&got2, x, y, 0), white, "{fmt:?} code 2 ({x}, {y}) is not opaque");
            }
        }
        // the known pixels are the same in both codes
        same_inside(&got2, &got3, 3).unwrap();
    }
}

#[test]
fn the_blur_tail_follows_the_forward_codes() {
    let fmt = Fmt::U8;
    let (a, b) = pair(fmt);
    let mut sl = sel(1, 1.0);
    sl.s0 = 3.0;
    sl.s1 = 5.0;
    let flows = true_flows();
    for code in [2, 3] {
        let mut s = cfg(fmt);
        s.warp_mode = code;
        s.blur = 1.0;
        let got = compose_vectors(&s, &sl, &a, &b, &flows).unwrap();
        let mut off = s;
        off.blur = 0.0;
        let mut want = compose_vectors(&off, &sl, &a, &b, &flows).unwrap();
        crate::warp::blur_compensation(&mut want, flows.f_ab.as_ref().unwrap(), 3.0, 5.0, 1.0, true).unwrap();
        assert_eq!(got, want, "code {code}");
        assert_ne!(got, compose_vectors(&off, &sl, &a, &b, &flows).unwrap());
    }
}

/// Real flows from the solver on a translating smooth texture, the forward codes against the exact in between frame.
#[test]
fn forward_codes_with_solved_flows_are_closer_to_the_truth_than_the_frame_blend() {
    let (w, h) = (96usize, 64usize);
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let a = wave(fmt, w, h, 0.0, 0.0);
        let b = wave(fmt, w, h, 6.0, 3.0);
        let mut s = settings(w, h, fmt);
        s.vectors_quality = 4;
        let scale = match fmt {
            Fmt::U8 => 255.0,
            Fmt::U16 => 65535.0,
            Fmt::F32 => 1.0,
        };
        for wa in [0.25f32, 0.5, 0.75] {
            let frac = 1.0f32 - wa;
            let truth = wave(fmt, w, h, f64::from(frac) * 6.0, f64::from(frac) * 3.0);
            let mut selection = sel(1, wa);
            selection.vectors_quality = 4;
            let flows = solve_flows(&s, &selection, &a, &b).unwrap();
            let mut plain = Image::new(w, h, fmt);
            blend(&mut plain, &a, &b, frac);
            let err = |img: &Image| {
                let (mut sum, mut n) = (0.0, 0usize);
                for y in 12..h - 12 {
                    for x in 12..w - 12 {
                        for c in 1..4 {
                            sum += (sample(img, x, y, c) - sample(&truth, x, y, c)).abs();
                            n += 1;
                        }
                    }
                }
                sum / n as f64 / scale
            };
            let e_plain = err(&plain);
            for code in [2, 3] {
                for interp in 0..=1 {
                    let mut s = s.clone();
                    s.warp_mode = code;
                    let mut sl = selection;
                    sl.interp = interp;
                    // Nearest needs only the flow its rule names; the pair serves both
                    let got = compose_vectors(&s, &sl, &a, &b, &flows).unwrap();
                    let e = err(&got);
                    assert!(
                        e < 0.012 && e * 3.0 < e_plain,
                        "{fmt:?} code {code} interp {interp} wA {wa}: error {e:.5} against the frame blend {e_plain:.5}"
                    );
                }
            }
        }
    }
}

#[test]
fn the_forward_codes_give_the_same_bytes_in_pools_of_one_and_four_threads() {
    let pool = |n: usize| rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
    for fmt in [Fmt::U8, Fmt::F32] {
        let a = translucent(fmt, 67, 45, 31);
        let b = translucent(fmt, 67, 45, 32);
        let flows = Flows { f_ab: Some(uniform(34, 23, (3.0, 1.0))), f_ba: Some(uniform(34, 23, (-2.5, -1.5))) };
        for code in [2, 3] {
            for interp in 0..=2 {
                let mut s = settings(67, 45, fmt);
                s.vectors_quality = 4;
                s.warp_mode = code;
                s.blur = 1.0;
                let mut sl = sel(interp, 0.4);
                sl.s0 = 3.0;
                sl.s1 = 5.0;
                let run = |n: usize| pool(n).install(|| compose_vectors(&s, &sl, &a, &b, &flows).unwrap());
                assert_eq!(run(1), run(4), "{fmt:?} code {code} interp {interp}");
            }
        }
    }
}

/// Smart Blend (warping code 1) through `compose_vectors`: the compositions of `smartblend.rs` with the flows the
/// table of 14.2 names, for every mode and depth, and the variant selection of the "360 Video" checkbox.
#[test]
fn smart_blend_composes_with_the_compositions_of_smartblend_rs() {
    use crate::smartblend::{blend_smart, nearest_smart, weighted_blend_smart, SmartOpts};
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let a = translucent(fmt, W, H, 11);
        let b = translucent(fmt, W, H, 12);
        let flows = Flows { f_ab: Some(uniform(W, H, (6.0, 2.0))), f_ba: Some(uniform(W, H, (-5.0, -3.0))) };
        let (f_ab, f_ba) = (flows.f_ab.clone().unwrap(), flows.f_ba.clone().unwrap());
        for wa in [0.25f32, 0.5, 0.75] {
            for binary in [false, true] {
                let mut s = cfg(fmt);
                s.warp_mode = 1;
                s.smart_blend = true;
                s.video_360 = binary;
                let opts = SmartOpts { binary_alpha: binary, is360: false, ..SmartOpts::default() };
                // Nearest: the single flow of the rule, F_BA for wA >= 0.5, F_AB below
                let mut want = Image::new(W, H, fmt);
                let single = if wa >= 0.5 { &f_ba } else { &f_ab };
                nearest_smart(&mut want, &a, &b, single, wa, opts).unwrap();
                assert_eq!(compose_vectors(&s, &sel(0, wa), &a, &b, &flows).unwrap(), want, "{fmt:?} nearest wA {wa} binary {binary}");
                // Blend: the binary variant only here
                let mut want = Image::new(W, H, fmt);
                blend_smart(&mut want, &a, &f_ba, &b, &f_ab, wa, opts).unwrap();
                assert_eq!(compose_vectors(&s, &sel(1, wa), &a, &b, &flows).unwrap(), want, "{fmt:?} blend wA {wa} binary {binary}");
                // Weighted Blend ignores the options
                let mut want = Image::new(W, H, fmt);
                weighted_blend_smart(&mut want, &a, &f_ba, &b, &f_ab, wa, opts).unwrap();
                assert_eq!(compose_vectors(&s, &sel(2, wa), &a, &b, &flows).unwrap(), want, "{fmt:?} weighted wA {wa} binary {binary}");
            }
            // the checkbox changes the Blend result on translucent frames and nothing else
            let render = |interp: i32, binary: bool| {
                let mut s = cfg(fmt);
                s.warp_mode = 1;
                s.video_360 = binary;
                compose_vectors(&s, &sel(interp, wa), &a, &b, &flows).unwrap()
            };
            assert_ne!(render(1, false), render(1, true), "{fmt:?} wA {wa}: the binary alpha composite differs on translucent pixels");
            assert_eq!(render(0, false), render(0, true), "{fmt:?} nearest ignores the checkbox");
            assert_eq!(render(2, false), render(2, true), "{fmt:?} weighted blend ignores the checkbox");
        }
    }
}

#[test]
fn smart_blend_fills_what_the_warps_leave_empty_and_keeps_the_interior() {
    // B is A moved by (8, 4): with the true flows both warps (hold edges off) leave a strip without data at two
    // borders of the picture; Smart Blend takes the partner frame there and the hole fill closes what remains.
    // Every pixel is opaque and the interior is the exact in between frame
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let (a, b) = pair(fmt);
        let mut s = cfg(fmt);
        s.warp_mode = 1;
        let white = match fmt {
            Fmt::U8 => 255.0,
            Fmt::U16 => 65535.0,
            Fmt::F32 => 1.0,
        };
        for interp in 0..=1 {
            for wa in [0.25f32, 0.5, 0.75] {
                let got = compose_vectors(&s, &sel(interp, wa), &a, &b, &true_flows()).unwrap();
                for y in 0..H {
                    for x in 0..W {
                        assert_eq!(sample(&got, x, y, 0), white, "{fmt:?} interp {interp} wA {wa} ({x}, {y}) is not opaque");
                    }
                }
                let frac = 1.0f32 - wa;
                let want = moved(&a, ((frac * V.0 as f32) as i32, (frac * V.1 as f32) as i32));
                same_inside(&got, &want, MARGIN).unwrap_or_else(|e| panic!("{fmt:?} interp {interp} wA {wa}: {e}"));
            }
        }
    }
}

#[test]
fn flow_settings_carry_quality_sensitivity_prep_and_the_video_range_luma() {
    let mut s = cfg(Fmt::U8);
    s.sensitivity = 0.55;
    s.prep_contrast_edge = true;
    s.prep_delinearize = true;
    s.video_360 = true;
    s.quality_high = false;
    let mut sl = sel(1, 0.5);
    sl.vectors_quality = 2;
    let fs = flow_settings(&s, &sl);
    assert_eq!((fs.quality, fs.hiq, fs.sensitivity), (2, false, 0.55));
    assert!(fs.prep_contrast_edge && fs.prep_delinearize && fs.video_range_luma);
    assert!(!fs.track_alpha && !fs.skip_refine);
    // plugin 03's Track RGB+A: alpha in the solver's luma
    let mut t = s.clone();
    t.video_360 = false;
    t.track_alpha = true;
    assert!(flow_settings(&t, &sl).track_alpha && !flow_settings(&t, &sl).video_range_luma);
    // the quality is the one after the segment override, not the popup
    s.vectors_quality = 4;
    assert_eq!(flow_settings(&s, &sl).quality, 2);
    // defaults: Best, high quality render, plain luma
    let fs = flow_settings(&cfg(Fmt::U8), &sel(1, 0.5));
    assert_eq!((fs.quality, fs.hiq, fs.prep_contrast_edge, fs.prep_delinearize, fs.video_range_luma), (4, true, false, false, false));
}

/// A smooth texture that translates: the frame at shift `(dx, dy)`.
fn wave(fmt: Fmt, w: usize, h: usize, dx: f64, dy: f64) -> Image {
    make(fmt, w, h, |x, y| {
        let (px, py) = (x as f64 - dx, y as f64 - dy);
        let t = std::f64::consts::TAU;
        std::array::from_fn(|c| {
            let c = c as f64;
            0.5 + 0.5
                * (0.4 * (t * (px / 23.0 + py / 31.0) + c).sin() + 0.35 * (t * (px / 14.0 - py / 17.0) + 2.0 * c).sin() + 0.25 * (t * px / 9.0 + 1.0).sin())
        })
    })
}

#[test]
fn solve_flows_runs_the_solves_the_table_names() {
    let (w, h) = (64, 48);
    let a = wave(Fmt::U8, w, h, 0.0, 0.0);
    let b = wave(Fmt::U8, w, h, 3.0, 1.0);
    let s = cfg(Fmt::U8);
    let fs = flow_settings(&s, &sel(1, 0.5));
    let f_ab = crate::flow::solve(&a, &b, &fs).unwrap();
    let f_ba = crate::flow::solve(&b, &a, &fs).unwrap();
    // Blend and Weighted Blend: both flows, X = A, Y = B for F_AB and X = B, Y = A for F_BA
    for interp in [1, 2] {
        let flows = solve_flows(&s, &sel(interp, 0.5), &a, &b).unwrap();
        assert_eq!(flows.f_ab.as_ref(), Some(&f_ab));
        assert_eq!(flows.f_ba.as_ref(), Some(&f_ba));
    }
    // Nearest: one solve. wA >= 0.5 needs F_BA (X = B, Y = A), wA < 0.5 needs F_AB (X = A, Y = B)
    let flows = solve_flows(&s, &sel(0, 0.75), &a, &b).unwrap();
    assert_eq!((flows.f_ab.as_ref(), flows.f_ba.as_ref()), (None, Some(&f_ba)));
    let flows = solve_flows(&s, &sel(0, 0.25), &a, &b).unwrap();
    assert_eq!((flows.f_ab.as_ref(), flows.f_ba.as_ref()), (Some(&f_ab), None));
    // the direction convention: A(p) ~ B(p + F_AB(p)), so the interior mean of F_AB is the motion A -> B in
    // pixels of the flow grid (the quality Best halves the resolution: reduction 1)
    let mean = |f: &FlowField| {
        let (mut sx, mut sy, mut n) = (0.0f64, 0.0f64, 0usize);
        for y in 8..f.height - 8 {
            for x in 8..f.width - 8 {
                let v = f.at(x, y);
                sx += f64::from(v[0]);
                sy += f64::from(v[1]);
                n += 1;
            }
        }
        (sx / n as f64, sy / n as f64)
    };
    let (ab, ba) = (mean(&f_ab), mean(&f_ba));
    assert!((ab.0 - 1.5).abs() < 0.15 && (ab.1 - 0.5).abs() < 0.15, "F_AB {ab:?}");
    assert!((ba.0 + 1.5).abs() < 0.15 && (ba.1 + 0.5).abs() < 0.15, "F_BA {ba:?}");
}

/// A clip layer from engine images (opaque frames, so the wire frame is the image).
fn layer_of(images: &[&Image]) -> MemLayer {
    let frames = images
        .iter()
        .map(|img| match &img.samples {
            Samples::U8(v) => Frame {
                width: img.width,
                height: img.height,
                pixels: Pixels::U8(v.chunks_exact(4).map(|p| [p[1], p[2], p[3], p[0]]).collect()),
            },
            _ => unreachable!("8 bpc only"),
        })
        .collect();
    MemLayer::new(TimeBase::default_for_rate(Rational::new(30, 1).unwrap()), frames).unwrap()
}

fn run_render(input: &MemLayer, layers: &LayerSet, s: &EngineSettings, mock: &Mock) -> (Selection, Result<Rendered>) {
    let params = ParamCurves::new(Vec::new());
    let env = Env { params: &params, input, layers };
    let map = RemapTimeMap::new(mock, s.rates, s.current_time);
    let host = EnvHost::new(&env, s, &map, LayerIds::default()).unwrap();
    let selection = select_frames(s, &host).unwrap();
    (selection, render(s, &host))
}

#[test]
fn the_tracking_layer_feeds_the_solver_and_the_retimed_layer_feeds_the_output() {
    let (w, h) = (64, 48);
    // the retimed layer is still (the same noise twice): its own flow would be zero. The tracking layer moves.
    let still = make(Fmt::U8, w, h, |x, y| noise(x as i64, y as i64, 11));
    let moving0 = wave(Fmt::U8, w, h, 0.0, 0.0);
    let moving1 = wave(Fmt::U8, w, h, 3.0, 1.0);
    let input = layer_of(&[&still, &still]);
    let mut s = settings(w, h, Fmt::U8);
    s.vectors_quality = 4;
    s.current_time = 100; // output frame 1 at half speed: position 0.5, frames 0 and 1
    for interp in [0, 1, 2] {
        s.interp = interp;
        // without a tracking layer: the solver reads the retimed layer
        s.no_tracking_layer = true;
        let (selection, got) = run_render(&input, &LayerSet::new(), &s, &Mock::speed(50.0));
        assert_eq!((selection.a, selection.b, selection.wa, selection.exact), (0, 1, 0.5, false));
        let flows = solve_flows(&s, &selection, &still, &still).unwrap();
        let want = compose_vectors(&s, &selection, &still, &still, &flows).unwrap();
        let Ok(Rendered::Image(got_plain)) = got else { panic!("expected an image") };
        assert_eq!(got_plain, want, "interp {interp}, no tracking layer");
        // with the tracking layer: the flows come from it, the pixels from the retimed layer
        s.no_tracking_layer = false;
        let mut layers = LayerSet::new();
        layers.insert(crate::planner::LAYER_ALT_B, Arc::new(layer_of(&[&moving0, &moving1])));
        let (selection, got) = run_render(&input, &layers, &s, &Mock::speed(50.0));
        let flows = solve_flows(&s, &selection, &moving0, &moving1).unwrap();
        let want = compose_vectors(&s, &selection, &still, &still, &flows).unwrap();
        let Ok(Rendered::Image(got_tracked)) = got else { panic!("expected an image") };
        assert_eq!(got_tracked, want, "interp {interp}, tracking layer");
        assert_ne!(got_tracked, got_plain, "the tracking layer changes the flows ({interp})");
    }
}

#[test]
fn a_tracking_layer_of_another_size_is_the_red_fill_and_a_missing_frame_is_transparent_input() {
    let (w, h) = (64, 48);
    let img = wave(Fmt::U8, w, h, 0.0, 0.0);
    let input = layer_of(&[&img, &img]);
    let mut s = settings(w, h, Fmt::U8);
    s.vectors_quality = 4;
    s.current_time = 100;
    s.no_tracking_layer = false;
    let mut layers = LayerSet::new();
    let small = wave(Fmt::U8, 32, 24, 0.0, 0.0);
    layers.insert(crate::planner::LAYER_ALT_B, Arc::new(layer_of(&[&small, &small])));
    let (_, got) = run_render(&input, &layers, &s, &Mock::speed(50.0));
    let Ok(Rendered::Error(frame)) = got else { panic!("expected the size mismatch fill") };
    assert_eq!((frame.width, frame.height, frame.depth()), (w, h, Depth::U8));
    // the clip ends: B = frame 2 does not exist and is transparent; the vector path still runs and the
    // result is a defined image
    s.no_tracking_layer = true;
    s.current_time = 300; // position 1.5: frames 1 and 2 of a two frame clip
    let (selection, got) = run_render(&input, &LayerSet::new(), &s, &Mock::speed(50.0));
    assert_eq!((selection.a, selection.b), (1, 2));
    assert!(matches!(got, Ok(Rendered::Image(_))));
}

#[test]
fn frames_too_small_for_their_halvings_are_searched_at_the_last_size_of_at_least_2x2() {
    // 3 x 2 at Best was a 1 x 1 working size; the reduction never halves below 2 x 2 (the tiny frames
    // rule, parity gap G7), so it is searched at 3 x 2 and the composition runs on those flows
    let a = make(Fmt::U8, 3, 2, |x, y| noise(x as i64, y as i64, 1));
    let b = make(Fmt::U8, 3, 2, |x, y| noise(x as i64, y as i64, 2));
    let mut s = settings(3, 2, Fmt::U8);
    s.vectors_quality = 4;
    for interp in 0..=2 {
        let selection = sel(interp, 0.5);
        for code in 0..=3 {
            // Nearest solves the one flow its warp code reads
            s.warp_mode = code;
            let flows = solve_flows(&s, &selection, &a, &b).unwrap();
            let f = flows.blur_flow().expect("a flow");
            assert_eq!((f.width, f.height), (3, 2), "interp {interp} code {code}");
            let img = compose_vectors(&s, &selection, &a, &b, &flows).unwrap();
            assert_eq!((img.width, img.height), (3, 2), "interp {interp} code {code}");
        }
    }
    // a side of 1 is never searched: the front-ends pass No Vectors for it (`prep::searchable`), and the
    // solver itself still refuses it with its own message
    let a = make(Fmt::U8, 3, 1, |x, y| noise(x as i64, y as i64, 1));
    let b = make(Fmt::U8, 3, 1, |x, y| noise(x as i64, y as i64, 2));
    let mut s = settings(3, 1, Fmt::U8);
    s.vectors_quality = 4;
    match solve_flows(&s, &sel(1, 0.5), &a, &b) {
        Err(FxError::Unsupported(m)) => {
            assert_eq!(m, "optical flow: working size 3x1 (output 3x1, reduction 0) is below 2x2, which is not supported")
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn the_codes_and_modes_are_checked_with_exact_texts() {
    // the warping codes 0 to 3 are served, with or without the plugin 01 checkbox: the code is what the core reads
    let sel1 = sel(1, 0.5);
    for (code, checkbox) in [(0, false), (1, true), (1, false), (0, true), (2, false), (3, false)] {
        let mut s = cfg(Fmt::U8);
        s.warp_mode = code;
        s.smart_blend = checkbox;
        check_supported(&s, &sel1).unwrap();
    }
    // a bad warping code and a bad interpolation mode are bad parameters, with their numbers in the reason
    let mut s = cfg(Fmt::U8);
    s.warp_mode = 9;
    match check_supported(&s, &sel1) {
        Err(FxError::BadParam { id, reason }) => assert_eq!((id.as_str(), reason.as_str()), ("warp_mode", "warping code 9 is not 0 to 3")),
        other => panic!("{other:?}"),
    }
    match check_supported(&cfg(Fmt::U8), &sel(5, 0.5)) {
        Err(FxError::BadParam { id, reason }) => {
            assert_eq!((id.as_str(), reason.as_str()), ("frame_interp", "interpolation mode 5 is not 0 (Nearest), 1 (Blend) or 2 (Weighted Blend)"))
        }
        other => panic!("{other:?}"),
    }
    // the "360 Video" checkbox is the video range luma of the solver and refuses nothing
    let mut s = cfg(Fmt::U8);
    s.video_360 = true;
    check_supported(&s, &sel1).unwrap();
}

#[test]
fn the_limits_of_the_blur_tail_reach_the_caller_unchanged() {
    // a flow of a million pixels with the slider up: the blur needs more than 65536 taps per
    // pixel, the warp refuses instead of running for minutes, and the vector path passes the
    // error on
    let (a, b) = pair(Fmt::U8);
    let mut s = cfg(Fmt::U8);
    s.blur = 10.0;
    let mut sl = sel(1, 1.0);
    sl.s0 = 3.0;
    sl.s1 = 9.0;
    let huge = Flows { f_ab: Some(uniform(W, H, (1.0e6, 0.0))), f_ba: Some(uniform(W, H, (1.0e6, 0.0))) };
    match compose_vectors(&s, &sl, &a, &b, &huge) {
        Err(FxError::Unsupported(m)) => assert!(m.contains("65536 taps"), "{m}"),
        other => panic!("expected the tap limit, got {other:?}"),
    }
    // flows too small for the warp are an error, not a panic
    let tiny = Flows { f_ab: Some(uniform(1, 1, (1.0, 1.0))), f_ba: Some(uniform(1, 1, (1.0, 1.0))) };
    assert!(compose_vectors(&cfg(Fmt::U8), &sel(1, 0.5), &a, &b, &tiny).is_err());
}

#[test]
fn a_flow_at_the_solver_resolution_composes_like_the_same_motion_at_the_image_resolution() {
    // the flow of the solver is smaller than the image and holds the motion in ITS pixels: the
    // warp samples it at (x * rx, y * ry) with rx = (flow width - 1) / (width - 1) and divides
    // the amount by rx, ry. On a 49 x 37 image a 25 x 19 flow has rx = ry = 1/2 and a 13 x 10
    // flow 1/4, exactly, so (4, 2) at half size and (2, 1) at a quarter size are the (8, 4)
    // pixels of the full size flow
    let (w, h) = (49usize, 37usize);
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let a = make(fmt, w, h, |x, y| noise(x as i64, y as i64, 7));
        let b = make(fmt, w, h, |x, y| noise(x as i64 - 8, y as i64 - 4, 7));
        let s = cfg(fmt);
        let full = Flows { f_ab: Some(uniform(w, h, (8.0, 4.0))), f_ba: Some(uniform(w, h, (-8.0, -4.0))) };
        for ((fw, fh), v) in [((25usize, 19usize), (4.0f32, 2.0f32)), ((13, 10), (2.0, 1.0))] {
            let reduced = Flows { f_ab: Some(uniform(fw, fh, v)), f_ba: Some(uniform(fw, fh, (-v.0, -v.1))) };
            for interp in 0..=2 {
                for wa in [0.25f32, 0.5, 0.75] {
                    let want = compose_vectors(&s, &sel(interp, wa), &a, &b, &full).unwrap();
                    let got = compose_vectors(&s, &sel(interp, wa), &a, &b, &reduced).unwrap();
                    if let Err(e) = same_inside(&got, &want, 0) {
                        panic!("{fmt:?} flow {fw}x{fh}, interp {interp}, wA {wa}: {e}");
                    }
                }
            }
        }
        // and the motion is the real one: the Blend of the exact flows is the in-between frame of the scenario
        let got = compose_vectors(&s, &sel(1, 0.5), &a, &b, &Flows { f_ab: Some(uniform(25, 19, (4.0, 2.0))), f_ba: Some(uniform(25, 19, (-4.0, -2.0))) }).unwrap();
        same_inside(&got, &moved(&a, (4, 2)), MARGIN).unwrap_or_else(|e| panic!("{fmt:?}: {e}"));
    }
}

/// Premultiplied random pixels with every alpha class: transparent black, translucent and opaque.
fn translucent(fmt: Fmt, w: usize, h: usize, seed: u64) -> Image {
    let mut img = Image::new(w, h, fmt);
    for y in 0..h {
        for x in 0..w {
            let hsh = hash(x as i64, y as i64, seed);
            let alpha = match hsh % 4 {
                0 => 0.0,
                1 => 1.0,
                _ => ((hsh >> 8) & 0xff) as f64 / 255.0,
            };
            let c = noise(x as i64, y as i64, seed ^ 0x55);
            let i = (y * w + x) * 4;
            match &mut img.samples {
                Samples::U8(v) => {
                    v[i] = (alpha * 255.0 + 0.5) as u8;
                    for k in 0..3 {
                        v[i + 1 + k] = ((c[k] * alpha * 255.0) as u8).min(v[i]);
                    }
                }
                Samples::U16(v) => {
                    v[i] = (alpha * 65535.0 + 0.5) as u16;
                    for k in 0..3 {
                        v[i + 1 + k] = ((c[k] * alpha * 65535.0) as u16).min(v[i]);
                    }
                }
                Samples::F32(v) => {
                    v[i] = alpha as f32;
                    for k in 0..3 {
                        v[i + 1 + k] = (c[k] * alpha) as f32;
                    }
                }
            }
        }
    }
    img
}

#[test]
fn translucent_frames_stay_valid_premultiplied_images_in_every_mode() {
    // colours never exceed alpha (up to the rounding of the format), no sample is NaN, and
    // wherever the blur tail ran a transparent pixel is black
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let a = translucent(fmt, W, H, 1);
        let b = translucent(fmt, W, H, 2);
        let flows = Flows { f_ab: Some(uniform(W / 2, H / 2, (3.5, -2.25))), f_ba: Some(uniform(W / 2, H / 2, (-4.0, 1.5))) };
        for interp in 0..=2 {
            for (blur, s0, s1) in [(0.0f32, 3.0f32, 3.5f32), (1.0, 3.0, 5.0), (1.0, 3.0, 3.5)] {
                for wa in [0.2f32, 0.5, 0.9] {
                    let mut s = cfg(fmt);
                    s.blur = blur;
                    let mut sl = sel(interp, wa);
                    sl.s0 = s0;
                    sl.s1 = s1;
                    let out = compose_vectors(&s, &sl, &a, &b, &flows).unwrap();
                    // the float lerp has no colour clamp: the unsharp steps (Weighted Blend, slow motion blur tail) can overshoot
                    let overshoots = fmt == Fmt::F32 && (interp == 2 || (blur > 0.005 && (s1 - s0).abs() < 1.0));
                    let tol = if fmt == Fmt::F32 { 1e-5 } else { 1.0 };
                    for y in 0..H {
                        for x in 0..W {
                            let alpha = sample(&out, x, y, 0);
                            for c in 1..4 {
                                let v = sample(&out, x, y, c);
                                assert!(
                                    v.is_finite() && (overshoots || (v >= -tol && v <= alpha + tol)),
                                    "{fmt:?} interp {interp} blur {blur} wA {wa} ({x}, {y}): colour {v} alpha {alpha}"
                                );
                                if blur > 0.005 && (s1 - s0).abs() > 1e-6 && alpha == 0.0 {
                                    assert_eq!(v, 0.0, "a transparent pixel is black after the blur tail");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---- Vectors In: the vector path with externally supplied flows -------------------------------------------

/// The scenario as a two frame layer (A, B = A moved by V) and `retime_with_flows` at `current_time` with a speed
/// map (time 100 is output frame 1; time 0 is the exact position 0).
fn run_external(s: &EngineSettings, mock: &Mock, flows: impl FnOnce(&Selection) -> Result<Flows>) -> Result<Rendered> {
    let (a, b) = pair(Fmt::U8);
    let input = layer_of(&[&a, &b]);
    let layers = LayerSet::new();
    let params = ParamCurves::new(Vec::new());
    let env = Env { params: &params, input: &input, layers: &layers };
    let map = RemapTimeMap::new(mock, s.rates, s.current_time);
    let host = EnvHost::new(&env, s, &map, LayerIds::default()).unwrap();
    crate::compose::retime_with_flows(s, &host, flows)
}

fn external_cfg(interp: i32, current_time: i32) -> EngineSettings {
    let mut s = settings(W, H, Fmt::U8);
    s.vectors_quality = 4;
    s.interp = interp;
    s.current_time = current_time;
    s
}

#[test]
fn injected_flows_replace_the_solver_and_give_the_composition_of_those_flows() {
    let (a, b) = pair(Fmt::U8);
    for interp in 0..=2 {
        let s = external_cfg(interp, 100);
        let mut seen = None;
        let got = run_external(&s, &Mock::speed(50.0), |sel| {
            seen = Some((sel.a, sel.b, sel.wa, sel.exact));
            Ok(true_flows())
        })
        .unwrap();
        // the closure got the frame pair of the render: A = 0, B = 1, wA = 0.5
        assert_eq!(seen, Some((0, 1, 0.5, false)), "interp {interp}");
        let Rendered::Image(got) = got else { panic!("expected an image") };
        // the same flows through compose_vectors directly
        let selection = Selection { s0: 0.5, s1: 0.5, a: 0, b: 1, wa: 0.5, exact: false, vectors_quality: 4, interp };
        let want = compose_vectors(&s, &selection, &a, &b, &true_flows()).unwrap();
        assert_eq!(got, want, "interp {interp}");
        // and with the true flows that is the in between frame, A moved by half of V (Weighted Blend sharpens the
        // changed pixels of a noise texture on top, so it is only compared with the composition above)
        if interp != 2 {
            same_inside(&got, &moved(&a, (4, 2)), MARGIN).unwrap_or_else(|e| panic!("interp {interp}: {e}"));
        }
    }
}

#[test]
fn the_flow_source_is_not_asked_on_the_fast_path() {
    // exact position (output frame 0): the fast path returns frame A and never calls the closure
    let s = external_cfg(1, 0);
    let got = run_external(&s, &Mock::speed(50.0), |_| panic!("the loader runs only on the vector path")).unwrap();
    let (a, _) = pair(Fmt::U8);
    let Rendered::Image(got) = got else { panic!("expected an image") };
    assert_eq!(got, a);
    // Vectors = No Vectors (quality 0) between frames is the plain frame blend, closure not called either
    let mut s = external_cfg(1, 100);
    s.vectors_quality = 0;
    let got = run_external(&s, &Mock::speed(50.0), |_| panic!("the loader runs only on the vector path")).unwrap();
    assert!(matches!(got, Rendered::Image(_)));
}

#[test]
fn no_flow_at_all_is_an_error_and_a_single_flow_is_inverted_where_it_is_read() {
    let (a, _) = pair(Fmt::U8);
    let ab = true_flows().f_ab;
    let ba = true_flows().f_ba;
    // neither layer: plugin 04 never calls the engine in this case, here a mismatch error
    let s = external_cfg(1, 100);
    match run_external(&s, &Mock::speed(50.0), |_| Ok(Flows { f_ab: None, f_ba: None })) {
        Err(FxError::Mismatch(m)) => assert_eq!(m, "Vectors In: neither vector layer supplied a flow"),
        other => panic!("{other:?}"),
    }
    // one layer: Blend and Weighted Blend read both flows, so the missing one is the inversion of
    // the other (`invert_flow`). For a uniform flow the inversion is the negated uniform flow, so
    // the result is the one of the two true flows up to the last bit of the splat: every sample
    // within one level
    for interp in [1, 2] {
        let s = external_cfg(interp, 100);
        let want = run_external(&s, &Mock::speed(50.0), |_| Ok(true_flows())).unwrap();
        let Rendered::Image(want) = want else { panic!() };
        for only in [Flows { f_ab: ab.clone(), f_ba: None }, Flows { f_ab: None, f_ba: ba.clone() }] {
            let got = run_external(&s, &Mock::speed(50.0), |_| Ok(only)).unwrap();
            let Rendered::Image(got) = got else { panic!() };
            for y in 0..H {
                for x in 0..W {
                    for c in 0..4 {
                        let (g, w) = (sample(&got, x, y, c), sample(&want, x, y, c));
                        assert!((g - w).abs() <= 1.0, "interp {interp} ({x}, {y}) channel {c}: {g} against {w}");
                    }
                }
            }
        }
    }
    // Nearest at wA = 0.5 reads only F_BA: that single flow is enough and the result equals the two flow render
    let s = external_cfg(0, 100);
    let got = run_external(&s, &Mock::speed(50.0), |_| Ok(Flows { f_ab: None, f_ba: ba.clone() })).unwrap();
    let want = run_external(&s, &Mock::speed(50.0), |_| Ok(true_flows())).unwrap();
    assert_eq!(got, want);
    let Rendered::Image(img) = got else { panic!() };
    same_inside(&img, &moved(&a, (4, 2)), MARGIN).unwrap();
    // with only F_AB it needs F_BA: the inversion runs and gives the same picture inside
    let got = run_external(&s, &Mock::speed(50.0), |_| Ok(Flows { f_ab: ab.clone(), f_ba: None })).unwrap();
    let Rendered::Image(img) = got else { panic!() };
    same_inside(&img, &moved(&a, (4, 2)), MARGIN).unwrap();
    // speed 75 percent: output frame 1 is at position 0.75, wA = 0.25 < 0.5, Nearest reads only F_AB
    let got = run_external(&s, &Mock::speed(75.0), |sel| {
        assert!(sel.wa < 0.5);
        Ok(Flows { f_ab: ab.clone(), f_ba: None })
    })
    .unwrap();
    let Rendered::Image(img) = got else { panic!() };
    same_inside(&img, &moved(&a, (6, 3)), MARGIN).unwrap();
    // the blur tail reads F_AB (Nearest too, once both flows exist in Vectors In): with the slider up and a moving
    // source the missing F_AB is derived from F_BA
    let mut s = external_cfg(0, 100);
    s.blur = 1.0;
    let got = run_external(&s, &Mock::speed(150.0), |_| Ok(Flows { f_ab: None, f_ba: ba.clone() })).unwrap();
    let want = run_external(&s, &Mock::speed(150.0), |_| Ok(true_flows())).unwrap();
    let (Rendered::Image(got), Rendered::Image(want)) = (got, want) else { panic!() };
    for y in 0..H {
        for x in 0..W {
            for c in 0..4 {
                assert!((sample(&got, x, y, c) - sample(&want, x, y, c)).abs() <= 1.0, "blur tail ({x}, {y}) channel {c}");
            }
        }
    }
}

#[test]
fn the_flow_inversion_negates_a_flow_at_the_place_it_points_to() {
    // a uniform flow (4, 2): every pixel lands 4 right and 2 down carrying -(4, 2); the pixels nothing lands on (the
    // first columns and rows) are holes that the fill closes with the same constant: the whole field is (-4, -2)
    let inv = invert_flow(&uniform(40, 30, (4.0, 2.0))).unwrap();
    assert_eq!((inv.width, inv.height), (40, 30));
    for y in 0..30 {
        for x in 0..40 {
            let v = inv.at(x, y);
            assert!((v[0] + 4.0).abs() < 1e-3 && (v[1] + 2.0).abs() < 1e-3, "({x}, {y}): {v:?}");
        }
    }
    // a zero flow is its own inverse
    let inv = invert_flow(&uniform(16, 12, (0.0, 0.0))).unwrap();
    assert!(inv.data.iter().all(|v| *v == 0.0));
    // a smooth flow: the inverse at the landing point of p is -f(p), within the error of the gradient
    let (w, h) = (64usize, 48usize);
    let mut f = FlowField::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 2;
            f.data[i] = 2.0 + 0.02 * x as f32;
            f.data[i + 1] = 1.0 - 0.01 * y as f32;
        }
    }
    let inv = invert_flow(&f).unwrap();
    for y in 12..h - 12 {
        for x in 12..w - 12 {
            let want = f.at(x, y);
            let v = inv.bilerp(x as f32 + want[0], y as f32 + want[1]);
            assert!((v[0] + want[0]).abs() < 0.12 && (v[1] + want[1]).abs() < 0.12, "({x}, {y}): {v:?} against {want:?}");
        }
    }
    // a flow below 2 x 2 is the forward warp's refusal, not a panic
    assert!(invert_flow(&uniform(1, 1, (1.0, 1.0))).is_err());
}

#[test]
fn bad_codes_are_refused_before_the_flow_source_is_asked() {
    for code in [4, -1] {
        let mut s = external_cfg(1, 100);
        s.warp_mode = code;
        let r = run_external(&s, &Mock::speed(50.0), |_| panic!("refused before any loading"));
        assert!(matches!(r, Err(FxError::BadParam { .. })));
    }
}

#[test]
fn the_forward_codes_work_with_injected_flows() {
    // plugin 04's default Warping is a forward code: injected flows through retime_with_flows give the forward composition
    let (a, _) = pair(Fmt::U8);
    let (_, b) = pair(Fmt::U8);
    for code in [2, 3] {
        let mut s = external_cfg(1, 100);
        s.warp_mode = code;
        let got = run_external(&s, &Mock::speed(50.0), |_| Ok(true_flows())).unwrap();
        let Rendered::Image(got) = got else { panic!() };
        let selection = Selection { s0: 0.5, s1: 0.5, a: 0, b: 1, wa: 0.5, exact: false, vectors_quality: 4, interp: 1 };
        assert_eq!(got, compose_vectors(&s, &selection, &a, &b, &true_flows()).unwrap(), "code {code}");
        same_inside(&got, &moved(&a, (4, 2)), MARGIN).unwrap();
    }
}

#[test]
fn a_vector_layer_image_becomes_a_flow_decoded_hole_filled_and_scaled() {
    use crate::vectors::encode_vectors_image;
    // a uniform flow (3, -2) encoded with Max Displace 8 at the three depths; a block of alpha 0 pixels is a hole
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let (w, h) = (40usize, 30usize);
        let flow = uniform(w, h, (3.0, -2.0));
        let mut img = encode_vectors_image(w, h, fmt, &flow, 8.0).unwrap();
        for y in 10..18 {
            for x in 14..24 {
                let i = (y * w + x) * 4;
                match &mut img.samples {
                    Samples::U8(v) => v[i..i + 4].fill(0),
                    Samples::U16(v) => v[i..i + 4].fill(0),
                    Samples::F32(v) => v[i..i + 4].fill(0.0),
                }
            }
        }
        // the decode alone gives zero vectors in the hole
        let (raw, _) = crate::vectors::decode_vectors_with_holes(&img, 8.0);
        assert_eq!(raw.at(15, 12), [0.0, 0.0], "{fmt:?}");
        // the loader result: the hole is filled with the surrounding vector, and Vec Scale is applied after
        let tol = if fmt == Fmt::U8 { 0.07 } else { 1e-3 };
        let plain = flow_from_vector_image(&img, 8.0, (1.0, 1.0)).unwrap();
        for y in 0..h {
            for x in 0..w {
                let v = plain.at(x, y);
                assert!((v[0] - 3.0).abs() < tol && (v[1] + 2.0).abs() < tol, "{fmt:?} ({x}, {y}): {v:?}");
            }
        }
        let scaled = flow_from_vector_image(&img, 8.0, (2.0, 0.5)).unwrap();
        for (p, q) in plain.data.chunks_exact(2).zip(scaled.data.chunks_exact(2)) {
            assert_eq!((q[0], q[1]), (2.0f32 * p[0], 0.5f32 * p[1]), "{fmt:?}");
        }
    }
}

#[test]
#[ignore = "the G6 wrap rule (U) this test encodes is superseded by the 360 rules read from TwixtorPro_64 (parity/lab/re/op-residual.md, finding 6)"]
fn the_wrap_flag_reaches_the_solves_and_tracks_a_shift_across_the_seam() {
    // `EngineSettings::wrap_360` is `FlowSettings::wrap` of the render's solves. A texture periodic in x moved
    // 2.5 px to the right: with the flag both flows of `solve_flows` are the shift in every column, the two seam
    // columns included; the plain solve loses it on the side whose targets leave the frame
    let (w, h) = (128usize, 48usize);
    let wf = w as f64;
    let tex = |x: f64, y: f64| {
        let t = std::f64::consts::TAU;
        0.5 + 0.16 * (t * (3.0 * x / wf + 0.03 * y)).sin() + 0.12 * (t * (7.0 * x / wf - 0.05 * y) + 1.0).sin()
    };
    let a = make(Fmt::F32, w, h, |x, y| [tex(x as f64, y as f64); 3]);
    let b = make(Fmt::F32, w, h, |x, y| [tex(x as f64 - 2.5, y as f64); 3]);
    let mut s = settings(w, h, Fmt::F32);
    s.vectors_quality = 5;
    s.sensitivity = 1.0;
    let sl = Selection { vectors_quality: 5, ..sel(1, 0.5) };
    s.wrap_360 = true;
    let fs = flow_settings(&s, &sl);
    assert!(fs.wrap, "the flag reaches the solver settings");
    assert!(!flow_settings(&EngineSettings { wrap_360: false, ..s.clone() }, &sl).wrap);
    let run = |threads: usize| rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| solve_flows(&s, &sl, &a, &b).unwrap());
    let flows = run(1);
    assert_eq!(run(4), flows, "1 and 4 threads");
    let pair = solve_pair(&a, &b, &fs).unwrap();
    assert_eq!(flows, Flows::pair(pair), "the wrap solves of the pair");
    // the worst column of each flow against the true motion (+2.5 for F_AB, -2.5 for F_BA), rows 8 to h - 8
    let worst = |f: &FlowField, want: f64, cols: std::ops::Range<usize>| {
        cols.map(|c| {
            let e: f64 = (8..h - 8).map(|y| {
                let v = f.at(c, y);
                ((f64::from(v[0]) - want).powi(2) + f64::from(v[1]).powi(2)).sqrt()
            })
            .sum();
            e / (h - 16) as f64
        })
        .fold(0.0f64, f64::max)
    };
    let (f_ab, f_ba) = (flows.f_ab.as_ref().unwrap(), flows.f_ba.as_ref().unwrap());
    let (ab, ba) = (worst(f_ab, 2.5, 0..w), worst(f_ba, -2.5, 0..w));
    assert!(ab < 0.05 && ba < 0.05, "wrap: worst column F_AB {ab}, F_BA {ba}");
    let (ab_seam, ba_seam) = (worst(f_ab, 2.5, w - 2..w), worst(f_ba, -2.5, 0..2));
    assert!(ab_seam < 0.05 && ba_seam < 0.05, "wrap: seam F_AB {ab_seam}, F_BA {ba_seam}");
    // the same render without the flag: F_AB's targets leave the frame on the right, F_BA's on the left
    let plain = solve_flows(&EngineSettings { wrap_360: false, ..s.clone() }, &sl, &a, &b).unwrap();
    let (pab, pba) = (worst(plain.f_ab.as_ref().unwrap(), 2.5, w - 2..w), worst(plain.f_ba.as_ref().unwrap(), -2.5, 0..2));
    eprintln!("seam columns: wrap {ab_seam:.4} {ba_seam:.4}, plain {pab:.4} {pba:.4}");
    assert!(pab > 4.0 * ab_seam.max(0.01) && pba > 4.0 * ba_seam.max(0.01), "plain seam F_AB {pab}, F_BA {pba}");
}
