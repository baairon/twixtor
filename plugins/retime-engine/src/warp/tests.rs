//! Tests of the inverse warp and the blur compensation.
//!
//! Two kinds of evidence: (1) goldens worked out by hand from the formulas, (2) bit exact
//! comparison with an independent numpy float32 model (`tests/fixtures/gen_warp.py` writes
//! `warp_cases.hex`), plus invariance and property tests.

use super::*;
use crate::image::{Image, Samples};

mod blur;

// ---------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------

/// Four channel f32 image from a closure `(x, y, c) -> value`.
fn img_f32(w: usize, h: usize, mut f: impl FnMut(usize, usize, usize) -> f32) -> Image {
    let mut v = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            for c in 0..4 {
                v.push(f(x, y, c));
            }
        }
    }
    Image { width: w, height: h, samples: Samples::F32(v) }
}

fn img_u8(w: usize, h: usize, mut f: impl FnMut(usize, usize, usize) -> u8) -> Image {
    let mut v = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            for c in 0..4 {
                v.push(f(x, y, c));
            }
        }
    }
    Image { width: w, height: h, samples: Samples::U8(v) }
}

fn img_u16(w: usize, h: usize, mut f: impl FnMut(usize, usize, usize) -> u16) -> Image {
    let mut v = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            for c in 0..4 {
                v.push(f(x, y, c));
            }
        }
    }
    Image { width: w, height: h, samples: Samples::U16(v) }
}

fn f32s(img: &Image) -> &[f32] {
    match &img.samples {
        Samples::F32(v) => v,
        _ => panic!("not a float image"),
    }
}

fn u8s(img: &Image) -> &[u8] {
    match &img.samples {
        Samples::U8(v) => v,
        _ => panic!("not a u8 image"),
    }
}

fn u16s(img: &Image) -> &[u16] {
    match &img.samples {
        Samples::U16(v) => v,
        _ => panic!("not a u16 image"),
    }
}

fn const_flow(w: usize, h: usize, dx: f32, dy: f32) -> FlowField {
    let mut f = FlowField::new(w, h);
    for y in 0..h {
        for x in 0..w {
            f.set(x, y, [dx, dy]);
        }
    }
    f
}

fn px_f32(img: &Image, x: usize, y: usize) -> [f32; 4] {
    let v = f32s(img);
    let i = (y * img.width + x) * 4;
    [v[i], v[i + 1], v[i + 2], v[i + 3]]
}

fn zeros_like(img: &Image) -> Image {
    Image::new(img.width, img.height, img.fmt())
}

const HOLD: WarpOpts = WarpOpts { hold: true, is360: false, mode: 1 };
const FADE: WarpOpts = WarpOpts { hold: false, is360: false, mode: 1 };
const WRAP: WarpOpts = WarpOpts { hold: false, is360: true, mode: 1 };

// ---------------------------------------------------------------------------------------
// store rules
// ---------------------------------------------------------------------------------------

#[test]
fn sat_u8_matches_the_documented_clamp_over_a_dense_grid() {
    // clamp(trunc(v + 0.5), 0, 255) for every value that is not NaN or beyond the i32 range
    let mut v = -600.0f32;
    while v < 600.0 {
        let want = (v + 0.5f32).trunc().clamp(0.0, 255.0) as u8;
        assert_eq!(sat_u8(v), want, "v = {v}");
        v += 0.0625;
    }
    for bits in 0..2000u32 {
        // values around the rounding boundaries: x.5 minus and plus an ulp
        let base = (bits % 300) as f32 - 20.0 + 0.5;
        for v in [f32::from_bits(base.to_bits() - 1), base, f32::from_bits(base.to_bits() + 1)] {
            let want = (v + 0.5f32).trunc().clamp(0.0, 255.0) as u8;
            assert_eq!(sat_u8(v), want, "v = {v}");
        }
    }
}

#[test]
fn sat_u16_matches_the_documented_clamp_over_a_dense_grid() {
    let mut v = -300.0f32;
    while v < 70000.0 {
        let want = (v + 0.5f32).trunc().clamp(0.0, 65535.0) as u16;
        assert_eq!(sat_u16(v), want, "v = {v}");
        v += 7.0625;
    }
    for v in [65534.5f32, 65535.0, 65535.4, 65535.5, 65536.0, 1e9, -1e9, 0.49999997, 0.5, -0.5, -0.49999997] {
        let want = (v + 0.5f32).trunc().clamp(0.0, 65535.0) as u16;
        assert_eq!(sat_u16(v), want, "v = {v}");
    }
}

#[test]
fn integer_stores_of_overflow_and_nonfinite_values() {
    // large positive values saturate at the top (packssdw / packsswb saturation)
    assert_eq!(sat_u8(1e9), 255);
    assert_eq!(sat_u8(255.4), 255);
    assert_eq!(sat_u8(255.5), 255);
    assert_eq!(sat_u8(256.0), 255);
    assert_eq!(sat_u16(1e9), 65535);
    assert_eq!(sat_u16(65535.5), 65535);
    // large negative values saturate at zero
    assert_eq!(sat_u8(-1e9), 0);
    assert_eq!(sat_u8(-0.6), 0);
    assert_eq!(sat_u16(-1e9), 0);
    // the indefinite integer of the truncation (NaN, +-inf beyond 2^31) wraps through the
    // -128 / -32768 bias and lands on the maximum: a quirk of the saturation sequence
    assert_eq!(sat_u8(f32::NAN), 255);
    assert_eq!(sat_u8(f32::INFINITY), 255);
    assert_eq!(sat_u8(f32::NEG_INFINITY), 255);
    assert_eq!(sat_u16(f32::NAN), 65535);
    assert_eq!(sat_u16(f32::NEG_INFINITY), 65535);
    // exact halves round up, the rounding is trunc(v + 0.5)
    assert_eq!(sat_u8(0.5), 1);
    // the add is single precision: 0.49999997 + 0.5 rounds to 1.0, so the sample stores 1,
    // while one ulp lower stays below 1 and truncates to 0
    assert_eq!(sat_u8(0.49999997), 1);
    assert_eq!(sat_u8(0.49999994), 0);
    assert_eq!(sat_u8(127.5), 128);
    assert_eq!(sat_u16(32767.5), 32768);
}

#[test]
fn faded_scale_adds_the_half_in_double_precision() {
    // trunc((double)(b * f) + 0.5). With b * f = 0.49999997 (f32) the double sum
    // 0.99999997 truncates to 0, whereas a single precision sum would round to 1.0 and give 1.
    let f = 0.49999997f32;
    assert_eq!(0.49999997f32 + 0.5f32, 1.0f32);
    assert_eq!(scale_u8(1, f), 0);
    assert_eq!(scale_u16(1, f), 0);
    // ordinary values
    assert_eq!(scale_u8(200, 0.5), 100);
    assert_eq!(scale_u8(255, 1.0), 255);
    assert_eq!(scale_u8(255, 0.0), 0);
    assert_eq!(scale_u8(101, 0.5), 51); // 50.5 + 0.5 = 51
    assert_eq!(scale_u16(65535, 0.5), 32768); // 32767.5 + 0.5
    assert_eq!(scale_u16(1000, 0.3333), 333);
}

// ---------------------------------------------------------------------------------------
// samplers
// ---------------------------------------------------------------------------------------

#[test]
fn bilerp_goldens_with_known_fractions() {
    // 3x3 image, value = 100 * c + 10 * y + x, so every neighbour difference is known
    let img = img_f32(3, 3, |x, y, c| (100 * c + 10 * y + x) as f32);
    let s = f32s(&img);
    let at = |x: f32, y: f32| bilerp4::<f32>(s, 3, 2, 2, x, y);
    // exact grid points
    assert_eq!(at(1.0, 1.0), [11.0, 111.0, 211.0, 311.0]);
    // fx = 0.5, fy = 0: (p10 - p00) * 0.5 + p00 = 0.5 + p00
    assert_eq!(at(0.5, 0.0), [0.5, 100.5, 200.5, 300.5]);
    // fx = 0, fy = 0.25: 10 * 0.25 = 2.5 above p00
    assert_eq!(at(1.0, 0.25), [3.5, 103.5, 203.5, 303.5]);
    // both: 0.5 + 2.5
    assert_eq!(at(0.5, 0.25), [3.0, 103.0, 203.0, 303.0]);
    // x = 2.0 = w1: the column index becomes w1 - 1 = 1 with weight 1.0 (so p10 is read as column 2)
    assert_eq!(at(2.0, 0.0), [2.0, 102.0, 202.0, 302.0]);
    assert_eq!(at(2.0, 2.0), [22.0, 122.0, 222.0, 322.0]);
    // x = 2.75 (beyond w1, clamped): the weight is 1.0 again, not 0.75
    assert_eq!(at(2.75, 1.0), [12.0, 112.0, 212.0, 312.0]);
}

#[test]
fn bilerp_upper_clamp_uses_the_formula_not_the_neighbour() {
    // (b - a) * 1 + a is not b for floats in general: a = 1e8, b = 1 gives 0 in single precision.
    let img = img_f32(2, 2, |x, _y, _c| if x == 0 { 1.0e8 } else { 1.0 });
    let s = f32s(&img);
    let v = bilerp4::<f32>(s, 2, 1, 1, 1.0, 0.0);
    assert_eq!(v[0], 0.0, "(1 - 1e8) * 1 + 1e8 == 0 in f32");
    // integer images are exact
    let img8 = img_u8(2, 2, |x, _y, _c| if x == 0 { 200 } else { 7 });
    let v8 = bilerp4::<u8>(u8s(&img8), 2, 1, 1, 1.0, 0.0);
    assert_eq!(v8, [7.0; 4]);
}

#[test]
fn flow_sampler_resamples_the_flow_bilinearly() {
    // a 3x2 flow with dx = x, dy = 10 * y
    let mut f = FlowField::new(3, 2);
    for y in 0..2 {
        for x in 0..3 {
            f.set(x, y, [x as f32, 10.0 * y as f32]);
        }
    }
    let fl = Flow::new(&f);
    assert_eq!(fl.bilerp(0.5, 0.5), [0.5, 5.0]);
    assert_eq!(fl.bilerp(1.25, 0.0), [1.25, 0.0]);
    // beyond the last column the weight is 1.0 on the clamped neighbour
    assert_eq!(fl.bilerp(2.0, 1.0), [2.0, 10.0]);
    assert_eq!(fl.bilerp(2.5, 1.0), [2.0, 10.0]);
}

// ---------------------------------------------------------------------------------------
// clip_to_rect
// ---------------------------------------------------------------------------------------

#[test]
fn clip_to_rect_goldens() {
    // inside: untouched
    let (mut ix, mut iy) = (3.0f32, 2.0f32);
    assert!(clip_to_rect(1.0, 1.0, &mut ix, &mut iy, 10.0, 5.0));
    assert_eq!((ix, iy), (3.0, 2.0));
    // from pixel (2, 2) to (-2, 2): hits x = 0 after half the way, iy stays (2 - 0.5 * 0)
    let (mut ix, mut iy) = (-2.0f32, 2.0f32);
    clip_to_rect(2.0, 2.0, &mut ix, &mut iy, 10.0, 5.0);
    assert_eq!((ix, iy), (0.0, 2.0));
    // from (2, 4) to (-2, 0): the segment crosses x = 0 at y = 2
    let (mut ix, mut iy) = (-2.0f32, 0.0f32);
    clip_to_rect(2.0, 4.0, &mut ix, &mut iy, 10.0, 5.0);
    assert_eq!((ix, iy), (0.0, 2.0));
    // from (8, 1) to (14, 4) in a 10 x 5 rectangle (w1 = 10, h1 = 5): x = 10 is reached at t = 1/3, y = 2
    let (mut ix, mut iy) = (14.0f32, 4.0f32);
    clip_to_rect(8.0, 1.0, &mut ix, &mut iy, 10.0, 5.0);
    assert_eq!(ix, 10.0);
    assert!((iy - 1.0 - 3.0 * (2.0 / 6.0)).abs() < 1e-5, "iy = {iy}");
    // a corner: both coordinates leave; ends on the rectangle
    let (mut ix, mut iy) = (-4.0f32, -4.0f32);
    clip_to_rect(1.0, 1.0, &mut ix, &mut iy, 10.0, 5.0);
    assert_eq!((ix, iy), (0.0, 0.0));
    // a pixel on the border whose sample goes outward: the segment leaves at once, the other
    // coordinate collapses to the pixel's own value (t = 0)
    let (mut ix, mut iy) = (12.5f32, 3.0f32);
    clip_to_rect(10.0, 1.0, &mut ix, &mut iy, 10.0, 5.0);
    assert_eq!((ix, iy), (10.0, 1.0));
    // NaN falls through untouched
    let (mut ix, mut iy) = (f32::NAN, 2.0f32);
    assert!(clip_to_rect(1.0, 1.0, &mut ix, &mut iy, 10.0, 5.0));
    assert!(ix.is_nan());
}

// ---------------------------------------------------------------------------------------
// the warp: early outs, shifts, hold, fade, coverage
// ---------------------------------------------------------------------------------------

#[test]
fn integer_flow_is_an_exact_shift_in_all_depths() {
    // flow (3, -2) at amount 1: out(x, y) = src(x + 3, y - 2) wherever the source point is inside
    let (w, h) = (9, 7);
    let flow = const_flow(w, h, 3.0, -2.0);
    let f = img_f32(w, h, |x, y, c| (c * 1000 + y * 50 + x) as f32 * 0.25);
    let u = img_u8(w, h, |x, y, c| (c * 61 + y * 17 + x * 5) as u8);
    let s = img_u16(w, h, |x, y, c| (c * 15000 + y * 1700 + x * 55) as u16);
    for (name, src) in [("f32", &f), ("u8", &u), ("u16", &s)] {
        let mut out = zeros_like(src);
        let cov = inv_warp_coverage(&mut out, src, &flow, 1.0, FADE).unwrap();
        for y in 0..h {
            for x in 0..w {
                let (sx, sy) = (x as i32 + 3, y as i32 - 2);
                let inside = sx >= 0 && sx < w as i32 && sy >= 0 && sy < h as i32;
                if inside {
                    assert_eq!(cov.at(x, y), 255, "{name} ({x},{y})");
                    for c in 0..4 {
                        let i = (y * w + x) * 4 + c;
                        let j = (sy as usize * w + sx as usize) * 4 + c;
                        match (&out.samples, &src.samples) {
                            (Samples::F32(o), Samples::F32(s)) => assert_eq!(o[i].to_bits(), s[j].to_bits(), "{name} ({x},{y},{c})"),
                            (Samples::U8(o), Samples::U8(s)) => assert_eq!(o[i], s[j], "{name} ({x},{y},{c})"),
                            (Samples::U16(o), Samples::U16(s)) => assert_eq!(o[i], s[j], "{name} ({x},{y},{c})"),
                            _ => unreachable!(),
                        }
                    }
                } else {
                    // sample point at least one pixel outside (shift by whole pixels): transparent, coverage 0
                    // (a point exactly on the next pixel row/column beyond the border is -1 or W, which is "outside")
                    assert_eq!(cov.at(x, y), 0, "{name} ({x},{y})");
                    for c in 0..4 {
                        let i = (y * w + x) * 4 + c;
                        match &out.samples {
                            Samples::F32(o) => assert_eq!(o[i], 0.0),
                            Samples::U8(o) => assert_eq!(o[i], 0),
                            Samples::U16(o) => assert_eq!(o[i], 0),
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn half_pixel_flow_goldens_in_hold_mode() {
    // value = 100 * c + 10 * y + x (exactly representable), flow (0.5, 0.25) at amount 1
    let (w, h) = (4, 3);
    let src = img_f32(w, h, |x, y, c| (100 * c + 10 * y + x) as f32);
    let flow = const_flow(w, h, 0.5, 0.25);
    let mut out = zeros_like(&src);
    inv_warp(&mut out, &src, &flow, 1.0, HOLD).unwrap();
    // interior pixel (1, 0): ix = 1.5, iy = 0.25: p00 = 1, +0.5 in x, +2.5 in y => 4
    assert_eq!(px_f32(&out, 1, 0), [4.0, 104.0, 204.0, 304.0]);
    // pixel (0, 0): 3.0 above p00 = 0
    assert_eq!(px_f32(&out, 0, 0), [3.0, 103.0, 203.0, 303.0]);
    // last column x = 3 = w1: the sample ix = 3.5 > w1 is clipped along the segment to x = 3; the
    // segment (3, y) to (3.5, y + 0.25) meets x = 3 at its start, so iy collapses to y:
    // t = (x - w1) / (x - ix) = 0 / (-0.5) = -0.0, iy = y - (-0.0) * (y - iy) = y
    assert_eq!(px_f32(&out, 3, 1), [13.0, 113.0, 213.0, 313.0]);
    // last row y = 2 = h1: iy = 2.25 > h1 clipped to 2 with ix collapsing to x
    assert_eq!(px_f32(&out, 1, 2), [21.0, 121.0, 221.0, 321.0]);
    // bottom right corner stays at the corner
    assert_eq!(px_f32(&out, 3, 2), [23.0, 123.0, 223.0, 323.0]);
}

#[test]
fn hold_clips_far_samples_to_the_border_along_the_segment() {
    // flow (+100, 0) at amount 1: every pixel is pushed far to the right and clipped to column w1
    let (w, h) = (6, 4);
    let src = img_f32(w, h, |x, y, _| (10 * y + x) as f32);
    let flow = const_flow(w, h, 100.0, 0.0);
    let mut out = zeros_like(&src);
    inv_warp(&mut out, &src, &flow, 1.0, HOLD).unwrap();
    for y in 0..h {
        for x in 0..w {
            assert_eq!(px_f32(&out, x, y)[0], (10 * y + w - 1) as f32, "({x},{y})");
        }
    }
    // the same with a diagonal push: (x, y) + (-100, -100) lands in the top left corner region; the
    // clip slides the point along the segment to the rectangle, so the result is on the border and
    // lies between the corner pixel (0, 0) and the pixel itself
    let flow = const_flow(w, h, -100.0, -100.0);
    inv_warp(&mut out, &src, &flow, 1.0, HOLD).unwrap();
    assert_eq!(px_f32(&out, 0, 0)[0], 0.0);
    for y in 0..h {
        for x in 0..w {
            let v = px_f32(&out, x, y)[0];
            let (xi, yi) = (x as f32, y as f32);
            // the clipped point is (x - t * 100, y - t * 100) with t chosen to reach a border: the
            // nearer one, so the shift is min(x, y) in both axes
            let m = xi.min(yi);
            let want = 10.0 * (yi - m) + (xi - m);
            assert!((v - want).abs() < 1e-3, "({x},{y}) got {v}, want {want}");
        }
    }
}

#[test]
fn fade_goldens_and_coverage_bytes() {
    // 4x3 image all channels 200 (u8); flow dx = -0.5 for the left column, so the sample is at
    // x = -0.5 (inside the fade band): f = ix + 1 = 0.5; coverage = trunc(0.5 * 255 + 0.5) = 128
    let (w, h) = (4, 3);
    let src = img_u8(w, h, |_, _, _| 200);
    let mut flow = const_flow(w, h, 0.0, 0.0);
    flow.set(0, 1, [-0.5, 0.0]);
    flow.set(3, 1, [1.25, 0.0]); // x = 4.25 >= W = 4: outside, transparent
    flow.set(2, 1, [1.5, 0.0]); // x = 3.5 > w1 = 3: f = W - ix = 0.5
    flow.set(1, 0, [0.0, -0.75]); // y = -0.75: f = iy + 1 = 0.25, coverage = trunc(0.25 * 255 + 0.5) = 64
    let mut out = zeros_like(&src);
    let cov = inv_warp_coverage(&mut out, &src, &flow, 1.0, FADE).unwrap();
    let p = |x: usize, y: usize| -> Vec<u8> { u8s(&out)[(y * w + x) * 4..(y * w + x) * 4 + 4].to_vec() };
    // faded sample: the clip moves (-0.5, 1) to (0, 1), the sample is 200, stored 200 then scaled by 0.5:
    // trunc(100.0 + 0.5) = 100
    assert_eq!(p(0, 1), vec![100; 4]);
    assert_eq!(cov.at(0, 1), 128);
    assert_eq!(p(2, 1), vec![100; 4]);
    assert_eq!(cov.at(2, 1), 128);
    assert_eq!(p(3, 1), vec![0; 4]);
    assert_eq!(cov.at(3, 1), 0);
    assert_eq!(p(1, 0), vec![50; 4]);
    assert_eq!(cov.at(1, 0), 64);
    // untouched pixels are plain copies with coverage 255
    assert_eq!(p(1, 1), vec![200; 4]);
    assert_eq!(cov.at(1, 1), 255);
    assert_eq!(cov.at(0, 0), 255);
}

#[test]
fn fade_weight_multiplies_both_axes_and_coverage_saturates() {
    // a corner sample at (-0.5, -0.5): f = 0.5 * 0.5 = 0.25
    let src = img_f32(3, 3, |_, _, _| 1.0);
    let mut flow = const_flow(3, 3, 0.0, 0.0);
    flow.set(0, 0, [-0.5, -0.5]);
    let mut out = zeros_like(&src);
    let cov = inv_warp_coverage(&mut out, &src, &flow, 1.0, FADE).unwrap();
    assert_eq!(px_f32(&out, 0, 0), [0.25; 4]);
    assert_eq!(cov.at(0, 0), 64);
    // a sample just inside the band edge: ix = -0.0001 gives f = 0.9999 and c = 255.4, which is above
    // 255 only when f * 255 + 0.5 > 255; here 254.97 + 0.5 = 255.47 > 255 => 255
    flow.set(1, 1, [-1.0001, 0.0]); // x = 1 - 1.0001 = -0.0001
    inv_warp_coverage(&mut out, &src, &flow, 1.0, FADE).map(|c| assert_eq!(c.at(1, 1), 255)).unwrap();
    // exactly one pixel outside is transparent: ix = -1.0 is rejected (<= -1)
    flow.set(1, 1, [-2.0, 0.0]);
    let cov = inv_warp_coverage(&mut out, &src, &flow, 1.0, FADE).unwrap();
    assert_eq!(cov.at(1, 1), 0);
    assert_eq!(px_f32(&out, 1, 1), [0.0; 4]);
    // ix = W = 3 exactly is rejected too (>= W), ix just below W (but above w1 = 2) is faded
    flow.set(1, 1, [2.0, 0.0]); // x = 3.0
    let cov = inv_warp_coverage(&mut out, &src, &flow, 1.0, FADE).unwrap();
    assert_eq!(cov.at(1, 1), 0);
    flow.set(1, 1, [1.9990234, 0.0]); // x = 2.9990234: f = 3 - 2.9990234 = 0.0009766 -> c = 0.249 + 0.5 = 0.749 -> 0
    let cov = inv_warp_coverage(&mut out, &src, &flow, 1.0, FADE).unwrap();
    assert_eq!(cov.at(1, 1), 0);
    let f = 3.0f32 - (1.0f32 + 1.9990234f32);
    assert_eq!(px_f32(&out, 1, 1)[0], 1.0f32 * f);
}

#[test]
fn coverage_invert_and_combine() {
    let mut a = Coverage { width: 3, height: 1, data: vec![0, 100, 255] };
    let b = Coverage { width: 3, height: 1, data: vec![7, 50, 255] };
    let mut m = a.clone();
    m.combine_max(&b);
    assert_eq!(m.data, vec![7, 100, 255]);
    a.invert();
    assert_eq!(a.data, vec![255, 155, 0]);
    // the holes of the first Smart Blend warp: where the fade left a gap, 255 - coverage is what the hole fill gets
    let mut m2 = m.clone();
    m2.invert();
    assert_eq!(m2.data, vec![248, 155, 0]);
}

#[test]
fn hold_and_360_leave_the_coverage_at_255() {
    let src = img_f32(5, 4, |x, y, c| (x + y + c) as f32);
    let flow = const_flow(5, 4, 9.0, -9.0);
    let mut out = zeros_like(&src);
    for opts in [HOLD, WRAP] {
        let cov = inv_warp_coverage(&mut out, &src, &flow, 1.0, opts).unwrap();
        assert!(cov.data.iter().all(|&c| c == 255));
        assert_eq!((cov.width, cov.height), (5, 4));
    }
}

#[test]
fn small_amounts_copy_the_source_and_fill_the_coverage() {
    let src = img_u8(6, 5, |x, y, c| (x * 31 + y * 7 + c) as u8);
    let flow = const_flow(6, 5, 4.0, 4.0);
    for amount in [0.0f32, 0.004, -0.0049, 0.00499] {
        let mut out = zeros_like(&src);
        let cov = inv_warp_coverage(&mut out, &src, &flow, amount, FADE).unwrap();
        assert_eq!(out, src, "amount {amount}");
        assert!(cov.data.iter().all(|&c| c == 255), "amount {amount}");
    }
    // f32(0.005) is below the double 0.005 (it is 0.00499999988...), so it still takes the copy path
    let mut out = zeros_like(&src);
    inv_warp(&mut out, &src, &flow, 0.005f32, HOLD).unwrap();
    assert_eq!(out, src);
    // just above the threshold the flow acts (shifts of 0.00501 * 4 = 0.02 pixel change the 8 bit values)
    let mut out = zeros_like(&src);
    inv_warp(&mut out, &src, &flow, 0.0051, HOLD).unwrap();
    assert_ne!(out, src);
}

#[test]
fn flow_ratio_test_uses_the_flow_to_image_scale() {
    // image 11 wide (w1 = 10), flow 601 wide (w1 = 600): r = 60. amount 0.0059 passes the first test
    // (>= 0.005) but 0.0059 / 60 = 0.0000983 < 0.0001 copies; 0.0061 / 60 = 0.0001017 samples.
    let (w, h) = (11, 9);
    let src = img_f32(w, h, |x, y, c| (x * 3 + y * 5 + c) as f32 * 0.37);
    let mut flow = FlowField::new(601, 9);
    for y in 0..9 {
        for x in 0..601 {
            flow.set(x, y, [3.0, 2.0]);
        }
    }
    let mut out = zeros_like(&src);
    inv_warp(&mut out, &src, &flow, 0.0059, FADE).unwrap();
    assert_eq!(out, src);
    inv_warp(&mut out, &src, &flow, 0.0061, FADE).unwrap();
    assert_ne!(out, src);
}

#[test]
fn nan_flow_gives_transparent_pixels_in_fade_mode_and_an_error_in_hold_mode() {
    let src = img_f32(4, 4, |_, _, _| 1.0);
    let mut flow = const_flow(4, 4, 0.0, 0.0);
    flow.set(2, 2, [f32::NAN, 0.0]);
    let mut out = zeros_like(&src);
    let cov = inv_warp_coverage(&mut out, &src, &flow, 1.0, FADE).unwrap();
    assert_eq!(px_f32(&out, 2, 2), [0.0; 4]);
    assert_eq!(cov.at(2, 2), 0);
    // neighbouring pixels sample the flow bilinearly, so the NaN spreads to them as well (documented)
    let e = inv_warp(&mut out, &src, &flow, 1.0, HOLD).unwrap_err();
    assert!(matches!(e, FxError::Unsupported(_)), "{e}");
}

#[test]
fn argument_errors() {
    let a = img_f32(4, 4, |_, _, _| 0.0);
    let flow = const_flow(4, 4, 1.0, 1.0);
    let mut out = img_f32(5, 4, |_, _, _| 0.0);
    assert!(matches!(inv_warp(&mut out, &a, &flow, 1.0, HOLD), Err(FxError::Mismatch(_))));
    let mut out = img_u8(4, 4, |_, _, _| 0);
    assert!(matches!(inv_warp(&mut out, &a, &flow, 1.0, HOLD), Err(FxError::Mismatch(_))));
    let mut out = img_f32(4, 4, |_, _, _| 0.0);
    let tiny = const_flow(1, 4, 1.0, 1.0);
    assert!(matches!(inv_warp(&mut out, &a, &tiny, 1.0, HOLD), Err(FxError::Mismatch(_))));
    let one = img_f32(1, 4, |_, _, _| 0.0);
    let mut one_out = one.clone();
    assert!(matches!(inv_warp(&mut one_out, &one, &flow, 1.0, HOLD), Err(FxError::Mismatch(_))));
    // mip map sampling is refused, but only after the early outs (the copy needs no pyramid)
    let mode2 = WarpOpts { mode: 2, ..HOLD };
    assert!(matches!(inv_warp(&mut out, &a, &flow, 1.0, mode2), Err(FxError::Unsupported(_))));
    assert!(inv_warp(&mut out, &a, &flow, 0.001, mode2).is_ok());
}

// ---------------------------------------------------------------------------------------
// the independent numpy model
// ---------------------------------------------------------------------------------------

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn f32(&mut self) -> f32 {
        f32::from_bits(self.u32())
    }
    fn bytes(&mut self, n: usize) -> &'a [u8] {
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        s
    }
}

fn samples_from(depth: u32, b: &[u8], w: usize, h: usize) -> Image {
    let n = w * h * 4;
    let samples = match depth {
        0 => Samples::U8(b.to_vec()),
        1 => Samples::U16(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()),
        _ => Samples::F32(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
    };
    let img = Image { width: w, height: h, samples };
    assert_eq!(img.sample_count(), n);
    img
}

fn depth_bytes(depth: u32) -> usize {
    [1, 2, 4][depth as usize]
}

struct Case {
    src: Image,
    flow: FlowField,
    amount: f32,
    opts: WarpOpts,
    want_cov: bool,
    out: Image,
    cov: Option<Vec<u8>>,
}

fn load_cases() -> Vec<Case> {
    let blob = crate::fixture::hex(include_str!("../../tests/fixtures/warp_cases.hex"));
    assert_eq!(&blob[..4], b"WRP1");
    let mut r = Reader { b: &blob, p: 4 };
    let n = r.u32();
    let mut cases = Vec::new();
    for _ in 0..n {
        let depth = r.u32();
        let (w, h, fw, fh) = (r.u32() as usize, r.u32() as usize, r.u32() as usize, r.u32() as usize);
        let amount = r.f32();
        let flags = r.u32();
        let nb = w * h * 4 * depth_bytes(depth);
        let src = samples_from(depth, r.bytes(nb), w, h);
        let fb = r.bytes(fw * fh * 2 * 4);
        let flow = FlowField {
            width: fw,
            height: fh,
            data: fb.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        };
        let out = samples_from(depth, r.bytes(nb), w, h);
        let want_cov = flags & 4 != 0;
        let cov = if want_cov { Some(r.bytes(w * h).to_vec()) } else { None };
        cases.push(Case {
            src,
            flow,
            amount,
            opts: WarpOpts { hold: flags & 1 != 0, is360: flags & 2 != 0, mode: 1 },
            want_cov,
            out,
            cov,
        });
    }
    assert_eq!(r.p, blob.len(), "fixture fully consumed");
    cases
}

fn same_samples(a: &Image, b: &Image) -> Option<usize> {
    match (&a.samples, &b.samples) {
        (Samples::U8(x), Samples::U8(y)) => x.iter().zip(y).position(|(p, q)| p != q),
        (Samples::U16(x), Samples::U16(y)) => x.iter().zip(y).position(|(p, q)| p != q),
        (Samples::F32(x), Samples::F32(y)) => x.iter().zip(y).position(|(p, q)| p.to_bits() != q.to_bits()),
        _ => Some(0),
    }
}

#[test]
fn numpy_model_fixtures_are_reproduced_bit_for_bit() {
    let cases = load_cases();
    assert_eq!(cases.len(), 60);
    let mut with_cov = 0;
    let mut with_360 = 0;
    for (i, c) in cases.iter().enumerate() {
        let mut out = zeros_like(&c.src);
        let cov = if c.want_cov {
            with_cov += 1;
            Some(inv_warp_coverage(&mut out, &c.src, &c.flow, c.amount, c.opts).unwrap())
        } else {
            inv_warp(&mut out, &c.src, &c.flow, c.amount, c.opts).unwrap();
            None
        };
        if c.opts.is360 {
            with_360 += 1;
        }
        if let Some(pos) = same_samples(&out, &c.out) {
            panic!(
                "case {i} ({:?}, {}x{}, flow {}x{}, amount {}, {:?}): first difference at sample {pos}",
                c.src.fmt(),
                c.src.width,
                c.src.height,
                c.flow.width,
                c.flow.height,
                c.amount,
                c.opts
            );
        }
        if let (Some(cv), Some(want)) = (cov, &c.cov) {
            assert_eq!(&cv.data, want, "coverage of case {i}");
        }
    }
    assert!(with_cov >= 15 && with_360 >= 6, "fixture mix: {with_cov} coverage cases, {with_360} 360 cases");
}

// ---------------------------------------------------------------------------------------
// thread count invariance
// ---------------------------------------------------------------------------------------

fn in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(f)
}

#[test]
fn thread_count_does_not_change_the_output() {
    let cases = load_cases();
    for (i, c) in cases.iter().enumerate() {
        let run = |threads: usize| {
            in_pool(threads, || {
                let mut out = zeros_like(&c.src);
                let cov = inv_warp_coverage(&mut out, &c.src, &c.flow, c.amount, c.opts).unwrap();
                (out, cov)
            })
        };
        let (o1, c1) = run(1);
        let (o4, c4) = run(4);
        assert!(same_samples(&o1, &o4).is_none(), "case {i}");
        assert_eq!(c1, c4, "case {i}");
    }
}

// ---------------------------------------------------------------------------------------
// properties
// ---------------------------------------------------------------------------------------

#[test]
fn zero_flow_is_the_identity_for_integers() {
    let (w, h) = (13, 9);
    let u = img_u8(w, h, |x, y, c| (x * 19 + y * 7 + c * 3) as u8);
    let s = img_u16(w, h, |x, y, c| (x * 4999 + y * 711 + c * 33) as u16);
    for src in [&u, &s] {
        let flow = FlowField::new(w, h);
        for opts in [HOLD, FADE, WRAP] {
            let mut out = zeros_like(src);
            inv_warp(&mut out, src, &flow, 1.0, opts).unwrap();
            assert_eq!(&out, src, "{opts:?}");
        }
        // the flow may have another resolution
        let flow = FlowField::new(5, 4);
        let mut out = zeros_like(src);
        inv_warp(&mut out, src, &flow, 0.7, HOLD).unwrap();
        assert_eq!(&out, src);
    }
}

#[test]
fn zero_flow_keeps_float_images_except_where_the_clamped_formula_rounds() {
    let (w, h) = (7, 6);
    let src = img_f32(w, h, |x, y, c| 0.0371 * (x * 13 + y * 5 + c) as f32 - 0.2);
    let flow = FlowField::new(w, h);
    let mut out = zeros_like(&src);
    inv_warp(&mut out, &src, &flow, 1.0, HOLD).unwrap();
    for y in 0..h - 1 {
        for x in 0..w - 1 {
            assert_eq!(px_f32(&out, x, y), px_f32(&src, x, y), "({x},{y})");
        }
    }
    // the last row and column go through (b - a) * 1 + a and may differ in the last bit only
    for y in 0..h {
        for x in 0..w {
            let (o, s) = (px_f32(&out, x, y), px_f32(&src, x, y));
            for c in 0..4 {
                assert!((o[c] - s[c]).abs() <= s[c].abs() * 1.2e-7 + 1e-7);
            }
        }
    }
}


// ---------------------------------------------------------------------------------------
// cross depth consistency and pure axis flows
// ---------------------------------------------------------------------------------------

/// Deterministic pseudo random values for the property tests below.
fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64) / ((1u64 << 31) as f64)
}

#[test]
fn integer_depths_equal_the_float_warp_followed_by_the_store_rule() {
    // the sampling arithmetic is the same in all depths (integers convert exactly to float), so in hold
    // mode and in 360 mode the integer output is the store rule applied to the float output of the same data
    let mut seed = 0x5eed_u64;
    let (w, h, fw, fh) = (21, 15, 11, 8);
    let vals8: Vec<u8> = (0..w * h * 4).map(|_| (lcg(&mut seed) * 256.0) as u8).collect();
    let vals16: Vec<u16> = (0..w * h * 4).map(|_| (lcg(&mut seed) * 65536.0) as u16).collect();
    let mut flow = FlowField::new(fw, fh);
    for v in flow.data.iter_mut() {
        *v = ((lcg(&mut seed) - 0.5) * 60.0) as f32;
    }
    for opts in [HOLD, WRAP] {
        for amount in [1.0f32, 0.43, -0.8] {
            let s8 = Image { width: w, height: h, samples: Samples::U8(vals8.clone()) };
            let f8 = Image { width: w, height: h, samples: Samples::F32(vals8.iter().map(|&v| v as f32).collect()) };
            let (mut o8, mut of8) = (zeros_like(&s8), zeros_like(&f8));
            inv_warp(&mut o8, &s8, &flow, amount, opts).unwrap();
            inv_warp(&mut of8, &f8, &flow, amount, opts).unwrap();
            let want8: Vec<u8> = f32s(&of8).iter().map(|&v| sat_u8(v)).collect();
            assert_eq!(u8s(&o8), &want8[..], "u8 {opts:?} amount {amount}");

            let s16 = Image { width: w, height: h, samples: Samples::U16(vals16.clone()) };
            let f16 = Image { width: w, height: h, samples: Samples::F32(vals16.iter().map(|&v| v as f32).collect()) };
            let (mut o16, mut of16) = (zeros_like(&s16), zeros_like(&f16));
            inv_warp(&mut o16, &s16, &flow, amount, opts).unwrap();
            inv_warp(&mut of16, &f16, &flow, amount, opts).unwrap();
            let want16: Vec<u16> = f32s(&of16).iter().map(|&v| sat_u16(v)).collect();
            assert_eq!(u16s(&o16), &want16[..], "u16 {opts:?} amount {amount}");
        }
    }
}

#[test]
fn hold_with_an_axis_aligned_flow_replicates_the_border() {
    // horizontal flow: the clipped sample stays on its row and slides to the border column
    let (w, h) = (9, 5);
    let src = img_f32(w, h, |x, y, c| (100 * c + 10 * y + x) as f32);
    for (dx, dy) in [(3.0f32, 0.0f32), (-4.0, 0.0), (0.0, 2.0), (0.0, -3.0), (40.0, 0.0), (0.0, -40.0)] {
        let flow = const_flow(w, h, dx, dy);
        let mut out = zeros_like(&src);
        inv_warp(&mut out, &src, &flow, 1.0, HOLD).unwrap();
        for y in 0..h {
            for x in 0..w {
                let sx = (x as i32 + dx as i32).clamp(0, w as i32 - 1) as usize;
                let sy = (y as i32 + dy as i32).clamp(0, h as i32 - 1) as usize;
                assert_eq!(px_f32(&out, x, y), px_f32(&src, sx, sy), "flow ({dx},{dy}) pixel ({x},{y})");
            }
        }
    }
}

#[test]
fn composing_two_integer_shifts_equals_one_shift_inside_the_image() {
    let (w, h) = (16, 12);
    let src = img_u8(w, h, |x, y, c| (x * 13 + y * 29 + c * 7) as u8);
    let (mut a, mut b, mut c) = (zeros_like(&src), zeros_like(&src), zeros_like(&src));
    inv_warp(&mut a, &src, &const_flow(w, h, 2.0, 1.0), 1.0, HOLD).unwrap();
    inv_warp(&mut b, &a, &const_flow(w, h, 3.0, -1.0), 1.0, HOLD).unwrap();
    inv_warp(&mut c, &src, &const_flow(w, h, 5.0, 0.0), 1.0, HOLD).unwrap();
    for y in 1..h - 1 {
        for x in 0..w - 5 {
            let i = (y * w + x) * 4;
            assert_eq!(&u8s(&b)[i..i + 4], &u8s(&c)[i..i + 4], "({x},{y})");
        }
    }
}

#[test]
#[ignore = "timing probe: cargo test --release -- --ignored bench_1080p --nocapture"]
fn bench_1080p() {
    let (w, h) = (1920, 1080);
    let src = img_f32(w, h, |x, y, c| ((x * 7 + y * 3 + c) % 255) as f32 / 255.0);
    let mut flow = FlowField::new(w / 2, h / 2);
    for y in 0..h / 2 {
        for x in 0..w / 2 {
            flow.set(x, y, [((x + y) % 17) as f32 - 8.0, ((x * 3 + y) % 13) as f32 - 6.0]);
        }
    }
    let mut out = zeros_like(&src);
    for (name, opts) in [("hold", HOLD), ("fade", FADE), ("360", WRAP)] {
        let t = std::time::Instant::now();
        inv_warp(&mut out, &src, &flow, 0.5, opts).unwrap();
        println!("{name}: {:?}", t.elapsed());
    }
    let t = std::time::Instant::now();
    motion_blur_for_bench(&src, &flow);
    println!("motion blur: {:?}", t.elapsed());
}

fn motion_blur_for_bench(src: &Image, flow: &FlowField) {
    let mut out = zeros_like(src);
    crate::warp::blurcomp::motion_blur(&mut out, src, flow, 1.0, true, 0.5, false).unwrap();
}

// ---------------------------------------------------------------------------------------
// hostile values
// ---------------------------------------------------------------------------------------

#[test]
fn hostile_flow_and_amount_values_never_panic() {
    let specials = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        0.5,
        1e-30,
        -1e-30,
        1e7,
        -1e7,
        1e30,
        -1e30,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let (w, h) = (7, 6);
    let imgs = [
        img_u8(w, h, |x, y, c| (x * 30 + y * 11 + c) as u8),
        img_u16(w, h, |x, y, c| (x * 9000 + y * 700 + c) as u16),
        img_f32(w, h, |x, y, c| (x + y + c) as f32 * 0.1),
    ];
    let mut seed = 99u64;
    for img in &imgs {
        for opts in [HOLD, FADE, WRAP] {
            for &amount in &[1.0f32, -0.7, 1e30, f32::NAN, f32::INFINITY, 0.0] {
                // a flow whose entries are drawn from the special values, so every pair shows up
                let mut flow = FlowField::new(5, 4);
                for v in flow.data.iter_mut() {
                    *v = specials[(lcg(&mut seed) * specials.len() as f64) as usize % specials.len()];
                }
                let mut out = zeros_like(img);
                // errors are fine, panics are not; the coverage entry shares the code
                let _ = inv_warp(&mut out, img, &flow, amount, opts);
                let _ = inv_warp_coverage(&mut out, img, &flow, amount, opts);
                // the blur loops once per tap, so keep the finite magnitudes small (the tap limit has its own test)
                let mut small = flow.clone();
                for v in small.data.iter_mut() {
                    if v.is_finite() {
                        *v = v.clamp(-3000.0, 3000.0);
                    }
                }
                for &hiq in &[true, false] {
                    let _ = crate::warp::blurcomp::motion_blur(&mut out, img, &small, amount, hiq, 0.5, false);
                    let mut copy = img.clone();
                    let _ = crate::warp::blurcomp::compensate(&mut copy, &small, amount, hiq);
                }
            }
        }
    }
}

#[test]
fn absurd_blur_vectors_are_refused_instead_of_looping_for_minutes() {
    let src = img_f32(6, 5, |x, y, c| (x + y + c) as f32);
    let flow = const_flow(6, 5, 1e7, 0.0);
    let mut out = zeros_like(&src);
    for hiq in [true, false] {
        let e = crate::warp::blurcomp::motion_blur(&mut out, &src, &flow, 1.0, hiq, 0.5, false).unwrap_err();
        assert!(matches!(e, FxError::Unsupported(_)), "{e}");
    }
    // 50 000 pixels is within the limit and runs
    let flow = const_flow(6, 5, 50_000.0, 0.0);
    crate::warp::blurcomp::motion_blur(&mut out, &src, &flow, 1.0, true, 0.5, false).unwrap();
}
