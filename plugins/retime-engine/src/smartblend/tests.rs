//! Tests of the Smart Blend composites and compositions: hand goldens, exhaustive u8 comparison
//! against a step by step transcription of the formulas, random u16 and f32 comparisons, the
//! numpy float32 model fixtures (`tests/fixtures/f1_gen.py`), the compositions, thread count
//! invariance and a ground truth sanity test on a synthetic moving scene.

use super::*;
use crate::flow::{solve_pair, FlowSettings};
use crate::holefill::tests::{as_f64, first_diff, image_from, in_pool, load_cases, same_bits, Rng};
use crate::image::Fmt;

fn px_u8(img: &Image, i: usize) -> [u8; 4] {
    match &img.samples {
        Samples::U8(v) => [v[i * 4], v[i * 4 + 1], v[i * 4 + 2], v[i * 4 + 3]],
        _ => panic!("not u8"),
    }
}

fn px_f32(img: &Image, i: usize) -> [f32; 4] {
    match &img.samples {
        Samples::F32(v) => [v[i * 4], v[i * 4 + 1], v[i * 4 + 2], v[i * 4 + 3]],
        _ => panic!("not f32"),
    }
}

fn blend1(fmt: Fmt, p1: &[f32; 4], p2: &[f32; 4], w: f32, binary: bool) -> Image {
    // one pixel images in any depth from f32 values in native units
    let mk = |p: &[f32; 4]| -> Image {
        let mut img = Image::new(1, 1, fmt);
        match &mut img.samples {
            Samples::U8(v) => (0..4).for_each(|i| v[i] = p[i] as u8),
            Samples::U16(v) => (0..4).for_each(|i| v[i] = p[i] as u16),
            Samples::F32(v) => v.copy_from_slice(p),
        }
        img
    };
    let (a, b) = (mk(p1), mk(p2));
    let mut out = Image::new(1, 1, fmt);
    if binary {
        smart_blend_binary(&mut out, &a, &b, w).unwrap();
    } else {
        smart_blend(&mut out, &a, &b, w).unwrap();
    }
    out
}

// ------------------------------------------------------------------------------------------------
// step by step reference versions of the kernels
// ------------------------------------------------------------------------------------------------

mod reference {
    //! Step by step versions of the integer kernels (`white` 255 or 65535 held as integers),
    //! the f32 kernel and the binary alpha kernels.

    fn f(x: u32) -> f32 {
        x as f32
    }

    /// Integer store of the main path: `y = x + 0.5`, `0 > y` gives 0, `y > white` gives white,
    /// else the truncation.
    fn store(x: f32, white: u32) -> u32 {
        let y = x + 0.5f32;
        if 0.0f32 > y {
            0
        } else if y > white as f32 {
            white
        } else {
            (y as i64 as u32) & white // white is 0xff or 0xffff: the low byte or word
        }
    }

    fn raw(x: f32, white: u32) -> u32 {
        (x as i64 as u32) & white
    }

    pub fn smart_int(p1: [u32; 4], p2: [u32; 4], w: f32, white: u32) -> [u32; 4] {
        let wf = f(white);
        let (a1, a2) = (p1[0], p2[0]);
        // x4 / x6 / x7 / x8: alpha, r, g, b of over(P1, P2)
        let (x4, x6, x7, x8);
        if a1 == 0 {
            x6 = f(p2[1]);
            x7 = f(p2[2]);
            x8 = f(p2[3]);
            x4 = f(a2);
        } else if a1 == white {
            x6 = f(p1[1]);
            x7 = f(p1[2]);
            x8 = f(p1[3]);
            x4 = f(a1);
        } else {
            let x2 = f(a1) / wf;
            let x1 = 1.0f32 - x2;
            x6 = f(p2[1]) * x1 + f(p1[1]);
            x7 = f(p2[2]) * x1 + f(p1[2]);
            x8 = f(p2[3]) * x1 + f(p1[3]);
            let x0 = f(a2);
            x4 = (wf - x0) * x2 + x0;
        }
        // x5 / x10 / x11 / x12: alpha, r, g, b of over(P2, P1)
        let (x5, x10, x11, x12);
        if a2 == 0 {
            x10 = f(p1[1]);
            x11 = f(p1[2]);
            x12 = f(p1[3]);
            x5 = f(a1);
        } else if a2 == white {
            x10 = f(p2[1]);
            x11 = f(p2[2]);
            x12 = f(p2[3]);
            x5 = f(a2);
        } else {
            let x2 = f(a2) / wf;
            let x1 = 1.0f32 - x2;
            x10 = f(p1[1]) * x1 + f(p2[1]);
            x11 = f(p1[2]) * x1 + f(p2[2]);
            x12 = f(p1[3]) * x1 + f(p2[3]);
            let x0 = f(a1);
            x5 = (wf - x0) * x2 + x0;
        }
        let ecx = if a1 > a2 { a1 } else { a2 }; // the larger alpha
        let x9 = f(ecx);
        let x0 = (x10 - x6) * w + x6;
        let x1 = (x11 - x7) * w + x7;
        let x2 = (x12 - x8) * w + x8;
        let x3 = (x5 - x4) * w + x4;
        if x3 == x9 {
            return [store(x3, white), store(x0, white), store(x1, white), store(x2, white)];
        }
        if x3 == 0.0 {
            // c2 unless c1.alpha > c2.alpha
            return if x4 > x5 {
                [raw(x4, white), raw(x6, white), raw(x7, white), raw(x8, white)]
            } else {
                [raw(x5, white), raw(x10, white), raw(x11, white), raw(x12, white)]
            };
        }
        let (r, g, b) = ((x0 * x9) / x3, (x1 * x9) / x3, (x2 * x9) / x3);
        [store(x9, white), store(r, white), store(g, white), store(b, white)]
    }

    pub fn binary_int(p1: [u32; 4], p2: [u32; 4], w: f32, white: u32) -> [u32; 4] {
        let wf = f(white);
        let (a1, a2) = (p1[0], p2[0]); // r8b is P2.alpha, dl is P1.alpha in the u8 listing
        // x15 / x12 / x13 / x14: alpha, r, g, b of X
        let (x15, x12, x13, x14);
        if a2 == white {
            let x1 = 1.0f32 - f(a1) / wf;
            x12 = f(p2[1]) * x1 + f(p1[1]);
            x13 = f(p2[2]) * x1 + f(p1[2]);
            x14 = f(p2[3]) * x1 + f(p1[3]);
            x15 = f(a2);
        } else if a1 == white {
            x12 = f(p1[1]);
            x13 = f(p1[2]);
            x14 = f(p1[3]);
            x15 = f(a1);
        } else {
            x12 = 0.0;
            x13 = 0.0;
            x14 = 0.0;
            x15 = 0.0;
        }
        // x8 / x9 / x6 / x7: alpha, r, g, b of Y
        let (x8, x9, x6, x7);
        if a1 == white {
            let x1 = 1.0f32 - f(a2) / wf;
            x9 = f(p1[1]) * x1 + f(p2[1]);
            x6 = f(p1[2]) * x1 + f(p2[2]);
            x7 = f(p1[3]) * x1 + f(p2[3]);
            x8 = f(a1);
        } else if a2 == white {
            x9 = f(p2[1]);
            x6 = f(p2[2]);
            x7 = f(p2[3]);
            x8 = f(a2);
        } else {
            x9 = 0.0;
            x6 = 0.0;
            x7 = 0.0;
            x8 = 0.0;
        }
        let r = (x9 - x12) * w + x12;
        let g = (x6 - x13) * w + x13;
        let b = (x7 - x14) * w + x14;
        let a = (x8 - x15) * w + x15;
        [store(a, white), store(r, white), store(g, white), store(b, white)]
    }

    fn feq(a: f32, b: f32) -> bool {
        a == b
    }

    pub fn smart_f32(p1: [f32; 4], p2: [f32; 4], w: f32) -> [f32; 4] {
        let (a1, a2) = (p1[0], p2[0]);
        let (x5, x9, x10, x6); // c1 = over(P1, P2): alpha, r, g, b
        if feq(a1, 0.0) {
            x9 = p2[1];
            x10 = p2[2];
            x6 = p2[3];
            x5 = a2;
        } else if feq(a1, 1.0) {
            x9 = p1[1];
            x10 = p1[2];
            x6 = p1[3];
            x5 = a1;
        } else {
            let m6 = 1.0f32 - a1;
            let m5 = 1.0f32 - a2;
            x6 = m6 * p2[3] + p1[3];
            x9 = m6 * p2[1] + p1[1];
            x10 = m6 * p2[2] + p1[2];
            x5 = m5 * a1 + a2;
        }
        let (x7, x11, x12, x8); // c2 = over(P2, P1)
        if feq(a2, 0.0) {
            x11 = p1[1];
            x12 = p1[2];
            x8 = p1[3];
            x7 = a1;
        } else if feq(a2, 1.0) {
            x11 = p2[1];
            x12 = p2[2];
            x8 = p2[3];
            x7 = a2;
        } else {
            let m8 = 1.0f32 - a2;
            let m7 = 1.0f32 - a1;
            x8 = m8 * p1[3] + p2[3];
            x11 = m8 * p1[1] + p2[1];
            x12 = m8 * p1[2] + p2[2];
            x7 = m7 * a2 + a1;
        }
        let max_alpha = if a1 > a2 { a1 } else { a2 };
        let x4 = (x7 - x5) * w + x5;
        let x0 = (x11 - x9) * w + x9;
        let x2 = (x12 - x10) * w + x10;
        let x3 = (x8 - x6) * w + x6;
        if feq(x4, max_alpha) {
            return [x4 + 0.0, x0 + 0.0, x2 + 0.0, x3 + 0.0];
        }
        if feq(x4, 0.0) {
            return if x5 > x7 { [x5, x9, x10, x6] } else { [x7, x11, x12, x8] };
        }
        let (r, g, b) = ((x0 * max_alpha) / x4, (x2 * max_alpha) / x4, (x3 * max_alpha) / x4);
        [max_alpha + 0.0, r + 0.0, g + 0.0, b + 0.0]
    }

    pub fn binary_f32(p1: [f32; 4], p2: [f32; 4], w: f32) -> [f32; 4] {
        let (a1, a2) = (p1[0], p2[0]);
        let (x15, x12, x13, x7);
        if feq(a2, 1.0) {
            let m = 1.0f32 - a1;
            x7 = m * p2[3] + p1[3];
            x12 = m * p2[1] + p1[1];
            x13 = m * p2[2] + p1[2];
            x15 = a2;
        } else if feq(a1, 1.0) {
            x12 = p1[1];
            x13 = p1[2];
            x7 = p1[3];
            x15 = a1;
        } else {
            x15 = 0.0;
            x7 = 0.0;
            x13 = 0.0;
            x12 = 0.0;
        }
        let (x11, x8, x9, x6);
        if feq(a1, 1.0) {
            let m = 1.0f32 - a2;
            x6 = m * p1[3] + p2[3];
            x8 = m * p1[1] + p2[1];
            x9 = m * p1[2] + p2[2];
            x11 = a1;
        } else if feq(a2, 1.0) {
            x8 = p2[1];
            x9 = p2[2];
            x6 = p2[3];
            x11 = a2;
        } else {
            x11 = 0.0;
            x6 = 0.0;
            x9 = 0.0;
            x8 = 0.0;
        }
        let r = (x8 - x12) * w + x12;
        let g = (x9 - x13) * w + x13;
        let b = (x6 - x7) * w + x7;
        let a = (x11 - x15) * w + x15;
        [a + 0.0, r + 0.0, g + 0.0, b + 0.0]
    }
}

// ------------------------------------------------------------------------------------------------
// hand goldens of the composites
// ------------------------------------------------------------------------------------------------

#[test]
fn golden_two_opaque_pixels_mix_linearly() {
    // c1 = P1, c2 = P2, v = (c2 - c1) * 0.5 + c1 = (255, 100, 100, 125), alpha equals amax
    let out = blend1(Fmt::U8, &[255.0, 200.0, 100.0, 50.0], &[255.0, 0.0, 100.0, 200.0], 0.5, false);
    assert_eq!(px_u8(&out, 0), [255, 100, 100, 125]);
}

#[test]
fn golden_a_transparent_pixel_takes_the_other_pixel_whatever_the_weight() {
    for w in [0.0f32, 0.1, 0.5, 0.9, 1.0] {
        let out = blend1(Fmt::U8, &[0.0, 0.0, 0.0, 0.0], &[255.0, 10.0, 20.0, 30.0], w, false);
        assert_eq!(px_u8(&out, 0), [255, 10, 20, 30], "w {w}");
        let out = blend1(Fmt::U8, &[255.0, 10.0, 20.0, 30.0], &[0.0, 0.0, 0.0, 0.0], w, false);
        assert_eq!(px_u8(&out, 0), [255, 10, 20, 30], "w {w}");
    }
}

#[test]
fn golden_a_lone_translucent_pixel_keeps_its_alpha_and_colour() {
    // c1 = P1 over nothing = (128, 128, 0, 0), c2 = P1 (alpha of P2 is 0), amax = 128: the result is P1
    for w in [0.0f32, 0.3, 0.5, 1.0] {
        let out = blend1(Fmt::U8, &[128.0, 128.0, 0.0, 0.0], &[0.0, 0.0, 0.0, 0.0], w, false);
        assert_eq!(px_u8(&out, 0), [128, 128, 0, 0], "w {w}");
    }
}

#[test]
fn golden_two_translucent_pixels_are_renormalised_to_the_larger_alpha() {
    // red 50 percent and green 50 percent at w = 0.5: c1 = (191.75, 128, 63.75, 0), c2 = (191.75, 63.75, 128, 0),
    // v = (191.75, 95.875, 95.875, 0) has alpha 191.75 != amax = 128, so rgb = v.rgb * 128 / 191.75 = 64
    let out = blend1(Fmt::U8, &[128.0, 128.0, 0.0, 0.0], &[128.0, 0.0, 128.0, 0.0], 0.5, false);
    assert_eq!(px_u8(&out, 0), [128, 64, 64, 0]);
}

#[test]
fn golden_the_alpha_zero_fallback_stores_the_raw_colour_without_rounding() {
    // alpha 2.0 (outside 0..1): c1.alpha = (1 - 2) * 2 + 2 = 0 and c2.alpha = 0, v.alpha = 0 != amax = 2;
    // c1.alpha > c2.alpha is false so c2 is stored raw: (0, P1.rgb * -1 + P2.rgb) = (0, 3, 3, 3)
    let out = blend1(Fmt::F32, &[2.0, 1.0, 2.0, 3.0], &[2.0, 4.0, 5.0, 6.0], 0.5, false);
    assert_eq!(px_f32(&out, 0), [0.0, 3.0, 3.0, 3.0]);
    // the fallback pixel for c1.alpha > c2.alpha is c1; craft by hand with the generic kernel
    let (v, raw) = smart_pixel([2.0, 1.0, 1.0, 1.0], [2.0, 4.0, 5.0, 6.0], 0.0, 1.0);
    assert!(raw && v[0] == 0.0);
}

#[test]
fn integer_stores_of_both_paths() {
    // main path: x + 0.5, clamp, truncate; raw path: plain truncation (never reached for integer alphas, so tested directly)
    assert_eq!(<u8 as Px>::store_plain(255.4), 255);
    assert_eq!(<u8 as Px>::store_plain(255.6), 255);
    assert_eq!(<u8 as Px>::store_plain(-0.6), 0);
    assert_eq!(<u8 as Px>::store_plain(0.5), 1);
    assert_eq!(<u8 as Px>::store_plain(f32::NAN), 0);
    assert_eq!(<u8 as Px>::store_raw(127.9), 127);
    assert_eq!(<u8 as Px>::store_raw(257.0), 1); // low byte of the 64 bit integer
    assert_eq!(<u16 as Px>::store_plain(65535.4), 65535);
    assert_eq!(<u16 as Px>::store_plain(70000.0), 65535);
    assert_eq!(<u16 as Px>::store_plain(-5.0), 0);
    assert_eq!(<u16 as Px>::store_raw(65537.0), 1);
    assert_eq!(<f32 as Px>::store_plain(-0.0).to_bits(), 0); // + 0.0 normalises the sign
    assert_eq!(<f32 as Px>::store_raw(-0.0).to_bits(), (-0.0f32).to_bits());
}

#[test]
fn golden_binary_alpha_variant() {
    // both opaque: the same as the plain composite
    let out = blend1(Fmt::U8, &[255.0, 200.0, 100.0, 50.0], &[255.0, 0.0, 100.0, 200.0], 0.5, true);
    assert_eq!(px_u8(&out, 0), [255, 100, 100, 125]);
    // P1 translucent, P2 opaque: X = (255, P2.rgb * (1 - 100/255) + P1.rgb), Y = P2; w = 0.25
    // X = (255, 56.078, 62.157, 68.235), Y = (255, 10, 20, 30): R 44.56 -> 45, G 51.62 -> 52, B 58.68 -> 59
    let out = blend1(Fmt::U8, &[100.0, 50.0, 50.0, 50.0], &[255.0, 10.0, 20.0, 30.0], 0.25, true);
    assert_eq!(px_u8(&out, 0), [255, 45, 52, 59]);
    // neither alpha exactly white: X = Y = 0 and the pixel is cleared, whatever the colours
    let out = blend1(Fmt::U8, &[100.0, 50.0, 50.0, 50.0], &[200.0, 70.0, 70.0, 70.0], 0.5, true);
    assert_eq!(px_u8(&out, 0), [0, 0, 0, 0]);
    // the plain composite keeps something there
    let out = blend1(Fmt::U8, &[100.0, 50.0, 50.0, 50.0], &[200.0, 70.0, 70.0, 70.0], 0.5, false);
    assert!(px_u8(&out, 0)[0] >= 200);
    // float: alpha exactly 1.0 counts, 0.999 does not
    let out = blend1(Fmt::F32, &[0.999, 0.5, 0.5, 0.5], &[0.5, 0.2, 0.2, 0.2], 0.5, true);
    assert_eq!(px_f32(&out, 0), [0.0; 4]);
    let out = blend1(Fmt::F32, &[1.0, 0.5, 0.5, 0.5], &[0.5, 0.2, 0.2, 0.2], 0.0, true);
    // w = 0: X = P1 (A2 != 1, A1 == 1), out = X = P1
    assert_eq!(px_f32(&out, 0), [1.0, 0.5, 0.5, 0.5]);
}

#[test]
fn shape_mismatches_are_errors() {
    let mut out = Image::new(4, 4, Fmt::U8);
    let a = Image::new(4, 4, Fmt::U8);
    assert!(matches!(smart_blend(&mut out, &a, &Image::new(4, 5, Fmt::U8), 0.5), Err(FxError::Mismatch(_))));
    assert!(matches!(smart_blend(&mut out, &a, &Image::new(4, 4, Fmt::U16), 0.5), Err(FxError::Mismatch(_))));
    assert!(matches!(smart_blend_binary(&mut out, &Image::new(3, 4, Fmt::U8), &a, 0.5), Err(FxError::Mismatch(_))));
    // an empty image is fine
    let mut e = Image::new(0, 0, Fmt::U8);
    smart_blend(&mut e, &Image::new(0, 0, Fmt::U8), &Image::new(0, 0, Fmt::U8), 0.5).unwrap();
}

// ------------------------------------------------------------------------------------------------
// exhaustive and random comparisons with the transcriptions
// ------------------------------------------------------------------------------------------------

#[test]
fn u8_smart_blend_all_alpha_pairs_against_a_step_by_step_model() {
    // all 256 x 256 alpha pairs, four colour combinations (valid premultiplied extremes and invalid ones), seven weights
    let weights = [0.0f32, 0.25, 0.5, 0.75, 1.0, 0.1, 0.9];
    let colours: [([u8; 3], [u8; 3]); 4] = [
        ([0, 0, 0], [0, 0, 0]),
        ([255, 255, 255], [255, 0, 128]),
        ([17, 200, 3], [250, 5, 99]),
        ([1, 2, 3], [0, 255, 254]),
    ];
    let mut checked = 0usize;
    for &w in &weights {
        for &(c1, c2) in &colours {
            let mut p1 = Vec::with_capacity(65536 * 4);
            let mut p2 = Vec::with_capacity(65536 * 4);
            for a1 in 0..256u32 {
                for a2 in 0..256u32 {
                    p1.extend_from_slice(&[a1 as u8, c1[0], c1[1], c1[2]]);
                    p2.extend_from_slice(&[a2 as u8, c2[0], c2[1], c2[2]]);
                }
            }
            let a = Image { width: 256, height: 256, samples: Samples::U8(p1.clone()) };
            let b = Image { width: 256, height: 256, samples: Samples::U8(p2.clone()) };
            let mut out = Image::new(256, 256, Fmt::U8);
            smart_blend(&mut out, &a, &b, w).unwrap();
            let mut out_bin = Image::new(256, 256, Fmt::U8);
            smart_blend_binary(&mut out_bin, &a, &b, w).unwrap();
            let (Samples::U8(o), Samples::U8(ob)) = (&out.samples, &out_bin.samples) else { unreachable!() };
            for i in 0..65536 {
                let q1 = [p1[i * 4] as u32, p1[i * 4 + 1] as u32, p1[i * 4 + 2] as u32, p1[i * 4 + 3] as u32];
                let q2 = [p2[i * 4] as u32, p2[i * 4 + 1] as u32, p2[i * 4 + 2] as u32, p2[i * 4 + 3] as u32];
                let want = reference::smart_int(q1, q2, w, 255);
                let got = [o[i * 4] as u32, o[i * 4 + 1] as u32, o[i * 4 + 2] as u32, o[i * 4 + 3] as u32];
                assert_eq!(got, want, "smart: pixel {:?} {:?} w {w}", q1, q2);
                let wantb = reference::binary_int(q1, q2, w, 255);
                let gotb = [ob[i * 4] as u32, ob[i * 4 + 1] as u32, ob[i * 4 + 2] as u32, ob[i * 4 + 3] as u32];
                assert_eq!(gotb, wantb, "binary: pixel {:?} {:?} w {w}", q1, q2);
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 7 * 4 * 65536);
}

#[test]
fn u8_opaque_pairs_are_the_linear_mix_and_a_transparent_partner_passes_the_pixel_through() {
    let mut rng = Rng(4);
    for _ in 0..20000 {
        let p1 = [255u8, rng.below(256) as u8, rng.below(256) as u8, rng.below(256) as u8];
        let p2 = [255u8, rng.below(256) as u8, rng.below(256) as u8, rng.below(256) as u8];
        let w = rng.unit();
        let o = px_u8(&blend1(Fmt::U8, &p1.map(f32::from), &p2.map(f32::from), w, false), 0);
        assert_eq!(o[0], 255);
        for c in 1..4 {
            let (x, y) = (f32::from(p1[c]), f32::from(p2[c]));
            assert_eq!(o[c], ((y - x) * w + x + 0.5f32) as u8, "{p1:?} {p2:?} w {w}");
        }
        let clear = [0.0f32; 4];
        assert_eq!(px_u8(&blend1(Fmt::U8, &clear, &p1.map(f32::from), w, false), 0), p1);
        assert_eq!(px_u8(&blend1(Fmt::U8, &p1.map(f32::from), &clear, w, false), 0), p1);
    }
}

#[test]
fn u16_smart_blend_random_and_boundary_pixels_against_a_step_by_step_model() {
    let mut rng = Rng(16);
    let interesting = [0u16, 1, 2, 255, 256, 32767, 32768, 65534, 65535];
    let mut pix: Vec<([u32; 4], [u32; 4], f32)> = Vec::new();
    for &a1 in &interesting {
        for &a2 in &interesting {
            for &c in &[0u16, 7, 32768, 65535] {
                pix.push(([a1 as u32, c as u32, 65535 - c as u32, 3], [a2 as u32, 65535 - c as u32, c as u32, 40000], 0.37));
            }
        }
    }
    for _ in 0..200_000 {
        let r = |rng: &mut Rng| -> u32 {
            match rng.below(6) {
                0 => 0,
                1 => 65535,
                _ => rng.below(65536) as u32,
            }
        };
        pix.push(([r(&mut rng), r(&mut rng), r(&mut rng), r(&mut rng)], [r(&mut rng), r(&mut rng), r(&mut rng), r(&mut rng)], rng.unit()));
    }
    for (chunk, _) in pix.chunks(4096).zip(0..) {
        for &w in &[0.5f32, 0.37, 0.0, 1.0] {
            let n = chunk.len();
            let mut p1 = Vec::new();
            let mut p2 = Vec::new();
            for (a, b, _) in chunk {
                p1.extend(a.iter().map(|&v| v as u16));
                p2.extend(b.iter().map(|&v| v as u16));
            }
            let a = Image { width: n, height: 1, samples: Samples::U16(p1) };
            let b = Image { width: n, height: 1, samples: Samples::U16(p2) };
            let mut out = Image::new(n, 1, Fmt::U16);
            let mut outb = Image::new(n, 1, Fmt::U16);
            smart_blend(&mut out, &a, &b, w).unwrap();
            smart_blend_binary(&mut outb, &a, &b, w).unwrap();
            let (Samples::U16(o), Samples::U16(ob)) = (&out.samples, &outb.samples) else { unreachable!() };
            for (i, (p, q, _)) in chunk.iter().enumerate() {
                let want = reference::smart_int(*p, *q, w, 65535);
                let got = [o[i * 4] as u32, o[i * 4 + 1] as u32, o[i * 4 + 2] as u32, o[i * 4 + 3] as u32];
                assert_eq!(got, want, "smart u16 {p:?} {q:?} w {w}");
                let wantb = reference::binary_int(*p, *q, w, 65535);
                let gotb = [ob[i * 4] as u32, ob[i * 4 + 1] as u32, ob[i * 4 + 2] as u32, ob[i * 4 + 3] as u32];
                assert_eq!(gotb, wantb, "binary u16 {p:?} {q:?} w {w}");
            }
        }
    }
}

fn same_f(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

#[test]
fn f32_smart_blend_random_and_special_pixels_against_a_step_by_step_model() {
    let mut rng = Rng(32);
    let specials = [0.0f32, -0.0, 1.0, 0.5, 2.0, -0.25, 1e-30, 1e30, f32::NAN, f32::INFINITY, 0.999999, 1.000001, 0.25];
    let mut n = 0usize;
    let mk = |rng: &mut Rng| -> [f32; 4] {
        let r = |rng: &mut Rng| -> f32 {
            if rng.below(3) == 0 {
                specials[rng.below(specials.len() as u64) as usize]
            } else {
                rng.unit()
            }
        };
        [r(rng), r(rng), r(rng), r(rng)]
    };
    let count = 100_000;
    let (mut p1, mut p2) = (Vec::new(), Vec::new());
    for _ in 0..count {
        p1.extend_from_slice(&mk(&mut rng));
        p2.extend_from_slice(&mk(&mut rng));
    }
    for &w in &[0.5f32, 0.0, 1.0, 0.3333333, -0.5, 1.5] {
        let a = Image { width: count, height: 1, samples: Samples::F32(p1.clone()) };
        let b = Image { width: count, height: 1, samples: Samples::F32(p2.clone()) };
        let mut out = Image::new(count, 1, Fmt::F32);
        let mut outb = Image::new(count, 1, Fmt::F32);
        smart_blend(&mut out, &a, &b, w).unwrap();
        smart_blend_binary(&mut outb, &a, &b, w).unwrap();
        let (Samples::F32(o), Samples::F32(ob)) = (&out.samples, &outb.samples) else { unreachable!() };
        for i in 0..count {
            let q1 = [p1[i * 4], p1[i * 4 + 1], p1[i * 4 + 2], p1[i * 4 + 3]];
            let q2 = [p2[i * 4], p2[i * 4 + 1], p2[i * 4 + 2], p2[i * 4 + 3]];
            let want = reference::smart_f32(q1, q2, w);
            let wantb = reference::binary_f32(q1, q2, w);
            for c in 0..4 {
                assert!(same_f(o[i * 4 + c], want[c]), "smart f32 {q1:?} {q2:?} w {w} channel {c}: {} vs {}", o[i * 4 + c], want[c]);
                assert!(same_f(ob[i * 4 + c], wantb[c]), "binary f32 {q1:?} {q2:?} w {w} channel {c}: {} vs {}", ob[i * 4 + c], wantb[c]);
            }
            n += 1;
        }
    }
    assert_eq!(n, 6 * count);
}

// ------------------------------------------------------------------------------------------------
// numpy float32 model fixtures
// ------------------------------------------------------------------------------------------------

#[test]
fn model_fixtures_smart_blend_is_reproduced_bit_for_bit() {
    let cases = load_cases();
    let mut counts = [[0usize; 3]; 2];
    for (i, c) in cases.iter().enumerate().filter(|(_, c)| c.kind == 1 || c.kind == 2) {
        let mut out = Image::new(c.w, c.h, c.p1.fmt());
        let p2 = c.p2.as_ref().unwrap();
        if c.kind == 1 {
            smart_blend(&mut out, &c.p1, p2, c.weight).unwrap();
        } else {
            smart_blend_binary(&mut out, &c.p1, p2, c.weight).unwrap();
        }
        if let Some(pos) = first_diff(&out, &c.out) {
            panic!(
                "case {i} (kind {}, depth {}, w {}): first difference at sample {pos}: got {} want {}",
                c.kind,
                c.depth,
                c.weight,
                as_f64(&out)[pos],
                as_f64(&c.out)[pos]
            );
        }
        counts[(c.kind - 1) as usize][c.depth as usize] += 1;
    }
    assert!(counts.iter().all(|row| row.iter().all(|&n| n == 6)), "{counts:?}");
}

// ------------------------------------------------------------------------------------------------
// thread count invariance
// ------------------------------------------------------------------------------------------------

#[test]
fn composites_are_identical_for_one_and_four_threads() {
    let mut rng = Rng(8);
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let a = image_from(fmt, 77, 53, &mut rng);
        let b = image_from(fmt, 77, 53, &mut rng);
        let run = |threads: usize, binary: bool| {
            let mut out = Image::new(77, 53, fmt);
            in_pool(threads, || {
                if binary {
                    smart_blend_binary(&mut out, &a, &b, 0.37).unwrap()
                } else {
                    smart_blend(&mut out, &a, &b, 0.37).unwrap()
                }
            });
            out
        };
        for binary in [false, true] {
            assert!(same_bits(&run(1, binary), &run(4, binary)), "{fmt:?} binary {binary}");
        }
    }
}

// ------------------------------------------------------------------------------------------------
// compositions
// ------------------------------------------------------------------------------------------------

fn opaque(fmt: Fmt, w: usize, h: usize, rng: &mut Rng) -> Image {
    let mut img = image_from(fmt, w, h, rng);
    let white = fmt.white();
    match &mut img.samples {
        Samples::U8(v) => v.chunks_mut(4).for_each(|p| p[0] = 255),
        Samples::U16(v) => v.chunks_mut(4).for_each(|p| p[0] = 65535),
        Samples::F32(v) => v.chunks_mut(4).for_each(|p| p[0] = white),
    }
    img
}

fn zero_flow(w: usize, h: usize) -> FlowField {
    FlowField::new(w, h)
}

#[test]
fn nearest_smart_with_no_motion_copies_the_chosen_frame() {
    let mut rng = Rng(6);
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let a = opaque(fmt, 20, 12, &mut rng);
        let b = opaque(fmt, 20, 12, &mut rng);
        let flow = zero_flow(20, 12);
        let mut out = Image::new(20, 12, fmt);
        nearest_smart(&mut out, &a, &b, &flow, 0.7, SmartOpts::default()).unwrap();
        assert!(same_bits(&out, &a), "{fmt:?} wa 0.7 takes A");
        nearest_smart(&mut out, &a, &b, &flow, 0.3, SmartOpts::default()).unwrap();
        assert!(same_bits(&out, &b), "{fmt:?} wa 0.3 takes B");
        nearest_smart(&mut out, &a, &b, &flow, 0.5, SmartOpts::default()).unwrap();
        assert!(same_bits(&out, &a), "{fmt:?} wa 0.5 takes A (wa >= 0.5)");
    }
}

#[test]
fn nearest_smart_fills_what_the_warp_left_empty() {
    // a constant flow of +12 px with amount 0.5 samples 6 px to the right: the last columns have no source
    let mut rng = Rng(9);
    let (w, h) = (40usize, 16usize);
    let a = opaque(Fmt::U8, w, h, &mut rng);
    let b = opaque(Fmt::U8, w, h, &mut rng);
    let mut flow = FlowField::new(w, h);
    for i in 0..w * h {
        flow.data[i * 2] = 12.0;
    }
    let mut out = Image::new(w, h, Fmt::U8);
    nearest_smart(&mut out, &a, &b, &flow, 0.5, SmartOpts::default()).unwrap();
    // the warp alone, before the fill
    let mut warped = Image::new(w, h, Fmt::U8);
    let cov = inv_warp_coverage(&mut warped, &a, &flow, 0.5, WarpOpts { hold: false, is360: false, mode: 1 }).unwrap();
    assert!(cov.data.iter().any(|&c| c == 0) && cov.data.iter().any(|&c| c == 255));
    let mut holes = cov.clone();
    holes.invert();
    let mut expect = warped.clone();
    fill_holes(&mut expect, &holes, false).unwrap();
    assert!(same_bits(&out, &expect));
    // known pixels (coverage 255) are the shifted frame, and the strip is opaque after the fill
    for y in 0..h {
        for x in 0..w {
            let o = px_u8(&out, y * w + x);
            assert_eq!(o[0], 255, "alpha at {x},{y}");
            if cov.at(x, y) == 255 {
                assert_eq!(o, px_u8(&a, y * w + x + 6), "pixel {x},{y}");
            }
        }
    }
}

#[test]
fn blend_smart_with_no_motion_is_the_plain_blend_for_opaque_frames() {
    let mut rng = Rng(12);
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let a = opaque(fmt, 24, 14, &mut rng);
        let b = opaque(fmt, 24, 14, &mut rng);
        let flow = zero_flow(24, 14);
        for wa in [0.25f32, 0.5, 0.8] {
            let mut out = Image::new(24, 14, fmt);
            blend_smart(&mut out, &a, &flow, &b, &flow, wa, SmartOpts::default()).unwrap();
            let mut plain = Image::new(24, 14, fmt);
            blend(&mut plain, &a, &b, 1.0f32 - wa);
            if let Some(pos) = first_diff(&out, &plain) {
                panic!("{fmt:?} wa {wa}: first difference at {pos}: {} vs {}", as_f64(&out)[pos], as_f64(&plain)[pos]);
            }
        }
    }
}

#[test]
fn weighted_blend_smart_with_no_motion_is_the_smart_blend_of_the_frames() {
    // zero flow: the warps copy, the change masks are empty, the sharpening keeps the warped frames
    let mut rng = Rng(13);
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let a = opaque(fmt, 30, 18, &mut rng);
        let b = opaque(fmt, 30, 18, &mut rng);
        let flow = zero_flow(30, 18);
        for wa in [0.3f32, 0.5, 0.9] {
            let mut out = Image::new(30, 18, fmt);
            weighted_blend_smart(&mut out, &a, &flow, &b, &flow, wa, SmartOpts::default()).unwrap();
            let mut want = Image::new(30, 18, fmt);
            smart_blend(&mut want, &a, &b, 1.0f32 - wa).unwrap();
            assert!(same_bits(&out, &want), "{fmt:?} wa {wa}");
        }
    }
}

#[test]
fn weighted_variants_ignore_the_options() {
    // the weighted compositions use the plain composite and the flag false: binary_alpha and is360
    // change nothing
    let mut rng = Rng(14);
    let a = image_from(Fmt::U8, 26, 16, &mut rng);
    let b = image_from(Fmt::U8, 26, 16, &mut rng);
    let mut flow = FlowField::new(26, 16);
    for i in 0..26 * 16 {
        flow.data[i * 2] = 3.0;
        flow.data[i * 2 + 1] = -1.0;
    }
    let plain = {
        let mut out = Image::new(26, 16, Fmt::U8);
        weighted_blend_smart(&mut out, &a, &flow, &b, &flow, 0.4, SmartOpts::default()).unwrap();
        out
    };
    let flagged = {
        let mut out = Image::new(26, 16, Fmt::U8);
        weighted_blend_smart(&mut out, &a, &flow, &b, &flow, 0.4, SmartOpts { binary_alpha: true, is360: true }).unwrap();
        out
    };
    assert!(same_bits(&plain, &flagged));
}

#[test]
fn weighted_tail_sequence_and_the_plain_blend_without_coverage() {
    let mut rng = Rng(15);
    let (w, h) = (28usize, 17usize);
    let a = image_from(Fmt::U8, w, h, &mut rng);
    let b = image_from(Fmt::U8, w, h, &mut rng);
    // pretend warped frames: a slightly changed copy of each
    let warp = |src: &Image, rng: &mut Rng| {
        let mut t = src.clone();
        if let Samples::U8(v) = &mut t.samples {
            for s in v.iter_mut() {
                *s = s.saturating_add(rng.below(60) as u8);
            }
        }
        t
    };
    let (ta0, tb0) = (warp(&a, &mut rng), warp(&b, &mut rng));
    let wa = 0.35f32;
    let frac = 1.0f32 - wa;
    // without coverage: sharpen both, plain blend
    let (mut ta, mut tb) = (ta0.clone(), tb0.clone());
    let mut out = Image::new(w, h, Fmt::U8);
    weighted_tail(&mut out, &mut ta, &a, &mut tb, &b, wa, None, SmartOpts::default()).unwrap();
    let (mut sa, mut sb) = (ta0.clone(), tb0.clone());
    sharpen_step(&mut sa, &a, frac);
    sharpen_step(&mut sb, &b, wa);
    let mut want = Image::new(w, h, Fmt::U8);
    blend(&mut want, &sa, &sb, frac);
    assert!(same_bits(&out, &want));
    assert!(same_bits(&ta, &sa) && same_bits(&tb, &sb), "the warped frames are sharpened in place");
    // with coverage: smart blend, combine, invert, fill
    let mut c1 = Coverage::new(w, h, 255);
    let mut c2 = Coverage::new(w, h, 255);
    for x in 0..5 {
        c1.data[3 * w + x] = 0;
        c2.data[3 * w + x] = 0;
    }
    c1.data[10 * w + 10] = 0; // only one of the two: not a hole
    let (mut ta, mut tb) = (ta0.clone(), tb0.clone());
    let mut out = Image::new(w, h, Fmt::U8);
    let (mut c1m, c2r) = (c1.clone(), c2.clone());
    weighted_tail(&mut out, &mut ta, &a, &mut tb, &b, wa, Some((&mut c1m, &c2r)), SmartOpts::default()).unwrap();
    let mut want = Image::new(w, h, Fmt::U8);
    smart_blend(&mut want, &sa, &sb, frac).unwrap();
    let (mut cc1, cc2) = (c1.clone(), c2.clone());
    smart_finish(&mut want, &mut cc1, &cc2, false).unwrap();
    assert!(same_bits(&out, &want));
    // cov1 was left as the holes mask: 255 where both coverages were 0
    assert!(c1m.data.iter().enumerate().all(|(i, &v)| v == if (3 * w..3 * w + 5).contains(&i) { 255 } else { 0 }));
}

#[test]
fn blend_tail_picks_the_variant_and_finishes() {
    let mut rng = Rng(16);
    let (w, h) = (20usize, 11usize);
    // translucent frames so that the two composites differ
    let mut a = image_from(Fmt::U8, w, h, &mut rng);
    let mut b = image_from(Fmt::U8, w, h, &mut rng);
    for img in [&mut a, &mut b] {
        if let Samples::U8(v) = &mut img.samples {
            for p in v.chunks_mut(4) {
                p[0] = 100 + (p[0] % 100);
                for c in 1..4 {
                    p[c] %= p[0];
                }
            }
        }
    }
    let mut c1 = Coverage::new(w, h, 255);
    let c2 = Coverage::new(w, h, 255);
    c1.data[0] = 0; // not a hole after the combine
    for binary in [false, true] {
        let opts = SmartOpts { binary_alpha: binary, is360: false };
        let mut out = Image::new(w, h, Fmt::U8);
        let mut c1m = c1.clone();
        blend_tail(&mut out, &a, &b, 0.4, Some((&mut c1m, &c2)), opts).unwrap();
        let mut want = Image::new(w, h, Fmt::U8);
        if binary {
            smart_blend_binary(&mut want, &a, &b, 0.4).unwrap();
        } else {
            smart_blend(&mut want, &a, &b, 0.4).unwrap();
        }
        assert!(same_bits(&out, &want), "binary {binary} (no hole, the fill is the identity)");
    }
    let mut p = Image::new(w, h, Fmt::U8);
    let mut q = Image::new(w, h, Fmt::U8);
    smart_blend(&mut p, &a, &b, 0.4).unwrap();
    smart_blend_binary(&mut q, &a, &b, 0.4).unwrap();
    assert!(!same_bits(&p, &q), "translucent input: the variants differ");
    // no coverage: the plain blend
    let mut out = Image::new(w, h, Fmt::U8);
    blend_tail(&mut out, &a, &b, 0.4, None, SmartOpts { binary_alpha: true, is360: true }).unwrap();
    let mut want = Image::new(w, h, Fmt::U8);
    blend(&mut want, &a, &b, 0.4);
    assert!(same_bits(&out, &want));
}

#[test]
fn smart_finish_combines_inverts_and_fills() {
    let (w, h) = (16usize, 10usize);
    let mut rng = Rng(17);
    let img = opaque(Fmt::U8, w, h, &mut rng);
    let mut cov1 = Coverage::new(w, h, 255);
    let mut cov2 = Coverage::new(w, h, 255);
    // (4,4): both empty -> hole; (5,4): only cov1 empty -> kept; (6,4): partial in both -> hole (any non zero hole mask counts)
    cov1.data[4 * w + 4] = 0;
    cov2.data[4 * w + 4] = 0;
    cov1.data[4 * w + 5] = 0;
    cov1.data[4 * w + 6] = 100;
    cov2.data[4 * w + 6] = 30;
    let mut out = img.clone();
    smart_finish(&mut out, &mut cov1, &cov2, false).unwrap();
    assert_eq!(cov1.data[4 * w + 4], 255);
    assert_eq!(cov1.data[4 * w + 5], 0);
    assert_eq!(cov1.data[4 * w + 6], 255 - 100);
    let changed: Vec<usize> = (0..w * h).filter(|&i| px_u8(&out, i) != px_u8(&img, i)).collect();
    assert!(changed.iter().all(|&i| i == 4 * w + 4 || i == 4 * w + 6), "{changed:?}");
    assert!(changed.contains(&(4 * w + 4)));
    // a coverage of the wrong size
    let mut c = Coverage::new(w, h + 1, 0);
    assert!(matches!(smart_finish(&mut out, &mut c, &cov2, false), Err(FxError::Mismatch(_))));
    // the 360 flag with a hole is refused, without a hole accepted
    let mut full = Coverage::new(w, h, 255);
    let none = Coverage::new(w, h, 255);
    smart_finish(&mut out, &mut full, &none, true).unwrap();
    let mut some = Coverage::new(w, h, 0);
    assert!(matches!(smart_finish(&mut out, &mut some, &Coverage::new(w, h, 0), true), Err(FxError::Unsupported(_))));
}

#[test]
fn compositions_are_identical_for_one_and_four_threads() {
    let mut rng = Rng(18);
    let (w, h) = (64usize, 40usize);
    let a = image_from(Fmt::U8, w, h, &mut rng);
    let b = image_from(Fmt::U8, w, h, &mut rng);
    let mut fab = FlowField::new(w, h);
    let mut fba = FlowField::new(w, h);
    for i in 0..w * h {
        fab.data[i * 2] = 7.0 + (i % 5) as f32 * 0.25;
        fab.data[i * 2 + 1] = -2.5;
        fba.data[i * 2] = -6.0;
        fba.data[i * 2 + 1] = 1.5 - (i % 3) as f32 * 0.5;
    }
    type Run = fn(&Image, &Image, &FlowField, &FlowField) -> Image;
    let runs: [(&str, Run); 3] = [
        ("blend_smart", |a, b, fab, fba| {
            let mut out = Image::new(a.width, a.height, a.fmt());
            blend_smart(&mut out, a, fba, b, fab, 0.4, SmartOpts::default()).unwrap();
            out
        }),
        ("weighted_blend_smart", |a, b, fab, fba| {
            let mut out = Image::new(a.width, a.height, a.fmt());
            weighted_blend_smart(&mut out, a, fba, b, fab, 0.4, SmartOpts::default()).unwrap();
            out
        }),
        ("nearest_smart", |a, b, fab, _| {
            let mut out = Image::new(a.width, a.height, a.fmt());
            nearest_smart(&mut out, a, b, fab, 0.3, SmartOpts::default()).unwrap();
            out
        }),
    ];
    for (name, run) in runs {
        let one = in_pool(1, || run(&a, &b, &fab, &fba));
        let four = in_pool(4, || run(&a, &b, &fab, &fba));
        assert!(same_bits(&one, &four), "{name}");
        assert!(!same_bits(&one, &a), "{name} did something");
    }
}

// ------------------------------------------------------------------------------------------------
// flow dissolve modes 0, 5 and 6
// ------------------------------------------------------------------------------------------------

#[test]
fn flow_dissolve_prechecks_snap_and_modes() {
    let mut rng = Rng(19);
    let a = opaque(Fmt::U8, 16, 10, &mut rng);
    let b = opaque(Fmt::U8, 16, 10, &mut rng);
    let mut out = Image::new(16, 10, Fmt::U8);
    // without flows only the modes 0, 2 and 6 are accepted
    for mode in [1u32, 3, 4, 5, 7, 8] {
        assert!(matches!(flow_dissolve_inv(&mut out, &a, None, &b, None, 0.5, mode), Err(FxError::BadParam { .. })), "mode {mode}");
    }
    // the weight snap: outside [0.002, 0.998] the mode becomes 0, a copy of A for wa >= 0.5 else B
    let flow = FlowField::new(16, 10);
    flow_dissolve_inv(&mut out, &a, Some(&flow), &b, Some(&flow), 0.999, 5).unwrap();
    assert!(same_bits(&out, &a));
    flow_dissolve_inv(&mut out, &a, Some(&flow), &b, Some(&flow), 0.001, 5).unwrap();
    assert!(same_bits(&out, &b));
    flow_dissolve_inv(&mut out, &a, None, &b, None, 0.9995, 6).unwrap();
    assert!(same_bits(&out, &a));
    // modes the port does not implement are named
    assert!(matches!(flow_dissolve_inv(&mut out, &a, Some(&flow), &b, Some(&flow), 0.5, 3), Err(FxError::Unsupported(_))));
    // mode 5 needs both flows
    assert!(matches!(flow_dissolve_inv(&mut out, &a, Some(&flow), &b, None, 0.5, 5), Err(FxError::BadParam { .. })));
}

#[test]
fn flow_dissolve_mode_5_without_motion_is_the_plain_blend() {
    let mut rng = Rng(20);
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let a = opaque(fmt, 18, 12, &mut rng);
        let b = opaque(fmt, 18, 12, &mut rng);
        let flow = zero_flow(18, 12);
        let mut out = Image::new(18, 12, fmt);
        flow_dissolve_inv(&mut out, &a, Some(&flow), &b, Some(&flow), 0.35, 5).unwrap();
        let mut want = Image::new(18, 12, fmt);
        blend(&mut want, &a, &b, 1.0f32 - 0.35);
        assert!(same_bits(&out, &want), "{fmt:?}");
    }
}

#[test]
fn flow_dissolve_mode_6_sharpens_only_where_the_two_frames_differ() {
    let mut rng = Rng(22);
    // identical frames: the change mask is empty, the sharpened copies are discarded, the blend of equal frames is the frame
    let a = opaque(Fmt::U8, 24, 16, &mut rng);
    let mut out = Image::new(24, 16, Fmt::U8);
    flow_dissolve_inv(&mut out, &a, None, &a, None, 0.4, 6).unwrap();
    assert!(same_bits(&out, &a));
    // different frames in one block: pixels far from the block and from its blurred surroundings are blended plainly
    let (w, h) = (48usize, 16usize);
    let mut a = Image::new(w, h, Fmt::U8);
    let mut b = Image::new(w, h, Fmt::U8);
    if let (Samples::U8(x), Samples::U8(y)) = (&mut a.samples, &mut b.samples) {
        for i in 0..w * h {
            x[i * 4..i * 4 + 4].copy_from_slice(&[255, 100, 100, 100]);
            y[i * 4..i * 4 + 4].copy_from_slice(&[255, 100, 100, 100]);
        }
        for yy in 4..12 {
            for xx in 4..12 {
                y[(yy * w + xx) * 4 + 1..(yy * w + xx) * 4 + 4].copy_from_slice(&[240, 240, 240]);
            }
        }
    }
    let mut out = Image::new(w, h, Fmt::U8);
    flow_dissolve_inv(&mut out, &a, None, &b, None, 0.5, 6).unwrap();
    // far away (column 40): both frames are flat 100 and blend to 100
    assert_eq!(px_u8(&out, 8 * w + 40), [255, 100, 100, 100]);
    // inside the block the two frames blend towards the middle (with an unsharp overshoot at most)
    let mid = px_u8(&out, 8 * w + 8);
    assert!(mid[1] > 100, "{mid:?}");
}

// ------------------------------------------------------------------------------------------------
// ground truth sanity: a moving scene with an occluding square
// ------------------------------------------------------------------------------------------------

mod scene {
    //! A synthetic clip with a KNOWN motion: a smooth textured background that pans by `BG` pixels per frame
    //! and an opaque textured square that moves by `FG` pixels per frame over it. Everything is evaluated
    //! in closed form at integer shifts, so the in between frame at `t = 0.5` is known exactly.

    use crate::flow::FlowField;
    use crate::image::{Fmt, Image, Samples};

    pub const W: usize = 96;
    pub const H: usize = 56;
    pub const BG: i32 = 10;
    pub const FG: i32 = 20;
    pub const SQ: i32 = 24;
    pub const X0: i32 = 22;
    pub const Y0: i32 = 14;

    fn wave(c: usize, x: f64, y: f64, scale: f64) -> f64 {
        const WAVES: [(f64, f64, f64); 4] = [(26.0, 20.0, 0.30), (19.0, 110.0, 0.28), (13.0, 60.0, 0.22), (34.0, 160.0, 0.20)];
        let mut s = 0.0;
        for (i, &(lambda, deg, amp)) in WAVES.iter().enumerate() {
            let th = deg.to_radians();
            let phase = 1.7 * i as f64 + 2.3 * c as f64;
            s += amp * (std::f64::consts::TAU * ((x * th.cos() + y * th.sin()) / (lambda * scale)) + phase).sin();
        }
        0.5 + 0.5 * s
    }

    /// Colour of the scene at pixel `(x, y)` at time `t` (0 = frame A, 1 = frame B), `t` such that all shifts are integers.
    pub fn rgb(x: i32, y: i32, t: f64) -> [f64; 3] {
        let sq_x = X0 + (FG as f64 * t) as i32;
        let (u, v) = (x - sq_x, y - Y0);
        if u >= 0 && u < SQ && v >= 0 && v < SQ {
            // the square has its own texture, attached to the square
            return [wave(0, u as f64, v as f64, 0.5), wave(1, u as f64, v as f64, 0.5), wave(2, u as f64, v as f64, 0.5)];
        }
        let bx = x as f64 - (BG as f64 * t).floor();
        [wave(0, bx, y as f64, 1.0), wave(1, bx, y as f64, 1.0), wave(2, bx, y as f64, 1.0)]
    }

    pub fn frame(fmt: Fmt, t: f64) -> Image {
        let mut img = Image::new(W, H, fmt);
        for y in 0..H {
            for x in 0..W {
                let c = rgb(x as i32, y as i32, t);
                match &mut img.samples {
                    Samples::U8(v) => {
                        let i = (y * W + x) * 4;
                        v[i] = 255;
                        for k in 0..3 {
                            v[i + 1 + k] = (c[k] * 255.0 + 0.5) as u8;
                        }
                    }
                    Samples::U16(v) => {
                        let i = (y * W + x) * 4;
                        v[i] = 65535;
                        for k in 0..3 {
                            v[i + 1 + k] = (c[k] * 65535.0 + 0.5) as u16;
                        }
                    }
                    Samples::F32(v) => {
                        let i = (y * W + x) * 4;
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

    /// `F_AB` on the grid of A (`A(x) ~ B(x + F_AB(x))`) and `F_BA` on the grid of B (`B(x) ~ A(x + F_BA(x))`).
    pub fn exact_flows() -> (FlowField, FlowField) {
        let mut ab = FlowField::new(W, H);
        let mut ba = FlowField::new(W, H);
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                let in_a = x >= X0 && x < X0 + SQ && y >= Y0 && y < Y0 + SQ;
                let in_b = x >= X0 + FG && x < X0 + FG + SQ && y >= Y0 && y < Y0 + SQ;
                ab.set(x as usize, y as usize, [if in_a { FG as f32 } else { BG as f32 }, 0.0]);
                ba.set(x as usize, y as usize, [if in_b { -FG as f32 } else { -BG as f32 }, 0.0]);
            }
        }
        (ab, ba)
    }
}

/// Mean absolute error over RGB (in levels of 255) of `img` against the truth frame, over the pixels selected by `region`.
fn mean_error(img: &Image, truth: &Image, region: impl Fn(usize, usize) -> bool) -> f64 {
    let (a, b) = (as_f64(img), as_f64(truth));
    let scale = 255.0 / f64::from(img.fmt().white());
    let (mut sum, mut n) = (0.0, 0usize);
    for y in 0..img.height {
        for x in 0..img.width {
            if region(x, y) {
                for c in 1..4 {
                    let i = (y * img.width + x) * 4 + c;
                    sum += (a[i] - b[i]).abs() * scale;
                }
                n += 3;
            }
        }
    }
    sum / n as f64
}

#[test]
fn ground_truth_blend_smart_beats_the_plain_blend_in_border_and_occlusion_regions() {
    use scene::*;
    let (a, b, truth) = (frame(Fmt::U8, 0.0), frame(Fmt::U8, 1.0), frame(Fmt::U8, 0.5));
    let (f_ab, f_ba) = exact_flows();
    let wa = 0.5f32;
    // the candidates: plain blend of the unwarped frames, hold edge warps + plain blend (warp
    // code 0), Smart Blend (code 1)
    let mut plain = Image::new(W, H, Fmt::U8);
    blend(&mut plain, &a, &b, 1.0 - wa);
    let mut hold = Image::new(W, H, Fmt::U8);
    {
        let (mut ta, mut tb) = (Image::new(W, H, Fmt::U8), Image::new(W, H, Fmt::U8));
        inv_warp(&mut ta, &a, &f_ba, 1.0 - wa, WarpOpts::default()).unwrap();
        inv_warp(&mut tb, &b, &f_ab, wa, WarpOpts::default()).unwrap();
        blend(&mut hold, &ta, &tb, 1.0 - wa);
    }
    let mut smart = Image::new(W, H, Fmt::U8);
    blend_smart(&mut smart, &a, &f_ba, &b, &f_ab, wa, SmartOpts::default()).unwrap();
    let mut weighted = Image::new(W, H, Fmt::U8);
    weighted_blend_smart(&mut weighted, &a, &f_ba, &b, &f_ab, wa, SmartOpts::default()).unwrap();

    // the border strips: the background pans by 10 px per frame, 5 px at the midpoint
    let strip = |x: usize, _y: usize| x < 5 || x >= W - 5;
    // the occlusion band: the horizontal span of the square at A and at B (the pixels where the truth differs from the average)
    let band = |x: usize, y: usize| {
        (X0 as usize..(X0 + FG + SQ) as usize).contains(&x) && (Y0 as usize..(Y0 + SQ) as usize).contains(&y)
    };
    let all = |_x: usize, _y: usize| true;
    let names = ["plain", "hold", "smart", "weighted"];
    let imgs = [&plain, &hold, &smart, &weighted];
    let mut table = Vec::new();
    for (n, img) in names.iter().zip(imgs) {
        table.push((*n, mean_error(img, &truth, strip), mean_error(img, &truth, band), mean_error(img, &truth, all)));
    }
    eprintln!("[f1 ground truth] mean abs error in levels (strip, occlusion band, whole frame): {table:?}");
    let get = |n: &str| table.iter().find(|t| t.0 == n).unwrap().clone();
    let (plain_e, hold_e, smart_e) = (get("plain"), get("hold"), get("smart"));
    // border strips: Smart Blend takes the valid frame and is exact, the others smear or ghost
    assert!(smart_e.1 < 0.6, "smart strip error {}", smart_e.1);
    assert!(smart_e.1 * 20.0 < hold_e.1 && smart_e.1 * 20.0 < plain_e.1, "strip: smart {} hold {} plain {}", smart_e.1, hold_e.1, plain_e.1);
    // occlusion band: closer than the plain blend by a margin
    assert!(smart_e.2 < 0.75 * plain_e.2, "band: smart {} plain {}", smart_e.2, plain_e.2);
    // whole frame
    assert!(smart_e.3 < 0.3 * plain_e.3, "whole: smart {} plain {}", smart_e.3, plain_e.3);
    // Smart Blend leaves no pixel unfilled: alpha is 255 everywhere
    let Samples::U8(v) = &smart.samples else { unreachable!() };
    assert!(v.chunks(4).all(|p| p[0] == 255));
    let Samples::U8(v) = &weighted.samples else { unreachable!() };
    assert!(v.chunks(4).all(|p| p[0] == 255));
}

#[test]
fn ground_truth_both_frames_missing_leaves_nothing_unfilled_and_plausible() {
    use scene::*;
    // flows of the same sign (an occlusion artefact): both warps sample to the right and miss the right strip together
    let (a, b) = (frame(Fmt::U8, 0.0), frame(Fmt::U8, 1.0));
    let mut flow = FlowField::new(W, H);
    for i in 0..W * H {
        flow.data[i * 2] = 14.0;
    }
    let mut out = Image::new(W, H, Fmt::U8);
    blend_smart(&mut out, &a, &flow, &b, &flow, 0.5, SmartOpts::default()).unwrap();
    // the warps alone: where both coverages are 0 there is no data
    let wopts = WarpOpts { hold: false, is360: false, mode: 1 };
    let (mut ta, mut tb) = (Image::new(W, H, Fmt::U8), Image::new(W, H, Fmt::U8));
    let c1 = inv_warp_coverage(&mut ta, &a, &flow, 0.5, wopts).unwrap();
    let c2 = inv_warp_coverage(&mut tb, &b, &flow, 0.5, wopts).unwrap();
    let mut union = c1.clone();
    union.combine_max(&c2);
    let missing: Vec<usize> = (0..W * H).filter(|&i| union.data[i] != 255).collect();
    assert!(missing.len() >= 6 * H, "the right strip is missing in both frames: {} pixels", missing.len());
    let Samples::U8(v) = &out.samples else { unreachable!() };
    assert!(v.chunks(4).all(|p| p[0] == 255), "an unfilled (transparent) pixel is left");
    // the filled colours lie within the range of the known ones in each channel
    let known: Vec<usize> = (0..W * H).filter(|&i| union.data[i] == 255).collect();
    for c in 1..4 {
        let lo = known.iter().map(|&i| v[i * 4 + c]).min().unwrap();
        let hi = known.iter().map(|&i| v[i * 4 + c]).max().unwrap();
        for &i in &missing {
            assert!(v[i * 4 + c] >= lo && v[i * 4 + c] <= hi, "pixel {i} channel {c}: {} outside {lo}..{hi}", v[i * 4 + c]);
        }
    }
}

#[test]
fn ground_truth_with_solved_flows_smart_blend_beats_the_plain_blend() {
    use scene::*;
    // the same scene with the flows estimated by the Lucas-Kanade solver instead of the exact ones
    let (a, b, truth) = (frame(Fmt::U8, 0.0), frame(Fmt::U8, 1.0), frame(Fmt::U8, 0.5));
    let pair = solve_pair(&a, &b, &FlowSettings::new(3, true, 0.7)).unwrap();
    let wa = 0.5f32;
    let mut plain = Image::new(W, H, Fmt::U8);
    blend(&mut plain, &a, &b, 1.0 - wa);
    let mut smart = Image::new(W, H, Fmt::U8);
    blend_smart(&mut smart, &a, &pair.f_ba, &b, &pair.f_ab, wa, SmartOpts::default()).unwrap();
    let strip = |x: usize, _y: usize| x < 5 || x >= W - 5;
    let all = |_x: usize, _y: usize| true;
    let (ps, ss) = (mean_error(&plain, &truth, strip), mean_error(&smart, &truth, strip));
    let (pa, sa) = (mean_error(&plain, &truth, all), mean_error(&smart, &truth, all));
    eprintln!("[f1 ground truth, solved flows] strip: plain {ps:.2} smart {ss:.2}; whole frame: plain {pa:.2} smart {sa:.2}");
    assert!(ss < ps, "strip: smart {ss} plain {ps}");
    assert!(sa < pa, "whole frame: smart {sa} plain {pa}");
    let Samples::U8(v) = &smart.samples else { unreachable!() };
    assert!(v.chunks(4).all(|p| p[0] == 255), "an unfilled pixel is left");
}

#[test]
fn smart_blend_scene_holds_at_every_depth() {
    use scene::*;
    // the exact flows give the same story at 16 bpc and in float: smart is exact in the border strips
    for fmt in [Fmt::U16, Fmt::F32] {
        let (a, b, truth) = (frame(fmt, 0.0), frame(fmt, 1.0), frame(fmt, 0.5));
        let (f_ab, f_ba) = exact_flows();
        let mut smart = Image::new(W, H, fmt);
        blend_smart(&mut smart, &a, &f_ba, &b, &f_ab, 0.5, SmartOpts::default()).unwrap();
        let mut plain = Image::new(W, H, fmt);
        blend(&mut plain, &a, &b, 0.5);
        let strip = |x: usize, _y: usize| x < 5 || x >= W - 5;
        let (s, p) = (mean_error(&smart, &truth, strip), mean_error(&plain, &truth, strip));
        assert!(s < 0.6 && s * 20.0 < p, "{fmt:?}: smart {s} plain {p}");
    }
}

/// Timing probe (run with `--release -- --ignored --nocapture`): the Smart Blend composite and the whole Blend 1
/// composition at 1920x1080 u8 with a constant flow.
#[test]
#[ignore]
fn timing_probe_1080p() {
    let mut rng = Rng(6);
    let (w, h) = (1920usize, 1080usize);
    let a = image_from(Fmt::U8, w, h, &mut rng);
    let b = image_from(Fmt::U8, w, h, &mut rng);
    let mut out = Image::new(w, h, Fmt::U8);
    let t = std::time::Instant::now();
    smart_blend(&mut out, &a, &b, 0.4).unwrap();
    println!("[f1] smart_blend 1920x1080 u8: {:?}", t.elapsed());
    let mut flow = FlowField::new(w, h);
    for i in 0..w * h {
        flow.data[i * 2] = 12.0;
        flow.data[i * 2 + 1] = -5.0;
    }
    let t = std::time::Instant::now();
    blend_smart(&mut out, &a, &flow, &b, &flow, 0.4, SmartOpts::default()).unwrap();
    println!("[f1] blend_smart 1920x1080 u8 (two warps, composite, hole fill of the borders): {:?}", t.elapsed());
    let t = std::time::Instant::now();
    weighted_blend_smart(&mut out, &a, &flow, &b, &flow, 0.4, SmartOpts::default()).unwrap();
    println!("[f1] weighted_blend_smart 1920x1080 u8: {:?}", t.elapsed());
}
