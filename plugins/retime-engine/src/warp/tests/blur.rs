//! Tests of the motion blur primitive and of Blur Compensation (child of `warp::tests`, shares its helpers).

use super::*;
use crate::warp::blurcomp::{blur_amount, clear_rgb_where_transparent, compensate, motion_blur};

struct BlurCase {
    src: Image,
    flow: FlowField,
    amount: f32,
    shutter: f32,
    hiq: bool,
    out: Image,
}

fn load_blur_cases() -> Vec<BlurCase> {
    let blob = include_bytes!("../../../tests/fixtures/blur_cases.bin");
    assert_eq!(&blob[..4], b"BLR1");
    let mut r = Reader { b: blob, p: 4 };
    let n = r.u32();
    let mut cases = Vec::new();
    for _ in 0..n {
        let depth = r.u32();
        let (w, h, fw, fh) = (r.u32() as usize, r.u32() as usize, r.u32() as usize, r.u32() as usize);
        let amount = r.f32();
        let shutter = r.f32();
        let hiq = r.u32() != 0;
        let nb = w * h * 4 * depth_bytes(depth);
        let src = samples_from(depth, r.bytes(nb), w, h);
        let fb = r.bytes(fw * fh * 2 * 4);
        let flow = FlowField {
            width: fw,
            height: fh,
            data: fb.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        };
        let out = samples_from(depth, r.bytes(nb), w, h);
        cases.push(BlurCase { src, flow, amount, shutter, hiq, out });
    }
    assert_eq!(r.p, blob.len());
    cases
}

/// Like `same_samples` but two NaNs of any payload are equal (the model and the port both produce
/// `inf * 0` for a pixel without a single tap, whose payload is a hardware matter).
fn same_samples_nan_ok(a: &Image, b: &Image) -> Option<usize> {
    match (&a.samples, &b.samples) {
        (Samples::F32(x), Samples::F32(y)) => {
            x.iter().zip(y).position(|(p, q)| !(p.to_bits() == q.to_bits() || (p.is_nan() && q.is_nan())))
        }
        _ => same_samples(a, b),
    }
}

#[test]
fn blur_model_fixtures_are_reproduced_bit_for_bit() {
    let cases = load_blur_cases();
    assert_eq!(cases.len(), 42);
    for (i, c) in cases.iter().enumerate() {
        let mut out = zeros_like(&c.src);
        motion_blur(&mut out, &c.src, &c.flow, c.amount, c.hiq, c.shutter, false).unwrap();
        if let Some(pos) = same_samples_nan_ok(&out, &c.out) {
            panic!(
                "blur case {i} ({:?}, {}x{}, flow {}x{}, amount {}, shutter {}, hiq {}): first difference at sample {pos}",
                c.src.fmt(),
                c.src.width,
                c.src.height,
                c.flow.width,
                c.flow.height,
                c.amount,
                c.shutter,
                c.hiq
            );
        }
    }
}

#[test]
fn blur_thread_count_does_not_change_the_output() {
    for (i, c) in load_blur_cases().iter().enumerate() {
        let run = |threads: usize| {
            in_pool(threads, || {
                let mut out = zeros_like(&c.src);
                motion_blur(&mut out, &c.src, &c.flow, c.amount, c.hiq, c.shutter, false).unwrap();
                out
            })
        };
        let (a, b) = (run(1), run(4));
        assert!(same_samples_nan_ok(&a, &b).is_none(), "blur case {i}");
    }
}

/// 1-D ramp image: every channel of pixel x holds x (A, R, G, B all equal), 12 x 4.
fn ramp_f32() -> Image {
    img_f32(12, 4, |x, _y, _c| x as f32)
}

#[test]
fn bilinear_blur_of_a_ramp_goldens() {
    // flow dx = 4 (same size as the image), amount 1: kx = (0.5 / 1) * 1 = 0.5, vx = 2, d2 = 4, dist = 2,
    // n = trunc(3.0) = 3, step = 2 / 2 = 1: forward taps x, x+1, x+2 and backward taps x-1, x-2.
    let src = ramp_f32();
    let flow = const_flow(12, 4, 4.0, 0.0);
    let mut out = zeros_like(&src);
    motion_blur(&mut out, &src, &flow, 1.0, true, 0.5, false).unwrap();
    let inv5 = 1.0f32 / 5.0;
    for x in 2..10 {
        // interior: five taps, sum 5x, (1/5) * 5x
        assert_eq!(px_f32(&out, x, 1), [inv5 * (5.0 * x as f32); 4], "x = {x}");
    }
    // x = 0: taps 0, 1, 2 inside, -1 and -2 outside: count 3, sum 3
    assert_eq!(px_f32(&out, 0, 1), [(1.0f32 / 3.0) * 3.0; 4]);
    // x = 1: forward 1, 2, 3 and backward 0 inside, -1 outside: count 4, sum 6
    assert_eq!(px_f32(&out, 1, 1), [(1.0f32 / 4.0) * 6.0; 4]);
    // x = 11: forward 11 inside, 12 and 13 outside; backward 10 and 9 inside: count 3, sum 30
    assert_eq!(px_f32(&out, 11, 1), [(1.0f32 / 3.0) * 30.0; 4]);
}

#[test]
fn nearest_blur_counts_every_backward_tap() {
    // same setup, draft quality. Taps sit at trunc(x + 0.5 + t): the forward taps x, x+1, x+2 count when
    // inside; the backward taps x-1, x-2 count ALWAYS (and a position of -0.5 truncates to 0, inside).
    let src = ramp_f32();
    let flow = const_flow(12, 4, 4.0, 0.0);
    let mut out = zeros_like(&src);
    motion_blur(&mut out, &src, &flow, 1.0, false, 0.5, false).unwrap();
    let inv5 = 1.0f32 / 5.0;
    for x in 2..10 {
        assert_eq!(px_f32(&out, x, 1), [inv5 * (5.0 * x as f32); 4], "x = {x}");
    }
    // x = 0: forward 0.5, 1.5, 2.5 -> 0, 1, 2 (count 3, sum 3); backward -0.5 -> 0 inside (adds 0),
    // -1.5 -> -1 outside but counted: count 5, sum 3
    assert_eq!(px_f32(&out, 0, 1), [inv5 * 3.0; 4]);
    // x = 1: forward 1, 2, 3 (count 3, sum 6); backward 0.5 -> 0 inside, -0.5 -> 0 inside: count 5, sum 6
    assert_eq!(px_f32(&out, 1, 1), [inv5 * 6.0; 4]);
    // x = 11: forward 11 (inside), 12 and 13 outside: count 1, sum 11; backward 10, 9: count 3, sum 30
    // wait: the backward loop runs n - 1 = 2 taps, both counted: count 1 + 2 = 3, sum 11 + 10 + 9 = 30
    assert_eq!(px_f32(&out, 11, 1), [(1.0f32 / 3.0) * 30.0; 4]);
}

#[test]
fn nearest_blur_without_motion_averages_three_copies() {
    // no small vector shortcut: n = 2, step 0: two forward taps and one backward tap on the same pixel
    let src = img_f32(6, 5, |x, y, c| 0.1 * (x + 2 * y + c) as f32 + 0.05);
    let flow = FlowField::new(6, 5);
    let mut out = zeros_like(&src);
    motion_blur(&mut out, &src, &flow, 1.0, false, 0.5, false).unwrap();
    let inv3 = 1.0f32 / 3.0;
    for y in 0..5 {
        for x in 0..6 {
            let p = px_f32(&src, x, y);
            for c in 0..4 {
                // 0 + p + p + p, then (1/3) * sum
                let sum = ((0.0f32 + p[c]) + p[c]) + p[c];
                assert_eq!(px_f32(&out, x, y)[c].to_bits(), (inv3 * sum).to_bits());
            }
        }
    }
    // the bilinear leaf copies the pixel for a zero vector
    motion_blur(&mut out, &src, &flow, 1.0, true, 0.5, false).unwrap();
    assert_eq!(out, src);
}

#[test]
fn integer_blur_stores_round_and_average() {
    // u8 ramp 0, 20, ..., 220 with horizontal motion: interior pixels average five taps exactly
    let src = img_u8(12, 3, |x, _y, _c| (x * 20) as u8);
    let flow = const_flow(12, 3, 4.0, 0.0);
    let mut out = zeros_like(&src);
    motion_blur(&mut out, &src, &flow, 1.0, true, 0.5, false).unwrap();
    for x in 2..10 {
        let i = (12 + x) * 4;
        assert_eq!(u8s(&out)[i], (x * 20) as u8, "x = {x}");
    }
    // x = 0: taps 0, 20, 40 (count 3): 20
    assert_eq!(u8s(&out)[12 * 4], 20);
    // x = 11: taps 220, 200, 180 (count 3): 200
    assert_eq!(u8s(&out)[(12 + 11) * 4], 200);
    // 16 bit, same shape of data
    let src16 = img_u16(12, 3, |x, _y, _c| (x * 5000) as u16);
    let mut out16 = zeros_like(&src16);
    motion_blur(&mut out16, &src16, &flow, 1.0, true, 0.5, false).unwrap();
    assert_eq!(u16s(&out16)[(12 + 5) * 4], 25000);
    assert_eq!(u16s(&out16)[12 * 4], 5000);
}

#[test]
fn flow_resolution_scales_the_blur_length() {
    // a flow with half the columns: rx = 5 / 11 and kx = (0.5 / rx) * amount, so the vector in image pixels is
    // dx / rx * 0.5 * amount; a flow value of 4 * rx in a flow of rx times the size is the same as 4 in the full one
    let src = ramp_f32();
    let full = const_flow(12, 4, 4.0, 0.0);
    let half = const_flow(6, 4, 4.0 * (5.0 / 11.0), 0.0);
    let (mut a, mut b) = (zeros_like(&src), zeros_like(&src));
    motion_blur(&mut a, &src, &full, 1.0, true, 0.5, false).unwrap();
    motion_blur(&mut b, &src, &half, 1.0, true, 0.5, false).unwrap();
    for x in 0..12 {
        let (p, q) = (px_f32(&a, x, 1)[0], px_f32(&b, x, 1)[0]);
        assert!((p - q).abs() < 1e-4, "x = {x}: {p} vs {q}");
    }
}

#[test]
fn blur_amount_call_site_goldens() {
    // (|s0 - s1| - 1.0) * blur, only when |s0 - s1| > 1e-6 and blur > 0.005
    assert_eq!(blur_amount(2.0, 0.0, 0.5), Some(0.5));
    assert_eq!(blur_amount(0.0, 2.0, 0.5), Some(0.5));
    assert_eq!(blur_amount(10.5, 10.0, 1.0), Some(-0.5)); // slow motion: negative
    assert_eq!(blur_amount(4.0, 3.0, 1.0), Some(0.0)); // exactly one frame per frame
    assert_eq!(blur_amount(3.0, 3.0, 1.0), None);
    assert_eq!(blur_amount(3.0, 3.0 + 1e-7, 1.0), None);
    assert!(blur_amount(3.0, 3.0 + 3e-6, 1.0).is_some());
    assert_eq!(blur_amount(5.0, 0.0, 0.0), None);
    // f32(0.005) is below the double 0.005 used by the comparison
    assert_eq!(blur_amount(5.0, 0.0, 0.005), None);
    assert!(blur_amount(5.0, 0.0, 0.0051).is_some());
    assert_eq!(blur_amount(f32::NAN, 0.0, 1.0), None);
}

#[test]
fn compensation_clamps_the_amount_and_skips_tiny_amounts() {
    let src = img_u8(12, 6, |x, y, c| (10 + x * 9 + y * 13 + c) as u8);
    let mut flow = FlowField::new(12, 6);
    for y in 0..6 {
        for x in 0..12 {
            flow.set(x, y, [((x + y) % 5) as f32 - 2.0, 1.0]);
        }
    }
    // amounts above 20 behave as 20, below -10 as -10
    let run = |amount: f32| {
        let mut img = src.clone();
        compensate(&mut img, &flow, amount, true).unwrap();
        img
    };
    assert_eq!(run(20.0), run(95.0));
    assert_eq!(run(-10.0), run(-73.0));
    assert_ne!(run(20.0), run(19.0));
    // |amount| < 1e-6 leaves the image alone
    assert_eq!(run(5e-7), src);
    assert_eq!(run(-5e-7), src);
    // |amount| >= 1e-6 passes the early out, but the vectors (|dx| * 0.5 * 2e-6) are far below the 1e-6 squared length
    // of the bilinear leaf, which copies the pixel; the opaque test image is untouched by the post step
    assert_eq!(run(2e-6), src);
}

#[test]
fn positive_compensation_clears_the_colour_of_transparent_pixels() {
    // pixels with alpha 0 but non zero colour are cleared by the post step. A zero flow makes the bilinear leaf
    // copy every pixel, so only the post step acts.
    let (w, h) = (12, 8);
    let mut v = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            if x < 8 {
                v[i..i + 4].copy_from_slice(&[200, 100, 50, 25]);
            } else {
                v[i..i + 4].copy_from_slice(&[0, 9, 8, 7]); // garbage colour under zero alpha
            }
        }
    }
    let src = Image { width: w, height: h, samples: Samples::U8(v) };
    let flow = FlowField::new(w, h);
    let mut img = src.clone();
    compensate(&mut img, &flow, 3.0, true).unwrap();
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            let p = &u8s(&img)[i..i + 4];
            if x < 8 {
                assert_eq!(p, &[200, 100, 50, 25]);
            } else {
                assert_eq!(p, &[0, 0, 0, 0], "({x},{y})");
            }
        }
    }
    // the post step on its own, float: `A == 0.0` so -0.0 counts, and A itself is left alone
    let mut f = img_f32(3, 1, |x, _, c| if x == 1 && c == 0 { 0.0 } else { 0.5 });
    match &mut f.samples {
        Samples::F32(v) => v[0] = -0.0,
        _ => unreachable!(),
    }
    clear_rgb_where_transparent(&mut f);
    assert_eq!(f32s(&f)[0].to_bits(), (-0.0f32).to_bits());
    assert_eq!(&f32s(&f)[1..4], &[0.0, 0.0, 0.0]);
    assert_eq!(px_f32(&f, 1, 0), [0.0, 0.0, 0.0, 0.0]);
    assert_eq!(px_f32(&f, 2, 0), [0.5, 0.5, 0.5, 0.5]);
    // 16 bit
    let mut s = img_u16(2, 1, |x, _, c| if x == 0 && c == 0 { 0 } else { 777 });
    clear_rgb_where_transparent(&mut s);
    assert_eq!(u16s(&s), &[0, 0, 0, 0, 777, 777, 777, 777]);
}

#[test]
fn negative_compensation_is_an_unsharp_mask_of_the_blurred_copy() {
    // compensate(img, flow, a < 0) = lerp(motion_blur(img, flow, 0.5), img, a): rebuild it from the public pieces
    let src = img_f32(14, 9, |x, y, c| if c == 0 { 1.0 } else { 0.1 * ((x * 3 + y * 5 + c) % 7) as f32 });
    let mut flow = FlowField::new(14, 9);
    for y in 0..9 {
        for x in 0..14 {
            flow.set(x, y, [3.0 * ((x % 4) as f32 - 1.5), 2.0 * ((y % 3) as f32 - 1.0)]);
        }
    }
    for hiq in [true, false] {
        for amount in [-0.5f32, -2.0, -7.25] {
            let mut want = zeros_like(&src);
            motion_blur(&mut want, &src, &flow, 0.5, hiq, 0.5, false).unwrap();
            crate::blend::lerp(&mut want, &src, amount);
            let mut got = src.clone();
            compensate(&mut got, &flow, amount, hiq).unwrap();
            assert!(same_samples(&got, &want).is_none(), "hiq {hiq} amount {amount}");
        }
    }
}

#[test]
fn blur_compensation_entry_applies_the_call_site_condition() {
    let src = img_f32(10, 8, |x, y, c| 0.05 * (x + y + c) as f32 + 0.1);
    let flow = const_flow(10, 8, 6.0, -2.0);
    let mut img = src.clone();
    // not moving between the two positions: unchanged
    blur_compensation(&mut img, &flow, 5.0, 5.0, 1.0, true).unwrap();
    assert_eq!(img, src);
    // slider off
    blur_compensation(&mut img, &flow, 7.0, 5.0, 0.0, true).unwrap();
    assert_eq!(img, src);
    // active: equals the explicit chain
    let amount = blur_amount(7.0, 5.0, 0.8).unwrap();
    assert_eq!(amount, (2.0f32 - 1.0) * 0.8);
    let mut want = src.clone();
    compensate(&mut want, &flow, amount, true).unwrap();
    blur_compensation(&mut img, &flow, 7.0, 5.0, 0.8, true).unwrap();
    assert_eq!(img, want);
    assert_ne!(img, src);
    // 360 leaves are not reproduced
    let mut out = zeros_like(&src);
    assert!(matches!(motion_blur(&mut out, &src, &flow, 1.0, true, 0.5, true), Err(FxError::Unsupported(_))));
}

#[test]
fn blur_leaf_stores_follow_the_clamp_rule_not_the_pack_rule() {
    // maxps / minps based: a NaN or an out of int32 range value ends at 0 (indefinite integer clamped by
    // maxps), large values that fit an int32 end at the white
    assert_eq!(<u8 as Pix>::store_blur(f32::NAN), 0);
    assert_eq!(<u8 as Pix>::store_blur(1e6), 255);
    assert_eq!(<u8 as Pix>::store_blur(1e12), 0);
    assert_eq!(<u8 as Pix>::store_blur(-3.0), 0);
    assert_eq!(<u8 as Pix>::store_blur(254.5), 255);
    assert_eq!(<u8 as Pix>::store_blur(254.49), 254);
    assert_eq!(<u16 as Pix>::store_blur(65535.5), 65535);
    assert_eq!(<u16 as Pix>::store_blur(1e6), 65535);
    assert_eq!(<u16 as Pix>::store_blur(f32::NAN), 0);
    assert_eq!(<f32 as Pix>::store_blur(-3.5), -3.5);
}
