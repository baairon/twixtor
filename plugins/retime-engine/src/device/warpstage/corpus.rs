//! Test corpora of the composition stages and the simulated stages: the plans of the wrappers
//! run with the kernel transcriptions of [`super::rs`] in the work item simulator, so the
//! device-free tests and the device tests compare against the same CPU calls.

use super::blend::{blend_plan, lerp_plan, BlendPlan, LerpPlan};
use super::blurcomp::{blur_args, compensate_plan, CompPlan};
use super::rs::{self, Depth};
use super::warp::{warp_plan, WarpPlan};
use crate::device::kernels::prelude::rs as p;
use crate::device::selfcheck;
use crate::device::sim::{self, Global, Rng};
use crate::flow::FlowField;
use crate::image::{Fmt, Image, Samples};
use crate::warp::{blur_amount, Coverage, WarpOpts};

pub const FMTS: [Fmt; 3] = [Fmt::U8, Fmt::U16, Fmt::F32];

/// Image sizes: every small shape class from 2x2 to 17x16 and a few larger ones.
pub fn sizes() -> Vec<(usize, usize)> {
    vec![(2, 2), (3, 2), (2, 3), (4, 4), (5, 3), (7, 7), (8, 5), (9, 16), (16, 9), (17, 16), (33, 20), (64, 37)]
}

fn unit(rng: &mut Rng) -> f32 {
    (rng.next_u32() >> 8) as f32 / (1u32 << 24) as f32
}

/// An f32 sample: tame (`wild == false`: 0..1 with exact 0, 1 and ties) or anything adversarial.
fn f32_sample(rng: &mut Rng, wild: bool, specials: &[f32]) -> f32 {
    let r = rng.below(16);
    if wild && r < 3 {
        return specials[rng.below(specials.len() as u64) as usize];
    }
    match r {
        3 => 0.0,
        4 => 1.0,
        5 => 0.5,
        6 => -0.0,
        7 => unit(rng) * 2.0 - 0.5,
        _ => unit(rng),
    }
}

/// A four channel image. `class` 0: tame values; 1: adversarial (f32 specials, integer extremes);
/// 2: alpha only 0 or white (for the binary Smart Blend).
pub fn image(rng: &mut Rng, w: usize, h: usize, fmt: Fmt, class: u32) -> Image {
    let n = w * h * 4;
    let specials = selfcheck::f32_specials();
    let mut img = Image::new(w, h, fmt);
    for i in 0..n {
        let alpha = i % 4 == 0;
        let r = rng.below(8);
        match &mut img.samples {
            Samples::U8(v) => {
                v[i] = if (class == 2 && alpha) || (class == 1 && r == 0) {
                    if rng.below(2) == 0 { 0 } else { 255 }
                } else {
                    rng.next_u32() as u8
                }
            }
            Samples::U16(v) => {
                v[i] = if (class == 2 && alpha) || (class == 1 && r == 0) {
                    if rng.below(2) == 0 { 0 } else { 65535 }
                } else {
                    rng.next_u32() as u16
                }
            }
            Samples::F32(v) => {
                v[i] = if class == 2 && alpha {
                    if rng.below(2) == 0 { 0.0 } else { 1.0 }
                } else {
                    f32_sample(rng, class == 1, &specials)
                }
            }
        }
    }
    img
}

/// A flow of `w x h` vectors of up to `scale` pixels; `wild` mixes in exact integers, halves,
/// -0.0, and (when `nonfinite`) NaN and infinities.
pub fn flow(rng: &mut Rng, w: usize, h: usize, scale: f32, wild: bool, nonfinite: bool) -> FlowField {
    let mut f = FlowField::new(w, h);
    for v in f.data.iter_mut() {
        let r = rng.below(32);
        *v = if wild && r == 0 {
            -0.0
        } else if wild && r == 1 {
            (rng.below(9) as f32 - 4.0) * 0.5
        } else if wild && r == 2 {
            rng.below(2 * (scale as u64).max(1) + 1) as f32 - scale.trunc()
        } else if nonfinite && r == 3 {
            [f32::NAN, f32::INFINITY, f32::NEG_INFINITY][rng.below(3) as usize]
        } else {
            (unit(rng) * 2.0 - 1.0) * scale
        };
    }
    f
}

/// The bits of every sample (f32 as its bit pattern).
pub fn bits(img: &Image) -> Vec<u64> {
    match &img.samples {
        Samples::U8(v) => v.iter().map(|x| *x as u64).collect(),
        Samples::U16(v) => v.iter().map(|x| *x as u64).collect(),
        Samples::F32(v) => v.iter().map(|x| x.to_bits() as u64).collect(),
    }
}

/// The first sample where the bits differ, with both values. Two NaNs count as equal whatever
/// their payloads: neither the devices nor the Rust compiler fix the payload of a NaN result (the
/// compiler may swap the operands of a commutative operation), and a NaN output is never accepted
/// from a device (`StageError::NanOutput`); NaN positions must agree.
pub fn first_diff(want: &Image, got: &Image) -> Option<(usize, String, String)> {
    assert_eq!((want.width, want.height, want.fmt()), (got.width, got.height, got.fmt()));
    let (a, b) = (bits(want), bits(got));
    let nan = |v: u64| want.fmt() == Fmt::F32 && f32::from_bits(v as u32).is_nan();
    a.iter().zip(&b).position(|(x, y)| x != y && !(nan(*x) && nan(*y))).map(|i| (i, format!("{:#x}", a[i]), format!("{:#x}", b[i])))
}

// ---- simulated stages ---------------------------------------------------------------------------

fn run_1d<T: Copy>(items: usize, len: usize, fill: T, seed: u64, mut item: impl FnMut(usize, &mut Global<T>)) -> Vec<T> {
    let mut g = Global::new(len, fill);
    sim::launch_1d(items, seed, |i| item(i, &mut g));
    assert_eq!(g.written(), len, "every element is written once");
    g.into_vec()
}

fn with_samples<R>(img: &Image, u8f: impl FnOnce(&[u8]) -> R, u16f: impl FnOnce(&[u16]) -> R, f32f: impl FnOnce(&[f32]) -> R) -> R {
    match &img.samples {
        Samples::U8(v) => u8f(v),
        Samples::U16(v) => u16f(v),
        Samples::F32(v) => f32f(v),
    }
}

fn like(img: &Image, samples: Samples) -> Image {
    Image { width: img.width, height: img.height, samples }
}

/// `blend_enqueue` with the transcriptions.
pub fn sim_blend(a: &Image, b: &Image, w: f32, seed: u64) -> Image {
    match blend_plan(w) {
        BlendPlan::CopyA => a.clone(),
        BlendPlan::CopyB => b.clone(),
        BlendPlan::Extrapolate(w2) => {
            let mut out = a.clone();
            sim_lerp(&mut out, b, w2, seed);
            out
        }
        BlendPlan::Mix => {
            let n = a.sample_count();
            let nn = n as u32;
            let s = match (&a.samples, &b.samples) {
                (Samples::U8(x), Samples::U8(y)) => Samples::U8(run_1d(n, n, 0, seed, |i, g| rs::bl_blend_int_item(i, x, y, g, w, nn, p::trunc_u8_f))),
                (Samples::U16(x), Samples::U16(y)) => Samples::U16(run_1d(n, n, 0, seed, |i, g| rs::bl_blend_int_item(i, x, y, g, w, nn, p::trunc_u16_f))),
                (Samples::F32(x), Samples::F32(y)) => Samples::F32(run_1d(n, n, 0.0, seed, |i, g| rs::bl_blend_f32_item(i, x, y, g, w, nn))),
                _ => unreachable!(),
            };
            like(a, s)
        }
    }
}

/// `lerp_enqueue` with the transcriptions.
pub fn sim_lerp(first: &mut Image, second: &Image, w: f32, seed: u64) {
    match lerp_plan(w) {
        LerpPlan::Keep => {}
        LerpPlan::CopySecond => *first = second.clone(),
        LerpPlan::Rows => {
            let px = first.width * first.height;
            let n = first.sample_count();
            let s = match (&first.samples, &second.samples) {
                (Samples::U8(f), Samples::U8(s)) => Samples::U8(run_1d(px, n, 0, seed, |i, g| rs::bl_lerp_int_item(i, f, s, g, w, px as u32, 255.0, p::trunc_u8_f))),
                (Samples::U16(f), Samples::U16(s)) => Samples::U16(run_1d(px, n, 0, seed, |i, g| rs::bl_lerp_int_item(i, f, s, g, w, px as u32, 65535.0, p::trunc_u16_f))),
                (Samples::F32(f), Samples::F32(s)) => Samples::F32(run_1d(px, n, 0.0, seed, |i, g| rs::bl_lerp_f32_item(i, f, s, g, w, px as u32))),
                _ => unreachable!(),
            };
            first.samples = s;
        }
    }
}

/// `weighted_blend_noflow_enqueue` with the transcriptions: `None` when the flag is raised.
pub fn sim_weighted_noflow(a: &Image, wa: f32, b: &Image, seed: u64) -> Option<Image> {
    let t = 1.0f32 - wa;
    match (&a.samples, &b.samples) {
        (Samples::F32(x), Samples::F32(y)) => {
            let n = x.len();
            let mut flag = 0;
            for v in [x, y] {
                sim::launch_1d(n, seed, |i| rs::bl_unsound_f32_item(i, v, &mut flag, n as u32));
            }
            let a1 = like(a, Samples::F32(run_1d(n, n, 0.0, seed, |i, g| rs::bl_plus_zero_f32_item(i, x, g, n as u32))));
            let b1 = like(b, Samples::F32(run_1d(n, n, 0.0, seed, |i, g| rs::bl_plus_zero_f32_item(i, y, g, n as u32))));
            let out = sim_blend(&a1, &b1, t, seed);
            if flag != 0 {
                None
            } else {
                Some(out)
            }
        }
        _ => Some(sim_blend(a, b, t, seed)),
    }
}

/// What a simulated stage gave: its outputs, or the refusal (`Err(0)` a plan refusal, else the
/// flag word).
pub type SimResult<T> = Result<T, i32>;

fn warp_depth<T: Depth>(src: &[T], flow: &[f32], a: &rs::WarpArgs, seed: u64) -> (Vec<T>, Vec<u8>, i32) {
    let (w, h) = (a.w as usize, a.h as usize);
    let mut out = Global::new(w * h * 4, T::ZERO);
    let cov_len = if a.want_cov != 0 { w * h } else { 1 };
    let mut cov = Global::new(cov_len, 0xffu8);
    let mut flag = 0;
    sim::launch_2d(w, h, seed, |x, y| rs::wp_inv_warp_item(x, y, src, flow, &mut out, &mut cov, &mut flag, a));
    assert_eq!(out.written(), w * h * 4);
    if a.want_cov != 0 {
        assert_eq!(cov.written(), w * h);
    }
    (out.into_vec(), cov.into_vec(), flag)
}

/// `inv_warp_enqueue` (with a coverage when `want_cov`) with the transcriptions.
pub fn sim_warp(src: &Image, flow: &FlowField, amount: f32, opts: WarpOpts, want_cov: bool, seed: u64) -> SimResult<(Image, Option<Coverage>)> {
    let plan = warp_plan((src.width, src.height), (flow.width, flow.height), flow.data.len(), amount, opts).map_err(|_| 0)?;
    let full = Coverage::new(src.width, src.height, 0xff);
    match plan {
        WarpPlan::Copy => Ok((src.clone(), want_cov.then_some(full))),
        WarpPlan::Launch(mut a) => {
            a.want_cov = want_cov as i32;
            let (s, cov, flag) = with_samples(
                src,
                |v| {
                    let (o, c, f) = warp_depth(v, &flow.data, &a, seed);
                    (Samples::U8(o), c, f)
                },
                |v| {
                    let (o, c, f) = warp_depth(v, &flow.data, &a, seed);
                    (Samples::U16(o), c, f)
                },
                |v| {
                    let (o, c, f) = warp_depth(v, &flow.data, &a, seed);
                    (Samples::F32(o), c, f)
                },
            );
            if flag != 0 {
                return Err(flag);
            }
            let cov = want_cov.then(|| Coverage { width: src.width, height: src.height, data: cov });
            Ok((like(src, s), cov))
        }
    }
}

fn blur_depth<T: Depth>(src: &[T], flow: &[f32], a: &rs::BlurArgs, rows: usize, seed: u64) -> (Vec<T>, i32) {
    let (w, h) = (a.w as usize, a.h as usize);
    let mut out = Global::new(w * h * 4, T::ZERO);
    let mut flag = 0;
    let mut y0 = 0;
    while y0 < h {
        let band = rows.min(h - y0);
        let mut b = *a;
        b.y0 = y0 as i32;
        sim::launch_2d(w, band, seed ^ y0 as u64, |x, y| rs::bc_motion_blur_item(x, y, src, flow, &mut out, &mut flag, &b));
        y0 += band;
    }
    assert_eq!(out.written(), w * h * 4);
    (out.into_vec(), flag)
}

/// `motion_blur_enqueue` with the transcriptions, in bands of `rows` rows.
#[allow(clippy::too_many_arguments)]
pub fn sim_motion_blur(src: &Image, flow: &FlowField, amount: f32, hiq: bool, shutter: f32, is360: bool, rows: usize, seed: u64) -> SimResult<Image> {
    let a = blur_args((src.width, src.height), (flow.width, flow.height), flow.data.len(), amount, hiq, shutter, is360).map_err(|_| 0)?;
    let (s, flag) = with_samples(
        src,
        |v| {
            let (o, f) = blur_depth(v, &flow.data, &a, rows, seed);
            (Samples::U8(o), f)
        },
        |v| {
            let (o, f) = blur_depth(v, &flow.data, &a, rows, seed);
            (Samples::U16(o), f)
        },
        |v| {
            let (o, f) = blur_depth(v, &flow.data, &a, rows, seed);
            (Samples::F32(o), f)
        },
    );
    if flag != 0 {
        return Err(flag);
    }
    Ok(like(src, s))
}

/// `clear_rgb_enqueue` with the transcriptions.
pub fn sim_clear_rgb(src: &Image, seed: u64) -> Image {
    let px = src.width * src.height;
    let n = src.sample_count();
    let s = with_samples(
        src,
        |v| Samples::U8(run_1d(px, n, 0, seed, |i, g| rs::bc_clear_rgb_item(i, v, g, px as u32))),
        |v| Samples::U16(run_1d(px, n, 0, seed, |i, g| rs::bc_clear_rgb_item(i, v, g, px as u32))),
        |v| Samples::F32(run_1d(px, n, 0.0, seed, |i, g| rs::bc_clear_rgb_item(i, v, g, px as u32))),
    );
    like(src, s)
}

/// `blur_compensation_wrap_enqueue` with the transcriptions.
#[allow(clippy::too_many_arguments)]
pub fn sim_compensation(img: &Image, flow: &FlowField, s0: f32, s1: f32, blur: f32, hiq: bool, is360: bool, rows: usize, seed: u64) -> SimResult<Image> {
    let Some(amount) = blur_amount(s0, s1, blur) else { return Ok(img.clone()) };
    match compensate_plan(amount) {
        CompPlan::Nothing => Ok(img.clone()),
        CompPlan::Blur(a) => {
            let tmp = sim_motion_blur(img, flow, a, hiq, 0.5, is360, rows, seed)?;
            Ok(sim_clear_rgb(&tmp, seed))
        }
        CompPlan::Sharpen(a) => {
            let mut tmp = sim_motion_blur(img, flow, 0.5, hiq, 0.5, is360, rows, seed)?;
            sim_lerp(&mut tmp, img, a, seed);
            Ok(tmp)
        }
    }
}

/// `smart_blend_enqueue` / `smart_blend_binary_enqueue` with the transcriptions.
pub fn sim_smart(p1: &Image, p2: &Image, w: f32, binary: bool, seed: u64) -> Image {
    let px = p1.width * p1.height;
    let n = p1.sample_count();
    let b = binary as i32;
    let s = match (&p1.samples, &p2.samples) {
        (Samples::U8(x), Samples::U8(y)) => Samples::U8(run_1d(px, n, 0, seed, |i, g| rs::sb_smart_item(i, x, y, g, w, b, px as u32))),
        (Samples::U16(x), Samples::U16(y)) => Samples::U16(run_1d(px, n, 0, seed, |i, g| rs::sb_smart_item(i, x, y, g, w, b, px as u32))),
        (Samples::F32(x), Samples::F32(y)) => Samples::F32(run_1d(px, n, 0.0, seed, |i, g| rs::sb_smart_item(i, x, y, g, w, b, px as u32))),
        _ => unreachable!(),
    };
    like(p1, s)
}

/// `coverage_holes_enqueue` with the transcription.
pub fn sim_cov_holes(c1: &Coverage, c2: &Coverage, seed: u64) -> Coverage {
    let n = c1.data.len();
    let data = run_1d(n, n, 0u8, seed, |i, g| rs::sb_cov_max_invert_item(i, &c1.data, &c2.data, g, n as u32));
    Coverage { width: c1.width, height: c1.height, data }
}

/// `coverage_invert_enqueue` with the transcription.
pub fn sim_cov_invert(c: &Coverage, seed: u64) -> Coverage {
    let n = c.data.len();
    let data = run_1d(n, n, 0u8, seed, |i, g| rs::sb_cov_invert_item(i, &c.data, g, n as u32));
    Coverage { width: c.width, height: c.height, data }
}

/// Blend weights: snaps, both sides of the snap tolerance, ties, extrapolation, NaN.
pub fn weights() -> Vec<f32> {
    vec![0.0, -0.0, 0.004, 0.0051, 0.25, 0.5, 0.75, 1.0 / 3.0, 0.9949, 0.996, 1.0, 1.5, -0.5, 2.0, -3.25, 1e-7, f32::NAN, f32::INFINITY, f32::from_bits(1)]
}
