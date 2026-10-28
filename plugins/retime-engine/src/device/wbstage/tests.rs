//! Device-free tests of the Weighted Blend stages: the chains of the wrappers run with the
//! kernel transcriptions (`kernels::wbstage`, `kernels::conv`, the composition transcriptions of
//! `warpstage::corpus` and the hole fill on `convstage::sim::SimBackend`) in the work item
//! simulator, in every order of `sim::SEEDS`, and are compared bit for bit with the CPU functions
//! over random and adversarial images at 8, 16 and 32 bit, from 1x1 to 64x37 (NaN against NaN,
//! payloads aside). Plus checks of the plans and of the source.

use super::{border_args, clean_plan, dissolve_plan, program_parts, sharpen_scalars, DissolvePlan, MaskOp};
use crate::device::convstage::holefill::fill_holes_host;
use crate::device::convstage::sim::SimBackend;
use crate::device::image::StageError;
use crate::device::kernels::{self, conv as kconv, wbstage as kwb};
use crate::device::sim::{self, Global, Rng};
use crate::device::warpstage::corpus::{self as wc, first_diff};
use crate::filters::{self, Mask};
use crate::flow::FlowField;
use crate::image::{Fmt, Image, Samples};
use crate::smartblend::{self, SmartOpts};
use crate::warp::{Coverage, WarpOpts};

pub(super) const FMTS: [Fmt; 3] = [Fmt::U8, Fmt::U16, Fmt::F32];

/// The sizes of the device-free corpus: the degenerate ones of the mask clean-up (sides 1 to 6)
/// and the composition corpus.
pub(super) fn sizes() -> Vec<(usize, usize)> {
    let mut v = vec![(1, 1), (1, 4), (4, 1), (2, 5), (5, 2), (3, 3), (5, 5), (6, 6), (3, 7)];
    v.extend(wc::sizes());
    v
}

/// `img` with a few rectangles taken from `other` and a few single pixels changed, so the
/// difference mask has blobs, specks and untouched areas.
pub(super) fn perturb(rng: &mut Rng, img: &Image, other: &Image) -> Image {
    let (w, h) = (img.width, img.height);
    let mut out = img.clone();
    let copy_px = |out: &mut Image, x: usize, y: usize| {
        let i = (y * w + x) * 4;
        match (&mut out.samples, &other.samples) {
            (Samples::U8(d), Samples::U8(s)) => d[i..i + 4].copy_from_slice(&s[i..i + 4]),
            (Samples::U16(d), Samples::U16(s)) => d[i..i + 4].copy_from_slice(&s[i..i + 4]),
            (Samples::F32(d), Samples::F32(s)) => d[i..i + 4].copy_from_slice(&s[i..i + 4]),
            _ => unreachable!(),
        }
    };
    for _ in 0..3 {
        let (x0, y0) = (rng.below(w as u64) as usize, rng.below(h as u64) as usize);
        let (rw, rh) = (1 + rng.below(w as u64 / 2 + 1) as usize, 1 + rng.below(h as u64 / 2 + 1) as usize);
        for y in y0..(y0 + rh).min(h) {
            for x in x0..(x0 + rw).min(w) {
                copy_px(&mut out, x, y);
            }
        }
    }
    for _ in 0..(w * h / 8 + 1) {
        let (x, y) = (rng.below(w as u64) as usize, rng.below(h as u64) as usize);
        copy_px(&mut out, x, y);
    }
    out
}

/// Panics with the first differing sample (NaN against NaN, any payload).
pub(super) fn same(what: &str, want: &Image, got: &Image) {
    if let Some((i, a, b)) = first_diff(want, got) {
        panic!("{what}: sample {i}: cpu {a}, sim/device {b}");
    }
}

fn has_nan(img: &Image) -> bool {
    crate::device::image::has_nan(img)
}

// ---- the simulated chains -------------------------------------------------------------------------

fn run_1d<T: Copy>(items: usize, len: usize, fill: T, seed: u64, mut item: impl FnMut(usize, &mut Global<T>)) -> Vec<T> {
    let mut g = Global::new(len, fill);
    sim::launch_1d(items, seed, |i| item(i, &mut g));
    assert_eq!(g.written(), len, "every element is written once");
    g.into_vec()
}

fn run_2d<T: Copy>(w: usize, h: usize, len: usize, fill: T, seed: u64, mut item: impl FnMut(usize, usize, &mut Global<T>)) -> Vec<T> {
    let mut g = Global::new(len, fill);
    sim::launch_2d(w, h, seed, |x, y| item(x, y, &mut g));
    assert_eq!(g.written(), len, "every element is written once");
    g.into_vec()
}

/// `change_mask_enqueue` with the transcriptions.
pub(super) fn sim_change_mask(a: &Image, b: &Image, seed: u64) -> Mask {
    let (w, h) = (a.width, a.height);
    let n = w * h;
    let (limit, nn) = (filters::mask_limit(a.fmt(), filters::MASK_THRESHOLD), n as u32);
    let mut cur = match (&a.samples, &b.samples) {
        (Samples::U8(x), Samples::U8(y)) => run_1d(n, n, 0u8, seed, |i, g| kwb::wb_diff_mask_item(i, x, y, g, limit, nn)),
        (Samples::U16(x), Samples::U16(y)) => run_1d(n, n, 0u8, seed, |i, g| kwb::wb_diff_mask_item(i, x, y, g, limit, nn)),
        (Samples::F32(x), Samples::F32(y)) => run_1d(n, n, 0u8, seed, |i, g| kwb::wb_diff_mask_item(i, x, y, g, limit, nn)),
        _ => unreachable!(),
    };
    let (wi, hi) = (w as i32, h as i32);
    for op in clean_plan(w, h).unwrap() {
        match op {
            MaskOp::Prune(l) => cur = run_2d(w, h, n, 0u8, seed, |x, y, g| kwb::wb_mask_prune_item(x, y, &cur, g, wi, hi, l)),
            MaskOp::Dilate => cur = run_2d(w, h, n, 0u8, seed, |x, y, g| kwb::wb_mask_dilate_item(x, y, &cur, g, wi, hi)),
            MaskOp::Fill(l, v) => cur = run_2d(w, h, n, 0u8, seed, |x, y, g| kwb::wb_mask_fill_item(x, y, &cur, g, wi, hi, l, i32::from(v))),
            MaskOp::Border(a) => {
                sim::launch_1d(w, seed, |x| kwb::wb_border_rows_item(x, &mut cur, &a));
                sim::launch_1d(h, seed, |y| kwb::wb_border_cols_item(y, &mut cur, &a));
            }
        }
    }
    Mask { width: w, height: h, data: cur }
}

/// `blur_enqueue` with the transcriptions: `cv_hpass` and `cv_vpass` over four lanes, with the
/// integer loads and stores around them.
pub(super) fn sim_blur(img: &Image, seed: u64) -> Image {
    let (w, h) = (img.width, img.height);
    let n = img.sample_count();
    let (taps, _, _) = sharpen_scalars(img.fmt(), 0.5);
    let (wi, hi, nt, nn) = (w as i32, h as i32, taps.len() as i32, n as u32);
    let hpass = |src: &[f32]| run_2d(w * 4, h, n, 0.0f32, seed, |g, y, d| kconv::cv_hpass_item(g, y, src, d, wi, hi, 4, &taps, nt));
    let vpass = |src: &[f32]| run_2d(w * 4, h, n, 0.0f32, seed, |t, y, d| kconv::cv_vpass_item(t, y, src, d, wi, hi, 4, &taps, nt));
    let samples = match &img.samples {
        Samples::F32(v) => Samples::F32(vpass(&hpass(v))),
        Samples::U8(v) => {
            let f0 = run_1d(n, n, 0.0f32, seed, |i, g| kwb::wb_to_float_item(i, v, g, nn));
            let f1 = hpass(&f0);
            let f0 = run_1d(n, n, 0.0f32, seed, |i, g| kwb::wb_round_u8_item(i, &f1, g, nn));
            let f1 = vpass(&f0);
            Samples::U8(run_1d(n, n, 0u8, seed, |i, g| kwb::wb_from_float_u8_item(i, &f1, g, nn)))
        }
        Samples::U16(v) => {
            let f0 = run_1d(n, n, 0.0f32, seed, |i, g| kwb::wb_to_float_item(i, v, g, nn));
            let f1 = hpass(&f0);
            let f0 = run_1d(n, n, 0.0f32, seed, |i, g| kwb::wb_round_u16_item(i, &f1, g, nn));
            let f1 = vpass(&f0);
            Samples::U16(run_1d(n, n, 0u16, seed, |i, g| kwb::wb_from_float_u16_item(i, &f1, g, nn)))
        }
    };
    Image { width: w, height: h, samples }
}

fn sim_invert(m: &Mask, seed: u64) -> Mask {
    let n = m.data.len();
    Mask { width: m.width, height: m.height, data: run_1d(n, n, 0u8, seed, |i, g| kwb::wb_invert_mask_item(i, &m.data, g, n as u32)) }
}

fn sim_composite(dest: &Image, src: &Image, mask: &Mask, seed: u64) -> Image {
    let n = dest.sample_count();
    let nn = n as u32;
    let p = crate::device::kernels::prelude::rs::trunc_u8_f;
    let q = crate::device::kernels::prelude::rs::trunc_u16_f;
    let samples = match (&dest.samples, &src.samples) {
        (Samples::U8(d), Samples::U8(s)) => Samples::U8(run_1d(n, n, 0u8, seed, |i, g| kwb::wb_composite_int_item(i, d, g, s, &mask.data, nn, p))),
        (Samples::U16(d), Samples::U16(s)) => Samples::U16(run_1d(n, n, 0u16, seed, |i, g| kwb::wb_composite_int_item(i, d, g, s, &mask.data, nn, q))),
        (Samples::F32(d), Samples::F32(s)) => Samples::F32(run_1d(n, n, 0.0f32, seed, |i, g| kwb::wb_composite_f32_item(i, d, g, s, &mask.data, nn))),
        _ => unreachable!(),
    };
    Image { width: dest.width, height: dest.height, samples }
}

/// The blur and the lerp of `sharpen_step`.
fn sim_sharpened(frame: &Image, t: f32, seed: u64) -> Image {
    let mut sharp = sim_blur(frame, seed);
    wc::sim_lerp(&mut sharp, frame, filters::sharpen_weight(t), seed);
    sharp
}

/// `sharpen_step_enqueue` with the transcriptions.
pub(super) fn sim_sharpen_step(frame: &Image, original: &Image, t: f32, seed: u64) -> Image {
    let mask = sim_change_mask(frame, original, seed);
    let sharp = sim_sharpened(frame, t, seed);
    let inv = sim_invert(&mask, seed);
    sim_composite(frame, &sharp, &inv, seed)
}

/// `weighted_blend_noflow_reference_enqueue` with the transcriptions.
pub(super) fn sim_reference(a: &Image, wa: f32, b: &Image, seed: u64) -> Image {
    let t = 1.0f32 - wa;
    let a1 = sim_sharpen_step(a, a, t, seed);
    let b1 = sim_sharpen_step(b, b, wa, seed);
    wc::sim_blend(&a1, &b1, t, seed)
}

/// `weighted_tail_enqueue` with the transcriptions: `(out, ta', tb', holes)`, or the error of
/// the hole fill.
#[allow(clippy::type_complexity)]
pub(super) fn sim_tail(ta: &Image, a: &Image, tb: &Image, b: &Image, wa: f32, cov: Option<(&Coverage, &Coverage)>, seed: u64) -> Result<(Image, Image, Image, Option<Coverage>), StageError> {
    let frac = 1.0f32 - wa;
    let ta1 = sim_sharpen_step(ta, a, frac, seed);
    let tb1 = sim_sharpen_step(tb, b, wa, seed);
    match cov {
        Some((c1, c2)) => {
            let mut out = wc::sim_smart(&ta1, &tb1, frac, false, seed);
            let holes = wc::sim_cov_holes(c1, c2, seed);
            fill_holes_host(&SimBackend::new(seed), &mut out, &holes, false)?;
            Ok((out, ta1, tb1, Some(holes)))
        }
        None => Ok((wc::sim_blend(&ta1, &tb1, frac, seed), ta1, tb1, None)),
    }
}

/// `weighted_blend_smart_enqueue` with the transcriptions (`Err` where the device refuses).
pub(super) fn sim_weighted_smart(a: &Image, flow_ba: &FlowField, b: &Image, flow_ab: &FlowField, wa: f32, seed: u64) -> Result<Image, StageError> {
    let frac = 1.0f32 - wa;
    let wopts = WarpOpts { hold: false, is360: false, mode: 1 };
    let refused = |_| StageError::Unsupported("warp".into());
    let (ta, c1) = wc::sim_warp(a, flow_ba, frac, wopts, true, seed).map_err(refused)?;
    let (tb, c2) = wc::sim_warp(b, flow_ab, wa, wopts, true, seed).map_err(refused)?;
    let (c1, c2) = (c1.unwrap(), c2.unwrap());
    sim_tail(&ta, a, &tb, b, wa, Some((&c1, &c2)), seed).map(|r| r.0)
}

/// `flow_dissolve_inv_enqueue` with the transcriptions.
pub(super) fn sim_dissolve(a: &Image, flow_ab: Option<&FlowField>, b: &Image, flow_ba: Option<&FlowField>, wa: f32, mode: u32, seed: u64) -> Result<Image, StageError> {
    let frac = 1.0f32 - wa;
    match dissolve_plan(flow_ab.is_some(), flow_ba.is_some(), wa, mode)? {
        DissolvePlan::Copy { of_b } => Ok(if of_b { b.clone() } else { a.clone() }),
        DissolvePlan::Warped => {
            let wopts = WarpOpts { hold: true, is360: false, mode: 1 };
            let refused = |_| StageError::Unsupported("warp".into());
            let (ta, _) = wc::sim_warp(a, flow_ba.unwrap(), frac, wopts, false, seed).map_err(refused)?;
            let (tb, _) = wc::sim_warp(b, flow_ab.unwrap(), wa, wopts, false, seed).map_err(refused)?;
            let ta = sim_sharpen_step(&ta, a, frac, seed);
            let tb = sim_sharpen_step(&tb, b, wa, seed);
            Ok(wc::sim_blend(&ta, &tb, frac, seed))
        }
        DissolvePlan::Masked => {
            let inv = sim_invert(&sim_change_mask(a, b, seed), seed);
            let ta = sim_composite(a, &sim_sharpened(a, frac, seed), &inv, seed);
            let tb = sim_composite(b, &sim_sharpened(b, wa, seed), &inv, seed);
            Ok(wc::sim_blend(&ta, &tb, frac, seed))
        }
    }
}

/// The positions of the sharpening and blending tests: the frames, ties, the middle, values near
/// the dissolve's cut offs, outside 0..1 and NaN.
pub(super) fn positions() -> Vec<f32> {
    vec![0.0, 1.0, 0.5, 0.25, 0.75, 1.0 / 3.0, 0.001, 0.0021, 0.9979, 0.999, -0.25, 1.5, 0.004, 0.996, f32::NAN]
}

// ---- tests ------------------------------------------------------------------------------------------

#[test]
fn change_mask_transcriptions_equal_filters_in_every_order() {
    let mut rng = Rng::new(0x3B00_0001);
    for (w, h) in sizes() {
        for fmt in FMTS {
            for class in [0, 1] {
                let a = wc::image(&mut rng, w, h, fmt, class);
                let other = wc::image(&mut rng, w, h, fmt, class);
                let b = perturb(&mut rng, &a, &other);
                let want = filters::change_mask(&b, &a);
                let got = sim::same_for_every_order(|seed| sim_change_mask(&b, &a, seed));
                assert_eq!(got, want, "change mask {fmt:?} {w}x{h} class {class}");
                assert_eq!(sim_change_mask(&a, &a, 0), filters::change_mask(&a, &a));
            }
        }
    }
}

#[test]
fn every_mask_step_equals_its_cpu_function_on_random_masks() {
    // masks of 0, 128, 64 and 255 in random blobs, every size, every step on its own
    let mut rng = Rng::new(0x3B00_0002);
    for (w, h) in sizes() {
        for density in [1u64, 3, 6] {
            let data: Vec<u8> = (0..w * h).map(|_| if rng.below(8) < density { [255u8, 128, 64, 1][rng.below(4) as usize] } else { 0 }).collect();
            let m = Mask { width: w, height: h, data };
            let (wi, hi, n) = (w as i32, h as i32, w * h);
            for limit in [254, 509, 0, 2000] {
                let mut want = m.clone();
                filters::mask_prune(&mut want, limit);
                let got = if h >= 5 && w >= 3 { sim::same_for_every_order(|s| run_2d(w, h, n, 0u8, s, |x, y, g| kwb::wb_mask_prune_item(x, y, &m.data, g, wi, hi, limit))) } else { m.data.clone() };
                assert_eq!(got, want.data, "prune {w}x{h} {limit}");
                for value in [255u8, 128, 64] {
                    let mut want = m.clone();
                    filters::mask_fill(&mut want, limit, value);
                    let got = if h >= 3 && w >= 3 { sim::same_for_every_order(|s| run_2d(w, h, n, 0u8, s, |x, y, g| kwb::wb_mask_fill_item(x, y, &m.data, g, wi, hi, limit, i32::from(value)))) } else { m.data.clone() };
                    assert_eq!(got, want.data, "fill {w}x{h} {limit} {value}");
                }
            }
            let mut want = m.clone();
            filters::mask_dilate_rows(&mut want);
            let got = if h >= 3 { sim::same_for_every_order(|s| run_2d(w, h, n, 0u8, s, |x, y, g| kwb::wb_mask_dilate_item(x, y, &m.data, g, wi, hi))) } else { m.data.clone() };
            assert_eq!(got, want.data, "dilate {w}x{h}");
            let mut want = m.clone();
            filters::mask_replicate_border(&mut want);
            let a = border_args(w, h).unwrap();
            let got = sim::same_for_every_order(|s| {
                let mut cur = m.data.clone();
                sim::launch_1d(w, s, |x| kwb::wb_border_rows_item(x, &mut cur, &a));
                sim::launch_1d(h, s, |y| kwb::wb_border_cols_item(y, &mut cur, &a));
                cur
            });
            assert_eq!(got, want.data, "border {w}x{h}");
            let mut want = m.clone();
            filters::clean_mask(&mut want);
            let mut cur = m.data.clone();
            for op in clean_plan(w, h).unwrap() {
                match op {
                    MaskOp::Prune(l) => cur = run_2d(w, h, n, 0u8, 7, |x, y, g| kwb::wb_mask_prune_item(x, y, &cur, g, wi, hi, l)),
                    MaskOp::Dilate => cur = run_2d(w, h, n, 0u8, 7, |x, y, g| kwb::wb_mask_dilate_item(x, y, &cur, g, wi, hi)),
                    MaskOp::Fill(l, v) => cur = run_2d(w, h, n, 0u8, 7, |x, y, g| kwb::wb_mask_fill_item(x, y, &cur, g, wi, hi, l, i32::from(v))),
                    MaskOp::Border(a) => {
                        sim::launch_1d(w, 7, |x| kwb::wb_border_rows_item(x, &mut cur, &a));
                        sim::launch_1d(h, 7, |y| kwb::wb_border_cols_item(y, &mut cur, &a));
                    }
                }
            }
            assert_eq!(cur, want.data, "clean {w}x{h}");
        }
    }
}

#[test]
fn blur_and_composite_transcriptions_equal_filters_in_every_order() {
    let mut rng = Rng::new(0x3B00_0003);
    for (w, h) in sizes() {
        for fmt in FMTS {
            for class in [0, 1] {
                let img = wc::image(&mut rng, w, h, fmt, class);
                let mut want = img.clone();
                filters::blur(&mut want, filters::BLUR_RADIUS);
                let got = sim::same_for_every_order(|seed| wc::bits(&sim_blur(&img, seed)));
                same(&format!("blur {fmt:?} {w}x{h} class {class}"), &want, &Image { width: w, height: h, samples: from_bits(fmt, &got) });
                let src = wc::image(&mut rng, w, h, fmt, class);
                let mask = Mask { width: w, height: h, data: (0..w * h).map(|_| [0u8, 255, 1, 128, 254, rng.next_u32() as u8][rng.below(6) as usize]).collect() };
                let mut want = img.clone();
                filters::composite_masked(&mut want, &src, &mask);
                let got = sim::same_for_every_order(|seed| wc::bits(&sim_composite(&img, &src, &mask, seed)));
                same(&format!("composite {fmt:?} {w}x{h} class {class}"), &want, &Image { width: w, height: h, samples: from_bits(fmt, &got) });
            }
        }
    }
}

fn from_bits(fmt: Fmt, v: &[u64]) -> Samples {
    match fmt {
        Fmt::U8 => Samples::U8(v.iter().map(|x| *x as u8).collect()),
        Fmt::U16 => Samples::U16(v.iter().map(|x| *x as u16).collect()),
        Fmt::F32 => Samples::F32(v.iter().map(|x| f32::from_bits(*x as u32)).collect()),
    }
}

/// Runs `sim` for every seed, checks every order gives one result (NaN payloads aside) and
/// returns it.
fn every_order(sim: impl Fn(u64) -> Image) -> Image {
    let first = sim(sim::SEEDS[0]);
    for seed in &sim::SEEDS[1..] {
        let other = sim(*seed);
        assert!(first_diff(&first, &other).is_none(), "the result depends on the work item order (seed {seed:#x})");
    }
    first
}

#[test]
fn sharpen_step_and_the_reference_blend_equal_the_cpu_in_every_order() {
    let mut rng = Rng::new(0x3B00_0004);
    let mut count = 0usize;
    for (w, h) in sizes() {
        for fmt in FMTS {
            for class in [0, 1] {
                let original = wc::image(&mut rng, w, h, fmt, class);
                let other = wc::image(&mut rng, w, h, fmt, class);
                let frame = perturb(&mut rng, &original, &other);
                for &t in &positions()[..6] {
                    let mut want = frame.clone();
                    filters::sharpen_step(&mut want, &original, t);
                    let got = every_order(|seed| sim_sharpen_step(&frame, &original, t, seed));
                    same(&format!("sharpen {fmt:?} {w}x{h} class {class} t {t}"), &want, &got);
                    count += want.sample_count();
                }
                for wa in [0.5f32, 0.25, 0.9, 0.0, 1.0] {
                    let mut want = Image::new(w, h, fmt);
                    crate::blend::weighted_blend_noflow_reference(&mut want, &original, wa, &other);
                    let got = every_order(|seed| sim_reference(&original, wa, &other, seed));
                    same(&format!("reference {fmt:?} {w}x{h} class {class} wa {wa}"), &want, &got);
                }
            }
        }
    }
    assert!(count > 100_000, "{count}");
}

#[test]
fn weighted_tail_and_weighted_blend_smart_equal_the_cpu_in_every_order() {
    let mut rng = Rng::new(0x3B00_0005);
    for (w, h) in sizes().into_iter().filter(|(w, h)| *w >= 2 && *h >= 2) {
        for fmt in FMTS {
            for class in [0, 1] {
                let a = wc::image(&mut rng, w, h, fmt, class);
                let b = wc::image(&mut rng, w, h, fmt, class);
                let fab = wc::flow(&mut rng, w, h, 3.0, true, false);
                let fba = wc::flow(&mut rng, w, h, 3.0, true, false);
                for wa in [0.5f32, 0.3, 0.8] {
                    let mut want = Image::new(w, h, fmt);
                    let r = smartblend::weighted_blend_smart(&mut want, &a, &fba, &b, &fab, wa, SmartOpts::default());
                    let got = sim_weighted_smart(&a, &fba, &b, &fab, wa, 0);
                    match (r, got) {
                        (Ok(()), Ok(_)) => {
                            let g = every_order(|seed| sim_weighted_smart(&a, &fba, &b, &fab, wa, seed).unwrap());
                            same(&format!("weighted smart {fmt:?} {w}x{h} class {class} wa {wa}"), &want, &g);
                        }
                        (Ok(()), Err(StageError::NanOutput)) => assert!(has_nan(&want), "weighted smart {fmt:?} {w}x{h}: a NaN only on the sim"),
                        (r, g) => panic!("weighted smart {fmt:?} {w}x{h}: cpu {r:?}, sim {:?}", g.err()),
                    }
                    // the tail on its own, with and without coverage, on frames that differ from the sources
                    let ta = perturb(&mut rng, &a, &b);
                    let tb = perturb(&mut rng, &b, &a);
                    let c1 = Coverage { width: w, height: h, data: (0..w * h).map(|_| if rng.below(5) == 0 { 0 } else { 255 }).collect() };
                    let c2 = Coverage { width: w, height: h, data: (0..w * h).map(|_| if rng.below(5) == 0 { 0 } else { 255 }).collect() };
                    for with_cov in [false, true] {
                        let (mut out, mut ta1, mut tb1, mut cv1) = (Image::new(w, h, fmt), ta.clone(), tb.clone(), c1.clone());
                        let cov = with_cov.then_some((&mut cv1, &c2));
                        let r = smartblend::weighted_tail(&mut out, &mut ta1, &a, &mut tb1, &b, wa, cov, SmartOpts::default());
                        let got = sim_tail(&ta, &a, &tb, &b, wa, with_cov.then_some((&c1, &c2)), 0);
                        let what = format!("tail {fmt:?} {w}x{h} class {class} wa {wa} cov {with_cov}");
                        match (r, got) {
                            (Ok(()), Ok((o, t1, t2, holes))) => {
                                same(&what, &out, &o);
                                same(&what, &ta1, &t1);
                                same(&what, &tb1, &t2);
                                if let Some(hm) = holes {
                                    assert_eq!(hm, cv1, "{what}: holes");
                                }
                            }
                            (Ok(()), Err(StageError::NanOutput)) => assert!(has_nan(&out), "{what}: a NaN only on the sim"),
                            (r, g) => panic!("{what}: cpu {r:?}, sim {:?}", g.err()),
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn flow_dissolve_equals_the_cpu_in_every_mode_and_order() {
    let mut rng = Rng::new(0x3B00_0006);
    for (w, h) in sizes().into_iter().filter(|(w, h)| *w >= 2 && *h >= 2) {
        for fmt in FMTS {
            let a = wc::image(&mut rng, w, h, fmt, 0);
            let noise = wc::image(&mut rng, w, h, fmt, 1);
                let b = perturb(&mut rng, &a, &noise);
            let fab = wc::flow(&mut rng, w, h, 2.5, true, false);
            let fba = wc::flow(&mut rng, w, h, 2.5, true, false);
            for mode in [0u32, 5, 6, 2, 4, 7] {
                for (ab, ba) in [(true, true), (false, false), (true, false)] {
                    for wa in [0.5f32, 0.001, 0.999, 0.3, f32::NAN] {
                        let (fa, fb) = (ab.then_some(&fab), ba.then_some(&fba));
                        let mut want = Image::new(w, h, fmt);
                        let r = smartblend::flow_dissolve_inv(&mut want, &a, fa, &b, fb, wa, mode);
                        let what = format!("dissolve {fmt:?} {w}x{h} mode {mode} flows {ab} {ba} wa {wa}");
                        match (r, sim_dissolve(&a, fa, &b, fb, wa, mode, 0)) {
                            (Ok(()), Ok(_)) => {
                                let got = every_order(|seed| sim_dissolve(&a, fa, &b, fb, wa, mode, seed).unwrap());
                                same(&what, &want, &got);
                            }
                            (Err(_), Err(StageError::Unsupported(_))) => {}
                            (r, g) => panic!("{what}: cpu {r:?}, sim {:?}", g.err()),
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn the_plans_follow_the_cpu() {
    // clean_mask's limits and values, in its order, for a mask large enough for every step
    let plan = clean_plan(8, 8).unwrap();
    let kinds: Vec<String> = plan
        .iter()
        .map(|op| match op {
            MaskOp::Prune(l) => format!("prune {l}"),
            MaskOp::Dilate => "dilate".into(),
            MaskOp::Fill(l, v) => format!("fill {l} {v}"),
            MaskOp::Border(_) => "border".into(),
        })
        .collect();
    assert_eq!(kinds, ["prune 254", "prune 509", "dilate", "fill 254 255", "border", "fill 254 255", "border", "fill 254 128", "border", "fill 127 64", "border"]);
    // a mask of 2x2 only replicates its border; 1x1 too
    assert!(clean_plan(2, 2).unwrap().iter().all(|op| matches!(op, MaskOp::Border(_))));
    assert!(clean_plan(0, 4).unwrap().is_empty());
    // the border edges of an ordinary mask
    let a = border_args(10, 6).unwrap();
    assert_eq!((a.fx0, a.fx1, a.fy0, a.fy1, a.top, a.y1, a.bottom, a.left, a.x1, a.right), (1.0, 8.0, 1.0, 4.0, 1, 4, 4, 1, 8, 8));
    // the dissolve plan: the CPU's checks in its order
    assert_eq!(dissolve_plan(false, false, 0.5, 0).ok(), Some(DissolvePlan::Copy { of_b: false }));
    assert_eq!(dissolve_plan(false, false, 0.3, 0).ok(), Some(DissolvePlan::Copy { of_b: true }));
    assert_eq!(dissolve_plan(true, true, 0.001, 5).ok(), Some(DissolvePlan::Copy { of_b: true }));
    assert_eq!(dissolve_plan(true, true, f32::NAN, 5).ok(), Some(DissolvePlan::Warped));
    assert_eq!(dissolve_plan(false, false, 0.5, 6).ok(), Some(DissolvePlan::Masked));
    assert!(dissolve_plan(false, false, 0.5, 5).is_err());
    assert!(dissolve_plan(true, false, 0.5, 5).is_err());
    assert!(dissolve_plan(true, true, 0.5, 3).is_err());
}

/// The keyword of a kernel definition, spelled so this file holds no kernel source itself.
const KERNEL: &str = concat!("__", "kernel void ");

#[test]
fn the_source_keeps_its_prefix_bounds_and_constants() {
    let code: String = kwb::WB.lines().filter(|l| !l.trim_start().starts_with("/*") && !l.trim_start().starts_with("*")).collect::<Vec<_>>().join("\n");
    for line in code.lines() {
        for kw in [KERNEL, "int wb_", "float wb_"] {
            if let Some(rest) = line.trim_start().strip_prefix(kw) {
                assert!(kw.ends_with("wb_") || rest.starts_with("wb_"), "{line}");
            }
        }
    }
    // every function of the source carries the prefix, and none is defined twice
    let names: Vec<&str> = code.lines().filter_map(|l| l.split_once('(').map(|p| p.0)).filter_map(|h| h.rsplit(' ').next()).filter(|n| n.starts_with("wb_")).collect();
    for k in ["wb_diff_mask_u8", "wb_mask_prune", "wb_mask_fill", "wb_mask_dilate", "wb_border_rows", "wb_border_cols", "wb_invert_mask", "wb_round_u16", "wb_composite_f32", "wb_store_int"] {
        assert!(names.contains(&k), "{k}");
    }
    // the loop bounds match the plan's checks, the hex literals the CPU constants
    assert!(kwb::WB.contains("for (int q = 0; q < 2; q++)") && kwb::WB.contains("for (int q = 0; q < 3; q++)") && kwb::WB.contains("for (int j = 0; j < 3; j++)"));
    assert_eq!(f32::from_bits(0x437f_0000), 255.0);
    assert!(kwb::WB.contains("0x1.fep+7f") && kwb::WB.contains("0x1.fffep+15f") && kwb::WB.contains("0x1p-1f"));
    // the taps the host passes: four of blur_taps(3.0)
    let (taps, limit, weight) = sharpen_scalars(Fmt::U8, 0.5);
    assert_eq!(taps, filters::blur_taps(3.0));
    assert_eq!(taps.len(), 4);
    assert!(taps.len() <= kconv::CV_MAX_TAPS);
    assert_eq!(limit, filters::mask_limit(Fmt::U8, 0.4));
    assert_eq!(weight, -0.5);
    // the program holds every part it needs, the prefixes of the other stage modules included
    let program = kernels::program(&program_parts());
    for k in ["cv_hpass", "cv_vpass", "bl_lerp_f32", "bl_blend_f32", "wp_inv_warp_##SUF", "sb_smart_##SUF", "sb_cov_max_invert", "hf_combine", "wb_composite_u8"] {
        assert!(program.contains(&format!("{KERNEL}{k}(")), "{k}");
    }
}
