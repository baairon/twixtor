//! Device tests of the vector, matte and overlay stages on the OpenCL devices of the machine
//! (`probe::open(0)` and `probe::open(1)`): every stage against its CPU function over the corpora
//! of the device-free tests plus 512x512 images, compared as bit patterns, more than a million
//! output samples per stage and device. A float output is compared in full with the downloads
//! keeping NaNs (a NaN must be a NaN, its payload aside); then the wrapper itself must give
//! `StageError::NanOutput` exactly when the CPU result holds a NaN, and the CPU's bits otherwise.
//! With no OpenCL runtime or no usable device every test prints a note and passes (CI runners
//! have none). The `#[ignore]`d probes print GPU (transfers included) and CPU times at 1080p.

use super::tests::corpus::*;
use super::tests::{float_bits, image_bits};
use super::{geom as g, matte as mt, overlay as ov, prep as pp, program_parts, vectors as vc, KEEP_NAN};
use crate::device::image::StageError;
use crate::device::kernels;
use crate::device::probe;
use crate::device::selfcheck::Session;
use crate::device::sim::Rng;
use crate::flow::{FlowField, FlowSettings};
use crate::image::{Fmt, Image};
use crate::matte::Matte;
use crate::prep::Plane;
use crate::vectors::HoleMask;
use std::sync::Mutex;
use std::time::Instant;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// The usable devices 0 and 1 with a built program, or none (with a printed note).
fn sessions(test: &str) -> Vec<(String, Session)> {
    let mut out = Vec::new();
    for index in 0..2 {
        let (dev, info) = match probe::open(index) {
            Ok(d) => d,
            Err(e) => {
                println!("{test}: gpu={index} skipped, {e}");
                continue;
            }
        };
        let why = probe::requirements(&info);
        if !why.is_empty() {
            println!("{test}: gpu={index} {} skipped, {why:?}", info.name);
            continue;
        }
        let session = Session::open(&dev, &kernels::program(&program_parts())).unwrap_or_else(|e| panic!("{test}: gpu={index} {}: the program does not build: {e}", info.name));
        out.push((format!("gpu={index} {}", info.name), session));
    }
    out
}

/// Runs `f` with the downloads keeping NaNs.
fn raw<T>(f: impl FnOnce() -> T) -> T {
    KEEP_NAN.with(|k| k.set(true));
    let r = f();
    KEEP_NAN.with(|k| k.set(false));
    r
}

/// Compares bit patterns (NaN against NaN, any payload); panics at the first difference.
fn same(dev: &str, stage: &str, case: &str, want: &[u64], got: &[u64]) -> usize {
    assert_eq!(want.len(), got.len(), "{dev} {stage} {case}: lengths");
    if let Some(i) = want.iter().zip(got).position(|(a, b)| a != b) {
        panic!("{dev} {stage} {case}: sample {i} differs: cpu 0x{:x} device 0x{:x}", want[i], got[i]);
    }
    want.len()
}

/// The NaN rule of a wrapper: `NanOutput` when the CPU result holds a NaN, else the CPU's bits.
fn rule<T>(dev: &str, stage: &str, case: &str, cpu_nan: bool, r: Result<T, StageError>) -> Option<T> {
    match (cpu_nan, r) {
        (true, Err(StageError::NanOutput)) => None,
        (false, Ok(v)) => Some(v),
        (true, Ok(_)) => panic!("{dev} {stage} {case}: a NaN output was handed on"),
        (_, Err(e)) => panic!("{dev} {stage} {case}: {e}"),
    }
}

fn img_nan(img: &Image) -> bool {
    crate::device::image::has_nan(img)
}

fn nan(v: &[f32]) -> bool {
    v.iter().any(|x| x.is_nan())
}

fn mask_bits(m: &HoleMask) -> Vec<u64> {
    match m {
        HoleMask::U8(v) => v.iter().map(|x| *x as u64).collect(),
        HoleMask::U16(v) => v.iter().map(|x| *x as u64).collect(),
        HoleMask::F32(v) => float_bits(v),
    }
}

/// The corpus sizes plus two large ones.
fn sizes() -> Vec<(usize, usize)> {
    let mut v = SIZES.to_vec();
    v.extend_from_slice(&[(512, 512), (509, 257)]);
    v
}

const MILLION: usize = 1_000_000;

#[test]
fn vector_coding_on_the_device_equals_the_cpu() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    for (dev, s) in sessions("vector coding") {
        let mut rng = Rng::new(0xD0C0_0001);
        let (mut enc, mut mrg, mut mfl, mut dec, mut hol, mut scl, mut zer, mut blu) = (0, 0, 0, 0, 0, 0, 0, 0);
        for (dw, dh) in sizes() {
            let big = dw * dh > 10_000;
            for fmt in FMTS {
                // encode, from a flow of the same and of another size
                if dw >= 2 && dh >= 2 {
                    for (fw, fh, scale) in [(dw, dh, 8.0f32), (5, 3, 0.37), (dw / 2 + 2, dh / 3 + 2, -2.0)] {
                        let f = flow(&mut rng, fw, fh, 20.0, !big);
                        let mut want = Image::new(dw, dh, fmt);
                        crate::vectors::encode_vectors(&mut want, &f, scale).unwrap();
                        let got = raw(|| vc::encode_vectors_image_gpu(&s, dw, dh, fmt, &f, scale)).unwrap();
                        enc += same(&dev, "encode", &format!("{fmt:?} {dw}x{dh}"), &image_bits(&want), &image_bits(&got));
                        if let Some(img) = rule(&dev, "encode", "", img_nan(&want), vc::encode_vectors_image_gpu(&s, dw, dh, fmt, &f, scale)) {
                            same(&dev, "encode", "wrapper", &image_bits(&want), &image_bits(&img));
                        }
                    }
                    // the merge over 1 to 4 layers, and merge_flows
                    for nl in 1..=4usize {
                        if big && nl != 3 {
                            continue;
                        }
                        let flows: Vec<FlowField> = (0..nl).map(|k| flow(&mut rng, [dw, 2, 9, dw / 2 + 2][k], [dh, 2, 6, dh / 2 + 2][k], 30.0, !big)).collect();
                        let weights: Vec<Matte> = (0..nl).map(|_| matte(&mut rng, dw, dh, !big)).collect();
                        let layers: Vec<(&FlowField, &Matte)> = flows.iter().zip(&weights).collect();
                        let mut want = Image::new(dw, dh, fmt);
                        crate::vectors::encode_vectors_merged(&mut want, &layers, 6.0).unwrap();
                        let got = raw(|| vc::encode_vectors_merged_image_gpu(&s, dw, dh, fmt, &layers, 6.0)).unwrap();
                        mrg += same(&dev, "merge", &format!("{fmt:?} {dw}x{dh} {nl}"), &image_bits(&want), &image_bits(&got));
                        if let Some(img) = rule(&dev, "merge", "", img_nan(&want), vc::encode_vectors_merged_image_gpu(&s, dw, dh, fmt, &layers, 6.0)) {
                            same(&dev, "merge", "wrapper", &image_bits(&want), &image_bits(&img));
                        }
                        // create-vectors fix 03: a weight in the blue channel
                        let mut want = Image::new(dw, dh, fmt);
                        crate::vectors::encode_vectors_merged_blue(&mut want, &layers, 6.0, &weights[0]).unwrap();
                        let mut got = Image::new(dw, dh, fmt);
                        raw(|| vc::encode_vectors_merged_blue_gpu(&s, &mut got, &layers, 6.0, &weights[0])).unwrap();
                        blu += same(&dev, "merge blue", &format!("{fmt:?} {dw}x{dh} {nl}"), &image_bits(&want), &image_bits(&got));
                        let mut again = Image::new(dw, dh, fmt);
                        rule(&dev, "merge blue", "", img_nan(&want), vc::encode_vectors_merged_blue_gpu(&s, &mut again, &layers, 6.0, &weights[0]));
                        if fmt == Fmt::F32 || big {
                            let want = crate::vectors::merge_flows(dw, dh, &layers).unwrap();
                            let got = raw(|| vc::merge_flows_gpu(&s, dw, dh, &layers)).unwrap();
                            mfl += same(&dev, "merge_flows", &format!("{dw}x{dh} {nl}"), &float_bits(&want.data), &float_bits(&got.data));
                            rule(&dev, "merge_flows", "", nan(&want.data), vc::merge_flows_gpu(&s, dw, dh, &layers));
                        }
                    }
                }
                // decode with holes, vec scale, zero image
                let img = image(&mut rng, dw, dh, fmt, !big);
                for md in [32.0f32, -0.7, 0.0] {
                    let want = crate::vectors::decode_vectors(&img, md);
                    let holes = crate::vectors::vector_hole_mask(&img);
                    let (gf, gh) = raw(|| vc::decode_vectors_with_holes_gpu(&s, &img, md)).unwrap();
                    dec += same(&dev, "decode", &format!("{fmt:?} {dw}x{dh} md {md}"), &float_bits(&want.data), &float_bits(&gf.data));
                    hol += same(&dev, "holes", &format!("{fmt:?} {dw}x{dh}"), &mask_bits(&holes), &mask_bits(&gh));
                    rule(&dev, "decode", "", nan(&want.data), vc::decode_vectors_gpu(&s, &img, md));
                    let hnan = matches!(&holes, HoleMask::F32(v) if nan(v));
                    rule(&dev, "holes", "", hnan, vc::vector_hole_mask_gpu(&s, &img));
                }
                let zero = vc::zero_vectors_image_gpu(&s, dw, dh, fmt).unwrap();
                zer += same(&dev, "zero", &format!("{fmt:?}"), &image_bits(&crate::vectors::zero_vectors_image(dw, dh, fmt)), &image_bits(&zero));
            }
            for (sx, sy) in [(1.0f32, 1.0f32), (-0.5, 3.0), (0.0, f32::INFINITY)] {
                let f = flow(&mut rng, dw, dh, 50.0, !big);
                let mut want = f.clone();
                crate::vectors::apply_vec_scale(&mut want, sx, sy);
                let mut got = f.clone();
                raw(|| vc::apply_vec_scale_gpu(&s, &mut got, sx, sy)).unwrap();
                scl += same(&dev, "vec scale", &format!("{dw}x{dh}"), &float_bits(&want.data), &float_bits(&got.data));
                let mut again = f.clone();
                rule(&dev, "vec scale", "", nan(&want.data), vc::apply_vec_scale_gpu(&s, &mut again, sx, sy));
            }
        }
        // a blue weight of another size is the CPU's Mismatch, Unsupported before any launch
        let f = flow(&mut rng, 3, 2, 1.0, false);
        let one = Matte::filled(3, 2, 1.0);
        let small = Matte::filled(2, 2, 1.0);
        let mut dst = Image::new(3, 2, Fmt::U8);
        assert!(matches!(crate::vectors::encode_vectors_merged_blue(&mut dst, &[(&f, &one)], 1.0, &small), Err(fx_core::FxError::Mismatch(_))));
        assert!(matches!(vc::encode_vectors_merged_blue_gpu(&s, &mut dst, &[(&f, &one)], 1.0, &small), Err(StageError::Unsupported(_))));
        println!("{dev}: encode {enc}, merge {mrg}, merge blue {blu}, merge_flows {mfl}, decode {dec}, holes {hol}, vec scale {scl}, zero {zer} samples identical");
        for (name, n) in [("encode", enc), ("merge", mrg), ("merge blue", blu), ("merge_flows", mfl), ("decode", dec), ("holes", hol), ("vec scale", scl), ("zero", zer)] {
            assert!(n >= MILLION, "{dev} {name}: only {n} samples compared");
        }
    }
}

#[test]
fn matte_operations_on_the_device_equal_the_cpu() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    for (dev, s) in sessions("matte") {
        let mut rng = Rng::new(0xD0C0_0002);
        let (mut ext, mut pre, mut ovr, mut eff, mut ow, mut sw) = (0, 0, 0, 0, 0, 0);
        for (w, h) in sizes() {
            let big = w * h > 10_000;
            for fmt in FMTS {
                let img = image(&mut rng, w, h, fmt, !big);
                for chan in [1, 2, 3, 4] {
                    let want = crate::matte::extract(&img, chan);
                    let got = raw(|| mt::extract_gpu(&s, &img, chan)).unwrap();
                    ext += same(&dev, "extract", &format!("{fmt:?} {w}x{h} {chan}"), &float_bits(&want.data), &float_bits(&got.data));
                    rule(&dev, "extract", "", nan(&want.data), mt::extract_gpu(&s, &img, chan));
                }
                let m = matte(&mut rng, w, h, !big);
                for (mm, what) in [(&m, "corpus"), (&clamp_matte(&mut rng, w, h), "clamp corpus")] {
                    let want = crate::matte::premultiply(&img, mm).unwrap();
                    let got = raw(|| mt::premultiply_gpu(&s, &img, mm)).unwrap();
                    pre += same(&dev, "premultiply", &format!("{fmt:?} {w}x{h} {what}"), &image_bits(&want), &image_bits(&got));
                    rule(&dev, "premultiply", what, img_nan(&want), mt::premultiply_gpu(&s, &img, mm));
                }
                for src in [image(&mut rng, w, h, fmt, !big), crate::matte::premultiply(&img, &m).unwrap(), Image::new(w, h, fmt)] {
                    let dest = image(&mut rng, w, h, fmt, !big);
                    let mut want = dest.clone();
                    crate::matte::over(&mut want, &src).unwrap();
                    let mut got = dest.clone();
                    raw(|| mt::over_gpu(&s, &mut got, &src)).unwrap();
                    ovr += same(&dev, "over", &format!("{fmt:?} {w}x{h}"), &image_bits(&want), &image_bits(&got));
                    let mut again = dest.clone();
                    rule(&dev, "over", "", img_nan(&want), mt::over_gpu(&s, &mut again, &src));
                }
            }
            for (nfg, clamp) in [(0usize, false), (1, false), (2, false), (3, false), (1, true), (2, true), (3, true)] {
                let mattes: Vec<Matte> = (0..nfg).map(|_| if clamp { clamp_matte(&mut rng, w, h) } else { matte(&mut rng, w, h, !big) }).collect();
                let case = format!("{w}x{h} {nfg} clamp corpus {clamp}");
                let want = crate::matte::effective(&mattes, &vec![0.0; nfg], w, h).unwrap();
                let got = raw(|| mt::effective_gpu(&s, &mattes, &vec![0.0; nfg], w, h)).unwrap();
                assert_eq!(want.len(), got.len(), "{dev} effective {case}: entries");
                for (a, b) in want.iter().zip(&got) {
                    eff += same(&dev, "effective", &case, &float_bits(&a.data), &float_bits(&b.data));
                }
                let cpu_nan = want.iter().any(|m| nan(&m.data));
                if let Some(v) = rule(&dev, "effective", &case, cpu_nan, mt::effective_gpu(&s, &mattes, &vec![0.0; nfg], w, h)) {
                    assert_eq!(v, want, "{dev} effective {case}: wrapper");
                }
                for first in 0..=nfg {
                    for last in first..=nfg {
                        let want = crate::matte::over_weights(&mattes, first, last, w, h).unwrap();
                        let got = raw(|| mt::over_weights_gpu(&s, &mattes, first, last, w, h)).unwrap();
                        assert_eq!(want.len(), got.len(), "{dev} over_weights {case}: entries");
                        for (a, b) in want.iter().zip(&got) {
                            ow += same(&dev, "over_weights", &format!("{case} {first}..={last}"), &float_bits(&a.data), &float_bits(&b.data));
                        }
                        // create-vectors fix 03: the weights inside the whole stack
                        let want = crate::matte::stack_weights(&mattes, first, last, w, h).unwrap();
                        let got = raw(|| mt::stack_weights_gpu(&s, &mattes, first, last, w, h)).unwrap();
                        assert_eq!(want.len(), got.len(), "{dev} stack_weights {case}: entries");
                        for (a, b) in want.iter().zip(&got) {
                            sw += same(&dev, "stack_weights", &format!("{case} {first}..={last}"), &float_bits(&a.data), &float_bits(&b.data));
                        }
                    }
                }
            }
        }
        // the CPU's Mismatch cases are Unsupported before any launch, and so is a shrink
        let half = Matte::filled(2, 2, 0.5);
        assert!(matches!(mt::effective_gpu(&s, &[half.clone()], &[1.0], 2, 2), Err(StageError::Unsupported(_))), "a shrink needs the CPU");
        let short = Matte { width: 2, height: 2, data: vec![0.5; 3] };
        for (mattes, shrinks, w, h) in [(vec![half.clone()], vec![], 2, 2), (vec![half.clone()], vec![0.0], 3, 2), (vec![short.clone()], vec![0.0], 2, 2), (vec![half.clone(), Matte::filled(2, 3, 0.5)], vec![0.0, 0.0], 2, 2)] {
            assert!(matches!(crate::matte::effective(&mattes, &shrinks, w, h), Err(fx_core::FxError::Mismatch(_))));
            assert!(matches!(mt::effective_gpu(&s, &mattes, &shrinks, w, h), Err(StageError::Unsupported(_))), "{dev}: effective {w}x{h}");
        }
        for (first, last, w, h) in [(1, 0, 2, 2), (0, 2, 2, 2), (0, 1, 3, 2)] {
            assert!(matches!(crate::matte::over_weights(&[half.clone()], first, last, w, h), Err(fx_core::FxError::Mismatch(_))));
            assert!(matches!(mt::over_weights_gpu(&s, &[half.clone()], first, last, w, h), Err(StageError::Unsupported(_))), "{dev}: over_weights {first}..={last} {w}x{h}");
        }
        assert!(matches!(mt::over_weights_gpu(&s, &[short.clone()], 0, 1, 2, 2), Err(StageError::Unsupported(_))));
        for (first, last, w, h) in [(1, 0, 2, 2), (0, 2, 2, 2), (0, 1, 3, 2)] {
            assert!(matches!(crate::matte::stack_weights(&[half.clone()], first, last, w, h), Err(fx_core::FxError::Mismatch(_))));
            assert!(matches!(mt::stack_weights_gpu(&s, &[half.clone()], first, last, w, h), Err(StageError::Unsupported(_))), "{dev}: stack_weights {first}..={last} {w}x{h}");
        }
        for m in [Matte::filled(2, 3, 1.0), Matte { width: 3, height: 2, data: vec![1.0; 5] }] {
            assert!(matches!(crate::matte::premultiply(&Image::new(3, 2, Fmt::U8), &m), Err(fx_core::FxError::Mismatch(_))));
            assert!(matches!(mt::premultiply_gpu(&s, &Image::new(3, 2, Fmt::U8), &m), Err(StageError::Unsupported(_))));
        }
        assert!(matches!(mt::over_gpu(&s, &mut Image::new(3, 2, Fmt::U8), &Image::new(3, 2, Fmt::U16)), Err(StageError::Unsupported(_))));
        println!("{dev}: extract {ext}, premultiply {pre}, over {ovr}, effective {eff}, over_weights {ow}, stack_weights {sw} samples identical");
        for (name, n) in [("extract", ext), ("premultiply", pre), ("over", ovr), ("effective", eff), ("over_weights", ow), ("stack_weights", sw)] {
            assert!(n >= MILLION, "{dev} {name}: only {n} samples compared");
        }
    }
}

#[test]
fn debug_compositions_on_the_device_equal_the_cpu() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    for (dev, s) in sessions("debug views") {
        let mut rng = Rng::new(0xD0C0_0003);
        let (mut fl, mut bl, mut df, mut ol, mut lu, mut dt, mut ed) = (0, 0, 0, 0, 0, 0, 0);
        for (w, h) in sizes() {
            let big = w * h > 10_000;
            for fmt in FMTS {
                for _ in 0..2 {
                    let a = image(&mut rng, w, h, fmt, !big);
                    let b = image(&mut rng, w, h, fmt, !big);
                    let mut want = Image::new(w, h, fmt);
                    crate::blend::blend(&mut want, &a, &b, 0.5);
                    bl += same(&dev, "blend view", &format!("{fmt:?} {w}x{h}"), &image_bits(&want), &image_bits(&raw(|| ov::blend_view_gpu(&s, &a, &b)).unwrap()));
                    rule(&dev, "blend view", "", img_nan(&want), ov::blend_view_gpu(&s, &a, &b));
                    let mut want = Image::new(w, h, fmt);
                    crate::debug::diff(&mut want, &a, &b);
                    df += same(&dev, "diff view", &format!("{fmt:?} {w}x{h}"), &image_bits(&want), &image_bits(&raw(|| ov::diff_view_gpu(&s, &a, &b)).unwrap()));
                    rule(&dev, "diff view", "", img_nan(&want), ov::diff_view_gpu(&s, &a, &b));
                    // W': every third pixel of `a` transparent on top of the corpus alpha
                    let mut wa = a.clone();
                    super::tests::punch(&mut wa);
                    let mut want = wa.clone();
                    crate::debug::fill_transparent(&mut want, &b);
                    let mut got = wa.clone();
                    raw(|| ov::fill_transparent_gpu(&s, &mut got, &b)).unwrap();
                    fl += same(&dev, "fill", &format!("{fmt:?} {w}x{h}"), &image_bits(&want), &image_bits(&got));
                    let mut got = wa.clone();
                    rule(&dev, "fill", "", img_nan(&want), ov::fill_transparent_gpu(&s, &mut got, &b));
                    let want = crate::debug::debug_luma(&a);
                    lu += same(&dev, "debug luma", &format!("{fmt:?} {w}x{h}"), &float_bits(&want.data), &float_bits(&raw(|| ov::debug_luma_gpu(&s, &a)).unwrap().data));
                    rule(&dev, "debug luma", "", nan(&want.data), ov::debug_luma_gpu(&s, &a));
                }
                let r: Vec<f32> = (0..w * h).map(|_| matte_value(&mut rng, !big)).collect();
                let gg: Vec<f32> = (0..w * h).map(|_| matte_value(&mut rng, !big)).collect();
                let mut want = Image::new(w, h, fmt);
                crate::debug::overlay(&mut want, &r, &gg);
                ol += same(&dev, "overlay", &format!("{fmt:?} {w}x{h}"), &image_bits(&want), &image_bits(&raw(|| ov::overlay_gpu(&s, w, h, fmt, &r, &gg)).unwrap()));
                rule(&dev, "overlay", "", img_nan(&want), ov::overlay_gpu(&s, w, h, fmt, &r, &gg));
            }
            for k in 0..4 {
                // luma planes in the 0..1 range of the Debug views, then wider values
                let l = plane(&mut rng, w, h, if k < 2 { 1.0 } else { 300.0 }, !big);
                let want = crate::debug::detail(&l);
                dt += same(&dev, "detail", &format!("{w}x{h}"), &float_bits(&want), &float_bits(&raw(|| ov::detail_gpu(&s, &l)).unwrap()));
                rule(&dev, "detail", "", nan(&want), ov::detail_gpu(&s, &l));
                let want = crate::debug::edge(&l);
                ed += same(&dev, "edge", &format!("{w}x{h}"), &float_bits(&want), &float_bits(&raw(|| ov::edge_gpu(&s, &l)).unwrap()));
                rule(&dev, "edge", "", nan(&want), ov::edge_gpu(&s, &l));
            }
        }
        println!("{dev}: fill {fl}, blend view {bl}, diff view {df}, overlay {ol}, debug luma {lu}, detail {dt}, edge {ed} samples identical");
        for (name, n) in [("fill", fl), ("blend view", bl), ("diff view", df), ("overlay", ol), ("debug luma", lu), ("detail", dt), ("edge", ed)] {
            assert!(n >= MILLION, "{dev} {name}: only {n} samples compared");
        }
    }
}

#[test]
fn draw_geom_and_the_point_splat_on_the_device_equal_the_cpu() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    for (dev, s) in sessions("geometry") {
        let mut rng = Rng::new(0xD0C0_0004);
        let (mut dr, mut sp, mut st) = (0, 0, 0);
        for (w, h) in sizes() {
            let big = w * h > 10_000;
            for fmt in FMTS {
                for np in [1usize, 5, 12, 16] {
                    let pts: Vec<([f32; 2], [u8; 3])> = (0..np)
                        .map(|k| {
                            let p = if k % 4 == 3 { [float(&mut rng, 1e6, true), float(&mut rng, 1e6, true)] } else { [(rng.below(w as u64 + 4) as f32) - 2.0 + 0.25 * k as f32, rng.below(h as u64 + 4) as f32 - 2.0] };
                            (p, [rng.next_u32() as u8, rng.next_u32() as u8, rng.next_u32() as u8])
                        })
                        .collect();
                    for ratio in [1.0f32, 3.0, if big { 40.0 } else { 0.0 }] {
                        let img = image(&mut rng, w, h, fmt, !big);
                        let mut want = img.clone();
                        crate::geom::draw_points(&mut want, &pts, ratio);
                        let mut got = img.clone();
                        raw(|| g::draw_points_gpu(&s, &mut got, &pts, ratio)).unwrap();
                        dr += same(&dev, "draw", &format!("{fmt:?} {w}x{h} {np} ratio {ratio}"), &image_bits(&want), &image_bits(&got));
                        let mut again = img.clone();
                        rule(&dev, "draw", "", img_nan(&want), g::draw_points_gpu(&s, &mut again, &pts, ratio));
                    }
                }
            }
            for quality in [5u8, 4] {
                let Ok(geo) = FlowSettings::new(quality, true, 1.0).geometry(w, h) else { continue };
                for np in [0usize, 1, 3, 12, 16] {
                    let pairs: Vec<([f32; 2], [f32; 2])> = (0..np)
                        .map(|k| {
                            let from = if k % 5 == 4 { [float(&mut rng, 1e4, true), 3.0] } else { [rng.below(w as u64 + 2) as f32 - 1.0 + 0.3, rng.below(h as u64) as f32 + 0.5] };
                            let to = [from[0] + float(&mut rng, 6.0, false), from[1] - float(&mut rng, 6.0, k == 2)];
                            (from, to)
                        })
                        .collect();
                    let want = crate::points::user_displacement(&pairs, &geo, (w, h));
                    let got = raw(|| g::user_displacement_gpu(&s, &pairs, &geo, (w, h))).unwrap();
                    for (a, b) in want.levels.iter().zip(&got.levels) {
                        sp += same(&dev, "splat d", &format!("{w}x{h} {np}"), &float_bits(&a.d), &float_bits(&b.d));
                        sp += same(&dev, "splat w", &format!("{w}x{h} {np}"), &float_bits(&a.w), &float_bits(&b.w));
                    }
                    if let Ok(want) = crate::points::second_tracker(&pairs, &geo, (w, h)) {
                        let got = raw(|| g::second_tracker_gpu(&s, &pairs, &geo, (w, h))).unwrap();
                        st += same(&dev, "second tracker", &format!("{w}x{h} {np}"), &float_bits(&want.data), &float_bits(&got.data));
                    }
                }
            }
        }
        println!("{dev}: draw {dr}, splat {sp}, second tracker {st} samples identical");
        for (name, n) in [("draw", dr), ("splat", sp), ("second tracker", st)] {
            assert!(n >= MILLION, "{dev} {name}: only {n} samples compared");
        }
    }
}

#[test]
fn luma_and_edge_enhance_on_the_device_equal_the_cpu() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    for (dev, s) in sessions("prep") {
        let mut rng = Rng::new(0xD0C0_0005);
        let (mut lu, mut ee) = (0, 0);
        for (w, h) in sizes() {
            let big = w * h > 10_000;
            for fmt in FMTS {
                let img = image(&mut rng, w, h, fmt, !big);
                for (alpha, video) in [(false, false), (true, false), (false, true)] {
                    let want = crate::prep::luma(&img, alpha, video).unwrap();
                    let got = raw(|| pp::luma_gpu(&s, &img, alpha, video)).unwrap();
                    lu += same(&dev, "luma", &format!("{fmt:?} {w}x{h} {alpha} {video}"), &float_bits(&want.data), &float_bits(&got.data));
                    rule(&dev, "luma", "", nan(&want.data), pp::luma_gpu(&s, &img, alpha, video));
                }
            }
            for wrap in [false, true] {
                for _ in 0..3 {
                    let p = plane(&mut rng, w, h, 300.0, !big);
                    let mut want = p.clone();
                    crate::prep::edge_enhance_with(&mut want, wrap);
                    let mut b = Plane { width: w, height: h, data: p.data.clone() };
                    if wrap {
                        crate::flow::conv::blur_gaussian_wrapx(&mut b.data, w, h, 1, 2.0);
                    } else {
                        crate::flow::conv::blur_gaussian(&mut b.data, w, h, 1, 2.0);
                    }
                    let got = raw(|| pp::edge_enhance_tail_gpu(&s, &b, wrap)).unwrap();
                    ee += same(&dev, "edge enhance", &format!("{w}x{h} {wrap}"), &float_bits(&want.data), &float_bits(&got.data));
                    rule(&dev, "edge enhance", "", nan(&want.data), pp::edge_enhance_tail_gpu(&s, &b, wrap));
                }
            }
        }
        assert!(matches!(pp::luma_gpu(&s, &Image::new(2, 2, Fmt::U8), true, true), Err(StageError::Unsupported(_))));
        println!("{dev}: luma {lu}, edge enhance {ee} samples identical");
        for (name, n) in [("luma", lu), ("edge enhance", ee)] {
            assert!(n >= MILLION, "{dev} {name}: only {n} samples compared");
        }
    }
}

// ---- timing probes ----------------------------------------------------------------------------

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// The best of three runs of `f`, in milliseconds.
fn best(mut f: impl FnMut()) -> f64 {
    (0..3)
        .map(|_| {
            let t = Instant::now();
            f();
            ms(t)
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
#[ignore = "timing probe: run with --ignored --nocapture"]
fn timing_probe_1080p() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let (w, h) = (1920usize, 1080usize);
    let mut rng = Rng::new(0x7133);
    let fmt = Fmt::U8;
    let img = image(&mut rng, w, h, fmt, false);
    let img2 = image(&mut rng, w, h, fmt, false);
    let f = flow(&mut rng, 480, 270, 10.0, false);
    // mattes in 0..1, as the effects make them (an infinite sample would give a NaN product)
    let mattes: Vec<Matte> = (0..3).map(|_| Matte { width: w, height: h, data: (0..w * h).map(|_| (rng.next_u32() >> 8) as f32 / 16777216.0).collect() }).collect();
    let flows: Vec<FlowField> = (0..3).map(|_| flow(&mut rng, 480, 270, 10.0, false)).collect();
    let layers: Vec<(&FlowField, &Matte)> = flows.iter().zip(&mattes).collect();
    let pts: Vec<([f32; 2], [u8; 3])> = (0..12).map(|k| ([100.0 + 150.0 * k as f32, 500.0], [255, 0, 0])).collect();
    let mut row: Vec<(String, f64)> = Vec::new();
    let cpu = |name: &str, row: &mut Vec<(String, f64)>, t: f64| row.push((format!("cpu {name}"), t));
    cpu("encode", &mut row, best(|| drop(crate::vectors::encode_vectors_image(w, h, fmt, &f, 8.0).unwrap())));
    cpu("merge 3", &mut row, best(|| drop(crate::vectors::encode_vectors_merged_image(w, h, fmt, &layers, 8.0).unwrap())));
    cpu("decode", &mut row, best(|| drop(crate::vectors::decode_vectors(&img, 8.0))));
    cpu("extract", &mut row, best(|| drop(crate::matte::extract(&img, 4))));
    cpu("premultiply", &mut row, best(|| drop(crate::matte::premultiply(&img, &mattes[0]).unwrap())));
    cpu("over", &mut row, best(|| crate::matte::over(&mut img2.clone(), &img).unwrap()));
    cpu("effective 3", &mut row, best(|| drop(crate::matte::effective(&mattes, &[0.0; 3], w, h).unwrap())));
    cpu("overlay", &mut row, best(|| crate::debug::overlay(&mut Image::new(w, h, fmt), &mattes[0].data, &mattes[1].data)));
    cpu("draw 12", &mut row, best(|| crate::geom::draw_points(&mut img.clone(), &pts, 1.0)));
    cpu("luma", &mut row, best(|| drop(crate::prep::luma(&img, false, false).unwrap())));
    let dl = crate::debug::debug_luma(&img);
    cpu("debug edge", &mut row, best(|| drop(crate::debug::edge(&dl))));
    println!("1080p 8 bpc, best of 3, milliseconds:");
    for (name, t) in &row {
        println!("  {name:<16} {t:8.2}");
    }
    for (dev, s) in sessions("timing") {
        let mut out: Vec<(&str, f64)> = Vec::new();
        out.push(("encode", best(|| drop(vc::encode_vectors_image_gpu(&s, w, h, fmt, &f, 8.0).unwrap()))));
        out.push(("merge 3", best(|| drop(vc::encode_vectors_merged_image_gpu(&s, w, h, fmt, &layers, 8.0).unwrap()))));
        out.push(("decode", best(|| drop(vc::decode_vectors_gpu(&s, &img, 8.0).unwrap()))));
        out.push(("extract", best(|| drop(mt::extract_gpu(&s, &img, 4).unwrap()))));
        out.push(("premultiply", best(|| drop(mt::premultiply_gpu(&s, &img, &mattes[0]).unwrap()))));
        out.push(("over", best(|| mt::over_gpu(&s, &mut img2.clone(), &img).unwrap())));
        out.push(("effective 3", best(|| drop(mt::effective_gpu(&s, &mattes, &[0.0; 3], w, h).unwrap()))));
        out.push(("overlay", best(|| drop(ov::overlay_gpu(&s, w, h, fmt, &mattes[0].data, &mattes[1].data).unwrap()))));
        out.push(("draw 12", best(|| g::draw_points_gpu(&s, &mut img.clone(), &pts, 1.0).unwrap())));
        out.push(("luma", best(|| drop(pp::luma_gpu(&s, &img, false, false).unwrap()))));
        out.push(("debug edge", best(|| drop(ov::edge_gpu(&s, &dl).unwrap()))));
        println!("{dev}, transfers included:");
        for (name, t) in out {
            println!("  gpu {name:<12} {t:8.2}");
        }
    }
}
