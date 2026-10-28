//! Device tests of the Weighted Blend stages on the OpenCL devices of the machine
//! (`probe::open(0)` and `probe::open(1)`): every stage against its CPU function over the
//! corpora of the device-free tests plus 256x192 and 509x257 images at every depth, more than a
//! million compared output samples per stage and device. A wrapper must give the CPU's bits, or
//! `StageError::NanOutput` exactly when the CPU result holds a NaN. With no OpenCL runtime or no
//! usable device every test prints a note and passes. The `#[ignore]`d probe prints GPU
//! (transfers included) and CPU times at 1080p.

use super::tests::{perturb, positions, same, sizes, FMTS};
use super::{flow_dissolve_inv_gpu, program, sharpen_step_gpu, weighted_blend_noflow_reference_gpu, weighted_blend_smart_gpu, weighted_tail_gpu};
use crate::device::image::{has_nan, StageError};
use crate::device::probe;
use crate::device::selfcheck::Session;
use crate::device::sim::Rng;
use crate::device::warpstage::corpus as wc;
use crate::filters;
use crate::image::{Fmt, Image};
use crate::smartblend::{self, SmartOpts};
use crate::warp::Coverage;
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
        let session = Session::open(&dev, &program()).unwrap_or_else(|e| panic!("{test}: gpu={index} {}: the program does not build: {e}", info.name));
        out.push((format!("gpu={index} {}", info.name), session));
    }
    out
}

/// The CPU's bits or, when the CPU result holds a NaN, `NanOutput`; returns the samples compared.
fn check(dev: &str, what: &str, want: &Image, got: Result<Image, StageError>) -> usize {
    match got {
        Ok(g) => {
            assert!(!has_nan(&g), "{dev} {what}: a NaN was handed on");
            same(&format!("{dev} {what}"), want, &g);
        }
        Err(StageError::NanOutput) => assert!(has_nan(want), "{dev} {what}: NanOutput without a NaN on the CPU"),
        Err(e) => panic!("{dev} {what}: {e}"),
    }
    want.sample_count()
}

fn device_sizes() -> Vec<(usize, usize)> {
    let mut v = sizes();
    v.extend_from_slice(&[(256, 192), (509, 257)]);
    v
}

const MILLION: usize = 1_000_000;

#[test]
fn sharpen_step_and_the_reference_blend_on_the_device_equal_the_cpu() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    for (dev, s) in sessions("sharpen") {
        let t0 = Instant::now();
        let mut rng = Rng::new(0xDB00_0001);
        let (mut sh, mut rf) = (0usize, 0usize);
        for (w, h) in device_sizes() {
            let big = w * h > 10_000;
            for fmt in FMTS {
                for class in [0, 1] {
                    let original = wc::image(&mut rng, w, h, fmt, class);
                    let other = wc::image(&mut rng, w, h, fmt, class);
                    let frame = perturb(&mut rng, &original, &other);
                    let ts = if big { &positions()[..3] } else { &positions()[..] };
                    for &t in ts {
                        let mut want = frame.clone();
                        filters::sharpen_step(&mut want, &original, t);
                        let mut got = frame.clone();
                        let r = sharpen_step_gpu(&s, &mut got, &original, t).map(|_| got);
                        sh += check(&dev, &format!("sharpen {fmt:?} {w}x{h} class {class} t {t}"), &want, r);
                    }
                    for wa in [0.5f32, 0.25, 0.9] {
                        let mut want = Image::new(w, h, fmt);
                        crate::blend::weighted_blend_noflow_reference(&mut want, &original, wa, &frame);
                        let mut got = Image::new(w, h, fmt);
                        let r = weighted_blend_noflow_reference_gpu(&s, &mut got, &original, wa, &frame).map(|_| got);
                        rf += check(&dev, &format!("reference {fmt:?} {w}x{h} class {class} wa {wa}"), &want, r);
                    }
                }
            }
        }
        // refusals before anything runs: two shapes, no pixels
        let mut a = Image::new(4, 4, Fmt::U8);
        assert!(matches!(sharpen_step_gpu(&s, &mut a, &Image::new(4, 4, Fmt::U16), 0.5), Err(StageError::Unsupported(_))));
        assert!(matches!(sharpen_step_gpu(&s, &mut Image::new(0, 4, Fmt::U8), &Image::new(0, 4, Fmt::U8), 0.5), Err(StageError::Unsupported(_))));
        assert!(matches!(weighted_blend_noflow_reference_gpu(&s, &mut a, &Image::new(4, 4, Fmt::U8), 0.5, &Image::new(4, 3, Fmt::U8)), Err(StageError::Unsupported(_))));
        println!("{dev}: sharpen_step {sh}, weighted_blend_noflow_reference {rf} samples identical in {:.1} s", t0.elapsed().as_secs_f64());
        assert!(sh >= MILLION && rf >= MILLION, "{dev}: {sh} / {rf} samples");
    }
}

#[test]
fn weighted_tail_and_weighted_blend_smart_on_the_device_equal_the_cpu() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    for (dev, s) in sessions("weighted smart") {
        let t0 = Instant::now();
        let mut rng = Rng::new(0xDB00_0002);
        let (mut ws, mut tl) = (0usize, 0usize);
        for (w, h) in device_sizes().into_iter().filter(|(w, h)| *w >= 2 && *h >= 2) {
            for fmt in FMTS {
                for class in [0, 1] {
                    let a = wc::image(&mut rng, w, h, fmt, class);
                    let b = wc::image(&mut rng, w, h, fmt, class);
                    let fab = wc::flow(&mut rng, w, h, 3.0, true, false);
                    let fba = wc::flow(&mut rng, w, h, 3.0, true, false);
                    for wa in [0.5f32, 0.3, 0.8] {
                        let mut want = Image::new(w, h, fmt);
                        smartblend::weighted_blend_smart(&mut want, &a, &fba, &b, &fab, wa, SmartOpts::default()).unwrap();
                        let mut got = Image::new(w, h, fmt);
                        let r = weighted_blend_smart_gpu(&s, &mut got, &a, &fba, &b, &fab, wa, SmartOpts::default()).map(|_| got);
                        ws += check(&dev, &format!("weighted smart {fmt:?} {w}x{h} class {class} wa {wa}"), &want, r);
                    }
                    let ta = perturb(&mut rng, &a, &b);
                    let tb = perturb(&mut rng, &b, &a);
                    let c1 = Coverage { width: w, height: h, data: (0..w * h).map(|_| if rng.below(5) == 0 { 0 } else { 255 }).collect() };
                    let c2 = Coverage { width: w, height: h, data: (0..w * h).map(|_| if rng.below(5) == 0 { 0 } else { 255 }).collect() };
                    for with_cov in [false, true] {
                        let wa = 0.4f32;
                        let (mut out, mut ta1, mut tb1, mut cv1) = (Image::new(w, h, fmt), ta.clone(), tb.clone(), c1.clone());
                        smartblend::weighted_tail(&mut out, &mut ta1, &a, &mut tb1, &b, wa, with_cov.then_some((&mut cv1, &c2)), SmartOpts::default()).unwrap();
                        let (mut gout, mut gta, mut gtb, mut gc1) = (Image::new(w, h, fmt), ta.clone(), tb.clone(), c1.clone());
                        let r = weighted_tail_gpu(&s, &mut gout, &mut gta, &a, &mut gtb, &b, wa, with_cov.then_some((&mut gc1, &c2)), SmartOpts::default());
                        let what = format!("tail {fmt:?} {w}x{h} class {class} cov {with_cov}");
                        match r {
                            Ok(()) => {
                                tl += check(&dev, &what, &out, Ok(gout));
                                tl += check(&dev, &what, &ta1, Ok(gta));
                                tl += check(&dev, &what, &tb1, Ok(gtb));
                                if with_cov {
                                    assert_eq!(gc1, cv1, "{dev} {what}: holes");
                                }
                            }
                            Err(StageError::NanOutput) => assert!(has_nan(&out) || has_nan(&ta1) || has_nan(&tb1), "{dev} {what}: NanOutput without a NaN"),
                            Err(e) => panic!("{dev} {what}: {e}"),
                        }
                    }
                }
            }
        }
        let mut o = Image::new(4, 4, Fmt::U8);
        let f = wc::flow(&mut rng, 4, 4, 1.0, false, false);
        assert!(matches!(weighted_blend_smart_gpu(&s, &mut o, &Image::new(4, 4, Fmt::U8), &f, &Image::new(4, 4, Fmt::F32), &f, 0.5, SmartOpts::default()), Err(StageError::Unsupported(_))));
        println!("{dev}: weighted_blend_smart {ws}, weighted_tail {tl} samples identical in {:.1} s", t0.elapsed().as_secs_f64());
        assert!(ws >= MILLION && tl >= MILLION, "{dev}: {ws} / {tl} samples");
    }
}

#[test]
fn flow_dissolve_on_the_device_equals_the_cpu() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    for (dev, s) in sessions("flow dissolve") {
        let t0 = Instant::now();
        let mut rng = Rng::new(0xDB00_0003);
        let mut n = 0usize;
        for (w, h) in device_sizes().into_iter().filter(|(w, h)| *w >= 2 && *h >= 2) {
            for fmt in FMTS {
                let a = wc::image(&mut rng, w, h, fmt, 0);
                let noise = wc::image(&mut rng, w, h, fmt, 1);
                let b = perturb(&mut rng, &a, &noise);
                let fab = wc::flow(&mut rng, w, h, 2.5, true, false);
                let fba = wc::flow(&mut rng, w, h, 2.5, true, false);
                for mode in [0u32, 5, 6, 2] {
                    for (ab, ba) in [(true, true), (false, false)] {
                        for wa in [0.5f32, 0.001, 0.3] {
                            let (fa, fb) = (ab.then_some(&fab), ba.then_some(&fba));
                            let mut want = Image::new(w, h, fmt);
                            let r = smartblend::flow_dissolve_inv(&mut want, &a, fa, &b, fb, wa, mode);
                            let mut got = Image::new(w, h, fmt);
                            let g = flow_dissolve_inv_gpu(&s, &mut got, &a, fa, &b, fb, wa, mode).map(|_| got);
                            let what = format!("dissolve {fmt:?} {w}x{h} mode {mode} flows {ab} wa {wa}");
                            match (r, g) {
                                (Err(_), Err(StageError::Unsupported(_))) => {}
                                (Ok(()), g) => n += check(&dev, &what, &want, g),
                                (r, g) => panic!("{dev} {what}: cpu {r:?}, device {:?}", g.err()),
                            }
                        }
                    }
                }
            }
        }
        println!("{dev}: flow_dissolve_inv {n} samples identical in {:.1} s", t0.elapsed().as_secs_f64());
        assert!(n >= MILLION, "{dev}: {n} samples");
    }
}

// ---- timing probe -------------------------------------------------------------------------------

fn best(mut f: impl FnMut()) -> f64 {
    (0..3)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1000.0
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
#[ignore = "timing probe: run with --ignored --nocapture"]
fn timing_probe_1080p() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let (w, h) = (1920usize, 1080usize);
    let mut rng = Rng::new(0x7134);
    for fmt in [Fmt::U8, Fmt::F32] {
        let a = wc::image(&mut rng, w, h, fmt, 0);
        let noise = wc::image(&mut rng, w, h, fmt, 0);
                let b = perturb(&mut rng, &a, &noise);
        let fab = wc::flow(&mut rng, w, h, 6.0, false, false);
        let fba = wc::flow(&mut rng, w, h, 6.0, false, false);
        let opts = SmartOpts::default();
        let mut rows: Vec<(String, f64)> = Vec::new();
        rows.push(("sharpen_step".into(), best(|| filters::sharpen_step(&mut b.clone(), &a, 0.5))));
        rows.push(("weighted_blend_noflow_reference".into(), best(|| crate::blend::weighted_blend_noflow_reference(&mut a.clone(), &a, 0.5, &b))));
        rows.push(("weighted_blend_smart".into(), best(|| smartblend::weighted_blend_smart(&mut a.clone(), &a, &fba, &b, &fab, 0.5, opts).unwrap())));
        rows.push(("flow_dissolve_inv mode 5".into(), best(|| smartblend::flow_dissolve_inv(&mut a.clone(), &a, Some(&fab), &b, Some(&fba), 0.5, 5).unwrap())));
        rows.push(("flow_dissolve_inv mode 6".into(), best(|| smartblend::flow_dissolve_inv(&mut a.clone(), &a, None, &b, None, 0.5, 6).unwrap())));
        for (name, t) in &rows {
            println!("timing {fmt:?} 1080p | cpu | {name} | {t:.1} ms");
        }
        for (dev, s) in sessions("timing") {
            let g = [
                ("sharpen_step", best(|| sharpen_step_gpu(&s, &mut b.clone(), &a, 0.5).unwrap())),
                ("weighted_blend_noflow_reference", best(|| weighted_blend_noflow_reference_gpu(&s, &mut a.clone(), &a, 0.5, &b).unwrap())),
                ("weighted_blend_smart", best(|| weighted_blend_smart_gpu(&s, &mut a.clone(), &a, &fba, &b, &fab, 0.5, opts).unwrap())),
                ("flow_dissolve_inv mode 5", best(|| flow_dissolve_inv_gpu(&s, &mut a.clone(), &a, Some(&fab), &b, Some(&fba), 0.5, 5).unwrap())),
                ("flow_dissolve_inv mode 6", best(|| flow_dissolve_inv_gpu(&s, &mut a.clone(), &a, None, &b, None, 0.5, 6).unwrap())),
            ];
            for (name, t) in g {
                println!("timing {fmt:?} 1080p | {dev} | {name} | {t:.1} ms (transfers included)");
            }
        }
    }
}
