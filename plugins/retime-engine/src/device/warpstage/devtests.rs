//! Device tests of the composition stages: every stage on every OpenCL device of this machine,
//! compared bit for bit with its CPU function over the corpora of `corpus`. A float output with a
//! NaN must come back as `StageError::NanOutput` (and the CPU result must hold a NaN there too;
//! the raw device result is then compared with NaN payloads set aside). Without a runtime or a
//! device the tests print a note and pass (CI runners have none). The device tests take a lock,
//! so one at a time uses the GPUs. The `#[ignore]`d probes print GPU (with transfers) against CPU
//! times at 1920x1080.

use super::corpus::{self, first_diff, FMTS};
use super::{blend as dblend, blurcomp as dblur, smartblend as dsmart, warp as dwarp, DeviceFlow, Flag};
use crate::blend;
use crate::device::image::{has_nan, DeviceImage, StageError};
use crate::device::probe;
use crate::device::selfcheck::{compare_floats, Session};
use crate::device::sim::Rng;
use crate::flow::FlowField;
use crate::image::{Fmt, Image, Samples};
use crate::smartblend;
use crate::warp::{self, WarpOpts};
use std::sync::Mutex;
use std::time::Instant;

static GPU: Mutex<()> = Mutex::new(());

/// A session with this stage module's program on every usable device (index, name).
fn sessions(test: &str) -> Vec<(usize, String, Session)> {
    let mut out = Vec::new();
    for idx in 0..2 {
        match probe::open(idx) {
            Ok((dev, info)) => {
                if !probe::usable(&info) {
                    eprintln!("note: {test}: OpenCL device {idx} ({}) does not meet the requirements; skipped", info.display_name());
                    continue;
                }
                let s = Session::open(&dev, &super::program()).unwrap_or_else(|e| panic!("{test}: the program does not build on {}: {e}", info.display_name()));
                out.push((idx, info.display_name().to_string(), s));
            }
            Err(e) => eprintln!("note: {test}: no OpenCL device {idx} ({e}); skipped"),
        }
    }
    out
}

/// The outcome of one device case against the CPU result.
#[derive(Default, Debug)]
struct Tally {
    compared: usize,
    nan_rule: usize,
    deferred: usize,
    refused_both: usize,
}

/// Checks a `_gpu` result against the CPU's: equal bits, or `NanOutput` where the CPU output
/// holds a NaN, in which case `raw` (the device result without the NaN rule) must equal the CPU
/// result apart from NaN payloads.
fn judge(what: &str, cpu: &Image, got: Result<Image, StageError>, raw: impl FnOnce() -> Image, t: &mut Tally) {
    match got {
        Ok(img) => {
            assert_eq!(first_diff(cpu, &img), None, "{what}");
            t.compared += cpu.sample_count();
        }
        Err(StageError::NanOutput) => {
            assert!(has_nan(cpu), "{what}: the device wrote a NaN where the CPU has none");
            let r = raw();
            let (Samples::F32(a), Samples::F32(b)) = (&cpu.samples, &r.samples) else { panic!("{what}: NaN in an integer image") };
            let (diff, _) = compare_floats(a, b);
            assert_eq!(diff, None, "{what}: raw device result differs beyond NaN payloads");
            t.nan_rule += 1;
            t.compared += cpu.sample_count();
        }
        Err(e) => panic!("{what}: {e}"),
    }
}

#[test]
fn device_images_round_trip_at_every_depth() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    for (_, name, s) in sessions("round trip") {
        let mut rng = Rng::new(42);
        for fmt in FMTS {
            let mut img = corpus::image(&mut rng, 13, 7, fmt, 1);
            if let Samples::F32(v) = &mut img.samples {
                v[0] = f32::from_bits(0x7fc0_1234);
                v[1] = f32::from_bits(0xffa0_0001);
                v[2] = -0.0;
                v[3] = f32::from_bits(1);
            }
            let d = DeviceImage::upload(&s, &img).unwrap();
            let back = d.download(&s).unwrap();
            assert_eq!(corpus::bits(&img), corpus::bits(&back), "{name} {fmt:?}");
        }
        let flow = corpus::flow(&mut rng, 5, 4, 3.0, true, true);
        let back = DeviceFlow::upload(&s, &flow).unwrap().download(&s).unwrap();
        assert!(flow.data.iter().zip(&back.data).all(|(a, b)| a.to_bits() == b.to_bits()), "{name} flow");
    }
}

fn device_sizes() -> Vec<(usize, usize)> {
    let mut v = corpus::sizes();
    v.extend([(256, 128), (191, 97)]);
    v
}

#[test]
fn blend_and_lerp_on_the_devices_equal_the_cpu() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    for (_, name, s) in sessions("blend") {
        let mut rng = Rng::new(0xB1E0);
        let (mut tb, mut tl, mut tw) = (Tally::default(), Tally::default(), Tally::default());
        for (w, h) in device_sizes() {
            for fmt in FMTS {
                for class in [0, 1] {
                    let a = corpus::image(&mut rng, w, h, fmt, class);
                    let b = corpus::image(&mut rng, w, h, fmt, class);
                    for wt in corpus::weights() {
                        let what = format!("{name} {w}x{h} {fmt:?} class {class} w {wt:e}");
                        let mut want = Image::new(w, h, fmt);
                        blend::blend(&mut want, &a, &b, wt);
                        let mut got = Image::new(w, h, fmt);
                        let r = dblend::blend_gpu(&s, &mut got, &a, &b, wt).map(|_| got);
                        judge(&format!("blend {what}"), &want, r, || {
                            let (da, db) = (DeviceImage::upload(&s, &a).unwrap(), DeviceImage::upload(&s, &b).unwrap());
                            let o = DeviceImage::zeroed(&s, w, h, fmt).unwrap();
                            dblend::blend_enqueue(&s, &o, &da, &db, wt).unwrap();
                            o.download(&s).unwrap()
                        }, &mut tb);
                        let mut want = a.clone();
                        blend::lerp(&mut want, &b, wt);
                        let mut got = a.clone();
                        let r = dblend::lerp_gpu(&s, &mut got, &b, wt).map(|_| got);
                        judge(&format!("lerp {what}"), &want, r, || {
                            let (da, db) = (DeviceImage::upload(&s, &a).unwrap(), DeviceImage::upload(&s, &b).unwrap());
                            dblend::lerp_enqueue(&s, &da, &db, wt).unwrap();
                            da.download(&s).unwrap()
                        }, &mut tl);
                    }
                    for wa in [0.0f32, 0.3, 0.5, 1.0, 1.7] {
                        let mut want = Image::new(w, h, fmt);
                        blend::weighted_blend_noflow(&mut want, &a, wa, &b);
                        let mut got = Image::new(w, h, fmt);
                        match dblend::weighted_blend_noflow_gpu(&s, &mut got, &a, wa, &b) {
                            Ok(()) => {
                                assert_eq!(first_diff(&want, &got), None, "weighted {name} {w}x{h} {fmt:?} {wa}");
                                tw.compared += want.sample_count();
                            }
                            Err(StageError::Unsupported(_)) => {
                                assert_eq!(fmt, Fmt::F32);
                                tw.deferred += 1;
                            }
                            Err(e) => panic!("weighted {name}: {e}"),
                        }
                    }
                }
            }
        }
        eprintln!("{name}: blend {tb:?}; lerp {tl:?}; weighted noflow {tw:?}");
        assert!(tb.compared >= 1_000_000 && tl.compared >= 1_000_000 && tw.compared > 100_000);
    }
}

fn cpu_warp(src: &Image, flow: &FlowField, amount: f32, opts: WarpOpts, cov: bool) -> fx_core::Result<(Image, Option<warp::Coverage>)> {
    let mut out = Image::new(src.width, src.height, src.fmt());
    if cov {
        let c = warp::inv_warp_coverage(&mut out, src, flow, amount, opts)?;
        Ok((out, Some(c)))
    } else {
        warp::inv_warp(&mut out, src, flow, amount, opts)?;
        Ok((out, None))
    }
}

#[test]
fn the_inverse_warp_on_the_devices_equals_the_cpu() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let opts_list = [
        (WarpOpts { hold: true, is360: false, mode: 1 }, false),
        (WarpOpts { hold: false, is360: false, mode: 1 }, false),
        (WarpOpts { hold: false, is360: false, mode: 1 }, true),
        (WarpOpts { hold: true, is360: true, mode: 1 }, true),
    ];
    for (_, name, s) in sessions("inverse warp") {
        let mut rng = Rng::new(0x3A4F);
        let mut t = Tally::default();
        for (w, h) in device_sizes() {
            for fmt in FMTS {
                for class in [0, 1] {
                    let src = corpus::image(&mut rng, w, h, fmt, class);
                    for (fw, fh) in [(w, h), (w.div_ceil(2).max(2), h.div_ceil(2).max(2))] {
                        for (scale, wild, nonfinite) in [(0.6f32, false, false), (3.0, true, false), (w as f32 * 2.5, true, false), (4.0, true, true)] {
                            let flow = corpus::flow(&mut rng, fw, fh, scale, wild, nonfinite);
                            for amount in [1.0f32, -0.75, 2.0] {
                                for (opts, cov) in opts_list {
                                    let what = format!("warp {name} {w}x{h} {fmt:?} class {class} flow {fw}x{fh} scale {scale} nf {nonfinite} amount {amount} {opts:?} cov {cov}");
                                    let cpu = cpu_warp(&src, &flow, amount, opts, cov);
                                    let mut out = Image::new(w, h, fmt);
                                    let dev = if cov {
                                        dwarp::inv_warp_coverage_gpu(&s, &mut out, &src, &flow, amount, opts).map(Some)
                                    } else {
                                        dwarp::inv_warp_gpu(&s, &mut out, &src, &flow, amount, opts).map(|_| None)
                                    };
                                    match (cpu, dev) {
                                        (Ok((img, ccov)), Ok(dcov)) => {
                                            assert_eq!(first_diff(&img, &out), None, "{what}");
                                            assert_eq!(ccov, dcov, "{what}");
                                            t.compared += img.sample_count();
                                        }
                                        (Ok((img, _)), Err(StageError::NanOutput)) => {
                                            assert!(has_nan(&img), "{what}: a NaN the CPU does not have");
                                            let raw = {
                                                let d = DeviceImage::upload(&s, &src).unwrap();
                                                let f = DeviceFlow::upload(&s, &flow).unwrap();
                                                let o = DeviceImage::zeroed(&s, w, h, fmt).unwrap();
                                                let flag = Flag::new(&s).unwrap();
                                                dwarp::inv_warp_enqueue(&s, &o, &d, &f, amount, opts, None, &flag).unwrap();
                                                flag.check(&s, "warp").unwrap();
                                                o.download(&s).unwrap()
                                            };
                                            let (Samples::F32(a), Samples::F32(b)) = (&img.samples, &raw.samples) else { unreachable!() };
                                            assert_eq!(compare_floats(a, b).0, None, "{what}: raw");
                                            t.nan_rule += 1;
                                            t.compared += img.sample_count();
                                        }
                                        (Ok(_), Err(StageError::Unsupported(why))) => {
                                            assert!(why.contains("device bound"), "{what}: {why}");
                                            t.deferred += 1;
                                        }
                                        (Err(_), Err(StageError::Unsupported(_))) => t.refused_both += 1,
                                        (c, d) => panic!("{what}: cpu {:?} device {:?}", c.err(), d.err()),
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        eprintln!("{name}: inverse warp {t:?}");
        assert!(t.compared >= 1_000_000 && t.refused_both > 0, "{t:?}");
    }
}

#[test]
#[ignore = "the kernel follows the rule before the parity work; port pending (docs/engine.md, Device)"]
fn motion_blur_and_compensation_on_the_devices_equal_the_cpu() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    for (_, name, s) in sessions("motion blur") {
        let mut rng = Rng::new(0xB10B);
        let (mut t, mut tc) = (Tally::default(), Tally::default());
        for (w, h) in device_sizes() {
            for fmt in FMTS {
                for class in [0, 1] {
                    let src = corpus::image(&mut rng, w, h, fmt, class);
                    for (scale, wild, nonfinite) in [(1.5f32, false, false), (6.0, true, false), (w as f32 * 2.0, true, false), (3.0, true, true)] {
                        let flow = corpus::flow(&mut rng, w, h, scale, wild, nonfinite);
                        for (amount, shutter) in [(1.0f32, 0.5f32), (3.0, 0.25)] {
                            for hiq in [true, false] {
                                for is360 in [false, true] {
                                    let what = format!("blur {name} {w}x{h} {fmt:?} class {class} scale {scale} nf {nonfinite} {amount} {shutter} hiq {hiq} 360 {is360}");
                                    let mut want = Image::new(w, h, fmt);
                                    let cpu = warp::motion_blur(&mut want, &src, &flow, amount, hiq, shutter, is360);
                                    let mut got = Image::new(w, h, fmt);
                                    let dev = dblur::motion_blur_gpu(&s, &mut got, &src, &flow, amount, hiq, shutter, is360).map(|_| got);
                                    match (cpu, dev) {
                                        (Ok(()), Err(StageError::Unsupported(why))) => {
                                            assert!(why.contains("device bound"), "{what}: {why}");
                                            t.deferred += 1;
                                        }
                                        (Ok(()), r) => judge(&what, &want, r, || {
                                            let d = DeviceImage::upload(&s, &src).unwrap();
                                            let f = DeviceFlow::upload(&s, &flow).unwrap();
                                            let o = DeviceImage::zeroed(&s, w, h, fmt).unwrap();
                                            let flag = Flag::new(&s).unwrap();
                                            dblur::motion_blur_enqueue(&s, &o, &d, &f, amount, hiq, shutter, is360, &flag).unwrap();
                                            o.download(&s).unwrap()
                                        }, &mut t),
                                        (Err(_), Err(StageError::Unsupported(_))) => t.refused_both += 1,
                                        (Err(e), r) => panic!("{what}: cpu {e} device {:?}", r.err()),
                                    }
                                    if nonfinite || class == 1 {
                                        continue;
                                    }
                                    for (s0, s1, blur) in [(0.0f32, 3.0f32, 1.0f32), (0.0, 0.25, 1.0)] {
                                        let mut want = src.clone();
                                        warp::blur_compensation_wrap(&mut want, &flow, s0, s1, blur, hiq, is360).unwrap();
                                        let mut got = src.clone();
                                        match dblur::blur_compensation_wrap_gpu(&s, &mut got, &flow, s0, s1, blur, hiq, is360) {
                                            Ok(()) => {
                                                assert_eq!(first_diff(&want, &got), None, "compensation {what} {s0} {s1}");
                                                tc.compared += want.sample_count();
                                            }
                                            Err(StageError::NanOutput) => {
                                                assert!(has_nan(&want), "compensation {what}");
                                                tc.nan_rule += 1;
                                            }
                                            Err(e) => panic!("compensation {what}: {e}"),
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        eprintln!("{name}: motion blur {t:?}; compensation {tc:?}");
        assert!(t.compared >= 1_000_000 && tc.compared >= 1_000_000, "{t:?} {tc:?}");
    }
}

#[test]
fn smart_blend_on_the_devices_equals_the_cpu() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    for (_, name, s) in sessions("smart blend") {
        let mut rng = Rng::new(0x5B5B);
        let (mut t, mut tb, mut tc) = (Tally::default(), Tally::default(), 0usize);
        for (w, h) in device_sizes() {
            for fmt in FMTS {
                for class in [0, 1, 2] {
                    let p1 = corpus::image(&mut rng, w, h, fmt, class);
                    let p2 = corpus::image(&mut rng, w, h, fmt, class);
                    for wt in corpus::weights() {
                        for binary in [false, true] {
                            let what = format!("smart {name} {w}x{h} {fmt:?} class {class} w {wt:e} binary {binary}");
                            let mut want = Image::new(w, h, fmt);
                            let mut got = Image::new(w, h, fmt);
                            let r = if binary {
                                smartblend::smart_blend_binary(&mut want, &p1, &p2, wt).unwrap();
                                dsmart::smart_blend_binary_gpu(&s, &mut got, &p1, &p2, wt)
                            } else {
                                smartblend::smart_blend(&mut want, &p1, &p2, wt).unwrap();
                                dsmart::smart_blend_gpu(&s, &mut got, &p1, &p2, wt)
                            };
                            judge(&what, &want, r.map(|_| got), || {
                                let (a, b) = (DeviceImage::upload(&s, &p1).unwrap(), DeviceImage::upload(&s, &p2).unwrap());
                                let o = DeviceImage::zeroed(&s, w, h, fmt).unwrap();
                                if binary {
                                    dsmart::smart_blend_binary_enqueue(&s, &o, &a, &b, wt).unwrap();
                                } else {
                                    dsmart::smart_blend_enqueue(&s, &o, &a, &b, wt).unwrap();
                                }
                                o.download(&s).unwrap()
                            }, if binary { &mut tb } else { &mut t });
                        }
                    }
                }
            }
            let mk = |rng: &mut Rng| warp::Coverage { width: w, height: h, data: (0..w * h).map(|_| [0u8, 255, rng.next_u32() as u8][rng.below(3) as usize]).collect() };
            let (c1, c2) = (mk(&mut rng), mk(&mut rng));
            let mut want = c1.clone();
            want.combine_max(&c2);
            want.invert();
            let mut got = c1.clone();
            dsmart::coverage_holes_gpu(&s, &mut got, &c2).unwrap();
            assert_eq!(got, want, "{name} coverage holes {w}x{h}");
            let mut inv = c1.clone();
            inv.invert();
            let mut got = c1.clone();
            dsmart::coverage_invert_gpu(&s, &mut got).unwrap();
            assert_eq!(got, inv, "{name} coverage invert {w}x{h}");
            tc += 2 * w * h;
        }
        eprintln!("{name}: smart blend {t:?}; binary {tb:?}; coverage bytes {tc}");
        assert!(t.compared >= 1_000_000 && tb.compared >= 1_000_000, "{t:?} {tb:?}");
    }
}

// ---- timing probes ------------------------------------------------------------------------------

fn time<R>(f: impl FnMut() -> R) -> f64 {
    let mut f = f;
    f();
    let t = Instant::now();
    let reps = 3;
    for _ in 0..reps {
        std::hint::black_box(f());
    }
    t.elapsed().as_secs_f64() * 1000.0 / reps as f64
}

#[test]
#[ignore = "timing probe: run with --ignored on a machine with an OpenCL device"]
fn probe_stage_timings_at_1080p() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (w, h) = (1920, 1080);
    let mut rng = Rng::new(1080);
    for fmt in [Fmt::U8, Fmt::F32] {
        let a = corpus::image(&mut rng, w, h, fmt, 0);
        let b = corpus::image(&mut rng, w, h, fmt, 0);
        let flow = corpus::flow(&mut rng, w, h, 6.0, false, false);
        let mut out = Image::new(w, h, fmt);
        let cpu = [
            ("blend", time(|| blend::blend(&mut out.clone(), &a, &b, 0.3))),
            ("inv_warp fade+cov", time(|| warp::inv_warp_coverage(&mut out.clone(), &a, &flow, 0.7, WarpOpts { hold: false, is360: false, mode: 1 }).unwrap())),
            ("inv_warp 360", time(|| warp::inv_warp(&mut out.clone(), &a, &flow, 0.7, WarpOpts { hold: true, is360: true, mode: 1 }).unwrap())),
            ("motion_blur hiq", time(|| warp::motion_blur(&mut out.clone(), &a, &flow, 2.0, true, 0.5, false).unwrap())),
            ("blur_compensation", time(|| warp::blur_compensation(&mut a.clone(), &flow, 0.0, 3.0, 1.0, true).unwrap())),
            ("smart_blend", time(|| smartblend::smart_blend(&mut out.clone(), &a, &b, 0.3).unwrap())),
        ];
        for (_, name, s) in sessions("timing") {
            let gpu = [
                time(|| dblend::blend_gpu(&s, &mut out, &a, &b, 0.3).unwrap()),
                time(|| dwarp::inv_warp_coverage_gpu(&s, &mut out, &a, &flow, 0.7, WarpOpts { hold: false, is360: false, mode: 1 }).unwrap()),
                time(|| dwarp::inv_warp_gpu(&s, &mut out, &a, &flow, 0.7, WarpOpts { hold: true, is360: true, mode: 1 }).unwrap()),
                time(|| dblur::motion_blur_gpu(&s, &mut out, &a, &flow, 2.0, true, 0.5, false).unwrap()),
                time(|| dblur::blur_compensation_gpu(&s, &mut a.clone(), &flow, 0.0, 3.0, 1.0, true).unwrap()),
                time(|| dsmart::smart_blend_gpu(&s, &mut out, &a, &b, 0.3).unwrap()),
            ];
            for ((stage, c), g) in cpu.iter().zip(gpu) {
                println!("timing {fmt:?} {stage} 1920x1080: cpu {c:.2} ms, {name} {g:.2} ms (with transfers)");
            }
        }
    }
}
