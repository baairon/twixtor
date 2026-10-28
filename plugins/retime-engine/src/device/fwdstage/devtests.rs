//! Device tests of the forward warp stage: whole warps on every OpenCL device of this machine,
//! compared bit for bit with `forward::forward_warp_with` over flows the CPU solver computed for
//! synthetic moving scenes, the adversarial mesh classes of the device-free tests and random
//! corpora. Without a runtime or a device the tests print a note and pass. One test at a time
//! uses the GPUs (a lock). `forward_warp_timings_at_1080p` (ignored) prints CPU against device
//! times per device and depth.

use super::tests::{adversarial_flow, cpu_warp, judge, judge_panic, Tally, FLOW_CLASSES};
use super::{forward_warp_enqueue, forward_warp_with_gpu, plan, DevicePlan};
use crate::device::image::{DeviceImage, StageError};
use crate::device::probe;
use crate::device::selfcheck::Session;
use crate::device::sim::Rng;
use crate::device::warpstage::corpus::{self, FMTS};
use crate::device::warpstage::{DeviceCoverage, Flag};
use crate::flow::{self, FlowField, FlowSettings};
use crate::forward::{self, ForwardOpts};
use crate::image::{Fmt, Image, Samples};
use std::sync::Mutex;
use std::time::Instant;

static GPU: Mutex<()> = Mutex::new(());

fn sessions(test: &str) -> Vec<(usize, String, Session)> {
    let mut out = Vec::new();
    for idx in 0..2 {
        match probe::open(idx) {
            Ok((dev, info)) => {
                if !probe::usable(&info) {
                    println!("note: {test}: OpenCL device {idx} ({}) does not meet the requirements; skipped", info.display_name());
                    continue;
                }
                let s = Session::open(&dev, &super::program()).unwrap_or_else(|e| panic!("{test}: the program does not build on {}: {e}", info.display_name()));
                out.push((idx, info.display_name().to_string(), s));
            }
            Err(e) => println!("note: {test}: no OpenCL device {idx} ({e}); skipped"),
        }
    }
    out
}

/// A frame of a moving scene: a smooth background panned by `pan` and a textured square of side
/// `side` at `(sx, sy)`, in 0..255 (8 bpc), 0..65535 (16) or 0..1 (float), alpha opaque.
pub(crate) fn scene(w: usize, h: usize, fmt: Fmt, pan: f32, sx: f32, sy: f32, side: f32) -> Image {
    let mut v = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            let (xf, yf) = (x as f32, y as f32);
            let inside = xf >= sx && xf < sx + side && yf >= sy && yf < sy + side;
            v.push(1.0f32);
            for ch in 1..4 {
                let c = ch as f32;
                let t = if inside {
                    0.5 + 0.35 * ((xf - sx) * 0.21 + c * 0.7).cos() * ((yf - sy) * 0.17).sin()
                } else {
                    let u = xf - pan;
                    0.5 + 0.28 * (u * 0.037 + c).sin() + 0.2 * (yf * 0.029 + u * 0.005).cos()
                };
                v.push(t.clamp(0.0, 1.0));
            }
        }
    }
    let samples = match fmt {
        Fmt::U8 => Samples::U8(v.iter().map(|t| (t * 255.0).round() as u8).collect()),
        Fmt::U16 => Samples::U16(v.iter().map(|t| (t * 65535.0).round() as u16).collect()),
        Fmt::F32 => Samples::F32(v),
    };
    Image { width: w, height: h, samples }
}

/// Frame pairs of moving scenes with the CPU solver's flows (`solve_pair`, default settings).
fn solved_corpus() -> Vec<(usize, usize, [f32; 6], FlowField, FlowField)> {
    let mut out = Vec::new();
    let motions: [[f32; 6]; 4] = [
        [0.0, 3.0, 10.0, 8.0, 18.0, 9.0],
        [2.0, 5.0, 6.0, 14.0, 4.0, 12.0],
        [-3.0, 30.0, 12.0, 4.0, 12.0, 20.0],
        [1.5, 9.0, 3.0, 20.0, 9.5, 16.0],
    ];
    for &(w, h) in &[(48usize, 32usize), (96, 64), (160, 96)] {
        for m in motions {
            // m: pan of B, square x in A, square y in A, square x in B, square y in B, side
            let a = scene(w, h, Fmt::U8, 0.0, m[1], m[2], m[5]);
            let b = scene(w, h, Fmt::U8, m[0], m[3], m[4], m[5]);
            let pair = flow::solve_pair(&a, &b, &FlowSettings::default()).expect("the CPU solver");
            out.push((w, h, m, pair.f_ab, pair.f_ba));
        }
    }
    out
}

#[test]
fn device_forward_warps_match_the_cpu() {
    let solved = solved_corpus();
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    for (idx, name, s) in sessions("device forward warps") {
        let t0 = Instant::now();
        let mut t = Tally::default();
        let mut i = 0usize;
        // real flows: both frames of every pair at three amounts and three depths
        for (w, h, m, f_ab, f_ba) in &solved {
            for fmt in FMTS {
                let a = scene(*w, *h, fmt, 0.0, m[1], m[2], m[5]);
                let b = scene(*w, *h, fmt, m[0], m[3], m[4], m[5]);
                for amount in [0.25f32, 0.5, 1.0] {
                    for (img, fl) in [(&a, f_ab), (&b, f_ba)] {
                        i += 1;
                        let want_cov = i % 2 == 0;
                        let mut cpu_out = Image::new(*w, *h, fmt);
                        let cpu = forward::forward_warp_with(&mut cpu_out, img, fl, amount, ForwardOpts::default(), want_cov);
                        let mut out = Image::new(*w, *h, fmt);
                        let got = forward_warp_with_gpu(&s, &mut out, img, fl, amount, ForwardOpts::default(), want_cov).map(|c| (out, c));
                        judge(&format!("{name}: solved {w}x{h} {fmt:?} {m:?} amount {amount}"), &cpu, &cpu_out, got, || None, &mut t);
                    }
                }
            }
        }
        let real = t.compared;
        // adversarial meshes
        let mut rng = Rng::new(0xD_E71CE + idx as u64);
        for &(w, h) in &[(2usize, 2usize), (7, 7), (33, 20), (64, 48), (160, 120), (321, 181)] {
            for class in 0..FLOW_CLASSES {
                for fmt in FMTS {
                    let src = corpus::image(&mut rng, w, h, fmt, if class % 4 == 3 { 1 } else { 0 });
                    let (fl, amount) = adversarial_flow(&mut rng, w, h, class);
                    let want_cov = class % 2 == 0;
                    let mut out = Image::new(w, h, fmt);
                    let got = forward_warp_with_gpu(&s, &mut out, &src, &fl, amount, ForwardOpts::default(), want_cov).map(|c| (out, c));
                    let what = format!("{name}: adversarial {w}x{h} {fmt:?} class {class} amount {amount}");
                    match cpu_warp(&src, &fl, amount, want_cov) {
                        Some((cpu, cpu_out)) => judge(&what, &cpu, &cpu_out, got, || None, &mut t),
                        None => judge_panic(&what, got, &mut t),
                    }
                }
            }
        }
        // random corpora
        for j in 0..60 {
            let (w, h) = corpus::sizes()[j % corpus::sizes().len()];
            let fmt = FMTS[j % 3];
            let src = corpus::image(&mut rng, w, h, fmt, (j % 2) as u32);
            let fl = corpus::flow(&mut rng, w.max(3) - 1, h.max(3) - 1, [0.5f32, 2.0, 6.0][j % 3], j % 2 == 0, j % 5 == 0);
            let amount = [1.0f32, 0.5, -0.5, 1.7][j % 4];
            let mut out = Image::new(w, h, fmt);
            let got = forward_warp_with_gpu(&s, &mut out, &src, &fl, amount, ForwardOpts::default(), j % 2 == 1).map(|c| (out, c));
            let what = format!("{name}: random #{j} {w}x{h} {fmt:?}");
            match cpu_warp(&src, &fl, amount, j % 2 == 1) {
                Some((cpu, cpu_out)) => judge(&what, &cpu, &cpu_out, got, || None, &mut t),
                None => judge_panic(&what, got, &mut t),
            }
        }
        println!("{name}: {} whole warps compared bit for bit ({real} on solver flows), {:?} in {:.2} s", t.compared, t, t0.elapsed().as_secs_f64());
        assert!(t.compared >= 200, "{name}: {t:?}");
    }
}

fn best_ms(runs: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut best = f64::MAX;
    for _ in 0..runs {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn convert(img: &Image, fmt: Fmt) -> Image {
    let Samples::U8(v) = &img.samples else { panic!("an 8 bpc frame") };
    let samples = match fmt {
        Fmt::U8 => Samples::U8(v.clone()),
        Fmt::U16 => Samples::U16(v.iter().map(|x| *x as u16 * 257).collect()),
        Fmt::F32 => Samples::F32(v.iter().map(|x| *x as f32 / 255.0).collect()),
    };
    Image { width: img.width, height: img.height, samples }
}

#[test]
#[ignore]
fn forward_warp_timings_at_1080p() {
    let (w, h) = (1920usize, 1080usize);
    let a = scene(w, h, Fmt::U8, 0.0, 600.0, 300.0, 360.0);
    let b = scene(w, h, Fmt::U8, 4.0, 640.0, 310.0, 360.0);
    let t = Instant::now();
    let typical = flow::solve_pair(&a, &b, &FlowSettings::default()).expect("solver").f_ab;
    println!("typical flow: the CPU solver at 1080p, {}x{} vectors, {:.0} ms", typical.width, typical.height, t.elapsed().as_secs_f64() * 1e3);
    // stress: a zoom with a rotation over the whole frame, and a fold of a third of the frame
    let mut zoom = FlowField::new(w, h);
    let mut fold = FlowField::new(w, h);
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let i = (y * w + x) * 2;
            zoom.data[i] = dx * -0.2 - dy * 0.15;
            zoom.data[i + 1] = dx * 0.15 + dy * -0.2;
            fold.data[i] = if x < w / 3 { 320.0 } else { -40.0 };
            fold.data[i + 1] = 6.0 * ((x as f32) * 0.01).sin();
        }
    }
    let flows = [("typical", &typical), ("zoom", &zoom), ("fold", &fold)];
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let sessions = sessions("forward warp timings");
    println!("| device | depth | flow | CPU ms | host mesh ms | device, transfers ms | device, resident ms | device kernels ms |");
    for fmt in FMTS {
        let src = convert(&a, fmt);
        for (fname, fl) in flows {
            let amount = 0.5f32;
            let mut cpu_out = Image::new(w, h, fmt);
            let cpu_ms = best_ms(3, || {
                forward::forward_warp_with(&mut cpu_out, &src, fl, amount, ForwardOpts::default(), true).unwrap();
            });
            let mesh_ms = best_ms(3, || {
                plan(&src, &src, fl, amount, ForwardOpts::default()).unwrap();
            });
            for (_, name, s) in &sessions {
                let mut out = Image::new(w, h, fmt);
                let check = forward_warp_with_gpu(s, &mut out, &src, fl, amount, ForwardOpts::default(), true);
                match &check {
                    Ok(_) => assert_eq!(corpus::first_diff(&cpu_out, &out), None, "{name} {fmt:?} {fname}: differs"),
                    Err(StageError::NanOutput) => {}
                    Err(e) => {
                        println!("| {name} | {fmt:?} | {fname} | {cpu_ms:.1} | {mesh_ms:.1} | refused: {e} | | |");
                        continue;
                    }
                }
                let full_ms = best_ms(3, || {
                    let _ = forward_warp_with_gpu(s, &mut out, &src, fl, amount, ForwardOpts::default(), true);
                });
                let dsrc = DeviceImage::upload(s, &src).unwrap();
                let dout = DeviceImage::zeroed(s, w, h, fmt).unwrap();
                let dcov = DeviceCoverage::filled(s, w, h, 0).unwrap();
                let flag = Flag::new(s).unwrap();
                s.queue.finish().unwrap();
                let resident_ms = best_ms(3, || {
                    let p = plan(&src, &src, fl, amount, ForwardOpts::default()).unwrap();
                    let dp = DevicePlan::upload(s, &p).unwrap();
                    forward_warp_enqueue(s, &dp, &dsrc, &dout, Some(&dcov), &flag).unwrap();
                    s.queue.finish().unwrap();
                });
                let p = plan(&src, &src, fl, amount, ForwardOpts::default()).unwrap();
                let dp = DevicePlan::upload(s, &p).unwrap();
                s.queue.finish().unwrap();
                let kernel_ms = best_ms(3, || {
                    forward_warp_enqueue(s, &dp, &dsrc, &dout, Some(&dcov), &flag).unwrap();
                    s.queue.finish().unwrap();
                });
                flag.check(s, "timing").unwrap();
                println!("| {name} | {fmt:?} | {fname} | {cpu_ms:.1} | {mesh_ms:.1} | {full_ms:.1} | {resident_ms:.1} | {kernel_ms:.1} |");
            }
        }
    }
}

#[test]
#[ignore]
fn forward_warp_kernel_time_at_4k() {
    // the whole device part of one warp (every launch together) at 3840x2160 under stress flows:
    // when the sum stays far below 200 ms, so does every single launch
    let (w, h) = (3840usize, 2160usize);
    let a = scene(w, h, Fmt::U8, 0.0, 1200.0, 600.0, 720.0);
    let mut zoom = FlowField::new(w / 2, h / 2);
    let mut fold = FlowField::new(w / 2, h / 2);
    let (cx, cy) = (w as f32 / 4.0, h as f32 / 4.0);
    for y in 0..h / 2 {
        for x in 0..w / 2 {
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let i = (y * (w / 2) + x) * 2;
            zoom.data[i] = dx * -0.2 - dy * 0.15;
            zoom.data[i + 1] = dx * 0.15 + dy * -0.2;
            fold.data[i] = if x < w / 6 { 320.0 } else { -40.0 };
            fold.data[i + 1] = 6.0 * ((x as f32) * 0.01).sin();
        }
    }
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    for (_, name, s) in &sessions("forward warp at 4K") {
        for fmt in [Fmt::U8, Fmt::F32] {
            let src = convert(&a, fmt);
            for (fname, fl) in [("zoom", &zoom), ("fold", &fold)] {
                let p = plan(&src, &src, fl, 1.0, ForwardOpts::default()).unwrap();
                let dp = DevicePlan::upload(s, &p).unwrap();
                let dsrc = DeviceImage::upload(s, &src).unwrap();
                let dout = DeviceImage::zeroed(s, w, h, fmt).unwrap();
                let flag = Flag::new(s).unwrap();
                s.queue.finish().unwrap();
                let ms = best_ms(2, || {
                    forward_warp_enqueue(s, &dp, &dsrc, &dout, None, &flag).unwrap();
                    s.queue.finish().unwrap();
                });
                let ok = flag.check(s, "4K").is_ok();
                println!("{name}: 4K {fmt:?} {fname}: all launches of one warp {ms:.1} ms (flag clear: {ok})");
            }
        }
    }
}
