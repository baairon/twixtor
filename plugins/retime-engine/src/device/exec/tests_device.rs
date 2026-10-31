//! The executor on the real devices of this machine (every OpenCL device that meets the
//! requirements). Each test prints a note and passes when there is none, as on CI.

use super::backend::ClBackend;
use super::*;
use crate::create_vectors::{create_vectors, create_vectors_on, CreateVectorsSettings};
use crate::device::probe;
use crate::device::selfcheck;
use crate::image::{Fmt, Samples};
use crate::stages::{Cpu, StageSet};
use std::sync::Arc;

/// Every device that meets the requirements and passes the self-check, opened as an executor with
/// `placement` (the solver tests put every stage on the device; the default keeps the solver on
/// the CPU).
fn executors(test: &str, placement: Placement) -> Vec<Arc<Executor>> {
    let Ok(p) = probe::probe() else {
        eprintln!("{test}: no OpenCL runtime, skipped");
        return Vec::new();
    };
    let mut out = Vec::new();
    for info in p.devices.iter().filter(|d| probe::usable(d)) {
        let (dev, info) = probe::open(info.index).expect("listed device opens");
        if !selfcheck::selfcheck(&dev).is_usable() {
            eprintln!("{test}: gpu={} fails the self-check, skipped", info.index);
            continue;
        }
        let label = open::label(&info);
        out.push(Arc::new(Exec::new(ClBackend::open(&dev, info).expect("the program builds"), label, placement)));
    }
    if out.is_empty() {
        eprintln!("{test}: no usable OpenCL device, skipped");
    }
    out
}

fn frame(w: usize, h: usize, fmt: Fmt, t: f32) -> Image {
    let mut img = Image::new(w, h, fmt);
    let v = |x: usize, y: usize, c: usize| -> f32 {
        let (fx, fy) = (x as f32 - t * 2.5, y as f32 + t * 0.75);
        0.5 + 0.25 * ((fx * 0.21 + c as f32).sin() * (fy * 0.17).cos()) + 0.2 * ((fx - fy) * 0.05).sin()
    };
    let n = w * h;
    match &mut img.samples {
        Samples::U8(s) => (0..n * 4).for_each(|i| s[i] = if i % 4 == 0 { 255 } else { (v(i / 4 % w, i / 4 / w, i % 4) * 255.0) as u8 }),
        Samples::U16(s) => (0..n * 4).for_each(|i| s[i] = if i % 4 == 0 { 65535 } else { (v(i / 4 % w, i / 4 / w, i % 4) * 65535.0) as u16 }),
        Samples::F32(s) => (0..n * 4).for_each(|i| s[i] = if i % 4 == 0 { 1.0 } else { v(i / 4 % w, i / 4 / w, i % 4) }),
    }
    img
}

#[test]
#[ignore = "the kernel follows the rule before the parity work; port pending (docs/engine.md, Device)"]
fn device_solves_equal_the_cpu_at_every_depth_size_and_quality() {
    for e in executors("device_solves_equal_the_cpu", Placement::all_on_device()) {
        for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
            for (w, h) in [(2usize, 2usize), (5, 3), (33, 17), (64, 48), (97, 61)] {
                for quality in [1u8, 2, 3, 4, 5] {
                    for (hiq, wrap, delin, edge) in [(true, false, false, false), (false, true, true, false), (true, true, false, true)] {
                        let s = FlowSettings { wrap, prep_delinearize: delin, prep_contrast_edge: edge, ..FlowSettings::new(quality, hiq, 0.7) };
                        let (a, b) = (frame(w, h, fmt, 0.0), frame(w, h, fmt, 1.0));
                        let what = format!("{} {fmt:?} {w}x{h} q{quality} hiq {hiq} wrap {wrap}", e.label());
                        match (e.solve(&a, &b, &s), Cpu.solve(&a, &b, &s)) {
                            (Ok(g), Ok(c)) => assert!(g.data.iter().zip(&c.data).all(|(p, q)| p.to_bits() == q.to_bits()) && g.width == c.width, "{what}"),
                            (Err(g), Err(c)) => assert_eq!(g.to_string(), c.to_string(), "{what}"),
                            (g, c) => panic!("{what}: {g:?} against {c:?}"),
                        }
                        let (g, c) = (e.solve_pair(&a, &b, &s), Cpu.solve_pair(&a, &b, &s));
                        assert_eq!(format!("{g:?}"), format!("{c:?}"), "{what} pair");
                    }
                }
            }
        }
        eprintln!("{}", e.summary());
        assert!(e.report(Stage::SolvePair).device > 0, "{}", e.summary());
    }
}

#[test]
#[ignore = "the kernel follows the rule before the parity work; port pending (docs/engine.md, Device)"]
fn the_frame_cache_reuses_resident_pyramids_and_stays_exact() {
    for e in executors("the_frame_cache_reuses_resident_pyramids", Placement::all_on_device()) {
        let s = FlowSettings::new(4, true, 0.7);
        let frames: Vec<Image> = (0..4).map(|t| frame(120, 68, Fmt::U8, t as f32)).collect();
        // a slow motion pattern: each pair is solved twice, consecutive pairs share a frame
        for k in [0usize, 0, 1, 1, 2, 2] {
            let got = e.solve_pair(&frames[k], &frames[k + 1], &s).unwrap();
            assert_eq!(got, Cpu.solve_pair(&frames[k], &frames[k + 1], &s).unwrap(), "{} pair {k}", e.label());
        }
        let (hits, misses, _, bytes) = e.backend().cache_stats();
        assert_eq!(misses, 4, "{}: each frame is built once", e.label());
        assert_eq!(hits, 8, "{}", e.label());
        assert!(bytes > 0);
    }
}

#[test]
#[ignore = "the kernel follows the rule before the parity work; port pending (docs/engine.md, Device)"]
fn create_vectors_through_the_executor_equals_the_cpu_and_passes_the_cross_check() {
    for e in executors("create_vectors_through_the_executor", Placement::all_on_device()) {
        let set = StageSet::new(e.clone());
        for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
            for quality in [4u8, 5] {
                let s = CreateVectorsSettings { quality, hiq: true, sensitivity: 0.7, prep_contrast_edge: false, prep_delinearize: false, track_rgb_a: false, max_displace: 16.0 };
                for t in 0..3 {
                    let (x, y) = (frame(81, 47, fmt, t as f32), frame(81, 47, fmt, t as f32 + 1.0));
                    let (img, note) = cross_checked(e.as_ref(), |dev| create_vectors_on(if dev { set.get() } else { &Cpu }, &x, &y, &s).map_err(|e| e.to_string()));
                    assert_eq!(note, None, "{}", e.label());
                    assert_eq!(img.unwrap(), create_vectors(&x, &y, &s).unwrap(), "{} {fmt:?} q{quality} frame {t}", e.label());
                }
            }
        }
        assert!(e.is_active());
        assert!(e.report(Stage::Solve).device > 0);
    }
}

/// The vector frame of `compose::compose_vectors` for one setting, on every depth and on two
/// odd sizes, with a slow and a fast motion.
#[allow(clippy::too_many_arguments)]
fn compose_on(e: &Executor, fmt: Fmt, (w, h): (usize, usize), code: i32, interp: i32, wa: f32, blur: f32, binary: bool, wrap: bool) {
    use crate::compose::{self, Flows};
    use crate::timeline::Selection;
    let mut st = crate::host::test_support::settings(w, h, fmt);
    st.vectors_quality = 4;
    st.warp_mode = code;
    st.interp = interp;
    st.blur = blur;
    st.video_360 = binary;
    st.wrap_360 = wrap;
    let s0 = 3.0f32 + (1.0f32 - wa);
    let sel = Selection { s0, s1: s0 + if blur != 0.0 { 0.75 } else { 0.0 }, a: 3, b: 4, wa, exact: false, vectors_quality: 4, interp };
    let (a, b) = (frame(w, h, fmt, 0.0), frame(w, h, fmt, 3.0));
    let s = FlowSettings::new(4, true, 0.7);
    // a frame too small to solve gets a uniform flow of its size
    let uniform = |v: f32| crate::flow::FlowField { width: w, height: h, data: (0..w * h).flat_map(|_| [v, -0.5 * v]).collect() };
    let flows = match Cpu.solve_pair(&a, &b, &s) {
        Ok(p) => Flows::pair(p),
        Err(_) => Flows { f_ab: Some(uniform(0.75)), f_ba: Some(uniform(-0.75)) },
    };
    let what = format!("{} {fmt:?} {w}x{h} code {code} interp {interp} wa {wa} blur {blur} binary {binary} wrap {wrap}", e.label());
    let want = compose::compose_vectors(&st, &sel, &a, &b, &flows);
    let got = e.compose_vectors(&st, &sel, &a, &b, &flows);
    match (got, want) {
        (Ok(g), Ok(c)) => assert!(crate::device::exec::cache::same_bits(&g, &c), "{what}"),
        (Err(g), Err(c)) => assert_eq!(g.to_string(), c.to_string(), "{what}"),
        (g, c) => panic!("{what}: {:?} against {:?}", g.map(|_| ()), c.map(|_| ())),
    }
}

#[test]
fn the_compose_chain_equals_the_cpu_for_every_code_mode_depth_and_tail() {
    for e in executors("the_compose_chain_equals_the_cpu", Placement::default()) {
        for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
            for size in [(97usize, 61usize), (64, 33)] {
                for code in 0..4 {
                    for interp in 0..3 {
                        for wa in [0.25f32, 0.625] {
                            compose_on(&e, fmt, size, code, interp, wa, 0.0, false, false);
                        }
                        compose_on(&e, fmt, size, code, interp, 0.5, 1.5, false, false);
                        compose_on(&e, fmt, size, code, interp, 0.5, -0.75, false, false);
                    }
                    // the binary alpha Smart Blend ("360 Video") and the wrap-around flag
                    compose_on(&e, fmt, size, code, 1, 0.375, 0.0, true, false);
                    compose_on(&e, fmt, size, code, 1, 0.375, 1.0, false, true);
                }
            }
        }
        // tiny frames: the CPU's refusals and results
        for size in [(1usize, 1usize), (2, 2), (3, 1), (5, 3)] {
            for code in 0..4 {
                compose_on(&e, Fmt::U8, size, code, 1, 0.5, 0.0, false, false);
            }
        }
        eprintln!("{}", e.summary());
        let r = e.report(Stage::Compose);
        assert!(r.device > 0, "{}", e.summary());
        assert!(e.is_active());
    }
}

#[test]
fn the_device_forward_warp_equals_the_cpu_with_and_without_coverage() {
    use crate::forward::{forward_warp_with, ForwardOpts};
    for e in executors("the_device_forward_warp_equals_the_cpu", Placement::default()) {
        for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
            for (w, h) in [(2usize, 2usize), (5, 3), (97, 61), (160, 90)] {
                let (x, y) = (frame(w, h, fmt, 0.0), frame(w, h, fmt, 2.0));
                let flow = Cpu.solve(&x, &y, &FlowSettings::new(4, true, 0.7)).unwrap();
                for amount in [0.0f32, 0.5, 1.0, -0.75] {
                    for cov in [false, true] {
                        let mut want = Image::new(w, h, fmt);
                        let c = forward_warp_with(&mut want, &x, &flow, amount, ForwardOpts::default(), cov);
                        let mut got = Image::new(w, h, fmt);
                        let g = e.forward_warp_with(&mut got, &x, &flow, amount, ForwardOpts::default(), cov);
                        let what = format!("{} {fmt:?} {w}x{h} amount {amount} cov {cov}", e.label());
                        assert_eq!(format!("{g:?}"), format!("{c:?}"), "{what}");
                        assert!(crate::device::exec::cache::same_bits(&got, &want), "{what}");
                    }
                }
            }
        }
        eprintln!("{}", e.summary());
        assert!(e.report(Stage::ForwardWarp).device > 0, "{}", e.summary());
    }
}

#[test]
fn a_program_loaded_from_the_cache_gives_the_same_bytes_as_a_fresh_build() {
    use crate::device::progcache;
    let Ok(p) = probe::probe() else {
        eprintln!("a_program_loaded_from_the_cache: no OpenCL runtime, skipped");
        return;
    };
    for info in p.devices.iter().filter(|d| probe::usable(d)) {
        let (dev, info) = probe::open(info.index).expect("listed device opens");
        let source = backend::program();
        let opts = crate::device::kernels::BUILD_OPTIONS;
        let context = crate::device::cl::Context::new(&dev).unwrap();
        let fresh = crate::device::cl::Program::build(&context, &source, opts).unwrap();
        let binary = fresh.binary().unwrap();
        assert!(!binary.is_empty());
        let loaded = crate::device::cl::Program::from_binary(&context, &binary, opts).unwrap();
        assert_eq!(loaded.build_status().unwrap(), crate::device::cl::CL_BUILD_SUCCESS);
        // a binary the driver does not know is an error, not a program
        assert!(crate::device::cl::Program::from_binary(&context, b"not a binary", opts).is_err());
        // the cache entry of this device: written by a build, read by the next one
        let key = progcache::device_key(&dev);
        assert!(key.contains(&info.name), "{key}");
        let t = std::time::Instant::now();
        let a = ClBackend::open(&dev, info.clone()).expect("the program builds");
        let first = t.elapsed();
        let t = std::time::Instant::now();
        let b = ClBackend::open(&dev, info.clone()).expect("the program loads");
        eprintln!("gpu={}: open {} ms, again {} ms", info.index, first.as_millis(), t.elapsed().as_millis());
        let path = progcache::entry_path(&progcache::folder(), &key, opts, &source);
        let entry = std::fs::read(&path).expect("the entry was written");
        assert!(progcache::decode(&entry, &key, opts, &source).is_some());
        for backend in [a, b] {
            let e = Exec::new(backend, open::label(&info), Placement::default());
            for fmt in [Fmt::U8, Fmt::F32] {
                compose_on(&e, fmt, (97, 61), 1, 2, 0.375, 1.5, false, false);
                compose_on(&e, fmt, (97, 61), 2, 1, 0.625, 0.0, true, false);
            }
            assert_eq!(e.report(Stage::Compose).device, 4, "{}", e.summary());
        }
    }
}
