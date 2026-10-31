//! Device-free tests of the executor: a fake backend that serves, fails with every kind of
//! `StageError`, or answers wrongly, the frame cache, the cross check and the open order.

use super::backend::{program_parts, Backend};
use super::cache::{digest, same_bits, ImageKey, Lru};
use super::open::candidates;
use super::*;
use crate::create_vectors::{create_vectors, create_vectors_on, CreateVectorsSettings};
use crate::device::cl::ClError;
use crate::device::kernels;
use crate::device::probe::DeviceInfo;
use crate::image::{Fmt, Samples};
use crate::compose::{self, Flows};
use crate::host::EngineSettings;
use crate::stages::{Cpu, StageSet};
use crate::timeline::Selection;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

/// What the fake device does with a call.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    /// The CPU function's result (a bit exact device).
    Exact,
    Cl,
    Nan,
    Unsupported,
    /// The CPU's result with one flow value changed (a faulty stage).
    Wrong,
    /// Exact for the first `n` calls, then `Cl` (a stage that breaks mid run).
    BreakAfter(usize),
}

struct Fake {
    mode: Mode,
    calls: AtomicUsize,
}

impl Fake {
    fn new(mode: Mode) -> Fake {
        Fake { mode, calls: AtomicUsize::new(0) }
    }

    fn answer(&self, mut f: FlowField) -> Result<FlowField, StageError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            Mode::Exact => Ok(f),
            Mode::Cl => Err(StageError::Cl(ClError::Usage("injected".into()))),
            Mode::Nan => Err(StageError::NanOutput),
            Mode::Unsupported => Err(StageError::Unsupported("injected".into())),
            Mode::Wrong => {
                f.data[0] += 0.25;
                Ok(f)
            }
            Mode::BreakAfter(k) if n < k => Ok(f),
            Mode::BreakAfter(_) => Err(StageError::Cl(ClError::Usage("broken".into()))),
        }
    }
}

fn cpu_err(e: fx_core::FxError) -> StageError {
    StageError::Unsupported(e.to_string())
}

impl Backend for Fake {
    fn solve(&self, x: &Image, y: &Image, s: &FlowSettings) -> Result<FlowField, StageError> {
        self.answer(flow::solve(x, y, s).map_err(cpu_err)?)
    }
    fn solve_pair(&self, a: &Image, b: &Image, s: &FlowSettings) -> Result<FlowPair, StageError> {
        let p = flow::solve_pair(a, b, s).map_err(cpu_err)?;
        Ok(FlowPair { f_ab: self.answer(p.f_ab)?, f_ba: self.answer(p.f_ba)? })
    }
    fn solve_pyramids_with(&self, x: &Pyramid, y: &Pyramid, s: &FlowSettings, e: SolveExtras<'_>) -> Result<FlowField, StageError> {
        self.answer(solve_pyramids_with(x, y, s, e).map_err(cpu_err)?)
    }
    fn compose_vectors(&self, settings: &EngineSettings, sel: &Selection, a: &Image, b: &Image, flows: &Flows) -> Result<Image, StageError> {
        let mut img = compose::compose_vectors(settings, sel, a, b, flows).map_err(cpu_err)?;
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            Mode::Exact => Ok(img),
            Mode::Cl => Err(StageError::Cl(ClError::Usage("injected".into()))),
            Mode::Nan => Err(StageError::NanOutput),
            Mode::Unsupported => Err(StageError::Unsupported("injected".into())),
            Mode::Wrong => {
                if let Samples::U8(v) = &mut img.samples {
                    v[1] = v[1].wrapping_add(1);
                }
                Ok(img)
            }
            Mode::BreakAfter(k) if n < k => Ok(img),
            Mode::BreakAfter(_) => Err(StageError::Cl(ClError::Usage("broken".into()))),
        }
    }
}

/// The stages the fake serves.
const SOLVES: [Stage; 3] = [Stage::Solve, Stage::SolvePair, Stage::SolvePyramidsWith];

/// The executor of the dispatch tests: the fake with every stage on the device (the default
/// placement keeps the solver on the CPU, see
/// `the_default_placement_serves_the_solver_on_the_cpu_and_the_textured_forward_warp_on_the_device`).
fn exec(mode: Mode) -> Arc<Exec<Fake>> {
    Arc::new(Exec::new(Fake::new(mode), "gpu=9 (fake)", Placement::all_on_device()))
}

/// A smooth moving texture at 8, 16 or 32 bpc: frame `t` is the pattern shifted by `t` pixels.
fn frame(w: usize, h: usize, fmt: Fmt, t: f32) -> Image {
    let mut img = Image::new(w, h, fmt);
    let v = |x: usize, y: usize, c: usize| -> f32 {
        let (fx, fy) = (x as f32 - t * 1.5, y as f32 + t * 0.5);
        0.5 + 0.25 * ((fx * 0.31 + c as f32).sin() * (fy * 0.23).cos()) + 0.2 * ((fx + fy) * 0.07).sin()
    };
    let n = w * h;
    match &mut img.samples {
        Samples::U8(s) => (0..n * 4).for_each(|i| s[i] = if i % 4 == 0 { 255 } else { (v(i / 4 % w, i / 4 / w, i % 4) * 255.0) as u8 }),
        Samples::U16(s) => (0..n * 4).for_each(|i| s[i] = if i % 4 == 0 { 65535 } else { (v(i / 4 % w, i / 4 / w, i % 4) * 65535.0) as u16 }),
        Samples::F32(s) => (0..n * 4).for_each(|i| s[i] = if i % 4 == 0 { 1.0 } else { v(i / 4 % w, i / 4 / w, i % 4) }),
    }
    img
}

fn cv_settings() -> CreateVectorsSettings {
    CreateVectorsSettings { quality: 4, hiq: true, sensitivity: 0.7, prep_contrast_edge: false, prep_delinearize: false, track_rgb_a: false, max_displace: 16.0 }
}

fn fs() -> FlowSettings {
    FlowSettings::new(3, true, 0.7)
}

#[test]
fn an_exact_device_serves_every_stage_and_gives_the_cpu_result() {
    let e = exec(Mode::Exact);
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let (a, b) = (frame(40, 24, fmt, 0.0), frame(40, 24, fmt, 1.0));
        assert_eq!(e.solve(&a, &b, &fs()).unwrap(), Cpu.solve(&a, &b, &fs()).unwrap());
        assert_eq!(e.solve_pair(&a, &b, &fs()).unwrap(), Cpu.solve_pair(&a, &b, &fs()).unwrap());
        let (pa, pb) = (flow::build_pyramid(&a, &fs()).unwrap(), flow::build_pyramid(&b, &fs()).unwrap());
        assert_eq!(
            e.solve_pyramids_with(&pa, &pb, &fs(), SolveExtras::default()).unwrap(),
            Cpu.solve_pyramids_with(&pa, &pb, &fs(), SolveExtras::default()).unwrap()
        );
    }
    for s in SOLVES {
        let r = e.report(s);
        assert_eq!((r.device, r.fallbacks(), r.cpu), (3, 0, 0), "{s:?}");
    }
    assert!(e.is_active());
}

#[test]
fn every_stage_error_kind_reruns_the_stage_on_the_cpu() {
    for (mode, kind) in [(Mode::Cl, 0), (Mode::Nan, 1), (Mode::Unsupported, 2)] {
        let e = exec(mode);
        let (a, b) = (frame(33, 17, Fmt::U16, 0.0), frame(33, 17, Fmt::U16, 2.0));
        assert_eq!(e.solve(&a, &b, &fs()).unwrap(), Cpu.solve(&a, &b, &fs()).unwrap(), "{mode:?}");
        assert_eq!(e.solve_pair(&a, &b, &fs()).unwrap(), Cpu.solve_pair(&a, &b, &fs()).unwrap(), "{mode:?}");
        let (pa, pb) = (flow::build_pyramid(&a, &fs()).unwrap(), flow::build_pyramid(&b, &fs()).unwrap());
        assert_eq!(
            e.solve_pyramids_with(&pa, &pb, &fs(), SolveExtras::default()).unwrap(),
            Cpu.solve_pyramids_with(&pa, &pb, &fs(), SolveExtras::default()).unwrap()
        );
        for s in SOLVES {
            let r = e.report(s);
            assert_eq!(r.device, 0, "{mode:?} {s:?}");
            assert_eq!([r.fallback_cl, r.fallback_nan, r.fallback_unsupported][kind], 1, "{mode:?} {s:?}");
            assert_eq!(r.fallbacks(), 1, "{mode:?} {s:?}");
        }
        assert_eq!(e.fallbacks(), 3);
        // a stage error does not refuse the device: the next call tries it again
        assert!(e.is_active());
    }
}

#[test]
fn a_nan_output_never_leaves_the_device_and_the_cpu_computes_it() {
    // the CPU flow of NaN frames holds NaNs: the fake refuses it as the wrappers do, the CPU's
    // NaN payloads come back
    let e = exec(Mode::Nan);
    let mut a = frame(20, 12, Fmt::F32, 0.0);
    if let Samples::F32(s) = &mut a.samples {
        s[4 * 25 + 1] = f32::NAN;
    }
    let b = frame(20, 12, Fmt::F32, 1.0);
    let got = e.solve(&a, &b, &fs()).unwrap();
    let want = Cpu.solve(&a, &b, &fs()).unwrap();
    assert!(got.data.iter().zip(&want.data).all(|(p, q)| p.to_bits() == q.to_bits()));
    assert_eq!(e.report(Stage::Solve).fallback_nan, 1);
}

#[test]
fn errors_the_cpu_raises_stay_the_cpu_errors() {
    let e = exec(Mode::Exact);
    let (a, b) = (frame(20, 12, Fmt::U8, 0.0), frame(21, 12, Fmt::U8, 0.0));
    assert_eq!(e.solve(&a, &b, &fs()).unwrap_err().to_string(), Cpu.solve(&a, &b, &fs()).unwrap_err().to_string());
}

#[test]
fn the_cpu_placement_and_a_refused_device_never_call_the_device() {
    let e = Exec::new(Fake::new(Mode::Wrong), "gpu=9 (fake)", Placement::cpu_only());
    let (a, b) = (frame(20, 12, Fmt::U8, 0.0), frame(20, 12, Fmt::U8, 1.0));
    assert_eq!(e.solve(&a, &b, &fs()).unwrap(), Cpu.solve(&a, &b, &fs()).unwrap());
    assert_eq!(e.backend().calls.load(Ordering::SeqCst), 0);
    assert_eq!(e.report(Stage::Solve).cpu, 1);
    let e = exec(Mode::Wrong);
    assert!(e.refuse("test"));
    assert!(!e.refuse("again"), "only the first refusal counts");
    assert_eq!(e.refusal().as_deref(), Some("test"));
    assert_eq!(e.solve_pair(&a, &b, &fs()).unwrap(), Cpu.solve_pair(&a, &b, &fs()).unwrap());
    assert_eq!(e.backend().calls.load(Ordering::SeqCst), 0);
}

#[test]
fn the_default_placement_serves_the_solver_on_the_cpu_and_the_textured_forward_warp_on_the_device() {
    // the solver kernels follow the rules before the parity work (docs/engine.md, Device): the
    // default serves the three solver stages with the CPU functions, counted as CPU calls, not as
    // fallbacks, and never calls the device for them, not even one that answers wrongly
    let p = Placement::default();
    assert_eq!(SOLVES.map(|s| p.device(s)), [false; 3]);
    assert!(p.device(Stage::ForwardWarp) && p.device(Stage::Compose));
    let e = Exec::new(Fake::new(Mode::Wrong), "gpu=9 (fake)", p);
    let (a, b) = (frame(33, 17, Fmt::U16, 0.0), frame(33, 17, Fmt::U16, 1.5));
    assert_eq!(e.solve(&a, &b, &fs()).unwrap(), Cpu.solve(&a, &b, &fs()).unwrap());
    assert_eq!(e.solve_pair(&a, &b, &fs()).unwrap(), Cpu.solve_pair(&a, &b, &fs()).unwrap());
    let (pa, pb) = (flow::build_pyramid(&a, &fs()).unwrap(), flow::build_pyramid(&b, &fs()).unwrap());
    assert_eq!(
        e.solve_pyramids_with(&pa, &pb, &fs(), SolveExtras::default()).unwrap(),
        Cpu.solve_pyramids_with(&pa, &pb, &fs(), SolveExtras::default()).unwrap()
    );
    assert_eq!(e.backend().calls.load(Ordering::SeqCst), 0);
    for s in SOLVES {
        let r = e.report(s);
        assert_eq!((r.device, r.fallbacks(), r.cpu), (0, 0, 1), "{s:?}");
    }
    assert_eq!(e.device_attempts(), 0);
    // the device report names them as CPU calls
    let summary = e.summary();
    assert!(summary.contains("solve device 0 (0 ms) fallback 0 [cl 0 nan 0 unsupported 0] (0 ms) cpu 1 ("), "{summary}");
    // the textured forward warp still goes to the device
    let src = frame(24, 16, Fmt::U8, 0.0);
    let mut flow = FlowField::new(24, 16);
    flow.data.iter_mut().for_each(|v| *v = 1.0);
    let mut want = Image::new(24, 16, Fmt::U8);
    crate::forward::forward_warp(&mut want, &src, &flow, 0.5).unwrap();
    let e = Exec::new(Textured, "gpu=9 (fake)", Placement::default());
    let mut got = Image::new(24, 16, Fmt::U8);
    e.forward_warp(&mut got, &src, &flow, 0.5).unwrap();
    assert_eq!(got, want);
    let r = e.report(Stage::ForwardWarp);
    assert_eq!((r.device, r.fallbacks(), r.cpu), (1, 0, 0));
}

/// One "frame" of a run: a create vectors render on the device set, or on the CPU for `None`.
fn cv_frame(set: Option<&StageSet>, t: f32) -> Result<Image, String> {
    let (x, y) = (frame(48, 32, Fmt::U8, t), frame(48, 32, Fmt::U8, t + 1.0));
    let stages: &dyn Stages = match set {
        Some(s) => s.get(),
        None => &Cpu,
    };
    create_vectors_on(stages, &x, &y, &cv_settings()).map_err(|e| e.to_string())
}

#[test]
fn the_cross_check_passes_an_exact_device_once_and_keeps_it() {
    let e = exec(Mode::Exact);
    let set = StageSet::new(e.clone());
    let mut cpu_renders = 0;
    for t in 0..4 {
        let (img, note) = cross_checked(e.as_ref(), |dev| {
            if !dev {
                cpu_renders += 1;
            }
            cv_frame(dev.then_some(&set), t as f32)
        });
        assert_eq!(note, None);
        let x = frame(48, 32, Fmt::U8, t as f32);
        let y = frame(48, 32, Fmt::U8, t as f32 + 1.0);
        assert_eq!(img.unwrap(), create_vectors(&x, &y, &cv_settings()).unwrap());
    }
    assert_eq!(cpu_renders, 1, "only the first frame is rendered twice");
    assert!(e.is_active());
    assert_eq!(e.report(Stage::Solve).device, 4);
}

#[test]
fn the_cross_check_catches_a_faulty_stage_and_refuses_the_device() {
    let e = exec(Mode::Wrong);
    let set = StageSet::new(e.clone());
    let want = |t: f32| create_vectors(&frame(48, 32, Fmt::U8, t), &frame(48, 32, Fmt::U8, t + 1.0), &cv_settings()).unwrap();
    // a faulty device output differs from the CPU's (the injected change shows in the encoding)
    assert_ne!(cv_frame(Some(&set), 0.0).unwrap(), want(0.0));
    let e = exec(Mode::Wrong);
    let set = StageSet::new(e.clone());
    let (img, note) = cross_checked(e.as_ref(), |dev| cv_frame(dev.then_some(&set), 0.0));
    assert_eq!(img.unwrap(), want(0.0), "the CPU's frame is the output");
    let note = note.expect("a refusal note");
    assert!(note.starts_with("gpu=9 (fake): the first frame rendered with the device differs"), "{note}");
    assert!(!e.is_active());
    // the rest of the run is the CPU: the device is never called again, and no second note
    let calls = e.backend().calls.load(Ordering::SeqCst);
    for t in 1..4 {
        let (img, note) = cross_checked(e.as_ref(), |dev| cv_frame(dev.then_some(&set), t as f32));
        assert_eq!(img.unwrap(), want(t as f32));
        assert_eq!(note, None);
    }
    assert_eq!(e.backend().calls.load(Ordering::SeqCst), calls);
}

#[test]
fn frames_that_do_not_reach_the_device_wait_for_the_first_that_does() {
    let e = exec(Mode::Wrong);
    let set = StageSet::new(e.clone());
    let mut cpu_renders = 0;
    // two frames without a solve (an exact frame, No Vectors): nothing to compare
    for _ in 0..2 {
        let (v, note) = cross_checked(e.as_ref(), |dev| {
            cpu_renders += usize::from(!dev);
            7
        });
        assert_eq!((v, note), (7, None));
    }
    assert_eq!(cpu_renders, 0);
    assert!(!Gate::checked(e.as_ref()));
    // the first vector frame is the checked one
    let (_, note) = cross_checked(e.as_ref(), |dev| cv_frame(dev.then_some(&set), 0.0));
    assert!(note.is_some());
}

#[test]
fn a_stage_that_breaks_mid_run_keeps_every_frame_equal_to_the_cpu() {
    let e = exec(Mode::BreakAfter(2));
    let set = StageSet::new(e.clone());
    for t in 0..6 {
        let (img, note) = cross_checked(e.as_ref(), |dev| cv_frame(dev.then_some(&set), t as f32));
        assert_eq!(note, None);
        let want = create_vectors(&frame(48, 32, Fmt::U8, t as f32), &frame(48, 32, Fmt::U8, t as f32 + 1.0), &cv_settings()).unwrap();
        assert_eq!(img.unwrap(), want, "frame {t}");
    }
    let r = e.report(Stage::Solve);
    assert_eq!((r.device, r.fallback_cl), (2, 4));
    assert!(e.is_active(), "a failing stage falls back, it does not refuse the device");
}

#[test]
fn the_summary_names_each_stage_that_ran() {
    let e = exec(Mode::Nan);
    let (a, b) = (frame(20, 12, Fmt::U8, 0.0), frame(20, 12, Fmt::U8, 1.0));
    e.solve(&a, &b, &fs()).unwrap();
    let s = e.summary();
    assert!(s.starts_with("gpu=9 (fake): solve device 0"), "{s}");
    assert!(s.contains("fallback 1 [cl 0 nan 1 unsupported 0]"), "{s}");
    assert!(!s.contains("solve_pair"), "{s}");
    assert!(s.ends_with("cache hits 0 misses 0 evictions 0 bytes 0"), "{s}");
    assert!(s.contains("solve first refused: a float output holds a NaN"), "{s}");
    assert_eq!(crate::device::exec::Exec::first_refusal(&e, Stage::SolvePair), None);
    // the open times, once the gate set them
    e.set_open_times(OpenTimes { total: Duration::from_millis(250), selfcheck: Duration::from_millis(200), build: Duration::from_millis(240) });
    assert!(e.summary().ends_with("; open 250 ms (self-check 200 ms, program 240 ms)"), "{}", e.summary());
}

#[test]
fn the_wrap_around_flag_keeps_the_compose_chain_on_the_cpu_without_a_fallback() {
    let e = exec(Mode::Exact);
    let (mut st, sel, a, b, flows) = compose_case(Fmt::U8, 1, 1, false);
    st.wrap_360 = true;
    assert_eq!(e.compose_vectors(&st, &sel, &a, &b, &flows).unwrap(), compose::compose_vectors(&st, &sel, &a, &b, &flows).unwrap());
    let r = e.report(Stage::Compose);
    assert_eq!((r.device, r.fallbacks(), r.cpu), (0, 0, 1));
}

#[test]
fn the_lru_evicts_the_least_recently_used_inside_the_budget() {
    let mut c: Lru<u32, &str> = Lru::new(100);
    assert!(c.insert(1, "a", 40));
    assert!(c.insert(2, "b", 40));
    assert_eq!(c.get(|k| *k == 1), Some(&"a"), "a hit makes 1 the most recent");
    assert!(c.insert(3, "c", 40), "2 is evicted, not 1");
    assert_eq!(c.peek(|k| *k == 2), None);
    assert_eq!(c.peek(|k| *k == 1), Some(&"a"));
    assert_eq!((c.len(), c.used(), c.evictions), (2, 80, 1));
    assert!(!c.insert(4, "d", 101), "an entry over the whole budget is not kept");
    assert_eq!(c.len(), 2);
    assert!(c.insert(5, "e", 100), "a full budget entry evicts everything else");
    assert_eq!((c.len(), c.used(), c.evictions), (1, 100, 3));
    assert_eq!(c.get(|k| *k == 9), None);
    assert_eq!((c.hits, c.misses), (1, 1));
    c.clear();
    assert!(c.is_empty() && c.used() == 0);
}

#[test]
fn cache_keys_compare_every_sample_bit_for_bit() {
    let a = frame(16, 8, Fmt::F32, 0.0);
    let s = fs();
    let key = ImageKey::new(&a, &s);
    assert!(key.matches(&a, &s, digest(&a)));
    // -0.0 against 0.0, in a sample the digest does not sample
    let mut z = a.clone();
    let mut nz = a.clone();
    if let (Samples::F32(p), Samples::F32(q)) = (&mut z.samples, &mut nz.samples) {
        p[5] = 0.0;
        q[5] = -0.0;
    }
    assert!(!same_bits(&z, &nz));
    assert!(!ImageKey::new(&z, &s).matches(&nz, &s, digest(&nz)));
    // a NaN matches the same NaN bits only
    let mut n1 = a.clone();
    if let Samples::F32(p) = &mut n1.samples {
        p[7] = f32::from_bits(0x7fc0_0001);
    }
    let mut n2 = n1.clone();
    assert!(ImageKey::new(&n1, &s).matches(&n2, &s, digest(&n2)));
    if let Samples::F32(p) = &mut n2.samples {
        p[7] = f32::from_bits(0x7fc0_0002);
    }
    assert!(!ImageKey::new(&n1, &s).matches(&n2, &s, digest(&n2)));
    // other settings, size or format are other keys
    assert!(!key.matches(&a, &FlowSettings { sensitivity: 0.5, ..s }, digest(&a)));
    assert!(!key.matches(&a, &FlowSettings { wrap: true, ..s }, digest(&a)));
    let b = frame(16, 8, Fmt::U16, 0.0);
    assert!(!key.matches(&b, &s, digest(&b)));
}

fn info(index: usize, unified: bool, usable: bool) -> DeviceInfo {
    use crate::device::cl;
    DeviceInfo {
        index,
        name: format!("dev{index}"),
        profile: "FULL_PROFILE".into(),
        version: "OpenCL 3.0".into(),
        c_version: "OpenCL C 1.2".into(),
        little_endian: true,
        available: usable,
        compiler_available: true,
        unified_memory: unified,
        global_memory: 1 << 32,
        max_allocation: 1 << 30,
        extensions: vec!["cl_khr_fp64".into()],
        single_fp: cl::CL_FP_DENORM | cl::CL_FP_INF_NAN | cl::CL_FP_ROUND_TO_NEAREST | cl::CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT,
        double_fp: cl::CL_FP_DENORM | cl::CL_FP_INF_NAN | cl::CL_FP_ROUND_TO_NEAREST,
        device_type: cl::CL_DEVICE_TYPE_GPU,
        ..Default::default()
    }
}

#[test]
fn gpu_tries_devices_with_their_own_memory_first() {
    let list = vec![info(0, true, true), info(1, false, false), info(2, false, true), info(3, true, true)];
    assert!(crate::device::probe::usable(&list[0]), "{:?}", crate::device::probe::requirements(&list[0]));
    assert!(!crate::device::probe::usable(&list[1]));
    assert_eq!(candidates(&list), vec![2, 0, 3]);
    assert_eq!(crate::device::probe::preferred(&list), Some(2));
    assert_eq!(candidates(&[info(0, false, false)]), Vec::<usize>::new());
}

#[test]
fn the_program_holds_every_stage_source_once() {
    let parts = program_parts();
    for (name, src) in kernels::SOURCES {
        if name.starts_with("prelude::") || name.starts_with("starter::") {
            continue;
        }
        assert_eq!(parts.iter().filter(|p| **p == *src).count(), 1, "{name}");
    }
    assert_eq!(parts.len(), kernels::SOURCES.len() - 6);
    assert!(backend::program().starts_with("#pragma OPENCL FP_CONTRACT OFF\n"));
}

#[test]
fn the_forward_warp_is_placed_on_the_device_and_a_refusal_reruns_it_on_the_cpu() {
    use crate::forward::{forward_warp_coverage, ForwardOpts};
    let src = frame(24, 16, Fmt::U16, 0.0);
    let flow = Cpu.solve(&src, &frame(24, 16, Fmt::U16, 1.0), &fs()).unwrap();
    let mut want = Image::new(24, 16, Fmt::U16);
    let want_cov = forward_warp_coverage(&mut want, &src, &flow, 0.5).unwrap();
    // the default placement: on a device without the stage the refusal reruns it on the CPU
    // from the caller's image
    let e = Exec::new(Fake::new(Mode::Exact), "gpu=9 (fake)", Placement::default());
    let mut got = Image::new(24, 16, Fmt::U16);
    assert_eq!(e.forward_warp_coverage(&mut got, &src, &flow, 0.5).unwrap(), want_cov);
    assert_eq!(got, want);
    let r = e.report(Stage::ForwardWarp);
    assert_eq!((r.device, r.fallback_unsupported, r.cpu), (0, 1, 0));
    // kept on the CPU by the placement: the CPU function, counted as a CPU call
    let mut off = Placement::default();
    off.on_device[Stage::ForwardWarp as usize] = false;
    let e = Exec::new(Fake::new(Mode::Exact), "gpu=9 (fake)", off);
    let mut got = Image::new(24, 16, Fmt::U16);
    assert_eq!(e.forward_warp_with(&mut got, &src, &flow, 0.5, ForwardOpts::default(), true).unwrap(), Some(want_cov));
    assert_eq!(got, want);
    let r = e.report(Stage::ForwardWarp);
    assert_eq!((r.device, r.fallbacks(), r.cpu), (0, 0, 1));
}

/// One vector frame of a 40 x 24 pair: the settings for warping code `code` and mode `interp`,
/// the selection (with a blur compensation tail when `blur`), the frames and both flows.
fn compose_case(fmt: Fmt, code: i32, interp: i32, blur: bool) -> (EngineSettings, Selection, Image, Image, Flows) {
    let (w, h) = (40, 24);
    let mut st = crate::host::test_support::settings(w, h, fmt);
    st.vectors_quality = 4;
    st.warp_mode = code;
    st.interp = interp;
    st.blur = if blur { 1.5 } else { 0.0 };
    let wa = 0.375f32;
    let s0 = 3.0f32 + (1.0f32 - wa);
    let sel = Selection { s0, s1: if blur { s0 + 0.5 } else { s0 }, a: 3, b: 4, wa, exact: false, vectors_quality: 4, interp };
    let (a, b) = (frame(w, h, fmt, 0.0), frame(w, h, fmt, 1.5));
    let flows = Flows::pair(Cpu.solve_pair(&a, &b, &fs()).unwrap());
    (st, sel, a, b, flows)
}

#[test]
fn the_compose_chain_reruns_on_the_cpu_for_every_stage_error_kind() {
    for (mode, kind) in [(Mode::Exact, 3), (Mode::Cl, 0), (Mode::Nan, 1), (Mode::Unsupported, 2)] {
        let e = exec(mode);
        for code in 0..4 {
            for interp in 0..3 {
                for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
                    let (st, sel, a, b, flows) = compose_case(fmt, code, interp, code == 1);
                    let want = compose::compose_vectors(&st, &sel, &a, &b, &flows).unwrap();
                    assert_eq!(e.compose_vectors(&st, &sel, &a, &b, &flows).unwrap(), want, "{mode:?} code {code} interp {interp} {fmt:?}");
                }
            }
        }
        let r = e.report(Stage::Compose);
        let fallbacks = [r.fallback_cl, r.fallback_nan, r.fallback_unsupported];
        if kind == 3 {
            assert_eq!((r.device, r.fallbacks()), (36, 0), "{mode:?}");
        } else {
            assert_eq!(r.device, 0, "{mode:?}");
            assert_eq!(fallbacks[kind], 36, "{mode:?}");
            assert_eq!(r.fallbacks(), 36, "{mode:?}");
        }
        assert!(e.is_active(), "a fallback never refuses the device");
    }
}

#[test]
fn a_chain_that_breaks_mid_run_falls_back_from_then_on_and_a_cpu_error_stays_the_cpu_error() {
    let e = exec(Mode::BreakAfter(2));
    for t in 0..5 {
        let (st, mut sel, a, b, flows) = compose_case(Fmt::U8, 0, 2, false);
        sel.wa = 0.2 + 0.1 * t as f32;
        assert_eq!(e.compose_vectors(&st, &sel, &a, &b, &flows).unwrap(), compose::compose_vectors(&st, &sel, &a, &b, &flows).unwrap(), "frame {t}");
    }
    let r = e.report(Stage::Compose);
    assert_eq!((r.device, r.fallback_cl), (2, 3));
    // a frame the CPU refuses (a missing flow) gives the CPU's error
    let (st, sel, a, b, mut flows) = compose_case(Fmt::U16, 0, 1, false);
    flows.f_ab = None;
    let e = exec(Mode::Exact);
    let want = compose::compose_vectors(&st, &sel, &a, &b, &flows).unwrap_err().to_string();
    assert_eq!(e.compose_vectors(&st, &sel, &a, &b, &flows).unwrap_err().to_string(), want);
}

#[test]
fn the_cross_check_catches_a_faulty_compose_chain() {
    let e = exec(Mode::Wrong);
    let (st, sel, a, b, flows) = compose_case(Fmt::U8, 0, 1, false);
    let want = compose::compose_vectors(&st, &sel, &a, &b, &flows).unwrap();
    let (img, note) = cross_checked(e.as_ref(), |dev| {
        let stages: &dyn Stages = if dev { e.as_ref() } else { &Cpu };
        stages.compose_vectors(&st, &sel, &a, &b, &flows).unwrap()
    });
    assert_eq!(img, want, "the CPU's frame is written");
    assert!(note.unwrap().contains("differs from the CPU render"));
    assert!(!e.is_active());
    // refused: the chain is the CPU function from then on
    assert_eq!(e.compose_vectors(&st, &sel, &a, &b, &flows).unwrap(), want);
    assert_eq!(e.report(Stage::Compose).cpu, 1);
}

#[test]
fn the_cpu_stage_set_composes_with_the_cpu_function() {
    for code in 0..4 {
        let (st, sel, a, b, flows) = compose_case(Fmt::F32, code, 2, true);
        assert_eq!(Cpu.compose_vectors(&st, &sel, &a, &b, &flows).unwrap(), compose::compose_vectors(&st, &sel, &a, &b, &flows).unwrap());
    }
}

#[test]
fn the_working_set_bounds_grow_with_the_frame() {
    let small = Image::new(64, 32, Fmt::U8);
    let big = Image::new(128, 64, Fmt::F32);
    assert!(backend::forward_bytes(&big) > 4 * backend::forward_bytes(&small));
    let none = Flows { f_ab: None, f_ba: None };
    assert!(backend::compose_bytes(&big, &none) > backend::compose_bytes(&small, &none));
    // 4K float: inside the working budget of an 8 GB device, the forward warp well inside it
    let uhd = Image::new(3840, 2160, Fmt::F32);
    assert!(backend::compose_bytes(&uhd, &none) < (8u64 << 30) / 4 * 3 / 4 * 3);
    assert!(backend::forward_bytes(&uhd) < (1u64 << 31));
}

/// A device whose forward warp follows the device plan (`fwdstage::plan`): an argument the plan
/// refuses is its `StageError`, anything it accepts is the textured warp, the only variant the
/// kernels compute (a Gouraud call that reached it would come back textured).
struct Textured;

impl Backend for Textured {
    fn solve(&self, x: &Image, y: &Image, s: &FlowSettings) -> Result<FlowField, StageError> {
        flow::solve(x, y, s).map_err(cpu_err)
    }
    fn solve_pair(&self, a: &Image, b: &Image, s: &FlowSettings) -> Result<FlowPair, StageError> {
        flow::solve_pair(a, b, s).map_err(cpu_err)
    }
    fn solve_pyramids_with(&self, x: &Pyramid, y: &Pyramid, s: &FlowSettings, e: SolveExtras<'_>) -> Result<FlowField, StageError> {
        solve_pyramids_with(x, y, s, e).map_err(cpu_err)
    }
    fn forward_warp_with(
        &self,
        out: &mut Image,
        src: &Image,
        flow: &FlowField,
        amount: f32,
        opts: crate::forward::ForwardOpts,
        want_coverage: bool,
    ) -> Result<Option<crate::warp::Coverage>, StageError> {
        crate::device::fwdstage::plan(out, src, flow, amount, opts)?;
        crate::forward::forward_warp_with(out, src, flow, amount, crate::forward::ForwardOpts::default(), want_coverage).map_err(cpu_err)
    }
}

#[test]
fn the_gouraud_forward_warp_stays_on_the_cpu_and_the_textured_one_on_the_device() {
    use crate::forward::ForwardOpts;
    use crate::stages::GouraudForward;
    // vectors-in fix 02: every forward warp of vectors-in and of the create-vectors Debug views is
    // the Gouraud variant, which no kernel computes; a white impulse pushed half a pixel along the
    // diagonal tells the variants apart (two pixels at half value against four at a quarter)
    let (w, h) = (16usize, 12usize);
    let mut src = Image::new(w, h, Fmt::U8);
    if let Samples::U8(d) = &mut src.samples {
        for px in d.chunks_exact_mut(4) {
            px.copy_from_slice(&[255, 0, 0, 0]);
        }
        let i = (6 * w + 7) * 4;
        d[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
    }
    let mut flow = FlowField::new(w, h);
    flow.data.iter_mut().for_each(|v| *v = 1.0);
    let cpu = |opts: ForwardOpts| {
        let mut o = Image::new(w, h, Fmt::U8);
        let c = crate::forward::forward_warp_with(&mut o, &src, &flow, 0.5, opts, true).unwrap();
        (o, c)
    };
    let (gouraud, gouraud_cov) = cpu(ForwardOpts { pyramid: false });
    let (textured, _) = cpu(ForwardOpts::default());
    assert_ne!(gouraud, textured, "the two variants differ on this frame");
    let e = Arc::new(Exec::new(Textured, "gpu=9 (fake)", Placement::default()));
    // the stages of vectors-in: the Gouraud variant over the executor, a CPU call, not a failed
    // device attempt
    let set = GouraudForward::wrap(StageSet::new(e.clone()));
    let mut got = Image::new(w, h, Fmt::U8);
    let cov = set.get().forward_warp_coverage(&mut got, &src, &flow, 0.5).unwrap();
    assert_eq!(got, gouraud);
    assert_eq!(cov, gouraud_cov.unwrap());
    let r = e.report(Stage::ForwardWarp);
    assert_eq!((r.device, r.fallbacks(), r.cpu), (0, 0, 1));
    assert_eq!(e.device_attempts(), 0);
    // the textured warp of the same executor is served by the device
    let mut got = Image::new(w, h, Fmt::U8);
    e.forward_warp(&mut got, &src, &flow, 0.5).unwrap();
    assert_eq!(got, textured);
    assert_eq!(e.report(Stage::ForwardWarp).device, 1);
    // should a Gouraud call reach the device anyway, its plan refuses it (and the executor would
    // rerun it on the CPU)
    assert!(matches!(crate::device::fwdstage::plan(&src, &src, &flow, 0.5, ForwardOpts { pyramid: false }), Err(StageError::Unsupported(_))));
    let mut got = Image::new(w, h, Fmt::U8);
    assert!(matches!(Textured.forward_warp_with(&mut got, &src, &flow, 0.5, ForwardOpts { pyramid: false }, false), Err(StageError::Unsupported(_))));
}

/// A bit exact device (the CPU functions) that keeps every frame its solves and its compose
/// chain receive.
#[derive(Default)]
struct Recorder {
    seen: std::sync::Mutex<Vec<Image>>,
}

impl Recorder {
    fn keep(&self, img: &Image) {
        self.seen.lock().unwrap().push(img.clone());
    }
}

impl Backend for Recorder {
    fn solve(&self, x: &Image, y: &Image, s: &FlowSettings) -> Result<FlowField, StageError> {
        self.keep(x);
        self.keep(y);
        flow::solve(x, y, s).map_err(cpu_err)
    }
    fn solve_pair(&self, a: &Image, b: &Image, s: &FlowSettings) -> Result<FlowPair, StageError> {
        self.keep(a);
        self.keep(b);
        flow::solve_pair(a, b, s).map_err(cpu_err)
    }
    fn solve_pyramids_with(&self, x: &Pyramid, y: &Pyramid, s: &FlowSettings, e: SolveExtras<'_>) -> Result<FlowField, StageError> {
        solve_pyramids_with(x, y, s, e).map_err(cpu_err)
    }
    fn compose_vectors(&self, settings: &EngineSettings, sel: &Selection, a: &Image, b: &Image, flows: &Flows) -> Result<Image, StageError> {
        self.keep(a);
        self.keep(b);
        compose::compose_vectors(settings, sel, a, b, flows).map_err(cpu_err)
    }
}

/// The selection and the output frame of a whole render (`compose::render`) of an 8 bpc clip at
/// half speed.
fn render_frame(clip: &[&Image], s: &EngineSettings) -> (Selection, Image) {
    use crate::timeline::{test_support::Mock, RemapTimeMap};
    use fx_core::{Env, Frame, LayerSet, MemLayer, ParamCurves, Pixels, Rational, TimeBase};
    let frames = clip
        .iter()
        .map(|img| match &img.samples {
            Samples::U8(v) => Frame { width: img.width, height: img.height, pixels: Pixels::U8(v.chunks_exact(4).map(|p| [p[1], p[2], p[3], p[0]]).collect()) },
            _ => unreachable!("8 bpc only"),
        })
        .collect();
    let input = MemLayer::new(TimeBase::default_for_rate(Rational::new(30, 1).unwrap()), frames).unwrap();
    let params = ParamCurves::new(Vec::new());
    let layers = LayerSet::new();
    let env = Env { params: &params, input: &input, layers: &layers };
    let mock = Mock::speed(50.0);
    let map = RemapTimeMap::new(&mock, s.rates, s.current_time);
    let host = crate::host::EnvHost::new(&env, s, &map, crate::host::LayerIds::default()).unwrap();
    let sel = crate::compose::select_frames(s, &host).unwrap();
    match crate::compose::render(s, &host).unwrap() {
        crate::compose::Rendered::Image(img) => (sel, img),
        other => panic!("expected an image, got {other:?}"),
    }
}

#[test]
fn reversed_motion_rows_reach_the_device_bottom_row_first_and_the_frame_equals_the_cpu() {
    // retime-flow fix 03 (`EngineSettings::motion_rows_reversed`, plugin 01): the vector path turns
    // the frames upside down above the stage seam, so the device stages (the solve and the compose
    // chain) receive the reversed frames and the host turns the output back; the device needs no
    // change of its own (every stage on the device here, the solver included)
    let (w, h) = (40, 24);
    let (f0, f1) = (frame(w, h, Fmt::U8, 0.0), frame(w, h, Fmt::U8, 1.5));
    let mut s = crate::host::test_support::settings(w, h, Fmt::U8);
    s.vectors_quality = 4;
    s.current_time = 100;
    s.no_tracking_layer = true;
    for interp in [0, 1, 2] {
        s.interp = interp;
        s.motion_rows_reversed = true;
        let (sel, cpu) = render_frame(&[&f0, &f1], &s);
        assert_eq!((sel.a, sel.b, sel.wa, sel.exact, sel.vectors_quality), (0, 1, 0.5, false, 4), "interp {interp}: the vector path between frames 0 and 1");
        let rec = Arc::new(Exec::new(Recorder::default(), "gpu=9 (fake)", Placement::all_on_device()));
        let mut on_device = s.clone();
        on_device.stages = StageSet::new(rec.clone());
        assert_eq!(render_frame(&[&f0, &f1], &on_device).1, cpu, "interp {interp}");
        let r = (rec.report(Stage::Solve), rec.report(Stage::SolvePair), rec.report(Stage::Compose));
        assert!(r.0.device + r.1.device >= 1, "interp {interp}: the solve ran on the device");
        assert_eq!((r.2.device, r.2.fallbacks(), r.2.cpu), (1, 0, 0), "interp {interp}: the compose chain ran on the device");
        let seen = rec.backend().seen.lock().unwrap();
        assert!(seen.len() >= 4);
        let (r0, r1) = (f0.rows_reversed(), f1.rows_reversed());
        assert!(seen.iter().all(|img| *img == r0 || *img == r1), "interp {interp}: every device stage saw the reversed frames");
        drop(seen);
        // the CPU frame is the reversed path's own: the frames reversed, solved and composed, the
        // output reversed back
        let (a, b) = (f0.rows_reversed(), f1.rows_reversed());
        let flows = crate::compose::solve_flows(&s, &sel, &a, &b).unwrap();
        let want = crate::compose::compose_vectors(&s, &sel, &a, &b, &flows).unwrap().rows_reversed();
        assert_eq!(cpu, want, "interp {interp}: the reversed path");
        // and the flag matters for these frames
        s.motion_rows_reversed = false;
        assert_ne!(render_frame(&[&f0, &f1], &s).1, cpu, "interp {interp}: the top row first render differs");
    }
}
