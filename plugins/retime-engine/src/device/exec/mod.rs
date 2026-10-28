//! The device executor: what makes `--device gpu` render.
//!
//! One [`Exec`] serves a whole run. It owns the device ([`backend::ClBackend`]: the context, the
//! in-order queue behind a mutex, the program built once from every stage module's sources, the
//! memory budget and the device frame cache) and implements the engine's stage seam
//! (`stages::Stages`), stage by stage:
//!
//! * a stage the placement table puts on the device runs there; any [`StageError`] (a driver
//!   error, a float output holding a NaN, a case the kernels do not cover, the memory budget, a
//!   busy device on the calling thread) reruns that stage with the CPU function, so the result is
//!   the CPU's either way;
//! * a stage the table keeps on the CPU, and every engine function that is not behind the seam,
//!   is the CPU function itself;
//! * after [`Exec::refuse`] every stage is the CPU function for the rest of the run.
//!
//! The gates: a device is opened only after it passes `selfcheck::selfcheck` ([`open`]); the
//! first frame of the run that used the device is rendered again on the CPU and compared byte for
//! byte ([`cross_checked`]); any difference refuses the device for the rest of the run with one
//! note, never silently.
//!
//! Stage placement (decided by the timing study, 1080p, both GPUs of the development machine):
//!
//! | stage | runs on | why |
//! |---|---|---|
//! | `flow::solve`, `flow::solve_pair` (pyramids built on the device, kept resident in the frame cache) | device | the largest part of a vector frame (the measured per call times are in docs/engine.md, Device) |
//! | `flow::extras::solve_pyramids_with` (masks, user displacement, wrap; host pyramids uploaded) | device | the solve dominates its upload |
//! | `forward::forward_warp_with` (codes 2 and 3, the flow inversion, the Warp debug view; the mesh and its mip levels on the host) | device | 4 to 7 times faster than the single threaded CPU rasteriser |
//! | `compose::compose_vectors` (the warps, Blend, Weighted Blend, Smart Blend, the hole fill and the blur compensation of one vector frame) as one resident chain ([`chain`]) | device, except with the wrap-around (360) flag | the frames and flows go up once and only the finished frame comes back |
//! | the layered compositions of `retime-fgbg` with mattes, vector coding, mattes, overlays | CPU | proven exact on the device, but alone each costs more in transfers than it saves |
//! | `delinearize`, the f64 `powf` / `ln` scalars, `matte::effective` with shrink | CPU | not on the device |
//!
//! No global state: the executor travels in `TemporalContext::executor` (a `stages::StageSet`)
//! and from there in `EngineSettings::stages`. Bytes equal the CPU's by construction (every
//! device stage is a bit exact port, proven by the stage modules' tests on both GPUs) and by the
//! two gates above.

pub mod backend;
pub mod cache;
pub mod chain;
pub mod open;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_device;

use crate::device::image::StageError;
use crate::flow::extras::{solve_pyramids_with, SolveExtras};
use crate::compose::{self, Flows};
use crate::flow::{self, FlowField, FlowPair, FlowSettings, Pyramid};
use crate::forward::{self, ForwardOpts};
use crate::host::EngineSettings;
use crate::image::Image;
use crate::stages::Stages;
use crate::timeline::Selection;
use crate::warp::Coverage;
use backend::Backend;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub use open::{open, Opened};

/// The executor over the OpenCL device.
pub type Executor = Exec<backend::ClBackend>;

/// The stages behind the seam, for the placement table and the counters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Solve,
    SolvePair,
    SolvePyramidsWith,
    /// `forward::forward_warp_with` (codes 2 and 3, the flow inversion, the Warp debug view).
    ForwardWarp,
    /// `compose::compose_vectors` as one resident chain.
    Compose,
}

/// How many stages there are.
const STAGES: usize = 5;

impl Stage {
    pub const ALL: [Stage; STAGES] = [Stage::Solve, Stage::SolvePair, Stage::SolvePyramidsWith, Stage::ForwardWarp, Stage::Compose];

    pub fn name(self) -> &'static str {
        match self {
            Stage::Solve => "solve",
            Stage::SolvePair => "solve_pair",
            Stage::SolvePyramidsWith => "solve_pyramids_with",
            Stage::ForwardWarp => "forward_warp",
            Stage::Compose => "compose",
        }
    }

    fn slot(self) -> usize {
        self as usize
    }
}

/// Which stages run on the device. The default is the table of the module documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    pub on_device: [bool; STAGES],
}

impl Default for Placement {
    /// Every stage on the device (the table of the module documentation).
    fn default() -> Placement {
        Placement { on_device: [true; STAGES] }
    }
}

impl Placement {
    /// Every stage on the CPU (the executor then only counts).
    pub fn cpu_only() -> Placement {
        Placement { on_device: [false; STAGES] }
    }

    pub fn device(self, stage: Stage) -> bool {
        self.on_device[stage.slot()]
    }
}

/// Counters of one stage.
#[derive(Default)]
struct StageCounters {
    device: AtomicU64,
    device_ns: AtomicU64,
    fallback_cl: AtomicU64,
    fallback_nan: AtomicU64,
    fallback_unsupported: AtomicU64,
    fallback_ns: AtomicU64,
    cpu: AtomicU64,
    cpu_ns: AtomicU64,
    /// The text of the first device refusal, for the report.
    first_refusal: Mutex<Option<String>>,
}

/// A snapshot of the counters of one stage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageReport {
    /// Calls the device served.
    pub device: u64,
    pub device_time: Duration,
    /// Calls the device failed and the CPU then served, by kind of [`StageError`].
    pub fallback_cl: u64,
    pub fallback_nan: u64,
    pub fallback_unsupported: u64,
    /// Time in the CPU reruns (the failed device attempt not included).
    pub fallback_time: Duration,
    /// Calls served by the CPU directly (placement, or after a refusal).
    pub cpu: u64,
    pub cpu_time: Duration,
}

impl StageReport {
    pub fn fallbacks(&self) -> u64 {
        self.fallback_cl + self.fallback_nan + self.fallback_unsupported
    }
}

/// How long opening the device took: the whole gate, the self-check and the program build (the
/// two run at once).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpenTimes {
    pub total: Duration,
    pub selfcheck: Duration,
    pub build: Duration,
}

/// The executor: a device backend, its placement table and the run's gate state.
pub struct Exec<B: Backend> {
    backend: B,
    /// `gpu=N (name)`, for the notes.
    label: String,
    placement: Placement,
    counters: [StageCounters; STAGES],
    refused: Mutex<Option<String>>,
    active: AtomicBool,
    checked: AtomicBool,
    open_times: Mutex<Option<OpenTimes>>,
}

impl<B: Backend> Exec<B> {
    pub fn new(backend: B, label: impl Into<String>, placement: Placement) -> Exec<B> {
        Exec {
            backend,
            label: label.into(),
            placement,
            counters: Default::default(),
            refused: Mutex::new(None),
            active: AtomicBool::new(true),
            checked: AtomicBool::new(false),
            open_times: Mutex::new(None),
        }
    }

    /// Records how long the gate took (for the report).
    pub fn set_open_times(&self, t: OpenTimes) {
        *self.open_times.lock().unwrap_or_else(|e| e.into_inner()) = Some(t);
    }

    pub fn open_times(&self) -> Option<OpenTimes> {
        *self.open_times.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn placement(&self) -> Placement {
        self.placement
    }

    /// False once the device is refused: every stage is then the CPU function.
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::SeqCst)
    }

    /// Refuses the device for the rest of the run. True the first time (the caller prints the
    /// note then), false when it was refused already.
    pub fn refuse(&self, why: impl Into<String>) -> bool {
        let mut r = self.refused.lock().unwrap_or_else(|e| e.into_inner());
        self.active.store(false, Ordering::SeqCst);
        if r.is_some() {
            return false;
        }
        *r = Some(why.into());
        true
    }

    /// Why the device was refused, if it was.
    pub fn refusal(&self) -> Option<String> {
        self.refused.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Calls the device served so far, over every stage.
    pub fn device_calls(&self) -> u64 {
        self.counters.iter().map(|c| c.device.load(Ordering::SeqCst)).sum()
    }

    /// Device calls plus failed device attempts: every call that reached the device.
    pub fn device_attempts(&self) -> u64 {
        Stage::ALL.iter().map(|&s| {
            let r = self.report(s);
            r.device + r.fallbacks()
        }).sum()
    }

    pub fn report(&self, stage: Stage) -> StageReport {
        let c = &self.counters[stage.slot()];
        let ns = |a: &AtomicU64| Duration::from_nanos(a.load(Ordering::SeqCst));
        StageReport {
            device: c.device.load(Ordering::SeqCst),
            device_time: ns(&c.device_ns),
            fallback_cl: c.fallback_cl.load(Ordering::SeqCst),
            fallback_nan: c.fallback_nan.load(Ordering::SeqCst),
            fallback_unsupported: c.fallback_unsupported.load(Ordering::SeqCst),
            fallback_time: ns(&c.fallback_ns),
            cpu: c.cpu.load(Ordering::SeqCst),
            cpu_time: ns(&c.cpu_ns),
        }
    }

    /// Total CPU fallbacks after a device attempt, over every stage.
    pub fn fallbacks(&self) -> u64 {
        Stage::ALL.iter().map(|&s| self.report(s).fallbacks()).sum()
    }

    /// One line per stage that ran, plus the cache counters: the `VFX_DEVICE_REPORT` note.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        for s in Stage::ALL {
            let r = self.report(s);
            if r.device + r.fallbacks() + r.cpu == 0 {
                continue;
            }
            parts.push(format!(
                "{} device {} ({} ms) fallback {} [cl {} nan {} unsupported {}] ({} ms) cpu {} ({} ms)",
                s.name(),
                r.device,
                r.device_time.as_millis(),
                r.fallbacks(),
                r.fallback_cl,
                r.fallback_nan,
                r.fallback_unsupported,
                r.fallback_time.as_millis(),
                r.cpu,
                r.cpu_time.as_millis()
            ));
            if let Some(why) = self.first_refusal(s) {
                parts.push(format!("{} first refused: {}", s.name(), why.replace(';', ",")));
            }
        }
        let (hits, misses, evictions, bytes) = self.backend.cache_stats();
        parts.push(format!("cache hits {hits} misses {misses} evictions {evictions} bytes {bytes}"));
        if let Some(t) = self.open_times() {
            parts.push(format!("open {} ms (self-check {} ms, program {} ms)", t.total.as_millis(), t.selfcheck.as_millis(), t.build.as_millis()));
        }
        format!("{}: {}", self.label, parts.join("; "))
    }

    /// The text of the first device refusal of `stage` in this run.
    pub fn first_refusal(&self, stage: Stage) -> Option<String> {
        self.counters[stage.slot()].first_refusal.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Runs `stage`: on the device when it is placed there and the device is active, the CPU
    /// function `cpu` otherwise or when the device attempt fails.
    fn run<T>(&self, stage: Stage, dev: impl FnOnce(&B) -> Result<T, StageError>, cpu: impl FnOnce() -> fx_core::Result<T>) -> fx_core::Result<T> {
        self.run_if(stage, true, dev, cpu)
    }

    /// [`Exec::run`], with `eligible` false for a call whose settings the placement keeps on the
    /// CPU (counted as a CPU call, not as a fallback).
    fn run_if<T>(&self, stage: Stage, eligible: bool, dev: impl FnOnce(&B) -> Result<T, StageError>, cpu: impl FnOnce() -> fx_core::Result<T>) -> fx_core::Result<T> {
        let c = &self.counters[stage.slot()];
        let add = |a: &AtomicU64, t: Instant| {
            a.fetch_add(t.elapsed().as_nanos() as u64, Ordering::SeqCst);
        };
        if !eligible || !self.is_active() || !self.placement.device(stage) {
            let t = Instant::now();
            let r = cpu();
            c.cpu.fetch_add(1, Ordering::SeqCst);
            add(&c.cpu_ns, t);
            return r;
        }
        let t = Instant::now();
        match dev(&self.backend) {
            Ok(v) => {
                c.device.fetch_add(1, Ordering::SeqCst);
                add(&c.device_ns, t);
                Ok(v)
            }
            Err(e) => {
                let kind = match e {
                    StageError::Cl(_) => &c.fallback_cl,
                    StageError::NanOutput => &c.fallback_nan,
                    StageError::Unsupported(_) => &c.fallback_unsupported,
                };
                kind.fetch_add(1, Ordering::SeqCst);
                let mut first = c.first_refusal.lock().unwrap_or_else(|e| e.into_inner());
                if first.is_none() {
                    *first = Some(e.to_string());
                }
                drop(first);
                let t = Instant::now();
                let r = cpu();
                add(&c.fallback_ns, t);
                r
            }
        }
    }
}

impl<B: Backend> Stages for Exec<B> {
    fn solve(&self, x: &Image, y: &Image, s: &FlowSettings) -> fx_core::Result<FlowField> {
        self.run(Stage::Solve, |b| b.solve(x, y, s), || flow::solve(x, y, s))
    }

    fn solve_pair(&self, a: &Image, b: &Image, s: &FlowSettings) -> fx_core::Result<FlowPair> {
        self.run(Stage::SolvePair, |d| d.solve_pair(a, b, s), || flow::solve_pair(a, b, s))
    }

    fn solve_pyramids_with(&self, x: &Pyramid, y: &Pyramid, s: &FlowSettings, e: SolveExtras<'_>) -> fx_core::Result<FlowField> {
        self.run(Stage::SolvePyramidsWith, |b| b.solve_pyramids_with(x, y, s, e), || solve_pyramids_with(x, y, s, e))
    }

    fn forward_warp_with(&self, out: &mut Image, src: &Image, flow: &FlowField, amount: f32, opts: ForwardOpts, want_coverage: bool) -> fx_core::Result<Option<Coverage>> {
        // the device result is written to `out` only on success, so a CPU rerun starts from the
        // caller's image
        let start = (self.is_active() && self.placement.device(Stage::ForwardWarp)).then(|| out.clone());
        let mut dev_out: Option<Image> = None;
        let r = self.run(
            Stage::ForwardWarp,
            |b| {
                let mut tmp = start.ok_or_else(|| StageError::Unsupported("the device was refused".into()))?;
                let c = b.forward_warp_with(&mut tmp, src, flow, amount, opts, want_coverage)?;
                dev_out = Some(tmp);
                Ok(c)
            },
            || forward::forward_warp_with(out, src, flow, amount, opts, want_coverage),
        );
        if let Some(img) = dev_out {
            *out = img;
        }
        r
    }

    fn compose_vectors(&self, settings: &EngineSettings, sel: &Selection, a: &Image, b: &Image, flows: &Flows) -> fx_core::Result<Image> {
        // the wrap-around (360) variants compose on the CPU (`chain`)
        let eligible = !settings.wrap_360;
        self.run_if(Stage::Compose, eligible, |d| d.compose_vectors(settings, sel, a, b, flows), || compose::compose_vectors(settings, sel, a, b, flows))
    }
}

/// What a frame render needs from the executor for the run level cross check.
pub trait Gate {
    /// Calls that reached the device so far (served or failed).
    fn attempts(&self) -> u64;
    /// The cross check has run.
    fn checked(&self) -> bool;
    fn mark_checked(&self);
    /// See [`Exec::refuse`].
    fn refuse(&self, why: String) -> bool;
    fn active(&self) -> bool;
    fn label(&self) -> String;
}

impl<B: Backend> Gate for Exec<B> {
    fn attempts(&self) -> u64 {
        self.device_attempts()
    }
    fn checked(&self) -> bool {
        self.checked.load(Ordering::SeqCst)
    }
    fn mark_checked(&self) {
        self.checked.store(true, Ordering::SeqCst);
    }
    fn refuse(&self, why: String) -> bool {
        Exec::refuse(self, why)
    }
    fn active(&self) -> bool {
        self.is_active()
    }
    fn label(&self) -> String {
        self.label.clone()
    }
}

/// The run level cross check (the second half of the device gate). `render(true)` renders a frame with the device,
/// `render(false)` on the CPU alone. The first frame of the run whose device render reached the
/// device is rendered on the CPU too and the two results are compared (`T` is the frame's bytes
/// and outcome); a difference refuses the device for the rest of the run and the CPU's result is
/// the frame. Returns the frame and, when the device was refused here, the note to print.
pub fn cross_checked<T: PartialEq>(gate: &dyn Gate, mut render: impl FnMut(bool) -> T) -> (T, Option<String>) {
    if !gate.active() {
        return (render(false), None);
    }
    let before = gate.attempts();
    let on_device = render(true);
    if gate.checked() || gate.attempts() == before {
        return (on_device, None);
    }
    gate.mark_checked();
    let on_cpu = render(false);
    if on_cpu == on_device {
        return (on_device, None);
    }
    let note = format!(
        "{}: the first frame rendered with the device differs from the CPU render, the CPU renders the rest of the run",
        gate.label()
    );
    gate.refuse(note.clone());
    (on_cpu, Some(note))
}
