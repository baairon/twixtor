//! What the executor asks of a device: the device stages with the CPU functions' arguments, each
//! returning the CPU function's result or a [`StageError`]. [`ClBackend`] is the OpenCL device
//! (one session, one in-order queue behind a mutex, the device frame cache); the device-free
//! tests implement [`Backend`] with fakes.

use super::cache::{digest, ImageKey, Lru};
use crate::device::cl;
use crate::device::convstage::Taps;
use crate::device::image::StageError;
use crate::device::kernels;
use crate::device::probe::{self, DeviceInfo};
use crate::device::selfcheck::Session;
use crate::device::solver::{self, build_pyramid_on, download_flow, solve_pyramids_with_enqueue, DeviceExtras, DevicePyramid};
use crate::device::{convstage, fwdstage, vecstage, warpstage, wbstage};
use crate::flow::extras::SolveExtras;
use crate::flow::{FlowField, FlowPair, FlowSettings, Pyramid};
use crate::compose::Flows;
use crate::forward::ForwardOpts;
use crate::host::EngineSettings;
use crate::image::{Fmt, Image, CHANNELS};
use crate::timeline::Selection;
use crate::warp::Coverage;
use std::sync::{Mutex, MutexGuard};
use std::thread::{self, ThreadId};

/// The device side of the stages of `stages::Stages`.
pub trait Backend: Send + Sync + 'static {
    /// `flow::solve`.
    fn solve(&self, x: &Image, y: &Image, s: &FlowSettings) -> Result<FlowField, StageError>;
    /// `flow::solve_pair`.
    fn solve_pair(&self, a: &Image, b: &Image, s: &FlowSettings) -> Result<FlowPair, StageError>;
    /// `flow::extras::solve_pyramids_with`.
    fn solve_pyramids_with(&self, x: &Pyramid, y: &Pyramid, s: &FlowSettings, e: SolveExtras<'_>) -> Result<FlowField, StageError>;
    /// `forward::forward_warp_with`; writes `out` only on success.
    fn forward_warp_with(&self, _out: &mut Image, _src: &Image, _flow: &FlowField, _amount: f32, _opts: ForwardOpts, _want_coverage: bool) -> Result<Option<Coverage>, StageError> {
        Err(StageError::Unsupported("no forward warp stage on this device".into()))
    }
    /// `compose::compose_vectors` as one resident chain (`chain::compose_vectors_enqueue`).
    fn compose_vectors(&self, _settings: &EngineSettings, _sel: &Selection, _a: &Image, _b: &Image, _flows: &Flows) -> Result<Image, StageError> {
        Err(StageError::Unsupported("no compose chain on this device".into()))
    }
    /// The frame cache counters `(hits, misses, evictions, bytes held)`; zero without a cache.
    fn cache_stats(&self) -> (u64, u64, u64, u64) {
        (0, 0, 0, 0)
    }
}

/// Every kernel source of the executor's program, in program order: the convolution stages,
/// the solver, the vector stages, the warp stages, the Weighted Blend stages and the forward
/// warp, each part once.
pub fn program_parts() -> Vec<&'static str> {
    let mut parts: Vec<&'static str> = Vec::new();
    let lists = [
        convstage::program_parts(),
        solver::program_parts(),
        vecstage::program_parts(),
        warpstage::program_parts(),
        wbstage::program_parts(),
        fwdstage::program_parts(),
    ];
    for list in lists {
        for p in list {
            if !parts.contains(&p) {
                parts.push(p);
            }
        }
    }
    parts
}

/// The whole program text of the executor (`kernels::program` of [`program_parts`]).
pub fn program() -> String {
    kernels::program(&program_parts())
}

/// An upper bound of the device bytes one solve needs: per working pixel 64 floats (both
/// pyramids with their gradients, the tensor planes, the flows and the scratch planes of every
/// level fit in that), plus the two frames as floats.
pub fn solve_bytes(x: &Image, s: &FlowSettings) -> Option<u64> {
    let geo = s.geometry(x.width, x.height).ok()?;
    let working = (geo.width as u64) * (geo.height as u64);
    let frame = (x.sample_count() as u64) * 4;
    Some(working * 64 * 4 + 2 * frame)
}

/// An upper bound of the device bytes of one forward warp of `img`: the source, the output, the
/// pyramid levels and the integer staging image (at most three images and a half), the float
/// passes (32 bytes a pixel), the per pixel counts and offsets (12), the hit records (at most
/// `FW_RECORDS_PER_PIXEL` of 16 bytes) and the coverage (1).
pub fn forward_bytes(img: &Image) -> u64 {
    let px = (img.width as u64) * (img.height as u64);
    let pixel = CHANNELS as u64 * sample_bytes(img.fmt());
    px * (pixel * 4 + 32 + 12 + crate::device::kernels::fwdstage::FW_RECORDS_PER_PIXEL as u64 * 16 + 1)
}

/// Bytes of one sample of `fmt`.
fn sample_bytes(fmt: Fmt) -> u64 {
    match fmt {
        Fmt::U8 => 1,
        Fmt::U16 => 2,
        Fmt::F32 => 4,
    }
}

/// An upper bound of the device bytes of one resident compose chain (`chain`): ten images of the
/// frame's size (the two frames, the output, two warped frames and the sharpening and blur
/// temporaries), the hole fill's planes (40 bytes a pixel), two coverage images, the two flows
/// and the forward warp's working set.
pub fn compose_bytes(img: &Image, flows: &Flows) -> u64 {
    let px = (img.width as u64) * (img.height as u64);
    let frame = img.sample_count() as u64 * sample_bytes(img.fmt());
    let flow = |f: &Option<FlowField>| f.as_ref().map_or(0, |f| f.data.len() as u64 * 4);
    frame * 10 + px * 42 + flow(&flows.f_ab) + flow(&flows.f_ba) + forward_bytes(img)
}

/// Device bytes of a resident pyramid: every level's floats.
fn pyramid_bytes(p: &DevicePyramid<Session>) -> u64 {
    p.geometry.sizes.iter().map(|&(w, h)| (w as u64) * (h as u64) * 4).sum()
}

struct Inner {
    session: Session,
    taps5: Option<Taps<Session>>,
    /// Resident pyramids of the frames solved last, keyed by the exact frame and settings.
    pyramids: Lru<ImageKey, DevicePyramid<Session>>,
}

/// One OpenCL device opened for a run: its session (context, in-order queue, the program built
/// once), the memory budget and the frame cache. A stage holds the one lock for its whole queue
/// sequence, so stages never interleave on the queue.
pub struct ClBackend {
    pub info: DeviceInfo,
    /// Bytes this run may place on the device (`probe::memory_budget`, 75% of global memory). A
    /// quarter of it is the frame cache, the rest the working set of one stage.
    pub budget: u64,
    inner: Mutex<Inner>,
    /// The thread inside a stage: a stage reached again on that thread (a nested call) runs on the
    /// CPU instead of waiting for itself.
    owner: Mutex<Option<ThreadId>>,
}

impl ClBackend {
    /// Builds the executor's program on `device` (the cold build costs 0.5 to 3 s).
    pub fn open(device: &cl::Device, info: DeviceInfo) -> Result<ClBackend, cl::ClError> {
        let session = Session::open(device, &program())?;
        let budget = probe::memory_budget(&info);
        let pyramids = Lru::new(budget / 4);
        Ok(ClBackend { info, budget, inner: Mutex::new(Inner { session, taps5: None, pyramids }), owner: Mutex::new(None) })
    }

    fn lock(&self) -> Result<Locked<'_>, StageError> {
        let me = thread::current().id();
        if *self.owner.lock().unwrap_or_else(|e| e.into_inner()) == Some(me) {
            return Err(StageError::Unsupported("the device is busy with a stage on this thread".into()));
        }
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        *self.owner.lock().unwrap_or_else(|e| e.into_inner()) = Some(me);
        Ok(Locked { inner, owner: &self.owner })
    }

    /// `solves` solves of frames like `x` fit in the working part of the budget.
    fn check_budget(&self, x: &Image, s: &FlowSettings, solves: u64) -> Result<(), StageError> {
        let working = self.budget - self.budget / 4;
        match solve_bytes(x, s) {
            Some(b) if b.saturating_mul(solves) <= working => Ok(()),
            Some(b) => Err(StageError::Unsupported(format!("{} bytes over the device memory budget of {working}", b.saturating_mul(solves)))),
            None => Err(StageError::Unsupported("optical flow: no geometry for this frame".into())),
        }
    }
}

struct Locked<'a> {
    inner: MutexGuard<'a, Inner>,
    owner: &'a Mutex<Option<ThreadId>>,
}

impl Drop for Locked<'_> {
    fn drop(&mut self) {
        *self.owner.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

fn check_images(x: &Image, y: &Image) -> Result<(), StageError> {
    if x.width != y.width || x.height != y.height || x.fmt() != y.fmt() {
        return Err(StageError::Unsupported("optical flow: incompatible inputs".into()));
    }
    Ok(())
}

impl Inner {
    /// Makes the pyramid of `img` resident (built on the device on a miss) and returns the digest
    /// of `img` for [`Inner::find`].
    fn pyramid(&mut self, img: &Image, s: &FlowSettings) -> Result<u64, StageError> {
        let d = digest(img);
        if self.pyramids.get(|k| k.matches(img, s, d)).is_some() {
            return Ok(d);
        }
        if self.taps5.is_none() {
            self.taps5 = Some(Taps::binomial5(&self.session)?);
        }
        let taps = self.taps5.as_ref().expect("set above");
        let p = build_pyramid_on(&self.session, img, s, taps)?;
        let bytes = pyramid_bytes(&p);
        if !self.pyramids.insert(ImageKey::new(img, s), p, bytes) {
            return Err(StageError::Unsupported("a pyramid larger than the frame cache".into()));
        }
        Ok(d)
    }

    fn find(&self, img: &Image, s: &FlowSettings, d: u64) -> Result<&DevicePyramid<Session>, StageError> {
        self.pyramids
            .peek(|k| k.matches(img, s, d))
            .ok_or_else(|| StageError::Unsupported("a frame left the cache during its own solve".into()))
    }

    /// The plain solve `X = x`, `Y = y` of two resident pyramids, the flow downloaded (a NaN is
    /// `StageError::NanOutput`).
    fn solve_resident(&self, x: (&Image, u64), y: (&Image, u64), s: &FlowSettings) -> Result<FlowField, StageError> {
        let none = DeviceExtras::default();
        let px = self.find(x.0, s, x.1)?;
        let py = self.find(y.0, s, y.1)?;
        let flow = solve_pyramids_with_enqueue(&self.session, px, py, s, &none)?;
        download_flow(&self.session, &flow, &px.geometry)
    }
}

impl Backend for ClBackend {
    fn solve(&self, x: &Image, y: &Image, s: &FlowSettings) -> Result<FlowField, StageError> {
        check_images(x, y)?;
        self.check_budget(x, s, 1)?;
        let mut g = self.lock()?;
        let dx = g.inner.pyramid(x, s)?;
        let dy = g.inner.pyramid(y, s)?;
        g.inner.solve_resident((x, dx), (y, dy), s)
    }

    fn solve_pair(&self, a: &Image, b: &Image, s: &FlowSettings) -> Result<FlowPair, StageError> {
        check_images(a, b)?;
        self.check_budget(a, s, 2)?;
        let mut g = self.lock()?;
        let da = g.inner.pyramid(a, s)?;
        let db = g.inner.pyramid(b, s)?;
        // the CPU's order: F_BA (X = B) first, then F_AB (X = A)
        let f_ba = g.inner.solve_resident((b, db), (a, da), s)?;
        let f_ab = g.inner.solve_resident((a, da), (b, db), s)?;
        Ok(FlowPair { f_ab, f_ba })
    }

    fn solve_pyramids_with(&self, x: &Pyramid, y: &Pyramid, s: &FlowSettings, e: SolveExtras<'_>) -> Result<FlowField, StageError> {
        if let Some(b) = solve_bytes_of(x) {
            if b > self.budget - self.budget / 4 {
                return Err(StageError::Unsupported(format!("{b} bytes over the device memory budget")));
            }
        }
        let g = self.lock()?;
        solver::solve_pyramids_with_gpu(&g.inner.session, x, y, s, e)
    }

    fn forward_warp_with(&self, out: &mut Image, src: &Image, flow: &FlowField, amount: f32, opts: ForwardOpts, want_coverage: bool) -> Result<Option<Coverage>, StageError> {
        let working = self.budget - self.budget / 4;
        let need = forward_bytes(src);
        if need > working {
            return Err(StageError::Unsupported(format!("{need} bytes over the device memory budget of {working}")));
        }
        let g = self.lock()?;
        fwdstage::forward_warp_with_gpu(&g.inner.session, out, src, flow, amount, opts, want_coverage)
    }

    fn compose_vectors(&self, settings: &EngineSettings, sel: &Selection, a: &Image, b: &Image, flows: &Flows) -> Result<Image, StageError> {
        let working = self.budget - self.budget / 4;
        let need = compose_bytes(a, flows);
        if need > working {
            return Err(StageError::Unsupported(format!("{need} bytes over the device memory budget of {working}")));
        }
        let g = self.lock()?;
        super::chain::compose_vectors_gpu(&g.inner.session, settings, sel, a, b, flows)
    }

    fn cache_stats(&self) -> (u64, u64, u64, u64) {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        (g.pyramids.hits, g.pyramids.misses, g.pyramids.evictions, g.pyramids.used())
    }
}

/// [`solve_bytes`] for host pyramids (the working size is the finest level).
fn solve_bytes_of(p: &Pyramid) -> Option<u64> {
    let geo = p.geometry();
    Some((geo.width as u64) * (geo.height as u64) * 64 * 4)
}
