//! The motion solver on the device: `flow::solve`, `flow::solve_pair`, `flow::solve_pyramids` and
//! `flow::extras::solve_pyramids_with` (masks, user displacement, wrap) with the CPU's bytes.
//!
//! Kernel prefix `sv_` (`kernels/solver.rs`); the convolutions, the pyramid builders, the samplers
//! and the plain flow upsample are the convolution stages' (`convstage`: `blur_gaussian_above_on`,
//! `luma_upload` / `luma_enqueue`, `reduce_on`, `bands_on`, `upsample_flow_on`, the `rs_`
//! samplers). [`program_parts`] lists every source these stages need, the convolution stages'
//! first.
//!
//! Precision contract (V: the CPU solver is the reference and stays untouched): the driver below is
//! `flow::extras::solve_extended` (which is the plain driver when nothing extra is given) step for
//! step, one launch per CPU kernel call in the CPU's order, every kernel the CPU kernel's operation
//! order and precision per output (see `kernels/solver.rs`). In place CPU steps are kernels whose
//! items read and write only their own element (`new_disp`, the user displacement, the scales,
//! the zero fill of the range check) or, for the order dependent sweeps of the block matching,
//! their own row or column, stepped in the CPU's order and split into launches of at most
//! `SV_SWEEP_CHUNK` pixels (the flow buffer carries the state between launches, so the split
//! changes nothing). The range check and its `check` state stay on the device (a boolean OR, order
//! independent), so a solve queues without a host round trip. On the host, per solve: the start
//! level and the damping factor (`flow::start_level`, f64 `powf` and `ln`), the damping test
//! `(damp as f64 - 1.0).abs() >= 1e-6`, the Gaussian taps of radius 7 and 5
//! (`conv::gaussian_kernel`, f64 `powf`), the upsample scales; per pyramid, Delinearize and
//! Contrast / Edge Enhance between the device steps, as `convstage::pyramid` does.
//!
//! NaN rule: a result that holds a NaN is refused with [`StageError::NanOutput`] (the devices
//! write other NaN payloads than the CPU); NaN inputs are otherwise processed as on the CPU (every
//! decision is a comparison, a truncation through the range tested `cvtt` or a boolean, none
//! reads a payload), so a result without a NaN is the CPU's.
//!
//! Refused with `StageError::Unsupported` (the executor reruns the solve on the CPU): what the CPU
//! refuses (`FxError` from the geometry or a mismatch, mapped to its text), a level wider or taller
//! than `SV_LINE_MAX` or holding more than `convstage::MAX_SAMPLES` tensor samples, buffers
//! of another size than the geometry says, and the 360 solve (`FlowSettings::wrap`), whose rules
//! have no kernels yet.
//!
//! Entry points (the executor codes against these): [`solve_pyramids_with_enqueue`] on device
//! resident pyramids ([`DevicePyramid`], from [`build_pyramid_enqueue`] or
//! [`DevicePyramid::upload`]) returning the device flow at the working size, and the host in host
//! out [`solve_pyramids_with_gpu`], [`solve_pyramids_gpu`], [`solve_gpu`], [`solve_pair_gpu`]
//! with the CPU functions' arguments. Every `_gpu` function is a `_host` function over the
//! [`SolverBackend`] trait, which the device ([`Session`]) and the simulator of the tests
//! implement, so the simulator runs the very orchestration the device runs.
//!
//! Gaps: the solve records G6 under wrap and G1b where gated sweeps run, as the CPU does.

#[cfg(test)]
mod cases;
#[cfg(test)]
mod sim;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_device;

use super::cl::{Arg, Buffer, Range};
use super::convstage::conv::blur_gaussian_above_on;
use super::convstage::pyramid::{bands_on, luma_enqueue, luma_upload, reduce_on, upsample_flow_on, DevicePlane};
use super::convstage::{self, check_len, check_samples, Backend, Taps};
use super::image::{reject_nan, DeviceImage};
pub use super::image::StageError;
pub use super::selfcheck::Session;
use crate::device::kernels::{self, solver as ksv};
use crate::flow::extras::{MaskPyramid, SolveExtras, UserDisplacement};
use crate::flow::{self, FlowField, FlowGeometry, FlowPair, FlowSettings, Pyramid};
use crate::image::Image;
use crate::prep;

/// The sources these stages need, in program order (after the prelude): the convolution
/// stages' parts (convolution, samplers, resample, pyramid, hole fill), then the solver's.
pub fn program_parts() -> Vec<&'static str> {
    let mut parts = convstage::program_parts();
    parts.push(ksv::SOLVER);
    parts
}

/// The program of these stages alone: the prelude and [`program_parts`].
pub fn program() -> String {
    kernels::program(&program_parts())
}

/// One launch of a solver kernel. Optional buffers are passed as absent with a flag.
pub enum SvLaunch<'a, B: Backend + ?Sized> {
    /// `sv_grad`, `w` by `h` items.
    Grad { img: &'a B::Buf<f32>, mask: Option<&'a B::Buf<f32>>, wrap: bool, grad: &'a B::Buf<f32>, tensor: &'a B::Buf<f32>, w: usize, h: usize },
    /// `sv_calc_b`, `w` by `h` items.
    CalcB {
        i1: &'a B::Buf<f32>,
        i0: &'a B::Buf<f32>,
        grad: &'a B::Buf<f32>,
        flow: &'a B::Buf<f32>,
        mask: Option<&'a B::Buf<f32>>,
        wrap: bool,
        b: &'a B::Buf<f32>,
        w: usize,
        h: usize,
    },
    /// `sv_new_disp`, in place on `flow`, `n` pixels.
    NewDisp { flow: &'a B::Buf<f32>, a: &'a B::Buf<f32>, b: &'a B::Buf<f32>, n: usize },
    /// `sv_apply_user`, in place on `flow`, `n` pixels.
    ApplyUser { flow: &'a B::Buf<f32>, d: &'a B::Buf<f32>, wt: &'a B::Buf<f32>, n: usize },
    /// `sv_sweep_rows`, `h - 2` items, in place on `flow`.
    SweepRows { a: &'a B::Buf<f32>, b: &'a B::Buf<f32>, flow: &'a B::Buf<f32>, gate: Option<&'a B::Buf<f32>>, w: usize, h: usize, x0: usize, count: usize, dir: u8 },
    /// `sv_sweep_cols`, `ncols` items, in place on `flow`.
    SweepCols {
        a: &'a B::Buf<f32>,
        b: &'a B::Buf<f32>,
        flow: &'a B::Buf<f32>,
        gate: Option<&'a B::Buf<f32>>,
        w: usize,
        h: usize,
        xoff: usize,
        ncols: usize,
        y0: usize,
        count: usize,
        dir: u8,
        wrap: bool,
    },
    /// `sv_sweep_ring`, `w` by `h` items, `old` to `flow`.
    SweepRing { a: &'a B::Buf<f32>, b: &'a B::Buf<f32>, old: &'a B::Buf<f32>, flow: &'a B::Buf<f32>, gate: Option<&'a B::Buf<f32>>, w: usize, h: usize, right: bool },
    /// `sv_range_rows`, `h` items.
    RangeRows { flow: &'a B::Buf<f32>, state: &'a B::Buf<i32>, rows: &'a B::Buf<i32>, w: usize, h: usize, wrap: bool },
    /// `sv_range_decide`, one item, in place on `state`.
    RangeDecide { state: &'a B::Buf<i32>, rows: &'a B::Buf<i32>, h: usize },
    /// `sv_range_apply`, in place on the `n` values of `flow`.
    RangeApply { flow: &'a B::Buf<f32>, state: &'a B::Buf<i32>, n: usize },
    /// `sv_scale`, in place on `n` values.
    Scale { data: &'a B::Buf<f32>, f: f32, n: usize },
    /// `sv_zero`, `n` values.
    Zero { data: &'a B::Buf<f32>, n: usize },
    /// `sv_upsample_flow_wrapx`, `dw` by `dh` items; `scale` is `[sx, sy, rx, ry]`.
    UpsampleWrapx { src: &'a B::Buf<f32>, sw: usize, sh: usize, dst: &'a B::Buf<f32>, dw: usize, dh: usize, scale: [f32; 4] },
}

/// A backend that runs the solver kernels as well as the convolution stages' kernels.
pub trait SolverBackend: Backend {
    fn sv_launch(&self, l: SvLaunch<'_, Self>) -> Result<(), StageError>;
}

fn i(v: usize) -> Arg<'static> {
    Arg::I32(v as i32)
}

fn flag(v: bool) -> Arg<'static> {
    Arg::I32(v as i32)
}

impl SolverBackend for Session {
    fn sv_launch(&self, l: SvLaunch<'_, Self>) -> Result<(), StageError> {
        // an absent optional buffer is passed as another live buffer of the launch with flag 0
        let (name, range, args): (&str, Range, Vec<Arg<'_>>) = match l {
            SvLaunch::Grad { img, mask, wrap, grad, tensor, w, h } => (
                "sv_grad",
                Range::D2(w, h),
                vec![img.arg(), mask.unwrap_or(img).arg(), flag(mask.is_some()), flag(wrap), grad.arg(), tensor.arg(), i(w), i(h)],
            ),
            SvLaunch::CalcB { i1, i0, grad, flow, mask, wrap, b, w, h } => (
                "sv_calc_b",
                Range::D2(w, h),
                vec![i1.arg(), i0.arg(), grad.arg(), flow.arg(), mask.unwrap_or(i0).arg(), flag(mask.is_some()), flag(wrap), b.arg(), i(w), i(h)],
            ),
            SvLaunch::NewDisp { flow, a, b, n } => ("sv_new_disp", Range::D1(n), vec![flow.arg(), a.arg(), b.arg(), i(n)]),
            SvLaunch::ApplyUser { flow, d, wt, n } => ("sv_apply_user", Range::D1(n), vec![flow.arg(), d.arg(), wt.arg(), i(n)]),
            SvLaunch::SweepRows { a, b, flow, gate, w, h, x0, count, dir } => (
                "sv_sweep_rows",
                Range::D1(h - 2),
                vec![a.arg(), b.arg(), flow.arg(), gate.unwrap_or(a).arg(), flag(gate.is_some()), i(w), i(h), i(x0), i(count), i(dir as usize)],
            ),
            SvLaunch::SweepCols { a, b, flow, gate, w, h, xoff, ncols, y0, count, dir, wrap } => (
                "sv_sweep_cols",
                Range::D1(ncols),
                vec![
                    a.arg(),
                    b.arg(),
                    flow.arg(),
                    gate.unwrap_or(a).arg(),
                    flag(gate.is_some()),
                    i(w),
                    i(h),
                    i(xoff),
                    i(ncols),
                    i(y0),
                    i(count),
                    i(dir as usize),
                    flag(wrap),
                ],
            ),
            SvLaunch::SweepRing { a, b, old, flow, gate, w, h, right } => (
                "sv_sweep_ring",
                Range::D2(w, h),
                vec![a.arg(), b.arg(), old.arg(), flow.arg(), gate.unwrap_or(a).arg(), flag(gate.is_some()), i(w), i(h), flag(right)],
            ),
            SvLaunch::RangeRows { flow, state, rows, w, h, wrap } => {
                ("sv_range_rows", Range::D1(h), vec![flow.arg(), state.arg(), rows.arg(), i(w), i(h), flag(wrap)])
            }
            SvLaunch::RangeDecide { state, rows, h } => ("sv_range_decide", Range::D1(1), vec![state.arg(), rows.arg(), i(h)]),
            SvLaunch::RangeApply { flow, state, n } => ("sv_range_apply", Range::D1(n), vec![flow.arg(), state.arg(), i(n)]),
            SvLaunch::Scale { data, f, n } => ("sv_scale", Range::D1(n), vec![data.arg(), Arg::F32(f), i(n)]),
            SvLaunch::Zero { data, n } => ("sv_zero", Range::D1(n), vec![data.arg(), i(n)]),
            SvLaunch::UpsampleWrapx { src, sw, sh, dst, dw, dh, scale } => (
                "sv_upsample_flow_wrapx",
                Range::D2(dw, dh),
                vec![
                    src.arg(),
                    i(sw),
                    i(sh),
                    dst.arg(),
                    i(dw),
                    i(dh),
                    Arg::F32(scale[0]),
                    Arg::F32(scale[1]),
                    Arg::F32(scale[2]),
                    Arg::F32(scale[3]),
                ],
            ),
        };
        let mut kernel = self.kernel(name)?;
        self.queue.run(&mut kernel, range, &args)?;
        Ok(())
    }
}

// ---- device resident inputs ---------------------------------------------------------------------

/// The band pass pyramid of one frame on the backend: the CPU's `Pyramid` levels, finest first.
pub struct DevicePyramid<B: Backend + ?Sized> {
    pub geometry: FlowGeometry,
    pub levels: Vec<DevicePlane<B>>,
}

impl<B: Backend> DevicePyramid<B> {
    /// Uploads the levels of a host pyramid.
    pub fn upload(b: &B, p: &Pyramid) -> Result<DevicePyramid<B>, StageError> {
        let geometry = p.geometry().clone();
        let levels = (0..geometry.levels).map(|l| DevicePlane::upload(b, p.level(l))).collect::<Result<Vec<_>, _>>()?;
        Ok(DevicePyramid { geometry, levels })
    }
}

/// One level of a user displacement on the backend: `d` (2 values per pixel) and `w` (1).
pub struct DeviceUserLevel<B: Backend + ?Sized> {
    pub d: B::Buf<f32>,
    pub w: B::Buf<f32>,
}

/// The extra inputs of a layered solve on the backend (`SolveExtras` resident): the mask
/// pyramid levels of `X` and of `Y` (finest first) and the user displacement levels.
pub struct DeviceExtras<'a, B: Backend + ?Sized> {
    pub masks: Option<(&'a [DevicePlane<B>], &'a [DevicePlane<B>])>,
    pub user: Option<&'a [DeviceUserLevel<B>]>,
}

impl<B: Backend + ?Sized> Default for DeviceExtras<'_, B> {
    fn default() -> Self {
        DeviceExtras { masks: None, user: None }
    }
}

/// The levels of a host mask pyramid on the backend.
pub fn upload_masks<B: Backend>(b: &B, m: &MaskPyramid) -> Result<Vec<DevicePlane<B>>, StageError> {
    let g = m.geometry();
    (0..g.levels)
        .map(|l| {
            let (w, h) = g.sizes[l];
            DevicePlane::upload(b, &prep::Plane { width: w, height: h, data: m.level(l).to_vec() })
        })
        .collect()
}

/// The levels of a host user displacement on the backend (an empty one gives no levels).
pub fn upload_user<B: Backend>(b: &B, u: &UserDisplacement) -> Result<Vec<DeviceUserLevel<B>>, StageError> {
    u.levels
        .iter()
        .map(|lv| {
            if lv.d.is_empty() || lv.w.is_empty() {
                return Err(StageError::Unsupported("user displacement: an empty level".into()));
            }
            Ok(DeviceUserLevel { d: b.upload(&lv.d)?, w: b.upload(&lv.w)? })
        })
        .collect()
}

fn unsupported(e: fx_core::FxError) -> StageError {
    StageError::Unsupported(e.to_string())
}

/// `flow::build_pyramid` from the luma plane at the output size: the reduction, Delinearize and
/// Edge Enhance on the host when enabled, the band pass levels; resident on the backend.
fn pyramid_from_luma<B: Backend>(b: &B, luma: DevicePlane<B>, geometry: FlowGeometry, s: &FlowSettings, taps5: &Taps<B>) -> Result<DevicePyramid<B>, StageError> {
    let wrap = s.wrap;
    let mut level0 = reduce_on(b, luma, geometry.reduction, wrap, taps5)?;
    if s.prep_delinearize || s.prep_contrast_edge {
        // a NaN here may be one the CPU keeps: compare the bits after the host steps, not before
        let data = b.download(&level0.buf)?;
        let mut p = prep::Plane { width: level0.width, height: level0.height, data };
        if s.prep_delinearize {
            prep::delinearize(&mut p);
        }
        if s.prep_contrast_edge {
            prep::edge_enhance_with(&mut p, wrap);
        }
        level0 = DevicePlane::upload(b, &p)?;
    }
    let levels = bands_on(b, level0, &geometry.sizes[..geometry.levels], wrap, taps5)?;
    Ok(DevicePyramid { geometry, levels })
}

/// `flow::build_pyramid` of a host image, resident on the backend.
pub fn build_pyramid_on<B: Backend>(b: &B, img: &Image, s: &FlowSettings, taps5: &Taps<B>) -> Result<DevicePyramid<B>, StageError> {
    let geometry = s.geometry(img.width, img.height).map_err(unsupported)?;
    let luma = luma_upload(b, img, s.track_alpha, s.video_range_luma)?;
    pyramid_from_luma(b, luma, geometry, s, taps5)
}

/// `flow::build_pyramid` of a device image (the executor's resident frames).
pub fn build_pyramid_enqueue(session: &Session, img: &DeviceImage, s: &FlowSettings, taps5: &Taps<Session>) -> Result<DevicePyramid<Session>, StageError> {
    let geometry = s.geometry(img.width, img.height).map_err(unsupported)?;
    let luma = luma_enqueue(session, img, s.track_alpha, s.video_range_luma)?;
    pyramid_from_luma(session, luma, geometry, s, taps5)
}

// ---- the driver ---------------------------------------------------------------------------------

/// The Gaussian taps of the solver: radius 7 (tensor and B vector) and 5 (flow).
struct SolverTaps<B: Backend> {
    g7: Option<Taps<B>>,
    g5: Option<Taps<B>>,
}

/// Checks the inputs as `solve_pyramids_with` does (geometry mismatches are refused) plus the
/// device limits; returns the user levels to use (an empty user displacement is none).
fn check_inputs<'a, B: Backend>(b: &B, x: &DevicePyramid<B>, y: &DevicePyramid<B>, e: &DeviceExtras<'a, B>) -> Result<Option<&'a [DeviceUserLevel<B>]>, StageError> {
    let geo = &x.geometry;
    if x.geometry != y.geometry {
        return Err(StageError::Unsupported("optical flow: pyramids of different geometry".into()));
    }
    if geo.levels == 0 || geo.sizes.len() < geo.levels || x.levels.len() != geo.levels || y.levels.len() != geo.levels {
        return Err(StageError::Unsupported("optical flow: a device pyramid of another level count than its geometry".into()));
    }
    for l in 0..geo.levels {
        let (w, h) = geo.sizes[l];
        if w < 2 || h < 2 {
            return Err(StageError::Unsupported(format!("optical flow: a {w}x{h} level")));
        }
        if w > ksv::SV_LINE_MAX || h > ksv::SV_LINE_MAX {
            return Err(StageError::Unsupported(format!("optical flow: a {w}x{h} level (the device solver takes sides up to {})", ksv::SV_LINE_MAX)));
        }
        check_samples(w * h * 3, "optical flow")?;
        for p in [&x.levels[l], &y.levels[l]] {
            if (p.width, p.height) != (w, h) {
                return Err(StageError::Unsupported("optical flow: a device level of another size than its geometry".into()));
            }
            check_len(b, &p.buf, w * h, "optical flow")?;
        }
        if let Some((mx, my)) = e.masks {
            if mx.len() != geo.levels || my.len() != geo.levels {
                return Err(StageError::Unsupported("optical flow: a mask pyramid of another geometry than the image pyramids".into()));
            }
            for p in [&mx[l], &my[l]] {
                if (p.width, p.height) != (w, h) {
                    return Err(StageError::Unsupported("optical flow: a mask pyramid of another geometry than the image pyramids".into()));
                }
                check_len(b, &p.buf, w * h, "optical flow")?;
            }
        }
    }
    let user = match e.user {
        Some(u) if !u.is_empty() => {
            if u.len() != geo.levels {
                return Err(StageError::Unsupported(format!(
                    "optical flow: a user displacement of {} levels for a pyramid of {}",
                    u.len(),
                    geo.levels
                )));
            }
            for (l, lv) in u.iter().enumerate() {
                let (w, h) = geo.sizes[l];
                if b.length(&lv.d) != w * h * 2 || b.length(&lv.w) != w * h {
                    return Err(StageError::Unsupported(format!("optical flow: user displacement level {l} of another size than its {w}x{h} level")));
                }
            }
            Some(u)
        }
        _ => None,
    };
    Ok(user)
}

/// The block matching of one level (`block_match` / `block_match_with`): nothing below 3x3; the
/// gated sweeps record G1b.
#[allow(clippy::too_many_arguments)]
fn refine_on<B: SolverBackend>(b: &B, a: &B::Buf<f32>, bb: &B::Buf<f32>, flow: &B::Buf<f32>, w: usize, h: usize, gate: Option<&B::Buf<f32>>, wrap: bool) -> Result<(), StageError> {
    if w < 3 || h < 3 {
        return Ok(());
    }
    let chunk = ksv::SV_SWEEP_CHUNK;
    if wrap {
        // sweeps 1 and 2 of sweeps_ring: every pixel against the row as it stood before
        let old = b.alloc::<f32>(w * h * 2)?;
        for right in [false, true] {
            b.copy(flow, &old)?;
            b.sv_launch(SvLaunch::SweepRing { a, b: bb, old: &old, flow, gate, w, h, right })?;
        }
    } else {
        // sweep 1: x = 1 ..= w - 2 increasing
        let mut x0 = 1;
        while x0 <= w - 2 {
            let count = chunk.min(w - 1 - x0);
            b.sv_launch(SvLaunch::SweepRows { a, b: bb, flow, gate, w, h, x0, count, dir: 0 })?;
            x0 += count;
        }
        // sweep 2: x = w - 2 down to 1
        let mut x0 = w - 2;
        while x0 >= 1 {
            let count = chunk.min(x0);
            b.sv_launch(SvLaunch::SweepRows { a, b: bb, flow, gate, w, h, x0, count, dir: 1 })?;
            x0 -= count;
        }
    }
    // sweeps 3 and 4 along the columns: the interior ones, every one under wrap
    let (xoff, ncols) = if wrap { (0, w) } else { (1, w - 2) };
    let mut y0 = 1;
    while y0 <= h - 2 {
        let count = chunk.min(h - 1 - y0);
        b.sv_launch(SvLaunch::SweepCols { a, b: bb, flow, gate, w, h, xoff, ncols, y0, count, dir: 0, wrap })?;
        y0 += count;
    }
    let mut y0 = h - 2;
    while y0 >= 1 {
        let count = chunk.min(y0);
        b.sv_launch(SvLaunch::SweepCols { a, b: bb, flow, gate, w, h, xoff, ncols, y0, count, dir: 1, wrap })?;
        y0 -= count;
    }
    Ok(())
}

/// `flow::extras::level_step` of one level on the backend.
#[allow(clippy::too_many_arguments)]
fn level_on<B: SolverBackend>(
    b: &B,
    img_a: &B::Buf<f32>,
    img_b: &B::Buf<f32>,
    w: usize,
    h: usize,
    mask_x: Option<&B::Buf<f32>>,
    mask_y: Option<&B::Buf<f32>>,
    user: Option<&DeviceUserLevel<B>>,
    s: &FlowSettings,
    finest: bool,
    flow: &B::Buf<f32>,
    taps: &SolverTaps<B>,
) -> Result<(), StageError> {
    let wrap = s.wrap;
    if wrap {
        // the 360 rules of the solver (the column walk of its filters, the 360 gradient, the B
        // step and the block matching over the poles) have no kernels yet: the CPU solves
        return Err(StageError::Unsupported("the 360 solve is not ported to the device yet".into()));
    }
    let n = w * h;
    if let Some(u) = user {
        b.sv_launch(SvLaunch::ApplyUser { flow, d: &u.d, wt: &u.w, n })?;
    }
    let grad = b.alloc::<f32>(n * 2)?;
    let tensor = b.alloc::<f32>(n * 3)?;
    b.sv_launch(SvLaunch::Grad { img: img_a, mask: mask_x, wrap, grad: &grad, tensor: &tensor, w, h })?;
    let tmp3 = b.alloc::<f32>(n * 3)?;
    // the three Gaussians of a level take the tap above first in their vertical border rows
    // (`conv::blur_gaussian_above_first`)
    blur_gaussian_above_on(b, &tensor, &tmp3, &tensor, w, h, 3, taps.g7.as_ref())?;
    let bvec = b.alloc::<f32>(n * 2)?;
    let tmp2 = b.alloc::<f32>(n * 2)?;
    for it in 0..3 {
        b.sv_launch(SvLaunch::CalcB { i1: img_a, i0: img_b, grad: &grad, flow, mask: mask_y, wrap, b: &bvec, w, h })?;
        blur_gaussian_above_on(b, &bvec, &tmp2, &bvec, w, h, 2, taps.g7.as_ref())?;
        b.sv_launch(SvLaunch::NewDisp { flow, a: &tensor, b: &bvec, n })?;
        blur_gaussian_above_on(b, flow, &tmp2, flow, w, h, 2, taps.g5.as_ref())?;
        if it == 1 && finest && !s.skip_refine {
            refine_on(b, img_a, img_b, flow, w, h, mask_x, wrap)?;
        }
    }
    if !finest && !s.skip_refine {
        refine_on(b, img_a, img_b, flow, w, h, mask_x, wrap)?;
    }
    if let Some(u) = user {
        b.sv_launch(SvLaunch::ApplyUser { flow, d: &u.d, wt: &u.w, n })?;
    }
    Ok(())
}

/// `[sx, sy, rx, ry]` of `flow::upsample_flow_wrapx`: `sx = dw / sw`, `rx = sw / dw` in single
/// precision, `sy` and `ry` as in `upsample_flow`.
pub fn upsample_wrapx_scales(sw: usize, sh: usize, dw: usize, dh: usize) -> [f32; 4] {
    let sx = dw as f32 / sw as f32;
    let rx = sw as f32 / dw as f32;
    let sy = (dh as f32 - 1.0f32) / ((sh as i32 - 1) as f32);
    let ry = (1.0f64 / (sy as f64)) as f32;
    [sx, sy, rx, ry]
}

/// The driver of `solve_pyramids_with` on the backend; `trace(l, flow)` sees the flow of level
/// `l` after its range check and damping (before the upsample). Returns the device flow of the
/// finest level after the final negation.
fn solve_traced<B: SolverBackend>(
    b: &B,
    x: &DevicePyramid<B>,
    y: &DevicePyramid<B>,
    s: &FlowSettings,
    e: &DeviceExtras<'_, B>,
    trace: &mut dyn FnMut(usize, &B::Buf<f32>) -> Result<(), StageError>,
) -> Result<B::Buf<f32>, StageError> {
    let user = check_inputs(b, x, y, e)?;
    let geo = &x.geometry;
    let levels = geo.levels;
    let wrap = s.wrap;
    if !(s.sensitivity >= 0.0) || !s.sensitivity.is_finite() {
        return Err(StageError::Unsupported(format!("sensitivity {} must be finite and not negative", s.sensitivity)));
    }
    if wrap {
        fx_core::gaps::record(&crate::gaps::G6);
    }
    let taps = SolverTaps { g7: Taps::gaussian(b, 7.0)?, g5: Taps::gaussian(b, 5.0)? };
    let (start, damp) = flow::start_level(s.sensitivity, levels);
    let damped = (damp as f64 - 1.0).abs() >= 1e-6;
    let (sw0, sh0) = geo.sizes[start];
    let mut cur = b.alloc::<f32>(sw0 * sh0 * 2)?;
    b.sv_launch(SvLaunch::Zero { data: &cur, n: sw0 * sh0 * 2 })?;
    let state = b.upload(&[1i32, 0])?;
    for l in (0..=start).rev() {
        let (w, h) = geo.sizes[l];
        let mask_x = e.masks.map(|(m, _)| &m[l].buf);
        let mask_y = e.masks.map(|(_, m)| &m[l].buf);
        let u = user.map(|u| &u[l]);
        level_on(b, &x.levels[l].buf, &y.levels[l].buf, w, h, mask_x, mask_y, u, s, l == 0, &cur, &taps)?;
        let rows = b.alloc::<i32>(h)?;
        b.sv_launch(SvLaunch::RangeRows { flow: &cur, state: &state, rows: &rows, w, h, wrap })?;
        b.sv_launch(SvLaunch::RangeDecide { state: &state, rows: &rows, h })?;
        b.sv_launch(SvLaunch::RangeApply { flow: &cur, state: &state, n: w * h * 2 })?;
        if l == start && damped {
            b.sv_launch(SvLaunch::Scale { data: &cur, f: damp, n: w * h * 2 })?;
        }
        trace(l, &cur)?;
        if l > 0 {
            let (nw, nh) = geo.sizes[l - 1];
            let next = b.alloc::<f32>(nw * nh * 2)?;
            if wrap {
                b.sv_launch(SvLaunch::UpsampleWrapx { src: &cur, sw: w, sh: h, dst: &next, dw: nw, dh: nh, scale: upsample_wrapx_scales(w, h, nw, nh) })?;
            } else {
                upsample_flow_on(b, &cur, w, h, &next, nw, nh)?;
            }
            cur = next;
        }
    }
    b.sv_launch(SvLaunch::Scale { data: &cur, f: -1.0f32, n: geo.width * geo.height * 2 })?;
    Ok(cur)
}

/// `solve_pyramids_with` on device resident pyramids and extras: the queued flow of the finest
/// level (`geometry.width * geometry.height` vectors, interleaved), NaN not yet checked.
pub fn solve_pyramids_with_on<B: SolverBackend>(b: &B, x: &DevicePyramid<B>, y: &DevicePyramid<B>, s: &FlowSettings, e: &DeviceExtras<'_, B>) -> Result<B::Buf<f32>, StageError> {
    solve_traced(b, x, y, s, e, &mut |_, _| Ok(()))
}

/// `solve_pyramids_with` on the device, everything resident (see [`solve_pyramids_with_on`]).
pub fn solve_pyramids_with_enqueue(
    session: &Session,
    x: &DevicePyramid<Session>,
    y: &DevicePyramid<Session>,
    s: &FlowSettings,
    e: &DeviceExtras<'_, Session>,
) -> Result<Buffer<f32>, StageError> {
    solve_pyramids_with_on(session, x, y, s, e)
}

/// A device flow back on the host as a `FlowField` of the geometry's working size; a NaN is
/// `StageError::NanOutput`.
pub fn download_flow<B: Backend>(b: &B, buf: &B::Buf<f32>, geometry: &FlowGeometry) -> Result<FlowField, StageError> {
    let data = b.download(buf)?;
    if data.len() != geometry.width * geometry.height * 2 {
        return Err(StageError::Unsupported("optical flow: a device flow of another size than the geometry".into()));
    }
    reject_nan(&data)?;
    Ok(FlowField { width: geometry.width, height: geometry.height, data })
}

/// The pyramid geometry checks of the CPU entry points, before anything is uploaded.
fn check_host(x: &Pyramid, y: &Pyramid, e: &SolveExtras<'_>) -> Result<(), StageError> {
    if x.geometry() != y.geometry() {
        return Err(StageError::Unsupported("optical flow: pyramids of different geometry".into()));
    }
    if let Some((mx, my)) = e.masks {
        if mx.geometry() != x.geometry() || my.geometry() != x.geometry() {
            return Err(StageError::Unsupported("optical flow: a mask pyramid of another geometry than the image pyramids".into()));
        }
    }
    if let Some(u) = e.user {
        let geo = x.geometry();
        if !u.levels.is_empty() {
            if u.levels.len() != geo.levels {
                return Err(StageError::Unsupported("optical flow: a user displacement of another level count".into()));
            }
            for (lv, &(w, h)) in u.levels.iter().zip(geo.sizes.iter()) {
                if lv.d.len() != w * h * 2 || lv.w.len() != w * h {
                    return Err(StageError::Unsupported("optical flow: a user displacement level of another size".into()));
                }
            }
        }
    }
    Ok(())
}

/// `solve_pyramids_with` of host inputs on the backend, with the per level trace.
fn solve_pyramids_with_traced<B: SolverBackend>(
    b: &B,
    x: &Pyramid,
    y: &Pyramid,
    s: &FlowSettings,
    e: SolveExtras<'_>,
    trace: &mut dyn FnMut(usize, &B::Buf<f32>) -> Result<(), StageError>,
) -> Result<FlowField, StageError> {
    check_host(x, y, &e)?;
    let dx = DevicePyramid::upload(b, x)?;
    let dy = DevicePyramid::upload(b, y)?;
    let masks = match e.masks {
        Some((mx, my)) => Some((upload_masks(b, mx)?, upload_masks(b, my)?)),
        None => None,
    };
    let user = match e.user {
        Some(u) if !u.levels.is_empty() => Some(upload_user(b, u)?),
        _ => None,
    };
    let de = DeviceExtras { masks: masks.as_ref().map(|(a, c)| (a.as_slice(), c.as_slice())), user: user.as_deref() };
    let flow = solve_traced(b, &dx, &dy, s, &de, trace)?;
    download_flow(b, &flow, &dx.geometry)
}

/// `flow::extras::solve_pyramids_with` of host inputs on the backend.
pub fn solve_pyramids_with_host<B: SolverBackend>(b: &B, x: &Pyramid, y: &Pyramid, s: &FlowSettings, e: SolveExtras<'_>) -> Result<FlowField, StageError> {
    solve_pyramids_with_traced(b, x, y, s, e, &mut |_, _| Ok(()))
}

/// `flow::extras::solve_pyramids_with` on the device: its arguments, its result or a refusal.
pub fn solve_pyramids_with_gpu(session: &Session, x: &Pyramid, y: &Pyramid, s: &FlowSettings, e: SolveExtras<'_>) -> Result<FlowField, StageError> {
    solve_pyramids_with_host(session, x, y, s, e)
}

/// `flow::solve_pyramids` on the device (with wrap it is `solve_pyramids_with` and no extras, as on
/// the CPU).
pub fn solve_pyramids_gpu(session: &Session, x: &Pyramid, y: &Pyramid, s: &FlowSettings) -> Result<FlowField, StageError> {
    solve_pyramids_with_host(session, x, y, s, SolveExtras::default())
}

fn check_images(x: &Image, y: &Image) -> Result<(), StageError> {
    if x.width != y.width || x.height != y.height || x.fmt() != y.fmt() {
        return Err(StageError::Unsupported("optical flow: incompatible inputs".into()));
    }
    Ok(())
}

/// `flow::solve` of host images on the backend: both pyramids built and kept on the backend.
pub fn solve_host<B: SolverBackend>(b: &B, x: &Image, y: &Image, s: &FlowSettings) -> Result<FlowField, StageError> {
    check_images(x, y)?;
    let taps5 = Taps::binomial5(b)?;
    let px = build_pyramid_on(b, x, s, &taps5)?;
    let py = build_pyramid_on(b, y, s, &taps5)?;
    let flow = solve_pyramids_with_on(b, &px, &py, s, &DeviceExtras::default())?;
    download_flow(b, &flow, &px.geometry)
}

/// `flow::solve` on the device.
pub fn solve_gpu(session: &Session, x: &Image, y: &Image, s: &FlowSettings) -> Result<FlowField, StageError> {
    solve_host(session, x, y, s)
}

/// `flow::solve_pair` of host images on the backend: the pyramids built once and kept on the
/// backend, `f_ba` solved with `X = B`, then `f_ab` with `X = A` (one in-order queue).
pub fn solve_pair_host<B: SolverBackend>(b: &B, a: &Image, bimg: &Image, s: &FlowSettings) -> Result<FlowPair, StageError> {
    check_images(a, bimg)?;
    let taps5 = Taps::binomial5(b)?;
    let pa = build_pyramid_on(b, a, s, &taps5)?;
    let pb = build_pyramid_on(b, bimg, s, &taps5)?;
    let none = DeviceExtras::default();
    let f_ba = solve_pyramids_with_on(b, &pb, &pa, s, &none)?;
    let f_ab = solve_pyramids_with_on(b, &pa, &pb, s, &none)?;
    Ok(FlowPair { f_ab: download_flow(b, &f_ab, &pa.geometry)?, f_ba: download_flow(b, &f_ba, &pa.geometry)? })
}

/// `flow::solve_pair` on the device.
pub fn solve_pair_gpu(session: &Session, a: &Image, b: &Image, s: &FlowSettings) -> Result<FlowPair, StageError> {
    solve_pair_host(session, a, b, s)
}
