//! Device stages of the Weighted Blend chain: `filters::sharpen_step` (the
//! difference mask and its clean-up, the blur, the unsharp lerp, the masked composite),
//! `blend::weighted_blend_noflow_reference`, `smartblend::weighted_tail`,
//! `smartblend::weighted_blend_smart` and `smartblend::flow_dissolve_inv`, built from the kernels
//! of `kernels::wbstage` (prefix `wb_`), the separable passes of `kernels::conv` (`cv_hpass`,
//! `cv_vpass` over the four interleaved lanes), the blend, lerp, inverse warp and Smart Blend
//! stages of [`crate::device::warpstage`] and the hole fill of [`crate::device::convstage`].
//!
//! For every ported CPU function `foo` there are two entry points:
//! * `foo_enqueue(session, <device images, flows, coverages>, ..., flag)` queues kernels and
//!   buffer copies only (it creates the device buffers it needs and the small tap buffer, it
//!   never moves image data between host and device). A refusal at run time (the warps) raises
//!   the [`Flag`]; the executor reads it after the chain ([`Flag::check`]) and reruns the stage on
//!   the CPU when it is raised. A float image downloaded after a chain holding a NaN is never
//!   accepted (the devices write other NaN payloads than the CPU).
//! * `foo_gpu(session, <the CPU function's arguments>)` uploads, enqueues, downloads, checks the
//!   flag and the NaN rule and returns the CPU function's result type with a [`StageError`] in
//!   place of its error. Every output argument is written only on success.
//!
//! Every argument the CPU refuses (`FxError::Mismatch`, `BadParam`, `Unsupported`) or panics on
//! (images of two shapes) is `StageError::Unsupported`, checked on the host before anything is
//! queued; so is an image without pixels. The CPU then gives its own result or error.
//!
//! Precision contract (V: the CPU functions are the reference and stay untouched): the steps run
//! in the CPU's order, each kernel in the CPU's operation order and precision. The scalars are
//! computed on the host by the CPU's own expressions: the blur taps (`filters::blur_taps`, an
//! f32 `powf`), the mask limit (`filters::mask_limit`), the lerp weight
//! (`filters::sharpen_weight`), the clean-up limits and values and every edge and source index
//! of `filters::mask_replicate_border` ([`border_args`]). The mask operations read a snapshot,
//! so they run from one buffer into another ([`clean_plan`]). The blur's two passes are
//! `cv_hpass` and `cv_vpass` with four channels (the sentinel `-1.0` and the renormalised border
//! of `filters::conv_pixel` are the convolution's multi channel border rule, the same
//! operations in the same order); at 8 and 16 bpc the samples are loaded as floats, stored and
//! reloaded between the passes and stored after the second one by the blur's `store_int`.
//! The hole fill that ends the compositions with coverage always runs on the device: without a
//! hole it is the identity there too (every pixel has the weight 1.0 and comes back as
//! `v / 1.0`, stored exactly), which saves reading the mask back on the host.
//!
//! Nothing here is left on the CPU; nothing here is a parity gap.

#[cfg(test)]
mod devtests;
#[cfg(test)]
mod tests;

use super::cl::{Access, Arg, Buffer, Range};
use super::convstage::holefill::{fill_image_enqueue, DeviceFiller, MaskSource};
use super::convstage::{self, Backend, Launch, Taps};
use super::image::{DeviceImage, DeviceSamples, StageError};
use super::kernels::wbstage::{self as kwb, BorderArgs};
use super::selfcheck::Session;
use super::warpstage::blend::{blend_enqueue, copy_enqueue, lerp_enqueue};
use super::warpstage::smartblend::{coverage_holes_enqueue, smart_blend_enqueue};
use super::warpstage::warp::inv_warp_enqueue;
use super::warpstage::{self, download_checked, image_arg, items, same_shape, suffix, DeviceCoverage, DeviceFlow, Flag};
use crate::filters::{blur_taps, mask_limit, sharpen_weight, BLUR_RADIUS, MASK_THRESHOLD};
use crate::flow::FlowField;
use crate::image::{Fmt, Image, CHANNELS};
use crate::smartblend::SmartOpts;
use crate::warp::{Coverage, WarpOpts};

/// The kernel sources these stages need, in program order (after the prelude): the convolution,
/// resample, pyramid and hole fill parts, the composition parts and [`kwb::WB`].
pub fn program_parts() -> Vec<&'static str> {
    let mut v = convstage::program_parts();
    v.extend(warpstage::program_parts());
    v.push(kwb::WB);
    v
}

/// The prelude and [`program_parts`].
pub fn program() -> String {
    super::kernels::program(&program_parts())
}

// ---- host plans ---------------------------------------------------------------------------------

/// One step of `filters::clean_mask` on a mask of the plan's size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaskOp {
    /// `mask_prune(limit)`.
    Prune(i32),
    /// `mask_dilate_rows`.
    Dilate,
    /// `mask_fill(limit, value)`.
    Fill(i32, u8),
    /// `mask_replicate_border`.
    Border(BorderArgs),
}

/// The arguments of `mask_replicate_border` on a `w x h` mask, by the CPU's expressions:
/// `p2 = p3 = 1.0`, `p4 = (w - 2) as f32`, `p5 = (h - 2) as f32`, the ordered edges and the
/// source rows and columns as `as i32`. `Unsupported` when one of its loops would run more often
/// than the kernels' constant bounds (sides beyond the f32 integer range only).
pub fn border_args(w: usize, h: usize) -> Result<BorderArgs, StageError> {
    let (wi, hi) = (w as i64, h as i64);
    let (p2, p3, p4, p5) = (1.0f32, 1.0f32, (wi - 2) as f32, (hi - 2) as f32);
    let (mut fy0, mut fy1) = (p3, p5);
    if p5 < p3 {
        fy0 = p5;
        fy1 = p3;
    }
    let (mut fx0, mut fx1) = (p2, p4);
    if p4 < p2 {
        fx0 = p4;
        fx1 = p2;
    }
    let a = BorderArgs {
        w: warpstage::side(w)?,
        h: warpstage::side(h)?,
        fx0,
        fx1,
        fy0,
        fy1,
        top: p3 as i32,
        y1: fy1 as i32,
        bottom: p5 as i32,
        left: p2 as i32,
        x1: fx1 as i32,
        right: p4 as i32,
    };
    // the CPU's loop counts: rows (float)y < fy0 from 0, rows y1..h, columns (float)x < fx0
    // from 0, columns x1 + j < w
    let above = (0..3).take_while(|y| (*y as f32) < fy0).count();
    let below = hi - i64::from(a.y1);
    let lefts = (0..3).take_while(|x| (*x as f32) < fx0).count();
    let rights = wi - i64::from(a.x1);
    if above > 2 || below > 3 || lefts > 2 || rights > 3 {
        return Err(StageError::Unsupported(format!("the border of a {w}x{h} mask: loops beyond the kernel bounds")));
    }
    Ok(a)
}

/// `filters::clean_mask` on a `w x h` mask as a list of steps, without the steps the CPU makes
/// the identity for this size (`mask_prune` needs `h >= 5` and `w >= 3`, `mask_fill` `h >= 3`
/// and `w >= 3`, `mask_dilate_rows` `h >= 3`, `mask_replicate_border` a pixel): the prune
/// limits 254 and 509, the dilation, twice the fill (254, 255) and the border, then the fill
/// (254, 128) and the border and, with the levels halved (`(x as f64 * 0.5 + 0.5) as i32`), the
/// fill (127, 64) and the border.
pub fn clean_plan(w: usize, h: usize) -> Result<Vec<MaskOp>, StageError> {
    let mut v = Vec::new();
    if w == 0 || h == 0 {
        // every step of the CPU is the identity on an empty mask
        return Ok(v);
    }
    let border = border_args(w, h)?;
    let can_fill = h >= 3 && w >= 3;
    for pass in 0..2i32 {
        if h >= 5 && w >= 3 {
            v.push(MaskOp::Prune(254 + 255 * pass));
        }
    }
    if h >= 3 {
        v.push(MaskOp::Dilate);
    }
    let mut fills = vec![(254i32, 255i32), (254, 255)];
    let (mut limit, mut value) = (254i32, 128i32);
    for _ in 0..2 {
        fills.push((limit, value));
        limit = (f64::from(limit) * 0.5 + 0.5) as i32;
        value = (f64::from(value) * 0.5 + 0.5) as i32;
    }
    for (limit, value) in fills {
        if can_fill {
            v.push(MaskOp::Fill(limit, value as u8));
        }
        v.push(MaskOp::Border(border));
    }
    Ok(v)
}

/// The scalars of `sharpen_step` at the position `t`: the blur taps, the mask limit of `fmt`
/// and the lerp weight.
pub fn sharpen_scalars(fmt: Fmt, t: f32) -> (Vec<f32>, f32, f32) {
    (blur_taps(BLUR_RADIUS), mask_limit(fmt, MASK_THRESHOLD), sharpen_weight(t))
}

// ---- launches -----------------------------------------------------------------------------------

fn run(s: &Session, name: &str, range: Range, args: &[Arg<'_>]) -> Result<(), StageError> {
    let mut k = s.kernel(name)?;
    s.queue.run(&mut k, range, args)?;
    Ok(())
}

fn n_arg(n: usize) -> Result<Arg<'static>, StageError> {
    Ok(Arg::U32(items(n)?))
}

fn shape_error(what: &str) -> StageError {
    StageError::Unsupported(format!("{what}: images of different shapes or without pixels"))
}

fn check_device(what: &str, imgs: &[&DeviceImage]) -> Result<(), StageError> {
    let first = imgs[0];
    if first.sample_count() == 0 || !imgs.iter().all(|i| same_shape(first, i)) {
        return Err(shape_error(what));
    }
    convstage::check_samples(first.sample_count(), what)
}

fn check_host(what: &str, imgs: &[&Image]) -> Result<(), StageError> {
    let first = imgs[0];
    if first.sample_count() == 0 || !imgs.iter().all(|i| first.same_shape(i)) {
        return Err(shape_error(what));
    }
    convstage::check_samples(first.sample_count(), what)
}

fn zeroed_like(s: &Session, img: &DeviceImage) -> Result<DeviceImage, StageError> {
    DeviceImage::zeroed(s, img.width, img.height, img.fmt())
}

/// `filters::change_mask(a, b)`: the difference mask and its clean-up, in a new mask buffer of
/// `a`'s size (`a` and `b` of one shape).
pub fn change_mask_enqueue(s: &Session, a: &DeviceImage, b: &DeviceImage) -> Result<Buffer<u8>, StageError> {
    check_device("change mask", &[a, b])?;
    let (w, h) = (a.width, a.height);
    let n = w * h;
    let plan = clean_plan(w, h)?;
    let m0 = Buffer::<u8>::new(&s.context, n, Access::ReadWrite)?;
    let m1 = Buffer::<u8>::new(&s.context, n, Access::ReadWrite)?;
    let limit = mask_limit(a.fmt(), MASK_THRESHOLD);
    run(s, &format!("wb_diff_mask_{}", suffix(a.fmt())), Range::D1(n), &[image_arg(a), image_arg(b), m0.arg(), Arg::F32(limit), n_arg(n)?])?;
    let bufs = [m0, m1];
    let mut cur = 0usize;
    let (wi, hi) = (Arg::I32(warpstage::side(w)?), Arg::I32(warpstage::side(h)?));
    for op in plan {
        let (src, dst) = (&bufs[cur], &bufs[1 - cur]);
        match op {
            MaskOp::Prune(limit) => {
                run(s, "wb_mask_prune", Range::D2(w, h), &[src.arg(), dst.arg(), wi, hi, Arg::I32(limit)])?;
                cur = 1 - cur;
            }
            MaskOp::Dilate => {
                run(s, "wb_mask_dilate", Range::D2(w, h), &[src.arg(), dst.arg(), wi, hi])?;
                cur = 1 - cur;
            }
            MaskOp::Fill(limit, value) => {
                run(s, "wb_mask_fill", Range::D2(w, h), &[src.arg(), dst.arg(), wi, hi, Arg::I32(limit), Arg::I32(i32::from(value))])?;
                cur = 1 - cur;
            }
            MaskOp::Border(a) => {
                let args = [
                    src.arg(),
                    wi,
                    hi,
                    Arg::F32(a.fx0),
                    Arg::F32(a.fx1),
                    Arg::F32(a.fy0),
                    Arg::F32(a.fy1),
                    Arg::I32(a.top),
                    Arg::I32(a.y1),
                    Arg::I32(a.bottom),
                ];
                run(s, "wb_border_rows", Range::D1(w), &args)?;
                run(s, "wb_border_cols", Range::D1(h), &[src.arg(), wi, hi, Arg::F32(a.fx0), Arg::I32(a.left), Arg::I32(a.x1), Arg::I32(a.right)])?;
            }
        }
    }
    let [b0, b1] = bufs;
    Ok(if cur == 0 { b0 } else { b1 })
}

/// `filters::blur(copy of src, 3.0)` into `dst` (one shape, distinct images).
pub fn blur_enqueue(s: &Session, src: &DeviceImage, dst: &DeviceImage) -> Result<(), StageError> {
    check_device("blur", &[src, dst])?;
    let (w, h) = (src.width, src.height);
    let n = src.sample_count();
    let taps = Taps::upload(s, &blur_taps(BLUR_RADIUS))?;
    match (&src.samples, &dst.samples) {
        (DeviceSamples::F32(sb), DeviceSamples::F32(db)) => {
            let tmp = s.alloc::<f32>(n)?;
            s.launch(Launch::HPass { src: sb, dst: &tmp, w, h, c: CHANNELS, taps: &taps, wrap: false })?;
            s.launch(Launch::VPass { src: &tmp, dst: db, w, h, c: CHANNELS, taps: &taps })
        }
        _ => {
            let sfx = suffix(src.fmt());
            let f0 = s.alloc::<f32>(n)?;
            let f1 = s.alloc::<f32>(n)?;
            run(s, &format!("wb_to_float_{sfx}"), Range::D1(n), &[image_arg(src), f0.arg(), n_arg(n)?])?;
            s.launch(Launch::HPass { src: &f0, dst: &f1, w, h, c: CHANNELS, taps: &taps, wrap: false })?;
            run(s, &format!("wb_round_{sfx}"), Range::D1(n), &[f1.arg(), f0.arg(), n_arg(n)?])?;
            s.launch(Launch::VPass { src: &f0, dst: &f1, w, h, c: CHANNELS, taps: &taps })?;
            run(s, &format!("wb_from_float_{sfx}"), Range::D1(n), &[f1.arg(), image_arg(dst), n_arg(n)?])
        }
    }
}

/// `filters::invert_mask` of a device mask of `n` pixels.
pub fn invert_mask_enqueue(s: &Session, mask: &Buffer<u8>, n: usize) -> Result<(), StageError> {
    run(s, "wb_invert_mask", Range::D1(n), &[mask.arg(), n_arg(n)?])
}

/// `filters::composite_masked(dest, src, mask)`, in place on `dest`.
pub fn composite_masked_enqueue(s: &Session, dest: &DeviceImage, src: &DeviceImage, mask: &Buffer<u8>) -> Result<(), StageError> {
    check_device("masked composite", &[dest, src])?;
    if mask.len() != dest.width * dest.height {
        return Err(StageError::Unsupported("masked composite: a mask of another size".into()));
    }
    let n = dest.sample_count();
    run(s, &format!("wb_composite_{}", suffix(dest.fmt())), Range::D1(n), &[image_arg(dest), image_arg(src), mask.arg(), n_arg(n)?])
}

/// The blur and the unsharp lerp of `sharpen_step` (`sharp = blur(frame)`,
/// `lerp(sharp, frame, sharpen_weight(t))`) into a new image.
fn sharpened(s: &Session, frame: &DeviceImage, t: f32) -> Result<DeviceImage, StageError> {
    let sharp = zeroed_like(s, frame)?;
    blur_enqueue(s, frame, &sharp)?;
    lerp_enqueue(s, &sharp, frame, sharpen_weight(t))?;
    Ok(sharp)
}

// ---- sharpen_step ---------------------------------------------------------------------------------

/// `filters::sharpen_step(frame, original, t)`, in place on `frame`: the change mask of `frame`
/// against `original`, the blurred and unsharp copy, the inverted mask and the masked composite.
/// No step raises `flag`.
pub fn sharpen_step_enqueue(s: &Session, frame: &DeviceImage, original: &DeviceImage, t: f32, flag: &Flag) -> Result<(), StageError> {
    let _ = flag;
    check_device("sharpen step", &[frame, original])?;
    let mask = change_mask_enqueue(s, frame, original)?;
    let sharp = sharpened(s, frame, t)?;
    invert_mask_enqueue(s, &mask, frame.width * frame.height)?;
    composite_masked_enqueue(s, frame, &sharp, &mask)
}

/// `filters::sharpen_step` on the device; `frame` changes only on success.
pub fn sharpen_step_gpu(s: &Session, frame: &mut Image, original: &Image, t: f32) -> Result<(), StageError> {
    check_host("sharpen step", &[frame, original])?;
    let f = DeviceImage::upload(s, frame)?;
    let o = DeviceImage::upload(s, original)?;
    let flag = Flag::new(s)?;
    sharpen_step_enqueue(s, &f, &o, t, &flag)?;
    flag.check(s, "sharpen step")?;
    *frame = download_checked(s, &f)?;
    Ok(())
}

// ---- weighted_blend_noflow_reference --------------------------------------------------------------

/// `blend::weighted_blend_noflow_reference(out, a, wa, b)`: `a' = sharpen_step(copy a, a, 1 - wa)`,
/// `b' = sharpen_step(copy b, b, wa)`, `out = blend(a', b', 1 - wa)`. No step raises `flag`.
pub fn weighted_blend_noflow_reference_enqueue(s: &Session, out: &DeviceImage, a: &DeviceImage, wa: f32, b: &DeviceImage, flag: &Flag) -> Result<(), StageError> {
    check_device("weighted blend reference", &[out, a, b])?;
    let t = 1.0f32 - wa;
    let a1 = zeroed_like(s, a)?;
    copy_enqueue(s, &a1, a)?;
    sharpen_step_enqueue(s, &a1, a, t, flag)?;
    let b1 = zeroed_like(s, b)?;
    copy_enqueue(s, &b1, b)?;
    sharpen_step_enqueue(s, &b1, b, wa, flag)?;
    blend_enqueue(s, out, &a1, &b1, t)
}

/// `blend::weighted_blend_noflow_reference` on the device (images of two shapes, which the CPU
/// panics on, are `Unsupported`).
pub fn weighted_blend_noflow_reference_gpu(s: &Session, out: &mut Image, a: &Image, wa: f32, b: &Image) -> Result<(), StageError> {
    check_host("weighted blend reference", &[out, a, b])?;
    let da = DeviceImage::upload(s, a)?;
    let db = DeviceImage::upload(s, b)?;
    let o = DeviceImage::zeroed(s, a.width, a.height, a.fmt())?;
    let flag = Flag::new(s)?;
    weighted_blend_noflow_reference_enqueue(s, &o, &da, wa, &db, &flag)?;
    flag.check(s, "weighted blend reference")?;
    *out = download_checked(s, &o)?;
    Ok(())
}

// ---- weighted_tail and weighted_blend_smart -------------------------------------------------------

/// The hole fill of `smart_finish` with the flag `false` on a device image, holes `holes`.
fn fill_enqueue(s: &Session, out: &DeviceImage, holes: &DeviceCoverage) -> Result<(), StageError> {
    if holes.width != out.width || holes.height != out.height {
        return Err(StageError::Unsupported("hole fill: a holes mask of another size".into()));
    }
    let filler = DeviceFiller::new(s, MaskSource::U8(&holes.data), out.width, out.height, false)?;
    fill_image_enqueue(s, &filler, out)
}

/// `smartblend::weighted_tail(out, ta, a, tb, b, wa, cov, opts)`: `ta` and `tb` (the warped
/// frames) are sharpened in place against `a` and `b`; with coverage `out = smart_blend(ta, tb,
/// 1 - wa)`, `cov.0` becomes the holes mask (`255 - max(cov.0, cov.1)`) and the hole fill runs
/// with the flag `false`; without coverage `out = blend(ta, tb, 1 - wa)`. `opts` is ignored, as
/// on the CPU. No step raises `flag`.
#[allow(clippy::too_many_arguments)]
pub fn weighted_tail_enqueue(
    s: &Session,
    out: &DeviceImage,
    ta: &DeviceImage,
    a: &DeviceImage,
    tb: &DeviceImage,
    b: &DeviceImage,
    wa: f32,
    cov: Option<(&DeviceCoverage, &DeviceCoverage)>,
    opts: SmartOpts,
    flag: &Flag,
) -> Result<(), StageError> {
    let _ = opts;
    check_device("weighted blend tail", &[out, ta, a, tb, b])?;
    if let Some((c1, c2)) = cov {
        if c1.width != c2.width || c1.height != c2.height || c1.width != out.width || c1.height != out.height {
            return Err(StageError::Unsupported("weighted blend tail: coverage images of another size".into()));
        }
    }
    let frac = 1.0f32 - wa;
    sharpen_step_enqueue(s, ta, a, frac, flag)?;
    sharpen_step_enqueue(s, tb, b, wa, flag)?;
    match cov {
        Some((c1, c2)) => {
            smart_blend_enqueue(s, out, ta, tb, frac)?;
            coverage_holes_enqueue(s, c1, c2)?;
            fill_enqueue(s, out, c1)
        }
        None => blend_enqueue(s, out, ta, tb, frac),
    }
}

/// `smartblend::weighted_tail` on the device: `out`, `ta`, `tb` and the first coverage image are
/// written on success only.
#[allow(clippy::too_many_arguments)]
pub fn weighted_tail_gpu(
    s: &Session,
    out: &mut Image,
    ta: &mut Image,
    a: &Image,
    tb: &mut Image,
    b: &Image,
    wa: f32,
    cov: Option<(&mut Coverage, &Coverage)>,
    opts: SmartOpts,
) -> Result<(), StageError> {
    check_host("weighted blend tail", &[out, ta, a, tb, b])?;
    let d = [DeviceImage::upload(s, ta)?, DeviceImage::upload(s, a)?, DeviceImage::upload(s, tb)?, DeviceImage::upload(s, b)?];
    let o = DeviceImage::zeroed(s, a.width, a.height, a.fmt())?;
    let flag = Flag::new(s)?;
    let (c1, c2, cov) = match cov {
        Some((c1, c2)) => {
            let (u1, u2) = (DeviceCoverage::upload(s, c1)?, DeviceCoverage::upload(s, c2)?);
            (Some(u1), Some(u2), Some(c1))
        }
        None => (None, None, None),
    };
    let pair = match (&c1, &c2) {
        (Some(x), Some(y)) => Some((x, y)),
        _ => None,
    };
    weighted_tail_enqueue(s, &o, &d[0], &d[1], &d[2], &d[3], wa, pair, opts, &flag)?;
    flag.check(s, "weighted blend tail")?;
    let new_out = download_checked(s, &o)?;
    let new_ta = download_checked(s, &d[0])?;
    let new_tb = download_checked(s, &d[2])?;
    let new_cov = match &c1 {
        Some(c) => Some(c.download(s)?),
        None => None,
    };
    *out = new_out;
    *ta = new_ta;
    *tb = new_tb;
    if let (Some(dst), Some(v)) = (cov, new_cov) {
        *dst = v;
    }
    Ok(())
}

/// `smartblend::weighted_blend_smart(out, a, flow_ba, b, flow_ab, wa, opts)`:
/// `A' = W(A, flow_ba, 1 - wa)` and `B' = W(B, flow_ab, wa)` (hold off, no wrap, mode 1, one
/// coverage image each), then [`weighted_tail_enqueue`] with coverage. A warp the CPU refuses or
/// one beyond the device bounds raises `flag`.
#[allow(clippy::too_many_arguments)]
pub fn weighted_blend_smart_enqueue(
    s: &Session,
    out: &DeviceImage,
    a: &DeviceImage,
    flow_ba: &DeviceFlow,
    b: &DeviceImage,
    flow_ab: &DeviceFlow,
    wa: f32,
    opts: SmartOpts,
    flag: &Flag,
) -> Result<(), StageError> {
    check_device("weighted blend smart", &[out, a, b])?;
    let frac = 1.0f32 - wa;
    let wopts = WarpOpts { hold: false, is360: false, mode: 1 };
    let ta = zeroed_like(s, a)?;
    let c1 = DeviceCoverage::filled(s, a.width, a.height, 0xff)?;
    inv_warp_enqueue(s, &ta, a, flow_ba, frac, wopts, Some(&c1), flag)?;
    let tb = zeroed_like(s, b)?;
    let c2 = DeviceCoverage::filled(s, b.width, b.height, 0xff)?;
    inv_warp_enqueue(s, &tb, b, flow_ab, wa, wopts, Some(&c2), flag)?;
    weighted_tail_enqueue(s, out, &ta, a, &tb, b, wa, Some((&c1, &c2)), opts, flag)
}

/// `smartblend::weighted_blend_smart` on the device.
#[allow(clippy::too_many_arguments)]
pub fn weighted_blend_smart_gpu(
    s: &Session,
    out: &mut Image,
    a: &Image,
    flow_ba: &FlowField,
    b: &Image,
    flow_ab: &FlowField,
    wa: f32,
    opts: SmartOpts,
) -> Result<(), StageError> {
    check_host("weighted blend smart", &[out, a, b])?;
    let wopts = WarpOpts { hold: false, is360: false, mode: 1 };
    // the warp plans first, so a refusal needs no upload
    warpstage::warp::warp_plan((a.width, a.height), (flow_ba.width, flow_ba.height), flow_ba.data.len(), 1.0f32 - wa, wopts)?;
    warpstage::warp::warp_plan((b.width, b.height), (flow_ab.width, flow_ab.height), flow_ab.data.len(), wa, wopts)?;
    let da = DeviceImage::upload(s, a)?;
    let db = DeviceImage::upload(s, b)?;
    let fba = DeviceFlow::upload(s, flow_ba)?;
    let fab = DeviceFlow::upload(s, flow_ab)?;
    let o = DeviceImage::zeroed(s, a.width, a.height, a.fmt())?;
    let flag = Flag::new(s)?;
    weighted_blend_smart_enqueue(s, &o, &da, &fba, &db, &fab, wa, opts, &flag)?;
    flag.check(s, "weighted blend smart")?;
    *out = download_checked(s, &o)?;
    Ok(())
}

// ---- flow_dissolve_inv ------------------------------------------------------------------------------

/// What `smartblend::flow_dissolve_inv` does for its scalars (the CPU's checks in its order).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DissolvePlan {
    /// Mode 0 (or a weight outside `[0.002, 0.998]`): a copy of A, or of B when `0.5 > wa`.
    Copy { of_b: bool },
    /// Mode 5: the two warps with hold, the sharpening steps and the blend.
    Warped,
    /// Mode 6: the change mask of the sources, the masked sharpening and the blend.
    Masked,
}

/// The host part of `flow_dissolve_inv`; its `BadParam` and `Unsupported` cases are
/// `StageError::Unsupported`.
pub fn dissolve_plan(has_ab: bool, has_ba: bool, wa: f32, mode: u32) -> Result<DissolvePlan, StageError> {
    if !has_ab && !has_ba && ((mode & 0xffff_fff9) != 0 || mode == 4) {
        return Err(StageError::Unsupported("flow dissolve: this mode needs flows".into()));
    }
    let mut mode = mode;
    if f64::from(wa) < 0.002 || 0.998 < f64::from(wa) {
        mode = 0;
    }
    match mode {
        0 => Ok(DissolvePlan::Copy { of_b: 0.5f32 > wa }),
        5 if has_ab && has_ba => Ok(DissolvePlan::Warped),
        5 => Err(StageError::Unsupported("flow dissolve mode 5 needs both flows".into())),
        6 => Ok(DissolvePlan::Masked),
        other => Err(StageError::Unsupported(format!("flow dissolve mode {other}"))),
    }
}

/// `smartblend::flow_dissolve_inv(out, a, flow_ab, b, flow_ba, wa, mode)`: the branch of
/// [`dissolve_plan`]. A warp the CPU refuses raises `flag`.
#[allow(clippy::too_many_arguments)]
pub fn flow_dissolve_inv_enqueue(
    s: &Session,
    out: &DeviceImage,
    a: &DeviceImage,
    flow_ab: Option<&DeviceFlow>,
    b: &DeviceImage,
    flow_ba: Option<&DeviceFlow>,
    wa: f32,
    mode: u32,
    flag: &Flag,
) -> Result<(), StageError> {
    check_device("flow dissolve", &[out, a, b])?;
    let frac = 1.0f32 - wa;
    match dissolve_plan(flow_ab.is_some(), flow_ba.is_some(), wa, mode)? {
        DissolvePlan::Copy { of_b } => copy_enqueue(s, out, if of_b { b } else { a }),
        DissolvePlan::Warped => {
            let (Some(fab), Some(fba)) = (flow_ab, flow_ba) else {
                return Err(StageError::Unsupported("flow dissolve mode 5 needs both flows".into()));
            };
            let wopts = WarpOpts { hold: true, is360: false, mode: 1 };
            let ta = zeroed_like(s, a)?;
            inv_warp_enqueue(s, &ta, a, fba, frac, wopts, None, flag)?;
            let tb = zeroed_like(s, b)?;
            inv_warp_enqueue(s, &tb, b, fab, wa, wopts, None, flag)?;
            sharpen_step_enqueue(s, &ta, a, frac, flag)?;
            sharpen_step_enqueue(s, &tb, b, wa, flag)?;
            blend_enqueue(s, out, &ta, &tb, frac)
        }
        DissolvePlan::Masked => {
            let mask = change_mask_enqueue(s, a, b)?;
            invert_mask_enqueue(s, &mask, a.width * a.height)?;
            let ta = zeroed_like(s, a)?;
            copy_enqueue(s, &ta, a)?;
            let sa = sharpened(s, &ta, frac)?;
            composite_masked_enqueue(s, &ta, &sa, &mask)?;
            let tb = zeroed_like(s, b)?;
            copy_enqueue(s, &tb, b)?;
            let sb = sharpened(s, &tb, wa)?;
            composite_masked_enqueue(s, &tb, &sb, &mask)?;
            blend_enqueue(s, out, &ta, &tb, frac)
        }
    }
}

/// `smartblend::flow_dissolve_inv` on the device.
#[allow(clippy::too_many_arguments)]
pub fn flow_dissolve_inv_gpu(
    s: &Session,
    out: &mut Image,
    a: &Image,
    flow_ab: Option<&FlowField>,
    b: &Image,
    flow_ba: Option<&FlowField>,
    wa: f32,
    mode: u32,
) -> Result<(), StageError> {
    check_host("flow dissolve", &[out, a, b])?;
    let plan = dissolve_plan(flow_ab.is_some(), flow_ba.is_some(), wa, mode)?;
    if let (DissolvePlan::Warped, Some(fab), Some(fba)) = (plan, flow_ab, flow_ba) {
        let wopts = WarpOpts { hold: true, is360: false, mode: 1 };
        warpstage::warp::warp_plan((a.width, a.height), (fba.width, fba.height), fba.data.len(), 1.0f32 - wa, wopts)?;
        warpstage::warp::warp_plan((b.width, b.height), (fab.width, fab.height), fab.data.len(), wa, wopts)?;
    }
    let da = DeviceImage::upload(s, a)?;
    let db = DeviceImage::upload(s, b)?;
    // the flows are uploaded where the plan warps; elsewhere a stand-in tells the enqueue which
    // flows the caller passed (the CPU never reads them there, whatever they hold)
    let flow = |f: Option<&FlowField>| -> Result<Option<DeviceFlow>, StageError> {
        match (f, plan) {
            (None, _) => Ok(None),
            (Some(f), DissolvePlan::Warped) => DeviceFlow::upload(s, f).map(Some),
            (Some(_), _) => Ok(Some(DeviceFlow { width: 2, height: 2, data: Buffer::from_slice(&s.context, &[0.0f32; 8], Access::ReadOnly)? })),
        }
    };
    let (fab, fba) = (flow(flow_ab)?, flow(flow_ba)?);
    let o = DeviceImage::zeroed(s, a.width, a.height, a.fmt())?;
    let flag = Flag::new(s)?;
    flow_dissolve_inv_enqueue(s, &o, &da, fab.as_ref(), &db, fba.as_ref(), wa, mode, &flag)?;
    flag.check(s, "flow dissolve")?;
    *out = download_checked(s, &o)?;
    Ok(())
}
