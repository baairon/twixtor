//! Device stages of `blend.rs`: [`copy_gpu`], [`blend_gpu`], [`lerp_gpu`] and
//! [`weighted_blend_noflow_gpu`] (its fast path; the full sequence with the sharpening step is
//! `wbstage::weighted_blend_noflow_reference_gpu`). The scalar branches of `blend` and `lerp` are taken on the host
//! ([`blend_plan`], [`lerp_plan`]), exactly as the CPU takes them; the per sample branch runs
//! in the kernels `bl_blend_*` and `bl_lerp_*`.

use super::{copy_image, download_checked, image_arg, items, same_shape, suffix, Flag};
use crate::device::cl::{Arg, Range};
use crate::device::image::{DeviceImage, StageError};
use crate::device::selfcheck::Session;
use crate::image::{Fmt, Image};

/// The branch of `blend::blend` for a weight (the CPU's comparisons, in its order).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlendPlan {
    /// `0.005 > |w|`: a copy of A.
    CopyA,
    /// `0.005 > |w - 1|`: a copy of B.
    CopyB,
    /// `w` outside 0..=1: a copy of A, then `lerp(out, B, w2)` with `w2 = (1 - w)` narrowed.
    Extrapolate(f32),
    /// The per sample mix.
    Mix,
}

pub fn blend_plan(w: f32) -> BlendPlan {
    const SNAP: f64 = 0.005;
    if SNAP > f64::from(w.abs()) {
        return BlendPlan::CopyA;
    }
    if SNAP > (f64::from(w) - 1.0).abs() {
        return BlendPlan::CopyB;
    }
    if w < 0.0 || w > 1.0 {
        return BlendPlan::Extrapolate((1.0f64 - f64::from(w)) as f32);
    }
    BlendPlan::Mix
}

/// The branch of `blend::lerp` for a weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LerpPlan {
    /// `w == 1.0`: `first` unchanged.
    Keep,
    /// `w == 0.0`: `first = second`.
    CopySecond,
    /// The row kernels.
    Rows,
}

pub fn lerp_plan(w: f32) -> LerpPlan {
    if w == 1.0 {
        LerpPlan::Keep
    } else if w == 0.0 {
        LerpPlan::CopySecond
    } else {
        LerpPlan::Rows
    }
}

fn check2(what: &str, a: &DeviceImage, b: &DeviceImage) -> Result<(), StageError> {
    if same_shape(a, b) {
        Ok(())
    } else {
        Err(StageError::Unsupported(format!("{what}: images of different shapes")))
    }
}

fn check_host(what: &str, a: &Image, b: &Image) -> Result<(), StageError> {
    if a.same_shape(b) && a.sample_count() > 0 {
        Ok(())
    } else {
        Err(StageError::Unsupported(format!("{what}: images of different shapes or without pixels")))
    }
}

/// `blend::copy`: queues a copy of `src` into `out`.
pub fn copy_enqueue(session: &Session, out: &DeviceImage, src: &DeviceImage) -> Result<(), StageError> {
    copy_image(session, out, src)
}

/// `blend::copy` on the device.
pub fn copy_gpu(session: &Session, out: &mut Image, src: &Image) -> Result<(), StageError> {
    check_host("copy", out, src)?;
    let s = DeviceImage::upload(session, src)?;
    let o = DeviceImage::zeroed(session, src.width, src.height, src.fmt())?;
    copy_enqueue(session, &o, &s)?;
    *out = download_checked(session, &o)?;
    Ok(())
}

/// `blend::lerp` (in place on `first`): queues the branch of [`lerp_plan`].
pub fn lerp_enqueue(session: &Session, first: &DeviceImage, second: &DeviceImage, w: f32) -> Result<(), StageError> {
    check2("lerp", first, second)?;
    match lerp_plan(w) {
        LerpPlan::Keep => Ok(()),
        LerpPlan::CopySecond => copy_image(session, first, second),
        LerpPlan::Rows => {
            let pixels = first.width * first.height;
            let mut kernel = session.kernel(&format!("bl_lerp_{}", suffix(first.fmt())))?;
            let args = [image_arg(first), image_arg(second), Arg::F32(w), Arg::U32(items(pixels)?)];
            session.queue.run(&mut kernel, Range::D1(pixels), &args)?;
            Ok(())
        }
    }
}

/// `blend::lerp` on the device.
pub fn lerp_gpu(session: &Session, first: &mut Image, second: &Image, w: f32) -> Result<(), StageError> {
    check_host("lerp", first, second)?;
    let f = DeviceImage::upload(session, first)?;
    let s = DeviceImage::upload(session, second)?;
    lerp_enqueue(session, &f, &s, w)?;
    *first = download_checked(session, &f)?;
    Ok(())
}

/// The per sample mix (branch 4) of `blend::blend`.
fn mix_enqueue(session: &Session, out: &DeviceImage, a: &DeviceImage, b: &DeviceImage, w: f32) -> Result<(), StageError> {
    let n = out.sample_count();
    let mut kernel = session.kernel(&format!("bl_blend_{}", suffix(out.fmt())))?;
    let args = [image_arg(a), image_arg(b), image_arg(out), Arg::F32(w), Arg::U32(items(n)?)];
    session.queue.run(&mut kernel, Range::D1(n), &args)?;
    Ok(())
}

/// `blend::blend`: queues the branch of [`blend_plan`]. `out` must not be `a` or `b`.
pub fn blend_enqueue(session: &Session, out: &DeviceImage, a: &DeviceImage, b: &DeviceImage, w: f32) -> Result<(), StageError> {
    check2("blend", out, a)?;
    check2("blend", out, b)?;
    match blend_plan(w) {
        BlendPlan::CopyA => copy_image(session, out, a),
        BlendPlan::CopyB => copy_image(session, out, b),
        BlendPlan::Extrapolate(w2) => {
            copy_image(session, out, a)?;
            lerp_enqueue(session, out, b, w2)
        }
        BlendPlan::Mix => mix_enqueue(session, out, a, b, w),
    }
}

/// `blend::blend` on the device.
pub fn blend_gpu(session: &Session, out: &mut Image, a: &Image, b: &Image, w: f32) -> Result<(), StageError> {
    check_host("blend", out, a)?;
    check_host("blend", out, b)?;
    let da = DeviceImage::upload(session, a)?;
    let db = DeviceImage::upload(session, b)?;
    let o = DeviceImage::zeroed(session, a.width, a.height, a.fmt())?;
    blend_enqueue(session, &o, &da, &db, w)?;
    *out = download_checked(session, &o)?;
    Ok(())
}

/// `blend::weighted_blend_noflow`, its proven fast path: `blend(a + 0.0, b + 0.0, 1 - wa)` for
/// float images whose samples are all finite with `|v| <= 1e30` (else `bl_unsound_f32` raises
/// bit 1 of `flag` and the CPU runs the full sequence), `blend(a, b, 1 - wa)` for 8 and 16 bit.
pub fn weighted_blend_noflow_enqueue(
    session: &Session,
    out: &DeviceImage,
    a: &DeviceImage,
    wa: f32,
    b: &DeviceImage,
    flag: &Flag,
) -> Result<(), StageError> {
    check2("weighted blend", out, a)?;
    check2("weighted blend", out, b)?;
    let t = 1.0f32 - wa;
    if out.fmt() != Fmt::F32 {
        return blend_enqueue(session, out, a, b, t);
    }
    let n = out.sample_count();
    let mut unsound = session.kernel("bl_unsound_f32")?;
    let mut plus = session.kernel("bl_plus_zero_f32")?;
    let a1 = DeviceImage::zeroed(session, a.width, a.height, Fmt::F32)?;
    let b1 = DeviceImage::zeroed(session, b.width, b.height, Fmt::F32)?;
    for (src, dst) in [(a, &a1), (b, &b1)] {
        session.queue.run(&mut unsound, Range::D1(n), &[image_arg(src), flag.buf.arg(), Arg::U32(items(n)?)])?;
        session.queue.run(&mut plus, Range::D1(n), &[image_arg(src), image_arg(dst), Arg::U32(items(n)?)])?;
    }
    blend_enqueue(session, out, &a1, &b1, t)
}

/// `blend::weighted_blend_noflow` on the device (`Unsupported` where the CPU needs its full
/// sequence with the sharpening step, which `wbstage::weighted_blend_noflow_reference_gpu` runs).
pub fn weighted_blend_noflow_gpu(session: &Session, out: &mut Image, a: &Image, wa: f32, b: &Image) -> Result<(), StageError> {
    check_host("weighted blend", out, a)?;
    check_host("weighted blend", out, b)?;
    let da = DeviceImage::upload(session, a)?;
    let db = DeviceImage::upload(session, b)?;
    let o = DeviceImage::zeroed(session, a.width, a.height, a.fmt())?;
    let flag = Flag::new(session)?;
    weighted_blend_noflow_enqueue(session, &o, &da, wa, &db, &flag)?;
    flag.check(session, "weighted blend without flows: a float sample is not finite or above 1e30")?;
    *out = download_checked(session, &o)?;
    Ok(())
}

