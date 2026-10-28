//! Device stages of `smartblend.rs`: the composites [`smart_blend_gpu`] and
//! [`smart_blend_binary_gpu`] (`sb_smart_*`), and the coverage steps of the compositions:
//! [`coverage_holes_gpu`] (the combine and invert of `smart_finish`, `sb_cov_max_invert`) and
//! [`coverage_invert_gpu`] (`Coverage::invert` of `nearest_smart`, `sb_cov_invert`).
//!
//! The compositions themselves (`nearest_smart`, `blend_smart`, `weighted_blend_smart`,
//! `blend_tail`) are chains the executor builds: the warps with coverage
//! (`warp::inv_warp_enqueue`), the composite here, the coverage step here and the hole fill of
//! its own stage. `weighted_tail` and `weighted_blend_smart`, which need `filters::sharpen_step`,
//! are the stages of `device::wbstage`.

use super::{download_checked, image_arg, items, same_shape, suffix, DeviceCoverage};
use crate::device::cl::{Arg, Range};
use crate::device::image::{DeviceImage, StageError};
use crate::device::selfcheck::Session;
use crate::image::Image;
use crate::warp::Coverage;

fn composite_enqueue(session: &Session, out: &DeviceImage, p1: &DeviceImage, p2: &DeviceImage, w: f32, binary: bool) -> Result<(), StageError> {
    if !(same_shape(out, p1) && same_shape(out, p2)) {
        return Err(StageError::Unsupported("smart blend: images of different shapes".into()));
    }
    let pixels = out.width * out.height;
    let mut kernel = session.kernel(&format!("sb_smart_{}", suffix(out.fmt())))?;
    let args = [image_arg(p1), image_arg(p2), image_arg(out), Arg::F32(w), Arg::I32(binary as i32), Arg::U32(items(pixels)?)];
    session.queue.run(&mut kernel, Range::D1(pixels), &args)?;
    Ok(())
}

/// `smartblend::smart_blend`: queues the composite. `out` must not be `p1` or `p2`.
pub fn smart_blend_enqueue(session: &Session, out: &DeviceImage, p1: &DeviceImage, p2: &DeviceImage, w: f32) -> Result<(), StageError> {
    composite_enqueue(session, out, p1, p2, w, false)
}

/// `smartblend::smart_blend_binary`: queues the composite.
pub fn smart_blend_binary_enqueue(session: &Session, out: &DeviceImage, p1: &DeviceImage, p2: &DeviceImage, w: f32) -> Result<(), StageError> {
    composite_enqueue(session, out, p1, p2, w, true)
}

fn composite_gpu(session: &Session, out: &mut Image, p1: &Image, p2: &Image, w: f32, binary: bool) -> Result<(), StageError> {
    if !(out.same_shape(p1) && out.same_shape(p2)) || p1.sample_count() == 0 {
        return Err(StageError::Unsupported("smart blend: images of different shapes or without pixels".into()));
    }
    let a = DeviceImage::upload(session, p1)?;
    let b = DeviceImage::upload(session, p2)?;
    let o = DeviceImage::zeroed(session, p1.width, p1.height, p1.fmt())?;
    composite_enqueue(session, &o, &a, &b, w, binary)?;
    *out = download_checked(session, &o)?;
    Ok(())
}

/// `smartblend::smart_blend` on the device.
pub fn smart_blend_gpu(session: &Session, out: &mut Image, p1: &Image, p2: &Image, w: f32) -> Result<(), StageError> {
    composite_gpu(session, out, p1, p2, w, false)
}

/// `smartblend::smart_blend_binary` on the device.
pub fn smart_blend_binary_gpu(session: &Session, out: &mut Image, p1: &Image, p2: &Image, w: f32) -> Result<(), StageError> {
    composite_gpu(session, out, p1, p2, w, true)
}

/// `cov1.combine_max(cov2); cov1.invert()`: the holes mask of `smart_finish`, in place on `cov1`.
pub fn coverage_holes_enqueue(session: &Session, cov1: &DeviceCoverage, cov2: &DeviceCoverage) -> Result<(), StageError> {
    if cov1.width != cov2.width || cov1.height != cov2.height {
        return Err(StageError::Unsupported("coverage images of different sizes".into()));
    }
    let n = cov1.width * cov1.height;
    let mut kernel = session.kernel("sb_cov_max_invert")?;
    session.queue.run(&mut kernel, Range::D1(n), &[cov1.data.arg(), cov2.data.arg(), Arg::U32(items(n)?)])?;
    Ok(())
}

/// The holes mask of `smart_finish` on the device.
pub fn coverage_holes_gpu(session: &Session, cov1: &mut Coverage, cov2: &Coverage) -> Result<(), StageError> {
    let a = DeviceCoverage::upload(session, cov1)?;
    let b = DeviceCoverage::upload(session, cov2)?;
    coverage_holes_enqueue(session, &a, &b)?;
    *cov1 = a.download(session)?;
    Ok(())
}

/// `Coverage::invert`, in place.
pub fn coverage_invert_enqueue(session: &Session, cov: &DeviceCoverage) -> Result<(), StageError> {
    let n = cov.width * cov.height;
    let mut kernel = session.kernel("sb_cov_invert")?;
    session.queue.run(&mut kernel, Range::D1(n), &[cov.data.arg(), Arg::U32(items(n)?)])?;
    Ok(())
}

/// `Coverage::invert` on the device.
pub fn coverage_invert_gpu(session: &Session, cov: &mut Coverage) -> Result<(), StageError> {
    let a = DeviceCoverage::upload(session, cov)?;
    coverage_invert_enqueue(session, &a)?;
    *cov = a.download(session)?;
    Ok(())
}
