//! The Debug view compositions of `debug.rs` on the device (kernels `ov_*`): the Blend view
//! (`blend(W, Y, 0.5)`), the Diff view, the R/G overlay of Detail and Edge Compare, and the per
//! pixel tails of the detail (`clamp01((L - blur5(L)) / 255 + 0.5)`) and the edge measure (the
//! gradient of the blurred luma, `m <= 20 ? m * 0.05 : 1`). The forward warp, the luma blurs and
//! the Gaussian are other stages: these take their outputs as inputs.

use super::{check_pixels, count, download_floats, download_image, image_output, launch, samples_arg, same_size, suffix, upload_image, DevicePlane};
use crate::device::cl::{Arg, Range};
use crate::device::image::{DeviceImage, StageError};
use crate::device::selfcheck::Session;
use crate::image::{Fmt, Image, CHANNELS};

fn pair(session: &Session, name: &str, a: &DeviceImage, b: &DeviceImage, out: &DeviceImage, per_sample: bool) -> Result<(), StageError> {
    let px = check_pixels(a.width, a.height)?;
    same_size((a.width, a.height), (b.width, b.height), name)?;
    same_size((a.width, a.height), (out.width, out.height), name)?;
    if a.fmt() != b.fmt() || a.fmt() != out.fmt() {
        return Err(StageError::Unsupported(format!("{name}: images of two formats")));
    }
    let n = if per_sample { px * CHANNELS } else { px };
    let args = [samples_arg(&a.samples), samples_arg(&b.samples), samples_arg(&out.samples), count(n)];
    launch(session, &format!("{name}_{}", suffix(a.fmt())), Range::D1(n), &args)
}

fn pair_gpu(session: &Session, name: &str, a: &Image, b: &Image, per_sample: bool) -> Result<Image, StageError> {
    if !a.same_shape(b) {
        return Err(StageError::Unsupported(format!("{name}: incompatible frames")));
    }
    let (da, db) = (upload_image(session, a)?, upload_image(session, b)?);
    let out = image_output(session, a.width, a.height, a.fmt())?;
    pair(session, name, &da, &db, &out, per_sample)?;
    download_image(session, &out)
}

/// Queues `ov_blend_*`: `out` receives the Blend view `blend(warped, y, 0.5)`.
pub fn blend_view_enqueue(session: &Session, warped: &DeviceImage, y: &DeviceImage, out: &DeviceImage) -> Result<(), StageError> {
    pair(session, "ov_blend", warped, y, out, true)
}

/// The Blend view of `debug::debug_view` (`blend::blend(out, warped, y, 0.5)`) on the device.
pub fn blend_view_gpu(session: &Session, warped: &Image, y: &Image) -> Result<Image, StageError> {
    pair_gpu(session, "ov_blend", warped, y, true)
}

/// Queues `ov_diff_*`: `out` receives the Diff view of `warped` and `y`.
pub fn diff_view_enqueue(session: &Session, warped: &DeviceImage, y: &DeviceImage, out: &DeviceImage) -> Result<(), StageError> {
    pair(session, "ov_diff", warped, y, out, false)
}

/// The Diff view of `debug::debug_view` (`|W - Y|` per colour, alpha white) on the device.
pub fn diff_view_gpu(session: &Session, warped: &Image, y: &Image) -> Result<Image, StageError> {
    pair_gpu(session, "ov_diff", warped, y, false)
}

/// Queues `ov_overlay_*`: `out` receives the R/G overlay of the planes `r` and `g`.
pub fn overlay_enqueue(session: &Session, r: &DevicePlane, g: &DevicePlane, out: &DeviceImage) -> Result<(), StageError> {
    let n = check_pixels(out.width, out.height)?;
    same_size((out.width, out.height), (r.width, r.height), "overlay")?;
    same_size((out.width, out.height), (g.width, g.height), "overlay")?;
    let args = [r.data.arg(), g.data.arg(), samples_arg(&out.samples), count(n)];
    launch(session, &format!("ov_overlay_{}", suffix(out.fmt())), Range::D1(n), &args)
}

/// The R/G overlay of the Detail and Edge Compare views (`debug::overlay`) into a new image of
/// `width x height` in `fmt`.
pub fn overlay_gpu(session: &Session, width: usize, height: usize, fmt: Fmt, r: &[f32], g: &[f32]) -> Result<Image, StageError> {
    let dr = DevicePlane::upload(session, width, height, r)?;
    let dg = DevicePlane::upload(session, width, height, g)?;
    let out = image_output(session, width, height, fmt)?;
    overlay_enqueue(session, &dr, &dg, &out)?;
    download_image(session, &out)
}

/// Queues `ov_detail`: `out` receives the detail of the luma `l` from its blur `b`.
pub fn detail_tail_enqueue(session: &Session, l: &DevicePlane, b: &DevicePlane, out: &DevicePlane) -> Result<(), StageError> {
    let n = check_pixels(l.width, l.height)?;
    same_size((l.width, l.height), (b.width, b.height), "detail")?;
    same_size((l.width, l.height), (out.width, out.height), "detail")?;
    launch(session, "ov_detail", Range::D1(n), &[l.data.arg(), b.data.arg(), out.data.arg(), count(n)])
}

/// The per pixel tail of `debug::detail`: `clamp01(((l - b) / 255) + 0.5)` of the luma `l` and its
/// pyramid blur `b` (`prep::blur5`), both `width x height`.
pub fn detail_tail_gpu(session: &Session, width: usize, height: usize, l: &[f32], b: &[f32]) -> Result<Vec<f32>, StageError> {
    let dl = DevicePlane::upload(session, width, height, l)?;
    let db = DevicePlane::upload(session, width, height, b)?;
    let out = DevicePlane::output(session, width, height)?;
    detail_tail_enqueue(session, &dl, &db, &out)?;
    download_floats(session, &out.data)
}

/// Queues `ov_edge`: `out` receives the edge measure of the blurred luma `b`.
pub fn edge_tail_enqueue(session: &Session, b: &DevicePlane, out: &DevicePlane) -> Result<(), StageError> {
    check_pixels(b.width, b.height)?;
    same_size((b.width, b.height), (out.width, out.height), "edge")?;
    let args = [b.data.arg(), out.data.arg(), Arg::U32(b.width as u32), Arg::U32(b.height as u32)];
    launch(session, "ov_edge", Range::D2(b.width, b.height), &args)
}

/// The tail of `debug::edge` after its Gaussian of radius 2.0: the gradient (`flow::calc_grad`)
/// of the blurred luma `b` and `m <= 20 ? m * 0.05 : 1` per pixel.
pub fn edge_tail_gpu(session: &Session, width: usize, height: usize, b: &[f32]) -> Result<Vec<f32>, StageError> {
    let db = DevicePlane::upload(session, width, height, b)?;
    let out = DevicePlane::output(session, width, height)?;
    edge_tail_enqueue(session, &db, &out)?;
    download_floats(session, &out.data)
}
