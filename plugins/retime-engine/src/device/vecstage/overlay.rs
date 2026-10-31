//! The Debug view compositions of `debug.rs` on the device (kernels `ov_*`): `W'`
//! (`debug::fill_transparent`), the Blend view (`blend(W', Y, 0.5)`), the grey Diff view
//! (`debug::diff`), the red/cyan overlay of Detail and Edge Compare (`debug::overlay`), the luma of
//! the Debug views (`debug::debug_luma`) and its detail and edge measures (`debug::detail`,
//! `debug::edge`). The forward warp that makes `W` is another stage: these take its output as an
//! input. In the order of `debug::debug_view_on`: fill `W` from `Y`, then the view (Detail and
//! Edge: the luma of `W'` and of `Y`, the measure of each, the overlay).

use super::{check_pixels, count, download_floats, download_image, image_output, launch, samples_arg, same_size, suffix, upload_image, DevicePlane};
use crate::device::cl::{Arg, Range};
use crate::device::image::{DeviceImage, StageError};
use crate::device::selfcheck::Session;
use crate::image::{Fmt, Image, CHANNELS};
use crate::prep::Plane;

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

/// Queues `ov_fill_*`: the pixels of `w` whose alpha is 0 take the pixel of `y` (`W'`), in place.
pub fn fill_transparent_enqueue(session: &Session, w: &DeviceImage, y: &DeviceImage) -> Result<(), StageError> {
    let n = check_pixels(w.width, w.height)?;
    same_size((w.width, w.height), (y.width, y.height), "fill")?;
    if w.fmt() != y.fmt() {
        return Err(StageError::Unsupported("fill: images of two formats".into()));
    }
    let args = [samples_arg(&w.samples), samples_arg(&y.samples), count(n)];
    launch(session, &format!("ov_fill_{}", suffix(w.fmt())), Range::D1(n), &args)
}

/// `debug::fill_transparent` on the device: `w` becomes `W'`. Frames of two shapes are
/// `Unsupported` (the CPU function requires one shape). On an error `w` is unchanged.
pub fn fill_transparent_gpu(session: &Session, w: &mut Image, y: &Image) -> Result<(), StageError> {
    if !w.same_shape(y) {
        return Err(StageError::Unsupported("fill: incompatible frames".into()));
    }
    let (dw, dy) = (upload_image(session, w)?, upload_image(session, y)?);
    fill_transparent_enqueue(session, &dw, &dy)?;
    *w = download_image(session, &dw)?;
    Ok(())
}

/// Queues `ov_blend_*`: `out` receives the Blend view `blend(warped, y, 0.5)` (`warped` is `W'`).
pub fn blend_view_enqueue(session: &Session, warped: &DeviceImage, y: &DeviceImage, out: &DeviceImage) -> Result<(), StageError> {
    pair(session, "ov_blend", warped, y, out, true)
}

/// The Blend view of `debug::debug_view` (`blend::blend(out, warped, y, 0.5)` of `W'` and `Y`) on
/// the device.
pub fn blend_view_gpu(session: &Session, warped: &Image, y: &Image) -> Result<Image, StageError> {
    pair_gpu(session, "ov_blend", warped, y, true)
}

/// Queues `ov_diff_*`: `out` receives the Diff view of `warped` (`W'`) and `y`.
pub fn diff_view_enqueue(session: &Session, warped: &DeviceImage, y: &DeviceImage, out: &DeviceImage) -> Result<(), StageError> {
    pair(session, "ov_diff", warped, y, out, false)
}

/// The Diff view of `debug::debug_view` (`debug::diff`: grey, alpha white) on the device.
pub fn diff_view_gpu(session: &Session, warped: &Image, y: &Image) -> Result<Image, StageError> {
    pair_gpu(session, "ov_diff", warped, y, false)
}

/// Queues `ov_overlay_*`: `out` receives the red/cyan overlay of the planes `r` and `g`.
pub fn overlay_enqueue(session: &Session, r: &DevicePlane, g: &DevicePlane, out: &DeviceImage) -> Result<(), StageError> {
    let n = check_pixels(out.width, out.height)?;
    same_size((out.width, out.height), (r.width, r.height), "overlay")?;
    same_size((out.width, out.height), (g.width, g.height), "overlay")?;
    let args = [r.data.arg(), g.data.arg(), samples_arg(&out.samples), count(n)];
    launch(session, &format!("ov_overlay_{}", suffix(out.fmt())), Range::D1(n), &args)
}

/// The red/cyan overlay of the Detail and Edge Compare views (`debug::overlay`) into a new image
/// of `width x height` in `fmt`.
pub fn overlay_gpu(session: &Session, width: usize, height: usize, fmt: Fmt, r: &[f32], g: &[f32]) -> Result<Image, StageError> {
    let dr = DevicePlane::upload(session, width, height, r)?;
    let dg = DevicePlane::upload(session, width, height, g)?;
    let out = image_output(session, width, height, fmt)?;
    overlay_enqueue(session, &dr, &dg, &out)?;
    download_image(session, &out)
}

/// Queues `ov_luma_*`: `out` receives the luma of the Debug views of `img`.
pub fn debug_luma_enqueue(session: &Session, img: &DeviceImage, out: &DevicePlane) -> Result<(), StageError> {
    let n = check_pixels(img.width, img.height)?;
    same_size((img.width, img.height), (out.width, out.height), "debug luma")?;
    let args = [samples_arg(&img.samples), out.data.arg(), count(n)];
    launch(session, &format!("ov_luma_{}", suffix(img.fmt())), Range::D1(n), &args)
}

/// `debug::debug_luma` on the device.
pub fn debug_luma_gpu(session: &Session, img: &Image) -> Result<Plane, StageError> {
    let d = upload_image(session, img)?;
    let out = DevicePlane::output(session, img.width, img.height)?;
    debug_luma_enqueue(session, &d, &out)?;
    Ok(Plane { width: img.width, height: img.height, data: download_floats(session, &out.data)? })
}

/// Queues `ov_detail`: `out` receives the detail measure of the luma plane `l`.
pub fn detail_enqueue(session: &Session, l: &DevicePlane, out: &DevicePlane) -> Result<(), StageError> {
    check_pixels(l.width, l.height)?;
    same_size((l.width, l.height), (out.width, out.height), "detail")?;
    let args = [l.data.arg(), out.data.arg(), Arg::U32(l.width as u32), Arg::U32(l.height as u32)];
    launch(session, "ov_detail", Range::D2(l.width, l.height), &args)
}

/// `debug::detail` of the luma plane `l` on the device.
pub fn detail_gpu(session: &Session, l: &Plane) -> Result<Vec<f32>, StageError> {
    let dl = DevicePlane::upload(session, l.width, l.height, &l.data)?;
    let out = DevicePlane::output(session, l.width, l.height)?;
    detail_enqueue(session, &dl, &out)?;
    download_floats(session, &out.data)
}

/// Queues `ov_edge`: `out` receives the edge measure of the luma plane `l`.
pub fn edge_enqueue(session: &Session, l: &DevicePlane, out: &DevicePlane) -> Result<(), StageError> {
    check_pixels(l.width, l.height)?;
    same_size((l.width, l.height), (out.width, out.height), "edge")?;
    let args = [l.data.arg(), out.data.arg(), Arg::U32(l.width as u32), Arg::U32(l.height as u32)];
    launch(session, "ov_edge", Range::D2(l.width, l.height), &args)
}

/// `debug::edge` of the luma plane `l` on the device.
pub fn edge_gpu(session: &Session, l: &Plane) -> Result<Vec<f32>, StageError> {
    let dl = DevicePlane::upload(session, l.width, l.height, &l.data)?;
    let out = DevicePlane::output(session, l.width, l.height)?;
    edge_enqueue(session, &dl, &out)?;
    download_floats(session, &out.data)
}
