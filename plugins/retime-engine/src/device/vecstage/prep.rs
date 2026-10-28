//! The per pixel prep on the device: colour to luma (`prep::luma` at 8, 16 and 32 bit with its
//! Track RGB+A and video range flags; kernels `pp_luma_*`, f32 at 8 bpc and f64 above, as the CPU)
//! and the tail of Contrast / Edge Enhance after its Gaussian (`prep::edge_enhance` and
//! `edge_enhance_with`: the gradient, plain or circular, and `(v * 255 + p) * 0.5`; `pp_edge_tail`).
//! Delinearize (`prep::delinearize`) stays on the CPU: its double precision `pow` takes its bits
//! from the host C runtime.

use super::{check_pixels, count, download_floats, launch, samples_arg, same_size, suffix, upload_image, DevicePlane};
use crate::device::cl::{Arg, Range};
use crate::device::image::{DeviceImage, StageError};
use crate::device::selfcheck::Session;
use crate::image::Image;
use crate::prep::Plane;

/// Queues `pp_luma_*`: `out` (the size of `img`) receives its luma.
pub fn luma_enqueue(session: &Session, img: &DeviceImage, track_alpha: bool, video_range: bool, out: &DevicePlane) -> Result<(), StageError> {
    if video_range && track_alpha {
        return Err(StageError::Unsupported("optical flow luma: video range conversion together with Track RGB+A is not supported".into()));
    }
    let n = check_pixels(img.width, img.height)?;
    same_size((img.width, img.height), (out.width, out.height), "luma")?;
    let args = [samples_arg(&img.samples), out.data.arg(), Arg::I32(track_alpha as i32), Arg::I32(video_range as i32), count(n)];
    launch(session, &format!("pp_luma_{}", suffix(img.fmt())), Range::D1(n), &args)
}

/// `prep::luma` on the device. The CPU's refusal of video range with Track RGB+A is
/// `Unsupported` with its text; an image without pixels gives the empty plane, as on the CPU.
pub fn luma_gpu(session: &Session, img: &Image, track_alpha: bool, video_range: bool) -> Result<Plane, StageError> {
    if video_range && track_alpha {
        return Err(StageError::Unsupported("optical flow luma: video range conversion together with Track RGB+A is not supported".into()));
    }
    if img.width == 0 || img.height == 0 {
        return Ok(Plane::new(img.width, img.height));
    }
    let d = upload_image(session, img)?;
    let out = DevicePlane::output(session, img.width, img.height)?;
    luma_enqueue(session, &d, track_alpha, video_range, &out)?;
    Ok(Plane { width: img.width, height: img.height, data: download_floats(session, &out.data)? })
}

/// Queues `pp_edge_tail`: `out` receives the Edge Enhance result of the blurred plane `b`.
pub fn edge_enhance_tail_enqueue(session: &Session, b: &DevicePlane, wrap: bool, out: &DevicePlane) -> Result<(), StageError> {
    check_pixels(b.width, b.height)?;
    same_size((b.width, b.height), (out.width, out.height), "edge enhance")?;
    let args = [b.data.arg(), out.data.arg(), Arg::U32(b.width as u32), Arg::U32(b.height as u32), Arg::I32(wrap as i32)];
    launch(session, "pp_edge_tail", Range::D2(b.width, b.height), &args)
}

/// The tail of `prep::edge_enhance_with(p, wrap)` on the device: `blurred` is `p` after the
/// Gaussian of radius 2.0 (`conv::blur_gaussian`, or `blur_gaussian_wrapx` with `wrap`); the
/// result is the plane the CPU function leaves in `p`.
pub fn edge_enhance_tail_gpu(session: &Session, blurred: &Plane, wrap: bool) -> Result<Plane, StageError> {
    let b = DevicePlane::upload(session, blurred.width, blurred.height, &blurred.data)?;
    let out = DevicePlane::output(session, blurred.width, blurred.height)?;
    edge_enhance_tail_enqueue(session, &b, wrap, &out)?;
    Ok(Plane { width: blurred.width, height: blurred.height, data: download_floats(session, &out.data)? })
}
