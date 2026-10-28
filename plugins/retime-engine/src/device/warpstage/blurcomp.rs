//! Device stages of `warp/blurcomp.rs`: [`motion_blur_gpu`] (bilinear and nearest taps, plain
//! and 360), [`clear_rgb_where_transparent_gpu`] and the compensation chain
//! [`blur_compensation_wrap_gpu`] / [`blur_compensation_gpu`] (condition, clamp, positive blur
//! and clear, or negative blur and lerp). The host takes the condition and the clamp
//! ([`compensate_plan`]) and the prologue of the leaves ([`blur_args`]).
//!
//! The kernel runs at most `BC_TAPS` taps per direction; a pixel needing more raises bit 2 of the
//! flag and the CPU runs the stage. Launches are split into bands of rows so that no launch does
//! more than [`TAP_BUDGET`] tap evaluations even when every pixel takes the most taps.

use super::rs::BlurArgs;
use super::{download_checked, image_arg, items, same_shape, side, suffix, DeviceFlow, Flag};
use crate::device::cl::{Arg, Range};
use crate::device::image::{DeviceImage, StageError};
use crate::device::kernels::warpstage::BC_TAPS;
use crate::device::selfcheck::Session;
use crate::flow::FlowField;
use crate::image::Image;
use crate::warp::blur_amount;

/// Most tap evaluations one launch may do (two directions of `BC_TAPS` taps per pixel).
pub const TAP_BUDGET: usize = 1 << 26;

/// Rows per launch for an image `width` pixels wide.
pub fn band_rows(width: usize) -> usize {
    (TAP_BUDGET / (width.max(1) * 2 * BC_TAPS as usize)).max(1)
}

/// The prologue of the motion blur leaves (`blurcomp::run`), for images of `(width, height)` and
/// a flow of `(fw, fh)`; the checks of `motion_blur`.
#[allow(clippy::too_many_arguments)]
pub fn blur_args(
    (width, height): (usize, usize),
    (fw, fh): (usize, usize),
    flow_len: usize,
    amount: f32,
    hiq: bool,
    shutter: f32,
    is360: bool,
) -> Result<BlurArgs, StageError> {
    if width < 2 || height < 2 || fw < 2 || fh < 2 {
        return Err(StageError::Unsupported("motion blur: an image or flow below 2x2".into()));
    }
    if flow_len != fw * fh * 2 {
        return Err(StageError::Unsupported("motion blur: flow data length does not match its size".into()));
    }
    let (w, h, fwi, fhi) = (side(width)?, side(height)?, side(fw)?, side(fh)?);
    let w1i = w - 1;
    let h1i = h - 1;
    let rx = (fwi - 1) as f32 / w1i as f32;
    let ry = (fhi - 1) as f32 / h1i as f32;
    let kx = (0.5f32 / rx) * amount;
    let ky = (0.5f32 / ry) * amount;
    Ok(BlurArgs {
        w,
        h,
        fw: fwi,
        fh: fhi,
        rx,
        ry,
        kx,
        ky,
        shifted: (shutter != 0.5f32) as i32,
        k: shutter * 2.0f32 - 1.0f32,
        bilinear: hiq as i32,
        wrap: is360 as i32,
        y0: 0,
    })
}

/// `blurcomp::motion_blur`: queues the bands of `bc_motion_blur_*`. `dest` must not be `src`.
#[allow(clippy::too_many_arguments)]
pub fn motion_blur_enqueue(
    session: &Session,
    dest: &DeviceImage,
    src: &DeviceImage,
    flow: &DeviceFlow,
    amount: f32,
    hiq: bool,
    shutter: f32,
    is360: bool,
    flag: &Flag,
) -> Result<(), StageError> {
    if !same_shape(dest, src) {
        return Err(StageError::Unsupported("motion blur: images of different shapes".into()));
    }
    let a = blur_args((src.width, src.height), (flow.width, flow.height), flow.data.len(), amount, hiq, shutter, is360)?;
    if is360 {
        fx_core::gaps::record(&crate::gaps::G6);
    }
    let mut kernel = session.kernel(&format!("bc_motion_blur_{}", suffix(src.fmt())))?;
    let rows = band_rows(src.width);
    let mut y0 = 0usize;
    while y0 < src.height {
        let band = rows.min(src.height - y0);
        let args = [
            image_arg(src),
            flow.data.arg(),
            image_arg(dest),
            flag.buf.arg(),
            Arg::I32(a.w),
            Arg::I32(a.h),
            Arg::I32(a.fw),
            Arg::I32(a.fh),
            Arg::F32(a.rx),
            Arg::F32(a.ry),
            Arg::F32(a.kx),
            Arg::F32(a.ky),
            Arg::I32(a.shifted),
            Arg::F32(a.k),
            Arg::I32(a.bilinear),
            Arg::I32(a.wrap),
            Arg::I32(y0 as i32),
        ];
        session.queue.run(&mut kernel, Range::D2(src.width, band), &args)?;
        y0 += band;
    }
    Ok(())
}

/// `blurcomp::motion_blur` on the device.
#[allow(clippy::too_many_arguments)]
pub fn motion_blur_gpu(
    session: &Session,
    dest: &mut Image,
    src: &Image,
    flow: &FlowField,
    amount: f32,
    hiq: bool,
    shutter: f32,
    is360: bool,
) -> Result<(), StageError> {
    if !dest.same_shape(src) {
        return Err(StageError::Unsupported("motion blur: images of different shapes".into()));
    }
    blur_args((src.width, src.height), (flow.width, flow.height), flow.data.len(), amount, hiq, shutter, is360)?;
    let s = DeviceImage::upload(session, src)?;
    let f = DeviceFlow::upload(session, flow)?;
    let d = DeviceImage::zeroed(session, src.width, src.height, src.fmt())?;
    let flag = Flag::new(session)?;
    motion_blur_enqueue(session, &d, &s, &f, amount, hiq, shutter, is360, &flag)?;
    flag.check(session, "motion blur")?;
    *dest = download_checked(session, &d)?;
    Ok(())
}

/// `blurcomp::clear_rgb_where_transparent` from `src` into `dst` (equal shapes, distinct).
pub fn clear_rgb_enqueue(session: &Session, dst: &DeviceImage, src: &DeviceImage) -> Result<(), StageError> {
    if !same_shape(dst, src) {
        return Err(StageError::Unsupported("clear rgb: images of different shapes".into()));
    }
    let pixels = src.width * src.height;
    let mut kernel = session.kernel(&format!("bc_clear_rgb_{}", suffix(src.fmt())))?;
    session.queue.run(&mut kernel, Range::D1(pixels), &[image_arg(src), image_arg(dst), Arg::U32(items(pixels)?)])?;
    Ok(())
}

/// `blurcomp::clear_rgb_where_transparent` on the device.
pub fn clear_rgb_where_transparent_gpu(session: &Session, img: &mut Image) -> Result<(), StageError> {
    if img.sample_count() == 0 {
        return Err(StageError::Unsupported("an image without pixels".into()));
    }
    let s = DeviceImage::upload(session, img)?;
    let d = DeviceImage::zeroed(session, img.width, img.height, img.fmt())?;
    clear_rgb_enqueue(session, &d, &s)?;
    *img = download_checked(session, &d)?;
    Ok(())
}

/// What `compensate_wrap` does with an amount after `blur_amount`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CompPlan {
    /// The clamped amount is below 1e-6: nothing.
    Nothing,
    /// Motion blur with this amount and shutter 0.5, then the clear.
    Blur(f32),
    /// Motion blur with amount 0.5 and shutter 0.5, then `lerp(tmp, img, amount)`.
    Sharpen(f32),
}

/// The clamp of `compensate_wrap` (`20 < a` gives 20, else `a < -10` gives -10).
pub fn compensate_plan(amount: f32) -> CompPlan {
    let mut amount = amount;
    if 20.0f32 < amount {
        amount = 20.0;
    } else if amount < -10.0f32 {
        amount = -10.0;
    }
    if (amount.abs() as f64) < 1e-6 {
        return CompPlan::Nothing;
    }
    if 0.0f32 < amount {
        CompPlan::Blur(amount)
    } else {
        CompPlan::Sharpen(amount)
    }
}

/// `blurcomp::blur_compensation_wrap` on a device image in place.
#[allow(clippy::too_many_arguments)]
pub fn blur_compensation_wrap_enqueue(
    session: &Session,
    img: &DeviceImage,
    flow: &DeviceFlow,
    s0: f32,
    s1: f32,
    blur: f32,
    hiq: bool,
    is360: bool,
    flag: &Flag,
) -> Result<(), StageError> {
    let Some(amount) = blur_amount(s0, s1, blur) else { return Ok(()) };
    match compensate_plan(amount) {
        CompPlan::Nothing => Ok(()),
        CompPlan::Blur(a) => {
            let tmp = DeviceImage::zeroed(session, img.width, img.height, img.fmt())?;
            motion_blur_enqueue(session, &tmp, img, flow, a, hiq, 0.5, is360, flag)?;
            clear_rgb_enqueue(session, img, &tmp)
        }
        CompPlan::Sharpen(a) => {
            let tmp = DeviceImage::zeroed(session, img.width, img.height, img.fmt())?;
            motion_blur_enqueue(session, &tmp, img, flow, 0.5, hiq, 0.5, is360, flag)?;
            super::blend::lerp_enqueue(session, &tmp, img, a)?;
            super::copy_image(session, img, &tmp)
        }
    }
}

/// `blurcomp::blur_compensation_wrap` on the device.
#[allow(clippy::too_many_arguments)]
pub fn blur_compensation_wrap_gpu(
    session: &Session,
    img: &mut Image,
    flow: &FlowField,
    s0: f32,
    s1: f32,
    blur: f32,
    hiq: bool,
    is360: bool,
) -> Result<(), StageError> {
    let Some(amount) = blur_amount(s0, s1, blur) else { return Ok(()) };
    if compensate_plan(amount) == CompPlan::Nothing {
        return Ok(());
    }
    if img.sample_count() == 0 {
        return Err(StageError::Unsupported("an image without pixels".into()));
    }
    blur_args((img.width, img.height), (flow.width, flow.height), flow.data.len(), amount, hiq, 0.5, is360)?;
    let d = DeviceImage::upload(session, img)?;
    let f = DeviceFlow::upload(session, flow)?;
    let flag = Flag::new(session)?;
    blur_compensation_wrap_enqueue(session, &d, &f, s0, s1, blur, hiq, is360, &flag)?;
    flag.check(session, "blur compensation")?;
    *img = download_checked(session, &d)?;
    Ok(())
}

/// `blurcomp::blur_compensation` (the flag off) on the device.
pub fn blur_compensation_gpu(session: &Session, img: &mut Image, flow: &FlowField, s0: f32, s1: f32, blur: f32, hiq: bool) -> Result<(), StageError> {
    blur_compensation_wrap_gpu(session, img, flow, s0, s1, blur, hiq, false)
}
