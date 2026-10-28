//! Device stages of `warp.rs`: [`inv_warp_gpu`] and [`inv_warp_coverage_gpu`] in every mode the
//! compositor reaches (hold, fade with or without coverage, 360 wrap; all three depths). The
//! argument checks and the early copy outs are taken on the host by [`warp_plan`], in the CPU's
//! order; the row kernel is `wp_inv_warp_*`, one work item per output pixel.

use super::rs::WarpArgs;
use super::{copy_image, download_checked, image_arg, same_shape, side, suffix, DeviceCoverage, DeviceFlow, Flag};
use crate::device::cl::{Arg, Range};
use crate::device::image::{DeviceImage, StageError};
use crate::device::selfcheck::Session;
use crate::flow::FlowField;
use crate::image::Image;
use crate::warp::{Coverage, WarpOpts};

/// What the CPU's `inv_warp_impl` does for given shapes and scalars.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WarpPlan {
    /// A copy of the source (and a coverage of 255 everywhere).
    Copy,
    /// The row kernel with these arguments (`want_cov` is set by the caller).
    Launch(WarpArgs),
}

/// The host part of `inv_warp_impl`: shapes `(width, height)` of the images (already known to
/// be equal in size and format) and of the flow, `flow_len` the flow's data length. Errors where
/// the CPU returns one (the CPU then reports its own).
pub fn warp_plan(
    (width, height): (usize, usize),
    (fw, fh): (usize, usize),
    flow_len: usize,
    amount: f32,
    opts: WarpOpts,
) -> Result<WarpPlan, StageError> {
    if width < 2 || height < 2 || fw < 2 || fh < 2 {
        return Err(StageError::Unsupported("inverse warp: an image or flow below 2x2".into()));
    }
    if flow_len != fw * fh * 2 {
        return Err(StageError::Unsupported("inverse warp: flow data length does not match its size".into()));
    }
    let a = if amount < 0.0 { -amount } else { amount };
    let mut copy_only = 0.005f64 > a as f64;
    if !copy_only {
        let r = (fw as i32 - 1) as f32 / (width as i32 - 1) as f32;
        let mut q = ((amount as f64) / (r as f64)) as f32;
        if q < 0.0 {
            q = -q;
        }
        copy_only = 0.0001f64 > q as f64;
    }
    if copy_only {
        return Ok(WarpPlan::Copy);
    }
    if opts.mode != 1 {
        return Err(StageError::Unsupported("inverse warp with mip map sampling".into()));
    }
    let (w, h, fwi, fhi) = (side(width)?, side(height)?, side(fw)?, side(fh)?);
    let w1i = w - 1;
    let h1i = h - 1;
    let rx = (fwi - 1) as f32 / w1i as f32;
    let ry = (fhi - 1) as f32 / h1i as f32;
    let kind = if opts.is360 {
        2
    } else if opts.hold {
        0
    } else {
        1
    };
    Ok(WarpPlan::Launch(WarpArgs { w, h, fw: fwi, fh: fhi, rx, ry, sx: amount / rx, sy: amount / ry, kind, want_cov: 0 }))
}

/// `warp::inv_warp` (`cov` `None`) or `warp::inv_warp_coverage` (`cov` `Some`, every byte
/// written): queues the copy or the row kernel. Raises `flag` where the CPU refuses or a loop
/// exceeds the device bound. `out` must not be `src`.
#[allow(clippy::too_many_arguments)]
pub fn inv_warp_enqueue(
    session: &Session,
    out: &DeviceImage,
    src: &DeviceImage,
    flow: &DeviceFlow,
    amount: f32,
    opts: WarpOpts,
    cov: Option<&DeviceCoverage>,
    flag: &Flag,
) -> Result<(), StageError> {
    if !same_shape(out, src) {
        return Err(StageError::Unsupported("inverse warp: images of different shapes".into()));
    }
    if let Some(c) = cov {
        if c.width != out.width || c.height != out.height {
            return Err(StageError::Unsupported("inverse warp: a coverage image of another size".into()));
        }
    }
    match warp_plan((out.width, out.height), (flow.width, flow.height), flow.data.len(), amount, opts)? {
        WarpPlan::Copy => {
            copy_image(session, out, src)?;
            if let Some(c) = cov {
                session.queue.fill(&c.data, 0xffu8)?;
            }
            Ok(())
        }
        WarpPlan::Launch(mut a) => {
            let dummy;
            let cov_buf = match cov {
                Some(c) => {
                    a.want_cov = 1;
                    &c.data
                }
                None => {
                    dummy = DeviceCoverage::filled(session, 1, 1, 0xff)?;
                    &dummy.data
                }
            };
            let mut kernel = session.kernel(&format!("wp_inv_warp_{}", suffix(out.fmt())))?;
            let args = [
                image_arg(src),
                flow.data.arg(),
                image_arg(out),
                cov_buf.arg(),
                flag.buf.arg(),
                Arg::I32(a.w),
                Arg::I32(a.h),
                Arg::I32(a.fw),
                Arg::I32(a.fh),
                Arg::F32(a.rx),
                Arg::F32(a.ry),
                Arg::F32(a.sx),
                Arg::F32(a.sy),
                Arg::I32(a.kind),
                Arg::I32(a.want_cov),
            ];
            session.queue.run(&mut kernel, Range::D2(out.width, out.height), &args)?;
            Ok(())
        }
    }
}

fn warp_gpu(session: &Session, out: &mut Image, src: &Image, flow: &FlowField, amount: f32, opts: WarpOpts, want_cov: bool) -> Result<Option<Coverage>, StageError> {
    if !out.same_shape(src) {
        return Err(StageError::Unsupported("inverse warp: images of different shapes".into()));
    }
    // the plan first, so a refusal needs no upload
    warp_plan((src.width, src.height), (flow.width, flow.height), flow.data.len(), amount, opts)?;
    let s = DeviceImage::upload(session, src)?;
    let f = DeviceFlow::upload(session, flow)?;
    let o = DeviceImage::zeroed(session, src.width, src.height, src.fmt())?;
    let c = if want_cov { Some(DeviceCoverage::filled(session, src.width, src.height, 0xff)?) } else { None };
    let flag = Flag::new(session)?;
    inv_warp_enqueue(session, &o, &s, &f, amount, opts, c.as_ref(), &flag)?;
    flag.check(session, "inverse warp")?;
    let img = download_checked(session, &o)?;
    let cov = match c {
        Some(c) => Some(c.download(session)?),
        None => None,
    };
    *out = img;
    Ok(cov)
}

/// `warp::inv_warp` on the device.
pub fn inv_warp_gpu(session: &Session, out: &mut Image, src: &Image, flow: &FlowField, amount: f32, opts: WarpOpts) -> Result<(), StageError> {
    warp_gpu(session, out, src, flow, amount, opts, false).map(|_| ())
}

/// `warp::inv_warp_coverage` on the device.
pub fn inv_warp_coverage_gpu(session: &Session, out: &mut Image, src: &Image, flow: &FlowField, amount: f32, opts: WarpOpts) -> Result<Coverage, StageError> {
    warp_gpu(session, out, src, flow, amount, opts, true).map(|c| c.expect("a coverage was requested"))
}
