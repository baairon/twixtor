//! Device stages of the per pixel composition: the blend primitives, the inverse
//! warp, the motion blur compensation and the Smart Blend composites, on the kernels of
//! `kernels::warpstage`.
//!
//! For every ported CPU function `foo` there are two entry points:
//! * `foo_enqueue(session, <device images, flows, coverages>, ...)` only queues kernels and
//!   buffer copies on the session's in-order queue (no transfer to or from the host), so the
//!   executor can chain stages while the images stay on the device. Stages that can refuse at run
//!   time take a [`Flag`]; after the chain the executor reads it with [`Flag::check`], and a
//!   refusal means the stage is rerun on the CPU. An f32 image downloaded after a chain must be
//!   checked with `image::has_nan`: a NaN in a float output is never accepted
//!   (`StageError::NanOutput`), because the devices write other NaN payloads than the CPU.
//! * `foo_gpu(session, <the CPU function's arguments>)` uploads, enqueues, downloads, checks the
//!   flag and the NaN rule, and returns the CPU function's result or a [`StageError`]; the output
//!   arguments are written only on success.
//!
//! The host side of every stage (argument checks, the scalar branches such as the snap copies of
//! `blend`, the early copy outs of the warp, the amount clamp of the compensation) is a plain
//! function of this module tree (`*_plan`), shared by the wrappers and by the device-free tests,
//! which run the same plans with the kernel transcriptions of [`rs`] in the simulator.
//!
//! Precision contract (V): every stage gives the bytes of its CPU function. The kernels follow
//! the CPU's operation order in the same precision (`kernels::warpstage` states it per kernel).
//! Not here: `filters::sharpen_step` (the Weighted Blend sharpening: blur, difference mask and
//! its clean-up, masked composite) and the chains built on it are the stages of
//! `device::wbstage`, so `weighted_blend_noflow` runs here only on its proven fast path and
//! otherwise as `wbstage::weighted_blend_noflow_reference_gpu`; the hole fill that ends the Smart
//! Blend compositions belongs to the hole fill stage. Nothing here is a parity gap.

pub mod blend;
pub mod blurcomp;
pub mod rs;
pub mod smartblend;
pub mod warp;

#[cfg(test)]
mod devtests;
#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) mod corpus;

use super::cl::{Access, Arg, Buffer};
use super::image::{has_nan, DeviceImage, DeviceSamples, StageError};
use super::kernels::{self, warpstage as k};
use super::selfcheck::Session;
use crate::flow::FlowField;
use crate::image::{Fmt, Image};
use crate::warp::Coverage;

/// The kernel sources of this module in program order (the warp samplers precede the motion
/// blur that calls them). The executor concatenates them with the other stage modules' parts after
/// the prelude (`kernels::program`).
pub fn program_parts() -> Vec<&'static str> {
    vec![k::BLEND, k::WARP, k::BLURCOMP, k::SMARTBLEND]
}

/// The prelude and [`program_parts`]: a program with every kernel of this module.
pub fn program() -> String {
    kernels::program(&program_parts())
}

/// A two channel flow field on the device (`width * height * 2` floats, as `FlowField`).
pub struct DeviceFlow {
    pub width: usize,
    pub height: usize,
    pub data: Buffer<f32>,
}

impl DeviceFlow {
    pub fn upload(session: &Session, flow: &FlowField) -> Result<DeviceFlow, StageError> {
        if flow.data.is_empty() || flow.data.len() != flow.width * flow.height * 2 {
            return Err(StageError::Unsupported("a flow without vectors or of the wrong length".into()));
        }
        let data = Buffer::from_slice(&session.context, &flow.data, Access::ReadOnly)?;
        Ok(DeviceFlow { width: flow.width, height: flow.height, data })
    }

    pub fn download(&self, session: &Session) -> Result<FlowField, StageError> {
        Ok(FlowField { width: self.width, height: self.height, data: session.queue.read(&self.data)? })
    }
}

/// A one channel uint8 coverage image on the device (as `warp::Coverage`).
pub struct DeviceCoverage {
    pub width: usize,
    pub height: usize,
    pub data: Buffer<u8>,
}

impl DeviceCoverage {
    /// A coverage image with every byte `value`.
    pub fn filled(session: &Session, width: usize, height: usize, value: u8) -> Result<DeviceCoverage, StageError> {
        if width * height == 0 {
            return Err(StageError::Unsupported("a coverage image without pixels".into()));
        }
        let data = Buffer::new(&session.context, width * height, Access::ReadWrite)?;
        session.queue.fill(&data, value)?;
        Ok(DeviceCoverage { width, height, data })
    }

    pub fn upload(session: &Session, cov: &Coverage) -> Result<DeviceCoverage, StageError> {
        if cov.data.is_empty() || cov.data.len() != cov.width * cov.height {
            return Err(StageError::Unsupported("a coverage image without pixels or of the wrong length".into()));
        }
        let data = Buffer::from_slice(&session.context, &cov.data, Access::ReadWrite)?;
        Ok(DeviceCoverage { width: cov.width, height: cov.height, data })
    }

    pub fn download(&self, session: &Session) -> Result<Coverage, StageError> {
        Ok(Coverage { width: self.width, height: self.height, data: session.queue.read(&self.data)? })
    }
}

/// The refusal word of the stages that can refuse at run time (bit 1: the CPU refuses the input
/// too; bit 2: a loop needed more passes than the device bound, `kernels::warpstage`).
pub struct Flag {
    pub buf: Buffer<i32>,
}

impl Flag {
    /// A cleared flag.
    pub fn new(session: &Session) -> Result<Flag, StageError> {
        let buf = Buffer::new(&session.context, 1, Access::ReadWrite)?;
        session.queue.fill(&buf, 0i32)?;
        Ok(Flag { buf })
    }

    /// Reads the word (a blocking read, after everything queued before it).
    pub fn read(&self, session: &Session) -> Result<i32, StageError> {
        Ok(session.queue.read(&self.buf)?[0])
    }

    /// `Ok` when no stage raised the flag, else `StageError::Unsupported` naming the reason.
    pub fn check(&self, session: &Session, what: &str) -> Result<(), StageError> {
        flag_result(self.read(session)?, what)
    }
}

/// The meaning of a flag word.
pub fn flag_result(word: i32, what: &str) -> Result<(), StageError> {
    if word == 0 {
        return Ok(());
    }
    let mut why = Vec::new();
    if word & k::FLAG_CPU_REFUSES != 0 {
        why.push("an input the CPU refuses too");
    }
    if word & k::FLAG_DEVICE_BOUND != 0 {
        why.push("a loop longer than the device bound");
    }
    Err(StageError::Unsupported(format!("{what}: {}", why.join(", "))))
}

/// The kernel name suffix of a format.
pub fn suffix(fmt: Fmt) -> &'static str {
    match fmt {
        Fmt::U8 => "u8",
        Fmt::U16 => "u16",
        Fmt::F32 => "f32",
    }
}

/// The buffer of a device image as a kernel argument.
pub fn image_arg(img: &DeviceImage) -> Arg<'_> {
    match &img.samples {
        DeviceSamples::U8(b) => b.arg(),
        DeviceSamples::U16(b) => b.arg(),
        DeviceSamples::F32(b) => b.arg(),
    }
}

/// Same size and format.
pub fn same_shape(a: &DeviceImage, b: &DeviceImage) -> bool {
    a.width == b.width && a.height == b.height && a.fmt() == b.fmt()
}

/// Queues a copy of every sample of `src` into `dst` (equal shapes).
pub fn copy_image(session: &Session, dst: &DeviceImage, src: &DeviceImage) -> Result<(), StageError> {
    if !same_shape(dst, src) {
        return Err(StageError::Unsupported("a copy between images of different shapes".into()));
    }
    match (&src.samples, &dst.samples) {
        (DeviceSamples::U8(s), DeviceSamples::U8(d)) => session.queue.copy(s, d)?,
        (DeviceSamples::U16(s), DeviceSamples::U16(d)) => session.queue.copy(s, d)?,
        (DeviceSamples::F32(s), DeviceSamples::F32(d)) => session.queue.copy(s, d)?,
        _ => unreachable!("shapes were checked"),
    }
    Ok(())
}

/// Downloads `img` and applies the NaN rule.
pub fn download_checked(session: &Session, img: &DeviceImage) -> Result<Image, StageError> {
    let out = img.download(session)?;
    if has_nan(&out) {
        return Err(StageError::NanOutput);
    }
    Ok(out)
}

/// Item count of a 1D launch, refused when it does not fit the kernel's `uint n`.
pub(crate) fn items(n: usize) -> Result<u32, StageError> {
    u32::try_from(n).map_err(|_| StageError::Unsupported(format!("{n} work items")))
}

/// Image side as the kernels' `int`.
pub(crate) fn side(n: usize) -> Result<i32, StageError> {
    i32::try_from(n).map_err(|_| StageError::Unsupported(format!("a side of {n} pixels")))
}
