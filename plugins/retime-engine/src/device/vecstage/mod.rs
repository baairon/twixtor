//! Device stages for vector coding, the pointwise matte operations, the Debug view compositions,
//! Draw Geom, the tracking point splat and the per pixel prep: each one a kernel of
//! `kernels::vecstage` with a line for line transcription ([`items`]) and two wrappers per CPU
//! function `foo`:
//!
//! * `foo_enqueue(session, <device arguments>, out)` only queues kernels on data already on the
//!   device (no image transfers; a few launch constants travel in small buffers), so the executor
//!   can chain stages;
//! * `foo_gpu(session, <the CPU function's arguments>)` uploads, queues, downloads and returns the
//!   CPU function's result type with a [`StageError`] in place of its error.
//!
//! Identifier prefixes of the kernels and their helpers: `vc_` vector coding, `mt_` matte, `ov_`
//! overlays, `gm_` geometry and points, `pp_` prep and filters. [`program_parts`] gives the
//! sources in the order they build on each other; a program for these stages is
//! `kernels::program(&program_parts())`.
//!
//! Every stage gives the CPU function's bytes (V: the CPU code is the reference; the kernels keep
//! its operation order and precision, see `kernels::vecstage`). Rules shared by every wrapper:
//! * a float output holding a NaN is never handed on: the wrapper returns
//!   [`StageError::NanOutput`] and the CPU computes the stage (the devices write other NaN
//!   payloads than the CPU); integer outputs are unaffected;
//! * an input the CPU function refuses or panics on (mismatched sizes, a flow below 2x2, a group
//!   outside the layers), an empty image and anything above a documented maximum ([`MAX_LAYERS`],
//!   [`MAX_FG`], [`MAX_POINTS`], sides of at most `i32::MAX`, at most `cl::MAX_WORK_ITEMS` work
//!   items) is [`StageError::Unsupported`], checked before anything is queued; the CPU then gives
//!   its own result or error;
//! * a sum over layers or points runs in its fixed order inside one work item, the loop count a
//!   small argument checked against its maximum, never a parallel reduction;
//! * the wrappers record the parity gaps the CPU function records, under the same conditions.
//!
//! Not ported here: the matte choke (`matte::choke`, the `choke_gpu` stage of
//! `device::convstage`; so `matte::effective` with a shrink stays with the CPU), the blurs, resamples, the forward warp and the hole
//! fill (other stage modules), Delinearize (`prep::delinearize`, a double precision `pow` whose bits
//! come from the host C runtime). The stages that need one of them take its output as an input.

use crate::device::cl::{Access, Arg, Buffer, Range, MAX_WORK_ITEMS};
use crate::device::image::{DeviceImage, DeviceSamples, StageError};
use crate::device::kernels::vecstage;
use crate::device::selfcheck::Session;
use crate::image::{Fmt, Image, CHANNELS};

pub use crate::device::kernels::vecstage::{MAX_FG, MAX_LAYERS, MAX_POINTS};

pub mod geom;
pub mod items;
pub mod matte;
pub mod overlay;
pub mod prep;
pub mod vectors;

#[cfg(test)]
mod device_tests;
#[cfg(test)]
mod tests;

/// The kernel sources of these stages in program order (the prelude goes first, through
/// `kernels::program`).
pub fn program_parts() -> Vec<&'static str> {
    vecstage::PARTS.to_vec()
}

/// A one channel f32 plane on the device (a matte, a weight, a luma plane), row major.
pub struct DevicePlane {
    pub width: usize,
    pub height: usize,
    pub data: Buffer<f32>,
}

impl DevicePlane {
    /// Copies `data` (`width * height` values) to the device.
    pub fn upload(session: &Session, width: usize, height: usize, data: &[f32]) -> Result<DevicePlane, StageError> {
        let n = check_pixels(width, height)?;
        if data.len() != n {
            return Err(StageError::Unsupported(format!("a plane of {width}x{height} holding {} values", data.len())));
        }
        Ok(DevicePlane { width, height, data: Buffer::from_slice(&session.context, data, Access::ReadWrite)? })
    }

    /// A plane whose values the next kernel writes.
    pub fn output(session: &Session, width: usize, height: usize) -> Result<DevicePlane, StageError> {
        let n = check_pixels(width, height)?;
        Ok(DevicePlane { width, height, data: Buffer::new(&session.context, n, Access::ReadWrite)? })
    }

    /// Copies the values back, NaNs included.
    pub fn download(&self, session: &Session) -> Result<Vec<f32>, StageError> {
        Ok(session.queue.read(&self.data)?)
    }
}

/// A two channel flow on the device (`(dx, dy)` interleaved), row major.
pub struct DeviceFlow {
    pub width: usize,
    pub height: usize,
    pub data: Buffer<f32>,
}

impl DeviceFlow {
    /// Copies `flow` to the device.
    pub fn upload(session: &Session, flow: &crate::flow::FlowField) -> Result<DeviceFlow, StageError> {
        let n = check_pixels(flow.width, flow.height)?;
        if flow.data.len() != n * 2 {
            return Err(StageError::Unsupported(format!(
                "a flow of {}x{} holding {} values",
                flow.width,
                flow.height,
                flow.data.len()
            )));
        }
        Ok(DeviceFlow { width: flow.width, height: flow.height, data: Buffer::from_slice(&session.context, &flow.data, Access::ReadWrite)? })
    }

    /// A flow whose values the next kernel writes.
    pub fn output(session: &Session, width: usize, height: usize) -> Result<DeviceFlow, StageError> {
        let n = check_pixels(width, height)?;
        Ok(DeviceFlow { width, height, data: Buffer::new(&session.context, n * 2, Access::ReadWrite)? })
    }

    /// Copies the flow back, NaNs included.
    pub fn download(&self, session: &Session) -> Result<crate::flow::FlowField, StageError> {
        Ok(crate::flow::FlowField { width: self.width, height: self.height, data: session.queue.read(&self.data)? })
    }
}

/// `width * height` when both sides are at least 1, at most `i32::MAX`, and four samples per
/// pixel stay within one launch; else `Unsupported`.
pub(crate) fn check_pixels(width: usize, height: usize) -> Result<usize, StageError> {
    if width == 0 || height == 0 {
        return Err(StageError::Unsupported("an image without pixels".into()));
    }
    let side = i32::MAX as usize;
    let n = width.checked_mul(height).filter(|n| width <= side && height <= side && n.checked_mul(CHANNELS).is_some_and(|s| s <= MAX_WORK_ITEMS));
    n.ok_or_else(|| StageError::Unsupported(format!("an image of {width}x{height} is above the launch limit")))
}

/// An uninitialised device image of `fmt` whose samples the next kernel writes.
pub(crate) fn image_output(session: &Session, width: usize, height: usize, fmt: Fmt) -> Result<DeviceImage, StageError> {
    let n = check_pixels(width, height)? * CHANNELS;
    let c = &session.context;
    let samples = match fmt {
        Fmt::U8 => DeviceSamples::U8(Buffer::new(c, n, Access::ReadWrite)?),
        Fmt::U16 => DeviceSamples::U16(Buffer::new(c, n, Access::ReadWrite)?),
        Fmt::F32 => DeviceSamples::F32(Buffer::new(c, n, Access::ReadWrite)?),
    };
    Ok(DeviceImage { width, height, samples })
}

/// [`DeviceImage::upload`] after the size checks of [`check_pixels`].
pub(crate) fn upload_image(session: &Session, img: &Image) -> Result<DeviceImage, StageError> {
    check_pixels(img.width, img.height)?;
    DeviceImage::upload(session, img)
}

#[cfg(test)]
thread_local! {
    /// Tests only: the downloads keep NaNs, so a device output can be compared in full.
    pub(crate) static KEEP_NAN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether a NaN output is refused (always outside the tests).
pub(crate) fn refuse_nan() -> bool {
    #[cfg(test)]
    {
        !KEEP_NAN.with(|k| k.get())
    }
    #[cfg(not(test))]
    {
        true
    }
}

/// Downloads `img`; a float image holding a NaN is `NanOutput`.
pub(crate) fn download_image(session: &Session, img: &DeviceImage) -> Result<Image, StageError> {
    let out = img.download(session)?;
    if refuse_nan() && crate::device::image::has_nan(&out) {
        return Err(StageError::NanOutput);
    }
    Ok(out)
}

/// Downloads a float buffer; a NaN is `NanOutput`.
pub(crate) fn download_floats(session: &Session, buf: &Buffer<f32>) -> Result<Vec<f32>, StageError> {
    let v = session.queue.read(buf)?;
    if refuse_nan() {
        crate::device::image::reject_nan(&v)?;
    }
    Ok(v)
}

/// The kernel argument of a sample buffer.
pub(crate) fn samples_arg(s: &DeviceSamples) -> Arg<'_> {
    match s {
        DeviceSamples::U8(b) => b.arg(),
        DeviceSamples::U16(b) => b.arg(),
        DeviceSamples::F32(b) => b.arg(),
    }
}

/// The kernel name suffix of a format.
pub(crate) fn suffix(fmt: Fmt) -> &'static str {
    match fmt {
        Fmt::U8 => "u8",
        Fmt::U16 => "u16",
        Fmt::F32 => "f32",
    }
}

/// Queues the kernel `name` over `range` with `args`.
pub(crate) fn launch(session: &Session, name: &str, range: Range, args: &[Arg<'_>]) -> Result<(), StageError> {
    let mut k = session.kernel(name)?;
    session.queue.run(&mut k, range, args)?;
    Ok(())
}

/// `n` as the `uint` item count argument.
pub(crate) fn count(n: usize) -> Arg<'static> {
    Arg::U32(n as u32)
}

/// `Unsupported` unless `a` and `b` have one size.
pub(crate) fn same_size(a: (usize, usize), b: (usize, usize), what: &str) -> Result<(), StageError> {
    if a != b {
        return Err(StageError::Unsupported(format!("{what}: {}x{} against {}x{}", b.0, b.1, a.0, a.1)));
    }
    Ok(())
}
