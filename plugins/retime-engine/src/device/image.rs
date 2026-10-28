//! Engine images on the device: the shared plumbing of every stage wrapper.
//!
//! A [`DeviceImage`] holds the samples of an [`Image`] (interleaved A, R, G, B in the engine's
//! sample type) in one buffer. Uploading and downloading copy the samples unchanged, so a round
//! trip returns the same bytes, NaN payloads and negative zeros included.

use super::cl::{Access, Buffer, ClError};
use super::selfcheck::Session;
use crate::image::{Fmt, Image, Samples};

/// Why a device stage did not produce a result. The executor reruns the stage on the CPU for
/// every variant except a failure it chooses to report.
#[derive(Debug)]
pub enum StageError {
    /// The driver or a kernel build reported an error.
    Cl(ClError),
    /// A float output holds a NaN. The self-check found that the devices write other NaN payloads
    /// than the CPU, so such an output is never accepted; the CPU computes the stage instead.
    NanOutput,
    /// The stage cannot run on the device for this input (an empty or oversized image, a setting
    /// the kernel does not cover); the text says why.
    Unsupported(String),
}

impl From<ClError> for StageError {
    fn from(e: ClError) -> StageError {
        StageError::Cl(e)
    }
}

impl std::fmt::Display for StageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StageError::Cl(e) => write!(f, "device error: {e}"),
            StageError::NanOutput => write!(f, "a float output holds a NaN"),
            StageError::Unsupported(why) => write!(f, "not supported on the device: {why}"),
        }
    }
}

impl std::error::Error for StageError {}

/// The samples of a device image, by format.
pub enum DeviceSamples {
    U8(Buffer<u8>),
    U16(Buffer<u16>),
    F32(Buffer<f32>),
}

/// An image whose samples live in device memory.
pub struct DeviceImage {
    pub width: usize,
    pub height: usize,
    pub samples: DeviceSamples,
}

impl DeviceImage {
    /// Copies `img` to the device.
    pub fn upload(session: &Session, img: &Image) -> Result<DeviceImage, StageError> {
        if img.sample_count() == 0 {
            return Err(StageError::Unsupported("an image without pixels".into()));
        }
        let samples = match &img.samples {
            Samples::U8(v) => DeviceSamples::U8(Buffer::from_slice(&session.context, v, Access::ReadWrite)?),
            Samples::U16(v) => DeviceSamples::U16(Buffer::from_slice(&session.context, v, Access::ReadWrite)?),
            Samples::F32(v) => DeviceSamples::F32(Buffer::from_slice(&session.context, v, Access::ReadWrite)?),
        };
        Ok(DeviceImage { width: img.width, height: img.height, samples })
    }

    /// A cleared image of the given size and format.
    pub fn zeroed(session: &Session, width: usize, height: usize, fmt: Fmt) -> Result<DeviceImage, StageError> {
        let n = width * height * crate::image::CHANNELS;
        if n == 0 {
            return Err(StageError::Unsupported("an image without pixels".into()));
        }
        let samples = match fmt {
            Fmt::U8 => DeviceSamples::U8(Buffer::new(&session.context, n, Access::ReadWrite)?),
            Fmt::U16 => DeviceSamples::U16(Buffer::new(&session.context, n, Access::ReadWrite)?),
            Fmt::F32 => DeviceSamples::F32(Buffer::new(&session.context, n, Access::ReadWrite)?),
        };
        let img = DeviceImage { width, height, samples };
        match &img.samples {
            DeviceSamples::U8(b) => session.queue.fill(b, 0u8)?,
            DeviceSamples::U16(b) => session.queue.fill(b, 0u16)?,
            DeviceSamples::F32(b) => session.queue.fill(b, 0.0f32)?,
        }
        Ok(img)
    }

    pub fn fmt(&self) -> Fmt {
        match self.samples {
            DeviceSamples::U8(_) => Fmt::U8,
            DeviceSamples::U16(_) => Fmt::U16,
            DeviceSamples::F32(_) => Fmt::F32,
        }
    }

    /// Number of samples (width x height x 4).
    pub fn sample_count(&self) -> usize {
        self.width * self.height * crate::image::CHANNELS
    }

    /// Copies the image back. A float image is returned as is, NaNs included; the stage wrappers
    /// reject a NaN output through [`has_nan`] before they hand a result on.
    pub fn download(&self, session: &Session) -> Result<Image, StageError> {
        let samples = match &self.samples {
            DeviceSamples::U8(b) => Samples::U8(session.queue.read(b)?),
            DeviceSamples::U16(b) => Samples::U16(session.queue.read(b)?),
            DeviceSamples::F32(b) => Samples::F32(session.queue.read(b)?),
        };
        Ok(Image { width: self.width, height: self.height, samples })
    }
}

/// True when any float sample is a NaN (integer images never are).
pub fn has_nan(img: &Image) -> bool {
    match &img.samples {
        Samples::F32(v) => v.iter().any(|x| x.is_nan()),
        _ => false,
    }
}

/// `Err(StageError::NanOutput)` when `plane` holds a NaN.
pub fn reject_nan(plane: &[f32]) -> Result<(), StageError> {
    if plane.iter().any(|x| x.is_nan()) {
        Err(StageError::NanOutput)
    } else {
        Ok(())
    }
}
