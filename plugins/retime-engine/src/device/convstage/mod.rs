//! Device stages of the convolution, resampling, pyramid, hole fill and matte choke code: the
//! building blocks the device motion solver chains with its data resident on the device.
//!
//! Kernel prefixes (every kernel and device helper of these sources carries one, so the sources
//! concatenate with any other stage module's into one program): `cv_` convolution and the matte
//! choke maps (`kernels/conv.rs`), `rs_` resampling and the samplers (`kernels/resample.rs`),
//! `pm_` pyramids and the flow upsample (`kernels/pyramid.rs`), `hf_` hole fill
//! (`kernels/holefill.rs`). [`program_parts`] lists the sources in the order a program needs them
//! (`resample::SAMPLERS` holds only plain functions, `rs_bilinear1`, `rs_bilinear2`,
//! `rs_bilinear1_wrapx`, `rs_cell` and `rs_wrapx_cell`, for other programs to include).
//!
//! Precision contract (V: the CPU functions are the reference and stay untouched): every kernel
//! reproduces its CPU function's operation order and precision per output sample (the module
//! headers of the kernel sources state each one); a multi pass algorithm runs one launch per pass
//! in the CPU's order with an intermediate buffer between passes; an in place CPU step is a
//! kernel whose items each read and write only their own element, which is equivalent. Taps are
//! computed on the host (`conv::gaussian_kernel`, `conv::BINOMIAL5`) and the scales of the
//! resamples by the CPU's own expressions. Integer outputs never hold a NaN; every f32 result
//! that holds a NaN is refused with [`StageError::NanOutput`] (the devices write other NaN
//! payloads than the CPU), and the caller reruns the stage on the CPU. A NaN payload cannot change
//! any value that is not a NaN (no step compares payloads or selects by them), so checking the
//! final result is enough.
//!
//! Entry points, per CPU function `foo`: `foo_enqueue(session, <device buffers>, ...)` queues the
//! kernels only (it may create device buffers and the small tap buffers, it never moves plane
//! data between host and device), and `foo_gpu(session, <the CPU function's arguments>)` uploads,
//! queues, downloads and returns the CPU function's result or a [`StageError`]. On an error the
//! host data are left as they were. Both are written once over the [`Backend`] trait, which the
//! device ([`Session`]) and the device-free simulator (`sim::SimBackend`, tests only) implement,
//! so the simulator runs the very orchestration the device runs.
//!
//! Stages left on the CPU (they need a libm transcendental whose bits come from the host's C
//! runtime): `conv::gaussian_kernel` (the taps, f64 `powf`), `prep::delinearize` (f64 `powf`),
//! `flow::start_level` (f64 `powf` and `ln`). `prep::edge_enhance` (the solver's gradient) is the
//! solver stage's; `build_pyramid_gpu` runs it and Delinearize on the host between device steps.

pub mod conv;
pub mod holefill;
pub mod pyramid;
pub mod sample;

#[cfg(test)]
pub(crate) mod corpus;
#[cfg(test)]
pub(crate) mod sim;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_device;

use super::cl::{self, Access, Arg, Buffer, Pod, Range};
pub use super::image::StageError;
pub use super::selfcheck::Session;
use crate::device::kernels::{self, conv as kconv, holefill as khf, pyramid as kpm, resample as krs};

/// The sources of these stages in program order (after the prelude).
pub fn program_parts() -> Vec<&'static str> {
    vec![kconv::CV_WRAPX_TAP, kconv::CONV, kconv::CHOKE, krs::SAMPLERS, krs::RESAMPLE, kpm::PYRAMID, khf::HOLEFILL]
}

/// The program of these stages alone: the prelude and [`program_parts`].
pub fn program() -> String {
    kernels::program(&program_parts())
}

/// The most samples one stage buffer may hold: one launch item per sample, and every index the
/// kernels compute fits an `int`.
pub const MAX_SAMPLES: usize = cl::MAX_WORK_ITEMS;

/// The scalar arguments and buffers of one kernel launch. `n` is the item count of a 1D launch.
pub enum Launch<'a, B: Backend + ?Sized> {
    /// `cv_hpass` or, with `wrap`, `cv_hpass_wrapx`: `src` to `dst`, `w * c` by `h` items.
    HPass { src: &'a B::Buf<f32>, dst: &'a B::Buf<f32>, w: usize, h: usize, c: usize, taps: &'a Taps<B>, wrap: bool },
    /// `cv_vpass`.
    VPass { src: &'a B::Buf<f32>, dst: &'a B::Buf<f32>, w: usize, h: usize, c: usize, taps: &'a Taps<B> },
    /// `rs_resample1` or `rs_resample1_wrapx`, `dw` by `dh` items.
    Resample { src: &'a B::Buf<f32>, sw: usize, sh: usize, dst: &'a B::Buf<f32>, dw: usize, dh: usize, sx: f32, sy: f32, wrap: bool },
    /// `rs_gather1` (`lanes` 1), `rs_gather2` (2) or `rs_gather1_wrapx` (1, `wrap`).
    Gather { img: &'a B::Buf<f32>, w: usize, h: usize, xs: &'a B::Buf<f32>, ys: &'a B::Buf<f32>, out: &'a B::Buf<f32>, n: usize, lanes: usize, wrap: bool },
    /// `pm_upsample_flow`; `scale` is `[sx, sy, rx, ry]`.
    Upsample { src: &'a B::Buf<f32>, sw: usize, sh: usize, dst: &'a B::Buf<f32>, dw: usize, dh: usize, scale: [f32; 4] },
    LumaU8 { img: &'a B::Buf<u8>, out: &'a B::Buf<f32>, track_alpha: bool, video: bool, n: usize },
    LumaU16 { img: &'a B::Buf<u16>, out: &'a B::Buf<f32>, track_alpha: bool, video: bool, n: usize },
    LumaF32 { img: &'a B::Buf<f32>, out: &'a B::Buf<f32>, track_alpha: bool, video: bool, n: usize },
    /// `pm_subtract`: `a = a - b`.
    Subtract { a: &'a B::Buf<f32>, b: &'a B::Buf<f32>, n: usize },
    /// `pm_binarize`, in place.
    Binarize { p: &'a B::Buf<f32>, n: usize },
    FlagsU8 { m: &'a B::Buf<u8>, hole: &'a B::Buf<u8>, eps: f64, n: usize },
    FlagsU16 { m: &'a B::Buf<u16>, hole: &'a B::Buf<u8>, eps: f64, n: usize },
    FlagsF32 { m: &'a B::Buf<f32>, hole: &'a B::Buf<u8>, eps: f64, n: usize },
    Weight0 { hole: &'a B::Buf<u8>, wt: &'a B::Buf<f32>, n: usize },
    Load { plane: &'a B::Buf<f32>, hole: &'a B::Buf<u8>, v0: &'a B::Buf<f32>, n: usize },
    /// `hf_combine`, in place on `sol`.
    Combine { sol: &'a B::Buf<f32>, value: &'a B::Buf<f32>, weight: &'a B::Buf<f32>, n: usize },
    ExtractU8 { img: &'a B::Buf<u8>, plane: &'a B::Buf<f32>, stride: usize, c: usize, n: usize },
    ExtractU16 { img: &'a B::Buf<u16>, plane: &'a B::Buf<f32>, stride: usize, c: usize, n: usize },
    ExtractF32 { img: &'a B::Buf<f32>, plane: &'a B::Buf<f32>, stride: usize, c: usize, n: usize },
    StoreU8 { plane: &'a B::Buf<f32>, img: &'a B::Buf<u8>, stride: usize, c: usize, n: usize },
    StoreU16 { plane: &'a B::Buf<f32>, img: &'a B::Buf<u16>, stride: usize, c: usize, n: usize },
    StoreF32 { plane: &'a B::Buf<f32>, img: &'a B::Buf<f32>, stride: usize, c: usize, n: usize },
    /// `cv_choke_invert`, in place.
    Invert { data: &'a B::Buf<f32>, n: usize },
    /// `cv_choke_threshold`, in place.
    Threshold { data: &'a B::Buf<f32>, t: f32, den: f32, n: usize },
    /// `cv_choke_mix`, in place on `data`.
    Mix { data: &'a B::Buf<f32>, c: &'a B::Buf<f32>, t: f32, it: f32, n: usize },
    /// `cv_choke_clamp`, in place.
    Clamp { data: &'a B::Buf<f32>, n: usize },
}

/// Where the stages run: the device ([`Session`]) or the simulator of the tests. Buffers are
/// shared references; a launch writes through them in queue order.
pub trait Backend {
    type Buf<T: Pod>;
    fn upload<T: Pod>(&self, data: &[T]) -> Result<Self::Buf<T>, StageError>;
    /// A buffer of `len` (at least 1) elements whose content is unspecified until written.
    fn alloc<T: Pod>(&self, len: usize) -> Result<Self::Buf<T>, StageError>;
    fn download<T: Pod>(&self, buf: &Self::Buf<T>) -> Result<Vec<T>, StageError>;
    fn length<T: Pod>(&self, buf: &Self::Buf<T>) -> usize;
    /// Copies all of `src` into `dst` (equal lengths, distinct buffers).
    fn copy<T: Pod>(&self, src: &Self::Buf<T>, dst: &Self::Buf<T>) -> Result<(), StageError>;
    fn launch(&self, l: Launch<'_, Self>) -> Result<(), StageError>;
}

/// A half kernel resident on the backend: `n` taps, `k[0]` the centre.
pub struct Taps<B: Backend + ?Sized> {
    pub buf: B::Buf<f32>,
    pub n: usize,
}

impl<B: Backend> Taps<B> {
    /// Uploads `k` (1 to [`kconv::CV_MAX_TAPS`] taps, else `StageError::Unsupported`).
    pub fn upload(b: &B, k: &[f32]) -> Result<Taps<B>, StageError> {
        if k.is_empty() || k.len() > kconv::CV_MAX_TAPS {
            return Err(StageError::Unsupported(format!(
                "a half kernel of {} taps (the device passes take 1 to {})",
                k.len(),
                kconv::CV_MAX_TAPS
            )));
        }
        Ok(Taps { buf: b.upload(k)?, n: k.len() })
    }

    /// The taps of `blur5` (`conv::BINOMIAL5`).
    pub fn binomial5(b: &B) -> Result<Taps<B>, StageError> {
        Taps::upload(b, &crate::flow::conv::BINOMIAL5)
    }

    /// The taps of `blur_gaussian` of radius `r`, or `None` when `blur_gaussian` does nothing
    /// (`r < BLUR_RADIUS_EPS`). The kernel is `conv::gaussian_kernel(r)` (host, f64 `powf`).
    pub fn gaussian(b: &B, r: f32) -> Result<Option<Taps<B>>, StageError> {
        if (r as f64) < crate::flow::conv::BLUR_RADIUS_EPS {
            return Ok(None);
        }
        Taps::upload(b, &crate::flow::conv::gaussian_kernel(r)).map(Some)
    }
}

/// `Err(Unsupported)` unless `n` samples fit one stage buffer (and `n > 0`).
pub(crate) fn check_samples(n: usize, what: &str) -> Result<(), StageError> {
    if n == 0 {
        return Err(StageError::Unsupported(format!("{what}: an empty plane")));
    }
    if n > MAX_SAMPLES {
        return Err(StageError::Unsupported(format!("{what}: {n} samples, the device stages take at most {MAX_SAMPLES}")));
    }
    Ok(())
}

/// `Err(Unsupported)` unless `buf` holds at least `n` elements.
pub(crate) fn check_len<B: Backend, T: Pod>(b: &B, buf: &B::Buf<T>, n: usize, what: &str) -> Result<(), StageError> {
    let len = b.length(buf);
    if len < n {
        return Err(StageError::Unsupported(format!("{what}: a buffer of {len} elements for {n} samples")));
    }
    Ok(())
}

fn i(v: usize) -> Arg<'static> {
    Arg::I32(v as i32)
}

fn flag(v: bool) -> Arg<'static> {
    Arg::I32(v as i32)
}

impl Backend for Session {
    type Buf<T: Pod> = Buffer<T>;

    fn upload<T: Pod>(&self, data: &[T]) -> Result<Buffer<T>, StageError> {
        Ok(Buffer::from_slice(&self.context, data, Access::ReadWrite)?)
    }

    fn alloc<T: Pod>(&self, len: usize) -> Result<Buffer<T>, StageError> {
        Ok(Buffer::new(&self.context, len, Access::ReadWrite)?)
    }

    fn download<T: Pod>(&self, buf: &Buffer<T>) -> Result<Vec<T>, StageError> {
        Ok(self.queue.read(buf)?)
    }

    fn length<T: Pod>(&self, buf: &Buffer<T>) -> usize {
        buf.len()
    }

    fn copy<T: Pod>(&self, src: &Buffer<T>, dst: &Buffer<T>) -> Result<(), StageError> {
        Ok(self.queue.copy(src, dst)?)
    }

    fn launch(&self, l: Launch<'_, Self>) -> Result<(), StageError> {
        let (name, range, args): (&str, Range, Vec<Arg<'_>>) = match l {
            Launch::HPass { src, dst, w, h, c, taps, wrap } => (
                if wrap { "cv_hpass_wrapx" } else { "cv_hpass" },
                Range::D2(w * c, h),
                vec![src.arg(), dst.arg(), i(w), i(h), i(c), taps.buf.arg(), i(taps.n)],
            ),
            Launch::VPass { src, dst, w, h, c, taps } => {
                ("cv_vpass", Range::D2(w * c, h), vec![src.arg(), dst.arg(), i(w), i(h), i(c), taps.buf.arg(), i(taps.n)])
            }
            Launch::Resample { src, sw, sh, dst, dw, dh, sx, sy, wrap } => (
                if wrap { "rs_resample1_wrapx" } else { "rs_resample1" },
                Range::D2(dw, dh),
                vec![src.arg(), i(sw), i(sh), dst.arg(), i(dw), i(dh), Arg::F32(sx), Arg::F32(sy)],
            ),
            Launch::Gather { img, w, h, xs, ys, out, n, lanes, wrap } => (
                match (lanes, wrap) {
                    (2, _) => "rs_gather2",
                    (_, true) => "rs_gather1_wrapx",
                    _ => "rs_gather1",
                },
                Range::D1(n),
                vec![img.arg(), i(w), i(h), xs.arg(), ys.arg(), out.arg(), i(n)],
            ),
            Launch::Upsample { src, sw, sh, dst, dw, dh, scale } => (
                "pm_upsample_flow",
                Range::D2(dw, dh),
                vec![
                    src.arg(),
                    i(sw),
                    i(sh),
                    dst.arg(),
                    i(dw),
                    i(dh),
                    Arg::F32(scale[0]),
                    Arg::F32(scale[1]),
                    Arg::F32(scale[2]),
                    Arg::F32(scale[3]),
                ],
            ),
            Launch::LumaU8 { img, out, track_alpha, video, n } => {
                ("pm_luma_u8", Range::D1(n), vec![img.arg(), out.arg(), flag(track_alpha), flag(video), i(n)])
            }
            Launch::LumaU16 { img, out, track_alpha, video, n } => {
                ("pm_luma_u16", Range::D1(n), vec![img.arg(), out.arg(), flag(track_alpha), flag(video), i(n)])
            }
            Launch::LumaF32 { img, out, track_alpha, video, n } => {
                ("pm_luma_f32", Range::D1(n), vec![img.arg(), out.arg(), flag(track_alpha), flag(video), i(n)])
            }
            Launch::Subtract { a, b, n } => ("pm_subtract", Range::D1(n), vec![a.arg(), b.arg(), i(n)]),
            Launch::Binarize { p, n } => ("pm_binarize", Range::D1(n), vec![p.arg(), i(n)]),
            Launch::FlagsU8 { m, hole, eps, n } => ("hf_flags_u8", Range::D1(n), vec![m.arg(), hole.arg(), Arg::F64(eps), i(n)]),
            Launch::FlagsU16 { m, hole, eps, n } => ("hf_flags_u16", Range::D1(n), vec![m.arg(), hole.arg(), Arg::F64(eps), i(n)]),
            Launch::FlagsF32 { m, hole, eps, n } => ("hf_flags_f32", Range::D1(n), vec![m.arg(), hole.arg(), Arg::F64(eps), i(n)]),
            Launch::Weight0 { hole, wt, n } => ("hf_weight0", Range::D1(n), vec![hole.arg(), wt.arg(), i(n)]),
            Launch::Load { plane, hole, v0, n } => ("hf_load", Range::D1(n), vec![plane.arg(), hole.arg(), v0.arg(), i(n)]),
            Launch::Combine { sol, value, weight, n } => ("hf_combine", Range::D1(n), vec![sol.arg(), value.arg(), weight.arg(), i(n)]),
            Launch::ExtractU8 { img, plane, stride, c, n } => ("hf_extract_u8", Range::D1(n), vec![img.arg(), plane.arg(), i(stride), i(c), i(n)]),
            Launch::ExtractU16 { img, plane, stride, c, n } => ("hf_extract_u16", Range::D1(n), vec![img.arg(), plane.arg(), i(stride), i(c), i(n)]),
            Launch::ExtractF32 { img, plane, stride, c, n } => ("hf_extract_f32", Range::D1(n), vec![img.arg(), plane.arg(), i(stride), i(c), i(n)]),
            Launch::StoreU8 { plane, img, stride, c, n } => ("hf_store_u8", Range::D1(n), vec![plane.arg(), img.arg(), i(stride), i(c), i(n)]),
            Launch::StoreU16 { plane, img, stride, c, n } => ("hf_store_u16", Range::D1(n), vec![plane.arg(), img.arg(), i(stride), i(c), i(n)]),
            Launch::StoreF32 { plane, img, stride, c, n } => ("hf_store_f32", Range::D1(n), vec![plane.arg(), img.arg(), i(stride), i(c), i(n)]),
            Launch::Invert { data, n } => ("cv_choke_invert", Range::D1(n), vec![data.arg(), i(n)]),
            Launch::Threshold { data, t, den, n } => ("cv_choke_threshold", Range::D1(n), vec![data.arg(), Arg::F32(t), Arg::F32(den), i(n)]),
            Launch::Mix { data, c, t, it, n } => ("cv_choke_mix", Range::D1(n), vec![data.arg(), c.arg(), Arg::F32(t), Arg::F32(it), i(n)]),
            Launch::Clamp { data, n } => ("cv_choke_clamp", Range::D1(n), vec![data.arg(), i(n)]),
        };
        let mut kernel = self.kernel(name)?;
        self.queue.run(&mut kernel, range, &args)?;
        Ok(())
    }
}
