//! Device stages of `flow/sample.rs`: `resample1` and `resample1_wrapx` (the `rs_resample1*`
//! kernels), and the samplers `bilinear1`, `bilinear2` and `bilinear1_wrapx` at a list of points
//! (the `rs_gather*` kernels, which call the device helpers of `resample::SAMPLERS` that the
//! solver's kernels call too).
//!
//! Precision contract: that of `kernels/resample.rs`; the scales are the CPU's expressions,
//! evaluated here on the host ([`resample1_scales`], [`resample1_wrapx_scales`]).
//!
//! Sizes: the plain samplers need `w, h >= 2` and the wrap sampler `w >= 1, h >= 2` (smaller
//! planes index outside the CPU's slices, so they are `StageError::Unsupported`); an empty
//! destination is the CPU's empty result.

use super::{check_len, check_samples, Backend, Launch, Session, StageError};
use crate::device::cl::Buffer;
use crate::device::image::reject_nan;

/// `(sx, sy)` of `sample::resample1`: `((f32) sw - 1.0) / (f32)(dw - 1)` and the same for y.
pub fn resample1_scales(sw: usize, sh: usize, dw: usize, dh: usize) -> (f32, f32) {
    let sx = (sw as f32 - 1.0) / ((dw as i32 - 1) as f32);
    let sy = (sh as f32 - 1.0) / ((dh as i32 - 1) as f32);
    (sx, sy)
}

/// `(sx, sy)` of `sample::resample1_wrapx`: `(f32) sw / (f32) dw` and the corner aligned `sy`.
pub fn resample1_wrapx_scales(sw: usize, sh: usize, dw: usize, dh: usize) -> (f32, f32) {
    let sx = (sw as i32 as f32) / (dw as i32 as f32);
    let sy = (sh as f32 - 1.0) / ((dh as i32 - 1) as f32);
    (sx, sy)
}

fn check_source(sw: usize, sh: usize, wrap: bool, what: &str) -> Result<(), StageError> {
    let min_w = if wrap { 1 } else { 2 };
    if sw < min_w || sh < 2 {
        return Err(StageError::Unsupported(format!("{what}: a {sw}x{sh} source (the sampler needs at least {min_w}x2)")));
    }
    Ok(())
}

/// [`resample1_enqueue`] over any backend.
#[allow(clippy::too_many_arguments)]
pub fn resample1_on<B: Backend>(
    b: &B,
    src: &B::Buf<f32>,
    sw: usize,
    sh: usize,
    dst: &B::Buf<f32>,
    dw: usize,
    dh: usize,
    wrap: bool,
) -> Result<(), StageError> {
    check_source(sw, sh, wrap, "resample")?;
    let n_src = sw.checked_mul(sh).unwrap_or(usize::MAX);
    let n_dst = dw.checked_mul(dh).unwrap_or(usize::MAX);
    check_samples(n_src, "resample")?;
    check_samples(n_dst, "resample")?;
    check_len(b, src, n_src, "resample")?;
    check_len(b, dst, n_dst, "resample")?;
    let (sx, sy) = if wrap { resample1_wrapx_scales(sw, sh, dw, dh) } else { resample1_scales(sw, sh, dw, dh) };
    b.launch(Launch::Resample { src, sw, sh, dst, dw, dh, sx, sy, wrap })
}

/// `sample::resample1` from one device plane into another (`src` `sw x sh`, `dst` `dw x dh`).
pub fn resample1_enqueue(s: &Session, src: &Buffer<f32>, sw: usize, sh: usize, dst: &Buffer<f32>, dw: usize, dh: usize) -> Result<(), StageError> {
    resample1_on(s, src, sw, sh, dst, dw, dh, false)
}

/// `sample::resample1_wrapx` from one device plane into another.
pub fn resample1_wrapx_enqueue(s: &Session, src: &Buffer<f32>, sw: usize, sh: usize, dst: &Buffer<f32>, dw: usize, dh: usize) -> Result<(), StageError> {
    resample1_on(s, src, sw, sh, dst, dw, dh, true)
}

/// The resample of a host plane on the backend.
pub fn resample1_host<B: Backend>(b: &B, src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize, wrap: bool) -> Result<Vec<f32>, StageError> {
    if dw == 0 || dh == 0 {
        return Ok(vec![0.0f32; dw * dh]);
    }
    if Some(src.len()) != sw.checked_mul(sh) {
        return Err(StageError::Unsupported(format!("resample: {} samples for {sw}x{sh}", src.len())));
    }
    check_source(sw, sh, wrap, "resample")?;
    let sbuf = b.upload(src)?;
    let dbuf = b.alloc::<f32>(dw * dh)?;
    resample1_on(b, &sbuf, sw, sh, &dbuf, dw, dh, wrap)?;
    let out = b.download(&dbuf)?;
    reject_nan(&out)?;
    Ok(out)
}

/// `sample::resample1` on the device.
pub fn resample1_gpu(s: &Session, src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Result<Vec<f32>, StageError> {
    resample1_host(s, src, sw, sh, dw, dh, false)
}

/// `sample::resample1_wrapx` on the device.
pub fn resample1_wrapx_gpu(s: &Session, src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Result<Vec<f32>, StageError> {
    resample1_host(s, src, sw, sh, dw, dh, true)
}

/// Which sampler [`gather_host`] evaluates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sampler {
    /// `sample::bilinear1`, one value per point.
    Bilinear1,
    /// `sample::bilinear2` of an interleaved two channel image, two values per point.
    Bilinear2,
    /// `sample::bilinear1_wrapx`, one value per point.
    Bilinear1Wrapx,
}

/// The sampler at every point `(xs[i], ys[i])` of a `w x h` image (`img` holds `w * h` values,
/// twice that for [`Sampler::Bilinear2`]) on the backend.
pub fn gather_host<B: Backend>(b: &B, sampler: Sampler, img: &[f32], w: usize, h: usize, xs: &[f32], ys: &[f32]) -> Result<Vec<f32>, StageError> {
    let lanes = if sampler == Sampler::Bilinear2 { 2 } else { 1 };
    let wrap = sampler == Sampler::Bilinear1Wrapx;
    check_source(w, h, wrap, "gather")?;
    if Some(img.len()) != w.checked_mul(h).and_then(|v| v.checked_mul(lanes)) {
        return Err(StageError::Unsupported(format!("gather: {} samples for {w}x{h}x{lanes}", img.len())));
    }
    if xs.len() != ys.len() {
        return Err(StageError::Unsupported("gather: as many x as y coordinates".into()));
    }
    if xs.is_empty() {
        return Ok(Vec::new());
    }
    check_samples(img.len(), "gather")?;
    check_samples(xs.len() * lanes, "gather")?;
    let ib = b.upload(img)?;
    let xb = b.upload(xs)?;
    let yb = b.upload(ys)?;
    let ob = b.alloc::<f32>(xs.len() * lanes)?;
    b.launch(Launch::Gather { img: &ib, w, h, xs: &xb, ys: &yb, out: &ob, n: xs.len(), lanes, wrap })?;
    let out = b.download(&ob)?;
    reject_nan(&out)?;
    Ok(out)
}

/// The samplers on the device at a list of points (see [`gather_host`]).
pub fn gather_gpu(s: &Session, sampler: Sampler, img: &[f32], w: usize, h: usize, xs: &[f32], ys: &[f32]) -> Result<Vec<f32>, StageError> {
    gather_host(s, sampler, img, w, h, xs, ys)
}
