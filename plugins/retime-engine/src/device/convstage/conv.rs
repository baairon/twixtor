//! Device stages of `flow/conv.rs` (`convolve`, `blur_gaussian`, `blur5` and their `_wrapx`
//! variants) and of `matte::choke`, built on the `cv_` kernels.
//!
//! Precision contract: that of `kernels/conv.rs`. A convolution is two launches, the horizontal
//! pass from the source into an intermediate buffer and the vertical pass from it into the
//! destination, as `conv::convolve` does with its temporary; the destination may be the source.
//! The choke runs `matte::choke`'s steps in its order (invert, the shrink of `floor` and of
//! `ceil` of the amount on a copy, the mix, invert, clamp), every scalar (`r`, `fr`, the radius,
//! `T`, `1.0 - T`, `t`, `1.0 - t`) computed on the host by the CPU's expressions.

use super::{check_len, check_samples, Backend, Launch, Session, StageError, Taps};
use crate::device::cl::Buffer;
use crate::device::image::reject_nan;
use crate::flow::conv::{gaussian_kernel, BINOMIAL5};
use crate::matte::{Matte, CHOKE_MAX};

fn check_plane<B: Backend>(b: &B, bufs: &[&B::Buf<f32>], w: usize, h: usize, c: usize, what: &str) -> Result<(), StageError> {
    if !(1..=3).contains(&c) {
        return Err(StageError::Unsupported(format!("{what}: {c} channels (the passes take 1 to 3)")));
    }
    let n = w.checked_mul(h).and_then(|v| v.checked_mul(c)).unwrap_or(usize::MAX);
    check_samples(n, what)?;
    for buf in bufs {
        check_len(b, *buf, n, what)?;
    }
    Ok(())
}

/// [`convolve_enqueue`] over any backend.
#[allow(clippy::too_many_arguments)]
pub fn convolve_on<B: Backend>(
    b: &B,
    src: &B::Buf<f32>,
    tmp: &B::Buf<f32>,
    dst: &B::Buf<f32>,
    w: usize,
    h: usize,
    c: usize,
    taps: &Taps<B>,
    wrap: bool,
) -> Result<(), StageError> {
    if std::ptr::eq(tmp, src) || std::ptr::eq(tmp, dst) {
        return Err(StageError::Unsupported("convolve: the intermediate buffer must differ from the source and the destination".into()));
    }
    check_plane(b, &[src, tmp, dst], w, h, c, "convolve")?;
    b.launch(Launch::HPass { src, dst: tmp, w, h, c, taps, wrap })?;
    b.launch(Launch::VPass { src: tmp, dst, w, h, c, taps })
}

/// `conv::convolve` on device buffers of `w * h * c` samples: the horizontal pass `src -> tmp`,
/// the vertical pass `tmp -> dst` (`dst` may be `src`; `tmp` is a third buffer).
#[allow(clippy::too_many_arguments)]
pub fn convolve_enqueue(
    s: &Session,
    src: &Buffer<f32>,
    tmp: &Buffer<f32>,
    dst: &Buffer<f32>,
    w: usize,
    h: usize,
    c: usize,
    taps: &Taps<Session>,
) -> Result<(), StageError> {
    convolve_on(s, src, tmp, dst, w, h, c, taps, false)
}

/// `conv::convolve_wrapx` on device buffers (the circular horizontal pass, the plain vertical one).
#[allow(clippy::too_many_arguments)]
pub fn convolve_wrapx_enqueue(
    s: &Session,
    src: &Buffer<f32>,
    tmp: &Buffer<f32>,
    dst: &Buffer<f32>,
    w: usize,
    h: usize,
    c: usize,
    taps: &Taps<Session>,
) -> Result<(), StageError> {
    convolve_on(s, src, tmp, dst, w, h, c, taps, true)
}

/// `conv::blur5` on device buffers (`taps` from `Taps::binomial5`).
pub fn blur5_enqueue(s: &Session, src: &Buffer<f32>, tmp: &Buffer<f32>, dst: &Buffer<f32>, w: usize, h: usize, taps: &Taps<Session>) -> Result<(), StageError> {
    convolve_on(s, src, tmp, dst, w, h, 1, taps, false)
}

/// `conv::blur5_wrapx` on device buffers.
pub fn blur5_wrapx_enqueue(s: &Session, src: &Buffer<f32>, tmp: &Buffer<f32>, dst: &Buffer<f32>, w: usize, h: usize, taps: &Taps<Session>) -> Result<(), StageError> {
    convolve_on(s, src, tmp, dst, w, h, 1, taps, true)
}

/// `conv::blur_gaussian` on device buffers; `taps` from `Taps::gaussian(r)`, `None` (a radius
/// below `BLUR_RADIUS_EPS`) copies `src` to `dst` (nothing when they are one buffer).
#[allow(clippy::too_many_arguments)]
pub fn blur_gaussian_on<B: Backend>(
    b: &B,
    src: &B::Buf<f32>,
    tmp: &B::Buf<f32>,
    dst: &B::Buf<f32>,
    w: usize,
    h: usize,
    c: usize,
    taps: Option<&Taps<B>>,
    wrap: bool,
) -> Result<(), StageError> {
    match taps {
        Some(t) => convolve_on(b, src, tmp, dst, w, h, c, t, wrap),
        None if std::ptr::eq(src, dst) => Ok(()),
        None => b.copy(src, dst),
    }
}

/// `conv::blur_gaussian` on device buffers (see [`blur_gaussian_on`]).
#[allow(clippy::too_many_arguments)]
pub fn blur_gaussian_enqueue(
    s: &Session,
    src: &Buffer<f32>,
    tmp: &Buffer<f32>,
    dst: &Buffer<f32>,
    w: usize,
    h: usize,
    c: usize,
    taps: Option<&Taps<Session>>,
) -> Result<(), StageError> {
    blur_gaussian_on(s, src, tmp, dst, w, h, c, taps, false)
}

/// `conv::blur_gaussian_wrapx` on device buffers.
#[allow(clippy::too_many_arguments)]
pub fn blur_gaussian_wrapx_enqueue(
    s: &Session,
    src: &Buffer<f32>,
    tmp: &Buffer<f32>,
    dst: &Buffer<f32>,
    w: usize,
    h: usize,
    c: usize,
    taps: Option<&Taps<Session>>,
) -> Result<(), StageError> {
    blur_gaussian_on(s, src, tmp, dst, w, h, c, taps, true)
}

// ---- host wrappers ----------------------------------------------------------------------------

/// The in place convolution of host data on the backend: `data` changes only on success.
pub fn convolve_host<B: Backend>(b: &B, data: &mut [f32], w: usize, h: usize, c: usize, k: &[f32], wrap: bool) -> Result<(), StageError> {
    if !(1..=3).contains(&c) {
        return Err(StageError::Unsupported(format!("convolve: {c} channels are not used by the solver")));
    }
    if Some(data.len()) != w.checked_mul(h).and_then(|v| v.checked_mul(c)) {
        return Err(StageError::Unsupported(format!("convolve: {} samples for {w}x{h}x{c}", data.len())));
    }
    if data.is_empty() || k.is_empty() {
        return Ok(());
    }
    check_samples(data.len(), "convolve")?;
    let taps = Taps::upload(b, k)?;
    let buf = b.upload(data)?;
    let tmp = b.alloc::<f32>(data.len())?;
    convolve_on(b, &buf, &tmp, &buf, w, h, c, &taps, wrap)?;
    let out = b.download(&buf)?;
    reject_nan(&out)?;
    data.copy_from_slice(&out);
    Ok(())
}

/// `conv::convolve` on the device.
pub fn convolve_gpu(s: &Session, data: &mut [f32], w: usize, h: usize, c: usize, k: &[f32]) -> Result<(), StageError> {
    convolve_host(s, data, w, h, c, k, false)
}

/// `conv::convolve_wrapx` on the device.
pub fn convolve_wrapx_gpu(s: &Session, data: &mut [f32], w: usize, h: usize, c: usize, k: &[f32]) -> Result<(), StageError> {
    convolve_host(s, data, w, h, c, k, true)
}

/// `blur_gaussian` / `blur_gaussian_wrapx` of host data on the backend.
pub fn blur_gaussian_host<B: Backend>(b: &B, data: &mut [f32], w: usize, h: usize, c: usize, r: f32, wrap: bool) -> Result<(), StageError> {
    if (r as f64) < crate::flow::conv::BLUR_RADIUS_EPS {
        return Ok(());
    }
    convolve_host(b, data, w, h, c, &gaussian_kernel(r), wrap)
}

/// `conv::blur_gaussian` on the device.
pub fn blur_gaussian_gpu(s: &Session, data: &mut [f32], w: usize, h: usize, c: usize, r: f32) -> Result<(), StageError> {
    blur_gaussian_host(s, data, w, h, c, r, false)
}

/// `conv::blur_gaussian_wrapx` on the device.
pub fn blur_gaussian_wrapx_gpu(s: &Session, data: &mut [f32], w: usize, h: usize, c: usize, r: f32) -> Result<(), StageError> {
    blur_gaussian_host(s, data, w, h, c, r, true)
}

/// `conv::blur5` on the device.
pub fn blur5_gpu(s: &Session, data: &mut [f32], w: usize, h: usize) -> Result<(), StageError> {
    convolve_host(s, data, w, h, 1, &BINOMIAL5, false)
}

/// `conv::blur5_wrapx` on the device.
pub fn blur5_wrapx_gpu(s: &Session, data: &mut [f32], w: usize, h: usize) -> Result<(), StageError> {
    convolve_host(s, data, w, h, 1, &BINOMIAL5, true)
}

// ---- matte choke ------------------------------------------------------------------------------

/// The scalars of one shrink step of `matte::choke` by the whole number `n >= 1`: the Gaussian
/// radius, `T = 1.0 - (k[n] + k[n + 1])` and `1.0 - T`, by the CPU's expressions.
fn shrink_scalars(n: f32) -> (f32, f32, f32) {
    let r = (n + 1.0f32).ceil();
    let mut fr = r - (n + 1.0f32);
    if fr < 0.01f32 {
        fr = fr + 0.01f32;
    }
    let radius = (1.0f32 - fr) + r;
    let k = gaussian_kernel(radius);
    let ni = n as usize;
    let t = 1.0f32 - (k[ni] + k[ni + 1]);
    let den = 1.0f32 - t;
    (radius, t, den)
}

/// One shrink step in place on `data` (`tmp` the intermediate of the blur).
fn shrink_on<B: Backend>(b: &B, data: &B::Buf<f32>, tmp: &B::Buf<f32>, w: usize, h: usize, n: f32) -> Result<(), StageError> {
    if !(n >= 1.0) {
        return Ok(());
    }
    let (radius, t, den) = shrink_scalars(n);
    let taps = Taps::gaussian(b, radius)?;
    blur_gaussian_on(b, data, tmp, data, w, h, 1, taps.as_ref(), false)?;
    b.launch(Launch::Threshold { data, t, den, n: w * h })
}

/// `matte::choke` on a device plane of `w * h` samples, in place. A NaN, infinite or zero amount
/// queues nothing (the CPU's identity). Records gap G1 as the CPU does when the choke runs.
pub fn choke_on<B: Backend>(b: &B, data: &B::Buf<f32>, w: usize, h: usize, amount: f32) -> Result<(), StageError> {
    if amount == 0.0 || !amount.is_finite() {
        return Ok(());
    }
    let n = w.checked_mul(h).unwrap_or(usize::MAX);
    check_samples(n, "choke")?;
    check_len(b, data, n, "choke")?;
    let grow = amount < 0.0;
    let mut a = if grow { -amount } else { amount };
    if a > CHOKE_MAX {
        a = CHOKE_MAX;
    }
    let lo = a.floor();
    let hi = a.ceil();
    // the largest kernel the steps need must fit the device passes
    let need = shrink_taps(hi);
    if need > crate::device::kernels::conv::CV_MAX_TAPS {
        return Err(StageError::Unsupported(format!("choke of {a} pixels: a half kernel of {need} taps")));
    }
    fx_core::gaps::record(&crate::gaps::G1);
    let tmp = b.alloc::<f32>(n)?;
    if grow {
        b.launch(Launch::Invert { data, n })?;
    }
    if lo == hi {
        shrink_on(b, data, &tmp, w, h, lo)?;
    } else {
        let c = b.alloc::<f32>(n)?;
        b.copy(data, &c)?;
        shrink_on(b, data, &tmp, w, h, lo)?;
        shrink_on(b, &c, &tmp, w, h, hi)?;
        let t = a - lo;
        let it = 1.0f32 - t;
        b.launch(Launch::Mix { data, c: &c, t, it, n })?;
    }
    if grow {
        b.launch(Launch::Invert { data, n })?;
    }
    b.launch(Launch::Clamp { data, n })
}

/// The taps of the shrink step by `n` (0 when it is the identity).
fn shrink_taps(n: f32) -> usize {
    if !(n >= 1.0) {
        return 0;
    }
    let (radius, _, _) = shrink_scalars(n);
    gaussian_kernel(radius).len()
}

/// `matte::choke` on a device plane (see [`choke_on`]).
pub fn choke_enqueue(s: &Session, data: &Buffer<f32>, w: usize, h: usize, amount: f32) -> Result<(), StageError> {
    choke_on(s, data, w, h, amount)
}

/// `matte::choke` of a host matte on the backend; the matte changes only on success. A matte
/// that does not hold `width * height` values is `Unsupported` whatever the amount (the CPU's
/// `Mismatch`, checked first as the CPU does), before anything is uploaded.
pub fn choke_host<B: Backend>(b: &B, m: &mut Matte, amount: f32) -> Result<(), StageError> {
    if m.width.checked_mul(m.height) != Some(m.data.len()) {
        return Err(StageError::Unsupported(format!("choke: a {}x{} matte holds {} values", m.width, m.height, m.data.len())));
    }
    if amount == 0.0 || !amount.is_finite() {
        return Ok(());
    }
    if m.data.is_empty() {
        fx_core::gaps::record(&crate::gaps::G1);
        return Ok(());
    }
    let buf = b.upload(&m.data)?;
    choke_on(b, &buf, m.width, m.height, amount)?;
    let out = b.download(&buf)?;
    reject_nan(&out)?;
    m.data.copy_from_slice(&out);
    Ok(())
}

/// `matte::choke` on the device.
pub fn choke_gpu(s: &Session, m: &mut Matte, amount: f32) -> Result<(), StageError> {
    choke_host(s, m, amount)
}
