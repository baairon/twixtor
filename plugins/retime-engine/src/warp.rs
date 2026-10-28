//! Inverse (backward) warp of the engine and the Blur Compensation tail.
//!
//! `W(p) = img(p + F(p) * k)` with the bilinear samplers, edge fade, coverage image and the
//! store rules of every pixel depth.
//!
//! Numerics: single precision in the order written here, no fused multiply add, truncating
//! conversions with the saturation documented per depth, plus the double precision steps noted
//! at the functions. Rows are independent and run through rayon row chunks, so the output does
//! not depend on the thread count.

use crate::flow::FlowField;
use crate::image::{Image, Samples, CHANNELS};
use fx_core::{par_rows, FxError, Result};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(test)]
mod tests;

mod blurcomp;
pub use blurcomp::{blur_amount, blur_compensation, blur_compensation_wrap};
#[cfg(test)]
pub(crate) use blurcomp::{clear_rgb_where_transparent, motion_blur};

// ---------------------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------------------

/// Options of the inverse warp: `hold`, `mode` and `is360`. The coverage image is the separate
/// entry point [`inv_warp_coverage`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WarpOpts {
    /// When true the sample point is clipped to the image rectangle (`clip_to_rect`) and nothing
    /// fades; when false samples that leave the image fade out and the fade is written to the
    /// coverage image (Smart Blend, warp code 1). The compositor uses `hold = true` for warp
    /// code 0.
    pub hold: bool,
    /// `is360`: horizontal wrap and pole reflection for equirectangular video (the same code in
    /// all three depths). Takes precedence over `hold`.
    pub is360: bool,
    /// 1 = plain bilinear (every call of the compositor). Any other value (mip map sampling) is
    /// refused with [`FxError::Unsupported`].
    pub mode: i32,
}

impl Default for WarpOpts {
    /// The call the compositor makes for warp mode 0: `hold = 1`, `mode = 1`, `is360 = 0`.
    fn default() -> Self {
        WarpOpts { hold: true, is360: false, mode: 1 }
    }
}

/// The one channel uint8 coverage image of the inverse warp (warp code 1, Smart Blend):
/// 255 where the source was sampled inside the image, 0 where the sample left the image
/// by one pixel or more, and the faded edge weight in between.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Coverage {
    pub width: usize,
    pub height: usize,
    /// `width * height` bytes, row major, top row first.
    pub data: Vec<u8>,
}

impl Coverage {
    /// A coverage image filled with `fill`.
    pub fn new(width: usize, height: usize, fill: u8) -> Coverage {
        Coverage { width, height, data: vec![fill; width * height] }
    }

    /// The byte at pixel `(x, y)`.
    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.width + x]
    }

    /// `255 - b` per byte: the "holes" mask the compositor derives from a coverage image.
    pub fn invert(&mut self) {
        for b in self.data.iter_mut() {
            *b = 0xff - *b;
        }
    }

    /// Per byte maximum with `other` (`a = max(a, b)`): the coverage combine of the two Smart Blend
    /// warps. Panics when the sizes differ (a programming error).
    pub fn combine_max(&mut self, other: &Coverage) {
        assert!(
            self.width == other.width && self.height == other.height,
            "coverage images need equal sizes: {}x{} against {}x{}",
            self.width,
            self.height,
            other.width,
            other.height
        );
        for (a, &b) in self.data.iter_mut().zip(&other.data) {
            *a = (*a).max(b);
        }
    }
}

// ---------------------------------------------------------------------------------------
// Numeric helpers
// ---------------------------------------------------------------------------------------

/// Truncation toward zero; NaN and values outside the 32 bit range give the integer indefinite
/// `i32::MIN`.
#[inline(always)]
pub(crate) fn cvtt32(x: f32) -> i32 {
    if x.is_nan() || x >= 2147483648.0 || x < -2147483648.0 {
        i32::MIN
    } else {
        x as i32
    }
}

/// Truncation toward zero of a double; NaN and out of range values give `i64::MIN`.
#[inline(always)]
fn cvtt64(x: f64) -> i64 {
    if x.is_nan() || x >= 9.223372036854775808e18 || x < -9.223372036854775808e18 {
        i64::MIN
    } else {
        x as i64
    }
}

/// Store rule of the 8 bpc kernels for an interpolated sample: add 0.5 (single precision),
/// truncate, bias by -128 (wrapping), saturate to the signed 16 bit range and then the signed
/// 8 bit range, add the bias back. The result is `clamp(trunc(v + 0.5), 0, 255)`, with the
/// indefinite integer of a NaN or out of range value wrapping through the bias to 255.
#[inline(always)]
pub(crate) fn sat_u8(v: f32) -> u8 {
    let i = cvtt32(v + 0.5f32);
    let t = i.wrapping_add(-128);
    let w = t.clamp(-32768, 32767);
    let b = w.clamp(-128, 127);
    (b + 128) as u8
}

/// Store rule of the 16 bpc kernels: the same chain with the bias -32768 and a single
/// saturation to the signed 16 bit range; `clamp(trunc(v + 0.5), 0, 65535)`, an indefinite
/// integer gives 65535.
#[inline(always)]
pub(crate) fn sat_u16(v: f32) -> u16 {
    let i = cvtt32(v + 0.5f32);
    let t = i.wrapping_add(-32768);
    let w = t.clamp(-32768, 32767);
    (w + 32768) as u16
}

/// Scale of an already stored 8 bpc sample by the edge fade `f`:
/// `trunc((double)((float)b * f) + 0.5)`, the low byte stored. The addition is in double
/// precision.
#[inline(always)]
fn scale_u8(b: u8, f: f32) -> u8 {
    let v = b as f32 * f;
    cvtt64(v as f64 + 0.5) as u8
}

/// The 16 bpc version of [`scale_u8`], the low word stored.
#[inline(always)]
fn scale_u16(b: u16, f: f32) -> u16 {
    let v = b as f32 * f;
    cvtt64(v as f64 + 0.5) as u16
}

/// Upper bound of the iterations of `clip_to_rect`: a legitimate clip settles within a handful of
/// passes. Inputs that never settle stop at the bound.
const CLIP_LIMIT: u32 = 4096;

/// Upper bound of the additions and subtractions of the 360 wrap loops, in total per pixel. A flow
/// of 100 000 image widths still wraps; beyond that (or non finite) the loops would run for
/// minutes per pixel or forever, so the call fails with `Unsupported`.
const WRAP_LIMIT: u32 = 1 << 16;

/// `clip_to_rect(x, y, ix, iy, w1, h1)`: slides the sample point `(ix, iy)` along the segment
/// from the output pixel `(x, y)` until it lies on the rectangle `[0, w1] x [0, h1]`. Operation
/// order of every branch: `t = (a - bound) / (a - cur)`, `t * (b - cur_other)`, `b - that`, all
/// single precision. A NaN coordinate falls through every test untouched. Returns false only when
/// the bound of the iterations was hit (never for finite data).
pub(crate) fn clip_to_rect(x: f32, y: f32, ix: &mut f32, iy: &mut f32, w1: f32, h1: f32) -> bool {
    for _ in 0..CLIP_LIMIT {
        if *ix < 0.0 {
            let dy = y - *iy;
            let dxs = x - *ix;
            let num = x - 0.0;
            *ix = 0.0;
            *iy = y - (num / dxs) * dy;
            continue;
        }
        if *ix > w1 {
            let dy = y - *iy;
            let dxs = x - *ix;
            let num = x - w1;
            *ix = w1;
            *iy = y - (num / dxs) * dy;
            continue;
        }
        if *iy < 0.0 {
            let dyy = y - *iy;
            let num = y - 0.0;
            let dx = x - *ix;
            *ix = x - (num / dyy) * dx;
            *iy = 0.0;
            continue;
        }
        if *iy > h1 {
            let dyy = y - *iy;
            let num = y - h1;
            let dx = x - *ix;
            *ix = x - (num / dyy) * dx;
            *iy = h1;
            continue;
        }
        return true;
    }
    false
}

// ---------------------------------------------------------------------------------------
// Pixel depth abstraction and the samplers
// ---------------------------------------------------------------------------------------

/// One sample type of the engine images with the rules that differ per depth.
pub(crate) trait Pix: Copy + Send + Sync + 'static {
    const ZERO: Self;
    /// Sample to float (`cvtdq2ps` of the integer, exact; identity for float).
    fn to_f32(self) -> f32;
    /// Store of an in range sample (u8, u16: saturating `trunc(v + 0.5)`; f32: raw).
    fn store(v: f32) -> Self;
    /// Store of a faded edge sample with weight `f` (f32: `v * f`; integers: the sample is
    /// stored first and the stored value is scaled with the double precision rounding).
    fn store_faded(v: f32, f: f32) -> Self;
    /// Store of the motion blur taps: f32 raw; integers `trunc32(v + 0.5)` converted back to float,
    /// clamped to `[0, white]`, the indefinite integer of a NaN or out of range value therefore
    /// giving 0 (not the maximum, unlike the saturation of [`sat_u8`] and [`sat_u16`]).
    fn store_blur(v: f32) -> Self;
}

impl Pix for f32 {
    const ZERO: f32 = 0.0;
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline(always)]
    fn store(v: f32) -> f32 {
        v
    }
    #[inline(always)]
    fn store_faded(v: f32, f: f32) -> f32 {
        v * f
    }
    #[inline(always)]
    fn store_blur(v: f32) -> f32 {
        v
    }
}

impl Pix for u8 {
    const ZERO: u8 = 0;
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn store(v: f32) -> u8 {
        sat_u8(v)
    }
    #[inline(always)]
    fn store_faded(v: f32, f: f32) -> u8 {
        scale_u8(sat_u8(v), f)
    }
    #[inline(always)]
    fn store_blur(v: f32) -> u8 {
        (cvtt32(v + 0.5f32) as f32).max(0.0).min(255.0) as u8
    }
}

impl Pix for u16 {
    const ZERO: u16 = 0;
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn store(v: f32) -> u16 {
        sat_u16(v)
    }
    #[inline(always)]
    fn store_faded(v: f32, f: f32) -> u16 {
        scale_u16(sat_u16(v), f)
    }
    #[inline(always)]
    fn store_blur(v: f32) -> u16 {
        (cvtt32(v + 0.5f32) as f32).max(0.0).min(65535.0) as u16
    }
}

/// Bilinear sampler of a four channel image, the same arithmetic for all three depths (only the
/// loads differ).
///
/// `w1` and `h1` are `width - 1` and `height - 1`. Truncating conversion, `fx = x - (float)ix`;
/// when `ix >= w1` the tap column becomes `w1 - 1` with `fx = 1.0` (so column `w1` is read through
/// the formula, which for float data is `(p10 - p00) * 1 + p00`, not exactly `p10`); same for
/// rows. Then per channel, in single precision: `top = (p10 - p00) * fx + p00`,
/// `bot = (p11 - p01) * fx + p01`, `out = (bot - top) * fy + top`. There is no lower clamp:
/// negative coordinates are never passed (callers clip first); they would index out of bounds and
/// panic here.
#[inline(always)]
pub(crate) fn bilerp4<P: Pix>(src: &[P], w: usize, w1: i32, h1: i32, x: f32, y: f32) -> [f32; 4] {
    let mut ix = cvtt32(x);
    let mut iy = cvtt32(y);
    let mut fx = x - ix as f32;
    let mut fy = y - iy as f32;
    if ix >= w1 {
        ix = w1 - 1;
        fx = 1.0;
    }
    if iy >= h1 {
        iy = h1 - 1;
        fy = 1.0;
    }
    let i00 = (iy as usize * w + ix as usize) * CHANNELS;
    let i01 = i00 + w * CHANNELS;
    mix4(
        &src[i00..i00 + CHANNELS],
        &src[i00 + CHANNELS..i00 + 2 * CHANNELS],
        &src[i01..i01 + CHANNELS],
        &src[i01 + CHANNELS..i01 + 2 * CHANNELS],
        fx,
        fy,
    )
}

/// The three lerps shared by the plain and the 360 samplers: `a` = (ix, iy), `b` = (ix + 1,
/// iy), `c` = (ix, iy + 1), `d` = (ix + 1, iy + 1).
#[inline(always)]
fn mix4<P: Pix>(a: &[P], b: &[P], c: &[P], d: &[P], fx: f32, fy: f32) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    for k in 0..CHANNELS {
        let p00 = a[k].to_f32();
        let p10 = b[k].to_f32();
        let p01 = c[k].to_f32();
        let p11 = d[k].to_f32();
        let top = (p10 - p00) * fx + p00;
        let bot = (p11 - p01) * fx + p01;
        out[k] = (bot - top) * fy + top;
    }
    out
}

/// The 360 degree bilinear sampler, the same control flow for all three depths. The coordinates
/// are already wrapped into `[0, W) x [0, H)`. The right neighbour wraps to column 0 when
/// `ix + 1 > w1`. When `iy + 1 > h1` both lower taps are taken from the last row `h1`, shifted by
/// half the width: columns `(W / 2 + ix) % W` and `(W / 2 + x1) % W` (`W / 2` is the signed
/// halving). The weights are those of [`bilerp4`] without any clamp.
///
/// A coordinate that rounds up to exactly `H` (`H - iy` of a tiny `iy`) would read one row past
/// the image, so the row is clamped to `h1`.
#[inline(always)]
fn bilerp360<P: Pix>(src: &[P], w: usize, w1: i32, h1: i32, x: f32, y: f32) -> [f32; 4] {
    let ix = cvtt32(x);
    let iy = cvtt32(y);
    let fx = x - ix as f32;
    let fy = y - iy as f32;
    let x1 = if ix + 1 > w1 { 0 } else { ix + 1 };
    let (mut xc, mut xd, yc);
    if iy + 1 > h1 {
        let half = (w as i32) / 2;
        xc = (half + ix) % w as i32;
        xd = (half + x1) % w as i32;
        yc = h1;
    } else {
        xc = ix;
        xd = x1;
        yc = iy + 1;
    }
    let ya = iy.min(h1);
    xc = xc.max(0);
    xd = xd.max(0);
    let at = |xx: i32, yy: i32| -> &[P] {
        let i = (yy as usize * w + xx as usize) * CHANNELS;
        &src[i..i + CHANNELS]
    };
    mix4(at(ix, ya), at(x1, ya), at(xc, yc), at(xd, yc), fx, fy)
}

/// A borrowed two channel flow field with the constants of the flow sampler.
#[derive(Clone, Copy)]
pub(crate) struct Flow<'a> {
    data: &'a [f32],
    w: usize,
    w1: i32,
    h1: i32,
}

impl<'a> Flow<'a> {
    pub(crate) fn new(f: &'a FlowField) -> Flow<'a> {
        Flow { data: &f.data, w: f.width, w1: f.width as i32 - 1, h1: f.height as i32 - 1 }
    }

    /// The two channel bilinear sampler: the same truncation, the same upper clamp with weight 1.0
    /// and the same `((b - a) * f + a)` chain as [`bilerp4`], on `(dx, dy)` pairs.
    #[inline(always)]
    pub(crate) fn bilerp(&self, x: f32, y: f32) -> [f32; 2] {
        let mut ix = cvtt32(x);
        let mut iy = cvtt32(y);
        let mut fx = x - ix as f32;
        let mut fy = y - iy as f32;
        if ix >= self.w1 {
            ix = self.w1 - 1;
            fx = 1.0;
        }
        if iy >= self.h1 {
            iy = self.h1 - 1;
            fy = 1.0;
        }
        let i00 = (iy as usize * self.w + ix as usize) * 2;
        let i01 = i00 + self.w * 2;
        let d = self.data;
        let mut out = [0.0f32; 2];
        for k in 0..2 {
            let p00 = d[i00 + k];
            let p10 = d[i00 + 2 + k];
            let p01 = d[i01 + k];
            let p11 = d[i01 + 2 + k];
            let top = (p10 - p00) * fx + p00;
            let bot = (p11 - p01) * fx + p01;
            out[k] = (bot - top) * fy + top;
        }
        out
    }
}

// ---------------------------------------------------------------------------------------
// The row kernel
// ---------------------------------------------------------------------------------------

/// Everything the row kernel derives once before the first row.
struct Kernel<'a, P> {
    src: &'a [P],
    flow: Flow<'a>,
    w: usize,
    /// `(float)W`, `(float)H`, `(float)w1`, `(float)h1`.
    wf: f32,
    hf: f32,
    w1: f32,
    h1: f32,
    w1i: i32,
    h1i: i32,
    /// `rx = flow.w1 / src.w1`, `ry = flow.h1 / src.h1` (f32 divisions).
    rx: f32,
    ry: f32,
    /// `sx = amount / rx`, `sy = amount / ry`.
    sx: f32,
    sy: f32,
    hold: bool,
    is360: bool,
    /// Set when a sample coordinate is NaN in hold mode, or a bounded loop hit its limit: states
    /// that would read outside the image or never terminate.
    bad: AtomicBool,
}

impl<'a, P: Pix> Kernel<'a, P> {
    /// One output row. `cov` is the row of the coverage image when one is requested (the
    /// hold and 360 modes leave the pre-filled 255 untouched).
    fn row(&self, y: usize, row: &mut [P], mut cov: Option<&mut [u8]>) {
        let yf = y as f32;
        let fy = yf * self.ry;
        for x in 0..self.w {
            let xf = x as f32;
            let [dx, dy] = self.flow.bilerp(xf * self.rx, fy);
            let ix = dx * self.sx + xf;
            let iy = dy * self.sy + yf;
            let px = &mut row[x * CHANNELS..(x + 1) * CHANNELS];
            if self.is360 {
                self.pixel_360(xf, yf, ix, iy, px);
            } else if self.hold {
                self.pixel_hold(xf, yf, ix, iy, px);
            } else {
                let c = self.pixel_fade(xf, yf, ix, iy, px);
                if let Some(cov) = cov.as_deref_mut() {
                    cov[x] = c;
                }
            }
        }
    }

    /// `hold` set: clip, then sample. A NaN coordinate would read at a wild address; it is reported
    /// through `bad` instead.
    #[inline(always)]
    fn pixel_hold(&self, x: f32, y: f32, mut ix: f32, mut iy: f32, px: &mut [P]) {
        if !clip_to_rect(x, y, &mut ix, &mut iy, self.w1, self.h1) || ix.is_nan() || iy.is_nan() {
            self.bad.store(true, Ordering::Relaxed);
            px.fill(P::ZERO);
            return;
        }
        let s = bilerp4(self.src, self.w, self.w1i, self.h1i, ix, iy);
        for k in 0..CHANNELS {
            px[k] = P::store(s[k]);
        }
    }

    /// `hold` clear: transparent beyond one pixel, faded in the border band, plain inside. Returns
    /// the coverage byte.
    #[inline(always)]
    fn pixel_fade(&self, x: f32, y: f32, mut ix: f32, mut iy: f32, px: &mut [P]) -> u8 {
        // `ix <= -1 || W <= ix || iy <= -1 || H <= iy || isnan(ix) || isnan(iy)`; written as the
        // negation of a conjunction so that a NaN takes this branch.
        if !(ix > -1.0 && ix < self.wf && iy > -1.0 && iy < self.hf) {
            px.fill(P::ZERO);
            return 0;
        }
        if ix >= 0.0 && ix <= self.w1 && iy >= 0.0 && iy <= self.h1 {
            let s = bilerp4(self.src, self.w, self.w1i, self.h1i, ix, iy);
            for k in 0..CHANNELS {
                px[k] = P::store(s[k]);
            }
            return 255;
        }
        let mut f = if ix < 0.0 {
            ix + 1.0
        } else if ix <= self.w1 {
            1.0
        } else {
            self.wf - ix
        };
        if iy < 0.0 {
            f *= iy + 1.0;
        } else if iy > self.h1 {
            f *= self.hf - iy;
        }
        if !clip_to_rect(x, y, &mut ix, &mut iy, self.w1, self.h1) || ix.is_nan() || iy.is_nan() {
            // cannot happen for finite data; the clip would loop forever or the sample would read
            // outside the image
            self.bad.store(true, Ordering::Relaxed);
            px.fill(P::ZERO);
            return 0;
        }
        let s = bilerp4(self.src, self.w, self.w1i, self.h1i, ix, iy);
        for k in 0..CHANNELS {
            px[k] = P::store_faded(s[k], f);
        }
        // coverage: `f * 255 + 0.5` in single precision, `> 255` gives 255, else the truncation.
        let c = f * 255.0 + 0.5;
        if c <= 255.0 {
            cvtt64(c as f64) as u8
        } else {
            255
        }
    }

    /// `is360` set: wrap, pole reflection, 360 sampler.
    #[inline(always)]
    fn pixel_360(&self, _x: f32, _y: f32, ix0: f32, iy0: f32, px: &mut [P]) {
        let (wf, hf) = (self.wf, self.hf);
        let (mut ix, mut iy) = (ix0, iy0);
        let mut guard = 0u32;
        while ix < 0.0 {
            ix += wf;
            guard += 1;
            if guard > WRAP_LIMIT {
                break;
            }
        }
        while wf <= ix {
            ix -= wf;
            guard += 1;
            if guard > WRAP_LIMIT {
                break;
            }
        }
        let mut n = 0u32;
        while iy < 0.0 {
            iy += hf;
            n += 1;
            guard += 1;
            if guard > WRAP_LIMIT {
                break;
            }
        }
        while hf <= iy {
            iy -= hf;
            n += 1;
            guard += 1;
            if guard > WRAP_LIMIT {
                break;
            }
        }
        if guard > WRAP_LIMIT || !(ix >= 0.0 && ix < wf) || iy.is_nan() {
            // non finite or astronomically large flow: the wrap loops would not terminate or the
            // sample would read out of range
            self.bad.store(true, Ordering::Relaxed);
            px.fill(P::ZERO);
            return;
        }
        if n & 1 != 0 {
            // all three depths shift by half the width
            ix = wf * 0.5 + ix;
            if ix >= wf {
                ix -= wf;
            }
            if iy != 0.0 {
                iy = hf - iy;
            }
        }
        let s = bilerp360(self.src, self.w, self.w1i, self.h1i, ix, iy);
        for k in 0..CHANNELS {
            px[k] = P::store(s[k]);
        }
    }
}

fn run<P: Pix>(
    out: &mut [P],
    cov: Option<&mut [u8]>,
    src: &[P],
    flow: &FlowField,
    amount: f32,
    opts: WarpOpts,
    width: usize,
    height: usize,
) -> bool {
    let w1i = width as i32 - 1;
    let h1i = height as i32 - 1;
    let rx = (flow.width as i32 - 1) as f32 / w1i as f32;
    let ry = (flow.height as i32 - 1) as f32 / h1i as f32;
    let k = Kernel {
        src,
        flow: Flow::new(flow),
        w: width,
        wf: width as f32,
        hf: height as f32,
        w1: w1i as f32,
        h1: h1i as f32,
        w1i,
        h1i,
        rx,
        ry,
        sx: amount / rx,
        sy: amount / ry,
        hold: opts.hold,
        is360: opts.is360,
        bad: AtomicBool::new(false),
    };
    let rowlen = width * CHANNELS;
    match cov {
        None => par_rows(out, rowlen, |y, row| k.row(y, row, None)),
        Some(c) => out
            .par_chunks_mut(rowlen)
            .zip(c.par_chunks_mut(width))
            .enumerate()
            .for_each(|(y, (row, crow))| k.row(y, row, Some(crow))),
    }
    !k.bad.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------------------

/// Inverse warp without a coverage image: `out(p) = src(p + flow(p) * amount)` with bilinear
/// sampling. This is `W(img, F, k)` of the retime: the flow lives on the grid of the other frame
/// and is used at the output pixel as if it were the flow at the sampled pixel; the flow may have
/// another resolution than the images (it is sampled bilinearly at `(x * rx, y * ry)` and
/// `amount` is divided by `rx`, `ry`).
///
/// Sequence: `|amount| < 0.005` (compared in double) or `|amount / rx| < 0.0001` (the division in
/// double, narrowed to single, then compared in double) copies `src` into `out`. Otherwise the
/// row kernel of the depth runs, with these store rules: 8 and 16 bpc
/// `clamp(trunc(v + 0.5), 0, max)` for in range samples, faded edge samples scaled as stored
/// (`trunc((double)(stored * f) + 0.5)`, low byte or word), float stored raw. `hold` clips the
/// sample point to the image (no fade), otherwise samples that leave the image fade; without a
/// coverage image the fade still applies. `is360` wraps instead: `ix` and `iy` are wrapped into
/// `[0, W) x [0, H)` by repeated addition or subtraction (counting the `iy` wraps `n`), and when
/// `n` is odd the point is shifted by half the width and `iy` is reflected (`H - iy`, unless it
/// is 0), then the 360 sampler is used (the same for all three depths).
///
/// Errors: [`FxError::Mismatch`] when `out` and `src` differ in size or format, or when an image
/// or the flow is smaller than 2x2 (the ratios would be infinite); [`FxError::Unsupported`] for
/// `mode != 1` (mip map sampling) and for a sample coordinate that cannot be handled (NaN in hold
/// mode, a flow that never terminates the 360 wrap loops).
pub fn inv_warp(out: &mut Image, src: &Image, flow: &FlowField, amount: f32, opts: WarpOpts) -> Result<()> {
    inv_warp_impl(out, src, flow, amount, opts, None)
}

/// Inverse warp that also returns the uint8 coverage image (warp code 1, Smart Blend, uses it
/// with `hold = false`).
///
/// Coverage per output pixel: 255 inside the image, 0 when the sample is one pixel or more
/// outside (the pixel is transparent black), and in the border band `c = f * 255 + 0.5`
/// (single precision) with `c <= 255` giving the truncated byte and anything above 255
/// giving 255, `f` being the fade weight. With `hold` or `is360` the whole image is 255
/// (the coverage starts filled with 0xff) and so it is after the `|amount|` early
/// outs. Everything else is as in [`inv_warp`].
pub fn inv_warp_coverage(
    out: &mut Image,
    src: &Image,
    flow: &FlowField,
    amount: f32,
    opts: WarpOpts,
) -> Result<Coverage> {
    let mut cov = Coverage::new(out.width, out.height, 0xff);
    inv_warp_impl(out, src, flow, amount, opts, Some(&mut cov.data))?;
    Ok(cov)
}

fn inv_warp_impl(
    out: &mut Image,
    src: &Image,
    flow: &FlowField,
    amount: f32,
    opts: WarpOpts,
    cov: Option<&mut [u8]>,
) -> Result<()> {
    // argument check: same size, type and channels; the flow is a two channel float image
    if !out.same_shape(src) {
        return Err(FxError::Mismatch(format!(
            "inverse warp: destination {}x{} {:?} against source {}x{} {:?}",
            out.width,
            out.height,
            out.fmt(),
            src.width,
            src.height,
            src.fmt()
        )));
    }
    if out.width < 2 || out.height < 2 || flow.width < 2 || flow.height < 2 {
        return Err(FxError::Mismatch(format!(
            "inverse warp needs images and a flow of at least 2x2 (image {}x{}, flow {}x{})",
            out.width, out.height, flow.width, flow.height
        )));
    }
    if flow.data.len() != flow.width * flow.height * 2 {
        return Err(FxError::Mismatch("inverse warp: flow data length does not match its size".into()));
    }
    let (width, height) = (out.width, out.height);

    // early outs: the amount test is a double compare of |amount| against 0.005, the ratio test
    // divides in double and narrows
    let a = if amount < 0.0 { -amount } else { amount };
    let mut copy_only = 0.005f64 > a as f64;
    if !copy_only {
        let r = (flow.width as i32 - 1) as f32 / (width as i32 - 1) as f32;
        let mut q = ((amount as f64) / (r as f64)) as f32;
        if q < 0.0 {
            q = -q;
        }
        copy_only = 0.0001f64 > q as f64;
    }
    if copy_only {
        crate::blend::copy(out, src);
        return Ok(());
    }
    if opts.mode != 1 {
        return Err(FxError::Unsupported(
            "inverse warp with mip map sampling (mode != 1) is not implemented"
                .into(),
        ));
    }

    let ok = match (&mut out.samples, &src.samples) {
        (Samples::U8(o), Samples::U8(s)) => run::<u8>(o, cov, s, flow, amount, opts, width, height),
        (Samples::U16(o), Samples::U16(s)) => run::<u16>(o, cov, s, flow, amount, opts, width, height),
        (Samples::F32(o), Samples::F32(s)) => run::<f32>(o, cov, s, flow, amount, opts, width, height),
        _ => unreachable!("shapes were checked"),
    };
    if ok {
        Ok(())
    } else {
        Err(FxError::Unsupported(
            "inverse warp: a sample coordinate is NaN or never converges, which is not supported"
                .into(),
        ))
    }
}
