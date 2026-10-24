//! Forward warp of the engine and the warp code 2 and 3 compositions.
//!
//! # What the forward warp does
//!
//! 1. Size, type and channel checks. When `|amount|` is below `1e-5` (a double precision compare
//!    of the widened float) or the flow is all zero (every `|dx|` and `|dy|` widened and compared
//!    with the double `0.0001`; NaN counts as zero), the result is a plain copy of `src` and a
//!    coverage of 255.
//! 2. A float accumulator (4 channels, the size of `dest`, cleared) and a one channel u8 hit
//!    counter (cleared). A mip pyramid of `src` is built (see [`pyramid`]).
//! 3. The mesh render covers the whole image: rows `0..h-1` and columns `0..w-1` (cells), in
//!    ascending sequential order, so the accumulation order, and with it the last bits, does not
//!    depend on the thread count.
//!    * vertices `[x', y']` ([`mesh::position_row`]), `u = c / (w - 1)`, `v = r / (h - 1)` and the
//!      mip level `d` ([`mesh::mip_level`], base 0);
//!    * per cell `T1 = (v[r][c], v[r][c+1], v[r+1][c])` then
//!      `T2 = (v[r+1][c+1], v[r][c+1], v[r+1][c])`, each scan converted by [`raster::triangle`];
//!    * every covered pixel samples the pyramid ([`pyramid::Pyramid::sample`]) and updates the
//!      running mean: `n == 0`: `acc = c + 0.0`, `n == 255`: `(acc * 255 + c) * 0.00390625 + 0.0`,
//!      else `((n * acc + c) * r) + 0.0` with `r = (float)(1.0 / ((double)n + 1.0))` (the
//!      reciprocal is a double precision division, narrowed), `n + 1` stored.
//! 4. Finish: coverage `cnt != 0 ? 255 : 0`; `dest = convert(acc)`: float `acc + 0.0`, u8
//!    `(u8)(i64)(acc + 0.5f)`, u16 `(u16)(i64)(acc + 0.5f)` (truncation of the biased value to
//!    64 bits, the low byte or word stored, no saturation).
//!
//! Textures are read in the native units of the depth (0..255, 0..65535, 0..1) and the sums are
//! float, so the integer depths round once at the very end. The Gouraud variant
//! (`pyramid = false`) and a supersampled path are not supported.

mod mesh;
mod pyramid;
mod raster;

#[cfg(test)]
mod tests;

use crate::blend::copy;
use crate::flow::FlowField;
use crate::holefill::fill_holes;
use crate::image::{Image, Samples, CHANNELS};
use crate::smartblend::{blend_tail, weighted_tail, SmartOpts};
use crate::warp::Coverage;
use fx_core::{FxError, Result};
use raster::{Sink, Vtx};

/// Options of the forward warp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForwardOpts {
    /// Build a mip pyramid of the source and rasterise textured triangles. The compositor always
    /// sets it; `false` selects the Gouraud rasteriser, which is not supported
    /// ([`FxError::Unsupported`]).
    pub pyramid: bool,
}

impl Default for ForwardOpts {
    /// The call the compositor makes: pyramid on.
    fn default() -> Self {
        ForwardOpts { pyramid: true }
    }
}

/// Forward warp as the compositor calls it, without a coverage image (warp code 3). `out` is
/// overwritten; it needs the size and format of `src`. `flow` may have any size of at least 2x2
/// and is resampled bilinearly like the inverse warp does.
pub fn forward_warp(out: &mut Image, src: &Image, flow: &FlowField, amount: f32) -> Result<()> {
    forward_warp_with(out, src, flow, amount, ForwardOpts::default(), false).map(|_| ())
}

/// The same with the coverage image requested (warp code 2): 255 where at least one triangle
/// covered the pixel, 0 elsewhere (no intermediate values).
pub fn forward_warp_coverage(
    out: &mut Image,
    src: &Image,
    flow: &FlowField,
    amount: f32,
) -> Result<Coverage> {
    match forward_warp_with(out, src, flow, amount, ForwardOpts::default(), true)? {
        Some(c) => Ok(c),
        None => Err(FxError::Unsupported("forward warp: coverage was not produced".into())),
    }
}

/// The accumulate step: samples the pyramid at the pixel's texture coordinates and folds the
/// colour into the running mean with the u8 hit counter.
struct Accum<'a> {
    src: &'a Image,
    pyr: std::cell::OnceCell<pyramid::Pyramid>,
    acc: Vec<f32>,
    cnt: Vec<u8>,
    w: usize,
    outside: bool,
}

impl Sink for Accum<'_> {
    #[inline]
    fn hit(&mut self, x: usize, y: usize, u: f32, v: f32, d: f32) {
        let src = self.src;
        let c = if 0.0 < d {
            self.pyr.get_or_init(|| pyramid::Pyramid::build(src)).sample(src, u, v, d)
        } else {
            pyramid::tap(src, u, v)
        };
        let i = y * self.w + x;
        let n = self.cnt[i];
        let a = &mut self.acc[i * CHANNELS..i * CHANNELS + CHANNELS];
        if n == 0 {
            for k in 0..CHANNELS {
                a[k] = c[k] + 0.0f32;
            }
            self.cnt[i] = 1;
        } else if n == 0xff {
            for k in 0..CHANNELS {
                a[k] = (a[k] * 255.0f32 + c[k]) * 0.00390625f32 + 0.0f32;
            }
        } else {
            let nf = n as f32;
            // the reciprocal is a double precision division narrowed to single
            let r = (1.0f64 / ((nf as f64) + 1.0f64)) as f32;
            for k in 0..CHANNELS {
                a[k] = (nf * a[k] + c[k]) * r + 0.0f32;
            }
            self.cnt[i] = n + 1;
        }
    }

    fn outside(&mut self) {
        self.outside = true;
    }
}

/// Truncation toward zero; NaN and out of range values give `i64::MIN`.
#[inline(always)]
fn cvtt64(x: f32) -> i64 {
    if x.is_nan() || x >= 9.223372036854775808e18 || x < -9.223372036854775808e18 {
        i64::MIN
    } else {
        x as i64
    }
}

/// The accumulator to destination conversion: float `acc + 0.0`, u8 `(u8)(i64)(acc + 0.5f)`,
/// u16 `(u16)(i64)(acc + 0.5f)`: the low byte or word of the truncated 64 bit integer, no
/// saturation.
fn convert(out: &mut Image, acc: &[f32]) {
    match &mut out.samples {
        Samples::F32(d) => {
            for (o, &a) in d.iter_mut().zip(acc) {
                *o = a + 0.0f32;
            }
        }
        Samples::U8(d) => {
            for (o, &a) in d.iter_mut().zip(acc) {
                *o = cvtt64(a + 0.5f32) as u8;
            }
        }
        Samples::U16(d) => {
            for (o, &a) in d.iter_mut().zip(acc) {
                *o = cvtt64(a + 0.5f32) as u16;
            }
        }
    }
}

/// True when every flow component is within `0.0001` of zero (compared as doubles; NaN passes
/// the test, since only `0.0001 < |v|` fails it).
fn flow_is_zero(flow: &FlowField) -> bool {
    for &v in &flow.data {
        let a = if v < 0.0 { -v } else { v };
        if 0.0001f64 < (a as f64) {
            return false;
        }
    }
    true
}

/// Full control over the optional arguments; `want_coverage` returns `Some(coverage)`.
///
/// `out` and `src` must have one size (at least 2x2) and format (`FxError::Mismatch` otherwise),
/// `flow` at least 2x2. Errors: `Unsupported` for `opts.pyramid == false` (the Gouraud rasteriser
/// is not implemented) and for the one geometry that would write outside the image (the
/// degenerate last row pass with a sub pixel span right of the image).
pub fn forward_warp_with(
    out: &mut Image,
    src: &Image,
    flow: &FlowField,
    amount: f32,
    opts: ForwardOpts,
    want_coverage: bool,
) -> Result<Option<Coverage>> {
    if !out.same_shape(src) {
        return Err(FxError::Mismatch(format!(
            "forward warp: destination {}x{} {:?} against source {}x{} {:?}",
            out.width,
            out.height,
            out.fmt(),
            src.width,
            src.height,
            src.fmt()
        )));
    }
    if src.width < 2 || src.height < 2 || flow.width < 2 || flow.height < 2 {
        return Err(FxError::Mismatch(format!(
            "forward warp needs images and a flow of at least 2x2 (image {}x{}, flow {}x{})",
            src.width, src.height, flow.width, flow.height
        )));
    }
    if flow.data.len() != flow.width * flow.height * 2 {
        return Err(FxError::Mismatch("forward warp: flow data length does not match its size".into()));
    }
    let (w, h) = (src.width, src.height);
    // early out: `1e-5 > (double)|amount|` or an all zero flow
    let a = if 0.0f32 > amount { -amount } else { amount };
    if 1e-05f64 > (a as f64) || flow_is_zero(flow) {
        copy(out, src);
        return Ok(want_coverage.then(|| Coverage::new(w, h, 0xff)));
    }
    if !opts.pyramid {
        return Err(FxError::Unsupported(
            "forward warp without a pyramid (the Gouraud rasteriser) is not implemented; \
             every call of the compositor builds the pyramid"
                .into(),
        ));
    }
    let grid = mesh::Grid::new(w, h, flow, amount);
    let pos = mesh::positions(&grid, flow);
    let levels = pyramid::level_count(w, h);
    let mips = if levels > 1 { Some(mesh::mip_levels(&grid, flow)) } else { None };
    let mut sink = Accum {
        src,
        pyr: std::cell::OnceCell::new(),
        acc: vec![0.0f32; w * h * CHANNELS],
        cnt: vec![0u8; w * h],
        w,
        outside: false,
    };
    let (w1, h1) = (w as i32 - 1, h as i32 - 1);
    let (w1f, h1f) = (w1 as f32, h1 as f32);
    let vtx = |r: usize, c: usize| -> Vtx {
        let p = pos[r * w + c];
        Vtx {
            x: p[0],
            y: p[1],
            u: (c as i32 as f32) / w1f,
            v: (r as i32 as f32) / h1f,
            d: match &mips {
                Some(m) => m[r * w + c],
                None => 0.0,
            },
        }
    };
    // rows 0..h-1 and columns 0..w-1 exclusive (the full region of interest), ascending
    for r in 0..h - 1 {
        for c in 0..w - 1 {
            let v00 = vtx(r, c);
            let v01 = vtx(r, c + 1);
            let v10 = vtx(r + 1, c);
            let v11 = vtx(r + 1, c + 1);
            raster::triangle(&[v00, v01, v10], w1, h1, &mut sink);
            raster::triangle(&[v11, v01, v10], w1, h1, &mut sink);
        }
    }
    if sink.outside {
        return Err(FxError::Unsupported(
            "forward warp: the degenerate last row pass of the rasteriser would write outside the \
             image, which is not supported"
                .into(),
        ));
    }
    convert(out, &sink.acc);
    Ok(want_coverage.then(|| {
        let mut cov = Coverage::new(w, h, 0);
        for (c, &n) in cov.data.iter_mut().zip(&sink.cnt) {
            *c = if n != 0 { 0xff } else { 0 };
        }
        cov
    }))
}

// ------------------------------------------------------------------------------------------------
// warp codes 2 and 3 of ENGINE-timeline 14.2
// ------------------------------------------------------------------------------------------------

/// Nearest, warp code 3 (`FW` is the forward warp without coverage): `wa >= 0.5` (a NaN `wa`
/// takes this branch, like the inverse variant): `out = FW(A, flow, 1 - wa)`, otherwise
/// `out = FW(B, flow, wa)`. `flow` is the ONE flow of the chosen half: `F_AB` when `wa >= 0.5`
/// and `F_BA` otherwise (`compose::vector::nearest_uses_f_ab(wa, 3)` says which one to solve).
/// Holes stay transparent black.
pub fn nearest_forward(out: &mut Image, a: &Image, b: &Image, flow: &FlowField, wa: f32) -> Result<()> {
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(0.5f32 > wa) {
        forward_warp(out, a, flow, 1.0f32 - wa)
    } else {
        forward_warp(out, b, flow, wa)
    }
}

/// Nearest, warp code 2: as [`nearest_forward`] with a coverage image, then `invert` and the
/// hole fill ([`fill_holes`]) with the flag `opts.is360`.
pub fn nearest_forward_fill(
    out: &mut Image,
    a: &Image,
    b: &Image,
    flow: &FlowField,
    wa: f32,
    opts: SmartOpts,
) -> Result<()> {
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    let mut cov = if !(0.5f32 > wa) {
        forward_warp_coverage(out, a, flow, 1.0f32 - wa)?
    } else {
        forward_warp_coverage(out, b, flow, wa)?
    };
    cov.invert();
    fill_holes(out, &cov, opts.is360)
}

/// Blend, warp code 3: `tA = FW(A, F_AB, frac)`, `tB = FW(B, F_BA, wa)`,
/// `out = blend(tA, tB, frac)`, `frac = 1.0f32 - wa`. Note the flow order: the pushed frame uses
/// its OWN flow, `f_ab` with A and `f_ba` with B (the inverse variants in [`crate::smartblend`]
/// take `flow_ba` first).
pub fn blend_forward(
    out: &mut Image,
    a: &Image,
    f_ab: &FlowField,
    b: &Image,
    f_ba: &FlowField,
    wa: f32,
) -> Result<()> {
    let frac = 1.0f32 - wa;
    let mut ta = Image::new(a.width, a.height, a.fmt());
    forward_warp(&mut ta, a, f_ab, frac)?;
    let mut tb = Image::new(b.width, b.height, b.fmt());
    forward_warp(&mut tb, b, f_ba, wa)?;
    blend_tail(out, &ta, &tb, frac, None, SmartOpts::default())
}

/// Blend, warp code 2: `tA = FW(A, F_AB, frac, cov1)`, `tB = FW(B, F_BA, wa, cov2)`, then
/// [`blend_tail`] with the coverage images (Smart Blend or its binary alpha variant per
/// `opts.binary_alpha`, combine, invert, hole fill).
pub fn blend_forward_fill(
    out: &mut Image,
    a: &Image,
    f_ab: &FlowField,
    b: &Image,
    f_ba: &FlowField,
    wa: f32,
    opts: SmartOpts,
) -> Result<()> {
    let frac = 1.0f32 - wa;
    let mut ta = Image::new(a.width, a.height, a.fmt());
    let mut cov1 = forward_warp_coverage(&mut ta, a, f_ab, frac)?;
    let mut tb = Image::new(b.width, b.height, b.fmt());
    let cov2 = forward_warp_coverage(&mut tb, b, f_ba, wa)?;
    blend_tail(out, &ta, &tb, frac, Some((&mut cov1, &cov2)), opts)
}

/// Weighted Blend, warp code 3 (without coverage images): `A' = FW(A, F_AB, frac)`,
/// `B' = FW(B, F_BA, wa)`, then [`weighted_tail`] without coverage (sharpening and
/// `blend(A', B', frac)`).
pub fn weighted_forward(
    out: &mut Image,
    a: &Image,
    f_ab: &FlowField,
    b: &Image,
    f_ba: &FlowField,
    wa: f32,
) -> Result<()> {
    let frac = 1.0f32 - wa;
    let mut ta = Image::new(a.width, a.height, a.fmt());
    forward_warp(&mut ta, a, f_ab, frac)?;
    let mut tb = Image::new(b.width, b.height, b.fmt());
    forward_warp(&mut tb, b, f_ba, wa)?;
    weighted_tail(out, &mut ta, a, &mut tb, b, wa, None, SmartOpts::default())
}

/// Weighted Blend, warp code 2 (with `cov1`, `cov2`): the same two warps with coverage, then
/// [`weighted_tail`] with coverage (sharpening, `smart_blend(A', B', frac)`, combine, invert, hole
/// fill with the flag 0; `opts` is ignored, as [`weighted_tail`] ignores it).
pub fn weighted_forward_fill(
    out: &mut Image,
    a: &Image,
    f_ab: &FlowField,
    b: &Image,
    f_ba: &FlowField,
    wa: f32,
    opts: SmartOpts,
) -> Result<()> {
    let frac = 1.0f32 - wa;
    let mut ta = Image::new(a.width, a.height, a.fmt());
    let mut cov1 = forward_warp_coverage(&mut ta, a, f_ab, frac)?;
    let mut tb = Image::new(b.width, b.height, b.fmt());
    let cov2 = forward_warp_coverage(&mut tb, b, f_ba, wa)?;
    weighted_tail(out, &mut ta, a, &mut tb, b, wa, Some((&mut cov1, &cov2)), opts)
}
