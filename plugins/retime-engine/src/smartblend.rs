//! Smart Blend composites and the compositions of warp code 1 ("inverse warp with Smart Blend").
//!
//! # The two composites
//!
//! * [`smart_blend`]. Per pixel, with `white` = 255, 65535 or 1.0 and `over(a, b)` = `a` over `b`
//!   (both premultiplied): `a.alpha == 0` gives `b`, `a.alpha == white` gives `(a.alpha, a.rgb)`,
//!   else `ia = 1 - a.alpha / white`, `alpha = (white - b.alpha) * (a.alpha / white) + b.alpha`,
//!   `rgb = b.rgb * ia + a.rgb`. Then `c1 = over(P1, P2)`, `c2 = over(P2, P1)`,
//!   `amax = max(P1.alpha, P2.alpha)`, `v = (c2 - c1) * w + c1` on all four values; when
//!   `v.alpha != amax`: with `v.alpha == 0` the pixel becomes the plain truncation of the
//!   colour with the larger alpha (`c2` unless `c1.alpha > c2.alpha`, stored without `+ 0.5`,
//!   without clamp, f32 raw), otherwise `rgb = (v.rgb * amax) / v.alpha` and `alpha = amax`.
//!   Stores: f32 `x + 0.0f`; u8 and u16 `y = x + 0.5f`, `0 > y` gives 0, `y > white` gives
//!   white, else the truncation. All three depths run the same float arithmetic (the integer
//!   kernels convert the integers to f32 and divide the alphas by white); the f32 kernel is
//!   the white = 1.0 case (`x / 1.0` and `1.0 - x` are exact in both). `amax` is `P2.alpha` when
//!   the compare `P1.alpha > P2.alpha` is false (also for NaN).
//! * [`smart_blend_binary`], selected by [`SmartOpts::binary_alpha`] (the "360 Video" checkbox).
//!   A composite for alpha that is exactly `white` or not at all: with
//!   `X = (P2 opaque) ? (white, P2.rgb * (1 - P1.alpha / white) + P1.rgb) : (P1 opaque) ? P1 :
//!   0` and `Y = (P1 opaque) ? (white, P1.rgb * (1 - P2.alpha / white) + P2.rgb) : (P2 opaque)
//!   ? P2 : 0` the pixel is `(Y - X) * w + X` with the same stores as above (so a pixel where
//!   neither alpha is exactly `white` comes out as zero).
//!
//! A shape mismatch (size or format of the three images) is [`FxError::Mismatch`].
//!
//! # The compositions (warp code 1: Frame Interp Nearest, Blend and Weighted Blend)
//!
//! `frac = 1.0f32 - wa` is the position after frame A. The inverse warps run with
//! `hold = false`, mode 1, and return a coverage image each ([`inv_warp_coverage`]); the
//! finish is `combine` (per byte max), invert (`255 - b`) and the hole fill
//! ([`crate::holefill::fill_holes`]) on the composited output.
//!
//! The `is360` flag (the engine's wrap-around flag) reaches the warps and the hole
//! fill. With it the inverse warps wrap and leave their coverage at 255, so the combined holes
//! mask is empty and the hole fill is the identity: a 360 Smart Blend is the 360 warps and the
//! composite. [`smart_finish`] with the flag and a holes mask that does have holes runs the
//! wrap-around hole fill ([`crate::holefill`], parity gap G6).
//!
//! The weighted compositions use the plain composite (never the binary alpha variant) and the
//! hole fill with the flag `false`, and their warps run with `is360 = false`, so they ignore
//! [`SmartOpts`] entirely. The parameter is kept so the six compositions share one calling
//! convention.
//!
//! Exactness: all kernels use single precision arithmetic only, with no fused multiply add and
//! no reciprocal estimates.

use crate::blend::{blend, cvtt};
use crate::nanrule as nr;
use crate::filters::{sharpen_step, sharpen_step_with};
use crate::flow::FlowField;
use crate::holefill::{fill_holes_with, VerticalTaps};
use crate::image::{Image, Samples, CHANNELS};
use crate::warp::{inv_warp, inv_warp_coverage, Coverage, WarpOpts};
use fx_core::{par_rows, FxError, Result};

#[cfg(test)]
mod tests;

/// The core arguments that choose a variant of the Smart Blend compositions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SmartOpts {
    /// The "360 Video" checkbox of plugin 01: selects the binary alpha composite
    /// ([`smart_blend_binary`]) instead of [`smart_blend`] in Blend code 1. Ignored by Nearest and
    /// by the weighted variants.
    pub binary_alpha: bool,
    /// The `is360` argument of the warps and the flag of the hole fill: wrap-around (360) warps,
    /// and the wrap-around hole fill wherever a holes mask has holes. Ignored by the weighted
    /// variants.
    pub is360: bool,
    /// The vertical tap order of the hole fill (`holefill::VerticalTaps`, the engine's order by
    /// default); every composition with a hole fill uses it, the weighted ones included.
    pub taps: VerticalTaps,
}

// ------------------------------------------------------------------------------------------------
// pixel arithmetic shared by the three depths
// ------------------------------------------------------------------------------------------------

/// The depth specific load and stores of the kernels. `store_plain` is the store of a value
/// that went through the main path (`+ 0.0f` for f32, `+ 0.5f` and saturation for the integers),
/// `store_raw` the store of the "alpha is 0" fallback colour (f32 as is, integers the plain
/// truncation of the 64 bit conversion, low byte or word).
pub(crate) trait Px: Copy + Send + Sync {
    const WHITE: f32;
    fn load(self) -> f32;
    fn store_plain(x: f32) -> Self;
    fn store_raw(x: f32) -> Self;
}

impl Px for u8 {
    const WHITE: f32 = 255.0;
    #[inline(always)]
    fn load(self) -> f32 {
        f32::from(self)
    }
    #[inline(always)]
    fn store_plain(x: f32) -> u8 {
        let y = x + 0.5f32;
        if 0.0 > y {
            0
        } else if y > 255.0 {
            255
        } else {
            cvtt(y) as u8
        }
    }
    #[inline(always)]
    fn store_raw(x: f32) -> u8 {
        cvtt(x) as u8
    }
}

impl Px for u16 {
    const WHITE: f32 = 65535.0;
    #[inline(always)]
    fn load(self) -> f32 {
        f32::from(self)
    }
    #[inline(always)]
    fn store_plain(x: f32) -> u16 {
        let y = x + 0.5f32;
        if 0.0 > y {
            0
        } else if y > 65535.0 {
            65535
        } else {
            cvtt(y) as u16
        }
    }
    #[inline(always)]
    fn store_raw(x: f32) -> u16 {
        cvtt(x) as u16
    }
}

impl Px for f32 {
    const WHITE: f32 = 1.0;
    #[inline(always)]
    fn load(self) -> f32 {
        self
    }
    #[inline(always)]
    fn store_plain(x: f32) -> f32 {
        x + 0.0f32
    }
    #[inline(always)]
    fn store_raw(x: f32) -> f32 {
        x
    }
}

/// `a` over `b` (`[alpha, r, g, b]`, premultiplied, alpha in units of `white`): the two `over`
/// blocks of the kernels. The compares are exact equality tests (a NaN alpha is neither 0 nor
/// `white`).
#[inline(always)]
pub(crate) fn over(a: [f32; 4], b: [f32; 4], white: f32) -> [f32; 4] {
    let r = over_with::<nr::Plain>(a, b, white);
    // a NaN in the result: the same arithmetic again with the NaN rule (`crate::nanrule`); numbers are the same bits
    if nr::any_nan(&r) {
        over_with::<nr::Rule>(a, b, white)
    } else {
        r
    }
}

/// [`over`] over a set of operations.
#[inline(always)]
pub(crate) fn over_with<A: nr::Arith>(a: [f32; 4], b: [f32; 4], white: f32) -> [f32; 4] {
    if a[0] == 0.0 {
        return b;
    }
    if a[0] == white {
        return a;
    }
    let ia = A::sub(1.0f32, A::div(a[0], white));
    let alpha = A::add(A::mul(A::sub(white, b[0]), A::div(a[0], white)), b[0]);
    [alpha, A::add(A::mul(b[1], ia), a[1]), A::add(A::mul(b[2], ia), a[2]), A::add(A::mul(b[3], ia), a[3])]
}

/// One pixel of the Smart Blend composite. Returns the four values and whether they take the raw
/// store (the "result alpha is 0" fallback).
/// Runs with the processor's own operations and again with the NaN rule (`crate::nanrule`) when the result holds a NaN.
fn smart_pixel(p1: [f32; 4], p2: [f32; 4], w: f32, white: f32) -> ([f32; 4], bool) {
    let r = smart_pixel_with::<nr::Plain>(p1, p2, w, white);
    if nr::any_nan(&r.0) {
        smart_pixel_with::<nr::Rule>(p1, p2, w, white)
    } else {
        r
    }
}

#[inline(always)]
fn smart_pixel_with<A: nr::Arith>(p1: [f32; 4], p2: [f32; 4], w: f32, white: f32) -> ([f32; 4], bool) {
    let c1 = over_with::<A>(p1, p2, white);
    let c2 = over_with::<A>(p2, p1, white);
    let amax = if p1[0] > p2[0] { p1[0] } else { p2[0] };
    let mut v = [0.0f32; 4];
    for i in 0..4 {
        v[i] = nr::lerp_g::<A>(c1[i], c2[i], w);
    }
    if v[0] == amax {
        return (v, false);
    }
    if v[0] == 0.0 {
        // c2 unless c1.alpha > c2.alpha
        return (if c1[0] > c2[0] { c1 } else { c2 }, true);
    }
    let alpha = v[0];
    v[1] = A::div(A::mul(v[1], amax), alpha);
    v[2] = A::div(A::mul(v[2], amax), alpha);
    v[3] = A::div(A::mul(v[3], amax), alpha);
    v[0] = amax;
    (v, false)
}

/// One pixel of the binary alpha composite (see the module documentation).
fn binary_pixel(p1: [f32; 4], p2: [f32; 4], w: f32, white: f32) -> ([f32; 4], bool) {
    let r = binary_pixel_with::<nr::Plain>(p1, p2, w, white);
    if nr::any_nan(&r.0) {
        binary_pixel_with::<nr::Rule>(p1, p2, w, white)
    } else {
        r
    }
}

#[inline(always)]
fn binary_pixel_with<A: nr::Arith>(p1: [f32; 4], p2: [f32; 4], w: f32, white: f32) -> ([f32; 4], bool) {
    let x: [f32; 4] = if p2[0] == white {
        let ia = A::sub(1.0f32, A::div(p1[0], white));
        [p2[0], A::add(A::mul(p2[1], ia), p1[1]), A::add(A::mul(p2[2], ia), p1[2]), A::add(A::mul(p2[3], ia), p1[3])]
    } else if p1[0] == white {
        p1
    } else {
        [0.0; 4]
    };
    let y: [f32; 4] = if p1[0] == white {
        let ia = A::sub(1.0f32, A::div(p2[0], white));
        [p1[0], A::add(A::mul(p1[1], ia), p2[1]), A::add(A::mul(p1[2], ia), p2[2]), A::add(A::mul(p1[3], ia), p2[3])]
    } else if p2[0] == white {
        p2
    } else {
        [0.0; 4]
    };
    let mut v = [0.0f32; 4];
    for i in 0..4 {
        v[i] = nr::lerp_g::<A>(x[i], y[i], w);
    }
    (v, false)
}

type PixelKernel = fn([f32; 4], [f32; 4], f32, f32) -> ([f32; 4], bool);

fn composite_rows<P: Px>(out: &mut [P], p1: &[P], p2: &[P], row_len: usize, w: f32, kernel: PixelKernel) {
    par_rows(out, row_len, |y, row| {
        let r1 = &p1[y * row_len..(y + 1) * row_len];
        let r2 = &p2[y * row_len..(y + 1) * row_len];
        for x in (0..row_len).step_by(CHANNELS) {
            let a = [r1[x].load(), r1[x + 1].load(), r1[x + 2].load(), r1[x + 3].load()];
            let b = [r2[x].load(), r2[x + 1].load(), r2[x + 2].load(), r2[x + 3].load()];
            let (v, raw) = kernel(a, b, w, P::WHITE);
            for c in 0..CHANNELS {
                row[x + c] = if raw { P::store_raw(v[c]) } else { P::store_plain(v[c]) };
            }
        }
    });
}

fn check_shapes3(what: &str, out: &Image, p1: &Image, p2: &Image) -> Result<()> {
    if out.same_shape(p1) && out.same_shape(p2) {
        Ok(())
    } else {
        Err(FxError::Mismatch(format!(
            "{what}: images of {}x{} {:?}, {}x{} {:?} and {}x{} {:?} must have one shape",
            out.width,
            out.height,
            out.fmt(),
            p1.width,
            p1.height,
            p1.fmt(),
            p2.width,
            p2.height,
            p2.fmt()
        )))
    }
}

fn composite(what: &str, out: &mut Image, p1: &Image, p2: &Image, w: f32, kernel: PixelKernel) -> Result<()> {
    check_shapes3(what, out, p1, p2)?;
    let row_len = out.row_len();
    if row_len == 0 {
        return Ok(());
    }
    match (&mut out.samples, &p1.samples, &p2.samples) {
        (Samples::U8(o), Samples::U8(a), Samples::U8(b)) => composite_rows(o, a, b, row_len, w, kernel),
        (Samples::U16(o), Samples::U16(a), Samples::U16(b)) => composite_rows(o, a, b, row_len, w, kernel),
        (Samples::F32(o), Samples::F32(a), Samples::F32(b)) => composite_rows(o, a, b, row_len, w, kernel),
        _ => unreachable!("shapes were checked"),
    }
    Ok(())
}

/// `out = smart_blend(p1, p2, w)`; see the module documentation for the pixel formula. `w` is
/// `frac`, the weight of the second image, in every call.
pub fn smart_blend(out: &mut Image, p1: &Image, p2: &Image, w: f32) -> Result<()> {
    composite("smart blend", out, p1, p2, w, smart_pixel)
}

/// `out = smart_blend(p1, p2, w)` of the binary alpha variant; see the module documentation.
pub fn smart_blend_binary(out: &mut Image, p1: &Image, p2: &Image, w: f32) -> Result<()> {
    composite("smart blend (binary alpha)", out, p1, p2, w, binary_pixel)
}

/// The variant selected by `opts.binary_alpha`.
fn smart_composite(out: &mut Image, p1: &Image, p2: &Image, w: f32, opts: SmartOpts) -> Result<()> {
    if opts.binary_alpha {
        smart_blend_binary(out, p1, p2, w)
    } else {
        smart_blend(out, p1, p2, w)
    }
}

// ------------------------------------------------------------------------------------------------
// coverage handling and hole fill
// ------------------------------------------------------------------------------------------------

/// The finish shared by every Smart Blend composition: `cov1 = max(cov1, cov2)` per byte, invert
/// (`255 - b`), then the hole fill of the composited output with `is360`. `cov1` is left
/// holding the holes mask (255 = nothing was warped there).
pub fn smart_finish(out: &mut Image, cov1: &mut Coverage, cov2: &Coverage, is360: bool) -> Result<()> {
    smart_finish_with(out, cov1, cov2, is360, VerticalTaps::BelowFirst)
}

/// [`smart_finish`] with the vertical tap order `taps` of the hole fill.
pub fn smart_finish_with(out: &mut Image, cov1: &mut Coverage, cov2: &Coverage, is360: bool, taps: VerticalTaps) -> Result<()> {
    if cov1.width != cov2.width || cov1.height != cov2.height {
        return Err(FxError::Mismatch(format!(
            "coverage images of {}x{} and {}x{}",
            cov1.width, cov1.height, cov2.width, cov2.height
        )));
    }
    cov1.combine_max(cov2);
    cov1.invert();
    fill_holes_with(out, cov1, is360, taps)
}

fn warp_opts(opts: SmartOpts) -> WarpOpts {
    WarpOpts { hold: false, is360: opts.is360, mode: 1 }
}

/// Warp code 1 with Frame Interp Nearest: `wa >= 0.5`: `W(A, flow, 1 - wa)`, otherwise
/// `W(B, flow, wa)`, both with `hold = false` and a coverage image; then invert and hole fill.
/// `flow` is the single flow solved for this half: `F_BA` for `wa >= 0.5`, `F_AB` otherwise.
/// `out` is overwritten.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn nearest_smart(
    out: &mut Image,
    a: &Image,
    b: &Image,
    flow: &FlowField,
    wa: f32,
    opts: SmartOpts,
) -> Result<()> {
    // the A branch is taken unless 0.5 > wa, so a NaN weight takes A
    let mut cov = if !(0.5f32 > wa) {
        inv_warp_coverage(out, a, flow, 1.0f32 - wa, warp_opts(opts))?
    } else {
        inv_warp_coverage(out, b, flow, wa, warp_opts(opts))?
    };
    cov.invert();
    fill_holes_with(out, &cov, opts.is360, opts.taps)
}

/// Warp code 1 with Frame Interp Blend: `tA = W(A, F_BA, frac)`, `tB = W(B, F_AB, wa)` with
/// `hold = false` and coverage, `out = smart_blend(tA, tB, frac)` (the binary alpha variant when
/// `opts.binary_alpha`), then [`smart_finish`]. `frac = 1.0f32 - wa`.
pub fn blend_smart(
    out: &mut Image,
    a: &Image,
    flow_ba: &FlowField,
    b: &Image,
    flow_ab: &FlowField,
    wa: f32,
    opts: SmartOpts,
) -> Result<()> {
    let frac = 1.0f32 - wa;
    let mut ta = Image::new(a.width, a.height, a.fmt());
    let mut cov1 = inv_warp_coverage(&mut ta, a, flow_ba, frac, warp_opts(opts))?;
    let mut tb = Image::new(b.width, b.height, b.fmt());
    let cov2 = inv_warp_coverage(&mut tb, b, flow_ab, wa, warp_opts(opts))?;
    smart_composite(out, &ta, &tb, frac, opts)?;
    smart_finish_with(out, &mut cov1, &cov2, opts.is360, opts.taps)
}

/// Warp code 1 with Frame Interp Weighted Blend: `A' = W(A, F_BA, frac)` and
/// `B' = W(B, F_AB, wa)` with `hold = false`, `is360 = false` and one coverage image each, then
/// [`weighted_tail`] with coverage. The flows are taken in the order of [`blend_smart`].
pub fn weighted_blend_smart(
    out: &mut Image,
    a: &Image,
    flow_ba: &FlowField,
    b: &Image,
    flow_ab: &FlowField,
    wa: f32,
    opts: SmartOpts,
) -> Result<()> {
    let frac = 1.0f32 - wa;
    let wopts = WarpOpts { hold: false, is360: false, mode: 1 };
    let mut ta = Image::new(a.width, a.height, a.fmt());
    let mut cov1 = inv_warp_coverage(&mut ta, a, flow_ba, frac, wopts)?;
    let mut tb = Image::new(b.width, b.height, b.fmt());
    let cov2 = inv_warp_coverage(&mut tb, b, flow_ab, wa, wopts)?;
    weighted_tail(out, &mut ta, a, &mut tb, b, wa, Some((&mut cov1, &cov2)), opts)
}

/// Everything the weighted compositions do after their two warps (the forward variants reuse
/// it). `ta` and `tb` are the warped frames and `a` and `b` the unwarped frames. In order:
/// 1. `ta = sharpen_step(ta, a, frac)` and `tb = sharpen_step(tb, b, wa)` ([`crate::filters`]:
///    difference mask against the unwarped frame, blur 3.0, unsharp lerp, inverted mask, masked
///    composite), `frac = 1.0f32 - wa`;
/// 2. with coverage (`Some((cov1, cov2))`): `out = smart_blend(ta, tb, frac)` (always the non
///    binary kernel here) then [`smart_finish`] with the flag `false`; without coverage (the
///    forward variant with code 3): `out = blend(ta, tb, frac)`.
///
/// Of `opts` only the tap order (`opts.taps`, of the sharpening blurs and of the hole fill) is
/// used: neither a binary alpha selector nor a 360 flag reaches this code.
pub fn weighted_tail(
    out: &mut Image,
    ta: &mut Image,
    a: &Image,
    tb: &mut Image,
    b: &Image,
    wa: f32,
    cov: Option<(&mut Coverage, &Coverage)>,
    opts: SmartOpts,
) -> Result<()> {
    check_shapes3("weighted blend tail", out, ta, tb)?;
    check_shapes3("weighted blend tail", out, a, b)?;
    let frac = 1.0f32 - wa;
    sharpen_step_with(ta, a, frac, opts.taps);
    sharpen_step_with(tb, b, wa, opts.taps);
    match cov {
        Some((cov1, cov2)) => {
            smart_blend(out, ta, tb, frac)?;
            smart_finish_with(out, cov1, cov2, false, opts.taps)
        }
        None => {
            blend(out, ta, tb, frac);
            Ok(())
        }
    }
}

/// What the Blend compositions do after the warps (codes 1 and 2 with coverage, codes 0 and 3
/// without): with `Some((cov1, cov2))` the composite `smart_blend(ta, tb, frac)` (the binary
/// alpha variant when `opts.binary_alpha`) and [`smart_finish`] with `opts.is360`; with `None`
/// the plain [`crate::blend::blend`] of `ta` and `tb` at `frac`.
pub fn blend_tail(
    out: &mut Image,
    ta: &Image,
    tb: &Image,
    frac: f32,
    cov: Option<(&mut Coverage, &Coverage)>,
    opts: SmartOpts,
) -> Result<()> {
    check_shapes3("blend tail", out, ta, tb)?;
    match cov {
        Some((cov1, cov2)) => {
            smart_composite(out, ta, tb, frac, opts)?;
            smart_finish_with(out, cov1, cov2, opts.is360, opts.taps)
        }
        None => {
            blend(out, ta, tb, frac);
            Ok(())
        }
    }
}

// ------------------------------------------------------------------------------------------------
// the flow dissolve helper (modes 5 and 6)
// ------------------------------------------------------------------------------------------------

/// Flow dissolve, modes 5 and 6: the two Weighted Blend styles that need no Smart Blend. The
/// compositions do not use it; it is provided for completeness. `frac = 1.0f32 - wa`.
///
/// * `mode == 5`: `tA = W(A, flow_ba, frac)`, `tB = W(B, flow_ab, wa)` with `hold = true`, no
///   coverage, then `sharpen_step` on both and `out = blend(tA', tB', frac)`.
/// * `mode == 6`: no flows. `mask = change_mask(A, B)` is built ONCE from the two source
///   frames; `A' = composite(A, sharp(A, frac), invert(mask))` and `B' = composite(B, sharp(B,
///   wa), the same inverted mask)`, `out = blend(A', B', frac)`. (`sharp(X, t)` is the blur 3.0
///   and unsharp lerp of `sharpen_step`; the mask is the difference of the two sources, not of a
///   warp and its source.)
///
/// Without flows only the modes 0, 2 and 6 are valid; any other mode is `FxError::BadParam` and
/// the destination is left alone. `wa` outside `[0.002, 0.998]` forces mode 0 (a copy of A for
/// `wa >= 0.5`, else of B). Modes other than 0, 5 and 6 are `Unsupported`.
pub fn flow_dissolve_inv(
    out: &mut Image,
    a: &Image,
    flow_ab: Option<&FlowField>,
    b: &Image,
    flow_ba: Option<&FlowField>,
    wa: f32,
    mode: u32,
) -> Result<()> {
    check_shapes3("flow dissolve", out, a, b)?;
    if flow_ab.is_none() && flow_ba.is_none() && ((mode & 0xffff_fff9) != 0 || mode == 4) {
        return Err(FxError::BadParam {
            id: "mode".into(),
            reason: "flow dissolve: this mode needs flows".into(),
        });
    }
    let mut mode = mode;
    if f64::from(wa) < 0.002 || 0.998 < f64::from(wa) {
        mode = 0;
    }
    let frac = 1.0f32 - wa;
    match mode {
        0 => {
            // A unless 0.5 > wa (a NaN weight copies A)
            crate::blend::copy(out, if 0.5f32 > wa { b } else { a });
            Ok(())
        }
        5 => {
            let (Some(fab), Some(fba)) = (flow_ab, flow_ba) else {
                return Err(FxError::BadParam {
                    id: "flow".into(),
                    reason: "flow dissolve mode 5 needs both flows".into(),
                });
            };
            let wopts = WarpOpts { hold: true, is360: false, mode: 1 };
            let mut ta = Image::new(a.width, a.height, a.fmt());
            inv_warp(&mut ta, a, fba, frac, wopts)?;
            let mut tb = Image::new(b.width, b.height, b.fmt());
            inv_warp(&mut tb, b, fab, wa, wopts)?;
            sharpen_step(&mut ta, a, frac);
            sharpen_step(&mut tb, b, wa);
            blend(out, &ta, &tb, frac);
            Ok(())
        }
        6 => {
            use crate::filters::{blur, change_mask, composite_masked, invert_mask, sharpen_weight};
            use crate::blend::lerp;
            let mut mask = change_mask(a, b);
            invert_mask(&mut mask);
            let sharpen = |frame: &Image, t: f32| -> Image {
                let mut sharp = frame.clone();
                blur(&mut sharp, 3.0);
                lerp(&mut sharp, frame, sharpen_weight(t));
                sharp
            };
            let mut ta = a.clone();
            let sa = sharpen(&ta, frac);
            composite_masked(&mut ta, &sa, &mask);
            let mut tb = b.clone();
            let sb = sharpen(&tb, wa);
            composite_masked(&mut tb, &sb, &mask);
            blend(out, &ta, &tb, frac);
            Ok(())
        }
        other => Err(FxError::Unsupported(format!(
            "flow dissolve mode {other}: only the cases 0, 5 and 6 are implemented"
        ))),
    }
}
