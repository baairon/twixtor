//! The FG matte layers of the layered paths. A [`Matte`] is one f32 channel at the output size in
//! 0..1. [`extract`] takes it from a fetched matte frame, [`choke`] shrinks or
//! grows it (rule G1), [`effective`] gives every layer's tracking matte (rule G1c),
//! [`over_weights`] the OVER weights of a group of layers (the weights of rule G1d),
//! [`premultiply`] and [`over`] make and compose the layer images, and [`alpha_holes`] gives
//! the holes of a composite.
//!
//! Claim tags: V established, I inferred, U not established (a rule of this project, pinned in
//! the parity gaps register until a reference render grades it).
//!
//! Precision contract: f32 evaluated in the order written, no fused multiply add, no
//! reassociation; rows run in parallel through rayon row chunks only and no float is summed
//! across threads, so every result is the same for any thread count; the layers are visited in a
//! fixed order, FG1 first. In detail:
//! - a matte sample is `v as f32 / white` (255.0, 65535.0 or 1.0), then [`clamp01`];
//! - [`effective`], [`over_weights`] and [`premultiply`] read every matte value through
//!   [`clamp_matte`] (U, ours: a matte outside 0..1 or NaN reaches them only through the API, as
//!   [`extract`] and [`choke`] clamp): the identity on 0..1, so an in range matte gives the bytes
//!   it gave before the clamp existed;
//! - [`choke`]: for a whole `n >= 1`, `r = ceil(n + 1.0)`, `fr = r - (n + 1.0)`, `fr += 0.01`
//!   when `fr < 0.01`, the radius `R = (1.0 - fr) + r`, the blur `conv::blur_gaussian` of the
//!   one channel plane with `R` and its half kernel `k = conv::gaussian_kernel(R)`,
//!   `T = 1.0 - (k[n] + k[n + 1])` and every sample `clamp01((b - T) / (1.0 - T))`; a fractional
//!   amount mixes `m * (1.0 - t) + c * t`, a negative one inverts with `1.0 - m` before and after,
//!   and the result is clamped by [`clamp01`];
//! - the visibility of the layers below layer `k` is `clamp01(inv_k * vis)`, the tracking matte
//!   `m_k * vis` (both factors in 0..1, so the product needs no clamp);
//! - the premultiply is `q(c * m)` of `image.rs` (`trunc((x as f64) + 0.5)`, low byte or word) at
//!   8 and 16 bpc and `c * m` in float;
//! - OVER is `smartblend::over` on the f32 loads and the plain stores of `smartblend::Px`
//!   (`x + 0.5` saturated and truncated at 8 and 16 bpc, `x + 0.0` in float).

use std::borrow::Cow;

use crate::flow::conv::{blur_gaussian, gaussian_kernel};
use crate::image::{q, Image, Samples, A, B, CHANNELS, G, R};
use crate::smartblend::{self, Px};
use crate::vectors::{vector_hole_mask, HoleMask};
use fx_core::{par_rows, FxError, Result};

#[cfg(test)]
mod tests;

/// One channel image in 0..1 at the output size, row major, `width * height` values.
#[derive(Clone, Debug, PartialEq)]
pub struct Matte {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl Matte {
    /// A matte of `width x height` samples, all `v` (Main_BG's matte is the constant 1.0).
    pub fn filled(width: usize, height: usize, v: f32) -> Matte {
        Matte { width, height, data: vec![v; width * height] }
    }
}

/// The clamp of every matte value to 0..1: above 1.0 gives 1.0, above 0.0 the value itself,
/// anything else 0.0 (a NaN too, and `-0.0` becomes `0.0`). U (ours): the original's treatment of
/// a NaN matte sample is not known.
#[inline(always)]
pub fn clamp01(x: f32) -> f32 {
    if x > 1.0 {
        1.0
    } else if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// The clamp of a matte value read by [`effective`], [`over_weights`] and [`premultiply`] (U,
/// ours): above 1.0 gives 1.0, a value of at least 0.0 itself (`-0.0` too, so every value in
/// 0..1 passes bit for bit), anything else 0.0 (a negative value or a NaN). Without it a value
/// above 1.0 wrapped the low byte or word of a premultiplied sample (`1.5 * 200` stored 44).
#[inline(always)]
pub fn clamp_matte(x: f32) -> f32 {
    if x > 1.0 {
        1.0
    } else if x >= 0.0 {
        x
    } else {
        0.0
    }
}

/// The values of `m` through [`clamp_matte`]: borrowed when every value is already in 0..1 (the
/// clamp would change nothing), else a clamped copy.
fn in_range(m: &[f32]) -> Cow<'_, [f32]> {
    if m.iter().all(|&v| (0.0..=1.0).contains(&v)) {
        Cow::Borrowed(m)
    } else {
        Cow::Owned(m.iter().map(|&v| clamp_matte(v)).collect())
    }
}

/// `FxError::Mismatch` unless the matte holds `width * height` values.
fn check_matte(m: &Matte, what: &str) -> Result<()> {
    if m.width.checked_mul(m.height) != Some(m.data.len()) {
        return Err(FxError::Mismatch(format!("{what}: a {}x{} matte holds {} values", m.width, m.height, m.data.len())));
    }
    Ok(())
}

/// The engine channel of the `fgk_matte_chan` popup value: 1 R, 2 G, 3 B, 4 alpha, which
/// are the engine indices 1, 2, 3 and 0. A value outside 1..=4 (the popup has none) reads alpha,
/// the last entry, as the front-ends' other popup maps do.
pub fn matte_channel(chan: i32) -> usize {
    match chan {
        1 => R,
        2 => G,
        3 => B,
        _ => A,
    }
}

/// The matte of one FG layer from its fetched matte frame `img` (the engine image of
/// `EngineHost::matte_frame`, V): the channel the `fgk_matte_chan` popup `chan` names (see
/// [`matte_channel`]) as `v as f32 / white`, then [`clamp01`] (at 8 and 16 bpc the clamp
/// changes nothing, a float sample outside 0..1 is clamped, I). The frame went through the input
/// preparation of the sources (premultiplied by its alpha, 16 bpc in 0..65535, I) before it
/// arrives here.
pub fn extract(img: &Image, chan: i32) -> Matte {
    let c = matte_channel(chan);
    let (w, h) = (img.width, img.height);
    let mut m = Matte::filled(w, h, 0.0);
    if w == 0 || h == 0 {
        return m;
    }
    match &img.samples {
        Samples::U8(s) => par_rows(&mut m.data, w, |y, row| {
            for (x, o) in row.iter_mut().enumerate() {
                *o = clamp01(f32::from(s[(y * w + x) * CHANNELS + c]) / 255.0f32);
            }
        }),
        Samples::U16(s) => par_rows(&mut m.data, w, |y, row| {
            for (x, o) in row.iter_mut().enumerate() {
                *o = clamp01(f32::from(s[(y * w + x) * CHANNELS + c]) / 65535.0f32);
            }
        }),
        Samples::F32(s) => par_rows(&mut m.data, w, |y, row| {
            for (x, o) in row.iter_mut().enumerate() {
                *o = clamp01(s[(y * w + x) * CHANNELS + c] / 1.0f32);
            }
        }),
    }
    m
}

/// The largest shrink [`choke`] runs, in pixels: a larger magnitude acts as this one. U (ours):
/// the shrink sliders of the three effects end at 10, so no control reaches it; it keeps the
/// kernel of the blur finite.
pub const CHOKE_MAX: f32 = 65536.0;

/// `fgk_inv_matte_shrink` on one matte (structure V, threshold and rescale U, rule G1):
/// `amount` pixels of shrink, a negative amount grows the matte. The steps (V):
/// - a negative amount inverts the matte (`m = 1.0 - m`) before and after the operation and
///   shrinks by `-amount`;
/// - with `lo = floor(amount)` and `hi = ceil(amount)`: `lo == hi` shrinks by `lo`; else a copy
///   `c` shrinks by `hi`, the matte by `lo`, and `m = m * (1.0 - t) + c * t` with
///   `t = amount - lo` (the mix is I);
/// - the result is clamped to 0..1 ([`clamp01`]).
///
/// The shrink step by `n` is the identity for `n = 0` and, for `n >= 1`, the blur and the
/// threshold of rule G1 (U), which erode a straight edge by exactly `n` pixels. So a choke of 0
/// is the identity (bit for bit, no clamp either), and so is a NaN or infinite amount (U, ours:
/// no control gives one). The amount is in pixels at the render's resolution
/// (`downsample_ratio_y * value` in the original, ratio 1 here); magnitudes above
/// [`CHOKE_MAX`] act as it. Records gap G1 when it runs.
///
/// Errors: `FxError::Mismatch` for a matte that does not hold `width * height` values, whatever
/// the amount (the matte is left as it is).
pub fn choke(m: &mut Matte, amount: f32) -> Result<()> {
    check_matte(m, "matte choke")?;
    if amount == 0.0 || !amount.is_finite() {
        return Ok(());
    }
    fx_core::gaps::record(&crate::gaps::G1);
    let (w, h) = (m.width, m.height);
    let grow = amount < 0.0;
    let mut a = if grow { -amount } else { amount };
    if a > CHOKE_MAX {
        a = CHOKE_MAX;
    }
    if grow {
        invert(&mut m.data, w);
    }
    let lo = a.floor();
    let hi = a.ceil();
    if lo == hi {
        shrink(&mut m.data, w, h, lo);
    } else {
        let mut c = m.data.clone();
        shrink(&mut m.data, w, h, lo);
        shrink(&mut c, w, h, hi);
        let t = a - lo;
        mix(&mut m.data, &c, w, t);
    }
    if grow {
        invert(&mut m.data, w);
    }
    map(&mut m.data, w, clamp01);
    Ok(())
}

/// One shrink step of [`choke`] on a `w x h` plane by the whole number of pixels `n` (rule G1,
/// U): nothing for `n < 1` (a choke of 0 is the identity); else `r = ceil(n + 1.0)`,
/// `fr = r - (n + 1.0)` (0.01 added when below 0.01, so 0.01 for every whole `n`), the radius
/// `R = (1.0 - fr) + r` (`n + 1.99`, V: the radius), the one channel `conv::blur_gaussian`
/// of radius `R` (the kernel `2^(-(i*i) / (R/3)^2)` normalised, half width `r = n + 1`),
/// then with its half kernel `k`, `T = 1.0 - (k[n] + k[n + 1])` and every sample
/// `b -> clamp01((b - T) / (1.0 - T))`. A straight edge blurs to `T` at the `n`-th pixel inside
/// it, so the `n` pixels next to the edge become 0 (up to the rounding of the blur), the next one
/// `k[n] / (k[n] + k[n + 1])` and the pixels further in about 1.
fn shrink(data: &mut [f32], w: usize, h: usize, n: f32) {
    if !(n >= 1.0) || data.is_empty() {
        return;
    }
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
    blur_gaussian(data, w, h, 1, radius);
    map(data, w, |b| clamp01((b - t) / den));
}

/// `x = f(x)` for every sample of a plane of rows of `w` samples, rows in parallel.
fn map(data: &mut [f32], w: usize, f: impl Fn(f32) -> f32 + Sync) {
    if w == 0 || data.is_empty() {
        return;
    }
    par_rows(data, w, |_, row| {
        for v in row.iter_mut() {
            *v = f(*v);
        }
    });
}

/// `m = 1.0 - m` for every sample.
fn invert(data: &mut [f32], w: usize) {
    map(data, w, |v| 1.0f32 - v);
}

/// `m = m * (1.0 - t) + c * t` for every sample (the mix, I).
fn mix(data: &mut [f32], c: &[f32], w: usize, t: f32) {
    if w == 0 || data.is_empty() {
        return;
    }
    let it = 1.0f32 - t;
    par_rows(data, w, |y, row| {
        let cr = &c[y * w..(y + 1) * w];
        for (v, &cv) in row.iter_mut().zip(cr) {
            *v = *v * it + cv * t;
        }
    });
}

/// `out = a * b` per sample (the tracking matte and the OVER weight of one layer).
fn product(a: &[f32], b: &[f32], w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; a.len()];
    if w == 0 || a.is_empty() {
        return out;
    }
    par_rows(&mut out, w, |y, row| {
        let (ar, br) = (&a[y * w..(y + 1) * w], &b[y * w..(y + 1) * w]);
        for ((o, &x), &v) in row.iter_mut().zip(ar).zip(br) {
            *o = x * v;
        }
    });
    out
}

/// `vis = clamp01(inv * vis)` per sample (`visible = clamp((1 - m) * visible, 0, 1)`).
fn occlude(vis: &mut [f32], inv: &[f32], w: usize) {
    if w == 0 || vis.is_empty() {
        return;
    }
    par_rows(vis, w, |y, row| {
        let ir = &inv[y * w..(y + 1) * w];
        for (v, &i) in row.iter_mut().zip(ir) {
            *v = clamp01(i * *v);
        }
    });
}

/// `FxError::Mismatch` unless every matte is `w x h` and holds `w * h` values.
fn check_same_size(mattes: &[Matte], w: usize, h: usize, what: &str) -> Result<()> {
    for (k, m) in mattes.iter().enumerate() {
        if !(m.width == w && m.height == h && Some(m.data.len()) == w.checked_mul(h)) {
            return Err(FxError::Mismatch(format!(
                "{what}: matte {k} is {}x{} ({} samples), the layers are {w}x{h}",
                m.width,
                m.height,
                m.data.len()
            )));
        }
    }
    Ok(())
}

/// The tracking matte of every layer (rule G1c): `nfg + 1`
/// entries for the `nfg` FG mattes `mattes` (raw, from [`extract`]), FG1 on top. With `vis` all
/// 1.0 at first, for `k` in `0..nfg`: entry `k` is `m_k * vis`; then `inv_k = 1.0 - m_k` is
/// choked by `shrinks[k]` ([`choke`]: the shrink acts on the inverse matte only, in the
/// visibility of the layers below, never on FG k's own tracking or compositing matte) and
/// `vis = clamp01(inv_k * vis)`. Entry `nfg` (Main_BG, matte 1) is the final `vis`, the product
/// of all inverses. Records gap G1c with two or more FG mattes.
///
/// Every matte is `width x height`, the size of the layers, and so is every entry: with no FG
/// matte (`nfg = 0`, the single layer path, which never calls this) the one entry is
/// Main_BG's constant matte 1.0 of that size. The matte values are read through
/// [`clamp_matte`] (an in range matte is used as it is).
///
/// Errors: `FxError::Mismatch` for a `shrinks` of another length than `mattes` or a matte of
/// another size (or holding another number of values).
pub fn effective(mattes: &[Matte], shrinks: &[f32], width: usize, height: usize) -> Result<Vec<Matte>> {
    let nfg = mattes.len();
    if shrinks.len() != nfg {
        return Err(FxError::Mismatch(format!("effective: {} shrinks for {nfg} FG mattes", shrinks.len())));
    }
    let (w, h) = (width, height);
    check_same_size(mattes, w, h, "effective")?;
    let Some(n) = w.checked_mul(h) else {
        return Err(FxError::Mismatch(format!("effective: a {w}x{h} matte is too large")));
    };
    if nfg >= 2 {
        fx_core::gaps::record(&crate::gaps::G1C);
    }
    let mut out = Vec::with_capacity(nfg + 1);
    let mut vis = vec![1.0f32; n];
    for (m, &s) in mattes.iter().zip(shrinks) {
        let md = in_range(&m.data);
        out.push(Matte { width: w, height: h, data: product(&md, &vis, w) });
        let mut inv = Matte { width: w, height: h, data: md.into_owned() };
        invert(&mut inv.data, w);
        choke(&mut inv, s)?;
        occlude(&mut vis, &inv.data, w);
    }
    out.push(Matte { width: w, height: h, data: vis });
    Ok(out)
}

/// The OVER weights of the layers `first..=last` of create-vectors (flow V, merge U; the
/// weights of rule G1d), from the raw mattes `mattes` (the `nfg` FG mattes, no shrink) with the
/// constant matte 1 for Main_BG (index `nfg`), each `w x h`. They are the shares of the layers in
/// the OVER composite of the group alone, FG1 on top (U, ours: the layers outside the group do
/// not occlude it): with `vis` all 1.0 at first, for `k` in `first..=last` the weight is
/// `m_k * vis` (`vis` itself for Main_BG), then `vis = clamp01((1.0 - m_k) * vis)`. Entry `i` is
/// layer `first + i`. A single FG layer gets its raw matte, Main_BG alone the constant 1, and
/// the full group `0..=nfg` the entries of [`effective`] with zero shrinks.
///
/// The matte values are read through [`clamp_matte`] (an in range matte is used as it is).
///
/// Errors: `FxError::Mismatch` unless `first <= last <= nfg` and every matte is `w x h` (holding
/// `w * h` values): weights at a working size against mattes of the output size are refused,
/// the callers pass the output size of their mattes.
pub fn over_weights(mattes: &[Matte], first: usize, last: usize, w: usize, h: usize) -> Result<Vec<Matte>> {
    let nfg = mattes.len();
    if !(first <= last && last <= nfg) {
        return Err(FxError::Mismatch(format!("over_weights: group {first}..={last} of {nfg} FG layers")));
    }
    check_same_size(mattes, w, h, "over_weights")?;
    let Some(n) = w.checked_mul(h) else {
        return Err(FxError::Mismatch(format!("over_weights: a {w}x{h} matte is too large")));
    };
    let mut out = Vec::with_capacity(last - first + 1);
    let mut vis = vec![1.0f32; n];
    for k in first..=last {
        if k == nfg {
            out.push(Matte { width: w, height: h, data: vis });
            break;
        }
        let md = in_range(&mattes[k].data);
        out.push(Matte { width: w, height: h, data: product(&md, &vis, w) });
        if k < last {
            let mut inv = md.into_owned();
            invert(&mut inv, w);
            occlude(&mut vis, &inv, w);
        }
    }
    Ok(out)
}

/// A layer image: every channel of the premultiplied engine image `src`, alpha included,
/// times the matte: `q(c as f32 * m)` stored as the low byte or word at 8 and 16 bpc (the rounding
/// of `image.rs`), `c * m` in float, with every matte value read through [`clamp_matte`]
/// (an in range matte is used as it is, so it gives the bytes it gave before the clamp).
///
/// Errors: `FxError::Mismatch` for a matte of another size than `src` (or holding another number
/// of values).
pub fn premultiply(src: &Image, m: &Matte) -> Result<Image> {
    let (w, h) = (src.width, src.height);
    if !(m.width == w && m.height == h && Some(m.data.len()) == w.checked_mul(h)) {
        return Err(FxError::Mismatch(format!(
            "premultiply: a {}x{} matte ({} values) for an image of {w}x{h}",
            m.width,
            m.height,
            m.data.len()
        )));
    }
    let mut out = Image::new(w, h, src.fmt());
    let row_len = src.row_len();
    if row_len == 0 || h == 0 {
        return Ok(out);
    }
    let md = in_range(&m.data);
    let md = &*md;
    match (&src.samples, &mut out.samples) {
        (Samples::U8(s), Samples::U8(o)) => par_rows(o, row_len, |y, row| {
            for x in 0..w {
                let mv = md[y * w + x];
                let i = (y * w + x) * CHANNELS;
                for c in 0..CHANNELS {
                    row[x * CHANNELS + c] = q(f32::from(s[i + c]) * mv) as u8;
                }
            }
        }),
        (Samples::U16(s), Samples::U16(o)) => par_rows(o, row_len, |y, row| {
            for x in 0..w {
                let mv = md[y * w + x];
                let i = (y * w + x) * CHANNELS;
                for c in 0..CHANNELS {
                    row[x * CHANNELS + c] = q(f32::from(s[i + c]) * mv) as u16;
                }
            }
        }),
        (Samples::F32(s), Samples::F32(o)) => par_rows(o, row_len, |y, row| {
            for x in 0..w {
                let mv = md[y * w + x];
                let i = (y * w + x) * CHANNELS;
                for c in 0..CHANNELS {
                    row[x * CHANNELS + c] = s[i + c] * mv;
                }
            }
        }),
        _ => unreachable!("Image::new follows the format of the source"),
    }
    Ok(out)
}

fn over_rows<P: Px>(dest: &mut [P], src: &[P], row_len: usize) {
    par_rows(dest, row_len, |y, row| {
        let s = &src[y * row_len..(y + 1) * row_len];
        for x in (0..row_len).step_by(CHANNELS) {
            let a = [s[x].load(), s[x + 1].load(), s[x + 2].load(), s[x + 3].load()];
            let b = [row[x].load(), row[x + 1].load(), row[x + 2].load(), row[x + 3].load()];
            let v = smartblend::over(a, b, P::WHITE);
            for c in 0..CHANNELS {
                row[x + c] = P::store_plain(v[c]);
            }
        }
    });
}

/// `src` OVER `dest`, both premultiplied: with `a` the alpha of `src` over white,
/// `dest.rgb = dest.rgb * (1 - a) + src.rgb` and `dest.alpha = dest.alpha + (1 - dest.alpha) * a`,
/// as `smartblend::over(src, dest, white)` (`a.alpha == 0` keeps `dest`, `a.alpha == white` gives
/// `src`, else `ia = 1 - a.alpha / white`, `alpha = (white - b.alpha) * (a.alpha / white) +
/// b.alpha`, `rgb = b.rgb * ia + a.rgb`) and the plain stores (8 and 16 bpc `x + 0.5`, saturated,
/// truncated; float `x + 0.0`). A transparent `src` leaves `dest` as it is (float `-0.0` becomes
/// `0.0`), and onto a transparent `dest` a premultiplied `src` comes out as it is. Images of
/// another size or format are `FxError::Mismatch`.
pub fn over(dest: &mut Image, src: &Image) -> Result<()> {
    if !dest.same_shape(src) {
        return Err(FxError::Mismatch(format!(
            "layer OVER: {}x{} {:?} onto {}x{} {:?}",
            src.width,
            src.height,
            src.fmt(),
            dest.width,
            dest.height,
            dest.fmt()
        )));
    }
    let row_len = dest.row_len();
    if row_len == 0 || dest.height == 0 {
        return Ok(());
    }
    match (&mut dest.samples, &src.samples) {
        (Samples::U8(d), Samples::U8(s)) => over_rows(d, s, row_len),
        (Samples::U16(d), Samples::U16(s)) => over_rows(d, s, row_len),
        (Samples::F32(d), Samples::F32(s)) => over_rows(d, s, row_len),
        _ => unreachable!("shapes were checked"),
    }
    Ok(())
}

/// The holes of a composite for the hole fill (holes derived from the alpha of the
/// composite): `white - alpha` per pixel in the sample type of `img` (`255 - A`, `65535 - A`,
/// `1.0 - A`), the mask `vectors::vector_hole_mask` builds; a partial alpha is a partial hole
/// value, which the hole fill counts as a hole (any non zero u8 or u16 value, `>= 3.1e-5` in
/// float).
pub fn alpha_holes(img: &Image) -> HoleMask {
    vector_hole_mask(img)
}
