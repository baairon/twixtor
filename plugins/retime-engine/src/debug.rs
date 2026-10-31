//! The Debug display of create-vectors (rule G4).
//! `debug_view_method` 1 Warp, 2 Blend, 3 Diff, 4 Detail Compare and 5 Edge Compare are the
//! engine view codes 4, 0, 1, 2 and 3. Every view starts from `W`, the forward warp of the first
//! frame `X` of the solve by its flow (the Warp view); the other four compose `W` with the
//! second frame `Y` (rule G4; the claim tags of each part below):
//!
//! * Warp (4): `W` itself, the forward warp at amount 1.0 with no coverage and no hole fill;
//! * the other four start from `W'`: a transparent image with `Y` laid OVER it, then `W` laid
//!   OVER that, with the plugin's image OVER ([`compose_over`]; READ and TRACED).
//!   Where `W` is opaque that is `W`, where it is transparent (the pixels the warp does not reach,
//!   and the whole frame when the solve has no pair) the pixel of `Y` (V);
//! * Blend (0): `lerp(W', Y, 0.5)` of `crate::blend` (the weight on `W'`, alpha normalised) (V);
//! * Diff (1): grey, alpha white; per pixel the luma difference `|L(W') - L(Y)|` times a gain
//!   of 2.5 (see [`DIFF_AMOUNT`]) with the Diff luma in double precision (V at 8, 16 and 32 bpc);
//! * Detail Compare (2): a red/cyan overlay, red the detail of `W'` and green and blue the detail
//!   of `Y`, alpha white (V); the detail of a frame with the luma `L` of its samples in 0..1 is
//!   `L` itself on the border pixels and `sqrt(S * 0.125)` inside, `S` the sum of `|L(p) - L(n)|`
//!   over the 8 neighbours `n`, then scaled by the overlay gain (V);
//! * Edge Compare (3): the same overlay of the edge measure, the gradient magnitude of the luma
//!   times 2 (V).
//!
//! Where the numbers come from: the rules below were read from the original's code and checked
//! against its own intermediate images under the emulator:
//! the images `W'` and `Y` that the plugin hands to its view generator, and the images it makes of
//! them, reproduce bit for bit at 8, 16 and 32 bpc for the Diff (all 192 pixels of the probe
//! frames), the luma, the Detail and Edge measures, the overlay and the Blend.
//!
//! The original works in single precision with these exceptions, which the rules below keep
//! (operation order and precision are part of the rule):
//! - the luma of the Diff is `(0.29 R + 0.59 G) + 0.12 B` evaluated in double precision on the
//!   samples and narrowed to single;
//! - the luma of the Detail and Edge views is the same in double precision on the samples scaled
//!   to 0..1 (`c * w / 255` for 8 bpc, `/ 65535` for 16 bpc, no division for float), each term
//!   divided separately, then narrowed and capped at 1.0 for 8 and 16 bpc;
//! - the overlay gain and the Diff gain are single precision narrowings of a double quotient;
//! - values are turned into 8 and 16 bpc samples as `trunc(v * 255 + 0.5)` and
//!   `trunc(v * 65535 + 0.5)` evaluated in double precision.
//!
//! The Debug frame roles (Previous: `X` is frame `n + 1`, `Y` frame `n`; Next: `X` is `n - 1`)
//! are the pair of `create_vectors::frame_pair`; with FG layers the flow is the merged one
//! (`create_vectors::job_flow`).
//!
//! Claim tags: V established, I inferred, U not established (a rule of this project, pinned in
//! the parity gaps register until a reference render grades it).
//!
//! Precision contract: f32 in the order written unless a double is named, no fused multiply
//! add, rows in parallel through rayon row chunks only, so the result is the same for any
//! thread count.

use crate::blend::{copy, lerp};
use crate::flow::FlowField;
use crate::image::{Image, Samples, A, B, CHANNELS, G, R};
use crate::prep::{self, Plane};
use fx_core::{par_rows, FxError, Result};

#[cfg(test)]
mod tests;

/// Engine view code of the Blend view.
pub const VIEW_BLEND: i32 = 0;
/// Engine view code of the Diff view.
pub const VIEW_DIFF: i32 = 1;
/// Engine view code of the Detail Compare view.
pub const VIEW_DETAIL: i32 = 2;
/// Engine view code of the Edge Compare view.
pub const VIEW_EDGE: i32 = 3;
/// Engine view code of the Warp view.
pub const VIEW_WARP: i32 = 4;

/// The amount the original passes to its Diff, Detail and Edge views (a constant of its Debug
/// code: 0.6).
pub const VIEW_AMOUNT: f32 = 0.6;
/// The weight on `W'` of the Blend view (0.5).
pub const BLEND_WEIGHT: f32 = 0.5;
/// The Diff passes `1.0 - amount` (a double subtraction narrowed to single) to its kernel.
pub const DIFF_AMOUNT: f32 = (1.0f64 - VIEW_AMOUNT as f64) as f32;

/// The Diff gain: `1.0 / |DIFF_AMOUNT|` in double precision narrowed to single (2.5000002).
pub fn diff_gain() -> f32 {
    (1.0f64 / f64::from(DIFF_AMOUNT.abs())) as f32
}

/// The overlay gain of the Detail and Edge views: the amount is raised to at least 0.001, halved
/// (single), subtracted from 1.0 (double narrowed to single), and the gain is the double quotient
/// `1.0 / that` narrowed to single (1.4285715).
pub fn overlay_gain() -> f32 {
    let amount = if 0.001f64 > f64::from(VIEW_AMOUNT) { 0.001f32 } else { VIEW_AMOUNT };
    let half = (f64::from(amount) * 0.5f64) as f32;
    let rest = (1.0f64 - f64::from(half)) as f32;
    (1.0f64 / f64::from(rest)) as f32
}

/// The debug view of engine view `code` (0 Blend, 1 Diff, 2 Detail Compare, 3 Edge Compare,
/// 4 Warp; any other code is Edge Compare, the last entry, as the popup map of the glue reads it)
/// for the frames `x` and `y` of the solve and its flow `X(p) ~ Y(p + flow(p))` (any size of at
/// least 2x2; with FG layers, the merged flow): an image of the size and format of `x` (see the
/// module documentation for each view). The four compositions record gap G4; Warp, which has
/// no open point, records nothing.
///
/// Tiny frames (U, ours, G7 family): a frame that is not [`crate::prep::searchable`] has no
/// search, and its Debug display shows what the Vectors display shows, the zero vector image
/// (`vectors::zero_vectors_image`); nothing is warped.
///
/// Errors: `Mismatch` for frames of different shapes; the forward warp's (a flow below 2x2, and
/// its one refused geometry).
pub fn debug_view(x: &Image, y: &Image, flow: &FlowField, code: i32) -> Result<Image> {
    debug_view_on(&crate::stages::Cpu, x, y, flow, code)
}

/// [`debug_view`] with the forward warp on `stages`.
pub fn debug_view_on(stages: &dyn crate::stages::Stages, x: &Image, y: &Image, flow: &FlowField, code: i32) -> Result<Image> {
    if !x.same_shape(y) {
        return Err(FxError::Mismatch(format!(
            "debug view: incompatible frames ({}x{} {:?} against {}x{} {:?})",
            x.width,
            x.height,
            x.fmt(),
            y.width,
            y.height,
            y.fmt()
        )));
    }
    let (w, h, fmt) = (x.width, x.height, x.fmt());
    if !prep::searchable(w, h) {
        return Ok(crate::vectors::zero_vectors_image(w, h, fmt));
    }
    let mut warped = Image::new(w, h, fmt);
    stages.forward_warp(&mut warped, x, flow, 1.0f32)?;
    if code == VIEW_WARP {
        return Ok(warped);
    }
    fx_core::gaps::record(&crate::gaps::G4);
    let wp = compose_over(&warped, y)?;
    Ok(views_of(&wp, y, code))
}

/// `W'` of the Debug views without FG mattes: a transparent image of the shape of `y`, `y` laid
/// OVER it, then `w` laid OVER the result, both with the plugin's image OVER
/// ([`crate::matte::over_pl`]). READ in `CreateMotionVectors_64` (the Debug branch of the per frame
/// driver zeroes the output, then calls the OVER with the frame `Y` and with the
/// forward warp `W`, in that order; TRACED bit for bit on the special value clip `specmix`).
/// With alphas of 0 and white only this is `W` where it is opaque and
/// `Y` elsewhere; a fractional, non finite or NaN alpha of either mixes them by the OVER's float
/// or integer arithmetic (an alpha of `W` above 1 can drive the alpha of `W'` to an infinity or a
/// NaN, which the Blend view then clears).
///
/// Errors: `Mismatch` for images of different shapes.
pub fn compose_over(w: &Image, y: &Image) -> Result<Image> {
    let mut out = Image::new(y.width, y.height, y.fmt());
    crate::matte::over_pl(&mut out, y)?;
    crate::matte::over_pl(&mut out, w)?;
    Ok(out)
}

/// The view `code` (Blend, Diff, Detail or Edge; any other code is Edge) of `wp` (the `W'` the
/// view generator gets) against `y` (same shape): the composition part of [`debug_view_on`].
pub fn views_of(wp: &Image, y: &Image, code: i32) -> Image {
    let mut out = Image::new(wp.width, wp.height, wp.fmt());
    match code {
        VIEW_BLEND => {
            copy(&mut out, wp);
            lerp(&mut out, y, BLEND_WEIGHT);
        }
        VIEW_DIFF => diff(&mut out, wp, y),
        VIEW_DETAIL => overlay(&mut out, &detail(&debug_luma(wp)), &detail(&debug_luma(y))),
        _ => overlay(&mut out, &edge(&debug_luma(wp)), &edge(&debug_luma(y))),
    }
    out
}

/// The alpha match the Debug display of a one layer group of an FG stack runs before its Diff,
/// Detail and Edge views (READ in `CreateMotionVectors_64`),
/// per pixel of `a` and `b` (same shape),
/// all single precision: `a1 = alpha(a) / white`, `a2 = alpha(b) / white` (white 255 or 65535; the
/// float alphas as they are); equal alphas change nothing; else when either is 0 both pixels
/// become 0 in every channel; else the pixel of the larger alpha is brought to the smaller one:
/// each colour `trunc((c / a_big) * a_small)` (no truncation in float) and its alpha the other
/// pixel's alpha sample.
pub fn match_alpha(a: &mut Image, b: &mut Image) {
    fn px<T: Copy>(pa: &mut [T], pb: &mut [T], to_f: impl Fn(T) -> f32, from_f: impl Fn(f32) -> T, white: f32, zero: T) {
        let (a1, a2) = (to_f(pa[A]) / white, to_f(pb[A]) / white);
        if a1 == a2 {
            return;
        }
        if a1 == 0.0 || a2 == 0.0 {
            pa.iter_mut().for_each(|v| *v = zero);
            pb.iter_mut().for_each(|v| *v = zero);
            return;
        }
        if a2 > a1 {
            for c in [R, G, B] {
                pb[c] = from_f((to_f(pb[c]) / a2) * a1);
            }
            pb[A] = pa[A];
        } else {
            for c in [R, G, B] {
                pa[c] = from_f((to_f(pa[c]) / a1) * a2);
            }
            pa[A] = pb[A];
        }
    }
    match (&mut a.samples, &mut b.samples) {
        (Samples::U8(sa), Samples::U8(sb)) => sa
            .chunks_exact_mut(CHANNELS)
            .zip(sb.chunks_exact_mut(CHANNELS))
            .for_each(|(pa, pb)| px(pa, pb, f32::from, |v| v as i64 as u8, 255.0, 0)),
        (Samples::U16(sa), Samples::U16(sb)) => sa
            .chunks_exact_mut(CHANNELS)
            .zip(sb.chunks_exact_mut(CHANNELS))
            .for_each(|(pa, pb)| px(pa, pb, f32::from, |v| v as i64 as u16, 65535.0, 0)),
        (Samples::F32(sa), Samples::F32(sb)) => sa
            .chunks_exact_mut(CHANNELS)
            .zip(sb.chunks_exact_mut(CHANNELS))
            .for_each(|(pa, pb)| px(pa, pb, |v| v, |v| v, 1.0, 0.0)),
        _ => {}
    }
}

/// The earlier rule for `W'` (the pixels of `w` with alpha 0 take the pixel of `y`, same shape):
/// equal to [`compose_over`] for alphas of 0 and white only. The CPU Debug views compose with
/// [`compose_over`]; this rule stays for the device transcriptions (`device/vecstage`), which
/// still follow it.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn fill_transparent(w: &mut Image, y: &Image) {
    match (&mut w.samples, &y.samples) {
        (Samples::U8(d), Samples::U8(s)) => fill_where(d, s, |a| a == 0),
        (Samples::U16(d), Samples::U16(s)) => fill_where(d, s, |a| a == 0),
        (Samples::F32(d), Samples::F32(s)) => fill_where(d, s, |a| a == 0.0),
        _ => unreachable!("the shapes were checked"),
    }
}

fn fill_where<T: Copy>(d: &mut [T], s: &[T], empty: impl Fn(T) -> bool) {
    for (dp, sp) in d.chunks_exact_mut(CHANNELS).zip(s.chunks_exact(CHANNELS)) {
        if empty(dp[A]) {
            dp.copy_from_slice(sp);
        }
    }
}

/// The luma of one pixel for the Diff: `(0.29 R + 0.59 G) + 0.12 B` in double precision on the
/// samples (their own scale), narrowed to single.
pub fn diff_luma(r: f64, g: f64, b: f64) -> f32 {
    (((r * 0.29f64) + (g * 0.59f64)) + (b * 0.12f64)) as f32
}

/// The Diff value of two lumas in the sample scale: `|l1 - l2|` (single) times the gain (single).
fn diff_scaled(l1: f32, l2: f32) -> f32 {
    (l1 - l2).abs() * diff_gain()
}

/// The 8 and 16 bpc Diff sample: the scaled value capped at `top` (`v >= top` gives `top`), then
/// `trunc(v + 0.5)` in double precision (the low word of the 64 bit integer; NaN gives 0).
fn diff_int(l1: f32, l2: f32, top: f32) -> i64 {
    let v = diff_scaled(l1, l2);
    let v = if v >= top { top } else { v };
    (f64::from(v) + 0.5f64) as i64
}

/// The float Diff sample: the scaled value capped at 1.0 (`v >= 1` gives 1.0; NaN stays NaN).
fn diff_float(l1: f32, l2: f32) -> f32 {
    let v = diff_scaled(l1, l2);
    if v >= 1.0 {
        1.0
    } else {
        v
    }
}

/// The luma of the Detail and Edge views: per pixel the double precision sum
/// `((R * 0.29 / S) + (G * 0.59 / S)) + (B * 0.12 / S)` narrowed to single and capped at 1.0 for
/// 8 and 16 bpc (`S` the white of the format, 255 or 65535); float: `(R * 0.29 + G * 0.59) + B *
/// 0.12` in double precision narrowed to single, no cap.
pub(crate) fn debug_luma(img: &Image) -> Plane {
    let scaled = |r: f64, g: f64, b: f64, s: f64| -> f32 {
        let v = (((r * 0.29f64) / s + (g * 0.59f64) / s) + (b * 0.12f64) / s) as f32;
        if v >= 1.0 {
            1.0
        } else {
            v
        }
    };
    let data: Vec<f32> = match &img.samples {
        Samples::U8(v) => v.chunks_exact(CHANNELS).map(|p| scaled(f64::from(p[R]), f64::from(p[G]), f64::from(p[B]), 255.0)).collect(),
        Samples::U16(v) => v.chunks_exact(CHANNELS).map(|p| scaled(f64::from(p[R]), f64::from(p[G]), f64::from(p[B]), 65535.0)).collect(),
        Samples::F32(v) => v.chunks_exact(CHANNELS).map(|p| float_luma(p[R], p[G], p[B])).collect(),
    };
    Plane { width: img.width, height: img.height, data }
}

/// The float luma of the Detail and Edge views: `(R * 0.29 + G * 0.59) + B * 0.12` in double precision narrowed to
/// single. A NaN result is computed again with the two sums through `crate::nanrule::add64`: which of two NaN terms
/// the sum returns is the NaN rule's (TRACED in `CreateMotionVectors_64`, the luma of the special value clip
/// `specmix`; the processor's own `addsd` returns whichever operand the compiler made the
/// destination). The products of a NaN by a constant return that NaN quieted under either rule.
pub(crate) fn float_luma(r: f32, g: f32, b: f32) -> f32 {
    let (tr, tg, tb) = (f64::from(r) * 0.29f64, f64::from(g) * 0.59f64, f64::from(b) * 0.12f64);
    let v = ((tr + tg) + tb) as f32;
    if v.is_nan() {
        use crate::nanrule::add64;
        add64(add64(tr, tg), tb) as f32
    } else {
        v
    }
}

/// The detail of a luma plane: the luma itself on the border (the original leaves the border of
/// the plane it measures untouched) and `sqrt(S * 0.125)` inside, `S` the sum of the absolute
/// differences to the 8 neighbours added in row order (rows above to below, columns left to
/// right, the pixel itself skipped), single precision.
pub(crate) fn detail(l: &Plane) -> Vec<f32> {
    let (w, h) = (l.width, l.height);
    let mut out = vec![0.0f32; w * h];
    if w == 0 || out.is_empty() {
        return out;
    }
    par_rows(&mut out, w, |y, row| {
        for (x, v) in row.iter_mut().enumerate() {
            let c = l.data[y * w + x];
            *v = if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
                c
            } else {
                let mut s: Option<f32> = None;
                for yy in y - 1..=y + 1 {
                    for xx in x - 1..=x + 1 {
                        if yy != y || xx != x {
                            let d = (c - l.data[yy * w + xx]).abs();
                            s = Some(match s {
                                None => d,
                                Some(acc) => acc + d,
                            });
                        }
                    }
                }
                (s.unwrap_or(0.0f32) * 0.125f32).sqrt()
            };
        }
    });
    out
}

/// The edge measure of a luma plane: with the border repeated outwards, the horizontal and
/// vertical Sobel sums
/// `gx = ((((-n(-1,-1) + n(1,-1)) - 2 n(-1,0)) + 2 n(1,0)) - n(-1,1)) + n(1,1)` and
/// `gy = ((((2 n(0,-1) + n(-1,-1)) + n(1,-1)) - n(-1,1)) - 2 n(0,1)) - n(1,1)` (`n(dx, dy)` the
/// luma at that offset, rows in the plane's order), `e = f64(sqrt(gy * gy + gx * gx)) *
/// 0.17677669529664` narrowed to single, and the view scales it by 2 (capped at 255).
pub(crate) fn edge(l: &Plane) -> Vec<f32> {
    let (w, h) = (l.width, l.height);
    let mut out = vec![0.0f32; w * h];
    if w == 0 || out.is_empty() {
        return out;
    }
    par_rows(&mut out, w, |y, row| {
        let n = |x: usize, dx: isize, dy: isize| -> f32 {
            let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
            let xx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
            l.data[yy * w + xx]
        };
        for (x, v) in row.iter_mut().enumerate() {
            let gx = ((((-n(x, -1, -1) + n(x, 1, -1)) - 2.0f32 * n(x, -1, 0)) + 2.0f32 * n(x, 1, 0)) - n(x, -1, 1)) + n(x, 1, 1);
            let gy = ((((2.0f32 * n(x, 0, -1) + n(x, -1, -1)) + n(x, 1, -1)) - n(x, -1, 1)) - 2.0f32 * n(x, 0, 1)) - n(x, 1, 1);
            let e = (f64::from((gy * gy + gx * gx).sqrt()) * 0.17677669529664f64) as f32;
            let scaled = 2.0f32 * e;
            *v = if scaled > 255.0f32 { 255.0f32 } else { scaled };
        }
    });
    out
}

/// One overlay value: the measure times the gain (single), clamped to 0..1 (`v > 1` gives 1, else
/// `v < 0` gives 0; NaN stays NaN).
fn overlay_value(m: f32, gain: f32) -> f32 {
    let v = gain * m;
    if v > 1.0f32 {
        1.0f32
    } else if v < 0.0f32 {
        0.0f32
    } else {
        v
    }
}

/// An overlay value as an integer sample: `trunc(v * top + 0.5)` in double precision (the low
/// byte or word of the 64 bit integer, NaN gives 0).
fn overlay_int(v: f32, top: f64) -> i64 {
    (f64::from(v) * top + 0.5f64) as i64
}

/// The red/cyan overlay: alpha white, red the measure `r`, green and blue the measure `g`, each
/// scaled by [`overlay_gain`] and clamped to 0..1, then stored per depth.
pub(crate) fn overlay(out: &mut Image, r: &[f32], g: &[f32]) {
    let w = out.width;
    let row_len = out.row_len();
    if row_len == 0 || out.height == 0 {
        return;
    }
    let gain = overlay_gain();
    match &mut out.samples {
        Samples::U8(o) => par_rows(o, row_len, |y, row| {
            for (x, px) in row.chunks_exact_mut(CHANNELS).enumerate() {
                let i = y * w + x;
                px[A] = 255;
                px[R] = overlay_int(overlay_value(r[i], gain), 255.0) as u8;
                px[G] = overlay_int(overlay_value(g[i], gain), 255.0) as u8;
                px[B] = px[G];
            }
        }),
        Samples::U16(o) => par_rows(o, row_len, |y, row| {
            for (x, px) in row.chunks_exact_mut(CHANNELS).enumerate() {
                let i = y * w + x;
                px[A] = 65535;
                px[R] = overlay_int(overlay_value(r[i], gain), 65535.0) as u16;
                px[G] = overlay_int(overlay_value(g[i], gain), 65535.0) as u16;
                px[B] = px[G];
            }
        }),
        Samples::F32(o) => par_rows(o, row_len, |y, row| {
            for (x, px) in row.chunks_exact_mut(CHANNELS).enumerate() {
                let i = y * w + x;
                px[A] = 1.0;
                px[R] = overlay_value(r[i], gain);
                px[G] = overlay_value(g[i], gain);
                px[B] = px[G];
            }
        }),
    }
}

/// The grey Diff of `W'` (`a`) and `Y` (`b`), alpha white (see the module documentation).
pub(crate) fn diff(out: &mut Image, a: &Image, b: &Image) {
    let row_len = out.row_len();
    if row_len == 0 || out.height == 0 {
        return;
    }
    match (&mut out.samples, &a.samples, &b.samples) {
        (Samples::U8(o), Samples::U8(a), Samples::U8(b)) => par_rows(o, row_len, |y, row| {
            let s = y * row_len;
            for (x, px) in row.chunks_exact_mut(CHANNELS).enumerate() {
                let i = s + x * CHANNELS;
                let l1 = diff_luma(f64::from(a[i + R]), f64::from(a[i + G]), f64::from(a[i + B]));
                let l2 = diff_luma(f64::from(b[i + R]), f64::from(b[i + G]), f64::from(b[i + B]));
                let v = diff_int(l1, l2, 255.0) as u8;
                px[A] = 255;
                px[R] = v;
                px[G] = v;
                px[B] = v;
            }
        }),
        (Samples::U16(o), Samples::U16(a), Samples::U16(b)) => par_rows(o, row_len, |y, row| {
            let s = y * row_len;
            for (x, px) in row.chunks_exact_mut(CHANNELS).enumerate() {
                let i = s + x * CHANNELS;
                let l1 = diff_luma(f64::from(a[i + R]), f64::from(a[i + G]), f64::from(a[i + B]));
                let l2 = diff_luma(f64::from(b[i + R]), f64::from(b[i + G]), f64::from(b[i + B]));
                let v = diff_int(l1, l2, 65535.0) as u16;
                px[A] = 65535;
                px[R] = v;
                px[G] = v;
                px[B] = v;
            }
        }),
        (Samples::F32(o), Samples::F32(a), Samples::F32(b)) => par_rows(o, row_len, |y, row| {
            let s = y * row_len;
            for (x, px) in row.chunks_exact_mut(CHANNELS).enumerate() {
                let i = s + x * CHANNELS;
                let l1 = diff_luma(f64::from(a[i + R]), f64::from(a[i + G]), f64::from(a[i + B]));
                let l2 = diff_luma(f64::from(b[i + R]), f64::from(b[i + G]), f64::from(b[i + B]));
                let v = diff_float(l1, l2);
                px[A] = 1.0;
                px[R] = v;
                px[G] = v;
                px[B] = v;
            }
        }),
        _ => unreachable!("the shapes were checked"),
    }
}
