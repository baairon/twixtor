//! The Debug display of create-vectors (rule G4).
//! `debug_view_method` 1 Warp, 2 Blend, 3 Diff, 4 Detail Compare and 5 Edge Compare are the
//! engine view codes 4, 0, 1, 2 and 3. Every view starts from `W`, the forward warp of the first
//! frame `X` of the solve by its flow (the Warp view); the other four compose `W` with the
//! second frame `Y` (U, ours, rule G4):
//!
//! * Warp (4): `W` itself, the forward warp at amount 1.0 with no coverage and no hole fill;
//! * Blend (0): `blend(W, Y, 0.5)` of `crate::blend` (the weight on `Y`);
//! * Diff (1): `|W - Y|` per colour channel of the premultiplied samples, alpha white (opaque);
//! * Detail Compare (2): an R/G overlay, red the detail of `W` and green the detail of `Y`, blue 0,
//!   alpha white, the detail of a frame `clamp01((L - blur5(L)) / 255 + 0.5)` of its luma `L`;
//! * Edge Compare (3): the same overlay of the edge measure `min(|grad(blur2(L))| * 0.05, 1)`, the
//!   edge term of the Contrast/Edge Enhance prep.
//!
//! With FG layers the flow is the merged one (`create_vectors::job_flow`). The Debug frame roles
//! (Previous: `X` is frame `n + 1`, `Y` frame `n`; Next: `X` is `n - 1`) are the pair of
//! `create_vectors::frame_pair`.
//!
//! Claim tags: V established, I inferred, U not established (a rule of this project, pinned in
//! the parity gaps register until a reference render grades it).
//!
//! Precision contract: the contracts of the stages it composes (the forward warp, `blend`, the
//! luma of `prep::luma`, the pyramid blur `prep::blur5`, the Gaussian `conv::blur_gaussian` and
//! the gradient `calc_grad` of the solver), f32 in the order written, no fused multiply add, rows
//! in parallel through rayon row chunks only, so the result is the same for any thread count. In
//! detail:
//! - the luma is `prep::luma(img, false, false)` (no alpha term, full range; values in 0..255);
//! - detail: `b = blur5(L)`, `v = clamp01(((L - b) / 255.0) + 0.5)` (`matte::clamp01`, a NaN
//!   gives 0.0);
//! - edge: `b` the one channel Gaussian of radius 2.0 of `L`, `(gx, gy) = calc_grad(b)`,
//!   `m = sqrt(gx * gx + gy * gy)`, `v = m <= 20.0 ? m * 0.05 : 1.0` (equal to
//!   `min(m * 0.05, 1.0)` for every number; a NaN gives 1.0, as in the prep);
//! - an overlay value `v` in 0..1 is stored as `q(v * 255.0)` at 8 bpc and `q(v * 65535.0)` at
//!   16 bpc (`q(x) = trunc((x as f64) + 0.5)`, the engine's rounding), `v` in float;
//! - Diff: `|w - y|` of the integers at 8 and 16 bpc (exact), `(w - y).abs()` in float.

use crate::blend::blend;
use crate::flow::{calc_grad, conv, FlowField};
use crate::image::{q, Image, Samples, A, B, CHANNELS, G, R};
use crate::matte::clamp01;
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
    let mut out = Image::new(w, h, fmt);
    match code {
        VIEW_BLEND => blend(&mut out, &warped, y, 0.5f32),
        VIEW_DIFF => diff(&mut out, &warped, y),
        VIEW_DETAIL => overlay(&mut out, &detail(&prep::luma(&warped, false, false)?), &detail(&prep::luma(y, false, false)?)),
        _ => overlay(&mut out, &edge(&prep::luma(&warped, false, false)?), &edge(&prep::luma(y, false, false)?)),
    }
    Ok(out)
}

/// The detail of a luma plane: `clamp01(((L - blur5(L)) / 255.0) + 0.5)` per sample (the band
/// pass of the solver's pyramid, centred at 0.5).
pub(crate) fn detail(l: &Plane) -> Vec<f32> {
    let mut b = l.clone();
    prep::blur5(&mut b);
    let mut out = vec![0.0f32; l.data.len()];
    if l.width == 0 || out.is_empty() {
        return out;
    }
    par_rows(&mut out, l.width, |y, row| {
        let o = y * l.width;
        for (x, v) in row.iter_mut().enumerate() {
            *v = clamp01(((l.data[o + x] - b.data[o + x]) / 255.0f32) + 0.5f32);
        }
    });
    out
}

/// The edge measure of a luma plane: the Gaussian of radius 2.0, the gradient, and
/// `m <= 20.0 ? m * 0.05 : 1.0` of its magnitude `m = sqrt(gx * gx + gy * gy)`.
pub(crate) fn edge(l: &Plane) -> Vec<f32> {
    let (w, h) = (l.width, l.height);
    let mut b = l.data.clone();
    conv::blur_gaussian(&mut b, w, h, 1, 2.0);
    let g = calc_grad(&b, w, h);
    let mut out = vec![0.0f32; w * h];
    if w == 0 || out.is_empty() {
        return out;
    }
    par_rows(&mut out, w, |y, row| {
        for (x, v) in row.iter_mut().enumerate() {
            let gx = g[(y * w + x) * 2];
            let gy = g[(y * w + x) * 2 + 1];
            let m = (gx * gx + gy * gy).sqrt();
            *v = if m <= 20.0f32 { m * 0.05f32 } else { 1.0f32 };
        }
    });
    out
}

/// The R/G overlay: alpha white, red `r`, green `g`, blue 0, each value in 0..1 stored per depth
/// (see the module documentation).
pub(crate) fn overlay(out: &mut Image, r: &[f32], g: &[f32]) {
    let w = out.width;
    let row_len = out.row_len();
    if row_len == 0 || out.height == 0 {
        return;
    }
    match &mut out.samples {
        Samples::U8(o) => par_rows(o, row_len, |y, row| {
            for (x, px) in row.chunks_exact_mut(CHANNELS).enumerate() {
                let i = y * w + x;
                px[A] = 255;
                px[R] = q(r[i] * 255.0f32) as u8;
                px[G] = q(g[i] * 255.0f32) as u8;
                px[B] = 0;
            }
        }),
        Samples::U16(o) => par_rows(o, row_len, |y, row| {
            for (x, px) in row.chunks_exact_mut(CHANNELS).enumerate() {
                let i = y * w + x;
                px[A] = 65535;
                px[R] = q(r[i] * 65535.0f32) as u16;
                px[G] = q(g[i] * 65535.0f32) as u16;
                px[B] = 0;
            }
        }),
        Samples::F32(o) => par_rows(o, row_len, |y, row| {
            for (x, px) in row.chunks_exact_mut(CHANNELS).enumerate() {
                let i = y * w + x;
                px[A] = 1.0;
                px[R] = r[i];
                px[G] = g[i];
                px[B] = 0.0;
            }
        }),
    }
}

/// `|W - Y|` per colour channel, alpha white.
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
                px[A] = 255;
                for c in [R, G, B] {
                    px[c] = a[i + c].abs_diff(b[i + c]);
                }
            }
        }),
        (Samples::U16(o), Samples::U16(a), Samples::U16(b)) => par_rows(o, row_len, |y, row| {
            let s = y * row_len;
            for (x, px) in row.chunks_exact_mut(CHANNELS).enumerate() {
                let i = s + x * CHANNELS;
                px[A] = 65535;
                for c in [R, G, B] {
                    px[c] = a[i + c].abs_diff(b[i + c]);
                }
            }
        }),
        (Samples::F32(o), Samples::F32(a), Samples::F32(b)) => par_rows(o, row_len, |y, row| {
            let s = y * row_len;
            for (x, px) in row.chunks_exact_mut(CHANNELS).enumerate() {
                let i = s + x * CHANNELS;
                px[A] = 1.0;
                for c in [R, G, B] {
                    px[c] = (a[i + c] - b[i + c]).abs();
                }
            }
        }),
        _ => unreachable!("the shapes were checked"),
    }
}
