//! Vector image coding of the engine.
//!
//! The "encoded vector image" is the four channel A, R, G, B image that Create Vectors writes
//! and Vectors In reads: `R` holds `dx` and `G` holds `dy`, centred (127, 32767 or 0.5 mean
//! zero) and scaled by `Max Displace`. This module holds the two conversions with their exact
//! arithmetic:
//!
//! * [`encode_vectors`]: `FlowField` -> encoded image.
//! * [`decode_vectors`]: encoded image -> `FlowField`; [`vector_hole_mask`] is the mask built
//!   from the alpha channel for the hole fill; [`apply_vec_scale`] applies Vec Scale X / Y.
//!
//! The push-pull hole fill of Vectors In is not here ([`crate::holefill`]). The order is
//! [`decode_vectors_with_holes`] -> hole fill (`flow`, `holes`) -> [`apply_vec_scale`].

use crate::flow::FlowField;
use crate::image::{Image, Samples, CHANNELS};
use crate::warp::Flow;
use fx_core::{par_rows, FxError, Result};

/// Truncation toward zero; NaN and out of range values give `i64::MIN`.
fn cvtt64(x: f64) -> i64 {
    if x.is_nan() || x >= 9.223372036854775808e18 || x < -9.223372036854775808e18 {
        i64::MIN
    } else {
        x as i64
    }
}

/// Encodes `flow` into the four channel image `dst` the way Create Vectors does. `scale` is
/// `Max Displace` in pixels of the OUTPUT resolution (Create Vectors passes `Max Displace` times
/// the horizontal downsample factor of the render).
///
/// With `dw1 = dst.width - 1`, `dh1 = dst.height - 1` and `fw1`, `fh1` the same for the flow (all
/// converted to f32), the encoder computes in single precision
/// `sx = fw1 / dw1`, `sy = fh1 / dh1`, `kx = (dw1 / fw1) / scale`, `ky = (dh1 / fh1) / scale`.
/// Per output pixel `(x, y)` the flow is sampled bilinearly at `(x * sx, y * sy)` (the two
/// channel flow sampler, so the flow may have another resolution), `fx = u * kx`, `fy = v * ky`,
/// and then, per depth:
///
/// * 8 bpc: `fx` and `fy` are clamped to `[-1, 1]` (`fx > 1` gives 1, else `-1 > fx` gives -1;
///   a NaN passes through), `+ 1.0` (single precision), then `R = low byte of
///   trunc((double)(fx + 1) * 127.0 + 0.5)`; same for `G`. Zero encodes as 127, `+scale` as 254,
///   `-scale` as 0.
/// * 16 bpc: the same with `32767.0` and the low word; zero is 32767, full scale 65534 and 0.
/// * float: NO clamp, `R = (fx + 1.0) * 0.5`, `G = (fy + 1.0) * 0.5`, zero is 0.5.
///
/// Alpha is the white of the depth (255, 65535, 1.0) and blue is 0 in every pixel, so a decode
/// of this image treats every vector as valid.
///
/// Errors: `Mismatch` when `dst` or `flow` is smaller than 2x2 (infinite or NaN ratios).
pub fn encode_vectors(dst: &mut Image, flow: &FlowField, scale: f32) -> Result<()> {
    if dst.width < 2 || dst.height < 2 || flow.width < 2 || flow.height < 2 {
        return Err(FxError::Mismatch(format!(
            "vector encode needs images and a flow of at least 2x2 (image {}x{}, flow {}x{})",
            dst.width, dst.height, flow.width, flow.height
        )));
    }
    if flow.data.len() != flow.width * flow.height * 2 {
        return Err(FxError::Mismatch("vector encode: flow data length does not match its size".into()));
    }
    let (dw1, dh1) = ((dst.width as i32 - 1) as f32, (dst.height as i32 - 1) as f32);
    let (fw1, fh1) = ((flow.width as i32 - 1) as f32, (flow.height as i32 - 1) as f32);
    let sx = fw1 / dw1;
    let sy = fh1 / dh1;
    let kx = (dw1 / fw1) / scale;
    let ky = (dh1 / fh1) / scale;
    let fl = Flow::new(flow);
    let width = dst.width;
    // the per pixel sample position and scaled vector, shared by the three depths
    let sample = move |x: usize, y: usize| -> (f32, f32) {
        let [u, v] = fl.bilerp(x as f32 * sx, y as f32 * sy);
        (u * kx, v * ky)
    };
    match &mut dst.samples {
        Samples::U8(out) => par_rows(out, width * CHANNELS, |y, row| {
            for (x, px) in row.chunks_exact_mut(CHANNELS).enumerate() {
                let (mut a, mut b) = sample(x, y);
                clamp_unit(&mut a);
                clamp_unit(&mut b);
                let (a, b) = (a + 1.0f32, b + 1.0f32);
                px[0] = 0xff;
                px[1] = cvtt64(a as f64 * 127.0 + 0.5) as u8;
                px[2] = cvtt64(b as f64 * 127.0 + 0.5) as u8;
                px[3] = 0;
            }
        }),
        Samples::U16(out) => par_rows(out, width * CHANNELS, |y, row| {
            for (x, px) in row.chunks_exact_mut(CHANNELS).enumerate() {
                let (mut a, mut b) = sample(x, y);
                clamp_unit(&mut a);
                clamp_unit(&mut b);
                let (a, b) = (a + 1.0f32, b + 1.0f32);
                px[0] = 0xffff;
                px[1] = cvtt64(a as f64 * 32767.0 + 0.5) as u16;
                px[2] = cvtt64(b as f64 * 32767.0 + 0.5) as u16;
                px[3] = 0;
            }
        }),
        Samples::F32(out) => par_rows(out, width * CHANNELS, |y, row| {
            for (x, px) in row.chunks_exact_mut(CHANNELS).enumerate() {
                let (a, b) = sample(x, y);
                px[0] = 1.0;
                px[1] = (a + 1.0f32) * 0.5f32;
                px[2] = (b + 1.0f32) * 0.5f32;
                px[3] = 0.0;
            }
        }),
    }
    Ok(())
}

/// Clamps to `[-1, 1]`: `v > 1.0` gives 1.0, else `-1.0 > v` gives -1.0; a NaN falls through
/// unchanged.
#[inline(always)]
fn clamp_unit(v: &mut f32) {
    if *v > 1.0 {
        *v = 1.0;
    } else if -1.0f32 > *v {
        *v = -1.0;
    }
}

/// Convenience: a new encoded vector image of the given size and format (all other bytes are
/// written by the encode as well, so the cleared start does not show).
pub fn encode_vectors_image(
    width: usize,
    height: usize,
    fmt: crate::image::Fmt,
    flow: &FlowField,
    scale: f32,
) -> Result<Image> {
    let mut img = Image::new(width, height, fmt);
    encode_vectors(&mut img, flow, scale)?;
    Ok(img)
}

/// Decodes an encoded vector image into a flow field of the same size. The integer depths are
/// computed in DOUBLE precision. `max_displace` is the `Max Displace` the layer reports, in
/// pixels of the layer.
///
/// Per pixel, `A == 0` (or `0.0` for float) gives the zero vector `(0.0, 0.0)`. Otherwise:
///
/// * 8 bpc: `u = (f32)(((double)R * 2.0 / 254.0 - 1.0) * (double)max_displace)`, `v` the same
///   with `G`.
/// * 16 bpc: the same with `65534.0`.
/// * float: `u = (R * 2.0 - 1.0) * max_displace` in single precision, `v` with `G`.
///
/// The vectors are in pixels of the layer. Vec Scale X / Y and the hole fill come after
/// ([`apply_vec_scale`], module docs).
pub fn decode_vectors(vec: &Image, max_displace: f32) -> FlowField {
    let (w, h) = (vec.width, vec.height);
    let mut flow = FlowField::new(w, h);
    if w == 0 || h == 0 {
        return flow;
    }
    let md = max_displace as f64;
    match &vec.samples {
        Samples::U8(v) => par_rows(&mut flow.data, w * 2, |y, row| {
            for (x, o) in row.chunks_exact_mut(2).enumerate() {
                let p = &v[(y * w + x) * CHANNELS..(y * w + x + 1) * CHANNELS];
                if p[0] == 0 {
                    o[0] = 0.0;
                    o[1] = 0.0;
                } else {
                    o[0] = dec_int(p[1] as f32, 254.0, md);
                    o[1] = dec_int(p[2] as f32, 254.0, md);
                }
            }
        }),
        Samples::U16(v) => par_rows(&mut flow.data, w * 2, |y, row| {
            for (x, o) in row.chunks_exact_mut(2).enumerate() {
                let p = &v[(y * w + x) * CHANNELS..(y * w + x + 1) * CHANNELS];
                if p[0] == 0 {
                    o[0] = 0.0;
                    o[1] = 0.0;
                } else {
                    o[0] = dec_int(p[1] as f32, 65534.0, md);
                    o[1] = dec_int(p[2] as f32, 65534.0, md);
                }
            }
        }),
        Samples::F32(v) => par_rows(&mut flow.data, w * 2, |y, row| {
            for (x, o) in row.chunks_exact_mut(2).enumerate() {
                let p = &v[(y * w + x) * CHANNELS..(y * w + x + 1) * CHANNELS];
                if p[0] == 0.0 {
                    o[0] = 0.0;
                    o[1] = 0.0;
                } else {
                    o[0] = (p[1] * 2.0f32 - 1.0f32) * max_displace;
                    o[1] = (p[2] * 2.0f32 - 1.0f32) * max_displace;
                }
            }
        }),
    }
    flow
}

/// `((c * 2.0) / den - 1.0) * scale` in double precision with one narrowing to single at the
/// end; the channel value converts to double exactly.
#[inline(always)]
fn dec_int(c: f32, den: f64, scale: f64) -> f32 {
    (((c as f64 * 2.0) / den - 1.0) * scale) as f32
}

/// The hole mask derived from the alpha channel for the hole fill: the alpha channel as a one
/// channel image of the same sample type, inverted: `255 - A`, `65535 - A` or `1.0 - A`. A pixel
/// with alpha zero is a full hole; partial alpha leaves a partial hole value.
#[derive(Clone, Debug, PartialEq)]
pub enum HoleMask {
    U8(Vec<u8>),
    U16(Vec<u16>),
    F32(Vec<f32>),
}

impl HoleMask {
    /// Number of mask samples (width * height).
    pub fn len(&self) -> usize {
        match self {
            HoleMask::U8(v) => v.len(),
            HoleMask::U16(v) => v.len(),
            HoleMask::F32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Builds the [`HoleMask`] of an encoded vector image (see the type).
pub fn vector_hole_mask(vec: &Image) -> HoleMask {
    match &vec.samples {
        Samples::U8(v) => HoleMask::U8(v.chunks_exact(CHANNELS).map(|p| 0xffu8.wrapping_sub(p[0])).collect()),
        Samples::U16(v) => HoleMask::U16(v.chunks_exact(CHANNELS).map(|p| 0xffffu16.wrapping_sub(p[0])).collect()),
        Samples::F32(v) => HoleMask::F32(v.chunks_exact(CHANNELS).map(|p| 1.0f32 - p[0]).collect()),
    }
}

/// The flow and the hole mask: the two inputs of the hole fill
/// ([`crate::holefill::fill_flow_holes`]). A caller can also use the flow as is (holes are zero
/// vectors).
pub fn decode_vectors_with_holes(vec: &Image, max_displace: f32) -> (FlowField, HoleMask) {
    (decode_vectors(vec, max_displace), vector_hole_mask(vec))
}

/// Vec Scale X and Y: `dx = sx * dx`, `dy = sy * dy` in single precision, applied unconditionally
/// (also for 1.0). The values are the layer's `Vec Scale X/Y` (default 1.0) and are applied
/// after the hole fill.
pub fn apply_vec_scale(flow: &mut FlowField, sx: f32, sy: f32) {
    for p in flow.data.chunks_exact_mut(2) {
        p[0] = sx * p[0];
        p[1] = sy * p[1];
    }
}

#[cfg(test)]
mod tests;
