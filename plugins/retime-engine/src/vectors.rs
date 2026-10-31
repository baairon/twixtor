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
//! * [`encode_vectors_merged`] and [`merge_flows`]: the matte weighted merge of the layered Create
//!   Vectors pass (U, a rule of this project that no effect runs: the layered pass composites
//!   its layers from the bottom up instead): the per layer vectors of [`encode_vectors`] summed with their
//!   weights in layer order from `0.0`, then today's clamp and store (or, for the Debug views, the
//!   sum in output pixels).
//!
//! The push-pull hole fill of Vectors In is not here ([`crate::holefill`]). The order is
//! [`decode_vectors_with_holes`] -> hole fill (`flow`, `holes`) -> [`apply_vec_scale`].

use crate::flow::FlowField;
use crate::image::{Image, Samples, CHANNELS};
use crate::matte::Matte;
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

/// The encoded zero vector image of the given size and format: every pixel holds what
/// [`encode_vectors`] writes for a zero flow, alpha the white of the depth, R and G the centre
/// (127, 32767 or 0.5), B 0. It has no size limit: Create Vectors writes it for a frame with a
/// side of 1, which has no motion search (U, ours, parity gap G7, see
/// [`crate::prep::searchable`]).
pub fn zero_vectors_image(width: usize, height: usize, fmt: crate::image::Fmt) -> Image {
    let mut img = Image::new(width, height, fmt);
    match &mut img.samples {
        Samples::U8(out) => out.chunks_exact_mut(CHANNELS).for_each(|px| px.copy_from_slice(&[0xff, 127, 127, 0])),
        Samples::U16(out) => out.chunks_exact_mut(CHANNELS).for_each(|px| px.copy_from_slice(&[0xffff, 32767, 32767, 0])),
        Samples::F32(out) => out.chunks_exact_mut(CHANNELS).for_each(|px| px.copy_from_slice(&[1.0, 0.5, 0.5, 0.0])),
    }
    img
}

/// The per layer constants of the merged encoder and of [`merge_flows`]: the sampler of one
/// layer's flow, its sample step `(sx, sy)`, its scale `(kx, ky)` and its weights.
struct MergeLayer<'a> {
    fl: Flow<'a>,
    sx: f32,
    sy: f32,
    kx: f32,
    ky: f32,
    weight: &'a [f32],
}

/// Checks the layers of a merge onto a `dw x dh` output and returns their constants: per layer
/// `sx = fw1 / dw1`, `sy = fh1 / dh1` and `kx = (dw1 / fw1) / scale`, `ky = (dh1 / fh1) / scale`
/// (`scale = None` leaves out the division: `kx = dw1 / fw1`), all in single precision with the
/// sizes minus one converted to f32, exactly as [`encode_vectors`] computes them.
fn merge_layers<'a>(
    dw: usize,
    dh: usize,
    layers: &[(&'a FlowField, &'a Matte)],
    scale: Option<f32>,
    what: &str,
) -> Result<Vec<MergeLayer<'a>>> {
    if layers.is_empty() {
        return Err(FxError::Mismatch(format!("{what}: no layer to merge")));
    }
    if dw < 2 || dh < 2 {
        return Err(FxError::Mismatch(format!("{what} needs an output of at least 2x2 (output {dw}x{dh})")));
    }
    let (dw1, dh1) = ((dw as i32 - 1) as f32, (dh as i32 - 1) as f32);
    let mut out = Vec::with_capacity(layers.len());
    for (k, &(flow, m)) in layers.iter().enumerate() {
        if flow.width < 2 || flow.height < 2 || flow.data.len() != flow.width * flow.height * 2 {
            return Err(FxError::Mismatch(format!(
                "{what}: layer {k} has a flow of {}x{} holding {} values (at least 2x2 needed)",
                flow.width,
                flow.height,
                flow.data.len()
            )));
        }
        if m.width != dw || m.height != dh || m.data.len() != dw * dh {
            return Err(FxError::Mismatch(format!(
                "{what}: the weight of layer {k} is {}x{} for an output of {dw}x{dh}",
                m.width, m.height
            )));
        }
        let (fw1, fh1) = ((flow.width as i32 - 1) as f32, (flow.height as i32 - 1) as f32);
        let (kx, ky) = match scale {
            Some(s) => ((dw1 / fw1) / s, (dh1 / fh1) / s),
            None => (dw1 / fw1, dh1 / fh1),
        };
        out.push(MergeLayer { fl: Flow::new(flow), sx: fw1 / dw1, sy: fh1 / dh1, kx, ky, weight: &m.data });
    }
    Ok(out)
}

/// The weighted sum of the layers at output pixel `(x, y)` of a `width` wide output: from
/// `(0.0, 0.0)`, for the layers in their order, `[u, v] = bilerp(x * sx, y * sy)` and
/// `acc_x = acc_x + w * (u * kx)`, `acc_y = acc_y + w * (v * ky)` (single precision, in this order).
#[inline(always)]
fn merged_at(per: &[MergeLayer<'_>], x: usize, y: usize, width: usize) -> (f32, f32) {
    let (mut ax, mut ay) = (0.0f32, 0.0f32);
    for l in per {
        let [u, v] = l.fl.bilerp(x as f32 * l.sx, y as f32 * l.sy);
        let wk = l.weight[y * width + x];
        ax = ax + wk * (u * l.kx);
        ay = ay + wk * (v * l.ky);
    }
    (ax, ay)
}

/// The vector image of a layered Create Vectors render (one flow per layer V, the matte
/// weighted merge U; no effect runs it): `layers` holds, for the layers `first..=last` of Layers Output in
/// that order, each layer's flow (`X(p) ~ Y(p + flow(p))`, any size of at least 2x2, usually the
/// working resolution) and its weight at the output size, the OVER weight of
/// `matte::over_weights` (raw mattes, Main_BG 1).
///
/// Per output pixel `(x, y)` (U, ours): every layer's vector is what [`encode_vectors`] computes
/// for it, `(u * kx, v * ky)` with `[u, v]` the flow sampled bilinearly at `(x * sx, y * sy)` and
/// `sx`, `sy`, `kx`, `ky` from the size of that layer's flow; the merged value is the sum over
/// the layers in their order, starting from `0.0`: `acc_x = acc_x + w_k * (u_k * kx_k)` (f32, no
/// fused multiply add), the same for `y`. The sum then takes today's clamp and store of
/// [`encode_vectors`] per depth; alpha is white and blue 0 in every pixel (the channel copy of
/// the merge gives the opaque alpha and the zero blue, I).
///
/// One layer of weight 1.0 everywhere is [`encode_vectors`] bit for bit (`0.0 + 1.0 * v` is `v`
/// up to the sign of a zero, which the store does not see).
///
/// Errors: `Mismatch` for an empty list, an output or a flow below 2x2, a flow whose data does not
/// match its size and a weight of another size than `dst`.
pub fn encode_vectors_merged(dst: &mut Image, layers: &[(&FlowField, &Matte)], scale: f32) -> Result<()> {
    let per = merge_layers(dst.width, dst.height, layers, Some(scale), "vector merge")?;
    let width = dst.width;
    let per = &per;
    let sample = move |x: usize, y: usize| -> (f32, f32) { merged_at(per, x, y, width) };
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

/// [`encode_vectors_merged`] with the blue channel holding `blue` (a weight in 0..1 at the size of
/// `dst`) instead of 0: `q(w * 255)` at 8 bpc, `q(w * 65535)` at 16 bpc (low byte or word, the
/// engine's rounding `q`) and `w` in float. MEASURED against `CreateMotionVectors_64` under the
/// emulator: when Layers Output selects one layer, the plugin's blue channel holds that layer's
/// weight. Errors: as
/// [`encode_vectors_merged`], and `Mismatch` for a `blue` of another size.
pub fn encode_vectors_merged_blue(dst: &mut Image, layers: &[(&FlowField, &Matte)], scale: f32, blue: &Matte) -> Result<()> {
    if blue.width != dst.width || blue.height != dst.height || blue.data.len() != dst.width * dst.height {
        return Err(FxError::Mismatch(format!(
            "vector merge: a {}x{} blue weight for an output of {}x{}",
            blue.width, blue.height, dst.width, dst.height
        )));
    }
    encode_vectors_merged(dst, layers, scale)?;
    write_blue(dst, blue);
    Ok(())
}

/// [`encode_vectors`] of one layer's flow, NOT weighted, with the blue channel holding `blue` (in
/// 0..1 at the size of `dst`, written as [`encode_vectors_merged_blue`] does). READ in
/// `CreateMotionVectors_64` (its per frame driver): the output group is composited from
/// its bottom layer up, the bottom layer's flow encoded plainly and every layer above
/// it laid over with its matte; a group of one layer is that plain encode alone, and
/// its blue channel is then overwritten by the layer's visible matte plane (channel 3).
/// Errors: as [`encode_vectors`], and `Mismatch` for a
/// `blue` of another size.
pub fn encode_vectors_blue(dst: &mut Image, flow: &FlowField, scale: f32, blue: &Matte) -> Result<()> {
    if blue.width != dst.width || blue.height != dst.height || blue.data.len() != dst.width * dst.height {
        return Err(FxError::Mismatch(format!(
            "vector encode: a {}x{} blue plane for an output of {}x{}",
            blue.width, blue.height, dst.width, dst.height
        )));
    }
    encode_vectors(dst, flow, scale)?;
    write_blue(dst, blue);
    Ok(())
}

/// Lays the vectors of one more layer over the vector image `dst` (an [`encode_vectors`] output or
/// an earlier composite) with that layer's raw matte `m` (one value per pixel of `dst` in the
/// integer scale of the depth: 0..255, 0..65535, or the float matte). READ in
/// `CreateMotionVectors_64` (the layer kernel and its per depth variants), all single precision: the layer's
/// vector `(a, b)` is
/// sampled as [`encode_vectors`] samples it and clamped to `[-1, 1]` the same way (also in float),
/// `e = (a + 1.0) * H` with `H` = 127, 32767 or 0.5, the weight `f = m * K` (`K` the f32 constants
/// `1/255`, `1/65535`, or 1 in float), and every vector channel becomes
/// `((e - old) * f + old) + 0.5` truncated at 8 and 16 bpc, `((e - old) * f + old) + 0.0` in float;
/// alpha is the white of the depth and blue 0. Errors: as [`encode_vectors`], and `Mismatch` for a
/// matte of another length.
pub fn composite_vectors(dst: &mut Image, flow: &FlowField, scale: f32, m: &[f32]) -> Result<()> {
    if dst.width < 2 || dst.height < 2 || flow.width < 2 || flow.height < 2 {
        return Err(FxError::Mismatch(format!(
            "vector composite needs images and a flow of at least 2x2 (image {}x{}, flow {}x{})",
            dst.width, dst.height, flow.width, flow.height
        )));
    }
    if flow.data.len() != flow.width * flow.height * 2 || m.len() != dst.width * dst.height {
        return Err(FxError::Mismatch("vector composite: a flow or matte of another size".into()));
    }
    let (dw1, dh1) = ((dst.width as i32 - 1) as f32, (dst.height as i32 - 1) as f32);
    let (fw1, fh1) = ((flow.width as i32 - 1) as f32, (flow.height as i32 - 1) as f32);
    let (sx, sy) = (fw1 / dw1, fh1 / dh1);
    let (kx, ky) = ((dw1 / fw1) / scale, (dh1 / fh1) / scale);
    let fl = Flow::new(flow);
    let width = dst.width;
    let sample = move |x: usize, y: usize| -> (f32, f32) {
        let [u, v] = fl.bilerp(x as f32 * sx, y as f32 * sy);
        let (mut a, mut b) = (u * kx, v * ky);
        clamp_unit(&mut a);
        clamp_unit(&mut b);
        (a + 1.0f32, b + 1.0f32)
    };
    match &mut dst.samples {
        Samples::U8(out) => par_rows(out, width * CHANNELS, |y, row| {
            for (x, px) in row.chunks_exact_mut(CHANNELS).enumerate() {
                let (a, b) = sample(x, y);
                let f = m[y * width + x] * crate::matte::PL_K8;
                for (c, e) in [(1usize, a * 127.0f32), (2, b * 127.0f32)] {
                    let old = f32::from(px[c]);
                    px[c] = (((e - old) * f + old) + 0.5f32) as i64 as u8;
                }
                px[0] = 0xff;
                px[3] = 0;
            }
        }),
        Samples::U16(out) => par_rows(out, width * CHANNELS, |y, row| {
            for (x, px) in row.chunks_exact_mut(CHANNELS).enumerate() {
                let (a, b) = sample(x, y);
                let f = m[y * width + x] * crate::matte::PL_K16;
                for (c, e) in [(1usize, a * 32767.0f32), (2, b * 32767.0f32)] {
                    let old = f32::from(px[c]);
                    px[c] = (((e - old) * f + old) + 0.5f32) as i64 as u16;
                }
                px[0] = 0xffff;
                px[3] = 0;
            }
        }),
        Samples::F32(out) => par_rows(out, width * CHANNELS, |y, row| {
            for (x, px) in row.chunks_exact_mut(CHANNELS).enumerate() {
                let (a, b) = sample(x, y);
                let f = m[y * width + x];
                for (c, e) in [(1usize, a * 0.5f32), (2, b * 0.5f32)] {
                    let old = px[c];
                    px[c] = ((e - old) * f + old) + 0.0f32;
                }
                px[0] = 1.0;
                px[3] = 0.0;
            }
        }),
    }
    Ok(())
}

/// Writes `blue` (same size as `dst`, checked by the callers) into the blue channel of `dst`:
/// `q(w * 255)`, `q(w * 65535)` or `w`.
fn write_blue(dst: &mut Image, blue: &Matte) {
    let width = dst.width;
    let b = &blue.data;
    match &mut dst.samples {
        Samples::U8(out) => par_rows(out, width * CHANNELS, |y, row| {
            for (x, px) in row.chunks_exact_mut(CHANNELS).enumerate() {
                px[3] = crate::image::q(b[y * width + x] * 255.0f32) as u8;
            }
        }),
        Samples::U16(out) => par_rows(out, width * CHANNELS, |y, row| {
            for (x, px) in row.chunks_exact_mut(CHANNELS).enumerate() {
                px[3] = crate::image::q(b[y * width + x] * 65535.0f32) as u16;
            }
        }),
        Samples::F32(out) => par_rows(out, width * CHANNELS, |y, row| {
            for (x, px) in row.chunks_exact_mut(CHANNELS).enumerate() {
                px[3] = b[y * width + x];
            }
        }),
    }
}

/// [`encode_vectors_merged`] into a new image of the given size and format.
pub fn encode_vectors_merged_image(
    width: usize,
    height: usize,
    fmt: crate::image::Fmt,
    layers: &[(&FlowField, &Matte)],
    scale: f32,
) -> Result<Image> {
    let mut img = Image::new(width, height, fmt);
    encode_vectors_merged(&mut img, layers, scale)?;
    Ok(img)
}

/// The merged flow of a layered render at the output size `width x height`, in pixels of the
/// output (`X(p) ~ Y(p + flow(p))`): the flow the Debug views use with FG layers (rule G4: "with
/// `nfg >= 1` the merged flow"). It is the sum of [`encode_vectors_merged`] before the division by
/// `Max Displace`, the clamp and the store (U, ours): per output pixel, from `0.0` and in the
/// order of `layers`, `acc_x = acc_x + w_k * (u_k * kx_k)` with `kx_k = dw1 / fw1_k` and
/// `ky_k = dh1 / fh1_k` (f32), the same sampling and weights as the encoder. No gap is recorded
/// here (the Debug view that uses it does). Errors as [`encode_vectors_merged`].
pub fn merge_flows(width: usize, height: usize, layers: &[(&FlowField, &Matte)]) -> Result<FlowField> {
    let per = merge_layers(width, height, layers, None, "flow merge")?;
    let mut out = FlowField::new(width, height);
    let per = &per;
    par_rows(&mut out.data, width * 2, |y, row| {
        for (x, o) in row.chunks_exact_mut(2).enumerate() {
            let (a, b) = merged_at(per, x, y, width);
            o[0] = a;
            o[1] = b;
        }
    });
    Ok(out)
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
#[cfg(test)]
mod merge_tests;
