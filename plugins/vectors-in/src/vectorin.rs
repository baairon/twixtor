//! The Vectors In path of plugin 04: the two vector layers become the two flow fields the core
//! composes with.
//!
//! 1. Presence is decided once per call at `t0 = current_time`: a vector layer counts as present
//!    when its checkout has a width and a height above 0. [`present_at`] is that probe. A layer
//!    keeps its size outside its clip (there its frames are transparent), so a set layer is present
//!    at every time.
//! 2. The core asks for the flows only on the vector path (`vectors_quality != 0 && !exact` after
//!    the segment override). It loads, in this order, `F_AB` from the layer `Vectors: Next` at
//!    frame `A` and `F_BA` from `Vectors: Prev` at frame `B = A + 1`. The fetch time of frame `k`
//!    is `T(k) = trunc(dur * k + 0.5)` ([`crate::timemap`]).
//! 3. Each loader decodes its image with `Max Displace` (read once at `t0`, scaled by the
//!    horizontal downsample ratio, which is 1 here), fills the alpha 0 holes of the flow with the
//!    pull push fill (its vertical taps in the order of `settings.stages`, the tap above first
//!    for this plugin, see `glue.rs`) and multiplies by `Vec Scale X / Y` read at the same `T(k)`
//!    ([`retime_engine::compose::flow_from_vector_image_with`]).
//! 4. With exactly one layer present the other flow is its inversion; with none the render writes
//!    the error fill before any engine call (`glue.rs`).
//! 5. With FG matte layers the loaders run once per layer of the displayed group
//!    ([`load_layer_flows`]), as the original's per layer vector loader does: each load first
//!    caps the alpha of the vector image by the layer's composition matte at the load's frame
//!    ([`matte_step`], `min(alpha, c)`), then decodes, fills and scales as above, so the layer's
//!    flow outside its matte (and in part where the matte is soft) is the hole fill of the flow
//!    inside (READ in the original's code).
//!
//! The vector image goes through the premultiplying prepare like a source frame
//! (`Image::from_frame`), so a vector layer has to be opaque except for alpha 0 holes: for alpha
//! below full the premultiply scales the encoded values and the decode is wrong.

use fx_core::{Env, FxError, Layer, ParamCurves, Result};
use retime_engine::compose::{flow_from_vector_image_with, Flows};
use retime_engine::flow::FlowField;
use retime_engine::frontend::fixed_reader;
use retime_engine::host::{layer_is_set, EngineSettings};
use retime_engine::image::{Image, Samples, A, CHANNELS};
use retime_engine::layers::LayerMattes;
use retime_engine::matte::Matte;
use retime_engine::timeline::{Selection, TimeMap};

use crate::prerender::{LAYER_VECTORS_NEXT, LAYER_VECTORS_PREV};

/// The probe of a vector layer at `t0`: the size of the checkout, `None` when the layer is not set or
/// has no pixels. A layer that returns `None` here is absent for the whole render, whatever it holds
/// at other times.
///
/// Outside its clip a layer keeps its size: the checkout reports the layer's size with transparent
/// pixels, so the layer is present and its loads decode to the zero flow ([`load_flow`]). That is
/// the plugin's behaviour under the harness's answer for a checkout outside a clip
/// (`max_result_rect` at the layer size, an empty `result_rect`; a documented assumption of the
/// harness): MEASURED against `TwixtorProVectorsIn_64` under the emulator, past the end of short
/// vector clips the plugin renders the frames
/// with a zero flow, not the error fill.
pub fn present_at(layer: Option<&dyn Layer>, time: i64) -> Result<Option<(usize, usize)>> {
    if !layer_is_set(layer) {
        return Ok(None);
    }
    let Some(layer) = layer else { return Ok(None) };
    let size = match layer.frame_at(time)? {
        Some(f) => (f.width, f.height),
        None => (layer.width(), layer.height()),
    };
    Ok(Some(size).filter(|&(w, h)| w > 0 && h > 0))
}

/// `Vectors: Max Displace` at `t0` as the loaders see it: the 16.16 reader (scale 1.0f) times the
/// horizontal downsample ratio `(f32)num / (f32)den`, which is 1.0 here because renders are always
/// at full resolution.
pub fn max_displace(curves: &ParamCurves, t0: i64) -> f32 {
    let value = fixed_reader(curves.fixed_at("vectors_max_displace", t0), 1.0f32);
    (1.0f32 / 1.0f32) * value
}

/// `Vec Scale X` and `Vec Scale Y` at `time`: `(f32)((f64)raw * 2^-16)` each, no further factor.
pub fn vec_scale(curves: &ParamCurves, time: i64) -> (f32, f32) {
    (fixed_reader(curves.fixed_at("vec_scale_x", time), 1.0f32), fixed_reader(curves.fixed_at("vec_scale_y", time), 1.0f32))
}

/// One loader: the flow of the vector layer `layer` (called `name` in messages) at layer time
/// `time`.
///
/// A frame of another size or depth than the render is an error. A time at which the layer has no
/// frame gives a transparent image, which decodes to the zero flow (alpha 0 means a zero vector,
/// and an all hole image fills to zeros).
pub fn load_flow(
    layer: &dyn Layer,
    name: &str,
    time: i32,
    settings: &EngineSettings,
    max_displace: f32,
    scale: (f32, f32),
) -> Result<FlowField> {
    flow_from_vector_image_with(&load_image(layer, name, time, settings)?, max_displace, scale, settings.stages.get().vertical_taps())
}

/// The engine image of the vector layer `layer` at `time` (see [`load_flow`]).
fn load_image(layer: &dyn Layer, name: &str, time: i32, settings: &EngineSettings) -> Result<Image> {
    Ok(match layer.frame_at(i64::from(time))? {
        Some(frame) => {
            if (frame.width, frame.height) != (settings.width, settings.height) {
                return Err(FxError::Mismatch(format!(
                    "{name} has a {}x{} frame at time {time} but the output is {}x{}",
                    frame.width, frame.height, settings.width, settings.height
                )));
            }
            if frame.depth() != settings.fmt.depth() {
                return Err(FxError::Mismatch(format!(
                    "{name} is {} bpc but the output is {} bpc",
                    frame.depth().bits(),
                    settings.fmt.depth().bits()
                )));
            }
            Image::from_frame(&frame)
        }
        None => Image::new(settings.width, settings.height, settings.fmt),
    })
}

/// The per layer matte of a vector load (READ, TwixtorProVectorsIn_64: the per layer
/// vector loader, its two loads and its row kernels at 8 bit, 16 bit and float): per pixel the
/// alpha of the vector image becomes `c` where `c < alpha`, with `c` the layer's matte `m` in the
/// units of the depth (`trunc(m * 255 + 0.5)`, `trunc(m * 65535 + 0.5)`, the float itself); a NaN
/// `c` keeps the alpha; the colour stays. The layered path hands the layer's composition matte
/// (`retime_engine::layers::LayerMattes`), whose units come back exactly. The decode then takes
/// `white - alpha` as the hole mask of the flow fill: whole holes where the layer is absent,
/// partial ones where its matte is soft. `m` has the size of `img` (otherwise
/// `FxError::Mismatch`).
pub fn matte_step(img: &mut Image, m: &Matte) -> Result<()> {
    if (m.width, m.height) != (img.width, img.height) || m.data.len() != img.width * img.height {
        return Err(FxError::Mismatch(format!(
            "the matte step: a {}x{} matte for a {}x{} vector image",
            m.width, m.height, img.width, img.height
        )));
    }
    match &mut img.samples {
        Samples::U8(s) => s.chunks_exact_mut(CHANNELS).zip(&m.data).for_each(|(px, &v)| {
            let c = ((v * 255.0f32 + 0.5f32) as i64).clamp(0, 255) as u8;
            if c < px[A] {
                px[A] = c;
            }
        }),
        Samples::U16(s) => s.chunks_exact_mut(CHANNELS).zip(&m.data).for_each(|(px, &v)| {
            let c = ((v * 65535.0f32 + 0.5f32) as i64).clamp(0, 65535) as u16;
            if c < px[A] {
                px[A] = c;
            }
        }),
        Samples::F32(s) => s.chunks_exact_mut(CHANNELS).zip(&m.data).for_each(|(px, &v)| {
            if v < px[A] {
                px[A] = v;
            }
        }),
    }
    Ok(())
}

/// One loader of a layer: [`load_flow`] with the [`matte_step`] of the layer's matte `m` at the
/// load's frame between the fetch and the decode.
pub fn load_layer_flow(
    layer: &dyn Layer,
    name: &str,
    time: i32,
    settings: &EngineSettings,
    max_displace: f32,
    scale: (f32, f32),
    m: &Matte,
) -> Result<FlowField> {
    let mut image = load_image(layer, name, time, settings)?;
    matte_step(&mut image, m)?;
    flow_from_vector_image_with(&image, max_displace, scale, settings.stages.get().vertical_taps())
}

/// The loaders of one layer of a layered render: as [`load_flows`], with `F_AB` from
/// `Vectors: Next` at `T(A)` capped by the layer's matte at `A` (`mattes.at_a`) and `F_BA` from
/// `Vectors: Prev` at `T(B)` by its matte at `B` ([`load_layer_flow`]).
pub fn load_layer_flows(
    env: &Env<'_>,
    settings: &EngineSettings,
    map: &dyn TimeMap,
    sel: &Selection,
    has_prev: bool,
    has_next: bool,
    mattes: &LayerMattes<'_>,
) -> Result<Flows> {
    let curves = env.params;
    let m = max_displace(curves, i64::from(settings.current_time));
    let mut flows = Flows { f_ab: None, f_ba: None };
    if has_next {
        let layer = env.layer(LAYER_VECTORS_NEXT).ok_or_else(|| missing(LAYER_VECTORS_NEXT))?;
        let time = map.frame_time(sel.a as f32);
        flows.f_ab = Some(load_layer_flow(layer, "Vectors: Next", time, settings, m, vec_scale(curves, i64::from(time)), mattes.at_a)?);
    }
    if has_prev {
        let layer = env.layer(LAYER_VECTORS_PREV).ok_or_else(|| missing(LAYER_VECTORS_PREV))?;
        let time = map.frame_time(sel.b as f32);
        flows.f_ba = Some(load_layer_flow(layer, "Vectors: Prev", time, settings, m, vec_scale(curves, i64::from(time)), mattes.at_b)?);
    }
    Ok(flows)
}

/// The loaders for the frame pair of `sel`: `F_AB` from `Vectors: Next` at `T(A)` first, then
/// `F_BA` from `Vectors: Prev` at `T(B)`. A layer that was not present at `t0` gives `None` (the
/// engine inverts the other one).
pub fn load_flows(
    env: &Env<'_>,
    settings: &EngineSettings,
    map: &dyn TimeMap,
    sel: &Selection,
    has_prev: bool,
    has_next: bool,
) -> Result<Flows> {
    let curves = env.params;
    let m = max_displace(curves, i64::from(settings.current_time));
    let mut flows = Flows { f_ab: None, f_ba: None };
    if has_next {
        let layer = env.layer(LAYER_VECTORS_NEXT).ok_or_else(|| missing(LAYER_VECTORS_NEXT))?;
        let time = map.frame_time(sel.a as f32);
        flows.f_ab = Some(load_flow(layer, "Vectors: Next", time, settings, m, vec_scale(curves, i64::from(time)))?);
    }
    if has_prev {
        let layer = env.layer(LAYER_VECTORS_PREV).ok_or_else(|| missing(LAYER_VECTORS_PREV))?;
        let time = map.frame_time(sel.b as f32);
        flows.f_ba = Some(load_flow(layer, "Vectors: Prev", time, settings, m, vec_scale(curves, i64::from(time)))?);
    }
    Ok(flows)
}

fn missing(id: &str) -> FxError {
    FxError::Mismatch(format!("the layer `{id}` was probed as present but the environment has none"))
}
