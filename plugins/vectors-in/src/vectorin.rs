//! The Vectors In path of plugin 04: the two vector layers become the two flow fields the core
//! composes with.
//!
//! 1. Presence is decided once per call at `t0 = current_time`: a vector layer counts as present
//!    when its checkout has a width and a height above 0. [`present_at`] is that probe (a layer
//!    that has no frame at `t0` is absent for the whole render).
//! 2. The core asks for the flows only on the vector path (`vectors_quality != 0 && !exact` after
//!    the segment override). It loads, in this order, `F_AB` from the layer `Vectors: Next` at
//!    frame `A` and `F_BA` from `Vectors: Prev` at frame `B = A + 1`. The fetch time of frame `k`
//!    is `T(k) = trunc(dur * k + 0.5)` ([`crate::timemap`]).
//! 3. Each loader decodes its image with `Max Displace` (read once at `t0`, scaled by the
//!    horizontal downsample ratio, which is 1 here), fills the alpha 0 holes of the flow with the
//!    pull push fill and multiplies by `Vec Scale X / Y` read at the same `T(k)`
//!    ([`retime_engine::compose::flow_from_vector_image`]).
//! 4. With exactly one layer present the other flow is its inversion; with none the render writes
//!    the error fill before any engine call (`glue.rs`).
//!
//! The vector image goes through the premultiplying prepare like a source frame
//! (`Image::from_frame`), so a vector layer has to be opaque except for alpha 0 holes: for alpha
//! below full the premultiply scales the encoded values and the decode is wrong.

use fx_core::{Env, FxError, Layer, ParamCurves, Result};
use retime_engine::compose::{flow_from_vector_image, Flows};
use retime_engine::flow::FlowField;
use retime_engine::frontend::fixed_reader;
use retime_engine::host::{layer_is_set, EngineSettings};
use retime_engine::image::Image;
use retime_engine::timeline::{Selection, TimeMap};

use crate::prerender::{LAYER_VECTORS_NEXT, LAYER_VECTORS_PREV};

/// The probe of a vector layer at `t0`: the size of the checkout, `None` when the layer is not set,
/// has no pixels or has no frame at `time`. A layer that returns `None` here is absent for the
/// whole render, whatever it holds at other times.
pub fn present_at(layer: Option<&dyn Layer>, time: i64) -> Result<Option<(usize, usize)>> {
    if !layer_is_set(layer) {
        return Ok(None);
    }
    let Some(layer) = layer else { return Ok(None) };
    Ok(layer.frame_at(time)?.filter(|f| f.width > 0 && f.height > 0).map(|f| (f.width, f.height)))
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
    let image = match layer.frame_at(i64::from(time))? {
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
    };
    flow_from_vector_image(&image, max_displace, scale)
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
