//! The pre-render checkout list of plugin 03, for the single layer configuration.
//!
//! Differences from plugin 01's list (`retime_engine::planner::checkouts`):
//!
//! * the probes also cover the FG matte layers (stopping at the first empty one);
//! * `Display == Source` registers one more checkout of Alt Source A at `X`, always Alt Source A
//!   and never the input layer, and has no `Src Time Offset`;
//! * when `register_layers` is set, every plan entry first registers Alt Source A, Alt Source B and the
//!   three FG matte layers at `X + off` and at `-X` (and again at `X + off2` with input fields),
//!   before its own checkout;
//! * the times of the plan checkouts are absolute layer times `T = trunc(f58 * n + 0.5)` (no
//!   `base`).
//!
//! Registrations that are not pixel inputs are not listed.

use fx_core::{Checkout, Result};
use retime_engine::planner::{plan, EntryKind, LAYER_ALT_A, LAYER_ALT_B};
use retime_engine::timeline::{frame_positions, TimeMap};

/// Layer ids of the three FG matte layers.
pub const LAYER_FG_MATTE: [&str; 3] = ["fg1_matte", "fg2_matte", "fg3_matte"];

/// What the pre-render of one output frame reads before it registers checkouts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreRender03 {
    /// The time of the output frame in ticks (`X`).
    pub current_time: i32,
    /// Ticks per output frame.
    pub time_step: i32,
    /// Ticks per frame of the layer.
    pub local_time_step: i32,
    /// `Display == Source` (popup value 2).
    pub display_source: bool,
    /// Alt Source A is set and has pixels (the probe of layer 13 gives a non empty size).
    pub alt_a_present: bool,
    /// Alt Source B is set and has pixels (`no_alt_b` is 0).
    pub alt_b_present: bool,
    /// Number of FG matte layers: the consecutive present layers starting at FG1 (the probe loop stops at the
    /// first empty one, so FG2 without FG1 does not count).
    pub fg_count: usize,
    /// `Input: Fields` is not None.
    pub input_fields: bool,
    /// Vectors popup minus one.
    pub vectors_quality: i32,
    /// Frame Interp popup minus one.
    pub interp: i32,
    /// Blur Compensation.
    pub blur: f32,
    /// Whether the registrations of the five layers before every plan entry are listed (see
    /// [`checkouts`]).
    pub register_layers: bool,
}

fn co(layer: Option<&'static str>, time: i32) -> Checkout {
    Checkout { layer, time: i64::from(time) }
}

/// The size probes, preceded by the checkout of the input layer: the input layer at `X`; Alt Source
/// A at `X`, and the input layer again when A has no pixels; Alt Source B at `X`; the FG matte
/// layers at `X` in order, up to and including the first empty one.
pub fn probes(req: &PreRender03) -> Vec<Checkout> {
    let cur = req.current_time;
    let mut out = vec![co(None, cur), co(Some(LAYER_ALT_A), cur)];
    if !req.alt_a_present {
        out.push(co(None, cur));
    }
    out.push(co(Some(LAYER_ALT_B), cur));
    for k in 0..LAYER_FG_MATTE.len() {
        out.push(co(Some(LAYER_FG_MATTE[k]), cur));
        if k >= req.fg_count {
            break;
        }
    }
    out
}

/// The complete list of registrations of one output frame, in issue order.
///
/// 1. [`probes`];
/// 2. `Display == Source`: one checkout of Alt Source A at `X` (even when A is empty), then nothing
///    else;
/// 3. otherwise the plan (`planner::plan` with `t0 = X / lts` and `t1 = (ts + X) / lts`) and for
///    every entry `k`, with `off = T(t_k)` and `off2 = T(t_k + 0.5)` (`map.frame_time`): with
///    `register_layers` set, Alt Source A, Alt Source B and the three FG matte layers each at `X + off` and
///    then at `-X`, with input fields the five layers again at `X + off2`; then the entry's own
///    checkout at `T(t_k)` (Alt Source A, or the input layer when A has no pixels, for the main
///    frames; Alt Source B for the tracking frames) and with input fields at `T(t_k + 0.5)`.
///
/// With FG matte layers and `Display != Source` the plan of the multi layer path is not known, so
/// the list is an `FxError::Unsupported` instead of a guess.
pub fn checkouts(req: &PreRender03, map: &dyn TimeMap, multi_layer_message: &str) -> Result<Vec<Checkout>> {
    let mut out = probes(req);
    let cur = req.current_time;
    if req.display_source {
        out.push(co(Some(LAYER_ALT_A), cur));
        return Ok(out);
    }
    if req.fg_count > 0 {
        return Err(fx_core::FxError::Unsupported(multi_layer_message.to_string()));
    }
    let (t0, t1) = frame_positions(cur, req.time_step, req.local_time_step);
    let p = plan(map, t0, t1, req.vectors_quality, req.interp, req.blur, !req.alt_b_present)?;
    let five: [Option<&'static str>; 5] =
        [Some(LAYER_ALT_A), Some(LAYER_ALT_B), Some(LAYER_FG_MATTE[0]), Some(LAYER_FG_MATTE[1]), Some(LAYER_FG_MATTE[2])];
    for entry in &p.entries {
        let t = entry.position as f32;
        // `(f32)t_k + 0.5` is formed in double and narrowed to f32.
        let half = ((entry.position as f64) + 0.5) as f32;
        if req.register_layers {
            let off = map.frame_time(t);
            for layer in five {
                out.push(co(layer, cur.wrapping_add(off)));
                out.push(co(layer, cur.wrapping_neg()));
            }
            if req.input_fields {
                let off2 = map.frame_time(half);
                for layer in five {
                    out.push(co(layer, cur.wrapping_add(off2)));
                }
            }
        }
        let layer = match entry.kind {
            EntryKind::Main => req.alt_a_present.then_some(LAYER_ALT_A),
            EntryKind::LayerB => Some(LAYER_ALT_B),
        };
        out.push(co(layer, map.frame_time(t)));
        if req.input_fields {
            out.push(co(layer, map.frame_time(half)));
        }
    }
    Ok(out)
}
