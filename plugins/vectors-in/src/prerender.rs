//! The pre-render checkout list of plugin 04, with or without FG matte layers.
//!
//! Layers: the input layer, `Source` (displayed as "Color Source"), `Vectors: Prev`, `Vectors: Next` and the FG
//! matte layers.
//! Plugin 04 has no Alt Source A or B and no mask hooks.
//!
//! The list, in issue order:
//!
//! 1. the checkout of the input layer at `X`;
//! 2. the probes at `X`: `Source`, `Vectors: Prev`, `Vectors: Next`, then FG1, FG2, FG3 matte up to
//!    and including the first empty one;
//! 3. `Display == Source`: `Source` at `X` once more and nothing else;
//! 4. otherwise the plan (`planner::plan` with `t0 = X / lts`, `t1 = (ts + X) / lts`, vectors
//!    quality constant 4, no layer B) and for each entry `k` with `off = T(frame[k])`:
//!    * with `register_layers` set: the dummy registrations, per layer a pair `X + off` and `-X`, in the
//!      order `Source`, `Vectors: Prev`, `Vectors: Next`, then the three FG matte slots whether
//!      used or not;
//!    * the entry's own checkouts: for a main entry `Source` at `off`, then `Vectors: Prev` and
//!      `Vectors: Next` at `off`, the pair `nfg + 1` times (one load per layer; the order
//!      of the repeats, a pair per layer, is U); for a kind 2 entry (FG matte `k`, after the main
//!      entries, `planner::plan_layers`) the matte layer at `off`.
//!
//! `T(k) = trunc(dur * k + 0.5)` is the time map's `frame_time` ([`crate::timemap`]). Registrations
//! that are not pixel inputs are not listed.
//!
//! The retimed layer is the `source` layer when it is set and the input layer otherwise, so entries
//! that name the source use `Some("source")` or `None` accordingly ([`PreRender04::src_layer`]);
//! the probe of the layer parameter itself always names `Some("source")`.

use fx_core::{Checkout, Result};
use retime_engine::planner::{plan_layers, EntryKind};
use retime_engine::timeline::{frame_positions, TimeMap};

/// Layer id of `Source`, displayed as "Color Source".
pub const LAYER_SOURCE: &str = "source";
/// Layer id of `Vectors: Prev`: the flow of frame `B` back to `A` (`F_BA`).
pub const LAYER_VECTORS_PREV: &str = "vectors_prev";
/// Layer id of `Vectors: Next`: the flow of frame `A` forward to `B` (`F_AB`).
pub const LAYER_VECTORS_NEXT: &str = "vectors_next";
/// Layer ids of the three FG matte layers.
pub const LAYER_FG_MATTE: [&str; 3] = ["fg1_matte", "fg2_matte", "fg3_matte"];

/// The vectors quality constant plugin 04 passes to the planner and the core. Only its being non
/// zero matters (the segment override of `timeline::select` can lower it to 0).
///
/// Tiny frames (U, ours, parity gap G7): the render passes 0 instead for a frame with a side of 1
/// (`retime_engine::prep::searched_quality`), which takes the fast path. The checkout list keeps
/// this constant: without layer B the plan entries do not depend on the quality, so the list of
/// such a frame is the one of a frame whose segment marker turns the vectors off.
pub const VECTORS_QUALITY: i32 = 4;

/// What the pre-render of one output frame reads before it registers checkouts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreRender04 {
    /// The time of the output frame in ticks (`X`).
    pub current_time: i32,
    /// Ticks per output frame.
    pub time_step: i32,
    /// Ticks per frame of the layer.
    pub local_time_step: i32,
    /// `Display == Source` (popup value 2).
    pub display_source: bool,
    /// The `Source` layer is set (else the input layer plays its part in the harness).
    pub source_present: bool,
    /// Number of FG matte layers: the consecutive present layers starting at FG1.
    pub fg_count: usize,
    /// Frame Interp popup minus one.
    pub interp: i32,
    /// Blur Compensation.
    pub blur: f32,
    /// Whether the dummy registrations of every plan entry are listed (see [`checkouts`]).
    pub register_layers: bool,
}

impl PreRender04 {
    /// The layer the retimed frames come from: the `source` layer when set, else the input (`None`).
    pub fn src_layer(&self) -> Option<&'static str> {
        self.source_present.then_some(LAYER_SOURCE)
    }
}

fn co(layer: Option<&'static str>, time: i32) -> Checkout {
    Checkout { layer, time: i64::from(time) }
}

/// The probes at `X`: the input layer, `Source`, `Vectors: Prev`, `Vectors: Next`, then the FG
/// mattes up to and including the first empty one.
pub fn probes(req: &PreRender04) -> Vec<Checkout> {
    let cur = req.current_time;
    let mut out = vec![co(None, cur), co(Some(LAYER_SOURCE), cur), co(Some(LAYER_VECTORS_PREV), cur), co(Some(LAYER_VECTORS_NEXT), cur)];
    for k in 0..LAYER_FG_MATTE.len() {
        out.push(co(Some(LAYER_FG_MATTE[k]), cur));
        if k >= req.fg_count {
            break;
        }
    }
    out
}

/// The complete list of registrations of one output frame, in issue order (module documentation).
pub fn checkouts(req: &PreRender04, map: &dyn TimeMap) -> Result<Vec<Checkout>> {
    let mut out = probes(req);
    let cur = req.current_time;
    let src = req.src_layer();
    if req.display_source {
        out.push(co(src, cur));
        return Ok(out);
    }
    let (t0, t1) = frame_positions(cur, req.time_step, req.local_time_step);
    let p = plan_layers(map, t0, t1, VECTORS_QUALITY, req.interp, req.blur, true, req.fg_count)?;
    let dummies: [Option<&'static str>; 6] = [
        src,
        Some(LAYER_VECTORS_PREV),
        Some(LAYER_VECTORS_NEXT),
        Some(LAYER_FG_MATTE[0]),
        Some(LAYER_FG_MATTE[1]),
        Some(LAYER_FG_MATTE[2]),
    ];
    for entry in &p.entries {
        let off = map.frame_time(entry.position as f32);
        if req.register_layers {
            for layer in dummies {
                out.push(co(layer, cur.wrapping_add(off)));
                out.push(co(layer, cur.wrapping_neg()));
            }
        }
        match entry.kind {
            EntryKind::Main => {
                out.push(co(src, off));
                // the two vector layers once per layer: nfg + 1 times
                for _ in 0..=req.fg_count {
                    out.push(co(Some(LAYER_VECTORS_PREV), off));
                    out.push(co(Some(LAYER_VECTORS_NEXT), off));
                }
            }
            // no layer B in plugin 04 (`noB` = 1): the planner emits no such entry
            EntryKind::LayerB => out.push(co(None, off)),
            // plan_layers gives at most three mattes
            EntryKind::Matte(k) => out.push(co(Some(LAYER_FG_MATTE[k]), off)),
        }
    }
    Ok(out)
}
