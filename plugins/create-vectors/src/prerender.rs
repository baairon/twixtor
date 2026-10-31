//! The checkout list of plugin 02: the layer fetches of one output frame, in issue order. `X` is
//! the current time.
//!
//! The list:
//!
//! 1. layer 0 at `X`;
//! 2. `Alt Motion Src` at `X`, and layer 0 at `X` again when `Alt Motion Src` has no pixels (the source becomes
//!    layer 0);
//! 3. the FG matte layers at `X` in order, up to and including the first empty one (FG k + 1 is
//!    ignored without FG k);
//! 4. `Display == Source`: the source at `X` once more, and nothing else; otherwise the source at
//!    `t_a` and at `t_b`, then every active FG matte at `t_a` and at `t_b` (FG1 first), where
//!    `(a, b)` is the frame pair ([`retime_engine::create_vectors::frame_pair`]) and `t = rnd(n)`
//!    is the frame time ([`retime_engine::create_vectors::frame_time`]).
//!
//! Unlike plugins 01 and 03, plugin 02 makes no extra registrations at `-X`.

use fx_core::Checkout;
use retime_engine::create_vectors::{frame_index, frame_pair, frame_time};

/// The id of `Alt Motion Src`. Plugins 01 and 03 call the layer that replaces the input as the source
/// `alt_source_a`; the role is the same here, so the id is shared (the parameter name is
/// `Alt Motion Src`).
pub const LAYER_ALT_SRC: &str = "alt_source_a";

/// Layer ids of `FG1: Matte`, `FG2: Matte` and `FG3: Matte`.
pub const LAYER_FG_MATTE: [&str; 3] = ["fg1_matte", "fg2_matte", "fg3_matte"];

/// What the pre-render of one output frame reads before it registers checkouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreRender02 {
    /// Current time in ticks (`X`).
    pub current_time: i32,
    /// Ticks per frame.
    pub time_step: i32,
    /// `Display == Source` (popup value 1).
    pub display_source: bool,
    /// `Display == Debug` (popup value 3).
    pub debug: bool,
    /// `Track Frame` popup value, 1 Previous, 2 Next.
    pub track_frame: i32,
    /// `Alt Motion Src` is set and has pixels.
    pub alt_present: bool,
    /// Number of active FG matte layers: the consecutive present layers starting at FG1.
    pub fg_count: usize,
}

fn co(layer: Option<&'static str>, time: i32) -> Checkout {
    Checkout { layer, time: i64::from(time) }
}

impl PreRender02 {
    /// The source layer the render reads: `Alt Motion Src` when it has pixels, else the input layer (`None`).
    pub fn source(&self) -> Option<&'static str> {
        self.alt_present.then_some(LAYER_ALT_SRC)
    }

    /// The frame numbers `(a, b)` of the pair.
    pub fn pair(&self) -> (i32, i32) {
        frame_pair(frame_index(self.current_time, self.time_step), self.track_frame, self.debug)
    }

    /// The layer times `(t_a, t_b)` of the pair.
    pub fn pair_times(&self) -> (i32, i32) {
        let (a, b) = self.pair();
        (frame_time(a, self.time_step), frame_time(b, self.time_step))
    }
}

/// Steps 1 to 3: layer 0, `Alt Motion Src` (and layer 0 again when it is empty) and the FG matte probes, all at `X`.
pub fn probes(req: &PreRender02) -> Vec<Checkout> {
    let cur = req.current_time;
    let mut out = vec![co(None, cur), co(Some(LAYER_ALT_SRC), cur)];
    if !req.alt_present {
        out.push(co(None, cur));
    }
    for (k, id) in LAYER_FG_MATTE.iter().enumerate() {
        out.push(co(Some(id), cur));
        if k >= req.fg_count {
            break;
        }
    }
    out
}

/// The complete list of registrations of one output frame, in issue order.
pub fn checkouts(req: &PreRender02) -> Vec<Checkout> {
    let mut out = probes(req);
    let src = req.source();
    if req.display_source {
        out.push(co(src, req.current_time));
        return out;
    }
    let (ta, tb) = req.pair_times();
    out.push(co(src, ta));
    out.push(co(src, tb));
    for id in LAYER_FG_MATTE.iter().take(req.fg_count) {
        out.push(co(Some(id), ta));
        out.push(co(Some(id), tb));
    }
    out
}
