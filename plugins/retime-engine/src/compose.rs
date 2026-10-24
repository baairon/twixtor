//! The render of one output frame: the error fills, the `Display == Source` copy and the retiming
//! core that picks the frames around the source position and combines them.
//!
//! Two paths. The fast path: the position lies on a frame (or Vectors is "No Vectors" after the
//! segment override), so the output is a copy or a crossfade of the two frames around the source
//! position, with no flow. The vector path ([`vector`]): optical flow motion interpolation, the
//! warps for the warping codes 0 to 3 in the three Frame Interp modes and the blur compensation
//! tail. What it cannot serve is an error, never a silent fallback.

use crate::blend::{blend, copy, weighted_blend_noflow};
use crate::host::{EngineHost, EngineSettings};
use crate::image::{error_frame, Image};
use crate::timeline::{frame_positions, select, source_display_time, Selection};
use fx_core::{Frame, FxError, Result};

mod vector;

pub use vector::{
    check_supported, compose_vectors, flow_from_vector_image, flow_settings, invert_flow, nearest_uses_f_ab,
    solve_flows, Flows,
};

/// What one render produces.
#[derive(Clone, Debug, PartialEq)]
pub enum Rendered {
    /// The engine image, PREMULTIPLIED (the engine domain). The output finish (`Image::to_frame`)
    /// still has to run.
    Image(Image),
    /// An error fill (`image::error_frame`), already a wire frame: it is the output as it is and
    /// skips the finish.
    Error(Frame),
}

impl Rendered {
    /// The wire frame: the output finish for an image, the fill itself for an error.
    pub fn into_frame(self) -> Frame {
        match self {
            Rendered::Image(image) => image.to_frame(),
            Rendered::Error(frame) => frame,
        }
    }
}

/// The error fill for the flags the host has set so far, if any: red for a size mismatch first,
/// then the missing layer fill.
fn issue_fill(settings: &EngineSettings, host: &dyn EngineHost) -> Option<Rendered> {
    let kind = host.issues().error_kind(settings.missing_layer_style)?;
    Some(Rendered::Error(error_frame(kind, settings.width, settings.height, settings.fmt.depth())))
}

/// Renders output frame `settings.current_time`:
/// 1. the error flags of the probes: size mismatch gives the half alpha red fill, a missing layer
///    the plugin's fill (half blue for plugin 01), both without the finish step and as a success;
/// 2. `Display == Source`: the frame of the retimed layer at [`source_display_time`]
///    (`current_time + trunc(Src Time Offset) * time_step`) goes through the premultiply and, in
///    the finish, the unpremultiply (a lossy round trip below full alpha);
/// 3. otherwise the core ([`retime`]).
///
/// Fetch problems found later (a frame of another size) turn into the same fills.
pub fn render(settings: &EngineSettings, host: &dyn EngineHost) -> Result<Rendered> {
    if let Some(fill) = issue_fill(settings, host) {
        return Ok(fill);
    }
    if settings.display_source {
        let time = source_display_time(settings.current_time, settings.time_step, settings.src_time_offset_fixed);
        let image = host.main_frame_at(time)?;
        return Ok(issue_fill(settings, host).unwrap_or(Rendered::Image(image)));
    }
    retime(settings, host)
}

/// The frame selection of the core: the output positions of this and the next frame
/// ([`frame_positions`]) and [`select`] with the host's time map, which calls `src_pos(t0)`,
/// `src_pos(t1)` and the segment markers in this order. The time map is left in the state the
/// later fetches need.
pub fn select_frames(settings: &EngineSettings, host: &dyn EngineHost) -> Result<Selection> {
    let (t0, t1) = frame_positions(settings.current_time, settings.time_step, settings.local_time_step);
    select(host.time_map(), t0, t1, settings.vectors_quality, settings.interp, settings.blur)
}

/// The retiming core:
/// 1. [`select_frames`]; when `vectors_quality != 0 && !exact` the vector path runs (optical
///    flow, warps, composition and blur compensation, the module [`vector`]) and its result is
///    the result; settings it cannot serve are an error;
/// 2. otherwise (the fast path) frame `a` and then frame `b = a + 1` are fetched at their layer
///    times;
/// 3. the output image is cleared and filled by [`interpolate`].
///
/// The output image is the engine image; the caller applies the finish. A size problem found by
/// the fetches gives the error fill instead.
pub fn retime(settings: &EngineSettings, host: &dyn EngineHost) -> Result<Rendered> {
    let sel = select_frames(settings, host)?;
    if sel.vectors_quality != 0 && !sel.exact {
        return vector::vector_path(settings, host, &sel);
    }
    fast_path(settings, host, &sel)
}

/// The core with the Vectors In branch (a front-end with vector layers): the same core as
/// [`retime`], but instead of running the optical flow solver it asks `flows` for the two flow
/// fields. Everything else is unchanged: the same [`select_frames`], the same fast path for exact
/// positions and `vectors_quality == 0` (the closure is not called there), the same warps,
/// composition modes and blur compensation tail.
///
/// `flows` receives the [`Selection`] (the frame pair `A = sel.a`, `B = sel.b`, the weight
/// `sel.wa`, the positions `s0`, `s1`) so the front-end can fetch `Vectors: Next` at frame `A`
/// (`F_AB`) and `Vectors: Prev` at frame `B` (`F_BA`) at the times the plugin's own time map
/// gives. It returns a [`Flows`]:
///
/// * both set: used as they are;
/// * exactly one set (only one vector layer exists): the missing one is the inversion of the
///   other ([`invert_flow`]: a forward warp of the negated vectors plus hole fill), derived only
///   when something reads it (the composition mode, or the blur tail for `F_AB`);
/// * neither set: `FxError::Mismatch`.
///
/// The flows are in pixels of the output image ([`flow_from_vector_image`] builds them from a
/// vector layer image); the warps resample a flow of any size of at least 2 x 2 and divide the
/// amounts by the size ratio, which is 1 here. Only the retimed layer's frame pair is fetched (no
/// tracking layer in this branch).
pub fn retime_with_flows<F>(settings: &EngineSettings, host: &dyn EngineHost, flows: F) -> Result<Rendered>
where
    F: FnOnce(&Selection) -> Result<Flows>,
{
    let sel = select_frames(settings, host)?;
    if sel.vectors_quality != 0 && !sel.exact {
        return vector::vector_path_with_flows(settings, host, &sel, flows);
    }
    fast_path(settings, host, &sel)
}

/// Steps 2 and 3 of [`retime`]: fetch `A` and `B`, clear the output, [`interpolate`].
fn fast_path(settings: &EngineSettings, host: &dyn EngineHost, sel: &Selection) -> Result<Rendered> {
    let a = host.main_frame(sel.a)?;
    let b = host.main_frame(sel.b)?;
    if let Some(fill) = issue_fill(settings, host) {
        return Ok(fill);
    }
    let mut out = Image::new(settings.width, settings.height, settings.fmt);
    interpolate(&mut out, sel, &a, &b)?;
    Ok(Rendered::Image(out))
}

/// The three modes of the fast path, on the interpolation mode the segment override left in
/// `sel.interp`:
/// - 0 Nearest: frame `a` when `wa >= 0.5`, else frame `b`, a plain copy;
/// - 1 Blend: `blend(a, b, (1.0f64 - wa as f64) as f32)` ([`Selection::blend_weight`]);
/// - 2 Weighted Blend without flows: `weighted_blend_noflow(a, wa, b)`, which equals the plain
///   Blend with the same weight.
///
/// All three images must have one size and format.
pub fn interpolate(out: &mut Image, sel: &Selection, a: &Image, b: &Image) -> Result<()> {
    match sel.interp {
        0 => copy(out, if sel.nearest_is_a() { a } else { b }),
        1 => blend(out, a, b, sel.blend_weight()),
        2 => weighted_blend_noflow(out, a, sel.wa, b),
        other => {
            return Err(FxError::BadParam {
                id: "frame_interp".into(),
                reason: format!("interpolation mode {other} is not 0 (Nearest), 1 (Blend) or 2 (Weighted Blend)"),
            })
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
