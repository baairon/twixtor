//! The tracking points (`ptN_use`, `ptN_pos`) as an input of the layered solves: [`point_layer`]
//! maps a `Pt N: Use` value to its layer, [`read_pos`] reads a point,
//! [`user_displacement`] turns point pairs into the user displacement of every pyramid level (the
//! points producer, rule G2) and [`second_tracker`] is the displacement interpolator of a layer whose
//! Track is off (rule G5).
//!
//! Claim tags: V established, I inferred, U not established (a rule of this project, pinned in
//! the parity gaps register until a reference render grades it).
//!
//! Sign: a pair is `(from, to)`, the point's position in the solve's frame `X` and in its frame
//! `Y`. The solver's own displacement is the negation of its output (`calc_b` samples `Y` at
//! `p - d`, and `solve_pyramids` negates at the end), so the user displacement it is set to at a
//! point is `from - to` (the solver's `flow = d / w` runs on that internal value), while
//! [`second_tracker`], which replaces a solve, returns the output convention
//! `X(p) ~ Y(p + flow(p))`, about `to - from` at the point.
//!
//! Precision contract: f32 evaluated in the order written, no fused multiply add, the points
//! accumulated in their order. At a level of `w_l x h_l` for an output of `W x H`:
//! `kx = (w_l - 1) as f32 / (W - 1) as f32` and `ky = (h_l - 1) as f32 / (H - 1) as f32`; the
//! pixel is `ix = cvtt(from_x * kx + 0.5)`, `iy = cvtt(from_y * ky + 0.5)` (truncation); then
//! `d.x = d.x + (from_x - to_x) * kx`, `d.y = d.y + (from_y - to_y) * ky` and `w = w + 1.0`.
//! The second tracker divides `d / w` at the pixels that got a point, runs the hole fill
//! `holefill::fill_flow_holes` (its own contract: rayon row chunks, the same result for any
//! thread count) and multiplies by `-1.0` as the solver does. A position is read with the 16.16
//! reader `((float_to_fixed(v) as f64) * (1.0 / 65536.0)) as f32` per axis.

use crate::flow::extras::{UserDisplacement, UserLevel};
use crate::flow::sample::cvtt;
use crate::flow::{FlowField, FlowGeometry};
use crate::frontend::fixed_reader;
use crate::holefill::{fill_flow_holes, HoleMaskRef};
use fx_core::{float_to_fixed, ParamCurves, Result};

#[cfg(test)]
mod tests;

/// One tracking point in use, at the times of the two frames of the pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UserPoint {
    /// The position at the layer time of frame `A`, in pixels of the output.
    pub at_a: [f32; 2],
    /// The position at the layer time of frame `B`.
    pub at_b: [f32; 2],
}

impl UserPoint {
    /// The pair `(from X, to Y)` of [`user_displacement`] and [`second_tracker`] for a solve whose
    /// frame `X` is frame `A` (`x_is_a`, the solve `A(p) ~ B(p + f_ab(p))`) or frame `B`.
    pub fn pair(&self, x_is_a: bool) -> ([f32; 2], [f32; 2]) {
        if x_is_a {
            (self.at_a, self.at_b)
        } else {
            (self.at_b, self.at_a)
        }
    }
}

/// The layer a `Pt N: Use` value puts its point on (1 Don't Use, 2 FG1, 3 FG2, 4 FG3,
/// 5 Main_BG): `None` for Don't Use (and any value below 2), else layer `min(value - 2, nfg)`, so
/// a value naming a layer that does not exist goes to the last layer, Main_BG (index `nfg`), and
/// with `nfg = 0` every value 2 to 5 is the only group. This is create-vectors' `point_belongs`:
/// a point belongs to exactly the layer returned here.
pub fn point_layer(use_value: i64, nfg: usize) -> Option<usize> {
    if use_value < 2 {
        return None;
    }
    let target = (use_value - 2) as u64;
    Some(if target > nfg as u64 { nfg } else { target as usize })
}

/// The point parameter `id` at `time` ticks, read the way the engine reads it: each axis
/// through the 16.16 reader, `((float_to_fixed(v) as f64) * (1.0 / 65536.0)) as f32`. The value
/// is in pixels of the output at the downsample ratio 1 (the harness has no downsampled
/// renders). Panics on an unknown id or a parameter that is not a point, as
/// `ParamCurves::point_at` does.
pub fn read_pos(c: &ParamCurves, id: &str, time: i32) -> [f32; 2] {
    let p = c.point_at(id, i64::from(time));
    [fixed_reader(float_to_fixed(p[0]), 1.0f32), fixed_reader(float_to_fixed(p[1]), 1.0f32)]
}

/// Whether the position `p` (output pixels, pixel centres at whole numbers) lies in the output
/// frame of `out = (W, H)`: its nearest pixel exists, `0.0 <= p + 0.5 < W` on x (f32) and the same
/// with `H` on y. A NaN or infinite position is outside. U (ours, rule G2: "points outside the
/// frame are skipped").
pub fn in_frame(p: [f32; 2], out: (usize, usize)) -> bool {
    let fx = p[0] + 0.5f32;
    let fy = p[1] + 0.5f32;
    fx >= 0.0 && fx < out.0 as f32 && fy >= 0.0 && fy < out.1 as f32
}

/// The splat of the pairs into one level of `wl x hl` (see [`user_displacement`]).
fn splat(pairs: &[([f32; 2], [f32; 2])], wl: usize, hl: usize, out: (usize, usize)) -> UserLevel {
    let mut lvl = UserLevel { d: vec![0.0f32; wl * hl * 2], w: vec![0.0f32; wl * hl] };
    if out.0 < 2 || out.1 < 2 || wl == 0 || hl == 0 {
        return lvl;
    }
    let kx = (wl - 1) as f32 / (out.0 - 1) as f32;
    let ky = (hl - 1) as f32 / (out.1 - 1) as f32;
    for &(from, to) in pairs {
        if !in_frame(from, out) || !(to[0].is_finite() && to[1].is_finite()) {
            continue;
        }
        let ix = cvtt(from[0] * kx + 0.5f32);
        let iy = cvtt(from[1] * ky + 0.5f32);
        // an in frame point lands inside every level (both scales are at most 1); kept as a guard
        if ix < 0 || iy < 0 || ix as usize >= wl || iy as usize >= hl {
            continue;
        }
        let i = iy as usize * wl + ix as usize;
        lvl.d[2 * i] = lvl.d[2 * i] + (from[0] - to[0]) * kx;
        lvl.d[2 * i + 1] = lvl.d[2 * i + 1] + (from[1] - to[1]) * ky;
        lvl.w[i] = lvl.w[i] + 1.0f32;
    }
    lvl
}

/// The user displacement of a layer from its point pairs `(from X, to Y)` in output pixels (`out`
/// is the output size `(W, H)`), one [`UserLevel`] per level of the solver geometry `geo`,
/// finest first (the points producer, U; rule G2). Per level, a nearest pixel splat in point order:
/// a pair whose `from` lies outside the output frame ([`in_frame`]) or whose `to` is not finite is
/// skipped (U, ours); else at the pixel `(cvtt(from_x * kx + 0.5), cvtt(from_y * ky + 0.5))` the
/// displacement grows by `((from_x - to_x) * kx, (from_y - to_y) * ky)` and the weight by 1.0, with
/// `kx = (w_l - 1) / (W - 1)` and `ky = (h_l - 1) / (H - 1)` (the corner aligned scale of the
/// solver's flow upsampling; ky is the vertical twin of the contract's `k_l`, U). Two points on
/// one pixel add up, so the consumer `flow = d / w` takes their mean. The values are in the
/// solver's own sign (see the module documentation). An output with a side below 2 (never
/// searched) gives empty levels. Records gap G2 when there is at least one pair.
pub fn user_displacement(pairs: &[([f32; 2], [f32; 2])], geo: &FlowGeometry, out: (usize, usize)) -> UserDisplacement {
    if !pairs.is_empty() {
        fx_core::gaps::record(&crate::gaps::G2);
    }
    UserDisplacement { levels: geo.sizes.iter().map(|&(wl, hl)| splat(pairs, wl, hl, out)).collect() }
}

/// The flow of a layer whose Track is off (role V, arithmetic U; rule G5): the displacement
/// interpolated from its point pairs `(from X, to Y)` in output pixels (`out` the output size)
/// instead of an optical flow estimate, at the working resolution `geo.width x geo.height`. The
/// level 0 splat of [`user_displacement`] gives `d / w` at every pixel with a point; the other
/// pixels are holes of a u8 mask (255), filled by the pull-push fill `holefill::fill_flow_holes`
/// (no 360 wrap), with no further blur; the field is then multiplied by `-1.0` into the solver's
/// output convention `X(p) ~ Y(p + flow(p))`, so a pixel with one point holds exactly
/// `(to - from) * k`. Without a point that lands (none given, or all outside the frame) the
/// result is the exact zero field, `FlowField::new`; one point gives a near uniform field.
/// Records gap G5. Errors: those of the fill (a working size below 2x2 with holes is
/// `FxError::Unsupported`; `FlowSettings::geometry` never gives one).
pub fn second_tracker(pairs: &[([f32; 2], [f32; 2])], geo: &FlowGeometry, out: (usize, usize)) -> Result<FlowField> {
    fx_core::gaps::record(&crate::gaps::G5);
    let (w0, h0) = (geo.width, geo.height);
    let lvl = splat(pairs, w0, h0, out);
    let mut flow = FlowField::new(w0, h0);
    if !lvl.w.iter().any(|&v| v != 0.0) {
        return Ok(flow);
    }
    let mut holes = vec![0u8; w0 * h0];
    for (i, &wt) in lvl.w.iter().enumerate() {
        if wt != 0.0 {
            flow.data[2 * i] = lvl.d[2 * i] / wt;
            flow.data[2 * i + 1] = lvl.d[2 * i + 1] / wt;
        } else {
            holes[i] = 255;
        }
    }
    fill_flow_holes(&mut flow, HoleMaskRef::U8(&holes), false)?;
    flow.scale(-1.0f32);
    Ok(flow)
}
