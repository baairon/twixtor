//! The extra inputs of the solver for the layered paths and the wrap-around solve:
//! [`MaskPyramid`], the binary pyramid of a layer's effective matte, [`UserDisplacement`],
//! the user displacement of a layer's tracking points at every level, and
//! [`solve_pyramids_with`], the solver driver that takes them and `FlowSettings::wrap` (rule
//! G6).
//!
//! Precision contract: the solver's (`flow.rs`): f32 evaluated in the order written, no fused
//! multiply add, no reassociation, rows parallel through rayon row chunks only and no float sum
//! across threads, so the output is the same for any thread count. The mask pyramid uses the
//! pyramid filters of the luma pyramid (`prep::blur5`, `prep::resample`, their circular variants
//! under `FlowSettings::wrap`) and the comparison `v >= 0.95` in f32; the user displacement
//! consumer is one f32 division per channel. With `SolveExtras::default()` and wrap off,
//! [`solve_pyramids_with`] IS [`super::solve_pyramids`] (it calls it), so the plain solve keeps
//! its bytes.
//!
//! Claim tags: V established, I inferred, U not established (the rule of this project; gap G6
//! records where the wrap runs; the gated block matching runs on no effect's path).

use rayon::prelude::*;

use super::{
    all_flow_out_of_range, all_flow_out_of_range_y, block_match, block_match_with, calc_a, calc_b, calc_b_with, calc_grad,
    calc_grad_wrap, conv, mask_gradient, new_disp, scale_values, start_level, upsample_flow, upsample_flow_wrapx, FlowField,
    FlowGeometry, FlowSettings, Pyramid, MASK_ON,
};
use crate::matte::Matte;
use crate::prep::{self, Plane};
use fx_core::{FxError, Result};

#[cfg(test)]
mod tests_wrap;

/// The mask pyramid of one layer: its effective matte is binarized (`v >= 0.95` gives
/// 1, else 0), then for every reduction step to the working size and for every pyramid level:
/// copy, the 5 tap binomial blur, the corner aligned resample, binarize (no band pass). One
/// plane per level of the luma pyramid of the same geometry, finest first, holding only 0.0
/// and 1.0.
///
/// - U: the matte goes through the colour to luma path (alpha flag off) and is scaled to
///   0..1 first. A [`Matte`] is already the matte channel in 0..1, so that path is taken
///   as the identity. A matte of 8 or 16 bits binarizes the same either way (its values lie at
///   least 4e-6 from 0.95); a float matte within one rounding of 0.95 could differ.
/// - I: every binarize uses the same threshold, 0.95, compared in f32.
/// - The reduction steps halve as the luma reduction does (`(w / 2, h / 2)`), and the levels
///   take the sizes of `FlowSettings::geometry`, so level `l` has the size of the band pass
///   image of level `l`.
/// - With `FlowSettings::wrap` (G6, U, the rule of the luma pyramid in `flow::build_pyramid`)
///   every step's blur and resample are the circular ones (`prep::blur5_with`,
///   `prep::resample_with` with the flag: horizontal taps modulo the width, the resample keeps
///   the period), so a matte that crosses the seam gives the masks of the same matte rotated
///   away from it (bit for bit when every resampled width is even and the rotation is a
///   multiple of `2^(R + L - 1)` columns). The binarize and the vertical passes are unchanged.
#[derive(Clone, Debug, PartialEq)]
pub struct MaskPyramid {
    /// The solver geometry the pyramid was built for.
    geometry: FlowGeometry,
    /// `w_l * h_l` values of 0.0 or 1.0 per level, finest first.
    levels: Vec<Vec<f32>>,
    /// Whether the mask of `X` also gates the block matching sweeps ([`MaskPyramid::build`]) or not
    /// ([`MaskPyramid::build_ungated`]).
    gate: bool,
}

/// `v >= 0.95 ? 1 : 0` on every value, rows in parallel (a NaN value gives 0).
fn binarize(p: &mut Plane) {
    let w = p.width.max(1);
    p.data.par_chunks_mut(w).for_each(|row| {
        for v in row.iter_mut() {
            *v = if *v >= MASK_ON { 1.0 } else { 0.0 };
        }
    });
}

/// One step of the mask pyramid: copy, `blur5`, the corner aligned resample to `(nw, nh)`,
/// binarize; with `wrap` the circular blur and the period keeping resample (G6, U).
fn mask_step(p: &Plane, nw: usize, nh: usize, wrap: bool) -> Plane {
    let mut t = p.clone();
    prep::blur5_with(&mut t, wrap);
    let mut r = prep::resample_with(&t, nw, nh, wrap);
    binarize(&mut r);
    r
}

impl MaskPyramid {
    /// The mask pyramid of the matte `m` (at the output size) for the solver geometry of `s` at
    /// that size. A matte whose data does not hold `width * height` values is
    /// `FxError::Mismatch`; a size the solver cannot search fails as `FlowSettings::geometry`
    /// does.
    pub fn build(m: &Matte, s: &FlowSettings) -> Result<MaskPyramid> {
        if m.data.len() != m.width * m.height {
            return Err(FxError::Mismatch(format!(
                "mask pyramid: a {}x{} matte holds {} values",
                m.width,
                m.height,
                m.data.len()
            )));
        }
        let geometry = s.geometry(m.width, m.height)?;
        let mut cur = Plane { width: m.width, height: m.height, data: m.data.clone() };
        binarize(&mut cur);
        for _ in 0..geometry.reduction {
            let (nw, nh) = (cur.width / 2, cur.height / 2);
            cur = mask_step(&cur, nw, nh, s.wrap);
        }
        debug_assert_eq!((cur.width, cur.height), (geometry.width, geometry.height));
        let n = geometry.levels;
        let mut levels = Vec::with_capacity(n);
        for i in 0..n {
            let next = if i + 1 < n {
                let (nw, nh) = geometry.sizes[i + 1];
                Some(mask_step(&cur, nw, nh, s.wrap))
            } else {
                None
            };
            levels.push(std::mem::take(&mut cur.data));
            match next {
                Some(p) => cur = p,
                None => break,
            }
        }
        Ok(MaskPyramid { geometry, levels, gate: true })
    }

    /// [`MaskPyramid::build`] for a solve whose block matching is not gated: the gradient masking and
    /// the B vector step read the mask as before, every block matching sweep runs. READ in the
    /// original's solver (`CreateMotionVectors_64`, the same code in all four plugins):
    /// the CPU block matching routines get null masks, only the device path passes the mask
    /// pyramids. The layers of Create Vectors take their
    /// masks this way.
    pub fn build_ungated(m: &Matte, s: &FlowSettings) -> Result<MaskPyramid> {
        let mut p = MaskPyramid::build(m, s)?;
        p.gate = false;
        Ok(p)
    }

    /// Whether the mask of `X` gates the block matching.
    pub fn gates(&self) -> bool {
        self.gate
    }

    /// The binary mask of level `l`, 0 the finest.
    pub fn level(&self, l: usize) -> &[f32] {
        &self.levels[l]
    }

    /// The solver geometry the pyramid was built for (it must equal the luma pyramids').
    pub fn geometry(&self) -> &FlowGeometry {
        &self.geometry
    }
}

/// The user displacement of one pyramid level: `d` holds the accumulated displacement, two
/// values `(dx, dy)` per pixel interleaved like [`FlowField::data`], and `w` the weight, one
/// value per pixel. At the start of every level and again at its end, before the range check, the
/// solver sets `flow = d / w` at every pixel where `w != 0`. `d` is in the solver's internal sign
/// (`X(p) ~ Y(p - d)`: `d += (from - to) * k_l`), so the returned flow of a pixel set at
/// level 0 is `-(d / w)`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UserLevel {
    pub d: Vec<f32>,
    pub w: Vec<f32>,
}

/// The user displacement of one layer: one [`UserLevel`] per level of the luma pyramid, finest
/// first (`points::user_displacement` builds it). An empty `levels` is no user displacement.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UserDisplacement {
    pub levels: Vec<UserLevel>,
}

/// What a layered solve adds to the plain one. The default adds nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct SolveExtras<'a> {
    /// The layer's mask pyramids at `X` and at `Y`, in that order (gradient masking and the
    /// block matching sweep gate read `X`'s, the B vector step needs the warped mask of `Y`).
    pub masks: Option<(&'a MaskPyramid, &'a MaskPyramid)>,
    /// The user displacement of the layer's tracking points.
    pub user: Option<&'a UserDisplacement>,
}

/// The user displacement consumer (V): `flow = d / w` per channel at every pixel whose weight is not 0
/// (`w != 0.0`: a weight of -0.0 writes nothing, a NaN weight writes NaN), rows in parallel.
pub(crate) fn apply_user(flow: &mut [f32], u: &UserLevel, w: usize) {
    flow.par_chunks_mut(w * 2).zip(u.d.par_chunks(w * 2)).zip(u.w.par_chunks(w)).for_each(|((f, d), wt)| {
        for x in 0..wt.len() {
            let k = wt[x];
            if k != 0.0 {
                f[x * 2] = d[x * 2] / k;
                f[x * 2 + 1] = d[x * 2 + 1] / k;
            }
        }
    });
}

/// [`super::solve_pyramids`] with the extra inputs of a layer: the flow of `X(p) ~ Y(p + out(p))`
/// at the working resolution.
///
/// With `SolveExtras::default()` (or an empty user displacement) and `s.wrap` off this calls
/// [`super::solve_pyramids`], the plain solve, bit for bit. Otherwise the driver is the plain one
/// with these hooks, each level `l` from the start level down to 0:
/// 1. (V) after the zero start or the upsample, `flow = d / w` where `w != 0` (level `l` of
///    the user displacement);
/// 2. the gradient of `X`: with wrap the central difference across the seam (G6), then with
///    masks the gradient masking by `X`'s level mask;
/// 3. the B vector step: with masks the warped mask of `Y` must be at least 0.999;
///    with wrap the target column wraps and only the row is range checked (G6);
/// 4. the block matching: with masks the sweep step of a pixel where `X`'s mask is 0 is skipped
///    and the cost stays unmasked (U, on no effect's path); with wrap the cost wraps its column and window and
///    the step tests the target row only (G6, U);
///    on the finest level `flow = d / w` is set again right after the block matching step of the
///    second iteration, whether or not that step runs (READ in the solver); a pyramid made by
///    [`MaskPyramid::build_ungated`] does not gate the block matching;
/// 5. (V) `flow = d / w` again at the end of the level, before the range check; the start level's damping
///    follows the range check as in the plain driver, so it scales the set pixels too (at
///    Sensitivity 100 there is no damping);
/// 6. the range check: with wrap it tests y only (G6).
///
/// With wrap (rule G6, U: a stated rule of this project, not a measurement of the
/// original) every horizontal step is periodic: the gradient is [`calc_grad`]'s arithmetic with
/// the columns modulo the width, the Gaussian blurs of the tensor, the B vector and the flow are
/// the circular `conv::blur_gaussian_wrapx` (no renormalisation at the left and right borders),
/// the target columns of the B step and the block cost are split exactly into a whole column
/// modulo the width and a fraction (`flow::wrap_target`), the block matching sweeps every column
/// with the horizontal sweeps reading the neighbours as they stood before the sweep, and the flow
/// upsample keeps the period (`flow::upsample_flow_wrapx`). The y direction keeps the plain
/// rules. So rotating both pyramids, the mask pyramids and the user displacement by `k_l` columns
/// at every level `l` rotates the flow of every level by `k_l`, bit for bit, whenever the
/// rotations agree with the upsample (`k_l = 2 k_{l + 1}` for a level twice as wide as the next);
/// with the periodic pyramids of `flow::build_pyramid` a rotation of both frames by a multiple of
/// `2^(R + L - 1)` columns, every resampled width even, rotates the solved flow by that multiple
/// divided by `2^R`. Each level's step alone ([`level_step`]) commutes with every rotation.
///
/// The update of every level is the plain one (`new_disp`: a pixel moves only where `|t| > 40`,
/// `t` the determinant term of the blurred tensor). The tensor of a masked solve is built from the
/// masked gradient, so it is zero outside the layer's mask and diluted near it by the radius 7
/// blur, and it does not depend on the flow: a layer whose tensor passes the test at no pixel of
/// the levels the solve visits (a small matte, whose mask levels also erode at every step) keeps
/// the zero start and gets the exact zero flow, every value `-0.0` after the final negation. That
/// is the plain solver's rule on the masked gradient, not a reset by the range check (a zero
/// flow is in range everywhere).
///
/// Records gap G6 when the wrap solve runs. Errors: pyramids, mask
/// pyramids or a user displacement of another geometry are `FxError::Mismatch`.
pub fn solve_pyramids_with(x: &Pyramid, y: &Pyramid, s: &FlowSettings, e: SolveExtras<'_>) -> Result<FlowField> {
    if x.geometry != y.geometry {
        return Err(FxError::Mismatch("optical flow: pyramids of different geometry".into()));
    }
    let geo = &x.geometry;
    if let Some((mx, my)) = e.masks {
        if mx.geometry != *geo || my.geometry != *geo {
            return Err(FxError::Mismatch("optical flow: a mask pyramid of another geometry than the image pyramids".into()));
        }
    }
    let user = match e.user {
        Some(u) if !u.levels.is_empty() => {
            check_user(u, geo)?;
            Some(u)
        }
        _ => None,
    };
    if e.masks.is_none() && user.is_none() && !s.wrap {
        return super::solve_pyramids(x, y, s);
    }
    Ok(solve_extended(x, y, s, e.masks, user))
}

fn check_user(u: &UserDisplacement, geo: &FlowGeometry) -> Result<()> {
    if u.levels.len() != geo.levels {
        return Err(FxError::Mismatch(format!(
            "optical flow: a user displacement of {} levels for a pyramid of {}",
            u.levels.len(),
            geo.levels
        )));
    }
    for (l, (lv, &(w, h))) in u.levels.iter().zip(geo.sizes.iter()).enumerate() {
        if lv.d.len() != w * h * 2 || lv.w.len() != w * h {
            return Err(FxError::Mismatch(format!(
                "optical flow: user displacement level {l} holds {} and {} values for a {w}x{h} level",
                lv.d.len(),
                lv.w.len()
            )));
        }
    }
    Ok(())
}

/// The block matching call of the extended driver: [`block_match`] when there is neither a gate
/// nor wrap, else [`block_match_with`].
fn refine(a: &[f32], b: &[f32], flow: &mut [f32], w: usize, h: usize, gate: Option<&[f32]>, wrap: bool) {
    if gate.is_none() && !wrap {
        block_match(a, b, flow, w, h);
    } else {
        block_match_with(a, b, flow, w, h, gate, wrap);
    }
}

/// The steps 1 to 5 of [`solve_pyramids_with`] on one level of `w x h` (`finest` for level 0,
/// where the block matching runs inside the iterations): the user displacement, the gradient of
/// `img_a` (with wrap the periodic one) and its masking by `mask_x`, the tensor and its blur, the
/// three iterations of the B vector step (`mask_y`, wrap), its blur, the update and the blur of
/// the flow, the block matching, and the user displacement again. The Gaussians are the circular
/// ones with wrap. Without masks, user displacement and wrap it is the level of the plain driver
/// bit for bit. With wrap, rotating every input of the level by `k` columns rotates `flow` by `k`
/// columns, bit for bit, for any `k`.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn level_step(
    img_a: &[f32],
    img_b: &[f32],
    w: usize,
    h: usize,
    mask_x: Option<&[f32]>,
    mask_y: Option<&[f32]>,
    user: Option<&UserLevel>,
    s: &FlowSettings,
    finest: bool,
    flow: &mut [f32],
) {
    level_step_gated(img_a, img_b, w, h, mask_x, mask_y, mask_x, user, s, finest, flow);
}

/// [`level_step`] with the block matching gate given on its own (`gate`: the mask of `X` of a gating
/// [`MaskPyramid`], or `None`); the gradient masking and the B vector step read `mask_x` and `mask_y`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn level_step_gated(
    img_a: &[f32],
    img_b: &[f32],
    w: usize,
    h: usize,
    mask_x: Option<&[f32]>,
    mask_y: Option<&[f32]>,
    gate: Option<&[f32]>,
    user: Option<&UserLevel>,
    s: &FlowSettings,
    finest: bool,
    flow: &mut [f32],
) {
    let wrap = s.wrap;
    let blur = |data: &mut [f32], c: usize, r: f32| {
        if wrap {
            conv::blur_gaussian_wrapx_above_first(data, w, h, c, r);
        } else {
            conv::blur_gaussian_above_first(data, w, h, c, r);
        }
    };
    if let Some(u) = user {
        apply_user(flow, u, w);
    }
    let mut grad = if wrap { calc_grad_wrap(img_a, w, h) } else { calc_grad(img_a, w, h) };
    if let Some(m) = mask_x {
        mask_gradient(&mut grad, m, w, h, wrap);
    }
    let mut a = calc_a(&grad, w, h);
    blur(&mut a, 3, 7.0);
    for it in 0..3 {
        let mut b = if mask_y.is_none() && !wrap {
            calc_b(img_a, img_b, &grad, flow, w, h)
        } else {
            calc_b_with(img_a, img_b, &grad, flow, w, h, mask_y, wrap)
        };
        blur(&mut b, 2, 7.0);
        new_disp(flow, &a, &b);
        blur(&mut *flow, 2, 5.0);
        if it == 1 && finest {
            if !s.skip_refine {
                refine(img_a, img_b, flow, w, h, gate, wrap);
            }
            // READ (the solver): the user displacement is set again right after the block
            // matching step of the finest level, also when that step is skipped
            if let Some(u) = user {
                apply_user(flow, u, w);
            }
        }
    }
    if !finest && !s.skip_refine {
        refine(img_a, img_b, flow, w, h, gate, wrap);
    }
    if let Some(u) = user {
        apply_user(flow, u, w);
    }
}

/// The driver of [`super::solve_pyramids`] with the hooks listed on [`solve_pyramids_with`]: per
/// level [`level_step`], then the range check (y only with wrap), the start level's damping and
/// the upsample (the period keeping one with wrap).
fn solve_extended(
    x: &Pyramid,
    y: &Pyramid,
    s: &FlowSettings,
    masks: Option<(&MaskPyramid, &MaskPyramid)>,
    user: Option<&UserDisplacement>,
) -> FlowField {
    let geo = &x.geometry;
    let levels = geo.levels;
    let wrap = s.wrap;
    if wrap {
        fx_core::gaps::record(&crate::gaps::G6);
    }
    let (start, damp) = start_level(s.sensitivity, levels);
    let (sw0, sh0) = geo.sizes[start];
    let mut flow = vec![0.0f32; sw0 * sh0 * 2];
    let mut check = true;
    for l in (0..=start).rev() {
        let (w, h) = geo.sizes[l];
        let mask_x = masks.map(|(m, _)| m.level(l));
        let mask_y = masks.map(|(_, m)| m.level(l));
        let gate = masks.and_then(|(m, _)| m.gates().then(|| m.level(l)));
        let u = user.map(|u| &u.levels[l]);
        level_step_gated(&x.levels[l].data, &y.levels[l].data, w, h, mask_x, mask_y, gate, u, s, l == 0, &mut flow);
        if check {
            // READ: the range check of the solver (TwixtorPro_64) takes no mode
            // argument, the 360 mode checks both axes as the plain driver does
            let out = all_flow_out_of_range(&flow, w, h);
            if out {
                flow.iter_mut().for_each(|v| *v = 0.0);
            } else {
                check = false;
            }
        }
        if l == start && (damp as f64 - 1.0).abs() >= 1e-6 {
            scale_values(&mut flow, damp);
        }
        if l > 0 {
            let (nw, nh) = geo.sizes[l - 1];
            flow = if wrap { upsample_flow_wrapx(&flow, w, h, nw, nh) } else { upsample_flow(&flow, w, h, nw, nh) };
        }
    }
    scale_values(&mut flow, -1.0f32);
    FlowField { width: geo.width, height: geo.height, data: flow }
}
