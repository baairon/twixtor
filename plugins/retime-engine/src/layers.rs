//! The n-layer retime path of the FG capable effects: the per layer controls
//! ([`LayerParams`]), the layers of one render and the group a display or output shows
//! ([`LayerJob`]), the mattes a per layer flow source gets ([`LayerMattes`]), and the bodies
//! of `compose::retime_layers` (the solver path of retime-fgbg) and
//! `compose::retime_layers_with_flows` (the Vectors In path).
//!
//! Claim tags: V established, I inferred, U not established (a rule of this project, pinned in
//! the parity gaps register `crate::gaps` until a reference render grades it).
//!
//! A layered frame (`nfg >= 1`) on the vector path, in this order:
//! 1. the fetches: the tracking frames (when there is a tracking layer), the main frames `A` and
//!    `B`, then the FG mattes at `A` and at `B` (at the layer times of the main frames),
//!    then the error fill check;
//! 2. the raw mattes (`matte::extract`) and the tracking mattes of every layer
//!    (`matte::effective`: FG1 on top, Main_BG the product of the inverses, rule G1c; the
//!    inverse matte shrink, rule G1);
//! 3. the luma pyramids of the two tracking frames, built ONCE for all layers;
//! 4. per layer `k` of the group, FG1 first: the solve with that layer's sensitivity, the mask
//!    pyramids of its tracking matte at `A` and at `B` (swapped for the reverse flow) and its
//!    tracking points as a user displacement (rule G2); the solves a frame needs are those of
//!    the single layer path (Nearest one, Blend and Weighted Blend two);
//! 5. from layer `last` down to `first` (bottom up, so FG1 ends on top): `A` and `B` are
//!    premultiplied by the composition matte (the raw matte, Main_BG the constant 1),
//!    warped by the calls the single layer path makes for the warping code (the warp table of
//!    [`compose_layers`]) and composed OVER into the running results `RA` and `RB`;
//! 6. codes 1 and 2 fill the holes of each running result once, the holes being `white - alpha`
//!    then the final stage of the warping code; then ONE blur compensation tail with
//!    the flow of layer `last` (rule G1f, U).
//!
//! The fast path of a layered frame (an exact position, No Vectors, a segment row, a frame that is
//! not `prep::searchable`) composes the displayed group of the two unwarped frames the same way,
//! with no warp and no fill (rule G1h, U), then interpolates as the single layer fast path.
//!
//! `nfg = 0` never comes here unless a tracking point is in use: `compose::retime_layers` calls
//! the single layer functions of today for it, and with points only the solve changes
//! ([`single_layer_with_points`]: the user displacement, then today's composition).
//!
//! Precision contract: integer group arithmetic only, plus `frac = 1.0f32 - wA` as the single
//! layer path forms it; every pixel and flow value comes from the stages called (the solver,
//! `matte`, the warps, `holefill`, `smartblend`, `blend`, `filters`, the blur tail), each with its
//! own contract (f32 in the written order, no fused multiply add, rows in parallel through rayon
//! row chunks only). The layers run in a fixed order (solves FG1 first, composition bottom up),
//! the two solves of a layer run as `flow::solve_pair` runs them, and no float is summed across
//! threads, so the output is the same for any thread count.

use std::borrow::Cow;

use crate::blend::blend;
use crate::compose::{
    check_supported, compose_vectors, flow_settings, interpolate, invert_flow_on, issue_fill, nearest_uses_f_ab, Flows,
    Rendered,
};
use crate::filters::sharpen_step;
use crate::flow::extras::{MaskPyramid, SolveExtras};
use crate::flow::{build_pyramid, FlowField, FlowSettings, Pyramid};
use crate::gaps;
use crate::holefill::{fill_holes_masked, HoleMaskRef};
use crate::host::{EngineHost, EngineSettings};
use crate::image::Image;
use crate::matte::{self, Matte};
use crate::planner::MAX_FG_LAYERS;
use crate::points::{second_tracker, user_displacement, UserPoint};
use crate::smartblend::{smart_blend, smart_blend_binary};
use crate::timeline::Selection;
use crate::vectors::HoleMask;
use crate::warp::{blur_amount, blur_compensation_wrap, inv_warp, inv_warp_coverage, WarpOpts};
use fx_core::{FxError, Result};

#[cfg(test)]
mod tests;

/// The controls of one layer: FG k's `fgk_matte_chan`, `fgk_sensitivity / 100`,
/// `fgk_inv_matte_shrink` in pixels and `fgk_track`; Main_BG's sensitivity and track, with no
/// matte channel and no shrink.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerParams {
    pub matte_chan: i32,
    pub sensitivity: f32,
    pub shrink: f32,
    pub track: bool,
}

/// The layers of one render: `layers` has `nfg + 1` entries, `layers[nfg]` is Main_BG, and
/// `first..=last` is the group the output shows.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerJob {
    pub nfg: usize,
    pub layers: Vec<LayerParams>,
    pub first: usize,
    pub last: usize,
}

impl LayerJob {
    /// The group `(first, last)` of the `display` popup of retime-fgbg and vectors-in:
    /// 1 all layers `0..=nfg`, 3 to 5 the single group `min(display - 3, nfg)`, 6 the background
    /// group `nfg`. With `nfg = 0` every value is the one group `(0, 0)` (without mattes 3 to
    /// 6 render as 1). Display 2 (Source) composes no layer; it and any value outside the popup
    /// give the full group, as 1 does (U, ours).
    pub fn display_group(display: i32, nfg: usize) -> (usize, usize) {
        match display {
            3..=5 => {
                let g = ((display - 3) as usize).min(nfg);
                (g, g)
            }
            6 => (nfg, nfg),
            _ => (0, nfg),
        }
    }

    /// The group `(first, last)` of create-vectors' `layers_output` popup `lo`: 1 All gives
    /// `(0, nfg)`, 2 to 5 the single layer `min(lo - 2, nfg)` (5 MAIN_BG is `nfg`). A value
    /// outside the popup is All (U, ours).
    pub fn output_group(lo: i32, nfg: usize) -> (usize, usize) {
        match lo {
            2..=5 => {
                let g = ((lo - 2) as usize).min(nfg);
                (g, g)
            }
            _ => (0, nfg),
        }
    }

    /// The job of a front-end: `fg` the FG layers FG1 first (`nfg = fg.len()`, at most three),
    /// `main_bg` Main_BG's controls and the group of the `display` popup ([`Self::display_group`]).
    pub fn for_display(fg: &[LayerParams], main_bg: LayerParams, display: i32) -> LayerJob {
        let nfg = fg.len();
        let mut layers = fg.to_vec();
        layers.push(main_bg);
        let (first, last) = Self::display_group(display, nfg);
        LayerJob { nfg, layers, first, last }
    }

    /// The structural checks of a job: at most three FG layers, `nfg + 1` layers and the group
    /// `first <= last <= nfg`; anything else is `FxError::Mismatch`.
    pub fn check(&self) -> Result<()> {
        if self.nfg > MAX_FG_LAYERS || self.layers.len() != self.nfg + 1 || self.first > self.last || self.last > self.nfg {
            return Err(FxError::Mismatch(format!(
                "layer job: {} FG layers, {} layer entries and the group {}..={} (at most {MAX_FG_LAYERS} FG layers, nfg + 1 entries, first <= last <= nfg)",
                self.nfg,
                self.layers.len(),
                self.first,
                self.last
            )));
        }
        Ok(())
    }

    /// The inverse matte shrink of every FG layer, FG1 first (`matte::effective`'s `shrinks`).
    fn shrinks(&self) -> Vec<f32> {
        self.layers[..self.nfg].iter().map(|l| l.shrink).collect()
    }
}

/// The two mattes of one layer for a flow source: at the layer times of frame `A` and frame `B`.
/// `compose::retime_layers_with_flows` passes the layer's tracking matte (`matte::effective`,
/// Main_BG the product of the inverses; the constant 1 when `nfg = 0`).
#[derive(Clone, Copy, Debug)]
pub struct LayerMattes<'a> {
    pub at_a: &'a Matte,
    pub at_b: &'a Matte,
}

fn bad_interp(other: i32) -> FxError {
    FxError::BadParam {
        id: "frame_interp".into(),
        reason: format!("interpolation mode {other} is not 0 (Nearest), 1 (Blend) or 2 (Weighted Blend)"),
    }
}

fn missing_flow(which: &str, k: usize) -> FxError {
    FxError::Mismatch(format!("the composition of layer {k} needs the flow {which} but it was not computed"))
}

/// The point lists of a render: one per layer (`nfg + 1`), else `FxError::Mismatch`.
pub(crate) fn check_point_lists(points: &[Vec<UserPoint>], job: &LayerJob) -> Result<()> {
    if points.len() != job.nfg + 1 {
        return Err(FxError::Mismatch(format!(
            "tracking points: {} lists for {} layers (one per layer, Main_BG last)",
            points.len(),
            job.nfg + 1
        )));
    }
    Ok(())
}

/// The luma pyramids of the two tracking frames, built once and in parallel as
/// `flow::solve_pair` builds them (a pyramid does not depend on the sensitivity).
fn pyramids(xa: &Image, xb: &Image, fs: &FlowSettings) -> Result<(Pyramid, Pyramid)> {
    if xa.width != xb.width || xa.height != xb.height || xa.fmt() != xb.fmt() {
        return Err(FxError::Mismatch(format!(
            "optical flow: incompatible inputs ({}x{} {:?} against {}x{} {:?})",
            xa.width,
            xa.height,
            xa.fmt(),
            xb.width,
            xb.height,
            xb.fmt()
        )));
    }
    let (pa, pb) = rayon::join(|| build_pyramid(xa, fs), || build_pyramid(xb, fs));
    Ok((pa?, pb?))
}

/// The flows of one layer for the frame pair (the solves of `compose::solve_flows` for the
/// interpolation mode: Nearest the one [`nearest_uses_f_ab`] names, Blend and Weighted Blend the
/// two, run as `flow::solve_pair` runs them). `pa` and `pb` are the luma pyramids of the tracking
/// frames `A` and `B`, `masks` the layer's mask pyramids at `A` and at `B` (the solve with
/// `X = A` takes them in that order, the one with `X = B` swapped), `points` the layer's
/// tracking points: the solve with `X = A` gets the pairs `UserPoint::pair(true)`, the other
/// `pair(false)` (the user displacement is in the solver's internal sign, its output
/// convention is `X(p) ~ Y(p + flow(p))`, so a point that moves from `p` in `A` to `q` in `B` sets
/// `F_AB` at its pixel to `(q - p) * k_0`). A layer whose Track is off (`track == false`, only
/// create-vectors has the control) takes `points::second_tracker` instead of a solve (rule
/// G5). An empty point list and no masks is the plain solve, bit for bit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_layer(
    settings: &EngineSettings,
    sel: &Selection,
    fs: &FlowSettings,
    track: bool,
    pa: &Pyramid,
    pb: &Pyramid,
    masks: Option<(&MaskPyramid, &MaskPyramid)>,
    points: &[UserPoint],
) -> Result<Flows> {
    let out = (settings.width, settings.height);
    let geo = pa.geometry();
    let pairs = |x_is_a: bool| -> Vec<([f32; 2], [f32; 2])> { points.iter().map(|p| p.pair(x_is_a)).collect() };
    let stages = settings.stages.get();
    let solve_ab = || -> Result<FlowField> {
        if !track {
            return second_tracker(&pairs(true), geo, out);
        }
        let user = (!points.is_empty()).then(|| user_displacement(&pairs(true), geo, out));
        stages.solve_pyramids_with(pa, pb, fs, SolveExtras { masks, user: user.as_ref() })
    };
    let solve_ba = || -> Result<FlowField> {
        if !track {
            return second_tracker(&pairs(false), geo, out);
        }
        let user = (!points.is_empty()).then(|| user_displacement(&pairs(false), geo, out));
        stages.solve_pyramids_with(pb, pa, fs, SolveExtras { masks: masks.map(|(ma, mb)| (mb, ma)), user: user.as_ref() })
    };
    match sel.interp {
        0 => {
            if nearest_uses_f_ab(sel.wa, settings.warp_mode) {
                Ok(Flows { f_ab: Some(solve_ab()?), f_ba: None })
            } else {
                Ok(Flows { f_ab: None, f_ba: Some(solve_ba()?) })
            }
        }
        1 | 2 => {
            // a device serves one solve at a time on its one queue: the two run in this order
            // there (the results are the same either way)
            let (f_ba, f_ab) = if settings.stages.is_cpu() { rayon::join(solve_ba, solve_ab) } else { (solve_ba(), solve_ab()) };
            Ok(Flows { f_ab: Some(f_ab?), f_ba: Some(f_ba?) })
        }
        other => Err(bad_interp(other)),
    }
}

/// The tracking frames of the pair: the tracking layer's when there is one, else `None` (the main
/// frames are tracked).
fn tracking_frames(settings: &EngineSettings, host: &dyn EngineHost, sel: &Selection) -> Result<Option<(Image, Image)>> {
    if settings.no_tracking_layer {
        return Ok(None);
    }
    let ta = host.tracking_frame(sel.a)?;
    let tb = host.tracking_frame(sel.b)?;
    Ok(Some((ta, tb)))
}

/// The single layer vector path (`nfg = 0`) with tracking points in use: the fetches and the
/// error fill check of today's vector path, the solves with the user displacement of the points
/// ([`solve_layer`] with `layer`'s sensitivity, no masks: the only layer's tracking matte is the
/// constant 1) and today's composition (`compose::compose_vectors`).
pub(crate) fn single_layer_with_points(
    settings: &EngineSettings,
    host: &dyn EngineHost,
    sel: &Selection,
    layer: &LayerParams,
    points: &[UserPoint],
) -> Result<Rendered> {
    check_supported(settings, sel)?;
    let tracking = tracking_frames(settings, host, sel)?;
    let a = host.main_frame(sel.a)?;
    let b = host.main_frame(sel.b)?;
    if let Some(fill) = issue_fill(settings, host) {
        return Ok(fill);
    }
    let (xa, xb) = match &tracking {
        Some((ta, tb)) => (ta, tb),
        None => (&a, &b),
    };
    let fs = FlowSettings { sensitivity: layer.sensitivity, ..flow_settings(settings, sel) };
    let (pa, pb) = pyramids(xa, xb, &fs)?;
    let flows = solve_layer(settings, sel, &fs, layer.track, &pa, &pb, None, points)?;
    Ok(Rendered::Image(compose_vectors(settings, sel, &a, &b, &flows)?))
}

/// The raw mattes of the FG layers at frame `A` and at frame `B` (`matte::extract` of
/// `EngineHost::matte_frame` with each layer's channel), FG1 first, A before B per layer. A matte
/// frame of another size than `like` is `FxError::Mismatch` (the layer primitives need one size).
fn fetch_mattes(host: &dyn EngineHost, job: &LayerJob, sel: &Selection, like: &Image) -> Result<(Vec<Matte>, Vec<Matte>)> {
    let mut at_a = Vec::with_capacity(job.nfg);
    let mut at_b = Vec::with_capacity(job.nfg);
    for k in 0..job.nfg {
        let chan = job.layers[k].matte_chan;
        for (pos, list) in [(sel.a, &mut at_a), (sel.b, &mut at_b)] {
            let img = host.matte_frame(k, pos)?;
            if (img.width, img.height) != (like.width, like.height) {
                return Err(FxError::Mismatch(format!(
                    "FG matte {k}: a {}x{} matte frame for {}x{} frames",
                    img.width, img.height, like.width, like.height
                )));
            }
            list.push(matte::extract(&img, chan));
        }
    }
    Ok((at_a, at_b))
}

/// The layer image of layer `k`: the frame premultiplied by the raw matte of FG `k`; for
/// Main_BG (`k == nfg`, matte 1) the frame itself (`q(c * 1.0)` is `c` at 8 and 16 bpc and
/// `c * 1.0` is `c` in float, so the premultiply is skipped). A matte of another size than the
/// frame is `FxError::Mismatch`.
fn layer_image<'f>(frame: &'f Image, mattes: &[Matte], k: usize, nfg: usize) -> Result<Cow<'f, Image>> {
    if k == nfg {
        Ok(Cow::Borrowed(frame))
    } else {
        Ok(Cow::Owned(matte::premultiply(frame, &mattes[k])?))
    }
}

/// The composite of the group `first..=last` of one unwarped frame (the fast path): from
/// layer `last` down to `first`, the layer image OVER the running result.
fn compose_group(frame: &Image, mattes: &[Matte], job: &LayerJob) -> Result<Image> {
    let mut r = Image::new(frame.width, frame.height, frame.fmt());
    for k in (job.first..=job.last).rev() {
        matte::over(&mut r, &*layer_image(frame, mattes, k, job.nfg)?)?;
    }
    Ok(r)
}

/// The fast path of a layered frame (rule G1h, U): the main frames `A` and `B`, the FG mattes
/// at both, the error fill check, the composite of the displayed group of each frame
/// ([`compose_group`], no warp, no fill) and `compose::interpolate` of the two composites (the
/// single layer fast path on them). Records G1 (a layered frame), G1h for a single group display
/// (`first == last`) and G1c with two or more FG layers (the composition order).
pub(crate) fn fast_path(settings: &EngineSettings, host: &dyn EngineHost, job: &LayerJob, sel: &Selection) -> Result<Rendered> {
    let a = host.main_frame(sel.a)?;
    let b = host.main_frame(sel.b)?;
    let (ma, mb) = fetch_mattes(host, job, sel, &a)?;
    if let Some(fill) = issue_fill(settings, host) {
        return Ok(fill);
    }
    fx_core::gaps::record(&gaps::G1);
    if job.first == job.last {
        fx_core::gaps::record(&gaps::G1H);
    }
    if job.nfg >= 2 {
        fx_core::gaps::record(&gaps::G1C);
    }
    let ra = compose_group(&a, &ma, job)?;
    let rb = compose_group(&b, &mb, job)?;
    let mut out = Image::new(a.width, a.height, a.fmt());
    interpolate(&mut out, sel, &ra, &rb)?;
    Ok(Rendered::Image(out))
}

/// The layered vector path of `compose::retime_layers` (`nfg >= 1`; module documentation, steps 1
/// to 6). `points` gets the selection and returns the tracking points of every layer (`nfg + 1`
/// lists, Main_BG last).
pub(crate) fn vector_path<G>(settings: &EngineSettings, host: &dyn EngineHost, job: &LayerJob, sel: &Selection, points: G) -> Result<Rendered>
where
    G: FnOnce(&Selection) -> Result<Vec<Vec<UserPoint>>>,
{
    check_supported(settings, sel)?;
    let pts = points(sel)?;
    check_point_lists(&pts, job)?;
    let tracking = tracking_frames(settings, host, sel)?;
    let a = host.main_frame(sel.a)?;
    let b = host.main_frame(sel.b)?;
    let mattes = fetch_mattes(host, job, sel, &a)?;
    if let Some(fill) = issue_fill(settings, host) {
        return Ok(fill);
    }
    fx_core::gaps::record(&gaps::G1);
    let (xa, xb) = match &tracking {
        Some((ta, tb)) => (ta, tb),
        None => (&a, &b),
    };
    let shrinks = job.shrinks();
    let ea = matte::effective(&mattes.0, &shrinks, a.width, a.height)?;
    let eb = matte::effective(&mattes.1, &shrinks, a.width, a.height)?;
    let base = flow_settings(settings, sel);
    let (pa, pb) = pyramids(xa, xb, &base)?;
    let mut flows = Vec::with_capacity(job.last - job.first + 1);
    for k in job.first..=job.last {
        let layer = &job.layers[k];
        let fs = FlowSettings { sensitivity: layer.sensitivity, ..base };
        let (ma, mb) = rayon::join(|| MaskPyramid::build(&ea[k], &fs), || MaskPyramid::build(&eb[k], &fs));
        let (ma, mb) = (ma?, mb?);
        flows.push(solve_layer(settings, sel, &fs, layer.track, &pa, &pb, Some((&ma, &mb)), &pts[k])?);
    }
    Ok(Rendered::Image(compose_layers(settings, sel, job, &a, &b, &mattes, &flows)?))
}

/// With exactly one flow of a layer loaded the other one is its inversion
/// (`compose::invert_flow`), derived only when this render reads it: the interpolation mode
/// needs it, or `tail` (the blur tail runs and this is layer `last`) and it is `F_AB`. The rule
/// of the single layer Vectors In path, per layer.
fn complete_layer_flows(settings: &EngineSettings, sel: &Selection, flows: &mut Flows, tail: bool, k: usize) -> Result<()> {
    let (need_ab, need_ba) = match sel.interp {
        0 => {
            let ab = nearest_uses_f_ab(sel.wa, settings.warp_mode);
            (ab || tail, !ab)
        }
        _ => (true, true),
    };
    match (&flows.f_ab, &flows.f_ba) {
        (None, None) => Err(FxError::Mismatch(format!("Vectors In: neither vector layer supplied a flow for layer {k}"))),
        (None, Some(ba)) => {
            if need_ab {
                flows.f_ab = Some(invert_flow_on(settings.stages.get(), ba)?);
            }
            Ok(())
        }
        (Some(ab), None) => {
            if need_ba {
                flows.f_ba = Some(invert_flow_on(settings.stages.get(), ab)?);
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// The layered Vectors In path of `compose::retime_layers_with_flows` (`nfg >= 1`): the main
/// frames, the FG mattes at `A` and `B`, the error fill check, the tracking mattes of every layer
/// (`matte::effective`), then per layer `k` of the group, FG1 first, `flows(sel, k, mattes)` with
/// the layer's tracking mattes at `A` and `B` (the front-end's per layer load, rule G1e) and the
/// inversion of a missing flow of that layer ([`complete_layer_flows`]), then the composition of
/// [`compose_layers`] with Vectors In's settings (no wrap).
pub(crate) fn vector_path_with_flows<F>(settings: &EngineSettings, host: &dyn EngineHost, job: &LayerJob, sel: &Selection, mut flows: F) -> Result<Rendered>
where
    F: FnMut(&Selection, usize, &LayerMattes<'_>) -> Result<Flows>,
{
    check_supported(settings, sel)?;
    let a = host.main_frame(sel.a)?;
    let b = host.main_frame(sel.b)?;
    let mattes = fetch_mattes(host, job, sel, &a)?;
    if let Some(fill) = issue_fill(settings, host) {
        return Ok(fill);
    }
    fx_core::gaps::record(&gaps::G1);
    let shrinks = job.shrinks();
    let ea = matte::effective(&mattes.0, &shrinks, a.width, a.height)?;
    let eb = matte::effective(&mattes.1, &shrinks, a.width, a.height)?;
    let tail = blur_amount(sel.s0, sel.s1, settings.blur).is_some();
    let mut per_layer = Vec::with_capacity(job.last - job.first + 1);
    for k in job.first..=job.last {
        let mut f = flows(sel, k, &LayerMattes { at_a: &ea[k], at_b: &eb[k] })?;
        complete_layer_flows(settings, sel, &mut f, tail && k == job.last, k)?;
        per_layer.push(f);
    }
    Ok(Rendered::Image(compose_layers(settings, sel, job, &a, &b, &mattes, &per_layer)?))
}

/// Whether the Nearest composition of warping code `code` takes frame `A` (else `B`), with the
/// compare each single layer composition makes: code 0 `0.5 <= wA`, codes 1 to 3 `!(0.5 > wA)`
/// (they differ only for a NaN weight).
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn nearest_takes_a(code: i32, wa: f32) -> bool {
    if code == 0 {
        0.5 <= wa
    } else {
        !(0.5f32 > wa)
    }
}

/// The flow that warps side `A` (`side_a`) or `B` of one layer: Nearest the one flow
/// [`nearest_uses_f_ab`] names; Blend and Weighted Blend `F_BA` for `A` and `F_AB` for `B` with the
/// inverse warps (codes 0 and 1), the frame's OWN flow with the forward warps (codes 2 and 3).
fn side_flow<'f>(flows: &'f Flows, code: i32, sel: &Selection, side_a: bool, k: usize) -> Result<&'f FlowField> {
    let ab = if sel.interp == 0 { nearest_uses_f_ab(sel.wa, code) } else { (code >= 2) == side_a };
    if ab {
        flows.f_ab.as_ref().ok_or_else(|| missing_flow("F_AB", k))
    } else {
        flows.f_ba.as_ref().ok_or_else(|| missing_flow("F_BA", k))
    }
}

/// One warp of a layer image `src` by `flow` and `amount`, with the call the single layer path
/// makes for the warping code (see [`compose_layers`]).
fn warp_layer(settings: &EngineSettings, weighted: bool, src: &Image, flow: &FlowField, amount: f32) -> Result<Image> {
    let mut t = Image::new(src.width, src.height, src.fmt());
    match settings.warp_mode {
        0 => inv_warp(&mut t, src, flow, amount, WarpOpts { is360: settings.wrap_360, ..WarpOpts::default() })?,
        1 => {
            let opts = WarpOpts { hold: false, is360: settings.wrap_360 && !weighted, mode: 1 };
            // the coverage is dropped: the holes come from the alpha of the composite
            let _ = inv_warp_coverage(&mut t, src, flow, amount, opts)?;
        }
        2 => {
            let _ = settings.stages.get().forward_warp_coverage(&mut t, src, flow, amount)?;
        }
        _ => settings.stages.get().forward_warp(&mut t, src, flow, amount)?,
    }
    Ok(t)
}

/// The hole fill of a running result (V): the holes are `white - alpha`
/// (`matte::alpha_holes`, any partial alpha counts) and `holefill::fill_holes_masked` fills them
/// with the flag `is360`.
fn fill_alpha_holes(r: &mut Image, is360: bool) -> Result<()> {
    let holes = matte::alpha_holes(r);
    let mask = match &holes {
        HoleMask::U8(m) => HoleMaskRef::U8(m),
        HoleMask::U16(m) => HoleMaskRef::U16(m),
        HoleMask::F32(m) => HoleMaskRef::F32(m),
    };
    fill_holes_masked(r, mask, is360)
}

/// The composition of a layered frame (module documentation, steps 5 and 6). `a` and
/// `b` are the main frames, `mattes` the raw FG mattes at `A` and at `B`, `flows[i]` the flows of
/// layer `job.first + i`. With `frac = 1.0f32 - wA`:
///
/// | code | warp of a layer image (side `A` by `frac`, side `B` by `wA`) | holes | final stage |
/// |---|---|---|---|
/// | 0 | inverse warp, hold edges, `is360 = wrap_360` | none | Nearest `R`; Blend `blend(RA, RB, frac)`; Weighted `sharpen_step` of `RA` against `A` at `frac` and of `RB` against `B` at `wA`, then `blend` |
/// | 1 | inverse warp without hold, coverage dropped, `is360 = wrap_360` (Weighted: off) | each running result, `white - alpha` | Nearest `R`; Blend `smart_blend` (the binary alpha variant with the "360 Video" checkbox); Weighted the sharpening, then `smart_blend` |
/// | 2 | forward warp, coverage dropped | as code 1 | as code 1 |
/// | 3 | forward warp | none | as code 0 |
///
/// Nearest has one running result, the side the single layer composition takes
/// ([`nearest_takes_a`]). The fill runs with `is360 = wrap_360` except under Weighted Blend (the
/// weighted compositions of the single layer path keep the wrap off). Then ONE blur compensation
/// tail with `wrap_360` and the blur flow of layer `last` (`F_AB`, or the one flow of Nearest; rule
/// G1f, U). Records G1g when a fill runs on a single group (`first == last`, the literal reading
/// of the composition: the fill floods an FG only view) and G1f when the tail changes pixels.
pub(crate) fn compose_layers(
    settings: &EngineSettings,
    sel: &Selection,
    job: &LayerJob,
    a: &Image,
    b: &Image,
    mattes: &(Vec<Matte>, Vec<Matte>),
    flows: &[Flows],
) -> Result<Image> {
    check_supported(settings, sel)?;
    job.check()?;
    if flows.len() != job.last - job.first + 1 || mattes.0.len() != job.nfg || mattes.1.len() != job.nfg || !a.same_shape(b) {
        return Err(FxError::Mismatch(format!(
            "layered composition: {} flows for the group {}..={}, {} and {} mattes for {} FG layers",
            flows.len(),
            job.first,
            job.last,
            mattes.0.len(),
            mattes.1.len(),
            job.nfg
        )));
    }
    for m in mattes.0.iter().chain(&mattes.1) {
        if (m.width, m.height) != (a.width, a.height) || m.data.len() != a.width * a.height {
            return Err(FxError::Mismatch(format!("layered composition: a {}x{} matte for {}x{} frames", m.width, m.height, a.width, a.height)));
        }
    }
    let wa = sel.wa;
    // frac = 1.0f - wA, the position after frame A; the amount of A and the weight of B
    let frac = 1.0f32 - wa;
    let code = settings.warp_mode;
    let weighted = sel.interp == 2;
    let fills = code == 1 || code == 2;
    let (side_a, side_b) = if sel.interp == 0 {
        let takes_a = nearest_takes_a(code, wa);
        (takes_a, !takes_a)
    } else {
        (true, true)
    };
    let mut ra = side_a.then(|| Image::new(a.width, a.height, a.fmt()));
    let mut rb = side_b.then(|| Image::new(b.width, b.height, b.fmt()));
    for k in (job.first..=job.last).rev() {
        let f = &flows[k - job.first];
        if let Some(r) = ra.as_mut() {
            let t = warp_layer(settings, weighted, &*layer_image(a, &mattes.0, k, job.nfg)?, side_flow(f, code, sel, true, k)?, frac)?;
            matte::over(r, &t)?;
        }
        if let Some(r) = rb.as_mut() {
            let t = warp_layer(settings, weighted, &*layer_image(b, &mattes.1, k, job.nfg)?, side_flow(f, code, sel, false, k)?, wa)?;
            matte::over(r, &t)?;
        }
    }
    if fills {
        if job.first == job.last {
            fx_core::gaps::record(&gaps::G1G);
        }
        let is360 = settings.wrap_360 && !weighted;
        for r in [ra.as_mut(), rb.as_mut()].into_iter().flatten() {
            fill_alpha_holes(r, is360)?;
        }
    }
    let mut out = match (ra, rb) {
        (Some(r), None) | (None, Some(r)) => r,
        (Some(mut ra), Some(mut rb)) => {
            if weighted {
                sharpen_step(&mut ra, a, frac);
                sharpen_step(&mut rb, b, wa);
            }
            let mut out = Image::new(a.width, a.height, a.fmt());
            if !fills {
                blend(&mut out, &ra, &rb, frac);
            } else if settings.video_360 && !weighted {
                smart_blend_binary(&mut out, &ra, &rb, frac)?;
            } else {
                smart_blend(&mut out, &ra, &rb, frac)?;
            }
            out
        }
        (None, None) => unreachable!("Nearest takes one side, the blends both"),
    };
    if let Some(flow) = flows[job.last - job.first].blur_flow() {
        if let Some(amount) = blur_amount(sel.s0, sel.s1, settings.blur) {
            if !((amount.abs() as f64) < 1e-6) {
                fx_core::gaps::record(&gaps::G1F);
            }
        }
        blur_compensation_wrap(&mut out, flow, sel.s0, sel.s1, settings.blur, settings.quality_high, settings.wrap_360)?;
    }
    Ok(out)
}
