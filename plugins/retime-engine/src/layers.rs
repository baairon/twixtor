//! The n-layer retime path of the FG capable effects: the per layer controls
//! ([`LayerParams`]), the layers of one render and the group a display or output shows
//! ([`LayerJob`]), the mattes a per layer flow source gets ([`LayerMattes`]), and the bodies
//! of `compose::retime_layers` (the solver path of retime-fgbg) and
//! `compose::retime_layers_with_flows` (the Vectors In path).
//!
//! Claim tags: V established, I inferred, U not established (a rule of this project). The effects
//! set `LayerJob::plates`, whose paths follow the original; the rules of this project below run
//! only on paths without it, which no effect takes.
//!
//! A layered frame (`nfg >= 1`) on the vector path, in this order:
//! 1. the fetches: the tracking frames (when there is a tracking layer), the main frames `A` and
//!    `B`, then the FG mattes at `A` and at `B` (at the layer times of the main frames),
//!    then the error fill check;
//! 2. the raw mattes (`matte::extract`) and the tracking mattes of every layer
//!    (`matte::effective`: FG1 on top, Main_BG the product of the inverses; the
//!    inverse matte shrink);
//! 3. the luma pyramids of the two tracking frames, built ONCE for all layers;
//! 4. per layer `k` of the group, FG1 first: the solve with that layer's sensitivity, the mask
//!    pyramids of its tracking matte at `A` and at `B` (swapped for the reverse flow) and its
//!    tracking points as a user displacement; the solves a frame needs are those of
//!    the single layer path (Nearest one, Blend and Weighted Blend two);
//! 5. from layer `last` down to `first` (bottom up, so FG1 ends on top): `A` and `B` are
//!    premultiplied by the composition matte (the raw matte, Main_BG the constant 1),
//!    warped by the calls the single layer path makes for the warping code (the warp table of
//!    [`compose_layers`]) and composed OVER into the running results `RA` and `RB`;
//! 6. codes 1 and 2 fill the holes of each running result once, the holes being `white - alpha`
//!    then the final stage of the warping code; then ONE blur compensation tail with
//!    the flow of layer `last` (U).
//!
//! The fast path of a layered frame (an exact position, No Vectors, a segment row, a frame that is
//! not `prep::searchable`) composes the displayed group of the two unwarped frames the same way,
//! with no warp and no fill (U), then interpolates as the single layer fast path.
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
use crate::holefill::{fill_holes_masked, fill_holes_masked_with, HoleMaskRef, VerticalTaps};
use crate::host::{EngineHost, EngineSettings};
use crate::image::{Image, Samples, A, CHANNELS};
use crate::matte::{self, Matte};
use crate::planner::MAX_FG_LAYERS;
use crate::points::{second_tracker, user_displacement_with, UserPoint};
use crate::smartblend::{smart_blend, smart_blend_binary};
use crate::timeline::Selection;
use crate::vectors::HoleMask;
use crate::warp::{blur_amount, blur_compensation_wrap, inv_warp, inv_warp_coverage, WarpOpts};
use fx_core::{FxError, Result};

mod plates;
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
    /// A single group display of the fast path rebuilds the part of the layer that the layers
    /// above cover ([`reconstructed_layer`]). Set by retime-fgbg (plugin 03), where it was
    /// measured; `for_display` leaves it off.
    pub rebuild_group: bool,
    /// The fast path of the full group (Display 1, `first == 0 && last == nfg`) with no inverse matte shrink on
    /// any FG layer reads no FG matte and is the single layer fast path ([`crate::compose::fast_path`]). Set by
    /// retime-fgbg (plugin 03), where it was measured on the call records (with a shrink the original reads the
    /// mattes); `for_display` leaves it off.
    pub full_group_reads_no_matte: bool,
    /// The layer images and the composition use the plugin's own pixel kernels ([`matte::premultiply_pl`],
    /// [`matte::over_pl`]: 8 and 16 bpc multiply by the f32 constants 1/255 and 1/65535 instead of dividing) in place
    /// of the Smart Blend kernels' arithmetic. Set by retime-fgbg (plugin 03), where it was read from the code;
    /// `for_display` leaves it off.
    pub pl_kernels: bool,
    /// The vector path composes the group as the original's per layer loop does ([`plates`]): every layer is
    /// interpolated on its own from its two plates (its layer image OVER the other frame's warped to its time,
    /// unpremultiplied, colour filled where it is not opaque, its alpha replaced by its own matte, premultiplied), and
    /// the interpolated layers are composed OVER, Main_BG first. Set by retime-fgbg (plugin 03), where it was read from
    /// the code; `for_display` leaves it off.
    pub plates: bool,
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
        LayerJob { nfg, layers, first, last, rebuild_group: false, full_group_reads_no_matte: false, pl_kernels: false, plates: false }
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

    /// The fast path of this job is the single layer one ([`crate::compose::fast_path`]): the full group of a job
    /// whose full group reads no matte there (`full_group_reads_no_matte`), with no inverse matte shrink on any FG
    /// layer.
    pub(crate) fn full_group_fast_path(&self) -> bool {
        self.full_group_reads_no_matte && self.first == 0 && self.last == self.nfg && self.layers[..self.nfg].iter().all(|l| l.shrink == 0.0)
    }
}

/// The two mattes of one layer for a flow source: at the layer times of frame `A` and frame `B`.
/// `compose::retime_layers_with_flows` passes the layer's tracking matte (`matte::effective`,
/// Main_BG the product of the inverses; the constant 1 when `nfg = 0`), and for a job with
/// `plates` the layer's composition matte (`plates::comp_mattes`: FG k's raw matte cut by every
/// FG layer above it, Main_BG white cut, divided by the white of the depth, so that
/// `trunc(m * white + 0.5)` gives the integer plane back at 8 and 16 bpc and float is the value).
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
        let user = (!points.is_empty()).then(|| user_displacement_with(&pairs(true), geo, out, fs.wrap));
        stages.solve_pyramids_with(pa, pb, fs, SolveExtras { masks, user: user.as_ref() })
    };
    let solve_ba = || -> Result<FlowField> {
        if !track {
            return second_tracker(&pairs(false), geo, out);
        }
        let user = (!points.is_empty()).then(|| user_displacement_with(&pairs(false), geo, out, fs.wrap));
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
/// constant 1) and today's composition (`compose::compose_vectors`). With
/// `EngineSettings::motion_rows_reversed` the whole path runs on bottom-up rows, as the vector path without points
/// does: the frames reversed, the points at `H - 1 - y` ([`points_rows_reversed`]), the output turned back.
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
    let fs = FlowSettings { sensitivity: layer.sensitivity, ..flow_settings(settings, sel) };
    if settings.motion_rows_reversed {
        let (ra, rb) = (a.rows_reversed(), b.rows_reversed());
        let rtracking = tracking.as_ref().map(|(ta, tb)| (ta.rows_reversed(), tb.rows_reversed()));
        let (xa, xb) = match &rtracking {
            Some((ta, tb)) => (ta, tb),
            None => (&ra, &rb),
        };
        let rpts = points_rows_reversed(points, a.height);
        let (pa, pb) = pyramids(xa, xb, &fs)?;
        let flows = solve_layer(settings, sel, &fs, layer.track, &pa, &pb, None, &rpts)?;
        return Ok(Rendered::Image(compose_vectors(settings, sel, &ra, &rb, &flows)?.rows_reversed()));
    }
    let (xa, xb) = match &tracking {
        Some((ta, tb)) => (ta, tb),
        None => (&a, &b),
    };
    let (pa, pb) = pyramids(xa, xb, &fs)?;
    let flows = solve_layer(settings, sel, &fs, layer.track, &pa, &pb, None, points)?;
    Ok(Rendered::Image(compose_vectors(settings, sel, &a, &b, &flows)?))
}

/// The points of a layer on the bottom-up rows of the motion path: `y` becomes `(H - 1) - y` (f32) at both frames, as
/// the original's point source flips them (READ in TwixtorPro_64: `(float)(H - 1) - y`).
fn points_rows_reversed(points: &[UserPoint], height: usize) -> Vec<UserPoint> {
    let last_row = height as f32 - 1.0;
    points.iter().map(|p| UserPoint { at_a: [p.at_a[0], last_row - p.at_a[1]], at_b: [p.at_b[0], last_row - p.at_b[1]] }).collect()
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
            // the per layer plates read the raw plane (a float matte keeps NaN and values outside 0..1)
            list.push(if job.plates { matte::extract_raw(&img, chan) } else { matte::extract(&img, chan) });
        }
    }
    Ok((at_a, at_b))
}

/// The layer image of layer `k`: the frame premultiplied by the raw matte of FG `k`; for
/// Main_BG (`k == nfg`, matte 1) the frame itself (`q(c * 1.0)` is `c` at 8 and 16 bpc and
/// `c * 1.0` is `c` in float, so the premultiply is skipped). A matte of another size than the
/// frame is `FxError::Mismatch`.
fn layer_image<'f>(job: &LayerJob, frame: &'f Image, mattes: &[Matte], k: usize) -> Result<Cow<'f, Image>> {
    if k == job.nfg {
        Ok(Cow::Borrowed(frame))
    } else if job.pl_kernels {
        Ok(Cow::Owned(matte::premultiply_pl(frame, &mattes[k])?))
    } else {
        Ok(Cow::Owned(matte::premultiply(frame, &mattes[k])?))
    }
}

/// `src` OVER `dest` for a layer of `job`: the plugin's own kernels when `job.pl_kernels`, else the Smart Blend OVER.
fn over_with(job: &LayerJob, dest: &mut Image, src: &Image) -> Result<()> {
    if job.pl_kernels {
        matte::over_pl(dest, src)
    } else {
        matte::over(dest, src)
    }
}

/// The composite of the group `first..=last` of one unwarped frame (the fast path): from
/// layer `last` down to `first`, the layer image OVER the running result.
fn compose_group(frame: &Image, mattes: &[Matte], job: &LayerJob) -> Result<Image> {
    let mut r = Image::new(frame.width, frame.height, frame.fmt());
    for k in (job.first..=job.last).rev() {
        over_with(job, &mut r, &*layer_image(job, frame, mattes, k)?)?;
    }
    Ok(r)
}

/// The fast path of a layered frame: the main frames `A` and `B`, the FG mattes
/// at both, the error fill check, the composite of the displayed group of each frame
/// ([`compose_group`], no warp, no fill) and `compose::interpolate` of the two composites (the
/// single layer fast path on them). A job with `plates` rebuilds a single group display as the
/// original does ([`rebuilt_plate`]).
pub(crate) fn fast_path(settings: &EngineSettings, host: &dyn EngineHost, job: &LayerJob, sel: &Selection) -> Result<Rendered> {
    if job.full_group_fast_path() {
        return crate::compose::fast_path(settings, host, sel);
    }
    let a = host.main_frame(sel.a)?;
    let b = host.main_frame(sel.b)?;
    let (ma, mb) = fetch_mattes(host, job, sel, &a)?;
    if let Some(fill) = issue_fill(settings, host) {
        return Ok(fill);
    }
    let (ra, rb) = if job.plates && job.rebuild_group && job.first == job.last {
        let fill = FillFrame { is360: settings.wrap_360, taps: settings.stages.get().vertical_taps(), reversed: settings.motion_rows_reversed };
        (rebuilt_plate(&a, &ma, job, fill)?, rebuilt_plate(&b, &mb, job, fill)?)
    } else if job.rebuild_group && job.first == job.last {
        let shrinks = job.shrinks();
        let ea = matte::effective(&ma, &shrinks, a.width, a.height)?;
        let eb = matte::effective(&mb, &shrinks, a.width, a.height)?;
        let k = job.first;
        let fill = FillFrame { is360: settings.wrap_360, taps: settings.stages.get().vertical_taps(), reversed: settings.motion_rows_reversed };
        (reconstructed_layer(&a, &ma, &ea[k], k, job.nfg, fill)?, reconstructed_layer(&b, &mb, &eb[k], k, job.nfg, fill)?)
    } else {
        (compose_group(&a, &ma, job)?, compose_group(&b, &mb, job)?)
    };
    let mut out = Image::new(a.width, a.height, a.fmt());
    interpolate(&mut out, sel, &ra, &rb)?;
    Ok(Rendered::Image(out))
}

/// The image of layer `k` in a single group display of the fast path (`LayerJob::rebuild_group`):
/// the pixels the layers above cover (the layer's visible matte `vis` from `matte::effective`
/// below 1) are holes; they are filled from the other pixels of the frame by the engine's
/// pull-push fill (`holefill::fill_holes_masked` with the mask `1 - vis`) and made opaque (with no
/// known pixel anywhere they become opaque black); the result is premultiplied by the layer's own
/// raw matte (Main_BG's is 1: no premultiply). A transparent pixel of the frame that no layer
/// above covers stays as it is. MEASURED for hard mattes on exact frames of the FG2, FG3 and
/// Background displays; partial alpha (soft mattes, an inverse matte shrink) and 32 bpc are not
/// settled. The fill runs in the frame of the motion path ([`FillFrame`]).
fn reconstructed_layer(frame: &Image, mattes: &[Matte], vis: &Matte, k: usize, nfg: usize, fill: FillFrame) -> Result<Image> {
    let holes: Vec<f32> = vis.data.iter().map(|&v| 1.0f32 - v).collect();
    let mut img = frame.clone();
    fill.fill(&mut img, &holes)?;
    let hole = |i: usize| f64::from(holes[i]) >= crate::holefill::WIDE_EPS;
    match &mut img.samples {
        Samples::U8(s) => s.chunks_exact_mut(CHANNELS).enumerate().filter(|(i, _)| hole(*i)).for_each(|(_, p)| p[A] = 255),
        Samples::U16(s) => s.chunks_exact_mut(CHANNELS).enumerate().filter(|(i, _)| hole(*i)).for_each(|(_, p)| p[A] = 65535),
        Samples::F32(s) => s.chunks_exact_mut(CHANNELS).enumerate().filter(|(i, _)| hole(*i)).for_each(|(_, p)| p[A] = 1.0),
    }
    if k == nfg {
        Ok(img)
    } else {
        matte::premultiply(&img, &mattes[k])
    }
}

/// The frame the hole fill of the layered fast path runs in: the rows of the motion path (`reversed`:
/// `EngineSettings::motion_rows_reversed`, bottom row first) with the vertical tap order of the render's stages (`taps`)
/// and the wrap flag `is360`. TwixtorPro_64's single channel fill walks the plugin's own rows, which
/// run from the bottom of the host frame; with the rows reversed and the tap above first the engine's fill gives the
/// plugin's filled plates bit for bit at 16 and 32 bpc (TRACED), where the
/// host rows give the last bits of the 32 bpc rebuilds.
#[derive(Clone, Copy, Debug)]
struct FillFrame {
    is360: bool,
    taps: VerticalTaps,
    reversed: bool,
}

impl FillFrame {
    /// `holefill::fill_holes_masked_with` of `img` with the hole mask `holes` (one value per pixel, host rows) in this
    /// frame; the image comes back in host rows.
    fn fill(self, img: &mut Image, holes: &[f32]) -> Result<()> {
        if !self.reversed {
            return fill_holes_masked_with(img, HoleMaskRef::F32(holes), self.is360, self.taps);
        }
        let w = img.width;
        let rev: Vec<f32> = if w == 0 { Vec::new() } else { holes.chunks(w).rev().flatten().copied().collect() };
        let mut r = img.rows_reversed();
        fill_holes_masked_with(&mut r, HoleMaskRef::F32(&rev), self.is360, self.taps)?;
        *img = r.rows_reversed();
        Ok(())
    }
}

/// The image of the single layer `k = job.first` of a single group display on the fast path of a render with plates
/// (`LayerJob::plates`): the degenerate plate of the original's render function, which on an exact frame composes the
/// layer image with ITSELF (no warp, `LA OVER LA`), refills it ([`plates::refill`]: unpremultiply, the pull-push fill
/// where the plate is not opaque, the alpha replaced by FG k's raw matte or Main_BG's frame alpha, premultiply) in the
/// frame of the motion path ([`FillFrame`]), and an FG layer is then composed OVER an empty image (TRACED,
/// TwixtorPro_64: `exact-d6-d32`, `shrink0-bg` and `mix00` frame 0, the hole mask is `white` minus the plate's alpha:
/// 0 of 240 mask samples differ, against 220 for the cut plane).
fn rebuilt_plate(frame: &Image, mattes: &[Matte], job: &LayerJob, fill: FillFrame) -> Result<Image> {
    let (w, h, fmt) = (frame.width, frame.height, frame.fmt());
    let k = job.first;
    let raws: Vec<plates::Plane> = mattes.iter().map(|m| plates::raw_plane(m, fmt)).collect();
    let l = plates::layer_times(frame, &plates::comp_plane(&raws, &job.shrinks(), k, fmt, w, h)?);
    let mut p = l.clone();
    matte::over_pl(&mut p, &l)?;
    let flip = |i: &Image| if fill.reversed { i.rows_reversed() } else { i.clone() };
    let mut q = flip(&p);
    if k == job.nfg {
        let fa = flip(frame);
        plates::refill(&mut q, plates::Alpha::Frame(&fa), fill.is360, fill.taps)?;
    } else {
        let r = &raws[k];
        let v: Vec<f32> = if fill.reversed && w > 0 { r.v.chunks(w).rev().flatten().copied().collect() } else { r.v.clone() };
        let rp = plates::Plane { white: r.white, v };
        plates::refill(&mut q, plates::Alpha::Plane(&rp), fill.is360, fill.taps)?;
    }
    let p = flip(&q);
    if k == job.nfg {
        return Ok(p);
    }
    let mut out = Image::new(w, h, fmt);
    matte::over_pl(&mut out, &p)?;
    Ok(out)
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
    let (xa, xb) = match &tracking {
        Some((ta, tb)) => (ta, tb),
        None => (&a, &b),
    };
    if settings.motion_rows_reversed {
        // the whole motion path on bottom-up rows, its output turned back (EngineSettings::motion_rows_reversed)
        let (ra, rb) = (a.rows_reversed(), b.rows_reversed());
        let rtracking = tracking.as_ref().map(|(ta, tb)| (ta.rows_reversed(), tb.rows_reversed()));
        let rmattes = (
            mattes.0.iter().map(Matte::rows_reversed).collect::<Vec<_>>(),
            mattes.1.iter().map(Matte::rows_reversed).collect::<Vec<_>>(),
        );
        let rpts: Vec<Vec<UserPoint>> = pts.iter().map(|l| points_rows_reversed(l, a.height)).collect();
        let (rxa, rxb) = match &rtracking {
            Some((ta, tb)) => (ta, tb),
            None => (&ra, &rb),
        };
        let out = solve_and_compose(settings, sel, job, &ra, &rb, rxa, rxb, &rmattes, &rpts)?;
        return Ok(Rendered::Image(out.rows_reversed()));
    }
    Ok(Rendered::Image(solve_and_compose(settings, sel, job, &a, &b, xa, xb, &mattes, &pts)?))
}

/// The solves of every layer of the group and the composition of [`compose_layers`] (steps 3 to 6 of the module
/// documentation) on the tracking frames `xa`, `xb` and the main frames `a`, `b`.
#[allow(clippy::too_many_arguments)]
fn solve_and_compose(
    settings: &EngineSettings,
    sel: &Selection,
    job: &LayerJob,
    a: &Image,
    b: &Image,
    xa: &Image,
    xb: &Image,
    mattes: &(Vec<Matte>, Vec<Matte>),
    pts: &[Vec<UserPoint>],
) -> Result<Image> {
    let shrinks = job.shrinks();
    if job.plates {
        return plates::solve_and_compose_plates(settings, sel, job, a, b, xa, xb, mattes, pts);
    }
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
    compose_layers(settings, sel, job, a, b, mattes, &flows)
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
/// frames, the FG mattes at `A` and `B`, the error fill check, the mattes of every layer for its
/// loads (the tracking mattes of `matte::effective`; with `plates` the composition mattes of
/// `plates::comp_mattes`, the planes the original's per layer vector loader caps the vector alpha
/// with, READ in TwixtorProVectorsIn_64), then per layer
/// `k` of the group, FG1 first, `flows(sel, k, mattes)` with the layer's mattes at `A` and `B` (the
/// front-end's per layer load) and the inversion of a missing flow of that layer
/// ([`complete_layer_flows`]), then the composition of [`compose_layers`] with Vectors In's settings
/// (no wrap).
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
    let shrinks = job.shrinks();
    let (ea, eb) = if job.plates {
        (plates::comp_mattes(&mattes.0, &shrinks, &a)?, plates::comp_mattes(&mattes.1, &shrinks, &a)?)
    } else {
        (matte::effective(&mattes.0, &shrinks, a.width, a.height)?, matte::effective(&mattes.1, &shrinks, a.width, a.height)?)
    };
    let tail = blur_amount(sel.s0, sel.s1, settings.blur).is_some();
    let mut per_layer = Vec::with_capacity(job.last - job.first + 1);
    for k in job.first..=job.last {
        let mut f = flows(sel, k, &LayerMattes { at_a: &ea[k], at_b: &eb[k] })?;
        complete_layer_flows(settings, sel, &mut f, tail && k == job.last, k)?;
        per_layer.push(f);
    }
    Ok(Rendered::Image(compose_layers(settings, sel, job, &a, &b, &mattes, &per_layer)?))
}

/// The flow loads of an exact frame of a render with external flows and `plates` whose fast path is not the single
/// layer one ([`LayerJob::full_group_fast_path`]): per layer `k` of the group, FG1 first, `flows(sel, k, mattes)` with
/// the layer's composition mattes at `A` and `B` (as [`vector_path_with_flows`] passes them), the results dropped. The
/// original's render function loads the vector layers of every layer of the group on such a frame too and composes the
/// fast path without them (TRACED, TwixtorProVectorsIn_64: the vector decode runs on the exact frames of a single group
/// display), so the reads are the original's; the output does not depend on them.
pub(crate) fn exact_frame_flow_loads<F>(settings: &EngineSettings, host: &dyn EngineHost, job: &LayerJob, sel: &Selection, mut flows: F) -> Result<()>
where
    F: FnMut(&Selection, usize, &LayerMattes<'_>) -> Result<Flows>,
{
    let like = Image::new(settings.width, settings.height, settings.fmt);
    let (ma, mb) = fetch_mattes(host, job, sel, &like)?;
    if issue_fill(settings, host).is_some() {
        return Ok(());
    }
    let shrinks = job.shrinks();
    let (ca, cb) = (plates::comp_mattes(&ma, &shrinks, &like)?, plates::comp_mattes(&mb, &shrinks, &like)?);
    for k in job.first..=job.last {
        let _ = flows(sel, k, &LayerMattes { at_a: &ca[k], at_b: &cb[k] })?;
    }
    Ok(())
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
/// tail with `wrap_360` and the blur flow of layer `last` (`F_AB`, or the one flow of Nearest;
/// U). Without `plates` (no effect) a fill on a single group floods an FG only view.
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
    if job.plates {
        return plates::compose_layers_plates(settings, sel, job, a, b, mattes, flows);
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
            let t = warp_layer(settings, weighted, &*layer_image(job, a, &mattes.0, k)?, side_flow(f, code, sel, true, k)?, frac)?;
            over_with(job, r, &t)?;
        }
        if let Some(r) = rb.as_mut() {
            let t = warp_layer(settings, weighted, &*layer_image(job, b, &mattes.1, k)?, side_flow(f, code, sel, false, k)?, wa)?;
            over_with(job, r, &t)?;
        }
    }
    if fills {
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
        blur_compensation_wrap(&mut out, flow, sel.s0, sel.s1, settings.blur, settings.quality_high, settings.wrap_360)?;
    }
    Ok(out)
}
