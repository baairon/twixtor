//! The vector path of the retiming core: optical flow motion interpolation.
//!
//! It runs when `vectors_quality != 0 && !exact` after the segment override. The solver reads the
//! frames `A` and `B = A + 1` of the tracking layer when one is set; the output is always built
//! from the retimed layer's frames. The steps are the solves ([`solve_flows`]), the warps and
//! composition ([`compose_vectors`]) and the blur compensation tail. The "360 Video" checkbox is
//! not the `is360` argument of the warps: it feeds the video range luma of the solver and selects
//! the binary alpha Smart Blend variant. The engine's wrap-around (360) flag
//! `EngineSettings::wrap_360` (retime-fgbg's `Track RGB+A`) is: it
//! reaches `FlowSettings::wrap`, the inverse warps of code 0, the `is360` of the Smart Blend
//! options (codes 1 and 2) and the blur tail ([`blur_compensation_wrap`]); the weighted Smart Blend
//! and forward compositions and the Vectors In flow steps keep it off. With the flag off every
//! call is the one this path made before the flag existed, bit for bit.

use super::{issue_fill, Rendered};
use crate::blend::blend;
use crate::filters::sharpen_step;
use crate::flow::{FlowField, FlowPair, FlowSettings};
use crate::forward::{blend_forward_fill_on, blend_forward_on, nearest_forward_fill_on, nearest_forward_on, weighted_forward_fill_on, weighted_forward_on};
use crate::stages::{Cpu, Stages};
use crate::holefill::{fill_flow_holes, HoleMaskRef};
use crate::host::{EngineHost, EngineSettings};
use crate::image::{Fmt, Image, Samples, CHANNELS};
use crate::smartblend::{blend_smart, nearest_smart, weighted_blend_smart, SmartOpts};
use crate::timeline::Selection;
use crate::vectors::{apply_vec_scale, decode_vectors_with_holes, HoleMask};
use crate::warp::{blur_amount, blur_compensation_wrap, inv_warp, WarpOpts};
use fx_core::{FxError, Result};

#[cfg(test)]
mod tests;

/// The flow fields the composition needs: `F_AB` and `F_BA`. Blend and Weighted Blend need both;
/// Nearest needs the one flow the chosen warp uses ([`nearest_uses_f_ab`]) and the other may be
/// missing. The fields have the solver's working resolution; the warps resample them.
#[derive(Clone, Debug, PartialEq)]
pub struct Flows {
    /// `A(x) ~ B(x + f_ab(x))`, on the grid of A (the solve with `X = A`, `Y = B`).
    pub f_ab: Option<FlowField>,
    /// `B(x) ~ A(x + f_ba(x))`, on the grid of B (the solve with `X = B`, `Y = A`).
    pub f_ba: Option<FlowField>,
}

impl Flows {
    /// Both flows of a Blend / Weighted Blend render.
    pub fn pair(pair: FlowPair) -> Flows {
        Flows { f_ab: Some(pair.f_ab), f_ba: Some(pair.f_ba) }
    }

    /// The flow the blur compensation tail uses: `F_AB` when it exists, else the solver's finest
    /// flow, which for Nearest is the single flow.
    pub fn blur_flow(&self) -> Option<&FlowField> {
        self.f_ab.as_ref().or(self.f_ba.as_ref())
    }
}

/// Which solve and which flow the Nearest warp uses: true when
/// `(wA < 0.5 && code < 2) || (wA >= 0.5 && code > 1)`. True gives the solve `X = A, Y = B` and
/// the flow `F_AB`, false `X = B, Y = A` and `F_BA`. `code` is the warping code, compared as an
/// unsigned number.
pub fn nearest_uses_f_ab(wa: f32, code: i32) -> bool {
    let code = code as u32;
    (wa < 0.5 && code < 2) || (0.5 <= wa && code > 1)
}

/// The settings that reach the solver for this render: the quality after the segment override,
/// the render quality flag, the sensitivity, the two Prep flags, the "360 Video" checkbox as
/// the video range luma and the engine's wrap-around flag `wrap_360` as `FlowSettings::wrap`
/// (off it is the plain solve). `Track RGB+A` is never alpha in the luma in the retime
/// (`track_alpha` false).
pub fn flow_settings(settings: &EngineSettings, sel: &Selection) -> FlowSettings {
    let quality = u8::try_from(sel.vectors_quality).unwrap_or(u8::MAX);
    FlowSettings {
        prep_contrast_edge: settings.prep_contrast_edge,
        prep_delinearize: settings.prep_delinearize,
        video_range_luma: settings.video_360,
        wrap: settings.wrap_360,
        ..FlowSettings::new(quality, settings.quality_high, settings.sensitivity)
    }
}

/// The settings the vector path cannot serve: an error that names what is missing, so a render
/// never falls back to the frame blend by itself. The warping code is `settings.warp_mode`
/// (plugin 01's front-end derives it from its Smart Blend checkbox, plugins 03 and 04 from their
/// Warping popup): 0 inverse warp, 1 inverse warp with Smart Blend, 2 forward warp with coverage
/// and hole fill, 3 forward warp without coverage are all served; any other code is a bad
/// parameter. An interpolation mode other than 0 to 2 is a bad parameter too. The "360 Video"
/// checkbox is not here: it is the video range luma of the solver and the binary alpha variant of
/// the Smart Blend composite, not the `is360` argument of the warps (see the module
/// documentation). What the compositions themselves cannot do (a frame below 2 x 2, the forward
/// warp geometry that writes outside the image) comes back from them as `FxError::Unsupported`
/// with their own text. The engine's wrap-around (360) flag `settings.wrap_360` is served (the
/// module documentation says where it reaches).
pub fn check_supported(settings: &EngineSettings, sel: &Selection) -> Result<()> {
    match settings.warp_mode {
        0..=3 => {}
        other => {
            return Err(FxError::BadParam { id: "warp_mode".into(), reason: format!("warping code {other} is not 0 to 3") })
        }
    }
    if !(0..=2).contains(&sel.interp) {
        return Err(bad_interp(sel.interp));
    }
    Ok(())
}

/// The compositions of the forward warp codes 2 (coverage, inverted coverage, hole fill) and 3
/// (no coverage), using the forward warp of [`crate::forward`] (the textured pyramid variant).
/// The pushed frame uses its OWN flow: `A` with `F_AB`, `B` with `F_BA`, Nearest the one flow
/// [`nearest_uses_f_ab`] names (with code 2 or 3 it is `F_AB` for `wA >= 0.5`, the opposite of
/// the inverse warps). Code 2 passes
/// `SmartOpts { binary_alpha: "360 Video" checkbox (Blend only), is360: wrap_360 }` (the forward
/// warp itself never wraps, its hole fill does; the weighted variant keeps the fill off); code 3
/// has no options. The blur compensation tail of [`compose_vectors`] follows as for the inverse
/// warps.
fn compose_forward(settings: &EngineSettings, sel: &Selection, a: &Image, b: &Image, flows: &Flows, out: &mut Image) -> Result<()> {
    let wa = sel.wa;
    let fill = settings.warp_mode == 2;
    let opts = SmartOpts { binary_alpha: settings.video_360, is360: settings.wrap_360 };
    let stages = settings.stages.get();
    match sel.interp {
        0 => {
            let flow = if nearest_uses_f_ab(wa, settings.warp_mode) {
                flows.f_ab.as_ref().ok_or_else(|| missing_flow("F_AB"))?
            } else {
                flows.f_ba.as_ref().ok_or_else(|| missing_flow("F_BA"))?
            };
            if fill {
                nearest_forward_fill_on(stages, out, a, b, flow, wa, opts)
            } else {
                nearest_forward_on(stages, out, a, b, flow, wa)
            }
        }
        1 | 2 => {
            let f_ab = flows.f_ab.as_ref().ok_or_else(|| missing_flow("F_AB"))?;
            let f_ba = flows.f_ba.as_ref().ok_or_else(|| missing_flow("F_BA"))?;
            match (sel.interp, fill) {
                (1, true) => blend_forward_fill_on(stages, out, a, f_ab, b, f_ba, wa, opts),
                (1, false) => blend_forward_on(stages, out, a, f_ab, b, f_ba, wa),
                (_, true) => weighted_forward_fill_on(stages, out, a, f_ab, b, f_ba, wa, opts),
                (_, false) => weighted_forward_on(stages, out, a, f_ab, b, f_ba, wa),
            }
        }
        other => Err(bad_interp(other)),
    }
}

fn bad_interp(other: i32) -> FxError {
    FxError::BadParam {
        id: "frame_interp".into(),
        reason: format!("interpolation mode {other} is not 0 (Nearest), 1 (Blend) or 2 (Weighted Blend)"),
    }
}

/// The solves of the vector path. `xa` and `xb` are the frames `A` and `B = A + 1` the solver
/// reads: the tracking layer's when there is one, else the main frames.
///
/// Nearest runs ONE solve, the one [`nearest_uses_f_ab`] selects, and fills only that flow. Blend
/// and Weighted Blend run the two solves (`X = B, Y = A` then `X = A, Y = B`) as one
/// [`solve_pair`], whose results are identical to two [`solve`] calls (the pyramid of a frame
/// does not depend on its role, so they are built once).
pub fn solve_flows(settings: &EngineSettings, sel: &Selection, xa: &Image, xb: &Image) -> Result<Flows> {
    check_supported(settings, sel)?;
    let fs = flow_settings(settings, sel);
    let stages = settings.stages.get();
    match sel.interp {
        0 => {
            if nearest_uses_f_ab(sel.wa, settings.warp_mode) {
                Ok(Flows { f_ab: Some(stages.solve(xa, xb, &fs)?), f_ba: None })
            } else {
                Ok(Flows { f_ab: None, f_ba: Some(stages.solve(xb, xa, &fs)?) })
            }
        }
        1 | 2 => Ok(Flows::pair(stages.solve_pair(xa, xb, &fs)?)),
        other => Err(bad_interp(other)),
    }
}

fn missing_flow(which: &str) -> FxError {
    FxError::Mismatch(format!("the composition needs the flow {which} but it was not computed"))
}

/// The warps, the composition and the blur compensation tail for given flows and the warping code
/// `settings.warp_mode`. `a` and `b` are the frames `A` and `B = A + 1` of the retimed layer
/// (premultiplied engine images of the output size), `sel` gives `wA`, the interpolation mode and
/// the source positions `s0`, `s1` of the blur tail. The flows may have any size of at least
/// 2 x 2; Nearest only reads the one [`nearest_uses_f_ab`] names, Blend and Weighted Blend read
/// both.
///
/// Code 0 is the inverse warp without coverage, with `frac = 1.0f - wA` in single precision and
/// `W` the inverse warp with hold edges, mode 1 and `is360 = wrap_360` ([`crate::warp::inv_warp`]
/// with the default options when the flag is off):
/// * Nearest: `wA >= 0.5` gives `W(A, F_BA, 1 - wA)`, else `W(B, F_AB, wA)`; the warp writes the
///   output;
/// * Blend: `tA = W(A, F_BA, frac)`, `tB = W(B, F_AB, wA)`, `out = blend(tA, tB, frac)`;
/// * Weighted Blend: the same two warps, each followed by the sharpening step
///   ([`crate::filters::sharpen_step`], strength position `frac` for A and `wA` for B), then
///   `out = blend(tA', tB', frac)`.
///
/// Code 1 (Smart Blend) runs the same warps with hold edges off and one coverage image each and
/// then the compositions of [`crate::smartblend`] (`nearest_smart`, `blend_smart`,
/// `weighted_blend_smart`: the Smart Blend composite, or the binary alpha composite for Blend
/// when the "360 Video" checkbox is set, the combined and inverted coverage, the pull-push hole
/// fill). Codes 2 and 3 push the frames with the forward warp ([`compose_forward`]). The blur
/// compensation tail ([`blur_compensation_wrap`] with `wrap_360`, which is
/// [`crate::warp::blur_compensation`] with the flag off) runs last on the composed image, with
/// the flow `F_AB` (Blend, Weighted Blend) or the single flow (Nearest).
pub fn compose_vectors(settings: &EngineSettings, sel: &Selection, a: &Image, b: &Image, flows: &Flows) -> Result<Image> {
    check_supported(settings, sel)?;
    let wa = sel.wa;
    // frac = 1.0f - wA, the position after frame A; the amount of A and the weight of B
    let frac = 1.0f32 - wa;
    let opts = WarpOpts { is360: settings.wrap_360, ..WarpOpts::default() };
    let mut out = Image::new(a.width, a.height, a.fmt());
    let smart = settings.warp_mode == 1;
    // The Smart Blend compositions take the "360 Video" checkbox as `binary_alpha` (the binary
    // alpha composite, Blend only) and the engine's wrap flag as `is360` (the weighted variant
    // ignores it).
    let smart_opts = SmartOpts { binary_alpha: settings.video_360, is360: settings.wrap_360 };
    if settings.warp_mode >= 2 {
        compose_forward(settings, sel, a, b, flows, &mut out)?;
    } else if smart {
        // warping code 1 (ENGINE-timeline 14.2): the same warps with hold edges off and one coverage image each, the
        // Smart Blend composite, the combined and inverted coverage, the hole fill (smartblend.rs, holefill.rs)
        match sel.interp {
            0 => {
                let flow = if nearest_uses_f_ab(wa, settings.warp_mode) {
                    flows.f_ab.as_ref().ok_or_else(|| missing_flow("F_AB"))?
                } else {
                    flows.f_ba.as_ref().ok_or_else(|| missing_flow("F_BA"))?
                };
                nearest_smart(&mut out, a, b, flow, wa, smart_opts)?;
            }
            1 | 2 => {
                let f_ab = flows.f_ab.as_ref().ok_or_else(|| missing_flow("F_AB"))?;
                let f_ba = flows.f_ba.as_ref().ok_or_else(|| missing_flow("F_BA"))?;
                if sel.interp == 1 {
                    blend_smart(&mut out, a, f_ba, b, f_ab, wa, smart_opts)?;
                } else {
                    weighted_blend_smart(&mut out, a, f_ba, b, f_ab, wa, smart_opts)?;
                }
            }
            other => return Err(bad_interp(other)),
        }
    } else {
        match sel.interp {
            0 => {
                // one warp straight into the output: A with F_BA for wA >= 0.5, else B with F_AB
                let flow = if nearest_uses_f_ab(wa, settings.warp_mode) {
                    flows.f_ab.as_ref().ok_or_else(|| missing_flow("F_AB"))?
                } else {
                    flows.f_ba.as_ref().ok_or_else(|| missing_flow("F_BA"))?
                };
                if 0.5 <= wa {
                    inv_warp(&mut out, a, flow, frac, opts)?;
                } else {
                    inv_warp(&mut out, b, flow, wa, opts)?;
                }
            }
            1 | 2 => {
                let f_ab = flows.f_ab.as_ref().ok_or_else(|| missing_flow("F_AB"))?;
                let f_ba = flows.f_ba.as_ref().ok_or_else(|| missing_flow("F_BA"))?;
                let mut ta = Image::new(a.width, a.height, a.fmt());
                let mut tb = Image::new(b.width, b.height, b.fmt());
                inv_warp(&mut ta, a, f_ba, frac, opts)?;
                inv_warp(&mut tb, b, f_ab, wa, opts)?;
                if sel.interp == 2 {
                    // the sharpening step of each warped frame against the unwarped frame, the
                    // strength position being frac for A and wA for B
                    sharpen_step(&mut ta, a, frac);
                    sharpen_step(&mut tb, b, wa);
                }
                blend(&mut out, &ta, &tb, frac);
            }
            other => return Err(bad_interp(other)),
        }
    }
    // 14.4: after the composition, with the flow F_AB (Nearest: the single flow)
    if let Some(flow) = flows.blur_flow() {
        blur_compensation_wrap(&mut out, flow, sel.s0, sel.s1, settings.blur, settings.quality_high, settings.wrap_360)?;
    }
    Ok(out)
}

/// The whole vector path of the core for one output frame: fetches (tracking layer first, then
/// the retimed layer), the error fills of the layer checks, [`solve_flows`] and
/// [`compose_vectors`]. The output is the engine image; the caller applies the finish.
pub(super) fn vector_path(settings: &EngineSettings, host: &dyn EngineHost, sel: &Selection) -> Result<Rendered> {
    check_supported(settings, sel)?;
    let tracking = if settings.no_tracking_layer {
        None
    } else {
        let ta = host.tracking_frame(sel.a)?;
        let tb = host.tracking_frame(sel.b)?;
        Some((ta, tb))
    };
    let a = host.main_frame(sel.a)?;
    let b = host.main_frame(sel.b)?;
    if let Some(fill) = issue_fill(settings, host) {
        return Ok(fill);
    }
    let (xa, xb) = match &tracking {
        Some((ta, tb)) => (ta, tb),
        None => (&a, &b),
    };
    let flows = solve_flows(settings, sel, xa, xb)?;
    Ok(Rendered::Image(settings.stages.get().compose_vectors(settings, sel, &a, &b, &flows)?))
}

/// The flow field of one vector layer image: the encoded image is converted with `max_displace`
/// (`Max Displace`, read once at `t0`; alpha 0 is the zero vector), the alpha 0 holes of the flow
/// are filled with the pull-push fill ([`fill_flow_holes`], 360 flag off), then the flow is
/// multiplied by `Vec Scale X / Y` (`scale`, read at the same frame, default `(1.0, 1.0)`;
/// applied even for 1.0). `vec` is the premultiplied engine image of the layer (opaque except
/// alpha 0 holes) and has the output size; the flow comes out at that size, in output pixels.
pub fn flow_from_vector_image(vec: &Image, max_displace: f32, scale: (f32, f32)) -> Result<FlowField> {
    let (mut flow, mask) = decode_vectors_with_holes(vec, max_displace);
    let mask_ref = match &mask {
        HoleMask::U8(m) => HoleMaskRef::U8(m),
        HoleMask::U16(m) => HoleMaskRef::U16(m),
        HoleMask::F32(m) => HoleMaskRef::F32(m),
    };
    fill_flow_holes(&mut flow, mask_ref, false)?;
    apply_vec_scale(&mut flow, scale.0, scale.1);
    Ok(flow)
}

/// The flow inversion that derives the missing flow of Vectors In when only one vector layer
/// exists:
///
/// 1. the flow data must match its size (the flow is the only argument and has its own size);
/// 2. a temporary four channel float image `T` is filled with 1.0, its channel 1 (`R`) receives
///    channel 0 of the flow and its channel 2 (`G`) channel 1, and then every channel is
///    multiplied by -1.0 (a product, so 0.0 gives -0.0): `T = (-1, -fx, -fy, -1)`;
/// 3. the forward warp with a coverage image (`forward::forward_warp_coverage`, textured pyramid
///    variant, amount 1.0) pushes `T` along the flow itself: the value of pixel `p` lands at
///    `p + flow(p)`, several landings average;
/// 4. the coverage is inverted (255 where nothing landed), channels 1 and 2 of the pushed image
///    become the two components of the output flow, and the holes are filled with the pull-push
///    fill ([`fill_flow_holes`], u8 mask, 360 flag off).
///
/// So `out(p + f(p)) = -f(p)`: the displacement back from the pixel that `p` moved to. Everything
/// the forward warp refuses (a flow below 2 x 2, the geometry that writes outside the image)
/// comes back as its `FxError::Unsupported`.
pub fn invert_flow(flow: &FlowField) -> Result<FlowField> {
    invert_flow_on(&Cpu, flow)
}

/// [`invert_flow`] with the forward warp on `stages`.
pub fn invert_flow_on(stages: &dyn Stages, flow: &FlowField) -> Result<FlowField> {
    let (w, h) = (flow.width, flow.height);
    if flow.data.len() != w * h * 2 {
        return Err(FxError::Mismatch("flow inversion: flow data length does not match its size".into()));
    }
    // T = (-1, -fx, -fy, -1): 1.0 fill, the flow into channels 1 and 2, times -1.0 (the product, so 0.0 gives -0.0)
    let mut t = Image::new(w, h, Fmt::F32);
    if let Samples::F32(v) = &mut t.samples {
        for (px, f) in v.chunks_exact_mut(CHANNELS).zip(flow.data.chunks_exact(2)) {
            px[0] = -1.0f32 * 1.0f32;
            px[1] = -1.0f32 * f[0];
            px[2] = -1.0f32 * f[1];
            px[3] = -1.0f32 * 1.0f32;
        }
    }
    let mut pushed = Image::new(w, h, Fmt::F32);
    let mut cov = stages.forward_warp_coverage(&mut pushed, &t, flow, 1.0)?;
    cov.invert();
    let mut out = FlowField::new(w, h);
    if let Samples::F32(v) = &pushed.samples {
        for (o, px) in out.data.chunks_exact_mut(2).zip(v.chunks_exact(CHANNELS)) {
            o[0] = px[1];
            o[1] = px[2];
        }
    }
    fill_flow_holes(&mut out, HoleMaskRef::U8(&cov.data), false)?;
    Ok(out)
}

/// The vector path with externally supplied flows ([`super::retime_with_flows`]): the Vectors In
/// branch of the core. The flows come from the caller instead of the solver, a missing one is
/// derived ([`complete_flows`]), then the composition is the same as with the solver.
pub(super) fn vector_path_with_flows<F>(settings: &EngineSettings, host: &dyn EngineHost, sel: &Selection, flows: F) -> Result<Rendered>
where
    F: FnOnce(&Selection) -> Result<Flows>,
{
    check_supported(settings, sel)?;
    let mut flows = flows(sel)?;
    complete_flows(settings, sel, &mut flows)?;
    let a = host.main_frame(sel.a)?;
    let b = host.main_frame(sel.b)?;
    if let Some(fill) = issue_fill(settings, host) {
        return Ok(fill);
    }
    Ok(Rendered::Image(settings.stages.get().compose_vectors(settings, sel, &a, &b, &flows)?))
}

/// With exactly one flow loaded the other one is its inversion ([`invert_flow`]). Done lazily:
/// the missing flow is derived only when this render reads it, which is the case when the
/// interpolation mode needs it (Blend and Weighted Blend both, Nearest the one
/// [`nearest_uses_f_ab`] names) or when the blur compensation tail will run (it reads `F_AB` when
/// it exists).
fn complete_flows(settings: &EngineSettings, sel: &Selection, flows: &mut Flows) -> Result<()> {
    let tail = blur_amount(sel.s0, sel.s1, settings.blur).is_some();
    let (need_ab, need_ba) = match sel.interp {
        0 => {
            let ab = nearest_uses_f_ab(sel.wa, settings.warp_mode);
            (ab || tail, !ab)
        }
        _ => (true, true),
    };
    match (&flows.f_ab, &flows.f_ba) {
        (None, None) => Err(FxError::Mismatch("Vectors In: neither vector layer supplied a flow".into())),
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
