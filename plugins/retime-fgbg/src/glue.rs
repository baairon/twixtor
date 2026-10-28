//! Plugin 03's wiring of the shared engine front-end: the single layer path and the n-layer path
//! of the FG matte layers.
//!
//! What the shared `retime_engine::frontend` entry points cannot express for plugin 03 is done
//! here from the public pieces of the engine: the Warping popup (a warp code, not plugin 01's Smart
//! Blend checkbox), the missing `Src Time Offset`, the FG matte probes of the pre-render, the layer
//! job of the FG matte layers, the tracking points and the Draw Geom overlay.
//!
//! The render of a frame, after the error fills of the probes:
//! * the Source display (`Display == 2`) shows Alt Source A at the current time `X`, or the input
//!   layer when there is no Alt Source A, then draws the tracking points of the Draw Geom group
//!   over it ([`draw_geom_points`]; rule G3);
//! * every other display runs `compose::retime_layers` with the layer job of the FG matte layers
//!   ([`layer_job`]: the count `n` of the probe loop, which stops at the first empty one, the per
//!   layer controls and the group of the Display popup) and the tracking points in use at the
//!   two source frame times ([`user_points`]). With `n = 0` and no point in use that is the
//!   single layer core of today, byte for byte, and the Display values 3 to 6 all
//!   select the one group `0..=0`, so they render exactly like the processed output.
//!
//! `Track RGB+A` is the engine's wrap-around (360) flag (`EngineSettings::wrap_360`, rule G6): the solver, the inverse warps, the hole fill and the blur tail wrap horizontally;
//! it does not add alpha to the luma. Mask paths (First/Last Mask, layer masks) feed the same
//! solver input as the points but are parameter kinds `fx-core` cannot express, so they are not
//! part of this crate's input.

use crate::prerender::{self, PreRender03, LAYER_FG_MATTE};
use crate::timemap::{CurveParams03, RetimeFgBgTimeMap};
use crate::{FG_CHAN_IDS, FG_SENS_IDS, FG_SHRINK_IDS, PT_COLOR_IDS, PT_POS_IDS, PT_USE_IDS};
use fx_core::{Checkout, Env, FieldOrder, Frame, FxError, ParamCurves, Result, TemporalContext};
use retime_engine::compose::{self, Rendered};
use retime_engine::frontend::{self, fixed_reader, SettingIds, TimeParamIds, Wiring};
use retime_engine::geom::draw_points;
use retime_engine::host::{layer_is_set, EngineHost, EngineSettings, EnvHost, LayerIds};
use retime_engine::image::{error_frame, MissingLayerStyle};
use retime_engine::layers::{LayerJob, LayerParams};
use retime_engine::planner::{LAYER_ALT_A, LAYER_ALT_B};
use retime_engine::points::{point_layer, read_pos, UserPoint};
use retime_engine::timeline::{frame_positions, Selection, TimeMap};

/// Plugin 03's parameter and layer ids for the shared settings reader (`frontend::read_settings`,
/// the only function of `frontend` this crate calls). Differences from plugin 01's table:
/// `Sensitivity` is `main_bg_sensitivity` (the single layer's `Main_BG: Sensitivity`); there is no
/// `Src Time Offset` and no `Smart Blend` checkbox (the Warping popup replaces it, see
/// [`warp_code`]); `time.src_time_offset` is never read (the time map of this crate has no offset)
/// and only names an existing parameter. The missing layer fill is the blue one with alpha 0 at 16
/// and 32 bpc.
pub const WIRING: Wiring = Wiring {
    settings: SettingIds {
        display: Some("display"),
        video_360: Some("video_360"),
        input_fields: Some("input_fields"),
        in_fps_is_out_fps: Some("in_fps_is_out_fps"),
        input_frame_rate: Some("input_frame_rate"),
        src_time_offset: None,
        vectors: Some("vectors"),
        prep: Some("prep"),
        sensitivity: Some("main_bg_sensitivity"),
        plenty_memory: Some("plenty_memory"),
        frame_interp: Some("frame_interp"),
        smart_blend: None,
        blur_compensation: Some("blur_compensation"),
    },
    time: TimeParamIds {
        remap_mode: "time_remap_mode",
        speed: "speed_percent",
        frame_num: "frame_num",
        src_time_offset: "speed_percent",
        mark_segs: "src_mark_segments",
    },
    layers: LayerIds { alt_a: LAYER_ALT_A, alt_b: LAYER_ALT_B },
    missing_layer: MissingLayerStyle::BlueAlphaSlip,
    register_layers: true,
};

/// The Warping popup as the warp code of the engine: 1 Inverse is 0, 2 Inverse w/ Smart Blend 1, 3
/// Forward 2, 4 Forward, no Smart Blend 3; any other value keeps 0.
pub fn warp_code(popup_value: i32) -> i32 {
    if (1..=4).contains(&popup_value) {
        popup_value - 1
    } else {
        0
    }
}

/// The FG matte probes of the pre-render: the count of consecutive present layers starting at FG1
/// (the loop stops at the first empty one) and whether one of them has another size than the
/// output.
pub fn fg_probe(env: &Env<'_>, width: usize, height: usize) -> (usize, bool) {
    let mut count = 0;
    let mut mismatch = false;
    for id in LAYER_FG_MATTE {
        let layer = env.layer(id);
        if !layer_is_set(layer) {
            break;
        }
        count += 1;
        if let Some(l) = layer {
            mismatch |= (l.width(), l.height()) != (width, height);
        }
    }
    (count, mismatch)
}

/// `Pt n: Use` points (1 based) in use at the two times: the value is not "Don't Use" at the first
/// time and the same at the second. With no FG layer every value 2 to 5 maps to the only
/// group.
pub fn points_in_use(curves: &ParamCurves, time1: i32, time2: i32) -> Vec<usize> {
    let mut out = Vec::new();
    for (i, id) in PT_USE_IDS.iter().enumerate() {
        let v1 = curves.int_at(id, i64::from(time1));
        if v1 != 1 && curves.int_at(id, i64::from(time2)) == v1 {
            out.push(i + 1);
        }
    }
    out
}

/// The settings of the engine for plugin 03: the shared reader plus the warp code of the Warping
/// popup and `Track RGB+A` as the engine's wrap-around flag, both read at `current_time`
/// (all popups and checkboxes are sampled there).
pub fn read_settings(cx: &TemporalContext, env: &Env<'_>) -> Result<EngineSettings> {
    let mut settings = frontend::read_settings(&WIRING, cx, env)?;
    settings.warp_mode = warp_code(env.params.int_at("warping", cx.current_time) as i32);
    settings.wrap_360 = env.params.bool_at("track_rgb_a", cx.current_time);
    Ok(settings)
}

/// The controls of FG layer `k` (0 is FG1) at `time`: `fgk_matte_chan`, `fgk_sensitivity`
/// through the 16.16 reader divided by 100 (as `main_bg_sensitivity` is read), and
/// `fgk_inv_matte_shrink` through the 16.16 reader, in pixels (`downsample_ratio_y * value`,
/// the ratio is 1 here; I: the slider is read as the engine's other sliders are). Plugin 03 has no
/// Track control: every layer tracks.
pub fn fg_layer(curves: &ParamCurves, time: i64, k: usize) -> LayerParams {
    LayerParams {
        matte_chan: curves.int_at(FG_CHAN_IDS[k], time) as i32,
        sensitivity: fixed_reader(curves.fixed_at(FG_SENS_IDS[k], time), 1.0f32) / 100.0f32,
        shrink: fixed_reader(curves.fixed_at(FG_SHRINK_IDS[k], time), 1.0f32),
        track: true,
    }
}

/// The layer job of a render with `fg_count` FG matte layers: FG1 to FG`n` from [`fg_layer`]
/// at `current_time`, Main_BG with the settings' sensitivity (`main_bg_sensitivity`, the one the
/// single layer path reads) and no matte, and the group of the Display popup
/// (`LayerJob::display_group`).
pub fn layer_job(settings: &EngineSettings, curves: &ParamCurves, fg_count: usize) -> LayerJob {
    let t = i64::from(settings.current_time);
    let fg: Vec<LayerParams> = (0..fg_count).map(|k| fg_layer(curves, t, k)).collect();
    let main_bg = LayerParams { matte_chan: 0, sensitivity: settings.sensitivity, shrink: 0.0, track: true };
    LayerJob::for_display(&fg, main_bg, curves.int_at("display", t) as i32)
}

/// The tracking points of the pair of `sel`: every point in use at the layer times
/// `T(a)` and `T(b)` ([`points_in_use`]) goes to the layer `points::point_layer` names (values 2 to
/// 5 to FG1, FG2, FG3 and Main_BG, a missing layer to Main_BG), in point order, with its position
/// read by the 16.16 reader at both times (`points::read_pos`). `nfg + 1` lists, Main_BG last.
pub fn user_points(curves: &ParamCurves, map: &dyn TimeMap, sel: &Selection, nfg: usize) -> Vec<Vec<UserPoint>> {
    let ta = map.frame_time(sel.a as f32);
    let tb = map.frame_time(sel.b as f32);
    let mut out = vec![Vec::new(); nfg + 1];
    for n in points_in_use(curves, ta, tb) {
        let v = curves.int_at(PT_USE_IDS[n - 1], i64::from(ta));
        if let Some(k) = point_layer(v, nfg) {
            let id = PT_POS_IDS[n - 1];
            out[k].push(UserPoint { at_a: read_pos(curves, id, ta), at_b: read_pos(curves, id, tb) });
        }
    }
    out
}

/// The layers whose points the `draw_geom` popup value `mode` draws: 2 All, every layer
/// `0..=nfg`; 3 to 6 the single group `min(mode - 3, nfg)` (FG1, FG2, FG3, Main_BG); 1 None and
/// any value outside the popup draw nothing (U, ours).
pub fn draw_geom_group(mode: i64, nfg: usize) -> Option<(usize, usize)> {
    match mode {
        2 => Some((0, nfg)),
        3..=6 => {
            let g = ((mode - 3) as usize).min(nfg);
            Some((g, g))
        }
        _ => None,
    }
}

/// The Draw Geom overlay of the Source display (rule G3): with
/// `draw_geom` other than None (read at `current_time`), the points in use at the rounded frame
/// time `T(t0)` (`t0` the output position of the frame) whose layer is in the Draw Geom group, in
/// point order, each at its position and in its `ptN_color` at that time, as
/// `geom::draw_points` draws them (an opaque disc of radius 1.5 times the downsample ratio, 1
/// here). Only an engine image is drawn on; an error fill stays as it is.
pub fn draw_geom_points(curves: &ParamCurves, map: &dyn TimeMap, settings: &EngineSettings, nfg: usize) -> Vec<([f32; 2], [u8; 3])> {
    let Some((first, last)) = draw_geom_group(curves.int_at("draw_geom", i64::from(settings.current_time)), nfg) else {
        return Vec::new();
    };
    let (t0, _) = frame_positions(settings.current_time, settings.time_step, settings.local_time_step);
    let at = map.frame_time(t0);
    let mut pts = Vec::new();
    for n in points_in_use(curves, at, at) {
        let v = curves.int_at(PT_USE_IDS[n - 1], i64::from(at));
        if let Some(k) = point_layer(v, nfg) {
            if first <= k && k <= last {
                pts.push((read_pos(curves, PT_POS_IDS[n - 1], at), curves.color_at(PT_COLOR_IDS[n - 1], i64::from(at))));
            }
        }
    }
    pts
}

fn time_map<'e>(settings: &EngineSettings, env: &Env<'e>) -> Result<RetimeFgBgTimeMap<CurveParams03<'e>>> {
    Ok(RetimeFgBgTimeMap::new(CurveParams03::new(env.params)?, settings.rates, settings.current_time))
}

/// `Effect::dependencies`: the pre-render list (`prerender::checkouts`).
pub(crate) fn dependencies(cx: &TemporalContext, env: &Env<'_>) -> Result<Vec<Checkout>> {
    let settings = read_settings(cx, env)?;
    let map = time_map(&settings, env)?;
    let (fg_count, _) = fg_probe(env, settings.width, settings.height);
    let req = PreRender03 {
        current_time: settings.current_time,
        time_step: settings.time_step,
        local_time_step: settings.local_time_step,
        display_source: settings.display_source,
        alt_a_present: settings.alt_a_present,
        alt_b_present: !settings.no_tracking_layer,
        fg_count,
        input_fields: settings.input_fields != FieldOrder::Progressive,
        vectors_quality: settings.vectors_quality,
        interp: settings.interp,
        blur: settings.blur,
        register_layers: WIRING.register_layers,
    };
    prerender::checkouts(&req, &map)
}

/// `Effect::render_temporal`: settings, time map, host, [`render`], the finish.
pub(crate) fn render_temporal(cx: &TemporalContext, env: &Env<'_>, output: &mut Frame) -> Result<()> {
    if (output.width, output.height, output.depth()) != (cx.width, cx.height, cx.depth) {
        return Err(FxError::Mismatch(format!(
            "the output frame is {}x{} at {} bpc but the context says {}x{} at {} bpc",
            output.width,
            output.height,
            output.depth().bits(),
            cx.width,
            cx.height,
            cx.depth.bits()
        )));
    }
    let settings = read_settings(cx, env)?;
    let map = time_map(&settings, env)?;
    let host = EnvHost::new(env, &settings, &map, WIRING.layers)?;
    let rendered = render(&settings, env, host, &map)?;
    frontend::finish(rendered, output);
    Ok(())
}

/// The render of one frame: error flags, then the Source display with the Draw Geom overlay, then
/// the layered core (`compose::retime_layers`, which is today's core for no FG layer and no point).
fn render<'e>(
    settings: &EngineSettings,
    env: &'e Env<'_>,
    host: EnvHost<'e>,
    map: &RetimeFgBgTimeMap<CurveParams03<'_>>,
) -> Result<Rendered> {
    // Error flags first: the size probes of the FG mattes set the same size mismatch flag as the
    // other layers; red wins over the missing layer fill.
    let (fg_count, fg_mismatch) = fg_probe(env, settings.width, settings.height);
    let mut issues = host.issues();
    issues.size_mismatch |= fg_mismatch;
    if let Some(kind) = issues.error_kind(settings.missing_layer_style) {
        return Ok(Rendered::Error(error_frame(kind, settings.width, settings.height, settings.fmt.depth())));
    }
    let curves = env.params;
    if settings.display_source {
        // 6.4: the Draw Geom overlay runs after the copy of the source; with no geometry it draws nothing
        let mut rendered = compose::render(settings, &host)?;
        let pts = draw_geom_points(curves, map, settings, fg_count);
        if let Rendered::Image(img) = &mut rendered {
            draw_points(img, &pts, 1.0f32);
        }
        return Ok(rendered);
    }
    let host = host.with_mattes(env, &LAYER_FG_MATTE[..fg_count])?;
    let job = layer_job(settings, curves, fg_count);
    compose::retime_layers(settings, &host, &job, |sel| Ok(user_points(curves, map, sel, fg_count)))
}
