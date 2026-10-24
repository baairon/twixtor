//! Plugin 03's wiring of the shared engine front-end, single layer use.
//!
//! What the shared `retime_engine::frontend` entry points cannot express for plugin 03 is done
//! here from the public pieces of the engine: the Warping popup (a warp code, not plugin 01's Smart
//! Blend checkbox), the missing `Src Time Offset`, the FG matte probes of the pre-render and the
//! checks that turn every feature of the multi layer path into an explicit `FxError::Unsupported`
//! instead of a silent fallback.
//!
//! Two things leave the single layer path and are refused with `FxError::Unsupported`:
//! * a FG Matte layer with pixels (the count `n` of the probe loop, which stops at the first empty
//!   one) selects the multi layer path (one flow per layer, mattes, Main_BG compositing);
//! * user geometry of the one group: a `Pt k: Use` point that is in use at the two source frame
//!   times (every value 2 to 5 belongs to the only group when `n = 0`) reaches the flow solver as
//!   user displacement. Mask paths (First/Last Mask) feed the same solver input but are parameter
//!   kinds `fx-core` cannot express, so they cannot be detected and are not part of this crate's
//!   input.
//!
//! The Source display (`Display == 2`) shows Alt Source A at the current time `X`, or the input
//! layer when there is no Alt Source A; the error flags are tested before it. With `n = 0` the
//! Display popup values 3 to 6 (FG1 to FG3 and Background processed) all select the single group
//! `min(Display - 3, 0) = 0`, which is the range `0..=0` of Display 1 as well, so they render
//! exactly like the processed output.

use crate::prerender::{self, PreRender03, LAYER_FG_MATTE};
use crate::timemap::{CurveParams03, RetimeFgBgTimeMap};
use fx_core::{Checkout, Env, FieldOrder, Frame, FxError, ParamCurves, Result, TemporalContext};
use retime_engine::compose::{self, Rendered};
use retime_engine::frontend::{self, SettingIds, TimeParamIds, Wiring};
use retime_engine::host::{layer_is_set, EngineHost, EngineSettings, EnvHost, LayerIds};
use retime_engine::image::{error_frame, MissingLayerStyle};
use retime_engine::planner::{LAYER_ALT_A, LAYER_ALT_B};
use retime_engine::timeline::{frame_positions, TimeMap};

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

/// The message of a FG Matte layer being set.
pub const FG_UNSUPPORTED: &str = "a FG Matte layer is set: the multi layer path (one flow per \
     layer, mattes, Main_BG compositing) is not implemented; remove the FG Matte layers (this render \
     handles the single layer case only)";

/// The message of user tracking points reaching the solver.
pub fn points_unsupported(points: &[usize]) -> String {
    let list: Vec<String> = points.iter().map(|n| format!("Pt {n}: Use")).collect();
    format!(
        "user tracking points feed the flow solver as user displacement, which is not \
         implemented; set {} to Don't Use",
        list.join(", ")
    )
}

/// The message of the Draw Geom overlay.
pub const DRAW_GEOM_UNSUPPORTED: &str = "Draw Geom would draw the geometry of the tracking points over the source view, \
     which is not implemented; set Draw Geom to None or Pt n: Use to Don't Use";

/// The message of Track RGB+A.
pub const TRACK_RGB_A_UNSUPPORTED: &str = "Track RGB+A drives the \
     engine's 360 degree flag (wrap around warps, solver and hole fill), which is not implemented; \
     uncheck Track RGB+A (it only matters when motion vectors are computed)";

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
/// time and the same at the second. With no FG layer every value 2 to 5 maps to the only group.
pub fn points_in_use(curves: &ParamCurves, time1: i32, time2: i32) -> Vec<usize> {
    let mut out = Vec::new();
    for (i, id) in crate::PT_USE_IDS.iter().enumerate() {
        let v1 = curves.int_at(id, i64::from(time1));
        if v1 != 1 && curves.int_at(id, i64::from(time2)) == v1 {
            out.push(i + 1);
        }
    }
    out
}

/// The settings of the engine for plugin 03: the shared reader plus the warp code of the Warping
/// popup read at `current_time` (all popups are sampled there).
pub fn read_settings(cx: &TemporalContext, env: &Env<'_>) -> Result<EngineSettings> {
    let mut settings = frontend::read_settings(&WIRING, cx, env)?;
    settings.warp_mode = warp_code(env.params.int_at("warping", cx.current_time) as i32);
    Ok(settings)
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
    prerender::checkouts(&req, &map, FG_UNSUPPORTED)
}

/// `Effect::render_temporal`: settings, time map, host, the checks of this module, the core, the finish.
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
    let rendered = render(&settings, env, &host, &map)?;
    frontend::finish(rendered, output);
    Ok(())
}

/// The render of one frame: error flags, then the Source display, then the core, with the
/// `Unsupported` checks of the single layer front-end where a setting leaves that path.
fn render(settings: &EngineSettings, env: &Env<'_>, host: &EnvHost<'_>, map: &RetimeFgBgTimeMap<CurveParams03<'_>>) -> Result<Rendered> {
    // Error flags first: the size probes of the FG mattes set the same size mismatch flag as the
    // other layers; red wins over the missing layer fill.
    let (fg_count, fg_mismatch) = fg_probe(env, settings.width, settings.height);
    let mut issues = host.issues();
    issues.size_mismatch |= fg_mismatch;
    if let Some(kind) = issues.error_kind(settings.missing_layer_style) {
        return Ok(Rendered::Error(error_frame(kind, settings.width, settings.height, settings.fmt.depth())));
    }
    let curves = env.params;
    let (t0, _) = frame_positions(settings.current_time, settings.time_step, settings.local_time_step);
    if settings.display_source {
        // 6.4: the Draw Geom overlay runs after the copy of the source; with no geometry it draws nothing.
        let draw_geom = curves.int_at("draw_geom", i64::from(settings.current_time));
        if draw_geom != 1 {
            let at = map.frame_time(t0);
            if !points_in_use(curves, at, at).is_empty() {
                return Err(FxError::Unsupported(DRAW_GEOM_UNSUPPORTED.into()));
            }
        }
        return compose::render(settings, host);
    }
    if fg_count > 0 {
        return Err(FxError::Unsupported(FG_UNSUPPORTED.into()));
    }
    let sel = compose::select_frames(settings, host)?;
    if sel.vectors_quality != 0 && !sel.exact {
        // The vector path: everything that reaches the solver or the warps must be one the engine has.
        let points = points_in_use(curves, map.frame_time(sel.a as f32), map.frame_time(sel.b as f32));
        if !points.is_empty() {
            return Err(FxError::Unsupported(points_unsupported(&points)));
        }
        if curves.bool_at("track_rgb_a", i64::from(settings.current_time)) {
            return Err(FxError::Unsupported(TRACK_RGB_A_UNSUPPORTED.into()));
        }
    }
    compose::retime(settings, host)
}
