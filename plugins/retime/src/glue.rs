//! Plugin 01's wiring of the shared engine front-end: which parameter feeds which setting, which
//! layers it takes and which time map it uses (the Speed / Frame Number map).

use fx_core::{Checkout, Env, Frame, Result, TemporalContext};
use retime_engine::frontend::{self, CurveTimeParams, SettingIds, TimeParamIds, Wiring};
use retime_engine::host::{EngineSettings, LayerIds};
use retime_engine::image::MissingLayerStyle;
use retime_engine::planner::{LAYER_ALT_A, LAYER_ALT_B};
use retime_engine::timeline::RemapTimeMap;

/// Plugin 01's parameter and layer ids (re-exported by the crate root so tests and the harness can
/// read the settings the same way the effect does). `register_layers` adds three extra checkouts at
/// `-current_time` to the dependency list: the input layer, the source layer and Alt Motion Source.
pub const WIRING: Wiring = Wiring {
    settings: SettingIds {
        display: Some("display"),
        video_360: Some("video_360"),
        input_fields: Some("input_fields"),
        in_fps_is_out_fps: Some("in_fps_is_out_fps"),
        input_frame_rate: Some("input_frame_rate"),
        src_time_offset: Some("src_time_offset"),
        vectors: Some("vectors"),
        prep: Some("prep"),
        sensitivity: Some("sensitivity"),
        plenty_memory: Some("plenty_memory"),
        frame_interp: Some("frame_interp"),
        smart_blend: Some("smart_blend"),
        blur_compensation: Some("blur_compensation"),
    },
    time: TimeParamIds {
        remap_mode: "time_remap_mode",
        speed: "speed_percent",
        frame_num: "frame_num",
        src_time_offset: "src_time_offset",
        mark_segs: "src_mark_segments",
    },
    layers: LayerIds { alt_a: LAYER_ALT_A, alt_b: LAYER_ALT_B },
    missing_layer: MissingLayerStyle::HalfBlue,
    register_layers: true,
    // MEASURED: the original's motion path works on bottom-up rows
    motion_rows_reversed: true,
};

/// The time map of plugin 01: `base` starts at 0, the settings give the rate model and the current
/// time.
fn time_map<'e>(settings: &EngineSettings, params: CurveTimeParams<'e>) -> RemapTimeMap<CurveTimeParams<'e>> {
    RemapTimeMap::new(params, settings.rates, settings.current_time)
}

/// `Effect::dependencies`: the checkout list.
pub(crate) fn dependencies(cx: &TemporalContext, env: &Env<'_>) -> Result<Vec<Checkout>> {
    frontend::dependencies(&WIRING, cx, env, time_map)
}

/// `Effect::render_temporal`: settings, time map, host, compose, finish, then the fetch of the input layer at
/// minus the current time.
///
/// The original checks out the pixels of the input layer at `-current_time` in the render of every output frame
/// (MEASURED on both module grids, which have no alternate layers and no `Display == Source` case); nothing of
/// those pixels reaches the output. The fetch is made
/// when the checkout list has that checkout (the pre-render's checkouts at `-current_time`), so a host is asked for
/// the same frames.
pub(crate) fn render_temporal(cx: &TemporalContext, env: &Env<'_>, output: &mut Frame) -> Result<()> {
    frontend::render_temporal(&WIRING, cx, env, output, time_map)?;
    let neg = Checkout { layer: None, time: cx.current_time.wrapping_neg() };
    if dependencies(cx, env)?.contains(&neg) {
        env.input.frame_at(neg.time)?;
    }
    Ok(())
}
