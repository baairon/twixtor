//! Plugin 04's wiring of the shared engine, single layer use with vectors from layers.
//!
//! The shared `retime_engine::frontend` entry points cannot express plugin 04 (its rate model
//! differs, it has no Src Time Offset, no Vectors popup, a Warping popup and two vector layers), so
//! the settings reader, the probes, the checks and the render are written here from the public
//! pieces of the engine, the way plugin 03's glue does.
//!
//! Order of a render:
//!
//! 1. the probes at `current_time` ([`probe`]): `Source` (size 0 is a missing layer, another size
//!    is a size mismatch), `Vectors: Prev` and `Vectors: Next` (a size other than the output is a
//!    size mismatch, and so is having neither present), the FG matte layers (a size other than the
//!    output is a size mismatch). The FIRST problem decides;
//! 2. the rate check (`out < 0.01 || in <= 0.01`: a size mismatch);
//! 3. `Display == Source`: the source frame at `current_time`;
//! 4. otherwise the core, with the Vectors In flows ([`compose::retime_with_flows`]), or with FG
//!    matte layers the layered core ([`compose::retime_layers_with_flows`]): the vector layers
//!    are loaded once per layer of the Display group with the layer's matte step (rule G1e), each
//!    layer is warped with its own flows and the group is composed bottom up.
//!
//! A failed probe or rate check writes the fill its error stands for (red at half alpha for a size
//! mismatch, the plugin's blue for a missing source) and the render still succeeds, so a failure is
//! visible.
//!
//! A FG Matte layer with pixels (the count `n` of the probe loop, which stops at the first empty
//! one) selects the n-layer path ([`layer_job`]) for every Display except the Source one; with
//! `n = 0` the render is the single layer core of today, byte for byte, and the Display values 3
//! to 6 select the one group, so they render as the output.
//!
//! The Warping popup "Forward" (warp code 2, the default) is served by the engine's forward warp
//! and a single vector layer by its flow inversion (`compose::invert_flow`), so neither is refused.
//!
//! The 360 flag and Track RGB+A are fixed at 0, the Sensitivity is the constant 0.7, and there is
//! no Prep, no field buffer and no tracking layer, so none of those can be reached from the
//! parameters.

use crate::prerender::{self, PreRender04, LAYER_FG_MATTE, LAYER_SOURCE, LAYER_VECTORS_NEXT, LAYER_VECTORS_PREV, VECTORS_QUALITY};
use crate::timemap::{CurveParams04, VectorsInTimeMap};
use crate::vectorin;
use fx_core::{float_to_fixed, Checkout, Env, FieldOrder, Frame, FxError, ParamCurves, ParamValue, Quality, Result, TemporalContext};
use retime_engine::compose::{self, Rendered};
use retime_engine::frontend::{self, fixed_reader, ticks_i32};
use retime_engine::host::{layer_is_set, EngineSettings, EnvHost, LayerIds};
use retime_engine::image::{error_frame, ErrorKind, Fmt, MissingLayerStyle};
use retime_engine::layers::{LayerJob, LayerParams};
use retime_engine::prep;
use retime_engine::timeline::Rates;

/// The ids the host adapter reads: the retimed layer is the `source` layer when set (else the input); the second
/// id is the tracking layer slot, which plugin 04 never has (`no_tracking_layer` is always true), so it is never
/// looked up.
pub const IDS: LayerIds = LayerIds { alt_a: LAYER_SOURCE, alt_b: LAYER_VECTORS_NEXT };

/// The constant Main_BG sensitivity plugin 04 passes to the core.
pub const SENSITIVITY: f32 = 0.7f32;

/// Whether the dummy registrations of every plan entry are listed (see [`prerender::checkouts`]).
pub const AE_13_2: bool = true;

/// The Warping popup as the warp code of the engine: 1 Inverse is 0, 2 Forward is 2, any other
/// value keeps 0. The warp codes 1 (Smart Blend) and 3 (forward warp without hole fill) never
/// occur.
pub fn warp_code(popup_value: i32) -> i32 {
    match popup_value {
        1 => 0,
        2 => 2,
        _ => 0,
    }
}

/// The ids of the matte channel popup and the inverse matte shrink slider of FG1 to FG3.
const FG_CHAN_IDS: [&str; 3] = ["fg1_matte_chan", "fg2_matte_chan", "fg3_matte_chan"];
const FG_SHRINK_IDS: [&str; 3] = ["fg1_inv_matte_shrink", "fg2_inv_matte_shrink", "fg3_inv_matte_shrink"];

/// The layer job of a render with `fg_count` FG matte layers: FG k's `fgk_matte_chan` and
/// `fgk_inv_matte_shrink` (the 16.16 reader, in pixels at the downsample ratio 1; I: as the other
/// sliders are read), read at `current_time`; plugin 04 solves nothing, so every layer has the
/// constant [`SENSITIVITY`] and tracks; the group is the Display popup's
/// (`LayerJob::display_group`).
pub fn layer_job(settings: &EngineSettings, curves: &ParamCurves, fg_count: usize) -> Result<LayerJob> {
    let t = i64::from(settings.current_time);
    let mut fg = Vec::with_capacity(fg_count);
    for k in 0..fg_count {
        fg.push(LayerParams {
            matte_chan: popup(curves, FG_CHAN_IDS[k], t)?,
            sensitivity: SENSITIVITY,
            shrink: fixed_reader(fixed(curves, FG_SHRINK_IDS[k], t)?, 1.0f32),
            track: true,
        });
    }
    let main_bg = LayerParams { matte_chan: 0, sensitivity: SENSITIVITY, shrink: 0.0, track: true };
    Ok(LayerJob::for_display(&fg, main_bg, popup(curves, "display", t)?))
}

/// The settings of one render: the engine's and the one flag the rate model raises.
#[derive(Clone, Debug)]
pub struct Settings {
    pub engine: EngineSettings,
    /// The rate check failed (`out < 0.01 || in <= 0.01`); `engine.rates` then holds a placeholder
    /// 30 fps model and nothing may be rendered.
    pub rate_error: bool,
}

fn wrong_kind(id: &str, wanted: &str, got: &ParamValue) -> FxError {
    FxError::BadParam { id: id.to_string(), reason: format!("expected {wanted}, found {got:?}") }
}

fn popup(curves: &ParamCurves, id: &str, time: i64) -> Result<i32> {
    match curves.at(id, time)? {
        ParamValue::Int(v) => Ok(v as i32),
        other => Err(wrong_kind(id, "a popup", &other)),
    }
}

fn checkbox(curves: &ParamCurves, id: &str, time: i64) -> Result<bool> {
    match curves.at(id, time)? {
        ParamValue::Bool(v) => Ok(v),
        other => Err(wrong_kind(id, "a checkbox", &other)),
    }
}

fn slider(curves: &ParamCurves, id: &str, time: i64) -> Result<f64> {
    match curves.at(id, time)? {
        ParamValue::Float(v) => Ok(v),
        ParamValue::Int(v) => Ok(v as f64),
        other => Err(wrong_kind(id, "a slider", &other)),
    }
}

/// A 16.16 fixed slider as the integer it is stored as (the sample converted by `float_to_fixed`).
fn fixed(curves: &ParamCurves, id: &str, time: i64) -> Result<i32> {
    Ok(float_to_fixed(slider(curves, id, time)?))
}

/// Reads the settings of one output frame at `cx.current_time`. Differences from
/// `frontend::read_settings` (plugin 01):
///
/// * rates: the input rate is 30.0 when `In FPS is Out FPS` is checked (the default) and `Input:
///   Frame Rate` otherwise; the output rate is always the composition rate, so the `Output: Frame
///   Rate` slider is read and ignored. Both snap to 29.97 / 59.94 / 23.976 and the rate check
///   raises `rate_error`. There is no field doubling;
/// * the vectors quality is the constant 4, the sensitivity the constant 0.7, Prep and Smart Blend
///   are off, there is no 360 flag, no Plenty Memory (`cache_last` false), no Src Time Offset, no
///   tracking layer;
/// * `warp_mode` is the Warping popup ([`warp_code`]).
pub fn read_settings(cx: &TemporalContext, env: &Env<'_>) -> Result<Settings> {
    let curves = env.params;
    let t = cx.current_time;
    let current_time = ticks_i32(cx.current_time, "current_time")?;
    let time_step = ticks_i32(cx.time_step(), "time_step")?;
    let local_time_step = ticks_i32(cx.local_time_step, "local_time_step")?;
    let time_scale = ticks_i32(cx.time_scale(), "time_scale")?;

    let in_is_out = checkbox(curves, "in_fps_is_out_fps", t)?;
    let typed_input = slider(curves, "input_frame_rate", t)?;
    // read and then ignored: the output rate is the composition rate
    let _typed_output = slider(curves, "output_frame_rate", t)?;
    let input_rate = if in_is_out { 30.0 } else { typed_input };
    let (rates, rate_error) = match Rates::new(input_rate, cx.fps.as_f64(), false, FieldOrder::Progressive, local_time_step) {
        Ok(rates) => (rates, false),
        Err(FxError::BadParam { .. }) => {
            (Rates::new(30.0, 30.0, false, FieldOrder::Progressive, local_time_step).expect("30 fps is a valid rate"), true)
        }
        Err(e) => return Err(e),
    };
    let engine = EngineSettings {
        current_time,
        time_step,
        local_time_step,
        time_scale,
        quality_high: cx.quality == Quality::High,
        width: cx.width,
        height: cx.height,
        fmt: Fmt::from_depth(cx.depth),
        rates,
        input_fields: FieldOrder::Progressive,
        display_source: popup(curves, "display", t)? == 2,
        src_time_offset_fixed: 0,
        // tiny frames (G7, U): a frame with a side of 1 has no motion search and renders as No Vectors
        vectors_quality: prep::searched_quality(VECTORS_QUALITY, cx.width, cx.height),
        interp: popup(curves, "frame_interp", t)? - 1,
        blur: fixed_reader(fixed(curves, "blur_compensation", t)?, 1.0f32),
        sensitivity: SENSITIVITY,
        prep_contrast_edge: false,
        prep_delinearize: false,
        smart_blend: false,
        warp_mode: warp_code(popup(curves, "warping", t)?),
        video_360: false,
        cache_last: false,
        alt_a_present: layer_is_set(env.layer(LAYER_SOURCE)),
        no_tracking_layer: true,
        missing_layer_style: MissingLayerStyle::BlueAlphaSlip,
        wrap_360: false,
        device: cx.device,
        stages: retime_engine::stages::StageSet::from_context(cx),
    };
    Ok(Settings { engine, rate_error })
}

/// The consecutive FG matte layers present, starting at FG1: the probe loop stops at the first
/// empty one.
pub fn fg_count(env: &Env<'_>) -> usize {
    LAYER_FG_MATTE.iter().take_while(|id| layer_is_set(env.layer(id))).count()
}

/// What the probes found at `current_time`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Probe {
    /// `Vectors: Prev` is present.
    pub has_prev: bool,
    /// `Vectors: Next` is present.
    pub has_next: bool,
    /// Number of FG matte layers (`nfg`).
    pub fg_count: usize,
    /// The first probe problem, as the fill its error stands for: `MissingLayer` for `Source` of
    /// size 0, `SizeMismatch` for every other.
    pub failure: Option<ErrorKind>,
}

/// The probes in order (the first problem wins). The `Source` probe looks at the `source` layer
/// when it is set and at the input layer otherwise.
pub fn probe(env: &Env<'_>, cx: &TemporalContext) -> Result<Probe> {
    let size = (cx.width, cx.height);
    let mut failure = None;
    let src: &dyn fx_core::Layer = match env.layer(LAYER_SOURCE) {
        Some(l) if layer_is_set(Some(l)) => l,
        _ => env.input,
    };
    if src.width() == 0 || src.height() == 0 {
        failure = Some(ErrorKind::MissingLayer(MissingLayerStyle::BlueAlphaSlip));
    } else if (src.width(), src.height()) != size {
        failure = Some(ErrorKind::SizeMismatch);
    }
    let prev = vectorin::present_at(env.layer(LAYER_VECTORS_PREV), cx.current_time)?;
    let next = vectorin::present_at(env.layer(LAYER_VECTORS_NEXT), cx.current_time)?;
    for found in [prev, next].into_iter().flatten() {
        if found != size && failure.is_none() {
            failure = Some(ErrorKind::SizeMismatch);
        }
    }
    if prev.is_none() && next.is_none() && failure.is_none() {
        failure = Some(ErrorKind::SizeMismatch);
    }
    let mut fg = 0;
    for id in LAYER_FG_MATTE {
        let layer = env.layer(id);
        if !layer_is_set(layer) {
            break;
        }
        fg += 1;
        if let Some(l) = layer {
            if (l.width(), l.height()) != size && failure.is_none() {
                failure = Some(ErrorKind::SizeMismatch);
            }
        }
    }
    Ok(Probe { has_prev: prev.is_some(), has_next: next.is_some(), fg_count: fg, failure })
}

fn time_map<'e>(settings: &EngineSettings, env: &Env<'e>) -> Result<VectorsInTimeMap<CurveParams04<'e>>> {
    Ok(VectorsInTimeMap::new(CurveParams04::new(env.params)?, settings.rates, settings.current_time))
}

/// `Effect::dependencies`: the pre-render list (`prerender::checkouts`). With a rate error the
/// planner cannot run (its times would be infinite), so only the probes are listed.
pub(crate) fn dependencies(cx: &TemporalContext, env: &Env<'_>) -> Result<Vec<Checkout>> {
    let Settings { engine: settings, rate_error } = read_settings(cx, env)?;
    let req = PreRender04 {
        current_time: settings.current_time,
        time_step: settings.time_step,
        local_time_step: settings.local_time_step,
        display_source: settings.display_source,
        source_present: settings.alt_a_present,
        fg_count: fg_count(env),
        interp: settings.interp,
        blur: settings.blur,
        register_layers: AE_13_2,
    };
    if rate_error {
        return Ok(prerender::probes(&req));
    }
    let map = time_map(&settings, env)?;
    prerender::checkouts(&req, &map)
}

/// `Effect::render_temporal`: probes, settings, time map, host, the checks of this module, the core, the finish.
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
    let probe = probe(env, cx)?;
    let Settings { engine: settings, rate_error } = read_settings(cx, env)?;
    let failure = probe.failure.or(rate_error.then_some(ErrorKind::SizeMismatch));
    if let Some(kind) = failure {
        *output = error_frame(kind, cx.width, cx.height, cx.depth);
        return Ok(());
    }
    let map = time_map(&settings, env)?;
    let host = EnvHost::new(env, &settings, &map, IDS)?;
    let rendered = render(&settings, env, host, &map, &probe)?;
    frontend::finish(rendered, output);
    Ok(())
}

/// The render of one frame once the probes passed: the Source display, then the core with the
/// Vectors In flows, single layer or layered.
fn render<'e>(
    settings: &EngineSettings,
    env: &'e Env<'_>,
    host: EnvHost<'e>,
    map: &VectorsInTimeMap<CurveParams04<'_>>,
    probe: &Probe,
) -> Result<Rendered> {
    if settings.display_source {
        // the source layer at current_time through the prepare; the overlay draws nothing
        return compose::render(settings, &host);
    }
    if probe.fg_count == 0 {
        return compose::retime_with_flows(settings, &host, |sel| {
            vectorin::load_flows(env, settings, map, sel, probe.has_prev, probe.has_next)
        });
    }
    let host = host.with_mattes(env, &LAYER_FG_MATTE[..probe.fg_count])?;
    let job = layer_job(settings, env.params, probe.fg_count)?;
    compose::retime_layers_with_flows(settings, &host, &job, |sel, _k, mattes| {
        vectorin::load_layer_flows(env, settings, map, sel, probe.has_prev, probe.has_next, mattes)
    })
}
