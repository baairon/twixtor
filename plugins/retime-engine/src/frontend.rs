//! Glue the four front-ends share: how the parameters, the time context and the layers of a
//! plugin become [`EngineSettings`], a [`TimeParams`] reader, an [`EngineHost`] and the
//! pre-render list, and how the engine's result becomes the output frame.
//!
//! Everything plugin specific (parameter ids, layer ids, which fill a missing layer shows, the
//! time map) is passed in through [`Wiring`] and a time map factory. Popups are read as integers,
//! 16.16 fixed sliders as the integer `float_to_fixed` gives for the sampled value and converted
//! with `((v as f64) * (1.0 / 65536.0)) as f32 * scale`, checkboxes as bools.

use crate::compose::{self, Rendered};
use crate::host::{layer_is_set, EngineHost, EngineSettings, EnvHost, LayerIds};
use crate::image::{Fmt, MissingLayerStyle};
use crate::planner::{checkouts, PreRender};
use crate::timeline::{Rates, TimeMap, TimeParams};
use fx_core::{float_to_fixed, Checkout, Env, FieldOrder, Frame, FxError, ParamCurves, ParamValue, Quality, Result, TemporalContext};

/// The ids of the parameters the time map reads (`TimeParams`).
#[derive(Clone, Copy, Debug)]
pub struct TimeParamIds {
    /// `Time Remap Mode` popup: 1 Speed, 2 Frame Number.
    pub remap_mode: &'static str,
    /// `Speed %`, a 16.16 fixed slider.
    pub speed: &'static str,
    /// `Frame Num`, a float slider.
    pub frame_num: &'static str,
    /// `Src Time Offset`, a 16.16 fixed slider.
    pub src_time_offset: &'static str,
    /// `Src: Mark Segments` popup 1..=5.
    pub mark_segs: &'static str,
}

/// [`TimeParams`] over `ParamCurves`: every method samples its parameter at the given layer time
/// (`ParamCurves` takes i64 ticks, the trait i32 ticks). The two 16.16 sliders go through
/// `ParamCurves::fixed_at`, so a keyframed or typed value is quantised to the 16.16 grid.
pub struct CurveTimeParams<'a> {
    curves: &'a ParamCurves,
    ids: TimeParamIds,
}

impl<'a> CurveTimeParams<'a> {
    /// An error when an id is not a parameter of the curves or has the wrong kind, so the
    /// reads below cannot panic.
    pub fn new(curves: &'a ParamCurves, ids: TimeParamIds) -> Result<CurveTimeParams<'a>> {
        for (id, popup) in [(ids.remap_mode, true), (ids.mark_segs, true), (ids.speed, false), (ids.frame_num, false), (ids.src_time_offset, false)] {
            let ok = match curves.at(id, 0)? {
                ParamValue::Int(_) => true,
                ParamValue::Float(_) => !popup,
                _ => false,
            };
            if !ok {
                return Err(FxError::BadParam {
                    id: id.to_string(),
                    reason: if popup { "the time map needs a popup".into() } else { "the time map needs a slider".into() },
                });
            }
        }
        Ok(CurveTimeParams { curves, ids })
    }
}

impl TimeParams for CurveTimeParams<'_> {
    fn remap_mode(&self, time: i32) -> i32 {
        self.curves.int_at(self.ids.remap_mode, i64::from(time)) as i32
    }

    fn speed_fixed(&self, time: i32) -> i32 {
        self.curves.fixed_at(self.ids.speed, i64::from(time))
    }

    fn frame_num(&self, time: i32) -> f64 {
        self.curves.float_at(self.ids.frame_num, i64::from(time))
    }

    fn src_time_offset_fixed(&self, time: i32) -> i32 {
        self.curves.fixed_at(self.ids.src_time_offset, i64::from(time))
    }

    fn mark_segs(&self, time: i32) -> i32 {
        self.curves.int_at(self.ids.mark_segs, i64::from(time)) as i32
    }
}

/// The ids of the parameters that become [`EngineSettings`]. `None` means the plugin has no such
/// parameter and the setting keeps the default noted on its field.
#[derive(Clone, Copy, Debug, Default)]
pub struct SettingIds {
    /// `Display` popup, 2 = Source. Default: Output.
    pub display: Option<&'static str>,
    /// `360 Video` checkbox. Default: off.
    pub video_360: Option<&'static str>,
    /// `Input: Fields` popup 1..=3. Default: None.
    pub input_fields: Option<&'static str>,
    /// `In FPS is Out FPS` checkbox. Default: off.
    pub in_fps_is_out_fps: Option<&'static str>,
    /// `Input: Frame Rate` float slider. Default: 15.0, the initial value of the source rate.
    pub input_frame_rate: Option<&'static str>,
    /// `Src Time Offset` 16.16 fixed slider, read for `Display == Source`. Default: 0.
    pub src_time_offset: Option<&'static str>,
    /// `Vectors` popup 1..=5. Default: No Vectors.
    pub vectors: Option<&'static str>,
    /// `Prep` popup 1..=3. Default: None.
    pub prep: Option<&'static str>,
    /// `Sensitivity` 16.16 fixed slider. Default: 0.
    pub sensitivity: Option<&'static str>,
    /// `Plenty Memory?` checkbox. Default: off.
    pub plenty_memory: Option<&'static str>,
    /// `Frame Interp` popup 1..=3. Default: Nearest.
    pub frame_interp: Option<&'static str>,
    /// `Smart Blend` checkbox. Default: off.
    pub smart_blend: Option<&'static str>,
    /// `Blur Compensation` 16.16 fixed slider. Default: 0.
    pub blur_compensation: Option<&'static str>,
}

/// How one plugin maps its parameters and layers onto the engine.
#[derive(Clone, Copy, Debug)]
pub struct Wiring {
    pub settings: SettingIds,
    pub time: TimeParamIds,
    pub layers: LayerIds,
    /// The fill of the missing layer error (`ErrorKind::MissingLayer`).
    pub missing_layer: MissingLayerStyle,
    /// The pre-render lists three extra checkouts at `-current_time`.
    pub register_layers: bool,
}

/// The 16.16 reader: `((v as f64) * (1.0 / 65536.0)) as f32 * scale`.
pub fn fixed_reader(fixed: i32, scale: f32) -> f32 {
    ((fixed as f64) * (1.0 / 65536.0)) as f32 * scale
}

/// Ticks are 32 bit in the engine; `fx_core` carries i64 ticks. Anything that does not fit is an
/// error rather than a silent wrap.
pub fn ticks_i32(ticks: i64, what: &str) -> Result<i32> {
    i32::try_from(ticks).map_err(|_| FxError::Unsupported(format!("{what} {ticks} does not fit 32 bit ticks")))
}

fn wrong_kind(id: &str, wanted: &str, got: &ParamValue) -> FxError {
    FxError::BadParam { id: id.to_string(), reason: format!("expected {wanted}, found {got:?}") }
}

/// Popup (integer) parameter sampled at `time`; `default` when the plugin has none.
fn popup(curves: &ParamCurves, id: Option<&str>, time: i64, default: i32) -> Result<i32> {
    let Some(id) = id else { return Ok(default) };
    match curves.at(id, time)? {
        ParamValue::Int(v) => Ok(v as i32),
        other => Err(wrong_kind(id, "a popup", &other)),
    }
}

fn checkbox(curves: &ParamCurves, id: Option<&str>, time: i64) -> Result<bool> {
    let Some(id) = id else { return Ok(false) };
    match curves.at(id, time)? {
        ParamValue::Bool(v) => Ok(v),
        other => Err(wrong_kind(id, "a checkbox", &other)),
    }
}

fn slider(curves: &ParamCurves, id: Option<&str>, time: i64, default: f64) -> Result<f64> {
    let Some(id) = id else { return Ok(default) };
    match curves.at(id, time)? {
        ParamValue::Float(v) => Ok(v),
        ParamValue::Int(v) => Ok(v as f64),
        other => Err(wrong_kind(id, "a slider", &other)),
    }
}

/// A 16.16 fixed slider as an integer (`float_to_fixed` of the sample).
fn fixed(curves: &ParamCurves, id: Option<&str>, time: i64) -> Result<i32> {
    Ok(float_to_fixed(slider(curves, id, time, 0.0)?))
}

/// Reads the settings of one output frame at `cx.current_time`, including the layer probes of the
/// pre-render (`alt_a_present`, `no_tracking_layer`). Errors: a time that does not fit 32 bits, a
/// parameter of the wrong kind, and error 5 of the rate model (`Rates::new`).
///
/// Order of the derivations: the rates (`Input: Frame Rate` read as `(f64 as f32)`, the
/// composition rate from `cx.fps`, `In FPS is Out FPS`, snapping, field doubling), then the
/// popups minus one, the fixed readers for `Blur Compensation` (scale 1.0f) and `Sensitivity`
/// (scale 1.0f, then `/ 100.0f`), `Prep` == 2 and == 3, the checkboxes.
pub fn read_settings(wiring: &Wiring, cx: &TemporalContext, env: &Env<'_>) -> Result<EngineSettings> {
    let curves = env.params;
    let ids = &wiring.settings;
    let t = cx.current_time;
    let current_time = ticks_i32(cx.current_time, "current_time")?;
    let time_step = ticks_i32(cx.time_step(), "time_step")?;
    let local_time_step = ticks_i32(cx.local_time_step, "local_time_step")?;
    let time_scale = ticks_i32(cx.time_scale(), "time_scale")?;

    let input_fields = match popup(curves, ids.input_fields, t, 1)? {
        2 => FieldOrder::UpperFirst,
        3 => FieldOrder::LowerFirst,
        _ => FieldOrder::Progressive,
    };
    let rates = Rates::new(
        slider(curves, ids.input_frame_rate, t, 15.0)?,
        cx.fps.as_f64(),
        checkbox(curves, ids.in_fps_is_out_fps, t)?,
        input_fields,
        local_time_step,
    )?;
    let smart_blend = checkbox(curves, ids.smart_blend, t)?;
    let prep = popup(curves, ids.prep, t, 1)?;
    Ok(EngineSettings {
        current_time,
        time_step,
        local_time_step,
        time_scale,
        quality_high: cx.quality == Quality::High,
        width: cx.width,
        height: cx.height,
        fmt: Fmt::from_depth(cx.depth),
        rates,
        input_fields,
        display_source: popup(curves, ids.display, t, 1)? == 2,
        src_time_offset_fixed: fixed(curves, ids.src_time_offset, t)?,
        vectors_quality: popup(curves, ids.vectors, t, 1)? - 1,
        interp: popup(curves, ids.frame_interp, t, 1)? - 1,
        blur: fixed_reader(fixed(curves, ids.blur_compensation, t)?, 1.0f32),
        sensitivity: fixed_reader(fixed(curves, ids.sensitivity, t)?, 1.0f32) / 100.0f32,
        prep_contrast_edge: prep == 2,
        prep_delinearize: prep == 3,
        smart_blend,
        warp_mode: i32::from(smart_blend),
        video_360: checkbox(curves, ids.video_360, t)?,
        cache_last: checkbox(curves, ids.plenty_memory, t)?,
        alt_a_present: layer_is_set(env.layer(wiring.layers.alt_a)),
        no_tracking_layer: !layer_is_set(env.layer(wiring.layers.alt_b)),
        missing_layer_style: wiring.missing_layer,
    })
}

/// The inputs of the pre-render planner from the settings (`planner::PreRender`).
pub fn pre_render(settings: &EngineSettings, register_layers: bool) -> PreRender {
    PreRender {
        current_time: settings.current_time,
        time_step: settings.time_step,
        local_time_step: settings.local_time_step,
        display_source: settings.display_source,
        src_time_offset_fixed: settings.src_time_offset_fixed,
        alt_a_present: settings.alt_a_present,
        alt_b_present: !settings.no_tracking_layer,
        input_fields: settings.input_fields != FieldOrder::Progressive,
        vectors_quality: settings.vectors_quality,
        interp: settings.interp,
        blur: settings.blur,
        register_layers,
    }
}

/// The complete checkout list of the pre-render for the settings (`planner::checkouts`).
pub fn checkout_list(settings: &EngineSettings, map: &dyn TimeMap, register_layers: bool) -> Result<Vec<Checkout>> {
    checkouts(&pre_render(settings, register_layers), map)
}

/// `Effect::dependencies` of a plugin that uses the Speed / Frame Number time map: reads the
/// settings, builds the plugin's time map with `make_map` (it gets the settings and the
/// parameter reader) and lists the checkouts.
pub fn dependencies<'e, M, F>(wiring: &Wiring, cx: &TemporalContext, env: &Env<'e>, make_map: F) -> Result<Vec<Checkout>>
where
    M: TimeMap,
    F: FnOnce(&EngineSettings, CurveTimeParams<'e>) -> M,
{
    let settings = read_settings(wiring, cx, env)?;
    let map = make_map(&settings, CurveTimeParams::new(env.params, wiring.time)?);
    checkout_list(&settings, &map, wiring.register_layers)
}

/// The output finish: an engine image is unpremultiplied into the wire frame (`Image::to_frame`),
/// an error fill replaces the frame as it is.
pub fn finish(rendered: Rendered, output: &mut Frame) {
    *output = rendered.into_frame();
}

/// `Effect::render_temporal` of a plugin that uses the Speed / Frame Number time map: settings,
/// time map, [`EnvHost`], [`compose::render`], [`finish`]. `output` must have the size and depth
/// of `cx`.
pub fn render_temporal<'e, M, F>(wiring: &Wiring, cx: &TemporalContext, env: &Env<'e>, output: &mut Frame, make_map: F) -> Result<()>
where
    M: TimeMap,
    F: FnOnce(&EngineSettings, CurveTimeParams<'e>) -> M,
{
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
    let settings = read_settings(wiring, cx, env)?;
    let map = make_map(&settings, CurveTimeParams::new(env.params, wiring.time)?);
    let host = EnvHost::new(env, &settings, &map, wiring.layers)?;
    let rendered = compose::render(&settings, &host as &dyn EngineHost)?;
    finish(rendered, output);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fx_core::{Keyframe, ParamKind, ParamSpec, Rational, TimeBase};

    fn slider_spec(id: &'static str, default: f64) -> ParamSpec {
        ParamSpec {
            id,
            name: id,
            kind: ParamKind::Float { valid_min: -1000.0, valid_max: 1000.0, slider_min: -1000.0, slider_max: 1000.0, default, precision: 3 },
        }
    }

    fn popup_spec(id: &'static str, default: u32) -> ParamSpec {
        ParamSpec { id, name: id, kind: ParamKind::Popup { choices: vec!["A".into(), "B".into(), "C".into()], default } }
    }

    const IDS: TimeParamIds =
        TimeParamIds { remap_mode: "mode", speed: "speed", frame_num: "frame", src_time_offset: "offset", mark_segs: "marks" };

    fn curves() -> ParamCurves {
        ParamCurves::new(vec![
            popup_spec("mode", 1),
            slider_spec("speed", 100.0),
            slider_spec("frame", 0.0),
            slider_spec("offset", 0.0),
            popup_spec("marks", 1),
        ])
    }

    #[test]
    fn the_fixed_reader_is_the_documented_expression() {
        assert_eq!(fixed_reader(6_553_600, 1.0), 100.0);
        assert_eq!(fixed_reader(0, 1.0), 0.0);
        assert_eq!(fixed_reader(-98_304, 1.0), -1.5);
        // 70 percent sensitivity: the slider is exact in 16.16, the division by 100.0f narrows once
        assert_eq!(fixed_reader(float_to_fixed(70.0), 1.0) / 100.0f32, 0.7f32);
        // 33.3 is stored as 2182349 / 65536 = 33.30000305175781 (it is not representable), narrowed to f32
        assert_eq!(float_to_fixed(33.3), 2_182_349);
        assert_eq!(fixed_reader(float_to_fixed(33.3), 1.0), (2_182_349.0f64 / 65536.0) as f32);
    }

    #[test]
    fn ticks_beyond_32_bits_are_errors() {
        assert_eq!(ticks_i32(-7, "t").unwrap(), -7);
        assert_eq!(ticks_i32(i64::from(i32::MAX), "t").unwrap(), i32::MAX);
        assert!(matches!(ticks_i32(i64::from(i32::MAX) + 1, "t"), Err(FxError::Unsupported(_))));
        assert!(ticks_i32(i64::MIN, "t").is_err());
    }

    #[test]
    fn curve_time_params_quantise_the_fixed_sliders_and_convert_ticks() {
        let mut c = curves();
        let base = TimeBase::default_for_rate(Rational::new(24, 1).unwrap());
        c.parse_cli("speed", "50@0;150@10", base).unwrap();
        c.set_constant("offset", ParamValue::Float(33.3)).unwrap();
        c.set_constant("frame", ParamValue::Float(2.5)).unwrap();
        c.set_constant("mode", ParamValue::Int(2)).unwrap();
        let p = CurveTimeParams::new(&c, IDS).unwrap();
        assert_eq!(p.remap_mode(0), 2);
        assert_eq!(p.frame_num(12345), 2.5);
        assert_eq!(p.src_time_offset_fixed(0), 2_182_349);
        assert_eq!(p.speed_fixed(-5), 50 * 65536, "before the first key the first value holds");
        assert_eq!(p.speed_fixed(500), 100 * 65536, "halfway between the keys at ticks 0 and 1000");
        assert_eq!(p.speed_fixed(1000), 150 * 65536);
        assert_eq!(p.speed_fixed(5000), 150 * 65536);
        // an odd linear value is rounded to the nearest 1/65536 away from zero at halves
        assert_eq!(p.speed_fixed(1), float_to_fixed(50.0 + 100.0 * 1.0 / 1000.0));
        assert_eq!(p.mark_segs(0), 1);
    }

    #[test]
    fn curve_time_params_reject_bad_wiring_instead_of_panicking() {
        let c = curves();
        let wrong = TimeParamIds { remap_mode: "speed", ..IDS };
        assert!(matches!(CurveTimeParams::new(&c, wrong), Err(FxError::BadParam { .. })));
        let missing = TimeParamIds { speed: "nope", ..IDS };
        assert!(matches!(CurveTimeParams::new(&c, missing), Err(FxError::UnknownParam(_))));
        let popup_as_slider = TimeParamIds { frame_num: "marks", ..IDS };
        assert!(CurveTimeParams::new(&c, popup_as_slider).is_ok(), "an integer sample is a valid slider reading");
    }

    #[test]
    fn hold_keys_on_a_popup_hold_and_floats_interpolate() {
        let mut c = curves();
        c.set_keys("marks", vec![
            Keyframe { time: 0, value: ParamValue::Int(1), hold: false, ease: false },
            Keyframe { time: 1000, value: ParamValue::Int(3), hold: false, ease: false },
        ])
        .unwrap();
        let p = CurveTimeParams::new(&c, IDS).unwrap();
        assert_eq!((p.mark_segs(999), p.mark_segs(1000)), (1, 3));
    }
}
