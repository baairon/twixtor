//! The mapping of plugin 04's parameters to the engine settings, read through
//! `glue::read_settings`, and the layer probes (`glue::probe`).

mod common;

use common::*;
use fx_core::{Depth, FieldOrder, Frame, ParamCurves, Quality, Rational};
use vectors_in::glue::{self, Settings};
use retime_engine::image::{ErrorKind, Fmt, MissingLayerStyle};

const W: usize = 8;
const H: usize = 6;

fn rig(params: &[(&str, &str)]) -> Rig {
    Rig::new(random_clip(Depth::U8, W, H, 6, 1), params).with_frames(6)
}

fn read(r: &Rig, n: u64) -> Settings {
    glue::read_settings(&r.cx(n), &r.env()).unwrap()
}

#[test]
fn defaults_map_to_the_engine_settings() {
    let s = read(&rig(&[]), 3);
    let e = s.engine;
    assert!(!s.rate_error);
    assert_eq!((e.current_time, e.time_step, e.local_time_step, e.time_scale), (300, 100, 100, 3000));
    assert_eq!((e.width, e.height, e.fmt), (W, H, Fmt::from_depth(Depth::U8)));
    assert!(e.quality_high);
    assert!(!e.display_source);
    assert_eq!(e.src_time_offset_fixed, 0);
    // the constants of the call (6.3): vectors quality 4, Sensitivity 0.7f, no Prep, no Smart Blend, no 360, no cache
    assert_eq!(e.vectors_quality, 4);
    assert_eq!(e.sensitivity.to_bits(), 0x3f33_3333);
    assert!(!e.prep_contrast_edge && !e.prep_delinearize && !e.smart_blend && !e.video_360 && !e.cache_last);
    assert_eq!(e.input_fields, FieldOrder::Progressive);
    // Frame Interp default Blend (popup 2) is 1, Warping default Forward (popup 2) is the warp code 2
    assert_eq!(e.interp, 1);
    assert_eq!(e.warp_mode, 2);
    assert_eq!(e.blur, 0.0);
    assert!(!e.alt_a_present);
    assert!(e.no_tracking_layer);
    assert_eq!(e.missing_layer_style, MissingLayerStyle::BlueAlphaSlip);
    // the rate model with the checkbox on: in = 30, out = the composition rate (30)
    assert_eq!((e.rates.in_fps, e.rates.out_fps, e.rates.dur, e.rates.local_time_step), (30.0, 30.0, 100.0, 100));
}

#[test]
fn popups_are_read_minus_one_and_warping_maps_to_the_core_codes() {
    for (popup, interp) in [("1", 0), ("2", 1), ("3", 2)] {
        assert_eq!(read(&rig(&[("frame_interp", popup)]), 0).engine.interp, interp);
    }
    for (popup, code) in [("1", 0), ("2", 2)] {
        assert_eq!(read(&rig(&[("warping", popup)]), 0).engine.warp_mode, code);
    }
    assert!(read(&rig(&[("display", "2")]), 0).engine.display_source);
    for display in ["1", "3", "4", "5", "6"] {
        assert!(!read(&rig(&[("display", display)]), 0).engine.display_source, "display {display}");
    }
}

#[test]
fn blur_compensation_is_quantised_to_16_16() {
    // 0.3 -> FLOAT2FIX = 19661 (0.3 * 65536 = 19660.8, rounded half away), read back as (f32)(19661 / 65536)
    let e = read(&rig(&[("blur_compensation", "0.3")]), 0).engine;
    assert_eq!(e.blur, (19661.0f64 * (1.0 / 65536.0)) as f32);
    assert_ne!(e.blur, 0.3f32);
    assert_eq!(read(&rig(&[("blur_compensation", "2.5")]), 0).engine.blur, 2.5);
}

#[test]
fn parameters_are_read_at_current_time() {
    let r = rig(&[("frame_interp", "1@0;3@2"), ("blur_compensation", "0@0;4@4")]);
    // popups hold between keys, the slider is linear: frame 3 is after the popup key at frame 2, and 3/4 of the way
    // along the slider
    let e = read(&r, 3).engine;
    assert_eq!(e.interp, 2);
    assert_eq!(e.blur, 3.0);
    let e = read(&r, 1).engine;
    assert_eq!(e.interp, 0);
    assert_eq!(e.blur, 1.0);
}

#[test]
fn rate_model_swaps_the_order_of_the_checkbox_and_the_composition_rate() {
    // checkbox on: in = 30.0 whatever is typed, out = composition rate
    let mut r = rig(&[("input_frame_rate", "12")]);
    let mut cx = r.cx(0);
    cx.fps = Rational::new(24, 1).unwrap();
    let e = glue::read_settings(&cx, &r.env()).unwrap().engine;
    assert_eq!((e.rates.in_fps, e.rates.out_fps), (30.0, 24.0));
    assert_eq!(e.rates.dur, 80.0);
    // checkbox off: in = the typed value, out = the composition rate
    r = rig(&[("in_fps_is_out_fps", "0"), ("input_frame_rate", "12")]);
    let e = glue::read_settings(&cx, &r.env()).unwrap().engine;
    assert_eq!((e.rates.in_fps, e.rates.out_fps, e.rates.dur), (12.0, 24.0, 200.0));
    // the plugin registers no Output: Frame Rate, so the id is refused
    use fx_core::Effect;
    let mut c = ParamCurves::new(vectors_in::VectorsIn.params());
    assert!(c.parse_cli("output_frame_rate", "50", time()).is_err());
}

#[test]
fn the_ntsc_rates_snap_and_the_typed_rate_goes_through_f32() {
    // 29.97 typed: the slider is an f32 (29.969999313...), the snap makes it exactly 29.97
    let r = rig(&[("in_fps_is_out_fps", "0"), ("input_frame_rate", "29.97")]);
    let e = read(&r, 0).engine;
    assert_eq!(e.rates.in_fps, 29.97);
    // the true NTSC rate 30000/1001 is 3e-5 away and is not snapped
    let mut cx = r.cx(0);
    cx.fps = Rational::new(30000, 1001).unwrap();
    let e = glue::read_settings(&cx, &r.env()).unwrap().engine;
    assert_eq!(e.rates.out_fps, 30000.0 / 1001.0);
    // 59.94 and 23.976 snap as well
    for (typed, snapped) in [("59.94", 59.94), ("23.976", 23.976)] {
        let e = read(&rig(&[("in_fps_is_out_fps", "0"), ("input_frame_rate", typed)]), 0).engine;
        assert_eq!(e.rates.in_fps, snapped, "{typed}");
    }
}

#[test]
fn a_rate_at_or_below_the_limits_raises_the_flag() {
    let r = rig(&[]);
    let mut cx = r.cx(0);
    for (num, den, error) in [(1, 1000, true), (1, 100, false), (29, 1, false)] {
        cx.fps = Rational::new(num, den).unwrap();
        assert_eq!(glue::read_settings(&cx, &r.env()).unwrap().rate_error, error, "{num}/{den}");
    }
    // the input rate: the slider cannot go below 0.1, so the input check never fires from the UI
    let r = rig(&[("in_fps_is_out_fps", "0"), ("input_frame_rate", "0.11")]);
    assert!(!read(&r, 0).rate_error);
}

#[test]
fn the_32_bit_tick_guard() {
    let r = rig(&[]);
    let mut cx = r.cx(0);
    cx.current_time = 1 << 40;
    assert!(glue::read_settings(&cx, &r.env()).is_err());
    cx = r.cx(0);
    cx.quality = Quality::Draft;
    assert!(!glue::read_settings(&cx, &r.env()).unwrap().engine.quality_high);
}

#[test]
fn the_source_layer_sets_the_main_layer_flag() {
    let r = rig(&[]).with_layer("source", random_clip(Depth::U8, W, H, 3, 2));
    assert!(read(&r, 0).engine.alt_a_present);
    assert!(!read(&rig(&[]), 0).engine.alt_a_present);
}

#[test]
fn the_probe_reports_presence_failures_and_the_fg_count() {
    let layer = |w: usize, h: usize| random_clip(Depth::U8, w, h, 12, 7);
    let r = rig(&[]);
    let p = glue::probe(&r.env(), &r.cx(2)).unwrap();
    assert_eq!((p.has_prev, p.has_next, p.fg_count), (false, false, 0));
    assert_eq!(p.failure, Some(ErrorKind::SizeMismatch), "no vector layer");
    let r = rig(&[]).with_layer("vectors_prev", layer(W, H));
    let p = glue::probe(&r.env(), &r.cx(2)).unwrap();
    assert_eq!((p.has_prev, p.has_next, p.failure), (true, false, None));
    let r = rig(&[]).with_layer("vectors_next", layer(W, H)).with_layer("fg1_matte", layer(W, H)).with_layer("fg2_matte", layer(W, H));
    let p = glue::probe(&r.env(), &r.cx(2)).unwrap();
    assert_eq!((p.has_prev, p.has_next, p.fg_count, p.failure), (false, true, 2, None));
    assert_eq!(glue::fg_count(&r.env()), 2);
    // the first problem decides: a wrong size Source before a wrong size vector layer, both red here
    let r = rig(&[]).with_layer("vectors_next", layer(W + 1, H));
    assert_eq!(glue::probe(&r.env(), &r.cx(2)).unwrap().failure, Some(ErrorKind::SizeMismatch));
    // a vector layer with no frame at the current time (a layer that is shorter than the clip) keeps its size
    // outside its clip: present
    let short: Vec<Frame> = layer(W, H).into_iter().take(2).collect();
    let r = rig(&[]).with_layer("vectors_next", short).with_layer("vectors_prev", layer(W, H));
    assert!(glue::probe(&r.env(), &r.cx(1)).unwrap().has_next);
    assert!(glue::probe(&r.env(), &r.cx(2)).unwrap().has_next);
}

#[test]
fn the_wiring_names_exist_in_params() {
    use fx_core::Effect;
    let ids: Vec<&str> = vectors_in::VectorsIn.params().iter().map(|p| p.id).collect();
    for id in [
        "display",
        "in_fps_is_out_fps",
        "input_frame_rate",
        "frame_interp",
        "warping",
        "blur_compensation",
        "vectors_max_displace",
        "vec_scale_x",
        "vec_scale_y",
        "time_remap_mode",
        "speed_percent",
        "frame_num",
        "src_mark_segments",
    ] {
        assert!(ids.contains(&id), "{id}");
    }
    // the layer ids of layers() are the ones the glue and the prerender use
    let layers: Vec<&str> = vectors_in::VectorsIn.layers().iter().map(|l| l.id).collect();
    assert_eq!(layers, ["source", "vectors_prev", "vectors_next", "fg1_matte", "fg2_matte", "fg3_matte"]);
    assert!(vectors_in::VectorsIn.layers().iter().all(|l| l.optional));
    assert_eq!(vectors_in::VectorsIn.kind(), fx_core::EffectKind::Temporal);
}

#[test]
fn the_wrap_flag_is_off_and_the_device_request_is_carried() {
    let r = rig(&[]);
    let e = read(&r, 2).engine;
    assert!(!e.wrap_360);
    assert_eq!(e.device, fx_core::DeviceRequest::Cpu);
    // every hole fill takes the tap above first in its vertical border rows
    assert_eq!(e.stages.get().vertical_taps(), retime_engine::holefill::VerticalTaps::AboveFirst);
    for device in [fx_core::DeviceRequest::Gpu, fx_core::DeviceRequest::GpuIndex(4)] {
        let mut cx = r.cx(2);
        cx.device = device;
        let got = glue::read_settings(&cx, &r.env()).unwrap();
        // the stages are the above-first fill around the Gouraud forward warp over the context's set, a new
        // handle per read
        assert!(!got.engine.stages.is_cpu());
        assert_eq!(got.engine.stages.get().vertical_taps(), retime_engine::holefill::VerticalTaps::AboveFirst);
        assert_eq!(got.engine, retime_engine::host::EngineSettings { device, stages: got.engine.stages.clone(), ..e.clone() });
        assert!(!got.rate_error);
    }
}
