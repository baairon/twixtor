//! The settings the engine derives from plugin 03's parameters at `current_time`: popups minus one,
//! the Warping popup as the warp code, the 16.16 sliders as the integers they are quantised to, the
//! rate model, the layer probes, and the table that wires parameter ids to the shared reader.

mod common;

use common::*;
use fx_core::{Depth, Effect, EffectKind, FieldOrder, Frame, FxError, ParamKind, Quality};
use retime_fgbg::glue::{points_in_use, read_settings, warp_code};
use retime_fgbg::{RetimeFgBg, WIRING};
use retime_engine::host::EngineSettings;
use retime_engine::image::{Fmt, MissingLayerStyle};

fn clip() -> Vec<Frame> {
    (0..4).map(|_| Frame::new(4, 3, Depth::U8)).collect()
}

fn settings_of(rig: &Rig, n: u64) -> EngineSettings {
    read_settings(&rig.cx(n), &rig.env()).unwrap()
}

#[test]
fn defaults_at_output_frame_seven() {
    let s = settings_of(&Rig::new(clip(), &[]), 7);
    assert_eq!((s.current_time, s.time_step, s.local_time_step, s.time_scale), (700, 100, 100, 3000));
    assert_eq!((s.width, s.height, s.fmt, s.quality_high), (4, 3, Fmt::U8, true));
    // In FPS is Out FPS is on: both rates are 30, a source frame lasts one layer frame
    assert_eq!((s.rates.in_fps, s.rates.out_fps, s.rates.dur, s.rates.local_time_step), (30.0, 30.0, 100.0, 100));
    assert_eq!((s.vectors_quality, s.interp), (4, 1), "Best and Blend are the popup defaults 5 and 2 minus one");
    assert_eq!((s.blur, s.sensitivity), (0.0, 0.7f32), "Main_BG Sensitivity 70 percent");
    assert_eq!((s.prep_contrast_edge, s.prep_delinearize, s.smart_blend, s.warp_mode), (false, false, false, 0), "Warping Inverse");
    assert_eq!((s.video_360, s.cache_last, s.display_source, s.src_time_offset_fixed), (false, true, false, 0));
    assert_eq!((s.alt_a_present, s.no_tracking_layer, s.input_fields), (false, true, FieldOrder::Progressive));
    assert_eq!(s.missing_layer_style, MissingLayerStyle::BlueAlphaSlip);
}

#[test]
fn popups_are_read_minus_one_and_checkboxes_as_flags() {
    let rig = Rig::new(
        clip(),
        &[("vectors", "2"), ("frame_interp", "3"), ("prep", "3"), ("display", "2"), ("video_360", "1"), ("plenty_memory", "0"), ("input_fields", "3")],
    );
    let s = settings_of(&rig, 0);
    assert_eq!((s.vectors_quality, s.interp), (1, 2));
    assert_eq!((s.prep_contrast_edge, s.prep_delinearize), (false, true));
    assert_eq!((s.display_source, s.video_360, s.cache_last, s.input_fields), (true, true, false, FieldOrder::LowerFirst));
    // Prep = Contrast/Edge Enhance
    let s = settings_of(&Rig::new(clip(), &[("prep", "2")]), 0);
    assert_eq!((s.prep_contrast_edge, s.prep_delinearize), (true, false));
    // every vectors quality and interpolation popup value
    for v in 1..=5 {
        assert_eq!(settings_of(&Rig::new(clip(), &[("vectors", &v.to_string())]), 0).vectors_quality, v - 1);
    }
    for v in 1..=3 {
        assert_eq!(settings_of(&Rig::new(clip(), &[("frame_interp", &v.to_string())]), 0).interp, v - 1);
    }
}

#[test]
fn display_is_the_source_only_for_the_value_two() {
    for (value, source) in [(1, false), (2, true), (3, false), (4, false), (5, false), (6, false)] {
        let s = settings_of(&Rig::new(clip(), &[("display", &value.to_string())]), 0);
        assert_eq!(s.display_source, source, "display {value}");
    }
}

#[test]
fn the_warping_popup_gives_the_warp_code() {
    // 1 Inverse 0, 2 Inverse w/ Smart Blend 1, 3 Forward 2, 4 Forward no Smart Blend 3
    for (value, code) in [(1, 0), (2, 1), (3, 2), (4, 3)] {
        let s = settings_of(&Rig::new(clip(), &[("warping", &value.to_string())]), 0);
        assert_eq!((s.warp_mode, s.smart_blend), (code, false), "warping {value}");
    }
    // a value outside 1..=4 keeps 0
    for value in [0, 5, 7, -1, i32::MAX] {
        assert_eq!(warp_code(value), 0, "{value}");
    }
    // the popup is sampled at current_time: a hold key from frame 5 on
    let rig = Rig::new(clip(), &[("warping", "1@0;3@5")]);
    assert_eq!(settings_of(&rig, 4).warp_mode, 0);
    assert_eq!(settings_of(&rig, 5).warp_mode, 2);
}

#[test]
fn the_sliders_are_quantised_to_16_16_like_after_effects_stores_them() {
    // Main_BG Sensitivity 33.3 -> FLOAT2FIX 2182349 -> (2182349 / 65536) as f32 = 33.30000305 -> / 100.0f32
    let s = settings_of(&Rig::new(clip(), &[("main_bg_sensitivity", "33.3")]), 0);
    assert_eq!(s.sensitivity, 0.3330000340938568f32);
    assert_ne!(s.sensitivity, 0.333f32);
    // Blur Compensation 0.004 -> 262 / 65536 = 0.003997802734375
    let s = settings_of(&Rig::new(clip(), &[("blur_compensation", "0.004")]), 0);
    assert_eq!(s.blur, 0.003997802734375f32);
    assert_eq!(settings_of(&Rig::new(clip(), &[("blur_compensation", "2.5")]), 0).blur, 2.5);
    // plugin 03 has no Sensitivity of its own and no Src Time Offset: the engine's single layer sensitivity is Main_BG's
    let ids: Vec<&str> = RetimeFgBg.params().iter().map(|p| p.id).collect();
    assert!(!ids.contains(&"sensitivity") && !ids.contains(&"src_time_offset") && !ids.contains(&"smart_blend"));
    // the other layers' sensitivities do not reach the single layer settings
    let s = settings_of(&Rig::new(clip(), &[("fg1_sensitivity", "10"), ("fg2_sensitivity", "20")]), 0);
    assert_eq!(s.sensitivity, 0.7f32);
}

#[test]
fn keyframes_are_read_at_current_time() {
    let rig = Rig::new(clip(), &[("blur_compensation", "0@0;4@10"), ("vectors", "5@0;2@3")]);
    assert_eq!(settings_of(&rig, 5).blur, 2.0);
    assert_eq!(settings_of(&rig, 2).vectors_quality, 4);
    assert_eq!(settings_of(&rig, 3).vectors_quality, 1, "a popup holds its key");
}

#[test]
fn the_rate_model_uses_the_input_frame_rate_unless_the_two_rates_are_tied() {
    // In FPS is Out FPS off, Input: Frame Rate 24 in a 30 fps composition: a source frame lasts 125 ticks
    let s = settings_of(&Rig::new(clip(), &[("in_fps_is_out_fps", "0"), ("input_frame_rate", "24")]), 0);
    assert_eq!((s.rates.in_fps, s.rates.out_fps, s.rates.dur), (24.0, 30.0, 125.0));
    // the stored 29.97 (an f32 widened) snaps to the NTSC double
    let s = settings_of(&Rig::new(clip(), &[("in_fps_is_out_fps", "0")]), 0);
    assert_eq!(s.rates.in_fps, 29.97);
    // with the tie on the Input: Frame Rate value is ignored
    let s = settings_of(&Rig::new(clip(), &[("input_frame_rate", "24")]), 0);
    assert_eq!((s.rates.in_fps, s.rates.out_fps), (30.0, 30.0));
    // fields double the input rate
    let s = settings_of(&Rig::new(clip(), &[("input_fields", "2")]), 0);
    assert_eq!((s.rates.in_fps, s.rates.dur, s.input_fields), (60.0, 50.0, FieldOrder::UpperFirst));
    // a composition rate below 0.01 is a `BadParam` error (only when the two rates are not tied)
    let rig = Rig::new(clip(), &[("in_fps_is_out_fps", "0")]);
    let mut cx = rig.cx(0);
    if let Ok(zero) = fx_core::Rational::new(1, 1000) {
        cx.fps = zero;
        assert!(matches!(read_settings(&cx, &rig.env()), Err(FxError::BadParam { .. })));
    }
}

#[test]
fn layer_probes_decide_alt_a_and_the_tracking_layer() {
    let same = || vec![Frame::new(4, 3, Depth::U8)];
    let zero = || vec![Frame::new(0, 0, Depth::U8)];
    let s = settings_of(&Rig::new(clip(), &[]).with_layer("alt_source_a", same()).with_layer("alt_source_b", same()), 0);
    assert_eq!((s.alt_a_present, s.no_tracking_layer), (true, false));
    let s = settings_of(&Rig::new(clip(), &[]).with_layer("alt_source_a", zero()).with_layer("alt_source_b", zero()), 0);
    assert_eq!((s.alt_a_present, s.no_tracking_layer), (false, true), "a layer without pixels is not set");
}

#[test]
fn fg_matte_probes_count_the_layers_from_fg1_and_flag_a_size_mismatch() {
    use retime_fgbg::glue::fg_probe;
    let same = || vec![Frame::new(4, 3, Depth::U8)];
    let wide = || vec![Frame::new(5, 3, Depth::U8)];
    let probe = |rig: Rig| fg_probe(&rig.env(), 4, 3);
    assert_eq!(probe(Rig::new(clip(), &[])), (0, false));
    assert_eq!(probe(Rig::new(clip(), &[]).with_layer("fg1_matte", same())), (1, false));
    assert_eq!(probe(Rig::new(clip(), &[]).with_layer("fg1_matte", same()).with_layer("fg2_matte", same())), (2, false));
    assert_eq!(probe(Rig::new(clip(), &[]).with_layer("fg1_matte", same()).with_layer("fg2_matte", same()).with_layer("fg3_matte", wide())), (3, true));
    // the loop stops at the first empty layer: FG2 without FG1 is nothing
    assert_eq!(probe(Rig::new(clip(), &[]).with_layer("fg2_matte", wide())), (0, false));
    assert_eq!(probe(Rig::new(clip(), &[]).with_layer("fg1_matte", wide()).with_layer("fg3_matte", wide())), (1, true));
}

#[test]
fn track_points_in_use_follow_the_two_time_rule() {
    // in use: not "Don't Use" (1) at the first time and the same value at the second
    let rig = Rig::new(clip(), &[("pt2_use", "3"), ("pt5_use", "1@0;2@3"), ("pt7_use", "2@0;5@3"), ("pt12_use", "5")]);
    let curves = &rig.curves;
    assert_eq!(points_in_use(curves, 0, 0), vec![2, 7, 12]);
    // from frame 3 (tick 300) on pt5 is 2 and pt7 is 5: both in use at (400, 400)
    assert_eq!(points_in_use(curves, 400, 400), vec![2, 5, 7, 12]);
    // pt7 changes value between the two times (2 then 5) and pt5 starts unused: neither is in use, in either order
    assert_eq!(points_in_use(curves, 0, 400), vec![2, 12]);
    assert_eq!(points_in_use(curves, 400, 0), vec![2, 12]);
}

#[test]
fn quality_and_time_come_from_the_context() {
    let rig = Rig::new(clip(), &[]);
    let mut cx = rig.cx(3);
    cx.quality = Quality::Draft;
    cx.local_time_step = 120;
    let s = read_settings(&cx, &rig.env()).unwrap();
    assert_eq!((s.quality_high, s.local_time_step, s.current_time), (false, 120, 300));
    cx.current_time = i64::from(i32::MAX) + 1;
    assert!(matches!(read_settings(&cx, &rig.env()), Err(FxError::Unsupported(_))));
}

#[test]
fn the_wiring_names_real_parameters_of_the_expected_kinds() {
    let params = RetimeFgBg.params();
    let kind = |id: &str| params.iter().find(|p| p.id == id).unwrap_or_else(|| panic!("no parameter {id}")).kind.clone();
    let w = &WIRING;
    for id in [w.settings.display, w.settings.input_fields, w.settings.vectors, w.settings.prep, w.settings.frame_interp].into_iter().flatten() {
        assert!(matches!(kind(id), ParamKind::Popup { .. }), "{id}");
    }
    for id in [w.settings.video_360, w.settings.in_fps_is_out_fps, w.settings.plenty_memory].into_iter().flatten() {
        assert!(matches!(kind(id), ParamKind::Checkbox { .. }), "{id}");
    }
    for id in [w.settings.input_frame_rate, w.settings.sensitivity, w.settings.blur_compensation].into_iter().flatten() {
        assert!(matches!(kind(id), ParamKind::Float { .. }), "{id}");
    }
    // plugin 03 has no Src Time Offset and no Smart Blend checkbox
    assert!(w.settings.src_time_offset.is_none() && w.settings.smart_blend.is_none());
    assert_eq!(w.settings.sensitivity, Some("main_bg_sensitivity"));
    assert!(matches!(kind(w.time.remap_mode), ParamKind::Popup { .. }));
    assert!(matches!(kind(w.time.mark_segs), ParamKind::Popup { .. }));
    for id in [w.time.speed, w.time.frame_num] {
        assert!(matches!(kind(id), ParamKind::Float { .. }), "{id}");
    }
    for id in ["warping", "draw_geom", "track_rgb_a"] {
        kind(id);
    }
    for n in 1..=12 {
        assert!(matches!(kind(&format!("pt{n}_use")), ParamKind::Popup { .. }));
    }
    // the layers the effect lists: the engine's two (same ids as plugin 01) and the three FG mattes
    let layers: Vec<_> = RetimeFgBg.layers().iter().map(|l| (l.id, l.name, l.optional)).collect();
    assert_eq!(
        layers,
        vec![
            (w.layers.alt_a, "Alt Source A", true),
            (w.layers.alt_b, "Alt Source B", true),
            ("fg1_matte", "FG1 Matte", true),
            ("fg2_matte", "FG2 Matte", true),
            ("fg3_matte", "FG3 Matte", true),
        ]
    );
    assert_eq!((w.layers.alt_a, w.layers.alt_b), ("alt_source_a", "alt_source_b"));
    assert_eq!(RetimeFgBg.kind(), EffectKind::Temporal);
}

#[test]
fn the_wrap_flag_is_off_and_the_device_request_is_carried_without_changing_a_pixel() {
    // half speed: output frame 1 lies between two source frames, so the vector path runs
    let rig = Rig::new(random_clip(Depth::U16, 16, 12, 4, 3), &[("speed_percent", "50")]);
    let plain = settings_of(&rig, 1);
    assert!(!plain.wrap_360);
    assert_eq!(plain.device, fx_core::DeviceRequest::Cpu);
    let want = rig.render(1).unwrap();
    for device in [fx_core::DeviceRequest::Gpu, fx_core::DeviceRequest::GpuIndex(0)] {
        let mut cx = rig.cx(1);
        cx.device = device;
        let s = read_settings(&cx, &rig.env()).unwrap();
        assert_eq!(s, EngineSettings { device, ..plain.clone() });
        assert_eq!(rig.render_cx(&cx).unwrap(), want, "{device}");
    }
    let s = settings_of(&Rig::new(clip(), &[("video_360", "1")]), 0);
    assert!(s.video_360 && !s.wrap_360);
}
