//! The settings the engine derives from plugin 01's parameters at `current_time`: popups minus one,
//! the 16.16 sliders as fixed point integers, the rate model, the layer probes.

mod common;

use common::*;
use fx_core::{float_to_fixed, Depth, FieldOrder, Frame, FxError, Quality, Rational};
use retime::WIRING;
use retime_engine::frontend::{fixed_reader, read_settings};
use retime_engine::host::EngineSettings;
use retime_engine::image::{Fmt, MissingLayerStyle};

fn clip() -> Vec<Frame> {
    (0..4).map(|_| Frame::new(4, 3, Depth::U8)).collect()
}

fn settings_of(rig: &Rig, n: u64) -> EngineSettings {
    read_settings(&WIRING, &rig.cx(n), &rig.env()).unwrap()
}

#[test]
fn defaults_at_output_frame_seven() {
    let s = settings_of(&Rig::new(clip(), &[]), 7);
    assert_eq!((s.current_time, s.time_step, s.local_time_step, s.time_scale), (700, 100, 100, 3000));
    assert_eq!((s.width, s.height, s.fmt, s.quality_high), (4, 3, Fmt::U8, true));
    // In FPS is Out FPS is on: both rates are 30, a source frame lasts one layer frame
    assert_eq!((s.rates.in_fps, s.rates.out_fps, s.rates.dur, s.rates.local_time_step), (30.0, 30.0, 100.0, 100));
    assert_eq!((s.vectors_quality, s.interp), (4, 1), "Best and Blend are the popup defaults 5 and 2 minus one");
    assert_eq!((s.blur, s.sensitivity), (0.0, 0.7f32));
    assert_eq!((s.prep_contrast_edge, s.prep_delinearize, s.smart_blend, s.warp_mode), (false, false, false, 0));
    assert_eq!((s.video_360, s.cache_last, s.display_source, s.src_time_offset_fixed), (false, true, false, 0));
    assert_eq!((s.alt_a_present, s.no_tracking_layer, s.input_fields), (false, true, FieldOrder::Progressive));
    assert_eq!(s.missing_layer_style, MissingLayerStyle::HalfBlue);
}

#[test]
fn popups_are_read_minus_one_and_checkboxes_as_flags() {
    let rig = Rig::new(
        clip(),
        &[
            ("vectors", "2"),
            ("frame_interp", "3"),
            ("prep", "3"),
            ("smart_blend", "1"),
            ("display", "2"),
            ("video_360", "1"),
            ("plenty_memory", "0"),
        ],
    );
    let s = settings_of(&rig, 0);
    assert_eq!((s.vectors_quality, s.interp), (1, 2));
    assert_eq!((s.prep_contrast_edge, s.prep_delinearize), (false, true));
    assert_eq!((s.smart_blend, s.warp_mode), (true, 1), "plugin 01's warping mode is the Smart Blend checkbox");
    assert_eq!((s.display_source, s.video_360, s.cache_last), (true, true, false));
    assert!(settings_of(&Rig::new(clip(), &[("prep", "2")]), 0).prep_contrast_edge);
}

#[test]
fn fixed_sliders_are_quantised_to_sixteen_sixteen() {
    let rig = Rig::new(clip(), &[("blur_compensation", "1.23456"), ("sensitivity", "33.3"), ("src_time_offset", "2.9")]);
    let s = settings_of(&rig, 0);
    // 1.23456 * 65536 = 80908.4 -> 80908 / 65536 = 1.23455810546875, exact in f32 (a multiple of 2^-16 below 2^8)
    assert_eq!(s.blur, (80_908.0f64 / 65_536.0) as f32);
    assert_ne!(s.blur, 1.23456f32);
    // 33.3 * 65536 = 2182348.8 -> 2182349; the reader narrows to f32, then the settings function divides by 100.0f
    assert_eq!(float_to_fixed(33.3), 2_182_349);
    assert_eq!(s.sensitivity, fixed_reader(2_182_349, 1.0) / 100.0f32);
    // 2.9 * 65536 = 190054.4 -> 190054; Display == Source truncates the frames later
    assert_eq!(s.src_time_offset_fixed, 190_054);
}

#[test]
fn parameters_are_read_at_the_current_time() {
    // Blur Compensation animated from 0 at frame 0 to 2 at frame 10: 1.0 at frame 5 (tick 500)
    let rig = Rig::new(clip(), &[("blur_compensation", "0@0;2@10")]);
    assert_eq!(settings_of(&rig, 5).blur, 1.0);
    assert_eq!(settings_of(&rig, 0).blur, 0.0);
    assert_eq!(settings_of(&rig, 20).blur, 2.0);
}

#[test]
fn the_rate_model_follows_the_input_rate_the_composition_rate_and_the_fields() {
    // 24 fps footage in a 30 fps composition: one source frame lasts 100 * 30 / 24 = 125 ticks
    let s = settings_of(&Rig::new(clip(), &[("in_fps_is_out_fps", "0"), ("input_frame_rate", "24")]), 0);
    assert_eq!((s.rates.in_fps, s.rates.out_fps, s.rates.dur), (24.0, 30.0, 125.0));
    // the default input rate 29.97f32 snaps to the NTSC double
    let s = settings_of(&Rig::new(clip(), &[("in_fps_is_out_fps", "0")]), 0);
    assert_eq!(s.rates.in_fps.to_bits(), 0x403d_f851_eb85_1eb8);
    // fields double the source rate
    let s = settings_of(&Rig::new(clip(), &[("input_fields", "2")]), 0);
    assert_eq!((s.input_fields, s.rates.in_fps, s.rates.out_fps), (FieldOrder::UpperFirst, 60.0, 30.0));
    let s = settings_of(&Rig::new(clip(), &[("input_fields", "3")]), 0);
    assert_eq!(s.input_fields, FieldOrder::LowerFirst);
    // a composition rate at or below 0.01 is error 5 (bad parameter)
    let rig = Rig::new(clip(), &[("in_fps_is_out_fps", "0")]);
    let mut cx = rig.cx(0);
    cx.fps = Rational::new(1, 200).unwrap();
    assert!(matches!(read_settings(&WIRING, &cx, &rig.env()), Err(FxError::BadParam { .. })));
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
fn quality_and_time_come_from_the_context() {
    let rig = Rig::new(clip(), &[]);
    let mut cx = rig.cx(3);
    cx.quality = Quality::Draft;
    cx.local_time_step = 120;
    let s = read_settings(&WIRING, &cx, &rig.env()).unwrap();
    assert_eq!((s.quality_high, s.local_time_step, s.current_time), (false, 120, 300));
    cx.current_time = i64::from(i32::MAX) + 1;
    assert!(matches!(read_settings(&WIRING, &cx, &rig.env()), Err(FxError::Unsupported(_))));
}

#[test]
fn the_wiring_names_real_parameters_of_the_expected_kinds() {
    use fx_core::{Effect, ParamKind};
    let params = retime::Retime.params();
    let kind = |id: &str| params.iter().find(|p| p.id == id).unwrap_or_else(|| panic!("no parameter {id}")).kind.clone();
    let w = &WIRING;
    for id in [w.settings.display, w.settings.input_fields, w.settings.vectors, w.settings.prep, w.settings.frame_interp].into_iter().flatten() {
        assert!(matches!(kind(id), ParamKind::Popup { .. }), "{id}");
    }
    for id in [w.settings.video_360, w.settings.in_fps_is_out_fps, w.settings.plenty_memory, w.settings.smart_blend].into_iter().flatten() {
        assert!(matches!(kind(id), ParamKind::Checkbox { .. }), "{id}");
    }
    for id in [w.settings.input_frame_rate, w.settings.src_time_offset, w.settings.sensitivity, w.settings.blur_compensation].into_iter().flatten() {
        assert!(matches!(kind(id), ParamKind::Float { .. }), "{id}");
    }
    assert!(matches!(kind(w.time.remap_mode), ParamKind::Popup { .. }));
    assert!(matches!(kind(w.time.mark_segs), ParamKind::Popup { .. }));
    for id in [w.time.speed, w.time.frame_num, w.time.src_time_offset] {
        assert!(matches!(kind(id), ParamKind::Float { .. }), "{id}");
    }
    // the layers the effect lists are the ones the engine fetches
    let layers: Vec<_> = retime::Retime.layers().iter().map(|l| (l.id, l.optional)).collect();
    assert_eq!(layers, vec![(w.layers.alt_a, true), (w.layers.alt_b, true)]);
    assert_eq!(retime::Retime.kind(), fx_core::EffectKind::Temporal);
}

#[test]
fn the_wrap_flag_is_off_and_the_device_request_is_carried_without_changing_a_pixel() {
    // half speed: output frame 1 lies between two source frames, so the vector path runs
    let rig = Rig::new(random_clip(Depth::U8, 16, 12, 4, 7), &[("speed_percent", "50")]);
    let plain = settings_of(&rig, 1);
    assert!(!plain.wrap_360);
    assert_eq!(plain.device, fx_core::DeviceRequest::Cpu);
    let want = rig.render(1).unwrap();
    for device in [fx_core::DeviceRequest::Gpu, fx_core::DeviceRequest::GpuIndex(2)] {
        let mut cx = rig.cx(1);
        cx.device = device;
        let s = read_settings(&WIRING, &cx, &rig.env()).unwrap();
        assert_eq!(s, EngineSettings { device, ..plain.clone() });
        // the engine only carries the request: the pixels are the CPU's
        assert_eq!(rig.render_cx(&cx).unwrap(), want, "{device}");
    }
    // the 360 Video checkbox is not the wrap flag
    let s = settings_of(&Rig::new(clip(), &[("video_360", "1")]), 0);
    assert!(s.video_360 && !s.wrap_360);
}
