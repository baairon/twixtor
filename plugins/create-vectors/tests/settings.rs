//! The parameter to settings mapping (`glue::read_settings`, `Settings02::vector_settings`): which
//! popup value is which engine argument, the 16.16 readers, keyframes sampled at `current_time`,
//! the depth and quality of the context, and the helpers that decide which parameters select the
//! paths this crate refuses.

mod common;

use common::*;
use fx_core::{float_to_fixed, Depth, Effect, FxError, Quality};
use create_vectors::glue::{overlay_layers, point_belongs, points_in_use, read_settings, DOWNSAMPLE};
use create_vectors::CreateVectors;
use retime_engine::create_vectors::CreateVectorsSettings;
use retime_engine::frontend::fixed_reader;
use retime_engine::image::Fmt;

fn rig(params: &[(&str, &str)]) -> Rig {
    Rig::new(random_clip(Depth::U8, 6, 4, 8, 1), params).with_frames(8)
}

fn settings(params: &[(&str, &str)], n: u64) -> create_vectors::glue::Settings02 {
    let r = rig(params);
    read_settings(&r.cx(n), &r.env()).unwrap()
}

#[test]
fn the_defaults_are_the_parameter_defaults() {
    let s = settings(&[], 3);
    assert_eq!((s.current_time, s.time_step, s.time_scale), (300, 100, 3000));
    assert_eq!((s.width, s.height, s.fmt), (6, 4, Fmt::U8));
    assert!(s.hiq, "the context is full quality");
    // Display Vectors, Vector Resolution Normal, Debug View Blend, Draw Geom All, Track Frame Previous, Prep None
    assert_eq!((s.display, s.vector_resolution, s.debug_view_method, s.draw_geom, s.track_frame, s.prep), (2, 1, 2, 2, 1, 1));
    assert_eq!(s.max_displace, 2048.0);
    assert!(!s.track_rgb_a && s.main_bg_track);
    assert_eq!(s.main_bg_sensitivity.to_bits(), (70.0f32 / 100.0f32).to_bits(), "the slider divided by 100.0f in single precision");
    assert!(!s.display_source() && !s.debug());
    let v = s.vector_settings();
    assert_eq!(
        v,
        CreateVectorsSettings {
            quality: 4,
            hiq: true,
            sensitivity: 70.0f32 / 100.0f32,
            prep_contrast_edge: false,
            prep_delinearize: false,
            track_rgb_a: false,
            max_displace: 2048.0
        }
    );
}

#[test]
fn popups_map_to_the_engine_arguments() {
    // Vector Resolution: Normal (half res) -> 4, Full res -> 5
    assert_eq!(settings(&[("vector_resolution", "1")], 0).vector_settings().quality, 4);
    assert_eq!(settings(&[("vector_resolution", "2")], 0).vector_settings().quality, 5);
    // Prep: Contrast/Edge Enhance and Delinearize
    let p = |v: &str| {
        let s = settings(&[("prep", v)], 0).vector_settings();
        (s.prep_contrast_edge, s.prep_delinearize)
    };
    assert_eq!((p("1"), p("2"), p("3")), ((false, false), (true, false), (false, true)));
    // Track RGB+A
    assert!(settings(&[("track_rgb_a", "1")], 0).vector_settings().track_rgb_a);
    // Debug View Method: Warp 4, Blend 0, Diff 1, Detail Compare 2, Edge Compare 3
    let codes: Vec<i32> = (1..=5).map(|v| settings(&[("debug_view_method", &v.to_string())], 0).debug_view_code()).collect();
    assert_eq!(codes, vec![4, 0, 1, 2, 3]);
    // Display: 1 Source, 2 Vectors, 3 Debug
    let d = |v: &str| {
        let s = settings(&[("display", v)], 0);
        (s.display_source(), s.debug())
    };
    assert_eq!((d("1"), d("2"), d("3")), ((true, false), (false, false), (false, true)));
    // Track Frame and the others are carried as popup values
    assert_eq!(settings(&[("track_frame", "2")], 0).track_frame, 2);
    assert_eq!(settings(&[("draw_geom", "4")], 0).draw_geom, 4);
}

#[test]
fn sliders_go_through_the_16_16_fixed_reader() {
    // Max Displace 100.123456: the stored 16.16 integer, then x * 2^-16 in double narrowed to single
    let s = settings(&[("max_displace", "100.123456")], 0);
    let raw = float_to_fixed(100.123456);
    assert_eq!(s.max_displace.to_bits(), fixed_reader(raw, 1.0).to_bits());
    assert_eq!(raw, 6_561_691, "100.123456 * 65536 = 6561691.2, rounded");
    assert!((s.max_displace - 100.123456).abs() < 1.0 / 65536.0);
    // Main_BG: Sensitivity 33.333: quantised, then divided by 100.0f
    let s = settings(&[("main_bg_sensitivity", "33.333")], 0);
    let raw = float_to_fixed(33.333);
    assert_eq!(s.main_bg_sensitivity.to_bits(), (fixed_reader(raw, 1.0) / 100.0f32).to_bits());
    // zero sensitivity is a valid slider value
    assert_eq!(settings(&[("main_bg_sensitivity", "0")], 0).main_bg_sensitivity, 0.0);
    assert_eq!(settings(&[("main_bg_sensitivity", "100")], 0).main_bg_sensitivity, 1.0);
}

#[test]
fn the_downsample_ratio_scales_max_displace() {
    // the harness renders at full resolution: downsample 1 / 1, the scale is Max Displace itself
    assert_eq!(DOWNSAMPLE, (1, 1));
    assert_eq!(settings(&[("max_displace", "37.5")], 0).vector_settings().max_displace, 37.5);
}

#[test]
fn parameters_are_read_at_current_time() {
    // keyframes: Max Displace 100 at frame 0 and 200 at frame 10 interpolates linearly, a popup holds
    let params = [("max_displace", "100@0;200@10"), ("track_frame", "1@0;2@4"), ("prep", "1@0;3@6")];
    let at = |n| settings(&params, n);
    assert_eq!(at(0).max_displace, 100.0);
    assert_eq!(at(5).max_displace, 150.0);
    assert_eq!((at(3).track_frame, at(4).track_frame), (1, 2));
    assert_eq!((at(5).prep, at(6).prep), (1, 3));
}

#[test]
fn the_context_decides_quality_depth_and_size() {
    let r = rig(&[]);
    let mut cx = r.cx(0);
    cx.quality = Quality::Draft;
    let s = read_settings(&cx, &r.env()).unwrap();
    assert!(!s.hiq && !s.vector_settings().hiq);
    for (depth, fmt) in [(Depth::U8, Fmt::U8), (Depth::U16, Fmt::U16), (Depth::F32, Fmt::F32)] {
        let r = Rig::new(random_clip(depth, 3, 2, 2, 1), &[]);
        let s = read_settings(&r.cx(0), &r.env()).unwrap();
        assert_eq!((s.fmt, s.width, s.height), (fmt, 3, 2));
    }
}

#[test]
fn a_time_beyond_32_bits_is_an_error_not_a_wrap() {
    let r = rig(&[]);
    let mut cx = r.cx(0);
    cx.current_time = 1 << 40;
    match read_settings(&cx, &r.env()) {
        Err(FxError::Unsupported(m)) => assert!(m.contains("current_time") && m.contains("32 bit"), "{m}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn every_parameter_the_glue_reads_exists() {
    // the ids of the settings table above
    let wanted = [
        "display",
        "vector_resolution",
        "debug_view_method",
        "draw_geom",
        "max_displace",
        "track_frame",
        "prep",
        "track_rgb_a",
        "main_bg_sensitivity",
        "main_bg_track",
        "layers_output",
        "fg1_matte_chan",
        "fg1_sensitivity",
        "fg1_inv_matte_shrink",
        "fg1_track",
        "fg3_track",
        "pt1_use",
        "pt12_use",
        "pt1_pos",
        "pt12_pos",
        "pt1_color",
        "pt12_color",
    ];
    for id in wanted {
        assert!(CreateVectors.params().iter().any(|p| p.id == id), "{id}");
    }
}

#[test]
fn a_point_belongs_to_the_layer_its_use_value_names() {
    // value 2 to 4 is FG1 to FG3, 5 is Main_BG (the last layer); a layer that does not exist maps
    // to the last one. With one layer (no FG) every value 2 to 5 is the only layer
    for v in 2..=5 {
        assert!(point_belongs(v, 0, 1), "value {v}");
    }
    // Don't Use (1) maps to no layer
    assert!(!point_belongs(1, 0, 1));
    // two FG layers and Main_BG: three layers 0, 1, 2
    let owners: Vec<Vec<usize>> = (2..=5).map(|v| (0..3).filter(|l| point_belongs(v, *l, 3)).collect()).collect();
    assert_eq!(owners, vec![vec![0], vec![1], vec![2], vec![2]]);
}

#[test]
fn points_are_in_use_only_when_the_value_holds_at_both_times() {
    // Pt 1 is FG2 at every time; Pt 2 is Don't Use until frame 4 and FG1 from there; Pt 3 is FG1 until frame 4 and FG2
    // from there (popups hold their key); Pt 4 is Don't Use. Times are ticks, frame 4 is tick 400.
    let r = rig(&[("pt1_use", "3"), ("pt2_use", "1@0;2@4"), ("pt3_use", "2@0;3@4"), ("pt4_use", "1")]);
    let c = &r.curves;
    // before the change: Pt 1 and Pt 3 hold, Pt 2 is Don't Use
    assert_eq!(points_in_use(c, 100, 200, 1, 0, 0), vec![1, 3]);
    // across the change: Pt 3 differs between the two times, Pt 2 is Don't Use at the first time
    assert_eq!(points_in_use(c, 300, 500, 1, 0, 0), vec![1]);
    assert_eq!(points_in_use(c, 100, 500, 1, 0, 0), vec![1]);
    // after the change all three hold
    assert_eq!(points_in_use(c, 400, 500, 1, 0, 0), vec![1, 2, 3]);
    // with the second time Don't Use the point is not in use either (value differs)
    let r = rig(&[("pt5_use", "2@0;1@4")]);
    assert_eq!(points_in_use(&r.curves, 100, 500, 1, 0, 0), Vec::<usize>::new());
    // with three layers the value picks the layer: Pt 1 (FG2, layer 1) is not in use for layer 0 only
    let r = rig(&[("pt1_use", "3")]);
    assert_eq!(points_in_use(&r.curves, 100, 100, 3, 0, 0), Vec::<usize>::new());
    assert_eq!(points_in_use(&r.curves, 100, 100, 3, 1, 1), vec![1]);
    assert_eq!(points_in_use(&r.curves, 100, 100, 3, 0, 2), vec![1]);
}

#[test]
fn the_draw_geom_overlay_covers_the_layers_the_popup_names() {
    // 1 None draws nothing, 2 All every layer, 3 to 6 one layer (FG1, FG2, FG3, Main_BG) clamped to the last
    assert_eq!(overlay_layers(1, 4), None);
    assert_eq!(overlay_layers(2, 4), Some((0, 3)));
    assert_eq!(overlay_layers(2, 1), Some((0, 0)));
    assert_eq!(overlay_layers(3, 4), Some((0, 0)));
    assert_eq!(overlay_layers(4, 4), Some((1, 1)));
    assert_eq!(overlay_layers(5, 4), Some((2, 2)));
    assert_eq!(overlay_layers(6, 4), Some((3, 3)));
    // with fewer layers the value clamps to the last (Main_BG)
    assert_eq!(overlay_layers(5, 2), Some((1, 1)));
    assert_eq!(overlay_layers(6, 1), Some((0, 0)));
}

#[test]
fn the_fg_controls_and_layers_output_are_read_at_current_time() {
    let s = settings(&[], 0);
    assert_eq!(s.layers_output, 1);
    for fg in s.fg {
        assert_eq!((fg.matte_chan, fg.track), (1, true));
        assert_eq!(fg.sensitivity.to_bits(), (70.0f32 / 100.0f32).to_bits());
        assert_eq!(fg.shrink, 0.0);
    }
    let s = settings(
        &[("layers_output", "4"), ("fg2_matte_chan", "4"), ("fg2_sensitivity", "33.333"), ("fg2_inv_matte_shrink", "2.5@0;4.5@4"), ("fg3_track", "0")],
        2,
    );
    assert_eq!(s.layers_output, 4);
    assert_eq!(s.fg[1].matte_chan, 4);
    assert_eq!(s.fg[1].sensitivity.to_bits(), (fixed_reader(float_to_fixed(33.333), 1.0) / 100.0f32).to_bits());
    assert_eq!(s.fg[1].shrink, 3.5, "keyed, read at the current time, in pixels at the downsample ratio 1");
    assert!(!s.fg[2].track && s.fg[0].track);
    // the layers of a render: FG1 to FGn, then Main_BG with its own sensitivity and Track
    let s = settings(&[("main_bg_sensitivity", "40"), ("main_bg_track", "0"), ("fg1_sensitivity", "10")], 0);
    let l = s.layers(1);
    assert_eq!(l.len(), 2);
    assert_eq!((l[0].sensitivity, l[1].sensitivity, l[1].track, l[1].shrink), (0.1, 0.4, false, 0.0));
    assert_eq!(s.layers(0).len(), 1);
    assert_eq!(s.layers(3).len(), 4);
}

#[test]
fn layer_points_put_each_point_on_its_layer_at_both_times() {
    use create_vectors::glue::layer_points;
    use retime_engine::points::UserPoint;
    // Pt 1 on FG2, Pt 2 on Main_BG, Pt 3 Don't Use, Pt 4 changes layer between the two times
    let r = rig(&[("pt1_use", "3"), ("pt1_pos", "10,20@1;12,21@2"), ("pt2_use", "5"), ("pt2_pos", "5,6"), ("pt4_use", "2@0;3@2")]);
    let c = &r.curves;
    let p = layer_points(c, 200, 100, 2);
    assert_eq!(p.len(), 3);
    assert_eq!(p[0], Vec::<UserPoint>::new());
    assert_eq!(p[1], vec![UserPoint { at_a: [12.0, 21.0], at_b: [10.0, 20.0] }]);
    assert_eq!(p[2], vec![UserPoint { at_a: [5.0, 6.0], at_b: [5.0, 6.0] }]);
    // with no FG layer every point in use is on the only layer, in point order
    let p = layer_points(c, 200, 100, 0);
    assert_eq!(p, vec![vec![UserPoint { at_a: [12.0, 21.0], at_b: [10.0, 20.0] }, UserPoint { at_a: [5.0, 6.0], at_b: [5.0, 6.0] }]]);
    // FG2 does not exist with one FG layer: Pt 1 goes to Main_BG
    assert_eq!(layer_points(c, 200, 100, 1)[1].len(), 2);
}
