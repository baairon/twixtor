//! The vector path through plugin 03's front-end (Warping Inverse).
//!
//! Two kinds of check. (1) The engine behind plugin 03 is the engine behind plugin 01, and with the
//! same settings the two front-ends must produce the same bytes: the settings mapping (Main_BG
//! Sensitivity as the solver's sensitivity, Prep, Blur Compensation, the interpolation modes, 360
//! Video, the tracking layer, the segment markers) is proven by rendering a moving clip through
//! both effects and comparing every output frame. (2) An analytic ground truth
//! (`common/moving.rs`): a texture that translates by a known number of pixels per frame, so the
//! exact in-between frame is known in closed form and the result is judged against it.

mod common;

use common::moving::*;
use common::*;
use fx_core::{Depth, Effect, Env, Frame, LayerSet, MemLayer, ParamCurves};
use retime::Retime;
use std::sync::Arc;

const W: usize = 48;
const H: usize = 32;
const MARGIN: usize = 8;

/// The output frame `n` of plugin 01 on a clip, with CLI style parameters (plugin 01's ids).
fn render01(clip: &[Frame], tracking: Option<&[Frame]>, params: &[(&str, &str)], n: u64, frames: u64) -> Frame {
    let mut curves = ParamCurves::new(Retime.params());
    for (id, text) in params {
        curves.parse_cli(id, text, time()).unwrap();
    }
    let input = MemLayer::new(time(), clip.to_vec()).unwrap();
    let mut layers = LayerSet::new();
    if let Some(t) = tracking {
        layers.insert("alt_source_b", Arc::new(MemLayer::new(time(), t.to_vec()).unwrap()));
    }
    let env = Env { params: &curves, input: &input, layers: &layers };
    let (w, h, d) = (clip[0].width, clip[0].height, clip[0].depth());
    let cx = fx_core::TemporalContext::new(w, h, d, time(), n, frames);
    let mut out = Frame::new(w, h, d);
    Retime.render_temporal(&cx, &env, &mut out).unwrap();
    out
}

/// Plugin 03 on the same clip with its own ids.
fn render03(clip: &[Frame], tracking: Option<&[Frame]>, params: &[(&str, &str)], n: u64, frames: u64) -> Frame {
    let mut rig = Rig::new(clip.to_vec(), params).with_frames(frames);
    if let Some(t) = tracking {
        rig = rig.with_layer("alt_source_b", t.to_vec());
    }
    rig.render(n).unwrap()
}

/// `(plugin 01 parameters, plugin 03 parameters)`: the same settings spelled with each plugin's ids.
type Pair = (Vec<(&'static str, &'static str)>, Vec<(&'static str, &'static str)>);

fn pairs() -> Vec<(String, Pair)> {
    let mut out = Vec::new();
    let base = |extra: &[(&'static str, &'static str)]| {
        let mut v = vec![("speed_percent", "50")];
        v.extend_from_slice(extra);
        v
    };
    for (name, interp) in [("nearest", "1"), ("blend", "2"), ("weighted", "3")] {
        out.push((format!("{name}"), (base(&[("frame_interp", interp)]), base(&[("frame_interp", interp)]))));
    }
    out.push(("blur compensation".into(), (base(&[("blur_compensation", "1")]), base(&[("blur_compensation", "1")]))));
    out.push(("prep contrast".into(), (base(&[("prep", "2")]), base(&[("prep", "2")]))));
    out.push(("prep delinearize weighted".into(), (base(&[("prep", "3"), ("frame_interp", "3")]), base(&[("prep", "3"), ("frame_interp", "3")]))));
    // plugin 01's Sensitivity is plugin 03's Main_BG Sensitivity on the single layer path
    out.push(("sensitivity".into(), (base(&[("sensitivity", "20")]), base(&[("main_bg_sensitivity", "20")]))));
    out.push(("video range".into(), (base(&[("video_360", "1")]), base(&[("video_360", "1")]))));
    out.push(("lower quality".into(), (base(&[("vectors", "3")]), base(&[("vectors", "3")]))));
    out.push(("speed curve".into(), (base(&[]).into_iter().filter(|p| p.0 != "speed_percent").chain([("speed_percent", "40@0;120@4")]).collect(), vec![("speed_percent", "40@0;120@4")])));
    out.push(("frame number".into(), (vec![("time_remap_mode", "2"), ("frame_num", "0@0;5@7")], vec![("time_remap_mode", "2"), ("frame_num", "0@0;5@7")])));
    out.push(("warping spelled out".into(), (base(&[]), base(&[("warping", "1")]))));
    out.push(("segment cut".into(), (base(&[("src_mark_segments", "1@0;2@3")]), base(&[("src_mark_segments", "1@0;2@3")]))));
    out
}

#[test]
fn plugin_03_renders_the_same_bytes_as_plugin_01_for_the_same_settings() {
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let clip = DEFAULT.clip(depth, W, H, 6);
        for (name, (p1, p3)) in pairs() {
            for n in 0..9u64 {
                let a = render01(&clip, None, &p1, n, 9);
                let b = render03(&clip, None, &p3, n, 9);
                assert_eq!(a, b, "{depth:?} {name} frame {n}");
            }
        }
    }
}

#[test]
fn the_tracking_layer_feeds_the_solver_the_same_way() {
    let clip = DEFAULT.clip(Depth::U8, W, H, 6);
    let other = Motion { vx: -2.0, vy: 1.5 }.clip(Depth::U8, W, H, 6);
    for n in [1u64, 3, 5] {
        let a = render01(&clip, Some(&other), &[("speed_percent", "50")], n, 9);
        let b = render03(&clip, Some(&other), &[("speed_percent", "50")], n, 9);
        assert_eq!(a, b, "frame {n}");
        let without = render03(&clip, None, &[("speed_percent", "50")], n, 9);
        assert_ne!(b, without, "the tracking layer changes the flow, frame {n}");
    }
}

#[test]
fn main_bg_sensitivity_reaches_the_solver_and_the_other_sensitivities_do_not() {
    let clip = DEFAULT.clip(Depth::U8, W, H, 6);
    let at = |extra: &[(&str, &str)]| {
        let mut p = vec![("speed_percent", "50")];
        p.extend_from_slice(extra);
        render03(&clip, None, &p, 3, 9)
    };
    let default = at(&[]);
    assert_eq!(at(&[("main_bg_sensitivity", "70")]), default);
    assert_ne!(at(&[("main_bg_sensitivity", "5")]), default, "Main_BG Sensitivity is the solver's sensitivity");
    assert_eq!(at(&[("fg1_sensitivity", "5"), ("fg2_sensitivity", "5"), ("fg3_sensitivity", "5")]), default, "the FG sensitivities are for the multi layer path");
}

#[test]
fn fg_and_background_displays_are_the_output_on_the_vector_path_too() {
    let clip = DEFAULT.clip(Depth::U8, W, H, 6);
    let base = render03(&clip, None, &[("speed_percent", "50")], 3, 9);
    for display in ["3", "4", "5", "6"] {
        assert_eq!(render03(&clip, None, &[("speed_percent", "50"), ("display", display)], 3, 9), base, "display {display}");
    }
}

#[test]
fn the_result_is_close_to_the_analytic_frame_and_beats_the_plain_blend() {
    let (w, h) = (96, 64);
    for depth in [Depth::U8, Depth::F32] {
        let clip = DEFAULT.clip(depth, w, h, 8);
        let rig = Rig::new(clip.clone(), &[("speed_percent", "50"), ("frame_interp", "2")]).with_frames(10);
        let plain = Rig::new(clip, &[("speed_percent", "50"), ("frame_interp", "2"), ("vectors", "1")]).with_frames(10);
        for n in [1u64, 3, 5] {
            let truth = DEFAULT.frame_at(depth, w, h, n as f64 / 2.0);
            let (flow, _) = error(&rig.render(n).unwrap(), &truth, 10);
            let (base, _) = error(&plain.render(n).unwrap(), &truth, 10);
            assert!(flow < 0.0045, "{depth:?} frame {n}: error {flow}");
            assert!(flow * 4.5 < base, "{depth:?} frame {n}: flow {flow} against plain blend {base}");
        }
    }
}

#[test]
fn a_still_clip_stays_still() {
    let still = Motion { vx: 0.0, vy: 0.0 };
    let clip = still.clip(Depth::U8, W, H, 6);
    let truth = still.frame_at(Depth::U8, W, H, 0.0);
    let rig = Rig::new(clip, &[("speed_percent", "50")]).with_frames(8);
    for n in 1..8u64 {
        let (e, m) = error(&rig.render(n).unwrap(), &truth, MARGIN);
        assert!(e < 0.0005 && m < 0.004, "frame {n}: {e} {m}");
    }
}
