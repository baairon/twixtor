//! The frames that never reach the flow code: exact positions, the segment markers that switch the vectors off, the
//! Source display and positions past the clip. Plugin 04 and plugin 01 share the engine and, on these frames, the
//! front-end arithmetic (equal rates, no Src Time Offset), so for the same settings the two effects must produce the
//! same bytes at 8, 16 and 32 bpc on clips with every alpha class. Plugin 01 runs with Vectors = No Vectors and its
//! own ids; plugin 04 runs with vector layers and Warping Inverse, which its exact frames never ask for.

mod common;

use common::*;
use fx_core::{Depth, Effect, Env, Frame, LayerSet, MemLayer, ParamCurves};
use retime::Retime;

const W: usize = 9;
const H: usize = 7;

fn render01(clip: &[Frame], params: &[(&str, &str)], n: u64, frames: u64) -> Frame {
    let mut curves = ParamCurves::new(Retime.params());
    for (id, text) in params {
        curves.parse_cli(id, text, time()).unwrap();
    }
    let input = MemLayer::new(time(), clip.to_vec()).unwrap();
    let layers = LayerSet::new();
    let env = Env { params: &curves, input: &input, layers: &layers };
    let (w, h, d) = (clip[0].width, clip[0].height, clip[0].depth());
    let cx = fx_core::TemporalContext::new(w, h, d, time(), n, frames);
    let mut out = Frame::new(w, h, d);
    Retime.render_temporal(&cx, &env, &mut out).unwrap();
    out
}

fn render04(clip: &[Frame], params: &[(&str, &str)], n: u64, frames: u64) -> Frame {
    let depth = clip[0].depth();
    let (prev, next) = motion_vectors(depth, W, H, 24, 1.0, 0.5, 16.0);
    let mut p = vec![("warping", "1"), ("vectors_max_displace", "16")];
    p.extend_from_slice(params);
    Rig::new(clip.to_vec(), &p).with_frames(frames).with_layer("vectors_prev", prev).with_layer("vectors_next", next).render(n).unwrap()
}

/// `(plugin 01 parameters, plugin 04 parameters)`.
type Pair = (Vec<(&'static str, &'static str)>, Vec<(&'static str, &'static str)>);

fn pairs() -> Vec<(String, Pair)> {
    let mut out = Vec::new();
    let nv = ("vectors", "1");
    for (name, interp) in [("nearest", "1"), ("blend", "2"), ("weighted", "3")] {
        for speed in ["100", "200", "-100"] {
            out.push((
                format!("{name} at {speed} percent"),
                (vec![nv, ("speed_percent", speed), ("frame_interp", interp)], vec![("speed_percent", speed), ("frame_interp", interp)]),
            ));
        }
        // the segment markers that switch the vectors off at half speed (No MV, Use Nearest / Frame Blend)
        for marker in ["4", "5"] {
            out.push((
                format!("{name} half speed, marker {marker}"),
                (
                    vec![nv, ("speed_percent", "50"), ("frame_interp", interp), ("src_mark_segments", marker)],
                    vec![("speed_percent", "50"), ("frame_interp", interp), ("src_mark_segments", marker)],
                ),
            ));
        }
    }
    out.push((
        "marker keys".into(),
        (vec![nv, ("speed_percent", "50"), ("src_mark_segments", "4@0;5@4")], vec![("speed_percent", "50"), ("src_mark_segments", "4@0;5@4")]),
    ));
    out.push((
        "frame number between frames, Frame Blend marker".into(),
        (
            vec![nv, ("time_remap_mode", "2"), ("frame_num", "0@0;5@7"), ("src_mark_segments", "5")],
            vec![("time_remap_mode", "2"), ("frame_num", "0@0;5@7"), ("src_mark_segments", "5")],
        ),
    ));
    out.push((
        "frame number, whole frames".into(),
        (vec![nv, ("time_remap_mode", "2"), ("frame_num", "2")], vec![("time_remap_mode", "2"), ("frame_num", "2")]),
    ));
    out.push(("speed curve, exact keys".into(), (vec![nv, ("speed_percent", "100@0;100@5")], vec![("speed_percent", "100@0;100@5")])));
    out.push(("source display".into(), (vec![nv, ("display", "2")], vec![("display", "2")])));
    out.push(("source display half speed".into(), (vec![nv, ("display", "2"), ("speed_percent", "50")], vec![("display", "2"), ("speed_percent", "50")])));
    // FG and Background displays take the plan like the output
    out.push(("background display".into(), (vec![nv, ("speed_percent", "100")], vec![("display", "6"), ("speed_percent", "100")])));
    out
}

#[test]
fn plugin_04_renders_the_same_bytes_as_plugin_01_where_no_flow_is_needed() {
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let clip = random_clip(depth, W, H, 6, 0x5eed);
        for (name, (p1, p4)) in pairs() {
            for n in 0..9u64 {
                let a = render01(&clip, &p1, n, 9);
                let b = render04(&clip, &p4, n, 9);
                assert_eq!(a, b, "{depth:?} {name} frame {n}");
            }
        }
    }
}

#[test]
fn the_exact_frames_of_a_full_speed_run_are_the_clip_through_the_engine_round_trip() {
    // 8 bpc, opaque frames come back bit for bit (the premultiply and unpremultiply are the identity at alpha 255)
    let clip: Vec<Frame> = (0..4).map(|i| f8(2, 2, &[[10 * i + 1, 2, 3, 255], [4, 5 * i + 5, 6, 255], [7, 8, 9 + i, 255], [250, 251, 252, 255]])).collect();
    let (prev, next) = motion_vectors(Depth::U8, 2, 2, 12, 1.0, 0.0, 16.0);
    let rig = Rig::new(clip.clone(), &[("vectors_max_displace", "16")]).with_frames(6).with_layer("vectors_prev", prev).with_layer("vectors_next", next);
    for n in 0..4u64 {
        assert_eq!(rig.render(n).unwrap(), clip[n as usize], "frame {n}");
    }
}

#[test]
fn past_the_clip_the_last_frame_fades_into_transparency_like_plugin_01() {
    let clip = random_clip(Depth::U8, W, H, 4, 77);
    for n in 0..10u64 {
        let a = render01(&clip, &[("vectors", "1"), ("speed_percent", "100")], n, 10);
        let b = render04(&clip, &[("speed_percent", "100")], n, 10);
        assert_eq!(a, b, "frame {n}");
    }
}
