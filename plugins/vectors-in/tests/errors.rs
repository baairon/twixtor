//! The error fills and the probes of plugin 04.
//!
//! A failing probe writes the fill its error stands for and the render succeeds, so a failure is
//! visible: red at half alpha for a size mismatch, the plugin's blue (alpha 0 at 16 and 32 bpc) for
//! a missing source. The first failing probe decides, in this order: Source, Vectors: Prev,
//! Vectors: Next, "neither vector layer", FG mattes, then the rate check.

mod common;

use common::moving::*;
use common::*;
use fx_core::{Depth, Effect, Frame, Rational};
use vectors_in::VectorsIn;

const W: usize = 32;
const H: usize = 24;
const MD: f32 = 16.0;

fn clip(depth: Depth, w: usize, h: usize) -> Vec<Frame> {
    DEFAULT.clip(depth, w, h, 6)
}

fn vectors(depth: Depth, w: usize, h: usize) -> (Vec<Frame>, Vec<Frame>) {
    motion_vectors(depth, w, h, 12, 3.4, 1.7, MD)
}

fn good(depth: Depth, extra: &[(&str, &str)]) -> Rig {
    let (prev, next) = vectors(depth, W, H);
    let mut p = vec![("warping", "1"), ("vectors_max_displace", "16"), ("speed_percent", "50")];
    p.extend_from_slice(extra);
    Rig::new(clip(depth, W, H), &p).with_frames(8).with_layer("vectors_prev", prev).with_layer("vectors_next", next)
}

#[test]
fn a_good_rig_renders_a_picture() {
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let out = good(depth, &[]).render(1).unwrap();
        assert_ne!(out, fill(depth, true, W, H));
        assert_ne!(out, fill(depth, false, W, H));
    }
}

#[test]
fn no_vector_layer_at_all_is_the_size_mismatch_fill() {
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let r = Rig::new(clip(depth, W, H), &[("warping", "1")]).with_frames(8);
        for n in [0u64, 1, 2] {
            assert_eq!(r.render(n).unwrap(), fill(depth, true, W, H), "{depth:?} frame {n}");
        }
        // also with the Source display: the probes come first, the display is read after them
        let r = Rig::new(clip(depth, W, H), &[("display", "2")]).with_frames(8);
        assert_eq!(r.render(1).unwrap(), fill(depth, true, W, H), "{depth:?} source display");
    }
}

#[test]
fn one_vector_layer_is_enough_for_the_probes() {
    let (prev, _) = vectors(Depth::U8, W, H);
    // exact frames never need the other flow
    let r = Rig::new(clip(Depth::U8, W, H), &[("warping", "1"), ("vectors_max_displace", "16")]).with_frames(8).with_layer("vectors_prev", prev);
    for n in 0..6 {
        let out = r.render(n).unwrap();
        assert_ne!(out, fill(Depth::U8, true, W, H));
    }
}

#[test]
fn a_vector_layer_of_another_size_is_the_size_mismatch_fill() {
    for which in ["vectors_prev", "vectors_next"] {
        let (prev, next) = vectors(Depth::U8, W, H);
        let (wrong_p, wrong_n) = vectors(Depth::U8, W + 2, H);
        let mut r = Rig::new(clip(Depth::U8, W, H), &[("warping", "1")]).with_frames(8);
        if which == "vectors_prev" {
            r = r.with_layer("vectors_prev", wrong_p).with_layer("vectors_next", next);
        } else {
            r = r.with_layer("vectors_prev", prev).with_layer("vectors_next", wrong_n);
        }
        assert_eq!(r.render(1).unwrap(), fill(Depth::U8, true, W, H), "{which}");
        assert_eq!(r.render(2).unwrap(), fill(Depth::U8, true, W, H), "{which} exact frame");
    }
}

#[test]
fn a_fg_matte_of_another_size_is_the_size_mismatch_fill() {
    let r = good(Depth::U8, &[]).with_layer("fg1_matte", clip(Depth::U8, W + 1, H));
    assert_eq!(r.render(1).unwrap(), fill(Depth::U8, true, W, H));
    // FG2 only: ignored, because the scan stops at the empty FG1
    let r = good(Depth::U8, &[]).with_layer("fg2_matte", clip(Depth::U8, W + 1, H));
    assert_ne!(r.render(1).unwrap(), fill(Depth::U8, true, W, H));
}

#[test]
fn a_source_layer_of_another_size_is_the_size_mismatch_fill() {
    let r = good(Depth::U8, &[]).with_layer("source", clip(Depth::U8, W, H + 4));
    assert_eq!(r.render(1).unwrap(), fill(Depth::U8, true, W, H));
}

#[test]
fn an_input_without_pixels_is_the_missing_source_fill() {
    // the Source probe finds size 0 first (blue), whatever the vector layers say; red would need a
    // size other than 0
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let (prev, next) = vectors(depth, W, H);
        let mut r = Rig::new(vec![Frame::new(0, 0, depth)], &[("warping", "1")])
            .with_frames(8)
            .with_layer("vectors_prev", prev)
            .with_layer("vectors_next", next);
        r.width = W;
        r.height = H;
        assert_eq!(r.render(1).unwrap(), fill(depth, false, W, H), "{depth:?}");
        assert_eq!(r.render(0).unwrap(), fill(depth, false, W, H), "{depth:?} exact frame");
    }
}

#[test]
fn a_bad_rate_is_the_size_mismatch_fill() {
    let r = good(Depth::U8, &[]);
    let mut cx = r.cx(1);
    cx.fps = Rational::new(1, 1000).unwrap(); // out = 0.001 <= 0.01
    assert_eq!(r.render_cx(&cx).unwrap(), fill(Depth::U8, true, W, H));
    // out == 0.01 exactly passes the rate check (it fails only for out < 0.01)
    cx.fps = Rational::new(1, 100).unwrap();
    assert_ne!(r.render_cx(&cx).unwrap(), fill(Depth::U8, true, W, H));
}

#[test]
fn a_vector_layer_that_is_empty_at_current_time_is_absent_for_the_whole_frame() {
    // Vectors: Next has two frames (ticks 0..200), Vectors: Prev is long: at output frame 5 (t0 = 500) Next is absent
    let (prev, next) = vectors(Depth::U8, W, H);
    let short: Vec<Frame> = next.into_iter().take(2).collect();
    let r = Rig::new(clip(Depth::U8, W, H), &[("warping", "1"), ("vectors_max_displace", "16"), ("speed_percent", "100")])
        .with_frames(8)
        .with_layer("vectors_prev", prev)
        .with_layer("vectors_next", short);
    // exact frames: fine with one layer
    assert_ne!(r.render(4).unwrap(), fill(Depth::U8, true, W, H));
    // and the dependencies still list both layers
    let deps = r.deps(4).unwrap();
    assert!(deps.iter().any(|c| c.layer == Some("vectors_next") && c.time == 400));
}

#[test]
fn a_vector_layer_without_a_frame_at_the_fetch_time_gives_a_zero_flow() {
    // the layers have 4 frames (ticks 0..400) and are present at t0 = 300; at 150 percent speed
    // output frame 3 is position 4.5, frames 4 and 5 at ticks 400 and 500: no frame there, so every
    // pixel is a hole and the flow is zero, which makes the result the plain blend
    let (prev, next) = vectors(Depth::U8, W, H);
    let short = |v: &Vec<Frame>| v.iter().take(4).cloned().collect::<Vec<_>>();
    let make = |extra: &[(&str, &str)]| {
        let mut p = vec![("warping", "1"), ("vectors_max_displace", "16"), ("speed_percent", "150")];
        p.extend_from_slice(extra);
        Rig::new(clip(Depth::U8, W, H), &p).with_frames(8).with_layer("vectors_prev", short(&prev)).with_layer("vectors_next", short(&next))
    };
    let got = make(&[]).render(3).unwrap();
    let plain = make(&[("src_mark_segments", "5")]).render(3).unwrap();
    let diff = |a: &Frame, b: &Frame| match (&a.pixels, &b.pixels) {
        (fx_core::Pixels::U8(x), fx_core::Pixels::U8(y)) => {
            x.iter().zip(y).flat_map(|(p, q)| (0..4).map(move |i| (p[i] as i32 - q[i] as i32).abs())).max().unwrap()
        }
        _ => unreachable!(),
    };
    assert_eq!(diff(&got, &plain), 0);
}

#[test]
fn the_output_frame_must_match_the_context() {
    let r = good(Depth::U8, &[]);
    let mut out = Frame::new(W + 1, H, Depth::U8);
    let e = VectorsIn.render_temporal(&r.cx(1), &r.env(), &mut out).unwrap_err();
    assert!(e.to_string().contains("the output frame is"), "{e}");
}

#[test]
fn vector_layers_shorter_than_the_output_run_are_the_red_fill_from_their_end_on() {
    // presence is decided at current_time: at half speed the output is twice as long as the source,
    // and layers that end with the source are absent for every output frame after that, whatever
    // the source holds
    let (prev, next) = vectors(Depth::U8, W, H);
    let (prev, next): (Vec<Frame>, Vec<Frame>) = (prev.into_iter().take(6).collect(), next.into_iter().take(6).collect());
    let r = Rig::new(clip(Depth::U8, W, H), &[("warping", "1"), ("vectors_max_displace", "16"), ("speed_percent", "50")])
        .with_frames(12)
        .with_layer("vectors_prev", prev)
        .with_layer("vectors_next", next);
    for n in 0..6 {
        assert_ne!(r.render(n).unwrap(), fill(Depth::U8, true, W, H), "frame {n}");
    }
    for n in 6..12 {
        assert_eq!(r.render(n).unwrap(), fill(Depth::U8, true, W, H), "frame {n}");
    }
}
