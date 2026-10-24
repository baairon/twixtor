//! `Effect::dependencies` of plugin 01: the checkout list, in issue order. Times are layer ticks of
//! a 30 fps composition (100 ticks per frame), so frame `p` of the source is at `100 * p` when the
//! input and composition rates are equal.

mod common;

use common::*;
use fx_core::{Checkout, Frame};

const A: Option<&str> = Some("alt_source_a");
const B: Option<&str> = Some("alt_source_b");

fn cp(layer: Option<&'static str>, time: i64) -> Checkout {
    Checkout { layer, time }
}

/// Twelve small 8 bpc frames; only the sizes matter for the dependencies.
fn clip() -> Vec<Frame> {
    (0..12).map(|_| Frame::new(2, 2, fx_core::Depth::U8)).collect()
}

#[test]
fn the_worked_dependency_example() {
    // Default parameters: Speed mode, Speed % = 100, Src Time Offset = 0, In FPS is Out FPS on (both 30), no
    // fields, no layer B, Vectors Best. Output frame 7: x = 7, simple, plan (1, 7) and (1, 8).
    let got = Rig::new(clip(), &[]).deps(7).unwrap();
    assert_eq!(
        got,
        vec![
            cp(None, 700),  // 1: layer 0 at cur
            cp(A, 700),     // 2: probe of layer 12
            cp(None, 700),  // 3: layer 12 is empty, probe of layer 0
            cp(B, 700),     // 4: probe of layer 13
            cp(None, -700), // 5 to 7: three extra checkouts at -cur: layer 0, src (= 0), layer 13
            cp(None, -700),
            cp(B, -700),
            cp(None, 700), // 8 on: one per plan entry through pos_to_time(p) = p * 100
            cp(None, 800),
        ]
    );
}

#[test]
fn every_output_frame_asks_for_its_own_source_frames() {
    let rig = Rig::new(clip(), &[]);
    for n in 0..12u64 {
        let got = rig.deps(n).unwrap();
        let t = 100 * n as i64;
        assert_eq!(got.len(), 9);
        assert_eq!(got[0], cp(None, t));
        assert_eq!(&got[7..], &[cp(None, t), cp(None, t + 100)], "frame {n}");
    }
}

#[test]
fn display_source_lists_the_probes_and_the_one_frame_at_the_offset() {
    // cur + trunc(Src Time Offset) * time_step = 700 + 2 * 100
    let rig = Rig::new(clip(), &[("display", "2"), ("src_time_offset", "2.9")]);
    assert_eq!(rig.deps(7).unwrap(), vec![cp(None, 700), cp(A, 700), cp(None, 700), cp(B, 700), cp(None, 900)]);
    // with Alt Source A set the frame comes from layer A and the second probe of the input is skipped
    let rig = Rig::new(clip(), &[("display", "2")]).with_layer("alt_source_a", clip());
    assert_eq!(rig.deps(7).unwrap(), vec![cp(None, 700), cp(A, 700), cp(B, 700), cp(A, 700)]);
}

#[test]
fn half_speed_asks_for_the_frames_around_the_position() {
    // output 7 at 50 percent is source position 3.5: frames 3 and 4 (no layer B, so one pair, even with Vectors on)
    let rig = Rig::new(clip(), &[("speed_percent", "50")]);
    let got = rig.deps(7).unwrap();
    assert_eq!(&got[..4], &[cp(None, 700), cp(A, 700), cp(None, 700), cp(B, 700)]);
    assert_eq!(&got[7..], &[cp(None, 300), cp(None, 400)]);
    assert_eq!(got.len(), 9);
}

#[test]
fn layer_b_doubles_the_plan_only_when_the_vector_path_would_run() {
    let b = |params: &[(&str, &str)]| Rig::new(clip(), params).with_layer("alt_source_b", clip());
    // between frames with Vectors on: the tracking layer first, then the source
    let got = b(&[("speed_percent", "50")]).deps(7).unwrap();
    assert_eq!(got.len(), 4 + 3 + 4);
    assert_eq!(&got[..4], &[cp(None, 700), cp(A, 700), cp(None, 700), cp(B, 700)]);
    assert_eq!(&got[4..7], &[cp(None, -700), cp(None, -700), cp(B, -700)]);
    assert_eq!(&got[7..], &[cp(B, 300), cp(B, 400), cp(None, 300), cp(None, 400)]);
    // an exact position needs no tracking frames
    assert_eq!(b(&[]).deps(7).unwrap().len(), 4 + 3 + 2);
    // No Vectors needs none either
    assert_eq!(b(&[("speed_percent", "50"), ("vectors", "1")]).deps(7).unwrap().len(), 4 + 3 + 2);
}

#[test]
fn alt_source_a_is_the_source_layer() {
    let rig = Rig::new(clip(), &[]).with_layer("alt_source_a", clip());
    assert_eq!(
        rig.deps(7).unwrap(),
        vec![cp(None, 700), cp(A, 700), cp(B, 700), cp(None, -700), cp(A, -700), cp(B, -700), cp(A, 700), cp(A, 800)]
    );
}

#[test]
fn input_fields_double_the_rate_and_fetch_every_entry_twice() {
    // Input: Fields = Upper Field First: in_fps = 2 * 30, out_fps = 30, so one source frame lasts 50 ticks and
    // output frame 7 is source position 14. Times: trunc(p * 50 + 50 * K) with K = 0.1, and the half frame positions p + 0.5.
    let rig = Rig::new(clip(), &[("input_fields", "2")]);
    let got = rig.deps(7).unwrap();
    assert_eq!(&got[7..], &[cp(None, 705), cp(None, 730), cp(None, 755), cp(None, 780)]);
}

#[test]
fn animated_speed_is_integrated_from_the_start_in_half_frame_steps() {
    // Speed 100 percent at frame 0 falling linearly to 50 at frame 2 (tick 200), then held at 50. Output frame 4
    // takes 8 half steps k = 1 ..= 8, sampled at ticks 50 * k: the speeds are 87.5, 75, 62.5, 50 and then 50 four
    // times (all exact in 16.16), and each adds speed / 2: x = (0.875 + 0.75 + 0.625 + 0.5 * 5) / 2 = 2.375.
    // So the pair is source frames 2 and 3, at ticks 200 and 300.
    let rig = Rig::new(clip(), &[("speed_percent", "100@0;50@2")]);
    let got = rig.deps(4).unwrap();
    assert_eq!(got.len(), 9, "{got:?}");
    assert_eq!(&got[7..], &[cp(None, 200), cp(None, 300)]);
}
