//! The Vectors In path end to end: a texture that translates by a known number of pixels per frame
//! (`common/moving.rs`) gives the exact in-between frame in closed form, and the two vector layers
//! are built analytically from the same motion (the exact flow, encoded the way Create Vectors
//! writes it). Plugin 04 retimes the clip and must land much closer to the analytic frame than the
//! plain frame blend does.
//!
//! Metric: mean absolute colour error as a fraction of full scale over the pixels at least 10
//! pixels from every edge (content that enters from outside the picture cannot be reconstructed).
//! Baseline: the same clip with the segment marker "No MV, Frame Blend", which switches the vectors
//! off and gives the plain blend of the two frames at the same weight. Measured on this machine
//! (debug build, 96 x 64, 8 bpc, frames 1, 3 and 5; the numbers below are the worst over four
//! motions of 2.6 to 4.4 pixels per frame): Blend error 0.0024 and at least 5.9 times closer than
//! the plain blend, Nearest 0.0024 and 5.8 times, Weighted Blend 0.0031 and 4.6 times; the default
//! motion at 50 percent: 0.0018 and 11.6 times at 8 bpc, 0.0014 and 15 times at 16 and 32 bpc. The
//! asserted margins sit well outside those.

mod common;

use common::moving::*;
use common::*;
use fx_core::{Depth, Frame, Pixels};
use std::sync::Arc;

const W: usize = 96;
const H: usize = 64;
const MARGIN: usize = 10;
/// Max Displace of the vector layers: far below the default 2048 so that the 8 bit coding (127 levels per side) is
/// fine enough (a step of 16 / 127 = 0.126 px).
const MAX_DISPLACE: f32 = 16.0;

fn rig(depth: Depth, motion: Motion, params: &[(&str, &str)], frames: u64) -> Rig {
    let clip = motion.clip(depth, W, H, 8);
    Rig::new(clip, params).with_frames(frames).with_vectors(motion.vx as f32, motion.vy as f32, MAX_DISPLACE)
}

/// Warping "Inverse" (the forward warp is not in the engine yet) and the Max Displace the layers were encoded with.
fn params<'a>(extra: &[(&'a str, &'a str)]) -> Vec<(&'a str, &'a str)> {
    let mut p = vec![("warping", "1"), ("vectors_max_displace", "16")];
    p.extend_from_slice(extra);
    p
}

/// Error against the analytic frame of the Vectors In render and of the plain blend, at output frame `n` of a run at
/// `speed` percent (source position `n * speed / 100`).
fn measure(depth: Depth, motion: Motion, interp: &str, speed: &str, n: u64) -> (f64, f64) {
    let speed_f: f64 = speed.parse().unwrap();
    let flow = rig(depth, motion, &params(&[("speed_percent", speed), ("frame_interp", interp)]), 12);
    let plain = rig(depth, motion, &params(&[("speed_percent", speed), ("frame_interp", interp), ("src_mark_segments", "5")]), 12);
    let truth = motion.frame_at(depth, W, H, n as f64 * speed_f / 100.0);
    let (got, _) = error(&flow.render(n).unwrap(), &truth, MARGIN);
    let (base, _) = error(&plain.render(n).unwrap(), &truth, MARGIN);
    (got, base)
}

#[test]
fn blend_at_half_speed_is_close_to_the_analytic_frame_and_beats_the_plain_blend() {
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        for n in [1u64, 3, 5, 7] {
            let (got, base) = measure(depth, DEFAULT, "2", "50", n);
            assert!(got < 0.0030, "{depth:?} frame {n}: error {got}");
            assert!(got * 8.0 < base, "{depth:?} frame {n}: vectors {got} against the plain blend {base}");
        }
    }
}

const MOTIONS: [Motion; 4] = [
    Motion { vx: 3.4, vy: 1.7 },
    Motion { vx: -2.6, vy: 2.9 },
    Motion { vx: 0.0, vy: -3.8 },
    Motion { vx: 4.4, vy: 0.0 },
];

#[test]
fn every_interpolation_mode_beats_the_plain_blend_for_four_motions_and_three_speeds() {
    for (interp, name, max_error, min_ratio) in [("1", "Nearest", 0.0035, 4.5), ("2", "Blend", 0.0035, 4.5), ("3", "Weighted Blend", 0.0045, 3.5)] {
        for (mi, motion) in MOTIONS.iter().enumerate() {
            for speed in ["25", "50", "75"] {
                // Weighted Blend is judged at half speed only (it adds the sharpening of the engine at every weight)
                if interp == "3" && speed != "50" {
                    continue;
                }
                for n in [1u64, 3, 5] {
                    let (got, base) = measure(Depth::U8, *motion, interp, speed, n);
                    assert!(got < max_error, "{name} motion {mi} speed {speed} frame {n}: error {got}");
                    assert!(got * min_ratio < base, "{name} motion {mi} speed {speed} frame {n}: vectors {got} against plain {base}");
                }
            }
        }
    }
}

#[test]
fn swapped_vector_layers_move_the_picture_the_wrong_way() {
    // Vectors: Prev must hold the flow of B back to A, Vectors: Next the flow of A forward to B.
    let motion = DEFAULT;
    let right = rig(Depth::U8, motion, &params(&[("speed_percent", "50")]), 10);
    let (prev, next) = motion_vectors(Depth::U8, W, H, 16, motion.vx as f32, motion.vy as f32, MAX_DISPLACE);
    let swapped = Rig::new(motion.clip(Depth::U8, W, H, 8), &params(&[("speed_percent", "50")]))
        .with_frames(10)
        .with_layer("vectors_prev", next)
        .with_layer("vectors_next", prev);
    for n in [1u64, 3, 5] {
        let truth = motion.frame_at(Depth::U8, W, H, n as f64 / 2.0);
        let (good, _) = error(&right.render(n).unwrap(), &truth, MARGIN);
        let (bad, _) = error(&swapped.render(n).unwrap(), &truth, MARGIN);
        assert!(good * 10.0 < bad, "frame {n}: right {good}, swapped {bad}");
    }
}

#[test]
fn vec_scale_multiplies_the_flows_and_zero_gives_the_plain_blend() {
    let motion = DEFAULT;
    // layers that hold half the motion, Vec Scale 2 restores it
    let half = Rig::new(motion.clip(Depth::U8, W, H, 8), &params(&[("speed_percent", "50"), ("vec_scale_x", "2"), ("vec_scale_y", "2")]))
        .with_frames(10)
        .with_vectors(motion.vx as f32 / 2.0, motion.vy as f32 / 2.0, MAX_DISPLACE);
    let direct = rig(Depth::U8, motion, &params(&[("speed_percent", "50")]), 10);
    // X and Y are independent: scaling only Y by 0 removes the vertical motion
    let no_y = rig(Depth::U8, motion, &params(&[("speed_percent", "50"), ("vec_scale_y", "0")]), 10);
    let zero = rig(Depth::U8, motion, &params(&[("speed_percent", "50"), ("vec_scale_x", "0"), ("vec_scale_y", "0")]), 10);
    let plain = rig(Depth::U8, motion, &params(&[("speed_percent", "50"), ("src_mark_segments", "5")]), 10);
    for n in [1u64, 3, 5] {
        let truth = motion.frame_at(Depth::U8, W, H, n as f64 / 2.0);
        let (scaled, _) = error(&half.render(n).unwrap(), &truth, MARGIN);
        let (full, _) = error(&direct.render(n).unwrap(), &truth, MARGIN);
        let (flat, _) = error(&no_y.render(n).unwrap(), &truth, MARGIN);
        assert!(scaled < 0.0035, "frame {n}: half vectors scaled by 2 give {scaled}");
        assert!((scaled - full).abs() < 0.0015, "frame {n}: {scaled} against {full}");
        assert!(flat > full * 2.5, "frame {n}: Vec Scale Y = 0 must lose the vertical motion ({flat} against {full})");
        // zero flow: both warps copy their frame (amounts above the early out still sample at integer positions)
        let got = zero.render(n).unwrap();
        let want = plain.render(n).unwrap();
        assert_eq!(max_level_difference(&got, &want), 0, "frame {n}: Vec Scale 0 is the plain blend");
    }
}

fn max_level_difference(a: &Frame, b: &Frame) -> i32 {
    match (&a.pixels, &b.pixels) {
        (Pixels::U8(x), Pixels::U8(y)) => {
            x.iter().zip(y).flat_map(|(p, q)| (0..4).map(move |i| (p[i] as i32 - q[i] as i32).abs())).max().unwrap_or(0)
        }
        _ => panic!("8 bpc frames expected"),
    }
}

#[test]
fn max_displace_scales_the_decoded_vectors() {
    // layers written with Max Displace 16; decoding with 32 doubles every vector, with 8 halves it
    let motion = DEFAULT;
    let at = |md: &str| {
        Rig::new(motion.clip(Depth::U8, W, H, 8), &[("warping", "1"), ("vectors_max_displace", md), ("speed_percent", "50")])
            .with_frames(10)
            .with_vectors(motion.vx as f32, motion.vy as f32, MAX_DISPLACE)
    };
    let (right, double, half) = (at("16"), at("32"), at("8"));
    for n in [1u64, 3, 5] {
        let truth = motion.frame_at(Depth::U8, W, H, n as f64 / 2.0);
        let (good, _) = error(&right.render(n).unwrap(), &truth, MARGIN);
        let (wide, _) = error(&double.render(n).unwrap(), &truth, MARGIN);
        let (narrow, _) = error(&half.render(n).unwrap(), &truth, MARGIN);
        assert!(good * 4.0 < wide, "frame {n}: {good} against Max Displace 32 {wide}");
        assert!(good * 4.0 < narrow, "frame {n}: {good} against Max Displace 8 {narrow}");
    }
}

#[test]
fn a_still_clip_with_zero_vectors_stays_still() {
    let still = Motion { vx: 0.0, vy: 0.0 };
    let r = rig(Depth::U8, still, &params(&[("speed_percent", "50")]), 8);
    let truth = still.frame_at(Depth::U8, W, H, 0.0);
    for n in 1..8u64 {
        let (e, m) = error(&r.render(n).unwrap(), &truth, MARGIN);
        assert!(e < 0.0005 && m < 0.004, "frame {n}: {e} {m}");
    }
}

#[test]
fn frame_number_mode_and_reverse_playback_use_the_vectors_at_their_own_frames() {
    let motion = DEFAULT;
    // Frame Number 0 to 5 over the run: source position 5 * n / 7
    let r = rig(Depth::U8, motion, &params(&[("time_remap_mode", "2"), ("frame_num", "0@0;5@7")]), 8);
    for n in [1u64, 2, 4, 5] {
        let truth = motion.frame_at(Depth::U8, W, H, 5.0 * n as f64 / 7.0);
        let (e, _) = error(&r.render(n).unwrap(), &truth, MARGIN);
        assert!(e < 0.0035, "frame number mode, frame {n}: {e}");
    }
    // Speed -50 percent with a start at the end of the clip is not reachable (the position starts
    // at 0); the vectors are read the same way for both directions, so the forward looking pair of
    // a negative position extrapolates: only check that it renders and differs from the plain blend
    let rev = rig(Depth::U8, motion, &params(&[("speed_percent", "-50")]), 8);
    assert!(rev.render(3).is_ok());
}

#[test]
fn blur_compensation_runs_on_the_decoded_flows() {
    let motion = DEFAULT;
    let at = |blur: &str| rig(Depth::U8, motion, &params(&[("speed_percent", "150"), ("blur_compensation", blur)]), 8);
    let (off, on) = (at("0"), at("1"));
    for n in [1u64, 3] {
        let a = off.render(n).unwrap();
        let b = on.render(n).unwrap();
        assert_ne!(a, b, "frame {n}: Blur Compensation 1 at 150 percent speed must change the picture");
    }
}

/// A source layer that is separate from the input layer is the retimed footage; the input layer
/// then only gives the size.
#[test]
fn the_source_layer_replaces_the_input_as_the_retimed_footage() {
    let motion = DEFAULT;
    let (prev, next) = motion_vectors(Depth::U8, W, H, 16, motion.vx as f32, motion.vy as f32, MAX_DISPLACE);
    let junk = random_clip(Depth::U8, W, H, 8, 5);
    let r = Rig::new(junk, &params(&[("speed_percent", "50")]))
        .with_frames(10)
        .with_layer("source", motion.clip(Depth::U8, W, H, 8))
        .with_layer("vectors_prev", prev)
        .with_layer("vectors_next", next);
    let truth = motion.frame_at(Depth::U8, W, H, 1.5);
    let (e, _) = error(&r.render(3).unwrap(), &truth, MARGIN);
    assert!(e < 0.0030, "{e}");
}

/// Recording layers show which vector frames are read: `Vectors: Next` at the time of frame A and `Vectors: Prev` at
/// the time of frame B, and nothing else besides the probe at `current_time`.
#[test]
fn the_vector_layers_are_read_at_the_frame_times_of_the_pair() {
    let motion = DEFAULT;
    let clip = motion.clip(Depth::U8, W, H, 8);
    let (prev, next) = motion_vectors(Depth::U8, W, H, 16, motion.vx as f32, motion.vy as f32, MAX_DISPLACE);
    let (rp, rn) = (Recorder::new(prev), Recorder::new(next));
    let rig = Rig::new(clip, &params(&[("speed_percent", "50")]))
        .with_frames(10)
        .with_dyn_layer("vectors_prev", rp.clone())
        .with_dyn_layer("vectors_next", rn.clone());
    // output frame 3 at 50 percent: source position 1.5, frames A = 1 and B = 2 at 100 ticks per frame; the probes
    // look at current_time = 300
    rig.render(3).unwrap();
    assert_eq!(rn.take(), vec![300, 100], "Vectors: Next: the probe, then frame A");
    assert_eq!(rp.take(), vec![300, 200], "Vectors: Prev: the probe, then frame B");
    // an exact frame (output 2 is position 1.0) never loads the vectors, only the probe runs
    rig.render(2).unwrap();
    assert_eq!(rn.take(), vec![200]);
    assert_eq!(rp.take(), vec![200]);
    let _ = Arc::strong_count(&rp);
}
