//! Ground truth of the Create Vectors output: a texture that translates by a known number of pixels
//! per frame (`common/moving.rs`, 3.4 px in x and 1.7 px in y per frame) must come out as vectors
//! that decode (with `vectors::decode_vectors` at the same Max Displace, as plugins 03 and 04 read
//! them) to the known motion.
//!
//! Sign convention: the flow is `X(p) ~ Y(p + flow(p))` with `X` the frame of the pair that comes
//! first (`a`) and `Y` the second (`b`). The content moves by `+v` per frame, so frame `k` shows
//! `pattern(p - v k)` and `Track Frame = Previous` (`Y` is the frame before) gives `flow = -v`,
//! `Next` gives `+v`. The Debug swap of the pair is covered by the dependencies tests (Debug itself
//! is refused).
//!
//! Max Displace is small (16 px) on purpose: the 8 bpc code has 127 steps per Max Displace, so the
//! default 2048 px would quantise a 3.4 px vector to zero
//! (`the_default_max_displace_cannot_carry_small_motion_at_8_bpc` pins that).
//!
//! Stated error at 96 x 64 (measured: mean 0.06 to 0.09 px, worst pixel 0.14 to 0.34 px, mean
//! vector within 0.05 px of the truth, all depths, both Vector Resolutions): mean error below 0.12
//! px, worst pixel below 0.5 px, mean vector within 0.1 px of the truth, over the pixels at least
//! 12 px from the edge.

mod common;

use common::moving::*;
use common::*;
use fx_core::{Depth, Frame, Pixels};
use retime_engine::create_vectors::{create_vectors, frame_index, frame_pair, frame_time, CreateVectorsSettings};
use retime_engine::image::Image;

const W: usize = 96;
const H: usize = 64;
const MARGIN: usize = 12;
const MAX: f32 = 16.0;
const MEAN_LIMIT: f64 = 0.12;
const WORST_LIMIT: f64 = 0.5;
const BIAS_LIMIT: f64 = 0.1;

fn rig(depth: Depth, params: &[(&str, &str)]) -> Rig {
    let mut p = vec![("max_displace", "16")];
    p.extend_from_slice(params);
    Rig::new(DEFAULT.clip(depth, W, H, 8), &p).with_frames(8)
}

/// `(mean error, max error, mean vector)` in pixels of the vectors in the output of frame `n`.
fn measure(depth: Depth, params: &[(&str, &str)], n: u64, truth: (f64, f64)) -> (f64, f64, (f64, f64)) {
    let out = rig(depth, params).render(n).unwrap();
    flow_error(&decode(&out, MAX), truth, MARGIN)
}

fn check(what: &str, got: (f64, f64, (f64, f64)), truth: (f64, f64)) {
    check_with(what, got, truth, (MEAN_LIMIT, WORST_LIMIT, BIAS_LIMIT));
}

fn check_with(what: &str, got: (f64, f64, (f64, f64)), truth: (f64, f64), limits: (f64, f64, f64)) {
    let (mean, max, avg) = got;
    eprintln!("{what}: mean {mean:.4} max {max:.4} avg ({:.3}, {:.3})", avg.0, avg.1);
    assert!(mean < limits.0, "{what}: mean error {mean}");
    assert!(max < limits.1, "{what}: worst error {max}");
    assert!((avg.0 - truth.0).abs() < limits.2 && (avg.1 - truth.1).abs() < limits.2, "{what}: mean vector {avg:?} against {truth:?}");
}

#[test]
fn track_previous_gives_minus_the_motion_at_every_depth() {
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        for res in ["1", "2"] {
            let truth = (-3.4, -1.7);
            let got = measure(depth, &[("vector_resolution", res), ("track_frame", "1")], 3, truth);
            check(&format!("previous {depth:?} res {res}"), got, truth);
        }
    }
}

#[test]
fn track_next_gives_plus_the_motion_at_every_depth() {
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        for res in ["1", "2"] {
            let truth = (3.4, 1.7);
            let got = measure(depth, &[("vector_resolution", res), ("track_frame", "2")], 3, truth);
            check(&format!("next {depth:?} res {res}"), got, truth);
        }
    }
}

#[test]
fn another_motion_and_another_frame_follow_the_same_law() {
    // a different velocity (negative x, fractional y) and a different output frame
    let m = Motion { vx: -2.6, vy: 0.8 };
    for depth in [Depth::U8, Depth::F32] {
        let rig = Rig::new(m.clip(depth, W, H, 8), &[("max_displace", "16")]).with_frames(8);
        let out = rig.render(5).unwrap();
        // measured: mean 0.11 px, worst 0.34 px, mean vector 0.11 px short of the truth (a slower texture is harder)
        let got = flow_error(&decode(&out, MAX), (2.6, -0.8), MARGIN);
        check_with(&format!("previous {depth:?} other motion"), got, (2.6, -0.8), (0.18, 0.5, 0.15));
    }
}

#[test]
fn the_prep_modes_and_track_rgb_a_keep_the_motion() {
    // Prep Contrast/Edge Enhance, Delinearize and Track RGB+A change the luma the solver sees, not the motion
    for params in [[("prep", "2"), ("track_rgb_a", "0")], [("prep", "3"), ("track_rgb_a", "0")], [("prep", "1"), ("track_rgb_a", "1")]] {
        // measured: Contrast/Edge Enhance mean 0.15 px, worst 0.40 px; the others as without Prep
        let got = measure(Depth::U8, &params, 3, (-3.4, -1.7));
        check_with(&format!("{params:?}"), got, (-3.4, -1.7), (0.25, 0.6, 0.15));
    }
}

#[test]
fn the_vectors_do_not_depend_on_the_max_displace_beyond_quantisation() {
    // the same flow encoded at three scales decodes to the same motion: scale and decode cancel
    let truth = (-3.4, -1.7);
    let mut avgs = Vec::new();
    for max in ["8", "16", "64"] {
        let out = rig(Depth::F32, &[("max_displace", max)]).render(3).unwrap();
        let flow = decode(&out, max.parse::<f32>().unwrap());
        let (mean, worst, avg) = flow_error(&flow, truth, MARGIN);
        assert!(mean < MEAN_LIMIT && worst < WORST_LIMIT, "max displace {max}: {mean} {worst}");
        avgs.push(avg);
    }
    for a in &avgs {
        assert!((a.0 - avgs[0].0).abs() < 1e-3 && (a.1 - avgs[0].1).abs() < 1e-3, "{avgs:?}");
    }
}

#[test]
fn the_default_max_displace_cannot_carry_small_motion_at_8_bpc() {
    // Max Displace 2048 (the default) at 8 bpc: one code step is 2048 / 127 = 16.1 px, so 3.4 px encodes as the zero code
    let out = Rig::new(DEFAULT.clip(Depth::U8, W, H, 8), &[]).with_frames(8).render(3).unwrap();
    let Pixels::U8(p) = &out.pixels else { panic!() };
    assert!(p.iter().all(|px| px[0] == 127 && px[1] == 127), "every vector rounds to the zero code");
    let flow = decode(&out, 2048.0);
    assert!(flow.data.iter().all(|v| *v == 0.0));
    // 16 bpc has 32767 steps per Max Displace: 0.06 px, so the motion survives
    let out = Rig::new(DEFAULT.clip(Depth::U16, W, H, 8), &[]).with_frames(8).render(3).unwrap();
    let (mean, _, avg) = flow_error(&decode(&out, 2048.0), (-3.4, -1.7), MARGIN);
    assert!(mean < 0.2 && avg.0 < -3.0, "{mean} {avg:?}");
}

#[test]
fn the_output_is_an_opaque_image_with_zero_blue_at_the_output_size() {
    for (depth, white) in [(Depth::U8, 255.0), (Depth::U16, 65535.0), (Depth::F32, 1.0)] {
        for res in ["1", "2"] {
            let out = rig(depth, &[("vector_resolution", res)]).render(3).unwrap();
            assert_eq!((out.width, out.height, out.depth()), (W, H, depth), "the encoder writes the OUTPUT size, whatever the working size");
            for y in 0..H {
                for x in 0..W {
                    let [_, _, b, a] = px(&out, x, y);
                    assert_eq!((b, a), (0.0, white), "{depth:?} res {res} at ({x}, {y})");
                }
            }
        }
    }
}

/// The wiring: the front-end output is the engine's pass on the premultiplied frames of the pair, finished.
#[test]
fn the_front_end_is_the_engine_pass_on_the_pair_of_frames() {
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        for (track, debug_free) in [(1, "1"), (2, "2")] {
            let clip = DEFAULT.clip(depth, W, H, 8);
            let n = 4u64;
            let rig = Rig::new(clip.clone(), &[("max_displace", "16"), ("track_frame", debug_free)]).with_frames(8);
            let got = rig.render(n).unwrap();
            let ts = 100;
            let (a, b) = frame_pair(frame_index((n as i32) * ts, ts), track, false);
            let fa = &clip[(frame_time(a, ts) / ts) as usize];
            let fb = &clip[(frame_time(b, ts) / ts) as usize];
            let settings = CreateVectorsSettings {
                quality: 4,
                hiq: true,
                sensitivity: 70.0f32 / 100.0f32,
                prep_contrast_edge: false,
                prep_delinearize: false,
                track_rgb_a: false,
                max_displace: 16.0,
            };
            let want = create_vectors(&Image::from_frame(fa), &Image::from_frame(fb), &settings).unwrap().to_frame();
            assert_eq!(got, want, "{depth:?} track {track}");
        }
    }
}

#[test]
fn quality_decides_the_working_resolution_and_the_vectors_stay_at_output_pixels() {
    // Draft quality halves the solver resolution once more; the encoder still gives output pixels
    let clip = DEFAULT.clip(Depth::U8, W, H, 8);
    let rig = Rig::new(clip, &[("max_displace", "16")]).with_frames(8);
    let mut cx = rig.cx(3);
    cx.quality = fx_core::Quality::Draft;
    let draft = rig.render_cx(&cx).unwrap();
    let full = rig.render(3).unwrap();
    assert_ne!(draft, full, "a different working resolution gives a different field");
    let (mean, _, avg) = flow_error(&decode(&draft, MAX), (-3.4, -1.7), MARGIN);
    assert!(mean < 0.25 && avg.0 < -3.0 && avg.1 < -1.4, "{mean} {avg:?}");
}

#[test]
fn the_first_and_last_frames_with_a_missing_neighbour_still_encode() {
    // frame 0 with Previous has no frame -1 (transparent) and the last frame with Next none past
    // the end
    for (n, track) in [(0u64, "1"), (7, "2")] {
        let out = rig(Depth::U8, &[("track_frame", track)]).render(n).unwrap();
        for y in 0..H {
            for x in 0..W {
                let [_, _, b, a] = px(&out, x, y);
                assert_eq!((b, a), (0.0, 255.0));
            }
        }
    }
}

#[test]
fn a_still_clip_has_zero_vectors() {
    let still = Motion { vx: 0.0, vy: 0.0 };
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let frames: Vec<Frame> = still.clip(depth, W, H, 6);
        let rig = Rig::new(frames, &[("max_displace", "16")]).with_frames(6);
        let out = rig.render(3).unwrap();
        let (mean, max, _) = flow_error(&decode(&out, MAX), (0.0, 0.0), MARGIN);
        assert!(mean < 0.01 && max < 0.05, "{depth:?}: {mean} {max}");
        if depth == Depth::U8 {
            // the zero code (127) everywhere inside: the solver noise is far below half a code (0.063 px)
            for y in MARGIN..H - MARGIN {
                for x in MARGIN..W - MARGIN {
                    assert_eq!(&px(&out, x, y)[..2], &[127.0, 127.0], "({x}, {y})");
                }
            }
        }
    }
}

#[test]
fn the_16_bit_wire_loses_at_most_one_level_of_the_after_effects_range() {
    // The output finish maps the engine range 0..65535 to the range 0..32768 and the consumer's
    // input prepare maps it back, so a 16 bpc code survives within one level of the coarser range:
    // the zero code 32767 comes back as 32768, which decodes to Max Displace / 32767 px. Pinned on
    // the encoder output through the wire.
    use retime_engine::flow::FlowField;
    use retime_engine::image::Fmt;
    use retime_engine::vectors::encode_vectors_image;
    let zero = FlowField::new(4, 4);
    let wire = encode_vectors_image(4, 4, Fmt::U16, &zero, 16.0).unwrap().to_frame();
    let Pixels::U16(p) = &wire.pixels else { panic!() };
    assert!(p.iter().all(|px| px[3] == 65535 && px[2] == 0), "opaque, blue 0");
    let wire_zero = p[0][0];
    assert_eq!(wire_zero, 32768, "the zero code on the wire (32767 in the engine, 16384 on the other 15 bit scale, a tie rounded up)");
    let back = decode(&wire, 16.0);
    let bias = back.at(1, 1)[0].abs() as f64;
    assert!(bias <= 16.0 / 32767.0 * 1.01, "bias {bias} px");
}
