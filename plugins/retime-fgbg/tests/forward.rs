//! The forward warp codes through plugin 03's front-end: Warping "Forward" (popup 3, warp code 2:
//! coverage image, Smart Blend composite and hole fill) and "Forward, no Smart Blend" (popup 4,
//! warp code 3: no coverage, no fill). Between two frames these render through the engine's forward
//! warp.
//!
//! Three kinds of check. (1) The render IS the forward composition: the same frames, flows and warp
//! code through the engine's own functions (`solve_flows`, `compose_vectors`) give the same bytes
//! as the effect, at every depth and Frame Interp mode. (2) An analytic ground truth
//! (`common/moving.rs`): a texture that translates by a known number of pixels per frame, so the
//! exact in-between frame is known in closed form; the forward Blend and Nearest results are much
//! closer to it than the plain frame blend. (3) The result does not depend on the thread count.

mod common;

use common::moving::*;
use common::*;
use fx_core::{Depth, Frame};
use retime_engine::compose::{compose_vectors, solve_flows, Rendered};
use retime_engine::image::Image;
use retime_engine::timeline::Selection;

const W: usize = 96;
const H: usize = 64;
const MARGIN: usize = 10;

/// The two forward popup values with their warp codes.
const FORWARD: [(&str, i32); 2] = [("3", 2), ("4", 3)];

/// Output frame `n` of half speed (n odd: between source frames `(n - 1) / 2` and `(n + 1) / 2`) computed with the
/// engine functions directly, from the settings the front-end reads.
fn engine_frame(rig: &Rig, n: u64, depth: Depth) -> Frame {
    let cx = rig.cx(n);
    let env = rig.env();
    let settings = retime_fgbg::glue::read_settings(&cx, &env).unwrap();
    let k = ((n - 1) / 2) as usize;
    let clip = &rig.input;
    let frame = |i: usize| Image::from_frame(&*fx_core::Layer::frame_at(clip, (i as i64) * 100).unwrap().expect("frame"));
    let (a, b) = (frame(k), frame(k + 1));
    let pos = n as f32 / 2.0;
    let sel = Selection { s0: pos, s1: pos + 0.5, a: k as i32, b: k as i32 + 1, wa: 0.5, exact: false, vectors_quality: settings.vectors_quality, interp: settings.interp };
    let flows = solve_flows(&settings, &sel, &a, &b).unwrap();
    let img = compose_vectors(&settings, &sel, &a, &b, &flows).unwrap();
    let _ = depth;
    Rendered::Image(img).into_frame()
}

#[test]
fn the_forward_warps_render_between_frames_and_are_the_forward_composition() {
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let clip = DEFAULT.clip(depth, W, H, 6);
        for (warping, code) in FORWARD {
            for interp in ["1", "2", "3"] {
                let rig = Rig::new(clip.clone(), &[("speed_percent", "50"), ("warping", warping), ("frame_interp", interp)]).with_frames(10);
                for n in [1u64, 3, 5] {
                    let got = rig.render(n).unwrap_or_else(|e| panic!("{depth:?} warping {warping} interp {interp} frame {n}: {e}"));
                    let want = engine_frame(&rig, n, depth);
                    assert_eq!(got, want, "{depth:?} warp code {code} interp {interp} frame {n}");
                }
                // an exact position never reaches a warp, whatever the code
                assert_eq!(rig.render(2).unwrap().to_bytes().len(), W * H * depth.bytes_per_pixel());
            }
        }
    }
}

#[test]
fn the_forward_warps_stay_close_to_the_analytic_frame_and_beat_the_frame_blend() {
    // Measured (release, 96 x 64, three motions of 3.4/1.7, -2.6/2.9 and 4.4/0 px per frame, half speed, frames 1, 3, 5): Blend
    // 0.0005 to 0.0027 against 0.014 to 0.022 for the frame blend (5.7 to 27 times closer), Nearest 0.0006 to 0.0031 against
    // 0.057 to 0.067 (18 to 109 times); the two forward codes agree to the last digit away from the holes, 16 and 32 bit
    // are slightly better than 8 bit. Asserted: error below 0.0045, Blend at least 4.5 times and Nearest at least 12 times
    // closer than the frame blend.
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        for motion in [DEFAULT, Motion { vx: -2.6, vy: 2.9 }, Motion { vx: 4.4, vy: 0.0 }] {
            let clip = motion.clip(depth, W, H, 8);
            let plain = Rig::new(clip.clone(), &[("speed_percent", "50"), ("vectors", "1")]).with_frames(10);
            for (warping, code) in FORWARD {
                for (interp, name, min_ratio) in [("2", "blend", 4.5), ("1", "nearest", 12.0)] {
                    let rig = Rig::new(clip.clone(), &[("speed_percent", "50"), ("warping", warping), ("frame_interp", interp)]).with_frames(10);
                    let plain_nearest = Rig::new(clip.clone(), &[("speed_percent", "50"), ("vectors", "1"), ("frame_interp", interp)]).with_frames(10);
                    for n in [1u64, 3, 5] {
                        let truth = motion.frame_at(depth, W, H, n as f64 / 2.0);
                        let got = rig.render(n).unwrap();
                        let (e, _) = error(&got, &truth, MARGIN);
                        let base = if interp == "2" { &plain } else { &plain_nearest };
                        let (eb, _) = error(&base.render(n).unwrap(), &truth, MARGIN);
                        assert!(e < 0.0045, "{depth:?} {motion:?} code {code} {name} frame {n}: error {e:.5}");
                        assert!(e * min_ratio < eb, "{depth:?} {motion:?} code {code} {name} frame {n}: {e:.5} against the plain {eb:.5}");
                    }
                }
            }
        }
    }
}

#[test]
fn forward_code_3_leaves_holes_transparent_and_code_2_fills_them() {
    // a pan of 6 pixels per frame: at half a frame the warped frame has a 3 pixel strip with no data at one edge
    let motion = Motion { vx: 6.0, vy: 0.0 };
    let clip = motion.clip(Depth::U8, W, H, 8);
    let alpha = |f: &Frame, x: usize, y: usize| match &f.pixels {
        fx_core::Pixels::U8(p) => p[y * W + x][3],
        _ => unreachable!(),
    };
    // Nearest at position 1.5 is wA = 0.5: frame A pushed by half the flow
    let c3 = Rig::new(clip.clone(), &[("speed_percent", "50"), ("warping", "4"), ("frame_interp", "1")]).with_frames(10).render(3).unwrap();
    let c2 = Rig::new(clip, &[("speed_percent", "50"), ("warping", "3"), ("frame_interp", "1")]).with_frames(10).render(3).unwrap();
    let transparent: usize = (0..H).map(|y| (0..W).filter(|&x| alpha(&c3, x, y) != 255).count()).sum();
    assert!(transparent > 0, "code 3 has holes at the edge the content moved away from");
    for y in 0..H {
        for x in 0..W {
            assert_eq!(alpha(&c2, x, y), 255, "code 2 is opaque everywhere ({x}, {y})");
        }
    }
}

#[test]
fn the_forward_codes_give_the_same_bytes_in_pools_of_one_and_four_threads() {
    let pool = |n: usize| rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
    for depth in [Depth::U8, Depth::F32] {
        let clip = DEFAULT.clip(depth, 67, 45, 6);
        for (warping, _) in FORWARD {
            for interp in ["1", "2", "3"] {
                let rig = Rig::new(clip.clone(), &[("speed_percent", "50"), ("warping", warping), ("frame_interp", interp), ("blur_compensation", "1")]).with_frames(9);
                let run = |n: usize| pool(n).install(|| (0..9u64).map(|f| rig.render(f).unwrap().to_bytes()).collect::<Vec<_>>());
                assert_eq!(run(1), run(4), "{depth:?} warping {warping} interp {interp}");
            }
        }
    }
}

/// Time of one interpolated frame of the forward codes at the two usual sizes (release build:
/// `cargo test -p retime-fgbg --test forward --release -- --ignored --nocapture timing`). Not a pass or fail test.
#[test]
#[ignore]
fn timing() {
    use std::time::Instant;
    println!("hardware threads: {}", rayon::current_num_threads());
    for (w, h) in [(1280usize, 720usize), (1920, 1080)] {
        for depth in [Depth::U8, Depth::F32] {
            let clip = DEFAULT.clip(depth, w, h, 3);
            for (warping, code) in FORWARD {
                for (interp, name) in [("1", "Nearest"), ("2", "Blend"), ("3", "Weighted Blend")] {
                    let rig = Rig::new(clip.clone(), &[("speed_percent", "50"), ("warping", warping), ("frame_interp", interp)]).with_frames(4);
                    let _ = rig.render(1).unwrap();
                    let runs = 3;
                    let t = Instant::now();
                    for _ in 0..runs {
                        let _ = rig.render(1).unwrap();
                    }
                    println!("{w}x{h} {depth:?} code {code} {name}: {:.0} ms per frame", t.elapsed().as_secs_f64() * 1000.0 / runs as f64);
                }
            }
        }
    }
}
