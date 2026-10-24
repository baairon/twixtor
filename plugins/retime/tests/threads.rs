//! The result of a render does not depend on the number of threads: rows are independent and every
//! reduction is in a fixed order. The same renders run in rayon pools of one and four threads, and
//! in the global pool, and must agree byte for byte.

mod common;

use common::*;
use fx_core::Depth;
use rayon::ThreadPoolBuilder;

const SCENARIOS: &[&[(&str, &str)]] = &[
    // exact frames with the default (Best, Weighted Blend)
    &[],
    &[("vectors", "1"), ("speed_percent", "50"), ("frame_interp", "1")],
    &[("vectors", "1"), ("speed_percent", "50"), ("frame_interp", "2")],
    &[("vectors", "1"), ("speed_percent", "50"), ("frame_interp", "3")],
    &[("vectors", "1"), ("speed_percent", "33.3"), ("frame_interp", "2")],
    &[("display", "2"), ("src_time_offset", "1")],
    // negative positions take the extrapolation branch of the blend
    &[("vectors", "1"), ("speed_percent", "-50"), ("src_time_offset", "5"), ("frame_interp", "2")],
    &[("vectors", "1"), ("time_remap_mode", "2"), ("frame_num", "-0.5")],
];

fn bytes_of(depth: Depth, params: &[(&str, &str)]) -> Vec<Vec<u8>> {
    // 67 x 45 so that rows do not divide evenly among four workers
    let rig = Rig::new(random_clip(depth, 67, 45, 6, 0xC0FFEE), params).with_frames(10);
    (0..10).map(|n| rig.render(n).unwrap().to_bytes()).collect()
}

#[test]
fn one_thread_four_threads_and_the_global_pool_agree() {
    let one = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let four = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        for params in SCENARIOS {
            let a = one.install(|| {
                assert_eq!(rayon::current_num_threads(), 1);
                bytes_of(depth, params)
            });
            let b = four.install(|| {
                assert_eq!(rayon::current_num_threads(), 4);
                bytes_of(depth, params)
            });
            assert_eq!(a, b, "{depth:?} {params:?}");
            assert_eq!(a, bytes_of(depth, params), "{depth:?} {params:?} global pool");
        }
    }
}

#[test]
fn the_scenarios_are_not_trivially_empty() {
    // guard against comparing blank frames: the half speed renders differ between frames and are not all zero
    let frames = bytes_of(Depth::U8, &[("vectors", "1"), ("speed_percent", "50"), ("frame_interp", "2")]);
    assert!(frames.iter().all(|f| f.iter().any(|b| *b != 0)));
    assert_ne!(frames[0], frames[1]);
    assert_ne!(frames[1], frames[2]);
}

/// The vector path: flow solver, inverse warps, sharpening and blur compensation on a clip with real motion. Each
/// scenario renders a run that crosses the end of the clip (the last positions blend with transparency).
const VECTOR_SCENARIOS: &[&[(&str, &str)]] = &[
    // the defaults: Best, Blend
    &[("speed_percent", "50")],
    &[("speed_percent", "25"), ("frame_interp", "1")],
    &[("speed_percent", "50"), ("frame_interp", "3")],
    // Blur Compensation: motion blur along the flow (fast), flow directed sharpening (slow)
    &[("speed_percent", "150"), ("blur_compensation", "1")],
    &[("speed_percent", "50"), ("blur_compensation", "1")],
    &[("speed_percent", "200"), ("blur_compensation", "0.5"), ("frame_interp", "3")],
    // solver settings: prep modes, sensitivity, video range luma, lower qualities
    &[("speed_percent", "50"), ("prep", "2"), ("sensitivity", "85")],
    &[("speed_percent", "50"), ("prep", "3"), ("frame_interp", "3")],
    &[("speed_percent", "50"), ("video_360", "1"), ("vectors", "4")],
    &[("speed_percent", "50"), ("vectors", "2")],
    // the retime of a speed curve and of Frame Number mode
    &[("speed_percent", "40@0;120@4"), ("blur_compensation", "1")],
    &[("time_remap_mode", "2"), ("frame_num", "0@0;6@7")],
    // Smart Blend (warping code 1): coverage warps, the Smart Blend composites, the pull-push hole fill
    &[("speed_percent", "50"), ("smart_blend", "1")],
    &[("speed_percent", "25"), ("smart_blend", "1"), ("frame_interp", "1")],
    &[("speed_percent", "50"), ("smart_blend", "1"), ("frame_interp", "3")],
    &[("speed_percent", "50"), ("smart_blend", "1"), ("video_360", "1")],
    &[("speed_percent", "150"), ("smart_blend", "1"), ("blur_compensation", "1")],
];

fn vector_bytes(depth: Depth, params: &[(&str, &str)], tracking: bool) -> Vec<Vec<u8>> {
    // 67 x 45 so that rows do not divide evenly among four workers; 7 source frames, 9 output frames
    let clip = common::moving::DEFAULT.clip(depth, 67, 45, 7);
    let mut rig = Rig::new(clip, params).with_frames(9);
    if tracking {
        // the solver reads a differently moving clip, the pixels come from the retimed layer
        let other = common::moving::Motion { vx: -2.0, vy: 2.5 }.clip(depth, 67, 45, 7);
        rig = rig.with_layer("alt_source_b", other);
    }
    (0..9).map(|n| rig.render(n).unwrap().to_bytes()).collect()
}

#[test]
fn the_whole_vector_path_gives_the_same_bytes_in_pools_of_one_and_four_threads() {
    let one = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let four = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        for (i, params) in VECTOR_SCENARIOS.iter().enumerate() {
            for tracking in [false, i == 0 || i == 2 || i == 12] {
                let a = one.install(|| {
                    assert_eq!(rayon::current_num_threads(), 1);
                    vector_bytes(depth, params, tracking)
                });
                let b = four.install(|| {
                    assert_eq!(rayon::current_num_threads(), 4);
                    vector_bytes(depth, params, tracking)
                });
                assert_eq!(a, b, "{depth:?} {params:?} tracking layer {tracking}");
                if i < 3 && !tracking {
                    // the default pool (every hardware thread) as well
                    assert_eq!(a, vector_bytes(depth, params, tracking), "{depth:?} {params:?} global pool");
                }
            }
        }
    }
}

#[test]
fn the_vector_scenarios_are_not_trivially_equal_or_empty() {
    // guard against comparing blank frames or two renders that are really the frame blend
    let flow = vector_bytes(Depth::U8, &[("speed_percent", "50")], false);
    let plain = vector_bytes(Depth::U8, &[("speed_percent", "50"), ("vectors", "1")], false);
    assert!(flow.iter().all(|f| f.iter().any(|b| *b != 0)));
    assert_eq!(flow[0], plain[0], "an exact position is the same frame with and without vectors");
    for n in [1usize, 3, 5] {
        assert_ne!(flow[n], plain[n], "frame {n} is interpolated by the flow");
    }
    // the blur compensation changes frames the slider does not touch otherwise
    let blurred = vector_bytes(Depth::U8, &[("speed_percent", "150"), ("blur_compensation", "1")], false);
    let sharp = vector_bytes(Depth::U8, &[("speed_percent", "150")], false);
    assert_ne!(blurred, sharp);
    // the tracking layer feeds the solver only: the result differs from the render without it
    let tracked = vector_bytes(Depth::U8, &[("speed_percent", "50")], true);
    assert_ne!(tracked[1], flow[1]);
    // Smart Blend fills the strips the hold edge warps replicate: the interpolated frames differ, the exact ones do not
    let smart = vector_bytes(Depth::U8, &[("speed_percent", "50"), ("smart_blend", "1")], false);
    assert_eq!(smart[0], flow[0]);
    for n in [1usize, 3, 5] {
        assert_ne!(smart[n], flow[n], "frame {n} is composed with coverage");
    }
}
