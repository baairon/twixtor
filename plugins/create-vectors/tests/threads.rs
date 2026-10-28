//! The result of a render does not depend on the number of threads: rows are independent and every
//! reduction is in a fixed order. The same renders run in rayon pools of one and four threads, and
//! in the global pool, and must agree byte for byte: the solver (`flow::solve`), the encoder and
//! the finish.

mod common;

use common::moving::*;
use common::*;
use fx_core::Depth;
use rayon::ThreadPoolBuilder;

const SCENARIOS: &[&[(&str, &str)]] = &[
    &[("max_displace", "16")],
    &[("max_displace", "16"), ("track_frame", "2")],
    &[("max_displace", "16"), ("vector_resolution", "2")],
    &[("max_displace", "30"), ("prep", "2")],
    &[("max_displace", "30"), ("prep", "3"), ("track_rgb_a", "1")],
    &[("display", "1")],
];

fn bytes_of(depth: Depth, params: &[(&str, &str)]) -> Vec<Vec<u8>> {
    // 67 x 45 so that rows do not divide evenly among four workers
    let rig = Rig::new(DEFAULT.clip(depth, 67, 45, 6), params).with_frames(6);
    (0..6).map(|n| rig.render(n).unwrap().to_bytes()).collect()
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
    // guard against comparing blank frames: the vector images differ between frames' parameters and are not all one code
    let frames = bytes_of(Depth::U8, &[("max_displace", "16")]);
    assert!(frames.iter().all(|f| f.iter().any(|b| *b != 0)));
    let distinct: std::collections::HashSet<u8> = frames[3].iter().copied().collect();
    assert!(distinct.len() > 4, "{} distinct bytes", distinct.len());
    assert_ne!(bytes_of(Depth::U8, &[("max_displace", "16"), ("track_frame", "2")])[3], frames[3]);
}

/// A matte clip of `w x h`: an opaque white disc on transparent black moving by (2, 1) px a frame.
fn disc_clip(depth: Depth, w: usize, h: usize, cx: f64, cy: f64, r: f64) -> Vec<fx_core::Frame> {
    (0..6)
        .map(|k| {
            let mut f = fx_core::Frame::new(w, h, depth);
            for i in 0..w * h {
                let (x, y) = ((i % w) as f64, (i / w) as f64);
                let inside = ((x - cx - 2.0 * k as f64).powi(2) + (y - cy - k as f64).powi(2)).sqrt() <= r;
                match &mut f.pixels {
                    fx_core::Pixels::U8(p) => p[i] = if inside { [255; 4] } else { [0; 4] },
                    fx_core::Pixels::U16(p) => p[i] = if inside { [65535; 4] } else { [0; 4] },
                    fx_core::Pixels::F32(p) => p[i] = if inside { [1.0; 4] } else { [0.0; 4] },
                }
            }
            f
        })
        .collect()
}

const LAYERED: &[&[(&str, &str)]] = &[
    &[("max_displace", "16")],
    &[("max_displace", "16"), ("fg1_inv_matte_shrink", "2.5"), ("fg2_track", "0"), ("pt1_use", "3"), ("pt1_pos", "30,20@0;36,23@5")],
    &[("max_displace", "16"), ("layers_output", "2"), ("pt2_use", "2"), ("pt2_pos", "20,18@0;25,20@5")],
    &[("max_displace", "16"), ("main_bg_track", "0"), ("pt3_use", "5"), ("pt3_pos", "50,30@0;46,32@5")],
    &[("display", "3"), ("debug_view_method", "1")],
    &[("display", "3"), ("debug_view_method", "4"), ("pt3_use", "2"), ("pt3_pos", "12,12")],
];

#[test]
fn the_layered_paths_agree_for_one_four_and_sixteen_threads() {
    let pools: Vec<_> = [1usize, 4, 16].iter().map(|&n| ThreadPoolBuilder::new().num_threads(n).build().unwrap()).collect();
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        for params in LAYERED {
            let run = || {
                let rig = Rig::new(DEFAULT.clip(depth, 67, 45, 6), params)
                    .with_frames(6)
                    .with_layer("fg1_matte", disc_clip(depth, 67, 45, 22.0, 18.0, 13.0))
                    .with_layer("fg2_matte", disc_clip(depth, 67, 45, 40.0, 24.0, 11.0));
                (0..6).map(|n| rig.render(n).unwrap().to_bytes()).collect::<Vec<_>>()
            };
            let one = pools[0].install(run);
            for pool in &pools[1..] {
                assert_eq!(pool.install(run), one, "{depth:?} {params:?}");
            }
        }
    }
}
