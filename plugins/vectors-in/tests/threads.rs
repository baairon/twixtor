//! The result of a render does not depend on the number of threads: rows are independent and every
//! reduction is in a fixed order. The same renders run in rayon pools of one and four threads and
//! in the global pool and must agree byte for byte: the fast path, the Vectors In path (decode,
//! hole fill, inverse warps, sharpening, blur compensation) and the Source display.

mod common;

use common::moving::*;
use common::*;
use fx_core::Depth;
use rayon::ThreadPoolBuilder;

const SCENARIOS: &[&[(&str, &str)]] = &[
    // exact frames
    &[("warping", "1")],
    &[("warping", "1"), ("speed_percent", "50"), ("frame_interp", "1")],
    &[("warping", "1"), ("speed_percent", "50"), ("frame_interp", "2")],
    &[("warping", "1"), ("speed_percent", "50"), ("frame_interp", "3")],
    &[("warping", "1"), ("speed_percent", "33.3"), ("frame_interp", "2")],
    &[("warping", "1"), ("speed_percent", "150"), ("blur_compensation", "1")],
    &[("warping", "1"), ("speed_percent", "50"), ("blur_compensation", "1")],
    &[("warping", "1"), ("speed_percent", "200"), ("blur_compensation", "0.5"), ("frame_interp", "3")],
    &[("warping", "1"), ("speed_percent", "50"), ("vec_scale_x", "1.5"), ("vec_scale_y", "-0.5")],
    &[("warping", "1"), ("speed_percent", "40@0;120@4"), ("blur_compensation", "1")],
    &[("warping", "1"), ("time_remap_mode", "2"), ("frame_num", "0@0;6@7")],
    // negative positions take the extrapolation branch of the blend
    &[("warping", "1"), ("speed_percent", "-50"), ("frame_interp", "2")],
    &[("warping", "1"), ("in_fps_is_out_fps", "0"), ("input_frame_rate", "24"), ("frame_interp", "2")],
    &[("warping", "1"), ("src_mark_segments", "5"), ("speed_percent", "50")],
    &[("display", "2")],
];

fn bytes_of(depth: Depth, params: &[(&str, &str)]) -> Vec<Vec<u8>> {
    // 67 x 45 so that rows do not divide evenly among four workers
    let mut p = vec![("vectors_max_displace", "16")];
    p.extend_from_slice(params);
    let rig = Rig::new(DEFAULT.clip(depth, 67, 45, 6), &p).with_frames(10).with_vectors(3.4, 1.7, 16.0);
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
fn the_vector_renders_differ_from_the_frame_blend_so_the_comparison_is_not_trivial() {
    // guard against comparing blank or identical frames: the vector path changes the in between frames
    let with = bytes_of(Depth::U8, &[("warping", "1"), ("speed_percent", "50")]);
    let plain = bytes_of(Depth::U8, &[("warping", "1"), ("speed_percent", "50"), ("src_mark_segments", "5")]);
    assert!(with.iter().all(|f| f.iter().any(|b| *b != 0)));
    assert_ne!(with[1], plain[1]);
    assert_ne!(with[3], plain[3]);
    assert_eq!(with[2], plain[2], "an exact frame is the same either way");
}
