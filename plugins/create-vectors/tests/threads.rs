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
