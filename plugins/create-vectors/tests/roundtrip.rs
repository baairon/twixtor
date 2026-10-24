//! Round trip: the vector image Create Vectors writes, read the way plugins 03 and 04 read it and
//! fed into the inverse warp, reproduces the OTHER frame of the pair much better than no warp. The
//! warp is `W(Y, flow, 1)`, `out(p) = Y(p + flow(p))` (`warp::inv_warp`), and the flow satisfies
//! `X(p) ~ Y(p + flow(p))`, so warping `Y` (the second frame of the pair) must give `X` (the first
//! one, the current frame).
//!
//! Metric: mean absolute colour error as a fraction of full scale over the pixels at least `MARGIN`
//! pixels from every edge (content entering from outside the picture cannot be reconstructed), the
//! frame itself against the warped other frame (`warped`) and against the unwarped other frame
//! (`plain`). The stated margin is `warped * 20 < plain` and `warped < 0.006`; measured at 96 x 64:
//! warped 0.0024 to 0.0032, unwarped 0.124 to 0.126, 39 to 53 times closer (printed by
//! `--nocapture`).

mod common;

use common::moving::*;
use common::*;
use fx_core::{Depth, Frame};
use retime_engine::image::Image;
use retime_engine::warp::{inv_warp, WarpOpts};

const W: usize = 96;
const H: usize = 64;
const MARGIN: usize = 12;
const MAX: f32 = 16.0;
/// The warped frame must be at least this many times closer than the unwarped one.
const MIN_GAIN: f64 = 20.0;
/// And absolutely close: mean error below this fraction of full scale.
const MAX_WARPED_ERROR: f64 = 0.006;

fn warp_other(depth: Depth, clip: &[Frame], n: usize, other: usize, params: &[(&str, &str)]) -> (Frame, Frame) {
    let mut p = vec![("max_displace", "16")];
    p.extend_from_slice(params);
    let rig = Rig::new(clip.to_vec(), &p).with_frames(clip.len() as u64);
    let vectors = rig.render(n as u64).unwrap();
    let flow = decode(&vectors, MAX);
    let src = Image::from_frame(&clip[other]);
    let mut out = Image::new(W, H, src.fmt());
    inv_warp(&mut out, &src, &flow, 1.0, WarpOpts::default()).unwrap();
    assert_eq!(depth, out.fmt().depth());
    (out.to_frame(), Image::from_frame(&clip[n]).to_frame())
}

#[test]
fn warping_the_other_frame_with_the_vectors_reproduces_the_current_frame() {
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let clip = DEFAULT.clip(depth, W, H, 8);
        for (track, other) in [("1", 2usize), ("2", 4usize)] {
            for res in ["1", "2"] {
                let (warped, current) = warp_other(depth, &clip, 3, other, &[("track_frame", track), ("vector_resolution", res)]);
                let (e_warp, _) = error(&warped, &current, MARGIN);
                let (e_plain, _) = error(&Image::from_frame(&clip[other]).to_frame(), &current, MARGIN);
                eprintln!("{depth:?} track {track} res {res}: warped {e_warp:.5}, unwarped {e_plain:.5}, gain {:.1}", e_plain / e_warp);
                assert!(e_warp < MAX_WARPED_ERROR, "{depth:?} track {track} res {res}: warped error {e_warp}");
                assert!(e_warp * MIN_GAIN < e_plain, "{depth:?} track {track} res {res}: warped {e_warp} against plain {e_plain}");
            }
        }
    }
}

#[test]
fn the_vectors_of_the_wrong_sign_do_not_reproduce_the_frame() {
    // the check above would also pass for a field that merely shrinks the error; the sign matters: warping the
    // frame on the OTHER side (Next vectors applied to the previous frame) is worse than not warping
    let depth = Depth::U8;
    let clip = DEFAULT.clip(depth, W, H, 8);
    let rig = Rig::new(clip.clone(), &[("max_displace", "16"), ("track_frame", "2")]).with_frames(8);
    let flow = decode(&rig.render(3).unwrap(), MAX);
    // frame 3 and its Previous neighbour 2: Next vectors point away from frame 2
    let src = Image::from_frame(&clip[2]);
    let mut out = Image::new(W, H, src.fmt());
    inv_warp(&mut out, &src, &flow, 1.0, WarpOpts::default()).unwrap();
    let current = Image::from_frame(&clip[3]).to_frame();
    let (wrong, _) = error(&out.to_frame(), &current, MARGIN);
    let (plain, _) = error(&Image::from_frame(&clip[2]).to_frame(), &current, MARGIN);
    assert!(wrong > plain, "wrong side {wrong} against unwarped {plain}");
}
