//! Tiny frames (parity gap G7, a rule of ours: the original divides by zero below 2x2): a clip of
//! any size from 1x1 up renders in every mode, without a panic and without an `Unsupported`. The
//! motion search never halves below 2x2, and a frame with a side of 1 has no search: it renders
//! exactly as `vectors=1` (No Vectors).
//!
//! Sizes 1x1 to 6x6, 1x17, 17x1 and 2x33 at the three depths (the float clips carry NaN,
//! infinite and negative samples), every Vectors quality, Frame Interp mode, Smart Blend,
//! Blur Compensation 0 and 2.5, both render qualities, the default Sensitivity (a one level
//! pyramid then solves to a zero flow) and 100 (it keeps the flow), Prep and 360 Video.

mod common;

use common::*;
use fx_core::{Depth, Frame, Pixels, Quality};
use std::panic::{catch_unwind, AssertUnwindSafe};

const SIZES: [(usize, usize); 9] = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (1, 17), (17, 1), (2, 33)];
const DEPTHS: [Depth; 3] = [Depth::U8, Depth::U16, Depth::F32];
/// Output frames per run: at 37.5 percent the positions are 0, 0.375, 0.75, 1.125, 1.5 and 1.875.
const FRAMES: u64 = 6;

/// Four frames of pseudo random pixels of every alpha class; at float also NaN, infinite and
/// negative samples.
fn tiny_clip(depth: Depth, w: usize, h: usize) -> Vec<Frame> {
    let mut clip = random_clip(depth, w, h, 4, (w * 131 + h * 7) as u64);
    for (k, f) in clip.iter_mut().enumerate() {
        if let Pixels::F32(px) = &mut f.pixels {
            for (i, p) in px.iter_mut().enumerate() {
                match (i + k) % 5 {
                    1 => p[0] = f32::NAN,
                    3 => {
                        p[1] = f32::INFINITY;
                        p[2] = -0.75;
                        p[3] = 1.0;
                    }
                    4 => p[3] = f32::NAN,
                    _ => {}
                }
            }
        }
    }
    clip
}

/// The parameter sets: every Vectors quality, Frame Interp and Smart Blend, Blur Compensation 0
/// and 2.5 and the two sensitivities, plus Prep, 360 Video and the Source display at Best.
fn modes() -> Vec<Vec<(&'static str, &'static str)>> {
    let mut out = Vec::new();
    for vectors in ["1", "2", "3", "4", "5"] {
        for interp in ["1", "2", "3"] {
            for smart in ["0", "1"] {
                for blur in ["0", "2.5"] {
                    for sensitivity in ["70", "100"] {
                        out.push(vec![
                            ("speed_percent", "37.5"),
                            ("vectors", vectors),
                            ("frame_interp", interp),
                            ("smart_blend", smart),
                            ("blur_compensation", blur),
                            ("sensitivity", sensitivity),
                        ]);
                    }
                }
            }
        }
    }
    for (prep, video_360) in [("2", "0"), ("3", "0"), ("1", "1"), ("2", "1")] {
        for interp in ["1", "2", "3"] {
            for smart in ["0", "1"] {
                out.push(vec![
                    ("speed_percent", "37.5"),
                    ("vectors", "5"),
                    ("frame_interp", interp),
                    ("smart_blend", smart),
                    ("blur_compensation", "2.5"),
                    ("sensitivity", "100"),
                    ("prep", prep),
                    ("video_360", video_360),
                ]);
            }
        }
    }
    out.push(vec![("speed_percent", "37.5"), ("display", "2")]);
    out
}

/// The same parameters with Vectors set to No Vectors.
fn no_vectors(params: &[(&'static str, &'static str)]) -> Vec<(&'static str, &'static str)> {
    let mut p: Vec<_> = params.iter().copied().filter(|(id, _)| *id != "vectors").collect();
    p.push(("vectors", "1"));
    p
}

fn render(rig: &Rig, n: u64, quality: Quality) -> Result<Frame, String> {
    let mut cx = rig.cx(n);
    cx.quality = quality;
    match catch_unwind(AssertUnwindSafe(|| rig.render_cx(&cx))) {
        Ok(Ok(f)) => Ok(f),
        Ok(Err(e)) => Err(format!("error: {e}")),
        Err(_) => Err("panicked".into()),
    }
}

#[test]
fn every_size_renders_in_every_mode() {
    let mut failures = Vec::new();
    let mut renders = 0;
    for depth in DEPTHS {
        for (w, h) in SIZES {
            let clip = tiny_clip(depth, w, h);
            let side_of_one = w < 2 || h < 2;
            for params in modes() {
                let rig = Rig::new(clip.clone(), &params).with_frames(FRAMES);
                let plain = Rig::new(clip.clone(), &no_vectors(&params)).with_frames(FRAMES);
                for quality in [Quality::High, Quality::Draft] {
                    for n in 0..FRAMES {
                        let case = format!("{depth:?} {w}x{h} {quality:?} frame {n} {params:?}");
                        match render(&rig, n, quality) {
                            Ok(got) => {
                                renders += 1;
                                // a side of 1 has no motion search: the frame is the No Vectors render, bit for bit
                                if side_of_one {
                                    let want = render(&plain, n, quality).expect("No Vectors renders");
                                    if got.to_bytes() != want.to_bytes() {
                                        failures.push(format!("{case}: differs from the No Vectors render"));
                                    }
                                }
                            }
                            Err(e) => failures.push(format!("{case}: {e}")),
                        }
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{} failures in {} renders:\n{}", failures.len(), renders + failures.len(), failures.join("\n"));
}

/// A frame that is searchable but too small for its halvings is searched at the last size of at
/// least 2x2: with motion and a sensitivity that keeps the flow, the vector path output differs from
/// the frame blend (it is not a disguised No Vectors render).
#[test]
fn small_searchable_frames_take_the_vector_path() {
    for depth in DEPTHS {
        for (w, h) in [(6usize, 6usize), (5, 5)] {
            let clip = common::moving::DEFAULT.clip(depth, w, h, 4);
            let params = [("speed_percent", "50"), ("vectors", "2"), ("frame_interp", "2"), ("sensitivity", "100")];
            let vec = Rig::new(clip.clone(), &params).with_frames(FRAMES);
            let plain = Rig::new(clip, &no_vectors(&params)).with_frames(FRAMES);
            let differs = (0..FRAMES).any(|n| vec.render(n).unwrap() != plain.render(n).unwrap());
            assert!(differs, "{depth:?} {w}x{h}: the vector path output equals the frame blend in every frame");
        }
    }
}
