//! Tiny frames (parity gap G7, a rule of ours: the original divides by zero below 2x2): a clip of
//! any size from 1x1 up renders in every mode, without a panic and without an `Unsupported`. The
//! motion search never halves below 2x2, and a frame with a side of 1 has no search: its vector
//! image is the zero vector (R and G at the centre, B 0, alpha opaque), the image a zero flow
//! encodes to.
//!
//! Sizes 1x1 to 6x6, 1x17, 17x1 and 2x33 at the three depths (the float clips carry NaN,
//! infinite and negative samples), both Vector Resolutions and Track Frames, every Prep, Track
//! RGB+A, both render qualities, the default `Main_BG: Sensitivity` and 100, and the Source display.

mod common;

use common::*;
use fx_core::{Depth, Frame, Pixels, Quality};
use std::panic::{catch_unwind, AssertUnwindSafe};

const SIZES: [(usize, usize); 9] = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (1, 17), (17, 1), (2, 33)];
const DEPTHS: [Depth; 3] = [Depth::U8, Depth::U16, Depth::F32];
const FRAMES: u64 = 4;

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

fn modes() -> Vec<Vec<(&'static str, &'static str)>> {
    let mut out = Vec::new();
    for resolution in ["1", "2"] {
        for track in ["1", "2"] {
            for prep in ["1", "2", "3"] {
                for rgba in ["0", "1"] {
                    for sensitivity in ["70", "100"] {
                        out.push(vec![
                            ("display", "2"),
                            ("vector_resolution", resolution),
                            ("track_frame", track),
                            ("prep", prep),
                            ("track_rgb_a", rgba),
                            ("main_bg_sensitivity", sensitivity),
                        ]);
                    }
                }
            }
        }
    }
    out.push(vec![("display", "2"), ("max_displace", "1.5"), ("main_bg_sensitivity", "100")]);
    out.push(vec![("display", "1")]);
    out
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

/// The wire pixel of the zero vector at each depth, read from a searched 2x2 frame of a still clip
/// (its flow is exactly zero): what the encoder writes for a zero flow.
fn zero_pixel(depth: Depth) -> Frame {
    let still = vec![random_clip(depth, 2, 2, 1, 3).remove(0); 4];
    let got = Rig::new(still, &[("display", "2")]).render(1).unwrap();
    let first = match &got.pixels {
        Pixels::U8(p) => Pixels::U8(vec![p[0]]),
        Pixels::U16(p) => Pixels::U16(vec![p[0]]),
        Pixels::F32(p) => Pixels::F32(vec![p[0]]),
    };
    Frame { width: 1, height: 1, pixels: first }
}

#[test]
fn every_size_renders_in_every_mode() {
    let mut failures = Vec::new();
    let mut renders = 0;
    for depth in DEPTHS {
        let zero = zero_pixel(depth);
        for (w, h) in SIZES {
            let clip = tiny_clip(depth, w, h);
            let side_of_one = w < 2 || h < 2;
            for params in modes() {
                let rig = Rig::new(clip.clone(), &params).with_frames(FRAMES);
                let vectors = params.contains(&("display", "2"));
                for quality in [Quality::High, Quality::Draft] {
                    for n in 0..FRAMES {
                        let case = format!("{depth:?} {w}x{h} {quality:?} frame {n} {params:?}");
                        match render(&rig, n, quality) {
                            Ok(got) => {
                                renders += 1;
                                // a side of 1 has no motion search: every pixel is the zero vector
                                let all_zero = match (&got.pixels, &zero.pixels) {
                                    (Pixels::U8(p), Pixels::U8(z)) => p.iter().all(|v| *v == z[0]),
                                    (Pixels::U16(p), Pixels::U16(z)) => p.iter().all(|v| *v == z[0]),
                                    (Pixels::F32(p), Pixels::F32(z)) => p.iter().all(|v| v.map(f32::to_bits) == z[0].map(f32::to_bits)),
                                    _ => false,
                                };
                                if side_of_one && vectors && !all_zero {
                                    failures.push(format!("{case}: not the zero vector"));
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

#[test]
fn the_zero_vector_is_the_centre_code() {
    // 8 bpc and float on the wire: R and G at the centre, B 0, opaque
    assert_eq!(zero_pixel(Depth::U8).pixels, Pixels::U8(vec![[127, 127, 0, 255]]));
    assert_eq!(zero_pixel(Depth::F32).pixels, Pixels::F32(vec![[0.5, 0.5, 0.0, 1.0]]));
    // a 1x1 clip with motion: the zero vector at every depth, whatever the content
    for depth in DEPTHS {
        let clip = common::moving::DEFAULT.clip(depth, 1, 1, 4);
        let got = Rig::new(clip, &[("display", "2")]).render(1).unwrap();
        assert_eq!(got.pixels, zero_pixel(depth).pixels, "{depth:?}");
    }
}

/// The layered settings: FG matte layers (with shrinks, Layers Output, FG Track off), points,
/// `Main_BG: Image Track` off and the five Debug views with the Draw Geom overlay.
fn layered_modes() -> Vec<(Vec<(&'static str, &'static str)>, usize)> {
    let mut out = Vec::new();
    for resolution in ["1", "2"] {
        for fg in [0usize, 1, 2] {
            out.push((vec![("display", "2"), ("vector_resolution", resolution), ("fg1_inv_matte_shrink", "1.5")], fg));
            out.push((vec![("display", "2"), ("vector_resolution", resolution), ("layers_output", "2"), ("fg1_track", "0")], fg));
            out.push((vec![("display", "2"), ("vector_resolution", resolution), ("main_bg_track", "0"), ("pt1_use", "5"), ("pt1_pos", "0,0@0;1,1@3")], fg));
            out.push((vec![("display", "2"), ("vector_resolution", resolution), ("pt2_use", "2"), ("pt2_pos", "1,0@0;0,1@3"), ("track_frame", "2")], fg));
            for view in ["1", "2", "3", "4", "5"] {
                out.push((vec![("display", "3"), ("vector_resolution", resolution), ("debug_view_method", view), ("pt3_use", "2"), ("pt3_pos", "0.5,0.5")], fg));
            }
        }
    }
    out.push((vec![("display", "1"), ("pt1_use", "2"), ("pt1_pos", "0,0")], 1));
    out
}

#[test]
fn every_size_renders_the_layered_settings() {
    // U, ours (G7 family): a side of 1 is the zero vector on the Vectors and the Debug display
    // whatever the FG layers, points, Track, Layers Output and Debug settings; every other size
    // runs the layered paths without a panic or an error
    let mut failures = Vec::new();
    let mut renders = 0;
    for depth in DEPTHS {
        let zero = zero_pixel(depth);
        for (w, h) in SIZES {
            let clip = tiny_clip(depth, w, h);
            let side_of_one = w < 2 || h < 2;
            for (params, fg) in layered_modes() {
                let mut rig = Rig::new(clip.clone(), &params).with_frames(FRAMES);
                for (k, id) in ["fg1_matte", "fg2_matte"].iter().enumerate().take(fg) {
                    rig = rig.with_layer(id, tiny_clip(depth, w, h).into_iter().rev().skip(k).chain(tiny_clip(depth, w, h)).take(4).collect());
                }
                let source = params.contains(&("display", "1"));
                for quality in [Quality::High, Quality::Draft] {
                    for n in 0..FRAMES {
                        let case = format!("{depth:?} {w}x{h} {quality:?} frame {n} fg {fg} {params:?}");
                        match render(&rig, n, quality) {
                            Ok(got) => {
                                renders += 1;
                                let all_zero = match (&got.pixels, &zero.pixels) {
                                    (Pixels::U8(p), Pixels::U8(z)) => p.iter().all(|v| *v == z[0]),
                                    (Pixels::U16(p), Pixels::U16(z)) => p.iter().all(|v| *v == z[0]),
                                    (Pixels::F32(p), Pixels::F32(z)) => p.iter().all(|v| v.map(f32::to_bits) == z[0].map(f32::to_bits)),
                                    _ => false,
                                };
                                if side_of_one && !source && !all_zero {
                                    failures.push(format!("{case}: not the zero vector"));
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
