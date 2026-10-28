//! Tiny frames (parity gap G7, a rule of ours: the original divides by zero below 2x2): a clip of
//! any size from 1x1 up renders in every mode, without a panic and without an `Unsupported`. A
//! frame with a side of 1 has no vector path: it renders as `vectors=1` would (the fast path),
//! which here equals the render whose segment marker turns the vectors off.
//!
//! Sizes 1x1 to 6x6, 1x17, 17x1 and 2x33 at the three depths, both Warping codes (the inverse
//! warp and the forward warp with its hole fill), every Frame Interp mode, Blur Compensation 0 and
//! 2.5, both render qualities, both vector layers or only one (the missing flow is the inversion
//! of the other), and three kinds of vector layer: pseudo random (holes and partial alpha), a shear
//! (horizontal motion growing down the rows, vertical motion exactly 0) and a constant motion. The
//! source clips carry NaN, infinite and negative float samples. NaN in a vector layer is a refusal
//! of the warps at every size that has a vector path, not a size limit
//! ([`nan_vector_samples_are_refused_at_every_size_with_a_vector_path`]).

mod common;

use common::*;
use fx_core::{Depth, Frame, Pixels, Quality};
use retime_engine::image::{Fmt, Image, Samples};
use std::panic::{catch_unwind, AssertUnwindSafe};

const SIZES: [(usize, usize); 9] = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (1, 17), (17, 1), (2, 33)];
const DEPTHS: [Depth; 3] = [Depth::U8, Depth::U16, Depth::F32];
/// Output frames per run: at 37.5 percent the positions are 0, 0.375, 0.75, 1.125, 1.5 and 1.875.
const FRAMES: u64 = 6;

/// Four source frames of pseudo random pixels of every alpha class; at float also NaN, infinite and
/// negative samples.
fn tiny_clip(depth: Depth, w: usize, h: usize) -> Vec<Frame> {
    let mut clip = random_clip(depth, w, h, 4, (w * 131 + h * 7) as u64);
    for (k, f) in clip.iter_mut().enumerate() {
        poison(f, k);
    }
    clip
}

/// NaN, infinite and negative samples in a float frame (other depths are left as they are).
fn poison(f: &mut Frame, k: usize) {
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

/// An opaque encoded vector frame from engine values: `code(x, y)` gives the R and G codes as
/// offsets from the centre in units of the depth's step (8 bpc: 1, 16 bpc: 256, float: 1 / 254).
fn coded(depth: Depth, w: usize, h: usize, code: impl Fn(usize, usize) -> (i32, i32)) -> Frame {
    let mut img = Image::new(w, h, Fmt::from_depth(depth));
    match &mut img.samples {
        Samples::U8(v) => {
            for (i, px) in v.chunks_exact_mut(4).enumerate() {
                let (r, g) = code(i % w, i / w);
                px.copy_from_slice(&[255, (127 + r).clamp(0, 254) as u8, (127 + g).clamp(0, 254) as u8, 0]);
            }
        }
        Samples::U16(v) => {
            for (i, px) in v.chunks_exact_mut(4).enumerate() {
                let (r, g) = code(i % w, i / w);
                px.copy_from_slice(&[65535, (32767 + 256 * r).clamp(0, 65534) as u16, (32767 + 256 * g).clamp(0, 65534) as u16, 0]);
            }
        }
        Samples::F32(v) => {
            for (i, px) in v.chunks_exact_mut(4).enumerate() {
                let (r, g) = code(i % w, i / w);
                px.copy_from_slice(&[1.0, 0.5 + r as f32 / 254.0, 0.5 + g as f32 / 254.0, 0.0]);
            }
        }
    }
    img.to_frame()
}

/// The vector layers `(prev, next)` of one kind, eight frames each.
fn vector_layers(kind: &str, depth: Depth, w: usize, h: usize) -> (Vec<Frame>, Vec<Frame>) {
    match kind {
        "random" => (random_clip(depth, w, h, 8, 77 + (w * 5 + h) as u64), random_clip(depth, w, h, 8, 91 + (w * 3 + h) as u64)),
        "nan" => {
            let mut prev = random_clip(depth, w, h, 8, 77);
            let mut next = random_clip(depth, w, h, 8, 91);
            for (k, f) in prev.iter_mut().chain(next.iter_mut()).enumerate() {
                poison(f, k);
            }
            (prev, next)
        }
        // dx grows by one step per row (8 px at the Max Displace of 1016 used below), dy is exactly 0
        "shear" => (
            (0..8).map(|_| coded(depth, w, h, |_, y| (-(y as i32), 0))).collect(),
            (0..8).map(|_| coded(depth, w, h, |_, y| (y as i32, 0))).collect(),
        ),
        _ => (
            (0..8).map(|_| coded(depth, w, h, |_, _| (-3, 2))).collect(),
            (0..8).map(|_| coded(depth, w, h, |_, _| (3, -2))).collect(),
        ),
    }
}

fn modes() -> Vec<Vec<(&'static str, &'static str)>> {
    let mut out = Vec::new();
    for warping in ["1", "2"] {
        for interp in ["1", "2", "3"] {
            for blur in ["0", "2.5"] {
                for displace in ["1016", "2048"] {
                    out.push(vec![
                        ("speed_percent", "37.5"),
                        ("warping", warping),
                        ("frame_interp", interp),
                        ("blur_compensation", blur),
                        ("vectors_max_displace", displace),
                    ]);
                }
            }
        }
        out.push(vec![("speed_percent", "37.5"), ("warping", warping), ("frame_interp", "2"), ("vec_scale_x", "-2"), ("vec_scale_y", "3")]);
    }
    for display in ["2", "3", "6"] {
        out.push(vec![("speed_percent", "37.5"), ("display", display), ("warping", "2")]);
    }
    out
}

/// The same parameters with the segment marker that turns the vectors off and keeps the
/// interpolation: "No MV, Use Nearest" for Nearest, "No MV, Frame Blend" otherwise.
fn no_vectors(params: &[(&'static str, &'static str)]) -> Vec<(&'static str, &'static str)> {
    let nearest = params.iter().any(|&(id, v)| id == "frame_interp" && v == "1");
    let mut p = params.to_vec();
    p.push(("src_mark_segments", if nearest { "4" } else { "5" }));
    p
}

fn finite(clip: &[Frame]) -> bool {
    clip.iter().all(|f| match &f.pixels {
        Pixels::F32(px) => px.iter().flatten().all(|v| v.is_finite()),
        _ => true,
    })
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
            let side_of_one = w < 2 || h < 2;
            for kind in ["random", "shear", "constant"] {
                // the source is a plain random clip for the shear and constant kinds, so the No
                // Vectors comparison of Weighted Blend (a plain blend only for finite samples) holds
                let clip = if kind == "random" { tiny_clip(depth, w, h) } else { random_clip(depth, w, h, 4, 5) };
                let (prev, next) = vector_layers(kind, depth, w, h);
                for layers in ["both", "prev", "next"] {
                    for params in modes() {
                        let build = |p: &[(&'static str, &'static str)]| {
                            let mut rig = Rig::new(clip.clone(), p).with_frames(FRAMES);
                            if layers != "next" {
                                rig = rig.with_layer("vectors_prev", prev.clone());
                            }
                            if layers != "prev" {
                                rig = rig.with_layer("vectors_next", next.clone());
                            }
                            rig
                        };
                        let rig = build(&params);
                        let weighted = params.iter().any(|&(id, v)| id == "frame_interp" && v == "3");
                        let compare = side_of_one && (!weighted || finite(&clip));
                        let plain = build(&no_vectors(&params));
                        for quality in [Quality::High, Quality::Draft] {
                            for n in 0..FRAMES {
                                let case = format!("{depth:?} {w}x{h} {kind} {layers} {quality:?} frame {n} {params:?}");
                                match render(&rig, n, quality) {
                                    Ok(got) => {
                                        renders += 1;
                                        if compare {
                                            let want = render(&plain, n, quality).expect("the marker render works");
                                            if got.to_bytes() != want.to_bytes() {
                                                failures.push(format!("{case}: differs from the render without vectors"));
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
        }
    }
    assert!(failures.is_empty(), "{} failures in {} renders:\n{}", failures.len(), renders + failures.len(), failures.join("\n"));
}

/// NaN vector samples make NaN flows, which the inverse warp and the motion blur refuse by name.
/// That refusal does not depend on the size: a 2x2 clip meets it exactly like a 16x16 clip (vectors-in
/// never searches motion, it takes its flows from the layers, and its vector path rendered 2x2
/// frames before the tiny frames rule too), and a frame with a side of 1 never reads the vector
/// layers (since the rule it takes the fast path).
#[test]
fn nan_vector_samples_are_refused_at_every_size_with_a_vector_path() {
    let params = [("speed_percent", "37.5"), ("warping", "1"), ("frame_interp", "2")];
    let outcome = |w: usize, h: usize| -> Vec<Result<(), String>> {
        let (prev, next) = vector_layers("nan", Depth::F32, w, h);
        let rig = Rig::new(random_clip(Depth::F32, w, h, 4, 5), &params)
            .with_frames(FRAMES)
            .with_layer("vectors_prev", prev)
            .with_layer("vectors_next", next);
        (0..FRAMES).map(|n| rig.render(n).map(|_| ()).map_err(|e| e.to_string())).collect()
    };
    let big = outcome(16, 16);
    assert!(big.iter().any(|r| r.as_ref().is_err_and(|m| m.contains("NaN"))), "{big:?}");
    let small = outcome(2, 2);
    assert!(small.iter().any(|r| r.as_ref().is_err_and(|m| m.contains("NaN"))), "{small:?}");
    for r in small.iter().chain(&big) {
        if let Err(m) = r {
            assert!(m.starts_with("unsupported: ") && m.contains("NaN"), "{m}");
        }
    }
    for (w, h) in [(1usize, 17usize), (17, 1), (1, 1)] {
        assert!(outcome(w, h).iter().all(|r| r.is_ok()), "{w}x{h}");
    }
}

/// The render fetches only what the dependencies announce, at the tiny sizes 1x4 (no vector path)
/// and 2x2 (the vector path) too.
#[test]
fn tiny_frames_fetch_only_what_the_dependencies_announce() {
    use fx_core::{Checkout, Effect, Env, LayerSet, TemporalContext};
    use vectors_in::VectorsIn;
    let mut cases = 0;
    for (w, h) in [(1usize, 4usize), (2, 2)] {
        for layers in ["both", "prev", "next"] {
            for params in modes() {
                for n in 0..FRAMES {
                    let input = Recorder::new(random_clip(Depth::U8, w, h, 4, 5));
                    let (p, q) = vector_layers("shear", Depth::U8, w, h);
                    let (prev, next) = (Recorder::new(p), Recorder::new(q));
                    let mut set = LayerSet::new();
                    if layers != "next" {
                        set.insert("vectors_prev", prev.clone());
                    }
                    if layers != "prev" {
                        set.insert("vectors_next", next.clone());
                    }
                    let c = curves(&params);
                    let env = Env { params: &c, input: &*input, layers: &set };
                    let cx = TemporalContext::new(w, h, Depth::U8, time(), n, FRAMES);
                    let deps = VectorsIn.dependencies(&cx, &env).unwrap();
                    let mut out = Frame::new(w, h, Depth::U8);
                    VectorsIn.render_temporal(&cx, &env, &mut out).unwrap();
                    for (layer, rec) in [(None, &input), (Some("vectors_prev"), &prev), (Some("vectors_next"), &next)] {
                        for t in rec.take() {
                            assert!(deps.contains(&Checkout { layer, time: t }), "{w}x{h} {layers} frame {n} {params:?}: {layer:?}@{t} not in {deps:?}");
                        }
                    }
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 2 * 3 * modes().len() * FRAMES as usize);
}
