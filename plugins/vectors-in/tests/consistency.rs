//! The render only fetches what the pre-render announced: every `(layer, time)` the render asks the
//! layers for is in `Effect::dependencies` of the same output frame. Run over many parameter sets,
//! layer combinations and output frames with recording layers (input, Source, Vectors: Prev,
//! Vectors: Next). Adapted from plugin 03's test: the vector path takes its flows from the layers
//! (no solver, so small clips are fine), there is no Src Time Offset, no tracking layer and no
//! fields.

mod common;

use common::moving::*;
use common::*;
use fx_core::{Checkout, Depth, Effect, Env, Frame, FxError, LayerSet};
use vectors_in::VectorsIn;
use std::sync::Arc;

const W: usize = 8;
const H: usize = 6;
const MD: f32 = 16.0;

fn clip() -> Vec<Frame> {
    DEFAULT.clip(Depth::U8, W, H, 8)
}

fn timings() -> Vec<Vec<(&'static str, &'static str)>> {
    vec![
        vec![],
        vec![("speed_percent", "50")],
        vec![("speed_percent", "25")],
        vec![("speed_percent", "200")],
        vec![("speed_percent", "33.3")],
        vec![("speed_percent", "0")],
        vec![("speed_percent", "-100")],
        vec![("speed_percent", "-50")],
        vec![("speed_percent", "100@0;50@4;200@8")],
        vec![("speed_percent", "75")],
        vec![("speed_percent", "75"), ("blur_compensation", "3")],
        vec![("time_remap_mode", "2"), ("frame_num", "0")],
        vec![("time_remap_mode", "2"), ("frame_num", "2.5")],
        vec![("time_remap_mode", "2"), ("frame_num", "-0.5")],
        vec![("time_remap_mode", "2"), ("frame_num", "0@0;7@8")],
        vec![("in_fps_is_out_fps", "0"), ("input_frame_rate", "24"), ("speed_percent", "100")],
        vec![("in_fps_is_out_fps", "0"), ("input_frame_rate", "24"), ("speed_percent", "40")],
        vec![("in_fps_is_out_fps", "0"), ("input_frame_rate", "12"), ("speed_percent", "100")],
        vec![("in_fps_is_out_fps", "0"), ("input_frame_rate", "60"), ("speed_percent", "100")],
        vec![("vec_scale_x", "1@0;3@8"), ("vectors_max_displace", "16@0;32@8"), ("speed_percent", "60")],
    ]
}

fn scenarios() -> Vec<Vec<(&'static str, &'static str)>> {
    let mut out = Vec::new();
    for t in timings() {
        for warping in ["1", "2"] {
            for interp in ["1", "2", "3"] {
                let mut p = t.clone();
                p.push(("warping", warping));
                p.push(("frame_interp", interp));
                out.push(p);
            }
        }
        // the Source display and an FG display (both ignore the plan or take it like the output)
        for display in ["2", "4"] {
            let mut p = t.clone();
            p.push(("display", display));
            p.push(("warping", "1"));
            out.push(p);
        }
        // a segment marker that cuts the clip at output frame 3, and the two No MV modes
        let mut p = t.clone();
        p.push(("src_mark_segments", "1@0;2@3;3@5"));
        p.push(("warping", "1"));
        out.push(p);
        let mut p = t.clone();
        p.push(("src_mark_segments", "1@0;4@3;5@6"));
        out.push(p);
    }
    out
}

struct Layers {
    input: Arc<Recorder>,
    source: Arc<Recorder>,
    prev: Arc<Recorder>,
    next: Arc<Recorder>,
}

impl Layers {
    fn new() -> Layers {
        let (prev, next) = motion_vectors(Depth::U8, W, H, 16, 3.4, 1.7, MD);
        Layers { input: Recorder::new(clip()), source: Recorder::new(clip()), prev: Recorder::new(prev), next: Recorder::new(next) }
    }
}

/// Asks the render and the dependencies for output frame `n` and compares them. Returns the number of vector layer
/// fetches at frame times (not the probes at `current_time`).
fn check(params: &[(&str, &str)], with_source: bool, with_prev: bool, with_next: bool, n: u64) -> Result<usize, String> {
    let l = Layers::new();
    let mut layers = LayerSet::new();
    if with_source {
        layers.insert("source", l.source.clone());
    }
    if with_prev {
        layers.insert("vectors_prev", l.prev.clone());
    }
    if with_next {
        layers.insert("vectors_next", l.next.clone());
    }
    let curves = curves(params);
    let env = Env { params: &curves, input: &*l.input, layers: &layers };
    let cx = fx_core::TemporalContext::new(W, H, Depth::U8, time(), n, 12);
    let deps = VectorsIn.dependencies(&cx, &env).map_err(|e| format!("dependencies: {e}"))?;
    let mut out = Frame::new(W, H, Depth::U8);
    match VectorsIn.render_temporal(&cx, &env, &mut out) {
        Ok(()) => {}
        // the forward warp and the flow inversion are engine features that may be missing; the fetches made before the
        // refusal still have to be announced
        Err(FxError::Unsupported(_)) => {}
        Err(e) => return Err(format!("render: {e}")),
    }
    let mut vector_fetches = 0;
    for (name, rec, layer) in [
        ("input", &l.input, None),
        ("source", &l.source, Some("source")),
        ("vectors_prev", &l.prev, Some("vectors_prev")),
        ("vectors_next", &l.next, Some("vectors_next")),
    ] {
        for t in rec.take() {
            if name.starts_with("vectors") && t != 100 * n as i64 {
                vector_fetches += 1;
            }
            if !deps.contains(&Checkout { layer, time: t }) {
                return Err(format!("{name} fetched at {t} but dependencies are {deps:?}"));
            }
        }
    }
    Ok(vector_fetches)
}

#[test]
fn every_fetch_of_the_render_was_announced_by_the_dependencies() {
    let mut cases = 0;
    let mut with_vector_fetches = 0;
    for params in scenarios() {
        for (with_source, with_prev, with_next) in [(false, true, true), (true, true, true), (false, true, false), (false, false, true)] {
            for n in 0..10 {
                match check(&params, with_source, with_prev, with_next, n) {
                    Ok(v) => with_vector_fetches += usize::from(v > 0),
                    Err(e) => panic!("{params:?} source {with_source} prev {with_prev} next {with_next} frame {n}: {e}"),
                }
                cases += 1;
            }
        }
    }
    assert!(cases > 5_000, "{cases}");
    // the vector path really ran in a good share of the cases (the others are exact frames, markers and refusals)
    assert!(with_vector_fetches > 500, "{with_vector_fetches}");
}

#[test]
fn the_fg_matte_probes_and_the_layers_without_any_vector_layer_are_announced_too() {
    // no vector layers: the render is the red fill, still only the probes are fetched
    for n in 0..4 {
        assert_eq!(check(&[("speed_percent", "50")], false, false, false, n).unwrap(), 0);
    }
    // an FG matte layer (refusal): the matte is only probed by size, nothing is fetched from it
    let l = Layers::new();
    let mut layers = LayerSet::new();
    layers.insert("vectors_prev", l.prev.clone());
    layers.insert("vectors_next", l.next.clone());
    layers.insert("fg1_matte", Arc::new(layer(clip())));
    let curves = curves(&[("warping", "1")]);
    let env = Env { params: &curves, input: &*l.input, layers: &layers };
    let cx = fx_core::TemporalContext::new(W, H, Depth::U8, time(), 1, 12);
    assert!(VectorsIn.dependencies(&cx, &env).is_err());
    let mut out = Frame::new(W, H, Depth::U8);
    assert!(VectorsIn.render_temporal(&cx, &env, &mut out).is_err());
    assert!(l.input.take().is_empty() && l.prev.take().len() <= 1);
}
