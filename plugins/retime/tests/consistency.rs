//! The render only fetches what `Effect::dependencies` announced: every `(layer, time)` the render
//! asks the layers for is in the dependency list of the same output frame. Run over many parameter
//! sets and output frames, with recording layers.

mod common;

use common::*;
use fx_core::{Checkout, Depth, Effect, Env, Frame, FxError, Layer, LayerSet};
use retime::Retime;
use std::sync::{Arc, Mutex};

/// A layer that logs the times it is asked for.
struct Recorder {
    inner: fx_core::MemLayer,
    log: Mutex<Vec<i64>>,
}

impl Recorder {
    fn new(frames: Vec<Frame>) -> Arc<Recorder> {
        Arc::new(Recorder { inner: layer(frames), log: Mutex::new(Vec::new()) })
    }

    fn take(&self) -> Vec<i64> {
        std::mem::take(&mut self.log.lock().unwrap())
    }
}

impl Layer for Recorder {
    fn width(&self) -> usize {
        self.inner.width()
    }
    fn height(&self) -> usize {
        self.inner.height()
    }
    fn depth(&self) -> Depth {
        self.inner.depth()
    }
    fn frame_at(&self, time: i64) -> fx_core::Result<Option<Arc<Frame>>> {
        self.log.lock().unwrap().push(time);
        self.inner.frame_at(time)
    }
}

/// The clip sizes: 5 x 4, and the tiny frames 1 x 4 (a side of 1: no search, the No Vectors render) and
/// 2 x 2 (searched at its own size).
const SIZES: [(usize, usize); 3] = [(5, 4), (1, 4), (2, 2)];

fn clip(w: usize, h: usize) -> Vec<Frame> {
    random_clip(Depth::U8, w, h, 8, 7)
}

/// Scenario parameter lists: speed or frame number, offsets, vectors, interpolation, display, fields, rates.
fn scenarios() -> Vec<Vec<(&'static str, &'static str)>> {
    let mut out = Vec::new();
    let timing: Vec<Vec<(&str, &str)>> = vec![
        vec![],
        vec![("speed_percent", "50")],
        vec![("speed_percent", "25")],
        vec![("speed_percent", "200")],
        vec![("speed_percent", "33.3")],
        vec![("speed_percent", "0")],
        vec![("speed_percent", "-100"), ("src_time_offset", "5")],
        vec![("speed_percent", "-50"), ("src_time_offset", "5")],
        vec![("speed_percent", "100@0;50@4;200@8")],
        vec![("speed_percent", "75"), ("src_time_offset", "2.5")],
        vec![("time_remap_mode", "2"), ("frame_num", "0")],
        vec![("time_remap_mode", "2"), ("frame_num", "2.5")],
        vec![("time_remap_mode", "2"), ("frame_num", "-0.5")],
        vec![("time_remap_mode", "2"), ("frame_num", "0@0;7@8")],
        vec![("in_fps_is_out_fps", "0"), ("input_frame_rate", "24"), ("speed_percent", "100")],
        vec![("in_fps_is_out_fps", "0"), ("input_frame_rate", "24"), ("speed_percent", "40")],
        vec![("input_fields", "2"), ("speed_percent", "50")],
    ];
    for t in &timing {
        for vectors in ["1", "3", "5"] {
            for interp in ["1", "3"] {
                let mut p = t.clone();
                p.push(("vectors", vectors));
                p.push(("frame_interp", interp));
                out.push(p);
            }
        }
        // the Source display
        let mut p = t.clone();
        p.push(("display", "2"));
        out.push(p);
        // a segment marker that cuts the clip at output frame 3
        let mut p = t.clone();
        p.push(("src_mark_segments", "1@0;2@3;3@5"));
        out.push(p);
    }
    out
}

/// Asks the render and the dependencies for output frame `n` and compares them.
fn check(params: &[(&str, &str)], with_alt: bool, with_b: bool, n: u64, (w, h): (usize, usize)) -> Result<(), String> {
    let input = Recorder::new(clip(w, h));
    let alt = Recorder::new(clip(w, h));
    let tracking = Recorder::new(clip(w, h));
    let mut layers = LayerSet::new();
    if with_alt {
        layers.insert("alt_source_a", alt.clone());
    }
    if with_b {
        layers.insert("alt_source_b", tracking.clone());
    }
    let curves = curves(params);
    let env = Env { params: &curves, input: &*input, layers: &layers };
    let cx = fx_core::TemporalContext::new(w, h, Depth::U8, time(), n, 12);
    let deps = Retime.dependencies(&cx, &env).map_err(|e| format!("dependencies: {e}"))?;
    let mut out = Frame::new(w, h, Depth::U8);
    // every size renders: the search never halves below 2 x 2 and a side of 1 is not searched
    Retime.render_temporal(&cx, &env, &mut out).map_err(|e| format!("render: {e}"))?;
    for (name, rec, layer) in [("input", &input, None), ("alt_source_a", &alt, Some("alt_source_a")), ("alt_source_b", &tracking, Some("alt_source_b"))] {
        for t in rec.take() {
            let wanted = Checkout { layer, time: t };
            // the source layer is the input unless Alt Color Source is set; fetches go through the dependencies of that layer
            if !deps.contains(&wanted) {
                return Err(format!("{name} fetched at {t} but dependencies are {deps:?}"));
            }
        }
    }
    Ok(())
}

#[test]
fn every_fetch_of_the_render_was_announced_by_the_dependencies() {
    let mut cases = 0;
    for params in scenarios() {
        for (with_alt, with_b) in [(false, false), (true, false), (false, true), (true, true)] {
            for size in SIZES {
                for n in 0..12 {
                    if let Err(e) = check(&params, with_alt, with_b, n, size) {
                        panic!("{params:?} alt {with_alt} tracking {with_b} {size:?} frame {n}: {e}");
                    }
                    cases += 1;
                }
            }
        }
    }
    assert!(cases > 18_000, "{cases}");
}

/// A clip with real motion, large enough for the solver, so that the vector path runs to the end.
fn moving_clip() -> Vec<Frame> {
    common::moving::DEFAULT.clip(Depth::U8, 32, 24, 8)
}

/// Like [`check`], on a clip the solver can work on. The render must succeed; the number of tracking layer fetches
/// is returned (an interpolated frame with Alt Motion Source set fetches it twice, an exact one not at all).
fn check_vector(params: &[(&str, &str)], with_alt: bool, with_b: bool, n: u64) -> Result<(usize, usize), String> {
    let input = Recorder::new(moving_clip());
    let alt = Recorder::new(moving_clip());
    let tracking = Recorder::new(common::moving::Motion { vx: -2.0, vy: 1.0 }.clip(Depth::U8, 32, 24, 8));
    let mut layers = LayerSet::new();
    if with_alt {
        layers.insert("alt_source_a", alt.clone());
    }
    if with_b {
        layers.insert("alt_source_b", tracking.clone());
    }
    let curves = curves(params);
    let env = Env { params: &curves, input: &*input, layers: &layers };
    let cx = fx_core::TemporalContext::new(32, 24, Depth::U8, time(), n, 12);
    let deps = Retime.dependencies(&cx, &env).map_err(|e| format!("dependencies: {e}"))?;
    let mut out = Frame::new(32, 24, Depth::U8);
    Retime.render_temporal(&cx, &env, &mut out).map_err(|e| format!("render: {e}"))?;
    let mut fetched = (0, 0);
    for (name, rec, layer) in [("input", &input, None), ("alt_source_a", &alt, Some("alt_source_a")), ("alt_source_b", &tracking, Some("alt_source_b"))] {
        let log = rec.take();
        if name == "alt_source_b" {
            fetched.1 = log.len();
        } else {
            fetched.0 += log.len();
        }
        for t in log {
            if !deps.contains(&(Checkout { layer, time: t })) {
                return Err(format!("{name} fetched at {t} but dependencies are {deps:?}"));
            }
        }
    }
    Ok(fetched)
}

#[test]
fn the_vector_path_fetches_only_what_the_dependencies_announced_including_the_tracking_layer() {
    let timings: Vec<Vec<(&str, &str)>> = vec![
        vec![("speed_percent", "50")],
        vec![("speed_percent", "150"), ("blur_compensation", "1")],
        vec![("speed_percent", "-100"), ("src_time_offset", "6")],
        vec![("speed_percent", "100@0;40@4"), ("blur_compensation", "0.5")],
        vec![("speed_percent", "75"), ("src_time_offset", "2.5")],
        vec![("time_remap_mode", "2"), ("frame_num", "0@0;7@9")],
        vec![("in_fps_is_out_fps", "0"), ("input_frame_rate", "24"), ("speed_percent", "40")],
        vec![("speed_percent", "50"), ("src_mark_segments", "1@0;2@3;3@5")],
    ];
    let (mut cases, mut vector_renders, mut tracked) = (0, 0, 0);
    for timing in &timings {
        for (vectors, interp) in [("5", "1"), ("5", "2"), ("5", "3"), ("2", "2")] {
            let mut params = timing.clone();
            params.push(("vectors", vectors));
            params.push(("frame_interp", interp));
            for (with_alt, with_b) in [(false, false), (true, false), (false, true), (true, true)] {
                for n in 0..9 {
                    let (main, track) = check_vector(&params, with_alt, with_b, n)
                        .unwrap_or_else(|e| panic!("{params:?} alt {with_alt} tracking {with_b} frame {n}: {e}"));
                    cases += 1;
                    // a vector render fetches the main frames twice (the pair) and, with the layer set, the tracking pair
                    if main >= 2 && !with_b {
                        vector_renders += 1;
                    }
                    if with_b && track > 0 {
                        assert_eq!(track, 2, "{params:?} frame {n}: the tracking layer is fetched as a pair");
                        assert!(main >= 2, "{params:?} frame {n}: tracking frames come with the main pair");
                        tracked += 1;
                    }
                }
            }
        }
    }
    assert!(cases > 1_000 && vector_renders > 100 && tracked > 100, "{cases} {vector_renders} {tracked}");
}
