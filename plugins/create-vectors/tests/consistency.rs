//! The render only fetches what `Effect::dependencies` announced: every `(layer, time)` the render
//! asks the layers for is in the dependency list of the same output frame. Run with recording
//! layers over Display, Track Frame, Alt Src, FG mattes, Vector Resolution, Prep, the first and
//! last frames and two time bases. Beyond the subset relation the exact fetches are pinned: the
//! vector display fetches the source at `t_a` then `t_b`, nothing else; the Source display the
//! source at `current_time`, nothing else.

mod common;

use common::*;
use fx_core::{Checkout, Depth, Effect, Env, Frame, Layer, LayerSet, Rational, TemporalContext, TimeBase};
use create_vectors::CreateVectors;
use std::sync::{Arc, Mutex};

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

fn clip() -> Vec<Frame> {
    random_clip(Depth::U8, 12, 10, 10, 7)
}

/// One run: returns (the dependencies, the fetches of every layer id, whether the render succeeded).
fn run(params: &[(&str, &str)], with_alt: bool, fg: usize, n: u64, base: TimeBase) -> (Vec<Checkout>, Vec<(Option<&'static str>, i64)>, bool) {
    let input = Recorder::new(clip());
    let alt = Recorder::new(clip());
    let fgs: Vec<Arc<Recorder>> = (0..3).map(|_| Recorder::new(clip())).collect();
    let mut layers = LayerSet::new();
    if with_alt {
        layers.insert("alt_source_a", alt.clone());
    }
    let ids = ["fg1_matte", "fg2_matte", "fg3_matte"];
    for k in 0..fg {
        layers.insert(ids[k], fgs[k].clone());
    }
    let curves = curves(params);
    let env = Env { params: &curves, input: &*input, layers: &layers };
    let cx = TemporalContext::new(12, 10, Depth::U8, base, n, 12);
    let deps = CreateVectors.dependencies(&cx, &env).unwrap();
    let mut out = Frame::new(12, 10, Depth::U8);
    let ok = CreateVectors.render_temporal(&cx, &env, &mut out).is_ok();
    let mut fetched = Vec::new();
    for t in input.take() {
        fetched.push((None, t));
    }
    for t in alt.take() {
        fetched.push((Some("alt_source_a"), t));
    }
    for (k, f) in fgs.iter().enumerate() {
        for t in f.take() {
            fetched.push((Some(ids[k]), t));
        }
    }
    (deps, fetched, ok)
}

#[test]
fn every_fetch_of_the_render_is_in_the_dependencies() {
    let bases = [TimeBase::default_for_rate(Rational::new(30, 1).unwrap()), TimeBase::default_for_rate(Rational::new(30000, 1001).unwrap())];
    let mut cases = 0;
    for base in bases {
        for display in ["1", "2", "3"] {
            for track in ["1", "2"] {
                for (res, prep) in [("1", "1"), ("2", "2"), ("1", "3")] {
                    for with_alt in [false, true] {
                        for fg in 0..3 {
                            for n in [0u64, 1, 5, 9, 11] {
                                let params = [("display", display), ("track_frame", track), ("vector_resolution", res), ("prep", prep), ("max_displace", "16")];
                                let (deps, fetched, _) = run(&params, with_alt, fg, n, base);
                                for (layer, time) in &fetched {
                                    assert!(
                                        deps.contains(&Checkout { layer: *layer, time: *time }),
                                        "display {display} track {track} alt {with_alt} fg {fg} frame {n}: fetch {layer:?}@{time} not announced in {deps:?}"
                                    );
                                }
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases, 2 * 3 * 2 * 3 * 2 * 3 * 5);
}

#[test]
fn the_vector_display_fetches_the_source_at_t_a_then_t_b_and_nothing_else() {
    let base = TimeBase::default_for_rate(Rational::new(30, 1).unwrap());
    for (track, other) in [("1", -1i64), ("2", 1)] {
        for with_alt in [false, true] {
            let (deps, fetched, ok) = run(&[("track_frame", track), ("max_displace", "16")], with_alt, 0, 4, base);
            assert!(ok);
            let src = with_alt.then_some("alt_source_a");
            assert_eq!(fetched, vec![(src, 400), (src, 400 + 100 * other)], "track {track} alt {with_alt}");
            // the same two entries close the dependencies
            let n = deps.len();
            assert_eq!(&deps[n - 2..], &[Checkout { layer: src, time: 400 }, Checkout { layer: src, time: 400 + 100 * other }]);
        }
    }
}

#[test]
fn the_source_display_fetches_the_source_at_the_current_time_only() {
    let base = TimeBase::default_for_rate(Rational::new(30, 1).unwrap());
    for with_alt in [false, true] {
        let (deps, fetched, ok) = run(&[("display", "1")], with_alt, 0, 6, base);
        assert!(ok);
        let src = with_alt.then_some("alt_source_a");
        assert_eq!(fetched, vec![(src, 600)]);
        assert_eq!(deps.last(), Some(&Checkout { layer: src, time: 600 }));
    }
}

#[test]
fn a_refused_render_fetches_no_pixels() {
    // FG matte, Debug, Track off: refused before any frame is fetched
    let base = TimeBase::default_for_rate(Rational::new(30, 1).unwrap());
    for (params, fg) in [(vec![("max_displace", "16")], 1), (vec![("display", "3")], 0), (vec![("main_bg_track", "0")], 0)] {
        let (_, fetched, ok) = run(&params, false, fg, 4, base);
        assert!(!ok);
        assert!(fetched.is_empty(), "{fetched:?}");
    }
}
