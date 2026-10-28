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

/// The clip sizes: 12 x 10, and the tiny frames 1 x 4 (a side of 1: the zero vector) and 2 x 2 (searched at
/// its own size).
const SIZES: [(usize, usize); 3] = [(12, 10), (1, 4), (2, 2)];

fn clip() -> Vec<Frame> {
    clip_of(12, 10)
}

fn clip_of(w: usize, h: usize) -> Vec<Frame> {
    random_clip(Depth::U8, w, h, 10, 7)
}

/// One run: returns (the dependencies, the fetches of every layer id, whether the render succeeded).
fn run(params: &[(&str, &str)], with_alt: bool, fg: usize, n: u64, base: TimeBase) -> (Vec<Checkout>, Vec<(Option<&'static str>, i64)>, bool) {
    run_at(params, with_alt, fg, n, base, (12, 10))
}

/// [`run`] on a clip of `w x h`.
fn run_at(
    params: &[(&str, &str)],
    with_alt: bool,
    fg: usize,
    n: u64,
    base: TimeBase,
    (w, h): (usize, usize),
) -> (Vec<Checkout>, Vec<(Option<&'static str>, i64)>, bool) {
    let input = Recorder::new(clip_of(w, h));
    let alt = Recorder::new(clip_of(w, h));
    let fgs: Vec<Arc<Recorder>> = (0..3).map(|_| Recorder::new(clip_of(w, h))).collect();
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
    let cx = TemporalContext::new(w, h, Depth::U8, base, n, 12);
    let deps = CreateVectors.dependencies(&cx, &env).unwrap();
    let mut out = Frame::new(w, h, Depth::U8);
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
                                for size in SIZES {
                                    let params = [("display", display), ("track_frame", track), ("vector_resolution", res), ("prep", prep), ("max_displace", "16")];
                                    let (deps, fetched, _) = run_at(&params, with_alt, fg, n, base, size);
                                    for (layer, time) in &fetched {
                                        assert!(
                                            deps.contains(&Checkout { layer: *layer, time: *time }),
                                            "display {display} track {track} alt {with_alt} fg {fg} {size:?} frame {n}: fetch {layer:?}@{time} not announced in {deps:?}"
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
    }
    assert_eq!(cases, 2 * 3 * 2 * 3 * 2 * 3 * 5 * 3);
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
fn layered_debug_and_tracker_renders_fetch_the_pair_then_the_mattes() {
    // FG mattes, Debug and Track off render (they were refused before any fetch): the source at t_a
    // then t_b, then every active matte at t_a and t_b, FG1 first, all announced
    let base = TimeBase::default_for_rate(Rational::new(30, 1).unwrap());
    let cases: [(Vec<(&str, &str)>, usize, (i64, i64)); 5] = [
        (vec![("max_displace", "16")], 1, (400, 300)),
        (vec![("max_displace", "16"), ("layers_output", "3")], 2, (400, 300)),
        (vec![("display", "3")], 0, (500, 400)),
        (vec![("display", "3"), ("track_frame", "2")], 2, (300, 400)),
        (vec![("main_bg_track", "0"), ("pt1_use", "5")], 0, (400, 300)),
    ];
    for (params, fg, (ta, tb)) in cases {
        let (deps, fetched, ok) = run(&params, false, fg, 4, base);
        assert!(ok, "{params:?}");
        let mut want = vec![(None, ta), (None, tb)];
        for id in ["fg1_matte", "fg2_matte", "fg3_matte"].iter().take(fg) {
            want.push((Some(*id), ta));
            want.push((Some(*id), tb));
        }
        assert_eq!(fetched, want, "{params:?} fg {fg}");
        for (layer, time) in &fetched {
            assert!(deps.contains(&Checkout { layer: *layer, time: *time }));
        }
    }
}
