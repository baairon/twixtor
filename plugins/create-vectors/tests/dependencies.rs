//! `Effect::dependencies` against hand worked examples. Time base: 30 fps with 100 ticks per frame,
//! so output frame `n` is `current_time = 100 n`,
//! `frame = (u32)(current_time / time_step + 0.5) = n`, `t(k) = rnd(k * 100)`.
//!
//! The list of one frame, in issue order (`X` is `current_time`; the layers are written as `L0` for
//! the input layer, `Alt` for `Alt Src`, `FG1` to `FG3` for the matte layers):
//!
//! 1. `L0@X`, `Alt@X`, `L0@X` again when `Alt` is empty, then `FG1@X`, `FG2@X`, `FG3@X` up to and
//!    including the first empty one;
//! 2. `Display == Source`: the source at `X` once more and nothing else;
//! 3. else the source at `t_a` and `t_b`, then `FG k` at `t_a` and `t_b` for every active FG layer.
//!
//! The pair is `(a, b) = (n, n - 1)` (Previous) or `(n, n + 1)` (Next); Debug swaps it to
//! `(n + 1, n)` and `(n - 1, n)`.

mod common;

use common::*;
use fx_core::{Checkout, Depth, Effect, Frame, Rational, TemporalContext, TimeBase};
use create_vectors::CreateVectors;

const ALT: &str = "alt_source_a";
const FG: [&str; 3] = ["fg1_matte", "fg2_matte", "fg3_matte"];

fn co(layer: Option<&'static str>, time: i64) -> Checkout {
    Checkout { layer, time }
}

fn clip() -> Vec<Frame> {
    random_clip(Depth::U8, 5, 4, 12, 3)
}

fn rig(params: &[(&str, &str)]) -> Rig {
    Rig::new(clip(), params).with_frames(12)
}

#[test]
fn vectors_previous_reads_the_source_at_frame_and_frame_minus_one() {
    // frame 7, Display Vectors (default), Track Frame Previous (default), no Alt Src, no FG: 4 probes and 2 fetches
    assert_eq!(
        rig(&[]).deps(7).unwrap(),
        vec![co(None, 700), co(Some(ALT), 700), co(None, 700), co(Some(FG[0]), 700), co(None, 700), co(None, 600)]
    );
}

#[test]
fn vectors_next_reads_the_source_at_frame_and_frame_plus_one() {
    assert_eq!(
        rig(&[("track_frame", "2")]).deps(7).unwrap(),
        vec![co(None, 700), co(Some(ALT), 700), co(None, 700), co(Some(FG[0]), 700), co(None, 700), co(None, 800)]
    );
}

#[test]
fn debug_swaps_the_roles_of_the_pair() {
    // Debug, Previous: (a, b) = (8, 7); Debug, Next: (6, 7)
    let probes = vec![co(None, 700), co(Some(ALT), 700), co(None, 700), co(Some(FG[0]), 700)];
    let mut previous = probes.clone();
    previous.extend([co(None, 800), co(None, 700)]);
    assert_eq!(rig(&[("display", "3"), ("track_frame", "1")]).deps(7).unwrap(), previous);
    let mut next = probes;
    next.extend([co(None, 600), co(None, 700)]);
    assert_eq!(rig(&[("display", "3"), ("track_frame", "2")]).deps(7).unwrap(), next);
}

#[test]
fn the_source_display_needs_the_current_time_only() {
    // Display Source: the probes and the source at X once more; Track Frame does not matter
    let want = vec![co(None, 700), co(Some(ALT), 700), co(None, 700), co(Some(FG[0]), 700), co(None, 700)];
    for track in ["1", "2"] {
        assert_eq!(rig(&[("display", "1"), ("track_frame", track)]).deps(7).unwrap(), want);
    }
}

#[test]
fn alt_src_replaces_the_input_as_the_source() {
    // Alt Src has pixels: no second L0 probe, the pair is read from Alt Src
    let r = rig(&[]).with_layer(ALT, clip());
    assert_eq!(
        r.deps(7).unwrap(),
        vec![co(None, 700), co(Some(ALT), 700), co(Some(FG[0]), 700), co(Some(ALT), 700), co(Some(ALT), 600)]
    );
    // and for the Source display: Alt Src at X once more
    let r = rig(&[("display", "1")]).with_layer(ALT, clip());
    assert_eq!(r.deps(7).unwrap(), vec![co(None, 700), co(Some(ALT), 700), co(Some(FG[0]), 700), co(Some(ALT), 700)]);
}

#[test]
fn an_empty_alt_src_layer_is_the_same_as_none() {
    // a layer of size 0 (width or height 0) has no pixels: the source falls back to layer 0
    let empty = vec![Frame::new(0, 0, Depth::U8)];
    let r = rig(&[]).with_layer(ALT, empty);
    assert_eq!(r.deps(7).unwrap(), rig(&[]).deps(7).unwrap());
}

#[test]
fn active_fg_mattes_are_read_at_both_times_in_order() {
    // FG1 and FG2 set: the probes stop at FG3 (the first empty one, included), then FG1 and FG2 at t_a and t_b
    let r = rig(&[]).with_layer(FG[0], clip()).with_layer(FG[1], clip());
    assert_eq!(
        r.deps(7).unwrap(),
        vec![
            co(None, 700),
            co(Some(ALT), 700),
            co(None, 700),
            co(Some(FG[0]), 700),
            co(Some(FG[1]), 700),
            co(Some(FG[2]), 700),
            co(None, 700),
            co(None, 600),
            co(Some(FG[0]), 700),
            co(Some(FG[0]), 600),
            co(Some(FG[1]), 700),
            co(Some(FG[1]), 600),
        ]
    );
    // all three, Next
    let r = rig(&[("track_frame", "2")]).with_layer(FG[0], clip()).with_layer(FG[1], clip()).with_layer(FG[2], clip());
    let d = r.deps(7).unwrap();
    assert_eq!(d.len(), 6 + 2 + 6);
    assert_eq!(&d[..6], &[co(None, 700), co(Some(ALT), 700), co(None, 700), co(Some(FG[0]), 700), co(Some(FG[1]), 700), co(Some(FG[2]), 700)]);
    assert_eq!(&d[6..], &[co(None, 700), co(None, 800), co(Some(FG[0]), 700), co(Some(FG[0]), 800), co(Some(FG[1]), 700), co(Some(FG[1]), 800), co(Some(FG[2]), 700), co(Some(FG[2]), 800)]);
    // the Source display reads the mattes at X only (as probes), not at the pair
    let r = rig(&[("display", "1")]).with_layer(FG[0], clip());
    assert_eq!(
        r.deps(7).unwrap(),
        vec![co(None, 700), co(Some(ALT), 700), co(None, 700), co(Some(FG[0]), 700), co(Some(FG[1]), 700), co(None, 700)]
    );
}

#[test]
fn fg2_without_fg1_is_ignored() {
    // the scan stops at the first empty matte: FG2 alone does not count and its probe is never issued
    let r = rig(&[]).with_layer(FG[1], clip());
    assert_eq!(r.deps(7).unwrap(), rig(&[]).deps(7).unwrap());
}

#[test]
fn every_output_frame_follows_the_formula() {
    for n in 0..12i64 {
        let d = rig(&[]).deps(n as u64).unwrap();
        assert_eq!(&d[4..], &[co(None, 100 * n), co(None, 100 * (n - 1))], "frame {n}");
        let d = rig(&[("track_frame", "2")]).deps(n as u64).unwrap();
        assert_eq!(&d[4..], &[co(None, 100 * n), co(None, 100 * (n + 1))], "frame {n}");
    }
    // frame 0 reaches back to time -100 (the host answers None for it, a transparent image)
    assert_eq!(rig(&[]).deps(0).unwrap()[5], co(None, -100));
}

#[test]
fn a_current_time_between_frames_rounds_to_the_nearest_frame() {
    // current_time 730 is frame 7.3: frame = (u32)(7.3 + 0.5) = 7, so the pair is 7 and 6 at 700 and 600, but the
    // probes and the Source display read at the current time itself
    let r = rig(&[]);
    let mut cx = r.cx(7);
    cx.current_time = 730;
    let d = CreateVectors.dependencies(&cx, &r.env()).unwrap();
    assert_eq!(d, vec![co(None, 730), co(Some(ALT), 730), co(None, 730), co(Some(FG[0]), 730), co(None, 700), co(None, 600)]);
    // 7.5 rounds up to frame 8
    cx.current_time = 750;
    let d = CreateVectors.dependencies(&cx, &r.env()).unwrap();
    assert_eq!(&d[4..], &[co(None, 800), co(None, 700)]);
    // the Source display at 730
    let r = rig(&[("display", "1")]);
    let mut cx = r.cx(7);
    cx.current_time = 730;
    let d = CreateVectors.dependencies(&cx, &r.env()).unwrap();
    assert_eq!(d.last(), Some(&co(None, 730)));
}

#[test]
fn an_ntsc_time_base_scales_the_times() {
    // 30000 / 1001: time_step 1001, frame 5 starts at 5005, t(k) = k * 1001
    let r = rig(&[]);
    let cx = TemporalContext::new(5, 4, Depth::U8, TimeBase::default_for_rate(Rational::new(30000, 1001).unwrap()), 5, 12);
    assert_eq!(cx.current_time, 5005);
    let d = CreateVectors.dependencies(&cx, &r.env()).unwrap();
    assert_eq!(&d[4..], &[co(None, 5005), co(None, 4004)]);
}

#[test]
fn display_and_track_frame_are_read_at_the_current_time() {
    // Track Frame Previous at the first key, Next from frame 5 on (popups hold their key)
    let r = rig(&[("track_frame", "1@0;2@5")]);
    assert_eq!(&r.deps(4).unwrap()[4..], &[co(None, 400), co(None, 300)]);
    assert_eq!(&r.deps(5).unwrap()[4..], &[co(None, 500), co(None, 600)]);
    assert_eq!(&r.deps(9).unwrap()[4..], &[co(None, 900), co(None, 1000)]);
    // Display Vectors until frame 7, Source from there: the pair disappears
    let r = rig(&[("display", "2@0;1@7")]);
    assert_eq!(r.deps(6).unwrap().len(), 6);
    assert_eq!(r.deps(7).unwrap().len(), 5);
}

#[test]
fn the_list_does_not_depend_on_the_other_parameters() {
    // no other parameter value changes the checkouts
    let base = rig(&[]).deps(7).unwrap();
    for p in [
        ("vector_resolution", "2"),
        ("max_displace", "100"),
        ("prep", "3"),
        ("track_rgb_a", "1"),
        ("main_bg_sensitivity", "10"),
        ("draw_geom", "1"),
        ("layers_output", "3"),
        ("pt1_use", "2"),
        ("fg1_sensitivity", "5"),
        ("main_bg_track", "0"),
    ] {
        assert_eq!(rig(&[p]).deps(7).unwrap(), base, "{p:?}");
    }
}

#[test]
fn the_layer_list_matches_the_spec() {
    let layers = CreateVectors.layers();
    let ids: Vec<(&str, &str, bool)> = layers.iter().map(|l| (l.id, l.name, l.optional)).collect();
    assert_eq!(
        ids,
        vec![
            ("alt_source_a", "Alt Src", true),
            ("fg1_matte", "FG1: Matte", true),
            ("fg2_matte", "FG2: Matte", true),
            ("fg3_matte", "FG3: Matte", true)
        ]
    );
    assert_eq!(CreateVectors.kind(), fx_core::EffectKind::Temporal);
}
