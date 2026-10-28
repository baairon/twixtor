//! `Effect::dependencies` of plugin 04 against hand worked examples. Every expected list below was
//! derived by hand from these formulas (the composition is 30 fps with 100 ticks per frame,
//! `TimeBase::default_for_rate`, so `lts = 100`, and `X` is the current time):
//!
//! * `dur = (f32)(lts * out_fps / in_fps)` where, with `In FPS is Out FPS` checked (the default),
//!   `in = 30.0` and `out` is the COMPOSITION rate; `T(k) = trunc(dur * k + 0.5)`;
//! * probes: the input layer, Source, Vectors: Prev, Vectors: Next, FG1 matte (each at `X`);
//! * per plan entry `k` with `off = T(k)`: the dummy registrations (Source, Prev, Next, FG1, FG2,
//!   FG3 each at `X + off` and `-X`), then Source, Prev, Next at `off`.

mod common;

use common::*;
use fx_core::{Checkout, Depth, Rational};

const SRC: Option<&str> = None;
const PREV: Option<&str> = Some("vectors_prev");
const NEXT: Option<&str> = Some("vectors_next");
const FG: [Option<&str>; 3] = [Some("fg1_matte"), Some("fg2_matte"), Some("fg3_matte")];

fn co(layer: Option<&'static str>, time: i64) -> Checkout {
    Checkout { layer, time }
}

fn rig(params: &[(&str, &str)]) -> Rig {
    Rig::new(random_clip(Depth::U8, 5, 4, 12, 3), params).with_frames(12)
}

/// The probes at `x` with no FG layer: layer 0, Source, Prev, Next, then FG1 (empty, which stops the scan).
fn probes(x: i64, src_probe: Option<&'static str>) -> Vec<Checkout> {
    vec![co(None, x), co(src_probe.or(Some("source")), x), co(PREV, x), co(NEXT, x), co(FG[0], x)]
}

/// One plan entry at layer time `off` of the frame at output time `x`.
fn entry(x: i64, off: i64, src: Option<&'static str>) -> Vec<Checkout> {
    let mut v = Vec::new();
    for layer in [src, PREV, NEXT, FG[0], FG[1], FG[2]] {
        v.push(co(layer, x + off));
        v.push(co(layer, -x));
    }
    v.extend([co(src, off), co(PREV, off), co(NEXT, off)]);
    v
}

fn expected(x: i64, offs: [i64; 2], src: Option<&'static str>) -> Vec<Checkout> {
    let mut v = probes(x, Some("source"));
    for off in offs {
        v.extend(entry(x, off, src));
    }
    v
}

#[test]
fn the_worked_example_output_frame_7_at_full_speed() {
    // X = 700, positions 7 and 8, dur = 100: T = 700 and 800
    let deps = rig(&[]).deps(7).unwrap();
    assert_eq!(deps.len(), 35);
    assert_eq!(deps, expected(700, [700, 800], SRC));
    // spelled out for the first entry
    assert_eq!(
        &deps[..20],
        &[
            co(None, 700),
            co(Some("source"), 700),
            co(PREV, 700),
            co(NEXT, 700),
            co(FG[0], 700),
            // entry for frame 7, off = 700: dummies at X + off = 1400 and -X = -700
            co(None, 1400),
            co(None, -700),
            co(PREV, 1400),
            co(PREV, -700),
            co(NEXT, 1400),
            co(NEXT, -700),
            co(FG[0], 1400),
            co(FG[0], -700),
            co(FG[1], 1400),
            co(FG[1], -700),
            co(FG[2], 1400),
            co(FG[2], -700),
            // its own checkouts at off
            co(None, 700),
            co(PREV, 700),
            co(NEXT, 700),
        ]
    );
}

#[test]
fn every_output_frame_lists_the_pair_around_its_source_position() {
    for n in 0..12i64 {
        let deps = rig(&[]).deps(n as u64).unwrap();
        let x = 100 * n;
        assert_eq!(deps, expected(x, [100 * n, 100 * (n + 1)], SRC), "frame {n}");
    }
}

#[test]
fn half_speed_reads_the_frames_around_position_3_point_5() {
    // output frame 7 at 50 percent: s0 = 3.5, frames 3 and 4
    let deps = rig(&[("speed_percent", "50")]).deps(7).unwrap();
    assert_eq!(deps, expected(700, [300, 400], SRC));
    // output frame 6: s0 = 3.0, s1 = 3.5: the pair is still (3, 4)
    assert_eq!(rig(&[("speed_percent", "50")]).deps(6).unwrap(), expected(600, [300, 400], SRC));
}

#[test]
fn the_source_layer_names_the_source_entries_when_it_is_set() {
    let r = rig(&[("speed_percent", "50")]).with_layer("source", random_clip(Depth::U8, 5, 4, 12, 4));
    let deps = r.deps(7).unwrap();
    assert_eq!(deps, expected(700, [300, 400], Some("source")));
    // the probe of the layer parameter is the same either way
    assert!(deps.contains(&co(Some("source"), 700)));
}

#[test]
fn the_composition_rate_wins_over_in_fps_is_out_fps() {
    // 24 fps composition, checkbox on: in = 30.0, out = 24: dur = 100 * 24 / 30 = 80; position = 7 * (30 / 24) = 8.75
    let r = rig(&[]);
    let mut cx = r.cx(7);
    cx.fps = Rational::new(24, 1).unwrap();
    let deps = fx_core::Effect::dependencies(&vectors_in::VectorsIn, &cx, &r.env()).unwrap();
    // frames 8 and 9: T(8) = 640, T(9) = trunc(720.5) = 720
    assert_eq!(deps, expected(700, [640, 720], SRC));
}

#[test]
fn unequal_input_rate_uses_the_plain_product_not_plugin_01s_bias() {
    // In FPS is Out FPS off, input 24, comp 30: dur = 125, position = 7 * 0.8 = 5.6, frames 5 and 6
    let deps = rig(&[("in_fps_is_out_fps", "0"), ("input_frame_rate", "24")]).deps(7).unwrap();
    assert_eq!(deps, expected(700, [625, 750], SRC));
}

#[test]
fn frame_number_mode_reads_the_frames_around_the_number() {
    // Frame Num 2.5 at every frame: position 2.5, frames 2 and 3
    for n in [0u64, 4, 9] {
        let deps = rig(&[("time_remap_mode", "2"), ("frame_num", "2.5")]).deps(n).unwrap();
        assert_eq!(deps, expected(100 * n as i64, [200, 300], SRC), "frame {n}");
    }
}

#[test]
fn the_source_display_registers_the_source_once_more_and_nothing_else() {
    let deps = rig(&[("display", "2")]).deps(7).unwrap();
    let mut want = probes(700, Some("source"));
    want.push(co(None, 700));
    assert_eq!(deps, want);
    let r = rig(&[("display", "2")]).with_layer("source", random_clip(Depth::U8, 5, 4, 12, 4));
    let deps = r.deps(7).unwrap();
    assert_eq!(deps.last(), Some(&co(Some("source"), 700)));
    assert_eq!(deps.len(), 6);
}

#[test]
fn the_fg_probe_stops_at_the_first_empty_matte() {
    let m = || random_clip(Depth::U8, 5, 4, 12, 9);
    // FG1 set, FG2 empty: probes end with FG1 and FG2 (the empty one is included); the Source display lists them
    let r = rig(&[("display", "2")]).with_layer("fg1_matte", m());
    let deps = r.deps(3).unwrap();
    assert_eq!(&deps[..6], &[co(None, 300), co(Some("source"), 300), co(PREV, 300), co(NEXT, 300), co(FG[0], 300), co(FG[1], 300)]);
    // FG2 without FG1 is ignored
    let r = rig(&[("display", "2")]).with_layer("fg2_matte", m());
    assert_eq!(r.deps(3).unwrap().len(), 6);
    // all three set: three probes and no stop
    let r = rig(&[("display", "2")]).with_layer("fg1_matte", m()).with_layer("fg2_matte", m()).with_layer("fg3_matte", m());
    let deps = r.deps(3).unwrap();
    assert_eq!(&deps[4..7], &[co(FG[0], 300), co(FG[1], 300), co(FG[2], 300)]);
    assert_eq!(deps.len(), 8);
}

/// One main plan entry at layer time `off` with `nfg` FG mattes: the dummies, Source, then the two
/// vector layers once per layer (`nfg + 1` times, a pair per layer).
fn layered_main(x: i64, off: i64, nfg: usize) -> Vec<Checkout> {
    let mut v = entry(x, off, SRC);
    v.truncate(v.len() - 2);
    for _ in 0..=nfg {
        v.extend([co(PREV, off), co(NEXT, off)]);
    }
    v
}

/// One kind 2 entry (FG matte `k`) at layer time `off`: the dummies, then the matte.
fn matte_entry(x: i64, off: i64, k: usize) -> Vec<Checkout> {
    let mut v = entry(x, off, SRC);
    v.truncate(v.len() - 3);
    v.push(co(FG[k], off));
    v
}

#[test]
fn fg_matte_layers_add_their_kind_2_entries_and_the_vector_layers_once_per_layer() {
    let m = || random_clip(Depth::U8, 5, 4, 12, 9);
    // FG1: the probes end with FG1 and the empty FG2; the main entries load Prev and Next twice (nfg + 1), then
    // the two matte entries of FG1 at the same two layer times (half speed: frame 3 is position 1.5, frames 1 and 2)
    let r = rig(&[("speed_percent", "50")]).with_layer("fg1_matte", m());
    let mut want = vec![co(None, 300), co(Some("source"), 300), co(PREV, 300), co(NEXT, 300), co(FG[0], 300), co(FG[1], 300)];
    for off in [100, 200] {
        want.extend(layered_main(300, off, 1));
    }
    for off in [100, 200] {
        want.extend(matte_entry(300, off, 0));
    }
    assert_eq!(r.deps(3).unwrap(), want);
    // all three: four loads of each vector layer per main entry and six matte entries, FG1 first; an exact frame
    // takes the same plan (the fast path composes the displayed group from the mattes too)
    let r = rig(&[]).with_layer("fg1_matte", m()).with_layer("fg2_matte", m()).with_layer("fg3_matte", m());
    let mut want = vec![co(None, 300), co(Some("source"), 300), co(PREV, 300), co(NEXT, 300), co(FG[0], 300), co(FG[1], 300), co(FG[2], 300)];
    for off in [300, 400] {
        want.extend(layered_main(300, off, 3));
    }
    for k in 0..3 {
        for off in [300, 400] {
            want.extend(matte_entry(300, off, k));
        }
    }
    assert_eq!(r.deps(3).unwrap(), want);
    // every Display but the Source one has the same list
    for display in ["1", "3", "4", "5", "6"] {
        let d = rig(&[("display", display)]).with_layer("fg1_matte", m()).with_layer("fg2_matte", m()).with_layer("fg3_matte", m());
        assert_eq!(d.deps(3).unwrap(), want, "display {display}");
    }
}

#[test]
fn the_vector_layers_are_always_listed_even_when_they_are_absent() {
    // the pre-render registers them whether or not the layers are set
    let r = rig(&[("speed_percent", "50")]);
    let deps = r.deps(5).unwrap();
    assert!(deps.contains(&co(PREV, 200)) && deps.contains(&co(NEXT, 300)));
    let r = r.with_vectors(2.0, 1.0, 16.0);
    assert_eq!(r.deps(5).unwrap(), deps);
}

#[test]
fn a_bad_rate_lists_only_the_probes() {
    let r = rig(&[]);
    let mut cx = r.cx(7);
    cx.fps = Rational::new(1, 1000).unwrap(); // 0.001 fps: out <= 0.01 is the rate error
    let deps = fx_core::Effect::dependencies(&vectors_in::VectorsIn, &cx, &r.env()).unwrap();
    assert_eq!(deps, probes(700, Some("source")));
}

#[test]
fn fifteen_checkouts_per_plan_entry_and_two_entries_per_frame() {
    // plugin 04 has 15 checkouts per plan entry (12 dummies and 3 own: Source, Prev, Next) and always 2 entries
    // (no layer B), plus the 5 probes
    let deps = rig(&[]).deps(2).unwrap();
    assert_eq!(deps.len(), 5 + 2 * 15);
}

#[test]
fn a_context_in_the_first_frame_has_negative_dummy_times_of_zero() {
    // X = 0: -X is 0 as well
    let deps = rig(&[]).deps(0).unwrap();
    assert_eq!(deps[5], co(None, 0));
    assert_eq!(deps[6], co(None, 0));
}

#[test]
fn a_fractional_duration_rounds_half_up_in_the_fetch_times() {
    // 25 fps composition, checkbox on: in = 30, out = 25: dur = (f32)(100 * 25 / 30) = 83.33333587646484 (f32 rounding
    // of 83.333...), position = 7 * 1.2 = 8.4, frames 8 and 9: T(8) = trunc(666.667 + 0.5) = 667, T(9) = trunc(750.0 + 0.5) = 750
    let r = rig(&[]);
    let mut cx = r.cx(7);
    cx.fps = Rational::new(25, 1).unwrap();
    let deps = fx_core::Effect::dependencies(&vectors_in::VectorsIn, &cx, &r.env()).unwrap();
    assert_eq!(deps, expected(700, [667, 750], SRC));
}
