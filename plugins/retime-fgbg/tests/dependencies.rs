//! `Effect::dependencies` of plugin 03 against hand worked examples. Every list below is written
//! out by hand from these formulas:
//!
//! * the first checkout: the input layer at `X`;
//! * size probes at `X`: Alt Source A, the input layer again when A has no pixels, Alt Source B,
//!   the FG Matte layers up to and including the first empty one;
//! * `Display == Source`: one more checkout of Alt Source A at `X` and nothing else;
//! * the plan with `T_k = trunc(f58 * t_k + 0.5)` (no `X`, no base) and, with input fields, `T2_k`
//!   at the half frame;
//! * before each entry, when `register_layers` is set: Alt Source A, Alt Source B and FG1 to FG3 at `X +
//!   off_k` and at `-X` (interleaved per layer), and with input fields the five layers again at `X
//!   + off2_k`.
//!
//! The composition is 30 fps with 100 ticks per frame, so output frame 7 has `X = 700`.

mod common;

use common::*;
use fx_core::{Checkout, Depth, FxError};

const A: Option<&str> = Some("alt_source_a");
const B: Option<&str> = Some("alt_source_b");
const FG1: Option<&str> = Some("fg1_matte");
const FG2: Option<&str> = Some("fg2_matte");
const FG3: Option<&str> = Some("fg3_matte");

fn cp(layer: Option<&'static str>, time: i64) -> Checkout {
    Checkout { layer, time }
}

fn clip() -> Vec<fx_core::Frame> {
    random_clip(Depth::U8, 5, 4, 8, 3)
}

fn rig(params: &[(&str, &str)]) -> Rig {
    Rig::new(clip(), params)
}

/// The five layers of the D registrations at `X + off` and `-X`, interleaved per layer, then (when `off2` is given)
/// the five layers again at `X + off2`.
fn d(x: i64, off: i64, off2: Option<i64>) -> Vec<Checkout> {
    let mut out = Vec::new();
    for layer in [A, B, FG1, FG2, FG3] {
        out.push(cp(layer, x + off));
        out.push(cp(layer, -x));
    }
    if let Some(off2) = off2 {
        for layer in [A, B, FG1, FG2, FG3] {
            out.push(cp(layer, x + off2));
        }
    }
    out
}

#[test]
fn defaults_output_frame_7_is_the_worked_example() {
    // Speed 100 percent, Vectors Best, Blend, no layers: s0 = 7, s1 = 8 (an exact frame), plan (Main 7), (Main 8)
    let got = rig(&[]).deps(7).unwrap();
    let want = vec![
        cp(None, 700), // wrapper: layer 0 at X
        cp(A, 700),    // probe of Alt Source A
        cp(None, 700), // A has no pixels: probe of layer 0
        cp(B, 700),    // probe of Alt Source B
        cp(FG1, 700),  // probe of FG1 Matte, empty: the loop stops
        // entry 0, position 7: off = trunc(100 * 7 + 0.5) = 700
        cp(A, 1400),
        cp(A, -700),
        cp(B, 1400),
        cp(B, -700),
        cp(FG1, 1400),
        cp(FG1, -700),
        cp(FG2, 1400),
        cp(FG2, -700),
        cp(FG3, 1400),
        cp(FG3, -700),
        cp(None, 700), // the entry itself: layer 0 at T = 700
        // entry 1, position 8: off = 800
        cp(A, 1500),
        cp(A, -700),
        cp(B, 1500),
        cp(B, -700),
        cp(FG1, 1500),
        cp(FG1, -700),
        cp(FG2, 1500),
        cp(FG2, -700),
        cp(FG3, 1500),
        cp(FG3, -700),
        cp(None, 800),
    ];
    assert_eq!(got, want);
}

#[test]
fn every_output_frame_follows_the_formula() {
    let r = rig(&[]);
    for n in 0..12i64 {
        let x = n * 100;
        let got = r.deps(n as u64).unwrap();
        let mut want = vec![cp(None, x), cp(A, x), cp(None, x), cp(B, x), cp(FG1, x)];
        for position in [n, n + 1] {
            want.extend(d(x, position * 100, None));
            want.push(cp(None, position * 100));
        }
        assert_eq!(got, want, "output frame {n}");
    }
}

#[test]
fn half_speed_between_frames_lists_the_two_source_frames_around_the_position() {
    // 50 percent at output frame 7: s0 = 3.5, s1 = 4.0; a = 3, b = 4; Vectors Best but no Alt Source B, so the plan
    // has the two Main entries (the flow runs on the main frames)
    let got = rig(&[("speed_percent", "50")]).deps(7).unwrap();
    let mut want = vec![cp(None, 700), cp(A, 700), cp(None, 700), cp(B, 700), cp(FG1, 700)];
    for position in [3i64, 4] {
        want.extend(d(700, position * 100, None));
        want.push(cp(None, position * 100));
    }
    assert_eq!(got, want);
}

#[test]
fn frame_number_mode_ignores_the_output_position_and_the_current_time_in_the_layer_times() {
    // Frame Num 2.5 at every output frame: position 2.5, a = 2, b = 3, T = 200 and 300 whatever X is
    for n in [0u64, 4, 9] {
        let x = n as i64 * 100;
        let got = rig(&[("time_remap_mode", "2"), ("frame_num", "2.5")]).deps(n).unwrap();
        let mut want = vec![cp(None, x), cp(A, x), cp(None, x), cp(B, x), cp(FG1, x)];
        for position in [2i64, 3] {
            want.extend(d(x, position * 100, None));
            want.push(cp(None, position * 100));
        }
        assert_eq!(got, want, "output frame {n}");
    }
}

#[test]
fn alt_source_a_replaces_layer_0_as_the_retimed_layer_and_ends_the_second_probe() {
    // A present: no layer 0 probe after it, the plan entries are layer 13; the D registrations are unchanged
    let r = rig(&[]).with_layer("alt_source_a", clip());
    let got = r.deps(7).unwrap();
    let mut want = vec![cp(None, 700), cp(A, 700), cp(B, 700), cp(FG1, 700)];
    for position in [7i64, 8] {
        want.extend(d(700, position * 100, None));
        want.push(cp(A, position * 100));
    }
    assert_eq!(got, want);
}

#[test]
fn a_tracking_layer_with_a_position_between_frames_adds_its_two_entries_first() {
    // Alt Source B set, Best, half speed: the plan is (B 3), (B 4), (Main 3), (Main 4) (kind 0 entries come first)
    let r = rig(&[("speed_percent", "50")]).with_layer("alt_source_b", clip());
    let got = r.deps(7).unwrap();
    let mut want = vec![cp(None, 700), cp(A, 700), cp(None, 700), cp(B, 700), cp(FG1, 700)];
    for (layer, position) in [(B, 3i64), (B, 4), (None, 3), (None, 4)] {
        want.extend(d(700, position * 100, None));
        want.push(cp(layer, position * 100));
    }
    assert_eq!(got, want);
    // on an exact frame the tracking layer is not needed: two Main entries
    let exact = rig(&[]).with_layer("alt_source_b", clip()).deps(7).unwrap();
    assert_eq!(exact.len(), 5 + 2 * 11);
    assert_eq!(exact[exact.len() - 1], cp(None, 800));
    // with No Vectors the tracking layer is never used either
    let none = rig(&[("speed_percent", "50"), ("vectors", "1")]).with_layer("alt_source_b", clip()).deps(7).unwrap();
    assert_eq!(none.len(), 5 + 2 * 11);
}

#[test]
fn unequal_rates_use_plugin_03s_frame_time_not_plugin_01s() {
    // Input 24 fps in a 30 fps composition: f58 = 125, output frame 7 is source position 5.6 (a = 5, b = 6), so the
    // layer times are trunc(125 * 5 + 0.5) = 625 and trunc(125 * 6 + 0.5) = 750. Plugin 01's map would give 637 and 762.
    let params = [("in_fps_is_out_fps", "0"), ("input_frame_rate", "24")];
    let got = rig(&params).deps(7).unwrap();
    let mut want = vec![cp(None, 700), cp(A, 700), cp(None, 700), cp(B, 700), cp(FG1, 700)];
    for off in [625i64, 750] {
        want.extend(d(700, off, None));
        want.push(cp(None, off));
    }
    assert_eq!(got, want);
}

#[test]
fn input_fields_register_the_half_frame_times_too() {
    // Lower Field First: in_fps 60 against 30, f58 = 50, output frame 7 is source position 14 (a = 14, b = 15):
    // T = 700 and 750, half frame T2 = trunc(50 * 14.5 + 0.5) = 725 and trunc(50 * 15.5 + 0.5) = 775
    let got = rig(&[("input_fields", "3")]).deps(7).unwrap();
    let mut want = vec![cp(None, 700), cp(A, 700), cp(None, 700), cp(B, 700), cp(FG1, 700)];
    for (off, off2) in [(700i64, 725i64), (750, 775)] {
        want.extend(d(700, off, Some(off2)));
        want.push(cp(None, off));
        want.push(cp(None, off2));
    }
    assert_eq!(got, want);
}

#[test]
fn display_source_registers_alt_source_a_once_and_nothing_else() {
    // probes, then one checkout of Alt Source A at X (even when A is empty)
    let got = rig(&[("display", "2")]).deps(7).unwrap();
    assert_eq!(got, vec![cp(None, 700), cp(A, 700), cp(None, 700), cp(B, 700), cp(FG1, 700), cp(A, 700)]);
    let with_a = rig(&[("display", "2")]).with_layer("alt_source_a", clip()).deps(3).unwrap();
    assert_eq!(with_a, vec![cp(None, 300), cp(A, 300), cp(B, 300), cp(FG1, 300), cp(A, 300)]);
    // no Src Time Offset in plugin 03 and the speed does not move the source: the Source display is at X
    let slow = rig(&[("display", "2"), ("speed_percent", "25")]).deps(5).unwrap();
    assert_eq!(slow.last(), Some(&cp(A, 500)));
    // the other Display values (FG1, FG2, FG3, Background processed) take the plan like the output
    for display in ["3", "4", "5", "6"] {
        assert_eq!(rig(&[("display", display)]).deps(7).unwrap(), rig(&[]).deps(7).unwrap(), "display {display}");
    }
}

#[test]
fn fg_matte_probes_stop_at_the_first_empty_layer() {
    // FG1 present, FG2 empty: probes of FG1 and FG2 only (FG3 is not probed); Display == Source keeps the list
    let r = rig(&[("display", "2")]).with_layer("fg1_matte", clip());
    assert_eq!(r.deps(2).unwrap(), vec![cp(None, 200), cp(A, 200), cp(None, 200), cp(B, 200), cp(FG1, 200), cp(FG2, 200), cp(A, 200)]);
    // FG1 and FG2 present: the third matte is the first empty one
    let r = rig(&[("display", "2")]).with_layer("fg1_matte", clip()).with_layer("fg2_matte", clip());
    assert_eq!(
        r.deps(2).unwrap(),
        vec![cp(None, 200), cp(A, 200), cp(None, 200), cp(B, 200), cp(FG1, 200), cp(FG2, 200), cp(FG3, 200), cp(A, 200)]
    );
    // all three present: three probes and no empty one
    let r = rig(&[("display", "2")]).with_layer("fg1_matte", clip()).with_layer("fg2_matte", clip()).with_layer("fg3_matte", clip());
    assert_eq!(
        r.deps(2).unwrap(),
        vec![cp(None, 200), cp(A, 200), cp(None, 200), cp(B, 200), cp(FG1, 200), cp(FG2, 200), cp(FG3, 200), cp(A, 200)]
    );
    // FG2 set without FG1: the loop stops at FG1, so no matte counts (a single layer render)
    let r = rig(&[("display", "2")]).with_layer("fg2_matte", clip());
    assert_eq!(r.deps(2).unwrap(), vec![cp(None, 200), cp(A, 200), cp(None, 200), cp(B, 200), cp(FG1, 200), cp(A, 200)]);
    let r = rig(&[]).with_layer("fg2_matte", clip());
    assert_eq!(r.deps(7).unwrap(), rig(&[]).deps(7).unwrap());
}

#[test]
fn the_multi_layer_plan_is_not_guessed() {
    // FG1 Matte present and the plan needed: the plan of the multi layer path has entries of kind 2 that this
    // front-end does not know, so the list is an error and not a guess
    let r = rig(&[]).with_layer("fg1_matte", clip());
    match r.deps(7) {
        Err(FxError::Unsupported(m)) => assert!(m.starts_with("a FG Matte layer is set"), "{m}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn earlier_hosts_have_no_extra_registrations() {
    // without the extra registrations (`PreRender03::register_layers = false`) the list is probes and plan
    // checkouts only
    use retime_fgbg::prerender::{checkouts, PreRender03};
    use retime_engine::timeline::Rates;
    let rates = Rates::new(15.0, 30.0, true, fx_core::FieldOrder::Progressive, 100).unwrap();
    let curves = curves(&[]);
    let params = retime_fgbg::timemap::CurveParams03::new(&curves).unwrap();
    let map = retime_fgbg::timemap::RetimeFgBgTimeMap::new(params, rates, 700);
    let req = PreRender03 {
        current_time: 700,
        time_step: 100,
        local_time_step: 100,
        display_source: false,
        alt_a_present: false,
        alt_b_present: false,
        fg_count: 0,
        input_fields: false,
        vectors_quality: 4,
        interp: 1,
        blur: 0.0,
        register_layers: false,
    };
    let got = checkouts(&req, &map, "unused").unwrap();
    assert_eq!(got, vec![cp(None, 700), cp(A, 700), cp(None, 700), cp(B, 700), cp(FG1, 700), cp(None, 700), cp(None, 800)]);
}

#[test]
fn an_animated_speed_curve_is_integrated_in_half_frame_steps() {
    // Speed 100 percent at frame 0 rising linearly to 200 percent at frame 4 (ticks 0 and 400), then held. Output frame
    // 6 (X = 600) is position 6.0: twelve half steps sampled at ticks 50 k, with the percentages
    // 112.5, 125, 137.5, 150, 162.5, 175, 187.5, 200 and then 200 four times; each adds speed / 2:
    // (12.5 + 4 * 2.0) / 2 = 10.25. So a = 10, b = 11, the weight is not on a frame, there is no Alt Source B and the
    // plan is (Main 10), (Main 11) at T = 1000 and 1100.
    let got = rig(&[("speed_percent", "100@0;200@4")]).deps(6).unwrap();
    let mut want = vec![cp(None, 600), cp(A, 600), cp(None, 600), cp(B, 600), cp(FG1, 600)];
    for position in [10i64, 11] {
        want.extend(d(600, position * 100, None));
        want.push(cp(None, position * 100));
    }
    assert_eq!(got, want);
}
