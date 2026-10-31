//! Plugin 04's time map: `T(k) = trunc(dur * k + 0.5)` for every fetch, the segment marker sampled
//! at `T`, and `src_pos` with no Src Time Offset. Golden values are worked out by hand from the
//! formulas; the differences from plugin 01's map are asserted on `RemapTimeMap` itself.

mod common;

use common::*;
use fx_core::FieldOrder;
use vectors_in::timemap::{CurveParams04, VectorsInTimeMap};
use retime_engine::timeline::{Rates, RemapTimeMap, TimeMap};

fn rates(input: f64, comp: f64) -> Rates {
    Rates::new(input, comp, false, FieldOrder::Progressive, 100).unwrap()
}

#[test]
fn frame_to_time_is_the_plain_rounded_product() {
    let c = curves(&[]);
    let p = CurveParams04::new(&c).unwrap();
    // (input rate, composition rate, dur, frame, T)
    let table: [(f64, f64, f32, f32, i32); 12] = [
        (30.0, 30.0, 100.0, 5.0, 500),
        (30.0, 30.0, 100.0, 0.0, 0),
        (30.0, 30.0, 100.0, -3.0, -299), // trunc(-300 + 0.5) = trunc(-299.5)
        (24.0, 30.0, 125.0, 5.0, 625),
        (24.0, 30.0, 125.0, 1.0, 125), // trunc(125.5)
        (24.0, 30.0, 125.0, -3.0, -374), // trunc(-374.5)
        (30.0, 24.0, 80.0, 7.0, 560),
        (30.0, 24.0, 80.0, 9.0, 720),
        (12.0, 24.0, 200.0, 2.0, 400),
        (60.0, 30.0, 50.0, 3.0, 150),
        (30.0, 30.0, 100.0, 2.5, 250),
        (24.0, 30.0, 125.0, 0.5, 63), // 125 * 0.5 + 0.5 = 63.0
    ];
    for (i, (input, comp, dur, frame, want)) in table.into_iter().enumerate() {
        let r = rates(input, comp);
        assert_eq!(r.dur, dur, "row {i}");
        let map = VectorsInTimeMap::new(p, r, 0);
        assert_eq!(map.frame_time(frame), want, "row {i}: dur {dur} frame {frame}");
        assert_eq!(map.time_of(frame), want);
    }
}

#[test]
fn it_is_not_plugin_01s_frame_time_when_the_rates_differ() {
    // 24 in 30: plugin 01 adds the FRAME_TIME_K bias on unequal rates (frame 5 gives 631), plugin 04 does not (625)
    let c = curves(&[]);
    let p = CurveParams04::new(&c).unwrap();
    let r = rates(24.0, 30.0);
    let ours = VectorsInTimeMap::new(p, r, 0);
    let theirs = RemapTimeMap::new(p, r, 0);
    assert_eq!(ours.frame_time(5.0), 625);
    assert_eq!(theirs.frame_time(5.0), 631);
    // equal rates agree
    let r = rates(30.0, 30.0);
    let (ours, theirs) = (VectorsInTimeMap::new(p, r, 0), RemapTimeMap::new(p, r, 0));
    for n in [-4.0f32, 0.0, 1.0, 7.0, 123.0] {
        assert_eq!(ours.frame_time(n), theirs.frame_time(n), "{n}");
    }
}

#[test]
fn the_marker_samples_the_popup_at_t_and_maps_one_to_five_onto_zero_to_four() {
    // keys hold between frames: 1 until frame 3, 2 from frame 3, 5 from frame 6
    let c = curves(&[("src_mark_segments", "1@0;2@3;5@6")]);
    let p = CurveParams04::new(&c).unwrap();
    let map = VectorsInTimeMap::new(p, rates(30.0, 30.0), 0);
    let at = |f: f32| map.marker(f).unwrap();
    assert_eq!([at(0.0), at(2.0), at(3.0), at(5.0), at(6.0), at(9.0)], [0, 0, 1, 1, 4, 4]);
    // a frame time of 24 in 30 is T(f) = 125 f: the key at frame 3 (tick 300) is reached by f = 3 (tick 375) but not f = 2 (250)
    let map = VectorsInTimeMap::new(p, rates(24.0, 30.0), 0);
    assert_eq!([map.marker(2.0).unwrap(), map.marker(3.0).unwrap()], [0, 1]);
    for value in 1..=5 {
        let c = curves(&[("src_mark_segments", &value.to_string())]);
        let map = VectorsInTimeMap::new(CurveParams04::new(&c).unwrap(), rates(30.0, 30.0), 0);
        assert_eq!(map.marker(4.0).unwrap(), value - 1);
    }
}

#[test]
fn src_pos_has_no_offset_and_no_base() {
    // Speed 100 percent, equal rates: the output position is the source position
    let c = curves(&[]);
    let map = VectorsInTimeMap::new(CurveParams04::new(&c).unwrap(), rates(30.0, 30.0), 0);
    for t in [0.0f32, 1.0, 7.0, 7.5, 20.0] {
        assert_eq!(map.src_pos(t).unwrap(), t, "{t}");
    }
    // Speed 50 percent
    let c = curves(&[("speed_percent", "50")]);
    let map = VectorsInTimeMap::new(CurveParams04::new(&c).unwrap(), rates(30.0, 30.0), 0);
    assert_eq!([map.src_pos(7.0).unwrap(), map.src_pos(8.0).unwrap()], [3.5, 4.0]);
    // 24 in 30: the factor in / out = 0.8 is applied once at the end: 7 * 0.8 in double, narrowed
    let c = curves(&[("in_fps_is_out_fps", "0"), ("input_frame_rate", "24")]);
    let map = VectorsInTimeMap::new(CurveParams04::new(&c).unwrap(), rates(24.0, 30.0), 0);
    assert_eq!(map.src_pos(7.0).unwrap(), (0.8f64 * 7.0) as f32);
    // Frame Number: Frame Num (2.5) times in / out
    let c = curves(&[("time_remap_mode", "2"), ("frame_num", "2.5")]);
    let map = VectorsInTimeMap::new(CurveParams04::new(&c).unwrap(), rates(24.0, 30.0), 0);
    assert_eq!(map.src_pos(3.0).unwrap(), 2.0);
    let map = VectorsInTimeMap::new(CurveParams04::new(&c).unwrap(), rates(30.0, 30.0), 0);
    assert_eq!(map.src_pos(3.0).unwrap(), 2.5);
    // Src Time Offset does not exist: the delegate never reads it (answers 0 through the TimeParams of this crate)
    use retime_engine::timeline::TimeParams;
    assert_eq!(CurveParams04::new(&c).unwrap().src_time_offset_fixed(0), 0);
}

#[test]
fn an_animated_speed_curve_is_integrated_in_half_frame_steps() {
    // Speed 100 -> 200 percent linearly over frames 0..4 (ticks 0..400): the loop samples the curve at
    // trunc(lts * f / 2 + K) for f = 1..: ticks 50, 100, 150, 200 for the first output frame position 2
    let c = curves(&[("speed_percent", "100@0;200@4")]);
    let map = VectorsInTimeMap::new(CurveParams04::new(&c).unwrap(), rates(30.0, 30.0), 0);
    // position 2.0: four half steps at ticks 50, 100, 150, 200 -> speeds 112.5, 125, 137.5, 150 percent;
    // each adds speed / 2 / 100... sum = (1.125 + 1.25 + 1.375 + 1.5) / 2 = 2.625
    assert_eq!(map.src_pos(2.0).unwrap(), 2.625);
    // the cache gives the same bits
    let cached = VectorsInTimeMap::new(CurveParams04::new(&c).unwrap(), rates(30.0, 30.0), 0).with_speed_cache();
    for k in 0..40 {
        let t = k as f32 * 0.5;
        assert_eq!(map.src_pos(t).unwrap().to_bits(), cached.src_pos(t).unwrap().to_bits(), "{t}");
    }
}
