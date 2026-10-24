//! Plugin 03's time map: `frame_time` and `marker` are `trunc((f64)f58 * (f64)(f32)n + 0.5)` with
//! no `base`, no unequal rate branch and an f64 product, `src_pos` is plugin 01's without `Src Time
//! Offset`.
//!
//! The expected numbers are worked out here from the formulas, with f32 and f64 values produced by
//! Python (struct round trips for the f32 narrowing), and the places where plugin 01's map answers
//! differently are asserted against `RemapTimeMap` itself so the difference is pinned from both
//! sides.

use fx_core::FieldOrder;
use retime_fgbg::timemap::RetimeFgBgTimeMap;
use retime_engine::timeline::{Rates, RemapTimeMap, TimeMap, TimeParams};
use std::cell::RefCell;

/// A parameter reader with closures of the sampling time; every read is logged.
struct Mock {
    mode: i32,
    speed: Box<dyn Fn(i32) -> i32>,
    frame: Box<dyn Fn(i32) -> f64>,
    marks: Box<dyn Fn(i32) -> i32>,
    log: RefCell<Vec<(&'static str, i32)>>,
}

impl Mock {
    fn new(mode: i32) -> Mock {
        Mock {
            mode,
            speed: Box::new(|_| 100 * 65536),
            frame: Box::new(|_| 0.0),
            marks: Box::new(|_| 1),
            log: RefCell::new(Vec::new()),
        }
    }

    fn speed(percent: i32) -> Mock {
        let mut m = Mock::new(1);
        m.speed = Box::new(move |_| percent * 65536);
        m
    }
}

impl TimeParams for Mock {
    fn remap_mode(&self, time: i32) -> i32 {
        self.log.borrow_mut().push(("mode", time));
        self.mode
    }
    fn speed_fixed(&self, time: i32) -> i32 {
        self.log.borrow_mut().push(("speed", time));
        (self.speed)(time)
    }
    fn frame_num(&self, time: i32) -> f64 {
        self.log.borrow_mut().push(("frame", time));
        (self.frame)(time)
    }
    fn src_time_offset_fixed(&self, time: i32) -> i32 {
        self.log.borrow_mut().push(("offset", time));
        7 * 65536
    }
    fn mark_segs(&self, time: i32) -> i32 {
        self.log.borrow_mut().push(("marks", time));
        (self.marks)(time)
    }
}

fn rates(input: f64, comp: f64, fields: FieldOrder, lts: i32) -> Rates {
    Rates::new(input, comp, false, fields, lts).unwrap()
}

#[test]
fn f58_is_the_duration_of_a_source_frame_in_ticks_narrowed_to_f32() {
    // (f32)(lts * out / in), computed in f64 first: the values below come from Python
    let cases: [(f64, f64, i32, f32); 6] = [
        (24.0, 30.0, 100, 125.0),
        (29.97, 23.976, 100, 80.0),
        (29.97, 59.94, 100, 200.0),
        (59.94, 30.0, 100, 50.050048828125),
        (59.94, 60.0, 100, 100.10009765625),
        (23.976, 24.0, 1001, 1002.0020141601562),
    ];
    let mock = Mock::speed(100);
    for (input, comp, lts, want) in cases {
        let m = RetimeFgBgTimeMap::new(&mock, rates(input, comp, FieldOrder::Progressive, lts), 0);
        assert_eq!(m.f58(), want, "in {input} out {comp} lts {lts}");
    }
}

#[test]
fn frame_time_is_trunc_of_f58_times_n_plus_half_with_no_base() {
    // (input rate, composition rate, lts, [(n, T)]); T = trunc(f58 * n + 0.5), f58 as above
    let table: [(f64, f64, i32, Vec<(f32, i32)>); 6] = [
        (24.0, 30.0, 100, vec![(0.0, 0), (1.0, 125), (2.0, 250), (3.0, 375), (5.0, 625), (7.0, 875), (10.0, 1250), (100.0, 12500)]),
        (29.97, 23.976, 100, vec![(0.0, 0), (1.0, 80), (3.0, 240), (7.0, 560), (10.0, 800)]),
        (29.97, 59.94, 100, vec![(1.0, 200), (5.0, 1000), (10.0, 2000)]),
        (59.94, 30.0, 100, vec![(1.0, 50), (10.0, 501), (100.0, 5005)]),
        (23.976, 24.0, 1001, vec![(1.0, 1002), (10.0, 10020), (100.0, 100200)]),
        // equal rates: 100 ticks per frame; a negative position truncates toward zero (trunc(-299.5) = -299)
        (30.0, 30.0, 100, vec![(0.0, 0), (4.0, 400), (-3.0, -299), (-1.0, -99)]),
    ];
    let mock = Mock::speed(100);
    for (input, comp, lts, rows) in table {
        let m = RetimeFgBgTimeMap::new(&mock, rates(input, comp, FieldOrder::Progressive, lts), 700);
        for (n, want) in rows {
            assert_eq!(m.frame_time(n), want, "in {input} out {comp} lts {lts} n {n}");
        }
    }
}

#[test]
fn frame_time_differs_from_plugin_01_when_the_rates_differ_and_agrees_when_they_do_not() {
    // 24 -> 30: plugin 01 adds `dur * K` to the f32 product in its unequal rate branch: 637 and 887
    // for frames 5 and 7, plugin 03 has no such branch: 625 and 875
    let mock = Mock::speed(100);
    let r = rates(24.0, 30.0, FieldOrder::Progressive, 100);
    let p1 = RemapTimeMap::new(&mock, r, 700);
    let p3 = RetimeFgBgTimeMap::new(&mock, r, 700);
    assert_eq!((p1.frame_time(5.0), p1.frame_time(7.0)), (637, 887));
    assert_eq!((p3.frame_time(5.0), p3.frame_time(7.0)), (625, 875));
    // equal rates: both are trunc(n * dur + 0.5) (an f64 product), identical
    let r = rates(30.0, 30.0, FieldOrder::Progressive, 100);
    let p1 = RemapTimeMap::new(&mock, r, 700);
    let p3 = RetimeFgBgTimeMap::new(&mock, r, 700);
    for n in [-3.0f32, 0.0, 1.0, 9.0, 123.0] {
        assert_eq!(p1.frame_time(n), p3.frame_time(n), "n {n}");
    }
}

#[test]
fn frame_time_ignores_the_base_that_plugin_01_subtracts() {
    // plugin 01 in Speed mode with Src Time Offset 7 leaves base = -7 and its frame_time uses (n - base); plugin 03
    // has no offset and no base at all
    let r = rates(30.0, 30.0, FieldOrder::Progressive, 100);
    let mock1 = Mock::speed(100);
    let p1 = RemapTimeMap::new(&mock1, r, 700);
    p1.src_pos(3.0).unwrap();
    assert_eq!(p1.base(), -7);
    assert_eq!(p1.frame_time(5.0), 1200, "(5 + 7) * 100");
    let mock = Mock::speed(100);
    let p3 = RetimeFgBgTimeMap::new(&mock, r, 700);
    p3.src_pos(3.0).unwrap();
    assert_eq!(p3.frame_time(5.0), 500);
    assert!(mock.log.borrow().iter().all(|(name, _)| *name != "offset"), "plugin 03 never reads an offset: {:?}", mock.log.borrow());
}

#[test]
fn input_fields_double_the_input_rate_and_halve_f58() {
    // 30 -> 30 with Upper Field First: in_fps 60, f58 = 100 * 30 / 60 = 50
    let mock = Mock::speed(100);
    let m = RetimeFgBgTimeMap::new(&mock, rates(30.0, 30.0, FieldOrder::UpperFirst, 100), 700);
    assert_eq!(m.f58(), 50.0);
    assert_eq!((m.frame_time(14.0), m.frame_time(14.5), m.frame_time(15.0), m.frame_time(15.5)), (700, 725, 750, 775));
    // Speed 100 percent at output position 7 is the source position 14 (60 / 30 times 7)
    assert_eq!(m.src_pos(7.0).unwrap(), 14.0);
}

#[test]
fn marker_samples_mark_segs_at_the_time_of_the_frame_and_maps_1_to_5_onto_0_to_4() {
    // 24 -> 30, f58 = 125: marker(5) reads at trunc(125 * 5 + 0.5) = 625 (the plugin 01 marker has base 0 and the
    // same value here), a popup keyed 1 before tick 600 and 3 from there on gives 0 then 2
    let mut mock = Mock::speed(100);
    mock.marks = Box::new(|t| if t < 600 { 1 } else { 3 });
    let m = RetimeFgBgTimeMap::new(&mock, rates(24.0, 30.0, FieldOrder::Progressive, 100), 700);
    assert_eq!(m.marker(4.0).unwrap(), 0, "tick 500");
    assert_eq!(m.marker(5.0).unwrap(), 2, "tick 625");
    assert_eq!(*mock.log.borrow(), vec![("marks", 500), ("marks", 625)]);
    // every popup value: 1..=5 map to 0..=4, anything else to 0
    for (value, want) in [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (0, 0), (6, 0), (-1, 0), (1000, 0)] {
        let mut mock = Mock::speed(100);
        mock.marks = Box::new(move |_| value);
        let m = RetimeFgBgTimeMap::new(&mock, rates(30.0, 30.0, FieldOrder::Progressive, 100), 0);
        assert_eq!(m.marker(2.0).unwrap(), want, "value {value}");
    }
}

#[test]
fn marker_multiplies_in_f64_where_plugin_01_multiplies_in_f32() {
    // 59.94 -> 60 (dur = 100.10009765625): at frame 984 the f64 product 98498.496 gives tick 98498,
    // the f32 product of plugin 01's marker rounds to 98499.0 and gives tick 98499 (found by a
    // Python search over 1..200000; 984, 994, 1004 and 1014 are the first four). A popup that
    // switches at tick 98499 tells them apart.
    let r = rates(59.94, 60.0, FieldOrder::Progressive, 100);
    assert_eq!(r.dur, 100.10009765625);
    let switch = |t: i32| if t < 98499 { 1 } else { 3 };
    let mut m3 = Mock::speed(100);
    m3.marks = Box::new(switch);
    let mut m1 = Mock::speed(100);
    m1.marks = Box::new(switch);
    let p3 = RetimeFgBgTimeMap::new(&m3, r, 0);
    let p1 = RemapTimeMap::new(&m1, r, 0);
    assert_eq!(p3.marker(984.0).unwrap(), 0);
    assert_eq!(m3.log.borrow().last(), Some(&("marks", 98498)));
    assert_eq!(p1.marker(984.0).unwrap(), 2);
    assert_eq!(m1.log.borrow().last(), Some(&("marks", 98499)));
    // frame_time uses the f64 product too: 984 * 100.10009765625 + 0.5 = 98498.996...
    assert_eq!(p3.frame_time(984.0), 98498);
}

#[test]
fn src_pos_is_plugin_01s_without_the_offset() {
    // Frame Number: (in_fps / out_fps) * Frame Num read at trunc(lts * t + K); 24 -> 30 gives 0.8 * 2.5 = 2.0
    let mut frame = Mock::new(2);
    frame.frame = Box::new(|_| 2.5);
    let m = RetimeFgBgTimeMap::new(&frame, rates(24.0, 30.0, FieldOrder::Progressive, 100), 700);
    assert_eq!(m.src_pos(7.0).unwrap(), 2.0);
    assert_eq!(*frame.log.borrow(), vec![("mode", 700), ("frame", 700)]);
    // Speed: 24 -> 30 at 100 percent, output position 10 is source position 8
    let speed = Mock::speed(100);
    let m = RetimeFgBgTimeMap::new(&speed, rates(24.0, 30.0, FieldOrder::Progressive, 100), 700);
    assert_eq!(m.src_pos(10.0).unwrap(), 8.0);
    assert!(speed.log.borrow().iter().all(|(n, _)| *n == "mode" || *n == "speed"), "{:?}", speed.log.borrow());
    // half speed samples every half frame: 50 percent at position 5 is 2.5
    let half = Mock::speed(50);
    let m = RetimeFgBgTimeMap::new(&half, rates(30.0, 30.0, FieldOrder::Progressive, 100), 700);
    assert_eq!(m.src_pos(5.0).unwrap(), 2.5);
    // an unknown mode is an error
    let bad = Mock::new(3);
    let m = RetimeFgBgTimeMap::new(&bad, rates(30.0, 30.0, FieldOrder::Progressive, 100), 700);
    assert!(m.src_pos(1.0).is_err());
}

#[test]
fn the_speed_cache_is_bit_identical() {
    let r = rates(30.0, 30.0, FieldOrder::Progressive, 100);
    let mut fast = Mock::new(1);
    fast.speed = Box::new(|t| if t < 3050 { 50 * 65536 } else { 200 * 65536 });
    let mut slow = Mock::new(1);
    slow.speed = Box::new(|t| if t < 3050 { 50 * 65536 } else { 200 * 65536 });
    let cached = RetimeFgBgTimeMap::new(&fast, r, 0).with_speed_cache();
    let plain = RetimeFgBgTimeMap::new(&slow, r, 0);
    for t in [0.0f32, 1.0, 2.5, 30.0, 31.0, 40.25, 99.0] {
        assert_eq!(cached.src_pos(t).unwrap().to_bits(), plain.src_pos(t).unwrap().to_bits(), "t {t}");
    }
}
