//! Tests of the tracking point primitives: the `Pt N: Use` table against create-vectors'
//! `point_belongs`, the 16.16 position reader, hand goldens of the per level splat (rule G2: the
//! pixel, the scale, the skip of points outside the frame, two points on one pixel), the second
//! tracker (rule G5: the exact zero field, a near uniform field from one point, exact values at the
//! points, the sign of the output convention) and thread count invariance.

use super::*;
use crate::flow::FlowSettings;
use fx_core::{ParamKind, ParamSpec, ParamValue, TimeBase};

fn in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(f)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// create-vectors' `point_belongs` (`plugins/create-vectors/src/glue.rs`), restated: value 2 to 4
/// are FG1 to FG3, 5 Main_BG, a value whose layer does not exist belongs to the last layer.
fn belongs(v: i64, layer: usize, layer_count: usize) -> bool {
    let target = v - 2;
    layer as i64 == target || (layer_count as i64 <= target && layer + 1 == layer_count)
}

/// The solver geometry of a full resolution solve (quality 5, R = 0) or of Best (quality 4, R = 1).
fn geometry(quality: u8, w: usize, h: usize) -> FlowGeometry {
    FlowSettings::new(quality, true, 1.0).geometry(w, h).unwrap()
}

#[test]
fn point_layer_follows_the_use_popup_for_every_layer_count() {
    // rows nfg 0 to 3, columns Pt N: Use 1 (Don't Use) to 5 (Main_BG)
    let table: [[Option<usize>; 5]; 4] = [
        [None, Some(0), Some(0), Some(0), Some(0)],
        [None, Some(0), Some(1), Some(1), Some(1)],
        [None, Some(0), Some(1), Some(2), Some(2)],
        [None, Some(0), Some(1), Some(2), Some(3)],
    ];
    for (nfg, row) in table.iter().enumerate() {
        for (i, want) in row.iter().enumerate() {
            assert_eq!(point_layer(i as i64 + 1, nfg), *want, "nfg {nfg}, value {}", i + 1);
        }
    }
    // the owner is the one layer point_belongs names, for values the popup never gives too
    for nfg in 0..=3usize {
        for v in -3i64..=9 {
            let owners: Vec<usize> = (0..=nfg).filter(|&l| belongs(v, l, nfg + 1)).collect();
            assert_eq!(point_layer(v, nfg).into_iter().collect::<Vec<_>>(), owners, "nfg {nfg}, value {v}");
        }
    }
    assert_eq!(point_layer(i64::MIN, 2), None);
    assert_eq!(point_layer(i64::MAX, 2), Some(2));
}

#[test]
fn a_user_point_gives_its_pair_for_either_solve() {
    let p = UserPoint { at_a: [1.0, 2.0], at_b: [3.0, 4.0] };
    assert_eq!(p.pair(true), ([1.0, 2.0], [3.0, 4.0]));
    assert_eq!(p.pair(false), ([3.0, 4.0], [1.0, 2.0]));
}

#[test]
fn positions_are_read_through_the_16_16_reader() {
    let spec = ParamSpec { id: "pt1_pos", name: "Pt 1: Pos", kind: ParamKind::Point { default: [-5.0, -5.0] } };
    let mut c = ParamCurves::new(vec![spec]);
    assert_eq!(read_pos(&c, "pt1_pos", 0), [-5.0, -5.0]);
    // 10.3 * 65536 = 675020.8 -> 675021; -2.7 * 65536 = -176947.2 -> -176947 (halves away from 0)
    c.set_constant("pt1_pos", ParamValue::Point([10.3, -2.7])).unwrap();
    let want = [(675_021.0f64 / 65536.0) as f32, (-176_947.0f64 / 65536.0) as f32];
    assert_eq!(read_pos(&c, "pt1_pos", 0), want);
    assert_ne!(want[0], 10.3f32);
    // keyed: half way between (0, 0) at frame 0 and (10, 20) at frame 10, a third of the way
    c.parse_cli("pt1_pos", "0,0@0;10,20@10", TimeBase { time_scale: 24, time_step: 1 }).unwrap();
    assert_eq!(read_pos(&c, "pt1_pos", 5), [5.0, 10.0]);
    let third = [(218_453.0f64 / 65536.0) as f32, (436_907.0f64 / 65536.0) as f32];
    let t = read_pos(&c, "pt1_pos", 0);
    assert_eq!(t, [0.0, 0.0]);
    // tick 10/3 does not exist; frame 10 * (1/3) is tick 3.33, so sample the exact third at 30 ticks a frame
    c.parse_cli("pt1_pos", "0,0@0;10,20@10", TimeBase { time_scale: 72, time_step: 3 }).unwrap();
    assert_eq!(read_pos(&c, "pt1_pos", 10), third);
}

#[test]
fn the_splat_puts_each_point_on_its_nearest_pixel_at_every_level() {
    // 64x48 at full resolution: levels 64x48, 32x24, 16x12, 8x6, 4x3
    let geo = geometry(5, 64, 48);
    assert_eq!(geo.sizes, vec![(64, 48), (32, 24), (16, 12), (8, 6), (4, 3)]);
    let ud = user_displacement(&[([40.0, 30.0], [46.0, 32.0])], &geo, (64, 48));
    assert_eq!(ud.levels.len(), 5);
    // the pixel of (40, 30) at each level: cvtt(40 * kx + 0.5), cvtt(30 * ky + 0.5)
    let pixels = [(40, 30), (20, 15), (10, 7), (4, 3), (2, 1)];
    for (l, (&(wl, hl), &(px, py))) in geo.sizes.iter().zip(&pixels).enumerate() {
        let lvl = &ud.levels[l];
        assert_eq!((lvl.d.len(), lvl.w.len()), (wl * hl * 2, wl * hl), "level {l}");
        let kx = (wl - 1) as f32 / 63.0f32;
        let ky = (hl - 1) as f32 / 47.0f32;
        let i = py * wl + px;
        assert_eq!(lvl.w[i], 1.0, "level {l}");
        assert_eq!([lvl.d[2 * i], lvl.d[2 * i + 1]], [-6.0f32 * kx, -2.0f32 * ky], "level {l}");
        // nothing anywhere else
        assert_eq!(lvl.w.iter().filter(|&&v| v != 0.0).count(), 1, "level {l}");
        assert_eq!(lvl.d.iter().filter(|&&v| v != 0.0).count(), 2, "level {l}");
    }
    // level 0 scale is 1; level 1 is (31/63, 23/47)
    assert_eq!([ud.levels[0].d[(30 * 64 + 40) * 2], ud.levels[0].d[(30 * 64 + 40) * 2 + 1]], [-6.0, -2.0]);
    assert_eq!(ud.levels[1].d[(15 * 32 + 20) * 2], -6.0f32 * (31.0f32 / 63.0f32));
    // a sub pixel position rounds to the nearest pixel: (40.4, 30.6) lands on (40, 31)
    let ud = user_displacement(&[([40.4, 30.6], [46.0, 32.0])], &geo, (64, 48));
    let i = 31 * 64 + 40;
    assert_eq!(ud.levels[0].w[i], 1.0);
    assert_eq!(ud.levels[0].d[2 * i], 40.4f32 - 46.0);
    assert_eq!(ud.levels[0].d[2 * i + 1], 30.6f32 - 32.0);
}

#[test]
fn two_points_on_one_pixel_add_up_to_their_mean() {
    let geo = geometry(5, 64, 48);
    // a half rounds up on both axes: (10.5, 20.5) lands on (11, 21), (10.49, 20.49) on (10, 20)
    let ud = user_displacement(&[([10.5, 20.5], [9.5, 20.5]), ([10.49, 20.49], [10.49, 19.49])], &geo, (64, 48));
    assert_eq!((ud.levels[0].w[21 * 64 + 11], ud.levels[0].d[(21 * 64 + 11) * 2]), (1.0, 1.0));
    assert_eq!((ud.levels[0].w[20 * 64 + 10], ud.levels[0].d[(20 * 64 + 10) * 2 + 1]), (1.0, 1.0));
    let pairs = [([40.0, 30.0], [46.0, 32.0]), ([40.2, 29.8], [44.2, 29.8]), ([10.0, 10.0], [11.0, 9.0])];
    let ud = user_displacement(&pairs, &geo, (64, 48));
    let i = 30 * 64 + 40;
    let lvl = &ud.levels[0];
    assert_eq!(lvl.w[i], 2.0);
    let dx = -6.0f32 + (40.2f32 - 44.2f32);
    let dy = -2.0f32 + (29.8f32 - 29.8f32);
    assert_eq!([lvl.d[2 * i], lvl.d[2 * i + 1]], [dx, dy]);
    // the consumer's d / w is the mean of the two displacements
    assert!((lvl.d[2 * i] / lvl.w[i] + 5.0).abs() < 1e-6);
    let j = 10 * 64 + 10;
    assert_eq!((lvl.w[j], lvl.d[2 * j], lvl.d[2 * j + 1]), (1.0, -1.0, 1.0));
    // the order of the points is the order of the accumulation
    let rev = [pairs[1], pairs[0], pairs[2]];
    let ud2 = user_displacement(&rev, &geo, (64, 48));
    assert_eq!(ud2.levels[0].w, lvl.w);
    assert_eq!(ud2.levels[0].d[2 * i], (40.2f32 - 44.2f32) + -6.0f32);
}

#[test]
fn points_outside_the_frame_are_skipped() {
    let geo = geometry(5, 64, 48);
    let skipped = [
        ([-0.6f32, 10.0f32], [0.0f32, 10.0f32]), // nearest pixel would be -1 (cvtt alone gives 0)
        ([63.6, 10.0], [60.0, 10.0]),             // nearest pixel 64
        ([10.0, -0.51], [10.0, 0.0]),
        ([10.0, 47.5], [10.0, 40.0]),
        ([f32::NAN, 3.0], [1.0, 1.0]),
        ([3.0, f32::INFINITY], [1.0, 1.0]),
        ([5.0, 5.0], [f32::INFINITY, 1.0]), // a target that is not a position
        ([5.0, 5.0], [1.0, f32::NAN]),
    ];
    for (k, p) in skipped.iter().enumerate() {
        let ud = user_displacement(std::slice::from_ref(p), &geo, (64, 48));
        for (l, lvl) in ud.levels.iter().enumerate() {
            assert!(lvl.w.iter().all(|&v| v == 0.0) && lvl.d.iter().all(|&v| v == 0.0), "pair {k}, level {l}");
        }
    }
    // just inside: the first and the last pixel
    let inside = [([-0.4f32, 47.4f32], [0.6f32, 47.4f32]), ([63.4, -0.5], [63.4, 0.5])];
    let ud = user_displacement(&inside, &geo, (64, 48));
    let lvl = &ud.levels[0];
    assert_eq!((lvl.w[47 * 64], lvl.d[47 * 64 * 2]), (1.0, -1.0));
    assert_eq!((lvl.w[63], lvl.d[63 * 2 + 1]), (1.0, -1.0));
    for (l, lvl) in ud.levels.iter().enumerate() {
        assert_eq!(lvl.w.iter().filter(|&&v| v != 0.0).count(), 2, "level {l}: in frame points land on every level");
    }
}

#[test]
fn no_points_give_empty_levels_of_the_pyramid_sizes() {
    let geo = geometry(4, 50, 34);
    let ud = user_displacement(&[], &geo, (50, 34));
    assert_eq!(ud.levels.len(), geo.levels);
    for (lvl, &(wl, hl)) in ud.levels.iter().zip(&geo.sizes) {
        assert_eq!((lvl.d.len(), lvl.w.len()), (wl * hl * 2, wl * hl));
        assert!(lvl.d.iter().chain(&lvl.w).all(|v| v.to_bits() == 0));
    }
    // an output that is never searched (a side below 2) lands nothing
    let ud = user_displacement(&[([0.0, 0.0], [1.0, 0.0])], &geo, (1, 34));
    assert!(ud.levels.iter().all(|l| l.w.iter().all(|&v| v == 0.0)));
}

#[test]
fn the_working_resolution_scales_the_points() {
    // Best (R = 1): 64x48 is searched at 32x24, k = (31/63, 23/47) on level 0
    let geo = geometry(4, 64, 48);
    assert_eq!((geo.width, geo.height), (32, 24));
    let ud = user_displacement(&[([40.0, 30.0], [46.0, 32.0])], &geo, (64, 48));
    let i = 15 * 32 + 20;
    assert_eq!(ud.levels[0].w[i], 1.0);
    assert_eq!(ud.levels[0].d[2 * i], -6.0f32 * (31.0f32 / 63.0f32));
    assert_eq!(ud.levels[0].d[2 * i + 1], -2.0f32 * (23.0f32 / 47.0f32));
}

#[test]
fn the_second_tracker_without_points_is_the_exact_zero_field() {
    for (q, w, h) in [(5u8, 64usize, 48usize), (4, 64, 48), (4, 9, 7)] {
        let geo = geometry(q, w, h);
        let f = second_tracker(&[], &geo, (w, h)).unwrap();
        assert_eq!(f, FlowField::new(geo.width, geo.height));
        assert!(f.data.iter().all(|v| v.to_bits() == 0));
        // points that all miss the frame count as none
        let f = second_tracker(&[([-3.0, 4.0], [0.0, 4.0]), ([f32::NAN, 0.0], [0.0, 0.0])], &geo, (w, h)).unwrap();
        assert!(f.data.iter().all(|v| v.to_bits() == 0));
    }
}

#[test]
fn one_point_gives_a_near_uniform_field_in_the_output_convention() {
    // full resolution, the point moves by (+6, +2) from X to Y: the flow X(p) ~ Y(p + f) is
    // about (+6, +2) everywhere
    let geo = geometry(5, 64, 48);
    let f = second_tracker(&[([40.0, 30.0], [46.0, 32.0])], &geo, (64, 48)).unwrap();
    assert_eq!((f.width, f.height), (64, 48));
    assert_eq!(f.at(40, 30), [6.0, 2.0]);
    for y in 0..48 {
        for x in 0..64 {
            let v = f.at(x, y);
            assert!((v[0] - 6.0).abs() < 1e-4 && (v[1] - 2.0).abs() < 1e-4, "({x}, {y}): {v:?}");
        }
    }
    // at the working resolution of Best the values are scaled like the solver's
    let geo = geometry(4, 64, 48);
    let f = second_tracker(&[([40.0, 30.0], [46.0, 32.0])], &geo, (64, 48)).unwrap();
    let want = [-1.0f32 * (-6.0f32 * (31.0f32 / 63.0f32)), -1.0f32 * (-2.0f32 * (23.0f32 / 47.0f32))];
    assert_eq!(f.at(20, 15), want);
    assert!(f.data.chunks(2).all(|v| (v[0] - want[0]).abs() < 1e-4 && (v[1] - want[1]).abs() < 1e-4));
}

#[test]
fn the_second_tracker_interpolates_between_points_and_keeps_them_exact() {
    let geo = geometry(5, 48, 32);
    let pairs = [([8.0, 8.0], [12.0, 8.0]), ([40.0, 24.0], [38.0, 25.0]), ([40.2, 24.1], [38.2, 23.1])];
    let f = second_tracker(&pairs, &geo, (48, 32)).unwrap();
    // the splat of level 0, negated: one point at (8, 8), two averaged at (40, 24)
    let ud = user_displacement(&pairs, &geo, (48, 32));
    let lvl = &ud.levels[0];
    for (x, y) in [(8usize, 8usize), (40, 24)] {
        let i = y * 48 + x;
        let want = [-1.0f32 * (lvl.d[2 * i] / lvl.w[i]), -1.0f32 * (lvl.d[2 * i + 1] / lvl.w[i])];
        assert_eq!(f.at(x, y), want, "({x}, {y})");
    }
    assert_eq!(f.at(8, 8), [4.0, 0.0]);
    assert_eq!(f.at(40, 24), [(-2.0f32 + -2.0f32) / 2.0, (1.0f32 + (23.1f32 - 24.1f32)) / 2.0]);
    // in between, every value lies between the values at the points (x from -2 to 4, y 0)
    for v in f.data.chunks(2) {
        assert!(v[0] >= -2.0 - 1e-4 && v[0] <= 4.0 + 1e-4 && v[1].abs() < 1e-4, "{v:?}");
    }
    assert!(f.data.chunks(2).any(|v| v[0] > 0.0) && f.data.chunks(2).any(|v| v[0] < 0.0));
}

#[test]
fn the_point_primitives_do_not_depend_on_the_thread_count() {
    let geo = geometry(5, 61, 37);
    let pairs = [([3.0, 4.0], [5.5, 4.25]), ([50.7, 30.2], [44.0, 33.0]), ([20.0, 20.0], [20.0, 18.0])];
    let run = |n: usize| in_pool(n, || (second_tracker(&pairs, &geo, (61, 37)).unwrap(), user_displacement(&pairs, &geo, (61, 37))));
    let (a, b) = (run(1), run(4));
    assert_eq!(bits(&a.0.data), bits(&b.0.data));
    assert_eq!(a.1, b.1);
}
