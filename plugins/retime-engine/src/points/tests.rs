//! Tests of the tracking point primitives: the `Pt N: Use` table against create-vectors'
//! `point_belongs`, the 16.16 position reader, the user displacement read from TwixtorPro_64 (the
//! level 0 splat, its mean and sign, the level bounds, a coarser level against a traced one), the second
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

/// The geometry of the traced case `pt-use2-n2` (32x24 at Best: levels 16x12, 8x6, 4x3).
fn traced_geometry() -> FlowGeometry {
    let geo = geometry(4, 32, 24);
    assert_eq!(geo.sizes, vec![(16, 12), (8, 6), (4, 3)]);
    geo
}

#[test]
fn level_0_holds_the_normalised_splat_scaled_by_minus_one() {
    // 64x48 at full resolution: k = 1 on level 0
    let geo = geometry(5, 64, 48);
    let ud = user_displacement(&[([40.0, 30.0], [46.0, 32.0])], &geo, (64, 48));
    assert_eq!(ud.levels.len(), 5);
    let i = 30 * 64 + 40;
    let lvl = &ud.levels[0];
    assert_eq!(lvl.w[i], 1.0);
    assert_eq!([lvl.d[2 * i], lvl.d[2 * i + 1]], [-1.0f32 * (46.0f32 - 40.0f32), -1.0f32 * (32.0f32 - 30.0f32)]);
    assert_eq!(lvl.w.iter().filter(|&&v| v != 0.0).count(), 1);
    // the pixel is (int)((double)(from * k) + 0.5): (40.4, 30.6) lands on (40, 31)
    let ud = user_displacement(&[([40.4, 30.6], [46.0, 32.0])], &geo, (64, 48));
    let i = 31 * 64 + 40;
    assert_eq!(ud.levels[0].w[i], 1.0);
    assert_eq!(ud.levels[0].d[2 * i], -1.0f32 * (46.0f32 - 40.4f32));
    // the difference of the two scaled positions, not the scaled difference: at Best k = (31/63, 23/47)
    let geo = geometry(4, 64, 48);
    let ud = user_displacement(&[([40.0, 30.0], [46.0, 32.0])], &geo, (64, 48));
    let (kx, ky) = (31.0f32 / 63.0f32, 23.0f32 / 47.0f32);
    let i = 15 * 32 + 20;
    assert_eq!(ud.levels[0].d[2 * i], -1.0f32 * (46.0f32 * kx - 40.0f32 * kx));
    assert_eq!(ud.levels[0].d[2 * i + 1], -1.0f32 * (32.0f32 * ky - 30.0f32 * ky));
}

#[test]
fn points_on_one_pixel_give_their_mean_and_a_weight_of_one() {
    let geo = geometry(5, 64, 48);
    let pairs = [([40.0, 30.0], [46.0, 32.0]), ([40.2, 29.8], [44.2, 29.8]), ([10.0, 10.0], [11.0, 9.0])];
    let ud = user_displacement(&pairs, &geo, (64, 48));
    let lvl = &ud.levels[0];
    let i = 30 * 64 + 40;
    assert_eq!(lvl.w[i], 1.0);
    let sx = (46.0f32 - 40.0f32) + 0.0f32;
    let sx = (44.2f32 - 40.2f32) + sx;
    let sy = ((32.0f32 - 30.0f32) + 0.0f32) + (29.8f32 - 29.8f32);
    assert_eq!([lvl.d[2 * i], lvl.d[2 * i + 1]], [-1.0f32 * (sx / 2.0f32), -1.0f32 * (sy / 2.0f32)]);
    let j = 10 * 64 + 10;
    assert_eq!((lvl.w[j], lvl.d[2 * j], lvl.d[2 * j + 1]), (1.0, -1.0, 1.0));
}

#[test]
fn only_the_level_bounds_of_the_start_pixel_drop_a_point() {
    let geo = geometry(5, 64, 48);
    let pairs = [
        ([-0.6f32, 10.0f32], [0.0f32, 10.0f32]), // (int)(-0.1) is 0: lands on the first column
        ([63.6, 10.0], [60.0, 10.0]),             // pixel 64
        ([10.0, 47.5], [10.0, 40.0]),             // pixel 48
        ([-1.6, 10.0], [0.0, 10.0]),              // pixel -1
        ([f32::NAN, 3.0], [1.0, 1.0]),
        ([3.0, f32::INFINITY], [1.0, 1.0]),
    ];
    let landed: Vec<bool> = pairs
        .iter()
        .map(|p| user_displacement(std::slice::from_ref(p), &geo, (64, 48)).levels[0].w.iter().any(|&v| v != 0.0))
        .collect();
    assert_eq!(landed, vec![true, false, false, false, false, false]);
    // the end position is not tested: an infinite end lands and gives an infinite displacement
    let ud = user_displacement(&[([5.0, 5.0], [f32::INFINITY, 1.0])], &geo, (64, 48));
    let i = 5 * 64 + 5;
    assert_eq!(ud.levels[0].w[i], 1.0);
    assert_eq!(ud.levels[0].d[2 * i], f32::NEG_INFINITY);
}

#[test]
fn no_points_give_zero_levels_of_the_pyramid_sizes() {
    let geo = geometry(4, 50, 34);
    let ud = user_displacement(&[], &geo, (50, 34));
    assert_eq!(ud.levels.len(), geo.levels);
    for (lvl, &(wl, hl)) in ud.levels.iter().zip(&geo.sizes) {
        assert_eq!((lvl.d.len(), lvl.w.len()), (wl * hl * 2, wl * hl));
        assert!(lvl.d.iter().chain(&lvl.w).all(|&v| v == 0.0));
    }
    // an output that is never searched (a side below 2) lands nothing
    let ud = user_displacement(&[([0.0, 0.0], [1.0, 0.0])], &geo, (1, 34));
    assert!(ud.levels.iter().all(|l| l.w.iter().all(|&v| v == 0.0)));
}

#[test]
fn the_coarser_levels_are_the_blurred_pyramid_of_level_0_as_traced() {
    // TwixtorPro_64, retime-fgbg.pt-use2-n2.box2-32x24.d8, frame 1, layer FG1, the solve with X = B: the point at
    // (12, 14) in B and (10, 15) in A (rows from the bottom, H - 1 - y). Level 1 (8x6) as the plugin's applier
    // received it, rows from the bottom: the nine samples of w and d around the point, nothing elsewhere.
    let geo = traced_geometry();
    let ud = user_displacement(&[([12.0, 14.0], [10.0, 15.0])], &geo, (32, 24));
    let l1 = &ud.levels[1];
    let w_want: [[u32; 3]; 3] = [
        [0x3b3e2be3, 0x3c03a83d, 0x3a2f8b03], // row 2
        [0x3d1a83a8, 0x3dd5f162, 0x3c0ea0f1], // row 3
        [0x3c3e2bdc, 0x3d03a838, 0x3b2f8afc], // row 4
    ];
    let d_want: [[[u32; 2]; 3]; 3] = [
        [[0x3aabc48c, 0xba255dd3], [0x3b6dd515, 0xbae4f816], [0x399e8e14, 0xb918a56a]],
        [[0x3c8b8fb1, 0xbc065c3c], [0x3d413d20, 0xbcba0990], [0x3b80d36f, 0xbaf80cca]],
        [[0x3babc485, 0xbb255dcd], [0x3c6dd50c, 0xbbe4f80c], [0x3a9e8e0c, 0xba18a565]],
    ];
    for y in 0..6usize {
        for x in 0..8usize {
            let i = y * 8 + x;
            if (2..=4).contains(&y) && (2..=4).contains(&x) {
                let (wv, dv) = (w_want[y - 2][x - 2], d_want[y - 2][x - 2]);
                assert_eq!(l1.w[i].to_bits(), wv, "w ({x}, {y})");
                assert_eq!([l1.d[2 * i].to_bits(), l1.d[2 * i + 1].to_bits()], dv, "d ({x}, {y})");
            } else {
                assert_eq!(l1.w[i], 0.0, "w ({x}, {y})");
                assert_eq!([l1.d[2 * i], l1.d[2 * i + 1]], [0.0, 0.0], "d ({x}, {y})");
            }
        }
    }
    // the consumer's d / w is the point's displacement at that level's scale where the weight reaches
    let i = 3 * 8 + 3;
    let want = [(7.0f32 / 15.0) * 2.0 * (15.0 / 31.0), (5.0f32 / 11.0) * -1.0 * (11.0 / 23.0)];
    assert!((l1.d[2 * i] / l1.w[i] - want[0]).abs() < 1e-5 && (l1.d[2 * i + 1] / l1.w[i] - want[1]).abs() < 1e-5);
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
    // the splat of level 0 (rule G5's splat), negated: one point at (8, 8), two averaged at (40, 24)
    let lvl = splat(&pairs, 48, 32, (48, 32));
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
