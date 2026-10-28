//! Tests of the solver hooks: the kernel variants of `flow.rs` and the driver of
//! `flow/extras.rs`. Identity: with no extras and wrap off every entry point gives the plain
//! solve bit for bit (against a frozen copy of the plain driver), and extras that cannot act
//! change nothing. Hand goldens of every masked and wrapped kernel, the mask pyramid and the
//! user displacement consumer; the numpy float32 model fixtures (`tests/fixtures/gen_mask.py`,
//! `mask_cases.hex`); thread count independence.

use super::extras::{solve_pyramids_with, MaskPyramid, SolveExtras, UserDisplacement, UserLevel};
use super::*;
use crate::image::{Fmt, Image, Samples};
use crate::matte::Matte;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn pool<T: Send>(n: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap().install(f)
}

/// A smooth texture with content at several scales, values about 0..255.
fn texture(x: f64, y: f64, seed: u32) -> f64 {
    let t = std::f64::consts::TAU;
    let s = seed as f64 * 0.7;
    128.0
        + 45.0 * (t * (x / 23.0 + y / 31.0) + s).sin()
        + 35.0 * (t * (x / 11.0 - y / 13.0) + 2.0 * s).sin()
        + 25.0 * (t * (x / 5.3 + y / 6.1) + 0.5 * s).sin()
        + 10.0 * (t * (x / 2.9 - y / 3.7) + 1.5 * s).sin()
}

/// Gray image of `texture(x - sx, y - sy)` at `fmt` (alpha white).
fn image(w: usize, h: usize, fmt: Fmt, sx: f64, sy: f64, seed: u32) -> Image {
    let mut im = Image::new(w, h, fmt);
    for y in 0..h {
        for x in 0..w {
            let v = texture(x as f64 - sx, y as f64 - sy, seed).clamp(0.0, 255.0);
            let i = (y * w + x) * 4;
            match &mut im.samples {
                Samples::U8(s) => {
                    let q = (v + 0.5).floor() as u8;
                    s[i..i + 4].copy_from_slice(&[255, q, q, q]);
                }
                Samples::U16(s) => {
                    let q = (v * 257.0 + 0.5).floor() as u16;
                    s[i..i + 4].copy_from_slice(&[65535, q, q, q]);
                }
                Samples::F32(s) => {
                    let q = (v / 255.0) as f32;
                    s[i..i + 4].copy_from_slice(&[1.0, q, q, q]);
                }
            }
        }
    }
    im
}

/// The plain driver as it stood before the solver hooks (the body of `solve_pyramids` with wrap
/// off), frozen here as the reference of the identity tests.
fn reference_solve(x: &Pyramid, y: &Pyramid, settings: &FlowSettings) -> FlowField {
    let geo = &x.geometry;
    let levels = geo.levels;
    let (start, damp) = start_level(settings.sensitivity, levels);
    let (sw0, sh0) = geo.sizes[start];
    let mut flow = vec![0.0f32; sw0 * sh0 * 2];
    let mut check = true;
    for l in (0..=start).rev() {
        let (w, h) = geo.sizes[l];
        let img_a = &x.levels[l].data;
        let img_b = &y.levels[l].data;
        let grad = calc_grad(img_a, w, h);
        let mut a = calc_a(&grad, w, h);
        conv::blur_gaussian(&mut a, w, h, 3, 7.0);
        for it in 0..3 {
            let mut b = calc_b(img_a, img_b, &grad, &flow, w, h);
            conv::blur_gaussian(&mut b, w, h, 2, 7.0);
            new_disp(&mut flow, &a, &b);
            conv::blur_gaussian(&mut flow, w, h, 2, 5.0);
            if it == 1 && l == 0 && !settings.skip_refine {
                block_match(img_a, img_b, &mut flow, w, h);
            }
        }
        if l != 0 && !settings.skip_refine {
            block_match(img_a, img_b, &mut flow, w, h);
        }
        if check {
            if all_flow_out_of_range(&flow, w, h) {
                flow.iter_mut().for_each(|v| *v = 0.0);
            } else {
                check = false;
            }
        }
        if l == start && (damp as f64 - 1.0).abs() >= 1e-6 {
            scale_values(&mut flow, damp);
        }
        if l > 0 {
            let (nw, nh) = geo.sizes[l - 1];
            flow = upsample_flow(&flow, w, h, nw, nh);
        }
    }
    scale_values(&mut flow, -1.0f32);
    FlowField { width: geo.width, height: geo.height, data: flow }
}

/// A user displacement that sets nothing: garbage displacements under weights of 0.0 and -0.0.
fn inert_user(geo: &FlowGeometry) -> UserDisplacement {
    let levels = geo
        .sizes
        .iter()
        .map(|&(w, h)| UserLevel {
            d: (0..w * h * 2).map(|i| (i % 7) as f32 * 3.5 - 9.0).collect(),
            w: (0..w * h).map(|i| if i % 2 == 0 { 0.0 } else { -0.0 }).collect(),
        })
        .collect();
    UserDisplacement { levels }
}

fn ones_mask(w: usize, h: usize, s: &FlowSettings) -> MaskPyramid {
    MaskPyramid::build(&Matte { width: w, height: h, data: vec![1.0; w * h] }, s).unwrap()
}

// ---------------------------------------------------------------------------------------
// Identity

#[test]
fn the_plain_settings_do_not_wrap_and_add_nothing() {
    assert!(!FlowSettings::new(4, true, 0.7).wrap);
    assert!(!FlowSettings::default().wrap);
    let e = SolveExtras::default();
    assert!(e.masks.is_none() && e.user.is_none());
    assert_eq!(UserDisplacement::default().levels, Vec::new());
}

#[test]
fn no_extras_and_inert_extras_give_the_old_solve_bit_for_bit() {
    let sizes = [(24usize, 16usize), (37, 29), (64, 48), (13, 9), (5, 3), (2, 2), (3, 7)];
    let mut n = 0;
    for (si, &(w, h)) in sizes.iter().enumerate() {
        for (di, fmt) in [Fmt::U8, Fmt::U16, Fmt::F32].into_iter().enumerate() {
            let a = image(w, h, fmt, 0.0, 0.0, si as u32);
            let b = image(w, h, fmt, 1.75, -0.5, si as u32);
            for q in 1..=5u8 {
                for hiq in [true, false] {
                    for (ki, sens) in [0.0f32, 0.35, 0.7, 1.0, 1.7].into_iter().enumerate() {
                        let k = si + di + q as usize + ki;
                        let mut s = FlowSettings::new(q, hiq, sens);
                        s.prep_contrast_edge = k % 4 == 1;
                        s.prep_delinearize = k % 4 == 2;
                        s.skip_refine = k % 5 == 3;
                        let case = format!("{w}x{h} {fmt:?} q {q} hiq {hiq} sens {sens} {s:?}");
                        let (pa, pb) = (build_pyramid(&a, &s).unwrap(), build_pyramid(&b, &s).unwrap());
                        let want = bits(&reference_solve(&pa, &pb, &s).data);
                        assert_eq!(bits(&solve_pyramids(&pa, &pb, &s).unwrap().data), want, "solve_pyramids {case}");
                        assert_eq!(bits(&solve(&a, &b, &s).unwrap().data), want, "solve {case}");
                        let e = SolveExtras::default();
                        assert_eq!(bits(&solve_pyramids_with(&pa, &pb, &s, e).unwrap().data), want, "default extras {case}");
                        // an empty user displacement is none
                        let empty = UserDisplacement::default();
                        let e = SolveExtras { masks: None, user: Some(&empty) };
                        assert_eq!(bits(&solve_pyramids_with(&pa, &pb, &s, e).unwrap().data), want, "empty user {case}");
                        // the extended driver itself: weights of 0 set nothing, an all ones mask
                        // gates nothing (finite inputs)
                        let user = inert_user(pa.geometry());
                        let ones = ones_mask(w, h, &s);
                        for e in [
                            SolveExtras { masks: None, user: Some(&user) },
                            SolveExtras { masks: Some((&ones, &ones)), user: None },
                            SolveExtras { masks: Some((&ones, &ones)), user: Some(&user) },
                        ] {
                            let got = solve_pyramids_with(&pa, &pb, &s, e).unwrap();
                            assert_eq!(bits(&got.data), want, "inert extras {case} masks {}", e.masks.is_some());
                        }
                        n += 1;
                    }
                }
            }
        }
    }
    assert_eq!(n, 7 * 3 * 10 * 5);
}

#[test]
fn solve_pair_equals_two_solves_through_every_path() {
    let (a, b) = (image(40, 30, Fmt::F32, 0.0, 0.0, 3), image(40, 30, Fmt::F32, -2.25, 1.0, 3));
    for wrap in [false, true] {
        let s = FlowSettings { wrap, ..FlowSettings::new(5, true, 0.9) };
        let p = pool(4, || solve_pair(&a, &b, &s).unwrap());
        assert_eq!(bits(&p.f_ab.data), bits(&pool(1, || solve(&a, &b, &s).unwrap()).data), "wrap {wrap}");
        assert_eq!(bits(&p.f_ba.data), bits(&pool(1, || solve(&b, &a, &s).unwrap()).data), "wrap {wrap}");
        let (pa, pb) = (build_pyramid(&a, &s).unwrap(), build_pyramid(&b, &s).unwrap());
        let e = SolveExtras::default();
        assert_eq!(solve_pyramids(&pa, &pb, &s).unwrap(), solve_pyramids_with(&pa, &pb, &s, e).unwrap(), "wrap {wrap}");
    }
}

#[test]
fn a_wrap_solve_runs_and_differs_from_the_plain_one_only_through_the_wrap() {
    let (a, b) = (image(48, 32, Fmt::U8, 0.0, 0.0, 5), image(48, 32, Fmt::U8, 2.0, 0.0, 5));
    let plain = FlowSettings::new(5, true, 0.9);
    let wrap = FlowSettings { wrap: true, ..plain };
    // the flag reaches the pyramids (rule G6): at full resolution level 0 differs only in
    // the three columns of each side that the circular 5 tap blur reads across the seam, the levels
    // below are resampled with the period kept; the geometry is the same
    let (pa, pb) = (build_pyramid(&a, &plain).unwrap(), build_pyramid(&b, &plain).unwrap());
    let (wa, wb) = (build_pyramid(&a, &wrap).unwrap(), build_pyramid(&b, &wrap).unwrap());
    assert_eq!(pa.geometry(), wa.geometry());
    for (p, q) in [(&pa, &wa), (&pb, &wb)] {
        let (w, h) = p.geometry().sizes[0];
        for y in 0..h {
            for x in 3..w - 3 {
                assert_eq!(p.level(0).at(x, y).to_bits(), q.level(0).at(x, y).to_bits(), "({x}, {y})");
            }
        }
        assert_ne!(bits(&p.level(0).data), bits(&q.level(0).data));
        assert_ne!(bits(&p.level(1).data), bits(&q.level(1).data));
    }
    // the wrap kernels on the plain pyramids, and on the periodic ones
    let fk = solve_pyramids(&pa, &pb, &wrap).unwrap();
    let fw = solve(&a, &b, &wrap).unwrap();
    assert_eq!(bits(&fw.data), bits(&solve_pyramids(&wa, &wb, &wrap).unwrap().data));
    let fp = solve(&a, &b, &plain).unwrap();
    assert_eq!((fw.width, fw.height), (fp.width, fp.height));
    assert_ne!(bits(&fk.data), bits(&fp.data), "the wrap kernels run");
    assert_ne!(bits(&fw.data), bits(&fk.data), "the periodic pyramids reach the solve");
    assert!(fw.data.iter().all(|v| v.is_finite()));
}

#[test]
fn an_all_ones_mask_can_change_only_a_b_vector_whose_target_is_nan() {
    // finite flows: the warped mask of an all ones level is exactly 1.0, so B is the plain one
    let (w, h) = (6usize, 5usize);
    let i1: Vec<f32> = (0..w * h).map(|i| (i * 13 % 17) as f32 * 1.5).collect();
    let i0: Vec<f32> = (0..w * h).map(|i| (i * 7 % 11) as f32 * 2.25).collect();
    let grad: Vec<f32> = (0..w * h * 2).map(|i| (i % 5) as f32 - 2.0).collect();
    let mut flow: Vec<f32> = (0..w * h * 2).map(|i| ((i * 37 % 19) as f32 - 9.0) * 0.37).collect();
    let ones = vec![1.0f32; w * h];
    let plain = calc_b(&i1, &i0, &grad, &flow, w, h);
    assert_eq!(bits(&calc_b_with(&i1, &i0, &grad, &flow, w, h, Some(&ones), false)), bits(&plain));
    assert_eq!(bits(&calc_b_with(&i1, &i0, &grad, &flow, w, h, None, false)), bits(&plain));
    // a NaN target: the plain step samples NaN and propagates it, the mask test fails and gives 0
    let p = 2 * w + 3;
    flow[p * 2] = f32::NAN;
    let plain = calc_b(&i1, &i0, &grad, &flow, w, h);
    let masked = calc_b_with(&i1, &i0, &grad, &flow, w, h, Some(&ones), false);
    assert!(plain[p * 2].is_nan() || plain[p * 2 + 1].is_nan(), "{:?}", &plain[p * 2..p * 2 + 2]);
    assert_eq!((masked[p * 2], masked[p * 2 + 1]), (0.0, 0.0));
    for i in 0..w * h {
        if i != p {
            assert_eq!(bits(&masked[i * 2..i * 2 + 2]), bits(&plain[i * 2..i * 2 + 2]), "pixel {i}");
        }
    }
    // the gradient masking and the sweep gate of an all ones mask never act
    let mut g = grad.clone();
    mask_gradient(&mut g, &ones, w, h, false);
    assert_eq!(bits(&g), bits(&grad));
    let img: Vec<f32> = (0..w * h).map(|i| ((i * 7 + 3) % 17) as f32 * 3.0).collect();
    let f0: Vec<f32> = (0..w * h * 2).map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.3).collect();
    let (mut fa, mut fb) = (f0.clone(), f0);
    block_match(&img, &i0, &mut fa, w, h);
    block_match_with(&img, &i0, &mut fb, w, h, Some(&ones), false);
    assert_eq!(bits(&fa), bits(&fb));
}

// ---------------------------------------------------------------------------------------
// Kernel goldens

#[test]
fn wrap_gradient_hand_values() {
    let (w, h) = (4usize, 3usize);
    let img = [1.0f32, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 3.0, 5.0, 7.0, 11.0];
    let g = calc_grad_wrap(&img, w, h);
    // gx = (I(x + 1 mod 4) - I(x - 1 mod 4)) / 2 on row 0
    let gx0: Vec<f32> = (0..4).map(|x| g[x * 2]).collect();
    assert_eq!(gx0, [(2.0 - 8.0) / 2.0, (4.0 - 1.0) / 2.0, (8.0 - 2.0) / 2.0, (1.0 - 4.0) / 2.0]);
    assert_eq!(g[(4 + 3) * 2], (16.0 - 64.0) / 2.0, "row 1, last column reads column 0");
    // gy is the plain one at every pixel, gx the plain one inside (an integer image is exact)
    let p = calc_grad(&img, w, h);
    for i in 0..w * h {
        assert_eq!(g[i * 2 + 1].to_bits(), p[i * 2 + 1].to_bits(), "gy {i}");
        if i % w != 0 && i % w != w - 1 {
            assert_eq!(g[i * 2].to_bits(), p[i * 2].to_bits(), "gx {i}");
        }
    }
    assert_eq!(p[0], 1.0, "the plain gradient is one sided at the seam");
    // a width of 2 reads the same column on both sides
    let g2 = calc_grad_wrap(&[1.0, 5.0, 2.0, 9.0], 2, 2);
    assert_eq!((g2[0], g2[2], g2[4], g2[6]), (0.0, 0.0, 0.0, 0.0));
    assert_eq!((g2[1], g2[3]), (1.0, 4.0));
}

#[test]
fn gradient_masking_hand_values() {
    let (w, h) = (5usize, 4usize);
    let grad: Vec<f32> = (0..w * h).flat_map(|i| [i as f32 + 1.0, -(i as f32) - 1.0]).collect();
    let zeroed = |mask: &[f32], wrap: bool| -> Vec<usize> {
        let mut g = grad.clone();
        mask_gradient(&mut g, mask, w, h, wrap);
        (0..w * h)
            .filter(|&i| {
                let z = (g[i * 2], g[i * 2 + 1]) == (0.0, 0.0);
                assert!(z || (g[i * 2], g[i * 2 + 1]) == (grad[i * 2], grad[i * 2 + 1]), "pixel {i} is either zeroed or untouched");
                z
            })
            .collect()
    };
    // a hole at (2, 1): the pixel and its four neighbours
    let mut m = vec![1.0f32; w * h];
    m[w + 2] = 0.0;
    assert_eq!(zeroed(&m, false), [2, w + 1, w + 2, w + 3, 2 * w + 2]);
    assert_eq!(zeroed(&m, true), [2, w + 1, w + 2, w + 3, 2 * w + 2]);
    // a hole on the first column: the neighbours that exist; with wrap the last column too
    let mut m = vec![1.0f32; w * h];
    m[2 * w] = 0.0;
    assert_eq!(zeroed(&m, false), [w, 2 * w, 2 * w + 1, 3 * w]);
    assert_eq!(zeroed(&m, true), [w, 2 * w, 2 * w + 1, 2 * w + 4, 3 * w]);
    // a corner hole
    let mut m = vec![1.0f32; w * h];
    m[w * h - 1] = 0.0;
    assert_eq!(zeroed(&m, false), [3 * w - 1, w * h - 2, w * h - 1]);
    assert_eq!(zeroed(&m, true), [3 * w - 1, 3 * w, w * h - 2, w * h - 1]);
    // the threshold: 0.95 is inside, the float below it is not
    let mut m = vec![1.0f32; w * h];
    m[w + 2] = 0.95;
    assert!(zeroed(&m, false).is_empty());
    m[w + 2] = f32::from_bits(0.95f32.to_bits() - 1);
    assert_eq!(zeroed(&m, false).len(), 5);
}

#[test]
fn masked_b_step_needs_the_warped_mask_at_0_999() {
    let (w, h) = (4usize, 3usize);
    let i1 = vec![0.0f32; w * h];
    let i0 = vec![10.0f32; w * h];
    let grad = [1.0f32, -1.0].repeat(w * h);
    // columns 0 and 1 inside, 2 and 3 outside
    let mask: Vec<f32> = (0..w * h).map(|i| if i % w < 2 { 1.0 } else { 0.0 }).collect();
    let mut flow = vec![0.0f32; w * h * 2];
    let set = |f: &mut Vec<f32>, x: usize, y: usize, v: [f32; 2]| {
        f[(y * w + x) * 2] = v[0];
        f[(y * w + x) * 2 + 1] = v[1];
    };
    // (1, 0): target x 1 + 2^-10, mask 1 - 2^-10 = 0.99902344 >= 0.999 passes
    set(&mut flow, 1, 0, [-0.0009765625, 0.0]);
    // (1, 1): target x 1 + 2^-9, mask 0.99804688 fails
    set(&mut flow, 1, 1, [-0.001953125, 0.0]);
    // (0, 2): target (0.5, 1.5): mask 1.0 between two inside columns
    set(&mut flow, 0, 2, [-0.5, 0.5]);
    let b = calc_b_with(&i1, &i0, &grad, &flow, w, h, Some(&mask), false);
    let at = |x: usize, y: usize| (b[(y * w + x) * 2], b[(y * w + x) * 2 + 1]);
    assert_eq!(at(1, 0), (10.0, -10.0));
    assert_eq!(at(1, 1), (0.0, 0.0));
    assert_eq!(at(0, 2), (10.0, -10.0));
    assert_eq!(at(0, 0), (10.0, -10.0));
    for y in 0..h {
        assert_eq!(at(2, y), (0.0, 0.0), "column 2 is outside");
        assert_eq!(at(3, y), (0.0, 0.0), "column 3 is outside");
    }
    // without the mask every in range target passes
    let plain = calc_b(&i1, &i0, &grad, &flow, w, h);
    assert!(plain.chunks(2).all(|v| v == [10.0, -10.0]));
}

#[test]
fn wrapped_b_step_reads_across_the_seam() {
    let (w, h) = (4usize, 3usize);
    let i1 = vec![0.0f32; w * h];
    let i0: Vec<f32> = (0..w * h).map(|i| [1.0f32, 2.0, 4.0, 8.0][i % w]).collect();
    let grad = [1.0f32, 2.0].repeat(w * h);
    let mut flow = vec![0.0f32; w * h * 2];
    let set = |f: &mut Vec<f32>, x: usize, y: usize, v: [f32; 2]| {
        f[(y * w + x) * 2] = v[0];
        f[(y * w + x) * 2 + 1] = v[1];
    };
    set(&mut flow, 0, 0, [0.5, 0.0]); // sx -0.5 -> 3.5: (1 - 8) * 0.5 + 8
    set(&mut flow, 3, 0, [-0.25, 0.0]); // sx 3.25: (1 - 8) * 0.25 + 8
    set(&mut flow, 1, 0, [5.0, 0.0]); // sx -4 -> 0
    set(&mut flow, 1, 1, [-6.5, 0.0]); // sx 7.5 -> 3.5
    set(&mut flow, 2, 1, [0.0, 1.5]); // sy -0.5: outside the rows
    set(&mut flow, 3, 2, [f32::NAN, 0.0]); // a column the loops cannot wrap
    set(&mut flow, 0, 1, [f32::NEG_INFINITY, 0.0]); // sx +inf: the loop bound
    set(&mut flow, 2, 2, [-1.0e30, 0.0]); // beyond the bound
    let b = calc_b_with(&i1, &i0, &grad, &flow, w, h, None, true);
    let at = |x: usize, y: usize| (b[(y * w + x) * 2], b[(y * w + x) * 2 + 1]);
    assert_eq!(at(0, 0), (4.5, 9.0));
    assert_eq!(at(3, 0), (6.25, 12.5));
    assert_eq!(at(1, 0), (1.0, 2.0));
    assert_eq!(at(1, 1), (4.5, 9.0));
    assert_eq!(at(2, 1), (0.0, 0.0));
    assert_eq!(at(3, 2), (0.0, 0.0));
    assert_eq!(at(0, 1), (0.0, 0.0));
    assert_eq!(at(2, 2), (0.0, 0.0));
    assert_eq!(at(2, 0), (4.0, 8.0), "zero flow samples its own pixel");
    // the plain step zeroes the targets left of 0 and right of w - 1
    let plain = calc_b(&i1, &i0, &grad, &flow, w, h);
    assert_eq!((plain[0], plain[1]), (0.0, 0.0));
    assert_eq!((plain[6], plain[7]), (0.0, 0.0));
    // the warped mask wraps the same way: column 0 inside, column 3 outside
    let mask: Vec<f32> = (0..w * h).map(|i| if i % w == 3 { 0.0 } else { 1.0 }).collect();
    let mut f = vec![0.0f32; w * h * 2];
    set(&mut f, 0, 0, [4.0, 0.0]); // sx -4 -> 0.0: mask 1
    set(&mut f, 1, 0, [1.5, 0.0]); // sx -0.5 -> 3.5: mask (1 - 0) * 0.5 + 0 = 0.5 fails
    let bm = calc_b_with(&i1, &i0, &grad, &f, w, h, Some(&mask), true);
    assert_eq!((bm[0], bm[1]), (1.0, 2.0));
    assert_eq!((bm[2], bm[3]), (0.0, 0.0));
}

#[test]
fn the_wrap_target_splits_the_column_and_keeps_its_bound() {
    // pixel column x displaced by -u: the whole column modulo the width and the fraction
    assert_eq!(wrap_target(0, 0.5, 4), Some((3, 0.5)), "x - u = -0.5 wraps to 3.5");
    assert_eq!(wrap_target(0, -4.0, 4), Some((0, 0.0)), "4.0 is column 0");
    assert_eq!(wrap_target(0, 4.0, 4), Some((0, 0.0)), "-4.0 is column 0");
    assert_eq!(wrap_target(1, -10.0, 4), Some((3, 0.0)), "11.0 is column 3");
    assert_eq!(wrap_target(2, 0.0, 4), Some((2, 0.0)));
    assert_eq!(wrap_target(2, -0.0, 4), Some((2, 0.0)));
    assert_eq!(wrap_target(3, -0.25, 4), Some((3, 0.25)));
    // a tiny negative displacement: the fraction rounds to 1.0, the column moves one right
    assert_eq!(wrap_target(0, 1.0e-9, 640), Some((0, 0.0)));
    assert_eq!(wrap_target(639, 1.0e-9, 640), Some((639, 0.0)));
    // the fraction depends on the flow only, never on the column: the same at every x
    for &u in &[0.3f32, -0.7, 2.6, -13.125, 1.0e-7] {
        let (c0, f0) = wrap_target(0, u, 50).unwrap();
        for x in 1..50 {
            let (c, f) = wrap_target(x, u, 50).unwrap();
            assert_eq!(f.to_bits(), f0.to_bits(), "u {u} x {x}");
            assert_eq!(c, (c0 + x) % 50, "u {u} x {x}");
        }
    }
    // where `x - u` is exact the split is that target wrapped into [0, W)
    for &(x, u) in &[(3usize, 0.5f32), (0, 1.25), (5, -2.75), (7, 9.5)] {
        let mut t = x as f32 - u;
        while t < 0.0 {
            t += 8.0;
        }
        while t >= 8.0 {
            t -= 8.0;
        }
        let (c, f) = wrap_target(x, u, 8).unwrap();
        assert_eq!((c as f32 + f).to_bits(), t.to_bits(), "({x}, {u})");
    }
    assert_eq!(wrap_target(0, f32::NAN, 4), None);
    assert_eq!(wrap_target(0, f32::INFINITY, 4), None);
    assert_eq!(wrap_target(0, f32::NEG_INFINITY, 4), None);
    // 65536 widths wrap, more do not
    assert_eq!(wrap_target(0, 4.0 * 65536.0, 4), Some((0, 0.0)));
    assert_eq!(wrap_target(0, 4.0 * 65537.0, 4), None);
    assert_eq!(wrap_target(1, -(4.0 * 65536.0), 4), Some((1, 0.0)));
}

#[test]
fn the_wrap_samplers_take_column_0_right_of_the_seam() {
    let (w, h) = (3usize, 2usize);
    let img = [0.0f32, 10.0, 20.0, 100.0, 110.0, 120.0];
    assert_eq!(bilinear1_wrapx_at(&img, w, h, 2, 0.5, 0.0), 10.0, "(0 - 20) * 0.5 + 20");
    assert_eq!(bilinear1_wrapx_at(&img, w, h, 2, 0.5, 0.5), 60.0);
    assert_eq!(bilinear1_wrapx_at(&img, w, h, 2, 0.0, 1.0), 120.0, "the last row moves the cell up with fy 1");
    for &(ix, fx, y) in &[(0usize, 0.0f32, 0.0f32), (0, 0.3, 0.7), (1, 0.75, 0.25), (1, 0.9375, 1.0), (0, 0.0, 0.999), (2, 0.25, 0.5)] {
        let x = ix as f32 + fx;
        assert_eq!(bilinear1_wrapx_at(&img, w, h, ix, fx, y).to_bits(), sample::bilinear1_wrapx(&img, w, h, x, y).to_bits(), "({x}, {y})");
        if x < 2.0 {
            assert_eq!(bilinear1_wrapx_at(&img, w, h, ix, fx, y).to_bits(), bilinear1(&img, w, h, x, y).to_bits(), "({x}, {y})");
        }
    }
    // the 2 channel sampler of the upsample: each lane as the 1 channel one
    let two: Vec<f32> = img.iter().flat_map(|&v| [v, -3.0 * v]).collect();
    for &(x, y) in &[(0.0f32, 0.0f32), (2.5, 0.25), (1.75, 1.0), (2.999, 0.5)] {
        let v = sample::bilinear2_wrapx(&two, w, h, x, y);
        let lane1: Vec<f32> = img.iter().map(|&v| -3.0 * v).collect();
        assert_eq!(v[0].to_bits(), sample::bilinear1_wrapx(&img, w, h, x, y).to_bits(), "({x}, {y})");
        assert_eq!(v[1].to_bits(), sample::bilinear1_wrapx(&lane1, w, h, x, y).to_bits(), "({x}, {y})");
    }
}

#[test]
fn wrap_block_cost_hand_values() {
    let (w, h) = (8usize, 8usize);
    let a = vec![0.0f32; w * h];
    let b: Vec<f32> = (0..w * h).map(|i| (i % w) as f32 + 10.0 * (i / w) as f32).collect();
    let bm = Block { a: &a, b: &b, w, h };
    // the cost at the target x2 of pixel column 0 (displaced by -u = x2)
    let cost = |bm: &Block, p: &[f32; 9], x2: f32, y2: f32| wrap_target(0, -x2, w).and_then(|(c, f)| bm.cost_wrap(p, c, f, y2));
    // x2 = 0.5, y2 = 3: window columns 7, 0, 1, 2 and rows 2..=5; fy = 0, so
    // q(a, b) = 0.5 * B(col a) + 0.5 * B(col a + 1) on row 2 + b
    let cols = [7.0f32, 0.0, 1.0, 2.0];
    let mut p = [0.0f32; 9];
    let mut n = 0;
    for r in 0..3 {
        for c in 0..3 {
            let row = 10.0 * (2 + r) as f32;
            p[n] = (0.5 * (cols[c] + row) + 0.5 * (cols[c + 1] + row)) + 0.0 * row;
            n += 1;
        }
    }
    assert_eq!(cost(&bm, &p, 0.5, 3.0), Some(0.0));
    p[4] += 2.0;
    assert_eq!(cost(&bm, &p, 0.5, 3.0), Some(4.0));
    p[4] -= 2.0;
    // the same target one width to the left or right
    assert_eq!(cost(&bm, &p, -7.5, 3.0), Some(0.0));
    assert_eq!(cost(&bm, &p, 16.5, 3.0), Some(0.0));
    // x2 = -0.5 wraps to 7.5: window columns 6, 7, 0, 1
    let cols = [6.0f32, 7.0, 0.0, 1.0];
    let mut n = 0;
    for r in 0..3 {
        for c in 0..3 {
            let row = 10.0 * (2 + r) as f32;
            p[n] = (0.5 * (cols[c] + row) + 0.5 * (cols[c + 1] + row)) + 0.0 * row;
            n += 1;
        }
    }
    assert_eq!(cost(&bm, &p, -0.5, 3.0), Some(0.0));
    // the rows keep the plain rules
    assert_eq!(cost(&bm, &p, 3.0, 0.5), None, "iy = 0");
    assert_eq!(cost(&bm, &p, 3.0, 6.01), None, "y2 > h - 2");
    assert!(cost(&bm, &p, 3.0, 6.0).is_some());
    assert_eq!(cost(&bm, &p, f32::NAN, 3.0), None);
    assert_eq!(cost(&bm, &p, 3.0, f32::NAN), None);
    // inside 1 <= x2 < w - 2 it is the plain cost
    let b2: Vec<f32> = (0..w * h).map(|i| ((i * 29 % 23) as f32 - 7.0) * 1.37).collect();
    let bm2 = Block { a: &a, b: &b2, w, h };
    let p2 = [1.5f32, -2.0, 3.25, 0.5, 7.0, -1.0, 2.0, 4.5, -3.0];
    for &(x2, y2) in &[(1.0f32, 1.0f32), (2.25, 4.75), (5.5, 2.5), (3.0, 6.0), (1.125, 5.5)] {
        assert_eq!(cost(&bm2, &p2, x2, y2).map(f32::to_bits), bm2.cost(&p2, x2, y2).map(f32::to_bits), "({x2}, {y2})");
    }
    // the wrapped patch of a seam column
    let img: Vec<f32> = (0..w * h).map(|i| i as f32).collect();
    let bm3 = Block { a: &img, b: &img, w, h };
    assert_eq!(bm3.patch_wrap(0, 1), [7.0, 0.0, 1.0, 15.0, 8.0, 9.0, 23.0, 16.0, 17.0]);
    assert_eq!(bm3.patch_wrap(3, 2), bm3.patch(3, 2));
}

fn block_image(w: usize, h: usize, f: impl Fn(usize, usize) -> f32) -> Vec<f32> {
    (0..h).flat_map(|y| (0..w).map(|x| f(x, y)).collect::<Vec<_>>()).collect()
}

#[test]
fn gated_sweeps_skip_the_pixels_outside_the_mask_only() {
    let (w, h) = (12usize, 10usize);
    let img = block_image(w, h, |x, y| ((x * 7 + y * 13) % 17) as f32 * 3.0 + (x as f32) * 0.5);
    let mut f0 = vec![0.0f32; w * h * 2];
    let c = (5 * w + 6) * 2;
    f0[c] = 2.0;
    f0[c + 1] = 1.0;
    let run = |gate: Option<&[f32]>| {
        let mut f = f0.clone();
        block_match_with(&img, &img, &mut f, w, h, gate, false);
        f
    };
    let mut plain = f0.clone();
    block_match(&img, &img, &mut plain, w, h);
    assert_eq!((plain[c], plain[c + 1]), (0.0, 0.0), "the plain sweeps fix the wrong vector");
    assert_eq!(bits(&run(None)), bits(&plain));
    let ones = vec![1.0f32; w * h];
    assert_eq!(bits(&run(Some(&ones))), bits(&plain));
    // the wrong pixel outside the mask keeps its vector; nothing else changes either
    let mut gate = ones.clone();
    gate[5 * w + 6] = 0.0;
    let got = run(Some(&gate));
    assert_eq!(bits(&got), bits(&f0));
    // the cost stays unmasked: a pixel inside the mask whose neighbour lies outside still takes
    // the neighbour's better vector (the gate reads only the pixel being stepped)
    let mut gate = vec![0.0f32; w * h];
    gate[5 * w + 6] = 1.0;
    let got = run(Some(&gate));
    assert_eq!((got[c], got[c + 1]), (0.0, 0.0));
    // all outside: nothing moves
    assert_eq!(bits(&run(Some(&vec![0.0f32; w * h]))), bits(&f0));
    // thread count independence of the gated wrap sweeps on a noisy field
    let noisy: Vec<f32> = (0..w * h * 2).map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.3).collect();
    let gate: Vec<f32> = (0..w * h).map(|i| if (i * 5 % 7) < 5 { 1.0 } else { 0.0 }).collect();
    let go = |n: usize| {
        let mut f = noisy.clone();
        pool(n, || block_match_with(&img, &img, &mut f, w, h, Some(&gate), true));
        f
    };
    assert_eq!(bits(&go(1)), bits(&go(4)));
    assert_eq!(bits(&go(1)), bits(&go(16)));
}

#[test]
fn wrap_sweeps_refine_a_vector_whose_target_crosses_the_seam() {
    let (w, h) = (12usize, 10usize);
    let img = block_image(w, h, |x, y| ((x * 7 + y * 13) % 17) as f32 * 3.0 + (x as f32) * 0.5);
    let mut f0 = vec![0.0f32; w * h * 2];
    // pixel (1, 4) points 2 to the right: its target x 1 - 2 = -1 lies across the seam
    let c = (4 * w + 1) * 2;
    f0[c] = 2.0;
    let mut plain = f0.clone();
    block_match(&img, &img, &mut plain, w, h);
    assert_eq!((plain[c], plain[c + 1]), (2.0, 0.0), "the plain step leaves a target outside alone");
    let mut wrapped = f0.clone();
    block_match_with(&img, &img, &mut wrapped, w, h, None, true);
    assert_eq!((wrapped[c], wrapped[c + 1]), (0.0, 0.0), "the wrap step compares it with the neighbour");
    // a target outside the rows is still left alone
    let mut f1 = vec![0.0f32; w * h * 2];
    f1[c + 1] = 6.0;
    let mut g = f1.clone();
    block_match_with(&img, &img, &mut g, w, h, None, true);
    assert_eq!((g[c], g[c + 1]), (0.0, 6.0));
}

#[test]
fn the_wrap_range_check_tests_y_only() {
    let (w, h) = (4usize, 4usize);
    let far_x = [100.0f32, 0.0].repeat(w * h);
    assert!(all_flow_out_of_range(&far_x, w, h));
    assert!(!all_flow_out_of_range_y(&far_x, w, h));
    let far_y = [0.0f32, 100.0].repeat(w * h);
    assert!(all_flow_out_of_range(&far_y, w, h));
    assert!(all_flow_out_of_range_y(&far_y, w, h));
    let mut one = far_y.clone();
    one[(3 * w + 2) * 2 + 1] = -3.0; // y 3 - 3 = 0 is in range
    assert!(!all_flow_out_of_range_y(&one, w, h));
    one[(3 * w + 2) * 2 + 1] = f32::NAN;
    assert!(all_flow_out_of_range_y(&one, w, h));
}

// ---------------------------------------------------------------------------------------
// Mask pyramid and user displacement

#[test]
fn mask_pyramid_binarizes_at_0_95_and_erodes_a_half_plane() {
    // full resolution 8x8: two levels, 8x8 and 4x4
    let s = FlowSettings::new(5, true, 0.7);
    let below = f32::from_bits(0.95f32.to_bits() - 1);
    let vals = [0.95f32, below, 1.0, 0.0, f32::NAN, 2.0, -1.0, 0.96];
    let data: Vec<f32> = (0..64).map(|i| vals[i % 8]).collect();
    let mp = MaskPyramid::build(&Matte { width: 8, height: 8, data }, &s).unwrap();
    assert_eq!(mp.geometry(), &s.geometry(8, 8).unwrap());
    assert_eq!(mp.geometry().sizes, vec![(8, 8), (4, 4)]);
    let want = [1.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
    assert_eq!(mp.level(0), want.repeat(8).as_slice());
    // a half plane, columns 0..=3: blur5 gives the rows [1, 1, 0.9375, 0.6875, 0.3125, 0.0625,
    // 0, 0] (border renormalised, all dyadic so exact); the resample takes x = 0, 7/3, 14/3, 7,
    // values 1, 0.854, 0.146, 0; binarized only column 0 is left
    let data: Vec<f32> = (0..64).map(|i| if i % 8 < 4 { 1.0 } else { 0.0 }).collect();
    let mp = MaskPyramid::build(&Matte { width: 8, height: 8, data: data.clone() }, &s).unwrap();
    assert_eq!(mp.level(0), data.as_slice());
    assert_eq!(mp.level(1), [1.0f32, 0.0, 0.0, 0.0].repeat(4).as_slice());
    // the intermediate values of that step, from the primitives the mask pyramid uses
    let mut t = Plane { width: 8, height: 8, data: data.clone() };
    prep::blur5(&mut t);
    assert_eq!(&t.data[..8], &[1.0, 1.0, 0.9375, 0.6875, 0.3125, 0.0625, 0.0, 0.0]);
    // constant mattes stay constant at every level
    for (v, want) in [(1.0f32, 1.0f32), (0.97, 1.0), (0.0, 0.0), (0.9, 0.0)] {
        let mp = MaskPyramid::build(&Matte { width: 37, height: 29, data: vec![v; 37 * 29] }, &s).unwrap();
        for l in 0..mp.geometry().levels {
            assert!(mp.level(l).iter().all(|&m| m == want), "{v} level {l}");
        }
    }
}

#[test]
fn mask_pyramid_follows_the_reduction_of_the_luma_pyramid() {
    for (q, hiq) in [(1u8, true), (3, true), (4, true), (4, false), (5, false), (5, true)] {
        let s = FlowSettings::new(q, hiq, 0.7);
        let (w, h) = (121usize, 83usize);
        let data: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 - 60.0, (i / w) as f32 - 40.0);
                if x * x + y * y < 30.0 * 30.0 { 1.0 } else { 0.0 }
            })
            .collect();
        let mp = MaskPyramid::build(&Matte { width: w, height: h, data }, &s).unwrap();
        let geo = s.geometry(w, h).unwrap();
        assert_eq!(mp.geometry(), &geo, "q {q} hiq {hiq}");
        let img = image(w, h, Fmt::U8, 0.0, 0.0, 1);
        let py = build_pyramid(&img, &s).unwrap();
        let mut ones = Vec::new();
        for l in 0..geo.levels {
            let (lw, lh) = geo.sizes[l];
            assert_eq!(mp.level(l).len(), lw * lh);
            assert_eq!((py.level(l).width, py.level(l).height), (lw, lh));
            assert!(mp.level(l).iter().all(|&m| m == 0.0 || m == 1.0));
            ones.push(mp.level(l).iter().filter(|&&m| m == 1.0).count());
        }
        // the disc shrinks level by level: every binarize after a blur erodes an edge by about 2
        // pixels of the finer size, so after R reduction steps the radius is about
        // (30 - 2 (2^R - 1)) / 2^R at the working size
        assert!(ones.windows(2).all(|p| p[1] <= p[0]), "q {q} hiq {hiq}: {ones:?}");
        let k = (1u32 << geo.reduction) as f64;
        let r = (30.0 - 2.0 * (k - 1.0)) / k;
        let area = std::f64::consts::PI * r * r;
        assert!((ones[0] as f64) < area * 1.3 && ones[0] as f64 > area * 0.6, "q {q} hiq {hiq}: {} vs {area}", ones[0]);
    }
}

#[test]
fn mask_pyramid_errors() {
    let s = FlowSettings::new(5, true, 0.7);
    let bad = Matte { width: 8, height: 8, data: vec![1.0; 63] };
    assert!(matches!(MaskPyramid::build(&bad, &s), Err(FxError::Mismatch(_))));
    let thin = Matte { width: 1, height: 5, data: vec![1.0; 5] };
    assert!(MaskPyramid::build(&thin, &s).is_err(), "a side of 1 has no motion search");
    let s0 = FlowSettings::new(0, true, 0.7);
    assert!(MaskPyramid::build(&Matte { width: 8, height: 8, data: vec![1.0; 64] }, &s0).is_err());
}

#[test]
fn user_displacement_consumer_hand_values() {
    let w = 3usize;
    let mut flow = vec![9.0f32; 6 * 2];
    let u = UserLevel {
        d: vec![1.0, 2.0, 3.0, -4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.5, -0.25],
        w: vec![0.0, 1.0, 2.0, -0.0, f32::NAN, 4.0],
    };
    super::extras::apply_user(&mut flow, &u, w);
    assert_eq!(&flow[0..2], &[9.0, 9.0], "weight 0 writes nothing");
    assert_eq!(&flow[2..4], &[3.0, -4.0]);
    assert_eq!(&flow[4..6], &[2.5, 3.0]);
    assert_eq!(&flow[6..8], &[9.0, 9.0], "weight -0.0 writes nothing");
    assert!(flow[8].is_nan() && flow[9].is_nan(), "a NaN weight is not 0");
    assert_eq!(&flow[10..12], &[0.125, -0.0625]);
}

#[test]
fn extras_of_another_geometry_are_refused() {
    let s = FlowSettings::new(5, true, 0.7);
    let (a, b) = (image(32, 24, Fmt::U8, 0.0, 0.0, 2), image(32, 24, Fmt::U8, 1.0, 0.0, 2));
    let (pa, pb) = (build_pyramid(&a, &s).unwrap(), build_pyramid(&b, &s).unwrap());
    let other = ones_mask(30, 24, &s);
    let ones = ones_mask(32, 24, &s);
    let e = SolveExtras { masks: Some((&ones, &other)), user: None };
    assert!(matches!(solve_pyramids_with(&pa, &pb, &s, e), Err(FxError::Mismatch(_))));
    let e = SolveExtras { masks: Some((&other, &ones)), user: None };
    assert!(matches!(solve_pyramids_with(&pa, &pb, &s, e), Err(FxError::Mismatch(_))));
    let mut user = inert_user(pa.geometry());
    user.levels.pop();
    let e = SolveExtras { masks: None, user: Some(&user) };
    assert!(matches!(solve_pyramids_with(&pa, &pb, &s, e), Err(FxError::Mismatch(_))));
    let mut user = inert_user(pa.geometry());
    user.levels[1].w.pop();
    let e = SolveExtras { masks: None, user: Some(&user) };
    assert!(matches!(solve_pyramids_with(&pa, &pb, &s, e), Err(FxError::Mismatch(_))));
    let q = image(30, 24, Fmt::U8, 0.0, 0.0, 2);
    let pq = build_pyramid(&q, &s).unwrap();
    assert!(matches!(solve_pyramids_with(&pa, &pq, &s, SolveExtras::default()), Err(FxError::Mismatch(_))));
}

#[test]
fn one_point_sets_its_pixel_exactly_and_two_points_average() {
    let (w, h) = (48usize, 36usize);
    let (a, b) = (image(w, h, Fmt::F32, 0.0, 0.0, 7), image(w, h, Fmt::F32, 1.5, 0.5, 7));
    for sens in [1.0f32, 0.7, 2.0] {
        let s = FlowSettings::new(5, true, sens);
        let (pa, pb) = (build_pyramid(&a, &s).unwrap(), build_pyramid(&b, &s).unwrap());
        let geo = pa.geometry().clone();
        let mut user = UserDisplacement {
            levels: geo.sizes.iter().map(|&(lw, lh)| UserLevel { d: vec![0.0; lw * lh * 2], w: vec![0.0; lw * lh] }).collect(),
        };
        // one point at (20, 11) and two on (5, 30) at level 0
        let (p, q) = (11 * w + 20, 30 * w + 5);
        let lv = &mut user.levels[0];
        lv.d[p * 2] = 0.8125;
        lv.d[p * 2 + 1] = -2.375;
        lv.w[p] = 1.0;
        lv.d[q * 2] = 3.0 + 0.5;
        lv.d[q * 2 + 1] = -1.0 + 1.25;
        lv.w[q] = 2.0;
        let e = SolveExtras { masks: None, user: Some(&user) };
        let f = solve_pyramids_with(&pa, &pb, &s, e).unwrap();
        // the returned flow is the negated internal flow; the start level is not level 0 here, so
        // its damping does not reach the level 0 values
        assert!(start_level(sens, geo.levels).0 > 0);
        assert_eq!(f.at(20, 11), [-0.8125, 2.375], "sens {sens}");
        assert_eq!(f.at(5, 30), [-(3.5f32 / 2.0), -(0.25f32 / 2.0)], "sens {sens}");
        // elsewhere the flow is a solve, not the plain one
        let plain = solve_pyramids(&pa, &pb, &s).unwrap();
        assert_ne!(bits(&f.data), bits(&plain.data));
    }
}

#[test]
fn the_extended_solve_is_thread_count_independent() {
    let (w, h) = (64usize, 48usize);
    let (a, b) = (image(w, h, Fmt::U16, 0.0, 0.0, 9), image(w, h, Fmt::U16, -2.0, 1.25, 9));
    let disc = |cx: f32| Matte {
        width: w,
        height: h,
        data: (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 - cx, (i / w) as f32 - 24.0);
                if x * x + y * y < 18.0 * 18.0 { 1.0 } else { 0.0 }
            })
            .collect(),
    };
    for wrap in [false, true] {
        let s = FlowSettings { wrap, ..FlowSettings::new(5, true, 0.8) };
        let (pa, pb) = (build_pyramid(&a, &s).unwrap(), build_pyramid(&b, &s).unwrap());
        let (mx, my) = (MaskPyramid::build(&disc(30.0), &s).unwrap(), MaskPyramid::build(&disc(28.0), &s).unwrap());
        let mut user = inert_user(pa.geometry());
        user.levels[0].w[10 * w + 12] = 1.0;
        user.levels[1].w[5 * (w / 2) + 6] = 3.0;
        let e = SolveExtras { masks: Some((&mx, &my)), user: Some(&user) };
        let f1 = pool(1, || solve_pyramids_with(&pa, &pb, &s, e).unwrap());
        let f4 = pool(4, || solve_pyramids_with(&pa, &pb, &s, e).unwrap());
        let f16 = pool(16, || solve_pyramids_with(&pa, &pb, &s, e).unwrap());
        assert_eq!(bits(&f1.data), bits(&f4.data), "wrap {wrap}");
        assert_eq!(bits(&f1.data), bits(&f16.data), "wrap {wrap}");
        let m1 = pool(1, || MaskPyramid::build(&disc(30.0), &s).unwrap());
        assert_eq!(pool(16, || MaskPyramid::build(&disc(30.0), &s).unwrap()), m1);
    }
}

// ---------------------------------------------------------------------------------------
// Numpy float32 model fixtures (`tests/fixtures/gen_mask.py`, format there)

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.b[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }

    fn f32s(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| f32::from_bits(self.u32())).collect()
    }
}

#[test]
fn model_fixtures() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mask_cases.hex");
    let text = std::fs::read_to_string(&path).expect("mask_cases.hex");
    let b = crate::fixture::hex(&text);
    assert_eq!(&b[0..4], b"MSK1");
    let mut r = Reader { b: &b, pos: 4 };
    let n = r.u32();
    let mut kinds = [0usize; 5];
    for case in 0..n {
        let kind = r.u32();
        let (w, h) = (r.u32() as usize, r.u32() as usize);
        let tag = format!("case {case} kind {kind} {w}x{h}");
        match kind {
            1 => {
                let (q, hiq) = (r.u32() as u8, r.u32() != 0);
                let data = r.f32s(w * h);
                let s = FlowSettings::new(q, hiq, 0.7);
                let mp = MaskPyramid::build(&Matte { width: w, height: h, data }, &s).unwrap();
                let levels = r.u32() as usize;
                assert_eq!(mp.geometry().levels, levels, "{tag} q {q} hiq {hiq}");
                for l in 0..levels {
                    let (lw, lh) = (r.u32() as usize, r.u32() as usize);
                    assert_eq!(mp.geometry().sizes[l], (lw, lh), "{tag} level {l}");
                    let want = r.f32s(lw * lh);
                    assert_eq!(bits(mp.level(l)), bits(&want), "{tag} q {q} hiq {hiq} level {l}");
                }
            }
            2 => {
                let (wrap, has_mask) = (r.u32() != 0, r.u32() != 0);
                let img = r.f32s(w * h);
                let mask = if has_mask { Some(r.f32s(w * h)) } else { None };
                let want = r.f32s(w * h * 2);
                let mut g = if wrap { calc_grad_wrap(&img, w, h) } else { calc_grad(&img, w, h) };
                if let Some(m) = &mask {
                    mask_gradient(&mut g, m, w, h, wrap);
                }
                assert_eq!(bits(&g), bits(&want), "{tag} wrap {wrap} mask {has_mask}");
            }
            3 => {
                let (wrap, has_mask) = (r.u32() != 0, r.u32() != 0);
                let (i1, i0) = (r.f32s(w * h), r.f32s(w * h));
                let (grad, flow) = (r.f32s(w * h * 2), r.f32s(w * h * 2));
                let mask = if has_mask { Some(r.f32s(w * h)) } else { None };
                let want = r.f32s(w * h * 2);
                let got = calc_b_with(&i1, &i0, &grad, &flow, w, h, mask.as_deref(), wrap);
                assert_eq!(bits(&got), bits(&want), "{tag} wrap {wrap} mask {has_mask}");
                if !wrap && !has_mask {
                    assert_eq!(bits(&calc_b(&i1, &i0, &grad, &flow, w, h)), bits(&want), "{tag} plain");
                }
            }
            4 => {
                let (wrap, has_gate) = (r.u32() != 0, r.u32() != 0);
                let (a, bb) = (r.f32s(w * h), r.f32s(w * h));
                let mut flow = r.f32s(w * h * 2);
                let gate = if has_gate { Some(r.f32s(w * h)) } else { None };
                let want = r.f32s(w * h * 2);
                if !wrap && !has_gate {
                    let mut plain = flow.clone();
                    block_match(&a, &bb, &mut plain, w, h);
                    assert_eq!(bits(&plain), bits(&want), "{tag} plain");
                }
                block_match_with(&a, &bb, &mut flow, w, h, gate.as_deref(), wrap);
                assert_eq!(bits(&flow), bits(&want), "{tag} wrap {wrap} gate {has_gate}");
            }
            k => panic!("{tag}: unknown kind {k}"),
        }
        kinds[kind as usize] += 1;
    }
    assert_eq!(r.pos, b.len(), "trailing bytes in the fixture");
    assert!(kinds[1..].iter().all(|&k| k >= 20), "{kinds:?}");
}

// ---------------------------------------------------------------------------------------
// The wrap solve on periodic pyramids

/// A texture of period `w` in x: whole cycles across the width, only the components of at most
/// `max_k` cycles (so a coarse level holds no alias), any frequency in y.
fn periodic_texture(x: f64, y: f64, w: f64, max_k: f64) -> f64 {
    let t = std::f64::consts::TAU;
    let mut v = 128.0;
    for (i, &(kx, fy, amp)) in [(1.0, 0.013, 30.0), (2.0, -0.021, 25.0), (3.0, 0.031, 22.0), (5.0, 0.047, 18.0), (8.0, -0.06, 14.0), (13.0, 0.08, 10.0), (21.0, -0.11, 7.0)]
        .iter()
        .enumerate()
    {
        if kx <= max_k {
            v += amp * (t * (kx * x / w + fy * y) + i as f64 * 1.3).sin();
        }
    }
    v
}

/// A pyramid whose every level is periodic in x: level `l` samples the texture shifted by
/// `shift` (full resolution pixels) at `x * W / w_l` (period preserving) and corner aligned rows,
/// band limited to periods of at least 6 pixels of the level. It isolates the wrap kernels of the
/// solver from the pyramid filters (the engine's own luma pyramid is periodic too under the wrap
/// flag, with the circular blur and the period keeping resample; `flow::tests_pyramid` tests it).
fn periodic_pyramid(geo: &FlowGeometry, shift: f64) -> Pyramid {
    let (w0, h0) = (geo.width as f64, geo.height as f64);
    let levels = geo
        .sizes
        .iter()
        .map(|&(w, h)| {
            let sx = w0 / w as f64;
            let sy = (h0 - 1.0) / (h as f64 - 1.0);
            let max_k = w as f64 / 6.0;
            let data = (0..w * h).map(|i| periodic_texture((i % w) as f64 * sx - shift, (i / w) as f64 * sy, w0, max_k) as f32).collect();
            Plane { width: w, height: h, data }
        })
        .collect();
    Pyramid { geometry: geo.clone(), levels }
}

#[test]
fn wrap_recovers_a_circular_shift_on_both_sides_of_the_seam_where_the_plain_solve_fails() {
    let (w, h) = (128usize, 64usize);
    for (shift, tol) in [(3.0f64, 0.005f64), (-5.0, 0.005), (9.5, 0.06)] {
        let s = FlowSettings::new(5, true, 1.0);
        let geo = s.geometry(w, h).unwrap();
        let (px, py) = (periodic_pyramid(&geo, 0.0), periodic_pyramid(&geo, shift));
        let fw = solve_pyramids(&px, &py, &FlowSettings { wrap: true, ..s }).unwrap();
        let fp = solve_pyramids(&px, &py, &s).unwrap();
        // mean distance to (shift, 0) of a column, away from the top and bottom rows
        let col = |f: &FlowField, c: usize| {
            let mut e = 0.0f64;
            for y in 8..h - 8 {
                let v = f.at(c, y);
                e += (((v[0] as f64) - shift).powi(2) + (v[1] as f64).powi(2)).sqrt();
            }
            e / (h - 16) as f64
        };
        let worst = |f: &FlowField, cols: &mut dyn Iterator<Item = usize>| cols.map(|c| col(f, c)).fold(0.0f64, f64::max);
        let (wl, wr) = (worst(&fw, &mut (0..8)), worst(&fw, &mut (w - 8..w)));
        let wm = worst(&fw, &mut (24..w - 24));
        assert!(wl < tol && wr < tol && wm < tol, "shift {shift}: wrap left {wl} middle {wm} right {wr}");
        // the plain solve fails on the side whose targets leave the image
        let bad = if shift > 0.0 { worst(&fp, &mut (w - 8..w)) } else { worst(&fp, &mut (0..8)) };
        assert!(bad > 8.0 * wl.max(wr) && bad > 0.04, "shift {shift}: plain {bad} against wrap {wl} {wr}");
    }
}

// ---------------------------------------------------------------------------------------
// Masked solves that never update

/// The texture of the create-vectors job tests: a sum of plane waves in 0..1, channel `c`.
fn wave_texture(x: f32, y: f32, c: usize) -> f32 {
    let waves = [(17.0f32, 0.35f32, 0.3f32), (11.0, 1.9, 0.25), (23.0, 2.7, 0.25)];
    let mut s = 0.0f32;
    for (i, (l, th, a)) in waves.iter().enumerate() {
        let ph = 1.3 * i as f32 + 2.1 * c as f32;
        s += a * ((x * th.cos() + y * th.sin()) * std::f32::consts::TAU / l + ph).sin();
    }
    (0.5 + 0.5 * s).clamp(0.0, 1.0)
}

/// Frame `k` of [`wave_texture`] moving by `(2, 1)` px a frame, opaque 16 bpc.
fn wave_frame(w: usize, h: usize, k: f32) -> Image {
    let mut im = Image::new(w, h, Fmt::U16);
    if let Samples::U16(s) = &mut im.samples {
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let v = |c: usize| (wave_texture(x as f32 - 2.0 * k, y as f32 - 1.0 * k, c) * 65535.0 + 0.5) as u16;
                s[i..i + 4].copy_from_slice(&[65535, v(0), v(1), v(2)]);
            }
        }
    }
    im
}

/// A disc matte of radius `r` at `(cx, cy)` with a one pixel soft edge.
fn soft_disc(w: usize, h: usize, cx: f32, cy: f32, r: f32) -> Matte {
    let data = (0..w * h)
        .map(|i| {
            let d = (((i % w) as f32 - cx).powi(2) + ((i / w) as f32 - cy).powi(2)).sqrt();
            (r - d + 0.5).clamp(0.0, 1.0)
        })
        .collect();
    Matte { width: w, height: h, data }
}

/// The largest `|t|` of `new_disp`'s update test (`t = b*b - dd*a` of the blurred tensor of the
/// masked gradient) at every level the solve visits, start level first. The tensor does not
/// depend on the flow, so a solve updates a pixel of level `l` only where this exceeds 40.
fn masked_update_peaks(x: &Pyramid, m: &MaskPyramid, s: &FlowSettings) -> Vec<f32> {
    let geo = &x.geometry;
    let (start, _) = start_level(s.sensitivity, geo.levels);
    (0..=start)
        .rev()
        .map(|l| {
            let (w, h) = geo.sizes[l];
            let mut grad = calc_grad(&x.levels[l].data, w, h);
            mask_gradient(&mut grad, m.level(l), w, h, false);
            let mut a = calc_a(&grad, w, h);
            conv::blur_gaussian(&mut a, w, h, 3, 7.0);
            a.chunks(3).fold(0.0f32, |p, t| p.max((t[1] * t[1] - t[2] * t[0]).abs()))
        })
        .collect()
}

#[test]
fn a_disc_whose_masked_tensor_never_passes_the_update_test_gives_the_exact_zero_flow() {
    // the reported case: a 14 px disc at (22, 20) of a 64x48 frame, Full res, Sensitivity 90.
    // Gradient masking leaves the tensor zero outside the mask, whose levels erode by the
    // blur5 and binarize of every step (empty from level 3 on), and the radius 7 blur dilutes
    // it: no pixel of any level passes `|t| > 40`, so the flow never leaves zero. It is not a
    // reset by the range check (a zero flow is in range everywhere).
    let (w, h) = (64usize, 48usize);
    let (x, y) = (wave_frame(w, h, 1.0), wave_frame(w, h, 0.0));
    let s = FlowSettings::new(5, true, 0.9);
    let (px, py) = (build_pyramid(&x, &s).unwrap(), build_pyramid(&y, &s).unwrap());
    let geo = px.geometry().clone();
    assert_eq!((geo.levels, start_level(s.sensitivity, geo.levels).0), (5, 4));
    let m = MaskPyramid::build(&soft_disc(w, h, 22.0, 20.0, 14.0), &s).unwrap();
    assert!(m.level(3).iter().chain(m.level(4)).all(|&v| v == 0.0), "the coarse masks are empty");
    let peaks = masked_update_peaks(&px, &m, &s);
    eprintln!("masked update peaks per level, start first: {peaks:?}");
    assert!(peaks.iter().all(|&t| t <= 40.0), "{peaks:?}");
    assert!(peaks[3] > 30.0, "level 1 comes closest: {peaks:?}");
    let f = solve_pyramids_with(&px, &py, &s, SolveExtras { masks: Some((&m, &m)), user: None }).unwrap();
    assert!(f.data.iter().all(|v| v.to_bits() == (-0.0f32).to_bits()), "the zero start, negated");
    // the unmasked solve of the pair tracks the motion (-2, -1) inside the disc, and through
    // `solve_pyramids_with` without extras it is the plain solve bit for bit
    let plain = solve_pyramids(&px, &py, &s).unwrap();
    assert_eq!(bits(&solve_pyramids_with(&px, &py, &s, SolveExtras::default()).unwrap().data), bits(&plain.data));
    assert_eq!(bits(&reference_solve(&px, &py, &s).data), bits(&plain.data));
    let (mut e, mut n) = (0.0f64, 0usize);
    for yy in 0..h {
        for xx in 0..w {
            if m.level(0)[yy * w + xx] == 1.0 {
                let v = plain.at(xx, yy);
                e += ((v[0] as f64 + 2.0).powi(2) + (v[1] as f64 + 1.0).powi(2)).sqrt();
                n += 1;
            }
        }
    }
    eprintln!("unmasked error inside the disc: {} over {n} px", e / n as f64);
    assert!(n > 500 && e / (n as f64) < 0.02, "unmasked mean error {}", e / n as f64);
}

#[test]
fn masked_solves_are_zero_exactly_where_no_level_passes_the_update_test() {
    // a sweep of 14 px discs over two frame sizes: the masked solve is the exact zero flow if and
    // only if the masked tensor passes `|t| > 40` nowhere on the levels the solve visits; both
    // outcomes occur
    let mut seen = (0usize, 0usize);
    for (w, h) in [(64usize, 48usize), (96, 72)] {
        let (x, y) = (wave_frame(w, h, 1.0), wave_frame(w, h, 0.0));
        let s = FlowSettings::new(5, true, 0.9);
        let (px, py) = (build_pyramid(&x, &s).unwrap(), build_pyramid(&y, &s).unwrap());
        for cy in (14..h - 13).step_by(4) {
            for cx in (14..w - 13).step_by(4) {
                let m = MaskPyramid::build(&soft_disc(w, h, cx as f32, cy as f32, 14.0), &s).unwrap();
                let f = solve_pyramids_with(&px, &py, &s, SolveExtras { masks: Some((&m, &m)), user: None }).unwrap();
                let zero = f.data.iter().all(|v| *v == 0.0);
                let passes = masked_update_peaks(&px, &m, &s).iter().any(|&t| t > 40.0);
                assert_eq!(zero, !passes, "{w}x{h} disc at ({cx}, {cy})");
                if zero {
                    seen.0 += 1;
                } else {
                    seen.1 += 1;
                }
            }
        }
    }
    eprintln!("zero flows {}, moving flows {}", seen.0, seen.1);
    assert!(seen.0 >= 10 && seen.1 >= 40, "{seen:?}");
}

// ---------------------------------------------------------------------------------------
// The order of the driver's hooks (G6), on pyramids of zero images: a zero band pass has
// a zero gradient, so the tensor is zero, no update passes `|t| > 40`, every B vector is zero and
// every block cost ties (a sweep replaces a vector only for a strictly lower cost). The flow of
// such a solve is the user displacement and the driver's own steps on it: the Gaussian blurs of
// the flow, the range check, the damping, the upsample and the final negation.

/// A pyramid of zero images of the geometry `geo`.
fn zero_pyramid(geo: &FlowGeometry) -> Pyramid {
    let levels = geo.sizes.iter().map(|&(w, h)| Plane { width: w, height: h, data: vec![0.0; w * h] }).collect();
    Pyramid { geometry: geo.clone(), levels }
}

/// A user displacement of `geo` that sets nothing (all weights 0).
fn empty_user(geo: &FlowGeometry) -> UserDisplacement {
    UserDisplacement { levels: geo.sizes.iter().map(|&(w, h)| UserLevel { d: vec![0.0; w * h * 2], w: vec![0.0; w * h] }).collect() }
}

#[test]
fn the_user_displacement_enters_each_level_before_its_iterations_spread_it() {
    // `flow = d / w` at the START of every level, before the three iterations, and again at its end. With one
    // point at level 0 only, the flow blurs (radius 5) of the iterations spread the set value to the neighbours of
    // the pixel, and the end of the level writes the pixel again; without the start step the neighbours stay zero
    let s = FlowSettings::new(5, true, 1.0);
    let (w, h) = (40usize, 32usize);
    let geo = s.geometry(w, h).unwrap();
    assert!(geo.levels >= 3 && start_level(1.0, geo.levels) == (geo.levels - 1, 1.0));
    let z = zero_pyramid(&geo);
    let mut user = empty_user(&geo);
    let p = 13 * w + 17;
    user.levels[0].d[p * 2] = 2.5;
    user.levels[0].d[p * 2 + 1] = -1.25;
    user.levels[0].w[p] = 1.0;
    let solve4 = || pool(4, || solve_pyramids_with(&z, &z, &s, SolveExtras { masks: None, user: Some(&user) }).unwrap());
    let got = pool(1, || solve_pyramids_with(&z, &z, &s, SolveExtras { masks: None, user: Some(&user) }).unwrap());
    assert_eq!(bits(&got.data), bits(&solve4().data), "1 and 4 threads");
    // the coarse levels stay zero; level 0: set, three blurs of the flow, set again, negate
    let mut want = vec![0.0f32; w * h * 2];
    want[p * 2] = 2.5;
    want[p * 2 + 1] = -1.25;
    for _ in 0..3 {
        conv::blur_gaussian(&mut want, w, h, 2, 5.0);
    }
    want[p * 2] = 2.5;
    want[p * 2 + 1] = -1.25;
    let want: Vec<f32> = want.iter().map(|v| -1.0f32 * v).collect();
    assert_eq!(bits(&got.data), bits(&want));
    assert_eq!(got.at(17, 13), [-2.5, 1.25]);
    // the neighbours carry the spread value: the same direction and less of it
    for (x, y) in [(16usize, 13usize), (18, 13), (17, 12), (17, 14), (20, 15)] {
        let v = got.at(x, y);
        assert!(v[0] < 0.0 && v[0] > -2.5 && v[1] > 0.0 && v[1] < 1.25, "({x}, {y}): {v:?}");
    }
    // the same through the wrap driver: the circular blurs, so a point on column 0 reaches the last column
    let sw = FlowSettings { wrap: true, ..s };
    let mut user = empty_user(&geo);
    let q = 13 * w;
    user.levels[0].d[q * 2] = 2.5;
    user.levels[0].w[q] = 1.0;
    let got = solve_pyramids_with(&z, &z, &sw, SolveExtras { masks: None, user: Some(&user) }).unwrap();
    let mut want = vec![0.0f32; w * h * 2];
    want[q * 2] = 2.5;
    for _ in 0..3 {
        conv::blur_gaussian_wrapx(&mut want, w, h, 2, 5.0);
    }
    want[q * 2] = 2.5;
    let want: Vec<f32> = want.iter().map(|v| -1.0f32 * v).collect();
    assert_eq!(bits(&got.data), bits(&want));
    assert!(got.at(w - 1, 13)[0] < 0.0, "the last column holds the spread of the point on column 0");
}

#[test]
fn the_start_level_damping_scales_the_pixels_the_user_set() {
    // `flow = d / w` at the end of a level comes BEFORE the range check and the start level's damping (the plain driver's
    // order), so a point set on the start level is damped like every other pixel. A sensitivity whose start level
    // is 0 makes the damped level the output: the point's pixel is -(damp * d / w), not -(d / w)
    let (w, h) = (40usize, 32usize);
    let levels = FlowSettings::new(5, true, 1.0).geometry(w, h).unwrap().levels;
    let sens = (2.0f64.powf((0.5 - levels as f64) / 2.0)) as f32;
    let (start, damp) = start_level(sens, levels);
    assert_eq!(start, 0, "sensitivity {sens} starts on level 0");
    assert!(damp > 0.4 && damp < 0.6 && (damp as f64 - 1.0).abs() >= 1e-6, "damping {damp}");
    for wrap in [false, true] {
        let s = FlowSettings { wrap, ..FlowSettings::new(5, true, sens) };
        let geo = s.geometry(w, h).unwrap();
        let z = zero_pyramid(&geo);
        let mut user = empty_user(&geo);
        let (p, q) = (13 * w + 17, 5 * w + 30);
        let lv = &mut user.levels[0];
        lv.d[p * 2] = 2.5;
        lv.d[p * 2 + 1] = -1.25;
        lv.w[p] = 1.0;
        lv.d[q * 2] = 3.0;
        lv.d[q * 2 + 1] = 1.0;
        lv.w[q] = 2.0;
        let e = SolveExtras { masks: None, user: Some(&user) };
        let got = pool(1, || solve_pyramids_with(&z, &z, &s, e).unwrap());
        assert_eq!(bits(&got.data), bits(&pool(4, || solve_pyramids_with(&z, &z, &s, e).unwrap()).data), "wrap {wrap}: 1 and 4 threads");
        assert_eq!(got.at(17, 13), [-1.0f32 * (damp * 2.5), -1.0f32 * (damp * -1.25)], "wrap {wrap}");
        assert_eq!(got.at(30, 5), [-1.0f32 * (damp * 1.5), -1.0f32 * (damp * 0.5)], "wrap {wrap}");
        // the whole field: set, blur three times, set again, damp, negate
        let set = |f: &mut Vec<f32>| {
            f[p * 2] = 2.5;
            f[p * 2 + 1] = -1.25;
            f[q * 2] = 3.0f32 / 2.0;
            f[q * 2 + 1] = 1.0f32 / 2.0;
        };
        let mut want = vec![0.0f32; w * h * 2];
        set(&mut want);
        for _ in 0..3 {
            if wrap {
                conv::blur_gaussian_wrapx(&mut want, w, h, 2, 5.0);
            } else {
                conv::blur_gaussian(&mut want, w, h, 2, 5.0);
            }
        }
        set(&mut want);
        let want: Vec<f32> = want.iter().map(|v| -1.0f32 * (damp * v)).collect();
        assert_eq!(bits(&got.data), bits(&want), "wrap {wrap}");
    }
}

#[test]
fn a_wrap_solve_keeps_a_flow_of_one_whole_width_that_the_plain_range_check_resets() {
    // G6: the range check of a wrap solve tests y only. A user displacement of one whole width at every pixel of
    // the start level points every pixel outside the frame in x (the plain check resets such a level to zero) but
    // onto its own column across the seam: the wrap solve keeps it, the upsample doubles it level by level and the
    // flow of the output is minus the frame width everywhere (the circular blurs of a constant field change it only
    // by their rounding)
    let (w, h) = (64usize, 32usize);
    let s = FlowSettings { wrap: true, ..FlowSettings::new(5, true, 1.0) };
    let geo = s.geometry(w, h).unwrap();
    let start = geo.levels - 1;
    assert!(start >= 2 && start_level(1.0, geo.levels) == (start, 1.0));
    assert!(geo.sizes.iter().all(|&(lw, _)| w % lw == 0), "every level width divides the frame width: {:?}", geo.sizes);
    let z = zero_pyramid(&geo);
    let mut user = empty_user(&geo);
    let (sw, sh) = geo.sizes[start];
    let top = &mut user.levels[start];
    for i in 0..sw * sh {
        top.d[i * 2] = sw as f32;
        top.w[i] = 1.0;
    }
    let e = SolveExtras { masks: None, user: Some(&user) };
    let got = pool(1, || solve_pyramids_with(&z, &z, &s, e).unwrap());
    assert_eq!(bits(&got.data), bits(&pool(4, || solve_pyramids_with(&z, &z, &s, e).unwrap()).data), "1 and 4 threads");
    for y in 0..h {
        for x in 0..w {
            let v = got.at(x, y);
            assert!((v[0] + w as f32).abs() < 1e-3 && v[1] == 0.0, "({x}, {y}): {v:?}");
        }
    }
    // the plain solve of the same input: every pixel's target leaves the frame, the level is reset and stays zero
    let plain = solve_pyramids_with(&z, &z, &FlowSettings { wrap: false, ..s }, e).unwrap();
    assert!(plain.data.iter().all(|&v| v == 0.0), "the plain range check resets the start level");
}

#[test]
fn the_b_step_gate_admits_a_warped_mask_of_exactly_0_999_and_not_the_float_below() {
    // `B` needs the warped mask of `Y` at least 0.999 (f32), so 0.999f32 itself passes and its predecessor
    // fails. Two ways to sample it: a binary mask between columns 0 (1.0) and 1 (0.0) at the fraction
    // `f = 1 - 0.999f32` (exact, a multiple of 2^-24), whose sample `(0 - 1) * f + 1` is 0.999f32 exactly, and
    // mask values of 0.999f32 and its two neighbours read at whole targets
    let at = 0.999f32;
    let below = f32::from_bits(at.to_bits() - 1);
    let above = f32::from_bits(at.to_bits() + 1);
    let f = 1.0f32 - at;
    // 2^-24, the spacing of the floats in [0.5, 1)
    let ulp = 1.0f32 / 16_777_216.0;
    assert_eq!(at.to_bits() - below.to_bits(), 1);
    assert_eq!(((0.0f32 - 1.0) * f + 1.0, (0.0f32 - 1.0) * (f + ulp) + 1.0, (0.0f32 - 1.0) * (f - ulp) + 1.0), (at, below, above));
    let (w, h) = (4usize, 3usize);
    let i1 = vec![0.0f32; w * h];
    let i0 = vec![10.0f32; w * h];
    let grad = [1.0f32, -1.0].repeat(w * h);
    let mut mask = vec![0.0f32; w * h];
    for y in 0..h {
        mask[y * w] = 1.0;
        mask[y * w + 2] = [at, below, above][y];
    }
    // column 0 of rows 0, 1, 2 targets x = f, f + 2^-24, f - 2^-24 (the flow is minus the displacement); the other
    // columns have the zero flow and read their own mask value
    let mut flow = vec![0.0f32; w * h * 2];
    for (y, fx) in [f, f + ulp, f - ulp].into_iter().enumerate() {
        flow[y * w * 2] = -fx;
    }
    for (wrap, threads) in [(false, 1), (true, 1), (false, 4), (true, 4)] {
        let b = pool(threads, || calc_b_with(&i1, &i0, &grad, &flow, w, h, Some(&mask), wrap));
        let px =|x: usize, y: usize| (b[(y * w + x) * 2], b[(y * w + x) * 2 + 1]);
        let pass = (10.0f32, -10.0f32);
        assert_eq!(px(0, 0), pass, "wrap {wrap}: a binary mask sampled to exactly 0.999 passes");
        assert_eq!(px(0, 1), (0.0, 0.0), "wrap {wrap}: the float below 0.999 fails");
        assert_eq!(px(0, 2), pass, "wrap {wrap}: the float above 0.999 passes");
        assert_eq!(px(2, 0), pass, "wrap {wrap}: a mask value of 0.999 passes");
        assert_eq!(px(2, 1), (0.0, 0.0), "wrap {wrap}: a mask value just below 0.999 fails");
        assert_eq!(px(2, 2), pass, "wrap {wrap}: a mask value just above 0.999 passes");
        for y in 0..h {
            assert_eq!(px(1, y), (0.0, 0.0), "wrap {wrap}: mask 0");
            assert_eq!(px(3, y), (0.0, 0.0), "wrap {wrap}: mask 0");
        }
    }
}
