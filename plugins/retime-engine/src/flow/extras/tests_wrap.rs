//! Tests of the periodic wrap solve (rule G6, U, ours): with `FlowSettings::wrap`
//! rotating both inputs by `k` columns rotates the solved flow by `k` columns, bit for bit.
//!
//! - every level's step ([`level_step`]: gradient, masking, tensor, B vector step, the circular
//!   Gaussians, the update, the ring sweeps of the block matching, the user displacement)
//!   commutes with every rotation `k` at every level of real pyramids, even and odd widths,
//!   qualities draft, High, Best and full resolution, without and with masks and points;
//! - the period keeping flow upsample rotates with the coarse level (`2 k` for `k`);
//! - the whole driver: a single level pyramid of an odd and of an even width for every `k`, and
//!   the periodic pyramids of `flow::build_pyramid` for rotations of both frames by multiples of
//!   `2^(R + L - 1)` columns, through every entry point;
//! - away from the seam the wrap gradient is the plain one bit for bit;
//! - with wrap off the layered driver is the one before the hooks were split into
//!   [`level_step`], and 1, 4 and 16 threads give identical bytes.

use super::*;
use crate::flow::{build_pyramid, solve, solve_pair, solve_pyramids, Pyramid};
use crate::image::{Fmt, Image, Samples};
use crate::prep::Plane;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn pool<T: Send>(n: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap().install(f)
}

/// A small deterministic generator in 0..1.
fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `v` (rows of `w` pixels of `c` interleaved values) rotated right by `s` columns: column `x`
/// of the result is column `x - s` modulo `w` of `v`.
fn rot<T: Copy>(v: &[T], w: usize, c: usize, s: usize) -> Vec<T> {
    let h = v.len() / (w * c);
    let mut out = v.to_vec();
    for y in 0..h {
        for x in 0..w {
            let from = (x + w - s % w) % w;
            for k in 0..c {
                out[(y * w + x) * c + k] = v[(y * w + from) * c + k];
            }
        }
    }
    out
}

fn rot_image(im: &Image, s: usize) -> Image {
    let mut out = im.clone();
    let w = im.width;
    match (&im.samples, &mut out.samples) {
        (Samples::U8(a), Samples::U8(b)) => *b = rot(a, w, 4, s),
        (Samples::U16(a), Samples::U16(b)) => *b = rot(a, w, 4, s),
        (Samples::F32(a), Samples::F32(b)) => *b = rot(a, w, 4, s),
        _ => unreachable!(),
    }
    out
}

/// A random-ish gray frame: a few plane waves and value noise, `shift` moves the waves right.
fn frame(w: usize, h: usize, fmt: Fmt, shift: (f64, f64), seed: u64) -> Image {
    let mut r = rng(seed);
    let waves: Vec<(f64, f64, f64, f64)> = (0..5).map(|_| (r() * 0.5 - 0.25, r() * 0.5 - 0.25, r() * 6.28, 12.0 + r() * 30.0)).collect();
    let noise: Vec<f64> = (0..w * h).map(|_| r() * 18.0 - 9.0).collect();
    let mut im = Image::new(w, h, fmt);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f64 - shift.0, y as f64 - shift.1);
            let mut v = 128.0 + noise[y * w + x];
            for &(kx, ky, ph, a) in &waves {
                v += a * (kx * fx + ky * fy + ph).sin();
            }
            let v = v.clamp(0.0, 255.0);
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

/// A disc matte on the cylinder (the distance in x is taken across the seam).
fn disc(w: usize, h: usize, c: (f64, f64), r: f64) -> Matte {
    let wf = w as f64;
    let data = (0..w * h)
        .map(|i| {
            let mut dx = (i % w) as f64 - c.0;
            while dx < -wf / 2.0 {
                dx += wf;
            }
            while dx >= wf / 2.0 {
                dx -= wf;
            }
            let dy = (i / w) as f64 - c.1;
            ((r - (dx * dx + dy * dy).sqrt() + 0.5).clamp(0.0, 1.0)) as f32
        })
        .collect();
    Matte { width: w, height: h, data }
}

/// A few user points per level, two of them on the seam columns.
fn points(geo: &FlowGeometry) -> UserDisplacement {
    let levels = geo
        .sizes
        .iter()
        .map(|&(w, h)| {
            let mut lv = UserLevel { d: vec![0.0; w * h * 2], w: vec![0.0; w * h] };
            for (px, py, dx, dy, k) in [(0usize, h / 2, 1.5f32, -0.25f32, 1.0f32), (w - 1, h / 3, -0.75, 0.5, 2.0), (w / 3, 1, 0.25, 0.125, 1.0)] {
                let i = py * w + px;
                lv.d[i * 2] = dx * k;
                lv.d[i * 2 + 1] = dy * k;
                lv.w[i] = k;
            }
            lv
        })
        .collect();
    UserDisplacement { levels }
}

fn rot_level(u: &UserLevel, w: usize, s: usize) -> UserLevel {
    UserLevel { d: rot(&u.d, w, 2, s), w: rot(&u.w, w, 1, s) }
}

fn wrap(q: u8, hiq: bool, sensitivity: f32) -> FlowSettings {
    FlowSettings { wrap: true, ..FlowSettings::new(q, hiq, sensitivity) }
}

/// Draft (Best at draft quality), High, Best and full resolution.
const QUALITIES: [(u8, bool); 4] = [(4, false), (3, true), (4, true), (5, true)];

/// The rotations tried on a row of `w`: 1, 2, 3, half and all but one column (those below `w`).
fn ks(w: usize) -> Vec<usize> {
    let mut v: Vec<usize> = [1, 2, 3, w / 2, w - 1].into_iter().filter(|&k| k >= 1 && k < w).collect();
    v.sort_unstable();
    v.dedup();
    v
}

// ---------------------------------------------------------------------------------------
// One level

#[test]
#[ignore = "the G6 wrap rule (U) this test encodes is superseded by the 360 rules read from TwixtorPro_64 (parity/lab/re/op-residual.md, finding 6)"]
fn every_level_step_commutes_with_every_rotation() {
    let mut n = 0;
    for (fw, fh) in [(64usize, 40usize), (75, 37)] {
        let (x, y) = (frame(fw, fh, Fmt::F32, (0.0, 0.0), 3), frame(fw, fh, Fmt::F32, (1.7, -0.6), 3));
        let (mx, my) = (disc(fw, fh, (3.0, fh as f64 * 0.5), fh as f64 * 0.35), disc(fw, fh, (4.5, fh as f64 * 0.5 - 0.5), fh as f64 * 0.35));
        for (q, hiq) in QUALITIES {
            let s = wrap(q, hiq, 1.0);
            let (px, py) = (build_pyramid(&x, &s).unwrap(), build_pyramid(&y, &s).unwrap());
            let (kx, ky) = (MaskPyramid::build(&mx, &s).unwrap(), MaskPyramid::build(&my, &s).unwrap());
            let geo = px.geometry().clone();
            let user = points(&geo);
            for l in 0..geo.levels {
                let (w, h) = geo.sizes[l];
                let (a, b) = (&px.level(l).data, &py.level(l).data);
                let mut r = rng(l as u64 + 17);
                let flow0: Vec<f32> = (0..w * h * 2).map(|_| (r() * 3.0 - 1.5) as f32).collect();
                for cfg in 0..3 {
                    let masks = (cfg >= 1).then(|| (kx.level(l), ky.level(l)));
                    let u = (cfg == 2).then(|| &user.levels[l]);
                    let mut base = flow0.clone();
                    level_step(a, b, w, h, masks.map(|m| m.0), masks.map(|m| m.1), u, &s, l == 0, &mut base);
                    assert_ne!(bits(&base), bits(&flow0), "the step runs");
                    for k in ks(w) {
                        let (ra, rb) = (rot(a, w, 1, k), rot(b, w, 1, k));
                        let rm = masks.map(|(m0, m1)| (rot(m0, w, 1, k), rot(m1, w, 1, k)));
                        let ru = u.map(|u| rot_level(u, w, k));
                        let mut got = rot(&flow0, w, 2, k);
                        level_step(&ra, &rb, w, h, rm.as_ref().map(|m| &m.0[..]), rm.as_ref().map(|m| &m.1[..]), ru.as_ref(), &s, l == 0, &mut got);
                        assert_eq!(bits(&got), bits(&rot(&base, w, 2, k)), "{fw}x{fh} q {q} hiq {hiq} level {l} ({w}x{h}) config {cfg} k {k}");
                        n += 1;
                    }
                }
            }
        }
    }
    assert!(n > 300, "{n}");
}

#[test]
#[ignore = "the G6 wrap rule (U) this test encodes is superseded by the 360 rules read from TwixtorPro_64 (parity/lab/re/op-residual.md, finding 6)"]
fn the_wrap_upsample_rotates_with_the_coarse_level() {
    let mut r = rng(5);
    for (sw, sh) in [(2usize, 2usize), (3, 4), (5, 3), (8, 6), (13, 7)] {
        let (dw, dh) = (sw * 2, sh * 2 + 1);
        let src: Vec<f32> = (0..sw * sh * 2).map(|_| (r() * 4.0 - 2.0) as f32).collect();
        let base = upsample_flow_wrapx(&src, sw, sh, dw, dh);
        for k in 0..sw {
            let got = upsample_flow_wrapx(&rot(&src, sw, 2, k), sw, sh, dw, dh);
            assert_eq!(bits(&got), bits(&rot(&base, dw, 2, 2 * k)), "{sw}x{sh} k {k}");
        }
        // the period is kept: the columns right of the last coarse column read column 0, the even
        // columns are the coarse vectors times 2 exactly
        for yy in 0..dh {
            for xx in (0..dw).step_by(2) {
                let i = (yy * dw + xx) * 2;
                let fy = (yy as i32 as f32) * ((1.0f64 / (((dh as f32 - 1.0) / ((sh as i32 - 1) as f32)) as f64)) as f32);
                let c = crate::flow::sample::bilinear2_wrapx(&src, sw, sh, (xx / 2) as f32, fy);
                assert_eq!(base[i].to_bits(), (c[0] * 2.0).to_bits());
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// The whole driver

/// A pyramid of one level made of level 0 of `p` (and the mask and user levels to match).
fn one_level(p: &Pyramid) -> Pyramid {
    let (w, h) = p.geometry().sizes[0];
    let geometry = FlowGeometry { reduction: 0, width: w, height: h, levels: 1, sizes: vec![(w, h)] };
    Pyramid { geometry, levels: vec![p.level(0).clone()] }
}

fn rot_pyramid(p: &Pyramid, k: usize) -> Pyramid {
    let levels = p.levels.iter().map(|lv| Plane { width: lv.width, height: lv.height, data: rot(&lv.data, lv.width, 1, k) }).collect();
    Pyramid { geometry: p.geometry.clone(), levels }
}

#[test]
#[ignore = "the G6 wrap rule (U) this test encodes is superseded by the 360 rules read from TwixtorPro_64 (parity/lab/re/op-residual.md, finding 6)"]
fn a_one_level_wrap_solve_rotates_with_every_rotation_of_odd_and_even_widths() {
    for (fw, fh) in [(37usize, 29usize), (40, 23), (9, 7)] {
        for fmt in [Fmt::U8, Fmt::F32] {
            let (x, y) = (frame(fw, fh, fmt, (0.0, 0.0), 11), frame(fw, fh, fmt, (-1.3, 0.8), 11));
            let s = wrap(5, true, 1.0);
            let (px, py) = (one_level(&build_pyramid(&x, &s).unwrap()), one_level(&build_pyramid(&y, &s).unwrap()));
            let geo = px.geometry().clone();
            let (mx, my) = (MaskPyramid::build(&disc(fw, fh, (1.0, fh as f64 * 0.5), fh as f64 * 0.4), &s).unwrap(), MaskPyramid::build(&disc(fw, fh, (2.0, fh as f64 * 0.5), fh as f64 * 0.4), &s).unwrap());
            let (kx, ky) = (MaskPyramid { geometry: geo.clone(), levels: vec![mx.levels[0].clone()], gate: true }, MaskPyramid { geometry: geo.clone(), levels: vec![my.levels[0].clone()], gate: true });
            let user = points(&geo);
            for sens in [1.0f32, 0.8] {
                let s = FlowSettings { sensitivity: sens, ..s };
                let solve3 = |px: &Pyramid, py: &Pyramid, kx: &MaskPyramid, ky: &MaskPyramid, u: &UserDisplacement| {
                    [
                        solve_pyramids(px, py, &s).unwrap(),
                        solve_pyramids_with(px, py, &s, SolveExtras { masks: Some((kx, ky)), user: None }).unwrap(),
                        solve_pyramids_with(px, py, &s, SolveExtras { masks: Some((kx, ky)), user: Some(u) }).unwrap(),
                    ]
                };
                let base = solve3(&px, &py, &kx, &ky, &user);
                assert!(base[0].data.iter().any(|v| v.abs() > 0.1), "{fw}x{fh} the solve moves");
                for k in 1..fw {
                    let rk = |m: &MaskPyramid| MaskPyramid { geometry: geo.clone(), levels: vec![rot(&m.levels[0], fw, 1, k)], gate: true };
                    let ru = UserDisplacement { levels: vec![rot_level(&user.levels[0], fw, k)] };
                    let got = solve3(&rot_pyramid(&px, k), &rot_pyramid(&py, k), &rk(&kx), &rk(&ky), &ru);
                    for c in 0..3 {
                        assert_eq!(bits(&got[c].data), bits(&rot(&base[c].data, fw, 2, k)), "{fw}x{fh} {fmt:?} sensitivity {sens} config {c} k {k}");
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "the G6 wrap rule (U) this test encodes is superseded by the 360 rules read from TwixtorPro_64 (parity/lab/re/op-residual.md, finding 6)"]
fn rotating_both_frames_rotates_the_periodic_wrap_solve_bit_for_bit() {
    // the periodic pyramids of `build_pyramid` rotate bit for bit for multiples of
    // 2^(R + L - 1) columns when every resampled width is even; the solve then rotates too,
    // through `solve`, `solve_pair`, `solve_pyramids` and `solve_pyramids_with` (masks, points)
    let (w, h) = (256usize, 56usize);
    let mut n = 0;
    for fmt in [Fmt::U8, Fmt::F32] {
        let (x, y) = (frame(w, h, fmt, (0.0, 0.0), 21), frame(w, h, fmt, (2.6, 0.7), 21));
        let (mx, my) = (disc(w, h, (5.0, 28.0), 17.0), disc(w, h, (7.5, 28.5), 17.0));
        for (q, hiq) in QUALITIES {
            for sens in [1.0f32, 0.7] {
                let s = wrap(q, hiq, sens);
                let geo = s.geometry(w, h).unwrap();
                let p = 1usize << (geo.reduction as usize + geo.levels - 1);
                assert_eq!(w % p, 0);
                let (px, py) = (build_pyramid(&x, &s).unwrap(), build_pyramid(&y, &s).unwrap());
                let (kx, ky) = (MaskPyramid::build(&mx, &s).unwrap(), MaskPyramid::build(&my, &s).unwrap());
                let user = points(&geo);
                let base = [
                    solve_pyramids(&px, &py, &s).unwrap(),
                    solve_pyramids_with(&px, &py, &s, SolveExtras { masks: Some((&kx, &ky)), user: None }).unwrap(),
                    solve_pyramids_with(&px, &py, &s, SolveExtras { masks: Some((&kx, &ky)), user: Some(&user) }).unwrap(),
                ];
                assert_eq!(bits(&solve(&x, &y, &s).unwrap().data), bits(&base[0].data));
                for k in [p, 3 * p, w / 2, w - p] {
                    let (rx, ry) = (rot_image(&x, k), rot_image(&y, k));
                    let (qx, qy) = (build_pyramid(&rx, &s).unwrap(), build_pyramid(&ry, &s).unwrap());
                    let rm = |m: &Matte| Matte { width: w, height: h, data: rot(&m.data, w, 1, k) };
                    let (jx, jy) = (MaskPyramid::build(&rm(&mx), &s).unwrap(), MaskPyramid::build(&rm(&my), &s).unwrap());
                    let ru = UserDisplacement {
                        levels: user.levels.iter().enumerate().map(|(l, u)| rot_level(u, geo.sizes[l].0, k >> (geo.reduction as usize + l))).collect(),
                    };
                    let got = [
                        solve_pyramids(&qx, &qy, &s).unwrap(),
                        solve_pyramids_with(&qx, &qy, &s, SolveExtras { masks: Some((&jx, &jy)), user: None }).unwrap(),
                        solve_pyramids_with(&qx, &qy, &s, SolveExtras { masks: Some((&jx, &jy)), user: Some(&ru) }).unwrap(),
                    ];
                    let k0 = k >> geo.reduction;
                    for c in 0..3 {
                        assert_eq!(bits(&got[c].data), bits(&rot(&base[c].data, geo.width, 2, k0)), "{fmt:?} q {q} hiq {hiq} sensitivity {sens} config {c} k {k}");
                        n += 1;
                    }
                    let pair = solve_pair(&rx, &ry, &s).unwrap();
                    assert_eq!(bits(&pair.f_ab.data), bits(&rot(&base[0].data, geo.width, 2, k0)), "{fmt:?} q {q} hiq {hiq} k {k}: solve_pair");
                }
            }
        }
    }
    assert_eq!(n, 2 * 4 * 2 * 4 * 3);
}

#[test]
#[ignore = "the G6 wrap rule (U) this test encodes is superseded by the 360 rules read from TwixtorPro_64 (parity/lab/re/op-residual.md, finding 6)"]
fn the_wrap_solve_tracks_a_shift_and_differs_from_the_plain_one() {
    // the rotation rule does not cost the solve its accuracy: a periodic texture shifted across the
    // seam is recovered in every column, also in the two seam columns
    let (w, h) = (128usize, 48usize);
    let wf = w as f64;
    let tex = |x: f64, y: f64| 128.0 + 40.0 * (std::f64::consts::TAU * (3.0 * x / wf + 0.03 * y)).sin() + 30.0 * (std::f64::consts::TAU * (7.0 * x / wf - 0.05 * y) + 1.0).sin();
    let gray = |dx: f64| {
        let mut im = Image::new(w, h, Fmt::F32);
        if let Samples::F32(s) = &mut im.samples {
            for yy in 0..h {
                for xx in 0..w {
                    let v = (tex(xx as f64 - dx, yy as f64) / 255.0) as f32;
                    s[(yy * w + xx) * 4..(yy * w + xx) * 4 + 4].copy_from_slice(&[1.0, v, v, v]);
                }
            }
        }
        im
    };
    let s = wrap(5, true, 1.0);
    let f = solve(&gray(0.0), &gray(2.5), &s).unwrap();
    let plain = solve(&gray(0.0), &gray(2.5), &FlowSettings { wrap: false, ..s }).unwrap();
    let mut worst = 0.0f64;
    for c in 0..w {
        let mut e = 0.0f64;
        for yy in 8..h - 8 {
            let v = f.at(c, yy);
            e += ((v[0] as f64 - 2.5).powi(2) + (v[1] as f64).powi(2)).sqrt();
        }
        worst = worst.max(e / (h - 16) as f64);
    }
    assert!(worst < 0.05, "worst column {worst}");
    assert_ne!(bits(&f.data), bits(&plain.data));
}

// ---------------------------------------------------------------------------------------
// Gradient, wrap off, threads

#[test]
#[ignore = "the G6 wrap rule (U) this test encodes is superseded by the 360 rules read from TwixtorPro_64 (parity/lab/re/op-residual.md, finding 6)"]
fn away_from_the_seam_the_wrap_gradient_is_the_plain_one_bit_for_bit() {
    let mut r = rng(31);
    for (w, h) in [(2usize, 3usize), (3, 2), (7, 5), (40, 17), (129, 9)] {
        let img: Vec<f32> = (0..w * h).map(|_| (r() * 255.0) as f32 * 0.37).collect();
        let (g, p) = (calc_grad_wrap(&img, w, h), calc_grad(&img, w, h));
        for yy in 0..h {
            for xx in 0..w {
                let i = (yy * w + xx) * 2;
                assert_eq!(g[i + 1].to_bits(), p[i + 1].to_bits(), "gy ({xx}, {yy}) {w}x{h}");
                if xx >= 1 && xx + 2 <= w {
                    assert_eq!(g[i].to_bits(), p[i].to_bits(), "gx ({xx}, {yy}) {w}x{h}");
                } else {
                    // the seam columns: the same chain with the columns modulo the width
                    let c = img[yy * w + xx];
                    let (left, right) = (img[yy * w + (xx + w - 1) % w], img[yy * w + (xx + 1) % w]);
                    let want = (((c - left) + 0.0f32) + (right - c)) / 2.0f32;
                    assert_eq!(g[i].to_bits(), want.to_bits(), "seam gx ({xx}, {yy}) {w}x{h}");
                }
            }
        }
    }
}

/// The layered driver with wrap off as it stood before its level was split into [`level_step`].
fn frozen_layered(x: &Pyramid, y: &Pyramid, s: &FlowSettings, masks: Option<(&MaskPyramid, &MaskPyramid)>, user: Option<&UserDisplacement>) -> FlowField {
    let geo = &x.geometry;
    let (start, damp) = start_level(s.sensitivity, geo.levels);
    let (sw0, sh0) = geo.sizes[start];
    let mut flow = vec![0.0f32; sw0 * sh0 * 2];
    let mut check = true;
    for l in (0..=start).rev() {
        let (w, h) = geo.sizes[l];
        let img_a = &x.levels[l].data;
        let img_b = &y.levels[l].data;
        let mask_x = masks.map(|(m, _)| m.level(l));
        let mask_y = masks.map(|(_, m)| m.level(l));
        if let Some(u) = user {
            apply_user(&mut flow, &u.levels[l], w);
        }
        let mut grad = calc_grad(img_a, w, h);
        if let Some(m) = mask_x {
            mask_gradient(&mut grad, m, w, h, false);
        }
        let mut a = calc_a(&grad, w, h);
        conv::blur_gaussian_above_first(&mut a, w, h, 3, 7.0);
        for it in 0..3 {
            let mut b = if mask_y.is_none() { calc_b(img_a, img_b, &grad, &flow, w, h) } else { calc_b_with(img_a, img_b, &grad, &flow, w, h, mask_y, false) };
            conv::blur_gaussian_above_first(&mut b, w, h, 2, 7.0);
            new_disp(&mut flow, &a, &b);
            conv::blur_gaussian_above_first(&mut flow, w, h, 2, 5.0);
            if it == 1 && l == 0 {
                if !s.skip_refine {
                    refine(img_a, img_b, &mut flow, w, h, mask_x.filter(|_| masks.is_some_and(|(m, _)| m.gates())), false);
                }
                // the user displacement again right after the finest block matching step, as the original's solver
                if let Some(u) = user {
                    apply_user(&mut flow, &u.levels[l], w);
                }
            }
        }
        if l != 0 && !s.skip_refine {
            refine(img_a, img_b, &mut flow, w, h, mask_x.filter(|_| masks.is_some_and(|(m, _)| m.gates())), false);
        }
        if let Some(u) = user {
            apply_user(&mut flow, &u.levels[l], w);
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

#[test]
fn with_wrap_off_the_layered_driver_is_the_frozen_one() {
    let (w, h) = (75usize, 50usize);
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let (x, y) = (frame(w, h, fmt, (0.0, 0.0), 41), frame(w, h, fmt, (2.2, -1.1), 41));
        for (q, hiq) in QUALITIES {
            for skip_refine in [false, true] {
                let s = FlowSettings { skip_refine, ..FlowSettings::new(q, hiq, 0.9) };
                let (px, py) = (build_pyramid(&x, &s).unwrap(), build_pyramid(&y, &s).unwrap());
                let (kx, ky) = (MaskPyramid::build(&disc(w, h, (30.0, 25.0), 16.0), &s).unwrap(), MaskPyramid::build(&disc(w, h, (32.0, 24.0), 16.0), &s).unwrap());
                let user = points(px.geometry());
                for (m, u) in [(Some((&kx, &ky)), None), (None, Some(&user)), (Some((&kx, &ky)), Some(&user))] {
                    let got = solve_pyramids_with(&px, &py, &s, SolveExtras { masks: m, user: u }).unwrap();
                    assert_eq!(bits(&got.data), bits(&frozen_layered(&px, &py, &s, m, u).data), "{fmt:?} q {q} hiq {hiq} skip {skip_refine}");
                }
            }
        }
    }
}

#[test]
fn the_wrap_solve_gives_identical_bytes_on_1_4_and_16_threads() {
    let (w, h) = (96usize, 40usize);
    let (x, y) = (frame(w, h, Fmt::F32, (0.0, 0.0), 51), frame(w, h, Fmt::F32, (-3.1, 0.4), 51));
    for (q, hiq) in QUALITIES {
        let s = wrap(q, hiq, 1.0);
        let run = || {
            let (px, py) = (build_pyramid(&x, &s).unwrap(), build_pyramid(&y, &s).unwrap());
            let (kx, ky) = (MaskPyramid::build(&disc(w, h, (2.0, 20.0), 14.0), &s).unwrap(), MaskPyramid::build(&disc(w, h, (95.0, 20.0), 14.0), &s).unwrap());
            let user = points(px.geometry());
            let a = solve_pyramids(&px, &py, &s).unwrap();
            let b = solve_pyramids_with(&px, &py, &s, SolveExtras { masks: Some((&kx, &ky)), user: Some(&user) }).unwrap();
            (a, b)
        };
        let one = pool(1, run);
        for n in [4, 16] {
            let other = pool(n, run);
            assert_eq!(bits(&one.0.data), bits(&other.0.data), "q {q} hiq {hiq} {n} threads");
            assert_eq!(bits(&one.1.data), bits(&other.1.data), "q {q} hiq {hiq} {n} threads, masks and points");
        }
    }
}

// ---- the equirectangular rules of the 360 mode, read from the original's code

#[test]
fn sphere_target_brings_a_position_onto_the_frame() {
    assert_eq!(crate::flow::sphere_target(-0.25, 1.0, 4.0, 3.0), (3.75, 1.0));
    // one step over a pole: half a turn and the row mirrored
    assert_eq!(crate::flow::sphere_target(1.0, -0.5, 4.0, 3.0), (3.0, 0.5));
    assert_eq!(crate::flow::sphere_target(3.5, 3.25, 4.0, 3.0), (1.5, 2.75));
    // two steps: no turn
    assert_eq!(crate::flow::sphere_target(1.0, 6.5, 4.0, 3.0), (1.0, 0.5));
}

#[test]
fn the_wrap_gradient_reads_the_pole_rows_half_a_turn_away() {
    let (w, h) = (4usize, 3usize);
    let img: Vec<f32> = (0..12).map(|i| ((i * 7) % 11) as f32).collect();
    let g = crate::flow::calc_grad_wrap(&img, w, h);
    let at = |x: usize, y: usize| img[y * w + x];
    for y in 0..h {
        for x in 0..w {
            let c = at(x, y);
            let (l, r) = (at((x + w - 1) % w, y), at((x + 1) % w, y));
            let p = (x + w / 2) % w;
            let u = if y >= 1 { at(x, y - 1) } else { at(p, 0) };
            let d = if y + 1 < h { at(x, y + 1) } else { at(p, h - 1) };
            assert_eq!(g[(y * w + x) * 2], ((c - l) + (r - c)) * 0.5, "gx ({x}, {y})");
            assert_eq!(g[(y * w + x) * 2 + 1], ((c - u) + (d - c)) * 0.5, "gy ({x}, {y})");
        }
    }
}
