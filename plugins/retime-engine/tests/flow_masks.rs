//! Integration tests of the solver hooks through the public API: the layered solve with
//! mask pyramids, the user displacement and the wrap-around solve (rule G6).
//!
//! - identity: an all ones matte, a point outside the frame and wrap off change no byte
//! - one point sets its pixel exactly at Sensitivity 100
//! - analytic: a masked disc translating over a differently moving background recovers the
//!   disc's motion inside its mask
//! - wrap: a horizontally periodic texture shifted across the seam is recovered on both sides of
//!   the seam, where the plain solve fails
//! - 1, 4 and 16 threads give identical bytes
//!
//! The user displacements are built here by the stated rule G2 (a nearest pixel splat per level,
//! `d += (from - to) * k_l`, `w += 1`, points outside skipped); the engine's own producer is
//! `points::user_displacement`.

use retime_engine::flow::extras::{solve_pyramids_with, MaskPyramid, SolveExtras, UserDisplacement, UserLevel};
use retime_engine::flow::{build_pyramid, solve, solve_pyramids, FlowField, FlowGeometry, FlowSettings};
use retime_engine::image::{Fmt, Image, Samples};
use retime_engine::matte::Matte;

fn bits(f: &FlowField) -> Vec<u32> {
    f.data.iter().map(|v| v.to_bits()).collect()
}

fn pool<T: Send>(n: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap().install(f)
}

/// A 1/f like texture (as in `flow_d.rs`): 40 plane waves with periods from 6 to 300 pixels,
/// seeded, values about 0..255.
fn natural(x: f64, y: f64, seed: u64) -> f64 {
    let mut s = 0x9e3779b9u64 ^ seed.wrapping_mul(0x2545f4914f6cdd1d);
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut v = 0.0;
    let mut tot = 0.0;
    for k in 0..40 {
        let period = 6.0 * (50.0f64).powf(k as f64 / 39.0);
        let ang = rnd() * std::f64::consts::TAU;
        let ph = rnd() * std::f64::consts::TAU;
        let amp = period.sqrt();
        tot += amp;
        v += amp * (std::f64::consts::TAU * (x * ang.cos() + y * ang.sin()) / period + ph).sin();
    }
    128.0 + 100.0 * v / tot * 2.5
}

/// A texture periodic in x with period `w`: whole numbers of cycles across the width at
/// several scales, any frequency in y.
fn periodic(x: f64, y: f64, w: f64) -> f64 {
    let t = std::f64::consts::TAU;
    let mut v = 0.0;
    for (i, &(kx, fy, amp)) in [(1.0, 0.013, 30.0), (2.0, -0.021, 25.0), (3.0, 0.031, 22.0), (5.0, 0.047, 18.0), (8.0, -0.06, 14.0), (13.0, 0.08, 10.0), (21.0, -0.11, 7.0)]
        .iter()
        .enumerate()
    {
        v += amp * (t * (kx * x / w + fy * y) + i as f64 * 1.3).sin();
    }
    128.0 + v
}

/// Gray image of `f(x, y)` at `fmt` (alpha white).
fn gray(w: usize, h: usize, fmt: Fmt, f: impl Fn(f64, f64) -> f64) -> Image {
    let mut im = Image::new(w, h, fmt);
    for y in 0..h {
        for x in 0..w {
            let v = f(x as f64, y as f64).clamp(0.0, 255.0);
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

/// The user displacement of point pairs `(from X, to Y)` (output pixels) by rule G2: at level
/// `l`, `k = (w_l - 1) / (W - 1)` (and the same with the heights for y), the pixel
/// `trunc(p * k + 0.5)`, `d += (from - to) * k`, `w += 1`; a point whose pixel lies outside the
/// level is skipped.
fn splat(pairs: &[([f32; 2], [f32; 2])], geo: &FlowGeometry, out: (usize, usize)) -> UserDisplacement {
    let levels = geo
        .sizes
        .iter()
        .map(|&(lw, lh)| {
            let kx = (lw - 1) as f32 / (out.0 - 1) as f32;
            let ky = (lh - 1) as f32 / (out.1 - 1) as f32;
            let mut lv = UserLevel { d: vec![0.0; lw * lh * 2], w: vec![0.0; lw * lh] };
            for &(from, to) in pairs {
                let (fx, fy) = (from[0] * kx + 0.5, from[1] * ky + 0.5);
                if !(fx >= 0.0 && fy >= 0.0) {
                    continue;
                }
                let (ix, iy) = (fx as usize, fy as usize);
                if ix >= lw || iy >= lh {
                    continue;
                }
                let i = iy * lw + ix;
                lv.d[i * 2] = lv.d[i * 2] + (from[0] - to[0]) * kx;
                lv.d[i * 2 + 1] = lv.d[i * 2 + 1] + (from[1] - to[1]) * ky;
                lv.w[i] = lv.w[i] + 1.0;
            }
            lv
        })
        .collect();
    UserDisplacement { levels }
}

fn disc_matte(w: usize, h: usize, c: (f64, f64), r: f64) -> Matte {
    let data = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f64 - c.0, (i / w) as f64 - c.1);
            if x * x + y * y < r * r {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    Matte { width: w, height: h, data }
}

#[test]
fn an_all_ones_matte_and_wrap_off_change_no_byte() {
    for (w, h) in [(64usize, 48usize), (37, 29), (90, 20)] {
        for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
            let a = gray(w, h, fmt, |x, y| natural(x, y, 1));
            let b = gray(w, h, fmt, |x, y| natural(x - 1.5, y + 0.75, 1));
            for (q, hiq, sens) in [(5u8, true, 0.7f32), (4, true, 1.0), (3, false, 0.85), (1, true, 0.7), (5, false, 1.3)] {
                let s = FlowSettings::new(q, hiq, sens);
                let case = format!("{w}x{h} {fmt:?} q {q} hiq {hiq} sens {sens}");
                let (pa, pb) = (build_pyramid(&a, &s).unwrap(), build_pyramid(&b, &s).unwrap());
                let plain = solve(&a, &b, &s).unwrap();
                let ones = Matte { width: w, height: h, data: vec![1.0; w * h] };
                let m = MaskPyramid::build(&ones, &s).unwrap();
                let e = SolveExtras { masks: Some((&m, &m)), user: None };
                assert_eq!(bits(&solve_pyramids_with(&pa, &pb, &s, e).unwrap()), bits(&plain), "all ones {case}");
                let off = FlowSettings { wrap: false, ..s };
                assert_eq!(bits(&solve_pyramids(&pa, &pb, &off).unwrap()), bits(&plain), "wrap off {case}");
            }
        }
    }
}

#[test]
fn a_point_outside_the_frame_changes_nothing() {
    let (w, h) = (80usize, 60usize);
    let a = gray(w, h, Fmt::U8, |x, y| natural(x, y, 2));
    let b = gray(w, h, Fmt::U8, |x, y| natural(x - 2.0, y - 1.0, 2));
    for (q, sens) in [(5u8, 1.0f32), (4, 0.7)] {
        let s = FlowSettings::new(q, true, sens);
        let (pa, pb) = (build_pyramid(&a, &s).unwrap(), build_pyramid(&b, &s).unwrap());
        let plain = solve_pyramids(&pa, &pb, &s).unwrap();
        for pair in [([-30.0f32, 20.0f32], [-25.0f32, 22.0f32]), ([40.0, 200.0], [41.0, 201.0]), ([1.0e6, 3.0], [1.0e6, 3.0])] {
            let u = splat(&[pair], pa.geometry(), (w, h));
            assert!(u.levels.iter().all(|l| l.w.iter().all(|&v| v == 0.0)), "{pair:?} is skipped");
            let e = SolveExtras { masks: None, user: Some(&u) };
            assert_eq!(bits(&solve_pyramids_with(&pa, &pb, &s, e).unwrap()), bits(&plain), "q {q} {pair:?}");
        }
        // a point inside does change the solve
        let u = splat(&[([40.0, 30.0], [46.0, 32.0])], pa.geometry(), (w, h));
        let e = SolveExtras { masks: None, user: Some(&u) };
        assert_ne!(bits(&solve_pyramids_with(&pa, &pb, &s, e).unwrap()), bits(&plain));
    }
}

#[test]
fn one_point_sets_its_pixel_exactly_at_sensitivity_100() {
    let (w, h) = (96usize, 72usize);
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        // the image moves by (+1, +0.5); the point says (+6, +2)
        let a = gray(w, h, fmt, |x, y| natural(x, y, 3));
        let b = gray(w, h, fmt, |x, y| natural(x - 1.0, y - 0.5, 3));
        for q in [5u8, 4] {
            let s = FlowSettings::new(q, true, 1.0);
            let (pa, pb) = (build_pyramid(&a, &s).unwrap(), build_pyramid(&b, &s).unwrap());
            let geo = pa.geometry().clone();
            let (from, to) = ([40.0f32, 30.0f32], [46.0f32, 32.0f32]);
            let u = splat(&[(from, to)], &geo, (w, h));
            let e = SolveExtras { masks: None, user: Some(&u) };
            let f = solve_pyramids_with(&pa, &pb, &s, e).unwrap();
            // level 0 of the user displacement holds d = (from - to) * k and w = 1 at one pixel
            let lv = &u.levels[0];
            let i = lv.w.iter().position(|&v| v == 1.0).unwrap();
            assert_eq!(lv.w.iter().filter(|&&v| v != 0.0).count(), 1);
            let (x, y) = (i % geo.width, i / geo.width);
            assert_eq!(f.at(x, y), [-lv.d[i * 2], -lv.d[i * 2 + 1]], "{fmt:?} q {q}");
            if q == 5 {
                assert_eq!((x, y), (40, 30));
                assert_eq!(f.at(x, y), [6.0, 2.0], "{fmt:?}: to - from");
            }
            // away from the point the solve still follows the image
            let v = f.at(10, 60 * geo.height / h);
            let k = if q == 5 { 1.0 } else { 0.5 };
            assert!((v[0] - k).abs() < 0.25 && (v[1] - 0.5 * k).abs() < 0.25, "{fmt:?} q {q}: {v:?}");
        }
    }
}

/// Frames of a disc of its own texture moving by `vf` over a background moving by `vb`: `X`
/// has the disc at `c`, `Y` at `c + vf`.
fn disc_scene(w: usize, h: usize, c: (f64, f64), r: f64, vf: (f64, f64), vb: (f64, f64)) -> (Image, Image) {
    let frame = |c: (f64, f64), b: (f64, f64)| {
        gray(w, h, Fmt::F32, move |x, y| {
            let (dx, dy) = (x - c.0, y - c.1);
            if dx * dx + dy * dy < r * r {
                natural(dx, dy, 11)
            } else {
                natural(x - b.0, y - b.1, 12)
            }
        })
    };
    (frame(c, (0.0, 0.0)), frame((c.0 + vf.0, c.1 + vf.1), vb))
}

/// Mean and worst distance of the flow to `v` over the pixels with `pick(x, y)`.
fn error_over(f: &FlowField, v: (f32, f32), pick: impl Fn(usize, usize) -> bool) -> (f64, f64) {
    let (mut sum, mut n, mut worst) = (0.0f64, 0usize, 0.0f64);
    for y in 0..f.height {
        for x in 0..f.width {
            if pick(x, y) {
                let p = f.at(x, y);
                let e = (((p[0] - v.0) as f64).powi(2) + ((p[1] - v.1) as f64).powi(2)).sqrt();
                sum += e;
                n += 1;
                worst = worst.max(e);
            }
        }
    }
    assert!(n > 0);
    (sum / n as f64, worst)
}

/// The disc spans enough of every searched level: a matte loses about 2 pixels of its edge at
/// every binarize after a blur, so a small disc vanishes from the coarse levels, where its
/// layer then has no data and keeps the upsampled flow (160x120 with a disc of radius 30
/// moving 3.6 pixels stops at about 83% of the motion).
#[test]
fn a_masked_disc_recovers_its_own_motion_over_a_differently_moving_background() {
    let (w, h) = (256usize, 192usize);
    let (c, r) = ((125.0, 98.0), 60.0);
    let (vf, vb) = ((3.0, -2.0), (-2.5, 1.5));
    let (x, y) = disc_scene(w, h, c, r, vf, vb);
    let s = FlowSettings::new(5, true, 0.7);
    let (px, py) = (build_pyramid(&x, &s).unwrap(), build_pyramid(&y, &s).unwrap());
    let mx = MaskPyramid::build(&disc_matte(w, h, c, r), &s).unwrap();
    let my = MaskPyramid::build(&disc_matte(w, h, (c.0 + vf.0, c.1 + vf.1), r), &s).unwrap();
    let e = SolveExtras { masks: Some((&mx, &my)), user: None };
    let masked = solve_pyramids_with(&px, &py, &s, e).unwrap();
    let plain = solve_pyramids(&px, &py, &s).unwrap();
    let band = |lo: f64, hi: f64| {
        move |x: usize, y: usize| {
            let d = ((x as f64 - c.0).powi(2) + (y as f64 - c.1).powi(2)).sqrt();
            d >= lo && d < hi
        }
    };
    let target = (vf.0 as f32, vf.1 as f32);
    let (m_in, m_worst) = error_over(&masked, target, band(0.0, r - 2.0));
    let (p_in, p_worst) = error_over(&plain, target, band(0.0, r - 2.0));
    let (m_core, _) = error_over(&masked, target, band(0.0, r * 0.5));
    let (m_ring, _) = error_over(&masked, target, band(r - 10.0, r - 2.0));
    let (p_ring, _) = error_over(&plain, target, band(r - 10.0, r - 2.0));
    eprintln!("disc: masked mean {m_in:.4} worst {m_worst:.4} core {m_core:.4} ring {m_ring:.4}; plain mean {p_in:.4} worst {p_worst:.4} ring {p_ring:.4}");
    assert!(m_core < 0.01, "masked core error {m_core}");
    assert!(m_in < 0.03, "masked error inside the mask {m_in}");
    assert!(m_ring < 0.6 * p_ring, "the mask keeps the background out of the edge of the disc: {m_ring} against {p_ring}");
    assert!(m_worst < 0.5 * p_worst, "{m_worst} against {p_worst}");
    // the inverse mask: the background follows its own motion around the disc
    let inv = |m: &Matte| Matte { width: m.width, height: m.height, data: m.data.iter().map(|v| 1.0 - v).collect() };
    let bx = MaskPyramid::build(&inv(&disc_matte(w, h, c, r + 4.0)), &s).unwrap();
    let by = MaskPyramid::build(&inv(&disc_matte(w, h, (c.0 + vf.0, c.1 + vf.1), r + 4.0)), &s).unwrap();
    let e = SolveExtras { masks: Some((&bx, &by)), user: None };
    let bg = solve_pyramids_with(&px, &py, &s, e).unwrap();
    let ring = |x: usize, y: usize| {
        let d = ((x as f64 - c.0).powi(2) + (y as f64 - c.1).powi(2)).sqrt();
        d > r + 8.0 && d < r + 20.0 && x >= 12 && y >= 12 && x < w - 12 && y < h - 12
    };
    let (b_ring, _) = error_over(&bg, (vb.0 as f32, vb.1 as f32), ring);
    let (p_bring, _) = error_over(&plain, (vb.0 as f32, vb.1 as f32), ring);
    eprintln!("background ring: masked {b_ring:.4}, plain {p_bring:.4}");
    assert!(b_ring < 0.12 && b_ring < p_bring, "{b_ring} against {p_bring}");
}

/// A textured disc across the seam of a static periodic background, moving 4 pixels to the right
/// (it wraps): the wrap solve follows it on both sides of the seam, the plain solve cannot match
/// the part whose targets cross the seam. Under the wrap flag the luma pyramid is periodic as
/// well (its blurs are circular and its resample keeps the period), so a uniform
/// shift of a periodic texture is recovered in the seam columns too (`flow::tests_pyramid`);
/// `flow::tests_extras` shows the wrap kernels on hand built periodic pyramids.
#[test]
fn wrap_follows_a_disc_across_the_seam_where_the_plain_solve_fails() {
    let (w, h, r) = (160usize, 96usize, 26.0f64);
    let wf = w as f64;
    let cy = h as f64 / 2.0;
    let around = move |x: f64, cx: f64| {
        let mut dx = x - cx;
        while dx < -wf / 2.0 {
            dx += wf;
        }
        while dx >= wf / 2.0 {
            dx -= wf;
        }
        dx
    };
    let frame = |cx: f64, fmt: Fmt| {
        gray(w, h, fmt, move |x, y| {
            let (dx, dy) = (around(x, cx), y - cy);
            if dx * dx + dy * dy < r * r {
                natural(dx, dy, 21)
            } else {
                periodic(x, y, wf)
            }
        })
    };
    let (cx, mv) = (wf - 2.0, 4.0);
    for fmt in [Fmt::U8, Fmt::F32] {
        let (x, y) = (frame(cx, fmt), frame(cx + mv, fmt));
        let plain_s = FlowSettings::new(5, true, 1.0);
        let wrap_s = FlowSettings { wrap: true, ..plain_s };
        let fw = solve(&x, &y, &wrap_s).unwrap();
        let fp = solve(&x, &y, &plain_s).unwrap();
        let v = (mv as f32, 0.0f32);
        let disc = |right: bool| {
            move |xx: usize, yy: usize| {
                let (dx, dy) = (around(xx as f64, cx), yy as f64 - cy);
                (dx * dx + dy * dy).sqrt() < r - 3.0 && (xx > w / 2) == right
            }
        };
        let (wr, _) = error_over(&fw, v, disc(true));
        let (wl, _) = error_over(&fw, v, disc(false));
        let (pr, _) = error_over(&fp, v, disc(true));
        let (pl, _) = error_over(&fp, v, disc(false));
        eprintln!("{fmt:?} disc across the seam: wrap right {wr:.4} left {wl:.4}; plain right {pr:.4} left {pl:.4}");
        assert!(wr < 0.25 && wl < 0.25, "{fmt:?}: wrap {wl} {wr}");
        assert!(wr < 0.75 * pr && wl < 0.75 * pl, "{fmt:?}: wrap {wl} {wr} against plain {pl} {pr}");
        // away from the seam the background is static for both
        let (wm, _) = error_over(&fw, (0.0, 0.0), |xx, yy| xx > 50 && xx < 110 && yy > 8 && yy < h - 8);
        assert!(wm < 0.02, "{fmt:?}: {wm}");
    }
}

#[test]
fn masks_points_and_wrap_give_identical_bytes_on_1_4_and_16_threads() {
    let (w, h) = (120usize, 90usize);
    let (c, r) = ((40.0, 45.0), 22.0);
    let (x, y) = disc_scene(w, h, c, r, (2.0, 1.0), (-1.0, 0.5));
    for wrap in [false, true] {
        for q in [5u8, 4] {
            let s = FlowSettings { wrap, ..FlowSettings::new(q, true, 0.8) };
            let (px, py) = (build_pyramid(&x, &s).unwrap(), build_pyramid(&y, &s).unwrap());
            let mx = MaskPyramid::build(&disc_matte(w, h, c, r), &s).unwrap();
            let my = MaskPyramid::build(&disc_matte(w, h, (c.0 + 2.0, c.1 + 1.0), r), &s).unwrap();
            let u = splat(&[([40.0, 45.0], [42.0, 46.0]), ([100.0, 10.0], [99.0, 10.5]), ([40.2, 45.1], [42.0, 46.0])], px.geometry(), (w, h));
            let e = SolveExtras { masks: Some((&mx, &my)), user: Some(&u) };
            let f1 = pool(1, || solve_pyramids_with(&px, &py, &s, e).unwrap());
            for n in [4, 16] {
                let fnn = pool(n, || solve_pyramids_with(&px, &py, &s, e).unwrap());
                assert_eq!(bits(&f1), bits(&fnn), "wrap {wrap} q {q} threads {n}");
            }
            let p1 = pool(1, || MaskPyramid::build(&disc_matte(w, h, c, r), &s).unwrap());
            assert_eq!(pool(16, || MaskPyramid::build(&disc_matte(w, h, c, r), &s).unwrap()), p1);
        }
    }
}
