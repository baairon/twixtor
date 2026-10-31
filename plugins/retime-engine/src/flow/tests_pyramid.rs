//! Tests of the periodic solver input (rule G6, U): under `FlowSettings::wrap` the
//! reduction to the working size, the band pass pyramid of `build_pyramid`, the level 0 prep
//! transforms and `MaskPyramid::build` are circular in x.
//!
//! - identity: with wrap off every pyramid, reduction, transform and mask pyramid equals a frozen
//!   copy of the code as it stood before the flag reached them, bit for bit, over sizes, depths,
//!   qualities, draft and prep options;
//! - shift equivariance: rotating the frame (or the matte) circularly in x by a multiple of
//!   `2^(R + L - 1)` columns rotates every level, bit for bit. The solved flow rotates bit for bit
//!   through a model driver whose four non periodic steps (Gaussians, upsample, target column,
//!   sweeps) are made circular (plain, with mask pyramids, with a user displacement); the
//!   engine's own driver is periodic as well (`flow/extras/tests_wrap.rs`), so it rotates bit for
//!   bit too;
//! - accuracy: a horizontally periodic texture shifted by a sub pixel amount (across the seam too)
//!   is recovered in every column with wrap, where the plain solve fails at the seam;
//! - the transforms and the reduction against their definitions, the seam columns, and 1, 4 and
//!   16 threads.

use super::extras::{apply_user, MaskPyramid, UserDisplacement, UserLevel};
use super::sample::cvtt;
use super::*;
use crate::image::{Fmt, Image, Samples};
use crate::matte::Matte;
use crate::prep::{self, Plane};

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn pool<T: Send>(n: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap().install(f)
}

fn wrap_of(s: FlowSettings) -> FlowSettings {
    FlowSettings { wrap: true, ..s }
}

// ---------------------------------------------------------------------------------------
// Frozen copies of the pyramid code before the wrap flag reached it

/// `prep::reduce` as it stood: `r` stages of `blur5` and the corner aligned resample to the
/// halved size.
fn frozen_reduce(mut t: Plane, r: u32) -> Plane {
    for _ in 0..r {
        conv::blur5_above_first(&mut t.data, t.width, t.height);
        let (nw, nh) = (t.width / 2, t.height / 2);
        t = Plane { width: nw, height: nh, data: sample::resample1(&t.data, t.width, t.height, nw, nh) };
    }
    t
}

/// `prep::edge_enhance` as it stood.
fn frozen_edge_enhance(p: &mut Plane) {
    let (w, h) = (p.width, p.height);
    conv::blur_gaussian_above_first(&mut p.data, w, h, 1, 2.0);
    let g = calc_grad(&p.data, w, h);
    for y in 0..h {
        for x in 0..w {
            let gx = g[(y * w + x) * 2];
            let gy = g[(y * w + x) * 2 + 1];
            let m = (gx * gx + gy * gy).sqrt();
            let f = if m <= 20.0f32 { m * 0.05f32 } else { 1.0f32 };
            let v = &mut p.data[y * w + x];
            *v = (f * 255.0f32 + *v) * 0.5f32;
        }
    }
}

/// `build_pyramid` as it stood: luma, the plain reduction, the transforms, then per level
/// `band = orig - blur5(orig)` and `orig' = resample(blur5(orig))`.
fn frozen_build_pyramid(img: &Image, s: &FlowSettings) -> Pyramid {
    let geometry = s.geometry(img.width, img.height).unwrap();
    let luma = prep::luma(img, s.track_alpha, s.video_range_luma).unwrap();
    let mut level0 = frozen_reduce(luma, geometry.reduction);
    if s.prep_delinearize {
        prep::delinearize(&mut level0);
    }
    if s.prep_contrast_edge {
        frozen_edge_enhance(&mut level0);
    }
    let n = geometry.levels;
    let mut orig = level0;
    let mut levels = Vec::with_capacity(n);
    for i in 0..n {
        let mut tmp = orig.clone();
        conv::blur5_above_first(&mut tmp.data, tmp.width, tmp.height);
        let next = if i + 1 < n {
            let (nw, nh) = geometry.sizes[i + 1];
            Some(Plane { width: nw, height: nh, data: sample::resample1(&tmp.data, tmp.width, tmp.height, nw, nh) })
        } else {
            None
        };
        let mut band = orig;
        for (a, b) in band.data.iter_mut().zip(tmp.data.iter()) {
            *a = *a - *b;
        }
        levels.push(band);
        match next {
            Some(p) => orig = p,
            None => break,
        }
    }
    Pyramid { geometry, levels }
}

/// `MaskPyramid::build` as it stood: binarize, then per reduction step and level copy, `blur5`,
/// the corner aligned resample, binarize.
fn frozen_mask_levels(m: &Matte, s: &FlowSettings) -> Vec<Vec<f32>> {
    let geometry = s.geometry(m.width, m.height).unwrap();
    let bin = |v: &mut Vec<f32>| v.iter_mut().for_each(|x| *x = if *x >= MASK_ON { 1.0 } else { 0.0 });
    let step = |p: &Plane, nw: usize, nh: usize| {
        let mut t = p.clone();
        conv::blur5_above_first(&mut t.data, t.width, t.height);
        let mut d = sample::resample1(&t.data, t.width, t.height, nw, nh);
        bin(&mut d);
        Plane { width: nw, height: nh, data: d }
    };
    let mut cur = Plane { width: m.width, height: m.height, data: m.data.clone() };
    bin(&mut cur.data);
    for _ in 0..geometry.reduction {
        let (nw, nh) = (cur.width / 2, cur.height / 2);
        cur = step(&cur, nw, nh);
    }
    let mut out = vec![cur.data.clone()];
    for i in 1..geometry.levels {
        let (nw, nh) = geometry.sizes[i];
        cur = step(&cur, nw, nh);
        out.push(cur.data.clone());
    }
    out
}

// ---------------------------------------------------------------------------------------
// Inputs and rotations

/// xorshift64 in 0..1.
fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = 0x9e3779b97f4a7c15u64 ^ seed.wrapping_mul(0x2545f4914f6cdd1d);
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A texture of period `w` in x: whole numbers of cycles across the width, odd counts among them
/// (so no rotation by a half or a quarter of the width maps it onto itself), any frequency in y,
/// and up to `max_k` cycles.
fn periodic(x: f64, y: f64, w: f64, max_k: f64) -> f64 {
    let t = std::f64::consts::TAU;
    let mut v = 128.0;
    for (i, &(kx, fy, amp)) in
        [(1.0, 0.013, 30.0), (2.0, -0.021, 25.0), (3.0, 0.031, 22.0), (5.0, 0.047, 18.0), (8.0, -0.06, 14.0), (13.0, 0.08, 10.0), (21.0, -0.11, 7.0)]
            .iter()
            .enumerate()
    {
        if kx <= max_k {
            v += amp * (t * (kx * x / w + fy * y) + i as f64 * 1.3).sin();
        }
    }
    v
}

/// Gray image of `f(x, y)` (clamped to 0..255) at `fmt`, alpha white.
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

/// A colour image with random channels and a random alpha (premultiplied colour), so the luma,
/// the alpha term of Track RGB+A and every depth see generic values.
fn noisy(w: usize, h: usize, fmt: Fmt, seed: u64) -> Image {
    let mut r = rng(seed);
    let mut im = Image::new(w, h, fmt);
    for i in 0..w * h {
        let a = 0.3 + 0.7 * r();
        let c = [a, a * r(), a * r(), a * r()];
        match &mut im.samples {
            Samples::U8(s) => (0..4).for_each(|k| s[i * 4 + k] = (c[k] * 255.0 + 0.5).floor() as u8),
            Samples::U16(s) => (0..4).for_each(|k| s[i * 4 + k] = (c[k] * 65535.0 + 0.5).floor() as u16),
            Samples::F32(s) => (0..4).for_each(|k| s[i * 4 + k] = c[k] as f32),
        }
    }
    im
}

/// `out(x) = v((x - s) mod w)` on rows of `w` elements of `c` values: the content moves `s`
/// columns to the right, across the seam.
fn rot<T: Copy>(v: &[T], w: usize, c: usize, s: usize) -> Vec<T> {
    let mut out = v.to_vec();
    for (y, row) in v.chunks(w * c).enumerate() {
        for x in 0..w {
            let to = (x + s) % w;
            out[(y * w + to) * c..(y * w + to + 1) * c].copy_from_slice(&row[x * c..(x + 1) * c]);
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

fn rot_matte(m: &Matte, s: usize) -> Matte {
    Matte { width: m.width, height: m.height, data: rot(&m.data, m.width, 1, s) }
}

/// The rotation of level `l` for a rotation of `s` frame columns (`s` divisible by `2^(R + l)`).
fn level_shift(geo: &FlowGeometry, l: usize, s: usize) -> usize {
    let k = 1usize << (geo.reduction as usize + l);
    assert_eq!(s % k, 0, "a rotation of {s} columns is not whole at level {l}");
    s / k
}

fn rot_user(u: &UserDisplacement, geo: &FlowGeometry, s: usize) -> UserDisplacement {
    let levels = u
        .levels
        .iter()
        .enumerate()
        .map(|(l, lv)| {
            let (w, _) = geo.sizes[l];
            let sl = level_shift(geo, l, s);
            UserLevel { d: rot(&lv.d, w, 2, sl), w: rot(&lv.w, w, 1, sl) }
        })
        .collect();
    UserDisplacement { levels }
}

/// A disc of radius `r` around `c` on a cylinder of circumference `w` (it crosses the seam when
/// `c.0` is within `r` of it).
fn disc_on_cylinder(w: usize, h: usize, c: (f64, f64), r: f64) -> Matte {
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
            if dx * dx + dy * dy < r * r {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    Matte { width: w, height: h, data }
}

/// Every setting the vector paths produce: the four vectors qualities, the full resolution of
/// Create Vectors (`q` 5), each at high quality and draft.
fn all_qualities() -> Vec<(u8, bool)> {
    let mut v = Vec::new();
    for q in 1..=5u8 {
        for hiq in [true, false] {
            v.push((q, hiq));
        }
    }
    v
}

// ---------------------------------------------------------------------------------------
// Identity with wrap off

#[test]
fn wrap_off_pyramids_reductions_and_masks_equal_the_frozen_code_bit_for_bit() {
    let sizes = [
        (2usize, 2usize),
        (3, 2),
        (2, 5),
        (5, 3),
        (7, 7),
        (13, 9),
        (16, 16),
        (24, 16),
        (31, 17),
        (37, 29),
        (40, 30),
        (64, 48),
        (65, 33),
        (96, 20),
        (127, 64),
    ];
    let mut pyramids = 0;
    let mut masks = 0;
    for (si, &(w, h)) in sizes.iter().enumerate() {
        for (di, fmt) in [Fmt::U8, Fmt::U16, Fmt::F32].into_iter().enumerate() {
            let img = noisy(w, h, fmt, (si * 3 + di) as u64);
            for (q, hiq) in all_qualities() {
                for prep_k in 0..4usize {
                    let k = si + di + q as usize + prep_k;
                    let mut s = FlowSettings::new(q, hiq, 0.7);
                    s.prep_delinearize = prep_k & 1 != 0;
                    s.prep_contrast_edge = prep_k & 2 != 0;
                    s.track_alpha = k % 3 == 1;
                    s.video_range_luma = k % 5 == 2 && !s.track_alpha;
                    if s.geometry(w, h).is_err() {
                        continue;
                    }
                    let case = format!("{w}x{h} {fmt:?} {s:?}");
                    let got = build_pyramid(&img, &s).unwrap();
                    let want = frozen_build_pyramid(&img, &s);
                    assert_eq!(got.geometry, want.geometry, "{case}");
                    for l in 0..want.levels.len() {
                        assert_eq!(bits(&got.levels[l].data), bits(&want.levels[l].data), "{case} level {l}");
                    }
                    pyramids += 1;
                }
            }
        }
        // mask pyramids: a disc, a soft ramp around the threshold and a random matte
        let mut r = rng(si as u64 + 100);
        let mattes = [
            disc_on_cylinder(w, h, (w as f64 * 0.4, h as f64 * 0.5), (w.min(h) as f64) * 0.3),
            Matte { width: w, height: h, data: (0..w * h).map(|i| 0.9 + 0.1 * ((i % w) as f32 / w as f32)).collect() },
            Matte { width: w, height: h, data: (0..w * h).map(|_| r() as f32).collect() },
        ];
        for m in &mattes {
            for (q, hiq) in all_qualities() {
                let s = FlowSettings::new(q, hiq, 0.7);
                if s.geometry(w, h).is_err() {
                    continue;
                }
                let got = MaskPyramid::build(m, &s).unwrap();
                let want = frozen_mask_levels(m, &s);
                assert_eq!(got.geometry().levels, want.len());
                for (l, lv) in want.iter().enumerate() {
                    assert_eq!(bits(got.level(l)), bits(lv), "mask {w}x{h} q {q} hiq {hiq} level {l}");
                }
                masks += 1;
            }
        }
    }
    // the reduction and the transforms on their own, every reduction count
    let mut steps = 0;
    for &(w, h) in &sizes {
        let mut r = rng((w * 1000 + h) as u64);
        let p = Plane { width: w, height: h, data: (0..w * h).map(|_| (r() * 255.0) as f32).collect() };
        for red in 0..=3u32 {
            if (w >> red) < 1 || (h >> red) < 1 {
                continue;
            }
            assert_eq!(bits(&prep::reduce_with(p.clone(), red, false).data), bits(&frozen_reduce(p.clone(), red).data), "{w}x{h} R {red}");
            assert_eq!(bits(&prep::reduce(p.clone(), red).data), bits(&frozen_reduce(p.clone(), red).data), "{w}x{h} R {red}");
            steps += 1;
        }
        let (mut a, mut b) = (p.clone(), p.clone());
        prep::edge_enhance_with(&mut a, false);
        frozen_edge_enhance(&mut b);
        assert_eq!(bits(&a.data), bits(&b.data), "edge enhance {w}x{h}");
        let mut c = p.clone();
        prep::blur5_with(&mut c, false);
        let mut d = p.clone();
        conv::blur5_above_first(&mut d.data, w, h);
        assert_eq!(bits(&c.data), bits(&d.data), "blur5 {w}x{h}");
        let (nw, nh) = ((w / 2).max(2), (h / 2).max(2));
        assert_eq!(bits(&prep::resample_with(&p, nw, nh, false).data), bits(&sample::resample1(&p.data, w, h, nw, nh)), "resample {w}x{h}");
        steps += 3;
    }
    eprintln!("wrap off identity: {pyramids} pyramids, {masks} mask pyramids, {steps} steps");
    assert!(pyramids >= 1050, "{pyramids} pyramid combinations");
    assert!(masks >= 300 && steps >= 80, "{masks} {steps}");
}

#[test]
fn the_solve_with_wrap_off_equals_the_frozen_pyramids_and_driver() {
    // the whole plain chain: frozen pyramids into the plain driver give `solve`'s bytes
    for (w, h) in [(48usize, 32usize), (37, 29), (64, 40)] {
        for fmt in [Fmt::U8, Fmt::F32] {
            let a = gray(w, h, fmt, |x, y| periodic(x, y, w as f64, 13.0));
            let b = gray(w, h, fmt, |x, y| periodic(x - 1.6, y + 0.4, w as f64, 13.0));
            for (q, hiq) in all_qualities() {
                let s = FlowSettings::new(q, hiq, 0.8);
                let (pa, pb) = (frozen_build_pyramid(&a, &s), frozen_build_pyramid(&b, &s));
                let want = solve_pyramids(&pa, &pb, &s).unwrap();
                assert_eq!(bits(&solve(&a, &b, &s).unwrap().data), bits(&want.data), "{w}x{h} {fmt:?} q {q} hiq {hiq}");
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// Shift equivariance of the solver input

/// Asserts that every level of `rotated` is level `l` of `base` rotated by `s / 2^(R + l)`.
fn assert_rotated_levels(base: &Pyramid, rotated: &Pyramid, s: usize, case: &str) {
    let geo = &base.geometry;
    assert_eq!(&rotated.geometry, geo, "{case}");
    for l in 0..geo.levels {
        let (w, _) = geo.sizes[l];
        let want = rot(&base.levels[l].data, w, 1, level_shift(geo, l, s));
        assert_eq!(bits(&rotated.levels[l].data), bits(&want), "{case}: level {l} ({w} wide) rotated by {s}");
    }
}

#[test]
#[ignore = "the G6 wrap rule (U) this test encodes is superseded by the 360 rules read from TwixtorPro_64 (parity/lab/re/op-residual.md, finding 6: the edge enhancement blur of the 360 mode is the plain one)"]
fn rotating_the_frame_rotates_every_wrap_pyramid_level_bit_for_bit() {
    // 256 x 128: every setting has R + L - 1 = 5, so rotations by multiples of 32 columns are
    // whole at every level (8 columns of 32 at the coarsest level)
    let (w, h) = (256usize, 128usize);
    let mut n = 0;
    for (di, fmt) in [Fmt::U8, Fmt::U16, Fmt::F32].into_iter().enumerate() {
        let img = noisy(w, h, fmt, 40 + di as u64);
        for (q, hiq) in all_qualities() {
            for prep_k in 0..4usize {
                let mut s = wrap_of(FlowSettings::new(q, hiq, 0.7));
                s.prep_delinearize = prep_k & 1 != 0;
                s.prep_contrast_edge = prep_k & 2 != 0;
                s.track_alpha = (q as usize + prep_k) % 2 == 1;
                let geo = s.geometry(w, h).unwrap();
                assert_eq!(geo.reduction as usize + geo.levels - 1, 5, "{s:?}");
                let base = build_pyramid(&img, &s).unwrap();
                for shift in [w / 2, w / 4, 32, w - 32] {
                    let rotated = build_pyramid(&rot_image(&img, shift), &s).unwrap();
                    assert_rotated_levels(&base, &rotated, shift, &format!("{fmt:?} {s:?}"));
                    n += 1;
                }
                // the plain pyramid is not periodic: its rotation differs (the test has teeth)
                if prep_k == 0 && di == 0 {
                    let p = FlowSettings { wrap: false, ..s };
                    let (pb, pr) = (build_pyramid(&img, &p).unwrap(), build_pyramid(&rot_image(&img, w / 2), &p).unwrap());
                    let l0 = rot(&pb.levels[0].data, geo.width, 1, geo.width / 2);
                    assert_ne!(bits(&pr.levels[0].data), bits(&l0), "plain {s:?}");
                }
            }
        }
    }
    assert_eq!(n, 3 * 10 * 4 * 4);
}

#[test]
#[ignore = "the G6 wrap rule (U) this test encodes is superseded by the 360 rules read from TwixtorPro_64 (parity/lab/re/op-residual.md, finding 6: the edge enhancement blur of the 360 mode is the plain one)"]
fn the_reduction_and_the_transforms_are_circular_at_any_even_rotation() {
    // the reduction alone: a rotation by a multiple of 2^r columns of an image whose halvings
    // are all even; the transforms (level 0, no resample): any rotation
    let (w, h) = (96usize, 40usize);
    let mut r = rng(7);
    let p = Plane { width: w, height: h, data: (0..w * h).map(|_| (r() * 255.0) as f32).collect() };
    for red in 0..=3u32 {
        let k = 1usize << red;
        let base = prep::reduce_with(p.clone(), red, true);
        assert_eq!((base.width, base.height), (w >> red, h >> red));
        for s in [k, 3 * k, w / 2, w - k] {
            let got = prep::reduce_with(Plane { width: w, height: h, data: rot(&p.data, w, 1, s) }, red, true);
            assert_eq!(bits(&got.data), bits(&rot(&base.data, base.width, 1, s / k)), "R {red} rotation {s}");
        }
    }
    for s in [1usize, 5, 47, 95] {
        let rp = Plane { width: w, height: h, data: rot(&p.data, w, 1, s) };
        let (mut a, mut b) = (p.clone(), rp.clone());
        prep::edge_enhance_with(&mut a, true);
        prep::edge_enhance_with(&mut b, true);
        assert_eq!(bits(&b.data), bits(&rot(&a.data, w, 1, s)), "edge enhance rotation {s}");
        let (mut a, mut b) = (p.clone(), rp);
        prep::blur5_with(&mut a, true);
        prep::blur5_with(&mut b, true);
        assert_eq!(bits(&b.data), bits(&rot(&a.data, w, 1, s)), "blur5 rotation {s}");
    }
    // the plain reduction and edge enhance are not circular (the test has teeth)
    let rp = Plane { width: w, height: h, data: rot(&p.data, w, 1, w / 2) };
    let (base, got) = (prep::reduce_with(p.clone(), 1, false), prep::reduce_with(rp.clone(), 1, false));
    assert_ne!(bits(&got.data), bits(&rot(&base.data, w / 2, 1, w / 4)));
    let (mut a, mut b) = (p.clone(), rp);
    prep::edge_enhance_with(&mut a, false);
    prep::edge_enhance_with(&mut b, false);
    assert_ne!(bits(&b.data), bits(&rot(&a.data, w, 1, w / 2)));
}

#[test]
fn a_mask_disc_across_the_seam_is_the_rotated_mask_of_the_centred_disc() {
    let (w, h) = (256usize, 128usize);
    for (q, hiq) in all_qualities() {
        let s = wrap_of(FlowSettings::new(q, hiq, 0.7));
        // a disc centred on the seam, and the same disc half a width away (in the middle)
        let across = disc_on_cylinder(w, h, (3.0, 61.0), 40.0);
        let centred = disc_on_cylinder(w, h, (3.0 + w as f64 / 2.0, 61.0), 40.0);
        assert_eq!(rot_matte(&centred, w / 2).data, across.data);
        let (ma, mc) = (MaskPyramid::build(&across, &s).unwrap(), MaskPyramid::build(&centred, &s).unwrap());
        let geo = ma.geometry().clone();
        let mut ones = 0;
        for l in 0..geo.levels {
            let (lw, lh) = geo.sizes[l];
            let want = rot(mc.level(l), lw, 1, level_shift(&geo, l, w / 2));
            assert_eq!(bits(ma.level(l)), bits(&want), "q {q} hiq {hiq} level {l}");
            ones += ma.level(l).iter().filter(|&&v| v == 1.0).count();
            // at the working size the disc covers both edge columns (it crosses the seam)
            if l == 0 {
                let col_on = |c: usize| (0..lh).any(|y| ma.level(l)[y * lw + c] == 1.0);
                assert!(col_on(0) && col_on(lw - 1), "q {q} hiq {hiq}");
            }
        }
        assert!(ones > 0);
        // the plain mask pyramid of the disc across the seam differs from the rotated centred one
        let p = FlowSettings { wrap: false, ..s };
        let (pa, pc) = (MaskPyramid::build(&across, &p).unwrap(), MaskPyramid::build(&centred, &p).unwrap());
        let differs = (0..geo.levels).any(|l| bits(pa.level(l)) != bits(&rot(pc.level(l), geo.sizes[l].0, 1, level_shift(&geo, l, w / 2))));
        assert!(differs, "plain q {q} hiq {hiq}");
    }
}

// ---------------------------------------------------------------------------------------
// Shift equivariance of the solve

/// 2 channel bilinear sampler periodic in x (the model driver's upsample): `ix = trunc(x)`,
/// `fx = x - ix`, the right neighbour of the last column is column 0; y keeps the plain rule (the
/// last row moves the cell up with `fy = 1`). The lerps of `bilinear2`.
fn bilinear2_periodic(img: &[f32], w: usize, h: usize, x: f32, y: f32) -> [f32; 2] {
    let ix = cvtt(x).max(0) as usize;
    let fx = x - (ix as f32);
    let x1 = if ix + 1 >= w { 0 } else { ix + 1 };
    let mut iy = cvtt(y);
    let mut fy = y - (iy as f32);
    if iy >= h as i32 - 1 {
        iy = h as i32 - 2;
        fy = 1.0;
    }
    let iy = iy.max(0) as usize;
    let mut out = [0.0f32; 2];
    for c in 0..2 {
        let p00 = img[(iy * w + ix) * 2 + c];
        let p10 = img[(iy * w + x1) * 2 + c];
        let p01 = img[((iy + 1) * w + ix) * 2 + c];
        let p11 = img[((iy + 1) * w + x1) * 2 + c];
        let top = (p10 - p00) * fx + p00;
        let bot = (p11 - p01) * fx + p01;
        out[c] = (bot - top) * fy + top;
    }
    out
}

/// The flow upsample of the model driver: the period keeping x of the wrap pyramid (`x * sw /
/// dw`, vectors times `dw / sw`), the corner aligned y of `upsample_flow`.
fn upsample_flow_periodic(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let sx = dw as f32 / sw as f32;
    let rx = sw as f32 / dw as f32;
    let sy = (dh as f32 - 1.0f32) / ((sh as i32 - 1) as f32);
    let ry = (1.0f64 / (sy as f64)) as f32;
    let mut dst = vec![0.0f32; dw * dh * 2];
    for y in 0..dh {
        let fy = (y as i32 as f32) * ry;
        for x in 0..dw {
            let v = bilinear2_periodic(src, sw, sh, (x as i32 as f32) * rx, fy);
            dst[(y * dw + x) * 2] = v[0] * sx;
            dst[(y * dw + x) * 2 + 1] = v[1] * sy;
        }
    }
    dst
}

/// The B vector step of the model driver: `calc_b_with` with wrap, except that the target column
/// is split exactly into the pixel's column plus the whole part of the displacement (integer
/// arithmetic modulo the width) and its fraction `t - floor(t)` (`t = -u`), instead of the single
/// precision `x - u`, whose rounding depends on the size of `x` (so no rotation can reproduce its
/// bits). The rows, the range check of `sy`, the warped mask test, the lerps and the products are
/// those of `calc_b_with`.
fn calc_b_split(i1: &[f32], i0: &[f32], grad: &[f32], flow: &[f32], w: usize, h: usize, mask_y: Option<&[f32]>) -> Vec<f32> {
    let mut b = vec![0.0f32; w * h * 2];
    let fh = h as f32;
    for y in 0..h {
        let fy = y as f32;
        for x in 0..w {
            let i = y * w + x;
            let t = -flow[i * 2];
            let sy = fy - flow[i * 2 + 1];
            if !t.is_finite() || t.abs() > 1.0e6 || sy < 0.0 || fh - 1.0 < sy {
                continue;
            }
            let whole = t.floor();
            let fx = t - whole;
            let col = (x as i64 + whole as i64).rem_euclid(w as i64) as usize;
            let (col, fx) = if fx >= 1.0 { ((col + 1) % w, fx - 1.0) } else { (col, fx) };
            let x1 = (col + 1) % w;
            let mut iy = cvtt(sy);
            let mut wy = sy - (iy as f32);
            if iy >= h as i32 - 1 {
                iy = h as i32 - 2;
                wy = 1.0;
            }
            let iy = iy.max(0) as usize;
            let sample = |img: &[f32]| {
                let (r0, r1) = (iy * w, (iy + 1) * w);
                let top = (img[r0 + x1] - img[r0 + col]) * fx + img[r0 + col];
                let bot = (img[r1 + x1] - img[r1 + col]) * fx + img[r1 + col];
                (bot - top) * wy + top
            };
            if let Some(m) = mask_y {
                if !(sample(m) >= WARPED_MASK_MIN) {
                    continue;
                }
            }
            let r = sample(i0) - i1[i];
            b[i * 2] = r * grad[i * 2];
            b[i * 2 + 1] = r * grad[i * 2 + 1];
        }
    }
    b
}

/// The wrap driver of `extras::solve_pyramids_with` with the steps that no rotation can follow
/// replaced: the Gaussian blurs of the tensor, the B vector and the flow through
/// `blur_gaussian_wrapx` (the engine's renormalise at the left and right borders), the flow
/// upsample through [`upsample_flow_periodic`] (the engine's is corner aligned), the B vector step
/// through [`calc_b_split`] (the engine's target column `x - u` rounds with the size of `x`, unless
/// `split` is false), and no block matching (its in place sweeps start at column 1 and skip the
/// edge columns). Every other step is the engine's own kernel: the wrap gradient, the gradient
/// masking, the tensor, the update, the user displacement, the range check of y, the damping. A
/// model only: fed with the periodic pyramids it shows that the solver input is circular.
fn circular_model_solve(
    x: &Pyramid,
    y: &Pyramid,
    s: &FlowSettings,
    masks: Option<(&MaskPyramid, &MaskPyramid)>,
    user: Option<&UserDisplacement>,
    split: bool,
) -> FlowField {
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
        let mut grad = calc_grad_wrap(img_a, w, h);
        if let Some(m) = mask_x {
            mask_gradient(&mut grad, m, w, h, true);
        }
        let mut a = calc_a(&grad, w, h);
        conv::blur_gaussian_wrapx_above_first(&mut a, w, h, 3, 7.0);
        for _ in 0..3 {
            let mut b = if split {
                calc_b_split(img_a, img_b, &grad, &flow, w, h, mask_y)
            } else {
                calc_b_with(img_a, img_b, &grad, &flow, w, h, mask_y, true)
            };
            conv::blur_gaussian_wrapx_above_first(&mut b, w, h, 2, 7.0);
            new_disp(&mut flow, &a, &b);
            conv::blur_gaussian_wrapx_above_first(&mut flow, w, h, 2, 5.0);
        }
        if let Some(u) = user {
            apply_user(&mut flow, &u.levels[l], w);
        }
        if check {
            if all_flow_out_of_range_y(&flow, w, h) {
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
            flow = upsample_flow_periodic(&flow, w, h, nw, nh);
        }
    }
    scale_values(&mut flow, -1.0f32);
    FlowField { width: geo.width, height: geo.height, data: flow }
}

/// Two frames on a cylinder: a periodic background moving `(0.7, -0.3)` and a textured disc that
/// crosses the seam moving `(2.5, 1.0)`; the matte of the disc at both frames.
fn seam_scene(w: usize, h: usize, fmt: Fmt) -> (Image, Image, Matte, Matte) {
    let wf = w as f64;
    let (c, r) = ((6.0, h as f64 * 0.45), h as f64 * 0.3);
    let frame = move |bx: f64, by: f64, dx: f64, dy: f64| {
        gray(w, h, fmt, move |x, y| {
            let mut ex = x - (c.0 + dx);
            while ex < -wf / 2.0 {
                ex += wf;
            }
            while ex >= wf / 2.0 {
                ex -= wf;
            }
            let ey = y - (c.1 + dy);
            if ex * ex + ey * ey < r * r {
                128.0 + 60.0 * ((ex * 0.31 + ey * 0.17).sin() * (ey * 0.23 - ex * 0.11).cos())
            } else {
                periodic(x - bx, y - by, wf, 21.0)
            }
        })
    };
    let x = frame(0.0, 0.0, 0.0, 0.0);
    let y = frame(0.7, -0.3, 2.5, 1.0);
    (x, y, disc_on_cylinder(w, h, c, r), disc_on_cylinder(w, h, (c.0 + 2.5, c.1 + 1.0), r))
}

/// A user displacement of a few points per level (rule G2's form: `d` and `w` per pixel), two of
/// them on the seam columns of every level.
fn seam_points(geo: &FlowGeometry) -> UserDisplacement {
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

#[test]
fn rotating_both_frames_rotates_the_circular_model_solve_bit_for_bit() {
    // the primary acceptance property, through the model driver (see its doc): plain, with mask
    // pyramids, and with mask pyramids and a user displacement, every setting, half and quarter
    // width rotations of both frames (and of the mattes and the points)
    let (w, h) = (256usize, 128usize);
    let mut n = 0;
    for fmt in [Fmt::U8, Fmt::F32] {
        let (x, y, mx, my) = seam_scene(w, h, fmt);
        for (q, hiq) in all_qualities() {
            let s = wrap_of(FlowSettings::new(q, hiq, 1.0));
            let (px, py) = (build_pyramid(&x, &s).unwrap(), build_pyramid(&y, &s).unwrap());
            let (kx, ky) = (MaskPyramid::build(&mx, &s).unwrap(), MaskPyramid::build(&my, &s).unwrap());
            let geo = px.geometry().clone();
            let user = seam_points(&geo);
            let base = [
                circular_model_solve(&px, &py, &s, None, None, true),
                circular_model_solve(&px, &py, &s, Some((&kx, &ky)), None, true),
                circular_model_solve(&px, &py, &s, Some((&kx, &ky)), Some(&user), true),
            ];
            // the flows are solves: the background moves about (0.7, -0.3) / 2^R, the points are set
            let k = (1u32 << geo.reduction) as f32;
            let bg = base[0].at(geo.width / 2, geo.height / 2);
            assert!((bg[0] * k - 0.7).abs() < 0.15 && (bg[1] * k + 0.3).abs() < 0.15, "q {q} hiq {hiq}: {bg:?}");
            assert_eq!(base[2].at(0, geo.height / 2), [-1.5, 0.25], "q {q} hiq {hiq}");
            for shift in [w / 2, w / 4] {
                let (rx, ry) = (rot_image(&x, shift), rot_image(&y, shift));
                let (qx, qy) = (build_pyramid(&rx, &s).unwrap(), build_pyramid(&ry, &s).unwrap());
                let (jx, jy) = (MaskPyramid::build(&rot_matte(&mx, shift), &s).unwrap(), MaskPyramid::build(&rot_matte(&my, shift), &s).unwrap());
                let ruser = rot_user(&user, &geo, shift);
                let got = [
                    circular_model_solve(&qx, &qy, &s, None, None, true),
                    circular_model_solve(&qx, &qy, &s, Some((&jx, &jy)), None, true),
                    circular_model_solve(&qx, &qy, &s, Some((&jx, &jy)), Some(&ruser), true),
                ];
                let s0 = shift >> geo.reduction;
                for (k, name) in ["plain", "masks", "masks and points"].iter().enumerate() {
                    let want = rot(&base[k].data, geo.width, 2, s0);
                    assert_eq!(bits(&got[k].data), bits(&want), "{fmt:?} q {q} hiq {hiq} rotation {shift}: {name}");
                    n += 1;
                }
            }
        }
    }
    assert_eq!(n, 2 * 10 * 2 * 3);
}

/// Largest and mean `|a - b|` over both channels.
fn max_mean_diff(a: &[f32], b: &[f32]) -> (f64, f64) {
    let mut m = 0.0f64;
    let mut t = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as f64 - *y as f64).abs();
        m = m.max(d);
        t += d;
    }
    (m, t / a.len() as f64)
}

#[test]
#[ignore = "the G6 wrap rule (U) this test encodes is superseded by the 360 rules read from TwixtorPro_64 (parity/lab/re/op-residual.md, finding 6)"]
fn the_engine_wrap_solve_rotates_bit_for_bit_with_the_periodic_pyramids() {
    // The engine's wrap driver is periodic (circular Gaussians, the period keeping
    // upsample, split target columns, ring sweeps), so on the periodic pyramids a rotated solve is
    // the rotated flow bit for bit, where the same driver on the pyramids as they stood and the
    // plain solve are not. The model with a single precision target column `x - u` (the B step
    // before the split target columns) stays within 1e-4 px of the rotation (the rounding alone).
    let (w, h) = (256usize, 128usize);
    let (x, y, mx, my) = seam_scene(w, h, Fmt::F32);
    let shift = w / 2;
    let (rx, ry) = (rot_image(&x, shift), rot_image(&y, shift));
    for (q, hiq) in [(5u8, true), (4, true), (3, true), (1, true), (4, false)] {
        for skip_refine in [false, true] {
            let s = FlowSettings { skip_refine, ..wrap_of(FlowSettings::new(q, hiq, 1.0)) };
            let geo = s.geometry(w, h).unwrap();
            let s0 = shift >> geo.reduction;
            let dev = |a: &FlowField, b: &FlowField| max_mean_diff(&b.data, &rot(&a.data, geo.width, 2, s0));
            let (px, py) = (build_pyramid(&x, &s).unwrap(), build_pyramid(&y, &s).unwrap());
            let (qx, qy) = (build_pyramid(&rx, &s).unwrap(), build_pyramid(&ry, &s).unwrap());
            let (dm, da) = dev(&solve_pyramids(&px, &py, &s).unwrap(), &solve_pyramids(&qx, &qy, &s).unwrap());
            let (ox, oy) = (frozen_build_pyramid(&x, &s), frozen_build_pyramid(&y, &s));
            let (orx, ory) = (frozen_build_pyramid(&rx, &s), frozen_build_pyramid(&ry, &s));
            let (om, oa) = dev(&solve_pyramids(&ox, &oy, &s).unwrap(), &solve_pyramids(&orx, &ory, &s).unwrap());
            let p = FlowSettings { wrap: false, ..s };
            let (pm, pa) = dev(&solve(&x, &y, &p).unwrap(), &solve(&rx, &ry, &p).unwrap());
            eprintln!("q {q} hiq {hiq} skip {skip_refine}: wrap max {dm:.4} mean {da:.5}; old pyramids max {om:.4} mean {oa:.5}; plain max {pm:.4} mean {pa:.5}");
            assert!(dm == 0.0 && da == 0.0 && oa > 0.0 && pa > 0.0, "q {q} hiq {hiq} skip {skip_refine}: wrap {dm} {da}, old {oa}, plain {pa}");
        }
        // the model with the engine's rounded B step
        let s = wrap_of(FlowSettings::new(q, hiq, 1.0));
        let geo = s.geometry(w, h).unwrap();
        let s0 = shift >> geo.reduction;
        let (px, py) = (build_pyramid(&x, &s).unwrap(), build_pyramid(&y, &s).unwrap());
        let (qx, qy) = (build_pyramid(&rx, &s).unwrap(), build_pyramid(&ry, &s).unwrap());
        let (kx, ky) = (MaskPyramid::build(&mx, &s).unwrap(), MaskPyramid::build(&my, &s).unwrap());
        let (jx, jy) = (MaskPyramid::build(&rot_matte(&mx, shift), &s).unwrap(), MaskPyramid::build(&rot_matte(&my, shift), &s).unwrap());
        for (a, b) in [(None, None), (Some((&kx, &ky)), Some((&jx, &jy)))] {
            let base = circular_model_solve(&px, &py, &s, a, None, false);
            let got = circular_model_solve(&qx, &qy, &s, b, None, false);
            let (dm, _) = max_mean_diff(&got.data, &rot(&base.data, geo.width, 2, s0));
            eprintln!("q {q} hiq {hiq} masks {}: model with the rounded B step max {dm:e}", a.is_some());
            assert!(dm < 1e-4, "q {q} hiq {hiq} masks {}: {dm}", a.is_some());
        }
    }
}

// ---------------------------------------------------------------------------------------
// Accuracy on a periodic texture

/// Mean distance of a column's vectors to `v`, rows 8 .. h - 8.
fn column_error(f: &FlowField, c: usize, v: (f64, f64)) -> f64 {
    let mut e = 0.0f64;
    for y in 8..f.height - 8 {
        let u = f.at(c, y);
        e += ((u[0] as f64 - v.0).powi(2) + (u[1] as f64 - v.1).powi(2)).sqrt();
    }
    e / (f.height - 16) as f64
}

fn worst(f: &FlowField, cols: impl Iterator<Item = usize>, v: (f64, f64)) -> f64 {
    cols.map(|c| column_error(f, c, v)).fold(0.0f64, f64::max)
}

#[test]
#[ignore = "the G6 wrap rule (U) this test encodes is superseded by the 360 rules read from TwixtorPro_64 (parity/lab/re/op-residual.md, finding 6)"]
fn a_periodic_shift_is_recovered_in_every_column_with_wrap() {
    // whole cycles across a 256 wide frame, shifted by sub pixel amounts (10.6 and -6.2 carry the
    // texture across the seam, 2.6 is vertical); errors in output pixels, the first and last 8
    // columns of the flow against the rest
    let (w, h) = (256usize, 96usize);
    let wf = w as f64;
    for fmt in [Fmt::U8, Fmt::F32] {
        for (dx, dy) in [(3.4f64, 0.0f64), (10.6, 0.0), (-6.2, 0.0), (0.0, 2.6)] {
            let x = gray(w, h, fmt, |xx, yy| periodic(xx, yy, wf, 21.0));
            let y = gray(w, h, fmt, move |xx, yy| periodic(xx - dx, yy - dy, wf, 21.0));
            // Best (half resolution, the default) and full resolution
            for (q, tol) in [(4u8, 0.05f64), (5, 0.06)] {
                let s = FlowSettings::new(q, true, 1.0);
                let k = (1u32 << s.geometry(w, h).unwrap().reduction) as f64;
                let v = (dx / k, dy / k);
                let fw = solve(&x, &y, &wrap_of(s)).unwrap();
                let fp = solve(&x, &y, &s).unwrap();
                // the wrap kernels on the pyramids as they stood (not periodic at the seam)
                let fo = solve_pyramids(&frozen_build_pyramid(&x, &s), &frozen_build_pyramid(&y, &s), &wrap_of(s)).unwrap();
                let fw0 = fw.width;
                let seam = |f: &FlowField| worst(f, (0..8).chain(fw0 - 8..fw0), v) * k;
                let (wl, wr, wm) = (worst(&fw, 0..8, v) * k, worst(&fw, fw0 - 8..fw0, v) * k, worst(&fw, 8..fw0 - 8, v) * k);
                let (ws, ps, os) = (wl.max(wr), seam(&fp), seam(&fo));
                eprintln!("{fmt:?} q {q} ({dx}, {dy}): wrap left {wl:.4} middle {wm:.4} right {wr:.4}; plain seam {ps:.4}; old pyramids seam {os:.4}");
                assert!(ws < tol && wm < tol, "{fmt:?} q {q} ({dx}, {dy}): wrap {wl} {wm} {wr}");
                // the seam columns are about as good as the rest (the driver is periodic, so they
                // are columns like any other), where the plain solve loses tenths of a pixel there
                assert!(ws < wm + 0.02, "{fmt:?} q {q} ({dx}, {dy}): seam {ws} middle {wm}");
                // the plain solve is worse at the seam; for the shifts across it, so were the wrap
                // kernels on the pyramids as they stood
                let factor = if dx != 0.0 { 4.0 } else { 2.0 };
                assert!(ps > factor * ws, "{fmt:?} q {q} ({dx}, {dy}): plain seam {ps} against wrap {ws}");
                if dx != 0.0 {
                    assert!(os > 4.0 * ws, "{fmt:?} q {q} ({dx}, {dy}): old pyramids seam {os} against wrap {ws}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// Hand values, seam columns and threads

/// The circular gradient of the wrap edge enhance, written out (`prep`'s own is private and has
/// its hand values there): the interior chain of `calc_grad` with the columns modulo the width,
/// `gy` one sided at the top and bottom rows.
fn edge_grad_model(img: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; w * h * 2];
    for y in 0..h {
        for x in 0..w {
            let c = img[y * w + x];
            let (l, r) = (img[y * w + (x + w - 1) % w], img[y * w + (x + 1) % w]);
            g[(y * w + x) * 2] = (((c - l) + 0.0f32) + (r - c)) / (1.0f32 + 1.0f32);
            let (mut ny, mut dy) = (0.0f32, 0.0f32);
            if y >= 1 {
                ny = (c - img[(y - 1) * w + x]) + 0.0f32;
                dy = 1.0;
            }
            if y + 1 < h {
                ny = ny + (img[(y + 1) * w + x] - c);
                dy = dy + 1.0;
            }
            g[(y * w + x) * 2 + 1] = ny / dy;
        }
    }
    g
}

#[test]
#[ignore = "the G6 wrap rule (U) this test encodes is superseded by the 360 rules read from TwixtorPro_64 (parity/lab/re/op-residual.md, finding 6)"]
fn the_wrap_transforms_and_reduction_equal_their_definitions_and_the_plain_ones_inside() {
    let (w, h) = (40usize, 22usize);
    let mut r = rng(11);
    let p = Plane { width: w, height: h, data: (0..w * h).map(|_| (r() * 255.0) as f32).collect() };
    // edge enhance: the wrapped Gaussian, the circular gradient, the per pixel map
    let mut got = p.clone();
    prep::edge_enhance_with(&mut got, true);
    let mut t = p.data.clone();
    conv::blur_gaussian_wrapx_above_first(&mut t, w, h, 1, 2.0);
    let g = edge_grad_model(&t, w, h);
    for i in 0..w * h {
        let (gx, gy) = (g[i * 2], g[i * 2 + 1]);
        let m = (gx * gx + gy * gy).sqrt();
        let f = if m <= 20.0f32 { m * 0.05f32 } else { 1.0f32 };
        assert_eq!(got.data[i].to_bits(), ((f * 255.0f32 + t[i]) * 0.5f32).to_bits(), "pixel {i}");
    }
    // away from the seam (the Gaussian's 3 taps, then the gradient's neighbour) it is the plain one
    let mut plain = p.clone();
    prep::edge_enhance(&mut plain);
    let mut seam = 0;
    for y in 0..h {
        for x in 0..w {
            let (a, b) = (got.data[y * w + x], plain.data[y * w + x]);
            if (4..w - 4).contains(&x) {
                assert_eq!(a.to_bits(), b.to_bits(), "({x}, {y})");
            } else if a != b {
                seam += 1;
            }
        }
    }
    assert!(seam > h, "{seam}");
    // the reduction: blur5_wrapx then the period keeping resample, per stage
    let got = prep::reduce_with(p.clone(), 2, true);
    let mut t = p.clone();
    for _ in 0..2 {
        conv::blur5_wrapx_above_first(&mut t.data, t.width, t.height);
        let (nw, nh) = (t.width / 2, t.height / 2);
        t = Plane { width: nw, height: nh, data: sample::resample1_wrapx(&t.data, t.width, t.height, nw, nh) };
    }
    assert_eq!((got.width, got.height), (10, 5));
    assert_eq!(bits(&got.data), bits(&t.data));
    // with an even width the resample reads the columns 2x exactly: one stage is blur5_wrapx taken
    // at (2x, y * (h - 1) / (h / 2 - 1)), the right neighbour weighted by a fraction of 0
    let one = prep::reduce_with(p.clone(), 1, true);
    let mut b = p.data.clone();
    conv::blur5_wrapx_above_first(&mut b, w, h);
    let sy = (h as f32 - 1.0) / ((h / 2) as f32 - 1.0);
    for y in 0..h / 2 {
        let yy = (y as f32) * sy;
        let (iy, fy) = if yy as usize >= h - 1 { (h - 2, 1.0f32) } else { (yy as usize, yy - (yy as usize) as f32) };
        for x in 0..w / 2 {
            let (p0, p1) = (b[iy * w + 2 * x], b[(iy + 1) * w + 2 * x]);
            let top = (b[iy * w + (2 * x + 1) % w] - p0) * 0.0f32 + p0;
            let bot = (b[(iy + 1) * w + (2 * x + 1) % w] - p1) * 0.0f32 + p1;
            assert_eq!(one.data[y * (w / 2) + x].to_bits(), ((bot - top) * fy + top).to_bits(), "({x}, {y})");
        }
    }
    // an odd width keeps the period: sx = 41 / 20
    let odd = Plane { width: 41, height: 6, data: (0..41 * 6).map(|i| (i % 41) as f32).collect() };
    let r1 = prep::reduce_with(odd.clone(), 1, true);
    let mut b = odd.data.clone();
    conv::blur5_wrapx_above_first(&mut b, 41, 6);
    assert_eq!((r1.width, r1.height), (20, 3));
    assert_eq!(bits(&r1.data), bits(&sample::resample1_wrapx(&b, 41, 6, 20, 3)));
}

#[test]
#[ignore = "the G6 wrap rule (U) this test encodes is superseded by the 360 rules read from TwixtorPro_64 (parity/lab/re/op-residual.md, finding 6)"]
fn the_wrap_pyramid_differs_from_the_plain_one_only_where_its_passes_differ() {
    // full resolution, no transforms: level 0 is luma - blur5(luma), so it differs only in the
    // three columns of each side that the 5 tap blur reads across the seam; the levels below are
    // resampled with the period kept, so they differ almost everywhere
    let (w, h) = (48usize, 32usize);
    let img = noisy(w, h, Fmt::U16, 77);
    let s = FlowSettings::new(5, true, 0.7);
    let (pp, wp) = (build_pyramid(&img, &s).unwrap(), build_pyramid(&img, &wrap_of(s)).unwrap());
    assert_eq!(pp.geometry, wp.geometry);
    let (a, b) = (&pp.levels[0].data, &wp.levels[0].data);
    let mut seam_diff = 0;
    for y in 0..h {
        for x in 0..w {
            let same = a[y * w + x].to_bits() == b[y * w + x].to_bits();
            if (3..w - 3).contains(&x) {
                assert!(same, "({x}, {y})");
            } else if !same {
                seam_diff += 1;
            }
        }
    }
    assert!(seam_diff > 4 * h, "{seam_diff}");
    let (lw, lh) = pp.geometry.sizes[1];
    let differ = (0..lw * lh).filter(|&i| pp.levels[1].data[i] != wp.levels[1].data[i]).count();
    assert!(differ > lw * lh / 2, "{differ} of {}", lw * lh);
}

#[test]
fn wrap_pyramids_masks_and_transforms_give_identical_bytes_on_1_4_and_16_threads() {
    let (w, h) = (256usize, 128usize);
    let img = noisy(w, h, Fmt::F32, 5);
    let m = disc_on_cylinder(w, h, (2.0, 60.0), 45.0);
    for (q, hiq) in [(5u8, true), (4, true), (1, false)] {
        let mut s = wrap_of(FlowSettings::new(q, hiq, 0.7));
        s.prep_contrast_edge = true;
        s.prep_delinearize = q == 4;
        let levels = |p: &Pyramid| p.levels.iter().map(|l| bits(&l.data)).collect::<Vec<_>>();
        let p1 = levels(&pool(1, || build_pyramid(&img, &s).unwrap()));
        let m1 = pool(1, || MaskPyramid::build(&m, &s).unwrap());
        for n in [4, 16] {
            assert_eq!(levels(&pool(n, || build_pyramid(&img, &s).unwrap())), p1, "q {q} threads {n}");
            assert_eq!(pool(n, || MaskPyramid::build(&m, &s).unwrap()), m1, "q {q} threads {n}");
        }
    }
    let p = Plane { width: 97, height: 61, data: (0..97 * 61).map(|i| ((i * 7919) % 255) as f32).collect() };
    let run = || {
        let mut e = p.clone();
        prep::edge_enhance_with(&mut e, true);
        (bits(&prep::reduce_with(p.clone(), 3, true).data), bits(&e.data))
    };
    let one = pool(1, run);
    for n in [4, 16] {
        assert_eq!(pool(n, run), one, "threads {n}");
    }
}
