//! Integration tests of the optical flow solver through its public API.
//!
//! - synthetic shift recovery at the three depths and at two qualities
//! - determinism and thread count invariance (rayon pools of 1 and 4 threads)
//! - the two flows of `solve_pair` equal two calls of `solve`
//! - differential test against the numpy model of the solver (tests/fixtures, made by
//!   `tests/fixtures/gen_flow_fixtures.py`)
//! - timing of a full HD solve (ignored by default)

use retime_engine::flow::{solve, solve_pair, FlowField, FlowSettings};
use retime_engine::image::{Fmt, Image, Samples};
use std::fmt::Write as _;

/// A 1/f like texture: 40 plane waves with periods from 6 to 300 pixels and amplitude
/// proportional to the square root of the period, values clamped to 0..255 (centre 128).
/// It has content at every pyramid scale, like natural footage. (A texture made of a few
/// narrow band sinusoids aliases at the coarse pyramid levels and defeats the solver for
/// shifts above about 4 pixels, in the numpy model as well.)
fn natural(x: f64, y: f64) -> f64 {
    let mut s = 0x9e3779b9u64;
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

/// Smooth narrow band texture (periods 7 to 53 pixels); only used for small shifts.
fn banded(x: f64, y: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    128.0
        + 40.0 * (tau * (x / 37.0 + y / 53.0)).sin()
        + 30.0 * (tau * (x / 19.0 - y / 29.0) + 1.0).sin()
        + 25.0 * (tau * (x / 11.0 + y / 13.0) + 2.0).sin()
        + 15.0 * (tau * (x / 7.3 - y / 9.1)).sin()
}

/// Gray image (R = G = B = v, alpha = white) of `tex(x - sx, y - sy)` at `fmt`. The 16 bit
/// and float images carry the same values scaled to their white.
fn gray_image_with(tex: fn(f64, f64) -> f64, w: usize, h: usize, fmt: Fmt, sx: f64, sy: f64) -> Image {
    let mut im = Image::new(w, h, fmt);
    for y in 0..h {
        for x in 0..w {
            let v = tex(x as f64 - sx, y as f64 - sy).clamp(0.0, 255.0);
            let i = (y * w + x) * 4;
            match &mut im.samples {
                Samples::U8(s) => {
                    let q = (v + 0.5).floor() as u8;
                    s[i] = 255;
                    s[i + 1] = q;
                    s[i + 2] = q;
                    s[i + 3] = q;
                }
                Samples::U16(s) => {
                    let q = (v * 257.0 + 0.5).floor() as u16;
                    s[i] = 65535;
                    s[i + 1] = q;
                    s[i + 2] = q;
                    s[i + 3] = q;
                }
                Samples::F32(s) => {
                    let q = (v / 255.0) as f32;
                    s[i] = 1.0;
                    s[i + 1] = q;
                    s[i + 2] = q;
                    s[i + 3] = q;
                }
            }
        }
    }
    im
}

fn gray_image(w: usize, h: usize, fmt: Fmt, sx: f64, sy: f64) -> Image {
    gray_image_with(natural, w, h, fmt, sx, sy)
}

/// Mean and worst deviation of the flow inside the central region (margin `m` pixels).
fn stats(f: &FlowField, m: usize, ex: f64, ey: f64) -> (f64, f64, f64) {
    let (mut sx, mut sy, mut n) = (0.0f64, 0.0f64, 0.0f64);
    let mut worst = 0.0f64;
    for y in m..f.height - m {
        for x in m..f.width - m {
            let v = f.at(x, y);
            sx += v[0] as f64;
            sy += v[1] as f64;
            n += 1.0;
            worst = worst.max((v[0] as f64 - ex).abs()).max((v[1] as f64 - ey).abs());
        }
    }
    (sx / n, sy / n, worst)
}

/// Mean error `(dx, dy)` and worst deviation (all in full resolution pixels) of the flow of
/// `A(p) = B(p + shift)` for the given texture, depth and quality.
fn recover(tex: fn(f64, f64) -> f64, fmt: Fmt, quality: u8, shift: (f64, f64)) -> (f64, f64, f64) {
    let (w, h) = (256, 192);
    let a = gray_image_with(tex, w, h, fmt, 0.0, 0.0);
    let b = gray_image_with(tex, w, h, fmt, shift.0, shift.1); // B(p) = A(p - s)
    let s = FlowSettings::new(quality, true, 1.0);
    let f = solve(&a, &b, &s).expect("solve");
    let div = if quality == 5 { 1 } else { 2 }; // quality 4 works at half resolution
    assert_eq!((f.width, f.height), (w / div, h / div));
    let m = if quality == 5 { 32 } else { 16 };
    let d = div as f64;
    let (mx, my, worst) = stats(&f, m, shift.0 / d, shift.1 / d);
    (mx * d - shift.0, my * d - shift.1, worst * d)
}

#[test]
fn recovers_known_shifts_at_the_three_depths() {
    for &fmt in &[Fmt::U8, Fmt::U16, Fmt::F32] {
        for &shift in &[(3.0, 2.0), (-2.5, 1.25), (0.0, -3.0), (6.5, -3.25), (12.0, 7.0), (-16.0, 10.5)] {
            let (ex, ey, worst) = recover(natural, fmt, 5, shift);
            eprintln!("{fmt:?} quality 5 shift {shift:?}: mean error ({ex:.4}, {ey:.4}), worst {worst:.4}");
            assert!(ex.abs() < 0.1 && ey.abs() < 0.1, "{fmt:?} {shift:?} mean error ({ex}, {ey})");
            assert!(worst < 0.5, "{fmt:?} {shift:?} worst deviation {worst}");
        }
    }
}

#[test]
fn recovers_small_shifts_of_a_banded_texture() {
    for &fmt in &[Fmt::U8, Fmt::U16, Fmt::F32] {
        for &shift in &[(3.0, 2.0), (-2.5, 1.25), (0.0, -3.0), (1.5, 0.75)] {
            let (ex, ey, worst) = recover(banded, fmt, 5, shift);
            eprintln!("banded {fmt:?} shift {shift:?}: mean error ({ex:.4}, {ey:.4}), worst {worst:.4}");
            assert!(ex.abs() < 0.02 && ey.abs() < 0.02, "{fmt:?} {shift:?} mean error ({ex}, {ey})");
            assert!(worst < 0.1, "{fmt:?} {shift:?} worst deviation {worst}");
        }
    }
}

#[test]
fn recovers_shifts_at_half_resolution() {
    for &fmt in &[Fmt::U8, Fmt::F32] {
        let (ex, ey, worst) = recover(natural, fmt, 4, (8.0, -4.0));
        eprintln!("{fmt:?} quality 4: mean error ({ex:.4}, {ey:.4}), worst {worst:.4}");
        assert!(ex.abs() < 0.3 && ey.abs() < 0.3, "{fmt:?} mean error ({ex}, {ey})");
    }
}

// ---------------------------------------------------------------------------------------
// Differential test against the numpy model (tests/fixtures/flow_*.bin)

struct Fixture {
    w: usize,
    h: usize,
    sens: f32,
    quality: u8,
    prep: u32,
    x: Image,
    y: Image,
    expected: Vec<f32>,
}

fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn load_fixture(bytes: &[u8]) -> Fixture {
    assert_eq!(rd_u32(bytes, 0), 0x32444c46, "fixture magic");
    let w = rd_u32(bytes, 4) as usize;
    let h = rd_u32(bytes, 8) as usize;
    let sens = f32::from_bits(rd_u32(bytes, 12));
    let quality = rd_u32(bytes, 16) as u8;
    let prep = rd_u32(bytes, 20);
    let n = w * h * 4;
    let mut o = 24;
    let mut x = Image::new(w, h, Fmt::U8);
    let mut y = Image::new(w, h, Fmt::U8);
    if let Samples::U8(s) = &mut x.samples {
        s.copy_from_slice(&bytes[o..o + n]);
    }
    o += n;
    if let Samples::U8(s) = &mut y.samples {
        s.copy_from_slice(&bytes[o..o + n]);
    }
    o += n;
    let count = (bytes.len() - o) / 4;
    let expected: Vec<f32> = (0..count).map(|i| f32::from_bits(rd_u32(bytes, o + i * 4))).collect();
    Fixture { w, h, sens, quality, prep, x, y, expected }
}

/// (max abs difference, mean abs difference) between the engine and the model.
fn compare_to_model(bytes: &[u8]) -> (f64, f64) {
    let fx = load_fixture(bytes);
    let mut s = FlowSettings::new(fx.quality, true, fx.sens);
    s.skip_refine = true; // the model has no block matching
    s.prep_contrast_edge = fx.prep == 1;
    s.prep_delinearize = fx.prep == 2;
    let f = solve(&fx.x, &fx.y, &s).unwrap();
    let r = if fx.quality == 4 { 1 } else { 0 };
    assert_eq!((f.width, f.height), (fx.w >> r, fx.h >> r));
    assert_eq!(f.data.len(), fx.expected.len());
    let (mut max, mut sum) = (0.0f64, 0.0f64);
    for (a, b) in f.data.iter().zip(fx.expected.iter()) {
        let d = (*a as f64 - *b as f64).abs();
        max = max.max(d);
        sum += d;
    }
    (max, sum / f.data.len() as f64)
}

#[test]
fn matches_the_numpy_model_without_block_matching() {
    let mut report = String::new();
    // (name, fixture, tolerance). The model sums in numpy float32 in a different order (no
    // (a + b) * k chain in the interior, no -1.0 sentinel) and uses numpy's power for
    // Delinearize, so only agreement to a small multiple of the float epsilon times the
    // iteration gain is expected (measured about 3e-6 without prep, see the printed report).
    for (name, bytes, tol) in [
        ("flow_a", &include_bytes!("fixtures/flow_a.bin")[..], 1e-4),
        ("flow_b", &include_bytes!("fixtures/flow_b.bin")[..], 1e-4),
        ("flow_c", &include_bytes!("fixtures/flow_c.bin")[..], 1e-4),
        ("flow_d (quality 4, odd size)", &include_bytes!("fixtures/flow_d.bin")[..], 1e-4),
        ("flow_e (edge enhance)", &include_bytes!("fixtures/flow_e.bin")[..], 1e-3),
        ("flow_f (delinearize)", &include_bytes!("fixtures/flow_f.bin")[..], 1e-3),
    ] {
        let (max, mean) = compare_to_model(bytes);
        writeln!(report, "{name}: max abs diff {max:.3e}, mean {mean:.3e}").unwrap();
        assert!(max < tol, "{name}: max abs diff {max}");
    }
    eprintln!("{report}");
}

// ---------------------------------------------------------------------------------------
// Determinism and thread count invariance

fn pool<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(f)
}

fn bits(f: &FlowField) -> Vec<u32> {
    f.data.iter().map(|v| v.to_bits()).collect()
}

fn pair_images(fmt: Fmt) -> (Image, Image) {
    (gray_image(160, 112, fmt, 0.0, 0.0), gray_image(160, 112, fmt, 2.25, -1.5))
}

#[test]
fn same_call_gives_identical_bits_for_one_and_four_threads() {
    for &fmt in &[Fmt::U8, Fmt::U16, Fmt::F32] {
        let (a, b) = pair_images(fmt);
        for &q in &[4u8, 5] {
            let s = FlowSettings::new(q, true, 0.7);
            let f1 = pool(1, || solve(&a, &b, &s).unwrap());
            let f4 = pool(4, || solve(&a, &b, &s).unwrap());
            let f4b = pool(4, || solve(&a, &b, &s).unwrap());
            assert_eq!(bits(&f1), bits(&f4), "{fmt:?} q{q}: 1 thread against 4 threads");
            assert_eq!(bits(&f4), bits(&f4b), "{fmt:?} q{q}: repeated call");
        }
    }
}

#[test]
fn prep_options_are_thread_count_invariant_too() {
    let (a, b) = pair_images(Fmt::U8);
    for (edge, gamma) in [(true, false), (false, true)] {
        let mut s = FlowSettings::new(5, true, 0.8);
        s.prep_contrast_edge = edge;
        s.prep_delinearize = gamma;
        let f1 = pool(1, || solve(&a, &b, &s).unwrap());
        let f3 = pool(3, || solve(&a, &b, &s).unwrap());
        assert_eq!(bits(&f1), bits(&f3), "edge {edge} gamma {gamma}");
    }
}

#[test]
fn solve_pair_equals_two_solves() {
    let (a, b) = pair_images(Fmt::F32);
    let s = FlowSettings::new(4, true, 0.7);
    let p = pool(4, || solve_pair(&a, &b, &s).unwrap());
    let f_ab = pool(1, || solve(&a, &b, &s).unwrap());
    let f_ba = pool(1, || solve(&b, &a, &s).unwrap());
    assert_eq!(bits(&p.f_ab), bits(&f_ab));
    assert_eq!(bits(&p.f_ba), bits(&f_ba));
    // opposite directions: the flow from B to A is about minus the flow from A to B
    let (m_ab, _, _) = stats(&p.f_ab, 12, 0.0, 0.0);
    let (m_ba, _, _) = stats(&p.f_ba, 12, 0.0, 0.0);
    assert!(m_ab * m_ba < 0.0, "{m_ab} {m_ba}");
}

#[test]
fn bad_inputs_are_reported() {
    let a = gray_image(64, 48, Fmt::U8, 0.0, 0.0);
    let b = gray_image(64, 40, Fmt::U8, 0.0, 0.0);
    let s = FlowSettings::new(4, true, 0.7);
    assert!(solve(&a, &b, &s).is_err(), "size mismatch");
    let c = gray_image(64, 48, Fmt::F32, 0.0, 0.0);
    assert!(solve(&a, &c, &s).is_err(), "format mismatch");
    assert!(solve(&a, &a, &FlowSettings::new(0, true, 0.7)).is_err(), "quality 0 is no vectors");
    assert!(solve(&a, &a, &FlowSettings::new(6, true, 0.7)).is_err());
    assert!(solve(&a, &a, &FlowSettings::new(4, true, f32::NAN)).is_err());
    assert!(solve(&a, &a, &FlowSettings::new(4, true, -0.5)).is_err());
    // 8x6 at quality 3 (High, quarter size) would be 2x1
    let t = gray_image(8, 6, Fmt::U8, 0.0, 0.0);
    assert!(solve(&t, &t, &FlowSettings::new(3, true, 0.7)).is_err(), "working size below 2x2");
}

#[test]
fn identical_frames_give_a_zero_flow() {
    let a = gray_image(96, 64, Fmt::U8, 0.0, 0.0);
    let f = solve(&a, &a, &FlowSettings::new(5, true, 1.0)).unwrap();
    assert!(f.data.iter().all(|v| *v == 0.0));
}

#[test]
fn zero_sensitivity_gives_a_zero_flow_even_with_motion() {
    let (a, b) = pair_images(Fmt::U8);
    let f = solve(&a, &b, &FlowSettings::new(5, true, 0.0)).unwrap();
    assert!(f.data.iter().all(|v| *v == 0.0), "start level 0 with damping 0");
}

#[test]
fn draft_quality_halves_the_resolution_once_more() {
    let a = gray_image(128, 96, Fmt::U8, 0.0, 0.0);
    let hi = solve(&a, &a, &FlowSettings::new(4, true, 0.7)).unwrap();
    let draft = solve(&a, &a, &FlowSettings::new(4, false, 0.7)).unwrap();
    assert_eq!((hi.width, hi.height), (64, 48));
    assert_eq!((draft.width, draft.height), (32, 24));
}

#[test]
fn odd_and_tiny_sizes_run_without_panics_and_give_finite_flows() {
    let mut count = 0;
    for &(w, h) in &[(2usize, 2usize), (3, 3), (4, 5), (5, 4), (7, 9), (16, 3), (3, 16), (33, 31), (64, 63), (101, 77)] {
        for &fmt in &[Fmt::U8, Fmt::U16, Fmt::F32] {
            for &q in &[4u8, 5] {
                let a = gray_image(w, h, fmt, 0.0, 0.0);
                let b = gray_image(w, h, fmt, 1.5, -0.5);
                let s = FlowSettings::new(q, true, 1.0);
                match solve(&a, &b, &s) {
                    Ok(f) => {
                        assert!(f.data.iter().all(|v| v.is_finite()), "{w}x{h} {fmt:?} q{q}");
                        count += 1;
                    }
                    Err(e) => {
                        // only a working size below 2x2 may be refused
                        let r = if q == 4 { 1 } else { 0 };
                        assert!((w >> r) < 2 || (h >> r) < 2, "{w}x{h} q{q}: {e}");
                    }
                }
            }
        }
    }
    assert!(count > 40, "{count}");
}

#[test]
fn nan_and_infinite_pixels_do_not_panic() {
    let mut a = gray_image(48, 40, Fmt::F32, 0.0, 0.0);
    let b = gray_image(48, 40, Fmt::F32, 1.0, 1.0);
    if let Samples::F32(s) = &mut a.samples {
        for (i, v) in [(5usize, f32::NAN), (300, f32::INFINITY), (1000, f32::NEG_INFINITY), (1500, -1.0)] {
            s[i] = v;
        }
    }
    for &sens in &[1.0f32, 0.7] {
        let r = solve(&a, &b, &FlowSettings::new(5, true, sens));
        assert!(r.is_ok());
    }
}

/// Timing of a full HD solve at every quality, `cargo test --release -- --ignored --nocapture timing`.
#[test]
#[ignore]
fn timing_full_hd() {
    let (w, h) = (1920, 1080);
    let a = gray_image(w, h, Fmt::U8, 0.0, 0.0);
    let b = gray_image(w, h, Fmt::U8, 6.5, -3.25);
    for q in 1u8..=5 {
        for &threads in &[1usize, 0] {
            let s = FlowSettings::new(q, true, 0.7);
            let t = std::time::Instant::now();
            let f = if threads == 1 { pool(1, || solve(&a, &b, &s).unwrap()) } else { solve(&a, &b, &s).unwrap() };
            let dt = t.elapsed();
            let scale = (w / f.width) as f64;
            let (mx, my, worst) = stats(&f, 16, 6.5 / scale, -3.25 / scale);
            eprintln!(
                "1920x1080 quality {q} ({}x{}) threads {}: {:.0} ms, mean flow ({:.3}, {:.3}) at full res, worst {:.3}",
                f.width,
                f.height,
                if threads == 1 { "1".to_string() } else { format!("{}", rayon::current_num_threads()) },
                dt.as_secs_f64() * 1000.0,
                mx * scale,
                my * scale,
                worst * scale
            );
        }
    }
    // the two flows of a blend
    for &q in &[4u8, 5] {
        let s = FlowSettings::new(q, true, 0.7);
        let t = std::time::Instant::now();
        let p = solve_pair(&a, &b, &s).unwrap();
        eprintln!("1920x1080 quality {q} solve_pair (all threads): {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
        let t = std::time::Instant::now();
        let p1 = pool(1, || solve_pair(&a, &b, &s).unwrap());
        eprintln!("1920x1080 quality {q} solve_pair (1 thread): {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(bits(&p.f_ab), bits(&p1.f_ab));
    }
}
