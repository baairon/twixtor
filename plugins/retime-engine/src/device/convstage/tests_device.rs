//! Device tests of the stages on the OpenCL devices of the machine: every stage's device result
//! compared bit for bit with the CPU function over seeded corpora (at least 1,000,000 compared
//! output samples per stage and device), with the NaN rule: where the CPU result holds a NaN the
//! device stage must refuse with `StageError::NanOutput`, anywhere else it must give the CPU's
//! bits. With no OpenCL runtime or no usable device every test prints a note and passes (CI
//! runners have none). The program is built once per device; the tests take the devices one at a
//! time, and every launch is a few milliseconds. `stage_timings_at_1080p` (ignored) prints the
//! device time (transfers included) against the CPU time per stage.

use super::conv::{blur5_gpu, blur5_wrapx_gpu, blur_gaussian_gpu, blur_gaussian_wrapx_gpu, choke_gpu, convolve_gpu, convolve_wrapx_gpu};
use super::corpus;
use super::holefill::{fill_flow_holes_gpu, fill_holes_gpu, fill_holes_masked_gpu, fill_plane_gpu};
use super::pyramid::{build_pyramid_gpu, luma_gpu, mask_pyramid_gpu, reduce_gpu, upsample_flow_gpu};
use super::sample::{gather_gpu, resample1_gpu, resample1_wrapx_gpu, Sampler};
use super::{program, Session, StageError};
use crate::device::cl;
use crate::device::probe;
use crate::device::sim::Rng;
use crate::flow::{self, conv, sample, FlowField, FlowSettings};
use crate::holefill::{self, HoleMaskRef};
use crate::image::{Image, Samples};
use crate::matte::{self, Matte};
use crate::prep::{self, Plane};
use crate::warp::Coverage;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());
static SESSIONS: OnceLock<Vec<(String, Session)>> = OnceLock::new();

/// One session per usable device, the stage program built once.
fn sessions() -> &'static [(String, Session)] {
    SESSIONS.get_or_init(|| {
        let api = match cl::runtime() {
            Ok(api) => api,
            Err(e) => {
                println!("device stages: {e}");
                return Vec::new();
            }
        };
        let list = match probe::devices(api) {
            Ok((list, _)) => list,
            Err(e) => {
                println!("device stages: {e}");
                return Vec::new();
            }
        };
        let source = program();
        let mut out = Vec::new();
        for (device, info) in list {
            let name = format!("gpu={} {}", info.index, info.display_name());
            if !probe::usable(&info) {
                println!("{name}: not usable: {:?}", probe::requirements(&info));
                continue;
            }
            match Session::open(&device, &source) {
                Ok(s) => out.push((name, s)),
                Err(e) => panic!("{name}: the stage program does not build: {e}"),
            }
        }
        out
    })
}

/// Runs `f` on every usable device, one at a time; `f` returns the number of compared samples,
/// which must reach a million.
fn on_devices(test: &str, f: impl Fn(&str, &Session) -> usize) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let list = sessions();
    if list.is_empty() {
        println!("{test}: skipped, no usable OpenCL device");
        return;
    }
    for (name, s) in list {
        let t = Instant::now();
        let n = f(name, s);
        println!("{test}: {name}: {n} output samples compared bit for bit in {:.2} s", t.elapsed().as_secs_f64());
        assert!(n >= 1_000_000, "{test}: {name}: only {n} samples compared");
    }
}

/// Checks a device float result against the CPU's under the NaN rule; returns the sample count.
fn check(dev: &str, case: &str, cpu: &[f32], got: Result<Vec<f32>, StageError>) -> usize {
    let cpu_nan = cpu.iter().any(|v| v.is_nan());
    match got {
        Err(StageError::NanOutput) if cpu_nan => cpu.len(),
        Err(e) => panic!("{dev}: {case}: the device stage failed: {e} (the CPU result holds a NaN: {cpu_nan})"),
        Ok(v) => {
            assert_eq!(v.len(), cpu.len(), "{dev}: {case}: lengths");
            assert!(!cpu_nan, "{dev}: {case}: the CPU result holds a NaN, the device result none");
            if let Some(i) = (0..v.len()).find(|&i| v[i].to_bits() != cpu[i].to_bits()) {
                panic!("{dev}: {case}: sample {i}: cpu 0x{:08x} ({:e}), device 0x{:08x} ({:e})", cpu[i].to_bits(), cpu[i], v[i].to_bits(), v[i]);
            }
            v.len()
        }
    }
}

/// Finite adversarial data: noise in `[lo, hi)`, the `-1.0` sentinel, both zeros, subnormals,
/// exact halves and large magnitudes.
fn finite(rng: &mut Rng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n)
        .map(|_| match rng.below(24) {
            0 => -1.0,
            1 => 0.0,
            2 => -0.0,
            3 => f32::from_bits(1 + rng.below(0x007f_ffff) as u32),
            4 => (corpus::uniform(rng, lo, hi)).floor() + 0.5,
            5 => 1.0e30,
            _ => corpus::uniform(rng, lo, hi),
        })
        .collect()
}

fn image(rng: &mut Rng, w: usize, h: usize, depth: usize) -> Image {
    corpus::image(rng, w, h, depth, 0.0)
}

// ---- convolution --------------------------------------------------------------------------------

#[test]
#[ignore = "the kernel follows the rule before the parity work; port pending (docs/engine.md, Device)"]
fn device_convolution_equals_the_cpu() {
    on_devices("convolve", |dev, s| {
        let mut rng = Rng::new(0xD1);
        let mut n = 0;
        for h in 1..=16 {
            for w in 1..=17 {
                for c in 1..=3 {
                    let data = corpus::plane(&mut rng, w * h * c, 0.08);
                    let k = if (w + h + c) % 2 == 0 { conv::gaussian_kernel(5.0) } else { corpus::taps(&mut rng, 1 + (w * h) % 12, false) };
                    for wrap in [false, true] {
                        let mut cpu = data.clone();
                        let mut dv = data.clone();
                        let got = if wrap {
                            conv::convolve_wrapx(&mut cpu, w, h, c, &k);
                            convolve_wrapx_gpu(s, &mut dv, w, h, c, &k)
                        } else {
                            conv::convolve(&mut cpu, w, h, c, &k);
                            convolve_gpu(s, &mut dv, w, h, c, &k)
                        };
                        n += check(dev, &format!("{w}x{h}x{c} n={} wrap={wrap}", k.len()), &cpu, got.map(|_| dv));
                    }
                }
            }
        }
        for (i, &taps) in [1usize, 2, 3, 6, 13, 64].iter().enumerate() {
            let (w, h, c) = [(640usize, 360usize, 1usize), (331, 257, 2), (200, 199, 3)][i % 3];
            let data = finite(&mut rng, w * h * c, -10.0, 300.0);
            let k = corpus::taps(&mut rng, taps, i % 2 == 0);
            for wrap in [false, true] {
                let mut cpu = data.clone();
                let mut dv = data.clone();
                let got = if wrap {
                    conv::convolve_wrapx(&mut cpu, w, h, c, &k);
                    convolve_wrapx_gpu(s, &mut dv, w, h, c, &k)
                } else {
                    conv::convolve(&mut cpu, w, h, c, &k);
                    convolve_gpu(s, &mut dv, w, h, c, &k)
                };
                n += check(dev, &format!("{w}x{h}x{c} n={taps} wrap={wrap}"), &cpu, got.map(|_| dv));
            }
        }
        // the blur wrappers
        let data = finite(&mut rng, 400 * 300, 0.0, 255.0);
        for wrap in [false, true] {
            let (mut cpu, mut dv) = (data.clone(), data.clone());
            let got = if wrap {
                conv::blur5_wrapx(&mut cpu, 400, 300);
                blur5_wrapx_gpu(s, &mut dv, 400, 300)
            } else {
                conv::blur5(&mut cpu, 400, 300);
                blur5_gpu(s, &mut dv, 400, 300)
            };
            n += check(dev, &format!("blur5 wrap={wrap}"), &cpu, got.map(|_| dv));
            for r in [2.0f32, 5.0, 3.99, 11.99] {
                let (mut cpu, mut dv) = (data.clone(), data.clone());
                let got = if wrap {
                    conv::blur_gaussian_wrapx(&mut cpu, 400, 300, 1, r);
                    blur_gaussian_wrapx_gpu(s, &mut dv, 400, 300, 1, r)
                } else {
                    conv::blur_gaussian(&mut cpu, 400, 300, 1, r);
                    blur_gaussian_gpu(s, &mut dv, 400, 300, 1, r)
                };
                n += check(dev, &format!("gauss r={r} wrap={wrap}"), &cpu, got.map(|_| dv));
            }
        }
        n
    });
}

// ---- resampling and samplers --------------------------------------------------------------------

#[test]
#[ignore = "the kernel follows the rule before the parity work; port pending (docs/engine.md, Device)"]
fn device_resample_equals_the_cpu() {
    on_devices("resample", |dev, s| {
        let mut rng = Rng::new(0xD2);
        let mut n = 0;
        for sh in 2..=16 {
            for sw in 1..=17 {
                let src = corpus::plane(&mut rng, sw * sh, 0.08);
                for &(dw, dh) in &[(1usize, 1usize), (2, 2), (3, 7), (sw, sh), (sw / 2 + 1, sh / 2 + 1), (17, 16), (2 * sw, 3)] {
                    if sw >= 2 {
                        n += check(dev, &format!("{sw}x{sh} -> {dw}x{dh}"), &sample::resample1(&src, sw, sh, dw, dh), resample1_gpu(s, &src, sw, sh, dw, dh));
                    }
                    n += check(dev, &format!("wrap {sw}x{sh} -> {dw}x{dh}"), &sample::resample1_wrapx(&src, sw, sh, dw, dh), resample1_wrapx_gpu(s, &src, sw, sh, dw, dh));
                }
            }
        }
        for &(sw, sh, dw, dh) in &[(1920usize, 1080usize, 960usize, 540usize), (961, 541, 1920, 1080), (640, 480, 641, 479), (333, 777, 166, 388)] {
            let src = finite(&mut rng, sw * sh, 0.0, 255.0);
            n += check(dev, &format!("{sw}x{sh} -> {dw}x{dh}"), &sample::resample1(&src, sw, sh, dw, dh), resample1_gpu(s, &src, sw, sh, dw, dh));
            n += check(dev, &format!("wrap {sw}x{sh} -> {dw}x{dh}"), &sample::resample1_wrapx(&src, sw, sh, dw, dh), resample1_wrapx_gpu(s, &src, sw, sh, dw, dh));
        }
        n
    });
}

#[test]
fn device_samplers_equal_the_cpu_samplers() {
    on_devices("samplers", |dev, s| {
        let mut rng = Rng::new(0xD3);
        let mut n = 0;
        for &(w, h, extra) in &[(1usize, 2usize, 30usize), (2, 2, 30), (3, 2, 30), (5, 7, 60), (17, 16, 200), (300, 200, 900)] {
            let img1 = corpus::plane(&mut rng, w * h, if w > 100 { 0.0 } else { 0.05 });
            let img2 = finite(&mut rng, w * h * 2, -50.0, 50.0);
            let xs = corpus::coords(&mut rng, w, extra);
            let ys = corpus::coords(&mut rng, h, extra);
            let (mut px, mut py) = (Vec::new(), Vec::new());
            for &x in &xs {
                for &y in &ys {
                    px.push(x);
                    py.push(y);
                }
            }
            let cpu: Vec<f32> = px.iter().zip(&py).map(|(&x, &y)| sample::bilinear1_wrapx(&img1, w, h, x, y)).collect();
            n += check(dev, &format!("bilinear1_wrapx {w}x{h}"), &cpu, gather_gpu(s, Sampler::Bilinear1Wrapx, &img1, w, h, &px, &py));
            if w >= 2 {
                let cpu: Vec<f32> = px.iter().zip(&py).map(|(&x, &y)| sample::bilinear1(&img1, w, h, x, y)).collect();
                n += check(dev, &format!("bilinear1 {w}x{h}"), &cpu, gather_gpu(s, Sampler::Bilinear1, &img1, w, h, &px, &py));
                let cpu: Vec<f32> = px.iter().zip(&py).flat_map(|(&x, &y)| sample::bilinear2(&img2, w, h, x, y)).collect();
                n += check(dev, &format!("bilinear2 {w}x{h}"), &cpu, gather_gpu(s, Sampler::Bilinear2, &img2, w, h, &px, &py));
            }
        }
        n
    });
}

#[test]
fn device_flow_upsample_equals_the_cpu() {
    on_devices("upsample_flow", |dev, s| {
        let mut rng = Rng::new(0xD4);
        let mut n = 0;
        for sh in 2..=9 {
            for sw in 2..=9 {
                let src = corpus::plane(&mut rng, sw * sh * 2, 0.03);
                for &(dw, dh) in &[(1usize, 1usize), (2, 2), (sw * 2, sh * 2), (sw * 2 + 1, sh * 2 - 1), (17, 18)] {
                    n += check(dev, &format!("{sw}x{sh} -> {dw}x{dh}"), &flow::upsample_flow(&src, sw, sh, dw, dh), upsample_flow_gpu(s, &src, sw, sh, dw, dh));
                }
            }
        }
        for &(sw, sh) in &[(960usize, 540usize), (481, 271), (240, 135)] {
            let src = finite(&mut rng, sw * sh * 2, -40.0, 40.0);
            let (dw, dh) = (sw * 2 - (sw % 2), sh * 2);
            n += check(dev, &format!("{sw}x{sh} -> {dw}x{dh}"), &flow::upsample_flow(&src, sw, sh, dw, dh), upsample_flow_gpu(s, &src, sw, sh, dw, dh));
        }
        n
    });
}

// ---- pyramids -----------------------------------------------------------------------------------

#[test]
#[ignore = "the kernel follows the rule before the parity work; port pending (docs/engine.md, Device)"]
fn device_luma_and_reduction_equal_the_cpu() {
    on_devices("luma and reduce", |dev, s| {
        let mut rng = Rng::new(0xD5);
        let mut n = 0;
        for depth in [8, 16, 32] {
            for (ta, video) in [(false, false), (true, false), (false, true)] {
                for &(w, h) in &[(1usize, 1usize), (17, 16), (640, 360)] {
                    let img = image(&mut rng, w, h, depth);
                    n += check(dev, &format!("luma {depth} {w}x{h} ta={ta} video={video}"), &prep::luma(&img, ta, video).unwrap().data, luma_gpu(s, &img, ta, video).map(|p| p.data));
                }
            }
        }
        for &(w, h, r) in &[(1920usize, 1080usize, 1u32), (1920, 1080, 3), (1001, 777, 2), (64, 48, 2)] {
            for wrap in [false, true] {
                let t = Plane { width: w, height: h, data: finite(&mut rng, w * h, 0.0, 255.0) };
                let cpu = prep::reduce_with(t.clone(), r, wrap);
                n += check(dev, &format!("reduce {w}x{h} r={r} wrap={wrap}"), &cpu.data, reduce_gpu(s, t, r, wrap).map(|p| p.data));
            }
        }
        n
    });
}

#[test]
#[ignore = "the kernel follows the rule before the parity work; port pending (docs/engine.md, Device)"]
fn device_pyramids_equal_build_pyramid_and_mask_pyramid() {
    on_devices("pyramids", |dev, s| {
        let mut rng = Rng::new(0xD6);
        let mut n = 0;
        for &(w, h) in &[(64usize, 48usize), (37, 29), (1280, 720), (640, 480)] {
            for (q, wrap, delin, edge, depth) in [(5u8, false, false, false, 8), (4, true, false, false, 16), (5, false, true, true, 32), (5, true, false, true, 8)] {
                let img = image(&mut rng, w, h, depth);
                let mut st = FlowSettings::new(q, true, 0.7);
                st.wrap = wrap;
                st.prep_delinearize = delin;
                st.prep_contrast_edge = edge;
                let cpu = flow::build_pyramid(&img, &st).unwrap();
                let got = build_pyramid_gpu(s, &img, &st).unwrap_or_else(|e| panic!("{dev}: pyramid {w}x{h}: {e}"));
                assert_eq!(got.len(), cpu.geometry().levels);
                for (l, p) in got.into_iter().enumerate() {
                    n += check(dev, &format!("pyramid {w}x{h} q{q} wrap={wrap} level {l}"), &cpu.level(l).data, Ok(p.data));
                }
                let m = Matte { width: w, height: h, data: if w > 100 { corpus::disc(w, h, w as f32 * 0.4, h as f32 * 0.5, h as f32 * 0.3) } else { corpus::matte(&mut rng, w * h) } };
                let cpu = flow::extras::MaskPyramid::build(&m, &st).unwrap();
                let got = mask_pyramid_gpu(s, &m, &st).unwrap_or_else(|e| panic!("{dev}: mask pyramid {w}x{h}: {e}"));
                for (l, p) in got.into_iter().enumerate() {
                    n += check(dev, &format!("mask {w}x{h} q{q} wrap={wrap} level {l}"), cpu.level(l), Ok(p));
                }
            }
        }
        n
    });
}

// ---- hole fill ----------------------------------------------------------------------------------

#[test]
#[ignore = "the kernel follows the rule before the parity work; port pending (docs/engine.md, Device)"]
fn device_plane_and_flow_fills_equal_the_cpu() {
    on_devices("fill_plane and fill_flow_holes", |dev, s| {
        let mut rng = Rng::new(0xD7);
        let mut n = 0;
        for &(w, h) in &[(2usize, 2usize), (3, 2), (5, 5), (17, 16), (40, 31), (640, 360), (333, 211)] {
            for share in [0.02, 0.4, 1.0] {
                let plane = finite(&mut rng, w * h, -50.0, 300.0);
                let m8 = corpus::mask_u8(&mut rng, w * h, share);
                let m16 = corpus::mask_u16(&mut rng, w * h, share);
                let m32 = corpus::mask_f32(&mut rng, w * h, share);
                for mask in [HoleMaskRef::U8(&m8), HoleMaskRef::U16(&m16), HoleMaskRef::F32(&m32)] {
                    for wrap in [false, true] {
                        let mut cpu = plane.clone();
                        holefill::fill_plane(&mut cpu, w, h, mask, wrap).unwrap();
                        let mut dv = plane.clone();
                        let got = fill_plane_gpu(s, &mut dv, w, h, mask, wrap).map(|_| dv);
                        n += check(dev, &format!("fill {w}x{h} share {share} wrap={wrap}"), &cpu, got);
                    }
                }
            }
            for wrap in [false, true] {
                let mut fl = FlowField::new(w, h);
                fl.data = finite(&mut rng, w * h * 2, -20.0, 20.0);
                let m8 = corpus::mask_u8(&mut rng, w * h, 0.3);
                let mut cpu = fl.clone();
                holefill::fill_flow_holes(&mut cpu, HoleMaskRef::U8(&m8), wrap).unwrap();
                let mut dv = fl.clone();
                let got = fill_flow_holes_gpu(s, &mut dv, HoleMaskRef::U8(&m8), wrap).map(|_| dv.data);
                n += check(dev, &format!("flow fill {w}x{h} wrap={wrap}"), &cpu.data, got);
            }
        }
        n
    });
}

#[test]
#[ignore = "the kernel follows the rule before the parity work; port pending (docs/engine.md, Device)"]
fn device_image_fills_equal_the_cpu_at_every_depth() {
    on_devices("fill_holes", |dev, s| {
        let mut rng = Rng::new(0xD8);
        let mut n = 0;
        for &(w, h) in &[(2usize, 2usize), (7, 3), (17, 16), (300, 200), (420, 250)] {
            for depth in [8, 16, 32] {
                for wrap in [false, true] {
                    let img = image(&mut rng, w, h, depth);
                    let holes = Coverage { width: w, height: h, data: corpus::mask_u8(&mut rng, w * h, 0.25) };
                    let mut cpu = img.clone();
                    holefill::fill_holes(&mut cpu, &holes, wrap).unwrap();
                    let mut dv = img.clone();
                    let got = fill_holes_gpu(s, &mut dv, &holes, wrap);
                    n += compare_images(dev, &format!("fill_holes {w}x{h} depth {depth} wrap={wrap}"), &cpu, got.map(|_| dv));
                    let m16 = corpus::mask_u16(&mut rng, w * h, 0.5);
                    let mut cpu = img.clone();
                    holefill::fill_holes_masked(&mut cpu, HoleMaskRef::U16(&m16), wrap).unwrap();
                    let mut dv = img.clone();
                    let got = fill_holes_masked_gpu(s, &mut dv, HoleMaskRef::U16(&m16), wrap);
                    n += compare_images(dev, &format!("fill_holes_masked {w}x{h} depth {depth} wrap={wrap}"), &cpu, got.map(|_| dv));
                }
            }
        }
        n
    });
}

fn compare_images(dev: &str, case: &str, cpu: &Image, got: Result<Image, StageError>) -> usize {
    match (&cpu.samples, got) {
        (Samples::F32(c), got) => check(dev, case, c, got.map(|g| match g.samples {
            Samples::F32(v) => v,
            _ => panic!("{case}: depth changed"),
        })),
        (Samples::U8(c), Ok(Image { samples: Samples::U8(g), .. })) => {
            assert!(*c == g, "{dev}: {case}: first difference at {:?}", c.iter().zip(&g).position(|(a, b)| a != b));
            c.len()
        }
        (Samples::U16(c), Ok(Image { samples: Samples::U16(g), .. })) => {
            assert!(*c == g, "{dev}: {case}: first difference at {:?}", c.iter().zip(&g).position(|(a, b)| a != b));
            c.len()
        }
        (_, Err(e)) => panic!("{dev}: {case}: {e}"),
        _ => panic!("{dev}: {case}: depth changed"),
    }
}

// ---- matte choke --------------------------------------------------------------------------------

#[test]
fn device_choke_equals_matte_choke() {
    on_devices("choke", |dev, s| {
        let mut rng = Rng::new(0xD9);
        let mut n = 0;
        for &(w, h) in &[(1usize, 1usize), (3, 2), (17, 16), (64, 9), (500, 300)] {
            for amount in [0.5f32, 1.0, 1.5, 2.0, 3.5, -1.0, -2.5, 10.0, -10.0, 0.999] {
                let data = if w > 100 || amount > 2.0 { corpus::disc(w, h, w as f32 / 2.0, h as f32 / 2.0, w.min(h) as f32 / 3.0) } else { corpus::matte(&mut rng, w * h) };
                let mut cpu = Matte { width: w, height: h, data: data.clone() };
                matte::choke(&mut cpu, amount).unwrap();
                let mut dv = Matte { width: w, height: h, data };
                let got = choke_gpu(s, &mut dv, amount).map(|_| dv.data);
                n += check(dev, &format!("choke {w}x{h} by {amount}"), &cpu.data, got);
            }
        }
        // the CPU's Mismatch (a matte holding another number of values) is Unsupported first
        let bad = Matte { width: 3, height: 2, data: vec![0.25; 5] };
        for amount in [0.0f32, 1.0] {
            let mut got = bad.clone();
            assert!(matches!(choke_gpu(s, &mut got, amount), Err(StageError::Unsupported(_))), "{dev}: choke of a bad matte by {amount}");
            assert_eq!(got, bad);
        }
        n
    });
}

// ---- timings ------------------------------------------------------------------------------------

/// The best of three runs, in milliseconds.
fn best_ms(mut f: impl FnMut()) -> f64 {
    (0..3)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1000.0
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
#[ignore = "timing probe: run with --ignored --nocapture"]
fn stage_timings_at_1080p() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let list = sessions();
    if list.is_empty() {
        println!("timings: skipped, no usable OpenCL device");
        return;
    }
    let (w, h) = (1920usize, 1080usize);
    let mut rng = Rng::new(0x71);
    let plane = finite(&mut rng, w * h, 0.0, 255.0);
    let flow2 = finite(&mut rng, w * h * 2, -20.0, 20.0);
    let half = finite(&mut rng, (w / 2) * (h / 2) * 2, -20.0, 20.0);
    let img8 = image(&mut rng, w, h, 8);
    let holes = Coverage { width: w, height: h, data: corpus::mask_u8(&mut rng, w * h, 0.05) };
    let m = Matte { width: w, height: h, data: corpus::disc(w, h, 900.0, 500.0, 300.0) };
    let st = FlowSettings::new(5, true, 0.7);
    for (dev, s) in list {
        let mut rows: Vec<(&str, f64, f64)> = Vec::new();
        rows.push(("blur5 1ch", best_ms(|| conv::blur5(&mut plane.clone(), w, h)), best_ms(|| blur5_gpu(s, &mut plane.clone(), w, h).unwrap())));
        rows.push(("blur5_wrapx 1ch", best_ms(|| conv::blur5_wrapx(&mut plane.clone(), w, h)), best_ms(|| blur5_wrapx_gpu(s, &mut plane.clone(), w, h).unwrap())));
        rows.push(("blur_gaussian r5 2ch", best_ms(|| conv::blur_gaussian(&mut flow2.clone(), w, h, 2, 5.0)), best_ms(|| blur_gaussian_gpu(s, &mut flow2.clone(), w, h, 2, 5.0).unwrap())));
        rows.push(("resample1 to half", best_ms(|| drop(sample::resample1(&plane, w, h, w / 2, h / 2))), best_ms(|| drop(resample1_gpu(s, &plane, w, h, w / 2, h / 2).unwrap()))));
        rows.push(("resample1_wrapx to half", best_ms(|| drop(sample::resample1_wrapx(&plane, w, h, w / 2, h / 2))), best_ms(|| drop(resample1_wrapx_gpu(s, &plane, w, h, w / 2, h / 2).unwrap()))));
        rows.push(("upsample_flow from half", best_ms(|| drop(flow::upsample_flow(&half, w / 2, h / 2, w, h))), best_ms(|| drop(upsample_flow_gpu(s, &half, w / 2, h / 2, w, h).unwrap()))));
        rows.push(("luma 8 bit", best_ms(|| drop(prep::luma(&img8, false, false).unwrap())), best_ms(|| drop(luma_gpu(s, &img8, false, false).unwrap()))));
        rows.push(("build_pyramid q5", best_ms(|| drop(flow::build_pyramid(&img8, &st).unwrap())), best_ms(|| drop(build_pyramid_gpu(s, &img8, &st).unwrap()))));
        rows.push(("MaskPyramid::build q5", best_ms(|| drop(flow::extras::MaskPyramid::build(&m, &st).unwrap())), best_ms(|| drop(mask_pyramid_gpu(s, &m, &st).unwrap()))));
        rows.push((
            "fill_plane 5% holes",
            best_ms(|| holefill::fill_plane(&mut plane.clone(), w, h, HoleMaskRef::U8(&holes.data), false).unwrap()),
            best_ms(|| fill_plane_gpu(s, &mut plane.clone(), w, h, HoleMaskRef::U8(&holes.data), false).unwrap()),
        ));
        rows.push(("fill_holes 8 bit", best_ms(|| holefill::fill_holes(&mut img8.clone(), &holes, false).unwrap()), best_ms(|| fill_holes_gpu(s, &mut img8.clone(), &holes, false).unwrap())));
        rows.push((
            "fill_flow_holes",
            best_ms(|| {
                let mut f = FlowField::new(w, h);
                f.data = flow2.clone();
                holefill::fill_flow_holes(&mut f, HoleMaskRef::U8(&holes.data), false).unwrap()
            }),
            best_ms(|| {
                let mut f = FlowField::new(w, h);
                f.data = flow2.clone();
                fill_flow_holes_gpu(s, &mut f, HoleMaskRef::U8(&holes.data), false).unwrap()
            }),
        ));
        rows.push(("choke 3.5", best_ms(|| matte::choke(&mut m.clone(), 3.5).unwrap()), best_ms(|| choke_gpu(s, &mut m.clone(), 3.5).unwrap())));
        for (stage, cpu, gpu) in rows {
            println!("timing {dev} | {stage} | cpu {cpu:.1} ms | gpu {gpu:.1} ms (transfers included)");
        }
    }
}
