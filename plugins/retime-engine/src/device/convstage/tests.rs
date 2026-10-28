//! Device-free tests of the stages: the very orchestration the device runs, on the simulator
//! backend (every kernel's Rust transcription, shuffled work item orders, write-once outputs),
//! compared bit for bit (NaN payloads included, the simulator runs on the CPU's arithmetic) with
//! the CPU functions on random and adversarial inputs.

use super::conv::{blur_gaussian_host, choke_host, convolve_host, convolve_on};
use super::corpus;
use super::holefill::{fill_flow_holes_host, fill_holes_host, fill_holes_masked_host, fill_plane_host};
use super::pyramid::{build_pyramid_host, luma_host, mask_pyramid_host, reduce_host, upsample_flow_host};
use super::sample::{gather_host, resample1_host, Sampler};
use super::sim::SimBackend;
use super::{program, program_parts, Backend, StageError, Taps};
use crate::device::kernels::{self, conv as kconv, pyramid as kpm, SOURCES};
use crate::device::sim::{self, Rng};
use crate::flow::{self, conv, sample, FlowField, FlowSettings};
use crate::holefill::{self, HoleMaskRef};
use crate::image::Image;
use crate::matte::{self, Matte};
use crate::prep::{self, Plane};
use crate::warp::Coverage;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Panics with the first differing element. Floats compare by their bits, except that two NaNs
/// are equal whatever their payloads: the compiler may commute the operands of an add or a
/// multiply, which keeps every value but may pick the other operand's NaN payload, so payloads are
/// not part of any CPU result (and the stages refuse NaN outputs anyway).
fn same(what: &str, cpu: &[f32], got: &[f32]) {
    assert_eq!(cpu.len(), got.len(), "{what}: lengths");
    let differ = |a: f32, b: f32| a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan());
    if let Some(i) = (0..cpu.len()).find(|&i| differ(cpu[i], got[i])) {
        panic!("{what}: element {i}: cpu 0x{:08x} ({:e}), sim 0x{:08x} ({:e})", cpu[i].to_bits(), cpu[i], got[i].to_bits(), got[i]);
    }
}

fn seed_for(case: usize) -> u64 {
    sim::SEEDS[case % sim::SEEDS.len()]
}

/// The convolution through the backend's buffers, NaNs kept (no NaN rule).
fn sim_convolve(seed: u64, data: &[f32], w: usize, h: usize, c: usize, k: &[f32], wrap: bool) -> Vec<f32> {
    let b = SimBackend::new(seed);
    let taps = Taps::upload(&b, k).unwrap();
    let buf = b.upload(data).unwrap();
    let tmp = b.alloc::<f32>(data.len()).unwrap();
    convolve_on(&b, &buf, &tmp, &buf, w, h, c, &taps, wrap).unwrap();
    b.download(&buf).unwrap()
}

fn cpu_convolve(data: &[f32], w: usize, h: usize, c: usize, k: &[f32], wrap: bool) -> Vec<f32> {
    let mut v = data.to_vec();
    if wrap {
        conv::convolve_wrapx(&mut v, w, h, c, k);
    } else {
        conv::convolve(&mut v, w, h, c, k);
    }
    v
}

/// Images compared sample by sample, floats by their bits.
fn same_samples(what: &str, cpu: &Image, got: &Image) {
    use crate::image::Samples;
    assert_eq!((cpu.width, cpu.height), (got.width, got.height), "{what}");
    match (&cpu.samples, &got.samples) {
        (Samples::U8(a), Samples::U8(b)) => assert!(a == b, "{what}"),
        (Samples::U16(a), Samples::U16(b)) => assert!(a == b, "{what}"),
        (Samples::F32(a), Samples::F32(b)) => same(what, a, b),
        _ => panic!("{what}: the depth changed"),
    }
}

// ---- program and sources ------------------------------------------------------------------------

#[test]
fn the_program_parts_are_registered_sources_in_dependency_order() {
    let parts = program_parts();
    for p in &parts {
        assert!(SOURCES.iter().any(|(_, s)| s == p), "a part is not in kernels::SOURCES");
    }
    let text = program();
    assert!(text.starts_with("#pragma OPENCL FP_CONTRACT OFF\n"));
    // every helper is defined before its first use
    for helper in ["cv_wrapx_tap(", "rs_cell(", "rs_wrapx_cell(", "rs_bilinear1(", "rs_bilinear2(", "rs_bilinear1_wrapx(", "cv_interior(", "cv_border(", "cv_clamp01("] {
        let def = text.find(&format!(" {helper}")).unwrap_or_else(|| panic!("{helper} is not defined"));
        let first = text.find(helper).unwrap();
        assert!(def < first || def + 1 == first, "{helper} is used before its definition");
    }
}

#[test]
fn every_identifier_of_these_sources_carries_its_prefix() {
    let types = [concat!("__ker", "nel void "), "int ", "float ", "float2 ", "long ", "uchar ", "ushort ", "double "];
    let mut seen = 0;
    for (name, src) in SOURCES {
        let prefix = match name.split("::").next().unwrap() {
            "conv" => "cv_",
            "resample" => "rs_",
            "pyramid" => "pm_",
            "holefill" => "hf_",
            _ => continue,
        };
        for line in src.lines() {
            let Some(ty) = types.iter().find(|t| line.starts_with(*t)) else { continue };
            if !line.ends_with('{') || !line.contains('(') {
                continue;
            }
            let ident = line[ty.len()..].split('(').next().unwrap().trim();
            assert!(ident.starts_with(prefix), "{name}: `{ident}` lacks the prefix {prefix}");
            seen += 1;
        }
    }
    assert!(seen >= 35, "only {seen} functions found");
}

#[test]
fn the_tap_bound_and_the_constants_match_the_sources() {
    assert_eq!(kconv::CONV.matches(&format!("q < {}", kconv::CV_MAX_TAPS)).count(), 3);
    assert!(kconv::CV_MAX_TAPS <= 64);
    // the hexadecimal literals of the luma and the binarize equal the CPU's decimal constants
    assert_eq!(kpm::LUMA_F32_BITS, [0.3f32.to_bits(), 0.59f32.to_bits(), 0.11f32.to_bits()]);
    assert_eq!(kpm::LUMA_F64_BITS, [0.3f64.to_bits(), 0.59f64.to_bits(), 0.11f64.to_bits()]);
    assert_eq!(kpm::MASK_ON_BITS, 0.95f32.to_bits());
    for (lit, v) in [
        ("0x1.333334p-2f", 0.3f64),
        ("0x1.2e147ap-1f", 0.59f32 as f64),
        ("0x1.c28f5cp-4f", 0.11f32 as f64),
        ("0x1.e66666p-1f", 0.95f32 as f64),
        ("0x1.3333333333333p-2", 0.3f64),
        ("0x1.2e147ae147ae1p-1", 0.59f64),
        ("0x1.c28f5c28f5c29p-4", 0.11f64),
        ("0x1.fep+7f", 255.0),
        ("0x1.b6p+7f", 219.0),
        ("0x1.01p+8f", 257.0),
        ("0x1.fffep+15f", 65535.0),
        ("0x1p+4f", 16.0),
        ("0x1p-1f", 0.5),
    ] {
        let parsed = parse_hex_float(lit);
        let want = if lit.ends_with('f') { v as f32 as f64 } else { v };
        assert_eq!(parsed.to_bits(), want.to_bits(), "{lit}");
        assert!(kernels::program(&program_parts()).contains(lit), "{lit} is not in the sources");
    }
    assert_eq!(parse_hex_float("0x1.333334p-2f") as f32, 0.3f32);
}

/// A C hexadecimal float literal (`0x1.8p+1f`) as a double (exact for these literals).
fn parse_hex_float(s: &str) -> f64 {
    let t = s.trim_end_matches('f').trim_start_matches("0x");
    let (mant, exp) = t.split_once('p').unwrap();
    let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
    let mut v = u64::from_str_radix(int, 16).unwrap() as f64;
    let mut scale = 1.0 / 16.0;
    for ch in frac.chars() {
        v += ch.to_digit(16).unwrap() as f64 * scale;
        scale /= 16.0;
    }
    v * 2f64.powi(exp.parse::<i32>().unwrap())
}

// ---- convolution ------------------------------------------------------------------------------

#[test]
fn convolution_transcriptions_equal_the_cpu_on_every_small_size() {
    let mut rng = Rng::new(0xC0_4E01);
    let mut case = 0usize;
    for h in 1..=16 {
        for w in 1..=17 {
            for c in 1..=3 {
                let data = corpus::plane(&mut rng, w * h * c, 0.15);
                let kernels = [conv::BINOMIAL5.to_vec(), conv::gaussian_kernel(5.0), corpus::taps(&mut rng, 1 + (case % 9), false)];
                for k in &kernels {
                    for wrap in [false, true] {
                        let cpu = cpu_convolve(&data, w, h, c, k, wrap);
                        let got = sim_convolve(seed_for(case), &data, w, h, c, k, wrap);
                        same(&format!("{w}x{h}x{c} n={} wrap={wrap}", k.len()), &cpu, &got);
                        case += 1;
                    }
                }
            }
        }
    }
}

#[test]
fn convolution_transcriptions_equal_the_cpu_for_every_tap_count() {
    let mut rng = Rng::new(0x7A95);
    let mut case = 0usize;
    for n in 1..=kconv::CV_MAX_TAPS {
        for &(w, h) in &[(1usize, 1usize), (2, 3), (7, 5), (17, 16), (n * 2 + 1, 3), (3, n * 2 + 2)] {
            let c = 1 + case % 3;
            let data = corpus::plane(&mut rng, w * h * c, 0.1);
            let k = corpus::taps(&mut rng, n, case % 2 == 0);
            for wrap in [false, true] {
                let cpu = cpu_convolve(&data, w, h, c, &k, wrap);
                let got = sim_convolve(seed_for(case), &data, w, h, c, &k, wrap);
                same(&format!("{w}x{h}x{c} n={n} wrap={wrap}"), &cpu, &got);
            }
            case += 1;
        }
    }
}

#[test]
fn convolution_bytes_do_not_depend_on_the_work_item_order() {
    let mut rng = Rng::new(17);
    let data = corpus::plane(&mut rng, 23 * 19 * 3, 0.2);
    let k = conv::gaussian_kernel(7.0);
    for wrap in [false, true] {
        let got = sim::same_for_every_order(|seed| bits(&sim_convolve(seed, &data, 23, 19, 3, &k, wrap)));
        let got: Vec<f32> = got.into_iter().map(f32::from_bits).collect();
        same(&format!("order wrap={wrap}"), &cpu_convolve(&data, 23, 19, 3, &k, wrap), &got);
    }
}

#[test]
fn the_sentinel_drops_a_real_minus_one_only_in_multi_channel_borders() {
    // a row of -1.0 next to the border: the CPU drops it in 2 and 3 channel images; so must the device
    let (w, h) = (6usize, 1usize);
    for c in 1..=3 {
        let mut data = vec![3.0f32; w * h * c];
        for ch in 0..c {
            data[c + ch] = -1.0;
        }
        let k = conv::gaussian_kernel(2.0);
        for wrap in [false, true] {
            same(&format!("c={c} wrap={wrap}"), &cpu_convolve(&data, w, h, c, &k, wrap), &sim_convolve(1, &data, w, h, c, &k, wrap));
        }
    }
}

#[test]
fn blur_wrappers_follow_the_cpu_including_their_identities_and_refusals() {
    let mut rng = Rng::new(99);
    let b = SimBackend::new(5);
    for &(w, h, c, r) in &[(9usize, 7usize, 1usize, 2.0f32), (16, 3, 2, 5.0), (5, 11, 3, 0.4), (4, 4, 1, 0.0005), (8, 8, 1, 63.4)] {
        let data = corpus::finite(&mut rng, w * h * c, -10.0, 300.0);
        let mut cpu = data.clone();
        conv::blur_gaussian(&mut cpu, w, h, c, r);
        let mut got = data.clone();
        blur_gaussian_host(&b, &mut got, w, h, c, r, false).unwrap();
        same(&format!("gauss {w}x{h}x{c} r={r}"), &cpu, &got);
        let mut cpu = data.clone();
        conv::blur_gaussian_wrapx(&mut cpu, w, h, c, r);
        let mut got = data.clone();
        blur_gaussian_host(&b, &mut got, w, h, c, r, true).unwrap();
        same(&format!("gauss wrap {w}x{h}x{c} r={r}"), &cpu, &got);
    }
    let data = corpus::finite(&mut rng, 13 * 6, 0.0, 255.0);
    let mut cpu = data.clone();
    conv::blur5(&mut cpu, 13, 6);
    let mut got = data.clone();
    convolve_host(&b, &mut got, 13, 6, 1, &conv::BINOMIAL5, false).unwrap();
    same("blur5", &cpu, &got);
    // a radius whose kernel exceeds the device maximum, four channels and a size mismatch are refused
    let mut d = vec![0.0f32; 64];
    assert!(matches!(blur_gaussian_host(&b, &mut d, 8, 8, 1, 64.5, false), Err(StageError::Unsupported(_))));
    assert!(matches!(convolve_host(&b, &mut vec![0.0f32; 16], 2, 2, 4, &[1.0], false), Err(StageError::Unsupported(_))));
    assert!(matches!(convolve_host(&b, &mut vec![0.0f32; 15], 4, 4, 1, &[1.0], false), Err(StageError::Unsupported(_))));
    // a NaN output is refused and the data are left as they were
    let mut nan = vec![1.0f32; 16];
    nan[5] = f32::NAN;
    let keep = bits(&nan);
    assert!(matches!(convolve_host(&b, &mut nan, 4, 4, 1, &conv::BINOMIAL5, false), Err(StageError::NanOutput)));
    assert_eq!(bits(&nan), keep);
    // empty data and an empty kernel do nothing
    convolve_host(&b, &mut [], 0, 3, 1, &[1.0], false).unwrap();
    let mut d = vec![2.0f32; 4];
    convolve_host(&b, &mut d, 2, 2, 1, &[], false).unwrap();
    assert_eq!(d, vec![2.0; 4]);
}

// ---- resampling and samplers --------------------------------------------------------------------

/// The resample through the backend's buffers, NaNs kept.
fn sim_resample(seed: u64, src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize, wrap: bool) -> Vec<f32> {
    let b = SimBackend::new(seed);
    let s = b.upload(src).unwrap();
    let d = b.alloc::<f32>(dw * dh).unwrap();
    super::sample::resample1_on(&b, &s, sw, sh, &d, dw, dh, wrap).unwrap();
    b.download(&d).unwrap()
}

#[test]
fn resample_transcriptions_equal_the_cpu_on_every_small_size() {
    let mut rng = Rng::new(0x5A3);
    let mut case = 0usize;
    for sh in 2..=16 {
        for sw in 1..=17 {
            let src = corpus::plane(&mut rng, sw * sh, 0.1);
            for &(dw, dh) in &[(1usize, 1usize), (1, 5), (2, 2), (3, 7), (sw, sh), (sw / 2 + 1, sh / 2 + 1), (17, 16), (2 * sw, 3)] {
                if sw >= 2 {
                    let cpu = sample::resample1(&src, sw, sh, dw, dh);
                    same(&format!("{sw}x{sh} -> {dw}x{dh}"), &cpu, &sim_resample(seed_for(case), &src, sw, sh, dw, dh, false));
                }
                let cpu = sample::resample1_wrapx(&src, sw, sh, dw, dh);
                same(&format!("wrap {sw}x{sh} -> {dw}x{dh}"), &cpu, &sim_resample(seed_for(case), &src, sw, sh, dw, dh, true));
                case += 1;
            }
        }
    }
}

#[test]
fn resample_wrappers_refuse_what_the_cpu_cannot_index_and_keep_empty_results() {
    let b = SimBackend::new(0);
    assert!(matches!(resample1_host(&b, &[1.0; 3], 1, 3, 2, 2, false), Err(StageError::Unsupported(_))));
    assert!(matches!(resample1_host(&b, &[1.0; 3], 3, 1, 2, 2, true), Err(StageError::Unsupported(_))));
    assert!(resample1_host(&b, &[1.0; 3], 1, 3, 2, 2, true).is_ok());
    assert_eq!(resample1_host(&b, &[1.0; 4], 2, 2, 0, 5, false).unwrap(), sample::resample1(&[1.0; 4], 2, 2, 0, 5));
    // dw = 1 divides by zero: the CPU gives NaN, the wrapper refuses it
    assert!(matches!(resample1_host(&b, &[1.0; 4], 2, 2, 1, 2, false), Err(StageError::NanOutput)));
}

#[test]
fn sampler_helpers_equal_the_cpu_samplers_at_every_adversarial_coordinate() {
    let mut rng = Rng::new(0xB111);
    for &(w, h) in &[(1usize, 2usize), (2, 2), (3, 2), (2, 5), (5, 7), (17, 16)] {
        let img1 = corpus::plane(&mut rng, w * h, 0.1);
        let img2 = corpus::plane(&mut rng, w * h * 2, 0.1);
        let xs = corpus::coords(&mut rng, w, 40);
        let ys = corpus::coords(&mut rng, h, 40);
        let (mut px, mut py) = (Vec::new(), Vec::new());
        for &x in &xs {
            for &y in &ys {
                px.push(x);
                py.push(y);
            }
        }
        let b = SimBackend::new(seed_for(w * 31 + h));
        let gather = |s: Sampler, img: &[f32]| -> Vec<f32> {
            let lanes = if s == Sampler::Bilinear2 { 2 } else { 1 };
            let ib = b.upload(img).unwrap();
            let xb = b.upload(&px).unwrap();
            let yb = b.upload(&py).unwrap();
            let ob = b.alloc::<f32>(px.len() * lanes).unwrap();
            b.launch(super::Launch::Gather { img: &ib, w, h, xs: &xb, ys: &yb, out: &ob, n: px.len(), lanes, wrap: s == Sampler::Bilinear1Wrapx }).unwrap();
            b.download(&ob).unwrap()
        };
        let cpu: Vec<f32> = px.iter().zip(&py).map(|(&x, &y)| sample::bilinear1_wrapx(&img1, w, h, x, y)).collect();
        same(&format!("bilinear1_wrapx {w}x{h}"), &cpu, &gather(Sampler::Bilinear1Wrapx, &img1));
        if w >= 2 {
            let cpu: Vec<f32> = px.iter().zip(&py).map(|(&x, &y)| sample::bilinear1(&img1, w, h, x, y)).collect();
            same(&format!("bilinear1 {w}x{h}"), &cpu, &gather(Sampler::Bilinear1, &img1));
            let cpu: Vec<f32> = px.iter().zip(&py).flat_map(|(&x, &y)| sample::bilinear2(&img2, w, h, x, y)).collect();
            same(&format!("bilinear2 {w}x{h}"), &cpu, &gather(Sampler::Bilinear2, &img2));
        }
    }
    // the host wrapper refuses NaN results and planes the sampler cannot index
    let b = SimBackend::new(0);
    assert!(matches!(gather_host(&b, Sampler::Bilinear1, &[1.0; 4], 2, 2, &[f32::NAN], &[0.0]), Err(StageError::NanOutput)));
    assert!(matches!(gather_host(&b, Sampler::Bilinear1, &[1.0; 2], 1, 2, &[0.0], &[0.0]), Err(StageError::Unsupported(_))));
    assert_eq!(gather_host(&b, Sampler::Bilinear2, &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0], 2, 2, &[0.5], &[0.5]).unwrap(), vec![1.0, 2.0]);
}

#[test]
fn flow_upsample_transcription_equals_the_cpu() {
    let mut rng = Rng::new(0x0F10);
    let mut case = 0usize;
    for sh in 2..=9 {
        for sw in 2..=9 {
            let src = corpus::plane(&mut rng, sw * sh * 2, 0.05);
            for &(dw, dh) in &[(1usize, 1usize), (2, 2), (sw * 2, sh * 2), (sw * 2 + 1, sh * 2 - 1), (17, 18), (sw, sh)] {
                let cpu = flow::upsample_flow(&src, sw, sh, dw, dh);
                let b = SimBackend::new(seed_for(case));
                let s = b.upload(&src).unwrap();
                let d = b.alloc::<f32>(dw * dh * 2).unwrap();
                super::pyramid::upsample_flow_on(&b, &s, sw, sh, &d, dw, dh).unwrap();
                same(&format!("upsample {sw}x{sh} -> {dw}x{dh}"), &cpu, &b.download(&d).unwrap());
                case += 1;
            }
        }
    }
    let b = SimBackend::new(0);
    let src = corpus::finite(&mut rng, 5 * 4 * 2, -3.0, 3.0);
    assert_eq!(bits(&upsample_flow_host(&b, &src, 5, 4, 10, 8).unwrap()), bits(&flow::upsample_flow(&src, 5, 4, 10, 8)));
    assert!(matches!(upsample_flow_host(&b, &[0.0; 4], 1, 2, 4, 4), Err(StageError::Unsupported(_))));
}

// ---- pyramids -----------------------------------------------------------------------------------

#[test]
fn luma_transcriptions_equal_the_cpu_at_every_depth() {
    let mut rng = Rng::new(0x1A);
    let b = SimBackend::new(3);
    // every 8 bit value in every channel position
    let mut px = Vec::new();
    for v in 0..=255u8 {
        px.extend([v, v, 255 - v, v / 3]);
        px.extend([255 - v, v / 2, v, v]);
    }
    let all8 = Image { width: 64, height: 8, samples: crate::image::Samples::U8(px) };
    for (ta, video) in [(false, false), (true, false), (false, true)] {
        let cpu = prep::luma(&all8, ta, video).unwrap();
        same(&format!("u8 all ta={ta} video={video}"), &cpu.data, &luma_host(&b, &all8, ta, video).unwrap().data);
        for depth in [8, 16, 32] {
            for &(w, h) in &[(1usize, 1usize), (17, 16), (33, 9)] {
                let img = corpus::image(&mut rng, w, h, depth, 0.0);
                let cpu = prep::luma(&img, ta, video).unwrap();
                same(&format!("{depth} {w}x{h} ta={ta} video={video}"), &cpu.data, &luma_host(&b, &img, ta, video).unwrap().data);
            }
        }
    }
    assert!(matches!(luma_host(&b, &all8, true, true), Err(StageError::Unsupported(_))));
    let empty = Image { width: 0, height: 3, samples: crate::image::Samples::U8(vec![]) };
    assert_eq!(luma_host(&b, &empty, false, false).unwrap(), prep::luma(&empty, false, false).unwrap());
}

#[test]
fn reduction_transcriptions_equal_the_cpu() {
    let mut rng = Rng::new(0x4ED);
    for &(w, h) in &[(17usize, 16usize), (33, 20), (64, 48), (100, 7), (9, 37)] {
        for r in 1..=2u32 {
            if (w >> r) < 2 || (h >> r) < 2 {
                continue;
            }
            for wrap in [false, true] {
                let t = Plane { width: w, height: h, data: corpus::finite(&mut rng, w * h, 0.0, 255.0) };
                let cpu = prep::reduce_with(t.clone(), r, wrap);
                let got = reduce_host(&SimBackend::new(seed_for(w + r as usize)), &t, r, wrap).unwrap();
                assert_eq!((got.width, got.height), (cpu.width, cpu.height));
                same(&format!("reduce {w}x{h} r={r} wrap={wrap}"), &cpu.data, &got.data);
            }
        }
    }
}

fn settings(quality: u8, hiq: bool, wrap: bool, delin: bool, edge: bool) -> FlowSettings {
    let mut s = FlowSettings::new(quality, hiq, 0.7);
    s.wrap = wrap;
    s.prep_delinearize = delin;
    s.prep_contrast_edge = edge;
    s
}

#[test]
fn pyramid_levels_equal_build_pyramid() {
    let mut rng = Rng::new(0xBA4D);
    let mut case = 0usize;
    for &(w, h) in &[(64usize, 48usize), (37, 29), (120, 68), (16, 16)] {
        for depth in [8, 16, 32] {
            for (q, hiq, wrap, delin, edge) in [(5, true, false, false, false), (4, true, true, false, false), (3, true, false, true, true), (5, true, true, true, true), (4, false, false, false, true)] {
                let img = corpus::image(&mut rng, w, h, depth, 0.0);
                let s = settings(q, hiq, wrap, delin, edge);
                let cpu = flow::build_pyramid(&img, &s).unwrap();
                let got = build_pyramid_host(&SimBackend::new(seed_for(case)), &img, &s).unwrap();
                assert_eq!(got.len(), cpu.geometry().levels);
                for (l, p) in got.iter().enumerate() {
                    assert_eq!((p.width, p.height), (cpu.level(l).width, cpu.level(l).height));
                    same(&format!("{w}x{h} depth {depth} q{q} wrap={wrap} level {l}"), &cpu.level(l).data, &p.data);
                }
                case += 1;
            }
        }
    }
}

#[test]
fn mask_pyramid_levels_equal_mask_pyramid_build() {
    let mut rng = Rng::new(0x3A5C);
    let mut case = 0usize;
    for &(w, h) in &[(64usize, 48usize), (37, 29), (120, 68), (17, 16)] {
        for (q, wrap) in [(5u8, false), (4, true), (3, false), (2, true)] {
            for kind in 0..2 {
                let data = if kind == 0 { corpus::matte(&mut rng, w * h) } else { corpus::disc(w, h, w as f32 * 0.3, h as f32 * 0.6, h as f32 * 0.3) };
                let m = Matte { width: w, height: h, data };
                let s = settings(q, true, wrap, false, false);
                let cpu = flow::extras::MaskPyramid::build(&m, &s).unwrap();
                let got = mask_pyramid_host(&SimBackend::new(seed_for(case)), &m, &s).unwrap();
                assert_eq!(got.len(), cpu.geometry().levels);
                for (l, p) in got.iter().enumerate() {
                    same(&format!("mask {w}x{h} q{q} wrap={wrap} level {l}"), cpu.level(l), p);
                }
                case += 1;
            }
        }
    }
}

// ---- hole fill ----------------------------------------------------------------------------------

#[test]
fn plane_fill_equals_fill_plane_for_every_mask_type() {
    let mut rng = Rng::new(0x401E);
    let mut case = 0usize;
    for &(w, h) in &[(2usize, 2usize), (3, 2), (2, 7), (5, 5), (17, 16), (16, 17), (33, 8), (40, 31)] {
        for share in [0.02, 0.3, 0.9, 1.0] {
            let plane = corpus::finite(&mut rng, w * h, -50.0, 300.0);
            let m8 = corpus::mask_u8(&mut rng, w * h, share);
            let m16 = corpus::mask_u16(&mut rng, w * h, share);
            let m32 = corpus::mask_f32(&mut rng, w * h, share);
            for mask in [HoleMaskRef::U8(&m8), HoleMaskRef::U16(&m16), HoleMaskRef::F32(&m32)] {
                for wrap in [false, true] {
                    let mut cpu = plane.clone();
                    holefill::fill_plane(&mut cpu, w, h, mask, wrap).unwrap();
                    let mut got = plane.clone();
                    fill_plane_host(&SimBackend::new(seed_for(case)), &mut got, w, h, mask, wrap).unwrap();
                    same(&format!("fill {w}x{h} share {share} wrap={wrap} case {case}"), &cpu, &got);
                    case += 1;
                }
            }
        }
    }
    // no hole: the identity, also for a side of 1; a hole and a side of 1: refused as on the CPU
    let b = SimBackend::new(0);
    let mut p = vec![1.5f32; 5];
    fill_plane_host(&b, &mut p, 5, 1, HoleMaskRef::U8(&[0; 5]), false).unwrap();
    assert_eq!(p, vec![1.5; 5]);
    assert!(matches!(fill_plane_host(&b, &mut p, 5, 1, HoleMaskRef::U8(&[0, 255, 0, 0, 0]), false), Err(StageError::Unsupported(_))));
    assert!(matches!(fill_plane_host(&b, &mut p, 2, 2, HoleMaskRef::U8(&[0; 5]), false), Err(StageError::Unsupported(_))));
}

#[test]
fn image_and_flow_fills_equal_the_cpu_at_every_depth() {
    let mut rng = Rng::new(0x1A6E);
    let mut case = 0usize;
    for &(w, h) in &[(2usize, 2usize), (7, 3), (17, 16), (30, 21)] {
        for depth in [8, 16, 32] {
            for wrap in [false, true] {
                let img = corpus::image(&mut rng, w, h, depth, 0.0);
                let holes = Coverage { width: w, height: h, data: corpus::mask_u8(&mut rng, w * h, 0.25) };
                let mut cpu = img.clone();
                holefill::fill_holes(&mut cpu, &holes, wrap).unwrap();
                let mut got = img.clone();
                fill_holes_host(&SimBackend::new(seed_for(case)), &mut got, &holes, wrap).unwrap();
                same_samples(&format!("image fill {w}x{h} depth {depth} wrap={wrap}"), &cpu, &got);
                let m32 = corpus::mask_f32(&mut rng, w * h, 0.4);
                let mut cpu = img.clone();
                holefill::fill_holes_masked(&mut cpu, HoleMaskRef::F32(&m32), wrap).unwrap();
                let mut got = img.clone();
                fill_holes_masked_host(&SimBackend::new(seed_for(case + 1)), &mut got, HoleMaskRef::F32(&m32), wrap).unwrap();
                same_samples(&format!("masked fill {w}x{h} depth {depth} wrap={wrap}"), &cpu, &got);
                case += 2;
            }
        }
        for wrap in [false, true] {
            let mut flow = FlowField::new(w, h);
            flow.data = corpus::finite(&mut rng, w * h * 2, -20.0, 20.0);
            let m16 = corpus::mask_u16(&mut rng, w * h, 0.5);
            let mut cpu = flow.clone();
            holefill::fill_flow_holes(&mut cpu, HoleMaskRef::U16(&m16), wrap).unwrap();
            let mut got = flow.clone();
            fill_flow_holes_host(&SimBackend::new(seed_for(case)), &mut got, HoleMaskRef::U16(&m16), wrap).unwrap();
            same(&format!("flow fill {w}x{h} wrap={wrap}"), &cpu.data, &got.data);
            case += 1;
        }
    }
}

// ---- matte choke --------------------------------------------------------------------------------

#[test]
fn choke_equals_matte_choke() {
    let mut rng = Rng::new(0xC40E);
    let mut case = 0usize;
    for &(w, h) in &[(1usize, 1usize), (3, 2), (17, 16), (40, 30), (64, 9)] {
        for amount in [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.5, -1.0, -2.5, 10.0, -10.0, f32::NAN, f32::INFINITY, 0.999, 1.001] {
            let data = if case % 2 == 0 { corpus::matte(&mut rng, w * h) } else { corpus::disc(w, h, w as f32 / 2.0, h as f32 / 2.0, w.min(h) as f32 / 3.0) };
            let mut cpu = Matte { width: w, height: h, data: data.clone() };
            matte::choke(&mut cpu, amount).unwrap();
            let mut got = Matte { width: w, height: h, data };
            choke_host(&SimBackend::new(seed_for(case)), &mut got, amount).unwrap();
            same(&format!("choke {w}x{h} by {amount}"), &cpu.data, &got.data);
            case += 1;
        }
    }
    let b = SimBackend::new(0);
    let mut m = Matte { width: 4, height: 4, data: vec![0.5; 16] };
    assert!(matches!(choke_host(&b, &mut m, 100.0), Err(StageError::Unsupported(_))));
    assert_eq!(m.data, vec![0.5; 16]);
    // a matte holding another number of values: the CPU's Mismatch for every amount, zero and
    // NaN included, is Unsupported before anything runs, and the matte is left as it is
    for amount in [0.0f32, f32::NAN, 1.0, -2.5] {
        let bad = Matte { width: 3, height: 2, data: vec![0.25; 5] };
        assert!(matches!(matte::choke(&mut bad.clone(), amount), Err(fx_core::FxError::Mismatch(_))), "cpu choke by {amount}");
        let mut got = bad.clone();
        assert!(matches!(choke_host(&b, &mut got, amount), Err(StageError::Unsupported(_))), "device choke by {amount}");
        assert_eq!(got, bad);
    }
}
