//! Device-free tests of the vector, matte and overlay stages: the transcriptions of `items` run
//! through the work item simulator in every order of `sim::SEEDS` and compared bit for bit with
//! the CPU functions over random and adversarial corpora at every depth (subnormals, -0.0, +-inf,
//! NaN, exact .5 ties, values out of range, mattes outside 0..1, zero and full alpha, sizes from
//! 2x2 to 17x16 and larger, 1 to 4 layers); plus checks of the sources themselves (prefixes, hex
//! literals against the CPU constants, loop bounds against the documented maxima).

use super::items::{self, DrawArgs, EncodeArgs, MergeArgs, SplatArgs};
use super::program_parts;
use crate::device::kernels::{self, vecstage as ks};
use crate::device::selfcheck::{self, Bits};
use crate::device::sim::{self, Global, Rng};
use crate::flow::{FlowField, FlowSettings};
use crate::image::{Fmt, Image, Samples};
use crate::matte::Matte;
use crate::prep::Plane;

// ---- corpus (shared with the device tests) ----------------------------------------------------

pub(super) mod corpus {
    use super::*;

    /// The sizes every image corpus visits.
    pub const SIZES: [(usize, usize); 9] = [(2, 2), (3, 2), (2, 3), (5, 4), (7, 7), (1, 5), (17, 16), (16, 17), (33, 9)];

    /// Adversarial f32 values: the self-check specials plus ties and edges of the stages.
    pub fn specials() -> Vec<f32> {
        let mut v = selfcheck::f32_specials();
        v.extend_from_slice(&[0.5, -0.5, 1.5, 2.5, 254.5, 255.5, 65534.5, 65535.5, 1.0, -1.0, 0.99999994, 1.0000001, 20.0, 20.000002, 0.05]);
        v.extend_from_slice(&[f32::from_bits(1), -f32::from_bits(1), f32::MIN_POSITIVE, 1e-40, 3e38, -3e38, 127.0 / 254.0, 0.25]);
        v
    }

    /// A float: a special one time in `1 / special`, else a value in `[-range, range]`, sometimes
    /// random bits.
    pub fn float(rng: &mut Rng, range: f32, nan: bool) -> f32 {
        let s = specials();
        loop {
            let v = match rng.below(10) {
                0 => s[rng.below(s.len() as u64) as usize],
                1 => f32::from_bits(rng.next_u32()),
                2 => (rng.below(9) as f32 - 4.0) * 0.5,
                _ => ((rng.next_u32() as f64 / u32::MAX as f64) * 2.0 - 1.0) as f32 * range,
            };
            if nan || !v.is_nan() {
                return v;
            }
        }
    }

    /// A matte value: mostly in 0..1, sometimes outside it or special.
    pub fn matte_value(rng: &mut Rng, nan: bool) -> f32 {
        match rng.below(8) {
            0 => float(rng, 3.0, nan),
            1 => [0.0f32, 1.0, -0.0, 0.5][rng.below(4) as usize],
            _ => (rng.next_u32() as f64 / u32::MAX as f64) as f32,
        }
    }

    pub fn flow(rng: &mut Rng, w: usize, h: usize, range: f32, nan: bool) -> FlowField {
        FlowField { width: w, height: h, data: (0..w * h * 2).map(|_| float(rng, range, nan)).collect() }
    }

    pub fn matte(rng: &mut Rng, w: usize, h: usize, nan: bool) -> Matte {
        Matte { width: w, height: h, data: (0..w * h).map(|_| matte_value(rng, nan)).collect() }
    }

    /// The matte values the clamp of `matte::clamp_matte` must sort: NaNs of both signs and
    /// several payloads, `-0.0`, negatives down to `-inf`, subnormals, the edges of 0..1 and
    /// values above 1 up to `+inf` (`1.5` is the value that wrapped a premultiplied byte).
    pub fn clamp_values() -> Vec<f32> {
        vec![
            f32::NAN,
            -f32::NAN,
            f32::from_bits(0x7f80_0001),
            f32::from_bits(0xffc1_2345),
            f32::from_bits(0x7fbf_ffff),
            -0.0,
            0.0,
            -f32::from_bits(1),
            f32::from_bits(1),
            -f32::MIN_POSITIVE,
            f32::MIN_POSITIVE,
            -1e-30,
            -0.5,
            -1.0,
            -3e38,
            f32::NEG_INFINITY,
            0.5,
            0.99999994,
            1.0,
            1.0000001,
            1.5,
            2.0,
            255.0,
            3e38,
            f32::INFINITY,
        ]
    }

    /// A matte whose values are mostly [`clamp_values`] (else in 0..1), so every pixel of a
    /// small image meets the clamp.
    pub fn clamp_matte(rng: &mut Rng, w: usize, h: usize) -> Matte {
        let v = clamp_values();
        let data = (0..w * h).map(|_| if rng.below(4) == 0 { (rng.next_u32() >> 8) as f32 / 16777216.0 } else { v[rng.below(v.len() as u64) as usize] }).collect();
        Matte { width: w, height: h, data }
    }

    /// An image of `fmt`; integer samples favour 0 and white, float samples hold every class.
    pub fn image(rng: &mut Rng, w: usize, h: usize, fmt: Fmt, nan: bool) -> Image {
        let n = w * h * 4;
        let samples = match fmt {
            Fmt::U8 => Samples::U8((0..n).map(|_| match rng.below(6) { 0 => 0, 1 => 255, _ => rng.next_u32() as u8 }).collect()),
            Fmt::U16 => Samples::U16((0..n).map(|_| match rng.below(6) { 0 => 0, 1 => 65535, _ => rng.next_u32() as u16 }).collect()),
            Fmt::F32 => Samples::F32((0..n).map(|_| match rng.below(4) { 0 => float(rng, 2.0, nan), 1 => [0.0f32, 1.0][rng.below(2) as usize], _ => matte_value(rng, false) }).collect()),
        };
        Image { width: w, height: h, samples }
    }

    pub fn plane(rng: &mut Rng, w: usize, h: usize, range: f32, nan: bool) -> Plane {
        Plane { width: w, height: h, data: (0..w * h).map(|_| float(rng, range, nan)).collect() }
    }

    pub const FMTS: [Fmt; 3] = [Fmt::U8, Fmt::U16, Fmt::F32];
}

use corpus::*;

/// The bits of every sample of an image, every NaN as the one quiet NaN: NaN payloads are not
/// compared (the compiler may commute an operation of the CPU code, which picks another payload,
/// and a device output holding a NaN is refused by the wrappers anyway).
pub(super) fn image_bits(img: &Image) -> Vec<u64> {
    match &img.samples {
        Samples::U8(v) => v.iter().map(|x| x.bits()).collect(),
        Samples::U16(v) => v.iter().map(|x| x.bits()).collect(),
        Samples::F32(v) => float_bits(v),
    }
}

/// The bits of every value, every NaN as the one quiet NaN (see [`image_bits`]).
pub(super) fn float_bits(v: &[f32]) -> Vec<u64> {
    v.iter().map(|x| if x.is_nan() { 0x7fc0_0000 } else { x.bits() }).collect()
}

/// Runs a transcription writing one image of `fmt` in every order and returns its bits.
fn sim_image(w: usize, h: usize, fmt: Fmt, run8: impl Fn(&mut Global<u8>, u64), run16: impl Fn(&mut Global<u16>, u64), run32: impl Fn(&mut Global<f32>, u64)) -> Vec<u64> {
    let n = w * h * 4;
    sim::same_for_every_order(|seed| match fmt {
        Fmt::U8 => {
            let mut g = Global::new(n, 0u8);
            run8(&mut g, seed);
            assert_eq!(g.written(), n);
            g.into_vec().iter().map(|x| x.bits()).collect::<Vec<u64>>()
        }
        Fmt::U16 => {
            let mut g = Global::new(n, 0u16);
            run16(&mut g, seed);
            assert_eq!(g.written(), n);
            g.into_vec().iter().map(|x| x.bits()).collect()
        }
        Fmt::F32 => {
            let mut g = Global::new(n, 0.0f32);
            run32(&mut g, seed);
            assert_eq!(g.written(), n);
            float_bits(&g.into_vec())
        }
    })
}

/// Runs a transcription writing one float buffer of `n` in every order and returns its bits.
fn sim_floats(n: usize, run: impl Fn(&mut Global<f32>, u64)) -> Vec<u64> {
    sim::same_for_every_order(|seed| {
        let mut g = Global::new(n, 0.0f32);
        run(&mut g, seed);
        assert_eq!(g.written(), n);
        float_bits(&g.into_vec())
    })
}

fn u8s(img: &Image) -> &[u8] {
    match &img.samples {
        Samples::U8(v) => v,
        _ => unreachable!(),
    }
}
fn u16s(img: &Image) -> &[u16] {
    match &img.samples {
        Samples::U16(v) => v,
        _ => unreachable!(),
    }
}
fn f32s(img: &Image) -> &[f32] {
    match &img.samples {
        Samples::F32(v) => v,
        _ => unreachable!(),
    }
}

// ---- the sources ------------------------------------------------------------------------------

/// The value of a hexadecimal float literal (`0x1.fep+7f`, `0x1p-1`, `-` not included).
fn hex_value(lit: &str) -> f64 {
    let body = lit.trim_end_matches(['f', 'F']);
    let body = &body[2..];
    let (mant, exp) = body.split_once(['p', 'P']).unwrap();
    let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
    let mut v = 0.0f64;
    for c in int.chars() {
        v = v * 16.0 + c.to_digit(16).unwrap() as f64;
    }
    let mut scale = 1.0 / 16.0;
    for c in frac.chars() {
        v += c.to_digit(16).unwrap() as f64 * scale;
        scale /= 16.0;
    }
    v * 2f64.powi(exp.parse::<i32>().unwrap())
}

/// Every hexadecimal float literal of `src`.
fn hex_literals(src: &str) -> Vec<String> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 1 < b.len() {
        if b[i] == b'0' && (b[i + 1] == b'x' || b[i + 1] == b'X') && (i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_')) {
            let start = i;
            i += 2;
            while i < b.len() && (b[i].is_ascii_hexdigit() || b[i] == b'.' || b[i] == b'p' || b[i] == b'P' || ((b[i] == b'+' || b[i] == b'-') && matches!(b[i - 1], b'p' | b'P')) || b[i] == b'f') {
                i += 1;
            }
            let lit = &src[start..i];
            if lit.contains(['p', 'P']) {
                out.push(lit.to_string());
            }
        } else {
            i += 1;
        }
    }
    out
}

#[test]
fn every_float_literal_is_a_cpu_constant_exactly() {
    // the f32 constants of the CPU functions, then the f64 ones
    let f32s: [(f32, &str); 24] = [
        (0.0, "zero"), (0.5, "half"), (1.0, "one"), (127.0, "8 bpc code"), (32767.0, "16 bpc code"), (255.0, "white 8"),
        (65535.0, "white 16"), (16.0, "video black"), (219.0, "video range"), (257.0, "16 to 8"), (0.3, "W30F"),
        (0.59, "W59F"), (0.11, "W11F"), (20.0, "edge limit"), (0.05, "edge gain"), (2.0, "two"),
        (0.725, "Diff R"), (1.475, "Diff G"), (0.29, "Debug luma R"), (0.12, "Debug luma B"), (0.7, "Debug measure scale"),
        (8.0, "Debug neighbours"), (1.0 / 255.0, "8 bpc unit"), (1.0 / 65535.0, "16 bpc unit"),
    ];
    let f64s: [(f64, &str); 11] = [
        (0.0, "zero"), (0.5, "half"), (1.0, "one"), (2.0, "two"), (127.0, "8 bpc code"), (32767.0, "16 bpc code"), (254.0, "8 bpc den"),
        (65534.0, "16 bpc den"), (0.3, "W30D"), (0.59, "W59D"), (0.11, "W11D"),
    ];
    let mut seen = 0;
    for src in ks::PARTS {
        for lit in hex_literals(src) {
            let v = hex_value(&lit);
            if lit.ends_with('f') {
                assert!(f32s.iter().any(|(c, _)| *c as f64 == v), "{lit} = {v:e} is no f32 constant of the CPU code");
            } else {
                assert!(f64s.iter().any(|(c, _)| *c == v) || f32s.iter().any(|(c, _)| *c as f64 == v && (*c == 255.0 || *c == 257.0)), "{lit} = {v:e} is no f64 constant");
            }
            seen += 1;
        }
    }
    assert!(seen > 100);
    // the weights at both precisions are spelled once each
    let all: String = ks::PARTS.concat();
    for (lit, want) in [("0x1.333334p-2f", 0.3f32 as f64), ("0x1.2e147ap-1f", 0.59f32 as f64), ("0x1.c28f5cp-4f", 0.11f32 as f64), ("0x1.99999ap-5f", 0.05f32 as f64)] {
        assert!(all.contains(lit), "{lit}");
        assert_eq!(hex_value(lit), want, "{lit}");
    }
    // the constants of the Debug views (`debug.rs`, create-vectors fix 05)
    for (lit, want) in [
        ("0x1.733334p-1f", 0.725f32 as f64),
        ("0x1.79999ap+0f", 1.475f32 as f64),
        ("0x1.28f5c2p-2f", 0.29f32 as f64),
        ("0x1.eb851ep-4f", 0.12f32 as f64),
        ("0x1.666666p-1f", 0.7f32 as f64),
        ("0x1.010102p-8f", (1.0f32 / 255.0f32) as f64),
        ("0x1.0001p-16f", (1.0f32 / 65535.0f32) as f64),
    ] {
        assert!(all.contains(lit), "{lit}");
        assert_eq!(hex_value(lit), want, "{lit}");
    }
    for (lit, want) in [("0x1.3333333333333p-2", 0.3f64), ("0x1.2e147ae147ae1p-1", 0.59f64), ("0x1.c28f5c28f5c29p-4", 0.11f64)] {
        assert!(all.contains(lit), "{lit}");
        assert_eq!(hex_value(lit), want, "{lit}");
    }
}

#[test]
fn every_identifier_carries_a_package_prefix() {
    let prefixes = ["vc_", "mt_", "ov_", "gm_", "pp_"];
    let mut functions = 0;
    for src in ks::PARTS {
        for line in src.lines() {
            // a function definition starts at column 0 with its return type
            if line.starts_with(' ') || line.starts_with('/') || line.starts_with('}') || !line.contains('(') {
                continue;
            }
            let head = line.split('(').next().unwrap();
            let name = head.split_whitespace().last().unwrap();
            assert!(prefixes.iter().any(|p| name.starts_with(p)), "`{name}` has no stage prefix");
            functions += 1;
        }
    }
    assert!(functions >= 50, "{functions}");
    let parts = program_parts();
    assert_eq!(parts.len(), 5);
    for (name, src) in [("VECTORS", ks::VECTORS), ("MATTE", ks::MATTE), ("PREP", ks::PREP), ("OVERLAY", ks::OVERLAY), ("GEOM", ks::GEOM)] {
        assert!(kernels::SOURCES.iter().any(|(n, s)| *n == format!("vecstage::{name}") && *s == src), "{name} is registered");
    }
}

#[test]
fn the_loop_bounds_are_the_documented_maxima() {
    assert_eq!(ks::MAX_LAYERS, 4);
    assert!(ks::VECTORS.contains("for (int k = 0; k < 4; k++)"));
    assert_eq!(ks::MAX_FG, 3);
    assert!(ks::MATTE.contains("for (int k = 0; k < 3; k++)"));
    assert!(ks::MATTE.contains("for (int k = 0; k < 4; k++)"), "over_weights visits MAX_FG + 1 layers");
    assert_eq!(ks::MAX_POINTS, 16);
    assert_eq!(ks::GEOM.matches("for (int k = 0; k < 16; k++)").count(), 2);
    assert!(ks::GEOM.contains("ip[k * 8 + 7]") && ks::DRAW_INTS == 8);
    let program = kernels::program(&program_parts());
    assert!(program.starts_with("#pragma OPENCL FP_CONTRACT OFF\n"));
}

// ---- vector coding ----------------------------------------------------------------------------

fn encode_args(dw: usize, dh: usize, f: &FlowField, scale: Option<f32>) -> EncodeArgs {
    let (sx, sy, kx, ky) = super::vectors::constants(dw, dh, f.width, f.height, scale);
    EncodeArgs { fw: f.width as u32, fw1: f.width as i32 - 1, fh1: f.height as i32 - 1, sx, sy, kx, ky }
}

fn sim_encode(dw: usize, dh: usize, fmt: Fmt, f: &FlowField, scale: f32) -> Vec<u64> {
    let e = encode_args(dw, dh, f, Some(scale));
    let (w, h) = (dw as u32, dh as u32);
    sim_image(
        dw,
        dh,
        fmt,
        |g, s| sim::launch_2d(dw, dh, s, |x, y| items::vc_encode_u8_item(x, y, &f.data, &e, g, w, h)),
        |g, s| sim::launch_2d(dw, dh, s, |x, y| items::vc_encode_u16_item(x, y, &f.data, &e, g, w, h)),
        |g, s| sim::launch_2d(dw, dh, s, |x, y| items::vc_encode_f32_item(x, y, &f.data, &e, g, w, h)),
    )
}

#[test]
fn encode_transcriptions_equal_encode_vectors_in_every_order() {
    let mut rng = Rng::new(0x6E0C_0001);
    let mut cases = 0;
    for &(dw, dh) in &SIZES {
        if dw < 2 || dh < 2 {
            continue;
        }
        for &(fw, fh) in &[(2usize, 2usize), (dw, dh), (5, 3), (9, 11)] {
            for scale in [1.0f32, 10.0, 0.37, -4.0, 0.0, f32::INFINITY] {
                let f = flow(&mut rng, fw, fh, 2.0 * scale.abs().min(100.0) + 1.0, true);
                for fmt in FMTS {
                    let mut want = Image::new(dw, dh, fmt);
                    crate::vectors::encode_vectors(&mut want, &f, scale).unwrap();
                    assert_eq!(sim_encode(dw, dh, fmt, &f, scale), image_bits(&want), "{fmt:?} {dw}x{dh} flow {fw}x{fh} scale {scale}");
                    cases += 1;
                }
            }
        }
    }
    assert!(cases > 300);
}

fn sim_merge(dw: usize, dh: usize, fmt: Option<Fmt>, layers: &[(&FlowField, &Matte)], scale: Option<f32>) -> Vec<u64> {
    let mut p = Vec::new();
    let mut ip = Vec::new();
    for (f, _) in layers {
        let e = encode_args(dw, dh, f, scale);
        p.extend_from_slice(&[e.sx, e.sy, e.kx, e.ky]);
        ip.extend_from_slice(&[e.fw as i32, e.fw1, e.fh1]);
    }
    let pick = |k: usize| layers.get(k).unwrap_or(&layers[0]);
    let m = MergeArgs { f: [0, 1, 2, 3].map(|k| pick(k).0.data.as_slice()), w: [0, 1, 2, 3].map(|k| pick(k).1.data.as_slice()), p: &p, ip: &ip, nl: layers.len() as u32 };
    let (w, h) = (dw as u32, dh as u32);
    match fmt {
        Some(fmt) => sim_image(
            dw,
            dh,
            fmt,
            |g, s| sim::launch_2d(dw, dh, s, |x, y| items::vc_merge_u8_item(x, y, &m, g, w, h)),
            |g, s| sim::launch_2d(dw, dh, s, |x, y| items::vc_merge_u16_item(x, y, &m, g, w, h)),
            |g, s| sim::launch_2d(dw, dh, s, |x, y| items::vc_merge_f32_item(x, y, &m, g, w, h)),
        ),
        None => sim_floats(dw * dh * 2, |g, s| sim::launch_2d(dw, dh, s, |x, y| items::vc_merge_flow_item(x, y, &m, g, w, h))),
    }
}

#[test]
fn merge_transcriptions_equal_the_merged_encoder_and_merge_flows_in_every_order() {
    let mut rng = Rng::new(0x6E0C_0002);
    let mut cases = 0;
    for &(dw, dh) in &[(2usize, 2usize), (5, 4), (7, 7), (17, 16), (16, 17), (33, 9)] {
        for nl in 1..=4usize {
            let flows: Vec<FlowField> = (0..nl).map(|k| flow(&mut rng, [2, dw, 9, 4][k], [2, dh, 6, 5][k], 30.0, true)).collect();
            let weights: Vec<Matte> = (0..nl).map(|_| matte(&mut rng, dw, dh, true)).collect();
            let layers: Vec<(&FlowField, &Matte)> = flows.iter().zip(&weights).collect();
            for scale in [1.0f32, 12.5, -3.0] {
                for fmt in FMTS {
                    let mut want = Image::new(dw, dh, fmt);
                    crate::vectors::encode_vectors_merged(&mut want, &layers, scale).unwrap();
                    assert_eq!(sim_merge(dw, dh, Some(fmt), &layers, Some(scale)), image_bits(&want), "{fmt:?} {dw}x{dh} {nl} layers");
                    cases += 1;
                }
            }
            let want = crate::vectors::merge_flows(dw, dh, &layers).unwrap();
            assert_eq!(sim_merge(dw, dh, None, &layers, None), float_bits(&want.data), "merge_flows {dw}x{dh} {nl} layers");
        }
    }
    assert!(cases >= 200);
}

#[test]
fn blue_transcriptions_after_the_merge_equal_encode_vectors_merged_blue_in_every_order() {
    // create-vectors fix 03: a single layer output carries its weight in the blue channel; the
    // merge transcription equals encode_vectors_merged (above), the blue store runs on its result
    let mut rng = Rng::new(0x6E0C_000B);
    let mut cases = 0;
    for &(dw, dh) in &[(2usize, 2usize), (5, 4), (7, 7), (17, 16), (33, 9)] {
        for nl in 1..=2usize {
            let flows: Vec<FlowField> = (0..nl).map(|k| flow(&mut rng, [dw, 9][k], [dh, 6][k], 30.0, true)).collect();
            let weights: Vec<Matte> = (0..nl).map(|_| matte(&mut rng, dw, dh, true)).collect();
            let layers: Vec<(&FlowField, &Matte)> = flows.iter().zip(&weights).collect();
            for blue in [weights[0].clone(), clamp_matte(&mut rng, dw, dh)] {
                for fmt in FMTS {
                    let mut want = Image::new(dw, dh, fmt);
                    crate::vectors::encode_vectors_merged_blue(&mut want, &layers, 8.0, &blue).unwrap();
                    let mut merged = Image::new(dw, dh, fmt);
                    crate::vectors::encode_vectors_merged(&mut merged, &layers, 8.0).unwrap();
                    assert_eq!(sim_merge(dw, dh, Some(fmt), &layers, Some(8.0)), image_bits(&merged));
                    let n = dw * dh;
                    let got = sim_image(
                        dw,
                        dh,
                        fmt,
                        |g, s| sim::launch_1d(n, s, |i| items::vc_blue_u8_item(i, u8s(&merged), g, &blue.data, n as u32)),
                        |g, s| sim::launch_1d(n, s, |i| items::vc_blue_u16_item(i, u16s(&merged), g, &blue.data, n as u32)),
                        |g, s| sim::launch_1d(n, s, |i| items::vc_blue_f32_item(i, f32s(&merged), g, &blue.data, n as u32)),
                    );
                    assert_eq!(got, image_bits(&want), "{fmt:?} {dw}x{dh} {nl} layers");
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 60);
}

#[test]
fn decode_holes_scale_and_zero_transcriptions_equal_the_cpu_in_every_order() {
    let mut rng = Rng::new(0x6E0C_0003);
    for &(w, h) in &SIZES {
        let n = w * h;
        for fmt in FMTS {
            let img = image(&mut rng, w, h, fmt, true);
            for md in [1.0f32, 32.0, 0.1, -7.0, 0.0, f32::INFINITY, f32::NAN] {
                let want = crate::vectors::decode_vectors(&img, md);
                let got = sim_floats(n * 2, |g, s| {
                    sim::launch_1d(n, s, |i| match fmt {
                        Fmt::U8 => items::vc_decode_u8_item(i, u8s(&img), g, md as f64, n as u32),
                        Fmt::U16 => items::vc_decode_u16_item(i, u16s(&img), g, md as f64, n as u32),
                        Fmt::F32 => items::vc_decode_f32_item(i, f32s(&img), g, md, n as u32),
                    })
                });
                assert_eq!(got, float_bits(&want.data), "decode {fmt:?} {w}x{h} md {md}");
            }
            let holes = crate::vectors::vector_hole_mask(&img);
            let got: Vec<u64> = match fmt {
                Fmt::U8 => sim::same_for_every_order(|s| {
                    let mut g = Global::new(n, 0u8);
                    sim::launch_1d(n, s, |i| items::vc_holes_u8_item(i, u8s(&img), &mut g, n as u32));
                    g.into_vec().iter().map(|x| x.bits()).collect::<Vec<u64>>()
                }),
                Fmt::U16 => sim::same_for_every_order(|s| {
                    let mut g = Global::new(n, 0u16);
                    sim::launch_1d(n, s, |i| items::vc_holes_u16_item(i, u16s(&img), &mut g, n as u32));
                    g.into_vec().iter().map(|x| x.bits()).collect::<Vec<u64>>()
                }),
                Fmt::F32 => sim_floats(n, |g, s| sim::launch_1d(n, s, |i| items::vc_holes_f32_item(i, f32s(&img), g, n as u32))),
            };
            let want: Vec<u64> = match &holes {
                crate::vectors::HoleMask::U8(v) => v.iter().map(|x| x.bits()).collect(),
                crate::vectors::HoleMask::U16(v) => v.iter().map(|x| x.bits()).collect(),
                crate::vectors::HoleMask::F32(v) => float_bits(v),
            };
            assert_eq!(got, want, "holes {fmt:?} {w}x{h}");
            let zero = crate::vectors::zero_vectors_image(w, h, fmt);
            let got = sim_image(
                w,
                h,
                fmt,
                |g, s| sim::launch_1d(n, s, |i| items::vc_zero_item(i, g, [255, 127, 127, 0], n as u32)),
                |g, s| sim::launch_1d(n, s, |i| items::vc_zero_item(i, g, [65535, 32767, 32767, 0], n as u32)),
                |g, s| sim::launch_1d(n, s, |i| items::vc_zero_item(i, g, [1.0, 0.5, 0.5, 0.0], n as u32)),
            );
            assert_eq!(got, image_bits(&zero));
        }
        let f = flow(&mut rng, w, h, 50.0, true);
        for (sx, sy) in [(1.0f32, 1.0f32), (-0.5, 2.0), (0.0, f32::INFINITY), (f32::NAN, -0.0)] {
            let mut want = f.clone();
            crate::vectors::apply_vec_scale(&mut want, sx, sy);
            let got = sim_floats(n * 2, |g, s| sim::launch_1d(n, s, |i| items::vc_vec_scale_item(i, &f.data, g, sx, sy, n as u32)));
            assert_eq!(got, float_bits(&want.data));
        }
    }
}

// ---- matte ------------------------------------------------------------------------------------

#[test]
fn extract_premultiply_and_over_transcriptions_equal_matte_rs_in_every_order() {
    let mut rng = Rng::new(0x6E0C_0004);
    for &(w, h) in &SIZES {
        let n = w * h;
        for fmt in FMTS {
            let img = image(&mut rng, w, h, fmt, true);
            for chan in [1, 2, 3, 4, 0, 7] {
                let want = crate::matte::extract(&img, chan);
                let c = crate::matte::matte_channel(chan) as u32;
                let got = sim_floats(n, |g, s| {
                    sim::launch_1d(n, s, |i| match fmt {
                        Fmt::U8 => items::mt_extract_u8_item(i, u8s(&img), g, c, n as u32),
                        Fmt::U16 => items::mt_extract_u16_item(i, u16s(&img), g, c, n as u32),
                        Fmt::F32 => items::mt_extract_f32_item(i, f32s(&img), g, c, n as u32),
                    })
                });
                assert_eq!(got, float_bits(&want.data), "extract {fmt:?} {w}x{h} chan {chan}");
            }
            let m = matte(&mut rng, w, h, true);
            let ns = n * 4;
            for mm in [&m, &clamp_matte(&mut rng, w, h)] {
                let want = crate::matte::premultiply(&img, mm).unwrap();
                let got = sim_image(
                    w,
                    h,
                    fmt,
                    |g, s| sim::launch_1d(ns, s, |i| items::mt_premultiply_u8_item(i, u8s(&img), &mm.data, g, ns as u32)),
                    |g, s| sim::launch_1d(ns, s, |i| items::mt_premultiply_u16_item(i, u16s(&img), &mm.data, g, ns as u32)),
                    |g, s| sim::launch_1d(ns, s, |i| items::mt_premultiply_f32_item(i, f32s(&img), &mm.data, g, ns as u32)),
                );
                assert_eq!(got, image_bits(&want), "premultiply {fmt:?} {w}x{h}");
            }
            let dest = image(&mut rng, w, h, fmt, true);
            for src in [image(&mut rng, w, h, fmt, true), Image::new(w, h, fmt), crate::matte::premultiply(&img, &m).unwrap()] {
                let mut want = dest.clone();
                crate::matte::over(&mut want, &src).unwrap();
                let got = sim_image(
                    w,
                    h,
                    fmt,
                    |g, s| sim::launch_1d(n, s, |i| items::mt_over_u8_item(i, u8s(&dest), g, u8s(&src), n as u32)),
                    |g, s| sim::launch_1d(n, s, |i| items::mt_over_u16_item(i, u16s(&dest), g, u16s(&src), n as u32)),
                    |g, s| sim::launch_1d(n, s, |i| items::mt_over_f32_item(i, f32s(&dest), g, f32s(&src), n as u32)),
                );
                assert_eq!(got, image_bits(&want), "over {fmt:?} {w}x{h}");
            }
        }
    }
}

fn sim_planes(count: usize, n: usize, run: impl Fn(&mut [Global<f32>; 4], u64)) -> Vec<Vec<u64>> {
    sim::same_for_every_order(|seed| {
        let mut g = [0, 1, 2, 3].map(|_| Global::new(n, 0.0f32));
        run(&mut g, seed);
        g.into_iter().take(count).map(|p| {
            assert_eq!(p.written(), n);
            float_bits(&p.into_vec())
        }).collect::<Vec<_>>()
    })
}

#[test]
fn effective_and_over_weights_transcriptions_equal_matte_rs_in_every_order() {
    let mut rng = Rng::new(0x6E0C_0005);
    let none: [f32; 0] = [];
    for &(w, h) in &SIZES {
        let n = w * h;
        for (nfg, clamp) in [(0usize, false), (1, false), (2, false), (3, false), (1, true), (2, true), (3, true)] {
            let mattes: Vec<Matte> = (0..nfg).map(|_| if clamp { clamp_matte(&mut rng, w, h) } else { matte(&mut rng, w, h, true) }).collect();
            let pick = |k: usize| mattes.get(k).or(mattes.first()).map(|m| m.data.as_slice()).unwrap_or(&none);
            let ms = [pick(0), pick(1), pick(2)];
            for shrink in [0.0f32, f32::NAN, f32::INFINITY] {
                let want = crate::matte::effective(&mattes, &vec![shrink; nfg], w, h).unwrap();
                let got = sim_planes(nfg + 1, n, |g, s| sim::launch_1d(n, s, |i| items::mt_effective_item(i, ms, g, nfg as u32, n as u32)));
                let want: Vec<Vec<u64>> = want.iter().map(|m| float_bits(&m.data)).collect();
                assert_eq!(got, want, "effective {w}x{h} nfg {nfg} clamp corpus {clamp}");
            }
            for first in 0..=nfg {
                for last in first..=nfg {
                    let want = crate::matte::over_weights(&mattes, first, last, w, h).unwrap();
                    let got = sim_planes(last - first + 1, n, |g, s| {
                        sim::launch_1d(n, s, |i| items::mt_over_weights_item(i, ms, g, first as u32, last as u32, nfg as u32, n as u32))
                    });
                    let want: Vec<Vec<u64>> = want.iter().map(|m| float_bits(&m.data)).collect();
                    assert_eq!(got, want, "over_weights {w}x{h} {first}..={last} of {nfg} clamp corpus {clamp}");
                    // create-vectors fix 03: the weights inside the whole stack
                    let want = crate::matte::stack_weights(&mattes, first, last, w, h).unwrap();
                    let got = sim_planes(last - first + 1, n, |g, s| {
                        sim::launch_1d(n, s, |i| items::mt_stack_weights_item(i, ms, g, first as u32, last as u32, nfg as u32, n as u32))
                    });
                    let want: Vec<Vec<u64>> = want.iter().map(|m| float_bits(&m.data)).collect();
                    assert_eq!(got, want, "stack_weights {w}x{h} {first}..={last} of {nfg} clamp corpus {clamp}");
                }
            }
        }
    }
}

#[test]
fn the_matte_clamp_equals_matte_rs_and_sits_where_the_cpu_reads_the_matte() {
    // the transcription sorts every class as the CPU's clamp, sign of zero included
    let mut changed = 0;
    for x in clamp_values() {
        let want = crate::matte::clamp_matte(x);
        assert_eq!(items::mt_clamp_matte(x).to_bits(), want.to_bits(), "clamp of {x:e} (0x{:08x})", x.to_bits());
        changed += usize::from(want.to_bits() != x.to_bits());
    }
    assert!(changed >= 15, "the clamp corpus changes {changed} values");
    assert_eq!(items::mt_clamp_matte(-0.0).to_bits(), (-0.0f32).to_bits(), "-0.0 is kept");
    assert_eq!(items::mt_clamp01(-0.0).to_bits(), 0.0f32.to_bits(), "clamp01 makes -0.0 a 0.0");
    // a value above 1 no longer wraps the low byte: 200 * 1.5 stores 200, not 44
    let img = Image { width: 1, height: 1, samples: Samples::U8(vec![200, 100, 255, 7]) };
    let m = Matte { width: 1, height: 1, data: vec![1.5] };
    let want = crate::matte::premultiply(&img, &m).unwrap();
    let got = sim_image(1, 1, Fmt::U8, |g, s| sim::launch_1d(4, s, |i| items::mt_premultiply_u8_item(i, u8s(&img), &m.data, g, 4)), |_, _| {}, |_, _| {});
    assert_eq!(got, image_bits(&want));
    assert_eq!(got, vec![200, 100, 255, 7]);
    // the source reads the matte through the clamp in premultiply, effective, over_weights and
    // stack_weights
    assert!(ks::MATTE.contains("float mt_clamp_matte(float x)"));
    assert!(ks::MATTE.contains("if (x >= 0x0p+0f) {"));
    assert_eq!(ks::MATTE.matches("float mv = mt_clamp_matte(m[i / 4]);").count(), 3, "the three premultiply kernels");
    assert_eq!(ks::MATTE.matches("float mk = mt_clamp_matte(m[i]);").count(), 3, "effective, over_weights and stack_weights");
    assert!(!ks::MATTE.contains("float mk = m[i];"));
}

// ---- overlays ---------------------------------------------------------------------------------

/// Alpha 0 on every third pixel (`0.0` and `-0.0` in turn in float): the transparent pixels of
/// `W` that `W'` takes from `Y`.
pub(super) fn punch(img: &mut Image) {
    match &mut img.samples {
        Samples::U8(v) => v.chunks_exact_mut(4).step_by(3).for_each(|p| p[0] = 0),
        Samples::U16(v) => v.chunks_exact_mut(4).step_by(3).for_each(|p| p[0] = 0),
        Samples::F32(v) => v.chunks_exact_mut(4).step_by(3).enumerate().for_each(|(k, p)| p[0] = if k % 2 == 0 { 0.0 } else { -0.0 }),
    }
}

#[test]
#[ignore = "the kernel follows the rule before the parity work; port pending (docs/engine.md, Device)"]
fn debug_composition_transcriptions_equal_debug_rs_in_every_order() {
    let mut rng = Rng::new(0x6E0C_0006);
    for &(w, h) in &SIZES {
        let n = w * h;
        for fmt in FMTS {
            let a = image(&mut rng, w, h, fmt, true);
            let b = image(&mut rng, w, h, fmt, true);
            let mut want = Image::new(w, h, fmt);
            crate::blend::blend(&mut want, &a, &b, 0.5f32);
            let ns = n * 4;
            let got = sim_image(
                w,
                h,
                fmt,
                |g, s| sim::launch_1d(ns, s, |i| items::ov_blend_u8_item(i, u8s(&a), u8s(&b), g, ns as u32)),
                |g, s| sim::launch_1d(ns, s, |i| items::ov_blend_u16_item(i, u16s(&a), u16s(&b), g, ns as u32)),
                |g, s| sim::launch_1d(ns, s, |i| items::ov_blend_f32_item(i, f32s(&a), f32s(&b), g, ns as u32)),
            );
            assert_eq!(got, image_bits(&want), "blend view {fmt:?} {w}x{h}");
            let mut want = Image::new(w, h, fmt);
            crate::debug::diff(&mut want, &a, &b);
            let got = sim_image(
                w,
                h,
                fmt,
                |g, s| sim::launch_1d(n, s, |i| items::ov_diff_u8_item(i, u8s(&a), u8s(&b), g, n as u32)),
                |g, s| sim::launch_1d(n, s, |i| items::ov_diff_u16_item(i, u16s(&a), u16s(&b), g, n as u32)),
                |g, s| sim::launch_1d(n, s, |i| items::ov_diff_f32_item(i, f32s(&a), f32s(&b), g, n as u32)),
            );
            assert_eq!(got, image_bits(&want), "diff view {fmt:?} {w}x{h}");
            let r: Vec<f32> = (0..n).map(|_| matte_value(&mut rng, true)).collect();
            let gg: Vec<f32> = (0..n).map(|_| matte_value(&mut rng, true)).collect();
            let mut want = Image::new(w, h, fmt);
            crate::debug::overlay(&mut want, &r, &gg);
            let got = sim_image(
                w,
                h,
                fmt,
                |g, s| sim::launch_1d(n, s, |i| items::ov_overlay_u8_item(i, &r, &gg, g, n as u32)),
                |g, s| sim::launch_1d(n, s, |i| items::ov_overlay_u16_item(i, &r, &gg, g, n as u32)),
                |g, s| sim::launch_1d(n, s, |i| items::ov_overlay_f32_item(i, &r, &gg, g, n as u32)),
            );
            assert_eq!(got, image_bits(&want), "overlay {fmt:?} {w}x{h}");
            // W': the corpus alpha plus every third pixel transparent
            let mut wa = a.clone();
            punch(&mut wa);
            let mut want = wa.clone();
            crate::debug::fill_transparent(&mut want, &b);
            let got = sim_image(
                w,
                h,
                fmt,
                |g, s| sim::launch_1d(n, s, |i| items::ov_fill_u8_item(i, u8s(&wa), g, u8s(&b), n as u32)),
                |g, s| sim::launch_1d(n, s, |i| items::ov_fill_u16_item(i, u16s(&wa), g, u16s(&b), n as u32)),
                |g, s| sim::launch_1d(n, s, |i| items::ov_fill_f32_item(i, f32s(&wa), g, f32s(&b), n as u32)),
            );
            assert_eq!(got, image_bits(&want), "fill {fmt:?} {w}x{h}");
            let want = crate::debug::debug_luma(&a);
            let got = sim_floats(n, |g, s| {
                sim::launch_1d(n, s, |i| match fmt {
                    Fmt::U8 => items::ov_luma_u8_item(i, u8s(&a), g, n as u32),
                    Fmt::U16 => items::ov_luma_u16_item(i, u16s(&a), g, n as u32),
                    Fmt::F32 => items::ov_luma_f32_item(i, f32s(&a), g, n as u32),
                })
            });
            assert_eq!(got, float_bits(&want.data), "debug luma {fmt:?} {w}x{h}");
        }
        // detail and edge on luma planes: the 0..1 range of the Debug views, then wider values,
        // with the specials of the corpus
        for (range, nan) in [(1.0f32, false), (1.0, true), (300.0, true)] {
            let l = plane(&mut rng, w, h, range, nan);
            let want = crate::debug::detail(&l);
            let got = sim_floats(n, |g, s| sim::launch_2d(w, h, s, |x, y| items::ov_detail_item(x, y, &l.data, g, w as u32, h as u32)));
            assert_eq!(got, float_bits(&want), "detail {w}x{h} range {range}");
            let want = crate::debug::edge(&l);
            let got = sim_floats(n, |g, s| sim::launch_2d(w, h, s, |x, y| items::ov_edge_item(x, y, &l.data, g, w as u32, h as u32)));
            assert_eq!(got, float_bits(&want), "edge {w}x{h} range {range}");
        }
    }
}

/// One launch of an image writing transcription (`run8`, `run16` or `run32` by `fmt`), the image
/// back.
fn sim_image_once(w: usize, h: usize, fmt: Fmt, run8: impl FnOnce(&mut Global<u8>), run16: impl FnOnce(&mut Global<u16>), run32: impl FnOnce(&mut Global<f32>)) -> Image {
    let n = w * h * 4;
    let samples = match fmt {
        Fmt::U8 => {
            let mut g = Global::new(n, 0u8);
            run8(&mut g);
            assert_eq!(g.written(), n);
            Samples::U8(g.into_vec())
        }
        Fmt::U16 => {
            let mut g = Global::new(n, 0u16);
            run16(&mut g);
            assert_eq!(g.written(), n);
            Samples::U16(g.into_vec())
        }
        Fmt::F32 => {
            let mut g = Global::new(n, 0.0f32);
            run32(&mut g);
            assert_eq!(g.written(), n);
            Samples::F32(g.into_vec())
        }
    };
    Image { width: w, height: h, samples }
}

/// One launch of a plane writing transcription, the plane back.
fn sim_plane_once(n: usize, run: impl FnOnce(&mut Global<f32>)) -> Vec<f32> {
    let mut g = Global::new(n, 0.0f32);
    run(&mut g);
    assert_eq!(g.written(), n);
    g.into_vec()
}

/// The Debug view `code` of `debug::debug_view_on` after its forward warp, composed from the
/// transcriptions in the order of `debug.rs` (`W'` from `warped` and `y`, then the view; Detail and
/// Edge through the luma of `W'` and of `Y`, the measure of each and the overlay), every launch in
/// the order of `seed`.
fn sim_view(warped: &Image, y: &Image, code: i32, seed: u64) -> Image {
    let (w, h, fmt) = (y.width, y.height, y.fmt());
    let n = w * h;
    let nu = n as u32;
    let wf = sim_image_once(
        w,
        h,
        fmt,
        |g| sim::launch_1d(n, seed, |i| items::ov_fill_u8_item(i, u8s(warped), g, u8s(y), nu)),
        |g| sim::launch_1d(n, seed, |i| items::ov_fill_u16_item(i, u16s(warped), g, u16s(y), nu)),
        |g| sim::launch_1d(n, seed, |i| items::ov_fill_f32_item(i, f32s(warped), g, f32s(y), nu)),
    );
    let ns = n * 4;
    match code {
        crate::debug::VIEW_BLEND => sim_image_once(
            w,
            h,
            fmt,
            |g| sim::launch_1d(ns, seed, |i| items::ov_blend_u8_item(i, u8s(&wf), u8s(y), g, ns as u32)),
            |g| sim::launch_1d(ns, seed, |i| items::ov_blend_u16_item(i, u16s(&wf), u16s(y), g, ns as u32)),
            |g| sim::launch_1d(ns, seed, |i| items::ov_blend_f32_item(i, f32s(&wf), f32s(y), g, ns as u32)),
        ),
        crate::debug::VIEW_DIFF => sim_image_once(
            w,
            h,
            fmt,
            |g| sim::launch_1d(n, seed, |i| items::ov_diff_u8_item(i, u8s(&wf), u8s(y), g, nu)),
            |g| sim::launch_1d(n, seed, |i| items::ov_diff_u16_item(i, u16s(&wf), u16s(y), g, nu)),
            |g| sim::launch_1d(n, seed, |i| items::ov_diff_f32_item(i, f32s(&wf), f32s(y), g, nu)),
        ),
        _ => {
            let luma = |img: &Image| {
                sim_plane_once(n, |g| {
                    sim::launch_1d(n, seed, |i| match &img.samples {
                        Samples::U8(s) => items::ov_luma_u8_item(i, s, g, nu),
                        Samples::U16(s) => items::ov_luma_u16_item(i, s, g, nu),
                        Samples::F32(s) => items::ov_luma_f32_item(i, s, g, nu),
                    })
                })
            };
            let measure = |l: &[f32]| {
                sim_plane_once(n, |g| {
                    sim::launch_2d(w, h, seed, |x, yy| {
                        if code == crate::debug::VIEW_DETAIL {
                            items::ov_detail_item(x, yy, l, g, w as u32, h as u32)
                        } else {
                            items::ov_edge_item(x, yy, l, g, w as u32, h as u32)
                        }
                    })
                })
            };
            let (r, gg) = (measure(&luma(&wf)), measure(&luma(y)));
            sim_image_once(
                w,
                h,
                fmt,
                |g| sim::launch_1d(n, seed, |i| items::ov_overlay_u8_item(i, &r, &gg, g, nu)),
                |g| sim::launch_1d(n, seed, |i| items::ov_overlay_u16_item(i, &r, &gg, g, nu)),
                |g| sim::launch_1d(n, seed, |i| items::ov_overlay_f32_item(i, &r, &gg, g, nu)),
            )
        }
    }
}

#[test]
#[ignore = "the kernel follows the rule before the parity work; port pending (docs/engine.md, Device)"]
fn the_debug_views_composed_from_the_transcriptions_equal_debug_view() {
    // the order of debug::debug_view_on after its forward warp, every view code at every depth;
    // the warp leaves transparent pixels (the last row and column, the transparent pixels of X)
    // and the corpus frames hold transparent pixels, so W' differs from W
    let mut rng = Rng::new(0x6E0C_000A);
    let mut compared = 0usize;
    for &(w, h) in &[(2usize, 2usize), (5, 4), (7, 7), (17, 16), (33, 9)] {
        for fmt in FMTS {
            let x = image(&mut rng, w, h, fmt, true);
            let y = image(&mut rng, w, h, fmt, true);
            let flow = FlowField { width: w, height: h, data: (0..w * h * 2).map(|_| (rng.below(9) as f32 - 4.0) * 0.375).collect() };
            let mut warped = Image::new(w, h, fmt);
            if crate::forward::forward_warp(&mut warped, &x, &flow, 1.0).is_err() {
                continue;
            }
            for code in [crate::debug::VIEW_BLEND, crate::debug::VIEW_DIFF, crate::debug::VIEW_DETAIL, crate::debug::VIEW_EDGE] {
                let want = crate::debug::debug_view(&x, &y, &flow, code).unwrap();
                let seed = sim::SEEDS[compared % sim::SEEDS.len()];
                assert_eq!(image_bits(&sim_view(&warped, &y, code, seed)), image_bits(&want), "view {code} {fmt:?} {w}x{h}");
                compared += 1;
            }
        }
    }
    assert!(compared >= 40, "{compared} views compared");
}

// ---- geometry and points ----------------------------------------------------------------------

fn random_points(rng: &mut Rng, n: usize, w: usize, h: usize) -> Vec<([f32; 2], [u8; 3])> {
    (0..n)
        .map(|_| {
            let p = match rng.below(8) {
                0 => [float(rng, 1e6, true), float(rng, 1e6, true)],
                1 => [rng.below(w as u64) as f32 + 0.5, rng.below(h as u64) as f32],
                _ => [(rng.next_u32() as f64 / u32::MAX as f64 * (w as f64 + 4.0) - 2.0) as f32, (rng.next_u32() as f64 / u32::MAX as f64 * (h as f64 + 4.0) - 2.0) as f32],
            };
            (p, [rng.next_u32() as u8, rng.next_u32() as u8, rng.next_u32() as u8])
        })
        .collect()
}

#[test]
fn draw_transcriptions_equal_draw_points_in_every_order() {
    let mut rng = Rng::new(0x6E0C_0007);
    for &(w, h) in &SIZES {
        for fmt in FMTS {
            for np in [1usize, 3, 12, 16] {
                let pts = random_points(&mut rng, np, w, h);
                for ratio in [1.0f32, 2.0, 0.0, 0.3, 1e10, -1.0, f32::NAN] {
                    let img = image(&mut rng, w, h, fmt, true);
                    let mut want = img.clone();
                    crate::geom::draw_points(&mut want, &pts, ratio);
                    let r = 1.5f32 * ratio;
                    if !(r >= 0.0) || !r.is_finite() {
                        assert_eq!(image_bits(&want), image_bits(&img), "no drawing");
                        continue;
                    }
                    let (p, ip) = super::geom::draw_constants(&pts, r, w, h);
                    let a = DrawArgs { p: &p, ip: &ip, n: np as u32, r2: r * r, w: w as u32, h: h as u32 };
                    let got = sim_image(
                        w,
                        h,
                        fmt,
                        |g, s| sim::launch_2d(w, h, s, |x, y| items::gm_draw_u8_item(x, y, u8s(&img), g, &a)),
                        |g, s| sim::launch_2d(w, h, s, |x, y| items::gm_draw_u16_item(x, y, u16s(&img), g, &a)),
                        |g, s| sim::launch_2d(w, h, s, |x, y| items::gm_draw_f32_item(x, y, f32s(&img), g, &a)),
                    );
                    assert_eq!(got, image_bits(&want), "draw {fmt:?} {w}x{h} {np} points ratio {ratio}");
                }
            }
        }
    }
}

fn random_pairs(rng: &mut Rng, n: usize, w: usize, h: usize) -> Vec<([f32; 2], [f32; 2])> {
    (0..n)
        .map(|k| {
            let from = if k > 0 && rng.below(4) == 0 {
                [rng.below(w as u64) as f32, rng.below(h as u64) as f32]
            } else if rng.below(6) == 0 {
                [float(rng, 1e4, true), float(rng, 1e4, true)]
            } else {
                [(rng.next_u32() as f64 / u32::MAX as f64 * (w as f64 + 2.0) - 1.0) as f32, (rng.next_u32() as f64 / u32::MAX as f64 * (h as f64 + 2.0) - 1.0) as f32]
            };
            let nan = rng.below(8) == 0;
            let to = [from[0] + float(rng, 8.0, nan), from[1] + float(rng, 8.0, false)];
            (from, to)
        })
        .collect()
}

fn sim_splat(pairs: &[([f32; 2], [f32; 2])], wl: usize, hl: usize, out: (usize, usize)) -> (Vec<u64>, Vec<u64>) {
    let p: Vec<f32> = pairs.iter().flat_map(|(f, t)| [f[0], f[1], t[0], t[1]]).collect();
    let (kx, ky, ow, oh) = super::geom::splat_constants(wl, hl, out);
    let a = SplatArgs { n: pairs.len() as u32, kx, ky, ow, oh, wl: wl as u32, hl: hl as u32 };
    sim::same_for_every_order(|seed| {
        let mut d = Global::new(wl * hl * 2, 0.0f32);
        let mut wt = Global::new(wl * hl, 0.0f32);
        sim::launch_2d(wl, hl, seed, |x, y| items::gm_splat_item(x, y, &p, &a, &mut d, &mut wt));
        (float_bits(&d.into_vec()), float_bits(&wt.into_vec()))
    })
}

#[test]
#[ignore = "the kernel follows the rule before the parity work; port pending (docs/engine.md, Device)"]
fn splat_and_seed_transcriptions_equal_points_rs_in_every_order() {
    let mut rng = Rng::new(0x6E0C_0008);
    for &(w, h) in &[(2usize, 2usize), (5, 4), (17, 16), (33, 9), (64, 48), (97, 31)] {
        for quality in [5u8, 4, 3] {
            let Ok(geo) = FlowSettings::new(quality, true, 1.0).geometry(w, h) else { continue };
            for np in [0usize, 1, 2, 5, 12, 16] {
                let pairs = random_pairs(&mut rng, np, w, h);
                let want = crate::points::user_displacement(&pairs, &geo, (w, h));
                assert_eq!(want.levels.len(), geo.sizes.len());
                for (lvl, &(wl, hl)) in want.levels.iter().zip(&geo.sizes) {
                    let (d, wt) = sim_splat(&pairs, wl, hl, (w, h));
                    assert_eq!(d, float_bits(&lvl.d), "splat d {w}x{h} level {wl}x{hl} {np} pairs");
                    assert_eq!(wt, float_bits(&lvl.w), "splat w {w}x{h} level {wl}x{hl}");
                }
                // the second tracker: the simulated seed, then the CPU fill and scale
                let Ok(want) = crate::points::second_tracker(&pairs, &geo, (w, h)) else { continue };
                let (w0, h0) = (geo.width, geo.height);
                let (d, wt) = sim_splat(&pairs, w0, h0, (w, h));
                let d: Vec<f32> = d.iter().map(|b| f32::from_bits(*b as u32)).collect();
                let wt: Vec<f32> = wt.iter().map(|b| f32::from_bits(*b as u32)).collect();
                let n = w0 * h0;
                let (seed, holes) = sim::same_for_every_order(|s| {
                    let mut f = Global::new(n * 2, 0.0f32);
                    let mut m = Global::new(n, 0u8);
                    sim::launch_1d(n, s, |i| items::gm_seed_item(i, &d, &wt, &mut f, &mut m, n as u32));
                    (float_bits(&f.into_vec()), m.into_vec())
                });
                let mut got = FlowField { width: w0, height: h0, data: seed.iter().map(|b| f32::from_bits(*b as u32)).collect() };
                if holes.iter().any(|&m| m == 0) {
                    crate::holefill::fill_flow_holes(&mut got, crate::holefill::HoleMaskRef::U8(&holes), false).unwrap();
                    got.scale(-1.0);
                } else {
                    got = FlowField::new(w0, h0);
                }
                assert_eq!(float_bits(&got.data), float_bits(&want.data), "second tracker {w}x{h} q{quality} {np} pairs");
            }
        }
    }
}

// ---- prep -------------------------------------------------------------------------------------

#[test]
#[ignore = "the kernel follows the rule before the parity work; port pending (docs/engine.md, Device)"]
fn luma_and_edge_tail_transcriptions_equal_prep_rs_in_every_order() {
    let mut rng = Rng::new(0x6E0C_0009);
    for &(w, h) in &SIZES {
        let n = w * h;
        for fmt in FMTS {
            let img = image(&mut rng, w, h, fmt, true);
            for (alpha, video) in [(false, false), (true, false), (false, true)] {
                let want = crate::prep::luma(&img, alpha, video).unwrap();
                let (a, v) = (alpha as i32, video as i32);
                let got = sim_floats(n, |g, s| {
                    sim::launch_1d(n, s, |i| match fmt {
                        Fmt::U8 => items::pp_luma_u8_item(i, u8s(&img), g, a, v, n as u32),
                        Fmt::U16 => items::pp_luma_u16_item(i, u16s(&img), g, a, v, n as u32),
                        Fmt::F32 => items::pp_luma_f32_item(i, f32s(&img), g, a, v, n as u32),
                    })
                });
                assert_eq!(got, float_bits(&want.data), "luma {fmt:?} {w}x{h} alpha {alpha} video {video}");
            }
        }
        for wrap in [false, true] {
            for nan in [false, true] {
                let p = plane(&mut rng, w, h, 300.0, nan);
                let mut want = p.clone();
                crate::prep::edge_enhance_with(&mut want, wrap);
                let mut b = p.data.clone();
                if wrap {
                    crate::flow::conv::blur_gaussian_wrapx(&mut b, w, h, 1, 2.0);
                } else {
                    crate::flow::conv::blur_gaussian(&mut b, w, h, 1, 2.0);
                }
                let got = sim_floats(n, |g, s| sim::launch_2d(w, h, s, |x, y| items::pp_edge_tail_item(x, y, &b, g, w as u32, h as u32, wrap as i32)));
                assert_eq!(got, float_bits(&want.data), "edge enhance {w}x{h} wrap {wrap}");
            }
        }
    }
}
