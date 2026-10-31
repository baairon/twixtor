//! Tests of the vector image coding: hand worked goldens, an exhaustive decode table, round trips,
//! the numpy model fixtures (`tests/fixtures/gen_vectors.py`) and thread count invariance.

use super::*;
use crate::image::Fmt;

fn flat_flow(w: usize, h: usize, dx: f32, dy: f32) -> FlowField {
    let mut f = FlowField::new(w, h);
    for y in 0..h {
        for x in 0..w {
            f.set(x, y, [dx, dy]);
        }
    }
    f
}

fn px8(img: &Image, x: usize, y: usize) -> [u8; 4] {
    match &img.samples {
        Samples::U8(v) => {
            let i = (y * img.width + x) * 4;
            [v[i], v[i + 1], v[i + 2], v[i + 3]]
        }
        _ => panic!("not u8"),
    }
}

fn px16(img: &Image, x: usize, y: usize) -> [u16; 4] {
    match &img.samples {
        Samples::U16(v) => {
            let i = (y * img.width + x) * 4;
            [v[i], v[i + 1], v[i + 2], v[i + 3]]
        }
        _ => panic!("not u16"),
    }
}

fn pxf(img: &Image, x: usize, y: usize) -> [f32; 4] {
    match &img.samples {
        Samples::F32(v) => {
            let i = (y * img.width + x) * 4;
            [v[i], v[i + 1], v[i + 2], v[i + 3]]
        }
        _ => panic!("not f32"),
    }
}

// ---------------------------------------------------------------------------------------
// encode goldens
// ---------------------------------------------------------------------------------------

#[test]
fn zero_flow_encodes_as_the_centre_value_with_full_alpha_and_zero_blue() {
    let flow = FlowField::new(6, 4);
    let a = encode_vectors_image(6, 4, Fmt::U8, &flow, 10.0).unwrap();
    let b = encode_vectors_image(6, 4, Fmt::U16, &flow, 10.0).unwrap();
    let c = encode_vectors_image(6, 4, Fmt::F32, &flow, 10.0).unwrap();
    for y in 0..4 {
        for x in 0..6 {
            // 127 = trunc(1.0 * 127 + 0.5), 32767, 0.5; alpha is the white, blue is zero
            assert_eq!(px8(&a, x, y), [255, 127, 127, 0]);
            assert_eq!(px16(&b, x, y), [65535, 32767, 32767, 0]);
            assert_eq!(pxf(&c, x, y), [1.0, 0.5, 0.5, 0.0]);
        }
    }
}

#[test]
fn full_scale_and_clamping() {
    // flow (+scale, -scale): fx = +1, fy = -1: R = 254, G = 0 (8 bit), 65534 and 0 (16 bit), 1.0 and 0.0 (float)
    let flow = flat_flow(5, 5, 8.0, -8.0);
    let a = encode_vectors_image(5, 5, Fmt::U8, &flow, 8.0).unwrap();
    assert_eq!(px8(&a, 2, 2), [255, 254, 0, 0]);
    let b = encode_vectors_image(5, 5, Fmt::U16, &flow, 8.0).unwrap();
    assert_eq!(px16(&b, 2, 2), [65535, 65534, 0, 0]);
    let c = encode_vectors_image(5, 5, Fmt::F32, &flow, 8.0).unwrap();
    assert_eq!(pxf(&c, 2, 2), [1.0, 1.0, 0.0, 0.0]);
    // beyond the scale the integer depths clamp, the float depth does not
    let flow = flat_flow(5, 5, 30.0, -50.0);
    let a = encode_vectors_image(5, 5, Fmt::U8, &flow, 8.0).unwrap();
    assert_eq!(px8(&a, 0, 0), [255, 254, 0, 0]);
    let b = encode_vectors_image(5, 5, Fmt::U16, &flow, 8.0).unwrap();
    assert_eq!(px16(&b, 4, 4), [65535, 65534, 0, 0]);
    let c = encode_vectors_image(5, 5, Fmt::F32, &flow, 8.0).unwrap();
    // (30 / 8 + 1) * 0.5 = 2.375 and (-50 / 8 + 1) * 0.5 = -2.625
    assert_eq!(pxf(&c, 1, 3), [1.0, 2.375, -2.625, 0.0]);
}

#[test]
fn quarter_scale_values_are_the_hand_computed_ones() {
    // fx = 0.25 -> (1.25 as f64) * 127 + 0.5 = 159.25 -> 159; fy = -0.5 -> 0.5 * 127 + 0.5 = 64
    let flow = flat_flow(4, 4, 2.0, -4.0);
    let a = encode_vectors_image(4, 4, Fmt::U8, &flow, 8.0).unwrap();
    assert_eq!(px8(&a, 1, 1), [255, 159, 64, 0]);
    // 16 bit: 1.25 * 32767 + 0.5 = 40959.25 -> 40959; 0.5 * 32767 + 0.5 = 16384
    let b = encode_vectors_image(4, 4, Fmt::U16, &flow, 8.0).unwrap();
    assert_eq!(px16(&b, 1, 1), [65535, 40959, 16384, 0]);
    // float: (0.25 + 1) * 0.5 = 0.625 and (-0.5 + 1) * 0.5 = 0.25
    let c = encode_vectors_image(4, 4, Fmt::F32, &flow, 8.0).unwrap();
    assert_eq!(pxf(&c, 1, 1), [1.0, 0.625, 0.25, 0.0]);
}

#[test]
fn the_flow_resolution_is_taken_into_account() {
    // a flow with half the columns: a flow value of 3 flow pixels is 3 * (dst.w1 / flow.w1) destination pixels
    // dst 9 wide (w1 = 8), flow 5 wide (w1 = 4): ratio 2; vector (3, 0) -> 6 pixels of 8 -> fx = 0.75
    let flow = flat_flow(5, 9, 3.0, 0.0);
    let a = encode_vectors_image(9, 9, Fmt::U8, &flow, 8.0).unwrap();
    // (0.75 + 1) * 127 + 0.5 = 222.75 -> 222; dy 0 -> 127
    assert_eq!(px8(&a, 3, 4), [255, 222, 127, 0]);
    let c = encode_vectors_image(9, 9, Fmt::F32, &flow, 8.0).unwrap();
    assert_eq!(pxf(&c, 3, 4), [1.0, 0.875, 0.5, 0.0]);
}

#[test]
fn encode_resamples_a_ramp_flow_bilinearly() {
    // flow 3 wide with dx = 0, 4, 8; dst 5 wide: sampled at x * (2/4) = 0, 0.5, 1, 1.5, 2 -> 0, 2, 4, 6, 8 flow
    // pixels, times the ratio 2 over the scale 16 -> 0, 0.25, 0.5, 0.75, 1.0
    let mut flow = FlowField::new(3, 2);
    for y in 0..2 {
        for x in 0..3 {
            flow.set(x, y, [4.0 * x as f32, 0.0]);
        }
    }
    let c = encode_vectors_image(5, 2, Fmt::F32, &flow, 16.0).unwrap();
    let r: Vec<f32> = (0..5).map(|x| pxf(&c, x, 0)[1]).collect();
    assert_eq!(r, vec![0.5, 0.625, 0.75, 0.875, 1.0]);
}

#[test]
fn nan_flow_encodes_as_zero_in_the_integer_depths() {
    // a NaN passes the clamp unchanged; its truncation is the integer indefinite, whose low
    // byte / word is 0
    let mut flow = flat_flow(4, 4, 0.0, 0.0);
    flow.set(1, 1, [f32::NAN, 0.0]);
    let a = encode_vectors_image(4, 4, Fmt::U8, &flow, 5.0).unwrap();
    // pixel (1, 1) samples exactly the NaN pixel in x; the other pixels share bilinear weights with it
    assert_eq!(px8(&a, 1, 1)[1], 0);
    assert_eq!(px8(&a, 3, 3), [255, 127, 127, 0]);
}

#[test]
fn encode_rejects_degenerate_sizes() {
    let flow = FlowField::new(1, 4);
    assert!(matches!(encode_vectors_image(4, 4, Fmt::U8, &flow, 1.0), Err(FxError::Mismatch(_))));
    let flow = FlowField::new(4, 4);
    assert!(matches!(encode_vectors_image(4, 1, Fmt::U8, &flow, 1.0), Err(FxError::Mismatch(_))));
}

// ---------------------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------------------

fn img_u8_rg(w: usize, h: usize, f: impl Fn(usize, usize) -> [u8; 4]) -> Image {
    let mut v = Vec::new();
    for y in 0..h {
        for x in 0..w {
            v.extend_from_slice(&f(x, y));
        }
    }
    Image { width: w, height: h, samples: Samples::U8(v) }
}

#[test]
fn decode_goldens_8_bit() {
    let img = img_u8_rg(4, 1, |x, _| match x {
        0 => [255, 127, 127, 0], // centre: zero exactly
        1 => [255, 254, 0, 0],   // +scale, -scale
        2 => [255, 0, 254, 9],   // -scale, +scale (blue is ignored)
        _ => [0, 200, 10, 0],    // alpha 0: a hole, the zero vector whatever the colour says
    });
    let f = decode_vectors(&img, 12.0);
    assert_eq!(f.at(0, 0), [0.0, 0.0]);
    assert_eq!(f.at(1, 0), [12.0, -12.0]);
    assert_eq!(f.at(2, 0), [-12.0, 12.0]);
    assert_eq!(f.at(3, 0), [0.0, 0.0]);
    // a non central value: ((1 * 2) / 254 - 1) * 12 in double, narrowed once
    let img = img_u8_rg(1, 1, |_, _| [1, 1, 253, 0]);
    let f = decode_vectors(&img, 12.0);
    let want = (((1.0f64 * 2.0) / 254.0 - 1.0) * 12.0) as f32;
    assert_eq!(f.at(0, 0)[0].to_bits(), want.to_bits());
    assert_eq!(f.at(0, 0)[1].to_bits(), (((253.0f64 * 2.0) / 254.0 - 1.0) * 12.0).to_bits_f32());
}

trait ToBitsF32 {
    fn to_bits_f32(self) -> u32;
}
impl ToBitsF32 for f64 {
    fn to_bits_f32(self) -> u32 {
        (self as f32).to_bits()
    }
}

#[test]
fn decode_goldens_16_bit_and_float() {
    let v = vec![65535u16, 32767, 32767, 0, 65535, 65534, 0, 0, 0, 1, 1, 1, 7, 0, 65534, 0];
    let img = Image { width: 4, height: 1, samples: Samples::U16(v) };
    let f = decode_vectors(&img, 5.0);
    assert_eq!(f.at(0, 0), [0.0, 0.0]);
    assert_eq!(f.at(1, 0), [5.0, -5.0]);
    assert_eq!(f.at(2, 0), [0.0, 0.0]); // alpha 0
    assert_eq!(f.at(3, 0), [-5.0, 5.0]);
    let v = vec![1.0f32, 0.5, 0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.0, 0.25, 0.75, 0.125, 0.0];
    let img = Image { width: 4, height: 1, samples: Samples::F32(v) };
    let f = decode_vectors(&img, 8.0);
    assert_eq!(f.at(0, 0), [0.0, 0.0]);
    assert_eq!(f.at(1, 0), [8.0, -8.0]);
    assert_eq!(f.at(2, 0), [0.0, 0.0]);
    // (0.75 * 2 - 1) * 8 = 4 and (0.125 * 2 - 1) * 8 = -6
    assert_eq!(f.at(3, 0), [4.0, -6.0]);
}

#[test]
fn decode_every_8_bit_value_against_the_double_formula() {
    let img = img_u8_rg(256, 2, |x, y| [255, x as u8, 255 - x as u8, y as u8]);
    for &scale in &[1.0f32, 0.37, 12.5, 1000.0] {
        let f = decode_vectors(&img, scale);
        for x in 0..256usize {
            let u = (((x as f64 * 2.0) / 254.0 - 1.0) * scale as f64) as f32;
            let v = ((((255 - x) as f64 * 2.0) / 254.0 - 1.0) * scale as f64) as f32;
            assert_eq!(f.at(x, 0)[0].to_bits(), u.to_bits(), "R = {x}, scale {scale}");
            assert_eq!(f.at(x, 1)[1].to_bits(), v.to_bits(), "G = {}, scale {scale}", 255 - x);
        }
    }
}

#[test]
fn hole_mask_inverts_the_alpha_channel() {
    let img = img_u8_rg(3, 1, |x, _| [[0, 128, 255][x], 1, 2, 3]);
    assert_eq!(vector_hole_mask(&img), HoleMask::U8(vec![255, 127, 0]));
    let img = Image { width: 2, height: 1, samples: Samples::U16(vec![0, 1, 1, 1, 65535, 2, 2, 2]) };
    assert_eq!(vector_hole_mask(&img), HoleMask::U16(vec![65535, 0]));
    let img = Image { width: 2, height: 1, samples: Samples::F32(vec![0.25, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) };
    assert_eq!(vector_hole_mask(&img), HoleMask::F32(vec![0.75, 0.0]));
    let (f, m) = decode_vectors_with_holes(&img, 3.0);
    assert_eq!(m.len(), 2);
    assert_eq!(f.width, 2);
}

#[test]
fn vec_scale_multiplies_both_components() {
    let mut f = FlowField::new(2, 1);
    f.set(0, 0, [1.5, -2.0]);
    f.set(1, 0, [0.0, 4.0]);
    apply_vec_scale(&mut f, 2.0, 0.5);
    assert_eq!(f.at(0, 0), [3.0, -1.0]);
    assert_eq!(f.at(1, 0), [0.0, 2.0]);
    apply_vec_scale(&mut f, 1.0, 1.0);
    assert_eq!(f.at(0, 0), [3.0, -1.0]);
}

// ---------------------------------------------------------------------------------------
// round trips
// ---------------------------------------------------------------------------------------

#[test]
fn decode_then_encode_is_the_identity_for_all_8_bit_values() {
    // R, G in 0..=254 with alpha 255 (255 is outside the coded range)
    let img = img_u8_rg(255, 255, |x, y| [255, x as u8, y as u8, 0]);
    for &scale in &[1.0f32, 7.0, 33.3, 500.0] {
        let flow = decode_vectors(&img, scale);
        let back = encode_vectors_image(255, 255, Fmt::U8, &flow, scale).unwrap();
        assert_eq!(back, img, "scale {scale}");
    }
}

#[test]
fn decode_then_encode_is_the_identity_for_all_16_bit_values() {
    let w = 65535usize;
    let v: Vec<u16> = (0..w).flat_map(|x| [65535u16, x as u16, (65534 - x) as u16, 0]).collect();
    let img = Image { width: w, height: 2, samples: Samples::U16([v.clone(), v.clone()].concat()) };
    for &scale in &[1.0f32, 19.0] {
        let flow = decode_vectors(&img, scale);
        let back = encode_vectors_image(w, 2, Fmt::U16, &flow, scale).unwrap();
        assert_eq!(back, img, "scale {scale}");
    }
}

#[test]
fn encode_then_decode_stays_within_a_quantisation_step() {
    let mut flow = FlowField::new(20, 12);
    for y in 0..12 {
        for x in 0..20 {
            flow.set(x, y, [((x * 7 + y * 3) % 23) as f32 - 11.0 + 0.123, ((x * 5 + y * 11) % 19) as f32 - 9.0 - 0.321]);
        }
    }
    let scale = 12.0f32;
    for (fmt, step) in [(Fmt::U8, scale / 127.0), (Fmt::U16, scale / 32767.0), (Fmt::F32, scale * 2e-7)] {
        let img = encode_vectors_image(20, 12, fmt, &flow, scale).unwrap();
        let back = decode_vectors(&img, scale);
        for y in 0..12 {
            for x in 0..20 {
                for k in 0..2 {
                    let (a, b) = (flow.at(x, y)[k], back.at(x, y)[k]);
                    assert!((a - b).abs() <= step * 0.5001 + 1e-6, "{fmt:?} ({x},{y},{k}): {a} vs {b}, step {step}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// numpy model and threads
// ---------------------------------------------------------------------------------------

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn f32(&mut self) -> f32 {
        f32::from_bits(self.u32())
    }
    fn bytes(&mut self, n: usize) -> &'a [u8] {
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        s
    }
    fn floats(&mut self, n: usize) -> Vec<f32> {
        self.bytes(n * 4).chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }
}

fn image_from(depth: u32, b: &[u8], w: usize, h: usize) -> Image {
    let samples = match depth {
        0 => Samples::U8(b.to_vec()),
        1 => Samples::U16(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()),
        _ => Samples::F32(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
    };
    Image { width: w, height: h, samples }
}

enum Case {
    Encode { fmt: Fmt, w: usize, h: usize, flow: FlowField, scale: f32, want: Image },
    Decode { img: Image, scale: f32, want: FlowField },
}

fn load_cases() -> Vec<Case> {
    let blob = crate::fixture::hex(include_str!("../../tests/fixtures/vectors_cases.hex"));
    assert_eq!(&blob[..4], b"VEC1");
    let mut r = Reader { b: &blob, p: 4 };
    let n = r.u32();
    let mut out = Vec::new();
    for _ in 0..n {
        let (kind, depth) = (r.u32(), r.u32());
        let (w, h, fw, fh) = (r.u32() as usize, r.u32() as usize, r.u32() as usize, r.u32() as usize);
        let scale = r.f32();
        let bytes = [1usize, 2, 4][depth as usize];
        if kind == 0 {
            let flow = FlowField { width: fw, height: fh, data: r.floats(fw * fh * 2) };
            let want = image_from(depth, r.bytes(w * h * 4 * bytes), w, h);
            let fmt = [Fmt::U8, Fmt::U16, Fmt::F32][depth as usize];
            out.push(Case::Encode { fmt, w, h, flow, scale, want });
        } else {
            let img = image_from(depth, r.bytes(w * h * 4 * bytes), w, h);
            let want = FlowField { width: w, height: h, data: r.floats(w * h * 2) };
            out.push(Case::Decode { img, scale, want });
        }
    }
    assert_eq!(r.p, blob.len());
    out
}

fn same_image(a: &Image, b: &Image) -> bool {
    match (&a.samples, &b.samples) {
        (Samples::U8(x), Samples::U8(y)) => x == y,
        (Samples::U16(x), Samples::U16(y)) => x == y,
        (Samples::F32(x), Samples::F32(y)) => x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()),
        _ => false,
    }
}

fn same_flow(a: &FlowField, b: &FlowField) -> bool {
    a.width == b.width && a.height == b.height && a.data.iter().zip(&b.data).all(|(p, q)| p.to_bits() == q.to_bits())
}

#[test]
fn model_fixtures_are_reproduced_bit_for_bit() {
    let cases = load_cases();
    assert_eq!(cases.len(), 21);
    let (mut enc, mut dec) = (0, 0);
    for (i, c) in cases.iter().enumerate() {
        match c {
            Case::Encode { fmt, w, h, flow, scale, want } => {
                enc += 1;
                let got = encode_vectors_image(*w, *h, *fmt, flow, *scale).unwrap();
                assert!(same_image(&got, want), "encode case {i} ({fmt:?} {w}x{h}, flow {}x{}, scale {scale})", flow.width, flow.height);
            }
            Case::Decode { img, scale, want } => {
                dec += 1;
                let got = decode_vectors(img, *scale);
                assert!(same_flow(&got, want), "decode case {i} ({:?} {}x{}, scale {scale})", img.fmt(), img.width, img.height);
            }
        }
    }
    assert_eq!((enc, dec), (12, 9));
}

#[test]
fn thread_count_does_not_change_the_vector_coding() {
    let pool = |n: usize| rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
    for (i, c) in load_cases().iter().enumerate() {
        match c {
            Case::Encode { fmt, w, h, flow, scale, .. } => {
                let a = pool(1).install(|| encode_vectors_image(*w, *h, *fmt, flow, *scale).unwrap());
                let b = pool(4).install(|| encode_vectors_image(*w, *h, *fmt, flow, *scale).unwrap());
                assert!(same_image(&a, &b), "case {i}");
            }
            Case::Decode { img, scale, .. } => {
                let a = pool(1).install(|| decode_vectors(img, *scale));
                let b = pool(4).install(|| decode_vectors(img, *scale));
                assert!(same_flow(&a, &b), "case {i}");
            }
        }
    }
}

// ---- the single layer output and the composite of a group, read from CreateMotionVectors_64

#[test]
fn composite_lays_a_layer_over_with_its_raw_matte_8_bit() {
    // zero flow underneath (127); the layer moves by the full scale (e = (1 + 1) * 127 = 254)
    let mut img = Image::new(3, 2, Fmt::U8);
    encode_vectors(&mut img, &flat_flow(3, 2, 0.0, 0.0), 4.0).unwrap();
    // mattes 0, 128 and 255 on the first row, 255 below
    let m = vec![0.0, 128.0, 255.0, 255.0, 255.0, 255.0];
    composite_vectors(&mut img, &flat_flow(3, 2, 4.0, -4.0), 4.0, &m).unwrap();
    assert_eq!(px8(&img, 0, 0), [255, 127, 127, 0]);
    // f = 128 * f32(1/255) = 0.50196081: (254 - 127) * f + 127 + 0.5 = 191.249 -> 191; G: (0 - 127) * f + 127.5 = 63.75 -> 63
    assert_eq!(px8(&img, 1, 0), [255, 191, 63, 0]);
    // f = 255 * f32(1/255) rounds to 1.0: the layer's own vector
    assert_eq!(px8(&img, 2, 0), [255, 254, 0, 0]);
}

#[test]
fn composite_16_bit_and_float_clamp_the_vector() {
    let mut img = Image::new(2, 2, Fmt::U16);
    encode_vectors(&mut img, &flat_flow(2, 2, 0.0, 0.0), 2.0).unwrap();
    composite_vectors(&mut img, &flat_flow(2, 2, 2.0, 1.0), 2.0, &[65535.0, 0.0, 32768.0, 65535.0]).unwrap();
    assert_eq!(px16(&img, 0, 0), [0xffff, 65534, 49151, 0]);
    assert_eq!(px16(&img, 1, 0), [0xffff, 32767, 32767, 0]);
    // f = 32768 * f32(1/65535) = 0.5000076: (65534 - 32767) * f + 32767 + 0.5 = 49151.25 -> 49151
    let f = 32768.0f32 * crate::matte::PL_K16;
    let want = (((65534.0f32 - 32767.0) * f + 32767.0) + 0.5f32) as i64 as u16;
    assert_eq!(px16(&img, 0, 1)[1], want);
    // float: the composite clamps (the plain float encoder does not): 3 scales -> 1.0 -> e = 1.0
    let mut img = Image::new(2, 2, Fmt::F32);
    encode_vectors(&mut img, &flat_flow(2, 2, 0.0, 0.0), 1.0).unwrap();
    composite_vectors(&mut img, &flat_flow(2, 2, 3.0, -0.5), 1.0, &[1.0, 1.0, 0.0, 0.25]).unwrap();
    assert_eq!(pxf(&img, 0, 0), [1.0, 1.0, 0.25, 0.0]);
    assert_eq!(pxf(&img, 0, 1), [1.0, 0.5, 0.5, 0.0]);
    let g = ((0.25f32 - 0.5) * 0.25 + 0.5) + 0.0f32;
    assert_eq!(pxf(&img, 1, 1), [1.0, ((1.0f32 - 0.5) * 0.25 + 0.5) + 0.0, g, 0.0]);
    // a matte of another size is refused
    assert!(composite_vectors(&mut img, &flat_flow(2, 2, 0.0, 0.0), 1.0, &[1.0]).is_err());
}

#[test]
fn encode_vectors_blue_is_the_plain_encode_with_the_blue_plane() {
    let flow = flat_flow(2, 2, 1.0, -0.5);
    let mut plain = Image::new(2, 2, Fmt::U8);
    encode_vectors(&mut plain, &flow, 4.0).unwrap();
    let blue = Matte { width: 2, height: 2, data: vec![0.0, 0.5, 1.0, 100.0 / 255.0] };
    let mut img = Image::new(2, 2, Fmt::U8);
    encode_vectors_blue(&mut img, &flow, 4.0, &blue).unwrap();
    for (i, b) in [0u8, 128, 255, 100].into_iter().enumerate() {
        let (x, y) = (i % 2, i / 2);
        let mut want = px8(&plain, x, y);
        want[3] = b;
        assert_eq!(px8(&img, x, y), want, "pixel {i}");
    }
    assert!(encode_vectors_blue(&mut img, &flow, 4.0, &Matte { width: 1, height: 1, data: vec![0.0] }).is_err());
}
