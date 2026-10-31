//! The matte weighted merge of create-vectors (a rule of this project that no effect runs):
//! `encode_vectors_merged` and `merge_flows` against the numpy model of `tests/fixtures/gen_merge.py`, the single layer
//! identity with `encode_vectors`, the errors and the thread count.

use super::*;
use crate::image::Fmt;

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

struct Case {
    fmt: Fmt,
    w: usize,
    h: usize,
    scale: f32,
    layers: Vec<(FlowField, Matte)>,
    want: Image,
    want_flow: FlowField,
}

fn load_cases() -> Vec<Case> {
    let blob = crate::fixture::hex(include_str!("../../tests/fixtures/merge_cases.hex"));
    assert_eq!(&blob[..4], b"MRG1");
    let mut r = Reader { b: &blob, p: 4 };
    let n = r.u32();
    let mut out = Vec::new();
    for _ in 0..n {
        let depth = r.u32();
        let (w, h) = (r.u32() as usize, r.u32() as usize);
        let scale = r.f32();
        let nl = r.u32();
        let mut layers = Vec::new();
        for _ in 0..nl {
            let (fw, fh) = (r.u32() as usize, r.u32() as usize);
            let flow = FlowField { width: fw, height: fh, data: r.floats(fw * fh * 2) };
            let m = Matte { width: w, height: h, data: r.floats(w * h) };
            layers.push((flow, m));
        }
        let bytes = [1usize, 2, 4][depth as usize];
        let want = image_from(depth, r.bytes(w * h * 4 * bytes), w, h);
        let want_flow = FlowField { width: w, height: h, data: r.floats(w * h * 2) };
        let fmt = [Fmt::U8, Fmt::U16, Fmt::F32][depth as usize];
        out.push(Case { fmt, w, h, scale, layers, want, want_flow });
    }
    assert_eq!(r.p, blob.len());
    out
}

fn bits(img: &Image) -> Vec<u32> {
    match &img.samples {
        Samples::U8(v) => v.iter().map(|&x| u32::from(x)).collect(),
        Samples::U16(v) => v.iter().map(|&x| u32::from(x)).collect(),
        Samples::F32(v) => v.iter().map(|x| x.to_bits()).collect(),
    }
}

fn refs(layers: &[(FlowField, Matte)]) -> Vec<(&FlowField, &Matte)> {
    layers.iter().map(|(f, m)| (f, m)).collect()
}

#[test]
fn model_fixtures_are_reproduced_bit_for_bit() {
    let cases = load_cases();
    assert_eq!(cases.len(), 11);
    for (i, c) in cases.iter().enumerate() {
        let layers = refs(&c.layers);
        let got = encode_vectors_merged_image(c.w, c.h, c.fmt, &layers, c.scale).unwrap();
        assert_eq!(bits(&got), bits(&c.want), "case {i} ({:?} {}x{}, {} layers)", c.fmt, c.w, c.h, layers.len());
        let flow = merge_flows(c.w, c.h, &layers).unwrap();
        let fb: Vec<u32> = flow.data.iter().map(|v| v.to_bits()).collect();
        let wb: Vec<u32> = c.want_flow.data.iter().map(|v| v.to_bits()).collect();
        assert_eq!(fb, wb, "merged flow of case {i}");
    }
}

/// A pseudo random flow of `w x h` with values in `-spread..spread`, a few far beyond the clamp.
fn flow(w: usize, h: usize, spread: f32, seed: u32) -> FlowField {
    let mut s = seed;
    let mut f = FlowField::new(w, h);
    for (i, v) in f.data.iter_mut().enumerate() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let u = (s >> 8) as f32 / 16777216.0;
        *v = (u * 2.0 - 1.0) * spread * if i % 17 == 3 { 40.0 } else { 1.0 };
    }
    f
}

#[test]
fn one_layer_of_weight_one_is_encode_vectors_bit_for_bit() {
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        for (w, h, fw, fh) in [(2usize, 2usize, 2usize, 2usize), (13, 9, 13, 9), (16, 12, 8, 6), (10, 11, 21, 23), (33, 17, 16, 8)] {
            for (scale, spread) in [(5.0f32, 4.0f32), (2048.0, 30.0), (0.3, 0.01)] {
                let f = flow(fw, fh, spread, (w * 31 + fw) as u32);
                let one = Matte::filled(w, h, 1.0);
                let want = encode_vectors_image(w, h, fmt, &f, scale).unwrap();
                let got = encode_vectors_merged_image(w, h, fmt, &[(&f, &one)], scale).unwrap();
                assert_eq!(bits(&got), bits(&want), "{fmt:?} {w}x{h} flow {fw}x{fh} scale {scale}");
            }
        }
    }
}

#[test]
fn weights_mix_the_layers_and_a_zero_weight_drops_one() {
    let (w, h) = (12usize, 8usize);
    let (a, b) = (flow(6, 4, 3.0, 1), flow(6, 4, 3.0, 2));
    let (one, zero) = (Matte::filled(w, h, 1.0), Matte::filled(w, h, 0.0));
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let only_a = encode_vectors_image(w, h, fmt, &a, 4.0).unwrap();
        // a layer of weight 0 adds +0.0 (its vectors are finite), so only the other one shows
        assert_eq!(encode_vectors_merged_image(w, h, fmt, &[(&a, &one), (&b, &zero)], 4.0).unwrap(), only_a);
        assert_eq!(encode_vectors_merged_image(w, h, fmt, &[(&b, &zero), (&a, &one)], 4.0).unwrap(), only_a);
    }
    // two halves sum to the mean of two equal layers
    let half = Matte::filled(w, h, 0.5);
    let m = merge_flows(w, h, &[(&a, &half), (&a, &half)]).unwrap();
    let full = merge_flows(w, h, &[(&a, &one)]).unwrap();
    assert_eq!(m, full, "0.5 * v + 0.5 * v is v exactly");
}

#[test]
fn the_merged_flow_of_one_layer_is_its_resampled_flow_in_output_pixels() {
    // a uniform flow (2, -1) at half resolution is (2 * 15 / 7, -1 * 11 / 5) px of the output
    let (w, h) = (16usize, 12usize);
    let mut f = FlowField::new(8, 6);
    for p in f.data.chunks_exact_mut(2) {
        p[0] = 2.0;
        p[1] = -1.0;
    }
    let m = merge_flows(w, h, &[(&f, &Matte::filled(w, h, 1.0))]).unwrap();
    let (kx, ky) = (15.0f32 / 7.0f32, 11.0f32 / 5.0f32);
    assert!(m.data.chunks_exact(2).all(|p| p[0] == 2.0f32 * kx && p[1] == -1.0f32 * ky));
    // the encoder of that field at its own size gives the merged encoder's codes up to the rounding of the scale
    let enc = encode_vectors_merged_image(w, h, Fmt::F32, &[(&f, &Matte::filled(w, h, 1.0))], 4.0).unwrap();
    let Samples::F32(v) = &enc.samples else { panic!() };
    assert!(v.chunks_exact(4).all(|p| p[0] == 1.0 && (p[1] - (2.0 * kx / 4.0 + 1.0) * 0.5).abs() < 1e-6 && p[3] == 0.0));
}

#[test]
fn merge_errors_are_mismatches() {
    let f = flow(4, 4, 1.0, 3);
    let m = Matte::filled(4, 4, 1.0);
    let mut img = Image::new(4, 4, Fmt::U8);
    assert!(matches!(encode_vectors_merged(&mut img, &[], 1.0), Err(FxError::Mismatch(_))));
    assert!(matches!(encode_vectors_merged(&mut img, &[(&f, &Matte::filled(3, 4, 1.0))], 1.0), Err(FxError::Mismatch(_))));
    assert!(matches!(encode_vectors_merged(&mut img, &[(&FlowField::new(1, 4), &m)], 1.0), Err(FxError::Mismatch(_))));
    let mut small = Image::new(1, 4, Fmt::U8);
    assert!(matches!(encode_vectors_merged(&mut small, &[(&f, &Matte::filled(1, 4, 1.0))], 1.0), Err(FxError::Mismatch(_))));
    let bad = FlowField { width: 4, height: 4, data: vec![0.0; 3] };
    assert!(matches!(merge_flows(4, 4, &[(&bad, &m)]), Err(FxError::Mismatch(_))));
    assert!(matches!(merge_flows(4, 4, &[]), Err(FxError::Mismatch(_))));
}

#[test]
fn thread_count_does_not_change_the_merge() {
    let pool = |n: usize| rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
    for c in load_cases() {
        let layers = refs(&c.layers);
        let one = pool(1).install(|| (encode_vectors_merged_image(c.w, c.h, c.fmt, &layers, c.scale).unwrap(), merge_flows(c.w, c.h, &layers).unwrap()));
        for n in [4, 16] {
            let other = pool(n).install(|| (encode_vectors_merged_image(c.w, c.h, c.fmt, &layers, c.scale).unwrap(), merge_flows(c.w, c.h, &layers).unwrap()));
            assert_eq!(bits(&one.0), bits(&other.0));
            assert_eq!(one.1, other.1);
        }
    }
}
