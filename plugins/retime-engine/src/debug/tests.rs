//! The Debug views (rule G4) against the numpy model of `tests/fixtures/gen_debug.py`, the Warp
//! view against the forward warp, the compositions against their primitives, tiny frames and the
//! thread count.

use super::*;
use crate::blend::{copy, lerp};
use crate::forward::forward_warp;
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
    fn bytes(&mut self, n: usize) -> &'a [u8] {
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        s
    }
    fn floats(&mut self, n: usize) -> Vec<f32> {
        self.bytes(n * 4).chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }
}

fn image_from(fmt: u32, b: &[u8], w: usize, h: usize) -> Image {
    let samples = match fmt {
        0 => Samples::U8(b.to_vec()),
        1 => Samples::U16(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()),
        _ => Samples::F32(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
    };
    Image { width: w, height: h, samples }
}

struct Case {
    code: i32,
    x: Image,
    y: Image,
    flow: FlowField,
    want: Image,
}

fn load_cases() -> Vec<Case> {
    let blob = crate::fixture::hex(include_str!("../../tests/fixtures/debug_cases.hex"));
    assert_eq!(&blob[..4], b"DBG1");
    let mut r = Reader { b: &blob, p: 4 };
    let n = r.u32();
    let mut out = Vec::new();
    for _ in 0..n {
        let fmt = r.u32();
        let (w, h, fw, fh) = (r.u32() as usize, r.u32() as usize, r.u32() as usize, r.u32() as usize);
        let code = r.u32() as i32;
        let bytes = [1usize, 2, 4][fmt as usize];
        let x = image_from(fmt, r.bytes(w * h * 4 * bytes), w, h);
        let y = image_from(fmt, r.bytes(w * h * 4 * bytes), w, h);
        let flow = FlowField { width: fw, height: fh, data: r.floats(fw * fh * 2) };
        let want = image_from(fmt, r.bytes(w * h * 4 * bytes), w, h);
        out.push(Case { code, x, y, flow, want });
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

#[test]
fn model_fixtures_are_reproduced_bit_for_bit() {
    // The numpy model predates the Debug view rules read from the original: the Warp view still
    // follows it bit for bit; the other views are checked against their rules in
    // `the_compositions_are_their_primitives_over_the_warp` and the tests below.
    let cases = load_cases();
    assert_eq!(cases.len(), 35);
    let mut per_code = [0usize; 5];
    for (i, c) in cases.iter().enumerate() {
        let got = debug_view(&c.x, &c.y, &c.flow, c.code).unwrap();
        let (g, w) = (bits(&got), bits(&c.want));
        match c.code {
            VIEW_WARP => assert_eq!(g, w, "case {i}: view {} {:?} {}x{}", c.code, c.x.fmt(), c.x.width, c.x.height),
            VIEW_DETAIL | VIEW_EDGE => {
                assert_eq!(g.len(), w.len());
                for (k, gp) in g.chunks(4).enumerate() {
                    assert_eq!(gp[B], gp[G], "case {i}: view {} pixel {k}", c.code);
                }
            }
            _ => {}
        }
        per_code[c.code as usize] += 1;
    }
    assert_eq!(per_code, [7; 5]);
}

#[test]
fn the_warp_view_is_the_forward_warp_of_the_first_frame_at_amount_one() {
    for c in load_cases().iter().filter(|c| c.code == VIEW_WARP) {
        let mut want = Image::new(c.x.width, c.x.height, c.x.fmt());
        forward_warp(&mut want, &c.x, &c.flow, 1.0).unwrap();
        assert_eq!(debug_view(&c.x, &c.y, &c.flow, VIEW_WARP).unwrap(), want);
        // the second frame plays no part in it
        assert_eq!(debug_view(&c.x, &c.x, &c.flow, VIEW_WARP).unwrap(), want);
    }
}

#[test]
fn the_compositions_are_their_primitives_over_the_warp() {
    for c in load_cases() {
        let mut warped = Image::new(c.x.width, c.x.height, c.x.fmt());
        forward_warp(&mut warped, &c.x, &c.flow, 1.0).unwrap();
        let warped = compose_over(&warped, &c.y).unwrap();
        let got = debug_view(&c.x, &c.y, &c.flow, c.code).unwrap();
        match c.code {
            VIEW_BLEND => {
                let mut want = Image::new(c.x.width, c.x.height, c.x.fmt());
                copy(&mut want, &warped);
                lerp(&mut want, &c.y, 0.5);
                assert_eq!(got, want);
            }
            VIEW_DIFF | VIEW_DETAIL | VIEW_EDGE => {
                // opaque; the Diff grey, the overlays cyan
                let white = c.x.fmt().white();
                let px = |img: &Image, i: usize, ch: usize| -> f32 {
                    match &img.samples {
                        Samples::U8(v) => f32::from(v[i * CHANNELS + ch]),
                        Samples::U16(v) => f32::from(v[i * CHANNELS + ch]),
                        Samples::F32(v) => v[i * CHANNELS + ch],
                    }
                };
                for i in 0..c.x.width * c.x.height {
                    assert_eq!(px(&got, i, A), white);
                    if c.code == VIEW_DIFF {
                        // the Diff written out: double precision lumas, single precision gain
                        let luma = |img: &Image| -> f32 {
                            let (r, g, b) = (f64::from(px(img, i, R)), f64::from(px(img, i, G)), f64::from(px(img, i, B)));
                            (((r * 0.29) + (g * 0.59)) + (b * 0.12)) as f32
                        };
                        let gain = (1.0f64 / f64::from(((1.0f64 - f64::from(0.6f32)) as f32).abs())) as f32;
                        let v = (luma(&warped) - luma(&c.y)).abs() * gain;
                        let want = match c.x.fmt() {
                            Fmt::U8 => (f64::from(if v < 255.0 { v } else { 255.0 }) + 0.5) as i64 as f32,
                            Fmt::U16 => (f64::from(if v < 65535.0 { v } else { 65535.0 }) + 0.5) as i64 as f32,
                            Fmt::F32 => {
                                if v < 1.0 {
                                    v
                                } else {
                                    1.0
                                }
                            }
                        };
                        for ch in [R, G, B] {
                            assert_eq!(px(&got, i, ch).to_bits(), want.to_bits(), "pixel {i} channel {ch}");
                        }
                    } else {
                        assert_eq!(px(&got, i, B).to_bits(), px(&got, i, G).to_bits());
                    }
                }
            }
            _ => {}
        }
    }
}

/// A smooth opaque frame of `w x h`, its pattern shifted by `dx`.
fn frame(fmt: Fmt, w: usize, h: usize, dx: f32) -> Image {
    let mut img = Image::new(w, h, fmt);
    let white = fmt.white();
    for y in 0..h {
        for x in 0..w {
            let v = 0.5 + 0.4 * ((x as f32 - dx) * 0.45).sin() * (y as f32 * 0.3).cos();
            let i = (y * w + x) * CHANNELS;
            match &mut img.samples {
                Samples::U8(s) => s[i..i + 4].copy_from_slice(&[255, (v * 255.0) as u8, (v * 200.0) as u8, (v * 100.0) as u8]),
                Samples::U16(s) => s[i..i + 4].copy_from_slice(&[65535, (v * 65535.0) as u16, (v * 50000.0) as u16, (v * 20000.0) as u16]),
                Samples::F32(s) => s[i..i + 4].copy_from_slice(&[white, v, v * 0.8, v * 0.4]),
            }
        }
    }
    img
}

#[test]
fn a_zero_flow_compares_the_two_frames_themselves() {
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let (x, y) = (frame(fmt, 16, 12, 0.0), frame(fmt, 16, 12, 2.0));
        let zero = FlowField::new(8, 6);
        // the warp of a zero flow is a copy of X
        assert_eq!(debug_view(&x, &y, &zero, VIEW_WARP).unwrap(), x);
        // the same frame twice: no difference, and the two overlay channels agree everywhere
        let d = debug_view(&x, &x, &zero, VIEW_DIFF).unwrap();
        assert!(bits(&d).chunks(4).all(|p| p[1..] == [0, 0, 0]), "{fmt:?}");
        for code in [VIEW_DETAIL, VIEW_EDGE] {
            let o = debug_view(&x, &x, &zero, code).unwrap();
            assert!(bits(&o).chunks(4).all(|p| p[1] == p[2] && p[3] == p[2]), "{fmt:?} view {code}");
            // two different frames do not agree everywhere
            let o = debug_view(&x, &y, &zero, code).unwrap();
            assert!(bits(&o).chunks(4).any(|p| p[1] != p[2]), "{fmt:?} view {code}");
        }
    }
}

#[test]
fn a_flow_that_aligns_the_frames_shrinks_the_difference() {
    // Y is X moved 2 px to the right: the flow (2, 0) of X(p) ~ Y(p + flow) warps X onto Y
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let (x, y) = (frame(fmt, 24, 16, 0.0), frame(fmt, 24, 16, 2.0));
        let mut f = FlowField::new(24, 16);
        for p in f.data.chunks_exact_mut(2) {
            p[0] = 2.0;
        }
        let sum = |img: &Image| -> f64 {
            let w = img.width;
            let mut s = 0.0f64;
            for yy in 2..img.height - 2 {
                for xx in 4..w - 4 {
                    let i = (yy * w + xx) * CHANNELS + R;
                    s += match &img.samples {
                        Samples::U8(v) => f64::from(v[i]) / 255.0,
                        Samples::U16(v) => f64::from(v[i]) / 65535.0,
                        Samples::F32(v) => f64::from(v[i]),
                    };
                }
            }
            s
        };
        let aligned = sum(&debug_view(&x, &y, &f, VIEW_DIFF).unwrap());
        let plain = sum(&debug_view(&x, &y, &FlowField::new(24, 16), VIEW_DIFF).unwrap());
        assert!(aligned * 10.0 < plain, "{fmt:?}: aligned {aligned}, unaligned {plain}");
    }
}

#[test]
fn transparent_warp_pixels_show_the_second_frame() {
    // a zero flow over a transparent X: W is transparent everywhere, so W' is Y
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let y = frame(fmt, 16, 12, 1.0);
        let x = Image::new(16, 12, fmt);
        let zero = FlowField::new(16, 12);
        assert_eq!(debug_view(&x, &y, &zero, VIEW_BLEND).unwrap(), y, "{fmt:?}");
        let d = debug_view(&x, &y, &zero, VIEW_DIFF).unwrap();
        assert!(bits(&d).chunks(4).all(|p| p[1..] == [0, 0, 0]), "{fmt:?}");
        for code in [VIEW_DETAIL, VIEW_EDGE] {
            let o = debug_view(&x, &y, &zero, code).unwrap();
            assert!(bits(&o).chunks(4).all(|p| p[1] == p[2] && p[2] == p[3]), "{fmt:?} view {code}");
        }
    }
}

// DIFF8: 1 pixels where the old rule differed (of 1), then others
const DIFF8: [([u8; 4], [u8; 4], u8); 3] = [
    ([255, 41, 62, 91], [255, 44, 53, 83], 13),
    ([255, 74, 77, 66], [255, 73, 89, 77], 20),
    ([255, 68, 69, 68], [255, 71, 79, 70], 18),
];
// DIFF16: 0 pixels where the old rule differed (of 0), then others
const DIFF16: [([u16; 4], [u16; 4], u16); 2] = [
    ([65535, 18901, 19594, 16882], [65535, 18784, 22914, 19866], 5707),
    ([65535, 17533, 17599, 17293], [65535, 18176, 20206, 17960], 4512),
];
// DIFF32: 5 pixels where the old rule differed (of 152), then others
const DIFF32: [([u32; 4], [u32; 4], u32); 7] = [
    ([0x3f800000, 0x3e93ab7a, 0x3e991449, 0x3e83e340], [0x3f800000, 0x3e92c070, 0x3eb302b5, 0x3e9b3453], 0x3db24ff9),
    ([0x3f800000, 0x3e88f985, 0x3e897d93, 0x3e871aa7], [0x3f800000, 0x3e8e017e, 0x3e9ddcd6, 0x3e8c4f31], 0x3d8d0839),
    ([0x3f800000, 0x3e85687f, 0x3e800b9d, 0x3e8aa1fe], [0x3f800000, 0x3e8546d8, 0x3e8ccd44, 0x3e89fd81], 0x3d1438b1),
    ([0x3f800000, 0x3e73f9fd, 0x3e37c4d8, 0x3e77cc2e], [0x3f800000, 0x3e875c27, 0x3e6a9904, 0x3e81de0b], 0x3dc3e29b),
    ([0x3f800000, 0x3e4315cd, 0x3e0a9911, 0x3e762767], [0x3f800000, 0x3e811fb6, 0x3e399a21, 0x3e67f655], 0x3dddbba3),
    ([0x3f800000, 0x3e847cf1, 0x3e64ed89, 0x3e855317], [0x3f800000, 0x3e84014c, 0x3e82bb6e, 0x3e8a337d], 0x3d48de09),
    ([0x3f800000, 0x3e23e252, 0x3e792246, 0x3eb57a2e], [0x3f800000, 0x3e2f7482, 0x3e52eb18, 0x3ea5fa41], 0x3d651d71),
];
const LUMA8: [([u8; 3], u32); 5] = [
    ([74, 77, 66], 0x3e9634ed),
    ([48, 35, 62], 0x3e28b2f0),
    ([66, 74, 62], 0x3e8d07e9),
    ([90, 75, 44], 0x3e97da6a),
    ([95, 68, 31], 0x3e8f5705),
];
const LUMA16: [([u16; 3], u32); 5] = [
    ([18901, 19594, 16882], 0x3e94f7c3),
    ([12486, 8871, 15754], 0x3e27e5e5),
    ([16830, 19139, 15788], 0x3e8d2717),
    ([23132, 19350, 11400], 0x3e984a28),
    ([24470, 17596, 7963], 0x3e900390),
];
const LUMA32: [([u32; 3], u32); 5] = [
    ([0x3e93ab7a, 0x3e991449, 0x3e83e340], 0x3e94f7b0),
    ([0x3e4315cd, 0x3e0a9911, 0x3e762767], 0x3e27e2cd),
    ([0x3e837a6f, 0x3e958491, 0x3e76ae6e], 0x3e8d2525),
    ([0x3eb4b69a, 0x3e972bd1, 0x3e321f38], 0x3e984900),
    ([0x3ebf2ae6, 0x3e897853, 0x3df8d3f1], 0x3e9002c0),
];
// plugin row order: row y-1 first; value is the detail of the centre pixel
const DETAIL: [([u32; 9], u32); 4] = [
    ([0x3e95a831, 0x3e99511f, 0x3e91f7d8, 0x3e823858, 0x3e7cc429, 0x3e6efebe, 0x3e5863f0, 0x3e4839a2, 0x3ed3c3ca], 0x3e680ee1),
    ([0x3e91ba98, 0x3e8e6cd2, 0x3e88b404, 0x3ee7ff5b, 0x3ed80680, 0x3ee0d925, 0x3f1909cc, 0x3f17b940, 0x3f19cddc], 0x3eb5e9d2),
    ([0x3e2716b7, 0x3e0cc483, 0x3e1cb058, 0x3e3eabe0, 0x3e271020, 0x3e3d361a, 0x3eb6bfc2, 0x3e5608b9, 0x3e6aef54], 0x3e61102f),
    ([0x3e374c04, 0x3e6b2ce0, 0x3f1ef870, 0x3e2ccb37, 0x3e93dfbe, 0x3f1fd61e, 0x3e3b310c, 0x3e99c01f, 0x3ef43f6b], 0x3ecb647c),
];
// the measure before the factor 2 of the view
const EDGE: [([u32; 9], u32); 4] = [
    ([0x3e95a831, 0x3e99511f, 0x3e91f7d8, 0x3e823858, 0x3e7cc429, 0x3e6efebe, 0x3e5863f0, 0x3e4839a2, 0x3ed3c3ca], 0x3d208cfa),
    ([0x3e91ba98, 0x3e8e6cd2, 0x3e88b404, 0x3ee7ff5b, 0x3ed80680, 0x3ee0d925, 0x3f1909cc, 0x3f17b940, 0x3f19cddc], 0x3e671a97),
    ([0x3e2716b7, 0x3e0cc483, 0x3e1cb058, 0x3e3eabe0, 0x3e271020, 0x3e3d361a, 0x3eb6bfc2, 0x3e5608b9, 0x3e6aef54], 0x3d9e07f9),
    ([0x3e374c04, 0x3e6b2ce0, 0x3f1ef870, 0x3e2ccb37, 0x3e93dfbe, 0x3f1fd61e, 0x3e3b310c, 0x3e99c01f, 0x3ef43f6b], 0x3e951c49),
];
const OVERLAY8: [(u32, u32, [u8; 4]); 4] = [
    (0x3e9634ed, 0x3ea67d88, [255, 107, 118, 118]),
    (0x3e89b7cc, 0x3e97cafe, [255, 98, 108, 108]),
    (0x3e837418, 0x3e89fa9e, [255, 94, 98, 98]),
    (0x3e7431a2, 0x3e840e4c, [255, 87, 94, 94]),
];
const OVERLAY16: [(u32, u32, [u16; 4]); 4] = [
    (0x3e94f7c3, 0x3ea6cdbb, [65535, 27239, 30501, 30501]),
    (0x3e890ed1, 0x3e972827, [65535, 25062, 27640, 27640]),
    (0x3e82df97, 0x3e8a482e, [65535, 23931, 25285, 25285]),
    (0x3e73e846, 0x3e83ffea, [65535, 22300, 24137, 24137]),
];
const OVERLAY32: [(u32, u32, [u32; 4]); 4] = [
    (0x3e94f7b0, 0x3ea6cc7c, [0x3f800000, 0x3ed4cf8e, 0x3eee48b2, 0x3eee48b2]),
    (0x3e890df8, 0x3e972864, [0x3f800000, 0x3ec3cad0, 0x3ed7f08f, 0x3ed7f08f]),
    (0x3e82deff, 0x3e8a483b, [0x3f800000, 0x3ebaf56d, 0x3ec58bc2, 0x3ec58bc2]),
    (0x3e73e88e, 0x3e83ff61, [0x3f800000, 0x3eae3866, 0x3ebc9166, 0x3ebc9166]),
];
const BLEND8: [([u8; 4], [u8; 4], [u8; 4]); 3] = [
    ([255, 66, 57, 67], [255, 66, 65, 69], [255, 66, 61, 68]),
    ([255, 48, 35, 62], [255, 64, 46, 58], [255, 56, 41, 60]),
    ([255, 29, 40, 79], [255, 37, 27, 63], [255, 33, 34, 71]),
];
const BLEND16: [([u16; 4], [u16; 4], [u16; 4]); 3] = [
    ([65535, 16958, 14651, 17066], [65535, 16896, 16734, 17690], [65535, 16927, 15693, 17378]),
    ([65535, 12486, 8871, 15754], [65535, 16528, 11878, 14846], [65535, 14507, 10375, 15300]),
    ([65535, 7388, 10199, 20104], [65535, 9500, 7050, 16092], [65535, 8444, 8625, 18098]),
];
const BLEND32: [([u32; 4], [u32; 4], [u32; 4]); 3] = [
    ([0x3f800000, 0x3e847cf1, 0x3e64ed89, 0x3e855317], [0x3f800000, 0x3e84014c, 0x3e82bb6e, 0x3e8a337d], [0x3f800000, 0x3e843f1e, 0x3e753232, 0x3e87c34a]),
    ([0x3f800000, 0x3e4315cd, 0x3e0a9911, 0x3e762767], [0x3f800000, 0x3e811fb6, 0x3e399a21, 0x3e67f655], [0x3f800000, 0x3e62aa9c, 0x3e221999, 0x3e6f0ede]),
    ([0x3f800000, 0x3de6dbac, 0x3e1f5acc, 0x3e9d0f8e], [0x3f800000, 0x3e14717c, 0x3ddc5158, 0x3e7b6fe5], [0x3f800000, 0x3e03efa9, 0x3e06c1bc, 0x3e8d63c0]),
];

fn px1(fmt: Fmt, a: [u32; 4]) -> Image {
    let mut img = Image::new(1, 1, fmt);
    match &mut img.samples {
        Samples::U8(v) => v.iter_mut().zip(a).for_each(|(d, s)| *d = s as u8),
        Samples::U16(v) => v.iter_mut().zip(a).for_each(|(d, s)| *d = s as u16),
        Samples::F32(v) => v.iter_mut().zip(a).for_each(|(d, s)| *d = f32::from_bits(s)),
    }
    img
}

fn u32s<T: Copy + Into<u32>>(a: [T; 4]) -> [u32; 4] {
    [a[0].into(), a[1].into(), a[2].into(), a[3].into()]
}

fn plane3(patch: &[u32; 9]) -> Plane {
    Plane { width: 3, height: 3, data: patch.iter().map(|&b| f32::from_bits(b)).collect() }
}

#[test]
fn the_diff_is_the_gain_times_the_luma_difference() {
    // pixel pairs (W', Y) and the value of the original, read from the images of its view
    // generator; the first of DIFF8 and the first five of
    // DIFF32 are pixels where the earlier integer and weight rules gave another value
    for (a, b, want) in DIFF8 {
        let mut out = Image::new(1, 1, Fmt::U8);
        diff(&mut out, &px1(Fmt::U8, u32s(a)), &px1(Fmt::U8, u32s(b)));
        assert_eq!(out.samples, Samples::U8(vec![255, want, want, want]), "{a:?} {b:?}");
    }
    for (a, b, want) in DIFF16 {
        let mut out = Image::new(1, 1, Fmt::U16);
        diff(&mut out, &px1(Fmt::U16, u32s(a)), &px1(Fmt::U16, u32s(b)));
        assert_eq!(out.samples, Samples::U16(vec![65535, want, want, want]), "{a:?} {b:?}");
    }
    for (a, b, want) in DIFF32 {
        let mut out = Image::new(1, 1, Fmt::F32);
        diff(&mut out, &px1(Fmt::F32, a), &px1(Fmt::F32, b));
        assert_eq!(bits(&out), vec![0x3f800000, want, want, want], "{a:x?} {b:x?}");
    }
    // the gain is the narrowing of 1 / (1 - 0.6) taken in double precision: a hair above 2.5
    assert_eq!(diff_gain().to_bits(), 0x4020_0001);
    let mut out = Image::new(1, 1, Fmt::U8);
    diff(&mut out, &px1(Fmt::U8, [255, 0, 0, 0]), &px1(Fmt::U8, [255, 0, 0, 0]));
    assert_eq!(out.samples, Samples::U8(vec![255, 0, 0, 0]));
    // float is capped at 1.0
    let mut out = Image::new(1, 1, Fmt::F32);
    diff(&mut out, &px1(Fmt::F32, [0x3f800000, 0x40800000, 0x40800000, 0x40800000]), &px1(Fmt::F32, [0x3f800000, 0, 0, 0]));
    assert_eq!(bits(&out), vec![0x3f800000; 4]);
}

#[test]
fn the_luma_divides_each_term_in_double_precision() {
    for (rgb, want) in LUMA8 {
        let img = px1(Fmt::U8, [255, rgb[0].into(), rgb[1].into(), rgb[2].into()]);
        assert_eq!(debug_luma(&img).data[0].to_bits(), want, "{rgb:?}");
    }
    for (rgb, want) in LUMA16 {
        let img = px1(Fmt::U16, [65535, rgb[0].into(), rgb[1].into(), rgb[2].into()]);
        assert_eq!(debug_luma(&img).data[0].to_bits(), want, "{rgb:?}");
    }
    for (rgb, want) in LUMA32 {
        let img = px1(Fmt::F32, [0x3f800000, rgb[0], rgb[1], rgb[2]]);
        assert_eq!(debug_luma(&img).data[0].to_bits(), want, "{rgb:x?}");
    }
    // white is capped at 1.0 at 8 and 16 bpc
    assert_eq!(debug_luma(&px1(Fmt::U8, [255, 255, 255, 255])).data[0], 1.0);
    assert_eq!(debug_luma(&px1(Fmt::U16, [65535, 65535, 65535, 65535])).data[0], 1.0);
}

#[test]
fn detail_and_edge_follow_the_arithmetic_of_the_original() {
    // 3 x 3 patches of its luma plane (its row order) and the measure of the centre
    for (patch, want) in DETAIL {
        assert_eq!(detail(&plane3(&patch))[4].to_bits(), want, "{patch:x?}");
    }
    for (patch, want) in EDGE {
        let e = edge(&plane3(&patch))[4];
        assert_eq!(e.to_bits(), (2.0f32 * f32::from_bits(want)).to_bits(), "{patch:x?}");
    }
    // the border of the detail is the luma itself; inside sqrt(S * 0.125)
    let mut l = Plane::new(4, 3);
    l.data[5] = 0.25;
    let d = detail(&l);
    assert_eq!(d[0], 0.0);
    assert_eq!(d[5], (2.0f32 * 0.125).sqrt());
    assert_eq!(d[6], (0.25f32 * 0.125).sqrt());
    let mut b = Plane::new(3, 3);
    b.data.iter_mut().for_each(|v| *v = 0.35);
    assert!(detail(&b).iter().enumerate().all(|(i, &v)| v == if i == 4 { 0.0 } else { 0.35 }));
    // a flat plane has no edge up to the rounding of the single precision sums (their order is the
    // original's, so 0.35 does not cancel exactly); a step of 1 gives sqrt(16) * 0.17677669529664,
    // times the view's 2
    assert!(edge(&b).iter().all(|&v| v < 1e-6));
    let mut s = Plane::new(4, 3);
    for y in 0..3 {
        s.data[y * 4 + 2] = 1.0;
        s.data[y * 4 + 3] = 1.0;
    }
    let e = edge(&s);
    assert_eq!(e[1], 2.0 * ((4.0f64 * 0.17677669529664) as f32));
    assert_eq!(e[0], 0.0);
    // the border is repeated outwards: a step between the two columns of a 2 x 2 plane is seen
    // by both rows with the same weight as in the middle of a taller plane
    let mut c = Plane::new(2, 2);
    c.data.copy_from_slice(&[0.0, 1.0, 0.0, 1.0]);
    let e = edge(&c);
    assert_eq!(e[0], 2.0 * ((4.0f64 * 0.17677669529664) as f32));
    assert_eq!(e[0], e[1]);
}

#[test]
fn the_overlay_scales_by_the_gain_and_stores_per_depth() {
    for (r, g, want) in OVERLAY8 {
        let mut out = Image::new(1, 1, Fmt::U8);
        overlay(&mut out, &[f32::from_bits(r)], &[f32::from_bits(g)]);
        assert_eq!(out.samples, Samples::U8(want.to_vec()), "{r:x} {g:x}");
    }
    for (r, g, want) in OVERLAY16 {
        let mut out = Image::new(1, 1, Fmt::U16);
        overlay(&mut out, &[f32::from_bits(r)], &[f32::from_bits(g)]);
        assert_eq!(out.samples, Samples::U16(want.to_vec()), "{r:x} {g:x}");
    }
    for (r, g, want) in OVERLAY32 {
        let mut out = Image::new(1, 1, Fmt::F32);
        overlay(&mut out, &[f32::from_bits(r)], &[f32::from_bits(g)]);
        assert_eq!(bits(&out), want.to_vec(), "{r:x} {g:x}");
    }
    // 1 / (1 - 0.3) with the quotient taken in double precision
    assert_eq!(overlay_gain(), 1.4285715f32);
    // a measure above 1 / gain saturates at 1, a negative one at 0
    let mut out = Image::new(1, 1, Fmt::U8);
    overlay(&mut out, &[0.9], &[-0.5]);
    assert_eq!(out.samples, Samples::U8(vec![255, 255, 0, 0]));
}

#[test]
fn the_blend_view_is_a_lerp_with_the_weight_on_the_warped_frame() {
    for (a, b, want) in BLEND8 {
        let mut out = Image::new(1, 1, Fmt::U8);
        copy(&mut out, &px1(Fmt::U8, u32s(a)));
        lerp(&mut out, &px1(Fmt::U8, u32s(b)), BLEND_WEIGHT);
        assert_eq!(out.samples, Samples::U8(want.to_vec()), "{a:?} {b:?}");
    }
    for (a, b, want) in BLEND16 {
        let mut out = Image::new(1, 1, Fmt::U16);
        copy(&mut out, &px1(Fmt::U16, u32s(a)));
        lerp(&mut out, &px1(Fmt::U16, u32s(b)), BLEND_WEIGHT);
        assert_eq!(out.samples, Samples::U16(want.to_vec()), "{a:?} {b:?}");
    }
    for (a, b, want) in BLEND32 {
        let mut out = Image::new(1, 1, Fmt::F32);
        copy(&mut out, &px1(Fmt::F32, a));
        lerp(&mut out, &px1(Fmt::F32, b), BLEND_WEIGHT);
        assert_eq!(bits(&out), want.to_vec(), "{a:x?} {b:x?}");
    }
}

#[test]
fn every_other_code_is_edge_compare() {
    let c = &load_cases()[3];
    let edge = debug_view(&c.x, &c.y, &c.flow, VIEW_EDGE).unwrap();
    for code in [-1, 5, 9] {
        assert_eq!(debug_view(&c.x, &c.y, &c.flow, code).unwrap(), edge, "code {code}");
    }
}

#[test]
fn tiny_frames_show_the_zero_vector_and_shapes_must_match() {
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        for (w, h) in [(1usize, 1usize), (1, 17), (17, 1)] {
            let (x, y) = (frame(fmt, w, h, 0.0), frame(fmt, w, h, 1.0));
            for code in 0..=4 {
                let got = debug_view(&x, &y, &FlowField::new(w, h), code).unwrap();
                assert_eq!(got, crate::vectors::zero_vectors_image(w, h, fmt), "{fmt:?} {w}x{h} view {code}");
            }
        }
        let other = frame(fmt, 5, 4, 0.0);
        assert!(matches!(debug_view(&frame(fmt, 4, 4, 0.0), &other, &FlowField::new(4, 4), 0), Err(FxError::Mismatch(_))));
    }
}

#[test]
fn thread_count_does_not_change_the_views() {
    let pool = |n: usize| rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
    for c in load_cases() {
        let one = pool(1).install(|| debug_view(&c.x, &c.y, &c.flow, c.code).unwrap());
        for n in [4, 16] {
            assert_eq!(bits(&pool(n).install(|| debug_view(&c.x, &c.y, &c.flow, c.code).unwrap())), bits(&one));
        }
    }
}

// ---- the alpha match of a one layer group's Debug display (read in CreateMotionVectors_64)

fn one_px(fmt: crate::image::Fmt, p: [f64; 4]) -> Image {
    let mut img = Image::new(1, 1, fmt);
    match &mut img.samples {
        Samples::U8(v) => v.iter_mut().zip(p).for_each(|(d, s)| *d = s as u8),
        Samples::U16(v) => v.iter_mut().zip(p).for_each(|(d, s)| *d = s as u16),
        Samples::F32(v) => v.iter_mut().zip(p).for_each(|(d, s)| *d = s as f32),
    }
    img
}

#[test]
fn match_alpha_brings_the_larger_alpha_down() {
    use crate::image::Fmt;
    // 8 bpc, alpha 255 against 128: a2 = 128 / 255; R = trunc((200 / 1) * 0.50196) = 100, G trunc(50 * 0.50196) = 25
    let (mut a, mut b) = (one_px(Fmt::U8, [255.0, 200.0, 50.0, 7.0]), one_px(Fmt::U8, [128.0, 90.0, 80.0, 70.0]));
    match_alpha(&mut a, &mut b);
    assert_eq!(a, one_px(Fmt::U8, [128.0, 100.0, 25.0, 3.0]));
    assert_eq!(b, one_px(Fmt::U8, [128.0, 90.0, 80.0, 70.0]));
    // the other way round scales the second image
    let (mut a, mut b) = (one_px(Fmt::U8, [128.0, 90.0, 80.0, 70.0]), one_px(Fmt::U8, [255.0, 200.0, 50.0, 7.0]));
    match_alpha(&mut a, &mut b);
    assert_eq!(b, one_px(Fmt::U8, [128.0, 100.0, 25.0, 3.0]));
    // a transparent pixel clears both; equal alphas change nothing
    let (mut a, mut b) = (one_px(Fmt::U8, [0.0, 9.0, 9.0, 9.0]), one_px(Fmt::U8, [200.0, 90.0, 80.0, 70.0]));
    match_alpha(&mut a, &mut b);
    assert_eq!((a, b), (one_px(Fmt::U8, [0.0; 4]), one_px(Fmt::U8, [0.0; 4])));
    let (mut a, mut b) = (one_px(Fmt::U16, [300.0, 9.0, 9.0, 9.0]), one_px(Fmt::U16, [300.0, 90.0, 80.0, 70.0]));
    match_alpha(&mut a, &mut b);
    assert_eq!((a, b), (one_px(Fmt::U16, [300.0, 9.0, 9.0, 9.0]), one_px(Fmt::U16, [300.0, 90.0, 80.0, 70.0])));
    // float: no truncation
    let (mut a, mut b) = (one_px(Fmt::F32, [1.0, 0.8, 0.2, 0.1]), one_px(Fmt::F32, [0.5, 0.3, 0.3, 0.3]));
    match_alpha(&mut a, &mut b);
    let s = |c: f32| (c / 1.0f32) * 0.5f32;
    assert_eq!(a, one_px(Fmt::F32, [0.5, s(0.8) as f64, s(0.2) as f64, s(0.1) as f64]));
}

#[test]
fn views_of_is_the_composition_part_of_debug_view() {
    use crate::image::Fmt;
    use crate::stages::Stages;
    let mk = |s: f64| {
        let mut img = Image::new(5, 4, Fmt::U16);
        if let Samples::U16(v) = &mut img.samples {
            for (i, px) in v.chunks_exact_mut(4).enumerate() {
                px.copy_from_slice(&[65535, ((i as f64 * 1234.5 + s) % 65535.0) as u16, ((i as f64 * 777.0 + s) % 65535.0) as u16, 999]);
            }
        }
        img
    };
    let (x, y) = (mk(0.0), mk(5000.0));
    let flow = FlowField::new(5, 4);
    for code in [VIEW_BLEND, VIEW_DIFF, VIEW_DETAIL, VIEW_EDGE] {
        let mut w = Image::new(5, 4, Fmt::U16);
        crate::stages::Cpu.forward_warp(&mut w, &x, &flow, 1.0).unwrap();
        let w = compose_over(&w, &y).unwrap();
        assert_eq!(views_of(&w, &y, code), debug_view(&x, &y, &flow, code).unwrap(), "view {code}");
    }
}

/// One row of float pixels `[A, R, G, B]` from bit patterns.
fn f32_row(px: &[[u32; 4]]) -> Image {
    Image { width: px.len(), height: 1, samples: Samples::F32(px.iter().flat_map(|p| p.map(f32::from_bits)).collect()) }
}

// Two memory rows of frame 0 of `create-vectors.mix-dbg.specmix-6x4.d32.20503049` (Debug Blend of the special value
// clip), TRACED in `CreateMotionVectors_64`: the frame `Y` and the forward warp `W` handed
// to the two OVER calls of the Debug branch, the `W'` they leave, and the Blend view the lerp makes of `W'` and `Y`.
const SPEC_Y: [[[u32; 4]; 6]; 2] = [
    [
        [0xff800000, 0xffc00000, 0xff800000, 0xff800000],
        [0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000],
        [0xffc00000, 0xffc00000, 0xffc00000, 0xffc00000],
        [0x00000000, 0x00000000, 0x00000000, 0xffc00000],
        [0x80000000, 0x80000000, 0x80000000, 0x7fc00000],
        [0x00000001, 0x00000001, 0x7f800000, 0xffc00000],
    ],
    [
        [0x007fffff, 0x011fffff, 0xff800000, 0x00000000],
        [0x00800000, 0x407fffff, 0x7fc00000, 0x80000000],
        [0x3f800000, 0x7f800000, 0xffc00000, 0x00000001],
        [0x40200000, 0xff800000, 0x00000000, 0x011fffff],
        [0x7f7fffff, 0x7fc00000, 0x80000000, 0x407fffff],
        [0x7f800000, 0xffc00000, 0x7f800000, 0x7f800000],
    ],
];
const SPEC_W: [[[u32; 4]; 6]; 2] = [
    [
        [0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000],
        [0xffc00000, 0xffc00000, 0xffc00000, 0xffc00000],
        [0x00000000, 0x00000000, 0x00000000, 0xffc00000],
        [0x80000000, 0x80000000, 0x80000000, 0x7fc00000],
        [0x00000001, 0x00000001, 0x7f800000, 0xffc00000],
        [0x007fffff, 0x011fffff, 0xff800000, 0x00000000],
    ],
    [
        [0x00800000, 0x407fffff, 0x7fc00000, 0x80000000],
        [0x3f800000, 0x7f800000, 0xffc00000, 0x00000001],
        [0x40200000, 0xff800000, 0x00000000, 0x011fffff],
        [0x7f7fffff, 0x7fc00000, 0x80000000, 0x407fffff],
        [0x7f800000, 0xffc00000, 0x7f800000, 0x7f800000],
        [0xff800000, 0xffc00000, 0xff800000, 0xff800000],
    ],
];
const SPEC_WP: [[[u32; 4]; 6]; 2] = [
    [
        [0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000],
        [0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000],
        [0xffc00000, 0xffc00000, 0xffc00000, 0xffc00000],
        [0x00000000, 0x00000000, 0x00000000, 0x00000000],
        [0x00000001, 0x00000001, 0x7f800000, 0xffc00000],
        [0x00800000, 0x01200000, 0xffc00000, 0xffc00000],
    ],
    [
        [0x00ffffff, 0x407fffff, 0x7fc00000, 0x00000000],
        [0x3f800000, 0x7f800000, 0xffc00000, 0x00000001],
        [0x3f800000, 0xff800000, 0xffc00000, 0x011ffffe],
        [0xff800000, 0x7fc00000, 0x00000000, 0xc0bffffc],
        [0xff800000, 0x7fc00000, 0xffc00000, 0xffc00000],
        [0x7f800000, 0xffc00000, 0xffc00000, 0xffc00000],
    ],
];
const SPEC_BLEND: [[[u32; 4]; 6]; 2] = [
    [[0; 4], [0; 4], [0; 4], [0; 4], [0; 4], [0x00400001, 0x00a00001, 0xffc00000, 0xffc00000]],
    [
        [0x00bfffff, 0x3fffffff, 0x7fc00000, 0x00000000],
        [0x3f000000, 0x7f800000, 0x7fc00000, 0x00000000],
        [0x3f800000, 0xffc00000, 0xffc00000, 0x009fffff],
        [0; 4],
        [0; 4],
        [0; 4],
    ],
];

#[cfg(feature = "emulator-nan")]
#[test]
fn w_prime_is_y_then_w_over_a_transparent_image_on_traced_special_values() {
    for row in 0..2 {
        let (y, w) = (f32_row(&SPEC_Y[row]), f32_row(&SPEC_W[row]));
        let wp = compose_over(&w, &y).unwrap();
        assert_eq!(bits(&wp), bits(&f32_row(&SPEC_WP[row])), "W' row {row}");
        assert_eq!(bits(&views_of(&wp, &y, VIEW_BLEND)), bits(&f32_row(&SPEC_BLEND[row])), "Blend row {row}");
    }
    // the earlier rule (W where its alpha is not 0, else Y) is not what the plugin composes here
    let mut filled = f32_row(&SPEC_W[1]);
    fill_transparent(&mut filled, &f32_row(&SPEC_Y[1]));
    assert_ne!(bits(&filled), bits(&f32_row(&SPEC_WP[1])));
}

// The float luma planes of the Detail view the plugin made of the two `W'` rows above (TRACED, the luma routine of
// frame 0 of `create-vectors.mix-dbg.specmix-6x4.d32.e7ab55b9`).
const SPEC_WP_LUMA: [[u32; 6]; 2] = [
    [0x7fc00000, 0x7fc00000, 0xffc00000, 0x00000000, 0xffc00000, 0xffc00000],
    [0x7fc00000, 0xffc00000, 0xffc00000, 0x7fc00000, 0x7fc00000, 0xffc00000],
];

#[cfg(feature = "emulator-nan")]
#[test]
fn the_float_luma_picks_a_nan_by_the_nan_rule() {
    for row in 0..2 {
        let l = debug_luma(&f32_row(&SPEC_WP[row]));
        assert_eq!(l.data.iter().map(|v| v.to_bits()).collect::<Vec<u32>>(), SPEC_WP_LUMA[row].to_vec(), "row {row}");
    }
    // a negative and a positive NaN term: the positive one, whatever the order of the terms
    let (pos, neg) = (f32::from_bits(0x7fc0_0000), f32::from_bits(0xffc0_0000));
    for (r, g, b) in [(neg, pos, 1.0f32), (pos, neg, 1.0), (neg, 1.0, pos), (1.0, neg, pos), (neg, neg, pos)] {
        assert_eq!(float_luma(r, g, b).to_bits(), 0x7fc0_0000, "{:08x} {:08x} {:08x}", r.to_bits(), g.to_bits(), b.to_bits());
    }
    assert_eq!(float_luma(neg, neg, 1.0).to_bits(), 0xffc0_0000);
    // numbers are the plain double precision sum
    for (r, g, b) in [(0.25f32, 0.5f32, 0.75f32), (f32::MAX, f32::MAX, -1.0), (1.0e-40, -3.0, 7.5)] {
        let want = (((f64::from(r) * 0.29) + (f64::from(g) * 0.59)) + (f64::from(b) * 0.12)) as f32;
        assert_eq!(float_luma(r, g, b).to_bits(), want.to_bits());
    }
}

#[test]
fn w_prime_with_alphas_of_zero_and_white_is_w_where_opaque_else_y() {
    // the opaque and transparent pixels of every depth: the OVER chain equals the earlier rule
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let y = frame(fmt, 7, 5, 1.0);
        let mut w = frame(fmt, 7, 5, 3.0);
        let white = fmt.white();
        match &mut w.samples {
            Samples::U8(s) => s.chunks_exact_mut(CHANNELS).step_by(3).for_each(|p| p.fill(0)),
            Samples::U16(s) => s.chunks_exact_mut(CHANNELS).step_by(3).for_each(|p| p.fill(0)),
            Samples::F32(s) => s.chunks_exact_mut(CHANNELS).step_by(3).for_each(|p| p.fill(0.0)),
        }
        let mut want = w.clone();
        fill_transparent(&mut want, &y);
        assert_eq!(compose_over(&w, &y).unwrap(), want, "{fmt:?} white {white}");
    }
    // shapes must agree
    assert!(compose_over(&frame(Fmt::U8, 4, 4, 0.0), &frame(Fmt::U8, 5, 4, 0.0)).is_err());
}
