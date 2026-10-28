//! The Debug views (rule G4) against the numpy model of `tests/fixtures/gen_debug.py`, the Warp
//! view against the forward warp, the compositions against their primitives, tiny frames and the
//! thread count.

use super::*;
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
    let cases = load_cases();
    assert_eq!(cases.len(), 35);
    let mut per_code = [0usize; 5];
    for (i, c) in cases.iter().enumerate() {
        let got = debug_view(&c.x, &c.y, &c.flow, c.code).unwrap();
        assert_eq!(bits(&got), bits(&c.want), "case {i}: view {} {:?} {}x{}", c.code, c.x.fmt(), c.x.width, c.x.height);
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
        let got = debug_view(&c.x, &c.y, &c.flow, c.code).unwrap();
        match c.code {
            VIEW_BLEND => {
                let mut want = Image::new(c.x.width, c.x.height, c.x.fmt());
                blend(&mut want, &warped, &c.y, 0.5);
                assert_eq!(got, want);
            }
            VIEW_DIFF | VIEW_DETAIL | VIEW_EDGE => {
                // opaque, and blue is 0 in the overlays
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
                        for ch in [R, G, B] {
                            assert_eq!(px(&got, i, ch), (px(&warped, i, ch) - px(&c.y, i, ch)).abs(), "pixel {i} channel {ch}");
                        }
                    } else {
                        assert_eq!(px(&got, i, B), 0.0);
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
            assert!(bits(&o).chunks(4).all(|p| p[1] == p[2] && p[3] == 0), "{fmt:?} view {code}");
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
