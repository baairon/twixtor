//! Tests of the forward warp: hand goldens, the numpy float32 model fixture, thread invariance,
//! the compositions against explicit call sequences and a ground truth scene.

use super::mesh::{mip_level, position_row, Grid};
use super::pyramid::level_count;
use super::raster::{triangle, Sink, Vtx};
use super::*;
use crate::holefill::fill_holes;
use crate::image::{Fmt, Image, Samples};
use crate::smartblend::SmartOpts;

// ------------------------------------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------------------------------------

fn const_img(w: usize, h: usize, fmt: Fmt, px: [f32; 4]) -> Image {
    let mut img = Image::new(w, h, fmt);
    match &mut img.samples {
        Samples::U8(d) => d.chunks_exact_mut(4).for_each(|p| (0..4).for_each(|k| p[k] = px[k] as u8)),
        Samples::U16(d) => d.chunks_exact_mut(4).for_each(|p| (0..4).for_each(|k| p[k] = px[k] as u16)),
        Samples::F32(d) => d.chunks_exact_mut(4).for_each(|p| p.copy_from_slice(&px)),
    }
    img
}

fn const_flow(w: usize, h: usize, v: [f32; 2]) -> FlowField {
    let mut f = FlowField::new(w, h);
    for y in 0..h {
        for x in 0..w {
            f.set(x, y, v);
        }
    }
    f
}

/// A deterministic textured image of any depth (values spread over the whole range).
fn textured(w: usize, h: usize, fmt: Fmt, seed: u32) -> Image {
    let mut img = Image::new(w, h, fmt);
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s >> 8) as f32 / (1u32 << 24) as f32
    };
    let n = w * h * 4;
    let vals: Vec<f32> = (0..n)
        .map(|i| {
            let p = i / 4;
            let (x, y) = ((p % w) as f32, (p / w) as f32);
            (0.5 + 0.3 * (x * 0.7 + 0.2 * (i % 4) as f32).sin() + 0.15 * (y * 0.5).cos() + 0.05 * next()).clamp(0.0, 1.0)
        })
        .collect();
    match &mut img.samples {
        Samples::U8(d) => d.iter_mut().zip(&vals).for_each(|(o, &v)| *o = (v * 255.0) as u8),
        Samples::U16(d) => d.iter_mut().zip(&vals).for_each(|(o, &v)| *o = (v * 65535.0) as u16),
        Samples::F32(d) => d.copy_from_slice(&vals),
    }
    img
}

fn u8_at(img: &Image, x: usize, y: usize, k: usize) -> i32 {
    match &img.samples {
        Samples::U8(d) => d[(y * img.width + x) * 4 + k] as i32,
        _ => panic!("u8 only"),
    }
}

fn bytes_of(img: &Image) -> Vec<u8> {
    match &img.samples {
        Samples::U8(d) => d.clone(),
        Samples::U16(d) => d.iter().flat_map(|v| v.to_le_bytes()).collect(),
        Samples::F32(d) => d.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect(),
    }
}

fn in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(f)
}

// ------------------------------------------------------------------------------------------------
// hand goldens
// ------------------------------------------------------------------------------------------------

#[test]
fn translation_of_a_constant_image_covers_the_expected_pixels() {
    // flow (2, 0), amount 1: x' = c + 2 (column 0 unshifted, the others + 1e-4), y' = r (+ 1e-4 for r >= 1).
    // The column 0 vertices sit exactly at x = 2 and spans are closed, so pixels 2..w-1 are covered and 0, 1
    // are holes; every row is covered (the strip between the rows 0 and 1 covers the scanlines 0 and 1,
    // the strip between r and r + 1 covers r + 1).
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let px = match fmt {
            Fmt::U8 => [255.0, 200.0, 100.0, 50.0],
            Fmt::U16 => [65535.0, 51000.0, 25000.0, 12800.0],
            Fmt::F32 => [1.0, 0.8, 0.4, 0.2],
        };
        let src = const_img(10, 8, fmt, px);
        let flow = const_flow(10, 8, [2.0, 0.0]);
        let mut out = Image::new(10, 8, fmt);
        let cov = forward_warp_coverage(&mut out, &src, &flow, 1.0).unwrap();
        let mut expected = Image::new(10, 8, fmt);
        for y in 0..8 {
            for x in 0..10 {
                let covered = x >= 2;
                assert_eq!(cov.at(x, y), if covered { 255 } else { 0 }, "{fmt:?} coverage at {x},{y}");
                if covered {
                    match &mut expected.samples {
                        Samples::U8(d) => (0..4).for_each(|k| d[(y * 10 + x) * 4 + k] = px[k] as u8),
                        Samples::U16(d) => (0..4).for_each(|k| d[(y * 10 + x) * 4 + k] = px[k] as u16),
                        Samples::F32(d) => (0..4).for_each(|k| d[(y * 10 + x) * 4 + k] = px[k]),
                    }
                }
            }
        }
        match (&out.samples, &expected.samples) {
            (Samples::F32(a), Samples::F32(b)) => {
                for (i, (&p, &q)) in a.iter().zip(b).enumerate() {
                    assert!((p - q).abs() <= 1e-6, "f32 sample {i}: {p} against {q}");
                }
            }
            _ => assert_eq!(out, expected, "{fmt:?}"),
        }
    }
}

#[test]
fn vertical_translation_leaves_the_top_rows_as_holes() {
    // flow (0, 3): the row 0 vertices are at y = 3 exactly; strip (0, 1) covers 3 and 4 (the row 1 vertices
    // are at 4.0001), strip (r, r + 1) covers r + 4; rows 0..2 are holes, 3..7 covered.
    let src = const_img(8, 8, Fmt::U8, [255.0, 10.0, 20.0, 30.0]);
    let flow = const_flow(8, 8, [0.0, 3.0]);
    let mut out = Image::new(8, 8, Fmt::U8);
    let cov = forward_warp_coverage(&mut out, &src, &flow, 1.0).unwrap();
    for y in 0..8 {
        for x in 0..8 {
            assert_eq!(cov.at(x, y), if y >= 3 { 255 } else { 0 }, "{x},{y}");
            assert_eq!(u8_at(&out, x, y, 0), if y >= 3 { 255 } else { 0 });
        }
    }
}

#[test]
fn flow_beyond_the_image_gives_no_coverage_and_clips_at_the_edge() {
    let src = const_img(10, 8, Fmt::U8, [255.0, 1.0, 2.0, 3.0]);
    // everything lands right of the image: every triangle is rejected (all x > w1)
    let mut out = Image::new(10, 8, Fmt::U8);
    let cov = forward_warp_coverage(&mut out, &src, &const_flow(10, 8, [50.0, 0.0]), 1.0).unwrap();
    assert!(cov.data.iter().all(|&c| c == 0));
    assert_eq!(out, Image::new(10, 8, Fmt::U8));
    // a flow of -3: pixels 0..=w-4 are covered, the last three columns are holes (the right vertices are at
    // x = 6 + 1e-4 for the rows >= 1 and exactly 6 for the row 0: the pixel 6 is covered by the closed spans)
    let cov = forward_warp_coverage(&mut out, &src, &const_flow(10, 8, [-3.0, 0.0]), 1.0).unwrap();
    for y in 0..8 {
        for x in 0..10 {
            assert_eq!(cov.at(x, y) != 0, x <= 6, "{x},{y}");
        }
    }
}

#[test]
fn early_outs_copy_the_source_and_report_full_coverage() {
    let src = textured(9, 7, Fmt::U8, 3);
    let flow = const_flow(9, 7, [1.0, 0.5]);
    // a zero flow
    let mut out = Image::new(9, 7, Fmt::U8);
    let cov = forward_warp_coverage(&mut out, &src, &const_flow(9, 7, [0.0, 0.0]), 1.0).unwrap();
    assert_eq!(out, src);
    assert!(cov.data.iter().all(|&c| c == 255));
    // |amount| below 1e-5 (the comparison is a double compare of the widened float: the float nearest to
    // 1e-5 is 9.99999974737875e-06 < 1e-05, so it is an early out, 1.1e-5 is not)
    for a in [0.0f32, 1e-6, -1e-6, 1e-5] {
        let mut o = Image::new(9, 7, Fmt::U8);
        forward_warp(&mut o, &src, &flow, a).unwrap();
        assert_eq!(o, src, "amount {a}");
    }
    let mut o = Image::new(9, 7, Fmt::U8);
    forward_warp(&mut o, &src, &flow, 1.1e-5).unwrap();
    assert_ne!(o, src);
    // the flow threshold: components of 0.0001f (9.99999974737875e-05 < 1e-4 as a double) still count as zero
    let mut o = Image::new(9, 7, Fmt::U8);
    forward_warp(&mut o, &src, &const_flow(9, 7, [1e-4, -1e-4]), 1.0).unwrap();
    assert_eq!(o, src);
    forward_warp(&mut o, &src, &const_flow(9, 7, [1.0001e-4, 0.0]), 1.0).unwrap();
    assert_ne!(o, src);
    // one large component anywhere defeats the zero test
    let mut f = const_flow(9, 7, [0.0, 0.0]);
    f.set(8, 6, [0.0, 0.5]);
    forward_warp(&mut o, &src, &f, 1.0).unwrap();
    assert_ne!(o, src);
}

#[test]
fn tie_epsilon_is_a_double_addition_on_rows_and_columns_but_the_first() {
    // zero flow vertices: x' = c, y' = r; the epsilon is added to the columns 1.. of the rows 1.. only
    let flow = const_flow(5, 4, [0.0, 0.0]);
    let g = Grid::new(5, 4, &flow, 1.0);
    let mut row = vec![[0.0f32; 2]; 5];
    position_row(&g, &flow, 0, &mut row);
    for (c, p) in row.iter().enumerate() {
        assert_eq!(*p, [c as f32, 0.0], "row 0 is not shifted");
    }
    position_row(&g, &flow, 2, &mut row);
    assert_eq!(row[0], [0.0, 2.0], "column 0 is not shifted");
    for c in 1..5 {
        assert_eq!(row[c], [((c as f64) + 0.0001) as f32, ((2.0f64) + 0.0001) as f32]);
    }
    assert_eq!(row[1][0].to_bits(), 1.0001f32.to_bits());
    // the addition is in double: for a tiny coordinate the double sum and the float sum differ in the last
    // bits (the numpy search of the fixture generator found 1.5424167941091582e-05)
    let tiny = 1.5424167941091582e-05f32;
    assert_eq!(tiny.to_bits(), 931226406);
    let double = ((tiny as f64) + 0.0001f64) as f32;
    let single = tiny + 0.0001f32;
    assert_ne!(double.to_bits(), single.to_bits());
    assert_eq!(double, 0.0001154241690528579f32);
    // through the real row builder: a flow that puts a vertex at that coordinate
    let mut f2 = const_flow(5, 4, [0.0, 0.0]);
    for y in 0..4 {
        for x in 0..5 {
            f2.set(x, y, [tiny - x as f32, 0.0]); // x' = fx * 1 + c = tiny (up to rounding of the sum)
        }
    }
    let g2 = Grid::new(5, 4, &f2, 1.0);
    position_row(&g2, &f2, 1, &mut row);
    let base = (tiny - 1.0f32) * 1.0f32 + 1.0f32;
    assert_eq!(row[1][0], ((base as f64) + 0.0001f64) as f32);
}

#[test]
fn pyramid_level_counts_follow_the_halving_rule() {
    // 1 + the number of times both sizes halve to at least 2
    assert_eq!(level_count(2, 2), 1);
    assert_eq!(level_count(3, 3), 1);
    assert_eq!(level_count(4, 4), 2);
    assert_eq!(level_count(8, 8), 3);
    assert_eq!(level_count(16, 12), 3);
    assert_eq!(level_count(5, 100), 2);
    assert_eq!(level_count(3, 100), 1);
    assert_eq!(level_count(1920, 1080), 10);
}

#[test]
fn mip_level_of_a_uniform_squeeze_matches_the_closed_form() {
    // flow fx = -0.5 x, fy = 0, amount 1: x' = x / 2 and y' = y. Interior vertex: the four axis neighbours are at
    // distances 0.5 (left, right) and 1 (up, down), the diagonals at sqrt(1.25):
    // m = (4 sqrt(1.25) / sqrt(2) + 2 * 1 + 2 * 0.5) / 8 and d = -log2(m).
    let (w, h) = (16, 12);
    let mut flow = FlowField::new(w, h);
    for y in 0..h {
        for x in 0..w {
            flow.set(x, y, [-0.5 * x as f32, 0.0]);
        }
    }
    let g = Grid::new(w, h, &flow, 1.0);
    let m = (4.0f64 * 1.25f64.sqrt() * 0.5f64.sqrt() + 2.0 + 1.0) / 8.0;
    let want = -m.log2();
    for (c, r) in [(5, 5), (3, 7), (10, 2)] {
        let d = mip_level(&g, &flow, c, r) as f64;
        assert!((d - want).abs() < 2e-6, "d {d} against {want} at {c},{r}");
    }
    assert!((want - 0.3766).abs() < 1e-3);
    // a translation by whole pixels is a level 0 vertex: |d| is a few ulps of the log
    let t = const_flow(w, h, [2.0, 1.0]);
    let g = Grid::new(w, h, &t, 1.0);
    for (c, r) in [(0, 0), (5, 5), (15, 11), (0, 11)] {
        assert!(mip_level(&g, &t, c, r).abs() < 3e-7, "{c},{r}");
    }
    // an expansion has a negative level
    let mut e = FlowField::new(w, h);
    for y in 0..h {
        for x in 0..w {
            e.set(x, y, [x as f32, y as f32]);
        }
    }
    let g = Grid::new(w, h, &e, 1.0);
    assert!((mip_level(&g, &e, 5, 5) + 1.0).abs() < 1e-6, "doubling is one level up: -log2(2)");
}

struct Rec {
    hits: Vec<(usize, usize, f32, f32, f32)>,
    outside: bool,
}

impl Sink for Rec {
    fn hit(&mut self, x: usize, y: usize, u: f32, v: f32, d: f32) {
        self.hits.push((x, y, u, v, d));
    }
    fn outside(&mut self) {
        self.outside = true;
    }
}

fn vt(x: f32, y: f32, u: f32, v: f32, d: f32) -> Vtx {
    Vtx { x, y, u, v, d }
}

#[test]
fn right_triangle_scan_conversion_by_hand() {
    // (0,0) (4,0) (0,4): rows 0..3 have 5, 4, 3, 2 pixels, row 4 is the end of the triangle (the edge
    // test is `y < y_b`) and gives nothing
    let mut rec = Rec { hits: vec![], outside: false };
    triangle(&[vt(0.0, 0.0, 0.0, 0.0, 0.0), vt(4.0, 0.0, 1.0, 0.0, 0.0), vt(0.0, 4.0, 0.0, 1.0, 0.0)], 7, 7, &mut rec);
    let mut counts = [0usize; 8];
    for h in &rec.hits {
        counts[h.1] += 1;
    }
    assert_eq!(counts, [5, 4, 3, 2, 0, 0, 0, 0]);
    // row 1: the pixels 0..=3, the right edge (4,0) -> (0,4) is at x = 3, u = 0.75, v = 0.25: u(x) = 0.75 x / 3
    let row1: Vec<_> = rec.hits.iter().filter(|h| h.1 == 1).collect();
    assert_eq!(row1.iter().map(|h| h.0).collect::<Vec<_>>(), vec![0, 1, 2, 3]);
    for h in &row1 {
        assert!((h.2 - 0.25 * h.0 as f32).abs() < 1e-6, "u at {}", h.0);
        assert!((h.3 - 0.25).abs() < 1e-6);
    }
    assert!(!rec.outside);
}

#[test]
fn degenerate_last_row_pass() {
    // both lower vertices exactly on the last row (y = h1 = 7): the main pass never reaches that row (the
    // edges end at y = 7) and the final pass writes the pixels between them: ceil(3.2) = 4 .. floor(5.9) = 5
    let mut rec = Rec { hits: vec![], outside: false };
    triangle(&[vt(4.0, 0.0, 0.5, 0.0, 0.0), vt(3.2, 7.0, 0.0, 1.0, 0.0), vt(5.9, 7.0, 1.0, 1.0, 0.0)], 9, 7, &mut rec);
    let last: Vec<_> = rec.hits.iter().filter(|h| h.1 == 7).collect();
    assert_eq!(last.iter().map(|h| h.0).collect::<Vec<_>>(), vec![4, 5]);
    // u runs from 0 at x = 3.2 to 1 at x = 5.9: at x = 4 it is 0.8 / 2.7
    assert!((last[0].2 - 0.8 / 2.7).abs() < 1e-5, "{}", last[0].2);
    assert!(rec.hits.iter().filter(|h| h.1 < 7).count() > 0, "the main pass covers the rows above");
    // no write at y = 7 when the lower vertices are not exactly on the row
    let mut rec = Rec { hits: vec![], outside: false };
    triangle(&[vt(4.0, 0.0, 0.5, 0.0, 0.0), vt(3.2, 6.9, 0.0, 1.0, 0.0), vt(5.9, 6.9, 1.0, 1.0, 0.0)], 9, 7, &mut rec);
    assert!(rec.hits.iter().all(|h| h.1 <= 6));
    // the sub pixel span beyond the right border would reach x = 10 > w1: nothing is written and
    // the sink is told
    let mut rec = Rec { hits: vec![], outside: false };
    triangle(&[vt(4.0, 0.0, 0.5, 0.0, 0.0), vt(9.2, 7.0, 0.0, 1.0, 0.0), vt(9.6, 7.0, 1.0, 1.0, 0.0)], 7, 7, &mut rec);
    assert!(rec.outside);
    assert!(rec.hits.iter().all(|h| h.0 <= 7));
}

#[test]
fn rejected_and_clipped_triangles() {
    let mut rec = Rec { hits: vec![], outside: false };
    // all x < 0
    triangle(&[vt(-3.0, 0.0, 0.0, 0.0, 0.0), vt(-1.0, 4.0, 0.0, 0.0, 0.0), vt(-2.0, 6.0, 0.0, 0.0, 0.0)], 7, 7, &mut rec);
    // all y > h1
    triangle(&[vt(1.0, 8.0, 0.0, 0.0, 0.0), vt(3.0, 9.0, 0.0, 0.0, 0.0), vt(2.0, 12.0, 0.0, 0.0, 0.0)], 7, 7, &mut rec);
    assert!(rec.hits.is_empty());
    // a big triangle clipped to the image: all hits inside
    triangle(&[vt(-20.0, -20.0, 0.0, 0.0, 0.0), vt(40.0, -10.0, 1.0, 0.0, 0.0), vt(5.0, 50.0, 0.0, 1.0, 0.0)], 7, 7, &mut rec);
    assert_eq!(rec.hits.len(), 64, "the whole 8x8 image lies inside");
    assert!(rec.hits.iter().all(|h| h.0 <= 7 && h.1 <= 7));
}

fn flat_src(w: usize, h: usize) -> Image {
    // 4 channel u8 image whose channel 1 is 10 * x so that the sampled value tells the texture coordinate
    let mut img = Image::new(w, h, Fmt::U8);
    if let Samples::U8(d) = &mut img.samples {
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                d[i] = 255;
                d[i + 1] = (10 * x) as u8;
                d[i + 2] = 0;
                d[i + 3] = 0;
            }
        }
    }
    img
}

#[test]
fn running_mean_and_the_saturating_hit_counter() {
    let src = flat_src(11, 2);
    let mut s = Accum {
        src: &src,
        pyr: std::cell::OnceCell::new(),
        acc: vec![0.0; 11 * 2 * 4],
        cnt: vec![0; 11 * 2],
        w: 11,
        outside: false,
    };
    // texture coordinate u = x / 10 gives the channel 1 value 10 x (bilinear at integer columns)
    let hit = |s: &mut Accum, x: f32| s.hit(3, 1, x / 10.0, 0.0, 0.0);
    hit(&mut s, 1.0); // 10
    assert_eq!((s.acc[(11 + 3) * 4 + 1], s.cnt[11 + 3]), (10.0, 1));
    hit(&mut s, 2.0); // (1 * 10 + 20) * (1 / 2) = 15
    assert_eq!((s.acc[(11 + 3) * 4 + 1], s.cnt[11 + 3]), (15.0, 2));
    hit(&mut s, 5.0); // (2 * 15 + 50) * f32(1 / 3) = 80 * 0.33333334 = 26.666668
    assert_eq!(s.acc[(11 + 3) * 4 + 1], (2.0f32 * 15.0 + 50.0) * 0.33333334f32);
    assert_eq!(s.cnt[11 + 3], 3);
    // alpha channel: three hits of 255
    assert!((s.acc[(11 + 3) * 4] - 255.0).abs() < 1e-3);
    // counter: 252 more hits reach 255, then it stays and the running mean is acc * 255 / 256 + c / 256
    for _ in 0..252 {
        hit(&mut s, 5.0);
    }
    assert_eq!(s.cnt[11 + 3], 255);
    let before = s.acc[(11 + 3) * 4 + 1];
    hit(&mut s, 10.0);
    assert_eq!(s.cnt[11 + 3], 255, "the counter saturates");
    assert_eq!(s.acc[(11 + 3) * 4 + 1], (before * 255.0 + 100.0) * 0.00390625 + 0.0);
    // the neighbour pixels were not touched
    assert_eq!(s.cnt.iter().map(|&c| c as usize).sum::<usize>(), 255);
}

#[test]
fn reciprocal_of_the_running_mean_is_a_double_division_narrowed() {
    // 1 / (n + 1) in double narrowed to float equals the correctly rounded float quotient for every n; pin the
    // values used by the accumulate step for the first counts
    for n in 1u32..=254 {
        let r = (1.0f64 / (n as f64 + 1.0)) as f32;
        let q = 1.0f32 / (n as f32 + 1.0);
        assert_eq!(r, q, "n = {n}");
    }
}

#[test]
fn convert_rules() {
    // u8 / u16: truncation of acc + 0.5 to 64 bits, low byte or word, no saturation; float: + 0.0
    let acc = [0.49f32, 0.5, 254.5, 255.4, 255.5, 300.7, -0.4, 65535.5, 70000.2, 0.0];
    let mut o8 = Image::new(10, 1, Fmt::U8);
    let mut acc4 = vec![0.0f32; 40];
    for (i, a) in acc.iter().enumerate() {
        acc4[i * 4 + 1] = *a;
    }
    convert(&mut o8, &acc4);
    let got: Vec<i32> = (0..10).map(|x| u8_at(&o8, x, 0, 1)).collect();
    assert_eq!(got, vec![0, 1, 255, 255, 0, 45, 0, 0, 112, 0]);
    let mut o16 = Image::new(10, 1, Fmt::U16);
    convert(&mut o16, &acc4);
    if let Samples::U16(d) = &o16.samples {
        let got: Vec<u16> = (0..10).map(|x| d[x * 4 + 1]).collect();
        assert_eq!(got, vec![0, 1, 255, 255, 256, 301, 0, 0, 4464, 0]);
    }
    let mut of = Image::new(10, 1, Fmt::F32);
    let mut z = acc4.clone();
    z[0] = -0.0;
    convert(&mut of, &z);
    if let Samples::F32(d) = &of.samples {
        assert!(d[0].is_sign_positive(), "-0.0 + 0.0 is +0.0");
        assert_eq!(d[1], 0.49);
    }
}

#[test]
fn gouraud_variant_interpolates_the_vertex_colours() {
    // a white impulse at (7, 6) of an opaque black 16 x 12 frame pushed by half of (1, 1): the cell split runs from
    // (c + 1, r) to (c, r + 1), so the impulse lands on the two pixels of that diagonal at half value (127 after the
    // tie breaking epsilon of the vertices), where the textured variant spreads it over four pixels at a quarter
    let (w, h) = (16usize, 12usize);
    let mut src = Image::new(w, h, Fmt::U8);
    if let Samples::U8(d) = &mut src.samples {
        for px in d.chunks_exact_mut(CHANNELS) {
            px.copy_from_slice(&[255, 0, 0, 0]);
        }
        let i = (6 * w + 7) * CHANNELS;
        d[i..i + CHANNELS].copy_from_slice(&[255, 255, 255, 255]);
    }
    let flow = const_flow(w, h, [1.0, 1.0]);
    let red = |img: &Image, x: usize, y: usize| match &img.samples {
        Samples::U8(d) => d[(y * w + x) * CHANNELS + 1],
        _ => unreachable!(),
    };
    let mut out = Image::new(w, h, Fmt::U8);
    let cov = forward_warp_with(&mut out, &src, &flow, 0.5, ForwardOpts { pyramid: false }, true).unwrap().unwrap();
    for y in 0..h {
        for x in 0..w {
            let want = if (x, y) == (8, 6) || (x, y) == (7, 7) { 127 } else { 0 };
            assert_eq!(red(&out, x, y), want, "Gouraud ({x}, {y})");
        }
    }
    assert_eq!(cov.data[0], 0, "the top row is not reached by a push of half a pixel down");
    let mut tex = Image::new(w, h, Fmt::U8);
    forward_warp_with(&mut tex, &src, &flow, 0.5, ForwardOpts::default(), false).unwrap();
    for (x, y) in [(7, 6), (8, 6), (7, 7), (8, 7)] {
        assert_eq!(red(&tex, x, y), 64, "textured ({x}, {y})");
    }
    // a zero flow is still the plain copy (the early outs come first)
    forward_warp_with(&mut out, &src, &const_flow(w, h, [0.0, 0.0]), 1.0, ForwardOpts { pyramid: false }, false).unwrap();
    assert_eq!(out, src);
}

#[test]
fn shape_errors() {
    let src = textured(6, 6, Fmt::U8, 1);
    let flow = const_flow(6, 6, [1.0, 0.0]);
    let mut other = Image::new(6, 7, Fmt::U8);
    assert!(forward_warp(&mut other, &src, &flow, 1.0).is_err());
    let mut f = Image::new(6, 6, Fmt::F32);
    assert!(forward_warp(&mut f, &src, &flow, 1.0).is_err());
    let tiny = textured(1, 6, Fmt::U8, 1);
    let mut o = Image::new(1, 6, Fmt::U8);
    assert!(forward_warp(&mut o, &tiny, &flow, 1.0).is_err());
    let mut o = Image::new(6, 6, Fmt::U8);
    assert!(forward_warp(&mut o, &src, &const_flow(1, 6, [1.0, 0.0]), 1.0).is_err());
}

#[test]
fn hostile_flows_never_panic() {
    let src = textured(9, 7, Fmt::U8, 4);
    for v in [f32::NAN, f32::INFINITY, -f32::INFINITY, 1e30, -1e30, 3e9] {
        let mut flow = const_flow(9, 7, [0.5, 0.25]);
        flow.set(3, 3, [v, 0.0]);
        flow.set(4, 4, [0.0, v]);
        for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
            let s = textured(9, 7, fmt, 4);
            let mut out = Image::new(9, 7, fmt);
            let _ = forward_warp_coverage(&mut out, &s, &flow, 1.0);
            let _ = forward_warp_coverage(&mut out, &s, &flow, f32::NAN);
            let _ = forward_warp_coverage(&mut out, &s, &flow, 1e30);
        }
    }
    let mut out = Image::new(9, 7, Fmt::U8);
    let _ = forward_warp(&mut out, &src, &const_flow(9, 7, [f32::NAN, f32::NAN]), 1.0);
}

// ------------------------------------------------------------------------------------------------
// the numpy model
// ------------------------------------------------------------------------------------------------

struct Rd<'a>(&'a [u8], usize);

impl Rd<'_> {
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.0[self.1..self.1 + 4].try_into().unwrap());
        self.1 += 4;
        v
    }
    fn f32(&mut self) -> f32 {
        f32::from_bits(self.u32())
    }
    fn bytes(&mut self, n: usize) -> &[u8] {
        let s = &self.0[self.1..self.1 + n];
        self.1 += n;
        s
    }
}

fn read_image(r: &mut Rd, w: usize, h: usize, fmt: u32) -> Image {
    let mut img = Image::new(w, h, [Fmt::U8, Fmt::U16, Fmt::F32][fmt as usize]);
    let n = w * h * 4;
    match &mut img.samples {
        Samples::U8(d) => d.copy_from_slice(r.bytes(n)),
        Samples::U16(d) => d.iter_mut().zip(r.bytes(n * 2).chunks_exact(2)).for_each(|(o, b)| *o = u16::from_le_bytes([b[0], b[1]])),
        Samples::F32(d) => d.iter_mut().zip(r.bytes(n * 4).chunks_exact(4)).for_each(|(o, b)| *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
    }
    img
}

#[test]
fn model_fixture() {
    // f2_gen.py: 26 cases over the three depths, translations, shears, squeezes (levels above 0), a swirl,
    // random flows, a half resolution flow, negative and tiny amounts, a zero amount. u8, u16 and the
    // coverage must be bit exact; f32 may differ where the numpy `log` differs from the CRT `logf` in the last
    // bit of the mip level (it only moves the blend of two levels by that fraction).
    let data = crate::fixture::hex(include_str!("../../tests/fixtures/f2_cases.hex"));
    let mut r = Rd(&data, 0);
    assert_eq!(r.bytes(4), b"F2C1");
    let n = r.u32();
    assert_eq!(n, 26);
    let mut f32_worst = 0.0f32;
    let mut exact = 0;
    for case in 0..n {
        let (fmt, w, h, fw, fh) = (r.u32(), r.u32() as usize, r.u32() as usize, r.u32() as usize, r.u32() as usize);
        let amount = r.f32();
        let want_cov = r.u32();
        assert_eq!(want_cov, 1);
        let src = read_image(&mut r, w, h, fmt);
        let mut flow = FlowField::new(fw, fh);
        for v in flow.data.iter_mut() {
            *v = r.f32();
        }
        let want = read_image(&mut r, w, h, fmt);
        let want_cov: Vec<u8> = r.bytes(w * h).to_vec();
        let mut out = Image::new(w, h, src.fmt());
        let cov = forward_warp_coverage(&mut out, &src, &flow, amount).unwrap();
        assert_eq!(cov.data, want_cov, "case {case}: coverage");
        match (&out.samples, &want.samples) {
            (Samples::F32(a), Samples::F32(b)) => {
                let mut worst = 0.0f32;
                for (i, (&p, &q)) in a.iter().zip(b).enumerate() {
                    let d = (p - q).abs();
                    worst = worst.max(d);
                    assert!(d <= 2e-6, "case {case} f32 sample {i}: {p} against {q}");
                }
                if worst == 0.0 {
                    exact += 1;
                }
                f32_worst = f32_worst.max(worst);
            }
            _ => {
                assert_eq!(bytes_of(&out), bytes_of(&want), "case {case} ({fmt}, {w}x{h}, flow {fw}x{fh}, amount {amount})");
                exact += 1;
            }
        }
    }
    println!("f2 model: {exact}/{n} cases bit exact, worst f32 difference {f32_worst:e}");
}

// ------------------------------------------------------------------------------------------------
// threads
// ------------------------------------------------------------------------------------------------

#[test]
fn thread_count_does_not_change_a_byte() {
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let (w, h) = (53, 41);
        let src = textured(w, h, fmt, 9);
        let mut flow = FlowField::new(27, 21);
        for y in 0..21 {
            for x in 0..27 {
                flow.set(x, y, [3.0 * ((y as f32) * 0.4).sin() - 0.2 * x as f32, 2.5 * ((x as f32) * 0.3).cos()]);
            }
        }
        let run = |threads: usize| {
            in_pool(threads, || {
                let mut out = Image::new(w, h, fmt);
                let cov = forward_warp_coverage(&mut out, &src, &flow, 0.9).unwrap();
                (bytes_of(&out), cov.data)
            })
        };
        let a = run(1);
        let b = run(4);
        assert_eq!(a, b, "{fmt:?}");
        assert!(a.1.iter().any(|&c| c == 255) && a.1.iter().any(|&c| c == 0), "the case has covered and uncovered pixels");
    }
}

// ------------------------------------------------------------------------------------------------
// compositions against explicit call sequences
// ------------------------------------------------------------------------------------------------

fn two_frames(fmt: Fmt) -> (Image, Image, FlowField, FlowField) {
    let a = textured(24, 18, fmt, 21);
    let b = textured(24, 18, fmt, 22);
    // asymmetric flows so that a swapped flow is visible
    let f_ab = const_flow(24, 18, [3.0, 1.0]);
    let f_ba = const_flow(24, 18, [-2.0, 0.5]);
    (a, b, f_ab, f_ba)
}

#[test]
fn nearest_forward_picks_the_frame_and_the_amount() {
    for fmt in [Fmt::U8, Fmt::F32] {
        let (a, b, f_ab, f_ba) = two_frames(fmt);
        for (wa, flow) in [(0.8f32, &f_ab), (0.5, &f_ab), (0.3, &f_ba), (0.0, &f_ba), (1.0, &f_ab)] {
            let mut out = Image::new(24, 18, fmt);
            nearest_forward(&mut out, &a, &b, flow, wa).unwrap();
            let mut want = Image::new(24, 18, fmt);
            if !(0.5f32 > wa) {
                forward_warp(&mut want, &a, flow, 1.0f32 - wa).unwrap();
            } else {
                forward_warp(&mut want, &b, flow, wa).unwrap();
            }
            assert_eq!(out, want, "{fmt:?} wa {wa}");
        }
        // at wa = 1 the amount is 1 - 1 = 0: a plain copy of A; at wa = 0 amount 0: a copy of B
        let mut out = Image::new(24, 18, fmt);
        nearest_forward(&mut out, &a, &b, &f_ab, 1.0).unwrap();
        assert_eq!(out, a);
        nearest_forward(&mut out, &a, &b, &f_ba, 0.0).unwrap();
        assert_eq!(out, b);
    }
}

#[test]
fn nearest_forward_fill_is_warp_invert_fill() {
    let (a, b, f_ab, _) = two_frames(Fmt::U8);
    let mut out = Image::new(24, 18, Fmt::U8);
    nearest_forward_fill(&mut out, &a, &b, &f_ab, 0.7, SmartOpts::default()).unwrap();
    let mut want = Image::new(24, 18, Fmt::U8);
    let mut cov = forward_warp_coverage(&mut want, &a, &f_ab, 1.0f32 - 0.7f32).unwrap();
    assert!(cov.data.iter().any(|&c| c == 0), "there are holes to fill");
    cov.invert();
    fill_holes(&mut want, &cov, false).unwrap();
    assert_eq!(out, want);
    // the plain variant keeps the holes transparent black
    let mut plain = Image::new(24, 18, Fmt::U8);
    nearest_forward(&mut plain, &a, &b, &f_ab, 0.7).unwrap();
    assert_eq!(u8_at(&plain, 0, 0, 0), 0);
    assert_ne!(u8_at(&out, 0, 0, 0), 0);
}

#[test]
fn blend_and_weighted_forward_pair_each_frame_with_its_own_flow() {
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let (a, b, f_ab, f_ba) = two_frames(fmt);
        let wa = 0.65f32;
        let frac = 1.0f32 - wa;
        let mut ta = Image::new(24, 18, fmt);
        let mut tb = Image::new(24, 18, fmt);
        let cov1 = forward_warp_coverage(&mut ta, &a, &f_ab, frac).unwrap();
        let cov2 = forward_warp_coverage(&mut tb, &b, &f_ba, wa).unwrap();
        // code 3 Blend
        let mut out = Image::new(24, 18, fmt);
        blend_forward(&mut out, &a, &f_ab, &b, &f_ba, wa).unwrap();
        let mut want = Image::new(24, 18, fmt);
        blend_tail(&mut want, &ta, &tb, frac, None, SmartOpts::default()).unwrap();
        assert_eq!(out, want, "{fmt:?} blend code 3");
        // swapping the flows changes the picture (the pairing is not symmetric)
        let mut swapped = Image::new(24, 18, fmt);
        blend_forward(&mut swapped, &a, &f_ba, &b, &f_ab, wa).unwrap();
        assert_ne!(out, swapped);
        // code 2 Blend, both option sets
        for opts in [SmartOpts::default(), SmartOpts { binary_alpha: true, is360: false, ..SmartOpts::default() }] {
            let mut out = Image::new(24, 18, fmt);
            blend_forward_fill(&mut out, &a, &f_ab, &b, &f_ba, wa, opts).unwrap();
            let mut want = Image::new(24, 18, fmt);
            let (mut c1, c2) = (cov1.clone(), cov2.clone());
            blend_tail(&mut want, &ta, &tb, frac, Some((&mut c1, &c2)), opts).unwrap();
            assert_eq!(out, want, "{fmt:?} blend code 2 {opts:?}");
        }
        // code 3 and code 2 Weighted Blend
        let mut out = Image::new(24, 18, fmt);
        weighted_forward(&mut out, &a, &f_ab, &b, &f_ba, wa).unwrap();
        let mut want = Image::new(24, 18, fmt);
        let (mut ta3, mut tb3) = (ta.clone(), tb.clone());
        weighted_tail(&mut want, &mut ta3, &a, &mut tb3, &b, wa, None, SmartOpts::default()).unwrap();
        assert_eq!(out, want, "{fmt:?} weighted code 3");
        let mut out = Image::new(24, 18, fmt);
        weighted_forward_fill(&mut out, &a, &f_ab, &b, &f_ba, wa, SmartOpts::default()).unwrap();
        let mut want = Image::new(24, 18, fmt);
        let (mut ta2, mut tb2) = (ta.clone(), tb.clone());
        let (mut c1, c2) = (cov1.clone(), cov2.clone());
        weighted_tail(&mut want, &mut ta2, &a, &mut tb2, &b, wa, Some((&mut c1, &c2)), SmartOpts::default()).unwrap();
        assert_eq!(out, want, "{fmt:?} weighted code 2");
    }
}

/// Code 3 Weighted Blend sharpens with the vertical tap order of its stage set, as the inverse and code 2 weighted
/// compositions do (the plugin's Weighted Blend blur is the shared separable routine for every warping code; MEASURED:
/// fgbg-core `interp3-w4.pan-16x12.d32` identical with it).
#[test]
fn weighted_forward_on_takes_the_vertical_taps_of_its_stage_set() {
    use crate::holefill::VerticalTaps;
    use crate::stages::{AboveFirstTaps, StageSet, Stages};
    let (a, b, f_ab, f_ba) = two_frames(Fmt::F32);
    let wa = 0.65f32;
    let frac = 1.0f32 - wa;
    let above = AboveFirstTaps::wrap(StageSet::cpu());
    let mut out = Image::new(24, 18, Fmt::F32);
    weighted_forward_on(above.get(), &mut out, &a, &f_ab, &b, &f_ba, wa).unwrap();
    let (mut ta, mut tb) = (Image::new(24, 18, Fmt::F32), Image::new(24, 18, Fmt::F32));
    above.get().forward_warp(&mut ta, &a, &f_ab, frac).unwrap();
    above.get().forward_warp(&mut tb, &b, &f_ba, wa).unwrap();
    let mut want = Image::new(24, 18, Fmt::F32);
    let opts = SmartOpts { taps: VerticalTaps::AboveFirst, ..SmartOpts::default() };
    weighted_tail(&mut want, &mut ta.clone(), &a, &mut tb.clone(), &b, wa, None, opts).unwrap();
    assert_eq!(out, want);
    let mut below = Image::new(24, 18, Fmt::F32);
    weighted_tail(&mut below, &mut ta, &a, &mut tb, &b, wa, None, SmartOpts::default()).unwrap();
    assert_ne!(out, below, "the border rows of the sharpening blur round otherwise with the tap below first");
}

#[test]
fn compositions_in_threads_are_identical() {
    let (a, b, f_ab, f_ba) = two_frames(Fmt::U8);
    let run = |t: usize| {
        in_pool(t, || {
            let mut o1 = Image::new(24, 18, Fmt::U8);
            blend_forward_fill(&mut o1, &a, &f_ab, &b, &f_ba, 0.4, SmartOpts::default()).unwrap();
            let mut o2 = Image::new(24, 18, Fmt::U8);
            weighted_forward_fill(&mut o2, &a, &f_ab, &b, &f_ba, 0.4, SmartOpts::default()).unwrap();
            (o1, o2)
        })
    };
    assert_eq!(run(1), run(4));
}

// ------------------------------------------------------------------------------------------------
// ground truth: a panning textured background and a square that moves twice as fast
// ------------------------------------------------------------------------------------------------

const GW: usize = 64;
const GH: usize = 48;

/// The background texture as a function of the (shifted) coordinate: smooth and non periodic enough.
fn bg(x: i32, y: i32, ch: usize) -> f32 {
    let (xf, yf) = (x as f32, y as f32);
    let v = 0.5 + 0.28 * (xf * 0.37 + ch as f32).sin() + 0.2 * (yf * 0.29 + xf * 0.05).cos();
    (v * 255.0).clamp(0.0, 255.0)
}

/// The square's own texture in square local coordinates.
fn sq(i: i32, j: i32, ch: usize) -> f32 {
    let v = 0.5 + 0.35 * ((i as f32) * 0.8 + ch as f32 * 0.7).cos() * ((j as f32) * 0.6).sin();
    (v * 255.0).clamp(0.0, 255.0)
}

/// Frame at pan offset `p` (the background moved right by `p` pixels) with the 16x16 square whose left edge is
/// at `sx` and top edge at 14.
fn scene(p: i32, sx: i32) -> Image {
    let mut img = Image::new(GW, GH, Fmt::U8);
    if let Samples::U8(d) = &mut img.samples {
        for y in 0..GH as i32 {
            for x in 0..GW as i32 {
                let i = ((y as usize) * GW + x as usize) * 4;
                let inside = x >= sx && x < sx + 16 && (14..30).contains(&y);
                d[i] = 255;
                for ch in 1..4 {
                    d[i + ch] = if inside { sq(x - sx, y - 14, ch) } else { bg(x - p, y, ch) } as u8;
                }
            }
        }
    }
    img
}

#[test]
fn ground_truth_square_lands_where_the_motion_says() {
    // frame A: background at pan 0, square at x 10..26. Frame B: pan 10, square at 30..46 (20 pixels per frame).
    // F_AB on the grid of A: (10, 0) on the background, (20, 0) inside the square. At the midpoint (amount 0.5)
    // the background is shifted by 5 and the square by 10, i.e. it sits at x 20..36.
    let a = scene(0, 10);
    let mut flow = FlowField::new(GW, GH);
    for y in 0..GH {
        for x in 0..GW {
            let inside = (10..26).contains(&x) && (14..30).contains(&y);
            flow.set(x, y, if inside { [20.0, 0.0] } else { [10.0, 0.0] });
        }
    }
    let mut out = Image::new(GW, GH, Fmt::U8);
    let cov = forward_warp_coverage(&mut out, &a, &flow, 0.5).unwrap();

    // holes: exactly the five leftmost columns (the left vertices are at x = 5.0 exactly)
    let holes: Vec<(usize, usize)> = (0..GH).flat_map(|y| (0..GW).map(move |x| (x, y))).filter(|&(x, y)| cov.at(x, y) == 0).collect();
    assert_eq!(holes.len(), 5 * GH, "holes only in the strip the pan uncovered");
    assert!(holes.iter().all(|&(x, _)| x < 5));

    // the square interior at its analytic position: 2 pixel margin from the stretched cell on its left (x 14..20),
    // the cell between its last column (x' = 35) and the first background column (x' = 31) folds over x 31..35
    let mut worst = 0;
    for y in 16..28i32 {
        for x in 22..31i32 {
            for ch in 1..4 {
                let want = sq(x - 20, y - 14, ch) as i32;
                let got = u8_at(&out, x as usize, y as usize, ch);
                worst = worst.max((got - want).abs());
            }
        }
    }
    assert!(worst <= 1, "square interior off by {worst}");
    // the square is NOT where it was in A (x 10..26 plus nothing) and not at the full motion either
    let at_a = (16..28i32).flat_map(|y| (12..18i32).map(move |x| (x, y))).filter(|&(x, y)| {
        (1..4).all(|ch| (u8_at(&out, x as usize, y as usize, ch) - sq(x - 10, y - 14, ch) as i32).abs() <= 1)
    });
    assert_eq!(at_a.count(), 0);

    // background away from the square: shifted by 5 on both sides
    let mut worst = 0;
    for y in 0..GH as i32 {
        for x in (7..13i32).chain(37..GW as i32) {
            for ch in 1..4 {
                let want = bg(x - 5, y, ch) as i32;
                let got = u8_at(&out, x as usize, y as usize, ch);
                worst = worst.max((got - want).abs());
            }
        }
    }
    assert!(worst <= 1, "background off by {worst}");

    // the filled result (warp code 2, Nearest): no transparent pixel is left, known pixels are unchanged
    let b = scene(10, 30);
    let mut filled = Image::new(GW, GH, Fmt::U8);
    nearest_forward_fill(&mut filled, &a, &b, &flow, 0.5, SmartOpts::default()).unwrap();
    for y in 0..GH {
        for x in 0..GW {
            assert_eq!(u8_at(&filled, x, y, 0), 255, "alpha at {x},{y}");
            if cov.at(x, y) != 0 {
                for ch in 0..4 {
                    assert_eq!(u8_at(&filled, x, y, ch), u8_at(&out, x, y, ch), "known pixel {x},{y}");
                }
            }
        }
    }
    // the filled strip is a smooth blend of the known pixels: inside their range per channel, and it differs
    // from transparent black (the plain variant leaves 0 there)
    for ch in 1..4 {
        let known: Vec<i32> =
            (0..GH).flat_map(|y| (5..GW).map(move |x| (x, y))).map(|(x, y)| u8_at(&out, x, y, ch)).collect();
        let (lo, hi) = (*known.iter().min().unwrap(), *known.iter().max().unwrap());
        for y in 0..GH {
            for x in 0..5 {
                let v = u8_at(&filled, x, y, ch);
                assert!(v >= lo - 1 && v <= hi + 1, "filled {x},{y},{ch} = {v} not in {lo}..{hi}");
            }
        }
    }
}

#[test]
fn ground_truth_without_motion_blur_of_the_background() {
    // a pure pan of the whole image by 6 pixels at amount 1: the image shifted, six leftmost columns holes
    let a = scene(0, 100); // the square lies outside the image
    let flow = const_flow(GW, GH, [6.0, 0.0]);
    let mut out = Image::new(GW, GH, Fmt::U8);
    let cov = forward_warp_coverage(&mut out, &a, &flow, 1.0).unwrap();
    for y in 0..GH {
        for x in 0..GW {
            assert_eq!(cov.at(x, y) != 0, x >= 6);
        }
    }
    let mut worst = 0;
    for y in 0..GH as i32 {
        for x in 6..GW as i32 {
            for ch in 1..4 {
                worst = worst.max((u8_at(&out, x as usize, y as usize, ch) - bg(x - 6, y, ch) as i32).abs());
            }
        }
    }
    assert!(worst <= 1, "pan off by {worst}");
}

/// Timing probe (release): `cargo test -p retime-engine --release --lib forward::tests::timing -- --ignored --nocapture`.
#[test]
#[ignore]
fn timing() {
    for (w, h) in [(1280usize, 720usize), (1920, 1080)] {
        for fmt in [Fmt::U8, Fmt::F32] {
            let src = textured(w, h, fmt, 5);
            let (fw, fh) = (w / 8, h / 8);
            let mut flow = FlowField::new(fw, fh);
            for y in 0..fh {
                for x in 0..fw {
                    flow.set(x, y, [6.0 * ((y as f32) * 0.2).sin(), 5.0 * ((x as f32) * 0.15).cos()]);
                }
            }
            for threads in [1usize, 16] {
                let t = std::time::Instant::now();
                let (mut out, cov) = in_pool(threads, || {
                    let mut out = Image::new(w, h, fmt);
                    let cov = forward_warp_coverage(&mut out, &src, &flow, 0.5).unwrap();
                    (out, cov)
                });
                let _ = &mut out;
                println!("{w}x{h} {fmt:?} {threads} threads: {:?} (covered {})", t.elapsed(), cov.data.iter().filter(|&&c| c != 0).count());
            }
        }
    }
}
