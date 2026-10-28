//! Tests of the Draw Geom overlay (rule G3): the disc footprint on and off pixel centres and at
//! other ratios (against a brute force of the coverage test), the colours per depth, later
//! points on top, clipping at the borders and the inputs that draw nothing.

use super::*;
use crate::image::Fmt;

const FMTS: [Fmt; 3] = [Fmt::U8, Fmt::U16, Fmt::F32];

/// The pixels with a non zero alpha, row by row.
fn covered(img: &Image) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for y in 0..img.height {
        for x in 0..img.width {
            let i = (y * img.width + x) * CHANNELS;
            let a = match &img.samples {
                Samples::U8(s) => f32::from(s[i]),
                Samples::U16(s) => f32::from(s[i]),
                Samples::F32(s) => s[i],
            };
            if a != 0.0 {
                out.push((x, y));
            }
        }
    }
    out
}

/// Every pixel of the image tested against the disc of radius `1.5 * ratio` at `p`.
fn brute(w: usize, h: usize, p: [f32; 2], ratio: f32) -> Vec<(usize, usize)> {
    let r = 1.5f32 * ratio;
    let r2 = r * r;
    let mut out = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (x as f32 - p[0], y as f32 - p[1]);
            if dx * dx + dy * dy <= r2 {
                out.push((x, y));
            }
        }
    }
    out
}

fn draw(w: usize, h: usize, fmt: Fmt, pts: &[([f32; 2], [u8; 3])], ratio: f32) -> Image {
    let mut img = Image::new(w, h, fmt);
    draw_points(&mut img, pts, ratio);
    img
}

fn block(x0: usize, x1: usize, y0: usize, y1: usize) -> Vec<(usize, usize)> {
    (y0..=y1).flat_map(|y| (x0..=x1).map(move |x| (x, y))).collect()
}

#[test]
fn a_point_on_a_pixel_centre_covers_the_3x3_block() {
    // offsets with dx*dx + dy*dy <= 2.25: the corners (1 + 1 = 2) are in, (2, 0) is not
    let img = draw(21, 21, Fmt::U8, &[([10.0, 10.0], [0, 255, 0])], 1.0);
    assert_eq!(covered(&img), block(9, 11, 9, 11));
}

#[test]
fn sub_pixel_points_cover_the_pixels_within_one_and_a_half() {
    // (10.5, 10.5): the four pixels around it (0.25 + 0.25), not (12, 10) (2.25 + 0.25)
    let img = draw(21, 21, Fmt::U8, &[([10.5, 10.5], [1, 2, 3])], 1.0);
    assert_eq!(covered(&img), block(10, 11, 10, 11));
    // (10.25, 10): column 9 only on the row of the point, columns 10 and 11 on three rows, no 12
    let img = draw(21, 21, Fmt::U8, &[([10.25, 10.0], [1, 2, 3])], 1.0);
    assert_eq!(covered(&img), vec![(10, 9), (11, 9), (9, 10), (10, 10), (11, 10), (10, 11), (11, 11)]);
    // a value exactly on the circle is covered: (11.5, 10) reaches (10, 10) at dx = -1.5
    let img = draw(21, 21, Fmt::U8, &[([11.5, 10.0], [1, 2, 3])], 1.0);
    assert!(covered(&img).contains(&(10, 10)));
    assert!(covered(&img).contains(&(13, 10)));
    assert!(!covered(&img).contains(&(10, 11)));
}

#[test]
fn the_disc_radius_scales_with_the_ratio() {
    // ratio 2: radius 3, the 29 pixels of x*x + y*y <= 9
    let img = draw(21, 21, Fmt::U8, &[([10.0, 10.0], [9, 9, 9])], 2.0);
    assert_eq!(covered(&img).len(), 29);
    // any position and ratio: exactly the pixels the coverage test admits
    let mut seed = 0x1234_5678u32;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed >> 8) as f32 / 16_777_216.0
    };
    for _ in 0..200 {
        let p = [next() * 30.0 - 5.0, next() * 24.0 - 5.0];
        let ratio = [0.5f32, 1.0, 1.0, 2.0, 3.25][(next() * 5.0) as usize % 5];
        let img = draw(20, 14, Fmt::F32, &[(p, [200, 100, 50])], ratio);
        assert_eq!(covered(&img), brute(20, 14, p, ratio), "point {p:?} ratio {ratio}");
    }
}

#[test]
fn the_colour_is_opaque_in_every_depth() {
    let col = [255u8, 128, 7];
    for fmt in FMTS {
        let img = draw(5, 5, fmt, &[([2.0, 2.0], col)], 1.0);
        let i = (2 * 5 + 2) * CHANNELS;
        match &img.samples {
            Samples::U8(s) => assert_eq!(&s[i..i + 4], &[255, 255, 128, 7]),
            Samples::U16(s) => assert_eq!(&s[i..i + 4], &[65535, 65535, 128 * 257, 7 * 257]),
            Samples::F32(s) => assert_eq!(&s[i..i + 4], &[1.0, 1.0, 128.0f32 / 255.0, 7.0f32 / 255.0]),
        }
    }
    // the default colours of the twelve points reach 16 bpc exactly as c * 257
    assert_eq!(255u16 * 257, 65535);
}

#[test]
fn a_later_point_is_drawn_on_top() {
    let pts = [([10.0f32, 10.0f32], [255u8, 0, 0]), ([11.0, 10.0], [0, 0, 255])];
    for fmt in FMTS {
        let img = draw(21, 21, fmt, &pts, 1.0);
        let px = |x: usize, y: usize| -> [f32; 4] {
            let i = (y * 21 + x) * CHANNELS;
            match &img.samples {
                Samples::U8(s) => [s[i], s[i + 1], s[i + 2], s[i + 3]].map(f32::from),
                Samples::U16(s) => [s[i], s[i + 1], s[i + 2], s[i + 3]].map(f32::from),
                Samples::F32(s) => [s[i], s[i + 1], s[i + 2], s[i + 3]],
            }
        };
        let white = fmt.white();
        let red = [white, white, 0.0, 0.0];
        let blue = [white, 0.0, 0.0, white];
        // the overlap (columns 10 and 11) is blue, column 9 keeps the first point's red
        for y in 9..=11 {
            assert_eq!(px(9, y), red, "{fmt:?}");
            assert_eq!(px(10, y), blue, "{fmt:?}");
            assert_eq!(px(12, y), blue, "{fmt:?}");
        }
        // drawn the other way round, red is on top
        let img2 = draw(21, 21, fmt, &[pts[1], pts[0]], 1.0);
        assert_eq!(covered(&img2), covered(&img));
        let i = (10 * 21 + 11) * CHANNELS;
        match &img2.samples {
            Samples::U8(s) => assert_eq!(s[i + 1], 255),
            Samples::U16(s) => assert_eq!(s[i + 1], 65535),
            Samples::F32(s) => assert_eq!(s[i + 1], 1.0),
        }
    }
}

#[test]
fn only_the_disc_is_touched() {
    // a drawn image keeps every other pixel bit for bit, a disc off the image draws nothing
    let mut img = Image::new(9, 6, Fmt::F32);
    if let Samples::F32(s) = &mut img.samples {
        for (i, v) in s.iter_mut().enumerate() {
            *v = i as f32 * 0.01 - 0.3;
        }
    }
    let before = img.clone();
    draw_points(&mut img, &[], 1.0);
    assert_eq!(img, before);
    draw_points(&mut img, &[([-5.0, -5.0], [1, 2, 3]), ([10.6, 3.0], [1, 2, 3]), ([3.0, 7.6], [1, 2, 3])], 1.0);
    assert_eq!(img, before);
    draw_points(&mut img, &[([4.0, 2.0], [1, 2, 3])], 1.0);
    let (Samples::F32(a), Samples::F32(b)) = (&img.samples, &before.samples) else { unreachable!() };
    for p in 0..9 * 6 {
        let (x, y) = (p % 9, p / 9);
        let inside = (3..=5).contains(&x) && (1..=3).contains(&y);
        assert_eq!(a[p * 4..p * 4 + 4] == b[p * 4..p * 4 + 4], !inside, "pixel ({x}, {y})");
    }
}

#[test]
fn discs_are_clipped_at_the_borders() {
    let img = draw(6, 5, Fmt::U16, &[([0.0, 0.0], [10, 20, 30]), ([5.0, 4.0], [10, 20, 30])], 1.0);
    assert_eq!(covered(&img), vec![(0, 0), (1, 0), (0, 1), (1, 1), (4, 3), (5, 3), (4, 4), (5, 4)]);
    // a point just off the image still reaches it
    let img = draw(6, 5, Fmt::U8, &[([-1.0, 2.0], [1, 1, 1])], 1.0);
    assert_eq!(covered(&img), vec![(0, 1), (0, 2), (0, 3)]);
    // an empty image takes any point
    let mut empty = Image::new(0, 3, Fmt::U8);
    draw_points(&mut empty, &[([0.0, 0.0], [1, 1, 1])], 1.0);
    assert_eq!(empty.sample_count(), 0);
}

#[test]
fn inputs_that_are_not_numbers_draw_nothing() {
    for p in [[f32::NAN, 2.0], [2.0, f32::INFINITY], [f32::NEG_INFINITY, 2.0]] {
        assert!(covered(&draw(6, 5, Fmt::U8, &[(p, [1, 1, 1])], 1.0)).is_empty(), "{p:?}");
    }
    for ratio in [f32::NAN, -1.0, f32::INFINITY] {
        assert!(covered(&draw(6, 5, Fmt::U8, &[([2.0, 2.0], [1, 1, 1])], ratio)).is_empty(), "ratio {ratio}");
    }
    // a ratio of 0 covers the pixel the point sits on exactly, and only it
    assert_eq!(covered(&draw(6, 5, Fmt::U8, &[([2.0, 2.0], [1, 1, 1])], 0.0)), vec![(2, 2)]);
    assert!(covered(&draw(6, 5, Fmt::U8, &[([2.5, 2.0], [1, 1, 1])], 0.0)).is_empty());
}

#[test]
fn coordinates_and_radii_beyond_the_integer_range_saturate() {
    // the ends of the box saturate instead of overflowing (an overflow panicked in a debug build)
    assert_eq!(span(1.0e30, 1.5, 4), None);
    assert_eq!(span(-1.0e30, 1.5, 4), None);
    assert_eq!(span(9.3e18, 1.5, 4), None);
    assert_eq!(span(-9.3e18, 1.5, 4), None);
    assert_eq!(span(1.0, 1.0e20, 4), Some((0, 3)));
    assert_eq!(span(1.0, 1.5, 4), Some((0, 3)));
    assert_eq!(span(-3.0, 1.5, 4), Some((0, 0)), "the box reaches one pixel further, the disc test decides");
    let pts = [([1.0e30, 1.0], [1, 2, 3]), ([-1.0e30, 1.0], [1, 2, 3]), ([9.3e18, 1.0], [1, 2, 3])];
    assert!(covered(&draw(4, 4, Fmt::U8, &pts, 1.0)).is_empty());
    // a radius of 1.5e20 covers the whole frame (its square is infinite in single precision)
    assert_eq!(covered(&draw(4, 4, Fmt::U8, &[([1.0, 1.0], [1, 2, 3])], 1.0e20)), block(0, 3, 0, 3));
}
