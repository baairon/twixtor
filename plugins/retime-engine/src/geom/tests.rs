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

// ---- draw_crosses: the glyph of CreateMotionVectors_64, read from its code. The alpha lists and
// the plot samples below are the plot calls of the original (traced under the emulator) or the
// same rules evaluated outside the rewrite; the
// tables of whole frames are the values MEASURED in
// the reference renders of the probe set `geom`.

/// The pixels a cross draws on a `w x h` frame: `(x, y, alpha)` of each of its two arms, from the
/// alpha rule alone.
fn arm_list(a: [f32; 2], b: [f32; 2], w: usize, h: usize) -> Vec<(usize, usize, u8)> {
    let mut out = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let al = arm_alpha(a, b, 1.5, [x as f32, y as f32]);
            if al > 0 {
                out.push((x, y, al));
            }
        }
    }
    out
}

#[test]
fn the_arms_of_a_cross_have_the_alpha_the_original_plots() {
    // point 1 of the probe case draw-geom2 (6.5, 4.5 in the host's rows, 6.5 in the plugin's): the
    // horizontal arm from 3.5 to 9.5 and the vertical one from 3.5 to 9.5, lines of width 1.5
    let h: [(usize, usize, u8); 16] = [
        (3, 6, 138), (4, 6, 191), (5, 6, 191), (6, 6, 191), (7, 6, 191), (8, 6, 191), (9, 6, 191), (10, 6, 138),
        (3, 7, 138), (4, 7, 191), (5, 7, 191), (6, 7, 191), (7, 7, 191), (8, 7, 191), (9, 7, 191), (10, 7, 138),
    ];
    assert_eq!(arm_list([3.5, 6.5], [9.5, 6.5], 16, 12), h.to_vec());
    let v: [(usize, usize, u8); 16] = [
        (6, 3, 138), (7, 3, 138), (6, 4, 191), (7, 4, 191), (6, 5, 191), (7, 5, 191), (6, 6, 191), (7, 6, 191),
        (6, 7, 191), (7, 7, 191), (6, 8, 191), (7, 8, 191), (6, 9, 191), (7, 9, 191), (6, 10, 138), (7, 10, 138),
    ];
    assert_eq!(arm_list([6.5, 3.5], [6.5, 9.5], 16, 12), v.to_vec());
    // a point at the corner of a 6 x 5 frame: the arms reach 3 pixels in, a quarter one further
    let corner_h: [(usize, usize, u8); 9] = [(0, 0, 255), (1, 0, 255), (2, 0, 255), (3, 0, 255), (4, 0, 64), (0, 1, 64), (1, 1, 64), (2, 1, 64), (3, 1, 64)];
    assert_eq!(arm_list([-3.0, 0.0], [3.0, 0.0], 6, 5), corner_h.to_vec());
    // a point outside the frame still reaches pixels 0 and 1 (x = -3, y = 4 in the plugin's rows 7)
    assert_eq!(arm_list([-6.0, 4.0], [0.0, 4.0], 16, 12), vec![(0, 3, 64), (0, 4, 255), (1, 4, 64), (0, 5, 64)]);
}

#[test]
fn the_plot_blends_per_depth_as_the_original_does() {
    let pixel = |img: &Image| -> Vec<f64> {
        match &img.samples {
            Samples::U8(s) => s.iter().map(|&v| f64::from(v)).collect(),
            Samples::U16(s) => s.iter().map(|&v| f64::from(v)).collect(),
            Samples::F32(s) => s.iter().map(|&v| f64::from(v.to_bits())).collect(),
        }
    };
    let one = |fmt: Fmt, px: [f64; 4], col: [u8; 3], a: u8| -> Vec<f64> {
        let mut img = Image::new(1, 1, fmt);
        match &mut img.samples {
            Samples::U8(s) => s.iter_mut().zip(px).for_each(|(d, v)| *d = v as u8),
            Samples::U16(s) => s.iter_mut().zip(px).for_each(|(d, v)| *d = v as u16),
            Samples::F32(s) => s.iter_mut().zip(px).for_each(|(d, v)| *d = v as f32),
        }
        plot(&mut img, 0, col, a);
        pixel(&img)
    };
    // 8 bpc: [A, R, G, B]
    assert_eq!(one(Fmt::U8, [255.0, 10.0, 200.0, 128.0], [255, 0, 0], 64), vec![255.0, 71.0, 150.0, 96.0]);
    assert_eq!(one(Fmt::U8, [128.0, 64.0, 32.0, 16.0], [0, 255, 0], 112), vec![184.0, 36.0, 130.0, 9.0]);
    assert_eq!(one(Fmt::U8, [255.0, 90.0, 90.0, 90.0], [10, 20, 30], 255), vec![255.0, 10.0, 20.0, 30.0]);
    // 16 bpc: the colour bytes and the alpha are shifted up by 8 first, and the quotient truncated
    assert_eq!(one(Fmt::U16, [65535.0, 1000.0, 40000.0, 30000.0], [255, 0, 0], 64), vec![65535.0, 17070.0, 29999.0, 22499.0]);
    assert_eq!(one(Fmt::U16, [32768.0, 20000.0, 10000.0, 5000.0], [0, 255, 0], 255), vec![65407.0, 77.0, 65064.0, 19.0]);
    assert_eq!(one(Fmt::U16, [0.0, 0.0, 0.0, 0.0], [255, 255, 255], 112), vec![28672.0, 28560.0, 28560.0, 28560.0]);
    // float: the bits of [A, R, G, B]
    let bits = |v: [u32; 4]| -> Vec<f64> { v.iter().map(|&b| f64::from(b)).collect() };
    assert_eq!(one(Fmt::F32, [1.0, 0.25, 0.5, 0.75], [255, 0, 0], 64), bits([0x3f800000, 0x3ee06061, 0x3ebfbfc0, 0x3f0fcfd0]));
    assert_eq!(one(Fmt::F32, [0.5, 0.1, 0.2, 0.3], [0, 255, 0], 112), bits([0x3f383838, 0x3d65b27f, 0x3f0d26c1, 0x3e2c45e0]));
    assert_eq!(one(Fmt::F32, [1.0, 0.1, 0.2, 0.3], [10, 20, 30], 255), bits([0x3f800000, 0x3d20a0a1, 0x3da0a0a1, 0x3df0f0f1]));
    // alpha 0 touches nothing
    assert_eq!(one(Fmt::U8, [255.0, 1.0, 2.0, 3.0], [9, 9, 9], 0), vec![255.0, 1.0, 2.0, 3.0]);
}

/// A `w x h` opaque black frame of `depth` through `Image::from_frame`, the crosses drawn, back
/// through `to_frame`.
fn cross_frame(w: usize, h: usize, depth: fx_core::Depth, pts: &[([f32; 2], [u8; 3])]) -> fx_core::Frame {
    let mut f = fx_core::Frame::new(w, h, depth);
    match &mut f.pixels {
        fx_core::Pixels::U8(p) => p.iter_mut().for_each(|v| *v = [0, 0, 0, 255]),
        fx_core::Pixels::U16(p) => p.iter_mut().for_each(|v| *v = [0, 0, 0, 65535]),
        fx_core::Pixels::F32(p) => p.iter_mut().for_each(|v| *v = [0.0, 0.0, 0.0, 1.0]),
    }
    let mut img = Image::from_frame(&f);
    draw_crosses(&mut img, pts, 1.0);
    img.to_frame()
}

#[test]
fn cross_of_a_white_point_on_a_pixel_centre_at_8_bpc() {
    let want: [(usize, usize, u8); 37] = [
        (12, 6, 64), (11, 7, 64), (12, 7, 255), (13, 7, 64), (11, 8, 64), (12, 8, 255), (13, 8, 64), (9, 9, 64),
        (10, 9, 64), (11, 9, 112), (12, 9, 255), (13, 9, 112), (14, 9, 64), (15, 9, 64), (8, 10, 64), (9, 10, 255),
        (10, 10, 255), (11, 10, 255), (12, 10, 255), (13, 10, 255), (14, 10, 255), (15, 10, 255), (16, 10, 64),
        (9, 11, 64), (10, 11, 64), (11, 11, 112), (12, 11, 255), (13, 11, 112), (14, 11, 64), (15, 11, 64),
        (11, 12, 64), (12, 12, 255), (13, 12, 64), (11, 13, 64), (12, 13, 255), (13, 13, 64), (12, 14, 64),
    ];
    let f = cross_frame(24, 20, fx_core::Depth::U8, &[([12.0, 10.0], [255, 255, 255])]);
    let fx_core::Pixels::U8(p) = &f.pixels else { unreachable!() };
    let mut got = Vec::new();
    for (i, v) in p.iter().enumerate() {
        assert_eq!(v[3], 255);
        assert!(v[0] == v[1] && v[1] == v[2]);
        if v[0] != 0 {
            got.push((i % 24, i / 24, v[0]));
        }
    }
    assert_eq!(got, want.to_vec());
}

#[test]
fn cross_off_the_pixel_centres_at_32_bpc_is_bit_exact() {
    // point (12.3, 10.7): the bits of R (G and B equal R, alpha 1.0)
    let want: [(usize, usize, u32); 28] = [
        (12, 7, 0x3efafafb), (13, 7, 0x3e848485), (12, 8, 0x3f72f2f3), (13, 8, 0x3f0c8c8d), (12, 9, 0x3f72f2f3),
        (13, 9, 0x3f0c8c8d), (9, 10, 0x3efafafb), (10, 10, 0x3f0c8c8d), (11, 10, 0x3f0c8c8d), (12, 10, 0x3f7a1d40),
        (13, 10, 0x3f4bef12), (14, 10, 0x3f0c8c8d), (15, 10, 0x3f0c8c8d), (16, 10, 0x3e848485), (9, 11, 0x3f53d3d4),
        (10, 11, 0x3f72f2f3), (11, 11, 0x3f72f2f3), (12, 11, 0x3f7f55ac), (13, 11, 0x3f7a1d40), (14, 11, 0x3f72f2f3),
        (15, 11, 0x3f72f2f3), (16, 11, 0x3efafafb), (12, 12, 0x3f72f2f3), (13, 12, 0x3f0c8c8d), (12, 13, 0x3f72f2f3),
        (13, 13, 0x3f0c8c8d), (12, 14, 0x3f53d3d4), (13, 14, 0x3efafafb),
    ];
    // the 16.16 reader of 12.3 and 10.7
    let p = [(12.3f64 * 65536.0).round() as f32 / 65536.0, (10.7f64 * 65536.0).round() as f32 / 65536.0];
    let f = cross_frame(24, 20, fx_core::Depth::F32, &[(p, [255, 255, 255])]);
    let fx_core::Pixels::F32(px) = &f.pixels else { unreachable!() };
    let mut got = Vec::new();
    for (i, v) in px.iter().enumerate() {
        assert_eq!(v[3], 1.0);
        assert!(v[0].to_bits() == v[1].to_bits() && v[1].to_bits() == v[2].to_bits());
        if v[0] != 0.0 {
            got.push((i % 24, i / 24, v[0].to_bits()));
        }
    }
    assert_eq!(got, want.to_vec());
}

#[test]
fn cross_at_16_bpc_in_the_host_range() {
    // white at (12, 10): the wire values of R
    let want: [(usize, usize, u16); 37] = [
        (12, 6, 16320), (11, 7, 16320), (12, 7, 65025), (13, 7, 16320), (11, 8, 16320), (12, 8, 65025),
        (13, 8, 16320), (9, 9, 16320), (10, 9, 16320), (11, 9, 28560), (12, 9, 65089), (13, 9, 28560),
        (14, 9, 16320), (15, 9, 16320), (8, 10, 16320), (9, 10, 65025), (10, 10, 65025), (11, 10, 65087),
        (12, 10, 65279), (13, 10, 65087), (14, 10, 65025), (15, 10, 65025), (16, 10, 16320), (9, 11, 16320),
        (10, 11, 16320), (11, 11, 28560), (12, 11, 65089), (13, 11, 28560), (14, 11, 16320), (15, 11, 16320),
        (11, 12, 16320), (12, 12, 65025), (13, 12, 16320), (11, 13, 16320), (12, 13, 65025), (13, 13, 16320),
        (12, 14, 16320),
    ];
    let f = cross_frame(24, 20, fx_core::Depth::U16, &[([12.0, 10.0], [255, 255, 255])]);
    let fx_core::Pixels::U16(p) = &f.pixels else { unreachable!() };
    let got: Vec<(usize, usize, u16)> = p.iter().enumerate().filter(|(_, v)| v[0] != 0).map(|(i, v)| (i % 24, i / 24, v[0])).collect();
    assert_eq!(got, want.to_vec());
}

#[test]
fn cross_colour_blend_and_transparent_pixels_at_8_bpc() {
    // red over grey 128: the side band and the double side band (MEASURED 160 / 96 and 184 / 72)
    let mut f = fx_core::Frame::new(24, 20, fx_core::Depth::U8);
    let fx_core::Pixels::U8(p) = &mut f.pixels else { unreachable!() };
    p.iter_mut().for_each(|v| *v = [128, 128, 128, 255]);
    let mut img = Image::from_frame(&f);
    draw_crosses(&mut img, &[([12.0, 10.0], [255, 0, 0])], 1.0);
    let f = img.to_frame();
    let fx_core::Pixels::U8(p) = &f.pixels else { unreachable!() };
    assert_eq!(p[9 * 24 + 10], [160, 96, 96, 255]);
    assert_eq!(p[9 * 24 + 11], [184, 72, 72, 255]);
    assert_eq!(p[10 * 24 + 12], [255, 0, 0, 255]);
    // a transparent clip: the colour itself with the arm's alpha (MEASURED 255 with 64 and 112)
    let f = fx_core::Frame::new(24, 20, fx_core::Depth::U8);
    let mut img = Image::from_frame(&f);
    draw_crosses(&mut img, &[([12.0, 10.0], [255, 255, 255])], 1.0);
    let f = img.to_frame();
    let fx_core::Pixels::U8(p) = &f.pixels else { unreachable!() };
    assert_eq!(p[9 * 24 + 10], [255, 255, 255, 64]);
    assert_eq!(p[9 * 24 + 11], [255, 255, 255, 112]);
    assert_eq!(p[0], [0, 0, 0, 0]);
}

#[test]
fn crosses_skip_points_that_are_not_numbers_and_clip_at_the_borders() {
    let f = cross_frame(6, 5, fx_core::Depth::U8, &[([f32::NAN, 2.0], [255, 255, 255]), ([2.0, f32::INFINITY], [255, 255, 255]), ([4.0e9, 1.0], [255, 255, 255])]);
    let fx_core::Pixels::U8(p) = &f.pixels else { unreachable!() };
    assert!(p.iter().all(|v| *v == [0, 0, 0, 255]));
    // a point at the corner: the arms reach x 0..=3 on row 0 and rows 0..=3 on column 0
    let f = cross_frame(6, 5, fx_core::Depth::U8, &[([0.0, 0.0], [255, 255, 255])]);
    let fx_core::Pixels::U8(p) = &f.pixels else { unreachable!() };
    assert_eq!((0..6).map(|x| p[x][0]).collect::<Vec<_>>(), vec![255, 255, 255, 255, 64, 0]);
    assert_eq!((0..5).map(|y| p[y * 6][0]).collect::<Vec<_>>(), vec![255, 255, 255, 255, 64]);
    let mut empty = Image::new(0, 3, Fmt::U8);
    draw_crosses(&mut empty, &[([0.0, 0.0], [1, 1, 1])], 1.0);
    assert_eq!(empty.sample_count(), 0);
}
