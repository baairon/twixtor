//! The Draw Geom overlay: the tracking points drawn over the
//! Source display and the create-vectors Debug display, after the source copy. [`draw_points`]
//! draws opaque discs in the point colours; [`draw_crosses`] is the glyph of
//! `CreateMotionVectors_64`, read from its code: two
//! antialiased segments per point.
//!
//! Claim tags: V established, I inferred, U not established (a rule of this project, pinned in
//! the parity gaps register until a reference render grades it).
//!
//! Precision contract: f32 evaluated in the order written, no fused multiply add, the points
//! drawn one after the other in their order (a later point on top); no arithmetic reaches the
//! pixels, which are overwritten. With `r = 1.5 * ratio` (f32) a pixel `(x, y)` is covered when
//! `dx * dx + dy * dy <= r * r` with `dx = x as f32 - px` and `dy = y as f32 - py` (pixel centres
//! at whole numbers; `<= 2.25` at the ratio 1). The colour `[c0, c1, c2]` (`ptN_color`) is stored
//! opaque: alpha white and `c` at 8 bpc, `c * 257` at 16 bpc, `c as f32 / 255.0` in float.

use crate::image::{Image, Samples, CHANNELS};

#[cfg(test)]
mod tests;

/// Draws the points `pts` (position in output pixels, `ptN_color`) into the engine image `img`
/// (rule G3, U): each point an opaque disc of radius `1.5 * ratio` in its colour, covering the
/// pixels with `dx * dx + dy * dy <= r * r`, drawn in the order of `pts`, so a later point is on
/// top. `ratio` is the downsample ratio (1 in this harness; V: the line widths are 1.5 and
/// 3.0 times it, and the disc takes the first). A point with a NaN or infinite coordinate, or a
/// ratio that is negative, NaN or infinite, draws nothing (U, ours). Records gap G3 when there
/// is a point to draw.
pub fn draw_points(img: &mut Image, pts: &[([f32; 2], [u8; 3])], ratio: f32) {
    if pts.is_empty() {
        return;
    }
    fx_core::gaps::record(&crate::gaps::G3);
    let r = 1.5f32 * ratio;
    if !(r >= 0.0) || !r.is_finite() {
        return;
    }
    let r2 = r * r;
    let (w, h) = (img.width, img.height);
    if w == 0 || h == 0 {
        return;
    }
    for &(p, col) in pts {
        if !(p[0].is_finite() && p[1].is_finite()) {
            continue;
        }
        // the box of the disc, one pixel wider on each side; the test below decides
        let Some((x0, x1)) = span(p[0], r, w) else { continue };
        let Some((y0, y1)) = span(p[1], r, h) else { continue };
        for y in y0..=y1 {
            let dy = y as f32 - p[1];
            for x in x0..=x1 {
                let dx = x as f32 - p[0];
                if dx * dx + dy * dy <= r2 {
                    put(img, y * w + x, col);
                }
            }
        }
    }
}

/// Half length of each arm of the Draw Geom cross at the ratio 1, in output pixels.
pub const CROSS_HALF_LENGTH: f32 = 3.0;
/// Line width of the arms at the ratio 1.
pub const CROSS_WIDTH: f32 = 1.5;

/// The Draw Geom glyph of `CreateMotionVectors_64` (V: read from the code of the
/// original and reproduced call for call: for every segment it draws, the list of pixels it plots
/// with their alpha is the list computed here, 264 segments of nine probe cases at all three
/// depths). Every point of `pts` (position in output pixels of the image as the plugin sees it,
/// pixel centres at whole numbers; `ptN_color`) is drawn, in the order of `pts`, as the horizontal
/// segment from `x - h` to `x + h` and then the vertical one from `y - h` to `y + h`,
/// `h = ratio * 3.0` and the line width `w = ratio * 1.5` (single precision products).
///
/// Per segment ([`segment`]): `r = (w - 1.0) * 0.5` (the subtraction in single, the product in
/// double, narrowed) when `w > 1.0`, else 0; `r1 = r + 1.0` (double, narrowed). The pixels visited
/// are `x` from `trunc((floor(min x) - r1) - 1.0)` to `trunc((ceil(max x) + r1) + 1.0)` and `y`
/// likewise (single precision, truncation towards zero; a value outside the 32 bit range or NaN
/// gives the smallest integer, so a segment with such an end draws nothing). With `D` the distance
/// of the pixel (`x`, `y` as floats) to the segment ([`distance`]) and `d2 = D * D` (single): the
/// alpha byte is 255 when `d2 <= r * r`, `trunc(double((r1 - sqrt(d2)) * 255.0) + 0.5)` when
/// `d2 < r1 * r1`, and the pixel is not touched otherwise (or when the alpha is 0).
///
/// The plot ([`plot`]) works in the premultiplied engine image and clips at its borders. 8 bpc:
/// alpha 255 stores the colour with alpha 255; else `t = a / 255.0` and each of the colour channels
/// and the alpha channel `d` becomes `trunc(((c - d) * t + d) + 0.5)` (`c` the colour byte, 255 for
/// the alpha; single products and sums, the +0.5 in double). 16 bpc: the colour bytes and `a` are
/// multiplied by 256 first (so a full alpha byte is 65280, not 65535), `t = (a * 256) / 65535.0`,
/// the same formula without the +0.5 (truncation), alpha channel with 65535. Float: the colour is
/// `c / 255.0`, `t = a / 255.0`; `t == 1.0` stores colour and alpha 1.0, else the same formula
/// (alpha channel with 1.0), no rounding.
///
/// A point with an end outside the 32 bit range or not a number draws nothing. It follows the
/// original and records no gap. [`draw_points`] (the disc, rule G3) is not this plugin's glyph and
/// stays as it was.
pub fn draw_crosses(img: &mut Image, pts: &[([f32; 2], [u8; 3])], ratio: f32) {
    if pts.is_empty() {
        return;
    }
    let half = ratio * CROSS_HALF_LENGTH;
    let width = ratio * CROSS_WIDTH;
    for &(p, col) in pts {
        segment(img, [p[0] - half, p[1]], [p[0] + half, p[1]], width, col);
        segment(img, [p[0], p[1] - half], [p[0], p[1] + half], width, col);
    }
}

/// `cvttss2si` with a 32 bit result: truncation; NaN and out of range give `i32::MIN`.
fn cvtt32(x: f32) -> i32 {
    if x.is_nan() || x >= 2147483648.0f32 || x < -2147483648.0f32 {
        i32::MIN
    } else {
        x as i32
    }
}

/// The distance of the point `p` to the segment `a b` (single precision, in the order of the
/// original): with `d = b - a`, `q = p - a` and `t = qy * dy + qx * dx`, a `t` that is not above
/// 0.0001 (compared in double) measures to `a` (`sqrt(qy * qy + qx * qx)`); else with
/// `u = (by - py) * dy + (bx - px) * dx`, a negative `u` measures to `b`
/// (`sqrt((py - by)^2 + (px - bx)^2)`); else the distance to the line,
/// `sqrt((qy * dx - dy * qx)^2 / (dy * dy + dx * dx))`.
pub(crate) fn distance(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let (qx, qy) = (p[0] - a[0], p[1] - a[1]);
    let t = qy * dy + qx * dx;
    let d2 = if !(f64::from(t) > 0.0001f64) {
        qy * qy + qx * qx
    } else {
        let u = (b[1] - p[1]) * dy + (b[0] - p[0]) * dx;
        if 0.0f32 > u {
            let (ex, ey) = (p[0] - b[0], p[1] - b[1]);
            ey * ey + ex * ex
        } else {
            let cross = qy * dx - dy * qx;
            (cross * cross) / (dy * dy + dx * dx)
        }
    };
    d2.sqrt()
}

/// The line radius `r` and `r1 = r + 1` of a line width `w` (see [`draw_crosses`]).
fn radii(w: f32) -> (f32, f32) {
    let r = if 1.0f32 < w { (f64::from(w - 1.0f32) * 0.5f64) as f32 } else { 0.0f32 };
    (r, (f64::from(r) + 1.0f64) as f32)
}

/// The alpha byte of the pixel `p` for a segment `a b` of line width `w` (see [`draw_crosses`]); 0
/// when it is not plotted.
pub(crate) fn arm_alpha(a: [f32; 2], b: [f32; 2], w: f32, p: [f32; 2]) -> u8 {
    let (r, r1) = radii(w);
    let d = distance(a, b, p);
    let d2 = d * d;
    if r * r >= d2 {
        255
    } else if r1 * r1 > d2 {
        ((f64::from((r1 - d2.sqrt()) * 255.0f32) + 0.5f64) as i64) as u8
    } else {
        0
    }
}

/// One antialiased segment of the cross (see [`draw_crosses`]).
fn segment(img: &mut Image, a: [f32; 2], b: [f32; 2], w: f32, col: [u8; 3]) {
    let (iw, ih) = (img.width, img.height);
    if iw == 0 || ih == 0 {
        return;
    }
    let (_, r1) = radii(w);
    let (min_x, max_x) = if b[0] <= a[0] { (b[0], a[0]) } else { (a[0], b[0]) };
    let (min_y, max_y) = if b[1] <= a[1] { (b[1], a[1]) } else { (a[1], b[1]) };
    let x_lo = cvtt32((min_x.floor() - r1) - 1.0f32);
    let x_hi = cvtt32((max_x.ceil() + r1) + 1.0f32);
    let y_lo = cvtt32((min_y.floor() - r1) - 1.0f32);
    let y_hi = cvtt32((max_y.ceil() + r1) + 1.0f32);
    // pixels outside the image are not plotted, so only the part of the box inside it is visited
    let (x0, x1) = (i64::from(x_lo).max(0), i64::from(x_hi).min(iw as i64 - 1));
    let (y0, y1) = (i64::from(y_lo).max(0), i64::from(y_hi).min(ih as i64 - 1));
    for y in y0..=y1 {
        for x in x0..=x1 {
            let alpha = arm_alpha(a, b, w, [x as f32, y as f32]);
            if alpha > 0 {
                plot(img, y as usize * iw + x as usize, col, alpha);
            }
        }
    }
}

/// The plot of the original on pixel `i` of the premultiplied image (see [`draw_crosses`]).
pub(crate) fn plot(img: &mut Image, i: usize, col: [u8; 3], a: u8) {
    if a == 0 {
        return;
    }
    let o = i * CHANNELS;
    match &mut img.samples {
        Samples::U8(s) => {
            if a == 255 {
                s[o..o + CHANNELS].copy_from_slice(&[255, col[0], col[1], col[2]]);
                return;
            }
            let t = f32::from(a) / 255.0f32;
            let mix = |c: f32, d: f32| -> u8 { ((f64::from((c - d) * t + d) + 0.5f64) as i64) as u8 };
            let (da, dr, dg, db) = (f32::from(s[o]), f32::from(s[o + 1]), f32::from(s[o + 2]), f32::from(s[o + 3]));
            s[o + 1] = mix(f32::from(col[0]), dr);
            s[o + 2] = mix(f32::from(col[1]), dg);
            s[o + 3] = mix(f32::from(col[2]), db);
            s[o] = mix(255.0f32, da);
        }
        Samples::U16(s) => {
            let c16 = |c: u8| f32::from(u16::from(c) << 8);
            let a16 = u16::from(a) << 8;
            if a16 == 0xffff {
                s[o..o + CHANNELS].copy_from_slice(&[0xffff, c16(col[0]) as u16, c16(col[1]) as u16, c16(col[2]) as u16]);
                return;
            }
            let t = f32::from(a16) / 65535.0f32;
            let mix = |c: f32, d: f32| -> u16 { (crate::blend::cvtt((c - d) * t + d)) as u16 };
            let (da, dr, dg, db) = (f32::from(s[o]), f32::from(s[o + 1]), f32::from(s[o + 2]), f32::from(s[o + 3]));
            s[o + 1] = mix(c16(col[0]), dr);
            s[o + 2] = mix(c16(col[1]), dg);
            s[o + 3] = mix(c16(col[2]), db);
            s[o] = mix(65535.0f32, da);
        }
        Samples::F32(s) => {
            let t = f32::from(a) / 255.0f32;
            let cf = |c: u8| f32::from(c) / 255.0f32;
            if t == 1.0f32 {
                s[o..o + CHANNELS].copy_from_slice(&[t, cf(col[0]), cf(col[1]), cf(col[2])]);
                return;
            }
            let mix = |c: f32, d: f32| -> f32 { (c - d) * t + d };
            let (da, dr, dg, db) = (s[o], s[o + 1], s[o + 2], s[o + 3]);
            s[o + 1] = mix(cf(col[0]), dr);
            s[o + 2] = mix(cf(col[1]), dg);
            s[o + 3] = mix(cf(col[2]), db);
            s[o] = mix(1.0f32, da);
        }
    }
}

/// The pixels `lo..=hi` of a line of `len` that a disc of radius `r` at `c` can reach (one more on
/// each side), or `None` when it misses the line. The ends saturate (a coordinate or radius beyond
/// the range of `i64` gives `i64::MIN` or `i64::MAX`, not an overflow).
pub(crate) fn span(c: f32, r: f32, len: usize) -> Option<(usize, usize)> {
    let lo = ((c - r).floor() as i64).saturating_sub(1);
    let hi = ((c + r).ceil() as i64).saturating_add(1);
    let last = len as i64 - 1;
    if hi < 0 || lo > last {
        return None;
    }
    Some((lo.max(0) as usize, hi.min(last) as usize))
}

/// Pixel `i` becomes the opaque colour `col` in the depth of `img`.
fn put(img: &mut Image, i: usize, col: [u8; 3]) {
    let o = i * CHANNELS;
    match &mut img.samples {
        Samples::U8(s) => s[o..o + CHANNELS].copy_from_slice(&[255, col[0], col[1], col[2]]),
        Samples::U16(s) => {
            let c16 = |c: u8| u16::from(c) * 257;
            s[o..o + CHANNELS].copy_from_slice(&[65535, c16(col[0]), c16(col[1]), c16(col[2])]);
        }
        Samples::F32(s) => {
            let cf = |c: u8| f32::from(c) / 255.0f32;
            s[o..o + CHANNELS].copy_from_slice(&[1.0, cf(col[0]), cf(col[1]), cf(col[2])]);
        }
    }
}
