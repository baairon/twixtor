//! The Draw Geom overlay (rule G3): the tracking points drawn over the
//! Source display and the create-vectors Debug display, after the source copy, as opaque discs in
//! the point colours.
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
