//! Filters of the Weighted Blend step: the separable `blur`, the `difference_mask` of two frames
//! and its clean-up (`clean_mask`), `invert_mask`, the masked `composite_masked`, and
//! `sharpen_weight` / `sharpen_step`, the arithmetic and the order of calls of one half of
//! Weighted Blend.
//!
//! The masks are 1 channel 8 bit images in every pixel format (`Mask`). Rows are processed in
//! parallel with `fx_core::par_rows`; every output pixel depends only on input data, so the result
//! does not depend on the thread count.

use crate::blend::{cvtt, lerp};
use crate::holefill::VerticalTaps;
use crate::nanrule as nr;
use crate::image::{Fmt, Image, Samples, CHANNELS};
use fx_core::par_rows;

/// A 1 channel 8 bit image, tightly packed, top row first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mask {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl Mask {
    /// An all zero mask.
    pub fn new(width: usize, height: usize) -> Mask {
        Mask { width, height, data: vec![0; width * height] }
    }

    /// A mask with every pixel set to `value`.
    pub fn filled(width: usize, height: usize, value: u8) -> Mask {
        Mask { width, height, data: vec![value; width * height] }
    }

    /// The pixel at column `x`, row `y` (row 0 on top).
    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.width + x]
    }

    /// Sets the pixel at column `x`, row `y`.
    pub fn set(&mut self, x: usize, y: usize, v: u8) {
        self.data[y * self.width + x] = v;
    }

    /// True when every pixel equals `value`.
    pub fn is_uniform(&self, value: u8) -> bool {
        self.data.iter().all(|v| *v == value)
    }
}

// ------------------------------------------------------------------------------------------------
// blur
// ------------------------------------------------------------------------------------------------

/// Radius of the blur in the sharpening step.
pub const BLUR_RADIUS: f32 = 3.0;

/// Truncation toward zero to `i32`; NaN and values out of range give `i32::MIN`.
#[inline(always)]
fn cvtt32(x: f32) -> i32 {
    if x.is_nan() || x >= 2_147_483_648.0 || x < -2_147_483_648.0 {
        i32::MIN
    } else {
        x as i32
    }
}

/// The half kernel of the blur. With `count = (int)radius + 1`:
/// `t = (radius / 3.0f) * (radius / 3.0f)`, `w[i] = powf(2.0f, -((i * i) / t))` for `i < count`,
/// `sum = w[0] + w[1] + ...` accumulated in single precision in that order, and every tap is
/// divided by `sum * 2.0f - w[0]`, so that the symmetric kernel
/// `w[count-1] .. w[1] w[0] w[1] .. w[count-1]` sums to one. All of it in single precision. For
/// `radius = 3.0` the exponents are the integers 0, -1, -4, -9 and the taps are
/// `[1, 1/2, 1/16, 1/512] / 2.12890625`.
pub fn blur_taps(radius: f32) -> Vec<f32> {
    let t = radius / 3.0f32;
    let t2 = t * t;
    let count = (radius as i32).max(0) as usize + 1;
    let mut w = Vec::with_capacity(count);
    let mut sum = 0.0f32;
    for i in 0..count {
        let fi = i as f32;
        let x = fi * fi;
        let x = x / t2;
        let x = -x;
        let p = 2.0f32.powf(x);
        w.push(p);
        sum += p;
    }
    let denom = sum * 2.0f32 - w[0];
    for p in &mut w {
        *p /= denom;
    }
    w
}

/// A sample type of the blur: how it is turned into the float lanes of the line buffer and how a result
/// lane is stored.
trait Sample: Copy + Send + Sync {
    /// Value of the padding around a line: taps outside the line read it, and in the edge formula a
    /// tap counts as outside the image when its lane equals it. It is -1.0 in all lanes for every
    /// sample type and only matters for pixel values that are exactly the sentinel.
    const SENTINEL: [f32; 4] = [-1.0; 4];
    /// True for the float type: only its lanes can hold a NaN (`crate::nanrule`).
    const FLOAT: bool;
    fn load(self) -> f32;
    fn store(v: f32) -> Self;
}

/// The store of the integer kernels: `trunc(v + 0.5)`, converted to float, clamped to `0..=max`,
/// then `trunc(x + 0.5)` again (identity on integers).
#[inline(always)]
fn store_int(v: f32, max: f32) -> i32 {
    let t = cvtt32(v + 0.5f32);
    let f = t as f32;
    let f = if f > 0.0 { f } else { 0.0 };
    let f = if f < max { f } else { max };
    cvtt32(f + 0.5f32)
}

impl Sample for u8 {
    const FLOAT: bool = false;
    fn load(self) -> f32 {
        self as f32
    }
    fn store(v: f32) -> u8 {
        store_int(v, 255.0) as u8
    }
}

impl Sample for u16 {
    const FLOAT: bool = false;
    fn load(self) -> f32 {
        self as f32
    }
    fn store(v: f32) -> u16 {
        store_int(v, 65535.0) as u16
    }
}

impl Sample for f32 {
    const FLOAT: bool = true;
    fn load(self) -> f32 {
        self
    }
    fn store(v: f32) -> f32 {
        v
    }
}

#[inline(always)]
fn load4<T: Sample>(px: &[T]) -> [f32; 4] {
    [px[0].load(), px[1].load(), px[2].load(), px[3].load()]
}

/// One output pixel of the 1D filter of the blur, for rows and columns alike. `get(j)` returns
/// the lanes of the pixel `j` of the line of `n` pixels. All four lanes are independent and use
/// single precision.
///
/// Three regions, by the position `i` in the line (`count = taps.len()`):
/// * the first `min(n, count)` pixels and the last `n - min(n, count) - max(0, n - 2 * count)`
///   pixels use the edge formula: `acc = c * w0`, `wsum = w0`; for `k = count - 1` down to 1:
///   `wr = (right != S) * w[k]` (the comparison gives 1.0 or 0.0), `wsum = wsum + wr`,
///   `rt = wr * right`, `wl = (left != S) * w[k]`, `lt = left * wl`, `wsum = wsum + wl`,
///   `acc = (acc + rt) + lt`; the result is `acc / wsum`, which renormalises the kernel over the
///   taps that lie inside the image (taps outside hold the sentinel `S`);
/// * the pixels in between use the plain formula `acc = c * w0`, then for `k = count - 1` down to 1
///   `acc = acc + (right + left) * w[k]`, no division.
///
/// With `left_first` the edge formula takes the two taps of each distance the other way round
/// (`left` in the place of `right` and back: `wsum` adds the left weight first and
/// `acc = (acc + lt) + rt`); the plain formula is the same either way.
#[inline]
fn conv_pixel(i: usize, n: usize, taps: &[f32], s: [f32; 4], get: impl Fn(usize) -> [f32; 4], left_first: bool, float: bool) -> [f32; 4] {
    let v = conv_pixel_with::<nr::Plain>(i, n, taps, s, &get, left_first);
    // a NaN in the result (float images only): the same arithmetic again with the NaN rule (`crate::nanrule`); numbers are the same bits
    if float && nr::any_nan(&v) {
        conv_pixel_with::<nr::Rule>(i, n, taps, s, &get, left_first)
    } else {
        v
    }
}

#[inline(always)]
fn conv_pixel_with<A: nr::Arith>(i: usize, n: usize, taps: &[f32], s: [f32; 4], get: &impl Fn(usize) -> [f32; 4], left_first: bool) -> [f32; 4] {
    let count = taps.len();
    let left = n.min(count);
    let mid = n.saturating_sub(2 * count);
    let c = get(i);
    let w0 = taps[0];
    let mut acc = [A::mul(c[0], w0), A::mul(c[1], w0), A::mul(c[2], w0), A::mul(c[3], w0)];
    if i >= left && i < left + mid {
        for k in (1..count).rev() {
            let r = get(i + k);
            let l = get(i - k);
            let wk = taps[k];
            for lane in 0..4 {
                acc[lane] = A::add(acc[lane], A::mul(A::add(r[lane], l[lane]), wk));
            }
        }
        return acc;
    }
    let mut wsum = [w0; 4];
    for k in (1..count).rev() {
        let r = if i + k < n { get(i + k) } else { s };
        let l = if i >= k { get(i - k) } else { s };
        let (r, l) = if left_first { (l, r) } else { (r, l) };
        let wk = taps[k];
        for lane in 0..4 {
            let wr = (if r[lane] != s[lane] { 1.0f32 } else { 0.0f32 }) * wk;
            wsum[lane] = wsum[lane] + wr;
            let rt = A::mul(wr, r[lane]);
            let wl = (if l[lane] != s[lane] { 1.0f32 } else { 0.0f32 }) * wk;
            let lt = A::mul(l[lane], wl);
            wsum[lane] = wsum[lane] + wl;
            acc[lane] = A::add(A::add(acc[lane], rt), lt);
        }
    }
    for lane in 0..4 {
        acc[lane] = A::div(acc[lane], wsum[lane]);
    }
    acc
}

/// The separable blur of one image: rows, then columns; `vtaps` is the tap order of the edge rows
/// of the column pass ([`blur_with`]).
fn blur_plane<T: Sample>(data: &mut [T], width: usize, height: usize, taps: &[f32], vtaps: VerticalTaps) {
    if width == 0 || height == 0 {
        return;
    }
    let rl = width * CHANNELS;
    // Pass 1, along the rows: each row is read completely into the line buffer, then written.
    par_rows(data, rl, |_, row| {
        let line: Vec<[f32; 4]> = row.chunks_exact(CHANNELS).map(load4::<T>).collect();
        for x in 0..width {
            let v = conv_pixel(x, width, taps, T::SENTINEL, |j| line[j], false, T::FLOAT);
            for lane in 0..4 {
                row[x * CHANNELS + lane] = T::store(v[lane]);
            }
        }
    });
    // Pass 2, along the columns, on the result of pass 1.
    let above_first = vtaps == VerticalTaps::AboveFirst;
    let snap: Vec<T> = data.to_vec();
    par_rows(data, rl, |y, row| {
        for x in 0..width {
            let v = conv_pixel(y, height, taps, T::SENTINEL, |j| load4::<T>(&snap[j * rl + x * CHANNELS..]), above_first, T::FLOAT);
            for lane in 0..4 {
                row[x * CHANNELS + lane] = T::store(v[lane]);
            }
        }
    });
}

/// Separable blur in place, rows then columns. All four channels (A, R, G, B of the premultiplied
/// data) are filtered alike.
///
/// Rounding of the integer formats: `clamp(trunc(v + 0.5), 0, max)`, no other adjustment; float
/// images are stored as computed. `radius < 0` does nothing (0.0 is an assumed cutoff, it only
/// matters for radii below 1e-153). The engine only uses 3.0.
pub fn blur(img: &mut Image, radius: f32) {
    blur_with(img, radius, VerticalTaps::BelowFirst)
}

/// [`blur`] with the tap order of the edge rows of the column pass: with
/// [`VerticalTaps::AboveFirst`] an edge row takes the row above (`y - k`) before the row below,
/// as `TwixtorProVectorsIn_64` does (MEASURED under the emulator); the row pass and the plain formula are
/// unchanged.
pub fn blur_with(img: &mut Image, radius: f32, vtaps: VerticalTaps) {
    if 0.0 > f64::from(radius) {
        return;
    }
    let taps = blur_taps(radius);
    let (w, h) = (img.width, img.height);
    match &mut img.samples {
        Samples::U8(d) => blur_plane(d, w, h, &taps, vtaps),
        Samples::U16(d) => blur_plane(d, w, h, &taps, vtaps),
        Samples::F32(d) => blur_plane(d, w, h, &taps, vtaps),
    }
}

// ------------------------------------------------------------------------------------------------
// masks
// ------------------------------------------------------------------------------------------------

/// Threshold passed to [`difference_mask`] by [`change_mask`].
pub const MASK_THRESHOLD: f32 = 0.4;

/// The limit of the summed channel difference:
/// `(((threshold * white) * threshold) * threshold) * 3.0f` in single precision, with `white` the
/// white level of the format (255, 65535 or 1.0). For 0.4: 48.96 (8 bpc), 12582.72 (16 bpc),
/// 0.192 (float).
pub fn mask_limit(fmt: Fmt, threshold: f32) -> f32 {
    threshold * fmt.white() * threshold * threshold * 3.0f32
}

/// The difference mask: 255 where `(|dR| + |dG|) + |dB| >= mask_limit` with
/// `d = warped - original` per channel, else 0. Alpha does not take part. Differences are formed in
/// float for every format; a NaN sum gives 0. [`change_mask`] follows it with the clean-up of
/// [`clean_mask`].
pub fn difference_mask(warped: &Image, original: &Image, threshold: f32) -> Mask {
    assert!(
        warped.same_shape(original),
        "difference_mask needs equal size and format"
    );
    let (w, h) = (warped.width, warped.height);
    let mut mask = Mask::new(w, h);
    if w == 0 || h == 0 {
        return mask;
    }
    let limit = mask_limit(warped.fmt(), threshold);
    let rl = w * CHANNELS;
    fn rows<T: Copy + Into<f32> + Sync>(mask: &mut Mask, a: &[T], b: &[T], rl: usize, limit: f32) {
        let w = mask.width;
        par_rows(&mut mask.data, w, |y, row| {
            let ar = &a[y * rl..(y + 1) * rl];
            let br = &b[y * rl..(y + 1) * rl];
            for x in 0..w {
                let p = x * CHANNELS;
                let d = |c: usize| -> f32 {
                    let (va, vb): (f32, f32) = (ar[p + c].into(), br[p + c].into());
                    (va - vb).abs()
                };
                let sum = (d(1) + d(2)) + d(3);
                row[x] = if sum >= limit { 0xff } else { 0 };
            }
        });
    }
    match (&warped.samples, &original.samples) {
        (Samples::U8(a), Samples::U8(b)) => rows(&mut mask, a, b, rl, limit),
        (Samples::U16(a), Samples::U16(b)) => rows(&mut mask, a, b, rl, limit),
        (Samples::F32(a), Samples::F32(b)) => rows(&mut mask, a, b, rl, limit),
        _ => unreachable!("shapes were checked"),
    }
    mask
}

/// `255 - b` for every pixel.
pub fn invert_mask(mask: &mut Mask) {
    for v in &mut mask.data {
        *v = 255 - *v;
    }
}

/// Sum of the eight samples read from the snapshot `t` around the destination `(x + 1, ...)`: the
/// columns `x`, `x + 1`, `x + 2` of the rows `r0`, `r1`, `r2` without the middle sample of `r1`.
/// It is the common core of [`mask_prune`] and [`mask_fill`].
#[inline(always)]
fn ring_sum(t: &Mask, x: usize, r0: usize, r1: usize, r2: usize) -> i32 {
    let at = |y: usize, dx: usize| -> i32 { i32::from(t.data[y * t.width + x + dx]) };
    at(r0, 1) + at(r0, 0) + at(r0, 2) + at(r1, 0) + at(r1, 2) + at(r2, 0) + at(r2, 1) + at(r2, 2)
}

/// Despeckle: from a snapshot of the mask, every destination pixel `(c, r)` with
/// `1 <= c <= width - 2` and `2 <= r <= height - 3` that is not zero is set to zero when the sum of
/// its eight neighbours is below `limit`. The neighbours are the rows `r - 2`, `r` and `r + 2` (not
/// `r - 1`, `r + 1`) and the columns `c - 1`, `c`, `c + 1`, without the centre: `(r - 2, c - 1)`,
/// `(r - 2, c)`, `(r - 2, c + 1)`, `(r, c - 1)`, `(r, c + 1)`, `(r + 2, c - 1)`, `(r + 2, c)`,
/// `(r + 2, c + 1)`. Needs `height >= 5` and `width >= 3`.
pub fn mask_prune(mask: &mut Mask, limit: i32) {
    let (w, h) = (mask.width, mask.height);
    if h < 5 || w < 3 {
        return;
    }
    let t = mask.clone();
    for y0 in 0..h - 4 {
        for x in 0..w - 2 {
            let sum = ring_sum(&t, x, y0, y0 + 2, y0 + 4);
            let dst = (y0 + 2) * w + x + 1;
            if sum < limit && mask.data[dst] != 0 {
                mask.data[dst] = 0;
            }
        }
    }
}

/// `passes` rounds of `mask_prune` with the limits 254, 509, ... (`254 + 255 * pass`).
pub fn mask_prune_passes(mask: &mut Mask, passes: u32) {
    for pass in 0..passes as i32 {
        mask_prune(mask, 254 + 255 * pass);
    }
}

/// Fill: from a snapshot of the mask, every destination pixel `(c, r)` with `1 <= c <= width - 2`
/// and `1 <= r <= height - 2` that is zero is set to `value` when the sum of its eight neighbours
/// (the full 3 x 3 ring) is above `limit`. Needs `height >= 3` and `width >= 3`.
pub fn mask_fill(mask: &mut Mask, limit: i32, value: u8) {
    let (w, h) = (mask.width, mask.height);
    if h < 3 || w < 3 {
        return;
    }
    let t = mask.clone();
    for y0 in 0..h - 2 {
        for x in 0..w - 2 {
            let sum = ring_sum(&t, x, y0, y0 + 1, y0 + 2);
            let dst = (y0 + 1) * w + x + 1;
            if limit < sum && mask.data[dst] == 0 {
                mask.data[dst] = value;
            }
        }
    }
}

/// Vertical growth of full pixels: from a snapshot, every pixel of the rows `1 ..= height - 2`
/// becomes 255 when the pixel above or below it is 255. Needs `height >= 3`.
pub fn mask_dilate_rows(mask: &mut Mask) {
    let (w, h) = (mask.width, mask.height);
    if h < 3 {
        return;
    }
    let t = mask.clone();
    for y in 1..h - 1 {
        for x in 0..w {
            if t.data[(y - 1) * w + x] == 255 || t.data[(y + 1) * w + x] == 255 {
                mask.data[y * w + x] = 255;
            }
        }
    }
}

/// Copy of the inner pixels to the outer frame: the inner rectangle spans the columns
/// `1 ..= width - 2` and the rows `1 ..= height - 2`. Row 0 gets row 1, the last row gets the row
/// before it, then in every row column 0 gets column 1 and the last column the column before it.
/// The rectangle tests are done in float and the algorithm is applied as is for any size.
pub fn mask_replicate_border(mask: &mut Mask) {
    let (w, h) = (mask.width as i64, mask.height as i64);
    if w == 0 || h == 0 {
        return;
    }
    let (p2, p3, p4, p5) = (1.0f32, 1.0f32, (w - 2) as f32, (h - 2) as f32);
    let iv14 = p5 as i32 as i64;
    let (mut fy0, mut fy1) = (p3, p5);
    if p5 < p3 {
        fy0 = p5;
        fy1 = p3;
    }
    let (mut fx0, mut fx1) = (p2, p4);
    if p4 < p2 {
        fx0 = p4;
        fx1 = p2;
    }
    let idx = |x: i64, y: i64| -> Option<usize> {
        if x >= 0 && x < w && y >= 0 && y < h {
            Some((y * w + x) as usize)
        } else {
            None
        }
    };
    let outside = |x: i64, y: i64| -> bool {
        let (xf, yf) = (x as f32, y as f32);
        yf < fy0 || fy1 < yf || fx1 < xf || xf < fx0
    };
    // rows above the rectangle take the row (int)p3
    let mut y = 0i64;
    while (y as f32) < fy0 {
        for x in 0..w {
            if outside(x, y) {
                if let (Some(d), Some(s)) = (idx(x, y), idx(x, p3 as i32 as i64)) {
                    mask.data[d] = mask.data[s];
                }
            }
        }
        y += 1;
    }
    // rows from (int)fy1 take the row (int)p5
    let mut y = fy1 as i32 as i64;
    while y < h {
        for x in 0..w {
            if outside(x, y) {
                if let (Some(d), Some(s)) = (idx(x, y), idx(x, iv14)) {
                    mask.data[d] = mask.data[s];
                }
            }
        }
        y += 1;
    }
    // columns: the left ones take the column (int)p2, the right ones from (int)fx1 on take (int)p4
    for y in 0..h {
        let mut x = 0i64;
        while (x as f32) < fx0 {
            if let (Some(d), Some(s)) = (idx(x, y), idx(p2 as i32 as i64, y)) {
                mask.data[d] = mask.data[s];
            }
            x += 1;
        }
        // the counter runs from (int)fx1 to the width, the destination starts at column (int)p4
        let first = fx1 as i32 as i64;
        let mut j = 0i64;
        while first + j < w {
            if let (Some(d), Some(s)) = (idx(p4 as i32 as i64 + j, y), idx(p4 as i32 as i64, y)) {
                mask.data[d] = mask.data[s];
            }
            j += 1;
        }
    }
}

/// The clean-up of the difference mask, in order:
/// 1. `mask_prune_passes(mask, 2)`: `mask_prune` with the limits 254 and 509;
/// 2. `mask_dilate_rows`;
/// 3. twice `mask_fill(254, 255)`, each followed by `mask_replicate_border`;
/// 4. `mask_fill(254, 128)` + border, then with the levels halved
///    (`(int)(x * 0.5 + 0.5)`: 127 and 64) `mask_fill(127, 64)` + border.
///
/// A mask that is zero everywhere stays zero (every step needs set pixels), which is the only
/// case the frame retime without vectors produces.
pub fn clean_mask(mask: &mut Mask) {
    mask_prune_passes(mask, 2);
    mask_dilate_rows(mask);
    for _ in 0..2 {
        mask_fill(mask, 254, 255);
        mask_replicate_border(mask);
    }
    let (mut limit, mut value) = (254i32, 128i32);
    for _ in 0..2 {
        mask_fill(mask, limit, value as u8);
        mask_replicate_border(mask);
        limit = (f64::from(limit) * 0.5 + 0.5) as i32;
        value = (f64::from(value) * 0.5 + 0.5) as i32;
    }
}

/// The difference mask of `warped` and `original` followed by `clean_mask`. 255 marks the pixels
/// the warp changed, 0 the pixels that agree with `original`.
pub fn change_mask(warped: &Image, original: &Image) -> Mask {
    let mut mask = difference_mask(warped, original, MASK_THRESHOLD);
    clean_mask(&mut mask);
    mask
}

// ------------------------------------------------------------------------------------------------
// masked composite
// ------------------------------------------------------------------------------------------------

/// Masked composite of `src` into `dest`: for every pixel with mask value `m`
/// * `m == 0`: `dest = src` (all four channels);
/// * otherwise `f = (float)m / 255.0f`, `g = (float)(1.0 - (double)f)`, and per channel
///   u8, u16: `dest = trunc(((src * g) + (dest * f)) + 0.5f)` (low byte or word of the 64 bit
///   integer); f32: `dest = ((g * src) + (f * dest)) + 0.0f`.
///
/// So the mask is the weight of `dest`: 255 keeps `dest` (exactly, for finite `src`), 0 takes
/// `src`.
pub fn composite_masked(dest: &mut Image, src: &Image, mask: &Mask) {
    assert!(dest.same_shape(src), "composite_masked needs equal size and format");
    assert!(
        mask.width == dest.width && mask.height == dest.height,
        "composite_masked needs a mask of the image size"
    );
    let rl = dest.row_len();
    if rl == 0 {
        return;
    }
    let w = dest.width;
    match (&mut dest.samples, &src.samples) {
        (Samples::U8(d), Samples::U8(s)) => par_rows(d, rl, |y, row| {
            let sr = &s[y * rl..(y + 1) * rl];
            for x in 0..w {
                let m = mask.data[y * w + x];
                for c in 0..CHANNELS {
                    let i = x * CHANNELS + c;
                    if m == 0 {
                        row[i] = sr[i];
                    } else {
                        let (f, g) = mask_weights(m);
                        row[i] = cvtt((sr[i] as f32 * g + row[i] as f32 * f) + 0.5f32) as u8;
                    }
                }
            }
        }),
        (Samples::U16(d), Samples::U16(s)) => par_rows(d, rl, |y, row| {
            let sr = &s[y * rl..(y + 1) * rl];
            for x in 0..w {
                let m = mask.data[y * w + x];
                for c in 0..CHANNELS {
                    let i = x * CHANNELS + c;
                    if m == 0 {
                        row[i] = sr[i];
                    } else {
                        let (f, g) = mask_weights(m);
                        row[i] = cvtt((sr[i] as f32 * g + row[i] as f32 * f) + 0.5f32) as u16;
                    }
                }
            }
        }),
        (Samples::F32(d), Samples::F32(s)) => par_rows(d, rl, |y, row| {
            let sr = &s[y * rl..(y + 1) * rl];
            for x in 0..w {
                let m = mask.data[y * w + x];
                for c in 0..CHANNELS {
                    let i = x * CHANNELS + c;
                    if m == 0 {
                        row[i] = sr[i];
                    } else {
                        let (f, g) = mask_weights(m);
                        let plain = (g * sr[i] + f * row[i]) + 0.0f32;
                        row[i] = if plain.is_nan() {
                            nr::add(nr::add(nr::mul(g, sr[i]), nr::mul(f, row[i])), 0.0f32)
                        } else {
                            plain
                        };
                    }
                }
            }
        }),
        _ => unreachable!("shapes were checked"),
    }
}

/// `(f, g)` of the composite for a non zero mask value: `f = m / 255.0f`, `g = 1.0 - f` computed
/// in double and narrowed to single.
#[inline(always)]
fn mask_weights(m: u8) -> (f32, f32) {
    let f = m as f32 / 255.0f32;
    let g = (1.0f64 - f64::from(f)) as f32;
    (f, g)
}

// ------------------------------------------------------------------------------------------------
// one half of Weighted Blend
// ------------------------------------------------------------------------------------------------

/// Strength of the unsharp step for the interpolation position `t`:
/// `c = -(1.0 - |0.5 - t| * 2.0) * 0.5` where everything is double precision except that
/// `1.0 - |0.5 - t| * 2.0` is narrowed to float and negated there. `c` is `-0.0` on a frame (`t` =
/// 0 or 1), which makes `lerp` copy its second image, and `-0.5` in the middle of an interval. For
/// float `t` this equals `-min(t, 1 - t)` exactly.
pub fn sharpen_weight(t: f32) -> f32 {
    let x = 1.0f64 - (0.5f64 - f64::from(t)).abs() * 2.0;
    let neg = -(x as f32);
    (f64::from(neg) * 0.5) as f32
}

/// One half of Weighted Blend: `frame` is the (possibly warped) frame, `original` the unwarped
/// input, `t` the position that selects the strength (`1 - wa` for the first frame, `wa` for the
/// second). In order:
/// 1. `mask = change_mask(frame, original)` (255 where the warp changed the pixel);
/// 2. `sharp = copy(frame)`, then `blur(sharp, 3.0)`;
/// 3. `base = copy(frame)`; `lerp(sharp, base, sharpen_weight(t))`, i.e.
///    `sharp = (blurred - base) * c + base` with `c <= 0`, an unsharp mask of strength `-c`;
/// 4. `invert_mask(mask)`;
/// 5. `composite_masked(frame, sharp, mask)`: `frame` keeps its value where the inverted mask
///    is 255 (no change by the warp) and takes the sharpened value where it is 0.
pub fn sharpen_step(frame: &mut Image, original: &Image, t: f32) {
    sharpen_step_with(frame, original, t, VerticalTaps::BelowFirst)
}

/// [`sharpen_step`] whose blur takes the edge rows of its column pass in the order `vtaps`
/// ([`blur_with`]).
pub fn sharpen_step_with(frame: &mut Image, original: &Image, t: f32, vtaps: VerticalTaps) {
    let mut mask = change_mask(frame, original);
    let mut sharp = frame.clone();
    blur_with(&mut sharp, BLUR_RADIUS, vtaps);
    let base = frame.clone();
    lerp(&mut sharp, &base, sharpen_weight(t));
    invert_mask(&mut mask);
    composite_masked(frame, &sharp, &mask);
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------- helpers

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    fn u8_image(width: usize, height: usize, data: Vec<u8>) -> Image {
        assert_eq!(data.len(), width * height * 4);
        Image { width, height, samples: Samples::U8(data) }
    }
    fn u16_image(width: usize, height: usize, data: Vec<u16>) -> Image {
        assert_eq!(data.len(), width * height * 4);
        Image { width, height, samples: Samples::U16(data) }
    }
    fn f32_image(width: usize, height: usize, data: Vec<f32>) -> Image {
        assert_eq!(data.len(), width * height * 4);
        Image { width, height, samples: Samples::F32(data) }
    }
    fn u8s(img: &Image) -> &[u8] {
        match &img.samples {
            Samples::U8(v) => v,
            _ => panic!("not u8"),
        }
    }
    fn u16s(img: &Image) -> &[u16] {
        match &img.samples {
            Samples::U16(v) => v,
            _ => panic!("not u16"),
        }
    }
    fn f32s(img: &Image) -> &[f32] {
        match &img.samples {
            Samples::F32(v) => v,
            _ => panic!("not f32"),
        }
    }
    fn bits(img: &Image) -> Vec<u32> {
        f32s(img).iter().map(|v| v.to_bits()).collect()
    }
    fn random_image(fmt: Fmt, w: usize, h: usize, rng: &mut Rng) -> Image {
        let n = w * h * 4;
        match fmt {
            Fmt::U8 => u8_image(w, h, (0..n).map(|_| (rng.next() >> 56) as u8).collect()),
            Fmt::U16 => u16_image(w, h, (0..n).map(|_| (rng.next() >> 48) as u16).collect()),
            Fmt::F32 => f32_image(w, h, (0..n).map(|_| rng.unit() * 2.0 - 0.5).collect()),
        }
    }

    // ------------------------------------------------------------------- taps

    #[test]
    fn above_first_changes_only_the_edge_rows_of_the_column_pass() {
        let mut rng = Rng(0x0005_beef);
        for fmt in [Fmt::F32, Fmt::U16, Fmt::U8] {
            let (w, h) = (13usize, 11usize);
            let img = random_image(fmt, w, h, &mut rng);
            let (mut a, mut b) = (img.clone(), img.clone());
            blur(&mut a, 3.0);
            blur_with(&mut b, 3.0, VerticalTaps::AboveFirst);
            let count = blur_taps(3.0).len();
            // the interior rows (every tap inside the column) agree bit for bit
            let rl = w * CHANNELS;
            let same_row = |y: usize| match (&a.samples, &b.samples) {
                (Samples::F32(p), Samples::F32(q)) => p[y * rl..(y + 1) * rl].iter().zip(&q[y * rl..(y + 1) * rl]).all(|(u, v)| u.to_bits() == v.to_bits()),
                (Samples::U16(p), Samples::U16(q)) => p[y * rl..(y + 1) * rl] == q[y * rl..(y + 1) * rl],
                (Samples::U8(p), Samples::U8(q)) => p[y * rl..(y + 1) * rl] == q[y * rl..(y + 1) * rl],
                _ => false,
            };
            for y in count..h - count {
                assert!(same_row(y), "{fmt:?} interior row {y}");
            }
            if fmt == Fmt::F32 {
                // an edge row is the edge formula with the two taps of each distance swapped
                let mut rows = img.clone();
                let taps = blur_taps(3.0);
                if let Samples::F32(d) = &mut rows.samples {
                    let src = d.clone();
                    for y in 0..h {
                        let line: Vec<[f32; 4]> = src[y * rl..(y + 1) * rl].chunks_exact(4).map(|p| [p[0], p[1], p[2], p[3]]).collect();
                        for x in 0..w {
                            let v = conv_pixel(x, w, &taps, [-1.0; 4], |j| line[j], false, false);
                            d[y * rl + x * 4..y * rl + x * 4 + 4].copy_from_slice(&v);
                        }
                    }
                }
                let snap = f32s(&rows).to_vec();
                for y in (0..count).chain(h - count..h) {
                    for x in 0..w {
                        let col = |j: usize| [snap[j * rl + x * 4], snap[j * rl + x * 4 + 1], snap[j * rl + x * 4 + 2], snap[j * rl + x * 4 + 3]];
                        let c = col(y);
                        let w0 = taps[0];
                        for lane in 0..4 {
                            let mut acc = c[lane] * w0;
                            let mut wsum = w0;
                            for k in (1..count).rev() {
                                let up = if y >= k { col(y - k)[lane] } else { -1.0 };
                                let down = if y + k < h { col(y + k)[lane] } else { -1.0 };
                                let wu = (if up != -1.0 { 1.0f32 } else { 0.0 }) * taps[k];
                                wsum = wsum + wu;
                                let ut = wu * up;
                                let wd = (if down != -1.0 { 1.0f32 } else { 0.0 }) * taps[k];
                                let dt = down * wd;
                                wsum = wsum + wd;
                                acc = (acc + ut) + dt;
                            }
                            assert_eq!(f32s(&b)[y * rl + x * 4 + lane].to_bits(), (acc / wsum).to_bits(), "edge row {y} col {x} lane {lane}");
                        }
                    }
                }
                assert_ne!(bits(&a), bits(&b), "the two orders differ somewhere in the edge rows");
            }
        }
        // sharpen_step is sharpen_step_with the engine's order
        let mut rng = Rng(77);
        let orig = random_image(Fmt::F32, 9, 8, &mut rng);
        let warped = random_image(Fmt::F32, 9, 8, &mut rng);
        let (mut p, mut q) = (warped.clone(), warped.clone());
        sharpen_step(&mut p, &orig, 0.3);
        sharpen_step_with(&mut q, &orig, 0.3, VerticalTaps::BelowFirst);
        assert_eq!(bits(&p), bits(&q));
    }

    #[test]
    fn blur_taps_for_radius_three_are_exact() {
        // numpy float32 emulation of the taps: powf(2, -i*i) = 1, 1/2, 1/16, 1/512,
        // sum 1.564453125, divisor sum * 2 - w0 = 2.12890625
        let taps = blur_taps(3.0);
        assert_eq!(taps, vec![0.46972477f32, 0.23486239, 0.029357798, 0.0009174312]);
        let d = 2.12890625f32;
        assert_eq!(taps[0], 1.0f32 / d);
        assert_eq!(taps[1], 0.5f32 / d);
        assert_eq!(taps[2], 0.0625f32 / d);
        assert_eq!(taps[3], 0.001953125f32 / d);
    }

    #[test]
    fn blur_taps_of_other_radii_sum_to_one() {
        for r in [1.0f32, 2.0, 4.5, 6.0, 9.0] {
            let t = blur_taps(r);
            assert_eq!(t.len(), r as usize + 1);
            let total = t[0] + 2.0 * t[1..].iter().sum::<f32>();
            assert!((total - 1.0).abs() < 1e-5, "r={r} sum={total}");
            assert!(t.windows(2).all(|p| p[0] > p[1]), "r={r} is decreasing");
        }
    }

    // ------------------------------------------------------------------- blur

    #[test]
    fn blur_u8_impulse_golden() {
        // Hand computation with the taps above, lane R only. Row pass of a 255 impulse:
        // trunc(255 * w[d] + 0.5) = 120, 60, 7, 0 for d = 0..3. Column pass of those:
        // trunc(h * w[e] + 0.5).
        let (w, h) = (21usize, 21usize);
        let mut d = vec![0u8; w * h * 4];
        d[(10 * w + 10) * 4 + 1] = 255;
        let mut img = u8_image(w, h, d);
        blur(&mut img, 3.0);
        let row = [120u32, 60, 7, 0];
        let tapsf = [0.46972477f32, 0.23486239, 0.029357798, 0.0009174312];
        let out = u8s(&img);
        for dy in -4i32..=4 {
            for dx in -4i32..=4 {
                let x = (10 + dx) as usize;
                let y = (10 + dy) as usize;
                let hv = (if dx.abs() <= 3 { row[dx.unsigned_abs() as usize] } else { 0 }) as f32;
                let want = if dy.abs() <= 3 {
                    (hv * tapsf[dy.unsigned_abs() as usize] + 0.5).trunc() as u8
                } else {
                    0
                };
                assert_eq!(out[(y * w + x) * 4 + 1], want, "dx={dx} dy={dy}");
            }
        }
        // the quadrant spelt out
        let at = |dx: usize, dy: usize| out[((10 + dy) * w + 10 + dx) * 4 + 1];
        assert_eq!([at(0, 0), at(0, 1), at(0, 2), at(0, 3)], [56, 28, 4, 0]);
        assert_eq!([at(1, 0), at(1, 1), at(1, 2), at(1, 3)], [28, 14, 2, 0]);
        assert_eq!([at(2, 0), at(2, 1), at(2, 2), at(2, 3)], [3, 2, 0, 0]);
        // the other lanes stay zero everywhere
        for p in out.chunks(4) {
            assert_eq!((p[0], p[2], p[3]), (0, 0, 0));
        }
    }

    #[test]
    fn blur_edge_formula_golden() {
        // 3 x 1 image, R = [100, 0, 0]. The row pass uses the edge formula everywhere (the line is
        // shorter than the kernel), the taps outside the line drop out and the rest is renormalised:
        // x = 0: 100 * w0 / (w0 + w2 + w1) = 100 / 1.5625 = 64, x = 1: 100 * w1 / (w0 + 2 w1) = 25,
        // x = 2: 100 * w2 / (w0 + w1 + w2) = 4. The column pass of a one row image changes nothing.
        let mut d = vec![0u8; 12];
        d[1] = 100;
        let mut img = u8_image(3, 1, d);
        blur(&mut img, 3.0);
        let r: Vec<u8> = u8s(&img).chunks(4).map(|p| p[1]).collect();
        assert_eq!(r, vec![64, 25, 4]);
        // 16 bit: 40000 / 1.5625 = 25600, 40000 / 4 = 10000, 40000 * 0.04 = 1600
        let mut d = vec![0u16; 12];
        d[1] = 40000;
        let mut img = u16_image(3, 1, d);
        blur(&mut img, 3.0);
        let r: Vec<u16> = u16s(&img).chunks(4).map(|p| p[1]).collect();
        assert_eq!(r, vec![25600, 10000, 1600]);
    }

    #[test]
    fn blur_integer_constants_are_fixed_points() {
        let mut rng = Rng(11);
        for (w, h) in [(1usize, 1usize), (3, 5), (8, 8), (9, 9), (17, 6), (40, 33)] {
            let c8: Vec<u8> = (0..4).map(|_| (rng.next() >> 56) as u8).collect();
            let mut img = u8_image(w, h, (0..w * h).flat_map(|_| c8.clone()).collect());
            let before = img.clone();
            blur(&mut img, 3.0);
            assert_eq!(img, before, "u8 {w}x{h}");
            let c16: Vec<u16> = (0..4).map(|_| (rng.next() >> 48) as u16).collect();
            let mut img = u16_image(w, h, (0..w * h).flat_map(|_| c16.clone()).collect());
            let before = img.clone();
            blur(&mut img, 3.0);
            assert_eq!(img, before, "u16 {w}x{h}");
        }
        let mut img = f32_image(20, 20, vec![0.3; 20 * 20 * 4]);
        blur(&mut img, 3.0);
        assert!(f32s(&img).iter().all(|v| (v - 0.3).abs() < 1e-6));
    }

    /// Independent restatement of one line of the filter with positional edge tests instead of the
    /// sentinel compares, returning the pre rounding float of each pixel.
    fn oracle_line(vals: &[f32], taps: &[f32]) -> Vec<f32> {
        let n = vals.len();
        let count = taps.len();
        (0..n)
            .map(|i| {
                let interior = n > 2 * count && i >= count && i < n - count;
                let mut acc = vals[i] * taps[0];
                if interior {
                    for k in (1..count).rev() {
                        acc = acc + (vals[i + k] + vals[i - k]) * taps[k];
                    }
                    acc
                } else {
                    let mut wsum = taps[0];
                    for k in (1..count).rev() {
                        if i + k < n {
                            wsum = wsum + taps[k];
                            acc = acc + taps[k] * vals[i + k];
                        }
                        if i >= k {
                            acc = acc + vals[i - k] * taps[k];
                            wsum = wsum + taps[k];
                        }
                    }
                    acc / wsum
                }
            })
            .collect()
    }

    fn oracle_blur(plane: &[f32], w: usize, h: usize, taps: &[f32], store: impl Fn(f32) -> f32) -> Vec<f32> {
        let mut cur = plane.to_vec();
        for y in 0..h {
            let line = oracle_line(&cur[y * w..(y + 1) * w], taps);
            for x in 0..w {
                cur[y * w + x] = store(line[x]);
            }
        }
        for x in 0..w {
            let col: Vec<f32> = (0..h).map(|y| cur[y * w + x]).collect();
            let line = oracle_line(&col, taps);
            for y in 0..h {
                cur[y * w + x] = store(line[y]);
            }
        }
        cur
    }

    #[test]
    fn blur_matches_the_positional_oracle_in_every_format() {
        let taps = blur_taps(3.0);
        let mut rng = Rng(2718);
        let sizes = [1usize, 2, 3, 4, 5, 7, 8, 9, 12, 17];
        let store8 = |v: f32| {
            let t = (v + 0.5f32).trunc();
            let t = t.max(0.0).min(255.0);
            (t + 0.5f32).trunc()
        };
        let store16 = |v: f32| {
            let t = (v + 0.5f32).trunc();
            let t = t.max(0.0).min(65535.0);
            (t + 0.5f32).trunc()
        };
        for &w in &sizes {
            for &h in &sizes {
                for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
                    let src = random_image(fmt, w, h, &mut rng);
                    let mut img = src.clone();
                    blur(&mut img, 3.0);
                    for lane in 0..4 {
                        let plane: Vec<f32> = match &src.samples {
                            Samples::U8(d) => d.iter().skip(lane).step_by(4).map(|v| *v as f32).collect(),
                            Samples::U16(d) => d.iter().skip(lane).step_by(4).map(|v| *v as f32).collect(),
                            Samples::F32(d) => d.iter().skip(lane).step_by(4).copied().collect(),
                        };
                        let want = match fmt {
                            Fmt::U8 => oracle_blur(&plane, w, h, &taps, store8),
                            Fmt::U16 => oracle_blur(&plane, w, h, &taps, store16),
                            Fmt::F32 => oracle_blur(&plane, w, h, &taps, |v| v),
                        };
                        let got: Vec<f32> = match &img.samples {
                            Samples::U8(d) => d.iter().skip(lane).step_by(4).map(|v| *v as f32).collect(),
                            Samples::U16(d) => d.iter().skip(lane).step_by(4).map(|v| *v as f32).collect(),
                            Samples::F32(d) => d.iter().skip(lane).step_by(4).copied().collect(),
                        };
                        let gb: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
                        let wb: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
                        assert_eq!(gb, wb, "{fmt:?} {w}x{h} lane {lane}");
                    }
                }
            }
        }
    }

    #[test]
    fn blur_ignores_a_negative_radius_and_empty_images() {
        let mut rng = Rng(5);
        let src = random_image(Fmt::U8, 6, 6, &mut rng);
        let mut img = src.clone();
        blur(&mut img, -1.0);
        assert_eq!(img, src);
        for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
            let mut e = Image::new(0, 0, fmt);
            blur(&mut e, 3.0);
        }
    }

    /// Two NaNs of the other sign meet in the taps of one pixel: the reference keeps the positive one (`nanrule`),
    /// whichever side it comes from. Finite lanes beside them stay finite.
    #[cfg(feature = "emulator-nan")]
    #[test]
    fn blur_f32_nan_pair_keeps_the_positive_nan() {
        let pos = f32::from_bits(0x7fc0_0000);
        let neg = f32::from_bits(0xffc0_0000);
        for (left, right) in [(pos, neg), (neg, pos)] {
            let mut d = vec![0.5f32; 12 * 4];
            d[3 * 4 + 1] = left;
            d[5 * 4 + 1] = right;
            let mut img = f32_image(12, 1, d);
            blur(&mut img, BLUR_RADIUS);
            let b = bits(&img);
            assert_eq!(b[4 * 4 + 1], 0x7fc0_0000, "R lane between the two NaNs");
            assert!(f32s(&img)[4 * 4 + 2].is_finite(), "G lane untouched by the NaNs");
        }
    }

    // ------------------------------------------------------------------ masks

    #[test]
    fn mask_limits_for_the_three_formats() {
        // numpy float32: 48.96, 12582.721 (12582.720703125), 0.192
        assert_eq!(mask_limit(Fmt::U8, 0.4), 48.96f32);
        assert_eq!(mask_limit(Fmt::U16, 0.4), 12582.721f32);
        assert_eq!(mask_limit(Fmt::F32, 0.4), 0.192f32);
    }

    #[test]
    fn difference_mask_threshold_is_the_sum_of_the_colour_channels() {
        // 8 bit: limit 48.96, so a sum of 48 gives 0 and 49 gives 255; alpha is ignored; order of
        // the channels is irrelevant for the integer sums.
        let a = u8_image(4, 1, vec![
            9, 10, 20, 18, // sum 48
            9, 10, 20, 19, // sum 49
            200, 49, 0, 0, // sum 49 in R alone
            0, 0, 0, 0, // sum 0 although alpha differs below
        ]);
        let b = u8_image(4, 1, vec![
            0, 0, 0, 0, //
            0, 0, 0, 0, //
            0, 0, 0, 0, //
            255, 0, 0, 0,
        ]);
        let m = difference_mask(&a, &b, 0.4);
        assert_eq!(m.data, vec![0, 255, 255, 0]);
        // the sign of the difference does not matter
        let m2 = difference_mask(&b, &a, 0.4);
        assert_eq!(m2.data, m.data);
        // 16 bit: limit 12582.72
        let a = u16_image(2, 1, vec![0, 6000, 6000, 582, 0, 6000, 6000, 583]);
        let b = u16_image(2, 1, vec![0; 8]);
        assert_eq!(difference_mask(&a, &b, 0.4).data, vec![0, 255]);
        // float: limit 0.192
        let a = f32_image(2, 1, vec![0.0, 0.1, 0.09, 0.0, 0.0, 0.1, 0.0925, 0.0]);
        let b = f32_image(2, 1, vec![0.0; 8]);
        assert_eq!(difference_mask(&a, &b, 0.4).data, vec![0, 255]);
        // NaN never reaches the limit
        let a = f32_image(1, 1, vec![0.0, f32::NAN, 1.0, 1.0]);
        assert_eq!(difference_mask(&a, &b_one(), 0.4).data, vec![0]);
    }

    fn b_one() -> Image {
        f32_image(1, 1, vec![0.0; 4])
    }

    #[test]
    fn difference_mask_of_identical_images_is_empty() {
        let mut rng = Rng(8);
        for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
            let a = random_image(fmt, 13, 7, &mut rng);
            let m = difference_mask(&a, &a.clone(), 0.4);
            assert_eq!((m.width, m.height), (13, 7));
            assert!(m.is_uniform(0));
        }
    }

    #[test]
    fn invert_mask_is_255_minus() {
        let mut m = Mask { width: 3, height: 2, data: vec![0, 1, 64, 127, 128, 255] };
        invert_mask(&mut m);
        assert_eq!(m.data, vec![255, 254, 191, 128, 127, 0]);
    }

    fn mask_from(w: usize, h: usize, pixels: &[(usize, usize)]) -> Mask {
        let mut m = Mask::new(w, h);
        for &(x, y) in pixels {
            m.set(x, y, 255);
        }
        m
    }

    #[test]
    fn mask_prune_removes_pixels_with_too_few_neighbours() {
        // neighbours of (c, r): rows r-2, r, r+2, columns c-1..c+1 (centre of row r excluded)
        let mut m = mask_from(12, 12, &[(5, 5)]);
        mask_prune(&mut m, 254);
        assert!(m.is_uniform(0), "an isolated pixel has no neighbour");
        // a neighbour at (c+1, r), (c, r+2) and (c-1, r-2) counts, one at (c, r+1) does not
        for (nx, ny, kept) in [(6, 5, true), (4, 5, true), (5, 7, true), (5, 3, true), (4, 3, true), (6, 7, true), (5, 6, false), (5, 4, false), (7, 5, false)] {
            let mut m = mask_from(12, 12, &[(5, 5), (nx, ny)]);
            mask_prune(&mut m, 254);
            assert_eq!(m.get(5, 5) == 255, kept, "neighbour ({nx},{ny})");
        }
        // limit 509 needs two neighbours
        let mut m = mask_from(12, 12, &[(5, 5), (6, 5)]);
        mask_prune(&mut m, 509);
        assert!(m.is_uniform(0));
        let mut m = mask_from(12, 12, &[(4, 5), (5, 5), (6, 5)]);
        mask_prune(&mut m, 509);
        assert_eq!(m.data.iter().filter(|v| **v != 0).count(), 1);
        assert_eq!(m.get(5, 5), 255);
        // the outer frame is never touched and a mask smaller than 5 rows is left alone
        let mut m = mask_from(12, 12, &[(0, 0), (11, 11), (5, 1)]);
        let before = m.clone();
        mask_prune(&mut m, 254);
        assert_eq!(m, before);
        let mut small = mask_from(8, 4, &[(3, 2)]);
        let before = small.clone();
        mask_prune(&mut small, 254);
        assert_eq!(small, before);
    }

    #[test]
    fn mask_fill_grows_into_the_full_ring() {
        let mut m = mask_from(9, 9, &[(4, 4)]);
        mask_fill(&mut m, 254, 255);
        for y in 0..9 {
            for x in 0..9 {
                let inside = (3..=5).contains(&x) && (3..=5).contains(&y);
                assert_eq!(m.get(x, y) == 255, inside, "({x},{y})");
            }
        }
        // 128 alone is above the limit 127 and below the limit 254
        let mut m = Mask::new(9, 9);
        m.set(4, 4, 128);
        let mut above = m.clone();
        mask_fill(&mut above, 127, 64);
        assert_eq!(above.get(5, 4), 64);
        let mut below = m.clone();
        mask_fill(&mut below, 254, 64);
        assert_eq!(below, m);
        // the border is never written
        let mut m = mask_from(6, 6, &[(1, 1)]);
        mask_fill(&mut m, 254, 255);
        assert_eq!(m.get(0, 0), 0);
        assert_eq!(m.get(2, 2), 255);
    }

    #[test]
    fn mask_dilate_rows_copies_full_pixels_up_and_down() {
        let mut m = mask_from(5, 6, &[(2, 2), (0, 0), (4, 5)]);
        mask_dilate_rows(&mut m);
        let on: Vec<(usize, usize)> = (0..6).flat_map(|y| (0..5).map(move |x| (x, y))).filter(|&(x, y)| m.get(x, y) == 255).collect();
        // (2,2) -> (2,1) and (2,3); (0,0) -> (0,1); (4,5) -> (4,4); the sources stay
        assert_eq!(on, vec![(0, 0), (0, 1), (2, 1), (2, 2), (2, 3), (4, 4), (4, 5)]);
        // values other than 255 do not spread
        let mut m = Mask::new(3, 3);
        m.set(1, 1, 254);
        mask_dilate_rows(&mut m);
        assert_eq!(m.data.iter().filter(|v| **v != 0).count(), 1);
    }

    #[test]
    fn mask_replicate_border_copies_the_inner_ring_outwards() {
        let (w, h) = (6usize, 5usize);
        let mut m = Mask::new(w, h);
        for y in 0..h {
            for x in 0..w {
                m.set(x, y, (10 * y + x) as u8);
            }
        }
        mask_replicate_border(&mut m);
        for y in 0..h {
            for x in 0..w {
                let sy = y.clamp(1, h - 2);
                let sx = x.clamp(1, w - 2);
                assert_eq!(m.get(x, y), (10 * sy + sx) as u8, "({x},{y})");
            }
        }
    }

    #[test]
    fn clean_mask_keeps_zero_and_drops_specks() {
        for (w, h) in [(1usize, 1usize), (2, 2), (3, 3), (5, 5), (16, 9), (31, 30)] {
            let mut m = Mask::new(w, h);
            clean_mask(&mut m);
            assert!(m.is_uniform(0), "{w}x{h}");
        }
        for pixels in [vec![(15, 15)], vec![(15, 15), (16, 15)], vec![(15, 15), (15, 16)], vec![(15, 15), (17, 15)]] {
            let mut m = mask_from(30, 30, &pixels);
            clean_mask(&mut m);
            assert!(m.is_uniform(0), "{pixels:?}");
        }
    }

    #[test]
    fn clean_mask_grows_a_block_into_three_levels() {
        // Block 6 x 6 at columns 12..18, rows 12..18 survives both prune passes (every pixel has
        // two neighbours), the vertical growth adds the rows 11 and 18, two full fills add two pixels
        // on every side, then rings of 128 and 64 follow (Chebyshev distance 1 and 2 from the core).
        let mut m = Mask::new(30, 30);
        for y in 12..18 {
            for x in 12..18 {
                m.set(x, y, 255);
            }
        }
        clean_mask(&mut m);
        for y in 0..30usize {
            for x in 0..30usize {
                let dx = if x < 10 { 10 - x } else if x > 19 { x - 19 } else { 0 };
                let dy = if y < 9 { 9 - y } else if y > 20 { y - 20 } else { 0 };
                let want = match dx.max(dy) {
                    0 => 255,
                    1 => 128,
                    2 => 64,
                    _ => 0,
                };
                assert_eq!(m.get(x, y), want, "({x},{y})");
            }
        }
    }

    #[test]
    fn clean_mask_commutes_with_a_half_turn() {
        let mut rng = Rng(99);
        for (w, h) in [(25usize, 25usize), (18, 31), (9, 12)] {
            let mut m = Mask::new(w, h);
            for v in m.data.iter_mut() {
                *v = if rng.next() % 5 < 2 { 255 } else { 0 };
            }
            let mut rot = Mask::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    rot.set(w - 1 - x, h - 1 - y, m.get(x, y));
                }
            }
            let mut a = m.clone();
            clean_mask(&mut a);
            let mut b = rot;
            clean_mask(&mut b);
            let mut b_back = Mask::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    b_back.set(w - 1 - x, h - 1 - y, b.get(x, y));
                }
            }
            assert_eq!(a, b_back, "{w}x{h}");
        }
    }

    // ------------------------------------------------------------- composite

    #[test]
    fn composite_u8_golden_values() {
        // numpy float32 emulation of trunc(((src * g) + (dest * f)) + 0.5), f = m / 255,
        // g = float(1.0 - double(f)); mask 0 takes src, 255 keeps dest.
        let cases: [(u8, u8, u8, u8); 10] = [
            (100, 200, 128, 150),
            (100, 200, 51, 180),
            (100, 200, 1, 200),
            (100, 200, 254, 100),
            (0, 255, 64, 191),
            (255, 0, 191, 191),
            (10, 20, 127, 15),
            (77, 33, 200, 68),
            (100, 200, 0, 200),
            (100, 200, 255, 100),
        ];
        for (dest, src, m, want) in cases {
            let mut d = u8_image(1, 1, vec![dest; 4]);
            let s = u8_image(1, 1, vec![src; 4]);
            composite_masked(&mut d, &s, &Mask::filled(1, 1, m));
            assert_eq!(u8s(&d), &[want; 4], "dest={dest} src={src} m={m}");
        }
    }

    #[test]
    fn composite_u8_every_pair_for_edge_masks() {
        let n = 65536usize;
        let a: Vec<u8> = (0..n).map(|i| (i / 256) as u8).collect();
        let b: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
        let src = u8_image(64, 256, b.clone());
        for m in [0u8, 1, 2, 51, 127, 128, 200, 254, 255] {
            let mut d = u8_image(64, 256, a.clone());
            composite_masked(&mut d, &src, &Mask::filled(64, 256, m));
            for i in 0..n {
                let want = if m == 0 {
                    b[i]
                } else {
                    let f = (m as f32) / 255.0f32;
                    let g = (1.0f64 - f as f64) as f32;
                    let s: f32 = b[i] as f32 * g;
                    let t: f32 = a[i] as f32 * f;
                    ((s + t) + 0.5f32).trunc() as i64 as u8
                };
                assert_eq!(u8s(&d)[i], want, "m={m} dest={} src={}", a[i], b[i]);
            }
            if m == 255 {
                assert_eq!(u8s(&d), &a[..], "mask 255 keeps the destination");
            }
        }
    }

    #[test]
    fn composite_u16_and_f32_golden_values() {
        let mut d = u16_image(1, 1, vec![10000; 4]);
        let s = u16_image(1, 1, vec![60000; 4]);
        composite_masked(&mut d, &s, &Mask::filled(1, 1, 128));
        // f = 128 / 255, 60000 * (1 - f) + 10000 * f + 0.5
        let f = 128.0f32 / 255.0;
        let g = (1.0f64 - f as f64) as f32;
        let want = ((60000.0f32 * g + 10000.0f32 * f) + 0.5f32).trunc() as u16;
        assert_eq!(u16s(&d), &[want; 4]);
        assert!((want as i32 - 34902).abs() <= 1, "about 34901.96: {want}");

        // numpy float32 emulation: ((g * src) + (f * dest)) + 0.0
        let cases: [(f32, f32, u8, u32); 3] = [
            (0.25, 0.75, 128, 0x3eff7f7e),
            (0.5, 1.0, 64, 0x3f5fdfe0),
            (0.125, -0.5, 200, 0xbc20a098),
        ];
        for (dest, src, m, want) in cases {
            let mut d = f32_image(1, 1, vec![dest; 4]);
            let s = f32_image(1, 1, vec![src; 4]);
            composite_masked(&mut d, &s, &Mask::filled(1, 1, m));
            assert_eq!(bits(&d), vec![want; 4], "dest={dest} src={src} m={m}");
        }
    }

    #[test]
    fn composite_float_mask_255_keeps_dest_and_clears_negative_zero() {
        let dest = f32_image(1, 1, vec![-0.0, 0.5, -3.0, f32::INFINITY]);
        let src = f32_image(1, 1, vec![0.9, -7.0, 123.0, 0.25]);
        let mut d = dest.clone();
        composite_masked(&mut d, &src, &Mask::filled(1, 1, 255));
        // (0 * src + 1 * dest) + 0.0: equal to dest, with -0.0 turned into +0.0
        assert_eq!(bits(&d), vec![0u32, 0.5f32.to_bits(), (-3.0f32).to_bits(), f32::INFINITY.to_bits()]);
        // mask 0 copies src bit for bit
        let mut d = dest.clone();
        composite_masked(&mut d, &src, &Mask::new(1, 1));
        assert_eq!(bits(&d), bits(&src));
        // a non finite source spreads: 0 * inf is NaN
        let mut d = f32_image(1, 1, vec![0.5; 4]);
        let s = f32_image(1, 1, vec![f32::INFINITY; 4]);
        composite_masked(&mut d, &s, &Mask::filled(1, 1, 255));
        assert!(f32s(&d).iter().all(|v| v.is_nan()));
    }

    #[test]
    fn composite_uses_the_mask_per_pixel() {
        let mut d = u8_image(3, 1, vec![10; 12]);
        let s = u8_image(3, 1, vec![250; 12]);
        let m = Mask { width: 3, height: 1, data: vec![0, 128, 255] };
        composite_masked(&mut d, &s, &m);
        let px: Vec<u8> = u8s(&d).chunks(4).map(|p| p[0]).collect();
        assert_eq!(px[0], 250);
        assert_eq!(px[2], 10);
        assert!(px[1] > 10 && px[1] < 250);
        assert!(u8s(&d).chunks(4).all(|p| p.iter().all(|v| *v == p[0])));
    }

    // -------------------------------------------------------------- sharpening

    #[test]
    fn sharpen_weight_golden_values() {
        assert_eq!(sharpen_weight(0.0).to_bits(), 0x8000_0000, "-0.0 on frame A");
        assert_eq!(sharpen_weight(1.0).to_bits(), 0x8000_0000, "-0.0 on frame B");
        assert_eq!(sharpen_weight(0.5), -0.5);
        assert_eq!(sharpen_weight(0.25), -0.25);
        assert_eq!(sharpen_weight(0.75), -0.25);
        assert_eq!(sharpen_weight(0.3), -0.3f32);
    }

    #[test]
    fn sharpen_weight_is_minus_the_distance_to_the_nearest_frame() {
        let mut rng = Rng(4);
        for _ in 0..5000 {
            let t = rng.unit();
            let exact = -(f64::from(t).min(1.0 - f64::from(t)));
            assert_eq!(sharpen_weight(t), exact as f32, "t={t}");
        }
    }

    #[test]
    fn sharpen_step_on_a_frame_changes_nothing() {
        // t = 0: c = -0.0, lerp copies the unblurred copy, so the composite writes the
        // frame itself.
        let mut rng = Rng(21);
        for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
            let original = random_image(fmt, 14, 11, &mut rng);
            // a frame that differs from the unwarped one everywhere, so that the mask is full
            let frame0 = random_image(fmt, 14, 11, &mut rng);
            for t in [0.0f32, 1.0] {
                let mut frame = frame0.clone();
                sharpen_step(&mut frame, &original, t);
                match fmt {
                    Fmt::F32 => {
                        // partial mask values blend src and dest, which are equal here, as g*d + f*d
                        for (g, w) in f32s(&frame).iter().zip(f32s(&frame0)) {
                            assert!((g - w).abs() <= 2e-7 * w.abs().max(1.0), "{g} {w}");
                        }
                    }
                    _ => assert_eq!(frame, frame0, "{fmt:?} t={t}"),
                }
            }
        }
    }

    #[test]
    fn sharpen_step_of_an_unchanged_frame_keeps_the_frame() {
        let mut rng = Rng(22);
        for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
            let original = random_image(fmt, 14, 11, &mut rng);
            for t in [0.25f32, 0.5, 0.9] {
                let mut frame = original.clone();
                sharpen_step(&mut frame, &original, t);
                match fmt {
                    Fmt::F32 => {
                        let want: Vec<u32> = f32s(&original).iter().map(|v| (*v + 0.0).to_bits()).collect();
                        assert_eq!(bits(&frame), want);
                    }
                    _ => assert_eq!(frame, original, "{fmt:?} t={t}"),
                }
            }
        }
    }

    #[test]
    fn sharpen_step_replaces_changed_pixels_by_the_unsharp_mask() {
        // original: flat grey. frame: the same with a bright 8 x 8 square; the square and a margin are
        // "changed", far from it the frame is unchanged. In the core of the changed area the result is
        // the unsharp mask lerp(blur(frame), frame, c); far from it it is the frame.
        let (w, h) = (40usize, 40usize);
        let original = u8_image(w, h, vec![100; w * h * 4]);
        let mut d = vec![100u8; w * h * 4];
        for y in 16..24 {
            for x in 16..24 {
                for c in 0..4 {
                    d[(y * w + x) * 4 + c] = if c == 0 { 255 } else { 200 };
                }
            }
        }
        let frame0 = u8_image(w, h, d);
        let t = 0.5f32;
        let mut frame = frame0.clone();
        sharpen_step(&mut frame, &original, t);
        // expected, spelt out from the pieces
        let mut sharp = frame0.clone();
        blur(&mut sharp, BLUR_RADIUS);
        let base = frame0.clone();
        lerp(&mut sharp, &base, sharpen_weight(t));
        let mut mask = change_mask(&frame0, &original);
        invert_mask(&mut mask);
        let mut want = frame0.clone();
        composite_masked(&mut want, &sharp, &mask);
        assert_eq!(frame, want);
        // far away: untouched
        assert_eq!(u8s(&frame)[(2 * w + 2) * 4 + 1], 100);
        // core of the square: replaced by the sharpened value, which differs from the frame there
        let core = (20 * w + 20) * 4 + 1;
        assert_eq!(mask.data[20 * w + 20], 0);
        assert_eq!(u8s(&frame)[core], u8s(&sharp)[core]);
        assert_ne!(u8s(&frame0)[(16 * w + 16) * 4 + 1], u8s(&frame)[(16 * w + 16) * 4 + 1], "the corner is sharpened");
    }

    // ------------------------------------------------------ thread independence

    fn with_threads<R: Send>(n: usize, f: impl FnOnce() -> R + Send) -> R {
        rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap().install(f)
    }

    #[test]
    fn filters_do_not_depend_on_the_thread_count() {
        let mut rng = Rng(1234);
        for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
            let a = random_image(fmt, 61, 47, &mut rng);
            let b = random_image(fmt, 61, 47, &mut rng);
            let run = |threads: usize| {
                with_threads(threads, || {
                    let mut blurred = a.clone();
                    blur(&mut blurred, 3.0);
                    let mask = difference_mask(&a, &b, 0.4);
                    let mut comp = a.clone();
                    composite_masked(&mut comp, &b, &mask);
                    let mut step = a.clone();
                    sharpen_step(&mut step, &b, 0.4);
                    (blurred, mask, comp, step)
                })
            };
            // images with NaN would not compare equal, the random data has none
            assert_eq!(run(1), run(4), "{fmt:?}");
        }
    }
}
