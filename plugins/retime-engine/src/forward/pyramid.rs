//! The mip pyramid of the textured forward warp: construction of the levels and the level blend
//! used to sample it.
//!
//! * level 0 is the source itself (both sizes at least 2);
//! * level `i >= 1`: a copy of level `i - 1` is blurred in place with the 5 tap kernel
//!   `[3/8, 1/4, 1/16]` (rows then columns, the same separable driver as the Gaussian blur of
//!   `filters.rs`), then resampled to the level size (corner aligned bilinear:
//!   `dst(x, y) = bilerp4(src, x * sx, y * sy)` with
//!   `sx = ((float)src_w - 1.0f) / (float)(dst_w - 1)`, stores `sat(trunc(v + 0.5))` for integers
//!   and raw floats);
//! * the number of levels is 1 plus the number of times both sizes can be halved while staying at
//!   least 2, level sizes `max(prev / 2, 2)`.

use crate::image::{Image, Samples, CHANNELS};
use crate::warp::{bilerp4, Pix};
use fx_core::par_rows;

/// Number of levels of the pyramid of a `w x h` source: 1 plus the number of times both sizes can
/// be halved while staying at least 2.
pub(crate) fn level_count(w: usize, h: usize) -> usize {
    let mut n = 1usize;
    let mut a = w.max(2) as i32;
    let mut b = h.max(2) as i32;
    loop {
        a /= 2;
        if a < 2 {
            break;
        }
        b /= 2;
        if b < 2 {
            break;
        }
        n += 1;
    }
    n
}

/// Levels 1.. of the pyramid; level 0 is the source and is passed separately to [`Pyramid::sample`].
pub(crate) struct Pyramid {
    pub levels: Vec<Image>,
    /// Total number of levels, level 0 included.
    pub count: usize,
}

impl Pyramid {
    pub fn build(src: &Image) -> Pyramid {
        let count = level_count(src.width, src.height);
        let mut levels: Vec<Image> = Vec::with_capacity(count.saturating_sub(1));
        let mut cw = src.width.max(2);
        let mut ch = src.height.max(2);
        for i in 1..count {
            let prev: &Image = if i == 1 { src } else { &levels[i - 2] };
            let mut tmp = prev.clone();
            blur5(&mut tmp);
            cw /= 2;
            ch /= 2;
            if cw < 2 {
                cw = 2;
            }
            if ch < 2 {
                ch = 2;
            }
            let mut lvl = Image::new(cw, ch, src.fmt());
            resample(&mut lvl, &tmp);
            levels.push(lvl);
        }
        Pyramid { levels, count }
    }

    /// The colour at `(u, v)` of level `d` (linear blend of two levels).
    pub fn sample(&self, src: &Image, u: f32, v: f32, d: f32) -> [f32; 4] {
        let level = |i: usize| -> &Image {
            if i == 0 {
                src
            } else {
                &self.levels[i - 1]
            }
        };
        if 0.0 < d {
            if d < ((self.count as i32) - 1) as f32 {
                let i = super::raster::cvtt32(d);
                let frac = d - (i as f32);
                let i = i as usize;
                let a = tap(level(i), u, v);
                let b = tap(level(i + 1), u, v);
                let mut out = [0.0f32; 4];
                for k in 0..4 {
                    out[k] = (b[k] - a[k]) * frac + a[k];
                }
                return out;
            }
            return tap(level(self.count - 1), u, v);
        }
        tap(src, u, v)
    }
}

/// One bilinear tap of a level at `(w1 * u, h1 * v)`.
#[inline(always)]
pub(crate) fn tap(img: &Image, u: f32, v: f32) -> [f32; 4] {
    let w1 = img.width as i32 - 1;
    let h1 = img.height as i32 - 1;
    let x = (w1 as f32) * u;
    let y = (h1 as f32) * v;
    match &img.samples {
        Samples::U8(d) => bilerp4::<u8>(d, img.width, w1, h1, x, y),
        Samples::U16(d) => bilerp4::<u16>(d, img.width, w1, h1, x, y),
        Samples::F32(d) => bilerp4::<f32>(d, img.width, w1, h1, x, y),
    }
}

// ------------------------------------------------------------------------------------------------
// blur5: the separable blur driver with the 5 tap binomial kernel (a copy of the private
// `filters::blur_plane`; here only the taps differ)
// ------------------------------------------------------------------------------------------------

/// The half kernel of the 5 tap binomial filter: 3 taps `[0.375, 0.25, 0.0625]`.
pub(crate) const BLUR5_TAPS: [f32; 3] = [0.375, 0.25, 0.0625];

/// Truncation toward zero, used by the integer stores of the blur.
#[inline(always)]
fn cvtt(x: f32) -> i32 {
    super::raster::cvtt32(x)
}

trait Sample: Copy + Send + Sync {
    fn load(self) -> f32;
    fn store(v: f32) -> Self;
}

#[inline(always)]
fn store_int(v: f32, max: f32) -> i32 {
    let t = cvtt(v + 0.5f32);
    let f = t as f32;
    let f = if f > 0.0 { f } else { 0.0 };
    let f = if f < max { f } else { max };
    cvtt(f + 0.5f32)
}

impl Sample for u8 {
    fn load(self) -> f32 {
        self as f32
    }
    fn store(v: f32) -> u8 {
        store_int(v, 255.0) as u8
    }
}

impl Sample for u16 {
    fn load(self) -> f32 {
        self as f32
    }
    fn store(v: f32) -> u16 {
        store_int(v, 65535.0) as u16
    }
}

impl Sample for f32 {
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

/// One output pixel of the 1D filter (see `conv_pixel` in `filters.rs`): edge pixels renormalise
/// over the taps inside the line (taps outside hold the sentinel -1.0 and are dropped when equal
/// to it), interior pixels use the plain formula.
#[inline]
fn conv_pixel(i: usize, n: usize, taps: &[f32], s: [f32; 4], get: impl Fn(usize) -> [f32; 4]) -> [f32; 4] {
    let count = taps.len();
    let left = n.min(count);
    let mid = n.saturating_sub(2 * count);
    let c = get(i);
    let w0 = taps[0];
    let mut acc = [c[0] * w0, c[1] * w0, c[2] * w0, c[3] * w0];
    if i >= left && i < left + mid {
        for k in (1..count).rev() {
            let r = get(i + k);
            let l = get(i - k);
            let wk = taps[k];
            for lane in 0..4 {
                acc[lane] = acc[lane] + (r[lane] + l[lane]) * wk;
            }
        }
        return acc;
    }
    let mut wsum = [w0; 4];
    for k in (1..count).rev() {
        let r = if i + k < n { get(i + k) } else { s };
        let l = if i >= k { get(i - k) } else { s };
        let wk = taps[k];
        for lane in 0..4 {
            let wr = (if r[lane] != s[lane] { 1.0f32 } else { 0.0f32 }) * wk;
            wsum[lane] = wsum[lane] + wr;
            let rt = wr * r[lane];
            let wl = (if l[lane] != s[lane] { 1.0f32 } else { 0.0f32 }) * wk;
            let lt = l[lane] * wl;
            wsum[lane] = wsum[lane] + wl;
            acc[lane] = (acc[lane] + rt) + lt;
        }
    }
    for lane in 0..4 {
        acc[lane] = acc[lane] / wsum[lane];
    }
    acc
}

fn blur_plane<T: Sample>(data: &mut [T], width: usize, height: usize, taps: &[f32]) {
    if width == 0 || height == 0 {
        return;
    }
    let sentinel = [-1.0f32; 4];
    let rl = width * CHANNELS;
    par_rows(data, rl, |_, row| {
        let line: Vec<[f32; 4]> = row.chunks_exact(CHANNELS).map(load4::<T>).collect();
        for x in 0..width {
            let v = conv_pixel(x, width, taps, sentinel, |j| line[j]);
            for lane in 0..4 {
                row[x * CHANNELS + lane] = T::store(v[lane]);
            }
        }
    });
    let snap: Vec<T> = data.to_vec();
    par_rows(data, rl, |y, row| {
        for x in 0..width {
            let v = conv_pixel(y, height, taps, sentinel, |j| load4::<T>(&snap[j * rl + x * CHANNELS..]));
            for lane in 0..4 {
                row[x * CHANNELS + lane] = T::store(v[lane]);
            }
        }
    });
}

/// Blurs a four channel image with the 5 tap binomial kernel.
pub(crate) fn blur5(img: &mut Image) {
    let (w, h) = (img.width, img.height);
    match &mut img.samples {
        Samples::U8(d) => blur_plane(d, w, h, &BLUR5_TAPS),
        Samples::U16(d) => blur_plane(d, w, h, &BLUR5_TAPS),
        Samples::F32(d) => blur_plane(d, w, h, &BLUR5_TAPS),
    }
}

// ------------------------------------------------------------------------------------------------
// resample, mode 3
// ------------------------------------------------------------------------------------------------

fn resample_plane<P: Pix>(dst: &mut [P], dw: usize, dh: usize, src: &[P], sw: usize, sh: usize) {
    let sx = ((sw as i32) as f32 - 1.0f32) / ((dw as i32 - 1) as f32);
    let sy = ((sh as i32) as f32 - 1.0f32) / ((dh as i32 - 1) as f32);
    let (w1, h1) = (sw as i32 - 1, sh as i32 - 1);
    par_rows(dst, dw * CHANNELS, |y, row| {
        let fy = (y as i32 as f32) * sy;
        for x in 0..dw {
            let v = bilerp4::<P>(src, sw, w1, h1, (x as i32 as f32) * sx, fy);
            for k in 0..CHANNELS {
                row[x * CHANNELS + k] = P::store(v[k]);
            }
        }
    });
}

/// Resamples `src` to the size of `dst` (corner aligned bilinear) for two images of one format;
/// the white points are equal, so there is no scaling.
pub(crate) fn resample(dst: &mut Image, src: &Image) {
    let (dw, dh, sw, sh) = (dst.width, dst.height, src.width, src.height);
    match (&mut dst.samples, &src.samples) {
        (Samples::U8(d), Samples::U8(s)) => resample_plane(d, dw, dh, s, sw, sh),
        (Samples::U16(d), Samples::U16(s)) => resample_plane(d, dw, dh, s, sw, sh),
        (Samples::F32(d), Samples::F32(s)) => resample_plane(d, dw, dh, s, sw, sh),
        _ => unreachable!("pyramid levels have the format of the source"),
    }
}
