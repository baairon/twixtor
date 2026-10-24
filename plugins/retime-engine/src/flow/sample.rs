//! Bilinear samplers (1 and 2 channels) and corner aligned resampling (1 channel) of float images.

/// Truncation toward zero to `i32`; NaN and values out of range give `i32::MIN`.
#[inline(always)]
pub(crate) fn cvtt(x: f32) -> i32 {
    if x.is_nan() || x >= 2147483648.0 || x < -2147483648.0 {
        i32::MIN
    } else {
        x as i32
    }
}

/// Index and fraction of one coordinate: `i = trunc(x)`, `f = x - (f32) i`; when `i >= max_index`
/// (the image width minus 1 for x, the height minus 1 for y) the cell is moved one to the left and
/// the fraction is 1.0. The returned index is clamped into `0..=len-2` to stay memory safe for
/// negative or NaN coordinates; in range inputs are untouched.
#[inline(always)]
fn cell(x: f32, len: usize) -> (usize, f32) {
    let mut i = cvtt(x);
    let mut f = x - (i as f32);
    let last = len as i32 - 1;
    if i >= last {
        i = last - 1;
        f = 1.0;
    }
    let i = if i < 0 { 0 } else { i as usize };
    (i.min(len.saturating_sub(2)), f)
}

/// 1 channel bilinear sampler: with `p00, p10, p01, p11` the four pixels of the cell,
/// `top = (p10 - p00) * fx + p00`, `bot = (p11 - p01) * fx + p01`, result
/// `(bot - top) * fy + top`, all single precision with no fused multiply add. `w, h >= 2`.
#[inline]
pub(crate) fn bilinear1(img: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let (ix, fx) = cell(x, w);
    let (iy, fy) = cell(y, h);
    let i = iy * w + ix;
    let p00 = img[i];
    let p10 = img[i + 1];
    let p01 = img[i + w];
    let p11 = img[i + w + 1];
    let top = (p10 - p00) * fx + p00;
    let bot = (p11 - p01) * fx + p01;
    (bot - top) * fy + top
}

/// 2 channel bilinear sampler: the arithmetic of `bilinear1` on both lanes, `img` interleaved
/// `(c0, c1)`.
#[inline]
pub(crate) fn bilinear2(img: &[f32], w: usize, h: usize, x: f32, y: f32) -> [f32; 2] {
    let (ix, fx) = cell(x, w);
    let (iy, fy) = cell(y, h);
    let i = (iy * w + ix) * 2;
    let r = w * 2;
    let mut out = [0.0f32; 2];
    for c in 0..2 {
        let p00 = img[i + c];
        let p10 = img[i + 2 + c];
        let p01 = img[i + r + c];
        let p11 = img[i + r + 2 + c];
        let top = (p10 - p00) * fx + p00;
        let bot = (p11 - p01) * fx + p01;
        out[c] = (bot - top) * fy + top;
    }
    out
}

/// Corner aligned bilinear resampling of a 1 channel float image:
/// `dst(x, y) = bilinear(src, x * sx, y * sy)` with `sx = ((f32) src_w - 1.0) / (f32)(dst_w - 1)`
/// and the same for `sy`; the product `x * sx` is a single precision multiply of the converted
/// integer.
pub(crate) fn resample1(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let sx = (sw as f32 - 1.0) / ((dw as i32 - 1) as f32);
    let sy = (sh as f32 - 1.0) / ((dh as i32 - 1) as f32);
    let mut dst = vec![0.0f32; dw * dh];
    if dw == 0 || dh == 0 {
        return dst;
    }
    dst.par_chunks_mut(dw).enumerate().for_each(|(y, row)| {
        let fy = (y as i32 as f32) * sy;
        for (x, v) in row.iter_mut().enumerate() {
            *v = bilinear1(src, sw, sh, (x as i32 as f32) * sx, fy);
        }
    });
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_matches_hand_values() {
        // 3x2 image
        let img = [0.0f32, 10.0, 20.0, 100.0, 110.0, 120.0];
        assert_eq!(bilinear1(&img, 3, 2, 0.0, 0.0), 0.0);
        assert_eq!(bilinear1(&img, 3, 2, 1.0, 0.0), 10.0);
        // x = 0.5, y = 0.5: top = 5, bot = 105, result 55
        assert_eq!(bilinear1(&img, 3, 2, 0.5, 0.5), 55.0);
        // last column and row: the cell moves left/up and the fraction is 1.0
        assert_eq!(bilinear1(&img, 3, 2, 2.0, 1.0), 120.0);
        // top = 17.5, bot = 117.5, result 17.5 + 0.25 * 100
        assert_eq!(bilinear1(&img, 3, 2, 1.75, 0.25), 42.5);
    }

    #[test]
    fn two_channel_equals_two_one_channel_samplers() {
        let a = [0.0f32, 10.0, 20.0, 100.0, 110.0, 120.0];
        let b = [5.0f32, -3.0, 7.5, 2.0, 1.0, 9.0];
        let mut ab = [0.0f32; 12];
        for i in 0..6 {
            ab[2 * i] = a[i];
            ab[2 * i + 1] = b[i];
        }
        for &(x, y) in &[(0.3f32, 0.7f32), (1.9, 0.1), (2.0, 1.0), (0.0, 0.99)] {
            let r = bilinear2(&ab, 3, 2, x, y);
            assert_eq!(r[0].to_bits(), bilinear1(&a, 3, 2, x, y).to_bits());
            assert_eq!(r[1].to_bits(), bilinear1(&b, 3, 2, x, y).to_bits());
        }
    }

    #[test]
    fn resample_is_corner_aligned() {
        let src = [0.0f32, 1.0, 2.0, 3.0, 4.0]; // 5x1 is not valid (h < 2): use 5x2
        let s2: Vec<f32> = src.iter().chain(src.iter()).copied().collect();
        let d = resample1(&s2, 5, 2, 3, 2);
        // dst x = 0, 1, 2 maps to src x = 0, 2, 4
        assert_eq!(&d[0..3], &[0.0, 2.0, 4.0]);
    }
}
