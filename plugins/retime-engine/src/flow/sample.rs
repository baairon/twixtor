//! Bilinear samplers (1 and 2 channels) and corner aligned resampling (1 channel) of float images.
//!
//! The `_wrapx` variants ([`bilinear1_wrapx`], [`resample1_wrapx`]) are the horizontal
//! wrap-around (360) mode of the hole fill (parity gap G6; its arithmetic is
//! U): the image is periodic in x, the column right of the last one is column 0, and the
//! horizontal resample keeps the period (`sx = sw / dw` instead of the corner aligned
//! `(sw - 1) / (dw - 1)`). The vertical axis keeps the rules of the plain samplers. Precision: the
//! lerp chain of [`bilinear1`], single precision, no fused multiply add; resample rows run through
//! rayon row chunks.

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
pub(crate) fn cell(x: f32, len: usize) -> (usize, f32) {
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

/// 1 channel bilinear sampler of an image that is periodic in x (G6, U): `ix = trunc(x)`,
/// `fx = x - (f32) ix`, the right neighbour is column `ix + 1`, or column 0 when
/// `ix + 1 > w - 1` (no cell move and no fraction 1.0 in x, unlike [`bilinear1`]); y takes the
/// rule of [`bilinear1`] (the cell moves up at the last row with the fraction 1.0). Then
/// `top = (p10 - p00) * fx + p00`, `bot = (p11 - p01) * fx + p01`, `(bot - top) * fy + top` with
/// `p10`, `p11` the right neighbours, single precision, no fused multiply add. `x` lies in
/// `[0, w)` for every caller, `w >= 1`, `h >= 2`; a column outside (a negative or NaN `x`) is
/// clamped into the row to stay memory safe, in range inputs are untouched.
#[inline]
pub(crate) fn bilinear1_wrapx(img: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let mut ix = cvtt(x);
    let fx = x - (ix as f32);
    let w1 = w as i32 - 1;
    if ix < 0 {
        ix = 0;
    } else if ix > w1 {
        ix = w1;
    }
    let x1 = if ix + 1 > w1 { 0 } else { ix + 1 };
    let (iy, fy) = cell(y, h);
    let r0 = iy * w;
    let r1 = r0 + w;
    let (ix, x1) = (ix as usize, x1 as usize);
    let p00 = img[r0 + ix];
    let p10 = img[r0 + x1];
    let p01 = img[r1 + ix];
    let p11 = img[r1 + x1];
    let top = (p10 - p00) * fx + p00;
    let bot = (p11 - p01) * fx + p01;
    (bot - top) * fy + top
}

/// [`bilinear1_wrapx`] at a column given as a whole column and a fraction (the B vector step and
/// the block cost of the wrap solve, rule G6, U): `ix` in `0..w` with the fraction `fx` (the
/// target column of `flow::wrap_target`, split exactly, so the sample does not depend on how far
/// from column 0 the pixel lies), the right neighbour `ix + 1`, or column 0 when `ix + 1 > w - 1`;
/// y and the lerps are those of [`bilinear1_wrapx`]. With `x = ix as f32 + fx` exact it equals
/// [`bilinear1_wrapx`] at `x` bit for bit. `w >= 1`, `h >= 2`, `ix < w`.
#[inline]
pub(crate) fn bilinear1_wrapx_at(img: &[f32], w: usize, h: usize, ix: usize, fx: f32, y: f32) -> f32 {
    let x1 = if ix + 1 > w - 1 { 0 } else { ix + 1 };
    let (iy, fy) = cell(y, h);
    let r0 = iy * w;
    let r1 = r0 + w;
    let p00 = img[r0 + ix];
    let p10 = img[r0 + x1];
    let p01 = img[r1 + ix];
    let p11 = img[r1 + x1];
    let top = (p10 - p00) * fx + p00;
    let bot = (p11 - p01) * fx + p01;
    (bot - top) * fy + top
}

/// 2 channel bilinear sampler of a field that is periodic in x (the flow upsample of the wrap
/// solve, rule G6, U): the column rule of [`bilinear1_wrapx`] (`ix = trunc(x)`, `fx = x - ix`, the
/// right neighbour of the last column is column 0, no cell move in x), the row rule of
/// [`bilinear2`], and its lerps on both lanes of the interleaved `img`. `x` lies in `[0, w)` for
/// every caller, `w >= 1`, `h >= 2`; a column outside is clamped into the row to stay memory safe.
#[inline]
pub(crate) fn bilinear2_wrapx(img: &[f32], w: usize, h: usize, x: f32, y: f32) -> [f32; 2] {
    let mut ix = cvtt(x);
    let fx = x - (ix as f32);
    let w1 = w as i32 - 1;
    if ix < 0 {
        ix = 0;
    } else if ix > w1 {
        ix = w1;
    }
    let x1 = if ix + 1 > w1 { 0 } else { ix + 1 };
    let (iy, fy) = cell(y, h);
    let (ix, x1) = (ix as usize, x1 as usize);
    let (r0, r1) = (iy * w, (iy + 1) * w);
    let mut out = [0.0f32; 2];
    for c in 0..2 {
        let p00 = img[(r0 + ix) * 2 + c];
        let p10 = img[(r0 + x1) * 2 + c];
        let p01 = img[(r1 + ix) * 2 + c];
        let p11 = img[(r1 + x1) * 2 + c];
        let top = (p10 - p00) * fx + p00;
        let bot = (p11 - p01) * fx + p01;
        out[c] = (bot - top) * fy + top;
    }
    out
}

/// Resampling of a 1 channel float image that is periodic in x (G6, U):
/// `dst(x, y) = bilinear1_wrapx(src, x * sx, y * sy)` with `sx = (f32) sw / (f32) dw` (the period
/// is kept: column `dw` of the destination would be column `sw` of the source, which is column 0)
/// and the corner aligned `sy = ((f32) sh - 1.0) / (f32)(dh - 1)` of [`resample1`]; the products
/// are single precision multiplies of the converted integers. Rows in parallel; the result does
/// not depend on the thread count.
pub(crate) fn resample1_wrapx(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let mut dst = vec![0.0f32; dw * dh];
    if dw == 0 || dh == 0 {
        return dst;
    }
    let sx = (sw as i32 as f32) / (dw as i32 as f32);
    let sy = (sh as f32 - 1.0) / ((dh as i32 - 1) as f32);
    dst.par_chunks_mut(dw).enumerate().for_each(|(y, row)| {
        let fy = (y as i32 as f32) * sy;
        for (x, v) in row.iter_mut().enumerate() {
            *v = bilinear1_wrapx(src, sw, sh, (x as i32 as f32) * sx, fy);
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

    #[test]
    fn periodic_bilinear_hand_values() {
        // 3x2 image; the column right of column 2 is column 0
        let img = [0.0f32, 10.0, 20.0, 100.0, 110.0, 120.0];
        assert_eq!(bilinear1_wrapx(&img, 3, 2, 0.0, 0.0), 0.0);
        assert_eq!(bilinear1_wrapx(&img, 3, 2, 0.5, 0.5), 55.0);
        // x = 2.5 lies between column 2 and column 0: top = (0 - 20) * 0.5 + 20 = 10, bot = 110
        assert_eq!(bilinear1_wrapx(&img, 3, 2, 2.5, 0.0), 10.0);
        assert_eq!(bilinear1_wrapx(&img, 3, 2, 2.5, 1.0), 110.0);
        assert_eq!(bilinear1_wrapx(&img, 3, 2, 2.5, 0.5), 60.0);
        // x = 2.0 exactly: the fraction is 0 (no cell move), so the value is column 2 exactly
        assert_eq!(bilinear1_wrapx(&img, 3, 2, 2.0, 0.0), 20.0);
        // y keeps the plain rule: the last row moves the cell up with the fraction 1.0
        assert_eq!(bilinear1_wrapx(&img, 3, 2, 1.0, 1.0), bilinear1(&img, 3, 2, 1.0, 1.0));
        // one column: the right neighbour is the column itself
        let col = [3.0f32, 5.0, 9.0];
        assert_eq!(bilinear1_wrapx(&col, 1, 3, 0.5, 1.5), 7.0);
    }

    #[test]
    fn periodic_bilinear_equals_bilinear1_left_of_the_last_column() {
        let (w, h) = (7usize, 5usize);
        let img: Vec<f32> = (0..w * h).map(|i| ((i * 37) % 23) as f32 * 0.37 - 2.0).collect();
        for xi in 0..60 {
            let x = xi as f32 * ((w - 1) as f32 / 60.0);
            for yi in 0..=20 {
                let y = yi as f32 * ((h - 1) as f32 / 20.0);
                assert_eq!(bilinear1_wrapx(&img, w, h, x, y).to_bits(), bilinear1(&img, w, h, x, y).to_bits(), "{x} {y}");
            }
        }
    }

    #[test]
    fn the_periodic_resample_keeps_the_period() {
        // 4 -> 2 columns: sx = 2, dst 0 and 1 read src 0 and 2; rows stay corner aligned
        let src: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0];
        assert_eq!(resample1_wrapx(&src, 4, 2, 2, 2), vec![0.0, 2.0, 10.0, 12.0]);
        // 2 -> 4 columns: sx = 0.5, dst 3 lies half way from src 1 back to src 0
        let s2: Vec<f32> = vec![0.0, 8.0, 4.0, 12.0];
        assert_eq!(resample1_wrapx(&s2, 2, 2, 4, 2), vec![0.0, 4.0, 8.0, 4.0, 4.0, 8.0, 12.0, 8.0]);
        // the corner aligned resample would end on the last column instead
        assert_eq!(resample1(&s2, 2, 2, 4, 2)[3], 8.0);
        // a constant image stays constant, an empty destination is empty
        let c = vec![2.5f32; 15];
        assert!(resample1_wrapx(&c, 5, 3, 2, 2).iter().all(|&v| v == 2.5));
        assert!(resample1_wrapx(&c, 5, 3, 0, 2).is_empty());
    }

    #[test]
    fn periodic_resample_model_fixtures_are_reproduced_bit_for_bit() {
        let mut n = 0;
        for c in crate::holefill::tests::load_wrap_cases() {
            let crate::holefill::tests::WrapCase::Resample { sw, sh, dw, dh, src, out } = c else { continue };
            let got = resample1_wrapx(&src, sw, sh, dw, dh);
            let gb: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            let wb: Vec<u32> = out.iter().map(|v| v.to_bits()).collect();
            assert_eq!(gb, wb, "{sw}x{sh} -> {dw}x{dh}");
            n += 1;
        }
        assert!(n >= 10, "{n} resample cases");
    }
}
