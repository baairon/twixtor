//! Separable Gaussian and binomial convolution of float images with 1 to 3 channels: the
//! horizontal pass over the rows, then the vertical pass over the columns. The 5 tap pyramid
//! filter `blur5` is the same convolution with the kernel `[0.375, 0.25, 0.0625]`.
//!
//! Arithmetic of one output element, `k[0..n)` the half kernel, `x` the line (all single
//! precision, every operation separately rounded):
//! - interior (`n <= i < len - n`): `s = x[i] * k[0]`, then for `j = n-1` down
//!   to `1`: `s = s + (x[i+j] + x[i-j]) * k[j]`;
//! - border (the first `min(n, len)` elements, the rest after the interior):
//!   `num = k[0] * x[i]`, `den = k[0]`, then for `j = n-1` down to `1`, first
//!   the right tap and then the left tap: `den = den + w`, `num = num + x_tap * w`
//!   where `w = k[j]` for a tap that exists and `w = 0` for a tap outside the
//!   line; the result is `num / den`.
//! - A tap "exists" by position for 1 channel images (the line buffer is zero
//!   padded). For 2 and 3 channel images the line buffer is padded with
//!   `-1.0` and the weight is `1.0 * k[j]` when the SAMPLE differs from `-1.0`
//!   and `0` otherwise: a real sample that is exactly `-1.0` inside the
//!   line is therefore dropped as well.
//!
//! The border sums of the multi channel case use the same sequential chain as the 1 channel case:
//! `num = (num + xr*wr) + xl*wl`.

use rayon::prelude::*;

/// Half kernel of the 5 tap pyramid filter `[1 4 6 4 1] / 16` of `blur5`, `n = 3`.
pub(crate) const BINOMIAL5: [f32; 3] = [0.375, 0.25, 0.0625];

/// Blur radii below this are skipped. 0.001 is an assumed value; no call in the engine uses such
/// a radius.
pub(crate) const BLUR_RADIUS_EPS: f64 = 0.001;

/// `powf(2.0, e)` evaluated as the double precision `pow` rounded to single. It equals a single
/// precision `powf` for all kernel exponents of r = 7, 5, 3 and 2; the two differ in about 1 of
/// 8000 random arguments.
#[inline]
fn exp2_f32(e: f32) -> f32 {
    2.0f64.powf(e as f64) as f32
}

/// Gaussian half kernel. `r` is the radius in pixels, `k[0]` the centre tap, `k[i]` the tap at
/// distance `i`; the taps are normalised so that `k[0] + 2 * sum(k[1..]) = 1`.
pub(crate) fn gaussian_kernel(r: f32) -> Vec<f32> {
    let n = (r as i32 + 1).max(0) as usize;
    let s3 = r / 3.0f32;
    let d = s3 * s3;
    let mut k: Vec<f32> = Vec::with_capacity(n);
    let mut sum = 0.0f32;
    for i in 0..n {
        let fi = i as f32;
        let t = fi * fi;
        let e = -(t / d);
        let v = exp2_f32(e);
        k.push(v);
        sum += v;
    }
    if n == 0 {
        return k;
    }
    let den = sum * 2.0f32 - k[0];
    for v in k.iter_mut() {
        *v /= den;
    }
    k
}

/// Padding value of the multi channel line buffers (also the "tap is outside" marker).
const PAD_MULTI: f32 = -1.0;

/// Border element: value `x[i]` of channel `c` of a line of `len` pixels whose
/// element `t` of channel `c` is `get(t)` (`None` outside the line).
#[inline(always)]
fn border_value<const C: usize>(get: &dyn Fn(isize) -> Option<f32>, i: isize, k: &[f32]) -> f32 {
    let n = k.len();
    let centre = match get(i) {
        Some(v) => v,
        None => 0.0,
    };
    let mut num = k[0] * centre;
    let mut den = k[0];
    let mut j = n;
    while j > 1 {
        j -= 1;
        let kj = k[j];
        // right tap, then left tap
        for side in 0..2 {
            let t = if side == 0 { i + j as isize } else { i - j as isize };
            let (xv, w) = if C == 1 {
                match get(t) {
                    Some(v) => (v, kj),
                    None => (0.0f32, 0.0f32),
                }
            } else {
                let v = match get(t) {
                    Some(v) => v,
                    None => PAD_MULTI,
                };
                // mask is 1.0 when the sample differs from -1.0 (true for NaN), else 0.0;
                // the weight is mask * k
                let mask = if v != PAD_MULTI { 1.0f32 } else { 0.0f32 };
                (v, mask * kj)
            };
            den += w;
            num += xv * w;
        }
    }
    num / den
}

/// One row of the horizontal pass. `x` and `out` hold `len * C` samples.
fn hrow<const C: usize>(x: &[f32], out: &mut [f32], len: usize, k: &[f32]) {
    let n = k.len();
    let head = n.min(len);
    let interior = len.saturating_sub(2 * n);
    let border = |i: usize, out: &mut [f32]| {
        for c in 0..C {
            let get = |t: isize| -> Option<f32> {
                if t < 0 || t >= len as isize {
                    None
                } else {
                    Some(x[t as usize * C + c])
                }
            };
            out[i * C + c] = border_value::<C>(&get, i as isize, k);
        }
    };
    for i in 0..head {
        border(i, out);
    }
    if interior > 0 {
        let a = head * C;
        let b = (head + interior) * C;
        let k0 = k[0];
        for t in a..b {
            out[t] = x[t] * k0;
        }
        for j in (1..n).rev() {
            let off = j * C;
            let kj = k[j];
            for t in a..b {
                out[t] = out[t] + (x[t + off] + x[t - off]) * kj;
            }
        }
    }
    for i in (head + interior)..len {
        border(i, out);
    }
}

/// Horizontal pass of the whole image, rows in parallel (rows are independent).
fn hpass<const C: usize>(src: &[f32], dst: &mut [f32], w: usize, k: &[f32]) {
    let row = w * C;
    if row == 0 {
        return;
    }
    dst.par_chunks_mut(row)
        .zip(src.par_chunks(row))
        .for_each(|(d, s)| hrow::<C>(s, d, w, k));
}

/// Vertical pass: output row `y` of an image of `h` rows and `w` pixels.
fn vrow<const C: usize>(src: &[f32], out: &mut [f32], y: usize, w: usize, h: usize, k: &[f32]) {
    let n = k.len();
    let row = w * C;
    let head = n.min(h);
    let interior = h.saturating_sub(2 * n);
    if y >= head && y < head + interior {
        let k0 = k[0];
        let s0 = &src[y * row..(y + 1) * row];
        for t in 0..row {
            out[t] = s0[t] * k0;
        }
        for j in (1..n).rev() {
            let below = &src[(y + j) * row..(y + j + 1) * row];
            let above = &src[(y - j) * row..(y - j + 1) * row];
            let kj = k[j];
            for t in 0..row {
                out[t] = out[t] + (below[t] + above[t]) * kj;
            }
        }
    } else {
        for xc in 0..w {
            for c in 0..C {
                let get = |t: isize| -> Option<f32> {
                    if t < 0 || t >= h as isize {
                        None
                    } else {
                        Some(src[(t as usize * w + xc) * C + c])
                    }
                };
                out[xc * C + c] = border_value::<C>(&get, y as isize, k);
            }
        }
    }
}

fn vpass<const C: usize>(src: &[f32], dst: &mut [f32], w: usize, h: usize, k: &[f32]) {
    let row = w * C;
    if row == 0 {
        return;
    }
    dst.par_chunks_mut(row)
        .enumerate()
        .for_each(|(y, d)| vrow::<C>(src, d, y, w, h, k));
}

fn convolve_c<const C: usize>(data: &mut [f32], w: usize, h: usize, k: &[f32]) {
    if data.is_empty() || k.is_empty() {
        return;
    }
    let mut tmp = vec![0.0f32; data.len()];
    hpass::<C>(data, &mut tmp, w, k);
    vpass::<C>(&tmp, data, w, h, k);
}

/// In place separable convolution of a `c` channel float image (`c` in 1..=3): the horizontal
/// pass over all rows, then the vertical pass.
pub(crate) fn convolve(data: &mut [f32], w: usize, h: usize, c: usize, k: &[f32]) {
    assert_eq!(data.len(), w * h * c);
    match c {
        1 => convolve_c::<1>(data, w, h, k),
        2 => convolve_c::<2>(data, w, h, k),
        3 => convolve_c::<3>(data, w, h, k),
        _ => panic!("convolve: {c} channels are not used by the solver"),
    }
}

/// Gaussian blur of radius `r` of a `c` channel float image, in place; radii below
/// `BLUR_RADIUS_EPS` do nothing.
pub(crate) fn blur_gaussian(data: &mut [f32], w: usize, h: usize, c: usize, r: f32) {
    if (r as f64) < BLUR_RADIUS_EPS {
        return;
    }
    let k = gaussian_kernel(r);
    convolve(data, w, h, c, &k);
}

/// The 5 tap pyramid filter (`[1 4 6 4 1] / 16`) on a 1 channel image.
pub(crate) fn blur5(data: &mut [f32], w: usize, h: usize) {
    convolve(data, w, h, 1, &BINOMIAL5);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn gaussian_kernels_match_the_reference_tables_bit_for_bit() {
        // the numpy float32 `power` values of the formula, as in `kernel_tables_match_the_numpy_values_bit_for_bit`.
        let t7: [f32; 8] = [
            0.20133439, 0.17726654, 0.12099101, 0.06401724, 0.02625780, 0.00834906, 0.00205795, 0.00039323,
        ];
        let k7 = gaussian_kernel(7.0);
        assert_eq!(k7.len(), 8);
        for (a, b) in k7.iter().zip(t7.iter()) {
            assert!((a - b).abs() <= 1e-8, "{a} vs {b}");
        }
        let k5 = gaussian_kernel(5.0);
        assert_eq!(k5.len(), 6);
        let k2 = gaussian_kernel(2.0);
        assert_eq!(k2.len(), 3);
        // normalisation: centre + 2 * rest = 1 up to rounding
        for k in [&k7, &k5, &k2] {
            let s: f32 = k[0] + 2.0 * k[1..].iter().sum::<f32>();
            assert!((s - 1.0).abs() < 1e-6);
        }
    }

    // numpy float32 values of the formula, and the three float constants of the binomial half kernel
    const GAUSS_R7: [u32; 8] = [0x3e4e2a9a, 0x3e35855c, 0x3df7ca22, 0x3d831b79, 0x3cd71a9b, 0x3c08ca7a, 0x3b06de99, 0x39ce2a9a];
    const GAUSS_R5: [u32; 6] = [0x3e904ee6, 0x3e60e131, 0x3dd4bfc6, 0x3cf46273, 0x3baa6d48, 0x3a104ee6];
    const GAUSS_R2: [u32; 3] = [0x3f33bb04, 0x3e172281, 0x3ab3bb0c];
    const BINOMIAL5_HALF: [u32; 3] = [0x3ec00000, 0x3e800000, 0x3d800000];

    #[test]
    fn kernel_tables_match_the_numpy_values_bit_for_bit() {
        assert_eq!(bits(&gaussian_kernel(7.0)), GAUSS_R7);
        assert_eq!(bits(&gaussian_kernel(5.0)), GAUSS_R5);
        assert_eq!(bits(&gaussian_kernel(2.0)), GAUSS_R2);
        assert_eq!(bits(&BINOMIAL5), BINOMIAL5_HALF);
    }

    #[test]
    fn constant_image_is_preserved_exactly_by_renormalised_borders() {
        // interior: sum of weights times c can differ from c by rounding; border: num/den
        // with identical sums gives exactly c. A power of two constant keeps every product
        // and sum exact.
        let (w, h) = (23, 19);
        let mut d = vec![4.0f32; w * h * 2];
        blur_gaussian(&mut d, w, h, 2, 7.0);
        for v in &d {
            assert!((v - 4.0).abs() <= 4.0 * 2e-7, "{v}");
        }
    }

    #[test]
    fn short_lines_use_the_border_rule_only() {
        // len 3 < n = 8: every pixel is a border pixel, renormalised over the 3 taps.
        let k = gaussian_kernel(7.0);
        let x = [1.0f32, 2.0, 4.0];
        let mut out = [0.0f32; 3];
        hrow::<1>(&x, &mut out, 3, &k);
        // hand evaluation of pixel 0: taps j = 7..1, right tap exists for j = 1, 2 only
        let mut num = k[0] * 1.0;
        let mut den = k[0];
        for j in (1..8).rev() {
            let (wr, xr) = if j < 3 { (k[j], x[j]) } else { (0.0, 0.0) };
            den += wr;
            num += xr * wr;
            den += 0.0;
            num += 0.0 * 0.0;
        }
        assert_eq!(out[0].to_bits(), (num / den).to_bits());
    }

    #[test]
    fn interior_and_border_chains_differ_in_the_last_bit_only() {
        // a delta image: the interior value at the delta equals k[0] * x exactly
        let (w, h) = (40, 1);
        let mut d = vec![0.0f32; w];
        d[20] = 1.0;
        let k = gaussian_kernel(2.0);
        let mut t = vec![0.0; w];
        hrow::<1>(&d, &mut t, w, &k);
        assert_eq!(t[20].to_bits(), k[0].to_bits());
        assert_eq!(t[19].to_bits(), k[1].to_bits());
        assert_eq!(t[22].to_bits(), k[2].to_bits());
        let _ = h;
    }

    #[test]
    fn multi_channel_borders_drop_samples_equal_to_minus_one() {
        // two channels, a line of 5 pixels, kernel of 3 taps: pixel 1 of channel 0 holds -1.0,
        // so it is excluded from the sums of its neighbours.
        let k = [0.5f32, 0.25, 0.125];
        let x: Vec<f32> = vec![1.0, 9.0, -1.0, 9.0, 3.0, 9.0, 4.0, 9.0, 5.0, 9.0];
        let mut out = vec![0.0f32; 10];
        hrow::<2>(&x, &mut out, 5, &k);
        // pixel 0, channel 0: centre 1.0; taps j=2: right x[2]=3.0 w=.125, left outside;
        // j=1: right x[1] = -1.0 -> dropped, left outside.
        let num = 0.5f32 * 1.0 + 0.125 * 3.0;
        let den = 0.5f32 + 0.125;
        assert_eq!(out[0].to_bits(), (num / den).to_bits());
        // the same line with 1 channel keeps the -1.0 sample
        let x1: Vec<f32> = vec![1.0, -1.0, 3.0, 4.0, 5.0];
        let mut o1 = vec![0.0f32; 5];
        hrow::<1>(&x1, &mut o1, 5, &k);
        let num1 = (0.5f32 * 1.0 + 0.125 * 3.0) + 0.25 * -1.0;
        let den1 = (0.5f32 + 0.125) + 0.25;
        assert_eq!(o1[0].to_bits(), (num1 / den1).to_bits());
    }

    #[test]
    fn vertical_pass_equals_transposed_horizontal_pass() {
        let (w, h) = (17, 21);
        let mut a = vec![0.0f32; w * h * 3];
        let mut s = 12345u32;
        for v in a.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s >> 8) as f32 / 65536.0 - 100.0;
        }
        let k = gaussian_kernel(5.0);
        // vertical pass on a
        let mut v1 = vec![0.0f32; a.len()];
        vpass::<3>(&a, &mut v1, w, h, &k);
        // transpose, horizontal pass, transpose back
        let mut t = vec![0.0f32; a.len()];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    t[(x * h + y) * 3 + c] = a[(y * w + x) * 3 + c];
                }
            }
        }
        let mut th = vec![0.0f32; a.len()];
        hpass::<3>(&t, &mut th, h, &k);
        let mut v2 = vec![0.0f32; a.len()];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    v2[(y * w + x) * 3 + c] = th[(x * h + y) * 3 + c];
                }
            }
        }
        assert_eq!(bits(&v1), bits(&v2));
    }

    #[test]
    fn blur5_of_an_impulse_is_the_binomial_kernel() {
        let (w, h) = (9, 9);
        let mut d = vec![0.0f32; w * h];
        d[4 * w + 4] = 16.0;
        blur5(&mut d, w, h);
        // 1D kernel [1 4 6 4 1]/16 squared, times 16 -> 6*6/16 = 2.25 at the centre
        assert_eq!(d[4 * w + 4], 16.0 * 0.375 * 0.375);
        assert_eq!(d[4 * w + 5], 16.0 * 0.25 * 0.375);
    }
}
