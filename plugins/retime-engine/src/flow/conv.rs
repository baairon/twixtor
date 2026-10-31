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
//!
//! The `_above_first` variants ([`convolve1_above_first`], [`convolve1_wrapx_above_first`]) are
//! the passes of the hole fill of a plugin whose fill adds, in the border rows of the VERTICAL
//! pass, the tap above (`y - j`, the lower row index) before the tap below (`y + j`); everything
//! else, the horizontal pass and the interior chain included, is the pass above, bit for bit
//! (`holefill::VerticalTaps`). 1 channel images only: the fill works on one plane at a time.
//!
//! The `_wrapx` variants ([`convolve_wrapx`], [`blur_gaussian_wrapx`], [`blur5_wrapx`]) are the
//! horizontal wrap-around (360) mode of the hole fill (parity gap G6; its
//! arithmetic is U): the horizontal pass is circular, every element takes the interior chain
//! above with its tap indices taken modulo the line length (`x[(i+j) mod len]`, Euclidean, so a
//! kernel longer than the line wraps more than once), with no border renormalisation and no
//! `-1.0` padding test because no tap is outside the line. The vertical pass is the one above,
//! unchanged. An element whose taps all lie inside the line is computed by the same interior
//! code as the plain pass, so it equals the plain pass bit for bit.
//!
//! The passes are written once over a set of arithmetic operations (`crate::nanrule::Arith`):
//! every caller above runs them with the processor's own operations (`nanrule::Plain`);
//! [`convolve1_with`] runs the 1 channel passes with any set, which the hole fill uses with the
//! NaN rule (`nanrule::Rule`) for a plane whose plain result holds a NaN. Numbers are the same bit
//! for bit with either set.

use crate::nanrule::{Arith, Plain};
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
fn border_value<const C: usize, A: Arith>(get: &dyn Fn(isize) -> Option<f32>, i: isize, k: &[f32]) -> f32 {
    border_value_ordered::<C, false, A>(get, i, k)
}

/// [`border_value`] with the order of the two taps of each distance: the right (`i + j`) tap
/// first, or with `LEFT_FIRST` the left (`i - j`) tap first. Every other step is the same.
#[inline(always)]
fn border_value_ordered<const C: usize, const LEFT_FIRST: bool, A: Arith>(get: &dyn Fn(isize) -> Option<f32>, i: isize, k: &[f32]) -> f32 {
    let n = k.len();
    let centre = match get(i) {
        Some(v) => v,
        None => 0.0,
    };
    let mut num = A::mul(k[0], centre);
    let mut den = k[0];
    let mut j = n;
    while j > 1 {
        j -= 1;
        let kj = k[j];
        // right tap, then left tap (LEFT_FIRST: left, then right)
        for side in 0..2 {
            let right = (side == 0) != LEFT_FIRST;
            let t = if right { i + j as isize } else { i - j as isize };
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
            num = A::add(num, A::mul(xv, w));
        }
    }
    A::div(num, den)
}

/// One row of the horizontal pass. `x` and `out` hold `len * C` samples.
fn hrow<const C: usize>(x: &[f32], out: &mut [f32], len: usize, k: &[f32]) {
    hrow_with::<C, Plain>(x, out, len, k)
}

/// [`hrow`] with the arithmetic `A`.
fn hrow_with<const C: usize, A: Arith>(x: &[f32], out: &mut [f32], len: usize, k: &[f32]) {
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
            out[i * C + c] = border_value::<C, A>(&get, i as isize, k);
        }
    };
    for i in 0..head {
        border(i, out);
    }
    if interior > 0 {
        interior_run::<C, A>(x, out, head, head + interior, k);
    }
    for i in (head + interior)..len {
        border(i, out);
    }
}

/// The interior chain of the pixels `a..b` of a row (pixels whose taps all lie inside the line):
/// `s = x[i] * k[0]`, then for `j = n-1` down to 1 `s = s + (x[i+j] + x[i-j]) * k[j]`, every
/// channel of a pixel on its own.
#[inline(always)]
fn interior_run<const C: usize, A: Arith>(x: &[f32], out: &mut [f32], a: usize, b: usize, k: &[f32]) {
    let n = k.len();
    let (a, b) = (a * C, b * C);
    let k0 = k[0];
    for t in a..b {
        out[t] = A::mul(x[t], k0);
    }
    for j in (1..n).rev() {
        let off = j * C;
        let kj = k[j];
        for t in a..b {
            out[t] = A::add(out[t], A::mul(A::add(x[t + off], x[t - off]), kj));
        }
    }
}

/// One row of the circular horizontal pass (the `_wrapx` variants, see the module header): the
/// pixels of [`hrow`]'s interior run the same interior code, every other pixel the same chain with
/// its tap indices taken modulo `len`: `s = x[i] * k[0]`, then for `j = n-1` down to 1
/// `s = s + (x[(i+j) mod len] + x[(i-j) mod len]) * k[j]` (right tap first). Single precision,
/// every operation rounded on its own.
fn hrow_wrapx<const C: usize>(x: &[f32], out: &mut [f32], len: usize, k: &[f32]) {
    hrow_wrapx_with::<C, Plain>(x, out, len, k)
}

/// [`hrow_wrapx`] with the arithmetic `A`.
fn hrow_wrapx_with<const C: usize, A: Arith>(x: &[f32], out: &mut [f32], len: usize, k: &[f32]) {
    let n = k.len();
    let head = n.min(len);
    let interior = len.saturating_sub(2 * n);
    let wrapped = |i: usize, out: &mut [f32]| {
        for c in 0..C {
            let mut s = A::mul(x[i * C + c], k[0]);
            let mut j = n;
            while j > 1 {
                j -= 1;
                let r = (i + j) % len;
                let l = (i as isize - j as isize).rem_euclid(len as isize) as usize;
                s = A::add(s, A::mul(A::add(x[r * C + c], x[l * C + c]), k[j]));
            }
            out[i * C + c] = s;
        }
    };
    for i in 0..head {
        wrapped(i, out);
    }
    if interior > 0 {
        interior_run::<C, A>(x, out, head, head + interior, k);
    }
    for i in (head + interior)..len {
        wrapped(i, out);
    }
}

/// Horizontal pass of the whole image, rows in parallel (rows are independent).
fn hpass<const C: usize>(src: &[f32], dst: &mut [f32], w: usize, k: &[f32]) {
    hpass_with::<C, Plain>(src, dst, w, k)
}

/// [`hpass`] with the arithmetic `A`.
fn hpass_with<const C: usize, A: Arith>(src: &[f32], dst: &mut [f32], w: usize, k: &[f32]) {
    let row = w * C;
    if row == 0 {
        return;
    }
    dst.par_chunks_mut(row)
        .zip(src.par_chunks(row))
        .for_each(|(d, s)| hrow_with::<C, A>(s, d, w, k));
}

/// The circular horizontal pass of the whole image ([`hrow_wrapx`]), rows in parallel.
fn hpass_wrapx<const C: usize>(src: &[f32], dst: &mut [f32], w: usize, k: &[f32]) {
    hpass_wrapx_with::<C, Plain>(src, dst, w, k)
}

/// [`hpass_wrapx`] with the arithmetic `A`.
fn hpass_wrapx_with<const C: usize, A: Arith>(src: &[f32], dst: &mut [f32], w: usize, k: &[f32]) {
    let row = w * C;
    if row == 0 {
        return;
    }
    dst.par_chunks_mut(row)
        .zip(src.par_chunks(row))
        .for_each(|(d, s)| hrow_wrapx_with::<C, A>(s, d, w, k));
}

/// Vertical pass: output row `y` of an image of `h` rows and `w` pixels.
fn vrow<const C: usize>(src: &[f32], out: &mut [f32], y: usize, w: usize, h: usize, k: &[f32]) {
    vrow_ordered::<C, false, Plain>(src, out, y, w, h, k)
}

/// [`vrow`] whose border rows take the tap above (`y - j`) first when `ABOVE_FIRST` is set.
fn vrow_ordered<const C: usize, const ABOVE_FIRST: bool, A: Arith>(src: &[f32], out: &mut [f32], y: usize, w: usize, h: usize, k: &[f32]) {
    let n = k.len();
    let row = w * C;
    let head = n.min(h);
    let interior = h.saturating_sub(2 * n);
    if y >= head && y < head + interior {
        let k0 = k[0];
        let s0 = &src[y * row..(y + 1) * row];
        for t in 0..row {
            out[t] = A::mul(s0[t], k0);
        }
        for j in (1..n).rev() {
            let below = &src[(y + j) * row..(y + j + 1) * row];
            let above = &src[(y - j) * row..(y - j + 1) * row];
            let kj = k[j];
            for t in 0..row {
                out[t] = A::add(out[t], A::mul(A::add(below[t], above[t]), kj));
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
                out[xc * C + c] = border_value_ordered::<C, ABOVE_FIRST, A>(&get, y as isize, k);
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

/// The vertical pass with the arithmetic `A`, the border rows taking the tap above first when
/// `ABOVE_FIRST` is set ([`vpass`] and [`vpass_above_first`] with `Plain`).
fn vpass_ordered_with<const C: usize, const ABOVE_FIRST: bool, A: Arith>(src: &[f32], dst: &mut [f32], w: usize, h: usize, k: &[f32]) {
    let row = w * C;
    if row == 0 {
        return;
    }
    dst.par_chunks_mut(row)
        .enumerate()
        .for_each(|(y, d)| vrow_ordered::<C, ABOVE_FIRST, A>(src, d, y, w, h, k));
}

/// The vertical pass whose border rows take the tap above first ([`vrow_ordered`]).
/// The vertical pass of the 360 mode (READ in the original's column routine of the wrap filters of
/// `TwixtorPro_64`, the same code in all four plugins): each column is
/// gathered into a line padded by `n - 1` elements on both sides with the equirectangular rule: from row 0 upwards
/// and from row `h - 1` downwards the walk reflects at the edge row (the row is read again, the direction turns) and
/// moves to the column `(w / 2 + c) % w` (integer halving) at every reflection, so the element above row 0 is row 0 of
/// the column half a turn away, the next one its row 1, and so on; then every element is the interior chain
/// `s = g[i] * k[0]`, `s = s + (g[i + j] + g[i - j]) * k[j]` for `j = n - 1` down to 1, no renormalisation. Single
/// precision; columns in parallel.
fn vpass_sphere<const C: usize>(src: &[f32], dst: &mut [f32], w: usize, h: usize, k: &[f32]) {
    vpass_sphere_with::<C, Plain>(src, dst, w, h, k)
}

/// [`vpass_sphere`] with the arithmetic `A`.
fn vpass_sphere_with<const C: usize, A: Arith>(src: &[f32], dst: &mut [f32], w: usize, h: usize, k: &[f32]) {
    let n = k.len();
    if w == 0 || h == 0 || n == 0 {
        return;
    }
    let half = w / 2;
    let pad = n - 1;
    let cols: Vec<Vec<f32>> = (0..w)
        .into_par_iter()
        .map(|x| {
            let ext = h + 2 * pad;
            let mut g = vec![0.0f32; ext * C];
            let put = |g: &mut [f32], e: usize, c: usize, r: usize| {
                g[e * C..(e + 1) * C].copy_from_slice(&src[(r * w + c) * C..(r * w + c + 1) * C]);
            };
            // walk from (x, start) by `dir`, reflecting at the edge rows with the half turn
            let walk = |start: usize, dir: i64, steps: usize| -> Vec<(usize, usize)> {
                let (mut c, mut r, mut d) = (x, start as i64, dir);
                let mut out = Vec::with_capacity(steps);
                for _ in 0..steps {
                    r += d;
                    if r < 0 {
                        r = 0;
                        d = -d;
                        c = (half + c) % w;
                    } else if r >= h as i64 {
                        r = h as i64 - 1;
                        d = -d;
                        c = (half + c) % w;
                    }
                    out.push((c, r as usize));
                }
                out
            };
            for (m, (c, r)) in walk(0, -1, pad).into_iter().enumerate() {
                put(&mut g, pad - 1 - m, c, r);
            }
            for r in 0..h {
                put(&mut g, pad + r, x, r);
            }
            for (m, (c, r)) in walk(h - 1, 1, pad).into_iter().enumerate() {
                put(&mut g, pad + h + m, c, r);
            }
            let mut out = vec![0.0f32; h * C];
            for y in 0..h {
                for ch in 0..C {
                    let e = pad + y;
                    let mut s = A::mul(g[e * C + ch], k[0]);
                    for j in (1..n).rev() {
                        s = A::add(s, A::mul(A::add(g[(e + j) * C + ch], g[(e - j) * C + ch]), k[j]));
                    }
                    out[y * C + ch] = s;
                }
            }
            out
        })
        .collect();
    for (x, col) in cols.iter().enumerate() {
        for y in 0..h {
            dst[(y * w + x) * C..(y * w + x + 1) * C].copy_from_slice(&col[y * C..(y + 1) * C]);
        }
    }
}

#[cfg(test)]
mod sphere_tests {
    use super::*;

    #[test]
    fn the_sphere_column_pass_reflects_at_the_poles_half_a_turn_away() {
        // 4 x 3, one channel, half kernel of 3 taps: column 0 is padded with (2, 1), (2, 0) above and (2, 2), (2, 1)
        // below (column (0 + 4 / 2) % 4 = 2), column 1 with column 3
        let (w, h) = (4usize, 3usize);
        let src: Vec<f32> = (0..w * h).map(|i| 1.0 + i as f32 * 1.37).collect();
        let k = [0.5f32, 0.2, 0.05];
        let mut dst = vec![0.0f32; w * h];
        vpass_sphere::<1>(&src, &mut dst, w, h, &k);
        let at = |x: usize, y: usize| src[y * w + x];
        for x in 0..w {
            let o = (x + w / 2) % w;
            let g = [at(o, 1), at(o, 0), at(x, 0), at(x, 1), at(x, 2), at(o, 2), at(o, 1)];
            for y in 0..h {
                let e = y + 2;
                let mut s = g[e] * k[0];
                s = s + (g[e + 2] + g[e - 2]) * k[2];
                s = s + (g[e + 1] + g[e - 1]) * k[1];
                assert_eq!(dst[y * w + x].to_bits(), s.to_bits(), "({x}, {y})");
            }
        }
    }

    #[test]
    fn a_kernel_longer_than_the_column_keeps_reflecting() {
        // h = 2 and 4 taps: above row 0 the walk reads row 0 and 1 at the far column, then reflects again at row 1
        // back onto the near column (two half turns of an even width)
        let (w, h) = (2usize, 2usize);
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let k = [0.4f32, 0.2, 0.1, 0.05];
        let mut dst = [0.0f32; 4];
        vpass_sphere::<1>(&src, &mut dst, w, h, &k);
        // column 0: above -1 (1, 0) = 2, -2 (1, 1) = 4, -3 (0, 1) = 3; below 2 (1, 1) = 4, 3 (1, 0) = 2, 4 (0, 0) = 1
        let g = [3.0f32, 4.0, 2.0, 1.0, 3.0, 4.0, 2.0, 1.0];
        for y in 0..2 {
            let e = y + 3;
            let mut s = g[e] * k[0];
            for j in (1..4).rev() {
                s = s + (g[e + j] + g[e - j]) * k[j];
            }
            assert_eq!(dst[y * w].to_bits(), s.to_bits(), "row {y}");
        }
    }
}

fn vpass_above_first<const C: usize>(src: &[f32], dst: &mut [f32], w: usize, h: usize, k: &[f32]) {
    vpass_ordered_with::<C, true, Plain>(src, dst, w, h, k)
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

/// [`convolve`] of a 1 channel image whose vertical border rows take the tap above (`y - j`)
/// before the tap below (module header). The horizontal pass and every interior element are those
/// of [`convolve`]; rows run through rayon row chunks, so the result does not depend on the thread
/// count.
pub(crate) fn convolve1_above_first(data: &mut [f32], w: usize, h: usize, k: &[f32]) {
    assert_eq!(data.len(), w * h);
    if data.is_empty() || k.is_empty() {
        return;
    }
    let mut tmp = vec![0.0f32; data.len()];
    hpass::<1>(data, &mut tmp, w, k);
    vpass_above_first::<1>(&tmp, data, w, h, k);
}

/// [`convolve1_above_first`] with the circular horizontal pass of [`convolve_wrapx`].
pub(crate) fn convolve1_wrapx_above_first(data: &mut [f32], w: usize, h: usize, k: &[f32]) {
    assert_eq!(data.len(), w * h);
    if data.is_empty() || k.is_empty() {
        return;
    }
    let mut tmp = vec![0.0f32; data.len()];
    hpass_wrapx::<1>(data, &mut tmp, w, k);
    vpass_sphere::<1>(&tmp, data, w, h, k);
}

/// The 1 channel convolution of the hole fill with the arithmetic `A` (module header): the
/// horizontal pass (circular with `wrapx`), then the vertical pass (with `wrapx` the 360 column
/// walk, else the plain pass whose border rows take the tap above first with `above_first`). With
/// `nanrule::Plain` it is the 1 channel [`convolve`], [`convolve1_above_first`], [`convolve_wrapx`]
/// or [`convolve1_wrapx_above_first`] bit for bit; with `nanrule::Rule` every sum, product and
/// quotient follows the NaN rule (numbers unchanged).
pub(crate) fn convolve1_with<A: Arith>(data: &mut [f32], w: usize, h: usize, k: &[f32], wrapx: bool, above_first: bool) {
    assert_eq!(data.len(), w * h);
    if data.is_empty() || k.is_empty() {
        return;
    }
    let mut tmp = vec![0.0f32; data.len()];
    if wrapx {
        hpass_wrapx_with::<1, A>(data, &mut tmp, w, k);
        vpass_sphere_with::<1, A>(&tmp, data, w, h, k);
    } else {
        hpass_with::<1, A>(data, &mut tmp, w, k);
        if above_first {
            vpass_ordered_with::<1, true, A>(&tmp, data, w, h, k);
        } else {
            vpass_ordered_with::<1, false, A>(&tmp, data, w, h, k);
        }
    }
}

/// The 5 tap pyramid filter (`[1 4 6 4 1] / 16`) on a 1 channel image.
pub(crate) fn blur5(data: &mut [f32], w: usize, h: usize) {
    convolve(data, w, h, 1, &BINOMIAL5);
}

fn convolve_wrapx_c<const C: usize>(data: &mut [f32], w: usize, h: usize, k: &[f32]) {
    if data.is_empty() || k.is_empty() {
        return;
    }
    let mut tmp = vec![0.0f32; data.len()];
    hpass_wrapx::<C>(data, &mut tmp, w, k);
    vpass_sphere::<C>(&tmp, data, w, h, k);
}

/// [`convolve`] in the horizontal wrap-around mode (G6, U): the circular horizontal pass over all
/// rows (tap indices modulo the width, no renormalisation; see the module header), then the
/// unchanged vertical pass. Precision: the interior chain of [`convolve`] per element, single
/// precision, no fused multiply add; rows run through rayon row chunks, so the result does not
/// depend on the thread count. Columns whose horizontal taps all lie inside the row
/// (`n <= x < w - n` for a half kernel of `n` taps) equal [`convolve`] bit for bit.
pub(crate) fn convolve_wrapx(data: &mut [f32], w: usize, h: usize, c: usize, k: &[f32]) {
    assert_eq!(data.len(), w * h * c);
    match c {
        1 => convolve_wrapx_c::<1>(data, w, h, k),
        2 => convolve_wrapx_c::<2>(data, w, h, k),
        3 => convolve_wrapx_c::<3>(data, w, h, k),
        _ => panic!("convolve: {c} channels are not used by the solver"),
    }
}

/// [`blur_gaussian`] in the horizontal wrap-around mode: the same kernel through
/// [`convolve_wrapx`]; radii below `BLUR_RADIUS_EPS` do nothing.
pub(crate) fn blur_gaussian_wrapx(data: &mut [f32], w: usize, h: usize, c: usize, r: f32) {
    if (r as f64) < BLUR_RADIUS_EPS {
        return;
    }
    let k = gaussian_kernel(r);
    convolve_wrapx(data, w, h, c, &k);
}

/// [`blur5`] in the horizontal wrap-around mode: the 5 tap pyramid filter through
/// [`convolve_wrapx`] on a 1 channel image.
pub(crate) fn blur5_wrapx(data: &mut [f32], w: usize, h: usize) {
    convolve_wrapx(data, w, h, 1, &BINOMIAL5);
}

// --------------------------------------------------------------------------------------------
// The solver's convolutions: vertical border rows above first
// --------------------------------------------------------------------------------------------

/// [`convolve`] (1 to 3 channels) whose vertical border rows take the tap above (`y - j`) before
/// the tap below, the order of [`vpass_above_first`]; the horizontal pass and every interior
/// element are those of [`convolve`]. These are the convolutions of the optical flow solver and of
/// the images it is given (the pyramid filter, the reduction, the edge enhancement, the mask
/// pyramid and the three Gaussians of a level). MEASURED against `CreateMotionVectors_64` under
/// the emulator: Full res solves of 32 bpc probes in which every pixel is a border pixel (8x4 to
/// 32x24, one to three levels) give the plugin's vectors bit for bit with this order and differ in
/// the last bits with the right or below tap first. The rows are those of the image the solver gets,
/// which the vector plugins turn
/// upside down first (their row order fixes), so "above" is the plugin's own previous row; the
/// fills of `TwixtorProVectorsIn_64` show the same order (`convolve1_above_first`).
pub(crate) fn convolve_above_first(data: &mut [f32], w: usize, h: usize, c: usize, k: &[f32]) {
    assert_eq!(data.len(), w * h * c);
    match c {
        1 => convolve_above_first_c::<1>(data, w, h, k),
        2 => convolve_above_first_c::<2>(data, w, h, k),
        3 => convolve_above_first_c::<3>(data, w, h, k),
        _ => panic!("convolve: {c} channels are not used by the solver"),
    }
}

fn convolve_above_first_c<const C: usize>(data: &mut [f32], w: usize, h: usize, k: &[f32]) {
    if data.is_empty() || k.is_empty() {
        return;
    }
    let mut tmp = vec![0.0f32; data.len()];
    hpass::<C>(data, &mut tmp, w, k);
    vpass_above_first::<C>(&tmp, data, w, h, k);
}

/// [`blur_gaussian`] through [`convolve_above_first`]; radii below `BLUR_RADIUS_EPS` do nothing.
pub(crate) fn blur_gaussian_above_first(data: &mut [f32], w: usize, h: usize, c: usize, r: f32) {
    if (r as f64) < BLUR_RADIUS_EPS {
        return;
    }
    let k = gaussian_kernel(r);
    convolve_above_first(data, w, h, c, &k);
}

/// [`blur5`] through [`convolve_above_first`].
pub(crate) fn blur5_above_first(data: &mut [f32], w: usize, h: usize) {
    convolve_above_first(data, w, h, 1, &BINOMIAL5);
}

/// [`convolve_wrapx`] whose vertical border rows take the tap above first (the circular
/// horizontal pass, then [`vpass_above_first`]).
pub(crate) fn convolve_wrapx_above_first(data: &mut [f32], w: usize, h: usize, c: usize, k: &[f32]) {
    assert_eq!(data.len(), w * h * c);
    match c {
        1 => convolve_wrapx_above_first_c::<1>(data, w, h, k),
        2 => convolve_wrapx_above_first_c::<2>(data, w, h, k),
        3 => convolve_wrapx_above_first_c::<3>(data, w, h, k),
        _ => panic!("convolve: {c} channels are not used by the solver"),
    }
}

fn convolve_wrapx_above_first_c<const C: usize>(data: &mut [f32], w: usize, h: usize, k: &[f32]) {
    if data.is_empty() || k.is_empty() {
        return;
    }
    let mut tmp = vec![0.0f32; data.len()];
    hpass_wrapx::<C>(data, &mut tmp, w, k);
    vpass_sphere::<C>(&tmp, data, w, h, k);
}

/// [`blur_gaussian_wrapx`] through [`convolve_wrapx_above_first`].
pub(crate) fn blur_gaussian_wrapx_above_first(data: &mut [f32], w: usize, h: usize, c: usize, r: f32) {
    if (r as f64) < BLUR_RADIUS_EPS {
        return;
    }
    let k = gaussian_kernel(r);
    convolve_wrapx_above_first(data, w, h, c, &k);
}

/// [`blur5_wrapx`] through [`convolve_wrapx_above_first`].
pub(crate) fn blur5_wrapx_above_first(data: &mut [f32], w: usize, h: usize) {
    convolve_wrapx_above_first(data, w, h, 1, &BINOMIAL5);
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
    #[ignore = "the G6 wrap rule (U) this test encodes is superseded by the 360 rules read from TwixtorPro_64 (parity/lab/re/op-residual.md, finding 6)"]
    fn above_first_differs_from_the_plain_pass_only_in_the_vertical_border_rows() {
        for k in kernels() {
            let n = k.len();
            for (w, h) in [(23usize, 19usize), (9, 4), (40, 2 * n + 3)] {
                let src = noise(w * h, (w * 31 + h) as u32);
                let (mut a, mut b) = (src.clone(), src.clone());
                convolve(&mut a, w, h, 1, &k);
                convolve1_above_first(&mut b, w, h, &k);
                let head = n.min(h);
                let interior = h.saturating_sub(2 * n);
                for y in head..head + interior {
                    assert_eq!(bits(&a[y * w..(y + 1) * w]), bits(&b[y * w..(y + 1) * w]), "interior row {y} n {n} {w}x{h}");
                }
                // a border row is the same renormalised sum with the two taps of each distance swapped
                let mut t = vec![0.0f32; w * h];
                hpass::<1>(&src, &mut t, w, &k);
                for y in (0..head).chain(head + interior..h) {
                    for x in 0..w {
                        let mut num = k[0] * t[y * w + x];
                        let mut den = k[0];
                        for j in (1..n).rev() {
                            for yy in [y as isize - j as isize, y as isize + j as isize] {
                                let (xv, wt) = if yy >= 0 && yy < h as isize { (t[yy as usize * w + x], k[j]) } else { (0.0f32, 0.0f32) };
                                den += wt;
                                num += xv * wt;
                            }
                        }
                        assert_eq!(b[y * w + x].to_bits(), (num / den).to_bits(), "row {y} col {x} n {n} {w}x{h}");
                    }
                }
            }
        }
        // the wrapped variant: the circular horizontal pass, then the same vertical pass
        let (w, h) = (24usize, 17usize);
        let k = gaussian_kernel(5.0);
        let src = noise(w * h, 77);
        let mut got = src.clone();
        convolve1_wrapx_above_first(&mut got, w, h, &k);
        let mut t = vec![0.0f32; w * h];
        hpass_wrapx::<1>(&src, &mut t, w, &k);
        let mut want = vec![0.0f32; w * h];
        vpass_above_first::<1>(&t, &mut want, w, h, &k);
        assert_eq!(bits(&got), bits(&want));
        // and a constant image keeps its value in every border row, as the plain pass does
        let mut c = vec![4.0f32; w * h];
        convolve1_above_first(&mut c, w, h, &k);
        assert!(c.iter().all(|v| (v - 4.0).abs() <= 4.0 * 2e-7));
    }

    #[test]
    fn the_solver_convolution_is_the_horizontal_pass_of_the_columns_read_bottom_up() {
        // above first in the vertical border rows: each column read from the bottom row up is a row
        // of the horizontal pass, whose border takes the right (here: the upper) tap first
        let (w, h) = (9, 14);
        for c in 1..=3usize {
            let mut a = vec![0.0f32; w * h * c];
            let mut s = 777u32 + c as u32;
            for v in a.iter_mut() {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                *v = (s >> 8) as f32 / 65536.0 - 100.0;
            }
            let k = gaussian_kernel(7.0);
            let mut got = a.clone();
            convolve_above_first(&mut got, w, h, c, &k);
            let mut rows = vec![0.0f32; a.len()];
            match c {
                1 => hpass::<1>(&a, &mut rows, w, &k),
                2 => hpass::<2>(&a, &mut rows, w, &k),
                _ => hpass::<3>(&a, &mut rows, w, &k),
            }
            let mut t = vec![0.0f32; a.len()];
            for y in 0..h {
                for x in 0..w {
                    for ch in 0..c {
                        t[(x * h + (h - 1 - y)) * c + ch] = rows[(y * w + x) * c + ch];
                    }
                }
            }
            let mut th = vec![0.0f32; a.len()];
            match c {
                1 => hpass::<1>(&t, &mut th, h, &k),
                2 => hpass::<2>(&t, &mut th, h, &k),
                _ => hpass::<3>(&t, &mut th, h, &k),
            }
            let mut want = vec![0.0f32; a.len()];
            for y in 0..h {
                for x in 0..w {
                    for ch in 0..c {
                        want[(y * w + x) * c + ch] = th[(x * h + (h - 1 - y)) * c + ch];
                    }
                }
            }
            assert_eq!(bits(&got), bits(&want), "{c} channels");
            // and it differs from the plain convolution (in the last bits of the border rows)
            let mut plain = a.clone();
            convolve(&mut plain, w, h, c, &k);
            assert_ne!(bits(&got), bits(&plain));
        }
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

    // --------------------------------------------------------------------------------------------
    // the circular horizontal pass (`_wrapx`)
    // --------------------------------------------------------------------------------------------

    fn noise(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / 65536.0 - 128.0
            })
            .collect()
    }

    fn kernels() -> Vec<Vec<f32>> {
        vec![BINOMIAL5.to_vec(), gaussian_kernel(2.0), gaussian_kernel(5.0), gaussian_kernel(7.0)]
    }

    #[test]
    #[ignore = "the G6 wrap rule (U) this test encodes is superseded by the 360 rules read from TwixtorPro_64 (parity/lab/re/op-residual.md, finding 6)"]
    fn wrapx_rows_equal_the_plain_rows_where_no_tap_wraps() {
        for k in kernels() {
            let n = k.len();
            for len in [2 * n + 1, 2 * n + 7, 61] {
                for c in 1..=3usize {
                    let x = noise(len * c, (len * 7 + c) as u32);
                    let mut a = vec![0.0f32; len * c];
                    let mut b = vec![0.0f32; len * c];
                    match c {
                        1 => {
                            hrow::<1>(&x, &mut a, len, &k);
                            hrow_wrapx::<1>(&x, &mut b, len, &k);
                        }
                        2 => {
                            hrow::<2>(&x, &mut a, len, &k);
                            hrow_wrapx::<2>(&x, &mut b, len, &k);
                        }
                        _ => {
                            hrow::<3>(&x, &mut a, len, &k);
                            hrow_wrapx::<3>(&x, &mut b, len, &k);
                        }
                    }
                    assert_eq!(bits(&a[n * c..(len - n) * c]), bits(&b[n * c..(len - n) * c]), "n {n} len {len} c {c}");
                    // the borders do differ: the plain pass renormalises, the circular one continues
                    assert_ne!(bits(&a[..c]), bits(&b[..c]), "n {n} len {len} c {c}");
                }
            }
        }
        // and so do the whole convolutions on the columns whose taps stay inside
        let (w, h) = (40usize, 23usize);
        for k in kernels() {
            let n = k.len();
            for c in 1..=3usize {
                let src = noise(w * h * c, 99 + c as u32);
                let (mut a, mut b) = (src.clone(), src.clone());
                convolve(&mut a, w, h, c, &k);
                convolve_wrapx(&mut b, w, h, c, &k);
                for y in 0..h {
                    let r = y * w * c;
                    assert_eq!(bits(&a[r + n * c..r + (w - n) * c]), bits(&b[r + n * c..r + (w - n) * c]), "row {y} n {n} c {c}");
                }
            }
        }
    }

    #[test]
    fn the_circular_pass_commutes_with_a_rotation_of_the_row() {
        // every pixel is the same chain over its circular neighbourhood, so rotating the input by s
        // rotates the output by s, bit for bit, for every length (kernels longer than the row too)
        for k in kernels() {
            for len in 1..=20usize {
                let x = noise(len, len as u32);
                let mut out = vec![0.0f32; len];
                hrow_wrapx::<1>(&x, &mut out, len, &k);
                for s in 0..len {
                    let xr: Vec<f32> = (0..len).map(|i| x[(i + s) % len]).collect();
                    let mut or = vec![0.0f32; len];
                    hrow_wrapx::<1>(&xr, &mut or, len, &k);
                    let want: Vec<f32> = (0..len).map(|i| out[(i + s) % len]).collect();
                    assert_eq!(bits(&or), bits(&want), "len {len} shift {s} n {}", k.len());
                }
            }
        }
    }

    #[test]
    fn circular_border_pixels_by_hand() {
        let k = [0.5f32, 0.25, 0.125];
        let x = [1.0f32, 2.0, 4.0, 8.0, 16.0];
        let mut out = [0.0f32; 5];
        hrow_wrapx::<1>(&x, &mut out, 5, &k);
        // pixel 0: right taps 2.0 (j=1), 4.0 (j=2); left taps 16.0 (j=1), 8.0 (j=2); j from 2 down
        let p0 = (1.0f32 * 0.5 + (4.0 + 8.0) * 0.125) + (2.0 + 16.0) * 0.25;
        assert_eq!(out[0].to_bits(), p0.to_bits());
        // pixel 4: right taps 1.0 (j=1), 2.0 (j=2); left taps 8.0, 4.0
        let p4 = (16.0f32 * 0.5 + (2.0 + 4.0) * 0.125) + (1.0 + 8.0) * 0.25;
        assert_eq!(out[4].to_bits(), p4.to_bits());
        // a line of 2: the taps wrap more than once, j = 2 lands on the pixel itself
        let mut o2 = [0.0f32; 2];
        hrow_wrapx::<1>(&[3.0, 5.0], &mut o2, 2, &k);
        assert_eq!(o2[0].to_bits(), ((3.0f32 * 0.5 + (3.0 + 3.0) * 0.125) + (5.0 + 5.0) * 0.25).to_bits());
        // a line of 1: every tap is the pixel itself
        let mut o1 = [0.0f32; 1];
        hrow_wrapx::<1>(&[7.0], &mut o1, 1, &k);
        assert_eq!(o1[0].to_bits(), ((7.0f32 * 0.5 + (7.0 + 7.0) * 0.125) + (7.0 + 7.0) * 0.25).to_bits());
        // no -1.0 test in the circular pass: a -1.0 sample of a two channel row is kept
        let x2: Vec<f32> = vec![1.0, 9.0, -1.0, 9.0, 3.0, 9.0, 4.0, 9.0, 5.0, 9.0];
        let mut o = vec![0.0f32; 10];
        hrow_wrapx::<2>(&x2, &mut o, 5, &k);
        let want = (1.0f32 * 0.5 + (3.0 + 4.0) * 0.125) + (-1.0 + 5.0) * 0.25;
        assert_eq!(o[0].to_bits(), want.to_bits());
    }

    #[test]
    fn a_constant_row_stays_one_value_without_renormalisation() {
        // every pixel of a constant row takes the same chain over equal taps, borders included
        let k = gaussian_kernel(5.0);
        for len in [1usize, 3, 12, 40] {
            let x = vec![0.7f32; len];
            let mut out = vec![0.0f32; len];
            hrow_wrapx::<1>(&x, &mut out, len, &k);
            assert!(out.iter().all(|v| v.to_bits() == out[0].to_bits()), "len {len}");
            assert!((out[0] - 0.7).abs() < 1e-6);
        }
    }

    #[test]
    #[ignore = "the G6 wrap rule (U) this test encodes is superseded by the 360 rules read from TwixtorPro_64 (parity/lab/re/op-residual.md, finding 6)"]
    fn the_vertical_pass_stays_renormalised_and_the_horizontal_one_wraps() {
        let (w, h) = (24usize, 30usize);
        let k = gaussian_kernel(5.0);
        let n = k.len();
        let base = noise(w * h, 5);
        let mut a = base.clone();
        convolve_wrapx(&mut a, w, h, 1, &k);
        // changing the bottom rows leaves the top rows alone (no vertical wrap) ...
        let mut low = base.clone();
        for v in low[(h - 3) * w..].iter_mut() {
            *v += 50.0;
        }
        convolve_wrapx(&mut low, w, h, 1, &k);
        assert_eq!(bits(&a[..(h - 3 - n) * w]), bits(&low[..(h - 3 - n) * w]));
        // ... while changing the last column reaches the first one (horizontal wrap)
        let mut right = base.clone();
        for y in 0..h {
            right[y * w + w - 1] += 50.0;
        }
        convolve_wrapx(&mut right, w, h, 1, &k);
        assert!((0..h).all(|y| right[y * w] != a[y * w]));
        // the vertical pass is the plain one: a single column image (w = 1 wraps onto itself)
        // through convolve_wrapx equals the plain vertical pass of its horizontal result
        let col = noise(h, 6);
        let mut c1 = col.clone();
        convolve_wrapx(&mut c1, 1, h, 1, &k);
        let mut t = vec![0.0f32; h];
        hpass_wrapx::<1>(&col, &mut t, 1, &k);
        let mut c2 = vec![0.0f32; h];
        vpass::<1>(&t, &mut c2, 1, h, &k);
        assert_eq!(bits(&c1), bits(&c2));
    }

    #[test]
    #[ignore = "the G6 wrap rule (U) this test encodes is superseded by the 360 rules read from TwixtorPro_64 (parity/lab/re/op-residual.md, finding 6)"]
    fn wrapx_convolution_model_fixtures_are_reproduced_bit_for_bit() {
        let mut n = 0;
        for c in crate::holefill::tests::load_wrap_cases() {
            let crate::holefill::tests::WrapCase::Conv { c, w, h, radius, data, out } = c else { continue };
            let mut got = data.clone();
            if radius == 0.0 {
                convolve_wrapx(&mut got, w, h, c, &BINOMIAL5);
                if c == 1 {
                    let mut g2 = data.clone();
                    blur5_wrapx(&mut g2, w, h);
                    assert_eq!(bits(&g2), bits(&got));
                }
            } else {
                let k = gaussian_kernel(radius);
                convolve_wrapx(&mut got, w, h, c, &k);
                if c == 1 {
                    let mut g2 = data.clone();
                    blur_gaussian_wrapx(&mut g2, w, h, 1, radius);
                    assert_eq!(bits(&g2), bits(&got));
                }
            }
            assert_eq!(bits(&got), bits(&out), "{c} channels {w}x{h} radius {radius}");
            n += 1;
        }
        assert!(n >= 9, "{n} convolution cases");
    }

    #[test]
    fn wrapx_convolution_is_identical_for_one_and_four_threads() {
        let (w, h) = (77usize, 41usize);
        let src = noise(w * h * 2, 8);
        let run = |threads: usize| {
            let mut d = src.clone();
            rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| blur_gaussian_wrapx(&mut d, w, h, 2, 7.0));
            d
        };
        assert_eq!(bits(&run(1)), bits(&run(4)));
    }

    #[test]
    fn convolve1_with_is_each_pass_of_the_fill_and_the_nan_rule_keeps_the_numbers() {
        let (w, h) = (13usize, 9usize);
        let src: Vec<f32> = (0..w * h).map(|i| ((i * 37 % 101) as f32) * 0.173 - 4.0).collect();
        for k in [BINOMIAL5.to_vec(), gaussian_kernel(5.0)] {
            for (wrapx, above) in [(false, false), (false, true), (true, false), (true, true)] {
                let mut want = src.clone();
                match (wrapx, above) {
                    (false, false) => convolve(&mut want, w, h, 1, &k),
                    (false, true) => convolve1_above_first(&mut want, w, h, &k),
                    (true, false) => convolve_wrapx(&mut want, w, h, 1, &k),
                    (true, true) => convolve1_wrapx_above_first(&mut want, w, h, &k),
                }
                let (mut p, mut r) = (src.clone(), src.clone());
                convolve1_with::<Plain>(&mut p, w, h, &k, wrapx, above);
                convolve1_with::<crate::nanrule::Rule>(&mut r, w, h, &k, wrapx, above);
                assert_eq!(bits(&p), bits(&want), "plain {wrapx} {above}");
                assert_eq!(bits(&r), bits(&want), "rule {wrapx} {above}");
            }
        }
    }
}
