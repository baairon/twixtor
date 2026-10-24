//! Image mixing primitives of the engine: [`blend`], [`lerp`], [`copy`] and the Weighted Blend
//! without motion vectors. All functions work in the engine domain (premultiplied) and require
//! equal size and format of their images; a mismatch is a programming error and panics.
//!
//! Numerics are single precision in the order written, with no fused multiply add and a truncating
//! conversion that stores the low byte or word without clamping. Rows are independent, so they run
//! in parallel with `fx_core::par_rows` and the result does not depend on the thread count.

use crate::image::{Image, Samples, CHANNELS};
use fx_core::par_rows;

/// Snap tolerance of [`blend`]: a weight closer than this to 0 or to 1 gives a plain copy.
const SNAP: f64 = 0.005;

/// Truncation toward zero; NaN and values outside the 64 bit range give `i64::MIN`. The row
/// kernels store the low byte or word of this integer (`as u8` / `as u16`).
#[inline(always)]
pub(crate) fn cvtt(x: f32) -> i64 {
    const TWO63: f32 = 9_223_372_036_854_775_808.0;
    if x.is_nan() || x >= TWO63 || x < -TWO63 {
        i64::MIN
    } else {
        x as i64
    }
}

fn assert_shape2(out: &Image, other: &Image) {
    assert!(
        out.same_shape(other),
        "image primitives need equal size and format: {}x{} {:?} against {}x{} {:?}",
        out.width,
        out.height,
        out.fmt(),
        other.width,
        other.height,
        other.fmt()
    );
}

/// `blend(out, A, B, w)`: the mix of two frames, including the 0.005 snap to a plain copy and the
/// extrapolation branch for `w` outside 0..=1.
///
/// Branches, in order:
/// 1. `0.005 > |w|` (compared in double): `out = copy(A)`.
/// 2. `0.005 > |(double)w - 1.0|`: `out = copy(B)`.
/// 3. `w < 0.0 || w > 1.0` (single precision compares): `out = copy(A)`, then
///    `lerp(out, B, (1.0f64 - w as f64) as f32)` (the weight is narrowed from double), which is
///    `A + (B - A) * w` with the alpha normalisation of [`lerp`].
/// 4. otherwise per sample (all four channels incl. alpha, same formula), in single precision
///    in exactly this order (subtract, multiply, add, add): u8 and u16
///    `out = trunc(((B - A) * w + A) + 0.5f32)` (low byte or word of the 64 bit integer, no
///    clamping), f32 `out = ((B - A) * w + A) + 0.0f32` (the `+ 0.0` turns -0.0 into +0.0).
///
/// A NaN weight fails every comparison and reaches branch 4.
pub fn blend(out: &mut Image, a: &Image, b: &Image, w: f32) {
    assert_shape2(out, a);
    assert_shape2(out, b);
    if SNAP > f64::from(w.abs()) {
        copy(out, a);
        return;
    }
    if SNAP > (f64::from(w) - 1.0).abs() {
        copy(out, b);
        return;
    }
    if w < 0.0 || w > 1.0 {
        copy(out, a);
        lerp(out, b, (1.0f64 - f64::from(w)) as f32);
        return;
    }
    blend_rows(out, a, b, w);
}

/// The common branch of `blend`: the per sample mix of every row.
fn blend_rows(out: &mut Image, a: &Image, b: &Image, w: f32) {
    let row_len = out.row_len();
    if row_len == 0 {
        return;
    }
    match (&mut out.samples, &a.samples, &b.samples) {
        (Samples::U8(o), Samples::U8(a), Samples::U8(b)) => par_rows(o, row_len, |y, row| {
            let ar = &a[y * row_len..(y + 1) * row_len];
            let br = &b[y * row_len..(y + 1) * row_len];
            for i in 0..row_len {
                let af = ar[i] as f32;
                let bf = br[i] as f32;
                row[i] = cvtt((bf - af) * w + af + 0.5f32) as u8;
            }
        }),
        (Samples::U16(o), Samples::U16(a), Samples::U16(b)) => par_rows(o, row_len, |y, row| {
            let ar = &a[y * row_len..(y + 1) * row_len];
            let br = &b[y * row_len..(y + 1) * row_len];
            for i in 0..row_len {
                let af = ar[i] as f32;
                let bf = br[i] as f32;
                row[i] = cvtt((bf - af) * w + af + 0.5f32) as u16;
            }
        }),
        (Samples::F32(o), Samples::F32(a), Samples::F32(b)) => par_rows(o, row_len, |y, row| {
            let ar = &a[y * row_len..(y + 1) * row_len];
            let br = &b[y * row_len..(y + 1) * row_len];
            for i in 0..row_len {
                row[i] = (br[i] - ar[i]) * w + ar[i] + 0.0f32;
            }
        }),
        _ => unreachable!("shapes were checked"),
    }
}

/// Plain copy of the full image `src` into `out`.
pub fn copy(out: &mut Image, src: &Image) {
    assert_shape2(out, src);
    match (&mut out.samples, &src.samples) {
        (Samples::U8(o), Samples::U8(s)) => o.copy_from_slice(s),
        (Samples::U16(o), Samples::U16(s)) => o.copy_from_slice(s),
        (Samples::F32(o), Samples::F32(s)) => o.copy_from_slice(s),
        _ => unreachable!("shapes were checked"),
    }
}

/// Linear interpolation in place: `first = second + (first - second) * w`, with the weight on the
/// FIRST image, alpha normalised for premultiplied data and a pixel with non positive alpha
/// cleared.
///
/// `w == 1.0` leaves `first` unchanged and `w == 0.0` copies `second` into `first`; a NaN weight
/// fails both comparisons and reaches the row kernels (and clears every pixel there, because its
/// alpha is NaN). Row kernels for the 4 channel images:
///
/// * f32: `a = (f.a - s.a) * w + s.a`; if not `a > 0.0` the pixel becomes zero;
///   else `c = (f.c - s.c) * w + s.c` for r, g, b; if `a > 1.0` then `c = c / a` and `a = 1.0`;
///   the pixel is stored as `[a + 0.0, r + 0.0, g + 0.0, b + 0.0]`.
/// * u8 and u16: the same in the sample scale. With `S = 255.0` or `65535.0` (single precision):
///   `a = (f.a - s.a) as f32 * w + s.a as f32` (the difference is taken in integers),
///   `n = a / S`; if not `n > 0.0` the pixel becomes zero; else each colour
///   `c = (f.c - s.c) as f32 * w + s.c as f32`; if `n > 1.0` then `c = c / n` and `a = S`;
///   each colour is clamped to `0..=a` (`c > a` gives `a`, else `0.0 > c` gives `0.0`), and every
///   channel (also `a`) is stored as the low byte or word of `trunc(x + 0.5f32)`.
pub fn lerp(first: &mut Image, second: &Image, w: f32) {
    assert_shape2(first, second);
    if w == 1.0 {
        return;
    }
    if w == 0.0 {
        copy(first, second);
        return;
    }
    let row_len = first.row_len();
    if row_len == 0 {
        return;
    }
    match (&mut first.samples, &second.samples) {
        (Samples::U8(f), Samples::U8(s)) => par_rows(f, row_len, |y, row| {
            let sr = &s[y * row_len..(y + 1) * row_len];
            for (fp, sp) in row.chunks_exact_mut(CHANNELS).zip(sr.chunks_exact(CHANNELS)) {
                lerp_pixel_int(fp, sp, w, 255.0f32, |v| cvtt(v) as u8);
            }
        }),
        (Samples::U16(f), Samples::U16(s)) => par_rows(f, row_len, |y, row| {
            let sr = &s[y * row_len..(y + 1) * row_len];
            for (fp, sp) in row.chunks_exact_mut(CHANNELS).zip(sr.chunks_exact(CHANNELS)) {
                lerp_pixel_int(fp, sp, w, 65535.0f32, |v| cvtt(v) as u16);
            }
        }),
        (Samples::F32(f), Samples::F32(s)) => par_rows(f, row_len, |y, row| {
            let sr = &s[y * row_len..(y + 1) * row_len];
            for (fp, sp) in row.chunks_exact_mut(CHANNELS).zip(sr.chunks_exact(CHANNELS)) {
                lerp_pixel_f32(fp, sp, w);
            }
        }),
        _ => unreachable!("shapes were checked"),
    }
}

/// One pixel of the f32 case of [`lerp`].
#[inline(always)]
fn lerp_pixel_f32(f: &mut [f32], s: &[f32], w: f32) {
    let mut a = (f[0] - s[0]) * w + s[0];
    if !(a > 0.0) {
        f.fill(0.0);
        return;
    }
    let mut r = (f[1] - s[1]) * w + s[1];
    let mut g = (f[2] - s[2]) * w + s[2];
    let mut b = (f[3] - s[3]) * w + s[3];
    if a > 1.0 {
        r = r / a;
        g = g / a;
        b = b / a;
        a = 1.0;
    }
    f[1] = r + 0.0;
    f[2] = g + 0.0;
    f[3] = b + 0.0;
    f[0] = a + 0.0;
}

/// One pixel of the integer cases of [`lerp`] (u8 with `scale = 255.0`, u16 with
/// `scale = 65535.0`).
#[inline(always)]
fn lerp_pixel_int<T: Copy + Into<i32>>(
    f: &mut [T],
    s: &[T],
    w: f32,
    scale: f32,
    store: impl Fn(f32) -> T,
) {
    let int = |v: T| -> i32 { v.into() };
    let diff = |i: usize| -> f32 { (int(f[i]) - int(s[i])) as f32 };
    let sf = |i: usize| -> f32 { int(s[i]) as f32 };
    let mut a = diff(0) * w + sf(0);
    let n = a / scale;
    if !(n > 0.0) {
        let zero = store(0.0);
        f.fill(zero);
        return;
    }
    let mut r = diff(1) * w + sf(1);
    let mut g = diff(2) * w + sf(2);
    let mut b = diff(3) * w + sf(3);
    if n > 1.0 {
        r = r / n;
        g = g / n;
        b = b / n;
        a = scale;
    }
    let clamp = |c: f32| -> f32 {
        if c > a {
            a
        } else if 0.0 > c {
            0.0
        } else {
            c
        }
    };
    let (r, g, b) = (clamp(r), clamp(g), clamp(b));
    f[1] = store(r + 0.5f32);
    f[2] = store(g + 0.5f32);
    f[3] = store(b + 0.5f32);
    f[0] = store(a + 0.5f32);
}

/// Weighted Blend without motion vectors. `wa` is the weight of `a`; the result is
/// `blend(a', b', 1 - wa)` where `a'` and `b'` are the frames after the sharpening step
/// (`filters::sharpen_step`).
///
/// Without flows the warped frame is a plain copy of the input, so the difference mask is empty
/// (`(|dR| + |dG|) + |dB| = 0` or NaN never reaches the positive limit), the clean-up keeps a zero
/// mask zero, the inverted mask is 255 everywhere and the masked composite keeps its destination
/// (mask 255 is the weight of the destination, see `filters::composite_masked`). The blurred and
/// sharpened copy is therefore computed and discarded: the result is exactly `blend(a, b, 1 - wa)`,
/// a plain Blend with no unsharp mask.
///
/// Two details keep that exact: the float composite ends with `+ 0.0f`, which turns a -0.0 of the
/// frame into +0.0 (applied here to the inputs; it matters for the snap copies of `blend`), and a
/// non finite sample would spread NaN through the discarded copy into the result (`0 * NaN`). Float
/// images with a non finite sample or a magnitude above 1e30 (where the blur could overflow)
/// therefore take `weighted_blend_noflow_reference`, the full sequence. 8 and 16 bit images are
/// always exact.
///
/// `1 - wa` is evaluated in single precision.
pub fn weighted_blend_noflow(out: &mut Image, a: &Image, wa: f32, b: &Image) {
    assert_shape2(out, a);
    assert_shape2(out, b);
    let t = 1.0f32 - wa;
    match (&a.samples, &b.samples) {
        (Samples::F32(da), Samples::F32(db)) => {
            if !(sound_float(da) && sound_float(db)) {
                weighted_blend_noflow_reference(out, a, wa, b);
                return;
            }
            let plus_zero = |d: &Vec<f32>| -> Image {
                Image { width: a.width, height: a.height, samples: Samples::F32(d.iter().map(|v| *v + 0.0f32).collect()) }
            };
            let (a1, b1) = (plus_zero(da), plus_zero(db));
            blend(out, &a1, &b1, t);
        }
        _ => blend(out, a, b, t),
    }
}

/// True when the float samples are finite and small enough that the discarded sharpened copy of
/// `weighted_blend_noflow` cannot produce a non finite value.
fn sound_float(d: &[f32]) -> bool {
    d.iter().all(|v| v.abs() <= 1.0e30)
}

/// The full sequence of Weighted Blend without flows, with every filter run: copy of `a`,
/// sharpening step with `t = 1 - wa`, copy of `b`, sharpening step with `t = wa`,
/// `blend(out, a', b', 1 - wa)`. Slow (two blurs and mask passes per call);
/// `weighted_blend_noflow` returns the same result without them whenever that is provable.
/// Public as the reference for tests and for non finite float data.
pub fn weighted_blend_noflow_reference(out: &mut Image, a: &Image, wa: f32, b: &Image) {
    assert_shape2(out, a);
    assert_shape2(out, b);
    let t = 1.0f32 - wa;
    let mut a1 = a.clone();
    crate::filters::sharpen_step(&mut a1, a, t);
    let mut b1 = b.clone();
    crate::filters::sharpen_step(&mut b1, b, wa);
    blend(out, &a1, &b1, t);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Fmt, Image, Samples};

    // ---------------------------------------------------------------- helpers

    /// xorshift64*, deterministic test data.
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

    /// 64 pixels x 256 rows: sample `a * 256 + b` holds `a` in the first image and `b` in the
    /// second, so every (A, B) pair of bytes occurs exactly once.
    fn all_u8_pairs() -> (Image, Image) {
        let a: Vec<u8> = (0..65536usize).map(|i| (i / 256) as u8).collect();
        let b: Vec<u8> = (0..65536usize).map(|i| (i % 256) as u8).collect();
        (u8_image(64, 256, a), u8_image(64, 256, b))
    }

    // ----------------------------------------------- independent oracles
    // Written again from the formulas of ENGINE-timeline section 7 and ENGINE-warp section 5,
    // with explicit temporaries and `trunc` instead of the shared helper.

    /// 8 and 16 bpc formula for one sample, returned as the truncated 64 bit integer.
    fn oracle_mix_int(a: i64, b: i64, w: f32) -> i64 {
        let af = a as f32;
        let bf = b as f32;
        let d: f32 = bf - af;
        let m: f32 = d * w;
        let s: f32 = m + af;
        let r: f32 = s + 0.5f32;
        r.trunc() as i64
    }

    /// Integer `lerp` kernel for one pixel, returning the truncated integers (A, R, G, B).
    fn oracle_lerp_px_int(f: [i64; 4], s: [i64; 4], w: f32, scale: f32) -> [i64; 4] {
        let mix = |i: usize| -> f32 { ((f[i] - s[i]) as f32) * w + (s[i] as f32) };
        let alpha = mix(0);
        let norm = alpha / scale;
        if norm <= 0.0 || norm.is_nan() {
            return [0; 4];
        }
        let (mut r, mut g, mut b) = (mix(1), mix(2), mix(3));
        let mut a = alpha;
        if norm > 1.0 {
            r = r / norm;
            g = g / norm;
            b = b / norm;
            a = scale;
        }
        let lim = |c: f32| -> f32 {
            let c = if c > a { a } else { c };
            if c < 0.0 {
                0.0
            } else {
                c
            }
        };
        let t = |c: f32| -> i64 { (c + 0.5f32).trunc() as i64 };
        [t(a), t(lim(r)), t(lim(g)), t(lim(b))]
    }

    /// Float `lerp` kernel for one pixel.
    fn oracle_lerp_px_f32(f: [f32; 4], s: [f32; 4], w: f32) -> [f32; 4] {
        let mix = |i: usize| -> f32 { (f[i] - s[i]) * w + s[i] };
        let alpha = mix(0);
        if alpha <= 0.0 || alpha.is_nan() {
            return [0.0; 4];
        }
        let (mut r, mut g, mut b, mut a) = (mix(1), mix(2), mix(3), alpha);
        if a > 1.0 {
            r /= a;
            g /= a;
            b /= a;
            a = 1.0;
        }
        [a + 0.0, r + 0.0, g + 0.0, b + 0.0]
    }

    /// Whole image blend of two u8 images, branch by branch from the document.
    fn oracle_blend_u8(a: &[u8], b: &[u8], w: f32) -> Vec<u8> {
        if (w.abs() as f64) < 0.005 {
            return a.to_vec();
        }
        if ((w as f64) - 1.0).abs() < 0.005 {
            return b.to_vec();
        }
        if w < 0.0 || w > 1.0 {
            // copy of A, then lerp(out, B, 1 - w) with the weight narrowed from double
            let wl = (1.0f64 - w as f64) as f32;
            let mut out = Vec::with_capacity(a.len());
            for (pa, pb) in a.chunks(4).zip(b.chunks(4)) {
                let f = [pa[0] as i64, pa[1] as i64, pa[2] as i64, pa[3] as i64];
                let s = [pb[0] as i64, pb[1] as i64, pb[2] as i64, pb[3] as i64];
                // w == 1.0 leaves the copy of A untouched, w == 0.0 copies B
                let px = if wl == 1.0 {
                    f
                } else if wl == 0.0 {
                    s
                } else {
                    oracle_lerp_px_int(f, s, wl, 255.0)
                };
                out.extend(px.iter().map(|v| *v as u8));
            }
            return out;
        }
        a.iter().zip(b).map(|(x, y)| oracle_mix_int(*x as i64, *y as i64, w) as u8).collect()
    }

    const WEIGHTS: [f32; 14] =
        [0.0, 0.004, 0.005, 0.25, 0.5, 0.996, 1.0, 0.1, 0.3, 0.7, 0.99, 0.995, 0.00501, 0.9949];

    // ------------------------------------------------------------- blend: u8

    #[test]
    fn blend_u8_hand_computed_values() {
        // (B - A) * w + A + 0.5, truncated; every row is "A B w expected".
        let cases: [(u8, u8, f32, u8); 8] = [
            (10, 20, 0.5, 15),     // 5 + 10 + 0.5 = 15.5
            (0, 255, 0.5, 128),    // 127.5 + 0 + 0.5 = 128.0
            (255, 0, 0.25, 191),   // -63.75 + 255 + 0.5 = 191.75
            (7, 8, 0.5, 8),        // 0.5 + 7 + 0.5 = 8.0, half rounds up
            (8, 7, 0.5, 8),        // -0.5 + 8 + 0.5 = 8.0
            (0, 1, 0.25, 0),       // 0.25 + 0 + 0.5 = 0.75
            (0, 1, 0.75, 1),       // 0.75 + 0 + 0.5 = 1.25
            (200, 100, 0.75, 125), // -75 + 200 + 0.5 = 125.5
        ];
        for (a, b, w, want) in cases {
            let ia = u8_image(1, 1, vec![a; 4]);
            let ib = u8_image(1, 1, vec![b; 4]);
            let mut out = Image::new(1, 1, Fmt::U8);
            blend(&mut out, &ia, &ib, w);
            assert_eq!(u8s(&out), &[want; 4], "A={a} B={b} w={w}");
        }
    }

    #[test]
    fn blend_u8_every_pair_matches_the_oracle_for_all_weights() {
        let (a, b) = all_u8_pairs();
        for w in WEIGHTS {
            let mut out = Image::new(64, 256, Fmt::U8);
            blend(&mut out, &a, &b, w);
            assert_eq!(u8s(&out), &oracle_blend_u8(u8s(&a), u8s(&b), w)[..], "w={w}");
        }
    }

    #[test]
    fn blend_u8_weights_beside_the_snaps() {
        let (a, b) = all_u8_pairs();
        // f32(0.005) is 0.00499999988..., below the double 0.005, so it snaps to A; the next f32
        // up does not. The same on the B side, where the test is |(double)w - 1| < 0.005.
        let w_low = 0.005f32;
        assert!((w_low as f64) < 0.005);
        let w_low_next = f32::from_bits(w_low.to_bits() + 1);
        assert!((w_low_next as f64) > 0.005);
        let w_high = 0.995f32; // 0.99500000476..., distance to 1 is 0.004999995 < 0.005
        assert!(((w_high as f64) - 1.0).abs() < 0.005);
        let w_outside = f32::from_bits(w_high.to_bits() - 4);
        assert!(((w_outside as f64) - 1.0).abs() >= 0.005);

        let mut out = Image::new(64, 256, Fmt::U8);
        blend(&mut out, &a, &b, w_low);
        assert_eq!(out, a, "0.005f32 is a copy of A");
        blend(&mut out, &a, &b, -w_low);
        assert_eq!(out, a, "-0.005f32 is a copy of A (|w| is tested)");
        blend(&mut out, &a, &b, w_low_next);
        assert_ne!(out, a, "the f32 after 0.005 is a real blend");
        assert_eq!(u8s(&out), &oracle_blend_u8(u8s(&a), u8s(&b), w_low_next)[..]);
        blend(&mut out, &a, &b, w_high);
        assert_eq!(out, b, "0.995f32 is a copy of B");
        blend(&mut out, &a, &b, 1.004);
        assert_eq!(out, b, "1.004 is a copy of B, not an extrapolation");
        blend(&mut out, &a, &b, -0.004);
        assert_eq!(out, a, "-0.004 is a copy of A, not an extrapolation");
        blend(&mut out, &a, &b, w_outside);
        assert_eq!(u8s(&out), &oracle_blend_u8(u8s(&a), u8s(&b), w_outside)[..]);
        assert_ne!(out, b);
    }

    #[test]
    fn blend_u8_extrapolation_goes_through_lerp() {
        // Pixels: alpha takes every (A, B) pair once (65536 pixels), the colours follow
        // deterministic functions of the pair so that clamping and normalisation are reached.
        let n = 65536usize;
        let mut da = Vec::with_capacity(n * 4);
        let mut db = Vec::with_capacity(n * 4);
        for p in 0..n {
            let (x, y) = ((p / 256) as u32, (p % 256) as u32);
            da.extend_from_slice(&[x as u8, ((x * 7 + y) & 255) as u8, ((x + 3 * y) & 255) as u8, (y ^ x) as u8]);
            db.extend_from_slice(&[y as u8, ((y * 5 + x) & 255) as u8, (255 - x) as u8, ((x * y) & 255) as u8]);
        }
        let a = u8_image(256, 256, da);
        let b = u8_image(256, 256, db);
        for w in [-0.5f32, 1.5, -2.0, 3.0, -0.0051, 1.0051, -0.01, 1.7] {
            let mut out = Image::new(256, 256, Fmt::U8);
            blend(&mut out, &a, &b, w);
            assert_eq!(u8s(&out), &oracle_blend_u8(u8s(&a), u8s(&b), w)[..], "w={w}");
        }
        // The same call spelt out: copy then lerp with the narrowed weight.
        let mut expect = Image::new(256, 256, Fmt::U8);
        copy(&mut expect, &a);
        lerp(&mut expect, &b, -0.5);
        let mut out = Image::new(256, 256, Fmt::U8);
        blend(&mut out, &a, &b, 1.5);
        assert_eq!(out, expect);
    }

    #[test]
    fn blend_u8_extrapolation_covers_every_colour_pair_for_opaque_pixels() {
        // Alpha 255 in both images (no normalisation), so the colour lanes are plain clamped mixes:
        // red takes every (A, B) pair, green the transposed pair, blue a rotated one.
        let n = 65536usize;
        let mut da = Vec::with_capacity(n * 4);
        let mut db = Vec::with_capacity(n * 4);
        for p in 0..n {
            let (x, y) = ((p / 256) as u8, (p % 256) as u8);
            da.extend_from_slice(&[255, x, y, x.wrapping_add(y)]);
            db.extend_from_slice(&[255, y, x, y.wrapping_sub(x)]);
        }
        let a = u8_image(256, 256, da);
        let b = u8_image(256, 256, db);
        for w in [-0.5f32, 1.5, -2.0, 3.0, 1.25, -0.125] {
            let mut out = Image::new(256, 256, Fmt::U8);
            blend(&mut out, &a, &b, w);
            assert_eq!(u8s(&out), &oracle_blend_u8(u8s(&a), u8s(&b), w)[..], "w={w}");
            // ENGINE-timeline section 7: the extrapolation is mathematically A + (B - A) * w, here
            // clamped to 0..=255 by the lerp; the 8 bit result is within one step of the real value.
            for (i, px) in u8s(&out).chunks(4).enumerate() {
                for c in 1..4 {
                    let (va, vb) = (u8s(&a)[i * 4 + c] as f64, u8s(&b)[i * 4 + c] as f64);
                    let real = (va + (vb - va) * f64::from(w)).clamp(0.0, 255.0);
                    assert!((px[c] as f64 - real).abs() <= 1.0, "w={w} pixel {i} lane {c}: {} vs {real}", px[c]);
                }
                assert_eq!(px[0], 255);
            }
        }
    }

    #[test]
    fn blend_u8_extrapolation_golden() {
        // Opaque, w = 1.5 -> lerp weight -0.5: (A - B) * -0.5 + B = 1.5 B - 0.5 A.
        let a = u8_image(1, 1, vec![255, 100, 40, 0]);
        let b = u8_image(1, 1, vec![255, 120, 50, 8]);
        let mut out = Image::new(1, 1, Fmt::U8);
        blend(&mut out, &a, &b, 1.5);
        // alpha: (0) * -0.5 + 255 = 255, n = 1.0 so no normalisation;
        // r: (100-120) * -0.5 + 120 = 130; g: (40-50) * -0.5 + 50 = 55; b: (0-8) * -0.5 + 8 = 12
        assert_eq!(u8s(&out), &[255, 130, 55, 12]);
        // w = -0.5 -> lerp weight 1.5: (A - B) * 1.5 + B = A + (B - A) * -0.5
        blend(&mut out, &a, &b, -0.5);
        // r: (100-120) * 1.5 + 120 = 90; g: (40-50) * 1.5 + 50 = 35; b: (0-8) * 1.5 + 8 = -4 -> 0
        assert_eq!(u8s(&out), &[255, 90, 35, 0]);
    }

    // ------------------------------------------------------------ blend: u16

    #[test]
    fn blend_u16_boundaries_and_random_match_the_oracle() {
        let edge: [u16; 10] = [0, 1, 2, 255, 256, 32767, 32768, 32769, 65534, 65535];
        let mut da = Vec::new();
        let mut db = Vec::new();
        for &x in &edge {
            for &y in &edge {
                da.push(x);
                db.push(y);
            }
        }
        let mut rng = Rng(0x1234_5678_9abc_def1);
        for _ in 0..20_000 {
            da.push((rng.next() >> 20) as u16);
            db.push((rng.next() >> 20) as u16);
        }
        while da.len() % 4 != 0 {
            da.push(0);
            db.push(65535);
        }
        let px = da.len() / 4;
        let a = u16_image(px, 1, da.clone());
        let b = u16_image(px, 1, db.clone());
        for w in WEIGHTS.iter().copied().filter(|w| *w > 0.005 && *w < 0.995) {
            let mut out = Image::new(px, 1, Fmt::U16);
            blend(&mut out, &a, &b, w);
            let want: Vec<u16> = da
                .iter()
                .zip(&db)
                .map(|(x, y)| oracle_mix_int(*x as i64, *y as i64, w) as u16)
                .collect();
            assert_eq!(u16s(&out), &want[..], "w={w}");
        }
        for (w, src) in [(0.0f32, &a), (0.004, &a), (0.005, &a), (0.996, &b), (1.0, &b)] {
            let mut out = Image::new(px, 1, Fmt::U16);
            blend(&mut out, &a, &b, w);
            assert_eq!(&out, src, "w={w}");
        }
    }

    #[test]
    fn blend_u16_hand_computed_values() {
        let cases: [(u16, u16, f32, u16); 5] = [
            (0, 65535, 0.5, 32768),     // 32767.5 + 0.5 = 32768.0
            (65535, 0, 0.25, 49151),    // -16383.75 + 65535 = 49151.25, + 0.5 = 49151.75
            (1000, 2000, 0.5, 1500),    // 500 + 1000 + 0.5 = 1500.5
            (32768, 32769, 0.5, 32769), // 0.5 + 32768 + 0.5 = 32769.0
            (32769, 32768, 0.5, 32769), // -0.5 + 32769 + 0.5 = 32769.0
        ];
        for (a, b, w, want) in cases {
            let ia = u16_image(1, 1, vec![a; 4]);
            let ib = u16_image(1, 1, vec![b; 4]);
            let mut out = Image::new(1, 1, Fmt::U16);
            blend(&mut out, &ia, &ib, w);
            assert_eq!(u16s(&out), &[want; 4], "A={a} B={b} w={w}");
        }
    }

    #[test]
    fn blend_u16_extrapolation_matches_the_lerp_oracle() {
        let mut rng = Rng(77);
        let n = 5000usize;
        let mut da = Vec::new();
        let mut db = Vec::new();
        for _ in 0..n {
            // premultiplied-looking data: colour at most alpha for most pixels
            let aa = (rng.next() >> 48) as u16;
            let ab = (rng.next() >> 48) as u16;
            da.extend_from_slice(&[aa, ((rng.next() >> 48) as u16).min(aa), (rng.next() >> 48) as u16, 0]);
            db.extend_from_slice(&[ab, ((rng.next() >> 48) as u16).min(ab), (rng.next() >> 48) as u16, 65535]);
        }
        let a = u16_image(n, 1, da.clone());
        let b = u16_image(n, 1, db.clone());
        for w in [-0.5f32, 1.5, -3.0, 2.25] {
            let mut out = Image::new(n, 1, Fmt::U16);
            blend(&mut out, &a, &b, w);
            let wl = (1.0f64 - w as f64) as f32;
            let mut want = Vec::new();
            for (pa, pb) in da.chunks(4).zip(db.chunks(4)) {
                let f = [pa[0] as i64, pa[1] as i64, pa[2] as i64, pa[3] as i64];
                let s = [pb[0] as i64, pb[1] as i64, pb[2] as i64, pb[3] as i64];
                want.extend(oracle_lerp_px_int(f, s, wl, 65535.0).iter().map(|v| *v as u16));
            }
            assert_eq!(u16s(&out), &want[..], "w={w}");
        }
    }

    // ------------------------------------------------------------ blend: f32

    #[test]
    fn blend_f32_exact_bits() {
        let a = f32_image(1, 2, vec![0.25, 0.5, 1.0, 0.0, -0.5, 2.0, 0.125, 8.0]);
        let b = f32_image(1, 2, vec![0.75, 0.0, 0.0, 1.0, 0.5, 3.0, 0.375, -8.0]);
        let mut out = Image::new(1, 2, Fmt::F32);
        blend(&mut out, &a, &b, 0.5);
        // (B - A) * 0.5 + A, all exact in binary
        let want = [0.5f32, 0.25, 0.5, 0.5, 0.0, 2.5, 0.25, 0.0];
        assert_eq!(bits(&out), want.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        blend(&mut out, &a, &b, 0.25);
        let want = [0.375f32, 0.375, 0.75, 0.25, -0.25, 2.25, 0.1875, 4.0];
        assert_eq!(bits(&out), want.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }

    #[test]
    fn blend_f32_adds_zero_to_remove_negative_zero() {
        // A = -0.0, B = -1.4e-45 (smallest denormal, negative), w = 0.25:
        // (B - A) = -1.4e-45, times 0.25 rounds to -0.0, plus A (-0.0) is -0.0, plus 0.0 is +0.0.
        let tiny = f32::from_bits(0x8000_0001);
        let a = f32_image(1, 1, vec![-0.0, -0.0, -0.0, -0.0]);
        let b = f32_image(1, 1, vec![tiny; 4]);
        let mut out = Image::new(1, 1, Fmt::F32);
        blend(&mut out, &a, &b, 0.25);
        assert_eq!(bits(&out), vec![0u32; 4]);
        // The snaps are plain copies and keep the -0.0.
        blend(&mut out, &a, &b, 0.0);
        assert_eq!(bits(&out), vec![0x8000_0000u32; 4]);
    }

    #[test]
    fn blend_f32_random_matches_the_oracle_bitwise() {
        let mut rng = Rng(0xdead_beef_cafe_f00d);
        let n = 3000usize;
        let da: Vec<f32> = (0..n * 4).map(|_| rng.unit() * 4.0 - 1.5).collect();
        let db: Vec<f32> = (0..n * 4).map(|_| rng.unit() * 4.0 - 1.5).collect();
        let a = f32_image(n, 1, da.clone());
        let b = f32_image(n, 1, db.clone());
        for w in WEIGHTS.iter().copied().filter(|w| *w > 0.005 && *w < 0.995) {
            let mut out = Image::new(n, 1, Fmt::F32);
            blend(&mut out, &a, &b, w);
            let want: Vec<u32> = da
                .iter()
                .zip(&db)
                .map(|(x, y)| {
                    let d: f32 = *y - *x;
                    let m: f32 = d * w;
                    let s: f32 = m + *x;
                    (s + 0.0f32).to_bits()
                })
                .collect();
            assert_eq!(bits(&out), want, "w={w}");
        }
    }

    #[test]
    fn blend_f32_extrapolation_matches_the_lerp_oracle() {
        let mut rng = Rng(5150);
        let n = 3000usize;
        let mut da = Vec::new();
        let mut db = Vec::new();
        for _ in 0..n {
            let (aa, ab) = (rng.unit(), rng.unit());
            da.extend_from_slice(&[aa, rng.unit() * aa, rng.unit(), rng.unit()]);
            db.extend_from_slice(&[ab, rng.unit() * ab, rng.unit(), rng.unit()]);
        }
        let a = f32_image(n, 1, da.clone());
        let b = f32_image(n, 1, db.clone());
        for w in [-0.5f32, 1.5, -2.0, 1.25] {
            let mut out = Image::new(n, 1, Fmt::F32);
            blend(&mut out, &a, &b, w);
            let wl = (1.0f64 - w as f64) as f32;
            let mut want: Vec<u32> = Vec::new();
            for (pa, pb) in da.chunks(4).zip(db.chunks(4)) {
                let px = oracle_lerp_px_f32([pa[0], pa[1], pa[2], pa[3]], [pb[0], pb[1], pb[2], pb[3]], wl);
                want.extend(px.iter().map(|v| v.to_bits()));
            }
            assert_eq!(bits(&out), want, "w={w}");
        }
    }

    // ------------------------------------------------------------------ lerp

    #[test]
    fn lerp_f32_golden_values() {
        // a = (0.5 - 1.0) * 0.5 + 1.0 = 0.75, r = (0.25 - 0.75) * 0.5 + 0.75 = 0.5,
        // g = (0.5 - 0.25) * 0.5 + 0.25 = 0.375, b = (1.0 - 0.0) * 0.5 + 0.0 = 0.5
        let mut first = f32_image(1, 1, vec![0.5, 0.25, 0.5, 1.0]);
        let second = f32_image(1, 1, vec![1.0, 0.75, 0.25, 0.0]);
        lerp(&mut first, &second, 0.5);
        assert_eq!(f32s(&first), &[0.75, 0.5, 0.375, 0.5]);

        // alpha above one is normalised: a = 2.0 -> colours divided by 2, alpha 1.0
        let mut first = f32_image(1, 1, vec![1.0, 1.0, 0.5, 0.0]);
        let second = f32_image(1, 1, vec![0.0; 4]);
        lerp(&mut first, &second, 2.0);
        assert_eq!(f32s(&first), &[1.0, 1.0, 0.5, 0.0]);

        // non positive alpha clears the whole pixel, colour included
        let mut first = f32_image(1, 1, vec![0.0, 0.7, 0.8, 0.9]);
        let second = f32_image(1, 1, vec![0.0, 0.1, 0.2, 0.3]);
        lerp(&mut first, &second, 0.5);
        assert_eq!(bits(&first), vec![0u32; 4]);
        let mut first = f32_image(1, 1, vec![0.1, 0.7, 0.8, 0.9]);
        let second = f32_image(1, 1, vec![0.9, 0.1, 0.2, 0.3]);
        lerp(&mut first, &second, 2.0); // alpha = (0.1 - 0.9) * 2 + 0.9 = -0.7
        assert_eq!(bits(&first), vec![0u32; 4]);
    }

    #[test]
    fn lerp_u8_golden_values() {
        // plain mix, no clamp: a = 255, r = 50, g = 25, b = 0, then + 0.5 truncated
        let mut first = u8_image(1, 1, vec![255, 100, 50, 0]);
        let second = u8_image(1, 1, vec![255, 0, 0, 0]);
        lerp(&mut first, &second, 0.5);
        assert_eq!(u8s(&first), &[255, 50, 25, 0]);

        // alpha above 255: a = 255 * 2 = 510, n = 2.0, colours divided by 2, alpha 255
        let mut first = u8_image(1, 1, vec![255, 100, 40, 0]);
        let second = u8_image(1, 1, vec![0, 0, 0, 0]);
        lerp(&mut first, &second, 2.0);
        assert_eq!(u8s(&first), &[255, 100, 40, 0]);

        // colour clamped to alpha: a = 100, r = 127.5 -> 100 (100.5 truncates to 100)
        let mut first = u8_image(1, 1, vec![100, 255, 0, 0]);
        let second = u8_image(1, 1, vec![100, 0, 0, 0]);
        lerp(&mut first, &second, 0.5);
        assert_eq!(u8s(&first), &[100, 100, 0, 0]);

        // zero alpha clears the pixel
        let mut first = u8_image(1, 1, vec![0, 200, 30, 5]);
        let second = u8_image(1, 1, vec![0, 100, 10, 5]);
        lerp(&mut first, &second, 0.5);
        assert_eq!(u8s(&first), &[0, 0, 0, 0]);

        // negative colour clamps to zero: b = (0 - 100) * 3 + 100 = -200
        let mut first = u8_image(1, 1, vec![200, 0, 0, 0]);
        let second = u8_image(1, 1, vec![200, 0, 0, 100]);
        lerp(&mut first, &second, 3.0);
        assert_eq!(u8s(&first), &[200, 0, 0, 0]);
    }

    #[test]
    fn lerp_u16_golden_values() {
        let mut first = u16_image(1, 1, vec![65535, 20000, 10000, 0]);
        let second = u16_image(1, 1, vec![65535, 0, 0, 0]);
        lerp(&mut first, &second, 0.5);
        assert_eq!(u16s(&first), &[65535, 10000, 5000, 0]);

        let mut first = u16_image(1, 1, vec![65535, 20000, 10000, 0]);
        let second = u16_image(1, 1, vec![0, 0, 0, 0]);
        lerp(&mut first, &second, 2.0); // a = 131070, n = 2.0
        assert_eq!(u16s(&first), &[65535, 20000, 10000, 0]);
    }

    #[test]
    fn lerp_weight_one_and_zero() {
        let mut rng = Rng(9);
        let da: Vec<u8> = (0..400).map(|_| (rng.next() >> 56) as u8).collect();
        let db: Vec<u8> = (0..400).map(|_| (rng.next() >> 56) as u8).collect();
        let first0 = u8_image(10, 10, da);
        let second = u8_image(10, 10, db);
        let mut first = first0.clone();
        lerp(&mut first, &second, 1.0);
        assert_eq!(first, first0, "w == 1 leaves the image unchanged, even where alpha is 0");
        lerp(&mut first, &second, 0.0);
        assert_eq!(first, second, "w == 0 copies the second image");
        let mut first = first0.clone();
        lerp(&mut first, &second, -0.0);
        assert_eq!(first, second, "-0.0 == 0.0");
    }

    #[test]
    fn lerp_nan_weight_clears_every_pixel() {
        let mut first = f32_image(2, 1, vec![0.5; 8]);
        let second = f32_image(2, 1, vec![0.25; 8]);
        lerp(&mut first, &second, f32::NAN);
        assert_eq!(bits(&first), vec![0u32; 8]);
        let mut first = u8_image(2, 1, vec![7; 8]);
        let second = u8_image(2, 1, vec![9; 8]);
        lerp(&mut first, &second, f32::NAN);
        assert_eq!(u8s(&first), &[0u8; 8]);
    }

    #[test]
    fn lerp_u8_exhaustive_alpha_pairs_match_the_oracle() {
        let n = 65536usize;
        let mut df = Vec::new();
        let mut ds = Vec::new();
        for p in 0..n {
            let (x, y) = ((p / 256) as u32, (p % 256) as u32);
            df.extend_from_slice(&[x as u8, ((x * 3 + y * 5) & 255) as u8, (y ^ 0x5a) as u8, (x.wrapping_mul(y) & 255) as u8]);
            ds.extend_from_slice(&[y as u8, ((x ^ y) & 255) as u8, ((x + y) & 255) as u8, (255 - y) as u8]);
        }
        let second = u8_image(256, 256, ds.clone());
        for w in [0.5f32, 0.25, 0.1, 0.9, -0.5, 1.5, 2.0, -1.0, 0.3, 0.999, 3.5] {
            let mut first = u8_image(256, 256, df.clone());
            lerp(&mut first, &second, w);
            let mut want = Vec::new();
            for (pf, ps) in df.chunks(4).zip(ds.chunks(4)) {
                let f = [pf[0] as i64, pf[1] as i64, pf[2] as i64, pf[3] as i64];
                let s = [ps[0] as i64, ps[1] as i64, ps[2] as i64, ps[3] as i64];
                want.extend(oracle_lerp_px_int(f, s, w, 255.0).iter().map(|v| *v as u8));
            }
            assert_eq!(u8s(&first), &want[..], "w={w}");
        }
    }

    #[test]
    fn lerp_u16_and_f32_random_match_the_oracles() {
        let mut rng = Rng(31337);
        let n = 4000usize;
        let d16f: Vec<u16> = (0..n * 4).map(|_| (rng.next() >> 48) as u16).collect();
        let d16s: Vec<u16> = (0..n * 4).map(|_| (rng.next() >> 48) as u16).collect();
        let dff: Vec<f32> = (0..n * 4).map(|_| rng.unit() * 2.0 - 0.25).collect();
        let dfs: Vec<f32> = (0..n * 4).map(|_| rng.unit() * 2.0 - 0.25).collect();
        for w in [0.5f32, 0.37, -0.5, 1.5, 2.5, 0.01, 0.99] {
            let mut first = u16_image(n, 1, d16f.clone());
            lerp(&mut first, &u16_image(n, 1, d16s.clone()), w);
            let mut want = Vec::new();
            for (pf, ps) in d16f.chunks(4).zip(d16s.chunks(4)) {
                let f = [pf[0] as i64, pf[1] as i64, pf[2] as i64, pf[3] as i64];
                let s = [ps[0] as i64, ps[1] as i64, ps[2] as i64, ps[3] as i64];
                want.extend(oracle_lerp_px_int(f, s, w, 65535.0).iter().map(|v| *v as u16));
            }
            assert_eq!(u16s(&first), &want[..], "u16 w={w}");

            let mut first = f32_image(n, 1, dff.clone());
            lerp(&mut first, &f32_image(n, 1, dfs.clone()), w);
            let mut want: Vec<u32> = Vec::new();
            for (pf, ps) in dff.chunks(4).zip(dfs.chunks(4)) {
                let px = oracle_lerp_px_f32([pf[0], pf[1], pf[2], pf[3]], [ps[0], ps[1], ps[2], ps[3]], w);
                want.extend(px.iter().map(|v| v.to_bits()));
            }
            assert_eq!(bits(&first), want, "f32 w={w}");
        }
    }

    #[test]
    fn lerp_is_blend_with_swapped_images_for_opaque_float_data() {
        // ENGINE-warp section 5: lerp(first, second, w) is second + (first - second) * w; for opaque
        // float pixels (alpha 1 in both images, so neither normalisation nor clearing happens)
        // this is blend(second, first, w) bit for bit: both evaluate (x - y) * w + y (+ 0.0).
        let mut rng = Rng(2024);
        let n = 2000usize;
        let mut da = Vec::new();
        let mut db = Vec::new();
        for _ in 0..n {
            da.extend_from_slice(&[1.0, rng.unit(), rng.unit(), rng.unit()]);
            db.extend_from_slice(&[1.0, rng.unit(), rng.unit(), rng.unit()]);
        }
        let first0 = f32_image(n, 1, da);
        let second = f32_image(n, 1, db);
        for w in [0.1f32, 0.3, 0.5, 0.77, 0.9] {
            let mut first = first0.clone();
            lerp(&mut first, &second, w);
            let mut out = Image::new(n, 1, Fmt::F32);
            blend(&mut out, &second, &first0, w);
            assert_eq!(bits(&first), bits(&out), "w={w}");
        }
    }

    // ------------------------------------------------------ copy and contracts

    #[test]
    fn copy_copies_every_format() {
        let a = u8_image(2, 2, (0..16).collect());
        let mut o = Image::new(2, 2, Fmt::U8);
        copy(&mut o, &a);
        assert_eq!(o, a);
        let a = u16_image(2, 2, (0..16).map(|v| v * 4097).collect());
        let mut o = Image::new(2, 2, Fmt::U16);
        copy(&mut o, &a);
        assert_eq!(o, a);
        let a = f32_image(2, 2, (0..16).map(|v| v as f32 * -0.1).collect());
        let mut o = Image::new(2, 2, Fmt::F32);
        copy(&mut o, &a);
        assert_eq!(o, a);
    }

    #[test]
    #[should_panic(expected = "equal size and format")]
    fn blend_rejects_a_size_mismatch() {
        let a = Image::new(2, 2, Fmt::U8);
        let b = Image::new(2, 3, Fmt::U8);
        let mut out = Image::new(2, 2, Fmt::U8);
        blend(&mut out, &a, &b, 0.5);
    }

    #[test]
    #[should_panic(expected = "equal size and format")]
    fn blend_rejects_a_format_mismatch() {
        let a = Image::new(2, 2, Fmt::U8);
        let b = Image::new(2, 2, Fmt::U16);
        let mut out = Image::new(2, 2, Fmt::U8);
        blend(&mut out, &a, &b, 0.5);
    }

    #[test]
    fn empty_images_are_fine() {
        for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
            let a = Image::new(0, 0, fmt);
            let b = Image::new(0, 0, fmt);
            let mut out = Image::new(0, 0, fmt);
            blend(&mut out, &a, &b, 0.5);
            blend(&mut out, &a, &b, 1.5);
            lerp(&mut out, &b, 0.5);
            copy(&mut out, &a);
        }
    }

    #[test]
    fn nan_weight_follows_the_ordered_comparisons() {
        // Every comparison in `blend` is false for NaN, so the common branch runs: the u8 and
        // u16 stores see `i64::MIN` from `cvtt` (low bits zero), f32 stays NaN.
        let a = u8_image(1, 1, vec![9; 4]);
        let b = u8_image(1, 1, vec![200; 4]);
        let mut out = Image::new(1, 1, Fmt::U8);
        blend(&mut out, &a, &b, f32::NAN);
        assert_eq!(u8s(&out), &[0u8; 4]);
        let a = f32_image(1, 1, vec![0.5; 4]);
        let b = f32_image(1, 1, vec![0.25; 4]);
        let mut out = Image::new(1, 1, Fmt::F32);
        blend(&mut out, &a, &b, f32::NAN);
        assert!(f32s(&out).iter().all(|v| v.is_nan()));
    }

    #[test]
    fn cvtt_truncates_toward_zero_and_maps_invalid_to_min() {
        assert_eq!(cvtt(1.9), 1);
        assert_eq!(cvtt(-1.9), -1);
        assert_eq!(cvtt(0.99999994), 0);
        assert_eq!(cvtt(f32::NAN), i64::MIN);
        assert_eq!(cvtt(f32::INFINITY), i64::MIN);
        assert_eq!(cvtt(-f32::INFINITY), i64::MIN);
        assert_eq!(cvtt(9.3e18), i64::MIN);
        assert_eq!(cvtt(-9_223_372_036_854_775_808.0), i64::MIN);
        assert_eq!(cvtt(256.5) as u8, 0); // low byte of 256
        assert_eq!(cvtt(-0.5) as u8, 0);
        assert_eq!(cvtt(-1.5) as u8, 255); // low byte of -1, no clamping
        assert_eq!(cvtt(65536.7) as u16, 0);
    }

    // ----------------------------------------------- thread count independence

    fn with_threads<R: Send>(n: usize, f: impl FnOnce() -> R + Send) -> R {
        rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap().install(f)
    }

    #[test]
    fn results_do_not_depend_on_the_thread_count() {
        let mut rng = Rng(424242);
        let (w, h) = (97usize, 61usize);
        let mk8 = |rng: &mut Rng| u8_image(w, h, (0..w * h * 4).map(|_| (rng.next() >> 56) as u8).collect());
        let mk16 = |rng: &mut Rng| u16_image(w, h, (0..w * h * 4).map(|_| (rng.next() >> 48) as u16).collect());
        let mkf = |rng: &mut Rng| f32_image(w, h, (0..w * h * 4).map(|_| rng.unit() * 2.0 - 0.5).collect());
        let images: Vec<(Image, Image)> =
            vec![(mk8(&mut rng), mk8(&mut rng)), (mk16(&mut rng), mk16(&mut rng)), (mkf(&mut rng), mkf(&mut rng))];
        for (a, b) in &images {
            for weight in [0.3f32, 1.5, -0.5] {
                let run = |threads: usize| {
                    with_threads(threads, || {
                        let mut out = Image::new(w, h, a.fmt());
                        blend(&mut out, a, b, weight);
                        let mut l = a.clone();
                        lerp(&mut l, b, weight);
                        (out, l)
                    })
                };
                let one = run(1);
                let four = run(4);
                assert_eq!(one, four, "{:?} w={weight}", a.fmt());
            }
        }
    }

    // ----------------------------------------------------------- weighted blend

    fn random_image(fmt: Fmt, w: usize, h: usize, rng: &mut Rng) -> Image {
        let n = w * h * 4;
        match fmt {
            Fmt::U8 => u8_image(w, h, (0..n).map(|_| (rng.next() >> 56) as u8).collect()),
            Fmt::U16 => u16_image(w, h, (0..n).map(|_| (rng.next() >> 48) as u16).collect()),
            Fmt::F32 => f32_image(w, h, (0..n).map(|_| rng.unit() * 3.0 - 1.0).collect()),
        }
    }

    const WA: [f32; 11] = [0.0, 0.003, 0.1, 0.25, 0.33, 0.5, 0.67, 0.75, 0.9, 0.996, 1.0];

    #[test]
    fn weighted_blend_without_flows_is_the_plain_blend_for_integer_formats() {
        // The masked composite keeps its destination where the inverted mask is 255, which is
        // everywhere when the warped frame is a copy of the input, so no sharpening reaches the
        // result.
        let mut rng = Rng(808);
        for fmt in [Fmt::U8, Fmt::U16] {
            let a = random_image(fmt, 23, 17, &mut rng);
            let b = random_image(fmt, 23, 17, &mut rng);
            for wa in WA {
                let mut out = Image::new(23, 17, fmt);
                weighted_blend_noflow(&mut out, &a, wa, &b);
                let mut plain = Image::new(23, 17, fmt);
                blend(&mut plain, &a, &b, 1.0f32 - wa);
                assert_eq!(out, plain, "{fmt:?} wa={wa}");
            }
        }
    }

    #[test]
    fn weighted_blend_fast_path_equals_the_literal_sequence() {
        let mut rng = Rng(909);
        for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
            for (w, h) in [(1usize, 1usize), (3, 2), (9, 9), (23, 17)] {
                let a = random_image(fmt, w, h, &mut rng);
                let b = random_image(fmt, w, h, &mut rng);
                for wa in WA {
                    let mut fast = Image::new(w, h, fmt);
                    weighted_blend_noflow(&mut fast, &a, wa, &b);
                    let mut literal = Image::new(w, h, fmt);
                    weighted_blend_noflow_reference(&mut literal, &a, wa, &b);
                    match fmt {
                        Fmt::F32 => assert_eq!(bits(&fast), bits(&literal), "{fmt:?} {w}x{h} wa={wa}"),
                        _ => assert_eq!(fast, literal, "{fmt:?} {w}x{h} wa={wa}"),
                    }
                }
            }
        }
    }

    #[test]
    fn weighted_blend_on_a_frame_is_the_frame() {
        let mut rng = Rng(11);
        for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
            let a = random_image(fmt, 19, 13, &mut rng);
            let b = random_image(fmt, 19, 13, &mut rng);
            let mut out = Image::new(19, 13, fmt);
            weighted_blend_noflow(&mut out, &a, 1.0, &b);
            match fmt {
                Fmt::F32 => {
                    let want: Vec<u32> = f32s(&a).iter().map(|v| (*v + 0.0).to_bits()).collect();
                    assert_eq!(bits(&out), want);
                }
                _ => assert_eq!(out, a, "{fmt:?} wa = 1 is frame A"),
            }
            weighted_blend_noflow(&mut out, &a, 0.0, &b);
            match fmt {
                Fmt::F32 => {
                    let want: Vec<u32> = f32s(&b).iter().map(|v| (*v + 0.0).to_bits()).collect();
                    assert_eq!(bits(&out), want);
                }
                _ => assert_eq!(out, b, "{fmt:?} wa = 0 is frame B"),
            }
        }
    }

    #[test]
    fn weighted_blend_of_identical_images_gives_the_image() {
        let mut rng = Rng(12);
        for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
            let a = random_image(fmt, 11, 7, &mut rng);
            for wa in WA {
                let mut out = Image::new(11, 7, fmt);
                weighted_blend_noflow(&mut out, &a, wa, &a.clone());
                match fmt {
                    Fmt::F32 => {
                        let want: Vec<u32> = f32s(&a).iter().map(|v| (*v + 0.0).to_bits()).collect();
                        assert_eq!(bits(&out), want, "wa={wa}");
                    }
                    _ => assert_eq!(out, a, "{fmt:?} wa={wa}"),
                }
            }
        }
    }

    #[test]
    fn weighted_blend_float_negative_zero_is_normalised_like_the_composite() {
        // The float composite ends with "+ 0.0", so the sharpened frames never keep a -0.0, also where
        // blend takes a snap and copies one of them.
        let a = f32_image(2, 1, vec![-0.0, -0.0, 0.5, -0.0, -0.0, 1.0, -0.0, 0.25]);
        let b = f32_image(2, 1, vec![-0.0, 0.25, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0]);
        for wa in [0.0f32, 1.0, 0.5] {
            let mut fast = Image::new(2, 1, Fmt::F32);
            weighted_blend_noflow(&mut fast, &a, wa, &b);
            let mut literal = Image::new(2, 1, Fmt::F32);
            weighted_blend_noflow_reference(&mut literal, &a, wa, &b);
            assert_eq!(bits(&fast), bits(&literal), "wa={wa}");
            assert!(f32s(&fast).iter().all(|v| v.to_bits() != 0x8000_0000), "no -0.0 left, wa={wa}");
        }
    }

    #[test]
    fn weighted_blend_non_finite_floats_take_the_literal_sequence() {
        let mut da = vec![0.5f32; 6 * 6 * 4];
        da[(2 * 6 + 3) * 4 + 1] = f32::NAN;
        da[(4 * 6 + 1) * 4 + 2] = f32::INFINITY;
        let a = f32_image(6, 6, da);
        let b = f32_image(6, 6, vec![0.25; 6 * 6 * 4]);
        for wa in [0.5f32, 0.2] {
            let mut x = Image::new(6, 6, Fmt::F32);
            weighted_blend_noflow(&mut x, &a, wa, &b);
            let mut y = Image::new(6, 6, Fmt::F32);
            weighted_blend_noflow_reference(&mut y, &a, wa, &b);
            assert_eq!(bits(&x), bits(&y));
        }
    }

    #[test]
    fn weighted_blend_literal_sequence_does_not_depend_on_the_thread_count() {
        let mut rng = Rng(77);
        for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
            let a = random_image(fmt, 53, 31, &mut rng);
            let b = random_image(fmt, 53, 31, &mut rng);
            let run = |threads: usize| {
                with_threads(threads, || {
                    let mut out = Image::new(53, 31, fmt);
                    weighted_blend_noflow_reference(&mut out, &a, 0.37, &b);
                    let mut fast = Image::new(53, 31, fmt);
                    weighted_blend_noflow(&mut fast, &a, 0.37, &b);
                    (out, fast)
                })
            };
            assert_eq!(run(1), run(4), "{fmt:?}");
        }
    }
}
