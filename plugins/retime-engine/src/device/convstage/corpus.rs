//! Seeded test inputs of the stage tests (device-free and device): planes with every adversarial
//! class mixed in, masks, mattes, images and sampler coordinates.

use crate::device::sim::Rng;
use crate::image::{Image, Samples};

/// The special values a plane may hold: zeros of both signs, subnormals, the `-1.0` padding
/// sentinel of the multi channel border rule, infinities, NaNs (quiet, negative, with a payload),
/// exact halves, the largest finite values.
pub fn specials() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        -1.0,
        -1.0,
        1.0,
        0.5,
        -0.5,
        2.5,
        254.5,
        255.0,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::from_bits(0x007f_ffff),
        f32::MIN_POSITIVE,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc0_0000),
        f32::from_bits(0xffc0_0000),
        f32::from_bits(0x7fc0_1234),
        f32::MAX,
        f32::MIN,
        1.0e-30,
        -1.0e30,
    ]
}

/// A uniform float in `[lo, hi)`.
pub fn uniform(rng: &mut Rng, lo: f32, hi: f32) -> f32 {
    let u = (rng.next_u32() >> 8) as f32 / 16_777_216.0;
    lo + (hi - lo) * u
}

/// `n` values: smooth noise in `[-300, 300)` plus, with probability `rate` per value, a special.
pub fn plane(rng: &mut Rng, n: usize, rate: f64) -> Vec<f32> {
    let sp = specials();
    let cut = (rate * u32::MAX as f64) as u32;
    (0..n)
        .map(|_| if rng.next_u32() < cut { sp[rng.below(sp.len() as u64) as usize] } else { uniform(rng, -300.0, 300.0) })
        .collect()
}

/// `n` finite values in `[lo, hi)`, a share of them exact halves and the `-1.0` sentinel.
pub fn finite(rng: &mut Rng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n)
        .map(|_| match rng.below(16) {
            0 => -1.0,
            1 => (uniform(rng, lo, hi)).floor() + 0.5,
            2 => 0.0,
            3 => -0.0,
            _ => uniform(rng, lo, hi),
        })
        .collect()
}

/// A half kernel of `n` taps: random positive taps (`gauss = false`) or the engine's Gaussian
/// with `n` taps.
pub fn taps(rng: &mut Rng, n: usize, gauss: bool) -> Vec<f32> {
    if gauss {
        let k = crate::flow::conv::gaussian_kernel(n as f32 - 0.5);
        assert_eq!(k.len(), n);
        return k;
    }
    (0..n).map(|_| uniform(rng, 0.001, 1.0)).collect()
}

/// A u8 hole mask of `n` entries with about `share` holes (255), the rest 0.
pub fn mask_u8(rng: &mut Rng, n: usize, share: f64) -> Vec<u8> {
    let cut = (share * u32::MAX as f64) as u32;
    (0..n).map(|_| if rng.next_u32() < cut { 255 } else { 0 }).collect()
}

/// A u16 hole mask: holes 65535, a few small non zero values (holes too), else 0.
pub fn mask_u16(rng: &mut Rng, n: usize, share: f64) -> Vec<u16> {
    let cut = (share * u32::MAX as f64) as u32;
    (0..n)
        .map(|_| if rng.next_u32() < cut { if rng.below(4) == 0 { 1 + rng.below(3) as u16 } else { 65535 } } else { 0 })
        .collect()
}

/// An f32 hole mask: holes 1.0, values around the `3.1e-5` threshold, NaN (a hole), else 0.
pub fn mask_f32(rng: &mut Rng, n: usize, share: f64) -> Vec<f32> {
    let cut = (share * u32::MAX as f64) as u32;
    (0..n)
        .map(|_| {
            if rng.next_u32() < cut {
                match rng.below(5) {
                    0 => 3.1e-5,
                    1 => f32::NAN,
                    2 => 0.5,
                    _ => 1.0,
                }
            } else {
                match rng.below(8) {
                    0 => 3.0e-5,
                    1 => -0.0,
                    _ => 0.0,
                }
            }
        })
        .collect()
}

/// A matte in 0..1 with values at and around 0.95 and a share of exact 0 and 1.
pub fn matte(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| match rng.below(10) {
            0 => 0.95,
            1 => f32::from_bits(0.95f32.to_bits() - 1),
            2 => 1.0,
            3 => 0.0,
            _ => uniform(rng, 0.0, 1.0),
        })
        .collect()
}

/// A smooth matte: a disc of radius `r` at `(cx, cy)` with a soft edge.
pub fn disc(w: usize, h: usize, cx: f32, cy: f32, r: f32) -> Vec<f32> {
    let mut v = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let d = (dx * dx + dy * dy).sqrt();
            v.push(((r - d) * 0.5 + 0.5).clamp(0.0, 1.0));
        }
    }
    v
}

/// An engine image of the given depth with random samples (float: noise in 0..1 with a few
/// specials at `rate`).
pub fn image(rng: &mut Rng, w: usize, h: usize, depth: usize, rate: f64) -> Image {
    let n = w * h * 4;
    let samples = match depth {
        8 => Samples::U8((0..n).map(|_| rng.next_u32() as u8).collect()),
        16 => Samples::U16((0..n).map(|_| rng.next_u32() as u16).collect()),
        _ => {
            let sp = specials();
            let cut = (rate * u32::MAX as f64) as u32;
            Samples::F32((0..n).map(|_| if rng.next_u32() < cut { sp[rng.below(sp.len() as u64) as usize] } else { uniform(rng, -0.1, 1.1) }).collect())
        }
    };
    Image { width: w, height: h, samples }
}

/// Sampler coordinates for an axis of `len` samples: every adversarial class.
pub fn coords(rng: &mut Rng, len: usize, extra: usize) -> Vec<f32> {
    let l = len as f32;
    let mut v = vec![
        f32::NEG_INFINITY,
        -1.0e10,
        -2147483648.0,
        -2.5,
        -1.0,
        -0.5,
        -0.0,
        0.0,
        f32::from_bits(1),
        0.25,
        0.5,
        1.0,
        l - 2.0,
        l - 1.5,
        l - 1.0,
        f32::from_bits((l - 1.0).to_bits().wrapping_sub(1)),
        l - 0.5,
        l,
        l + 3.0,
        2147483648.0,
        1.0e10,
        f32::INFINITY,
        f32::NAN,
    ];
    for _ in 0..extra {
        v.push(uniform(rng, -0.5, l + 0.5));
    }
    v
}
