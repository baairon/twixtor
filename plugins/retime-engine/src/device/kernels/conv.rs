//! Device kernels of the separable convolution (`flow/conv.rs`) and of the matte choke
//! (`matte.rs`), prefix `cv_`, with a line for line Rust transcription of every kernel and helper
//! (`*_item`, one work item, statement by statement) that the simulator runs.
//!
//! Precision contract (V: `flow::conv` and `matte::choke` are the reference). One work item
//! computes one output sample of one pass, serially, in the CPU's order:
//! * `cv_hpass` / `cv_vpass`: the horizontal and the vertical pass of `conv::convolve`. A sample
//!   is interior exactly when the CPU's `head <= i < head + interior` holds (`head = min(n, len)`,
//!   `interior = len - 2n` saturating); interior samples take the chain `s = x[i] * k[0]`, then
//!   for `j = n - 1` down to 1 `s = s + (x[i + j] + x[i - j]) * k[j]`; every other sample the
//!   border rule (`num = k[0] * x[i]`, `den = k[0]`, then for `j = n - 1` down to 1 the right tap
//!   and then the left tap, `den = den + w`, `num = num + xv * w`, result `num / den`) with the
//!   zero padding of 1 channel lines (`xv = 0`, `w = 0` outside) and the `-1.0` sentinel of 2 and
//!   3 channel lines (`w = mask * k[j]`, the mask 0 for a sample equal to `-1.0`, outside or not).
//! * `cv_vpass_above`: the vertical pass of `conv::convolve_above_first`, the same as `cv_vpass`
//!   except that the border rule takes the tap above (`y - j`) before the tap below
//!   (`cv_border_ordered` with `left_first`; `holefill::VerticalTaps::AboveFirst`).
//! * `cv_hpass_wrapx`: the circular horizontal pass of `conv::convolve_wrapx`, the interior chain
//!   at every sample with the tap columns from [`CV_WRAPX_TAP`]'s helper `cv_wrapx_tap`, the one
//!   place the horizontal wrap rule of the convolution lives.
//! * `cv_choke_*`: the per sample maps of `matte::choke`: `1.0 - v`, `clamp01((b - t) / den)`,
//!   `v * it + c * t` and `clamp01(v)`, with `t`, `den` and `it` computed on the host by the CPU's
//!   expressions.
//!
//! Taps are passed as a buffer of `n` floats computed on the host (`conv::gaussian_kernel`,
//! `conv::BINOMIAL5`); `n` is at most [`CV_MAX_TAPS`], the constant bound of every tap loop.

use crate::device::sim::Global;

/// The largest half kernel the device passes take; a longer one is refused by the wrappers
/// (`StageError::Unsupported`). It is the constant bound of every tap loop (`q < 64`).
pub const CV_MAX_TAPS: usize = 64;

/// The horizontal wrap rule of the convolution, alone so a periodic variant changes one place.
pub const CV_WRAPX_TAP: &str = r#"
/* The column of a circular tap: t modulo len, Euclidean (len >= 1). */
int cv_wrapx_tap(int t, int len) {
    int m = t % len;
    return m < 0 ? m + len : m;
}
"#;

pub const CONV: &str = r#"
/* Interior chain of conv.rs: s = x[i] * k[0], then s = s + (x[i + j] + x[i - j]) * k[j] for
   j = n - 1 down to 1. Element t of the line is src[base + t * stride]. */
float cv_interior(__global const float* src, int base, int stride, int i, __global const float* k, int n) {
    float s = src[base + i * stride] * k[0];
    for (int q = 0; q < 64; q++) {
        int j = n - 1 - q;
        if (j < 1) {
            break;
        }
        s = s + (src[base + (i + j) * stride] + src[base + (i - j) * stride]) * k[j];
    }
    return s;
}

/* Border rule of conv.rs (border_value_ordered): the renormalised chain, right tap then left tap,
   or with left_first != 0 the left tap then the right tap. With multi == 0 a tap outside the line
   adds (0, 0); with multi != 0 the line is padded with -1.0 and a sample equal to -1.0 gets the
   weight 0. */
float cv_border_ordered(__global const float* src, int base, int stride, int len, int i, __global const float* k, int n, int multi, int left_first) {
    float centre = src[base + i * stride];
    float num = k[0] * centre;
    float den = k[0];
    for (int q = 0; q < 64; q++) {
        int j = n - 1 - q;
        if (j < 1) {
            break;
        }
        float kj = k[j];
        for (int side = 0; side < 2; side++) {
            int right = (side == 0) != (left_first != 0);
            int t = right ? i + j : i - j;
            int inside = t >= 0 && t < len;
            float xv;
            float w;
            if (multi == 0) {
                if (inside) {
                    xv = src[base + t * stride];
                    w = kj;
                } else {
                    xv = 0x0p+0f;
                    w = 0x0p+0f;
                }
            } else {
                float v = inside ? src[base + t * stride] : -0x1p+0f;
                float mask = v != -0x1p+0f ? 0x1p+0f : 0x0p+0f;
                xv = v;
                w = mask * kj;
            }
            den = den + w;
            num = num + xv * w;
        }
    }
    return num / den;
}

/* Border rule of conv.rs (border_value): the right tap first. */
float cv_border(__global const float* src, int base, int stride, int len, int i, __global const float* k, int n, int multi) {
    return cv_border_ordered(src, base, stride, len, i, k, n, multi, 0);
}

/* Horizontal pass of conv.rs (hrow): item (g, y), g = x * c + channel. */
__kernel void cv_hpass(__global const float* src, __global float* dst, int w, int h, int c, __global const float* k, int n) {
    int g = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (g >= w * c || y >= h) {
        return;
    }
    int i = g / c;
    int ch = g - i * c;
    int base = y * w * c + ch;
    int head = n < w ? n : w;
    int interior = w > 2 * n ? w - 2 * n : 0;
    float v;
    if (i >= head && i < head + interior) {
        v = cv_interior(src, base, c, i, k, n);
    } else {
        v = cv_border(src, base, c, w, i, k, n, c != 1);
    }
    dst[base + i * c] = v;
}

/* Vertical pass of conv.rs (vrow): item (t, y), t = x * c + channel. */
__kernel void cv_vpass(__global const float* src, __global float* dst, int w, int h, int c, __global const float* k, int n) {
    int t = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    int row = w * c;
    if (t >= row || y >= h) {
        return;
    }
    int head = n < h ? n : h;
    int interior = h > 2 * n ? h - 2 * n : 0;
    float v;
    if (y >= head && y < head + interior) {
        v = cv_interior(src, t, row, y, k, n);
    } else {
        v = cv_border(src, t, row, h, y, k, n, c != 1);
    }
    dst[y * row + t] = v;
}

/* Vertical pass of conv.rs whose border rows take the tap above first (vrow_ordered with
   ABOVE_FIRST, the vertical pass of convolve_above_first): item (t, y) as cv_vpass. */
__kernel void cv_vpass_above(__global const float* src, __global float* dst, int w, int h, int c, __global const float* k, int n) {
    int t = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    int row = w * c;
    if (t >= row || y >= h) {
        return;
    }
    int head = n < h ? n : h;
    int interior = h > 2 * n ? h - 2 * n : 0;
    float v;
    if (y >= head && y < head + interior) {
        v = cv_interior(src, t, row, y, k, n);
    } else {
        v = cv_border_ordered(src, t, row, h, y, k, n, c != 1, 1);
    }
    dst[y * row + t] = v;
}

/* Circular horizontal pass of conv.rs (hrow_wrapx): the interior chain at every sample with the
   tap columns taken by cv_wrapx_tap. */
__kernel void cv_hpass_wrapx(__global const float* src, __global float* dst, int w, int h, int c, __global const float* k, int n) {
    int g = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (g >= w * c || y >= h) {
        return;
    }
    int i = g / c;
    int ch = g - i * c;
    int base = y * w * c + ch;
    float s = src[base + i * c] * k[0];
    for (int q = 0; q < 64; q++) {
        int j = n - 1 - q;
        if (j < 1) {
            break;
        }
        int r = cv_wrapx_tap(i + j, w);
        int l = cv_wrapx_tap(i - j, w);
        s = s + (src[base + r * c] + src[base + l * c]) * k[j];
    }
    dst[base + i * c] = s;
}
"#;

pub const CHOKE: &str = r#"
/* clamp01 of matte.rs: above 1 gives 1, above 0 the value, anything else (NaN, -0.0) 0. */
float cv_clamp01(float x) {
    if (x > 0x1p+0f) {
        return 0x1p+0f;
    }
    if (x > 0x0p+0f) {
        return x;
    }
    return 0x0p+0f;
}

__kernel void cv_choke_invert(__global float* data, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    data[i] = 0x1p+0f - data[i];
}

__kernel void cv_choke_threshold(__global float* data, float t, float den, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    data[i] = cv_clamp01((data[i] - t) / den);
}

__kernel void cv_choke_mix(__global float* data, __global const float* c, float t, float it, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    data[i] = data[i] * it + c[i] * t;
}

__kernel void cv_choke_clamp(__global float* data, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    data[i] = cv_clamp01(data[i]);
}
"#;

// ---- transcriptions ---------------------------------------------------------------------------

pub fn cv_wrapx_tap(t: i32, len: i32) -> i32 {
    let m = t % len;
    if m < 0 {
        m + len
    } else {
        m
    }
}

pub fn cv_interior(src: &[f32], base: i32, stride: i32, i: i32, k: &[f32], n: i32) -> f32 {
    let mut s = src[(base + i * stride) as usize] * k[0];
    for q in 0..CV_MAX_TAPS as i32 {
        let j = n - 1 - q;
        if j < 1 {
            break;
        }
        s = s + (src[(base + (i + j) * stride) as usize] + src[(base + (i - j) * stride) as usize]) * k[j as usize];
    }
    s
}

#[allow(clippy::too_many_arguments)]
pub fn cv_border_ordered(src: &[f32], base: i32, stride: i32, len: i32, i: i32, k: &[f32], n: i32, multi: bool, left_first: bool) -> f32 {
    let centre = src[(base + i * stride) as usize];
    let mut num = k[0] * centre;
    let mut den = k[0];
    for q in 0..CV_MAX_TAPS as i32 {
        let j = n - 1 - q;
        if j < 1 {
            break;
        }
        let kj = k[j as usize];
        for side in 0..2 {
            let right = (side == 0) != left_first;
            let t = if right { i + j } else { i - j };
            let inside = t >= 0 && t < len;
            let xv;
            let w;
            if !multi {
                if inside {
                    xv = src[(base + t * stride) as usize];
                    w = kj;
                } else {
                    xv = 0.0f32;
                    w = 0.0f32;
                }
            } else {
                let v = if inside { src[(base + t * stride) as usize] } else { -1.0f32 };
                let mask = if v != -1.0f32 { 1.0f32 } else { 0.0f32 };
                xv = v;
                w = mask * kj;
            }
            den = den + w;
            num = num + xv * w;
        }
    }
    num / den
}

#[allow(clippy::too_many_arguments)]
pub fn cv_border(src: &[f32], base: i32, stride: i32, len: i32, i: i32, k: &[f32], n: i32, multi: bool) -> f32 {
    cv_border_ordered(src, base, stride, len, i, k, n, multi, false)
}

#[allow(clippy::too_many_arguments)]
pub fn cv_hpass_item(g: usize, y: usize, src: &[f32], dst: &mut Global<f32>, w: i32, h: i32, c: i32, k: &[f32], n: i32) {
    let (g, y) = (g as i32, y as i32);
    if g >= w * c || y >= h {
        return;
    }
    let i = g / c;
    let ch = g - i * c;
    let base = y * w * c + ch;
    let head = if n < w { n } else { w };
    let interior = if w > 2 * n { w - 2 * n } else { 0 };
    let v = if i >= head && i < head + interior {
        cv_interior(src, base, c, i, k, n)
    } else {
        cv_border(src, base, c, w, i, k, n, c != 1)
    };
    dst.set((base + i * c) as usize, v);
}

#[allow(clippy::too_many_arguments)]
pub fn cv_vpass_item(t: usize, y: usize, src: &[f32], dst: &mut Global<f32>, w: i32, h: i32, c: i32, k: &[f32], n: i32) {
    let (t, y) = (t as i32, y as i32);
    let row = w * c;
    if t >= row || y >= h {
        return;
    }
    let head = if n < h { n } else { h };
    let interior = if h > 2 * n { h - 2 * n } else { 0 };
    let v = if y >= head && y < head + interior {
        cv_interior(src, t, row, y, k, n)
    } else {
        cv_border(src, t, row, h, y, k, n, c != 1)
    };
    dst.set((y * row + t) as usize, v);
}

#[allow(clippy::too_many_arguments)]
pub fn cv_vpass_above_item(t: usize, y: usize, src: &[f32], dst: &mut Global<f32>, w: i32, h: i32, c: i32, k: &[f32], n: i32) {
    let (t, y) = (t as i32, y as i32);
    let row = w * c;
    if t >= row || y >= h {
        return;
    }
    let head = if n < h { n } else { h };
    let interior = if h > 2 * n { h - 2 * n } else { 0 };
    let v = if y >= head && y < head + interior {
        cv_interior(src, t, row, y, k, n)
    } else {
        cv_border_ordered(src, t, row, h, y, k, n, c != 1, true)
    };
    dst.set((y * row + t) as usize, v);
}

#[allow(clippy::too_many_arguments)]
pub fn cv_hpass_wrapx_item(g: usize, y: usize, src: &[f32], dst: &mut Global<f32>, w: i32, h: i32, c: i32, k: &[f32], n: i32) {
    let (g, y) = (g as i32, y as i32);
    if g >= w * c || y >= h {
        return;
    }
    let i = g / c;
    let ch = g - i * c;
    let base = y * w * c + ch;
    let mut s = src[(base + i * c) as usize] * k[0];
    for q in 0..CV_MAX_TAPS as i32 {
        let j = n - 1 - q;
        if j < 1 {
            break;
        }
        let r = cv_wrapx_tap(i + j, w);
        let l = cv_wrapx_tap(i - j, w);
        s = s + (src[(base + r * c) as usize] + src[(base + l * c) as usize]) * k[j as usize];
    }
    dst.set((base + i * c) as usize, s);
}

pub fn cv_clamp01(x: f32) -> f32 {
    if x > 1.0f32 {
        return 1.0f32;
    }
    if x > 0.0f32 {
        return x;
    }
    0.0f32
}

pub fn cv_choke_invert_item(i: usize, data: &[f32], out: &mut Global<f32>, n: i32) {
    if i as i32 >= n {
        return;
    }
    out.set(i, 1.0f32 - data[i]);
}

pub fn cv_choke_threshold_item(i: usize, data: &[f32], out: &mut Global<f32>, t: f32, den: f32, n: i32) {
    if i as i32 >= n {
        return;
    }
    out.set(i, cv_clamp01((data[i] - t) / den));
}

pub fn cv_choke_mix_item(i: usize, data: &[f32], c: &[f32], out: &mut Global<f32>, t: f32, it: f32, n: i32) {
    if i as i32 >= n {
        return;
    }
    out.set(i, data[i] * it + c[i] * t);
}

pub fn cv_choke_clamp_item(i: usize, data: &[f32], out: &mut Global<f32>, n: i32) {
    if i as i32 >= n {
        return;
    }
    out.set(i, cv_clamp01(data[i]));
}
