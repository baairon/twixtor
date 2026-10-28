//! Device samplers and resampling kernels of `flow/sample.rs`, prefix `rs_`, with a line for line
//! Rust transcription of every function (`*_item` for the kernels) that the simulator runs.
//!
//! [`SAMPLERS`] holds plain OpenCL C functions and no kernel, so any program that includes it
//! (after the prelude) can call them from its own kernels; the motion solver does:
//! * `int rs_cell(float x, int len, float* f)`: the `cell` of `sample.rs` (index and fraction of
//!   one coordinate, the cell moved left with the fraction 1.0 at the last index, the index
//!   clamped into `0..=len-2`);
//! * `int rs_wrapx_cell(float x, int w, int* x1, float* f)`: the x cell of a plane periodic in x
//!   (index clamped into `0..w`, right neighbour column 0 after the last column); the one place
//!   the horizontal wrap rule of the samplers lives;
//! * `float rs_bilinear1(img, w, h, x, y)`, `float2 rs_bilinear2(img, w, h, x, y)` and
//!   `float rs_bilinear1_wrapx(img, w, h, x, y)`: `bilinear1`, `bilinear2` and
//!   `bilinear1_wrapx` (`w, h >= 2`, and `w >= 1, h >= 2` for the wrap sampler).
//!
//! Precision contract (V: `flow::sample` is the reference): coordinates convert with the
//! prelude's `cvtt_i32_f` (truncation toward zero, `INT_MIN` for NaN and out of range, x86
//! `cvttss2si`), the fraction is `x - (float)i`, the lerp chain is
//! `top = (p10 - p00) * fx + p00`, `bot = (p11 - p01) * fx + p01`, `(bot - top) * fy + top`, in
//! single precision with every operation rounded on its own. The resample kernels take the
//! scales `sx`, `sy` computed on the host by the CPU's expressions and sample at
//! `((float)x * sx, (float)y * sy)`.

use crate::device::kernels::prelude::rs;
use crate::device::sim::Global;

pub const SAMPLERS: &str = r#"
/* cell of sample.rs: i = trunc(x), f = x - (float)i; i >= len - 1 moves the cell left with the
   fraction 1.0; the index is clamped into 0..=len-2. */
int rs_cell(float x, int len, float* f) {
    int i = cvtt_i32_f(x);
    float fr = x - (float)i;
    int last = len - 1;
    if (i >= last) {
        i = last - 1;
        fr = 0x1p+0f;
    }
    if (i < 0) {
        i = 0;
    }
    int cap = len >= 2 ? len - 2 : 0;
    if (i > cap) {
        i = cap;
    }
    *f = fr;
    return i;
}

/* The x cell of a plane periodic in x (bilinear1_wrapx): ix = trunc(x), f = x - (float)ix with
   the unclamped ix, then ix clamped into 0..=w-1; the right neighbour is ix + 1, or column 0
   after the last column. */
int rs_wrapx_cell(float x, int w, int* x1, float* f) {
    int ix = cvtt_i32_f(x);
    float fx = x - (float)ix;
    int w1 = w - 1;
    if (ix < 0) {
        ix = 0;
    } else if (ix > w1) {
        ix = w1;
    }
    *x1 = ix + 1 > w1 ? 0 : ix + 1;
    *f = fx;
    return ix;
}

float rs_bilinear1(__global const float* img, int w, int h, float x, float y) {
    float fx;
    float fy;
    int ix = rs_cell(x, w, &fx);
    int iy = rs_cell(y, h, &fy);
    int i = iy * w + ix;
    float p00 = img[i];
    float p10 = img[i + 1];
    float p01 = img[i + w];
    float p11 = img[i + w + 1];
    float top = (p10 - p00) * fx + p00;
    float bot = (p11 - p01) * fx + p01;
    return (bot - top) * fy + top;
}

float2 rs_bilinear2(__global const float* img, int w, int h, float x, float y) {
    float fx;
    float fy;
    int ix = rs_cell(x, w, &fx);
    int iy = rs_cell(y, h, &fy);
    int i = (iy * w + ix) * 2;
    int r = w * 2;
    float a00 = img[i];
    float a10 = img[i + 2];
    float a01 = img[i + r];
    float a11 = img[i + r + 2];
    float atop = (a10 - a00) * fx + a00;
    float abot = (a11 - a01) * fx + a01;
    float b00 = img[i + 1];
    float b10 = img[i + 3];
    float b01 = img[i + r + 1];
    float b11 = img[i + r + 3];
    float btop = (b10 - b00) * fx + b00;
    float bbot = (b11 - b01) * fx + b01;
    return (float2)((abot - atop) * fy + atop, (bbot - btop) * fy + btop);
}

float rs_bilinear1_wrapx(__global const float* img, int w, int h, float x, float y) {
    int x1;
    float fx;
    float fy;
    int ix = rs_wrapx_cell(x, w, &x1, &fx);
    int iy = rs_cell(y, h, &fy);
    int r0 = iy * w;
    int r1 = r0 + w;
    float p00 = img[r0 + ix];
    float p10 = img[r0 + x1];
    float p01 = img[r1 + ix];
    float p11 = img[r1 + x1];
    float top = (p10 - p00) * fx + p00;
    float bot = (p11 - p01) * fx + p01;
    return (bot - top) * fy + top;
}
"#;

pub const RESAMPLE: &str = r#"
/* resample1: dst(x, y) = rs_bilinear1(src, (float)x * sx, (float)y * sy). */
__kernel void rs_resample1(__global const float* src, int sw, int sh, __global float* dst, int dw, int dh, float sx, float sy) {
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= dw || y >= dh) {
        return;
    }
    float fy = (float)y * sy;
    dst[y * dw + x] = rs_bilinear1(src, sw, sh, (float)x * sx, fy);
}

/* resample1_wrapx: the same with the periodic sampler. */
__kernel void rs_resample1_wrapx(__global const float* src, int sw, int sh, __global float* dst, int dw, int dh, float sx, float sy) {
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= dw || y >= dh) {
        return;
    }
    float fy = (float)y * sy;
    dst[y * dw + x] = rs_bilinear1_wrapx(src, sw, sh, (float)x * sx, fy);
}

/* The samplers at a list of points: out[i] = sampler(img, xs[i], ys[i]). */
__kernel void rs_gather1(__global const float* img, int w, int h, __global const float* xs, __global const float* ys, __global float* out, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    out[i] = rs_bilinear1(img, w, h, xs[i], ys[i]);
}

__kernel void rs_gather2(__global const float* img, int w, int h, __global const float* xs, __global const float* ys, __global float* out, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    float2 v = rs_bilinear2(img, w, h, xs[i], ys[i]);
    out[i * 2] = v.x;
    out[i * 2 + 1] = v.y;
}

__kernel void rs_gather1_wrapx(__global const float* img, int w, int h, __global const float* xs, __global const float* ys, __global float* out, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    out[i] = rs_bilinear1_wrapx(img, w, h, xs[i], ys[i]);
}
"#;

// ---- transcriptions ---------------------------------------------------------------------------

pub fn rs_cell(x: f32, len: i32) -> (i32, f32) {
    let mut i = rs::cvtt_i32_f(x);
    let mut fr = x - i as f32;
    let last = len - 1;
    if i >= last {
        i = last - 1;
        fr = 1.0f32;
    }
    if i < 0 {
        i = 0;
    }
    let cap = if len >= 2 { len - 2 } else { 0 };
    if i > cap {
        i = cap;
    }
    (i, fr)
}

/// `(ix, x1, fx)`.
pub fn rs_wrapx_cell(x: f32, w: i32) -> (i32, i32, f32) {
    let mut ix = rs::cvtt_i32_f(x);
    let fx = x - ix as f32;
    let w1 = w - 1;
    if ix < 0 {
        ix = 0;
    } else if ix > w1 {
        ix = w1;
    }
    let x1 = if ix + 1 > w1 { 0 } else { ix + 1 };
    (ix, x1, fx)
}

pub fn rs_bilinear1(img: &[f32], w: i32, h: i32, x: f32, y: f32) -> f32 {
    let (ix, fx) = rs_cell(x, w);
    let (iy, fy) = rs_cell(y, h);
    let i = (iy * w + ix) as usize;
    let w = w as usize;
    let p00 = img[i];
    let p10 = img[i + 1];
    let p01 = img[i + w];
    let p11 = img[i + w + 1];
    let top = (p10 - p00) * fx + p00;
    let bot = (p11 - p01) * fx + p01;
    (bot - top) * fy + top
}

pub fn rs_bilinear2(img: &[f32], w: i32, h: i32, x: f32, y: f32) -> [f32; 2] {
    let (ix, fx) = rs_cell(x, w);
    let (iy, fy) = rs_cell(y, h);
    let i = ((iy * w + ix) * 2) as usize;
    let r = (w * 2) as usize;
    let a00 = img[i];
    let a10 = img[i + 2];
    let a01 = img[i + r];
    let a11 = img[i + r + 2];
    let atop = (a10 - a00) * fx + a00;
    let abot = (a11 - a01) * fx + a01;
    let b00 = img[i + 1];
    let b10 = img[i + 3];
    let b01 = img[i + r + 1];
    let b11 = img[i + r + 3];
    let btop = (b10 - b00) * fx + b00;
    let bbot = (b11 - b01) * fx + b01;
    [(abot - atop) * fy + atop, (bbot - btop) * fy + btop]
}

pub fn rs_bilinear1_wrapx(img: &[f32], w: i32, h: i32, x: f32, y: f32) -> f32 {
    let (ix, x1, fx) = rs_wrapx_cell(x, w);
    let (iy, fy) = rs_cell(y, h);
    let r0 = iy * w;
    let r1 = r0 + w;
    let p00 = img[(r0 + ix) as usize];
    let p10 = img[(r0 + x1) as usize];
    let p01 = img[(r1 + ix) as usize];
    let p11 = img[(r1 + x1) as usize];
    let top = (p10 - p00) * fx + p00;
    let bot = (p11 - p01) * fx + p01;
    (bot - top) * fy + top
}

#[allow(clippy::too_many_arguments)]
pub fn rs_resample1_item(x: usize, y: usize, src: &[f32], sw: i32, sh: i32, dst: &mut Global<f32>, dw: i32, dh: i32, sx: f32, sy: f32) {
    let (x, y) = (x as i32, y as i32);
    if x >= dw || y >= dh {
        return;
    }
    let fy = y as f32 * sy;
    dst.set((y * dw + x) as usize, rs_bilinear1(src, sw, sh, x as f32 * sx, fy));
}

#[allow(clippy::too_many_arguments)]
pub fn rs_resample1_wrapx_item(x: usize, y: usize, src: &[f32], sw: i32, sh: i32, dst: &mut Global<f32>, dw: i32, dh: i32, sx: f32, sy: f32) {
    let (x, y) = (x as i32, y as i32);
    if x >= dw || y >= dh {
        return;
    }
    let fy = y as f32 * sy;
    dst.set((y * dw + x) as usize, rs_bilinear1_wrapx(src, sw, sh, x as f32 * sx, fy));
}
