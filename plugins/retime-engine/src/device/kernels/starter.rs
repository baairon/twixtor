//! The starter kernels the self-check runs on a device before it is used: plain copies, the
//! common branch of `blend::blend` at every depth, the prelude's conversions, and a dependent
//! loop. Each OpenCL C kernel has a Rust transcription below (`*_item`, one work item, statement by
//! statement) that the simulator runs and the tests compare with the CPU code.
//!
//! Precision contracts (V: the CPU functions named are the reference):
//! * `copy_u8`, `copy_u16`, `copy_u32`: bit copies; the f32 copy moves 32 bit words, so NaN
//!   payloads, -0.0 and subnormals pass unchanged, as `blend::copy` does.
//! * `blend_u8`, `blend_u16`: branch 4 of `blend::blend`, single precision in the order subtract,
//!   multiply, add, add `0.5f`, then the blend store (`trunc_u8_f`, `trunc_u16_f`).
//! * `blend_f32`: branch 4 of `blend::blend` in f32, `((b - a) * w + a) + 0.0f`.
//! * `convert_f32`, `convert_f64`: every conversion helper of the prelude on one input each.
//! * `index_2d`: writes `y * w + x` at `(x, y)`, the 2D range layout (`get_global_id(0)` is x).
//! * `chain_f32`: `v = v * k + x` repeated `CHAIN_STEPS` times in f32 (a self-check probe, no CPU
//!   stage), so contraction or flushing inside a loop shows up. The loop count is a constant.

use crate::device::kernels::prelude::rs;
use crate::device::sim::Global;

/// Iterations of `chain_f32`; a constant, so every launch is bounded.
pub const CHAIN_STEPS: usize = 64;
/// Outputs of `convert_f32` per input value (`CONVERT_F32_OUTPUTS` longs each).
pub const CONVERT_F32_OUTPUTS: usize = 9;
/// Outputs of `convert_f64` per input value.
pub const CONVERT_F64_OUTPUTS: usize = 2;

pub const COPY: &str = r#"
__kernel void copy_u8(__global const uchar* src, __global uchar* dst, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    dst[i] = src[i];
}

__kernel void copy_u16(__global const ushort* src, __global ushort* dst, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    dst[i] = src[i];
}

__kernel void copy_u32(__global const uint* src, __global uint* dst, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    dst[i] = src[i];
}
"#;

pub const BLEND: &str = r#"
__kernel void blend_u8(__global const uchar* a, __global const uchar* b, __global uchar* out, float w, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    float af = (float)a[i];
    float bf = (float)b[i];
    out[i] = trunc_u8_f((bf - af) * w + af + 0x1p-1f);
}

__kernel void blend_u16(__global const ushort* a, __global const ushort* b, __global ushort* out, float w, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    float af = (float)a[i];
    float bf = (float)b[i];
    out[i] = trunc_u16_f((bf - af) * w + af + 0x1p-1f);
}

__kernel void blend_f32(__global const float* a, __global const float* b, __global float* out, float w, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    out[i] = (b[i] - a[i]) * w + a[i] + 0x0p+0f;
}
"#;

pub const CONVERT: &str = r#"
__kernel void convert_f32(__global const float* x, __global long* out, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    float v = x[i];
    size_t o = i * 9;
    out[o] = (long)cvtt_i32_f(v);
    out[o + 1] = cvtt_i64_f(v);
    out[o + 2] = q_round(v);
    out[o + 3] = (long)store_q_u8(v);
    out[o + 4] = (long)store_q_u16(v);
    out[o + 5] = (long)trunc_u8_f(v);
    out[o + 6] = (long)trunc_u16_f(v);
    out[o + 7] = (long)sat_u8(v);
    out[o + 8] = (long)sat_u16(v);
}

__kernel void convert_f64(__global const double* x, __global long* out, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    double v = x[i];
    size_t o = i * 2;
    out[o] = (long)cvtt_i32_d(v);
    out[o + 1] = cvtt_i64_d(v);
}
"#;

pub const INDEX: &str = r#"
__kernel void index_2d(__global uint* out, uint w, uint h) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    if (x >= w || y >= h) {
        return;
    }
    out[y * w + x] = (uint)(y * w + x);
}
"#;

pub const CHAIN: &str = r#"
__kernel void chain_f32(__global const float* x, __global const float* k, __global float* out, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    float xi = x[i];
    float ki = k[i];
    float v = xi;
    for (int s = 0; s < 64; s++) {
        v = v * ki + xi;
    }
    out[i] = v;
}
"#;

// ---- transcriptions ---------------------------------------------------------------------------

pub fn copy_item<T: Copy>(i: usize, src: &[T], dst: &mut Global<T>, n: u32) {
    if i >= n as usize {
        return;
    }
    dst.set(i, src[i]);
}

pub fn blend_u8_item(i: usize, a: &[u8], b: &[u8], out: &mut Global<u8>, w: f32, n: u32) {
    if i >= n as usize {
        return;
    }
    let af = a[i] as f32;
    let bf = b[i] as f32;
    out.set(i, rs::trunc_u8_f((bf - af) * w + af + 0.5f32));
}

pub fn blend_u16_item(i: usize, a: &[u16], b: &[u16], out: &mut Global<u16>, w: f32, n: u32) {
    if i >= n as usize {
        return;
    }
    let af = a[i] as f32;
    let bf = b[i] as f32;
    out.set(i, rs::trunc_u16_f((bf - af) * w + af + 0.5f32));
}

pub fn blend_f32_item(i: usize, a: &[f32], b: &[f32], out: &mut Global<f32>, w: f32, n: u32) {
    if i >= n as usize {
        return;
    }
    out.set(i, (b[i] - a[i]) * w + a[i] + 0.0f32);
}

pub fn convert_f32_item(i: usize, x: &[f32], out: &mut Global<i64>, n: u32) {
    if i >= n as usize {
        return;
    }
    let v = x[i];
    let o = i * CONVERT_F32_OUTPUTS;
    out.set(o, rs::cvtt_i32_f(v) as i64);
    out.set(o + 1, rs::cvtt_i64_f(v));
    out.set(o + 2, rs::q_round(v));
    out.set(o + 3, rs::store_q_u8(v) as i64);
    out.set(o + 4, rs::store_q_u16(v) as i64);
    out.set(o + 5, rs::trunc_u8_f(v) as i64);
    out.set(o + 6, rs::trunc_u16_f(v) as i64);
    out.set(o + 7, rs::sat_u8(v) as i64);
    out.set(o + 8, rs::sat_u16(v) as i64);
}

pub fn convert_f64_item(i: usize, x: &[f64], out: &mut Global<i64>, n: u32) {
    if i >= n as usize {
        return;
    }
    let v = x[i];
    let o = i * CONVERT_F64_OUTPUTS;
    out.set(o, rs::cvtt_i32_d(v) as i64);
    out.set(o + 1, rs::cvtt_i64_d(v));
}

pub fn index_2d_item(x: usize, y: usize, out: &mut Global<u32>, w: u32, h: u32) {
    if x >= w as usize || y >= h as usize {
        return;
    }
    out.set(y * w as usize + x, (y * w as usize + x) as u32);
}

pub fn chain_f32_item(i: usize, x: &[f32], k: &[f32], out: &mut Global<f32>, n: u32) {
    if i >= n as usize {
        return;
    }
    let xi = x[i];
    let ki = k[i];
    let mut v = xi;
    for _ in 0..CHAIN_STEPS {
        v = v * ki + xi;
    }
    out.set(i, v);
}
