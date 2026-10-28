//! Device kernels of the hole fill (`holefill.rs`), prefix `hf_`, with a line for line Rust
//! transcription of every kernel (`*_item`) that the simulator runs. The pull and push passes are
//! the `cv_` and `rs_` kernels; these are the per sample steps between them.
//!
//! Precision contract (V: `holefill.rs` is the reference):
//! * `hf_flags_u8`, `hf_flags_u16`, `hf_flags_f32`: the hole test `!(eps > (double)m)` (a NaN
//!   mask value is a hole), `eps` passed from the host (`U8_EPS`, `WIDE_EPS`); 1 for a hole.
//! * `hf_weight0`: the level 0 weight, `0.0f` at a hole, else `1.0f`.
//! * `hf_load`: the level 0 value, `0.0f` at a hole, else the sample.
//! * `hf_combine`: `if (w != 0.0f) sol = v / w` (a NaN weight divides), in place on `sol`.
//! * `hf_extract_*`: channel `c` of an interleaved image or flow as floats (`(float)byte`,
//!   `(float)word`, a copy).
//! * `hf_store_u8`, `hf_store_u16`: the write back `y = x + 0.5f`, `0 > y` gives 0, `y > white`
//!   gives white, else the low byte or word of `cvtt(y)` (`blend::cvtt`, 64 bit truncation);
//!   `hf_store_f32` a copy.

use crate::device::kernels::prelude::rs;
use crate::device::sim::Global;

pub const HOLEFILL: &str = r#"
__kernel void hf_flags_u8(__global const uchar* m, __global uchar* hole, double eps, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    hole[i] = !(eps > (double)m[i]) ? 1 : 0;
}

__kernel void hf_flags_u16(__global const ushort* m, __global uchar* hole, double eps, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    hole[i] = !(eps > (double)m[i]) ? 1 : 0;
}

__kernel void hf_flags_f32(__global const float* m, __global uchar* hole, double eps, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    hole[i] = !(eps > (double)m[i]) ? 1 : 0;
}

__kernel void hf_weight0(__global const uchar* hole, __global float* wt, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    wt[i] = hole[i] != 0 ? 0x0p+0f : 0x1p+0f;
}

__kernel void hf_load(__global const float* plane, __global const uchar* hole, __global float* v0, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    v0[i] = hole[i] != 0 ? 0x0p+0f : plane[i];
}

__kernel void hf_combine(__global float* sol, __global const float* value, __global const float* weight, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    float w = weight[i];
    if (w != 0x0p+0f) {
        sol[i] = value[i] / w;
    }
}

__kernel void hf_extract_u8(__global const uchar* img, __global float* plane, int stride, int c, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    plane[i] = (float)img[i * stride + c];
}

__kernel void hf_extract_u16(__global const ushort* img, __global float* plane, int stride, int c, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    plane[i] = (float)img[i * stride + c];
}

__kernel void hf_extract_f32(__global const float* img, __global float* plane, int stride, int c, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    plane[i] = img[i * stride + c];
}

__kernel void hf_store_u8(__global const float* plane, __global uchar* img, int stride, int c, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    float y = plane[i] + 0x1p-1f;
    uchar v;
    if (0x0p+0f > y) {
        v = 0;
    } else if (y > 0x1.fep+7f) {
        v = 255;
    } else {
        v = (uchar)cvtt_i64_f(y);
    }
    img[i * stride + c] = v;
}

__kernel void hf_store_u16(__global const float* plane, __global ushort* img, int stride, int c, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    float y = plane[i] + 0x1p-1f;
    ushort v;
    if (0x0p+0f > y) {
        v = 0;
    } else if (y > 0x1.fffep+15f) {
        v = 65535;
    } else {
        v = (ushort)cvtt_i64_f(y);
    }
    img[i * stride + c] = v;
}

__kernel void hf_store_f32(__global const float* plane, __global float* img, int stride, int c, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    img[i * stride + c] = plane[i];
}
"#;

// ---- transcriptions ---------------------------------------------------------------------------

pub fn hf_flags_item(i: usize, m: &[f64], hole: &mut Global<u8>, eps: f64, n: i32) {
    if i as i32 >= n {
        return;
    }
    hole.set(i, if !(eps > m[i]) { 1 } else { 0 });
}

pub fn hf_weight0_item(i: usize, hole: &[u8], wt: &mut Global<f32>, n: i32) {
    if i as i32 >= n {
        return;
    }
    wt.set(i, if hole[i] != 0 { 0.0f32 } else { 1.0f32 });
}

pub fn hf_load_item(i: usize, plane: &[f32], hole: &[u8], v0: &mut Global<f32>, n: i32) {
    if i as i32 >= n {
        return;
    }
    v0.set(i, if hole[i] != 0 { 0.0f32 } else { plane[i] });
}

/// `hf_combine` on a copy: `sol` is read, the result written to `out` (every item its own element).
pub fn hf_combine_item(i: usize, sol: &[f32], value: &[f32], weight: &[f32], out: &mut Global<f32>, n: i32) {
    if i as i32 >= n {
        return;
    }
    let w = weight[i];
    let mut s = sol[i];
    if w != 0.0f32 {
        s = value[i] / w;
    }
    out.set(i, s);
}

pub fn hf_extract_item<T: Copy>(i: usize, img: &[T], plane: &mut Global<f32>, stride: i32, c: i32, n: i32, to_f32: fn(T) -> f32) {
    if i as i32 >= n {
        return;
    }
    plane.set(i, to_f32(img[i * stride as usize + c as usize]));
}

pub fn hf_store_u8_item(i: usize, plane: &[f32], img: &mut Global<u8>, stride: i32, c: i32, n: i32) {
    if i as i32 >= n {
        return;
    }
    let y = plane[i] + 0.5f32;
    let v: u8 = if 0.0f32 > y {
        0
    } else if y > 255.0f32 {
        255
    } else {
        rs::cvtt_i64_f(y) as u8
    };
    img.set(i * stride as usize + c as usize, v);
}

pub fn hf_store_u16_item(i: usize, plane: &[f32], img: &mut Global<u16>, stride: i32, c: i32, n: i32) {
    if i as i32 >= n {
        return;
    }
    let y = plane[i] + 0.5f32;
    let v: u16 = if 0.0f32 > y {
        0
    } else if y > 65535.0f32 {
        65535
    } else {
        rs::cvtt_i64_f(y) as u16
    };
    img.set(i * stride as usize + c as usize, v);
}
