//! Device kernels of the solver's pyramids (`prep.rs`, `flow.rs`, `flow/extras.rs`), prefix `pm_`,
//! with a line for line Rust transcription of every kernel (`*_item`) that the simulator runs.
//! The blurs and resamples of the pyramids are the `cv_` and `rs_` kernels; these are the steps
//! between them.
//!
//! Precision contract (V: the CPU functions named are the reference):
//! * `pm_luma_u8`, `pm_luma_u16`, `pm_luma_f32`: `prep::luma` per pixel, the 8 bit sum in single
//!   precision `(r * 0.3f + g * 0.59f) + b * 0.11f` (with Track Alpha widened and `a * 0.3` added
//!   in double), the 16 bit and float sums in double precision `((g * 0.59 + r * 0.3) + b * 0.11)`
//!   (`+ a * 0.3`), then `/ 257.0` or `* 255.0` and the conversion to single (round to nearest);
//!   the video range rule in single precision. The weights are the hexadecimal forms of the CPU's
//!   decimal constants (a test checks every literal).
//! * `pm_subtract`: `a = a - b` (the band pass `orig - blur5(orig)` of `flow::build_pyramid`).
//! * `pm_binarize`: `v >= 0.95f ? 1.0f : 0.0f` (`flow::MASK_ON`, the mask pyramid, a NaN gives 0).
//! * `pm_upsample_flow`: `flow::upsample_flow`, the corner aligned two channel upsample:
//!   `rs_bilinear2` at `((float)x * rx, (float)y * ry)`, the lanes multiplied by `sx` and `sy`;
//!   `sx`, `sy`, `rx`, `ry` computed on the host by the CPU's expressions.

use crate::device::kernels::resample;
use crate::device::sim::Global;

/// `prep.rs`'s luma weights as bits, single then double precision: 0.3, 0.59, 0.11.
pub const LUMA_F32_BITS: [u32; 3] = [0x3e99_999a, 0x3f17_0a3d, 0x3de1_47ae];
pub const LUMA_F64_BITS: [u64; 3] = [0x3fd3_3333_3333_3333, 0x3fe2_e147_ae14_7ae1, 0x3fbc_28f5_c28f_5c29];

pub const PYRAMID: &str = r#"
/* prep.rs luma, 8 bit: video range ((r - 16) * 255) / 219, else (r * 0.3f + g * 0.59f) + b * 0.11f,
   with track_alpha widened and a * 0.3 added in double. */
__kernel void pm_luma_u8(__global const uchar* s, __global float* out, int track_alpha, int video, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    float a = (float)s[i * 4];
    float r = (float)s[i * 4 + 1];
    float g = (float)s[i * 4 + 2];
    float b = (float)s[i * 4 + 3];
    float v;
    if (video) {
        v = ((r - 0x1p+4f) * 0x1.fep+7f) / 0x1.b6p+7f;
    } else {
        float sum = (r * 0x1.333334p-2f + g * 0x1.2e147ap-1f) + b * 0x1.c28f5cp-4f;
        if (track_alpha) {
            v = (float)((double)sum + (double)a * 0x1.3333333333333p-2);
        } else {
            v = sum;
        }
    }
    out[i] = v;
}

/* prep.rs luma, 16 bit: the double sum divided by 257; video range in single precision. */
__kernel void pm_luma_u16(__global const ushort* s, __global float* out, int track_alpha, int video, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    double a = (double)s[i * 4];
    double r = (double)s[i * 4 + 1];
    double g = (double)s[i * 4 + 2];
    double b = (double)s[i * 4 + 3];
    float v;
    if (video) {
        v = (((float)s[i * 4 + 1] / 0x1.01p+8f - 0x1p+4f) * 0x1.fep+7f) / 0x1.b6p+7f;
    } else {
        double sum = (g * 0x1.2e147ae147ae1p-1 + r * 0x1.3333333333333p-2) + b * 0x1.c28f5c28f5c29p-4;
        if (track_alpha) {
            sum = sum + a * 0x1.3333333333333p-2;
        }
        v = (float)(sum / 0x1.01p+8);
    }
    out[i] = v;
}

/* prep.rs luma, float: the double sum times 255; video range in single precision. */
__kernel void pm_luma_f32(__global const float* s, __global float* out, int track_alpha, int video, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    double a = (double)s[i * 4];
    double r = (double)s[i * 4 + 1];
    double g = (double)s[i * 4 + 2];
    double b = (double)s[i * 4 + 3];
    float v;
    if (video) {
        v = ((s[i * 4 + 1] * 0x1.fep+7f - 0x1p+4f) * 0x1.fep+7f) / 0x1.b6p+7f;
    } else {
        double sum = (g * 0x1.2e147ae147ae1p-1 + r * 0x1.3333333333333p-2) + b * 0x1.c28f5c28f5c29p-4;
        if (track_alpha) {
            sum = sum + a * 0x1.3333333333333p-2;
        }
        v = (float)(sum * 0x1.fep+7);
    }
    out[i] = v;
}

/* a = a - b (the band pass of build_pyramid). */
__kernel void pm_subtract(__global float* a, __global const float* b, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    a[i] = a[i] - b[i];
}

/* v >= 0.95f ? 1 : 0 (MASK_ON of flow.rs). */
__kernel void pm_binarize(__global float* p, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    p[i] = p[i] >= 0x1.e66666p-1f ? 0x1p+0f : 0x0p+0f;
}

/* upsample_flow of flow.rs: the corner aligned two channel upsample, lanes scaled by (sx, sy). */
__kernel void pm_upsample_flow(__global const float* src, int sw, int sh, __global float* dst, int dw, int dh, float sx, float sy, float rx, float ry) {
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= dw || y >= dh) {
        return;
    }
    float fy = (float)y * ry;
    float2 v = rs_bilinear2(src, sw, sh, (float)x * rx, fy);
    int o = (y * dw + x) * 2;
    dst[o] = v.x * sx;
    dst[o + 1] = v.y * sy;
}
"#;

// ---- transcriptions ---------------------------------------------------------------------------

fn f(bits: u32) -> f32 {
    f32::from_bits(bits)
}

fn d(bits: u64) -> f64 {
    f64::from_bits(bits)
}

pub fn pm_luma_u8_item(i: usize, s: &[u8], out: &mut Global<f32>, track_alpha: bool, video: bool, n: i32) {
    if i as i32 >= n {
        return;
    }
    let a = s[i * 4] as f32;
    let r = s[i * 4 + 1] as f32;
    let g = s[i * 4 + 2] as f32;
    let b = s[i * 4 + 3] as f32;
    let v = if video {
        ((r - 16.0f32) * 255.0f32) / 219.0f32
    } else {
        let sum = (r * f(LUMA_F32_BITS[0]) + g * f(LUMA_F32_BITS[1])) + b * f(LUMA_F32_BITS[2]);
        if track_alpha {
            (sum as f64 + a as f64 * d(LUMA_F64_BITS[0])) as f32
        } else {
            sum
        }
    };
    out.set(i, v);
}

pub fn pm_luma_u16_item(i: usize, s: &[u16], out: &mut Global<f32>, track_alpha: bool, video: bool, n: i32) {
    if i as i32 >= n {
        return;
    }
    let a = s[i * 4] as f64;
    let r = s[i * 4 + 1] as f64;
    let g = s[i * 4 + 2] as f64;
    let b = s[i * 4 + 3] as f64;
    let v = if video {
        ((s[i * 4 + 1] as f32 / 257.0f32 - 16.0f32) * 255.0f32) / 219.0f32
    } else {
        let mut sum = (g * d(LUMA_F64_BITS[1]) + r * d(LUMA_F64_BITS[0])) + b * d(LUMA_F64_BITS[2]);
        if track_alpha {
            sum = sum + a * d(LUMA_F64_BITS[0]);
        }
        (sum / 257.0f64) as f32
    };
    out.set(i, v);
}

pub fn pm_luma_f32_item(i: usize, s: &[f32], out: &mut Global<f32>, track_alpha: bool, video: bool, n: i32) {
    if i as i32 >= n {
        return;
    }
    let a = s[i * 4] as f64;
    let r = s[i * 4 + 1] as f64;
    let g = s[i * 4 + 2] as f64;
    let b = s[i * 4 + 3] as f64;
    let v = if video {
        ((s[i * 4 + 1] * 255.0f32 - 16.0f32) * 255.0f32) / 219.0f32
    } else {
        let mut sum = (g * d(LUMA_F64_BITS[1]) + r * d(LUMA_F64_BITS[0])) + b * d(LUMA_F64_BITS[2]);
        if track_alpha {
            sum = sum + a * d(LUMA_F64_BITS[0]);
        }
        (sum * 255.0f64) as f32
    };
    out.set(i, v);
}

pub fn pm_subtract_item(i: usize, a: &[f32], b: &[f32], out: &mut Global<f32>, n: i32) {
    if i as i32 >= n {
        return;
    }
    out.set(i, a[i] - b[i]);
}

/// `0.95f32` as bits (`flow::MASK_ON`).
pub const MASK_ON_BITS: u32 = 0x3f73_3333;

pub fn pm_binarize_item(i: usize, p: &[f32], out: &mut Global<f32>, n: i32) {
    if i as i32 >= n {
        return;
    }
    out.set(i, if p[i] >= f(MASK_ON_BITS) { 1.0f32 } else { 0.0f32 });
}

#[allow(clippy::too_many_arguments)]
pub fn pm_upsample_flow_item(
    x: usize,
    y: usize,
    src: &[f32],
    sw: i32,
    sh: i32,
    dst: &mut Global<f32>,
    dw: i32,
    dh: i32,
    scale: [f32; 4],
) {
    let [sx, sy, rx, ry] = scale;
    let (x, y) = (x as i32, y as i32);
    if x >= dw || y >= dh {
        return;
    }
    let fy = y as f32 * ry;
    let v = resample::rs_bilinear2(src, sw, sh, x as f32 * rx, fy);
    let o = ((y * dw + x) * 2) as usize;
    dst.set(o, v[0] * sx);
    dst.set(o + 1, v[1] * sy);
}
