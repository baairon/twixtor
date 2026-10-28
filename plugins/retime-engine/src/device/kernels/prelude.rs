//! The OpenCL C prelude every device program starts with, and its line for line Rust
//! transcription ([`rs`]).
//!
//! Precision contract (V: the CPU helpers it mirrors are the reference). `FP_CONTRACT OFF`, so no
//! expression is fused into a multiply add; `cl_khr_fp64` for the double precision steps. Every
//! float to integer conversion goes through a range test first, because a C cast of an out of
//! range value is undefined: inside the range the cast truncates toward zero, outside it and for
//! NaN the result is the minimum of the target type, which is what x86 `cvtt` gives and what the
//! CPU helpers return (`blend::cvtt`, `warp::cvtt32`, `image::q`, `timeline::cvtt_i32`). A
//! narrowing store keeps the low byte or word (C conversion to an unsigned type is modulo). Float
//! literals are written in hexadecimal only, so no decimal to binary rounding is involved.
//!
//! Helpers:
//! * `cvtt_i32_f(float)`, `cvtt_i64_f(float)`, `cvtt_i32_d(double)`, `cvtt_i64_d(double)`:
//!   truncation, `INT_MIN` / `LONG_MIN` out of range or NaN.
//! * `q_round(float x)`: `cvtt_i64_d((double)x + 0.5)`, the `q` of `image.rs`; `store_q_u8` and
//!   `store_q_u16` keep its low byte or word.
//! * `trunc_u8_f(float v)` / `trunc_u16_f`: the low byte or word of `cvtt_i64_f(v)`, the store of
//!   `blend.rs` (callers add `0.5f` first).
//! * `sat_u8(float v)` / `sat_u16`: the saturating store chain of `warp.rs`.

/// The prelude source. It must stay the first text of every program (`kernels::program`).
pub const PRELUDE: &str = r#"#pragma OPENCL FP_CONTRACT OFF
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/* Truncation toward zero; outside [-2^31, 2^31) and NaN give INT_MIN (x86 cvttss2si). */
int cvtt_i32_f(float x) {
    if (x >= -0x1p31f && x < 0x1p31f) {
        return (int)x;
    }
    return INT_MIN;
}

/* Truncation toward zero; outside [-2^63, 2^63) and NaN give LONG_MIN. */
long cvtt_i64_f(float x) {
    if (x >= -0x1p63f && x < 0x1p63f) {
        return (long)x;
    }
    return LONG_MIN;
}

/* Truncation toward zero of a double; outside (-2^31 - 1, 2^31) and NaN give INT_MIN. */
int cvtt_i32_d(double x) {
    if (x > -0x1.00000002p31 && x < 0x1p31) {
        return (int)x;
    }
    return INT_MIN;
}

/* Truncation toward zero of a double; outside [-2^63, 2^63) and NaN give LONG_MIN. */
long cvtt_i64_d(double x) {
    if (x >= -0x1p63 && x < 0x1p63) {
        return (long)x;
    }
    return LONG_MIN;
}

/* The q rounding of image.rs: trunc((double)x + 0.5). */
long q_round(float x) {
    return cvtt_i64_d((double)x + 0x1p-1);
}

/* The q store: the low byte or word of q_round. */
uchar store_q_u8(float x) {
    return (uchar)q_round(x);
}

ushort store_q_u16(float x) {
    return (ushort)q_round(x);
}

/* The blend store: the low byte or word of the truncated 64 bit integer, no clamp. */
uchar trunc_u8_f(float v) {
    return (uchar)cvtt_i64_f(v);
}

ushort trunc_u16_f(float v) {
    return (ushort)cvtt_i64_f(v);
}

/* The saturating store of warp.rs: add one half, truncate, bias by -128 (wrapping), saturate to
   the signed 16 bit and then the signed 8 bit range, add the bias back. */
uchar sat_u8(float v) {
    int i = cvtt_i32_f(v + 0x1p-1f);
    int t = as_int(as_uint(i) + 0xffffff80u);
    int w = t < -32768 ? -32768 : (t > 32767 ? 32767 : t);
    int b = w < -128 ? -128 : (w > 127 ? 127 : w);
    return (uchar)(b + 128);
}

/* The same chain with the bias -32768 and one saturation to the signed 16 bit range. */
ushort sat_u16(float v) {
    int i = cvtt_i32_f(v + 0x1p-1f);
    int t = as_int(as_uint(i) + 0xffff8000u);
    int w = t < -32768 ? -32768 : (t > 32767 ? 32767 : t);
    return (ushort)(w + 32768);
}
"#;

/// Rust transcriptions of the prelude, one function per OpenCL C function, statement by statement.
/// The simulator runs kernel transcriptions built from these; the tests compare them with the CPU
/// helpers bit for bit.
pub mod rs {
    pub fn cvtt_i32_f(x: f32) -> i32 {
        if x >= -2147483648.0f32 && x < 2147483648.0f32 {
            return x as i32;
        }
        i32::MIN
    }

    pub fn cvtt_i64_f(x: f32) -> i64 {
        if x >= -9223372036854775808.0f32 && x < 9223372036854775808.0f32 {
            return x as i64;
        }
        i64::MIN
    }

    pub fn cvtt_i32_d(x: f64) -> i32 {
        if x > -2147483649.0f64 && x < 2147483648.0f64 {
            return x as i32;
        }
        i32::MIN
    }

    pub fn cvtt_i64_d(x: f64) -> i64 {
        if x >= -9223372036854775808.0f64 && x < 9223372036854775808.0f64 {
            return x as i64;
        }
        i64::MIN
    }

    pub fn q_round(x: f32) -> i64 {
        cvtt_i64_d(x as f64 + 0.5f64)
    }

    pub fn store_q_u8(x: f32) -> u8 {
        q_round(x) as u8
    }

    pub fn store_q_u16(x: f32) -> u16 {
        q_round(x) as u16
    }

    pub fn trunc_u8_f(v: f32) -> u8 {
        cvtt_i64_f(v) as u8
    }

    pub fn trunc_u16_f(v: f32) -> u16 {
        cvtt_i64_f(v) as u16
    }

    pub fn sat_u8(v: f32) -> u8 {
        let i = cvtt_i32_f(v + 0.5f32);
        let t = (i as u32).wrapping_add(0xffff_ff80u32) as i32;
        let w = if t < -32768 { -32768 } else if t > 32767 { 32767 } else { t };
        let b = if w < -128 { -128 } else if w > 127 { 127 } else { w };
        (b + 128) as u8
    }

    pub fn sat_u16(v: f32) -> u16 {
        let i = cvtt_i32_f(v + 0.5f32);
        let t = (i as u32).wrapping_add(0xffff_8000u32) as i32;
        let w = if t < -32768 { -32768 } else if t > 32767 { 32767 } else { t };
        (w + 32768) as u16
    }
}
