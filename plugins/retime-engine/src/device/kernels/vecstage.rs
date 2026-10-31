//! Kernel sources of the vector, matte and overlay stages: vector image coding, the pointwise matte
//! operations, the Debug view compositions, the Draw Geom discs and the tracking point splat, and
//! the per pixel parts of the luma and Edge Enhance prep. Their line for line Rust transcriptions
//! (one function per work item, statement by statement) are in `device::vecstage::items`.
//!
//! Identifier prefixes: every kernel and helper here starts with one of `vc_` (vector coding),
//! `mt_` (matte), `ov_` (overlays), `gm_` (geometry and points) or `pp_` (prep and filters), so the
//! executor can concatenate the sources of every stage module into one program. The sources build on
//! each other in the order of [`PARTS`]: every part uses only the prelude.
//!
//! Precision contracts (V: the CPU function named is the reference; every kernel keeps its
//! operation order and precision, f32 or f64 as the CPU writes it, no fused multiply add, the
//! prelude's range tested conversions for every float to integer step, hexadecimal literals):
//! * `vc_encode_*`: `vectors::encode_vectors` (the `warp::Flow::bilerp` sampler, `u * kx`, the
//!   clamp to `[-1, 1]` at 8 and 16 bpc, `+ 1.0f`, the f64 store `trunc((double)a * 127.0 + 0.5)`
//!   or `* 32767.0`, `(a + 1.0f) * 0.5f` in float); `sx`, `sy`, `kx`, `ky` arrive as arguments
//!   computed on the host by the CPU's own expressions.
//! * `vc_merge_*`, `vc_merge_flow`: `vectors::encode_vectors_merged` and `vectors::merge_flows`:
//!   per pixel the sum over at most [`MAX_LAYERS`] layers in their order from `0.0f`,
//!   `acc = acc + w * (u * kx)`, then the encoder's clamp and store (or the plain sum).
//!   `vc_blue_*`: the blue store of `vectors::encode_vectors_merged_blue` after the merge, in
//!   place (`q(w * 255.0f)`, `q(w * 65535.0f)`, `w` in float; create-vectors fix 03).
//! * `vc_decode_*`: `vectors::decode_vectors` (`A == 0` gives zero; integer depths in f64,
//!   `(float)((((double)c * 2.0) / den - 1.0) * scale)`; float `(c * 2.0f - 1.0f) * md`).
//! * `vc_holes_*`: `vectors::vector_hole_mask` (`white - A`), also `matte::alpha_holes`.
//! * `vc_vec_scale`: `vectors::apply_vec_scale` (`s * d`). `vc_zero_*`: `zero_vectors_image`.
//! * `mt_extract_*`: `matte::extract` (`clamp01(v / white)`). `mt_premultiply_*`:
//!   `matte::premultiply` (`q(c * m)`, `c * m` in float, `m` read through `mt_clamp_matte`).
//!   `mt_over_*`: `matte::over` (`smartblend::over` and the plain stores). `mt_effective`,
//!   `mt_over_weights`, `mt_stack_weights`: the products and visibilities of `matte::effective`
//!   (zero shrinks only, the choke needs a blur), `matte::over_weights` and `matte::stack_weights`
//!   (create-vectors fix 03: the loop starts at layer 0, so the FG layers above the group occlude
//!   it, and only the group's weights are stored), the layers in their order inside one work
//!   item, every matte value read through `mt_clamp_matte` before its first use, as the CPU's
//!   `matte::clamp_matte` (`x > 1` gives 1, `x >= 0` the value, `-0.0` included, anything else
//!   0, a NaN too).
//! * the Debug views of `debug.rs` (create-vectors fix 05): `ov_fill_*`: `debug::fill_transparent`
//!   (`W'`: a pixel with alpha 0, `-0.0` included, takes the pixel of `Y`; in place).
//!   `ov_blend_*`: branch 4 of `blend::blend` at the weight 0.5 (the Blend view). `ov_diff_*`:
//!   `debug::diff`, grey, at 8 and 16 bpc `min((|725 dR + 1475 dG + 300 dB| + 500) / 1000, white)`
//!   in 64 bit integers, in float `fabs((dR * 0.725f + dG * 1.475f) + dB * 0.3f)` of the f32
//!   differences; alpha white. `ov_overlay_*`: `debug::overlay`, red `r`, green and blue `g`
//!   (`q(v * 255.0f)`, `q(v * 65535.0f)`, `v` in float). `ov_luma_*`: `debug::debug_luma`, each
//!   colour sample times the f32 `1.0f / 255.0f` or `1.0f / 65535.0f` (1 in float), then
//!   `(r * 0.29f + g * 0.59f) + b * 0.12f`. `ov_detail`: `debug::detail`, `l / 0.7f` on the border,
//!   inside `sqrt(s / 8.0f) / 0.7f` with `s` summed from `0.0f` over the 8 neighbours in row order
//!   (`s = s + fabs(n - l)`); `ov_edge`: `debug::edge`, the Sobel sums of the luma with the
//!   coordinates clamped into the plane, each sum left to right, `sqrt((sx * sx + sy * sy) / 8.0f)
//!   / 0.7f`; both then `v < 1.0f ? v : 1.0f` (the CPU's `f32::min`, a NaN gives 1).
//! * `gm_draw_*`: `geom::draw_points` (the point boxes come from the host's `geom::span`; per
//!   pixel the last point in order whose disc covers it wins, which is the CPU's overwrite in
//!   point order). `gm_splat`: the splat of `points::user_displacement` (per pixel the points in
//!   order, `d = d + (from - to) * k`, `w = w + 1.0f`). `gm_seed`: the `d / w` start and the holes
//!   of `points::second_tracker`.
//! * `pp_luma_*`: `prep::luma` at every depth with its flags (f32 at 8 bpc, f64 at 16 bpc and in
//!   float, the video range branch in f32). `pp_edge_tail`: the gradient and per pixel map of
//!   `prep::edge_enhance` / `edge_enhance_with` after its blur (`calc_grad` or the circular
//!   `calc_grad_wrapx`).

/// The most layers one merge launch takes (FG1 to FG3 and Main_BG); the loops of `vc_merge_*`
/// are bounded by this literal.
pub const MAX_LAYERS: usize = 4;
/// The most FG mattes `mt_effective`, `mt_over_weights` and `mt_stack_weights` take.
pub const MAX_FG: usize = 3;
/// The most points one `gm_draw_*` or `gm_splat` launch takes (the effects have 12).
pub const MAX_POINTS: usize = 16;
/// Ints per point of `gm_draw_*`: the box `x0, x1, y0, y1`, the colour `c0, c1, c2` and a flag.
pub const DRAW_INTS: usize = 8;

/// The sources in program order (see the module documentation).
pub const PARTS: [&str; 5] = [MATTE, VECTORS, PREP, OVERLAY, GEOM];

pub const VECTORS: &str = r#"
/* clamp_unit of vectors.rs: v > 1 gives 1, else -1 > v gives -1; a NaN falls through. */
float vc_clamp_unit(float v) {
    if (v > 0x1p+0f) {
        return 0x1p+0f;
    }
    if (-0x1p+0f > v) {
        return -0x1p+0f;
    }
    return v;
}

/* The 8 bpc vector code of one value: clamp, + 1, trunc((double)a * 127.0 + 0.5), low byte. */
uchar vc_code_u8(float a) {
    float c = vc_clamp_unit(a);
    float d = c + 0x1p+0f;
    return (uchar)cvtt_i64_d((double)d * 0x1.fcp+6 + 0x1p-1);
}

/* The 16 bpc vector code: the same with 32767.0 and the low word. */
ushort vc_code_u16(float a) {
    float c = vc_clamp_unit(a);
    float d = c + 0x1p+0f;
    return (ushort)cvtt_i64_d((double)d * 0x1.fffcp+14 + 0x1p-1);
}

/* The float vector code: no clamp, (a + 1) * 0.5. */
float vc_code_f32(float a) {
    return (a + 0x1p+0f) * 0x1p-1f;
}

/* Flow::bilerp of warp.rs on a flow w pixels wide (w1 = w - 1, h1 = height - 1). */
void vc_bilerp(__global const float* d, uint w, int w1, int h1, float x, float y, float* u, float* v) {
    int ix = cvtt_i32_f(x);
    int iy = cvtt_i32_f(y);
    float fx = x - (float)ix;
    float fy = y - (float)iy;
    if (ix >= w1) {
        ix = w1 - 1;
        fx = 0x1p+0f;
    }
    if (iy >= h1) {
        iy = h1 - 1;
        fy = 0x1p+0f;
    }
    size_t i00 = ((size_t)iy * (size_t)w + (size_t)ix) * 2;
    size_t i01 = i00 + (size_t)w * 2;
    float out[2];
    for (int k = 0; k < 2; k++) {
        float p00 = d[i00 + k];
        float p10 = d[i00 + 2 + k];
        float p01 = d[i01 + k];
        float p11 = d[i01 + 2 + k];
        float top = (p10 - p00) * fx + p00;
        float bot = (p11 - p01) * fx + p01;
        out[k] = (bot - top) * fy + top;
    }
    *u = out[0];
    *v = out[1];
}

/* The scaled vector of encode_vectors at output pixel (x, y). */
void vc_sample(__global const float* flow, uint fw, int fw1, int fh1, size_t x, size_t y, float sx, float sy, float kx, float ky, float* a, float* b) {
    float u;
    float v;
    vc_bilerp(flow, fw, fw1, fh1, (float)x * sx, (float)y * sy, &u, &v);
    *a = u * kx;
    *b = v * ky;
}

__kernel void vc_encode_u8(__global const float* flow, uint fw, int fw1, int fh1, __global uchar* dst, uint dw, uint dh, float sx, float sy, float kx, float ky) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    if (x >= dw || y >= dh) {
        return;
    }
    float a;
    float b;
    vc_sample(flow, fw, fw1, fh1, x, y, sx, sy, kx, ky, &a, &b);
    size_t o = (y * dw + x) * 4;
    dst[o] = (uchar)255;
    dst[o + 1] = vc_code_u8(a);
    dst[o + 2] = vc_code_u8(b);
    dst[o + 3] = (uchar)0;
}

__kernel void vc_encode_u16(__global const float* flow, uint fw, int fw1, int fh1, __global ushort* dst, uint dw, uint dh, float sx, float sy, float kx, float ky) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    if (x >= dw || y >= dh) {
        return;
    }
    float a;
    float b;
    vc_sample(flow, fw, fw1, fh1, x, y, sx, sy, kx, ky, &a, &b);
    size_t o = (y * dw + x) * 4;
    dst[o] = (ushort)65535;
    dst[o + 1] = vc_code_u16(a);
    dst[o + 2] = vc_code_u16(b);
    dst[o + 3] = (ushort)0;
}

__kernel void vc_encode_f32(__global const float* flow, uint fw, int fw1, int fh1, __global float* dst, uint dw, uint dh, float sx, float sy, float kx, float ky) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    if (x >= dw || y >= dh) {
        return;
    }
    float a;
    float b;
    vc_sample(flow, fw, fw1, fh1, x, y, sx, sy, kx, ky, &a, &b);
    size_t o = (y * dw + x) * 4;
    dst[o] = 0x1p+0f;
    dst[o + 1] = vc_code_f32(a);
    dst[o + 2] = vc_code_f32(b);
    dst[o + 3] = 0x0p+0f;
}

/* merged_at of vectors.rs: the weighted sum of nl <= 4 layers in their order, from 0.0f. Per
   layer k, p holds sx, sy, kx, ky at 4 k and ip holds the flow width, w1 and h1 at 3 k. */
void vc_merged_at(__global const float* f0, __global const float* f1, __global const float* f2, __global const float* f3,
                  __global const float* w0, __global const float* w1, __global const float* w2, __global const float* w3,
                  __global const float* p, __global const int* ip, uint nl, size_t x, size_t y, uint width, float* ax, float* ay) {
    float sum_x = 0x0p+0f;
    float sum_y = 0x0p+0f;
    for (int k = 0; k < 4; k++) {
        if (k >= (int)nl) {
            break;
        }
        __global const float* f = k == 0 ? f0 : (k == 1 ? f1 : (k == 2 ? f2 : f3));
        __global const float* wt = k == 0 ? w0 : (k == 1 ? w1 : (k == 2 ? w2 : w3));
        float u;
        float v;
        vc_bilerp(f, (uint)ip[k * 3], ip[k * 3 + 1], ip[k * 3 + 2], (float)x * p[k * 4], (float)y * p[k * 4 + 1], &u, &v);
        float wk = wt[y * width + x];
        sum_x = sum_x + wk * (u * p[k * 4 + 2]);
        sum_y = sum_y + wk * (v * p[k * 4 + 3]);
    }
    *ax = sum_x;
    *ay = sum_y;
}

__kernel void vc_merge_u8(__global const float* f0, __global const float* f1, __global const float* f2, __global const float* f3,
                          __global const float* w0, __global const float* w1, __global const float* w2, __global const float* w3,
                          __global const float* p, __global const int* ip, uint nl, __global uchar* dst, uint dw, uint dh) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    if (x >= dw || y >= dh) {
        return;
    }
    float a;
    float b;
    vc_merged_at(f0, f1, f2, f3, w0, w1, w2, w3, p, ip, nl, x, y, dw, &a, &b);
    size_t o = (y * dw + x) * 4;
    dst[o] = (uchar)255;
    dst[o + 1] = vc_code_u8(a);
    dst[o + 2] = vc_code_u8(b);
    dst[o + 3] = (uchar)0;
}

__kernel void vc_merge_u16(__global const float* f0, __global const float* f1, __global const float* f2, __global const float* f3,
                           __global const float* w0, __global const float* w1, __global const float* w2, __global const float* w3,
                           __global const float* p, __global const int* ip, uint nl, __global ushort* dst, uint dw, uint dh) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    if (x >= dw || y >= dh) {
        return;
    }
    float a;
    float b;
    vc_merged_at(f0, f1, f2, f3, w0, w1, w2, w3, p, ip, nl, x, y, dw, &a, &b);
    size_t o = (y * dw + x) * 4;
    dst[o] = (ushort)65535;
    dst[o + 1] = vc_code_u16(a);
    dst[o + 2] = vc_code_u16(b);
    dst[o + 3] = (ushort)0;
}

__kernel void vc_merge_f32(__global const float* f0, __global const float* f1, __global const float* f2, __global const float* f3,
                           __global const float* w0, __global const float* w1, __global const float* w2, __global const float* w3,
                           __global const float* p, __global const int* ip, uint nl, __global float* dst, uint dw, uint dh) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    if (x >= dw || y >= dh) {
        return;
    }
    float a;
    float b;
    vc_merged_at(f0, f1, f2, f3, w0, w1, w2, w3, p, ip, nl, x, y, dw, &a, &b);
    size_t o = (y * dw + x) * 4;
    dst[o] = 0x1p+0f;
    dst[o + 1] = vc_code_f32(a);
    dst[o + 2] = vc_code_f32(b);
    dst[o + 3] = 0x0p+0f;
}

/* merge_flows: the plain sum, two channels per pixel. */
__kernel void vc_merge_flow(__global const float* f0, __global const float* f1, __global const float* f2, __global const float* f3,
                            __global const float* w0, __global const float* w1, __global const float* w2, __global const float* w3,
                            __global const float* p, __global const int* ip, uint nl, __global float* dst, uint dw, uint dh) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    if (x >= dw || y >= dh) {
        return;
    }
    float a;
    float b;
    vc_merged_at(f0, f1, f2, f3, w0, w1, w2, w3, p, ip, nl, x, y, dw, &a, &b);
    size_t o = (y * dw + x) * 2;
    dst[o] = a;
    dst[o + 1] = b;
}

/* encode_vectors_merged_blue after the merge, in place: the blue sample of every pixel holds the
   weight w of its pixel, q(w * 255) or q(w * 65535) (the low byte or word), w in float; the other
   samples are written back unchanged. */
__kernel void vc_blue_u8(__global uchar* dst, __global const float* blue, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t o = i * 4;
    for (int c = 0; c < 3; c++) {
        dst[o + c] = dst[o + c];
    }
    dst[o + 3] = store_q_u8(blue[i] * 0x1.fep+7f);
}

__kernel void vc_blue_u16(__global ushort* dst, __global const float* blue, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t o = i * 4;
    for (int c = 0; c < 3; c++) {
        dst[o + c] = dst[o + c];
    }
    dst[o + 3] = store_q_u16(blue[i] * 0x1.fffep+15f);
}

__kernel void vc_blue_f32(__global float* dst, __global const float* blue, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t o = i * 4;
    for (int c = 0; c < 3; c++) {
        dst[o + c] = dst[o + c];
    }
    dst[o + 3] = blue[i];
}

/* dec_int of vectors.rs: ((c * 2.0) / den - 1.0) * scale in double, one narrowing at the end. */
float vc_dec_int(float c, double den, double scale) {
    return (float)((((double)c * 0x1p+1) / den - 0x1p+0) * scale);
}

__kernel void vc_decode_u8(__global const uchar* vec, __global float* flow, double md, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t p = i * 4;
    if (vec[p] == 0) {
        flow[2 * i] = 0x0p+0f;
        flow[2 * i + 1] = 0x0p+0f;
    } else {
        flow[2 * i] = vc_dec_int((float)vec[p + 1], 0x1.fcp+7, md);
        flow[2 * i + 1] = vc_dec_int((float)vec[p + 2], 0x1.fcp+7, md);
    }
}

__kernel void vc_decode_u16(__global const ushort* vec, __global float* flow, double md, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t p = i * 4;
    if (vec[p] == 0) {
        flow[2 * i] = 0x0p+0f;
        flow[2 * i + 1] = 0x0p+0f;
    } else {
        flow[2 * i] = vc_dec_int((float)vec[p + 1], 0x1.fffcp+15, md);
        flow[2 * i + 1] = vc_dec_int((float)vec[p + 2], 0x1.fffcp+15, md);
    }
}

__kernel void vc_decode_f32(__global const float* vec, __global float* flow, float md, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t p = i * 4;
    if (vec[p] == 0x0p+0f) {
        flow[2 * i] = 0x0p+0f;
        flow[2 * i + 1] = 0x0p+0f;
    } else {
        flow[2 * i] = (vec[p + 1] * 0x1p+1f - 0x1p+0f) * md;
        flow[2 * i + 1] = (vec[p + 2] * 0x1p+1f - 0x1p+0f) * md;
    }
}

/* vector_hole_mask: white - A, one sample per pixel. */
__kernel void vc_holes_u8(__global const uchar* img, __global uchar* holes, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    holes[i] = (uchar)(255 - (int)img[i * 4]);
}

__kernel void vc_holes_u16(__global const ushort* img, __global ushort* holes, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    holes[i] = (ushort)(65535 - (int)img[i * 4]);
}

__kernel void vc_holes_f32(__global const float* img, __global float* holes, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    holes[i] = 0x1p+0f - img[i * 4];
}

/* apply_vec_scale: dx = sx * dx, dy = sy * dy, in place. */
__kernel void vc_vec_scale(__global float* flow, float sx, float sy, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    flow[2 * i] = sx * flow[2 * i];
    flow[2 * i + 1] = sy * flow[2 * i + 1];
}

/* zero_vectors_image: alpha white, R and G the centre, B 0. */
__kernel void vc_zero_u8(__global uchar* img, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    img[i * 4] = (uchar)255;
    img[i * 4 + 1] = (uchar)127;
    img[i * 4 + 2] = (uchar)127;
    img[i * 4 + 3] = (uchar)0;
}

__kernel void vc_zero_u16(__global ushort* img, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    img[i * 4] = (ushort)65535;
    img[i * 4 + 1] = (ushort)32767;
    img[i * 4 + 2] = (ushort)32767;
    img[i * 4 + 3] = (ushort)0;
}

__kernel void vc_zero_f32(__global float* img, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    img[i * 4] = 0x1p+0f;
    img[i * 4 + 1] = 0x1p-1f;
    img[i * 4 + 2] = 0x1p-1f;
    img[i * 4 + 3] = 0x0p+0f;
}
"#;

pub const MATTE: &str = r#"
/* clamp01 of matte.rs: above 1 gives 1, above 0 the value, anything else 0 (NaN and -0.0 too). */
float mt_clamp01(float x) {
    if (x > 0x1p+0f) {
        return 0x1p+0f;
    }
    if (x > 0x0p+0f) {
        return x;
    }
    return 0x0p+0f;
}

/* clamp_matte of matte.rs: above 1 gives 1, at least 0 the value (-0.0 kept), anything else 0
   (a negative value or a NaN). */
float mt_clamp_matte(float x) {
    if (x > 0x1p+0f) {
        return 0x1p+0f;
    }
    if (x >= 0x0p+0f) {
        return x;
    }
    return 0x0p+0f;
}

/* The plain stores of smartblend::Px: x + 0.5 saturated and truncated, x + 0.0 in float. */
uchar mt_store_plain_u8(float x) {
    float y = x + 0x1p-1f;
    if (0x0p+0f > y) {
        return (uchar)0;
    }
    if (y > 0x1.fep+7f) {
        return (uchar)255;
    }
    return trunc_u8_f(y);
}

ushort mt_store_plain_u16(float x) {
    float y = x + 0x1p-1f;
    if (0x0p+0f > y) {
        return (ushort)0;
    }
    if (y > 0x1.fffep+15f) {
        return (ushort)65535;
    }
    return trunc_u16_f(y);
}

/* smartblend::over: a over b, premultiplied [alpha, r, g, b], alpha in units of white. */
void mt_over4(const float* a, const float* b, float white, float* out) {
    if (a[0] == 0x0p+0f) {
        for (int c = 0; c < 4; c++) {
            out[c] = b[c];
        }
        return;
    }
    if (a[0] == white) {
        for (int c = 0; c < 4; c++) {
            out[c] = a[c];
        }
        return;
    }
    float ia = 0x1p+0f - a[0] / white;
    float alpha = (white - b[0]) * (a[0] / white) + b[0];
    out[0] = alpha;
    out[1] = b[1] * ia + a[1];
    out[2] = b[2] * ia + a[2];
    out[3] = b[3] * ia + a[3];
}

__kernel void mt_extract_u8(__global const uchar* img, __global float* m, uint c, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    m[i] = mt_clamp01((float)img[i * 4 + c] / 0x1.fep+7f);
}

__kernel void mt_extract_u16(__global const ushort* img, __global float* m, uint c, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    m[i] = mt_clamp01((float)img[i * 4 + c] / 0x1.fffep+15f);
}

__kernel void mt_extract_f32(__global const float* img, __global float* m, uint c, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    m[i] = mt_clamp01(img[i * 4 + c] / 0x1p+0f);
}

/* premultiply: every sample times the clamped matte of its pixel, q rounding at 8 and 16 bpc. */
__kernel void mt_premultiply_u8(__global const uchar* src, __global const float* m, __global uchar* out, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    float mv = mt_clamp_matte(m[i / 4]);
    out[i] = store_q_u8((float)src[i] * mv);
}

__kernel void mt_premultiply_u16(__global const ushort* src, __global const float* m, __global ushort* out, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    float mv = mt_clamp_matte(m[i / 4]);
    out[i] = store_q_u16((float)src[i] * mv);
}

__kernel void mt_premultiply_f32(__global const float* src, __global const float* m, __global float* out, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    float mv = mt_clamp_matte(m[i / 4]);
    out[i] = src[i] * mv;
}

/* over: src OVER dest in place, one pixel per work item. */
__kernel void mt_over_u8(__global uchar* dest, __global const uchar* src, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t o = i * 4;
    float a[4];
    float b[4];
    float v[4];
    for (int c = 0; c < 4; c++) {
        a[c] = (float)src[o + c];
        b[c] = (float)dest[o + c];
    }
    mt_over4(a, b, 0x1.fep+7f, v);
    for (int c = 0; c < 4; c++) {
        dest[o + c] = mt_store_plain_u8(v[c]);
    }
}

__kernel void mt_over_u16(__global ushort* dest, __global const ushort* src, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t o = i * 4;
    float a[4];
    float b[4];
    float v[4];
    for (int c = 0; c < 4; c++) {
        a[c] = (float)src[o + c];
        b[c] = (float)dest[o + c];
    }
    mt_over4(a, b, 0x1.fffep+15f, v);
    for (int c = 0; c < 4; c++) {
        dest[o + c] = mt_store_plain_u16(v[c]);
    }
}

__kernel void mt_over_f32(__global float* dest, __global const float* src, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t o = i * 4;
    float a[4];
    float b[4];
    float v[4];
    for (int c = 0; c < 4; c++) {
        a[c] = src[o + c];
        b[c] = dest[o + c];
    }
    mt_over4(a, b, 0x1p+0f, v);
    for (int c = 0; c < 4; c++) {
        dest[o + c] = v[c] + 0x0p+0f;
    }
}

/* effective with zero shrinks: for the nfg <= 3 FG mattes in order, with m_k the clamped matte
   (clamp_matte), entry k = m_k * vis, then vis = clamp01((1 - m_k) * vis); entry nfg = vis. */
__kernel void mt_effective(__global const float* m0, __global const float* m1, __global const float* m2,
                           __global float* e0, __global float* e1, __global float* e2, __global float* e3, uint nfg, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    float vis = 0x1p+0f;
    for (int k = 0; k < 3; k++) {
        if (k >= (int)nfg) {
            break;
        }
        __global const float* m = k == 0 ? m0 : (k == 1 ? m1 : m2);
        __global float* e = k == 0 ? e0 : (k == 1 ? e1 : e2);
        float mk = mt_clamp_matte(m[i]);
        e[i] = mk * vis;
        float inv = 0x1p+0f - mk;
        vis = mt_clamp01(inv * vis);
    }
    __global float* last = nfg == 0 ? e0 : (nfg == 1 ? e1 : (nfg == 2 ? e2 : e3));
    last[i] = vis;
}

/* over_weights: for k in first..=last (nfg <= 3, last <= nfg), with m_k the clamped matte
   (clamp_matte), out entry k - first is m_k * vis (vis itself for Main_BG, k == nfg), then
   vis = clamp01((1 - m_k) * vis) while k < last. */
__kernel void mt_over_weights(__global const float* m0, __global const float* m1, __global const float* m2,
                              __global float* o0, __global float* o1, __global float* o2, __global float* o3,
                              uint first, uint last, uint nfg, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    float vis = 0x1p+0f;
    for (int k = 0; k < 4; k++) {
        if (k < (int)first) {
            continue;
        }
        if (k > (int)last) {
            break;
        }
        int j = k - (int)first;
        __global float* o = j == 0 ? o0 : (j == 1 ? o1 : (j == 2 ? o2 : o3));
        if (k == (int)nfg) {
            o[i] = vis;
            break;
        }
        __global const float* m = k == 0 ? m0 : (k == 1 ? m1 : m2);
        float mk = mt_clamp_matte(m[i]);
        o[i] = mk * vis;
        if (k < (int)last) {
            float inv = 0x1p+0f - mk;
            vis = mt_clamp01(inv * vis);
        }
    }
}

/* matte::stack_weights: the OVER weights of the layers first..=last inside the whole stack. The
   loop starts at layer 0, so the FG layers above first occlude the group too; only the group's
   weights are stored. For first == 0 it is mt_over_weights. */
__kernel void mt_stack_weights(__global const float* m0, __global const float* m1, __global const float* m2,
                               __global float* o0, __global float* o1, __global float* o2, __global float* o3,
                               uint first, uint last, uint nfg, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    float vis = 0x1p+0f;
    for (int k = 0; k < 4; k++) {
        if (k > (int)last) {
            break;
        }
        int j = k - (int)first;
        if (k == (int)nfg) {
            __global float* o = j == 0 ? o0 : (j == 1 ? o1 : (j == 2 ? o2 : o3));
            o[i] = vis;
            break;
        }
        __global const float* m = k == 0 ? m0 : (k == 1 ? m1 : m2);
        float mk = mt_clamp_matte(m[i]);
        if (k >= (int)first) {
            __global float* o = j == 0 ? o0 : (j == 1 ? o1 : (j == 2 ? o2 : o3));
            o[i] = mk * vis;
        }
        if (k < (int)last) {
            float inv = 0x1p+0f - mk;
            vis = mt_clamp01(inv * vis);
        }
    }
}
"#;

pub const PREP: &str = r#"
/* The gradient of flow::calc_grad at (x, y) of a w x h plane, one sided at the edges; with wrap
   the horizontal difference of prep's calc_grad_wrapx (columns modulo w, always two terms). */
void pp_grad(__global const float* img, size_t x, size_t y, size_t w, size_t h, int wrap, float* gx, float* gy) {
    size_t i = y * w + x;
    float c = img[i];
    float nx = 0x0p+0f;
    float dx = 0x0p+0f;
    float ny = 0x0p+0f;
    float dy = 0x0p+0f;
    if (wrap != 0) {
        float left = img[y * w + (x + w - 1) % w];
        float right = img[y * w + (x + 1) % w];
        nx = ((c - left) + 0x0p+0f) + (right - c);
        dx = 0x1p+0f + 0x1p+0f;
    } else {
        if (x >= 1) {
            nx = (c - img[i - 1]) + 0x0p+0f;
            dx = 0x1p+0f;
        }
    }
    if (y >= 1) {
        ny = (c - img[i - w]) + 0x0p+0f;
        dy = 0x1p+0f;
    }
    if (wrap == 0) {
        if (x + 1 <= w - 1) {
            nx = nx + (img[i + 1] - c);
            dx = dx + 0x1p+0f;
        }
    }
    if (y + 1 <= h - 1) {
        ny = ny + (img[i + w] - c);
        dy = dy + 0x1p+0f;
    }
    *gx = nx / dx;
    *gy = ny / dy;
}

/* The edge measure of the Contrast / Edge Enhance prep: m <= 20 ? m * 0.05 : 1 of the gradient
   magnitude. */
float pp_edge_value(float gx, float gy) {
    float m = sqrt(gx * gx + gy * gy);
    return m <= 0x1.4p+4f ? m * 0x1.99999ap-5f : 0x1p+0f;
}

/* The tail of edge_enhance after its blur: p = (v * 255 + b) * 0.5 of the blurred plane b. */
__kernel void pp_edge_tail(__global const float* b, __global float* out, uint w, uint h, int wrap) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    if (x >= w || y >= h) {
        return;
    }
    float gx;
    float gy;
    pp_grad(b, x, y, (size_t)w, (size_t)h, wrap, &gx, &gy);
    float f = pp_edge_value(gx, gy);
    size_t i = y * w + x;
    out[i] = (f * 0x1.fep+7f + b[i]) * 0x1p-1f;
}

/* prep::luma at 8 bpc: f32 weights, the alpha term widened to f64, the video range in f32. */
__kernel void pp_luma_u8(__global const uchar* s, __global float* out, int alpha, int video, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t p = i * 4;
    float a = (float)s[p];
    float r = (float)s[p + 1];
    float g = (float)s[p + 2];
    float b = (float)s[p + 3];
    float v;
    if (video != 0) {
        v = ((r - 0x1p+4f) * 0x1.fep+7f) / 0x1.b6p+7f;
    } else {
        float sum = (r * 0x1.333334p-2f + g * 0x1.2e147ap-1f) + b * 0x1.c28f5cp-4f;
        if (alpha != 0) {
            v = (float)((double)sum + (double)a * 0x1.3333333333333p-2);
        } else {
            v = sum;
        }
    }
    out[i] = v;
}

/* prep::luma at 16 bpc: f64 weights and the division by 257.0, the video range in f32. */
__kernel void pp_luma_u16(__global const ushort* s, __global float* out, int alpha, int video, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t p = i * 4;
    double a = (double)s[p];
    double r = (double)s[p + 1];
    double g = (double)s[p + 2];
    double b = (double)s[p + 3];
    float v;
    if (video != 0) {
        v = ((((float)s[p + 1]) / 0x1.01p+8f - 0x1p+4f) * 0x1.fep+7f) / 0x1.b6p+7f;
    } else {
        double sum = (g * 0x1.2e147ae147ae1p-1 + r * 0x1.3333333333333p-2) + b * 0x1.c28f5c28f5c29p-4;
        if (alpha != 0) {
            sum = sum + a * 0x1.3333333333333p-2;
        }
        v = (float)(sum / 0x1.01p+8);
    }
    out[i] = v;
}

/* prep::luma in float: f64 weights times 255.0, the video range in f32. */
__kernel void pp_luma_f32(__global const float* s, __global float* out, int alpha, int video, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t p = i * 4;
    double a = (double)s[p];
    double r = (double)s[p + 1];
    double g = (double)s[p + 2];
    double b = (double)s[p + 3];
    float v;
    if (video != 0) {
        v = ((s[p + 1] * 0x1.fep+7f - 0x1p+4f) * 0x1.fep+7f) / 0x1.b6p+7f;
    } else {
        double sum = (g * 0x1.2e147ae147ae1p-1 + r * 0x1.3333333333333p-2) + b * 0x1.c28f5c28f5c29p-4;
        if (alpha != 0) {
            sum = sum + a * 0x1.3333333333333p-2;
        }
        v = (float)(sum * 0x1.fep+7);
    }
    out[i] = v;
}
"#;

pub const OVERLAY: &str = r#"
/* W' of the Debug views (debug::fill_transparent), in place: a pixel of w whose alpha is 0 (-0.0
   included) takes the pixel of y; the others are written back unchanged. */
__kernel void ov_fill_u8(__global uchar* w, __global const uchar* y, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t o = i * 4;
    if (w[o] == 0) {
        for (int c = 0; c < 4; c++) {
            w[o + c] = y[o + c];
        }
        return;
    }
    for (int c = 0; c < 4; c++) {
        w[o + c] = w[o + c];
    }
}

__kernel void ov_fill_u16(__global ushort* w, __global const ushort* y, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t o = i * 4;
    if (w[o] == 0) {
        for (int c = 0; c < 4; c++) {
            w[o + c] = y[o + c];
        }
        return;
    }
    for (int c = 0; c < 4; c++) {
        w[o + c] = w[o + c];
    }
}

__kernel void ov_fill_f32(__global float* w, __global const float* y, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t o = i * 4;
    if (w[o] == 0x0p+0f) {
        for (int c = 0; c < 4; c++) {
            w[o + c] = y[o + c];
        }
        return;
    }
    for (int c = 0; c < 4; c++) {
        w[o + c] = w[o + c];
    }
}

/* The Blend view: branch 4 of blend::blend at the weight 0.5, then the blend store. */
__kernel void ov_blend_u8(__global const uchar* a, __global const uchar* b, __global uchar* out, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    float af = (float)a[i];
    float bf = (float)b[i];
    out[i] = trunc_u8_f((bf - af) * 0x1p-1f + af + 0x1p-1f);
}

__kernel void ov_blend_u16(__global const ushort* a, __global const ushort* b, __global ushort* out, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    float af = (float)a[i];
    float bf = (float)b[i];
    out[i] = trunc_u16_f((bf - af) * 0x1p-1f + af + 0x1p-1f);
}

__kernel void ov_blend_f32(__global const float* a, __global const float* b, __global float* out, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    out[i] = (b[i] - a[i]) * 0x1p-1f + a[i] + 0x0p+0f;
}

/* diff_int of debug.rs: |725 dR + 1475 dG + 300 dB| / 1000 rounded half up, at most top, exact in
   64 bit integers. */
long ov_diff_int(long dr, long dg, long db, long top) {
    long s = 725 * dr + 1475 * dg + 300 * db;
    long v = s < 0 ? -s : s;
    long r = (v + 500) / 1000;
    return r < top ? r : top;
}

/* diff_value of debug.rs: |(dR * 0.725 + dG * 1.475) + dB * 0.3| in f32. */
float ov_diff_value(float dr, float dg, float db) {
    return fabs((dr * 0x1.733334p-1f + dg * 0x1.79999ap+0f) + db * 0x1.333334p-2f);
}

/* The Diff view (debug::diff): the grey value of the channel differences of w and y in R, G and
   B, alpha white. */
__kernel void ov_diff_u8(__global const uchar* a, __global const uchar* b, __global uchar* out, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t o = i * 4;
    long dr = (long)a[o + 1] - (long)b[o + 1];
    long dg = (long)a[o + 2] - (long)b[o + 2];
    long db = (long)a[o + 3] - (long)b[o + 3];
    uchar v = (uchar)ov_diff_int(dr, dg, db, 255);
    out[o] = (uchar)255;
    out[o + 1] = v;
    out[o + 2] = v;
    out[o + 3] = v;
}

__kernel void ov_diff_u16(__global const ushort* a, __global const ushort* b, __global ushort* out, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t o = i * 4;
    long dr = (long)a[o + 1] - (long)b[o + 1];
    long dg = (long)a[o + 2] - (long)b[o + 2];
    long db = (long)a[o + 3] - (long)b[o + 3];
    ushort v = (ushort)ov_diff_int(dr, dg, db, 65535);
    out[o] = (ushort)65535;
    out[o + 1] = v;
    out[o + 2] = v;
    out[o + 3] = v;
}

__kernel void ov_diff_f32(__global const float* a, __global const float* b, __global float* out, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t o = i * 4;
    float v = ov_diff_value(a[o + 1] - b[o + 1], a[o + 2] - b[o + 2], a[o + 3] - b[o + 3]);
    out[o] = 0x1p+0f;
    out[o + 1] = v;
    out[o + 2] = v;
    out[o + 3] = v;
}

/* The red/cyan overlay (debug::overlay): alpha white, red r, green and blue g. */
__kernel void ov_overlay_u8(__global const float* r, __global const float* g, __global uchar* out, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t o = i * 4;
    uchar gv = store_q_u8(g[i] * 0x1.fep+7f);
    out[o] = (uchar)255;
    out[o + 1] = store_q_u8(r[i] * 0x1.fep+7f);
    out[o + 2] = gv;
    out[o + 3] = gv;
}

__kernel void ov_overlay_u16(__global const float* r, __global const float* g, __global ushort* out, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t o = i * 4;
    ushort gv = store_q_u16(g[i] * 0x1.fffep+15f);
    out[o] = (ushort)65535;
    out[o + 1] = store_q_u16(r[i] * 0x1.fffep+15f);
    out[o + 2] = gv;
    out[o + 3] = gv;
}

__kernel void ov_overlay_f32(__global const float* r, __global const float* g, __global float* out, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t o = i * 4;
    out[o] = 0x1p+0f;
    out[o + 1] = r[i];
    out[o + 2] = g[i];
    out[o + 3] = g[i];
}

/* The luma of the Debug views (debug::debug_luma) from colour samples scaled to 0..1. */
float ov_lum(float r, float g, float b) {
    return (r * 0x1.28f5c2p-2f + g * 0x1.2e147ap-1f) + b * 0x1.eb851ep-4f;
}

__kernel void ov_luma_u8(__global const uchar* s, __global float* out, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t p = i * 4;
    float k = 0x1.010102p-8f;
    out[i] = ov_lum((float)s[p + 1] * k, (float)s[p + 2] * k, (float)s[p + 3] * k);
}

__kernel void ov_luma_u16(__global const ushort* s, __global float* out, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t p = i * 4;
    float k = 0x1.0001p-16f;
    out[i] = ov_lum((float)s[p + 1] * k, (float)s[p + 2] * k, (float)s[p + 3] * k);
}

__kernel void ov_luma_f32(__global const float* s, __global float* out, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t p = i * 4;
    out[i] = ov_lum(s[p + 1], s[p + 2], s[p + 3]);
}

/* The detail of a luma plane (debug::detail): l / 0.7 on the border, inside sqrt(s / 8) / 0.7 of
   the sum s of |l(n) - l(p)| over the 8 neighbours n in row order, then at most 1. */
__kernel void ov_detail(__global const float* l, __global float* out, uint w, uint h) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    if (x >= w || y >= h) {
        return;
    }
    size_t wd = (size_t)w;
    float c = l[y * wd + x];
    float d;
    if (x == 0 || y == 0 || x + 1 == wd || y + 1 == (size_t)h) {
        d = c / 0x1.666666p-1f;
    } else {
        float s = 0x0p+0f;
        for (int r = 0; r < 3; r++) {
            for (int k = 0; k < 3; k++) {
                size_t yy = y + (size_t)r - 1;
                size_t xx = x + (size_t)k - 1;
                if (yy != y || xx != x) {
                    s = s + fabs(l[yy * wd + xx] - c);
                }
            }
        }
        d = sqrt(s / 0x1p+3f) / 0x1.666666p-1f;
    }
    out[y * wd + x] = d < 0x1p+0f ? d : 0x1p+0f;
}

/* The luma at (x + dx, y + dy) with the coordinates clamped into the w x h plane. */
float ov_at(__global const float* l, size_t x, size_t y, int dy, int dx, uint w, uint h) {
    long yy = (long)y + (long)dy;
    long xx = (long)x + (long)dx;
    long ym = (long)h - 1;
    long xm = (long)w - 1;
    yy = yy < 0 ? 0 : (yy > ym ? ym : yy);
    xx = xx < 0 ? 0 : (xx > xm ? xm : xx);
    return l[(size_t)yy * (size_t)w + (size_t)xx];
}

/* The edge measure of a luma plane (debug::edge): the 3 x 3 Sobel sums with the border repeated
   outwards, sqrt((sx * sx + sy * sy) / 8) / 0.7, then at most 1. */
__kernel void ov_edge(__global const float* l, __global float* out, uint w, uint h) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    if (x >= w || y >= h) {
        return;
    }
    float sx = (ov_at(l, x, y, -1, 1, w, h) + 0x1p+1f * ov_at(l, x, y, 0, 1, w, h) + ov_at(l, x, y, 1, 1, w, h))
        - (ov_at(l, x, y, -1, -1, w, h) + 0x1p+1f * ov_at(l, x, y, 0, -1, w, h) + ov_at(l, x, y, 1, -1, w, h));
    float sy = (ov_at(l, x, y, 1, -1, w, h) + 0x1p+1f * ov_at(l, x, y, 1, 0, w, h) + ov_at(l, x, y, 1, 1, w, h))
        - (ov_at(l, x, y, -1, -1, w, h) + 0x1p+1f * ov_at(l, x, y, -1, 0, w, h) + ov_at(l, x, y, -1, 1, w, h));
    float m = sqrt((sx * sx + sy * sy) / 0x1p+3f) / 0x1.666666p-1f;
    out[y * (size_t)w + x] = m < 0x1p+0f ? m : 0x1p+0f;
}
"#;

pub const GEOM: &str = r#"
/* Draw Geom: per pixel the last point in order (n <= 16) whose box holds the pixel and whose disc
   dx * dx + dy * dy <= r2 covers it; p holds px, py at 2 k, ip the box x0, x1, y0, y1, the
   colour and a flag at 8 k. An uncovered pixel is written back unchanged. */
int gm_hit(__global const float* p, __global const int* ip, uint n, float r2, size_t x, size_t y) {
    int hit = -1;
    for (int k = 0; k < 16; k++) {
        if (k >= (int)n) {
            break;
        }
        if (ip[k * 8 + 7] == 0) {
            continue;
        }
        if ((int)x < ip[k * 8] || (int)x > ip[k * 8 + 1] || (int)y < ip[k * 8 + 2] || (int)y > ip[k * 8 + 3]) {
            continue;
        }
        float dy = (float)y - p[k * 2 + 1];
        float dx = (float)x - p[k * 2];
        if (dx * dx + dy * dy <= r2) {
            hit = k;
        }
    }
    return hit;
}

__kernel void gm_draw_u8(__global uchar* img, __global const float* p, __global const int* ip, uint n, float r2, uint w, uint h) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    if (x >= w || y >= h) {
        return;
    }
    size_t o = (y * w + x) * 4;
    int k = gm_hit(p, ip, n, r2, x, y);
    if (k < 0) {
        for (int c = 0; c < 4; c++) {
            img[o + c] = img[o + c];
        }
        return;
    }
    img[o] = (uchar)255;
    img[o + 1] = (uchar)ip[k * 8 + 4];
    img[o + 2] = (uchar)ip[k * 8 + 5];
    img[o + 3] = (uchar)ip[k * 8 + 6];
}

__kernel void gm_draw_u16(__global ushort* img, __global const float* p, __global const int* ip, uint n, float r2, uint w, uint h) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    if (x >= w || y >= h) {
        return;
    }
    size_t o = (y * w + x) * 4;
    int k = gm_hit(p, ip, n, r2, x, y);
    if (k < 0) {
        for (int c = 0; c < 4; c++) {
            img[o + c] = img[o + c];
        }
        return;
    }
    img[o] = (ushort)65535;
    img[o + 1] = (ushort)(ip[k * 8 + 4] * 257);
    img[o + 2] = (ushort)(ip[k * 8 + 5] * 257);
    img[o + 3] = (ushort)(ip[k * 8 + 6] * 257);
}

__kernel void gm_draw_f32(__global float* img, __global const float* p, __global const int* ip, uint n, float r2, uint w, uint h) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    if (x >= w || y >= h) {
        return;
    }
    size_t o = (y * w + x) * 4;
    int k = gm_hit(p, ip, n, r2, x, y);
    if (k < 0) {
        for (int c = 0; c < 4; c++) {
            img[o + c] = img[o + c];
        }
        return;
    }
    img[o] = 0x1p+0f;
    img[o + 1] = (float)ip[k * 8 + 4] / 0x1.fep+7f;
    img[o + 2] = (float)ip[k * 8 + 5] / 0x1.fep+7f;
    img[o + 3] = (float)ip[k * 8 + 6] / 0x1.fep+7f;
}

/* The splat of points::user_displacement into one level of wl x hl: per pixel the n <= 16 pairs
   in order (p holds from x, from y, to x, to y at 4 k); a pair whose from is outside the output
   frame (ow, oh as floats) or whose to is not finite is skipped; a pair landing on the pixel adds
   (from - to) * k to d and 1.0 to w. */
__kernel void gm_splat(__global const float* p, uint n, float kx, float ky, float ow, float oh, __global float* d, __global float* wt, uint wl, uint hl) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    if (x >= wl || y >= hl) {
        return;
    }
    float sx = 0x0p+0f;
    float sy = 0x0p+0f;
    float sw = 0x0p+0f;
    for (int k = 0; k < 16; k++) {
        if (k >= (int)n) {
            break;
        }
        float fx = p[k * 4];
        float fy = p[k * 4 + 1];
        float tx = p[k * 4 + 2];
        float ty = p[k * 4 + 3];
        float ax = fx + 0x1p-1f;
        float ay = fy + 0x1p-1f;
        int inside = ax >= 0x0p+0f && ax < ow && ay >= 0x0p+0f && ay < oh;
        if (!inside || !(isfinite(tx) && isfinite(ty))) {
            continue;
        }
        int ix = cvtt_i32_f(fx * kx + 0x1p-1f);
        int iy = cvtt_i32_f(fy * ky + 0x1p-1f);
        if (ix < 0 || iy < 0 || ix >= (int)wl || iy >= (int)hl) {
            continue;
        }
        if ((size_t)ix != x || (size_t)iy != y) {
            continue;
        }
        sx = sx + (fx - tx) * kx;
        sy = sy + (fy - ty) * ky;
        sw = sw + 0x1p+0f;
    }
    size_t i = y * wl + x;
    d[2 * i] = sx;
    d[2 * i + 1] = sy;
    wt[i] = sw;
}

/* The start of points::second_tracker from a level 0 splat: d / w where a point landed, else a
   zero vector and a hole (255). */
__kernel void gm_seed(__global const float* d, __global const float* wt, __global float* flow, __global uchar* holes, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    float w = wt[i];
    if (w != 0x0p+0f) {
        flow[2 * i] = d[2 * i] / w;
        flow[2 * i + 1] = d[2 * i + 1] / w;
        holes[i] = (uchar)0;
    } else {
        flow[2 * i] = 0x0p+0f;
        flow[2 * i + 1] = 0x0p+0f;
        holes[i] = (uchar)255;
    }
}
"#;
