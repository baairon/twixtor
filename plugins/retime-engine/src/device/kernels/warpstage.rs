//! Kernel sources of the per pixel composition stages: the blend primitives of
//! `blend.rs`, the inverse warp of `warp.rs` with its samplers, the motion blur of
//! `warp/blurcomp.rs` and the Smart Blend composites of `smartblend.rs`. Their Rust transcriptions
//! (statement by statement) are `device::warpstage::rs`; the stage wrappers are
//! `device::warpstage`.
//!
//! Identifier prefixes (every function, kernel and macro defined here, so the sources of all
//! stage modules can be concatenated into one program): `bl_` / `BL_` blend, `wp_` / `WP_` inverse
//! warp, `bc_` / `BC_` blur compensation, `sb_` / `SB_` Smart Blend. The three depths share one
//! body through a preprocessor macro per source, instantiated for `u8` (uchar), `u16` (ushort)
//! and `f32` (float); the macro arguments are the depth's stores, as the `Pix` and `Px` traits of
//! the CPU modules are.
//!
//! Precision contracts (V: the CPU function named is the reference, bit for bit):
//! * `bl_blend_*`: branch 4 of `blend::blend` (the host takes branches 1 to 3); `bl_lerp_*`: the
//!   row kernels of `blend::lerp`, in place on the first image; `bl_plus_zero_f32` and
//!   `bl_unsound_f32`: the `+ 0.0f` copies and the `sound_float` test of
//!   `blend::weighted_blend_noflow`.
//! * `wp_inv_warp_*`: the row kernel of `warp::inv_warp` / `inv_warp_coverage` (hold, fade with
//!   coverage, 360 wrap), with `clip_to_rect`, `bilerp4`, `bilerp360` and the flow sampler. The
//!   host takes the early copy outs and the argument checks.
//! * `bc_motion_blur_*`: `blurcomp::motion_blur` (bilinear and nearest taps, plain and 360);
//!   `bc_clear_rgb_*`: `clear_rgb_where_transparent`, from one buffer into another.
//! * `sb_smart_*`: `smartblend::smart_blend` and `smart_blend_binary`; `sb_cov_*`: the coverage
//!   combine and invert of `smart_finish` and `nearest_smart`.
//!
//! Bounded loops. Every loop has a constant bound. Where the CPU loops for a count that depends
//! on the data (the clip of `clip_to_rect` up to 4096 passes, the 360 wrap loops up to 65536
//! steps, the motion blur taps up to 65536), the kernel loops at most a smaller constant
//! (`WP_CLIP_PASSES` 64, `WP_WRAP_STEPS` 64 per loop, `BC_TAPS` 1024 taps per direction) and
//! raises bit 2 of the flag word when that is not enough; the wrapper then refuses the stage
//! (`StageError::Unsupported`) and the CPU runs it. Inside those bounds the iterations are the
//! CPU's. Bit 1 of the flag word marks the states the CPU itself refuses (a NaN sample point in
//! hold mode, a 360 coordinate that never wraps, a NaN bilinear tap, more than 65536 taps).

/// Passes of the clip loop on the device (the CPU allows 4096).
pub const WP_CLIP_PASSES: u32 = 64;
/// Steps of each 360 wrap loop on the device (the CPU allows 65536 in total).
pub const WP_WRAP_STEPS: u32 = 64;
/// Motion blur taps per direction on the device (the CPU allows 65536).
pub const BC_TAPS: i32 = 1024;
/// Flag bit: a state the CPU refuses too (its result would be an `Unsupported` error).
pub const FLAG_CPU_REFUSES: i32 = 1;
/// Flag bit: a loop needed more passes than the device bound; the CPU must run the stage.
pub const FLAG_DEVICE_BOUND: i32 = 2;

pub const BLEND: &str = r#"
/* branch 4 of blend::blend: subtract, multiply, add, then the depth's store */
#define BL_BLEND_INT(SUF, T, STORE) \
__kernel void bl_blend_##SUF(__global const T* a, __global const T* b, __global T* out, float w, uint n) { \
    size_t i = get_global_id(0); \
    if (i >= n) { \
        return; \
    } \
    float af = (float)a[i]; \
    float bf = (float)b[i]; \
    out[i] = STORE((bf - af) * w + af + 0x1p-1f); \
}

BL_BLEND_INT(u8, uchar, trunc_u8_f)
BL_BLEND_INT(u16, ushort, trunc_u16_f)

__kernel void bl_blend_f32(__global const float* a, __global const float* b, __global float* out, float w, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    out[i] = (b[i] - a[i]) * w + a[i] + 0x0p+0f;
}

/* the integer row kernel of blend::lerp, one pixel per work item, in place on f */
#define BL_LERP_INT(SUF, T, SCALE, STORE) \
__kernel void bl_lerp_##SUF(__global T* f, __global const T* s, float w, uint n) { \
    size_t p = get_global_id(0); \
    if (p >= n) { \
        return; \
    } \
    size_t o = p * 4; \
    int f0 = (int)f[o]; \
    int f1 = (int)f[o + 1]; \
    int f2 = (int)f[o + 2]; \
    int f3 = (int)f[o + 3]; \
    int s0 = (int)s[o]; \
    int s1 = (int)s[o + 1]; \
    int s2 = (int)s[o + 2]; \
    int s3 = (int)s[o + 3]; \
    float a = (float)(f0 - s0) * w + (float)s0; \
    float nn = a / SCALE; \
    if (!(nn > 0x0p+0f)) { \
        T z = STORE(0x0p+0f); \
        f[o] = z; \
        f[o + 1] = z; \
        f[o + 2] = z; \
        f[o + 3] = z; \
        return; \
    } \
    float r = (float)(f1 - s1) * w + (float)s1; \
    float g = (float)(f2 - s2) * w + (float)s2; \
    float b = (float)(f3 - s3) * w + (float)s3; \
    if (nn > 0x1p+0f) { \
        r = r / nn; \
        g = g / nn; \
        b = b / nn; \
        a = SCALE; \
    } \
    r = r > a ? a : (0x0p+0f > r ? 0x0p+0f : r); \
    g = g > a ? a : (0x0p+0f > g ? 0x0p+0f : g); \
    b = b > a ? a : (0x0p+0f > b ? 0x0p+0f : b); \
    f[o + 1] = STORE(r + 0x1p-1f); \
    f[o + 2] = STORE(g + 0x1p-1f); \
    f[o + 3] = STORE(b + 0x1p-1f); \
    f[o] = STORE(a + 0x1p-1f); \
}

BL_LERP_INT(u8, uchar, 0x1.fep+7f, trunc_u8_f)
BL_LERP_INT(u16, ushort, 0x1.fffep+15f, trunc_u16_f)

/* the float row kernel of blend::lerp */
__kernel void bl_lerp_f32(__global float* f, __global const float* s, float w, uint n) {
    size_t p = get_global_id(0);
    if (p >= n) {
        return;
    }
    size_t o = p * 4;
    float f0 = f[o];
    float f1 = f[o + 1];
    float f2 = f[o + 2];
    float f3 = f[o + 3];
    float a = (f0 - s[o]) * w + s[o];
    if (!(a > 0x0p+0f)) {
        f[o] = 0x0p+0f;
        f[o + 1] = 0x0p+0f;
        f[o + 2] = 0x0p+0f;
        f[o + 3] = 0x0p+0f;
        return;
    }
    float r = (f1 - s[o + 1]) * w + s[o + 1];
    float g = (f2 - s[o + 2]) * w + s[o + 2];
    float b = (f3 - s[o + 3]) * w + s[o + 3];
    if (a > 0x1p+0f) {
        r = r / a;
        g = g / a;
        b = b / a;
        a = 0x1p+0f;
    }
    f[o + 1] = r + 0x0p+0f;
    f[o + 2] = g + 0x0p+0f;
    f[o + 3] = b + 0x0p+0f;
    f[o] = a + 0x0p+0f;
}

/* the + 0.0f copy of weighted_blend_noflow */
__kernel void bl_plus_zero_f32(__global const float* src, __global float* out, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    out[i] = src[i] + 0x0p+0f;
}

/* sound_float of weighted_blend_noflow: bit 1 of flag when |v| <= 1e30f fails (NaN too) */
__kernel void bl_unsound_f32(__global const float* src, __global int* flag, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    if (!(fabs(src[i]) <= 0x1.93e594p+99f)) {
        atomic_or(flag, 1);
    }
}
"#;

pub const WARP: &str = r#"
/* the saturating stores of warp.rs (Pix::store, Pix::store_faded) */
uchar wp_store_u8(float v) {
    return sat_u8(v);
}

ushort wp_store_u16(float v) {
    return sat_u16(v);
}

float wp_store_f32(float v) {
    return v;
}

/* scale_u8 / scale_u16: the stored sample times the fade, rounded in double */
uchar wp_faded_u8(float v, float f) {
    return store_q_u8((float)sat_u8(v) * f);
}

ushort wp_faded_u16(float v, float f) {
    return store_q_u16((float)sat_u16(v) * f);
}

float wp_faded_f32(float v, float f) {
    return v * f;
}

/* clip_to_rect with at most 64 passes: 1 settled, 0 the bound was hit */
int wp_clip(float x, float y, float* ix, float* iy, float w1, float h1) {
    for (int s = 0; s < 64; s++) {
        if (*ix < 0x0p+0f) {
            float dy = y - *iy;
            float dxs = x - *ix;
            float num = x - 0x0p+0f;
            *ix = 0x0p+0f;
            *iy = y - (num / dxs) * dy;
            continue;
        }
        if (*ix > w1) {
            float dy = y - *iy;
            float dxs = x - *ix;
            float num = x - w1;
            *ix = w1;
            *iy = y - (num / dxs) * dy;
            continue;
        }
        if (*iy < 0x0p+0f) {
            float dyy = y - *iy;
            float num = y - 0x0p+0f;
            float dx = x - *ix;
            *ix = x - (num / dyy) * dx;
            *iy = 0x0p+0f;
            continue;
        }
        if (*iy > h1) {
            float dyy = y - *iy;
            float num = y - h1;
            float dx = x - *ix;
            *ix = x - (num / dyy) * dx;
            *iy = h1;
            continue;
        }
        return 1;
    }
    return 0;
}

/* Flow::bilerp: the two channel sampler with the upper clamp of weight 1 */
void wp_flow_bilerp(__global const float* d, int fw, int w1, int h1, float x, float y, float* out) {
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
    size_t i00 = ((size_t)iy * (size_t)fw + (size_t)ix) * 2;
    size_t i01 = i00 + (size_t)fw * 2;
    for (int k = 0; k < 2; k++) {
        float p00 = d[i00 + k];
        float p10 = d[i00 + 2 + k];
        float p01 = d[i01 + k];
        float p11 = d[i01 + 2 + k];
        float top = (p10 - p00) * fx + p00;
        float bot = (p11 - p01) * fx + p01;
        out[k] = (bot - top) * fy + top;
    }
}

/* the samplers and the inverse warp of one depth */
#define WP_DEPTH(SUF, T) \
void wp_mix4_##SUF(__global const T* src, size_t ia, size_t ib, size_t ic, size_t id, float fx, float fy, float* out) { \
    for (int k = 0; k < 4; k++) { \
        float p00 = (float)src[ia + k]; \
        float p10 = (float)src[ib + k]; \
        float p01 = (float)src[ic + k]; \
        float p11 = (float)src[id + k]; \
        float top = (p10 - p00) * fx + p00; \
        float bot = (p11 - p01) * fx + p01; \
        out[k] = (bot - top) * fy + top; \
    } \
} \
void wp_bilerp4_##SUF(__global const T* src, int w, int w1, int h1, float x, float y, float* out) { \
    int ix = cvtt_i32_f(x); \
    int iy = cvtt_i32_f(y); \
    float fx = x - (float)ix; \
    float fy = y - (float)iy; \
    if (ix >= w1) { \
        ix = w1 - 1; \
        fx = 0x1p+0f; \
    } \
    if (iy >= h1) { \
        iy = h1 - 1; \
        fy = 0x1p+0f; \
    } \
    size_t i00 = ((size_t)iy * (size_t)w + (size_t)ix) * 4; \
    size_t i01 = i00 + (size_t)w * 4; \
    wp_mix4_##SUF(src, i00, i00 + 4, i01, i01 + 4, fx, fy, out); \
} \
void wp_bilerp360_##SUF(__global const T* src, int w, int w1, int h1, float x, float y, float* out) { \
    int ix = cvtt_i32_f(x); \
    int iy = cvtt_i32_f(y); \
    float fx = x - (float)ix; \
    float fy = y - (float)iy; \
    int x1 = ix + 1 > w1 ? 0 : ix + 1; \
    int xc; \
    int xd; \
    int yc; \
    if (iy + 1 > h1) { \
        int hw = w / 2; \
        xc = (hw + ix) % w; \
        xd = (hw + x1) % w; \
        yc = h1; \
    } else { \
        xc = ix; \
        xd = x1; \
        yc = iy + 1; \
    } \
    int ya = iy < h1 ? iy : h1; \
    xc = xc > 0 ? xc : 0; \
    xd = xd > 0 ? xd : 0; \
    size_t ia = ((size_t)ya * (size_t)w + (size_t)ix) * 4; \
    size_t ib = ((size_t)ya * (size_t)w + (size_t)x1) * 4; \
    size_t ic = ((size_t)yc * (size_t)w + (size_t)xc) * 4; \
    size_t id = ((size_t)yc * (size_t)w + (size_t)xd) * 4; \
    wp_mix4_##SUF(src, ia, ib, ic, id, fx, fy, out); \
} \
__kernel void wp_inv_warp_##SUF(__global const T* src, __global const float* flow, __global T* out, __global uchar* cov, \
                                __global int* flag, int w, int h, int fw, int fh, float rx, float ry, float sx, float sy, \
                                int kind, int want_cov) { \
    int x = (int)get_global_id(0); \
    int y = (int)get_global_id(1); \
    if (x >= w || y >= h) { \
        return; \
    } \
    float wf = (float)w; \
    float hf = (float)h; \
    int w1i = w - 1; \
    int h1i = h - 1; \
    float w1 = (float)w1i; \
    float h1 = (float)h1i; \
    float yf = (float)y; \
    float fyy = yf * ry; \
    float xf = (float)x; \
    float d[2]; \
    wp_flow_bilerp(flow, fw, fw - 1, fh - 1, xf * rx, fyy, d); \
    float ix = d[0] * sx + xf; \
    float iy = d[1] * sy + yf; \
    size_t o = ((size_t)y * (size_t)w + (size_t)x) * 4; \
    size_t ci = (size_t)y * (size_t)w + (size_t)x; \
    float s[4]; \
    uchar c = 255; \
    if (kind == 2) { \
        int far = 0; \
        int n = 0; \
        for (int t = 0; t < 64; t++) { \
            if (!(ix < 0x0p+0f)) { \
                break; \
            } \
            ix += wf; \
        } \
        far |= ix < 0x0p+0f; \
        for (int t = 0; t < 64; t++) { \
            if (!(wf <= ix)) { \
                break; \
            } \
            ix -= wf; \
        } \
        far |= wf <= ix; \
        for (int t = 0; t < 64; t++) { \
            if (!(iy < 0x0p+0f)) { \
                break; \
            } \
            iy += hf; \
            n += 1; \
        } \
        far |= iy < 0x0p+0f; \
        for (int t = 0; t < 64; t++) { \
            if (!(hf <= iy)) { \
                break; \
            } \
            iy -= hf; \
            n += 1; \
        } \
        far |= hf <= iy; \
        if (far) { \
            atomic_or(flag, 2); \
            out[o] = 0; out[o + 1] = 0; out[o + 2] = 0; out[o + 3] = 0; \
        } else if (!(ix >= 0x0p+0f && ix < wf) || isnan(iy)) { \
            atomic_or(flag, 1); \
            out[o] = 0; out[o + 1] = 0; out[o + 2] = 0; out[o + 3] = 0; \
        } else { \
            if ((n & 1) != 0) { \
                ix = wf * 0x1p-1f + ix; \
                if (ix >= wf) { \
                    ix -= wf; \
                } \
                if (iy != 0x0p+0f) { \
                    iy = hf - iy; \
                } \
            } \
            wp_bilerp360_##SUF(src, w, w1i, h1i, ix, iy, s); \
            for (int k = 0; k < 4; k++) { \
                out[o + k] = wp_store_##SUF(s[k]); \
            } \
        } \
    } else if (kind == 0) { \
        int settled = wp_clip(xf, yf, &ix, &iy, w1, h1); \
        if (!settled || isnan(ix) || isnan(iy)) { \
            atomic_or(flag, settled ? 1 : 2); \
            out[o] = 0; out[o + 1] = 0; out[o + 2] = 0; out[o + 3] = 0; \
        } else { \
            wp_bilerp4_##SUF(src, w, w1i, h1i, ix, iy, s); \
            for (int k = 0; k < 4; k++) { \
                out[o + k] = wp_store_##SUF(s[k]); \
            } \
        } \
    } else { \
        if (!(ix > -0x1p+0f && ix < wf && iy > -0x1p+0f && iy < hf)) { \
            out[o] = 0; out[o + 1] = 0; out[o + 2] = 0; out[o + 3] = 0; \
            c = 0; \
        } else if (ix >= 0x0p+0f && ix <= w1 && iy >= 0x0p+0f && iy <= h1) { \
            wp_bilerp4_##SUF(src, w, w1i, h1i, ix, iy, s); \
            for (int k = 0; k < 4; k++) { \
                out[o + k] = wp_store_##SUF(s[k]); \
            } \
            c = 255; \
        } else { \
            float f = ix < 0x0p+0f ? ix + 0x1p+0f : (ix <= w1 ? 0x1p+0f : wf - ix); \
            if (iy < 0x0p+0f) { \
                f = f * (iy + 0x1p+0f); \
            } else if (iy > h1) { \
                f = f * (hf - iy); \
            } \
            int settled = wp_clip(xf, yf, &ix, &iy, w1, h1); \
            if (!settled || isnan(ix) || isnan(iy)) { \
                atomic_or(flag, settled ? 1 : 2); \
                out[o] = 0; out[o + 1] = 0; out[o + 2] = 0; out[o + 3] = 0; \
                c = 0; \
            } else { \
                wp_bilerp4_##SUF(src, w, w1i, h1i, ix, iy, s); \
                for (int k = 0; k < 4; k++) { \
                    out[o + k] = wp_faded_##SUF(s[k], f); \
                } \
                float cf = f * 0x1.fep+7f + 0x1p-1f; \
                c = cf <= 0x1.fep+7f ? (uchar)cvtt_i64_d((double)cf) : 255; \
            } \
        } \
    } \
    if (want_cov) { \
        cov[ci] = c; \
    } \
}

WP_DEPTH(u8, uchar)
WP_DEPTH(u16, ushort)
WP_DEPTH(f32, float)
"#;

pub const BLURCOMP: &str = r#"
/* Pix::store_blur: trunc(v + 0.5) back to float, clamped to [0, white]; f32 raw */
uchar bc_store_u8(float v) {
    float t = (float)cvtt_i32_f(v + 0x1p-1f);
    t = t > 0x0p+0f ? t : 0x0p+0f;
    t = t < 0x1.fep+7f ? t : 0x1.fep+7f;
    return (uchar)cvtt_i32_f(t);
}

ushort bc_store_u16(float v) {
    float t = (float)cvtt_i32_f(v + 0x1p-1f);
    t = t > 0x0p+0f ? t : 0x0p+0f;
    t = t < 0x1.fffep+15f ? t : 0x1.fffep+15f;
    return (ushort)cvtt_i32_f(t);
}

float bc_store_f32(float v) {
    return v;
}

/* Blur::wrap_x with at most 64 steps per loop: 0 wrapped into [0, W), 1 outside (the CPU
   refuses), 2 the device bound was hit */
int bc_wrap_x(float qx, float wf, float* out) {
    float x = qx;
    for (int t = 0; t < 64; t++) {
        if (!(x < 0x0p+0f)) {
            break;
        }
        x += wf;
    }
    if (x < 0x0p+0f) {
        return 2;
    }
    for (int t = 0; t < 64; t++) {
        if (!(wf <= x)) {
            break;
        }
        x -= wf;
    }
    if (wf <= x) {
        return 2;
    }
    if (x >= 0x0p+0f && x < wf) {
        *out = x;
        return 0;
    }
    return 1;
}

#define BC_DEPTH(SUF, T) \
__kernel void bc_motion_blur_##SUF(__global const T* src, __global const float* flow, __global T* dest, __global int* flag, \
                                   int w, int h, int fw, int fh, float rx, float ry, float kx, float ky, int shifted, \
                                   float k, int bilinear, int wrap, int y0) { \
    int x = (int)get_global_id(0); \
    int y = y0 + (int)get_global_id(1); \
    if (x >= w || y >= h) { \
        return; \
    } \
    float wf = (float)w; \
    float hf = (float)h; \
    int w1i = w - 1; \
    int h1i = h - 1; \
    float yf = (float)y; \
    float fyy = yf * ry; \
    float xf = (float)x; \
    float d[2]; \
    wp_flow_bilerp(flow, fw, fw - 1, fh - 1, xf * rx, fyy, d); \
    float vx = d[0] * kx; \
    float vy = d[1] * ky; \
    float d2 = vy * vy + vx * vx; \
    size_t o = ((size_t)y * (size_t)w + (size_t)x) * 4; \
    if (bilinear && 0x1.0c6f7ap-20f > d2) { \
        for (int c = 0; c < 4; c++) { \
            dest[o + c] = src[o + c]; \
        } \
        return; \
    } \
    float dist = sqrt(d2); \
    float cx = xf; \
    float cy = yf; \
    if (shifted) { \
        cx = vx * k + xf; \
        cy = vy * k + yf; \
    } \
    if (!bilinear) { \
        cx = cx + 0x1p-1f; \
        cy = cy + 0x1p-1f; \
    } \
    int n = cvtt_i32_f(dist + 0x1p+0f); \
    float sx; \
    float sy; \
    if (n < 2) { \
        n = 2; \
        sx = vx; \
        sy = vy; \
    } else { \
        float dd = (float)(n - 1); \
        sx = vx / dd; \
        sy = vy / dd; \
    } \
    if (n > 65536) { \
        atomic_or(flag, 1); \
        dest[o] = 0; dest[o + 1] = 0; dest[o + 2] = 0; dest[o + 3] = 0; \
        return; \
    } \
    if (n > 1024) { \
        atomic_or(flag, 2); \
        dest[o] = 0; dest[o + 1] = 0; dest[o + 2] = 0; dest[o + 3] = 0; \
        return; \
    } \
    float sum[4]; \
    for (int c = 0; c < 4; c++) { \
        sum[c] = 0x0p+0f; \
    } \
    int cnt = 0; \
    int bad = 0; \
    float t4[4]; \
    float qx = cx; \
    float qy = cy; \
    float nsx = -sx; \
    float nsy = -sy; \
    for (int dir = 0; dir < 2; dir++) { \
        for (int t = 0; t < 1024; t++) { \
            if (t >= n - dir) { \
                break; \
            } \
            if (bilinear) { \
                if (isnan(qx) || isnan(qy)) { \
                    bad |= 1; \
                } else if (wrap) { \
                    if (0x0p+0f <= qy && qy < hf) { \
                        float wx = 0x0p+0f; \
                        int r = bc_wrap_x(qx, wf, &wx); \
                        if (r != 0) { \
                            bad |= r; \
                        } else { \
                            cnt += 1; \
                            wp_bilerp360_##SUF(src, w, w1i, h1i, wx, qy, t4); \
                            for (int c = 0; c < 4; c++) { \
                                sum[c] += t4[c]; \
                            } \
                        } \
                    } \
                } else if (0x0p+0f <= qx && qx < wf && 0x0p+0f <= qy && qy < hf) { \
                    cnt += 1; \
                    wp_bilerp4_##SUF(src, w, w1i, h1i, qx, qy, t4); \
                    for (int c = 0; c < 4; c++) { \
                        sum[c] += t4[c]; \
                    } \
                } \
            } else { \
                int iy = cvtt_i32_f(qy); \
                int inside = 0; \
                int ix = 0; \
                if (wrap) { \
                    if (iy >= 0 && iy < h) { \
                        float wx = 0x0p+0f; \
                        int r = bc_wrap_x(qx, wf, &wx); \
                        if (r != 0) { \
                            bad |= r; \
                        } else { \
                            ix = cvtt_i32_f(wx); \
                            inside = 1; \
                        } \
                    } \
                } else { \
                    ix = cvtt_i32_f(qx); \
                    inside = ix >= 0 && ix < w && iy >= 0 && iy < h; \
                } \
                if (inside) { \
                    if (dir == 0) { \
                        cnt += 1; \
                    } \
                    size_t i = ((size_t)iy * (size_t)w + (size_t)ix) * 4; \
                    for (int c = 0; c < 4; c++) { \
                        sum[c] += (float)src[i + c]; \
                    } \
                } \
                if (dir == 1) { \
                    cnt += 1; \
                } \
            } \
            if (dir == 0) { \
                qx += sx; \
                qy += sy; \
            } else { \
                qx += nsx; \
                qy += nsy; \
            } \
        } \
        qx = cx - sx; \
        qy = cy - sy; \
    } \
    if (bad != 0) { \
        atomic_or(flag, bad); \
    } \
    float inv = 0x1p+0f / (float)cnt; \
    for (int c = 0; c < 4; c++) { \
        dest[o + c] = bc_store_##SUF(inv * sum[c]); \
    } \
} \
__kernel void bc_clear_rgb_##SUF(__global const T* src, __global T* dst, uint n) { \
    size_t p = get_global_id(0); \
    if (p >= n) { \
        return; \
    } \
    size_t o = p * 4; \
    T a = src[o]; \
    dst[o] = a; \
    if (a == 0) { \
        dst[o + 1] = 0; \
        dst[o + 2] = 0; \
        dst[o + 3] = 0; \
    } else { \
        dst[o + 1] = src[o + 1]; \
        dst[o + 2] = src[o + 2]; \
        dst[o + 3] = src[o + 3]; \
    } \
}

BC_DEPTH(u8, uchar)
BC_DEPTH(u16, ushort)
BC_DEPTH(f32, float)
"#;

pub const SMARTBLEND: &str = r#"
/* smartblend::over: a over b, alpha in units of white */
void sb_over(const float* a, const float* b, float white, float* o) {
    if (a[0] == 0x0p+0f) {
        for (int c = 0; c < 4; c++) {
            o[c] = b[c];
        }
        return;
    }
    if (a[0] == white) {
        for (int c = 0; c < 4; c++) {
            o[c] = a[c];
        }
        return;
    }
    float ia = 0x1p+0f - a[0] / white;
    float alpha = (white - b[0]) * (a[0] / white) + b[0];
    o[0] = alpha;
    o[1] = b[1] * ia + a[1];
    o[2] = b[2] * ia + a[2];
    o[3] = b[3] * ia + a[3];
}

/* smart_pixel: returns 1 when the raw store applies */
int sb_smart_pixel(const float* p1, const float* p2, float w, float white, float* v) {
    float c1[4];
    float c2[4];
    sb_over(p1, p2, white, c1);
    sb_over(p2, p1, white, c2);
    float amax = p1[0] > p2[0] ? p1[0] : p2[0];
    for (int i = 0; i < 4; i++) {
        v[i] = (c2[i] - c1[i]) * w + c1[i];
    }
    if (v[0] == amax) {
        return 0;
    }
    if (v[0] == 0x0p+0f) {
        if (c1[0] > c2[0]) {
            for (int i = 0; i < 4; i++) {
                v[i] = c1[i];
            }
        } else {
            for (int i = 0; i < 4; i++) {
                v[i] = c2[i];
            }
        }
        return 1;
    }
    float alpha = v[0];
    v[1] = (v[1] * amax) / alpha;
    v[2] = (v[2] * amax) / alpha;
    v[3] = (v[3] * amax) / alpha;
    v[0] = amax;
    return 0;
}

/* binary_pixel */
int sb_binary_pixel(const float* p1, const float* p2, float w, float white, float* v) {
    float x[4];
    float y[4];
    if (p2[0] == white) {
        float ia = 0x1p+0f - p1[0] / white;
        x[0] = p2[0];
        x[1] = p2[1] * ia + p1[1];
        x[2] = p2[2] * ia + p1[2];
        x[3] = p2[3] * ia + p1[3];
    } else if (p1[0] == white) {
        for (int i = 0; i < 4; i++) {
            x[i] = p1[i];
        }
    } else {
        for (int i = 0; i < 4; i++) {
            x[i] = 0x0p+0f;
        }
    }
    if (p1[0] == white) {
        float ia = 0x1p+0f - p2[0] / white;
        y[0] = p1[0];
        y[1] = p1[1] * ia + p2[1];
        y[2] = p1[2] * ia + p2[2];
        y[3] = p1[3] * ia + p2[3];
    } else if (p2[0] == white) {
        for (int i = 0; i < 4; i++) {
            y[i] = p2[i];
        }
    } else {
        for (int i = 0; i < 4; i++) {
            y[i] = 0x0p+0f;
        }
    }
    for (int i = 0; i < 4; i++) {
        v[i] = (y[i] - x[i]) * w + x[i];
    }
    return 0;
}

/* Px::store_plain and Px::store_raw */
uchar sb_plain_u8(float x) {
    float y = x + 0x1p-1f;
    if (0x0p+0f > y) {
        return 0;
    }
    if (y > 0x1.fep+7f) {
        return 255;
    }
    return trunc_u8_f(y);
}

ushort sb_plain_u16(float x) {
    float y = x + 0x1p-1f;
    if (0x0p+0f > y) {
        return 0;
    }
    if (y > 0x1.fffep+15f) {
        return 65535;
    }
    return trunc_u16_f(y);
}

float sb_plain_f32(float x) {
    return x + 0x0p+0f;
}

uchar sb_raw_u8(float x) {
    return trunc_u8_f(x);
}

ushort sb_raw_u16(float x) {
    return trunc_u16_f(x);
}

float sb_raw_f32(float x) {
    return x;
}

#define SB_DEPTH(SUF, T, WHITE) \
__kernel void sb_smart_##SUF(__global const T* p1, __global const T* p2, __global T* out, float w, int binary, uint n) { \
    size_t p = get_global_id(0); \
    if (p >= n) { \
        return; \
    } \
    size_t o = p * 4; \
    float a[4]; \
    float b[4]; \
    float v[4]; \
    for (int c = 0; c < 4; c++) { \
        a[c] = (float)p1[o + c]; \
        b[c] = (float)p2[o + c]; \
    } \
    int raw = binary ? sb_binary_pixel(a, b, w, WHITE, v) : sb_smart_pixel(a, b, w, WHITE, v); \
    for (int c = 0; c < 4; c++) { \
        out[o + c] = raw ? sb_raw_##SUF(v[c]) : sb_plain_##SUF(v[c]); \
    } \
}

SB_DEPTH(u8, uchar, 0x1.fep+7f)
SB_DEPTH(u16, ushort, 0x1.fffep+15f)
SB_DEPTH(f32, float, 0x1p+0f)

/* smart_finish before the hole fill: cov1 = 255 - max(cov1, cov2) */
__kernel void sb_cov_max_invert(__global uchar* cov1, __global const uchar* cov2, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    uchar a = cov1[i];
    uchar b = cov2[i];
    uchar m = a > b ? a : b;
    cov1[i] = (uchar)(255 - m);
}

/* Coverage::invert */
__kernel void sb_cov_invert(__global uchar* cov, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    cov[i] = (uchar)(255 - cov[i]);
}
"#;
