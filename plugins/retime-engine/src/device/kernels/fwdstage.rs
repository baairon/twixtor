//! Device kernels of the forward warp's rasteriser, prefix `fw_` / `FW_`. Their Rust
//! transcriptions (statement by statement) are `device::fwdstage::sim`; the stage wrappers are
//! `device::fwdstage`.
//!
//! The mesh (vertex positions and the per vertex mip level, whose `logf` takes its bits from the
//! host C runtime) is computed on the host by `forward::mesh` and uploaded; these kernels do the
//! scan conversion, the pyramid sampling and the running mean. The CPU folds the hits of a pixel
//! in the order of the triangles (cell `(r, c)` row major, `T1` then `T2`, triangle index
//! `2 * (r * (w - 1) + c) + {0, 1}`); the device reproduces that order per pixel:
//! * `fw_count`: one item per triangle runs `raster::triangle` with a counting sink (an atomic
//!   increment of the pixel's hit count) and adds its hits to a global total;
//! * `fw_scan_chunks`, `fw_scan_supers`, `fw_scan_top`: the exclusive prefix sum of the counts
//!   (64 pixel chunks, 4096 pixel super chunks, one item over the super chunks); the first hit
//!   slot of pixel `p` is `off[p] + cpre[p / 64] + sbase[p / 4096]`;
//! * `fw_emit`: the same scan conversion again (the same arithmetic, so the same hits), each hit
//!   stored as the record `(triangle index, u, v, d)` at the next free slot of its pixel (the slot
//!   order is the arrival order, which the device does not fix);
//! * `fw_fold_*`: one item per pixel sorts its records by triangle index (insertion sort, stable;
//!   two records with one index raise the bound flag, a pixel is hit at most once per triangle)
//!   and folds them in that order with the CPU's sampler and running mean, then stores the
//!   destination sample and the coverage.
//! * `fw_load_*`, `fw_resample_*`: the pyramid levels (`forward::pyramid::Pyramid::build`): a
//!   level is loaded as floats, blurred by `cv_hpass` and `cv_vpass` with the taps of `blur5`
//!   (with the integer store and reload between the passes, `wb_round_*`, and after them,
//!   `wb_from_float_*`), then resampled by `fw_resample_*` into one buffer holding every level.
//!
//! Precision contract (V: the CPU functions are the reference): `fw_triangle`, `fw_span` and
//! `fw_edge_init` are `raster::triangle`, `span` and `edge_init` operation for operation in
//! single precision (the three tests against double constants widen first, the reciprocals are
//! correctly rounded divisions); `fw_vertex` is the `vtx` closure of `forward_warp_with`
//! (`u = (float)c / (float)(w - 1)`); `fw_sample_*` is `Pyramid::sample` and `pyramid::tap`
//! through `wp_bilerp4_*` (the CPU's `warp::bilerp4`); `fw_fold` is `Accum::hit` (the reciprocal
//! `1.0 / ((double)n + 1.0)` divided in double and narrowed); `fw_out_*` is `forward::convert`;
//! `fw_resample_*` is `pyramid::resample_plane` (`Pix::store` is `wp_store_*`).
//!
//! Bounded loops. The scanline and span loops are bounded by the image size, which the host
//! limits to [`FW_MAX_DIM`] in both directions, so the constant bound 16384 is never the reason a
//! loop ends. The data dependent counts have device bounds: [`FW_TRI_CAP`] hits per triangle,
//! [`FW_PIXEL_CAP`] hits per pixel, [`FW_RECORDS_PER_PIXEL`] records per pixel on average; past
//! one of them, at a tap the CPU would read through a negative index (a texture coordinate below
//! `-1 / (w - 1)`, which only a NaN or a far extrapolation gives) or at two records with one
//! triangle index, bit 2 of the flag word is raised ([`FLAG_DEVICE_BOUND`]) and the CPU runs the
//! warp. Bit 1 ([`FLAG_CPU_REFUSES`]) is the degenerate row write outside the image, which the
//! CPU refuses too.

pub use super::warpstage::{FLAG_CPU_REFUSES, FLAG_DEVICE_BOUND};

/// Hits of one triangle on the device.
pub const FW_TRI_CAP: u32 = 16384;
/// Hits of one pixel on the device (the insertion sort and the fold loop bound).
pub const FW_PIXEL_CAP: u32 = 512;
/// Records per pixel on average (the record buffer of `w * h * 4` records at most).
pub const FW_RECORDS_PER_PIXEL: usize = 4;
/// The largest width and height (the scanline and span loop bounds).
pub const FW_MAX_DIM: usize = 16384;
/// Pixels per chunk and per super chunk of the prefix sum.
pub const FW_CHUNK: usize = 64;
pub const FW_SUPER: usize = 4096;
/// Super chunks the one item scan loops over at most (`2^26` pixels).
pub const FW_MAX_SUPERS: usize = 16384;

pub const FORWARD: &str = r#"
/* One vertex of the payload [x, y, u, v, d] (raster::Vtx) and one active edge (raster::Edge). */
typedef struct fw_vtx_s {
    float x;
    float y;
    float u;
    float v;
    float d;
} fw_vtx;

typedef struct fw_edge_s {
    float ay;
    float by;
    float x;
    float u;
    float v;
    float d;
    float dx;
    float du;
    float dv;
    float dd;
} fw_edge;

/* The vertex (r, c) of the mesh: the uploaded position and level, u = c / w1, v = r / h1. */
fw_vtx fw_vertex(__global const float* pos, __global const float* dl, int has_d, int w, int h, int r, int c) {
    size_t i = (size_t)r * (size_t)w + (size_t)c;
    fw_vtx v;
    v.x = pos[i * 2];
    v.y = pos[i * 2 + 1];
    v.u = (float)c / (float)(w - 1);
    v.v = (float)r / (float)(h - 1);
    v.d = has_d ? dl[i] : 0x0p+0f;
    return v;
}

/* The sink. mode 0 counts the hit, mode 1 stores the record at the pixel's next slot. Returns 0
   when the triangle passes its bound of hits (the bound flag is raised). */
int fw_hit(int mode, int x, int y, float u, float v, float d, uint key, int w, int* nh,
           __global uint* cnt, __global uint* fill, __global const uint* off, __global const uint* cpre,
           __global const uint* sbase, __global uint4* rec, uint total, __global int* flag) {
    *nh = *nh + 1;
    if (*nh > 16384) {
        atomic_or(flag, 2);
        return 0;
    }
    uint p = (uint)y * (uint)w + (uint)x;
    if (mode == 0) {
        atomic_inc(&cnt[p]);
        return 1;
    }
    uint slot = atomic_inc(&fill[p]);
    uint idx = off[p] + cpre[p >> 6] + sbase[p >> 12] + slot;
    if (slot >= cnt[p] || idx >= total) {
        atomic_or(flag, 2);
        return 0;
    }
    rec[idx] = (uint4)(key, as_uint(u), as_uint(v), as_uint(d));
    return 1;
}

/* raster::edge_init: activates a -> b when a.y <= y < b.y, slot 0 when no edge is active. */
void fw_edge_init(fw_vtx a, fw_vtx b, float y, fw_edge* e0, fw_edge* e1, uint* count) {
    if (a.y <= y && y < b.y) {
        int slot = *count != 0u ? 1 : 0;
        *count = *count + 1u;
        float dx = b.x - a.x;
        float du = b.u - a.u;
        float dv = b.v - a.v;
        float dd = b.d - a.d;
        float inv = 0x1p+0f / (b.y - a.y);
        float t = ((y - 0x1p+0f) - a.y) * inv;
        fw_edge e;
        e.ay = a.y;
        e.by = b.y;
        e.x = dx * t + a.x;
        e.u = du * t + a.u;
        e.v = dv * t + a.v;
        e.d = dd * t + a.d;
        e.dx = dx * inv;
        e.du = du * inv;
        e.dv = dv * inv;
        e.dd = dd * inv;
        if (slot == 0) {
            *e0 = e;
        } else {
            *e1 = e;
        }
    }
}

/* raster::span: one span of the main pass; returns 0 when the hit bound was passed. */
int fw_span(int mode, fw_edge l, fw_edge r, float y, int w1, uint key, int w, int* nh,
            __global uint* cnt, __global uint* fill, __global const uint* off, __global const uint* cpre,
            __global const uint* sbase, __global uint4* rec, uint total, __global int* flag) {
    int x0 = cvtt_i32_f(ceil(l.x));
    x0 = x0 > 0 ? x0 : 0;
    int x1 = cvtt_i32_f(floor(r.x));
    x1 = x1 < w1 ? x1 : w1;
    float wid = r.x - l.x;
    if (x0 > x1) {
        return 1;
    }
    if (!((double)wid > 0x1.5798ee2308c3ap-27)) {
        return 1;
    }
    int yi = cvtt_i32_f(y);
    if (x0 == x1) {
        float t = ((float)x0 - l.x) * (0x1p+0f / wid);
        float u = (r.u - l.u) * t + l.u;
        float v = (r.v - l.v) * t + l.v;
        float d = (r.d - l.d) * t + l.d;
        return fw_hit(mode, x0, yi, u, v, d, key, w, nh, cnt, fill, off, cpre, sbase, rec, total, flag);
    }
    float inv = 0x1p+0f / wid;
    float du = r.u - l.u;
    float dv = r.v - l.v;
    float dd = r.d - l.d;
    float t = ((float)x0 - l.x) * inv;
    float u = du * t + l.u;
    float v = dv * t + l.v;
    float d = dd * t + l.d;
    int x = x0;
    for (int s = 0; s < 16384; s++) {
        if (!(x < x1)) {
            break;
        }
        if (!fw_hit(mode, x, yi, u, v, d, key, w, nh, cnt, fill, off, cpre, sbase, rec, total, flag)) {
            return 0;
        }
        u += du * inv;
        v += dv * inv;
        d += dd * inv;
        x += 1;
    }
    return fw_hit(mode, x, yi, u, v, d, key, w, nh, cnt, fill, off, cpre, sbase, rec, total, flag);
}

/* One hit of the degenerate pass: outside the image raises the CPU refusal and writes nothing. */
int fw_emit_hit(int mode, int x, int yi, int w1, int h1, float u, float v, float d, uint key, int w, int* nh,
                __global uint* cnt, __global uint* fill, __global const uint* off, __global const uint* cpre,
                __global const uint* sbase, __global uint4* rec, uint total, __global int* flag) {
    if (x < 0 || x > w1 || yi < 0 || yi > h1) {
        atomic_or(flag, 1);
        return 1;
    }
    return fw_hit(mode, x, yi, u, v, d, key, w, nh, cnt, fill, off, cpre, sbase, rec, total, flag);
}

/* raster::triangle; returns 0 when the hit bound was passed. */
int fw_triangle(int mode, fw_vtx a0, fw_vtx a1, fw_vtx a2, int w1, int h1, uint key, int w, int* nh,
                __global uint* cnt, __global uint* fill, __global const uint* off, __global const uint* cpre,
                __global const uint* sbase, __global uint4* rec, uint total, __global int* flag) {
    float w1f = (float)w1;
    float h1f = (float)h1;
    int ok = (0x0p+0f <= a0.x || 0x0p+0f <= a1.x || 0x0p+0f <= a2.x)
          && (0x0p+0f <= a0.y || 0x0p+0f <= a1.y || 0x0p+0f <= a2.y)
          && (a0.x <= w1f || a1.x <= w1f || a2.x <= w1f)
          && (a0.y <= h1f || a1.y <= h1f || a2.y <= h1f);
    if (!ok) {
        return 1;
    }
    fw_vtx v0;
    fw_vtx v1;
    fw_vtx v2;
    if (a1.y < a0.y) {
        if (a2.y < a0.y) {
            if (a2.y < a1.y) {
                v0 = a2;
                v1 = a1;
                v2 = a0;
            } else {
                v0 = a1;
                v1 = a2;
                v2 = a0;
            }
        } else {
            v0 = a1;
            v1 = a0;
            v2 = a2;
        }
    } else if (a2.y < a1.y) {
        if (a0.y < a2.y) {
            v0 = a0;
            v1 = a2;
            v2 = a1;
        } else {
            v0 = a2;
            v1 = a0;
            v2 = a1;
        }
    } else {
        v0 = a0;
        v1 = a1;
        v2 = a2;
    }
    float y = ceil(v0.y);
    float y_end = floor(v2.y);
    if (y < 0x0p+0f) {
        y = 0x0p+0f;
    }
    if (h1f < y_end) {
        y_end = h1f;
    }
    if (y <= y_end) {
        int vi = 0;
        uint count = 0u;
        fw_edge e0;
        e0.ay = 0x0p+0f; e0.by = 0x0p+0f; e0.x = 0x0p+0f; e0.u = 0x0p+0f; e0.v = 0x0p+0f;
        e0.d = 0x0p+0f; e0.dx = 0x0p+0f; e0.du = 0x0p+0f; e0.dv = 0x0p+0f; e0.dd = 0x0p+0f;
        fw_edge e1 = e0;
        for (int s = 0; s < 16384; s++) {
            if (vi < 3) {
                for (int q = 0; q < 3; q++) {
                    fw_vtx cur = vi == 0 ? v0 : (vi == 1 ? v1 : v2);
                    if (y < cur.y) {
                        break;
                    }
                    if (count != 0u) {
                        if (count == 1u) {
                            if (e0.ay <= y && e0.by <= y) {
                                count = 0u;
                            }
                        } else if (count == 2u) {
                            if (y < e1.ay || y < e1.by) {
                                if (!(y < e0.ay || y < e0.by)) {
                                    e0 = e1;
                                    count = 1u;
                                }
                            } else if (e0.ay <= y && e0.by <= y) {
                                count = 0u;
                            } else {
                                count = 1u;
                            }
                        }
                    }
                    if (vi == 0) {
                        fw_edge_init(v0, v1, y, &e0, &e1, &count);
                        fw_edge_init(v0, v2, y, &e0, &e1, &count);
                    } else if (vi == 1) {
                        fw_edge_init(v1, v2, y, &e0, &e1, &count);
                        fw_edge_init(v1, v0, y, &e0, &e1, &count);
                    } else {
                        fw_edge_init(v2, v0, y, &e0, &e1, &count);
                        fw_edge_init(v2, v1, y, &e0, &e1, &count);
                    }
                    vi += 1;
                    if (vi >= 3) {
                        break;
                    }
                }
            }
            e0.x += e0.dx;
            e0.u += e0.du;
            e0.v += e0.dv;
            e0.d += e0.dd;
            e1.x += e1.dx;
            e1.u += e1.du;
            e1.v += e1.dv;
            e1.d += e1.dd;
            if (count == 2u) {
                if (e1.x < e0.x) {
                    if (!fw_span(mode, e1, e0, y, w1, key, w, nh, cnt, fill, off, cpre, sbase, rec, total, flag)) {
                        return 0;
                    }
                } else {
                    if (!fw_span(mode, e0, e1, y, w1, key, w, nh, cnt, fill, off, cpre, sbase, rec, total, flag)) {
                        return 0;
                    }
                }
            }
            y += 0x1p+0f;
            if (!(y <= y_end)) {
                break;
            }
        }
    }
    /* the degenerate pass: both lower vertices exactly on the last row */
    if (v1.y == h1f && v1.y == v2.y) {
        fw_vtx l;
        fw_vtx r;
        if (v2.x <= v1.x) {
            l = v2;
            r = v1;
        } else {
            l = v1;
            r = v2;
        }
        float yy = l.y;
        float yt = (float)cvtt_i32_f(yy);
        if (0x0p+0f <= yt && yt <= h1f && !((double)(yy - yt) > 0x1.47ae147ae147bp-7)) {
            int x0 = cvtt_i32_f(ceil(l.x));
            x0 = x0 > 0 ? x0 : 0;
            int x1 = cvtt_i32_f(floor(r.x));
            x1 = x1 < w1 ? x1 : w1;
            float wid = r.x - l.x;
            if ((double)wid > 0x1.4f8b588e368f1p-17) {
                float inv = 0x1p+0f / wid;
                int yi = cvtt_i32_f(yt);
                if (x0 == x1) {
                    float tt = ((float)x0 - l.x) * inv;
                    float u = (r.u - l.u) * tt + l.u;
                    float v = (r.v - l.v) * tt + l.v;
                    float d = (r.d - l.d) * tt + l.d;
                    return fw_emit_hit(mode, x0, yi, w1, h1, u, v, d, key, w, nh, cnt, fill, off, cpre, sbase, rec, total, flag);
                }
                float du = r.u - l.u;
                float dv = r.v - l.v;
                float dd = r.d - l.d;
                float tt = ((float)x0 - l.x) * inv;
                float u = du * tt + l.u;
                float v = dv * tt + l.v;
                float d = dd * tt + l.d;
                int x = x0;
                for (int s = 0; s < 16384; s++) {
                    if (!(x < x1)) {
                        break;
                    }
                    if (!fw_emit_hit(mode, x, yi, w1, h1, u, v, d, key, w, nh, cnt, fill, off, cpre, sbase, rec, total, flag)) {
                        return 0;
                    }
                    u += du * inv;
                    v += dv * inv;
                    d += dd * inv;
                    x += 1;
                }
                return fw_emit_hit(mode, x, yi, w1, h1, u, v, d, key, w, nh, cnt, fill, off, cpre, sbase, rec, total, flag);
            }
        }
    }
    return 1;
}

/* The triangle `key` of the mesh: cell key / 2 (row major over (h - 1) x (w - 1) cells), T1 for an
   even key, T2 for an odd one. */
int fw_mesh_triangle(int mode, uint key, __global const float* pos, __global const float* dl, int has_d, int w, int h, int* nh,
                     __global uint* cnt, __global uint* fill, __global const uint* off, __global const uint* cpre,
                     __global const uint* sbase, __global uint4* rec, uint total, __global int* flag) {
    uint cell = key >> 1;
    int r = (int)(cell / (uint)(w - 1));
    int c = (int)(cell % (uint)(w - 1));
    fw_vtx v01 = fw_vertex(pos, dl, has_d, w, h, r, c + 1);
    fw_vtx v10 = fw_vertex(pos, dl, has_d, w, h, r + 1, c);
    if ((key & 1u) == 0u) {
        fw_vtx v00 = fw_vertex(pos, dl, has_d, w, h, r, c);
        return fw_triangle(mode, v00, v01, v10, w - 1, h - 1, key, w, nh, cnt, fill, off, cpre, sbase, rec, total, flag);
    }
    fw_vtx v11 = fw_vertex(pos, dl, has_d, w, h, r + 1, c + 1);
    return fw_triangle(mode, v11, v01, v10, w - 1, h - 1, key, w, nh, cnt, fill, off, cpre, sbase, rec, total, flag);
}

/* Pass 1: the hits of the triangles key0 .. key0 + n - 1 counted per pixel and in total[0];
   nothing is counted once total[0] passed maxrec (the bound flag is raised then). */
__kernel void fw_count(__global const float* pos, __global const float* dl, int has_d, int w, int h,
                       __global uint* cnt, __global uint* total, uint maxrec, __global int* flag, uint key0, uint n) {
    uint g = (uint)get_global_id(0);
    if (g >= n) {
        return;
    }
    if (atomic_add(&total[0], 0u) > maxrec) {
        return;
    }
    int nh = 0;
    fw_mesh_triangle(0, key0 + g, pos, dl, has_d, w, h, &nh, cnt, cnt, cnt, cnt, cnt, (__global uint4*)cnt, 0u, flag);
    if (nh > 0) {
        uint old = atomic_add(&total[0], (uint)nh);
        if (old + (uint)nh > maxrec || old + (uint)nh < old) {
            atomic_or(flag, 2);
        }
    }
}

/* Pass 2: the exclusive prefix sum of 64 counts per item, the chunk sum to csum. */
__kernel void fw_scan_chunks(__global const uint* cnt, __global uint* off, __global uint* csum, uint n, uint nc) {
    uint k = (uint)get_global_id(0);
    if (k >= nc) {
        return;
    }
    uint s = 0u;
    for (int j = 0; j < 64; j++) {
        uint i = k * 64u + (uint)j;
        if (i >= n) {
            break;
        }
        off[i] = s;
        s += cnt[i];
    }
    csum[k] = s;
}

/* The exclusive prefix sum of 64 chunk sums per item (4096 pixels), the sum to ssum. */
__kernel void fw_scan_supers(__global const uint* csum, __global uint* cpre, __global uint* ssum, uint nc, uint ns) {
    uint k = (uint)get_global_id(0);
    if (k >= ns) {
        return;
    }
    uint s = 0u;
    for (int j = 0; j < 64; j++) {
        uint i = k * 64u + (uint)j;
        if (i >= nc) {
            break;
        }
        cpre[i] = s;
        s += csum[i];
    }
    ssum[k] = s;
}

/* The exclusive prefix sum of the super chunk sums by one item; the sum to total[1] (saturated). */
__kernel void fw_scan_top(__global const uint* ssum, __global uint* sbase, __global uint* total, uint ns) {
    if (get_global_id(0) != 0) {
        return;
    }
    ulong s = 0UL;
    for (int j = 0; j < 16384; j++) {
        if ((uint)j >= ns) {
            break;
        }
        sbase[j] = (uint)s;
        s += (ulong)ssum[j];
    }
    total[1] = s > 0xffffffffUL ? 0xffffffffu : (uint)s;
}

/* Pass 3: the records of the triangles key0 .. key0 + n - 1; `total` is the record count. */
__kernel void fw_emit(__global const float* pos, __global const float* dl, int has_d, int w, int h,
                      __global uint* cnt, __global uint* fill, __global const uint* off, __global const uint* cpre,
                      __global const uint* sbase, __global uint4* rec, uint total, __global int* flag, uint key0, uint n) {
    uint g = (uint)get_global_id(0);
    if (g >= n) {
        return;
    }
    int nh = 0;
    fw_mesh_triangle(1, key0 + g, pos, dl, has_d, w, h, &nh, cnt, fill, off, cpre, sbase, rec, total, flag);
}

/* Accum::hit: the running mean with the saturating u8 hit counter n. */
void fw_fold(float* acc, int* n, const float* c) {
    if (*n == 0) {
        for (int k = 0; k < 4; k++) {
            acc[k] = c[k] + 0x0p+0f;
        }
        *n = 1;
    } else if (*n == 255) {
        for (int k = 0; k < 4; k++) {
            acc[k] = (acc[k] * 0x1.fep+7f + c[k]) * 0x1p-8f + 0x0p+0f;
        }
    } else {
        float nf = (float)(*n);
        float r = (float)(0x1p+0 / ((double)nf + 0x1p+0));
        for (int k = 0; k < 4; k++) {
            acc[k] = (nf * acc[k] + c[k]) * r + 0x0p+0f;
        }
        *n = *n + 1;
    }
}

/* forward::convert per sample */
float fw_out_f32(float a) {
    return a + 0x0p+0f;
}

uchar fw_out_u8(float a) {
    return trunc_u8_f(a + 0x1p-1f);
}

ushort fw_out_u16(float a) {
    return trunc_u16_f(a + 0x1p-1f);
}

/* The per depth kernels: the pyramid sampler (pyramid::tap and Pyramid::sample, level 0 is the
   source, level i >= 1 at lt[3 * i] in lev, of lt[3 * i + 1] x lt[3 * i + 2] pixels), the fold
   and the pyramid steps. */
#define FW_DEPTH(SUF, T) \
int fw_tap_##SUF(__global const T* img, int w, int h, float u, float v, float* c) { \
    int w1 = w - 1; \
    int h1 = h - 1; \
    float x = (float)w1 * u; \
    float y = (float)h1 * v; \
    if (cvtt_i32_f(x) < 0 || cvtt_i32_f(y) < 0) { \
        for (int k = 0; k < 4; k++) { \
            c[k] = 0x0p+0f; \
        } \
        return 0; \
    } \
    wp_bilerp4_##SUF(img, w, w1, h1, x, y, c); \
    return 1; \
} \
int fw_level_tap_##SUF(__global const T* src, __global const T* lev, __global const int* lt, int i, float u, float v, float* c) { \
    if (i == 0) { \
        return fw_tap_##SUF(src, lt[1], lt[2], u, v, c); \
    } \
    return fw_tap_##SUF(lev + lt[i * 3], lt[i * 3 + 1], lt[i * 3 + 2], u, v, c); \
} \
int fw_sample_##SUF(__global const T* src, __global const T* lev, __global const int* lt, int nlev, float u, float v, float d, float* c) { \
    if (0x0p+0f < d) { \
        if (d < (float)(nlev - 1)) { \
            int i = cvtt_i32_f(d); \
            float frac = d - (float)i; \
            float a[4]; \
            float b[4]; \
            int ok = fw_level_tap_##SUF(src, lev, lt, i, u, v, a); \
            ok = fw_level_tap_##SUF(src, lev, lt, i + 1, u, v, b) & ok; \
            for (int k = 0; k < 4; k++) { \
                c[k] = (b[k] - a[k]) * frac + a[k]; \
            } \
            return ok; \
        } \
        return fw_level_tap_##SUF(src, lev, lt, nlev - 1, u, v, c); \
    } \
    return fw_level_tap_##SUF(src, lev, lt, 0, u, v, c); \
} \
__kernel void fw_fold_##SUF(__global const T* src, __global const T* lev, __global const int* lt, int nlev, \
                            __global uint4* rec, __global const uint* cnt, __global const uint* off, __global const uint* cpre, \
                            __global const uint* sbase, __global T* out, __global uchar* cov, int want_cov, \
                            __global int* flag, int w, int h, int y0, int rows) { \
    int x = (int)get_global_id(0); \
    int gy = (int)get_global_id(1); \
    int y = y0 + gy; \
    if (x >= w || gy >= rows || y >= h) { \
        return; \
    } \
    uint p = (uint)y * (uint)w + (uint)x; \
    uint n = cnt[p]; \
    if (n > 512u) { \
        atomic_or(flag, 2); \
        n = 0u; \
    } \
    uint base = off[p] + cpre[p >> 6] + sbase[p >> 12]; \
    for (int i = 1; i < 512; i++) { \
        if ((uint)i >= n) { \
            break; \
        } \
        uint4 cur = rec[base + (uint)i]; \
        int j = i; \
        for (int s = 0; s < 512; s++) { \
            if (j <= 0) { \
                break; \
            } \
            uint4 prev = rec[base + (uint)(j - 1)]; \
            if (prev.x <= cur.x) { \
                break; \
            } \
            rec[base + (uint)j] = prev; \
            j -= 1; \
        } \
        rec[base + (uint)j] = cur; \
    } \
    float acc[4]; \
    for (int k = 0; k < 4; k++) { \
        acc[k] = 0x0p+0f; \
    } \
    int hits = 0; \
    uint last = 0u; \
    for (int i = 0; i < 512; i++) { \
        if ((uint)i >= n) { \
            break; \
        } \
        uint4 r = rec[base + (uint)i]; \
        if (i > 0 && r.x == last) { \
            atomic_or(flag, 2); \
        } \
        last = r.x; \
        float c[4]; \
        if (!fw_sample_##SUF(src, lev, lt, nlev, as_float(r.y), as_float(r.z), as_float(r.w), c)) { \
            atomic_or(flag, 2); \
        } \
        fw_fold(acc, &hits, c); \
    } \
    size_t o = (size_t)p * 4; \
    for (int k = 0; k < 4; k++) { \
        out[o + k] = fw_out_##SUF(acc[k]); \
    } \
    if (want_cov) { \
        cov[p] = hits != 0 ? (uchar)255 : (uchar)0; \
    } \
} \
__kernel void fw_load_##SUF(__global const T* src, uint off, __global float* dst, uint n) { \
    uint i = (uint)get_global_id(0); \
    if (i >= n) { \
        return; \
    } \
    dst[i] = (float)src[off + i]; \
} \
__kernel void fw_resample_##SUF(__global const T* src, int sw, int sh, __global T* lev, uint off, int dw, int dh, float sx, float sy) { \
    int x = (int)get_global_id(0); \
    int y = (int)get_global_id(1); \
    if (x >= dw || y >= dh) { \
        return; \
    } \
    float fy = (float)y * sy; \
    float v[4]; \
    wp_bilerp4_##SUF(src, sw, sw - 1, sh - 1, (float)x * sx, fy, v); \
    size_t o = (size_t)off + ((size_t)y * (size_t)dw + (size_t)x) * 4; \
    for (int k = 0; k < 4; k++) { \
        lev[o + k] = wp_store_##SUF(v[k]); \
    } \
}

FW_DEPTH(u8, uchar)
FW_DEPTH(u16, ushort)
FW_DEPTH(f32, float)
"#;
