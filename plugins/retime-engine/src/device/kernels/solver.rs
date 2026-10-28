//! Device kernels of the motion solver (`flow.rs`, `flow/extras.rs`), prefix `sv_`, with a line
//! for line Rust transcription of every kernel and helper (`*_item` for the kernels, one work
//! item, statement by statement) that the simulator runs.
//!
//! Precision contract (V: `flow::solve_pyramids` and `flow::extras::solve_pyramids_with` are the
//! reference and stay untouched). Single precision in the CPU's order, every operation rounded on
//! its own (`FP_CONTRACT OFF`, divisions correctly rounded by the build options), the prelude's
//! range tested `cvtt_i32_f` for every float to integer conversion (x86 `cvttss2si`), `floor`
//! and `fabs` only (both exact):
//! * `sv_grad`: `calc_grad` (one sided at the edges, `+ 0.0` kept) or `calc_grad_wrap` (columns
//!   modulo the width, divisor 2), then the gradient masking of `mask_gradient` (`(0, 0)` where
//!   the level mask is below 0.95 at the pixel or one of its four neighbours, the left and right
//!   ones modulo the width under wrap), then the structure tensor of `calc_a`
//!   (`gx * gx, gx * gy, gy * gy`) of the masked gradient; one item per pixel.
//! * `sv_calc_b`: `calc_b` and `calc_b_with` (the warped mask of `Y` at least 0.999; under wrap
//!   the row range test, `wrap_target` and the sampler `sv_bilinear1_wrapx_at`); per pixel.
//! * `sv_new_disp`: `new_disp`, `t = b * b - dd * a`, the update only where `|t| > 40.0`.
//! * `sv_apply_user`: the user displacement consumer `flow = d / w` where `w != 0.0`.
//! * `sv_sweep_rows`, `sv_sweep_cols`: the in place sweeps of `block_match` / `block_match_with`
//!   (`Block::step`, `Block::cost`, the patch, the G1b gate: a pixel whose level mask of `X` is
//!   below 0.95 keeps its vector). One item per interior row (sweeps 1 and 2) or per column
//!   (sweeps 3 and 4, every column under wrap with `Block::step_with`), stepping in the CPU's
//!   order over at most [`SV_SWEEP_CHUNK`] pixels per launch from a start the host passes; the
//!   flow buffer carries the updated neighbour from one launch to the next, so a sweep split into
//!   consecutive chunks is the CPU's sweep exactly. An item reads and writes only its own row or
//!   column of the flow.
//! * `sv_sweep_ring`: sweeps 1 and 2 of the wrap solve (`sweeps_ring`): every pixel of rows
//!   `1 ..= h - 2` against its ring neighbour as both stood BEFORE the sweep (a copy `old`), so
//!   the items are independent.
//! * `sv_range_rows`, `sv_range_decide`, `sv_range_apply`: `all_flow_out_of_range` (or its y only
//!   variant under wrap) and the driver's `check` state, kept on the device in `state[0]` (still
//!   checking) and `state[1]` (zero this level): one item per row ORs its pixels (a boolean OR, so
//!   the order does not matter), one item ORs the rows and updates the state, one item per value
//!   zeroes the flow when the state says so (`+0.0`, as the CPU's fill).
//! * `sv_scale` (`scale_values`, `f * v`), `sv_zero` (the zero start) and
//!   `sv_upsample_flow_wrapx` (`upsample_flow_wrapx` with `bilinear2_wrapx`; the plain upsample
//!   is `pm_upsample_flow`).
//!
//! Every per level and per frame scalar that needs a libm transcendental (`start_level`, the
//! Gaussian taps) or a host expression (the upsample scales, the damping test) is computed on the
//! host and passed in. Loops: every `for` loop compares against a literal (the lint); the pixel
//! loops of a row or column stop at [`SV_LINE_MAX`], and the wrappers refuse wider or taller
//! levels with `StageError::Unsupported`.

use crate::device::kernels::prelude::rs;
use crate::device::kernels::resample::{rs_bilinear1, rs_cell};
use crate::device::sim::Global;

/// The most pixels one sweep item steps in one launch (the literal bound of the sweep loops).
pub const SV_SWEEP_CHUNK: usize = 512;

/// The widest and tallest level the solver kernels take (the literal bound of the range check
/// loops); a larger level is `StageError::Unsupported`.
pub const SV_LINE_MAX: usize = 16384;

pub const SOLVER: &str = r#"
/* flow::wrap_target: the target column of pixel column x displaced by -u, split into a whole
   column modulo w and a fraction; 0 (None) for a NaN or more than 2^16 widths. */
int sv_wrap_target(int x, float u, int w, int* col, float* frac) {
    float d = -u;
    if (!(fabs(d) <= 0x1p+16f * (float)w)) {
        return 0;
    }
    float whole = floor(d);
    float f = d - whole;
    long t = (long)x + (long)whole;
    long m = t % (long)w;
    if (m < 0) {
        m = m + (long)w;
    }
    int c = (int)m;
    if (f >= 0x1p+0f) {
        c = c + 1 == w ? 0 : c + 1;
        f = 0x0p+0f;
    }
    *col = c;
    *frac = f;
    return 1;
}

/* sample::bilinear1_wrapx_at: column ix with the fraction fx, right neighbour ix + 1 or column 0,
   the row rule of rs_cell. */
float sv_bilinear1_wrapx_at(__global const float* img, int w, int h, int ix, float fx, float y) {
    int x1 = ix + 1 > w - 1 ? 0 : ix + 1;
    float fy;
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

/* sample::bilinear2_wrapx: the column rule of rs_wrapx_cell, the row rule of rs_cell. */
float2 sv_bilinear2_wrapx(__global const float* img, int w, int h, float x, float y) {
    int x1;
    float fx;
    float fy;
    int ix = rs_wrapx_cell(x, w, &x1, &fx);
    int iy = rs_cell(y, h, &fy);
    int r0 = iy * w;
    int r1 = (iy + 1) * w;
    float a00 = img[(r0 + ix) * 2];
    float a10 = img[(r0 + x1) * 2];
    float a01 = img[(r1 + ix) * 2];
    float a11 = img[(r1 + x1) * 2];
    float atop = (a10 - a00) * fx + a00;
    float abot = (a11 - a01) * fx + a01;
    float b00 = img[(r0 + ix) * 2 + 1];
    float b10 = img[(r0 + x1) * 2 + 1];
    float b01 = img[(r1 + ix) * 2 + 1];
    float b11 = img[(r1 + x1) * 2 + 1];
    float btop = (b10 - b00) * fx + b00;
    float bbot = (b11 - b01) * fx + b01;
    return (float2)((abot - atop) * fy + atop, (bbot - btop) * fy + btop);
}

/* The mask test of the solver: below MASK_ON (0.95) counts as outside. */
int sv_off(__global const float* m, int w, int x, int y) {
    return m[y * w + x] < 0x1.e66666p-1f;
}

/* Block::patch (cols x - 1, x, x + 1) and Block::patch_wrap (cols modulo w), rows y - 1 .. y + 1. */
void sv_patch(__global const float* a, int w, int x, int y, int wrap, float* p) {
    int c0 = wrap ? (x + w - 1) % w : x - 1;
    int c2 = wrap ? (x + 1) % w : x + 1;
    for (int r = 0; r < 3; r++) {
        int base = (y + r - 1) * w;
        p[r * 3] = a[base + c0];
        p[r * 3 + 1] = a[base + x];
        p[r * 3 + 2] = a[base + c2];
    }
}

/* The weights, squares and sum of Block::cost / Block::cost_wrap over the window rows
   rbase / w .. + 3 and the window columns c[0..4]. */
float sv_cost_sum(__global const float* b, int w, const float* p, int rbase, const int* c, float fx, float fy) {
    float w11 = fy * fx;
    float w10 = fx - w11;
    float w01 = fy - w11;
    float w00 = ((0x1p+0f - fx) - fy) + w11;
    float sq[9];
    for (int j = 0; j < 3; j++) {
        int r0 = rbase + j * w;
        int r1 = r0 + w;
        for (int i = 0; i < 3; i++) {
            float q = ((w00 * b[r0 + c[i]] + w10 * b[r0 + c[i + 1]]) + w01 * b[r1 + c[i]]) + w11 * b[r1 + c[i + 1]];
            float d = p[j * 3 + i] - q;
            sq[j * 3 + i] = d * d;
        }
    }
    float s = sq[0] + sq[1];
    for (int k = 2; k < 8; k++) {
        s = s + sq[k];
    }
    return sq[8] + s;
}

/* The row of a block cost: 0 when y2 > h - 2 or the cell row is below 1. */
int sv_cost_row(int h, float y2, int* iy, float* fy) {
    float hm2 = (float)(h - 2);
    if (y2 > hm2) {
        return 0;
    }
    if (y2 == hm2) {
        *iy = cvtt_i32_f(hm2 - 0x1p+0f);
        *fy = 0x1p+0f;
    } else {
        int i = cvtt_i32_f(y2);
        *iy = i;
        *fy = y2 - (float)i;
    }
    return *iy >= 1;
}

/* Block::cost: 1 and the cost in *out, or 0 (None). */
int sv_cost(__global const float* b, int w, int h, const float* p, float x2, float y2, float* out) {
    float wm2 = (float)(w - 2);
    float hm2 = (float)(h - 2);
    if (x2 > wm2 || y2 > hm2) {
        return 0;
    }
    int ix;
    float fx;
    if (x2 == wm2) {
        ix = cvtt_i32_f(wm2 - 0x1p+0f);
        fx = 0x1p+0f;
    } else {
        ix = cvtt_i32_f(x2);
        fx = x2 - (float)ix;
    }
    int iy;
    float fy;
    int row = sv_cost_row(h, y2, &iy, &fy);
    if (ix < 1 || !row) {
        return 0;
    }
    int c[4] = {ix - 1, ix, ix + 1, ix + 2};
    *out = sv_cost_sum(b, w, p, (iy - 1) * w, c, fx, fy);
    return 1;
}

/* Block::cost_wrap at the column col with the fraction fx. */
int sv_cost_wrap(__global const float* b, int w, int h, const float* p, int col, float fx, float y2, float* out) {
    int iy;
    float fy;
    if (!sv_cost_row(h, y2, &iy, &fy)) {
        return 0;
    }
    int c[4] = {(col + w - 1) % w, col, (col + 1) % w, (col + 2) % w};
    *out = sv_cost_sum(b, w, p, (iy - 1) * w, c, fx, fy);
    return 1;
}

/* Block::step (wrap 0) and Block::step_with (wrap 1): 1 when the neighbour's vector (nx, ny)
   costs less than the current one (cx, cy). */
int sv_step(__global const float* a, __global const float* b, int w, int h, int x, int y, float cx, float cy, float nx, float ny, int wrap) {
    float h1 = (float)(h - 1);
    float p[9];
    float k0;
    float k1;
    if (wrap) {
        float ty = (float)y - cy;
        if (!(0x0p+0f <= ty && ty <= h1)) {
            return 0;
        }
        sv_patch(a, w, x, y, 1, p);
        int col;
        float f;
        if (!sv_wrap_target(x, cx, w, &col, &f) || !sv_cost_wrap(b, w, h, p, col, f, ty, &k0)) {
            return 0;
        }
        float ny2 = (float)y - ny;
        if (!sv_wrap_target(x, nx, w, &col, &f) || !sv_cost_wrap(b, w, h, p, col, f, ny2, &k1)) {
            return 0;
        }
        return k1 < k0;
    }
    float tx = (float)x - cx;
    float ty = (float)y - cy;
    float w1 = (float)(w - 1);
    if (!(0x0p+0f <= tx && tx <= w1 && 0x0p+0f <= ty && ty <= h1)) {
        return 0;
    }
    sv_patch(a, w, x, y, 0, p);
    if (!sv_cost(b, w, h, p, tx, ty, &k0)) {
        return 0;
    }
    if (!sv_cost(b, w, h, p, (float)x - nx, (float)y - ny, &k1)) {
        return 0;
    }
    return k1 < k0;
}

/* calc_grad / calc_grad_wrap, mask_gradient and calc_a of one pixel. */
__kernel void sv_grad(__global const float* img, __global const float* mask, int has_mask, int wrap, __global float* grad, __global float* tensor, int w, int h) {
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= w || y >= h) {
        return;
    }
    float c = img[y * w + x];
    float nx = 0x0p+0f;
    float dx = 0x0p+0f;
    float ny = 0x0p+0f;
    float dy = 0x0p+0f;
    if (wrap) {
        float left = img[y * w + (x + w - 1) % w];
        float right = img[y * w + (x + 1) % w];
        nx = (c - left) + 0x0p+0f;
        dx = 0x1p+0f;
        if (y >= 1) {
            ny = (c - img[(y - 1) * w + x]) + 0x0p+0f;
            dy = 0x1p+0f;
        }
        nx = nx + (right - c);
        dx = dx + 0x1p+0f;
        if (y + 1 <= h - 1) {
            ny = ny + (img[(y + 1) * w + x] - c);
            dy = dy + 0x1p+0f;
        }
    } else {
        if (x >= 1) {
            nx = (c - img[y * w + x - 1]) + 0x0p+0f;
            dx = 0x1p+0f;
        }
        if (y >= 1) {
            ny = (c - img[(y - 1) * w + x]) + 0x0p+0f;
            dy = 0x1p+0f;
        }
        if (x + 1 <= w - 1) {
            nx = nx + (img[y * w + x + 1] - c);
            dx = dx + 0x1p+0f;
        }
        if (y + 1 <= h - 1) {
            ny = ny + (img[(y + 1) * w + x] - c);
            dy = dy + 0x1p+0f;
        }
    }
    float gx = nx / dx;
    float gy = ny / dy;
    if (has_mask) {
        int out = sv_off(mask, w, x, y) || (y >= 1 && sv_off(mask, w, x, y - 1)) || (y + 1 < h && sv_off(mask, w, x, y + 1));
        if (wrap) {
            out = out || sv_off(mask, w, (x + w - 1) % w, y) || sv_off(mask, w, (x + 1) % w, y);
        } else {
            out = out || (x >= 1 && sv_off(mask, w, x - 1, y)) || (x + 1 < w && sv_off(mask, w, x + 1, y));
        }
        if (out) {
            gx = 0x0p+0f;
            gy = 0x0p+0f;
        }
    }
    int o = y * w + x;
    grad[o * 2] = gx;
    grad[o * 2 + 1] = gy;
    tensor[o * 3] = gx * gx;
    tensor[o * 3 + 1] = gx * gy;
    tensor[o * 3 + 2] = gy * gy;
}

/* calc_b / calc_b_with of one pixel: I1 = i1, I0 = i0, the flow, the level mask of Y. */
__kernel void sv_calc_b(__global const float* i1, __global const float* i0, __global const float* grad, __global const float* flow, __global const float* mask, int has_mask, int wrap, __global float* b, int w, int h) {
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= w || y >= h) {
        return;
    }
    int i = y * w + x;
    float fy = (float)y;
    float fh = (float)h;
    int pass = 0;
    float s = 0x0p+0f;
    if (wrap) {
        float sy = fy - flow[i * 2 + 1];
        int col;
        float fx;
        if (!(sy < 0x0p+0f || fh - 0x1p+0f < sy) && sv_wrap_target(x, flow[i * 2], w, &col, &fx)) {
            if (!has_mask || sv_bilinear1_wrapx_at(mask, w, h, col, fx, sy) >= 0x1.ff7ceep-1f) {
                s = sv_bilinear1_wrapx_at(i0, w, h, col, fx, sy);
                pass = 1;
            }
        }
    } else {
        float fx = (float)x;
        float fw = (float)w;
        float sx = fx - flow[i * 2];
        float sy = fy - flow[i * 2 + 1];
        if (!(sx < 0x0p+0f || fw - 0x1p+0f < sx || sy < 0x0p+0f || fh - 0x1p+0f < sy)) {
            if (!has_mask || rs_bilinear1(mask, w, h, sx, sy) >= 0x1.ff7ceep-1f) {
                s = rs_bilinear1(i0, w, h, sx, sy);
                pass = 1;
            }
        }
    }
    if (pass) {
        float r = s - i1[i];
        b[i * 2] = r * grad[i * 2];
        b[i * 2 + 1] = r * grad[i * 2 + 1];
    } else {
        b[i * 2] = 0x0p+0f;
        b[i * 2 + 1] = 0x0p+0f;
    }
}

/* new_disp of one pixel, in place on the flow (every item writes its own vector). */
__kernel void sv_new_disp(__global float* flow, __global const float* a, __global const float* b, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    float aa = a[i * 3];
    float ab = a[i * 3 + 1];
    float ad = a[i * 3 + 2];
    float bx = b[i * 2];
    float by = b[i * 2 + 1];
    float d0 = flow[i * 2];
    float d1 = flow[i * 2 + 1];
    float t = ab * ab - ad * aa;
    if (fabs(t) > 0x1.4p+5f) {
        d0 = (ab * by - ad * bx) / t + d0;
        d1 = (ab * bx - aa * by) / t + d1;
    }
    flow[i * 2] = d0;
    flow[i * 2 + 1] = d1;
}

/* apply_user of one pixel: flow = d / wt where wt != 0. */
__kernel void sv_apply_user(__global float* flow, __global const float* d, __global const float* wt, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    float k = wt[i];
    float f0 = flow[i * 2];
    float f1 = flow[i * 2 + 1];
    if (k != 0x0p+0f) {
        f0 = d[i * 2] / k;
        f1 = d[i * 2 + 1] / k;
    }
    flow[i * 2] = f0;
    flow[i * 2 + 1] = f1;
}

/* One chunk of sweep 1 (dir 0: x = x0, x0 + 1, .., neighbour x - 1) or sweep 2 (dir 1:
   x = x0, x0 - 1, .., neighbour x + 1) of the interior row g + 1, in place. */
__kernel void sv_sweep_rows(__global const float* a, __global const float* b, __global float* flow, __global const float* gate, int has_gate, int w, int h, int x0, int count, int dir) {
    int y = (int)get_global_id(0) + 1;
    if (y > h - 2) {
        return;
    }
    for (int k = 0; k < 512; k++) {
        if (k >= count) {
            break;
        }
        int x = dir == 0 ? x0 + k : x0 - k;
        int i = y * w + x;
        if (has_gate && gate[i] < 0x1.e66666p-1f) {
            continue;
        }
        int j = dir == 0 ? i - 1 : i + 1;
        float cx = flow[i * 2];
        float cy = flow[i * 2 + 1];
        float nx = flow[j * 2];
        float ny = flow[j * 2 + 1];
        if (sv_step(a, b, w, h, x, y, cx, cy, nx, ny, 0)) {
            flow[i * 2] = nx;
            flow[i * 2 + 1] = ny;
        }
    }
}

/* One chunk of sweep 3 (dir 0: y = y0, y0 + 1, .., neighbour y - 1) or sweep 4 (dir 1: y = y0,
   y0 - 1, .., neighbour y + 1) of the column g + xoff, in place; wrap selects Block::step_with. */
__kernel void sv_sweep_cols(__global const float* a, __global const float* b, __global float* flow, __global const float* gate, int has_gate, int w, int h, int xoff, int ncols, int y0, int count, int dir, int wrap) {
    int g = (int)get_global_id(0);
    if (g >= ncols) {
        return;
    }
    int x = g + xoff;
    for (int k = 0; k < 512; k++) {
        if (k >= count) {
            break;
        }
        int y = dir == 0 ? y0 + k : y0 - k;
        int i = y * w + x;
        if (has_gate && gate[i] < 0x1.e66666p-1f) {
            continue;
        }
        int j = dir == 0 ? i - w : i + w;
        float cx = flow[i * 2];
        float cy = flow[i * 2 + 1];
        float nx = flow[j * 2];
        float ny = flow[j * 2 + 1];
        if (sv_step(a, b, w, h, x, y, cx, cy, nx, ny, wrap)) {
            flow[i * 2] = nx;
            flow[i * 2 + 1] = ny;
        }
    }
}

/* Sweep 1 (right 0, neighbour x - 1 mod w) or 2 (right 1, x + 1 mod w) of the wrap solve: the
   pixel against its neighbour as both stood before the sweep (old), rows 1 .. h - 2. */
__kernel void sv_sweep_ring(__global const float* a, __global const float* b, __global const float* old, __global float* flow, __global const float* gate, int has_gate, int w, int h, int right) {
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= w || y >= h) {
        return;
    }
    int i = y * w + x;
    float o0 = old[i * 2];
    float o1 = old[i * 2 + 1];
    if (y >= 1 && y <= h - 2 && !(has_gate && gate[i] < 0x1.e66666p-1f)) {
        int nxc = right ? (x + 1) % w : (x + w - 1) % w;
        int j = y * w + nxc;
        float n0 = old[j * 2];
        float n1 = old[j * 2 + 1];
        if (sv_step(a, b, w, h, x, y, o0, o1, n0, n1, 1)) {
            o0 = n0;
            o1 = n1;
        }
    }
    flow[i * 2] = o0;
    flow[i * 2 + 1] = o1;
}

/* Whether row y holds a vector in range (all_flow_out_of_range, y only with wrap); 1 without a
   scan once the check is over (state[0] == 0). */
__kernel void sv_range_rows(__global const float* flow, __global const int* state, __global int* rows, int w, int h, int wrap) {
    int y = (int)get_global_id(0);
    if (y >= h) {
        return;
    }
    if (state[0] == 0) {
        rows[y] = 1;
        return;
    }
    float fw = (float)(w - 1);
    float fh = (float)(h - 1);
    float fy = (float)y;
    int any = 0;
    for (int x = 0; x < 16384; x++) {
        if (x >= w || any) {
            break;
        }
        float px = (float)x + flow[(y * w + x) * 2];
        float py = fy + flow[(y * w + x) * 2 + 1];
        if (wrap) {
            any = 0x0p+0f <= py && py <= fh;
        } else {
            any = 0x0p+0f <= px && px <= fw && 0x0p+0f <= py && py <= fh;
        }
    }
    rows[y] = any;
}

/* The driver's check: while state[0] (checking), a level with no row in range zeroes the flow
   (state[1] = 1), the first level with one ends the check. One item. */
__kernel void sv_range_decide(__global int* state, __global const int* rows, int h) {
    if (get_global_id(0) != 0) {
        return;
    }
    int check = state[0];
    int zero = 0;
    if (check != 0) {
        int any = 0;
        for (int y = 0; y < 16384; y++) {
            if (y >= h || any) {
                break;
            }
            any = rows[y] != 0;
        }
        if (any) {
            check = 0;
        } else {
            zero = 1;
        }
    }
    state[0] = check;
    state[1] = zero;
}

/* The zero fill of the check, in place on the n values of the flow. */
__kernel void sv_range_apply(__global float* flow, __global const int* state, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    float v = flow[i];
    if (state[1] != 0) {
        v = 0x0p+0f;
    }
    flow[i] = v;
}

/* scale_values: v = f * v. */
__kernel void sv_scale(__global float* data, float f, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    data[i] = f * data[i];
}

/* The zero start of the flow. */
__kernel void sv_zero(__global float* data, int n) {
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    data[i] = 0x0p+0f;
}

/* upsample_flow_wrapx: (sx, sy, rx, ry) from the host, source (x * rx, y * ry) with
   sv_bilinear2_wrapx, the vector times (sx, sy). */
__kernel void sv_upsample_flow_wrapx(__global const float* src, int sw, int sh, __global float* dst, int dw, int dh, float sx, float sy, float rx, float ry) {
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= dw || y >= dh) {
        return;
    }
    float fy = (float)y * ry;
    float2 v = sv_bilinear2_wrapx(src, sw, sh, (float)x * rx, fy);
    int o = (y * dw + x) * 2;
    dst[o] = v.x * sx;
    dst[o + 1] = v.y * sy;
}
"#;

// ---- transcriptions ---------------------------------------------------------------------------

/// The level mask threshold of the kernels (`0x1.e66666p-1f`), `flow::MASK_ON`.
pub const SV_MASK_ON: f32 = f32::from_bits(0x3f73_3333);
/// The warped mask threshold of `sv_calc_b` (`0x1.ff7ceep-1f`), `0.999` of `calc_b_with`.
pub const SV_WARPED_MASK_MIN: f32 = f32::from_bits(0x3f7f_be77);
/// The update threshold of `sv_new_disp` (`0x1.4p+5f`).
pub const SV_UPDATE_MIN: f32 = 40.0;

pub fn sv_wrap_target(x: i32, u: f32, w: i32) -> Option<(i32, f32)> {
    let d = -u;
    if !(d.abs() <= 65536.0f32 * w as f32) {
        return None;
    }
    let whole = d.floor();
    let mut f = d - whole;
    let t = x as i64 + whole as i64;
    let mut m = t % w as i64;
    if m < 0 {
        m += w as i64;
    }
    let mut c = m as i32;
    if f >= 1.0f32 {
        c = if c + 1 == w { 0 } else { c + 1 };
        f = 0.0f32;
    }
    Some((c, f))
}

pub fn sv_bilinear1_wrapx_at(img: &[f32], w: i32, h: i32, ix: i32, fx: f32, y: f32) -> f32 {
    let x1 = if ix + 1 > w - 1 { 0 } else { ix + 1 };
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

pub fn sv_bilinear2_wrapx(img: &[f32], w: i32, h: i32, x: f32, y: f32) -> [f32; 2] {
    let (ix, x1, fx) = crate::device::kernels::resample::rs_wrapx_cell(x, w);
    let (iy, fy) = rs_cell(y, h);
    let r0 = iy * w;
    let r1 = (iy + 1) * w;
    let at = |c: i32, k: i32| img[(c * 2 + k) as usize];
    let a00 = at(r0 + ix, 0);
    let a10 = at(r0 + x1, 0);
    let a01 = at(r1 + ix, 0);
    let a11 = at(r1 + x1, 0);
    let atop = (a10 - a00) * fx + a00;
    let abot = (a11 - a01) * fx + a01;
    let b00 = at(r0 + ix, 1);
    let b10 = at(r0 + x1, 1);
    let b01 = at(r1 + ix, 1);
    let b11 = at(r1 + x1, 1);
    let btop = (b10 - b00) * fx + b00;
    let bbot = (b11 - b01) * fx + b01;
    [(abot - atop) * fy + atop, (bbot - btop) * fy + btop]
}

pub fn sv_off(m: &[f32], w: i32, x: i32, y: i32) -> bool {
    m[(y * w + x) as usize] < SV_MASK_ON
}

pub fn sv_patch(a: &[f32], w: i32, x: i32, y: i32, wrap: bool) -> [f32; 9] {
    let c0 = if wrap { (x + w - 1) % w } else { x - 1 };
    let c2 = if wrap { (x + 1) % w } else { x + 1 };
    let mut p = [0.0f32; 9];
    for r in 0..3 {
        let base = (y + r - 1) * w;
        p[(r * 3) as usize] = a[(base + c0) as usize];
        p[(r * 3 + 1) as usize] = a[(base + x) as usize];
        p[(r * 3 + 2) as usize] = a[(base + c2) as usize];
    }
    p
}

pub fn sv_cost_sum(b: &[f32], w: i32, p: &[f32; 9], rbase: i32, c: [i32; 4], fx: f32, fy: f32) -> f32 {
    let w11 = fy * fx;
    let w10 = fx - w11;
    let w01 = fy - w11;
    let w00 = ((1.0f32 - fx) - fy) + w11;
    let mut sq = [0.0f32; 9];
    for j in 0..3i32 {
        let r0 = rbase + j * w;
        let r1 = r0 + w;
        for i in 0..3usize {
            let t = |r: i32, k: usize| b[(r + c[k]) as usize];
            let q = ((w00 * t(r0, i) + w10 * t(r0, i + 1)) + w01 * t(r1, i)) + w11 * t(r1, i + 1);
            let d = p[(j * 3) as usize + i] - q;
            sq[(j * 3) as usize + i] = d * d;
        }
    }
    let mut s = sq[0] + sq[1];
    for v in sq.iter().take(8).skip(2) {
        s = s + *v;
    }
    sq[8] + s
}

pub fn sv_cost_row(h: i32, y2: f32) -> Option<(i32, f32)> {
    let hm2 = (h - 2) as f32;
    if y2 > hm2 {
        return None;
    }
    let (iy, fy) = if y2 == hm2 {
        (rs::cvtt_i32_f(hm2 - 1.0f32), 1.0f32)
    } else {
        let i = rs::cvtt_i32_f(y2);
        (i, y2 - i as f32)
    };
    if iy >= 1 {
        Some((iy, fy))
    } else {
        None
    }
}

pub fn sv_cost(b: &[f32], w: i32, h: i32, p: &[f32; 9], x2: f32, y2: f32) -> Option<f32> {
    let wm2 = (w - 2) as f32;
    let hm2 = (h - 2) as f32;
    if x2 > wm2 || y2 > hm2 {
        return None;
    }
    let (ix, fx) = if x2 == wm2 {
        (rs::cvtt_i32_f(wm2 - 1.0f32), 1.0f32)
    } else {
        let i = rs::cvtt_i32_f(x2);
        (i, x2 - i as f32)
    };
    let row = sv_cost_row(h, y2);
    let (iy, fy) = match row {
        Some(r) if ix >= 1 => r,
        _ => return None,
    };
    Some(sv_cost_sum(b, w, p, (iy - 1) * w, [ix - 1, ix, ix + 1, ix + 2], fx, fy))
}

#[allow(clippy::too_many_arguments)]
pub fn sv_cost_wrap(b: &[f32], w: i32, h: i32, p: &[f32; 9], col: i32, fx: f32, y2: f32) -> Option<f32> {
    let (iy, fy) = sv_cost_row(h, y2)?;
    let c = [(col + w - 1) % w, col, (col + 1) % w, (col + 2) % w];
    Some(sv_cost_sum(b, w, p, (iy - 1) * w, c, fx, fy))
}

#[allow(clippy::too_many_arguments)]
pub fn sv_step(a: &[f32], b: &[f32], w: i32, h: i32, x: i32, y: i32, cx: f32, cy: f32, nx: f32, ny: f32, wrap: bool) -> bool {
    let h1 = (h - 1) as f32;
    if wrap {
        let ty = y as f32 - cy;
        if !(0.0f32 <= ty && ty <= h1) {
            return false;
        }
        let p = sv_patch(a, w, x, y, true);
        let Some(k0) = sv_wrap_target(x, cx, w).and_then(|(col, f)| sv_cost_wrap(b, w, h, &p, col, f, ty)) else {
            return false;
        };
        let ny2 = y as f32 - ny;
        let Some(k1) = sv_wrap_target(x, nx, w).and_then(|(col, f)| sv_cost_wrap(b, w, h, &p, col, f, ny2)) else {
            return false;
        };
        return k1 < k0;
    }
    let tx = x as f32 - cx;
    let ty = y as f32 - cy;
    let w1 = (w - 1) as f32;
    if !(0.0f32 <= tx && tx <= w1 && 0.0f32 <= ty && ty <= h1) {
        return false;
    }
    let p = sv_patch(a, w, x, y, false);
    let Some(k0) = sv_cost(b, w, h, &p, tx, ty) else {
        return false;
    };
    let Some(k1) = sv_cost(b, w, h, &p, x as f32 - nx, y as f32 - ny) else {
        return false;
    };
    k1 < k0
}

#[allow(clippy::too_many_arguments)]
pub fn sv_grad_item(x: usize, y: usize, img: &[f32], mask: &[f32], has_mask: bool, wrap: bool, grad: &mut Global<f32>, tensor: &mut Global<f32>, w: i32, h: i32) {
    let (x, y) = (x as i32, y as i32);
    if x >= w || y >= h {
        return;
    }
    let at = |xx: i32, yy: i32| img[(yy * w + xx) as usize];
    let c = at(x, y);
    let mut nx = 0.0f32;
    let mut dx = 0.0f32;
    let mut ny = 0.0f32;
    let mut dy = 0.0f32;
    if wrap {
        let left = at((x + w - 1) % w, y);
        let right = at((x + 1) % w, y);
        nx = (c - left) + 0.0f32;
        dx = 1.0f32;
        if y >= 1 {
            ny = (c - at(x, y - 1)) + 0.0f32;
            dy = 1.0f32;
        }
        nx = nx + (right - c);
        dx = dx + 1.0f32;
        if y + 1 <= h - 1 {
            ny = ny + (at(x, y + 1) - c);
            dy = dy + 1.0f32;
        }
    } else {
        if x >= 1 {
            nx = (c - at(x - 1, y)) + 0.0f32;
            dx = 1.0f32;
        }
        if y >= 1 {
            ny = (c - at(x, y - 1)) + 0.0f32;
            dy = 1.0f32;
        }
        if x + 1 <= w - 1 {
            nx = nx + (at(x + 1, y) - c);
            dx = dx + 1.0f32;
        }
        if y + 1 <= h - 1 {
            ny = ny + (at(x, y + 1) - c);
            dy = dy + 1.0f32;
        }
    }
    let mut gx = nx / dx;
    let mut gy = ny / dy;
    if has_mask {
        let mut out = sv_off(mask, w, x, y) || (y >= 1 && sv_off(mask, w, x, y - 1)) || (y + 1 < h && sv_off(mask, w, x, y + 1));
        if wrap {
            out = out || sv_off(mask, w, (x + w - 1) % w, y) || sv_off(mask, w, (x + 1) % w, y);
        } else {
            out = out || (x >= 1 && sv_off(mask, w, x - 1, y)) || (x + 1 < w && sv_off(mask, w, x + 1, y));
        }
        if out {
            gx = 0.0f32;
            gy = 0.0f32;
        }
    }
    let o = (y * w + x) as usize;
    grad.set(o * 2, gx);
    grad.set(o * 2 + 1, gy);
    tensor.set(o * 3, gx * gx);
    tensor.set(o * 3 + 1, gx * gy);
    tensor.set(o * 3 + 2, gy * gy);
}

#[allow(clippy::too_many_arguments)]
pub fn sv_calc_b_item(
    x: usize,
    y: usize,
    i1: &[f32],
    i0: &[f32],
    grad: &[f32],
    flow: &[f32],
    mask: &[f32],
    has_mask: bool,
    wrap: bool,
    b: &mut Global<f32>,
    w: i32,
    h: i32,
) {
    let (x, y) = (x as i32, y as i32);
    if x >= w || y >= h {
        return;
    }
    let i = (y * w + x) as usize;
    let fy = y as f32;
    let fh = h as f32;
    let mut pass = false;
    let mut s = 0.0f32;
    if wrap {
        let sy = fy - flow[i * 2 + 1];
        let target = if !(sy < 0.0f32 || fh - 1.0f32 < sy) { sv_wrap_target(x, flow[i * 2], w) } else { None };
        if let Some((col, fx)) = target {
            if !has_mask || sv_bilinear1_wrapx_at(mask, w, h, col, fx, sy) >= SV_WARPED_MASK_MIN {
                s = sv_bilinear1_wrapx_at(i0, w, h, col, fx, sy);
                pass = true;
            }
        }
    } else {
        let fx = x as f32;
        let fw = w as f32;
        let sx = fx - flow[i * 2];
        let sy = fy - flow[i * 2 + 1];
        if !(sx < 0.0f32 || fw - 1.0f32 < sx || sy < 0.0f32 || fh - 1.0f32 < sy)
            && (!has_mask || rs_bilinear1(mask, w, h, sx, sy) >= SV_WARPED_MASK_MIN)
        {
            s = rs_bilinear1(i0, w, h, sx, sy);
            pass = true;
        }
    }
    if pass {
        let r = s - i1[i];
        b.set(i * 2, r * grad[i * 2]);
        b.set(i * 2 + 1, r * grad[i * 2 + 1]);
    } else {
        b.set(i * 2, 0.0f32);
        b.set(i * 2 + 1, 0.0f32);
    }
}

pub fn sv_new_disp_item(i: usize, flow: &[f32], a: &[f32], b: &[f32], out: &mut Global<f32>, n: i32) {
    if i as i32 >= n {
        return;
    }
    let aa = a[i * 3];
    let ab = a[i * 3 + 1];
    let ad = a[i * 3 + 2];
    let bx = b[i * 2];
    let by = b[i * 2 + 1];
    let mut d0 = flow[i * 2];
    let mut d1 = flow[i * 2 + 1];
    let t = ab * ab - ad * aa;
    if t.abs() > SV_UPDATE_MIN {
        d0 = (ab * by - ad * bx) / t + d0;
        d1 = (ab * bx - aa * by) / t + d1;
    }
    out.set(i * 2, d0);
    out.set(i * 2 + 1, d1);
}

pub fn sv_apply_user_item(i: usize, flow: &[f32], d: &[f32], wt: &[f32], out: &mut Global<f32>, n: i32) {
    if i as i32 >= n {
        return;
    }
    let k = wt[i];
    let mut f0 = flow[i * 2];
    let mut f1 = flow[i * 2 + 1];
    if k != 0.0f32 {
        f0 = d[i * 2] / k;
        f1 = d[i * 2 + 1] / k;
    }
    out.set(i * 2, f0);
    out.set(i * 2 + 1, f1);
}

/// What a sweep transcription reads and writes the flow through (the simulator checks that an
/// item touches only elements no other item of the launch touches).
pub trait SweepFlow {
    fn get(&mut self, i: usize) -> f32;
    fn set(&mut self, i: usize, v: f32);
}

#[allow(clippy::too_many_arguments)]
pub fn sv_sweep_rows_item(g: usize, a: &[f32], b: &[f32], flow: &mut dyn SweepFlow, gate: &[f32], has_gate: bool, w: i32, h: i32, x0: i32, count: i32, dir: i32) {
    let y = g as i32 + 1;
    if y > h - 2 {
        return;
    }
    for k in 0..SV_SWEEP_CHUNK as i32 {
        if k >= count {
            break;
        }
        let x = if dir == 0 { x0 + k } else { x0 - k };
        let i = (y * w + x) as usize;
        if has_gate && gate[i] < SV_MASK_ON {
            continue;
        }
        let j = if dir == 0 { i - 1 } else { i + 1 };
        let cx = flow.get(i * 2);
        let cy = flow.get(i * 2 + 1);
        let nx = flow.get(j * 2);
        let ny = flow.get(j * 2 + 1);
        if sv_step(a, b, w, h, x, y, cx, cy, nx, ny, false) {
            flow.set(i * 2, nx);
            flow.set(i * 2 + 1, ny);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn sv_sweep_cols_item(
    g: usize,
    a: &[f32],
    b: &[f32],
    flow: &mut dyn SweepFlow,
    gate: &[f32],
    has_gate: bool,
    w: i32,
    h: i32,
    xoff: i32,
    ncols: i32,
    y0: i32,
    count: i32,
    dir: i32,
    wrap: bool,
) {
    let g = g as i32;
    if g >= ncols {
        return;
    }
    let x = g + xoff;
    for k in 0..SV_SWEEP_CHUNK as i32 {
        if k >= count {
            break;
        }
        let y = if dir == 0 { y0 + k } else { y0 - k };
        let i = (y * w + x) as usize;
        if has_gate && gate[i] < SV_MASK_ON {
            continue;
        }
        let j = if dir == 0 { i - w as usize } else { i + w as usize };
        let cx = flow.get(i * 2);
        let cy = flow.get(i * 2 + 1);
        let nx = flow.get(j * 2);
        let ny = flow.get(j * 2 + 1);
        if sv_step(a, b, w, h, x, y, cx, cy, nx, ny, wrap) {
            flow.set(i * 2, nx);
            flow.set(i * 2 + 1, ny);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn sv_sweep_ring_item(x: usize, y: usize, a: &[f32], b: &[f32], old: &[f32], flow: &mut Global<f32>, gate: &[f32], has_gate: bool, w: i32, h: i32, right: bool) {
    let (x, y) = (x as i32, y as i32);
    if x >= w || y >= h {
        return;
    }
    let i = (y * w + x) as usize;
    let mut o0 = old[i * 2];
    let mut o1 = old[i * 2 + 1];
    if y >= 1 && y <= h - 2 && !(has_gate && gate[i] < SV_MASK_ON) {
        let nxc = if right { (x + 1) % w } else { (x + w - 1) % w };
        let j = (y * w + nxc) as usize;
        let n0 = old[j * 2];
        let n1 = old[j * 2 + 1];
        if sv_step(a, b, w, h, x, y, o0, o1, n0, n1, true) {
            o0 = n0;
            o1 = n1;
        }
    }
    flow.set(i * 2, o0);
    flow.set(i * 2 + 1, o1);
}

pub fn sv_range_rows_item(y: usize, flow: &[f32], state: &[i32], rows: &mut Global<i32>, w: i32, h: i32, wrap: bool) {
    let y = y as i32;
    if y >= h {
        return;
    }
    if state[0] == 0 {
        rows.set(y as usize, 1);
        return;
    }
    let fw = (w - 1) as f32;
    let fh = (h - 1) as f32;
    let fy = y as f32;
    let mut any = false;
    for x in 0..SV_LINE_MAX as i32 {
        if x >= w || any {
            break;
        }
        let px = x as f32 + flow[((y * w + x) * 2) as usize];
        let py = fy + flow[((y * w + x) * 2 + 1) as usize];
        any = if wrap { 0.0f32 <= py && py <= fh } else { 0.0f32 <= px && px <= fw && 0.0f32 <= py && py <= fh };
    }
    rows.set(y as usize, any as i32);
}

pub fn sv_range_decide_item(g: usize, state: &[i32], rows: &[i32], out: &mut Global<i32>, h: i32) {
    if g != 0 {
        return;
    }
    let mut check = state[0];
    let mut zero = 0;
    if check != 0 {
        let mut any = false;
        for y in 0..SV_LINE_MAX as i32 {
            if y >= h || any {
                break;
            }
            any = rows[y as usize] != 0;
        }
        if any {
            check = 0;
        } else {
            zero = 1;
        }
    }
    out.set(0, check);
    out.set(1, zero);
}

pub fn sv_range_apply_item(i: usize, flow: &[f32], state: &[i32], out: &mut Global<f32>, n: i32) {
    if i as i32 >= n {
        return;
    }
    let mut v = flow[i];
    if state[1] != 0 {
        v = 0.0f32;
    }
    out.set(i, v);
}

pub fn sv_scale_item(i: usize, data: &[f32], f: f32, out: &mut Global<f32>, n: i32) {
    if i as i32 >= n {
        return;
    }
    out.set(i, f * data[i]);
}

pub fn sv_zero_item(i: usize, out: &mut Global<f32>, n: i32) {
    if i as i32 >= n {
        return;
    }
    out.set(i, 0.0f32);
}

#[allow(clippy::too_many_arguments)]
pub fn sv_upsample_flow_wrapx_item(x: usize, y: usize, src: &[f32], sw: i32, sh: i32, dst: &mut Global<f32>, dw: i32, dh: i32, scale: [f32; 4]) {
    let [sx, sy, rx, ry] = scale;
    let (x, y) = (x as i32, y as i32);
    if x >= dw || y >= dh {
        return;
    }
    let fy = y as f32 * ry;
    let v = sv_bilinear2_wrapx(src, sw, sh, x as f32 * rx, fy);
    let o = ((y * dw + x) * 2) as usize;
    dst.set(o, v[0] * sx);
    dst.set(o + 1, v[1] * sy);
}
