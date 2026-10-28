//! Device kernels of the Weighted Blend chain (`filters.rs`), prefix `wb_`, with a line for line
//! Rust transcription of every kernel and helper (`*_item`, one work item, statement by
//! statement) that the simulator runs. The separable passes of the blur are the `cv_hpass` and
//! `cv_vpass` kernels of [`super::conv`] (four interleaved lanes, the `-1.0` sentinel), the lerp,
//! blend, warps and Smart Blend those of [`super::warpstage`]; this source holds the rest.
//!
//! Precision contract (V: `filters.rs` is the reference; every kernel keeps its operation order
//! and precision, hexadecimal literals, the prelude's range tested conversions):
//! * `wb_diff_mask_*`: `filters::difference_mask`: `d = fabs(a - b)` in float for the colour
//!   channels 1, 2, 3, `sum = (d1 + d2) + d3`, `255` when `sum >= limit` (a NaN sum gives 0);
//!   `limit` is `filters::mask_limit`, computed on the host.
//! * `wb_mask_prune`, `wb_mask_fill`, `wb_mask_dilate`: `mask_prune`, `mask_fill` and
//!   `mask_dilate_rows`, from a snapshot (`src`) into `dst`, one work item per pixel, the integer
//!   ring sum of `filters::ring_sum`. The host skips the launch where the CPU function returns
//!   at once (too small a mask).
//! * `wb_border_rows`, `wb_border_cols`: `mask_replicate_border` in place, its two phases: the
//!   rows (one work item per column, the rows above and below the inner rectangle in the CPU's
//!   order) and then the columns (one work item per row, the left then the right columns). The
//!   rectangle edges `fy0`, `fy1`, `fx0`, `fx1` and the source rows and columns arrive from the
//!   host, computed by the CPU's expressions; every item touches only its own column or row, so
//!   the result is the CPU's sequential one. The loops run at most 2 (rows above), 3 (rows
//!   below), 2 (left) and 3 (right) times, which the host checks against the CPU's counts.
//! * `wb_invert_mask`: `invert_mask` (`255 - b`).
//! * `wb_to_float_*`, `wb_round_*`, `wb_from_float_*`: the loads and stores of `filters::blur`
//!   around the two passes at 8 and 16 bpc: `(float)v`, then after the horizontal pass
//!   `(float)store_int(v)` (the stored and reloaded sample), after the vertical pass
//!   `store_int(v)`, with `store_int(v, max)`: `t = cvtt(v + 0.5f)`, `f = (float)t`, `f > 0 ? f
//!   : 0`, `f < max ? f : max`, `cvtt(f + 0.5f)`. Float images run the passes on their samples.
//! * `wb_composite_*`: `composite_masked`: `m == 0` takes `src`; else `f = (float)m / 255.0f`,
//!   `g = (float)(1.0 - (double)f)` and `trunc((src * g + dest * f) + 0.5f)` (low byte or word)
//!   at 8 and 16 bpc, `(g * src + f * dest) + 0.0f` in float.

use crate::device::kernels::prelude::rs;
use crate::device::sim::Global;

pub const WB: &str = r#"
/* difference_mask: 255 where (|dR| + |dG|) + |dB| >= limit, d = a - b in float. */
__kernel void wb_diff_mask_u8(__global const uchar* a, __global const uchar* b, __global uchar* mask, float limit, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t p = i * 4;
    float d1 = fabs((float)a[p + 1] - (float)b[p + 1]);
    float d2 = fabs((float)a[p + 2] - (float)b[p + 2]);
    float d3 = fabs((float)a[p + 3] - (float)b[p + 3]);
    float sum = (d1 + d2) + d3;
    mask[i] = sum >= limit ? (uchar)255 : (uchar)0;
}

__kernel void wb_diff_mask_u16(__global const ushort* a, __global const ushort* b, __global uchar* mask, float limit, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t p = i * 4;
    float d1 = fabs((float)a[p + 1] - (float)b[p + 1]);
    float d2 = fabs((float)a[p + 2] - (float)b[p + 2]);
    float d3 = fabs((float)a[p + 3] - (float)b[p + 3]);
    float sum = (d1 + d2) + d3;
    mask[i] = sum >= limit ? (uchar)255 : (uchar)0;
}

__kernel void wb_diff_mask_f32(__global const float* a, __global const float* b, __global uchar* mask, float limit, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    size_t p = i * 4;
    float d1 = fabs(a[p + 1] - b[p + 1]);
    float d2 = fabs(a[p + 2] - b[p + 2]);
    float d3 = fabs(a[p + 3] - b[p + 3]);
    float sum = (d1 + d2) + d3;
    mask[i] = sum >= limit ? (uchar)255 : (uchar)0;
}

/* ring_sum of filters.rs: the columns x, x + 1, x + 2 of the rows r0, r1, r2 without the middle
   sample of r1, in the CPU's order. */
int wb_ring(__global const uchar* t, int w, int x, int r0, int r1, int r2) {
    int s = (int)t[r0 * w + x + 1] + (int)t[r0 * w + x];
    s = s + (int)t[r0 * w + x + 2];
    s = s + (int)t[r1 * w + x];
    s = s + (int)t[r1 * w + x + 2];
    s = s + (int)t[r2 * w + x];
    s = s + (int)t[r2 * w + x + 1];
    s = s + (int)t[r2 * w + x + 2];
    return s;
}

/* mask_prune: a pixel with 1 <= x <= w - 2 and 2 <= y <= h - 3 that is not zero becomes zero
   when the ring of the rows y - 2, y, y + 2 sums below limit (h >= 5, w >= 3). */
__kernel void wb_mask_prune(__global const uchar* src, __global uchar* dst, int w, int h, int limit) {
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= w || y >= h) {
        return;
    }
    uchar v = src[y * w + x];
    if (x >= 1 && x <= w - 2 && y >= 2 && y <= h - 3 && v != (uchar)0) {
        int s = wb_ring(src, w, x - 1, y - 2, y, y + 2);
        if (s < limit) {
            v = (uchar)0;
        }
    }
    dst[y * w + x] = v;
}

/* mask_fill: a zero pixel with 1 <= x <= w - 2 and 1 <= y <= h - 2 becomes value when the full
   ring sums above limit (h >= 3, w >= 3). */
__kernel void wb_mask_fill(__global const uchar* src, __global uchar* dst, int w, int h, int limit, int value) {
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= w || y >= h) {
        return;
    }
    uchar v = src[y * w + x];
    if (x >= 1 && x <= w - 2 && y >= 1 && y <= h - 2 && v == (uchar)0) {
        int s = wb_ring(src, w, x - 1, y - 1, y, y + 1);
        if (limit < s) {
            v = (uchar)value;
        }
    }
    dst[y * w + x] = v;
}

/* mask_dilate_rows: a pixel of the rows 1 ..= h - 2 becomes 255 when the pixel above or below
   is 255 (h >= 3). */
__kernel void wb_mask_dilate(__global const uchar* src, __global uchar* dst, int w, int h) {
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= w || y >= h) {
        return;
    }
    uchar v = src[y * w + x];
    if (y >= 1 && y <= h - 2) {
        if (src[(y - 1) * w + x] == (uchar)255 || src[(y + 1) * w + x] == (uchar)255) {
            v = (uchar)255;
        }
    }
    dst[y * w + x] = v;
}

/* The outside test of mask_replicate_border, in float. */
int wb_outside(int x, int y, float fx0, float fx1, float fy0, float fy1) {
    float xf = (float)x;
    float yf = (float)y;
    return yf < fy0 || fy1 < yf || fx1 < xf || xf < fx0;
}

/* mask_replicate_border, the row phase, in place, one work item per column x: the rows y with
   (float)y < fy0 take the row top (when it exists), then the rows from y1 = (int)fy1 to h - 1
   take the row bottom, every one only where the pixel is outside the inner rectangle. */
__kernel void wb_border_rows(__global uchar* m, int w, int h, float fx0, float fx1, float fy0, float fy1, int top, int y1, int bottom) {
    int x = (int)get_global_id(0);
    if (x >= w) {
        return;
    }
    for (int q = 0; q < 2; q++) {
        int y = q;
        if (!((float)y < fy0)) {
            break;
        }
        if (wb_outside(x, y, fx0, fx1, fy0, fy1) && y >= 0 && y < h && top >= 0 && top < h) {
            m[y * w + x] = m[top * w + x];
        }
    }
    for (int q = 0; q < 3; q++) {
        int y = y1 + q;
        if (y >= h) {
            break;
        }
        if (wb_outside(x, y, fx0, fx1, fy0, fy1) && y >= 0 && bottom >= 0 && bottom < h) {
            m[y * w + x] = m[bottom * w + x];
        }
    }
}

/* mask_replicate_border, the column phase, in place, one work item per row y: the columns x
   with (float)x < fx0 take the column left, then for j from 0 the column right + j takes the
   column right as long as x1 + j < w. */
__kernel void wb_border_cols(__global uchar* m, int w, int h, float fx0, int left, int x1, int right) {
    int y = (int)get_global_id(0);
    if (y >= h) {
        return;
    }
    for (int q = 0; q < 2; q++) {
        int x = q;
        if (!((float)x < fx0)) {
            break;
        }
        if (x >= 0 && x < w && left >= 0 && left < w) {
            m[y * w + x] = m[y * w + left];
        }
    }
    for (int j = 0; j < 3; j++) {
        if (x1 + j >= w) {
            break;
        }
        int d = right + j;
        if (d >= 0 && d < w && right >= 0 && right < w) {
            m[y * w + d] = m[y * w + right];
        }
    }
}

/* invert_mask: 255 - b, in place. */
__kernel void wb_invert_mask(__global uchar* m, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    m[i] = (uchar)(255 - (int)m[i]);
}

/* The store of the integer blur: trunc(v + 0.5) as float, clamped to 0..max, trunc(x + 0.5). */
int wb_store_int(float v, float max) {
    int t = cvtt_i32_f(v + 0x1p-1f);
    float f = (float)t;
    f = f > 0x0p+0f ? f : 0x0p+0f;
    f = f < max ? f : max;
    return cvtt_i32_f(f + 0x1p-1f);
}

__kernel void wb_to_float_u8(__global const uchar* src, __global float* dst, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    dst[i] = (float)src[i];
}

__kernel void wb_to_float_u16(__global const ushort* src, __global float* dst, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    dst[i] = (float)src[i];
}

/* The sample pass 1 stores and pass 2 loads again. */
__kernel void wb_round_u8(__global const float* src, __global float* dst, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    dst[i] = (float)(uchar)wb_store_int(src[i], 0x1.fep+7f);
}

__kernel void wb_round_u16(__global const float* src, __global float* dst, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    dst[i] = (float)(ushort)wb_store_int(src[i], 0x1.fffep+15f);
}

__kernel void wb_from_float_u8(__global const float* src, __global uchar* dst, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    dst[i] = (uchar)wb_store_int(src[i], 0x1.fep+7f);
}

__kernel void wb_from_float_u16(__global const float* src, __global ushort* dst, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    dst[i] = (ushort)wb_store_int(src[i], 0x1.fffep+15f);
}

/* composite_masked, in place on dest, one work item per sample; mask is per pixel. */
__kernel void wb_composite_u8(__global uchar* dest, __global const uchar* src, __global const uchar* mask, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    uchar m = mask[i / 4];
    if (m == (uchar)0) {
        dest[i] = src[i];
        return;
    }
    float f = (float)m / 0x1.fep+7f;
    float g = (float)(0x1p+0 - (double)f);
    dest[i] = trunc_u8_f(((float)src[i] * g + (float)dest[i] * f) + 0x1p-1f);
}

__kernel void wb_composite_u16(__global ushort* dest, __global const ushort* src, __global const uchar* mask, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    uchar m = mask[i / 4];
    if (m == (uchar)0) {
        dest[i] = src[i];
        return;
    }
    float f = (float)m / 0x1.fep+7f;
    float g = (float)(0x1p+0 - (double)f);
    dest[i] = trunc_u16_f(((float)src[i] * g + (float)dest[i] * f) + 0x1p-1f);
}

__kernel void wb_composite_f32(__global float* dest, __global const float* src, __global const uchar* mask, uint n) {
    size_t i = get_global_id(0);
    if (i >= n) {
        return;
    }
    uchar m = mask[i / 4];
    if (m == (uchar)0) {
        dest[i] = src[i];
        return;
    }
    float f = (float)m / 0x1.fep+7f;
    float g = (float)(0x1p+0 - (double)f);
    dest[i] = (g * src[i] + f * dest[i]) + 0x0p+0f;
}
"#;

// ---- transcriptions ---------------------------------------------------------------------------

/// The lanes of one sample type: `as f32` of the stored value.
pub trait WbSample: Copy {
    fn f(self) -> f32;
}

impl WbSample for u8 {
    fn f(self) -> f32 {
        self as f32
    }
}

impl WbSample for u16 {
    fn f(self) -> f32 {
        self as f32
    }
}

impl WbSample for f32 {
    fn f(self) -> f32 {
        self
    }
}

pub fn wb_diff_mask_item<T: WbSample>(i: usize, a: &[T], b: &[T], mask: &mut Global<u8>, limit: f32, n: u32) {
    if i >= n as usize {
        return;
    }
    let p = i * 4;
    let d1 = (a[p + 1].f() - b[p + 1].f()).abs();
    let d2 = (a[p + 2].f() - b[p + 2].f()).abs();
    let d3 = (a[p + 3].f() - b[p + 3].f()).abs();
    let sum = (d1 + d2) + d3;
    mask.set(i, if sum >= limit { 255 } else { 0 });
}

pub fn wb_ring(t: &[u8], w: i32, x: i32, r0: i32, r1: i32, r2: i32) -> i32 {
    let at = |r: i32, dx: i32| t[(r * w + x + dx) as usize] as i32;
    let mut s = at(r0, 1) + at(r0, 0);
    s = s + at(r0, 2);
    s = s + at(r1, 0);
    s = s + at(r1, 2);
    s = s + at(r2, 0);
    s = s + at(r2, 1);
    s = s + at(r2, 2);
    s
}

pub fn wb_mask_prune_item(x: usize, y: usize, src: &[u8], dst: &mut Global<u8>, w: i32, h: i32, limit: i32) {
    let (x, y) = (x as i32, y as i32);
    if x >= w || y >= h {
        return;
    }
    let mut v = src[(y * w + x) as usize];
    if x >= 1 && x <= w - 2 && y >= 2 && y <= h - 3 && v != 0 {
        let s = wb_ring(src, w, x - 1, y - 2, y, y + 2);
        if s < limit {
            v = 0;
        }
    }
    dst.set((y * w + x) as usize, v);
}

#[allow(clippy::too_many_arguments)]
pub fn wb_mask_fill_item(x: usize, y: usize, src: &[u8], dst: &mut Global<u8>, w: i32, h: i32, limit: i32, value: i32) {
    let (x, y) = (x as i32, y as i32);
    if x >= w || y >= h {
        return;
    }
    let mut v = src[(y * w + x) as usize];
    if x >= 1 && x <= w - 2 && y >= 1 && y <= h - 2 && v == 0 {
        let s = wb_ring(src, w, x - 1, y - 1, y, y + 1);
        if limit < s {
            v = value as u8;
        }
    }
    dst.set((y * w + x) as usize, v);
}

pub fn wb_mask_dilate_item(x: usize, y: usize, src: &[u8], dst: &mut Global<u8>, w: i32, h: i32) {
    let (x, y) = (x as i32, y as i32);
    if x >= w || y >= h {
        return;
    }
    let mut v = src[(y * w + x) as usize];
    if y >= 1 && y <= h - 2 && (src[((y - 1) * w + x) as usize] == 255 || src[((y + 1) * w + x) as usize] == 255) {
        v = 255;
    }
    dst.set((y * w + x) as usize, v);
}

pub fn wb_outside(x: i32, y: i32, fx0: f32, fx1: f32, fy0: f32, fy1: f32) -> bool {
    let xf = x as f32;
    let yf = y as f32;
    yf < fy0 || fy1 < yf || fx1 < xf || xf < fx0
}

/// The arguments of the two border kernels, computed on the host (see `wbstage::border_args`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BorderArgs {
    pub w: i32,
    pub h: i32,
    pub fx0: f32,
    pub fx1: f32,
    pub fy0: f32,
    pub fy1: f32,
    pub top: i32,
    pub y1: i32,
    pub bottom: i32,
    pub left: i32,
    pub x1: i32,
    pub right: i32,
}

/// `wb_border_rows` in place: `m` is the mask, read and written by this item only in column `x`
/// (the simulator gives every item the whole buffer, so the item order cannot matter).
pub fn wb_border_rows_item(x: usize, m: &mut [u8], a: &BorderArgs) {
    let x = x as i32;
    let w = a.w;
    let h = a.h;
    if x >= w {
        return;
    }
    for q in 0..2 {
        let y = q;
        if !((y as f32) < a.fy0) {
            break;
        }
        if wb_outside(x, y, a.fx0, a.fx1, a.fy0, a.fy1) && y >= 0 && y < h && a.top >= 0 && a.top < h {
            m[(y * w + x) as usize] = m[(a.top * w + x) as usize];
        }
    }
    for q in 0..3 {
        let y = a.y1 + q;
        if y >= h {
            break;
        }
        if wb_outside(x, y, a.fx0, a.fx1, a.fy0, a.fy1) && y >= 0 && a.bottom >= 0 && a.bottom < h {
            m[(y * w + x) as usize] = m[(a.bottom * w + x) as usize];
        }
    }
}

/// `wb_border_cols` in place, row `y` only.
pub fn wb_border_cols_item(y: usize, m: &mut [u8], a: &BorderArgs) {
    let y = y as i32;
    let w = a.w;
    if y >= a.h {
        return;
    }
    for q in 0..2 {
        let x = q;
        if !((x as f32) < a.fx0) {
            break;
        }
        if x >= 0 && x < w && a.left >= 0 && a.left < w {
            m[(y * w + x) as usize] = m[(y * w + a.left) as usize];
        }
    }
    for j in 0..3 {
        if a.x1 + j >= w {
            break;
        }
        let d = a.right + j;
        if d >= 0 && d < w && a.right >= 0 && a.right < w {
            m[(y * w + d) as usize] = m[(y * w + a.right) as usize];
        }
    }
}

pub fn wb_invert_mask_item(i: usize, old: &[u8], m: &mut Global<u8>, n: u32) {
    if i >= n as usize {
        return;
    }
    m.set(i, (255 - old[i] as i32) as u8);
}

pub fn wb_store_int(v: f32, max: f32) -> i32 {
    let t = rs::cvtt_i32_f(v + 0.5f32);
    let mut f = t as f32;
    f = if f > 0.0f32 { f } else { 0.0f32 };
    f = if f < max { f } else { max };
    rs::cvtt_i32_f(f + 0.5f32)
}

pub fn wb_to_float_item<T: WbSample>(i: usize, src: &[T], dst: &mut Global<f32>, n: u32) {
    if i >= n as usize {
        return;
    }
    dst.set(i, src[i].f());
}

pub fn wb_round_u8_item(i: usize, src: &[f32], dst: &mut Global<f32>, n: u32) {
    if i >= n as usize {
        return;
    }
    dst.set(i, wb_store_int(src[i], 255.0f32) as u8 as f32);
}

pub fn wb_round_u16_item(i: usize, src: &[f32], dst: &mut Global<f32>, n: u32) {
    if i >= n as usize {
        return;
    }
    dst.set(i, wb_store_int(src[i], 65535.0f32) as u16 as f32);
}

pub fn wb_from_float_u8_item(i: usize, src: &[f32], dst: &mut Global<u8>, n: u32) {
    if i >= n as usize {
        return;
    }
    dst.set(i, wb_store_int(src[i], 255.0f32) as u8);
}

pub fn wb_from_float_u16_item(i: usize, src: &[f32], dst: &mut Global<u16>, n: u32) {
    if i >= n as usize {
        return;
    }
    dst.set(i, wb_store_int(src[i], 65535.0f32) as u16);
}

/// `wb_composite_u8` / `_u16`: `old` is `dest` before the launch.
pub fn wb_composite_int_item<T: WbSample>(i: usize, old: &[T], dest: &mut Global<T>, src: &[T], mask: &[u8], n: u32, store: fn(f32) -> T) {
    if i >= n as usize {
        return;
    }
    let m = mask[i / 4];
    if m == 0 {
        dest.set(i, src[i]);
        return;
    }
    let f = m as f32 / 255.0f32;
    let g = (1.0f64 - f as f64) as f32;
    dest.set(i, store((src[i].f() * g + old[i].f() * f) + 0.5f32));
}

pub fn wb_composite_f32_item(i: usize, old: &[f32], dest: &mut Global<f32>, src: &[f32], mask: &[u8], n: u32) {
    if i >= n as usize {
        return;
    }
    let m = mask[i / 4];
    if m == 0 {
        dest.set(i, src[i]);
        return;
    }
    let f = m as f32 / 255.0f32;
    let g = (1.0f64 - f as f64) as f32;
    dest.set(i, (g * src[i] + f * old[i]) + 0.0f32);
}
