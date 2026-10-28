//! Line for line Rust transcriptions of the kernels of `kernels::vecstage`: one function per
//! OpenCL C function, statement by statement, the same names without the source's types. A
//! kernel's transcription (`*_item`) is one work item; the simulator (`device::sim`) runs it over
//! the range in shuffled orders, writing through `Global` buffers, and the tests compare the
//! result with the CPU function bit for bit. An in place kernel reads the old samples from a copy
//! (`src`) and writes every element it owns once.

use crate::device::kernels::prelude::rs;
use crate::device::sim::Global;

// ---- VECTORS ----------------------------------------------------------------------------------

pub fn vc_clamp_unit(v: f32) -> f32 {
    if v > 1.0f32 {
        return 1.0f32;
    }
    if -1.0f32 > v {
        return -1.0f32;
    }
    v
}

pub fn vc_code_u8(a: f32) -> u8 {
    let c = vc_clamp_unit(a);
    let d = c + 1.0f32;
    rs::cvtt_i64_d(d as f64 * 127.0f64 + 0.5f64) as u8
}

pub fn vc_code_u16(a: f32) -> u16 {
    let c = vc_clamp_unit(a);
    let d = c + 1.0f32;
    rs::cvtt_i64_d(d as f64 * 32767.0f64 + 0.5f64) as u16
}

pub fn vc_code_f32(a: f32) -> f32 {
    (a + 1.0f32) * 0.5f32
}

#[allow(clippy::too_many_arguments)]
pub fn vc_bilerp(d: &[f32], w: u32, w1: i32, h1: i32, x: f32, y: f32) -> (f32, f32) {
    let mut ix = rs::cvtt_i32_f(x);
    let mut iy = rs::cvtt_i32_f(y);
    let mut fx = x - ix as f32;
    let mut fy = y - iy as f32;
    if ix >= w1 {
        ix = w1 - 1;
        fx = 1.0f32;
    }
    if iy >= h1 {
        iy = h1 - 1;
        fy = 1.0f32;
    }
    let i00 = (iy as usize * w as usize + ix as usize) * 2;
    let i01 = i00 + w as usize * 2;
    let mut out = [0.0f32; 2];
    for k in 0..2 {
        let p00 = d[i00 + k];
        let p10 = d[i00 + 2 + k];
        let p01 = d[i01 + k];
        let p11 = d[i01 + 2 + k];
        let top = (p10 - p00) * fx + p00;
        let bot = (p11 - p01) * fx + p01;
        out[k] = (bot - top) * fy + top;
    }
    (out[0], out[1])
}

/// The constants of `vc_encode_*` and of one layer of `vc_merge_*`.
#[derive(Clone, Copy, Debug)]
pub struct EncodeArgs {
    pub fw: u32,
    pub fw1: i32,
    pub fh1: i32,
    pub sx: f32,
    pub sy: f32,
    pub kx: f32,
    pub ky: f32,
}

#[allow(clippy::too_many_arguments)]
pub fn vc_sample(flow: &[f32], a: &EncodeArgs, x: usize, y: usize) -> (f32, f32) {
    let (u, v) = vc_bilerp(flow, a.fw, a.fw1, a.fh1, x as f32 * a.sx, y as f32 * a.sy);
    (u * a.kx, v * a.ky)
}

pub fn vc_encode_u8_item(x: usize, y: usize, flow: &[f32], e: &EncodeArgs, dst: &mut Global<u8>, dw: u32, dh: u32) {
    if x >= dw as usize || y >= dh as usize {
        return;
    }
    let (a, b) = vc_sample(flow, e, x, y);
    let o = (y * dw as usize + x) * 4;
    dst.set(o, 255);
    dst.set(o + 1, vc_code_u8(a));
    dst.set(o + 2, vc_code_u8(b));
    dst.set(o + 3, 0);
}

pub fn vc_encode_u16_item(x: usize, y: usize, flow: &[f32], e: &EncodeArgs, dst: &mut Global<u16>, dw: u32, dh: u32) {
    if x >= dw as usize || y >= dh as usize {
        return;
    }
    let (a, b) = vc_sample(flow, e, x, y);
    let o = (y * dw as usize + x) * 4;
    dst.set(o, 65535);
    dst.set(o + 1, vc_code_u16(a));
    dst.set(o + 2, vc_code_u16(b));
    dst.set(o + 3, 0);
}

pub fn vc_encode_f32_item(x: usize, y: usize, flow: &[f32], e: &EncodeArgs, dst: &mut Global<f32>, dw: u32, dh: u32) {
    if x >= dw as usize || y >= dh as usize {
        return;
    }
    let (a, b) = vc_sample(flow, e, x, y);
    let o = (y * dw as usize + x) * 4;
    dst.set(o, 1.0f32);
    dst.set(o + 1, vc_code_f32(a));
    dst.set(o + 2, vc_code_f32(b));
    dst.set(o + 3, 0.0f32);
}

/// The layers of `vc_merged_at`: flows `f`, weights `w`, the float constants `p` (sx, sy, kx, ky
/// per layer) and the int constants `ip` (flow width, w1, h1 per layer).
pub struct MergeArgs<'a> {
    pub f: [&'a [f32]; 4],
    pub w: [&'a [f32]; 4],
    pub p: &'a [f32],
    pub ip: &'a [i32],
    pub nl: u32,
}

pub fn vc_merged_at(m: &MergeArgs<'_>, x: usize, y: usize, width: u32) -> (f32, f32) {
    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    for k in 0..4usize {
        if k as i32 >= m.nl as i32 {
            break;
        }
        let f = m.f[k];
        let wt = m.w[k];
        let (u, v) = vc_bilerp(f, m.ip[k * 3] as u32, m.ip[k * 3 + 1], m.ip[k * 3 + 2], x as f32 * m.p[k * 4], y as f32 * m.p[k * 4 + 1]);
        let wk = wt[y * width as usize + x];
        sum_x = sum_x + wk * (u * m.p[k * 4 + 2]);
        sum_y = sum_y + wk * (v * m.p[k * 4 + 3]);
    }
    (sum_x, sum_y)
}

pub fn vc_merge_u8_item(x: usize, y: usize, m: &MergeArgs<'_>, dst: &mut Global<u8>, dw: u32, dh: u32) {
    if x >= dw as usize || y >= dh as usize {
        return;
    }
    let (a, b) = vc_merged_at(m, x, y, dw);
    let o = (y * dw as usize + x) * 4;
    dst.set(o, 255);
    dst.set(o + 1, vc_code_u8(a));
    dst.set(o + 2, vc_code_u8(b));
    dst.set(o + 3, 0);
}

pub fn vc_merge_u16_item(x: usize, y: usize, m: &MergeArgs<'_>, dst: &mut Global<u16>, dw: u32, dh: u32) {
    if x >= dw as usize || y >= dh as usize {
        return;
    }
    let (a, b) = vc_merged_at(m, x, y, dw);
    let o = (y * dw as usize + x) * 4;
    dst.set(o, 65535);
    dst.set(o + 1, vc_code_u16(a));
    dst.set(o + 2, vc_code_u16(b));
    dst.set(o + 3, 0);
}

pub fn vc_merge_f32_item(x: usize, y: usize, m: &MergeArgs<'_>, dst: &mut Global<f32>, dw: u32, dh: u32) {
    if x >= dw as usize || y >= dh as usize {
        return;
    }
    let (a, b) = vc_merged_at(m, x, y, dw);
    let o = (y * dw as usize + x) * 4;
    dst.set(o, 1.0f32);
    dst.set(o + 1, vc_code_f32(a));
    dst.set(o + 2, vc_code_f32(b));
    dst.set(o + 3, 0.0f32);
}

pub fn vc_merge_flow_item(x: usize, y: usize, m: &MergeArgs<'_>, dst: &mut Global<f32>, dw: u32, dh: u32) {
    if x >= dw as usize || y >= dh as usize {
        return;
    }
    let (a, b) = vc_merged_at(m, x, y, dw);
    let o = (y * dw as usize + x) * 2;
    dst.set(o, a);
    dst.set(o + 1, b);
}

pub fn vc_dec_int(c: f32, den: f64, scale: f64) -> f32 {
    ((((c as f64) * 2.0f64) / den - 1.0f64) * scale) as f32
}

pub fn vc_decode_u8_item(i: usize, vec: &[u8], flow: &mut Global<f32>, md: f64, n: u32) {
    if i >= n as usize {
        return;
    }
    let p = i * 4;
    if vec[p] == 0 {
        flow.set(2 * i, 0.0f32);
        flow.set(2 * i + 1, 0.0f32);
    } else {
        flow.set(2 * i, vc_dec_int(vec[p + 1] as f32, 254.0f64, md));
        flow.set(2 * i + 1, vc_dec_int(vec[p + 2] as f32, 254.0f64, md));
    }
}

pub fn vc_decode_u16_item(i: usize, vec: &[u16], flow: &mut Global<f32>, md: f64, n: u32) {
    if i >= n as usize {
        return;
    }
    let p = i * 4;
    if vec[p] == 0 {
        flow.set(2 * i, 0.0f32);
        flow.set(2 * i + 1, 0.0f32);
    } else {
        flow.set(2 * i, vc_dec_int(vec[p + 1] as f32, 65534.0f64, md));
        flow.set(2 * i + 1, vc_dec_int(vec[p + 2] as f32, 65534.0f64, md));
    }
}

pub fn vc_decode_f32_item(i: usize, vec: &[f32], flow: &mut Global<f32>, md: f32, n: u32) {
    if i >= n as usize {
        return;
    }
    let p = i * 4;
    if vec[p] == 0.0f32 {
        flow.set(2 * i, 0.0f32);
        flow.set(2 * i + 1, 0.0f32);
    } else {
        flow.set(2 * i, (vec[p + 1] * 2.0f32 - 1.0f32) * md);
        flow.set(2 * i + 1, (vec[p + 2] * 2.0f32 - 1.0f32) * md);
    }
}

pub fn vc_holes_u8_item(i: usize, img: &[u8], holes: &mut Global<u8>, n: u32) {
    if i >= n as usize {
        return;
    }
    holes.set(i, (255 - img[i * 4] as i32) as u8);
}

pub fn vc_holes_u16_item(i: usize, img: &[u16], holes: &mut Global<u16>, n: u32) {
    if i >= n as usize {
        return;
    }
    holes.set(i, (65535 - img[i * 4] as i32) as u16);
}

pub fn vc_holes_f32_item(i: usize, img: &[f32], holes: &mut Global<f32>, n: u32) {
    if i >= n as usize {
        return;
    }
    holes.set(i, 1.0f32 - img[i * 4]);
}

pub fn vc_vec_scale_item(i: usize, src: &[f32], flow: &mut Global<f32>, sx: f32, sy: f32, n: u32) {
    if i >= n as usize {
        return;
    }
    flow.set(2 * i, sx * src[2 * i]);
    flow.set(2 * i + 1, sy * src[2 * i + 1]);
}

pub fn vc_zero_item<T: Copy>(i: usize, img: &mut Global<T>, px: [T; 4], n: u32) {
    if i >= n as usize {
        return;
    }
    img.set(i * 4, px[0]);
    img.set(i * 4 + 1, px[1]);
    img.set(i * 4 + 2, px[2]);
    img.set(i * 4 + 3, px[3]);
}

// ---- MATTE ------------------------------------------------------------------------------------

pub fn mt_clamp01(x: f32) -> f32 {
    if x > 1.0f32 {
        return 1.0f32;
    }
    if x > 0.0f32 {
        return x;
    }
    0.0f32
}

pub fn mt_clamp_matte(x: f32) -> f32 {
    if x > 1.0f32 {
        return 1.0f32;
    }
    if x >= 0.0f32 {
        return x;
    }
    0.0f32
}

pub fn mt_store_plain_u8(x: f32) -> u8 {
    let y = x + 0.5f32;
    if 0.0f32 > y {
        return 0;
    }
    if y > 255.0f32 {
        return 255;
    }
    rs::trunc_u8_f(y)
}

pub fn mt_store_plain_u16(x: f32) -> u16 {
    let y = x + 0.5f32;
    if 0.0f32 > y {
        return 0;
    }
    if y > 65535.0f32 {
        return 65535;
    }
    rs::trunc_u16_f(y)
}

pub fn mt_over4(a: &[f32; 4], b: &[f32; 4], white: f32, out: &mut [f32; 4]) {
    if a[0] == 0.0f32 {
        for c in 0..4 {
            out[c] = b[c];
        }
        return;
    }
    if a[0] == white {
        for c in 0..4 {
            out[c] = a[c];
        }
        return;
    }
    let ia = 1.0f32 - a[0] / white;
    let alpha = (white - b[0]) * (a[0] / white) + b[0];
    out[0] = alpha;
    out[1] = b[1] * ia + a[1];
    out[2] = b[2] * ia + a[2];
    out[3] = b[3] * ia + a[3];
}

pub fn mt_extract_u8_item(i: usize, img: &[u8], m: &mut Global<f32>, c: u32, n: u32) {
    if i >= n as usize {
        return;
    }
    m.set(i, mt_clamp01(img[i * 4 + c as usize] as f32 / 255.0f32));
}

pub fn mt_extract_u16_item(i: usize, img: &[u16], m: &mut Global<f32>, c: u32, n: u32) {
    if i >= n as usize {
        return;
    }
    m.set(i, mt_clamp01(img[i * 4 + c as usize] as f32 / 65535.0f32));
}

pub fn mt_extract_f32_item(i: usize, img: &[f32], m: &mut Global<f32>, c: u32, n: u32) {
    if i >= n as usize {
        return;
    }
    m.set(i, mt_clamp01(img[i * 4 + c as usize] / 1.0f32));
}

pub fn mt_premultiply_u8_item(i: usize, src: &[u8], m: &[f32], out: &mut Global<u8>, n: u32) {
    if i >= n as usize {
        return;
    }
    let mv = mt_clamp_matte(m[i / 4]);
    out.set(i, rs::store_q_u8(src[i] as f32 * mv));
}

pub fn mt_premultiply_u16_item(i: usize, src: &[u16], m: &[f32], out: &mut Global<u16>, n: u32) {
    if i >= n as usize {
        return;
    }
    let mv = mt_clamp_matte(m[i / 4]);
    out.set(i, rs::store_q_u16(src[i] as f32 * mv));
}

pub fn mt_premultiply_f32_item(i: usize, src: &[f32], m: &[f32], out: &mut Global<f32>, n: u32) {
    if i >= n as usize {
        return;
    }
    let mv = mt_clamp_matte(m[i / 4]);
    out.set(i, src[i] * mv);
}

/// `mt_over_*`: `old` is the destination before the launch, `dest` the destination after.
pub fn mt_over_u8_item(i: usize, old: &[u8], dest: &mut Global<u8>, src: &[u8], n: u32) {
    if i >= n as usize {
        return;
    }
    let o = i * 4;
    let mut a = [0.0f32; 4];
    let mut b = [0.0f32; 4];
    let mut v = [0.0f32; 4];
    for c in 0..4 {
        a[c] = src[o + c] as f32;
        b[c] = old[o + c] as f32;
    }
    mt_over4(&a, &b, 255.0f32, &mut v);
    for c in 0..4 {
        dest.set(o + c, mt_store_plain_u8(v[c]));
    }
}

pub fn mt_over_u16_item(i: usize, old: &[u16], dest: &mut Global<u16>, src: &[u16], n: u32) {
    if i >= n as usize {
        return;
    }
    let o = i * 4;
    let mut a = [0.0f32; 4];
    let mut b = [0.0f32; 4];
    let mut v = [0.0f32; 4];
    for c in 0..4 {
        a[c] = src[o + c] as f32;
        b[c] = old[o + c] as f32;
    }
    mt_over4(&a, &b, 65535.0f32, &mut v);
    for c in 0..4 {
        dest.set(o + c, mt_store_plain_u16(v[c]));
    }
}

pub fn mt_over_f32_item(i: usize, old: &[f32], dest: &mut Global<f32>, src: &[f32], n: u32) {
    if i >= n as usize {
        return;
    }
    let o = i * 4;
    let mut a = [0.0f32; 4];
    let mut b = [0.0f32; 4];
    let mut v = [0.0f32; 4];
    for c in 0..4 {
        a[c] = src[o + c];
        b[c] = old[o + c];
    }
    mt_over4(&a, &b, 1.0f32, &mut v);
    for c in 0..4 {
        dest.set(o + c, v[c] + 0.0f32);
    }
}

/// `mt_effective`: `m` the FG mattes (unused ones may repeat), `e` the four outputs.
pub fn mt_effective_item(i: usize, m: [&[f32]; 3], e: &mut [Global<f32>; 4], nfg: u32, n: u32) {
    if i >= n as usize {
        return;
    }
    let mut vis = 1.0f32;
    for k in 0..3usize {
        if k as i32 >= nfg as i32 {
            break;
        }
        let mk = mt_clamp_matte(m[k][i]);
        e[k].set(i, mk * vis);
        let inv = 1.0f32 - mk;
        vis = mt_clamp01(inv * vis);
    }
    let last = if nfg == 0 { 0 } else if nfg == 1 { 1 } else if nfg == 2 { 2 } else { 3 };
    e[last].set(i, vis);
}

#[allow(clippy::too_many_arguments)]
pub fn mt_over_weights_item(i: usize, m: [&[f32]; 3], o: &mut [Global<f32>; 4], first: u32, last: u32, nfg: u32, n: u32) {
    if i >= n as usize {
        return;
    }
    let mut vis = 1.0f32;
    for k in 0..4i32 {
        if k < first as i32 {
            continue;
        }
        if k > last as i32 {
            break;
        }
        let j = (k - first as i32) as usize;
        if k == nfg as i32 {
            o[j].set(i, vis);
            break;
        }
        let mk = mt_clamp_matte(m[k as usize][i]);
        o[j].set(i, mk * vis);
        if k < last as i32 {
            let inv = 1.0f32 - mk;
            vis = mt_clamp01(inv * vis);
        }
    }
}

// ---- PREP -------------------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn pp_grad(img: &[f32], x: usize, y: usize, w: usize, h: usize, wrap: i32) -> (f32, f32) {
    let i = y * w + x;
    let c = img[i];
    let mut nx = 0.0f32;
    let mut dx = 0.0f32;
    let mut ny = 0.0f32;
    let mut dy = 0.0f32;
    if wrap != 0 {
        let left = img[y * w + (x + w - 1) % w];
        let right = img[y * w + (x + 1) % w];
        nx = ((c - left) + 0.0f32) + (right - c);
        dx = 1.0f32 + 1.0f32;
    } else if x >= 1 {
        nx = (c - img[i - 1]) + 0.0f32;
        dx = 1.0f32;
    }
    if y >= 1 {
        ny = (c - img[i - w]) + 0.0f32;
        dy = 1.0f32;
    }
    if wrap == 0 && x + 1 <= w - 1 {
        nx = nx + (img[i + 1] - c);
        dx = dx + 1.0f32;
    }
    if y + 1 <= h - 1 {
        ny = ny + (img[i + w] - c);
        dy = dy + 1.0f32;
    }
    (nx / dx, ny / dy)
}

pub fn pp_edge_value(gx: f32, gy: f32) -> f32 {
    let m = (gx * gx + gy * gy).sqrt();
    if m <= 20.0f32 {
        m * 0.05f32
    } else {
        1.0f32
    }
}

#[allow(clippy::too_many_arguments)]
pub fn pp_edge_tail_item(x: usize, y: usize, b: &[f32], out: &mut Global<f32>, w: u32, h: u32, wrap: i32) {
    if x >= w as usize || y >= h as usize {
        return;
    }
    let (gx, gy) = pp_grad(b, x, y, w as usize, h as usize, wrap);
    let f = pp_edge_value(gx, gy);
    let i = y * w as usize + x;
    out.set(i, (f * 255.0f32 + b[i]) * 0.5f32);
}

pub fn pp_luma_u8_item(i: usize, s: &[u8], out: &mut Global<f32>, alpha: i32, video: i32, n: u32) {
    if i >= n as usize {
        return;
    }
    let p = i * 4;
    let a = s[p] as f32;
    let r = s[p + 1] as f32;
    let g = s[p + 2] as f32;
    let b = s[p + 3] as f32;
    let v;
    if video != 0 {
        v = ((r - 16.0f32) * 255.0f32) / 219.0f32;
    } else {
        let sum = (r * 0.3f32 + g * 0.59f32) + b * 0.11f32;
        if alpha != 0 {
            v = (sum as f64 + a as f64 * 0.3f64) as f32;
        } else {
            v = sum;
        }
    }
    out.set(i, v);
}

pub fn pp_luma_u16_item(i: usize, s: &[u16], out: &mut Global<f32>, alpha: i32, video: i32, n: u32) {
    if i >= n as usize {
        return;
    }
    let p = i * 4;
    let a = s[p] as f64;
    let r = s[p + 1] as f64;
    let g = s[p + 2] as f64;
    let b = s[p + 3] as f64;
    let v;
    if video != 0 {
        v = (((s[p + 1] as f32) / 257.0f32 - 16.0f32) * 255.0f32) / 219.0f32;
    } else {
        let mut sum = (g * 0.59f64 + r * 0.3f64) + b * 0.11f64;
        if alpha != 0 {
            sum = sum + a * 0.3f64;
        }
        v = (sum / 257.0f64) as f32;
    }
    out.set(i, v);
}

pub fn pp_luma_f32_item(i: usize, s: &[f32], out: &mut Global<f32>, alpha: i32, video: i32, n: u32) {
    if i >= n as usize {
        return;
    }
    let p = i * 4;
    let a = s[p] as f64;
    let r = s[p + 1] as f64;
    let g = s[p + 2] as f64;
    let b = s[p + 3] as f64;
    let v;
    if video != 0 {
        v = ((s[p + 1] * 255.0f32 - 16.0f32) * 255.0f32) / 219.0f32;
    } else {
        let mut sum = (g * 0.59f64 + r * 0.3f64) + b * 0.11f64;
        if alpha != 0 {
            sum = sum + a * 0.3f64;
        }
        v = (sum * 255.0f64) as f32;
    }
    out.set(i, v);
}

// ---- OVERLAY ----------------------------------------------------------------------------------

pub fn ov_blend_u8_item(i: usize, a: &[u8], b: &[u8], out: &mut Global<u8>, n: u32) {
    if i >= n as usize {
        return;
    }
    let af = a[i] as f32;
    let bf = b[i] as f32;
    out.set(i, rs::trunc_u8_f((bf - af) * 0.5f32 + af + 0.5f32));
}

pub fn ov_blend_u16_item(i: usize, a: &[u16], b: &[u16], out: &mut Global<u16>, n: u32) {
    if i >= n as usize {
        return;
    }
    let af = a[i] as f32;
    let bf = b[i] as f32;
    out.set(i, rs::trunc_u16_f((bf - af) * 0.5f32 + af + 0.5f32));
}

pub fn ov_blend_f32_item(i: usize, a: &[f32], b: &[f32], out: &mut Global<f32>, n: u32) {
    if i >= n as usize {
        return;
    }
    out.set(i, (b[i] - a[i]) * 0.5f32 + a[i] + 0.0f32);
}

pub fn ov_diff_u8_item(i: usize, a: &[u8], b: &[u8], out: &mut Global<u8>, n: u32) {
    if i >= n as usize {
        return;
    }
    let o = i * 4;
    out.set(o, 255);
    for c in 1..4 {
        let x = a[o + c];
        let y = b[o + c];
        out.set(o + c, if x > y { x - y } else { y - x });
    }
}

pub fn ov_diff_u16_item(i: usize, a: &[u16], b: &[u16], out: &mut Global<u16>, n: u32) {
    if i >= n as usize {
        return;
    }
    let o = i * 4;
    out.set(o, 65535);
    for c in 1..4 {
        let x = a[o + c];
        let y = b[o + c];
        out.set(o + c, if x > y { x - y } else { y - x });
    }
}

pub fn ov_diff_f32_item(i: usize, a: &[f32], b: &[f32], out: &mut Global<f32>, n: u32) {
    if i >= n as usize {
        return;
    }
    let o = i * 4;
    out.set(o, 1.0f32);
    for c in 1..4 {
        out.set(o + c, (a[o + c] - b[o + c]).abs());
    }
}

pub fn ov_overlay_u8_item(i: usize, r: &[f32], g: &[f32], out: &mut Global<u8>, n: u32) {
    if i >= n as usize {
        return;
    }
    let o = i * 4;
    out.set(o, 255);
    out.set(o + 1, rs::store_q_u8(r[i] * 255.0f32));
    out.set(o + 2, rs::store_q_u8(g[i] * 255.0f32));
    out.set(o + 3, 0);
}

pub fn ov_overlay_u16_item(i: usize, r: &[f32], g: &[f32], out: &mut Global<u16>, n: u32) {
    if i >= n as usize {
        return;
    }
    let o = i * 4;
    out.set(o, 65535);
    out.set(o + 1, rs::store_q_u16(r[i] * 65535.0f32));
    out.set(o + 2, rs::store_q_u16(g[i] * 65535.0f32));
    out.set(o + 3, 0);
}

pub fn ov_overlay_f32_item(i: usize, r: &[f32], g: &[f32], out: &mut Global<f32>, n: u32) {
    if i >= n as usize {
        return;
    }
    let o = i * 4;
    out.set(o, 1.0f32);
    out.set(o + 1, r[i]);
    out.set(o + 2, g[i]);
    out.set(o + 3, 0.0f32);
}

pub fn ov_detail_item(i: usize, l: &[f32], b: &[f32], out: &mut Global<f32>, n: u32) {
    if i >= n as usize {
        return;
    }
    out.set(i, mt_clamp01(((l[i] - b[i]) / 255.0f32) + 0.5f32));
}

pub fn ov_edge_item(x: usize, y: usize, b: &[f32], out: &mut Global<f32>, w: u32, h: u32) {
    if x >= w as usize || y >= h as usize {
        return;
    }
    let (gx, gy) = pp_grad(b, x, y, w as usize, h as usize, 0);
    out.set(y * w as usize + x, pp_edge_value(gx, gy));
}

// ---- GEOM -------------------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn gm_hit(p: &[f32], ip: &[i32], n: u32, r2: f32, x: usize, y: usize) -> i32 {
    let mut hit = -1;
    for k in 0..16usize {
        if k as i32 >= n as i32 {
            break;
        }
        if ip[k * 8 + 7] == 0 {
            continue;
        }
        if (x as i32) < ip[k * 8] || x as i32 > ip[k * 8 + 1] || (y as i32) < ip[k * 8 + 2] || y as i32 > ip[k * 8 + 3] {
            continue;
        }
        let dy = y as f32 - p[k * 2 + 1];
        let dx = x as f32 - p[k * 2];
        if dx * dx + dy * dy <= r2 {
            hit = k as i32;
        }
    }
    hit
}

/// The geometry arguments of `gm_draw_*`.
pub struct DrawArgs<'a> {
    pub p: &'a [f32],
    pub ip: &'a [i32],
    pub n: u32,
    pub r2: f32,
    pub w: u32,
    pub h: u32,
}

pub fn gm_draw_u8_item(x: usize, y: usize, old: &[u8], img: &mut Global<u8>, a: &DrawArgs<'_>) {
    if x >= a.w as usize || y >= a.h as usize {
        return;
    }
    let o = (y * a.w as usize + x) * 4;
    let k = gm_hit(a.p, a.ip, a.n, a.r2, x, y);
    if k < 0 {
        for c in 0..4 {
            img.set(o + c, old[o + c]);
        }
        return;
    }
    let k = k as usize;
    img.set(o, 255);
    img.set(o + 1, a.ip[k * 8 + 4] as u8);
    img.set(o + 2, a.ip[k * 8 + 5] as u8);
    img.set(o + 3, a.ip[k * 8 + 6] as u8);
}

pub fn gm_draw_u16_item(x: usize, y: usize, old: &[u16], img: &mut Global<u16>, a: &DrawArgs<'_>) {
    if x >= a.w as usize || y >= a.h as usize {
        return;
    }
    let o = (y * a.w as usize + x) * 4;
    let k = gm_hit(a.p, a.ip, a.n, a.r2, x, y);
    if k < 0 {
        for c in 0..4 {
            img.set(o + c, old[o + c]);
        }
        return;
    }
    let k = k as usize;
    img.set(o, 65535);
    img.set(o + 1, (a.ip[k * 8 + 4] * 257) as u16);
    img.set(o + 2, (a.ip[k * 8 + 5] * 257) as u16);
    img.set(o + 3, (a.ip[k * 8 + 6] * 257) as u16);
}

pub fn gm_draw_f32_item(x: usize, y: usize, old: &[f32], img: &mut Global<f32>, a: &DrawArgs<'_>) {
    if x >= a.w as usize || y >= a.h as usize {
        return;
    }
    let o = (y * a.w as usize + x) * 4;
    let k = gm_hit(a.p, a.ip, a.n, a.r2, x, y);
    if k < 0 {
        for c in 0..4 {
            img.set(o + c, old[o + c]);
        }
        return;
    }
    let k = k as usize;
    img.set(o, 1.0f32);
    img.set(o + 1, a.ip[k * 8 + 4] as f32 / 255.0f32);
    img.set(o + 2, a.ip[k * 8 + 5] as f32 / 255.0f32);
    img.set(o + 3, a.ip[k * 8 + 6] as f32 / 255.0f32);
}

/// The scalar arguments of `gm_splat`.
#[derive(Clone, Copy, Debug)]
pub struct SplatArgs {
    pub n: u32,
    pub kx: f32,
    pub ky: f32,
    pub ow: f32,
    pub oh: f32,
    pub wl: u32,
    pub hl: u32,
}

pub fn gm_splat_item(x: usize, y: usize, p: &[f32], a: &SplatArgs, d: &mut Global<f32>, wt: &mut Global<f32>) {
    if x >= a.wl as usize || y >= a.hl as usize {
        return;
    }
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    let mut sw = 0.0f32;
    for k in 0..16usize {
        if k as i32 >= a.n as i32 {
            break;
        }
        let fx = p[k * 4];
        let fy = p[k * 4 + 1];
        let tx = p[k * 4 + 2];
        let ty = p[k * 4 + 3];
        let ax = fx + 0.5f32;
        let ay = fy + 0.5f32;
        let inside = ax >= 0.0f32 && ax < a.ow && ay >= 0.0f32 && ay < a.oh;
        if !inside || !(tx.is_finite() && ty.is_finite()) {
            continue;
        }
        let ix = rs::cvtt_i32_f(fx * a.kx + 0.5f32);
        let iy = rs::cvtt_i32_f(fy * a.ky + 0.5f32);
        if ix < 0 || iy < 0 || ix >= a.wl as i32 || iy >= a.hl as i32 {
            continue;
        }
        if ix as usize != x || iy as usize != y {
            continue;
        }
        sx = sx + (fx - tx) * a.kx;
        sy = sy + (fy - ty) * a.ky;
        sw = sw + 1.0f32;
    }
    let i = y * a.wl as usize + x;
    d.set(2 * i, sx);
    d.set(2 * i + 1, sy);
    wt.set(i, sw);
}

pub fn gm_seed_item(i: usize, d: &[f32], wt: &[f32], flow: &mut Global<f32>, holes: &mut Global<u8>, n: u32) {
    if i >= n as usize {
        return;
    }
    let w = wt[i];
    if w != 0.0f32 {
        flow.set(2 * i, d[2 * i] / w);
        flow.set(2 * i + 1, d[2 * i + 1] / w);
        holes.set(i, 0);
    } else {
        flow.set(2 * i, 0.0f32);
        flow.set(2 * i + 1, 0.0f32);
        holes.set(i, 255);
    }
}
