//! Line for line Rust transcriptions of the kernels of `kernels::warpstage`, one function per
//! OpenCL C function (`*_item` for a kernel's work item), statement by statement, generic over
//! the depth exactly where the OpenCL C source is a macro over the depth. The simulator
//! (`device::sim`) runs the items in shuffled orders and the tests compare the result with the
//! CPU functions bit for bit. Outputs are `sim::Global` buffers (one write per element); the flag
//! word is a plain integer that the items OR into, as `atomic_or` does on the device.

use crate::device::kernels::prelude::rs as p;
use crate::device::sim::Global;

/// The depth parameters of the macros: the load (`(float)x`) and the stores of each source.
pub trait Depth: Copy + Default + PartialEq + std::fmt::Debug + Send + Sync + 'static {
    const ZERO: Self;
    /// `(float)x`.
    fn ld(self) -> f32;
    /// `wp_store_##SUF`.
    fn wp_store(v: f32) -> Self;
    /// `wp_faded_##SUF`.
    fn wp_faded(v: f32, f: f32) -> Self;
    /// `bc_store_##SUF`.
    fn bc_store(v: f32) -> Self;
    /// `sb_plain_##SUF`.
    fn sb_plain(x: f32) -> Self;
    /// `sb_raw_##SUF`.
    fn sb_raw(x: f32) -> Self;
    /// The `WHITE` argument of `SB_DEPTH`.
    const WHITE: f32;
}

impl Depth for u8 {
    const ZERO: u8 = 0;
    const WHITE: f32 = 255.0;
    fn ld(self) -> f32 {
        self as f32
    }
    fn wp_store(v: f32) -> u8 {
        p::sat_u8(v)
    }
    fn wp_faded(v: f32, f: f32) -> u8 {
        p::store_q_u8(p::sat_u8(v) as f32 * f)
    }
    fn bc_store(v: f32) -> u8 {
        let mut t = p::cvtt_i32_f(v + 0.5f32) as f32;
        t = if t > 0.0f32 { t } else { 0.0f32 };
        t = if t < 255.0f32 { t } else { 255.0f32 };
        p::cvtt_i32_f(t) as u8
    }
    fn sb_plain(x: f32) -> u8 {
        let y = x + 0.5f32;
        if 0.0f32 > y {
            return 0;
        }
        if y > 255.0f32 {
            return 255;
        }
        p::trunc_u8_f(y)
    }
    fn sb_raw(x: f32) -> u8 {
        p::trunc_u8_f(x)
    }
}

impl Depth for u16 {
    const ZERO: u16 = 0;
    const WHITE: f32 = 65535.0;
    fn ld(self) -> f32 {
        self as f32
    }
    fn wp_store(v: f32) -> u16 {
        p::sat_u16(v)
    }
    fn wp_faded(v: f32, f: f32) -> u16 {
        p::store_q_u16(p::sat_u16(v) as f32 * f)
    }
    fn bc_store(v: f32) -> u16 {
        let mut t = p::cvtt_i32_f(v + 0.5f32) as f32;
        t = if t > 0.0f32 { t } else { 0.0f32 };
        t = if t < 65535.0f32 { t } else { 65535.0f32 };
        p::cvtt_i32_f(t) as u16
    }
    fn sb_plain(x: f32) -> u16 {
        let y = x + 0.5f32;
        if 0.0f32 > y {
            return 0;
        }
        if y > 65535.0f32 {
            return 65535;
        }
        p::trunc_u16_f(y)
    }
    fn sb_raw(x: f32) -> u16 {
        p::trunc_u16_f(x)
    }
}

impl Depth for f32 {
    const ZERO: f32 = 0.0;
    const WHITE: f32 = 1.0;
    fn ld(self) -> f32 {
        self
    }
    fn wp_store(v: f32) -> f32 {
        v
    }
    fn wp_faded(v: f32, f: f32) -> f32 {
        v * f
    }
    fn bc_store(v: f32) -> f32 {
        v
    }
    fn sb_plain(x: f32) -> f32 {
        x + 0.0f32
    }
    fn sb_raw(x: f32) -> f32 {
        x
    }
}

// ---- BLEND ------------------------------------------------------------------------------------

/// `bl_blend_u8` / `bl_blend_u16` (`BL_BLEND_INT`).
pub fn bl_blend_int_item<T: Depth>(i: usize, a: &[T], b: &[T], out: &mut Global<T>, w: f32, n: u32, store: fn(f32) -> T) {
    if i >= n as usize {
        return;
    }
    let af = a[i].ld();
    let bf = b[i].ld();
    out.set(i, store((bf - af) * w + af + 0.5f32));
}

/// `bl_blend_f32`.
pub fn bl_blend_f32_item(i: usize, a: &[f32], b: &[f32], out: &mut Global<f32>, w: f32, n: u32) {
    if i >= n as usize {
        return;
    }
    out.set(i, (b[i] - a[i]) * w + a[i] + 0.0f32);
}

/// `bl_lerp_u8` / `bl_lerp_u16` (`BL_LERP_INT`). `f` is the buffer before the launch and `out`
/// the same buffer after it; each item reads only its own pixel before writing it.
pub fn bl_lerp_int_item<T: Depth + Into<i32>>(pix: usize, f: &[T], s: &[T], out: &mut Global<T>, w: f32, n: u32, scale: f32, store: fn(f32) -> T) {
    if pix >= n as usize {
        return;
    }
    let o = pix * 4;
    let f0: i32 = f[o].into();
    let f1: i32 = f[o + 1].into();
    let f2: i32 = f[o + 2].into();
    let f3: i32 = f[o + 3].into();
    let s0: i32 = s[o].into();
    let s1: i32 = s[o + 1].into();
    let s2: i32 = s[o + 2].into();
    let s3: i32 = s[o + 3].into();
    let mut a = (f0 - s0) as f32 * w + s0 as f32;
    let nn = a / scale;
    if !(nn > 0.0f32) {
        let z = store(0.0f32);
        out.set(o, z);
        out.set(o + 1, z);
        out.set(o + 2, z);
        out.set(o + 3, z);
        return;
    }
    let mut r = (f1 - s1) as f32 * w + s1 as f32;
    let mut g = (f2 - s2) as f32 * w + s2 as f32;
    let mut b = (f3 - s3) as f32 * w + s3 as f32;
    if nn > 1.0f32 {
        r = r / nn;
        g = g / nn;
        b = b / nn;
        a = scale;
    }
    r = if r > a { a } else if 0.0f32 > r { 0.0f32 } else { r };
    g = if g > a { a } else if 0.0f32 > g { 0.0f32 } else { g };
    b = if b > a { a } else if 0.0f32 > b { 0.0f32 } else { b };
    out.set(o + 1, store(r + 0.5f32));
    out.set(o + 2, store(g + 0.5f32));
    out.set(o + 3, store(b + 0.5f32));
    out.set(o, store(a + 0.5f32));
}

/// `bl_lerp_f32`, in place as [`bl_lerp_int_item`].
pub fn bl_lerp_f32_item(pix: usize, f: &[f32], s: &[f32], out: &mut Global<f32>, w: f32, n: u32) {
    if pix >= n as usize {
        return;
    }
    let o = pix * 4;
    let f0 = f[o];
    let f1 = f[o + 1];
    let f2 = f[o + 2];
    let f3 = f[o + 3];
    let mut a = (f0 - s[o]) * w + s[o];
    if !(a > 0.0f32) {
        out.set(o, 0.0f32);
        out.set(o + 1, 0.0f32);
        out.set(o + 2, 0.0f32);
        out.set(o + 3, 0.0f32);
        return;
    }
    let mut r = (f1 - s[o + 1]) * w + s[o + 1];
    let mut g = (f2 - s[o + 2]) * w + s[o + 2];
    let mut b = (f3 - s[o + 3]) * w + s[o + 3];
    if a > 1.0f32 {
        r = r / a;
        g = g / a;
        b = b / a;
        a = 1.0f32;
    }
    out.set(o + 1, r + 0.0f32);
    out.set(o + 2, g + 0.0f32);
    out.set(o + 3, b + 0.0f32);
    out.set(o, a + 0.0f32);
}

/// `bl_plus_zero_f32`.
pub fn bl_plus_zero_f32_item(i: usize, src: &[f32], out: &mut Global<f32>, n: u32) {
    if i >= n as usize {
        return;
    }
    out.set(i, src[i] + 0.0f32);
}

/// `bl_unsound_f32`.
pub fn bl_unsound_f32_item(i: usize, src: &[f32], flag: &mut i32, n: u32) {
    if i >= n as usize {
        return;
    }
    if !(src[i].abs() <= f32::from_bits(0x7149_f2ca)) {
        *flag |= 1;
    }
}

// ---- WARP -------------------------------------------------------------------------------------

/// `wp_clip`: 1 settled, 0 the bound of 64 passes was hit.
pub fn wp_clip(x: f32, y: f32, ix: &mut f32, iy: &mut f32, w1: f32, h1: f32) -> i32 {
    for _s in 0..64 {
        if *ix < 0.0f32 {
            let dy = y - *iy;
            let dxs = x - *ix;
            let num = x - 0.0f32;
            *ix = 0.0f32;
            *iy = y - (num / dxs) * dy;
            continue;
        }
        if *ix > w1 {
            let dy = y - *iy;
            let dxs = x - *ix;
            let num = x - w1;
            *ix = w1;
            *iy = y - (num / dxs) * dy;
            continue;
        }
        if *iy < 0.0f32 {
            let dyy = y - *iy;
            let num = y - 0.0f32;
            let dx = x - *ix;
            *ix = x - (num / dyy) * dx;
            *iy = 0.0f32;
            continue;
        }
        if *iy > h1 {
            let dyy = y - *iy;
            let num = y - h1;
            let dx = x - *ix;
            *ix = x - (num / dyy) * dx;
            *iy = h1;
            continue;
        }
        return 1;
    }
    0
}

/// `wp_flow_bilerp`.
pub fn wp_flow_bilerp(d: &[f32], fw: i32, w1: i32, h1: i32, x: f32, y: f32, out: &mut [f32; 2]) {
    let mut ix = p::cvtt_i32_f(x);
    let mut iy = p::cvtt_i32_f(y);
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
    let i00 = (iy as usize * fw as usize + ix as usize) * 2;
    let i01 = i00 + fw as usize * 2;
    for k in 0..2 {
        let p00 = d[i00 + k];
        let p10 = d[i00 + 2 + k];
        let p01 = d[i01 + k];
        let p11 = d[i01 + 2 + k];
        let top = (p10 - p00) * fx + p00;
        let bot = (p11 - p01) * fx + p01;
        out[k] = (bot - top) * fy + top;
    }
}

/// `wp_mix4_##SUF`.
#[allow(clippy::too_many_arguments)]
pub fn wp_mix4<T: Depth>(src: &[T], ia: usize, ib: usize, ic: usize, id: usize, fx: f32, fy: f32, out: &mut [f32; 4]) {
    for k in 0..4 {
        let p00 = src[ia + k].ld();
        let p10 = src[ib + k].ld();
        let p01 = src[ic + k].ld();
        let p11 = src[id + k].ld();
        let top = (p10 - p00) * fx + p00;
        let bot = (p11 - p01) * fx + p01;
        out[k] = (bot - top) * fy + top;
    }
}

/// `wp_bilerp4_##SUF`.
pub fn wp_bilerp4<T: Depth>(src: &[T], w: i32, w1: i32, h1: i32, x: f32, y: f32, out: &mut [f32; 4]) {
    let mut ix = p::cvtt_i32_f(x);
    let mut iy = p::cvtt_i32_f(y);
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
    let i00 = (iy as usize * w as usize + ix as usize) * 4;
    let i01 = i00 + w as usize * 4;
    wp_mix4(src, i00, i00 + 4, i01, i01 + 4, fx, fy, out);
}

/// `wp_bilerp360_##SUF`.
pub fn wp_bilerp360<T: Depth>(src: &[T], w: i32, w1: i32, h1: i32, x: f32, y: f32, out: &mut [f32; 4]) {
    let ix = p::cvtt_i32_f(x);
    let iy = p::cvtt_i32_f(y);
    let fx = x - ix as f32;
    let fy = y - iy as f32;
    let x1 = if ix + 1 > w1 { 0 } else { ix + 1 };
    let mut xc;
    let mut xd;
    let yc;
    if iy + 1 > h1 {
        let hw = w / 2;
        xc = (hw + ix) % w;
        xd = (hw + x1) % w;
        yc = h1;
    } else {
        xc = ix;
        xd = x1;
        yc = iy + 1;
    }
    let ya = if iy < h1 { iy } else { h1 };
    xc = if xc > 0 { xc } else { 0 };
    xd = if xd > 0 { xd } else { 0 };
    let ia = (ya as usize * w as usize + ix as usize) * 4;
    let ib = (ya as usize * w as usize + x1 as usize) * 4;
    let ic = (yc as usize * w as usize + xc as usize) * 4;
    let id = (yc as usize * w as usize + xd as usize) * 4;
    wp_mix4(src, ia, ib, ic, id, fx, fy, out);
}

/// The scalar arguments of `wp_inv_warp_##SUF`, in kernel order after the buffers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WarpArgs {
    pub w: i32,
    pub h: i32,
    pub fw: i32,
    pub fh: i32,
    pub rx: f32,
    pub ry: f32,
    pub sx: f32,
    pub sy: f32,
    /// 0 hold, 1 fade, 2 wrap (360).
    pub kind: i32,
    pub want_cov: i32,
}

fn zero4<T: Depth>(out: &mut Global<T>, o: usize) {
    out.set(o, T::ZERO);
    out.set(o + 1, T::ZERO);
    out.set(o + 2, T::ZERO);
    out.set(o + 3, T::ZERO);
}

/// `wp_inv_warp_##SUF`, one work item at `(x, y)`.
#[allow(clippy::too_many_arguments)]
pub fn wp_inv_warp_item<T: Depth>(gx: usize, gy: usize, src: &[T], flow: &[f32], out: &mut Global<T>, cov: &mut Global<u8>, flag: &mut i32, a: &WarpArgs) {
    let x = gx as i32;
    let y = gy as i32;
    if x >= a.w || y >= a.h {
        return;
    }
    let (w, h) = (a.w, a.h);
    let wf = w as f32;
    let hf = h as f32;
    let w1i = w - 1;
    let h1i = h - 1;
    let w1 = w1i as f32;
    let h1 = h1i as f32;
    let yf = y as f32;
    let fyy = yf * a.ry;
    let xf = x as f32;
    let mut d = [0.0f32; 2];
    wp_flow_bilerp(flow, a.fw, a.fw - 1, a.fh - 1, xf * a.rx, fyy, &mut d);
    let mut ix = d[0] * a.sx + xf;
    let mut iy = d[1] * a.sy + yf;
    let o = (y as usize * w as usize + x as usize) * 4;
    let ci = y as usize * w as usize + x as usize;
    let mut s = [0.0f32; 4];
    let mut c: u8 = 255;
    if a.kind == 2 {
        let mut far = 0i32;
        let mut n = 0i32;
        for _t in 0..64 {
            if !(ix < 0.0f32) {
                break;
            }
            ix += wf;
        }
        far |= (ix < 0.0f32) as i32;
        for _t in 0..64 {
            if !(wf <= ix) {
                break;
            }
            ix -= wf;
        }
        far |= (wf <= ix) as i32;
        for _t in 0..64 {
            if !(iy < 0.0f32) {
                break;
            }
            iy += hf;
            n += 1;
        }
        far |= (iy < 0.0f32) as i32;
        for _t in 0..64 {
            if !(hf <= iy) {
                break;
            }
            iy -= hf;
            n += 1;
        }
        far |= (hf <= iy) as i32;
        if far != 0 {
            *flag |= 2;
            zero4(out, o);
        } else if !(ix >= 0.0f32 && ix < wf) || iy.is_nan() {
            *flag |= 1;
            zero4(out, o);
        } else {
            if (n & 1) != 0 {
                ix = wf * 0.5f32 + ix;
                if ix >= wf {
                    ix -= wf;
                }
                if iy != 0.0f32 {
                    iy = hf - iy;
                }
            }
            wp_bilerp360(src, w, w1i, h1i, ix, iy, &mut s);
            for k in 0..4 {
                out.set(o + k, T::wp_store(s[k]));
            }
        }
    } else if a.kind == 0 {
        let settled = wp_clip(xf, yf, &mut ix, &mut iy, w1, h1);
        if settled == 0 || ix.is_nan() || iy.is_nan() {
            *flag |= if settled != 0 { 1 } else { 2 };
            zero4(out, o);
        } else {
            wp_bilerp4(src, w, w1i, h1i, ix, iy, &mut s);
            for k in 0..4 {
                out.set(o + k, T::wp_store(s[k]));
            }
        }
    } else if !(ix > -1.0f32 && ix < wf && iy > -1.0f32 && iy < hf) {
        zero4(out, o);
        c = 0;
    } else if ix >= 0.0f32 && ix <= w1 && iy >= 0.0f32 && iy <= h1 {
        wp_bilerp4(src, w, w1i, h1i, ix, iy, &mut s);
        for k in 0..4 {
            out.set(o + k, T::wp_store(s[k]));
        }
        c = 255;
    } else {
        let mut f = if ix < 0.0f32 { ix + 1.0f32 } else if ix <= w1 { 1.0f32 } else { wf - ix };
        if iy < 0.0f32 {
            f = f * (iy + 1.0f32);
        } else if iy > h1 {
            f = f * (hf - iy);
        }
        let settled = wp_clip(xf, yf, &mut ix, &mut iy, w1, h1);
        if settled == 0 || ix.is_nan() || iy.is_nan() {
            *flag |= if settled != 0 { 1 } else { 2 };
            zero4(out, o);
            c = 0;
        } else {
            wp_bilerp4(src, w, w1i, h1i, ix, iy, &mut s);
            for k in 0..4 {
                out.set(o + k, T::wp_faded(s[k], f));
            }
            let cf = f * 255.0f32 + 0.5f32;
            c = if cf <= 255.0f32 { p::cvtt_i64_d(cf as f64) as u8 } else { 255 };
        }
    }
    if a.want_cov != 0 {
        cov.set(ci, c);
    }
}

// ---- BLURCOMP ---------------------------------------------------------------------------------

/// `bc_wrap_x`: 0 wrapped, 1 outside, 2 the device bound was hit.
pub fn bc_wrap_x(qx: f32, wf: f32, out: &mut f32) -> i32 {
    let mut x = qx;
    for _t in 0..64 {
        if !(x < 0.0f32) {
            break;
        }
        x += wf;
    }
    if x < 0.0f32 {
        return 2;
    }
    for _t in 0..64 {
        if !(wf <= x) {
            break;
        }
        x -= wf;
    }
    if wf <= x {
        return 2;
    }
    if x >= 0.0f32 && x < wf {
        *out = x;
        return 0;
    }
    1
}

/// The scalar arguments of `bc_motion_blur_##SUF`, in kernel order after the buffers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlurArgs {
    pub w: i32,
    pub h: i32,
    pub fw: i32,
    pub fh: i32,
    pub rx: f32,
    pub ry: f32,
    pub kx: f32,
    pub ky: f32,
    pub shifted: i32,
    pub k: f32,
    pub bilinear: i32,
    pub wrap: i32,
    pub y0: i32,
}

/// `bc_motion_blur_##SUF`, one work item (`gy` is the row within the band starting at `y0`).
#[allow(clippy::too_many_arguments)]
pub fn bc_motion_blur_item<T: Depth>(gx: usize, gy: usize, src: &[T], flow: &[f32], dest: &mut Global<T>, flag: &mut i32, a: &BlurArgs) {
    let x = gx as i32;
    let y = a.y0 + gy as i32;
    if x >= a.w || y >= a.h {
        return;
    }
    let (w, h) = (a.w, a.h);
    let wf = w as f32;
    let hf = h as f32;
    let w1i = w - 1;
    let h1i = h - 1;
    let yf = y as f32;
    let fyy = yf * a.ry;
    let xf = x as f32;
    let mut d = [0.0f32; 2];
    wp_flow_bilerp(flow, a.fw, a.fw - 1, a.fh - 1, xf * a.rx, fyy, &mut d);
    let vx = d[0] * a.kx;
    let vy = d[1] * a.ky;
    let d2 = vy * vy + vx * vx;
    let o = (y as usize * w as usize + x as usize) * 4;
    if a.bilinear != 0 && f32::from_bits(0x3586_37bd) > d2 {
        for c in 0..4 {
            dest.set(o + c, src[o + c]);
        }
        return;
    }
    let dist = d2.sqrt();
    let mut cx = xf;
    let mut cy = yf;
    if a.shifted != 0 {
        cx = vx * a.k + xf;
        cy = vy * a.k + yf;
    }
    if a.bilinear == 0 {
        cx = cx + 0.5f32;
        cy = cy + 0.5f32;
    }
    let mut n = p::cvtt_i32_f(dist + 1.0f32);
    let sx;
    let sy;
    if n < 2 {
        n = 2;
        sx = vx;
        sy = vy;
    } else {
        let dd = (n - 1) as f32;
        sx = vx / dd;
        sy = vy / dd;
    }
    if n > 65536 {
        *flag |= 1;
        zero4(dest, o);
        return;
    }
    if n > 1024 {
        *flag |= 2;
        zero4(dest, o);
        return;
    }
    let mut sum = [0.0f32; 4];
    let mut cnt = 0i32;
    let mut bad = 0i32;
    let mut t4 = [0.0f32; 4];
    let mut qx = cx;
    let mut qy = cy;
    let nsx = -sx;
    let nsy = -sy;
    for dir in 0..2i32 {
        for t in 0..1024i32 {
            if t >= n - dir {
                break;
            }
            if a.bilinear != 0 {
                if qx.is_nan() || qy.is_nan() {
                    bad |= 1;
                } else if a.wrap != 0 {
                    if 0.0f32 <= qy && qy < hf {
                        let mut wx = 0.0f32;
                        let r = bc_wrap_x(qx, wf, &mut wx);
                        if r != 0 {
                            bad |= r;
                        } else {
                            cnt += 1;
                            wp_bilerp360(src, w, w1i, h1i, wx, qy, &mut t4);
                            for c in 0..4 {
                                sum[c] += t4[c];
                            }
                        }
                    }
                } else if 0.0f32 <= qx && qx < wf && 0.0f32 <= qy && qy < hf {
                    cnt += 1;
                    wp_bilerp4(src, w, w1i, h1i, qx, qy, &mut t4);
                    for c in 0..4 {
                        sum[c] += t4[c];
                    }
                }
            } else {
                let iy = p::cvtt_i32_f(qy);
                let mut inside = 0i32;
                let mut ix = 0i32;
                if a.wrap != 0 {
                    if iy >= 0 && iy < h {
                        let mut wx = 0.0f32;
                        let r = bc_wrap_x(qx, wf, &mut wx);
                        if r != 0 {
                            bad |= r;
                        } else {
                            ix = p::cvtt_i32_f(wx);
                            inside = 1;
                        }
                    }
                } else {
                    ix = p::cvtt_i32_f(qx);
                    inside = (ix >= 0 && ix < w && iy >= 0 && iy < h) as i32;
                }
                if inside != 0 {
                    if dir == 0 {
                        cnt += 1;
                    }
                    let i = (iy as usize * w as usize + ix as usize) * 4;
                    for c in 0..4 {
                        sum[c] += src[i + c].ld();
                    }
                }
                if dir == 1 {
                    cnt += 1;
                }
            }
            if dir == 0 {
                qx += sx;
                qy += sy;
            } else {
                qx += nsx;
                qy += nsy;
            }
        }
        qx = cx - sx;
        qy = cy - sy;
    }
    if bad != 0 {
        *flag |= bad;
    }
    let inv = 1.0f32 / cnt as f32;
    for c in 0..4 {
        dest.set(o + c, T::bc_store(inv * sum[c]));
    }
}

/// `bc_clear_rgb_##SUF`, from `src` into `dst`.
pub fn bc_clear_rgb_item<T: Depth>(pix: usize, src: &[T], dst: &mut Global<T>, n: u32) {
    if pix >= n as usize {
        return;
    }
    let o = pix * 4;
    let a = src[o];
    dst.set(o, a);
    if a == T::ZERO {
        dst.set(o + 1, T::ZERO);
        dst.set(o + 2, T::ZERO);
        dst.set(o + 3, T::ZERO);
    } else {
        dst.set(o + 1, src[o + 1]);
        dst.set(o + 2, src[o + 2]);
        dst.set(o + 3, src[o + 3]);
    }
}

// ---- SMARTBLEND -------------------------------------------------------------------------------

/// `sb_over`.
pub fn sb_over(a: &[f32; 4], b: &[f32; 4], white: f32, o: &mut [f32; 4]) {
    if a[0] == 0.0f32 {
        for c in 0..4 {
            o[c] = b[c];
        }
        return;
    }
    if a[0] == white {
        for c in 0..4 {
            o[c] = a[c];
        }
        return;
    }
    let ia = 1.0f32 - a[0] / white;
    let alpha = (white - b[0]) * (a[0] / white) + b[0];
    o[0] = alpha;
    o[1] = b[1] * ia + a[1];
    o[2] = b[2] * ia + a[2];
    o[3] = b[3] * ia + a[3];
}

/// `sb_smart_pixel`.
pub fn sb_smart_pixel(p1: &[f32; 4], p2: &[f32; 4], w: f32, white: f32, v: &mut [f32; 4]) -> i32 {
    let mut c1 = [0.0f32; 4];
    let mut c2 = [0.0f32; 4];
    sb_over(p1, p2, white, &mut c1);
    sb_over(p2, p1, white, &mut c2);
    let amax = if p1[0] > p2[0] { p1[0] } else { p2[0] };
    for i in 0..4 {
        v[i] = (c2[i] - c1[i]) * w + c1[i];
    }
    if v[0] == amax {
        return 0;
    }
    if v[0] == 0.0f32 {
        if c1[0] > c2[0] {
            for i in 0..4 {
                v[i] = c1[i];
            }
        } else {
            for i in 0..4 {
                v[i] = c2[i];
            }
        }
        return 1;
    }
    let alpha = v[0];
    v[1] = (v[1] * amax) / alpha;
    v[2] = (v[2] * amax) / alpha;
    v[3] = (v[3] * amax) / alpha;
    v[0] = amax;
    0
}

/// `sb_binary_pixel`.
pub fn sb_binary_pixel(p1: &[f32; 4], p2: &[f32; 4], w: f32, white: f32, v: &mut [f32; 4]) -> i32 {
    let mut x = [0.0f32; 4];
    let mut y = [0.0f32; 4];
    if p2[0] == white {
        let ia = 1.0f32 - p1[0] / white;
        x[0] = p2[0];
        x[1] = p2[1] * ia + p1[1];
        x[2] = p2[2] * ia + p1[2];
        x[3] = p2[3] * ia + p1[3];
    } else if p1[0] == white {
        for i in 0..4 {
            x[i] = p1[i];
        }
    } else {
        for i in 0..4 {
            x[i] = 0.0f32;
        }
    }
    if p1[0] == white {
        let ia = 1.0f32 - p2[0] / white;
        y[0] = p1[0];
        y[1] = p1[1] * ia + p2[1];
        y[2] = p1[2] * ia + p2[2];
        y[3] = p1[3] * ia + p2[3];
    } else if p2[0] == white {
        for i in 0..4 {
            y[i] = p2[i];
        }
    } else {
        for i in 0..4 {
            y[i] = 0.0f32;
        }
    }
    for i in 0..4 {
        v[i] = (y[i] - x[i]) * w + x[i];
    }
    0
}

/// `sb_smart_##SUF`.
pub fn sb_smart_item<T: Depth>(pix: usize, p1: &[T], p2: &[T], out: &mut Global<T>, w: f32, binary: i32, n: u32) {
    if pix >= n as usize {
        return;
    }
    let o = pix * 4;
    let mut a = [0.0f32; 4];
    let mut b = [0.0f32; 4];
    let mut v = [0.0f32; 4];
    for c in 0..4 {
        a[c] = p1[o + c].ld();
        b[c] = p2[o + c].ld();
    }
    let raw = if binary != 0 { sb_binary_pixel(&a, &b, w, T::WHITE, &mut v) } else { sb_smart_pixel(&a, &b, w, T::WHITE, &mut v) };
    for c in 0..4 {
        out.set(o + c, if raw != 0 { T::sb_raw(v[c]) } else { T::sb_plain(v[c]) });
    }
}

/// `sb_cov_max_invert`, in place on `cov1`.
pub fn sb_cov_max_invert_item(i: usize, cov1: &[u8], cov2: &[u8], out: &mut Global<u8>, n: u32) {
    if i >= n as usize {
        return;
    }
    let a = cov1[i];
    let b = cov2[i];
    let m = if a > b { a } else { b };
    out.set(i, (255 - m as i32) as u8);
}

/// `sb_cov_invert`, in place.
pub fn sb_cov_invert_item(i: usize, cov: &[u8], out: &mut Global<u8>, n: u32) {
    if i >= n as usize {
        return;
    }
    out.set(i, (255 - cov[i] as i32) as u8);
}
