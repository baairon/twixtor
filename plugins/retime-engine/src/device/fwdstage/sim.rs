//! Line for line Rust transcriptions of the kernels of `kernels::fwdstage` (`*_item` per
//! kernel, the helpers under their OpenCL C names) and the device-free run of the whole stage:
//! the orchestration of [`super::forward_warp_enqueue`] with every launch in a shuffled work item
//! order (`device::sim`). The atomics of the counting and emitting passes run as plain read,
//! modify, write steps, one item at a time, which is one of the serialisations a device may give;
//! the slot order of the records therefore follows the shuffled item order, and the sort of the
//! fold must remove it.

use super::{level_dims, level_table, max_records, resample_scales, row_bands, triangle_bands, HostMesh, Plan};
use crate::device::convstage::sim::SimBackend;
use crate::device::convstage::{Backend, Launch, Taps};
use crate::device::image::StageError;
use crate::device::kernels::fwdstage as k;
use crate::device::kernels::prelude::rs as p;
use crate::device::kernels::wbstage as kwb;
use crate::device::sim::{self, Global};
use crate::device::warpstage::flag_result;
use crate::device::warpstage::rs::{wp_bilerp4, Depth};
use crate::forward::pyramid;
use crate::image::{Image, Samples, CHANNELS};
use crate::warp::Coverage;
use std::cell::RefCell;

/// The depth parameters of `FW_DEPTH` and the integer blur stores it is chained with.
pub trait FwDepth: Depth {
    const INT: bool;
    /// `fw_out_##SUF`.
    fn fw_out(a: f32) -> Self;
    /// `wb_round_##SUF` (integers only).
    fn round_item(i: usize, src: &[f32], dst: &mut Global<f32>, n: u32);
    /// `wb_from_float_##SUF` (integers only).
    fn from_float_item(i: usize, src: &[f32], dst: &mut Global<Self>, n: u32);
    /// `f32` to `Self` for the f32 path, where the blurred level is already float.
    fn from_f32_vec(v: Vec<f32>) -> Vec<Self>;
}

impl FwDepth for u8 {
    const INT: bool = true;
    fn fw_out(a: f32) -> u8 {
        p::trunc_u8_f(a + 0.5f32)
    }
    fn round_item(i: usize, src: &[f32], dst: &mut Global<f32>, n: u32) {
        kwb::wb_round_u8_item(i, src, dst, n)
    }
    fn from_float_item(i: usize, src: &[f32], dst: &mut Global<u8>, n: u32) {
        kwb::wb_from_float_u8_item(i, src, dst, n)
    }
    fn from_f32_vec(_: Vec<f32>) -> Vec<u8> {
        unreachable!("integer levels are stored by wb_from_float")
    }
}

impl FwDepth for u16 {
    const INT: bool = true;
    fn fw_out(a: f32) -> u16 {
        p::trunc_u16_f(a + 0.5f32)
    }
    fn round_item(i: usize, src: &[f32], dst: &mut Global<f32>, n: u32) {
        kwb::wb_round_u16_item(i, src, dst, n)
    }
    fn from_float_item(i: usize, src: &[f32], dst: &mut Global<u16>, n: u32) {
        kwb::wb_from_float_u16_item(i, src, dst, n)
    }
    fn from_f32_vec(_: Vec<f32>) -> Vec<u16> {
        unreachable!("integer levels are stored by wb_from_float")
    }
}

impl FwDepth for f32 {
    const INT: bool = false;
    fn fw_out(a: f32) -> f32 {
        a + 0.0f32
    }
    fn round_item(_: usize, _: &[f32], _: &mut Global<f32>, _: u32) {
        unreachable!("no integer store at 32 bpc")
    }
    fn from_float_item(_: usize, _: &[f32], _: &mut Global<f32>, _: u32) {
        unreachable!("no integer store at 32 bpc")
    }
    fn from_f32_vec(v: Vec<f32>) -> Vec<f32> {
        v
    }
}

// ---- the scan converter -------------------------------------------------------------------------

/// `fw_vtx`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Vtx {
    pub x: f32,
    pub y: f32,
    pub u: f32,
    pub v: f32,
    pub d: f32,
}

/// `fw_edge`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Edge {
    ay: f32,
    by: f32,
    x: f32,
    u: f32,
    v: f32,
    d: f32,
    dx: f32,
    du: f32,
    dv: f32,
    dd: f32,
}

/// The sink arguments of `fw_hit` (`cnt`, `fill`, `off`, `cpre`, `sbase`, `rec`, `total`, `flag`)
/// and the item's hit counter `nh`.
pub struct Sink<'a> {
    pub mode: i32,
    pub w: i32,
    pub nh: i32,
    pub cnt: &'a mut [u32],
    pub fill: &'a mut [u32],
    pub off: &'a [u32],
    pub cpre: &'a [u32],
    pub sbase: &'a [u32],
    pub rec: Option<&'a mut Global<[u32; 4]>>,
    pub total: u32,
    pub flag: &'a mut i32,
}

/// `fw_vertex`.
pub fn fw_vertex(pos: &[f32], dl: &[f32], has_d: bool, w: i32, h: i32, r: i32, c: i32) -> Vtx {
    let i = r as usize * w as usize + c as usize;
    Vtx {
        x: pos[i * 2],
        y: pos[i * 2 + 1],
        u: c as f32 / (w - 1) as f32,
        v: r as f32 / (h - 1) as f32,
        d: if has_d { dl[i] } else { 0.0f32 },
    }
}

/// `fw_hit`.
pub fn fw_hit(s: &mut Sink, x: i32, y: i32, u: f32, v: f32, d: f32, key: u32) -> bool {
    s.nh += 1;
    if s.nh > 16384 {
        *s.flag |= 2;
        return false;
    }
    let p = (y as u32).wrapping_mul(s.w as u32).wrapping_add(x as u32) as usize;
    if s.mode == 0 {
        s.cnt[p] = s.cnt[p].wrapping_add(1);
        return true;
    }
    let slot = s.fill[p];
    s.fill[p] = slot.wrapping_add(1);
    let idx = s.off[p].wrapping_add(s.cpre[p >> 6]).wrapping_add(s.sbase[p >> 12]).wrapping_add(slot);
    if slot >= s.cnt[p] || idx >= s.total {
        *s.flag |= 2;
        return false;
    }
    s.rec.as_mut().expect("records in mode 1").set(idx as usize, [key, u.to_bits(), v.to_bits(), d.to_bits()]);
    true
}

/// `fw_edge_init`.
fn fw_edge_init(a: &Vtx, b: &Vtx, y: f32, e0: &mut Edge, e1: &mut Edge, count: &mut u32) {
    if a.y <= y && y < b.y {
        let slot = if *count != 0 { 1 } else { 0 };
        *count += 1;
        let dx = b.x - a.x;
        let du = b.u - a.u;
        let dv = b.v - a.v;
        let dd = b.d - a.d;
        let inv = 1.0f32 / (b.y - a.y);
        let t = ((y - 1.0f32) - a.y) * inv;
        let e = Edge {
            ay: a.y,
            by: b.y,
            x: dx * t + a.x,
            u: du * t + a.u,
            v: dv * t + a.v,
            d: dd * t + a.d,
            dx: dx * inv,
            du: du * inv,
            dv: dv * inv,
            dd: dd * inv,
        };
        if slot == 0 {
            *e0 = e;
        } else {
            *e1 = e;
        }
    }
}

/// `fw_span`.
fn fw_span(s: &mut Sink, l: Edge, r: Edge, y: f32, w1: i32, key: u32) -> bool {
    let mut x0 = p::cvtt_i32_f(l.x.ceil());
    x0 = if x0 > 0 { x0 } else { 0 };
    let mut x1 = p::cvtt_i32_f(r.x.floor());
    x1 = if x1 < w1 { x1 } else { w1 };
    let wid = r.x - l.x;
    if x0 > x1 {
        return true;
    }
    if !((wid as f64) > 1e-08f64) {
        return true;
    }
    let yi = p::cvtt_i32_f(y);
    if x0 == x1 {
        let t = ((x0 as f32) - l.x) * (1.0f32 / wid);
        let u = (r.u - l.u) * t + l.u;
        let v = (r.v - l.v) * t + l.v;
        let d = (r.d - l.d) * t + l.d;
        return fw_hit(s, x0, yi, u, v, d, key);
    }
    let inv = 1.0f32 / wid;
    let du = r.u - l.u;
    let dv = r.v - l.v;
    let dd = r.d - l.d;
    let t = ((x0 as f32) - l.x) * inv;
    let mut u = du * t + l.u;
    let mut v = dv * t + l.v;
    let mut d = dd * t + l.d;
    let mut x = x0;
    for _ in 0..16384 {
        if !(x < x1) {
            break;
        }
        if !fw_hit(s, x, yi, u, v, d, key) {
            return false;
        }
        u += du * inv;
        v += dv * inv;
        d += dd * inv;
        x += 1;
    }
    fw_hit(s, x, yi, u, v, d, key)
}

/// `fw_emit_hit`.
#[allow(clippy::too_many_arguments)]
fn fw_emit_hit(s: &mut Sink, x: i32, yi: i32, w1: i32, h1: i32, u: f32, v: f32, d: f32, key: u32) -> bool {
    if x < 0 || x > w1 || yi < 0 || yi > h1 {
        *s.flag |= 1;
        return true;
    }
    fw_hit(s, x, yi, u, v, d, key)
}

/// `fw_triangle`.
pub fn fw_triangle(s: &mut Sink, a0: Vtx, a1: Vtx, a2: Vtx, w1: i32, h1: i32, key: u32) -> bool {
    let w1f = w1 as f32;
    let h1f = h1 as f32;
    let ok = (0.0f32 <= a0.x || 0.0f32 <= a1.x || 0.0f32 <= a2.x)
        && (0.0f32 <= a0.y || 0.0f32 <= a1.y || 0.0f32 <= a2.y)
        && (a0.x <= w1f || a1.x <= w1f || a2.x <= w1f)
        && (a0.y <= h1f || a1.y <= h1f || a2.y <= h1f);
    if !ok {
        return true;
    }
    let (v0, v1, v2);
    if a1.y < a0.y {
        if a2.y < a0.y {
            if a2.y < a1.y {
                (v0, v1, v2) = (a2, a1, a0);
            } else {
                (v0, v1, v2) = (a1, a2, a0);
            }
        } else {
            (v0, v1, v2) = (a1, a0, a2);
        }
    } else if a2.y < a1.y {
        if a0.y < a2.y {
            (v0, v1, v2) = (a0, a2, a1);
        } else {
            (v0, v1, v2) = (a2, a0, a1);
        }
    } else {
        (v0, v1, v2) = (a0, a1, a2);
    }
    let mut y = v0.y.ceil();
    let mut y_end = v2.y.floor();
    if y < 0.0f32 {
        y = 0.0f32;
    }
    if h1f < y_end {
        y_end = h1f;
    }
    if y <= y_end {
        let mut vi = 0i32;
        let mut count = 0u32;
        let mut e0 = Edge::default();
        let mut e1 = e0;
        for _ in 0..16384 {
            if vi < 3 {
                for _ in 0..3 {
                    let cur = if vi == 0 { v0 } else if vi == 1 { v1 } else { v2 };
                    if y < cur.y {
                        break;
                    }
                    if count != 0 {
                        if count == 1 {
                            if e0.ay <= y && e0.by <= y {
                                count = 0;
                            }
                        } else if count == 2 {
                            if y < e1.ay || y < e1.by {
                                if !(y < e0.ay || y < e0.by) {
                                    e0 = e1;
                                    count = 1;
                                }
                            } else if e0.ay <= y && e0.by <= y {
                                count = 0;
                            } else {
                                count = 1;
                            }
                        }
                    }
                    if vi == 0 {
                        fw_edge_init(&v0, &v1, y, &mut e0, &mut e1, &mut count);
                        fw_edge_init(&v0, &v2, y, &mut e0, &mut e1, &mut count);
                    } else if vi == 1 {
                        fw_edge_init(&v1, &v2, y, &mut e0, &mut e1, &mut count);
                        fw_edge_init(&v1, &v0, y, &mut e0, &mut e1, &mut count);
                    } else {
                        fw_edge_init(&v2, &v0, y, &mut e0, &mut e1, &mut count);
                        fw_edge_init(&v2, &v1, y, &mut e0, &mut e1, &mut count);
                    }
                    vi += 1;
                    if vi >= 3 {
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
            if count == 2 {
                if e1.x < e0.x {
                    if !fw_span(s, e1, e0, y, w1, key) {
                        return false;
                    }
                } else if !fw_span(s, e0, e1, y, w1, key) {
                    return false;
                }
            }
            y += 1.0f32;
            if !(y <= y_end) {
                break;
            }
        }
    }
    if v1.y == h1f && v1.y == v2.y {
        let (l, r) = if v2.x <= v1.x { (v2, v1) } else { (v1, v2) };
        let yy = l.y;
        let yt = p::cvtt_i32_f(yy) as f32;
        if 0.0f32 <= yt && yt <= h1f && !(((yy - yt) as f64) > 0.01f64) {
            let mut x0 = p::cvtt_i32_f(l.x.ceil());
            x0 = if x0 > 0 { x0 } else { 0 };
            let mut x1 = p::cvtt_i32_f(r.x.floor());
            x1 = if x1 < w1 { x1 } else { w1 };
            let wid = r.x - l.x;
            if (wid as f64) > 1e-05f64 {
                let inv = 1.0f32 / wid;
                let yi = p::cvtt_i32_f(yt);
                if x0 == x1 {
                    let tt = ((x0 as f32) - l.x) * inv;
                    let u = (r.u - l.u) * tt + l.u;
                    let v = (r.v - l.v) * tt + l.v;
                    let d = (r.d - l.d) * tt + l.d;
                    return fw_emit_hit(s, x0, yi, w1, h1, u, v, d, key);
                }
                let du = r.u - l.u;
                let dv = r.v - l.v;
                let dd = r.d - l.d;
                let tt = ((x0 as f32) - l.x) * inv;
                let mut u = du * tt + l.u;
                let mut v = dv * tt + l.v;
                let mut d = dd * tt + l.d;
                let mut x = x0;
                for _ in 0..16384 {
                    if !(x < x1) {
                        break;
                    }
                    if !fw_emit_hit(s, x, yi, w1, h1, u, v, d, key) {
                        return false;
                    }
                    u += du * inv;
                    v += dv * inv;
                    d += dd * inv;
                    x += 1;
                }
                return fw_emit_hit(s, x, yi, w1, h1, u, v, d, key);
            }
        }
    }
    true
}

/// `fw_mesh_triangle`.
pub fn fw_mesh_triangle(s: &mut Sink, key: u32, pos: &[f32], dl: &[f32], has_d: bool, w: i32, h: i32) -> bool {
    let cell = key >> 1;
    let r = (cell / (w - 1) as u32) as i32;
    let c = (cell % (w - 1) as u32) as i32;
    let v01 = fw_vertex(pos, dl, has_d, w, h, r, c + 1);
    let v10 = fw_vertex(pos, dl, has_d, w, h, r + 1, c);
    if key & 1 == 0 {
        let v00 = fw_vertex(pos, dl, has_d, w, h, r, c);
        return fw_triangle(s, v00, v01, v10, w - 1, h - 1, key);
    }
    let v11 = fw_vertex(pos, dl, has_d, w, h, r + 1, c + 1);
    fw_triangle(s, v11, v01, v10, w - 1, h - 1, key)
}

// ---- the passes ---------------------------------------------------------------------------------

/// The mesh arguments of the triangle passes.
pub struct MeshArgs<'a> {
    pub pos: &'a [f32],
    pub dl: &'a [f32],
    pub has_d: bool,
    pub w: i32,
    pub h: i32,
}

/// `fw_count`.
#[allow(clippy::too_many_arguments)]
pub fn fw_count_item(g: usize, m: &MeshArgs, cnt: &mut [u32], total: &mut [u32], maxrec: u32, flag: &mut i32, key0: u32, n: u32) {
    let g = g as u32;
    if g >= n {
        return;
    }
    if total[0] > maxrec {
        return;
    }
    let mut s = Sink { mode: 0, w: m.w, nh: 0, cnt, fill: &mut [], off: &[], cpre: &[], sbase: &[], rec: None, total: 0, flag };
    fw_mesh_triangle(&mut s, key0.wrapping_add(g), m.pos, m.dl, m.has_d, m.w, m.h);
    let nh = s.nh;
    if nh > 0 {
        let old = total[0];
        total[0] = old.wrapping_add(nh as u32);
        if old.wrapping_add(nh as u32) > maxrec || old.wrapping_add(nh as u32) < old {
            *flag |= 2;
        }
    }
}

/// `fw_scan_chunks`.
pub fn fw_scan_chunks_item(k: usize, cnt: &[u32], off: &mut Global<u32>, csum: &mut Global<u32>, n: u32, nc: u32) {
    let k = k as u32;
    if k >= nc {
        return;
    }
    let mut s = 0u32;
    for j in 0..64u32 {
        let i = k * 64 + j;
        if i >= n {
            break;
        }
        off.set(i as usize, s);
        s = s.wrapping_add(cnt[i as usize]);
    }
    csum.set(k as usize, s);
}

/// `fw_scan_supers`.
pub fn fw_scan_supers_item(k: usize, csum: &[u32], cpre: &mut Global<u32>, ssum: &mut Global<u32>, nc: u32, ns: u32) {
    let k = k as u32;
    if k >= ns {
        return;
    }
    let mut s = 0u32;
    for j in 0..64u32 {
        let i = k * 64 + j;
        if i >= nc {
            break;
        }
        cpre.set(i as usize, s);
        s = s.wrapping_add(csum[i as usize]);
    }
    ssum.set(k as usize, s);
}

/// `fw_scan_top`.
pub fn fw_scan_top_item(g: usize, ssum: &[u32], sbase: &mut Global<u32>, total: &mut [u32], ns: u32) {
    if g != 0 {
        return;
    }
    let mut s = 0u64;
    for j in 0..16384u32 {
        if j >= ns {
            break;
        }
        sbase.set(j as usize, s as u32);
        s += ssum[j as usize] as u64;
    }
    total[1] = if s > 0xffff_ffff { 0xffff_ffff } else { s as u32 };
}

/// `fw_emit`.
#[allow(clippy::too_many_arguments)]
pub fn fw_emit_item(
    g: usize,
    m: &MeshArgs,
    cnt: &mut [u32],
    fill: &mut [u32],
    off: &[u32],
    cpre: &[u32],
    sbase: &[u32],
    rec: &mut Global<[u32; 4]>,
    total: u32,
    flag: &mut i32,
    key0: u32,
    n: u32,
) {
    let g = g as u32;
    if g >= n {
        return;
    }
    let mut s = Sink { mode: 1, w: m.w, nh: 0, cnt, fill, off, cpre, sbase, rec: Some(rec), total, flag };
    fw_mesh_triangle(&mut s, key0.wrapping_add(g), m.pos, m.dl, m.has_d, m.w, m.h);
}

/// `fw_fold`.
pub fn fw_fold(acc: &mut [f32; 4], n: &mut i32, c: &[f32; 4]) {
    if *n == 0 {
        for k in 0..4 {
            acc[k] = c[k] + 0.0f32;
        }
        *n = 1;
    } else if *n == 255 {
        for k in 0..4 {
            acc[k] = (acc[k] * 255.0f32 + c[k]) * 0.00390625f32 + 0.0f32;
        }
    } else {
        let nf = *n as f32;
        let r = (1.0f64 / ((nf as f64) + 1.0f64)) as f32;
        for k in 0..4 {
            acc[k] = (nf * acc[k] + c[k]) * r + 0.0f32;
        }
        *n += 1;
    }
}

/// `fw_tap_##SUF`.
pub fn fw_tap<T: FwDepth>(img: &[T], w: i32, h: i32, u: f32, v: f32, c: &mut [f32; 4]) -> i32 {
    let w1 = w - 1;
    let h1 = h - 1;
    let x = w1 as f32 * u;
    let y = h1 as f32 * v;
    if p::cvtt_i32_f(x) < 0 || p::cvtt_i32_f(y) < 0 {
        *c = [0.0f32; 4];
        return 0;
    }
    wp_bilerp4(img, w, w1, h1, x, y, c);
    1
}

/// `fw_level_tap_##SUF`.
pub fn fw_level_tap<T: FwDepth>(src: &[T], lev: &[T], lt: &[i32], i: i32, u: f32, v: f32, c: &mut [f32; 4]) -> i32 {
    let i = i as usize;
    if i == 0 {
        return fw_tap(src, lt[1], lt[2], u, v, c);
    }
    fw_tap(&lev[lt[i * 3] as usize..], lt[i * 3 + 1], lt[i * 3 + 2], u, v, c)
}

/// `fw_sample_##SUF`.
#[allow(clippy::too_many_arguments)]
pub fn fw_sample<T: FwDepth>(src: &[T], lev: &[T], lt: &[i32], nlev: i32, u: f32, v: f32, d: f32, c: &mut [f32; 4]) -> i32 {
    if 0.0f32 < d {
        if d < (nlev - 1) as f32 {
            let i = p::cvtt_i32_f(d);
            let frac = d - i as f32;
            let mut a = [0.0f32; 4];
            let mut b = [0.0f32; 4];
            let mut ok = fw_level_tap(src, lev, lt, i, u, v, &mut a);
            ok &= fw_level_tap(src, lev, lt, i + 1, u, v, &mut b);
            for k in 0..4 {
                c[k] = (b[k] - a[k]) * frac + a[k];
            }
            return ok;
        }
        return fw_level_tap(src, lev, lt, nlev - 1, u, v, c);
    }
    fw_level_tap(src, lev, lt, 0, u, v, c)
}

/// The arguments of `fw_fold_##SUF` other than the item and its outputs.
pub struct FoldArgs<'a, T> {
    pub src: &'a [T],
    pub lev: &'a [T],
    pub lt: &'a [i32],
    pub nlev: i32,
    pub cnt: &'a [u32],
    pub off: &'a [u32],
    pub cpre: &'a [u32],
    pub sbase: &'a [u32],
    pub want_cov: bool,
    pub w: i32,
    pub h: i32,
    pub y0: i32,
    pub rows: i32,
}

/// `fw_fold_##SUF`: the records of a pixel are its own, so the in place sort writes `rec` freely.
pub fn fw_fold_item<T: FwDepth>(x: usize, gy: usize, a: &FoldArgs<T>, rec: &mut [[u32; 4]], out: &mut Global<T>, cov: &mut Global<u8>, flag: &mut i32) {
    let x = x as i32;
    let gy = gy as i32;
    let y = a.y0 + gy;
    if x >= a.w || gy >= a.rows || y >= a.h {
        return;
    }
    let pp = (y as u32).wrapping_mul(a.w as u32).wrapping_add(x as u32);
    let p = pp as usize;
    let mut n = a.cnt[p];
    if n > 512 {
        *flag |= 2;
        n = 0;
    }
    let base = a.off[p].wrapping_add(a.cpre[p >> 6]).wrapping_add(a.sbase[p >> 12]) as usize;
    for i in 1..512u32 {
        if i >= n {
            break;
        }
        let cur = rec[base + i as usize];
        let mut j = i as i32;
        for _ in 0..512 {
            if j <= 0 {
                break;
            }
            let prev = rec[base + (j - 1) as usize];
            if prev[0] <= cur[0] {
                break;
            }
            rec[base + j as usize] = prev;
            j -= 1;
        }
        rec[base + j as usize] = cur;
    }
    let mut acc = [0.0f32; 4];
    let mut hits = 0i32;
    let mut last = 0u32;
    for i in 0..512u32 {
        if i >= n {
            break;
        }
        let r = rec[base + i as usize];
        if i > 0 && r[0] == last {
            *flag |= 2;
        }
        last = r[0];
        let mut c = [0.0f32; 4];
        if fw_sample(a.src, a.lev, a.lt, a.nlev, f32::from_bits(r[1]), f32::from_bits(r[2]), f32::from_bits(r[3]), &mut c) == 0 {
            *flag |= 2;
        }
        fw_fold(&mut acc, &mut hits, &c);
    }
    let o = p * 4;
    for k in 0..4 {
        out.set(o + k, T::fw_out(acc[k]));
    }
    if a.want_cov {
        cov.set(p, if hits != 0 { 255 } else { 0 });
    }
}

/// `fw_load_##SUF`.
pub fn fw_load_item<T: FwDepth>(i: usize, src: &[T], off: u32, dst: &mut Global<f32>, n: u32) {
    if i as u32 >= n {
        return;
    }
    dst.set(i, src[off as usize + i].ld());
}

/// `fw_resample_##SUF`.
#[allow(clippy::too_many_arguments)]
pub fn fw_resample_item<T: FwDepth>(x: usize, y: usize, src: &[T], sw: i32, sh: i32, lev: &mut Global<T>, off: u32, dw: i32, dh: i32, sx: f32, sy: f32) {
    let (x, y) = (x as i32, y as i32);
    if x >= dw || y >= dh {
        return;
    }
    let fy = y as f32 * sy;
    let mut v = [0.0f32; 4];
    wp_bilerp4(src, sw, sw - 1, sh - 1, x as f32 * sx, fy, &mut v);
    let o = off as usize + (y as usize * dw as usize + x as usize) * 4;
    for k in 0..4 {
        lev.set(o + k, T::wp_store(v[k]));
    }
}

// ---- the stage ----------------------------------------------------------------------------------

/// A fresh order seed per launch (identity for seed 0).
struct Seeds(u64, u64);

impl Seeds {
    fn next(&mut self) -> u64 {
        self.1 += 1;
        if self.0 == 0 {
            0
        } else {
            self.0 ^ self.1.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        }
    }
}

fn written<T: Copy>(g: Global<T>, what: &str) -> Vec<T> {
    assert_eq!(g.written(), g.len(), "{what}: not every element was written");
    g.into_vec()
}

/// [`super::pyramid_enqueue`] in the simulator: the levels 1.. in one vector and the table.
pub fn sim_pyramid<T: FwDepth>(src: &[T], w: usize, h: usize, count: usize, seed: u64) -> (Vec<T>, Vec<i32>) {
    let dims = level_dims(w, h, count);
    let (lt, len) = level_table(&dims).expect("level table");
    let mut lev = Global::new(len, T::ZERO);
    if count < 2 {
        return (lev.into_vec(), lt);
    }
    let mut seeds = Seeds(seed, 0);
    let b = SimBackend::new(seed);
    let taps = Taps::upload(&b, &pyramid::BLUR5_TAPS).expect("taps");
    for i in 1..count {
        let (sw, sh) = dims[i - 1];
        let (dw, dh) = dims[i];
        let ns = sw * sh * CHANNELS;
        let snap: Vec<T> = if i == 1 { src.to_vec() } else { lev.clone().into_vec() };
        let from_off = if i == 1 { 0u32 } else { lt[(i - 1) * 3] as u32 };
        let mut f0 = Global::new(ns, 0.0f32);
        sim::launch_1d(ns, seeds.next(), |j| fw_load_item(j, &snap, from_off, &mut f0, ns as u32));
        let f0 = RefCell::new(written(f0, "fw_load"));
        let f1 = RefCell::new(vec![0.0f32; ns]);
        b.launch(Launch::HPass { src: &f0, dst: &f1, w: sw, h: sh, c: CHANNELS, taps: &taps, wrap: false }).expect("hpass");
        let blurred: Vec<T> = if T::INT {
            let mut r = Global::new(ns, 0.0f32);
            let f1v = f1.borrow().clone();
            sim::launch_1d(ns, seeds.next(), |j| T::round_item(j, &f1v, &mut r, ns as u32));
            *f0.borrow_mut() = written(r, "wb_round");
            b.launch(Launch::VPass { src: &f0, dst: &f1, w: sw, h: sh, c: CHANNELS, taps: &taps }).expect("vpass");
            let f1v = f1.borrow().clone();
            let mut t = Global::new(ns, T::ZERO);
            sim::launch_1d(ns, seeds.next(), |j| T::from_float_item(j, &f1v, &mut t, ns as u32));
            written(t, "wb_from_float")
        } else {
            b.launch(Launch::VPass { src: &f1, dst: &f0, w: sw, h: sh, c: CHANNELS, taps: &taps }).expect("vpass");
            T::from_f32_vec(f0.borrow().clone())
        };
        let (sx, sy) = resample_scales(sw, sh, dw, dh);
        let off = lt[i * 3] as u32;
        sim::launch_2d(dw, dh, seeds.next(), |x, y| fw_resample_item(x, y, &blurred, sw as i32, sh as i32, &mut lev, off, dw as i32, dh as i32, sx, sy));
    }
    assert_eq!(lev.written(), len, "fw_resample: not every level sample was written");
    (lev.into_vec(), lt)
}

/// The outcome of the simulated stage: the destination samples, the coverage, the refusal word.
pub struct SimOut<T> {
    pub out: Vec<T>,
    pub cov: Option<Vec<u8>>,
    pub flag: i32,
    /// The hit total (`total[0]`), or `None` when the counting pass refused.
    pub hits: Option<u32>,
}

/// [`super::forward_warp_enqueue`] of a warp plan in the simulator.
pub fn sim_warp<T: FwDepth>(src: &[T], m: &HostMesh, want_cov: bool, seed: u64) -> SimOut<T> {
    let (w, h) = (m.width, m.height);
    let (lev, lt) = sim_pyramid(src, w, h, m.count, seed);
    let mut seeds = Seeds(seed ^ 0x5151, 0);
    let npx = w * h;
    let nc = npx.div_ceil(k::FW_CHUNK);
    let ns = npx.div_ceil(k::FW_SUPER);
    let zero = [0.0f32];
    let ma = MeshArgs { pos: &m.pos, dl: m.levels.as_deref().unwrap_or(&zero), has_d: m.levels.is_some(), w: w as i32, h: h as i32 };
    let mut cnt = vec![0u32; npx];
    let mut total = vec![0u32; 2];
    let mut iflag = 0i32;
    let maxrec = max_records(w, h);
    let bands = triangle_bands(w, h);
    for &(key0, n) in &bands {
        sim::launch_1d(n as usize, seeds.next(), |g| fw_count_item(g, &ma, &mut cnt, &mut total, maxrec, &mut iflag, key0, n));
    }
    let mut off = Global::new(npx, 0u32);
    let mut csum = Global::new(nc, 0u32);
    sim::launch_1d(nc, seeds.next(), |kk| fw_scan_chunks_item(kk, &cnt, &mut off, &mut csum, npx as u32, nc as u32));
    let (off, csum) = (written(off, "fw_scan_chunks off"), written(csum, "fw_scan_chunks csum"));
    let mut cpre = Global::new(nc, 0u32);
    let mut ssum = Global::new(ns, 0u32);
    sim::launch_1d(ns, seeds.next(), |kk| fw_scan_supers_item(kk, &csum, &mut cpre, &mut ssum, nc as u32, ns as u32));
    let (cpre, ssum) = (written(cpre, "fw_scan_supers cpre"), written(ssum, "fw_scan_supers ssum"));
    let mut sbase = Global::new(ns, 0u32);
    sim::launch_1d(1, seeds.next(), |g| fw_scan_top_item(g, &ssum, &mut sbase, &mut total, ns as u32));
    let sbase = written(sbase, "fw_scan_top");
    if iflag != 0 || total[0] > maxrec || total[0] != total[1] {
        return SimOut { out: Vec::new(), cov: None, flag: iflag | 2, hits: None };
    }
    let nrec = total[0];
    let mut rec = Global::new((nrec as usize).max(1), [0u32; 4]);
    let mut fill = vec![0u32; npx];
    let mut flag = 0i32;
    for &(key0, n) in &bands {
        sim::launch_1d(n as usize, seeds.next(), |g| fw_emit_item(g, &ma, &mut cnt, &mut fill, &off, &cpre, &sbase, &mut rec, nrec, &mut flag, key0, n));
    }
    assert_eq!(rec.written(), nrec as usize, "fw_emit: every record slot is written once");
    let mut rec = rec.into_vec();
    let mut out = Global::new(npx * CHANNELS, T::ZERO);
    let mut cov = Global::new(npx, 0u8);
    for (y0, rows) in row_bands(w, h) {
        let fa = FoldArgs {
            src,
            lev: &lev,
            lt: &lt,
            nlev: m.count as i32,
            cnt: &cnt,
            off: &off,
            cpre: &cpre,
            sbase: &sbase,
            want_cov,
            w: w as i32,
            h: h as i32,
            y0: y0 as i32,
            rows: rows as i32,
        };
        sim::launch_2d(w, rows, seeds.next(), |x, gy| fw_fold_item(x, gy, &fa, &mut rec, &mut out, &mut cov, &mut flag));
    }
    let out = written(out, "fw_fold out");
    let cov = if want_cov { Some(written(cov, "fw_fold cov")) } else { None };
    SimOut { out, cov, flag, hits: Some(nrec) }
}

fn samples_of<T: FwDepth>(img: &Image) -> Vec<T>
where
    Vec<T>: FromSamples,
{
    <Vec<T> as FromSamples>::from(&img.samples)
}

/// Typed access to the samples of an image.
pub trait FromSamples: Sized {
    fn from(s: &Samples) -> Self;
    fn into_samples(self) -> Samples;
}

impl FromSamples for Vec<u8> {
    fn from(s: &Samples) -> Vec<u8> {
        match s {
            Samples::U8(v) => v.clone(),
            _ => unreachable!("format"),
        }
    }
    fn into_samples(self) -> Samples {
        Samples::U8(self)
    }
}

impl FromSamples for Vec<u16> {
    fn from(s: &Samples) -> Vec<u16> {
        match s {
            Samples::U16(v) => v.clone(),
            _ => unreachable!("format"),
        }
    }
    fn into_samples(self) -> Samples {
        Samples::U16(self)
    }
}

impl FromSamples for Vec<f32> {
    fn from(s: &Samples) -> Vec<f32> {
        match s {
            Samples::F32(v) => v.clone(),
            _ => unreachable!("format"),
        }
    }
    fn into_samples(self) -> Samples {
        Samples::F32(self)
    }
}

fn run_depth<T: FwDepth>(src: &Image, plan: &Plan, want_cov: bool, seed: u64) -> Result<(Image, Option<Coverage>), StageError>
where
    Vec<T>: FromSamples,
{
    let (w, h) = (src.width, src.height);
    let data: Vec<T> = samples_of(src);
    let (img, cov) = match plan {
        Plan::Copy { .. } => (Image { width: w, height: h, samples: data.into_samples() }, want_cov.then(|| Coverage::new(w, h, 0xff))),
        Plan::Warp(m) => {
            let r = sim_warp(&data, m, want_cov, seed);
            flag_result(r.flag, "forward warp")?;
            (Image { width: w, height: h, samples: r.out.into_samples() }, r.cov.map(|data| Coverage { width: w, height: h, data }))
        }
    };
    if crate::device::image::has_nan(&img) {
        return Err(StageError::NanOutput);
    }
    Ok((img, cov))
}

/// [`super::forward_warp_with_gpu`] in the simulator, with the raw result when the NaN rule
/// refuses it (`Err((NanOutput, Some(raw)))`).
pub fn sim_forward_warp_with(
    out: &Image,
    src: &Image,
    flow: &crate::flow::FlowField,
    amount: f32,
    opts: crate::forward::ForwardOpts,
    want_cov: bool,
    seed: u64,
) -> Result<(Image, Option<Coverage>), StageError> {
    let plan = super::plan(out, src, flow, amount, opts)?;
    match src.samples {
        Samples::U8(_) => run_depth::<u8>(src, &plan, want_cov, seed),
        Samples::U16(_) => run_depth::<u16>(src, &plan, want_cov, seed),
        Samples::F32(_) => run_depth::<f32>(src, &plan, want_cov, seed),
    }
}

/// The raw simulated output of a warp plan without the flag and NaN rules (for the NaN rule
/// tests): `None` for the copy plan or a refused counting pass.
pub fn sim_raw(src: &Image, flow: &crate::flow::FlowField, amount: f32, seed: u64) -> Option<(Image, i32)> {
    let plan = super::plan(src, src, flow, amount, crate::forward::ForwardOpts::default()).ok()?;
    let Plan::Warp(m) = plan else { return None };
    let (w, h) = (src.width, src.height);
    let (samples, flag) = match &src.samples {
        Samples::U8(v) => {
            let r = sim_warp(v, &m, false, seed);
            r.hits?;
            (Samples::U8(r.out), r.flag)
        }
        Samples::U16(v) => {
            let r = sim_warp(v, &m, false, seed);
            r.hits?;
            (Samples::U16(r.out), r.flag)
        }
        Samples::F32(v) => {
            let r = sim_warp(v, &m, false, seed);
            r.hits?;
            (Samples::F32(r.out), r.flag)
        }
    };
    Some((Image { width: w, height: h, samples }, flag))
}
