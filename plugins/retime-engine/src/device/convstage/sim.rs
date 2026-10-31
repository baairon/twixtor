//! The device-free backend of the stage tests: every launch runs the kernel's Rust transcription
//! (`device/kernels/*.rs`, `*_item`) once per work item through `device::sim`, in an order drawn
//! from the backend's seed (a new shuffle per launch), with each output a write-once
//! `sim::Global` (a second write to one element is a race and panics). A launch must write every
//! element of its output exactly once, except the strided image stores, which write their
//! channel and keep the other samples.

use super::{Backend, Launch, StageError};
use crate::device::cl::Pod;
use crate::device::kernels::{conv as kc, holefill as kh, pyramid as kp, resample as kr};
use crate::device::sim::{self, Global};
use std::cell::{Cell, RefCell};

pub struct SimBackend {
    seed: u64,
    launches: Cell<u64>,
}

impl SimBackend {
    pub fn new(seed: u64) -> SimBackend {
        SimBackend { seed, launches: Cell::new(0) }
    }

    /// The order seed of the next launch (0 stays the identity order for every launch).
    fn next_seed(&self) -> u64 {
        let k = self.launches.get();
        self.launches.set(k + 1);
        if self.seed == 0 {
            0
        } else {
            self.seed ^ k.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        }
    }
}

fn snapshot<T: Copy>(b: &RefCell<Vec<T>>) -> Vec<T> {
    b.borrow().clone()
}

/// Stores `out` into `dst`, after checking that every element was written once.
fn finish<T: Copy>(dst: &RefCell<Vec<T>>, out: Global<T>, what: &str) {
    assert_eq!(out.written(), out.len(), "{what}: not every element was written");
    *dst.borrow_mut() = out.into_vec();
}

fn global_f32(len: usize) -> Global<f32> {
    Global::new(len, f32::from_bits(0x7fc0_dead))
}

impl Backend for SimBackend {
    type Buf<T: Pod> = RefCell<Vec<T>>;

    fn upload<T: Pod>(&self, data: &[T]) -> Result<RefCell<Vec<T>>, StageError> {
        assert!(!data.is_empty(), "a buffer needs at least one element");
        Ok(RefCell::new(data.to_vec()))
    }

    fn alloc<T: Pod>(&self, len: usize) -> Result<RefCell<Vec<T>>, StageError> {
        assert!(len > 0, "a buffer needs at least one element");
        Ok(RefCell::new(vec![T::default(); len]))
    }

    fn download<T: Pod>(&self, buf: &RefCell<Vec<T>>) -> Result<Vec<T>, StageError> {
        Ok(snapshot(buf))
    }

    fn length<T: Pod>(&self, buf: &RefCell<Vec<T>>) -> usize {
        buf.borrow().len()
    }

    fn copy<T: Pod>(&self, src: &RefCell<Vec<T>>, dst: &RefCell<Vec<T>>) -> Result<(), StageError> {
        assert!(!std::ptr::eq(src, dst), "copy of a buffer onto itself");
        let s = snapshot(src);
        assert_eq!(s.len(), dst.borrow().len());
        *dst.borrow_mut() = s;
        Ok(())
    }

    fn launch(&self, l: Launch<'_, Self>) -> Result<(), StageError> {
        let seed = self.next_seed();
        match l {
            Launch::HPass { src, dst, w, h, c, taps, wrap } => {
                let s = snapshot(src);
                let k = snapshot(&taps.buf);
                let mut out = global_f32(dst.borrow().len());
                let (wi, hi, ci, ni) = (w as i32, h as i32, c as i32, taps.n as i32);
                sim::launch_2d(w * c, h, seed, |g, y| {
                    if wrap {
                        kc::cv_hpass_wrapx_item(g, y, &s, &mut out, wi, hi, ci, &k, ni)
                    } else {
                        kc::cv_hpass_item(g, y, &s, &mut out, wi, hi, ci, &k, ni)
                    }
                });
                finish(dst, out, "hpass");
            }
            Launch::VPass { src, dst, w, h, c, taps } => {
                let s = snapshot(src);
                let k = snapshot(&taps.buf);
                let mut out = global_f32(dst.borrow().len());
                let (wi, hi, ci, ni) = (w as i32, h as i32, c as i32, taps.n as i32);
                sim::launch_2d(w * c, h, seed, |t, y| kc::cv_vpass_item(t, y, &s, &mut out, wi, hi, ci, &k, ni));
                finish(dst, out, "vpass");
            }
            Launch::VPassAbove { src, dst, w, h, c, taps } => {
                let s = snapshot(src);
                let k = snapshot(&taps.buf);
                let mut out = global_f32(dst.borrow().len());
                let (wi, hi, ci, ni) = (w as i32, h as i32, c as i32, taps.n as i32);
                sim::launch_2d(w * c, h, seed, |t, y| kc::cv_vpass_above_item(t, y, &s, &mut out, wi, hi, ci, &k, ni));
                finish(dst, out, "vpass above");
            }
            Launch::Resample { src, sw, sh, dst, dw, dh, sx, sy, wrap } => {
                let s = snapshot(src);
                let mut out = global_f32(dst.borrow().len());
                let (swi, shi, dwi, dhi) = (sw as i32, sh as i32, dw as i32, dh as i32);
                sim::launch_2d(dw, dh, seed, |x, y| {
                    if wrap {
                        kr::rs_resample1_wrapx_item(x, y, &s, swi, shi, &mut out, dwi, dhi, sx, sy)
                    } else {
                        kr::rs_resample1_item(x, y, &s, swi, shi, &mut out, dwi, dhi, sx, sy)
                    }
                });
                finish(dst, out, "resample");
            }
            Launch::Gather { img, w, h, xs, ys, out: o, n, lanes, wrap } => {
                let (im, xv, yv) = (snapshot(img), snapshot(xs), snapshot(ys));
                let mut out = global_f32(o.borrow().len());
                let (wi, hi) = (w as i32, h as i32);
                sim::launch_1d(n, seed, |i| {
                    if lanes == 2 {
                        let v = kr::rs_bilinear2(&im, wi, hi, xv[i], yv[i]);
                        out.set(i * 2, v[0]);
                        out.set(i * 2 + 1, v[1]);
                    } else if wrap {
                        out.set(i, kr::rs_bilinear1_wrapx(&im, wi, hi, xv[i], yv[i]));
                    } else {
                        out.set(i, kr::rs_bilinear1(&im, wi, hi, xv[i], yv[i]));
                    }
                });
                finish(o, out, "gather");
            }
            Launch::Upsample { src, sw, sh, dst, dw, dh, scale } => {
                let s = snapshot(src);
                let mut out = global_f32(dst.borrow().len());
                let (swi, shi, dwi, dhi) = (sw as i32, sh as i32, dw as i32, dh as i32);
                sim::launch_2d(dw, dh, seed, |x, y| kp::pm_upsample_flow_item(x, y, &s, swi, shi, &mut out, dwi, dhi, scale));
                finish(dst, out, "upsample");
            }
            Launch::LumaU8 { img, out: o, track_alpha, video, n } => {
                let s = snapshot(img);
                let mut out = global_f32(o.borrow().len());
                sim::launch_1d(n, seed, |i| kp::pm_luma_u8_item(i, &s, &mut out, track_alpha, video, n as i32));
                finish(o, out, "luma");
            }
            Launch::LumaU16 { img, out: o, track_alpha, video, n } => {
                let s = snapshot(img);
                let mut out = global_f32(o.borrow().len());
                sim::launch_1d(n, seed, |i| kp::pm_luma_u16_item(i, &s, &mut out, track_alpha, video, n as i32));
                finish(o, out, "luma");
            }
            Launch::LumaF32 { img, out: o, track_alpha, video, n } => {
                let s = snapshot(img);
                let mut out = global_f32(o.borrow().len());
                sim::launch_1d(n, seed, |i| kp::pm_luma_f32_item(i, &s, &mut out, track_alpha, video, n as i32));
                finish(o, out, "luma");
            }
            Launch::Subtract { a, b, n } => {
                let (av, bv) = (snapshot(a), snapshot(b));
                let mut out = global_f32(av.len());
                sim::launch_1d(n, seed, |i| kp::pm_subtract_item(i, &av, &bv, &mut out, n as i32));
                finish(a, out, "subtract");
            }
            Launch::Binarize { p, n } => {
                let pv = snapshot(p);
                let mut out = global_f32(pv.len());
                sim::launch_1d(n, seed, |i| kp::pm_binarize_item(i, &pv, &mut out, n as i32));
                finish(p, out, "binarize");
            }
            Launch::FlagsU8 { m, hole, eps, n } => flags(seed, snapshot(m).iter().map(|&v| v as f64).collect(), hole, eps, n),
            Launch::FlagsU16 { m, hole, eps, n } => flags(seed, snapshot(m).iter().map(|&v| v as f64).collect(), hole, eps, n),
            Launch::FlagsF32 { m, hole, eps, n } => flags(seed, snapshot(m).iter().map(|&v| v as f64).collect(), hole, eps, n),
            Launch::Weight0 { hole, wt, n } => {
                let hv = snapshot(hole);
                let mut out = global_f32(wt.borrow().len());
                sim::launch_1d(n, seed, |i| kh::hf_weight0_item(i, &hv, &mut out, n as i32));
                finish(wt, out, "weight0");
            }
            Launch::Load { plane, hole, v0, n } => {
                let (pv, hv) = (snapshot(plane), snapshot(hole));
                let mut out = global_f32(v0.borrow().len());
                sim::launch_1d(n, seed, |i| kh::hf_load_item(i, &pv, &hv, &mut out, n as i32));
                finish(v0, out, "load");
            }
            Launch::Combine { sol, value, weight, n } => {
                let (sv, vv, wv) = (snapshot(sol), snapshot(value), snapshot(weight));
                let mut out = global_f32(sv.len());
                sim::launch_1d(n, seed, |i| kh::hf_combine_item(i, &sv, &vv, &wv, &mut out, n as i32));
                finish(sol, out, "combine");
            }
            Launch::ExtractU8 { img, plane, stride, c, n } => extract(seed, snapshot(img), plane, stride, c, n, |v| v as f32),
            Launch::ExtractU16 { img, plane, stride, c, n } => extract(seed, snapshot(img), plane, stride, c, n, |v| v as f32),
            Launch::ExtractF32 { img, plane, stride, c, n } => extract(seed, snapshot(img), plane, stride, c, n, |v| v),
            Launch::StoreU8 { plane, img, stride, c, n } => {
                let pv = snapshot(plane);
                let mut out = Global::new(img.borrow().len(), 0u8);
                sim::launch_1d(n, seed, |i| kh::hf_store_u8_item(i, &pv, &mut out, stride as i32, c as i32, n as i32));
                merge_channel(img, out, stride, c, n);
            }
            Launch::StoreU16 { plane, img, stride, c, n } => {
                let pv = snapshot(plane);
                let mut out = Global::new(img.borrow().len(), 0u16);
                sim::launch_1d(n, seed, |i| kh::hf_store_u16_item(i, &pv, &mut out, stride as i32, c as i32, n as i32));
                merge_channel(img, out, stride, c, n);
            }
            Launch::StoreF32 { plane, img, stride, c, n } => {
                let pv = snapshot(plane);
                let mut out = global_f32(img.borrow().len());
                sim::launch_1d(n, seed, |i| {
                    if i < n {
                        out.set(i * stride + c, pv[i]);
                    }
                });
                merge_channel(img, out, stride, c, n);
            }
            Launch::Invert { data, n } => {
                let dv = snapshot(data);
                let mut out = global_f32(dv.len());
                sim::launch_1d(n, seed, |i| kc::cv_choke_invert_item(i, &dv, &mut out, n as i32));
                finish(data, out, "invert");
            }
            Launch::Threshold { data, t, den, n } => {
                let dv = snapshot(data);
                let mut out = global_f32(dv.len());
                sim::launch_1d(n, seed, |i| kc::cv_choke_threshold_item(i, &dv, &mut out, t, den, n as i32));
                finish(data, out, "threshold");
            }
            Launch::Mix { data, c, t, it, n } => {
                let (dv, cv) = (snapshot(data), snapshot(c));
                let mut out = global_f32(dv.len());
                sim::launch_1d(n, seed, |i| kc::cv_choke_mix_item(i, &dv, &cv, &mut out, t, it, n as i32));
                finish(data, out, "mix");
            }
            Launch::Clamp { data, n } => {
                let dv = snapshot(data);
                let mut out = global_f32(dv.len());
                sim::launch_1d(n, seed, |i| kc::cv_choke_clamp_item(i, &dv, &mut out, n as i32));
                finish(data, out, "clamp");
            }
        }
        Ok(())
    }
}

fn flags(seed: u64, m: Vec<f64>, hole: &RefCell<Vec<u8>>, eps: f64, n: usize) {
    let mut out = Global::new(hole.borrow().len(), 7u8);
    sim::launch_1d(n, seed, |i| kh::hf_flags_item(i, &m, &mut out, eps, n as i32));
    finish(hole, out, "flags");
}

fn extract<T: Copy>(seed: u64, img: Vec<T>, plane: &RefCell<Vec<f32>>, stride: usize, c: usize, n: usize, to_f32: fn(T) -> f32) {
    let mut out = global_f32(plane.borrow().len());
    sim::launch_1d(n, seed, |i| kh::hf_extract_item(i, &img, &mut out, stride as i32, c as i32, n as i32, to_f32));
    finish(plane, out, "extract");
}

/// The strided store: element `i * stride + c` from `out` for `i < n`, every other one kept.
fn merge_channel<T: Copy>(img: &RefCell<Vec<T>>, out: Global<T>, stride: usize, c: usize, n: usize) {
    assert_eq!(out.written(), n, "store: not every sample of the channel was written");
    let v = out.into_vec();
    let mut dst = img.borrow_mut();
    for i in 0..n {
        dst[i * stride + c] = v[i * stride + c];
    }
}
