//! The device-free backend of the solver tests: the convolution stages' `SimBackend` (its own
//! launches in shuffled orders) with the solver launches added, each running the kernel's Rust
//! transcription (`kernels/solver.rs`, `*_item`) once per work item through `device::sim` in an
//! order drawn from the seed of [`backend`] (a new shuffle per launch). Per pixel outputs are
//! write-once `sim::Global`s that must be written completely; the in place sweeps run on an
//! [`Owned`] flow, which panics when two items of one launch touch one element (a race on a
//! device), so a sweep that read another item's row or column would fail here.

use super::{SolverBackend, StageError, SvLaunch};
use crate::device::convstage::sim::SimBackend;
use crate::device::kernels::solver::{self as ks, SweepFlow};
use crate::device::sim::{self, Global};
use std::cell::{Cell, RefCell};

thread_local! {
    static SEED: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
}

/// A simulator backend: the convolution launches shuffled by `SimBackend::new(seed)`, the solver
/// launches by the same seed (0 keeps every launch in the identity order).
pub fn backend(seed: u64) -> SimBackend {
    SEED.with(|s| s.set((seed, 0)));
    SimBackend::new(seed)
}

fn next_seed() -> u64 {
    SEED.with(|s| {
        let (seed, k) = s.get();
        s.set((seed, k + 1));
        if seed == 0 {
            0
        } else {
            seed ^ (k.wrapping_add(0x51).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        }
    })
}

fn snapshot<T: Copy>(b: &RefCell<Vec<T>>) -> Vec<T> {
    b.borrow().clone()
}

fn finish<T: Copy>(dst: &RefCell<Vec<T>>, out: Global<T>, what: &str) {
    assert_eq!(out.len(), dst.borrow().len(), "{what}: output of another length");
    assert_eq!(out.written(), out.len(), "{what}: not every element was written");
    *dst.borrow_mut() = out.into_vec();
}

fn global_f32(len: usize) -> Global<f32> {
    Global::new(len, f32::from_bits(0x7fc0_beef))
}

const NONE: usize = usize::MAX;

/// The flow of an in place sweep: reads see writes, and every element belongs to the first item
/// that touches it during the launch.
pub struct Owned {
    data: Vec<f32>,
    owner: Vec<usize>,
    item: usize,
}

impl Owned {
    fn new(data: Vec<f32>) -> Owned {
        let n = data.len();
        Owned { data, owner: vec![NONE; n], item: 0 }
    }

    fn touch(&mut self, i: usize) {
        let o = self.owner[i];
        if o == NONE {
            self.owner[i] = self.item;
        } else {
            assert_eq!(o, self.item, "element {i} touched by the items {o} and {} of one launch", self.item);
        }
    }
}

impl SweepFlow for Owned {
    fn get(&mut self, i: usize) -> f32 {
        self.touch(i);
        self.data[i]
    }

    fn set(&mut self, i: usize, v: f32) {
        self.touch(i);
        self.data[i] = v;
    }
}

fn opt(b: Option<&RefCell<Vec<f32>>>) -> (Vec<f32>, bool) {
    match b {
        Some(m) => (snapshot(m), true),
        None => (Vec::new(), false),
    }
}

impl SolverBackend for SimBackend {
    fn sv_launch(&self, l: SvLaunch<'_, Self>) -> Result<(), StageError> {
        let seed = next_seed();
        match l {
            SvLaunch::Grad { img, mask, wrap, grad, tensor, w, h } => {
                let im = snapshot(img);
                let (m, has) = opt(mask);
                let mut g = global_f32(grad.borrow().len());
                let mut t = global_f32(tensor.borrow().len());
                sim::launch_2d(w, h, seed, |x, y| ks::sv_grad_item(x, y, &im, &m, has, wrap, &mut g, &mut t, w as i32, h as i32));
                finish(grad, g, "grad");
                finish(tensor, t, "tensor");
            }
            SvLaunch::CalcB { i1, i0, grad, flow, mask, wrap, b, w, h } => {
                let (v1, v0, g, f) = (snapshot(i1), snapshot(i0), snapshot(grad), snapshot(flow));
                let (m, has) = opt(mask);
                let mut out = global_f32(b.borrow().len());
                sim::launch_2d(w, h, seed, |x, y| ks::sv_calc_b_item(x, y, &v1, &v0, &g, &f, &m, has, wrap, &mut out, w as i32, h as i32));
                finish(b, out, "calc_b");
            }
            SvLaunch::NewDisp { flow, a, b, n } => {
                let (f, av, bv) = (snapshot(flow), snapshot(a), snapshot(b));
                let mut out = global_f32(f.len());
                sim::launch_1d(n, seed, |i| ks::sv_new_disp_item(i, &f, &av, &bv, &mut out, n as i32));
                finish(flow, out, "new_disp");
            }
            SvLaunch::ApplyUser { flow, d, wt, n } => {
                let (f, dv, wv) = (snapshot(flow), snapshot(d), snapshot(wt));
                let mut out = global_f32(f.len());
                sim::launch_1d(n, seed, |i| ks::sv_apply_user_item(i, &f, &dv, &wv, &mut out, n as i32));
                finish(flow, out, "apply_user");
            }
            SvLaunch::SweepRows { a, b, flow, gate, w, h, x0, count, dir } => {
                let (av, bv) = (snapshot(a), snapshot(b));
                let (g, has) = opt(gate);
                let mut f = Owned::new(snapshot(flow));
                sim::launch_1d(h - 2, seed, |item| {
                    f.item = item;
                    ks::sv_sweep_rows_item(item, &av, &bv, &mut f, &g, has, w as i32, h as i32, x0 as i32, count as i32, dir as i32)
                });
                *flow.borrow_mut() = f.data;
            }
            SvLaunch::SweepCols { a, b, flow, gate, w, h, xoff, ncols, y0, count, dir, wrap } => {
                let (av, bv) = (snapshot(a), snapshot(b));
                let (g, has) = opt(gate);
                let mut f = Owned::new(snapshot(flow));
                sim::launch_1d(ncols, seed, |item| {
                    f.item = item;
                    ks::sv_sweep_cols_item(
                        item, &av, &bv, &mut f, &g, has, w as i32, h as i32, xoff as i32, ncols as i32, y0 as i32, count as i32, dir as i32, wrap,
                    )
                });
                *flow.borrow_mut() = f.data;
            }
            SvLaunch::SweepRing { a, b, old, flow, gate, w, h, right } => {
                let (av, bv, ov) = (snapshot(a), snapshot(b), snapshot(old));
                let (g, has) = opt(gate);
                let mut out = global_f32(flow.borrow().len());
                sim::launch_2d(w, h, seed, |x, y| ks::sv_sweep_ring_item(x, y, &av, &bv, &ov, &mut out, &g, has, w as i32, h as i32, right));
                finish(flow, out, "sweep_ring");
            }
            SvLaunch::RangeRows { flow, state, rows, w, h, wrap } => {
                let (f, st) = (snapshot(flow), snapshot(state));
                let mut out = Global::new(rows.borrow().len(), 7i32);
                sim::launch_1d(h, seed, |y| ks::sv_range_rows_item(y, &f, &st, &mut out, w as i32, h as i32, wrap));
                finish(rows, out, "range_rows");
            }
            SvLaunch::RangeDecide { state, rows, h } => {
                let (st, r) = (snapshot(state), snapshot(rows));
                let mut out = Global::new(st.len(), 7i32);
                sim::launch_1d(1, seed, |g| ks::sv_range_decide_item(g, &st, &r, &mut out, h as i32));
                finish(state, out, "range_decide");
            }
            SvLaunch::RangeApply { flow, state, n } => {
                let (f, st) = (snapshot(flow), snapshot(state));
                let mut out = global_f32(f.len());
                sim::launch_1d(n, seed, |i| ks::sv_range_apply_item(i, &f, &st, &mut out, n as i32));
                finish(flow, out, "range_apply");
            }
            SvLaunch::Scale { data, f, n } => {
                let d = snapshot(data);
                let mut out = global_f32(d.len());
                sim::launch_1d(n, seed, |i| ks::sv_scale_item(i, &d, f, &mut out, n as i32));
                finish(data, out, "scale");
            }
            SvLaunch::Zero { data, n } => {
                let mut out = global_f32(data.borrow().len());
                sim::launch_1d(n, seed, |i| ks::sv_zero_item(i, &mut out, n as i32));
                finish(data, out, "zero");
            }
            SvLaunch::UpsampleWrapx { src, sw, sh, dst, dw, dh, scale } => {
                let s = snapshot(src);
                let mut out = global_f32(dst.borrow().len());
                sim::launch_2d(dw, dh, seed, |x, y| ks::sv_upsample_flow_wrapx_item(x, y, &s, sw as i32, sh as i32, &mut out, dw as i32, dh as i32, scale));
                finish(dst, out, "upsample_wrapx");
            }
        }
        Ok(())
    }
}
