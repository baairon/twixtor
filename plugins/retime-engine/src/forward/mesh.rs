//! The displaced vertex grid of the forward warp: the vertex positions and the per vertex mip
//! level `d`. Both only depend on the flow, so every row is independent and they are computed in
//! parallel (the result does not depend on the thread count).

use crate::flow::FlowField;
use rayon::prelude::*;

/// Geometry shared by the two builders: source size and the flow ratio (`rx = (flow.w - 1) /
/// (src.w - 1)`, `sx = amount / rx`, both as `(float)(int)` quotients).
#[derive(Clone, Copy)]
pub(crate) struct Grid {
    pub w: usize,
    pub h: usize,
    pub rx: f32,
    pub ry: f32,
    pub sx: f32,
    pub sy: f32,
}

impl Grid {
    pub fn new(w: usize, h: usize, flow: &FlowField, amount: f32) -> Grid {
        let rx = (flow.width as i32 - 1) as f32 / (w as i32 - 1) as f32;
        let ry = (flow.height as i32 - 1) as f32 / (h as i32 - 1) as f32;
        Grid { w, h, rx, ry, sx: amount / rx, sy: amount / ry }
    }
}

/// Positions of the vertices of row `r`: `x' = fx * sx + c`, `y' = fy * sy + r` with the flow
/// sampled at `(c * rx, r * ry)`, then, for `r != 0` and the columns `1..w`, the tie breaking
/// epsilon `x' = (float)((double)x' + 0.0001)` (a double precision addition, narrowed), the same
/// for `y'`.
pub(crate) fn position_row(grid: &Grid, flow: &FlowField, r: usize, row: &mut [[f32; 2]]) {
    let fy_pos = (r as i32 as f32) * grid.ry;
    let rf = r as i32 as f32;
    for (c, p) in row.iter_mut().enumerate() {
        let cf = c as i32 as f32;
        let [fx, fy] = flow.bilerp(cf * grid.rx, fy_pos);
        *p = [fx * grid.sx + cf, fy * grid.sy + rf];
    }
    if r != 0 {
        for p in row.iter_mut().skip(1) {
            p[0] = ((p[0] as f64) + 0.0001f64) as f32;
            p[1] = ((p[1] as f64) + 0.0001f64) as f32;
        }
    }
}

/// The flow term of one neighbour in [`mip_level`]: `(fx * sx + idx_x) - x'c`,
/// `(fy * sy + idx_y) - y'c`.
#[inline(always)]
fn neighbour(grid: &Grid, flow: &FlowField, ci: i32, ri: i32, xc: f32, yc: f32) -> (f32, f32) {
    let [fx, fy] = flow.bilerp((ci as f32) * grid.rx, (ri as f32) * grid.ry);
    ((fx * grid.sx + ci as f32) - xc, (fy * grid.sy + ri as f32) - yc)
}

/// The mip level of the vertex `(c, r)`: the 8 neighbours with out of range indices mirrored
/// (`-1 -> 1`, `w -> w - 2`, `h -> h - 2`), the distance of each displaced neighbour from the
/// displaced vertex without the epsilon; the four diagonals are summed in single precision in the
/// order right-up, left-up, left-down, right-down, widened, multiplied by `0.7071067811865476`,
/// the four axis neighbours (up, down, left, right) are added in double precision, the sum is
/// narrowed, multiplied by 0.125 (through a double, exact), and
/// `d = base - logf(m) * (1.0f / logf(2.0f))` with `base = 0`.
pub(crate) fn mip_level(grid: &Grid, flow: &FlowField, c: usize, r: usize) -> f32 {
    let (w, h) = (grid.w as i32, grid.h as i32);
    let (c, r) = (c as i32, r as i32);
    let up = if r - 1 < 0 { 1 } else { r - 1 };
    let down = if h - 1 < r + 1 { h - 2 } else { r + 1 };
    let left = if c - 1 < 0 { 1 } else { c - 1 };
    let right = if w - 1 < c + 1 { w - 2 } else { c + 1 };
    let [fx, fy] = flow.bilerp((c as f32) * grid.rx, (r as f32) * grid.ry);
    let xc = fx * grid.sx + c as f32;
    let yc = fy * grid.sy + r as f32;
    let len = |t: (f32, f32)| -> f32 { (t.1 * t.1 + t.0 * t.0).sqrt() };
    let ru = len(neighbour(grid, flow, right, up, xc, yc));
    let lu = len(neighbour(grid, flow, left, up, xc, yc));
    let ld = len(neighbour(grid, flow, left, down, xc, yc));
    let rd = len(neighbour(grid, flow, right, down, xc, yc));
    let u = len(neighbour(grid, flow, c, up, xc, yc));
    let dn = len(neighbour(grid, flow, c, down, xc, yc));
    let l = len(neighbour(grid, flow, left, r, xc, yc));
    let rr = len(neighbour(grid, flow, right, r, xc, yc));
    let diag = ((ru + lu) + ld) + rd;
    let mut t = (diag as f64) * 0.7071067811865476f64;
    t += u as f64;
    t += dn as f64;
    t += l as f64;
    t += rr as f64;
    let m = (((t as f32) as f64) * 0.125f64) as f32;
    let inv_ln2 = 1.0f32 / 2.0f32.ln();
    -(m.ln() * inv_ln2) + 0.0f32
}

/// All positions (`h` rows of `w`), rows computed in parallel.
pub(crate) fn positions(grid: &Grid, flow: &FlowField) -> Vec<[f32; 2]> {
    let mut pos = vec![[0.0f32; 2]; grid.w * grid.h];
    pos.par_chunks_mut(grid.w).enumerate().for_each(|(r, row)| position_row(grid, flow, r, row));
    pos
}

/// All mip levels, rows computed in parallel.
pub(crate) fn mip_levels(grid: &Grid, flow: &FlowField) -> Vec<f32> {
    let mut d = vec![0.0f32; grid.w * grid.h];
    d.par_chunks_mut(grid.w).enumerate().for_each(|(r, row)| {
        for (c, v) in row.iter_mut().enumerate() {
            *v = mip_level(grid, flow, c, r);
        }
    });
    d
}
