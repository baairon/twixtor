//! The optical flow solver and the vector field type shared with the warps. Entry points:
//! [`solve`] (one flow, `X(p) ~ Y(p + out(p))`), [`solve_pair`] (the two flows of a blend, the
//! pyramids are built once) and the lower level [`build_pyramid`] / [`solve_pyramids`].
//!
//! The returned flow lives at the WORKING resolution of the estimator (`FlowSettings::geometry`),
//! not at the output resolution: it has the size `(W >> R, H >> R)` and every warp resamples it
//! bilinearly.

use fx_core::FxError;
use rayon::prelude::*;

use crate::image::Image;
use crate::prep::{self, Plane};

pub(crate) mod conv;
pub(crate) mod sample;

use sample::{bilinear1, bilinear2, cvtt};

/// A two channel f32 vector field, row major, interleaved `(dx, dy)` per pixel. Units are pixels
/// of the image the field belongs to.
#[derive(Clone, Debug, PartialEq)]
pub struct FlowField {
    pub width: usize,
    pub height: usize,
    /// `width * height * 2` values.
    pub data: Vec<f32>,
}

impl FlowField {
    /// A field of zero vectors.
    pub fn new(width: usize, height: usize) -> FlowField {
        FlowField { width, height, data: vec![0.0; width * height * 2] }
    }

    /// The vector at pixel `(x, y)`.
    pub fn at(&self, x: usize, y: usize) -> [f32; 2] {
        let i = (y * self.width + x) * 2;
        [self.data[i], self.data[i + 1]]
    }

    pub fn set(&mut self, x: usize, y: usize, v: [f32; 2]) {
        let i = (y * self.width + x) * 2;
        self.data[i] = v[0];
        self.data[i + 1] = v[1];
    }

    /// Bilinear sample of both channels in single precision, for the warps that resample the
    /// flow at `(x * rx, y * ry)`. Callers must pass non negative coordinates inside the field
    /// (otherwise the cell index is clamped).
    pub fn bilerp(&self, x: f32, y: f32) -> [f32; 2] {
        bilinear2(&self.data, self.width, self.height, x, y)
    }

    /// Multiplies every value by `f` in single precision.
    pub fn scale(&mut self, f: f32) {
        scale_values(&mut self.data, f);
    }
}

// ---------------------------------------------------------------------------------------
// Settings and geometry

/// Everything that reaches the estimator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlowSettings {
    /// Vectors quality: 1 Sloppy, 2 Medium, 3 High, 4 Best (half resolution), 5 full resolution.
    /// 0 means "no vectors" and is refused.
    pub quality: u8,
    /// True for a high quality render; draft quality halves the flow resolution once more.
    pub hiq: bool,
    /// The Sensitivity slider already divided by 100 (2.2).
    pub sensitivity: f32,
    /// Prep popup "Contrast/Edge Enhance" (2.3, 5.2).
    pub prep_contrast_edge: bool,
    /// Prep popup "Delinearize" (2.3, 5.1).
    pub prep_delinearize: bool,
    /// "Track RGB+A" of plugin 02: alpha is added to the luma (section 3). Always false in
    /// the retime path.
    pub track_alpha: bool,
    /// Use the video range luma conversion (see [`prep::luma`]). Default false.
    pub video_range_luma: bool,
    /// Skips the block matching refinement. The engine always passes false.
    pub skip_refine: bool,
}

impl FlowSettings {
    /// The retime path: no prep, no alpha, normal luma, block matching on.
    pub fn new(quality: u8, hiq: bool, sensitivity: f32) -> FlowSettings {
        FlowSettings {
            quality,
            hiq,
            sensitivity,
            prep_contrast_edge: false,
            prep_delinearize: false,
            track_alpha: false,
            video_range_luma: false,
            skip_refine: false,
        }
    }

    /// Working geometry for an output image of `out_width x out_height`.
    pub fn geometry(&self, out_width: usize, out_height: usize) -> Result<FlowGeometry, FxError> {
        if self.quality == 0 || self.quality > 5 {
            return Err(FxError::BadParam {
                id: "vectors_quality".into(),
                reason: format!("flow quality {} is outside 1..=5 (0 disables the estimator)", self.quality),
            });
        }
        if !(self.sensitivity >= 0.0) || !self.sensitivity.is_finite() {
            return Err(FxError::BadParam {
                id: "sensitivity".into(),
                reason: format!("sensitivity {} must be finite and not negative", self.sensitivity),
            });
        }
        let (reduction, w0, h0) = prep::flow_resolution(self.quality, self.hiq, out_width, out_height);
        if w0 < 2 || h0 < 2 {
            return Err(FxError::Unsupported(format!(
                "optical flow: working size {w0}x{h0} (output {out_width}x{out_height}, reduction {reduction}) is below 2x2, which is not supported"
            )));
        }
        let levels = prep::level_count(w0, h0, 2);
        let sizes = prep::level_sizes(w0, h0, levels);
        Ok(FlowGeometry { reduction, width: w0, height: h0, levels, sizes })
    }
}

impl Default for FlowSettings {
    /// Plugin 01 defaults: Best (4), high quality render, Sensitivity 70.
    fn default() -> FlowSettings {
        FlowSettings::new(4, true, 70.0f32 / 100.0f32)
    }
}

/// Resolution and pyramid layout of one solve.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlowGeometry {
    /// Number of halvings `R` of the output size.
    pub reduction: u32,
    /// Working size `(W >> R, H >> R)`, the size of the returned flow.
    pub width: usize,
    pub height: usize,
    /// Number of pyramid levels `L`.
    pub levels: usize,
    /// Size of every level, finest first.
    pub sizes: Vec<(usize, usize)>,
}

/// Start level and damping factor from the sensitivity: `x = log2(s * s) + L`; `x < 0` gives
/// `(0, 0.0)`, `x >= L` gives `(L - 1, 1.0)`, else `(floor(x), x - floor(x))`. `powf` and `logf`
/// are evaluated in double precision and rounded to single, which gives the same values as a
/// single precision evaluation for every slider value `k / 100`.
pub fn start_level(sensitivity: f32, levels: usize) -> (usize, f32) {
    let p = ((sensitivity as f64).powf(2.0)) as f32;
    let l1 = (p as f64).ln() as f32;
    let l2 = (2.0f64).ln() as f32;
    let lv = levels as f32;
    let x = l1 / l2 + lv;
    if 0.0 <= x || x.is_nan() {
        if x < lv || x.is_nan() {
            let f = x.floor();
            let s = cvtt(f);
            let s = if s < 0 { 0 } else { s as usize };
            (s.min(levels - 1), x - (s as f32))
        } else {
            (levels - 1, 1.0)
        }
    } else {
        (0, 0.0)
    }
}

// ---------------------------------------------------------------------------------------
// Pyramid

/// The Laplacian pyramid of one frame: the band pass images the solver works on
/// (ENGINE-motion section 6). Opaque; built by [`build_pyramid`].
#[derive(Clone, Debug)]
pub struct Pyramid {
    geometry: FlowGeometry,
    levels: Vec<Plane>,
}

impl Pyramid {
    pub fn geometry(&self) -> &FlowGeometry {
        &self.geometry
    }

    /// The band pass image of level `l` (0 is the finest). For tests and debugging.
    pub fn level(&self, l: usize) -> &Plane {
        &self.levels[l]
    }
}

/// `a - b` elementwise, in place on `a`.
fn subtract_in_place(a: &mut [f32], b: &[f32]) {
    a.par_iter_mut().zip(b.par_iter()).for_each(|(x, y)| *x = *x - *y);
}

/// Pyramid builder: luma, reduction to the working size, Delinearize then Contrast/Edge Enhance on
/// level 0 (when enabled), then for every level `band_i = orig_i - blur5(orig_i)` with
/// `orig_{i+1} = resample(blur5(orig_i))`.
pub fn build_pyramid(img: &Image, settings: &FlowSettings) -> Result<Pyramid, FxError> {
    let geometry = settings.geometry(img.width, img.height)?;
    let luma = prep::luma(img, settings.track_alpha, settings.video_range_luma)?;
    let mut level0 = prep::reduce(luma, geometry.reduction);
    debug_assert_eq!((level0.width, level0.height), (geometry.width, geometry.height));
    if settings.prep_delinearize {
        prep::delinearize(&mut level0);
    }
    if settings.prep_contrast_edge {
        prep::edge_enhance(&mut level0);
    }
    let n = geometry.levels;
    let mut orig = level0;
    let mut bands: Vec<Plane> = Vec::with_capacity(n);
    for i in 0..n {
        let mut tmp = orig.clone();
        prep::blur5(&mut tmp);
        let next = if i + 1 < n {
            let (nw, nh) = geometry.sizes[i + 1];
            Some(prep::resample(&tmp, nw, nh))
        } else {
            None
        };
        let mut band = orig;
        subtract_in_place(&mut band.data, &tmp.data);
        bands.push(band);
        match next {
            Some(p) => orig = p,
            None => break,
        }
    }
    Ok(Pyramid { geometry, levels: bands })
}

// ---------------------------------------------------------------------------------------
// Per level kernels (all single precision, ENGINE-motion section 7.1)

/// Gradient: interleaved `(gx, gy)`, one sided at the edges.
/// `gx = ((I(x,y) - I(x-1,y)) + 0.0 [x >= 1]) + (I(x+1,y) - I(x,y)) [x <= w-2]` divided
/// by the number of terms; the same for `gy` with the rows.
pub(crate) fn calc_grad(img: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; w * h * 2];
    if w == 0 || h == 0 {
        return g;
    }
    g.par_chunks_mut(w * 2).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let c = img[y * w + x];
            let mut nx = 0.0f32;
            let mut dx = 0.0f32;
            let mut ny = 0.0f32;
            let mut dy = 0.0f32;
            if x >= 1 {
                nx = (c - img[y * w + x - 1]) + 0.0f32;
                dx = 1.0;
            }
            if y >= 1 {
                ny = (c - img[(y - 1) * w + x]) + 0.0f32;
                dy = 1.0;
            }
            if x + 1 <= w - 1 {
                nx = nx + (img[y * w + x + 1] - c);
                dx = dx + 1.0;
            }
            if y + 1 <= h - 1 {
                ny = ny + (img[(y + 1) * w + x] - c);
                dy = dy + 1.0;
            }
            row[x * 2] = nx / dx;
            row[x * 2 + 1] = ny / dy;
        }
    });
    g
}

/// Structure tensor: `(gx*gx, gx*gy, gy*gy)`, 3 channels.
fn calc_a(grad: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; w * h * 3];
    a.par_chunks_mut(3).zip(grad.par_chunks(2)).for_each(|(o, g)| {
        let (gx, gy) = (g[0], g[1]);
        o[0] = gx * gx;
        o[1] = gx * gy;
        o[2] = gy * gy;
    });
    a
}

/// Right hand side: `I1 = imgA`, `I0 = imgB`, `d` the flow.
/// `(sx, sy) = (x - d.x, y - d.y)`; outside `[0, w-1] x [0, h-1]` the result is `(0, 0)`,
/// else `r = bilinear(I0, sx, sy) - I1(x, y)` and `B = (r * gx, r * gy)`. The pixel
/// coordinates are single precision values (exact for these sizes).
fn calc_b(i1: &[f32], i0: &[f32], grad: &[f32], flow: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut b = vec![0.0f32; w * h * 2];
    let fw = w as f32;
    let fh = h as f32;
    b.par_chunks_mut(w * 2).enumerate().for_each(|(y, row)| {
        let fy = y as f32;
        for x in 0..w {
            let fx = x as f32;
            let i = y * w + x;
            let sx = fx - flow[i * 2];
            let sy = fy - flow[i * 2 + 1];
            if sx < 0.0 || fw - 1.0 < sx || sy < 0.0 || fh - 1.0 < sy {
                row[x * 2] = 0.0;
                row[x * 2 + 1] = 0.0;
            } else {
                let s = bilinear1(i0, w, h, sx, sy);
                let r = s - i1[i];
                row[x * 2] = r * grad[i * 2];
                row[x * 2 + 1] = r * grad[i * 2 + 1];
            }
        }
    });
    b
}

/// New displacement: `t = b*b - dd*a`; if `|t| > 40.0`
/// `d.x = (b*by - dd*bx) / t + d.x`, `d.y = (b*bx - a*by) / t + d.y`.
fn new_disp(flow: &mut [f32], a: &[f32], b: &[f32]) {
    flow.par_chunks_mut(2).zip(a.par_chunks(3)).zip(b.par_chunks(2)).for_each(|((d, t3), bb)| {
        let (aa, ab, ad) = (t3[0], t3[1], t3[2]);
        let (bx, by) = (bb[0], bb[1]);
        let t = ab * ab - ad * aa;
        if t.abs() > 40.0f32 {
            d[0] = (ab * by - ad * bx) / t + d[0];
            d[1] = (ab * bx - aa * by) / t + d[1];
        }
    });
}

/// Upsampling of the flow to the next finer level: corner aligned,
/// `sx = ((f32) dst_w - 1.0) / (f32)(src_w - 1)`, `rx = (f32)(1.0 / (f64) sx)`, the source
/// position is `(x * rx, y * ry)`, the sampled vector is multiplied by `(sx, sy)`.
fn upsample_flow(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let sx = (dw as f32 - 1.0f32) / ((sw as i32 - 1) as f32);
    let sy = (dh as f32 - 1.0f32) / ((sh as i32 - 1) as f32);
    let rx = (1.0f64 / (sx as f64)) as f32;
    let ry = (1.0f64 / (sy as f64)) as f32;
    let mut dst = vec![0.0f32; dw * dh * 2];
    dst.par_chunks_mut(dw * 2).enumerate().for_each(|(y, row)| {
        let fy = (y as i32 as f32) * ry;
        for x in 0..dw {
            let v = bilinear2(src, sw, sh, (x as i32 as f32) * rx, fy);
            row[x * 2] = v[0] * sx;
            row[x * 2 + 1] = v[1] * sy;
        }
    });
    dst
}

/// `p = f * p` for every value; `f == 1.0` leaves the data untouched.
fn scale_values(data: &mut [f32], f: f32) {
    if f == 1.0 {
        return;
    }
    data.par_iter_mut().for_each(|v| *v = f * *v);
}

/// True when no pixel satisfies `0 <= x + u <= w-1 && 0 <= y + v <= h-1` (note the plus sign).
fn all_flow_out_of_range(flow: &[f32], w: usize, h: usize) -> bool {
    let fw = (w - 1) as f32;
    let fh = (h - 1) as f32;
    let any = flow.par_chunks(w * 2).enumerate().any(|(y, row)| {
        let fy = y as f32;
        (0..w).any(|x| {
            let px = (x as i32 as f32) + row[x * 2];
            let py = fy + row[x * 2 + 1];
            0.0 <= px && px <= fw && 0.0 <= py && py <= fh
        })
    });
    !any
}

// ---------------------------------------------------------------------------------------
// Block matching (ENGINE-motion section 8)

struct Block<'a> {
    a: &'a [f32],
    b: &'a [f32],
    w: usize,
    h: usize,
}

impl<'a> Block<'a> {
    /// The 3x3 patch of imgA around `(x, y)` in row major order: rows `y-1, y, y+1`, columns
    /// `x-1, x, x+1`.
    #[inline]
    fn patch(&self, x: usize, y: usize) -> [f32; 9] {
        let w = self.w;
        let i = (y - 1) * w + x - 1;
        [
            self.a[i],
            self.a[i + 1],
            self.a[i + 2],
            self.a[i + w],
            self.a[i + w + 1],
            self.a[i + w + 2],
            self.a[i + 2 * w],
            self.a[i + 2 * w + 1],
            self.a[i + 2 * w + 2],
        ]
    }

    /// Block cost: `None` when the target position is not inside `[1, w-2] x [1, h-2]`, else the
    /// sum of squared differences between the patch and the bilinear interpolation of imgB cells
    /// around `(x2, y2)`.
    #[inline]
    fn cost(&self, p: &[f32; 9], x2: f32, y2: f32) -> Option<f32> {
        let wm2 = (self.w as i32 - 2) as f32;
        let hm2 = (self.h as i32 - 2) as f32;
        if x2 > wm2 || y2 > hm2 {
            return None;
        }
        let (ix, fx) = if x2 == wm2 {
            (cvtt(wm2 - 1.0f32), 1.0f32)
        } else {
            let i = cvtt(x2);
            (i, x2 - (i as f32))
        };
        let (iy, fy) = if y2 == hm2 {
            (cvtt(hm2 - 1.0f32), 1.0f32)
        } else {
            let i = cvtt(y2);
            (i, y2 - (i as f32))
        };
        if ix < 1 || iy < 1 {
            return None;
        }
        let (ix, iy) = (ix as usize, iy as usize);
        let w = self.w;
        let w11 = fy * fx;
        let w10 = fx - w11;
        let w01 = fy - w11;
        let w00 = ((1.0f32 - fx) - fy) + w11;
        // T(i, j) = imgB(ix + i, iy + j); the 4x4 block starts at (ix - 1, iy - 1)
        let base = (iy - 1) * w + ix - 1;
        let t = |i: usize, j: usize| self.b[base + j * w + i]; // i, j in 0..4 (offset by 1)
        let mut sq = [0.0f32; 9];
        let mut n = 0;
        for b in 0..3usize {
            for a in 0..3usize {
                let q = ((w00 * t(a, b) + w10 * t(a + 1, b)) + w01 * t(a, b + 1)) + w11 * t(a + 1, b + 1);
                let d = p[n] - q;
                sq[n] = d * d;
                n += 1;
            }
        }
        // the first eight squares are accumulated in order, the ninth is added last
        let mut s = sq[0] + sq[1];
        for k in 2..8 {
            s = s + sq[k];
        }
        Some(sq[8] + s)
    }

    /// One pixel step of a sweep: `flow(x, y)` is replaced by the neighbour's vector `n`
    /// when the cost at the neighbour's target is lower.
    #[inline]
    fn step(&self, x: usize, y: usize, cur: [f32; 2], n: [f32; 2]) -> Option<[f32; 2]> {
        let tx = x as f32 - cur[0];
        let ty = y as f32 - cur[1];
        let w1 = (self.w - 1) as f32;
        let h1 = (self.h - 1) as f32;
        if 0.0 <= tx && tx <= w1 && 0.0 <= ty && ty <= h1 {
            let patch = self.patch(x, y);
            if let Some(c0) = self.cost(&patch, tx, ty) {
                if let Some(c1) = self.cost(&patch, x as f32 - n[0], y as f32 - n[1]) {
                    if c1 < c0 {
                        return Some(n);
                    }
                }
            }
        }
        None
    }
}

/// The four sweeps of one level: left to right, right to left, top to bottom, bottom to top over
/// the interior pixels, in place, each pixel compared with the already updated neighbour behind
/// it. Rows (sweeps 1 and 2) and columns (3 and 4) are independent.
fn block_match(a: &[f32], b: &[f32], flow: &mut [f32], w: usize, h: usize) {
    if w < 3 || h < 3 {
        return;
    }
    let bm = Block { a, b, w, h };
    // sweep 1: x = 1..=w-2 increasing, neighbour (x-1, y)
    flow.par_chunks_mut(w * 2).enumerate().for_each(|(y, row)| {
        if y >= 1 && y <= h - 2 {
            for x in 1..=(w - 2) {
                let cur = [row[x * 2], row[x * 2 + 1]];
                let n = [row[(x - 1) * 2], row[(x - 1) * 2 + 1]];
                if let Some(v) = bm.step(x, y, cur, n) {
                    row[x * 2] = v[0];
                    row[x * 2 + 1] = v[1];
                }
            }
        }
    });
    // sweep 2: x = w-2 down to 1, neighbour (x+1, y)
    flow.par_chunks_mut(w * 2).enumerate().for_each(|(y, row)| {
        if y >= 1 && y <= h - 2 {
            for x in (1..=(w - 2)).rev() {
                let cur = [row[x * 2], row[x * 2 + 1]];
                let n = [row[(x + 1) * 2], row[(x + 1) * 2 + 1]];
                if let Some(v) = bm.step(x, y, cur, n) {
                    row[x * 2] = v[0];
                    row[x * 2 + 1] = v[1];
                }
            }
        }
    });
    // sweeps 3 and 4 run on the transposed field so that a column is contiguous
    let mut ft = vec![0.0f32; flow.len()];
    for y in 0..h {
        for x in 0..w {
            ft[(x * h + y) * 2] = flow[(y * w + x) * 2];
            ft[(x * h + y) * 2 + 1] = flow[(y * w + x) * 2 + 1];
        }
    }
    // sweep 3: y = 1..=h-2 increasing, neighbour (x, y-1)
    ft.par_chunks_mut(h * 2).enumerate().for_each(|(x, col)| {
        if x >= 1 && x <= w - 2 {
            for y in 1..=(h - 2) {
                let cur = [col[y * 2], col[y * 2 + 1]];
                let n = [col[(y - 1) * 2], col[(y - 1) * 2 + 1]];
                if let Some(v) = bm.step(x, y, cur, n) {
                    col[y * 2] = v[0];
                    col[y * 2 + 1] = v[1];
                }
            }
        }
    });
    // sweep 4: y = h-2 down to 1, neighbour (x, y+1)
    ft.par_chunks_mut(h * 2).enumerate().for_each(|(x, col)| {
        if x >= 1 && x <= w - 2 {
            for y in (1..=(h - 2)).rev() {
                let cur = [col[y * 2], col[y * 2 + 1]];
                let n = [col[(y + 1) * 2], col[(y + 1) * 2 + 1]];
                if let Some(v) = bm.step(x, y, cur, n) {
                    col[y * 2] = v[0];
                    col[y * 2 + 1] = v[1];
                }
            }
        }
    });
    for y in 0..h {
        for x in 0..w {
            flow[(y * w + x) * 2] = ft[(x * h + y) * 2];
            flow[(y * w + x) * 2 + 1] = ft[(x * h + y) * 2 + 1];
        }
    }
}

// ---------------------------------------------------------------------------------------
// Driver

/// Solver driver (`X` is the pyramid `imgA`, `Y` the pyramid `imgB`): returns the flow of the
/// finest level after the final negation, `X(p) ~ Y(p + out(p))`.
///
/// Both pyramids must come from [`build_pyramid`] with settings that give the same
/// geometry; `settings` supplies the sensitivity and `skip_refine`.
pub fn solve_pyramids(x: &Pyramid, y: &Pyramid, settings: &FlowSettings) -> Result<FlowField, FxError> {
    if x.geometry != y.geometry {
        return Err(FxError::Mismatch("optical flow: pyramids of different geometry".into()));
    }
    let geo = &x.geometry;
    let levels = geo.levels;
    let (start, damp) = start_level(settings.sensitivity, levels);
    let (sw0, sh0) = geo.sizes[start];
    let mut flow = vec![0.0f32; sw0 * sh0 * 2];
    let mut check = true;
    for l in (0..=start).rev() {
        let (w, h) = geo.sizes[l];
        let img_a = &x.levels[l].data;
        let img_b = &y.levels[l].data;
        let grad = calc_grad(img_a, w, h);
        let mut a = calc_a(&grad, w, h);
        conv::blur_gaussian(&mut a, w, h, 3, 7.0);
        for it in 0..3 {
            let mut b = calc_b(img_a, img_b, &grad, &flow, w, h);
            conv::blur_gaussian(&mut b, w, h, 2, 7.0);
            new_disp(&mut flow, &a, &b);
            conv::blur_gaussian(&mut flow, w, h, 2, 5.0);
            if it == 1 && l == 0 && !settings.skip_refine {
                block_match(img_a, img_b, &mut flow, w, h);
            }
        }
        if l != 0 && !settings.skip_refine {
            block_match(img_a, img_b, &mut flow, w, h);
        }
        if check {
            if all_flow_out_of_range(&flow, w, h) {
                flow.iter_mut().for_each(|v| *v = 0.0);
            } else {
                check = false;
            }
        }
        if l == start && (damp as f64 - 1.0).abs() >= 1e-6 {
            scale_values(&mut flow, damp);
        }
        if l > 0 {
            let (nw, nh) = geo.sizes[l - 1];
            flow = upsample_flow(&flow, w, h, nw, nh);
        }
    }
    scale_values(&mut flow, -1.0f32);
    Ok(FlowField { width: geo.width, height: geo.height, data: flow })
}

fn check_inputs(x: &Image, y: &Image) -> Result<(), FxError> {
    if x.width != y.width || x.height != y.height || x.fmt() != y.fmt() {
        return Err(FxError::Mismatch(format!(
            "optical flow: incompatible inputs ({}x{} {:?} against {}x{} {:?})",
            x.width,
            x.height,
            x.fmt(),
            y.width,
            y.height,
            y.fmt()
        )));
    }
    Ok(())
}

/// The flow computation: both pyramids are built (`X` -> `imgA`, `Y` -> `imgB`) and solved. The
/// result has the solver's convention `X(p) ~ Y(p + out(p))`, is defined on the grid of `X` and has
/// the working resolution of `FlowSettings::geometry`. `x` and `y` are premultiplied engine images
/// of equal size and format.
pub fn solve(x: &Image, y: &Image, settings: &FlowSettings) -> Result<FlowField, FxError> {
    check_inputs(x, y)?;
    let (px, py) = rayon::join(|| build_pyramid(x, settings), || build_pyramid(y, settings));
    solve_pyramids(&px?, &py?, settings)
}

/// The two flows of a blend: `f_ba` is the solve with `X = B`, `Y = A`, `f_ab` the solve with
/// `X = A`, `Y = B`.
#[derive(Clone, Debug, PartialEq)]
pub struct FlowPair {
    /// `A(x) ~ B(x + f_ab(x))`, on the grid of A.
    pub f_ab: FlowField,
    /// `B(x) ~ A(x + f_ba(x))`, on the grid of B.
    pub f_ba: FlowField,
}

/// Both flows of a Blend / Weighted Blend render: the same values as `solve(B, A)` and
/// `solve(A, B)` (the pyramid of a frame does not depend on its role), with the pyramids built
/// once and the two solves run in parallel.
pub fn solve_pair(a: &Image, b: &Image, settings: &FlowSettings) -> Result<FlowPair, FxError> {
    check_inputs(a, b)?;
    let (pa, pb) = rayon::join(|| build_pyramid(a, settings), || build_pyramid(b, settings));
    let (pa, pb) = (pa?, pb?);
    let (f_ba, f_ab) = rayon::join(|| solve_pyramids(&pb, &pa, settings), || solve_pyramids(&pa, &pb, settings));
    Ok(FlowPair { f_ab: f_ab?, f_ba: f_ba? })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_start_at_zero_and_store_vectors() {
        let mut f = FlowField::new(3, 2);
        assert_eq!(f.data.len(), 12);
        assert_eq!(f.at(2, 1), [0.0, 0.0]);
        f.set(2, 1, [1.5, -2.0]);
        assert_eq!(f.at(2, 1), [1.5, -2.0]);
        assert_eq!(f.at(1, 1), [0.0, 0.0]);
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn field_sampler_and_scale_helpers() {
        let mut f = FlowField::new(3, 2);
        f.set(0, 0, [1.0, 10.0]);
        f.set(1, 0, [3.0, 30.0]);
        f.set(0, 1, [5.0, 50.0]);
        f.set(1, 1, [7.0, 70.0]);
        assert_eq!(f.bilerp(0.0, 0.0), [1.0, 10.0]);
        assert_eq!(f.bilerp(0.5, 0.0), [2.0, 20.0]);
        assert_eq!(f.bilerp(0.5, 0.5), [4.0, 40.0]);
        f.scale(-2.0);
        assert_eq!(f.at(1, 1), [-14.0, -140.0]);
    }

    #[test]
    fn gradient_of_ramps_and_parabola() {
        // ramp 3x + 5y on 6x5: one sided differences at the edges equal the slope
        let (w, h) = (6, 5);
        let img: Vec<f32> = (0..h).flat_map(|y| (0..w).map(move |x| 3.0 * x as f32 + 5.0 * y as f32)).collect();
        let g = calc_grad(&img, w, h);
        for i in 0..w * h {
            assert_eq!((g[i * 2], g[i * 2 + 1]), (3.0, 5.0));
        }
        // parabola x*x along x: gx = 1 at x = 0, 2x inside, 2x - 1 at x = w - 1
        let img: Vec<f32> = (0..h).flat_map(|_| (0..w).map(|x| (x * x) as f32)).collect();
        let g = calc_grad(&img, w, h);
        assert_eq!(g[0], 1.0);
        assert_eq!(g[2 * 3], 6.0);
        assert_eq!(g[2 * (w - 1)], (2 * (w - 1) - 1) as f32);
        // gy of a constant column is zero (the adds of 0.0 are kept)
        assert_eq!(g[1], 0.0);
    }

    #[test]
    fn tensor_and_right_hand_side_of_a_tiny_patch() {
        let g = [2.0f32, -3.0, 0.5, 4.0];
        let a = calc_a(&g, 2, 1);
        assert_eq!(a, vec![4.0, -6.0, 9.0, 0.25, 2.0, 16.0]);
        // calc_b on 3x3 with a zero flow: B = (I0 - I1) * grad
        let (w, h) = (3, 3);
        let i1: Vec<f32> = (0..9).map(|v| v as f32).collect();
        let i0: Vec<f32> = (0..9).map(|v| (v * 2) as f32).collect();
        let grad = vec![1.0f32; 18];
        let flow = vec![0.0f32; 18];
        let b = calc_b(&i1, &i0, &grad, &flow, w, h);
        for p in 0..9 {
            assert_eq!(b[p * 2], p as f32, "I0 - I1 = v");
            assert_eq!(b[p * 2 + 1], p as f32);
        }
        // a flow of (1, 0): pixel x = 0 samples x - 1 = -1 and is zeroed; pixel (1, 1) samples
        // I0(0, 1) = 6
        let flow = [1.0f32, 0.0].repeat(9);
        let b = calc_b(&i1, &i0, &grad, &flow, w, h);
        assert_eq!((b[0], b[1]), (0.0, 0.0));
        assert_eq!(b[(3 + 1) * 2], 6.0 - 4.0, "I0(0,1) - I1(1,1)");
        // a flow beyond the edge: sx = x - d.x = 2 + 2.5 > w - 1 is zeroed
        let flow = [-2.5f32, 0.0].repeat(9);
        let b = calc_b(&i1, &i0, &grad, &flow, w, h);
        assert_eq!((b[2 * 2], b[2 * 2 + 1]), (0.0, 0.0), "sx = 4.5 > w - 1");
    }

    #[test]
    fn new_displacement_update_and_threshold() {
        // A = (a, b, dd) = (100, 0, 100): t = -10000, d += (b*by - dd*bx)/t, (b*bx - a*by)/t
        let mut flow = vec![1.0f32, 2.0];
        new_disp(&mut flow, &[100.0, 0.0, 100.0], &[50.0, -20.0]);
        assert_eq!(flow, vec![1.5, 1.8]);
        // |t| = 39 does not exceed the epsilon 40.0: unchanged
        let mut flow = vec![1.0f32, 2.0];
        new_disp(&mut flow, &[6.0, 0.0, 6.5], &[50.0, -20.0]); // t = -39
        assert_eq!(flow, vec![1.0, 2.0]);
        let mut flow = vec![0.0f32, 0.0];
        new_disp(&mut flow, &[8.0, 3.0, 8.0], &[1.0, 1.0]); // t = 9 - 64 = -55
        assert_eq!(flow[0], (3.0f32 * 1.0 - 8.0 * 1.0) / -55.0);
        assert_eq!(flow[1], (3.0f32 * 1.0 - 8.0 * 1.0) / -55.0);
    }

    #[test]
    fn flow_upsampling_scales_the_vectors_by_the_size_ratio() {
        // constant flow (1, 2) on 2x2 -> 3x3: sx = (3 - 1) / (2 - 1) = 2, sy = 2
        let src = [1.0f32, 2.0].repeat(4);
        let d = upsample_flow(&src, 2, 2, 3, 3);
        for p in 0..9 {
            assert_eq!((d[p * 2], d[p * 2 + 1]), (2.0, 4.0));
        }
        // a horizontal ramp in u on 3x2 -> 5x2: sx = 4 / 2 = 2, positions x * 0.5
        let src = [0.0f32, 0.0, 2.0, 0.0, 4.0, 0.0].repeat(2);
        let d = upsample_flow(&src, 3, 2, 5, 2);
        for x in 0..5 {
            assert_eq!(d[x * 2], 2.0 * x as f32);
        }
    }

    #[test]
    fn range_check_and_scaling() {
        // all vectors point out of the image: true; one in range: false
        let (w, h) = (4, 4);
        let out = [10.0f32, 10.0].repeat(16);
        assert!(all_flow_out_of_range(&out, w, h));
        let mut ok = out.clone();
        ok[5 * 2] = -1.0; // pixel (1, 1): x + u = 0 but y + v = 11, still out
        assert!(all_flow_out_of_range(&ok, w, h));
        ok[0] = 0.0;
        ok[1] = 0.0; // pixel (0, 0) stays in place
        assert!(!all_flow_out_of_range(&ok, w, h));
        // scaling by 1.0 leaves the data alone, by -1.0 flips the sign of zero as well
        let mut v = vec![0.0f32, 1.5, -2.0];
        scale_values(&mut v, 1.0);
        assert_eq!(bits(&v), bits(&[0.0, 1.5, -2.0]));
        scale_values(&mut v, -1.0);
        assert_eq!(bits(&v), bits(&[-0.0, -1.5, 2.0]));
    }

    fn block_image(w: usize, h: usize, f: impl Fn(usize, usize) -> f32) -> Vec<f32> {
        (0..h).flat_map(|y| (0..w).map(|x| f(x, y)).collect::<Vec<_>>()).collect()
    }

    #[test]
    fn block_cost_golden_values() {
        let (w, h) = (8, 8);
        let zeros = vec![0.0f32; w * h];
        let a = zeros.clone();
        let bm = Block { a: &a, b: &zeros, w, h };
        // integer position: q = T, d = p
        let p = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq!(bm.cost(&p, 3.0, 3.0), Some(285.0));
        // positions outside [1, w-2]: None
        assert_eq!(bm.cost(&p, 0.5, 3.0), None, "ix = 0");
        assert_eq!(bm.cost(&p, 3.0, 0.0), None);
        assert_eq!(bm.cost(&p, 6.01, 3.0), None, "x2 > w - 2");
        assert!(bm.cost(&p, 6.0, 6.0).is_some(), "x2 == w - 2 takes the cell ix = w - 3 with fx = 1");
        // the ninth square is added last: d9 = 4096, the others +-1
        let mut p = [1.0f32; 9];
        p[8] = 4096.0;
        assert_eq!(bm.cost(&p, 3.0, 3.0), Some(16777224.0));
        // bilinear weights: B = x + 10 y, position (2.5, 3.25) gives q(a,b) = (2.5 + a) + 10 (3.25 + b)
        let b = block_image(w, h, |x, y| x as f32 + 10.0 * y as f32);
        let bm = Block { a: &a, b: &b, w, h };
        let mut p = [0.0f32; 9];
        let mut n = 0;
        for bb in -1i32..=1 {
            for aa in -1i32..=1 {
                p[n] = (2.5 + aa as f32) + 10.0 * (3.25 + bb as f32);
                n += 1;
            }
        }
        assert_eq!(bm.cost(&p, 2.5, 3.25), Some(0.0));
        p[4] += 2.0;
        assert_eq!(bm.cost(&p, 2.5, 3.25), Some(4.0));
    }

    #[test]
    fn block_matching_replaces_a_bad_vector_by_the_better_neighbour_and_spares_the_border() {
        let (w, h) = (12, 10);
        let img = block_image(w, h, |x, y| ((x * 7 + y * 13) % 17) as f32 * 3.0 + (x as f32) * 0.5);
        let mut flow = vec![0.0f32; w * h * 2];
        // a wrong vector in the middle, nonzero vectors on the first and last row
        let c = (5 * w + 6) * 2;
        flow[c] = 2.0;
        flow[c + 1] = 1.0;
        for x in 0..w {
            flow[x * 2] = 0.25;
            flow[((h - 1) * w + x) * 2] = 0.25;
        }
        block_match(&img, &img, &mut flow, w, h);
        assert_eq!((flow[c], flow[c + 1]), (0.0, 0.0), "the correct zero vector of the neighbour wins");
        for x in 0..w {
            assert_eq!(flow[x * 2], 0.25, "first row never changes");
            assert_eq!(flow[((h - 1) * w + x) * 2], 0.25, "last row never changes");
        }
        // thread count independence of the sweeps on a noisy field
        let mut f1 = (0..w * h * 2).map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.3).collect::<Vec<_>>();
        let mut f2 = f1.clone();
        let pool = |n| rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
        pool(1).install(|| block_match(&img, &img, &mut f1, w, h));
        pool(4).install(|| block_match(&img, &img, &mut f2, w, h));
        assert_eq!(bits(&f1), bits(&f2));
    }

    #[test]
    fn start_level_table() {
        // default sensitivity 0.7: start L-2, damping 0.9709 (ENGINE-motion 2.2)
        let (s, d) = start_level(0.7, 8);
        assert_eq!(s, 6);
        assert!((d - 0.9709).abs() < 1e-4, "{d}");
        assert_eq!(start_level(1.0, 8), (7, 1.0));
        assert_eq!(start_level(2.0, 8), (7, 1.0));
        assert_eq!(start_level(0.0, 8), (0, 0.0));
        // very low sensitivity on a short pyramid: x < 0 gives start 0, damping 0
        assert_eq!(start_level(0.1, 3), (0, 0.0));
        // 0.5: log2(0.25) = -2, x = L - 2 exactly: start L-2, damping 0
        assert_eq!(start_level(0.5, 8), (6, 0.0));
    }
}
