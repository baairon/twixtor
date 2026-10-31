//! Image preparation for the motion estimator: colour to luma, the quality to resolution mapping,
//! the reduction to the working resolution, and the two optional transforms, Delinearize and
//! Contrast / Edge Enhance.
//!
//! The engine images arriving here are PREMULTIPLIED A, R, G, B images (8 bpc: 0..255, 16 bpc:
//! 0..65535, float: white 1.0). The output of the luma stage is a single channel float image in
//! the range 0..255 (`Plane`).
//!
//! The `_with` functions ([`blur5_with`], [`resample_with`], [`reduce_with`],
//! [`edge_enhance_with`]) take the solver's wrap-around (360) flag, `FlowSettings::wrap` (rule G6,
//! U: the 360 variants of the pyramid filters are not established; this project states that
//! under the flag every horizontal blur and resample that
//! builds the solver's input is circular). With the flag off each one IS the plain function (it
//! calls it), so the plain path keeps its bytes. With the flag on the horizontal passes are the
//! `_wrapx` primitives of `flow::conv` and `flow::sample` (tap indices modulo the width, no border
//! renormalisation; the horizontal resample keeps the period, `sx = sw / dw`, the right neighbour
//! of the last column is column 0) and the vertical passes are the solver's (border rows above
//! first, `conv::convolve_above_first`). Precision: those
//! primitives' (single precision in the order written, no fused multiply add, rows through rayon
//! row chunks, so the output does not depend on the thread count). When every halving of a frame
//! has an even width, a circular rotation of the input by a multiple of `2^r` columns rotates the
//! output of [`reduce_with`] by that multiple divided by `2^r`, bit for bit (every value is computed
//! from the same neighbours by the same chain).

use fx_core::FxError;
use rayon::prelude::*;

use crate::flow::{conv, sample};
use crate::image::{Image, Samples};

/// A single channel float image, row major, top row first, with luma values in 0..255.
#[derive(Clone, Debug, PartialEq)]
pub struct Plane {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl Plane {
    pub fn new(width: usize, height: usize) -> Plane {
        Plane { width, height, data: vec![0.0; width * height] }
    }

    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }
}

/// Quality to resolution mapping: returns the number of halvings `R` and the working size
/// `(W >> R, H >> R)` obtained by `R` truncating halvings.
///
/// `q` is the vectors quality: 1 Sloppy, 2 Medium, 3 High, 4 Best (half resolution), 5 full
/// resolution; 0 means "no vectors" and never reaches the solver. `hiq` is true for a high quality
/// render; for lower quality one more halving is applied (R 0 -> 1, 1 -> 2, 2 -> 3, 3 stays).
/// Values of `q` above 5 are rejected by [`crate::flow::FlowSettings::geometry`].
///
/// Tiny frames (U, ours, parity gap G7: the original divides by zero in its resample below 2x2):
/// the reduction never halves below 2x2. While `R > 0` and `(W >> R, H >> R)` has a side below
/// 2, `R` drops by one, so a frame too small for its halvings is searched at the last size of at
/// least 2x2 (6x6 at High, `q` 3: 3x3, not 1x1). Every size whose working size was at least 2x2
/// keeps its `R`. A frame with a side below 2 ends at `R = 0` with that side; it is not
/// [`searchable`] and never reaches the solver.
pub fn flow_resolution(q: u8, hiq: bool, w: usize, h: usize) -> (u32, usize, usize) {
    let mut r: u32 = match q {
        0 | 1 | 2 => 3,
        3 => 2,
        4 => 1,
        _ => 0,
    };
    if !hiq {
        r = match r {
            0 => 1,
            1 => 2,
            2 => 3,
            x => x,
        };
    }
    while r > 0 && ((w >> r) < 2 || (h >> r) < 2) {
        r -= 1;
    }
    let (mut w0, mut h0) = (w, h);
    for _ in 0..r {
        w0 /= 2;
        h0 /= 2;
    }
    (r, w0, h0)
}

/// Whether a frame of `w x h` pixels has a motion search: both sides are at least 2 (U, ours, parity
/// gap G7). With the reduction of [`flow_resolution`] a searchable frame always has a working size
/// of at least 2x2.
///
/// Contract for every caller: the optical flow solver, the layered (FG matte)
/// paths and the Debug views run only when `searchable(width, height)` holds for the output
/// frame. A frame that is not searchable (a side of 1, or an empty side) takes the fast path,
/// exactly as `Vectors` = No Vectors (`vectors=1`): Nearest copies, Blend and Weighted Blend
/// crossfade the two frames; Create Vectors writes the zero vector (127, 32767 or 0.5 in R and G,
/// B 0, alpha opaque). The front-ends apply it by passing a vectors quality of 0
/// ([`searched_quality`]).
pub fn searchable(w: usize, h: usize) -> bool {
    w >= 2 && h >= 2
}

/// The vectors quality a front-end passes to the planner and the core for an output frame of
/// `w x h`: `vectors_quality` itself when the frame is [`searchable`], else 0 (No Vectors).
pub fn searched_quality(vectors_quality: i32, w: usize, h: usize) -> i32 {
    if searchable(w, h) {
        vectors_quality
    } else {
        0
    }
}

/// Number of pyramid levels for the working size `(w0, h0)`; `min_size` is 2 in every caller.
pub fn level_count(w0: usize, h0: usize, min_size: usize) -> usize {
    let mut levels = 1;
    let (mut w, mut h) = (w0, h0);
    loop {
        w /= 2;
        if w < min_size {
            break;
        }
        h /= 2;
        if h < min_size {
            break;
        }
        if w > 2 && h > 2 && !(w == 2 && h == 2) {
            levels += 1;
        } else {
            break;
        }
    }
    levels
}

/// Sizes of the pyramid levels: `size[0] = (W0, H0)`, `size[i+1] = (max(2, w/2), max(2, h/2))`.
pub fn level_sizes(w0: usize, h0: usize, levels: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::with_capacity(levels);
    let (mut w, mut h) = (w0, h0);
    for _ in 0..levels {
        v.push((w, h));
        w = (w / 2).max(2);
        h = (h / 2).max(2);
    }
    v
}

// Luma constants. 8 bit: single precision. 16 bit and float: DOUBLE precision, with the weights
// 0.59, 0.3 and 0.11, then a division by 257.0 or a multiplication by 255.0, narrowed to single
// precision.
const W30F: f32 = 0.3;
const W59F: f32 = 0.59;
const W11F: f32 = 0.11;
const W30D: f64 = 0.3;
const W59D: f64 = 0.59;
const W11D: f64 = 0.11;

/// Colour to luma. Output values are in 0..255.
///
/// - 8 bit: `(R*0.3 + G*0.59) + B*0.11` in f32; with `track_alpha` the sum is widened and
///   `A*0.3` is added in f64 before narrowing.
/// - 16 bit: `(((G*0.59 + R*0.3) + B*0.11) [+ A*0.3]) / 257.0` in f64, narrowed to f32.
/// - float (white 1.0): `(((G*0.59 + R*0.3) + B*0.11) [+ A*0.3]) * 255.0` in f64, narrowed to f32.
/// - `video_range`: only the R channel is used, `((R - 16.0) * 255.0) / 219.0` (8 bit),
///   `((R / 257.0 - 16.0) * 255.0) / 219.0` (16 bit), `((R * 255.0 - 16.0) * 255.0) / 219.0`
///   (float), all f32. `video_range` together with `track_alpha` is not supported and returns
///   `FxError::Unsupported`.
pub fn luma(img: &Image, track_alpha: bool, video_range: bool) -> Result<Plane, FxError> {
    if video_range && track_alpha {
        return Err(FxError::Unsupported(
            "optical flow luma: video range conversion together with Track RGB+A is not supported".into(),
        ));
    }
    let (w, h) = (img.width, img.height);
    let mut out = Plane::new(w, h);
    if w == 0 || h == 0 {
        return Ok(out);
    }
    let row = w * 4;
    match &img.samples {
        Samples::U8(s) => {
            out.data.par_chunks_mut(w).zip(s.par_chunks(row)).for_each(|(o, p)| {
                for (x, v) in o.iter_mut().enumerate() {
                    let px = &p[x * 4..x * 4 + 4];
                    let (a, r, g, b) = (px[0] as f32, px[1] as f32, px[2] as f32, px[3] as f32);
                    *v = if video_range {
                        ((r - 16.0f32) * 255.0f32) / 219.0f32
                    } else {
                        let sum = (r * W30F + g * W59F) + b * W11F;
                        if track_alpha {
                            ((sum as f64) + (a as f64) * W30D) as f32
                        } else {
                            sum
                        }
                    };
                }
            });
        }
        Samples::U16(s) => {
            out.data.par_chunks_mut(w).zip(s.par_chunks(row)).for_each(|(o, p)| {
                for (x, v) in o.iter_mut().enumerate() {
                    let px = &p[x * 4..x * 4 + 4];
                    let (a, r, g, b) = (px[0] as f64, px[1] as f64, px[2] as f64, px[3] as f64);
                    *v = if video_range {
                        (((px[1] as f32) / 257.0f32 - 16.0f32) * 255.0f32) / 219.0f32
                    } else {
                        let mut sum = (g * W59D + r * W30D) + b * W11D;
                        if track_alpha {
                            sum += a * W30D;
                        }
                        (sum / 257.0f64) as f32
                    };
                }
            });
        }
        Samples::F32(s) => {
            out.data.par_chunks_mut(w).zip(s.par_chunks(row)).for_each(|(o, p)| {
                for (x, v) in o.iter_mut().enumerate() {
                    let px = &p[x * 4..x * 4 + 4];
                    let (a, r, g, b) = (px[0] as f64, px[1] as f64, px[2] as f64, px[3] as f64);
                    *v = if video_range {
                        ((px[1] * 255.0f32 - 16.0f32) * 255.0f32) / 219.0f32
                    } else {
                        let mut sum = (g * W59D + r * W30D) + b * W11D;
                        if track_alpha {
                            sum += a * W30D;
                        }
                        (sum * 255.0f64) as f32
                    };
                }
            });
        }
    }
    Ok(out)
}

/// The 5 tap binomial blur `[1 4 6 4 1] / 16` of the pyramid code, its vertical border rows with
/// the tap above first (`conv::convolve_above_first`, MEASURED against the original); the reduction, the edge
/// enhancement and the pyramid
/// take it.
pub fn blur5(p: &mut Plane) {
    conv::blur5_above_first(&mut p.data, p.width, p.height);
}

/// Corner aligned bilinear resampling to `(dw, dh)`.
pub fn resample(src: &Plane, dw: usize, dh: usize) -> Plane {
    Plane { width: dw, height: dh, data: sample::resample1(&src.data, src.width, src.height, dw, dh) }
}

/// Reduction of the luma image to the working resolution: `r` stages of `blur5` followed by a
/// corner aligned resample to the halved size (`w / 2`, truncating).
pub fn reduce(mut t: Plane, r: u32) -> Plane {
    for _ in 0..r {
        blur5(&mut t);
        let (nw, nh) = (t.width / 2, t.height / 2);
        t = resample(&t, nw, nh);
    }
    t
}

/// [`blur5`], or with `wrap` its horizontal wrap-around variant (G6, U; module header): the
/// circular horizontal pass of `conv::blur5_wrapx_above_first`, then the vertical pass with the
/// border rows above first. Without `wrap`
/// it is [`blur5`].
pub fn blur5_with(p: &mut Plane, wrap: bool) {
    if !wrap {
        blur5(p);
        return;
    }
    conv::blur5_wrapx_above_first(&mut p.data, p.width, p.height);
}

/// [`resample`], or with `wrap` the resample of a plane that is periodic in x (G6, U; module
/// header): `sample::resample1_wrapx`, `sx = sw / dw` with column 0 right of the last column, the
/// rows corner aligned as in [`resample`]. Without `wrap` it is [`resample`].
pub fn resample_with(src: &Plane, dw: usize, dh: usize, wrap: bool) -> Plane {
    if !wrap {
        return resample(src, dw, dh);
    }
    Plane { width: dw, height: dh, data: sample::resample1_wrapx(&src.data, src.width, src.height, dw, dh) }
}

/// [`reduce`], or with `wrap` the reduction of a frame that is periodic in x (G6, U): `r` stages
/// of [`blur5_with`] followed by [`resample_with`] to the halved size (`w / 2`, `h / 2`,
/// truncating, the sizes of [`reduce`]), both with the flag. Without `wrap` it is [`reduce`].
/// With an even width at every stage the horizontal resample reads the columns `2 x` exactly
/// (`sx = 2`, fraction 0); an odd width keeps the period with `sx = w / (w / 2)`.
pub fn reduce_with(t: Plane, r: u32, wrap: bool) -> Plane {
    if !wrap {
        return reduce(t, r);
    }
    let mut t = t;
    for _ in 0..r {
        blur5_with(&mut t, true);
        let (nw, nh) = (t.width / 2, t.height / 2);
        t = resample_with(&t, nw, nh, true);
    }
    t
}

/// Delinearize: `p = (max(p, 0) / 255) ^ (1 / 2.2) * 255` in single precision with
/// `gamma = 2.2f`. The power is the `powf` of the plugin's runtime (`crt_powf::powf`), not the correctly
/// rounded one: they differ by one unit in the last place for rare arguments.
pub fn delinearize(p: &mut Plane) {
    let gamma: f32 = 2.2;
    let inv = 1.0f32 / gamma;
    p.data.par_iter_mut().for_each(|v| {
        if *v <= 0.0 && *v != 0.0 {
            *v = 0.0;
        }
        let base = *v / 255.0f32;
        *v = crate::crt_powf::powf(base, inv) * 255.0f32;
    });
}

/// Contrast / Edge Enhance: the image is blurred in place with the Gaussian of radius 2.0, its
/// gradient is taken (the solver's `calc_grad`), and per pixel `m = sqrt(gx*gx + gy*gy)`,
/// `v = m <= 20.0 ? m * 0.05 : 1.0`, `p = (v * 255.0 + p) * 0.5` (single precision).
pub fn edge_enhance(p: &mut Plane) {
    let (w, h) = (p.width, p.height);
    conv::blur_gaussian_above_first(&mut p.data, w, h, 1, 2.0);
    let g = crate::flow::calc_grad(&p.data, w, h);
    p.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            let gx = g[(y * w + x) * 2];
            let gy = g[(y * w + x) * 2 + 1];
            let m = (gx * gx + gy * gy).sqrt();
            let f = if m <= 20.0f32 { m * 0.05f32 } else { 1.0f32 };
            *v = (f * 255.0f32 + *v) * 0.5f32;
        }
    });
}

/// The gradient of [`edge_enhance`] (the solver's `calc_grad`) with its horizontal difference
/// circular (G6, U): at every column `gx = (((I(x) - I(x - 1)) + 0.0) + (I(x + 1) - I(x))) / 2`
/// with the columns taken modulo the width, the interior chain of `calc_grad` (whose interior
/// columns divide by `1.0 + 1.0`), so every column `1 <= x <= w - 2` equals `calc_grad` bit for
/// bit and the columns 0 and `w - 1` read across the seam. `gy` is that of `calc_grad` (one sided
/// at the top and bottom rows). U: it is not the solver's wrap gradient `(I(x + 1) - I(x - 1)) /
/// 2` (`flow::calc_grad_wrap`, the variant of the solver), which rounds differently
/// at every column; the transform only becomes circular. A width of 1 gives `gx = 0`.
#[allow(dead_code)]
fn calc_grad_wrapx(img: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; w * h * 2];
    if w == 0 || h == 0 {
        return g;
    }
    g.par_chunks_mut(w * 2).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let c = img[y * w + x];
            let left = img[y * w + (x + w - 1) % w];
            let right = img[y * w + (x + 1) % w];
            let nx = ((c - left) + 0.0f32) + (right - c);
            let dx = 1.0f32 + 1.0f32;
            let mut ny = 0.0f32;
            let mut dy = 0.0f32;
            if y >= 1 {
                ny = (c - img[(y - 1) * w + x]) + 0.0f32;
                dy = 1.0;
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

/// [`edge_enhance`], or with `wrap` its 360 variant (TRACED): the Gaussian of radius 2.0 through
/// `conv::blur_gaussian_wrapx_above_first` (circular horizontal pass, the vertical pass over the poles),
/// the solver's 360 gradient `flow::calc_grad_wrap`, then the per pixel map of [`edge_enhance`] unchanged
/// (`m = sqrt(gx*gx + gy*gy)`, `v = m <= 20.0 ? m * 0.05 : 1.0`, `p = (v * 255.0 + p) * 0.5`,
/// single precision). Without `wrap` it is [`edge_enhance`].
pub fn edge_enhance_with(p: &mut Plane, wrap: bool) {
    if !wrap {
        edge_enhance(p);
        return;
    }
    // TRACED (TwixtorPro_64, the edge enhancement on `mix03.movebox-17x13.d32`, frame 0, level 0 8 x 6): the blur of
    // radius 2.0 is the 360 one, circular rows and the column walk over
    // the poles (0 of 48 samples differ from the traced blurred plane, 40 with the plain blur), and the gradient is the
    // solver's 360 gradient (0 of 96 up to the sign of gy, which the plugin's rows from the bottom flip)
    let (w, h) = (p.width, p.height);
    conv::blur_gaussian_wrapx_above_first(&mut p.data, w, h, 1, 2.0);
    let g = crate::flow::calc_grad_wrap(&p.data, w, h);
    p.data.par_chunks_mut(w.max(1)).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            let gx = g[(y * w + x) * 2];
            let gy = g[(y * w + x) * 2 + 1];
            let m = (gx * gx + gy * gy).sqrt();
            let f = if m <= 20.0f32 { m * 0.05f32 } else { 1.0f32 };
            *v = (f * 255.0f32 + *v) * 0.5f32;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Fmt, Image};

    #[test]
    fn quality_to_resolution_table() {
        // (q, hiq) -> R, for a 1920x1080 output
        for &(q, hiq, r, w0, h0) in &[
            (1u8, true, 3u32, 240usize, 135usize),
            (2, true, 3, 240, 135),
            (3, true, 2, 480, 270),
            (4, true, 1, 960, 540),
            (5, true, 0, 1920, 1080),
            (4, false, 2, 480, 270),
            (3, false, 3, 240, 135),
            (5, false, 1, 960, 540),
            (1, false, 3, 240, 135),
        ] {
            assert_eq!(flow_resolution(q, hiq, 1920, 1080), (r, w0, h0), "q={q} hiq={hiq}");
        }
        // truncating halving
        assert_eq!(flow_resolution(4, true, 1921, 1081), (1, 960, 540));
        assert_eq!(flow_resolution(1, true, 100, 75), (3, 12, 9));
    }

    /// The reduction rule before the tiny frames rule: the table, one more halving at draft, no
    /// floor at 2x2.
    fn old_flow_resolution(q: u8, hiq: bool, w: usize, h: usize) -> (u32, usize, usize) {
        let mut r: u32 = match q {
            0 | 1 | 2 => 3,
            3 => 2,
            4 => 1,
            _ => 0,
        };
        if !hiq && r < 3 {
            r += 1;
        }
        (r, w >> r, h >> r)
    }

    #[test]
    fn searched_sizes_keep_their_reduction() {
        let mut kept = 0;
        let mut lowered = 0;
        for w in 1..=64usize {
            for h in 1..=64usize {
                for q in 1..=5u8 {
                    for hiq in [true, false] {
                        let (r_old, w_old, h_old) = old_flow_resolution(q, hiq, w, h);
                        let (r, w0, h0) = flow_resolution(q, hiq, w, h);
                        let case = format!("{w}x{h} q={q} hiq={hiq}");
                        // the working size is always the truncating halvings of the frame size
                        assert_eq!((w0, h0), (w >> r, h >> r), "{case}");
                        assert!(r <= r_old, "{case}");
                        if w_old >= 2 && h_old >= 2 {
                            // every size that rendered before the rule keeps its reduction, so its
                            // geometry and every value of its solve stay as they were
                            assert_eq!((r, w0, h0), (r_old, w_old, h_old), "{case}");
                            kept += 1;
                        } else if searchable(w, h) {
                            // the largest reduction that leaves at least 2x2
                            assert!(w0 >= 2 && h0 >= 2, "{case}: {w0}x{h0}");
                            assert!(r == r_old || (w >> (r + 1)) < 2 || (h >> (r + 1)) < 2, "{case}: R {r}");
                            lowered += 1;
                        } else {
                            // a side of 1: no halving at all, and the frame is not searched
                            assert_eq!((r, w0, h0), (0, w, h), "{case}");
                        }
                        let geometry = crate::flow::FlowSettings::new(q, hiq, 0.7).geometry(w, h);
                        assert_eq!(geometry.is_ok(), searchable(w, h), "{case}: {geometry:?}");
                    }
                }
            }
        }
        assert!(kept > 0 && lowered > 0, "kept {kept}, lowered {lowered}");
        // the examples of the documentation: 6x6 at High (q 3, the popup value 4) is searched at 3x3
        // (the old R 2 gave 1x1), 2x2 at Sloppy at 2x2, 100x3 at full resolution draft at 100x3 (R 0,
        // the old R 1 gave 50x1)
        assert_eq!(old_flow_resolution(3, true, 6, 6), (2, 1, 1));
        assert_eq!(flow_resolution(3, true, 6, 6), (1, 3, 3));
        assert_eq!(flow_resolution(1, true, 2, 2), (0, 2, 2));
        assert_eq!(flow_resolution(1, false, 17, 16), (3, 2, 2));
        assert_eq!(flow_resolution(5, false, 100, 3), (0, 100, 3));
        assert_eq!(flow_resolution(5, true, 1, 17), (0, 1, 17));
        assert_eq!(flow_resolution(1, false, 17, 1), (0, 17, 1));
    }

    #[test]
    fn side_of_one_is_no_vectors() {
        for (w, h) in [(1usize, 1usize), (1, 2), (2, 1), (1, 17), (17, 1), (1, 1000), (0, 5), (5, 0)] {
            assert!(!searchable(w, h), "{w}x{h}");
            for vq in 0..=4 {
                assert_eq!(searched_quality(vq, w, h), 0, "{w}x{h} vq {vq}");
            }
        }
        for (w, h) in [(2usize, 2usize), (2, 33), (33, 2), (3, 3), (1920, 1080)] {
            assert!(searchable(w, h), "{w}x{h}");
            for vq in 0..=4 {
                assert_eq!(searched_quality(vq, w, h), vq, "{w}x{h} vq {vq}");
            }
        }
    }

    #[test]
    fn pyramid_geometry_of_the_documented_example() {
        let l = level_count(960, 540, 2);
        assert_eq!(l, 8);
        assert_eq!(
            level_sizes(960, 540, l),
            vec![(960, 540), (480, 270), (240, 135), (120, 67), (60, 33), (30, 16), (15, 8), (7, 4)]
        );
        assert_eq!(level_count(4, 4, 2), 1);
        assert_eq!(level_count(8, 8, 2), 2);
        assert_eq!(level_count(16, 3, 2), 1);
    }

    fn rgba8(w: usize, h: usize, px: [u8; 4]) -> Image {
        let mut im = Image::new(w, h, Fmt::U8);
        if let Samples::U8(v) = &mut im.samples {
            for p in v.chunks_mut(4) {
                p.copy_from_slice(&px);
            }
        }
        im
    }

    #[test]
    fn luma_weights_at_each_depth() {
        // 8 bit: (R*0.3 + G*0.59) + B*0.11 in f32
        let im = rgba8(2, 2, [255, 100, 200, 50]);
        let y = luma(&im, false, false).unwrap();
        let want = (100.0f32 * 0.3f32 + 200.0f32 * 0.59f32) + 50.0f32 * 0.11f32;
        assert_eq!(y.data[0].to_bits(), want.to_bits());
        // with alpha: widened, A*0.3 added in f64
        let ya = luma(&im, true, false).unwrap();
        assert_eq!(ya.data[0], ((want as f64) + 255.0f64 * 0.3f64) as f32);
        // 16 bit, white maps to 255 within one f32 ulp
        let mut im16 = Image::new(2, 2, Fmt::U16);
        if let Samples::U16(v) = &mut im16.samples {
            for p in v.chunks_mut(4) {
                p.copy_from_slice(&[65535, 65535, 65535, 65535]);
            }
        }
        let y16 = luma(&im16, false, false).unwrap();
        assert!((y16.data[0] - 255.0).abs() <= 255.0 * 6e-8, "{}", y16.data[0]);
        // mid grey with the documented double precision chain
        if let Samples::U16(v) = &mut im16.samples {
            for p in v.chunks_mut(4) {
                p.copy_from_slice(&[65535, 1000, 30000, 7]);
            }
        }
        let y16 = luma(&im16, false, false).unwrap();
        let want16 = (((30000.0f64 * 0.59 + 1000.0f64 * 0.3) + 7.0f64 * 0.11) / 257.0) as f32;
        assert_eq!(y16.data[0].to_bits(), want16.to_bits());
        // float: white 1.0 -> 255
        let mut imf = Image::new(2, 2, Fmt::F32);
        if let Samples::F32(v) = &mut imf.samples {
            for p in v.chunks_mut(4) {
                p.copy_from_slice(&[1.0, 0.25, 0.5, 0.125]);
            }
        }
        let yf = luma(&imf, false, false).unwrap();
        let wantf = (((0.5f32 as f64 * 0.59 + 0.25f32 as f64 * 0.3) + 0.125f32 as f64 * 0.11) * 255.0) as f32;
        assert_eq!(yf.data[0].to_bits(), wantf.to_bits());
    }

    #[test]
    fn luma_video_range() {
        let im = rgba8(1, 1, [255, 16, 0, 0]);
        let y = luma(&im, false, true).unwrap();
        assert_eq!(y.data[0], 0.0);
        let im = rgba8(1, 1, [255, 235, 0, 0]);
        let y = luma(&im, false, true).unwrap();
        assert_eq!(y.data[0], ((235.0f32 - 16.0) * 255.0) / 219.0);
        assert!(luma(&im, true, true).is_err());
    }

    #[test]
    fn delinearize_formula() {
        let mut p = Plane { width: 4, height: 1, data: vec![0.0, 255.0, -5.0, 127.5] };
        delinearize(&mut p);
        assert_eq!(p.data[0], 0.0);
        assert_eq!(p.data[1], 255.0);
        assert_eq!(p.data[2], 0.0);
        let inv = 1.0f32 / 2.2f32;
        let want = crate::crt_powf::powf(127.5f32 / 255.0f32, inv) * 255.0;
        assert_eq!(p.data[3].to_bits(), want.to_bits());
        // 0.5^(1/2.2) * 255 is about 186.1
        assert!((p.data[3] - 186.1).abs() < 0.1, "{}", p.data[3]);
    }

    #[test]
    fn reduce_halves_and_keeps_constants() {
        let p = Plane { width: 17, height: 9, data: vec![64.0; 17 * 9] };
        let r = reduce(p, 2);
        assert_eq!((r.width, r.height), (4, 2));
        for v in &r.data {
            assert!((v - 64.0).abs() < 1e-4);
        }
    }

    #[test]
    fn circular_edge_gradient_hand_values() {
        let (w, h) = (4usize, 3usize);
        let img = [1.0f32, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 3.0, 5.0, 7.0, 11.0];
        let g = calc_grad_wrapx(&img, w, h);
        // gx = (((c - l) + 0) + (r - c)) / 2 with the columns modulo 4
        let gx = |y: usize| (0..w).map(|x| g[(y * w + x) * 2]).collect::<Vec<f32>>();
        assert_eq!(gx(0), vec![-3.0, 1.5, 3.0, -1.5]);
        assert_eq!(gx(1), vec![-48.0, 24.0, 48.0, -24.0]);
        // gy is one sided at the top and bottom rows, as in calc_grad
        assert_eq!((g[1], g[(4 + 1) * 2 + 1]), (15.0, ((32.0f32 - 2.0) + (5.0 - 32.0)) / 2.0));
        // the interior columns and every gy are calc_grad's bit for bit; the seam columns are not
        let plain = crate::flow::calc_grad(&img, w, h);
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 2;
                assert_eq!(g[i + 1].to_bits(), plain[i + 1].to_bits(), "gy ({x}, {y})");
                assert_eq!(g[i].to_bits() == plain[i].to_bits(), x >= 1 && x + 1 < w, "gx ({x}, {y})");
            }
        }
        // a width of 2 reads the other column on both sides, a width of 1 gives 0
        assert_eq!(calc_grad_wrapx(&[1.0, 5.0, 2.0, 9.0], 2, 2)[0], 0.0);
        assert_eq!(calc_grad_wrapx(&[3.0, 7.0], 1, 2)[0], 0.0);
    }

    /// TRACED (TwixtorPro_64, `retime-fgbg.mix03.movebox-17x13.d32`, frame 0: the solver's level 0 of frame A, 8 x 6,
    /// at the edge enhancement and after it at the level builder, in the plugin's logical rows; 360
    /// Video on): the 360 edge enhancement blurs over the poles and takes the
    /// solver's 360 gradient, bit for bit; the plain blur of the earlier reading differs.
    #[test]
    fn the_360_edge_enhancement_is_the_plugins_on_a_traced_level() {
        let input: [u32; 48] = [
            0x4331312a, 0x4347bc31, 0x43468d55, 0x4338ef71, 0x433b09f7, 0x433d5987, 0x432ca342, 0x4323341f,
            0x433b1b73, 0x43514721, 0x43561733, 0x433c4cb2, 0x43298616, 0x432e9570, 0x4331974a, 0x433272c2,
            0x43472c15, 0x4347e03d, 0x434d8345, 0x433fa9db, 0x43315db0, 0x4335bdfe, 0x43511cbc, 0x4351ffe5,
            0x4330624f, 0x432d4a83, 0x43465302, 0x4346e7e6, 0x433ea1f3, 0x4343eb6c, 0x434c1c64, 0x434230fd,
            0x432d96c6, 0x43357050, 0x434031cf, 0x434211c4, 0x4340dff0, 0x43358287, 0x4326e4a6, 0x432b69db,
            0x434c0228, 0x43518ebb, 0x4343e645, 0x4337e6f5, 0x43325e04, 0x432339dd, 0x4319610c, 0x43356fec,
        ];
        let want: [u32; 48] = [
            0x432ab661, 0x432a6db3, 0x43449153, 0x431f1ef1, 0x42d9ba1a, 0x431b079a, 0x43195e04, 0x42c7faf2,
            0x4332fb56, 0x431d46af, 0x431bbfd0, 0x43421364, 0x42fc14ea, 0x42d02444, 0x4323e480, 0x433ddfc1,
            0x42fb521e, 0x432dc652, 0x430d93a6, 0x432788c0, 0x430d377e, 0x4329e846, 0x432dd5ec, 0x4307dfd4,
            0x4319578f, 0x431800ae, 0x431a0936, 0x42e283e7, 0x42f8ff7e, 0x42f2ea68, 0x433a1770, 0x43409830,
            0x430a9f33, 0x431be2ae, 0x42fdc776, 0x4303291a, 0x430c8e95, 0x432ba54b, 0x433a6d16, 0x43032060,
            0x43177b16, 0x430a7594, 0x4338e3ef, 0x430bfa33, 0x4305df51, 0x431f365c, 0x430d691e, 0x433711fe,
        ];
        let plane = || Plane { width: 8, height: 6, data: input.iter().map(|&b| f32::from_bits(b)).collect() };
        let mut p = plane();
        edge_enhance_with(&mut p, true);
        assert_eq!(p.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want.to_vec());
        // the plain blur with the 360 gradient (the earlier reading) is not the plugin's
        let mut q = plane();
        conv::blur_gaussian_above_first(&mut q.data, 8, 6, 1, 2.0);
        let g = crate::flow::calc_grad_wrap(&q.data, 8, 6);
        for (i, v) in q.data.iter_mut().enumerate() {
            let m = (g[i * 2] * g[i * 2] + g[i * 2 + 1] * g[i * 2 + 1]).sqrt();
            let f = if m <= 20.0f32 { m * 0.05f32 } else { 1.0f32 };
            *v = (f * 255.0f32 + *v) * 0.5f32;
        }
        assert_ne!(q.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want.to_vec());
    }

    #[test]
    fn edge_enhance_keeps_flat_images_at_half_plus_zero() {
        // flat image: gradient 0, v = 0 -> p = (0 * 255 + p) * 0.5
        let mut p = Plane { width: 12, height: 10, data: vec![100.0; 120] };
        edge_enhance(&mut p);
        for v in &p.data {
            assert!((v - 50.0).abs() < 1e-3, "{v}");
        }
    }
}
