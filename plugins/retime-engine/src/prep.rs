//! Image preparation for the motion estimator: colour to luma, the quality to resolution mapping,
//! the reduction to the working resolution, and the two optional transforms, Delinearize and
//! Contrast / Edge Enhance.
//!
//! The engine images arriving here are PREMULTIPLIED A, R, G, B images (8 bpc: 0..255, 16 bpc:
//! 0..65535, float: white 1.0). The output of the luma stage is a single channel float image in
//! the range 0..255 (`Plane`).

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
    let (mut w0, mut h0) = (w, h);
    for _ in 0..r {
        w0 /= 2;
        h0 /= 2;
    }
    (r, w0, h0)
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

/// The 5 tap binomial blur `[1 4 6 4 1] / 16` of the pyramid code.
pub fn blur5(p: &mut Plane) {
    conv::blur5(&mut p.data, p.width, p.height);
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

/// Delinearize: `p = (max(p, 0) / 255) ^ (1 / 2.2) * 255` in single precision with
/// `gamma = 2.2f`. The power is the double precision `pow` rounded to single, which differs from a
/// single precision `powf` for about 1 in 8000 arguments (1 ulp).
pub fn delinearize(p: &mut Plane) {
    let gamma: f32 = 2.2;
    let inv = 1.0f32 / gamma;
    p.data.par_iter_mut().for_each(|v| {
        if *v <= 0.0 && *v != 0.0 {
            *v = 0.0;
        }
        let base = *v / 255.0f32;
        *v = (base as f64).powf(inv as f64) as f32 * 255.0f32;
    });
}

/// Contrast / Edge Enhance: the image is blurred in place with the Gaussian of radius 2.0, its
/// gradient is taken (the solver's `calc_grad`), and per pixel `m = sqrt(gx*gx + gy*gy)`,
/// `v = m <= 20.0 ? m * 0.05 : 1.0`, `p = (v * 255.0 + p) * 0.5` (single precision).
pub fn edge_enhance(p: &mut Plane) {
    let (w, h) = (p.width, p.height);
    conv::blur_gaussian(&mut p.data, w, h, 1, 2.0);
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
        let want = ((127.5f32 / 255.0f32) as f64).powf(inv as f64) as f32 * 255.0;
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
    fn edge_enhance_keeps_flat_images_at_half_plus_zero() {
        // flat image: gradient 0, v = 0 -> p = (0 * 255 + p) * 0.5
        let mut p = Plane { width: 12, height: 10, data: vec![100.0; 120] };
        edge_enhance(&mut p);
        for v in &p.data {
            assert!((v - 50.0).abs() < 1e-3, "{v}");
        }
    }
}
