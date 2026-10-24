//! Blur Compensation, the post warp tail of the compositor.
//!
//! The chain: [`blur_amount`] decides whether to run and computes the amount; [`compensate`]
//! clamps it and either blurs along the flow (positive) or builds a flow directed unsharp mask
//! (negative), using the motion blur primitive [`motion_blur`]; the post step is
//! [`clear_rgb_where_transparent`].

use super::{bilerp4, cvtt32, Flow, Pix};
use crate::flow::FlowField;
use crate::image::{Image, Samples, CHANNELS};
use fx_core::{par_rows, FxError, Result};
use std::sync::atomic::{AtomicBool, Ordering};

/// Largest tap count `n` of one pixel that is run. A longer vector fails the call with
/// `Unsupported` instead of computing for minutes (the count can reach 2^31 for absurd vectors).
/// A flow of 65 000 pixels at blur amount 1 is still handled.
const TAP_LIMIT: i32 = 1 << 16;

/// `1e-6` as a single precision constant: the squared vector length below which the bilinear
/// variant copies the pixel.
const TINY_VECTOR: f32 = f32::from_bits(0x3586_37bd);

/// The condition and amount of the compensation: with `d = s0 - s1` (single precision), it runs
/// only when `|d| > 1e-6` (compared in double) and the slider `blur > 0.005` (compared in
/// double), and a flow exists (which the caller knows); the amount is `(|d| - 1.0) * blur` in
/// single precision. It is positive when the source advances more than one frame per output
/// frame (motion blur along the flow) and negative in slow motion (flow directed sharpening).
pub fn blur_amount(s0: f32, s1: f32, blur: f32) -> Option<f32> {
    let a = (s0 - s1).abs();
    if !((a as f64) > 1e-6) || !((blur as f64) > 0.005) {
        return None;
    }
    Some((a - 1.0f32) * blur)
}

/// Blur Compensation of the compositor tail: `img` is the composed output, `flow` the flow handed
/// over by the caller (else the solver's finest flow), `s0` and `s1` the two source positions of
/// the output frame (`Selection::{s0, s1}`), `blur` the slider, `hiq` the high quality render
/// flag (selects the bilinear taps; false selects the nearest taps).
///
/// Does nothing when the condition fails ([`blur_amount`] is `None`). Otherwise [`compensate`]
/// runs on the amount and the result replaces `img` (it blurs into a temporary and copies it
/// back). The 360 variant is not supported: see [`motion_blur`].
pub fn blur_compensation(
    img: &mut Image,
    flow: &FlowField,
    s0: f32,
    s1: f32,
    blur: f32,
    hiq: bool,
) -> Result<()> {
    match blur_amount(s0, s1, blur) {
        Some(amount) => compensate(img, flow, amount, hiq),
        None => Ok(()),
    }
}

/// Applies a compensation amount. The amount is clamped to `[-10, 20]` (`20.0 < amount` gives
/// 20, then `amount < -10` gives -10; NaN passes). `|amount| < 1e-6` (double compare) leaves the
/// image unchanged. A positive amount runs [`motion_blur`] with that amount and shutter 0.5 into
/// a temporary, then [`clear_rgb_where_transparent`]. Anything else (negative, NaN) runs the
/// motion blur with amount 0.5 and then `lerp(temp, img, amount)` (weight on the blurred copy,
/// so `img + (blur - img) * amount`: a flow directed unsharp mask of strength `|amount|`). The
/// temporary then replaces `img`.
pub fn compensate(img: &mut Image, flow: &FlowField, amount: f32, hiq: bool) -> Result<()> {
    let mut amount = amount;
    if 20.0f32 < amount {
        amount = 20.0;
    } else if amount < -10.0f32 {
        amount = -10.0;
    }
    if (amount.abs() as f64) < 1e-6 {
        return Ok(());
    }
    let mut tmp = Image::new(img.width, img.height, img.fmt());
    if 0.0f32 < amount {
        motion_blur(&mut tmp, img, flow, amount, hiq, 0.5, false)?;
        clear_rgb_where_transparent(&mut tmp);
    } else {
        motion_blur(&mut tmp, img, flow, 0.5, hiq, 0.5, false)?;
        crate::blend::lerp(&mut tmp, img, amount);
    }
    crate::blend::copy(img, &tmp);
    Ok(())
}

/// The post step of a positive compensation: where alpha is exactly zero (`A == 0`; for float
/// `-0.0` counts and NaN does not) the colour channels are set to zero (positive zero). Four
/// channel images only, which is all the engine has.
pub fn clear_rgb_where_transparent(img: &mut Image) {
    match &mut img.samples {
        Samples::U8(v) => v.chunks_exact_mut(CHANNELS).for_each(|p| {
            if p[0] == 0 {
                p[1] = 0;
                p[2] = 0;
                p[3] = 0;
            }
        }),
        Samples::U16(v) => v.chunks_exact_mut(CHANNELS).for_each(|p| {
            if p[0] == 0 {
                p[1] = 0;
                p[2] = 0;
                p[3] = 0;
            }
        }),
        Samples::F32(v) => v.chunks_exact_mut(CHANNELS).for_each(|p| {
            if p[0] == 0.0 {
                p[1] = 0.0;
                p[2] = 0.0;
                p[3] = 0.0;
            }
        }),
    }
}

/// The flow directed motion blur primitive, over the whole image.
///
/// Variant selection: `hiq` picks the bilinear tap variant, otherwise the nearest tap variant.
/// The 360 wrap variants are not supported: `is360` returns [`FxError::Unsupported`].
///
/// With `rx = flow.w1 / src.w1`, `ry = flow.h1 / src.h1`, `kx = (0.5 / rx) * amount`,
/// `ky = (0.5 / ry) * amount`, `k = shutter * 2.0 - 1.0` (all f32), per pixel `(x, y)`:
///
/// * `(dx, dy)` = the flow sampled at `(x * rx, y * ry)`, `vx = dx * kx`, `vy = dy * ky`,
///   `d2 = vy * vy + vx * vx`.
/// * Bilinear variant: `1e-6 > d2` (f32 compare) copies the source pixel. Otherwise
///   `n = trunc32(sqrtf(d2) + 1.0)`, if `n < 2` then `n = 2` and the step is `(vx, vy)`, else the
///   step is `(vx, vy) / (n - 1)`. The centre is `(x, y)`, or `(vx * k + x, vy * k + y)` when
///   `shutter != 0.5`. `n` forward taps start at the centre and advance by the step (accumulated
///   by repeated addition), then `n - 1` backward taps start at `centre - step` and advance by
///   `-step`; a tap counts and adds `bilerp(src, px, py)` only when `0 <= px < W` and
///   `0 <= py < H`. The result is `(1.0 / count) * sum` per channel, stored with
///   `Pix::store_blur`.
/// * Nearest variant: no small vector shortcut, taps at `trunc32(p + 0.5)` (signed truncation), a
///   tap is inside when `0 <= ix < W` and `0 <= iy < H` after the truncation; the forward taps
///   count only when inside, the backward taps ALWAYS count (the increment sits after the range
///   test in the backward loop). The centre of the nearest taps is `centre + 0.5`.
///
/// A NaN tap coordinate in the bilinear variant would index outside the image, and a vector of
/// more than 65 536 taps would run for minutes per pixel; both are reported as
/// [`FxError::Unsupported`].
pub fn motion_blur(
    dest: &mut Image,
    src: &Image,
    flow: &FlowField,
    amount: f32,
    hiq: bool,
    shutter: f32,
    is360: bool,
) -> Result<()> {
    if !dest.same_shape(src) {
        return Err(FxError::Mismatch(format!(
            "motion blur: destination {}x{} {:?} against source {}x{} {:?}",
            dest.width,
            dest.height,
            dest.fmt(),
            src.width,
            src.height,
            src.fmt()
        )));
    }
    if src.width < 2 || src.height < 2 || flow.width < 2 || flow.height < 2 {
        return Err(FxError::Mismatch(format!(
            "motion blur needs images and a flow of at least 2x2 (image {}x{}, flow {}x{})",
            src.width, src.height, flow.width, flow.height
        )));
    }
    if is360 {
        return Err(FxError::Unsupported(
            "motion blur with the 360 flag (wrapping sample lookups) is not implemented".into(),
        ));
    }
    let ok = match (&mut dest.samples, &src.samples) {
        (Samples::U8(d), Samples::U8(s)) => run::<u8>(d, s, src.width, src.height, flow, amount, hiq, shutter),
        (Samples::U16(d), Samples::U16(s)) => run::<u16>(d, s, src.width, src.height, flow, amount, hiq, shutter),
        (Samples::F32(d), Samples::F32(s)) => run::<f32>(d, s, src.width, src.height, flow, amount, hiq, shutter),
        _ => unreachable!("shapes were checked"),
    };
    if ok {
        Ok(())
    } else {
        Err(FxError::Unsupported(
            "motion blur: a tap coordinate is NaN or a vector needs more than 65536 taps, which is not supported".into(),
        ))
    }
}

struct Blur<'a, P> {
    src: &'a [P],
    flow: Flow<'a>,
    w: usize,
    wf: f32,
    hf: f32,
    w1i: i32,
    h1i: i32,
    rx: f32,
    ry: f32,
    kx: f32,
    ky: f32,
    /// `shutter != 0.5` and `k = shutter * 2.0 - 1.0`
    shifted: bool,
    k: f32,
    bilinear: bool,
    bad: AtomicBool,
}

fn run<P: Pix>(
    dest: &mut [P],
    src: &[P],
    width: usize,
    height: usize,
    flow: &FlowField,
    amount: f32,
    hiq: bool,
    shutter: f32,
) -> bool {
    let w1i = width as i32 - 1;
    let h1i = height as i32 - 1;
    // rx, ry, kx, ky exactly as the prologue of the leaves: f32 divisions, `(0.5 / r) * amount`
    let rx = (flow.width as i32 - 1) as f32 / w1i as f32;
    let ry = (flow.height as i32 - 1) as f32 / h1i as f32;
    let kx = (0.5f32 / rx) * amount;
    let ky = (0.5f32 / ry) * amount;
    let b = Blur {
        src,
        flow: Flow::new(flow),
        w: width,
        wf: width as f32,
        hf: height as f32,
        w1i,
        h1i,
        rx,
        ry,
        kx,
        ky,
        shifted: shutter != 0.5f32,
        k: shutter * 2.0f32 - 1.0f32,
        bilinear: hiq,
        bad: AtomicBool::new(false),
    };
    par_rows(dest, width * CHANNELS, |y, row| b.row(y, row));
    !b.bad.load(Ordering::Relaxed)
}

impl<'a, P: Pix> Blur<'a, P> {
    fn row(&self, y: usize, row: &mut [P]) {
        let yf = y as f32;
        let fy = yf * self.ry;
        for x in 0..self.w {
            let xf = x as f32;
            let [dx, dy] = self.flow.bilerp(xf * self.rx, fy);
            let vx = dx * self.kx;
            let vy = dy * self.ky;
            let d2 = vy * vy + vx * vx;
            let px = &mut row[x * CHANNELS..(x + 1) * CHANNELS];
            if self.bilinear {
                if TINY_VECTOR > d2 {
                    let i = (y * self.w + x) * CHANNELS;
                    px.copy_from_slice(&self.src[i..i + CHANNELS]);
                } else {
                    self.blur_bilinear(xf, yf, vx, vy, d2, px);
                }
            } else {
                self.blur_nearest(xf, yf, vx, vy, d2, px);
            }
        }
    }

    #[inline(always)]
    fn blur_bilinear(&self, xf: f32, yf: f32, vx: f32, vy: f32, d2: f32, px: &mut [P]) {
        let dist = d2.sqrt();
        let (cx, cy) = if self.shifted { (vx * self.k + xf, vy * self.k + yf) } else { (xf, yf) };
        let mut n = cvtt32(dist + 1.0f32);
        let (sx, sy);
        if n < 2 {
            n = 2;
            sx = vx;
            sy = vy;
        } else {
            let d = (n - 1) as f32;
            sx = vx / d;
            sy = vy / d;
        }
        if n > TAP_LIMIT {
            self.bad.store(true, Ordering::Relaxed);
            px.fill(P::ZERO);
            return;
        }
        let mut sum = [0.0f32; CHANNELS];
        let mut cnt = 0i32;
        let tap = |qx: f32, qy: f32, sum: &mut [f32; CHANNELS], cnt: &mut i32| {
            if qx.is_nan() || qy.is_nan() {
                self.bad.store(true, Ordering::Relaxed);
                return;
            }
            if 0.0 <= qx && qx < self.wf && 0.0 <= qy && qy < self.hf {
                *cnt += 1;
                let t = bilerp4(self.src, self.w, self.w1i, self.h1i, qx, qy);
                for c in 0..CHANNELS {
                    sum[c] += t[c];
                }
            }
        };
        let (mut qx, mut qy) = (cx, cy);
        for _ in 0..n {
            tap(qx, qy, &mut sum, &mut cnt);
            qx += sx;
            qy += sy;
        }
        let (nsx, nsy) = (-sx, -sy);
        let (mut qx, mut qy) = (cx - sx, cy - sy);
        for _ in 1..n {
            tap(qx, qy, &mut sum, &mut cnt);
            qx += nsx;
            qy += nsy;
        }
        let inv = 1.0f32 / cnt as f32;
        for c in 0..CHANNELS {
            px[c] = P::store_blur(inv * sum[c]);
        }
    }

    /// Nearest taps, no small vector shortcut, and the backward taps are counted whether or not
    /// they are inside.
    #[inline(always)]
    fn blur_nearest(&self, xf: f32, yf: f32, vx: f32, vy: f32, d2: f32, px: &mut [P]) {
        let dist = d2.sqrt();
        let (cx, cy) = if self.shifted { (vx * self.k + xf, vy * self.k + yf) } else { (xf, yf) };
        let cx = cx + 0.5f32;
        let cy = cy + 0.5f32;
        let mut n = cvtt32(dist + 1.0f32);
        let (sx, sy);
        if n < 2 {
            n = 2;
            sx = vx;
            sy = vy;
        } else {
            let d = (n - 1) as f32;
            sx = vx / d;
            sy = vy / d;
        }
        if n > TAP_LIMIT {
            self.bad.store(true, Ordering::Relaxed);
            px.fill(P::ZERO);
            return;
        }
        let mut sum = [0.0f32; CHANNELS];
        let mut cnt = 0i32;
        let w = self.w as i32;
        let h = self.hf as i32;
        let (mut qx, mut qy) = (cx, cy);
        for _ in 0..n {
            let (ix, iy) = (cvtt32(qx), cvtt32(qy));
            if ix >= 0 && ix < w && iy >= 0 && iy < h {
                cnt += 1;
                let i = (iy as usize * self.w + ix as usize) * CHANNELS;
                for c in 0..CHANNELS {
                    sum[c] += self.src[i + c].to_f32();
                }
            }
            qx += sx;
            qy += sy;
        }
        let (nsx, nsy) = (-sx, -sy);
        let (mut qx, mut qy) = (cx - sx, cy - sy);
        for _ in 1..n {
            let (ix, iy) = (cvtt32(qx), cvtt32(qy));
            if ix >= 0 && ix < w && iy >= 0 && iy < h {
                let i = (iy as usize * self.w + ix as usize) * CHANNELS;
                for c in 0..CHANNELS {
                    sum[c] += self.src[i + c].to_f32();
                }
            }
            cnt += 1;
            qx += nsx;
            qy += nsy;
        }
        let inv = 1.0f32 / cnt as f32;
        for c in 0..CHANNELS {
            px[c] = P::store_blur(inv * sum[c]);
        }
    }
}
