//! Blur Compensation, the post warp tail of the compositor.
//!
//! The chain: [`blur_amount`] decides whether to run and computes the amount; [`compensate`]
//! clamps it and either blurs along the flow (positive) or builds a flow directed unsharp mask
//! (negative), using the motion blur primitive [`motion_blur`]; the post step is
//! [`clear_rgb_where_transparent`].
//!
//! The wrap-around (360) variant ([`blur_compensation_wrap`] with `is360`, parity gap G6): the bilinear taps are READ
//! (the float 360 leaf of TwixtorPro_64's blur compensation): every tap counts, its x is
//! wrapped into `[0, W)` by the loops of the 360 inverse warp and its position is put on the sphere
//! ([`sphere_tap`]: y by whole heights, after an odd count x half a turn and y reflected), then sampled with the 360
//! sampler of the warp (`bilerp360`). The nearest taps keep the most constrained reading (U): the taps of the plain
//! leaf with the column `trunc(x)` of the wrapped coordinate and the plain y test. The flow lookup and the forward and
//! backward tap order are those of the plain leaves. A pixel whose taps all stay inside the image (`0 <= x < W - 1`,
//! `0 <= y < H - 1` for the bilinear taps, `0 <= x < W` for the nearest taps) comes out bit identical to the plain blur.
//!
//! Numerics: single precision in the written order, no fused multiply add, truncating stores as
//! documented per function; rows run through rayon row chunks, so the result does not depend on
//! the thread count.

use super::{bilerp360, bilerp4, cvtt32, Flow, Pix, WRAP_LIMIT};
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
/// back). The 360 variant is [`blur_compensation_wrap`]; this is its call with the flag off.
pub fn blur_compensation(
    img: &mut Image,
    flow: &FlowField,
    s0: f32,
    s1: f32,
    blur: f32,
    hiq: bool,
) -> Result<()> {
    blur_compensation_wrap(img, flow, s0, s1, blur, hiq, false)
}

/// [`blur_compensation`] with the engine's wrap-around (360) flag for the taps of the motion
/// blur. `is360 == false` is the compensation above, bit for bit. With the flag the
/// same chain runs (condition, clamp, positive blur and clear, or negative blur and lerp) with the
/// wrapped taps of [`motion_blur`] (G6, U; see the module header); one whose condition fails, or
/// whose clamped amount is below 1e-6, does nothing as without the flag.
pub fn blur_compensation_wrap(
    img: &mut Image,
    flow: &FlowField,
    s0: f32,
    s1: f32,
    blur: f32,
    hiq: bool,
    is360: bool,
) -> Result<()> {
    match blur_amount(s0, s1, blur) {
        Some(amount) if !is360 => compensate(img, flow, amount, hiq),
        Some(amount) => compensate_wrap(img, flow, amount, hiq, true),
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
    compensate_wrap(img, flow, amount, hiq, false)
}

/// [`compensate`] with the 360 flag of the motion blur taps.
fn compensate_wrap(img: &mut Image, flow: &FlowField, amount: f32, hiq: bool, is360: bool) -> Result<()> {
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
        motion_blur(&mut tmp, img, flow, amount, hiq, 0.5, is360)?;
        clear_rgb_where_transparent(&mut tmp);
    } else {
        motion_blur(&mut tmp, img, flow, 0.5, hiq, 0.5, is360)?;
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
/// Variant selection: `hiq` picks the bilinear tap variant, otherwise the nearest tap variant;
/// `is360` picks the wrap-around taps of either (described after the plain ones).
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
/// * Wrap-around taps (`is360`, G6): the tap positions are accumulated exactly as above; per tap a copy of its x is
///   wrapped into `[0, W)` by the loops of the 360 inverse warp (`while x < 0 { x += W }`, then
///   `while W <= x { x -= W }`, single precision). Bilinear (READ): EVERY tap counts and adds
///   `bilerp360(src, x', y')` at its position on the sphere ([`sphere_tap`]), the 360 sampler of the inverse warp
///   (right neighbour of the last column is column 0, the row below the last row is the last row shifted by half the
///   width). Nearest (U, the most constrained reading) (`q = p + 0.5` the accumulated position): a tap is inside when
///   `0 <= trunc32(q.y) < H`, and an inside tap reads column `trunc32` of its wrapped `q.x`, which
///   is its column modulo `W`; the forward and backward counting rules stay. The x of a tap whose
///   y is outside is not wrapped.
///
/// A NaN tap coordinate in the bilinear variant would index outside the image, and a vector of
/// more than 65 536 taps would run for minutes per pixel; both are reported as
/// [`FxError::Unsupported`]. With `is360` the same holds for a tap x that the wrap loops cannot
/// bring into `[0, W)` within their bound (NaN, infinite or astronomically far, as the 360 inverse
/// warp refuses it).
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
        fx_core::gaps::record(&crate::gaps::G6);
    }
    let ok = match (&mut dest.samples, &src.samples) {
        (Samples::U8(d), Samples::U8(s)) => run::<u8>(d, s, src.width, src.height, flow, amount, hiq, shutter, is360),
        (Samples::U16(d), Samples::U16(s)) => run::<u16>(d, s, src.width, src.height, flow, amount, hiq, shutter, is360),
        (Samples::F32(d), Samples::F32(s)) => run::<f32>(d, s, src.width, src.height, flow, amount, hiq, shutter, is360),
        _ => unreachable!("shapes were checked"),
    };
    if ok {
        Ok(())
    } else if is360 {
        Err(FxError::Unsupported(
            "motion blur: a tap coordinate is NaN or never wraps, or a vector needs more than 65536 taps, which is not supported".into(),
        ))
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
    is360: bool,
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
    if is360 {
        par_rows(dest, width * CHANNELS, |y, row| b.row::<true>(y, row));
    } else {
        par_rows(dest, width * CHANNELS, |y, row| b.row::<false>(y, row));
    }
    !b.bad.load(Ordering::Relaxed)
}

/// The position of a bilinear wrap-around tap on the sphere (READ, the float 360 leaf of the blur compensation,
/// TwixtorPro_64; TRACED on `mix08.movebox-31x21.d32`: the sampler receives the backward tap of pixel (1, 0)
/// at x = 16.5016, half a turn from 1.0016): `x` is already in `[0, W)`; `y += H`
/// while `y < 0` and `y -= H` while `H <= y`, counting the steps; after an odd count `x = W * 0.5 + x` (then `W` less
/// unless it is below `W`) and a `y` other than 0 becomes `H - y`. `None` when the loops exceed their bound or the
/// result is not inside the frame (NaN, infinite, or a reflection that rounds to `H`).
#[inline(always)]
pub(crate) fn sphere_tap(x: f32, qy: f32, wf: f32, hf: f32) -> Option<(f32, f32)> {
    let mut y = qy;
    let mut steps = 0u32;
    while y < 0.0 {
        y += hf;
        steps += 1;
        if steps > WRAP_LIMIT {
            return None;
        }
    }
    while hf <= y {
        y -= hf;
        steps += 1;
        if steps > WRAP_LIMIT {
            return None;
        }
    }
    let mut x = x;
    if steps % 2 == 1 {
        x = wf * 0.5f32 + x;
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(x < wf) {
            x -= wf;
        }
        if y != 0.0 {
            y = hf - y;
        }
    }
    if x >= 0.0 && x < wf && y >= 0.0 && y < hf {
        Some((x, y))
    } else {
        None
    }
}

impl<'a, P: Pix> Blur<'a, P> {
    /// The x of a wrap-around tap brought into `[0, W)` by the loops of the 360 inverse warp
    /// (`x += W` while `x < 0`, then `x -= W` while `W <= x`, single precision), or `None` when
    /// the loops exceed their bound or the result is outside (NaN, infinite or astronomically far).
    #[inline(always)]
    fn wrap_x(&self, qx: f32) -> Option<f32> {
        let wf = self.wf;
        let mut x = qx;
        let mut guard = 0u32;
        while x < 0.0 {
            x += wf;
            guard += 1;
            if guard > WRAP_LIMIT {
                return None;
            }
        }
        while wf <= x {
            x -= wf;
            guard += 1;
            if guard > WRAP_LIMIT {
                return None;
            }
        }
        if x >= 0.0 && x < wf {
            Some(x)
        } else {
            None
        }
    }

    /// One output row; `WRAP` selects the wrap-around taps.
    fn row<const WRAP: bool>(&self, y: usize, row: &mut [P]) {
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
                    self.blur_bilinear::<WRAP>(xf, yf, vx, vy, d2, px);
                }
            } else {
                self.blur_nearest::<WRAP>(xf, yf, vx, vy, d2, px);
            }
        }
    }

    /// Bilinear taps; `WRAP` wraps the x of a tap whose y is inside and samples it with
    /// `bilerp360` (the module header).
    #[inline(always)]
    fn blur_bilinear<const WRAP: bool>(&self, xf: f32, yf: f32, vx: f32, vy: f32, d2: f32, px: &mut [P]) {
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
            if WRAP {
                // every tap counts; its position is put on the sphere ([`sphere_tap`]) and sampled with the 360 sampler
                let Some((wx, wy)) = self.wrap_x(qx).and_then(|wx| sphere_tap(wx, qy, self.wf, self.hf)) else {
                    self.bad.store(true, Ordering::Relaxed);
                    return;
                };
                *cnt += 1;
                let t = bilerp360(self.src, self.w, self.w1i, self.h1i, wx, wy);
                for c in 0..CHANNELS {
                    sum[c] += t[c];
                }
            } else if 0.0 <= qx && qx < self.wf && 0.0 <= qy && qy < self.hf {
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
    /// they are inside. `WRAP` replaces the x test by the wrap of the tap x (the module header).
    #[inline(always)]
    fn blur_nearest<const WRAP: bool>(&self, xf: f32, yf: f32, vx: f32, vy: f32, d2: f32, px: &mut [P]) {
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
        // the column of an inside wrap-around tap: `trunc32` of its x wrapped into [0, W), so
        // `0 <= ix < W`; an x that cannot be wrapped fails the call
        let wrapped_column = |qx: f32| -> Option<usize> {
            match self.wrap_x(qx) {
                Some(x) => Some(cvtt32(x) as usize),
                None => {
                    self.bad.store(true, Ordering::Relaxed);
                    None
                }
            }
        };
        let (mut qx, mut qy) = (cx, cy);
        for _ in 0..n {
            if WRAP {
                let iy = cvtt32(qy);
                if iy >= 0 && iy < h {
                    if let Some(ix) = wrapped_column(qx) {
                        cnt += 1;
                        let i = (iy as usize * self.w + ix) * CHANNELS;
                        for c in 0..CHANNELS {
                            sum[c] += self.src[i + c].to_f32();
                        }
                    }
                }
            } else {
                let (ix, iy) = (cvtt32(qx), cvtt32(qy));
                if ix >= 0 && ix < w && iy >= 0 && iy < h {
                    cnt += 1;
                    let i = (iy as usize * self.w + ix as usize) * CHANNELS;
                    for c in 0..CHANNELS {
                        sum[c] += self.src[i + c].to_f32();
                    }
                }
            }
            qx += sx;
            qy += sy;
        }
        let (nsx, nsy) = (-sx, -sy);
        let (mut qx, mut qy) = (cx - sx, cy - sy);
        for _ in 1..n {
            if WRAP {
                let iy = cvtt32(qy);
                if iy >= 0 && iy < h {
                    if let Some(ix) = wrapped_column(qx) {
                        let i = (iy as usize * self.w + ix) * CHANNELS;
                        for c in 0..CHANNELS {
                            sum[c] += self.src[i + c].to_f32();
                        }
                    }
                }
            } else {
                let (ix, iy) = (cvtt32(qx), cvtt32(qy));
                if ix >= 0 && ix < w && iy >= 0 && iy < h {
                    let i = (iy as usize * self.w + ix as usize) * CHANNELS;
                    for c in 0..CHANNELS {
                        sum[c] += self.src[i + c].to_f32();
                    }
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
