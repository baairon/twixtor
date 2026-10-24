//! Hole fill of the engine, used by the Smart Blend modes and by Vectors In.
//!
//! A pull-push pyramid fill of every channel on its own:
//!
//! 1. nothing happens unless `holes` has the size of `image`;
//! 2. the pyramid is built for the size `W x H`: level 0 is `W x H`, level `k + 1` is
//!    `max(2, w_k / 2) x max(2, h_k / 2)` (integer division). `n` counts the loop that builds the
//!    level sizes (until both dimensions are 2), the push starts at level `n - 1`;
//! 3. for every channel `c` of the image, in order (alpha first, a flow has 2 channels):
//!    * load: the channel becomes the f32 value image of level 0 in the native units of the depth
//!      (`(float)byte`, `(float)word`, a bit copy). The weight image of level 0 and the value
//!      image are set up together: for every pixel with `(double)mask >= eps` (a hole) value and
//!      weight are 0, else the weight is 1.0 (the value is kept). `eps` is [`U8_EPS`] for a u8
//!      mask and [`WIDE_EPS`] (3.1e-5) for u16 and f32 masks;
//!    * pull (levels `1 ..= n`): for the value image and, separately, for the weight image: copy
//!      level `k - 1`, `blur5` it (the `[1 4 6 4 1] / 16` filter of the flow pyramids) and resample
//!      it to the size of level `k` (corner aligned bilinear). Level `n` is built but never used;
//!    * push: the solution image of the top level `n - 1` is cleared; then for `level = n - 1`
//!      down to 0: `sol[level] = (weight != 0) ? value / weight : sol[level]`, then ONE Gaussian
//!      blur of radius 5.0; at level 0 the same `value / weight` step runs again (so every pixel
//!      with a non zero weight is restored), at any higher level the solution is resampled into
//!      the level below;
//!    * write back: u8 and u16 `y = x + 0.5f; 0 > y ? 0 : y > white ? white : trunc(y)`, f32 a
//!      bit copy.
//!
//! A pixel that is not a hole comes back unchanged at every depth (weight 1.0, `v / 1.0`), so the
//! fill is the identity without holes and `fill_holes` returns early then. A hole pixel receives
//! a smooth blend of the surrounding known pixels (a low pass of the weighted average at the
//! coarsest level that has support). Colour and alpha are filled independently: a transparent
//! hole next to opaque pixels becomes opaque.
//!
//! The 360 variant (`is360`, which wraps horizontally) is not supported: with at least one hole
//! it returns `FxError::Unsupported`.
//!
//! Numerics: every arithmetic step is a single precision operation of the reused primitives
//! (`crate::flow::conv`, `crate::flow::sample`, no fused multiply add, no reciprocal estimate) or
//! the division of the combine step. Rows run through rayon row chunks, so the result does not
//! depend on the thread count.

use crate::blend::cvtt;
use crate::flow::conv::{blur5, blur_gaussian};
use crate::flow::sample::resample1;
use crate::flow::FlowField;
use crate::image::{Image, Samples, CHANNELS};
use crate::warp::Coverage;
use fx_core::{FxError, Result};

#[cfg(test)]
pub(crate) mod tests;

/// Threshold of the u8 hole mask test `eps <= (double)m`; the value is about 6.0e-154. Any value
/// in `(0, 1]` gives the same decision for integer masks (`m != 0` is a hole), so the exact
/// value does not change a u8 result.
pub const U8_EPS: f64 = f64::from_bits(0x2020_2020_2020_20fc);
/// Threshold of the hole mask test for u16 and f32 masks (`3.1e-5`).
pub const WIDE_EPS: f64 = f64::from_bits(0x3f00_40bf_e3b0_3e21);

/// The `holes` image: 255 (u8), 65535 (u16) or 1.0 (f32) where the warp left nothing, 0 where the
/// pixel is valid. For Smart Blend this is the inverted coverage.
#[derive(Clone, Copy, Debug)]
pub enum HoleMaskRef<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
    F32(&'a [f32]),
}

impl HoleMaskRef<'_> {
    pub fn len(&self) -> usize {
        match self {
            HoleMaskRef::U8(m) => m.len(),
            HoleMaskRef::U16(m) => m.len(),
            HoleMaskRef::F32(m) => m.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A pixel is a hole when `eps <= m` (compared as doubles) or the compare is unordered (a NaN
    /// float mask value).
    #[inline]
    fn is_hole(&self, i: usize) -> bool {
        match self {
            HoleMaskRef::U8(m) => !(U8_EPS > f64::from(m[i])),
            HoleMaskRef::U16(m) => !(WIDE_EPS > f64::from(m[i])),
            HoleMaskRef::F32(m) => !(WIDE_EPS > f64::from(m[i])),
        }
    }
}

/// Levels of the hole fill pyramid. `dims[k]` is the size of level `k` for `k` in `0 ..= n`; the
/// push starts at `top = n - 1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub dims: Vec<(usize, usize)>,
    pub top: usize,
}

/// The level sizes of the hole fill pyramid for an image of `w x h`.
pub fn geometry(w: usize, h: usize) -> Geometry {
    // do { a /= 2; b /= 2; clamp both to >= 2; count += 1 }
    // while (a > 1 && b > 1 && (a != 2 || b != 2))
    let (mut a, mut b) = (w as i64, h as i64);
    let mut count = 1usize;
    loop {
        a /= 2;
        b /= 2;
        if a < 2 {
            a = 2;
        }
        if b < 2 {
            b = 2;
        }
        count += 1;
        if !(a > 1 && b > 1 && (a != 2 || b != 2)) {
            break;
        }
    }
    let n = count;
    let mut dims = Vec::with_capacity(n + 1);
    let (mut cw, mut ch) = (w, h);
    for _ in 0..=n {
        dims.push((cw, ch));
        cw = (cw / 2).max(2);
        ch = (ch / 2).max(2);
    }
    Geometry { dims, top: n - 1 }
}

/// The part of the fill that depends only on the hole mask: the weight pyramid. It is the same
/// for every channel, so it is built once.
struct Filler {
    geo: Geometry,
    hole: Vec<bool>,
    wts: Vec<Vec<f32>>,
}

impl Filler {
    /// `None` when the mask has no hole (the fill is the identity then).
    fn new(w: usize, h: usize, mask: HoleMaskRef<'_>, is360: bool) -> Result<Option<Filler>> {
        if mask.len() != w * h {
            return Err(FxError::Mismatch(format!(
                "hole fill: mask has {} entries for an image of {}x{}",
                mask.len(),
                w,
                h
            )));
        }
        let hole: Vec<bool> = (0..w * h).map(|i| mask.is_hole(i)).collect();
        if !hole.iter().any(|&b| b) {
            return Ok(None);
        }
        if is360 {
            return Err(FxError::Unsupported(
                "hole fill with the 360 flag set: the wrapping blur and resample modes are not implemented".into(),
            ));
        }
        if w < 2 || h < 2 {
            return Err(FxError::Unsupported(format!(
                "hole fill of a {w}x{h} image: the pyramid resample divides by zero for a dimension of 1"
            )));
        }
        let geo = geometry(w, h);
        let mut wts: Vec<Vec<f32>> = Vec::with_capacity(geo.top + 1);
        wts.push(hole.iter().map(|&b| if b { 0.0f32 } else { 1.0f32 }).collect());
        for k in 1..=geo.top {
            let (pw, ph) = geo.dims[k - 1];
            let (cw, ch) = geo.dims[k];
            let mut tmp = wts[k - 1].clone();
            blur5(&mut tmp, pw, ph);
            wts.push(resample1(&tmp, pw, ph, cw, ch));
        }
        Ok(Some(Filler { geo, hole, wts }))
    }

    /// Fills one channel in place. `plane` holds the native units of the depth (0..255,
    /// 0..65535, any float).
    fn fill(&self, plane: &mut [f32]) {
        let geo = &self.geo;
        // load: the value image of level 0, holes zeroed
        let mut vals: Vec<Vec<f32>> = Vec::with_capacity(geo.top + 1);
        let mut v0 = plane.to_vec();
        for (v, &h) in v0.iter_mut().zip(&self.hole) {
            if h {
                *v = 0.0;
            }
        }
        vals.push(v0);
        // pull
        for k in 1..=geo.top {
            let (pw, ph) = geo.dims[k - 1];
            let (cw, ch) = geo.dims[k];
            let mut tmp = vals[k - 1].clone();
            blur5(&mut tmp, pw, ph);
            vals.push(resample1(&tmp, pw, ph, cw, ch));
        }
        // push
        let (tw, th) = geo.dims[geo.top];
        let mut sol = vec![0.0f32; tw * th];
        for level in (0..=geo.top).rev() {
            let (lw, lh) = geo.dims[level];
            combine(&mut sol, &vals[level], &self.wts[level]);
            blur_gaussian(&mut sol, lw, lh, 1, GAUSS_RADIUS);
            if level == 0 {
                combine(&mut sol, &vals[0], &self.wts[0]);
            } else {
                let (nw, nh) = geo.dims[level - 1];
                sol = resample1(&sol, lw, lh, nw, nh);
            }
        }
        plane.copy_from_slice(&sol);
    }
}

/// Radius of the Gaussian blur of the push step.
pub const GAUSS_RADIUS: f32 = 5.0;

/// `if (weight != 0.0) sol = value / weight`; a NaN weight is "not equal" and divides.
fn combine(sol: &mut [f32], value: &[f32], weight: &[f32]) {
    for ((s, &v), &w) in sol.iter_mut().zip(value).zip(weight) {
        if w != 0.0 {
            *s = v / w;
        }
    }
}

/// Write back rule for u8: `y = x + 0.5f32`, `0 > y` gives 0, `y > 255` gives 255, else the
/// truncation.
#[inline]
fn store_u8(x: f32) -> u8 {
    let y = x + 0.5f32;
    if 0.0 > y {
        0
    } else if y > 255.0 {
        255
    } else {
        cvtt(y) as u8
    }
}

/// Write back rule for u16: the same as [`store_u8`] with 65535 as the upper bound.
#[inline]
fn store_u16(x: f32) -> u16 {
    let y = x + 0.5f32;
    if 0.0 > y {
        0
    } else if y > 65535.0 {
        65535
    } else {
        cvtt(y) as u16
    }
}

/// Hole fill of a single float channel plane in native units (the f32 core of the hole fill,
/// without the depth conversions). `mask` has `w * h` entries.
pub fn fill_plane(plane: &mut [f32], w: usize, h: usize, mask: HoleMaskRef<'_>, is360: bool) -> Result<()> {
    if plane.len() != w * h {
        return Err(FxError::Mismatch("hole fill: plane length does not match its size".into()));
    }
    if let Some(f) = Filler::new(w, h, mask, is360)? {
        f.fill(plane);
    }
    Ok(())
}

/// Hole fill with the inverted coverage of a warp as the mask (the u8 mask variant). `holes` is
/// 255 where nothing was warped; see [`fill_holes_masked`]. Sizes that differ give
/// `FxError::Mismatch`.
pub fn fill_holes(img: &mut Image, holes: &Coverage, is360: bool) -> Result<()> {
    if holes.width != img.width || holes.height != img.height {
        return Err(FxError::Mismatch(format!(
            "hole fill: image {}x{} against holes {}x{}",
            img.width, img.height, holes.width, holes.height
        )));
    }
    fill_holes_masked(img, HoleMaskRef::U8(&holes.data), is360)
}

/// The hole fill of an engine image with a hole mask of u8, u16 or f32 type. Without a hole the
/// image is returned unchanged (the fill is the identity then).
pub fn fill_holes_masked(img: &mut Image, mask: HoleMaskRef<'_>, is360: bool) -> Result<()> {
    let (w, h) = (img.width, img.height);
    let Some(filler) = Filler::new(w, h, mask, is360)? else {
        return Ok(());
    };
    let n = w * h;
    let mut plane = vec![0.0f32; n];
    for c in 0..CHANNELS {
        match &mut img.samples {
            Samples::U8(s) => {
                for i in 0..n {
                    plane[i] = f32::from(s[i * CHANNELS + c]);
                }
                filler.fill(&mut plane);
                for i in 0..n {
                    s[i * CHANNELS + c] = store_u8(plane[i]);
                }
            }
            Samples::U16(s) => {
                for i in 0..n {
                    plane[i] = f32::from(s[i * CHANNELS + c]);
                }
                filler.fill(&mut plane);
                for i in 0..n {
                    s[i * CHANNELS + c] = store_u16(plane[i]);
                }
            }
            Samples::F32(s) => {
                for i in 0..n {
                    plane[i] = s[i * CHANNELS + c];
                }
                filler.fill(&mut plane);
                for i in 0..n {
                    s[i * CHANNELS + c] = plane[i];
                }
            }
        }
    }
    Ok(())
}

/// The hole fill applied to a two channel float flow image (Vectors In: decode with
/// `vectors::decode_vectors_with_holes`, fill, then `vectors::apply_vec_scale`). The mask has the
/// size of the flow.
pub fn fill_flow_holes(flow: &mut FlowField, mask: HoleMaskRef<'_>, is360: bool) -> Result<()> {
    let (w, h) = (flow.width, flow.height);
    if flow.data.len() != w * h * 2 {
        return Err(FxError::Mismatch("hole fill: flow data length does not match its size".into()));
    }
    let Some(filler) = Filler::new(w, h, mask, is360)? else {
        return Ok(());
    };
    let n = w * h;
    let mut plane = vec![0.0f32; n];
    for c in 0..2 {
        for i in 0..n {
            plane[i] = flow.data[i * 2 + c];
        }
        filler.fill(&mut plane);
        for i in 0..n {
            flow.data[i * 2 + c] = plane[i];
        }
    }
    Ok(())
}
