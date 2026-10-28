//! Device stages of `holefill.rs`: the pull-push fill of one plane (`fill_plane`), of an engine
//! image (`fill_holes`, `fill_holes_masked`) and of a two channel flow (`fill_flow_holes`), with
//! the 360 flag (`is360`, the wrapped passes).
//!
//! Precision contract: that of the `hf_`, `cv_` and `rs_` kernels in `holefill.rs`'s order. The
//! level sizes are `holefill::geometry` (integer, host). The weight pyramid is built once per
//! mask: level 0 from the hole flags, then per level `blur5` (two passes from the previous level
//! into a new buffer, which equals the CPU's blur of a copy) and the resample. Per channel: the
//! load, the same pull for the values, then the push from the top level, `combine`, the Gaussian
//! of radius 5 (taps `gaussian_kernel(5.0)`), at level 0 `combine` again, at a higher level the
//! resample into the level below; then the channel's write back. The identity case (no hole) is
//! decided on the host with the CPU's predicate, as is the refusal of a side of 1 with a hole.

use super::conv::convolve_on;
use super::sample::resample1_on;
use super::{check_samples, Backend, Launch, Session, StageError, Taps};
use crate::device::cl::Buffer;
use crate::device::image::{reject_nan, DeviceImage, DeviceSamples};
use crate::flow::FlowField;
use crate::holefill::{geometry, Geometry, HoleMaskRef, GAUSS_RADIUS, U8_EPS, WIDE_EPS};
use crate::image::{Image, Samples, CHANNELS};
use crate::warp::Coverage;

/// The hole mask on the backend, by type (255 / 65535 / 1.0 at a hole).
pub enum MaskSource<'a, B: Backend + ?Sized> {
    U8(&'a B::Buf<u8>),
    U16(&'a B::Buf<u16>),
    F32(&'a B::Buf<f32>),
}

/// The CPU's hole predicate (`HoleMaskRef::is_hole`): `!(eps > (double) m)`.
fn is_hole(mask: HoleMaskRef<'_>, i: usize) -> bool {
    match mask {
        HoleMaskRef::U8(m) => !(U8_EPS > f64::from(m[i])),
        HoleMaskRef::U16(m) => !(WIDE_EPS > f64::from(m[i])),
        HoleMaskRef::F32(m) => !(WIDE_EPS > f64::from(m[i])),
    }
}

/// The part of the fill that depends only on the mask, on the backend: the level sizes, the hole
/// flags and the weight pyramid (`wts[k]` for `k` in `0..=top`).
pub struct DeviceFiller<B: Backend + ?Sized> {
    pub geo: Geometry,
    pub hole: B::Buf<u8>,
    pub wts: Vec<B::Buf<f32>>,
    pub wrap: bool,
    taps5: Taps<B>,
    gauss: Taps<B>,
}

impl<B: Backend> DeviceFiller<B> {
    /// Builds the weight pyramid of a `w x h` mask resident on the backend. The caller has decided
    /// that the mask has a hole (else the fill is the identity) and `w, h >= 2`.
    pub fn new(b: &B, mask: MaskSource<'_, B>, w: usize, h: usize, wrap: bool) -> Result<DeviceFiller<B>, StageError> {
        if w < 2 || h < 2 {
            return Err(StageError::Unsupported(format!(
                "hole fill of a {w}x{h} image: the pyramid resample divides by zero for a dimension of 1"
            )));
        }
        let n = w * h;
        check_samples(n * CHANNELS, "hole fill")?;
        let geo = geometry(w, h);
        let hole = b.alloc::<u8>(n)?;
        match mask {
            MaskSource::U8(m) => b.launch(Launch::FlagsU8 { m, hole: &hole, eps: U8_EPS, n })?,
            MaskSource::U16(m) => b.launch(Launch::FlagsU16 { m, hole: &hole, eps: WIDE_EPS, n })?,
            MaskSource::F32(m) => b.launch(Launch::FlagsF32 { m, hole: &hole, eps: WIDE_EPS, n })?,
        }
        let taps5 = Taps::binomial5(b)?;
        let gauss = Taps::gaussian(b, GAUSS_RADIUS)?
            .ok_or_else(|| StageError::Unsupported("hole fill: the push radius is below the blur epsilon".into()))?;
        let w0 = b.alloc::<f32>(n)?;
        b.launch(Launch::Weight0 { hole: &hole, wt: &w0, n })?;
        let mut wts = vec![w0];
        for k in 1..=geo.top {
            let next = pull(b, &wts[k - 1], geo.dims[k - 1], geo.dims[k], wrap, &taps5)?;
            wts.push(next);
        }
        Ok(DeviceFiller { geo, hole, wts, wrap, taps5, gauss })
    }

    /// Fills one plane of `w * h` floats in place (`holefill`'s `Filler::fill`).
    pub fn fill(&self, b: &B, plane: &B::Buf<f32>) -> Result<(), StageError> {
        let geo = &self.geo;
        let (w0, h0) = geo.dims[0];
        let n0 = w0 * h0;
        // load
        let v0 = b.alloc::<f32>(n0)?;
        b.launch(Launch::Load { plane, hole: &self.hole, v0: &v0, n: n0 })?;
        let mut vals = vec![v0];
        // pull
        for k in 1..=geo.top {
            let next = pull(b, &vals[k - 1], geo.dims[k - 1], geo.dims[k], self.wrap, &self.taps5)?;
            vals.push(next);
        }
        // push
        let (tw, th) = geo.dims[geo.top];
        // the cleared top level solution (the top level holds a handful of samples)
        let mut sol = b.upload(&vec![0.0f32; tw * th])?;
        for level in (0..=geo.top).rev() {
            let (lw, lh) = geo.dims[level];
            let n = lw * lh;
            b.launch(Launch::Combine { sol: &sol, value: &vals[level], weight: &self.wts[level], n })?;
            let tmp = b.alloc::<f32>(n)?;
            convolve_on(b, &sol, &tmp, &sol, lw, lh, 1, &self.gauss, self.wrap)?;
            if level == 0 {
                b.launch(Launch::Combine { sol: &sol, value: &vals[0], weight: &self.wts[0], n })?;
            } else {
                let (nw, nh) = geo.dims[level - 1];
                let next = b.alloc::<f32>(nw * nh)?;
                resample1_on(b, &sol, lw, lh, &next, nw, nh, self.wrap)?;
                sol = next;
            }
        }
        b.copy(&sol, plane)
    }
}

/// One pull step: `blur5` of `src` (`dims_src`) into a new buffer, resampled to `dims_dst`.
fn pull<B: Backend>(b: &B, src: &B::Buf<f32>, (pw, ph): (usize, usize), (cw, ch): (usize, usize), wrap: bool, taps5: &Taps<B>) -> Result<B::Buf<f32>, StageError> {
    let n = pw * ph;
    let tmp = b.alloc::<f32>(n)?;
    let blurred = b.alloc::<f32>(n)?;
    convolve_on(b, src, &tmp, &blurred, pw, ph, 1, taps5, wrap)?;
    let out = b.alloc::<f32>(cw * ch)?;
    resample1_on(b, &blurred, pw, ph, &out, cw, ch, wrap)?;
    Ok(out)
}

/// Uploads a host mask.
fn upload_mask<B: Backend>(b: &B, mask: HoleMaskRef<'_>) -> Result<MaskBuf<B>, StageError> {
    Ok(match mask {
        HoleMaskRef::U8(m) => MaskBuf::U8(b.upload(m)?),
        HoleMaskRef::U16(m) => MaskBuf::U16(b.upload(m)?),
        HoleMaskRef::F32(m) => MaskBuf::F32(b.upload(m)?),
    })
}

enum MaskBuf<B: Backend + ?Sized> {
    U8(B::Buf<u8>),
    U16(B::Buf<u16>),
    F32(B::Buf<f32>),
}

impl<B: Backend> MaskBuf<B> {
    fn source(&self) -> MaskSource<'_, B> {
        match self {
            MaskBuf::U8(v) => MaskSource::U8(v),
            MaskBuf::U16(v) => MaskSource::U16(v),
            MaskBuf::F32(v) => MaskSource::F32(v),
        }
    }
}

/// `None` when the fill is the identity (no hole), else the filler of the host mask on the
/// backend; checks the mask size and the side of 1 as `Filler::new` does, and records G6 when the
/// wrapped passes run.
fn host_filler<B: Backend>(b: &B, w: usize, h: usize, mask: HoleMaskRef<'_>, is360: bool) -> Result<Option<DeviceFiller<B>>, StageError> {
    if mask.len() != w * h {
        return Err(StageError::Unsupported(format!("hole fill: mask has {} entries for an image of {w}x{h}", mask.len())));
    }
    if !(0..w * h).any(|i| is_hole(mask, i)) {
        return Ok(None);
    }
    if w < 2 || h < 2 {
        return Err(StageError::Unsupported(format!(
            "hole fill of a {w}x{h} image: the pyramid resample divides by zero for a dimension of 1"
        )));
    }
    check_samples(w * h * CHANNELS, "hole fill")?;
    if is360 {
        fx_core::gaps::record(&crate::gaps::G6);
    }
    let m = upload_mask(b, mask)?;
    DeviceFiller::new(b, m.source(), w, h, is360).map(Some)
}

/// `holefill::fill_plane` on the backend; the plane changes only on success.
pub fn fill_plane_host<B: Backend>(b: &B, plane: &mut [f32], w: usize, h: usize, mask: HoleMaskRef<'_>, is360: bool) -> Result<(), StageError> {
    if plane.len() != w * h {
        return Err(StageError::Unsupported("hole fill: plane length does not match its size".into()));
    }
    let Some(f) = host_filler(b, w, h, mask, is360)? else {
        return Ok(());
    };
    let buf = b.upload(plane)?;
    f.fill(b, &buf)?;
    let out = b.download(&buf)?;
    reject_nan(&out)?;
    plane.copy_from_slice(&out);
    Ok(())
}

/// `holefill::fill_plane` on the device.
pub fn fill_plane_gpu(s: &Session, plane: &mut [f32], w: usize, h: usize, mask: HoleMaskRef<'_>, is360: bool) -> Result<(), StageError> {
    fill_plane_host(s, plane, w, h, mask, is360)
}

/// Fills a device plane in place with a filler built by [`DeviceFiller::new`].
pub fn fill_plane_enqueue(s: &Session, filler: &DeviceFiller<Session>, plane: &Buffer<f32>) -> Result<(), StageError> {
    filler.fill(s, plane)
}

/// The interleaved samples of an image or flow on the backend.
pub enum Interleaved<'a, B: Backend + ?Sized> {
    U8(&'a B::Buf<u8>),
    U16(&'a B::Buf<u16>),
    F32(&'a B::Buf<f32>),
}

impl<B: Backend + ?Sized> Clone for Interleaved<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B: Backend + ?Sized> Copy for Interleaved<'_, B> {}

/// Fills every channel of an interleaved buffer of `channels` channels in place: extract, fill,
/// write back, channel by channel in order (`fill_holes_masked`, `fill_flow_holes`).
pub fn fill_interleaved_on<B: Backend>(b: &B, filler: &DeviceFiller<B>, data: Interleaved<'_, B>, channels: usize) -> Result<(), StageError> {
    let (w, h) = filler.geo.dims[0];
    let n = w * h;
    let plane = b.alloc::<f32>(n)?;
    for c in 0..channels {
        match data {
            Interleaved::U8(img) => {
                b.launch(Launch::ExtractU8 { img, plane: &plane, stride: channels, c, n })?;
                filler.fill(b, &plane)?;
                b.launch(Launch::StoreU8 { plane: &plane, img, stride: channels, c, n })?;
            }
            Interleaved::U16(img) => {
                b.launch(Launch::ExtractU16 { img, plane: &plane, stride: channels, c, n })?;
                filler.fill(b, &plane)?;
                b.launch(Launch::StoreU16 { plane: &plane, img, stride: channels, c, n })?;
            }
            Interleaved::F32(img) => {
                b.launch(Launch::ExtractF32 { img, plane: &plane, stride: channels, c, n })?;
                filler.fill(b, &plane)?;
                b.launch(Launch::StoreF32 { plane: &plane, img, stride: channels, c, n })?;
            }
        }
    }
    Ok(())
}

/// `holefill::fill_holes_masked` of a [`DeviceImage`] in place, with a filler of its size.
pub fn fill_image_enqueue(s: &Session, filler: &DeviceFiller<Session>, img: &DeviceImage) -> Result<(), StageError> {
    if filler.geo.dims[0] != (img.width, img.height) {
        return Err(StageError::Unsupported("hole fill: the filler and the image differ in size".into()));
    }
    let data = match &img.samples {
        DeviceSamples::U8(v) => Interleaved::U8(v),
        DeviceSamples::U16(v) => Interleaved::U16(v),
        DeviceSamples::F32(v) => Interleaved::F32(v),
    };
    fill_interleaved_on(s, filler, data, CHANNELS)
}

/// `holefill::fill_flow_holes` of a device flow (`w * h * 2` floats) in place.
pub fn fill_flow_enqueue(s: &Session, filler: &DeviceFiller<Session>, flow: &Buffer<f32>) -> Result<(), StageError> {
    fill_interleaved_on(s, filler, Interleaved::F32(flow), 2)
}

/// `holefill::fill_holes_masked` on the backend; the image changes only on success.
pub fn fill_holes_masked_host<B: Backend>(b: &B, img: &mut Image, mask: HoleMaskRef<'_>, is360: bool) -> Result<(), StageError> {
    let (w, h) = (img.width, img.height);
    let Some(f) = host_filler(b, w, h, mask, is360)? else {
        return Ok(());
    };
    if img.sample_count() != w * h * CHANNELS {
        return Err(StageError::Unsupported("hole fill: the image holds another sample count than its size".into()));
    }
    let samples = match &img.samples {
        Samples::U8(v) => {
            let buf = b.upload(v)?;
            fill_interleaved_on(b, &f, Interleaved::U8(&buf), CHANNELS)?;
            Samples::U8(b.download(&buf)?)
        }
        Samples::U16(v) => {
            let buf = b.upload(v)?;
            fill_interleaved_on(b, &f, Interleaved::U16(&buf), CHANNELS)?;
            Samples::U16(b.download(&buf)?)
        }
        Samples::F32(v) => {
            let buf = b.upload(v)?;
            fill_interleaved_on(b, &f, Interleaved::F32(&buf), CHANNELS)?;
            let out = b.download(&buf)?;
            reject_nan(&out)?;
            Samples::F32(out)
        }
    };
    img.samples = samples;
    Ok(())
}

/// `holefill::fill_holes_masked` on the device.
pub fn fill_holes_masked_gpu(s: &Session, img: &mut Image, mask: HoleMaskRef<'_>, is360: bool) -> Result<(), StageError> {
    fill_holes_masked_host(s, img, mask, is360)
}

/// `holefill::fill_holes` on the backend (the u8 coverage holes).
pub fn fill_holes_host<B: Backend>(b: &B, img: &mut Image, holes: &Coverage, is360: bool) -> Result<(), StageError> {
    if holes.width != img.width || holes.height != img.height {
        return Err(StageError::Unsupported(format!(
            "hole fill: image {}x{} against holes {}x{}",
            img.width, img.height, holes.width, holes.height
        )));
    }
    fill_holes_masked_host(b, img, HoleMaskRef::U8(&holes.data), is360)
}

/// `holefill::fill_holes` on the device.
pub fn fill_holes_gpu(s: &Session, img: &mut Image, holes: &Coverage, is360: bool) -> Result<(), StageError> {
    fill_holes_host(s, img, holes, is360)
}

/// `holefill::fill_flow_holes` on the backend; the flow changes only on success.
pub fn fill_flow_holes_host<B: Backend>(b: &B, flow: &mut FlowField, mask: HoleMaskRef<'_>, is360: bool) -> Result<(), StageError> {
    let (w, h) = (flow.width, flow.height);
    if flow.data.len() != w * h * 2 {
        return Err(StageError::Unsupported("hole fill: flow data length does not match its size".into()));
    }
    let Some(f) = host_filler(b, w, h, mask, is360)? else {
        return Ok(());
    };
    let buf = b.upload(&flow.data)?;
    fill_interleaved_on(b, &f, Interleaved::F32(&buf), 2)?;
    let out = b.download(&buf)?;
    reject_nan(&out)?;
    flow.data.copy_from_slice(&out);
    Ok(())
}

/// `holefill::fill_flow_holes` on the device.
pub fn fill_flow_holes_gpu(s: &Session, flow: &mut FlowField, mask: HoleMaskRef<'_>, is360: bool) -> Result<(), StageError> {
    fill_flow_holes_host(s, flow, mask, is360)
}
