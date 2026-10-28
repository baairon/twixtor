//! Device stages of the solver's pyramids: `prep::luma`, the reduction `prep::reduce_with`, the
//! band pass levels of `flow::build_pyramid`, the steps of `MaskPyramid::build` and
//! `flow::upsample_flow`.
//!
//! Precision contract: that of the `pm_`, `cv_` and `rs_` kernels, in the CPU's order: a
//! reduction stage is `blur5` (two passes) then the resample to `(w / 2, h / 2)`; a band level is
//! `tmp = blur5(orig)`, the next level `resample(tmp)` and `band = orig - tmp`; a mask step is
//! `blur5`, resample, binarize. Delinearize (an f64 `powf` per pixel) and Contrast / Edge Enhance
//! (the solver's gradient) run on the host in [`build_pyramid_gpu`], between the device steps.

use super::conv::convolve_on;
use super::sample::resample1_on;
use super::{check_len, check_samples, Backend, Launch, Session, StageError, Taps};
use crate::device::cl::Buffer;
use crate::device::image::{reject_nan, DeviceImage, DeviceSamples};
use crate::flow::{FlowGeometry, FlowSettings};
use crate::image::{Image, Samples};
use crate::matte::Matte;
use crate::prep::{self, Plane};

/// A plane on the backend: its buffer and size.
pub struct DevicePlane<B: Backend + ?Sized> {
    pub buf: B::Buf<f32>,
    pub width: usize,
    pub height: usize,
}

impl<B: Backend> DevicePlane<B> {
    pub fn upload(b: &B, p: &Plane) -> Result<DevicePlane<B>, StageError> {
        if p.data.len() != p.width * p.height {
            return Err(StageError::Unsupported(format!("a {}x{} plane holds {} values", p.width, p.height, p.data.len())));
        }
        check_samples(p.data.len(), "plane")?;
        Ok(DevicePlane { buf: b.upload(&p.data)?, width: p.width, height: p.height })
    }

    /// The plane back on the host; a NaN is `StageError::NanOutput`.
    pub fn download(&self, b: &B) -> Result<Plane, StageError> {
        let data = b.download(&self.buf)?;
        reject_nan(&data)?;
        Ok(Plane { width: self.width, height: self.height, data })
    }
}

/// `blur5` (or `blur5_wrapx`) of `p` into a new plane.
fn blur5_new<B: Backend>(b: &B, p: &DevicePlane<B>, taps5: &Taps<B>, wrap: bool) -> Result<DevicePlane<B>, StageError> {
    let n = p.width * p.height;
    let tmp = b.alloc::<f32>(n)?;
    let out = b.alloc::<f32>(n)?;
    convolve_on(b, &p.buf, &tmp, &out, p.width, p.height, 1, taps5, wrap)?;
    Ok(DevicePlane { buf: out, width: p.width, height: p.height })
}

/// `resample` (or `resample_with` and the flag) of `p` to `(dw, dh)` into a new plane.
fn resample_new<B: Backend>(b: &B, p: &DevicePlane<B>, dw: usize, dh: usize, wrap: bool) -> Result<DevicePlane<B>, StageError> {
    check_samples(dw * dh, "resample")?;
    let out = b.alloc::<f32>(dw * dh)?;
    resample1_on(b, &p.buf, p.width, p.height, &out, dw, dh, wrap)?;
    Ok(DevicePlane { buf: out, width: dw, height: dh })
}

// ---- luma ---------------------------------------------------------------------------------------

/// `prep::luma` of a device image into `out` (`width * height` floats). Video range together with
/// Track Alpha is `StageError::Unsupported`, as on the CPU.
pub fn luma_on<B: Backend>(
    b: &B,
    samples: LumaSource<'_, B>,
    n: usize,
    out: &B::Buf<f32>,
    track_alpha: bool,
    video: bool,
) -> Result<(), StageError> {
    if video && track_alpha {
        return Err(StageError::Unsupported("optical flow luma: video range conversion together with Track RGB+A".into()));
    }
    check_samples(n * 4, "luma")?;
    check_len(b, out, n, "luma")?;
    match samples {
        LumaSource::U8(img) => b.launch(Launch::LumaU8 { img, out, track_alpha, video, n }),
        LumaSource::U16(img) => b.launch(Launch::LumaU16 { img, out, track_alpha, video, n }),
        LumaSource::F32(img) => b.launch(Launch::LumaF32 { img, out, track_alpha, video, n }),
    }
}

/// The interleaved A, R, G, B samples a luma launch reads.
pub enum LumaSource<'a, B: Backend + ?Sized> {
    U8(&'a B::Buf<u8>),
    U16(&'a B::Buf<u16>),
    F32(&'a B::Buf<f32>),
}

/// `prep::luma` of a [`DeviceImage`] into a new device plane.
pub fn luma_enqueue(s: &Session, img: &DeviceImage, track_alpha: bool, video: bool) -> Result<DevicePlane<Session>, StageError> {
    let n = img.width * img.height;
    check_samples(n, "luma")?;
    let out = s.alloc::<f32>(n)?;
    let src = match &img.samples {
        DeviceSamples::U8(v) => LumaSource::U8(v),
        DeviceSamples::U16(v) => LumaSource::U16(v),
        DeviceSamples::F32(v) => LumaSource::F32(v),
    };
    luma_on(s, src, n, &out, track_alpha, video)?;
    Ok(DevicePlane { buf: out, width: img.width, height: img.height })
}

/// `prep::luma` of a host image on the backend, as a device plane.
pub fn luma_upload<B: Backend>(b: &B, img: &Image, track_alpha: bool, video: bool) -> Result<DevicePlane<B>, StageError> {
    let n = img.width * img.height;
    check_samples(n, "luma")?;
    if img.sample_count() != n * 4 {
        return Err(StageError::Unsupported("luma: the image holds another sample count than its size".into()));
    }
    let out = b.alloc::<f32>(n)?;
    match &img.samples {
        Samples::U8(v) => {
            let buf = b.upload(v)?;
            luma_on(b, LumaSource::U8(&buf), n, &out, track_alpha, video)?;
        }
        Samples::U16(v) => {
            let buf = b.upload(v)?;
            luma_on(b, LumaSource::U16(&buf), n, &out, track_alpha, video)?;
        }
        Samples::F32(v) => {
            let buf = b.upload(v)?;
            luma_on(b, LumaSource::F32(&buf), n, &out, track_alpha, video)?;
        }
    }
    Ok(DevicePlane { buf: out, width: img.width, height: img.height })
}

/// `prep::luma` on the backend (an image without pixels gives the CPU's empty plane).
pub fn luma_host<B: Backend>(b: &B, img: &Image, track_alpha: bool, video: bool) -> Result<Plane, StageError> {
    if video && track_alpha {
        return Err(StageError::Unsupported("optical flow luma: video range conversion together with Track RGB+A".into()));
    }
    if img.width == 0 || img.height == 0 {
        return Ok(Plane::new(img.width, img.height));
    }
    luma_upload(b, img, track_alpha, video)?.download(b)
}

/// `prep::luma` on the device.
pub fn luma_gpu(s: &Session, img: &Image, track_alpha: bool, video: bool) -> Result<Plane, StageError> {
    luma_host(s, img, track_alpha, video)
}

// ---- reduction and band pass levels -------------------------------------------------------------

/// `prep::reduce_with`: `r` stages of `blur5` then the resample to the halved size. A size below
/// 2x2 before a stage is `StageError::Unsupported` (the sampler needs 2x2).
pub fn reduce_on<B: Backend>(b: &B, t: DevicePlane<B>, r: u32, wrap: bool, taps5: &Taps<B>) -> Result<DevicePlane<B>, StageError> {
    let mut t = t;
    for _ in 0..r {
        let blurred = blur5_new(b, &t, taps5, wrap)?;
        let (nw, nh) = (t.width / 2, t.height / 2);
        if nw == 0 || nh == 0 {
            return Err(StageError::Unsupported(format!("reduce: a {}x{} stage halves to an empty plane", t.width, t.height)));
        }
        t = resample_new(b, &blurred, nw, nh, wrap)?;
    }
    Ok(t)
}

/// `prep::reduce_with` of a device plane.
pub fn reduce_enqueue(s: &Session, t: DevicePlane<Session>, r: u32, wrap: bool, taps5: &Taps<Session>) -> Result<DevicePlane<Session>, StageError> {
    reduce_on(s, t, r, wrap, taps5)
}

/// `prep::reduce_with` of a host plane on the backend.
pub fn reduce_host<B: Backend>(b: &B, t: &Plane, r: u32, wrap: bool) -> Result<Plane, StageError> {
    if r == 0 {
        return Ok(t.clone());
    }
    let taps5 = Taps::binomial5(b)?;
    let p = DevicePlane::upload(b, t)?;
    reduce_on(b, p, r, wrap, &taps5)?.download(b)
}

/// `prep::reduce_with` on the device.
pub fn reduce_gpu(s: &Session, t: Plane, r: u32, wrap: bool) -> Result<Plane, StageError> {
    reduce_host(s, &t, r, wrap)
}

/// The band pass levels of `flow::build_pyramid` from its level 0 (after Delinearize and Edge
/// Enhance): for `i` in `0..levels`, `tmp = blur5(orig_i)`, `orig_{i+1} = resample(tmp)` to
/// `sizes[i + 1]`, `band_i = orig_i - tmp` (in place on `orig_i`). Finest first.
pub fn bands_on<B: Backend>(b: &B, level0: DevicePlane<B>, sizes: &[(usize, usize)], wrap: bool, taps5: &Taps<B>) -> Result<Vec<DevicePlane<B>>, StageError> {
    let n = sizes.len();
    let mut orig = level0;
    let mut bands = Vec::with_capacity(n);
    for i in 0..n {
        let tmp = blur5_new(b, &orig, taps5, wrap)?;
        let next = if i + 1 < n {
            let (nw, nh) = sizes[i + 1];
            Some(resample_new(b, &tmp, nw, nh, wrap)?)
        } else {
            None
        };
        b.launch(Launch::Subtract { a: &orig.buf, b: &tmp.buf, n: orig.width * orig.height })?;
        bands.push(orig);
        match next {
            Some(p) => orig = p,
            None => break,
        }
    }
    Ok(bands)
}

/// The band pass levels of a device level 0 (see [`bands_on`]).
pub fn bands_enqueue(s: &Session, level0: DevicePlane<Session>, geometry: &FlowGeometry, wrap: bool, taps5: &Taps<Session>) -> Result<Vec<DevicePlane<Session>>, StageError> {
    bands_on(s, level0, &geometry.sizes[..geometry.levels], wrap, taps5)
}

/// `flow::build_pyramid` on the backend: the band pass planes, finest first (the CPU's
/// `Pyramid::level(l)` for every level; `geometry` is `settings.geometry` of the image size).
/// Luma, the reduction and the levels run on the backend, Delinearize and Edge Enhance on the
/// host between them.
pub fn build_pyramid_host<B: Backend>(b: &B, img: &Image, settings: &FlowSettings) -> Result<Vec<Plane>, StageError> {
    let geometry = settings.geometry(img.width, img.height).map_err(|e| StageError::Unsupported(e.to_string()))?;
    let wrap = settings.wrap;
    let taps5 = Taps::binomial5(b)?;
    let luma = luma_upload(b, img, settings.track_alpha, settings.video_range_luma)?;
    let mut level0 = reduce_on(b, luma, geometry.reduction, wrap, &taps5)?;
    if settings.prep_delinearize || settings.prep_contrast_edge {
        let mut p = level0.download(b)?;
        if settings.prep_delinearize {
            prep::delinearize(&mut p);
        }
        if settings.prep_contrast_edge {
            prep::edge_enhance_with(&mut p, wrap);
        }
        level0 = DevicePlane::upload(b, &p)?;
    }
    let bands = bands_on(b, level0, &geometry.sizes[..geometry.levels], wrap, &taps5)?;
    bands.iter().map(|p| p.download(b)).collect()
}

/// `flow::build_pyramid` on the device (see [`build_pyramid_host`]).
pub fn build_pyramid_gpu(s: &Session, img: &Image, settings: &FlowSettings) -> Result<Vec<Plane>, StageError> {
    build_pyramid_host(s, img, settings)
}

// ---- mask pyramid -------------------------------------------------------------------------------

/// One step of the mask pyramid: `blur5`, resample to `(nw, nh)`, binarize.
fn mask_step<B: Backend>(b: &B, p: &DevicePlane<B>, nw: usize, nh: usize, wrap: bool, taps5: &Taps<B>) -> Result<DevicePlane<B>, StageError> {
    let t = blur5_new(b, p, taps5, wrap)?;
    if nw == 0 || nh == 0 {
        return Err(StageError::Unsupported(format!("mask pyramid: a {}x{} step halves to an empty plane", p.width, p.height)));
    }
    let r = resample_new(b, &t, nw, nh, wrap)?;
    b.launch(Launch::Binarize { p: &r.buf, n: nw * nh })?;
    Ok(r)
}

/// The levels of `MaskPyramid::build` from a device matte at the output size (binarized in
/// place first), finest first: the reduction steps to the working size, then one step per level.
pub fn mask_pyramid_on<B: Backend>(b: &B, m: DevicePlane<B>, geometry: &FlowGeometry, wrap: bool, taps5: &Taps<B>) -> Result<Vec<DevicePlane<B>>, StageError> {
    let mut cur = m;
    b.launch(Launch::Binarize { p: &cur.buf, n: cur.width * cur.height })?;
    for _ in 0..geometry.reduction {
        let (nw, nh) = (cur.width / 2, cur.height / 2);
        cur = mask_step(b, &cur, nw, nh, wrap, taps5)?;
    }
    let n = geometry.levels;
    let mut levels = Vec::with_capacity(n);
    for i in 0..n {
        let next = if i + 1 < n {
            let (nw, nh) = geometry.sizes[i + 1];
            Some(mask_step(b, &cur, nw, nh, wrap, taps5)?)
        } else {
            None
        };
        levels.push(cur);
        match next {
            Some(p) => cur = p,
            None => break,
        }
    }
    Ok(levels)
}

/// `MaskPyramid::build`'s levels of a device matte (see [`mask_pyramid_on`]).
pub fn mask_pyramid_enqueue(s: &Session, m: DevicePlane<Session>, geometry: &FlowGeometry, wrap: bool, taps5: &Taps<Session>) -> Result<Vec<DevicePlane<Session>>, StageError> {
    mask_pyramid_on(s, m, geometry, wrap, taps5)
}

/// `MaskPyramid::build` on the backend: the level planes, finest first (`MaskPyramid::level`).
pub fn mask_pyramid_host<B: Backend>(b: &B, m: &Matte, settings: &FlowSettings) -> Result<Vec<Vec<f32>>, StageError> {
    if m.data.len() != m.width * m.height {
        return Err(StageError::Unsupported(format!("mask pyramid: a {}x{} matte holds {} values", m.width, m.height, m.data.len())));
    }
    let geometry = settings.geometry(m.width, m.height).map_err(|e| StageError::Unsupported(e.to_string()))?;
    let taps5 = Taps::binomial5(b)?;
    let p = DevicePlane::upload(b, &Plane { width: m.width, height: m.height, data: m.data.clone() })?;
    let levels = mask_pyramid_on(b, p, &geometry, settings.wrap, &taps5)?;
    levels.iter().map(|l| l.download(b).map(|p| p.data)).collect()
}

/// `MaskPyramid::build` on the device (see [`mask_pyramid_host`]).
pub fn mask_pyramid_gpu(s: &Session, m: &Matte, settings: &FlowSettings) -> Result<Vec<Vec<f32>>, StageError> {
    mask_pyramid_host(s, m, settings)
}

// ---- flow upsample ------------------------------------------------------------------------------

/// `[sx, sy, rx, ry]` of `flow::upsample_flow`: `sx = ((f32) dw - 1.0) / (f32)(sw - 1)`,
/// `rx = (f32)(1.0 / (f64) sx)`, the same for y.
pub fn upsample_scales(sw: usize, sh: usize, dw: usize, dh: usize) -> [f32; 4] {
    let sx = (dw as f32 - 1.0f32) / ((sw as i32 - 1) as f32);
    let sy = (dh as f32 - 1.0f32) / ((sh as i32 - 1) as f32);
    let rx = (1.0f64 / (sx as f64)) as f32;
    let ry = (1.0f64 / (sy as f64)) as f32;
    [sx, sy, rx, ry]
}

/// `flow::upsample_flow` from a device flow of `sw x sh` vectors into one of `dw x dh`.
#[allow(clippy::too_many_arguments)]
pub fn upsample_flow_on<B: Backend>(b: &B, src: &B::Buf<f32>, sw: usize, sh: usize, dst: &B::Buf<f32>, dw: usize, dh: usize) -> Result<(), StageError> {
    if sw < 2 || sh < 2 {
        return Err(StageError::Unsupported(format!("flow upsample: a {sw}x{sh} source (the sampler needs 2x2)")));
    }
    check_samples(sw * sh * 2, "flow upsample")?;
    check_samples(dw * dh * 2, "flow upsample")?;
    check_len(b, src, sw * sh * 2, "flow upsample")?;
    check_len(b, dst, dw * dh * 2, "flow upsample")?;
    b.launch(Launch::Upsample { src, sw, sh, dst, dw, dh, scale: upsample_scales(sw, sh, dw, dh) })
}

/// `flow::upsample_flow` on device flows.
pub fn upsample_flow_enqueue(s: &Session, src: &Buffer<f32>, sw: usize, sh: usize, dst: &Buffer<f32>, dw: usize, dh: usize) -> Result<(), StageError> {
    upsample_flow_on(s, src, sw, sh, dst, dw, dh)
}

/// `flow::upsample_flow` of a host flow on the backend.
pub fn upsample_flow_host<B: Backend>(b: &B, src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Result<Vec<f32>, StageError> {
    if dw == 0 || dh == 0 {
        return Ok(Vec::new());
    }
    if Some(src.len()) != sw.checked_mul(sh).and_then(|v| v.checked_mul(2)) {
        return Err(StageError::Unsupported(format!("flow upsample: {} values for {sw}x{sh}", src.len())));
    }
    if sw < 2 || sh < 2 {
        return Err(StageError::Unsupported(format!("flow upsample: a {sw}x{sh} source (the sampler needs 2x2)")));
    }
    let sb = b.upload(src)?;
    let db = b.alloc::<f32>(dw * dh * 2)?;
    upsample_flow_on(b, &sb, sw, sh, &db, dw, dh)?;
    let out = b.download(&db)?;
    reject_nan(&out)?;
    Ok(out)
}

/// `flow::upsample_flow` on the device.
pub fn upsample_flow_gpu(s: &Session, src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Result<Vec<f32>, StageError> {
    upsample_flow_host(s, src, sw, sh, dw, dh)
}
