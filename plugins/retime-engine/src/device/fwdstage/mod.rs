//! Device stage of the forward warp: `forward::forward_warp_with` with the mesh on
//! the host and the rasteriser, the pyramid and the running mean on the device, on the kernels of
//! `kernels::fwdstage` (prefix `fw_`), the separable passes of `kernels::conv`, the integer
//! stores of `kernels::wbstage` and the sampler of `kernels::warpstage`.
//!
//! Entry points:
//! * [`plan`]: the host side, shared with the device-free tests: the CPU's argument checks (every
//!   argument the CPU refuses is `StageError::Unsupported`), the early copy out and the mesh
//!   (`forward::mesh::positions` and `mip_levels`, computed by the CPU's own functions: the mip
//!   level's `logf` takes its bits from the host C runtime, so it never runs on the device).
//! * [`forward_warp_enqueue`]`(session, plan, src, out, cov, flag)` queues the stage on device
//!   images. It moves no image data between host and device apart from the mesh it is given,
//!   and reads back two scalars in the middle (the refusal word of the counting pass and the hit
//!   total, which sizes the record buffer); a refusal found there is returned as
//!   `StageError::Unsupported`, one found later raises `flag` ([`Flag::check`] after the chain).
//! * [`forward_warp_with_gpu`], [`forward_warp_gpu`], [`forward_warp_coverage_gpu`]: the CPU
//!   functions' arguments and results, with a [`StageError`] in place of the error; `out` is
//!   written only on success. An f32 result holding a NaN is `StageError::NanOutput`.
//!
//! The order of the fold: the CPU folds the hits of a pixel triangle by triangle in mesh order.
//! The device scan converts every triangle on its own (the CPU's arithmetic, so the same hits
//! with the same `u`, `v`, `d` bits), stores each hit as a record keyed by its triangle index in
//! its pixel's list, sorts each list by the key and folds it in that order: the CPU's fold for
//! every pixel, whatever order the work items ran in.
//!
//! Precision contract (V: `forward::forward_warp_with` is the reference and stays untouched):
//! `kernels::fwdstage` states it per kernel. Refusals (the executor reruns the warp on the CPU):
//! images wider or taller than `FW_MAX_DIM` or of more than `cl::MAX_WORK_ITEMS` pixels; a
//! triangle of more than `FW_TRI_CAP` hits; a pixel of more than `FW_PIXEL_CAP` hits; more than
//! `FW_RECORDS_PER_PIXEL` hits per pixel on average; a texture tap at a negative index; the
//! degenerate row write outside the image (the CPU refuses it too); the Gouraud variant
//! (`ForwardOpts { pyramid: false }`, `forward::gouraud`, vectors-in fix 02), which has no kernel
//! (the executor keeps it on the CPU before the device is asked, `exec::Exec::forward_warp_with`).

#[cfg(test)]
mod devtests;
#[cfg(test)]
pub(crate) mod sim;
#[cfg(test)]
mod tests;

use super::cl::{Access, Arg, Buffer, Range, MAX_WORK_ITEMS};
use super::convstage::{Backend, Launch, Taps};
use super::image::{has_nan, DeviceImage, DeviceSamples, StageError};
use super::kernels::{self, fwdstage as k};
use super::selfcheck::Session;
use super::warpstage::{flag_result, suffix, DeviceCoverage, Flag};
use super::wbstage;
use crate::flow::FlowField;
use crate::forward::{self, mesh, pyramid, ForwardOpts};
use crate::image::{Fmt, Image, CHANNELS};
use crate::warp::Coverage;

/// The kernel sources this stage needs, in program order (after the prelude): the parts of the
/// Weighted Blend stages (the convolution passes, the warp samplers and stores, the integer blur
/// stores) and [`k::FORWARD`].
pub fn program_parts() -> Vec<&'static str> {
    let mut v = wbstage::program_parts();
    v.push(k::FORWARD);
    v
}

/// The prelude and [`program_parts`].
pub fn program() -> String {
    kernels::program(&program_parts())
}

/// The mesh of one warp, computed on the host by the CPU's functions.
#[derive(Clone, Debug)]
pub struct HostMesh {
    pub width: usize,
    pub height: usize,
    /// `x', y'` of every vertex, row major (`forward::mesh::positions`).
    pub pos: Vec<f32>,
    /// The mip level of every vertex (`forward::mesh::mip_levels`); `None` when the pyramid has
    /// one level (the CPU then uses `d = 0`).
    pub levels: Option<Vec<f32>>,
    /// Levels of the pyramid, level 0 included (`forward::pyramid::level_count`).
    pub count: usize,
}

/// What the warp does for these arguments.
#[derive(Clone, Debug)]
pub enum Plan {
    /// The early out: a copy of the source and a coverage of 255.
    Copy { width: usize, height: usize },
    /// The rasterised warp of this mesh.
    Warp(HostMesh),
}

/// The CPU's checks, early out and mesh for `forward_warp_with(out, src, flow, amount, opts, _)`,
/// plus the device limits. `out` is only checked for its shape.
pub fn plan(out: &Image, src: &Image, flow: &FlowField, amount: f32, opts: ForwardOpts) -> Result<Plan, StageError> {
    if !out.same_shape(src) {
        return Err(StageError::Unsupported("forward warp: destination and source of different shapes".into()));
    }
    if src.width < 2 || src.height < 2 || flow.width < 2 || flow.height < 2 {
        return Err(StageError::Unsupported("forward warp: an image or a flow below 2x2".into()));
    }
    if flow.data.len() != flow.width * flow.height * 2 {
        return Err(StageError::Unsupported("forward warp: flow data length does not match its size".into()));
    }
    let (w, h) = (src.width, src.height);
    let a = if 0.0f32 > amount { -amount } else { amount };
    if 1e-05f64 > (a as f64) || forward::flow_is_zero(flow) {
        return Ok(Plan::Copy { width: w, height: h });
    }
    if !opts.pyramid {
        return Err(StageError::Unsupported("forward warp without a pyramid (the Gouraud variant has no device kernel)".into()));
    }
    if w > k::FW_MAX_DIM || h > k::FW_MAX_DIM || w * h > MAX_WORK_ITEMS {
        return Err(StageError::Unsupported(format!("forward warp: {w}x{h} is above the device bound")));
    }
    let grid = mesh::Grid::new(w, h, flow, amount);
    let pos: Vec<f32> = mesh::positions(&grid, flow).into_iter().flatten().collect();
    let count = pyramid::level_count(w, h);
    let levels = if count > 1 { Some(mesh::mip_levels(&grid, flow)) } else { None };
    Ok(Plan::Warp(HostMesh { width: w, height: h, pos, levels, count }))
}

/// The pyramid level sizes of a `w x h` source (`Pyramid::build`), level 0 first.
pub fn level_dims(w: usize, h: usize, count: usize) -> Vec<(usize, usize)> {
    let mut dims = vec![(w, h)];
    let (mut cw, mut ch) = (w.max(2), h.max(2));
    for _ in 1..count {
        cw /= 2;
        ch /= 2;
        if cw < 2 {
            cw = 2;
        }
        if ch < 2 {
            ch = 2;
        }
        dims.push((cw, ch));
    }
    dims
}

/// The level table of the fold kernels: `[0, w0, h0, off1, w1, h1, ...]` (offsets in samples of
/// the buffer holding levels 1..), and that buffer's length (at least 1).
pub fn level_table(dims: &[(usize, usize)]) -> Result<(Vec<i32>, usize), StageError> {
    let mut lt = Vec::with_capacity(dims.len() * 3);
    let mut off = 0usize;
    for (i, &(lw, lh)) in dims.iter().enumerate() {
        let o = if i == 0 { 0 } else { off };
        lt.push(side(o)?);
        lt.push(side(lw)?);
        lt.push(side(lh)?);
        if i > 0 {
            off += lw * lh * CHANNELS;
        }
    }
    Ok((lt, off.max(1)))
}

/// The resample scales of one level step (`pyramid::resample_plane`).
pub fn resample_scales(sw: usize, sh: usize, dw: usize, dh: usize) -> (f32, f32) {
    let sx = ((sw as i32) as f32 - 1.0f32) / ((dw as i32 - 1) as f32);
    let sy = ((sh as i32) as f32 - 1.0f32) / ((dh as i32 - 1) as f32);
    (sx, sy)
}

/// The launch bands of the triangle passes: `(first key, count)`, at most 2^20 triangles each.
pub fn triangle_bands(w: usize, h: usize) -> Vec<(u32, u32)> {
    let ntri = 2 * (w - 1) * (h - 1);
    let band = 1usize << 20;
    (0..ntri).step_by(band).map(|k0| (k0 as u32, (ntri - k0).min(band) as u32)).collect()
}

/// The launch bands of the fold: `(first row, rows)`, about 2^20 pixels each.
pub fn row_bands(w: usize, h: usize) -> Vec<(usize, usize)> {
    let rows = ((1usize << 20) / w).max(1);
    (0..h).step_by(rows).map(|y0| (y0, (h - y0).min(rows))).collect()
}

/// The record budget of a `w x h` warp.
pub fn max_records(w: usize, h: usize) -> u32 {
    (w * h * k::FW_RECORDS_PER_PIXEL).min(u32::MAX as usize / 2) as u32
}

/// The mesh on the device.
pub struct DeviceMesh {
    pub width: usize,
    pub height: usize,
    pub pos: Buffer<f32>,
    /// The mip levels, or one unused element when the pyramid has one level.
    pub levels: Buffer<f32>,
    pub has_levels: bool,
    pub count: usize,
}

impl DeviceMesh {
    pub fn upload(s: &Session, m: &HostMesh) -> Result<DeviceMesh, StageError> {
        let pos = Buffer::from_slice(&s.context, &m.pos, Access::ReadOnly)?;
        let (levels, has_levels) = match &m.levels {
            Some(d) => (Buffer::from_slice(&s.context, d, Access::ReadOnly)?, true),
            None => (Buffer::from_slice(&s.context, &[0.0f32], Access::ReadOnly)?, false),
        };
        Ok(DeviceMesh { width: m.width, height: m.height, pos, levels, has_levels, count: m.count })
    }
}

/// The plan on the device.
pub enum DevicePlan {
    Copy,
    Warp(DeviceMesh),
}

impl DevicePlan {
    pub fn upload(s: &Session, p: &Plan) -> Result<DevicePlan, StageError> {
        Ok(match p {
            Plan::Copy { .. } => DevicePlan::Copy,
            Plan::Warp(m) => DevicePlan::Warp(DeviceMesh::upload(s, m)?),
        })
    }
}

fn run(s: &Session, name: &str, range: Range, args: &[Arg<'_>]) -> Result<(), StageError> {
    let mut kern = s.kernel(name)?;
    s.queue.run(&mut kern, range, args)?;
    Ok(())
}

fn side(n: usize) -> Result<i32, StageError> {
    i32::try_from(n).map_err(|_| StageError::Unsupported(format!("forward warp: a size of {n}")))
}

fn uint(n: usize) -> Result<Arg<'static>, StageError> {
    Ok(Arg::U32(u32::try_from(n).map_err(|_| StageError::Unsupported(format!("forward warp: a count of {n}")))?))
}

fn image_arg(img: &DeviceImage) -> Arg<'_> {
    match &img.samples {
        DeviceSamples::U8(b) => b.arg(),
        DeviceSamples::U16(b) => b.arg(),
        DeviceSamples::F32(b) => b.arg(),
    }
}

/// The pyramid levels 1.. of `src` in one buffer of the source's format, and the level table.
pub struct DevicePyramid {
    pub levels: DeviceSamples,
    pub table: Buffer<i32>,
    pub count: usize,
}

/// `Pyramid::build(src)` on the device: per level the float load, `cv_hpass` and `cv_vpass` with
/// the taps of `blur5` (with the integer store and reload between them and after them at 8 and
/// 16 bpc), then `fw_resample_*` into the level buffer.
pub fn pyramid_enqueue(s: &Session, src: &DeviceImage, count: usize) -> Result<DevicePyramid, StageError> {
    let (w, h) = (src.width, src.height);
    let dims = level_dims(w, h, count);
    let (lt, len) = level_table(&dims)?;
    let table = Buffer::from_slice(&s.context, &lt, Access::ReadOnly)?;
    let levels = match src.fmt() {
        Fmt::U8 => DeviceSamples::U8(Buffer::new(&s.context, len, Access::ReadWrite)?),
        Fmt::U16 => DeviceSamples::U16(Buffer::new(&s.context, len, Access::ReadWrite)?),
        Fmt::F32 => DeviceSamples::F32(Buffer::new(&s.context, len, Access::ReadWrite)?),
    };
    let pyr = DevicePyramid { levels, table, count };
    if count < 2 {
        return Ok(pyr);
    }
    let sfx = suffix(src.fmt());
    let n0 = w * h * CHANNELS;
    let taps = Taps::upload(s, &pyramid::BLUR5_TAPS)?;
    let f0 = Buffer::<f32>::new(&s.context, n0, Access::ReadWrite)?;
    let f1 = Buffer::<f32>::new(&s.context, n0, Access::ReadWrite)?;
    let tmp = match src.fmt() {
        Fmt::U8 => Some(DeviceSamples::U8(Buffer::new(&s.context, n0, Access::ReadWrite)?)),
        Fmt::U16 => Some(DeviceSamples::U16(Buffer::new(&s.context, n0, Access::ReadWrite)?)),
        Fmt::F32 => None,
    };
    let lev_arg = || match &pyr.levels {
        DeviceSamples::U8(b) => b.arg(),
        DeviceSamples::U16(b) => b.arg(),
        DeviceSamples::F32(b) => b.arg(),
    };
    for i in 1..count {
        let (sw, sh) = dims[i - 1];
        let (dw, dh) = dims[i];
        let ns = sw * sh * CHANNELS;
        let (from, from_off) = if i == 1 { (image_arg(src), 0usize) } else { (lev_arg(), lt[(i - 1) * 3] as usize) };
        run(s, &format!("fw_load_{sfx}"), Range::D1(ns), &[from, uint(from_off)?, f0.arg(), uint(ns)?])?;
        s.launch(Launch::HPass { src: &f0, dst: &f1, w: sw, h: sh, c: CHANNELS, taps: &taps, wrap: false })?;
        let blurred = match &tmp {
            Some(t) => {
                run(s, &format!("wb_round_{sfx}"), Range::D1(ns), &[f1.arg(), f0.arg(), uint(ns)?])?;
                s.launch(Launch::VPass { src: &f0, dst: &f1, w: sw, h: sh, c: CHANNELS, taps: &taps })?;
                let targ = match t {
                    DeviceSamples::U8(b) => b.arg(),
                    DeviceSamples::U16(b) => b.arg(),
                    DeviceSamples::F32(b) => b.arg(),
                };
                run(s, &format!("wb_from_float_{sfx}"), Range::D1(ns), &[f1.arg(), targ, uint(ns)?])?;
                targ
            }
            None => {
                s.launch(Launch::VPass { src: &f1, dst: &f0, w: sw, h: sh, c: CHANNELS, taps: &taps })?;
                f0.arg()
            }
        };
        let (sx, sy) = resample_scales(sw, sh, dw, dh);
        let args = [blurred, Arg::I32(side(sw)?), Arg::I32(side(sh)?), lev_arg(), uint(lt[i * 3] as usize)?, Arg::I32(side(dw)?), Arg::I32(side(dh)?), Arg::F32(sx), Arg::F32(sy)];
        run(s, &format!("fw_resample_{sfx}"), Range::D2(dw, dh), &args)?;
    }
    Ok(pyr)
}

fn check_shapes(src: &DeviceImage, out: &DeviceImage, cov: Option<&DeviceCoverage>, w: usize, h: usize) -> Result<(), StageError> {
    let ok = src.width == w
        && src.height == h
        && out.width == w
        && out.height == h
        && src.fmt() == out.fmt()
        && cov.is_none_or(|c| c.width == w && c.height == h);
    if ok {
        Ok(())
    } else {
        Err(StageError::Unsupported("forward warp: device images of different shapes".into()))
    }
}

/// Queues `forward_warp_with` of `src` into `out` (and the coverage into `cov` when given) for
/// the plan uploaded with [`DevicePlan::upload`]. See the module documentation for the two scalar
/// read backs and the refusals.
pub fn forward_warp_enqueue(s: &Session, plan: &DevicePlan, src: &DeviceImage, out: &DeviceImage, cov: Option<&DeviceCoverage>, flag: &Flag) -> Result<(), StageError> {
    let mesh = match plan {
        DevicePlan::Copy => {
            check_shapes(src, out, cov, src.width, src.height)?;
            match (&src.samples, &out.samples) {
                (DeviceSamples::U8(a), DeviceSamples::U8(b)) => s.queue.copy(a, b)?,
                (DeviceSamples::U16(a), DeviceSamples::U16(b)) => s.queue.copy(a, b)?,
                (DeviceSamples::F32(a), DeviceSamples::F32(b)) => s.queue.copy(a, b)?,
                _ => return Err(StageError::Unsupported("forward warp: formats differ".into())),
            }
            if let Some(c) = cov {
                s.queue.fill(&c.data, 0xffu8)?;
            }
            return Ok(());
        }
        DevicePlan::Warp(m) => m,
    };
    let (w, h) = (mesh.width, mesh.height);
    check_shapes(src, out, cov, w, h)?;
    let sfx = suffix(src.fmt());
    let pyr = pyramid_enqueue(s, src, mesh.count)?;
    let npx = w * h;
    let nc = npx.div_ceil(k::FW_CHUNK);
    let ns = npx.div_ceil(k::FW_SUPER);
    if ns > k::FW_MAX_SUPERS {
        return Err(StageError::Unsupported(format!("forward warp: {w}x{h} is above the device bound")));
    }
    let ctx = &s.context;
    let cnt = Buffer::<u32>::new(ctx, npx, Access::ReadWrite)?;
    let fill = Buffer::<u32>::new(ctx, npx, Access::ReadWrite)?;
    let off = Buffer::<u32>::new(ctx, npx, Access::ReadWrite)?;
    let csum = Buffer::<u32>::new(ctx, nc, Access::ReadWrite)?;
    let cpre = Buffer::<u32>::new(ctx, nc, Access::ReadWrite)?;
    let ssum = Buffer::<u32>::new(ctx, ns, Access::ReadWrite)?;
    let sbase = Buffer::<u32>::new(ctx, ns, Access::ReadWrite)?;
    let total = Buffer::<u32>::new(ctx, 2, Access::ReadWrite)?;
    let iflag = Buffer::<i32>::new(ctx, 1, Access::ReadWrite)?;
    s.queue.fill(&cnt, 0u32)?;
    s.queue.fill(&fill, 0u32)?;
    s.queue.fill(&total, 0u32)?;
    s.queue.fill(&iflag, 0i32)?;
    let (wi, hi) = (Arg::I32(side(w)?), Arg::I32(side(h)?));
    let has_d = Arg::I32(i32::from(mesh.has_levels));
    let maxrec = max_records(w, h);
    let bands = triangle_bands(w, h);
    for &(key0, n) in &bands {
        let args = [mesh.pos.arg(), mesh.levels.arg(), has_d, wi, hi, cnt.arg(), total.arg(), Arg::U32(maxrec), iflag.arg(), Arg::U32(key0), Arg::U32(n)];
        run(s, "fw_count", Range::D1(n as usize), &args)?;
    }
    run(s, "fw_scan_chunks", Range::D1(nc), &[cnt.arg(), off.arg(), csum.arg(), uint(npx)?, uint(nc)?])?;
    run(s, "fw_scan_supers", Range::D1(ns), &[csum.arg(), cpre.arg(), ssum.arg(), uint(nc)?, uint(ns)?])?;
    run(s, "fw_scan_top", Range::D1(1), &[ssum.arg(), sbase.arg(), total.arg(), uint(ns)?])?;
    let word = s.queue.read(&iflag)?[0];
    flag_result(word, "forward warp")?;
    let t = s.queue.read(&total)?;
    if t[0] > maxrec || t[0] != t[1] {
        return Err(StageError::Unsupported(format!("forward warp: {} hits, more than the device holds", t[0])));
    }
    let rec = Buffer::<u32>::new(ctx, (t[0] as usize).max(1) * 4, Access::ReadWrite)?;
    for &(key0, n) in &bands {
        let args = [
            mesh.pos.arg(),
            mesh.levels.arg(),
            has_d,
            wi,
            hi,
            cnt.arg(),
            fill.arg(),
            off.arg(),
            cpre.arg(),
            sbase.arg(),
            rec.arg(),
            Arg::U32(t[0]),
            flag.buf.arg(),
            Arg::U32(key0),
            Arg::U32(n),
        ];
        run(s, "fw_emit", Range::D1(n as usize), &args)?;
    }
    let lev = match &pyr.levels {
        DeviceSamples::U8(b) => b.arg(),
        DeviceSamples::U16(b) => b.arg(),
        DeviceSamples::F32(b) => b.arg(),
    };
    let (cov_arg, want) = match cov {
        Some(c) => (c.data.arg(), 1),
        None => (cnt.arg(), 0),
    };
    for (y0, rows) in row_bands(w, h) {
        let args = [
            image_arg(src),
            lev,
            pyr.table.arg(),
            Arg::I32(side(pyr.count)?),
            rec.arg(),
            cnt.arg(),
            off.arg(),
            cpre.arg(),
            sbase.arg(),
            image_arg(out),
            cov_arg,
            Arg::I32(want),
            flag.buf.arg(),
            wi,
            hi,
            Arg::I32(side(y0)?),
            Arg::I32(side(rows)?),
        ];
        run(s, &format!("fw_fold_{sfx}"), Range::D2(w, rows), &args)?;
    }
    Ok(())
}

/// `forward::forward_warp_with` on the device.
pub fn forward_warp_with_gpu(
    s: &Session,
    out: &mut Image,
    src: &Image,
    flow: &FlowField,
    amount: f32,
    opts: ForwardOpts,
    want_coverage: bool,
) -> Result<Option<Coverage>, StageError> {
    let p = plan(out, src, flow, amount, opts)?;
    let dplan = DevicePlan::upload(s, &p)?;
    let dsrc = DeviceImage::upload(s, src)?;
    let dout = DeviceImage::zeroed(s, src.width, src.height, src.fmt())?;
    let dcov = if want_coverage { Some(DeviceCoverage::filled(s, src.width, src.height, 0)?) } else { None };
    let flag = Flag::new(s)?;
    forward_warp_enqueue(s, &dplan, &dsrc, &dout, dcov.as_ref(), &flag)?;
    flag.check(s, "forward warp")?;
    let img = dout.download(s)?;
    if has_nan(&img) {
        return Err(StageError::NanOutput);
    }
    let cov = match dcov {
        Some(c) => Some(c.download(s)?),
        None => None,
    };
    *out = img;
    Ok(cov)
}

/// `forward::forward_warp` (warp code 3) on the device.
pub fn forward_warp_gpu(s: &Session, out: &mut Image, src: &Image, flow: &FlowField, amount: f32) -> Result<(), StageError> {
    forward_warp_with_gpu(s, out, src, flow, amount, ForwardOpts::default(), false).map(|_| ())
}

/// `forward::forward_warp_coverage` (warp code 2) on the device.
pub fn forward_warp_coverage_gpu(s: &Session, out: &mut Image, src: &Image, flow: &FlowField, amount: f32) -> Result<Coverage, StageError> {
    match forward_warp_with_gpu(s, out, src, flow, amount, ForwardOpts::default(), true)? {
        Some(c) => Ok(c),
        None => Err(StageError::Unsupported("forward warp: coverage was not produced".into())),
    }
}
