//! Vector image coding on the device: `vectors::encode_vectors`, `encode_vectors_image`,
//! `encode_vectors_merged`, `encode_vectors_merged_image`, `encode_vectors_merged_blue`,
//! `merge_flows`, `decode_vectors`,
//! `vector_hole_mask`, `decode_vectors_with_holes`, `apply_vec_scale` and `zero_vectors_image`
//! (kernels `vc_*`). The per launch constants (`sx`, `sy`, `kx`, `ky` per layer) are computed on
//! the host by the CPU's own expressions ([`constants`]), so every kernel sees the CPU's values.

use super::{check_pixels, count, download_floats, download_image, image_output, launch, samples_arg, same_size, suffix, upload_image, DeviceFlow, DevicePlane, MAX_LAYERS};
use crate::device::cl::{Access, Arg, Buffer, Range};
use crate::device::image::{DeviceImage, DeviceSamples, StageError};
use crate::device::selfcheck::Session;
use crate::flow::FlowField;
use crate::image::{Fmt, Image};
use crate::matte::Matte;
use crate::vectors::HoleMask;

/// `(sx, sy, kx, ky)` of a `fw x fh` flow onto a `dw x dh` output, as `vectors.rs` computes them:
/// the sizes minus one as f32, `sx = fw1 / dw1`, `sy = fh1 / dh1`, `kx = (dw1 / fw1) / scale`,
/// `ky = (dh1 / fh1) / scale` (without a scale `kx = dw1 / fw1`, `ky = dh1 / fh1`).
pub fn constants(dw: usize, dh: usize, fw: usize, fh: usize, scale: Option<f32>) -> (f32, f32, f32, f32) {
    let (dw1, dh1) = ((dw as i32 - 1) as f32, (dh as i32 - 1) as f32);
    let (fw1, fh1) = ((fw as i32 - 1) as f32, (fh as i32 - 1) as f32);
    let (kx, ky) = match scale {
        Some(s) => ((dw1 / fw1) / s, (dh1 / fh1) / s),
        None => (dw1 / fw1, dh1 / fh1),
    };
    (fw1 / dw1, fh1 / dh1, kx, ky)
}

fn check_flow(w: usize, h: usize, len: usize, what: &str) -> Result<(), StageError> {
    if w < 2 || h < 2 || len != w * h * 2 {
        return Err(StageError::Unsupported(format!("{what}: a flow of {w}x{h} holding {len} values (at least 2x2 needed)")));
    }
    check_pixels(w, h)?;
    Ok(())
}

fn check_output(w: usize, h: usize, what: &str) -> Result<(), StageError> {
    if w < 2 || h < 2 {
        return Err(StageError::Unsupported(format!("{what} needs an output of at least 2x2 (output {w}x{h})")));
    }
    check_pixels(w, h)?;
    Ok(())
}

/// Queues `vc_encode_*`: `dst` (any format) receives the encoded image of `flow` at `scale`.
pub fn encode_vectors_enqueue(session: &Session, dst: &DeviceImage, flow: &DeviceFlow, scale: f32) -> Result<(), StageError> {
    check_output(dst.width, dst.height, "vector encode")?;
    check_flow(flow.width, flow.height, flow.data.len(), "vector encode")?;
    let (sx, sy, kx, ky) = constants(dst.width, dst.height, flow.width, flow.height, Some(scale));
    let args = [
        flow.data.arg(),
        Arg::U32(flow.width as u32),
        Arg::I32(flow.width as i32 - 1),
        Arg::I32(flow.height as i32 - 1),
        samples_arg(&dst.samples),
        Arg::U32(dst.width as u32),
        Arg::U32(dst.height as u32),
        Arg::F32(sx),
        Arg::F32(sy),
        Arg::F32(kx),
        Arg::F32(ky),
    ];
    launch(session, &format!("vc_encode_{}", suffix(dst.fmt())), Range::D2(dst.width, dst.height), &args)
}

/// `vectors::encode_vectors` on the device. On an error `dst` is unchanged.
pub fn encode_vectors_gpu(session: &Session, dst: &mut Image, flow: &FlowField, scale: f32) -> Result<(), StageError> {
    check_output(dst.width, dst.height, "vector encode")?;
    check_flow(flow.width, flow.height, flow.data.len(), "vector encode")?;
    let f = DeviceFlow::upload(session, flow)?;
    let out = image_output(session, dst.width, dst.height, dst.fmt())?;
    encode_vectors_enqueue(session, &out, &f, scale)?;
    *dst = download_image(session, &out)?;
    Ok(())
}

/// `vectors::encode_vectors_image` on the device.
pub fn encode_vectors_image_gpu(session: &Session, width: usize, height: usize, fmt: Fmt, flow: &FlowField, scale: f32) -> Result<Image, StageError> {
    let mut img = Image::new(width, height, fmt);
    encode_vectors_gpu(session, &mut img, flow, scale)?;
    Ok(img)
}

/// The layer constants of a merge: per layer `sx, sy, kx, ky` and `fw, fw1, fh1`.
fn merge_constants(dw: usize, dh: usize, layers: &[(&DeviceFlow, &DevicePlane)], scale: Option<f32>, what: &str) -> Result<(Vec<f32>, Vec<i32>), StageError> {
    if layers.is_empty() || layers.len() > MAX_LAYERS {
        return Err(StageError::Unsupported(format!("{what}: {} layers (1 to {MAX_LAYERS} run on the device)", layers.len())));
    }
    check_output(dw, dh, what)?;
    let mut p = Vec::with_capacity(4 * MAX_LAYERS);
    let mut ip = Vec::with_capacity(3 * MAX_LAYERS);
    for &(f, m) in layers {
        check_flow(f.width, f.height, f.data.len(), what)?;
        same_size((dw, dh), (m.width, m.height), what)?;
        if m.data.len() != dw * dh {
            return Err(StageError::Unsupported(format!("{what}: a weight buffer of {} values", m.data.len())));
        }
        let (sx, sy, kx, ky) = constants(dw, dh, f.width, f.height, scale);
        p.extend_from_slice(&[sx, sy, kx, ky]);
        ip.extend_from_slice(&[f.width as i32, f.width as i32 - 1, f.height as i32 - 1]);
    }
    Ok((p, ip))
}

/// Queues one of `vc_merge_*` / `vc_merge_flow` with `dst` the output samples.
fn merge_enqueue(session: &Session, name: &str, dw: usize, dh: usize, dst: Arg<'_>, layers: &[(&DeviceFlow, &DevicePlane)], scale: Option<f32>, what: &str) -> Result<(), StageError> {
    let (p, ip) = merge_constants(dw, dh, layers, scale, what)?;
    let pb = Buffer::from_slice(&session.context, &p, Access::ReadOnly)?;
    let ib = Buffer::from_slice(&session.context, &ip, Access::ReadOnly)?;
    // unused layer slots repeat the first layer; the loop never reads them
    let f = |k: usize| layers.get(k).unwrap_or(&layers[0]).0.data.arg();
    let w = |k: usize| layers.get(k).unwrap_or(&layers[0]).1.data.arg();
    let args = [f(0), f(1), f(2), f(3), w(0), w(1), w(2), w(3), pb.arg(), ib.arg(), Arg::U32(layers.len() as u32), dst, Arg::U32(dw as u32), Arg::U32(dh as u32)];
    launch(session, name, Range::D2(dw, dh), &args)
}

/// Queues `vc_merge_*`: `dst` receives the merged encoded image of `layers` (flow, weight) in
/// their order, at most [`MAX_LAYERS`].
pub fn encode_vectors_merged_enqueue(session: &Session, dst: &DeviceImage, layers: &[(&DeviceFlow, &DevicePlane)], scale: f32) -> Result<(), StageError> {
    let name = format!("vc_merge_{}", suffix(dst.fmt()));
    merge_enqueue(session, &name, dst.width, dst.height, samples_arg(&dst.samples), layers, Some(scale), "vector merge")?;
    Ok(())
}

fn upload_layers(session: &Session, layers: &[(&FlowField, &Matte)]) -> Result<Vec<(DeviceFlow, DevicePlane)>, StageError> {
    let mut out = Vec::with_capacity(layers.len());
    for &(f, m) in layers {
        check_flow(f.width, f.height, f.data.len(), "vector merge")?;
        out.push((DeviceFlow::upload(session, f)?, DevicePlane::upload(session, m.width, m.height, &m.data)?));
    }
    Ok(out)
}

fn check_layers(dw: usize, dh: usize, layers: &[(&FlowField, &Matte)], what: &str) -> Result<(), StageError> {
    if layers.is_empty() || layers.len() > MAX_LAYERS {
        return Err(StageError::Unsupported(format!("{what}: {} layers (1 to {MAX_LAYERS} run on the device)", layers.len())));
    }
    check_output(dw, dh, what)?;
    for &(f, m) in layers {
        check_flow(f.width, f.height, f.data.len(), what)?;
        if m.width != dw || m.height != dh || m.data.len() != dw * dh {
            return Err(StageError::Unsupported(format!("{what}: a weight of {}x{} for an output of {dw}x{dh}", m.width, m.height)));
        }
    }
    Ok(())
}

/// `vectors::encode_vectors_merged` on the device. On an error `dst` is unchanged.
pub fn encode_vectors_merged_gpu(session: &Session, dst: &mut Image, layers: &[(&FlowField, &Matte)], scale: f32) -> Result<(), StageError> {
    check_layers(dst.width, dst.height, layers, "vector merge")?;
    let up = upload_layers(session, layers)?;
    let refs: Vec<(&DeviceFlow, &DevicePlane)> = up.iter().map(|(f, m)| (f, m)).collect();
    let out = image_output(session, dst.width, dst.height, dst.fmt())?;
    encode_vectors_merged_enqueue(session, &out, &refs, scale)?;
    *dst = download_image(session, &out)?;
    Ok(())
}

/// `vectors::encode_vectors_merged_image` on the device.
pub fn encode_vectors_merged_image_gpu(session: &Session, width: usize, height: usize, fmt: Fmt, layers: &[(&FlowField, &Matte)], scale: f32) -> Result<Image, StageError> {
    let mut img = Image::new(width, height, fmt);
    encode_vectors_merged_gpu(session, &mut img, layers, scale)?;
    Ok(img)
}

/// Queues `vc_merge_*` and then `vc_blue_*`: `dst` receives the merged encoded image of `layers`
/// with the weight plane `blue` (the output size) in its blue channel.
pub fn encode_vectors_merged_blue_enqueue(session: &Session, dst: &DeviceImage, layers: &[(&DeviceFlow, &DevicePlane)], scale: f32, blue: &DevicePlane) -> Result<(), StageError> {
    let n = check_pixels(dst.width, dst.height)?;
    same_size((dst.width, dst.height), (blue.width, blue.height), "vector merge blue")?;
    encode_vectors_merged_enqueue(session, dst, layers, scale)?;
    let args = [samples_arg(&dst.samples), blue.data.arg(), count(n)];
    launch(session, &format!("vc_blue_{}", suffix(dst.fmt())), Range::D1(n), &args)
}

/// `vectors::encode_vectors_merged_blue` on the device (create-vectors fix 03: a single layer
/// output carries its weight in the blue channel). A `blue` of another size is `Unsupported`
/// (the CPU's `Mismatch`), checked before anything is uploaded, as are the refusals of
/// [`encode_vectors_merged_gpu`]. On an error `dst` is unchanged.
pub fn encode_vectors_merged_blue_gpu(session: &Session, dst: &mut Image, layers: &[(&FlowField, &Matte)], scale: f32, blue: &Matte) -> Result<(), StageError> {
    if blue.width != dst.width || blue.height != dst.height || blue.data.len() != dst.width * dst.height {
        return Err(StageError::Unsupported(format!("vector merge: a {}x{} blue weight for an output of {}x{}", blue.width, blue.height, dst.width, dst.height)));
    }
    check_layers(dst.width, dst.height, layers, "vector merge")?;
    let up = upload_layers(session, layers)?;
    let refs: Vec<(&DeviceFlow, &DevicePlane)> = up.iter().map(|(f, m)| (f, m)).collect();
    let db = DevicePlane::upload(session, blue.width, blue.height, &blue.data)?;
    let out = image_output(session, dst.width, dst.height, dst.fmt())?;
    encode_vectors_merged_blue_enqueue(session, &out, &refs, scale, &db)?;
    *dst = download_image(session, &out)?;
    Ok(())
}

/// Queues `vc_merge_flow`: `out` (the output size) receives the merged flow of `layers`.
pub fn merge_flows_enqueue(session: &Session, out: &DeviceFlow, layers: &[(&DeviceFlow, &DevicePlane)]) -> Result<(), StageError> {
    merge_enqueue(session, "vc_merge_flow", out.width, out.height, out.data.arg(), layers, None, "flow merge")
}

/// `vectors::merge_flows` on the device.
pub fn merge_flows_gpu(session: &Session, width: usize, height: usize, layers: &[(&FlowField, &Matte)]) -> Result<FlowField, StageError> {
    check_layers(width, height, layers, "flow merge")?;
    let up = upload_layers(session, layers)?;
    let refs: Vec<(&DeviceFlow, &DevicePlane)> = up.iter().map(|(f, m)| (f, m)).collect();
    let out = DeviceFlow::output(session, width, height)?;
    merge_flows_enqueue(session, &out, &refs)?;
    Ok(FlowField { width, height, data: download_floats(session, &out.data)? })
}

/// Queues `vc_decode_*`: `out` (the size of `vec`) receives the decoded flow.
pub fn decode_vectors_enqueue(session: &Session, vec: &DeviceImage, max_displace: f32, out: &DeviceFlow) -> Result<(), StageError> {
    let n = check_pixels(vec.width, vec.height)?;
    same_size((vec.width, vec.height), (out.width, out.height), "vector decode")?;
    let md = match vec.fmt() {
        Fmt::F32 => Arg::F32(max_displace),
        _ => Arg::F64(max_displace as f64),
    };
    let args = [samples_arg(&vec.samples), out.data.arg(), md, count(n)];
    launch(session, &format!("vc_decode_{}", suffix(vec.fmt())), Range::D1(n), &args)
}

/// `vectors::decode_vectors` on the device (an image without pixels gives the empty flow, as on
/// the CPU, without a launch).
pub fn decode_vectors_gpu(session: &Session, vec: &Image, max_displace: f32) -> Result<FlowField, StageError> {
    if vec.width == 0 || vec.height == 0 {
        return Ok(FlowField::new(vec.width, vec.height));
    }
    let v = upload_image(session, vec)?;
    let out = DeviceFlow::output(session, vec.width, vec.height)?;
    decode_vectors_enqueue(session, &v, max_displace, &out)?;
    Ok(FlowField { width: vec.width, height: vec.height, data: download_floats(session, &out.data)? })
}

/// A one channel buffer of `n` samples in the sample type of `fmt`.
pub fn mask_output(session: &Session, n: usize, fmt: Fmt) -> Result<DeviceSamples, StageError> {
    let c = &session.context;
    Ok(match fmt {
        Fmt::U8 => DeviceSamples::U8(Buffer::new(c, n, Access::ReadWrite)?),
        Fmt::U16 => DeviceSamples::U16(Buffer::new(c, n, Access::ReadWrite)?),
        Fmt::F32 => DeviceSamples::F32(Buffer::new(c, n, Access::ReadWrite)?),
    })
}

/// Queues `vc_holes_*`: `out` (one sample per pixel of `vec`, its sample type) receives
/// `white - A`.
pub fn vector_hole_mask_enqueue(session: &Session, vec: &DeviceImage, out: &DeviceSamples) -> Result<(), StageError> {
    let n = check_pixels(vec.width, vec.height)?;
    let len = match out {
        DeviceSamples::U8(b) => (b.len(), Fmt::U8),
        DeviceSamples::U16(b) => (b.len(), Fmt::U16),
        DeviceSamples::F32(b) => (b.len(), Fmt::F32),
    };
    if len != (n, vec.fmt()) {
        return Err(StageError::Unsupported("hole mask: the output does not match the image".into()));
    }
    let args = [samples_arg(&vec.samples), samples_arg(out), count(n)];
    launch(session, &format!("vc_holes_{}", suffix(vec.fmt())), Range::D1(n), &args)
}

/// Downloads a hole mask; a float mask holding a NaN is `NanOutput`.
pub fn download_mask(session: &Session, m: &DeviceSamples) -> Result<HoleMask, StageError> {
    Ok(match m {
        DeviceSamples::U8(b) => HoleMask::U8(session.queue.read(b)?),
        DeviceSamples::U16(b) => HoleMask::U16(session.queue.read(b)?),
        DeviceSamples::F32(b) => HoleMask::F32(download_floats(session, b)?),
    })
}

/// `vectors::vector_hole_mask` (and `matte::alpha_holes`) on the device.
pub fn vector_hole_mask_gpu(session: &Session, vec: &Image) -> Result<HoleMask, StageError> {
    let v = upload_image(session, vec)?;
    let out = mask_output(session, vec.width * vec.height, vec.fmt())?;
    vector_hole_mask_enqueue(session, &v, &out)?;
    download_mask(session, &out)
}

/// `vectors::decode_vectors_with_holes` on the device (one upload for both kernels).
pub fn decode_vectors_with_holes_gpu(session: &Session, vec: &Image, max_displace: f32) -> Result<(FlowField, HoleMask), StageError> {
    let v = upload_image(session, vec)?;
    let flow = DeviceFlow::output(session, vec.width, vec.height)?;
    let holes = mask_output(session, vec.width * vec.height, vec.fmt())?;
    decode_vectors_enqueue(session, &v, max_displace, &flow)?;
    vector_hole_mask_enqueue(session, &v, &holes)?;
    let f = FlowField { width: vec.width, height: vec.height, data: download_floats(session, &flow.data)? };
    Ok((f, download_mask(session, &holes)?))
}

/// Queues `vc_vec_scale` on `flow` in place.
pub fn apply_vec_scale_enqueue(session: &Session, flow: &DeviceFlow, sx: f32, sy: f32) -> Result<(), StageError> {
    let n = check_pixels(flow.width, flow.height)?;
    if flow.data.len() != n * 2 {
        return Err(StageError::Unsupported("vec scale: the flow buffer does not match its size".into()));
    }
    launch(session, "vc_vec_scale", Range::D1(n), &[flow.data.arg(), Arg::F32(sx), Arg::F32(sy), count(n)])
}

/// `vectors::apply_vec_scale` on the device. On an error `flow` is unchanged.
pub fn apply_vec_scale_gpu(session: &Session, flow: &mut FlowField, sx: f32, sy: f32) -> Result<(), StageError> {
    let f = DeviceFlow::upload(session, flow)?;
    apply_vec_scale_enqueue(session, &f, sx, sy)?;
    flow.data = download_floats(session, &f.data)?;
    Ok(())
}

/// Queues `vc_zero_*`: `out` becomes the zero vector image.
pub fn zero_vectors_image_enqueue(session: &Session, out: &DeviceImage) -> Result<(), StageError> {
    let n = check_pixels(out.width, out.height)?;
    launch(session, &format!("vc_zero_{}", suffix(out.fmt())), Range::D1(n), &[samples_arg(&out.samples), count(n)])
}

/// `vectors::zero_vectors_image` on the device.
pub fn zero_vectors_image_gpu(session: &Session, width: usize, height: usize, fmt: Fmt) -> Result<Image, StageError> {
    let out = image_output(session, width, height, fmt)?;
    zero_vectors_image_enqueue(session, &out)?;
    download_image(session, &out)
}
