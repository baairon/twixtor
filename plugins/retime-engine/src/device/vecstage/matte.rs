//! The pointwise matte operations on the device: `matte::extract`, `premultiply`, `over`,
//! `effective` (shrinks of zero only: the choke needs the Gaussian blur and stays on the CPU),
//! `over_weights` and `alpha_holes` (kernels `mt_*`, and `vc_holes_*` for the holes). The
//! kernels read every matte value through `mt_clamp_matte` where the CPU reads it through
//! `matte::clamp_matte` (premultiply, effective, over_weights). The wrappers check sizes first
//! and answer `Unsupported` where the CPU function returns `FxError::Mismatch`, before anything
//! is uploaded or queued.

use super::vectors::{download_mask, mask_output, vector_hole_mask_enqueue};
use super::{check_pixels, count, download_floats, download_image, image_output, launch, samples_arg, same_size, suffix, upload_image, DevicePlane, MAX_FG};
use crate::device::cl::{Arg, Range};
use crate::device::image::{DeviceImage, StageError};
use crate::device::selfcheck::Session;
use crate::image::Image;
use crate::matte::{matte_channel, Matte};
use crate::vectors::HoleMask;

/// Queues `mt_extract_*`: `out` (the size of `img`) receives the matte of channel popup `chan`.
pub fn extract_enqueue(session: &Session, img: &DeviceImage, chan: i32, out: &DevicePlane) -> Result<(), StageError> {
    let n = check_pixels(img.width, img.height)?;
    same_size((img.width, img.height), (out.width, out.height), "matte extract")?;
    let c = matte_channel(chan) as u32;
    let args = [samples_arg(&img.samples), out.data.arg(), Arg::U32(c), count(n)];
    launch(session, &format!("mt_extract_{}", suffix(img.fmt())), Range::D1(n), &args)
}

/// `matte::extract` on the device (an image without pixels gives the empty matte, as on the CPU).
pub fn extract_gpu(session: &Session, img: &Image, chan: i32) -> Result<Matte, StageError> {
    if img.width == 0 || img.height == 0 {
        return Ok(Matte::filled(img.width, img.height, 0.0));
    }
    let d = upload_image(session, img)?;
    let out = DevicePlane::output(session, img.width, img.height)?;
    extract_enqueue(session, &d, chan, &out)?;
    Ok(Matte { width: img.width, height: img.height, data: download_floats(session, &out.data)? })
}

/// Queues `mt_premultiply_*`: `out` (shape of `src`) receives `src` times the matte `m`.
pub fn premultiply_enqueue(session: &Session, src: &DeviceImage, m: &DevicePlane, out: &DeviceImage) -> Result<(), StageError> {
    let n = check_pixels(src.width, src.height)? * crate::image::CHANNELS;
    same_size((src.width, src.height), (m.width, m.height), "premultiply")?;
    same_size((src.width, src.height), (out.width, out.height), "premultiply")?;
    if src.fmt() != out.fmt() {
        return Err(StageError::Unsupported("premultiply: the output has another format".into()));
    }
    let args = [samples_arg(&src.samples), m.data.arg(), samples_arg(&out.samples), count(n)];
    launch(session, &format!("mt_premultiply_{}", suffix(src.fmt())), Range::D1(n), &args)
}

/// `matte::premultiply` on the device; a matte of another size (or holding another number of
/// values) is `Unsupported` (the CPU's `Mismatch`).
pub fn premultiply_gpu(session: &Session, src: &Image, m: &Matte) -> Result<Image, StageError> {
    if !matte_is(m, src.width, src.height) {
        return Err(StageError::Unsupported(format!("premultiply: matte {}x{} for an image of {}x{}", m.width, m.height, src.width, src.height)));
    }
    let s = upload_image(session, src)?;
    let dm = DevicePlane::upload(session, m.width, m.height, &m.data)?;
    let out = image_output(session, src.width, src.height, src.fmt())?;
    premultiply_enqueue(session, &s, &dm, &out)?;
    download_image(session, &out)
}

/// Queues `mt_over_*`: `src` OVER `dest`, in place on `dest`.
pub fn over_enqueue(session: &Session, dest: &DeviceImage, src: &DeviceImage) -> Result<(), StageError> {
    let n = check_pixels(dest.width, dest.height)?;
    same_size((dest.width, dest.height), (src.width, src.height), "layer OVER")?;
    if dest.fmt() != src.fmt() {
        return Err(StageError::Unsupported("layer OVER: images of two formats".into()));
    }
    let args = [samples_arg(&dest.samples), samples_arg(&src.samples), count(n)];
    launch(session, &format!("mt_over_{}", suffix(dest.fmt())), Range::D1(n), &args)
}

/// `matte::over` on the device; images of another shape are `Unsupported` (the CPU's
/// `Mismatch`). On an error `dest` is unchanged.
pub fn over_gpu(session: &Session, dest: &mut Image, src: &Image) -> Result<(), StageError> {
    if !dest.same_shape(src) {
        return Err(StageError::Unsupported(format!("layer OVER: {}x{} {:?} onto {}x{} {:?}", src.width, src.height, src.fmt(), dest.width, dest.height, dest.fmt())));
    }
    let d = upload_image(session, dest)?;
    let s = upload_image(session, src)?;
    over_enqueue(session, &d, &s)?;
    *dest = download_image(session, &d)?;
    Ok(())
}

/// Whether `m` is `w x h` and holds `w * h` values (the CPU's size check).
fn matte_is(m: &Matte, w: usize, h: usize) -> bool {
    m.width == w && m.height == h && Some(m.data.len()) == w.checked_mul(h)
}

fn check_mattes(mattes: &[&DevicePlane], w: usize, h: usize, what: &str) -> Result<(), StageError> {
    for m in mattes {
        same_size((w, h), (m.width, m.height), what)?;
    }
    Ok(())
}

/// Queues `mt_effective`: `out` (`mattes.len() + 1` planes of `w x h`) receives the tracking
/// mattes of `matte::effective` with every shrink zero, for 0 to [`MAX_FG`] FG mattes of
/// `w x h` (none gives Main_BG's constant 1.0).
pub fn effective_enqueue(session: &Session, mattes: &[&DevicePlane], w: usize, h: usize, out: &[&DevicePlane]) -> Result<(), StageError> {
    let nfg = mattes.len();
    if nfg > MAX_FG || out.len() != nfg + 1 {
        return Err(StageError::Unsupported(format!("effective: {nfg} FG mattes and {} outputs (0 to {MAX_FG} FG run on the device)", out.len())));
    }
    let n = check_pixels(w, h)?;
    check_mattes(mattes, w, h, "effective")?;
    check_mattes(out, w, h, "effective")?;
    // with no FG matte every matte slot repeats the first output (the loop never reads a matte)
    let m = |k: usize| mattes.get(k).or(mattes.first()).map(|p| p.data.arg()).unwrap_or_else(|| out[0].data.arg());
    let o = |k: usize| out.get(k).unwrap_or(&out[0]).data.arg();
    let args = [m(0), m(1), m(2), o(0), o(1), o(2), o(3), Arg::U32(nfg as u32), count(n)];
    launch(session, "mt_effective", Range::D1(n), &args)?;
    if nfg >= 2 {
        fx_core::gaps::record(&crate::gaps::G1C);
    }
    Ok(())
}

fn upload_mattes(session: &Session, mattes: &[Matte]) -> Result<Vec<DevicePlane>, StageError> {
    mattes.iter().map(|m| DevicePlane::upload(session, m.width, m.height, &m.data)).collect()
}

fn download_all(session: &Session, planes: &[DevicePlane]) -> Result<Vec<Matte>, StageError> {
    planes.iter().map(|p| Ok(Matte { width: p.width, height: p.height, data: download_floats(session, &p.data)? })).collect()
}

/// `matte::effective` on the device for shrinks that are all zero (or not finite, which the
/// choke also treats as zero); any other shrink is `Unsupported` (the choke runs a blur). The
/// CPU's `Mismatch` cases (a `shrinks` of another length, a matte that is not
/// `width x height` or holds another number of values, a size whose sample count overflows) are
/// `Unsupported`, as is an empty size; all of them are checked before anything is uploaded. No
/// FG matte gives Main_BG's constant 1.0 at `width x height`, computed on the device.
pub fn effective_gpu(session: &Session, mattes: &[Matte], shrinks: &[f32], width: usize, height: usize) -> Result<Vec<Matte>, StageError> {
    if shrinks.len() != mattes.len() {
        return Err(StageError::Unsupported(format!("effective: {} shrinks for {} FG mattes", shrinks.len(), mattes.len())));
    }
    if let Some(k) = mattes.iter().position(|m| !matte_is(m, width, height)) {
        let m = &mattes[k];
        return Err(StageError::Unsupported(format!("effective: matte {k} is {}x{} ({} samples), the layers are {width}x{height}", m.width, m.height, m.data.len())));
    }
    check_pixels(width, height)?;
    if shrinks.iter().any(|s| *s != 0.0 && s.is_finite()) {
        return Err(StageError::Unsupported("effective: a matte shrink needs the blur of the choke".into()));
    }
    if mattes.len() > MAX_FG {
        return Err(StageError::Unsupported(format!("effective: {} FG mattes (at most {MAX_FG} run on the device)", mattes.len())));
    }
    let up = upload_mattes(session, mattes)?;
    let out = (0..=mattes.len()).map(|_| DevicePlane::output(session, width, height)).collect::<Result<Vec<_>, _>>()?;
    effective_enqueue(session, &up.iter().collect::<Vec<_>>(), width, height, &out.iter().collect::<Vec<_>>())?;
    download_all(session, &out)
}

/// Queues `mt_over_weights`: `out` (`last - first + 1` planes) receives the OVER weights of the
/// layers `first..=last` of `matte::over_weights` over the `mattes.len()` FG mattes.
pub fn over_weights_enqueue(session: &Session, mattes: &[&DevicePlane], first: usize, last: usize, w: usize, h: usize, out: &[&DevicePlane]) -> Result<(), StageError> {
    let nfg = mattes.len();
    if nfg > MAX_FG || first > last || last > nfg || out.len() != last - first + 1 {
        return Err(StageError::Unsupported(format!("over_weights: group {first}..={last} of {nfg} FG layers, {} outputs", out.len())));
    }
    let n = check_pixels(w, h)?;
    check_mattes(mattes, w, h, "over_weights")?;
    check_mattes(out, w, h, "over_weights")?;
    // with no FG matte every slot repeats the first output (the loop never reads a matte)
    let m = |k: usize| mattes.get(k).or(mattes.first()).map(|p| p.data.arg()).unwrap_or_else(|| out[0].data.arg());
    let o = |k: usize| out.get(k).unwrap_or(&out[0]).data.arg();
    let args = [m(0), m(1), m(2), o(0), o(1), o(2), o(3), Arg::U32(first as u32), Arg::U32(last as u32), Arg::U32(nfg as u32), count(n)];
    launch(session, "mt_over_weights", Range::D1(n), &args)
}

/// `matte::over_weights` on the device; the CPU's `Mismatch` cases (a group outside the layers,
/// a matte that is not `w x h` or holds another number of values, an overflowing size) are
/// `Unsupported`, as is an empty size, all checked before anything is uploaded.
pub fn over_weights_gpu(session: &Session, mattes: &[Matte], first: usize, last: usize, w: usize, h: usize) -> Result<Vec<Matte>, StageError> {
    let nfg = mattes.len();
    if first > last || last > nfg {
        return Err(StageError::Unsupported(format!("over_weights: group {first}..={last} of {nfg} FG layers")));
    }
    if mattes.iter().any(|m| !matte_is(m, w, h)) {
        return Err(StageError::Unsupported("over_weights: a matte of another size".into()));
    }
    check_pixels(w, h)?;
    if nfg > MAX_FG {
        return Err(StageError::Unsupported(format!("over_weights: {nfg} FG mattes (at most {MAX_FG} run on the device)")));
    }
    let up = upload_mattes(session, mattes)?;
    let out = (first..=last).map(|_| DevicePlane::output(session, w, h)).collect::<Result<Vec<_>, _>>()?;
    over_weights_enqueue(session, &up.iter().collect::<Vec<_>>(), first, last, w, h, &out.iter().collect::<Vec<_>>())?;
    download_all(session, &out)
}

/// `matte::alpha_holes` on the device (the hole mask of `vectors::vector_hole_mask`).
pub fn alpha_holes_gpu(session: &Session, img: &Image) -> Result<HoleMask, StageError> {
    let d = upload_image(session, img)?;
    let out = mask_output(session, img.width * img.height, img.fmt())?;
    vector_hole_mask_enqueue(session, &d, &out)?;
    download_mask(session, &out)
}
