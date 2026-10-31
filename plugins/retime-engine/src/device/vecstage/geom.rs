//! Draw Geom and the tracking point splat on the device: `geom::draw_points` (kernels
//! `gm_draw_*`), the per level splat of `points::user_displacement` (`gm_splat`) and the start of
//! `points::second_tracker` (`gm_splat` at level 0 and `gm_seed`; its pull-push hole fill stays on
//! the CPU, another module's stage). The per point boxes of Draw Geom come from the CPU's own
//! `geom::span`, so the kernels see the CPU's integers.

use super::{check_pixels, count, download_floats, download_image, launch, samples_arg, upload_image, DeviceFlow, DevicePlane, MAX_POINTS};
use crate::device::cl::{Access, Arg, Buffer, Range};
use crate::device::image::{DeviceImage, StageError};
use crate::device::kernels::vecstage::DRAW_INTS;
use crate::device::selfcheck::Session;
use crate::flow::extras::{UserDisplacement, UserLevel};
use crate::flow::{FlowField, FlowGeometry};
use crate::holefill::{fill_flow_holes, HoleMaskRef};
use crate::image::Image;

/// The launch constants of `gm_draw_*` for `pts` on a `w x h` image: per point `px, py` and the
/// box, colour and flag (flag 0 for a point that is not finite or whose box misses the image).
pub fn draw_constants(pts: &[([f32; 2], [u8; 3])], r: f32, w: usize, h: usize) -> (Vec<f32>, Vec<i32>) {
    let mut p = Vec::with_capacity(pts.len() * 2);
    let mut ip = Vec::with_capacity(pts.len() * DRAW_INTS);
    for &(pt, col) in pts {
        p.extend_from_slice(&[pt[0], pt[1]]);
        let boxed = if pt[0].is_finite() && pt[1].is_finite() {
            match (crate::geom::span(pt[0], r, w), crate::geom::span(pt[1], r, h)) {
                (Some((x0, x1)), Some((y0, y1))) => Some([x0 as i32, x1 as i32, y0 as i32, y1 as i32]),
                _ => None,
            }
        } else {
            None
        };
        let (b, flag) = match boxed {
            Some(b) => (b, 1),
            None => ([0; 4], 0),
        };
        ip.extend_from_slice(&[b[0], b[1], b[2], b[3], col[0] as i32, col[1] as i32, col[2] as i32, flag]);
    }
    (p, ip)
}

/// Queues `gm_draw_*` on `img` in place; at most [`MAX_POINTS`] points. Records gap G3 when there
/// is a point.
pub fn draw_points_enqueue(session: &Session, img: &DeviceImage, pts: &[([f32; 2], [u8; 3])], ratio: f32) -> Result<(), StageError> {
    if pts.len() > MAX_POINTS {
        return Err(StageError::Unsupported(format!("draw geom: {} points (at most {MAX_POINTS} on the device)", pts.len())));
    }
    check_pixels(img.width, img.height)?;
    if pts.is_empty() {
        return Ok(());
    }
    fx_core::gaps::record(&crate::gaps::G3);
    let r = 1.5f32 * ratio;
    if !(r >= 0.0) || !r.is_finite() {
        return Ok(());
    }
    let r2 = r * r;
    let (p, ip) = draw_constants(pts, r, img.width, img.height);
    let pb = Buffer::from_slice(&session.context, &p, Access::ReadOnly)?;
    let ib = Buffer::from_slice(&session.context, &ip, Access::ReadOnly)?;
    let args = [samples_arg(&img.samples), pb.arg(), ib.arg(), Arg::U32(pts.len() as u32), Arg::F32(r2), Arg::U32(img.width as u32), Arg::U32(img.height as u32)];
    launch(session, &format!("gm_draw_{}", super::suffix(img.fmt())), Range::D2(img.width, img.height), &args)
}

/// `geom::draw_points` on the device. An image without pixels or no point changes nothing, as on
/// the CPU. On an error `img` is unchanged.
pub fn draw_points_gpu(session: &Session, img: &mut Image, pts: &[([f32; 2], [u8; 3])], ratio: f32) -> Result<(), StageError> {
    if pts.len() > MAX_POINTS {
        return Err(StageError::Unsupported(format!("draw geom: {} points (at most {MAX_POINTS} on the device)", pts.len())));
    }
    if pts.is_empty() {
        return Ok(());
    }
    if img.width == 0 || img.height == 0 {
        fx_core::gaps::record(&crate::gaps::G3);
        return Ok(());
    }
    let d = upload_image(session, img)?;
    draw_points_enqueue(session, &d, pts, ratio)?;
    *img = download_image(session, &d)?;
    Ok(())
}

/// The scalar arguments of `gm_splat` for one `wl x hl` level of an output of `out`:
/// `(kx, ky, W as f32, H as f32)` as `points.rs` computes them.
pub fn splat_constants(wl: usize, hl: usize, out: (usize, usize)) -> (f32, f32, f32, f32) {
    let kx = (wl - 1) as f32 / (out.0 - 1) as f32;
    let ky = (hl - 1) as f32 / (out.1 - 1) as f32;
    (kx, ky, out.0 as f32, out.1 as f32)
}

fn pair_values(pairs: &[([f32; 2], [f32; 2])]) -> Vec<f32> {
    pairs.iter().flat_map(|(f, t)| [f[0], f[1], t[0], t[1]]).collect()
}

/// Queues `gm_splat`: the level `d` (two channels) and `w` of `wl x hl` receive the splat of the
/// pairs whose values `pb` holds (`n` pairs, at most [`MAX_POINTS`]) for an output of `out`.
pub fn splat_enqueue(session: &Session, pb: &Buffer<f32>, n: usize, out: (usize, usize), d: &DeviceFlow, w: &DevicePlane) -> Result<(), StageError> {
    let (wl, hl) = (d.width, d.height);
    if n > MAX_POINTS || pb.len() < n * 4 || (w.width, w.height) != (wl, hl) {
        return Err(StageError::Unsupported(format!("point splat: {n} pairs (at most {MAX_POINTS}) or a weight of another size")));
    }
    check_pixels(wl, hl)?;
    if out.0 < 2 || out.1 < 2 {
        return Err(StageError::Unsupported("point splat: an output below 2x2 has no search".into()));
    }
    let (kx, ky, ow, oh) = splat_constants(wl, hl, out);
    let args = [pb.arg(), Arg::U32(n as u32), Arg::F32(kx), Arg::F32(ky), Arg::F32(ow), Arg::F32(oh), d.data.arg(), w.data.arg(), Arg::U32(wl as u32), Arg::U32(hl as u32)];
    launch(session, "gm_splat", Range::D2(wl, hl), &args)
}

/// The upload of the pair values (at least one value, so a buffer exists for no pair too).
fn pair_buffer(session: &Session, pairs: &[([f32; 2], [f32; 2])]) -> Result<Buffer<f32>, StageError> {
    let mut v = pair_values(pairs);
    if v.is_empty() {
        v.push(0.0);
    }
    Ok(Buffer::from_slice(&session.context, &v, Access::ReadOnly)?)
}

/// `points::user_displacement` on the device: one splat launch per level. An output below 2x2 or
/// a level without pixels gives the CPU's empty level without a launch.
pub fn user_displacement_gpu(session: &Session, pairs: &[([f32; 2], [f32; 2])], geo: &FlowGeometry, out: (usize, usize)) -> Result<UserDisplacement, StageError> {
    if pairs.len() > MAX_POINTS {
        return Err(StageError::Unsupported(format!("point splat: {} pairs (at most {MAX_POINTS})", pairs.len())));
    }
    let pb = pair_buffer(session, pairs)?;
    let mut levels = Vec::with_capacity(geo.sizes.len());
    for &(wl, hl) in &geo.sizes {
        if out.0 < 2 || out.1 < 2 || wl == 0 || hl == 0 {
            levels.push(UserLevel { d: vec![0.0; wl * hl * 2], w: vec![0.0; wl * hl] });
            continue;
        }
        let d = DeviceFlow::output(session, wl, hl)?;
        let w = DevicePlane::output(session, wl, hl)?;
        splat_enqueue(session, &pb, pairs.len(), out, &d, &w)?;
        levels.push(UserLevel { d: download_floats(session, &d.data)?, w: download_floats(session, &w.data)? });
    }
    Ok(UserDisplacement { levels })
}

/// Queues `gm_seed`: from the level 0 splat `d`, `w`, `flow` receives `d / w` where a point
/// landed (else zero) and `holes` 0 there (else 255).
pub fn second_tracker_seed_enqueue(session: &Session, d: &DeviceFlow, w: &DevicePlane, flow: &DeviceFlow, holes: &Buffer<u8>) -> Result<(), StageError> {
    let n = check_pixels(d.width, d.height)?;
    if (w.width, w.height) != (d.width, d.height) || (flow.width, flow.height) != (d.width, d.height) || holes.len() != n {
        return Err(StageError::Unsupported("second tracker: buffers of other sizes".into()));
    }
    launch(session, "gm_seed", Range::D1(n), &[d.data.arg(), w.data.arg(), flow.data.arg(), holes.arg(), count(n)])
}

/// `points::second_tracker` with its splat and `d / w` start on the device and the pull-push
/// fill (`holefill::fill_flow_holes`) and the `-1.0` scale on the CPU. A fill error is
/// `Unsupported` with the CPU's text.
pub fn second_tracker_gpu(session: &Session, pairs: &[([f32; 2], [f32; 2])], geo: &FlowGeometry, out: (usize, usize)) -> Result<FlowField, StageError> {
    if pairs.len() > MAX_POINTS {
        return Err(StageError::Unsupported(format!("second tracker: {} pairs (at most {MAX_POINTS})", pairs.len())));
    }
    let (w0, h0) = (geo.width, geo.height);
    check_pixels(w0, h0)?;
    if out.0 < 2 || out.1 < 2 {
        return Err(StageError::Unsupported("second tracker: an output below 2x2 has no search".into()));
    }
    fx_core::gaps::record(&crate::gaps::G5);
    let pb = pair_buffer(session, pairs)?;
    let d = DeviceFlow::output(session, w0, h0)?;
    let w = DevicePlane::output(session, w0, h0)?;
    splat_enqueue(session, &pb, pairs.len(), out, &d, &w)?;
    let seed = DeviceFlow::output(session, w0, h0)?;
    let holes = Buffer::<u8>::new(&session.context, w0 * h0, Access::ReadWrite)?;
    second_tracker_seed_enqueue(session, &d, &w, &seed, &holes)?;
    let mask = session.queue.read(&holes)?;
    if !mask.iter().any(|&m| m == 0) {
        return Ok(FlowField::new(w0, h0));
    }
    let mut flow = FlowField { width: w0, height: h0, data: download_floats(session, &seed.data)? };
    fill_flow_holes(&mut flow, HoleMaskRef::U8(&mask), false).map_err(|e| StageError::Unsupported(e.to_string()))?;
    flow.scale(-1.0f32);
    Ok(flow)
}
