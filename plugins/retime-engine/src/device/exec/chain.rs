//! The resident compose chain: `compose::compose_vectors` (the warps, the composition and the
//! blur compensation tail of one vector frame) as one queue sequence on the device. The two
//! frames and the flows are uploaded once, every intermediate image, coverage and hole mask stays
//! on the device, and only the finished frame is read back.
//!
//! The chain is built from the stage modules' `*_enqueue` functions in the order
//! `compose::compose_vectors` calls the CPU functions, with the CPU's scalars (`frac = 1.0f32 -
//! wA`, the branch tests, the warp options) computed by the same expressions on the host:
//!
//! | warping code | Nearest | Blend | Weighted Blend |
//! |---|---|---|---|
//! | 0 inverse | one warp into the output | two warps, `blend` | two warps, two `sharpen_step`, `blend` |
//! | 1 Smart Blend | warp with coverage, invert, hole fill | two warps with coverage, Smart Blend (binary alpha with "360 Video"), combine, invert, hole fill | `weighted_blend_smart` |
//! | 2 forward with fill | forward warp with coverage, invert, hole fill | two forward warps with coverage, then as code 1 | two forward warps with coverage, `weighted_tail` |
//! | 3 forward | one forward warp | two forward warps, `blend` | two forward warps, `weighted_tail` without coverage |
//!
//! then the blur compensation tail with the flow `F_AB` (Nearest: the single flow).
//!
//! What the chain leaves to the CPU (`StageError::Unsupported`, the executor then runs
//! `compose::compose_vectors` itself with the same inputs): the engine's wrap-around flag
//! `wrap_360` (the 360 variants of the warps, the fill and the blur, which also record the parity
//! gap note G6), every argument the CPU refuses (a missing flow, a bad code or mode, a frame or
//! flow below 2 x 2, the forward warp geometry it refuses) and every refusal of a stage (the
//! raised [`Flag`], the forward warp's device bounds). A finished float frame holding a NaN is
//! `StageError::NanOutput`. So the result is the CPU's bytes or a refusal.
//!
//! Precision contract (V): none of its own; each step is a stage the stage modules' tests prove bit
//! exact, in the CPU's order, and the run level cross check compares the first frame.

use crate::compose::{check_supported, nearest_uses_f_ab, Flows};
use crate::device::convstage::holefill::{fill_image_enqueue, DeviceFiller, MaskSource};
use crate::device::fwdstage::{self, DevicePlan};
use crate::device::image::{DeviceImage, StageError};
use crate::device::selfcheck::Session;
use crate::device::warpstage::blend::blend_enqueue;
use crate::device::warpstage::blurcomp::blur_compensation_wrap_enqueue;
use crate::device::warpstage::smartblend::{coverage_holes_enqueue, coverage_invert_enqueue, smart_blend_binary_enqueue, smart_blend_enqueue};
use crate::device::warpstage::warp::inv_warp_enqueue;
use crate::device::warpstage::{download_checked, DeviceCoverage, DeviceFlow, Flag};
use crate::device::wbstage::{sharpen_step_enqueue, weighted_blend_smart_enqueue, weighted_tail_enqueue};
use crate::flow::FlowField;
use crate::forward::ForwardOpts;
use crate::host::EngineSettings;
use crate::image::Image;
use crate::smartblend::SmartOpts;
use crate::timeline::Selection;
use crate::warp::WarpOpts;

/// The frames and flows of one chain on the device.
struct Inputs<'a> {
    a: (&'a Image, DeviceImage),
    b: (&'a Image, DeviceImage),
    f_ab: Option<(&'a FlowField, DeviceFlow)>,
    f_ba: Option<(&'a FlowField, DeviceFlow)>,
}

fn missing(which: &str) -> StageError {
    StageError::Unsupported(format!("compose chain: the flow {which} was not computed"))
}

impl Inputs<'_> {
    fn ab(&self) -> Result<&(&FlowField, DeviceFlow), StageError> {
        self.f_ab.as_ref().ok_or_else(|| missing("F_AB"))
    }
    fn ba(&self) -> Result<&(&FlowField, DeviceFlow), StageError> {
        self.f_ba.as_ref().ok_or_else(|| missing("F_BA"))
    }
}

fn zeroed_like(s: &Session, img: &Image) -> Result<DeviceImage, StageError> {
    DeviceImage::zeroed(s, img.width, img.height, img.fmt())
}

/// `holefill::fill_holes(out, holes, false)` on the device. Without a hole the fill is the
/// identity on the device too (every weight is 1.0 and every sample comes back as `v / 1.0`).
fn fill_enqueue(s: &Session, out: &DeviceImage, holes: &DeviceCoverage) -> Result<(), StageError> {
    if holes.width != out.width || holes.height != out.height {
        return Err(StageError::Unsupported("hole fill: a holes mask of another size".into()));
    }
    let filler = DeviceFiller::new(s, MaskSource::U8(&holes.data), out.width, out.height, false)?;
    fill_image_enqueue(s, &filler, out)
}

/// One forward warp (`forward::forward_warp_with` with the default options) of `src` into `out`,
/// the coverage into `cov` when given. The mesh is computed on the host by the CPU's functions.
fn forward(s: &Session, out: &DeviceImage, src: (&Image, &DeviceImage), flow: &FlowField, amount: f32, cov: Option<&DeviceCoverage>, flag: &Flag) -> Result<(), StageError> {
    let plan = fwdstage::plan(src.0, src.0, flow, amount, ForwardOpts::default())?;
    let dplan = DevicePlan::upload(s, &plan)?;
    fwdstage::forward_warp_enqueue(s, &dplan, src.1, out, cov, flag)
}

/// Smart Blend (the binary alpha variant when `opts.binary_alpha`) of `ta` and `tb` into `out`,
/// then the finish: `cov1 = 255 - max(cov1, cov2)` and the hole fill (`smartblend::blend_tail`
/// with coverage, the 360 flag off).
fn smart_tail(s: &Session, out: &DeviceImage, ta: &DeviceImage, tb: &DeviceImage, frac: f32, covs: (&DeviceCoverage, &DeviceCoverage), opts: SmartOpts) -> Result<(), StageError> {
    if opts.binary_alpha {
        smart_blend_binary_enqueue(s, out, ta, tb, frac)?;
    } else {
        smart_blend_enqueue(s, out, ta, tb, frac)?;
    }
    coverage_holes_enqueue(s, covs.0, covs.1)?;
    fill_enqueue(s, out, covs.0)
}

/// Queues `compose::compose_vectors` into `out`. See the module documentation.
fn enqueue(s: &Session, settings: &EngineSettings, sel: &Selection, d: &Inputs<'_>, out: &DeviceImage, flag: &Flag) -> Result<(), StageError> {
    let wa = sel.wa;
    let frac = 1.0f32 - wa;
    let (a, b) = (&d.a, &d.b);
    let code = settings.warp_mode;
    // the device fill kernels have the engine's tap order only (a set with another order runs the CPU composition)
    let smart_opts = SmartOpts { binary_alpha: settings.video_360, is360: settings.wrap_360, ..SmartOpts::default() };
    let filled = |img: &Image| DeviceCoverage::filled(s, img.width, img.height, 0xff);
    if code >= 2 {
        // compose_forward: the pushed frame uses its own flow
        let fill = code == 2;
        match sel.interp {
            0 => {
                let flow = if nearest_uses_f_ab(wa, code) { d.ab()? } else { d.ba()? };
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                let (src, amount) = if !(0.5f32 > wa) { (a, 1.0f32 - wa) } else { (b, wa) };
                if fill {
                    let cov = filled(src.0)?;
                    forward(s, out, (src.0, &src.1), flow.0, amount, Some(&cov), flag)?;
                    coverage_invert_enqueue(s, &cov)?;
                    fill_enqueue(s, out, &cov)
                } else {
                    forward(s, out, (src.0, &src.1), flow.0, amount, None, flag)
                }
            }
            1 | 2 => {
                let (f_ab, f_ba) = (d.ab()?, d.ba()?);
                let ta = zeroed_like(s, a.0)?;
                let tb = zeroed_like(s, b.0)?;
                let covs = if fill { Some((filled(a.0)?, filled(b.0)?)) } else { None };
                forward(s, &ta, (a.0, &a.1), f_ab.0, frac, covs.as_ref().map(|c| &c.0), flag)?;
                forward(s, &tb, (b.0, &b.1), f_ba.0, wa, covs.as_ref().map(|c| &c.1), flag)?;
                let pair = covs.as_ref().map(|(c1, c2)| (c1, c2));
                if sel.interp == 1 {
                    match pair {
                        Some(p) => smart_tail(s, out, &ta, &tb, frac, p, smart_opts),
                        None => blend_enqueue(s, out, &ta, &tb, frac),
                    }
                } else {
                    weighted_tail_enqueue(s, out, &ta, &a.1, &tb, &b.1, wa, pair, smart_opts, flag)
                }
            }
            other => Err(StageError::Unsupported(format!("compose chain: interpolation mode {other}"))),
        }
    } else if code == 1 {
        let wopts = WarpOpts { hold: false, is360: smart_opts.is360, mode: 1 };
        match sel.interp {
            0 => {
                let flow = if nearest_uses_f_ab(wa, code) { d.ab()? } else { d.ba()? };
                let cov = filled(a.0)?;
                // the A branch is taken unless 0.5 > wa, so a NaN weight takes A
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                if !(0.5f32 > wa) {
                    inv_warp_enqueue(s, out, &a.1, &flow.1, 1.0f32 - wa, wopts, Some(&cov), flag)?;
                } else {
                    inv_warp_enqueue(s, out, &b.1, &flow.1, wa, wopts, Some(&cov), flag)?;
                }
                coverage_invert_enqueue(s, &cov)?;
                fill_enqueue(s, out, &cov)
            }
            1 => {
                let (f_ab, f_ba) = (d.ab()?, d.ba()?);
                let ta = zeroed_like(s, a.0)?;
                let c1 = filled(a.0)?;
                inv_warp_enqueue(s, &ta, &a.1, &f_ba.1, frac, wopts, Some(&c1), flag)?;
                let tb = zeroed_like(s, b.0)?;
                let c2 = filled(b.0)?;
                inv_warp_enqueue(s, &tb, &b.1, &f_ab.1, wa, wopts, Some(&c2), flag)?;
                smart_tail(s, out, &ta, &tb, frac, (&c1, &c2), smart_opts)
            }
            2 => {
                let (f_ab, f_ba) = (d.ab()?, d.ba()?);
                weighted_blend_smart_enqueue(s, out, &a.1, &f_ba.1, &b.1, &f_ab.1, wa, smart_opts, flag)
            }
            other => Err(StageError::Unsupported(format!("compose chain: interpolation mode {other}"))),
        }
    } else {
        let opts = WarpOpts { is360: settings.wrap_360, ..WarpOpts::default() };
        match sel.interp {
            0 => {
                let flow = if nearest_uses_f_ab(wa, code) { d.ab()? } else { d.ba()? };
                if 0.5 <= wa {
                    inv_warp_enqueue(s, out, &a.1, &flow.1, frac, opts, None, flag)
                } else {
                    inv_warp_enqueue(s, out, &b.1, &flow.1, wa, opts, None, flag)
                }
            }
            1 | 2 => {
                let (f_ab, f_ba) = (d.ab()?, d.ba()?);
                let ta = zeroed_like(s, a.0)?;
                let tb = zeroed_like(s, b.0)?;
                inv_warp_enqueue(s, &ta, &a.1, &f_ba.1, frac, opts, None, flag)?;
                inv_warp_enqueue(s, &tb, &b.1, &f_ab.1, wa, opts, None, flag)?;
                if sel.interp == 2 {
                    sharpen_step_enqueue(s, &ta, &a.1, frac, flag)?;
                    sharpen_step_enqueue(s, &tb, &b.1, wa, flag)?;
                }
                blend_enqueue(s, out, &ta, &tb, frac)
            }
            other => Err(StageError::Unsupported(format!("compose chain: interpolation mode {other}"))),
        }
    }
}

/// `compose::compose_vectors(settings, sel, a, b, flows)` on the device: its result, or a
/// refusal after which the CPU composes (see the module documentation).
pub fn compose_vectors_gpu(s: &Session, settings: &EngineSettings, sel: &Selection, a: &Image, b: &Image, flows: &Flows) -> Result<Image, StageError> {
    check_supported(settings, sel).map_err(|e| StageError::Unsupported(e.to_string()))?;
    if settings.wrap_360 {
        return Err(StageError::Unsupported("compose chain: the wrap-around (360) variants run on the CPU".into()));
    }
    if !a.same_shape(b) {
        return Err(StageError::Unsupported("compose chain: frames of different shapes".into()));
    }
    let upload = |f: &Option<FlowField>| -> Result<Option<DeviceFlow>, StageError> {
        match f {
            Some(f) => DeviceFlow::upload(s, f).map(Some),
            None => Ok(None),
        }
    };
    let d = Inputs {
        a: (a, DeviceImage::upload(s, a)?),
        b: (b, DeviceImage::upload(s, b)?),
        f_ab: flows.f_ab.as_ref().zip(upload(&flows.f_ab)?),
        f_ba: flows.f_ba.as_ref().zip(upload(&flows.f_ba)?),
    };
    let out = zeroed_like(s, a)?;
    let flag = Flag::new(s)?;
    enqueue(s, settings, sel, &d, &out, &flag)?;
    // the tail: after the composition, with the flow F_AB (Nearest: the single flow)
    let tail = match (&d.f_ab, &d.f_ba) {
        (Some(f), _) | (None, Some(f)) => Some(&f.1),
        (None, None) => None,
    };
    if let Some(flow) = tail {
        blur_compensation_wrap_enqueue(s, &out, flow, sel.s0, sel.s1, settings.blur, settings.quality_high, false, &flag)?;
    }
    flag.check(s, "compose chain")?;
    download_checked(s, &out)
}
