//! The stage-set seam: the few engine stages a render device may take over, behind one trait.
//!
//! [`Stages`] lists them with the CPU functions' arguments and results. [`Cpu`] is the reference:
//! every method is the CPU function itself, called as the engine called it before the seam
//! existed, so a render through [`Cpu`] is today's render, bit for bit. A device implementation
//! (`device::exec::Exec`) runs a stage on the device and gives the CPU function's result, or runs
//! the CPU function itself when the device cannot.
//!
//! [`StageSet`] is the handle the engine carries (`host::EngineSettings::stages`, the create
//! vectors `_on` functions): the CPU unless the host passed a device executor in
//! `TemporalContext::executor` ([`StageSet::from_context`]). Nothing here is global; the handle
//! travels with the settings of each render.
//!
//! Precision contract: none of its own. Every method's result is the result of the CPU function
//! named in its documentation (V for [`Cpu`]; for a device implementation by the stage modules'
//! construction, the self-check at device open and the run level cross check).

use crate::compose::{self, Flows};
use crate::flow::extras::{solve_pyramids_with, SolveExtras};
use crate::flow::{self, FlowField, FlowPair, FlowSettings, Pyramid};
use crate::forward::{self, ForwardOpts};
use crate::holefill::VerticalTaps;
use crate::host::EngineSettings;
use crate::image::Image;
use crate::timeline::Selection;
use crate::warp::Coverage;
use fx_core::{FxError, Result, TemporalContext};
use std::fmt;
use std::sync::Arc;

/// The engine stages a device may run. Each method returns what the CPU function of the same name
/// returns for the same arguments.
pub trait Stages: Send + Sync {
    /// `flow::solve`.
    fn solve(&self, x: &Image, y: &Image, s: &FlowSettings) -> Result<FlowField>;
    /// `flow::solve_pair`.
    fn solve_pair(&self, a: &Image, b: &Image, s: &FlowSettings) -> Result<FlowPair>;
    /// `flow::extras::solve_pyramids_with`.
    fn solve_pyramids_with(&self, x: &Pyramid, y: &Pyramid, s: &FlowSettings, e: SolveExtras<'_>) -> Result<FlowField>;
    /// `forward::forward_warp_with`, the forward warp of codes 2 and 3, the flow inversion and the
    /// Warp debug view.
    fn forward_warp_with(&self, out: &mut Image, src: &Image, flow: &FlowField, amount: f32, opts: ForwardOpts, want_coverage: bool) -> Result<Option<Coverage>>;

    /// `forward::forward_warp` on [`Stages::forward_warp_with`].
    fn forward_warp(&self, out: &mut Image, src: &Image, flow: &FlowField, amount: f32) -> Result<()> {
        self.forward_warp_with(out, src, flow, amount, ForwardOpts::default(), false).map(|_| ())
    }

    /// `forward::forward_warp_coverage` on [`Stages::forward_warp_with`].
    fn forward_warp_coverage(&self, out: &mut Image, src: &Image, flow: &FlowField, amount: f32) -> Result<Coverage> {
        match self.forward_warp_with(out, src, flow, amount, ForwardOpts::default(), true)? {
            Some(c) => Ok(c),
            None => Err(FxError::Unsupported("forward warp: coverage was not produced".into())),
        }
    }

    /// `compose::compose_vectors`: the warps, the composition and the blur compensation tail of
    /// one vector frame. The default is the CPU function itself (its forward warps go through
    /// `settings.stages`, as they always did).
    fn compose_vectors(&self, settings: &EngineSettings, sel: &Selection, a: &Image, b: &Image, flows: &Flows) -> Result<Image> {
        compose::compose_vectors(settings, sel, a, b, flows)
    }

    /// The vertical tap order (`holefill::VerticalTaps`) of the separable filters of the render:
    /// every pull-push hole fill (the image fills of the forward warps and of the Smart Blend
    /// compositions, the flow fills of the vector layers and of the flow inversion) and the blur of
    /// the Weighted Blend sharpening (`filters::sharpen_step_with`). The default is the engine's
    /// order.
    fn vertical_taps(&self) -> VerticalTaps {
        VerticalTaps::BelowFirst
    }
}

/// The CPU: every stage is the CPU function, unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cpu;

impl Stages for Cpu {
    fn solve(&self, x: &Image, y: &Image, s: &FlowSettings) -> Result<FlowField> {
        flow::solve(x, y, s)
    }

    fn solve_pair(&self, a: &Image, b: &Image, s: &FlowSettings) -> Result<FlowPair> {
        flow::solve_pair(a, b, s)
    }

    fn solve_pyramids_with(&self, x: &Pyramid, y: &Pyramid, s: &FlowSettings, e: SolveExtras<'_>) -> Result<FlowField> {
        solve_pyramids_with(x, y, s, e)
    }

    fn forward_warp_with(&self, out: &mut Image, src: &Image, flow: &FlowField, amount: f32, opts: ForwardOpts, want_coverage: bool) -> Result<Option<Coverage>> {
        forward::forward_warp_with(out, src, flow, amount, opts, want_coverage)
    }
}

/// The stages of another set with every forward warp taken as the Gouraud variant
/// (`ForwardOpts { pyramid: false }`, [`forward::forward_warp_with`]); the solves are the inner
/// set's, and `compose_vectors` is the CPU composition, whose forward warps come back here through
/// `settings.stages`. A plugin whose forward warps interpolate the vertex colours puts it in its
/// settings ([`GouraudForward::wrap`]).
pub struct GouraudForward(pub StageSet);

impl GouraudForward {
    /// `inner` with its forward warps taken as the Gouraud variant.
    pub fn wrap(inner: StageSet) -> StageSet {
        StageSet::new(Arc::new(GouraudForward(inner)))
    }
}

impl Stages for GouraudForward {
    fn solve(&self, x: &Image, y: &Image, s: &FlowSettings) -> Result<FlowField> {
        self.0.get().solve(x, y, s)
    }

    fn solve_pair(&self, a: &Image, b: &Image, s: &FlowSettings) -> Result<FlowPair> {
        self.0.get().solve_pair(a, b, s)
    }

    fn solve_pyramids_with(&self, x: &Pyramid, y: &Pyramid, s: &FlowSettings, e: SolveExtras<'_>) -> Result<FlowField> {
        self.0.get().solve_pyramids_with(x, y, s, e)
    }

    fn forward_warp_with(&self, out: &mut Image, src: &Image, flow: &FlowField, amount: f32, _opts: ForwardOpts, want_coverage: bool) -> Result<Option<Coverage>> {
        self.0.get().forward_warp_with(out, src, flow, amount, ForwardOpts { pyramid: false }, want_coverage)
    }

    fn vertical_taps(&self) -> VerticalTaps {
        self.0.get().vertical_taps()
    }
}

/// The stages of another set whose separable filters ([`Stages::vertical_taps`]: the pull-push
/// hole fills and the Weighted Blend blur) take the tap above first in the border rows of their
/// vertical passes (`holefill::VerticalTaps::AboveFirst`); everything else is the inner set's,
/// and `compose_vectors` is the CPU composition, whose filters read the order back through
/// `settings.stages`. A plugin whose filters add their taps that way puts it in its settings
/// ([`AboveFirstTaps::wrap`]). The device fill and blur kernels follow the engine's order only,
/// so the device composition is not used under this set (the CPU composition runs).
pub struct AboveFirstTaps(pub StageSet);

impl AboveFirstTaps {
    /// `inner` with its hole fills taking the tap above first.
    pub fn wrap(inner: StageSet) -> StageSet {
        StageSet::new(Arc::new(AboveFirstTaps(inner)))
    }
}

impl Stages for AboveFirstTaps {
    fn solve(&self, x: &Image, y: &Image, s: &FlowSettings) -> Result<FlowField> {
        self.0.get().solve(x, y, s)
    }

    fn solve_pair(&self, a: &Image, b: &Image, s: &FlowSettings) -> Result<FlowPair> {
        self.0.get().solve_pair(a, b, s)
    }

    fn solve_pyramids_with(&self, x: &Pyramid, y: &Pyramid, s: &FlowSettings, e: SolveExtras<'_>) -> Result<FlowField> {
        self.0.get().solve_pyramids_with(x, y, s, e)
    }

    fn forward_warp_with(&self, out: &mut Image, src: &Image, flow: &FlowField, amount: f32, opts: ForwardOpts, want_coverage: bool) -> Result<Option<Coverage>> {
        self.0.get().forward_warp_with(out, src, flow, amount, opts, want_coverage)
    }

    fn vertical_taps(&self) -> VerticalTaps {
        VerticalTaps::AboveFirst
    }
}

/// The stages a render uses: the CPU ([`StageSet::default`]) or a device implementation shared
/// by every frame of a run. Cloning shares the implementation. Two sets are equal when both are
/// the CPU or both share one implementation.
#[derive(Clone, Default)]
pub struct StageSet(Option<Arc<dyn Stages>>);

impl StageSet {
    /// The CPU.
    pub fn cpu() -> StageSet {
        StageSet(None)
    }

    /// A device implementation.
    pub fn new(stages: Arc<dyn Stages>) -> StageSet {
        StageSet(Some(stages))
    }

    /// The set the host passed in `cx.executor` (a `StageSet` behind the `Any`), else the CPU.
    pub fn from_context(cx: &TemporalContext) -> StageSet {
        cx.executor.as_ref().and_then(|any| any.downcast_ref::<StageSet>()).cloned().unwrap_or_default()
    }

    /// The stages to call.
    pub fn get(&self) -> &dyn Stages {
        match &self.0 {
            Some(s) => s.as_ref(),
            None => &Cpu,
        }
    }

    /// True for the CPU.
    pub fn is_cpu(&self) -> bool {
        self.0.is_none()
    }
}

impl fmt::Debug for StageSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(if self.is_cpu() { "StageSet(cpu)" } else { "StageSet(device)" })
    }
}

impl PartialEq for StageSet {
    fn eq(&self, other: &StageSet) -> bool {
        match (&self.0, &other.0) {
            (None, None) => true,
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fx_core::{Depth, Rational, TimeBase};

    fn cx() -> TemporalContext {
        TemporalContext::new(4, 4, Depth::U8, TimeBase::default_for_rate(Rational::new(24, 1).unwrap()), 0, 1)
    }

    #[test]
    fn a_context_without_an_executor_is_the_cpu() {
        let set = StageSet::from_context(&cx());
        assert!(set.is_cpu());
        assert_eq!(set, StageSet::cpu());
        assert_eq!(format!("{set:?}"), "StageSet(cpu)");
    }

    #[test]
    fn the_vertical_taps_follow_the_wrappers() {
        assert_eq!(StageSet::cpu().get().vertical_taps(), VerticalTaps::BelowFirst);
        assert_eq!(GouraudForward::wrap(StageSet::cpu()).get().vertical_taps(), VerticalTaps::BelowFirst);
        let both = AboveFirstTaps::wrap(GouraudForward::wrap(StageSet::cpu()));
        assert_eq!(both.get().vertical_taps(), VerticalTaps::AboveFirst);
        // a Gouraud set around it passes the order on
        assert_eq!(GouraudForward::wrap(AboveFirstTaps::wrap(StageSet::cpu())).get().vertical_taps(), VerticalTaps::AboveFirst);
    }

    #[test]
    fn a_context_carries_a_shared_set() {
        let shared = StageSet::new(Arc::new(Cpu));
        let mut c = cx();
        c.executor = Some(Arc::new(shared.clone()));
        let got = StageSet::from_context(&c);
        assert!(!got.is_cpu());
        assert_eq!(got, shared);
        assert_ne!(got, StageSet::new(Arc::new(Cpu)), "another implementation is another set");
        // something else behind the Any is ignored
        c.executor = Some(Arc::new(17u32));
        assert!(StageSet::from_context(&c).is_cpu());
    }
}
