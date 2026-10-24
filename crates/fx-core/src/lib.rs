//! Shared effect interface for every plugin family.
//!
//! Frames keep the bit depth they arrived in, so an effect can run exact integer math at 8 and 16
//! bpc instead of round tripping through float.

mod curves;
mod fixed;
mod frame;
mod layer;
mod params;
mod time;

pub use curves::{Keyframe, ParamCurves};
pub use fixed::{fixed_to_float, float_to_fixed};
pub use frame::{Depth, Frame, Pixels};
pub use layer::{Checkout, Env, Layer, LayerSet, LayerSpec, MemLayer};
pub use params::{ParamKind, ParamSpec, ParamValue, Params};
pub use time::{FieldOrder, Quality, Rational, TemporalContext, TimeBase};

use serde::Serialize;
use std::fmt;

/// Per-render context that does not belong to the pixels themselves.
#[derive(Clone, Copy, Debug, Default)]
pub struct RenderContext {
    /// Zero-based frame number within the clip.
    pub frame_index: u64,
    /// Time in seconds of this frame.
    pub time: f64,
    /// Frames per second of the clip.
    pub fps: f64,
}

#[derive(Debug)]
pub enum FxError {
    UnknownParam(String),
    BadParam { id: String, reason: String },
    Mismatch(String),
    Unsupported(String),
    /// A layer or file could not be read. Hosts map it to their I/O exit code.
    Io(String),
}

impl fmt::Display for FxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FxError::UnknownParam(id) => write!(f, "unknown param `{id}`"),
            FxError::BadParam { id, reason } => write!(f, "bad value for `{id}`: {reason}"),
            FxError::Mismatch(msg) => write!(f, "frame mismatch: {msg}"),
            FxError::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            FxError::Io(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for FxError {}

pub type Result<T> = std::result::Result<T, FxError>;

/// How a host has to drive an effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectKind {
    /// One output frame from the one input frame at the same time: [`Effect::render`].
    Frame,
    /// Needs other frames, extra layers or parameters sampled over time: [`Effect::render_temporal`].
    Temporal,
}

pub trait Effect: Send + Sync {
    /// Stable CLI identifier, for example `retime`.
    fn id(&self) -> &'static str;
    /// Human readable name.
    fn name(&self) -> &'static str;
    /// Parameters, in the order `vfx list` shows them.
    fn params(&self) -> Vec<ParamSpec>;
    /// Renders `input` into `output`. Both frames share size and depth. A temporal
    /// effect can leave the default, which reports that it needs [`Effect::render_temporal`].
    fn render(
        &self,
        _ctx: &RenderContext,
        _params: &Params,
        _input: &Frame,
        _output: &mut Frame,
    ) -> Result<()> {
        Err(FxError::Unsupported(format!("{} is a temporal effect and needs the temporal interface", self.id())))
    }

    /// Which entry point a host calls. The default is the single frame [`Effect::render`].
    fn kind(&self) -> EffectKind {
        EffectKind::Frame
    }

    /// Extra input layers a temporal effect can use besides the main input.
    fn layers(&self) -> Vec<LayerSpec> {
        Vec::new()
    }

    /// The layer fetches the render of `cx.frame_index` will make, in issue order. A caller may use
    /// it to prefetch; tests use it to pin the times a render reads. Temporal effects only.
    fn dependencies(&self, _cx: &TemporalContext, _env: &Env<'_>) -> Result<Vec<Checkout>> {
        Ok(Vec::new())
    }

    /// Renders the frame described by `cx`. `output` has the size and depth of `cx`.
    /// Temporal effects only.
    fn render_temporal(&self, _cx: &TemporalContext, _env: &Env<'_>, _output: &mut Frame) -> Result<()> {
        Err(FxError::Unsupported(format!("{} is not a temporal effect", self.id())))
    }
}

/// Fills in defaults for any param the caller left out and rejects unknown ids.
pub fn resolve_params(effect: &dyn Effect, given: &Params) -> Result<Params> {
    let specs = effect.params();
    for id in given.keys() {
        if !specs.iter().any(|s| s.id == id) {
            return Err(FxError::UnknownParam(id.to_string()));
        }
    }
    let mut out = Params::default();
    for spec in &specs {
        let value = match given.get(spec.id) {
            Some(v) => spec.check(v)?,
            None => spec.default_value(),
        };
        out.set(spec.id, value);
    }
    Ok(out)
}

/// Runs `f` once per row in parallel. `f` gets the row index and the output row.
pub fn par_rows<T: Send>(data: &mut [T], width: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    use rayon::prelude::*;
    data.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| f(y, row));
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Temporal;

    impl Effect for Temporal {
        fn id(&self) -> &'static str {
            "t"
        }
        fn name(&self) -> &'static str {
            "t"
        }
        fn params(&self) -> Vec<ParamSpec> {
            Vec::new()
        }
    }

    #[test]
    fn an_effect_may_leave_the_single_frame_render_to_the_default() {
        let mut out = Frame::new(1, 1, Depth::U8);
        let r = Temporal.render(&RenderContext::default(), &Params::default(), &Frame::new(1, 1, Depth::U8), &mut out);
        assert!(matches!(r, Err(FxError::Unsupported(m)) if m.contains("temporal")));
        assert_eq!(Temporal.kind(), EffectKind::Frame);
    }

    #[test]
    fn io_errors_display_their_message_only() {
        assert_eq!(FxError::Io("input clip a.raw: unexpected end of file".into()).to_string(), "input clip a.raw: unexpected end of file");
    }
}
