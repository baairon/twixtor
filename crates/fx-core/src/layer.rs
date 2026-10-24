//! Sources an effect can pull frames from at times of its choosing.
//!
//! A temporal effect gets the main input plus any extra named layers its parameters point at.
//! Asking for a time outside a layer gives `Ok(None)`.

use crate::{Depth, Frame, FxError, ParamCurves, Result, TimeBase};
use serde::Serialize;
use std::collections::BTreeMap;
use std::sync::Arc;

pub trait Layer: Send + Sync {
    fn width(&self) -> usize;
    fn height(&self) -> usize;
    fn depth(&self) -> Depth;
    /// The frame shown at `time` ticks of layer time, `None` outside the layer.
    fn frame_at(&self, time: i64) -> Result<Option<Arc<Frame>>>;
}

/// A layer held in memory: frame `i` covers the ticks `i * time_step .. (i + 1) * time_step`.
pub struct MemLayer {
    time: TimeBase,
    frames: Vec<Arc<Frame>>,
}

impl MemLayer {
    pub fn new(time: TimeBase, frames: Vec<Frame>) -> Result<MemLayer> {
        let first = frames.first().ok_or_else(|| FxError::Mismatch("a layer needs at least one frame".into()))?;
        let (w, h, d) = (first.width, first.height, first.depth());
        if frames.iter().any(|f| (f.width, f.height, f.depth()) != (w, h, d)) {
            return Err(FxError::Mismatch("all frames of a layer must share size and depth".into()));
        }
        Ok(MemLayer { time, frames: frames.into_iter().map(Arc::new).collect() })
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
}

impl Layer for MemLayer {
    fn width(&self) -> usize {
        self.frames[0].width
    }

    fn height(&self) -> usize {
        self.frames[0].height
    }

    fn depth(&self) -> Depth {
        self.frames[0].depth()
    }

    fn frame_at(&self, time: i64) -> Result<Option<Arc<Frame>>> {
        let n = self.time.frame_of(time);
        Ok(usize::try_from(n).ok().and_then(|i| self.frames.get(i)).cloned())
    }
}

/// An extra input layer an effect can use, as listed by `vfx list`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct LayerSpec {
    /// Stable id used on the command line, in snake_case.
    pub id: &'static str,
    pub name: &'static str,
    /// An optional layer may be left out; the effect then falls back to its documented default.
    pub optional: bool,
}

/// The extra layers supplied to one render, by id.
#[derive(Default)]
pub struct LayerSet(BTreeMap<String, Arc<dyn Layer>>);

impl LayerSet {
    pub fn new() -> LayerSet {
        LayerSet::default()
    }

    pub fn insert(&mut self, id: &str, layer: Arc<dyn Layer>) {
        self.0.insert(id.to_string(), layer);
    }

    pub fn get(&self, id: &str) -> Option<&Arc<dyn Layer>> {
        self.0.get(id)
    }

    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(String::as_str)
    }
}

/// Everything besides the frame description that a temporal render reads.
pub struct Env<'a> {
    pub params: &'a ParamCurves,
    /// The main input layer.
    pub input: &'a dyn Layer,
    pub layers: &'a LayerSet,
}

impl Env<'_> {
    /// The extra layer `id`, if the host supplied one.
    pub fn layer(&self, id: &str) -> Option<&dyn Layer> {
        self.layers.get(id).map(|l| l.as_ref())
    }
}

/// One layer fetch a render will make: the equivalent of a pre-render checkout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Checkout {
    /// `None` is the main input layer.
    pub layer: Option<&'static str>,
    /// Ticks of layer time.
    pub time: i64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rational;

    fn clip(n: usize) -> MemLayer {
        let frames = (0..n)
            .map(|i| {
                let mut f = Frame::new(2, 2, Depth::U8);
                if let crate::Pixels::U8(px) = &mut f.pixels {
                    px.iter_mut().for_each(|p| *p = [i as u8, 0, 0, 255]);
                }
                f
            })
            .collect();
        MemLayer::new(TimeBase::from_fps(Rational::new(24000, 1001).unwrap()), frames).unwrap()
    }

    fn marker(layer: &MemLayer, t: i64) -> Option<u8> {
        layer.frame_at(t).unwrap().map(|f| match &f.pixels {
            crate::Pixels::U8(px) => px[0][0],
            _ => unreachable!(),
        })
    }

    #[test]
    fn frame_spans_are_half_open() {
        let l = clip(3);
        assert_eq!(marker(&l, 0), Some(0));
        assert_eq!(marker(&l, 1000), Some(0));
        assert_eq!(marker(&l, 1001), Some(1));
        assert_eq!(marker(&l, 3002), Some(2));
    }

    #[test]
    fn outside_the_layer_is_empty() {
        let l = clip(3);
        assert_eq!(marker(&l, -1), None);
        assert_eq!(marker(&l, 3003), None);
        assert_eq!(marker(&l, i64::MIN), None);
        assert_eq!((l.width(), l.height(), l.depth(), l.frame_count()), (2, 2, Depth::U8, 3));
    }

    #[test]
    fn mixed_frames_are_rejected() {
        let base = TimeBase::from_fps(Rational::new(24, 1).unwrap());
        assert!(MemLayer::new(base, vec![]).is_err());
        assert!(MemLayer::new(base, vec![Frame::new(2, 2, Depth::U8), Frame::new(3, 2, Depth::U8)]).is_err());
        assert!(MemLayer::new(base, vec![Frame::new(2, 2, Depth::U8), Frame::new(2, 2, Depth::U16)]).is_err());
    }

    #[test]
    fn layer_sets_hand_out_what_was_inserted() {
        let mut set = LayerSet::new();
        set.insert("alt_source_a", Arc::new(clip(1)));
        assert!(set.get("alt_source_a").is_some());
        assert!(set.get("alt_source_b").is_none());
        assert_eq!(set.ids().collect::<Vec<_>>(), ["alt_source_a"]);
    }
}
