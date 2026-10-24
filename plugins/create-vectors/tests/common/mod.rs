//! Helpers shared by the integration tests of plugin 02: a rig that holds the clip, the extra layers and the
//! parameter curves of one run and renders or lists the dependencies of one output frame through the `Effect`
//! trait, exactly as a host does, plus the decoding and error measures of the ground truth tests.
#![allow(dead_code)]

pub mod moving;

use fx_core::{
    Checkout, Depth, Effect, Env, Frame, LayerSet, MemLayer, ParamCurves, Pixels, Rational, Result, TemporalContext,
    TimeBase,
};
use create_vectors::CreateVectors;
use retime_engine::flow::FlowField;
use retime_engine::image::Image;
use retime_engine::vectors::decode_vectors;
use std::sync::Arc;

/// 30 fps with 100 ticks per frame (`TimeBase::default_for_rate`).
pub fn time() -> TimeBase {
    TimeBase::default_for_rate(Rational::new(30, 1).unwrap())
}

/// The default parameters of plugin 02 with `--param ID=VALUE` style overrides (plain values and `V@F;V@F` keys).
pub fn curves(params: &[(&str, &str)]) -> ParamCurves {
    let mut c = ParamCurves::new(CreateVectors.params());
    for (id, text) in params {
        c.parse_cli(id, text, time()).unwrap_or_else(|e| panic!("{id}={text}: {e}"));
    }
    c
}

pub fn layer(frames: Vec<Frame>) -> MemLayer {
    MemLayer::new(time(), frames).unwrap()
}

/// One run: the input clip, extra layers and parameters, at one output size and depth.
pub struct Rig {
    pub curves: ParamCurves,
    pub input: MemLayer,
    pub layers: LayerSet,
    pub depth: Depth,
    pub width: usize,
    pub height: usize,
    pub frames: u64,
}

impl Rig {
    pub fn new(clip: Vec<Frame>, params: &[(&str, &str)]) -> Rig {
        let (width, height, depth) = (clip[0].width, clip[0].height, clip[0].depth());
        let frames = clip.len() as u64;
        Rig { curves: curves(params), input: layer(clip), layers: LayerSet::new(), depth, width, height, frames }
    }

    pub fn with_layer(mut self, id: &str, clip: Vec<Frame>) -> Rig {
        self.layers.insert(id, Arc::new(layer(clip)));
        self
    }

    pub fn with_frames(mut self, frames: u64) -> Rig {
        self.frames = frames;
        self
    }

    pub fn cx(&self, n: u64) -> TemporalContext {
        TemporalContext::new(self.width, self.height, self.depth, time(), n, self.frames)
    }

    pub fn env(&self) -> Env<'_> {
        Env { params: &self.curves, input: &self.input, layers: &self.layers }
    }

    pub fn render_cx(&self, cx: &TemporalContext) -> Result<Frame> {
        let mut out = Frame::new(cx.width, cx.height, cx.depth);
        CreateVectors.render_temporal(cx, &self.env(), &mut out)?;
        Ok(out)
    }

    /// Output frame `n`.
    pub fn render(&self, n: u64) -> Result<Frame> {
        self.render_cx(&self.cx(n))
    }

    pub fn deps(&self, n: u64) -> Result<Vec<Checkout>> {
        CreateVectors.dependencies(&self.cx(n), &self.env())
    }
}

/// The vector field a Create Vectors output frame carries, as a consumer (plugin 03 or 04) reads it: the wire frame goes
/// through the input prepare (`Image::from_frame`) and the decode with the `Max Displace` the effect was run with.
pub fn decode(frame: &Frame, max_displace: f32) -> FlowField {
    decode_vectors(&Image::from_frame(frame), max_displace)
}

/// Error of a flow field against a constant true flow over the pixels at least `margin` away from
/// every edge: the mean and the maximum of the Euclidean error, and the mean vector.
pub fn flow_error(flow: &FlowField, truth: (f64, f64), margin: usize) -> (f64, f64, (f64, f64)) {
    let (mut sum, mut max, mut n) = (0.0f64, 0.0f64, 0usize);
    let (mut mx, mut my) = (0.0f64, 0.0f64);
    for y in margin..flow.height - margin {
        for x in margin..flow.width - margin {
            let [u, v] = flow.at(x, y);
            let (u, v) = (u as f64, v as f64);
            let e = ((u - truth.0).powi(2) + (v - truth.1).powi(2)).sqrt();
            sum += e;
            max = max.max(e);
            mx += u;
            my += v;
            n += 1;
        }
    }
    let n = n as f64;
    (sum / n, max, (mx / n, my / n))
}

/// One pixel of a wire frame as `[R, G, B, A]` in the stored values.
pub fn px(f: &Frame, x: usize, y: usize) -> [f64; 4] {
    let i = y * f.width + x;
    match &f.pixels {
        Pixels::U8(p) => p[i].map(|v| v as f64),
        Pixels::U16(p) => p[i].map(|v| v as f64),
        Pixels::F32(p) => p[i].map(|v| v as f64),
    }
}

/// A deterministic pseudo random frame with every alpha class (0, 1, nearly full, full, random).
pub fn random_frame(depth: Depth, width: usize, height: usize, seed: &mut u64) -> Frame {
    fn next(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *seed >> 33
    }
    let mut f = Frame::new(width, height, depth);
    match &mut f.pixels {
        Pixels::U8(px) => {
            for p in px.iter_mut() {
                let a = [0u8, 1, 254, 255, (next(seed) & 255) as u8][(next(seed) % 5) as usize];
                *p = [(next(seed) & 255) as u8, (next(seed) & 255) as u8, (next(seed) & 255) as u8, a];
            }
        }
        Pixels::U16(px) => {
            for p in px.iter_mut() {
                let a = [0u16, 1, 65534, 65535, (next(seed) & 0xffff) as u16][(next(seed) % 5) as usize];
                *p = [(next(seed) & 0xffff) as u16, (next(seed) & 0xffff) as u16, (next(seed) & 0xffff) as u16, a];
            }
        }
        Pixels::F32(px) => {
            for p in px.iter_mut() {
                let u = |s: &mut u64| (next(s) & 0xffffff) as f32 / 16777216.0;
                let a = [0.0f32, 1e-13, 1.0, u(seed), u(seed)][(next(seed) % 5) as usize];
                *p = [u(seed), u(seed) * 2.0 - 0.5, u(seed), a];
            }
        }
    }
    f
}

/// `count` random frames of one size.
pub fn random_clip(depth: Depth, width: usize, height: usize, count: usize, seed: u64) -> Vec<Frame> {
    let mut seed = seed;
    (0..count).map(|_| random_frame(depth, width, height, &mut seed)).collect()
}
