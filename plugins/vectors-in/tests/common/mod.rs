//! Helpers shared by the integration tests of plugin 04: a rig that holds the source clip, the vector layers, the
//! other layers and the parameter curves of one run and renders or lists the dependencies of one output frame
//! through the `Effect` trait, exactly as a host does.
#![allow(dead_code)]

pub mod moving;

use fx_core::{
    Checkout, Depth, Effect, Env, Frame, Layer, LayerSet, MemLayer, ParamCurves, Pixels, Rational, Result, TemporalContext, TimeBase,
};
use vectors_in::VectorsIn;
use retime_engine::flow::FlowField;
use retime_engine::image::Fmt;
use retime_engine::vectors::encode_vectors_image;
use std::sync::{Arc, Mutex};

/// 30 fps with 100 ticks per frame (`TimeBase::default_for_rate`).
pub fn time() -> TimeBase {
    TimeBase::default_for_rate(Rational::new(30, 1).unwrap())
}

/// The default parameters of plugin 04 with `--param ID=VALUE` style overrides (plain values and `V@F;V@F` keys).
pub fn curves(params: &[(&str, &str)]) -> ParamCurves {
    let mut c = ParamCurves::new(VectorsIn.params());
    for (id, text) in params {
        c.parse_cli(id, text, time()).unwrap_or_else(|e| panic!("{id}={text}: {e}"));
    }
    c
}

pub fn f8(width: usize, height: usize, px: &[[u8; 4]]) -> Frame {
    assert_eq!(px.len(), width * height);
    Frame { width, height, pixels: Pixels::U8(px.to_vec()) }
}

pub fn layer(frames: Vec<Frame>) -> MemLayer {
    MemLayer::new(time(), frames).unwrap()
}

/// A layer that logs the times it is asked for.
pub struct Recorder {
    inner: MemLayer,
    log: Mutex<Vec<i64>>,
}

impl Recorder {
    pub fn new(frames: Vec<Frame>) -> Arc<Recorder> {
        Arc::new(Recorder { inner: layer(frames), log: Mutex::new(Vec::new()) })
    }

    /// The logged times, oldest first, and clears the log.
    pub fn take(&self) -> Vec<i64> {
        std::mem::take(&mut self.log.lock().unwrap())
    }
}

impl Layer for Recorder {
    fn width(&self) -> usize {
        self.inner.width()
    }
    fn height(&self) -> usize {
        self.inner.height()
    }
    fn depth(&self) -> Depth {
        self.inner.depth()
    }
    fn frame_at(&self, time: i64) -> Result<Option<Arc<Frame>>> {
        self.log.lock().unwrap().push(time);
        self.inner.frame_at(time)
    }
}

/// An encoded vector frame (wire, opaque) holding the constant flow `(vx, vy)` pixels, scaled by `max_displace` as
/// Create Vectors writes it (`vectors::encode_vectors`).
pub fn vector_frame(depth: Depth, width: usize, height: usize, vx: f32, vy: f32, max_displace: f32) -> Frame {
    let mut flow = FlowField::new(width, height);
    for y in 0..height {
        for x in 0..width {
            flow.set(x, y, [vx, vy]);
        }
    }
    encode_vectors_image(width, height, Fmt::from_depth(depth), &flow, max_displace).unwrap().to_frame()
}

/// `count` identical vector frames of a layer that carries the constant flow `(vx, vy)` of the frame as the host sees
/// it. The plugins work bottom row first (`vectors_in::glue`, row order), so the frames hold the vertical component
/// pointing up the rows: `vector_frame` of `(vx, -vy)`.
pub fn vector_clip(depth: Depth, width: usize, height: usize, count: usize, vx: f32, vy: f32, max_displace: f32) -> Vec<Frame> {
    (0..count).map(|_| vector_frame(depth, width, height, vx, -vy, max_displace)).collect()
}

/// The vector layers of a clip that translates by `(vx, vy)` pixels per frame (`moving::Motion`): `Vectors: Next`
/// holds the flow of `A` to `B` (`A(x) = B(x + v)`, so `+v`), `Vectors: Prev` the flow of `B` back to `A` (`-v`).
/// Returns `(prev, next)`.
pub fn motion_vectors(depth: Depth, width: usize, height: usize, count: usize, vx: f32, vy: f32, max_displace: f32) -> (Vec<Frame>, Vec<Frame>) {
    (vector_clip(depth, width, height, count, -vx, -vy, max_displace), vector_clip(depth, width, height, count, vx, vy, max_displace))
}

/// One run: the source clip (the input layer), extra layers and parameters, at one output size and depth.
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

    pub fn with_dyn_layer(mut self, id: &str, l: Arc<dyn Layer>) -> Rig {
        self.layers.insert(id, l);
        self
    }

    /// The vector layers of a translating clip, see [`motion_vectors`].
    pub fn with_vectors(self, vx: f32, vy: f32, max_displace: f32) -> Rig {
        let (prev, next) = motion_vectors(self.depth, self.width, self.height, self.frames as usize + 8, vx, vy, max_displace);
        self.with_layer("vectors_prev", prev).with_layer("vectors_next", next)
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
        VectorsIn.render_temporal(cx, &self.env(), &mut out)?;
        Ok(out)
    }

    /// Output frame `n`.
    pub fn render(&self, n: u64) -> Result<Frame> {
        self.render_cx(&self.cx(n))
    }

    pub fn deps(&self, n: u64) -> Result<Vec<Checkout>> {
        VectorsIn.dependencies(&self.cx(n), &self.env())
    }
}

/// The error fill as a wire frame. Size mismatch (red): half alpha red at every depth. Missing
/// layer, plugin 04: (A, R, G, B) = (128, 0, 0, 255) at 8 bpc, (0, 0, 0, 32768) at 16 bpc (engine
/// range, alpha 0) and (0.0, 0, 0, 1.0) at 32 bpc: blue, with alpha 0 at 16 and 32 bpc.
pub fn fill(depth: Depth, red: bool, width: usize, height: usize) -> Frame {
    let n = width * height;
    let pixels = match (depth, red) {
        (Depth::U8, true) => Pixels::U8(vec![[255, 0, 0, 128]; n]),
        (Depth::U8, false) => Pixels::U8(vec![[0, 0, 255, 128]; n]),
        (Depth::U16, true) => Pixels::U16(vec![[65535, 0, 0, 32768]; n]),
        (Depth::U16, false) => Pixels::U16(vec![[0, 0, 65535, 0]; n]),
        (Depth::F32, true) => Pixels::F32(vec![[1.0, 0.0, 0.0, 0.5]; n]),
        (Depth::F32, false) => Pixels::F32(vec![[0.0, 0.0, 1.0, 0.0]; n]),
    };
    Frame { width, height, pixels }
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
