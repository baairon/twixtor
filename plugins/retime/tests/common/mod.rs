//! Helpers shared by the integration tests of plugin 01: a rig that holds the clip, the extra layers and
//! the parameter curves of one run and renders or lists the dependencies of one output frame through the
//! `Effect` trait, exactly as a host does.
#![allow(dead_code)]

pub mod goldens;
pub mod moving;

use fx_core::{Checkout, Depth, Effect, Env, Frame, LayerSet, MemLayer, ParamCurves, Pixels, Rational, Result, TemporalContext, TimeBase};
use retime::Retime;
use std::sync::Arc;

/// 30 fps with 100 ticks per frame (`TimeBase::default_for_rate`), the time base the tests use.
pub fn time() -> TimeBase {
    TimeBase::default_for_rate(Rational::new(30, 1).unwrap())
}

/// The default parameters of plugin 01 with `--param ID=VALUE` style overrides (plain values and `V@F;V@F` keys).
pub fn curves(params: &[(&str, &str)]) -> ParamCurves {
    let mut c = ParamCurves::new(Retime.params());
    for (id, text) in params {
        c.parse_cli(id, text, time()).unwrap_or_else(|e| panic!("{id}={text}: {e}"));
    }
    c
}

pub fn f8(width: usize, height: usize, px: &[[u8; 4]]) -> Frame {
    assert_eq!(px.len(), width * height);
    Frame { width, height, pixels: Pixels::U8(px.to_vec()) }
}

pub fn f16(width: usize, height: usize, px: &[[u16; 4]]) -> Frame {
    assert_eq!(px.len(), width * height);
    Frame { width, height, pixels: Pixels::U16(px.to_vec()) }
}

pub fn f32s(width: usize, height: usize, px: &[[f32; 4]]) -> Frame {
    assert_eq!(px.len(), width * height);
    Frame { width, height, pixels: Pixels::F32(px.to_vec()) }
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
        Retime.render_temporal(cx, &self.env(), &mut out)?;
        Ok(out)
    }

    /// Output frame `n`.
    pub fn render(&self, n: u64) -> Result<Frame> {
        self.render_cx(&self.cx(n))
    }

    pub fn deps(&self, n: u64) -> Result<Vec<Checkout>> {
        Retime.dependencies(&self.cx(n), &self.env())
    }
}

/// The golden frames of one depth, built from `goldens`.
pub struct Set {
    pub depth: Depth,
    pub width: usize,
    pub height: usize,
    /// The input clip, frames 0 to 3.
    pub clip: Vec<Frame>,
    /// The round trip of each clip frame.
    pub rt: Vec<Frame>,
    pub b01: Frame,
    pub b12: Frame,
    pub b23: Frame,
    pub b3t: Frame,
    pub b01_25: Frame,
    pub b01_75: Frame,
}

impl Set {
    pub fn transparent(&self) -> Frame {
        Frame::new(self.width, self.height, self.depth)
    }

    /// The round trip of clip frame `i`, the transparent frame outside the clip.
    pub fn rt_or_clear(&self, i: usize) -> Frame {
        self.rt.get(i).cloned().unwrap_or_else(|| self.transparent())
    }

    /// The blend of frames `k` and `k + 1` at weight 0.5 (the last frame blends with transparency).
    pub fn half(&self, k: usize) -> Frame {
        match k {
            0 => self.b01.clone(),
            1 => self.b12.clone(),
            2 => self.b23.clone(),
            3 => self.b3t.clone(),
            _ => self.transparent(),
        }
    }

    pub fn rig(&self, params: &[(&str, &str)]) -> Rig {
        Rig::new(self.clip.clone(), params)
    }
}

pub fn sets() -> Vec<Set> {
    use goldens::{d16, d32, d8};
    let s8 = |px: &[[u8; 4]]| f8(3, 2, px);
    let s16 = |px: &[[u16; 4]]| f16(2, 2, px);
    let s32 = |px: &[[f32; 4]]| f32s(2, 2, px);
    vec![
        Set {
            depth: Depth::U8,
            width: 3,
            height: 2,
            clip: vec![s8(&d8::F0), s8(&d8::F1), s8(&d8::F2), s8(&d8::F3)],
            rt: vec![s8(&d8::R0), s8(&d8::R1), s8(&d8::R2), s8(&d8::R3)],
            b01: s8(&d8::B01),
            b12: s8(&d8::B12),
            b23: s8(&d8::B23),
            b3t: s8(&d8::B3T),
            b01_25: s8(&d8::B01_25),
            b01_75: s8(&d8::B01_75),
        },
        Set {
            depth: Depth::U16,
            width: 2,
            height: 2,
            clip: vec![s16(&d16::F0), s16(&d16::F1), s16(&d16::F2), s16(&d16::F3)],
            rt: vec![s16(&d16::R0), s16(&d16::R1), s16(&d16::R2), s16(&d16::R3)],
            b01: s16(&d16::B01),
            b12: s16(&d16::B12),
            b23: s16(&d16::B23),
            b3t: s16(&d16::B3T),
            b01_25: s16(&d16::B01_25),
            b01_75: s16(&d16::B01_75),
        },
        Set {
            depth: Depth::F32,
            width: 2,
            height: 2,
            clip: vec![s32(&d32::F0), s32(&d32::F1), s32(&d32::F2), s32(&d32::F3)],
            rt: vec![s32(&d32::R0), s32(&d32::R1), s32(&d32::R2), s32(&d32::R3)],
            b01: s32(&d32::B01),
            b12: s32(&d32::B12),
            b23: s32(&d32::B23),
            b3t: s32(&d32::B3T),
            b01_25: s32(&d32::B01_25),
            b01_75: s32(&d32::B01_75),
        },
    ]
}

/// The error fill as a wire frame: half alpha red (size mismatch) or half alpha blue (missing layer).
pub fn fill(depth: Depth, red: bool, width: usize, height: usize) -> Frame {
    let n = width * height;
    let pixels = match (depth, red) {
        (Depth::U8, true) => Pixels::U8(vec![[255, 0, 0, 128]; n]),
        (Depth::U8, false) => Pixels::U8(vec![[0, 0, 255, 128]; n]),
        (Depth::U16, true) => Pixels::U16(vec![[65535, 0, 0, 32768]; n]),
        (Depth::U16, false) => Pixels::U16(vec![[0, 0, 65535, 32768]; n]),
        (Depth::F32, true) => Pixels::F32(vec![[1.0, 0.0, 0.0, 0.5]; n]),
        (Depth::F32, false) => Pixels::F32(vec![[0.0, 0.0, 1.0, 0.5]; n]),
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
