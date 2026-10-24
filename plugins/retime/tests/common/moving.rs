//! A synthetic clip with a KNOWN motion, the ground truth of the vector path tests: a smooth
//! textured pattern (a sum of plane waves, different phases per colour channel) that translates by
//! `(vx, vy)` pixels per frame. Frame `k` of the clip is the pattern shifted by `k` frames and the
//! frame at any fractional source position `s` is known in closed form, so a retimed frame can be
//! compared with the analytically exact in-between frame.
#![allow(dead_code)]

use fx_core::{Depth, Frame, Pixels};

/// Motion per frame, in pixels of the full resolution image (positive x is to the right, positive y down).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Motion {
    pub vx: f64,
    pub vy: f64,
}

/// The velocity of the main tests: 3.4 px per frame in x and 1.7 px in y.
pub const DEFAULT: Motion = Motion { vx: 3.4, vy: 1.7 };

/// One plane wave: wavelength in pixels, direction in degrees, amplitude.
const WAVES: [(f64, f64, f64); 4] = [(26.0, 20.0, 0.30), (19.0, 110.0, 0.28), (13.0, 60.0, 0.22), (34.0, 160.0, 0.20)];

/// The pattern of colour channel `c` (0 R, 1 G, 2 B) at the real position `(x, y)`, in `0..=1`.
pub fn pattern(c: usize, x: f64, y: f64) -> f64 {
    let mut s = 0.0;
    for (i, &(lambda, deg, amp)) in WAVES.iter().enumerate() {
        let th = deg.to_radians();
        let phase = 1.7 * i as f64 + 2.3 * c as f64;
        s += amp * (std::f64::consts::TAU * ((x * th.cos() + y * th.sin()) / lambda) + phase).sin();
    }
    0.5 + 0.5 * s
}

impl Motion {
    /// The exact colour of pixel `(x, y)` of the frame at source position `s` (in frames; frame `k` is `s = k`).
    pub fn rgb(self, x: usize, y: usize, s: f64) -> [f64; 3] {
        let (px, py) = (x as f64 - self.vx * s, y as f64 - self.vy * s);
        [pattern(0, px, py), pattern(1, px, py), pattern(2, px, py)]
    }

    /// The wire frame (opaque) at source position `s`, quantised to the depth.
    pub fn frame_at(self, depth: Depth, width: usize, height: usize, s: f64) -> Frame {
        let mut f = Frame::new(width, height, depth);
        for y in 0..height {
            for x in 0..width {
                let [r, g, b] = self.rgb(x, y, s);
                match &mut f.pixels {
                    Pixels::U8(px) => px[y * width + x] = [q8(r), q8(g), q8(b), 255],
                    Pixels::U16(px) => px[y * width + x] = [q16(r), q16(g), q16(b), 65535],
                    Pixels::F32(px) => px[y * width + x] = [r as f32, g as f32, b as f32, 1.0],
                }
            }
        }
        f
    }

    /// The wire frame of the pattern seen through a shutter: the mean of the exact frames at the source positions
    /// `s - half ..= s + half` (`samples + 1` evenly spaced positions), quantised to the depth. This is what a
    /// motion blur of `2 * half` source frames centred on `s` looks like.
    pub fn blurred_frame_at(self, depth: Depth, width: usize, height: usize, s: f64, half: f64, samples: usize) -> Frame {
        let mut f = Frame::new(width, height, depth);
        for y in 0..height {
            for x in 0..width {
                let mut acc = [0.0f64; 3];
                for i in 0..=samples {
                    let t = s - half + 2.0 * half * i as f64 / samples as f64;
                    let c = self.rgb(x, y, t);
                    for k in 0..3 {
                        acc[k] += c[k];
                    }
                }
                let [r, g, b] = acc.map(|v| v / (samples + 1) as f64);
                match &mut f.pixels {
                    Pixels::U8(px) => px[y * width + x] = [q8(r), q8(g), q8(b), 255],
                    Pixels::U16(px) => px[y * width + x] = [q16(r), q16(g), q16(b), 65535],
                    Pixels::F32(px) => px[y * width + x] = [r as f32, g as f32, b as f32, 1.0],
                }
            }
        }
        f
    }

    /// A clip of `count` frames, frame `k` at position `k`.
    pub fn clip(self, depth: Depth, width: usize, height: usize, count: usize) -> Vec<Frame> {
        (0..count).map(|k| self.frame_at(depth, width, height, k as f64)).collect()
    }
}

/// [`DEFAULT`] motion.
pub fn frame_at(depth: Depth, width: usize, height: usize, s: f64) -> Frame {
    DEFAULT.frame_at(depth, width, height, s)
}

/// [`DEFAULT`] motion.
pub fn clip(depth: Depth, width: usize, height: usize, count: usize) -> Vec<Frame> {
    DEFAULT.clip(depth, width, height, count)
}

fn q8(v: f64) -> u8 {
    (v * 255.0 + 0.5) as u8
}

fn q16(v: f64) -> u16 {
    (v * 65535.0 + 0.5) as u16
}

/// The three colour channels of a pixel as fractions of full scale.
fn norm(f: &Frame, x: usize, y: usize) -> [f64; 3] {
    let i = y * f.width + x;
    match &f.pixels {
        Pixels::U8(p) => [p[i][0] as f64 / 255.0, p[i][1] as f64 / 255.0, p[i][2] as f64 / 255.0],
        Pixels::U16(p) => [p[i][0] as f64 / 65535.0, p[i][1] as f64 / 65535.0, p[i][2] as f64 / 65535.0],
        Pixels::F32(p) => [p[i][0] as f64, p[i][1] as f64, p[i][2] as f64],
    }
}

/// Mean and maximum absolute colour error, as fractions of full scale, over the pixels at least `margin`
/// pixels away from every edge (content that enters from outside the picture cannot be reconstructed).
pub fn error(got: &Frame, want: &Frame, margin: usize) -> (f64, f64) {
    assert_eq!((got.width, got.height), (want.width, want.height));
    let (mut sum, mut max, mut n) = (0.0f64, 0.0f64, 0usize);
    for y in margin..got.height - margin {
        for x in margin..got.width - margin {
            let (g, w) = (norm(got, x, y), norm(want, x, y));
            for c in 0..3 {
                let e = (g[c] - w[c]).abs();
                sum += e;
                max = max.max(e);
                n += 1;
            }
        }
    }
    (sum / n as f64, max)
}

/// Like [`error`] over the pixels at least `margin` pixels from every edge for which `region(x, y)` is true; returns
/// `(mean, max, pixel count)`.
pub fn error_where(got: &Frame, want: &Frame, margin: usize, region: impl Fn(usize, usize) -> bool) -> (f64, f64, usize) {
    let (mut sum, mut max, mut n) = (0.0f64, 0.0f64, 0usize);
    for y in margin..got.height - margin {
        for x in margin..got.width - margin {
            if !region(x, y) {
                continue;
            }
            let (g, w) = (norm(got, x, y), norm(want, x, y));
            for c in 0..3 {
                let e = (g[c] - w[c]).abs();
                sum += e;
                max = max.max(e);
                n += 1;
            }
        }
    }
    (if n == 0 { 0.0 } else { sum / n as f64 }, max, n)
}

/// The mean absolute colour error over the pixels of `region` with no margin (the border strips are part of the region).
pub fn error_in(got: &Frame, want: &Frame, region: impl Fn(usize, usize) -> bool) -> f64 {
    error_where(got, want, 0, region).0
}

/// True when every pixel of the frame has full alpha (a wire frame: 255, 65535 or 1.0).
pub fn all_opaque(f: &Frame) -> bool {
    match &f.pixels {
        Pixels::U8(p) => p.iter().all(|px| px[3] == 255),
        Pixels::U16(p) => p.iter().all(|px| px[3] == 65535),
        Pixels::F32(p) => p.iter().all(|px| px[3] == 1.0),
    }
}

/// A scene with an OCCLUSION: a textured background that pans by `bg` pixels per frame and an opaque textured
/// rectangle that moves by `fg` pixels per frame over it (a different texture, so a wrong correspondence shows). Frame
/// `k` is the scene at `s = k`; the exact frame at any fractional `s` is known in closed form, including which pixels
/// show the rectangle. The rectangle covers `[x0 + fg.vx * s, + w) x [y0 + fg.vy * s, + h)`.
#[derive(Clone, Copy, Debug)]
pub struct Scene {
    pub bg: Motion,
    pub fg: Motion,
    pub x0: f64,
    pub y0: f64,
    pub w: f64,
    pub h: f64,
}

/// The scene of the Smart Blend tests: a background panning by (3.5, 1) and a 30 x 22 rectangle moving by (-7, 3).
pub const OCCLUSION: Scene = Scene { bg: Motion { vx: 8.0, vy: 2.0 }, fg: Motion { vx: -12.0, vy: 5.0 }, x0: 52.0, y0: 14.0, w: 30.0, h: 22.0 };

impl Scene {
    /// Does the rectangle cover pixel `(x, y)` at source position `s`?
    pub fn covers(self, x: usize, y: usize, s: f64) -> bool {
        let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
        let (rx, ry) = (self.x0 + self.fg.vx * s, self.y0 + self.fg.vy * s);
        px >= rx && px < rx + self.w && py >= ry && py < ry + self.h
    }

    /// The exact colour of pixel `(x, y)` at position `s`.
    pub fn rgb(self, x: usize, y: usize, s: f64) -> [f64; 3] {
        if self.covers(x, y, s) {
            let (px, py) = (x as f64 - self.fg.vx * s + 37.0, y as f64 - self.fg.vy * s + 11.0);
            [pattern(2, px, py), pattern(0, px * 1.3, py * 0.8), pattern(1, py, px)]
        } else {
            let (px, py) = (x as f64 - self.bg.vx * s, y as f64 - self.bg.vy * s);
            [pattern(0, px, py), pattern(1, px, py), pattern(2, px, py)]
        }
    }

    /// The wire frame (opaque) at position `s`, quantised to the depth.
    pub fn frame_at(self, depth: Depth, width: usize, height: usize, s: f64) -> Frame {
        let mut f = Frame::new(width, height, depth);
        for y in 0..height {
            for x in 0..width {
                let [r, g, b] = self.rgb(x, y, s);
                match &mut f.pixels {
                    Pixels::U8(px) => px[y * width + x] = [q8(r), q8(g), q8(b), 255],
                    Pixels::U16(px) => px[y * width + x] = [q16(r), q16(g), q16(b), 65535],
                    Pixels::F32(px) => px[y * width + x] = [r as f32, g as f32, b as f32, 1.0],
                }
            }
        }
        f
    }

    /// A clip of `count` frames, frame `k` at position `k`.
    pub fn clip(self, depth: Depth, width: usize, height: usize, count: usize) -> Vec<Frame> {
        (0..count).map(|k| self.frame_at(depth, width, height, k as f64)).collect()
    }
}
