//! Seeded solver cases shared by the device-free and the device tests: frame pairs of smooth
//! textures (translated, rotated, still), mattes, tracking points, adversarial pixels, the CPU
//! reference of a case and its per level trace.

use crate::device::sim::Rng;
use crate::flow::extras::{self, MaskPyramid, SolveExtras, UserDisplacement};
use crate::flow::{self, FlowField, FlowSettings, Pyramid};
use crate::image::{Image, Samples};
use crate::matte::Matte;

/// One solve: the frames, the settings, the optional mattes and point pairs.
#[derive(Clone)]
pub struct Case {
    pub name: String,
    pub a: Image,
    pub b: Image,
    pub s: FlowSettings,
    pub mattes: Option<(Matte, Matte)>,
    pub points: Vec<([f32; 2], [f32; 2])>,
}

/// The texture value at `(u, v)`: a sum of seeded plane waves in 0..1.
fn wave(waves: &[(f64, f64, f64, f64)], u: f64, v: f64) -> f64 {
    let mut s = 0.5;
    for &(a, ku, kv, p) in waves {
        s += a * (ku * u + kv * v + p).sin();
    }
    s
}

/// A frame of a smooth texture moved by `(dx, dy)` and rotated by `angle` radians about the
/// centre; `depth` 8, 16 or 32 (float); `period` makes the texture periodic in x over the width
/// (wavenumbers multiple of `2 pi / w`, for the wrap cases).
#[allow(clippy::too_many_arguments)]
pub fn texture(w: usize, h: usize, depth: usize, seed: u64, dx: f64, dy: f64, angle: f64, period: bool) -> Image {
    let mut rng = Rng::new(seed);
    let mut unit = || (rng.next_u32() as f64) / (u32::MAX as f64);
    let waves: Vec<(f64, f64, f64, f64)> = (0..6)
        .map(|k| {
            let a = 0.08 + 0.04 * unit();
            let (ku, kv) = if period {
                let n = 1.0 + (k % 3) as f64 + (unit() * 2.0).floor();
                (std::f64::consts::TAU * n / w as f64, (unit() - 0.5) * 0.9)
            } else {
                ((unit() - 0.5) * 1.2, (unit() - 0.5) * 1.2)
            };
            (a, ku, kv, unit() * 6.0)
        })
        .collect();
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let (sn, cs) = angle.sin_cos();
    let n = w * h * 4;
    let mut vals = Vec::with_capacity(n);
    for y in 0..h {
        for x in 0..w {
            let (px, py) = (x as f64 - cx, y as f64 - cy);
            let u = cs * px - sn * py + cx - dx;
            let v = sn * px + cs * py + cy - dy;
            let t = wave(&waves, u, v);
            vals.push((1.0, t, wave(&waves, u + 3.0, v), wave(&waves, u, v + 5.0)));
        }
    }
    let samples = match depth {
        8 => Samples::U8(vals.iter().flat_map(|&(a, r, g, b)| [a, r, g, b].map(|c: f64| (c.clamp(0.0, 1.0) * 255.0).round() as u8)).collect()),
        16 => Samples::U16(vals.iter().flat_map(|&(a, r, g, b)| [a, r, g, b].map(|c: f64| (c.clamp(0.0, 1.0) * 32768.0).round() as u16)).collect()),
        // float frames in 0..255 for odd seeds (the update test |t| > 40 needs gradients of that
        // size to move anything), in 0..1 otherwise
        _ => {
            let k = if seed % 2 == 1 { 255.0 } else { 1.0 };
            Samples::F32(vals.iter().flat_map(|&(a, r, g, b)| [a, r, g, b].map(|c: f64| (c * k) as f32)).collect())
        }
    };
    Image { width: w, height: h, samples }
}

/// Puts `v` into the R, G and B samples of `count` seeded pixels of a float image.
pub fn spike(img: &mut Image, rng: &mut Rng, count: usize, v: f32) {
    let n = img.width * img.height;
    if let Samples::F32(s) = &mut img.samples {
        for _ in 0..count {
            let p = rng.below(n as u64) as usize;
            for c in 1..4 {
                s[p * 4 + c] = v;
            }
        }
    }
}

/// A soft disc matte.
pub fn disc(w: usize, h: usize, cx: f32, cy: f32, r: f32) -> Matte {
    Matte { width: w, height: h, data: crate::device::convstage::corpus::disc(w, h, cx, cy, r) }
}

/// Everything the CPU computes for a case: the pyramids, the extras and the flow.
pub struct Reference {
    pub px: Pyramid,
    pub py: Pyramid,
    pub masks: Option<(MaskPyramid, MaskPyramid)>,
    pub user: Option<UserDisplacement>,
    pub flow: FlowField,
}

impl Reference {
    pub fn extras(&self) -> SolveExtras<'_> {
        SolveExtras { masks: self.masks.as_ref().map(|(a, b)| (a, b)), user: self.user.as_ref() }
    }
}

/// The CPU solve of a case (`None` when the CPU refuses it).
pub fn reference(c: &Case) -> Option<Reference> {
    let px = flow::build_pyramid(&c.a, &c.s).ok()?;
    let py = flow::build_pyramid(&c.b, &c.s).ok()?;
    let masks = match &c.mattes {
        Some((ma, mb)) => Some((MaskPyramid::build(ma, &c.s).ok()?, MaskPyramid::build(mb, &c.s).ok()?)),
        None => None,
    };
    let user = if c.points.is_empty() { None } else { Some(crate::points::user_displacement(&c.points, px.geometry(), (c.a.width, c.a.height))) };
    let e = SolveExtras { masks: masks.as_ref().map(|(a, b)| (a, b)), user: user.as_ref() };
    let flow = extras::solve_pyramids_with(&px, &py, &c.s, e).ok()?;
    Some(Reference { px, py, masks, user, flow })
}

/// The CPU driver of `solve_pyramids_with` level by level: the flow of every visited level after
/// its range check and damping, coarsest first (the steps of `solve_extended`).
pub fn cpu_trace(r: &Reference, s: &FlowSettings) -> Vec<(usize, Vec<f32>)> {
    let geo = r.px.geometry();
    let (start, damp) = flow::start_level(s.sensitivity, geo.levels);
    let (sw0, sh0) = geo.sizes[start];
    let mut f = vec![0.0f32; sw0 * sh0 * 2];
    let mut check = true;
    let mut out = Vec::new();
    let user = r.user.as_ref().filter(|u| !u.levels.is_empty());
    for l in (0..=start).rev() {
        let (w, h) = geo.sizes[l];
        let mx = r.masks.as_ref().map(|(m, _)| m.level(l));
        let my = r.masks.as_ref().map(|(_, m)| m.level(l));
        extras::level_step(&r.px.level(l).data, &r.py.level(l).data, w, h, mx, my, user.map(|u| &u.levels[l]), s, l == 0, &mut f);
        if check {
            let all_out = if s.wrap { flow::all_flow_out_of_range_y(&f, w, h) } else { flow::all_flow_out_of_range(&f, w, h) };
            if all_out {
                f.iter_mut().for_each(|v| *v = 0.0);
            } else {
                check = false;
            }
        }
        if l == start && (damp as f64 - 1.0).abs() >= 1e-6 {
            flow::scale_values(&mut f, damp);
        }
        out.push((l, f.clone()));
        if l > 0 {
            let (nw, nh) = geo.sizes[l - 1];
            f = if s.wrap { flow::upsample_flow_wrapx(&f, w, h, nw, nh) } else { flow::upsample_flow(&f, w, h, nw, nh) };
        }
    }
    out
}

/// Equal bits, or both NaN (the devices write other NaN payloads).
pub fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// The first value where two traces differ: `(level, pixel x, pixel y, channel, cpu, device)`.
pub fn first_trace_difference(cpu: &[(usize, Vec<f32>)], dev: &[(usize, Vec<f32>)], sizes: &[(usize, usize)]) -> Option<String> {
    for ((lc, vc), (ld, vd)) in cpu.iter().zip(dev.iter()) {
        if lc != ld || vc.len() != vd.len() {
            return Some(format!("level {lc} against {ld}: {} against {} values", vc.len(), vd.len()));
        }
        if let Some(i) = (0..vc.len()).find(|&i| !same(vc[i], vd[i])) {
            let w = sizes[*lc].0;
            let p = i / 2;
            return Some(format!(
                "first difference at level {lc}, pixel ({}, {}), channel {}: cpu {:e} ({:#010x}), device {:e} ({:#010x})",
                p % w,
                p / w,
                i % 2,
                vc[i],
                vc[i].to_bits(),
                vd[i],
                vd[i].to_bits()
            ));
        }
    }
    if cpu.len() != dev.len() {
        return Some(format!("{} levels against {}", cpu.len(), dev.len()));
    }
    None
}

/// Settings of a case: `(quality, hiq)` and the sensitivity.
pub fn settings(q: (u8, bool), sens: f32, wrap: bool) -> FlowSettings {
    let mut s = FlowSettings::new(q.0, q.1, sens);
    s.wrap = wrap;
    s
}

/// The qualities of the corpus: Best, High, draft (Best at draft), full resolution, Sloppy.
pub const QUALITIES: [(u8, bool); 5] = [(4, true), (3, true), (4, false), (5, true), (1, true)];

/// A seeded corpus of `count` cases with sides up to `max_side` (every class of the brief).
pub fn corpus(seed: u64, count: usize, max_side: usize) -> Vec<Case> {
    let mut rng = Rng::new(seed);
    let sizes: Vec<(usize, usize)> = [
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 3),
        (3, 7),
        (8, 8),
        (16, 9),
        (17, 13),
        (24, 18),
        (32, 24),
        (33, 17),
        (48, 32),
        (64, 48),
        (96, 72),
        (128, 96),
        (160, 120),
        (320, 180),
        (640, 360),
    ]
    .into_iter()
    .filter(|&(w, h)| w.max(h) <= max_side)
    .collect();
    let mut out = Vec::with_capacity(count);
    for k in 0..count {
        let (w, h) = sizes[k % sizes.len()];
        let depth = [8, 16, 32][rng.below(3) as usize];
        let q = QUALITIES[rng.below(QUALITIES.len() as u64) as usize];
        let sens = [0.7f32, 1.0, 1.0, 0.85, 0.3, 0.95, 1.0, 0.9][rng.below(8) as usize];
        let class = k % 10;
        let wrap = class == 6 || class == 7 || rng.below(8) == 0;
        let tseed = rng.next_u64();
        let unit = |r: &mut Rng| (r.next_u32() as f64) / (u32::MAX as f64) - 0.5;
        let (mut dx, mut dy, mut ang) = (unit(&mut rng) * 3.0, unit(&mut rng) * 2.0, 0.0);
        let mut name = String::new();
        match class {
            0 => name.push_str("translate"),
            1 => {
                ang = unit(&mut rng) * 0.15;
                name.push_str("rotate");
            }
            2 => {
                dx = 0.0;
                dy = 0.0;
                name.push_str("still");
            }
            3 => {
                dx = (w as f64) * 0.6 + 7.0;
                dy = (h as f64) * 0.5 + 3.0;
                name.push_str("large");
            }
            _ => name.push_str("translate"),
        }
        let a = texture(w, h, depth, tseed, 0.0, 0.0, 0.0, wrap);
        let mut b = texture(w, h, depth, tseed, dx, dy, ang, wrap);
        let mut a = a;
        if class == 8 && depth == 32 {
            let v = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0e30][rng.below(4) as usize];
            spike(&mut a, &mut rng, 1 + (w * h) / 50, v);
            spike(&mut b, &mut rng, 1, v);
            name.push_str(&format!("+spike({v:e})"));
        }
        let mattes = if class == 4 || class == 7 || class == 9 {
            let r = (w.min(h) as f32) * 0.35 + 1.0;
            let (cx, cy) = (w as f32 * 0.45, h as f32 * 0.5);
            name.push_str("+mask");
            Some((disc(w, h, cx, cy, r), disc(w, h, cx + dx as f32, cy + dy as f32, r)))
        } else {
            None
        };
        let points = if class == 5 || class == 9 {
            name.push_str("+points");
            let n = 1 + rng.below(3) as usize;
            (0..n)
                .map(|_| {
                    let fx = (rng.next_u32() as f32 / u32::MAX as f32) * w as f32;
                    let fy = (rng.next_u32() as f32 / u32::MAX as f32) * h as f32;
                    ([fx, fy], [fx + dx as f32, fy + dy as f32])
                })
                .collect()
        } else {
            Vec::new()
        };
        let mut points = points;
        if class == 5 && (k / 10) % 3 == 2 {
            // an infinite target: the user displacement turns the flow infinite, then NaN
            points[0].1[0] = f32::INFINITY;
            name.push_str("+inf point");
        }
        if wrap {
            name.push_str("+wrap");
        }
        let s = settings(q, sens, wrap);
        out.push(Case { name: format!("#{k} {name} {w}x{h} d{depth} q{}{} s{sens}", q.0, if q.1 { "" } else { "draft" }), a, b, s, mattes, points });
    }
    out
}

/// A user displacement that sets the vector of one pixel per level to `(v, v)` (weight 1): with a
/// NaN `v` the CPU flow holds a NaN, the case of the `NanOutput` rule.
pub fn spot_user(geo: &crate::flow::FlowGeometry, v: f32) -> UserDisplacement {
    let levels = geo.sizes[..geo.levels]
        .iter()
        .map(|&(w, h)| {
            let mut d = vec![0.0f32; w * h * 2];
            let mut wt = vec![0.0f32; w * h];
            let p = (h / 2) * w + w / 2;
            d[p * 2] = v;
            d[p * 2 + 1] = v;
            wt[p] = 1.0;
            extras::UserLevel { d, w: wt }
        })
        .collect();
    UserDisplacement { levels }
}
