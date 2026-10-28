//! The n-layer path of plugin 04 with exact vector images: the per layer matte step (rule G1e) gives
//! each layer the motion of its own region everywhere, and the layered render of a two layer scene
//! is the exact in-between frame on the FG core and on the BG far from the disc.

mod common;

use common::*;
use fx_core::{Depth, Frame, Pixels};
use retime_engine::compose::flow_from_vector_image;
use retime_engine::flow::FlowField;
use retime_engine::image::{Fmt, Image};
use retime_engine::matte::Matte;
use retime_engine::vectors::encode_vectors_image;
use vectors_in::vectorin::matte_step;

const W: usize = 96;
const H: usize = 64;
const MD: f32 = 16.0;
/// The disc of the FG layer: centre `(40 + 3 k, 32)` at frame `k`, radius 18.
const R: f64 = 18.0;

fn centre(s: f64) -> (f64, f64) {
    (40.0 + 3.0 * s, 32.0)
}

fn inside(x: f64, y: f64, s: f64) -> bool {
    let (cx, cy) = centre(s);
    (x - cx) * (x - cx) + (y - cy) * (y - cy) <= R * R
}

fn wave(x: f64, y: f64, seed: f64) -> [f64; 3] {
    std::array::from_fn(|c| 0.5 + 0.22 * (x * 0.31 + seed + c as f64).sin() + 0.2 * (y * 0.27 + 2.0 * seed + 1.7 * c as f64).cos())
}

/// The scene at source position `s`: the FG texture moving (+3, 0) per frame inside the disc, the BG (-2, +1) outside.
fn colour(x: usize, y: usize, s: f64) -> [f64; 3] {
    let (xf, yf) = (x as f64, y as f64);
    if inside(xf, yf, s) {
        wave(xf - 3.0 * s, yf, 0.0)
    } else {
        wave(xf + 2.0 * s, yf - s, 1.1)
    }
}

fn frame_of(f: impl Fn(usize, usize) -> [f64; 3]) -> Frame {
    let mut fr = Frame::new(W, H, Depth::U8);
    if let Pixels::U8(px) = &mut fr.pixels {
        for y in 0..H {
            for x in 0..W {
                let c = f(x, y);
                px[y * W + x] = [(c[0] * 255.0 + 0.5) as u8, (c[1] * 255.0 + 0.5) as u8, (c[2] * 255.0 + 0.5) as u8, 255];
            }
        }
    }
    fr
}

/// The exact vector image at frame `k`: `Vectors: Next` (`next`, the motion from `k` to `k + 1`) or `Vectors: Prev`
/// (the motion from `k` back to `k - 1`), FG or BG by the region at frame `k`.
fn vectors(k: usize, next: bool) -> Frame {
    let sign = if next { 1.0f32 } else { -1.0 };
    let mut flow = FlowField::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let v = if inside(x as f64, y as f64, k as f64) { [3.0, 0.0] } else { [-2.0, 1.0] };
            flow.set(x, y, [sign * v[0], sign * v[1]]);
        }
    }
    encode_vectors_image(W, H, Fmt::U8, &flow, MD).unwrap().to_frame()
}

fn disc(k: usize) -> Frame {
    frame_of(|x, y| if inside(x as f64, y as f64, k as f64) { [1.0; 3] } else { [0.0; 3] })
}

#[test]
fn the_matte_step_gives_each_layer_the_motion_of_its_own_region_everywhere() {
    let img = Image::from_frame(&vectors(2, true));
    let fg = Matte { width: W, height: H, data: (0..W * H).map(|i| if inside((i % W) as f64, (i / W) as f64, 2.0) { 1.0 } else { 0.0 }).collect() };
    let bg = Matte { width: W, height: H, data: fg.data.iter().map(|v| 1.0 - v).collect() };
    for (m, want) in [(&fg, [3.0f32, 0.0]), (&bg, [-2.0, 1.0])] {
        let mut stepped = img.clone();
        matte_step(&mut stepped, m).unwrap();
        let flow = flow_from_vector_image(&stepped, MD, (1.0, 1.0)).unwrap();
        // the 8 bpc coding quantises a vector to MD / 127 = 0.126 px; the hole fill spreads the region's constant
        for y in 0..H {
            for x in 0..W {
                let f = flow.at(x, y);
                assert!((f[0] - want[0]).abs() < 0.07 && (f[1] - want[1]).abs() < 0.07, "({x}, {y}): {f:?} against {want:?}");
            }
        }
    }
    // a matte of another size is an error, not a panic
    assert!(matte_step(&mut img.clone(), &Matte::filled(W - 1, H, 1.0)).is_err());
}

fn error_on(got: &Frame, s: f64, keep: impl Fn(f64, f64) -> bool) -> f64 {
    let (mut sum, mut n) = (0.0, 0usize);
    if let Pixels::U8(px) = &got.pixels {
        for y in 0..H {
            for x in 0..W {
                if !keep(x as f64, y as f64) {
                    continue;
                }
                let want = colour(x, y, s);
                for c in 0..3 {
                    sum += (px[y * W + x][c] as f64 / 255.0 - want[c]).abs();
                    n += 1;
                }
            }
        }
    }
    sum / n as f64
}

#[test]
fn exact_vector_images_render_the_exact_in_between_frame_on_both_layers() {
    let clip: Vec<Frame> = (0..8).map(|k| frame_of(|x, y| colour(x, y, k as f64))).collect();
    let rig = |p: &[(&str, &str)]| {
        Rig::new(clip.clone(), p)
            .with_layer("vectors_next", (0..10).map(|k| vectors(k, true)).collect())
            .with_layer("vectors_prev", (0..10).map(|k| vectors(k, false)).collect())
            .with_layer("fg1_matte", (0..10).map(disc).collect())
    };
    let dist = |x: f64, y: f64, s: f64| {
        let (cx, cy) = centre(s);
        ((x - cx) * (x - cx) + (y - cy) * (y - cy)).sqrt()
    };
    for warping in ["1", "2"] {
        let r = rig(&[("vectors_max_displace", "16"), ("speed_percent", "50"), ("warping", warping)]);
        // output frame 3 is position 1.5
        let got = r.render(3).unwrap();
        let core = error_on(&got, 1.5, |x, y| dist(x, y, 1.5) < R - 3.0);
        let far = error_on(&got, 1.5, |x, y| dist(x, y, 1.5) > R + 4.0 && x >= 4.0 && y >= 4.0 && x < 92.0 && y < 60.0);
        // measured 0.0021 and 0.0016 of white for both warpings (the sampling floor); a frame blend is about 0.1
        assert!(core < 0.004 && far < 0.004, "warping {warping}: FG core {core:.5}, BG far {far:.5}");
    }
}
