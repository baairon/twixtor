//! The Vectors In decode of plugin 04: known vector images converted to flows and checked against
//! independent arithmetic, the alpha 0 holes filled, `Vec Scale` and `Max Displace` applied, and
//! the fetch times and parameter times of the two loaders.

mod common;

use common::*;
use fx_core::{float_to_fixed, Depth, Frame, Pixels};
use vectors_in::glue;
use vectors_in::timemap::{CurveParams04, VectorsInTimeMap};
use vectors_in::vectorin::{self, load_flow, load_flows, max_displace, present_at};
use retime_engine::flow::FlowField;
use retime_engine::frontend::fixed_reader;
use retime_engine::host::EngineSettings;
use retime_engine::timeline::Selection;

const W: usize = 4;
const H: usize = 3;

fn settings(w: usize, h: usize, depth: Depth) -> EngineSettings {
    let r = Rig::new(vec![Frame::new(w, h, depth)], &[]);
    glue::read_settings(&r.cx(0), &r.env()).unwrap().engine
}

fn flow_of(frame: Frame, scale: (f32, f32), max: f32) -> FlowField {
    let (w, h, d) = (frame.width, frame.height, frame.depth());
    let l = layer(vec![frame]);
    load_flow(&l, "test", 0, &settings(w, h, d), max, scale).unwrap()
}

/// What the decode of one 8 bit channel value must give: `((c * 2.0) / 254.0 - 1.0) * scale` in
/// double, narrowed once to f32.
fn dec8(c: u8, scale: f32) -> f32 {
    (((c as f64 * 2.0) / 254.0 - 1.0) * scale as f64) as f32
}

fn frame8(w: usize, h: usize, px: Vec<[u8; 4]>) -> Frame {
    f8(w, h, &px)
}

#[test]
fn an_8_bit_vector_image_decodes_with_max_displace() {
    // R = dx, G = dy, B unused, opaque
    let px: Vec<[u8; 4]> = vec![
        [127, 127, 9, 255], [254, 0, 9, 255], [0, 254, 9, 255], [200, 50, 9, 255],
        [128, 126, 0, 255], [1, 253, 0, 255], [127, 254, 0, 255], [254, 127, 0, 255],
        [64, 190, 0, 255], [10, 20, 0, 255], [250, 240, 0, 255], [100, 100, 255, 255],
    ];
    let flow = flow_of(frame8(W, H, px.clone()), (1.0, 1.0), 16.0);
    assert_eq!((flow.width, flow.height), (W, H));
    for (i, p) in px.iter().enumerate() {
        let got = flow.at(i % W, i / W);
        assert_eq!(got, [dec8(p[0], 16.0), dec8(p[1], 16.0)], "pixel {i}");
    }
    // the corners of the code: 127 is zero, 254 is +M, 0 is -M
    assert_eq!(flow.at(0, 0), [0.0, 0.0]);
    assert_eq!(flow.at(1, 0), [16.0, -16.0]);
    assert_eq!(flow.at(2, 0), [-16.0, 16.0]);
    // another Max Displace scales every vector
    let wide = flow_of(frame8(W, H, px.clone()), (1.0, 1.0), 127.0);
    assert_eq!(wide.at(3, 0), [dec8(200, 127.0), dec8(50, 127.0)]);
    assert_eq!(wide.at(3, 0)[0], 73.0); // 200 - 127: with Max Displace 127 the code is the integer c - 127
}

#[test]
fn a_float_vector_image_decodes_exactly() {
    let px: Vec<[f32; 4]> = (0..W * H)
        .map(|i| match i {
            0 => [0.5, 0.5, 0.0, 1.0],
            1 => [0.75, 0.25, 0.0, 1.0],
            2 => [1.0, 0.0, 0.0, 1.0],
            3 => [0.0, 1.0, 0.0, 1.0],
            _ => [0.5 + i as f32 / 64.0, 0.5 - i as f32 / 128.0, 0.0, 1.0],
        })
        .collect();
    let frame = Frame { width: W, height: H, pixels: Pixels::F32(px.clone()) };
    let flow = flow_of(frame, (1.0, 1.0), 10.0);
    for (i, p) in px.iter().enumerate() {
        // (R * 2 - 1) * M in single precision
        assert_eq!(flow.at(i % W, i / W), [(p[0] * 2.0 - 1.0) * 10.0, (p[1] * 2.0 - 1.0) * 10.0], "pixel {i}");
    }
    assert_eq!(flow.at(1, 0), [5.0, -5.0]);
}

#[test]
fn a_16_bit_vector_image_written_by_create_vectors_decodes_to_the_flow_it_encodes() {
    // the encode and the decode through the 16 bit range of the engine (0..32768 on the way):
    // within one level of that range
    let (vx, vy, m) = (3.25f32, -7.5f32, 16.0f32);
    let frame = vector_frame(Depth::U16, 8, 6, vx, vy, m);
    let flow = flow_of(frame, (1.0, 1.0), m);
    let tol = 2.0 * m / 16384.0;
    for y in 0..6 {
        for x in 0..8 {
            let [u, v] = flow.at(x, y);
            assert!((u - vx).abs() < tol && (v - vy).abs() < tol, "({x},{y}): {u} {v}");
        }
    }
}

#[test]
fn alpha_zero_pixels_are_holes_that_take_the_surrounding_flow() {
    let (w, h) = (8, 6);
    let base = [200u8, 60, 0, 255];
    let mut px = vec![base; w * h];
    // an isolated hole and a 2 x 2 block, with junk colours that must not leak into the flow
    for (x, y) in [(3, 2), (5, 3), (6, 3), (5, 4), (6, 4)] {
        px[y * w + x] = [7, 250, 99, 0];
    }
    let flow = flow_of(frame8(w, h, px.clone()), (1.0, 1.0), 16.0);
    let want = [dec8(200, 16.0), dec8(60, 16.0)];
    for y in 0..h {
        for x in 0..w {
            let got = flow.at(x, y);
            if px[y * w + x][3] == 255 {
                assert_eq!(got, want, "known pixel ({x},{y}) is returned unchanged");
            } else {
                assert_ne!(got, [0.0, 0.0], "hole ({x},{y}) is filled, not left at the zero vector");
                assert!((got[0] - want[0]).abs() < 1e-3 && (got[1] - want[1]).abs() < 1e-3, "hole ({x},{y}): {got:?} against {want:?}");
            }
        }
    }
}

#[test]
fn a_hole_between_two_flows_takes_a_value_between_them() {
    // left half pushes right (c = 254), right half pushes left (c = 0), one column of holes between
    let (w, h) = (9, 4);
    let mut px = vec![[0u8, 127, 0, 255]; w * h];
    for y in 0..h {
        for x in 0..4 {
            px[y * w + x] = [254, 127, 0, 255];
        }
        px[y * w + 4] = [127, 127, 0, 0];
    }
    let flow = flow_of(frame8(w, h, px), (1.0, 1.0), 10.0);
    for y in 0..h {
        let u = flow.at(4, y)[0];
        assert!((-10.0..=10.0).contains(&u), "{u}");
        assert!(flow.at(3, y)[0] == 10.0 && flow.at(5, y)[0] == -10.0);
        assert!(u.abs() < 5.0, "the hole sits between +10 and -10: {u}");
    }
}

#[test]
fn vec_scale_multiplies_after_the_fill() {
    let (w, h) = (6, 4);
    let mut px = vec![[200u8, 60, 0, 255]; w * h];
    px[2 * w + 2] = [0, 0, 0, 0];
    let plain = flow_of(frame8(w, h, px.clone()), (1.0, 1.0), 16.0);
    let scaled = flow_of(frame8(w, h, px), (2.0, -0.5), 16.0);
    for i in 0..w * h {
        let (x, y) = (i % w, i / w);
        let (a, b) = (plain.at(x, y), scaled.at(x, y));
        assert_eq!(b, [2.0 * a[0], -0.5 * a[1]], "pixel ({x},{y})");
    }
    // scale 0 gives the zero flow
    let zero = flow_of(frame8(4, 4, vec![[200, 60, 0, 255]; 16]), (0.0, 0.0), 16.0);
    assert!(zero.data.iter().all(|v| *v == 0.0));
}

#[test]
fn a_frame_of_the_wrong_size_or_depth_is_refused_and_no_frame_is_a_zero_flow() {
    let l = layer(vec![frame8(W, H, vec![[200, 60, 0, 255]; W * H])]);
    let s = settings(W + 1, H, Depth::U8);
    let e = load_flow(&l, "Vectors: Next", 0, &s, 16.0, (1.0, 1.0)).unwrap_err();
    assert!(e.to_string().contains("Vectors: Next has a 4x3 frame at time 0 but the output is 5x3"), "{e}");
    let s = settings(W, H, Depth::U16);
    let e = load_flow(&l, "Vectors: Prev", 0, &s, 16.0, (1.0, 1.0)).unwrap_err();
    assert!(e.to_string().contains("Vectors: Prev is 8 bpc but the output is 16 bpc"), "{e}");
    // time past the layer: transparent image, every pixel a hole, zero flow
    let s = settings(W, H, Depth::U8);
    let flow = load_flow(&l, "x", 100_000, &s, 16.0, (1.0, 1.0)).unwrap();
    assert!(flow.data.iter().all(|v| *v == 0.0));
}

#[test]
fn presence_is_the_checkout_at_the_given_time() {
    let l = layer(vec![frame8(W, H, vec![[0, 0, 0, 255]; W * H]); 2]);
    assert_eq!(present_at(Some(&l), 0).unwrap(), Some((W, H)));
    assert_eq!(present_at(Some(&l), 199).unwrap(), Some((W, H)));
    // outside its clip the layer keeps its size (transparent pixels): present
    assert_eq!(present_at(Some(&l), 200).unwrap(), Some((W, H)), "no frame at the time");
    assert_eq!(present_at(Some(&l), -1).unwrap(), Some((W, H)));
    assert_eq!(present_at(None, 0).unwrap(), None);
    let empty = layer(vec![Frame::new(0, 0, Depth::U8)]);
    assert_eq!(present_at(Some(&empty), 0).unwrap(), None);
}

/// A rig for the loaders: the vector layers hold, in frame `i`, the constant flow `(i, 2 i)` (Next) and `(-i, -2 i)`
/// (Prev), written with Max Displace 127 so that integer flows are exact (`c - 127`).
fn loader_rig(params: &[(&str, &str)], frames: usize) -> Rig {
    let (w, h) = (6, 4);
    let next: Vec<Frame> = (0..frames).map(|i| vector_frame(Depth::U8, w, h, i as f32, 2.0 * i as f32, 127.0)).collect();
    let prev: Vec<Frame> = (0..frames).map(|i| vector_frame(Depth::U8, w, h, -(i as f32), -2.0 * i as f32, 127.0)).collect();
    Rig::new(vec![Frame::new(w, h, Depth::U8)], params).with_layer("vectors_next", next).with_layer("vectors_prev", prev)
}

fn selection(a: i32) -> Selection {
    Selection { s0: a as f32 + 0.5, s1: a as f32 + 1.0, a, b: a + 1, wa: 0.5, exact: false, vectors_quality: 4, interp: 1 }
}

fn loaders(r: &Rig, a: i32, has_prev: bool, has_next: bool) -> retime_engine::compose::Flows {
    let cx = r.cx(0);
    let env = r.env();
    let s = glue::read_settings(&cx, &env).unwrap().engine;
    let map = VectorsInTimeMap::new(CurveParams04::new(env.params).unwrap(), s.rates, s.current_time);
    load_flows(&env, &s, &map, &selection(a), has_prev, has_next).unwrap()
}

#[test]
fn next_is_read_at_frame_a_and_prev_at_frame_b() {
    let r = loader_rig(&[("vectors_max_displace", "127")], 12);
    // dur = 100: T(k) = 100 k, frame index k
    let f = loaders(&r, 3, true, true);
    assert_eq!(f.f_ab.as_ref().unwrap().at(0, 0), [3.0, 6.0], "F_AB comes from Vectors: Next at frame A = 3");
    assert_eq!(f.f_ba.as_ref().unwrap().at(5, 3), [-4.0, -8.0], "F_BA comes from Vectors: Prev at frame B = 4");
    // 24 in 30: dur = 125, T(4) = 500 (frame 5), T(5) = 625 (frame 6)
    let r = loader_rig(&[("vectors_max_displace", "127"), ("in_fps_is_out_fps", "0"), ("input_frame_rate", "24")], 12);
    let f = loaders(&r, 4, true, true);
    assert_eq!(f.f_ab.as_ref().unwrap().at(1, 1), [5.0, 10.0]);
    assert_eq!(f.f_ba.as_ref().unwrap().at(1, 1), [-6.0, -12.0]);
}

#[test]
fn an_absent_layer_gives_no_flow() {
    let r = loader_rig(&[("vectors_max_displace", "127")], 12);
    let f = loaders(&r, 3, true, false);
    assert!(f.f_ab.is_none() && f.f_ba.is_some());
    let f = loaders(&r, 3, false, true);
    assert!(f.f_ab.is_some() && f.f_ba.is_none());
    let f = loaders(&r, 3, false, false);
    assert!(f.f_ab.is_none() && f.f_ba.is_none());
}

#[test]
fn vec_scale_is_read_at_the_fetch_time_of_each_loader() {
    // Vec Scale X runs from 1 at frame 0 to 3 at frame 10 (ticks 0 and 1000): at T(4) = 500 it is 2.0, at T(5) = 625
    // (24 in 30) 2.25; Vec Scale Y is the constant 0.5
    let r = loader_rig(
        &[("vectors_max_displace", "127"), ("in_fps_is_out_fps", "0"), ("input_frame_rate", "24"), ("vec_scale_x", "1@0;3@10"), ("vec_scale_y", "0.5")],
        12,
    );
    let f = loaders(&r, 4, true, true);
    assert_eq!(f.f_ab.as_ref().unwrap().at(0, 0), [5.0 * 2.0, 10.0 * 0.5]);
    assert_eq!(f.f_ba.as_ref().unwrap().at(0, 0), [-6.0 * 2.25, -12.0 * 0.5]);
}

#[test]
fn max_displace_is_the_16_16_value_at_the_given_time() {
    let c = curves(&[("vectors_max_displace", "16@0;32@10")]);
    assert_eq!(max_displace(&c, 500), 24.0);
    let at_100 = max_displace(&c, 100);
    assert_eq!(at_100, fixed_reader(float_to_fixed(17.6), 1.0));
    assert_ne!(at_100, 17.6f32);
    // the default is 2048
    assert_eq!(max_displace(&curves(&[]), 0), 2048.0);
    // Vec Scale: the 16.16 reader without a multiplier, default 1.0
    assert_eq!(vectorin::vec_scale(&curves(&[]), 0), (1.0, 1.0));
    assert_eq!(vectorin::vec_scale(&curves(&[("vec_scale_x", "-2.5"), ("vec_scale_y", "0.3")]), 0), (-2.5, fixed_reader(float_to_fixed(0.3), 1.0)));
}

#[test]
fn each_layered_load_is_stepped_by_the_matte_of_its_own_frame() {
    // the per layer matte step: F_AB is loaded from Vectors: Next at T(A) and stepped by the layer's matte AT A, F_BA from
    // Vectors: Prev at T(B) by its matte AT B. The vector layers hold one flow on the left half and another on the
    // right half; the layer's matte covers the left half at A and the right half at B. F_AB keeps the left half's
    // vectors exactly and its holes (the right half) take them from the fill; F_BA the other way round
    let (w, h) = (8usize, 4usize);
    let split = |left: [f32; 2], right: [f32; 2]| {
        let mut flow = FlowField::new(w, h);
        for y in 0..h {
            for x in 0..w {
                flow.set(x, y, if x < w / 2 { left } else { right });
            }
        }
        retime_engine::vectors::encode_vectors_image(w, h, retime_engine::image::Fmt::U8, &flow, 127.0).unwrap().to_frame()
    };
    let next: Vec<Frame> = (0..12).map(|_| split([3.0, 1.0], [-2.0, 0.0])).collect();
    let prev: Vec<Frame> = (0..12).map(|_| split([-3.0, -1.0], [2.0, -1.0])).collect();
    let r = Rig::new(vec![Frame::new(w, h, Depth::U8)], &[("vectors_max_displace", "127")]).with_layer("vectors_next", next).with_layer("vectors_prev", prev);
    let half = |left: bool| retime_engine::matte::Matte { width: w, height: h, data: (0..w * h).map(|i| if (i % w < w / 2) == left { 1.0 } else { 0.0 }).collect() };
    let (at_a, at_b) = (half(true), half(false));
    let cx = r.cx(0);
    let env = r.env();
    let s = glue::read_settings(&cx, &env).unwrap().engine;
    let map = VectorsInTimeMap::new(CurveParams04::new(env.params).unwrap(), s.rates, s.current_time);
    let mattes = retime_engine::layers::LayerMattes { at_a: &at_a, at_b: &at_b };
    let load = |threads: usize| {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
            let map = VectorsInTimeMap::new(CurveParams04::new(env.params).unwrap(), s.rates, s.current_time);
            vectorin::load_layer_flows(&env, &s, &map, &selection(3), true, true, &mattes).unwrap()
        })
    };
    let f = load(1);
    assert_eq!(load(4), f, "1 and 4 threads");
    let (ab, ba) = (f.f_ab.unwrap(), f.f_ba.unwrap());
    let near = |v: [f32; 2], want: [f32; 2]| (v[0] - want[0]).abs() < 1e-3 && (v[1] - want[1]).abs() < 1e-3;
    for y in 0..h {
        for x in 0..w {
            let left = x < w / 2;
            let (vab, vba) = (ab.at(x, y), ba.at(x, y));
            if left {
                assert_eq!(vab, [3.0, 1.0], "F_AB ({x}, {y}): inside the matte at A, as loaded");
                assert!(near(vba, [2.0, -1.0]), "F_BA ({x}, {y}): a hole of the matte at B, filled: {vba:?}");
            } else {
                assert!(near(vab, [3.0, 1.0]), "F_AB ({x}, {y}): a hole of the matte at A, filled: {vab:?}");
                assert_eq!(vba, [2.0, -1.0], "F_BA ({x}, {y}): inside the matte at B, as loaded");
            }
        }
    }
    // the same mattes the other way round give the other halves
    let swapped = retime_engine::layers::LayerMattes { at_a: &at_b, at_b: &at_a };
    let g = vectorin::load_layer_flows(&env, &s, &map, &selection(3), true, true, &swapped).unwrap();
    assert!(near(g.f_ab.unwrap().at(0, 0), [-2.0, 0.0]) && near(g.f_ba.unwrap().at(w - 1, 0), [-3.0, -1.0]));
}
