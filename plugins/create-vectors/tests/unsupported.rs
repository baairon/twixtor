//! What the plugin once refused now renders, through the `Effect` trait exactly as a host runs it:
//! FG matte layers (one flow per layer, the matte weighted merge), `Layers Output`, tracking
//! points (user displacement), `Main_BG: Image Track` and `FGk: Image Track` unchecked (the second tracker),
//! Display Debug (five views) and the Draw Geom overlay on the Source and Debug displays. Each
//! render equals the engine call it stands for, made on the frames upside down with the result turned back (the
//! plugin works bottom row first, `create_vectors::glue`). An error fill (size mismatch, missing layer) still
//! wins over everything, as the flags are tested first.

mod common;

use common::*;
use create_vectors::glue::{flip_image, flip_pos};
use fx_core::{Depth, Frame, Pixels};
use retime_engine::create_vectors::{create_vectors, create_vectors_job, frame_time, job_flow, CreateVectorsJob, LayerParams};
use retime_engine::debug::debug_view_on;
use retime_engine::stages::{GouraudForward, StageSet};
use retime_engine::image::{Fmt, Image};
use retime_engine::matte::extract;
use retime_engine::points::UserPoint;
use retime_engine::vectors::zero_vectors_image;

const FG1: &str = "fg1_matte";
const FG2: &str = "fg2_matte";
const W: usize = 48;
const H: usize = 36;

fn clip() -> Vec<Frame> {
    moving::DEFAULT.clip(Depth::U8, W, H, 6)
}

/// A matte clip: an opaque white disc of radius `r` (in every channel) on transparent black, the
/// disc moving by `(2, 1)` px a frame.
fn disc_clip(cx: f64, cy: f64, r: f64) -> Vec<Frame> {
    (0..6)
        .map(|k| {
            let mut f = Frame::new(W, H, Depth::U8);
            if let Pixels::U8(p) = &mut f.pixels {
                for (i, px) in p.iter_mut().enumerate() {
                    let (x, y) = ((i % W) as f64, (i / W) as f64);
                    let d = ((x - cx - 2.0 * k as f64).powi(2) + (y - cy - k as f64).powi(2)).sqrt();
                    *px = if d <= r { [255, 255, 255, 255] } else { [0, 0, 0, 0] };
                }
            }
            f
        })
        .collect()
}

fn rig(params: &[(&str, &str)]) -> Rig {
    Rig::new(clip(), params).with_frames(6)
}

/// The engine image of a frame, upside down (the row order of the plugin).
fn engine(f: &Frame) -> Image {
    let mut img = Image::from_frame(f);
    flip_image(&mut img);
    img
}

/// The stages of the Debug views: the CPU with the Gouraud forward warp (`create_vectors::glue`, Debug warp).
fn gouraud() -> StageSet {
    GouraudForward::wrap(StageSet::cpu())
}

/// An engine result turned back to the rows of the frame, as the wire frame the effect writes.
fn back(mut img: Image) -> Frame {
    flip_image(&mut img);
    img.to_frame()
}

/// The settings of the default parameters with `max_displace` 16 (the rigs below set it).
fn base() -> retime_engine::create_vectors::CreateVectorsSettings {
    retime_engine::create_vectors::CreateVectorsSettings {
        quality: 4,
        hiq: true,
        sensitivity: 0.7,
        prep_contrast_edge: false,
        prep_delinearize: false,
        track_rgb_a: false,
        max_displace: 16.0,
    }
}

fn layer(track: bool) -> LayerParams {
    LayerParams { matte_chan: 1, sensitivity: 0.7, shrink: 0.0, track }
}

/// Output frame 2 with Previous: X is frame 2, Y frame 1.
fn pair() -> (Image, Image) {
    let c = clip();
    (engine(&c[2]), engine(&c[1]))
}

#[test]
fn fg_matte_layers_render_the_merged_vectors() {
    let m1 = disc_clip(18.0, 16.0, 11.0);
    let m2 = disc_clip(30.0, 18.0, 9.0);
    let (x, y) = pair();
    let mat = |c: &[Frame], k: usize| extract(&engine(&c[k]), 1);
    for (lo, first, last) in [("1", 0usize, 2usize), ("2", 0, 0), ("3", 1, 1), ("4", 2, 2), ("5", 2, 2)] {
        let r = rig(&[("max_displace", "16"), ("layers_output", lo)]).with_layer(FG1, m1.clone()).with_layer(FG2, m2.clone());
        let got = r.render(2).unwrap();
        let job = CreateVectorsJob {
            settings: base(),
            layers: vec![layer(true), layer(true), layer(true)],
            mattes_x: vec![mat(&m1, 2), mat(&m2, 2)],
            mattes_y: vec![mat(&m1, 1), mat(&m2, 1)],
            points: Vec::new(),
            first,
            last,
        };
        assert_eq!(got, back(create_vectors_job(&x, &y, &job).unwrap()), "layers output {lo}");
    }
    // FG2 alone is not active (the scan stops at the first empty matte): the single layer output
    let plain = rig(&[("max_displace", "16")]).render(2).unwrap();
    assert_eq!(rig(&[("max_displace", "16")]).with_layer(FG2, m2).render(2).unwrap(), plain);
    assert_eq!(plain, back(create_vectors(&x, &y, &base()).unwrap()));
    // a black FG1 matte with Layers Output All: Main_BG takes the whole frame, the single layer output
    let black: Vec<Frame> = (0..6).map(|_| Frame::new(W, H, Depth::U8)).collect();
    assert_eq!(rig(&[("max_displace", "16")]).with_layer(FG1, black).render(2).unwrap(), plain);
}

#[test]
fn the_fg_controls_reach_their_layers() {
    let m1 = disc_clip(18.0, 16.0, 11.0);
    let (x, y) = pair();
    let mat = |k: usize, chan: i32| extract(&engine(&m1[k]), chan);
    let params = [
        ("max_displace", "16"),
        ("fg1_sensitivity", "50"),
        ("fg1_matte_chan", "4"),
        ("fg1_inv_matte_shrink", "2.5"),
        ("fg1_track", "0"),
        ("main_bg_sensitivity", "90"),
    ];
    let got = rig(&params).with_layer(FG1, m1.clone()).render(2).unwrap();
    let job = CreateVectorsJob {
        settings: CreateVectorsSettingsExt::with_sensitivity(base(), 0.9),
        layers: vec![
            LayerParams { matte_chan: 4, sensitivity: 0.5, shrink: 2.5, track: false },
            LayerParams { matte_chan: 0, sensitivity: 0.9, shrink: 0.0, track: true },
        ],
        mattes_x: vec![mat(2, 4)],
        mattes_y: vec![mat(1, 4)],
        points: Vec::new(),
        first: 0,
        last: 1,
    };
    assert_eq!(got, back(create_vectors_job(&x, &y, &job).unwrap()));
}

/// `CreateVectorsSettings` with another Main_BG sensitivity.
trait CreateVectorsSettingsExt {
    fn with_sensitivity(self, s: f32) -> Self;
}

impl CreateVectorsSettingsExt for retime_engine::create_vectors::CreateVectorsSettings {
    fn with_sensitivity(mut self, s: f32) -> Self {
        self.sensitivity = s;
        self
    }
}

#[test]
fn the_source_display_ignores_fg_mattes_without_draw_geom() {
    // Display Source copies the source whatever the mattes (the layer count only matters for the overlay)
    let with = rig(&[("display", "1")]).with_layer(FG1, clip());
    let without = rig(&[("display", "1")]);
    assert_eq!(with.render(2).unwrap(), without.render(2).unwrap());
}

#[test]
fn every_debug_view_renders_the_engine_view() {
    let (x, y) = pair();
    for (view, code) in [("1", 4), ("2", 0), ("3", 1), ("4", 2), ("5", 3)] {
        // Debug swaps the roles: Previous reads X = frame 3, Y = frame 2 for output frame 2
        let c = clip();
        let (dx, dy) = (engine(&c[3]), engine(&c[2]));
        let job = CreateVectorsJob::single(base());
        let want = back(debug_view_on(gouraud().get(), &dx, &dy, &job_flow(&dx, &dy, &job).unwrap(), code).unwrap());
        let got = rig(&[("display", "3"), ("debug_view_method", view), ("max_displace", "16"), ("draw_geom", "1")]).render(2).unwrap();
        assert_eq!(got, want, "view {view}");
    }
    // the view popup does nothing when Display is not Debug
    assert_eq!(rig(&[("debug_view_method", "1"), ("max_displace", "16")]).render(2).unwrap(), back(create_vectors(&x, &y, &base()).unwrap()));
    // with an FG matte the Debug display composes the layers, as the original's code does
    let m1 = disc_clip(18.0, 16.0, 11.0);
    let c = clip();
    let (dx, dy) = (engine(&c[3]), engine(&c[2]));
    let job = CreateVectorsJob {
        settings: base(),
        layers: vec![layer(true), layer(true)],
        mattes_x: vec![extract(&engine(&m1[3]), 1)],
        mattes_y: vec![extract(&engine(&m1[2]), 1)],
        points: Vec::new(),
        first: 0,
        last: 1,
    };
    let want = back(retime_engine::create_vectors::debug_job_view_on(StageSet::cpu().get(), gouraud().get(), &dx, &dy, &job, 4).unwrap());
    let got = rig(&[("display", "3"), ("debug_view_method", "1"), ("max_displace", "16")]).with_layer(FG1, m1).render(2).unwrap();
    assert_eq!(got, want);
}

#[test]
fn main_bg_track_unchecked_is_the_second_tracker() {
    // no point: the zero vector everywhere (127 in R and G, B 0, opaque)
    let got = rig(&[("main_bg_track", "0")]).render(2).unwrap();
    assert_eq!(got, zero_vectors_image(W, H, Fmt::U8).to_frame());
    let Pixels::U8(p) = &got.pixels else { panic!() };
    assert!(p.iter().all(|px| *px == [127, 127, 0, 255]));
    // a point on Main_BG: the field follows it
    let got = rig(&[("main_bg_track", "0"), ("max_displace", "16"), ("pt1_use", "5"), ("pt1_pos", "10,10@1;13,11@2")]).render(2).unwrap();
    let flow = decode(&got, 16.0);
    let [u, v] = flow.at(10, 10);
    assert!((u + 3.0).abs() < 0.2 && (v + 1.0).abs() < 0.2, "{u} {v}");
    // the Source display never reaches a tracker
    assert!(rig(&[("main_bg_track", "0"), ("display", "1")]).render(2).is_ok());
    // the FG Track checkboxes only matter for FG layers
    let plain = rig(&[]).render(2).unwrap();
    assert_eq!(rig(&[("fg1_track", "0"), ("fg2_track", "0"), ("fg3_track", "0")]).render(2).unwrap(), plain);
}

#[test]
fn tracking_points_in_use_at_both_frames_reach_the_solver() {
    let (x, y) = pair();
    let plain = back(create_vectors(&x, &y, &base()).unwrap());
    let at = |t: i32| frame_time(t, 100);
    // Pt 1 on FG1: with one layer it belongs to the only layer; at frame 2 (Previous: ticks 200 and 100)
    let params = [("max_displace", "16"), ("pt1_use", "2"), ("pt1_pos", "20,15@1;24,17@2")];
    let got = rig(&params).render(2).unwrap();
    let mut job = CreateVectorsJob::single(base());
    job.points = vec![vec![UserPoint { at_a: flip_pos([24.0, 17.0], H), at_b: flip_pos([20.0, 15.0], H) }]];
    assert_eq!((at(2), at(1)), (200, 100));
    assert_eq!(got, back(create_vectors_job(&x, &y, &job).unwrap()));
    assert_ne!(got, plain);
    // Don't Use (the default) and a value that differs at the two frame times change nothing
    assert_eq!(rig(&[("max_displace", "16"), ("pt1_use", "1")]).render(2).unwrap(), plain);
    assert_eq!(rig(&[("max_displace", "16"), ("pt1_use", "2@0;3@2"), ("pt1_pos", "20,15")]).render(2).unwrap(), plain);
    // a point that never moves holds its pixel still (Full res: the point's pixel is an output pixel)
    let still = rig(&[("max_displace", "16"), ("vector_resolution", "2"), ("pt1_use", "5"), ("pt1_pos", "20,15")]).render(2).unwrap();
    assert_eq!(decode(&still, 16.0).at(20, 15), [0.0, 0.0]);
}

#[test]
fn draw_geom_draws_over_the_source_and_the_debug_display() {
    // default Draw Geom is All: with no point in use the overlay draws nothing and the Source display is a plain copy
    let source = rig(&[("display", "1")]).render(2).unwrap();
    assert_eq!(source, clip()[2]);
    // a point in use at the frame time: the plugin's cross in its colour (Pt 3 is white), arms of
    // full coverage 3 pixels each way, a quarter one pixel further, nothing beyond
    let got = rig(&[("display", "1"), ("pt3_use", "2"), ("pt3_pos", "10,12")]).render(2).unwrap();
    let Pixels::U8(p) = &got.pixels else { panic!() };
    let Pixels::U8(s) = &source.pixels else { panic!() };
    assert_eq!(p[12 * W + 10], [255, 255, 255, 255]);
    assert_eq!(p[12 * W + 11], [255, 255, 255, 255]);
    assert_eq!(p[12 * W + 13], [255, 255, 255, 255]);
    assert_eq!(p[9 * W + 10], [255, 255, 255, 255]);
    assert_ne!(p[12 * W + 14], s[12 * W + 14]);
    assert_ne!(p[12 * W + 14], [255, 255, 255, 255]);
    assert_eq!(p[12 * W + 15], s[12 * W + 15]);
    // Draw Geom None switches the overlay off; one layer: FG1 to Main_BG all name the only layer
    assert_eq!(rig(&[("display", "1"), ("pt3_use", "2"), ("pt3_pos", "10,12"), ("draw_geom", "1")]).render(2).unwrap(), source);
    for g in ["3", "4", "5", "6"] {
        assert_eq!(rig(&[("display", "1"), ("pt3_use", "2"), ("pt3_pos", "10,12"), ("draw_geom", g)]).render(2).unwrap(), got, "draw geom {g}");
    }
    // with FG1 present, Draw Geom Main_BG does not draw a point of FG1
    let m1 = disc_clip(18.0, 16.0, 11.0);
    assert_eq!(rig(&[("display", "1"), ("pt3_use", "2"), ("pt3_pos", "10,12"), ("draw_geom", "6")]).with_layer(FG1, m1).render(2).unwrap(), source);
    // the Debug display draws it over the view
    let debug = rig(&[("display", "3"), ("pt3_use", "2"), ("pt3_pos", "10,12"), ("draw_geom", "1")]).render(2).unwrap();
    let drawn = rig(&[("display", "3"), ("pt3_use", "2"), ("pt3_pos", "10,12")]).render(2).unwrap();
    assert_ne!(debug, drawn);
    let Pixels::U8(p) = &drawn.pixels else { panic!() };
    assert_eq!(p[12 * W + 10], [255, 255, 255, 255]);
    // Display Vectors never draws the overlay
    let vectors = rig(&[("pt3_use", "2"), ("pt3_pos", "10,12"), ("max_displace", "16")]).render(2).unwrap();
    let Pixels::U8(p) = &vectors.pixels else { panic!() };
    assert_ne!(p[12 * W + 10], [255, 255, 255, 255]);
}

#[test]
fn a_side_of_one_is_the_zero_vector_whatever_the_settings() {
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        for (w, h) in [(1usize, 1usize), (1, 17), (17, 1)] {
            let c = random_clip(depth, w, h, 6, 4);
            let zero = zero_vectors_image(w, h, Fmt::from_depth(depth)).to_frame();
            let cases: [&[(&str, &str)]; 6] = [
                &[("main_bg_track", "0")],
                &[("pt1_use", "5"), ("pt1_pos", "0,0@1;0,1@2")],
                &[("display", "3"), ("debug_view_method", "1")],
                &[("display", "3"), ("debug_view_method", "4"), ("pt2_use", "2")],
                &[("layers_output", "2"), ("fg1_track", "0")],
                &[("layers_output", "5"), ("fg1_inv_matte_shrink", "3")],
            ];
            for params in cases {
                for fg in [false, true] {
                    let mut r = Rig::new(c.clone(), params).with_frames(6);
                    if fg {
                        r = r.with_layer(FG1, random_clip(depth, w, h, 6, 5));
                    }
                    assert_eq!(r.render(2).unwrap(), zero, "{depth:?} {w}x{h} fg {fg} {params:?}");
                }
            }
        }
    }
}

#[test]
fn an_error_fill_wins_over_everything() {
    // a FG matte of another size is a size mismatch (red), whatever the other settings
    let small = random_clip(Depth::U8, 8, 8, 6, 9);
    let r = rig(&[("display", "3"), ("main_bg_track", "0"), ("pt1_use", "2")]).with_layer(FG1, small);
    let out = r.render(2).unwrap();
    let Pixels::U8(p) = &out.pixels else { panic!() };
    assert!(p.iter().all(|px| *px == [255, 0, 0, 128]), "50 percent red at every pixel");
    // and the Source display with a mismatching Alt Motion Src too
    let alt = random_clip(Depth::U8, 8, 8, 6, 9);
    let out = rig(&[("display", "1")]).with_layer("alt_source_a", alt).render(2).unwrap();
    let Pixels::U8(p) = &out.pixels else { panic!() };
    assert!(p.iter().all(|px| *px == [255, 0, 0, 128]));
}
