//! What plugin 03's front-end once refused now renders: a FG Matte layer (the n-layer path), user
//! tracking points reaching the solver, the Draw Geom overlay of such points over the Source
//! display, and Track RGB+A (the engine's wrap-around flag). Each is checked to render where it
//! applies and to change nothing where the render stays on the single layer fast path (exact
//! positions, Vectors off, the segment markers that switch vectors off, the Source display).
//!
//! The clips are the tiny golden clips where no real motion is needed (the motion search never
//! halves below 2x2, so these sizes are searched at their own size), and a textured moving clip
//! where a render must differ.

mod common;

use common::*;
use fx_core::{Depth, Frame, Pixels};

/// An opaque clip of `n` frames of `w x h` whose red channel (the default matte channel) is `red`.
fn flat(w: usize, h: usize, n: usize, depth: Depth, red: f64) -> Vec<Frame> {
    (0..n)
        .map(|_| {
            let mut f = Frame::new(w, h, depth);
            match &mut f.pixels {
                Pixels::U8(px) => px.fill([(red * 255.0) as u8, 0, 0, 255]),
                Pixels::U16(px) => px.fill([(red * 65535.0) as u16, 0, 0, 65535]),
                Pixels::F32(px) => px.fill([red as f32, 0.0, 0.0, 1.0]),
            }
            f
        })
        .collect()
}

/// The textured clip of the moving scene, large enough for the solver.
fn moving() -> Vec<Frame> {
    common::moving::DEFAULT.clip(Depth::U8, 32, 24, 8)
}

/// The render passes every check of the front-end and renders.
fn passes(rig: &Rig, n: u64) {
    if let Err(e) = rig.render(n) {
        panic!("frame {n}: {e}");
    }
}

#[test]
fn every_warping_code_passes_the_checks_of_the_front_end() {
    // The engine serves all four warp codes (inverse, inverse with Smart Blend, forward, forward
    // without Smart Blend). A render between frames passes the checks and renders the tiny clips
    // (searched at their own size); real renders of the forward codes are in `forward.rs`.
    for set in sets() {
        for warping in ["1", "2", "3", "4"] {
            for interp in ["1", "2", "3"] {
                let rig = set.rig(&[("speed_percent", "50"), ("warping", warping), ("frame_interp", interp)]);
                passes(&rig, 1);
                passes(&rig, 3);
            }
            let rig = set.rig(&[("speed_percent", "50"), ("warping", warping)]);
            // an exact position never reaches a warp, and neither does No Vectors
            assert_eq!(rig.render(2).unwrap(), set.rt[1], "{:?} warping {warping}", set.depth);
            let plain = set.rig(&[("speed_percent", "50"), ("warping", warping), ("vectors", "1")]);
            assert_eq!(plain.render(1).unwrap(), set.b01);
        }
    }
}

#[test]
fn blur_compensation_makes_exact_positions_take_the_vector_path() {
    // exact_position: a source position that moves between the two output frames with blur above 0.005 is not exact
    let set = &sets()[0];
    let rig = set.rig(&[("warping", "3"), ("blur_compensation", "2")]);
    passes(&rig, 1);
    // a still position (speed 0) stays exact whatever the blur is
    assert_eq!(set.rig(&[("warping", "3"), ("blur_compensation", "2"), ("speed_percent", "0")]).render(1).unwrap(), set.rt[0]);
    // blur at or below 0.005 does not count (0.004 is stored as 262 / 65536; 0.005 would be 328 / 65536, above the limit)
    assert_eq!(set.rig(&[("warping", "3"), ("blur_compensation", "0.004")]).render(1).unwrap(), set.rt[1]);
    passes(&set.rig(&[("warping", "3"), ("blur_compensation", "0.005")]), 1);
}

#[test]
fn the_segment_markers_that_switch_vectors_off_also_switch_the_refusals_off() {
    // Src: Mark Segments 4 (No MV, Use Nearest): the override gives (vectors 0, Nearest), so no warp runs
    let set = &sets()[0];
    let rig = set.rig(&[("speed_percent", "50"), ("warping", "3"), ("src_mark_segments", "4")]);
    assert_eq!(rig.render(1).unwrap(), set.rt[0], "Nearest picks A at weight 0.5");
    // 5 (No MV, Frame Blend): the plain blend
    let rig = set.rig(&[("speed_percent", "50"), ("warping", "3"), ("src_mark_segments", "5")]);
    assert_eq!(rig.render(1).unwrap(), set.b01);
    // a cut (marker A on frame 0 and B on frame 1) also drops the vectors: Nearest
    let rig = set.rig(&[("speed_percent", "50"), ("warping", "3"), ("src_mark_segments", "1@0;2@1")]);
    assert_eq!(rig.render(1).unwrap(), set.rt[0]);
}

#[test]
fn fg_matte_layers_render_on_every_frame_and_the_groups_of_the_fast_path_are_exact() {
    for set in sets() {
        let (w, h, d) = (set.width, set.height, set.depth);
        // between frames and on them, with one or three mattes, Vectors on or off, every Display: the render succeeds
        for speed in ["100", "50"] {
            for vectors in ["1", "5"] {
                for display in ["1", "3", "4", "5", "6"] {
                    let p = [("speed_percent", speed), ("vectors", vectors), ("display", display)];
                    let one = set.rig(&p).with_layer("fg1_matte", set.clip.clone());
                    let all = set.rig(&p).with_layer("fg1_matte", set.clip.clone()).with_layer("fg2_matte", set.clip.clone()).with_layer("fg3_matte", set.clip.clone());
                    for n in 0..4 {
                        one.render(n).unwrap_or_else(|e| panic!("{d:?} {p:?} frame {n}: {e}"));
                        all.render(n).unwrap_or_else(|e| panic!("{d:?} {p:?} three mattes, frame {n}: {e}"));
                    }
                }
            }
        }
        // the fast path composes the displayed group of the unwarped frames. On an opaque clip, a white matte
        // (1 everywhere): FG1 covers all, so every group shows the frame itself; a black matte (0): FG1 alone is
        // transparent and the others show the frame (Main_BG's composition matte is 1)
        let opaque = common::moving::DEFAULT.clip(d, w, h, 4);
        let frame = Rig::new(opaque.clone(), &[]).render(1).unwrap();
        let on = |display: &str, matte: &Vec<Frame>| Rig::new(opaque.clone(), &[("display", display)]).with_layer("fg1_matte", matte.clone()).render(1).unwrap();
        let white = flat(w, h, 4, d, 1.0);
        let black = flat(w, h, 4, d, 0.0);
        for display in ["1", "3", "6"] {
            assert_eq!(on(display, &white), frame, "{d:?} white matte, display {display}");
        }
        for (display, want) in [("1", frame.clone()), ("3", set.transparent()), ("6", frame.clone())] {
            assert_eq!(on(display, &black), want, "{d:?} black matte, display {display}");
        }
        // with one matte the displays 4 and 5 (FG2, FG3) are the last group, Main_BG, as 6 is
        for display in ["4", "5"] {
            assert_eq!(on(display, &black), frame, "{d:?} display {display}");
        }
        // a semi transparent frame is composed over itself: FG1's layer image (the frame times 1) OVER Main_BG's
        // (the frame) raises the alpha, taken literally
        let rt = set.rig(&[]).with_layer("fg1_matte", white.clone()).render(1).unwrap();
        assert_ne!(rt, set.rt[1], "{d:?}: the golden frame has partial alpha");
        // FG2 without FG1 is not counted (the probe loop stops at FG1)
        assert_eq!(set.rig(&[]).with_layer("fg2_matte", set.clip.clone()).render(1).unwrap(), set.rt[1]);
        // the Source display copies the frame
        assert_eq!(set.rig(&[("display", "2")]).with_layer("fg1_matte", set.clip.clone()).render(1).unwrap(), set.rt[1]);
    }
}

#[test]
fn a_fg_matte_layer_changes_the_vector_path_render() {
    // a textured matte splits the moving clip into two layers with their own flows
    let plain = Rig::new(moving(), &[("speed_percent", "50")]);
    let matte = common::moving::Motion { vx: -1.0, vy: 0.5 }.clip(Depth::U8, 32, 24, 8);
    let layered = Rig::new(moving(), &[("speed_percent", "50")]).with_layer("fg1_matte", matte);
    assert_ne!(layered.render(1).unwrap(), plain.render(1).unwrap());
    // the exact frame of display 1 with a hard matte is the frame itself, as without the matte
    let white = flat(32, 24, 8, Depth::U8, 1.0);
    let layered = Rig::new(moving(), &[("speed_percent", "50")]).with_layer("fg1_matte", white);
    assert_eq!(layered.render(2).unwrap(), plain.render(2).unwrap());
}

#[test]
fn track_points_reach_the_solver_only_on_the_vector_path() {
    let set = &sets()[0];
    // Pt 2 on FG 1 and Pt 5 on Main_BG Layer: with no FG layer every used point belongs to the only group
    let rig = set.rig(&[("speed_percent", "50"), ("pt2_use", "2"), ("pt5_use", "5"), ("pt2_pos", "1,1"), ("pt5_pos", "2,0")]);
    passes(&rig, 1);
    // exact frames and No Vectors never build the user displacement
    assert_eq!(rig.render(2).unwrap(), set.rt[1]);
    assert_eq!(set.rig(&[("speed_percent", "50"), ("pt2_use", "2"), ("vectors", "1")]).render(1).unwrap(), set.b01);
    // on a textured clip a point that moves against the scene changes the flow, so the in between frame
    let plain = Rig::new(moving(), &[("speed_percent", "50")]);
    let pointed = Rig::new(moving(), &[("speed_percent", "50"), ("pt1_use", "5"), ("pt1_pos", "16,12@0;10,14@1")]);
    assert_ne!(pointed.render(1).unwrap(), plain.render(1).unwrap());
    assert_eq!(pointed.render(2).unwrap(), plain.render(2).unwrap(), "an exact frame");
    // a point that is unused ("Don't Use", the default) or whose use changes between the two frame times is not geometry:
    // output frame 1 of half speed reads source frames 0 and 1, the times 0 and 100, and "Don't Use" at tick 0 with FG 2
    // from tick 100 on (a popup holds its key) is unused at the first time
    let changing = Rig::new(moving(), &[("speed_percent", "50"), ("pt3_use", "1@0;3@1"), ("pt3_pos", "16,12@0;10,14@1")]);
    assert_eq!(changing.render(1).unwrap(), plain.render(1).unwrap());
    // output frame 3 reads source frames 1 and 2, the times 100 and 200: pt3 is 3 at both, so it is in use there
    let held = Rig::new(moving(), &[("speed_percent", "50"), ("pt3_use", "1@0;3@1"), ("pt3_pos", "16,12@0;10,14@1;4,16@2")]);
    assert_ne!(held.render(3).unwrap(), plain.render(3).unwrap());
}

/// The pixels of `f` (8 bpc) that differ from `g`.
fn changed(f: &Frame, g: &Frame) -> Vec<(usize, [u8; 4])> {
    match (&f.pixels, &g.pixels) {
        (Pixels::U8(a), Pixels::U8(b)) => a.iter().zip(b).enumerate().filter(|(_, (x, y))| x != y).map(|(i, (x, _))| (i, *x)).collect(),
        _ => panic!("8 bpc frames"),
    }
}

#[test]
fn the_draw_geom_overlay_draws_the_points_in_use_over_the_source_display() {
    let clip = random_clip(Depth::U8, 16, 12, 4, 5);
    let src = Rig::new(clip.clone(), &[("display", "2")]).render(1).unwrap();
    let rig = |p: &[(&str, &str)]| Rig::new(clip.clone(), p);
    // Draw Geom defaults to All: Pt 4 (yellow by default) at (5, 4) is an opaque disc of radius 1.5, the 3 x 3 block
    let drawn = rig(&[("display", "2"), ("pt4_use", "5"), ("pt4_pos", "5,4")]).render(1).unwrap();
    let want: Vec<(usize, [u8; 4])> = [3usize, 4, 5].iter().flat_map(|&y| [4usize, 5, 6].map(|x| (y * 16 + x, [255, 255, 0, 255]))).collect();
    assert_eq!(changed(&drawn, &src), want);
    // with no FG layer every group value 2 to 6 holds the point (they clamp to the one layer); later points on top
    for geom in ["2", "3", "4", "5", "6"] {
        let r = rig(&[("display", "2"), ("pt4_use", "5"), ("pt4_pos", "5,4"), ("draw_geom", geom)]);
        assert_eq!(r.render(1).unwrap(), drawn, "draw_geom {geom}");
    }
    let two = rig(&[("display", "2"), ("pt4_use", "5"), ("pt4_pos", "5,4"), ("pt6_use", "2"), ("pt6_pos", "6,4")]).render(1).unwrap();
    assert_eq!(&two.to_bytes()[(4 * 16 + 6) * 4..(4 * 16 + 7) * 4], &[0, 255, 255, 255], "Pt 6 (cyan) is drawn after Pt 4");
    // None draws nothing; an unused point or one outside the frame (the default position) draws nothing
    assert_eq!(rig(&[("display", "2"), ("pt4_use", "5"), ("pt4_pos", "5,4"), ("draw_geom", "1")]).render(1).unwrap(), src);
    assert_eq!(rig(&[("display", "2"), ("pt4_pos", "5,4")]).render(1).unwrap(), src);
    assert_eq!(rig(&[("display", "2"), ("pt4_use", "5")]).render(1).unwrap(), src);
    // with FG1 the groups split: a Main_BG point is drawn by All and Main_BG (6), not by FG1 (3)
    let matte = random_clip(Depth::U8, 16, 12, 4, 9);
    let fg = |geom: &str| rig(&[("display", "2"), ("pt4_use", "5"), ("pt4_pos", "5,4"), ("draw_geom", geom)]).with_layer("fg1_matte", matte.clone()).render(1).unwrap();
    assert_eq!(fg("2"), drawn);
    assert_eq!(fg("6"), drawn);
    assert_eq!(fg("3"), src);
    // the overlay only exists in the Source display: the processed output with a point on an exact frame is the frame
    let set = &sets()[0];
    assert_eq!(set.rig(&[("pt4_use", "5"), ("pt4_pos", "1,1")]).render(1).unwrap(), set.rt[1]);
}

#[test]
fn track_rgb_a_is_the_wrap_flag_where_vectors_are_computed() {
    let set = &sets()[0];
    let rig = set.rig(&[("speed_percent", "50"), ("track_rgb_a", "1")]);
    passes(&rig, 1);
    assert_eq!(rig.render(2).unwrap(), set.rt[1]);
    assert_eq!(set.rig(&[("speed_percent", "50"), ("track_rgb_a", "1"), ("vectors", "1")]).render(1).unwrap(), set.b01);
    // on the textured clip the wrapped warps and solve read across the left and right borders
    let plain = Rig::new(moving(), &[("speed_percent", "50")]);
    let wrapped = Rig::new(moving(), &[("speed_percent", "50"), ("track_rgb_a", "1")]);
    assert_ne!(wrapped.render(1).unwrap(), plain.render(1).unwrap());
    assert_eq!(wrapped.render(2).unwrap(), plain.render(2).unwrap(), "an exact frame has no motion to wrap");
}

#[test]
fn the_360_video_checkbox_is_not_a_refusal() {
    // plugin 01's engine reading of the checkbox (the video range luma of the solver) applies; it
    // only changes what the solver reads, so with the tiny clips the render reaches the solver
    for set in sets() {
        passes(&set.rig(&[("speed_percent", "50"), ("video_360", "1")]), 1);
        assert_eq!(set.rig(&[("video_360", "1")]).render(2).unwrap(), set.rt[2]);
    }
}

#[test]
fn every_feature_once_refused_renders_together() {
    // FG mattes, a point on FG1, Track RGB+A and every warping code, between frames
    for warping in ["1", "2", "3", "4"] {
        let p = [("speed_percent", "50"), ("warping", warping), ("track_rgb_a", "1"), ("pt1_use", "2"), ("pt1_pos", "16,12@0;18,12@1")];
        let matte = common::moving::Motion { vx: 1.0, vy: 0.0 }.clip(Depth::U8, 32, 24, 8);
        let r = Rig::new(moving(), &p).with_layer("fg1_matte", matte);
        r.render(1).unwrap_or_else(|e| panic!("warping {warping}: {e}"));
    }
}

#[test]
fn an_error_fill_wins_over_every_refusal() {
    // the flags of the probes are tested before the engine is entered, so a missing input is blue
    // and a layer of the wrong size red even when the settings could not be rendered anyway
    let set = &sets()[0];
    let (w, h) = (set.width, set.height);
    let rig = set.rig(&[("speed_percent", "50"), ("warping", "3"), ("pt1_use", "2")]).with_layer("alt_source_b", vec![fx_core::Frame::new(w + 1, h, set.depth)]);
    assert_eq!(rig.render(1).unwrap(), fill(set.depth, true, w, h));
}
