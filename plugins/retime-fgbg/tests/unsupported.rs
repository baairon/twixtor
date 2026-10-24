//! What plugin 03's front-end refuses, with the exact messages: a FG Matte layer (the multi layer
//! path), user tracking points reaching the solver, the Draw Geom overlay of such points, and Track
//! RGB+A. Every refusal is an `FxError::Unsupported`, never a silent fallback; every one of them is
//! also checked to NOT fire where the render stays on the single layer fast path (exact positions,
//! Vectors off, the segment markers that switch vectors off, the Source display).
//!
//! The clips are the tiny golden clips: the checks run before the solver, so a refusal needs no
//! real motion, and a render that passes the checks reaches the solver's own "below 2x2" refusal at
//! these sizes (the tests tell the two apart by the message).

mod common;

use common::*;
use fx_core::FxError;

const FG: &str = "a FG Matte layer is set: the multi layer path (one flow per layer, mattes, Main_BG compositing) is not implemented; remove the FG Matte layers (this render handles the single layer case only)";
const DRAW_GEOM: &str = "Draw Geom would draw the geometry of the tracking points over the source view, which is not implemented; set Draw Geom to None or Pt n: Use to Don't Use";
const TRACK_RGB_A: &str = "Track RGB+A drives the engine's 360 degree flag (wrap around warps, solver and hole fill), which is not implemented; uncheck Track RGB+A (it only matters when motion vectors are computed)";

fn points(list: &str) -> String {
    format!("user tracking points feed the flow solver as user displacement, which is not implemented; set {list} to Don't Use")
}

/// Asserts that rendering output frame `n` is the `Unsupported` error with exactly this message.
fn refused(rig: &Rig, n: u64, message: &str) {
    match rig.render(n) {
        Err(FxError::Unsupported(m)) => assert_eq!(m, message),
        other => panic!("frame {n}: expected Unsupported({message}), got {other:?}"),
    }
}

/// The render passes every check of the front-end: it either succeeds or reaches the solver of the engine.
fn passes(rig: &Rig, n: u64) {
    match rig.render(n) {
        Ok(_) => {}
        Err(FxError::Unsupported(m)) => assert!(m.contains("below 2x2"), "{m}"),
        Err(e) => panic!("frame {n}: {e}"),
    }
}

#[test]
fn every_warping_code_passes_the_checks_of_the_front_end() {
    // The engine serves all four warp codes (inverse, inverse with Smart Blend, forward, forward
    // without Smart Blend). A render between frames passes the checks and reaches the solver of the
    // engine, which refuses the tiny clips by name; real renders of the forward codes are in
    // `forward.rs`.
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
fn fg_matte_layers_are_refused_even_on_exact_frames_but_not_in_the_source_display() {
    for set in sets() {
        let rig = set.rig(&[]).with_layer("fg1_matte", set.clip.clone());
        refused(&rig, 0, FG);
        refused(&rig, 2, FG);
        // with Vectors off too: the multi layer path is chosen by the layer count
        refused(&set.rig(&[("vectors", "1")]).with_layer("fg1_matte", set.clip.clone()), 0, FG);
        // any of the three, with or without the others
        let all = set.rig(&[]).with_layer("fg1_matte", set.clip.clone()).with_layer("fg2_matte", set.clip.clone()).with_layer("fg3_matte", set.clip.clone());
        refused(&all, 1, FG);
        // the FG displays take the multi layer path too
        refused(&set.rig(&[("display", "3")]).with_layer("fg1_matte", set.clip.clone()), 0, FG);
        refused(&set.rig(&[("display", "6")]).with_layer("fg1_matte", set.clip.clone()), 0, FG);
        // FG2 without FG1 is not counted (the probe loop stops at FG1)
        assert_eq!(set.rig(&[]).with_layer("fg2_matte", set.clip.clone()).render(1).unwrap(), set.rt[1]);
        // the Source display copies the frame
        assert_eq!(set.rig(&[("display", "2")]).with_layer("fg1_matte", set.clip.clone()).render(1).unwrap(), set.rt[1]);
    }
}

#[test]
fn track_points_reach_the_solver_only_on_the_vector_path() {
    let set = &sets()[0];
    // Pt 2 on FG 1 and Pt 5 on Main_BG Layer: with no FG layer every used point belongs to the only group
    let rig = set.rig(&[("speed_percent", "50"), ("pt2_use", "2"), ("pt5_use", "5")]);
    refused(&rig, 1, &points("Pt 2: Use, Pt 5: Use"));
    // exact frames and No Vectors never build the user displacement
    assert_eq!(rig.render(2).unwrap(), set.rt[1]);
    assert_eq!(set.rig(&[("speed_percent", "50"), ("pt2_use", "2"), ("vectors", "1")]).render(1).unwrap(), set.b01);
    // a point that is unused ("Don't Use", the default) or whose use changes between the two frame times is not geometry
    passes(&set.rig(&[("speed_percent", "50"), ("pt3_use", "1")]), 1);
    passes(&set.rig(&[("speed_percent", "50"), ("pt3_use", "1@0;3@1")]), 1);
    // output frame 1 of half speed reads source frames 0 and 1, the times 0 and 100: "Don't Use" at tick 0 and FG 2 from tick
    // 100 on (a popup holds its key) is unused at the first time, so it is not geometry there
    let rig = set.rig(&[("speed_percent", "50"), ("pt3_use", "1@0;3@1")]);
    passes(&rig, 1);
    // output frame 3 reads source frames 1 and 2, the times 100 and 200: pt3 is 3 at both
    refused(&rig, 3, &points("Pt 3: Use"));
}

#[test]
fn the_draw_geom_overlay_of_track_points_is_refused_in_the_source_display() {
    let set = &sets()[0];
    // Draw Geom defaults to All, so a used point draws over the source view
    let rig = set.rig(&[("display", "2"), ("pt4_use", "5")]);
    refused(&rig, 1, DRAW_GEOM);
    for geom in ["2", "3", "4", "5", "6"] {
        refused(&set.rig(&[("display", "2"), ("pt4_use", "5"), ("draw_geom", geom)]), 1, DRAW_GEOM);
    }
    // None draws nothing; an unused point draws nothing
    assert_eq!(set.rig(&[("display", "2"), ("pt4_use", "5"), ("draw_geom", "1")]).render(1).unwrap(), set.rt[1]);
    assert_eq!(set.rig(&[("display", "2"), ("draw_geom", "2")]).render(1).unwrap(), set.rt[1]);
    // the overlay only exists in the Source display: the processed output with a point on an exact frame renders
    assert_eq!(set.rig(&[("pt4_use", "5")]).render(1).unwrap(), set.rt[1]);
}

#[test]
fn track_rgb_a_is_refused_only_where_vectors_are_computed() {
    let set = &sets()[0];
    let rig = set.rig(&[("speed_percent", "50"), ("track_rgb_a", "1")]);
    refused(&rig, 1, TRACK_RGB_A);
    assert_eq!(rig.render(2).unwrap(), set.rt[1]);
    assert_eq!(set.rig(&[("speed_percent", "50"), ("track_rgb_a", "1"), ("vectors", "1")]).render(1).unwrap(), set.b01);
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
fn several_unsupported_features_report_the_first_in_a_fixed_order() {
    // multi layer, then tracking points, then Track RGB+A (every warp code is served, so nothing follows)
    let set = &sets()[0];
    let all = [("speed_percent", "50"), ("warping", "3"), ("track_rgb_a", "1"), ("pt1_use", "2")];
    let with_fg = set.rig(&all).with_layer("fg1_matte", set.clip.clone());
    refused(&with_fg, 1, FG);
    refused(&set.rig(&all), 1, &points("Pt 1: Use"));
    refused(&set.rig(&[("speed_percent", "50"), ("warping", "3"), ("track_rgb_a", "1")]), 1, TRACK_RGB_A);
    passes(&set.rig(&[("speed_percent", "50"), ("warping", "3")]), 1);
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
