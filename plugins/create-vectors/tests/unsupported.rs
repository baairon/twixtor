//! Each unsupported feature gives an `FxError::Unsupported` with its own message, pinned literally
//! here, and only when the parameters really select that path (`glue.rs` has the list): a FG matte
//! layer with pixels, Display Debug (five views), `Main_BG: Track` unchecked, user tracking points
//! reaching the solver, and the Draw Geom overlay over the Source display. An error fill (size
//! mismatch, missing layer) wins over every refusal, as the flags are tested first. Nothing is
//! approximated.

mod common;

use common::*;
use fx_core::{Depth, Frame, FxError, Pixels};

const FG1: &str = "fg1_matte";
const FG2: &str = "fg2_matte";

fn clip() -> Vec<Frame> {
    random_clip(Depth::U8, 24, 16, 6, 5)
}

fn rig(params: &[(&str, &str)]) -> Rig {
    Rig::new(clip(), params).with_frames(6)
}

fn unsupported(r: &Rig, n: u64) -> String {
    match r.render(n) {
        Err(FxError::Unsupported(m)) => m,
        Err(e) => panic!("another error: {e}"),
        Ok(_) => panic!("rendered, expected Unsupported"),
    }
}

const FG_MESSAGE: &str = "a FG Matte layer is set: the multi layer path (one flow per layer, matte weighted \
merge of the vector images) is not implemented; remove the FG matte layers (Create Vectors \
handles the single layer case only)";

const TRACK_MESSAGE: &str = "Main_BG: Track is unchecked: the second tracker (a displacement \
interpolator driven by user geometry, used instead of the optical flow) is not implemented; check Main_BG: Track";

const DRAW_GEOM_MESSAGE: &str = "Draw Geom would draw the geometry of the tracking points over the source view, which is \
not implemented; set Draw Geom to None or Pt n: Use to Don't Use";

#[test]
fn a_fg_matte_layer_is_refused_for_the_vector_display() {
    let r = rig(&[]).with_layer(FG1, clip());
    assert_eq!(unsupported(&r, 2), FG_MESSAGE);
    // FG2 alone is not active (the scan stops at the first empty matte), so nothing is refused
    assert!(rig(&[]).with_layer(FG2, clip()).render(2).is_ok());
    // every Track Frame, and Debug too (the FG check comes first)
    assert_eq!(unsupported(&rig(&[("track_frame", "2")]).with_layer(FG1, clip()), 2), FG_MESSAGE);
    assert_eq!(unsupported(&rig(&[("display", "3")]).with_layer(FG1, clip()), 2), FG_MESSAGE);
}

#[test]
fn the_source_display_ignores_fg_mattes_without_draw_geom() {
    // Display Source copies the source whatever the mattes (the layer count only matters for the overlay)
    let with = rig(&[("display", "1")]).with_layer(FG1, clip());
    let without = rig(&[("display", "1")]);
    assert_eq!(with.render(2).unwrap(), without.render(2).unwrap());
}

#[test]
fn every_debug_view_is_refused_with_its_own_message() {
    let want = [
        ("1", "Display Debug with Debug View Method \"Warp\" draws the forward warp of the first frame with the flow, which is not implemented; set Display to Vectors or Source"),
        ("2", "Display Debug with Debug View Method \"Blend\" draws the blend view of the two frames, which is not implemented; set Display to Vectors or Source"),
        ("3", "Display Debug with Debug View Method \"Diff\" draws the difference view of the two frames, which is not implemented; set Display to Vectors or Source"),
        ("4", "Display Debug with Debug View Method \"Detail Compare\" draws the detail comparison view, which is not implemented; set Display to Vectors or Source"),
        ("5", "Display Debug with Debug View Method \"Edge Compare\" draws the edge comparison view, which is not implemented; set Display to Vectors or Source"),
    ];
    for (view, message) in want {
        for track in ["1", "2"] {
            assert_eq!(unsupported(&rig(&[("display", "3"), ("debug_view_method", view), ("track_frame", track)]), 2), message, "view {view}");
        }
    }
    // the view popup does nothing when Display is not Debug
    assert!(rig(&[("debug_view_method", "1")]).render(2).is_ok());
}

#[test]
fn main_bg_track_unchecked_is_the_second_tracker() {
    assert_eq!(unsupported(&rig(&[("main_bg_track", "0")]), 2), TRACK_MESSAGE);
    // the Source display never reaches a tracker
    assert!(rig(&[("main_bg_track", "0"), ("display", "1")]).render(2).is_ok());
    // the FG Track checkboxes only matter for FG layers
    assert!(rig(&[("fg1_track", "0"), ("fg2_track", "0"), ("fg3_track", "0")]).render(2).is_ok());
}

#[test]
fn tracking_points_in_use_at_both_frames_are_refused() {
    let m = |list: &str| {
        format!(
            "user tracking points feed the flow solver as user displacement, which is not implemented; set {list} to Don't Use"
        )
    };
    // Pt 1 is FG1 and Pt 4 is Main_BG: with one layer (no FG matte) both belong to the only layer
    assert_eq!(unsupported(&rig(&[("pt1_use", "2")]), 2), m("Pt 1: Use"));
    assert_eq!(unsupported(&rig(&[("pt1_use", "2"), ("pt4_use", "5")]), 2), m("Pt 1: Use, Pt 4: Use"));
    assert_eq!(unsupported(&rig(&[("pt12_use", "4"), ("track_frame", "2")]), 2), m("Pt 12: Use"));
    // Don't Use (the default) is fine, and so is a point whose value differs at the two frame times:
    // frame 3 with Previous reads frames 3 and 2 (ticks 300 and 200), the key switches at frame 3 (tick 300)
    assert!(rig(&[("pt1_use", "1")]).render(2).is_ok());
    assert!(rig(&[("pt1_use", "1@0;2@3")]).render(3).is_ok(), "Don't Use at the first time (tick 300 is the key: value 2) differs from tick 200");
    assert!(rig(&[("pt1_use", "2@0;3@3")]).render(3).is_ok(), "FG1 at tick 200, FG2 at tick 300: the values differ");
    assert!(rig(&[("pt1_use", "2@0;3@3")]).render(2).is_err(), "frames 2 and 1 are both before the key: FG1 at both");
    // the Source display does not run the solver
    assert!(rig(&[("pt1_use", "2"), ("display", "1"), ("draw_geom", "1")]).render(2).is_ok());
}

#[test]
fn draw_geom_over_the_source_view_needs_geometry_to_be_refused() {
    // default Draw Geom is All: with no point in use the overlay draws nothing and the Source display is a plain copy
    assert!(rig(&[("display", "1")]).render(2).is_ok());
    // a point in use at the frame time: refused; Draw Geom None switches the overlay off
    assert_eq!(unsupported(&rig(&[("display", "1"), ("pt3_use", "2")]), 2), DRAW_GEOM_MESSAGE);
    assert!(rig(&[("display", "1"), ("pt3_use", "2"), ("draw_geom", "1")]).render(2).is_ok());
    // one layer only: FG1, FG2, FG3 and Main_BG all name the only layer
    for g in ["3", "4", "5", "6"] {
        assert_eq!(unsupported(&rig(&[("display", "1"), ("pt3_use", "2"), ("draw_geom", g)]), 2), DRAW_GEOM_MESSAGE, "draw geom {g}");
    }
    // Display Vectors never draws the overlay: with the point it is the solver that refuses
    let m = unsupported(&rig(&[("pt3_use", "2"), ("draw_geom", "1")]), 2);
    assert!(m.starts_with("user tracking points feed the flow solver"), "{m}");
}

#[test]
fn the_order_of_the_refusals_is_fg_then_debug_then_track_then_points() {
    let all = [("display", "3"), ("main_bg_track", "0"), ("pt1_use", "2")];
    assert_eq!(unsupported(&rig(&all).with_layer(FG1, clip()), 2), FG_MESSAGE);
    assert!(unsupported(&rig(&all), 2).starts_with("Display Debug"));
    assert_eq!(unsupported(&rig(&[("main_bg_track", "0"), ("pt1_use", "2")]), 2), TRACK_MESSAGE);
}

#[test]
fn an_error_fill_wins_over_every_refusal() {
    // a FG matte of another size is a size mismatch (red), even though a FG matte would be refused
    let small = random_clip(Depth::U8, 8, 8, 6, 9);
    let r = rig(&[("display", "3"), ("main_bg_track", "0"), ("pt1_use", "2")]).with_layer(FG1, small);
    let out = r.render(2).unwrap();
    let Pixels::U8(p) = &out.pixels else { panic!() };
    assert!(p.iter().all(|px| *px == [255, 0, 0, 128]), "50 percent red at every pixel");
    // and the Source display with a mismatching Alt Src too
    let alt = random_clip(Depth::U8, 8, 8, 6, 9);
    let out = rig(&[("display", "1")]).with_layer("alt_source_a", alt).render(2).unwrap();
    let Pixels::U8(p) = &out.pixels else { panic!() };
    assert!(p.iter().all(|px| *px == [255, 0, 0, 128]));
}
