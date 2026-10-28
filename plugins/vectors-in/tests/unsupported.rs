//! What plugin 04 once refused now renders: a FG Matte layer takes the n-layer path of the engine,
//! the Warping popup "Forward" (its default, warp code 2) is served by the engine's forward warp
//! and a single vector layer by its flow inversion; the tests below pin that those renders succeed
//! and are close to the analytic frame of the moving clip, and that what is still an error is one
//! with a precise message, never a silent fallback.

mod common;

use common::moving::*;
use common::*;
use fx_core::Depth;
use vectors_in::glue;

const W: usize = 48;
const H: usize = 32;
const MD: f32 = 16.0;

fn base<'a>(extra: &[(&'a str, &'a str)]) -> Vec<(&'a str, &'a str)> {
    let mut p = vec![("vectors_max_displace", "16"), ("speed_percent", "50")];
    p.extend_from_slice(extra);
    p
}

fn rig(extra: &[(&str, &str)]) -> Rig {
    Rig::new(DEFAULT.clip(Depth::U8, W, H, 8), &base(extra)).with_frames(10).with_vectors(3.4, 1.7, MD)
}

/// Mean absolute error of output frame `n` of a half speed run against the exact scene at position `n / 2`.
fn truth_error(r: &Rig, n: u64) -> f64 {
    let truth = DEFAULT.frame_at(Depth::U8, W, H, n as f64 / 2.0);
    error(&r.render(n).unwrap(), &truth, 6).0
}

#[test]
fn the_forward_warp_is_the_default_and_renders_between_frames_close_to_the_truth() {
    // Warping defaults to "Forward" (popup 2, warp code 2), so the default parameter set takes the forward warp between
    // frames; the vector layers are the exact motion of the moving clip, so the result is the in between frame
    for n in [1u64, 3, 5] {
        let e = truth_error(&rig(&[]), n);
        assert!(e < 0.003, "default Forward frame {n}: error {e:.5} (measured 0.0017 to 0.0018, the frame blend is 0.021)");
        for warping in ["1", "2"] {
            let e = truth_error(&rig(&[("warping", warping)]), n);
            assert!(e < 0.003, "warping {warping} frame {n}: error {e:.5}");
        }
    }
    // the forward and the inverse warp agree on this clip to within the sampling of their different warps
    let (f, i) = (rig(&[]).render(3).unwrap(), rig(&[("warping", "1")]).render(3).unwrap());
    assert!(error(&f, &i, 6).0 < 0.003);
}

#[test]
fn exact_positions_and_markers_that_switch_the_vectors_off_never_reach_the_warp() {
    // 100 percent speed: every frame is exact, the engine never asks for a warp or for flows
    let full = Rig::new(DEFAULT.clip(Depth::U8, W, H, 8), &[("vectors_max_displace", "16")]).with_frames(8).with_vectors(3.4, 1.7, MD);
    for n in 0..8 {
        assert!(full.render(n).is_ok(), "exact frame {n} with the default Forward warping");
    }
    // the segment markers "No MV, Use Nearest" and "No MV, Frame Blend" (popup 4 and 5) switch the vectors off
    for marker in ["4", "5"] {
        let r = rig(&[("src_mark_segments", marker)]);
        for n in 0..8 {
            assert!(r.render(n).is_ok(), "marker {marker} frame {n}");
        }
    }
}

#[test]
fn a_fg_matte_layer_renders_the_n_layer_path() {
    let matte = DEFAULT.clip(Depth::U8, W, H, 8);
    let plain = rig(&[("warping", "1")]);
    // every frame, exact or not, every Display and both warpings render
    for warping in ["1", "2"] {
        let r = rig(&[("warping", warping)]).with_layer("fg1_matte", matte.clone());
        for n in [0u64, 1, 2, 3] {
            r.render(n).unwrap_or_else(|e| panic!("warping {warping} frame {n}: {e}"));
        }
        for display in ["1", "3", "4", "5", "6"] {
            let r = rig(&[("warping", warping), ("display", display)]).with_layer("fg1_matte", matte.clone());
            r.render(1).unwrap_or_else(|e| panic!("warping {warping} display {display}: {e}"));
        }
    }
    // a textured matte splits the frame into two layers: the output is not the single layer one
    let layered = rig(&[("warping", "1")]).with_layer("fg1_matte", matte.clone());
    assert_ne!(layered.render(1).unwrap(), plain.render(1).unwrap());
    // the Source display copies the source: the matte changes nothing there
    let source = |r: Rig| r.render(1).unwrap();
    assert_eq!(source(rig(&[("warping", "1"), ("display", "2")]).with_layer("fg1_matte", matte.clone())), source(rig(&[("warping", "1"), ("display", "2")])));
    // FG2 without FG1 is ignored (the probe loop stops at the first empty layer)
    let r = rig(&[("warping", "1")]).with_layer("fg2_matte", matte.clone());
    assert_eq!(r.render(1).unwrap(), plain.render(1).unwrap());
}

#[test]
fn an_all_black_fg_matte_gives_the_single_layer_render_with_the_inverse_warp() {
    // FG1's matte is 0 everywhere: its layer image is transparent and its tracking matte 0 (every vector a hole),
    // while Main_BG's tracking matte is 1 (the matte step changes nothing), so with the inverse warp (no fill) the
    // layered render is the single layer one bit for bit, between frames and on them, for every Frame Interp
    let black: Vec<fx_core::Frame> = (0..8).map(|_| fx_core::Frame::new(W, H, Depth::U8)).collect();
    let black = black
        .into_iter()
        .map(|mut f| {
            if let fx_core::Pixels::U8(px) = &mut f.pixels {
                px.fill([0, 0, 0, 255]);
            }
            f
        })
        .collect::<Vec<_>>();
    for interp in ["1", "2", "3"] {
        for blur in ["0", "1"] {
            let params = [("warping", "1"), ("frame_interp", interp), ("blur_compensation", blur)];
            let plain = rig(&params);
            let layered = rig(&params).with_layer("fg1_matte", black.clone());
            for n in [1u64, 2, 3] {
                assert_eq!(layered.render(n).unwrap(), plain.render(n).unwrap(), "interp {interp} blur {blur} frame {n}");
            }
        }
    }
}

#[test]
fn the_fg_matte_parameters_alone_change_nothing() {
    // matte channel and inverse matte shrink only matter with a matte layer
    let plain = rig(&[("warping", "1")]).render(1).unwrap();
    let with = rig(&[("warping", "1"), ("fg1_matte_chan", "4"), ("fg1_inv_matte_shrink", "3"), ("fg2_matte_chan", "2")]).render(1).unwrap();
    assert_eq!(plain, with);
}

#[test]
fn one_vector_layer_is_enough_the_other_flow_is_its_inversion() {
    let (prev, next) = motion_vectors(Depth::U8, W, H, 16, 3.4, 1.7, MD);
    let only = |layer: &str, clip: &Vec<fx_core::Frame>, extra: &[(&str, &str)]| {
        Rig::new(DEFAULT.clip(Depth::U8, W, H, 8), &base(extra))
            .with_frames(10)
            .with_layer(layer, clip.clone())
    };
    let both = rig(&[]);
    for (name, r) in [("Prev", only("vectors_prev", &prev, &[])), ("Next", only("vectors_next", &next, &[]))] {
        for n in [1u64, 3, 5] {
            let got = r.render(n).unwrap_or_else(|e| panic!("only Vectors: {name}, frame {n}: {e}"));
            let truth = DEFAULT.frame_at(Depth::U8, W, H, n as f64 / 2.0);
            let (e, _) = error(&got, &truth, 6);
            let two = truth_error(&both, n);
            // the inverted flow of a constant motion is the constant opposite motion: within a small margin of two layers
            assert!(e < 0.005 && e < two * 2.0 + 0.002, "only Vectors: {name}, frame {n}: error {e:.5} against {two:.5} with both");
        }
    }
    // Nearest at wA >= 0.5 (output frame 1 at 50 percent is position 0.5, wA = 0.5) warps frame A with the one flow it
    // reads: Vectors: Prev alone is enough there, and Next alone works too through the inversion
    for (layer, clip) in [("vectors_prev", &prev), ("vectors_next", &next)] {
        assert!(only(layer, clip, &[("frame_interp", "1")]).render(1).is_ok(), "{layer}");
    }
}

#[test]
fn the_warp_code_of_the_popup() {
    // 1 Inverse -> 0, 2 Forward -> 2, any other value keeps 0
    assert_eq!(glue::warp_code(1), 0);
    assert_eq!(glue::warp_code(2), 2);
    for other in [0, 3, 4, -1, 99] {
        assert_eq!(glue::warp_code(other), 0, "{other}");
    }
}

#[test]
fn nothing_unsupported_is_reported_for_a_clean_inverse_render() {
    // all three interpolation modes, both vector layers, the inverse warp
    for interp in ["1", "2", "3"] {
        assert!(rig(&[("warping", "1"), ("frame_interp", interp)]).render(1).is_ok(), "interp {interp}");
    }
}
