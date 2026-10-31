//! The retime of plugin 03 through the `Effect` trait (`render_temporal`), compared with literal frames from the
//! independent numpy float32 script `tests/common/gen_goldens.py` (the same script and clip as plugin 01's tests: the
//! pixel path behind the two front-ends is the same engine, so the same scenarios give the same frames; what differs
//! is the front-end, which these tests exercise with plugin 03's own parameters) at 8, 16 and 32 bpc.
//!
//! Parameter values are the ones a user types: popups are 1 based (`vectors=1` is No Vectors,
//! `frame_interp=1/2/3` is Nearest / Blend / Weighted Blend, `warping=1..4`), percentages are plain numbers.

mod common;

use common::*;
use fx_core::{Effect, Frame, FxError, TemporalContext};
use retime_fgbg::RetimeFgBg;

const NO_VECTORS: (&str, &str) = ("vectors", "1");
const NEAREST: (&str, &str) = ("frame_interp", "1");
const BLEND: (&str, &str) = ("frame_interp", "2");
const WEIGHTED: (&str, &str) = ("frame_interp", "3");

#[test]
fn defaults_play_the_clip_through_the_documented_lossy_round_trip() {
    // defaults: Speed mode, 100 percent, In FPS is Out FPS, Vectors Best, Blend, Warping Inverse: every position is on a frame
    for set in sets() {
        let rig = set.rig(&[]);
        for n in 0..4u64 {
            assert_eq!(rig.render(n).unwrap(), set.rt[n as usize], "{:?} frame {n}", set.depth);
        }
        // past the end of the clip the source is transparent
        assert_eq!(rig.render(4).unwrap(), set.transparent());
        assert_eq!(rig.render(9).unwrap(), set.transparent());
    }
}

#[test]
fn half_speed_without_vectors() {
    for set in sets() {
        for (interp, name) in [(NEAREST, "nearest"), (BLEND, "blend"), (WEIGHTED, "weighted blend")] {
            let rig = set.rig(&[NO_VECTORS, ("speed_percent", "50"), interp]);
            for n in 0..10u64 {
                let k = (n / 2) as usize;
                let want = if n % 2 == 0 || interp == NEAREST { set.rt_or_clear(k) } else { set.half(k) };
                assert_eq!(rig.render(n).unwrap(), want, "{:?} {name} frame {n}", set.depth);
            }
        }
    }
}

#[test]
fn quarter_speed_uses_the_fraction_as_the_weight() {
    for set in sets() {
        let at = |interp, n| set.rig(&[NO_VECTORS, ("speed_percent", "25"), interp]).render(n).unwrap();
        assert_eq!((at(NEAREST, 1), at(NEAREST, 2), at(NEAREST, 3), at(NEAREST, 4)), (set.rt[0].clone(), set.rt[0].clone(), set.rt[1].clone(), set.rt[1].clone()));
        for interp in [BLEND, WEIGHTED] {
            assert_eq!(at(interp, 1), set.b01_25, "{:?}", set.depth);
            assert_eq!(at(interp, 2), set.b01);
            assert_eq!(at(interp, 3), set.b01_75);
            assert_eq!(at(interp, 4), set.rt[1]);
        }
    }
}

#[test]
fn frame_number_mode() {
    for set in sets() {
        // Frame Num 2.5 shows the middle of frames 2 and 3 at every output frame
        for n in [0, 1, 7] {
            let blend = set.rig(&[NO_VECTORS, ("time_remap_mode", "2"), ("frame_num", "2.5"), BLEND]).render(n).unwrap();
            assert_eq!(blend, set.b23, "{:?} frame {n}", set.depth);
            let near = set.rig(&[NO_VECTORS, ("time_remap_mode", "2"), ("frame_num", "2.5"), NEAREST]).render(n).unwrap();
            assert_eq!(near, set.rt[2]);
            // an exact frame number works with Vectors on (the default Best)
            let exact = set.rig(&[("time_remap_mode", "2"), ("frame_num", "1")]).render(n).unwrap();
            assert_eq!(exact, set.rt[1]);
            // a frame number beyond the clip is transparent
            let beyond = set.rig(&[("time_remap_mode", "2"), ("frame_num", "6")]).render(n).unwrap();
            assert_eq!(beyond, set.transparent());
        }
        // an animated frame number: frame 3 at output frame 0 and frame 0 at output frame 3 (linear keys)
        let walk = set.rig(&[("time_remap_mode", "2"), ("frame_num", "3@0;0@3")]);
        let got: Vec<_> = (0..4).map(|n| walk.render(n).unwrap()).collect();
        assert_eq!(got, vec![set.rt[3].clone(), set.rt[2].clone(), set.rt[1].clone(), set.rt[0].clone()], "{:?}", set.depth);
    }
}

#[test]
fn reverse_playback_starts_at_frame_zero_and_runs_off_the_clip() {
    // Plugin 03 has no Src Time Offset, so Speed -100 percent shows frame 0 and then positions before the clip
    for set in sets() {
        let rig = set.rig(&[("speed_percent", "-100")]);
        let got: Vec<_> = (0..4).map(|n| rig.render(n).unwrap()).collect();
        assert_eq!(got, vec![set.rt[0].clone(), set.transparent(), set.transparent(), set.transparent()], "{:?}", set.depth);
        // the same positions in Frame Number mode, reading frame numbers from a decreasing curve
        let rig = set.rig(&[("time_remap_mode", "2"), ("frame_num", "3@0;-1@4")]);
        let got: Vec<_> = (0..5).map(|n| rig.render(n).unwrap()).collect();
        assert_eq!(got, vec![set.rt[3].clone(), set.rt[2].clone(), set.rt[1].clone(), set.rt[0].clone(), set.transparent()], "{:?}", set.depth);
    }
}

#[test]
fn display_source_shows_alt_color_source_at_the_current_time_without_retiming() {
    for set in sets() {
        // no offset parameter: output frame n shows frame n of Alt Color Source whatever the speed is
        let rig = set.rig(&[("display", "2"), ("speed_percent", "50")]).with_layer("alt_source_a", set.clip.clone());
        let got: Vec<_> = (0..5).map(|n| rig.render(n).unwrap()).collect();
        assert_eq!(got, vec![set.rt[0].clone(), set.rt[1].clone(), set.rt[2].clone(), set.rt[3].clone(), set.transparent()], "{:?}", set.depth);
        // the Draw Geom overlay with no tracking point in use draws nothing, at any setting
        for geom in ["1", "2", "3", "6"] {
            let rig = set.rig(&[("display", "2"), ("draw_geom", geom)]).with_layer("alt_source_a", set.clip.clone());
            assert_eq!(rig.render(1).unwrap(), set.rt[1], "{:?} draw_geom {geom}", set.depth);
        }
        // without Alt Color Source the Source display is the missing layer fill, never the input layer
        let (w, h) = (set.rt[0].width, set.rt[0].height);
        let rig = set.rig(&[("display", "2"), ("speed_percent", "50"), ("pt1_use", "5"), ("pt1_pos", "1,1")]);
        for n in 0..5 {
            assert_eq!(rig.render(n).unwrap(), fill(set.depth, false, w, h), "{:?} frame {n}", set.depth);
        }
    }
}

#[test]
fn the_fg_and_background_displays_render_like_the_output_when_there_is_one_layer() {
    // Display 3 to 6 select the group min(Display - 3, n) with n = 0 FG layers: the single group 0,
    // the range 0..=0 of Display 1
    for set in sets() {
        let base = [NO_VECTORS, ("speed_percent", "50"), BLEND];
        let want: Vec<Frame> = (0..4).map(|n| set.rig(&base).render(n).unwrap()).collect();
        for display in ["3", "4", "5", "6"] {
            let mut params = base.to_vec();
            params.push(("display", display));
            let rig = set.rig(&params);
            let got: Vec<Frame> = (0..4).map(|n| rig.render(n).unwrap()).collect();
            assert_eq!(got, want, "{:?} display {display}", set.depth);
        }
    }
}

#[test]
fn the_warping_popup_does_not_matter_where_no_vectors_are_computed() {
    // exact positions and No Vectors never reach a warp: all four codes give the frame blend
    for set in sets() {
        for warping in ["1", "2", "3", "4"] {
            let exact = set.rig(&[("warping", warping)]);
            assert_eq!(exact.render(2).unwrap(), set.rt[2], "{:?} warping {warping}", set.depth);
            let plain = set.rig(&[NO_VECTORS, ("speed_percent", "50"), BLEND, ("warping", warping)]);
            assert_eq!(plain.render(1).unwrap(), set.b01, "{:?} warping {warping}", set.depth);
        }
    }
}

/// Output frame `n` of half speed (n odd: between source frames `(n - 1) / 2` and `(n + 1) / 2`)
/// computed from explicit inputs: the two source frames of the clip, the selection of the half
/// speed position with the vectors quality of Best (4), `solve_flows` and `compose_vectors`. Only
/// the warping code, interpolation, sensitivity and render quality come from the settings the
/// front-end reads; its vectors quality and the routing of the core are not used, so a render that
/// falls back to the frame blend does not match it.
fn engine_frame(rig: &Rig, n: u64) -> Frame {
    use retime_engine::compose::{compose_vectors, solve_flows, Rendered};
    use retime_engine::image::Image;
    use retime_engine::timeline::Selection;
    let cx = rig.cx(n);
    let env = rig.env();
    let settings = retime_fgbg::glue::read_settings(&cx, &env).unwrap();
    let k = ((n - 1) / 2) as usize;
    // the vector path runs on bottom-up rows (`EngineSettings::motion_rows_reversed`)
    assert!(settings.motion_rows_reversed);
    let frame = |i: usize| Image::from_frame(&*fx_core::Layer::frame_at(&rig.input, (i as i64) * 100).unwrap().expect("frame")).rows_reversed();
    let (a, b) = (frame(k), frame(k + 1));
    let pos = n as f32 / 2.0;
    let sel = Selection { s0: pos, s1: pos + 0.5, a: k as i32, b: k as i32 + 1, wa: 0.5, exact: false, vectors_quality: 4, interp: settings.interp };
    let flows = solve_flows(&settings, &sel, &a, &b).unwrap();
    Rendered::Image(compose_vectors(&settings, &sel, &a, &b, &flows).unwrap().rows_reversed()).into_frame()
}

#[test]
fn a_position_between_frames_with_vectors_on_runs_the_vector_path_on_tiny_frames() {
    // Frames of 3 x 2 and 2 x 2 pixels of a texture that moves by a sub-pixel amount. The motion search never halves
    // below 2 x 2 (the tiny frames rule, parity gap G7), so Best searches them at their own size: a position between
    // frames takes the vector path with every warping code. Its output is the solve and the composition computed
    // from explicit inputs, and it differs from the No Vectors render of the same clip, so a silent fall back to the
    // frame blend fails here. Real motion is in `vector.rs`.
    for depth in [fx_core::Depth::U8, fx_core::Depth::U16, fx_core::Depth::F32] {
        for (w, h) in [(3usize, 2usize), (2, 2)] {
            let clip = common::moving::tiny_texture_clip(depth, w, h, 4);
            for warping in ["1", "2", "3", "4"] {
                for interp in ["1", "2", "3"] {
                    let params = [("speed_percent", "50"), ("main_bg_sensitivity", "100"), ("warping", warping), ("frame_interp", interp)];
                    let rig = Rig::new(clip.clone(), &params).with_frames(6);
                    let mut no_vectors = params.to_vec();
                    no_vectors.push(("vectors", "1"));
                    let plain = Rig::new(clip.clone(), &no_vectors).with_frames(6);
                    for n in [1u64, 3, 5] {
                        let case = format!("{depth:?} {w}x{h} warping {warping} interp {interp} frame {n}");
                        let got = rig.render(n).unwrap_or_else(|e| panic!("{case}: {e}"));
                        assert_eq!(got, engine_frame(&rig, n), "{case}");
                        assert_ne!(got, plain.render(n).unwrap(), "{case}: the frame blend");
                    }
                    // the even frames are on frames: the fast path, never the vector path
                    assert_eq!(rig.render(2).unwrap(), plain.render(2).unwrap());
                }
            }
        }
    }
}

#[test]
fn mismatched_sizes_give_the_red_fill_and_a_missing_input_the_blue_one_with_plugin_03s_alpha() {
    for set in sets() {
        let (w, h) = (set.width, set.height);
        // the host asks for a wider output than the clip has
        let rig = set.rig(&[]);
        let mut cx = rig.cx(1);
        cx.width += 1;
        assert_eq!(rig.render_cx(&cx).unwrap(), fill(set.depth, true, w + 1, h), "{:?}", set.depth);
        // Alt Source A of another size replaces the input and does not fit
        let wide = vec![Frame::new(w + 1, h, set.depth)];
        let rig = set.rig(&[]).with_layer("alt_source_a", wide.clone());
        assert_eq!(rig.render(0).unwrap(), fill(set.depth, true, w, h));
        // a tracking layer of another size is flagged as well
        let rig = set.rig(&[]).with_layer("alt_source_b", wide.clone());
        assert_eq!(rig.render(0).unwrap(), fill(set.depth, true, w, h));
        // a FG matte of another size sets the same flag (the size probes of the pre-render), even
        // though a FG layer is otherwise not rendered
        let rig = set.rig(&[]).with_layer("fg1_matte", wide.clone());
        assert_eq!(rig.render(0).unwrap(), fill(set.depth, true, w, h));
        let rig = set.rig(&[("display", "2")]).with_layer("fg2_matte", set.clip.clone()).with_layer("fg1_matte", wide);
        assert_eq!(rig.render(0).unwrap(), fill(set.depth, true, w, h));
        // a zero size input is a missing layer: blue, with alpha 0 at 16 and 32 bpc
        let empty = Rig::new(vec![Frame::new(0, 0, set.depth)], &[]);
        let mut cx = empty.cx(0);
        cx.width = w;
        cx.height = h;
        assert_eq!(empty.render_cx(&cx).unwrap(), fill(set.depth, false, w, h));
        // red wins over blue, and an error fill wins over every unsupported feature
        let both = Rig::new(vec![Frame::new(0, 0, set.depth)], &[("warping", "3"), ("speed_percent", "50")]).with_layer("fg1_matte", vec![Frame::new(w + 1, h, set.depth)]);
        let mut cx = both.cx(1);
        cx.width = w;
        cx.height = h;
        assert_eq!(both.render_cx(&cx).unwrap(), fill(set.depth, true, w, h));
    }
}

#[test]
fn the_missing_layer_fill_is_the_documented_one_of_plugin_03() {
    // missing layer fill: blue at half alpha at 8 bpc, blue with alpha 0 at 16 and 32 bpc
    for (depth, want) in [
        (fx_core::Depth::U8, Frame { width: 1, height: 1, pixels: fx_core::Pixels::U8(vec![[0, 0, 255, 128]]) }),
        (fx_core::Depth::U16, Frame { width: 1, height: 1, pixels: fx_core::Pixels::U16(vec![[0, 0, 65535, 0]]) }),
        (fx_core::Depth::F32, Frame { width: 1, height: 1, pixels: fx_core::Pixels::F32(vec![[0.0, 0.0, 1.0, 0.0]]) }),
    ] {
        let rig = Rig::new(vec![Frame::new(0, 0, depth)], &[]);
        let mut cx = rig.cx(0);
        cx.width = 1;
        cx.height = 1;
        assert_eq!(rig.render_cx(&cx).unwrap(), want, "{depth:?}");
    }
}

#[test]
fn alt_source_a_is_retimed_instead_of_the_input_and_an_empty_one_is_ignored() {
    let sets = sets();
    for set in &sets {
        // the input holds the clip backwards, Alt Source A the clip forwards
        let backwards: Vec<_> = set.clip.iter().rev().cloned().collect();
        let rig = Rig::new(backwards.clone(), &[]).with_layer("alt_source_a", set.clip.clone());
        for n in 0..4u64 {
            assert_eq!(rig.render(n).unwrap(), set.rt[n as usize], "{:?} frame {n}", set.depth);
        }
        // a zero size Alt Source A is "not set": the input is used
        let rig = Rig::new(backwards, &[]).with_layer("alt_source_a", vec![Frame::new(0, 0, set.depth)]);
        assert_eq!(rig.render(0).unwrap(), set.rt[3]);
        // Alt Source B of the right size does not change a fast path render
        let rig = set.rig(&[NO_VECTORS, ("speed_percent", "50"), BLEND]).with_layer("alt_source_b", set.clip.clone());
        assert_eq!(rig.render(1).unwrap(), set.b01);
    }
}

#[test]
fn fg_matte_layers_do_not_change_a_source_display() {
    // the Source display copies the source frame and does not take the multi layer path
    for set in sets() {
        let rig = set
            .rig(&[("display", "2")])
            .with_layer("alt_source_a", set.clip.clone())
            .with_layer("fg1_matte", set.clip.clone())
            .with_layer("fg2_matte", set.clip.clone());
        assert_eq!(rig.render(2).unwrap(), set.rt[2], "{:?}", set.depth);
    }
}

#[test]
fn a_run_longer_than_the_clip_keeps_producing_frames() {
    for set in sets() {
        let rig = set.rig(&[NO_VECTORS, ("speed_percent", "50")]).with_frames(12);
        assert_eq!(rig.render(7).unwrap(), set.b3t, "{:?}", set.depth);
        assert_eq!(rig.render(11).unwrap(), set.transparent());
    }
}

#[test]
fn the_output_frame_is_replaced_whatever_it_held() {
    for set in sets() {
        let rig = set.rig(&[]);
        let cx = rig.cx(0);
        let mut dirty = fill(set.depth, true, set.width, set.height);
        RetimeFgBg.render_temporal(&cx, &rig.env(), &mut dirty).unwrap();
        assert_eq!(dirty, set.rt[0]);
    }
}

#[test]
fn a_context_that_disagrees_with_the_output_frame_is_rejected() {
    let set = &sets()[0];
    let rig = set.rig(&[]);
    let cx: TemporalContext = rig.cx(0);
    let mut small = Frame::new(1, 1, set.depth);
    let r = RetimeFgBg.render_temporal(&cx, &rig.env(), &mut small);
    assert!(matches!(r, Err(FxError::Mismatch(_))));
}

#[test]
fn an_empty_composition_renders_without_panicking() {
    for depth in [fx_core::Depth::U8, fx_core::Depth::U16, fx_core::Depth::F32] {
        let rig = Rig::new(vec![Frame::new(0, 0, depth)], &[]);
        let out = rig.render(0).unwrap();
        assert_eq!((out.width, out.height, out.depth()), (0, 0, depth));
    }
}

#[test]
fn unequal_rates_fetch_at_plugin_03s_layer_times() {
    // Input 3 fps in a 30 fps composition (100 ticks per frame): f58 = 1000 ticks per source frame. Output frame 15 is the
    // source position 15 * 3 / 30 = 1.5 (a = 1, b = 2, weight 0.5), whose layer times are T(1) = 1000 and T(2) = 2000, i.e.
    // clip frames 10 and 20 (a `MemLayer` frame covers 100 ticks). Plugin 01's frame to time function would add
    // `dur * K` = 100 ticks and read frames 11 and 21. Frame i of the clip has red 8 i, so the blend is (80 + 160) / 2.
    use fx_core::Pixels;
    let clip: Vec<Frame> = (0..30u8).map(|i| Frame { width: 1, height: 1, pixels: Pixels::U8(vec![[i * 8, 0, 0, 255]]) }).collect();
    let rig = Rig::new(clip, &[NO_VECTORS, ("in_fps_is_out_fps", "0"), ("input_frame_rate", "3"), BLEND]).with_frames(20);
    assert_eq!(rig.render(15).unwrap(), f8(1, 1, &[[120, 0, 0, 255]]));
    // an exact position: output frame 10 is source position 1.0, clip frame 10
    assert_eq!(rig.render(10).unwrap(), f8(1, 1, &[[80, 0, 0, 255]]));
}
