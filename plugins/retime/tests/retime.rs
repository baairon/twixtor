//! The retime of plugin 01 through the `Effect` trait (`render_temporal`), compared with literal
//! frames from an independent numpy float32 script (`tests/common/gen_goldens.py`) at 8, 16 and 32 bpc.
//!
//! Parameter values are the ones a user types: popups are 1 based (`vectors=1` is No Vectors,
//! `frame_interp=1/2/3` is Nearest / Blend / Weighted Blend), percentages are plain numbers.

mod common;

use common::*;
use fx_core::{FxError, TemporalContext};

const NO_VECTORS: (&str, &str) = ("vectors", "1");
const NEAREST: (&str, &str) = ("frame_interp", "1");
const BLEND: (&str, &str) = ("frame_interp", "2");
const WEIGHTED: (&str, &str) = ("frame_interp", "3");

#[test]
fn defaults_play_the_clip_through_the_documented_lossy_round_trip() {
    // defaults: Speed mode, 100 percent, In FPS is Out FPS, Vectors Best, Weighted Blend: every position is on a frame
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
        // Nearest takes A while the weight of A is at least one half
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
        // Frame Num 2.5 shows the middle of frames 2 and 3 at every output frame; the default offset is ignored
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
fn reverse_playback_with_a_start_offset() {
    for set in sets() {
        let rig = set.rig(&[("speed_percent", "-100"), ("src_time_offset", "3")]);
        let got: Vec<_> = (0..5).map(|n| rig.render(n).unwrap()).collect();
        let want = vec![set.rt[3].clone(), set.rt[2].clone(), set.rt[1].clone(), set.rt[0].clone(), set.transparent()];
        assert_eq!(got, want, "{:?}", set.depth);
    }
}

#[test]
fn src_time_offset_shifts_the_clip_ahead() {
    for set in sets() {
        let rig = set.rig(&[("src_time_offset", "2")]);
        assert_eq!((rig.render(0).unwrap(), rig.render(1).unwrap(), rig.render(2).unwrap()), (set.rt[2].clone(), set.rt[3].clone(), set.transparent()));
    }
}

#[test]
fn negative_offsets_are_not_a_parameter_value() {
    // The slider is valid from 0; delaying a clip is tested on the engine with a negative reading.
    let mut c = curves(&[]);
    assert!(matches!(c.parse_cli("src_time_offset", "-2", time()), Err(FxError::BadParam { .. })));
}

#[test]
fn display_source_shows_the_frame_at_the_offset_without_retiming() {
    for set in sets() {
        let rig = set.rig(&[("display", "2"), ("speed_percent", "50"), ("src_time_offset", "2")]);
        assert_eq!((rig.render(0).unwrap(), rig.render(1).unwrap(), rig.render(2).unwrap()), (set.rt[2].clone(), set.rt[3].clone(), set.transparent()), "{:?}", set.depth);
        // Display == Source needs neither vectors nor an exact position
        let plain = set.rig(&[("display", "2")]);
        assert_eq!(plain.render(1).unwrap(), set.rt[1]);
    }
}

#[test]
fn a_position_between_frames_with_vectors_on_runs_the_solver_and_tiny_frames_are_refused_by_name() {
    // The golden frames are 3 x 2 and 2 x 2 pixels: below the 2 x 2 working size the solver needs at Best
    // (half resolution). The position between frames takes the vector path (it does not fall back to the
    // frame blend) and the solver's message says why it cannot run. Real motion is in `vector.rs`.
    for set in sets() {
        let rig = set.rig(&[("speed_percent", "50")]);
        for n in [1u64, 3, 5] {
            match rig.render(n) {
                Err(FxError::Unsupported(m)) => assert!(m.contains("below 2x2"), "{m}"),
                other => panic!("frame {n}: expected Unsupported, got {other:?}"),
            }
        }
        // the even frames of the same run are on frames and render
        assert_eq!(rig.render(2).unwrap(), set.rt[1]);
        let msg = rig.render(1).unwrap_err().to_string();
        assert!(msg.starts_with("unsupported: optical flow: working size"), "{msg}");
        // Smart Blend runs the same solver: the same refusal for tiny frames, and exact frames never reach the vector path
        let sb = set.rig(&[("speed_percent", "50"), ("smart_blend", "1")]);
        match sb.render(1) {
            Err(FxError::Unsupported(m)) => assert!(m.contains("below 2x2"), "{m}"),
            other => panic!("expected the solver's Unsupported, got {other:?}"),
        }
        assert_eq!(sb.render(2).unwrap(), set.rt[1], "an exact frame never reaches the vector path");
    }
}

#[test]
fn mismatched_sizes_give_the_red_fill_and_a_missing_input_the_blue_one() {
    for set in sets() {
        let (w, h) = (set.width, set.height);
        // the host asks for a wider output than the clip has
        let rig = set.rig(&[]);
        let mut cx = rig.cx(1);
        cx.width += 1;
        assert_eq!(rig.render_cx(&cx).unwrap(), fill(set.depth, true, w + 1, h), "{:?}", set.depth);
        // Alt Source A of another size replaces the input and does not fit
        let wide = vec![fx_core::Frame::new(w + 1, h, set.depth)];
        let rig = set.rig(&[]).with_layer("alt_source_a", wide.clone());
        assert_eq!(rig.render(0).unwrap(), fill(set.depth, true, w, h));
        // a tracking layer of another size is flagged as well
        let rig = set.rig(&[]).with_layer("alt_source_b", wide);
        assert_eq!(rig.render(0).unwrap(), fill(set.depth, true, w, h));
        // a zero size input is a missing layer
        let empty = common::Rig::new(vec![fx_core::Frame::new(0, 0, set.depth)], &[]);
        let mut cx = empty.cx(0);
        cx.width = w;
        cx.height = h;
        assert_eq!(empty.render_cx(&cx).unwrap(), fill(set.depth, false, w, h));
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
        let rig = Rig::new(backwards, &[]).with_layer("alt_source_a", vec![fx_core::Frame::new(0, 0, set.depth)]);
        assert_eq!(rig.render(0).unwrap(), set.rt[3]);
        // Alt Source B of the right size does not change a fast path render
        let rig = set.rig(&[NO_VECTORS, ("speed_percent", "50"), BLEND]).with_layer("alt_source_b", set.clip.clone());
        assert_eq!(rig.render(1).unwrap(), set.b01);
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
    // a host may pass a dirty frame: the render overwrites it entirely
    for set in sets() {
        let rig = set.rig(&[]);
        let cx = rig.cx(0);
        let mut dirty = fill(set.depth, true, set.width, set.height);
        use fx_core::Effect;
        retime::Retime.render_temporal(&cx, &rig.env(), &mut dirty).unwrap();
        assert_eq!(dirty, set.rt[0]);
    }
}

#[test]
fn a_context_that_disagrees_with_the_output_frame_is_rejected() {
    use fx_core::Effect;
    let set = &sets()[0];
    let rig = set.rig(&[]);
    let cx: TemporalContext = rig.cx(0);
    let mut small = fx_core::Frame::new(1, 1, set.depth);
    let r = retime::Retime.render_temporal(&cx, &rig.env(), &mut small);
    assert!(matches!(r, Err(FxError::Mismatch(_))));
}

#[test]
fn an_empty_composition_renders_without_panicking() {
    // a 0 x 0 frame has no rows to split among workers; the input is "missing" and the fill is empty
    for depth in [fx_core::Depth::U8, fx_core::Depth::U16, fx_core::Depth::F32] {
        let rig = common::Rig::new(vec![fx_core::Frame::new(0, 0, depth)], &[]);
        let out = rig.render(0).unwrap();
        assert_eq!((out.width, out.height, out.depth()), (0, 0, depth));
    }
}
