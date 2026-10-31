//! The Source display (a copy of the source layer with the premultiply round trip of the input
//! prepare and the output finish), `Alt Motion Src` as the source, the error fills (red at half alpha for
//! a size mismatch, the GREY image for a missing layer) and the frames outside the clip.

mod common;

use common::*;
use fx_core::{Depth, Effect, Env, Frame, FxError, LayerSet, MemLayer, Pixels, TemporalContext};
use create_vectors::CreateVectors;
use retime_engine::image::{error_frame, ErrorKind, Image, MissingLayerStyle};
use std::sync::Arc;

const ALT: &str = "alt_source_a";

/// What a frame becomes through the engine: the input prepare (premultiply) and the output finish (unpremultiply).
fn round_trip(f: &Frame) -> Frame {
    Image::from_frame(f).to_frame()
}

#[test]
fn the_source_display_is_the_source_through_the_premultiply_round_trip() {
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        // random pixels of every alpha class: the premultiply round trip is lossy, and the display
        // shows its result
        let clip = random_clip(depth, 7, 5, 6, 11);
        let rig = Rig::new(clip.clone(), &[("display", "1")]).with_frames(6);
        for n in 0..6 {
            assert_eq!(rig.render(n as u64).unwrap(), round_trip(&clip[n]), "{depth:?} frame {n}");
        }
        // the vector parameters do not matter
        let other = Rig::new(clip.clone(), &[("display", "1"), ("max_displace", "5"), ("track_frame", "2"), ("vector_resolution", "2"), ("prep", "3")]).with_frames(6);
        assert_eq!(other.render(3).unwrap(), round_trip(&clip[3]));
    }
}

#[test]
fn the_source_display_reads_the_source_at_the_current_time() {
    // current_time 730 is inside frame 7: the clip frame at tick 730 is frame 7, whatever frame index the host says
    let clip = random_clip(Depth::U8, 4, 3, 10, 2);
    let rig = Rig::new(clip.clone(), &[("display", "1")]).with_frames(10);
    let mut cx = rig.cx(7);
    cx.current_time = 730;
    assert_eq!(rig.render_cx(&cx).unwrap(), round_trip(&clip[7]));
}

#[test]
fn alt_src_is_the_source_when_it_has_pixels() {
    let input = random_clip(Depth::U8, 6, 4, 4, 1);
    let alt = random_clip(Depth::U8, 6, 4, 4, 2);
    let rig = Rig::new(input.clone(), &[("display", "1")]).with_layer(ALT, alt.clone()).with_frames(4);
    assert_eq!(rig.render(2).unwrap(), round_trip(&alt[2]));
    // an Alt Motion Src of size 0 is no layer at all: the input is the source
    let empty = vec![Frame::new(0, 0, Depth::U8)];
    let rig = Rig::new(input.clone(), &[("display", "1")]).with_layer(ALT, empty).with_frames(4);
    assert_eq!(rig.render(2).unwrap(), round_trip(&input[2]));
    // the vector display reads the pair from Alt Motion Src: the same clip moved by a known amount gives vectors, the input is ignored
    let moving = DEFAULT_MOTION.clip(Depth::U8, 64, 48, 6);
    let noise = random_clip(Depth::U8, 64, 48, 6, 3);
    let via_alt = Rig::new(noise, &[("max_displace", "16")]).with_layer(ALT, moving.clone()).with_frames(6);
    let direct = Rig::new(moving, &[("max_displace", "16")]).with_frames(6);
    assert_eq!(via_alt.render(3).unwrap(), direct.render(3).unwrap());
}

const DEFAULT_MOTION: common::moving::Motion = common::moving::DEFAULT;

fn px8(f: &Frame) -> &[[u8; 4]] {
    match &f.pixels {
        Pixels::U8(p) => p,
        _ => panic!("not 8 bpc"),
    }
}

#[test]
fn a_source_of_another_size_is_the_red_error_image() {
    let input = random_clip(Depth::U8, 6, 4, 4, 1);
    let alt = random_clip(Depth::U8, 5, 4, 4, 2);
    for display in ["1", "2"] {
        let rig = Rig::new(input.clone(), &[("display", display)]).with_layer(ALT, alt.clone()).with_frames(4);
        let out = rig.render(2).unwrap();
        assert_eq!(out, error_frame(ErrorKind::SizeMismatch, 6, 4, Depth::U8));
        // 8 bpc (A, R, G, B) = (0x80, 0xff, 0, 0): wire RGBA (255, 0, 0, 128)
        assert!(px8(&out).iter().all(|p| *p == [255, 0, 0, 128]));
    }
    // 16 bpc: (0x4000, 0x8000, 0, 0) in the range 0..=32768, 32 bpc (0.5, 1.0, 0, 0)
    for depth in [Depth::U16, Depth::F32] {
        let input = random_clip(depth, 6, 4, 4, 1);
        let alt = random_clip(depth, 5, 4, 4, 2);
        let out = Rig::new(input, &[]).with_layer(ALT, alt).with_frames(4).render(2).unwrap();
        assert_eq!(out, error_frame(ErrorKind::SizeMismatch, 6, 4, depth));
        match &out.pixels {
            Pixels::U16(p) => assert!(p.iter().all(|px| *px == [65535, 0, 0, Frame::q15_to_u16(0x4000)])),
            Pixels::F32(p) => assert!(p.iter().all(|px| *px == [1.0, 0.0, 0.0, 0.5])),
            _ => unreachable!(),
        }
    }
}

/// A render of an empty (0 x 0) input layer into a 6 x 4 output, with extra layers.
fn render_empty_input(depth: Depth, layers: &LayerSet) -> Frame {
    let empty = MemLayer::new(time(), vec![Frame::new(0, 0, depth)]).unwrap();
    let curves = curves(&[]);
    let env = Env { params: &curves, input: &empty, layers };
    let cx = TemporalContext::new(6, 4, depth, time(), 1, 4);
    let mut out = Frame::new(6, 4, depth);
    CreateVectors.render_temporal(&cx, &env, &mut out).unwrap();
    out
}

#[test]
fn a_missing_source_layer_is_the_grey_error_image() {
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let out = render_empty_input(depth, &LayerSet::new());
        assert_eq!(out, error_frame(ErrorKind::MissingLayer(MissingLayerStyle::Grey), 6, 4, depth));
        match &out.pixels {
            // 8 bpc (A, R, G, B) = (0xff, 0x7f, 0x7f, 0x7f)
            Pixels::U8(p) => assert!(p.iter().all(|px| *px == [0x7f, 0x7f, 0x7f, 0xff])),
            // 16 bpc (1, 0x4000, 0x4000, 0x4000): the alpha is 1 on the 0..=32768 scale
            Pixels::U16(p) => {
                let g = Frame::q15_to_u16(0x4000);
                assert!(p.iter().all(|px| *px == [g, g, g, Frame::q15_to_u16(1)]));
            }
            // 32 bpc: grey 0.5, and the alpha slot is 0.0
            Pixels::F32(p) => assert!(p.iter().all(|px| *px == [0.5, 0.5, 0.5, 0.0])),
        }
    }
}

#[test]
fn red_is_tested_before_grey() {
    // no source (grey) and a FG matte of another size (red): the size mismatch flag wins
    let mut layers = LayerSet::new();
    layers.insert("fg1_matte", Arc::new(MemLayer::new(time(), vec![Frame::new(3, 3, Depth::U8)]).unwrap()));
    let out = render_empty_input(Depth::U8, &layers);
    assert_eq!(out, error_frame(ErrorKind::SizeMismatch, 6, 4, Depth::U8));
    // an empty Alt Motion Src falls back to the (empty) input: still grey
    let mut layers = LayerSet::new();
    layers.insert(ALT, Arc::new(MemLayer::new(time(), vec![Frame::new(0, 0, Depth::U8)]).unwrap()));
    assert_eq!(render_empty_input(Depth::U8, &layers), error_frame(ErrorKind::MissingLayer(MissingLayerStyle::Grey), 6, 4, Depth::U8));
}

#[test]
fn frames_outside_the_clip_are_transparent_and_give_a_valid_vector_image() {
    // output frame 9 of an 8 frame clip: both frames of the pair are past the end (transparent), the flow is zero
    let clip = DEFAULT_MOTION.clip(Depth::U8, 32, 24, 8);
    let rig = Rig::new(clip, &[("max_displace", "16")]).with_frames(12);
    let out = rig.render(10).unwrap();
    assert!(px8(&out).iter().all(|p| *p == [127, 127, 0, 255]), "a zero vector everywhere, blue 0, opaque");
    // the Source display of a frame past the end is the transparent image
    let rig = Rig::new(DEFAULT_MOTION.clip(Depth::U8, 32, 24, 8), &[("display", "1")]).with_frames(12);
    assert!(px8(&rig.render(10).unwrap()).iter().all(|p| *p == [0, 0, 0, 0]));
}

#[test]
fn the_output_frame_must_match_the_context() {
    let clip = random_clip(Depth::U8, 6, 4, 3, 1);
    let rig = Rig::new(clip, &[]);
    let cx = rig.cx(1);
    let mut wrong = Frame::new(5, 4, Depth::U8);
    assert!(matches!(CreateVectors.render_temporal(&cx, &rig.env(), &mut wrong), Err(FxError::Mismatch(_))));
    let mut wrong = Frame::new(6, 4, Depth::U16);
    assert!(matches!(CreateVectors.render_temporal(&cx, &rig.env(), &mut wrong), Err(FxError::Mismatch(_))));
    // a dirty output of the right shape is simply replaced
    let mut dirty = Frame::new(6, 4, Depth::U8);
    if let Pixels::U8(p) = &mut dirty.pixels {
        p.fill([9, 9, 9, 9]);
    }
    CreateVectors.render_temporal(&cx, &rig.env(), &mut dirty).unwrap();
    assert_eq!(dirty, rig.render(1).unwrap());
}

#[test]
fn a_layer_of_another_depth_is_a_configuration_error() {
    let input = random_clip(Depth::U8, 6, 4, 3, 1);
    let alt = random_clip(Depth::U16, 6, 4, 3, 1);
    let rig = Rig::new(input, &[]).with_layer(ALT, alt);
    assert!(matches!(rig.render(1), Err(FxError::Mismatch(_))));
}
