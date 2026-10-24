//! Goldens of the vector encoding Create Vectors uses (`vectors::encode_vectors_image`), worked out
//! by hand, for known flow fields at the three depths: the centred code (127, 32767 or 0.5 for a
//! zero vector), the scale by Max Displace, the clamp of the integer depths, no clamp for float,
//! blue 0, alpha white, and the resampling of a working resolution flow to the output size (one
//! working pixel of displacement is more than one output pixel). Also the decode inverse
//! (`vectors::decode_vectors`) and the quantisation bound.

use retime_engine::flow::FlowField;
use retime_engine::image::{Fmt, Image, Samples};
use retime_engine::vectors::{decode_vectors, encode_vectors_image};

fn constant(w: usize, h: usize, v: [f32; 2]) -> FlowField {
    let mut f = FlowField::new(w, h);
    for y in 0..h {
        for x in 0..w {
            f.set(x, y, v);
        }
    }
    f
}

/// `[A, R, G, B]` of the pixel (x, y), as f64 of the stored values.
fn argb(img: &Image, x: usize, y: usize) -> [f64; 4] {
    let i = (y * img.width + x) * 4;
    match &img.samples {
        Samples::U8(v) => std::array::from_fn(|k| v[i + k] as f64),
        Samples::U16(v) => std::array::from_fn(|k| v[i + k] as f64),
        Samples::F32(v) => std::array::from_fn(|k| v[i + k] as f64),
    }
}

fn all_pixels(img: &Image, want: [f64; 4]) {
    for y in 0..img.height {
        for x in 0..img.width {
            assert_eq!(argb(img, x, y), want, "pixel ({x}, {y})");
        }
    }
}

#[test]
fn a_constant_flow_encodes_to_the_hand_computed_codes() {
    // flow (1, -2), scale 4: fx = 0.25, fy = -0.5
    let flow = constant(4, 3, [1.0, -2.0]);
    // 8 bpc: R = trunc((0.25 + 1) * 127 + 0.5) = trunc(159.25) = 159, G = trunc(0.5 * 127 + 0.5) = 64
    all_pixels(&encode_vectors_image(4, 3, Fmt::U8, &flow, 4.0).unwrap(), [255.0, 159.0, 64.0, 0.0]);
    // 16 bpc: R = trunc(1.25 * 32767 + 0.5) = trunc(40959.25) = 40959, G = trunc(0.5 * 32767 + 0.5) = 16384
    all_pixels(&encode_vectors_image(4, 3, Fmt::U16, &flow, 4.0).unwrap(), [65535.0, 40959.0, 16384.0, 0.0]);
    // float: R = (0.25 + 1) * 0.5 = 0.625, G = (-0.5 + 1) * 0.5 = 0.25
    all_pixels(&encode_vectors_image(4, 3, Fmt::F32, &flow, 4.0).unwrap(), [1.0, 0.625, 0.25, 0.0]);
}

#[test]
fn the_zero_vector_is_the_centre_code() {
    let flow = constant(5, 4, [0.0, 0.0]);
    all_pixels(&encode_vectors_image(5, 4, Fmt::U8, &flow, 100.0).unwrap(), [255.0, 127.0, 127.0, 0.0]);
    all_pixels(&encode_vectors_image(5, 4, Fmt::U16, &flow, 100.0).unwrap(), [65535.0, 32767.0, 32767.0, 0.0]);
    all_pixels(&encode_vectors_image(5, 4, Fmt::F32, &flow, 100.0).unwrap(), [1.0, 0.5, 0.5, 0.0]);
}

#[test]
fn full_scale_is_254_65534_and_the_integer_depths_clamp_but_float_does_not() {
    // +-scale: R = 254, G = 0 (8 bpc), 65534 and 0 (16 bpc), 1.0 and 0.0 (float)
    let edge = constant(3, 3, [4.0, -4.0]);
    all_pixels(&encode_vectors_image(3, 3, Fmt::U8, &edge, 4.0).unwrap(), [255.0, 254.0, 0.0, 0.0]);
    all_pixels(&encode_vectors_image(3, 3, Fmt::U16, &edge, 4.0).unwrap(), [65535.0, 65534.0, 0.0, 0.0]);
    all_pixels(&encode_vectors_image(3, 3, Fmt::F32, &edge, 4.0).unwrap(), [1.0, 1.0, 0.0, 0.0]);
    // ten times the scale: the integer depths stop at full scale, float continues linearly
    let far = constant(3, 3, [40.0, -40.0]);
    all_pixels(&encode_vectors_image(3, 3, Fmt::U8, &far, 4.0).unwrap(), [255.0, 254.0, 0.0, 0.0]);
    all_pixels(&encode_vectors_image(3, 3, Fmt::U16, &far, 4.0).unwrap(), [65535.0, 65534.0, 0.0, 0.0]);
    all_pixels(&encode_vectors_image(3, 3, Fmt::F32, &far, 4.0).unwrap(), [1.0, 5.5, -4.5, 0.0]);
}

#[test]
fn a_working_resolution_flow_is_resampled_and_scaled_to_the_output_size() {
    // a 2x2 flow of (1, 1) for a 3x3 output: the displacement is multiplied by (3 - 1) / (2 - 1) = 2, scale 4: fx = 0.5
    let flow = constant(2, 2, [1.0, 1.0]);
    all_pixels(&encode_vectors_image(3, 3, Fmt::U8, &flow, 4.0).unwrap(), [255.0, 191.0, 191.0, 0.0]);
    all_pixels(&encode_vectors_image(3, 3, Fmt::F32, &flow, 4.0).unwrap(), [1.0, 0.75, 0.75, 0.0]);
    // the half resolution case of the plugin: a 49 x 33 flow for a 97 x 65 output scales by 96 / 48 = 2 and 64 / 32 = 2:
    // (0.5, -0.25) becomes (1, -0.5) output pixels, fx = 0.25 gives R = 0.625 and fy = -0.125 gives G = 0.4375
    let half = constant(49, 33, [0.5, -0.25]);
    all_pixels(&encode_vectors_image(97, 65, Fmt::F32, &half, 4.0).unwrap(), [1.0, 0.625, 0.4375, 0.0]);
}

#[test]
fn a_gradient_flow_is_sampled_bilinearly_at_the_scaled_position() {
    // flow.x = x (0 to 4 over 5 columns) for a 9 column output: output column X samples at X * 4 / 8 = X / 2
    // and the displacement doubles (8 / 4): the value at output column X is (X / 2) * 2 = X output pixels
    let mut flow = FlowField::new(5, 2);
    for y in 0..2 {
        for x in 0..5 {
            flow.set(x, y, [x as f32, 0.0]);
        }
    }
    let img = encode_vectors_image(9, 3, Fmt::F32, &flow, 16.0).unwrap();
    for x in 0..9 {
        let want = (x as f64 / 16.0 + 1.0) * 0.5;
        assert!((argb(&img, x, 1)[1] - want).abs() < 1e-6, "column {x}: {} against {want}", argb(&img, x, 1)[1]);
    }
}

#[test]
fn decode_inverts_encode_within_half_a_code_step() {
    // 8 bpc: one code step is scale / 127 (127 codes per scale), so the error is at most scale / 254; 16 bpc scale / 65534
    let scale = 16.0f32;
    for (fmt, step) in [(Fmt::U8, scale as f64 / 254.0), (Fmt::U16, scale as f64 / 65534.0), (Fmt::F32, 1e-5)] {
        let mut worst = 0.0f64;
        for i in 0..=320 {
            let v = -16.0 + i as f32 * 0.1;
            let flow = constant(2, 2, [v, -v * 0.5]);
            let back = decode_vectors(&encode_vectors_image(2, 2, fmt, &flow, scale).unwrap(), scale);
            let [u, w] = back.at(1, 1);
            worst = worst.max((u as f64 - v as f64).abs()).max((w as f64 + v as f64 * 0.5).abs());
        }
        assert!(worst <= step * 1.001 + 1e-5, "{fmt:?}: worst error {worst}, bound {step}");
        eprintln!("{fmt:?}: worst round trip error {worst:.6} px at Max Displace 16");
    }
}

#[test]
fn decode_reads_alpha_zero_as_no_vector() {
    // the Create Vectors output is opaque, so every vector is valid; a transparent pixel would decode to zero
    let flow = constant(2, 2, [3.0, 3.0]);
    let mut img = encode_vectors_image(2, 2, Fmt::U8, &flow, 16.0).unwrap();
    if let Samples::U8(v) = &mut img.samples {
        v[0] = 0;
    }
    let back = decode_vectors(&img, 16.0);
    assert_eq!(back.at(0, 0), [0.0, 0.0]);
    assert_ne!(back.at(1, 0), [0.0, 0.0]);
}
