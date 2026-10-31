//! The layered Create Vectors pass: the single layer identity, the black matte identity, two
//! textures under complementary disc mattes (each region decodes to its own velocity), the point
//! pixel, the second tracker without points, the Layers Output groups, tiny frames (the zero
//! vector) and the thread count.

use super::*;
use crate::flow::extras::solve_pyramids_with;
use crate::image::{Fmt, Samples, CHANNELS};
use crate::vectors::{decode_vectors, encode_vectors_image, zero_vectors_image};

const FMTS: [Fmt; 3] = [Fmt::U8, Fmt::U16, Fmt::F32];

/// A smooth texture: a sum of plane waves, `salt` picks the phases, in 0..1.
fn texture(x: f32, y: f32, c: usize, salt: f32) -> f32 {
    let waves = [(17.0f32, 0.35f32, 0.3f32), (11.0, 1.9, 0.25), (23.0, 2.7, 0.25)];
    let mut s = 0.0f32;
    for (i, (l, th, a)) in waves.iter().enumerate() {
        let ph = 1.3 * i as f32 + 2.1 * c as f32 + salt;
        s += a * ((x * th.cos() + y * th.sin()) * std::f32::consts::TAU / l + ph).sin();
    }
    (0.5 + 0.5 * s).clamp(0.0, 1.0)
}

/// Pixel values in 0..1 to an opaque engine image of `fmt`.
fn opaque(fmt: Fmt, w: usize, h: usize, f: impl Fn(usize, usize, usize) -> f32) -> Image {
    let mut img = Image::new(w, h, fmt);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * CHANNELS;
            let v = [1.0, f(x, y, 0), f(x, y, 1), f(x, y, 2)];
            match &mut img.samples {
                Samples::U8(s) => (0..4).for_each(|c| s[i + c] = (v[c] * 255.0 + 0.5) as u8),
                Samples::U16(s) => (0..4).for_each(|c| s[i + c] = (v[c] * 65535.0 + 0.5) as u16),
                Samples::F32(s) => (0..4).for_each(|c| s[i + c] = v[c]),
            }
        }
    }
    img
}

/// Frame `k` of a texture moving by `v` per frame.
fn moving(fmt: Fmt, w: usize, h: usize, v: (f32, f32), k: f32, salt: f32) -> Image {
    opaque(fmt, w, h, |x, y, c| texture(x as f32 - v.0 * k, y as f32 - v.1 * k, c, salt))
}

fn settings(quality: u8, sensitivity: f32, max_displace: f32) -> CreateVectorsSettings {
    CreateVectorsSettings {
        quality,
        hiq: true,
        sensitivity,
        prep_contrast_edge: false,
        prep_delinearize: false,
        track_rgb_a: false,
        max_displace,
    }
}

fn layer(sensitivity: f32, track: bool, shrink: f32) -> LayerParams {
    LayerParams { matte_chan: 1, sensitivity, shrink, track }
}

/// A job with `mattes` (the same at X and Y) and every layer at `sensitivity` with Track on.
fn job_with(s: CreateVectorsSettings, mattes: Vec<Matte>, first: usize, last: usize) -> CreateVectorsJob {
    let nfg = mattes.len();
    CreateVectorsJob {
        settings: s,
        layers: (0..=nfg).map(|_| layer(s.sensitivity, true, 0.0)).collect(),
        mattes_x: mattes.clone(),
        mattes_y: mattes,
        points: Vec::new(),
        first,
        last,
    }
}

#[test]
fn the_single_layer_job_is_create_vectors_bit_for_bit() {
    for fmt in FMTS {
        let (x, y) = (moving(fmt, 40, 30, (2.0, 1.0), 1.0, 0.0), moving(fmt, 40, 30, (2.0, 1.0), 0.0, 0.0));
        for quality in [QUALITY_NORMAL, QUALITY_FULL] {
            for (prep_contrast_edge, track_rgb_a) in [(false, false), (true, true)] {
                let s = CreateVectorsSettings { prep_contrast_edge, track_rgb_a, ..settings(quality, 0.7, 16.0) };
                let job = CreateVectorsJob::single(s);
                assert!(job.is_single_layer());
                assert_eq!(create_vectors_job(&x, &y, &job).unwrap(), create_vectors(&x, &y, &s).unwrap(), "{fmt:?} q{quality}");
                // and its flow is the plain solve
                assert_eq!(layer_flows(&x, &y, &job).unwrap(), vec![solve_flow(&x, &y, &s).unwrap()]);
                assert_eq!(job_flow(&x, &y, &job).unwrap(), solve_flow(&x, &y, &s).unwrap());
            }
        }
    }
}

#[test]
fn a_black_fg_matte_with_layers_output_all_is_the_single_layer_output() {
    for fmt in FMTS {
        let (w, h) = (40usize, 30usize);
        let (x, y) = (moving(fmt, w, h, (2.0, 1.0), 1.0, 0.0), moving(fmt, w, h, (2.0, 1.0), 0.0, 0.0));
        let s = settings(QUALITY_NORMAL, 0.7, 16.0);
        let single = create_vectors(&x, &y, &s).unwrap();
        for nfg in 1..=3 {
            let job = job_with(s, vec![Matte::filled(w, h, 0.0); nfg], 0, nfg);
            assert_eq!(create_vectors_job(&x, &y, &job).unwrap(), single, "{fmt:?} nfg {nfg}");
            // Layers Output MAIN_BG alone is the same picture with its weight, 1 everywhere, in the blue channel
            let job = job_with(s, vec![Matte::filled(w, h, 0.0); nfg], nfg, nfg);
            let mut want = single.clone();
            match &mut want.samples {
                Samples::U8(v) => v.chunks_exact_mut(CHANNELS).for_each(|p| p[3] = 255),
                Samples::U16(v) => v.chunks_exact_mut(CHANNELS).for_each(|p| p[3] = 65535),
                Samples::F32(v) => v.chunks_exact_mut(CHANNELS).for_each(|p| p[3] = 1.0),
            }
            assert_eq!(create_vectors_job(&x, &y, &job).unwrap(), want, "{fmt:?} nfg {nfg} MAIN_BG");
        }
    }
}

/// `(mean, max)` Euclidean error of `flow` against `truth` over the pixels where `keep` holds.
fn region_error(flow: &FlowField, truth: (f32, f32), keep: impl Fn(usize, usize) -> bool) -> (f64, f64, usize) {
    let (mut sum, mut max, mut n) = (0.0f64, 0.0f64, 0usize);
    for y in 0..flow.height {
        for x in 0..flow.width {
            if !keep(x, y) {
                continue;
            }
            let [u, v] = flow.at(x, y);
            let e = (((u - truth.0) as f64).powi(2) + ((v - truth.1) as f64).powi(2)).sqrt();
            sum += e;
            max = max.max(e);
            n += 1;
        }
    }
    (sum / n as f64, max, n)
}

#[test]
fn fg_and_bg_regions_decode_to_their_own_velocities() {
    // FG1: a texture inside a disc of radius 22 that moves with it by (+3, 0) per frame; Main_BG:
    // another texture moving by (-2, +1). X is frame 1 and Y frame 0 (Track Frame Previous), so the
    // true flows of X(p) ~ Y(p + flow) are (-3, 0) and (+2, -1).
    let (w, h) = (96usize, 80usize);
    let (vf, vb) = ((3.0f32, 0.0f32), (-2.0f32, 1.0f32));
    let (cx, cy, r) = (46.0f32, 40.0f32, 22.0f32);
    let inside = |x: usize, y: usize, k: f32| ((x as f32 - (cx + vf.0 * k)).powi(2) + (y as f32 - (cy + vf.1 * k)).powi(2)).sqrt() <= r;
    let matte = |k: f32| Matte { width: w, height: h, data: (0..w * h).map(|i| if inside(i % w, i / w, k) { 1.0 } else { 0.0 }).collect() };
    let frame = |k: f32| {
        opaque(Fmt::F32, w, h, |x, y, c| {
            if inside(x, y, k) {
                texture(x as f32 - vf.0 * k, y as f32 - vf.1 * k, c, 0.0)
            } else {
                texture(x as f32 - vb.0 * k, y as f32 - vb.1 * k, c, 3.0)
            }
        })
    };
    let (x, y) = (frame(1.0), frame(0.0));
    let s = settings(QUALITY_FULL, 1.0, 16.0);
    let mut job = job_with(s, vec![matte(1.0)], 0, 1);
    job.mattes_y = vec![matte(0.0)];
    let out = create_vectors_job(&x, &y, &job).unwrap();
    let got = decode_vectors(&out, 16.0);
    let near = |x: usize, y: usize, rad: f32| ((x as f32 - (cx + vf.0)).powi(2) + (y as f32 - cy).powi(2)).sqrt() <= rad;
    let core = |x: usize, y: usize| near(x, y, 12.0);
    let far = |x: usize, y: usize| !near(x, y, 34.0) && x >= 8 && y >= 8 && x < w - 8 && y < h - 8;
    // the two sides of the matte edge, where one flow for both regions mixes the two motions
    let fg_ring = |x: usize, y: usize| near(x, y, r - 1.0) && !near(x, y, r - 7.0);
    let bg_ring = |x: usize, y: usize| !near(x, y, r + 1.0) && near(x, y, r + 7.0);
    let (fg_mean, fg_max, n_fg) = region_error(&got, (-3.0, 0.0), core);
    let (bg_mean, bg_max, n_bg) = region_error(&got, (2.0, -1.0), far);
    eprintln!("layered: FG mean {fg_mean:.4} max {fg_max:.4} ({n_fg} px), BG mean {bg_mean:.4} max {bg_max:.4} ({n_bg} px)");
    assert!(n_fg > 300 && n_bg > 1000);
    // measured: FG mean 0.0020 (max 0.0059), BG mean 0.0018 (max 0.021)
    assert!(fg_mean < 0.02 && fg_max < 0.1, "FG core: mean {fg_mean}, max {fg_max}");
    assert!(bg_mean < 0.02 && bg_max < 0.1, "BG: mean {bg_mean}, max {bg_max}");
    // the single layer solve mixes the two motions at the matte edge: there the layered one is
    // much the better on both sides
    let single = decode_vectors(&create_vectors(&x, &y, &s).unwrap(), 16.0);
    let ring = |f: &FlowField| (region_error(f, (-3.0, 0.0), fg_ring).0, region_error(f, (2.0, -1.0), bg_ring).0);
    let (lf, lb) = ring(&got);
    let (sf, sb) = ring(&single);
    eprintln!("edge rings: layered FG {lf:.4} BG {lb:.4}, single layer FG {sf:.4} BG {sb:.4}");
    // measured (every layer on its own layer image, as the original's code solves it): layered FG 0.0008, BG 1.08; single
    // layer FG 0.31, BG 1.55. Main_BG solves on the frames times its visible matte, whose black FG area keeps a
    // static edge at the matte boundary, so its ring stays well above the FG one, still below the single layer's
    assert!(lf < 0.05 && lb < 1.5, "layered rings: FG {lf}, BG {lb}");
    assert!(lf * 10.0 < sf && lb < sb, "layered FG {lf} BG {lb} against single FG {sf} BG {sb}");
    // each layer alone (Layers Output FG1 and MAIN_BG) carries its own region; a layer alone is its own flow,
    // not weighted, with its visible matte in blue, as the original's code writes it
    let fg_job = CreateVectorsJob { first: 0, last: 0, ..job.clone() };
    let fg_out = create_vectors_job(&x, &y, &fg_job).unwrap();
    let fg_only = decode_vectors(&fg_out, 16.0);
    assert!(region_error(&fg_only, (-3.0, 0.0), core).0 < 0.02);
    let (vis, _) = matte::layer_solve_mattes(&fg_job.mattes_x, &[0.0], 0, Fmt::F32, w, h).unwrap();
    let mut want = Image::new(w, h, Fmt::F32);
    vectors::encode_vectors_blue(&mut want, &layer_flows(&x, &y, &fg_job).unwrap()[0], 16.0, &vis).unwrap();
    assert_eq!(fg_out, want, "FG1 alone is its own flow, also outside its matte");
    let bg_only = decode_vectors(&create_vectors_job(&x, &y, &CreateVectorsJob { first: 1, last: 1, ..job.clone() }).unwrap(), 16.0);
    assert!(region_error(&bg_only, (2.0, -1.0), far).0 < 0.02);
}

#[test]
fn the_point_pixel_carries_the_point_displacement_at_full_res() {
    // Vector Resolution Full res (no reduction), Sensitivity 100: the user displacement sets the
    // flow of the point's pixel at the end of level 0, so it decodes to `to - from` exactly
    let (w, h) = (64usize, 48usize);
    for fmt in FMTS {
        let (x, y) = (moving(fmt, w, h, (1.0, 0.5), 1.0, 0.0), moving(fmt, w, h, (1.0, 0.5), 0.0, 0.0));
        let s = settings(QUALITY_FULL, 1.0, 16.0);
        let mut job = CreateVectorsJob::single(s);
        job.points = vec![vec![UserPoint { at_a: [40.0, 30.0], at_b: [46.0, 32.0] }]];
        assert!(!job.is_single_layer());
        let out = create_vectors_job(&x, &y, &job).unwrap();
        let i = (30 * w + 40) * CHANNELS;
        match &out.samples {
            Samples::U8(v) => assert_eq!(&v[i..i + 4], &[255, 175, 143, 0]),
            Samples::U16(v) => assert_eq!(&v[i..i + 4], &[65535, 45055, 36863, 0]),
            Samples::F32(v) => assert_eq!(&v[i..i + 4], &[1.0, 0.6875, 0.5625, 0.0]),
        }
        if fmt == Fmt::F32 {
            assert_eq!(decode_vectors(&out, 16.0).at(40, 30), [6.0, 2.0]);
        }
        // the same point on Main_BG of a layered job (a black FG matte): the pixel is Main_BG's
        let mut layered = job_with(s, vec![Matte::filled(w, h, 0.0)], 0, 1);
        layered.points = vec![Vec::new(), job.points[0].clone()];
        let out = create_vectors_job(&x, &y, &layered).unwrap();
        let j = (30 * w + 40) * CHANNELS;
        match &out.samples {
            Samples::U8(v) => assert_eq!(&v[j..j + 4], &[255, 175, 143, 0]),
            Samples::U16(v) => assert_eq!(&v[j..j + 4], &[65535, 45055, 36863, 0]),
            Samples::F32(v) => assert_eq!(&v[j..j + 4], &[1.0, 0.6875, 0.5625, 0.0]),
        }
    }
}

#[test]
fn track_off_without_points_is_the_zero_vector_everywhere() {
    for fmt in FMTS {
        let (w, h) = (24usize, 18usize);
        let (x, y) = (moving(fmt, w, h, (2.0, 1.0), 1.0, 0.0), moving(fmt, w, h, (2.0, 1.0), 0.0, 0.0));
        let s = settings(QUALITY_NORMAL, 0.7, 16.0);
        let zero = zero_vectors_image(w, h, fmt);
        let mut job = CreateVectorsJob::single(s);
        job.layers[0].track = false;
        assert_eq!(create_vectors_job(&x, &y, &job).unwrap(), zero, "{fmt:?}");
        // the literal codes: 127, 32767 or 0.5 in R and G, B 0, alpha white
        let want: [f32; 4] = [fmt.white(), [127.0, 32767.0, 0.5][fmt.code() as usize], [127.0, 32767.0, 0.5][fmt.code() as usize], 0.0];
        let px: Vec<f32> = match &zero.samples {
            Samples::U8(v) => v[..4].iter().map(|&c| f32::from(c)).collect(),
            Samples::U16(v) => v[..4].iter().map(|&c| f32::from(c)).collect(),
            Samples::F32(v) => v[..4].to_vec(),
        };
        assert_eq!(px, want);
        // every layer of a layered job off: the merge of zero fields
        let mut layered = job_with(s, vec![disc(w, h, 12.0, 9.0, 5.0)], 0, 1);
        layered.layers.iter_mut().for_each(|l| l.track = false);
        assert_eq!(create_vectors_job(&x, &y, &layered).unwrap(), zero, "{fmt:?} layered");
        // Main_BG tracking again: not the zero vector any more
        layered.layers[1].track = true;
        assert_ne!(create_vectors_job(&x, &y, &layered).unwrap(), zero, "{fmt:?}");
    }
}

#[test]
fn the_second_tracker_moves_with_its_points() {
    // one point on a layer whose Track is off: its flow is near uniform `to - from` (rule G5)
    let (w, h) = (32usize, 24usize);
    let x = moving(Fmt::F32, w, h, (0.0, 0.0), 0.0, 0.0);
    let mut job = CreateVectorsJob::single(settings(QUALITY_FULL, 0.7, 16.0));
    job.layers[0].track = false;
    job.points = vec![vec![UserPoint { at_a: [10.0, 8.0], at_b: [13.0, 7.0] }]];
    let got = decode_vectors(&create_vectors_job(&x, &x, &job).unwrap(), 16.0);
    let (mean, _, _) = region_error(&got, (3.0, -1.0), |_, _| true);
    assert!(mean < 1e-3, "mean {mean}");
    assert_eq!(got.at(10, 8), [3.0, -1.0]);
}

/// A disc matte of radius `r` at `(cx, cy)` with a soft edge.
fn disc(w: usize, h: usize, cx: f32, cy: f32, r: f32) -> Matte {
    let data = (0..w * h)
        .map(|i| {
            let d = (((i % w) as f32 - cx).powi(2) + ((i / w) as f32 - cy).powi(2)).sqrt();
            (r - d + 0.5).clamp(0.0, 1.0)
        })
        .collect();
    Matte { width: w, height: h, data }
}

#[test]
fn layers_output_picks_the_group() {
    assert_eq!(output_group(1, 0), (0, 0));
    assert_eq!(output_group(1, 3), (0, 3));
    assert_eq!((2..=5).map(|v| output_group(v, 3)).collect::<Vec<_>>(), [(0, 0), (1, 1), (2, 2), (3, 3)]);
    assert_eq!((2..=5).map(|v| output_group(v, 1)).collect::<Vec<_>>(), [(0, 0), (1, 1), (1, 1), (1, 1)]);
    assert_eq!((2..=5).map(|v| output_group(v, 0)).collect::<Vec<_>>(), [(0, 0); 4]);
    assert_eq!(output_group(0, 2), (0, 2));
    // the flows of a layer do not depend on the group, and a group's output is the merge of its layers
    let (w, h) = (64usize, 48usize);
    let fmt = Fmt::U16;
    let (x, y) = (moving(fmt, w, h, (2.0, 1.0), 1.0, 0.0), moving(fmt, w, h, (2.0, 1.0), 0.0, 0.0));
    let s = settings(QUALITY_FULL, 0.9, 16.0);
    let mattes = vec![disc(w, h, 32.0, 24.0, 14.0), disc(w, h, 46.0, 30.0, 12.0)];
    let mut all = job_with(s, mattes.clone(), 0, 2);
    all.layers[1].sensitivity = 0.4;
    all.layers[0].shrink = 1.5;
    let flows = layer_flows(&x, &y, &all).unwrap();
    assert_eq!(flows.len(), 3);
    for k in 0..=2 {
        let one = CreateVectorsJob { first: k, last: k, ..all.clone() };
        assert_eq!(layer_flows(&x, &y, &one).unwrap(), vec![flows[k].clone()], "layer {k}");
        // a single layer: its own flow, its visible matte in the blue channel
        let shrinks: Vec<f32> = all.layers[..2].iter().map(|l| l.shrink).collect();
        let (vis, _) = matte::layer_solve_mattes(&mattes, &shrinks, k, fmt, w, h).unwrap();
        let mut want = Image::new(w, h, fmt);
        vectors::encode_vectors_blue(&mut want, &flows[k], 16.0, &vis).unwrap();
        assert_eq!(create_vectors_job(&x, &y, &one).unwrap(), want, "layer {k}");
    }
    // the group: the bottom layer encoded, the others laid over it bottom up with their raw mattes (finding 4)
    let mut want = Image::new(w, h, fmt);
    vectors::encode_vectors(&mut want, &flows[2], 16.0).unwrap();
    for k in [1usize, 0] {
        vectors::composite_vectors(&mut want, &flows[k], 16.0, &matte::integer_plane(&mattes[k], fmt)).unwrap();
    }
    assert_eq!(create_vectors_job(&x, &y, &all).unwrap(), want);
    let wts = matte::over_weights(&mattes, 0, 2, w, h).unwrap();
    let layers: Vec<(&FlowField, &Matte)> = flows.iter().zip(wts.iter()).collect();
    // the layers differ: FG2 has another sensitivity and FG1's shrink only changes the layers below it
    let peak = |f: &FlowField| f.data.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(peak(&flows[0]) > 0.5 && peak(&flows[2]) > 0.5);
    assert_ne!(flows[0], flows[1]);
    assert_ne!(flows[0], flows[2]);
    let unshrunk = layer_flows(&x, &y, &CreateVectorsJob { layers: all.layers.iter().map(|l| LayerParams { shrink: 0.0, ..*l }).collect(), ..all.clone() }).unwrap();
    assert_eq!(unshrunk[0], flows[0], "FG1's own tracking matte is unchoked");
    // the Debug flow of the group is the merged flow
    assert_eq!(job_flow(&x, &y, &all).unwrap(), vectors::merge_flows(w, h, &layers).unwrap());
}

#[test]
fn a_side_of_one_is_the_zero_vector_whatever_the_job() {
    // U, ours (G7 family): no solver, no second tracker, no merge on a frame with a side of 1
    for fmt in FMTS {
        for (w, h) in [(1usize, 1usize), (1, 17), (17, 1)] {
            let (x, y) = (moving(fmt, w, h, (2.0, 1.0), 1.0, 0.0), moving(fmt, w, h, (2.0, 1.0), 0.0, 0.0));
            let zero = zero_vectors_image(w, h, fmt);
            let s = settings(QUALITY_NORMAL, 0.7, 16.0);
            for nfg in 0..=3 {
                for track in [true, false] {
                    for (first, last) in [(0, nfg), (nfg, nfg), (0, 0)] {
                        let mut job = job_with(s, vec![Matte::filled(w, h, 0.5); nfg], first, last);
                        job.layers.iter_mut().for_each(|l| l.track = track);
                        job.points = (0..=nfg).map(|_| vec![UserPoint { at_a: [0.0, 0.0], at_b: [0.0, 1.0] }]).collect();
                        assert_eq!(create_vectors_job(&x, &y, &job).unwrap(), zero, "{fmt:?} {w}x{h} nfg {nfg}");
                        let flows = layer_flows(&x, &y, &job).unwrap();
                        assert_eq!(flows.len(), last - first + 1);
                        assert!(flows.iter().all(|f| *f == FlowField::new(w, h)));
                        assert_eq!(job_flow(&x, &y, &job).unwrap(), FlowField::new(w, h));
                    }
                }
            }
        }
    }
}

#[test]
fn a_job_that_does_not_fit_its_frames_is_a_mismatch() {
    let (w, h) = (16usize, 12usize);
    let x = moving(Fmt::U8, w, h, (1.0, 0.0), 0.0, 0.0);
    let s = settings(QUALITY_NORMAL, 0.7, 16.0);
    let mismatch = |job: &CreateVectorsJob| matches!(create_vectors_job(&x, &x, job), Err(fx_core::FxError::Mismatch(_)));
    assert!(mismatch(&job_with(s, vec![Matte::filled(w + 1, h, 0.0)], 0, 1)));
    assert!(mismatch(&CreateVectorsJob { mattes_y: Vec::new(), ..job_with(s, vec![Matte::filled(w, h, 0.0)], 0, 1) }));
    assert!(mismatch(&CreateVectorsJob { first: 1, last: 0, ..job_with(s, vec![Matte::filled(w, h, 0.0)], 0, 1) }));
    assert!(mismatch(&CreateVectorsJob { last: 2, ..job_with(s, vec![Matte::filled(w, h, 0.0)], 0, 1) }));
    assert!(mismatch(&CreateVectorsJob { layers: vec![layer(0.7, true, 0.0)], ..job_with(s, vec![Matte::filled(w, h, 0.0)], 0, 1) }));
    assert!(mismatch(&CreateVectorsJob { points: vec![Vec::new()], ..job_with(s, vec![Matte::filled(w, h, 0.0)], 0, 1) }));
    let other = moving(Fmt::U8, w, h + 1, (1.0, 0.0), 0.0, 0.0);
    assert!(matches!(create_vectors_job(&x, &other, &CreateVectorsJob::single(s)), Err(fx_core::FxError::Mismatch(_))));
    assert!(matches!(layer_flows(&x, &other, &CreateVectorsJob::single(s)), Err(fx_core::FxError::Mismatch(_))));
}

#[test]
fn thread_count_does_not_change_a_layered_job() {
    let pool = |n: usize| rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
    let (w, h) = (48usize, 36usize);
    for fmt in FMTS {
        let (x, y) = (moving(fmt, w, h, (2.0, 1.0), 1.0, 0.0), moving(fmt, w, h, (2.0, 1.0), 0.0, 0.0));
        let s = settings(QUALITY_NORMAL, 0.8, 16.0);
        let mut job = job_with(s, vec![disc(w, h, 16.0, 14.0, 9.0), disc(w, h, 30.0, 20.0, 10.0)], 0, 2);
        job.layers[0].shrink = 2.5;
        job.layers[1].track = false;
        job.points = vec![vec![UserPoint { at_a: [16.0, 14.0], at_b: [18.0, 15.0] }], vec![UserPoint { at_a: [30.0, 20.0], at_b: [29.0, 22.0] }], Vec::new()];
        let one = pool(1).install(|| (create_vectors_job(&x, &y, &job).unwrap(), job_flow(&x, &y, &job).unwrap()));
        for n in [4, 16] {
            let other = pool(n).install(|| (create_vectors_job(&x, &y, &job).unwrap(), job_flow(&x, &y, &job).unwrap()));
            assert_eq!(one.0, other.0, "{fmt:?} {n} threads");
            assert_eq!(one.1, other.1, "{fmt:?} {n} threads");
        }
        // and the encode of nfg = 0 with points is the encode of its flow
        let mut single = CreateVectorsJob::single(s);
        single.points = vec![job.points[0].clone()];
        let flow = layer_flows(&x, &y, &single).unwrap().remove(0);
        assert_eq!(create_vectors_job(&x, &y, &single).unwrap(), encode_vectors_image(w, h, fmt, &flow, 16.0).unwrap());
    }
}

/// A small layer can get the exact zero flow, and that is the solver's own update test, not a
/// reset: on this smooth texture moving by (2, 1) px a frame, the masked solve of a 14 px disc at
/// (22, 20) or (26, 22) of a 64x48 frame (Full res, Sensitivity 90) never updates a pixel, while
/// the same disc at (32, 24), (22, 24) or (32, 20) moves toward it. Gradient masking
/// leaves the structure tensor zero outside the layer's mask, whose levels erode at every step
/// (empty from level 3 on here), and the radius 7 blur dilutes what is left, so at those two
/// positions no pixel of any level passes the update test `|t| > 40` of the plain solver: the
/// flow stays at its zero start (all `-0.0` after the final negation) and the range check never
/// fires (a zero flow is in range everywhere). `flow::tests_extras` pins the property over a sweep
/// of positions: a masked solve is the zero flow exactly where no level passes the test.
#[test]
fn small_masked_discs_keep_the_zero_flow_where_the_update_test_never_passes() {
    let (w, h) = (64usize, 48usize);
    let fmt = Fmt::U16;
    let (x, y) = (moving(fmt, w, h, (2.0, 1.0), 1.0, 0.0), moving(fmt, w, h, (2.0, 1.0), 0.0, 0.0));
    let s = settings(QUALITY_FULL, 0.9, 16.0).flow_settings();
    let (px, py) = (flow::build_pyramid(&x, &s).unwrap(), flow::build_pyramid(&y, &s).unwrap());
    let plain = flow::solve_pyramids(&px, &py, &s).unwrap();
    for (cx, cy, moves) in [(22.0f32, 20.0f32, false), (26.0, 22.0, false), (32.0, 24.0, true), (22.0, 24.0, true), (32.0, 20.0, true)] {
        let matte = disc(w, h, cx, cy, 14.0);
        let mp = MaskPyramid::build(&matte, &s).unwrap();
        let f = solve_pyramids_with(&px, &py, &s, SolveExtras { masks: Some((&mp, &mp)), user: None }).unwrap();
        if moves {
            // inside the disc the masked flow moves toward the true (-2, -1) (the zero flow is
            // sqrt(5) away); measured: mean error 0.46, 1.42 and 1.73
            let (mean, _, n) = region_error(&f, (-2.0, -1.0), |x, y| mp.level(0)[y * w + x] == 1.0);
            assert!(n > 500 && mean < 2.0, "disc at ({cx}, {cy}): mean error {mean}");
        } else {
            assert!(f.data.iter().all(|v| v.to_bits() == (-0.0f32).to_bits()), "disc at ({cx}, {cy})");
        }
        // the unmasked solve tracks the motion there either way
        let (mean, _, _) = region_error(&plain, (-2.0, -1.0), |x, y| mp.level(0)[y * w + x] == 1.0);
        assert!(mean < 0.02, "unmasked at ({cx}, {cy}): {mean}");
    }
}

#[test]
fn the_debug_display_of_an_fg_stack_composes_the_layers() {
    // the original's order: bottom up, Y_k OVER the output, then the forward warp of X_k OVER it
    use crate::debug::{debug_view_on, views_of, VIEW_DIFF, VIEW_WARP};
    use crate::stages::Stages;
    let (w, h) = (24usize, 16usize);
    let fmt = Fmt::U8;
    let (x, y) = (moving(fmt, w, h, (1.0, 0.5), 1.0, 0.0), moving(fmt, w, h, (1.0, 0.5), 0.0, 0.0));
    let s = settings(QUALITY_FULL, 1.0, 16.0);
    let mattes = vec![disc(w, h, 12.0, 8.0, 5.0)];
    let all = job_with(s, mattes.clone(), 0, 1);
    let flows = layer_flows(&x, &y, &all).unwrap();
    let mut want = Image::new(w, h, fmt);
    for k in [1usize, 0] {
        let (vx, _) = matte::layer_solve_mattes(&mattes, &[0.0], k, fmt, w, h).unwrap();
        let (vy, _) = matte::layer_solve_mattes(&mattes, &[0.0], k, fmt, w, h).unwrap();
        matte::over_pl(&mut want, &matte::premultiply_pl(&y, &vy).unwrap()).unwrap();
        let mut wk = Image::new(w, h, fmt);
        Cpu.forward_warp(&mut wk, &matte::premultiply_pl(&x, &vx).unwrap(), &flows[k], 1.0).unwrap();
        matte::over_pl(&mut want, &wk).unwrap();
    }
    assert_eq!(debug_job_view_on(&Cpu, &Cpu, &x, &y, &all, VIEW_WARP).unwrap(), want);
    assert_eq!(debug_job_view_on(&Cpu, &Cpu, &x, &y, &all, VIEW_DIFF).unwrap(), views_of(&want, &y, VIEW_DIFF));
    // a one layer group's Warp view is that layer's warp alone
    let bg = CreateVectorsJob { first: 1, last: 1, ..all.clone() };
    let (vx, _) = matte::layer_solve_mattes(&mattes, &[0.0], 1, fmt, w, h).unwrap();
    let mut wk = Image::new(w, h, fmt);
    Cpu.forward_warp(&mut wk, &matte::premultiply_pl(&x, &vx).unwrap(), &flows[1], 1.0).unwrap();
    assert_eq!(debug_job_view_on(&Cpu, &Cpu, &x, &y, &bg, VIEW_WARP).unwrap(), wk);
    // without FG mattes it is the single layer Debug view
    let one = CreateVectorsJob::single(s);
    let flow = job_flow(&x, &y, &one).unwrap();
    assert_eq!(debug_job_view_on(&Cpu, &Cpu, &x, &y, &one, VIEW_DIFF).unwrap(), debug_view_on(&Cpu, &x, &y, &flow, VIEW_DIFF).unwrap());
}
