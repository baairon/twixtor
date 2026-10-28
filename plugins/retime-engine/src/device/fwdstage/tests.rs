//! Device-free tests of the forward warp stage: the scan converter transcription against
//! `forward::raster::triangle` hit by hit, the prefix sum on its own, the pyramid levels, and
//! whole warps of the simulated stage in shuffled work item orders against
//! `forward::forward_warp_with`, over random and adversarial meshes at every depth.

use super::sim::{self, Sink, Vtx};
use super::{plan, Plan};
use crate::device::image::StageError;
use crate::device::kernels::fwdstage as k;
use crate::device::selfcheck::compare_floats;
use crate::device::sim::{Global, Rng, SEEDS};
use crate::device::warpstage::corpus::{self, first_diff, FMTS};
use crate::flow::FlowField;
use crate::forward::{self, pyramid, raster, ForwardOpts};
use crate::image::{Fmt, Image, Samples};
use fx_core::FxError;

fn unit(rng: &mut Rng) -> f32 {
    (rng.next_u32() >> 8) as f32 / (1u32 << 24) as f32
}

// ---- the scan converter -------------------------------------------------------------------------

#[derive(Default)]
struct Recorder {
    hits: Vec<(usize, usize, f32, f32, f32)>,
    outside: bool,
}

impl raster::Sink for Recorder {
    fn hit(&mut self, x: usize, y: usize, u: f32, v: f32, d: f32) {
        self.hits.push((x, y, u, v, d));
    }
    fn outside(&mut self) {
        self.outside = true;
    }
}

/// A coordinate in or around `0..=m`: uniform, integers, halves, the edges, far away, specials.
fn coord(rng: &mut Rng, m: f32) -> f32 {
    match rng.below(20) {
        0 => rng.below(m as u64 + 1) as f32,
        1 => rng.below(2 * m as u64 + 1) as f32 * 0.5,
        2 => m,
        3 => 0.0,
        4 => -0.0,
        5 => [f32::NAN, f32::INFINITY, f32::NEG_INFINITY][rng.below(3) as usize],
        6 => [1e30, -1e30, 3e9, -3e9, 1e6][rng.below(5) as usize],
        7 => m + 0.0001,
        8 => (rng.below(m as u64 + 1) as f32) - 0.0001,
        9 => unit(rng) * 4.0 * m - 1.5 * m,
        _ => unit(rng) * (m + 6.0) - 3.0,
    }
}

fn vtx(rng: &mut Rng, w1: f32, h1: f32) -> raster::Vtx {
    raster::Vtx { x: coord(rng, w1), y: coord(rng, h1), u: unit(rng), v: unit(rng), d: unit(rng) * 11.0 - 3.0 }
}

/// Runs the transcription in mode 1 with one record slot per pixel (`off[p] = p`), returns the
/// records, the refusal word and the return value.
fn transcribed(t: &[raster::Vtx; 3], w: usize, h: usize, key: u32) -> (Vec<[u32; 4]>, usize, i32, bool) {
    let npx = w * h;
    let mut cnt = vec![1u32; npx];
    let mut fill = vec![0u32; npx];
    let off: Vec<u32> = (0..npx as u32).collect();
    let cpre = vec![0u32; npx.div_ceil(64)];
    let sbase = vec![0u32; npx.div_ceil(4096)];
    let mut rec = Global::new(npx, [0u32; 4]);
    let mut flag = 0i32;
    let cv = |v: &raster::Vtx| Vtx { x: v.x, y: v.y, u: v.u, v: v.v, d: v.d };
    let ok = {
        let mut s = Sink { mode: 1, w: w as i32, nh: 0, cnt: &mut cnt, fill: &mut fill, off: &off, cpre: &cpre, sbase: &sbase, rec: Some(&mut rec), total: npx as u32, flag: &mut flag };
        sim::fw_triangle(&mut s, cv(&t[0]), cv(&t[1]), cv(&t[2]), w as i32 - 1, h as i32 - 1, key)
    };
    let n = rec.written();
    (rec.into_vec(), n, flag, ok)
}

fn check_triangle(t: &[raster::Vtx; 3], w: usize, h: usize, what: &str) -> usize {
    let mut cpu = Recorder::default();
    raster::triangle(t, w as i32 - 1, h as i32 - 1, &mut cpu);
    let key = 0x1234_5679u32;
    let (rec, n, flag, ok) = transcribed(t, w, h, key);
    let mut seen = std::collections::HashSet::new();
    for &(x, y, ..) in &cpu.hits {
        assert!(seen.insert((x, y)), "{what}: the CPU hit ({x}, {y}) twice in one triangle: {t:?}");
    }
    if cpu.hits.len() > k::FW_TRI_CAP as usize {
        assert!(!ok && flag & 2 != 0, "{what}: {} hits must pass the device bound", cpu.hits.len());
        return 0;
    }
    assert!(ok && flag & 2 == 0, "{what}: refused ({flag}) a triangle of {} hits: {t:?}", cpu.hits.len());
    assert_eq!(flag & 1 != 0, cpu.outside, "{what}: the outside refusal differs: {t:?}");
    assert_eq!(n, cpu.hits.len(), "{what}: hit count: {t:?}");
    for &(x, y, u, v, d) in &cpu.hits {
        let r = rec[y * w + x];
        assert_eq!(r, [key, u.to_bits(), v.to_bits(), d.to_bits()], "{what}: hit ({x}, {y}) of {t:?}");
    }
    cpu.hits.len()
}

#[test]
fn the_scan_converter_transcription_gives_the_cpu_hits() {
    let mut rng = Rng::new(0xF0_4A4D);
    let mut hits = 0usize;
    let mut tris = 0usize;
    for &(w, h) in &[(2usize, 2usize), (3, 2), (2, 3), (5, 4), (9, 7), (17, 16), (33, 20), (64, 48)] {
        let (w1, h1) = (w as f32 - 1.0, h as f32 - 1.0);
        for i in 0..1500 {
            let mut t = [vtx(&mut rng, w1, h1), vtx(&mut rng, w1, h1), vtx(&mut rng, w1, h1)];
            // ties on purpose: shared y, shared x, both lower vertices on the last row, a flat or
            // a point triangle
            match i % 9 {
                0 => t[1].y = t[0].y,
                1 => t[2].y = t[1].y,
                2 => {
                    t[1].y = h1;
                    t[2].y = h1;
                }
                3 => {
                    t[0].y = h1;
                    t[2].y = h1;
                }
                4 => t[2] = t[1],
                5 => {
                    t[1].y = t[0].y;
                    t[2].y = t[0].y;
                }
                6 => t[1].x = t[0].x,
                _ => {}
            }
            hits += check_triangle(&t, w, h, &format!("{w}x{h} #{i}"));
            tris += 1;
        }
    }
    println!("scan converter: {tris} triangles, {hits} hits compared bit for bit");
    assert!(hits > 100_000);
}

#[test]
fn a_triangle_past_the_device_bound_raises_the_flag() {
    let (w, h) = (260usize, 260usize);
    let t = [
        raster::Vtx { x: 0.0, y: 0.0, u: 0.0, v: 0.0, d: 0.0 },
        raster::Vtx { x: 259.0, y: 0.0, u: 1.0, v: 0.0, d: 0.0 },
        raster::Vtx { x: 0.0, y: 259.0, u: 0.0, v: 1.0, d: 0.0 },
    ];
    let (_, _, flag, ok) = transcribed(&t, w, h, 0);
    assert!(!ok);
    assert_eq!(flag, 2);
}

// ---- the prefix sum -----------------------------------------------------------------------------

#[test]
fn the_three_level_prefix_sum_is_the_exclusive_scan() {
    let mut rng = Rng::new(7);
    for &npx in &[1usize, 2, 63, 64, 65, 127, 4095, 4096, 4097, 8191, 20000, 70001] {
        let cnt: Vec<u32> = (0..npx).map(|_| if rng.below(3) == 0 { 0 } else { rng.below(9) as u32 }).collect();
        let nc = npx.div_ceil(k::FW_CHUNK);
        let ns = npx.div_ceil(k::FW_SUPER);
        let base = crate::device::sim::same_for_every_order(|seed| {
            let mut off = Global::new(npx, 0u32);
            let mut csum = Global::new(nc, 0u32);
            crate::device::sim::launch_1d(nc, seed, |kk| sim::fw_scan_chunks_item(kk, &cnt, &mut off, &mut csum, npx as u32, nc as u32));
            let (off, csum) = (off.into_vec(), csum.into_vec());
            let mut cpre = Global::new(nc, 0u32);
            let mut ssum = Global::new(ns, 0u32);
            crate::device::sim::launch_1d(ns, seed ^ 3, |kk| sim::fw_scan_supers_item(kk, &csum, &mut cpre, &mut ssum, nc as u32, ns as u32));
            let (cpre, ssum) = (cpre.into_vec(), ssum.into_vec());
            let mut sbase = Global::new(ns, 0u32);
            let mut total = vec![0u32; 2];
            sim::fw_scan_top_item(0, &ssum, &mut sbase, &mut total, ns as u32);
            let sbase = sbase.into_vec();
            let b: Vec<u32> = (0..npx).map(|p| off[p] + cpre[p >> 6] + sbase[p >> 12]).collect();
            (b, total[1])
        });
        let mut s = 0u32;
        for p in 0..npx {
            assert_eq!(base.0[p], s, "{npx} pixels: pixel {p}");
            s += cnt[p];
        }
        assert_eq!(base.1, s, "{npx} pixels: total");
    }
}

#[test]
fn bands_cover_every_triangle_and_row_once() {
    for &(w, h) in &[(2usize, 2usize), (1920, 1080), (3840, 2160), (5, 300000)] {
        let tb = super::triangle_bands(w, h);
        let mut next = 0u64;
        for &(k0, n) in &tb {
            assert_eq!(k0 as u64, next);
            assert!(n as usize <= 1 << 20 && n > 0);
            next += n as u64;
        }
        assert_eq!(next, 2 * (w as u64 - 1) * (h as u64 - 1));
        let rb = super::row_bands(w, h);
        let mut y = 0;
        for &(y0, rows) in &rb {
            assert_eq!(y0, y);
            y += rows;
        }
        assert_eq!(y, h);
    }
}

// ---- the pyramid --------------------------------------------------------------------------------

fn level_bits(img: &Image) -> Vec<u64> {
    corpus::bits(img)
}

#[test]
fn the_pyramid_levels_are_the_cpus() {
    let mut rng = Rng::new(11);
    let mut levels = 0;
    for &(w, h) in &[(2usize, 2usize), (4, 4), (5, 9), (8, 5), (17, 16), (33, 20), (64, 48), (70, 9)] {
        for fmt in FMTS {
            for class in 0..2 {
                let img = corpus::image(&mut rng, w, h, fmt, class);
                let cpu = pyramid::Pyramid::build(&img);
                let want: Vec<u64> = cpu.levels.iter().flat_map(level_bits).collect();
                for &seed in &SEEDS[..2] {
                    let got: Vec<u64> = match &img.samples {
                        Samples::U8(v) => sim::sim_pyramid(v, w, h, cpu.count, seed).0.iter().map(|x| *x as u64).collect(),
                        Samples::U16(v) => sim::sim_pyramid(v, w, h, cpu.count, seed).0.iter().map(|x| *x as u64).collect(),
                        Samples::F32(v) => sim::sim_pyramid(v, w, h, cpu.count, seed).0.iter().map(|x| x.to_bits() as u64).collect(),
                    };
                    let got = if want.is_empty() { Vec::new() } else { got };
                    let nan = |b: u64| fmt == Fmt::F32 && f32::from_bits(b as u32).is_nan();
                    let diff = want.iter().zip(&got).position(|(a, b)| a != b && !(nan(*a) && nan(*b)));
                    assert_eq!(want.len(), got.len(), "{w}x{h} {fmt:?}");
                    assert_eq!(diff, None, "{w}x{h} {fmt:?} class {class}: pyramid sample differs");
                }
                levels += cpu.levels.len();
            }
        }
    }
    assert!(levels > 50);
}

// ---- whole warps --------------------------------------------------------------------------------

/// The flow classes of the corpus.
pub(crate) const FLOW_CLASSES: usize = 12;

/// A flow of class `class` for a `w x h` image (flow size `fw x fh`).
pub(crate) fn adversarial_flow(rng: &mut Rng, w: usize, h: usize, class: usize) -> (FlowField, f32) {
    let same = rng.below(2) == 0 || class >= 6;
    let (fw, fh) = if same { (w, h) } else { (2 + rng.below(w as u64) as usize, 2 + rng.below(h as u64) as usize) };
    let mut f = FlowField::new(fw, fh);
    let (cx, cy) = (fw as f32 / 2.0, fh as f32 / 2.0);
    let amounts = [1.0f32, 0.5, -1.0, 2.0, 0.25, -0.7, 1e-5, 1.5];
    let amount = amounts[rng.below(amounts.len() as u64) as usize];
    for y in 0..fh {
        for x in 0..fw {
            let (xf, yf) = (x as f32, y as f32);
            let v = match class {
                // smooth random motion
                0 => [(unit(rng) * 2.0 - 1.0) * 1.5, (unit(rng) * 2.0 - 1.0) * 1.5],
                // a zoom in or out about the centre
                1 => {
                    let s = [-0.6f32, 0.5, 1.0, -0.3][((x + y) % 4).min(3) * 0 + (fw % 4)];
                    [(xf - cx) * s, (yf - cy) * s]
                }
                // a fold: the left half moves right over the right half
                2 => [if x < fw / 2 { 0.4 * fw as f32 } else { -0.1 * fw as f32 }, 0.3],
                // a stretch: a step of many pixels across a column
                3 => [if x > fw / 3 { 9.0 } else { 0.0 }, if y > fh / 2 { -7.0 } else { 0.0 }],
                // exact integers and halves (vertices on pixel centres and edges)
                4 => [(rng.below(5) as f32 - 2.0) * 0.5, (rng.below(5) as f32 - 2.0) * 0.5],
                // far outside
                5 => {
                    if rng.below(5) == 0 {
                        [[1e6f32, -1e6, 3e9, 1e30][rng.below(4) as usize], unit(rng) * 3.0]
                    } else {
                        [unit(rng), unit(rng)]
                    }
                }
                // NaN and infinite vectors
                6 => {
                    if rng.below(8) == 0 {
                        [[f32::NAN, f32::INFINITY, f32::NEG_INFINITY][rng.below(3) as usize], unit(rng)]
                    } else {
                        [unit(rng) - 0.5, unit(rng) - 0.5]
                    }
                }
                // the first row onto the last row (the degenerate row pass)
                7 => {
                    if y == 0 {
                        [(rng.below(5) as f32) * 0.5 - 1.0, (h - 1) as f32]
                    } else {
                        [unit(rng) - 0.5, -(yf.min(1.0)) + unit(rng) * 0.5]
                    }
                }
                // the degenerate pass right of the image (the CPU refuses it)
                8 => {
                    if y == 0 {
                        [w as f32 + 5.0, (h - 1) as f32]
                    } else {
                        [0.0, 0.25]
                    }
                }
                // everything collapses towards a point
                9 => [(cx - xf) * 0.9, (cy - yf) * 0.9],
                // a fold over the whole frame (every pixel covered twice) and a shear
                10 => [if y % 2 == 0 { 0.0 } else { 0.7 * fw as f32 }, xf * 0.3],
                // a rotation by about 30 degrees
                _ => [(xf - cx) * -0.134 - (yf - cy) * 0.5, (xf - cx) * 0.5 + (yf - cy) * -0.134],
            };
            let i = (y * fw + x) * 2;
            f.data[i] = v[0];
            f.data[i + 1] = v[1];
        }
    }
    if same && class < 6 {
        // the zero flow early out now and then
        if rng.below(25) == 0 {
            f.data.iter_mut().for_each(|v| *v = 0.0);
        }
    }
    (f, if class == 7 || class == 8 { 1.0 } else { amount })
}

/// The outcome counts of a comparison run.
#[derive(Default, Debug)]
pub(crate) struct Tally {
    pub compared: usize,
    pub samples: usize,
    pub nan_rule: usize,
    pub refused_both: usize,
    pub device_bound: usize,
    /// Inputs on which the CPU function panics (the device must refuse them).
    pub cpu_panics: usize,
    pub first_panic: Option<String>,
    /// The inputs refused at a device bound (the first few).
    pub bound_cases: Vec<String>,
}

/// `forward_warp_with` on the CPU, `None` when it panics.
pub(crate) fn cpu_warp(src: &Image, flow: &FlowField, amount: f32, want_cov: bool) -> Option<(Result<Option<crate::warp::Coverage>, FxError>, Image)> {
    std::panic::catch_unwind(|| {
        let mut out = Image::new(src.width, src.height, src.fmt());
        let r = forward::forward_warp_with(&mut out, src, flow, amount, ForwardOpts::default(), want_cov);
        (r, out)
    })
    .ok()
}

/// Judges a device result for an input the CPU panics on: it must be refused.
pub(crate) fn judge_panic(what: &str, got: Result<(Image, Option<crate::warp::Coverage>), StageError>, t: &mut Tally) {
    match got {
        Err(StageError::Unsupported(_)) => {}
        Ok(_) => panic!("{what}: the CPU panics and the device gives a result"),
        Err(e) => panic!("{what}: the CPU panics, the device error is {e}"),
    }
    t.cpu_panics += 1;
    if t.first_panic.is_none() {
        t.first_panic = Some(what.to_string());
    }
}

/// Judges one stage result against the CPU's: equal bits (two NaNs equal), NaN output only where
/// the CPU has a NaN, the CPU's refusals refused, device bounds counted.
pub(crate) fn judge(
    what: &str,
    cpu: &Result<Option<crate::warp::Coverage>, FxError>,
    cpu_out: &Image,
    got: Result<(Image, Option<crate::warp::Coverage>), StageError>,
    raw: impl FnOnce() -> Option<Image>,
    t: &mut Tally,
) {
    match (cpu, got) {
        (Ok(cc), Ok((img, gc))) => {
            assert_eq!(first_diff(cpu_out, &img), None, "{what}: image differs");
            assert_eq!(cc.as_ref().map(|c| &c.data), gc.as_ref().map(|c| &c.data), "{what}: coverage differs");
            t.compared += 1;
            t.samples += img.sample_count();
        }
        (Ok(_), Err(StageError::NanOutput)) => {
            assert!(crate::device::image::has_nan(cpu_out), "{what}: NaN output where the CPU has none");
            if let Some(r) = raw() {
                let (Samples::F32(a), Samples::F32(b)) = (&cpu_out.samples, &r.samples) else { panic!("{what}: NaN in an integer image") };
                let (diff, _) = compare_floats(a, b);
                assert_eq!(diff, None, "{what}: raw result differs beyond NaN payloads");
            }
            t.nan_rule += 1;
            t.compared += 1;
        }
        (Ok(_), Err(StageError::Unsupported(why))) => {
            assert!(why.contains("device bound") || why.contains("more than the device holds"), "{what}: refused: {why}");
            t.device_bound += 1;
            if t.bound_cases.len() < 40 {
                t.bound_cases.push(what.to_string());
            }
        }
        (Err(e), Err(StageError::Unsupported(_))) => {
            assert!(matches!(e, FxError::Unsupported(_) | FxError::Mismatch(_)), "{what}: CPU error {e}");
            t.refused_both += 1;
        }
        (Err(e), Ok(_)) => panic!("{what}: the CPU refuses ({e}) and the device does not"),
        (c, Err(e)) => panic!("{what}: CPU {:?}, device error {e}", c.as_ref().map(|_| ())),
    }
}

#[test]
fn simulated_warps_match_the_cpu_in_every_order() {
    let mut rng = Rng::new(0x6_F0D);
    let mut t = Tally::default();
    let sizes = [(2usize, 2usize), (3, 2), (2, 3), (4, 4), (5, 3), (7, 7), (9, 16), (16, 9), (17, 16), (33, 20), (64, 37), (64, 48)];
    for (si, &(w, h)) in sizes.iter().enumerate() {
        for class in 0..FLOW_CLASSES {
            let fmt = FMTS[(si + class) % 3];
            let src = corpus::image(&mut rng, w, h, fmt, if class % 4 == 3 { 1 } else { 0 });
            let (flow, amount) = adversarial_flow(&mut rng, w, h, class);
            let want_cov = (si + class) % 2 == 0;
            let mut cpu_out = Image::new(w, h, fmt);
            let cpu = forward::forward_warp_with(&mut cpu_out, &src, &flow, amount, ForwardOpts::default(), want_cov);
            for (n, &seed) in SEEDS.iter().enumerate() {
                if n >= 2 && w * h > 1200 {
                    break;
                }
                let got = sim::sim_forward_warp_with(&cpu_out, &src, &flow, amount, ForwardOpts::default(), want_cov, seed);
                let what = format!("{w}x{h} {fmt:?} class {class} amount {amount} seed {seed:#x}");
                judge(&what, &cpu, &cpu_out, got, || sim::sim_raw(&src, &flow, amount, seed).map(|r| r.0), &mut t);
            }
        }
    }
    println!("simulated forward warps: {t:?}");
    assert!(t.compared >= 250, "{t:?}");
    assert!(t.refused_both >= 1, "{t:?}");
}

#[test]
fn simulated_warps_of_random_corpora_match_the_cpu() {
    let mut rng = Rng::new(0xC0_4A55);
    let mut t = Tally::default();
    for i in 0..150 {
        let (w, h) = corpus::sizes()[i % corpus::sizes().len()];
        let fmt = FMTS[i % 3];
        let src = corpus::image(&mut rng, w, h, fmt, (i % 2) as u32);
        let (fw, fh) = if i % 3 == 0 { (w, h) } else { (2 + (i % 7), 2 + (i % 5)) };
        let flow = corpus::flow(&mut rng, fw, fh, [0.5f32, 2.0, 6.0][i % 3], i % 2 == 0, i % 5 == 0);
        let amount = [1.0f32, 0.5, -0.5, 1.7][i % 4];
        let want_cov = i % 2 == 1;
        let mut cpu_out = Image::new(w, h, fmt);
        let cpu = forward::forward_warp_with(&mut cpu_out, &src, &flow, amount, ForwardOpts::default(), want_cov);
        let seed = SEEDS[1 + i % 3];
        let got = sim::sim_forward_warp_with(&cpu_out, &src, &flow, amount, ForwardOpts::default(), want_cov, seed);
        judge(&format!("random #{i} {w}x{h} {fmt:?}"), &cpu, &cpu_out, got, || sim::sim_raw(&src, &flow, amount, seed).map(|r| r.0), &mut t);
    }
    println!("simulated random warps: {t:?}");
    assert!(t.compared >= 140, "{t:?}");
}

#[test]
fn the_plan_takes_the_cpus_early_outs_and_refusals() {
    let src = Image::new(5, 4, Fmt::U8);
    let flow = FlowField::new(3, 3);
    assert!(matches!(plan(&src, &src, &flow, 1.0, ForwardOpts::default()), Ok(Plan::Copy { .. })));
    let mut moving = FlowField::new(3, 3);
    moving.data[0] = 0.5;
    assert!(matches!(plan(&src, &src, &moving, 1e-6, ForwardOpts::default()), Ok(Plan::Copy { .. })));
    assert!(matches!(plan(&src, &src, &moving, 1.0, ForwardOpts::default()), Ok(Plan::Warp(_))));
    assert!(matches!(plan(&src, &src, &moving, 1.0, ForwardOpts { pyramid: false }), Err(StageError::Unsupported(_))));
    let other = Image::new(4, 4, Fmt::U8);
    assert!(matches!(plan(&other, &src, &moving, 1.0, ForwardOpts::default()), Err(StageError::Unsupported(_))));
    let tiny = Image::new(1, 4, Fmt::U8);
    assert!(matches!(plan(&tiny, &tiny, &moving, 1.0, ForwardOpts::default()), Err(StageError::Unsupported(_))));
    let wide = Image { width: k::FW_MAX_DIM + 1, height: 2, samples: Samples::U8(vec![0; (k::FW_MAX_DIM + 1) * 8]) };
    assert!(matches!(plan(&wide, &wide, &moving, 1.0, ForwardOpts::default()), Err(StageError::Unsupported(_))));
}
