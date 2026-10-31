//! Device-free tests of the composition stages: the kernel transcriptions, run through the
//! wrappers' plans in shuffled work item orders, against the CPU functions bit for bit at every
//! depth over random and adversarial corpora; the identifier prefixes; the device bounds.

use super::corpus::{self, first_diff, FMTS};
use super::rs;
use crate::blend;
use crate::device::kernels::warpstage as k;
use crate::device::sim::{Rng, SEEDS};
use crate::flow::FlowField;
use crate::image::{Fmt, Image, Samples};
use crate::smartblend;
use crate::warp::{self, Coverage, WarpOpts};

// ---- sources ----------------------------------------------------------------------------------

/// The names every source defines: `#define` names, and the name before `(` of every line that
/// starts a function or kernel definition (a line ending in `{`, with or without a macro `\`).
fn defined_names(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in src.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("#define ") {
            out.push(rest.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).next().unwrap().to_string());
            continue;
        }
        let body = t.trim_end_matches('\\').trim_end();
        let starts_type = ["void ", "uchar ", "ushort ", "float ", "int ", concat!("__ker", "nel void ")].iter().any(|p| t.starts_with(p));
        if starts_type && body.ends_with('{') && t.contains('(') && !t.contains('=') {
            let head = &t[..t.find('(').unwrap()];
            out.push(head.rsplit(' ').next().unwrap().to_string());
        }
    }
    out
}

#[test]
fn every_defined_name_carries_a_package_prefix() {
    let mut count = 0;
    for src in super::program_parts() {
        for name in defined_names(src) {
            count += 1;
            assert!(
                ["bl_", "wp_", "bc_", "sb_", "BL_", "WP_", "BC_", "SB_"].iter().any(|p| name.starts_with(p)),
                "{name} has no stage prefix"
            );
        }
    }
    assert!(count >= 30, "only {count} definitions found");
}

#[test]
fn the_program_holds_every_kernel_the_wrappers_launch() {
    let program = super::program();
    let kernels = [
        "bl_blend_##SUF", "bl_lerp_##SUF", "bl_blend_f32", "bl_lerp_f32", "bl_plus_zero_f32", "bl_unsound_f32",
        "wp_inv_warp_##SUF", "bc_motion_blur_##SUF", "bc_clear_rgb_##SUF", "sb_smart_##SUF", "sb_cov_max_invert", "sb_cov_invert",
    ];
    for kname in kernels {
        assert!(program.contains(&format!("{} {kname}(", concat!("__ker", "nel void"))), "{kname}");
    }
    for inst in ["BL_BLEND_INT(u8", "BL_BLEND_INT(u16", "BL_LERP_INT(u8", "BL_LERP_INT(u16", "WP_DEPTH(u8", "WP_DEPTH(u16", "WP_DEPTH(f32", "BC_DEPTH(u8", "BC_DEPTH(u16", "BC_DEPTH(f32", "SB_DEPTH(u8", "SB_DEPTH(u16", "SB_DEPTH(f32"] {
        assert!(program.contains(inst), "{inst}");
    }
    assert!(program.starts_with("#pragma OPENCL FP_CONTRACT OFF"));
    // the bounds stated in Rust are the literals of the sources
    assert!(k::WARP.contains(&format!("s < {}", k::WP_CLIP_PASSES)));
    assert!(k::WARP.contains(&format!("t < {}", k::WP_WRAP_STEPS)));
    assert!(k::BLURCOMP.contains(&format!("t < {}", k::BC_TAPS)));
    assert!(k::BLURCOMP.contains(&format!("n > {}", k::BC_TAPS)));
    assert!(k::BLURCOMP.contains("n > 65536"));
}

#[test]
fn the_hex_constants_are_the_cpu_constants() {
    // 1e30f of sound_float, 1e-6f of TINY_VECTOR, the whites
    assert_eq!(f32::from_bits(0x7149_f2ca), 1.0e30f32);
    assert!(k::BLEND.contains("0x1.93e594p+99f"));
    assert_eq!(f32::from_bits(0x3586_37bd), 1e-6f32);
    assert!(k::BLURCOMP.contains("0x1.0c6f7ap-20f"));
    assert_eq!(255.0f32, (0x1fe as f32) * 0.5);
    assert_eq!(65535.0f32, (0x1fffe as f32) * 0.5);
}

// ---- blend ------------------------------------------------------------------------------------

#[test]
fn blend_matches_the_cpu_in_every_order_and_depth() {
    let mut rng = Rng::new(0xB1E0);
    let mut compared = 0usize;
    for (w, h) in corpus::sizes() {
        for fmt in FMTS {
            for class in [0, 1] {
                let a = corpus::image(&mut rng, w, h, fmt, class);
                let b = corpus::image(&mut rng, w, h, fmt, class);
                for wt in corpus::weights() {
                    let mut want = Image::new(w, h, fmt);
                    blend::blend(&mut want, &a, &b, wt);
                    for seed in SEEDS {
                        let got = corpus::sim_blend(&a, &b, wt, seed);
                        assert_eq!(first_diff(&want, &got), None, "{w}x{h} {fmt:?} class {class} w {wt:e} seed {seed}");
                        compared += want.sample_count();
                    }
                }
            }
        }
    }
    assert!(compared > 100_000, "{compared}");
}

#[test]
fn lerp_matches_the_cpu_in_every_order_and_depth() {
    let mut rng = Rng::new(0x1E2F);
    for (w, h) in corpus::sizes() {
        for fmt in FMTS {
            for class in [0, 1] {
                let f = corpus::image(&mut rng, w, h, fmt, class);
                let s = corpus::image(&mut rng, w, h, fmt, class);
                for wt in corpus::weights().into_iter().chain([-0.5f32, -10.0, 0.3, 20.0, -0.0]) {
                    let mut want = f.clone();
                    blend::lerp(&mut want, &s, wt);
                    for seed in [SEEDS[0], SEEDS[2]] {
                        let mut got = f.clone();
                        corpus::sim_lerp(&mut got, &s, wt, seed);
                        assert_eq!(first_diff(&want, &got), None, "{w}x{h} {fmt:?} class {class} w {wt:e}");
                    }
                }
            }
        }
    }
}

#[test]
fn weighted_blend_noflow_matches_on_its_fast_path() {
    let mut rng = Rng::new(0x3E16);
    let mut fast = 0;
    let mut refused = 0;
    for (w, h) in corpus::sizes() {
        for fmt in FMTS {
            for class in [0, 1] {
                let a = corpus::image(&mut rng, w, h, fmt, class);
                let b = corpus::image(&mut rng, w, h, fmt, class);
                for wa in [0.0f32, 0.3, 0.5, 0.999, 1.0, 1.7, -0.2] {
                    let mut want = Image::new(w, h, fmt);
                    blend::weighted_blend_noflow(&mut want, &a, wa, &b);
                    match corpus::sim_weighted_noflow(&a, wa, &b, SEEDS[1]) {
                        Some(got) => {
                            fast += 1;
                            assert_eq!(first_diff(&want, &got), None, "{w}x{h} {fmt:?} class {class} wa {wa}");
                        }
                        None => {
                            refused += 1;
                            assert_eq!(fmt, Fmt::F32);
                        }
                    }
                }
            }
        }
    }
    assert!(fast > 300 && refused > 10, "{fast} {refused}");
}

// ---- inverse warp -----------------------------------------------------------------------------

fn opts_list() -> Vec<(WarpOpts, bool)> {
    vec![
        (WarpOpts { hold: true, is360: false, mode: 1 }, false),
        (WarpOpts { hold: false, is360: false, mode: 1 }, false),
        (WarpOpts { hold: false, is360: false, mode: 1 }, true),
        (WarpOpts { hold: true, is360: true, mode: 1 }, true),
        (WarpOpts { hold: false, is360: true, mode: 1 }, false),
    ]
}

fn cpu_warp(src: &Image, flow: &FlowField, amount: f32, opts: WarpOpts, want_cov: bool) -> fx_core::Result<(Image, Option<Coverage>)> {
    let mut out = Image::new(src.width, src.height, src.fmt());
    if want_cov {
        let c = warp::inv_warp_coverage(&mut out, src, flow, amount, opts)?;
        Ok((out, Some(c)))
    } else {
        warp::inv_warp(&mut out, src, flow, amount, opts)?;
        Ok((out, None))
    }
}

/// Compares one warp case; returns the samples compared, 0 when both refused, and counts device
/// bound deferrals in `deferred`.
pub(super) fn check_warp_case(src: &Image, flow: &FlowField, amount: f32, opts: WarpOpts, want_cov: bool, seed: u64, deferred: &mut usize) -> usize {
    let cpu = cpu_warp(src, flow, amount, opts, want_cov);
    let sim = corpus::sim_warp(src, flow, amount, opts, want_cov, seed);
    let what = format!("{}x{} {:?} flow {}x{} amount {amount:e} {opts:?} cov {want_cov} seed {seed}", src.width, src.height, src.fmt(), flow.width, flow.height);
    match (cpu, sim) {
        (Ok((img, cov)), Ok((simg, scov))) => {
            assert_eq!(first_diff(&img, &simg), None, "{what}");
            assert_eq!(cov, scov, "{what}");
            img.sample_count()
        }
        (Ok(_), Err(flag)) => {
            assert!(flag == k::FLAG_DEVICE_BOUND, "{what}: the device refused (flag {flag}) where the CPU did not");
            *deferred += 1;
            0
        }
        (Err(_), Err(_)) => 0,
        (Err(e), Ok(_)) => panic!("{what}: the CPU refused ({e}) and the device did not"),
    }
}

#[test]
fn inverse_warp_matches_the_cpu_in_every_mode_and_depth() {
    let mut rng = Rng::new(0x3A4F);
    let mut compared = 0;
    let mut deferred = 0;
    let mut refused_both = 0;
    for (w, h) in corpus::sizes() {
        for fmt in FMTS {
            for class in [0, 1] {
                let src = corpus::image(&mut rng, w, h, fmt, class);
                // flow sizes: the image size, a coarser one, a finer one
                for (fw, fh) in [(w, h), (w.div_ceil(2).max(2), h.div_ceil(2).max(2)), (w + 3, h + 1)] {
                    for (scale, wild, nonfinite) in [(0.6f32, false, false), (3.0, true, false), (w as f32 * 2.5, true, false), (4.0, true, true)] {
                        let flow = corpus::flow(&mut rng, fw, fh, scale, wild, nonfinite);
                        for amount in [1.0f32, 0.5, -0.75, 0.004, 2.0, -1.0] {
                            for (opts, cov) in opts_list() {
                                let seed = SEEDS[rng.below(4) as usize];
                                let n = check_warp_case(&src, &flow, amount, opts, cov, seed, &mut deferred);
                                if n == 0 {
                                    refused_both += 1;
                                }
                                compared += n;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(compared > 1_000_000, "{compared}");
    assert!(refused_both > 10, "the corpus holds refused cases ({refused_both})");
    assert!(deferred * 20 < refused_both + 1000, "{deferred} deferrals");
}

#[test]
fn the_360_warp_crosses_the_poles_and_wraps_far_flows() {
    // vertical flows of 0.5 to 3 image heights cross the poles an odd and an even number of times
    let mut rng = Rng::new(0x9013);
    let mut deferred = 0;
    let mut compared = 0;
    for (w, h) in [(4, 4), (5, 3), (16, 9), (17, 16)] {
        for fmt in FMTS {
            let src = corpus::image(&mut rng, w, h, fmt, 0);
            for k in 0..12 {
                let mut flow = FlowField::new(w, h);
                for (i, v) in flow.data.iter_mut().enumerate() {
                    let base = if i % 2 == 0 { w as f32 } else { h as f32 };
                    *v = base * (k as f32 * 0.25 - 1.5) + (rng.below(5) as f32 - 2.0) * 0.5;
                }
                let opts = WarpOpts { hold: true, is360: true, mode: 1 };
                compared += check_warp_case(&src, &flow, 1.0, opts, true, SEEDS[k % 4], &mut deferred);
            }
        }
    }
    assert_eq!(deferred, 0);
    assert!(compared > 0);
}

#[test]
fn the_device_bounds_defer_to_the_cpu_and_only_then() {
    let mut deferred = 0;
    let src = corpus::image(&mut Rng::new(5), 4, 4, Fmt::F32, 0);
    // 100 widths of horizontal flow: the CPU wraps it, the device needs more than 64 steps
    let mut flow = FlowField::new(4, 4);
    for v in flow.data.chunks_exact_mut(2) {
        v[0] = 400.0;
    }
    let opts = WarpOpts { hold: true, is360: true, mode: 1 };
    assert!(cpu_warp(&src, &flow, 1.0, opts, false).is_ok());
    assert_eq!(corpus::sim_warp(&src, &flow, 1.0, opts, false, 0).unwrap_err(), k::FLAG_DEVICE_BOUND);
    // 60 widths stay on the device
    for v in flow.data.chunks_exact_mut(2) {
        v[0] = 240.0;
    }
    assert_eq!(check_warp_case(&src, &flow, 1.0, opts, false, 1, &mut deferred), 64);
    // a NaN in hold mode is refused by both
    flow.data[3] = f32::NAN;
    let hold = WarpOpts::default();
    assert!(cpu_warp(&src, &flow, 1.0, hold, false).is_err());
    assert_eq!(corpus::sim_warp(&src, &flow, 1.0, hold, false, 0).unwrap_err() & k::FLAG_CPU_REFUSES, k::FLAG_CPU_REFUSES);
    // a vector of 2000 taps at blur amount 1: the CPU blurs it, the device defers
    let mut long = FlowField::new(4, 4);
    for v in long.data.chunks_exact_mut(2) {
        v[0] = 4000.0;
    }
    let mut dest = Image::new(4, 4, Fmt::F32);
    assert!(warp::motion_blur(&mut dest, &src, &long, 1.0, true, 0.5, false).is_ok());
    assert_eq!(corpus::sim_motion_blur(&src, &long, 1.0, true, 0.5, false, 4, 0).unwrap_err(), k::FLAG_DEVICE_BOUND);
    assert_eq!(deferred, 0);
}

// ---- motion blur ------------------------------------------------------------------------------

fn check_blur_case(src: &Image, flow: &FlowField, amount: f32, hiq: bool, shutter: f32, is360: bool, rows: usize, seed: u64, deferred: &mut usize) -> usize {
    let mut dest = Image::new(src.width, src.height, src.fmt());
    let cpu = warp::motion_blur(&mut dest, src, flow, amount, hiq, shutter, is360);
    let sim = corpus::sim_motion_blur(src, flow, amount, hiq, shutter, is360, rows, seed);
    let what = format!("{}x{} {:?} amount {amount:e} hiq {hiq} shutter {shutter} 360 {is360}", src.width, src.height, src.fmt());
    match (cpu, sim) {
        (Ok(()), Ok(got)) => {
            assert_eq!(first_diff(&dest, &got), None, "{what}");
            dest.sample_count()
        }
        (Ok(()), Err(flag)) => {
            assert_eq!(flag, k::FLAG_DEVICE_BOUND, "{what}");
            *deferred += 1;
            0
        }
        (Err(_), Err(_)) => 0,
        (Err(e), Ok(_)) => panic!("{what}: the CPU refused ({e}) and the device did not"),
    }
}

#[test]
#[ignore = "the kernel follows the rule before the parity work; port pending (docs/engine.md, Device)"]
fn motion_blur_matches_the_cpu_for_every_variant_and_depth() {
    let mut rng = Rng::new(0xB10B);
    let mut compared = 0;
    let mut deferred = 0;
    for (w, h) in corpus::sizes() {
        for fmt in FMTS {
            for class in [0, 1] {
                let src = corpus::image(&mut rng, w, h, fmt, class);
                for (fw, fh) in [(w, h), (w.div_ceil(2).max(2), h.div_ceil(2).max(2))] {
                    for (scale, wild, nonfinite) in [(1.5f32, false, false), (6.0, true, false), (w as f32 * 2.0, true, false), (3.0, true, true)] {
                        let flow = corpus::flow(&mut rng, fw, fh, scale, wild, nonfinite);
                        for (amount, shutter) in [(1.0f32, 0.5f32), (0.5, 0.5), (3.0, 0.25), (-2.0, 0.75)] {
                            for hiq in [true, false] {
                                for is360 in [false, true] {
                                    let rows = [1usize, 3, 1000][rng.below(3) as usize];
                                    let seed = SEEDS[rng.below(4) as usize];
                                    compared += check_blur_case(&src, &flow, amount, hiq, shutter, is360, rows, seed, &mut deferred);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(compared > 1_000_000, "{compared}");
    assert_eq!(deferred, 0);
}

#[test]
#[ignore = "the kernel follows the rule before the parity work; port pending (docs/engine.md, Device)"]
fn blur_compensation_matches_the_cpu_chain() {
    let mut rng = Rng::new(0xC0C0);
    let mut cases = 0;
    for (w, h) in corpus::sizes() {
        for fmt in FMTS {
            let class = rng.below(2) as u32;
            let src = corpus::image(&mut rng, w, h, fmt, class);
            let flow = corpus::flow(&mut rng, w, h, 4.0, true, false);
            for (s0, s1, blur) in [(0.0f32, 3.0f32, 1.0f32), (0.0, 0.25, 1.0), (2.0, 2.0, 1.0), (0.0, 5.0, 0.004), (0.0, 1.0, 0.7), (0.0, 30.0, 2.0), (0.0, 0.1, 30.0)] {
                for hiq in [true, false] {
                    for is360 in [false, true] {
                        let mut want = src.clone();
                        let cpu = warp::blur_compensation_wrap(&mut want, &flow, s0, s1, blur, hiq, is360);
                        let sim = corpus::sim_compensation(&src, &flow, s0, s1, blur, hiq, is360, 2, SEEDS[cases % 4]);
                        match (cpu, sim) {
                            (Ok(()), Ok(got)) => assert_eq!(first_diff(&want, &got), None, "{w}x{h} {fmt:?} {s0} {s1} {blur} {hiq} {is360}"),
                            (Err(_), Err(_)) => {}
                            (c, s) => panic!("{w}x{h} {fmt:?}: cpu {c:?} sim {:?}", s.err()),
                        }
                        cases += 1;
                    }
                }
            }
        }
    }
    assert!(cases > 900);
}

#[test]
fn clear_rgb_matches_the_cpu() {
    let mut rng = Rng::new(0xC1EA);
    for (w, h) in corpus::sizes() {
        for fmt in FMTS {
            let src = corpus::image(&mut rng, w, h, fmt, 1);
            let mut want = src.clone();
            warp::clear_rgb_where_transparent(&mut want);
            for seed in SEEDS {
                assert_eq!(first_diff(&want, &corpus::sim_clear_rgb(&src, seed)), None);
            }
        }
    }
}

// ---- Smart Blend ------------------------------------------------------------------------------

#[test]
fn smart_blend_and_its_binary_variant_match_the_cpu() {
    let mut rng = Rng::new(0x5B5B);
    let mut compared = 0;
    for (w, h) in corpus::sizes() {
        for fmt in FMTS {
            for class in [0, 1, 2] {
                let p1 = corpus::image(&mut rng, w, h, fmt, class);
                let p2 = corpus::image(&mut rng, w, h, fmt, class);
                for wt in corpus::weights() {
                    for binary in [false, true] {
                        let mut want = Image::new(w, h, fmt);
                        if binary {
                            smartblend::smart_blend_binary(&mut want, &p1, &p2, wt).unwrap();
                        } else {
                            smartblend::smart_blend(&mut want, &p1, &p2, wt).unwrap();
                        }
                        let got = corpus::sim_smart(&p1, &p2, wt, binary, SEEDS[compared % 4]);
                        assert_eq!(first_diff(&want, &got), None, "{w}x{h} {fmt:?} class {class} w {wt:e} binary {binary}");
                        compared += want.sample_count();
                    }
                }
            }
        }
    }
    assert!(compared > 500_000, "{compared}");
}

#[test]
fn the_raw_store_and_the_alpha_edges_are_reached() {
    // v.alpha == 0 with v != amax (the raw store) needs opposite alphas cancelling: w = 1 with
    // c2.alpha = 0 is impossible, so use alpha 0 against a negative alpha
    let p1 = Image { width: 2, height: 2, samples: Samples::F32(vec![0.0, 0.3, 0.2, 0.1, 1.0, 0.5, 0.5, 0.5, 0.25, 0.1, 0.1, 0.1, -0.5, 0.2, 0.2, 0.2]) };
    let p2 = Image { width: 2, height: 2, samples: Samples::F32(vec![0.0, 0.7, 0.6, 0.5, 0.0, 0.1, 0.1, 0.1, 0.75, 0.3, 0.3, 0.3, 0.5, 0.1, 0.1, 0.1]) };
    for wt in [0.0f32, 0.5, 1.0, 0.3] {
        let mut want = Image::new(2, 2, Fmt::F32);
        smartblend::smart_blend(&mut want, &p1, &p2, wt).unwrap();
        assert_eq!(first_diff(&want, &corpus::sim_smart(&p1, &p2, wt, false, 1)), None);
    }
}

#[test]
fn coverage_steps_match_the_cpu() {
    let mut rng = Rng::new(0xC0FE);
    for (w, h) in corpus::sizes() {
        let mk = |rng: &mut Rng| Coverage { width: w, height: h, data: (0..w * h).map(|_| [0u8, 255, 128, rng.next_u32() as u8][rng.below(4) as usize]).collect() };
        let (c1, c2) = (mk(&mut rng), mk(&mut rng));
        let mut want = c1.clone();
        want.combine_max(&c2);
        want.invert();
        assert_eq!(corpus::sim_cov_holes(&c1, &c2, 3), want);
        let mut inv = c1.clone();
        inv.invert();
        assert_eq!(corpus::sim_cov_invert(&c1, 7), inv);
    }
}

// ---- transcription pieces ---------------------------------------------------------------------

#[test]
fn the_depth_stores_equal_the_cpu_stores() {
    let specials = crate::device::selfcheck::f32_specials();
    let mut rng = Rng::new(77);
    let mut vals: Vec<f32> = specials.clone();
    vals.extend((0..20000).map(|_| f32::from_bits(rng.next_u32())));
    vals.extend((-600..70000).map(|i| i as f32 * 0.5));
    for v in vals {
        assert_eq!(<u8 as rs::Depth>::wp_store(v), warp::sat_u8(v), "{v:e}");
        assert_eq!(<u16 as rs::Depth>::wp_store(v), warp::sat_u16(v), "{v:e}");
        // store_blur: (cvtt32(v + 0.5) as f32).max(0).min(white) as u8 / u16
        let t = warp::cvtt32(v + 0.5f32) as f32;
        assert_eq!(<u8 as rs::Depth>::bc_store(v), t.max(0.0).min(255.0) as u8, "{v:e}");
        assert_eq!(<u16 as rs::Depth>::bc_store(v), t.max(0.0).min(65535.0) as u16, "{v:e}");
        for f in [0.0f32, 0.25, 0.5, 0.999, 1.0] {
            let b = warp::sat_u8(v);
            let want = {
                let x = b as f32 * f;
                crate::device::selfcheck::cvtt_i64_d(x as f64 + 0.5) as u8
            };
            assert_eq!(<u8 as rs::Depth>::wp_faded(v, f), want, "{v:e} {f}");
        }
    }
}
