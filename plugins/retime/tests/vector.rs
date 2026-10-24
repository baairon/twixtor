//! The vector path against an ANALYTIC ground truth (`common/moving.rs`): a smooth texture that
//! translates by a known number of pixels per frame, so the exact in-between frame at any
//! fractional source position is known in closed form and the result can be judged against it
//! directly.
//!
//! The effect runs with its defaults (Vectors Best, Smart Blend off, Blur Compensation 0) except
//! the speed and the Frame Interp mode. The same clip is rendered once more with `vectors=1` (No
//! Vectors), which is the plain frame blend (Blend, Weighted Blend) or the nearest frame (Nearest);
//! that render is the baseline the flow result must beat.
//!
//! Metrics, per depth and per mode: the mean absolute colour error against the analytic frame
//! (fraction of full scale, pixels at least 10 px from every edge, content that enters from outside
//! cannot be reconstructed), the same for the baseline, and the error against the frame at the
//! MIRRORED position (a wrong motion direction would be about as good as the right one on that
//! frame, so the right direction is asserted by a large margin).
//!
//! Measured (release build, 96 x 64, 8 frame clips, 3 to 7 interpolated source positions per case;
//! the worst mean error and the smallest improvement over the baseline of each depth and mode,
//! printed by the main test with `--nocapture`): U8 Blend 0.0023 (6.0 times closer than the plain
//! blend), Nearest 0.0028 (13.0 times), Weighted Blend 0.0032 (4.5 times; the unsharp step costs
//! accuracy on a smooth texture by design); U16 and F32 are 10 to 20 percent better. The baseline
//! errors are 0.014 to 0.070 at 3 to 4 px per frame. The asserted margins are 1.3 to 1.8 times
//! looser than these.

mod common;

use common::moving::*;
use common::*;
use fx_core::Depth;

const W: usize = 96;
const H: usize = 64;
const MARGIN: usize = 10;

#[derive(Clone, Copy, Debug)]
enum Mode {
    Nearest,
    Blend,
    Weighted,
}

impl Mode {
    /// The popup value (1 based).
    fn popup(self) -> &'static str {
        match self {
            Mode::Nearest => "1",
            Mode::Blend => "2",
            Mode::Weighted => "3",
        }
    }

    /// Largest mean error against the analytic frame, as a fraction of full scale.
    fn max_error(self) -> f64 {
        match self {
            Mode::Nearest | Mode::Blend => 0.0045,
            Mode::Weighted => 0.0055,
        }
    }

    /// Smallest ratio of the baseline error to the flow error for a motion of at least 2.5 px per frame.
    fn min_ratio(self) -> f64 {
        match self {
            Mode::Nearest => 10.0,
            Mode::Blend => 4.5,
            Mode::Weighted => 3.5,
        }
    }
}

const MOTIONS: [Motion; 4] = [DEFAULT, Motion { vx: -2.6, vy: 2.9 }, Motion { vx: 0.0, vy: -3.8 }, Motion { vx: 4.4, vy: 0.0 }];

/// One case: the worst flow error, the smallest ratio against the baseline and the smallest margin against the mirrored
/// frame over the interpolated output frames of the run.
struct Result3 {
    worst_error: f64,
    worst_max: f64,
    min_ratio: f64,
    min_vs_mirrored: f64,
    frames: usize,
}

fn run_case(depth: Depth, motion: Motion, speed: &str, mode: Mode, extra: &[(&str, &str)]) -> Result3 {
    let clip = motion.clip(depth, W, H, 8);
    let mut params = vec![("speed_percent", speed), ("frame_interp", mode.popup())];
    params.extend_from_slice(extra);
    let rig = Rig::new(clip.clone(), &params).with_frames(10);
    let mut plain_params = params.clone();
    plain_params.push(("vectors", "1"));
    let plain = Rig::new(clip, &plain_params).with_frames(10);
    let step = speed.parse::<f64>().unwrap() / 100.0;
    let mut r = Result3 { worst_error: 0.0, worst_max: 0.0, min_ratio: f64::MAX, min_vs_mirrored: f64::MAX, frames: 0 };
    for n in 1..8u64 {
        let s = (step as f32 * n as f32) as f64;
        if (s - s.round()).abs() < 0.01 {
            continue; // an exact frame: no interpolation happens
        }
        let truth = motion.frame_at(depth, W, H, s);
        let mirrored = motion.frame_at(depth, W, H, -s);
        let got = rig.render(n).unwrap();
        let (ef, mf) = error(&got, &truth, MARGIN);
        let (eb, _) = error(&plain.render(n).unwrap(), &truth, MARGIN);
        let (em, _) = error(&got, &mirrored, MARGIN);
        r.worst_error = r.worst_error.max(ef);
        r.worst_max = r.worst_max.max(mf);
        r.min_ratio = r.min_ratio.min(eb / ef.max(1e-9));
        r.min_vs_mirrored = r.min_vs_mirrored.min(em / ef.max(1e-9));
        r.frames += 1;
    }
    r
}

fn fast(m: Motion) -> bool {
    m.vx.abs().max(m.vy.abs()) >= 2.5
}

#[test]
fn the_flow_result_is_close_to_the_analytic_frame_and_beats_the_plain_blend_at_every_depth_and_mode() {
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        for mode in [Mode::Blend, Mode::Nearest, Mode::Weighted] {
            let (mut worst, mut ratio) = (0.0f64, f64::MAX);
            for motion in MOTIONS {
                for speed in ["50", "25", "75"] {
                    if matches!(mode, Mode::Weighted) && speed != "50" {
                        continue; // the other speeds repeat the same pair of frames at other weights
                    }
                    let r = run_case(depth, motion, speed, mode, &[]);
                    assert!(r.frames >= 2, "{depth:?} {mode:?} {motion:?} speed {speed}: only {} interpolated frames", r.frames);
                    let case = format!("{depth:?} {mode:?} {motion:?} speed {speed}");
                    assert!(r.worst_error < mode.max_error(), "{case}: error {:.5} (max {:.4})", r.worst_error, r.worst_max);
                    assert!(r.worst_max < 0.04, "{case}: worst pixel {:.4}", r.worst_max);
                    assert!(r.min_ratio > 1.0 && (!fast(motion) || r.min_ratio >= mode.min_ratio()), "{case}: only {:.1} times closer than the plain result", r.min_ratio);
                    assert!(r.min_vs_mirrored > 10.0, "{case}: the mirrored motion is only {:.1} times worse", r.min_vs_mirrored);
                    worst = worst.max(r.worst_error);
                    ratio = ratio.min(r.min_ratio);
                }
            }
            eprintln!("{depth:?} {mode:?}: worst mean error {worst:.5}, smallest improvement {ratio:.1}x");
        }
    }
}

#[test]
fn a_slow_motion_is_still_tracked_and_never_made_worse() {
    // 1.2 and 0.7 px per frame: the plain blend is already good, the flow result must not be worse than it
    let motion = Motion { vx: -1.2, vy: -0.7 };
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        for mode in [Mode::Blend, Mode::Nearest] {
            let r = run_case(depth, motion, "50", mode, &[]);
            assert!(r.worst_error < 0.004, "{depth:?} {mode:?}: error {:.5}", r.worst_error);
            assert!(r.min_ratio > 1.0, "{depth:?} {mode:?}: the flow result is worse than the plain one ({:.2})", r.min_ratio);
            assert!(r.min_vs_mirrored > 8.0, "{depth:?} {mode:?}: {:.1}", r.min_vs_mirrored);
        }
    }
}

#[test]
fn the_video_range_checkbox_estimates_from_the_red_channel_and_still_tracks() {
    // "360 Video" of plugin 01 selects the video range luma of the solver (red channel only), not a 360 warp
    for depth in [Depth::U8, Depth::F32] {
        for motion in [DEFAULT, Motion { vx: 0.0, vy: -3.8 }] {
            let r = run_case(depth, motion, "50", Mode::Blend, &[("video_360", "1")]);
            assert!(r.worst_error < 0.006 && r.min_ratio >= 4.0 && r.min_vs_mirrored > 8.0, "{depth:?} {motion:?}: {:.5} {:.1} {:.1}", r.worst_error, r.min_ratio, r.min_vs_mirrored);
        }
    }
}

#[test]
fn the_lower_flow_qualities_track_a_coarser_but_still_correct_motion() {
    // Vectors popup 2 and 3 (Sloppy, Medium) solve at the same resolution, 1/8 of the image (12 x 8 at 96 x 64),
    // popup 4 (High) at 1/4: the flow is coarser, the result is still much closer to the truth than the plain blend
    for (vectors, limit, ratio) in [("2", 0.012, 1.8), ("3", 0.012, 1.8), ("4", 0.004, 5.0)] {
        let r = run_case(Depth::U8, DEFAULT, "50", Mode::Blend, &[("vectors", vectors)]);
        eprintln!("vectors {vectors}: error {:.5}, {:.1}x closer than plain, {:.1}x better than mirrored", r.worst_error, r.min_ratio, r.min_vs_mirrored);
        assert!(r.worst_error < limit && r.min_ratio > ratio && r.min_vs_mirrored > 6.0, "vectors {vectors}: {:.5} {:.1} {:.1}", r.worst_error, r.min_ratio, r.min_vs_mirrored);
    }
}

#[test]
fn blur_compensation_approaches_the_shutter_blur_of_the_motion_as_the_amount_grows() {
    // Speed % 300 advances 3 source frames per output frame; with Blur Compensation b the amount is
    // (3 - 1) * b, the motion blur runs along the flow with taps spanning +-0.5 * amount source
    // frames of motion. The blur uses a handful of discrete taps (a 3 tap box at amount 1, which
    // blurs about twice as much as the continuous box), so the result is only compared with the
    // continuous shutter (the mean of the exact frames over that window) where many taps make the
    // two agree: amounts 2 and 4. Measured at U8: residual 0.0075 to 0.0082 against 0.025 to 0.031
    // for the sharp frame at amount 2, 0.0035 to 0.0053 against 0.067 to 0.088 at amount 4. A wrong
    // direction or size gives a residual as large as the sharp frame's.
    for depth in [Depth::U8, Depth::F32] {
        for motion in [DEFAULT, Motion { vx: -2.6, vy: 2.9 }, Motion { vx: 0.0, vy: -3.8 }] {
            for (blur, amount, max_ratio) in [("1", 2.0f64, 0.40f64), ("2", 4.0, 0.12)] {
                let rig = Rig::new(motion.clip(depth, W, H, 12), &[("speed_percent", "300"), ("blur_compensation", blur)]).with_frames(4);
                for n in 1..4u64 {
                    let s = 3.0 * n as f64;
                    let got = rig.render(n).unwrap();
                    let (residual, _) = error(&got, &motion.blurred_frame_at(depth, W, H, s, 0.5 * amount, 64), MARGIN);
                    let (unblurred, _) = error(&motion.frame_at(depth, W, H, s), &motion.blurred_frame_at(depth, W, H, s, 0.5 * amount, 64), MARGIN);
                    assert!(residual < max_ratio * unblurred, "{depth:?} {motion:?} blur {blur} frame {n}: residual {residual:.4}, sharp frame {unblurred:.4}");
                }
            }
        }
    }
}

#[test]
fn slow_motion_blur_compensation_runs_on_the_flow_and_keeps_the_frame_close_to_the_truth() {
    // below 100 percent the amount is negative: a flow directed unsharp mask. It changes the frame, and on this smooth
    // pattern the result stays within 0.004 of the analytic frame (without the slider the error is 0.0016 to 0.0022)
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let clip = DEFAULT.clip(depth, W, H, 8);
        let with = Rig::new(clip.clone(), &[("speed_percent", "50"), ("blur_compensation", "1")]).with_frames(8);
        let without = Rig::new(clip, &[("speed_percent", "50")]).with_frames(8);
        for n in [1u64, 2, 3, 5] {
            let (a, b) = (with.render(n).unwrap(), without.render(n).unwrap());
            assert_ne!(a, b, "{depth:?} frame {n}: the slider changes nothing");
            let (e, _) = error(&a, &DEFAULT.frame_at(depth, W, H, n as f64 / 2.0), MARGIN);
            assert!(e < 0.004, "{depth:?} frame {n}: error {e:.5}");
        }
    }
}

#[test]
fn a_still_clip_stays_still() {
    // a clip that does not move: every in-between frame equals the frame itself (zero flow), at every depth
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let still = Motion { vx: 0.0, vy: 0.0 };
        let rig = Rig::new(still.clip(depth, W, H, 6), &[("speed_percent", "50")]).with_frames(8);
        let truth = still.frame_at(depth, W, H, 0.0);
        for n in 1..8u64 {
            let got = rig.render(n).unwrap();
            let (e, m) = error(&got, &truth, MARGIN);
            assert!(e < 0.0005 && m < 0.004, "{depth:?} frame {n}: {e} {m}");
        }
    }
}

// ---- Smart Blend (warping code 1) on a scene with an occlusion -------------------------------------------------

/// The regions of the occlusion scene at output position `s` (a pixel is `(x, y)` of a 96 x 64 frame): the border strips
/// where content enters the picture (8 px wide), the occlusion band (pixels the rectangle covers in exactly one of the two
/// source frames around `s`, away from the borders) and everything.
struct Regions {
    k: f64,
}

impl Regions {
    fn strip(&self, x: usize, y: usize) -> bool {
        x < 8 || x >= W - 8 || y < 8 || y >= H - 8
    }

    fn band(&self, x: usize, y: usize) -> bool {
        !self.strip(x, y) && OCCLUSION.covers(x, y, self.k) != OCCLUSION.covers(x, y, self.k + 1.0)
    }
}

/// Error in 8 bit levels (the mean absolute difference) of a result against the exact scene in a region.
fn levels(got: &fx_core::Frame, truth: &fx_core::Frame, region: impl Fn(usize, usize) -> bool) -> f64 {
    error_in(got, truth, region) * 255.0
}

/// Smart Blend (the Smart Blend checkbox, warping code 1) against the same render without it (warping code 0, hold
/// edge warps; "the plain flow blend") and against the frame blend, on a background panning by (8, 2) px per frame
/// with a rectangle moving by (-12, 5) over it. The solver's own flows (Best), half speed, output frames 1, 3 and 5.
///
/// Measured in 8 bit levels, mean absolute error, release build (U8, U16 and F32 agree to 0.1 level):
///
/// | mode | strips: plain / flow / Smart | occlusion band: plain / flow / Smart | whole frame: plain / flow / Smart |
/// |---|---|---|---|
/// | Blend | 12.8 to 17.7 / 3.1 to 4.1 / 1.3 to 1.8 | 37 to 38 / 33 to 42 / the same as flow | 19.4 to 20.7 / 9.5 to 10.7 / 8.8 to 9.8 |
/// | Weighted Blend | the same / 3.3 to 4.5 / 1.7 to 2.2 | the same | the same / 9.9 to 11.1 / 9.3 to 10.2 |
/// | Nearest | 23.6 to 33.3 / 3.9 to 5.7 / 5.2 to 6.5 | 41 to 50 / 37 to 46 / the same as flow | 32 to 35 / 11.4 to 11.9 / 11.7 to 12.3 |
///
/// The occlusion band is where no frame has the right content: Smart Blend leaves the flow result there as it is (the
/// hole fill only touches pixels without coverage). In the strips the content that entered from outside the picture is
/// taken from the partner frame: Blend and Weighted Blend halve the error of the hold edge warps, Nearest has only one
/// frame and its edge fade is replaced by the smooth pull-push fill (a little worse than replicated edge pixels on
/// this smooth texture, 5.2 to 6.5 levels against 3.9 to 5.7).
#[test]
fn smart_blend_fills_the_strips_keeps_the_occlusion_and_leaves_no_unfilled_pixel() {
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let clip = OCCLUSION.clip(depth, W, H, 8);
        for mode in [Mode::Blend, Mode::Weighted, Mode::Nearest] {
            let run = |extra: &[(&str, &str)]| {
                let mut p = vec![("speed_percent", "50"), ("frame_interp", mode.popup())];
                p.extend_from_slice(extra);
                Rig::new(clip.clone(), &p).with_frames(10)
            };
            let (code0, smart, plain) = (run(&[]), run(&[("smart_blend", "1")]), run(&[("vectors", "1")]));
            for n in [1u64, 3, 5] {
                let s = n as f64 / 2.0;
                let r = Regions { k: s.floor() };
                let truth = OCCLUSION.frame_at(depth, W, H, s);
                let (g0, g1, gp) = (code0.render(n).unwrap(), smart.render(n).unwrap(), plain.render(n).unwrap());
                let case = format!("{depth:?} {mode:?} frame {n}");
                assert!(all_opaque(&g1), "{case}: Smart Blend left a pixel that is not opaque");
                let strip = |x, y| r.strip(x, y);
                let band = |x, y| r.band(x, y);
                let all = |_, _| true;
                let (s0, s1, sp) = (levels(&g0, &truth, strip), levels(&g1, &truth, strip), levels(&gp, &truth, strip));
                let (b0, b1) = (levels(&g0, &truth, band), levels(&g1, &truth, band));
                let (a0, a1, ap) = (levels(&g0, &truth, all), levels(&g1, &truth, all), levels(&gp, &truth, all));
                // the occlusion band: not worse than the plain flow blend (it is the same pixels)
                assert!(b1 <= b0 * 1.02 + 0.05, "{case}: band {b1:.2} against {b0:.2}");
                // the whole frame: better than the plain frame blend by a wide margin, not worse than the flow blend
                assert!(a1 < ap * 0.6, "{case}: whole frame {a1:.2} against the frame blend {ap:.2}");
                match mode {
                    Mode::Nearest => {
                        assert!(s1 <= s0 * 1.5 && s1 < sp * 0.3, "{case}: strips {s1:.2}, flow {s0:.2}, frame {sp:.2}");
                        assert!(a1 <= a0 * 1.06, "{case}: whole frame {a1:.2} against {a0:.2}");
                    }
                    _ => {
                        assert!(s1 <= s0 * 0.7 && s1 < sp * 0.2, "{case}: strips {s1:.2}, flow {s0:.2}, frame {sp:.2}");
                        assert!(a1 <= a0, "{case}: whole frame {a1:.2} against {a0:.2}");
                    }
                }
            }
        }
    }
}

#[test]
fn smart_blend_on_the_translating_texture_keeps_the_accuracy_and_fills_the_strips() {
    // the plain translating texture of the main tests (6 / 3 px per frame): Smart Blend keeps the interior accuracy of the
    // flow result (0.0018 to 0.0023) and beats the hold edge warps in the strips where the content enters, 1.1 to 1.3
    // levels against 2.9 to 3.4 (Blend: the partner frame has it)
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let motion = Motion { vx: 6.0, vy: 3.0 };
        let clip = motion.clip(depth, W, H, 8);
        let on = Rig::new(clip.clone(), &[("speed_percent", "50"), ("smart_blend", "1")]).with_frames(10);
        let off = Rig::new(clip, &[("speed_percent", "50")]).with_frames(10);
        for n in [1u64, 3, 5] {
            let truth = motion.frame_at(depth, W, H, n as f64 / 2.0);
            let (g_on, g_off) = (on.render(n).unwrap(), off.render(n).unwrap());
            assert!(all_opaque(&g_on));
            let (inner, _) = error(&g_on, &truth, MARGIN);
            assert!(inner < 0.003, "{depth:?} frame {n}: interior error {inner:.5}");
            let strip = |x: usize, y: usize| x < 8 || x >= W - 8 || y < 8 || y >= H - 8;
            let (e_on, e_off) = (levels(&g_on, &truth, strip), levels(&g_off, &truth, strip));
            assert!(e_on < e_off * 0.55, "{depth:?} frame {n}: strips {e_on:.2} against {e_off:.2} with hold edges");
        }
    }
}

#[test]
fn smart_blend_with_the_360_video_checkbox_runs_the_binary_alpha_composite() {
    // "360 Video" with Smart Blend selects the binary alpha composite in Blend (and the video range
    // luma in the solver): on opaque frames it is the same kind of result, within the accuracy of
    // the flow blend: 9.4 to 10.5 levels over the whole frame against 9.0 to 9.8 for the Smart
    // Blend composite and 9.5 to 10.7 without Smart Blend
    for depth in [Depth::U8, Depth::F32] {
        let clip = OCCLUSION.clip(depth, W, H, 8);
        let rig = Rig::new(clip, &[("speed_percent", "50"), ("smart_blend", "1"), ("video_360", "1")]).with_frames(10);
        for n in [1u64, 3, 5] {
            let got = rig.render(n).unwrap();
            assert!(all_opaque(&got), "{depth:?} frame {n}");
            let truth = OCCLUSION.frame_at(depth, W, H, n as f64 / 2.0);
            let a1 = levels(&got, &truth, |_, _| true);
            assert!(a1 < 12.0, "{depth:?} frame {n}: whole frame {a1:.2} levels");
        }
    }
}

/// Timing of one half speed render with the default parameters at the two usual sizes, in the global pool (every
/// hardware thread) and in a pool of one thread for the 8 bpc Blend (release build:
/// `cargo test -p retime --test vector --release -- --ignored --nocapture timing`). Not a pass or fail test.
#[test]
#[ignore]
fn timing() {
    use std::time::Instant;
    let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    println!("hardware threads: {}", rayon::current_num_threads());
    for (w, h) in [(1280usize, 720usize), (1920, 1080)] {
        for depth in [Depth::U8, Depth::U16, Depth::F32] {
            for (mode, name) in [(Mode::Blend, "Blend"), (Mode::Nearest, "Nearest"), (Mode::Weighted, "Weighted Blend")] {
                let clip = DEFAULT.clip(depth, w, h, 3);
                let rig = Rig::new(clip, &[("speed_percent", "50"), ("frame_interp", mode.popup())]).with_frames(4);
                let _ = rig.render(1).unwrap(); // warm up the thread pool and the allocator
                let runs = 3;
                let t = Instant::now();
                for _ in 0..runs {
                    let _ = rig.render(1).unwrap();
                }
                let ms = t.elapsed().as_secs_f64() * 1000.0 / runs as f64;
                let single = if depth == Depth::U8 {
                    let t = Instant::now();
                    one.install(|| {
                        let _ = rig.render(1).unwrap();
                    });
                    format!(", one thread {:.0} ms", t.elapsed().as_secs_f64() * 1000.0)
                } else {
                    String::new()
                };
                println!("{w}x{h} {depth:?} {name}: {ms:.0} ms per frame{single}");
            }
        }
    }
}

/// Where the time of one 1080p half speed Blend render goes (release build, `--ignored --nocapture breakdown`):
/// the wire to engine conversion of two frames, the two solves, the two warps, the blend, the engine to wire
/// conversion, and the sharpen step that Weighted Blend adds per warped frame.
#[test]
#[ignore]
fn breakdown() {
    use retime_engine::blend::blend;
    use retime_engine::filters::sharpen_step;
    use retime_engine::flow::{solve_pair, FlowSettings};
    use retime_engine::image::Image;
    use retime_engine::warp::{inv_warp, WarpOpts};
    use std::time::Instant;
    let (w, h) = (1920usize, 1080usize);
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1000.0;
    for depth in [Depth::U8, Depth::F32] {
        let (fa, fb) = (DEFAULT.frame_at(depth, w, h, 0.0), DEFAULT.frame_at(depth, w, h, 1.0));
        let t = Instant::now();
        let (a, b) = (Image::from_frame(&fa), Image::from_frame(&fb));
        let t_in = ms(t);
        let fs = FlowSettings::new(4, true, 0.7);
        let _ = solve_pair(&a, &b, &fs).unwrap();
        let t = Instant::now();
        let pair = solve_pair(&a, &b, &fs).unwrap();
        let t_solve = ms(t);
        let (mut ta, mut tb) = (Image::new(w, h, a.fmt()), Image::new(w, h, a.fmt()));
        let t = Instant::now();
        inv_warp(&mut ta, &a, &pair.f_ba, 0.5, WarpOpts::default()).unwrap();
        inv_warp(&mut tb, &b, &pair.f_ab, 0.5, WarpOpts::default()).unwrap();
        let t_warp = ms(t);
        let mut out = Image::new(w, h, a.fmt());
        let t = Instant::now();
        blend(&mut out, &ta, &tb, 0.5);
        let t_blend = ms(t);
        let t = Instant::now();
        let _ = out.to_frame();
        let t_out = ms(t);
        let t = Instant::now();
        sharpen_step(&mut ta, &a, 0.5);
        sharpen_step(&mut tb, &b, 0.5);
        let t_sharp = ms(t);
        println!("{depth:?} 1920x1080: from_frame x2 {t_in:.0} ms, solve_pair {t_solve:.0}, inv_warp x2 {t_warp:.0}, blend {t_blend:.0}, to_frame {t_out:.0}, sharpen_step x2 {t_sharp:.0}");
    }
}
