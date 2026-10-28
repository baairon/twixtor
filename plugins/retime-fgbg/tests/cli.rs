//! The first slice end to end through the real binary: an ffmpeg `testsrc2` clip as raw frames, `vfx apply retime-fgbg`
//! with a frame blend retime, and the output compared with values computed independently here.
//!
//! The test builds `vfx` with the cargo that runs it (`cargo build -p fx-cli`, in the same target directory and profile)
//! and needs `ffmpeg` on the PATH; without ffmpeg it prints a note and passes, so a machine without it stays green.
//! The pixel formats are the ones `vfx` reads and writes: `rgba`, `rgba64le`, `gbrapf32le` (planar G B R A).

mod common;

use fx_core::{Depth, Frame, Pixels};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

const W: usize = 64;
const H: usize = 48;
const SOURCE_FRAMES: usize = 12;

fn workspace() -> PathBuf {
    // plugins/retime-fgbg -> the workspace root
    Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).expect("workspace root").to_path_buf()
}

/// Builds `vfx` once and returns its path, or None when it cannot be built.
fn vfx() -> Option<&'static Path> {
    static BIN: OnceLock<Option<PathBuf>> = OnceLock::new();
    BIN.get_or_init(|| {
        let exe = std::env::current_exe().ok()?;
        let profile_dir = exe.parent()?.parent()?.to_path_buf(); // target/<profile>/deps/cli-hash -> target/<profile>
        let release = profile_dir.file_name().is_some_and(|n| n == "release");
        let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
        let mut build = Command::new(cargo);
        build.args(["build", "-p", "fx-cli", "--quiet"]).current_dir(workspace());
        if release {
            build.arg("--release");
        }
        if !build.status().ok()?.success() {
            return None;
        }
        let bin = profile_dir.join(if cfg!(windows) { "vfx.exe" } else { "vfx" });
        bin.exists().then_some(bin)
    })
    .as_deref()
}

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok_and(|o| o.status.success())
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("retime-fgbg-cli-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn pix_fmt(depth: Depth) -> &'static str {
    match depth {
        Depth::U8 => "rgba",
        Depth::U16 => "rgba64le",
        Depth::F32 => "gbrapf32le",
    }
}

/// A `testsrc2` clip of `SOURCE_FRAMES` frames as raw frames in the wire format of `depth`.
fn make_clip(dir: &Scratch, depth: Depth) -> PathBuf {
    let out = dir.path("clip.raw");
    let status = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
        .arg(format!("testsrc2=size={W}x{H}:rate=24:duration=2"))
        .args(["-frames:v", &SOURCE_FRAMES.to_string(), "-pix_fmt", pix_fmt(depth), "-f", "rawvideo"])
        .arg(&out)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg could not make the clip");
    out
}

fn apply(bin: &Path, depth: Depth, input: &Path, output: &Path, frames: usize, params: &[&str]) -> Output {
    let mut cmd = Command::new(bin);
    cmd.args(["apply", "retime-fgbg", "--width", &W.to_string(), "--height", &H.to_string(), "--depth", &depth.bits().to_string()])
        .args(["--fps", "24", "--in"])
        .arg(input)
        .arg("--out")
        .arg(output)
        .args(["--frames", &frames.to_string()]);
    for p in params {
        cmd.args(["--param", p]);
    }
    cmd.output().expect("run vfx")
}

fn read_frames(path: &Path, depth: Depth) -> Vec<Frame> {
    let bytes = std::fs::read(path).unwrap();
    let len = W * H * depth.bytes_per_pixel();
    assert_eq!(bytes.len() % len, 0, "a trailing partial frame");
    bytes.chunks(len).map(|c| Frame::from_bytes(W, H, depth, c)).collect()
}

/// The blend of two OPAQUE 8 bpc pixels at weight one half, per channel (premultiply is the identity at alpha 255).
fn mix8(a: [u8; 4], b: [u8; 4]) -> [u8; 4] {
    std::array::from_fn(|i| ((((b[i] as f32 - a[i] as f32) * 0.5f32) + a[i] as f32) + 0.5f32) as i64 as u8)
}

/// Half of an opaque 8 bpc pixel against the transparent frame, then the output finish: premultiplied
/// `trunc((0 - c) * 0.5 + c + 0.5)`, alpha 128, and unpremultiply with `255 / 128`.
fn fade8(a: [u8; 4]) -> [u8; 4] {
    let alpha = ((((0.0f32 - a[3] as f32) * 0.5f32) + a[3] as f32) + 0.5f32) as i64 as u8;
    let s = 255.0f32 / alpha as f32;
    let mut out = [0u8; 4];
    for i in 0..3 {
        let c = ((((0.0f32 - a[i] as f32) * 0.5f32) + a[i] as f32) + 0.5f32) as i64 as u8;
        let mut v = c as f32 * s;
        if 255.0f32 < v {
            v = 255.0;
        }
        out[i] = ((v as f64) + 0.5) as i64 as u8;
    }
    out[3] = alpha;
    out
}

/// What a frame becomes when it passes the engine on an exact position. 8 and 32 bpc opaque pixels
/// come back bit for bit. At 16 bpc the engine works in the range 0..=32768, which has half as many
/// levels as the wire range 0..=65535 (`Frame::u16_to_q15`, `Frame::q15_to_u16`): an opaque pixel
/// comes back as the nearest level of that coarser range.
fn exact_pass(f: &Frame) -> Frame {
    let mut out = f.clone();
    if let Pixels::U16(px) = &mut out.pixels {
        for p in px.iter_mut() {
            *p = p.map(|v| Frame::q15_to_u16(Frame::u16_to_q15(v)));
        }
    }
    out
}

fn px8(f: &Frame) -> &[[u8; 4]] {
    match &f.pixels {
        Pixels::U8(p) => p,
        _ => panic!("not an 8 bpc frame"),
    }
}

#[test]
fn half_speed_frame_blend_of_a_testsrc2_clip_at_8_bpc() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg is not on the PATH: skipping the end to end test");
        return;
    }
    let bin = vfx().expect("fx-cli builds");
    let dir = Scratch::new("half");
    let clip = make_clip(&dir, Depth::U8);
    let out = dir.path("out.raw");
    let r = apply(bin, Depth::U8, &clip, &out, 2 * SOURCE_FRAMES, &["vectors=1", "speed_percent=50", "frame_interp=2"]);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let src = read_frames(&clip, Depth::U8);
    let got = read_frames(&out, Depth::U8);
    assert_eq!(src.len(), SOURCE_FRAMES);
    assert_eq!(got.len(), 2 * SOURCE_FRAMES, "the frame count is --frames");
    let distinct = src.iter().enumerate().filter(|(i, f)| src[..*i].iter().all(|g| g.pixels != f.pixels)).count();
    assert!(distinct >= 4, "the test clip moves: {distinct} distinct frames");
    let mut real_blends = 0;
    for (n, g) in got.iter().enumerate() {
        let k = n / 2;
        let want: Vec<[u8; 4]> = if n % 2 == 0 {
            // output frame 2k is source frame k, bit for bit (testsrc2 is opaque)
            px8(&src[k]).to_vec()
        } else if k + 1 < SOURCE_FRAMES {
            px8(&src[k]).iter().zip(px8(&src[k + 1])).map(|(&a, &b)| mix8(a, b)).collect()
        } else {
            // past the last frame the neighbour is transparent
            px8(&src[k]).iter().map(|&a| fade8(a)).collect()
        };
        assert_eq!(px8(g), &want[..], "output frame {n}");
        if n % 2 == 1 && k + 1 < SOURCE_FRAMES && g.pixels != src[k].pixels && g.pixels != src[k + 1].pixels {
            real_blends += 1;
        }
    }
    assert!(real_blends >= 2, "{real_blends} odd frames differ from both neighbours");
}

#[test]
fn the_other_blend_modes_and_depths_run_end_to_end() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg is not on the PATH: skipping the end to end test");
        return;
    }
    let bin = vfx().expect("fx-cli builds");
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let dir = Scratch::new(&format!("modes{}", depth.bits()));
        let clip = make_clip(&dir, depth);
        let src = read_frames(&clip, depth);
        // exact frames: the default parameters (Vectors Best, Weighted Blend) copy the clip through the round trip
        let out = dir.path("exact.raw");
        let r = apply(bin, depth, &clip, &out, SOURCE_FRAMES, &[]);
        assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
        let expected: Vec<Frame> = src.iter().map(exact_pass).collect();
        assert_eq!(read_frames(&out, depth), expected, "{depth:?} opaque frames survive the engine on exact positions");
        // Nearest at half speed repeats every frame twice
        let out = dir.path("nearest.raw");
        let r = apply(bin, depth, &clip, &out, 2 * SOURCE_FRAMES, &["vectors=1", "speed_percent=50", "frame_interp=1"]);
        assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
        let got = read_frames(&out, depth);
        assert_eq!(got.len(), 2 * SOURCE_FRAMES);
        for (n, g) in got.iter().enumerate() {
            assert_eq!(g, &expected[n / 2], "{depth:?} nearest frame {n}");
        }
        // Weighted Blend without vectors is the plain Blend
        let (wb, bl) = (dir.path("wb.raw"), dir.path("blend.raw"));
        for (file, interp) in [(&wb, "frame_interp=3"), (&bl, "frame_interp=2")] {
            let r = apply(bin, depth, &clip, file, 2 * SOURCE_FRAMES, &["vectors=1", "speed_percent=50", interp]);
            assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
        }
        assert_eq!(std::fs::read(&wb).unwrap(), std::fs::read(&bl).unwrap(), "{depth:?}");
        // the blended odd frames lie between their neighbours (opaque data: per channel min and max)
        let blended = read_frames(&bl, depth);
        for n in (1..2 * SOURCE_FRAMES - 1).step_by(2) {
            let (a, b) = (&src[n / 2], &src[n / 2 + 1]);
            match (&blended[n].pixels, &a.pixels, &b.pixels) {
                (Pixels::U8(g), Pixels::U8(a), Pixels::U8(b)) => {
                    for ((g, a), b) in g.iter().zip(a).zip(b) {
                        for i in 0..3 {
                            assert!(g[i] >= a[i].min(b[i]) && g[i] <= a[i].max(b[i]), "8 bpc frame {n}");
                        }
                    }
                }
                (Pixels::U16(g), Pixels::U16(a), Pixels::U16(b)) => {
                    for ((g, a), b) in g.iter().zip(a).zip(b) {
                        for i in 0..3 {
                            // the 15 bit range of the engine and the rounding of the chain move a value by a level or two
                            assert!(g[i] as i32 >= a[i].min(b[i]) as i32 - 3 && g[i] as i32 <= a[i].max(b[i]) as i32 + 3, "16 bpc frame {n}");
                        }
                    }
                }
                (Pixels::F32(g), Pixels::F32(a), Pixels::F32(b)) => {
                    for ((g, a), b) in g.iter().zip(a).zip(b) {
                        for i in 0..3 {
                            let (lo, hi) = (a[i].min(b[i]), a[i].max(b[i]));
                            let want = ((b[i] - a[i]) * 0.5f32 + a[i]) + 0.0f32;
                            assert_eq!(g[i], want, "32 bpc frame {n}");
                            assert!(g[i] >= lo - 1e-6 && g[i] <= hi + 1e-6);
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
    }
}

/// The synthetic clip with a known motion (`common/moving.rs`) as raw frames in the wire format of `depth`.
fn make_moving_clip(dir: &Scratch, depth: Depth) -> PathBuf {
    let out = dir.path("moving.raw");
    let bytes: Vec<u8> = common::moving::DEFAULT.clip(depth, W, H, SOURCE_FRAMES).iter().flat_map(|f| f.to_bytes()).collect();
    std::fs::write(&out, bytes).unwrap();
    out
}

/// The default parameters (Vectors Best, Frame Interp Blend, Smart Blend off) with only the speed changed:
/// the optical flow path through the real binary. The run succeeds, the frame count is `--frames`, the even
/// output frames (exact positions) are the source frames, and the odd ones are much closer to the analytic
/// in-between frame of the moving pattern than the plain frame blend of the same run with No Vectors.
#[test]
fn half_speed_with_the_default_vectors_runs_the_optical_flow_path_end_to_end() {
    let bin = vfx().expect("fx-cli builds");
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let dir = Scratch::new(&format!("flow{}", depth.bits()));
        let clip = make_moving_clip(&dir, depth);
        let src = read_frames(&clip, depth);
        assert_eq!(src.len(), SOURCE_FRAMES);
        let frames = 2 * SOURCE_FRAMES - 2; // positions 0 to 10.5: every odd frame has both neighbours
        let out = dir.path("flow.raw");
        let r = apply(bin, depth, &clip, &out, frames, &["speed_percent=50"]);
        assert!(r.status.success(), "{depth:?}: {}", String::from_utf8_lossy(&r.stderr));
        let got = read_frames(&out, depth);
        assert_eq!(got.len(), frames, "the frame count is --frames");
        let plain_out = dir.path("plain.raw");
        let r = apply(bin, depth, &clip, &plain_out, frames, &["speed_percent=50", "vectors=1"]);
        assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
        let plain = read_frames(&plain_out, depth);
        let (mut worst, mut closest) = (0.0f64, f64::MAX);
        for (n, g) in got.iter().enumerate() {
            if n % 2 == 0 {
                assert_eq!(g, &exact_pass(&src[n / 2]), "{depth:?} output frame {n} is source frame {}", n / 2);
            } else {
                let truth = common::moving::DEFAULT.frame_at(depth, W, H, n as f64 / 2.0);
                let (flow_err, _) = common::moving::error(g, &truth, 9);
                let (plain_err, _) = common::moving::error(&plain[n], &truth, 9);
                worst = worst.max(flow_err);
                closest = closest.min(plain_err / flow_err);
                // measured at 64 x 48: flow error 0.0013 to 0.0017, plain blend 0.019 to 0.023, 11.8 to 17 times closer
                assert!(flow_err < 0.0025, "{depth:?} frame {n}: flow error {flow_err}");
                assert!(flow_err * 8.0 < plain_err, "{depth:?} frame {n}: flow error {flow_err} against plain blend error {plain_err}");
            }
        }
        eprintln!("{depth:?}: worst flow error {worst:.5}, at least {closest:.1}x closer than the plain blend");
    }
}

#[test]
fn the_effect_is_listed_as_temporal_with_its_five_layers() {
    let bin = vfx().expect("fx-cli builds");
    let out = Command::new(bin).args(["list", "--json"]).output().unwrap();
    assert!(out.status.success());
    let list: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let entry = list
        .as_array()
        .expect("a list of effects")
        .iter()
        .find(|e| e["id"] == "retime-fgbg")
        .expect("retime-fgbg is registered");
    assert_eq!(entry["kind"], "temporal");
    let layers: Vec<(&str, bool)> =
        entry["layers"].as_array().unwrap().iter().map(|l| (l["id"].as_str().unwrap(), l["optional"].as_bool().unwrap())).collect();
    assert_eq!(
        layers,
        vec![("alt_source_a", true), ("alt_source_b", true), ("fg1_matte", true), ("fg2_matte", true), ("fg3_matte", true)]
    );
}

/// Warping Forward and Forward, no Smart Blend render between frames through `vfx` and succeed. A
/// run whose frames are all on exact positions never needs a warp, so the clip passes through
/// unchanged.
#[test]
fn the_forward_warp_renders_between_frames_through_the_binary_and_exact_frames_pass_through() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg is not on the PATH: skipping the end to end test");
        return;
    }
    let bin = vfx().expect("fx-cli builds");
    let dir = Scratch::new("warp");
    let clip = make_clip(&dir, Depth::U8);
    let out = dir.path("out.raw");
    for warping in ["3", "4"] {
        let r = apply(bin, Depth::U8, &clip, &out, 4, &["speed_percent=50", &format!("warping={warping}")]);
        assert!(r.status.success(), "warping {warping}: {}", String::from_utf8_lossy(&r.stderr));
        assert_eq!(read_frames(&out, Depth::U8).len(), 4);
    }
    // every frame on a source frame: the warp is never reached and the clip passes through
    let r = apply(bin, Depth::U8, &clip, &out, SOURCE_FRAMES, &["warping=3"]);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    assert_eq!(read_frames(&out, Depth::U8), read_frames(&clip, Depth::U8));
}

/// Writes `frames` as one raw clip file.
fn write_raw(path: PathBuf, frames: &[Frame]) -> PathBuf {
    let bytes: Vec<u8> = frames.iter().flat_map(|f| f.to_bytes()).collect();
    std::fs::write(&path, bytes).unwrap();
    path
}

/// The disc matte of the layered tests: opaque white inside a disc of radius 12 that moves 2 px right per frame.
fn disc_matte(depth: Depth, count: usize) -> Vec<Frame> {
    (0..count)
        .map(|k| {
            let mut f = Frame::new(W, H, depth);
            if let Pixels::U8(px) = &mut f.pixels {
                for y in 0..H {
                    for x in 0..W {
                        let (dx, dy) = (x as f64 - 24.0 - 2.0 * k as f64, y as f64 - 24.0);
                        let v = if dx * dx + dy * dy <= 144.0 { 255 } else { 0 };
                        px[y * W + x] = [v, v, v, 255];
                    }
                }
            }
            f
        })
        .collect()
}

#[test]
fn a_fg_matte_layer_file_renders_end_to_end_as_the_engine_does_and_notes_its_rules_once() {
    let bin = vfx().expect("fx-cli builds");
    let dir = Scratch::new("layered");
    let clip_frames = common::moving::DEFAULT.clip(Depth::U8, W, H, SOURCE_FRAMES);
    let matte_frames = disc_matte(Depth::U8, SOURCE_FRAMES);
    let clip = write_raw(dir.path("clip.raw"), &clip_frames);
    let matte = write_raw(dir.path("matte.raw"), &matte_frames);
    let out = dir.path("out.raw");
    for (params, notes) in [
        (vec!["speed_percent=50"], vec!["G1", "G1b"]),
        // frame 0 is exact: the fast path of the FG1 group notes G1h first; the notes of a frame come in id order
        (vec!["speed_percent=50", "track_rgb_a=1", "warping=2", "display=3"], vec!["G1", "G1h", "G1b", "G1g", "G6"]),
        (vec!["speed_percent=50", "pt1_use=2", "pt1_pos=24,24@0;30,24@6", "blur_compensation=1"], vec!["G1", "G1b", "G1f", "G2"]),
        (vec!["vectors=1", "display=6"], vec!["G1", "G1h"]),
    ] {
        let frames = 6;
        let mut cmd = Command::new(bin);
        cmd.args(["apply", "retime-fgbg", "--width", &W.to_string(), "--height", &H.to_string(), "--depth", "8", "--fps", "24", "--in"])
            .arg(&clip)
            .arg("--out")
            .arg(&out)
            .args(["--frames", &frames.to_string(), "--layer"])
            .arg(format!("fg1_matte={}", matte.display()));
        for p in &params {
            cmd.args(["--param", p]);
        }
        let r = cmd.output().expect("run vfx");
        let stderr = String::from_utf8_lossy(&r.stderr).to_string();
        assert!(r.status.success(), "{params:?}: {stderr}");
        // the bytes of the engine called in process with the same clips and parameters
        let pairs: Vec<(&str, &str)> = params.iter().map(|p| p.split_once('=').unwrap()).collect();
        let fps24 = fx_core::TimeBase::default_for_rate(fx_core::Rational::new(24, 1).unwrap());
        let mut curves = fx_core::ParamCurves::new(fx_core::Effect::params(&retime_fgbg::RetimeFgBg));
        for (id, text) in &pairs {
            curves.parse_cli(id, text, fps24).unwrap();
        }
        let input = fx_core::MemLayer::new(fps24, clip_frames.clone()).unwrap();
        let mut layers = fx_core::LayerSet::new();
        layers.insert("fg1_matte", std::sync::Arc::new(fx_core::MemLayer::new(fps24, matte_frames.clone()).unwrap()));
        let env = fx_core::Env { params: &curves, input: &input, layers: &layers };
        let got = read_frames(&out, Depth::U8);
        assert_eq!(got.len(), frames);
        for (n, g) in got.iter().enumerate() {
            let cx = fx_core::TemporalContext::new(W, H, Depth::U8, fps24, n as u64, frames as u64);
            let mut want = Frame::new(W, H, Depth::U8);
            fx_core::Effect::render_temporal(&retime_fgbg::RetimeFgBg, &cx, &env, &mut want).unwrap();
            assert_eq!(g, &want, "{params:?} frame {n}");
        }
        // each open rule the run reached is noted exactly once, and no other
        let noted: Vec<&str> = stderr.lines().filter_map(|l| l.strip_prefix("vfx: note: retime-fgbg: ")).map(|l| l.split(' ').next().unwrap()).collect();
        assert_eq!(noted, notes, "{params:?}: {stderr}");
    }
}

/// The `vfx: note:` line of gap `g` for retime-fgbg.
fn note_line(g: &fx_core::OpenGap) -> String {
    format!("vfx: note: retime-fgbg: {} {}: a stated rule that no Twixtor reference render has graded yet (README: limits)", g.id, g.title)
}

#[test]
fn the_g1_note_prints_for_a_run_of_any_one_layered_frame_and_not_for_the_source_display() {
    // gaps.rs: G1 is noted on every frame rendered with an FG matte layer present, on any display but Source. vfx
    // prints a gap once per run, so each run here renders ONE frame (Frame Number mode): an in-between frame (the
    // layered vector path, which also notes G1b), an exact frame and a No Vectors frame (the layered fast path), and
    // the Source display, which composes no layer and notes nothing
    let bin = vfx().expect("fx-cli builds");
    let dir = Scratch::new("g1-note");
    let clip = write_raw(dir.path("clip.raw"), &common::moving::DEFAULT.clip(Depth::U8, W, H, SOURCE_FRAMES));
    let matte = write_raw(dir.path("matte.raw"), &disc_matte(Depth::U8, SOURCE_FRAMES));
    let out = dir.path("out.raw");
    let (g1, g1b) = (note_line(&retime_engine::gaps::G1), note_line(&retime_engine::gaps::G1B));
    for (params, want_g1, want_g1b) in [
        (vec!["time_remap_mode=2", "frame_num=2.5"], true, true),
        (vec!["time_remap_mode=2", "frame_num=3"], true, false),
        (vec!["time_remap_mode=2", "frame_num=2.5", "vectors=1"], true, false),
        (vec!["time_remap_mode=2", "frame_num=2.5", "display=2"], false, false),
    ] {
        let mut cmd = Command::new(bin);
        cmd.args(["apply", "retime-fgbg", "--width", &W.to_string(), "--height", &H.to_string(), "--depth", "8", "--fps", "24", "--in"])
            .arg(&clip)
            .arg("--out")
            .arg(&out)
            .args(["--frames", "1", "--layer"])
            .arg(format!("fg1_matte={}", matte.display()));
        for p in &params {
            cmd.args(["--param", p]);
        }
        let r = cmd.output().expect("run vfx");
        let stderr = String::from_utf8_lossy(&r.stderr).to_string();
        assert!(r.status.success(), "{params:?}: {stderr}");
        let count = |line: &str| stderr.lines().filter(|l| l.trim_end() == line).count();
        assert_eq!(count(&g1), usize::from(want_g1), "{params:?}: {stderr}");
        assert_eq!(count(&g1b), usize::from(want_g1b), "{params:?}: {stderr}");
        if !want_g1 {
            assert!(!stderr.contains("vfx: note:"), "{params:?}: the Source display notes nothing: {stderr}");
        }
    }
}
