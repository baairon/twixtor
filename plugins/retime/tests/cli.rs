//! End to end through the real `vfx` program: an ffmpeg `testsrc2` clip as raw frames,
//! `vfx apply retime` with a frame blend retime, and the output compared with values computed
//! independently here.
//!
//! The test builds `vfx` with the cargo that runs it (`cargo build -p fx-cli`, in the same target
//! directory and profile) and needs `ffmpeg` on the PATH; without ffmpeg it prints a note and
//! passes, so a machine without it stays green. The pixel formats are the ones `vfx` reads and
//! writes: `rgba`, `rgba64le`, `gbrapf32le` (planar G B R A).

mod common;

use fx_core::{Depth, Frame, Pixels};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

const W: usize = 64;
const H: usize = 48;
const SOURCE_FRAMES: usize = 12;

fn workspace() -> PathBuf {
    // plugins/retime -> the workspace root
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
        let dir = std::env::temp_dir().join(format!("retime-cli-{}-{name}", std::process::id()));
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
    apply_sized(bin, depth, (W, H), input, output, frames, params)
}

fn apply_sized(bin: &Path, depth: Depth, (w, h): (usize, usize), input: &Path, output: &Path, frames: usize, params: &[&str]) -> Output {
    let mut cmd = Command::new(bin);
    cmd.args(["apply", "retime", "--width", &w.to_string(), "--height", &h.to_string(), "--depth", &depth.bits().to_string()])
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
    read_frames_sized(path, depth, (W, H))
}

fn read_frames_sized(path: &Path, depth: Depth, (w, h): (usize, usize)) -> Vec<Frame> {
    let bytes = std::fs::read(path).unwrap();
    let len = w * h * depth.bytes_per_pixel();
    assert_eq!(bytes.len() % len, 0, "a trailing partial frame");
    bytes.chunks(len).map(|c| Frame::from_bytes(w, h, depth, c)).collect()
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
/// levels as the wire (`Frame::u16_to_q15`, `Frame::q15_to_u16`): an opaque pixel comes back as the
/// nearest level of that coarser range.
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

/// What the user sees when the flow path cannot serve a request: exit code 3, the frame that failed and the reason. A 3 x 2
/// clip is below the 2 x 2 working size the solver needs at Best; the exact frames of the same run are fine.
#[test]
fn the_flow_path_refuses_what_it_cannot_serve_with_exit_code_3_and_the_reason() {
    let bin = vfx().expect("fx-cli builds");
    let dir = Scratch::new("refuse");
    let tiny = dir.path("tiny.raw");
    let frames: Vec<Frame> = (0..4u8).map(|k| common::f8(3, 2, &[[10 * k, 20, 30, 255]; 6])).collect();
    std::fs::write(&tiny, frames.iter().flat_map(|f| f.to_bytes()).collect::<Vec<u8>>()).unwrap();
    let run = |speed: &str| {
        Command::new(bin)
            .args(["apply", "retime", "--width", "3", "--height", "2", "--depth", "8", "--fps", "24", "--in"])
            .arg(&tiny)
            .arg("--out")
            .arg(dir.path("out.raw"))
            .args(["--frames", "4", "--param", &format!("speed_percent={speed}")])
            .output()
            .expect("run vfx")
    };
    let r = run("50");
    assert_eq!(r.status.code(), Some(3), "{}", String::from_utf8_lossy(&r.stderr));
    let stderr = String::from_utf8_lossy(&r.stderr);
    assert!(stderr.contains("frame 1"), "{stderr}");
    assert!(
        stderr.contains("unsupported: optical flow: working size 1x1 (output 3x2, reduction 1) is below 2x2, which is not supported"),
        "{stderr}"
    );
    // exact positions never reach the solver
    let r = run("100");
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
}

/// Smart Blend (`smart_blend=1`) through the real binary at the three depths, on the moving texture and on the scene with
/// an occlusion: success, the exact frames are the source frames, every pixel is opaque, and the in-between frames are close
/// to the analytic frame (the texture: within 0.0025, 8 times closer than the frame blend; the occlusion scene: the whole
/// frame error at most 0.6 of the frame blend's, measured about 0.45).
#[test]
fn smart_blend_runs_end_to_end() {
    let bin = vfx().expect("fx-cli builds");
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let dir = Scratch::new(&format!("smart{}", depth.bits()));
        let frames = 2 * SOURCE_FRAMES - 2;
        let (w, h) = (96usize, 64usize);
        for (name, size, clip_frames) in [
            ("texture", (W, H), common::moving::DEFAULT.clip(depth, W, H, SOURCE_FRAMES)),
            ("occlusion", (w, h), common::moving::OCCLUSION.clip(depth, w, h, SOURCE_FRAMES)),
        ] {
            let clip = dir.path(&format!("{name}.raw"));
            std::fs::write(&clip, clip_frames.iter().flat_map(|f| f.to_bytes()).collect::<Vec<u8>>()).unwrap();
            let (out, plain_out) = (dir.path("smart.raw"), dir.path("plain.raw"));
            let r = apply_sized(bin, depth, size, &clip, &out, frames, &["speed_percent=50", "smart_blend=1"]);
            assert!(r.status.success(), "{depth:?} {name}: {}", String::from_utf8_lossy(&r.stderr));
            let r = apply_sized(bin, depth, size, &clip, &plain_out, frames, &["speed_percent=50", "vectors=1"]);
            assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
            let (got, plain) = (read_frames_sized(&out, depth, size), read_frames_sized(&plain_out, depth, size));
            assert_eq!(got.len(), frames);
            for (n, g) in got.iter().enumerate() {
                if n % 2 == 0 {
                    assert_eq!(g, &exact_pass(&clip_frames[n / 2]), "{depth:?} {name} frame {n} is source frame {}", n / 2);
                    continue;
                }
                assert!(common::moving::all_opaque(g), "{depth:?} {name} frame {n} has a pixel that is not opaque");
                let s = n as f64 / 2.0;
                let truth = if name == "texture" {
                    common::moving::DEFAULT.frame_at(depth, W, H, s)
                } else {
                    common::moving::OCCLUSION.frame_at(depth, w, h, s)
                };
                if name == "texture" {
                    let (flow_err, _) = common::moving::error(g, &truth, 9);
                    let (plain_err, _) = common::moving::error(&plain[n], &truth, 9);
                    assert!(flow_err < 0.0025 && flow_err * 8.0 < plain_err, "{depth:?} frame {n}: smart {flow_err}, plain blend {plain_err}");
                } else {
                    let all = |_: usize, _: usize| true;
                    let (smart_err, plain_err) = (common::moving::error_in(g, &truth, all), common::moving::error_in(&plain[n], &truth, all));
                    assert!(smart_err < plain_err * 0.6, "{depth:?} frame {n}: smart {smart_err}, plain blend {plain_err}");
                }
            }
        }
    }
}

/// A real video test pattern (ffmpeg `testsrc2`: sharp edges, text, a moving gradient and counters) with the default
/// parameters except the speed: the flow path runs to the end on content that is not a smooth texture, every depth
/// of the pipeline keeps the exact frames, and the in-between frames are no frame blend.
#[test]
fn the_default_parameters_run_on_a_testsrc2_clip() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg is not on the PATH: skipping the end to end test");
        return;
    }
    let bin = vfx().expect("fx-cli builds");
    let dir = Scratch::new("testsrc");
    let clip = make_clip(&dir, Depth::U8);
    let src = read_frames(&clip, Depth::U8);
    let frames = 2 * SOURCE_FRAMES - 2;
    let out = dir.path("flow.raw");
    let r = apply(bin, Depth::U8, &clip, &out, frames, &["speed_percent=50"]);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let got = read_frames(&out, Depth::U8);
    assert_eq!(got.len(), frames);
    let plain_out = dir.path("plain.raw");
    let r = apply(bin, Depth::U8, &clip, &plain_out, frames, &["speed_percent=50", "vectors=1"]);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let plain = read_frames(&plain_out, Depth::U8);
    let (mut different, mut moving) = (0, 0);
    for (n, g) in got.iter().enumerate() {
        if n % 2 == 0 {
            assert_eq!(px8(g), px8(&src[n / 2]), "output frame {n} is source frame {}", n / 2);
        } else {
            assert!(px8(g).iter().all(|p| p[3] == 255), "frame {n} is opaque");
            // testsrc2 repeats frames at this size: between two equal frames there is nothing to interpolate
            if src[n / 2].pixels != src[n / 2 + 1].pixels {
                moving += 1;
                if g.pixels != plain[n].pixels {
                    different += 1;
                }
            } else {
                assert_eq!(g.pixels, plain[n].pixels, "frame {n} lies between equal frames and is that frame");
            }
        }
    }
    assert!(moving >= 3 && different == moving, "{different} of {moving} in-between frames between different frames differ from the frame blend");
}

#[test]
fn the_effect_is_listed_as_temporal_with_its_two_layers() {
    let bin = vfx().expect("fx-cli builds");
    let out = Command::new(bin).args(["list", "--json"]).output().unwrap();
    assert!(out.status.success());
    let list: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let entry = list
        .as_array()
        .expect("a list of effects")
        .iter()
        .find(|e| e["id"] == "retime")
        .expect("retime is registered");
    assert_eq!(entry["kind"], "temporal");
    let layers: Vec<(&str, bool)> =
        entry["layers"].as_array().unwrap().iter().map(|l| (l["id"].as_str().unwrap(), l["optional"].as_bool().unwrap())).collect();
    assert_eq!(layers, vec![("alt_source_a", true), ("alt_source_b", true)]);
}
