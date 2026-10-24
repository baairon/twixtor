//! Plugin 04 end to end through the real binary: raw clips for the input, the vector layers (`--layer`) and
//! the output, `vfx apply vectors-in`, and the output compared with values computed independently here. The tests that
//! need `ffmpeg` (a `testsrc2` clip) skip with a note without it; the ones on the analytic moving clip do not.
//!
//! The test builds `vfx` with the cargo that runs it (`cargo build -p fx-cli`, in the same target directory and profile)
//! and needs `ffmpeg` on the PATH; without ffmpeg it prints a note and passes, so a machine without it stays green.
//! The pixel formats are the ones `vfx` reads and writes: `rgba`, `rgba64le`, `gbrapf32le` (planar G B R A).

mod common;

use common::*;
use fx_core::{Depth, Frame, Pixels};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

const W: usize = 64;
const H: usize = 48;
const SOURCE_FRAMES: usize = 12;

fn workspace() -> PathBuf {
    // plugins/vectors-in -> the workspace root
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
        let dir = std::env::temp_dir().join(format!("vectors-in-cli-{}-{name}", std::process::id()));
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

/// `vfx apply vectors-in` on `input` with `--param` values and `--layer ID=FILE` layers. The
/// composition rate is 30: with `In FPS is Out FPS` checked (the default) plugin 04 takes the input
/// rate as 30 and the output rate from the composition, so a 24 fps composition would speed a 100
/// percent retime up by 30 / 24.
fn apply(bin: &Path, depth: Depth, input: &Path, output: &Path, frames: usize, params: &[&str], layers: &[(&str, &Path)]) -> Output {
    let mut cmd = Command::new(bin);
    cmd.args(["apply", "vectors-in", "--width", &W.to_string(), "--height", &H.to_string(), "--depth", &depth.bits().to_string()])
        .args(["--fps", "30", "--in"])
        .arg(input)
        .arg("--out")
        .arg(output)
        .args(["--frames", &frames.to_string()]);
    for p in params {
        cmd.args(["--param", p]);
    }
    for (id, file) in layers {
        cmd.arg("--layer").arg(format!("{id}={}", file.display()));
    }
    cmd.output().expect("run vfx")
}

fn write_clip(path: PathBuf, frames: &[Frame]) -> PathBuf {
    let bytes: Vec<u8> = frames.iter().flat_map(|f| f.to_bytes()).collect();
    std::fs::write(&path, bytes).unwrap();
    path
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


const MD: f32 = 16.0;

/// The moving clip, its two vector layers (the exact flow, `common::motion_vectors`) as raw files in `dir`.
fn moving_files(dir: &Scratch, depth: Depth) -> (PathBuf, PathBuf, PathBuf) {
    let clip = write_clip(dir.path("moving.raw"), &common::moving::DEFAULT.clip(depth, W, H, SOURCE_FRAMES));
    // the vector layers must cover the OUTPUT times: presence is decided at current_time, so layers
    // as short as the source would vanish when a slow motion run goes past the end of the clip
    let (prev, next) = motion_vectors(depth, W, H, 2 * SOURCE_FRAMES, 3.4, 1.7, MD);
    (clip, write_clip(dir.path("prev.raw"), &prev), write_clip(dir.path("next.raw"), &next))
}

/// Half speed with the two vector layers through the real binary, Warping Inverse: the run succeeds, the frame
/// count is `--frames`, the even frames (exact positions) are the source frames and the odd ones are much closer to
/// the analytic in-between frame than the plain frame blend (the same run with the segment marker "No MV, Frame
/// Blend"). Measured at 64 x 48: Vectors In error 0.0014 to 0.0018, plain blend 0.019 to 0.022.
#[test]
fn half_speed_with_vector_layers_runs_the_vectors_in_path_end_to_end() {
    let bin = vfx().expect("fx-cli builds");
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let dir = Scratch::new(&format!("vin{}", depth.bits()));
        let (clip, prev, next) = moving_files(&dir, depth);
        let src = read_frames(&clip, depth);
        assert_eq!(src.len(), SOURCE_FRAMES);
        let frames = 2 * SOURCE_FRAMES - 2;
        let params = ["speed_percent=50", "warping=1", "vectors_max_displace=16"];
        let layers = [("vectors_prev", prev.as_path()), ("vectors_next", next.as_path())];
        let out = dir.path("vin.raw");
        let r = apply(bin, depth, &clip, &out, frames, &params, &layers);
        assert!(r.status.success(), "{depth:?}: {}", String::from_utf8_lossy(&r.stderr));
        let got = read_frames(&out, depth);
        assert_eq!(got.len(), frames, "the frame count is --frames");
        let plain_out = dir.path("plain.raw");
        let mut plain_params = params.to_vec();
        plain_params.push("src_mark_segments=5");
        let r = apply(bin, depth, &clip, &plain_out, frames, &plain_params, &layers);
        assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
        let plain = read_frames(&plain_out, depth);
        for (n, g) in got.iter().enumerate() {
            if n % 2 == 0 {
                assert_eq!(g, &exact_pass(&src[n / 2]), "{depth:?} output frame {n} is source frame {}", n / 2);
            } else {
                let truth = common::moving::DEFAULT.frame_at(depth, W, H, n as f64 / 2.0);
                let (flow_err, _) = common::moving::error(g, &truth, 9);
                let (plain_err, _) = common::moving::error(&plain[n], &truth, 9);
                assert!(flow_err < 0.0030, "{depth:?} frame {n}: Vectors In error {flow_err}");
                assert!(flow_err * 8.0 < plain_err, "{depth:?} frame {n}: {flow_err} against the plain blend {plain_err}");
            }
        }
    }
}

#[test]
fn the_effect_is_listed_as_temporal_with_its_six_layers() {
    let bin = vfx().expect("fx-cli builds");
    let out = Command::new(bin).args(["list", "--json"]).output().unwrap();
    assert!(out.status.success());
    let list: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let entry = list.as_array().expect("a list of effects").iter().find(|e| e["id"] == "vectors-in").expect("vectors-in is registered");
    assert_eq!(entry["kind"], "temporal");
    let layers: Vec<(&str, bool)> =
        entry["layers"].as_array().unwrap().iter().map(|l| (l["id"].as_str().unwrap(), l["optional"].as_bool().unwrap())).collect();
    assert_eq!(
        layers,
        vec![("source", true), ("vectors_prev", true), ("vectors_next", true), ("fg1_matte", true), ("fg2_matte", true), ("fg3_matte", true)]
    );
}

/// Without a vector layer the render succeeds with the red fill (half alpha red) on every frame.
#[test]
fn no_vector_layer_gives_the_red_fill_through_the_binary() {
    let bin = vfx().expect("fx-cli builds");
    let dir = Scratch::new("none");
    let clip = write_clip(dir.path("moving.raw"), &common::moving::DEFAULT.clip(Depth::U8, W, H, 4));
    let out = dir.path("out.raw");
    let r = apply(bin, Depth::U8, &clip, &out, 4, &[], &[]);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    for f in read_frames(&out, Depth::U8) {
        assert_eq!(f, fill(Depth::U8, true, W, H));
    }
}

/// What the front-end cannot do is a clean error of the binary (exit code 3, the message on stderr), not a wrong
/// picture: a FG Matte layer is the multi layer path. The default Warping is Forward, which renders between frames
/// (the engine has the forward warp), and so does a single vector layer; the exact frames pass through.
#[test]
fn unsupported_settings_are_a_clean_error_and_the_defaults_render_between_frames() {
    let bin = vfx().expect("fx-cli builds");
    let dir = Scratch::new("unsupported");
    let (clip, prev, next) = moving_files(&dir, Depth::U8);
    let layers = [("vectors_prev", prev.as_path()), ("vectors_next", next.as_path())];
    let out = dir.path("out.raw");
    let r = apply(bin, Depth::U8, &clip, &out, 4, &["speed_percent=50"], &layers);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    assert_eq!(read_frames(&out, Depth::U8).len(), 4);
    // one vector layer only: the other flow is its inversion
    let r = apply(bin, Depth::U8, &clip, &out, 4, &["speed_percent=50"], &[("vectors_next", next.as_path())]);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    // every frame on a source frame: the warp is never reached and the clip passes through
    let r = apply(bin, Depth::U8, &clip, &out, SOURCE_FRAMES, &[], &layers);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    assert_eq!(read_frames(&out, Depth::U8), read_frames(&clip, Depth::U8));
    // a FG Matte layer: the multi layer path
    let with_matte = [("vectors_prev", prev.as_path()), ("vectors_next", next.as_path()), ("fg1_matte", clip.as_path())];
    let r = apply(bin, Depth::U8, &clip, &out, 4, &["warping=1"], &with_matte);
    assert_eq!(r.status.code(), Some(3), "{}", String::from_utf8_lossy(&r.stderr));
    assert!(String::from_utf8_lossy(&r.stderr).contains("a FG Matte layer is set"));
}

/// A real `testsrc2` clip (needs ffmpeg) with vector layers of zero motion: exact frames pass the engine, the odd
/// frames of a half speed run with the segment marker "No MV, Frame Blend" are the 50/50 mix computed independently
/// here, and with the inverse warp and zero vectors they are the same mix within a level.
#[test]
fn testsrc2_clip_with_zero_vectors_at_8_bpc() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg is not on the PATH: skipping the end to end test");
        return;
    }
    let bin = vfx().expect("fx-cli builds");
    let dir = Scratch::new("testsrc");
    let clip = make_clip(&dir, Depth::U8);
    let zero = write_clip(dir.path("zero.raw"), &vector_clip(Depth::U8, W, H, 2 * SOURCE_FRAMES, 0.0, 0.0, MD));
    let layers = [("vectors_prev", zero.as_path()), ("vectors_next", zero.as_path())];
    let src = read_frames(&clip, Depth::U8);
    let out = dir.path("out.raw");
    let params = ["speed_percent=50", "warping=1", "vectors_max_displace=16", "src_mark_segments=5"];
    let r = apply(bin, Depth::U8, &clip, &out, 2 * SOURCE_FRAMES, &params, &layers);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let got = read_frames(&out, Depth::U8);
    assert_eq!(got.len(), 2 * SOURCE_FRAMES);
    for (n, g) in got.iter().enumerate() {
        let k = n / 2;
        let want: Vec<[u8; 4]> = if n % 2 == 0 {
            px8(&src[k]).to_vec()
        } else if k + 1 < SOURCE_FRAMES {
            px8(&src[k]).iter().zip(px8(&src[k + 1])).map(|(&a, &b)| mix8(a, b)).collect()
        } else {
            px8(&src[k]).iter().map(|&a| fade8(a)).collect()
        };
        assert_eq!(px8(g), &want[..], "output frame {n}");
    }
    let r = apply(bin, Depth::U8, &clip, &out, 2 * SOURCE_FRAMES - 2, &params[..3], &layers);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    for (n, g) in read_frames(&out, Depth::U8).iter().enumerate().filter(|(n, _)| n % 2 == 1) {
        let k = n / 2;
        for (p, (a, b)) in px8(g).iter().zip(px8(&src[k]).iter().zip(px8(&src[k + 1]))) {
            let m = mix8(*a, *b);
            for i in 0..4 {
                assert!((p[i] as i32 - m[i] as i32).abs() <= 1, "frame {n}: {p:?} against {m:?}");
            }
        }
    }
}
