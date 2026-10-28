//! Create Vectors end to end through the real `vfx` program, and the OUTPUT CONTRACT of the vector
//! image.
//!
//! The output of plugin 02 is an ordinary RGBA frame whose red and green channels CARRY DATA (the
//! motion, centred and scaled by Max Displace). Anything that treats the frame as a picture damages
//! it: `tools/apply.py` encodes with libx264 CRF 18 yuv420p (chroma subsampling, limited range,
//! lossy: most red and green samples change, and at the default Max Displace of 2048 px one 8 bpc
//! code is 16 px), so its mp4 is NOT a usable vector layer. A lossless path keeps every byte:
//! `vfx apply ... --out FILE` writes the raw frames and ffmpeg stores them with ffv1 (`rgba` as
//! `bgra`, `rgba64le`) or png, and decoding gives the same bytes back. The tests below run `vfx` on
//! a moving clip, decode the raw output, and pin both behaviours (they skip the ffmpeg parts
//! without ffmpeg).
//!
//! The test builds `vfx` with the cargo that runs it (`cargo build -p fx-cli`, same target
//! directory and profile).

mod common;

use common::moving::DEFAULT;
use common::{decode, flow_error};
use fx_core::{Depth, Frame};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

const W: usize = 64;
const H: usize = 48;
const SOURCE_FRAMES: usize = 8;
const MAX: f32 = 16.0;

fn workspace() -> PathBuf {
    // plugins/create-vectors -> the workspace root
    Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).expect("workspace root").to_path_buf()
}

/// Builds `vfx` once and returns its path, or None when it cannot be built.
fn vfx() -> Option<&'static Path> {
    static BIN: OnceLock<Option<PathBuf>> = OnceLock::new();
    BIN.get_or_init(|| {
        let exe = std::env::current_exe().ok()?;
        let profile_dir = exe.parent()?.parent()?.to_path_buf();
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
        let dir = std::env::temp_dir().join(format!("create-vectors-cli-{}-{name}", std::process::id()));
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

/// The synthetic clip with a known motion as raw frames in the wire format of `depth`.
fn make_moving_clip(dir: &Scratch, depth: Depth) -> PathBuf {
    let out = dir.path("moving.raw");
    let bytes: Vec<u8> = DEFAULT.clip(depth, W, H, SOURCE_FRAMES).iter().flat_map(|f| f.to_bytes()).collect();
    std::fs::write(&out, bytes).unwrap();
    out
}

fn apply(bin: &Path, depth: Depth, input: &Path, output: &Path, frames: usize, params: &[&str], layers: &[(&str, &Path)]) -> Output {
    let mut cmd = Command::new(bin);
    cmd.args(["apply", "create-vectors", "--width", &W.to_string(), "--height", &H.to_string(), "--depth", &depth.bits().to_string()])
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

fn read_frames(path: &Path, depth: Depth) -> Vec<Frame> {
    let bytes = std::fs::read(path).unwrap();
    let len = W * H * depth.bytes_per_pixel();
    assert_eq!(bytes.len() % len, 0, "a trailing partial frame");
    bytes.chunks(len).map(|c| Frame::from_bytes(W, H, depth, c)).collect()
}

/// `(mean error, worst error)` in pixels of the vectors read from the frames `from..to`, against
/// the known motion.
fn vector_error(frames: &[Frame], range: std::ops::Range<usize>, truth: (f64, f64)) -> (f64, f64) {
    let (mut mean, mut worst) = (0.0f64, 0.0f64);
    for f in &frames[range.clone()] {
        let (m, w, _) = flow_error(&decode(f, MAX), truth, 10);
        mean += m;
        worst = worst.max(w);
    }
    (mean / range.len() as f64, worst)
}

#[test]
fn the_effect_is_listed_as_temporal_with_its_four_layers() {
    let bin = vfx().expect("fx-cli builds");
    let out = Command::new(bin).args(["list", "--json"]).output().unwrap();
    assert!(out.status.success());
    let list: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let entry = list.as_array().expect("a list of effects").iter().find(|e| e["id"] == "create-vectors").expect("create-vectors is registered");
    assert_eq!(entry["kind"], "temporal");
    let layers: Vec<(&str, bool)> =
        entry["layers"].as_array().unwrap().iter().map(|l| (l["id"].as_str().unwrap(), l["optional"].as_bool().unwrap())).collect();
    assert_eq!(layers, vec![("alt_source_a", true), ("fg1_matte", true), ("fg2_matte", true), ("fg3_matte", true)]);
}

#[test]
fn the_binary_writes_vectors_that_decode_to_the_known_motion_at_every_depth() {
    let bin = vfx().expect("fx-cli builds");
    for depth in [Depth::U8, Depth::U16, Depth::F32] {
        let dir = Scratch::new(&format!("truth{}", depth.bits()));
        let clip = make_moving_clip(&dir, depth);
        for (track, truth) in [("1", (-3.4, -1.7)), ("2", (3.4, 1.7))] {
            let out = dir.path(&format!("vec{track}.raw"));
            let r = apply(bin, depth, &clip, &out, SOURCE_FRAMES, &["max_displace=16", &format!("track_frame={track}")], &[]);
            assert!(r.status.success(), "{depth:?}: {}", String::from_utf8_lossy(&r.stderr));
            let frames = read_frames(&out, depth);
            assert_eq!(frames.len(), SOURCE_FRAMES, "the frame count is --frames");
            // frames 2 to 5 have both neighbours inside the clip
            let (mean, worst) = vector_error(&frames, 2..6, truth);
            eprintln!("{depth:?} track {track}: mean {mean:.4} px, worst {worst:.4} px");
            assert!(mean < 0.15 && worst < 0.6, "{depth:?} track {track}: mean {mean}, worst {worst}");
        }
    }
}

#[test]
fn the_lossless_path_keeps_every_byte_and_the_libx264_path_damages_the_vectors() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg is not on the PATH: skipping the contract test");
        return;
    }
    let bin = vfx().expect("fx-cli builds");
    for depth in [Depth::U8, Depth::U16] {
        let dir = Scratch::new(&format!("contract{}", depth.bits()));
        let clip = make_moving_clip(&dir, depth);
        let raw = dir.path("vec.raw");
        let r = apply(bin, depth, &clip, &raw, SOURCE_FRAMES, &["max_displace=16"], &[]);
        assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
        let clean = read_frames(&raw, depth);
        let (clean_mean, _) = vector_error(&clean, 2..6, (-3.4, -1.7));
        let size = format!("{W}x{H}");
        let fmt = pix_fmt(depth);
        // lossless: ffv1 in matroska, then converted back to the same raw format
        let mkv = dir.path("vec.mkv");
        let enc = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", fmt, "-video_size", &size, "-framerate", "30", "-i"])
            .arg(&raw)
            .args(["-c:v", "ffv1"])
            .arg(&mkv)
            .status()
            .unwrap();
        assert!(enc.success());
        let back = dir.path("back.raw");
        let dec = Command::new("ffmpeg").args(["-v", "error", "-y", "-i"]).arg(&mkv).args(["-pix_fmt", fmt, "-f", "rawvideo"]).arg(&back).status().unwrap();
        assert!(dec.success());
        assert_eq!(std::fs::read(&back).unwrap(), std::fs::read(&raw).unwrap(), "{depth:?}: ffv1 round trip is bit exact");
        // what tools/apply.py does: libx264 CRF 18 yuv420p (tv range, bt709 matrix)
        let mp4 = dir.path("vec.mp4");
        let enc = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", fmt, "-video_size", &size, "-framerate", "30", "-i"])
            .arg(&raw)
            .args(["-vf", "scale=out_color_matrix=bt709:out_range=tv", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"])
            .arg(&mp4)
            .status()
            .unwrap();
        assert!(enc.success());
        let lossy = dir.path("lossy.raw");
        let dec = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-i"])
            .arg(&mp4)
            .args(["-vf", "scale=in_color_matrix=bt709", "-pix_fmt", fmt, "-f", "rawvideo"])
            .arg(&lossy)
            .status()
            .unwrap();
        assert!(dec.success());
        let degraded = read_frames(&lossy, depth);
        let (lossy_mean, lossy_worst) = vector_error(&degraded, 2..6, (-3.4, -1.7));
        // the damage in CODES of the red and green channels (interior pixels of frames 2 to 5): one 8 bpc code is
        // Max Displace / 127 px, so at the default Max Displace of 2048 px every code of damage is 16 px of error
        let (mut sum, mut differing, mut count) = (0.0f64, 0usize, 0usize);
        for k in 2..6 {
            for y in 10..H - 10 {
                for x in 10..W - 10 {
                    for c in 0..2 {
                        let (a, b) = (common::px(&clean[k], x, y)[c], common::px(&degraded[k], x, y)[c]);
                        sum += (a - b).abs();
                        differing += (a != b) as usize;
                        count += 1;
                    }
                }
            }
        }
        let (mean_codes, fraction) = (sum / count as f64, differing as f64 / count as f64);
        eprintln!(
            "{depth:?}: lossless mean error {clean_mean:.3} px; libx264 yuv420p mean {lossy_mean:.3} px (worst {lossy_worst:.2}),              {:.1}% of the R/G samples changed, mean change {mean_codes:.2} codes",
            fraction * 100.0
        );
        // The contract is the damage to the stored codes. Whether that damage moves the mean error against the true
        // motion depends on the encoder build (on this smooth clip one x264 build lands 0.3% closer), so it is
        // printed above but not asserted.
        assert!(fraction > 0.1, "{depth:?}: only {fraction} of the samples changed");
        let one_code_in_px = if depth == Depth::U8 { 1.0 } else { 1.0 / 257.0 };
        assert!(mean_codes > 0.1 * one_code_in_px, "{depth:?}: mean change {mean_codes} codes");
    }
}

#[test]
fn alt_src_through_the_binary_replaces_the_input() {
    let bin = vfx().expect("fx-cli builds");
    let dir = Scratch::new("alt");
    let clip = make_moving_clip(&dir, Depth::U8);
    // the input is a still black clip, Alt Src carries the motion: the output equals running the moving clip as the input
    let black = dir.path("black.raw");
    std::fs::write(&black, vec![0u8; W * H * 4 * SOURCE_FRAMES]).unwrap();
    let (a, b) = (dir.path("a.raw"), dir.path("b.raw"));
    assert!(apply(bin, Depth::U8, &black, &a, SOURCE_FRAMES, &["max_displace=16"], &[("alt_source_a", &clip)]).status.success());
    assert!(apply(bin, Depth::U8, &clip, &b, SOURCE_FRAMES, &["max_displace=16"], &[]).status.success());
    assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
}

/// A matte clip as raw 8 bpc frames: an opaque white disc of radius `r` on transparent black,
/// moving by `(2, 1)` px a frame from `(cx, cy)`.
fn make_disc_matte(dir: &Scratch, name: &str, cx: f64, cy: f64, r: f64) -> PathBuf {
    let out = dir.path(name);
    let mut bytes = Vec::with_capacity(W * H * 4 * SOURCE_FRAMES);
    for k in 0..SOURCE_FRAMES {
        for i in 0..W * H {
            let (x, y) = ((i % W) as f64, (i / W) as f64);
            let d = ((x - cx - 2.0 * k as f64).powi(2) + (y - cy - k as f64).powi(2)).sqrt();
            bytes.extend_from_slice(if d <= r { &[255, 255, 255, 255] } else { &[0, 0, 0, 0] });
        }
    }
    std::fs::write(&out, bytes).unwrap();
    out
}

/// The note line `vfx` prints for gap `id` with its register title.
fn note(id: &str) -> String {
    let title = match id {
        "G1" => "matte shrink tail",
        "G1b" => "mask-aware block matching cost",
        "G1c" => "layer visibility direction",
        "G1d" => "create-vectors merge",
        "G2" => "tracking point displacement",
        "G3" => "Draw Geom drawing",
        "G4" => "four Debug view compositions",
        "G5" => "second tracker arithmetic",
        other => panic!("no gap {other}"),
    };
    format!("vfx: note: create-vectors: {id} {title}: a stated rule that no Twixtor reference render has graded yet (README: limits)")
}

#[test]
fn the_layered_paths_render_through_the_binary_and_note_each_gap_once() {
    let bin = vfx().expect("fx-cli builds");
    let dir = Scratch::new("layered");
    let clip = make_moving_clip(&dir, Depth::U8);
    let fg1 = make_disc_matte(&dir, "fg1.raw", 22.0, 20.0, 12.0);
    let fg2 = make_disc_matte(&dir, "fg2.raw", 38.0, 26.0, 10.0);
    let out = dir.path("out.raw");
    let one: &[(&str, &Path)] = &[("fg1_matte", &fg1)];
    let two: &[(&str, &Path)] = &[("fg1_matte", &fg1), ("fg2_matte", &fg2)];
    let cases: Vec<(&[&str], &[(&str, &Path)], &[&str])> = vec![
        (&[], &[], &[]),
        (&["display=1"], one, &[]),
        (&["display=1", "pt1_use=2", "pt1_pos=20,15", "draw_geom=1"], &[], &[]),
        (&["display=3", "debug_view_method=1"], &[], &[]),
        (&["fg1_track=0", "fg2_sensitivity=10"], &[], &[]),
        (&[], one, &["G1", "G1b", "G1d"]),
        (&["fg1_inv_matte_shrink=2"], one, &["G1", "G1b", "G1d"]),
        (&["layers_output=3"], two, &["G1", "G1b", "G1c", "G1d"]),
        // no layer of the group tracks: no solve, so neither the masks nor the visibility order run
        (&["fg1_track=0", "fg2_track=0", "main_bg_track=0"], two, &["G1", "G1d", "G5"]),
        (&["display=3"], &[], &["G4"]),
        (&["display=3", "debug_view_method=3"], one, &["G1", "G1b", "G1d", "G4"]),
        (&["main_bg_track=0"], &[], &["G5"]),
        (&["main_bg_track=0", "pt1_use=5", "pt1_pos=20,15@0;26,18@7"], &[], &["G5"]),
        (&["pt1_use=5", "pt1_pos=20,15@0;26,18@7"], &[], &["G2"]),
        (&["display=1", "pt1_use=2", "pt1_pos=20,15"], &[], &["G3"]),
        (&["display=3", "debug_view_method=1", "pt1_use=2", "pt1_pos=20,15"], &[], &["G2", "G3"]),
    ];
    for (params, layers, gaps) in cases {
        let mut p = vec!["max_displace=16"];
        p.extend_from_slice(params);
        let r = apply(bin, Depth::U8, &clip, &out, SOURCE_FRAMES, &p, layers);
        let err = String::from_utf8_lossy(&r.stderr).to_string();
        assert_eq!(r.status.code(), Some(0), "{params:?}: {err}");
        assert_eq!(read_frames(&out, Depth::U8).len(), SOURCE_FRAMES);
        let mut lines: Vec<&str> = err.lines().collect();
        lines.sort_unstable();
        let mut want: Vec<String> = gaps.iter().map(|g| note(g)).collect();
        want.sort_unstable();
        assert_eq!(lines, want, "{params:?} with {} layers: each note once", layers.len());
    }
    // FG layers move the vectors: the layered output is not the single layer one
    let plain = dir.path("plain.raw");
    assert!(apply(bin, Depth::U8, &clip, &plain, SOURCE_FRAMES, &["max_displace=16"], &[]).status.success());
    assert!(apply(bin, Depth::U8, &clip, &out, SOURCE_FRAMES, &["max_displace=16", "layers_output=2"], one).status.success());
    assert_ne!(std::fs::read(&out).unwrap(), std::fs::read(&plain).unwrap());
}

#[test]
fn a_side_of_one_is_the_zero_vector_through_the_binary_and_notes_nothing() {
    let bin = vfx().expect("fx-cli builds");
    let dir = Scratch::new("side1");
    for (w, h) in [(1usize, 17usize), (17, 1), (1, 1)] {
        let clip = dir.path("in.raw");
        let matte = dir.path("matte.raw");
        let bytes: Vec<u8> = (0..w * h * 4 * 4).map(|i| (i * 37 % 251) as u8).collect();
        std::fs::write(&clip, &bytes).unwrap();
        std::fs::write(&matte, bytes.iter().rev().copied().collect::<Vec<u8>>()).unwrap();
        for params in [&["display=3", "debug_view_method=2"][..], &["main_bg_track=0", "pt1_use=5"], &["layers_output=2", "fg1_inv_matte_shrink=3"]] {
            let out = dir.path("out.raw");
            let mut cmd = Command::new(bin);
            cmd.args(["apply", "create-vectors", "--width", &w.to_string(), "--height", &h.to_string(), "--depth", "8", "--fps", "30", "--in"])
                .arg(&clip)
                .arg("--out")
                .arg(&out)
                .args(["--frames", "4"])
                .arg("--layer")
                .arg(format!("fg1_matte={}", matte.display()));
            for p in params {
                cmd.args(["--param", p]);
            }
            let r = cmd.output().unwrap();
            assert_eq!(r.status.code(), Some(0), "{w}x{h} {params:?}: {}", String::from_utf8_lossy(&r.stderr));
            assert!(r.stderr.is_empty(), "{w}x{h} {params:?}: {}", String::from_utf8_lossy(&r.stderr));
            let got = std::fs::read(&out).unwrap();
            assert_eq!(got.len(), w * h * 4 * 4);
            assert!(got.chunks(4).all(|p| p == [127, 127, 0, 255]), "{w}x{h} {params:?}");
        }
    }
}
