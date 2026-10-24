//! Black box tests of `vfx` as a command line program, the same way ffmpeg pipes drive it. The
//! generic ones hold for any effect whatever its math; the exact ones use the hidden `selftest-*`
//! effects.

use serde_json::Value;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::Mutex;

static SPAWN: Mutex<()> = Mutex::new(());

/// Windows lets a child inherit every inheritable handle in the process, so a spawn racing
/// another thread's would hold that thread's pipe open and delay its end of input.
fn spawn(args: &[&str], env: &[(&str, &str)]) -> Child {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vfx"));
    cmd.args(args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    for (k, v) in env {
        cmd.env(k, v);
    }
    let _one_at_a_time = SPAWN.lock().unwrap_or_else(|p| p.into_inner());
    cmd.spawn().expect("spawn vfx")
}

fn vfx_env(args: &[&str], stdin: &[u8], env: &[(&str, &str)]) -> Output {
    let mut child = spawn(args, env);
    let mut pipe = child.stdin.take().unwrap();
    let data = stdin.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = pipe.write_all(&data);
    });
    let out = child.wait_with_output().expect("wait vfx");
    writer.join().unwrap();
    out
}

fn vfx(args: &[&str], stdin: &[u8]) -> Output {
    vfx_env(args, stdin, &[])
}

fn list(all: bool) -> Vec<Value> {
    let args: &[&str] = if all { &["list", "--json", "--all"] } else { &["list", "--json"] };
    let out = vfx(args, &[]);
    assert_eq!(out.status.code(), Some(0));
    serde_json::from_slice::<Value>(&out.stdout).unwrap().as_array().unwrap().clone()
}

/// The effects `vfx list` shows by default.
fn effects() -> Vec<Value> {
    list(false)
}

/// A frame effect that is always there, whatever the plugins turn into.
const FRAME: &str = "selftest-invert";
const SHIFT: &str = "selftest-shift";
const INTEGRATE: &str = "selftest-integrate";

fn depth_bytes(depth: u32) -> usize {
    match depth {
        8 => 4,
        16 => 8,
        _ => 16,
    }
}

/// Finite, in range test pixels for every wire format.
fn frames(depth: u32, w: usize, h: usize, count: usize) -> Vec<u8> {
    let n = w * h * count;
    let mut out = Vec::new();
    match depth {
        8 => out.extend((0..n * 4).map(|i| (i * 37 % 251) as u8)),
        16 => out.extend((0..n * 4).flat_map(|i| ((i * 4099 % 65521) as u16).to_le_bytes())),
        _ => {
            for f in 0..count {
                for plane in 0..4 {
                    for i in 0..w * h {
                        let v = ((f * 31 + plane * 7 + i * 13) % 101) as f32 / 100.0;
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }
    }
    assert_eq!(out.len(), w * h * count * depth_bytes(depth));
    out
}

fn stderr_line(out: &Output) -> String {
    let text = String::from_utf8_lossy(&out.stderr).to_string();
    assert_eq!(text.trim_end().lines().count(), 1, "stderr should be one line: {text:?}");
    text.trim_end().to_string()
}

fn scratch_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("vfx-cli-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("vfx-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn names(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    v.sort();
    v
}

// ---- listing ------------------------------------------------------------------------------

#[test]
fn list_json_names_the_four_effects() {
    let list = effects();
    let ids: Vec<&str> = list.iter().map(|e| e["id"].as_str().unwrap()).collect();
    for want in ["retime", "create-vectors", "retime-fgbg", "vectors-in"] {
        assert!(ids.contains(&want), "{want} missing from {ids:?}");
    }
    for e in &list {
        assert!(e["name"].is_string() && e["family"].is_string() && e["params"].is_array());
        assert!(matches!(e["kind"].as_str(), Some("frame" | "temporal")), "{e}");
        assert!(e["layers"].is_array());
        assert_eq!(e["hidden"], false);
    }
}

#[test]
fn hidden_effects_need_all_but_always_run() {
    let shown: Vec<String> = effects().iter().map(|e| e["id"].as_str().unwrap().to_string()).collect();
    assert!(shown.iter().all(|id| !id.starts_with("selftest-")), "{shown:?}");
    let all = list(true);
    let find = |id: &str| all.iter().find(|e| e["id"] == id).unwrap_or_else(|| panic!("{id} missing from --all")).clone();
    for id in [FRAME, SHIFT, INTEGRATE] {
        assert_eq!(find(id)["hidden"], true);
    }
    assert_eq!(find(FRAME)["kind"], "frame");
    assert_eq!(find(SHIFT)["kind"], "temporal");
    assert_eq!(find(SHIFT)["layers"], serde_json::json!([{"id": "overlay", "name": "Overlay", "optional": true}]));
    assert_eq!(find(INTEGRATE)["layers"], serde_json::json!([]));
    assert_eq!(find("retime")["hidden"], false);
    let out = vfx(&["apply", FRAME, "--width", "1", "--height", "1", "--depth", "8"], &[1, 2, 3, 4]);
    assert_eq!(out.status.code(), Some(0));
    assert_eq!(out.stdout, [254, 253, 252, 4]);
}

#[test]
fn list_table_names_every_effect_and_its_kind() {
    let out = vfx(&["list", "--all"], &[]);
    assert_eq!(out.status.code(), Some(0));
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.starts_with("id"));
    assert_eq!(text.lines().next().unwrap().split_whitespace().collect::<Vec<_>>(), ["id", "name", "family", "kind", "params", "layers"]);
    for e in list(true) {
        let id = e["id"].as_str().unwrap();
        let row = text.lines().find(|l| l.starts_with(&format!("{id} "))).unwrap_or_else(|| panic!("{id} has no row"));
        assert!(row.contains(e["kind"].as_str().unwrap()), "{row}");
    }
    assert!(text.contains("selftest-shift layers"));
    let shown = String::from_utf8(vfx(&["list"], &[]).stdout).unwrap();
    assert!(!shown.contains("selftest-"));
}

// ---- the frame path, unchanged ---------------------------------------------------------------

/// `apply` arguments for any listed effect: temporal ones also need a rate.
fn apply_args(e: &Value, w: &str, h: &str, depth: &str) -> Vec<String> {
    let mut args: Vec<String> =
        ["apply", e["id"].as_str().unwrap(), "--width", w, "--height", h, "--depth", depth].iter().map(|s| s.to_string()).collect();
    if e["kind"] == "temporal" {
        args.extend(["--fps".to_string(), "24".to_string()]);
    }
    args
}

fn run_args(args: &[String], stdin: &[u8]) -> Output {
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    vfx(&refs, stdin)
}

/// The generic loops below feed every effect the same bytes; an effect that cannot run without
/// a layer of its own has tests of its own.
fn has_required_layer(e: &Value) -> bool {
    e["layers"].as_array().is_some_and(|l| l.iter().any(|layer| layer["optional"] == false))
}

/// A listed (non hidden) temporal effect may honestly say it does not support a parameter set yet.
fn unsupported(out: &Output) -> bool {
    out.status.code() == Some(3) && String::from_utf8_lossy(&out.stderr).contains("unsupported")
}

#[test]
fn every_effect_keeps_the_frame_size_at_every_depth() {
    for e in list(true).into_iter().filter(|e| !has_required_layer(e)) {
        let id = e["id"].as_str().unwrap();
        for depth in [8u32, 16, 32] {
            let input = frames(depth, 5, 3, 2);
            let args = apply_args(&e, "5", "3", &depth.to_string());
            let out = run_args(&args, &input);
            if e["hidden"] == false && e["kind"] == "temporal" && unsupported(&out) {
                continue;
            }
            assert_eq!(out.status.code(), Some(0), "{id} @{depth}: {}", String::from_utf8_lossy(&out.stderr));
            assert_eq!(out.stdout.len(), input.len(), "{id} @{depth}");
            assert_eq!(run_args(&args, &input).stdout, out.stdout, "{id} @{depth} is not deterministic");
        }
    }
}

#[test]
fn one_by_one_frames_work() {
    for depth in [8u32, 16, 32] {
        let input = frames(depth, 1, 1, 1);
        let d = depth.to_string();
        let out = vfx(&["apply", FRAME, "--width", "1", "--height", "1", "--depth", &d], &input);
        assert_eq!(out.status.code(), Some(0));
        assert_eq!(out.stdout.len(), input.len());
    }
}

#[test]
fn empty_input_is_zero_frames() {
    let out = vfx(&["apply", FRAME, "--width", "4", "--height", "4", "--depth", "8"], &[]);
    assert_eq!(out.status.code(), Some(0));
    assert!(out.stdout.is_empty());
    let out = vfx(&["apply", SHIFT, "--width", "4", "--height", "4", "--depth", "8", "--fps", "24"], &[]);
    assert_eq!(out.status.code(), Some(0));
    assert!(out.stdout.is_empty());
}

#[test]
fn trailing_partial_frame_is_an_io_error() {
    let mut input = frames(8, 4, 4, 1);
    input.extend_from_slice(&[1, 2, 3]);
    let out = vfx(&["apply", FRAME, "--width", "4", "--height", "4", "--depth", "8"], &input);
    assert_eq!(out.status.code(), Some(4));
    assert!(stderr_line(&out).contains("partial frame"));
    // a temporal effect refuses the same input before it renders anything
    let out = vfx(&["apply", SHIFT, "--width", "4", "--height", "4", "--depth", "8", "--fps", "24"], &input);
    assert_eq!(out.status.code(), Some(4));
    assert!(stderr_line(&out).contains("partial frame (3 of 64 bytes) after 1 whole frames"));
    assert!(out.stdout.is_empty());
}

#[test]
fn unknown_effect_and_param_exit_3() {
    let out = vfx(&["apply", "no-such-effect", "--width", "1", "--height", "1", "--depth", "8"], &[]);
    assert_eq!(out.status.code(), Some(3));
    stderr_line(&out);

    for id in [FRAME, SHIFT] {
        let out = vfx(&["apply", id, "--width", "1", "--height", "1", "--depth", "8", "--fps", "24", "--param", "no_such_param=1"], &[]);
        assert_eq!(out.status.code(), Some(3), "{id}");
        assert!(stderr_line(&out).contains("no_such_param"));
    }
}

#[test]
fn every_param_default_round_trips_through_the_cli() {
    for e in list(true).into_iter().filter(|e| !has_required_layer(e)) {
        let id = e["id"].as_str().unwrap();
        for p in e["params"].as_array().unwrap() {
            let text = match p["kind"].as_str().unwrap() {
                "color" | "point" => p["default"].as_array().unwrap().iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","),
                "checkbox" => u8::from(p["default"].as_bool().unwrap()).to_string(),
                _ => p["default"].to_string(),
            };
            let kv = format!("{}={}", p["id"].as_str().unwrap(), text);
            let mut args = apply_args(&e, "2", "2", "8");
            args.extend(["--param".to_string(), kv.clone()]);
            let out = run_args(&args, &[]);
            assert_eq!(out.status.code(), Some(0), "{id} {kv}: {}", String::from_utf8_lossy(&out.stderr));
        }
    }
}

#[test]
fn usage_errors_exit_2() {
    for args in [
        &[][..],
        &["apply"],
        &["apply", "x", "--width", "4", "--height", "4"],
        &["apply", "x", "--width", "4", "--height", "4", "--depth", "9"],
        &["list", "--nope"],
        &["wat"],
        &["apply", SHIFT, "--width", "4", "--height", "4", "--depth", "8", "--frames", "0"],
        &["apply", SHIFT, "--width", "4", "--height", "4", "--depth", "8", "--layer", "novalue"],
        &["apply", SHIFT, "--width", "4", "--height", "4", "--depth", "8", "--quality", "ultra"],
    ] {
        let out = vfx(args, &[]);
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        stderr_line(&out);
    }
}

#[test]
fn help_and_version_succeed() {
    for flag in ["--help", "--version"] {
        let out = vfx(&[flag], &[]);
        assert_eq!(out.status.code(), Some(0));
        assert!(!out.stdout.is_empty());
    }
    let help = String::from_utf8(vfx(&["--help"], &[]).stdout).unwrap();
    for word in ["--fps N[/D]", "--frames N", "--time-scale", "--layer ID=FILE", "--tmp-dir", "--check", "quote the argument", "V@F;V@F"] {
        assert!(help.contains(word), "the help should mention {word}");
    }
}

#[test]
fn file_input_and_output_work_and_may_not_alias() {
    let (src, dst) = (scratch("in.rgba"), scratch("out.rgba"));
    let input = frames(8, 3, 2, 3);
    std::fs::write(&src, &input).unwrap();
    let (s, d) = (src.to_str().unwrap(), dst.to_str().unwrap());
    let out = vfx(&["apply", FRAME, "--width", "3", "--height", "2", "--depth", "8", "--in", s, "--out", d], &[]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(std::fs::read(&dst).unwrap().len(), input.len());

    let out = vfx(&["apply", FRAME, "--width", "3", "--height", "2", "--depth", "8", "--in", s, "--out", s], &[]);
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(std::fs::read(&src).unwrap(), input, "input must survive");

    let out = vfx(&["apply", FRAME, "--width", "3", "--height", "2", "--depth", "8", "--in", "does-not-exist.raw"], &[]);
    assert_eq!(out.status.code(), Some(4));
    let _ = std::fs::remove_dir_all(src.parent().unwrap());
}

#[test]
fn a_frame_effect_refuses_temporal_only_options_and_keyframes() {
    let base = ["apply", FRAME, "--width", "2", "--height", "2", "--depth", "8"];
    for extra in [&["--frames", "2"][..], &["--layer", "a=b.raw"], &["--time-scale", "2400"], &["--quality", "draft"], &["--field", "upper"], &["--tmp-dir", "."]] {
        let args: Vec<&str> = base.iter().chain(extra.iter()).copied().collect();
        let out = vfx(&args, &[]);
        assert_eq!(out.status.code(), Some(2), "{extra:?}");
        assert!(stderr_line(&out).contains("only apply to temporal effects"));
    }
    let args: Vec<&str> = base.iter().copied().chain(["--param", "invert_alpha=1@0;0@5"]).collect();
    let out = vfx(&args, &[]);
    assert_eq!(out.status.code(), Some(3));
    assert!(stderr_line(&out).contains("keyframes need a temporal effect"));
    // --fps stays optional, and stays a plain number for a frame effect
    for fps in [None, Some("24"), Some("30000/1001"), Some("29.97002997002997")] {
        let mut args: Vec<&str> = base.to_vec();
        if let Some(f) = fps {
            args.extend(["--fps", f]);
        }
        assert_eq!(vfx(&args, &[0; 16]).status.code(), Some(0), "{fps:?}");
    }
}

// ---- temporal effects ------------------------------------------------------------------------

const W: usize = 3;
const H: usize = 2;

fn base_args<'a>(id: &'a str, depth: &'a str) -> Vec<&'a str> {
    vec!["apply", id, "--width", "3", "--height", "2", "--depth", depth, "--fps", "24"]
}

/// Frame `i` of a clip of `frames(depth, W, H, count)`.
fn frame_of(clip: &[u8], depth: u32, i: usize) -> &[u8] {
    let len = W * H * depth_bytes(depth);
    &clip[i * len..(i + 1) * len]
}

fn shift_oracle(clip: &[u8], depth: u32, count: usize, frames: usize, offset: i64) -> Vec<u8> {
    let len = W * H * depth_bytes(depth);
    let mut out = Vec::new();
    for n in 0..frames as i64 {
        let src = n + offset;
        if (0..count as i64).contains(&src) {
            out.extend_from_slice(frame_of(clip, depth, src as usize));
        } else {
            out.extend(std::iter::repeat_n(0u8, len));
        }
    }
    out
}

#[test]
fn shift_matches_a_hand_computed_oracle_at_every_depth_by_file_and_by_pipe() {
    let dir = scratch_dir("shift");
    for depth in [8u32, 16, 32] {
        let clip = frames(depth, W, H, 6);
        let path = dir.join(format!("in{depth}.raw"));
        std::fs::write(&path, &clip).unwrap();
        for offset in [-8i64, -2, 0, 1, 3, 8] {
            let d = depth.to_string();
            let off = format!("offset={offset}");
            let want = shift_oracle(&clip, depth, 6, 6, offset);
            let mut args = base_args(SHIFT, &d);
            args.extend(["--param", &off]);
            let piped = vfx(&args, &clip);
            assert_eq!(piped.status.code(), Some(0), "{}", String::from_utf8_lossy(&piped.stderr));
            assert_eq!(piped.stdout, want, "pipe, depth {depth}, offset {offset}");
            args.extend(["--in", path.to_str().unwrap()]);
            let filed = vfx(&args, &[]);
            assert_eq!(filed.stdout, want, "file, depth {depth}, offset {offset}");
        }
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn frames_option_cuts_or_extends_the_output() {
    let clip = frames(8, W, H, 6);
    for (n, offset) in [(3usize, 0i64), (9, 0), (9, 2), (1, 5)] {
        let (frames_arg, off) = (n.to_string(), format!("offset={offset}"));
        let mut args = base_args(SHIFT, "8");
        args.extend(["--frames", &frames_arg, "--param", &off]);
        let out = vfx(&args, &clip);
        assert_eq!(out.status.code(), Some(0));
        assert_eq!(out.stdout, shift_oracle(&clip, 8, 6, n, offset), "frames {n} offset {offset}");
    }
}

#[test]
fn the_overlay_layer_is_mixed_in_and_may_have_another_length() {
    let dir = scratch_dir("overlay");
    let clip = frames(8, W, H, 5);
    let mut other = frames(8, W, H, 3);
    other.iter_mut().for_each(|b| *b = b.wrapping_mul(3).wrapping_add(11));
    let (cp, op) = (dir.join("clip.raw"), dir.join("over.raw"));
    std::fs::write(&cp, &clip).unwrap();
    std::fs::write(&op, &other).unwrap();
    let spec = format!("overlay={}", op.display());
    let mut args = base_args(SHIFT, "8");
    args.extend(["--param", "offset=1", "--in", cp.to_str().unwrap(), "--layer", &spec]);
    let out = vfx(&args, &[]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let shifted = shift_oracle(&clip, 8, 5, 5, 1);
    let len = W * H * 4;
    let mut want = Vec::new();
    for n in 0..5 {
        for i in 0..len {
            let over = if n < 3 { other[n * len + i] } else { 0 };
            want.push(((shifted[n * len + i] as u16 + over as u16 + 1) >> 1) as u8);
        }
    }
    assert_eq!(out.stdout, want);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn layer_mistakes_have_their_own_exit_codes() {
    let dir = scratch_dir("layers");
    let clip = frames(8, W, H, 2);
    let good = dir.join("good.raw");
    std::fs::write(&good, &clip).unwrap();
    let mut ragged = clip.clone();
    ragged.push(7);
    let bad = dir.join("bad.raw");
    std::fs::write(&bad, &ragged).unwrap();
    let empty = dir.join("empty.raw");
    std::fs::write(&empty, []).unwrap();
    let try_layer = |id: &str, path: &Path| {
        let mut args = base_args(SHIFT, "8");
        let spec = format!("{id}={}", path.display());
        args.extend(["--layer", &spec]);
        vfx(&args, &clip)
    };
    let out = try_layer("nope", &good);
    assert_eq!(out.status.code(), Some(3));
    assert!(stderr_line(&out).contains("unknown layer `nope`") && stderr_line(&out).contains("overlay"));
    let out = try_layer("overlay", &dir.join("missing.raw"));
    assert_eq!(out.status.code(), Some(4));
    stderr_line(&out);
    let out = try_layer("overlay", &bad);
    assert_eq!(out.status.code(), Some(4));
    assert!(stderr_line(&out).contains("layer overlay") && stderr_line(&out).contains("partial frame"));
    let out = try_layer("overlay", &empty);
    assert_eq!(out.status.code(), Some(4));
    assert!(stderr_line(&out).contains("holds no frames"));
    let out = try_layer("overlay", &good);
    assert_eq!(out.status.code(), Some(0));
    // the same layer twice is a usage error
    let mut args = base_args(SHIFT, "8");
    let spec = format!("overlay={}", good.display());
    args.extend(["--layer", &spec, "--layer", &spec]);
    assert_eq!(vfx(&args, &clip).status.code(), Some(2));
    // the output may not be one of the inputs
    let mut args = base_args(SHIFT, "8");
    args.extend(["--layer", &spec, "--out", good.to_str().unwrap()]);
    assert_eq!(vfx(&args, &clip).status.code(), Some(2));
    assert_eq!(std::fs::read(&good).unwrap(), clip, "the layer must survive");
    // an effect without layers rejects any
    let mut args = base_args(INTEGRATE, "8");
    args.extend(["--layer", &spec]);
    let out = vfx(&args, &clip);
    assert_eq!(out.status.code(), Some(3));
    assert!(stderr_line(&out).contains("has no layers"));
    let _ = std::fs::remove_dir_all(dir);
}

/// 8 bpc frames that are `value` in every channel.
fn solid(value: u8, count: usize) -> Vec<u8> {
    vec![value; W * H * 4 * count]
}

fn firsts(out: &[u8]) -> Vec<u8> {
    out.chunks(W * H * 4).map(|f| f[0]).collect()
}

#[test]
fn keyframed_parameters_are_sampled_at_every_earlier_frame() {
    let clip = solid(100, 6);
    let run = |param: &str, extra: &[&str]| {
        let mut args = base_args(INTEGRATE, "8");
        args.extend(["--param", param]);
        args.extend(extra);
        let out = vfx(&args, &clip);
        assert_eq!(out.status.code(), Some(0), "{param}: {}", String::from_utf8_lossy(&out.stderr));
        firsts(&out.stdout)
    };
    // constant gain: frame n gains n * 2
    assert_eq!(run("gain=2", &[]), [100, 102, 104, 106, 108, 110]);
    // a ramp 0 -> 10 over frames 0..4 samples 0, 2.5, 5, 7.5, 10; the sums are 0, 0, 2.5, 7.5, 15, 25
    assert_eq!(run("gain=0@0;10@4", &[]), [100, 100, 103, 108, 115, 125]);
    // a hold key keeps 4 until the next key: samples 4, 4, 8, 8, 8, 8
    assert_eq!(run("gain=4@0h;8@2", &[]), [100, 104, 108, 116, 124, 132]);
    // keys may come in any order and sit between frames
    assert_eq!(run("gain=10@4;0@0", &[]), [100, 100, 103, 108, 115, 125]);
}

#[test]
fn the_time_base_changes_what_a_between_frames_key_means() {
    let clip = solid(100, 5);
    let run = |extra: &[&str]| {
        let mut args = base_args(INTEGRATE, "8");
        args.extend(["--param", "gain=0@0;10@2.5"]);
        args.extend(extra);
        let out = vfx(&args, &clip);
        assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
        firsts(&out.stdout)
    };
    // 24 fps default: 100 ticks a frame, the key sits at tick 250, samples 0, 4, 8, 10
    assert_eq!(run(&[]), [100, 100, 104, 112, 122]);
    assert_eq!(run(&["--time-scale", "2400"]), run(&[]));
    // one tick a frame: the key rounds to tick 3, samples 0, 3.33, 6.67, 10
    assert_eq!(run(&["--time-scale", "24"]), [100, 100, 103, 110, 120]);
    // 1000 ticks a second is 41.67 ticks a frame
    let mut args = base_args(INTEGRATE, "8");
    args.extend(["--time-scale", "1000"]);
    let out = vfx(&args, &clip);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr_line(&out).contains("whole number of ticks"));
}

#[test]
fn fps_is_required_exact_and_parsed_like_a_rate() {
    let args = ["apply", SHIFT, "--width", "3", "--height", "2", "--depth", "8"];
    let out = vfx(&args, &[]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr_line(&out).contains("needs --fps"));
    let mut inexact = args.to_vec();
    inexact.extend(["--fps", "29.97002997002997"]);
    let out = vfx(&inexact, &[]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr_line(&out).contains("not an exact rate"));
    for fps in ["24", "29.97", "30000/1001", "60000/2002"] {
        let mut ok = args.to_vec();
        ok.extend(["--fps", fps]);
        assert_eq!(vfx(&ok, &solid(1, 2)).status.code(), Some(0), "{fps}");
    }
}

#[test]
fn quality_and_field_are_accepted_by_temporal_effects() {
    for extra in [&["--quality", "draft"][..], &["--quality", "high", "--field", "upper"], &["--field", "lower"]] {
        let mut args = base_args(INTEGRATE, "8");
        args.extend(extra);
        let out = vfx(&args, &solid(5, 2));
        assert_eq!(out.status.code(), Some(0), "{extra:?}");
        assert_eq!(out.stdout.len(), 2 * W * H * 4);
    }
}

#[test]
fn check_validates_and_touches_nothing() {
    let dir = scratch_dir("check");
    let out_path = dir.join("never.raw");
    let spec = format!("overlay={}", dir.join("not-there.raw").display());
    let mut args = base_args(SHIFT, "8");
    args.extend(["--check", "--param", "offset=-2@0;3@10", "--layer", &spec, "--in", "nope.raw", "--out", out_path.to_str().unwrap()]);
    let out = vfx(&args, &[]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(out.stdout.is_empty() && !out_path.exists());
    for (extra, code) in [(&["--param", "offset=99"][..], 3), (&["--layer", "nope=x"], 3), (&["--time-scale", "1000"], 2), (&["--param", "offset=1@0;2@0"], 3)] {
        let mut args = base_args(SHIFT, "8");
        args.push("--check");
        args.extend(extra);
        assert_eq!(vfx(&args, &[]).status.code(), Some(code), "{extra:?}");
    }
    // frame effects check too
    let out = vfx(&["apply", FRAME, "--width", "2", "--height", "2", "--depth", "8", "--check", "--param", "invert_alpha=1"], &[]);
    assert_eq!(out.status.code(), Some(0));
    let out = vfx(&["apply", FRAME, "--width", "2", "--height", "2", "--depth", "8", "--check", "--param", "x=1"], &[]);
    assert_eq!(out.status.code(), Some(3));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_spool_file_is_gone_after_success_and_after_every_failure() {
    let dir = scratch_dir("spool");
    let tmp = dir.join("tmp");
    std::fs::create_dir_all(&tmp).unwrap();
    let t = tmp.to_str().unwrap();
    let clip = frames(8, W, H, 4);
    let run = |extra: &[&str], stdin: &[u8]| {
        let mut args = base_args(SHIFT, "8");
        args.extend(["--tmp-dir", t]);
        args.extend(extra);
        let out = vfx(&args, stdin);
        assert!(names(&tmp).is_empty(), "left behind after {extra:?}: {:?}", names(&tmp));
        out
    };
    assert_eq!(run(&["--param", "offset=1"], &clip).status.code(), Some(0));
    // the layer fails after the input was already spooled
    let ragged = dir.join("ragged.raw");
    std::fs::write(&ragged, [1, 2, 3]).unwrap();
    let spec = format!("overlay={}", ragged.display());
    assert_eq!(run(&["--layer", &spec], &clip).status.code(), Some(4));
    // a trailing partial frame in the piped input
    let mut partial = clip.clone();
    partial.push(9);
    assert_eq!(run(&[], &partial).status.code(), Some(4));
    // a bad parameter never even creates it
    assert_eq!(run(&["--param", "offset=99"], &clip).status.code(), Some(3));
    // the spool is also deleted when stdout is closed on it mid run: write to a missing directory
    let nowhere = dir.join("no").join("out.raw");
    assert_eq!(run(&["--out", nowhere.to_str().unwrap()], &clip).status.code(), Some(4));
    // a --tmp-dir that does not exist is an io error
    let absent = dir.join("absent");
    let mut args = base_args(SHIFT, "8");
    args.extend(["--tmp-dir", absent.to_str().unwrap()]);
    let out = vfx(&args, &clip);
    assert_eq!(out.status.code(), Some(4));
    assert!(stderr_line(&out).contains("cannot create a spool file"));
    let _ = std::fs::remove_dir_all(dir);
}

/// Starts vfx, feeds it a clip but keeps stdin open, and waits until a spool file shows up
/// in `watch`. Then it closes stdin, lets the run finish and returns what it saw there.
fn spooled_names_while_running(args: &[&str], env: &[(&str, &str)], watch: &Path, clip: &[u8]) -> (Vec<String>, Output) {
    let mut child = spawn(args, env);
    let mut stdin = child.stdin.take().unwrap();
    stdin.write_all(clip).unwrap();
    stdin.flush().unwrap();
    let mut seen = Vec::new();
    for _ in 0..200 {
        seen = names(watch);
        if seen.iter().any(|n| n.starts_with("vfx-spool-")) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    drop(stdin);
    (seen, child.wait_with_output().unwrap())
}

#[test]
fn the_default_spool_directory_is_the_os_temp_dir() {
    let dir = scratch_dir("ostmp");
    let clip = frames(8, W, H, 2);
    let d = dir.to_str().unwrap();
    let env = [("TMP", d), ("TEMP", d), ("TMPDIR", d)];
    let (seen, out) = spooled_names_while_running(&base_args(INTEGRATE, "8"), &env, &dir, &clip);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(seen.len(), 1, "{seen:?}");
    assert!(seen[0].starts_with("vfx-spool-") && seen[0].ends_with(".raw"), "{seen:?}");
    assert!(names(&dir).is_empty(), "{:?}", names(&dir));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn tmp_dir_moves_the_spool_file() {
    let dir = scratch_dir("tmpdir");
    let (here, elsewhere) = (dir.join("here"), dir.join("elsewhere"));
    std::fs::create_dir_all(&here).unwrap();
    std::fs::create_dir_all(&elsewhere).unwrap();
    let e = elsewhere.to_str().unwrap();
    let env = [("TMP", e), ("TEMP", e), ("TMPDIR", e)];
    let mut args = base_args(INTEGRATE, "8");
    args.extend(["--tmp-dir", here.to_str().unwrap()]);
    let (seen, out) = spooled_names_while_running(&args, &env, &here, &frames(8, W, H, 2));
    assert_eq!(out.status.code(), Some(0));
    assert_eq!(seen.len(), 1, "{seen:?}");
    assert!(names(&here).is_empty() && names(&elsewhere).is_empty());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn parallel_runs_never_share_a_spool_file() {
    let dir = scratch_dir("parallel");
    let t = dir.to_str().unwrap().to_string();
    std::thread::scope(|scope| {
        for k in 0..6u8 {
            let t = t.clone();
            scope.spawn(move || {
                let clip = solid(10 + k, 3);
                for _ in 0..3 {
                    let mut args = base_args(INTEGRATE, "8");
                    args.extend(["--tmp-dir", &t, "--param", "gain=1"]);
                    let out = vfx(&args, &clip);
                    assert_eq!(out.status.code(), Some(0));
                    assert_eq!(firsts(&out.stdout), [10 + k, 11 + k, 12 + k], "runner {k}");
                }
            });
        }
    });
    assert!(names(&dir).is_empty(), "{:?}", names(&dir));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn output_is_identical_for_any_rayon_thread_count() {
    let (w, h, count) = (64usize, 48usize, 4usize);
    for depth in [8u32, 16, 32] {
        let clip = frames(depth, w, h, count);
        let overlay = frames(depth, w, h, count).iter().map(|b| b.wrapping_add(9)).collect::<Vec<u8>>();
        let dir = scratch_dir(&format!("threads{depth}"));
        let op = dir.join("o.raw");
        std::fs::write(&op, &overlay).unwrap();
        let (ws, hs, ds) = (w.to_string(), h.to_string(), depth.to_string());
        for id in [SHIFT, INTEGRATE] {
            let spec = format!("overlay={}", op.display());
            let mut args = vec!["apply", id, "--width", &ws, "--height", &hs, "--depth", &ds, "--fps", "24", "--param"];
            args.push(if id == SHIFT { "offset=-1" } else { "gain=0@0;40@3" });
            if id == SHIFT {
                args.extend(["--layer", &spec]);
            }
            let default = vfx_env(&args, &clip, &[]);
            assert_eq!(default.status.code(), Some(0), "{}", String::from_utf8_lossy(&default.stderr));
            for threads in ["1", "2", "7"] {
                let other = vfx_env(&args, &clip, &[("RAYON_NUM_THREADS", threads)]);
                assert_eq!(other.stdout, default.stdout, "{id} @{depth} with {threads} threads");
            }
            assert_eq!(vfx_env(&args, &clip, &[]).stdout, default.stdout, "{id} @{depth} twice");
        }
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[test]
fn output_to_a_file_matches_stdout() {
    let dir = scratch_dir("out");
    let clip = frames(16, W, H, 3);
    let path = dir.join("o.raw");
    let mut args = base_args(SHIFT, "16");
    args.extend(["--param", "offset=1"]);
    let piped = vfx(&args, &clip);
    args.extend(["--out", path.to_str().unwrap()]);
    let filed = vfx(&args, &clip);
    assert_eq!(filed.status.code(), Some(0));
    assert!(filed.stdout.is_empty());
    assert_eq!(std::fs::read(&path).unwrap(), piped.stdout);
    let _ = std::fs::remove_dir_all(dir);
}
