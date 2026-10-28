//! Black box tests of the render device surface of `vfx`: `vfx devices` and `vfx apply --device`.
//! `vfx devices` lists the CPU and whatever OpenCL devices the machine has (none on CI). `gpu`
//! renders on the best device that passes the gate or on the CPU after one note, `gpu=N` is strict
//! (exit 3 with the reasons), and every device gives the bytes of a run on the CPU.

use serde_json::Value;
use std::io::Write;
use std::process::{Command, Output, Stdio};
use std::sync::Mutex;

static SPAWN: Mutex<()> = Mutex::new(());

const SHIFT: &str = "selftest-shift";
const FRAME: &str = "selftest-invert";

fn vfx(args: &[&str], stdin: &[u8]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_vfx"));
    cmd.args(args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    // one spawn at a time, so no child inherits another test's pipe (Windows)
    let mut child = {
        let _one_at_a_time = SPAWN.lock().unwrap_or_else(|p| p.into_inner());
        cmd.spawn().expect("spawn vfx")
    };
    let mut pipe = child.stdin.take().unwrap();
    let data = stdin.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = pipe.write_all(&data);
    });
    let out = child.wait_with_output().expect("wait vfx");
    writer.join().unwrap();
    out
}

fn stderr_line(out: &Output) -> String {
    let text = String::from_utf8_lossy(&out.stderr).to_string();
    assert_eq!(text.trim_end().lines().count(), 1, "stderr should be one line: {text:?}");
    text.trim_end().to_string()
}

/// `count` frames of `w x h` at `depth` bits whose content moves from frame to frame.
fn clip(depth: u32, w: usize, h: usize, count: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for f in 0..count {
        match depth {
            8 => out.extend((0..w * h * 4).map(|i| ((i * 37 + f * 11) % 251) as u8)),
            16 => out.extend((0..w * h * 4).flat_map(|i| (((i * 4099 + f * 977) % 65521) as u16).to_le_bytes())),
            _ => {
                for plane in 0..4 {
                    for i in 0..w * h {
                        let v = ((f * 31 + plane * 7 + i * 13) % 101) as f32 / 100.0;
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }
    }
    out
}

fn shift_args<'a>(extra: &[&'a str]) -> Vec<&'a str> {
    let mut args = vec!["apply", SHIFT, "--width", "3", "--height", "2", "--depth", "8", "--fps", "24"];
    args.extend_from_slice(extra);
    args
}

#[test]
fn devices_lists_the_cpu_and_every_opencl_device() {
    let out = vfx(&["devices", "--json"], &[]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    let list = v.as_array().unwrap();
    assert_eq!(list[0], serde_json::json!({"id": "cpu", "name": "CPU", "kind": "cpu", "default": true}));
    let json_note = String::from_utf8(out.stderr).unwrap();
    let text_out = vfx(&["devices"], &[]);
    assert_eq!(text_out.status.code(), Some(0));
    let text = String::from_utf8(text_out.stdout).unwrap();
    let note = String::from_utf8(text_out.stderr).unwrap();
    assert_eq!(note, json_note, "the same notes for both forms");
    if list.len() == 1 {
        // no OpenCL device here (CI): the CPU and one note
        println!("no OpenCL device: {note}");
        assert_eq!(text, "id   name  kind  default
cpu  CPU   cpu   yes
");
        assert_eq!(note.lines().count(), 1, "{note:?}");
        assert!(note.starts_with("vfx: note: ") && note.contains("listing the CPU only"), "{note:?}");
    } else {
        println!("{text}");
        assert!(note.is_empty() || note.lines().all(|l| l.starts_with("vfx: note: OpenCL platform skipped")), "{note:?}");
        assert!(text.starts_with("id "), "{text}");
        assert!(text.lines().next().unwrap().ends_with("verdict"));
    }
    for (n, d) in list.iter().enumerate().skip(1) {
        assert_eq!(d["id"], format!("gpu={}", n - 1), "numbered in order");
        assert_eq!(d["kind"], "gpu");
        assert_eq!(d["default"], false);
        for key in ["platform", "vendor", "version", "driver", "global_memory", "single_fp", "double_fp", "verdict", "reasons"] {
            assert!(d.get(key).is_some(), "{key} in {d}");
        }
        let verdict = d["verdict"].as_str().unwrap();
        let reasons = d["reasons"].as_array().unwrap();
        assert!(verdict == "ok" && reasons.is_empty() || verdict == "refused" && !reasons.is_empty(), "{d}");
        let id = d["id"].as_str().unwrap();
        let name = d["name"].as_str().unwrap();
        assert!(text.contains(&format!("
{id}  {name}")), "the table names {id}");
        // a listed device that meets the requirements passes the check (the gate runs), one that
        // does not is refused with its reasons
        let out = vfx(&shift_args(&["--device", id, "--check"]), &[]);
        if verdict == "ok" {
            assert_eq!(out.status.code(), Some(0), "{id}: {}", String::from_utf8_lossy(&out.stderr));
            assert!(out.stderr.is_empty(), "{id}");
        } else {
            assert_eq!(out.status.code(), Some(3), "{id}");
            assert!(stderr_line(&out).starts_with(&format!("vfx: error: unsupported: --device {id}: ")));
        }
    }
    // the CPU id is a value --device takes
    let out = vfx(&shift_args(&["--device", "cpu", "--check"]), &[]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn devices_refuses_other_arguments() {
    for args in [&["devices", "--all"][..], &["devices", "cpu"], &["devices", "--json", "x"]] {
        let out = vfx(args, &[]);
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        assert!(stderr_line(&out).contains("for devices"), "{args:?}");
    }
    assert_eq!(vfx(&["devices", "--help"], &[]).status.code(), Some(0));
}

/// The listed OpenCL devices and whether each meets the requirements.
fn listed_gpus() -> Vec<(String, bool)> {
    let v: Value = serde_json::from_slice(&vfx(&["devices", "--json"], &[]).stdout).unwrap();
    v.as_array().unwrap().iter().skip(1).map(|d| (d["id"].as_str().unwrap().to_string(), d["verdict"] == "ok")).collect()
}

#[test]
fn gpu_n_is_strict_and_gpu_falls_back_with_one_note() {
    let input = clip(8, 3, 2, 3);
    // a number nobody lists: exit 3 with the reason, nothing written, under --check too
    for device in [&["--device", "gpu=4000"][..], &["--device=gpu=4000"]] {
        for check in [false, true] {
            let mut extra = device.to_vec();
            if check {
                extra.push("--check");
            }
            let out = vfx(&shift_args(&extra), &input);
            assert_eq!(out.status.code(), Some(3), "{extra:?}");
            assert!(stderr_line(&out).starts_with("vfx: error: unsupported: --device gpu=4000: "), "{extra:?}");
            assert!(out.stdout.is_empty(), "{extra:?}: no frame is written");
        }
    }
    // gpu: a device that passes the gate renders silently, else the CPU after ONE note; the bytes
    // are the CPU's either way
    let gpus = listed_gpus();
    for retime in [false, true] {
        let args: Vec<&str> = if retime {
            vec!["apply", "retime", "--width", "3", "--height", "2", "--depth", "8", "--fps", "24", "--param", "speed_percent=50", "--device", "gpu"]
        } else {
            shift_args(&["--device", "gpu"])
        };
        let out = vfx(&args, &input);
        assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
        let mut cpu = args.clone();
        cpu.truncate(cpu.len() - 2);
        let want = vfx(&cpu, &input);
        assert_eq!(out.stdout, want.stdout, "the CPU's bytes");
        let err = String::from_utf8_lossy(&out.stderr).to_string();
        if gpus.iter().any(|(_, ok)| *ok) {
            assert_eq!(err, String::from_utf8_lossy(&want.stderr), "a usable device: no note");
        } else {
            println!("no usable OpenCL device: {err}");
            assert_eq!(err.lines().count(), 1 + String::from_utf8_lossy(&want.stderr).lines().count(), "{err:?}");
            assert!(err.starts_with("vfx: note: --device gpu: ") && err.lines().next().unwrap().ends_with("rendering on the CPU"), "{err:?}");
        }
    }
}

#[test]
fn every_usable_gpu_renders_the_cpu_bytes_of_every_effect() {
    let usable: Vec<String> = listed_gpus().into_iter().filter(|(_, ok)| *ok).map(|(id, _)| id).collect();
    if usable.is_empty() {
        println!("no usable OpenCL device, skipped");
        return;
    }
    let (w, h) = (24usize, 16usize);
    let ws = w.to_string();
    let hs = h.to_string();
    let runs: [(&str, &[&str]); 5] = [
        ("retime", &["--param", "speed_percent=37.5"]),
        ("retime", &["--param", "speed_percent=50", "--param", "frame_interp=3", "--param", "smart_blend=1"]),
        ("retime-fgbg", &["--param", "speed_percent=60", "--param", "warping=2"]),
        ("create-vectors", &[]),
        ("create-vectors", &["--param", "vector_resolution=2", "--param", "track_frame=2"]),
    ];
    for depth in [8u32, 16, 32] {
        let d = depth.to_string();
        let input = clip(depth, w, h, 5);
        for (id, params) in runs {
            let mut base: Vec<&str> = vec!["apply", id, "--width", &ws, "--height", &hs, "--depth", &d, "--fps", "24"];
            base.extend_from_slice(params);
            let cpu = vfx(&base, &input);
            assert_eq!(cpu.status.code(), Some(0), "{id} {params:?}: {}", String::from_utf8_lossy(&cpu.stderr));
            for dev in &usable {
                let mut args = base.clone();
                args.extend_from_slice(&["--device", dev]);
                let out = vfx(&args, &input);
                assert_eq!(out.status.code(), Some(0), "{id} {dev}");
                assert_eq!(out.stderr, cpu.stderr, "{id} {params:?} @{depth} {dev}");
                assert!(out.stdout == cpu.stdout, "{id} {params:?} @{depth} {dev}: other bytes than the CPU");
            }
        }
    }
}

#[test]
fn bad_device_values_are_usage_errors() {
    for value in ["opencl", "gpu=", "gpu=-1", "gpu=+1", "gpu=1.0", "GPU", "cpu=0", ""] {
        let out = vfx(&shift_args(&["--device", value, "--check"]), &[]);
        assert_eq!(out.status.code(), Some(2), "{value:?}");
        assert!(stderr_line(&out).contains("`--device` expects cpu, gpu or gpu=N"), "{value:?}");
    }
    let out = vfx(&shift_args(&["--device"]), &[]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr_line(&out).contains("needs a value"));
}

#[test]
fn a_frame_effect_refuses_the_device_flag() {
    for value in ["cpu", "gpu", "gpu=0"] {
        let out = vfx(&["apply", FRAME, "--width", "2", "--height", "2", "--depth", "8", "--device", value], &[0; 16]);
        assert_eq!(out.status.code(), Some(2), "{value}");
        assert_eq!(stderr_line(&out), format!("vfx: error: {FRAME} is a frame effect, --device only apply to temporal effects"));
        assert!(out.stdout.is_empty());
    }
}

#[test]
fn the_cpu_device_gives_the_bytes_of_a_run_without_the_flag() {
    let listed: Value = serde_json::from_slice(&vfx(&["list", "--json", "--all"], &[]).stdout).unwrap();
    let mut runs: Vec<(String, Vec<String>)> = Vec::new();
    for e in listed.as_array().unwrap() {
        let required = e["layers"].as_array().is_some_and(|l| l.iter().any(|layer| layer["optional"] == false));
        if e["kind"] != "temporal" || required {
            continue;
        }
        runs.push((e["id"].as_str().unwrap().to_string(), Vec::new()));
    }
    // the vector path of retime too: half speed puts the frames between source frames
    runs.push(("retime".into(), vec!["--param".into(), "speed_percent=50".into()]));
    assert!(runs.iter().any(|(id, _)| id == "retime"), "retime is listed");
    for (id, params) in &runs {
        for depth in [8u32, 16, 32] {
            let input = clip(depth, 16, 12, 3);
            let d = depth.to_string();
            let mut base: Vec<&str> = vec!["apply", id, "--width", "16", "--height", "12", "--depth", &d, "--fps", "24"];
            base.extend(params.iter().map(String::as_str));
            let plain = vfx(&base, &input);
            for flag in [&["--device", "cpu"][..], &["--device=cpu"]] {
                let mut args = base.clone();
                args.extend_from_slice(flag);
                let out = vfx(&args, &input);
                assert_eq!(out.status.code(), plain.status.code(), "{id} @{depth} {flag:?}");
                assert_eq!(out.stderr, plain.stderr, "{id} @{depth} {flag:?}");
                assert_eq!(out.stdout, plain.stdout, "{id} @{depth} {flag:?}: other bytes on the cpu device");
            }
        }
    }
}

#[test]
fn the_help_names_the_device_surface() {
    let help = String::from_utf8(vfx(&["--help"], &[]).stdout).unwrap();
    for word in ["vfx devices [--json]", "--device cpu|gpu|gpu=N", "passes"] {
        assert!(help.contains(word), "the help should mention {word}");
    }
}
