//! `fx-parity devcheck`: the byte gate of the GPU backend. Every temporal case of one or more test
//! sets runs through `vfx` with `--device cpu` and with `--device gpu=N` for each device, and the
//! two runs must agree byte for byte: the output clip, the exit code and stderr (minus the device's
//! own note lines: the device report this command asks for with `VFX_DEVICE_REPORT=1`, a
//! refusal note, the `gpu` fallback note). A device run also reports how many stages fell back to
//! the CPU, which is counted but is not a difference (a fallback gives the CPU's bytes). No
//! references are needed.

use crate::cases::Case;
use crate::run::load_testset;
use crate::selfcheck::{describe_difference, pick};
use crate::vfx::{clip_args, Vfx};
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub struct DevcheckOptions {
    pub testsets: Vec<PathBuf>,
    pub effect: Option<String>,
    /// How many cases per test set; 0 runs every temporal case.
    pub max_cases: usize,
    /// The `--device` values to check; `None` takes every listed device whose verdict is `ok`.
    pub devices: Option<Vec<String>>,
    pub vfx: Option<PathBuf>,
}

/// What one `vfx` run gave.
#[derive(Clone, Debug, PartialEq)]
pub struct RunOutput {
    pub code: Option<i32>,
    pub stdout: Vec<u8>,
    /// The stderr lines that are not the device's own notes.
    pub stderr: Vec<String>,
    /// Stages that fell back to the CPU, from the device report (0 without one).
    pub fallbacks: u64,
    /// The device's own note lines (the report, a refusal), as printed.
    pub device_notes: Vec<String>,
}

/// True for a note line that only a device run prints.
pub fn is_device_note(line: &str) -> bool {
    let l = line.trim();
    ["vfx: note: device report: ", "vfx: note: gpu=", "vfx: note: --device "].iter().any(|p| l.starts_with(p))
}

/// The sum of the `fallback N` counts of a device report line.
pub fn report_fallbacks(line: &str) -> u64 {
    let mut total = 0;
    let mut words = line.split_whitespace().peekable();
    while let Some(w) = words.next() {
        if w == "fallback" {
            if let Some(n) = words.peek().and_then(|n| n.parse::<u64>().ok()) {
                total += n;
            }
        }
    }
    total
}

/// Splits a run's stderr into the compared lines and the device notes.
pub fn split_stderr(stderr: &[u8]) -> (Vec<String>, Vec<String>, u64) {
    let text = String::from_utf8_lossy(stderr);
    let (mut kept, mut notes, mut fallbacks) = (Vec::new(), Vec::new(), 0);
    for line in text.lines() {
        if is_device_note(line) {
            if line.trim().starts_with("vfx: note: device report: ") {
                fallbacks += report_fallbacks(line);
            }
            notes.push(line.trim().to_string());
        } else {
            kept.push(line.to_string());
        }
    }
    (kept, notes, fallbacks)
}

fn run_case(vfx: &Vfx, case: &Case, testset: &Path, device: &str) -> Result<RunOutput, String> {
    let mut cmd = Command::new(&vfx.path);
    cmd.args(clip_args(case, testset)).args(["--device", device]).env("VFX_DEVICE_REPORT", "1");
    let out = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("cannot run {}: {e}", vfx.path.display()))?;
    let (stderr, device_notes, fallbacks) = split_stderr(&out.stderr);
    Ok(RunOutput { code: out.status.code(), stdout: out.stdout, stderr, fallbacks, device_notes })
}

/// How one case compared on one device.
#[derive(Clone, Debug)]
pub struct CaseResult {
    pub case: String,
    pub device: String,
    /// `None`: identical; else what differs.
    pub difference: Option<String>,
    pub fallbacks: u64,
    pub refused: bool,
}

/// Compares a device run with the CPU run of the same case.
pub fn compare(case: &Case, device: &str, cpu: &RunOutput, dev: &RunOutput) -> CaseResult {
    let refused = dev.device_notes.iter().any(|l| !l.starts_with("vfx: note: device report: "));
    let difference = if cpu.code != dev.code {
        Some(format!("exit code {:?} on the CPU, {:?} on {device}", cpu.code, dev.code))
    } else if cpu.stderr != dev.stderr {
        Some(format!("stderr differs: {:?} on the CPU, {:?} on {device}", cpu.stderr, dev.stderr))
    } else {
        describe_difference(case, device, &cpu.stdout, &dev.stdout).map(|d| d.detail)
    };
    CaseResult { case: case.id.clone(), device: device.to_string(), difference, fallbacks: dev.fallbacks, refused }
}

/// The tally of one device.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    pub cases: usize,
    pub identical: usize,
    pub different: usize,
    /// Cases with at least one stage fallback, and the stage fallbacks in all.
    pub cases_with_fallbacks: usize,
    pub fallbacks: u64,
    /// Runs where the device refused itself (the cross check).
    pub refusals: usize,
}

pub fn tally(results: &[CaseResult], device: &str) -> Tally {
    let mut t = Tally::default();
    for r in results.iter().filter(|r| r.device == device) {
        t.cases += 1;
        if r.difference.is_none() {
            t.identical += 1;
        } else {
            t.different += 1;
        }
        if r.fallbacks > 0 {
            t.cases_with_fallbacks += 1;
        }
        t.fallbacks += r.fallbacks;
        t.refusals += usize::from(r.refused);
    }
    t
}

/// The devices `vfx devices --json` lists with the verdict `ok`.
pub fn listed_devices(vfx: &Vfx) -> Result<Vec<String>, String> {
    let out = Command::new(&vfx.path).args(["devices", "--json"]).output().map_err(|e| format!("cannot run {}: {e}", vfx.path.display()))?;
    let v: Value = serde_json::from_slice(&out.stdout).map_err(|e| format!("vfx devices --json: {e}"))?;
    Ok(v.as_array()
        .map(|a| a.iter().filter(|d| d["kind"] == "gpu" && d["verdict"] == "ok").filter_map(|d| d["id"].as_str().map(str::to_string)).collect())
        .unwrap_or_default())
}

/// Runs the comparison; `progress` gets one line per case.
pub fn run(opts: &DevcheckOptions, progress: &dyn Fn(&str)) -> Result<(Vec<String>, Vec<CaseResult>), String> {
    let vfx = Vfx::locate(opts.vfx.as_deref())?;
    let devices = match &opts.devices {
        Some(d) => d.clone(),
        None => listed_devices(&vfx)?,
    };
    let mut results = Vec::new();
    for testset in &opts.testsets {
        let set = load_testset(testset)?;
        let picked = pick(&set.cases, opts.effect.as_deref(), opts.max_cases);
        for (i, case) in picked.iter().enumerate() {
            let cpu = run_case(&vfx, case, testset, "cpu")?;
            for device in &devices {
                // one device process at a time: the runs never share a GPU
                let dev = run_case(&vfx, case, testset, device)?;
                let r = compare(case, device, &cpu, &dev);
                progress(&format!(
                    "{} {}/{} {} {device}: {}{}",
                    testset.display(),
                    i + 1,
                    picked.len(),
                    case.id,
                    r.difference.as_deref().unwrap_or("identical"),
                    if r.fallbacks > 0 { format!(" ({} fallbacks)", r.fallbacks) } else { String::new() }
                ));
                results.push(r);
            }
        }
    }
    Ok((devices, results))
}

pub fn all_identical(results: &[CaseResult]) -> bool {
    results.iter().all(|r| r.difference.is_none())
}

pub fn write_human(devices: &[String], results: &[CaseResult], out: &mut impl Write) -> std::io::Result<()> {
    if devices.is_empty() {
        writeln!(out, "no usable OpenCL device: nothing to compare")?;
    }
    writeln!(out, "device  cases  identical  different  cases with fallbacks  fallbacks  refusals")?;
    for d in devices {
        let t = tally(results, d);
        writeln!(out, "{d:<7} {:>5}  {:>9}  {:>9}  {:>20}  {:>9}  {:>8}", t.cases, t.identical, t.different, t.cases_with_fallbacks, t.fallbacks, t.refusals)?;
    }
    for r in results.iter().filter(|r| r.difference.is_some()) {
        writeln!(out, "DIFFERENT {} {}: {}", r.device, r.case, r.difference.as_deref().unwrap_or(""))?;
    }
    Ok(())
}

pub fn to_json(devices: &[String], results: &[CaseResult]) -> Value {
    let per: Vec<Value> = devices
        .iter()
        .map(|d| {
            let t = tally(results, d);
            json!({"device": d, "cases": t.cases, "identical": t.identical, "different": t.different,
                   "cases_with_fallbacks": t.cases_with_fallbacks, "fallbacks": t.fallbacks, "refusals": t.refusals})
        })
        .collect();
    let diffs: Vec<Value> = results
        .iter()
        .filter(|r| r.difference.is_some())
        .map(|r| json!({"device": r.device, "case": r.case, "difference": r.difference}))
        .collect();
    json!({"devices": per, "differences": diffs})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn out(code: i32, stdout: &[u8], stderr: &str) -> RunOutput {
        let (stderr, device_notes, fallbacks) = split_stderr(stderr.as_bytes());
        RunOutput { code: Some(code), stdout: stdout.to_vec(), stderr, fallbacks, device_notes }
    }

    fn case() -> Case {
        serde_json::from_value(json!({
            "id": "retime-x", "effect": "retime", "label": "x", "kind": "temporal", "width": 2, "height": 1, "depth": 8,
            "params": [], "source": "s", "input": "in.raw", "reference": "r.raw", "frames": 2, "fps": "24"
        }))
        .unwrap_or_else(|e| panic!("case json: {e}"))
    }

    #[test]
    fn device_notes_are_split_off_and_their_fallbacks_counted() {
        let (kept, notes, fb) = split_stderr(
            b"vfx: note: retime: G6 wrap: a stated rule\nvfx: note: device report: gpu=0 (X): solve device 3 (1 ms) fallback 2 [cl 0 nan 2 unsupported 0] (0 ms) cpu 0 (0 ms); solve_pair device 1 (1 ms) fallback 1 [cl 1 nan 0 unsupported 0] (0 ms) cpu 0 (0 ms); cache hits 0 misses 0 evictions 0 bytes 0\n",
        );
        assert_eq!(kept, vec!["vfx: note: retime: G6 wrap: a stated rule".to_string()]);
        assert_eq!(notes.len(), 1);
        assert_eq!(fb, 3);
        assert!(is_device_note("vfx: note: gpu=1 (Y): the first frame rendered with the device differs"));
        assert!(is_device_note("vfx: note: --device gpu: no OpenCL device passes the gate (...), rendering on the CPU"));
        assert!(!is_device_note("vfx: error: unsupported: --device gpu=9: no OpenCL device 9"));
    }

    #[test]
    fn identical_runs_compare_equal_and_each_difference_is_named() {
        let c = case();
        let cpu = out(0, &[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8], "vfx: note: retime: G6 x\n");
        let same = out(0, &[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8], "vfx: note: retime: G6 x\nvfx: note: device report: gpu=0 (X): solve device 1 (1 ms) fallback 0 [cl 0 nan 0 unsupported 0] (0 ms) cpu 0 (0 ms)\n");
        let r = compare(&c, "gpu=0", &cpu, &same);
        assert_eq!((r.difference, r.fallbacks, r.refused), (None, 0, false));
        let bytes = out(0, &[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 9], "vfx: note: retime: G6 x\n");
        assert!(compare(&c, "gpu=0", &cpu, &bytes).difference.unwrap().contains("1 of 2 frames differ"));
        let code = out(3, &[], "vfx: note: retime: G6 x\n");
        assert!(compare(&c, "gpu=0", &cpu, &code).difference.unwrap().starts_with("exit code"));
        let err = out(0, &cpu.stdout, "vfx: note: retime: G7 y\n");
        assert!(compare(&c, "gpu=0", &cpu, &err).difference.unwrap().starts_with("stderr differs"));
        let refused = out(0, &cpu.stdout, "vfx: note: retime: G6 x\nvfx: note: gpu=0 (X): the first frame rendered with the device differs from the CPU render, the CPU renders the rest of the run\n");
        let r = compare(&c, "gpu=0", &cpu, &refused);
        assert!(r.difference.is_none() && r.refused, "a refusal keeps the CPU's bytes and is counted");
        let t = tally(&[r, compare(&c, "gpu=0", &cpu, &bytes), compare(&c, "gpu=1", &cpu, &same)], "gpu=0");
        assert_eq!(t, Tally { cases: 2, identical: 1, different: 1, cases_with_fallbacks: 0, fallbacks: 0, refusals: 1 });
    }
}
