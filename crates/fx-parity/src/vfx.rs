//! Finding and driving the `vfx` binary. A still goes through its stdin and
//! stdout, the same way ffmpeg pipes feed it; a clip is read from files, like the
//! temporal effects want it.

use crate::cases::Case;
use crate::effects::{self, EffectInfo};
use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;

static SPAWN: Mutex<()> = Mutex::new(());

#[derive(Clone, Debug)]
pub struct Vfx {
    pub path: PathBuf,
}

/// The rayon thread count a run of `vfx` gets through `RAYON_NUM_THREADS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Threads {
    /// Whatever the environment holds.
    Inherit,
    /// The variable removed: rayon picks its own default (one thread per core).
    Default,
    Fixed(usize),
}

fn exe_name() -> &'static str {
    if cfg!(windows) {
        "vfx.exe"
    } else {
        "vfx"
    }
}

fn existing(path: &Path) -> Result<Vfx, String> {
    if path.is_file() {
        Ok(Vfx { path: path.to_path_buf() })
    } else {
        Err(format!("vfx not found at {}", path.display()))
    }
}

/// The `vfx apply` words common to stills and clips: effect, size, depth and params.
fn base_args(case: &Case) -> Vec<OsString> {
    let mut args: Vec<OsString> = ["apply", &case.effect, "--width", &case.width.to_string(), "--height", &case.height.to_string(), "--depth", &case.depth.to_string()]
        .iter()
        .map(OsString::from)
        .collect();
    for p in &case.params {
        args.push("--param".into());
        args.push(p.into());
    }
    args
}

/// The full `vfx apply` argument list of a temporal case. `testset` locates the clip
/// files; without it the paths stay relative (for `--check`, which reads none).
pub fn clip_args(case: &Case, testset: &Path) -> Vec<OsString> {
    let mut args = base_args(case);
    if let Some(fps) = &case.fps {
        args.extend(["--fps".into(), fps.into()]);
    }
    args.extend(["--frames".into(), case.frames.to_string().into()]);
    for (id, file) in &case.layers {
        args.push("--layer".into());
        args.push(format!("{id}={}", testset.join(file).display()).into());
    }
    args.extend(["--in".into(), testset.join(&case.input).into_os_string()]);
    args
}

impl Vfx {
    pub fn at(path: impl Into<PathBuf>) -> Vfx {
        Vfx { path: path.into() }
    }

    /// `--vfx`, then `VFX_PATH`, then next to this executable, then PATH.
    pub fn locate(explicit: Option<&Path>) -> Result<Vfx, String> {
        if let Some(p) = explicit {
            return existing(p);
        }
        if let Some(p) = std::env::var_os("VFX_PATH").filter(|p| !p.is_empty()) {
            return existing(Path::new(&p));
        }
        if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
            // a test binary sits one level down, in target/<profile>/deps
            for d in [Some(dir.as_path()), dir.parent()].into_iter().flatten() {
                if let Ok(v) = existing(&d.join(exe_name())) {
                    return Ok(v);
                }
            }
        }
        if let Some(paths) = std::env::var_os("PATH") {
            for d in std::env::split_paths(&paths) {
                if let Ok(v) = existing(&d.join(exe_name())) {
                    return Ok(v);
                }
            }
        }
        Err("vfx not found: pass --vfx PATH, set VFX_PATH, or build it with `cargo build --release -p fx-cli`".into())
    }

    fn list_with(&self, extra: &[&str]) -> Result<Vec<EffectInfo>, String> {
        let out = Command::new(&self.path)
            .args(["list", "--json"])
            .args(extra)
            .stdin(Stdio::null())
            .output()
            .map_err(|e| format!("cannot run {}: {e}", self.path.display()))?;
        if !out.status.success() {
            return Err(format!("`vfx list --json` failed: {}", failure(&out.status.code(), &out.stderr)));
        }
        effects::parse_list(&String::from_utf8_lossy(&out.stdout))
    }

    /// The effects `vfx list` shows by default.
    pub fn list(&self) -> Result<Vec<EffectInfo>, String> {
        self.list_with(&[])
    }

    /// Every effect, the hidden self-test ones included (`vfx list --all`). A vfx that does
    /// not know `--all` has no hidden effects, so its plain list is the answer.
    pub fn list_all(&self) -> Result<Vec<EffectInfo>, String> {
        self.list_with(&["--all"]).or_else(|_| self.list())
    }

    /// Runs one still case on `input` and returns the output frame bytes.
    pub fn apply(&self, case: &Case, input: &[u8]) -> Result<Vec<u8>, String> {
        if case.is_temporal() {
            return Err(format!("case {} is temporal, it runs from files with apply_clip", case.id));
        }
        let mut cmd = Command::new(&self.path);
        cmd.args(base_args(case));
        // Windows lets a child inherit every inheritable handle in the process, so a spawn
        // racing another thread's would hold that thread's pipe open and delay its end of input
        let mut child = {
            let _one_at_a_time = SPAWN.lock().unwrap_or_else(|p| p.into_inner());
            cmd.stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| format!("cannot run {}: {e}", self.path.display()))?
        };
        let mut stdin = child.stdin.take().expect("piped stdin");
        let out = std::thread::scope(|scope| {
            // a separate writer keeps a large frame from deadlocking against the child's output
            scope.spawn(move || {
                let _ = stdin.write_all(input);
            });
            child.wait_with_output()
        })
        .map_err(|e| format!("waiting for vfx failed: {e}"))?;
        if out.status.success() {
            Ok(out.stdout)
        } else {
            Err(format!("vfx {}", failure(&out.status.code(), &out.stderr)))
        }
    }

    /// Runs one temporal case from the clip files of `testset` and returns the output clip.
    pub fn apply_clip(&self, case: &Case, testset: &Path, threads: Threads) -> Result<Vec<u8>, String> {
        let mut cmd = Command::new(&self.path);
        cmd.args(clip_args(case, testset));
        match threads {
            Threads::Inherit => {}
            Threads::Default => {
                cmd.env_remove("RAYON_NUM_THREADS");
            }
            Threads::Fixed(n) => {
                cmd.env("RAYON_NUM_THREADS", n.to_string());
            }
        }
        let out = {
            let _one_at_a_time = SPAWN.lock().unwrap_or_else(|p| p.into_inner());
            cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()
        }
        .map_err(|e| format!("cannot run {}: {e}", self.path.display()))?
        .wait_with_output()
        .map_err(|e| format!("waiting for vfx failed: {e}"))?;
        if out.status.success() {
            Ok(out.stdout)
        } else {
            Err(format!("vfx {}", failure(&out.status.code(), &out.stderr)))
        }
    }

    /// Asks vfx to validate a case, effect id, params, keyframes, layers and rate, without
    /// reading a file (`vfx apply --check`).
    pub fn check(&self, case: &Case) -> Result<(), String> {
        let mut args = if case.is_temporal() { clip_args(case, Path::new("")) } else { base_args(case) };
        args.push("--check".into());
        let out = Command::new(&self.path)
            .args(args)
            .stdin(Stdio::null())
            .output()
            .map_err(|e| format!("cannot run {}: {e}", self.path.display()))?;
        if out.status.success() {
            Ok(())
        } else {
            Err(format!("vfx {}", failure(&out.status.code(), &out.stderr)))
        }
    }
}

/// "exited with code 3: <first stderr line>"
fn failure(code: &Option<i32>, stderr: &[u8]) -> String {
    let line = String::from_utf8_lossy(stderr).lines().next().unwrap_or("").trim().to_string();
    let how = match code {
        Some(c) => format!("exited with code {c}"),
        None => "was killed".to_string(),
    };
    if line.is_empty() {
        how
    } else {
        format!("{how}: {line}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cases::CaseKind;
    use std::collections::BTreeMap;

    fn temporal_case() -> Case {
        Case {
            id: "fx.all-min.testsrc2-16x12.d8.deadbeef".into(),
            effect: "fx".into(),
            label: "all-min".into(),
            params: vec!["gain=0@0;10@11".into(), "offset=2".into()],
            depth: 8,
            width: 16,
            height: 12,
            source: "testsrc2".into(),
            input: "inputs/testsrc2-16x12.d8.f12.rgba".into(),
            reference: "r".into(),
            kind: CaseKind::Temporal,
            frames: 12,
            fps: Some("30000/1001".into()),
            layers: BTreeMap::from([("overlay".to_string(), "inputs/noisefx-16x12.d8.f12.rgba".to_string())]),
        }
    }

    fn text(args: &[OsString]) -> Vec<String> {
        args.iter().map(|a| a.to_string_lossy().into_owned()).collect()
    }

    #[test]
    fn failure_text_keeps_the_first_stderr_line() {
        assert_eq!(failure(&Some(3), b"vfx: error: nope\nmore\n"), "exited with code 3: vfx: error: nope");
        assert_eq!(failure(&Some(4), b""), "exited with code 4");
        assert_eq!(failure(&None, b"x"), "was killed: x");
    }

    #[test]
    fn explicit_paths_must_exist() {
        assert!(Vfx::locate(Some(Path::new("definitely/not/here"))).is_err());
    }

    #[test]
    fn a_clip_case_becomes_files_rate_frames_layers_and_params() {
        let args = text(&clip_args(&temporal_case(), Path::new("set")));
        let sep = std::path::MAIN_SEPARATOR;
        let mut want: Vec<String> = [
            "apply", "fx", "--width", "16", "--height", "12", "--depth", "8", "--param", "gain=0@0;10@11", "--param", "offset=2",
            "--fps", "30000/1001", "--frames", "12", "--layer",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        want.push(format!("overlay=set{sep}inputs/noisefx-16x12.d8.f12.rgba"));
        want.push("--in".to_string());
        want.push(format!("set{sep}inputs/testsrc2-16x12.d8.f12.rgba"));
        assert_eq!(args, want);
    }

    #[test]
    fn a_still_cannot_run_as_a_clip_and_a_clip_cannot_run_as_a_still() {
        let vfx = Vfx::at("not-run");
        let err = vfx.apply(&temporal_case(), &[]).unwrap_err();
        assert!(err.contains("apply_clip"), "{err}");
    }
}
