//! `fx-parity gen`: render the input frames and clips with ffmpeg and write `cases.json`.

use crate::cases::{self, InputFile, Plan, TestSet};
use crate::effects::{EffectInfo, EffectKind, Kind};
use crate::sources;
use crate::vfx::Vfx;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

pub struct GenOptions {
    pub out: PathBuf,
    pub vfx: Option<PathBuf>,
    /// Effect ids to include; everything `vfx list` reports when None.
    pub effects: Option<Vec<String>>,
    pub plan: Plan,
    pub jobs: usize,
}

pub struct GenSummary {
    pub effects: usize,
    pub inputs: usize,
    pub cases: usize,
    pub manifest: PathBuf,
}

/// `FFMPEG` from the environment, else `ffmpeg` on PATH.
pub fn ffmpeg_path() -> PathBuf {
    std::env::var_os("FFMPEG").filter(|v| !v.is_empty()).map(PathBuf::from).unwrap_or_else(|| "ffmpeg".into())
}

/// First line of `ffmpeg -version`, which also proves the binary runs.
pub fn ffmpeg_version(ffmpeg: &Path) -> Result<String, String> {
    let out = Command::new(ffmpeg).arg("-version").stdin(Stdio::null()).output().map_err(|e| {
        format!("cannot run ffmpeg ({}): {e}; set FFMPEG or put it on PATH", ffmpeg.display())
    })?;
    if !out.status.success() {
        return Err(format!("`{} -version` failed", ffmpeg.display()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).lines().next().unwrap_or("ffmpeg").trim().to_string())
}

/// Renders one still into `dest` as raw frame bytes in the depth's wire format.
pub fn render_input(ffmpeg: &Path, source: &str, depth: u32, w: usize, h: usize, dest: &Path) -> Result<(), String> {
    let part = dest.with_extension(format!("{}.part", dest.extension().and_then(|e| e.to_str()).unwrap_or("raw")));
    let out = Command::new(ffmpeg)
        .args(["-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i"])
        .arg(sources::graph(source, depth, w, h))
        .args(["-frames:v", "1", "-an", "-pix_fmt", sources::pix_fmt(depth), "-f", "rawvideo"])
        .arg(&part)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("cannot run ffmpeg: {e}"))?;
    let describe = format!("{source} {w}x{h} at {depth} bpc");
    if !out.status.success() {
        let _ = std::fs::remove_file(&part);
        let msg = String::from_utf8_lossy(&out.stderr);
        let hint = if depth == 32 { " (32 bpc needs an ffmpeg whose geq filter accepts float formats)" } else { "" };
        return Err(format!("ffmpeg failed for {describe}: {}{hint}", msg.lines().next().unwrap_or("no message")));
    }
    let want = w * h * fx_core::Depth::from_bits(depth).expect("known depth").bytes_per_pixel();
    let got = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if got != want as u64 {
        let _ = std::fs::remove_file(&part);
        return Err(format!("ffmpeg wrote {got} bytes for {describe}, expected {want}"));
    }
    std::fs::rename(&part, dest).map_err(|e| format!("cannot move {} into place: {e}", part.display()))
}

/// Renders one clip of `frames` frames into `dest` as raw bytes in the depth's wire format.
pub fn render_clip(ffmpeg: &Path, source: &str, depth: u32, w: usize, h: usize, frames: usize, dest: &Path) -> Result<(), String> {
    let part = dest.with_extension(format!("{}.part", dest.extension().and_then(|e| e.to_str()).unwrap_or("raw")));
    let out = Command::new(ffmpeg)
        .args(["-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i"])
        .arg(sources::clip_graph(source, depth, w, h, frames))
        .args(["-frames:v", &frames.to_string(), "-an", "-fps_mode", "passthrough", "-pix_fmt", sources::pix_fmt(depth), "-f", "rawvideo"])
        .arg(&part)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("cannot run ffmpeg: {e}"))?;
    let describe = format!("{source} {w}x{h} x {frames} frames at {depth} bpc");
    if !out.status.success() {
        let _ = std::fs::remove_file(&part);
        let msg = String::from_utf8_lossy(&out.stderr);
        return Err(format!("ffmpeg failed for {describe}: {}", msg.lines().next().unwrap_or("no message")));
    }
    let want = frames * w * h * fx_core::Depth::from_bits(depth).expect("known depth").bytes_per_pixel();
    let got = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if got != want as u64 {
        let _ = std::fs::remove_file(&part);
        return Err(format!("ffmpeg wrote {got} bytes for {describe}, expected {want}"));
    }
    std::fs::rename(&part, dest).map_err(|e| format!("cannot move {} into place: {e}", part.display()))
}

/// Renders a still or a clip, whichever the source is.
fn render_file(ffmpeg: &Path, f: &InputFile, dest: &Path) -> Result<(), String> {
    if sources::find_clip(&f.source).is_some() {
        render_clip(ffmpeg, &f.source, f.depth, f.width, f.height, f.frames, dest)
    } else {
        render_input(ffmpeg, &f.source, f.depth, f.width, f.height, dest)
    }
}

/// Runs `work` for every index on `jobs` threads and returns the first error.
fn parallel(count: usize, jobs: usize, work: impl Fn(usize) -> Result<(), String> + Sync) -> Result<(), String> {
    let next = AtomicUsize::new(0);
    let first_error: Mutex<Option<String>> = Mutex::new(None);
    std::thread::scope(|scope| {
        for _ in 0..jobs.max(1).min(count.max(1)) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= count || first_error.lock().unwrap().is_some() {
                    break;
                }
                if let Err(e) = work(i) {
                    first_error.lock().unwrap().get_or_insert(e);
                }
            });
        }
    });
    match first_error.into_inner().unwrap() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Keeps the requested effects, in `vfx list` order, and rejects ids it does not have.
/// Without a request every effect `vfx list` shows by default is kept; the hidden
/// self-test effects take part only when named.
pub fn select_effects(all: Vec<EffectInfo>, wanted: Option<&[String]>) -> Result<Vec<EffectInfo>, String> {
    let Some(wanted) = wanted else {
        let shown: Vec<EffectInfo> = all.into_iter().filter(|e| !e.hidden).collect();
        return if shown.is_empty() { Err("`vfx list` reported no effects".into()) } else { Ok(shown) };
    };
    if let Some(missing) = wanted.iter().find(|w| !all.iter().any(|e| &e.id == *w)) {
        let known: Vec<&str> = all.iter().map(|e| e.id.as_str()).collect();
        return Err(format!("vfx has no effect `{missing}`, it has: {}", known.join(", ")));
    }
    Ok(all.into_iter().filter(|e| wanted.contains(&e.id)).collect())
}

pub fn generate(opts: &GenOptions, log: &(dyn Fn(&str) + Sync)) -> Result<GenSummary, String> {
    let vfx = Vfx::locate(opts.vfx.as_deref())?;
    let ffmpeg = ffmpeg_path();
    let version = ffmpeg_version(&ffmpeg)?;
    let effects = select_effects(vfx.list_all()?, opts.effects.as_deref())?;
    for e in &effects {
        for p in &e.params {
            if let Kind::Unknown(kind) = &p.kind {
                log(&format!("warning: {}.{} has the param kind `{kind}`, which gen does not know; it stays at its default", e.id, p.id));
            }
        }
    }

    let built = cases::build_all(&effects, &opts.plan);
    // stills only for the frame effects, and only the clips the kept temporal cases use
    let mut inputs = if effects.iter().any(|e| e.kind == EffectKind::Frame) { opts.plan.inputs() } else { Vec::new() };
    inputs.extend(built.clip_inputs.iter().cloned());
    if inputs.is_empty() {
        return Err("the sources, sizes and depths given leave nothing to render (the range source exists only at 32 bpc)".into());
    }
    let dir = opts.out.join("inputs");
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let clips = inputs.iter().filter(|f| f.frames > 1 || sources::find_clip(&f.source).is_some()).count();
    log(&format!("rendering {} inputs ({} clips) with {version}", inputs.len(), clips));
    parallel(inputs.len(), opts.jobs, |i| {
        let f = &inputs[i];
        render_file(&ffmpeg, f, &opts.out.join(&f.file))
    })?;

    let set = TestSet { format: cases::FORMAT, ffmpeg: version, inputs: inputs.clone(), cases: built.cases };
    let manifest = opts.out.join("cases.json");
    let mut text = serde_json::to_string_pretty(&set).map_err(|e| format!("cannot encode cases.json: {e}"))?;
    text.push('\n');
    std::fs::write(&manifest, text).map_err(|e| format!("cannot write {}: {e}", manifest.display()))?;
    Ok(GenSummary { effects: effects.len(), inputs: inputs.len(), cases: set.cases.len(), manifest })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn effect(id: &str) -> EffectInfo {
        EffectInfo::new(id, vec![])
    }

    #[test]
    fn effect_selection_keeps_list_order_and_rejects_strangers() {
        let all = || vec![effect("a"), effect("b"), effect("c")];
        let ids = |v: Vec<EffectInfo>| v.into_iter().map(|e| e.id).collect::<Vec<_>>();
        assert_eq!(ids(select_effects(all(), None).unwrap()), ["a", "b", "c"]);
        assert_eq!(ids(select_effects(all(), Some(&["c".to_string(), "a".to_string()])).unwrap()), ["a", "c"]);
        let err = select_effects(all(), Some(&["z".to_string()])).unwrap_err();
        assert!(err.contains("`z`") && err.contains("a, b, c"), "{err}");
        assert!(select_effects(vec![], None).is_err());
    }

    #[test]
    fn hidden_effects_join_only_when_named() {
        let all = || {
            let mut hidden = effect("selftest-x");
            hidden.hidden = true;
            vec![effect("a"), hidden, effect("b")]
        };
        let ids = |v: Vec<EffectInfo>| v.into_iter().map(|e| e.id).collect::<Vec<_>>();
        assert_eq!(ids(select_effects(all(), None).unwrap()), ["a", "b"]);
        assert_eq!(ids(select_effects(all(), Some(&["selftest-x".to_string()])).unwrap()), ["selftest-x"]);
        assert_eq!(ids(select_effects(all(), Some(&["b".to_string(), "selftest-x".to_string()])).unwrap()), ["selftest-x", "b"]);
        let mut only_hidden = effect("h");
        only_hidden.hidden = true;
        assert!(select_effects(vec![only_hidden], None).is_err());
    }
}
