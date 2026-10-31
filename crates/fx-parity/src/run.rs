//! `fx-parity run`: apply every case through `vfx` and compare with its reference.

use crate::cases::{Case, TestSet};
use crate::compare::{self, Comparison};
use crate::families;
use crate::vfx::{self, Threads, Vfx};
use fx_core::{Depth, Frame};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

pub struct RunOptions {
    pub testset: PathBuf,
    pub refs: PathBuf,
    pub effect: Option<String>,
    pub vfx: Option<PathBuf>,
    pub jobs: usize,
    /// The accepted distance in representable floats at 32 bpc; 0 is bit for bit.
    pub max_ulp: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Pass,
    Fail,
    NoReference,
}

impl Status {
    pub fn word(self) -> &'static str {
        match self {
            Status::Pass => "pass",
            Status::Fail => "FAIL",
            Status::NoReference => "no reference",
        }
    }
}

#[derive(Debug)]
pub struct CaseResult {
    pub case: Case,
    pub status: Status,
    /// Set when the case could not be compared at all (vfx failed, a file had the wrong size).
    pub error: Option<String>,
    /// A still: what differs. A clip: what differs in its first failing frame, or for a pass
    /// the largest gaps over the whole clip (no pixels listed).
    pub comparison: Option<Comparison>,
    /// Clips: the first frame that fails, counting from 0.
    pub frame: Option<usize>,
    /// Clips: how many frames fail.
    pub failing_frames: usize,
    /// Pixels outside the accepted tolerance over the whole case (a clip's frames added up).
    pub differing_pixels: usize,
    /// Mean absolute error as a fraction of full scale: the frame's for a still, the clip's
    /// (the mean of its frames' means) for a clip. None when nothing was compared: no
    /// reference, or an error.
    pub mean_abs_diff: Option<f64>,
    /// Clips: the frame with the largest mean error and that error; None for a still and
    /// for a clip without any error.
    pub worst_frame: Option<(usize, f64)>,
    pub testset: PathBuf,
    pub input_path: PathBuf,
    pub reference_path: PathBuf,
    /// The `vfx: note:` lines of the run; empty when vfx did not run or printed none.
    pub notes: Vec<String>,
}

/// One word of a command line, quoted when a shell would split or act on it.
pub fn quote(word: &str) -> String {
    if word.contains(|c: char| c.is_whitespace() || ";&|<>()$`'\"".contains(c)) {
        format!("\"{word}\"")
    } else {
        word.to_string()
    }
}

impl CaseResult {
    /// Compared, and without any error: the output equals the reference.
    pub fn is_exact(&self) -> bool {
        self.mean_abs_diff == Some(0.0)
    }

    /// The command that reproduces this case's output, for handing to an effect author.
    pub fn repro(&self) -> String {
        let c = &self.case;
        if c.is_temporal() {
            let words: Vec<String> = vfx::clip_args(c, &self.testset).iter().map(|a| quote(&a.to_string_lossy())).collect();
            return format!("vfx {}", words.join(" "));
        }
        let mut cmd = format!("vfx apply {} --width {} --height {} --depth {}", c.effect, c.width, c.height, c.depth);
        for p in &c.params {
            cmd.push_str(&format!(" --param {}", quote(p)));
        }
        cmd.push_str(&format!(" --in {}", quote(&self.input_path.display().to_string())));
        cmd
    }
}

#[derive(Debug, Default, PartialEq)]
pub struct Totals {
    pub pass: usize,
    pub fail: usize,
    pub no_reference: usize,
}

pub fn totals(results: &[CaseResult]) -> Totals {
    let mut t = Totals::default();
    for r in results {
        match r.status {
            Status::Pass => t.pass += 1,
            Status::Fail => t.fail += 1,
            Status::NoReference => t.no_reference += 1,
        }
    }
    t
}

pub fn load_testset(dir: &Path) -> Result<TestSet, String> {
    let path = dir.join("cases.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e} (run `fx-parity gen --out {}` first)", path.display(), dir.display()))?;
    let set: TestSet = serde_json::from_str(&text).map_err(|e| format!("{} is not a valid test set: {e}", path.display()))?;
    if set.format != crate::cases::FORMAT {
        return Err(format!("{} has format {}, this build reads format {}", path.display(), set.format, crate::cases::FORMAT));
    }
    Ok(set)
}

/// A frame file as raw bytes plus the [`Frame`] read from them.
fn read_frame(path: &Path, case: &Case, what: &str) -> Result<(Vec<u8>, Frame), String> {
    let depth = Depth::from_bits(case.depth).ok_or_else(|| format!("case {} has depth {}", case.id, case.depth))?;
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {what} {}: {e}", path.display()))?;
    let want = case.width * case.height * depth.bytes_per_pixel();
    if bytes.len() != want {
        return Err(format!(
            "{what} {} has {} bytes, a {}x{} frame at {} bpc is {want}",
            path.display(),
            bytes.len(),
            case.width,
            case.height,
            case.depth
        ));
    }
    let frame = Frame::from_bytes(case.width, case.height, depth, &bytes);
    Ok((bytes, frame))
}

/// A clip file as raw bytes, which must hold exactly the case's frames.
fn read_clip(path: &Path, case: &Case, what: &str) -> Result<Vec<u8>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {what} {}: {e}", path.display()))?;
    let want = case.frames * case.frame_len();
    if bytes.len() != want {
        return Err(format!(
            "{what} {} has {} bytes, {} frames of {}x{} at {} bpc are {want}",
            path.display(),
            bytes.len(),
            case.frames,
            case.width,
            case.height,
            case.depth
        ));
    }
    Ok(bytes)
}

/// Runs one case with the default window of 1 ULP at 32 bpc.
pub fn run_case(vfx: &Vfx, testset: &Path, refs: &Path, case: &Case) -> CaseResult {
    run_case_within(vfx, testset, refs, case, compare::F32_MAX_ULP)
}

/// Runs one case. `vfx` is driven as a subprocess, exactly like a real pipe.
pub fn run_case_within(vfx: &Vfx, testset: &Path, refs: &Path, case: &Case, max_ulp: u64) -> CaseResult {
    let input_path = testset.join(&case.input);
    let reference_path = refs.join(&case.reference);
    let result = |status, error: Option<String>, comparison: Option<Comparison>| CaseResult {
        case: case.clone(),
        status,
        error,
        mean_abs_diff: comparison.as_ref().map(|c| c.mean_abs_diff),
        comparison,
        frame: None,
        failing_frames: 0,
        differing_pixels: 0,
        worst_frame: None,
        testset: testset.to_path_buf(),
        input_path: input_path.clone(),
        reference_path: reference_path.clone(),
        notes: Vec::new(),
    };
    let failed = |error: String| result(Status::Fail, Some(error), None);

    if !reference_path.is_file() {
        return result(Status::NoReference, None, None);
    }
    if case.is_temporal() {
        return run_clip(vfx, testset, case, max_ulp, &input_path, &reference_path, result, failed);
    }
    let (input_bytes, input) = match read_frame(&input_path, case, "input") {
        Ok(f) => f,
        Err(e) => return failed(e),
    };
    let expected = match read_frame(&reference_path, case, "reference") {
        Ok((_, f)) => f,
        Err(e) => return failed(e),
    };
    let output = match vfx.apply(case, &input_bytes) {
        Ok(o) => o,
        Err(e) => return failed(e),
    };
    if output.len() != input_bytes.len() {
        return failed(format!("vfx wrote {} bytes, a frame is {}", output.len(), input_bytes.len()));
    }
    let actual = Frame::from_bytes(case.width, case.height, input.depth(), &output);
    match compare::compare_within(&input, &expected, &actual, max_ulp) {
        Ok(c) => {
            let differing = c.differing_pixels;
            let mut r = result(if c.is_match() { Status::Pass } else { Status::Fail }, None, Some(c));
            r.differing_pixels = differing;
            r
        }
        Err(e) => failed(e),
    }
}

/// A temporal case: the clip and its layers go to vfx as files and the output clip is
/// compared with the reference clip frame by frame.
fn run_clip(
    vfx: &Vfx,
    testset: &Path,
    case: &Case,
    max_ulp: u64,
    input_path: &Path,
    reference_path: &Path,
    result: impl Fn(Status, Option<String>, Option<Comparison>) -> CaseResult,
    failed: impl Fn(String) -> CaseResult,
) -> CaseResult {
    let Some(depth) = Depth::from_bits(case.depth) else {
        return failed(format!("case {} has depth {}", case.id, case.depth));
    };
    let input = match read_clip(input_path, case, "input") {
        Ok(b) => b,
        Err(e) => return failed(e),
    };
    let expected = match read_clip(reference_path, case, "reference") {
        Ok(b) => b,
        Err(e) => return failed(e),
    };
    let (output, notes) = match vfx.apply_clip_noted(case, testset, Threads::Inherit) {
        Ok(o) => o,
        Err(e) => return failed(e),
    };
    let failed = |error: String| CaseResult { notes: notes.clone(), ..failed(error) };
    let result = |status, error, comparison| CaseResult { notes: notes.clone(), ..result(status, error, comparison) };
    if output.len() != expected.len() {
        return failed(format!(
            "vfx wrote {} bytes, {} frames of {}x{} at {} bpc are {}",
            output.len(),
            case.frames,
            case.width,
            case.height,
            case.depth,
            expected.len()
        ));
    }
    let clip = match compare::compare_clip_within((case.width, case.height, depth), case.frames, &input, &expected, &output, max_ulp) {
        Ok(c) => c,
        Err(e) => return failed(e),
    };
    let mut r = match &clip.first {
        // a failing clip carries the details of its first failing frame
        Some((_, first)) => result(Status::Fail, None, Some(first.clone())),
        None => {
            let gaps = Comparison {
                max_abs_diff: clip.max_abs_diff,
                mean_abs_diff: clip.mean_abs_diff,
                max_ulp: clip.max_ulp,
                ..Comparison::default()
            };
            result(Status::Pass, None, Some(gaps))
        }
    };
    r.frame = clip.first.as_ref().map(|(f, _)| *f);
    r.failing_frames = clip.failing_frames;
    r.differing_pixels = clip.differing_pixels;
    // the clip's own error, not that of its first failing frame
    r.mean_abs_diff = Some(clip.mean_abs_diff);
    r.worst_frame = clip.worst_frame;
    r
}

/// Files in the references directory that no case in the test set asks for,
/// which usually means a misnamed reference.
pub fn unmatched_references(cases: &[Case], refs: &Path) -> Vec<String> {
    let wanted: HashSet<&str> = cases.iter().map(|c| c.reference.as_str()).collect();
    let mut stray: Vec<String> = std::fs::read_dir(refs)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().is_file())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|name| !wanted.contains(name.as_str()))
        .collect();
    stray.sort();
    stray
}

/// Every case of the (optionally filtered) test set, in manifest order.
/// `warn` gets a line for anything odd about the references directory.
pub fn run(opts: &RunOptions, warn: &dyn Fn(&str)) -> Result<Vec<CaseResult>, String> {
    let set = load_testset(&opts.testset)?;
    if !opts.refs.is_dir() {
        return Err(format!("the references directory {} does not exist", opts.refs.display()));
    }
    let stray = unmatched_references(&set.cases, &opts.refs);
    if !stray.is_empty() {
        warn(&format!("{} files in {} match no case, for example {}", stray.len(), opts.refs.display(), stray[0]));
    }
    let cases: Vec<Case> = set.cases.into_iter().filter(|c| opts.effect.as_ref().is_none_or(|e| &c.effect == e)).collect();
    if cases.is_empty() {
        return Err(match &opts.effect {
            Some(e) => format!("no cases for effect `{e}` in {}", opts.testset.display()),
            None => format!("{} has no cases", opts.testset.display()),
        });
    }
    let vfx = Vfx::locate(opts.vfx.as_deref())?;
    let results = run_cases(&vfx, opts, &cases);
    for line in missing_notes(&results) {
        warn(&line);
    }
    Ok(results)
}

/// Runs every case on `opts.jobs` threads, one result per case in case order.
fn run_cases(vfx: &Vfx, opts: &RunOptions, cases: &[Case]) -> Vec<CaseResult> {
    let slots: Vec<Mutex<Option<CaseResult>>> = cases.iter().map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..opts.jobs.clamp(1, cases.len()) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= cases.len() {
                    break;
                }
                let r = run_case_within(vfx, &opts.testset, &opts.refs, &cases[i], opts.max_ulp);
                *slots[i].lock().unwrap() = Some(r);
            });
        }
    });
    slots.into_iter().map(|s| s.into_inner().unwrap().expect("every case ran")).collect()
}

/// The gap ids whose note a family case must print: its own list in [`families`], or for a
/// case this build does not know, any note at all unless its family only observes. None when
/// no note is expected.
fn expected_notes(case: &Case) -> Option<Vec<String>> {
    let family = case.family.as_deref()?;
    match (families::find(family), families::family_case(case)) {
        (_, Some(f)) if f.notes.is_empty() => None,
        (_, Some(f)) => Some(f.notes.iter().map(|g| g.to_string()).collect()),
        (Some(i), None) if i.observe_only => None,
        _ => Some(Vec::new()),
    }
}

/// One warning per family case whose vfx run printed no note, or not the note of a gap it is
/// there to settle: either its settings no longer reach the rule, or the gap was retired and the
/// family still names it.
pub fn missing_notes(results: &[CaseResult]) -> Vec<String> {
    let mut out = Vec::new();
    for r in results {
        // only a run that went through has notes to judge
        if r.status == Status::NoReference || r.error.is_some() {
            continue;
        }
        let Some(want) = expected_notes(&r.case) else { continue };
        let family = r.case.family.as_deref().unwrap_or("");
        let printed: Vec<&str> = r.notes.iter().filter_map(|l| crate::vfx::note_gap(l)).collect();
        if want.is_empty() {
            if printed.is_empty() {
                out.push(format!("family case {} ({family}) printed no note: its settings may no longer reach an open rule", r.case.id));
            }
            continue;
        }
        let missing: Vec<&str> = want.iter().map(String::as_str).filter(|g| !printed.contains(g)).collect();
        if !missing.is_empty() {
            out.push(format!(
                "family case {} ({family}) printed no note for {}: its settings no longer reach the rule, or the gap was retired and the family still names it",
                r.case.id,
                missing.iter().map(|g| format!("{g} ({})", families::title(g))).collect::<Vec<_>>().join(", ")
            ));
        }
    }
    out
}

/// What the references say about one family of the run.
#[derive(Clone, Debug, PartialEq)]
pub struct FamilyGrade {
    pub family: String,
    pub cases: usize,
    /// Cases compared with a reference.
    pub compared: usize,
    /// Compared cases that are bit exact.
    pub exact: usize,
    /// Cases that FAIL: a difference, or a run or file that went wrong.
    pub failed: usize,
    /// Cases without a reference.
    pub missing: usize,
    /// The cases the family has in this build of fx-parity; 0 for a family it does not know.
    pub full: usize,
}

impl FamilyGrade {
    /// Every case compared and bit exact, and every case of the family present (a run narrowed to
    /// one effect settles nothing): the family's rule matches the original, its note retires.
    pub fn settled(&self) -> bool {
        self.cases > 0 && self.exact == self.cases && self.cases >= self.full
    }
}

/// One grade per family with a case in `results`, in the order the families first appear; empty
/// when no case belongs to a family.
pub fn family_grades(results: &[CaseResult]) -> Vec<FamilyGrade> {
    let mut grades: Vec<FamilyGrade> = Vec::new();
    for r in results {
        let Some(family) = &r.case.family else { continue };
        let at = match grades.iter().position(|g| &g.family == family) {
            Some(at) => at,
            None => {
                let full = families::cases_of(family).len();
                grades.push(FamilyGrade { family: family.clone(), cases: 0, compared: 0, exact: 0, failed: 0, missing: 0, full });
                grades.len() - 1
            }
        };
        let g = &mut grades[at];
        g.cases += 1;
        g.compared += usize::from(r.mean_abs_diff.is_some());
        g.exact += usize::from(r.is_exact());
        g.failed += usize::from(r.status == Status::Fail);
        g.missing += usize::from(r.status == Status::NoReference);
    }
    grades
}

fn write_families(results: &[CaseResult], out: &mut impl Write) -> std::io::Result<()> {
    let grades = family_grades(results);
    if grades.is_empty() {
        return Ok(());
    }
    writeln!(out, "\nfamilies")?;
    let width = grades.iter().map(|g| g.family.len()).max().unwrap_or(0);
    for g in &grades {
        writeln!(
            out,
            "  {:<width$}  {} cases, {} compared, {} bit exact, {} FAIL, {} no reference: {}",
            g.family,
            g.cases,
            g.compared,
            g.exact,
            g.failed,
            g.missing,
            if g.settled() { "settled, its note retires" } else { "open" }
        )?;
    }
    Ok(())
}

fn params_text(case: &Case) -> String {
    if case.params.is_empty() {
        "defaults".to_string()
    } else {
        case.params.join(" ")
    }
}

fn ulp_text(max_ulp: Option<u64>) -> String {
    match max_ulp {
        Some(u64::MAX) => ", max ulp: NaN mismatch".to_string(),
        Some(u) => format!(", max ulp {u}"),
        None => String::new(),
    }
}

/// What a status line says about a case that was compared and is not exact: how far off a
/// pass is, the mean error, and for a clip its worst frame.
fn grade_note(r: &CaseResult) -> Option<String> {
    let mean = r.mean_abs_diff.filter(|m| *m > 0.0)?;
    let mut note = String::new();
    if let (Status::Pass, Some(cmp)) = (r.status, &r.comparison) {
        note.push_str(&format!("within tolerance, max abs diff {}{}, ", compare::diff_text(cmp.max_abs_diff), ulp_text(cmp.max_ulp)));
    }
    note.push_str(&format!("mean error {} of full scale", compare::percent_text(mean)));
    if let Some((frame, worst)) = r.worst_frame {
        note.push_str(&format!(", worst frame {frame} ({})", compare::percent_text(worst)));
    }
    Some(note)
}

/// How many cases the `graded error` block lists as the worst.
pub const WORST_LISTED: usize = 10;

/// The graded error of one effect's compared cases.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectGrade {
    pub effect: String,
    /// Cases that were compared with a reference: not those without one, nor those that errored.
    pub compared: usize,
    /// Compared cases with no error at all.
    pub exact: usize,
    /// The mean of the compared cases' mean errors.
    pub mean_abs_diff: f64,
    /// The compared case with the largest mean error (the first on a tie); None when all are exact.
    pub worst: Option<(String, f64)>,
}

/// One grade per effect that has a compared case, in the order the effects first appear.
pub fn graded(results: &[CaseResult]) -> Vec<EffectGrade> {
    let mut grades: Vec<(EffectGrade, f64)> = Vec::new();
    for r in results {
        let Some(mean) = r.mean_abs_diff else { continue };
        let at = match grades.iter().position(|(g, _)| g.effect == r.case.effect) {
            Some(at) => at,
            None => {
                let empty = EffectGrade { effect: r.case.effect.clone(), compared: 0, exact: 0, mean_abs_diff: 0.0, worst: None };
                grades.push((empty, 0.0));
                grades.len() - 1
            }
        };
        let (g, sum) = &mut grades[at];
        g.compared += 1;
        g.exact += usize::from(r.is_exact());
        *sum += mean;
        if mean > g.worst.as_ref().map_or(0.0, |w| w.1) {
            g.worst = Some((r.case.id.clone(), mean));
        }
    }
    grades
        .into_iter()
        .map(|(mut g, sum)| {
            g.mean_abs_diff = sum / g.compared as f64;
            g
        })
        .collect()
}

/// The `n` compared cases with the largest mean error, worst first. Cases without error are
/// never listed and a tie keeps the report order.
pub fn worst_cases(results: &[CaseResult], n: usize) -> Vec<&CaseResult> {
    let mut bad: Vec<&CaseResult> = results.iter().filter(|r| r.mean_abs_diff.is_some_and(|m| m > 0.0)).collect();
    bad.sort_by(|a, b| b.mean_abs_diff.unwrap_or(0.0).total_cmp(&a.mean_abs_diff.unwrap_or(0.0)));
    bad.truncate(n);
    bad
}

fn write_graded(results: &[CaseResult], out: &mut impl Write) -> std::io::Result<()> {
    writeln!(out, "\ngraded error")?;
    let grades = graded(results);
    if grades.is_empty() {
        return writeln!(out, "  no case was compared with a reference");
    }
    let width = grades.iter().map(|g| g.effect.len()).max().unwrap_or(0);
    for g in &grades {
        write!(
            out,
            "  {:<width$}  {} compared, {} bit exact, mean error {} of full scale",
            g.effect,
            g.compared,
            g.exact,
            compare::percent_text(g.mean_abs_diff)
        )?;
        if let Some((id, worst)) = &g.worst {
            write!(out, ", worst {id} ({})", compare::percent_text(*worst))?;
        }
        writeln!(out)?;
    }
    let worst = worst_cases(results, WORST_LISTED);
    if !worst.is_empty() {
        writeln!(out, "\n{} worst case{} by mean error", worst.len(), if worst.len() == 1 { "" } else { "s" })?;
        for (i, r) in worst.iter().enumerate() {
            writeln!(out, "  {:>2}  {:>9}  {}", i + 1, compare::percent_text(r.mean_abs_diff.unwrap_or(0.0)), r.case.id)?;
        }
    }
    Ok(())
}

/// The human report. One line per case, details under each failure, then the graded error
/// of the cases that were compared and the totals. `brief` leaves the passes and the cases
/// without a reference out of the case lines.
pub fn write_human(results: &[CaseResult], brief: bool, out: &mut impl Write) -> std::io::Result<()> {
    const PAD: &str = "              ";
    for r in results {
        if brief && r.status != Status::Fail {
            continue;
        }
        write!(out, "{:<12}  {}", r.status.word(), r.case.id)?;
        if let Some(note) = grade_note(r) {
            write!(out, "  ({note})")?;
        }
        writeln!(out)?;
        if r.status != Status::Fail {
            continue;
        }
        let c = &r.case;
        writeln!(
            out,
            "{PAD}effect {}  {}x{}  {} bpc  source {}  params: {}",
            c.effect,
            c.width,
            c.height,
            c.depth,
            c.source,
            params_text(c)
        )?;
        if c.is_temporal() {
            let layers: Vec<String> = c.layers.keys().cloned().collect();
            writeln!(
                out,
                "{PAD}clip: {} frames at {} fps  layers: {}",
                c.frames,
                c.fps.as_deref().unwrap_or("?"),
                if layers.is_empty() { "none".to_string() } else { layers.join(", ") }
            )?;
        }
        writeln!(out, "{PAD}input:     {}", r.input_path.display())?;
        writeln!(out, "{PAD}reference: {}", r.reference_path.display())?;
        writeln!(out, "{PAD}repro:     {}", r.repro())?;
        if let Some(e) = &r.error {
            writeln!(out, "{PAD}{e}")?;
        }
        if let (Some(frame), true) = (r.frame, c.is_temporal()) {
            writeln!(out, "{PAD}first failing frame: {frame} of {} ({} frames fail)", c.frames, r.failing_frames)?;
        }
        if let Some(cmp) = &r.comparison {
            writeln!(
                out,
                "{PAD}{} of {} pixels differ, max abs diff {}{}",
                cmp.differing_pixels,
                c.width * c.height,
                compare::diff_text(cmp.max_abs_diff),
                ulp_text(cmp.max_ulp)
            )?;
            for d in &cmp.first {
                writeln!(out, "{PAD}{d}")?;
            }
            if cmp.differing_pixels > cmp.first.len() {
                writeln!(out, "{PAD}... {} more", cmp.differing_pixels - cmp.first.len())?;
            }
        }
    }
    write_graded(results, out)?;
    write_families(results, out)?;
    let t = totals(results);
    writeln!(out, "\n{} pass, {} FAIL, {} no reference, {} cases", t.pass, t.fail, t.no_reference, results.len())
}

fn number(v: f64) -> Value {
    serde_json::Number::from_f64(v).map(Value::Number).unwrap_or_else(|| Value::String(v.to_string()))
}

/// The graded error per effect, as the JSON report holds it.
fn graded_json(results: &[CaseResult]) -> Value {
    let mut by_effect = serde_json::Map::new();
    for g in graded(results) {
        let worst = g.worst.map(|(id, mean)| json!({ "id": id, "mean_abs_diff": number(mean) }));
        by_effect.insert(
            g.effect,
            json!({ "compared": g.compared, "exact": g.exact, "mean_abs_diff": number(g.mean_abs_diff), "worst": worst }),
        );
    }
    Value::Object(by_effect)
}

/// The JSON report. `brief` leaves out everything but the failures, as in the human report;
/// the summary and the graded error still count every case.
pub fn to_json(results: &[CaseResult], brief: bool) -> Value {
    let t = totals(results);
    let cases: Vec<Value> = results
        .iter()
        .filter(|r| !brief || r.status == Status::Fail)
        .map(|r| {
            let c = &r.case;
            let mut v = json!({
                "id": c.id,
                "effect": c.effect,
                "label": c.label,
                "params": c.params,
                "depth": c.depth,
                "width": c.width,
                "height": c.height,
                "source": c.source,
                "input": c.input,
                "reference": c.reference,
                "status": r.status.word(),
                "kind": c.kind,
                "frames": c.frames,
            });
            if c.is_temporal() {
                v["fps"] = json!(c.fps);
                v["layers"] = json!(c.layers);
                if let Some(frame) = r.frame {
                    v["frame"] = json!(frame);
                    v["failing_frames"] = json!(r.failing_frames);
                }
                if r.comparison.is_some() {
                    v["clip_differing_pixels"] = json!(r.differing_pixels);
                }
            }
            if r.status == Status::Fail {
                v["input_path"] = json!(r.input_path.display().to_string());
                v["reference_path"] = json!(r.reference_path.display().to_string());
                v["repro"] = json!(r.repro());
            }
            if let Some(e) = &r.error {
                v["error"] = json!(e);
            }
            if let Some(mean) = r.mean_abs_diff {
                v["mean_abs_diff"] = number(mean);
            }
            if let Some((frame, mean)) = r.worst_frame {
                v["worst_frame"] = json!({ "frame": frame, "mean_abs_diff": number(mean) });
            }
            if let Some(family) = &c.family {
                v["family"] = json!(family);
                v["notes"] = json!(r.notes);
            }
            if let Some(cmp) = &r.comparison {
                v["differing_pixels"] = json!(cmp.differing_pixels);
                v["max_abs_diff"] = number(cmp.max_abs_diff);
                if let Some(u) = cmp.max_ulp {
                    v["max_ulp"] = if u == u64::MAX { json!("nan mismatch") } else { json!(u) };
                }
                v["mismatches"] = Value::Array(cmp.first.iter().map(|d| d.to_json()).collect());
            }
            v
        })
        .collect();
    let mut report = json!({
        "summary": { "pass": t.pass, "fail": t.fail, "no_reference": t.no_reference, "total": results.len() },
        "graded": graded_json(results),
        "cases": cases,
    });
    let grades = family_grades(results);
    if !grades.is_empty() {
        let mut by_family = serde_json::Map::new();
        for g in grades {
            let v = json!({
                "cases": g.cases, "compared": g.compared, "exact": g.exact, "failed": g.failed, "missing": g.missing, "settled": g.settled(),
            });
            by_family.insert(g.family, v);
        }
        report["families"] = Value::Object(by_family);
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cases::CaseKind;
    use crate::compare::{PixelDiff, Quad};
    use std::collections::BTreeMap;

    fn case(params: &[&str]) -> Case {
        Case {
            id: "fx.all-min.noise-2x1.d8.deadbeef".into(),
            effect: "fx".into(),
            label: "all-min".into(),
            params: params.iter().map(|s| s.to_string()).collect(),
            depth: 8,
            width: 2,
            height: 1,
            source: "noise".into(),
            input: "inputs/x".into(),
            reference: "r".into(),
            kind: CaseKind::Frame,
            frames: 1,
            fps: None,
            layers: BTreeMap::new(),
            family: None,
        }
    }

    fn result(status: Status, error: Option<&str>, comparison: Option<Comparison>) -> CaseResult {
        CaseResult {
            case: case(&["amount=0", "mode=1"]),
            status,
            error: error.map(str::to_string),
            mean_abs_diff: comparison.as_ref().map(|c| c.mean_abs_diff),
            comparison,
            frame: None,
            failing_frames: 0,
            differing_pixels: 0,
            worst_frame: None,
            testset: PathBuf::from("set"),
            input_path: PathBuf::from("set/inputs/x"),
            reference_path: PathBuf::from("refs/r"),
            notes: Vec::new(),
        }
    }

    /// A 12 frame clip case with one layer whose frame 5 is the first to fail.
    fn clip_failing() -> CaseResult {
        let mut r = failing();
        r.case.kind = CaseKind::Temporal;
        r.case.frames = 12;
        r.case.fps = Some("30000/1001".into());
        r.case.params = vec!["gain=0@0;10@11".into()];
        r.case.input = "inputs/x.f12".into();
        r.case.layers = BTreeMap::from([("overlay".to_string(), "inputs/o.f12".to_string())]);
        r.input_path = PathBuf::from("set/inputs/x.f12");
        r.frame = Some(5);
        r.failing_frames = 3;
        // the clip's own error and its worst frame, not those of frame 5
        r.mean_abs_diff = Some(0.01);
        r.worst_frame = Some((8, 0.03));
        r
    }

    /// One channel of the 2x1 frame is off by one level: 1 / (8 * 255) of full scale.
    fn failing() -> CaseResult {
        let first = vec![PixelDiff { x: 1, y: 0, input: Quad::U8([1, 2, 3, 4]), expected: Quad::U8([5, 6, 7, 8]), actual: Quad::U8([5, 6, 7, 9]) }];
        let cmp = Comparison { differing_pixels: 1, max_abs_diff: 1.0, mean_abs_diff: 1.0 / (8.0 * 255.0), max_ulp: None, first };
        result(Status::Fail, None, Some(cmp))
    }

    fn human(results: &[CaseResult], brief: bool) -> String {
        let mut buf = Vec::new();
        write_human(results, brief, &mut buf).unwrap();
        String::from_utf8(buf).unwrap()
    }

    #[test]
    fn human_report_has_what_an_effect_author_needs() {
        let ok = result(Status::Pass, None, Some(Comparison::default()));
        let none = result(Status::NoReference, None, None);
        let text = human(&[ok, failing(), none], false);
        assert!(text.contains("pass          fx.all-min"), "{text}");
        assert!(text.contains("no reference  fx.all-min"), "{text}");
        assert!(text.contains("FAIL          fx.all-min"), "{text}");
        assert!(text.contains("params: amount=0 mode=1"), "{text}");
        assert!(text.contains("1 of 2 pixels differ, max abs diff 1"), "{text}");
        assert!(text.contains("x=1 y=0 in=[1, 2, 3, 4] expected=[5, 6, 7, 8] actual=[5, 6, 7, 9]"), "{text}");
        assert!(text.contains("input:     set/inputs/x"), "{text}");
        assert!(text.contains("reference: refs/r"), "{text}");
        assert!(text.contains("repro:     vfx apply fx --width 2 --height 1 --depth 8 --param amount=0 --param mode=1 --in set/inputs/x"), "{text}");
        assert!(text.contains("1 pass, 1 FAIL, 1 no reference, 3 cases"), "{text}");
    }

    #[test]
    fn a_pass_that_is_not_bit_exact_says_so() {
        let near = Comparison { max_abs_diff: 5.9604645e-8, mean_abs_diff: 1.5e-8, max_ulp: Some(1), ..Comparison::default() };
        let text = human(&[result(Status::Pass, None, Some(near))], false);
        assert!(text.contains("(within tolerance, max abs diff 5.9604645e-8, max ulp 1, mean error 1.50e-6% of full scale)"), "{text}");
        let exact = human(&[result(Status::Pass, None, Some(Comparison::default()))], false);
        assert!(!exact.contains("tolerance"), "{exact}");
    }

    #[test]
    fn repro_quotes_paths_with_spaces() {
        let mut r = failing();
        r.input_path = PathBuf::from("my set/inputs/x");
        assert!(r.repro().ends_with("--in \"my set/inputs/x\""), "{}", r.repro());
    }

    #[test]
    fn a_clip_failure_names_the_frame_the_layers_and_the_command_to_rerun_it() {
        let text = human(&[clip_failing()], false);
        assert!(text.contains("clip: 12 frames at 30000/1001 fps  layers: overlay"), "{text}");
        assert!(text.contains("first failing frame: 5 of 12 (3 frames fail)"), "{text}");
        assert!(text.contains("1 of 2 pixels differ"), "{text}");
        assert!(text.contains("x=1 y=0 in=[1, 2, 3, 4] expected=[5, 6, 7, 8] actual=[5, 6, 7, 9]"), "{text}");
        let sep = std::path::MAIN_SEPARATOR;
        let repro = text.lines().find(|l| l.contains("repro:")).unwrap();
        assert!(repro.contains("vfx apply fx --width 2 --height 1 --depth 8 --param \"gain=0@0;10@11\" --fps 30000/1001 --frames 12 --layer "), "{repro}");
        assert!(repro.ends_with(&format!("--in set{sep}inputs/x.f12")), "{repro}");
        assert!(repro.contains(&format!("overlay=set{sep}inputs/o.f12")), "{repro}");
        // a still never mentions frames
        assert!(!human(&[failing()], false).contains("first failing frame"));
    }

    #[test]
    fn a_clip_failure_is_in_the_json_too() {
        let v = to_json(&[clip_failing()], false);
        let c = &v["cases"][0];
        assert_eq!((c["kind"].as_str(), c["frames"].as_u64(), c["fps"].as_str()), (Some("temporal"), Some(12), Some("30000/1001")));
        assert_eq!((c["frame"].as_u64(), c["failing_frames"].as_u64()), (Some(5), Some(3)));
        assert_eq!(c["layers"]["overlay"], "inputs/o.f12");
        assert_eq!(c["params"][0], "gain=0@0;10@11");
        assert_eq!(c["mismatches"][0]["x"], 1);
        let still = to_json(&[failing()], false);
        assert_eq!(still["cases"][0]["kind"], "frame");
        assert_eq!(still["cases"][0]["frames"], 1);
        assert!(still["cases"][0].get("frame").is_none() && still["cases"][0].get("layers").is_none());
    }

    #[test]
    fn quoting_covers_what_a_shell_would_act_on() {
        assert_eq!(quote("plain=1"), "plain=1");
        assert_eq!(quote("a=1@0;2@3"), "\"a=1@0;2@3\"");
        assert_eq!(quote("two words"), "\"two words\"");
        assert_eq!(quote("a|b"), "\"a|b\"");
    }

    #[test]
    fn brief_lists_only_failures_but_still_counts_everything() {
        let ok = result(Status::Pass, None, Some(Comparison::default()));
        let none = result(Status::NoReference, None, None);
        let text = human(&[ok, failing(), none], true);
        assert!(!text.contains("pass          fx") && !text.contains("no reference  fx"), "{text}");
        assert!(text.contains("FAIL          fx.all-min"), "{text}");
        assert!(text.contains("1 pass, 1 FAIL, 1 no reference, 3 cases"), "{text}");
    }

    #[test]
    fn json_report_carries_the_same_facts() {
        let v = to_json(&[failing()], false);
        assert_eq!(v["summary"]["fail"], 1);
        let c = &v["cases"][0];
        assert_eq!(c["status"], "FAIL");
        assert_eq!(c["params"][1], "mode=1");
        assert_eq!(c["differing_pixels"], 1);
        assert_eq!(c["mismatches"][0]["x"], 1);
        assert_eq!(c["mismatches"][0]["actual"][3], 9);
        assert!(c["repro"].as_str().unwrap().starts_with("vfx apply fx "));
        assert_eq!(c["reference_path"], "refs/r");
    }

    #[test]
    fn brief_json_keeps_the_summary_and_drops_the_rest() {
        let ok = result(Status::Pass, None, Some(Comparison::default()));
        let v = to_json(&[ok, failing(), result(Status::NoReference, None, None)], true);
        assert_eq!(v["summary"]["total"], 3);
        assert_eq!(v["cases"].as_array().unwrap().len(), 1);
        assert_eq!(v["cases"][0]["status"], "FAIL");
    }

    #[test]
    fn errors_are_failures_not_passes() {
        let r = result(Status::Fail, Some("vfx exited with code 3: boom"), None);
        assert_eq!(r.status, Status::Fail);
        assert!(human(&[r], false).contains("vfx exited with code 3: boom"));
    }

    #[test]
    fn stray_reference_files_are_reported() {
        let dir = std::env::temp_dir().join(format!("fx-parity-stray-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("r"), b"x").unwrap();
        std::fs::write(dir.join("typo.rgba"), b"x").unwrap();
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        assert_eq!(unmatched_references(&[case(&[])], &dir), ["typo.rgba"]);
        assert!(unmatched_references(&[case(&[])], &dir.join("missing")).is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn totals_count_each_status() {
        let mk = |status| result(status, None, None);
        let t = totals(&[mk(Status::Pass), mk(Status::Pass), mk(Status::Fail), mk(Status::NoReference)]);
        assert_eq!(t, Totals { pass: 2, fail: 1, no_reference: 1 });
    }

    /// A compared case of `effect` with the given mean error; a pass when the error is zero.
    fn graded_case(effect: &str, n: usize, mean: f64) -> CaseResult {
        let cmp = Comparison { max_abs_diff: mean, mean_abs_diff: mean, ..Comparison::default() };
        let mut r = result(if mean == 0.0 { Status::Pass } else { Status::Fail }, None, Some(cmp));
        r.case.effect = effect.to_string();
        r.case.id = format!("{effect}.case{n}");
        r
    }

    #[test]
    fn every_non_exact_line_shows_its_mean_error_and_exact_ones_do_not() {
        let fail = human(&[failing()], false);
        assert!(fail.contains("FAIL          fx.all-min.noise-2x1.d8.deadbeef  (mean error 0.049% of full scale)\n"), "{fail}");
        let ok = human(&[result(Status::Pass, None, Some(Comparison::default()))], false);
        let first = ok.lines().next().unwrap();
        assert!(first.starts_with("pass          fx.all-min") && !first.contains("mean error"), "{first}");
        // no reference and a failed run have nothing to grade
        let none = human(&[result(Status::NoReference, None, None), result(Status::Fail, Some("boom"), None)], false);
        assert!(!none.lines().take(3).any(|l| l.contains("mean error")), "{none}");
        // a clip adds its worst frame, which can differ from the first failing frame
        let clip = human(&[clip_failing()], false);
        assert!(clip.contains("(mean error 1% of full scale, worst frame 8 (3%))"), "{clip}");
        assert!(clip.contains("first failing frame: 5 of 12"), "{clip}");
    }

    #[test]
    fn the_graded_block_has_a_line_per_effect_and_the_worst_cases() {
        let mut rs = vec![
            graded_case("alpha", 1, 0.0),
            graded_case("alpha", 2, 0.02),
            graded_case("alpha", 3, 0.0),
            graded_case("alpha", 4, 0.01),
            graded_case("beta", 1, 0.0),
            graded_case("beta", 2, 0.0),
            result(Status::NoReference, None, None),
            result(Status::Fail, Some("vfx exited with code 3"), None),
        ];
        let text = human(&rs, true);
        let block: Vec<&str> = text.lines().skip_while(|l| *l != "graded error").collect();
        assert_eq!(block[0], "graded error");
        assert_eq!(
            block[1],
            "  alpha  4 compared, 2 bit exact, mean error 0.75% of full scale, worst alpha.case2 (2%)"
        );
        assert_eq!(block[2], "  beta   2 compared, 2 bit exact, mean error 0% of full scale");
        let row = |n: usize, pct: &str, id: &str| format!("  {n:>2}  {pct:>9}  {id}");
        assert_eq!(block[4], "2 worst cases by mean error");
        assert_eq!(block[5], row(1, "2%", "alpha.case2"));
        assert_eq!(block[6], row(2, "1%", "alpha.case4"));
        assert_eq!(block.len(), 9, "{block:?}");
        assert_eq!(block[8], "4 pass, 3 FAIL, 1 no reference, 8 cases", "the totals stay last");
        // the block is printed with --brief and without it
        assert_eq!(human(&rs, false).lines().filter(|l| *l == "graded error").count(), 1);

        // only the ten worst are listed, worst first
        rs = (0..14).map(|i| graded_case("gamma", i, 0.001 * (1 + i % 12) as f64)).collect();
        let text = human(&rs, true);
        let tail = text.split("10 worst cases by mean error
").nth(1).expect("the heading names ten cases");
        let listed: Vec<&str> = tail.lines().take_while(|l| !l.is_empty()).collect();
        assert_eq!(listed.len(), 10, "{text}");
        assert_eq!(listed[0], row(1, "1.2%", "gamma.case11"));
        assert_eq!(listed[1], row(2, "1.1%", "gamma.case10"));
        assert_eq!(listed[9], row(10, "0.3%", "gamma.case2"));
    }

    #[test]
    fn a_report_with_nothing_compared_says_so_and_keeps_the_totals() {
        let text = human(&[result(Status::NoReference, None, None)], false);
        assert!(text.contains("\ngraded error\n  no case was compared with a reference\n"), "{text}");
        assert!(text.ends_with("0 pass, 0 FAIL, 1 no reference, 1 cases\n"), "{text}");
        assert!(!text.contains("worst case"));
        assert!(graded(&[]).is_empty());
        assert!(worst_cases(&[], 10).is_empty());
    }

    #[test]
    fn grades_count_exact_cases_and_average_the_mean_errors() {
        let rs = vec![graded_case("a", 1, 0.0), graded_case("b", 1, 0.5), graded_case("a", 2, 0.25), graded_case("a", 3, 0.25)];
        let g = graded(&rs);
        assert_eq!(g.len(), 2);
        assert_eq!((g[0].effect.as_str(), g[0].compared, g[0].exact), ("a", 3, 1));
        assert!((g[0].mean_abs_diff - 0.5 / 3.0).abs() < 1e-12);
        assert_eq!(g[0].worst, Some(("a.case2".to_string(), 0.25)), "the first of two equal cases");
        assert_eq!((g[1].effect.as_str(), g[1].exact, g[1].worst.clone()), ("b", 0, Some(("b.case1".to_string(), 0.5))));
        let all_exact = graded(&[graded_case("a", 1, 0.0)]);
        assert_eq!(all_exact[0].worst, None);
        let order: Vec<&str> = worst_cases(&rs, 2).iter().map(|r| r.case.id.as_str()).collect();
        assert_eq!(order, ["b.case1", "a.case2"]);
    }

    #[test]
    fn the_json_report_has_the_mean_per_case_and_the_graded_block_next_to_the_summary() {
        let mut rs = vec![graded_case("alpha", 1, 0.0), graded_case("alpha", 2, 0.02), clip_failing(), result(Status::NoReference, None, None)];
        rs[2].case.id = "fx.clip".into();
        let v = to_json(&rs, false);
        let alpha = &v["graded"]["alpha"];
        assert_eq!((alpha["compared"].as_u64(), alpha["exact"].as_u64()), (Some(2), Some(1)));
        assert_eq!(alpha["mean_abs_diff"].as_f64(), Some(0.01));
        assert_eq!(alpha["worst"]["id"], "alpha.case2");
        assert_eq!(alpha["worst"]["mean_abs_diff"].as_f64(), Some(0.02));
        // clip_failing keeps the effect `fx`; its worst case is the clip
        assert_eq!(v["graded"]["fx"]["worst"]["id"], "fx.clip");
        assert_eq!(v["summary"]["total"], 4);
        let c = &v["cases"];
        assert_eq!(c[0]["mean_abs_diff"].as_f64(), Some(0.0));
        assert_eq!(c[1]["mean_abs_diff"].as_f64(), Some(0.02));
        assert!(c[1].get("worst_frame").is_none(), "a still has no worst frame");
        assert_eq!(c[2]["mean_abs_diff"].as_f64(), Some(0.01));
        assert_eq!(c[2]["worst_frame"]["frame"], 8);
        assert_eq!(c[2]["worst_frame"]["mean_abs_diff"].as_f64(), Some(0.03));
        assert!(c[3].get("mean_abs_diff").is_none(), "no reference, nothing graded");
        // an effect with only exact cases has a null worst
        let exact = to_json(&[graded_case("e", 1, 0.0)], false);
        assert_eq!(exact["graded"]["e"]["worst"], Value::Null);
        // --brief leaves out the passes but not their share of the graded block
        let brief = to_json(&rs, true);
        assert_eq!(brief["cases"].as_array().unwrap().len(), 2);
        assert_eq!(brief["graded"]["alpha"]["compared"], 2);
        assert!(to_json(&[], false)["graded"].as_object().unwrap().is_empty());
    }

    /// A temporal case of `family` with the given label, a result with `status` and these notes.
    fn family_result(family: &str, label: &str, status: Status, mean: Option<f64>, notes: &[&str]) -> CaseResult {
        let mut r = result(status, None, mean.map(|m| Comparison { max_abs_diff: m, mean_abs_diff: m, ..Comparison::default() }));
        r.case.family = Some(family.to_string());
        r.case.label = label.to_string();
        r.case.id = format!("fx.{label}");
        r.notes = notes.iter().map(|g| format!("vfx: note: retime-fgbg: {g} some title: a stated rule")).collect();
        r
    }

    #[test]
    fn family_grades_count_each_family_and_settle_only_when_every_case_is_exact() {
        let rs = vec![
            family_result("gap-a", "gap-a-x", Status::Pass, Some(0.0), &[]),
            family_result("gap-b", "gap-b-x", Status::Fail, Some(0.5), &[]),
            family_result("gap-a", "gap-a-y", Status::Pass, Some(0.0), &[]),
            family_result("gap-b", "gap-b-y", Status::NoReference, None, &[]),
            result(Status::Pass, None, Some(Comparison::default())),
        ];
        let g = family_grades(&rs);
        assert_eq!(g.len(), 2);
        assert_eq!((g[0].family.as_str(), g[0].cases, g[0].compared, g[0].exact, g[0].failed, g[0].missing), ("gap-a", 2, 2, 2, 0, 0));
        assert!(g[0].settled());
        assert_eq!((g[1].cases, g[1].compared, g[1].exact, g[1].failed, g[1].missing), (2, 1, 0, 1, 1));
        assert!(!g[1].settled());
        // a family without a single compared case is not settled
        assert!(!family_grades(&[family_result("gap-c", "gap-c-x", Status::NoReference, None, &[])])[0].settled());
        // nor is a known family of which only some cases ran, however exact they are
        let part = family_grades(&[family_result("gap-debug", "gap-debug-cv-blend-lo2-track1", Status::Pass, Some(0.0), &[])]);
        assert_eq!((part[0].cases, part[0].exact, part[0].full), (1, 1, 4));
        assert!(!part[0].settled());
        let every: Vec<CaseResult> = ["lo2-track1", "lo5-track1", "lo2-track2", "lo5-track2"]
            .iter()
            .map(|t| family_result("gap-debug", &format!("gap-debug-cv-blend-{t}"), Status::Pass, Some(0.0), &[]))
            .collect();
        assert!(family_grades(&every)[0].settled());
        let v = to_json(&rs, false);
        assert_eq!(v["families"]["gap-a"], json!({"cases": 2, "compared": 2, "exact": 2, "failed": 0, "missing": 0, "settled": true}));
        assert_eq!(v["families"]["gap-b"]["settled"], false);
        assert_eq!(v["cases"][0]["family"], "gap-a");
        assert_eq!(v["cases"][4].get("family"), None);
        let text = human(&rs, false);
        assert!(text.contains("\nfamilies\n  gap-a  2 cases, 2 compared, 2 bit exact, 0 FAIL, 0 no reference: settled, its note retires\n"), "{text}");
        assert!(text.contains("  gap-b  2 cases, 1 compared, 0 bit exact, 1 FAIL, 1 no reference: open\n"), "{text}");
        assert!(text.ends_with("3 pass, 1 FAIL, 1 no reference, 5 cases\n"), "{text}");
    }

    #[test]
    fn a_report_without_family_cases_is_the_report_of_before() {
        let rs = vec![result(Status::Pass, None, Some(Comparison::default())), failing(), result(Status::NoReference, None, None)];
        assert!(family_grades(&rs).is_empty());
        let v = to_json(&rs, false);
        assert!(v.get("families").is_none());
        assert!(v["cases"].as_array().unwrap().iter().all(|c| c.get("family").is_none() && c.get("notes").is_none()));
        assert_eq!(v.as_object().unwrap().keys().collect::<Vec<_>>(), ["cases", "graded", "summary"]);
        assert!(!human(&rs, false).contains("families"));
        assert!(missing_notes(&rs).is_empty());
    }

    #[test]
    fn a_family_case_that_prints_no_note_is_a_warning() {
        // gap-debug's cv-blend-lo2-track1 must note G4
        let label = "gap-debug-cv-blend-lo2-track1";
        let both = family_result("gap-debug", label, Status::Pass, Some(0.0), &["G4", "G5"]);
        assert!(missing_notes(&[both]).is_empty());
        let none = family_result("gap-debug", label, Status::Pass, Some(0.0), &[]);
        let w = missing_notes(&[none]);
        assert_eq!(w.len(), 1);
        assert!(w[0].contains("fx.gap-debug-cv-blend-lo2-track1 (gap-debug) printed no note for G4 (one-layer Debug Blend view)"), "{}", w[0]);
        let one = family_result("gap-debug", label, Status::Fail, Some(0.1), &["G5"]);
        assert!(missing_notes(&[one])[0].contains("printed no note for G4 (one-layer Debug Blend view):"), "a failing comparison still ran vfx");
        // no run, no judgement: no reference, or an error
        assert!(missing_notes(&[family_result("gap-debug", label, Status::NoReference, None, &[])]).is_empty());
        let mut broken = family_result("gap-debug", label, Status::Fail, None, &[]);
        broken.error = Some("vfx exited with code 3: vfx: error: unsupported: x".into());
        assert!(missing_notes(&[broken]).is_empty());
        // the observing families expect nothing
        assert!(missing_notes(&[family_result("gap-tiny", "gap-tiny-cv-1x1", Status::Pass, Some(0.0), &[])]).is_empty());
        // a case this build does not know: any note will do, unless its family only observes
        let unknown = family_result("gap-debug", "gap-debug-future", Status::Pass, Some(0.0), &[]);
        assert!(missing_notes(&[unknown])[0].contains("printed no note: its settings may no longer reach an open rule"));
        assert!(missing_notes(&[family_result("gap-debug", "gap-debug-future", Status::Pass, Some(0.0), &["G5"])]).is_empty());
        assert!(missing_notes(&[family_result("gap-keys", "gap-keys-future", Status::Pass, Some(0.0), &[])]).is_empty());
        assert!(missing_notes(&[family_result("gap-later", "gap-later-x", Status::Pass, Some(0.0), &[])])[0].contains("printed no note"));
    }

    #[test]
    fn the_report_text_is_plain_ascii() {
        let rs = vec![graded_case("alpha", 1, 0.0), graded_case("alpha", 2, 0.02), clip_failing(), failing()];
        assert!(human(&rs, false).is_ascii());
        assert!(to_json(&rs, false).to_string().is_ascii());
    }
}
