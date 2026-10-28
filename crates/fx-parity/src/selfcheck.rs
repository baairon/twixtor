//! `fx-parity selfcheck`: the engine must be deterministic and give the same bytes for
//! any thread count. A subset of the temporal cases is run again and again through
//! `vfx`, with the default rayon thread count and with `RAYON_NUM_THREADS` set, and
//! every output must equal the first one byte for byte. No references are needed.

use crate::cases::Case;
use crate::run::load_testset;
use crate::vfx::{Threads, Vfx};
use serde_json::{json, Value};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

pub struct SelfcheckOptions {
    pub testset: PathBuf,
    pub effect: Option<String>,
    /// How many cases to check; 0 checks every temporal case.
    pub max_cases: usize,
    /// Thread counts to compare with the default one.
    pub threads: Vec<usize>,
    pub vfx: Option<PathBuf>,
    pub jobs: usize,
}

/// Two outputs of one case that should have been the same and were not.
#[derive(Clone, Debug, PartialEq)]
pub struct Difference {
    /// What the first output was compared with, for example `repeat run` or `RAYON_NUM_THREADS=1`.
    pub against: String,
    pub detail: String,
}

#[derive(Debug)]
pub struct CaseCheck {
    pub case: Case,
    /// How many times vfx ran.
    pub runs: usize,
    /// Set when a run failed, which stops the case.
    pub error: Option<String>,
    pub differences: Vec<Difference>,
}

impl CaseCheck {
    pub fn ok(&self) -> bool {
        self.error.is_none() && self.differences.is_empty()
    }
}

/// At most `max` temporal cases, always at least one per effect, spread evenly through
/// each effect's cases (so the combined settings and the sweeps are both visited).
pub fn pick(cases: &[Case], effect: Option<&str>, max: usize) -> Vec<Case> {
    let mut groups: Vec<(&str, Vec<&Case>)> = Vec::new();
    for c in cases.iter().filter(|c| c.is_temporal() && effect.is_none_or(|e| c.effect == e)) {
        match groups.iter_mut().find(|g| g.0 == c.effect) {
            Some(g) => g.1.push(c),
            None => groups.push((&c.effect, vec![c])),
        }
    }
    let total: usize = groups.iter().map(|g| g.1.len()).sum();
    if max == 0 || total <= max {
        return groups.into_iter().flat_map(|g| g.1).cloned().collect();
    }
    let n = groups.len();
    let mut picked = Vec::new();
    for (gi, (_, list)) in groups.iter().enumerate() {
        // the budget splits evenly, the first effects take the remainder, nobody gets less than one
        let quota = (max / n + usize::from(gi < max % n)).clamp(1, list.len());
        picked.extend((0..quota).map(|k| list[k * list.len() / quota].clone()));
    }
    picked
}

/// Where two outputs first differ, or None when they are the same bytes.
pub fn describe_difference(case: &Case, against: &str, first: &[u8], other: &[u8]) -> Option<Difference> {
    if first == other {
        return None;
    }
    let detail = if first.len() != other.len() {
        format!("the output has {} bytes against {}", other.len(), first.len())
    } else {
        let len = case.frame_len().max(1);
        let differing_bytes = first.iter().zip(other).filter(|(a, b)| a != b).count();
        let frames: Vec<usize> = (0..first.len() / len).filter(|f| first[f * len..(f + 1) * len] != other[f * len..(f + 1) * len]).collect();
        let at = first.iter().zip(other).position(|(a, b)| a != b).unwrap_or(0);
        let (frame, offset) = (at / len, at % len);
        let plane = case.width * case.height * 4;
        let pixel = if case.depth == 32 { (offset % plane) / 4 } else { offset / (len / (case.width * case.height)) };
        format!(
            "{} of {} frames differ ({differing_bytes} bytes), the first is frame {frame}, at x={} y={}",
            frames.len(),
            first.len() / len,
            pixel % case.width,
            pixel / case.width
        )
    };
    Some(Difference { against: against.to_string(), detail })
}

/// Runs `run` for the default thread count twice and for each of `threads`, and compares
/// every output with the first.
pub fn check_case(case: &Case, threads: &[usize], run: &dyn Fn(Threads) -> Result<Vec<u8>, String>) -> CaseCheck {
    let mut check = CaseCheck { case: case.clone(), runs: 0, error: None, differences: Vec::new() };
    let go = |how: Threads, check: &mut CaseCheck| -> Option<Vec<u8>> {
        match run(how) {
            Ok(bytes) => {
                check.runs += 1;
                Some(bytes)
            }
            Err(e) => {
                check.error = Some(e);
                None
            }
        }
    };
    let Some(first) = go(Threads::Default, &mut check) else { return check };
    let compare = |how: Threads, against: String, check: &mut CaseCheck| {
        let Some(other) = go(how, check) else { return false };
        check.differences.extend(describe_difference(case, &against, &first, &other));
        true
    };
    if !compare(Threads::Default, "repeat run".to_string(), &mut check) {
        return check;
    }
    for &n in threads {
        if !compare(Threads::Fixed(n), format!("RAYON_NUM_THREADS={n}"), &mut check) {
            break;
        }
    }
    check
}

pub fn run(opts: &SelfcheckOptions) -> Result<Vec<CaseCheck>, String> {
    let set = load_testset(&opts.testset)?;
    let picked = pick(&set.cases, opts.effect.as_deref(), opts.max_cases);
    if picked.is_empty() {
        return Err(match &opts.effect {
            Some(e) => format!("no temporal cases for effect `{e}` in {}", opts.testset.display()),
            None => format!("{} has no temporal cases", opts.testset.display()),
        });
    }
    let vfx = Vfx::locate(opts.vfx.as_deref())?;
    let slots: Vec<Mutex<Option<CaseCheck>>> = picked.iter().map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..opts.jobs.clamp(1, picked.len()) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= picked.len() {
                    break;
                }
                let case = &picked[i];
                let result = check_case(case, &opts.threads, &|how| vfx.apply_clip(case, &opts.testset, how));
                *slots[i].lock().unwrap() = Some(result);
            });
        }
    });
    Ok(slots.into_iter().map(|s| s.into_inner().unwrap().expect("every case ran")).collect())
}

pub fn all_ok(checks: &[CaseCheck]) -> bool {
    checks.iter().all(CaseCheck::ok)
}

pub fn write_human(checks: &[CaseCheck], out: &mut impl Write) -> std::io::Result<()> {
    const PAD: &str = "      ";
    for c in checks {
        let word = if c.ok() { "ok" } else { "DIFF" };
        writeln!(out, "{word:<5} {}  ({} runs)", c.case.id, c.runs)?;
        if let Some(e) = &c.error {
            writeln!(out, "{PAD}{e}")?;
        }
        for d in &c.differences {
            writeln!(out, "{PAD}{}: {}", d.against, d.detail)?;
        }
    }
    let bad = checks.iter().filter(|c| !c.ok()).count();
    writeln!(out, "\n{} cases checked, {} identical, {} DIFFER", checks.len(), checks.len() - bad, bad)
}

pub fn to_json(checks: &[CaseCheck]) -> Value {
    let bad = checks.iter().filter(|c| !c.ok()).count();
    json!({
        "summary": { "checked": checks.len(), "identical": checks.len() - bad, "differ": bad },
        "cases": checks.iter().map(|c| json!({
            "id": c.case.id,
            "effect": c.case.effect,
            "label": c.case.label,
            "ok": c.ok(),
            "runs": c.runs,
            "error": c.error,
            "differences": c.differences.iter().map(|d| json!({ "against": d.against, "detail": d.detail })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cases::CaseKind;
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    fn case(effect: &str, label: &str, depth: u32) -> Case {
        Case {
            id: format!("{effect}.{label}.d{depth}"),
            effect: effect.into(),
            label: label.into(),
            params: vec![],
            depth,
            width: 3,
            height: 2,
            source: "testsrc2".into(),
            input: "i".into(),
            reference: "r".into(),
            kind: CaseKind::Temporal,
            frames: 4,
            fps: Some("24".into()),
            layers: BTreeMap::new(),
            family: None,
        }
    }

    fn still(effect: &str) -> Case {
        Case { kind: CaseKind::Frame, frames: 1, fps: None, ..case(effect, "all-default", 8) }
    }

    #[test]
    fn pick_takes_everything_when_it_fits_and_only_temporal_cases() {
        let cases: Vec<Case> = (0..5).map(|i| case("a", &format!("l{i}"), 8)).chain([still("a"), still("b")]).collect();
        let picked = pick(&cases, None, 24);
        assert_eq!(picked.len(), 5);
        assert!(picked.iter().all(Case::is_temporal));
        assert_eq!(pick(&cases, None, 0).len(), 5);
        assert_eq!(pick(&cases, Some("b"), 24).len(), 0);
        assert_eq!(pick(&cases, Some("a"), 24).len(), 5);
    }

    #[test]
    fn pick_spreads_evenly_and_gives_every_effect_a_share() {
        let mut cases: Vec<Case> = (0..30).map(|i| case("a", &format!("l{i}"), 8)).collect();
        cases.extend((0..30).map(|i| case("b", &format!("l{i}"), 8)));
        cases.extend((0..3).map(|i| case("c", &format!("l{i}"), 8)));
        let picked = pick(&cases, None, 10);
        let of = |e: &str| picked.iter().filter(|c| c.effect == e).map(|c| c.label.clone()).collect::<Vec<_>>();
        // 10 split three ways: 4, 3, 3
        assert_eq!((of("a").len(), of("b").len(), of("c").len()), (4, 3, 3));
        assert_eq!(of("a"), ["l0", "l7", "l15", "l22"]);
        assert_eq!(of("c"), ["l0", "l1", "l2"]);
        // a tiny budget still visits every effect
        let tiny = pick(&cases, None, 1);
        assert_eq!(tiny.iter().map(|c| c.effect.as_str()).collect::<Vec<_>>(), ["a", "b", "c"]);
        assert!(tiny.iter().all(|c| c.label == "l0"));
        assert_eq!(pick(&cases, None, 10), picked, "deterministic");
    }

    #[test]
    fn identical_outputs_have_no_difference_and_a_changed_byte_is_located() {
        let c = case("a", "x", 8);
        let a = vec![7u8; 4 * c.frame_len()];
        assert_eq!(describe_difference(&c, "repeat run", &a, &a), None);
        // frame 2, pixel 4 (x=1 y=1), second channel
        let mut b = a.clone();
        b[2 * c.frame_len() + 4 * 4 + 1] ^= 1;
        let d = describe_difference(&c, "repeat run", &a, &b).unwrap();
        assert_eq!(d.against, "repeat run");
        assert_eq!(d.detail, "1 of 4 frames differ (1 bytes), the first is frame 2, at x=1 y=1");
        // two frames, many bytes
        let mut e = a.clone();
        e[c.frame_len()..c.frame_len() + 5].fill(0);
        e[3 * c.frame_len() + 23] = 0;
        let d = describe_difference(&c, "RAYON_NUM_THREADS=1", &a, &e).unwrap();
        assert_eq!(d.detail, "2 of 4 frames differ (6 bytes), the first is frame 1, at x=0 y=0");
        let short = describe_difference(&c, "repeat run", &a, &a[..10]).unwrap();
        assert_eq!(short.detail, format!("the output has 10 bytes against {}", a.len()));
    }

    #[test]
    fn float_differences_are_located_in_the_planar_layout() {
        let c = case("a", "x", 32);
        let a = vec![0u8; 2 * c.frame_len()];
        // frame 1, plane 2 (R), pixel 5 (x=2 y=1)
        let mut b = a.clone();
        b[c.frame_len() + (2 * 6 + 5) * 4 + 3] = 9;
        let d = describe_difference(&c, "repeat run", &a, &b).unwrap();
        assert_eq!(d.detail, "1 of 2 frames differ (1 bytes), the first is frame 1, at x=2 y=1");
        // 16 bpc has 8 bytes a pixel
        let c16 = case("a", "x", 16);
        let a = vec![0u8; c16.frame_len()];
        let mut b = a.clone();
        b[8 * 4 + 7] = 1;
        assert!(describe_difference(&c16, "repeat run", &a, &b).unwrap().detail.ends_with("at x=1 y=1"));
    }

    /// A run closure that hands out the given outputs in order and records how it was called.
    fn scripted(outputs: Vec<Result<Vec<u8>, String>>, calls: &RefCell<Vec<Threads>>) -> impl Fn(Threads) -> Result<Vec<u8>, String> + '_ {
        let outputs = RefCell::new(outputs.into_iter());
        move |how| {
            calls.borrow_mut().push(how);
            outputs.borrow_mut().next().expect("no more runs than scripted")
        }
    }

    #[test]
    fn a_case_runs_twice_by_default_and_once_per_thread_count() {
        let c = case("a", "x", 8);
        let same = vec![1u8; 24];
        let calls = RefCell::new(Vec::new());
        let run = scripted(vec![Ok(same.clone()); 4], &calls);
        let check = check_case(&c, &[1, 3], &run);
        assert!(check.ok());
        assert_eq!(check.runs, 4);
        assert_eq!(*calls.borrow(), [Threads::Default, Threads::Default, Threads::Fixed(1), Threads::Fixed(3)]);
    }

    #[test]
    fn nondeterminism_and_thread_dependence_are_flagged_separately() {
        let c = case("a", "x", 8);
        let good = vec![1u8; 4 * c.frame_len()];
        let mut bad = good.clone();
        bad[3] = 2;
        let calls = RefCell::new(Vec::new());
        // the repeat run differs, the 1 thread run does not
        let check = check_case(&c, &[1], &scripted(vec![Ok(good.clone()), Ok(bad.clone()), Ok(good.clone())], &calls));
        assert!(!check.ok());
        assert_eq!(check.differences.iter().map(|d| d.against.as_str()).collect::<Vec<_>>(), ["repeat run"]);
        // the repeat is fine, the 3 thread run differs
        let calls = RefCell::new(Vec::new());
        let check = check_case(&c, &[1, 3], &scripted(vec![Ok(good.clone()), Ok(good.clone()), Ok(good.clone()), Ok(bad.clone())], &calls));
        assert_eq!(check.differences.iter().map(|d| d.against.as_str()).collect::<Vec<_>>(), ["RAYON_NUM_THREADS=3"]);
        assert!(check.differences[0].detail.contains("frame 0"));
        // both differ
        let calls = RefCell::new(Vec::new());
        let check = check_case(&c, &[1], &scripted(vec![Ok(good.clone()), Ok(bad.clone()), Ok(bad)], &calls));
        assert_eq!(check.differences.len(), 2);
    }

    #[test]
    fn a_failing_run_is_an_error_and_stops_the_case() {
        let c = case("a", "x", 8);
        let calls = RefCell::new(Vec::new());
        let check = check_case(&c, &[1], &scripted(vec![Ok(vec![0; 24]), Err("vfx exited with code 3: boom".into())], &calls));
        assert!(!check.ok());
        assert_eq!((check.runs, check.error.as_deref()), (1, Some("vfx exited with code 3: boom")));
        assert_eq!(calls.borrow().len(), 2);
        let calls = RefCell::new(Vec::new());
        let check = check_case(&c, &[1], &scripted(vec![Err("no vfx".into())], &calls));
        assert_eq!((check.runs, check.error.as_deref()), (0, Some("no vfx")));
    }

    #[test]
    fn the_reports_say_which_comparison_failed() {
        let c = case("a", "x", 8);
        let good = vec![1u8; 4 * c.frame_len()];
        let mut bad = good.clone();
        bad[0] = 9;
        let calls = RefCell::new(Vec::new());
        let differing = check_case(&c, &[1], &scripted(vec![Ok(good.clone()), Ok(good.clone()), Ok(bad)], &calls));
        let calls = RefCell::new(Vec::new());
        let fine = check_case(&case("a", "y", 8), &[1], &scripted(vec![Ok(good.clone()), Ok(good.clone()), Ok(good)], &calls));
        let checks = [fine, differing];
        assert!(!all_ok(&checks));
        let mut buf = Vec::new();
        write_human(&checks, &mut buf).unwrap();
        let text = String::from_utf8(buf).unwrap();
        assert!(text.contains("ok    a.y.d8  (3 runs)"), "{text}");
        assert!(text.contains("DIFF  a.x.d8  (3 runs)"), "{text}");
        assert!(text.contains("RAYON_NUM_THREADS=1: 1 of 4 frames differ (1 bytes), the first is frame 0, at x=0 y=0"), "{text}");
        assert!(text.contains("2 cases checked, 1 identical, 1 DIFFER"), "{text}");
        let v = to_json(&checks);
        assert_eq!((v["summary"]["checked"].as_u64(), v["summary"]["differ"].as_u64()), (Some(2), Some(1)));
        assert_eq!(v["cases"][1]["differences"][0]["against"], "RAYON_NUM_THREADS=1");
        assert_eq!(v["cases"][0]["ok"], true);
        assert!(all_ok(&checks[..1]));
    }
}
