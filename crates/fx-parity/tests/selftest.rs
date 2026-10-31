//! Self test of the harness: references made by `vfx` itself must all pass, and a deliberately
//! altered reference must be flagged with the right frame and pixel count. The tests that need
//! ffmpeg skip themselves when it is not installed.
//!
//! The frame tests hold for any effect, whatever its math and params; the temporal ones use the
//! hidden `selftest-*` effects of `vfx`. Sizes stay small and the sweeps are narrowed so the run
//! stays short however many params the effects have.

use fx_parity::cases::{self, Case, CaseKind, InputFile, Plan, TemporalPlan, TestSet};
use fx_parity::checklist;
use fx_parity::effects::{EffectInfo, EffectKind};
use fx_parity::families;
use fx_parity::run::{self, RunOptions, Status};
use fx_parity::selfcheck::{self, SelfcheckOptions};
use fx_parity::sources;
use fx_parity::testgen::{ffmpeg_path, ffmpeg_version, render_input};
use fx_parity::vfx::{Threads, Vfx};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Once, OnceLock};

const JOBS: usize = 8;

fn exe_name() -> &'static str {
    if cfg!(windows) {
        "vfx.exe"
    } else {
        "vfx"
    }
}

/// The vfx binary next to this test's profile directory, rebuilt if stale.
fn vfx() -> &'static Vfx {
    static VFX: OnceLock<Vfx> = OnceLock::new();
    VFX.get_or_init(|| {
        let exe = std::env::current_exe().unwrap();
        let profile_dir = exe.parent().and_then(Path::parent).expect("target/<profile>/deps").to_path_buf();
        let target_dir = profile_dir.parent().expect("target dir");
        let profile = profile_dir.file_name().and_then(|n| n.to_str()).expect("profile name");
        let mut build = Command::new(option_env!("CARGO").unwrap_or("cargo"));
        build.current_dir(env!("CARGO_MANIFEST_DIR")).args(["build", "-q", "-p", "fx-cli", "--target-dir"]).arg(target_dir);
        match profile {
            "debug" => {}
            "release" => {
                build.arg("--release");
            }
            other => {
                build.args(["--profile", other]);
            }
        }
        let status = build.status().expect("run cargo to build vfx");
        assert!(status.success(), "cargo build -p fx-cli failed");
        let path = profile_dir.join(exe_name());
        assert!(path.is_file(), "{} was not built", path.display());
        Vfx::at(path)
    })
}

/// The effects `vfx list` shows by default.
fn effects() -> &'static Vec<EffectInfo> {
    static EFFECTS: OnceLock<Vec<EffectInfo>> = OnceLock::new();
    EFFECTS.get_or_init(|| vfx().list().expect("vfx list"))
}

/// Every effect, the hidden self-test ones included.
fn all_effects() -> &'static Vec<EffectInfo> {
    static EFFECTS: OnceLock<Vec<EffectInfo>> = OnceLock::new();
    EFFECTS.get_or_init(|| vfx().list_all().expect("vfx list --all"))
}

fn effect(id: &str) -> EffectInfo {
    all_effects().iter().find(|e| e.id == id).unwrap_or_else(|| panic!("vfx has no effect {id}")).clone()
}

/// The frame effects whatever the plugins turn into: the listed ones that are frame
/// effects, and the hidden `selftest-invert`.
fn frame_effects() -> Vec<EffectInfo> {
    all_effects().iter().filter(|e| e.kind == EffectKind::Frame && (!e.hidden || e.id == "selftest-invert")).cloned().collect()
}

/// A fresh directory under the target dir. Directories left by earlier runs, which
/// are named after their process id, are swept the first time through.
fn scratch(name: &str) -> PathBuf {
    static SWEEP: Once = Once::new();
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    SWEEP.call_once(|| {
        let old = std::time::Duration::from_secs(600);
        for entry in std::fs::read_dir(&root).into_iter().flatten().flatten() {
            let stale = entry.metadata().and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).is_some_and(|age| age > old);
            if entry.file_name().to_string_lossy().starts_with("parity-") && stale {
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    });
    let dir = root.join(format!("parity-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Runs `f` on every item, `JOBS` at a time.
fn parallel<T: Sync>(items: &[T], f: impl Fn(&T) + Sync) {
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..JOBS.min(items.len().max(1)) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                match items.get(i) {
                    Some(item) => f(item),
                    None => break,
                }
            });
        }
    });
}

/// Deterministic, finite, in range test frames in the wire format of each depth.
fn synth_frame(depth: u32, w: usize, h: usize, seed: u64) -> Vec<u8> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut next = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let n = w * h;
    match depth {
        8 => (0..n * 4).map(|_| next() as u8).collect(),
        16 => (0..n * 4).flat_map(|_| (next() as u16).to_le_bytes()).collect(),
        _ => (0..n * 4).flat_map(|_| ((next() % 10_000) as f32 / 10_000.0).to_le_bytes()).collect(),
    }
}

/// A still, or a clip of `f.frames` such frames.
fn synth_input(f: &InputFile, seed: u64) -> Vec<u8> {
    (0..f.frames).flat_map(|frame| synth_frame(f.depth, f.width, f.height, seed * 1000 + frame as u64 + 1)).collect()
}

struct Set {
    dir: PathBuf,
    refs: PathBuf,
    cases: Vec<Case>,
}

impl Set {
    fn options(&self) -> RunOptions {
        RunOptions { testset: self.dir.clone(), refs: self.refs.clone(), effect: None, vfx: Some(vfx().path.clone()), jobs: JOBS, max_ulp: 1 }
    }

    fn input(&self, c: &Case) -> Vec<u8> {
        std::fs::read(self.dir.join(&c.input)).unwrap()
    }

    fn ref_path(&self, c: &Case) -> PathBuf {
        self.refs.join(&c.reference)
    }

    fn find(&self, label: &str, depth: u32, size: (usize, usize)) -> &Case {
        self.cases
            .iter()
            .find(|c| c.label == label && c.depth == depth && (c.width, c.height) == size)
            .unwrap_or_else(|| panic!("no {label} case at {size:?} {depth} bpc"))
    }

    fn find_clip(&self, effect: &str, label: &str, source: &str, depth: u32, size: (usize, usize)) -> &Case {
        self.cases
            .iter()
            .find(|c| c.effect == effect && c.label == label && c.source == source && c.depth == depth && (c.width, c.height) == size)
            .unwrap_or_else(|| panic!("no {effect} {label} case on {source} at {size:?} {depth} bpc"))
    }

    fn check(&self, c: &Case) -> run::CaseResult {
        run::run_case(vfx(), &self.dir, &self.refs, c)
    }

    /// Same inputs, private copy of the references, free to damage.
    fn fork(&self, name: &str) -> Set {
        let refs = scratch(name);
        for entry in std::fs::read_dir(&self.refs).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(entry.path(), refs.join(entry.file_name())).unwrap();
        }
        Set { dir: self.dir.clone(), refs, cases: self.cases.clone() }
    }
}

/// Writes inputs, `cases.json` and a reference per case taken from vfx's own output.
fn build_set(name: &str, plan: &Plan, effects: &[EffectInfo]) -> Set {
    let dir = scratch(name);
    let refs = dir.join("refs");
    std::fs::create_dir_all(dir.join("inputs")).unwrap();
    std::fs::create_dir_all(&refs).unwrap();
    let built = cases::build_all(effects, plan);
    let mut inputs = if effects.iter().any(|e| e.kind == EffectKind::Frame) { plan.inputs() } else { Vec::new() };
    inputs.extend(built.clip_inputs.iter().cloned());
    for (i, f) in inputs.iter().enumerate() {
        std::fs::write(dir.join(&f.file), synth_input(f, i as u64 + 1)).unwrap();
    }
    let set = TestSet { format: cases::FORMAT, ffmpeg: "synthetic".into(), inputs, cases: built.cases.clone() };
    std::fs::write(dir.join("cases.json"), serde_json::to_string_pretty(&set).unwrap()).unwrap();
    let set = Set { dir, refs, cases: built.cases };
    parallel(&set.cases, |c| {
        let out = if c.is_temporal() {
            vfx().apply_clip(c, &set.dir, Threads::Inherit)
        } else {
            vfx().apply(c, &set.input(c))
        };
        std::fs::write(set.ref_path(c), out.unwrap_or_else(|e| panic!("{}: {e}", c.id))).unwrap();
    });
    set
}

/// A small set for every frame effect: all-default, all-min and all-max at two sizes and every depth.
fn base() -> &'static Set {
    static BASE: OnceLock<Set> = OnceLock::new();
    BASE.get_or_init(|| {
        let noise = sources::find("noise").unwrap();
        let plan = Plan { sizes: vec![(1, 1), (5, 4)], sources: vec![noise], depths: vec![8, 16, 32], sweep: false, sweep_sources: vec![noise], ..Plan::default() };
        build_set("base", &plan, &frame_effects())
    })
}

/// Small clips for the temporal self-test effects: 6 frames at 5x4 and 3x2, every depth, every clip source.
fn clips() -> &'static Set {
    static CLIPS: OnceLock<Set> = OnceLock::new();
    CLIPS.get_or_init(|| {
        let plan = Plan {
            temporal: TemporalPlan { sizes: vec![(5, 4), (3, 2)], frames: 6, ..TemporalPlan::default() },
            ..Plan::default()
        };
        build_set("clips", &plan, &[effect("selftest-shift"), effect("selftest-integrate")])
    })
}

#[test]
fn the_self_test_effects_are_hidden_from_the_plain_list_and_typed_right() {
    assert!(effects().iter().all(|e| !e.hidden && !e.id.starts_with("selftest-")));
    assert!(!effects().is_empty());
    for id in ["selftest-invert", "selftest-shift", "selftest-integrate", "selftest-threads"] {
        let e = effect(id);
        assert!(e.hidden, "{id}");
        assert_eq!(e.kind, if id == "selftest-invert" { EffectKind::Frame } else { EffectKind::Temporal }, "{id}");
    }
    let shift = effect("selftest-shift");
    assert_eq!(shift.layers.len(), 1);
    assert_eq!((shift.layers[0].id.as_str(), shift.layers[0].optional), ("overlay", true));
    assert!(effect("selftest-integrate").layers.is_empty());
    // all_effects holds the listed ones too
    assert!(effects().iter().all(|e| all_effects().iter().any(|a| a.id == e.id)));
}

#[test]
fn references_made_by_vfx_itself_all_pass() {
    let set = base();
    assert!(!set.cases.is_empty());
    let results = run::run(&set.options(), &|_| {}).unwrap();
    assert_eq!(results.len(), set.cases.len());
    let t = run::totals(&results);
    let first_bad = results.iter().find(|r| r.status != Status::Pass).map(|r| (&r.case.id, &r.error));
    assert_eq!((t.pass, t.fail, t.no_reference), (set.cases.len(), 0, 0), "{first_bad:?}");
    // the report lists them in manifest order
    let ids: Vec<&str> = results.iter().map(|r| r.case.id.as_str()).collect();
    let want: Vec<&str> = set.cases.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(ids, want);
}

#[test]
fn a_corrupted_8_bit_reference_is_flagged_with_the_right_pixel_count() {
    let set = base().fork("dmg8");
    let c = set.find("all-default", 8, (5, 4)).clone();
    let mut bytes = std::fs::read(set.ref_path(&c)).unwrap();
    // three pixels: pixel 3 loses one level on two channels, pixel 7 and pixel 19 on one
    for off in [3 * 4, 3 * 4 + 2, 7 * 4 + 1, 19 * 4 + 3] {
        bytes[off] = bytes[off].wrapping_add(1);
    }
    std::fs::write(set.ref_path(&c), bytes).unwrap();

    // its effect's cases: exactly the altered one fails
    let mut opts = set.options();
    opts.effect = Some(c.effect.clone());
    let results = run::run(&opts, &|_| {}).unwrap();
    let t = run::totals(&results);
    assert_eq!((t.fail, t.pass), (1, results.len() - 1));
    let bad = results.iter().find(|r| r.status == Status::Fail).unwrap();
    assert_eq!(bad.case.id, c.id);
    let cmp = bad.comparison.as_ref().unwrap();
    assert_eq!(cmp.differing_pixels, 3);
    assert_eq!(cmp.max_abs_diff, 1.0);
    let at: Vec<(usize, usize)> = cmp.first.iter().map(|d| (d.x, d.y)).collect();
    assert_eq!(at, [(3, 0), (2, 1), (4, 3)]);
    assert_eq!((bad.frame, bad.failing_frames), (None, 0), "a still has no frames to name");
    // four of the 20 * 4 channel values are one level off
    let want = 4.0 / (20.0 * 4.0 * 255.0);
    assert!((cmp.mean_abs_diff - want).abs() < 1e-12 && bad.mean_abs_diff == Some(cmp.mean_abs_diff), "{}", cmp.mean_abs_diff);
    assert_eq!(bad.worst_frame, None);

    // the graded block ranks the damaged case first and counts the others as exact
    let mut text = Vec::new();
    run::write_human(&results, true, &mut text).unwrap();
    let text = String::from_utf8(text).unwrap();
    let block = text.split("\ngraded error\n").nth(1).expect("the graded block follows the case lines");
    let line = block.lines().next().unwrap();
    assert!(line.contains(&format!("{} compared, {} bit exact", results.len(), results.len() - 1)), "{line}");
    assert!(line.contains(&format!("worst {} (", c.id)), "{line}");
    assert!(block.contains("\n1 worst case by mean error\n") && block.lines().any(|l| l.ends_with(&c.id)), "{block}");
    let json = run::to_json(&results, true);
    let g = &json["graded"][c.effect.as_str()];
    assert_eq!(g["compared"].as_u64(), Some(results.len() as u64));
    assert_eq!(g["exact"].as_u64(), Some(results.len() as u64 - 1));
    assert_eq!(g["worst"]["id"], c.id.as_str());
    assert!((g["worst"]["mean_abs_diff"].as_f64().unwrap() - want).abs() < 1e-12);
}

#[test]
fn a_corrupted_16_bit_reference_is_flagged_and_the_gap_is_reported() {
    let set = base().fork("dmg16");
    let c = set.find("all-default", 16, (5, 4)).clone();
    let mut bytes = std::fs::read(set.ref_path(&c)).unwrap();
    let mut pixel = |index: usize, channel: usize, delta: i32| {
        let o = index * 8 + channel * 2;
        let v = u16::from_le_bytes([bytes[o], bytes[o + 1]]) as i32;
        let nv = (if v + delta > 65535 { v - delta } else { v + delta }) as u16;
        bytes[o..o + 2].copy_from_slice(&nv.to_le_bytes());
    };
    pixel(1, 0, 1);
    pixel(4, 3, 40);
    std::fs::write(set.ref_path(&c), bytes).unwrap();

    let r = set.check(&c);
    assert_eq!(r.status, Status::Fail);
    let cmp = r.comparison.unwrap();
    assert_eq!(cmp.differing_pixels, 2);
    assert_eq!(cmp.max_abs_diff, 40.0);
    assert_eq!(cmp.first.len(), 2);
}

/// Moves the first finite, non-zero value outside `not_pixel` by `ulps` and returns its pixel.
/// Planes are G, B, R, A, `n` values each.
fn bump_a_value(bytes: &mut [u8], n: usize, ulps: u32, not_pixel: Option<usize>) -> usize {
    for pixel in (0..n).filter(|p| Some(*p) != not_pixel) {
        for plane in 0..4 {
            let o = (plane * n + pixel) * 4;
            let bits = u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
            let v = f32::from_bits(bits);
            if v.is_finite() && v != 0.0 && v.abs() < 1e30 {
                bytes[o..o + 4].copy_from_slice(&(bits + ulps).to_le_bytes());
                return pixel;
            }
        }
    }
    panic!("no finite non-zero value to bump");
}

#[test]
fn float_references_tolerate_one_ulp_and_flag_two() {
    let set = base().fork("dmg32");
    let c = set.find("all-default", 32, (5, 4)).clone();
    let mut bytes = std::fs::read(set.ref_path(&c)).unwrap();
    let within = bump_a_value(&mut bytes, 5 * 4, 1, None);
    std::fs::write(set.ref_path(&c), &bytes).unwrap();
    assert_eq!(set.check(&c).status, Status::Pass, "one ULP is inside the tolerance");

    let beyond = bump_a_value(&mut bytes, 5 * 4, 2, Some(within));
    std::fs::write(set.ref_path(&c), bytes).unwrap();
    let r = set.check(&c);
    assert_eq!(r.status, Status::Fail, "two ULPs are not");
    let cmp = r.comparison.unwrap();
    assert_eq!(cmp.differing_pixels, 1);
    assert_eq!(cmp.max_ulp, Some(2));
    assert_eq!((cmp.first[0].x, cmp.first[0].y), (beyond % 5, beyond / 5));
}

#[test]
fn max_ulp_zero_is_bit_for_bit_and_wider_windows_widen() {
    let set = base().fork("ulp0");
    let c = set.find("all-default", 32, (5, 4)).clone();
    let check = |max_ulp| run::run_case_within(vfx(), &set.dir, &set.refs, &c, max_ulp);
    assert_eq!(check(0).status, Status::Pass, "an exact reference is bit for bit");
    let mut bytes = std::fs::read(set.ref_path(&c)).unwrap();
    let pixel = bump_a_value(&mut bytes, 5 * 4, 1, None);
    std::fs::write(set.ref_path(&c), &bytes).unwrap();
    assert_eq!(check(1).status, Status::Pass);
    let r = check(0);
    assert_eq!(r.status, Status::Fail, "one ULP is not bit for bit");
    let cmp = r.comparison.unwrap();
    assert_eq!((cmp.differing_pixels, cmp.max_ulp), (1, Some(1)));
    assert_eq!((cmp.first[0].x, cmp.first[0].y), (pixel % 5, pixel / 5));
    bump_a_value(&mut bytes, 5 * 4, 2, Some(pixel));
    std::fs::write(set.ref_path(&c), &bytes).unwrap();
    assert_eq!(check(1).status, Status::Fail, "a value 2 ULP away is outside 1");
    assert_eq!(check(2).status, Status::Pass);

    // a signed zero is the same value at any window but different bits at 0
    let zero = base().fork("ulp0zero");
    let mut bytes = std::fs::read(zero.ref_path(&c)).unwrap();
    let n = 5 * 4;
    let at = (0..n * 4).find(|i| bytes[i * 4..i * 4 + 4] == 0f32.to_le_bytes()).map(|i| i * 4);
    if let Some(o) = at {
        bytes[o..o + 4].copy_from_slice(&(-0.0f32).to_le_bytes());
        std::fs::write(zero.ref_path(&c), &bytes).unwrap();
        let strict = run::run_case_within(vfx(), &zero.dir, &zero.refs, &c, 0);
        assert_eq!(strict.status, Status::Fail, "-0.0 against 0.0 differs bit for bit");
        assert_eq!(strict.comparison.unwrap().differing_pixels, 1);
        assert_eq!(zero.check(&c).status, Status::Pass, "and passes within 1 ULP");
    }
}

#[test]
fn max_ulp_zero_counts_clip_pixels_over_all_frames() {
    let set = clips().fork("tulp0");
    let c = integrate_case(&set, 32);
    let len = c.frame_len();
    let mut bytes = std::fs::read(set.ref_path(&c)).unwrap();
    bump_a_value(&mut bytes[len..2 * len], 5 * 4, 1, None);
    bump_a_value(&mut bytes[3 * len..4 * len], 5 * 4, 1, None);
    std::fs::write(set.ref_path(&c), &bytes).unwrap();
    assert_eq!(set.check(&c).status, Status::Pass);
    let r = run::run_case_within(vfx(), &set.dir, &set.refs, &c, 0);
    assert_eq!((r.status, r.frame, r.failing_frames, r.differing_pixels), (Status::Fail, Some(1), 2, 2));
    let json = run::to_json(&[r], false);
    assert_eq!(json["cases"][0]["clip_differing_pixels"].as_u64(), Some(2));
    assert_eq!(json["cases"][0]["max_ulp"].as_u64(), Some(1));
}

#[test]
fn missing_wrong_sized_and_failing_cases_are_never_passes() {
    let set = base().fork("odd");
    let missing = set.find("all-default", 8, (1, 1)).clone();
    std::fs::remove_file(set.ref_path(&missing)).unwrap();
    let short = set.find("all-default", 16, (1, 1)).clone();
    let bytes = std::fs::read(set.ref_path(&short)).unwrap();
    std::fs::write(set.ref_path(&short), &bytes[..bytes.len() - 1]).unwrap();

    let r = set.check(&missing);
    assert_eq!((r.status, r.comparison.is_none()), (Status::NoReference, true));
    let r = set.check(&short);
    assert_eq!(r.status, Status::Fail);
    assert!(r.error.as_deref().unwrap().contains("bytes"), "{:?}", r.error);

    // a case vfx itself rejects is a failure carrying vfx's own message
    let mut broken = set.find("all-default", 8, (5, 4)).clone();
    broken.params = vec!["no_such_param=1".into()];
    let r = set.check(&broken);
    assert_eq!(r.status, Status::Fail);
    assert!(r.error.as_deref().unwrap().contains("exited with code 3"), "{:?}", r.error);
    assert!(r.error.as_deref().unwrap().contains("no_such_param"));

    // and in a whole run the two odd cases are counted, not passed
    let mut opts = set.options();
    opts.effect = Some(missing.effect.clone());
    let results = run::run(&opts, &|_| {}).unwrap();
    let t = run::totals(&results);
    assert_eq!((t.no_reference, t.fail, t.pass), (1, 1, results.len() - 2));
}

#[test]
fn the_effect_filter_selects_cases_and_unknown_effects_are_errors() {
    let set = base();
    let first = set.cases[0].effect.clone();
    let mut opts = set.options();
    opts.effect = Some(first.clone());
    let results = run::run(&opts, &|_| {}).unwrap();
    assert!(!results.is_empty() && results.iter().all(|r| r.case.effect == first));
    opts.effect = Some("no-such-effect".into());
    assert!(run::run(&opts, &|_| {}).unwrap_err().contains("no cases for effect"));
}

#[test]
fn every_generated_param_value_is_accepted_by_vfx() {
    // all the sweeps and ramps, one source and depth, a frame size that never gets asked for pixels
    let noise = sources::find("noise").unwrap();
    let clip = sources::find_clip("movegrad").unwrap();
    let plan = Plan {
        sizes: vec![(8, 8)],
        sources: vec![noise],
        depths: vec![8],
        sweep: true,
        sweep_sources: vec![noise],
        temporal: TemporalPlan { sizes: vec![(8, 8)], sources: vec![clip], sweep_sources: vec![clip], ..TemporalPlan::default() },
    };
    let cases = cases::build(all_effects(), &plan);
    let params: usize = all_effects().iter().map(|e| e.params.len()).sum();
    assert!(cases.len() >= all_effects().len() + params, "{} cases for {params} params", cases.len());
    assert!(cases.iter().any(|c| c.is_temporal() && c.params.iter().any(|p| p.contains('@'))), "the keyframed ramps are in the plan");
    parallel(&cases, |c| {
        // no frames in, so this only exercises the effect id, every --param and the layers
        let out = if c.is_temporal() { vfx().check(c).map(|_| Vec::new()) } else { vfx().apply(c, &[]) };
        assert!(out.is_ok(), "{} {:?}: {:?}", c.effect, c.params, out.err());
    });
}

// ---- temporal cases ------------------------------------------------------------------------

#[test]
fn temporal_references_made_by_vfx_itself_all_pass() {
    let set = clips();
    assert!(set.cases.len() > 150, "{} temporal cases", set.cases.len());
    assert!(set.cases.iter().all(|c| c.kind == CaseKind::Temporal && c.frames == 6));
    assert!(set.cases.iter().any(|c| c.params.iter().any(|p| p.contains("@0;"))), "ramps are part of the set");
    assert!(set.cases.iter().any(|c| c.effect == "selftest-shift" && c.layers.contains_key("overlay")));
    let results = run::run(&set.options(), &|_| {}).unwrap();
    assert_eq!(results.len(), set.cases.len());
    let t = run::totals(&results);
    let first_bad = results.iter().find(|r| r.status != Status::Pass).map(|r| (&r.case.id, &r.error));
    assert_eq!((t.pass, t.fail, t.no_reference), (set.cases.len(), 0, 0), "{first_bad:?}");
}

/// The 8 bit integrate clip, all params at their default, with every pixel of every frame known to the test.
fn integrate_case(set: &Set, depth: u32) -> Case {
    set.find_clip("selftest-integrate", "all-default", "testsrc2", depth, (5, 4)).clone()
}

#[test]
fn a_corrupted_8_bit_clip_is_flagged_with_its_first_frame_and_pixel_count() {
    let set = clips().fork("tdmg8");
    let c = integrate_case(&set, 8);
    let len = c.frame_len();
    let mut bytes = std::fs::read(set.ref_path(&c)).unwrap();
    assert_eq!(bytes.len(), 6 * len);
    // frame 3: pixel 3 is off by one level on two channels, pixel 7 and pixel 19 on one; frame 5: pixel 0
    for off in [3 * len + 3 * 4, 3 * len + 3 * 4 + 2, 3 * len + 7 * 4 + 1, 3 * len + 19 * 4 + 3, 5 * len] {
        bytes[off] ^= 1;
    }
    std::fs::write(set.ref_path(&c), bytes).unwrap();

    let mut opts = set.options();
    opts.effect = Some(c.effect.clone());
    let results = run::run(&opts, &|_| {}).unwrap();
    let t = run::totals(&results);
    assert_eq!((t.fail, t.pass), (1, results.len() - 1));
    let bad = results.iter().find(|r| r.status == Status::Fail).unwrap();
    assert_eq!(bad.case.id, c.id);
    assert_eq!((bad.frame, bad.failing_frames), (Some(3), 2));
    // frame 3 has four channel values one level off, frame 5 has one; the other frames are equal
    let values = 5.0 * 4.0 * 4.0 * 255.0;
    let (worst, worst_mean) = bad.worst_frame.expect("a damaged clip has a worst frame");
    assert_eq!(worst, 3);
    assert!((worst_mean - 4.0 / values).abs() < 1e-12, "{worst_mean}");
    assert!((bad.mean_abs_diff.unwrap() - 5.0 / (6.0 * values)).abs() < 1e-12, "{:?}", bad.mean_abs_diff);
    let cmp = bad.comparison.as_ref().unwrap();
    assert_eq!((cmp.differing_pixels, cmp.max_abs_diff), (3, 1.0));
    let at: Vec<(usize, usize)> = cmp.first.iter().map(|d| (d.x, d.y)).collect();
    assert_eq!(at, [(3, 0), (2, 1), (4, 3)]);
    // the report names the frame, the rate, the params and a command that reruns the case
    let mut buf = Vec::new();
    run::write_human(&results, true, &mut buf).unwrap();
    let text = String::from_utf8(buf).unwrap();
    assert!(text.contains("first failing frame: 3 of 6 (2 frames fail)"), "{text}");
    assert!(text.contains("worst frame 3 ("), "{text}");
    assert!(text.contains("3 of 20 pixels differ"), "{text}");
    assert!(text.contains(&format!("clip: 6 frames at {} fps", c.fps.as_deref().unwrap())), "{text}");
    assert!(text.contains(&format!("vfx apply {} --width 5 --height 4 --depth 8", c.effect)) && text.contains("--frames 6"), "{text}");
    let json = run::to_json(&results, true);
    assert_eq!((json["cases"][0]["frame"].as_u64(), json["cases"][0]["failing_frames"].as_u64()), (Some(3), Some(2)));
    assert_eq!(json["cases"][0]["worst_frame"]["frame"], 3);
    assert!(json["cases"][0]["mean_abs_diff"].as_f64().unwrap() > 0.0);
    assert_eq!(json["cases"][0]["mismatches"][2]["x"], 4);
}

#[test]
fn a_corrupted_16_bit_clip_is_flagged_and_the_gap_is_reported() {
    let set = clips().fork("tdmg16");
    let c = integrate_case(&set, 16);
    let len = c.frame_len();
    let mut bytes = std::fs::read(set.ref_path(&c)).unwrap();
    let mut bump = |frame: usize, pixel: usize, channel: usize, delta: i32| {
        let o = frame * len + pixel * 8 + channel * 2;
        let v = u16::from_le_bytes([bytes[o], bytes[o + 1]]) as i32;
        let nv = (if v + delta > 65535 { v - delta } else { v + delta }) as u16;
        bytes[o..o + 2].copy_from_slice(&nv.to_le_bytes());
    };
    bump(2, 1, 0, 1);
    bump(4, 4, 3, 40);
    std::fs::write(set.ref_path(&c), bytes).unwrap();
    let r = set.check(&c);
    assert_eq!(r.status, Status::Fail);
    assert_eq!((r.frame, r.failing_frames), (Some(2), 2));
    let cmp = r.comparison.unwrap();
    assert_eq!((cmp.differing_pixels, cmp.max_abs_diff), (1, 1.0), "the first failing frame is the one reported");
    assert_eq!((cmp.first[0].x, cmp.first[0].y), (1, 0));
}

#[test]
fn float_clips_tolerate_one_ulp_per_frame_and_flag_two() {
    let set = clips().fork("tdmg32");
    let c = integrate_case(&set, 32);
    let len = c.frame_len();
    let mut bytes = std::fs::read(set.ref_path(&c)).unwrap();
    let within = bump_a_value(&mut bytes[len..2 * len], 5 * 4, 1, None);
    let _ = within;
    std::fs::write(set.ref_path(&c), &bytes).unwrap();
    let r = set.check(&c);
    assert_eq!(r.status, Status::Pass, "one ULP in frame 1 is inside the tolerance");
    assert_eq!((r.frame, r.failing_frames), (None, 0));
    assert_eq!(r.comparison.as_ref().unwrap().max_ulp, Some(1), "the gap is still reported");

    let beyond = bump_a_value(&mut bytes[4 * len..5 * len], 5 * 4, 2, None);
    std::fs::write(set.ref_path(&c), bytes).unwrap();
    let r = set.check(&c);
    assert_eq!(r.status, Status::Fail, "two ULPs in frame 4 are not");
    assert_eq!((r.frame, r.failing_frames), (Some(4), 1));
    let cmp = r.comparison.unwrap();
    assert_eq!((cmp.differing_pixels, cmp.max_ulp), (1, Some(2)));
    assert_eq!((cmp.first[0].x, cmp.first[0].y), (beyond % 5, beyond / 5));
}

#[test]
fn missing_short_and_failing_clips_are_never_passes() {
    let set = clips().fork("todd");
    let missing = integrate_case(&set, 8);
    std::fs::remove_file(set.ref_path(&missing)).unwrap();
    let short = integrate_case(&set, 16);
    let bytes = std::fs::read(set.ref_path(&short)).unwrap();
    // one frame short is the commonest mistake of a reference generator
    std::fs::write(set.ref_path(&short), &bytes[..bytes.len() - short.frame_len()]).unwrap();
    let r = set.check(&missing);
    assert_eq!((r.status, r.comparison.is_none()), (Status::NoReference, true));
    let r = set.check(&short);
    assert_eq!(r.status, Status::Fail);
    let e = r.error.as_deref().unwrap();
    assert!(e.contains("reference") && e.contains("6 frames of 5x4 at 16 bpc"), "{e}");

    let mut broken = integrate_case(&set, 32);
    broken.params = vec!["no_such_param=1".into()];
    let r = set.check(&broken);
    assert_eq!(r.status, Status::Fail);
    assert!(r.error.as_deref().unwrap().contains("exited with code 3") && r.error.as_deref().unwrap().contains("no_such_param"));

    // a layer file that is not there is vfx's io error, reported as the case's failure
    let mut layered = set.find_clip("selftest-shift", "all-default", "testsrc2", 8, (5, 4)).clone();
    layered.layers.insert("overlay".into(), "inputs/not-there".into());
    let r = set.check(&layered);
    assert_eq!(r.status, Status::Fail);
    assert!(r.error.as_deref().unwrap().contains("exited with code 4"), "{:?}", r.error);

    // an input of the wrong length is a setup mistake, named as such
    let mut stale = integrate_case(&set, 16);
    stale.frames = 5;
    let r = set.check(&stale);
    assert_eq!(r.status, Status::Fail);
    assert!(r.error.as_deref().unwrap().contains("input"), "{:?}", r.error);

    let mut opts = set.options();
    opts.effect = Some("selftest-integrate".into());
    let results = run::run(&opts, &|_| {}).unwrap();
    let t = run::totals(&results);
    assert_eq!((t.no_reference, t.fail), (1, 1));
}

#[test]
fn selfcheck_finds_the_temporal_cases_identical_for_any_thread_count() {
    let set = clips();
    let opts = SelfcheckOptions {
        testset: set.dir.clone(),
        effect: None,
        max_cases: 16,
        threads: vec![1, 3],
        vfx: Some(vfx().path.clone()),
        jobs: JOBS,
    };
    let checks = selfcheck::run(&opts).unwrap();
    assert_eq!(checks.len(), 16);
    let efx: std::collections::HashSet<&str> = checks.iter().map(|c| c.case.effect.as_str()).collect();
    assert_eq!(efx.len(), 2, "both effects are visited");
    for c in &checks {
        assert_eq!(c.runs, 4, "twice with the default thread count, then 1 and 3: {}", c.case.id);
        assert!(c.ok(), "{}: {:?} {:?}", c.case.id, c.error, c.differences);
    }
    assert!(selfcheck::all_ok(&checks));
    // narrowed to one effect
    let one = selfcheck::run(&SelfcheckOptions { effect: Some("selftest-shift".into()), max_cases: 3, ..opts }).unwrap();
    assert_eq!(one.len(), 3);
    assert!(one.iter().all(|c| c.case.effect == "selftest-shift"));
}

#[test]
fn selfcheck_fires_on_an_effect_that_depends_on_the_thread_count() {
    let plan = Plan { temporal: TemporalPlan { sizes: vec![(4, 3)], frames: 3, ..TemporalPlan::default() }, depths: vec![8, 16], ..Plan::default() };
    let set = build_set("threads", &plan, &[effect("selftest-threads")]);
    let checks = selfcheck::run(&SelfcheckOptions {
        testset: set.dir.clone(),
        effect: None,
        max_cases: 4,
        threads: vec![1, 3],
        vfx: Some(vfx().path.clone()),
        jobs: JOBS,
    })
    .unwrap();
    assert_eq!(checks.len(), 4);
    for c in &checks {
        assert!(c.error.is_none(), "{:?}", c.error);
        assert!(!c.ok(), "{} should differ", c.case.id);
        // the same thread count twice is stable; the others cannot all match the default one
        assert!(c.differences.iter().all(|d| d.against.starts_with("RAYON_NUM_THREADS=")), "{:?}", c.differences);
        assert!(!c.differences.is_empty() && c.differences.iter().any(|d| d.detail.contains("frame 0") && d.detail.contains("x=0 y=0")));
    }
    // and the binary says so and exits 1
    let out = parity(&["selfcheck", "--testset", set.dir.to_str().unwrap(), "--vfx", vfx().path.to_str().unwrap(), "--max-cases", "2"]);
    assert_eq!(code(&out), 1);
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("DIFF") && text.contains("RAYON_NUM_THREADS=") && text.contains("2 DIFFER"), "{text}");
    let out = parity(&["selfcheck", "--testset", set.dir.to_str().unwrap(), "--vfx", vfx().path.to_str().unwrap(), "--max-cases", "1", "--json"]);
    assert_eq!(code(&out), 1);
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!((v["summary"]["differ"].as_u64(), v["cases"][0]["ok"].as_bool()), (Some(1), Some(false)));
}

// ---- the real binary and ffmpeg ------------------------------------------------------------

fn ffmpeg_or_skip() -> Option<PathBuf> {
    let ffmpeg = ffmpeg_path();
    match ffmpeg_version(&ffmpeg) {
        Ok(_) => Some(ffmpeg),
        Err(e) => {
            eprintln!("skipping, no ffmpeg: {e}");
            None
        }
    }
}

fn parity(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_fx-parity")).args(args).output().expect("run fx-parity")
}

fn code(out: &std::process::Output) -> i32 {
    out.status.code().unwrap_or(-1)
}

fn pixel(dir: &Path, f: &InputFile, x: usize, y: usize) -> [f64; 4] {
    let bytes = std::fs::read(dir.join(&f.file)).unwrap();
    let n = f.width * f.height;
    let i = y * f.width + x;
    let u16at = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]) as f64;
    let f32at = |o: usize| f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as f64;
    match f.depth {
        8 => std::array::from_fn(|c| bytes[i * 4 + c] as f64 / 255.0),
        16 => std::array::from_fn(|c| u16at(i * 8 + c * 2) / 65535.0),
        // planes are G, B, R, A
        _ => [f32at((2 * n + i) * 4), f32at(i * 4), f32at((n + i) * 4), f32at((3 * n + i) * 4)],
    }
}

fn noise(x: usize, y: usize, w: usize, c: u64) -> f64 {
    let m = sources::NOISE_MOD;
    let n = ((y * w + x) as u64) * 4 + c + 1;
    let s1 = (n * 40503 + 12345) % m;
    let s2 = (s1 * s1 + 7919) % m;
    let s3 = (s2 * 48271 + s1) % m;
    let s4 = (s3 * s3 + s2) % m;
    let s5 = (s4 * 69621 + s3) % m;
    s5 as f64 / m as f64
}

#[test]
fn generated_inputs_hold_the_documented_patterns_at_every_depth() {
    if ffmpeg_or_skip().is_none() {
        return;
    }
    let dir = scratch("gen");
    let out = parity(&[
        "gen", "--out", dir.to_str().unwrap(), "--vfx", vfx().path.to_str().unwrap(),
        "--effects", "selftest-invert", "--sizes", "1x1,2x3,37x23", "--jobs", "4", "--no-sweep",
    ]);
    assert_eq!(code(&out), 0, "{}", String::from_utf8_lossy(&out.stderr));
    let set: TestSet = serde_json::from_slice(&std::fs::read(dir.join("cases.json")).unwrap()).unwrap();
    assert_eq!(set.inputs.len(), 3 * (4 * 3 + 1));
    for f in &set.inputs {
        let bpp = fx_core::Depth::from_bits(f.depth).unwrap().bytes_per_pixel();
        assert_eq!(std::fs::metadata(dir.join(&f.file)).unwrap().len() as usize, f.width * f.height * bpp, "{}", f.file);
    }
    let input = |src: &str, depth: u32, size: (usize, usize)| {
        set.inputs.iter().find(|f| f.source == src && f.depth == depth && (f.width, f.height) == size).unwrap().clone()
    };
    for depth in [8u32, 16, 32] {
        let tol = match depth {
            8 => 0.5 / 255.0 + 1e-9,
            16 => 0.5 / 65535.0 + 1e-9,
            _ => 1e-6,
        };
        let close = |got: [f64; 4], want: [f64; 4], what: &str| {
            for c in 0..4 {
                assert!((got[c] - want[c]).abs() <= tol, "{what} @{depth} channel {c}: got {got:?}, want {want:?}");
            }
        };
        let (w, h) = (37usize, 23usize);
        let g = input("gradient", depth, (w, h));
        close(pixel(&dir, &g, 0, 0), [0.0, 0.0, 0.0, 1.0], "gradient origin");
        close(pixel(&dir, &g, w - 1, h - 1), [1.0, 1.0, 1.0, 1.0], "gradient far corner");
        close(pixel(&dir, &g, w - 1, 0), [1.0, 0.0, (w - 1) as f64 / (w + h - 2) as f64, 1.0], "gradient top right");
        close(pixel(&dir, &input("gradient", depth, (1, 1)), 0, 0), [0.5, 0.5, 0.5, 1.0], "gradient 1x1");
        let a = input("alpha", depth, (w, h));
        close(pixel(&dir, &a, 0, 0), [0.0, 1.0, 0.5, 0.0], "alpha origin");
        close(pixel(&dir, &a, w - 1, h - 1), [1.0, 0.0, 0.5, 1.0], "alpha far corner");
        close(pixel(&dir, &a, 18, 11), [0.5, 0.5, 0.5, 0.5], "alpha middle");
        let b = input("bars", depth, (w, h));
        close(pixel(&dir, &b, 0, 0), [0.75, 0.75, 0.75, 1.0], "white bar");
        close(pixel(&dir, &b, 6, 0), [0.75, 0.75, 0.0, 1.0], "yellow bar");
        close(pixel(&dir, &b, w - 1, 0), [0.0, 0.0, 0.75, 1.0], "blue bar");
        let nz = input("noise", depth, (w, h));
        for (x, y) in [(0, 0), (5, 3), (36, 22), (20, 10)] {
            close(
                pixel(&dir, &nz, x, y),
                [noise(x, y, w, 0), noise(x, y, w, 1), noise(x, y, w, 2), 1.0],
                &format!("noise {x},{y}"),
            );
        }
    }
    let r = input("range", 32, (37, 23));
    let corner = pixel(&dir, &r, 0, 0);
    assert_eq!([corner[0], corner[1], corner[2]], [-0.5, -0.5, 1.5]);
    assert_eq!(pixel(&dir, &r, 36, 22)[0], 1.5);

    // fixed seed: a second run writes the same bytes
    let again = scratch("gen-again");
    let out = parity(&[
        "gen", "--out", again.to_str().unwrap(), "--vfx", vfx().path.to_str().unwrap(),
        "--effects", "selftest-invert", "--sizes", "1x1,2x3,37x23", "--no-sweep",
    ]);
    assert_eq!(code(&out), 0);
    for f in &set.inputs {
        assert_eq!(std::fs::read(dir.join(&f.file)).unwrap(), std::fs::read(again.join(&f.file)).unwrap(), "{}", f.file);
    }
    assert_eq!(std::fs::read(dir.join("cases.json")).unwrap(), std::fs::read(again.join("cases.json")).unwrap());
}

#[test]
fn the_binary_runs_a_generated_set_end_to_end_params_and_all() {
    if ffmpeg_or_skip().is_none() {
        return;
    }
    // the frame effect with the fewest params, narrowed to a single source and depth so the sweeps stay short
    let effect = frame_effects().into_iter().filter(|e| !e.params.is_empty()).min_by_key(|e| e.params.len()).expect("a frame effect with params");
    let dir = scratch("e2e");
    let vfx_path = vfx().path.to_str().unwrap().to_string();
    let out = parity(&[
        "gen", "--out", dir.to_str().unwrap(), "--vfx", &vfx_path, "--effects", &effect.id,
        "--sizes", "2x3,37x23", "--sources", "noise", "--depths", "16",
    ]);
    assert_eq!(code(&out), 0, "{}", String::from_utf8_lossy(&out.stderr));
    let set: TestSet = serde_json::from_slice(&std::fs::read(dir.join("cases.json")).unwrap()).unwrap();
    assert!(set.cases.iter().all(|c| c.effect == effect.id));
    assert!(set.cases.len() > effect.params.len(), "{} cases for {} params", set.cases.len(), effect.params.len());
    let refs = dir.join("refs");
    // a references directory that is not there is a mistake, not a set with nothing rendered
    let out = parity(&["run", "--testset", dir.to_str().unwrap(), "--refs", refs.to_str().unwrap(), "--vfx", &vfx_path]);
    assert_eq!(code(&out), 3);
    assert!(String::from_utf8_lossy(&out.stderr).contains("does not exist"));
    std::fs::create_dir_all(&refs).unwrap();
    let args = |extra: &[&'static str]| {
        let mut v = vec!["run", "--testset", dir.to_str().unwrap(), "--refs", refs.to_str().unwrap(), "--vfx", &vfx_path];
        v.extend_from_slice(extra);
        parity(&v)
    };

    // no references yet: everything is listed as such, nothing passes
    let out = args(&["--json"]);
    assert_eq!(code(&out), 0);
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["summary"]["no_reference"], set.cases.len());
    assert_eq!(v["summary"]["pass"], 0);
    assert_eq!(code(&args(&["--strict"])), 1);

    // a misnamed reference is called out instead of silently never matching
    std::fs::write(refs.join("typo.rgba"), b"x").unwrap();
    let out = args(&[]);
    assert!(String::from_utf8_lossy(&out.stderr).contains("1 files in"), "{}", String::from_utf8_lossy(&out.stderr));
    std::fs::remove_file(refs.join("typo.rgba")).unwrap();

    parallel(&set.cases, |c| {
        let input = std::fs::read(dir.join(&c.input)).unwrap();
        let output = vfx().apply(c, &input).unwrap_or_else(|e| panic!("{}: {e}", c.id));
        std::fs::write(refs.join(&c.reference), output).unwrap();
    });
    let out = args(&["--json"]);
    assert_eq!(code(&out), 0, "{}", String::from_utf8_lossy(&out.stdout));
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["summary"]["pass"], set.cases.len());
    assert_eq!(v["summary"]["fail"], 0);

    // damage one reference in two pixels
    let c = set.cases.iter().find(|c| c.width == 37 && c.label == "all-default").unwrap();
    let p = refs.join(&c.reference);
    let mut bytes = std::fs::read(&p).unwrap();
    bytes[10 * 8] ^= 0x01;
    bytes[200 * 8 + 6] ^= 0x80;
    std::fs::write(&p, bytes).unwrap();
    let out = args(&["--json"]);
    assert_eq!(code(&out), 1);
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["summary"]["fail"], 1);
    let bad = v["cases"].as_array().unwrap().iter().find(|c| c["status"] == "FAIL").unwrap();
    assert_eq!(bad["id"], c.id.as_str());
    assert_eq!(bad["differing_pixels"], 2);
    assert_eq!(bad["mismatches"][0]["x"], 10);
    assert_eq!(bad["mismatches"][1]["y"], 200 / 37);
    assert_eq!(bad["params"], serde_json::json!(c.params));
    assert!(bad["mean_abs_diff"].as_f64().unwrap() > 0.0, "the damaged case has a mean error");
    assert_eq!(v["graded"][effect.id.as_str()]["worst"]["id"], c.id.as_str());
    assert_eq!(v["graded"][effect.id.as_str()]["compared"].as_u64(), Some(set.cases.len() as u64));
    let human = args(&[]);
    assert_eq!(code(&human), 1);
    let text = String::from_utf8_lossy(&human.stdout);
    assert!(text.contains("FAIL") && text.contains("2 of 851 pixels differ"), "{text}");
    assert!(text.contains("% of full scale)") && text.contains("\ngraded error\n"), "{text}");
    assert!(text.contains(&format!("worst {} (", c.id)), "{text}");
}

fn frames_of(bytes: &[u8], f: &InputFile) -> Vec<Vec<u8>> {
    let len = f.width * f.height * fx_core::Depth::from_bits(f.depth).unwrap().bytes_per_pixel();
    assert_eq!(bytes.len(), len * f.frames, "{}", f.file);
    bytes.chunks(len).map(<[u8]>::to_vec).collect()
}

#[test]
fn gen_renders_moving_clips_for_temporal_effects_and_the_whole_chain_runs_on_them() {
    if ffmpeg_or_skip().is_none() {
        return;
    }
    let dir = scratch("tgen");
    let vfx_path = vfx().path.to_str().unwrap().to_string();
    let out = parity(&[
        "gen", "--out", dir.to_str().unwrap(), "--vfx", &vfx_path, "--effects", "selftest-shift,selftest-integrate",
        "--sizes", "16x12,37x23", "--jobs", "4",
    ]);
    assert_eq!(code(&out), 0, "{}", String::from_utf8_lossy(&out.stderr));
    let set: TestSet = serde_json::from_slice(&std::fs::read(dir.join("cases.json")).unwrap()).unwrap();

    // the plan: clips of 12 frames, both rates, layers for the effect that lists one
    assert!(set.cases.iter().all(|c| c.kind == CaseKind::Temporal && c.frames == 12));
    let rates: std::collections::HashSet<&str> = set.cases.iter().map(|c| c.fps.as_deref().unwrap()).collect();
    assert_eq!(rates, ["24", "30000/1001"].into_iter().collect());
    let sizes: std::collections::HashSet<(usize, usize)> = set.cases.iter().map(|c| (c.width, c.height)).collect();
    assert_eq!(sizes, [(16, 12), (37, 23)].into_iter().collect());
    let ids: std::collections::HashSet<&str> = set.cases.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(ids.len(), set.cases.len());
    for c in set.cases.iter().filter(|c| c.effect == "selftest-shift") {
        assert_eq!(c.layers.contains_key("overlay"), c.label != "no-layers", "{}", c.id);
    }
    assert!(set.cases.iter().filter(|c| c.effect == "selftest-integrate").all(|c| c.layers.is_empty()));
    assert!(set.cases.iter().any(|c| c.params == ["gain=-8@0;8@11"]), "a two key ramp over the clip");
    for label in ["all-default", "all-min", "all-max"] {
        for depth in [8u32, 16, 32] {
            for size in [(16usize, 12usize), (37, 23)] {
                assert!(set.cases.iter().any(|c| c.effect == "selftest-integrate" && c.label == label && c.depth == depth && (c.width, c.height) == size), "{label} {depth} {size:?}");
            }
        }
    }

    // the clips: right length, and every one of them moves from frame to frame
    assert!(set.inputs.iter().all(|f| f.frames == 12));
    for f in &set.inputs {
        let bytes = std::fs::read(dir.join(&f.file)).unwrap();
        let frames = frames_of(&bytes, f);
        for (i, pair) in frames.windows(2).enumerate() {
            assert!(pair[0] != pair[1], "{} stands still between frames {i} and {}", f.file, i + 1);
        }
    }
    let input = |src: &str, depth: u32, size: (usize, usize)| {
        set.inputs.iter().find(|f| f.source == src && f.depth == depth && (f.width, f.height) == size).unwrap_or_else(|| panic!("no {src} clip")).clone()
    };
    let at = |f: &InputFile, frame: usize, x: usize, y: usize| -> [f64; 4] {
        // one frame at a time through the single frame reader
        let len = f.width * f.height * fx_core::Depth::from_bits(f.depth).unwrap().bytes_per_pixel();
        let bytes = std::fs::read(dir.join(&f.file)).unwrap();
        let single = InputFile { file: format!("{}.f{frame}", f.file), frames: 1, ..f.clone() };
        std::fs::write(dir.join(&single.file), &bytes[frame * len..(frame + 1) * len]).unwrap();
        pixel(&dir, &single, x, y)
    };
    for depth in [8u32, 16, 32] {
        let tol = match depth {
            8 => 0.5 / 255.0 + 1e-9,
            16 => 0.5 / 65535.0 + 1e-9,
            _ => 1e-6,
        };
        let size = (16usize, 12usize);
        // testsrc2 is opaque (swscale widens 255 to within a few levels of full scale at 16 bpc)
        let t = input("testsrc2", depth, size);
        for frame in [0, 11] {
            assert!((at(&t, frame, 3, 2)[3] - 1.0).abs() <= 2e-4, "testsrc2 alpha @{depth}");
        }
        // movegrad is opaque and its red channel moves by 0.07 a frame: pixel (0, 0) has FX = 0
        let m = input("movegrad", depth, size);
        for frame in 0..12 {
            let p = at(&m, frame, 0, 0);
            assert!((p[0] - 0.07 * frame as f64).abs() <= tol + 1e-6, "movegrad frame {frame} @{depth}: {p:?}");
            assert!((p[3] - 1.0).abs() <= 1e-6);
        }
        // alphaslide: alpha is clip(2 FX - N / 6 + 0.5, 0, 1), a ramp sliding to the left
        let a = input("alphaslide", depth, size);
        let want = [((0, 0, 0), 0.5), ((11, 0, 0), 0.0), ((0, 15, 0), 1.0), ((11, 15, 0), 2.5 - 11.0 / 6.0)];
        for ((frame, x, y), alpha) in want {
            let got = at(&a, frame, x, y)[3];
            assert!((got - alpha).abs() <= tol + 1e-6, "alphaslide alpha at frame {frame} x={x} @{depth}: {got}, want {alpha}");
        }
        // noisefx: frame 0 is the `noise` still
        let n = input("noisefx", depth, size);
        let still = scratch(&format!("still-{depth}")).join("noise");
        render_input(&ffmpeg_path(), "noise", depth, 16, 12, &still).unwrap();
        let len = 16 * 12 * fx_core::Depth::from_bits(depth).unwrap().bytes_per_pixel();
        assert_eq!(&std::fs::read(dir.join(&n.file)).unwrap()[..len], &std::fs::read(&still).unwrap()[..], "noisefx frame 0 @{depth}");
    }

    // gen is repeatable down to the byte
    let again = scratch("tgen-again");
    let out = parity(&[
        "gen", "--out", again.to_str().unwrap(), "--vfx", &vfx_path, "--effects", "selftest-shift,selftest-integrate",
        "--sizes", "16x12,37x23", "--jobs", "4",
    ]);
    assert_eq!(code(&out), 0);
    assert_eq!(std::fs::read(dir.join("cases.json")).unwrap(), std::fs::read(again.join("cases.json")).unwrap());
    for f in &set.inputs {
        assert_eq!(std::fs::read(dir.join(&f.file)).unwrap(), std::fs::read(again.join(&f.file)).unwrap(), "{}", f.file);
    }

    // selfcheck runs on the generated set and finds nothing
    let out = parity(&["selfcheck", "--testset", dir.to_str().unwrap(), "--vfx", &vfx_path, "--max-cases", "12"]);
    assert_eq!(code(&out), 0, "{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stdout).contains("12 cases checked, 12 identical, 0 DIFFER"));

    // with references made by vfx itself the whole set passes through `fx-parity run`, and an
    // altered frame is found
    let refs = dir.join("refs");
    std::fs::create_dir_all(&refs).unwrap();
    let picked = selfcheck::pick(&set.cases, None, 30);
    parallel(&picked, |c| {
        let out = vfx().apply_clip(c, &dir, Threads::Inherit).unwrap_or_else(|e| panic!("{}: {e}", c.id));
        std::fs::write(refs.join(&c.reference), out).unwrap();
    });
    let args = |extra: &[&str]| {
        let mut v = vec!["run", "--testset", dir.to_str().unwrap(), "--refs", refs.to_str().unwrap(), "--vfx", &vfx_path];
        v.extend_from_slice(extra);
        parity(&v)
    };
    let out = args(&["--json"]);
    assert_eq!(code(&out), 0);
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!((v["summary"]["pass"].as_u64(), v["summary"]["fail"].as_u64()), (Some(picked.len() as u64), Some(0)));
    let c = &picked[0];
    let p = refs.join(&c.reference);
    let mut bytes = std::fs::read(&p).unwrap();
    let len = c.frame_len();
    bytes[7 * len + 1] ^= 0x01;
    std::fs::write(&p, bytes).unwrap();
    let out = args(&["--json", "--brief"]);
    assert_eq!(code(&out), 1);
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    let bad = &v["cases"][0];
    assert_eq!((bad["id"].as_str(), bad["frame"].as_u64(), bad["differing_pixels"].as_u64()), (Some(c.id.as_str()), Some(7), Some(1)));
    assert_eq!(bad["frames"], 12);
    assert!(bad["repro"].as_str().unwrap().starts_with("vfx apply "));
}

#[test]
fn the_clip_flags_reach_the_plan() {
    if ffmpeg_or_skip().is_none() {
        return;
    }
    let dir = scratch("tflags");
    let out = parity(&[
        "gen", "--out", dir.to_str().unwrap(), "--vfx", vfx().path.to_str().unwrap(), "--effects", "selftest-shift",
        "--frames", "3", "--fps", "25", "--sizes", "8x6", "--depths", "8", "--clip-sources", "movegrad,noisefx",
        "--clip-sweep-sources", "noisefx", "--max-cases-per-effect", "1",
    ]);
    assert_eq!(code(&out), 0, "{}", String::from_utf8_lossy(&out.stderr));
    let set: TestSet = serde_json::from_slice(&std::fs::read(dir.join("cases.json")).unwrap()).unwrap();
    assert!(set.cases.iter().all(|c| c.frames == 3 && c.fps.as_deref() == Some("25") && (c.width, c.height, c.depth) == (8, 6, 8)));
    assert!(set.cases.iter().all(|c| ["movegrad", "noisefx"].contains(&c.source.as_str())));
    // the cap of 1 keeps what must stay: the four combined settings on both sources, and the single param's extremes
    let labels: std::collections::HashSet<&str> = set.cases.iter().map(|c| c.label.as_str()).collect();
    for want in ["all-default", "all-min", "all-max", "no-layers"] {
        assert!(labels.contains(want), "{want}");
    }
    assert_eq!(set.cases.iter().filter(|c| c.label == "all-default").count(), 2);
    assert!(set.inputs.iter().all(|f| f.frames == 3 && f.file.contains(".f3.")));
    assert!(set.inputs.iter().all(|f| std::fs::metadata(dir.join(&f.file)).unwrap().len() == 3 * 8 * 6 * 4));
    // and the frame stills were not rendered for a set of temporal effects
    assert!(set.inputs.iter().all(|f| sources::find_clip(&f.source).is_some()));
}

/// Runs ffmpeg with the arguments of a checklist command from inside the test set.
fn ffmpeg_in(ffmpeg: &Path, dir: &Path, args: &[String]) {
    let out = Command::new(ffmpeg).args(args).current_dir(dir).stdin(std::process::Stdio::null()).output().expect("run ffmpeg");
    assert!(out.status.success(), "{}: {}", checklist::shell_line(args), String::from_utf8_lossy(&out.stderr));
}

#[test]
fn the_checklist_commands_carry_every_input_through_a_program_and_back() {
    let Some(ffmpeg) = ffmpeg_or_skip() else { return };
    let dir = scratch("checklist");
    let vfx_path = vfx().path.to_str().unwrap().to_string();
    let out = parity(&[
        "gen", "--out", dir.to_str().unwrap(), "--vfx", &vfx_path, "--effects", "selftest-invert,selftest-shift",
        "--sizes", "5x4", "--frames", "3", "--no-sweep", "--max-cases-per-effect", "1", "--sources", "noise,alpha",
        "--clip-sources", "movegrad,alphaslide", "--jobs", "4",
    ]);
    assert_eq!(code(&out), 0, "{}", String::from_utf8_lossy(&out.stderr));
    let set: TestSet = serde_json::from_slice(&std::fs::read(dir.join("cases.json")).unwrap()).unwrap();
    let out = parity(&["checklist", "--testset", dir.to_str().unwrap(), "--vfx", &vfx_path]);
    assert_eq!(code(&out), 0, "{}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stdout).contains(&format!("{} cases for 2 effects", set.cases.len())));
    let md = std::fs::read_to_string(dir.join("CHECKLIST.md")).unwrap();
    assert!(md.is_ascii());
    // the effects and their layers are named as `vfx list` names them
    let shift = effect("selftest-shift");
    assert!(md.contains(&format!("## {} (`selftest-shift`)", shift.name)), "{md}");
    assert!(md.contains(&format!("  - {}: `inputs/", shift.layers[0].name)), "{md}");
    assert!(md.contains("| clip |") && md.contains("| still |"), "{md}");

    let exr = checklist::detect_exr(&ffmpeg);
    let layer_name = |id: &str| shift.layers.iter().find(|l| l.id == id).map_or(id.to_string(), |l| l.name.clone());
    std::fs::create_dir_all(dir.join(checklist::EXPORT_DIR)).unwrap();
    std::fs::create_dir_all(dir.join(checklist::REFS_DIR)).unwrap();
    let mut round_trips = 0;
    for c in &set.cases {
        assert_eq!(md.matches(&format!("- case: `{}`", c.id)).count(), 1, "{}", c.id);
        let Some(steps) = checklist::import_steps(c, &layer_name, exr) else {
            assert_eq!((c.depth, exr), (32, checklist::Exr::No));
            continue;
        };
        assert_eq!(steps.len(), 1 + c.layers.len(), "{}: the input and each layer", c.id);
        let (export_file, export_args) = checklist::export_step(c);
        for line in steps.iter().map(|s| checklist::shell_line(&s.args)).chain([checklist::shell_line(&export_args)]) {
            assert!(md.contains(&line), "missing from the file: {line}");
        }
        for s in &steps {
            ffmpeg_in(&ffmpeg, &dir, &s.args);
        }
        // the program opens the input and exports it again, unchanged
        for n in 0..c.frames {
            let pick = |name: &str| dir.join(name.replace("%04d", &format!("{n:04}")));
            std::fs::copy(pick(&steps[0].output), pick(&export_file)).unwrap();
        }
        ffmpeg_in(&ffmpeg, &dir, &export_args);
        let before = std::fs::read(dir.join(&c.input)).unwrap();
        let after = std::fs::read(dir.join(checklist::reference_file(c))).unwrap();
        assert_eq!(before.len(), after.len(), "{}", c.id);
        // colour under zero alpha does not survive an EXR, nothing else is lost
        let zero_alpha = c.depth == 32 && ["alpha", "alphaslide"].contains(&c.source.as_str());
        if !zero_alpha {
            assert!(before == after, "{} does not come back byte for byte", c.id);
            round_trips += 1;
        }
    }
    assert!(round_trips >= 8, "{round_trips} exact round trips");
    assert!(set.cases.iter().any(|c| c.is_temporal() && !c.layers.is_empty()), "a clip with a layer was in the set");
}

#[test]
fn command_line_errors_use_the_documented_exit_codes() {
    assert_eq!(code(&parity(&[])), 2);
    assert_eq!(code(&parity(&["run", "--testset", "x"])), 2);
    assert_eq!(code(&parity(&["bogus"])), 2);
    assert_eq!(code(&parity(&["selfcheck"])), 2);
    assert_eq!(code(&parity(&["checklist"])), 2);
    assert_eq!(code(&parity(&["checklist", "--testset", "x", "--nope"])), 2);
    assert_eq!(code(&parity(&["selfcheck", "--testset", "x", "--threads", "0"])), 2);
    let missing = scratch("missing");
    let out = parity(&["run", "--testset", missing.to_str().unwrap(), "--refs", missing.to_str().unwrap(), "--vfx", vfx().path.to_str().unwrap()]);
    assert_eq!(code(&out), 3);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(err.trim_end().lines().count(), 1, "{err}");
    assert!(err.contains("cases.json"));
    let out = parity(&["selfcheck", "--testset", missing.to_str().unwrap(), "--vfx", vfx().path.to_str().unwrap()]);
    assert_eq!(code(&out), 3);
    assert!(String::from_utf8_lossy(&out.stderr).contains("cases.json"));
    let out = parity(&["checklist", "--testset", missing.to_str().unwrap(), "--vfx", vfx().path.to_str().unwrap()]);
    assert_eq!(code(&out), 3);
    assert!(String::from_utf8_lossy(&out.stderr).contains("cases.json"));
    // a set of frame cases has nothing for selfcheck to run
    let frames_only = base();
    let out = parity(&["selfcheck", "--testset", frames_only.dir.to_str().unwrap(), "--vfx", vfx().path.to_str().unwrap()]);
    assert_eq!(code(&out), 3);
    assert!(String::from_utf8_lossy(&out.stderr).contains("has no temporal cases"), "{}", String::from_utf8_lossy(&out.stderr));
    let out = parity(&["selfcheck", "--testset", clips().dir.to_str().unwrap(), "--vfx", vfx().path.to_str().unwrap(), "--effect", "selftest-nothing"]);
    assert_eq!(code(&out), 3);
    assert!(String::from_utf8_lossy(&out.stderr).contains("no temporal cases for effect `selftest-nothing`"));
    let out = parity(&["gen", "--out", missing.to_str().unwrap(), "--vfx", "does/not/exist"]);
    assert_eq!(code(&out), 3);
    let out = parity(&["gen", "--out", missing.to_str().unwrap(), "--vfx", vfx().path.to_str().unwrap(), "--effects", "nope"]);
    assert_eq!(code(&out), 3);
    assert!(String::from_utf8_lossy(&out.stderr).contains("no effect `nope`"));
    let out = parity(&["gen", "--out", missing.to_str().unwrap(), "--vfx", vfx().path.to_str().unwrap(), "--effects", "selftest-invert", "--sources", "range", "--depths", "8"]);
    assert_eq!(code(&out), 3);
    assert!(String::from_utf8_lossy(&out.stderr).contains("nothing to render"));
    assert_eq!(code(&parity(&["--help"])), 0);
    let help = String::from_utf8_lossy(&parity(&["--help"]).stdout).to_string();
    assert!(help.contains("selfcheck") && help.contains("--max-cases-per-effect"));
    assert!(help.contains("fx-parity checklist --testset DIR") && help.contains("graded error"));
}

// ---- the reference case families ----------------------------------------------------------

/// `vfx list --json --all` as JSON, for the `gaps` lists the harness itself does not read.
fn list_json() -> Value {
    let out = Command::new(&vfx().path).args(["list", "--json", "--all"]).output().expect("run vfx list");
    assert!(out.status.success());
    serde_json::from_slice(&out.stdout).expect("vfx list --json is JSON")
}

#[test]
fn every_family_case_is_a_setting_the_built_vfx_takes_and_every_listed_gap_has_a_family() {
    // the families are built from the real effect list: none of their cases is dropped
    let ids: Vec<String> = families::FAMILIES.iter().map(|f| f.id.to_string()).collect();
    let built = families::build(effects(), &ids).expect("every family setting is known to vfx");
    for (id, n) in &built.counts {
        assert_eq!(*n, families::cases_of(id).len(), "{id} lost cases");
    }
    assert_eq!(built.cases.len(), families::all().len());
    // vfx itself validates every param, value, key and layer of every case
    let errors: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    parallel(&built.cases, |c| {
        if let Err(e) = vfx().check(c) {
            errors.lock().unwrap().push(format!("{}: {e}", c.id));
        }
    });
    let errors = errors.into_inner().unwrap();
    assert!(errors.is_empty(), "{errors:#?}");
    // the register as each effect lists it: every gap has a family and the same title
    let list = list_json();
    let mut listed: Vec<String> = Vec::new();
    for e in list.as_array().unwrap() {
        for g in e["gaps"].as_array().into_iter().flatten() {
            let (id, title) = (g["id"].as_str().unwrap(), g["title"].as_str().unwrap());
            if e["hidden"].as_bool() == Some(true) {
                continue;
            }
            assert_eq!(families::title(id), title, "{id} of {}", e["id"]);
            assert!(families::FAMILIES.iter().any(|f| f.gaps.contains(&id)), "{id} of {} has no reference case family", e["id"]);
            listed.push(id.to_string());
        }
    }
    // and every register gap a family names is listed by some effect (G7 and K1 are not in the register)
    for f in families::FAMILIES.iter().filter(|f| !f.observe_only) {
        for g in f.gaps {
            assert!(listed.iter().any(|l| l == g), "{} names {g}, which no effect lists", f.id);
        }
    }
}

#[test]
fn gen_families_writes_a_set_that_checklist_and_run_take_and_runnable_cases_note_their_gaps() {
    if ffmpeg_or_skip().is_none() {
        return;
    }
    let dir = scratch("families");
    let vfx_path = vfx().path.to_str().unwrap().to_string();
    let out = parity(&["gen", "--out", dir.to_str().unwrap(), "--families", "all", "--vfx", &vfx_path]);
    assert_eq!(code(&out), 0, "{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let set: TestSet = serde_json::from_str(&std::fs::read_to_string(dir.join("cases.json")).unwrap()).unwrap();
    let counts: Vec<(&str, usize)> = families::FAMILIES.iter().map(|f| (f.id, families::cases_of(f.id).len())).collect();
    assert_eq!(stdout.lines().nth(1), Some(families::summary(&counts).as_str()), "{stdout}");
    assert_eq!(set.cases.len(), families::all().len());
    assert!(set.cases.iter().all(|c| c.family.is_some() && c.is_temporal() && c.frames == families::FRAMES));
    for f in &set.inputs {
        let len = std::fs::metadata(dir.join(&f.file)).unwrap().len() as usize;
        assert_eq!(len, f.frames * f.width * f.height * fx_core::Depth::from_bits(f.depth).unwrap().bytes_per_pixel(), "{}", f.file);
    }
    // the matte is the disc of the main clip: frame 5 of fgdisc_matte is white exactly where the disc is
    let matte = set.inputs.iter().find(|f| f.file == "inputs/fgdisc_matte-64x48.d8.f12.rgba").expect("the hard matte");
    let (cx, cy, r) = (19.0 + 2.0 * 5.0, 16.0 + 5.0, 10.0);
    let bytes = std::fs::read(dir.join(&matte.file)).unwrap();
    let frame = &bytes[5 * 64 * 48 * 4..6 * 64 * 48 * 4];
    for y in 0..48 {
        for x in 0..64 {
            let inside = (x as f64 - cx) * (x as f64 - cx) + (y as f64 - cy) * (y as f64 - cy) <= r * r;
            assert_eq!(frame[(y * 64 + x) * 4], if inside { 255 } else { 0 }, "matte at {x},{y}");
        }
    }

    let out = parity(&["checklist", "--testset", dir.to_str().unwrap(), "--vfx", &vfx_path]);
    assert_eq!(code(&out), 0, "{}", String::from_utf8_lossy(&out.stderr));
    let md = std::fs::read_to_string(dir.join("CHECKLIST.md")).unwrap();
    assert!(md.contains("## Parity gaps these cases settle") && md.contains("FG Matte popup"));
    assert_eq!(md.matches("\n- settles: ").count(), set.cases.len());
    assert!(md.is_ascii());

    // references from vfx itself for every case it renders; a case it cannot render yet must be
    // refused as unsupported (a control the build does not support yet), never fail otherwise
    let refs = dir.join("refs");
    std::fs::create_dir_all(&refs).unwrap();
    let rendered = AtomicUsize::new(0);
    let odd: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    parallel(&set.cases, |c| match vfx().apply_clip_noted(c, &dir, Threads::Inherit) {
        Ok((clip, _)) => {
            std::fs::write(refs.join(&c.reference), clip).unwrap();
            rendered.fetch_add(1, Ordering::SeqCst);
        }
        Err(e) if e.contains("exited with code 3") && e.contains("unsupported") => {}
        Err(e) => odd.lock().unwrap().push(format!("{}: {e}", c.id)),
    });
    let odd = odd.into_inner().unwrap();
    assert!(odd.is_empty(), "{odd:#?}");
    let rendered = rendered.into_inner();
    assert!(rendered >= families::cases_of("gap-tiny").len(), "the tiny frames render today: {rendered}");

    let out = parity(&["run", "--testset", dir.to_str().unwrap(), "--refs", refs.to_str().unwrap(), "--json", "--vfx", &vfx_path]);
    assert_eq!(code(&out), 0, "{}", String::from_utf8_lossy(&out.stderr));
    // every family case that renders prints the notes of the gaps it is there to settle
    let warnings = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(!warnings.contains("printed no note"), "{warnings}");
    let report: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(report["summary"]["pass"].as_u64(), Some(rendered as u64));
    assert_eq!(report["summary"]["fail"].as_u64(), Some(0));
    let fams = report["families"].as_object().expect("a families block");
    assert_eq!(fams.len(), families::FAMILIES.len());
    let compared: u64 = fams.values().map(|f| f["compared"].as_u64().unwrap()).sum();
    assert_eq!(compared, rendered as u64);
    for (id, f) in fams {
        let n = families::cases_of(id).len() as u64;
        assert_eq!(f["cases"].as_u64(), Some(n), "{id}");
        assert_eq!(f["exact"], f["compared"], "{id}: references made by vfx itself are bit exact");
        assert_eq!(f["settled"].as_bool(), Some(f["compared"].as_u64() == Some(n)), "{id}");
    }
    assert_eq!(fams["gap-tiny"]["settled"], true);
    assert!(report["cases"].as_array().unwrap().iter().all(|c| c["family"].is_string() && c["notes"].is_array()));
}

#[test]
fn run_warns_when_a_family_case_prints_no_note_and_reads_the_real_note_lines() {
    // the hidden selftest-gap effect notes its gap T1 when `gap` is on; as cases of a family this
    // build does not know, any note is expected, so the case with `gap` off is the one to warn
    let plan = Plan { temporal: TemporalPlan { sizes: vec![(5, 4)], frames: 3, ..TemporalPlan::default() }, depths: vec![8], ..Plan::default() };
    let built = build_set("family-notes", &plan, &[effect("selftest-gap")]);
    let mut set: TestSet = serde_json::from_str(&std::fs::read_to_string(built.dir.join("cases.json")).unwrap()).unwrap();
    for c in set.cases.iter_mut() {
        c.family = Some("gap-later".into());
        c.label = format!("gap-later-{}", c.label);
    }
    std::fs::write(built.dir.join("cases.json"), serde_json::to_string_pretty(&set).unwrap()).unwrap();
    let on: Vec<&Case> = set.cases.iter().filter(|c| c.params.iter().any(|p| p == "gap=1")).collect();
    let off: Vec<&Case> = set.cases.iter().filter(|c| !c.params.iter().any(|p| p == "gap=1")).collect();
    assert!(!on.is_empty() && !off.is_empty());
    let (clip, notes) = vfx().apply_clip_noted(on[0], &built.dir, Threads::Inherit).unwrap();
    assert_eq!(clip, std::fs::read(built.refs.join(&on[0].reference)).unwrap(), "the clip is the one apply_clip gives");
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert_eq!(fx_parity::vfx::note_gap(&notes[0]), Some("T1"), "{}", notes[0]);
    let warnings = std::sync::Mutex::new(Vec::new());
    let results = run::run(&built.options(), &|m| warnings.lock().unwrap().push(m.to_string())).unwrap();
    assert!(results.iter().all(|r| r.status == Status::Pass));
    let warnings = warnings.into_inner().unwrap();
    assert_eq!(warnings.len(), off.len(), "{warnings:#?}");
    for c in &off {
        assert!(warnings.iter().any(|w| w.contains(&c.id) && w.contains("printed no note")), "{}: {warnings:#?}", c.id);
    }
    for c in &on {
        assert!(!warnings.iter().any(|w| w.contains(&c.id)), "{}", c.id);
        let r = results.iter().find(|r| r.case.id == c.id).unwrap();
        assert_eq!(r.notes.len(), 1);
    }
    let report = run::to_json(&results, false);
    assert_eq!(report["families"]["gap-later"]["settled"], true);
    assert_eq!(report["families"]["gap-later"]["exact"].as_u64(), Some(set.cases.len() as u64));
}
