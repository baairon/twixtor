//! The reference case families of the open parity gaps (`fx-parity gen --families`).
//!
//! Every control of an effect that runs a stated rule (an entry of the parity gaps register,
//! `retime_engine::gaps`, listed per effect as `gaps` by `vfx list --json`) comes with the cases
//! that, rendered once in the original plugin, settle that rule: the family. Two families observe
//! points that print no note, because there is nothing inside the plugin to grade them against:
//! `gap-tiny` (G7, frames of 1x1 to 5x5, where the original is undefined) and `gap-keys` (K1,
//! Bezier ease handles, which the host evaluates).
//!
//! A family case is an ordinary temporal case with `family` set: its own clip sources
//! ([`crate::sources::FAMILY_SOURCES`] and the matrix clips), its own size, depth, frame count
//! and rate, and every setting spelled with the param ids and values `vfx list --json` reports.
//! None of it touches the default matrix: without `--families` the test set is what it always was.
//!
//! Retiring a note: when `fx-parity run` reports a family `settled` (every case compared and bit
//! exact), its register entry in `plugins/retime-engine/src/gaps.rs` and its `record` calls are
//! deleted, and its Stated rules row in `docs/parity.md` goes.

use crate::cases::{self, Case, CaseKind, InputFile};
use crate::effects::{EffectInfo, Kind};
use crate::sources;
use std::collections::BTreeMap;

/// The frame size of the family cases that are not about the size: big enough for a disc of
/// radius 10 to cross the frame, small enough to stay quick at 32 bpc.
pub const SIZE: (usize, usize) = (64, 48);
/// Frames per family clip, and the rate every family case runs at.
pub const FRAMES: usize = 12;
pub const FPS: &str = "24";

/// One family: the open points it settles and the rule this project runs until then.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FamilyInfo {
    pub id: &'static str,
    /// Register ids (`G3`, `G4`, ...), or `G7` and `K1a` to `K1e` for the observing families.
    pub gaps: &'static [&'static str],
    /// What the references decide, in one line.
    pub settles: &'static str,
    /// The rule `vfx` runs until a graded pass settles it.
    pub rule: &'static str,
    /// True when no case of the family prints a note on purpose (G7 and K1).
    pub observe_only: bool,
}

/// One case of a family, before it is crossed with nothing: every field is fixed.
#[derive(Clone, Debug, PartialEq)]
pub struct Family {
    /// The family id, for example `gap-drawgeom`.
    pub gap: &'static str,
    /// What sets this case apart inside its family, for example `cv-lo5-shrink-0.5`.
    pub label: String,
    pub effect: &'static str,
    /// `k=v` strings exactly as `vfx apply --param` takes them.
    pub params: Vec<String>,
    /// Layer id and the clip source it gets.
    pub layers: Vec<(&'static str, &'static str)>,
    /// The clip source of the main input.
    pub main: &'static str,
    pub size: (usize, usize),
    pub depth: u32,
    /// The register ids whose `vfx: note:` this case must print; empty when it should print none.
    pub notes: Vec<&'static str>,
    /// The gap ids this case settles, when it settles fewer than its family's.
    pub settles: Vec<&'static str>,
}

/// Every family, in the order of the register.
pub const FAMILIES: &[FamilyInfo] = &[
    FamilyInfo {
        id: "gap-drawgeom",
        gaps: &["G3"],
        settles: "what retime-fgbg's Draw Geom draws for three points at sub pixel positions on its Source display with Alt Color Source set, at every depth",
        rule: "an opaque disc of radius 1.5 times the downsample ratio in the point colour, in point order, later points on top",
        observe_only: false,
    },
    FamilyInfo {
        id: "gap-debug",
        gaps: &["G4"],
        settles: "the Blend view of the Debug display for a one layer group of an FG stack (FG1 alone, MAIN_BG alone), for both Track Frames",
        rule: "the plain lerp(W', Y_k, 0.5) of the group's composite and its own layer image, with no alpha match",
        observe_only: false,
    },
    FamilyInfo {
        id: "gap-tracker2",
        gaps: &["G5"],
        settles: "the flow of a layer whose Track is off, built from 0, 1 or 2 points, at both Vector Resolutions",
        rule: "a pull-push fill of the level 0 splat of the points, with no extra blur",
        observe_only: false,
    },
    FamilyInfo {
        id: "gap-wrap",
        gaps: &["G6"],
        settles: "how 360 Video wraps the solver, the masks of a layered render, the hole fill and the blur across the left and right edges, and a pole crossing",
        rule: "the solver, the layer masks, the hole fill and the motion blur wrap horizontally (indices modulo the width, y range checked only); the forward warp itself does not wrap",
        observe_only: false,
    },
    FamilyInfo {
        id: "gap-tiny",
        gaps: &["G7"],
        settles: "what the original does with frames of 1x1 to 5x5, where it divides by zero",
        rule: "the search never halves below 2x2; a frame with a side of 1 has no search and renders as No Vectors (create-vectors: the zero vector)",
        observe_only: true,
    },
    FamilyInfo {
        id: "gap-keys",
        gaps: &["K1a", "K1b", "K1c", "K1d", "K1e"],
        settles: "how the host eases Bezier keys on Frame Num, Speed %, Sensitivity and a point, read out through Blend weights with No Vectors",
        rule: "a cubic Bezier per segment solved by 64 bisection steps; a plain end has the straight line speed at influence 1/3, Easy Ease is speed 0 at exactly 1/3; floats clamp to their valid range; a point moves on the straight line",
        observe_only: true,
    },
];

/// The one line titles of the gaps the families settle: the register's own titles for G3 to G6
/// (as `vfx list --json` prints them), and the observed points G7 and K1a to K1e.
pub const TITLES: &[(&str, &str)] = &[
    ("G3", "Draw Geom drawing"),
    ("G4", "one-layer Debug Blend view"),
    ("G5", "second tracker arithmetic"),
    ("G6", "solver, hole fill and blur wrap variants"),
    ("G7", "tiny frames"),
    ("K1a", "a Linear end next to a Bezier end"),
    ("K1b", "Easy Ease influence"),
    ("K1c", "overshoot past the valid range"),
    ("K1d", "the host's own Bezier solve"),
    ("K1e", "a point's spatial path between keys"),
];

/// The title of a gap id, or the id itself when it is not one of [`TITLES`].
pub fn title(gap: &str) -> &str {
    TITLES.iter().find(|(id, _)| *id == gap).map_or(gap, |(_, t)| *t)
}

pub fn find(id: &str) -> Option<&'static FamilyInfo> {
    FAMILIES.iter().find(|f| f.id == id)
}

/// `IDS|all` as `--families` takes it: family ids separated by commas, or `all`.
pub fn parse_ids(text: &str) -> Result<Vec<String>, String> {
    if text.trim() == "all" {
        return Ok(FAMILIES.iter().map(|f| f.id.to_string()).collect());
    }
    let mut out: Vec<String> = Vec::new();
    for id in text.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if find(id).is_none() {
            let known: Vec<&str> = FAMILIES.iter().map(|f| f.id).collect();
            return Err(format!("unknown family `{id}`, known: {}, or all", known.join(", ")));
        }
        if !out.iter().any(|o| o == id) {
            out.push(id.to_string());
        }
    }
    if out.is_empty() {
        return Err("`--families` is empty".into());
    }
    Ok(out)
}

// ---- the cases ----------------------------------------------------------------------------

fn p(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

/// A case at the family size and 8 bpc; the builders below change what differs.
fn case(gap: &'static str, label: impl Into<String>, effect: &'static str, main: &'static str, params: Vec<String>) -> Family {
    Family { gap, label: label.into(), effect, params, layers: Vec::new(), main, size: SIZE, depth: 8, notes: Vec::new(), settles: Vec::new() }
}

impl Family {
    fn layer(mut self, id: &'static str, source: &'static str) -> Family {
        self.layers.push((id, source));
        self
    }

    fn notes(mut self, gaps: &[&'static str]) -> Family {
        self.notes = gaps.to_vec();
        self
    }

    fn settles(mut self, gaps: &[&'static str]) -> Family {
        self.settles = gaps.to_vec();
        self
    }

    fn depth(mut self, depth: u32) -> Family {
        self.depth = depth;
        self
    }

    fn size(mut self, size: (usize, usize)) -> Family {
        self.size = size;
        self
    }
}

const CV: &str = "create-vectors";
const FGBG: &str = "retime-fgbg";
const VIN: &str = "vectors-in";
const RETIME: &str = "retime";

/// Half speed, so that every other output frame is an in-between frame on the vector path.
const HALF: &str = "speed_percent=50";

/// Three points at sub pixel positions on three layers' choices (all the only group without mattes).
const DRAWN: [&str; 6] = ["pt1_use=2", "pt1_pos=10.3,12.6", "pt2_use=3", "pt2_pos=30.5,20.5", "pt3_use=5", "pt3_pos=50.75,35.25"];

fn drawgeom() -> Vec<Family> {
    let mut out = Vec::new();
    for depth in [8, 16, 32] {
        for g in [2, 3, 6] {
            // the Source display shows Alt Color Source A with the points drawn over it (without it, the missing layer fill)
            let mut params = vec!["display=2".to_string(), format!("draw_geom={g}")];
            params.extend(p(&DRAWN));
            let c = case("gap-drawgeom", format!("fgbg-source-geom{g}-d{depth}"), FGBG, "flatgrey", params).layer("alt_source_a", "flatgrey");
            out.push(c.depth(depth).notes(&["G3"]));
        }
    }
    out
}

/// The Blend view (2) of a one layer group of an FG stack: FG1 alone (Layers Output 2) and MAIN_BG alone (5).
fn debug() -> Vec<Family> {
    let mut out = Vec::new();
    for track in [1, 2] {
        for lo in [2, 5] {
            let params = vec!["display=3".into(), "debug_view_method=2".into(), format!("layers_output={lo}"), format!("track_frame={track}")];
            let c = case("gap-debug", format!("cv-blend-lo{lo}-track{track}"), CV, "fgdisc", params).layer("fg1_matte", "fgdisc_matte");
            out.push(c.notes(&["G4"]));
        }
    }
    out
}

fn tracker2() -> Vec<Family> {
    let pts: [(&str, &[&str]); 3] = [
        ("0pt", &[]),
        ("1pt", &["pt1_use=5", "pt1_pos=40,30@0;46,32@11"]),
        ("2pt", &["pt1_use=5", "pt1_pos=40,30@0;46,32@11", "pt2_use=5", "pt2_pos=20,15@0;18,20@11"]),
    ];
    let mut out = Vec::new();
    for vr in [1, 2] {
        for (tag, set) in pts {
            let mut params = vec!["main_bg_track=0".to_string(), format!("vector_resolution={vr}")];
            params.extend(p(set));
            out.push(case("gap-tracker2", format!("cv-track-off-{tag}-vr{vr}"), CV, "stilltex", params).notes(&["G5"]));
        }
    }
    out
}

fn wrap() -> Vec<Family> {
    let mut out = Vec::new();
    for w in 1..=3 {
        for blur in [0, 1] {
            let params = vec!["video_360=1".into(), HALF.into(), format!("warping={w}"), format!("blur_compensation={blur}")];
            out.push(case("gap-wrap", format!("fgbg-warping{w}-blur{blur}"), FGBG, "wraptex", params).notes(&["G6"]));
        }
    }
    // a layered render: the masks of the layers below FG1 wrap too
    for w in [1, 2] {
        let params = vec!["video_360=1".into(), HALF.into(), format!("warping={w}")];
        let c = case("gap-wrap", format!("fgbg-layered-warping{w}"), FGBG, "wraptex", params).layer("fg1_matte", "fgdisc_matte");
        out.push(c.notes(&["G6"]));
    }
    out.push(case("gap-wrap", "fgbg-pole-blur1-d32", FGBG, "wrappole", p(&["video_360=1", HALF, "blur_compensation=1"])).depth(32).notes(&["G6"]));
    out
}

fn tiny() -> Vec<Family> {
    let mut out = Vec::new();
    for side in 1..=5 {
        let size = (side, side);
        let label = format!("{side}x{side}");
        out.push(case("gap-tiny", format!("retime-{label}"), RETIME, "movegrad", p(&[HALF])).size(size));
        out.push(case("gap-tiny", format!("fgbg-{label}"), FGBG, "movegrad", p(&[HALF])).size(size));
        out.push(case("gap-tiny", format!("cv-{label}"), CV, "movegrad", Vec::new()).size(size));
        let vin = case("gap-tiny", format!("vin-{label}"), VIN, "movegrad", vec![HALF.into(), format!("vectors_max_displace={}", sources::FAMILY_VECTOR_SCALE)]);
        out.push(vin.layer("vectors_prev", "fgvec_prev").layer("vectors_next", "fgvec_next").size(size));
    }
    out
}

/// Keyed with ease handles (`iS/I`, `oS/I`: speed per second and influence in percent) on the
/// movegrad clip, whose top left pixel brightens by the same step every frame, so with No
/// Vectors and Blend the output reads back the source position of every frame.
fn keys() -> Vec<Family> {
    let frames = |keys: &str| vec!["time_remap_mode=2".to_string(), "vectors=1".into(), format!("frame_num={keys}")];
    let mut out = vec![
        case("gap-keys", "frame-num-linear-into-bezier", RETIME, "movegrad", frames("0@0o0/50;11@11")).settles(&["K1a"]),
        case("gap-keys", "frame-num-easy-ease", RETIME, "movegrad", frames("0@0e;11@11e")).settles(&["K1b"]),
        case("gap-keys", "frame-num-typed-33.33", RETIME, "movegrad", frames("0@0o0/33.33;11@11i0/33.33")).settles(&["K1b"]),
        case("gap-keys", "speed-handles", RETIME, "movegrad", p(&["vectors=1", "speed_percent=100@0o-200/40;25@11i0/60"])).settles(&["K1d"]),
        case("gap-keys", "sensitivity-overshoot", RETIME, "movegrad", p(&[HALF, "sensitivity=20@0o400/100;90@11"])).settles(&["K1c"]),
    ];
    for v in out.iter_mut() {
        v.depth = 16;
    }
    for (tag, keys) in [("point-path", "10,10@0o20/50;50,40@11"), ("point-still-segment", "30,20@0o10/50;30,20@11")] {
        let params = vec!["display=2".to_string(), "pt1_use=5".into(), format!("pt1_pos={keys}")];
        let c = case("gap-keys", tag, FGBG, "flatgrey", params).layer("alt_source_a", "flatgrey");
        out.push(c.depth(16).notes(&["G3"]).settles(&["K1e"]));
    }
    out
}

/// The cases of one family, in a fixed order. Empty for an unknown id.
pub fn cases_of(id: &str) -> Vec<Family> {
    match id {
        "gap-drawgeom" => drawgeom(),
        "gap-debug" => debug(),
        "gap-tracker2" => tracker2(),
        "gap-wrap" => wrap(),
        "gap-tiny" => tiny(),
        "gap-keys" => keys(),
        _ => Vec::new(),
    }
}

/// Every case of every family.
pub fn all() -> Vec<Family> {
    FAMILIES.iter().flat_map(|f| cases_of(f.id)).collect()
}

/// The family case a test set case was made from, found by its family and label; None for a
/// matrix case, or a family case this build does not know (an older or newer test set).
pub fn family_case(case: &Case) -> Option<Family> {
    let family = case.family.as_deref()?;
    cases_of(family).into_iter().find(|f| format!("{}-{}", f.gap, f.label) == case.label)
}

/// The gap ids a test set case settles: its own list, else its family's; empty for a matrix case
/// or an unknown family.
pub fn settles(case: &Case) -> Vec<&'static str> {
    match (family_case(case), case.family.as_deref().and_then(find)) {
        (Some(f), _) if !f.settles.is_empty() => f.settles,
        (_, Some(info)) => info.gaps.to_vec(),
        _ => Vec::new(),
    }
}

// ---- into the test set --------------------------------------------------------------------

/// Whether `value` is a value the param takes: a popup in range, a number of the right shape,
/// keyframes passing `fx_core::split_key`. The checks `vfx apply --check` makes in full are left
/// to it; this one catches a misspelt family before any file is rendered.
fn plausible(kind: &Kind, value: &str) -> bool {
    if value.contains('@') {
        return value.split(';').map(str::trim).filter(|k| !k.is_empty()).all(|k| fx_core::split_key(k).is_ok_and(|t| plausible(kind, t.value)));
    }
    let numbers = |n: usize| {
        let v: Vec<Option<f64>> = value.split(',').map(|t| t.trim().parse::<f64>().ok()).collect();
        v.len() == n && v.iter().all(|x| x.is_some_and(f64::is_finite))
    };
    match kind {
        Kind::Popup { choices, .. } => value.parse::<usize>().is_ok_and(|v| v >= 1 && v <= *choices),
        Kind::Checkbox { .. } => matches!(value, "0" | "1"),
        Kind::Int { min, max, .. } => value.parse::<i64>().is_ok_and(|v| v >= *min && v <= *max),
        Kind::Float { valid_min, valid_max, .. } => {
            value.parse::<f64>().is_ok_and(|v| v.is_finite() && valid_min.is_none_or(|lo| v >= lo) && valid_max.is_none_or(|hi| v <= hi))
        }
        Kind::Angle { .. } => numbers(1),
        Kind::Point { .. } => numbers(2),
        Kind::Color { .. } => numbers(3),
        Kind::Unknown(_) => false,
    }
}

/// Checks a family case against the effect `vfx list --json` reports: every param id exists and
/// takes its value, every layer id exists.
pub fn check(f: &Family, effect: &EffectInfo) -> Result<(), String> {
    for spec in &f.params {
        let (id, value) = spec.split_once('=').ok_or_else(|| format!("{} {}: `{spec}` is not id=value", f.gap, f.label))?;
        let param = effect.params.iter().find(|p| p.id == id).ok_or_else(|| format!("{} {}: {} has no param `{id}`", f.gap, f.label, effect.id))?;
        if !plausible(&param.kind, value) {
            return Err(format!("{} {}: `{spec}` is not a value of {} `{id}`", f.gap, f.label, effect.id));
        }
    }
    for (id, source) in &f.layers {
        if !effect.layers.iter().any(|l| l.id == *id) {
            return Err(format!("{} {}: {} has no layer `{id}`", f.gap, f.label, effect.id));
        }
        if sources::find_family(source).is_none() && sources::find_clip(source).is_none() {
            return Err(format!("{} {}: no clip source `{source}`", f.gap, f.label));
        }
    }
    if sources::find_family(f.main).is_none() && sources::find_clip(f.main).is_none() {
        return Err(format!("{} {}: no clip source `{}`", f.gap, f.label, f.main));
    }
    Ok(())
}

fn input(source: &str, (w, h): (usize, usize), depth: u32) -> InputFile {
    InputFile {
        file: cases::clip_path(source, w, h, depth, FRAMES),
        source: source.to_string(),
        width: w,
        height: h,
        depth,
        pix_fmt: sources::pix_fmt(depth).to_string(),
        frames: FRAMES,
    }
}

/// The test set case of a family case. The id follows the temporal cases' form,
/// `effect.label.source-WxH.dD.hash`, with the family id leading the label and in the hash.
pub fn to_case(f: &Family) -> Case {
    let (width, height) = f.size;
    let label = format!("{}-{}", f.gap, f.label);
    let layers: BTreeMap<String, String> = f.layers.iter().map(|(id, src)| (id.to_string(), input(src, f.size, f.depth).file)).collect();
    let layer_key: Vec<String> = layers.iter().map(|(k, v)| format!("{k}={v}")).collect();
    let key = format!(
        "family:{}|{}|{label}|{}|{}|{width}x{height}|{}|t{FRAMES}|{FPS}|{}",
        f.gap,
        f.effect,
        f.params.join(","),
        f.main,
        f.depth,
        layer_key.join(",")
    );
    let hash = cases::fnv1a(&key);
    let id = format!("{}.{}.{}-{width}x{height}.d{}.{hash:08x}", cases::safe(f.effect), cases::safe(&label), f.main, f.depth);
    Case {
        reference: format!("{id}.{}", sources::pix_fmt(f.depth)),
        input: input(f.main, f.size, f.depth).file,
        id,
        effect: f.effect.to_string(),
        label,
        params: f.params.clone(),
        depth: f.depth,
        width,
        height,
        source: f.main.to_string(),
        kind: CaseKind::Temporal,
        frames: FRAMES,
        fps: Some(FPS.to_string()),
        layers,
        family: Some(f.gap.to_string()),
    }
}

/// The family test set: the cases of the families `ids` whose effect `vfx` lists, in family
/// order, with the clip files they use (each once, sorted by name), and per family how many cases
/// it got. A family case whose effect is missing is left out; one the effect refuses (an unknown
/// param, layer or value) is an error.
#[derive(Debug)]
pub struct Built {
    pub cases: Vec<Case>,
    pub inputs: Vec<InputFile>,
    pub counts: Vec<(&'static str, usize)>,
}

pub fn build(effects: &[EffectInfo], ids: &[String]) -> Result<Built, String> {
    let mut cases = Vec::new();
    let mut files: BTreeMap<String, InputFile> = BTreeMap::new();
    let mut counts = Vec::new();
    for info in FAMILIES.iter().filter(|f| ids.iter().any(|i| i == f.id)) {
        let mut count = 0;
        for f in cases_of(info.id) {
            let Some(effect) = effects.iter().find(|e| e.id == f.effect) else { continue };
            check(&f, effect)?;
            for src in std::iter::once(f.main).chain(f.layers.iter().map(|(_, s)| *s)) {
                let file = input(src, f.size, f.depth);
                files.entry(file.file.clone()).or_insert(file);
            }
            cases.push(to_case(&f));
            count += 1;
        }
        counts.push((info.id, count));
    }
    Ok(Built { cases, inputs: files.into_values().collect(), counts })
}

/// The summary line `gen` prints: the family count and the cases per family.
pub fn summary(counts: &[(&str, usize)]) -> String {
    let total: usize = counts.iter().map(|(_, n)| n).sum();
    let each: Vec<String> = counts.iter().map(|(id, n)| format!("{id} {n}")).collect();
    format!("families: {} with {total} cases ({})", counts.len(), each.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::parse_list;
    use std::collections::HashSet;

    #[test]
    fn every_family_has_cases_unique_labels_and_a_known_gap() {
        let mut ids = HashSet::new();
        for info in FAMILIES {
            assert!(ids.insert(info.id), "{} twice", info.id);
            assert!(info.id.starts_with("gap-") && info.id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'), "{}", info.id);
            let cases = cases_of(info.id);
            assert!(!cases.is_empty(), "{} has no cases", info.id);
            let mut labels = HashSet::new();
            for c in &cases {
                assert_eq!(c.gap, info.id);
                assert!(labels.insert(c.label.clone()), "{}: {} twice", info.id, c.label);
                assert!(c.notes.iter().all(|g| info.gaps.contains(g) || *g == "G3"), "{}: {:?}", c.label, c.notes);
                assert!(c.label.is_ascii() && !c.label.contains(' '), "{}", c.label);
            }
            assert!(!info.gaps.is_empty() && info.gaps.iter().all(|g| TITLES.iter().any(|(id, _)| id == g)), "{}", info.id);
            for text in [info.settles, info.rule] {
                assert!(text.is_ascii() && !text.contains('\u{2013}') && !text.contains('\u{2014}') && !text.contains('|'), "{text}");
            }
        }
        assert!(cases_of("gap-nope").is_empty());
        let order: Vec<&str> = FAMILIES.iter().map(|f| f.id).collect();
        assert_eq!(
            order,
["gap-drawgeom", "gap-debug", "gap-tracker2", "gap-wrap", "gap-tiny", "gap-keys"]
        );
    }

    #[test]
    fn every_register_gap_and_the_observed_points_have_a_family() {
        let covered: HashSet<&str> = FAMILIES.iter().flat_map(|f| f.gaps.iter().copied()).collect();
        for (id, _) in TITLES {
            assert!(covered.contains(id), "{id} has no family");
        }
        assert_eq!(title("G3"), "Draw Geom drawing");
        assert_eq!(title("G4"), "one-layer Debug Blend view");
        assert_eq!(title("G6"), "solver, hole fill and blur wrap variants");
        assert_eq!(title("Z9"), "Z9");
        // only the observing families print nothing on purpose; every other case expects a note
        for info in FAMILIES {
            for c in cases_of(info.id) {
                if !info.observe_only {
                    assert!(!c.notes.is_empty(), "{} {} expects no note", info.id, c.label);
                }
            }
        }
        assert!(cases_of("gap-tiny").iter().all(|c| c.notes.is_empty()));
    }

    #[test]
    fn the_cases_follow_the_rules_table() {
        let labels = |id: &str| cases_of(id).into_iter().map(|c| c.label).collect::<Vec<_>>();
        let has = |id: &str, label: &str, param: &str| {
            let c = cases_of(id).into_iter().find(|c| c.label == label).unwrap_or_else(|| panic!("{id} {label}"));
            assert!(c.params.iter().any(|p| p == param), "{id} {label}: {:?} lacks {param}", c.params);
            c
        };
        // G3: retime-fgbg's Source display with Alt Color Source on flat grey, Draw Geom 2, 3 and 6, three sub pixel points,
        // every depth
        let depths: HashSet<u32> = cases_of("gap-drawgeom").iter().map(|c| c.depth).collect();
        assert_eq!(depths, HashSet::from([8, 16, 32]));
        assert_eq!(labels("gap-drawgeom").len(), 9);
        for c in cases_of("gap-drawgeom") {
            assert_eq!((c.effect, c.main), ("retime-fgbg", "flatgrey"));
            assert!(c.params.contains(&"display=2".to_string()) && c.params.iter().filter(|p| p.contains("_pos=")).count() == 3);
            assert_eq!(c.layers, [("alt_source_a", "flatgrey")]);
        }
        // G4: the Blend view of FG1 alone and MAIN_BG alone, both Track Frames, with the hard matte
        assert_eq!(labels("gap-debug"), ["cv-blend-lo2-track1", "cv-blend-lo5-track1", "cv-blend-lo2-track2", "cv-blend-lo5-track2"]);
        let blend = has("gap-debug", "cv-blend-lo5-track2", "debug_view_method=2");
        assert!(blend.params.contains(&"layers_output=5".to_string()) && blend.layers == [("fg1_matte", "fgdisc_matte")]);
        // G5: 0, 1 and 2 points at both resolutions
        assert_eq!(labels("gap-tracker2").len(), 6);
        has("gap-tracker2", "cv-track-off-2pt-vr2", "main_bg_track=0");
        // G6: 360 Video, warping 1 to 3, blur 0 and 1 on wraptex, a layered render, a 32 bpc pole crossing with blur
        assert_eq!(labels("gap-wrap").len(), 9);
        assert!(cases_of("gap-wrap").iter().all(|c| c.params.contains(&"video_360=1".to_string())));
        assert_eq!(has("gap-wrap", "fgbg-layered-warping2", "warping=2").layers, [("fg1_matte", "fgdisc_matte")]);
        assert_eq!(cases_of("gap-wrap").iter().find(|c| c.main == "wrappole").map(|c| c.depth), Some(32));
        // G7: 1x1 to 5x5 in all four effects; K1: handles on frame_num and speed_percent
        let sizes: HashSet<(usize, usize)> = cases_of("gap-tiny").iter().map(|c| c.size).collect();
        assert_eq!(sizes, (1..=5).map(|s| (s, s)).collect());
        let effects: HashSet<&str> = cases_of("gap-tiny").iter().map(|c| c.effect).collect();
        assert_eq!(effects, HashSet::from(["retime", "retime-fgbg", "create-vectors", "vectors-in"]));
        let keyed: Vec<String> = cases_of("gap-keys").iter().flat_map(|c| c.params.clone()).filter(|p| p.contains('@')).collect();
        assert!(keyed.iter().any(|p| p.starts_with("frame_num=") && p.contains('o')));
        assert!(keyed.iter().any(|p| p.starts_with("speed_percent=") && p.contains('i')));
    }

    #[test]
    fn every_key_of_a_family_parses() {
        for c in all() {
            for spec in c.params.iter().filter(|s| s.contains('@')) {
                let (_, keys) = spec.split_once('=').unwrap();
                for k in keys.split(';') {
                    fx_core::split_key(k).unwrap_or_else(|e| panic!("{} {}: {k}: {e}", c.gap, c.label));
                }
            }
        }
    }

    #[test]
    fn ids_parse_with_all_and_unknown_ids_are_errors() {
        assert_eq!(parse_ids("all").unwrap().len(), FAMILIES.len());
        assert_eq!(parse_ids(" gap-wrap , gap-tiny,gap-wrap").unwrap(), ["gap-wrap", "gap-tiny"]);
        assert!(parse_ids("gap-wat").unwrap_err().contains("unknown family `gap-wat`"));
        assert!(parse_ids(",").is_err());
    }

    /// A slice of `vfx list --json` with the params and layers the families of create-vectors use.
    fn mini_list() -> Vec<EffectInfo> {
        parse_list(
            r#"[{"id":"create-vectors","kind":"temporal","params":[
              {"id":"display","name":"Display","kind":"popup","choices":["Vectors","Source","Debug"],"default":1},
              {"id":"debug_view_method","name":"Debug View Method","kind":"popup","choices":["Warp","Blend","Diff","Detail Compare","Edge Compare"],"default":2},
              {"id":"layers_output","name":"Layers Output","kind":"popup","choices":["All","FG1","FG2","FG3","MAIN_BG"],"default":1},
              {"id":"track_frame","name":"Track Frame","kind":"popup","choices":["Previous","Next"],"default":1}
             ],"layers":[{"id":"fg1_matte","name":"FG1: Matte","optional":true}]}]"#,
        )
        .unwrap()
    }

    #[test]
    fn build_keeps_the_listed_effects_and_refuses_a_setting_the_effect_lacks() {
        let built = build(&mini_list(), &["gap-debug".to_string(), "gap-tiny".to_string()]).unwrap();
        assert_eq!(built.counts, [("gap-debug", 4), ("gap-tiny", 5)], "only the create-vectors cases: the other effects are not listed");
        assert!(built.cases.iter().all(|c| c.effect == "create-vectors"));
        let files: Vec<&str> = built.inputs.iter().map(|f| f.file.as_str()).collect();
        assert_eq!(files.len(), 7, "{files:?}");
        assert_eq!(files[..2], ["inputs/fgdisc-64x48.d8.f12.rgba", "inputs/fgdisc_matte-64x48.d8.f12.rgba"]);
        let c = &built.cases[1];
        assert_eq!(c.family.as_deref(), Some("gap-debug"));
        assert_eq!(c.label, "gap-debug-cv-blend-lo5-track1");
        assert!(c.id.starts_with("create-vectors.gap-debug-cv-blend-lo5-track1.fgdisc-64x48.d8."), "{}", c.id);
        assert_eq!(c.reference, format!("{}.rgba", c.id));
        assert_eq!(c.layers["fg1_matte"], "inputs/fgdisc_matte-64x48.d8.f12.rgba");
        assert_eq!((c.frames, c.fps.as_deref(), c.kind), (12, Some("24"), CaseKind::Temporal));
        // a param the effect does not have is an error, so is a popup value out of range
        let err = build(&mini_list(), &["gap-tracker2".to_string()]).unwrap_err();
        assert!(err.contains("has no param `main_bg_track`"), "{err}");
        let f = case("gap-debug", "bad", CV, "fgdisc", vec!["layers_output=6".into()]);
        assert!(check(&f, &mini_list()[0]).unwrap_err().contains("is not a value"));
        let f = case("gap-debug", "bad", CV, "fgdisc", Vec::new()).layer("fg9_matte", "fgdisc_matte");
        assert!(check(&f, &mini_list()[0]).unwrap_err().contains("no layer `fg9_matte`"));
        let f = case("gap-debug", "bad", CV, "nosuch", Vec::new());
        assert!(check(&f, &mini_list()[0]).unwrap_err().contains("no clip source `nosuch`"));
        // ids are unique over every family, and stable
        let every: Vec<Case> = all().iter().map(to_case).collect();
        let ids: HashSet<&str> = every.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids.len(), every.len());
        assert_eq!(all().iter().map(to_case).collect::<Vec<_>>(), every);
    }

    #[test]
    fn values_are_checked_by_kind() {
        let float = Kind::Float { valid_min: Some(0.0), valid_max: Some(10.0), slider_min: None, slider_max: None, default: 0.0 };
        assert!(plausible(&float, "3.5") && !plausible(&float, "11") && !plausible(&float, "x"));
        assert!(plausible(&float, "2@0o5/50;9@11") && !plausible(&float, "2@0;12@11"));
        let popup = Kind::Popup { choices: 3, default: 1, names: vec![] };
        assert!(plausible(&popup, "3") && !plausible(&popup, "0") && !plausible(&popup, "4"));
        assert!(plausible(&Kind::Checkbox { default: true }, "0") && !plausible(&Kind::Checkbox { default: true }, "2"));
        assert!(plausible(&Kind::Point { default: [0.0, 0.0] }, "40.5,30") && !plausible(&Kind::Point { default: [0.0, 0.0] }, "40"));
        assert!(plausible(&Kind::Point { default: [0.0, 0.0] }, "10,10@0o20/50;50,40@11"));
    }

    #[test]
    fn a_test_set_case_finds_its_family_case_and_what_it_settles() {
        let every = all();
        for f in &every {
            let c = to_case(f);
            assert_eq!(family_case(&c).as_ref(), Some(f), "{}", c.id);
            let own = settles(&c);
            assert!(!own.is_empty() && own.iter().all(|g| find(f.gap).unwrap().gaps.contains(g)), "{}: {own:?}", c.id);
        }
        let keys: Vec<Vec<&str>> = cases_of("gap-keys").iter().map(|f| settles(&to_case(f))).collect();
        assert_eq!(keys, [vec!["K1a"], vec!["K1b"], vec!["K1b"], vec!["K1d"], vec!["K1c"], vec!["K1e"], vec!["K1e"]]);
        // a family this build lacks, a renamed case and a matrix case
        let mut c = to_case(&every[0]);
        c.label = "gap-drawgeom-renamed".into();
        assert_eq!((family_case(&c), settles(&c)), (None, vec!["G3"]));
        c.family = Some("gap-future".into());
        assert!(settles(&c).is_empty());
        c.family = None;
        assert!(family_case(&c).is_none() && settles(&c).is_empty());
    }

    #[test]
    fn the_summary_line_counts_families_and_cases() {
        assert_eq!(summary(&[("gap-a", 3), ("gap-b", 2)]), "families: 2 with 5 cases (gap-a 3, gap-b 2)");
    }
}
