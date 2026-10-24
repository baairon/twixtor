//! The test matrix and its `cases.json` form.
//!
//! A frame effect gets stills: every setting crossed with source, size and depth.
//! A temporal effect gets small clips, the same settings plus keyframed ramps, a
//! layer mode and the layers it takes, capped per effect by a deterministic
//! selection that always keeps the combined settings and every parameter's
//! boundary values.

use crate::effects::{EffectInfo, EffectKind};
use crate::sources::{self, ClipSource, Source};
use crate::sweep::{self, Setting};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const FORMAT: u32 = 1;
pub const DEFAULT_SIZES: [(usize, usize); 5] = [(1, 1), (2, 3), (37, 23), (255, 129), (320, 180)];
pub const DEFAULT_DEPTHS: [u32; 3] = [8, 16, 32];

/// Clip sizes: one even and one odd, small enough that 12 frames at 32 bpc stay light.
pub const TEMPORAL_SIZES: [(usize, usize); 2] = [(16, 12), (37, 23)];
pub const DEFAULT_CLIP_FRAMES: usize = 12;
/// Rates the clip sources take in turn: an integer rate and an NTSC one, so both time bases run.
pub const DEFAULT_CLIP_FPS: [&str; 2] = ["24", "30000/1001"];
/// Rich RGB with motion, and a sliding alpha ramp.
pub const DEFAULT_CLIP_SWEEP_SOURCES: [&str; 2] = ["testsrc2", "alphaslide"];
pub const DEFAULT_MAX_CASES_PER_EFFECT: usize = 400;

fn one() -> usize {
    1
}

/// Which `vfx apply` form a case uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseKind {
    /// One still frame in, one out.
    #[default]
    Frame,
    /// A clip in (and layers), a clip out, with `--fps` and `--frames`.
    Temporal,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Case {
    /// Unique and stable: effect, setting, source, size, depth and a hash of them.
    pub id: String,
    pub effect: String,
    /// Which setting this is, for example `all-min`, `amount-i2` or `speed-ramp`.
    pub label: String,
    /// `k=v` strings exactly as `vfx apply --param` takes them. Empty means all defaults.
    /// A temporal case may hold keyframes, `k=V@F;V@F`.
    pub params: Vec<String>,
    pub depth: u32,
    pub width: usize,
    pub height: usize,
    pub source: String,
    /// Input frame or clip, relative to the test set directory.
    pub input: String,
    /// Where the reference frame or clip is expected, relative to the references directory.
    pub reference: String,
    #[serde(default)]
    pub kind: CaseKind,
    /// Frames in the input clip, and in the reference. 1 for a still.
    #[serde(default = "one")]
    pub frames: usize,
    /// Frame rate for `--fps`, for example `24` or `30000/1001`. Temporal cases only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<String>,
    /// Extra layer clips by layer id, relative to the test set directory. Temporal cases only.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub layers: BTreeMap<String, String>,
}

impl Case {
    pub fn is_temporal(&self) -> bool {
        self.kind == CaseKind::Temporal
    }

    /// Bytes of one frame of this case.
    pub fn frame_len(&self) -> usize {
        self.width * self.height * fx_core::Depth::from_bits(self.depth).map_or(16, |d| d.bytes_per_pixel())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputFile {
    pub file: String,
    pub source: String,
    pub width: usize,
    pub height: usize,
    pub depth: u32,
    pub pix_fmt: String,
    /// Frames in the file: 1 for a still.
    #[serde(default = "one")]
    pub frames: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TestSet {
    pub format: u32,
    pub ffmpeg: String,
    pub inputs: Vec<InputFile>,
    pub cases: Vec<Case>,
}

/// What the temporal effects get; the depths, `sweep` and the sizes given by the user
/// are shared with the stills through [`Plan`].
pub struct TemporalPlan {
    pub sizes: Vec<(usize, usize)>,
    pub sources: Vec<&'static ClipSource>,
    /// The clip sources the one-at-a-time sweeps and ramps run on.
    pub sweep_sources: Vec<&'static ClipSource>,
    /// Frames per clip.
    pub frames: usize,
    /// Rates, handed to the clip sources in turn by their place in [`sources::CLIP_SOURCES`].
    pub fps: Vec<String>,
    /// Cases kept per temporal effect; 0 keeps them all.
    pub max_cases_per_effect: usize,
}

impl Default for TemporalPlan {
    fn default() -> TemporalPlan {
        TemporalPlan {
            sizes: TEMPORAL_SIZES.to_vec(),
            sources: sources::CLIP_SOURCES.iter().collect(),
            sweep_sources: DEFAULT_CLIP_SWEEP_SOURCES.iter().filter_map(|n| sources::find_clip(n)).collect(),
            frames: DEFAULT_CLIP_FRAMES,
            fps: DEFAULT_CLIP_FPS.iter().map(|s| s.to_string()).collect(),
            max_cases_per_effect: DEFAULT_MAX_CASES_PER_EFFECT,
        }
    }
}

pub struct Plan {
    pub sizes: Vec<(usize, usize)>,
    pub sources: Vec<&'static Source>,
    pub depths: Vec<u32>,
    /// Add the one-at-a-time param sweeps. Without it only all-default, all-min
    /// and all-max are produced.
    pub sweep: bool,
    /// The sources the sweeps run on. all-default, all-min and all-max run on
    /// every source; sweeping each param on all of them would multiply the
    /// matrix without finding more.
    pub sweep_sources: Vec<&'static Source>,
    pub temporal: TemporalPlan,
}

/// Rich RGB, and varying alpha: between them the two cover what a param can touch.
pub const DEFAULT_SWEEP_SOURCES: [&str; 2] = ["noise", "alpha"];

impl Default for Plan {
    fn default() -> Plan {
        Plan {
            sizes: DEFAULT_SIZES.to_vec(),
            sources: sources::SOURCES.iter().collect(),
            depths: DEFAULT_DEPTHS.to_vec(),
            sweep: true,
            sweep_sources: DEFAULT_SWEEP_SOURCES.iter().filter_map(|n| sources::find(n)).collect(),
            temporal: TemporalPlan::default(),
        }
    }
}

fn fnv1a(text: &str) -> u32 {
    text.bytes().fold(0x811c_9dc5u32, |h, b| (h ^ b as u32).wrapping_mul(0x0100_0193))
}

fn safe(text: &str) -> String {
    text.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect()
}

fn input_path(source: &str, w: usize, h: usize, depth: u32) -> String {
    format!("inputs/{source}-{w}x{h}.d{depth}.{}", sources::pix_fmt(depth))
}

fn clip_path(source: &str, w: usize, h: usize, depth: u32, frames: usize) -> String {
    format!("inputs/{source}-{w}x{h}.d{depth}.f{frames}.{}", sources::pix_fmt(depth))
}

impl Plan {
    /// The frame size that gets the one-at-a-time param sweeps, the largest one.
    /// Every other size runs only all-default, all-min and all-max.
    pub fn sweep_size(&self) -> Option<(usize, usize)> {
        if self.sweep {
            self.sizes.iter().copied().max_by_key(|(w, h)| w * h)
        } else {
            None
        }
    }

    /// The still input frames, for the frame effects.
    pub fn inputs(&self) -> Vec<InputFile> {
        let mut out = Vec::new();
        for src in &self.sources {
            for &(width, height) in &self.sizes {
                for &depth in src.depths.iter().filter(|d| self.depths.contains(d)) {
                    out.push(InputFile {
                        file: input_path(src.name, width, height, depth),
                        source: src.name.to_string(),
                        width,
                        height,
                        depth,
                        pix_fmt: sources::pix_fmt(depth).to_string(),
                        frames: 1,
                    });
                }
            }
        }
        out
    }
}

/// The cases of a set of effects, with the clip files the temporal ones need.
pub struct Built {
    pub cases: Vec<Case>,
    /// Clip files used by the kept temporal cases (main inputs and layers), each once.
    pub clip_inputs: Vec<InputFile>,
}

/// Every case for the given effects: frame effects get each setting crossed with
/// source, size and depth; temporal effects get the capped clip plan.
pub fn build(effects: &[EffectInfo], plan: &Plan) -> Vec<Case> {
    build_all(effects, plan).cases
}

pub fn build_all(effects: &[EffectInfo], plan: &Plan) -> Built {
    let mut cases = Vec::new();
    let mut clips: BTreeMap<String, InputFile> = BTreeMap::new();
    for effect in effects {
        match effect.kind {
            EffectKind::Frame => frame_cases(effect, plan, &mut cases),
            EffectKind::Temporal => cases.extend(temporal_cases(effect, plan, &mut clips)),
        }
    }
    let mut clip_inputs: Vec<InputFile> = clips.into_values().collect();
    clip_inputs.sort_by(|a, b| a.file.cmp(&b.file));
    Built { cases, clip_inputs }
}

fn frame_cases(effect: &EffectInfo, plan: &Plan, cases: &mut Vec<Case>) {
    let sweep_size = plan.sweep_size();
    let swept: Vec<&'static Source> =
        plan.sources.iter().filter(|s| plan.sweep_sources.iter().any(|w| w.name == s.name)).copied().collect();
    let mut sizes = plan.sizes.clone();
    sizes.sort_by_key(|(w, h)| std::cmp::Reverse(w * h));
    for &(width, height) in &sizes {
        for setting in sweep::settings(effect, width, height, Some((width, height)) == sweep_size) {
            for src in if setting.single { &swept } else { &plan.sources } {
                for &depth in src.depths.iter().filter(|d| plan.depths.contains(d)) {
                    let key = format!(
                        "{}|{}|{}|{}|{width}x{height}|{depth}",
                        effect.id,
                        setting.label,
                        setting.params.join(","),
                        src.name
                    );
                    let id = format!(
                        "{}.{}.{}-{width}x{height}.d{depth}.{:08x}",
                        safe(&effect.id),
                        safe(&setting.label),
                        src.name,
                        fnv1a(&key)
                    );
                    cases.push(Case {
                        reference: format!("{id}.{}", sources::pix_fmt(depth)),
                        input: input_path(src.name, width, height, depth),
                        id,
                        effect: effect.id.clone(),
                        label: setting.label.clone(),
                        params: setting.params.clone(),
                        depth,
                        width,
                        height,
                        source: src.name.to_string(),
                        kind: CaseKind::Frame,
                        frames: 1,
                        fps: None,
                        layers: BTreeMap::new(),
                    });
                }
            }
        }
    }
}

// ---- temporal effects ---------------------------------------------------------------------

/// A temporal case that may still be cut by the per-effect cap.
struct Candidate {
    case: Case,
    /// 0: all-default, all-min, all-max, no-layers (always kept on every combination).
    /// 1: a parameter's min or max (kept at least once each, whatever the cap).
    /// 2: every other single setting (kept once each while the cap allows).
    tier: u8,
    hash: u32,
    needs: Vec<InputFile>,
}

fn tier_of(setting: &Setting) -> u8 {
    if !setting.single {
        0
    } else if setting.label.ends_with("-min") || setting.label.ends_with("-max") {
        1
    } else {
        2
    }
}

fn clip_input(src: &ClipSource, (w, h): (usize, usize), depth: u32, frames: usize) -> InputFile {
    InputFile {
        file: clip_path(src.name, w, h, depth, frames),
        source: src.name.to_string(),
        width: w,
        height: h,
        depth,
        pix_fmt: sources::pix_fmt(depth).to_string(),
        frames,
    }
}

fn clip_index(src: &ClipSource) -> usize {
    sources::CLIP_SOURCES.iter().position(|s| s.name == src.name).expect("a clip source is in the registry")
}

/// Keeps every tier 0 and tier 1 candidate, then one per label of tier 2 while the cap
/// allows, then fills up to `cap` in a fixed pseudo random order. The order of the
/// survivors is the order they came in.
fn select(cands: Vec<Candidate>, cap: usize) -> Vec<Candidate> {
    if cap == 0 || cands.len() <= cap {
        return cands;
    }
    let n = cands.len();
    let mut keep = vec![false; n];
    let mut count = 0;
    for (i, c) in cands.iter().enumerate() {
        if c.tier == 0 {
            keep[i] = true;
            count += 1;
        }
    }
    // one case per label of tiers 1 and 2, rotating over the combinations a label runs on
    let mut groups: Vec<(&str, u8, Vec<usize>)> = Vec::new();
    for (i, c) in cands.iter().enumerate().filter(|(_, c)| c.tier > 0) {
        match groups.iter_mut().find(|g| g.0 == c.case.label) {
            Some(g) => g.2.push(i),
            None => groups.push((&c.case.label, c.tier, vec![i])),
        }
    }
    for tier in [1u8, 2] {
        for (gi, (_, t, idxs)) in groups.iter().enumerate() {
            if *t != tier || (tier == 2 && count >= cap) {
                continue;
            }
            keep[idxs[gi % idxs.len()]] = true;
            count += 1;
        }
    }
    let mut rest: Vec<usize> = (0..n).filter(|i| !keep[*i]).collect();
    rest.sort_by_key(|&i| (cands[i].hash, i));
    for i in rest {
        if count >= cap {
            break;
        }
        keep[i] = true;
        count += 1;
    }
    cands.into_iter().zip(keep).filter(|(_, k)| *k).map(|(c, _)| c).collect()
}

fn temporal_cases(effect: &EffectInfo, plan: &Plan, clips: &mut BTreeMap<String, InputFile>) -> Vec<Case> {
    let t = &plan.temporal;
    let mut sizes = t.sizes.clone();
    sizes.sort_by_key(|(w, h)| std::cmp::Reverse(w * h));
    let sweep_size = if plan.sweep { t.sizes.iter().copied().max_by_key(|(w, h)| w * h) } else { None };
    let swept: Vec<&'static ClipSource> =
        t.sources.iter().filter(|s| t.sweep_sources.iter().any(|w| w.name == s.name)).copied().collect();
    let has_optional_layer = effect.layers.iter().any(|l| l.optional);
    let rates: Vec<&str> = if t.fps.is_empty() { vec!["24"] } else { t.fps.iter().map(String::as_str).collect() };

    let mut cands = Vec::new();
    for &size in &sizes {
        let (width, height) = size;
        let is_sweep = Some(size) == sweep_size;
        // (setting, whether the optional layers are given)
        let mut settings: Vec<(Setting, bool)> =
            sweep::settings(effect, width, height, is_sweep).into_iter().map(|s| (s, true)).collect();
        if is_sweep {
            settings.extend(sweep::ramps(effect, width, height, t.frames).into_iter().map(|s| (s, true)));
        }
        if has_optional_layer {
            settings.push((Setting { label: "no-layers".into(), params: Vec::new(), single: false }, false));
        }
        for (setting, optional_layers) in settings {
            for &src in if setting.single { &swept } else { &t.sources } {
                let main = clip_index(src);
                let fps = rates[main % rates.len()].to_string();
                for &depth in &plan.depths {
                    let mut needs = vec![clip_input(src, size, depth, t.frames)];
                    let mut layers = BTreeMap::new();
                    for (j, layer) in effect.layers.iter().enumerate() {
                        if layer.optional && !optional_layers {
                            continue;
                        }
                        let from = &sources::CLIP_SOURCES[(main + 1 + j) % sources::CLIP_SOURCES.len()];
                        let file = clip_input(from, size, depth, t.frames);
                        layers.insert(layer.id.clone(), file.file.clone());
                        needs.push(file);
                    }
                    let layer_key: Vec<String> = layers.iter().map(|(k, v)| format!("{k}={v}")).collect();
                    let key = format!(
                        "{}|{}|{}|{}|{width}x{height}|{depth}|t{}|{fps}|{}",
                        effect.id,
                        setting.label,
                        setting.params.join(","),
                        src.name,
                        t.frames,
                        layer_key.join(",")
                    );
                    let hash = fnv1a(&key);
                    let id = format!(
                        "{}.{}.{}-{width}x{height}.d{depth}.{hash:08x}",
                        safe(&effect.id),
                        safe(&setting.label),
                        src.name
                    );
                    cands.push(Candidate {
                        tier: tier_of(&setting),
                        hash,
                        needs,
                        case: Case {
                            reference: format!("{id}.{}", sources::pix_fmt(depth)),
                            input: clip_path(src.name, width, height, depth, t.frames),
                            id,
                            effect: effect.id.clone(),
                            label: setting.label.clone(),
                            params: setting.params.clone(),
                            depth,
                            width,
                            height,
                            source: src.name.to_string(),
                            kind: CaseKind::Temporal,
                            frames: t.frames,
                            fps: Some(fps.clone()),
                            layers,
                        },
                    });
                }
            }
        }
    }
    select(cands, t.max_cases_per_effect)
        .into_iter()
        .map(|c| {
            for file in c.needs {
                clips.entry(file.file.clone()).or_insert(file);
            }
            c.case
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::{Kind, LayerInfo, Param};
    use std::collections::HashSet;

    fn effect(id: &str, params: Vec<Param>) -> EffectInfo {
        EffectInfo::new(id, params)
    }

    fn amount() -> Param {
        Param { id: "amount".into(), name: "Amount".into(), kind: Kind::Int { min: 0, max: 100, default: 10 } }
    }

    fn gain() -> Param {
        Param {
            id: "gain".into(),
            name: "Gain".into(),
            kind: Kind::Float { valid_min: Some(-255.0), valid_max: Some(255.0), slider_min: Some(-16.0), slider_max: Some(16.0), default: 0.0 },
        }
    }

    fn layer(id: &str, optional: bool) -> LayerInfo {
        LayerInfo { id: id.into(), name: id.into(), optional }
    }

    fn many(n: usize) -> EffectInfo {
        let params = (0..n)
            .map(|i| {
                let kind = if i % 3 == 0 {
                    Kind::Int { min: 0, max: 20, default: 5 }
                } else {
                    Kind::Float { valid_min: Some(0.0), valid_max: Some(100.0), slider_min: Some(0.0), slider_max: Some(100.0), default: 50.0 }
                };
                Param { id: format!("p{i}"), name: format!("P{i}"), kind }
            })
            .collect();
        effect("big", params).temporal(vec![layer("alt", true), layer("matte", false)])
    }

    #[test]
    fn param_less_effects_get_one_case_per_source_size_and_depth() {
        let cases = build(&[effect("a", vec![])], &Plan::default());
        // four patterns at three depths, plus the float only range pattern, at five sizes
        assert_eq!(cases.len(), (4 * 3 + 1) * 5);
        assert!(cases.iter().all(|c| c.label == "all-default" && c.params.is_empty()));
        assert!(cases.iter().all(|c| c.source != "range" || c.depth == 32));
        assert!(cases.iter().all(|c| c.kind == CaseKind::Frame && c.frames == 1 && c.fps.is_none() && c.layers.is_empty()));
    }

    #[test]
    fn ids_and_reference_names_are_unique_and_file_safe() {
        let cases = build(&[effect("a-1", vec![amount()]), effect("b_2", vec![])], &Plan::default());
        let mut ids = HashSet::new();
        for c in &cases {
            assert!(ids.insert(c.id.clone()), "duplicate id {}", c.id);
            assert!(c.id.chars().all(|ch| ch.is_ascii_alphanumeric() || "-_.".contains(ch)), "{}", c.id);
            assert_eq!(c.reference, format!("{}.{}", c.id, sources::pix_fmt(c.depth)));
            assert!(c.input.starts_with("inputs/") && c.input.ends_with(sources::pix_fmt(c.depth)));
        }
    }

    #[test]
    fn sweeps_run_at_the_largest_size_only() {
        let cases = build(&[effect("a", vec![amount()])], &Plan::default());
        let single: Vec<_> = cases.iter().filter(|c| c.label.starts_with("amount-")).collect();
        assert!(!single.is_empty());
        assert!(single.iter().all(|c| (c.width, c.height) == (320, 180)));
        for size in DEFAULT_SIZES {
            let at: HashSet<_> = cases.iter().filter(|c| (c.width, c.height) == size).map(|c| c.label.as_str()).collect();
            for want in ["all-default", "all-min", "all-max"] {
                assert!(at.contains(want), "{want} missing at {size:?}");
            }
        }
    }

    #[test]
    fn gaining_a_param_changes_ids_but_keeps_the_rest() {
        let before = build(&[effect("a", vec![])], &Plan::default());
        let after = build(&[effect("a", vec![amount()])], &Plan::default());
        let ids: HashSet<_> = after.iter().map(|c| c.id.clone()).collect();
        // all-default has no params in either set, so those references stay valid
        for c in before {
            assert!(ids.contains(&c.id), "{}", c.id);
        }
    }

    #[test]
    fn plan_filters_narrow_the_matrix() {
        let noise = sources::find("noise").unwrap();
        let plan = Plan { sizes: vec![(2, 3)], sources: vec![noise], depths: vec![16], sweep: true, sweep_sources: vec![noise], ..Plan::default() };
        let cases = build(&[effect("a", vec![amount()])], &plan);
        assert!(cases.iter().all(|c| c.source == "noise" && c.depth == 16 && (c.width, c.height) == (2, 3)));
        assert_eq!(plan.inputs().len(), 1);
        assert_eq!(plan.inputs()[0].file, "inputs/noise-2x3.d16.rgba64le");
        let none = Plan { depths: vec![8], sources: vec![sources::find("range").unwrap()], ..Plan::default() };
        assert!(none.inputs().is_empty());
        assert!(build(&[effect("a", vec![])], &none).is_empty());
    }

    #[test]
    fn sweeps_run_on_fewer_sources_than_the_combined_settings() {
        let cases = build(&[effect("a", vec![amount()])], &Plan::default());
        let sources_of = |single: bool| -> HashSet<&str> {
            cases.iter().filter(|c| c.label.starts_with("amount-") == single).map(|c| c.source.as_str()).collect()
        };
        assert_eq!(sources_of(true), HashSet::from(["noise", "alpha"]));
        assert_eq!(sources_of(false), HashSet::from(["gradient", "bars", "noise", "alpha", "range"]));
        // widening the sweeps to every source is just a different plan
        let all = Plan { sweep_sources: sources::SOURCES.iter().collect(), ..Plan::default() };
        let wide = build(&[effect("a", vec![amount()])], &all);
        assert!(wide.len() > cases.len());
        // a sweep source the plan does not render is simply absent
        let narrow = Plan { sources: vec![sources::find("bars").unwrap()], ..Plan::default() };
        assert!(build(&[effect("a", vec![amount()])], &narrow).iter().all(|c| !c.label.starts_with("amount-")));
    }

    #[test]
    fn no_sweep_leaves_only_the_combined_settings() {
        let plan = Plan { sweep: false, ..Plan::default() };
        let cases = build(&[effect("a", vec![amount()])], &plan);
        let labels: HashSet<_> = cases.iter().map(|c| c.label.as_str()).collect();
        assert_eq!(labels, HashSet::from(["all-default", "all-min", "all-max"]));
    }

    #[test]
    fn json_round_trips() {
        let cases = build(&[effect("a", vec![amount()])], &Plan::default());
        let set = TestSet { format: FORMAT, ffmpeg: "ffmpeg test".into(), inputs: Plan::default().inputs(), cases };
        let text = serde_json::to_string_pretty(&set).unwrap();
        assert_eq!(serde_json::from_str::<TestSet>(&text).unwrap(), set);
        assert!(!text.contains('\u{2013}') && !text.contains('\u{2014}'));
    }

    #[test]
    fn an_older_cases_file_still_loads_as_stills() {
        let old = r#"{"format":1,"ffmpeg":"x","inputs":[{"file":"inputs/a","source":"noise","width":2,"height":3,"depth":8,"pix_fmt":"rgba"}],
            "cases":[{"id":"i","effect":"e","label":"all-default","params":[],"depth":8,"width":2,"height":3,"source":"noise","input":"inputs/a","reference":"r"}]}"#;
        let set: TestSet = serde_json::from_str(old).unwrap();
        let c = &set.cases[0];
        assert_eq!((c.kind, c.frames, c.fps.clone(), c.layers.len()), (CaseKind::Frame, 1, None, 0));
        assert_eq!(set.inputs[0].frames, 1);
        assert_eq!(c.frame_len(), 2 * 3 * 4);
    }

    // ---- temporal plans ----

    fn shift_like() -> EffectInfo {
        effect(
            "shifty",
            vec![Param { id: "offset".into(), name: "Offset".into(), kind: Kind::Int { min: -8, max: 8, default: 0 } }],
        )
        .temporal(vec![layer("overlay", true)])
    }

    #[test]
    fn a_temporal_effect_gets_clip_cases_with_frames_fps_and_layers() {
        let built = build_all(&[shift_like()], &Plan::default());
        assert!(!built.cases.is_empty());
        for c in &built.cases {
            assert_eq!(c.kind, CaseKind::Temporal);
            assert_eq!(c.frames, 12);
            assert!(matches!(c.fps.as_deref(), Some("24" | "30000/1001")));
            assert!(c.input.contains(".f12.") && c.input.starts_with("inputs/"), "{}", c.input);
            assert!(sources::find_clip(&c.source).is_some());
            assert!([(16, 12), (37, 23)].contains(&(c.width, c.height)));
            assert_eq!(c.reference, format!("{}.{}", c.id, sources::pix_fmt(c.depth)));
            if c.label == "no-layers" {
                assert!(c.layers.is_empty());
            } else {
                // the layer comes from another source than the main clip, in the same shape
                let l = &c.layers["overlay"];
                assert!(l.contains(&format!("-{}x{}.d{}.f12.", c.width, c.height, c.depth)) && !l.contains(&c.source), "{l}");
            }
        }
        // every clip file a case names is among the inputs to render, and nothing else is
        let files: HashSet<&str> = built.clip_inputs.iter().map(|f| f.file.as_str()).collect();
        let named: HashSet<&str> =
            built.cases.iter().flat_map(|c| std::iter::once(c.input.as_str()).chain(c.layers.values().map(String::as_str))).collect();
        assert_eq!(files, named);
        assert!(built.clip_inputs.iter().all(|f| f.frames == 12));
    }

    #[test]
    fn rates_alternate_between_the_clip_sources() {
        let cases = build(&[shift_like()], &Plan::default());
        let rate_of = |src: &str| -> HashSet<&str> {
            cases.iter().filter(|c| c.source == src).map(|c| c.fps.as_deref().unwrap()).collect()
        };
        assert_eq!(rate_of("testsrc2"), HashSet::from(["24"]));
        assert_eq!(rate_of("movegrad"), HashSet::from(["30000/1001"]));
        assert_eq!(rate_of("noisefx"), HashSet::from(["24"]));
        assert_eq!(rate_of("alphaslide"), HashSet::from(["30000/1001"]));
        // a narrowed source list keeps each source's rate
        let mut plan = Plan::default();
        plan.temporal.sources = vec![sources::find_clip("movegrad").unwrap()];
        assert!(build(&[shift_like()], &plan).iter().all(|c| c.fps.as_deref() == Some("30000/1001")));
    }

    #[test]
    fn combined_settings_run_everywhere_and_sweeps_on_the_sweep_sources_at_the_big_size() {
        let cases = build(&[shift_like()], &Plan::default());
        for label in ["all-default", "all-min", "all-max", "no-layers"] {
            let at: HashSet<(String, usize, u32)> =
                cases.iter().filter(|c| c.label == label).map(|c| (c.source.clone(), c.width, c.depth)).collect();
            assert_eq!(at.len(), 4 * 2 * 3, "{label} on 4 sources, 2 sizes, 3 depths");
        }
        let single: Vec<_> = cases.iter().filter(|c| c.label.starts_with("offset-")).collect();
        assert!(!single.is_empty());
        assert!(single.iter().all(|c| c.width == 37 && ["testsrc2", "alphaslide"].contains(&c.source.as_str())));
        let ids: HashSet<&str> = cases.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids.len(), cases.len(), "ids are unique");
    }

    #[test]
    fn float_params_get_keyframed_ramps_over_the_whole_clip() {
        let e = effect("g", vec![gain(), amount()]).temporal(vec![]);
        let cases = build(&[e], &Plan::default());
        let ramps: Vec<_> = cases.iter().filter(|c| c.label == "gain-ramp").collect();
        assert!(!ramps.is_empty());
        assert!(ramps.iter().all(|c| c.params == ["gain=-8@0;8@11"] && c.width == 37));
        assert!(cases.iter().all(|c| c.label != "amount-ramp"), "an int holds between keys, a ramp says nothing");
        // no sweep, no ramps; one frame clips cannot ramp
        let plan = Plan { sweep: false, ..Plan::default() };
        assert!(build(&[effect("g", vec![gain()]).temporal(vec![])], &plan).iter().all(|c| !c.label.contains("ramp")));
        let mut one_frame = Plan::default();
        one_frame.temporal.frames = 1;
        assert!(build(&[effect("g", vec![gain()]).temporal(vec![])], &one_frame).iter().all(|c| !c.label.contains("ramp")));
    }

    #[test]
    fn required_layers_are_always_given_and_optional_ones_can_be_left_out() {
        let e = effect("l", vec![]).temporal(vec![layer("matte", false), layer("alt", true)]);
        let cases = build(&[e], &Plan::default());
        assert!(cases.iter().filter(|c| c.label == "no-layers").all(|c| c.layers.keys().collect::<Vec<_>>() == ["matte"]));
        assert!(cases.iter().filter(|c| c.label == "all-default").all(|c| c.layers.keys().collect::<Vec<_>>() == ["alt", "matte"]));
        // an effect with only required layers has no no-layers setting
        let e = effect("r", vec![]).temporal(vec![layer("matte", false)]);
        assert!(build(&[e], &Plan::default()).iter().all(|c| c.label != "no-layers"));
        // a temporal effect without layers neither
        assert!(build(&[effect("n", vec![]).temporal(vec![])], &Plan::default()).iter().all(|c| c.label != "no-layers" && c.layers.is_empty()));
    }

    #[test]
    fn the_cap_keeps_the_combined_settings_and_every_boundary_value() {
        let e = many(40);
        let mut uncapped = Plan::default();
        uncapped.temporal.max_cases_per_effect = 0;
        let full = build(std::slice::from_ref(&e), &uncapped);
        assert!(full.len() > 1200, "{} cases uncapped", full.len());
        let capped = build(std::slice::from_ref(&e), &Plan::default());
        assert_eq!(capped.len(), 400);
        // every kept case is one of the full plan, in the same order
        let order: Vec<&str> = full.iter().map(|c| c.id.as_str()).collect();
        let mut last = 0;
        for c in &capped {
            let at = order.iter().position(|id| *id == c.id).expect("a capped case is in the full plan");
            assert!(at >= last, "order kept");
            last = at;
        }
        // all-default, all-min, all-max and no-layers are kept on every source, size and depth
        for label in ["all-default", "all-min", "all-max", "no-layers"] {
            let want = full.iter().filter(|c| c.label == label).count();
            assert_eq!(capped.iter().filter(|c| c.label == label).count(), want, "{label}");
        }
        // one case per parameter and boundary value
        for p in &e.params {
            for role in ["min", "max"] {
                let label = format!("{}-{role}", p.id);
                assert!(capped.iter().any(|c| c.label == label), "{label} missing");
            }
        }
        // every other label survives too while the budget allows
        let labels = |cases: &[Case]| cases.iter().map(|c| c.label.clone()).collect::<HashSet<_>>();
        assert_eq!(labels(&capped), labels(&full));
        // deterministic
        assert_eq!(build(std::slice::from_ref(&e), &Plan::default()), capped);
    }

    #[test]
    fn a_tiny_cap_still_keeps_the_required_cases() {
        let e = many(10);
        let mut plan = Plan::default();
        plan.temporal.max_cases_per_effect = 1;
        let cases = build(std::slice::from_ref(&e), &plan);
        let combined = cases.iter().filter(|c| ["all-default", "all-min", "all-max", "no-layers"].contains(&c.label.as_str())).count();
        assert_eq!(combined, 4 * 24);
        for p in &e.params {
            for role in ["min", "max"] {
                assert!(cases.iter().any(|c| c.label == format!("{}-{role}", p.id)), "{}-{role}", p.id);
            }
        }
        // the required part is 96 combined cases plus 20 boundary cases; nothing else
        assert_eq!(cases.len(), 4 * 24 + 20);
    }

    #[test]
    fn a_cap_above_the_plan_changes_nothing() {
        let e = shift_like();
        let mut big = Plan::default();
        big.temporal.max_cases_per_effect = 100_000;
        let mut none = Plan::default();
        none.temporal.max_cases_per_effect = 0;
        assert_eq!(build(std::slice::from_ref(&e), &big), build(std::slice::from_ref(&e), &none));
    }

    #[test]
    fn the_temporal_plan_follows_the_narrowing_flags() {
        let mut plan = Plan::default();
        plan.temporal.sizes = vec![(16, 12)];
        plan.depths = vec![8];
        plan.temporal.frames = 5;
        plan.temporal.sources = vec![sources::find_clip("noisefx").unwrap()];
        plan.temporal.fps = vec!["25".into()];
        let built = build_all(&[shift_like()], &plan);
        // the sweep sources are not among the rendered ones, so only the combined settings remain
        assert!(built.cases.iter().all(|c| ["all-default", "all-min", "all-max", "no-layers"].contains(&c.label.as_str())));
        assert!(built.cases.iter().all(|c| c.source == "noisefx" && c.depth == 8 && c.frames == 5 && c.fps.as_deref() == Some("25")));
        assert!(built.clip_inputs.iter().all(|f| f.file.contains(".d8.f5.rgba") && f.width == 16));
    }

    #[test]
    fn frame_and_temporal_effects_mix_and_stills_stay_stills() {
        let both = build(&[effect("f", vec![amount()]), shift_like()], &Plan::default());
        assert!(both.iter().filter(|c| c.effect == "f").all(|c| c.kind == CaseKind::Frame));
        assert!(both.iter().filter(|c| c.effect == "shifty").all(|c| c.kind == CaseKind::Temporal));
        let alone = build(&[effect("f", vec![amount()])], &Plan::default());
        let frame_of_both: Vec<&Case> = both.iter().filter(|c| c.effect == "f").collect();
        assert_eq!(frame_of_both.len(), alone.len());
        assert!(frame_of_both.iter().zip(&alone).all(|(a, b)| *a == b));
    }
}
