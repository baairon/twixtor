//! Param values to try: one param at a time over its default, min, max and a few
//! interior points, plus all-default, all-min and all-max.

use crate::effects::{EffectInfo, Kind, Param};

/// Bounds beyond this count as "no practical limit"; the slider range is used instead.
const SANE: f64 = 1e9;

#[derive(Clone, Debug, PartialEq)]
pub struct Setting {
    pub label: String,
    /// `k=v` strings as the vfx CLI takes them. Empty means every param at its default.
    pub params: Vec<String>,
    /// One param moved on its own, as opposed to all-default, all-min and all-max.
    pub single: bool,
}

/// Shortest text that round trips, with -0 folded into 0.
fn exact(v: f64) -> String {
    let s = v.to_string();
    if s == "-0" {
        "0".into()
    } else {
        s
    }
}

/// Three decimals, trailing zeros trimmed. For computed interior points.
fn rounded(v: f64) -> String {
    let mut s = format!("{v:.3}");
    while s.ends_with('0') {
        s.pop();
    }
    if s.ends_with('.') {
        s.pop();
    }
    if s == "-0" {
        s = "0".into();
    }
    s
}

struct Probe {
    default: String,
    min: String,
    max: String,
    interior: Vec<String>,
}

fn quartiles(lo: f64, hi: f64) -> Vec<String> {
    (1..=3).map(|k| rounded((lo + (hi - lo) * k as f64 / 4.0).clamp(lo, hi))).collect()
}

fn probe(p: &Param, w: usize, h: usize) -> Option<Probe> {
    let usable = |v: Option<f64>| v.filter(|x| x.is_finite() && x.abs() <= SANE);
    Some(match &p.kind {
        Kind::Float { valid_min, valid_max, slider_min, slider_max, default } => {
            let lo = usable(*valid_min).or(usable(*slider_min)).unwrap_or(*default);
            let hi = usable(*valid_max).or(usable(*slider_max)).unwrap_or(*default);
            if lo > hi {
                return None;
            }
            // interior points follow the slider range where it sits inside the valid range
            let (a, b) = match (usable(*slider_min), usable(*slider_max)) {
                (Some(a), Some(b)) if a.max(lo) < b.min(hi) => (a.max(lo), b.min(hi)),
                _ => (lo, hi),
            };
            Probe { default: exact(*default), min: exact(lo), max: exact(hi), interior: quartiles(a, b) }
        }
        Kind::Int { min, max, default } => {
            let span = *max as i128 - *min as i128;
            let at = |k: i128| (*min as i128 + span * k / 4).to_string();
            Probe { default: default.to_string(), min: min.to_string(), max: max.to_string(), interior: vec![at(1), at(2), at(3)] }
        }
        Kind::Checkbox { default } => Probe {
            default: u8::from(*default).to_string(),
            min: "0".into(),
            max: "1".into(),
            interior: vec![],
        },
        Kind::Color { default } => Probe {
            default: format!("{},{},{}", default[0], default[1], default[2]),
            min: "0,0,0".into(),
            max: "255,255,255".into(),
            interior: vec!["255,0,0".into(), "0,255,0".into(), "0,0,255".into()],
        },
        Kind::Angle { default } => Probe {
            default: exact(*default),
            min: "-360".into(),
            max: "360".into(),
            interior: vec!["-180".into(), "0".into(), "180".into()],
        },
        Kind::Point { default } => {
            let (w, h) = (w as f64, h as f64);
            let at = |k: f64| format!("{},{}", rounded(w * k / 4.0), rounded(h * k / 4.0));
            Probe {
                default: format!("{},{}", exact(default[0]), exact(default[1])),
                min: "0,0".into(),
                max: format!("{},{}", rounded(w), rounded(h)),
                interior: vec![at(1.0), at(2.0), at(3.0)],
            }
        }
        Kind::Popup { choices, default, .. } => {
            if *choices == 0 {
                return None;
            }
            let n = *choices as i64;
            let interior = if n <= 8 {
                (2..n).map(|v| v.to_string()).collect()
            } else {
                (1..=3).map(|k| (1 + (n - 1) * k / 4).to_string()).collect()
            };
            Probe { default: default.to_string(), min: "1".into(), max: n.to_string(), interior }
        }
        Kind::Unknown(_) => return None,
    })
}

/// (role, value) pairs for one param: default, min, max, then interior points.
/// A value that repeats an earlier one is dropped.
pub fn values(p: &Param, w: usize, h: usize) -> Vec<(String, String)> {
    let Some(probe) = probe(p, w, h) else { return Vec::new() };
    let mut all = vec![("default".to_string(), probe.default), ("min".into(), probe.min), ("max".into(), probe.max)];
    all.extend(probe.interior.into_iter().enumerate().map(|(i, v)| (format!("i{}", i + 1), v)));
    let mut seen: Vec<String> = Vec::new();
    all.retain(|(_, v)| {
        let fresh = !seen.contains(v);
        if fresh {
            seen.push(v.clone());
        }
        fresh
    });
    all
}

/// Every param setting to run for one effect at one frame size.
///
/// `sweep` adds the one-at-a-time sweeps; without it only the three combined
/// settings are produced. A setting that would run the same params as an
/// earlier one is dropped.
pub fn settings(effect: &EffectInfo, w: usize, h: usize, sweep: bool) -> Vec<Setting> {
    let mut out = vec![Setting { label: "all-default".into(), params: Vec::new(), single: false }];
    let add = |out: &mut Vec<Setting>, label: String, params: Vec<String>, single: bool| {
        if !params.is_empty() && !out.iter().any(|s| s.params == params) {
            out.push(Setting { label, params, single });
        }
    };
    for role in ["min", "max"] {
        let params: Vec<String> = effect
            .params
            .iter()
            .filter_map(|p| {
                let probe = probe(p, w, h)?;
                let v = if role == "min" { probe.min } else { probe.max };
                Some(format!("{}={v}", p.id))
            })
            .collect();
        add(&mut out, format!("all-{role}"), params, false);
    }
    if sweep {
        for p in &effect.params {
            for (role, value) in values(p, w, h) {
                add(&mut out, format!("{}-{role}", p.id), vec![format!("{}={value}", p.id)], true);
            }
        }
    }
    out
}

/// Two key ramps for the params that animate between keys (float, angle and point):
/// `id=A@0;B@LAST`, with A and B the first and last interior points of the sweep, or
/// the default and the largest value when the interior collapsed into other values.
/// A clip needs two frames for a ramp.
pub fn ramps(effect: &EffectInfo, w: usize, h: usize, frames: usize) -> Vec<Setting> {
    if frames < 2 {
        return Vec::new();
    }
    let last = frames - 1;
    effect
        .params
        .iter()
        .filter(|p| matches!(p.kind, Kind::Float { .. } | Kind::Angle { .. } | Kind::Point { .. }))
        .filter_map(|p| {
            let vals = values(p, w, h);
            let get = |role: &str| vals.iter().find(|(r, _)| r == role).map(|(_, v)| v.clone());
            let (a, b) = match (get("i1"), get("i3")) {
                (Some(a), Some(b)) => (a, b),
                _ => (vals.first()?.1.clone(), get("max").or_else(|| vals.last().map(|(_, v)| v.clone()))?),
            };
            (a != b).then(|| Setting { label: format!("{}-ramp", p.id), params: vec![format!("{}={a}@0;{b}@{last}", p.id)], single: true })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn param(id: &str, kind: Kind) -> Param {
        Param { id: id.into(), name: id.into(), kind }
    }

    fn roles(p: &Param) -> Vec<(String, String)> {
        values(p, 320, 180)
    }

    fn pairs(v: &[(&str, &str)]) -> Vec<(String, String)> {
        v.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
    }

    #[test]
    fn float_uses_the_slider_range_for_interior_points_and_drops_repeats() {
        let p = param("a", Kind::Float { valid_min: Some(0.0), valid_max: Some(100.0), slider_min: Some(0.0), slider_max: Some(100.0), default: 50.0 });
        assert_eq!(roles(&p), pairs(&[("default", "50"), ("min", "0"), ("max", "100"), ("i1", "25"), ("i3", "75")]));
        let narrow = param("a", Kind::Float { valid_min: Some(-1000.0), valid_max: Some(1000.0), slider_min: Some(0.0), slider_max: Some(8.0), default: 1.5 });
        assert_eq!(
            roles(&narrow),
            pairs(&[("default", "1.5"), ("min", "-1000"), ("max", "1000"), ("i1", "2"), ("i2", "4"), ("i3", "6")])
        );
    }

    #[test]
    fn float_without_usable_valid_bounds_falls_back_to_the_slider() {
        let p = param("a", Kind::Float { valid_min: None, valid_max: Some(1e30), slider_min: Some(0.0), slider_max: Some(1.0), default: 0.25 });
        assert_eq!(roles(&p)[..3], pairs(&[("default", "0.25"), ("min", "0"), ("max", "1")])[..]);
    }

    #[test]
    fn float_values_always_stay_inside_the_valid_range() {
        let p = param("a", Kind::Float { valid_min: Some(0.0), valid_max: Some(0.0004), slider_min: Some(0.0), slider_max: Some(0.0004), default: 0.0 });
        for (_, v) in roles(&p) {
            let x: f64 = v.parse().unwrap();
            assert!((0.0..=0.0004).contains(&x), "{v}");
        }
    }

    #[test]
    fn int_checkbox_angle_color() {
        let int = param("n", Kind::Int { min: 0, max: 100, default: 10 });
        assert_eq!(roles(&int), pairs(&[("default", "10"), ("min", "0"), ("max", "100"), ("i1", "25"), ("i2", "50"), ("i3", "75")]));
        let tiny = param("n", Kind::Int { min: 0, max: 2, default: 1 });
        assert_eq!(roles(&tiny), pairs(&[("default", "1"), ("min", "0"), ("max", "2")]));
        let wide = param("n", Kind::Int { min: i64::MIN, max: i64::MAX, default: 0 });
        assert_eq!(roles(&wide).len(), 6);
        let check = param("c", Kind::Checkbox { default: true });
        assert_eq!(roles(&check), pairs(&[("default", "1"), ("min", "0")]));
        let angle = param("t", Kind::Angle { default: 45.0 });
        assert_eq!(
            roles(&angle),
            pairs(&[("default", "45"), ("min", "-360"), ("max", "360"), ("i1", "-180"), ("i2", "0"), ("i3", "180")])
        );
        let color = param("k", Kind::Color { default: [10, 20, 30] });
        assert_eq!(roles(&color)[0], ("default".to_string(), "10,20,30".to_string()));
        assert_eq!(roles(&color).len(), 6);
    }

    #[test]
    fn point_follows_the_frame_size() {
        let p = param("p", Kind::Point { default: [160.0, 90.0] });
        assert_eq!(
            roles(&p),
            pairs(&[("default", "160,90"), ("min", "0,0"), ("max", "320,180"), ("i1", "80,45"), ("i3", "240,135")])
        );
        assert_eq!(values(&p, 1, 1)[2].1, "1,1");
    }

    #[test]
    fn popup_lists_small_menus_whole_and_samples_big_ones() {
        let small = param("m", Kind::Popup { choices: 4, default: 2, names: vec![] });
        assert_eq!(roles(&small), pairs(&[("default", "2"), ("min", "1"), ("max", "4"), ("i2", "3")]));
        let big = param("m", Kind::Popup { choices: 20, default: 1, names: vec![] });
        assert_eq!(roles(&big), pairs(&[("default", "1"), ("max", "20"), ("i1", "5"), ("i2", "10"), ("i3", "15")]));
    }

    #[test]
    fn unknown_kinds_have_nothing_to_sweep() {
        assert!(roles(&param("u", Kind::Unknown("spline".into()))).is_empty());
    }

    fn effect(params: Vec<Param>) -> EffectInfo {
        EffectInfo::new("fx", params)
    }

    #[test]
    fn an_effect_without_params_gets_one_setting() {
        let s = settings(&effect(vec![]), 8, 8, true);
        assert_eq!(s, vec![Setting { label: "all-default".into(), params: vec![], single: false }]);
    }

    #[test]
    fn settings_cover_sweeps_and_the_three_combined_cases() {
        let e = effect(vec![
            param("amount", Kind::Int { min: 0, max: 100, default: 10 }),
            param("flip", Kind::Checkbox { default: false }),
        ]);
        let all = settings(&e, 8, 8, true);
        let labels: Vec<&str> = all.iter().map(|s| s.label.as_str()).collect();
        assert_eq!(
            labels,
            [
                "all-default", "all-min", "all-max", "amount-default", "amount-min", "amount-max", "amount-i1",
                "amount-i2", "amount-i3", "flip-default", "flip-max"
            ]
        );
        assert_eq!(all[1].params, ["amount=0", "flip=0"]);
        assert_eq!(all[2].params, ["amount=100", "flip=1"]);
        assert_eq!(all[4].params, ["amount=0"]);
        let singles: Vec<bool> = all.iter().map(|s| s.single).collect();
        assert_eq!(singles, [false, false, false, true, true, true, true, true, true, true, true]);

        let combos = settings(&e, 8, 8, false);
        let labels: Vec<&str> = combos.iter().map(|s| s.label.as_str()).collect();
        assert_eq!(labels, ["all-default", "all-min", "all-max"]);
    }

    #[test]
    fn a_single_param_effect_keeps_the_all_labels_and_drops_the_twin_sweeps() {
        let e = effect(vec![param("amount", Kind::Int { min: 0, max: 100, default: 10 })]);
        let labels: Vec<String> = settings(&e, 8, 8, true).into_iter().map(|s| s.label).collect();
        assert_eq!(labels, ["all-default", "all-min", "all-max", "amount-default", "amount-i1", "amount-i2", "amount-i3"]);
    }

    #[test]
    fn ramps_run_between_the_interior_points_over_the_whole_clip() {
        let e = effect(vec![
            param("speed", Kind::Float { valid_min: Some(0.0), valid_max: Some(100.0), slider_min: Some(0.0), slider_max: Some(100.0), default: 50.0 }),
            param("turn", Kind::Angle { default: 45.0 }),
            param("at", Kind::Point { default: [160.0, 90.0] }),
            param("n", Kind::Int { min: 0, max: 9, default: 1 }),
            param("on", Kind::Checkbox { default: false }),
            param("mode", Kind::Popup { choices: 3, default: 1, names: vec![] }),
            param("tint", Kind::Color { default: [1, 2, 3] }),
        ]);
        let ramps = ramps(&e, 320, 180, 12);
        let labels: Vec<&str> = ramps.iter().map(|s| s.label.as_str()).collect();
        assert_eq!(labels, ["speed-ramp", "turn-ramp", "at-ramp"], "only the params that interpolate");
        assert_eq!(ramps[0].params, ["speed=25@0;75@11"]);
        assert_eq!(ramps[1].params, ["turn=-180@0;180@11"]);
        assert_eq!(ramps[2].params, ["at=80,45@0;240,135@11"]);
        assert!(ramps.iter().all(|s| s.single));
        // a ramp needs two keys at different frames
        assert!(super::ramps(&e, 320, 180, 1).is_empty());
        assert_eq!(super::ramps(&e, 320, 180, 2)[0].params, ["speed=25@0;75@1"]);
    }

    #[test]
    fn a_ramp_falls_back_to_default_and_max_and_skips_a_flat_range() {
        // quarters of 0..0.0004 print as 0 at three decimals, so no interior point survives
        let tiny = effect(vec![param("a", Kind::Float { valid_min: Some(0.0), valid_max: Some(0.0004), slider_min: Some(0.0), slider_max: Some(0.0004), default: 0.0 })]);
        let r = ramps(&tiny, 8, 8, 5);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].params, ["a=0@0;0.0004@4"]);
        let flat = effect(vec![param("a", Kind::Float { valid_min: Some(1.0), valid_max: Some(1.0), slider_min: Some(1.0), slider_max: Some(1.0), default: 1.0 })]);
        assert!(ramps(&flat, 8, 8, 5).is_empty());
    }

    #[test]
    fn a_degenerate_effect_does_not_repeat_itself() {
        let e = effect(vec![param("n", Kind::Int { min: 3, max: 3, default: 3 })]);
        let all = settings(&e, 8, 8, true);
        let labels: Vec<&str> = all.iter().map(|s| s.label.as_str()).collect();
        assert_eq!(labels, ["all-default", "all-min"]);
    }
}
