//! `vfx list`: the registered effects, their kind, params and layers.

use crate::error::CliError;
use crate::registry::{self, Entry};
use fx_core::{EffectKind, LayerSpec, ParamKind, ParamSpec};
use serde::Serialize;
use std::io::Write;

/// Writes the listing. Hidden test effects are included only when `all` is set.
pub fn run(json: bool, all: bool, out: &mut impl Write) -> Result<(), CliError> {
    let entries = if all { registry::all() } else { registry::listed() };
    let text = if json { to_json(&entries) } else { to_table(&entries) };
    out.write_all(text.as_bytes())
        .and_then(|_| out.flush())
        .map_err(|e| CliError::io("write error", e))
}

#[derive(Serialize)]
struct EffectInfo {
    id: &'static str,
    name: &'static str,
    family: &'static str,
    /// `frame` or `temporal`.
    kind: EffectKind,
    /// True for the test effects that `vfx list` shows only with `--all`.
    hidden: bool,
    params: Vec<ParamSpec>,
    /// Extra input layers of a temporal effect; empty for a frame effect.
    layers: Vec<LayerSpec>,
    /// The open parity gaps the effect can note. Left out when it has none, so the entry of
    /// such an effect reads as it did before gaps existed.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    gaps: Vec<GapInfo>,
}

#[derive(Serialize)]
struct GapInfo {
    id: &'static str,
    title: &'static str,
    controls: &'static [&'static str],
}

pub fn to_json(entries: &[Entry]) -> String {
    let list: Vec<_> = entries
        .iter()
        .map(|e| EffectInfo {
            id: e.effect.id(),
            name: e.effect.name(),
            family: e.family,
            kind: e.effect.kind(),
            hidden: e.hidden,
            params: e.effect.params(),
            layers: e.effect.layers(),
            gaps: e.effect.gaps().iter().map(|g| GapInfo { id: g.id, title: g.title, controls: g.controls }).collect(),
        })
        .collect();
    let mut text = serde_json::to_string_pretty(&list).expect("effect list serializes");
    text.push('\n');
    text
}

fn kind_name(kind: EffectKind) -> &'static str {
    match kind {
        EffectKind::Frame => "frame",
        EffectKind::Temporal => "temporal",
    }
}

fn to_table(entries: &[Entry]) -> String {
    let mut rows = vec![strings(["id", "name", "family", "kind", "params", "layers"])];
    for e in entries {
        rows.push(vec![
            e.effect.id().to_string(),
            e.effect.name().to_string(),
            e.family.to_string(),
            kind_name(e.effect.kind()).to_string(),
            e.effect.params().len().to_string(),
            e.effect.layers().len().to_string(),
        ]);
    }
    let mut text = render(&rows);
    for e in entries {
        let params = e.effect.params();
        if !params.is_empty() {
            let mut rows = vec![strings(["param", "name", "kind", "range", "default"])];
            rows.extend(params.iter().map(|p| {
                let (kind, range, default) = describe(p);
                vec![p.id.to_string(), p.name.to_string(), kind, range, default]
            }));
            text.push_str(&format!("\n{} params\n", e.effect.id()));
            text.push_str(&render(&rows));
        }
        let layers = e.effect.layers();
        if !layers.is_empty() {
            let mut rows = vec![strings(["layer", "name", "optional"])];
            rows.extend(layers.iter().map(|l| vec![l.id.to_string(), l.name.to_string(), if l.optional { "yes" } else { "no" }.to_string()]));
            text.push_str(&format!("\n{} layers\n", e.effect.id()));
            text.push_str(&render(&rows));
        }
    }
    text
}

fn strings<const N: usize>(cells: [&str; N]) -> Vec<String> {
    cells.iter().map(|c| c.to_string()).collect()
}

/// Left aligned columns separated by two spaces, no trailing whitespace.
pub(crate) fn render(rows: &[Vec<String>]) -> String {
    let cols = rows.iter().map(Vec::len).max().unwrap_or(0);
    let widths: Vec<usize> = (0..cols)
        .map(|c| rows.iter().filter_map(|r| r.get(c)).map(|s| s.chars().count()).max().unwrap_or(0))
        .collect();
    let mut text = String::new();
    for row in rows {
        let line: Vec<String> = row.iter().enumerate().map(|(c, s)| format!("{s:<w$}", w = widths[c])).collect();
        text.push_str(line.join("  ").trim_end());
        text.push('\n');
    }
    text
}

/// Kind name, accepted range and default, as the CLI takes them.
fn describe(p: &ParamSpec) -> (String, String, String) {
    let (kind, range, default) = match &p.kind {
        ParamKind::Float { valid_min, valid_max, default, .. } => {
            ("float", format!("{valid_min}..{valid_max}"), default.to_string())
        }
        ParamKind::Int { min, max, default } => ("int", format!("{min}..{max}"), default.to_string()),
        ParamKind::Checkbox { default } => ("checkbox", "0|1".into(), u8::from(*default).to_string()),
        ParamKind::Color { default } => {
            ("color", "r,g,b 0..255".into(), format!("{},{},{}", default[0], default[1], default[2]))
        }
        ParamKind::Angle { default } => ("angle", "degrees".into(), default.to_string()),
        ParamKind::Point { default } => ("point", "x,y".into(), format!("{},{}", default[0], default[1])),
        ParamKind::Popup { choices, default } => {
            let list: Vec<String> = choices.iter().enumerate().map(|(i, c)| format!("{}={c}", i + 1)).collect();
            ("popup", list.join(" "), default.to_string())
        }
        // a kind added to fx-core after this table was written: show what it serialises to
        #[allow(unreachable_patterns)]
        other => {
            let v = serde_json::to_value(other).unwrap_or_default();
            return (v["kind"].as_str().unwrap_or("?").to_string(), String::new(), v["default"].to_string());
        }
    };
    (kind.to_string(), range, default)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(id: &'static str, kind: ParamKind) -> ParamSpec {
        ParamSpec { id, name: "Name", kind }
    }

    #[test]
    fn json_lists_every_effect_with_params_kind_and_layers() {
        let text = to_json(&registry::all());
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let list = v.as_array().unwrap();
        assert_eq!(list.len(), registry::all().len());
        for item in list {
            for key in ["id", "name", "family"] {
                assert!(item[key].is_string(), "{key}");
            }
            assert!(item["params"].is_array());
            assert!(item["layers"].is_array());
            assert!(item["hidden"].is_boolean());
            assert!(matches!(item["kind"].as_str(), Some("frame" | "temporal")), "{item}");
        }
        let find = |id: &str| list.iter().find(|e| e["id"] == id).unwrap_or_else(|| panic!("{id} missing"));
        assert_eq!(find("selftest-shift")["kind"], "temporal");
        assert_eq!(find("selftest-shift")["hidden"], true);
        assert_eq!(find("selftest-shift")["layers"][0], serde_json::json!({"id": "overlay", "name": "Overlay", "optional": true}));
        assert_eq!(find("selftest-invert")["kind"], "frame");
        assert_eq!(find("selftest-invert")["layers"], serde_json::json!([]));
        assert_eq!(find("retime")["hidden"], false);
    }

    #[test]
    fn an_entry_without_gaps_is_byte_identical_to_the_listing_before_gaps() {
        // the entry as it was before `gaps` existed, field for field
        #[derive(Serialize)]
        struct Before {
            id: &'static str,
            name: &'static str,
            family: &'static str,
            kind: EffectKind,
            hidden: bool,
            params: Vec<ParamSpec>,
            layers: Vec<LayerSpec>,
        }
        let plain: Vec<Entry> = registry::all().into_iter().filter(|e| e.effect.gaps().is_empty()).collect();
        assert!(plain.iter().any(|e| e.effect.id() == "retime") && plain.iter().any(|e| e.effect.id() == "selftest-invert"));
        let before: Vec<Before> = plain
            .iter()
            .map(|e| Before {
                id: e.effect.id(),
                name: e.effect.name(),
                family: e.family,
                kind: e.effect.kind(),
                hidden: e.hidden,
                params: e.effect.params(),
                layers: e.effect.layers(),
            })
            .collect();
        let mut want = serde_json::to_string_pretty(&before).unwrap();
        want.push('\n');
        assert_eq!(to_json(&plain), want);
    }

    #[test]
    fn gaps_come_last_as_id_title_and_controls_only_where_an_effect_has_some() {
        let text = to_json(&registry::all());
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let list = v.as_array().unwrap();
        let find = |id: &str| list.iter().find(|e| e["id"] == id).unwrap_or_else(|| panic!("{id} missing"));
        for id in ["retime", "selftest-invert", "selftest-shift", "selftest-integrate", "selftest-threads"] {
            assert!(find(id).get("gaps").is_none(), "{id}");
        }
        let gap = &find("selftest-gap")["gaps"];
        assert_eq!(gap, &serde_json::json!([{"id": "T1", "title": "notes plumbing test", "controls": ["gap"]}]));
        let ids = |id: &str| find(id)["gaps"].as_array().unwrap().iter().map(|g| g["id"].as_str().unwrap().to_string()).collect::<Vec<_>>();
        assert_eq!(ids("create-vectors"), ["G1", "G1b", "G1c", "G1d", "G2", "G3", "G4", "G5"]);
        assert_eq!(ids("retime-fgbg"), ["G1", "G1b", "G1c", "G1f", "G1g", "G1h", "G2", "G3", "G6"]);
        assert_eq!(ids("vectors-in"), ["G1", "G1c", "G1e", "G1f", "G1g", "G1h"]);
        // the top level keys of every entry in text order: `gaps` comes last, after `layers`
        let mut keys: Vec<Vec<&str>> = Vec::new();
        for line in text.lines() {
            if line == "  {" {
                keys.push(Vec::new());
            } else if let Some(rest) = line.strip_prefix("    \"") {
                keys.last_mut().unwrap().push(&rest[..rest.find('"').unwrap()]);
            }
        }
        assert_eq!(keys.len(), list.len());
        let before = ["id", "name", "family", "kind", "hidden", "params", "layers"];
        for (k, e) in keys.iter().zip(list) {
            let has = e.get("gaps").is_some();
            assert_eq!(k[..before.len()], before, "{}", e["id"]);
            let after: &[&str] = if has { &["gaps"] } else { &[] };
            assert_eq!(&k[before.len()..], after, "{}", e["id"]);
        }
    }

    #[test]
    fn every_gap_of_an_effect_names_controls_of_that_effect() {
        for e in registry::all() {
            let params: Vec<&str> = e.effect.params().iter().map(|p| p.id).collect();
            let layers: Vec<&str> = e.effect.layers().iter().map(|l| l.id).collect();
            let gaps = e.effect.gaps();
            assert!(gaps.windows(2).all(|w| w[0].id < w[1].id), "{}: gaps in id order", e.effect.id());
            for g in gaps {
                for c in g.controls {
                    assert!(params.contains(c) || layers.contains(c), "{}: {} names `{c}`, which it does not have", e.effect.id(), g.id);
                }
            }
        }
    }

    #[test]
    fn hidden_effects_need_all() {
        let ids = |all: bool| -> Vec<String> {
            let mut buf = Vec::new();
            run(true, all, &mut buf).unwrap();
            let v: serde_json::Value = serde_json::from_slice(&buf).unwrap();
            v.as_array().unwrap().iter().map(|e| e["id"].as_str().unwrap().to_string()).collect()
        };
        let (shown, everything) = (ids(false), ids(true));
        assert!(shown.iter().all(|id| !id.starts_with("selftest-")), "{shown:?}");
        assert!(everything.contains(&"selftest-shift".to_string()));
        assert!(everything.len() > shown.len());
        assert!(shown.iter().all(|id| everything.contains(id)));
    }

    #[test]
    fn the_table_has_a_kind_column_and_a_layers_section() {
        let text = to_table(&registry::all());
        let header = text.lines().next().unwrap();
        assert_eq!(header.split_whitespace().collect::<Vec<_>>(), ["id", "name", "family", "kind", "params", "layers"]);
        let shift = text.lines().find(|l| l.starts_with("selftest-shift")).unwrap();
        assert!(shift.contains("temporal"), "{shift}");
        let invert = text.lines().find(|l| l.starts_with("selftest-invert")).unwrap();
        assert!(invert.contains("frame"), "{invert}");
        assert!(text.contains("\nselftest-shift layers\nlayer    name     optional\noverlay  Overlay  yes\n"), "{text}");
        assert!(!text.contains("selftest-invert layers"));
    }

    #[test]
    fn param_json_uses_the_kind_tag() {
        let p = spec("amount", ParamKind::Int { min: 0, max: 9, default: 3 });
        let v = serde_json::to_value(&p).unwrap();
        assert_eq!(v["kind"], "int");
        assert_eq!(v["id"], "amount");
        assert_eq!(v["max"], 9);
    }

    #[test]
    fn describe_covers_every_kind() {
        let float = spec("a", ParamKind::Float { valid_min: 0.0, valid_max: 2.5, slider_min: 0.0, slider_max: 1.0, default: 1.0, precision: 1 });
        assert_eq!(describe(&float), ("float".to_string(), "0..2.5".to_string(), "1".to_string()));
        let popup = spec("b", ParamKind::Popup { choices: vec!["One".into(), "Two".into()], default: 2 });
        assert_eq!(describe(&popup), ("popup".to_string(), "1=One 2=Two".to_string(), "2".to_string()));
        let color = spec("c", ParamKind::Color { default: [1, 2, 3] });
        assert_eq!(describe(&color).2, "1,2,3");
        let check = spec("d", ParamKind::Checkbox { default: true });
        assert_eq!(describe(&check).2, "1");
    }

    #[test]
    fn table_has_no_trailing_spaces() {
        let text = render(&[strings(["a", "bb"]), strings(["ccc", "d"])]);
        assert_eq!(text, "a    bb\nccc  d\n");
    }
}
