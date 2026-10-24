//! The effect list as `vfx list --json` prints it.
//!
//! Parsed by hand so a field a future param kind adds, or a bound that
//! serialised to null because it was infinite, does not break the harness.
//! `kind`, `hidden` and `layers` came later than the rest: an older `vfx`
//! that leaves them out gives frame effects without layers.

use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    Float {
        valid_min: Option<f64>,
        valid_max: Option<f64>,
        slider_min: Option<f64>,
        slider_max: Option<f64>,
        default: f64,
    },
    Int { min: i64, max: i64, default: i64 },
    Checkbox { default: bool },
    Color { default: [u8; 3] },
    Angle { default: f64 },
    Point { default: [f64; 2] },
    /// `names` are the choices as the interface shows them, the first one is number 1.
    Popup { choices: usize, default: i64, names: Vec<String> },
    /// A kind this harness does not know. It is exercised at its default only.
    Unknown(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub id: String,
    pub name: String,
    pub kind: Kind,
}

/// How `vfx` drives an effect: one frame in, one frame out, or a clip with other frames and layers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EffectKind {
    #[default]
    Frame,
    Temporal,
}

/// An extra input layer of a temporal effect.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerInfo {
    pub id: String,
    pub name: String,
    pub optional: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EffectInfo {
    pub id: String,
    pub name: String,
    pub family: String,
    pub kind: EffectKind,
    /// Listed by `vfx list --all` only (the self-test effects).
    pub hidden: bool,
    pub params: Vec<Param>,
    pub layers: Vec<LayerInfo>,
}

impl EffectInfo {
    /// A visible frame effect with no layers, the common shape.
    pub fn new(id: &str, params: Vec<Param>) -> EffectInfo {
        EffectInfo {
            id: id.into(),
            name: id.into(),
            family: "f".into(),
            kind: EffectKind::Frame,
            hidden: false,
            params,
            layers: Vec::new(),
        }
    }

    pub fn temporal(mut self, layers: Vec<LayerInfo>) -> EffectInfo {
        self.kind = EffectKind::Temporal;
        self.layers = layers;
        self
    }
}

fn text(v: &Value, key: &str) -> Result<String, String> {
    v[key].as_str().map(str::to_string).ok_or_else(|| format!("missing string `{key}`"))
}

fn num(v: &Value, key: &str) -> Result<f64, String> {
    v[key].as_f64().ok_or_else(|| format!("missing number `{key}`"))
}

fn int(v: &Value, key: &str) -> Result<i64, String> {
    v[key].as_i64().ok_or_else(|| format!("missing integer `{key}`"))
}

fn pair<const N: usize>(v: &Value, key: &str) -> Result<[f64; N], String> {
    let items = v[key].as_array().filter(|a| a.len() == N);
    let items = items.ok_or_else(|| format!("`{key}` should hold {N} numbers"))?;
    let mut out = [0.0; N];
    for (slot, item) in out.iter_mut().zip(items) {
        *slot = item.as_f64().ok_or_else(|| format!("`{key}` should hold {N} numbers"))?;
    }
    Ok(out)
}

fn parse_param(v: &Value) -> Result<Param, String> {
    let id = text(v, "id")?;
    let ctx = |e: String| format!("param `{id}`: {e}");
    let kind_name = text(v, "kind").map_err(ctx)?;
    let kind = match kind_name.as_str() {
        "float" => Kind::Float {
            valid_min: v["valid_min"].as_f64(),
            valid_max: v["valid_max"].as_f64(),
            slider_min: v["slider_min"].as_f64(),
            slider_max: v["slider_max"].as_f64(),
            default: num(v, "default").map_err(ctx)?,
        },
        "int" => Kind::Int {
            min: int(v, "min").map_err(ctx)?,
            max: int(v, "max").map_err(ctx)?,
            default: int(v, "default").map_err(ctx)?,
        },
        "checkbox" => Kind::Checkbox {
            default: v["default"].as_bool().ok_or_else(|| ctx("missing bool `default`".into()))?,
        },
        "color" => {
            let c = pair::<3>(v, "default").map_err(ctx)?;
            Kind::Color { default: [c[0] as u8, c[1] as u8, c[2] as u8] }
        }
        "angle" => Kind::Angle { default: num(v, "default").map_err(ctx)? },
        "point" => Kind::Point { default: pair::<2>(v, "default").map_err(ctx)? },
        "popup" => {
            let names: Vec<String> = v["choices"]
                .as_array()
                .ok_or_else(|| ctx("missing `choices`".into()))?
                .iter()
                .map(|c| c.as_str().map_or_else(|| c.to_string(), str::to_string))
                .collect();
            Kind::Popup { choices: names.len(), default: int(v, "default").map_err(ctx)?, names }
        }
        other => Kind::Unknown(other.to_string()),
    };
    Ok(Param { id: id.clone(), name: text(v, "name").unwrap_or(id), kind })
}

fn parse_kind(e: &Value, id: &str) -> Result<EffectKind, String> {
    match e["kind"].as_str() {
        None | Some("frame") => Ok(EffectKind::Frame),
        Some("temporal") => Ok(EffectKind::Temporal),
        Some(other) => Err(format!("effect `{id}`: unknown kind `{other}`, this harness knows frame and temporal")),
    }
}

fn parse_layers(e: &Value, id: &str) -> Result<Vec<LayerInfo>, String> {
    let Some(items) = e.get("layers").filter(|v| !v.is_null()) else { return Ok(Vec::new()) };
    let items = items.as_array().ok_or_else(|| format!("effect `{id}`: `layers` should be an array"))?;
    items
        .iter()
        .map(|l| {
            let layer_id = text(l, "id").map_err(|m| format!("effect `{id}`: layer: {m}"))?;
            Ok(LayerInfo {
                name: text(l, "name").unwrap_or_else(|_| layer_id.clone()),
                optional: l["optional"].as_bool().unwrap_or(false),
                id: layer_id,
            })
        })
        .collect()
}

pub fn parse_list(json: &str) -> Result<Vec<EffectInfo>, String> {
    let root: Value = serde_json::from_str(json).map_err(|e| format!("effect list is not JSON: {e}"))?;
    let list = root.as_array().ok_or("effect list should be a JSON array")?;
    list.iter()
        .map(|e| -> Result<EffectInfo, String> {
            let id = text(e, "id")?;
            let params = e["params"].as_array().ok_or_else(|| format!("effect `{id}`: missing `params`"))?;
            Ok(EffectInfo {
                name: text(e, "name").unwrap_or_else(|_| id.clone()),
                family: text(e, "family").unwrap_or_default(),
                kind: parse_kind(e, &id)?,
                hidden: e["hidden"].as_bool().unwrap_or(false),
                params: params
                    .iter()
                    .map(|p| parse_param(p).map_err(|m| format!("effect `{id}`: {m}")))
                    .collect::<Result<_, _>>()?,
                layers: parse_layers(e, &id)?,
                id,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_kind_and_null_bounds() {
        let list = parse_list(
            r#"[{"id":"fx-1","name":"One","family":"f","params":[
            {"id":"a","name":"A","kind":"float","valid_min":null,"valid_max":100.0,"slider_min":0.0,"slider_max":50.0,"default":10.0,"precision":1},
            {"id":"b","name":"B","kind":"int","min":-2,"max":5,"default":1},
            {"id":"c","name":"C","kind":"checkbox","default":true},
            {"id":"d","name":"D","kind":"color","default":[1,2,3]},
            {"id":"e","name":"E","kind":"angle","default":90.0},
            {"id":"f","name":"F","kind":"point","default":[1.5,2.0]},
            {"id":"g","name":"G","kind":"popup","choices":["x","y","z"],"default":2},
            {"id":"h","name":"H","kind":"spline","default":0}
            ]}]"#,
        )
        .unwrap();
        assert_eq!(list.len(), 1);
        let p = &list[0].params;
        assert_eq!(p.len(), 8);
        assert_eq!(
            p[0].kind,
            Kind::Float { valid_min: None, valid_max: Some(100.0), slider_min: Some(0.0), slider_max: Some(50.0), default: 10.0 }
        );
        assert_eq!(p[1].kind, Kind::Int { min: -2, max: 5, default: 1 });
        assert_eq!(p[3].kind, Kind::Color { default: [1, 2, 3] });
        assert_eq!(p[5].kind, Kind::Point { default: [1.5, 2.0] });
        assert_eq!(p[6].kind, Kind::Popup { choices: 3, default: 2, names: vec!["x".into(), "y".into(), "z".into()] });
        assert_eq!(p[7].kind, Kind::Unknown("spline".into()));
    }

    #[test]
    fn kind_hidden_and_layers_are_read_and_default_for_older_vfx() {
        let list = parse_list(
            r#"[{"id":"old","params":[]},
            {"id":"frame","kind":"frame","hidden":false,"layers":[],"params":[]},
            {"id":"t","name":"T","family":"selftest","kind":"temporal","hidden":true,"params":[],
             "layers":[{"id":"overlay","name":"Overlay","optional":true},{"id":"matte"}],"future_field":1}]"#,
        )
        .unwrap();
        assert_eq!((list[0].kind, list[0].hidden, list[0].layers.len()), (EffectKind::Frame, false, 0));
        assert_eq!(list[1].kind, EffectKind::Frame);
        assert_eq!((list[2].kind, list[2].hidden), (EffectKind::Temporal, true));
        assert_eq!(
            list[2].layers,
            vec![
                LayerInfo { id: "overlay".into(), name: "Overlay".into(), optional: true },
                LayerInfo { id: "matte".into(), name: "matte".into(), optional: false }
            ]
        );
        let err = parse_list(r#"[{"id":"x","kind":"audio","params":[]}]"#).unwrap_err();
        assert!(err.contains("unknown kind `audio`"), "{err}");
        assert!(parse_list(r#"[{"id":"x","params":[],"layers":{}}]"#).is_err());
        assert!(parse_list(r#"[{"id":"x","params":[],"layers":[{"name":"n"}]}]"#).is_err());
        assert_eq!(parse_list(r#"[{"id":"x","params":[],"layers":null}]"#).unwrap()[0].layers, vec![]);
    }

    #[test]
    fn constructors_build_the_common_shapes() {
        let e = EffectInfo::new("a", vec![]);
        assert_eq!((e.kind, e.hidden, e.layers.len()), (EffectKind::Frame, false, 0));
        let t = e.temporal(vec![LayerInfo { id: "l".into(), name: "L".into(), optional: false }]);
        assert_eq!((t.kind, t.layers.len()), (EffectKind::Temporal, 1));
    }

    #[test]
    fn bad_documents_are_errors() {
        assert!(parse_list("not json").is_err());
        assert!(parse_list("{}").is_err());
        assert!(parse_list(r#"[{"name":"x","params":[]}]"#).is_err());
        assert!(parse_list(r#"[{"id":"x","params":[{"id":"p","kind":"int","min":0}]}]"#).is_err());
        assert_eq!(parse_list("[]").unwrap(), vec![]);
    }
}
