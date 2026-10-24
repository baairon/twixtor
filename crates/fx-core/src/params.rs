use crate::{FxError, Result};
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ParamKind {
    /// Float slider. `valid_*` is the hard range, `slider_*` the UI range.
    Float {
        valid_min: f64,
        valid_max: f64,
        slider_min: f64,
        slider_max: f64,
        default: f64,
        precision: u8,
    },
    /// Integer slider.
    Int { min: i64, max: i64, default: i64 },
    Checkbox { default: bool },
    /// RGB color, 0..=255 per channel.
    Color { default: [u8; 3] },
    /// Angle in degrees.
    Angle { default: f64 },
    /// Point in pixels relative to the frame.
    Point { default: [f64; 2] },
    /// Popup menu, values are 1-based.
    Popup { choices: Vec<String>, default: u32 },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ParamSpec {
    pub id: &'static str,
    pub name: &'static str,
    #[serde(flatten)]
    pub kind: ParamKind,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ParamValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    Color([u8; 3]),
    Point([f64; 2]),
}

impl ParamSpec {
    pub fn default_value(&self) -> ParamValue {
        match &self.kind {
            ParamKind::Float { default, .. } | ParamKind::Angle { default } => {
                ParamValue::Float(*default)
            }
            ParamKind::Int { default, .. } => ParamValue::Int(*default),
            ParamKind::Checkbox { default } => ParamValue::Bool(*default),
            ParamKind::Color { default } => ParamValue::Color(*default),
            ParamKind::Point { default } => ParamValue::Point(*default),
            ParamKind::Popup { default, .. } => ParamValue::Int(*default as i64),
        }
    }

    /// Validates a value against this spec's kind and range.
    pub fn check(&self, v: &ParamValue) -> Result<ParamValue> {
        let bad = |reason: String| FxError::BadParam { id: self.id.to_string(), reason };
        match (&self.kind, v) {
            (ParamKind::Float { valid_min, valid_max, .. }, ParamValue::Float(x)) => {
                if x < valid_min || x > valid_max || x.is_nan() {
                    return Err(bad(format!("{x} outside {valid_min}..={valid_max}")));
                }
                Ok(v.clone())
            }
            (ParamKind::Float { .. }, ParamValue::Int(x)) => self.check(&ParamValue::Float(*x as f64)),
            (ParamKind::Angle { .. }, ParamValue::Float(_)) => Ok(v.clone()),
            (ParamKind::Angle { .. }, ParamValue::Int(x)) => Ok(ParamValue::Float(*x as f64)),
            (ParamKind::Int { min, max, .. }, ParamValue::Int(x)) => {
                if x < min || x > max {
                    return Err(bad(format!("{x} outside {min}..={max}")));
                }
                Ok(v.clone())
            }
            (ParamKind::Checkbox { .. }, ParamValue::Bool(_)) => Ok(v.clone()),
            (ParamKind::Checkbox { .. }, ParamValue::Int(x)) if *x == 0 || *x == 1 => {
                Ok(ParamValue::Bool(*x == 1))
            }
            (ParamKind::Color { .. }, ParamValue::Color(_)) => Ok(v.clone()),
            (ParamKind::Point { .. }, ParamValue::Point(_)) => Ok(v.clone()),
            (ParamKind::Popup { choices, .. }, ParamValue::Int(x)) => {
                if *x < 1 || *x as usize > choices.len() {
                    return Err(bad(format!("{x} outside 1..={}", choices.len())));
                }
                Ok(v.clone())
            }
            _ => Err(bad(format!("wrong type {v:?}"))),
        }
    }

    /// Parses a CLI string such as `0.5`, `3`, `true`, `255,128,0` or `10,20`.
    pub fn parse(&self, s: &str) -> Result<ParamValue> {
        let bad = || FxError::BadParam { id: self.id.to_string(), reason: format!("cannot parse `{s}`") };
        let nums = |n: usize| -> Result<Vec<f64>> {
            let v: Vec<f64> = s.split(',').map(|p| p.trim().parse::<f64>()).collect::<std::result::Result<_, _>>().map_err(|_| bad())?;
            if v.len() == n { Ok(v) } else { Err(bad()) }
        };
        let v = match &self.kind {
            ParamKind::Float { .. } | ParamKind::Angle { .. } => ParamValue::Float(s.trim().parse().map_err(|_| bad())?),
            ParamKind::Int { .. } | ParamKind::Popup { .. } => ParamValue::Int(s.trim().parse().map_err(|_| bad())?),
            ParamKind::Checkbox { .. } => match s.trim() {
                "1" | "true" | "on" => ParamValue::Bool(true),
                "0" | "false" | "off" => ParamValue::Bool(false),
                _ => return Err(bad()),
            },
            ParamKind::Color { .. } => {
                let c = nums(3)?;
                if c.iter().any(|x| *x < 0.0 || *x > 255.0 || x.fract() != 0.0) {
                    return Err(bad());
                }
                ParamValue::Color([c[0] as u8, c[1] as u8, c[2] as u8])
            }
            ParamKind::Point { .. } => {
                let p = nums(2)?;
                ParamValue::Point([p[0], p[1]])
            }
        };
        self.check(&v)
    }
}

/// Resolved param values keyed by id.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Params(BTreeMap<String, ParamValue>);

impl Params {
    pub fn set(&mut self, id: &str, v: ParamValue) {
        self.0.insert(id.to_string(), v);
    }

    pub fn get(&self, id: &str) -> Option<&ParamValue> {
        self.0.get(id)
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(String::as_str)
    }

    pub fn float(&self, id: &str) -> f64 {
        match self.0.get(id) {
            Some(ParamValue::Float(x)) => *x,
            Some(ParamValue::Int(x)) => *x as f64,
            other => panic!("param `{id}` is not a float: {other:?}"),
        }
    }

    pub fn int(&self, id: &str) -> i64 {
        match self.0.get(id) {
            Some(ParamValue::Int(x)) => *x,
            other => panic!("param `{id}` is not an int: {other:?}"),
        }
    }

    pub fn bool(&self, id: &str) -> bool {
        match self.0.get(id) {
            Some(ParamValue::Bool(x)) => *x,
            other => panic!("param `{id}` is not a checkbox: {other:?}"),
        }
    }

    pub fn color(&self, id: &str) -> [u8; 3] {
        match self.0.get(id) {
            Some(ParamValue::Color(x)) => *x,
            other => panic!("param `{id}` is not a color: {other:?}"),
        }
    }

    pub fn point(&self, id: &str) -> [f64; 2] {
        match self.0.get(id) {
            Some(ParamValue::Point(x)) => *x,
            other => panic!("param `{id}` is not a point: {other:?}"),
        }
    }
}
