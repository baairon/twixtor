//! Parameters that change over time.
//!
//! Retiming effects read some parameters at times other than the current one (the speed curve is
//! integrated from the start of the layer), so a temporal effect gets a [`ParamCurves`] it can
//! sample at any tick.

use crate::{FxError, ParamKind, ParamSpec, ParamValue, Params, Result, TimeBase};
use std::collections::BTreeMap;

/// One keyframe. `hold` keeps the value until the next key instead of
/// interpolating towards it. `ease` gives the curve zero slope at this key, so
/// the value leaves or reaches it gently (Easy Ease).
#[derive(Clone, Debug, PartialEq)]
pub struct Keyframe {
    /// Ticks since the start of the layer.
    pub time: i64,
    pub value: ParamValue,
    pub hold: bool,
    pub ease: bool,
}

/// The value of every parameter of one effect as a function of time. A
/// parameter without keyframes is constant. Between keyframes float, angle and
/// point values interpolate (linearly, or along an eased curve where a key is
/// eased), every other kind holds.
#[derive(Clone, Debug)]
pub struct ParamCurves {
    specs: Vec<ParamSpec>,
    constants: Params,
    tracks: BTreeMap<String, Vec<Keyframe>>,
}

impl ParamCurves {
    /// Every parameter at its default.
    pub fn new(specs: Vec<ParamSpec>) -> ParamCurves {
        let mut constants = Params::default();
        for s in &specs {
            constants.set(s.id, s.default_value());
        }
        ParamCurves { specs, constants, tracks: BTreeMap::new() }
    }

    /// Constant curves holding `params` (already resolved against `specs`).
    pub fn from_params(specs: Vec<ParamSpec>, params: &Params) -> Result<ParamCurves> {
        let mut curves = ParamCurves::new(specs);
        let ids: Vec<&str> = params.keys().collect();
        for id in ids {
            curves.set_constant(id, params.get(id).expect("key just listed").clone())?;
        }
        Ok(curves)
    }

    fn spec(&self, id: &str) -> Result<&ParamSpec> {
        self.specs.iter().find(|s| s.id == id).ok_or_else(|| FxError::UnknownParam(id.to_string()))
    }

    pub fn specs(&self) -> &[ParamSpec] {
        &self.specs
    }

    pub fn set_constant(&mut self, id: &str, value: ParamValue) -> Result<()> {
        let value = self.spec(id)?.check(&value)?;
        let spec_id = self.spec(id)?.id;
        self.tracks.remove(id);
        self.constants.set(spec_id, value);
        Ok(())
    }

    /// Replaces the curve of `id` by `keys`. Keys are sorted by time; two keys at the same tick are an error.
    pub fn set_keys(&mut self, id: &str, mut keys: Vec<Keyframe>) -> Result<()> {
        let spec = self.spec(id)?.clone();
        if keys.is_empty() {
            return Err(FxError::BadParam { id: id.to_string(), reason: "a curve needs at least one keyframe".into() });
        }
        for k in &mut keys {
            k.value = spec.check(&k.value)?;
        }
        keys.sort_by_key(|k| k.time);
        if keys.windows(2).any(|w| w[0].time == w[1].time) {
            return Err(FxError::BadParam { id: id.to_string(), reason: "two keyframes at the same time".into() });
        }
        self.tracks.insert(id.to_string(), keys);
        Ok(())
    }

    /// Reads a CLI value: a plain value (`0.5`, `255,128,0`) sets a constant;
    /// `V@F;V@F;...` sets keyframes with `F` the frame number (fractions
    /// allowed). A trailing `h` on the frame makes that key hold, as in
    /// `100@0h;25@24`; a trailing `e` eases it, as in `100@0e;25@24e` (a key is
    /// either held or eased, not both).
    pub fn parse_cli(&mut self, id: &str, text: &str, time: TimeBase) -> Result<()> {
        let spec = self.spec(id)?.clone();
        let bad = |reason: String| FxError::BadParam { id: id.to_string(), reason };
        if !text.contains('@') {
            return self.set_constant(id, spec.parse(text)?);
        }
        let mut keys = Vec::new();
        for piece in text.split(';') {
            let (value, frame) = piece.rsplit_once('@').ok_or_else(|| bad(format!("`{piece}` has no @frame")))?;
            let frame = frame.trim();
            let (number, hold, ease) = match (frame.strip_suffix('h'), frame.strip_suffix('e')) {
                (Some(f), _) => (f, true, false),
                (_, Some(f)) => (f, false, true),
                _ => (frame, false, false),
            };
            let number: f64 = number
                .parse()
                .map_err(|_| bad(format!("`{frame}` is not a frame number (a frame may end in h to hold or e to ease, not both)")))?;
            keys.push(Keyframe { time: time.ticks_of_frame(number), value: spec.parse(value)?, hold, ease });
        }
        self.set_keys(id, keys)
    }

    /// Value of `id` at `time` ticks.
    pub fn at(&self, id: &str, time: i64) -> Result<ParamValue> {
        let spec = self.spec(id)?;
        let Some(keys) = self.tracks.get(id) else {
            return Ok(self.constants.get(id).cloned().unwrap_or_else(|| spec.default_value()));
        };
        let last = keys.len() - 1;
        if time <= keys[0].time {
            return Ok(keys[0].value.clone());
        }
        if time >= keys[last].time {
            return Ok(keys[last].value.clone());
        }
        let i = keys.partition_point(|k| k.time <= time) - 1;
        let (a, b) = (&keys[i], &keys[i + 1]);
        if a.hold || !matches!(spec.kind, ParamKind::Float { .. } | ParamKind::Angle { .. } | ParamKind::Point { .. }) {
            return Ok(a.value.clone());
        }
        let f = eased_progress((time - a.time) as f64 / (b.time - a.time) as f64, a.ease, b.ease);
        let mix = |x: f64, y: f64| x + (y - x) * f;
        Ok(match (&a.value, &b.value) {
            (ParamValue::Float(x), ParamValue::Float(y)) => ParamValue::Float(mix(*x, *y)),
            (ParamValue::Point(x), ParamValue::Point(y)) => ParamValue::Point([mix(x[0], y[0]), mix(x[1], y[1])]),
            _ => a.value.clone(),
        })
    }

    /// Every parameter at `time` as a resolved [`Params`].
    pub fn params_at(&self, time: i64) -> Params {
        let mut out = Params::default();
        for s in &self.specs {
            out.set(s.id, self.at(s.id, time).expect("spec ids are known"));
        }
        out
    }

    /// Typed samples for hot loops. Panic on an unknown id or a value of the wrong kind, like [`Params`].
    pub fn float_at(&self, id: &str, time: i64) -> f64 {
        match self.at(id, time).unwrap_or_else(|e| panic!("{e}")) {
            ParamValue::Float(x) => x,
            ParamValue::Int(x) => x as f64,
            other => panic!("param `{id}` is not a float: {other:?}"),
        }
    }

    /// A float parameter as a 16.16 fixed point integer (`float_to_fixed` of the sample).
    pub fn fixed_at(&self, id: &str, time: i64) -> i32 {
        crate::float_to_fixed(self.float_at(id, time))
    }

    pub fn int_at(&self, id: &str, time: i64) -> i64 {
        match self.at(id, time).unwrap_or_else(|e| panic!("{e}")) {
            ParamValue::Int(x) => x,
            other => panic!("param `{id}` is not an int: {other:?}"),
        }
    }

    pub fn bool_at(&self, id: &str, time: i64) -> bool {
        match self.at(id, time).unwrap_or_else(|e| panic!("{e}")) {
            ParamValue::Bool(x) => x,
            other => panic!("param `{id}` is not a checkbox: {other:?}"),
        }
    }

    pub fn point_at(&self, id: &str, time: i64) -> [f64; 2] {
        match self.at(id, time).unwrap_or_else(|e| panic!("{e}")) {
            ParamValue::Point(x) => x,
            other => panic!("param `{id}` is not a point: {other:?}"),
        }
    }

    pub fn color_at(&self, id: &str, time: i64) -> [u8; 3] {
        match self.at(id, time).unwrap_or_else(|e| panic!("{e}")) {
            ParamValue::Color(x) => x,
            other => panic!("param `{id}` is not a color: {other:?}"),
        }
    }
}

/// Progress `u` (0 to 1) along a segment, reshaped by the easing of its two keys.
///
/// The segment is a cubic Hermite curve whose slope is zero at an eased key and
/// the slope of the straight line between the keys at a plain one. Both plain
/// is `u` itself; both eased is smoothstep; one eased starts or ends gently
/// while keeping the straight line's slope at the other key. Every case is
/// monotonic and runs exactly from 0 to 1, so a segment never overshoots.
fn eased_progress(u: f64, ease_start: bool, ease_end: bool) -> f64 {
    match (ease_start, ease_end) {
        (false, false) => u,
        (true, true) => u * u * (3.0 - 2.0 * u),
        (true, false) => u * u * (2.0 - u),
        (false, true) => u * (1.0 + u * (1.0 - u)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rational;

    fn float(id: &'static str, default: f64) -> ParamSpec {
        ParamSpec {
            id,
            name: id,
            kind: ParamKind::Float { valid_min: 0.0, valid_max: 1000.0, slider_min: 0.0, slider_max: 200.0, default, precision: 1 },
        }
    }

    fn popup(id: &'static str) -> ParamSpec {
        ParamSpec { id, name: id, kind: ParamKind::Popup { choices: vec!["A".into(), "B".into(), "C".into()], default: 1 } }
    }

    fn base() -> TimeBase {
        TimeBase::from_fps(Rational::new(24, 1).unwrap()).scaled(100)
    }

    fn curves() -> ParamCurves {
        ParamCurves::new(vec![float("speed", 100.0), popup("mode")])
    }

    #[test]
    fn untouched_parameters_are_their_defaults() {
        let c = curves();
        assert_eq!(c.float_at("speed", 0), 100.0);
        assert_eq!(c.float_at("speed", 123456), 100.0);
        assert_eq!(c.int_at("mode", -5), 1);
    }

    #[test]
    fn constants_are_validated() {
        let mut c = curves();
        c.set_constant("speed", ParamValue::Float(25.0)).unwrap();
        assert_eq!(c.float_at("speed", 10), 25.0);
        assert!(c.set_constant("speed", ParamValue::Float(-1.0)).is_err());
        assert!(c.set_constant("mode", ParamValue::Int(4)).is_err());
        assert!(c.set_constant("nope", ParamValue::Int(1)).is_err());
    }

    #[test]
    fn floats_interpolate_linearly_and_clamp_outside() {
        let mut c = curves();
        c.parse_cli("speed", "100@0;50@10", base()).unwrap();
        assert_eq!(c.float_at("speed", -100), 100.0);
        assert_eq!(c.float_at("speed", 0), 100.0);
        assert_eq!(c.float_at("speed", 500), 75.0);
        assert_eq!(c.float_at("speed", 1000), 50.0);
        assert_eq!(c.float_at("speed", 5000), 50.0);
    }

    #[test]
    fn hold_keys_and_non_float_kinds_step() {
        let mut c = curves();
        c.parse_cli("speed", "100@0h;50@10", base()).unwrap();
        assert_eq!(c.float_at("speed", 999), 100.0);
        assert_eq!(c.float_at("speed", 1000), 50.0);
        c.parse_cli("mode", "1@0;3@10", base()).unwrap();
        assert_eq!(c.int_at("mode", 999), 1);
        assert_eq!(c.int_at("mode", 1000), 3);
    }

    #[test]
    fn cli_keys_accept_fractions_and_any_order() {
        let mut c = curves();
        c.parse_cli("speed", "50@10;100@0.5", base()).unwrap();
        assert_eq!(c.float_at("speed", 50), 100.0);
        assert_eq!(c.float_at("speed", 1000), 50.0);
    }

    #[test]
    fn cli_errors_name_the_parameter() {
        let mut c = curves();
        assert!(c.parse_cli("speed", "100@0;100@0", base()).is_err());
        assert!(c.parse_cli("speed", "100@x", base()).is_err());
        assert!(c.parse_cli("speed", "100@0;50", base()).is_err());
        assert!(c.parse_cli("speed", "5000@0", base()).is_err());
        assert!(c.parse_cli("nope", "1", base()).is_err());
    }

    #[test]
    fn a_constant_replaces_keys_and_params_at_resolves_everything() {
        let mut c = curves();
        c.parse_cli("speed", "100@0;50@10", base()).unwrap();
        c.parse_cli("speed", "30", base()).unwrap();
        let p = c.params_at(777);
        assert_eq!(p.float("speed"), 30.0);
        assert_eq!(p.int("mode"), 1);
    }

    #[test]
    fn fixed_samples_follow_the_sdk_rounding() {
        let mut c = curves();
        assert_eq!(c.fixed_at("speed", 0), 6_553_600);
        c.parse_cli("speed", "33.3", base()).unwrap();
        assert_eq!(c.fixed_at("speed", 0), 2_182_349);
    }

    #[test]
    fn points_interpolate_per_component() {
        let spec = ParamSpec { id: "pt", name: "pt", kind: ParamKind::Point { default: [0.0, 0.0] } };
        let mut c = ParamCurves::new(vec![spec]);
        c.parse_cli("pt", "0,0@0;10,20@10", base()).unwrap();
        assert_eq!(c.point_at("pt", 500), [5.0, 10.0]);
    }

    /// 1000 ticks between the keys, so tick `t` is progress `t / 1000`.
    fn between(keys: &str) -> ParamCurves {
        let mut c = curves();
        c.parse_cli("speed", keys, base()).unwrap();
        c
    }

    #[test]
    fn eased_keys_reach_their_values_exactly_and_start_and_end_gently() {
        let c = between("100@0e;300@10e");
        assert_eq!(c.float_at("speed", 0), 100.0);
        assert_eq!(c.float_at("speed", 1000), 300.0);
        // smoothstep: halfway is exactly the middle, and the first percent of the time moves far less than a percent of the value
        assert_eq!(c.float_at("speed", 500), 200.0);
        assert!(c.float_at("speed", 10) - 100.0 < 0.1, "{}", c.float_at("speed", 10));
        assert!(300.0 - c.float_at("speed", 990) < 0.1);
    }

    #[test]
    fn eased_curves_are_monotonic_and_symmetric() {
        let up = between("100@0e;300@10e");
        let down = between("300@0e;100@10e");
        let mut last = f64::NEG_INFINITY;
        for t in 0..=1000 {
            let v = up.float_at("speed", t);
            assert!(v >= last, "tick {t}: {v} < {last}");
            last = v;
            assert!((v + up.float_at("speed", 1000 - t) - 400.0).abs() < 1e-9, "tick {t}");
            assert!((v - down.float_at("speed", 1000 - t)).abs() < 1e-9, "tick {t}");
        }
    }

    #[test]
    fn one_eased_key_is_flat_at_that_key_and_keeps_the_straight_slope_at_the_other() {
        let start = between("100@0e;200@10");
        let end = between("100@0;200@10e");
        // u * u * (2 - u) and u * (1 + u * (1 - u)) at u = 0.5
        assert_eq!(start.float_at("speed", 500), 137.5);
        assert_eq!(end.float_at("speed", 500), 162.5);
        // the plain end arrives with the slope of the straight line (100 over 1000 ticks)
        assert!((start.float_at("speed", 1000) - start.float_at("speed", 999) - 0.1).abs() < 1e-3);
        assert!((end.float_at("speed", 1) - end.float_at("speed", 0) - 0.1).abs() < 1e-3);
        // and the eased end is flat: the first tick moves 1/500 of what the straight line would
        assert!(start.float_at("speed", 1) - start.float_at("speed", 0) < 1e-3);
        assert!(end.float_at("speed", 1000) - end.float_at("speed", 999) < 1e-3);
        // each is the other's mirror image
        for t in 0..=1000 {
            assert!((start.float_at("speed", t) + end.float_at("speed", 1000 - t) - 300.0).abs() < 1e-9, "tick {t}");
        }
    }

    #[test]
    fn a_plain_segment_is_still_exactly_linear() {
        let c = between("100@0;50@10");
        assert_eq!(c.float_at("speed", 500), 75.0);
        // two equal eased keys stay constant, with no rounding noise
        assert_eq!(between("40@0e;40@10e").float_at("speed", 333), 40.0);
    }

    #[test]
    fn hold_wins_over_ease_and_other_kinds_still_step() {
        let c = between("100@0h;200@10e");
        assert_eq!(c.float_at("speed", 999), 100.0);
        assert_eq!(c.float_at("speed", 1000), 200.0);
        let mut c = curves();
        c.parse_cli("mode", "1@0e;3@10e", base()).unwrap();
        assert_eq!(c.int_at("mode", 999), 1);
        assert_eq!(c.int_at("mode", 1000), 3);
    }

    #[test]
    fn eased_points_interpolate_per_component() {
        let spec = ParamSpec { id: "pt", name: "pt", kind: ParamKind::Point { default: [0.0, 0.0] } };
        let mut c = ParamCurves::new(vec![spec]);
        c.parse_cli("pt", "0,0@0e;10,20@10e", base()).unwrap();
        assert_eq!(c.point_at("pt", 500), [5.0, 10.0]);
        assert!(c.point_at("pt", 100)[0] < 1.0);
    }

    #[test]
    fn a_key_cannot_be_held_and_eased_at_once() {
        let mut c = curves();
        for text in ["100@0he;50@10", "100@0eh;50@10", "100@e;50@10", "100@0ee;50@10"] {
            let err = c.parse_cli("speed", text, base()).unwrap_err().to_string();
            assert!(err.contains("speed") && err.contains("not a frame number"), "{text}: {err}");
        }
    }

    #[test]
    fn mixed_keys_ease_only_the_segments_they_touch() {
        // 100e -> 200 plain -> 100e: the first segment starts gently and meets the plain key at full slope, the second mirrors it
        let c = between("100@0e;200@10;100@20e");
        assert_eq!(c.float_at("speed", 1000), 200.0);
        assert_eq!(c.float_at("speed", 500), 137.5);
        assert_eq!(c.float_at("speed", 1500), 137.5);
    }
}
