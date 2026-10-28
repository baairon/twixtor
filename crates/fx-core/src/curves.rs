//! Parameters that change over time.
//!
//! Retiming effects read some parameters at times other than the current one (the speed curve is
//! integrated from the start of the layer), so a temporal effect gets a [`ParamCurves`] it can
//! sample at any tick.
//!
//! Precision: values are f64 and times are i64 ticks. A segment with no ease handle at either end
//! is one of the four closed forms of `eased_progress`, evaluated exactly as before handles
//! existed. A segment with a handle at either end is the cubic Bezier of After Effects' Keyframe
//! Velocity model (a host feature), solved for its time parameter by 64 bisection steps
//! in f64 using only `+ - * /`, in the order written in `bezier`. `tools/speed_curve.py` writes
//! the same operations in the same order, and `tests/fixtures/bezier_cases.hex` holds its values,
//! which this module reproduces bit for bit.

use crate::{FxError, ParamKind, ParamSpec, ParamValue, Params, Result, TimeBase};
use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

/// A Bezier ease handle on one side of a key: the speed the curve has at the key and how far
/// into the segment that speed reaches, as After Effects' Keyframe Velocity dialog sets them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ease {
    /// Parameter units per tick; for a point, pixels per tick along the straight line between
    /// its keys. The CLI takes units per second, which [`ParamCurves::parse_cli`] divides by the
    /// ticks per second, once.
    pub speed: f64,
    /// Percent of the segment's duration, 0.1 to 100.
    pub influence: f64,
}

/// One keyframe. `hold` keeps the value until the next key instead of
/// interpolating towards it. `ease` gives the curve zero slope at this key, so
/// the value leaves or reaches it gently (Easy Ease). `ease_in` and `ease_out`
/// are Bezier handles: `ease_in` shapes the segment that arrives at this key,
/// `ease_out` the one that leaves it.
#[derive(Clone, Debug, PartialEq)]
pub struct Keyframe {
    /// Ticks since the start of the layer.
    pub time: i64,
    pub value: ParamValue,
    pub hold: bool,
    pub ease: bool,
    pub ease_in: Option<Ease>,
    pub ease_out: Option<Ease>,
}

impl Keyframe {
    /// A plain key: no hold, no ease, no handles.
    pub fn new(time: i64, value: ParamValue) -> Keyframe {
        Keyframe { time, value, hold: false, ease: false, ease_in: None, ease_out: None }
    }
}

/// One `iS/I` or `oS/I` handle of a key, each number as written and as read: `S` the speed in
/// parameter units per second, `I` the influence in percent.
#[derive(Clone, Debug, PartialEq)]
pub struct HandleText<'a> {
    pub speed_text: &'a str,
    pub influence_text: &'a str,
    pub speed: f64,
    pub influence: f64,
}

/// One `V@F` piece of a keyed CLI value as [`split_key`] splits it. The value stays text,
/// because only the parameter knows how to read it.
#[derive(Clone, Debug, PartialEq)]
pub struct KeyText<'a> {
    /// Everything before the last `@`.
    pub value: &'a str,
    /// The frame number as written, without handles or suffix.
    pub frame_text: &'a str,
    pub frame: f64,
    pub hold: bool,
    pub ease: bool,
    pub ease_in: Option<HandleText<'a>>,
    pub ease_out: Option<HandleText<'a>>,
}

/// Splits one key `V@F[iS/I][oS/I][h|e]` of a keyed CLI value. `F` is a frame number as `f64`
/// reads it. `iS/I` is the incoming handle and `oS/I` the outgoing one, `S` a speed in parameter
/// units per second and `I` an influence of 0.1 to 100 percent, both matching
/// `-?[0-9]+(\.[0-9]+)?`, the speed at most 1e9 either way. A trailing `h` holds the key, a
/// trailing `e` eases it. A handle starts
/// at the first `i` or `o` followed by a digit or `-`, which no frame number contains, so a key
/// without handles splits exactly as before. A key with handles cannot end in `e`, and a hold key
/// takes only an incoming handle (Bezier in, Hold out). An error is the reason alone, without
/// the parameter.
pub fn split_key(piece: &str) -> std::result::Result<KeyText<'_>, String> {
    let (value, frame) = piece.rsplit_once('@').ok_or_else(|| format!("`{piece}` has no @frame"))?;
    let frame = frame.trim();
    let not_a_frame = || format!("`{frame}` is not a frame number (a frame may end in h to hold or e to ease, not both)");
    let Some(start) = handle_start(frame) else {
        let (number, hold, ease) = match (frame.strip_suffix('h'), frame.strip_suffix('e')) {
            (Some(f), _) => (f, true, false),
            (_, Some(f)) => (f, false, true),
            _ => (frame, false, false),
        };
        let at: f64 = number.parse().map_err(|_| not_a_frame())?;
        return Ok(KeyText { value, frame_text: number, frame: at, hold, ease, ease_in: None, ease_out: None });
    };
    let number = &frame[..start];
    let at: f64 = number.parse().map_err(|_| not_a_frame())?;
    let mut rest = &frame[start..];
    let mut ease_in = None;
    if rest.starts_with('i') {
        let (h, after) = handle(frame, rest)?;
        ease_in = Some(h);
        rest = after;
    }
    let mut ease_out = None;
    if rest.starts_with('o') && handle_start(rest) == Some(0) {
        let (h, after) = handle(frame, rest)?;
        ease_out = Some(h);
        rest = after;
    }
    let hold = match rest {
        "" => false,
        "h" => true,
        "e" => return Err("a key with ease handles cannot also end in e".into()),
        _ => return Err(not_a_handle(frame, rest)),
    };
    if hold && ease_out.is_some() {
        return Err("a held key has no outgoing handle".into());
    }
    Ok(KeyText { value, frame_text: number, frame: at, hold, ease: false, ease_in, ease_out })
}

/// The largest handle speed either way, in units per second (U: After Effects' own bound is not
/// known). It keeps a handle's reach `((S/k)*(I/100))*n` finite on any segment of i64 ticks, so a
/// handle never turns a value into NaN or infinity.
const HANDLE_SPEED: f64 = 1e9;

/// The index of the first `i` or `o` of `frame` that is followed by a digit or `-`.
fn handle_start(frame: &str) -> Option<usize> {
    let b = frame.as_bytes();
    (0..b.len()).find(|&i| matches!(b[i], b'i' | b'o') && matches!(b.get(i + 1), Some(b'0'..=b'9' | b'-')))
}

/// The handle at the start of `text` (an `i` or `o`, then every following character a handle
/// can hold: digits, `.`, `/` and `-`) and the text after it. `frame` is for the message.
fn handle<'a>(frame: &str, text: &'a str) -> std::result::Result<(HandleText<'a>, &'a str), String> {
    let len = 1 + text[1..].bytes().take_while(|b| matches!(b, b'0'..=b'9' | b'.' | b'/' | b'-')).count();
    let (token, after) = text.split_at(len);
    let bad = || not_a_handle(frame, token);
    let (speed_text, influence_text) = token[1..].split_once('/').ok_or_else(bad)?;
    if !is_number(speed_text) || !is_number(influence_text) {
        return Err(bad());
    }
    let speed: f64 = speed_text.parse().map_err(|_| bad())?;
    let influence: f64 = influence_text.parse().map_err(|_| bad())?;
    if !(-HANDLE_SPEED..=HANDLE_SPEED).contains(&speed) {
        return Err(format!("handle speed {speed_text} is outside -{HANDLE_SPEED}..={HANDLE_SPEED}"));
    }
    if !(0.1..=100.0).contains(&influence) {
        return Err(format!("influence {influence_text} is outside 0.1..=100"));
    }
    Ok((HandleText { speed_text, influence_text, speed, influence }, after))
}

fn not_a_handle(frame: &str, text: &str) -> String {
    format!("`{frame}`: `{text}` is not an ease handle (i or o, then SPEED/INFLUENCE, as in i0/33.3)")
}

/// `-?[0-9]+(\.[0-9]+)?`, the numbers of a handle.
fn is_number(text: &str) -> bool {
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let unsigned = text.strip_prefix('-').unwrap_or(text);
    match unsigned.split_once('.') {
        Some((int, frac)) => digits(int) && digits(frac),
        None => digits(unsigned),
    }
}

/// How a message names a kind whose value steps from key to key; None for the kinds that
/// interpolate (float, angle, point), the only ones a handle can shape.
fn stepping_kind(kind: &ParamKind) -> Option<&'static str> {
    match kind {
        ParamKind::Float { .. } | ParamKind::Angle { .. } | ParamKind::Point { .. } => None,
        ParamKind::Int { .. } => Some("an int"),
        ParamKind::Checkbox { .. } => Some("a checkbox"),
        ParamKind::Color { .. } => Some("a color"),
        ParamKind::Popup { .. } => Some("a popup"),
    }
}

/// The value of every parameter of one effect as a function of time. A
/// parameter without keyframes is constant. Between keyframes float, angle and
/// point values interpolate (linearly, along an eased curve where a key is
/// eased, or along a Bezier curve next to a handle), every other kind holds.
#[derive(Clone, Debug)]
pub struct ParamCurves {
    specs: Vec<ParamSpec>,
    constants: Params,
    tracks: BTreeMap<String, Vec<Keyframe>>,
    solved: Memo,
}

/// The values of Bezier segments already solved, by parameter index and tick.
///
/// Speed mode reads the speed at every half frame from frame 0 again for each output frame, and a
/// Bezier read costs a 64 step solve, so a long render would spend most of its time solving the
/// same ticks. `at` is a function of the keys alone. Keys come in only through `set_keys` (which
/// `parse_cli` calls), and it empties the memo first; a parameter set to a constant by
/// `set_constant` is read without the memo, and gets keys again only through `set_keys`. So a
/// value taken from here is the very value `bezier_at` would compute again: the memo changes no
/// output. It holds at most `MEMO_CAP` values and starts empty again when full; a clone starts
/// empty.
#[derive(Default)]
struct Memo(Mutex<HashMap<(usize, i64), ParamValue>>);

const MEMO_CAP: usize = 1 << 20;

impl Memo {
    fn get(&self, key: (usize, i64)) -> Option<ParamValue> {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).get(&key).cloned()
    }

    fn put(&self, key: (usize, i64), value: ParamValue) {
        let mut map = self.0.lock().unwrap_or_else(|p| p.into_inner());
        if map.len() >= MEMO_CAP {
            map.clear();
        }
        map.insert(key, value);
    }

    fn clear(&mut self) {
        self.0.get_mut().unwrap_or_else(|p| p.into_inner()).clear();
    }
}

impl Clone for Memo {
    fn clone(&self) -> Memo {
        Memo::default()
    }
}

impl std::fmt::Debug for Memo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Memo")
    }
}

impl ParamCurves {
    /// Every parameter at its default.
    pub fn new(specs: Vec<ParamSpec>) -> ParamCurves {
        let mut constants = Params::default();
        for s in &specs {
            constants.set(s.id, s.default_value());
        }
        ParamCurves { specs, constants, tracks: BTreeMap::new(), solved: Memo::default() }
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
    /// As in [`split_key`], a handle's influence is 0.1 to 100 percent, an eased key has no
    /// handle and a held key no outgoing one.
    pub fn set_keys(&mut self, id: &str, mut keys: Vec<Keyframe>) -> Result<()> {
        let spec = self.spec(id)?.clone();
        if keys.is_empty() {
            return Err(FxError::BadParam { id: id.to_string(), reason: "a curve needs at least one keyframe".into() });
        }
        for k in &mut keys {
            k.value = spec.check(&k.value)?;
        }
        for k in &keys {
            let reason = if k.ease && (k.ease_in.is_some() || k.ease_out.is_some()) {
                Some("a key with ease handles cannot also end in e".to_string())
            } else if k.hold && k.ease_out.is_some() {
                Some("a held key has no outgoing handle".to_string())
            } else {
                let outside = k.ease_in.iter().chain(&k.ease_out).find(|h| !(0.1..=100.0).contains(&h.influence));
                outside.map(|h| format!("influence {} is outside 0.1..=100", h.influence))
            };
            if let Some(reason) = reason {
                return Err(FxError::BadParam { id: id.to_string(), reason });
            }
        }
        keys.sort_by_key(|k| k.time);
        if keys.windows(2).any(|w| w[0].time == w[1].time) {
            return Err(FxError::BadParam { id: id.to_string(), reason: "two keyframes at the same time".into() });
        }
        self.solved.clear();
        self.tracks.insert(id.to_string(), keys);
        Ok(())
    }

    /// Reads a CLI value: a plain value (`0.5`, `255,128,0`) sets a constant;
    /// `V@F;V@F;...` sets keyframes with `F` the frame number (fractions
    /// allowed). A trailing `h` on the frame makes that key hold, as in
    /// `100@0h;25@24`; a trailing `e` eases it, as in `100@0e;25@24e` (a key is
    /// either held or eased, not both). Bezier handles go between the frame and
    /// the suffix, `iS/I` into the key and `oS/I` out of it, as in
    /// `100@0o0/50;25@24i0/50` ([`split_key`]); their speed is per second and is
    /// divided here by the ticks per second of `time`. Only floats, angles and
    /// points take handles: every other kind holds its value between keys.
    pub fn parse_cli(&mut self, id: &str, text: &str, time: TimeBase) -> Result<()> {
        let spec = self.spec(id)?.clone();
        let bad = |reason: String| FxError::BadParam { id: id.to_string(), reason };
        if !text.contains('@') {
            return self.set_constant(id, spec.parse(text)?);
        }
        let per_second = time.time_scale as f64;
        let ease = |h: Option<HandleText<'_>>| h.map(|h| Ease { speed: h.speed / per_second, influence: h.influence });
        let mut keys = Vec::new();
        for piece in text.split(';') {
            let key = split_key(piece).map_err(bad)?;
            if let (true, Some(kind)) = (key.ease_in.is_some() || key.ease_out.is_some(), stepping_kind(&spec.kind)) {
                return Err(bad(format!("`{}` has ease handles, but {kind} holds its value between keys", piece.trim())));
            }
            keys.push(Keyframe {
                time: time.ticks_of_frame(key.frame),
                value: spec.parse(key.value)?,
                hold: key.hold,
                ease: key.ease,
                ease_in: ease(key.ease_in),
                ease_out: ease(key.ease_out),
            });
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
        if a.ease_out.is_some() || b.ease_in.is_some() {
            return Ok(self.solved_at(id, spec, a, b, time));
        }
        let f = eased_progress((time - a.time) as f64 / (b.time - a.time) as f64, a.ease, b.ease);
        let mix = |x: f64, y: f64| x + (y - x) * f;
        Ok(match (&a.value, &b.value) {
            (ParamValue::Float(x), ParamValue::Float(y)) => ParamValue::Float(mix(*x, *y)),
            (ParamValue::Point(x), ParamValue::Point(y)) => ParamValue::Point([mix(x[0], y[0]), mix(x[1], y[1])]),
            _ => a.value.clone(),
        })
    }

    /// [`bezier_at`] through the memo, out of line so that the closed forms keep their fast path.
    #[cold]
    #[inline(never)]
    fn solved_at(&self, id: &str, spec: &ParamSpec, a: &Keyframe, b: &Keyframe, time: i64) -> ParamValue {
        let slot = (self.specs.iter().position(|s| s.id == id).expect("the spec was found"), time);
        if let Some(v) = self.solved.get(slot) {
            return v;
        }
        let v = bezier_at(&spec.kind, a, b, time);
        self.solved.put(slot, v.clone());
        v
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

/// The value at `time` (`a.time <= time < b.time`) on a segment with a handle at either end;
/// the key's own value at `a.time`. A float or an angle follows [`bezier`] from `a`'s value to
/// `b`'s. A float is then clamped to its valid range, which an overshooting handle can leave (U:
/// whether After Effects clamps there is not known, gap K1c; the rule is that a slider never
/// holds a value outside its range, so sensitivity cannot go negative mid render, compared with
/// `<` and `>` in that order so that NaN stays NaN as in `f64::clamp`). An angle is not clamped.
/// A point moves on the straight line from `a` to `b`: its distance along the line follows
/// [`bezier`] from 0 to the line's length `L = sqrt(dx*dx + dy*dy)`, with the handle speeds in
/// pixels per tick, and the point is `a + d * (y / L)` per component; when `L` is 0 it stays at
/// `a` (U: After Effects' own spatial path, gap K1e).
fn bezier_at(kind: &ParamKind, a: &Keyframe, b: &Keyframe, time: i64) -> ParamValue {
    if time == a.time {
        return a.value.clone();
    }
    let n = (b.time - a.time) as f64;
    let u = (time - a.time) as f64 / n;
    match (&a.value, &b.value) {
        (ParamValue::Float(va), ParamValue::Float(vb)) => {
            let y = bezier(u, n, *va, *vb - *va, a, b);
            ParamValue::Float(match *kind {
                ParamKind::Float { valid_min, valid_max, .. } => {
                    if y < valid_min {
                        valid_min
                    } else if y > valid_max {
                        valid_max
                    } else {
                        y
                    }
                }
                _ => y,
            })
        }
        (ParamValue::Point(pa), ParamValue::Point(pb)) => {
            let (dx, dy) = (pb[0] - pa[0], pb[1] - pa[1]);
            let l = (dx * dx + dy * dy).sqrt();
            if l == 0.0 {
                return a.value.clone();
            }
            let f = bezier(u, n, 0.0, l, a, b) / l;
            ParamValue::Point([pa[0] + dx * f, pa[1] + dy * f])
        }
        _ => a.value.clone(),
    }
}

/// The cubic Bezier of one segment `n` ticks long at progress `u` (0 to 1), from the value `va`
/// to `va + d3`, in the operation order `tools/speed_curve.py` shares.
///
/// The control points are (0, 0), (x1, d1), (x2, d2), (1, d3), in progress and value relative to
/// `va`. The outgoing end of `a`: a handle (speed S per tick, influence I) gives
/// `(I/100, (S*(I/100))*n)`; an eased key `(1/3, 0)`; a plain one `(1/3, d3/3)`. The incoming
/// end of `b`: a handle gives `(1 - I/100, d3 - (S*(I/100))*n)`; an eased key `(1 - 1/3, d3)`; a
/// plain one `(1 - 1/3, d3 - d3/3)`. So an eased end is a handle of speed 0 and a plain end one
/// of the straight line's speed, both at an influence of exactly 1/3 (U: After Effects' Easy
/// Ease and how it eases a Linear end next to a Bezier one, gaps K1a and K1b); in exact
/// arithmetic those give the four closed forms of `eased_progress`, which handle free segments
/// keep. The time curve is monotonic for every influence in 0.1 to 100, so the curve parameter
/// is found by 64 bisection steps from `lo = 0`, `hi = 1`: `m = (lo + hi) * 0.5`, `r = 1 - m`,
/// `x = ((3*r*r*m)*x1 + (3*r*m*m)*x2) + m*m*m`, `lo = m` if `x < u` else `hi = m`; then
/// `s = (lo + hi) * 0.5`, `r = 1 - s` and the value `va + (((3*r*r*s)*d1 + (3*r*s*s)*d2) +
/// (s*s*s)*d3)` (U: After Effects' own solve, gap K1d). Overshoot is allowed here.
fn bezier(u: f64, n: f64, va: f64, d3: f64, a: &Keyframe, b: &Keyframe) -> f64 {
    let (x1, d1) = match a.ease_out {
        Some(h) => (h.influence / 100.0, (h.speed * (h.influence / 100.0)) * n),
        None if a.ease => (1.0 / 3.0, 0.0),
        None => (1.0 / 3.0, d3 / 3.0),
    };
    let (x2, d2) = match b.ease_in {
        Some(h) => (1.0 - h.influence / 100.0, d3 - (h.speed * (h.influence / 100.0)) * n),
        None if b.ease => (1.0 - 1.0 / 3.0, d3),
        None => (1.0 - 1.0 / 3.0, d3 - d3 / 3.0),
    };
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..64 {
        let m = (lo + hi) * 0.5;
        let r = 1.0 - m;
        let x = ((3.0 * r * r * m) * x1 + (3.0 * r * m * m) * x2) + m * m * m;
        if x < u {
            lo = m;
        } else {
            hi = m;
        }
    }
    let s = (lo + hi) * 0.5;
    let r = 1.0 - s;
    va + (((3.0 * r * r * s) * d1 + (3.0 * r * s * s) * d2) + (s * s * s) * d3)
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

    // ---- Bezier handles -------------------------------------------------------------------------

    /// The value between two handle free keys exactly as it was computed before handles existed.
    fn old_value(a: (i64, f64, bool, bool), b: (i64, f64, bool), time: i64) -> f64 {
        let ((ta, va, hold, ease_a), (tb, vb, ease_b)) = (a, b);
        if hold {
            return va;
        }
        let u = (time - ta) as f64 / (tb - ta) as f64;
        let f = match (ease_a, ease_b) {
            (false, false) => u,
            (true, true) => u * u * (3.0 - 2.0 * u),
            (true, false) => u * u * (2.0 - u),
            (false, true) => u * (1.0 + u * (1.0 - u)),
        };
        va + (vb - va) * f
    }

    fn slider(id: &'static str, lo: f64, hi: f64) -> ParamSpec {
        ParamSpec { id, name: id, kind: ParamKind::Float { valid_min: lo, valid_max: hi, slider_min: lo, slider_max: hi, default: 0.0, precision: 3 } }
    }

    fn point(id: &'static str) -> ParamSpec {
        ParamSpec { id, name: id, kind: ParamKind::Point { default: [0.0, 0.0] } }
    }

    #[test]
    fn plain_segments_bit_identical() {
        // the middle one of four keys has no handle at either end, while every side the rules
        // allow on its neighbours has one, so it must stay the closed form of before, bit for bit
        let h = Some(Ease { speed: 0.37, influence: 41.5 });
        let pairs = [(100.0, 50.0), (-3.25, 7.0e-3), (0.0, 999.9), (12.5, 12.5), (-1000.0, 1000.0), (1.0 / 3.0, 2.0 / 3.0), (0.1, -0.2)];
        let shapes = [(false, false, false), (true, true, false), (true, false, false), (false, true, false), (false, false, true)];
        let mut checked = 0;
        for (va, vb) in pairs {
            for n in [1i64, 2, 3, 7, 100, 1001, 2400] {
                for (ease_a, ease_b, hold) in shapes {
                    let (ta, tb) = (40, 40 + n);
                    let keys = |a: ParamValue, b: ParamValue, edge: ParamValue| {
                        let first = Keyframe { ease_out: h, ..Keyframe::new(0, edge.clone()) };
                        let mut k1 = Keyframe { ease: ease_a, hold, ..Keyframe::new(ta, a) };
                        if !ease_a {
                            k1.ease_in = h;
                        }
                        let mut k2 = Keyframe { ease: ease_b, ..Keyframe::new(tb, b) };
                        if !ease_b {
                            k2.ease_out = h;
                        }
                        let last = Keyframe { ease_in: h, ..Keyframe::new(tb + 50, edge) };
                        vec![first, k1, k2, last]
                    };
                    let mut c = ParamCurves::new(vec![slider("v", -1000.0, 1000.0), point("p")]);
                    c.set_keys("v", keys(ParamValue::Float(va), ParamValue::Float(vb), ParamValue::Float(-7.0))).unwrap();
                    c.set_keys("p", keys(ParamValue::Point([va, vb]), ParamValue::Point([vb, -va]), ParamValue::Point([3.0, 4.0]))).unwrap();
                    let stride = if n > 200 { 7 } else { 1 };
                    for t in (ta..tb).step_by(stride) {
                        let want = old_value((ta, va, hold, ease_a), (tb, vb, ease_b), t);
                        assert_eq!(c.float_at("v", t).to_bits(), want.to_bits(), "{va} {vb} n {n} {ease_a} {ease_b} {hold} tick {t}");
                        let p = c.point_at("p", t);
                        let want = [old_value((ta, va, hold, ease_a), (tb, vb, ease_b), t), old_value((ta, vb, hold, ease_a), (tb, -va, ease_b), t)];
                        assert_eq!(p.map(f64::to_bits), want.map(f64::to_bits), "point {va} {vb} n {n} tick {t}");
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 10_000, "{checked}");
        // the same through the key grammar: every segment between two handle free ends is the old closed form
        let c = between("0@0o5/50;100@1i0/50;37.5@2.4;12.25@3.1e;80@4i3/20;5@5.25h;9@6o1/2;20@7.5");
        for (a, b) in [((100, 100.0, false, false), (240, 37.5, false)), ((240, 37.5, false, false), (310, 12.25, true)), ((400, 80.0, false, false), (525, 5.0, false)), ((525, 5.0, true, false), (600, 9.0, false))] {
            for t in a.0..b.0 {
                let want = old_value(a, b, t);
                assert_eq!(c.float_at("speed", t).to_bits(), want.to_bits(), "tick {t}");
            }
        }
    }

    #[test]
    fn thirds_reproduce_e() {
        // 2400 ticks a second, 100 to 300 over 10 frames: the straight line runs at 480 a second
        let third = "33.33333333333333";
        let cases = [
            (format!("100@0o0/{third};300@10i0/{third}"), "100@0e;300@10e"),
            (format!("100@0o480/{third};300@10i480/{third}"), "100@0;300@10"),
            (format!("100@0o0/{third};300@10i480/{third}"), "100@0e;300@10"),
            (format!("100@0o480/{third};300@10i0/{third}"), "100@0;300@10e"),
            // an eased or a plain end on the other side of a handle is the same model
            (format!("100@0e;300@10i0/{third}"), "100@0e;300@10e"),
            (format!("100@0e;300@10i480/{third}"), "100@0e;300@10"),
            (format!("100@0;300@10i0/{third}"), "100@0;300@10e"),
            (format!("100@0o0/{third};300@10e"), "100@0e;300@10e"),
            (format!("100@0o480/{third};300@10"), "100@0;300@10"),
        ];
        for (handles, closed) in cases {
            let (h, c) = (between(&handles), between(closed));
            let worst = (0..=1000).map(|t| (h.float_at("speed", t) - c.float_at("speed", t)).abs()).fold(0.0, f64::max);
            assert!(worst < 1e-9, "{handles}: {worst}");
            assert_eq!(h.float_at("speed", 0), 100.0, "the key's value exactly");
        }
        // two equal keys with handles of speed 0 stay exactly constant, as two equal eased keys do
        let c = between("40@0o0/70;40@10i-0/20.5");
        assert!((0..=1000).all(|t| c.float_at("speed", t) == 40.0));
        // influence 100 on both ends: halfway in time is halfway in value and the ends are flat. The
        // time curve has zero slope halfway there, so the solve there is good to a few millionths of the segment
        let c = between("100@0o0/100;300@10i0/100");
        assert!((c.float_at("speed", 500) - 200.0).abs() < 1e-2, "{}", c.float_at("speed", 500));
        assert!(c.float_at("speed", 1) - 100.0 < 1e-3);
        assert!((0..=1000).all(|t| (c.float_at("speed", t) + c.float_at("speed", 1000 - t) - 400.0).abs() < 1e-2));
    }

    #[test]
    fn overshoot_clamps() {
        // speed is valid 0..=1000: an outgoing handle of 60000 a second at 50% overshoots the next key
        let up = between("900@0o60000/50;1000@10");
        let values: Vec<f64> = (0..=1000).map(|t| up.float_at("speed", t)).collect();
        assert!(values.iter().all(|v| (900.0..=1000.0).contains(v)), "clamped to the valid range");
        let top = values.iter().filter(|v| **v == 1000.0).count();
        assert!(top > 500, "most of the segment sits on the top of the range: {top}");
        // the unclamped curve really leaves the range
        let (a, b) = (Keyframe { ease_out: Some(Ease { speed: 25.0, influence: 50.0 }), ..Keyframe::new(0, ParamValue::Float(900.0)) }, Keyframe::new(1000, ParamValue::Float(1000.0)));
        assert!(bezier(0.5, 1000.0, 900.0, 100.0, &a, &b) > 2000.0);
        let down = between("100@0o-60000/50;0@10");
        assert!((0..=1000).all(|t| (0.0..=100.0).contains(&down.float_at("speed", t))));
        assert_eq!(down.float_at("speed", 500), 0.0);
        // an angle is not clamped
        let mut c = ParamCurves::new(vec![ParamSpec { id: "a", name: "a", kind: ParamKind::Angle { default: 0.0 } }]);
        c.parse_cli("a", "0@0o60000/50;90@10", base()).unwrap();
        let peak = (0..1000).map(|t| c.float_at("a", t)).fold(f64::MIN, f64::max);
        assert!(peak > 1000.0, "{peak}");
        // a point overshoots along the line through its keys, and a point that does not move stays put
        let mut c = ParamCurves::new(vec![point("p"), point("q")]);
        c.parse_cli("p", "0,0@0o24000/100;30,40@10", base()).unwrap();
        c.parse_cli("q", "5,-5@0o2400/50;5,-5@10i-2400/50", base()).unwrap();
        let far = (0..1000).map(|t| c.point_at("p", t)).fold([0.0, 0.0], |m, p| if p[0] > m[0] { p } else { m });
        assert!(far[0] > 30.0 && far[1] > 40.0, "{far:?}");
        assert!((far[0] * 4.0 - far[1] * 3.0).abs() < 1e-9, "on the line: {far:?}");
        assert!((0..=1000).all(|t| c.point_at("q", t) == [5.0, -5.0]));
    }

    #[test]
    fn speed_is_per_second() {
        // 240 a second is 0.1 a tick at 2400 ticks a second: the key leaves with that slope
        let out = between("0@0o240/50;500@100");
        assert!((out.float_at("speed", 1) - out.float_at("speed", 0) - 0.1).abs() < 1e-3, "{}", out.float_at("speed", 1));
        let into = between("0@0;500@100i240/50");
        assert!((into.float_at("speed", 10_000) - into.float_at("speed", 9_999) - 0.1).abs() < 1e-3);
        // ten times as many ticks a second give the same curve over the same seconds
        let fine = TimeBase::from_fps(Rational::new(24, 1).unwrap()).scaled(1000);
        for text in ["0@0o240/50;500@100", "0@0;500@100i240/50", "20@0o-30/10;700@3.5i900/90;40@12.25o5/0.1;40@20"] {
            let coarse = between(text);
            let mut c = curves();
            c.parse_cli("speed", text, fine).unwrap();
            for t in (0..2100).step_by(7) {
                let (a, b) = (coarse.float_at("speed", t), c.float_at("speed", 10 * t));
                assert!((a - b).abs() < 1e-9 * (1.0 + a.abs()), "{text} tick {t}: {a} {b}");
            }
        }
        // a point moves S pixels a second along its line: 2400 a second is a pixel a tick
        let mut c = ParamCurves::new(vec![point("p")]);
        c.parse_cli("p", "0,0@0o2400/50;300,400@100", base()).unwrap();
        let p = c.point_at("p", 1);
        assert!((p[0] - 0.6).abs() < 1e-3 && (p[1] - 0.8).abs() < 1e-3, "{p:?}");
    }

    fn hex(text: &str) -> Vec<u8> {
        let digits: Vec<u8> = text.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
        digits.chunks(2).map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap()).collect()
    }

    #[test]
    fn matches_python_fixture() {
        // little endian: b"BEZ1", u32 cases, then per case u32 ticks a frame, u32 ticks a second,
        // u32 length and the key text, f64 limit (the slider is -limit..=limit), u32 count and
        // that many (i64 tick, f64 value) pairs, as tools/speed_curve.py evaluates them
        let bytes = hex(include_str!("../tests/fixtures/bezier_cases.hex"));
        let mut at = 0;
        let mut take = |n: usize| {
            at += n;
            &bytes[at - n..at]
        };
        assert_eq!(take(4), b"BEZ1");
        let u32_of = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap());
        let cases = u32_of(take(4));
        let mut compared = 0;
        for _ in 0..cases {
            let step = u32_of(take(4));
            let per_second = u32_of(take(4));
            let len = u32_of(take(4)) as usize;
            let text = std::str::from_utf8(take(len)).unwrap().to_string();
            let limit = f64::from_le_bytes(take(8).try_into().unwrap());
            let mut c = ParamCurves::new(vec![slider("speed", -limit, limit)]);
            c.parse_cli("speed", &text, TimeBase { time_scale: per_second, time_step: step }).unwrap();
            for _ in 0..u32_of(take(4)) {
                let tick = i64::from_le_bytes(take(8).try_into().unwrap());
                let want = f64::from_le_bytes(take(8).try_into().unwrap());
                let got = c.float_at("speed", tick);
                assert_eq!(got.to_bits(), want.to_bits(), "{text} at tick {tick}: {got} against {want}");
                compared += 1;
            }
        }
        assert_eq!(at, bytes.len(), "the whole fixture was read");
        assert!(cases >= 40 && compared >= 5000, "{cases} cases, {compared} values");
    }

    #[test]
    fn the_memo_changes_no_value() {
        let text = "20@0o-30/10;700@3.5i900/90;40@12.25o5/0.1;40@20";
        let warm = between(text);
        let first: Vec<u64> = (0..2100).map(|t| warm.float_at("speed", t).to_bits()).collect();
        let again: Vec<u64> = (0..2100).map(|t| warm.float_at("speed", t).to_bits()).collect();
        // a fresh curve read once per tick, backwards, holds the same bits
        let cold = between(text);
        let mut backwards: Vec<u64> = (0..2100).rev().map(|t| cold.float_at("speed", t).to_bits()).collect();
        backwards.reverse();
        assert_eq!(first, again);
        assert_eq!(first, backwards);
        // new keys, a constant or a clone never see the values of the keys before
        let mut c = between(text);
        let _ = c.float_at("speed", 500);
        c.parse_cli("speed", "20@0o30/10;700@3.5", base()).unwrap();
        assert_eq!(c.float_at("speed", 500).to_bits(), between("20@0o30/10;700@3.5").float_at("speed", 500).to_bits());
        let copy = c.clone();
        c.parse_cli("speed", "33", base()).unwrap();
        assert_eq!((c.float_at("speed", 500), copy.float_at("speed", 500).to_bits()), (33.0, between("20@0o30/10;700@3.5").float_at("speed", 500).to_bits()));
        // the cap empties the memo and the reads go on unchanged
        let m = Memo::default();
        for t in 0..MEMO_CAP as i64 + 3 {
            m.put((0, t), ParamValue::Float(t as f64));
        }
        assert_eq!((m.get((0, 1)), m.get((0, MEMO_CAP as i64 + 2))), (None, Some(ParamValue::Float((MEMO_CAP + 2) as f64))));
    }

    #[test]
    fn new_keys_empty_the_memo() {
        // the reads at tick 500 go through the memo; without emptying it the old 200 would come back
        let mut c = between("100@0o0/50;300@10i0/50");
        assert_eq!(c.float_at("speed", 500), 200.0);
        c.parse_cli("speed", "100@0o0/50;500@10i0/50", base()).unwrap();
        assert_eq!(c.float_at("speed", 500), 300.0);
        // keys made by hand through set_keys, after a constant too
        let h = Some(Ease { speed: 0.0, influence: 50.0 });
        let key = |time, v: f64| Keyframe { ease_in: h, ease_out: h, ..Keyframe::new(time, ParamValue::Float(v)) };
        c.set_keys("speed", vec![key(0, 100.0), key(1000, 700.0)]).unwrap();
        assert_eq!(c.float_at("speed", 500), 400.0);
        c.set_constant("speed", ParamValue::Float(5.0)).unwrap();
        assert_eq!(c.float_at("speed", 500), 5.0);
        c.set_keys("speed", vec![key(0, 0.0), key(1000, 900.0)]).unwrap();
        assert_eq!(c.float_at("speed", 500), 450.0);
        // a refused curve leaves the keys and their memo as they were
        assert!(c.parse_cli("speed", "100@0o0/50;100@0", base()).is_err());
        assert_eq!(c.float_at("speed", 500), 450.0);
    }

    #[test]
    fn handle_speeds_are_bounded_so_values_stay_finite() {
        // the fastest handles allowed, on a segment of 10^12 ticks, keep every value finite
        let text = "0@0o1000000000/100;100@10000000000i-1000000000/100";
        let mut c = ParamCurves::new(vec![ParamSpec { id: "a", name: "a", kind: ParamKind::Angle { default: 0.0 } }, slider("v", -100.0, 100.0)]);
        c.parse_cli("a", text, base()).unwrap();
        c.parse_cli("v", text, base()).unwrap();
        for t in [1, 2, 1_000, 10_000_000, 500_000_000_000, 999_999_999_999] {
            let (a, v) = (c.float_at("a", t), c.float_at("v", t));
            assert!(a.is_finite() && (-100.0..=100.0).contains(&v), "tick {t}: {a} {v}");
        }
        let fastest = split_key("1@2o-1000000000.0/50").unwrap();
        assert_eq!(fastest.ease_out.unwrap().speed, -1e9);
        // one more is refused when the value is read, so --check sees it
        let nines = "9".repeat(300);
        for (text, speed) in [
            ("1@2o1000000000.5/50", "1000000000.5"),
            ("1@2i-1000000001/50", "-1000000001"),
            (&*format!("100@0o{nines}/100;100@10000000000i{nines}/100"), &*nines),
        ] {
            let e = curves().parse_cli("speed", text, base()).unwrap_err().to_string();
            assert_eq!(e, format!("bad value for `speed`: handle speed {speed} is outside -1000000000..=1000000000"), "{text}");
        }
    }

    #[test]
    fn handles_split_as_written() {
        let k = split_key(" 100@24.5i-2.5/33.3o0/100").unwrap();
        assert_eq!((k.value, k.frame_text, k.frame, k.hold, k.ease), (" 100", "24.5", 24.5, false, false));
        let i = k.ease_in.unwrap();
        assert_eq!((i.speed_text, i.influence_text, i.speed, i.influence), ("-2.5", "33.3", -2.5, 33.3));
        let o = k.ease_out.unwrap();
        assert_eq!((o.speed_text, o.influence_text, o.speed, o.influence), ("0", "100", 0.0, 100.0));
        // Bezier in, Hold out; an outgoing handle alone
        let k = split_key("5@3i7/0.1h").unwrap();
        assert!(k.hold && !k.ease && k.ease_in.is_some() && k.ease_out.is_none());
        let k = split_key("5@ 3o-0/50 ").unwrap();
        assert_eq!((k.frame_text, k.ease_in.is_none(), k.ease_out.unwrap().speed.to_bits()), ("3", true, (-0.0f64).to_bits()));
        // a frame that reads today reads the same: it has no i or o followed by a digit or -
        for (text, frame, hold, ease) in [
            ("1@inf", f64::INFINITY, false, false),
            ("1@-inf", f64::NEG_INFINITY, false, false),
            ("1@infh", f64::INFINITY, true, false),
            ("1@2e1", 20.0, false, false),
            ("1@1e", 1.0, false, true),
            ("1@2e1e", 20.0, false, true),
            ("1@ 7.25h ", 7.25, true, false),
        ] {
            let k = split_key(text).unwrap();
            assert_eq!((k.frame, k.hold, k.ease, k.ease_in, k.ease_out), (frame, hold, ease, None, None), "{text}");
        }
        // handles with no segment to shape are kept and change nothing
        let k = split_key("100@0i50/50").unwrap();
        assert!(k.ease_in.is_some());
        let (with, without) = (between("100@0i50/50;200@10o-70/20"), between("100@0;200@10"));
        for t in -5..1005 {
            assert_eq!(with.float_at("speed", t).to_bits(), without.float_at("speed", t).to_bits(), "tick {t}");
        }
        // a hold key with an incoming handle shapes the segment before it and holds after it
        let c = between("100@0o0/50;200@10i0/50h;300@20");
        assert!(c.float_at("speed", 500) > 140.0 && c.float_at("speed", 500) < 160.0);
        assert_eq!((c.float_at("speed", 1000), c.float_at("speed", 1999), c.float_at("speed", 2000)), (200.0, 200.0, 300.0));
        // on a key the curve is the key's value exactly, not the solve's nearest point
        assert_eq!(between("100@0;0@10o5/50;300@20").float_at("speed", 1000).to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn bad_handles_say_why() {
        let other = |kind| ParamSpec { id: "k", name: "k", kind };
        let err = |spec: ParamSpec, text: &str| ParamCurves::new(vec![spec.clone()]).parse_cli(spec.id, text, base()).unwrap_err().to_string();
        let speed = || float("speed", 100.0);
        let rule = "is not an ease handle (i or o, then SPEED/INFLUENCE, as in i0/33.3)";
        let long = "9".repeat(400);
        for (text, frame, token) in [
            ("100@24i0/", "24i0/", "i0/"),
            ("100@24i0", "24i0", "i0"),
            ("100@24o-", "24o-", "o-"),
            ("100@24i0/33.3x", "24i0/33.3x", "x"),
            ("100@24i0/33.3o", "24i0/33.3o", "o"),
            ("100@24o1/2i3/4", "24o1/2i3/4", "i3/4"),
            ("100@24i1/2i3/4", "24i1/2i3/4", "i3/4"),
            ("100@24i1/2/3", "24i1/2/3", "i1/2/3"),
            ("100@24i1./2", "24i1./2", "i1./2"),
            ("100@24i1/.5", "24i1/.5", "i1/.5"),
            ("100@24i-/2", "24i-/2", "i-/2"),
            ("100@24i1-2/3", "24i1-2/3", "i1-2/3"),
            ("100@24i1e5/50", "24i1e5/50", "i1"),
            ("100@24o1/2hh", "24o1/2hh", "hh"),
            ("100@24o1/2 h", "24o1/2 h", " h"),
        ] {
            assert_eq!(err(speed(), text), format!("bad value for `speed`: `{frame}`: `{token}` {rule}"), "{text}");
        }
        assert_eq!(err(speed(), &format!("1@2i{long}/5")), format!("bad value for `speed`: handle speed {long} is outside -1000000000..=1000000000"));
        for (text, influence) in [("100@24i0/0", "0"), ("100@24i0/0.09", "0.09"), ("100@24o5/100.5", "100.5"), ("100@24i0/-1", "-1"), ("100@1i0/50o2/0.0", "0.0")] {
            assert_eq!(err(speed(), text), format!("bad value for `speed`: influence {influence} is outside 0.1..=100"), "{text}");
        }
        for text in ["100@24i0/50e", "100@24o0/50e", "100@24i1/2o0/50e"] {
            assert_eq!(err(speed(), text), "bad value for `speed`: a key with ease handles cannot also end in e", "{text}");
        }
        for text in ["100@24o0/50h", "100@24i1/50o0/50h"] {
            assert_eq!(err(speed(), text), "bad value for `speed`: a held key has no outgoing handle", "{text}");
        }
        // the kinds that hold between keys refuse handles when the value is read, so --check sees it
        let ends = "holds its value between keys";
        assert_eq!(err(popup("mode"), "1@0i0/50;3@10"), format!("bad value for `mode`: `1@0i0/50` has ease handles, but a popup {ends}"));
        assert_eq!(err(other(ParamKind::Checkbox { default: false }), "0@0; 1@5o0/50"), format!("bad value for `k`: `1@5o0/50` has ease handles, but a checkbox {ends}"));
        assert_eq!(err(other(ParamKind::Int { min: 0, max: 9, default: 0 }), "2@0;5@5i1/50h"), format!("bad value for `k`: `5@5i1/50h` has ease handles, but an int {ends}"));
        assert_eq!(err(other(ParamKind::Color { default: [0, 0, 0] }), "1,2,3@0o5/50;4,5,6@5"), format!("bad value for `k`: `1,2,3@0o5/50` has ease handles, but a color {ends}"));
        // the messages of before are unchanged, also for text that only looks like a handle
        let frame = "is not a frame number (a frame may end in h to hold or e to ease, not both)";
        for (text, shown) in [("100@x", "x"), ("100@24i/5", "24i/5"), ("100@xi0/50", "xi0/50"), ("100@24hi0/50", "24hi0/50"), ("100@24i.5/50", "24i.5/50"), ("100@0he", "0he")] {
            assert_eq!(err(speed(), text), format!("bad value for `speed`: `{shown}` {frame}"), "{text}");
        }
        assert_eq!(err(speed(), "100@0;50"), "bad value for `speed`: `50` has no @frame");
        // set_keys holds a hand made key to the same rules
        let mut c = ParamCurves::new(vec![speed()]);
        let h = Some(Ease { speed: 0.0, influence: 50.0 });
        for (key, reason) in [
            (Keyframe { ease_in: Some(Ease { speed: 0.0, influence: 0.0 }), ..Keyframe::new(0, ParamValue::Float(1.0)) }, "influence 0 is outside 0.1..=100"),
            (Keyframe { ease: true, ease_out: h, ..Keyframe::new(0, ParamValue::Float(1.0)) }, "a key with ease handles cannot also end in e"),
            (Keyframe { hold: true, ease_out: h, ..Keyframe::new(0, ParamValue::Float(1.0)) }, "a held key has no outgoing handle"),
        ] {
            let e = c.set_keys("speed", vec![key, Keyframe::new(10, ParamValue::Float(2.0))]).unwrap_err().to_string();
            assert_eq!(e, format!("bad value for `speed`: {reason}"));
        }
        assert!(c.set_keys("speed", vec![Keyframe { hold: true, ease_in: h, ..Keyframe::new(0, ParamValue::Float(1.0)) }]).is_ok());
    }
}
