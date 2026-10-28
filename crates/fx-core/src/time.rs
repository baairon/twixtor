//! Integer time as it is passed to an effect.
//!
//! Effects work on integer ticks (`time_scale` ticks per second, `time_step` ticks per frame) and
//! truncate in several places, so time is passed as integers instead of seconds as `f64`.

use crate::{Depth, DeviceRequest, FxError, Result};
use std::any::Any;
use std::sync::Arc;

/// An exact frame rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rational {
    pub num: u32,
    pub den: u32,
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

impl Rational {
    /// Reduces `num / den`; both must be non zero.
    pub fn new(num: u32, den: u32) -> Result<Rational> {
        if num == 0 || den == 0 {
            return Err(FxError::Mismatch(format!("frame rate {num}/{den} must be positive")));
        }
        let g = gcd(num as u64, den as u64) as u32;
        Ok(Rational { num: num / g, den: den / g })
    }

    pub fn as_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }
}

impl std::fmt::Display for Rational {
    /// `24` for whole rates, `30000/1001` otherwise: the form `Rational::parse` and `--fps` accept.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

impl Rational {

    /// Parses `24`, `29.97` (read as the exact decimal 2997/100) or `30000/1001`.
    pub fn parse(s: &str) -> Result<Rational> {
        let bad = || FxError::Mismatch(format!("cannot read `{s}` as a frame rate"));
        let s = s.trim();
        if let Some((n, d)) = s.split_once('/') {
            return Rational::new(n.trim().parse().map_err(|_| bad())?, d.trim().parse().map_err(|_| bad())?);
        }
        let (int, frac) = s.split_once('.').unwrap_or((s, ""));
        let dangling_point = s.contains('.') && frac.is_empty();
        if dangling_point || frac.len() > 6 || int.is_empty() || !int.bytes().chain(frac.bytes()).all(|b| b.is_ascii_digit()) {
            return Err(bad());
        }
        let den = 10u64.pow(frac.len() as u32);
        let digits = format!("{int}{frac}");
        let num: u64 = digits.parse().map_err(|_| bad())?;
        if num > u32::MAX as u64 {
            return Err(bad());
        }
        Rational::new(num as u32, den as u32)
    }
}

/// A time base: `time_scale` ticks per second and a frame lasts `time_step` ticks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeBase {
    pub time_scale: u32,
    pub time_step: u32,
}

impl TimeBase {
    /// The coarsest exact base: `time_scale = num`, `time_step = den`.
    pub fn from_fps(fps: Rational) -> TimeBase {
        TimeBase { time_scale: fps.num, time_step: fps.den }
    }

    /// Same rate with every tick `k` times finer.
    pub fn scaled(self, k: u32) -> TimeBase {
        TimeBase { time_scale: self.time_scale * k, time_step: self.time_step * k }
    }

    /// The default time base: integer rates get 100 ticks per frame, NTSC style rates keep
    /// `num/den` (`den` of 1001). This is an assumed value; it only matters for keyframed values
    /// sampled between two frames.
    pub fn default_for_rate(fps: Rational) -> TimeBase {
        let base = TimeBase::from_fps(fps);
        if fps.den == 1 { base.scaled(100) } else { base }
    }

    /// The frame rate in lowest terms.
    pub fn fps(self) -> Rational {
        let g = gcd(self.time_scale as u64, self.time_step as u64).max(1) as u32;
        Rational { num: self.time_scale / g, den: self.time_step / g }
    }

    /// Tick at which frame `n` starts.
    pub fn frame_start(self, n: u64) -> i64 {
        n as i64 * self.time_step as i64
    }

    /// Index of the frame whose span contains `ticks`; negative before time 0.
    pub fn frame_of(self, ticks: i64) -> i64 {
        ticks.div_euclid(self.time_step as i64)
    }

    /// Converts a (possibly fractional) frame number to the nearest tick.
    pub fn ticks_of_frame(self, frame: f64) -> i64 {
        (frame * self.time_step as f64).round() as i64
    }
}

/// Draft or full quality render.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quality {
    Draft,
    High,
}

/// Field order of the footage being rendered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldOrder {
    Progressive,
    UpperFirst,
    LowerFirst,
}

/// Everything a temporal effect needs to know about the frame it renders.
#[derive(Clone, Debug)]
pub struct TemporalContext {
    /// Zero based index of the output frame.
    pub frame_index: u64,
    /// Number of frames the host renders.
    pub frame_count: u64,
    pub width: usize,
    pub height: usize,
    pub depth: Depth,
    pub time: TimeBase,
    /// Time of the frame in ticks since the start of the layer.
    pub current_time: i64,
    /// Ticks per frame of the layer. Equals `time.time_step` in a plain composition.
    pub local_time_step: i64,
    /// Composition (output) frame rate.
    pub fps: Rational,
    pub quality: Quality,
    pub field: FieldOrder,
    /// The render device the host asks for (`vfx apply --device`), [`DeviceRequest::Cpu`] unless
    /// the host sets it. The host has already refused what it cannot serve; an effect passes it on
    /// to its engine, and no pixel depends on it.
    pub device: DeviceRequest,
    /// The host's device executor for the whole run, opaque here (`None` unless the host opened a
    /// device for `device`). An engine downcasts it to the type it knows; fx-core never looks
    /// inside, and no pixel depends on it: a device gives the CPU's bytes or is not used.
    pub executor: Option<Arc<dyn Any + Send + Sync>>,
}

impl TemporalContext {
    /// A progressive, full quality context on the CPU for frame `frame_index` in a plain
    /// composition.
    pub fn new(width: usize, height: usize, depth: Depth, time: TimeBase, frame_index: u64, frame_count: u64) -> TemporalContext {
        TemporalContext {
            frame_index,
            frame_count,
            width,
            height,
            depth,
            time,
            current_time: time.frame_start(frame_index),
            local_time_step: time.time_step as i64,
            fps: time.fps(),
            quality: Quality::High,
            field: FieldOrder::Progressive,
            device: DeviceRequest::Cpu,
            executor: None,
        }
    }

    /// Ticks per frame.
    pub fn time_step(&self) -> i64 {
        self.time.time_step as i64
    }

    /// Ticks per second.
    pub fn time_scale(&self) -> i64 {
        self.time.time_scale as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_frame_rates() {
        assert_eq!(Rational::parse("24").unwrap(), Rational { num: 24, den: 1 });
        assert_eq!(Rational::parse("29.97").unwrap(), Rational { num: 2997, den: 100 });
        assert_eq!(Rational::parse("30000/1001").unwrap(), Rational { num: 30000, den: 1001 });
        assert_eq!(Rational::parse("60000/2002").unwrap(), Rational { num: 30000, den: 1001 });
        assert_eq!(Rational::parse(" 12.5 ").unwrap(), Rational { num: 25, den: 2 });
        for bad in ["", "0", "24/0", "-24", "1e3", "24.", ".5", "1.2345678", "a/b", "5000000000"] {
            assert!(Rational::parse(bad).is_err(), "`{bad}` should not parse");
        }
    }

    #[test]
    fn display_is_the_form_parse_accepts() {
        for text in ["24", "30000/1001", "25", "12500/3"] {
            let r = Rational::parse(text).unwrap();
            assert_eq!(r.to_string(), text);
            assert_eq!(Rational::parse(&r.to_string()).unwrap(), r);
        }
    }

    #[test]
    fn base_round_trips_frames() {
        let ntsc = TimeBase::from_fps(Rational::new(24000, 1001).unwrap());
        assert_eq!((ntsc.time_scale, ntsc.time_step), (24000, 1001));
        assert_eq!(ntsc.frame_start(3), 3003);
        assert_eq!(ntsc.frame_of(3003), 3);
        assert_eq!(ntsc.frame_of(3002), 2);
        assert_eq!(ntsc.frame_of(-1), -1);
        assert_eq!(ntsc.ticks_of_frame(2.5), 2503);
        assert_eq!(ntsc.fps(), Rational { num: 24000, den: 1001 });
    }

    #[test]
    fn default_base_refines_integer_rates_only() {
        let int = TimeBase::default_for_rate(Rational::new(24, 1).unwrap());
        assert_eq!((int.time_scale, int.time_step), (2400, 100));
        let ntsc = TimeBase::default_for_rate(Rational::new(30000, 1001).unwrap());
        assert_eq!((ntsc.time_scale, ntsc.time_step), (30000, 1001));
    }

    #[test]
    fn context_starts_at_the_frame() {
        let base = TimeBase::default_for_rate(Rational::new(25, 1).unwrap());
        let cx = TemporalContext::new(64, 48, Depth::U8, base, 7, 100);
        assert_eq!(cx.current_time, 700);
        assert_eq!(cx.local_time_step, 100);
        assert_eq!((cx.time_step(), cx.time_scale()), (100, 2500));
        assert_eq!(cx.fps, Rational { num: 25, den: 1 });
    }

    #[test]
    fn a_new_context_asks_for_the_cpu() {
        let base = TimeBase::default_for_rate(Rational::new(24, 1).unwrap());
        let cx = TemporalContext::new(4, 4, Depth::F32, base, 0, 1);
        assert_eq!(cx.device, DeviceRequest::Cpu);
        assert!(cx.executor.is_none());
        assert_eq!((cx.quality, cx.field), (Quality::High, FieldOrder::Progressive));
    }
}
