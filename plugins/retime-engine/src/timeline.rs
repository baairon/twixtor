//! Time model of the retiming engine: the rate model, the frame positions, the time map `src_pos`
//! with its frame-to-time companions, and the frame selection of the core render.
//!
//! Precision is part of the contract: where a comment says `f32` the operation is a single
//! precision operation, where it says `f64` the operands are widened first. Time is an `i32` tick
//! count, so the public interface uses `i32` ticks and `i32` frame numbers.

use fx_core::{FieldOrder, FxError, Result};
use std::cell::{Cell, RefCell};

/// Bias added before a sampling time is truncated to a whole tick, as in `trunc(x + K)`: the
/// sampling time of every parameter read in `src_pos` (Frame Number, the offset read, the half
/// frame steps and the partial step of Speed). MEASURED in the original's code, not by probing:
/// the time map functions of Twixtor_64, TwixtorPro_64 and TwixtorProVectorsIn_64 all add the double
/// 0.05. It is not observable on whole output frames
/// (`lts * t` is a whole number of ticks there), which is why no comparison measured it; a
/// fraction of a tick between 0.95 and 1.0 samples the next tick.
pub const K: f64 = 0.05;

/// The bias of `frame_time` for unequal rates (`trunc(d * dur + dur * FRAME_TIME_K)`) and the
/// tolerance of its frame rate equality test (`|in - out| < FRAME_TIME_K`). MEASURED from the
/// original's checkout times under the emulator:
/// the bias at durations 96, 1200 and 4800 ticks lies in [0.05, 0.0502), and rates that differ by
/// 0.0499992 are equal while rates that differ by 0.06 and 0.09 are not.
pub const FRAME_TIME_K: f64 = 0.05;

/// Snapping tolerance for typed frame rates, compared against the absolute value of the f32
/// narrowed difference.
pub const SNAP_TOLERANCE: f64 = 2e-5;
/// The three snapped rates, in the order they are tested.
pub const SNAP_RATES: [f64; 3] = [29.97, 59.94, 23.976];
/// Smallest usable frame rate.
pub const MIN_RATE: f64 = 0.01;
/// Exactness tolerance on the fraction of a source position.
pub const EXACT_TOLERANCE: f64 = 0.005;
/// Two source positions closer than this count as equal.
pub const SAME_POSITION_EPS: f64 = 1e-6;
/// Smallest remainder of the Speed integral worth a partial step.
pub const SPEED_REMAINDER_EPS: f64 = 0.0001;

/// Truncation toward zero as the x86 conversion does: NaN and values outside the 32 bit range
/// give the "integer indefinite" `i32::MIN` (Rust's `as` would saturate).
pub fn cvtt_i32(x: f64) -> i32 {
    if x > -2147483649.0 && x < 2147483648.0 {
        x as i32
    } else {
        i32::MIN
    }
}

/// The same for an f32 input, see [`cvtt_i32`].
pub fn cvtt_i32_f32(x: f32) -> i32 {
    if x > -2147483904.0 && x < 2147483648.0 {
        x as i32
    } else {
        i32::MIN
    }
}

fn bad_rate(in_fps: f64, out_fps: f64) -> FxError {
    FxError::BadParam {
        id: "frame_rate".into(),
        reason: format!(
            "error 5 (bad parameter): input rate {in_fps} and output rate {out_fps} must be above {MIN_RATE}"
        ),
    }
}

fn snap(mut rate: f64) -> f64 {
    for target in SNAP_RATES {
        if (((rate - target) as f32).abs() as f64) < SNAP_TOLERANCE {
            rate = target;
        }
    }
    rate
}

/// The two frame rates and the duration of one source frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rates {
    /// Source frame rate after snapping and field doubling.
    pub in_fps: f64,
    /// Composition frame rate after snapping.
    pub out_fps: f64,
    /// Duration of one source frame in time units, `((lts as f64 * out_fps) / in_fps) as f32`.
    pub dur: f32,
    /// Ticks per layer frame (`lts`).
    pub local_time_step: i32,
}

impl Rates {
    /// The rate model. Order of operations:
    /// 1. both rates start at 15.0; the source rate becomes the slider value narrowed through
    ///    f32; the output rate becomes the composition rate (`None` keeps 15.0);
    /// 2. `In FPS is Out FPS` makes both 30.0;
    /// 3. each rate (source first) snaps to 29.97, 59.94 and 23.976 when
    ///    `|(rate - c) as f32| < 2e-5` (the comparison is a double compare, so the f32 difference
    ///    is widened first); the true NTSC rate 30000/1001 is 3e-5 away from 29.97 and is not
    ///    snapped;
    /// 4. any input field order doubles the source rate;
    /// 5. `out < 0.01 || in <= 0.01` is error 5 (note the asymmetry, and that NaN passes). The
    ///    planner and the core repeat the test on the f32 narrowed rates, which differs only for
    ///    rates within 1e-9 of 0.01 and is not modelled.
    ///
    /// `input_rate` is the raw `Input: Frame Rate` slider value (f64), `comp_rate` the
    /// composition rate as a double (a plain `f64` or an `Option<f64>`), `local_time_step` the
    /// integer ticks per layer frame.
    pub fn new(
        input_rate: f64,
        comp_rate: impl Into<Option<f64>>,
        in_fps_is_out_fps: bool,
        input_fields: FieldOrder,
        local_time_step: i32,
    ) -> Result<Rates> {
        let mut in_fps = (input_rate as f32 * 1.0f32) as f64;
        let mut out_fps = comp_rate.into().unwrap_or(15.0);
        if in_fps_is_out_fps {
            in_fps = 30.0;
            out_fps = 30.0;
        }
        in_fps = snap(in_fps);
        out_fps = snap(out_fps);
        if input_fields != FieldOrder::Progressive {
            in_fps *= 2.0;
        }
        if MIN_RATE > out_fps || MIN_RATE >= in_fps {
            return Err(bad_rate(in_fps, out_fps));
        }
        let dur = ((local_time_step as f64 * out_fps) / in_fps) as f32;
        Ok(Rates { in_fps, out_fps, dur, local_time_step })
    }
}

/// Positions of the current and the next output frame in output frames since layer time 0:
/// `t0 = cur / lts` and `t1 = (time_step + cur) / lts`, all f32. A zero `lts` divides to infinity
/// or NaN.
pub fn frame_positions(current_time: i32, time_step: i32, local_time_step: i32) -> (f32, f32) {
    let cur = current_time as f32;
    let lts = local_time_step as f32;
    let t0 = cur / lts;
    let t1 = (time_step as f32 + cur) / lts;
    (t0, t1)
}

/// The time of the source frame the `Display == Source` branch copies:
/// `N = trunc((offset_fixed as f32) * (-1/65536))`, time `cur - N * time_step`, i.e.
/// `cur + trunc(offset) * time_step` with 32 bit wrapping multiply and subtract.
pub fn source_display_time(current_time: i32, time_step: i32, src_time_offset_fixed: i32) -> i32 {
    let n = cvtt_i32_f32(src_time_offset_fixed as f32 * -1.52587890625e-05f32);
    current_time.wrapping_sub(n.wrapping_mul(time_step))
}

/// What the time map reads from the parameters. Every method samples the parameter at a layer
/// time in ticks, so a keyframed parameter is evaluated at that tick. Values are raw: popups as
/// integers, 16.16 sliders as fixed point integers.
pub trait TimeParams {
    /// `Time Remap Mode` popup: 1 = Speed, 2 = Frame Number. The time map reads it
    /// at `current_time` only.
    fn remap_mode(&self, time: i32) -> i32;
    /// `Speed %` as the 16.16 fixed point percent (`65536 * 100` is 100 percent).
    fn speed_fixed(&self, time: i32) -> i32;
    /// `Frame Num` float slider as a double (narrowed to f32 by the reader).
    fn frame_num(&self, time: i32) -> f64;
    /// `Src Time Offset` as the 16.16 fixed point frame count.
    fn src_time_offset_fixed(&self, time: i32) -> i32;
    /// `Src: Mark Segs` popup value 1..=5 (the time map maps it to 0..=4).
    fn mark_segs(&self, time: i32) -> i32;
}

impl<T: TimeParams + ?Sized> TimeParams for &T {
    fn remap_mode(&self, time: i32) -> i32 {
        (**self).remap_mode(time)
    }
    fn speed_fixed(&self, time: i32) -> i32 {
        (**self).speed_fixed(time)
    }
    fn frame_num(&self, time: i32) -> f64 {
        (**self).frame_num(time)
    }
    fn src_time_offset_fixed(&self, time: i32) -> i32 {
        (**self).src_time_offset_fixed(time)
    }
    fn mark_segs(&self, time: i32) -> i32 {
        (**self).mark_segs(time)
    }
}

/// The plugin specific part of the engine that turns output frame positions into source positions
/// and source frame numbers into layer times.
///
/// Call order matters for implementations with state: the planner and the core call
/// `src_pos(t0)`, then `src_pos(t1)`, and only then `marker` and `frame_time`, so those see the
/// state left by the call for `t1`.
pub trait TimeMap {
    /// Fractional source frame number for the output position `t`.
    fn src_pos(&self, t: f32) -> Result<f32>;
    /// Layer time in ticks of the source frame `f` (an integer value as float).
    fn frame_time(&self, f: f32) -> i32;
    /// Segment marker of the frame `f`, 0..=4 (Cut A, Cut B, Cut C, No MV Use Nearest, No MV
    /// Frame Blend).
    fn marker(&self, f: f32) -> Result<i32>;
}

/// The Speed / Frame Number time map of plugin 01, with its `base` side effect modelled by a
/// [`Cell`].
///
/// Plugin 03 has the same shape for `src_pos`; the single difference is that it has no
/// `Src Time Offset`: Speed mode neither reads that parameter nor writes `base`.
/// [`Self::without_src_time_offset`] selects that variant for `src_pos` only. Plugin 03's
/// `frame_time` and marker are different functions (a plain `trunc(dur * n + 0.5)` without `base`
/// and without the K bias branch, and no K use at all) and are not provided here.
pub struct RemapTimeMap<P: TimeParams> {
    params: P,
    rates: Rates,
    current_time: i32,
    base: Cell<i32>,
    reads_offset: bool,
    /// Opt-in prefix sums of the Speed integral, `speed_cache[k]` = the value of `out` after `k` half steps.
    speed_cache: Option<RefCell<Vec<f32>>>,
}

impl<P: TimeParams> RemapTimeMap<P> {
    /// `current_time` is the time at which the mode popup is read. `base` starts at 0.
    pub fn new(params: P, rates: Rates, current_time: i32) -> Self {
        RemapTimeMap { params, rates, current_time, base: Cell::new(0), reads_offset: true, speed_cache: None }
    }

    /// Remembers the running sums of the Speed integral so that a later call does not sample the
    /// curve from time 0 again (without this the cost grows with the position: a clip of n frames
    /// costs O(n^2) parameter reads). The sum after `k` half steps depends only on the samples at
    /// `trunc(lts * k / 2 + K)`, never on `t`, and the recurrence is evaluated in the same order,
    /// so the results are bit identical. Only valid while `speed_fixed` is a pure function of the
    /// time for the life of the map (true for a frame render and for a whole run over immutable
    /// curves).
    pub fn with_speed_cache(mut self) -> Self {
        self.speed_cache = Some(RefCell::new(vec![0.0]));
        self
    }

    /// Plugin 03's variant of `src_pos`: no `Src Time Offset` read, `base` stays 0.
    pub fn without_src_time_offset(mut self) -> Self {
        self.reads_offset = false;
        self
    }

    pub fn rates(&self) -> &Rates {
        &self.rates
    }

    /// The frame offset the last `src_pos` call left behind.
    pub fn base(&self) -> i32 {
        self.base.get()
    }

    /// `trunc((lts_f * x) + K)` with the f32 product widened before the add.
    fn sample_time(lts_f: f32, x: f32) -> i32 {
        cvtt_i32((lts_f * x) as f64 + K)
    }
}

impl<P: TimeParams> TimeMap for RemapTimeMap<P> {
    /// Frame Number: `base = 0`, parameter time `trunc((lts_f * t) + K)`, value
    /// `(frame_num as f32)`, result `((in_fps / out_fps) * value as f64) as f32`. Speed:
    /// `base = -trunc((offset as f32) * 2^-16)` (offset read at the same time as the first Speed
    /// sample), then the integral of the speed curve over half frame steps: `out = sp / 2 + out`
    /// for `k = 1 ..= (t * 2) as i32`, with the sampling time `trunc(((lts_f * k) / 2) + K)` and
    /// `sp = (speed_fixed as f64 / 6553600) as f32`, then a partial step `((rem * sp) / 2) + out`
    /// when `rem = t * 2 - (k - 1) > 0.0001`, finally times `in_fps / out_fps`. Negative `t`
    /// gives 0. Another mode value is an error.
    fn src_pos(&self, t: f32) -> Result<f32> {
        let lts_f = self.rates.local_time_step as f32;
        let mode = self.params.remap_mode(self.current_time);
        match mode {
            2 => {
                self.base.set(0);
                let time = Self::sample_time(lts_f, t);
                let n = self.params.frame_num(time) as f32;
                Ok(((self.rates.in_fps / self.rates.out_fps) * n as f64) as f32)
            }
            1 => {
                let mut out = 0.0f32;
                if self.reads_offset {
                    let time0 = Self::sample_time(lts_f, t);
                    let offset = self.params.src_time_offset_fixed(time0);
                    let whole = cvtt_i32_f32(offset as f32 * 1.52587890625e-05f32);
                    self.base.set(whole.wrapping_neg());
                }
                let n2 = cvtt_i32_f32(t * 2.0);
                if n2 >= 1 << 24 {
                    // `k += 1.0` stops counting at 2^24, so the loop never ends for such
                    // positions.
                    return Err(FxError::Unsupported(
                        "output position beyond 2^23 frames is not supported".into(),
                    ));
                }
                let mut k = 1.0f32;
                if let Some(cache) = &self.speed_cache {
                    let mut sums = cache.borrow_mut();
                    while (sums.len() as i64) <= n2 as i64 {
                        let step = sums.len() as f32;
                        let time = cvtt_i32(((lts_f * step) / 2.0) as f64 + K);
                        let sp = ((self.params.speed_fixed(time) as f64) / 6553600.0) as f32;
                        let prev = sums[sums.len() - 1];
                        sums.push((sp / 2.0) + prev);
                    }
                    if n2 >= 1 {
                        out = sums[n2 as usize];
                        k = n2 as f32 + 1.0;
                    }
                } else {
                    while (n2 as f32) >= k {
                        let time = cvtt_i32(((lts_f * k) / 2.0) as f64 + K);
                        let sp = ((self.params.speed_fixed(time) as f64) / 6553600.0) as f32;
                        out = (sp / 2.0) + out;
                        k += 1.0;
                    }
                }
                let tt = t * 2.0;
                let rem: f32 = tt - (k + -1.0);
                if (rem as f64) > SPEED_REMAINDER_EPS {
                    let time = cvtt_i32((((tt * lts_f) / 2.0) as f64) + K);
                    let sp = ((self.params.speed_fixed(time) as f64) / 6553600.0) as f32;
                    out = ((rem * sp) / 2.0) + out;
                }
                Ok(((self.rates.in_fps / self.rates.out_fps) * out as f64) as f32)
            }
            other => Err(FxError::BadParam {
                id: "time_remap_mode".into(),
                reason: format!("mode {other} is neither 1 (Speed) nor 2 (Frame Number)"),
            }),
        }
    }

    /// Time of a source frame: `d = f - base` (f32); when `|in_fps - out_fps| >= FRAME_TIME_K`
    /// (NaN included) the time is `trunc((d * dur) as f64 + dur as f64 * FRAME_TIME_K)` with an
    /// f32 product, otherwise `trunc(d as f64 * dur as f64 + 0.5)` with an f64 product.
    fn frame_time(&self, f: f32) -> i32 {
        let d: f32 = f - self.base.get() as f32;
        let dur = self.rates.dur;
        if (self.rates.in_fps - self.rates.out_fps).abs() < FRAME_TIME_K {
            cvtt_i32((d as f64) * (dur as f64) + 0.5)
        } else {
            cvtt_i32((d * dur) as f64 + (dur as f64) * FRAME_TIME_K)
        }
    }

    /// `Src: Mark Segs` sampled at `trunc(((f - base) * dur) as f64 + 0.5)`, popup value 1..=5
    /// mapped to 0..=4, anything else 0. The sampler cannot fail.
    fn marker(&self, f: f32) -> Result<i32> {
        let d: f32 = f - self.base.get() as f32;
        let time = cvtt_i32(((d * self.rates.dur) as f64) + 0.5);
        let v = self.params.mark_segs(time);
        Ok(if (1..=5).contains(&v) { v - 1 } else { 0 })
    }
}

/// The result of the frame selection of the core render.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Selection {
    /// Source position of the output frame, `src_pos(t0)`.
    pub s0: f32,
    /// Source position of the next output frame, `src_pos(t1)` (only used by the exactness test).
    pub s1: f32,
    /// First source frame, `trunc(s0)`.
    pub a: i32,
    /// Second source frame, `a + 1`.
    pub b: i32,
    /// Weight of frame `a`: `1.0 - (s0 - a as f32)` in f32. Above 1 for negative positions.
    pub wa: f32,
    /// The position lies on a frame (within 0.005) or blur compensation is off:
    /// the vector path does not run, the fast path does.
    pub exact: bool,
    /// Vectors quality (0..=4) after the segment override.
    pub vectors_quality: i32,
    /// Frame interpolation (0 Nearest, 1 Blend, 2 Weighted Blend) after the segment override.
    pub interp: i32,
}

impl Selection {
    /// The weight the fast path passes to `blend(A, B, w)` for Frame Interp = Blend:
    /// `(1.0f64 - wa as f64) as f32`.
    pub fn blend_weight(&self) -> f32 {
        (1.0f64 - self.wa as f64) as f32
    }

    /// Nearest in the fast path: frame `a` when `wa >= 0.5`, else frame `b`.
    pub fn nearest_is_a(&self) -> bool {
        self.wa >= 0.5
    }
}

/// The exactness test (the planner calls its result `simple`): true unless the fraction
/// `s0 - floorf(s0)` is more than 0.005 away from both 0 and 1 (`|f| > 0.005` and
/// `|(f as f64 - 1.0) as f32| > 0.005`, absolute values taken in f32 and compared as doubles), or
/// the two consecutive positions differ by more than 1e-6 while `blur` (Blur Compensation) is
/// above 0.005. The planner and the core use this same expression.
pub fn exact_position(s0: f32, s1: f32, blur: f32) -> bool {
    let mut exact = true;
    let frac: f32 = s0 - s0.floor();
    if (frac.abs() as f64) > EXACT_TOLERANCE && ((((frac as f64) - 1.0) as f32).abs() as f64) > EXACT_TOLERANCE {
        exact = false;
    }
    if ((s0 - s1).abs() as f64) > SAME_POSITION_EPS && (blur as f64) > EXACT_TOLERANCE {
        exact = false;
    }
    exact
}

/// The segment override applied before planning and before rendering. `ma = marker(a)`,
/// `mb = marker(g)` with `a = trunc(s0)` and `g = trunc(s0 + 1.0)`: `ma != mb` gives (vq 0,
/// Nearest), then `ma == 3` gives (0, Nearest) and `ma == 4` gives (0, Blend) ("No MV, Use
/// Nearest" and "No MV, Frame Blend"); later tests win.
///
/// The planner replaces `g` by `a + 1` when `g < 1`; the core does not. The two differ only for
/// source positions below 0, where the marker of frame 0 is compared with the marker of frame 1
/// in the planner and with itself in the core. `planner` selects that variant. The `fields` flag
/// (always 0 in plugin 01) would add 0.5 instead of 1.0 to `s0` and is not modelled.
pub fn marker_override(map: &dyn TimeMap, s0: f32, vectors_quality: i32, interp: i32, planner: bool) -> Result<(i32, i32)> {
    let a = cvtt_i32_f32(s0);
    let mut g = cvtt_i32(s0 as f64 + 1.0);
    if planner && g < 1 {
        g = a.wrapping_add(1);
    }
    let ma = map.marker(a as f32)?;
    let mb = map.marker(g as f32)?;
    let (mut vq, mut ip) = (vectors_quality, interp);
    if ma != mb {
        vq = 0;
        ip = 0;
    }
    if ma == 3 {
        vq = 0;
        ip = 0;
    }
    if ma == 4 {
        vq = 0;
        ip = 1;
    }
    Ok((vq, ip))
}

/// Frame selection of the core render: the source positions of the two output frame positions (in
/// this order, so the time map state is the one left by `t1`), the segment override with the
/// core's form of `g`, the frame pair `a = trunc(s0)`, `b = a + 1`, the weight of `a` and the
/// exactness flag. `vectors_quality` is the raw popup value minus one (0..=4), `interp` likewise
/// (0..=2), `blur` the Blur Compensation value.
pub fn select(map: &dyn TimeMap, t0: f32, t1: f32, vectors_quality: i32, interp: i32, blur: f32) -> Result<Selection> {
    let s0 = map.src_pos(t0)?;
    let s1 = map.src_pos(t1)?;
    let (vq, ip) = marker_override(map, s0, vectors_quality, interp, false)?;
    let a = cvtt_i32_f32(s0);
    let frac: f32 = s0 - a as f32;
    let wa: f32 = 1.0f32 - frac;
    Ok(Selection {
        s0,
        s1,
        a,
        b: a.wrapping_add(1),
        wa,
        exact: exact_position(s0, s1, blur),
        vectors_quality: vq,
        interp: ip,
    })
}

/// Test doubles shared with the planner tests.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use std::cell::RefCell;

    pub type F<T> = Box<dyn Fn(i32) -> T>;

    /// A `TimeParams` whose parameters are closures of the sampling time; every read is logged.
    pub struct Mock {
        pub mode: Cell<i32>,
        pub speed: F<i32>,
        pub frame: F<f64>,
        pub offset: F<i32>,
        pub marks: F<i32>,
        pub log: RefCell<Vec<(&'static str, i32)>>,
    }

    /// 16.16 fixed point from a plain number, as fx-core provides it (`float_to_fixed`).
    pub fn fixed(v: f64) -> i32 {
        fx_core::float_to_fixed(v)
    }

    impl Mock {
        /// Speed mode with a constant speed in percent, no offset, no markers.
        pub fn speed(percent: f64) -> Mock {
            Mock {
                mode: Cell::new(1),
                speed: Box::new(move |_| fixed(percent)),
                frame: Box::new(|_| 0.0),
                offset: Box::new(|_| 0),
                marks: Box::new(|_| 1),
                log: RefCell::new(Vec::new()),
            }
        }

        /// Frame Number mode with a constant frame number.
        pub fn frame_number(value: f64) -> Mock {
            Mock { mode: Cell::new(2), frame: Box::new(move |_| value), ..Mock::speed(100.0) }
        }

        pub fn with_offset(mut self, f: impl Fn(i32) -> i32 + 'static) -> Mock {
            self.offset = Box::new(f);
            self
        }

        pub fn with_marks(mut self, f: impl Fn(i32) -> i32 + 'static) -> Mock {
            self.marks = Box::new(f);
            self
        }

        pub fn reads(&self) -> Vec<(&'static str, i32)> {
            self.log.borrow().clone()
        }
    }

    impl TimeParams for Mock {
        fn remap_mode(&self, time: i32) -> i32 {
            self.log.borrow_mut().push(("mode", time));
            self.mode.get()
        }
        fn speed_fixed(&self, time: i32) -> i32 {
            self.log.borrow_mut().push(("speed", time));
            (self.speed)(time)
        }
        fn frame_num(&self, time: i32) -> f64 {
            self.log.borrow_mut().push(("frame", time));
            (self.frame)(time)
        }
        fn src_time_offset_fixed(&self, time: i32) -> i32 {
            self.log.borrow_mut().push(("offset", time));
            (self.offset)(time)
        }
        fn mark_segs(&self, time: i32) -> i32 {
            self.log.borrow_mut().push(("marks", time));
            (self.marks)(time)
        }
    }

    pub fn rates(in_fps: f64, out_fps: f64, lts: i32) -> Rates {
        Rates::new(in_fps, out_fps, false, FieldOrder::Progressive, lts).unwrap()
    }

    /// 30 fps both ways, 100 ticks per frame.
    pub fn plain_rates() -> Rates {
        rates(30.0, 30.0, 100)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn k_is_the_documented_decision() {
        assert_eq!(K, 0.05);
        assert_eq!(FRAME_TIME_K, 0.05);
        assert_eq!(K.to_bits(), 0x3fa9_9999_9999_999a);
        assert_eq!(SNAP_RATES[0].to_bits(), 0x403d_f851_eb85_1eb8);
        assert_eq!(SNAP_RATES[1].to_bits(), 0x404d_f851_eb85_1eb8);
        assert_eq!(SNAP_RATES[2].to_bits(), 0x4037_f9db_22d0_e560);
        assert_eq!(SNAP_TOLERANCE.to_bits(), 0x3ef4_f8b5_88e3_68f1);
        assert_eq!(MIN_RATE.to_bits(), 0x3f84_7ae1_47ae_147b);
        assert_eq!(SPEED_REMAINDER_EPS.to_bits(), 0x3f1a_36e2_eb1c_432d);
    }

    #[test]
    fn truncating_conversions_follow_x86() {
        assert_eq!(cvtt_i32(2.9), 2);
        assert_eq!(cvtt_i32(-2.9), -2);
        assert_eq!(cvtt_i32(-0.5), 0);
        assert_eq!(cvtt_i32(2147483647.9), 2147483647);
        assert_eq!(cvtt_i32(2147483648.0), i32::MIN, "out of range is the integer indefinite");
        assert_eq!(cvtt_i32(-2147483648.0), i32::MIN);
        assert_eq!(cvtt_i32(f64::NAN), i32::MIN);
        assert_eq!(cvtt_i32(f64::INFINITY), i32::MIN);
        assert_eq!(cvtt_i32_f32(-1.5), -1);
        assert_eq!(cvtt_i32_f32(16777216.0), 16777216);
        assert_eq!(cvtt_i32_f32(2147483520.0), 2147483520);
        assert_eq!(cvtt_i32_f32(2147483648.0), i32::MIN);
        assert_eq!(cvtt_i32_f32(f32::NAN), i32::MIN);
        assert_eq!(cvtt_i32_f32(-2147483648.0), i32::MIN);
    }

    // ---- rate model ----

    fn new_rates(input: f64, comp: f64, same: bool, fields: FieldOrder, lts: i32) -> Result<Rates> {
        Rates::new(input, comp, same, fields, lts)
    }

    #[test]
    fn default_parameters_give_30_and_30() {
        // Input: Frame Rate defaults to the f32 29.97 and In FPS is Out FPS is on: both rates become 30.0.
        let r = new_rates(29.97f32 as f64, 30000.0 / 1001.0, true, FieldOrder::Progressive, 100).unwrap();
        assert_eq!((r.in_fps, r.out_fps, r.dur, r.local_time_step), (30.0, 30.0, 100.0, 100));
    }

    #[test]
    fn a_missing_composition_rate_keeps_the_default_15() {
        // Without a composition rate (`None`) the output rate stays at the default 15.0.
        let r = Rates::new(30.0, None, false, FieldOrder::Progressive, 100).unwrap();
        assert_eq!((r.in_fps, r.out_fps, r.dur), (30.0, 15.0, 50.0));
        let same = Rates::new(30.0, Some(24.0), false, FieldOrder::Progressive, 100).unwrap();
        assert_eq!(same, Rates::new(30.0, 24.0, false, FieldOrder::Progressive, 100).unwrap());
        // In FPS is Out FPS overrides it as well.
        let r = Rates::new(24.0, None, true, FieldOrder::Progressive, 100).unwrap();
        assert_eq!((r.in_fps, r.out_fps), (30.0, 30.0));
    }

    #[test]
    fn typed_ntsc_rates_snap_but_the_true_rates_do_not() {
        let snapped = |v: f64| new_rates(v, 30.0, false, FieldOrder::Progressive, 1).unwrap().in_fps;
        // The slider stores f32, so 29.97 arrives as 29.969999313354492 and snaps to the double 29.97.
        assert_eq!(snapped(29.97f32 as f64).to_bits(), 29.97f64.to_bits());
        assert_eq!(snapped(59.94f32 as f64).to_bits(), 59.94f64.to_bits());
        assert_eq!(snapped(23.976f32 as f64).to_bits(), 23.976f64.to_bits());
        // 30000/1001 is 2.997e-5 above 29.97, 60000/1001 is 5.99e-5 above 59.94, 24000/1001 is 2.4e-5 above 23.976.
        let ntsc = |num: f64| new_rates(30.0, num / 1001.0, false, FieldOrder::Progressive, 1).unwrap().out_fps;
        assert_eq!(ntsc(30000.0), 30000.0 / 1001.0);
        assert_eq!(ntsc(60000.0), 60000.0 / 1001.0);
        assert_eq!(ntsc(24000.0), 24000.0 / 1001.0);
        // Integer rates are left alone; the source rate goes through f32, the output rate does not.
        assert_eq!(snapped(24.0), 24.0);
        assert_eq!(snapped(25.0), 25.0);
        assert_eq!(snapped(0.1), 0.1f32 as f64);
        let out = |v: f64| new_rates(30.0, v, false, FieldOrder::Progressive, 1).unwrap().out_fps;
        assert_eq!(out(0.1), 0.1);
        // The comp rate snaps the same way (typed 29.97 as a double).
        assert_eq!(out(29.97 + 1e-9), 29.97);
        assert_eq!(out(59.94 - 1e-9), 59.94);
        assert_eq!(out(23.976 + 1e-9), 23.976);
    }

    #[test]
    fn snapping_tolerance_is_strictly_below_2e_minus_5() {
        let out = |v: f64| new_rates(30.0, v, false, FieldOrder::Progressive, 1).unwrap().out_fps;
        assert_eq!(out(29.97 + 1.9e-5), 29.97);
        assert_eq!(out(29.97 - 1.9e-5), 29.97);
        assert_eq!(out(29.97 + 2.1e-5), 29.97 + 2.1e-5);
        assert_eq!(out(59.94 + 1.9e-5), 59.94);
        assert_eq!(out(59.94 - 2.1e-5), 59.94 - 2.1e-5);
        assert_eq!(out(23.976 - 1.9e-5), 23.976);
        assert_eq!(out(23.976 + 2.1e-5), 23.976 + 2.1e-5);
        // The difference is narrowed to f32 before the comparison, and that f32 is widened and compared with
        // the DOUBLE 2e-5: the f32 nearest to 2e-5 (1.99999995e-5) is below it and still snaps.
        let diff = 1.9999999494757503e-5f64;
        assert!((diff as f32) as f64 <= 2e-5);
        assert_eq!(out(29.97 + diff), 29.97);
    }

    #[test]
    fn input_fields_double_the_source_rate() {
        for fields in [FieldOrder::UpperFirst, FieldOrder::LowerFirst] {
            let r = new_rates(25.0, 50.0, false, fields, 100).unwrap();
            assert_eq!((r.in_fps, r.out_fps), (50.0, 50.0));
            assert_eq!(r.dur, 100.0);
        }
        // Doubling happens after snapping: a typed 29.97 becomes 2 * 29.97 as a double.
        let r = new_rates(29.97f32 as f64, 30.0, false, FieldOrder::UpperFirst, 1).unwrap();
        assert_eq!(r.in_fps, 29.97 * 2.0);
    }

    #[test]
    fn duration_of_a_source_frame_is_lts_times_the_rate_ratio() {
        // 24 fps source in a 30 fps comp: a source frame lasts 100 * 30 / 24 = 125 ticks.
        assert_eq!(rates(24.0, 30.0, 100).dur, 125.0);
        // 1001 ticks per frame at 30000/1001, source typed 29.97 (snapped): (1001 * 29.97002997...) / 29.97.
        let r = new_rates(29.97f32 as f64, 30000.0 / 1001.0, false, FieldOrder::Progressive, 1001).unwrap();
        assert_eq!(r.dur, ((1001.0f64 * (30000.0 / 1001.0)) / 29.97) as f32);
        assert!((r.dur - 1001.001).abs() < 1e-3, "{}", r.dur);
        assert_eq!(rates(60.0, 59.94, 100).dur, 99.9f64 as f32);
    }

    #[test]
    fn bad_rates_are_error_5() {
        for (input, comp) in [(0.0, 30.0), (30.0, 0.0), (-1.0, 30.0), (30.0, -5.0), (0.01f32 as f64, 30.0)] {
            let e = new_rates(input, comp, false, FieldOrder::Progressive, 100).unwrap_err();
            assert!(e.to_string().contains("error 5"), "{e}");
            assert!(matches!(e, FxError::BadParam { .. }));
        }
        // In FPS is Out FPS rescues bad typed values.
        assert!(new_rates(0.0, 0.0, true, FieldOrder::Progressive, 100).is_ok());
        // The test is asymmetric: an output rate of exactly 0.01 passes (0.01 > 0.01 is false),
        // a source rate of exactly 0.01 does not (0.01 >= 0.01). The slider value is an f32, so use
        // the first f32 above 0.01 for the source.
        assert!(new_rates(30.0, 0.01, false, FieldOrder::Progressive, 100).is_ok());
        let above = f32::from_bits(0.01f32.to_bits() + 1);
        assert!(above as f64 > 0.01);
        assert!(new_rates(above as f64, 30.0, false, FieldOrder::Progressive, 100).is_ok());
        // NaN fails neither comparison (unordered), so it slips through.
        assert!(new_rates(30.0, f64::NAN, false, FieldOrder::Progressive, 100).is_ok());
    }

    // ---- frame positions (section 4) ----

    #[test]
    fn frame_positions_are_f32_quotients() {
        assert_eq!(frame_positions(0, 100, 100), (0.0, 1.0));
        assert_eq!(frame_positions(700, 100, 100), (7.0, 8.0));
        assert_eq!(frame_positions(250, 100, 100), (2.5, 3.5));
        assert_eq!(frame_positions(3003, 1001, 1001), (3.0, 4.0));
        assert_eq!(frame_positions(-300, 100, 100), (-3.0, -2.0));
        // t1 adds the f32 time step to the f32 current time before dividing.
        assert_eq!(frame_positions(1, 3, 3), (1.0f32 / 3.0, (3.0f32 + 1.0) / 3.0));
        // An f32 cannot hold every i32: 16777217 rounds to 16777216.
        assert_eq!(frame_positions(16777217, 1, 1).0, 16777216.0);
        assert!(frame_positions(5, 1, 0).0.is_infinite());
    }

    #[test]
    fn source_display_time_adds_the_truncated_offset_in_frames() {
        assert_eq!(source_display_time(700, 100, 0), 700);
        assert_eq!(source_display_time(700, 100, fixed(2.5)), 900, "N = trunc(-2.5) = -2");
        assert_eq!(source_display_time(700, 100, fixed(-1.5)), 600, "N = trunc(1.5) = 1");
        assert_eq!(source_display_time(700, 100, fixed(0.99)), 700);
        assert_eq!(source_display_time(700, 100, fixed(1.0)), 800);
        assert_eq!(source_display_time(i32::MAX, 100, fixed(1.0)), i32::MAX.wrapping_add(100), "32 bit wrap");
    }

    // ---- Speed and Frame Number (section 5) ----

    fn pos(mock: &Mock, rates: Rates, t: f32) -> f32 {
        RemapTimeMap::new(mock, rates, 0).src_pos(t).unwrap()
    }

    #[test]
    fn constant_100_percent_is_t_times_the_rate_ratio() {
        let m = Mock::speed(100.0);
        for t in [0.0f32, 0.5, 1.0, 5.0, 7.5, 123.0, 2.25, 0.25, 10.75] {
            assert_eq!(pos(&m, plain_rates(), t), t, "t {t}");
        }
        // Not a multiple of one half: the partial last step still gives t for a constant speed.
        assert_eq!(pos(&m, plain_rates(), 0.2), 0.2);
        let t = 3.3f32;
        assert!((pos(&m, plain_rates(), t) - t).abs() < 1e-6);
        // 24 fps source in a 30 fps comp: s = 0.8 t.
        assert_eq!(pos(&m, rates(24.0, 30.0, 100), 5.0), 4.0);
        assert_eq!(pos(&m, rates(30.0, 24.0, 100), 4.0), 5.0);
    }

    #[test]
    fn constant_speeds_scale_the_position() {
        assert_eq!(pos(&Mock::speed(50.0), plain_rates(), 5.0), 2.5);
        assert_eq!(pos(&Mock::speed(200.0), plain_rates(), 3.0), 6.0);
        assert_eq!(pos(&Mock::speed(-100.0), plain_rates(), 3.0), -3.0);
        assert_eq!(pos(&Mock::speed(0.0), plain_rates(), 3.0), 0.0);
        // 50 %, t = 5.3: ten half steps of 0.25 plus (rem * 0.5) / 2 with rem = 10.6 - 10.
        let rem = 5.3f32 * 2.0 - 10.0;
        let want = ((rem * 0.5) / 2.0) + 2.5;
        assert_eq!(pos(&Mock::speed(50.0), plain_rates(), 5.3), want);
        assert!((want - 2.65).abs() < 1e-5);
    }

    #[test]
    fn negative_positions_give_zero() {
        for t in [-0.25f32, -1.0, -7.5, -1000.0] {
            let s = pos(&Mock::speed(100.0), plain_rates(), t);
            assert_eq!(s, 0.0, "t {t}");
            assert!(s.is_sign_positive());
        }
        // The partial step is skipped when the remainder is not above 0.0001: t = 0.00004.
        assert_eq!(pos(&Mock::speed(100.0), plain_rates(), 0.00004), 0.0);
        assert!(pos(&Mock::speed(100.0), plain_rates(), 0.0002) > 0.0);
    }

    #[test]
    fn speed_curve_is_integrated_in_half_frame_steps() {
        // 100 ticks per frame, so the half frame step k is sampled at trunc(50 k + 0.05) = 50 k. Speed is 100 %
        // before time 3050 and 200 % from there: steps 1 to 60 add 0.5 each, steps 61 and 62 add 1.0 each.
        let m = Mock { speed: Box::new(|t| if t < 3050 { fixed(100.0) } else { fixed(200.0) }), ..Mock::speed(100.0) };
        assert_eq!(pos(&m, plain_rates(), 31.0), 30.0 + 1.0 + 1.0);
        // Sampling at the half frame: t = 1.25 reads speed at 50, 100 and (partial step) 125.
        let m = Mock::speed(100.0);
        pos(&m, plain_rates(), 1.25);
        assert_eq!(m.reads(), vec![("mode", 0), ("offset", 125), ("speed", 50), ("speed", 100), ("speed", 125)]);
    }

    #[test]
    fn parameter_sampling_time_uses_the_bias_k() {
        // lts * t = 0.96 samples at trunc(0.96 + 0.05) = 1 (needs K above 0.04),
        // lts * t = 0.94 samples at trunc(0.94 + 0.05) = 0 (needs K below 0.06).
        let m = Mock { frame: Box::new(|t| t as f64), ..Mock::frame_number(0.0) };
        assert_eq!(pos(&m, plain_rates(), 0.0096), 1.0);
        assert_eq!(pos(&m, plain_rates(), 0.0094), 0.0);
        assert_eq!(pos(&m, plain_rates(), 5.0), 500.0);
        assert_eq!(pos(&m, plain_rates(), 2.5), 250.0);
        // Speed mode: the first half step is sampled at trunc(7 / 2 + K) = 3, the offset at trunc(7 * 0.5 + K) = 3.
        let m = Mock::speed(100.0);
        pos(&m, rates(30.0, 30.0, 7), 0.5);
        assert_eq!(m.reads(), vec![("mode", 0), ("offset", 3), ("speed", 3)]);
    }

    #[test]
    fn frame_number_mode() {
        assert_eq!(pos(&Mock::frame_number(12.5), plain_rates(), 4.0), 12.5);
        assert_eq!(pos(&Mock::frame_number(12.5), rates(24.0, 30.0, 100), 4.0), 10.0);
        assert_eq!(pos(&Mock::frame_number(-3.25), plain_rates(), 0.0), -3.25, "negative frame numbers extrapolate");
        // The value goes through f32: 0.1 as a double becomes the f32 0.1.
        assert_eq!(pos(&Mock::frame_number(0.1), plain_rates(), 1.0), 0.1f32);
        assert_eq!(pos(&Mock::frame_number(1e10), plain_rates(), 1.0), 1e10f32);
        // The parameter is read at trunc(lts * t + K); the mode only at the current time.
        let m = Mock::frame_number(4.0);
        RemapTimeMap::new(&m, plain_rates(), 700).src_pos(7.0).unwrap();
        assert_eq!(m.reads(), vec![("mode", 700), ("frame", 700)]);
    }

    #[test]
    fn base_is_a_side_effect_of_the_last_src_pos() {
        // Src Time Offset animated: one frame before time 100, two frames afterwards.
        let m = Mock::speed(100.0).with_offset(|t| if t < 100 { fixed(1.0) } else { fixed(2.0) });
        let map = RemapTimeMap::new(&m, plain_rates(), 0);
        assert_eq!(map.base(), 0);
        map.src_pos(0.5).unwrap(); // offset read at trunc(50.1) = 50
        assert_eq!(map.base(), -1);
        map.src_pos(1.5).unwrap(); // offset read at trunc(150.1) = 150
        assert_eq!(map.base(), -2, "frame_time and marker see the value left by t1");
        assert_eq!(map.frame_time(5.0), 700, "(5 - (-2)) * 100 + 0.5");
        // Fractional offsets truncate toward zero; negative offsets give a positive base.
        for (frames, want) in [(3.7, -3), (-3.7, 3), (0.99, 0), (-0.99, 0), (3.0, -3), (-3.0, 3)] {
            let m = Mock::speed(100.0).with_offset(move |_| fixed(frames));
            let map = RemapTimeMap::new(&m, plain_rates(), 0);
            map.src_pos(1.0).unwrap();
            assert_eq!(map.base(), want, "offset {frames}");
        }
        // Frame Number mode resets it to 0.
        let m = Mock::speed(100.0).with_offset(|_| fixed(4.0));
        let map = RemapTimeMap::new(&m, plain_rates(), 0);
        map.src_pos(1.0).unwrap();
        assert_eq!(map.base(), -4);
        m.mode.set(2);
        map.src_pos(1.0).unwrap();
        assert_eq!(map.base(), 0);
        m.mode.set(1);
        map.src_pos(1.0).unwrap();
        assert_eq!(map.base(), -4);
    }

    #[test]
    fn plugin_03_flavour_has_no_offset() {
        let m = Mock::speed(100.0).with_offset(|_| fixed(5.0));
        let map = RemapTimeMap::new(&m, plain_rates(), 0).without_src_time_offset();
        assert_eq!(map.src_pos(2.0).unwrap(), 2.0);
        assert_eq!(map.base(), 0);
        assert!(!m.reads().iter().any(|r| r.0 == "offset"));
        // Frame Number is the same function in both flavours.
        let m = Mock::frame_number(7.0);
        let map = RemapTimeMap::new(&m, plain_rates(), 0).without_src_time_offset();
        assert_eq!(map.src_pos(2.0).unwrap(), 7.0);
    }

    #[test]
    fn unknown_modes_are_errors_here() {
        for mode in [0, 3, -1] {
            let m = Mock::speed(100.0);
            m.mode.set(mode);
            let e = RemapTimeMap::new(&m, plain_rates(), 0).src_pos(1.0).unwrap_err();
            assert!(matches!(e, FxError::BadParam { .. }), "{e}");
        }
    }

    #[test]
    fn runaway_positions_are_refused_instead_of_looping_forever() {
        // k += 1.0 stops counting at 2^24, so 2^23 frames (2^24 half steps) would never
        // terminate.
        let m = Mock::speed(100.0);
        let map = RemapTimeMap::new(&m, plain_rates(), 0);
        assert!(map.src_pos(8388608.0).is_err());
        assert!(m.reads().len() <= 2, "refused before any sampling beyond the mode and offset");
        // NaN and infinity skip the loop (the truncation gives i32::MIN) and give 0.
        assert_eq!(pos(&Mock::speed(100.0), plain_rates(), f32::NAN).to_bits(), 0.0f32.to_bits());
        assert_eq!(pos(&Mock::speed(100.0), plain_rates(), f32::NEG_INFINITY).to_bits(), 0.0f32.to_bits());
    }

    // ---- frame_time and marker (section 5) ----

    #[test]
    fn frame_time_equal_rates_rounds_to_nearest_in_f64() {
        let m = Mock::speed(100.0);
        let map = RemapTimeMap::new(&m, plain_rates(), 0);
        assert_eq!(map.frame_time(0.0), 0);
        assert_eq!(map.frame_time(7.0), 700);
        assert_eq!(map.frame_time(7.5), 750);
        assert_eq!(map.frame_time(-1.0), -99, "-99.5 truncates toward zero");
        assert_eq!(map.frame_time(-7.0), -699);
        // Rates within FRAME_TIME_K of each other count as equal: 30 against 29.95 (difference 0.05 - 7.6e-7 after the
        // f32 slider, dur = 100.16695).
        let map = RemapTimeMap::new(&m, rates(29.95, 30.0, 100), 0);
        assert_eq!(map.frame_time(1.0), 100, "trunc(100.167 + 0.5); the biased form would give 105");
        assert_eq!(map.frame_time(2.0), 200, "trunc(200.334 + 0.5), what the original asked for; the biased form would give 205");
        assert_eq!(map.frame_time(3.0), 301, "trunc(300.501 + 0.5); the biased form would give 305");
        // 60 against 59.94 differ by 0.06: unequal (the original asks for 5, 105, 205 at a 60 fps output).
        let map = RemapTimeMap::new(&m, rates(59.94, 60.0, 100), 0);
        assert_eq!((map.frame_time(0.0), map.frame_time(1.0), map.frame_time(2.0)), (5, 105, 205));
    }

    #[test]
    fn frame_time_unequal_rates_adds_the_k_bias_to_an_f32_product() {
        let m = Mock::speed(100.0);
        // 24 fps source in a 30 fps comp: dur = 125. 7 * 125 + 125 * 0.05 = 881.25 -> 881.
        let map = RemapTimeMap::new(&m, rates(24.0, 30.0, 100), 0);
        assert_eq!(map.frame_time(7.0), 881);
        assert_eq!(map.frame_time(0.0), 6, "bias only: 6.25 -> 6");
        assert_eq!(map.frame_time(-7.0), -868, "-875 + 6.25 = -868.75 truncates toward zero");
        // 30 against 30.1 differ by 0.1: unequal. dur = 100.333336, 3 * dur + 0.05 * dur = 306.02 (the equal
        // rate form would give 301).
        let map = RemapTimeMap::new(&m, rates(30.0, 30.1, 100), 0);
        assert_eq!(map.frame_time(3.0), 306);
        // The checkout times the original asked for (fix 01): 25 fps in 24 (dur 96), 2 fps in 24 (dur 1200),
        // 0.5 fps in 24 (dur 4800), 29.91 fps in 30.
        let map = RemapTimeMap::new(&m, rates(25.0, 24.0, 100), 0);
        assert_eq!((0..5).map(|f| map.frame_time(f as f32)).collect::<Vec<_>>(), vec![4, 100, 196, 292, 388]);
        let map = RemapTimeMap::new(&m, rates(2.0, 24.0, 100), 0);
        assert_eq!((map.frame_time(0.0), map.frame_time(1.0)), (60, 1260));
        let map = RemapTimeMap::new(&m, rates(0.5, 24.0, 100), 0);
        assert_eq!((map.frame_time(0.0), map.frame_time(1.0)), (240, 5040));
        let map = RemapTimeMap::new(&m, rates(29.91, 30.0, 100), 0);
        assert_eq!((0..4).map(|f| map.frame_time(f as f32)).collect::<Vec<_>>(), vec![5, 105, 205, 305]);
        // The product is an f32 product: f * dur differs between the f32 and the f64 product after the bias.
        let r = rates(30.0, 24.0, 33);
        let map = RemapTimeMap::new(&m, r, 0);
        let f = 79441.0f32;
        let f32_product = cvtt_i32((f * r.dur) as f64 + (r.dur as f64) * FRAME_TIME_K);
        assert_eq!(map.frame_time(f), f32_product);
        let differs = (1..200000).map(|n| n as f32).find(|&g| {
            cvtt_i32((g * r.dur) as f64 + (r.dur as f64) * FRAME_TIME_K) != cvtt_i32((g as f64) * (r.dur as f64) + (r.dur as f64) * FRAME_TIME_K)
        });
        let g = differs.expect("some frame number separates the f32 from the f64 product");
        assert_eq!(map.frame_time(g), cvtt_i32((g * r.dur) as f64 + (r.dur as f64) * FRAME_TIME_K));
    }

    #[test]
    fn frame_time_subtracts_base() {
        let m = Mock::speed(100.0).with_offset(|_| fixed(2.0));
        let map = RemapTimeMap::new(&m, plain_rates(), 0);
        assert_eq!(map.frame_time(5.0), 500, "base is 0 before the first src_pos");
        map.src_pos(1.0).unwrap();
        assert_eq!(map.frame_time(5.0), 700);
        assert_eq!(map.frame_time(2.0), 400, "frame 2 - (-2) = 4");
        let map = RemapTimeMap::new(&m, rates(24.0, 30.0, 100), 0);
        map.src_pos(1.0).unwrap();
        assert_eq!(map.frame_time(5.0), 881, "(5 + 2) * 125 + 6.25");
    }

    #[test]
    fn marker_reads_the_popup_at_the_rounded_frame_time() {
        let m = Mock::speed(100.0).with_marks(|t| match t {
            300 => 3,
            600 => 5,
            900 => 6,
            100 => 0,
            200 => -2,
            _ => 1,
        });
        let map = RemapTimeMap::new(&m, plain_rates(), 0);
        assert_eq!(map.marker(3.0).unwrap(), 2, "value 3 is Cut C");
        assert_eq!(map.marker(6.0).unwrap(), 4, "value 5 is No MV, Frame Blend");
        assert_eq!(map.marker(9.0).unwrap(), 0, "out of range values give 0");
        assert_eq!(map.marker(1.0).unwrap(), 0);
        assert_eq!(map.marker(2.0).unwrap(), 0);
        assert_eq!(map.marker(4.0).unwrap(), 0, "value 1 is Cut A");
        let reads: Vec<_> = m.reads().into_iter().filter(|r| r.0 == "marks").map(|r| r.1).collect();
        assert_eq!(reads, vec![300, 600, 900, 100, 200, 400], "trunc(f * 100 + 0.5)");
        // The offset base moves the sampling time: base -2 asks for frame 3 at time 500.
        let m = Mock::speed(100.0).with_offset(|_| fixed(2.0)).with_marks(|t| if t == 500 { 4 } else { 1 });
        let map = RemapTimeMap::new(&m, plain_rates(), 0);
        map.src_pos(1.0).unwrap();
        assert_eq!(map.marker(3.0).unwrap(), 3);
        // Unequal rates use dur for the marker too, always with the +0.5 rounding and never the K bias.
        let m = Mock::speed(100.0).with_marks(|t| if t == 625 { 2 } else { 1 });
        let map = RemapTimeMap::new(&m, rates(24.0, 30.0, 100), 0);
        assert_eq!(map.marker(5.0).unwrap(), 1, "5 * 125 + 0.5 = 625");
    }

    // ---- frame selection (section 6) ----

    /// A time map with scripted positions and markers, logging the order of the calls.
    struct Script {
        pos: Box<dyn Fn(f32) -> f32>,
        mark: Box<dyn Fn(f32) -> i32>,
        calls: RefCell<Vec<String>>,
    }

    impl Script {
        fn new(pos: impl Fn(f32) -> f32 + 'static, mark: impl Fn(f32) -> i32 + 'static) -> Script {
            Script { pos: Box::new(pos), mark: Box::new(mark), calls: RefCell::new(Vec::new()) }
        }
    }

    impl TimeMap for Script {
        fn src_pos(&self, t: f32) -> Result<f32> {
            self.calls.borrow_mut().push(format!("pos {t}"));
            Ok((self.pos)(t))
        }
        fn frame_time(&self, f: f32) -> i32 {
            f as i32
        }
        fn marker(&self, f: f32) -> Result<i32> {
            self.calls.borrow_mut().push(format!("marker {f}"));
            Ok((self.mark)(f))
        }
    }

    #[test]
    fn selection_orders_the_calls_and_computes_the_weights() {
        let s = Script::new(|t| if t == 7.0 { 5.25 } else { 5.5 }, |_| 0);
        let sel = select(&s, 7.0, 8.0, 4, 2, 0.0).unwrap();
        assert_eq!(*s.calls.borrow(), ["pos 7", "pos 8", "marker 5", "marker 6"]);
        assert_eq!((sel.s0, sel.s1, sel.a, sel.b, sel.wa), (5.25, 5.5, 5, 6, 0.75));
        assert_eq!((sel.vectors_quality, sel.interp, sel.exact), (4, 2, false));
        assert_eq!(sel.blend_weight(), 0.25);
        assert!(sel.nearest_is_a(), "0.75 >= 0.5 picks A");
    }

    #[test]
    fn selection_weights_for_special_positions() {
        let sel = |s0: f32| select(&Script::new(move |_| s0, |_| 0), 0.0, 1.0, 4, 1, 0.0).unwrap();
        let on_frame = sel(2.0);
        assert_eq!((on_frame.a, on_frame.b, on_frame.wa, on_frame.exact), (2, 3, 1.0, true));
        assert_eq!(on_frame.blend_weight(), 0.0);
        let half = sel(3.5);
        assert_eq!((half.wa, half.nearest_is_a(), half.blend_weight()), (0.5, true, 0.5), "frac == 0.5 picks A");
        let just_past = sel(f32::from_bits(3.5f32.to_bits() + 1));
        assert!(just_past.wa < 0.5 && !just_past.nearest_is_a());
        // Negative positions truncate toward zero, so the weight of A exceeds 1.
        let neg = sel(-0.5);
        assert_eq!((neg.a, neg.b, neg.wa, neg.blend_weight()), (0, 1, 1.5, -0.5));
        let neg2 = sel(-2.5);
        assert_eq!((neg2.a, neg2.b, neg2.wa), (-2, -1, 1.5));
        let neg3 = sel(-1.0);
        assert_eq!((neg3.a, neg3.b, neg3.wa, neg3.exact), (-1, 0, 1.0, true));
    }

    #[test]
    fn exactness_thresholds_on_the_fraction() {
        let ex = |s0: f32| exact_position(s0, s0, 0.0);
        assert!(ex(5.0));
        assert!(ex(5.004), "fraction 0.004");
        assert!(!ex(5.006), "fraction 0.006");
        assert!(!ex(5.5));
        assert!(!ex(5.994), "1 - 0.994 = 0.006");
        assert!(ex(5.996), "1 - 0.996 = 0.004");
        assert!(ex(-5.0));
        assert!(!ex(-5.5));
        // 5.005 is stored as 5.0050001 in f32, so its fraction is just above 0.005: not exact.
        assert!(!ex(5.005));
        assert!(ex(f32::from_bits(5.0f32.to_bits() + 1)));
        // One f32 step below 6.0 is 4.8e-7 away from the next frame: exact.
        assert!(ex(f32::from_bits(6.0f32.to_bits() - 1)));
    }

    #[test]
    fn exactness_with_blur_compensation() {
        // Two consecutive positions further apart than 1e-6 and blur above 0.005 force the vector path.
        assert!(!exact_position(5.0, 6.0, 0.01));
        assert!(exact_position(5.0, 6.0, 0.0));
        assert!(exact_position(5.0, 6.0, 0.005), "0.005 as f32 is 0.00499999989, not above 0.005");
        assert!(!exact_position(5.0, 6.0, 0.0051));
        // Positions within 1e-6 do not count as different, whatever the blur.
        assert!(exact_position(5.0, 5.0, 10.0));
        assert!(exact_position(5.0, 5.0 + f32::EPSILON * 8.0, 10.0), "9.5e-7 apart");
        assert!(!exact_position(5.0, 5.0 + f32::EPSILON * 16.0, 10.0), "1.9e-6 apart");
        assert!(!exact_position(5.5, 5.5, 0.0), "blur never makes a fractional position exact");
        assert!(exact_position(5.0, 4.0, 0.0), "the order of the arguments does not matter");
        assert!(!exact_position(5.0, 4.0, 1.0));
    }

    #[test]
    fn segment_override_table() {
        // a = 10, g = 11 for s0 = 10.5; the marker closure assigns the two frames.
        let run = |ma: i32, mb: i32, planner: bool| {
            let s = Script::new(|t| t, move |f| if f == 10.0 { ma } else { mb });
            marker_override(&s, 10.5, 4, 2, planner).unwrap()
        };
        for planner in [false, true] {
            assert_eq!(run(0, 0, planner), (4, 2), "no cut");
            assert_eq!(run(1, 1, planner), (4, 2));
            assert_eq!(run(2, 2, planner), (4, 2));
            assert_eq!(run(0, 1, planner), (0, 0), "a cut between the two frames: Nearest, no vectors");
            assert_eq!(run(1, 0, planner), (0, 0));
            assert_eq!(run(0, 4, planner), (0, 0));
            assert_eq!(run(3, 3, planner), (0, 0), "No MV, Use Nearest");
            assert_eq!(run(3, 0, planner), (0, 0));
            assert_eq!(run(4, 4, planner), (0, 1), "No MV, Frame Blend");
            assert_eq!(run(4, 0, planner), (0, 1), "the later test wins over the cut test");
            assert_eq!(run(4, 3, planner), (0, 1));
        }
        // Markers are asked for the frames a and g only.
        let s = Script::new(|t| t, |_| 0);
        marker_override(&s, 10.5, 4, 2, false).unwrap();
        assert_eq!(*s.calls.borrow(), ["marker 10", "marker 11"]);
    }

    #[test]
    fn planner_and_core_differ_for_negative_positions() {
        // s0 = -0.5: a = 0, the core tests g = trunc(0.5) = 0 (the same frame), the planner replaces g < 1 by a + 1 = 1.
        let s = Script::new(|t| t, |f| if f == 0.0 { 0 } else { 2 });
        assert_eq!(marker_override(&s, -0.5, 4, 2, false).unwrap(), (4, 2));
        assert_eq!(marker_override(&s, -0.5, 4, 2, true).unwrap(), (0, 0));
        // Everywhere else both forms agree (g = a + 1, or g = trunc(s0 + 1) = a + 1 below -1).
        for s0 in [0.0f32, 0.5, 3.5, -1.5, -2.5, -10.25] {
            let s = Script::new(|t| t, |f| if f == 0.0 { 0 } else { 2 });
            assert_eq!(
                marker_override(&s, s0, 4, 2, false).unwrap(),
                marker_override(&s, s0, 4, 2, true).unwrap(),
                "s0 {s0}"
            );
        }
    }

    #[test]
    fn selection_applies_the_core_override_and_reports_the_exact_flag() {
        // a = 3 is a "No MV, Frame Blend" frame: vectors off, Blend, whatever the settings were.
        let s = Script::new(|t| t * 0.5, |f| if f == 3.0 { 4 } else { 0 });
        let sel = select(&s, 7.0, 8.0, 4, 2, 0.0).unwrap();
        assert_eq!((sel.a, sel.b, sel.wa, sel.vectors_quality, sel.interp), (3, 4, 0.5, 0, 1));
        assert!(!sel.exact);
    }

    #[test]
    fn the_speed_cache_gives_bit_identical_positions_with_far_fewer_reads() {
        // An animated speed (a ramp with a few steps) over 3000 ticks of 100 per frame, odd lts to avoid aligned samples.
        let curve = |t: i32| fixed(40.0 + ((t / 37) % 17) as f64 * 11.5 - if t > 2500 { 90.0 } else { 0.0 });
        let make = |cache: bool| {
            let m = Mock { speed: Box::new(curve), ..Mock::speed(100.0) }.with_offset(|t| if t < 1000 { 0 } else { fixed(1.0) });
            let map = RemapTimeMap::new(m, rates(24.0, 30.0, 83), 0);
            if cache { map.with_speed_cache() } else { map }
        };
        let (plain, cached) = (make(false), make(true));
        // Out of order, repeated and negative positions, with fractional parts.
        let ts = [0.0f32, 0.25, 3.0, 17.5, 4.3, 120.9, 120.9, 60.1, -2.0, 0.1, 300.6, 299.5, 301.0, 1.0, 0.0001];
        for t in ts {
            let (a, b) = (plain.src_pos(t).unwrap(), cached.src_pos(t).unwrap());
            assert_eq!(a.to_bits(), b.to_bits(), "t {t}");
            assert_eq!(plain.base(), cached.base());
        }
        let count = |m: &RemapTimeMap<Mock>| m.params.reads().iter().filter(|r| r.0 == "speed").count();
        assert!(count(&cached) < count(&plain) / 3, "{} against {}", count(&cached), count(&plain));
        // The partial step still samples at its own time every call.
        let m = Mock::speed(100.0);
        let map = RemapTimeMap::new(&m, plain_rates(), 0).with_speed_cache();
        map.src_pos(1.25).unwrap();
        map.src_pos(1.25).unwrap();
        let speed: Vec<i32> = m.reads().into_iter().filter(|r| r.0 == "speed").map(|r| r.1).collect();
        assert_eq!(speed, vec![50, 100, 125, 125], "two cached steps, the partial step is sampled again");
    }


    /// Deterministic parameters for the cross check table below (integer formulas only, so the numpy
    /// reference and this test cannot disagree about a parameter value).
    struct Det {
        mode: i32,
    }

    impl TimeParams for Det {
        fn remap_mode(&self, _time: i32) -> i32 {
            self.mode
        }
        fn speed_fixed(&self, t: i32) -> i32 {
            ((t / 37) % 17) * 753664 + 2621440
        }
        fn frame_num(&self, t: i32) -> f64 {
            (t.abs() % 977) as f64 / 8.0 - 20.0
        }
        fn src_time_offset_fixed(&self, t: i32) -> i32 {
            let t = t.abs();
            65536 * ((t / 211) % 4) + (t % 7) * 1000
        }
        fn mark_segs(&self, t: i32) -> i32 {
            (t.abs() / 150) % 6
        }
    }

    // (lts, typed input rate, composition rate, mode, t bits, expected src_pos bits, expected base,
    //  a = trunc(s0), frame_time(a), frame_time(a + 1), marker(a), expected dur bits)
    // Produced by an independent implementation of the same operations in numpy float32 / Python
    // float64 (scratch script, seed 20261022): rate model with snapping, src_pos in both modes,
    // frame_time with both branches, marker. 160 cases over lts 7 to 3003, ten rate pairs,
    // animated speed and offset, random fractional positions and a few negative ones.
    const CASES: [(i32, f64, f64, i32, u32, u32, i32, i32, i32, i32, i32, u32); 160] = [
        (125, 30.0, 24.0, 1, 0x41f00000, 0x424ee99a, -1, 51, 5205, 5305, 3, 0x42c80000),
        (83, 29.969999313354492, 29.97002997002997, 1, 0x413d9ad8, 0x4166aec8, 0, 14, 1162, 1245, 0, 0x42a6000b),
        (83, 29.969999313354492, 29.97002997002997, 1, 0x426cc226, 0x429b1932, -3, 77, 6640, 6723, 1, 0x42a6000b),
        (3003, 24.0, 30.0, 1, 0x41600000, 0x416ec083, -3, 14, 64001, 67755, 4, 0x456a9c00),
        (125, 29.969999313354492, 29.97002997002997, 1, 0x42045240, 0x423175fa, -3, 44, 5875, 6000, 2, 0x42fa0010),
        (1001, 15.0, 25.0, 1, 0x41900000, 0x416ced90, -1, 14, 25108, 26776, 3, 0x44d08aab),
        (7, 60.0, 59.94, 1, 0x41e00000, 0x4192c96e, 0, 18, 126, 133, 0, 0x40dfc6a8),
        (125, 59.939998626708984, 60.0, 1, 0x421d3a0f, 0x4254a1c4, -3, 53, 7013, 7138, 3, 0x42fa4010),
        (100, 23.97599983215332, 23.976023976023978, 1, 0x428c7de3, 0x42b888dc, -1, 92, 9300, 9400, 1, 0x42c8000d),
        (40, 60.0, 59.94, 1, 0x4287d457, 0x42ac47ef, 0, 86, 3438, 3478, 3, 0x421fd70a),
        (1001, 24.0, 30.0, 1, 0x3f04d149, 0x3f49cb69, -2, 0, 2565, 3816, 3, 0x449c6800),
        (7, 24.0, 30.0, 1, 0xc0173585, 0x00000000, 0, 0, 0, 9, 0, 0x410c0000),
        (3003, 24.0, 30.0, 1, 0x4203a102, 0x42067482, 0, 33, 124061, 127815, 2, 0x456a9c00),
        (7, 15.0, 25.0, 1, 0x41ab4c60, 0x40eebe6f, 0, 7, 82, 93, 0, 0x413aaaab),
        (40, 30.0, 30.0, 2, 0x42840000, 0x42838000, 0, 65, 2600, 2640, 4, 0x42200000),
        (3003, 59.939998626708984, 60.0, 1, 0x40a00000, 0x40dc2dfb, -3, 6, 27204, 30210, 0, 0x453be019),
        (7, 25.0, 30.0, 1, 0xc09a801d, 0x00000000, 0, 0, 0, 8, 0, 0x41066666),
        (125, 25.0, 30.0, 1, 0xbf7c0085, 0x00000000, 0, 0, 7, 157, 0, 0x43160000),
        (7, 30.0, 30.0, 1, 0x4225a4a9, 0x420470aa, -1, 33, 238, 245, 0, 0x40e00000),
        (3003, 30.0, 24.0, 1, 0xbfbcf2ca, 0x00000000, -1, 0, 2522, 4924, 3, 0x45162666),
        (125, 15.0, 25.0, 2, 0x428283b4, 0x41593333, 0, 13, 2718, 2927, 0, 0x43505555),
        (1001, 48.0, 60.0, 2, 0x41e33b37, 0xc0c33333, 0, -6, -7444, -6193, 1, 0x449c6800),
        (40, 23.97599983215332, 23.976023976023978, 2, 0x41ffca0f, 0x418cfff7, 0, 17, 680, 720, 3, 0x4220000a),
        (83, 30.0, 24.0, 1, 0x42820000, 0x42d39fff, -1, 105, 7041, 7108, 3, 0x4284cccd),
        (7, 25.0, 30.0, 1, 0xc09944b7, 0x00000000, 0, 0, 0, 8, 0, 0x41066666),
        (3003, 48.0, 60.0, 1, 0x4241d0e6, 0x4247626d, -1, 49, 187875, 191628, 2, 0x456a9c00),
        (83, 59.939998626708984, 60.0, 2, 0x42780000, 0x4149cc4a, 0, 12, 1001, 1084, 0, 0x42a62a8a),
        (3003, 23.97599983215332, 23.976023976023978, 2, 0x415b49e6, 0xc093fff6, 0, -4, -12011, -9008, 1, 0x453bb00c),
        (125, 29.969999313354492, 29.97002997002997, 2, 0x4234d126, 0x42973ff6, 0, 75, 9375, 9500, 1, 0x42fa0010),
        (100, 30.0, 24.0, 1, 0x40e00000, 0x412c2665, -3, 10, 1044, 1124, 0, 0x42a00000),
        (40, 30.0, 24.0, 2, 0x418cbec9, 0x42a9b000, 0, 84, 2689, 2721, 4, 0x42000000),
        (3003, 25.0, 30.0, 1, 0x4225bffe, 0x42356a1e, -1, 45, 165945, 169549, 0, 0x4561399a),
        (125, 15.0, 25.0, 1, 0x41c80000, 0x41a6676e, -2, 20, 4593, 4802, 0, 0x43505555),
        (100, 23.97599983215332, 23.976023976023978, 1, 0x41f00000, 0x421c8f52, -2, 39, 4100, 4200, 2, 0x42c8000d),
        (7, 60.0, 59.94, 2, 0x42460000, 0x41ba2faa, 0, 23, 161, 168, 0, 0x40dfc6a8),
        (125, 30.0, 24.0, 2, 0x40911ddb, 0x427dc000, 0, 63, 6305, 6405, 0, 0x42c80000),
        (125, 48.0, 60.0, 1, 0x42100000, 0x421b8f5b, -1, 38, 6101, 6257, 3, 0x431c4000),
        (3003, 60.0, 59.94, 1, 0x420e25dc, 0x4237f5ce, -1, 45, 138149, 141149, 1, 0x453b7ff4),
        (125, 30.0, 30.0, 1, 0x41e00000, 0x421623d7, 0, 37, 4625, 4750, 0, 0x42fa0000),
        (125, 23.97599983215332, 23.976023976023978, 1, 0x42104843, 0x4242a0ab, -1, 48, 6125, 6250, 3, 0x42fa0010),
        (125, 15.0, 25.0, 2, 0xbec5942a, 0xc1066666, 0, -8, -1656, -1447, 4, 0x43505555),
        (3003, 24.0, 30.0, 1, 0x42842feb, 0x4288c13e, 0, 68, 255442, 259196, 2, 0x456a9c00),
        (125, 29.969999313354492, 29.97002997002997, 2, 0x41e0c6be, 0x42527ff2, 0, 52, 6500, 6625, 0, 0x42fa0010),
        (1001, 15.0, 25.0, 1, 0x420c0000, 0x41e41ba0, -2, 28, 50133, 51801, 2, 0x44d08aab),
        (1001, 15.0, 25.0, 2, 0x428c0000, 0x4222e666, 0, 40, 66816, 68485, 0, 0x44d08aab),
        (1001, 59.939998626708984, 60.0, 1, 0x42340000, 0x426f3355, -1, 59, 60170, 61172, 3, 0x447a8021),
        (3003, 30.0, 24.0, 1, 0x429c0000, 0x42fbeffc, -2, 125, 305224, 307627, 0, 0x45162666),
        (7, 60.0, 59.94, 1, 0x42220000, 0x42002d9f, -1, 32, 231, 238, 0, 0x40dfc6a8),
        (83, 23.97599983215332, 23.976023976023978, 2, 0x41906fee, 0x42347ff4, 0, 45, 3735, 3818, 0, 0x42a6000b),
        (3003, 29.969999313354492, 29.97002997002997, 2, 0x42080acc, 0x42367ff4, 0, 45, 135135, 138138, 0, 0x453bb00c),
        (1001, 48.0, 60.0, 1, 0x429c0000, 0x42a549bb, -2, 82, 105167, 106418, 3, 0x449c6800),
        (1001, 29.969999313354492, 29.97002997002997, 1, 0x4257501e, 0x428ead8a, -3, 71, 74074, 75075, 0, 0x447a4010),
        (100, 30.0, 24.0, 1, 0x4231711d, 0x42926c46, -1, 73, 5924, 6004, 2, 0x42a00000),
        (100, 48.0, 60.0, 1, 0x422a0000, 0x4230c28e, 0, 44, 5506, 5631, 0, 0x42fa0000),
        (1001, 59.939998626708984, 60.0, 1, 0xbf888189, 0x00000000, -1, 0, 1052, 2054, 0, 0x447a8021),
        (100, 25.0, 30.0, 1, 0x428a0000, 0x4298a999, 0, 76, 9126, 9246, 0, 0x42f00000),
        (83, 15.0, 25.0, 1, 0x41f926d0, 0x41c43848, 0, 24, 3326, 3465, 3, 0x430a5555),
        (100, 25.0, 30.0, 1, 0x424ec33e, 0x4260f711, 0, 56, 6726, 6846, 1, 0x42f00000),
        (83, 15.0, 25.0, 1, 0xbf2d05c6, 0x00000000, 0, 0, 6, 145, 0, 0x430a5555),
        (1001, 30.0, 30.0, 1, 0x406a5d3c, 0x40b80418, -1, 5, 6006, 7007, 3, 0x447a4000),
        (1001, 48.0, 60.0, 1, 0x428b0000, 0x4292820c, -1, 73, 92655, 93906, 4, 0x449c6800),
        (1001, 30.0, 30.0, 2, 0x40905dec, 0x42600000, 0, 56, 56056, 57057, 0, 0x447a4000),
        (83, 30.0, 24.0, 1, 0x42036380, 0x42534e11, 0, 52, 3456, 3522, 4, 0x4284cccd),
        (3003, 30.0, 24.0, 1, 0x420c0000, 0x42621ccc, -2, 56, 139459, 141861, 3, 0x45162666),
        (100, 29.969999313354492, 29.97002997002997, 2, 0x40e00000, 0x4286fff7, 0, 67, 6700, 6800, 1, 0x42c8000d),
        (7, 29.969999313354492, 29.97002997002997, 1, 0x41a40000, 0x413b51e5, 0, 11, 77, 84, 0, 0x40e0000f),
        (3003, 24.0, 30.0, 1, 0x42149306, 0x421a8728, 0, 38, 142830, 146583, 1, 0x456a9c00),
        (125, 30.0, 30.0, 1, 0x4097b656, 0x40cf5aff, -2, 6, 1000, 1125, 0, 0x42fa0000),
        (7, 30.0, 24.0, 1, 0x42960000, 0x42da7669, -2, 109, 621, 627, 3, 0x40b33333),
        (1001, 29.969999313354492, 29.97002997002997, 2, 0x42b184cc, 0x42bd7ff4, 0, 94, 94094, 95095, 2, 0x447a4010),
        (125, 15.0, 25.0, 2, 0x41df03e8, 0x41ebcccd, 0, 29, 6052, 6260, 3, 0x43505555),
        (40, 23.97599983215332, 23.976023976023978, 1, 0x41700000, 0x419d0515, -2, 19, 840, 880, 4, 0x4220000a),
        (40, 30.0, 30.0, 2, 0x42240000, 0x427b8000, 0, 62, 2480, 2520, 3, 0x42200000),
        (100, 29.969999313354492, 29.97002997002997, 1, 0x40400000, 0x402c7ad7, -1, 2, 300, 400, 1, 0x42c8000d),
        (3003, 25.0, 30.0, 1, 0x418f6366, 0x419a37ee, -3, 19, 79459, 83062, 0, 0x4561399a),
        (1001, 30.0, 24.0, 1, 0x427936b9, 0x42cf1727, -3, 103, 84924, 85725, 0, 0x44483333),
        (125, 60.0, 59.94, 2, 0x428badaa, 0x42bc703e, 0, 94, 11744, 11869, 0, 0x42f9c000),
        (7, 60.0, 59.94, 1, 0xc08802a2, 0x00000000, 0, 0, 0, 7, 0, 0x40dfc6a8),
        (125, 60.0, 59.94, 1, 0x428d79d1, 0x42bb128a, -1, 93, 11744, 11869, 0, 0x42f9c000),
        (100, 23.97599983215332, 23.976023976023978, 1, 0x41a00000, 0x41cd3846, -1, 25, 2600, 2700, 4, 0x42c8000d),
        (100, 60.0, 59.94, 2, 0xc068d11d, 0x41cb3405, 0, 25, 2502, 2602, 3, 0x42c7cccd),
        (40, 30.0, 24.0, 1, 0x41d80000, 0x4224f99a, -1, 41, 1345, 1377, 1, 0x42000000),
        (1001, 23.97599983215332, 23.976023976023978, 1, 0x41f00000, 0x4222fff1, -2, 40, 42042, 43043, 3, 0x447a4010),
        (83, 59.939998626708984, 60.0, 1, 0x42100000, 0x423b979c, -2, 46, 3992, 4075, 1, 0x42a62a8a),
        (100, 30.0, 30.0, 2, 0x428bb138, 0xbff00000, 0, -1, -99, 0, 0, 0x42c80000),
        (7, 60.0, 59.94, 2, 0x42200000, 0x41703d80, 0, 15, 105, 112, 0, 0x40dfc6a8),
        (1001, 25.0, 30.0, 1, 0x41300000, 0x41533bbb, 0, 13, 15675, 16876, 1, 0x44962666),
        (7, 59.939998626708984, 60.0, 1, 0x4266ef4e, 0x42613e56, -1, 56, 399, 406, 1, 0x40e03967),
        (3003, 48.0, 60.0, 1, 0x422d27ac, 0x4234c745, 0, 45, 169106, 172860, 3, 0x456a9c00),
        (83, 25.0, 30.0, 2, 0x4290aa5a, 0xbffd5555, 0, -1, -94, 4, 0, 0x42c73333),
        (7, 30.0, 24.0, 2, 0x41e48aa7, 0x40c80000, 0, 6, 33, 39, 0, 0x40b33333),
        (40, 29.969999313354492, 29.97002997002997, 1, 0x428c0000, 0x42b0dd64, -1, 88, 3560, 3600, 4, 0x4220000a),
        (125, 29.969999313354492, 29.97002997002997, 1, 0x40b00000, 0x40ea28e8, -3, 7, 1250, 1375, 1, 0x42fa0010),
        (100, 24.0, 30.0, 1, 0x42990000, 0x42a03020, 0, 80, 10006, 10131, 0, 0x42fa0000),
        (83, 59.939998626708984, 60.0, 1, 0x429ea4b9, 0x42ced40d, -3, 103, 8810, 8894, 3, 0x42a62a8a),
        (100, 29.969999313354492, 29.97002997002997, 2, 0x40900000, 0x4210fff6, 0, 36, 3600, 3700, 0, 0x42c8000d),
        (3003, 15.0, 25.0, 2, 0x41c79097, 0x4215b333, 0, 37, 185435, 190440, 3, 0x459c6800),
        (83, 60.0, 59.94, 1, 0x42860000, 0x42b0b632, -2, 88, 7466, 7549, 0, 0x42a5d581),
        (83, 48.0, 60.0, 2, 0x417268da, 0x41400000, 0, 12, 1250, 1353, 1, 0x42cf8000),
        (40, 25.0, 30.0, 1, 0x41a396bd, 0x41a021f2, -3, 20, 1106, 1154, 0, 0x42400000),
        (125, 25.0, 30.0, 1, 0x41f00000, 0x4209f111, -1, 34, 5257, 5407, 4, 0x43160000),
        (7, 23.97599983215332, 23.976023976023978, 1, 0x42860000, 0x4290732b, -2, 72, 518, 525, 2, 0x40e0000f),
        (125, 15.0, 25.0, 1, 0x4116fac9, 0x40f4ee4e, -1, 7, 1677, 1885, 4, 0x43505555),
        (83, 23.97599983215332, 23.976023976023978, 1, 0x423af9a1, 0x4271fe7a, -2, 60, 5146, 5229, 3, 0x42a6000b),
        (40, 23.97599983215332, 23.976023976023978, 1, 0x42640000, 0x428fb1e3, -2, 71, 2920, 2960, 0, 0x4220000a),
        (83, 30.0, 30.0, 2, 0x4206cd47, 0x42aac000, 0, 85, 7055, 7138, 4, 0x42a60000),
        (83, 48.0, 60.0, 1, 0x40900000, 0x40639582, -1, 3, 420, 523, 1, 0x42cf8000),
        (1001, 59.939998626708984, 60.0, 1, 0x42a516c6, 0x42db2c30, -3, 109, 112274, 113276, 3, 0x447a8021),
        (40, 59.939998626708984, 60.0, 1, 0x42000000, 0x42259303, -2, 41, 1723, 1763, 4, 0x42202900),
        (40, 48.0, 60.0, 1, 0x4299c4a4, 0x42a01d23, -2, 80, 4102, 4152, 2, 0x42480000),
        (83, 48.0, 60.0, 1, 0x41831df1, 0x4187a28c, -2, 16, 1872, 1976, 0, 0x42cf8000),
        (83, 29.969999313354492, 29.97002997002997, 1, 0x42766ad3, 0x42a1cd1f, 0, 80, 6640, 6723, 1, 0x42a6000b),
        (1001, 15.0, 25.0, 2, 0x420c0000, 0x424c0000, 0, 51, 85168, 86836, 2, 0x44d08aab),
        (7, 15.0, 25.0, 2, 0xc03c47c0, 0xc1280000, 0, -10, -116, -104, 0, 0x413aaaab),
        (125, 24.0, 30.0, 2, 0x41686510, 0x4287999a, 0, 67, 10476, 10632, 2, 0x431c4000),
        (100, 24.0, 30.0, 2, 0x4277e598, 0x418c0000, 0, 17, 2131, 2256, 1, 0x42fa0000),
        (3003, 25.0, 30.0, 1, 0x42940000, 0x42a04eee, -1, 80, 292071, 295675, 0, 0x4561399a),
        (1001, 24.0, 30.0, 1, 0x42918884, 0x429a1cc7, -1, 77, 97660, 98911, 1, 0x449c6800),
        (100, 29.969999313354492, 29.97002997002997, 1, 0x41fc0000, 0x4225a13d, -2, 41, 4300, 4400, 3, 0x42c8000d),
        (3003, 24.0, 30.0, 2, 0xbfbd616f, 0x4215999a, 0, 37, 139076, 142830, 0, 0x456a9c00),
        (3003, 24.0, 30.0, 1, 0x41300000, 0x41391eb8, 0, 11, 41478, 45232, 4, 0x456a9c00),
        (40, 15.0, 25.0, 1, 0x424fc1e4, 0x421cddb6, -1, 39, 2669, 2736, 4, 0x42855555),
        (100, 23.97599983215332, 23.976023976023978, 1, 0x42930000, 0x42c0a8e8, -2, 96, 9800, 9900, 4, 0x42c8000d),
        (40, 30.0, 30.0, 1, 0xc065f722, 0x00000000, 0, 0, 0, 40, 0, 0x42200000),
        (100, 48.0, 60.0, 1, 0x42080000, 0x420b0418, 0, 34, 4256, 4381, 3, 0x42fa0000),
        (125, 25.0, 30.0, 1, 0x40000000, 0x3fb15556, -1, 1, 307, 457, 1, 0x43160000),
        (7, 48.0, 60.0, 2, 0xc03b29dc, 0xc1600000, 0, -14, -122, -113, 0, 0x410c0000),
        (40, 25.0, 30.0, 1, 0x41900000, 0x41928446, -3, 18, 1010, 1058, 0, 0x42400000),
        (1001, 29.969999313354492, 29.97002997002997, 2, 0x40eb17ac, 0x42317ff4, 0, 44, 44044, 45045, 4, 0x447a4010),
        (1001, 29.969999313354492, 29.97002997002997, 1, 0x42940000, 0x42c46512, -3, 98, 101101, 102102, 1, 0x447a4010),
        (1001, 30.0, 30.0, 1, 0x42560000, 0x428d95c2, -1, 70, 71071, 72072, 4, 0x447a4000),
        (40, 29.969999313354492, 29.97002997002997, 1, 0x429fd6ea, 0x42d078dd, -3, 104, 4280, 4320, 3, 0x4220000a),
        (7, 30.0, 30.0, 2, 0x42960000, 0x42368000, 0, 45, 315, 322, 1, 0x40e00000),
        (40, 25.0, 30.0, 2, 0x42832a07, 0x42541555, 0, 53, 2546, 2594, 3, 0x42400000),
        (83, 29.969999313354492, 29.97002997002997, 2, 0x4260c0de, 0x4294bff6, 0, 74, 6142, 6225, 3, 0x42a6000b),
        (83, 60.0, 59.94, 2, 0x4284791c, 0x426239ea, 0, 56, 4647, 4730, 0, 0x42a5d581),
        (100, 60.0, 59.94, 1, 0xc08814d6, 0x00000000, -2, 0, 204, 304, 0, 0x42c7cccd),
        (7, 59.939998626708984, 60.0, 1, 0x427c0000, 0x4282146f, -2, 65, 469, 476, 2, 0x40e03967),
        (100, 25.0, 30.0, 1, 0x4225dd89, 0x4232359c, -3, 44, 5646, 5766, 0, 0x42f00000),
        (100, 30.0, 30.0, 1, 0x40685f1d, 0x406669c5, -1, 3, 400, 500, 1, 0x42c80000),
        (100, 30.0, 30.0, 1, 0x42080000, 0x422dc51e, 0, 43, 4300, 4400, 3, 0x42c80000),
        (7, 60.0, 59.94, 2, 0x4299cc64, 0x423d306f, 0, 47, 329, 336, 1, 0x40dfc6a8),
        (125, 15.0, 25.0, 1, 0x422c0000, 0x420a049a, -1, 34, 7302, 7510, 0, 0x43505555),
        (40, 60.0, 59.94, 1, 0x4294ea60, 0x42bebd89, -2, 95, 3878, 3918, 0, 0x421fd70a),
        (7, 48.0, 60.0, 1, 0x42500000, 0x42181eb8, -1, 38, 341, 350, 1, 0x410c0000),
        (83, 30.0, 24.0, 1, 0x427f51cf, 0x42cfab35, -1, 103, 6908, 6975, 3, 0x4284cccd),
        (1001, 48.0, 60.0, 1, 0x41900000, 0x419df3b6, -1, 19, 25087, 26338, 3, 0x449c6800),
        (40, 15.0, 25.0, 1, 0x3f800000, 0x3e8c8b43, 0, 0, 3, 69, 0, 0x42855555),
        (100, 60.0, 59.94, 1, 0x41f80000, 0x4224feb8, -2, 41, 4300, 4400, 3, 0x42c7cccd),
        (3003, 23.97599983215332, 23.976023976023978, 2, 0x42640000, 0x408ffff7, 0, 4, 12012, 15015, 1, 0x453bb00c),
        (100, 30.0, 30.0, 2, 0x42b3a000, 0x40600000, 0, 3, 300, 400, 1, 0x42c80000),
        (40, 48.0, 60.0, 1, 0x41300000, 0x41150e55, -2, 9, 552, 602, 2, 0x42480000),
        (40, 29.969999313354492, 29.97002997002997, 1, 0x42980000, 0x42c47ff3, -2, 98, 4000, 4040, 1, 0x4220000a),
        (40, 48.0, 60.0, 1, 0x414c1d71, 0x413e34d3, -2, 11, 652, 702, 3, 0x42480000),
        (83, 24.0, 30.0, 1, 0x41bac46f, 0x41c5a6f3, -1, 24, 2598, 2702, 4, 0x42cf8000),
        (3003, 30.0, 30.0, 1, 0x41b00000, 0x41e2570a, -1, 28, 87087, 90090, 3, 0x453bb000),
        (1001, 24.0, 30.0, 1, 0x422ed242, 0x423a57c4, -3, 46, 61373, 62625, 0, 0x449c6800),
        (40, 23.97599983215332, 23.976023976023978, 1, 0xbff09390, 0x00000000, 0, 0, 0, 40, 0, 0x4220000a),
        (3003, 23.97599983215332, 23.976023976023978, 1, 0x41e00000, 0x4213265c, -2, 36, 114114, 117117, 3, 0x453bb00c),
        (125, 25.0, 30.0, 2, 0x41f80000, 0x42a35555, 0, 81, 12157, 12307, 2, 0x43160000),
    ];

    #[test]
    // The ft_a and ft_b columns were recomputed in float32 with FRAME_TIME_K = 0.05;
    // the src_pos column of the Frame Number row `0x41ffca0f` with K = 0.05 (the sample is 1278.946 ticks, which
    // K = 0.1 took at tick 1279); the other columns are the independent implementation's.
    fn time_map_matches_an_independent_float32_implementation() {
        let (mut speed, mut number, mut based, mut negative, mut marked) = (0, 0, 0, 0, 0);
        for (i, c) in CASES.iter().enumerate() {
            let (lts, input, comp, mode, t_bits, s_bits, base, a, ft_a, ft_b, mk, dur_bits) = *c;
            let rates = Rates::new(input, comp, false, FieldOrder::Progressive, lts).unwrap();
            assert_eq!(rates.dur.to_bits(), dur_bits, "case {i} dur");
            let det = Det { mode };
            let map = RemapTimeMap::new(&det, rates, 0);
            let s = map.src_pos(f32::from_bits(t_bits)).unwrap();
            assert_eq!(s.to_bits(), s_bits, "case {i}: src_pos({}) = {s}", f32::from_bits(t_bits));
            assert_eq!(map.base(), base, "case {i} base");
            assert_eq!(cvtt_i32_f32(s), a, "case {i} a");
            assert_eq!(map.frame_time(a as f32), ft_a, "case {i} frame_time(a)");
            assert_eq!(map.frame_time((a + 1) as f32), ft_b, "case {i} frame_time(a + 1)");
            assert_eq!(map.marker(a as f32).unwrap(), mk, "case {i} marker");
            // The speed cache must not change a single bit either.
            let cached = RemapTimeMap::new(&det, rates, 0).with_speed_cache();
            assert_eq!(cached.src_pos(f32::from_bits(t_bits)).unwrap().to_bits(), s_bits, "case {i} cached");
            speed += (mode == 1) as i32;
            number += (mode == 2) as i32;
            based += (base != 0) as i32;
            negative += (a < 0) as i32;
            marked += (mk != 0) as i32;
        }
        // The table really exercises the branches it is meant to.
        assert_eq!((speed, number), (115, 45));
        assert!(based > 50 && negative > 3 && marked > 50, "{based} {negative} {marked}");
    }

    #[test]
    fn constant_speed_selects_every_frame_exactly() {
        let m = Mock::speed(100.0);
        let map = RemapTimeMap::new(&m, plain_rates(), 0);
        for n in 0..400i32 {
            let (t0, t1) = frame_positions(n * 100, 100, 100);
            let sel = select(&map, t0, t1, 4, 2, 0.0).unwrap();
            assert_eq!((sel.a, sel.b, sel.wa, sel.exact), (n, n + 1, 1.0, true), "frame {n}");
            assert_eq!((sel.s0, sel.s1), (n as f32, n as f32 + 1.0));
            assert_eq!((sel.vectors_quality, sel.interp), (4, 2));
            assert_eq!(map.frame_time(sel.a as f32), n * 100);
            assert_eq!(map.frame_time(sel.b as f32), (n + 1) * 100);
        }
        // Negative output frames map to source frame 0.
        let (t0, t1) = frame_positions(-300, 100, 100);
        let sel = select(&map, t0, t1, 4, 2, 0.0).unwrap();
        assert_eq!((sel.s0, sel.a, sel.wa), (0.0, 0, 1.0));
        assert_eq!(sel.s1, 0.0);
    }

    #[test]
    fn half_speed_alternates_between_frames_and_midpoints() {
        let m = Mock::speed(50.0);
        let map = RemapTimeMap::new(&m, plain_rates(), 0);
        for n in 0..40i32 {
            let (t0, t1) = frame_positions(n * 100, 100, 100);
            let sel = select(&map, t0, t1, 3, 1, 0.0).unwrap();
            assert_eq!(sel.a, n / 2);
            assert_eq!(sel.wa, if n % 2 == 0 { 1.0 } else { 0.5 }, "frame {n}");
            assert_eq!(sel.exact, n % 2 == 0);
            assert_eq!(sel.vectors_quality, 3);
        }
        // With blur compensation even the exact frames take the vector path (s0 != s1).
        let (t0, t1) = frame_positions(400, 100, 100);
        assert!(!select(&map, t0, t1, 3, 1, 1.0).unwrap().exact);
    }
}
