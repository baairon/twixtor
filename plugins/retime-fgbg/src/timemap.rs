//! Plugin 03's time map.
//!
//! Three methods make it up:
//!
//! * `src_pos`: the Speed / Frame Number map of plugin 01 without the `Src Time Offset` read and
//!   without the `base` write, delegated to `RemapTimeMap::without_src_time_offset`.
//! * `frame_time`, the frame to time conversion of every fetch and of the pre-render loop: `T =
//!   trunc((f64)f58 * (f64)(f32)n + 0.5)`. There is no `base` and no unequal rate branch (plugin 01
//!   adds `dur * K` there): plugin 03's `frame_time` is NOT plugin 01's when the two rates differ.
//! * `marker`: `Src: Mark Segs` sampled at the same `T` and mapped 1..=5 to 0..=4. Plugin 01's
//!   marker samples at `trunc((f - base) * dur + 0.5)`, which differs only through the f32 product.
//!
//! `f58 = (f32)(((f64)lts * out_fps) / in_fps)` is `Rates::dur`: the product by the output rate and
//! the division by the input rate are done in f64, in this order, and the result is narrowed to
//! f32.

use fx_core::{FxError, ParamCurves, ParamValue, Result};
use retime_engine::timeline::{cvtt_i32, Rates, RemapTimeMap, TimeMap, TimeParams};

/// The half in `trunc(x + 0.5)`.
const HALF: f64 = 0.5;

/// Plugin 03's [`TimeMap`]. `P` is the parameter reader; a shared reference to a
/// `CurveTimeParams` is `Copy` and implements [`TimeParams`].
pub struct RetimeFgBgTimeMap<P: TimeParams + Copy> {
    params: P,
    inner: RemapTimeMap<P>,
    f58: f32,
}

impl<P: TimeParams + Copy> RetimeFgBgTimeMap<P> {
    /// `current_time` is the time of the output frame, at which `Time Remap Mode` is read.
    pub fn new(params: P, rates: Rates, current_time: i32) -> Self {
        let f58 = rates.dur;
        RetimeFgBgTimeMap { params, inner: RemapTimeMap::new(params, rates, current_time).without_src_time_offset(), f58 }
    }

    /// Opt-in prefix sums of the Speed integral (bit identical, see `RemapTimeMap::with_speed_cache`).
    pub fn with_speed_cache(mut self) -> Self {
        self.inner = self.inner.with_speed_cache();
        self
    }

    /// The duration of one source frame in layer ticks, narrowed to f32.
    pub fn f58(&self) -> f32 {
        self.f58
    }

    /// `trunc((f64)f58 * (f64)n + 0.5)` for an f32 frame position `n`.
    fn time_of(&self, n: f32) -> i32 {
        cvtt_i32((self.f58 as f64) * (n as f64) + HALF)
    }
}

impl<P: TimeParams + Copy> TimeMap for RetimeFgBgTimeMap<P> {
    /// See `RemapTimeMap::src_pos`; no `Src Time Offset`, `base` stays 0.
    fn src_pos(&self, t: f32) -> Result<f32> {
        self.inner.src_pos(t)
    }

    /// The conversion to a layer time: `trunc((f64)f58 * (f64)(f32)n + 0.5)`.
    fn frame_time(&self, f: f32) -> i32 {
        self.time_of(f)
    }

    /// `Src: Mark Segs` at `trunc((f64)f58 * (f64)f + 0.5)`; the popup value 1..=5 gives 0..=4 and
    /// any other value 0.
    fn marker(&self, f: f32) -> Result<i32> {
        let v = self.params.mark_segs(self.time_of(f));
        Ok(if (1..=5).contains(&v) { v - 1 } else { 0 })
    }
}

/// Plugin 03's [`TimeParams`] over `ParamCurves`: the four parameters its time map reads, each
/// sampled at the given layer time (`ParamCurves` takes i64 ticks, the trait i32 ticks). Plugin 03
/// has no `Src Time Offset`, so that reader answers 0 and is never called by [`RetimeFgBgTimeMap`].
/// `Speed %` goes through `ParamCurves::fixed_at` like plugin 01's reader, so a typed or keyframed
/// value is quantised to 16.16.
#[derive(Clone, Copy)]
pub struct CurveParams03<'a> {
    curves: &'a ParamCurves,
}

impl<'a> CurveParams03<'a> {
    /// An error when one of the four ids is missing or has another kind, so the reads cannot panic.
    pub fn new(curves: &'a ParamCurves) -> Result<CurveParams03<'a>> {
        for (id, popup) in [("time_remap_mode", true), ("src_mark_segments", true), ("speed_percent", false), ("frame_num", false)] {
            let ok = match curves.at(id, 0)? {
                ParamValue::Int(_) => true,
                ParamValue::Float(_) => !popup,
                _ => false,
            };
            if !ok {
                return Err(FxError::BadParam {
                    id: id.to_string(),
                    reason: if popup { "the time map needs a popup".into() } else { "the time map needs a slider".into() },
                });
            }
        }
        Ok(CurveParams03 { curves })
    }
}

impl TimeParams for CurveParams03<'_> {
    fn remap_mode(&self, time: i32) -> i32 {
        self.curves.int_at("time_remap_mode", i64::from(time)) as i32
    }

    fn speed_fixed(&self, time: i32) -> i32 {
        self.curves.fixed_at("speed_percent", i64::from(time))
    }

    fn frame_num(&self, time: i32) -> f64 {
        self.curves.float_at("frame_num", i64::from(time))
    }

    /// Plugin 03 has no such parameter; never called by the plugin 03 time map.
    fn src_time_offset_fixed(&self, _time: i32) -> i32 {
        0
    }

    fn mark_segs(&self, time: i32) -> i32 {
        self.curves.int_at("src_mark_segments", i64::from(time)) as i32
    }
}
