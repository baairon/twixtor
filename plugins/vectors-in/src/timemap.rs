//! Plugin 04's time map.
//!
//! Three pieces make it up, the same operations as plugin 03's:
//!
//! * `src_pos`: the Speed / Frame Number map of plugin 01 without the `Src Time Offset` read and
//!   without the `base` write: Frame Number samples `Frame Num` at `trunc((f64)((f32)lts * x) + K)`
//!   and returns `(f32)((in_fps / out_fps) * (f64)v)`; Speed integrates the speed curve in half
//!   frame steps sampled at `trunc((f64)((f32)lts * f / 2) + K)` (loop test `f <= (f32)(i32)(x *
//!   2)`, tail step when the remainder exceeds 0.0001, final factor `in_fps / out_fps`). Delegated
//!   to `RemapTimeMap::without_src_time_offset`; `K` is the sampling bias of
//!   `retime_engine::timeline`.
//! * the frame to time conversion `T(k) = trunc((f64)dur * (f64)(f32)k + 0.5)` with `dur` =
//!   `Rates::dur`, used for every fetch and in the pre-render plan loops. No `base`, no unequal
//!   rate branch with `K`: not plugin 01's.
//! * `marker`: `Src: Mark Segments` sampled at `T(k)`, popup 1..=5 mapped to 0..=4.

use fx_core::{FxError, ParamCurves, ParamValue, Result};
use retime_engine::timeline::{cvtt_i32, Rates, RemapTimeMap, TimeMap, TimeParams};

/// The half in `trunc(x + 0.5)`.
const HALF: f64 = 0.5;

/// Plugin 04's [`TimeMap`]. `P` is the parameter reader; a shared reference to a `CurveParams04` is `Copy` and
/// implements [`TimeParams`].
pub struct VectorsInTimeMap<P: TimeParams + Copy> {
    params: P,
    inner: RemapTimeMap<P>,
    dur: f32,
}

impl<P: TimeParams + Copy> VectorsInTimeMap<P> {
    /// `current_time` is the time of the output frame, at which `Time Remap Mode` is read.
    pub fn new(params: P, rates: Rates, current_time: i32) -> Self {
        let dur = rates.dur;
        VectorsInTimeMap { params, inner: RemapTimeMap::new(params, rates, current_time).without_src_time_offset(), dur }
    }

    /// Opt-in prefix sums of the Speed integral (bit identical, see `RemapTimeMap::with_speed_cache`).
    pub fn with_speed_cache(mut self) -> Self {
        self.inner = self.inner.with_speed_cache();
        self
    }

    /// The duration of one source frame in layer ticks, narrowed to f32.
    pub fn dur(&self) -> f32 {
        self.dur
    }

    /// `trunc((f64)dur * (f64)n + 0.5)` for an f32 frame position `n`.
    pub fn time_of(&self, n: f32) -> i32 {
        cvtt_i32((self.dur as f64) * (n as f64) + HALF)
    }
}

impl<P: TimeParams + Copy> TimeMap for VectorsInTimeMap<P> {
    /// See `RemapTimeMap::src_pos`; no `Src Time Offset`, `base` stays 0.
    fn src_pos(&self, t: f32) -> Result<f32> {
        self.inner.src_pos(t)
    }

    /// The conversion `T(k)` of every fetch slot: `trunc((f64)dur * (f64)(f32)n + 0.5)`.
    fn frame_time(&self, f: f32) -> i32 {
        self.time_of(f)
    }

    /// `Src: Mark Segments` at `T(f)`; the popup value 1..=5 gives 0..=4 and any other value 0.
    fn marker(&self, f: f32) -> Result<i32> {
        let v = self.params.mark_segs(self.time_of(f));
        Ok(if (1..=5).contains(&v) { v - 1 } else { 0 })
    }
}

/// Plugin 04's [`TimeParams`] over `ParamCurves`: the four parameters its time map reads, each
/// sampled at the given layer time (`ParamCurves` takes i64 ticks, the trait i32 ticks). Plugin 04
/// has no `Src Time Offset`, so that reader answers 0 and is never called by [`VectorsInTimeMap`].
/// `Speed %` goes through `ParamCurves::fixed_at`, so a typed or keyframed value is quantised to
/// 16.16.
#[derive(Clone, Copy)]
pub struct CurveParams04<'a> {
    curves: &'a ParamCurves,
}

impl<'a> CurveParams04<'a> {
    /// An error when one of the four ids is missing or has another kind, so the reads cannot panic.
    pub fn new(curves: &'a ParamCurves) -> Result<CurveParams04<'a>> {
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
        Ok(CurveParams04 { curves })
    }
}

impl TimeParams for CurveParams04<'_> {
    fn remap_mode(&self, time: i32) -> i32 {
        self.curves.int_at("time_remap_mode", i64::from(time)) as i32
    }

    fn speed_fixed(&self, time: i32) -> i32 {
        self.curves.fixed_at("speed_percent", i64::from(time))
    }

    fn frame_num(&self, time: i32) -> f64 {
        self.curves.float_at("frame_num", i64::from(time))
    }

    /// Plugin 04 has no such parameter; never called by the plugin 04 time map.
    fn src_time_offset_fixed(&self, _time: i32) -> i32 {
        0
    }

    fn mark_segs(&self, time: i32) -> i32 {
        self.curves.int_at("src_mark_segments", i64::from(time)) as i32
    }
}
