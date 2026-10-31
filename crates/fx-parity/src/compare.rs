//! Frame comparison: exact at 8 and 16 bpc, within one ULP at 32 bpc.

use fx_core::{Depth, Frame, Pixels};
use serde_json::{json, Value};
use std::fmt;

/// Largest accepted distance, in representable floats, at 32 bpc, unless a run asks for another.
pub const F32_MAX_ULP: u64 = 1;
/// How many differing pixels a report lists.
pub const MAX_LISTED: usize = 10;

/// Float bits mapped onto integers that sort like the floats. -0.0 and 0.0 share a value.
fn ordered(bits: u32) -> i64 {
    if bits & 0x8000_0000 != 0 {
        -((bits & 0x7fff_ffff) as i64)
    } else {
        bits as i64
    }
}

/// Distance in representable floats, or None when either side is NaN.
pub fn ulp_distance(a: f32, b: f32) -> Option<u64> {
    if a.is_nan() || b.is_nan() {
        return None;
    }
    Some((ordered(a.to_bits()) - ordered(b.to_bits())).unsigned_abs())
}

/// The pass rule for one 32 bpc channel. NaN only matches a NaN with the same bits.
pub fn f32_matches(expected: f32, actual: f32) -> bool {
    f32_matches_within(expected, actual, F32_MAX_ULP)
}

/// The pass rule with `max_ulp` as the accepted distance. 0 means bit for bit: -0.0 against
/// 0.0 and NaNs with other bits differ.
pub fn f32_matches_within(expected: f32, actual: f32, max_ulp: u64) -> bool {
    if max_ulp == 0 {
        return expected.to_bits() == actual.to_bits();
    }
    match ulp_distance(expected, actual) {
        Some(d) => d <= max_ulp,
        None => expected.to_bits() == actual.to_bits(),
    }
}

fn f32_abs_diff(expected: f32, actual: f32) -> f64 {
    if expected == actual || (expected.is_nan() && expected.to_bits() == actual.to_bits()) {
        0.0
    } else if expected.is_nan() || actual.is_nan() {
        f64::INFINITY
    } else {
        (expected as f64 - actual as f64).abs()
    }
}

/// One RGBA pixel in the frame's own channel type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Quad {
    U8([u8; 4]),
    U16([u16; 4]),
    F32([f32; 4]),
}

impl fmt::Display for Quad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Quad::U8(p) => write!(f, "[{}, {}, {}, {}]", p[0], p[1], p[2], p[3]),
            Quad::U16(p) => write!(f, "[{}, {}, {}, {}]", p[0], p[1], p[2], p[3]),
            // Debug keeps tiny and huge floats in exponent form instead of a screenful of digits
            Quad::F32(p) => write!(f, "[{:?}, {:?}, {:?}, {:?}]", p[0], p[1], p[2], p[3]),
        }
    }
}

/// A difference for humans: whole numbers without a fraction, the rest in shortest form.
pub fn diff_text(v: f64) -> String {
    if v.is_finite() && v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v:?}")
    }
}

/// A fraction of full scale as a percentage with three significant digits, for example
/// `0.0123%`, `12.3%` or `5.96e-6%`. Zero prints as `0%`.
pub fn percent_text(fraction: f64) -> String {
    let p = fraction * 100.0;
    if p == 0.0 {
        return "0%".to_string();
    }
    let p = p.abs();
    if !p.is_finite() || !(1e-3..1e6).contains(&p) {
        return format!("{:.2e}%", fraction * 100.0);
    }
    let decimals = (2 - p.log10().floor() as i32).clamp(0, 12) as usize;
    let mut s = format!("{:.*}", decimals, fraction * 100.0);
    if s.contains('.') {
        s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    }
    format!("{s}%")
}

fn f32_json(v: f32) -> Value {
    if v.is_finite() {
        // the shortest f32 text keeps 0.1 as 0.1 instead of 0.10000000149011612
        if let Some(n) = format!("{v}").parse::<f64>().ok().and_then(serde_json::Number::from_f64) {
            return Value::Number(n);
        }
    }
    Value::String(v.to_string())
}

impl Quad {
    pub fn to_json(&self) -> Value {
        match self {
            Quad::U8(p) => json!(p),
            Quad::U16(p) => json!(p),
            Quad::F32(p) => Value::Array(p.iter().map(|v| f32_json(*v)).collect()),
        }
    }

    fn bits_json(&self) -> Option<Value> {
        match self {
            Quad::F32(p) => Some(json!(p.iter().map(|v| format!("0x{:08x}", v.to_bits())).collect::<Vec<_>>())),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PixelDiff {
    pub x: usize,
    pub y: usize,
    pub input: Quad,
    pub expected: Quad,
    pub actual: Quad,
}

impl PixelDiff {
    pub fn to_json(&self) -> Value {
        let mut v = json!({
            "x": self.x,
            "y": self.y,
            "input": self.input.to_json(),
            "expected": self.expected.to_json(),
            "actual": self.actual.to_json(),
        });
        if let (Some(e), Some(a)) = (self.expected.bits_json(), self.actual.bits_json()) {
            v["expected_bits"] = e;
            v["actual_bits"] = a;
        }
        v
    }
}

impl fmt::Display for PixelDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "x={} y={} in={} expected={} actual={}",
            self.x, self.y, self.input, self.expected, self.actual
        )
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Comparison {
    /// Pixels with at least one channel outside the accepted tolerance.
    pub differing_pixels: usize,
    /// Largest per channel absolute difference over the whole frame, tolerated or not.
    pub max_abs_diff: f64,
    /// Mean absolute difference over every pixel and all 4 channels, as a fraction of full
    /// scale: 8 bpc is divided by 255, 16 bpc by 65535, 32 bpc is taken as it is. A channel
    /// pair that has no finite difference (a NaN or an infinity against another value)
    /// counts as 1.0.
    pub mean_abs_diff: f64,
    /// Largest per channel ULP distance at 32 bpc; `u64::MAX` when a NaN did not match.
    pub max_ulp: Option<u64>,
    /// The first [`MAX_LISTED`] differing pixels in raster order.
    pub first: Vec<PixelDiff>,
}

impl Comparison {
    pub fn is_match(&self) -> bool {
        self.differing_pixels == 0
    }
}

struct Channel {
    differs: bool,
    abs: f64,
    ulp: Option<u64>,
    /// What the channel adds to the mean, in the channel's own units: its difference, or
    /// a full scale (1.0) when the difference is not a finite number.
    err: f64,
}

fn scan<T: Copy>(
    width: usize,
    input: &[[T; 4]],
    expected: &[[T; 4]],
    actual: &[[T; 4]],
    full_scale: f64,
    channel: impl Fn(T, T) -> Channel,
    wrap: impl Fn([T; 4]) -> Quad,
) -> Comparison {
    let mut out = Comparison::default();
    let mut total = 0.0f64;
    for (i, (e, a)) in expected.iter().zip(actual).enumerate() {
        let mut differs = false;
        for c in 0..4 {
            let r = channel(e[c], a[c]);
            differs |= r.differs;
            total += r.err;
            out.max_abs_diff = out.max_abs_diff.max(r.abs);
            if let Some(u) = r.ulp {
                out.max_ulp = Some(out.max_ulp.map_or(u, |m| m.max(u)));
            }
        }
        if differs {
            out.differing_pixels += 1;
            if out.first.len() < MAX_LISTED {
                out.first.push(PixelDiff {
                    x: i % width,
                    y: i / width,
                    input: wrap(input[i]),
                    expected: wrap(*e),
                    actual: wrap(*a),
                });
            }
        }
    }
    if !expected.is_empty() {
        out.mean_abs_diff = total / (expected.len() as f64 * 4.0 * full_scale);
    }
    out
}

fn int_channel(e: i64, a: i64) -> Channel {
    let abs = (e - a).unsigned_abs() as f64;
    Channel { differs: e != a, abs, ulp: None, err: abs }
}

/// Compares `actual` against `expected`; `input` is only carried into the report.
pub fn compare(input: &Frame, expected: &Frame, actual: &Frame) -> Result<Comparison, String> {
    compare_within(input, expected, actual, F32_MAX_ULP)
}

/// [`compare`] with `max_ulp` as the accepted distance at 32 bpc; 8 and 16 bpc stay exact.
pub fn compare_within(input: &Frame, expected: &Frame, actual: &Frame, max_ulp: u64) -> Result<Comparison, String> {
    let shape = |f: &Frame| (f.width, f.height, f.depth());
    if shape(input) != shape(expected) || shape(input) != shape(actual) {
        return Err(format!(
            "frames differ in shape: input {:?}, expected {:?}, actual {:?}",
            shape(input),
            shape(expected),
            shape(actual)
        ));
    }
    let w = input.width;
    Ok(match (&input.pixels, &expected.pixels, &actual.pixels) {
        (Pixels::U8(i), Pixels::U8(e), Pixels::U8(a)) => {
            scan(w, i, e, a, 255.0, |e, a| int_channel(e as i64, a as i64), Quad::U8)
        }
        (Pixels::U16(i), Pixels::U16(e), Pixels::U16(a)) => {
            scan(w, i, e, a, 65535.0, |e, a| int_channel(e as i64, a as i64), Quad::U16)
        }
        (Pixels::F32(i), Pixels::F32(e), Pixels::F32(a)) => scan(
            w,
            i,
            e,
            a,
            1.0,
            |e, a| {
                let abs = f32_abs_diff(e, a);
                Channel {
                    differs: !f32_matches_within(e, a, max_ulp),
                    abs,
                    ulp: Some(ulp_distance(e, a).unwrap_or(if e.to_bits() == a.to_bits() { 0 } else { u64::MAX })),
                    err: if abs.is_finite() { abs } else { 1.0 },
                }
            },
            Quad::F32,
        ),
        _ => unreachable!("shapes were checked above"),
    })
}

/// A clip compared frame by frame: exact at 8 and 16 bpc, within 1 ULP at 32 bpc.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClipComparison {
    pub frames: usize,
    /// Frames with at least one pixel outside the accepted tolerance.
    pub failing_frames: usize,
    /// Pixels outside the accepted tolerance, over all frames.
    pub differing_pixels: usize,
    /// The first such frame and what differs in it.
    pub first: Option<(usize, Comparison)>,
    /// Largest per channel absolute difference over the whole clip, tolerated or not.
    pub max_abs_diff: f64,
    /// Mean over all frames of each frame's [`Comparison::mean_abs_diff`]; a frame equal byte
    /// for byte counts 0.
    pub mean_abs_diff: f64,
    /// The frame with the largest mean error (the first one on a tie) and that mean; None
    /// when no frame has any error.
    pub worst_frame: Option<(usize, f64)>,
    /// Largest per channel ULP distance at 32 bpc; `u64::MAX` when a NaN did not match.
    pub max_ulp: Option<u64>,
}

impl ClipComparison {
    pub fn is_match(&self) -> bool {
        self.failing_frames == 0
    }
}

/// Compares `actual` with `expected`, `frames` whole frames of `width`x`height` at `depth`.
/// `input` is the clip fed to the effect; its frame at the same index is carried into the
/// report of each differing pixel.
pub fn compare_clip(
    shape: (usize, usize, Depth),
    frames: usize,
    input: &[u8],
    expected: &[u8],
    actual: &[u8],
) -> Result<ClipComparison, String> {
    compare_clip_within(shape, frames, input, expected, actual, F32_MAX_ULP)
}

/// [`compare_clip`] with `max_ulp` as the accepted distance at 32 bpc.
pub fn compare_clip_within(
    (width, height, depth): (usize, usize, Depth),
    frames: usize,
    input: &[u8],
    expected: &[u8],
    actual: &[u8],
    max_ulp: u64,
) -> Result<ClipComparison, String> {
    let len = width * height * depth.bytes_per_pixel();
    for (what, bytes) in [("input", input), ("expected", expected), ("actual", actual)] {
        if bytes.len() != frames * len {
            return Err(format!("the {what} clip has {} bytes, {frames} frames of {width}x{height} at {} bpc are {}", bytes.len(), depth.bits(), frames * len));
        }
    }
    let mut out = ClipComparison { frames, ..ClipComparison::default() };
    let mut total = 0.0f64;
    for f in 0..frames {
        let (e, a) = (&expected[f * len..(f + 1) * len], &actual[f * len..(f + 1) * len]);
        if e == a {
            continue;
        }
        let frame = |bytes: &[u8]| Frame::from_bytes(width, height, depth, bytes);
        let cmp = compare_within(&frame(&input[f * len..(f + 1) * len]), &frame(e), &frame(a), max_ulp)?;
        out.differing_pixels += cmp.differing_pixels;
        out.max_abs_diff = out.max_abs_diff.max(cmp.max_abs_diff);
        total += cmp.mean_abs_diff;
        if cmp.mean_abs_diff > out.worst_frame.map_or(0.0, |(_, m)| m) {
            out.worst_frame = Some((f, cmp.mean_abs_diff));
        }
        if let Some(u) = cmp.max_ulp {
            out.max_ulp = Some(out.max_ulp.map_or(u, |m| m.max(u)));
        }
        if !cmp.is_match() {
            out.failing_frames += 1;
            if out.first.is_none() {
                out.first = Some((f, cmp));
            }
        }
    }
    if frames > 0 {
        out.mean_abs_diff = total / frames as f64;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fx_core::Depth;

    fn next(v: f32) -> f32 {
        f32::from_bits(v.to_bits() + 1)
    }

    /// A frame holding exactly `pixels`, whatever else Frame carries.
    fn frame_of(w: usize, h: usize, depth: Depth, pixels: Pixels) -> Frame {
        let mut f = Frame::new(w, h, depth);
        f.pixels = pixels;
        f
    }

    fn f32_frame(px: &[[f32; 4]], w: usize) -> Frame {
        frame_of(w, px.len() / w, Depth::F32, Pixels::F32(px.to_vec()))
    }

    #[test]
    fn ulp_boundaries() {
        let x = 1.0f32;
        assert_eq!(ulp_distance(x, x), Some(0));
        assert_eq!(ulp_distance(x, next(x)), Some(1));
        assert_eq!(ulp_distance(x, next(next(x))), Some(2));
        assert!(f32_matches(x, next(x)));
        assert!(f32_matches(next(x), x));
        assert!(!f32_matches(x, next(next(x))));
        // across a power of two the spacing changes but the count of floats does not
        let below = f32::from_bits(2.0f32.to_bits() - 1);
        assert_eq!(ulp_distance(below, 2.0), Some(1));
        assert_eq!(ulp_distance(below, next(2.0)), Some(2));
        // across zero: -min, 0, +min are three consecutive floats
        let tiny = f32::from_bits(1);
        assert_eq!(ulp_distance(-tiny, 0.0), Some(1));
        assert_eq!(ulp_distance(0.0, tiny), Some(1));
        assert_eq!(ulp_distance(-tiny, tiny), Some(2));
        assert!(!f32_matches(-tiny, tiny));
        // negative values order the other way
        let n = -1.0f32;
        assert_eq!(ulp_distance(n, f32::from_bits(n.to_bits() + 1)), Some(1));
    }

    #[test]
    fn ulp_zero_is_bit_for_bit() {
        let tiny = f32::from_bits(1);
        assert!(f32_matches_within(1.0, 1.0, 0));
        assert!(!f32_matches_within(1.0, next(1.0), 0));
        assert!(f32_matches_within(1.0, next(1.0), 1));
        // signed zeros and NaN payloads are different bits
        assert!(!f32_matches_within(0.0, -0.0, 0));
        assert!(!f32_matches_within(-0.0, 0.0, 0));
        assert!(f32_matches_within(0.0, -0.0, 1));
        let nan = f32::from_bits(0x7fc0_0000);
        assert!(f32_matches_within(nan, nan, 0));
        assert!(!f32_matches_within(nan, f32::from_bits(0x7fc0_0001), 0));
        assert!(!f32_matches_within(nan, f32::from_bits(0xffc0_0000), 0));
        assert!(f32_matches_within(0.0, tiny, 1) && !f32_matches_within(0.0, tiny, 0));
        // a wider window accepts what 1 refuses
        assert!(f32_matches_within(1.0, next(next(1.0)), 2));
        assert!(!f32_matches_within(1.0, next(next(1.0)), 1));
    }

    #[test]
    fn frames_and_clips_honour_the_window() {
        let input = f32_frame(&[[0.0; 4]; 2], 2);
        let expected = f32_frame(&[[0.0, 1.0, 1.0, 1.0], [0.5; 4]], 2);
        let mut zero = expected.clone();
        let Pixels::F32(px) = &mut zero.pixels else { unreachable!() };
        px[0][0] = -0.0;
        px[1][2] = next(0.5);
        assert!(compare(&input, &expected, &zero).unwrap().is_match(), "the default window is 1");
        let strict = compare_within(&input, &expected, &zero, 0).unwrap();
        assert_eq!(strict.differing_pixels, 2);
        assert_eq!(strict.max_ulp, Some(1));
        assert_eq!(compare_within(&input, &expected, &zero, 1).unwrap().differing_pixels, 0);
        let bytes = |f: &Frame| -> Vec<u8> {
            let Pixels::F32(px) = &f.pixels else { unreachable!() };
            // planar G B R A
            [1usize, 2, 0, 3].iter().flat_map(|&c| px.iter().flat_map(move |p| p[c].to_le_bytes())).collect()
        };
        let (i, e, a) = (bytes(&input), bytes(&expected), bytes(&zero));
        let loose = compare_clip((2, 1, Depth::F32), 1, &i, &e, &a).unwrap();
        assert!(loose.is_match() && loose.differing_pixels == 0);
        let tight = compare_clip_within((2, 1, Depth::F32), 1, &i, &e, &a, 0).unwrap();
        assert_eq!((tight.failing_frames, tight.differing_pixels), (1, 2));
    }

    #[test]
    fn signed_zero_is_equal() {
        assert_eq!(ulp_distance(0.0, -0.0), Some(0));
        assert!(f32_matches(0.0, -0.0));
        assert!(f32_matches(-0.0, 0.0));
        assert_eq!(f32_abs_diff(0.0, -0.0), 0.0);
    }

    #[test]
    fn nan_matches_only_identical_bits() {
        let a = f32::from_bits(0x7fc0_0000);
        let b = f32::from_bits(0x7fc0_0001);
        let neg = f32::from_bits(0xffc0_0000);
        assert!(f32_matches(a, a));
        assert!(!f32_matches(a, b));
        assert!(!f32_matches(a, neg));
        assert!(!f32_matches(a, 1.0));
        assert!(!f32_matches(1.0, a));
        assert!(!f32_matches(a, f32::INFINITY));
        assert_eq!(ulp_distance(a, a), None);
        assert_eq!(f32_abs_diff(a, a), 0.0);
        assert_eq!(f32_abs_diff(a, b), f64::INFINITY);
    }

    #[test]
    fn infinities_and_extremes() {
        assert!(f32_matches(f32::INFINITY, f32::INFINITY));
        assert!(f32_matches(f32::MAX, f32::INFINITY));
        assert!(!f32_matches(f32::MAX, f32::from_bits(f32::INFINITY.to_bits() + 0x1000)));
        assert!(!f32_matches(f32::INFINITY, f32::NEG_INFINITY));
        assert_eq!(f32_abs_diff(f32::INFINITY, f32::INFINITY), 0.0);
        assert_eq!(f32_abs_diff(f32::INFINITY, 1.0), f64::INFINITY);
    }

    fn u8_frame(px: &[[u8; 4]], w: usize) -> Frame {
        frame_of(w, px.len() / w, Depth::U8, Pixels::U8(px.to_vec()))
    }

    #[test]
    fn identical_frames_match_with_zero_stats() {
        let f = u8_frame(&[[1, 2, 3, 4]; 6], 3);
        let c = compare(&f, &f, &f).unwrap();
        assert!(c.is_match());
        assert_eq!((c.differing_pixels, c.max_abs_diff, c.first.len()), (0, 0.0, 0));
        assert_eq!(c.max_ulp, None);
    }

    #[test]
    fn counts_pixels_not_channels_and_tracks_the_largest_gap() {
        let input = u8_frame(&[[9; 4]; 6], 3);
        let expected = u8_frame(&[[10, 20, 30, 40]; 6], 3);
        let mut actual = expected.clone();
        let Pixels::U8(px) = &mut actual.pixels else { unreachable!() };
        px[1] = [11, 21, 31, 41]; // four channels, one pixel
        px[4] = [10, 20, 30, 90]; // alpha off by 50
        let c = compare(&input, &expected, &actual).unwrap();
        assert_eq!(c.differing_pixels, 2);
        assert_eq!(c.max_abs_diff, 50.0);
        assert_eq!(c.first.len(), 2);
        assert_eq!((c.first[0].x, c.first[0].y), (1, 0));
        assert_eq!((c.first[1].x, c.first[1].y), (1, 1));
        assert_eq!(c.first[1].input, Quad::U8([9; 4]));
        assert_eq!(c.first[1].expected, Quad::U8([10, 20, 30, 40]));
        assert_eq!(c.first[1].actual, Quad::U8([10, 20, 30, 90]));
    }

    #[test]
    fn only_the_first_ten_are_listed_but_all_are_counted() {
        let input = u8_frame(&[[0; 4]; 20], 5);
        let expected = input.clone();
        let mut actual = input.clone();
        let Pixels::U8(px) = &mut actual.pixels else { unreachable!() };
        for p in px.iter_mut().skip(3).take(12) {
            p[0] = 1;
        }
        let c = compare(&input, &expected, &actual).unwrap();
        assert_eq!(c.differing_pixels, 12);
        assert_eq!(c.first.len(), MAX_LISTED);
        assert_eq!((c.first[0].x, c.first[0].y), (3, 0));
        assert_eq!((c.first[9].x, c.first[9].y), (2, 2));
    }

    #[test]
    fn sixteen_bit_is_exact() {
        let mk = |v: u16| frame_of(2, 1, Depth::U16, Pixels::U16(vec![[v; 4], [7; 4]]));
        let (input, a, b) = (mk(0), mk(65534), mk(65535));
        let c = compare(&input, &a, &b).unwrap();
        assert_eq!(c.differing_pixels, 1);
        assert_eq!(c.max_abs_diff, 1.0);
        assert!(compare(&input, &b, &b).unwrap().is_match());
    }

    #[test]
    fn float_tolerance_is_one_ulp_and_reports_the_gap_anyway() {
        let base = [0.25f32, 0.5, 1.0, 1.0];
        let input = f32_frame(&[[0.0; 4]; 3], 3);
        let expected = f32_frame(&[base; 3], 3);
        let mut actual = expected.clone();
        let Pixels::F32(px) = &mut actual.pixels else { unreachable!() };
        px[0][1] = next(0.5); // 1 ulp: tolerated
        px[1][2] = next(next(1.0)); // 2 ulp: differs
        px[2][0] = -0.0 + 0.25; // identical
        let c = compare(&input, &expected, &actual).unwrap();
        assert_eq!(c.differing_pixels, 1);
        assert_eq!(c.max_ulp, Some(2));
        assert_eq!((c.first[0].x, c.first[0].y), (1, 0));
        assert!(c.max_abs_diff > 0.0 && c.max_abs_diff < 1e-6);
    }

    #[test]
    fn float_nan_and_signed_zero_pixels() {
        let nan = f32::from_bits(0x7fc0_0000);
        let input = f32_frame(&[[0.0; 4]; 2], 2);
        let expected = f32_frame(&[[nan, 0.0, 0.0, 1.0], [0.0, -0.0, 0.0, 1.0]], 2);
        let same = f32_frame(&[[nan, 0.0, 0.0, 1.0], [-0.0, 0.0, 0.0, 1.0]], 2);
        assert!(compare(&input, &expected, &same).unwrap().is_match());
        let other_nan = f32_frame(&[[f32::from_bits(0x7fc0_0001), 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], 2);
        let c = compare(&input, &expected, &other_nan).unwrap();
        assert_eq!(c.differing_pixels, 1);
        assert_eq!(c.max_ulp, Some(u64::MAX));
        assert_eq!(c.max_abs_diff, f64::INFINITY);
    }

    #[test]
    fn display_keeps_floats_short() {
        let q = Quad::F32([0.5, 1.0, f32::from_bits(2), f32::NAN]);
        assert_eq!(q.to_string(), "[0.5, 1.0, 3e-45, NaN]");
        assert_eq!(Quad::U16([1, 2, 3, 65535]).to_string(), "[1, 2, 3, 65535]");
        assert_eq!(diff_text(300.0), "300");
        assert_eq!(diff_text(2.9802322387695313e-8), "2.9802322387695313e-8");
        assert_eq!(diff_text(f64::INFINITY), "inf");
    }

    fn u8_clip(frames: &[[[u8; 4]; 2]]) -> Vec<u8> {
        frames.iter().flat_map(|f| f.iter().flatten().copied()).collect()
    }

    #[test]
    fn clips_are_compared_frame_by_frame_and_the_first_failing_frame_is_reported() {
        let input = u8_clip(&[[[1; 4]; 2]; 4]);
        let expected = u8_clip(&[[[10, 20, 30, 40]; 2]; 4]);
        let mut actual = expected.clone();
        // frame 1 untouched; frame 2: pixel 1, two channels; frame 3: pixel 0
        actual[2 * 8 + 4] = 11;
        actual[2 * 8 + 6] = 31;
        actual[3 * 8 + 3] = 99;
        let c = compare_clip((2, 1, Depth::U8), 4, &input, &expected, &actual).unwrap();
        assert_eq!((c.frames, c.failing_frames), (4, 2));
        let (frame, first) = c.first.as_ref().unwrap();
        assert_eq!(*frame, 2);
        assert_eq!(first.differing_pixels, 1);
        assert_eq!((first.first[0].x, first.first[0].y), (1, 0));
        assert_eq!(first.first[0].input, Quad::U8([1; 4]));
        assert_eq!(first.first[0].expected, Quad::U8([10, 20, 30, 40]));
        assert_eq!(first.first[0].actual, Quad::U8([11, 20, 31, 40]));
        assert_eq!(c.max_abs_diff, 59.0, "the largest gap anywhere, here in frame 3");
        assert!(!c.is_match());
        // identical clips match with nothing to report
        let same = compare_clip((2, 1, Depth::U8), 4, &input, &expected, &expected).unwrap();
        assert!(same.is_match() && same.first.is_none() && same.max_abs_diff == 0.0 && same.max_ulp.is_none());
    }

    #[test]
    fn clips_use_the_tolerance_of_their_depth() {
        let to_bytes = |vals: &[f32]| -> Vec<u8> {
            // planar G B R A for two pixels: 8 floats per frame
            vals.iter().flat_map(|v| v.to_le_bytes()).collect()
        };
        let base: Vec<f32> = (0..16).map(|i| 0.25 + i as f32 * 0.01).collect();
        let input = to_bytes(&base);
        let mut one_ulp = base.clone();
        one_ulp[3] = next(one_ulp[3]);
        let c = compare_clip((2, 1, Depth::F32), 2, &input, &input, &to_bytes(&one_ulp)).unwrap();
        assert!(c.is_match(), "one ULP is tolerated");
        assert_eq!(c.max_ulp, Some(1));
        assert!(c.max_abs_diff > 0.0);
        let mut two_ulp = base.clone();
        two_ulp[9] = next(next(two_ulp[9]));
        let c = compare_clip((2, 1, Depth::F32), 2, &input, &input, &to_bytes(&two_ulp)).unwrap();
        assert_eq!((c.failing_frames, c.first.as_ref().unwrap().0), (1, 1));
        // 16 bpc is exact
        let a: Vec<u8> = [1u16, 2, 3, 4].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut b = a.clone();
        b[0] = 2;
        assert_eq!(compare_clip((1, 1, Depth::U16), 1, &a, &a, &b).unwrap().failing_frames, 1);
    }

    #[test]
    fn clips_of_the_wrong_length_are_errors() {
        let ok = vec![0u8; 16];
        assert!(compare_clip((2, 1, Depth::U8), 2, &ok, &ok, &ok).is_ok());
        for bad in [&ok[..15], &[0u8; 17][..]] {
            let err = compare_clip((2, 1, Depth::U8), 2, &ok, &ok, bad).unwrap_err();
            assert!(err.contains("actual clip") && err.contains("2 frames of 2x1 at 8 bpc are 16"), "{err}");
        }
        assert!(compare_clip((2, 1, Depth::U8), 2, &ok[..8], &ok, &ok).unwrap_err().contains("input clip"));
    }

    #[test]
    fn shape_mismatch_is_an_error_not_a_panic() {
        let a = Frame::new(2, 2, Depth::U8);
        let b = Frame::new(2, 2, Depth::U16);
        let c = Frame::new(4, 1, Depth::U8);
        assert!(compare(&a, &a, &b).is_err());
        assert!(compare(&a, &a, &c).is_err());
    }

    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12
    }

    #[test]
    fn mean_error_is_a_fraction_of_full_scale_over_every_channel_of_every_pixel() {
        // 8 bpc: two pixels, 8 channels; one channel is off by 51 levels, a fifth of full scale
        let input = u8_frame(&[[0; 4]; 2], 2);
        let expected = u8_frame(&[[10, 20, 30, 40]; 2], 2);
        let mut actual = expected.clone();
        let Pixels::U8(px) = &mut actual.pixels else { unreachable!() };
        px[1][2] = 30 + 51;
        let c = compare(&input, &expected, &actual).unwrap();
        assert!(near(c.mean_abs_diff, 51.0 / (8.0 * 255.0)), "{}", c.mean_abs_diff);
        assert!(near(c.mean_abs_diff, 0.025));
        // the same single channel error at 16 bpc
        let mk = |v: u16| frame_of(2, 1, Depth::U16, Pixels::U16(vec![[v; 4], [7; 4]]));
        let mut off = mk(1000);
        let Pixels::U16(px) = &mut off.pixels else { unreachable!() };
        px[0][3] = 1000 + 13107;
        let c = compare(&mk(0), &mk(1000), &off).unwrap();
        assert!(near(c.mean_abs_diff, 13107.0 / (8.0 * 65535.0)), "{}", c.mean_abs_diff);
        // at 32 bpc the difference is already a fraction
        let base = f32_frame(&[[0.25; 4]; 2], 2);
        let mut moved = base.clone();
        let Pixels::F32(px) = &mut moved.pixels else { unreachable!() };
        px[0][0] = 0.25 + 0.5;
        px[1][3] = 0.25 - 0.25;
        let c = compare(&base, &base, &moved).unwrap();
        assert!(near(c.mean_abs_diff, 0.75 / 8.0), "{}", c.mean_abs_diff);
    }

    #[test]
    fn equal_frames_have_exactly_zero_mean_error_and_a_tolerated_gap_is_small_but_not_zero() {
        let f = u8_frame(&[[1, 2, 3, 4]; 6], 3);
        assert_eq!(compare(&f, &f, &f).unwrap().mean_abs_diff, 0.0);
        // signed zeros are different bytes but no error
        let a = f32_frame(&[[0.0, 1.0, 1.0, 1.0]], 1);
        let b = f32_frame(&[[-0.0, 1.0, 1.0, 1.0]], 1);
        let c = compare(&a, &a, &b).unwrap();
        assert!(c.is_match() && c.mean_abs_diff == 0.0);
        // one ULP at 1.0 passes, and shows up as a tiny positive mean
        let x = f32_frame(&[[1.0; 4]], 1);
        let mut y = x.clone();
        let Pixels::F32(px) = &mut y.pixels else { unreachable!() };
        px[0][0] = next(1.0);
        let c = compare(&x, &x, &y).unwrap();
        assert!(c.is_match());
        assert!(c.mean_abs_diff > 0.0 && c.mean_abs_diff < 1e-7, "{}", c.mean_abs_diff);
    }

    #[test]
    fn a_pair_without_a_finite_difference_counts_a_full_scale() {
        let nan = f32::from_bits(0x7fc0_0000);
        let input = f32_frame(&[[0.0; 4]; 2], 2);
        // 8 channels: NaN against a number, infinity against a number, the same NaN twice
        let expected = f32_frame(&[[nan, f32::INFINITY, nan, 0.5], [0.0; 4]], 2);
        let actual = f32_frame(&[[0.5, 1.0, nan, 0.5], [0.0; 4]], 2);
        let c = compare(&input, &expected, &actual).unwrap();
        assert!(near(c.mean_abs_diff, 2.0 / 8.0), "{}", c.mean_abs_diff);
        assert!(c.mean_abs_diff.is_finite());
        // a NaN that differs from another NaN in its bits is a miss too
        let other = f32_frame(&[[f32::from_bits(0x7fc0_0001), 0.0, 0.0, 0.0], [0.0; 4]], 2);
        let only_nan = f32_frame(&[[nan, 0.0, 0.0, 0.0], [0.0; 4]], 2);
        assert!(near(compare(&input, &only_nan, &other).unwrap().mean_abs_diff, 1.0 / 8.0));
    }

    #[test]
    fn a_clip_mean_is_the_mean_of_its_frame_means_with_equal_frames_at_zero() {
        let input = u8_clip(&[[[1; 4]; 2]; 4]);
        let expected = u8_clip(&[[[10, 20, 30, 40]; 2]; 4]);
        let mut actual = expected.clone();
        // frame 1: one channel off by 200 levels; frame 3: one channel off by 51
        actual[8] = 10 + 200;
        actual[3 * 8 + 5] = 20 + 51;
        let c = compare_clip((2, 1, Depth::U8), 4, &input, &expected, &actual).unwrap();
        assert!(near(c.mean_abs_diff, (200.0 + 51.0) / (4.0 * 8.0 * 255.0)), "{}", c.mean_abs_diff);
        let (frame, mean) = c.worst_frame.unwrap();
        assert_eq!(frame, 1);
        assert!(near(mean, 200.0 / (8.0 * 255.0)), "{mean}");
        // the first failing frame is not necessarily the worst one, and its own stats stay per frame
        assert_eq!(c.first.as_ref().unwrap().0, 1);
        let same = compare_clip((2, 1, Depth::U8), 4, &input, &expected, &expected).unwrap();
        assert_eq!((same.mean_abs_diff, same.worst_frame), (0.0, None));
        // a tie is won by the earlier frame
        let mut tied = expected.clone();
        tied[2 * 8] ^= 1;
        tied[3 * 8] ^= 1;
        let c = compare_clip((2, 1, Depth::U8), 4, &input, &expected, &tied).unwrap();
        assert_eq!(c.worst_frame.map(|(f, _)| f), Some(2));
    }

    #[test]
    fn a_clip_whose_only_difference_is_a_signed_zero_has_no_worst_frame() {
        let to_bytes = |vals: &[f32]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
        let a = to_bytes(&[0.0, 1.0, 1.0, 1.0]);
        let b = to_bytes(&[-0.0, 1.0, 1.0, 1.0]);
        let c = compare_clip((1, 1, Depth::F32), 1, &a, &a, &b).unwrap();
        assert!(c.is_match());
        assert_eq!((c.mean_abs_diff, c.worst_frame), (0.0, None));
    }

    #[test]
    fn percentages_keep_three_significant_digits() {
        assert_eq!(percent_text(0.0), "0%");
        assert_eq!(percent_text(0.000123), "0.0123%");
        assert_eq!(percent_text(0.5), "50%");
        assert_eq!(percent_text(0.123456), "12.3%");
        assert_eq!(percent_text(1.0), "100%");
        assert_eq!(percent_text(0.025), "2.5%");
        assert_eq!(percent_text(0.00002), "0.002%");
        assert_eq!(percent_text(5.96e-8), "5.96e-6%");
        assert_eq!(percent_text(2.0e5), "2.00e7%");
        assert_eq!(percent_text(f64::INFINITY), "inf%");
        assert!(percent_text(0.000123).is_ascii());
    }

    #[test]
    fn json_keeps_short_float_text_and_marks_non_finite() {
        let q = Quad::F32([0.1, f32::NAN, f32::INFINITY, -0.0]);
        let v = q.to_json();
        assert_eq!(v[0].to_string(), "0.1");
        assert_eq!(v[1], "NaN");
        assert_eq!(v[2], "inf");
        let d = PixelDiff { x: 1, y: 2, input: q, expected: q, actual: q };
        let j = d.to_json();
        assert_eq!(j["expected_bits"][0], "0x3dcccccd");
        assert!(PixelDiff { input: Quad::U8([0; 4]), expected: Quad::U8([0; 4]), actual: Quad::U8([1; 4]), ..d }
            .to_json()
            .get("expected_bits")
            .is_none());
    }
}
