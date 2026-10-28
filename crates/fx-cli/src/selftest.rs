//! Hidden test effects. They exist to exercise the plumbing of `vfx` and of the
//! parity harness (temporal context, random access to the clip, extra layers,
//! parameter curves, thread count invariance), not to do anything useful.
//! `vfx list` hides them unless `--all` is passed; `vfx apply` runs them.

use fx_core::gaps::{record, OpenGap};
use fx_core::{
    par_rows, Checkout, Effect, EffectKind, Env, Frame, FxError, LayerSpec, ParamKind, ParamSpec, Params, Pixels,
    RenderContext, Result, TemporalContext,
};
use rayon::prelude::*;

pub fn effects() -> Vec<Box<dyn Effect>> {
    vec![Box::new(Invert), Box::new(Shift), Box::new(Integrate), Box::new(Threads), Box::new(Gap)]
}

/// The temporal effects have no single frame form.
fn not_a_frame_effect(id: &str) -> Result<()> {
    Err(FxError::Unsupported(format!("{id} is a temporal effect, it needs --fps")))
}

/// Replaces `out` by `src`, or by transparent black when the clip has no frame there.
fn load(out: &mut Frame, src: Option<&Frame>) -> Result<()> {
    match src {
        Some(f) => {
            if (f.width, f.height, f.depth()) != (out.width, out.height, out.depth()) {
                return Err(FxError::Mismatch(format!(
                    "a {}x{} {:?} source frame for a {}x{} {:?} output",
                    f.width,
                    f.height,
                    f.depth(),
                    out.width,
                    out.height,
                    out.depth()
                )));
            }
            out.pixels = f.pixels.clone();
        }
        None => *out = Frame::new(out.width, out.height, out.depth()),
    }
    Ok(())
}

// ---- selftest-invert: the frame effect -----------------------------------------------------

/// Inverts the colour channels of one frame (`max - v`), alpha too when `invert_alpha` is on.
pub struct Invert;

impl Effect for Invert {
    fn id(&self) -> &'static str {
        "selftest-invert"
    }

    fn name(&self) -> &'static str {
        "Selftest Invert"
    }

    fn params(&self) -> Vec<ParamSpec> {
        vec![ParamSpec { id: "invert_alpha", name: "Invert Alpha", kind: ParamKind::Checkbox { default: false } }]
    }

    fn render(&self, _ctx: &RenderContext, params: &Params, input: &Frame, output: &mut Frame) -> Result<()> {
        let channels = if params.bool("invert_alpha") { 4 } else { 3 };
        load(output, Some(input))?;
        let w = output.width;
        match &mut output.pixels {
            Pixels::U8(px) => par_rows(px, w, |_, row| row.iter_mut().for_each(|p| p[..channels].iter_mut().for_each(|c| *c = 255 - *c))),
            Pixels::U16(px) => par_rows(px, w, |_, row| row.iter_mut().for_each(|p| p[..channels].iter_mut().for_each(|c| *c = 65535 - *c))),
            Pixels::F32(px) => par_rows(px, w, |_, row| row.iter_mut().for_each(|p| p[..channels].iter_mut().for_each(|c| *c = 1.0 - *c))),
        }
        Ok(())
    }
}

// ---- selftest-shift: random access to the clip and an extra layer ---------------------------

/// Output frame `n` is input frame `n + offset`, transparent black outside the clip. The
/// optional layer `overlay`, when given, is mixed in at 50 percent: every channel becomes
/// `(result + overlay + 1) / 2` in integers (`(result + overlay) * 0.5` in float), with the
/// overlay frame at the same frame `n` and transparent black where the overlay has none.
pub struct Shift;

impl Shift {
    /// Layer time of the input frame a render of `cx` reads.
    fn source_time(cx: &TemporalContext, env: &Env<'_>) -> i64 {
        cx.current_time + env.params.int_at("offset", cx.current_time) * cx.time_step()
    }
}

impl Effect for Shift {
    fn id(&self) -> &'static str {
        "selftest-shift"
    }

    fn name(&self) -> &'static str {
        "Selftest Shift"
    }

    fn params(&self) -> Vec<ParamSpec> {
        vec![ParamSpec { id: "offset", name: "Offset", kind: ParamKind::Int { min: -8, max: 8, default: 0 } }]
    }

    fn kind(&self) -> EffectKind {
        EffectKind::Temporal
    }

    fn layers(&self) -> Vec<LayerSpec> {
        vec![LayerSpec { id: "overlay", name: "Overlay", optional: true }]
    }

    fn render(&self, _ctx: &RenderContext, _params: &Params, _input: &Frame, _output: &mut Frame) -> Result<()> {
        not_a_frame_effect(self.id())
    }

    fn dependencies(&self, cx: &TemporalContext, env: &Env<'_>) -> Result<Vec<Checkout>> {
        let mut fetches = vec![Checkout { layer: None, time: Shift::source_time(cx, env) }];
        if env.layer("overlay").is_some() {
            fetches.push(Checkout { layer: Some("overlay"), time: cx.current_time });
        }
        Ok(fetches)
    }

    fn render_temporal(&self, cx: &TemporalContext, env: &Env<'_>, output: &mut Frame) -> Result<()> {
        let src = env.input.frame_at(Shift::source_time(cx, env))?;
        load(output, src.as_deref())?;
        if let Some(layer) = env.layer("overlay") {
            let over = layer.frame_at(cx.current_time)?;
            let zero;
            let over = match over.as_deref() {
                Some(f) => f,
                None => {
                    zero = Frame::new(output.width, output.height, output.depth());
                    &zero
                }
            };
            if (over.width, over.height, over.depth()) != (output.width, output.height, output.depth()) {
                return Err(FxError::Mismatch("the overlay layer has another size or depth than the output".into()));
            }
            mix_half(output, over);
        }
        Ok(())
    }
}

fn mix_half(out: &mut Frame, over: &Frame) {
    let w = out.width;
    match (&mut out.pixels, &over.pixels) {
        (Pixels::U8(a), Pixels::U8(b)) => par_rows(a, w, |y, row| {
            for (x, p) in row.iter_mut().enumerate() {
                let q = b[y * w + x];
                for c in 0..4 {
                    p[c] = ((p[c] as u16 + q[c] as u16 + 1) >> 1) as u8;
                }
            }
        }),
        (Pixels::U16(a), Pixels::U16(b)) => par_rows(a, w, |y, row| {
            for (x, p) in row.iter_mut().enumerate() {
                let q = b[y * w + x];
                for c in 0..4 {
                    p[c] = ((p[c] as u32 + q[c] as u32 + 1) >> 1) as u16;
                }
            }
        }),
        (Pixels::F32(a), Pixels::F32(b)) => par_rows(a, w, |y, row| {
            for (x, p) in row.iter_mut().enumerate() {
                let q = b[y * w + x];
                for c in 0..4 {
                    p[c] = (p[c] + q[c]) * 0.5;
                }
            }
        }),
        _ => unreachable!("the depths were checked by the caller"),
    }
}

// ---- selftest-integrate: parameter curves sampled at every earlier frame --------------------

/// Output frame `n` is input frame `n` with every channel (alpha included) raised by
/// `S / 255` of the full range, where `S` is the sum of `gain` sampled at the start of each
/// frame `0 .. n` (frame `n` itself is not in the sum, so frame 0 is untouched), added in
/// that order in f64. In levels: 8 bpc `round(v + S)`, 16 bpc `round(v + S * 257)`, float
/// `v + S / 255`; integers clamp to their range and float to 0..1.
pub struct Integrate;

impl Effect for Integrate {
    fn id(&self) -> &'static str {
        "selftest-integrate"
    }

    fn name(&self) -> &'static str {
        "Selftest Integrate"
    }

    fn params(&self) -> Vec<ParamSpec> {
        vec![ParamSpec {
            id: "gain",
            name: "Gain",
            kind: ParamKind::Float {
                valid_min: -255.0,
                valid_max: 255.0,
                slider_min: -16.0,
                slider_max: 16.0,
                default: 0.0,
                precision: 2,
            },
        }]
    }

    fn kind(&self) -> EffectKind {
        EffectKind::Temporal
    }

    fn render(&self, _ctx: &RenderContext, _params: &Params, _input: &Frame, _output: &mut Frame) -> Result<()> {
        not_a_frame_effect(self.id())
    }

    fn dependencies(&self, cx: &TemporalContext, _env: &Env<'_>) -> Result<Vec<Checkout>> {
        Ok(vec![Checkout { layer: None, time: cx.current_time }])
    }

    fn render_temporal(&self, cx: &TemporalContext, env: &Env<'_>, output: &mut Frame) -> Result<()> {
        let step = cx.time_step();
        let mut sum = 0.0f64;
        for k in 0..cx.frame_index {
            sum += env.params.float_at("gain", k as i64 * step);
        }
        let src = env.input.frame_at(cx.current_time)?;
        load(output, src.as_deref())?;
        let w = output.width;
        match &mut output.pixels {
            Pixels::U8(px) => par_rows(px, w, |_, row| {
                row.iter_mut().for_each(|p| p.iter_mut().for_each(|c| *c = (*c as f64 + sum).round().clamp(0.0, 255.0) as u8))
            }),
            Pixels::U16(px) => par_rows(px, w, |_, row| {
                row.iter_mut().for_each(|p| p.iter_mut().for_each(|c| *c = (*c as f64 + sum * 257.0).round().clamp(0.0, 65535.0) as u16))
            }),
            Pixels::F32(px) => par_rows(px, w, |_, row| {
                row.iter_mut().for_each(|p| p.iter_mut().for_each(|c| *c = ((*c as f64 + sum / 255.0) as f32).clamp(0.0, 1.0)))
            }),
        }
        Ok(())
    }
}

// ---- selftest-threads: a deliberately thread dependent effect ----------------------------------

/// Output frame `n` is input frame `n` with the red channel of its first pixel replaced by the
/// number of rayon threads (8 bpc `min(n, 255)`, 16 bpc `n * 257` capped at 65535, float `n / 255`).
/// The engine must never do this; the effect exists so that `fx-parity selfcheck` can be seen
/// firing on a real process when `RAYON_NUM_THREADS` changes.
pub struct Threads;

impl Effect for Threads {
    fn id(&self) -> &'static str {
        "selftest-threads"
    }

    fn name(&self) -> &'static str {
        "Selftest Threads"
    }

    fn params(&self) -> Vec<ParamSpec> {
        Vec::new()
    }

    fn kind(&self) -> EffectKind {
        EffectKind::Temporal
    }

    fn render(&self, _ctx: &RenderContext, _params: &Params, _input: &Frame, _output: &mut Frame) -> Result<()> {
        not_a_frame_effect(self.id())
    }

    fn render_temporal(&self, cx: &TemporalContext, env: &Env<'_>, output: &mut Frame) -> Result<()> {
        let src = env.input.frame_at(cx.current_time)?;
        load(output, src.as_deref())?;
        let threads = rayon::current_num_threads();
        match &mut output.pixels {
            Pixels::U8(px) => px[0][0] = threads.min(255) as u8,
            Pixels::U16(px) => px[0][0] = (threads * 257).min(65535) as u16,
            Pixels::F32(px) => px[0][0] = threads as f32 / 255.0,
        }
        Ok(())
    }
}

// ---- selftest-gap: the note of an open parity gap ------------------------------------------------

/// The test gap of `selftest-gap`. It stands for no rule of the original: it exists so that the
/// path from `fx_core::gaps::record` to the `vfx: note:` line can be run on a real process.
pub static T1: OpenGap = OpenGap::new("T1", "notes plumbing test", &["gap"]);
static GAP_GAPS: &[&OpenGap] = &[&T1];

/// Output frame `n` is input frame `n`, transparent black past the clip. With the checkbox `gap`
/// on at the frame's time, the frame records the test gap T1 once from the main thread and once
/// per row from the rayon thread that visits it, so `vfx` has to fold many records from many
/// threads into its one note per run. The pixels are the same with `gap` on or off.
pub struct Gap;

impl Effect for Gap {
    fn id(&self) -> &'static str {
        "selftest-gap"
    }

    fn name(&self) -> &'static str {
        "Selftest Gap"
    }

    fn params(&self) -> Vec<ParamSpec> {
        vec![ParamSpec { id: "gap", name: "Gap", kind: ParamKind::Checkbox { default: false } }]
    }

    fn kind(&self) -> EffectKind {
        EffectKind::Temporal
    }

    fn gaps(&self) -> &'static [&'static OpenGap] {
        GAP_GAPS
    }

    fn render(&self, _ctx: &RenderContext, _params: &Params, _input: &Frame, _output: &mut Frame) -> Result<()> {
        not_a_frame_effect(self.id())
    }

    fn dependencies(&self, cx: &TemporalContext, _env: &Env<'_>) -> Result<Vec<Checkout>> {
        Ok(vec![Checkout { layer: None, time: cx.current_time }])
    }

    fn render_temporal(&self, cx: &TemporalContext, env: &Env<'_>, output: &mut Frame) -> Result<()> {
        let src = env.input.frame_at(cx.current_time)?;
        load(output, src.as_deref())?;
        if env.params.bool_at("gap", cx.current_time) {
            record(&T1);
            (0..output.height).into_par_iter().for_each(|_| record(&T1));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fx_core::{Depth, Layer, LayerSet, MemLayer, ParamCurves, ParamValue, Rational, TimeBase};
    use std::sync::Arc;

    fn base() -> TimeBase {
        TimeBase::default_for_rate(Rational::new(24, 1).unwrap())
    }

    /// A 2x2 frame whose first pixel is `[v, v + 1, v + 2, 255]`, the rest `[v; 4]`.
    fn frame8(v: u8) -> Frame {
        let mut f = Frame::new(2, 2, Depth::U8);
        if let Pixels::U8(px) = &mut f.pixels {
            px.iter_mut().for_each(|p| *p = [v; 4]);
            px[0] = [v, v + 1, v + 2, 255];
        }
        f
    }

    fn first8(f: &Frame) -> [u8; 4] {
        match &f.pixels {
            Pixels::U8(px) => px[0],
            _ => unreachable!(),
        }
    }

    fn clip8(values: &[u8]) -> MemLayer {
        MemLayer::new(base(), values.iter().map(|v| frame8(*v)).collect()).unwrap()
    }

    fn render(effect: &dyn Effect, curves: &ParamCurves, input: &MemLayer, layers: &LayerSet, n: u64, count: u64) -> Frame {
        let cx = TemporalContext::new(2, 2, input.depth(), base(), n, count);
        let env = Env { params: curves, input, layers };
        let mut out = Frame::new(2, 2, input.depth());
        effect.render_temporal(&cx, &env, &mut out).unwrap();
        out
    }

    #[test]
    fn the_registry_sees_one_frame_and_four_temporal_effects() {
        let kinds: Vec<(&str, EffectKind)> = effects().iter().map(|e| (e.id(), e.kind())).collect();
        assert_eq!(
            kinds,
            [
                ("selftest-invert", EffectKind::Frame),
                ("selftest-shift", EffectKind::Temporal),
                ("selftest-integrate", EffectKind::Temporal),
                ("selftest-threads", EffectKind::Temporal),
                ("selftest-gap", EffectKind::Temporal)
            ]
        );
        assert_eq!(Shift.layers(), vec![LayerSpec { id: "overlay", name: "Overlay", optional: true }]);
        assert!(Integrate.layers().is_empty());
        let ctx = RenderContext::default();
        let params = fx_core::resolve_params(&Shift, &Params::default()).unwrap();
        let mut out = Frame::new(2, 2, Depth::U8);
        assert!(matches!(Shift.render(&ctx, &params, &frame8(1), &mut out), Err(FxError::Unsupported(_))));
    }

    #[test]
    fn shift_reads_the_frame_offset_away_and_is_transparent_outside() {
        let clip = clip8(&[10, 20, 30, 40]);
        let layers = LayerSet::new();
        let mut curves = ParamCurves::new(Shift.params());
        let at = |curves: &ParamCurves, n: u64| first8(&render(&Shift, curves, &clip, &layers, n, 6));
        assert_eq!(at(&curves, 1), [20, 21, 22, 255]);
        curves.set_constant("offset", ParamValue::Int(2)).unwrap();
        assert_eq!(at(&curves, 0), [30, 31, 32, 255]);
        assert_eq!(at(&curves, 1), [40, 41, 42, 255]);
        assert_eq!(at(&curves, 2), [0, 0, 0, 0]);
        assert_eq!(at(&curves, 5), [0, 0, 0, 0]);
        curves.set_constant("offset", ParamValue::Int(-1)).unwrap();
        assert_eq!(at(&curves, 0), [0, 0, 0, 0]);
        assert_eq!(at(&curves, 3), [30, 31, 32, 255]);
        assert_eq!(at(&curves, 4), [40, 41, 42, 255]);
        assert_eq!(at(&curves, 5), [0, 0, 0, 0]);
    }

    #[test]
    fn shift_follows_a_stepped_offset_curve() {
        let clip = clip8(&[10, 20, 30, 40, 50]);
        let mut curves = ParamCurves::new(Shift.params());
        curves.parse_cli("offset", "0@0;2@2", base()).unwrap();
        let at = |n: u64| first8(&render(&Shift, &curves, &clip, &LayerSet::new(), n, 5))[0];
        // an int holds between keys: 0 until frame 2, then 2
        assert_eq!([at(0), at(1), at(2)], [10, 20, 50]);
    }

    #[test]
    fn the_overlay_is_mixed_at_fifty_percent_and_missing_frames_count_as_empty() {
        let clip = clip8(&[10, 20, 30]);
        let mut layers = LayerSet::new();
        layers.insert("overlay", Arc::new(clip8(&[100, 200])));
        let curves = ParamCurves::new(Shift.params());
        let at = |n: u64| first8(&render(&Shift, &curves, &clip, &layers, n, 3));
        assert_eq!(at(0), [55, 56, 57, 255]);
        // (20 + 200 + 1) / 2
        assert_eq!(at(1), [110, 111, 112, 255]);
        // the overlay has no frame 2: (30 + 0 + 1) / 2
        assert_eq!(at(2), [15, 16, 16, 128]);
    }

    #[test]
    fn dependencies_list_the_fetches() {
        let clip = clip8(&[1, 2, 3]);
        let mut curves = ParamCurves::new(Shift.params());
        curves.set_constant("offset", ParamValue::Int(-3)).unwrap();
        let cx = TemporalContext::new(2, 2, Depth::U8, base(), 4, 5);
        let none = LayerSet::new();
        let env = Env { params: &curves, input: &clip, layers: &none };
        assert_eq!(Shift.dependencies(&cx, &env).unwrap(), vec![Checkout { layer: None, time: 100 }]);
        let mut with = LayerSet::new();
        with.insert("overlay", Arc::new(clip8(&[1])));
        let env = Env { params: &curves, input: &clip, layers: &with };
        assert_eq!(
            Shift.dependencies(&cx, &env).unwrap(),
            vec![Checkout { layer: None, time: 100 }, Checkout { layer: Some("overlay"), time: 400 }]
        );
        let env = Env { params: &curves, input: &clip, layers: &none };
        assert_eq!(Integrate.dependencies(&cx, &env).unwrap(), vec![Checkout { layer: None, time: 400 }]);
    }

    #[test]
    fn integrate_adds_the_running_sum_and_leaves_frame_zero_alone() {
        let clip = clip8(&[10, 10, 10, 10, 10]);
        let mut curves = ParamCurves::new(Integrate.params());
        curves.set_constant("gain", ParamValue::Float(3.0)).unwrap();
        let at = |n: u64| first8(&render(&Integrate, &curves, &clip, &LayerSet::new(), n, 5));
        assert_eq!(at(0), [10, 11, 12, 255]);
        assert_eq!(at(1), [13, 14, 15, 255]);
        assert_eq!(at(4), [22, 23, 24, 255]);
        // alpha is a channel too, and everything clamps
        let loud = clip8(&[250; 3]);
        let mut curves = ParamCurves::new(Integrate.params());
        curves.set_constant("gain", ParamValue::Float(100.0)).unwrap();
        assert_eq!(first8(&render(&Integrate, &curves, &loud, &LayerSet::new(), 2, 3)), [255, 255, 255, 255]);
        curves.set_constant("gain", ParamValue::Float(-100.0)).unwrap();
        assert_eq!(first8(&render(&Integrate, &curves, &loud, &LayerSet::new(), 2, 3)), [50, 51, 52, 55]);
    }

    #[test]
    fn integrate_samples_the_curve_at_every_earlier_frame() {
        // gain ramps 0 -> 8 over frames 0..4, so the samples at frames 0..n are 0, 2, 4, 6
        let clip = clip8(&[100; 6]);
        let mut curves = ParamCurves::new(Integrate.params());
        curves.parse_cli("gain", "0@0;8@4", base()).unwrap();
        let at = |n: u64| first8(&render(&Integrate, &curves, &clip, &LayerSet::new(), n, 6))[0];
        assert_eq!([at(0), at(1), at(2), at(3), at(4)], [100, 100, 102, 106, 112]);
        // past the last key the gain stays 8
        assert_eq!(at(5), 112 + 8);
    }

    #[test]
    fn integrate_scales_to_16_bit_and_float() {
        let mut curves = ParamCurves::new(Integrate.params());
        curves.set_constant("gain", ParamValue::Float(2.0)).unwrap();
        let mut f16 = Frame::new(2, 2, Depth::U16);
        f16.pixels = Pixels::U16(vec![[1000, 2000, 3000, 65535]; 4]);
        let clip = MemLayer::new(base(), vec![f16.clone(), f16]).unwrap();
        let out = render(&Integrate, &curves, &clip, &LayerSet::new(), 1, 2);
        assert_eq!(out.pixels, Pixels::U16(vec![[1514, 2514, 3514, 65535]; 4]));

        let mut ff = Frame::new(2, 2, Depth::F32);
        ff.pixels = Pixels::F32(vec![[0.25, 0.5, 2.0, -1.0]; 4]);
        let clip = MemLayer::new(base(), vec![ff.clone(), ff]).unwrap();
        let out = render(&Integrate, &curves, &clip, &LayerSet::new(), 1, 2);
        let Pixels::F32(px) = out.pixels else { unreachable!() };
        assert_eq!(px[0], [(0.25f64 + 2.0 / 255.0) as f32, (0.5f64 + 2.0 / 255.0) as f32, 1.0, 0.0]);
    }

    #[test]
    fn a_frame_past_the_clip_is_transparent_black_plus_the_sum() {
        let clip = clip8(&[10, 10]);
        let mut curves = ParamCurves::new(Integrate.params());
        curves.set_constant("gain", ParamValue::Float(1.0)).unwrap();
        assert_eq!(first8(&render(&Integrate, &curves, &clip, &LayerSet::new(), 3, 4)), [3, 3, 3, 3]);
    }

    #[test]
    fn invert_flips_colour_and_optionally_alpha() {
        let mut input = Frame::new(1, 2, Depth::U8);
        input.pixels = Pixels::U8(vec![[0, 100, 255, 40], [1, 2, 3, 4]]);
        let run = |alpha: bool| {
            let mut given = Params::default();
            given.set("invert_alpha", ParamValue::Bool(alpha));
            let params = fx_core::resolve_params(&Invert, &given).unwrap();
            let mut out = Frame::new(1, 2, Depth::U8);
            Invert.render(&RenderContext::default(), &params, &input, &mut out).unwrap();
            out.pixels
        };
        assert_eq!(run(false), Pixels::U8(vec![[255, 155, 0, 40], [254, 253, 252, 4]]));
        assert_eq!(run(true), Pixels::U8(vec![[255, 155, 0, 215], [254, 253, 252, 251]]));
    }

    #[test]
    fn the_threads_effect_reports_the_pool_size_in_the_first_pixel_only() {
        let clip = clip8(&[10, 20]);
        let curves = ParamCurves::new(Vec::new());
        let out = render(&Threads, &curves, &clip, &LayerSet::new(), 1, 2);
        let n = rayon::current_num_threads().min(255) as u8;
        assert_eq!(first8(&out), [n, 21, 22, 255]);
        let Pixels::U8(px) = &out.pixels else { unreachable!() };
        assert_eq!(px[1], [20, 20, 20, 20]);
    }

    #[test]
    fn the_gap_effect_copies_its_input_with_the_gap_on_or_off() {
        let clip = clip8(&[10, 20, 30]);
        let mut curves = ParamCurves::new(Gap.params());
        let frames = |curves: &ParamCurves| (0..4).map(|n| render(&Gap, curves, &clip, &LayerSet::new(), n, 4)).collect::<Vec<_>>();
        let off = frames(&curves);
        assert_eq!(off[..3], [frame8(10), frame8(20), frame8(30)]);
        assert_eq!(off[3], Frame::new(2, 2, Depth::U8), "past the clip it is transparent black");
        // the record itself is process wide and only fx-core's tests read it; the note is pinned
        // through the binary (tests/cli.rs)
        curves.set_constant("gap", ParamValue::Bool(true)).unwrap();
        assert_eq!(frames(&curves), off);
        assert_eq!(Gap.gaps(), [&T1]);
        let none = LayerSet::new();
        let env = Env { params: &curves, input: &clip, layers: &none };
        let cx = TemporalContext::new(2, 2, Depth::U8, base(), 2, 4);
        assert_eq!(Gap.dependencies(&cx, &env).unwrap(), vec![Checkout { layer: None, time: 200 }]);
        let mut out = Frame::new(2, 2, Depth::U8);
        assert!(matches!(Gap.render(&RenderContext::default(), &Params::default(), &frame8(1), &mut out), Err(FxError::Unsupported(_))));
    }

    #[test]
    fn mismatched_sources_are_errors_not_panics() {
        let mut out = Frame::new(3, 3, Depth::U8);
        assert!(matches!(load(&mut out, Some(&frame8(1))), Err(FxError::Mismatch(_))));
        load(&mut out, None).unwrap();
        assert_eq!(out, Frame::new(3, 3, Depth::U8));
    }
}
