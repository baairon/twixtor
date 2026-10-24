//! `vfx apply` for temporal effects: random access to the clip, extra layers, parameter curves, and
//! frames rendered in order and written as they finish.
//!
//! What an effect author can rely on:
//! - `render_temporal` runs once per output frame `n` in `0..frame_count`, in order, on the main
//!   thread; the effect may use rayon inside (`fx_core::par_rows`) but the bytes must not depend on
//!   the thread count or on the run. `output` arrives zeroed, sized and typed like the clip; an
//!   `Err` or a panic ends the run with exit code 3 and names the frame.
//! - `cx.time` is `--time-scale` or `TimeBase::default_for_rate(fps)`: 100 ticks a frame at an integer
//!   rate, `num/den` at an NTSC style rate. `cx.current_time` is `n * time_step`,
//!   `cx.local_time_step` is `time_step`, `cx.fps` the reduced `--fps`, `cx.frame_count` `--frames`
//!   (default: the frames of the input clip), `cx.quality` and `cx.field` the flags.
//! - `env.input` and every layer given by `--layer ID=FILE` are clips of whole frames in the wire
//!   format of the depth (straight alpha; `rgba`, `rgba64le` 0..65535, planar `gbrapf32le`), all of
//!   the output's size and depth, each with its own length. Frame `i` covers the ticks
//!   `i * time_step .. (i + 1) * time_step`; `frame_at` gives `None` before the start and past the
//!   end (the effect decides what that shows, transparent black being the usual choice), and an
//!   `Err` only when the file cannot be read, which `vfx` reports as an io error (exit code 4).
//! - A layer the effect lists as required must be given, an optional one is simply absent from
//!   `env.layers`; ids not listed by `layers()` are refused (exit code 3).
//! - `env.params` samples every parameter at any tick: constants, or the keyframes `V@F;V@F` (`h`
//!   after `F` holds, `e` eases); floats, angles and points interpolate, other kinds hold.

use crate::apply::{frame_len, guarded, is_stdio, open_output, same_file};
use crate::cli::ApplyArgs;
use crate::clip::{spool, IoNote, RawClip, SpoolGuard};
use crate::error::CliError;
use fx_core::{Effect, Env, Frame, Layer, LayerSet, ParamCurves, Rational, TemporalContext, TimeBase};
use std::fs::File;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// `24` or `30000/1001`, as `--fps` takes it.
pub fn rate_text(fps: Rational) -> String {
    if fps.den == 1 {
        fps.num.to_string()
    } else {
        format!("{}/{}", fps.num, fps.den)
    }
}

/// The time base of a run: `--time-scale` if given, else the workspace default. A
/// frame must last a whole number of ticks, `time_scale * den / num`.
pub fn time_base(fps: Rational, time_scale: Option<u32>) -> Result<TimeBase, CliError> {
    let Some(scale) = time_scale else {
        if fps.den == 1 && fps.num as u64 * 100 > u32::MAX as u64 {
            return Err(CliError::usage(format!(
                "{} fps is too fast for the default time base (100 ticks per frame), give --time-scale",
                fps.num
            )));
        }
        return Ok(TimeBase::default_for_rate(fps));
    };
    let ticks = scale as u64 * fps.den as u64;
    let num = fps.num as u64;
    let per_frame = ticks / num;
    if ticks % num != 0 || per_frame == 0 || per_frame > u32::MAX as u64 {
        return Err(CliError::usage(format!(
            "--time-scale {scale} gives {:.3} ticks per frame at {} fps, but a frame must last a whole number of ticks: use a multiple of {num}",
            ticks as f64 / num as f64,
            rate_text(fps)
        )));
    }
    Ok(TimeBase { time_scale: scale, time_step: per_frame as u32 })
}

/// Every parameter at its default, then the `--param` values, plain or keyframed.
pub fn build_curves(effect: &dyn Effect, given: &[(String, String)], time: TimeBase) -> Result<ParamCurves, CliError> {
    let mut curves = ParamCurves::new(effect.params());
    for (id, text) in given {
        curves.parse_cli(id, text, time)?;
    }
    Ok(curves)
}

/// The `--layer` ids must be ones the effect lists, and every required layer must be there.
pub fn check_layers(effect: &dyn Effect, given: &[(String, PathBuf)]) -> Result<(), CliError> {
    let specs = effect.layers();
    let known = || {
        if specs.is_empty() {
            "no layers".to_string()
        } else {
            format!("the layers {}", specs.iter().map(|s| s.id).collect::<Vec<_>>().join(", "))
        }
    };
    for (id, _) in given {
        if !specs.iter().any(|s| s.id == id) {
            return Err(CliError::Effect(format!("unknown layer `{id}`: {} has {}", effect.id(), known())));
        }
    }
    for spec in specs.iter().filter(|s| !s.optional) {
        if !given.iter().any(|(id, _)| id == spec.id) {
            return Err(CliError::Effect(format!(
                "{} needs the layer `{}` ({}), pass --layer {}=FILE",
                effect.id(),
                spec.id,
                spec.name,
                spec.id
            )));
        }
    }
    Ok(())
}

/// Renders `frames` frames in order and writes each one as it is done. `note` is
/// where the clips leave read errors; one found after a frame wins over what the
/// effect made of it, so a vanished file is an io error and not a wrong picture.
#[allow(clippy::too_many_arguments)]
pub fn render_all(
    effect: &dyn Effect,
    args: &ApplyArgs,
    time: TimeBase,
    curves: &ParamCurves,
    input: &dyn Layer,
    layers: &LayerSet,
    frames: u64,
    note: &IoNote,
    out: &mut dyn Write,
) -> Result<(), CliError> {
    let (w, h, depth) = (args.width, args.height, args.depth);
    for n in 0..frames {
        let mut cx = TemporalContext::new(w, h, depth, time, n, frames);
        if let Some(quality) = args.quality {
            cx.quality = quality;
        }
        if let Some(field) = args.field {
            cx.field = field;
        }
        let env = Env { params: curves, input, layers };
        let mut frame = Frame::new(w, h, depth);
        let rendered = guarded(n, || effect.render_temporal(&cx, &env, &mut frame));
        if let Some(message) = note.take() {
            return Err(CliError::Io(message));
        }
        rendered?;
        if (frame.width, frame.height, frame.depth()) != (w, h, depth) {
            return Err(CliError::Effect(format!("frame {n}: effect changed the frame size or depth")));
        }
        out.write_all(&frame.to_bytes())
            .and_then(|_| out.flush())
            .map_err(|e| CliError::io("write error", e))?;
    }
    Ok(())
}

fn open_clip(path: &Path, what: String, args: &ApplyArgs, time: TimeBase, note: &IoNote) -> Result<RawClip, CliError> {
    let file = File::open(path).map_err(|e| CliError::io(format!("cannot open {what}"), e))?;
    RawClip::new(file, what, (args.width, args.height, args.depth), time, note.clone())
}

pub fn run(effect: &dyn Effect, args: &ApplyArgs) -> Result<(), CliError> {
    let fps = args
        .fps
        .ok_or_else(|| CliError::usage(format!("{} is a temporal effect and needs --fps N[/D], for example 24 or 30000/1001", effect.id())))?;
    let fps = fps.exact.ok_or_else(|| {
        CliError::usage(format!(
            "--fps {} is not an exact rate: write it as a fraction such as 30000/1001, or a decimal of at most 6 places",
            fps.value
        ))
    })?;
    let time = time_base(fps, args.time_scale)?;
    let curves = build_curves(effect, &args.params, time)?;
    check_layers(effect, &args.layers)?;
    frame_len(args)?;
    if args.check {
        return Ok(());
    }

    let (input, output) = (args.input.as_deref().filter(|p| !is_stdio(p)), args.output.as_deref());
    if let Some(o) = output.filter(|o| !is_stdio(o)) {
        let inputs = input.into_iter().chain(args.layers.iter().map(|(_, p)| p.as_path()));
        if let Some(clash) = inputs.into_iter().find(|i| same_file(i, o)) {
            return Err(CliError::usage(format!("--out and {} are the same file", clash.display())));
        }
    }

    let note = IoNote::default();
    // declared before the clips so that it drops after them: the spool file is
    // closed by then, which Windows needs before it lets go of the name
    let mut spooled: Option<SpoolGuard> = None;
    let main = match input {
        Some(path) => open_clip(path, format!("input {}", path.display()), args, time, &note)?,
        None => {
            if io::stdin().is_terminal() {
                return Err(CliError::usage("refusing to read raw frames from a terminal, pipe them in or use --in"));
            }
            let dir = args.tmp_dir.clone().unwrap_or_else(std::env::temp_dir);
            let (guard, file) = spool(&mut io::stdin().lock(), &dir)?;
            spooled = Some(guard);
            RawClip::new(file, "input", (args.width, args.height, args.depth), time, note.clone())?
        }
    };
    let mut layers = LayerSet::new();
    for (id, path) in &args.layers {
        let clip = open_clip(path, format!("layer {id} {}", path.display()), args, time, &note)?;
        if clip.frame_count() == 0 {
            return Err(CliError::Io(format!("layer {id} {} holds no frames", path.display())));
        }
        layers.insert(id, Arc::new(clip));
    }

    let frames = args.frames.unwrap_or_else(|| main.frame_count());
    let mut writer = open_output(output)?;
    let result = render_all(effect, args, time, &curves, &main, &layers, frames, &note, &mut writer);
    drop(writer);
    drop(layers);
    drop(main);
    drop(spooled);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::tests::TempDir;
    use crate::selftest::{Integrate, Shift};
    use fx_core::{Depth, MemLayer, ParamValue, Pixels, Quality};

    fn fps(text: &str) -> Rational {
        Rational::parse(text).unwrap()
    }

    #[test]
    fn rate_text_round_trips() {
        for t in ["24", "30000/1001", "2997/100"] {
            assert_eq!(rate_text(fps(t)), t);
        }
        assert_eq!(rate_text(fps("29.97")), "2997/100");
    }

    #[test]
    fn the_default_base_is_the_workspace_guess() {
        let b = time_base(fps("24"), None).unwrap();
        assert_eq!((b.time_scale, b.time_step), (2400, 100));
        let b = time_base(fps("30000/1001"), None).unwrap();
        assert_eq!((b.time_scale, b.time_step), (30000, 1001));
        assert_eq!(time_base(fps("42949672"), None).unwrap().time_step, 100);
        assert_eq!(time_base(fps("42949673"), None).unwrap_err().code(), 2);
    }

    #[test]
    fn a_time_scale_must_give_whole_ticks_per_frame() {
        let b = time_base(fps("24"), Some(24)).unwrap();
        assert_eq!((b.time_scale, b.time_step), (24, 1));
        let b = time_base(fps("24"), Some(48000)).unwrap();
        assert_eq!((b.time_scale, b.time_step), (48000, 2000));
        let b = time_base(fps("30000/1001"), Some(60000)).unwrap();
        assert_eq!((b.time_scale, b.time_step), (60000, 2002));
        let b = time_base(fps("29.97"), Some(2997)).unwrap();
        assert_eq!((b.time_scale, b.time_step), (2997, 100));
        for (rate, scale) in [("24", 1000), ("24", 1), ("30000/1001", 30001), ("30000/1001", 24000), ("29.97", 3000)] {
            let err = time_base(fps(rate), Some(scale)).unwrap_err();
            assert_eq!(err.code(), 2, "{rate} {scale}");
            assert!(err.to_string().contains("whole number of ticks"), "{err}");
        }
        let msg = time_base(fps("24"), Some(1000)).unwrap_err().to_string();
        assert!(msg.contains("41.667 ticks per frame at 24 fps") && msg.contains("multiple of 24"), "{msg}");
        // a frame longer than u32 ticks is refused instead of wrapping
        assert!(time_base(fps("1"), Some(u32::MAX)).is_ok());
        assert!(time_base(fps("1/4294967295"), Some(u32::MAX)).is_err());
    }

    #[test]
    fn curves_take_plain_and_keyframed_values() {
        let base = time_base(fps("24"), None).unwrap();
        let given = vec![("gain".to_string(), "0@0;10@4".to_string())];
        let curves = build_curves(&Integrate, &given, base).unwrap();
        assert_eq!(curves.float_at("gain", 200), 5.0);
        let plain = build_curves(&Integrate, &[("gain".to_string(), "7.5".to_string())], base).unwrap();
        assert_eq!(plain.float_at("gain", 12345), 7.5);
        assert_eq!(build_curves(&Integrate, &[], base).unwrap().float_at("gain", 5), 0.0);
        for (id, text, word) in [("nope", "1", "nope"), ("gain", "999", "gain"), ("gain", "1@0;2@0", "same time"), ("gain", "x", "gain")] {
            let err = build_curves(&Integrate, &[(id.to_string(), text.to_string())], base).unwrap_err();
            assert_eq!(err.code(), 3, "{id}={text}");
            assert!(err.to_string().contains(word), "{err}");
        }
    }

    #[test]
    fn layer_ids_and_required_layers_are_checked() {
        let one = |id: &str| vec![(id.to_string(), PathBuf::from("x.raw"))];
        assert!(check_layers(&Shift, &[]).is_ok());
        assert!(check_layers(&Shift, &one("overlay")).is_ok());
        let err = check_layers(&Shift, &one("nope")).unwrap_err();
        assert_eq!(err.code(), 3);
        assert_eq!(err.to_string(), "unknown layer `nope`: selftest-shift has the layers overlay");
        let err = check_layers(&Integrate, &one("overlay")).unwrap_err();
        assert!(err.to_string().ends_with("selftest-integrate has no layers"), "{err}");

        struct Needy;
        impl Effect for Needy {
            fn id(&self) -> &'static str {
                "needy"
            }
            fn name(&self) -> &'static str {
                "Needy"
            }
            fn params(&self) -> Vec<fx_core::ParamSpec> {
                Vec::new()
            }
            fn layers(&self) -> Vec<fx_core::LayerSpec> {
                vec![fx_core::LayerSpec { id: "matte", name: "Matte", optional: false }]
            }
            fn render(&self, _: &fx_core::RenderContext, _: &fx_core::Params, _: &Frame, _: &mut Frame) -> fx_core::Result<()> {
                Ok(())
            }
        }
        let err = check_layers(&Needy, &[]).unwrap_err();
        assert_eq!(err.code(), 3);
        assert_eq!(err.to_string(), "needy needs the layer `matte` (Matte), pass --layer matte=FILE");
        assert!(check_layers(&Needy, &one("matte")).is_ok());
    }

    fn frame8(v: u8) -> Frame {
        let mut f = Frame::new(2, 1, Depth::U8);
        f.pixels = Pixels::U8(vec![[v, v, v, 255]; 2]);
        f
    }

    fn render(effect: &dyn Effect, args: &ApplyArgs, curves: &ParamCurves, clip: &MemLayer, layers: &LayerSet, frames: u64, note: &IoNote) -> (Vec<u8>, Result<(), CliError>) {
        let mut out = Vec::new();
        let time = time_base(fps("24"), None).unwrap();
        let r = render_all(effect, args, time, curves, clip, layers, frames, note, &mut out);
        (out, r)
    }

    fn setup() -> (ApplyArgs, TimeBase, MemLayer) {
        let args = ApplyArgs::new("selftest-shift", 2, 1, Depth::U8);
        let time = time_base(fps("24"), None).unwrap();
        let clip = MemLayer::new(time, (0..4).map(|i| frame8(10 * (i + 1))).collect()).unwrap();
        (args, time, clip)
    }

    #[test]
    fn frames_come_out_in_order_and_may_outrun_the_clip() {
        let (args, _, clip) = setup();
        let mut curves = ParamCurves::new(Shift.params());
        curves.set_constant("offset", ParamValue::Int(1)).unwrap();
        let (out, r) = render(&Shift, &args, &curves, &clip, &LayerSet::new(), 6, &IoNote::default());
        r.unwrap();
        let firsts: Vec<u8> = out.chunks(8).map(|f| f[0]).collect();
        assert_eq!(firsts, [20, 30, 40, 0, 0, 0]);
    }

    #[test]
    fn an_effect_error_names_the_frame_and_stops_the_output() {
        let (args, _, clip) = setup();
        let mut curves = ParamCurves::new(Shift.params());
        curves.set_constant("offset", ParamValue::Int(0)).unwrap();
        // a layer of another size makes the overlay mix fail at frame 0
        let mut layers = LayerSet::new();
        layers.insert("overlay", Arc::new(MemLayer::new(time_base(fps("24"), None).unwrap(), vec![Frame::new(3, 3, Depth::U8)]).unwrap()));
        let (out, r) = render(&Shift, &args, &curves, &clip, &layers, 3, &IoNote::default());
        let err = r.unwrap_err();
        assert_eq!(err.code(), 3);
        assert!(err.to_string().starts_with("frame 0:") && err.to_string().contains("overlay"), "{err}");
        assert!(out.is_empty());
    }

    #[test]
    fn a_note_from_a_clip_turns_into_an_io_error() {
        let (args, _, clip) = setup();
        let curves = ParamCurves::new(Integrate.params());
        let note = IoNote::default();
        note.set("cannot read frame 2 of input x.raw: gone".into());
        let (out, r) = render(&Integrate, &args, &curves, &clip, &LayerSet::new(), 3, &note);
        let err = r.unwrap_err();
        assert_eq!(err.code(), 4);
        assert!(err.to_string().contains("gone"), "{err}");
        assert!(out.is_empty(), "the frame that met the error is not written");
    }

    #[test]
    fn quality_and_field_reach_the_context() {
        struct Spy;
        impl Effect for Spy {
            fn id(&self) -> &'static str {
                "spy"
            }
            fn name(&self) -> &'static str {
                "Spy"
            }
            fn params(&self) -> Vec<fx_core::ParamSpec> {
                Vec::new()
            }
            fn kind(&self) -> fx_core::EffectKind {
                fx_core::EffectKind::Temporal
            }
            fn render(&self, _: &fx_core::RenderContext, _: &fx_core::Params, _: &Frame, _: &mut Frame) -> fx_core::Result<()> {
                Ok(())
            }
            fn render_temporal(&self, cx: &TemporalContext, _: &Env<'_>, out: &mut Frame) -> fx_core::Result<()> {
                let q = u8::from(cx.quality == Quality::Draft);
                let f = match cx.field {
                    fx_core::FieldOrder::Progressive => 0,
                    fx_core::FieldOrder::UpperFirst => 1,
                    fx_core::FieldOrder::LowerFirst => 2,
                };
                let t = [cx.frame_index as u8, cx.frame_count as u8, (cx.current_time / 100) as u8, cx.local_time_step as u8];
                out.pixels = Pixels::U8(vec![[q, f, t[0], t[1]], [t[2], t[3], cx.fps.num as u8, cx.fps.den as u8]]);
                Ok(())
            }
        }
        let (mut args, _, clip) = setup();
        let curves = ParamCurves::new(Vec::new());
        let (out, r) = render(&Spy, &args, &curves, &clip, &LayerSet::new(), 2, &IoNote::default());
        r.unwrap();
        assert_eq!(out, [0, 0, 0, 2, 0, 100, 24, 1, 0, 0, 1, 2, 1, 100, 24, 1]);
        args.quality = Some(Quality::Draft);
        args.field = Some(fx_core::FieldOrder::LowerFirst);
        let (out, r) = render(&Spy, &args, &curves, &clip, &LayerSet::new(), 1, &IoNote::default());
        r.unwrap();
        assert_eq!(&out[..2], [1, 2]);
    }

    fn write_clip(dir: &TempDir, name: &str, frames: u8) -> PathBuf {
        let bytes: Vec<u8> = (0..frames).flat_map(|i| [10 * (i + 1), 0, 0, 255, 10 * (i + 1), 1, 0, 255]).collect();
        dir.file(name, &bytes)
    }

    #[test]
    fn run_reads_files_applies_layers_and_writes_the_output() {
        let dir = TempDir::new("run");
        let input = write_clip(&dir, "in.raw", 3);
        let over = write_clip(&dir, "over.raw", 3);
        let out = dir.0.join("out.raw");
        let mut args = ApplyArgs::new("selftest-shift", 2, 1, Depth::U8);
        args.fps = Some(crate::cli::FpsArg { exact: Some(fps("24")), value: 24.0 });
        args.input = Some(input.clone());
        args.output = Some(out.clone());
        args.params = vec![("offset".into(), "1".into())];
        args.layers = vec![("overlay".into(), over.clone())];
        run(&Shift, &args).unwrap();
        let got = std::fs::read(&out).unwrap();
        // frame n = input n+1 mixed with the overlay's frame n: (20 + 10 + 1) / 2 = 15 ...
        assert_eq!(got.len(), 3 * 8);
        assert_eq!([got[0], got[8], got[16]], [(20 + 10 + 1) / 2, (30 + 20 + 1) / 2, (0 + 30 + 1) / 2]);

        args.frames = Some(5);
        run(&Shift, &args).unwrap();
        assert_eq!(std::fs::read(&out).unwrap().len(), 5 * 8);
        args.frames = Some(1);
        run(&Shift, &args).unwrap();
        assert_eq!(std::fs::read(&out).unwrap().len(), 8);
    }

    #[test]
    fn run_reports_each_kind_of_mistake_with_its_exit_code() {
        let dir = TempDir::new("mistakes");
        let input = write_clip(&dir, "in.raw", 2);
        let mut ragged = std::fs::read(&input).unwrap();
        ragged.push(1);
        let bad = dir.file("bad.raw", &ragged);
        let empty = dir.file("empty.raw", &[]);
        // the output file exists, so that naming it as a layer is caught as the same file
        dir.file("o.raw", &std::fs::read(&input).unwrap());
        let base = || {
            let mut a = ApplyArgs::new("selftest-shift", 2, 1, Depth::U8);
            a.fps = Some(crate::cli::FpsArg { exact: Some(fps("24")), value: 24.0 });
            a.input = Some(input.clone());
            a.output = Some(dir.0.join("o.raw"));
            a
        };
        let code = |a: ApplyArgs| run(&Shift, &a).unwrap_err().code();

        let mut a = base();
        a.fps = None;
        assert_eq!(code(a), 2);
        let mut a = base();
        a.fps = Some(crate::cli::FpsArg { exact: None, value: 29.97002997 });
        assert_eq!(code(a), 2);
        let mut a = base();
        a.time_scale = Some(1000);
        assert_eq!(code(a), 2);
        let mut a = base();
        a.output = a.input.clone();
        assert_eq!(code(a), 2);
        let mut a = base();
        a.layers = vec![("overlay".into(), dir.0.join("o.raw"))];
        a.output = Some(dir.0.join("o.raw"));
        assert_eq!(code(a), 2);
        let mut a = base();
        a.params = vec![("offset".into(), "99".into())];
        assert_eq!(code(a), 3);
        let mut a = base();
        a.layers = vec![("nope".into(), input.clone())];
        assert_eq!(code(a), 3);
        let mut a = base();
        a.input = Some(dir.0.join("missing.raw"));
        assert_eq!(code(a), 4);
        let mut a = base();
        a.input = Some(bad.clone());
        let err = run(&Shift, &a).unwrap_err();
        assert_eq!(err.code(), 4);
        assert!(err.to_string().contains("partial frame (1 of 8 bytes) after 2 whole frames"), "{err}");
        let mut a = base();
        a.layers = vec![("overlay".into(), bad)];
        let err = run(&Shift, &a).unwrap_err();
        assert!(err.to_string().starts_with("layer overlay "), "{err}");
        assert_eq!(err.code(), 4);
        let mut a = base();
        a.layers = vec![("overlay".into(), empty.clone())];
        assert_eq!(code(a), 4);
        let mut a = base();
        a.layers = vec![("overlay".into(), dir.0.join("gone.raw"))];
        assert_eq!(code(a), 4);
        // an empty main input is zero frames, not an error
        let mut a = base();
        a.input = Some(empty);
        run(&Shift, &a).unwrap();
        assert_eq!(std::fs::read(dir.0.join("o.raw")).unwrap().len(), 0);
    }

    #[test]
    fn check_validates_without_touching_a_file() {
        let mut a = ApplyArgs::new("selftest-shift", 2, 1, Depth::U8);
        a.fps = Some(crate::cli::FpsArg { exact: Some(fps("30000/1001")), value: 29.97 });
        a.check = true;
        a.input = Some(PathBuf::from("does-not-exist.raw"));
        a.output = Some(PathBuf::from("/definitely/not/writable/out.raw"));
        a.layers = vec![("overlay".into(), PathBuf::from("neither.raw"))];
        a.params = vec![("offset".into(), "-2@0;3@10".into())];
        run(&Shift, &a).unwrap();
        a.params = vec![("offset".into(), "9".into())];
        assert_eq!(run(&Shift, &a).unwrap_err().code(), 3);
        a.params.clear();
        a.layers = vec![("nope".into(), PathBuf::from("x"))];
        assert_eq!(run(&Shift, &a).unwrap_err().code(), 3);
        a.layers.clear();
        a.fps = None;
        assert_eq!(run(&Shift, &a).unwrap_err().code(), 2);
    }
}
