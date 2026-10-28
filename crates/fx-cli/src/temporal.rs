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
//!   (default: the frames of the input clip), `cx.quality`, `cx.field` and `cx.device` the flags
//!   (`gpu=N` that cannot be served is refused before the first frame, `gpu` without a device
//!   that passes the gate renders on the CPU after one note). With a device, `cx.executor` is the
//!   run's executor (a `retime_engine::stages::StageSet`), and the first frame that used the
//!   device is rendered a second time on the CPU and compared byte for byte; a difference refuses
//!   the device for the rest of the run (one note) and the CPU's frame is written.
//! - `env.input` and every layer given by `--layer ID=FILE` are clips of whole frames in the wire
//!   format of the depth (straight alpha; `rgba`, `rgba64le` 0..65535, planar `gbrapf32le`), all of
//!   the output's size and depth, each with its own length. Frame `i` covers the ticks
//!   `i * time_step .. (i + 1) * time_step`, or, for a clip given its own rate R by `--clip-fps`
//!   (the input) or `--layer-fps ID=R` (a layer), the time `i / R .. (i + 1) / R` seconds
//!   (`clip::RawClip::with_rate`, exact integers); with another `--clip-fps` the default
//!   `--frames` is the output frames that start inside the input clip (`clip::output_frames`).
//!   `frame_at` gives `None` before the start and past the
//!   end (the effect decides what that shows, transparent black being the usual choice), and an
//!   `Err` only when the file cannot be read, which `vfx` reports as an io error (exit code 4).
//! - A layer the effect lists as required must be given, an optional one is simply absent from
//!   `env.layers`; ids not listed by `layers()` are refused (exit code 3).
//! - `env.params` samples every parameter at any tick: constants, or the keyframes `V@F;V@F` (`h`
//!   after `F` holds, `e` eases, `iS/I` and `oS/I` after `F` are Bezier handles); floats, angles
//!   and points interpolate, other kinds hold.

use crate::apply::{frame_len, guarded, is_stdio, open_output, same_file};
use crate::cli::ApplyArgs;
use crate::clip::{output_frames, spool, IoNote, RawClip, SpoolGuard};
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

/// Every `--layer-fps` id must be a layer the effect lists (exit code 3, as for `--layer`) and one
/// that `--layer` gives (a usage error).
pub fn check_layer_rates(effect: &dyn Effect, args: &ApplyArgs) -> Result<(), CliError> {
    let specs = effect.layers();
    for (id, rate) in &args.layer_fps {
        if !specs.iter().any(|s| s.id == id) {
            let known = if specs.is_empty() {
                "no layers".to_string()
            } else {
                format!("the layers {}", specs.iter().map(|s| s.id).collect::<Vec<_>>().join(", "))
            };
            return Err(CliError::Effect(format!("unknown layer `{id}` in --layer-fps: {} has {known}", effect.id())));
        }
        if !args.layers.iter().any(|(k, _)| k == id) {
            return Err(CliError::usage(format!(
                "--layer-fps {id}={} names a layer that no --layer {id}=FILE gives",
                rate_text(*rate)
            )));
        }
    }
    Ok(())
}

/// Renders `frames` frames in order and writes each one as it is done. `note` is
/// where the clips leave read errors; one found after a frame wins over what the
/// effect made of it, so a vanished file is an io error and not a wrong picture.
///
/// `notes` gets the note of every open parity gap a frame recorded (`fx_core::gaps`), once per
/// gap per run and right after the first frame that rendered with it: a frame that fails notes
/// nothing, a record left from before the run is dropped, and the notes never touch `out` or the
/// outcome of the run.
#[cfg(test)]
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
    notes: &mut dyn Write,
) -> Result<(), CliError> {
    render_all_on(effect, args, time, curves, input, layers, frames, note, out, notes, None)
}

/// One frame and how its render ended, compared by the frame's wire bytes and the error's code
/// and text (the cross check).
struct Outcome {
    frame: Frame,
    result: Result<(), CliError>,
}

impl PartialEq for Outcome {
    fn eq(&self, other: &Outcome) -> bool {
        let key = |r: &Result<(), CliError>| r.as_ref().err().map(|e| (e.code(), e.to_string()));
        (self.frame.width, self.frame.height) == (other.frame.width, other.frame.height)
            && self.frame.to_bytes() == other.frame.to_bytes()
            && key(&self.result) == key(&other.result)
    }
}

/// [`render_all`] with the run's device executor (`None`: the CPU, exactly [`render_all`]).
#[allow(clippy::too_many_arguments)]
pub fn render_all_on(
    effect: &dyn Effect,
    args: &ApplyArgs,
    time: TimeBase,
    curves: &ParamCurves,
    input: &dyn Layer,
    layers: &LayerSet,
    frames: u64,
    note: &IoNote,
    out: &mut dyn Write,
    notes: &mut dyn Write,
    device: Option<&Arc<retime_engine::device::exec::Executor>>,
) -> Result<(), CliError> {
    let (w, h, depth) = (args.width, args.height, args.depth);
    fx_core::gaps::take_recorded();
    let mut noted = std::collections::BTreeSet::new();
    let handle: Option<Arc<dyn std::any::Any + Send + Sync>> =
        device.map(|e| Arc::new(retime_engine::stages::StageSet::new(e.clone())) as Arc<dyn std::any::Any + Send + Sync>);
    for n in 0..frames {
        let mut cx = TemporalContext::new(w, h, depth, time, n, frames);
        if let Some(quality) = args.quality {
            cx.quality = quality;
        }
        if let Some(field) = args.field {
            cx.field = field;
        }
        if let Some(device) = args.device {
            cx.device = device;
        }
        let env = Env { params: curves, input, layers };
        let render = |with_device: bool| -> Outcome {
            let mut cx = cx.clone();
            if with_device {
                cx.executor = handle.clone();
            }
            let mut frame = Frame::new(w, h, depth);
            let result = guarded(n, || effect.render_temporal(&cx, &env, &mut frame));
            Outcome { frame, result }
        };
        let Outcome { frame, result: rendered } = match device {
            Some(e) => {
                let (outcome, refused) = retime_engine::device::exec::cross_checked(e.as_ref(), render);
                if let Some(why) = refused {
                    let _ = notes.write_all(format!("vfx: note: {why}\n").as_bytes());
                    let _ = notes.flush();
                }
                outcome
            }
            None => render(false),
        };
        if let Some(message) = note.take() {
            return Err(CliError::Io(message));
        }
        rendered?;
        if (frame.width, frame.height, frame.depth()) != (w, h, depth) {
            return Err(CliError::Effect(format!("frame {n}: effect changed the frame size or depth")));
        }
        print_notes(effect.id(), &fx_core::gaps::take_recorded(), &mut noted, notes);
        out.write_all(&frame.to_bytes())
            .and_then(|_| out.flush())
            .map_err(|e| CliError::io("write error", e))?;
    }
    Ok(())
}

/// The stderr line for an open parity gap that a render of `effect` used.
pub fn note_line(effect: &str, gap: &fx_core::OpenGap) -> String {
    format!(
        "vfx: note: {effect}: {} {}: a stated rule that no Twixtor reference render has graded yet (README: parity gaps)\n",
        gap.id, gap.title
    )
}

/// Writes the note of every gap of `used` that `noted` does not hold yet, and adds it there. A
/// note only informs: a failed write is ignored, so it never changes how a run ends.
fn print_notes(effect: &str, used: &[&'static fx_core::OpenGap], noted: &mut std::collections::BTreeSet<&'static str>, notes: &mut dyn Write) {
    let mut wrote = false;
    for gap in used {
        if noted.insert(gap.id) {
            let _ = notes.write_all(note_line(effect, gap).as_bytes());
            wrote = true;
        }
    }
    if wrote {
        let _ = notes.flush();
    }
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
    check_layer_rates(effect, args)?;
    frame_len(args)?;
    let device = crate::devices::open(args.device.unwrap_or_default())?;
    if let Some(why) = &device.note {
        eprintln!("vfx: note: {why}");
    }
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
    let main = match args.clip_fps {
        Some(rate) => main.with_rate(rate),
        None => main,
    };
    let mut layers = LayerSet::new();
    for (id, path) in &args.layers {
        let clip = open_clip(path, format!("layer {id} {}", path.display()), args, time, &note)?;
        if clip.frame_count() == 0 {
            return Err(CliError::Io(format!("layer {id} {} holds no frames", path.display())));
        }
        let clip = match args.layer_fps.iter().find(|(k, _)| k == id) {
            Some((_, rate)) => clip.with_rate(*rate),
            None => clip,
        };
        layers.insert(id, Arc::new(clip));
    }

    let frames = args.frames.unwrap_or_else(|| match args.clip_fps {
        Some(rate) => output_frames(main.frame_count(), rate, fps),
        None => main.frame_count(),
    });
    let mut writer = open_output(output)?;
    let exec = device.executor.as_ref();
    let result = render_all_on(effect, args, time, &curves, &main, &layers, frames, &note, &mut writer, &mut io::stderr(), exec);
    if let Some(e) = exec {
        if std::env::var_os("VFX_DEVICE_REPORT").is_some() {
            eprintln!("vfx: note: device report: {}", e.summary());
        }
    }
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

    /// The tests that run `render_all` take turns: it takes the process wide gap record, which a
    /// render running in parallel would otherwise empty under another test's feet.
    fn one_render_at_a_time() -> std::sync::MutexGuard<'static, ()> {
        static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());
        TURN.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn render(effect: &dyn Effect, args: &ApplyArgs, curves: &ParamCurves, clip: &MemLayer, layers: &LayerSet, frames: u64, note: &IoNote) -> (Vec<u8>, Result<(), CliError>) {
        let _turn = one_render_at_a_time();
        let mut out = Vec::new();
        let time = time_base(fps("24"), None).unwrap();
        let r = render_all(effect, args, time, curves, clip, layers, frames, note, &mut out, &mut io::sink());
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

    #[test]
    fn the_device_reaches_the_context() {
        struct DeviceSpy;
        impl Effect for DeviceSpy {
            fn id(&self) -> &'static str {
                "device-spy"
            }
            fn name(&self) -> &'static str {
                "Device Spy"
            }
            fn params(&self) -> Vec<fx_core::ParamSpec> {
                Vec::new()
            }
            fn kind(&self) -> fx_core::EffectKind {
                fx_core::EffectKind::Temporal
            }
            fn render_temporal(&self, cx: &TemporalContext, _: &Env<'_>, out: &mut Frame) -> fx_core::Result<()> {
                let code = match cx.device {
                    fx_core::DeviceRequest::Cpu => [0, 0],
                    fx_core::DeviceRequest::Gpu => [1, 0],
                    fx_core::DeviceRequest::GpuIndex(n) => [2, n as u8],
                };
                out.pixels = Pixels::U8(vec![[code[0], code[1], 0, 0]; 2]);
                Ok(())
            }
        }
        let (mut args, _, clip) = setup();
        let curves = ParamCurves::new(Vec::new());
        // the frame loop passes the request on as it is; `run` is what refuses what this build lacks
        for (device, want) in [(None, [0, 0]), (Some(fx_core::DeviceRequest::Cpu), [0, 0]), (Some(fx_core::DeviceRequest::Gpu), [1, 0]), (Some(fx_core::DeviceRequest::GpuIndex(5)), [2, 5])] {
            args.device = device;
            let (out, r) = render(&DeviceSpy, &args, &curves, &clip, &LayerSet::new(), 2, &IoNote::default());
            r.unwrap();
            assert_eq!([out[0], out[1], out[8], out[9]], [want[0], want[1], want[0], want[1]], "{device:?}");
        }
    }

    #[test]
    fn run_settles_the_device_before_any_file_and_under_check() {
        let mut a = ApplyArgs::new("selftest-shift", 2, 1, Depth::U8);
        a.fps = Some(crate::cli::FpsArg { exact: Some(fps("24")), value: 24.0 });
        a.input = Some(PathBuf::from("does-not-exist.raw"));
        a.output = Some(PathBuf::from("/definitely/not/writable/out.raw"));
        for check in [true, false] {
            a.check = check;
            // gpu=N past the listing is strict: unsupported with the reason, before any file
            a.device = Some(fx_core::DeviceRequest::GpuIndex(4000));
            let err = run(&Shift, &a).unwrap_err();
            assert_eq!(err.code(), 3, "check={check}");
            assert!(err.to_string().starts_with("unsupported: --device gpu=4000: "), "{err}");
            // gpu always has a device: a GPU that passes the gate or the CPU after a note
            a.device = Some(fx_core::DeviceRequest::Gpu);
            let r = run(&Shift, &a);
            if check {
                r.unwrap();
            } else {
                assert_eq!(r.unwrap_err().code(), 4, "then the missing input is the error");
            }
        }
        // the CPU passes the check; without --check the missing input is then the error
        a.device = Some(fx_core::DeviceRequest::Cpu);
        a.check = true;
        run(&Shift, &a).unwrap();
        a.check = false;
        assert_eq!(run(&Shift, &a).unwrap_err().code(), 4);
        // a usage error still comes first
        a.device = Some(fx_core::DeviceRequest::GpuIndex(4000));
        a.fps = None;
        assert_eq!(run(&Shift, &a).unwrap_err().code(), 2);
    }

    fn write_clip(dir: &TempDir, name: &str, frames: u8) -> PathBuf {
        let bytes: Vec<u8> = (0..frames).flat_map(|i| [10 * (i + 1), 0, 0, 255, 10 * (i + 1), 1, 0, 255]).collect();
        dir.file(name, &bytes)
    }

    #[test]
    fn run_reads_files_applies_layers_and_writes_the_output() {
        let _turn = one_render_at_a_time();
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
    fn run_reads_the_input_and_each_layer_at_its_own_rate() {
        let _turn = one_render_at_a_time();
        let dir = TempDir::new("rates");
        let input = write_clip(&dir, "in.raw", 12);
        let over = write_clip(&dir, "over.raw", 3);
        let out = dir.0.join("out.raw");
        let mut args = ApplyArgs::new("selftest-shift", 2, 1, Depth::U8);
        args.fps = Some(crate::cli::FpsArg { exact: Some(fps("24")), value: 24.0 });
        args.input = Some(input.clone());
        args.output = Some(out.clone());
        let firsts = |args: &ApplyArgs| {
            run(&Shift, args).unwrap();
            std::fs::read(&out).unwrap().chunks(8).map(|f| f[0]).collect::<Vec<u8>>()
        };
        // 12 frames at 48 fps last a quarter second: 6 output frames at 24, showing footage 2n
        args.clip_fps = Some(fps("48"));
        assert_eq!(firsts(&args), [10, 30, 50, 70, 90, 110]);
        // the output rate given as the clip rate changes nothing
        args.clip_fps = Some(fps("24"));
        assert_eq!(firsts(&args), [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]);
        // --frames still counts output frames; an offset counts output frames too
        args.clip_fps = Some(fps("48"));
        args.frames = Some(8);
        args.params = vec![("offset".into(), "1".into())];
        assert_eq!(firsts(&args), [30, 50, 70, 90, 110, 0, 0, 0]);
        // a 12 fps overlay of 3 frames: output frame n mixes overlay frame n / 2
        args.params.clear();
        args.frames = None;
        args.layers = vec![("overlay".into(), over.clone())];
        args.layer_fps = vec![("overlay".into(), fps("12"))];
        let overlay = [10u16, 10, 20, 20, 30, 30];
        let want: Vec<u8> = [10u16, 30, 50, 70, 90, 110].iter().zip(overlay).map(|(a, b)| ((a + b + 1) >> 1) as u8).collect();
        assert_eq!(firsts(&args), want);
        // without --layer-fps the layer stays at the output rate, whatever --clip-fps says
        args.layer_fps.clear();
        let want: Vec<u8> = [10u16, 30, 50, 70, 90, 110].iter().zip([10u16, 20, 30, 0, 0, 0]).map(|(a, b)| ((a + b + 1) >> 1) as u8).collect();
        assert_eq!(firsts(&args), want);
    }

    #[test]
    fn layer_rates_must_name_a_listed_layer_that_is_given() {
        let mut a = ApplyArgs::new("selftest-shift", 2, 1, Depth::U8);
        a.fps = Some(crate::cli::FpsArg { exact: Some(fps("24")), value: 24.0 });
        a.check = true;
        a.layer_fps = vec![("overlay".into(), fps("25"))];
        let err = run(&Shift, &a).unwrap_err();
        assert_eq!(err.code(), 2);
        assert_eq!(err.to_string(), "--layer-fps overlay=25 names a layer that no --layer overlay=FILE gives");
        a.layers = vec![("overlay".into(), PathBuf::from("o.raw"))];
        run(&Shift, &a).unwrap();
        a.layer_fps = vec![("matte".into(), fps("30000/1001"))];
        let err = run(&Shift, &a).unwrap_err();
        assert_eq!(err.code(), 3);
        assert_eq!(err.to_string(), "unknown layer `matte` in --layer-fps: selftest-shift has the layers overlay");
        a.layers.clear();
        let err = check_layer_rates(&Integrate, &a).unwrap_err();
        assert_eq!(err.to_string(), "unknown layer `matte` in --layer-fps: selftest-integrate has no layers");
        a.layer_fps.clear();
        a.clip_fps = Some(fps("30000/1001"));
        run(&Shift, &a).unwrap();
    }

    #[test]
    fn run_reports_each_kind_of_mistake_with_its_exit_code() {
        let _turn = one_render_at_a_time();
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

    // test gaps no other test records
    static X1: fx_core::OpenGap = fx_core::OpenGap::new("X1", "first open point", &["a"]);
    static X2: fx_core::OpenGap = fx_core::OpenGap::new("X2", "second open point", &["a"]);
    static X3: fx_core::OpenGap = fx_core::OpenGap::new("X3", "third open point", &["a"]);
    static X4: fx_core::OpenGap = fx_core::OpenGap::new("X4", "stale open point", &["a"]);

    #[test]
    fn a_note_line_names_the_effect_the_gap_and_the_readme_section() {
        assert_eq!(
            note_line("retime-fgbg", &X1),
            "vfx: note: retime-fgbg: X1 first open point: a stated rule that no Twixtor reference render has graded yet (README: parity gaps)\n"
        );
    }

    #[test]
    fn notes_print_once_per_gap_and_a_failed_write_is_ignored() {
        let mut noted = std::collections::BTreeSet::new();
        let mut sink = Vec::new();
        print_notes("fx", &[&X1, &X2], &mut noted, &mut sink);
        print_notes("fx", &[&X1], &mut noted, &mut sink);
        print_notes("fx", &[], &mut noted, &mut sink);
        print_notes("fx", &[&X2, &X3], &mut noted, &mut sink);
        let want = [note_line("fx", &X1), note_line("fx", &X2), note_line("fx", &X3)].concat();
        assert_eq!(String::from_utf8(sink).unwrap(), want);

        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("stderr is gone"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Err(io::Error::other("stderr is gone"))
            }
        }
        let mut noted = std::collections::BTreeSet::new();
        print_notes("fx", &[&X1], &mut noted, &mut Broken);
        assert!(noted.contains("X1"));
    }

    #[test]
    fn one_note_per_gap_per_run_after_the_frame_that_used_it() {
        struct Noter;
        impl Effect for Noter {
            fn id(&self) -> &'static str {
                "noter"
            }
            fn name(&self) -> &'static str {
                "Noter"
            }
            fn params(&self) -> Vec<fx_core::ParamSpec> {
                Vec::new()
            }
            fn kind(&self) -> fx_core::EffectKind {
                fx_core::EffectKind::Temporal
            }
            fn render_temporal(&self, cx: &TemporalContext, _: &Env<'_>, out: &mut Frame) -> fx_core::Result<()> {
                use rayon::prelude::*;
                let n = cx.frame_index;
                (0..64).into_par_iter().for_each(|i| fx_core::gaps::record(if n == 0 || i % 2 == 0 { &X1 } else { &X2 }));
                if n == 3 {
                    fx_core::gaps::record(&X3);
                    return Err(fx_core::FxError::Unsupported("frame three".into()));
                }
                out.pixels = Pixels::U8(vec![[n as u8; 4]; 2]);
                Ok(())
            }
        }
        let _turn = one_render_at_a_time();
        let (args, time, clip) = setup();
        let curves = ParamCurves::new(Vec::new());
        let run = |frames: u64| {
            let (mut out, mut notes) = (Vec::new(), Vec::new());
            // a record from before the run is not this run's
            fx_core::gaps::record(&X4);
            let r = render_all(&Noter, &args, time, &curves, &clip, &LayerSet::new(), frames, &IoNote::default(), &mut out, &mut notes);
            // only this test records X1 to X4; a parallel test of the crate may record others
            let text = String::from_utf8(notes).unwrap();
            let lines: Vec<String> = text.lines().filter(|l| l.starts_with("vfx: note: noter: X")).map(|l| format!("{l}\n")).collect();
            (out, r, lines)
        };
        let (out, r, lines) = run(3);
        r.unwrap();
        assert_eq!(out, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2], "the notes never reach the frames");
        assert_eq!(lines, [note_line("noter", &X1), note_line("noter", &X2)]);
        // every run notes afresh, and the frame that fails notes nothing
        let (out, r, lines) = run(5);
        assert_eq!(r.unwrap_err().code(), 3, "the exit code is the failure's");
        assert_eq!(out.len(), 3 * 8);
        assert_eq!(lines, [note_line("noter", &X1), note_line("noter", &X2)]);
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
