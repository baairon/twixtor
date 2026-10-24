//! `vfx apply`: run one effect over raw frames. A frame effect streams stdin to
//! stdout; a temporal effect (see `temporal`) gets random access to the clip.

use crate::cli::ApplyArgs;
use crate::error::CliError;
use crate::{registry, temporal};
use fx_core::{resolve_params, Effect, EffectKind, Frame, Params, RenderContext};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, IsTerminal, Read, Write};
use std::panic::{self, AssertUnwindSafe, PanicHookInfo};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

const IO_BUFFER: usize = 1 << 20;
/// Sanity cap on one frame so a typo in --width does not try to allocate the machine.
const MAX_FRAME_BYTES: usize = 1 << 32;

/// Outcome of trying to fill one frame buffer.
#[derive(Debug, PartialEq)]
enum Fill {
    Full,
    Eof,
    Partial(usize),
}

fn fill<R: Read + ?Sized>(reader: &mut R, buf: &mut [u8]) -> io::Result<Fill> {
    let mut got = 0;
    while got < buf.len() {
        match reader.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(match got {
        0 => Fill::Eof,
        n if n == buf.len() => Fill::Full,
        n => Fill::Partial(n),
    })
}

pub(crate) fn is_stdio(path: &Path) -> bool {
    path == Path::new("-")
}

fn open_input(path: Option<&Path>) -> Result<Box<dyn Read>, CliError> {
    match path.filter(|p| !is_stdio(p)) {
        Some(p) => {
            let file = File::open(p).map_err(|e| CliError::io(format!("cannot open {}", p.display()), e))?;
            Ok(Box::new(BufReader::with_capacity(IO_BUFFER, file)))
        }
        None => {
            if io::stdin().is_terminal() {
                return Err(CliError::usage("refusing to read raw frames from a terminal, pipe them in or use --in"));
            }
            Ok(Box::new(BufReader::with_capacity(IO_BUFFER, io::stdin().lock())))
        }
    }
}

pub(crate) fn open_output(path: Option<&Path>) -> Result<Box<dyn Write>, CliError> {
    match path.filter(|p| !is_stdio(p)) {
        Some(p) => {
            let file = File::create(p).map_err(|e| CliError::io(format!("cannot create {}", p.display()), e))?;
            Ok(Box::new(BufWriter::with_capacity(IO_BUFFER, file)))
        }
        None => {
            if io::stdout().is_terminal() {
                return Err(CliError::usage("refusing to write raw frames to a terminal, redirect stdout or use --out"));
            }
            Ok(Box::new(BufWriter::with_capacity(IO_BUFFER, io::stdout().lock())))
        }
    }
}

pub(crate) fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

pub(crate) fn frame_len(args: &ApplyArgs) -> Result<usize, CliError> {
    args.width
        .checked_mul(args.height)
        .and_then(|n| n.checked_mul(args.depth.bytes_per_pixel()))
        .filter(|n| *n <= MAX_FRAME_BYTES)
        .ok_or_else(|| CliError::usage(format!("a {}x{} frame is too large", args.width, args.height)))
}

/// Parses the `--param` strings against the effect's specs and fills in defaults.
/// A frame effect sees one value per frame, so keyframes (`V@F;V@F`) are refused.
fn resolve(effect: &dyn Effect, given: &[(String, String)]) -> Result<Params, CliError> {
    let specs = effect.params();
    let mut parsed = Params::default();
    for (key, text) in given {
        let spec = specs
            .iter()
            .find(|s| s.id == key)
            .ok_or_else(|| fx_core::FxError::UnknownParam(key.clone()))?;
        if text.contains('@') {
            return Err(CliError::Effect(format!(
                "bad value for `{key}`: keyframes need a temporal effect, and {} is a frame effect",
                effect.id()
            )));
        }
        parsed.set(key, spec.parse(text)?);
    }
    Ok(resolve_params(effect, &parsed)?)
}

static RENDERING: AtomicBool = AtomicBool::new(false);
static PANIC_NOTE: Mutex<String> = Mutex::new(String::new());

fn panic_line(info: &PanicHookInfo<'_>) -> String {
    let payload = info.payload();
    let msg = payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".into());
    let at = info.location().map(|l| format!(" ({}:{})", l.file(), l.line())).unwrap_or_default();
    format!("{msg}{at}")
}

/// Keeps panics to one stderr line. While an effect renders the message is held
/// back so it can ride along in the effect error instead of printing twice.
pub fn install_panic_hook() {
    panic::set_hook(Box::new(|info| {
        let line = panic_line(info);
        if RENDERING.load(Ordering::SeqCst) {
            *PANIC_NOTE.lock().unwrap_or_else(|p| p.into_inner()) = line;
        } else {
            eprintln!("vfx: error: panic: {}", line.split_whitespace().collect::<Vec<_>>().join(" "));
        }
    }));
}

/// Runs one render call, turning an effect panic into an ordinary effect error
/// and naming the frame in every effect error.
pub(crate) fn guarded<T>(index: u64, call: impl FnOnce() -> fx_core::Result<T>) -> Result<T, CliError> {
    RENDERING.store(true, Ordering::SeqCst);
    let result = panic::catch_unwind(AssertUnwindSafe(call));
    RENDERING.store(false, Ordering::SeqCst);
    match result {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(e)) => Err(CliError::Effect(format!("frame {index}: {e}"))),
        Err(_) => {
            let note = std::mem::take(&mut *PANIC_NOTE.lock().unwrap_or_else(|p| p.into_inner()));
            Err(CliError::Effect(format!("frame {index}: effect panicked: {note}")))
        }
    }
}

/// Renders one frame of a frame effect.
fn render_one(
    effect: &dyn Effect,
    ctx: &RenderContext,
    params: &Params,
    input: &Frame,
    output: &mut Frame,
) -> Result<(), CliError> {
    guarded(ctx.frame_index, || effect.render(ctx, params, input, output))
}

/// Runs the effect over every frame of `input`, returning the frame count.
pub fn stream(
    effect: &dyn Effect,
    args: &ApplyArgs,
    params: &Params,
    input: &mut dyn Read,
    output: &mut dyn Write,
) -> Result<u64, CliError> {
    let len = frame_len(args)?;
    let mut buf = vec![0u8; len];
    let mut index = 0u64;
    loop {
        match fill(input, &mut buf).map_err(|e| CliError::io("read error", e))? {
            Fill::Eof => break,
            Fill::Partial(n) => {
                return Err(CliError::Io(format!(
                    "input ends with a partial frame ({n} of {len} bytes) after {index} whole frames"
                )))
            }
            Fill::Full => {}
        }
        let frame = Frame::from_bytes(args.width, args.height, args.depth, &buf);
        let mut out = Frame::new(args.width, args.height, args.depth);
        // `..Default::default()` keeps this building when fx-core adds a context field
        #[allow(clippy::needless_update)]
        let fps = args.frame_rate();
        let ctx = RenderContext { frame_index: index, time: index as f64 / fps, fps, ..Default::default() };
        render_one(effect, &ctx, params, &frame, &mut out)?;
        if (out.width, out.height, out.depth()) != (args.width, args.height, args.depth) {
            return Err(CliError::Effect(format!("frame {index}: effect changed the frame size or depth")));
        }
        output.write_all(&out.to_bytes()).map_err(|e| CliError::io("write error", e))?;
        index += 1;
    }
    output.flush().map_err(|e| CliError::io("write error", e))?;
    Ok(index)
}

pub fn run(args: ApplyArgs) -> Result<(), CliError> {
    let entry = registry::find(&args.effect).ok_or_else(|| {
        CliError::Effect(format!("unknown effect `{}`, `vfx list` shows the registered ones", args.effect))
    })?;
    match entry.effect.kind() {
        EffectKind::Temporal => temporal::run(&*entry.effect, &args),
        EffectKind::Frame => run_frame(&*entry.effect, &args),
    }
}

fn run_frame(effect: &dyn Effect, args: &ApplyArgs) -> Result<(), CliError> {
    let temporal_only = args.temporal_only_flags();
    if !temporal_only.is_empty() {
        return Err(CliError::usage(format!(
            "{} is a frame effect, {} only apply to temporal effects",
            effect.id(),
            temporal_only.join(", ")
        )));
    }
    let params = resolve(effect, &args.params)?;
    frame_len(args)?;
    if args.check {
        return Ok(());
    }

    let (input, output) = (args.input.as_deref(), args.output.as_deref());
    if let (Some(i), Some(o)) = (input, output) {
        if !is_stdio(i) && !is_stdio(o) && same_file(i, o) {
            return Err(CliError::usage("--in and --out are the same file"));
        }
    }
    let mut reader = open_input(input)?;
    let mut writer = open_output(output)?;
    stream(effect, args, &params, &mut reader, &mut writer)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::FpsArg;
    use fx_core::{Depth, FxError, ParamSpec, Rational, Result as FxResult};
    use std::path::PathBuf;

    struct Invert;

    impl Effect for Invert {
        fn id(&self) -> &'static str {
            "invert"
        }
        fn name(&self) -> &'static str {
            "invert"
        }
        fn params(&self) -> Vec<ParamSpec> {
            Vec::new()
        }
        fn render(&self, ctx: &RenderContext, _p: &Params, input: &Frame, output: &mut Frame) -> FxResult<()> {
            if ctx.frame_index == 3 {
                return Err(FxError::Unsupported("frame three".into()));
            }
            output.clone_from(input);
            if let fx_core::Pixels::U8(px) = &mut output.pixels {
                px.iter_mut().for_each(|p| p[..3].iter_mut().for_each(|c| *c = 255 - *c));
            }
            Ok(())
        }
    }

    fn args(w: usize, h: usize) -> ApplyArgs {
        ApplyArgs::new("invert", w, h, Depth::U8)
    }

    /// A reader that hands out at most `step` bytes per call.
    struct Trickle<'a>(&'a [u8], usize);

    impl Read for Trickle<'_> {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            let n = self.0.len().min(self.1).min(out.len());
            out[..n].copy_from_slice(&self.0[..n]);
            self.0 = &self.0[n..];
            Ok(n)
        }
    }

    #[test]
    fn fill_reports_eof_partial_and_full() {
        let mut buf = [0u8; 4];
        assert_eq!(fill(&mut Trickle(&[], 3), &mut buf).unwrap(), Fill::Eof);
        assert_eq!(fill(&mut Trickle(&[1, 2], 3), &mut buf).unwrap(), Fill::Partial(2));
        assert_eq!(fill(&mut Trickle(&[1, 2, 3, 4, 5], 3), &mut buf).unwrap(), Fill::Full);
    }

    #[test]
    fn streams_frames_even_when_reads_are_tiny() {
        let a = args(2, 1);
        let input: Vec<u8> = (0..16).collect();
        let mut out = Vec::new();
        let n = stream(&Invert, &a, &Params::default(), &mut Trickle(&input, 3), &mut out).unwrap();
        assert_eq!(n, 2);
        assert_eq!(out, vec![255, 254, 253, 3, 251, 250, 249, 7, 247, 246, 245, 11, 243, 242, 241, 15]);
    }

    #[test]
    fn partial_trailing_frame_is_an_io_error_after_the_whole_frames() {
        let a = args(2, 1);
        let input = vec![0u8; 8 + 3];
        let mut out = Vec::new();
        let err = stream(&Invert, &a, &Params::default(), &mut Trickle(&input, 5), &mut out).unwrap_err();
        assert_eq!(err.code(), 4);
        assert!(err.to_string().contains("partial frame (3 of 8 bytes) after 1 whole frames"), "{err}");
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn effect_errors_exit_3_and_name_the_frame() {
        let a = args(1, 1);
        let input = vec![0u8; 4 * 5];
        let err = stream(&Invert, &a, &Params::default(), &mut Trickle(&input, 100), &mut Vec::new()).unwrap_err();
        assert_eq!(err.code(), 3);
        assert!(err.to_string().starts_with("frame 3:"), "{err}");
    }

    #[test]
    fn an_effect_panic_is_an_effect_error() {
        struct Boom;
        impl Effect for Boom {
            fn id(&self) -> &'static str {
                "boom"
            }
            fn name(&self) -> &'static str {
                "boom"
            }
            fn params(&self) -> Vec<ParamSpec> {
                Vec::new()
            }
            fn render(&self, _c: &RenderContext, _p: &Params, _i: &Frame, _o: &mut Frame) -> FxResult<()> {
                panic!("boom")
            }
        }
        let err = stream(&Boom, &args(1, 1), &Params::default(), &mut Trickle(&[0u8; 4], 4), &mut Vec::new())
            .unwrap_err();
        assert_eq!(err.code(), 3);
        assert!(err.to_string().starts_with("frame 0: effect panicked"), "{err}");
    }

    #[test]
    fn frame_time_follows_fps() {
        struct Clock;
        impl Effect for Clock {
            fn id(&self) -> &'static str {
                "clock"
            }
            fn name(&self) -> &'static str {
                "clock"
            }
            fn params(&self) -> Vec<ParamSpec> {
                Vec::new()
            }
            fn render(&self, ctx: &RenderContext, _p: &Params, _i: &Frame, out: &mut Frame) -> FxResult<()> {
                if let fx_core::Pixels::U8(px) = &mut out.pixels {
                    px[0] = [ctx.frame_index as u8, (ctx.time * 1000.0).round() as u8, ctx.fps as u8, 0];
                }
                Ok(())
            }
        }
        let mut a = args(1, 1);
        a.fps = Some(FpsArg { exact: Rational::new(20, 1).ok(), value: 20.0 });
        let mut out = Vec::new();
        stream(&Clock, &a, &Params::default(), &mut Trickle(&[0u8; 12], 12), &mut out).unwrap();
        assert_eq!(out, vec![0, 0, 20, 0, 1, 50, 20, 0, 2, 100, 20, 0]);
    }

    #[test]
    fn a_frame_effect_refuses_keyframes_and_temporal_options() {
        use crate::selftest::Invert as Real;
        let err = resolve(&Real, &[("invert_alpha".into(), "1@0;0@5".into())]).unwrap_err();
        assert_eq!(err.code(), 3);
        assert!(err.to_string().contains("keyframes need a temporal effect") && err.to_string().contains("selftest-invert"), "{err}");
        assert!(resolve(&Real, &[("invert_alpha".into(), "1".into())]).is_ok());

        let mut a = ApplyArgs::new("selftest-invert", 2, 2, Depth::U8);
        a.frames = Some(3);
        a.layers = vec![("overlay".into(), PathBuf::from("x.raw"))];
        let err = run_frame(&Real, &a).unwrap_err();
        assert_eq!(err.code(), 2);
        assert_eq!(err.to_string(), "selftest-invert is a frame effect, --frames, --layer only apply to temporal effects");
    }

    #[test]
    fn check_reads_and_writes_nothing_for_a_frame_effect() {
        use crate::selftest::Invert as Real;
        let mut a = ApplyArgs::new("selftest-invert", 2, 2, Depth::U8);
        a.check = true;
        a.input = Some(PathBuf::from("does-not-exist.raw"));
        a.output = Some(PathBuf::from("/definitely/not/writable/out.raw"));
        a.params = vec![("invert_alpha".into(), "1".into())];
        run_frame(&Real, &a).unwrap();
        a.params = vec![("nope".into(), "1".into())];
        assert_eq!(run_frame(&Real, &a).unwrap_err().code(), 3);
    }

    #[test]
    fn oversized_frames_are_a_usage_error() {
        let mut a = args(usize::MAX / 2, 4);
        assert_eq!(frame_len(&a).unwrap_err().code(), 2);
        a.width = 1 << 20;
        a.height = 1 << 20;
        assert_eq!(frame_len(&a).unwrap_err().code(), 2);
    }
}
