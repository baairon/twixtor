//! Hand rolled argument parsing for `vfx`.

use crate::error::CliError;
use fx_core::{Depth, FieldOrder, Quality, Rational};
use std::path::PathBuf;

pub const USAGE: &str = "\
usage:
  vfx list [--json] [--all]
  vfx apply <effect> --width W --height H --depth 8|16|32 [--fps N[/D]]
            [--param ID=VALUE ...] [--in FILE] [--out FILE]
            [--frames N] [--time-scale TICKS] [--layer ID=FILE ...]
            [--quality draft|high] [--field none|upper|lower]
            [--tmp-dir DIR] [--check]
  vfx --help | --version

list prints the registered effects with their kind (frame or temporal), params
and, for temporal effects, their extra layers. The hidden test family selftest
is listed only with --all; its effects always run.

apply reads raw frames and writes the same byte format: rgba for 8 bpc,
rgba64le for 16, gbrapf32le for 32 (planar G B R A), straight alpha. `-` means
stdin or stdout.

A frame effect streams: frames are read until EOF (stdin, or --in) and each is
written as it is done (stdout, or --out). --fps (default 24) only sets the time
of each frame.

A temporal effect (its kind in `vfx list`) needs --fps N[/D], for example 24,
29.97 (read as the exact decimal 2997/100) or 30000/1001, because it reads
other frames. The input is a seekable clip, whole frames back to back:
--in FILE, or stdin spooled to a temporary file in --tmp-dir (default: the OS
temp dir) that is removed on every exit path. Frames are rendered in order and
written as they finish.
  --frames N         frames to render (default: the frames in the input clip;
                     frames past the clip see an empty input)
  --time-scale T     ticks per second (default: 100 ticks per frame at integer
                     rates, num/den at NTSC style rates); a frame must last a
                     whole number of ticks
  --layer ID=FILE    an extra layer the effect lists, a raw clip of the same
                     size and depth; repeatable
  --quality, --field render quality (default high) and field order (default none)

--param ID=VALUE sets a parameter. A temporal effect also takes keyframes,
VALUE = V@F;V@F with F the frame number (fractions allowed) and a trailing h
on F for a hold key, for example --param 'speed_percent=100@0;50@24h'. A
trailing e on F eases the key (zero slope there, S-shaped between two eased
keys), for example --param 'speed_percent=100@0e;50@24e'. The ; ends a
command in a shell, so quote the argument.

--check validates the effect, parameters, layer ids and rates, then exits 0
without reading or writing any file.

exit codes: 0 ok, 2 usage, 3 unknown effect or bad param or effect failure,
4 io error (including a trailing partial frame)";

#[derive(Debug, PartialEq)]
pub enum Command {
    Help,
    Version,
    List { json: bool, all: bool },
    Apply(ApplyArgs),
}

/// A `--fps` value. `exact` is set when the text is a rational rate; a plain
/// decimal with more digits than a rate can carry only gives `value`, which a
/// frame effect accepts and a temporal one refuses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FpsArg {
    pub exact: Option<Rational>,
    pub value: f64,
}

#[derive(Debug, PartialEq)]
pub struct ApplyArgs {
    pub effect: String,
    pub width: usize,
    pub height: usize,
    pub depth: Depth,
    pub fps: Option<FpsArg>,
    pub time_scale: Option<u32>,
    pub frames: Option<u64>,
    pub params: Vec<(String, String)>,
    pub layers: Vec<(String, PathBuf)>,
    pub quality: Option<Quality>,
    pub field: Option<FieldOrder>,
    pub tmp_dir: Option<PathBuf>,
    pub check: bool,
    pub input: Option<PathBuf>,
    pub output: Option<PathBuf>,
}

impl ApplyArgs {
    /// Everything at its default, for callers that build the arguments by hand.
    pub fn new(effect: impl Into<String>, width: usize, height: usize, depth: Depth) -> ApplyArgs {
        ApplyArgs {
            effect: effect.into(),
            width,
            height,
            depth,
            fps: None,
            time_scale: None,
            frames: None,
            params: Vec::new(),
            layers: Vec::new(),
            quality: None,
            field: None,
            tmp_dir: None,
            check: false,
            input: None,
            output: None,
        }
    }

    /// The frame rate a frame effect sees, 24 when `--fps` was not given.
    pub fn frame_rate(&self) -> f64 {
        self.fps.map_or(24.0, |f| f.value)
    }

    /// Names of the options that only mean something to a temporal effect and were given.
    pub fn temporal_only_flags(&self) -> Vec<&'static str> {
        let mut flags = Vec::new();
        for (given, name) in [
            (self.frames.is_some(), "--frames"),
            (self.time_scale.is_some(), "--time-scale"),
            (!self.layers.is_empty(), "--layer"),
            (self.quality.is_some(), "--quality"),
            (self.field.is_some(), "--field"),
            (self.tmp_dir.is_some(), "--tmp-dir"),
        ] {
            if given {
                flags.push(name);
            }
        }
        flags
    }
}

pub fn parse<I: IntoIterator<Item = String>>(args: I) -> Result<Command, CliError> {
    let mut it = args.into_iter();
    let Some(cmd) = it.next() else {
        return Err(CliError::usage("missing command, try `vfx --help`"));
    };
    match cmd.as_str() {
        "-h" | "--help" | "help" => Ok(Command::Help),
        "-V" | "--version" => Ok(Command::Version),
        "list" => parse_list(it),
        "apply" => parse_apply(it),
        other => Err(CliError::usage(format!("unknown command `{other}`, try `vfx --help`"))),
    }
}

/// Splits `--flag=value` into its parts; anything else is returned whole.
fn split_flag(arg: String) -> (String, Option<String>) {
    match arg.split_once('=') {
        Some((flag, value)) if flag.starts_with("--") => (flag.to_string(), Some(value.to_string())),
        _ => (arg, None),
    }
}

fn value(
    flag: &str,
    inline: Option<String>,
    it: &mut impl Iterator<Item = String>,
) -> Result<String, CliError> {
    inline
        .or_else(|| it.next())
        .ok_or_else(|| CliError::usage(format!("`{flag}` needs a value")))
}

fn parse_list(it: impl Iterator<Item = String>) -> Result<Command, CliError> {
    let (mut json, mut all) = (false, false);
    for arg in it {
        match arg.as_str() {
            "--json" => json = true,
            "--all" => all = true,
            "-h" | "--help" => return Ok(Command::Help),
            other => return Err(CliError::usage(format!("unexpected argument `{other}` for list"))),
        }
    }
    Ok(Command::List { json, all })
}

fn parse_size(flag: &str, text: &str) -> Result<usize, CliError> {
    match text.trim().parse::<usize>() {
        Ok(n) if n > 0 => Ok(n),
        _ => Err(CliError::usage(format!("`{flag}` expects a positive integer, got `{text}`"))),
    }
}

fn parse_fps(text: &str) -> Result<FpsArg, CliError> {
    if let Ok(r) = Rational::parse(text) {
        return Ok(FpsArg { exact: Some(r), value: r.as_f64() });
    }
    match text.trim().parse::<f64>() {
        Ok(f) if f.is_finite() && f > 0.0 => Ok(FpsArg { exact: None, value: f }),
        _ => Err(CliError::usage(format!(
            "`--fps` expects a positive rate such as 24, 29.97 or 30000/1001, got `{text}`"
        ))),
    }
}

/// `ID=VALUE` pairs for `--param` and `--layer`; `flag` names the option in errors.
fn pair<'a>(flag: &str, text: &'a str) -> Result<(&'a str, &'a str), CliError> {
    match text.split_once('=') {
        Some((key, val)) if !key.is_empty() => Ok((key, val)),
        _ => Err(CliError::usage(format!("`{flag}` expects ID=VALUE, got `{text}`"))),
    }
}

fn parse_apply(mut it: impl Iterator<Item = String>) -> Result<Command, CliError> {
    let (mut effect, mut width, mut height, mut depth) = (None, None, None, None);
    let mut a = ApplyArgs::new("", 0, 0, Depth::U8);

    while let Some(arg) = it.next() {
        let (flag, inline) = split_flag(arg);
        match flag.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--width" => width = Some(parse_size("--width", &value(&flag, inline, &mut it)?)?),
            "--height" => height = Some(parse_size("--height", &value(&flag, inline, &mut it)?)?),
            "--depth" => {
                let text = value(&flag, inline, &mut it)?;
                let bits = text.trim().parse::<u32>().ok().and_then(Depth::from_bits);
                depth = Some(bits.ok_or_else(|| {
                    CliError::usage(format!("`--depth` expects 8, 16 or 32, got `{text}`"))
                })?);
            }
            "--fps" => a.fps = Some(parse_fps(&value(&flag, inline, &mut it)?)?),
            "--frames" => a.frames = Some(parse_size("--frames", &value(&flag, inline, &mut it)?)? as u64),
            "--time-scale" => {
                let text = value(&flag, inline, &mut it)?;
                a.time_scale = match text.trim().parse::<u32>() {
                    Ok(n) if n > 0 => Some(n),
                    _ => {
                        return Err(CliError::usage(format!(
                            "`--time-scale` expects a positive number of ticks per second, got `{text}`"
                        )))
                    }
                };
            }
            "--param" => {
                let text = value(&flag, inline, &mut it)?;
                let (key, val) = pair("--param", &text)?;
                if a.params.iter().any(|(k, _)| k == key) {
                    return Err(CliError::usage(format!("param `{key}` given more than once")));
                }
                a.params.push((key.to_string(), val.to_string()));
            }
            "--layer" => {
                let text = value(&flag, inline, &mut it)?;
                let (id, file) = pair("--layer", &text)?;
                if file.is_empty() {
                    return Err(CliError::usage(format!("`--layer` expects ID=FILE, got `{text}`")));
                }
                if a.layers.iter().any(|(k, _)| k == id) {
                    return Err(CliError::usage(format!("layer `{id}` given more than once")));
                }
                a.layers.push((id.to_string(), PathBuf::from(file)));
            }
            "--quality" => {
                let text = value(&flag, inline, &mut it)?;
                a.quality = Some(match text.trim() {
                    "draft" => Quality::Draft,
                    "high" => Quality::High,
                    _ => return Err(CliError::usage(format!("`--quality` expects draft or high, got `{text}`"))),
                });
            }
            "--field" => {
                let text = value(&flag, inline, &mut it)?;
                a.field = Some(match text.trim() {
                    "none" => FieldOrder::Progressive,
                    "upper" => FieldOrder::UpperFirst,
                    "lower" => FieldOrder::LowerFirst,
                    _ => {
                        return Err(CliError::usage(format!(
                            "`--field` expects none, upper or lower, got `{text}`"
                        )))
                    }
                });
            }
            "--tmp-dir" => a.tmp_dir = Some(PathBuf::from(value(&flag, inline, &mut it)?)),
            "--check" => a.check = true,
            "--in" => a.input = Some(PathBuf::from(value(&flag, inline, &mut it)?)),
            "--out" => a.output = Some(PathBuf::from(value(&flag, inline, &mut it)?)),
            s if s.starts_with('-') && s != "-" => {
                return Err(CliError::usage(format!("unknown option `{s}`")));
            }
            _ => {
                if effect.is_some() {
                    return Err(CliError::usage(format!("unexpected argument `{flag}`")));
                }
                effect = Some(flag);
            }
        }
    }

    let need = |what: &str| CliError::usage(format!("missing {what}"));
    a.effect = effect.ok_or_else(|| need("effect id"))?;
    a.width = width.ok_or_else(|| need("--width"))?;
    a.height = height.ok_or_else(|| need("--height"))?;
    a.depth = depth.ok_or_else(|| need("--depth"))?;
    Ok(Command::Apply(a))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(args: &[&str]) -> Result<Command, CliError> {
        parse(args.iter().map(|s| s.to_string()))
    }

    fn code(args: &[&str]) -> i32 {
        p(args).unwrap_err().code()
    }

    fn apply(extra: &[&str]) -> Result<ApplyArgs, CliError> {
        let mut line = vec!["apply", "e", "--width", "4", "--height", "4", "--depth", "8"];
        line.extend_from_slice(extra);
        match p(&line)? {
            Command::Apply(a) => Ok(a),
            other => panic!("not apply: {other:?}"),
        }
    }

    #[test]
    fn parses_a_full_apply_line() {
        let cmd = p(&[
            "apply", "retime", "--width", "320", "--height=180", "--depth", "16", "--fps", "29.97",
            "--param", "a=1", "--param=color=1,2,3", "--in", "a.raw", "--out", "-",
        ])
        .unwrap();
        let Command::Apply(a) = cmd else { panic!("not apply") };
        assert_eq!((a.width, a.height, a.depth), (320, 180, Depth::U16));
        assert_eq!(a.fps.unwrap().exact, Some(Rational { num: 2997, den: 100 }));
        assert_eq!(a.frame_rate(), 29.97);
        assert_eq!(
            a.params,
            vec![("a".to_string(), "1".to_string()), ("color".to_string(), "1,2,3".to_string())]
        );
        assert_eq!(a.input, Some(PathBuf::from("a.raw")));
        assert_eq!(a.output, Some(PathBuf::from("-")));
        assert!(a.temporal_only_flags().is_empty());
    }

    #[test]
    fn fps_is_absent_by_default_and_negative_param_values_pass_through() {
        let a = apply(&["--param", "angle=-90"]).unwrap();
        assert_eq!(a.fps, None);
        assert_eq!(a.frame_rate(), 24.0);
        assert_eq!(a.params[0].1, "-90");
    }

    #[test]
    fn fps_takes_integers_exact_decimals_and_fractions() {
        let exact = |t: &str| apply(&["--fps", t]).unwrap().fps.unwrap().exact;
        assert_eq!(exact("24"), Some(Rational { num: 24, den: 1 }));
        assert_eq!(exact("29.97"), Some(Rational { num: 2997, den: 100 }));
        assert_eq!(exact("30000/1001"), Some(Rational { num: 30000, den: 1001 }));
        assert_eq!(exact("60000/2002"), Some(Rational { num: 30000, den: 1001 }));
        // a decimal too long for a rate is still a usable number for a frame effect
        let long = apply(&["--fps", "29.97002997002997"]).unwrap().fps.unwrap();
        assert_eq!(long.exact, None);
        assert!((long.value - 29.97002997002997).abs() < 1e-12);
        for bad in ["0", "-24", "nan", "inf", "x", "24/0", ""] {
            assert_eq!(apply(&["--fps", bad]).unwrap_err().code(), 2, "{bad}");
        }
    }

    #[test]
    fn temporal_options_parse() {
        let a = apply(&[
            "--frames", "12", "--time-scale=2400", "--layer", "overlay=o.raw", "--layer=alt=b.raw",
            "--quality", "draft", "--field", "lower", "--tmp-dir", "spool", "--check",
        ])
        .unwrap();
        assert_eq!((a.frames, a.time_scale), (Some(12), Some(2400)));
        assert_eq!(
            a.layers,
            vec![("overlay".to_string(), PathBuf::from("o.raw")), ("alt".to_string(), PathBuf::from("b.raw"))]
        );
        assert_eq!((a.quality, a.field), (Some(Quality::Draft), Some(FieldOrder::LowerFirst)));
        assert_eq!(a.tmp_dir, Some(PathBuf::from("spool")));
        assert!(a.check);
        assert_eq!(a.temporal_only_flags(), ["--frames", "--time-scale", "--layer", "--quality", "--field", "--tmp-dir"]);
        assert_eq!(apply(&["--quality", "high", "--field", "upper"]).unwrap().field, Some(FieldOrder::UpperFirst));
        assert_eq!(apply(&["--field", "none"]).unwrap().field, Some(FieldOrder::Progressive));
    }

    #[test]
    fn keyframe_values_pass_through_untouched() {
        let a = apply(&["--param", "speed=100@0;50@24h", "--param", "pt=1,2@0;3,4@5"]).unwrap();
        assert_eq!(a.params[0], ("speed".to_string(), "100@0;50@24h".to_string()));
        assert_eq!(a.params[1].1, "1,2@0;3,4@5");
    }

    #[test]
    fn list_and_help() {
        assert_eq!(p(&["list"]).unwrap(), Command::List { json: false, all: false });
        assert_eq!(p(&["list", "--json"]).unwrap(), Command::List { json: true, all: false });
        assert_eq!(p(&["list", "--all", "--json"]).unwrap(), Command::List { json: true, all: true });
        assert_eq!(p(&["--help"]).unwrap(), Command::Help);
        assert_eq!(p(&["apply", "--help"]).unwrap(), Command::Help);
        assert_eq!(p(&["list", "-h"]).unwrap(), Command::Help);
        assert_eq!(p(&["--version"]).unwrap(), Command::Version);
    }

    #[test]
    fn usage_errors_exit_2() {
        assert_eq!(code(&[]), 2);
        assert_eq!(code(&["frobnicate"]), 2);
        assert_eq!(code(&["list", "--yaml"]), 2);
        let base = ["apply", "e", "--width", "4", "--height", "4", "--depth", "8"];
        assert!(p(&base).is_ok());
        for bad in [
            &["--depth", "12"][..],
            &["--width", "0"],
            &["--height", "-3"],
            &["--width", "abc"],
            &["--fps", "0"],
            &["--fps", "nan"],
            &["--param", "novalue"],
            &["--param", "=1"],
            &["--bogus"],
            &["extra"],
            &["--param", "a=1", "--param", "a=2"],
            &["--in"],
            &["--frames", "0"],
            &["--frames", "x"],
            &["--time-scale", "0"],
            &["--time-scale", "-5"],
            &["--layer", "novalue"],
            &["--layer", "=file"],
            &["--layer", "a="],
            &["--layer", "a=x", "--layer", "a=y"],
            &["--quality", "ultra"],
            &["--field", "both"],
            &["--tmp-dir"],
        ] {
            let args: Vec<&str> = base.iter().chain(bad.iter()).copied().collect();
            assert_eq!(code(&args), 2, "{bad:?}");
        }
        for missing in [
            &["apply", "--width", "4", "--height", "4", "--depth", "8"][..],
            &["apply", "e", "--height", "4", "--depth", "8"],
            &["apply", "e", "--width", "4", "--depth", "8"],
            &["apply", "e", "--width", "4", "--height", "4"],
        ] {
            assert_eq!(code(missing), 2, "{missing:?}");
        }
    }

    #[test]
    fn the_usage_text_is_plain_ascii() {
        assert!(USAGE.is_ascii());
    }
}
