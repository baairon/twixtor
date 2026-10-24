//! Command line for `fx-parity`.

use std::path::PathBuf;

pub const USAGE: &str = "\
usage:
  fx-parity gen --out DIR [--vfx PATH] [--effects ID,...] [--sizes WxH,...]
                [--sources NAME,...] [--depths 8,16,32] [--no-sweep]
                [--sweep-sources NAME,...|all] [--jobs N]
                [--frames N] [--fps RATE,...] [--clip-sources NAME,...]
                [--clip-sweep-sources NAME,...|all] [--max-cases-per-effect N]
  fx-parity run --testset DIR --refs DIR [--effect ID] [--json] [--brief]
                [--strict] [--vfx PATH] [--jobs N]
  fx-parity selfcheck --testset DIR [--effect ID] [--max-cases N]
                [--threads N,...] [--json] [--vfx PATH] [--jobs N]
  fx-parity checklist --testset DIR [--out FILE] [--vfx PATH]

gen renders the inputs with ffmpeg (FFMPEG, else PATH) and writes DIR/cases.json
for every effect `vfx list --json` reports (a hidden self-test effect only when
named in --effects).

A frame effect gets stills. Each param is swept one at a time over default, min,
max and interior values at the largest size, on the --sweep-sources (default
noise,alpha) at every depth. All-default, all-min and all-max run on every
source, size and depth. --no-sweep keeps only those three.

A temporal effect gets clips of --frames frames (default 12) at 16x12 and 37x23
(--sizes overrides both kinds), from the --clip-sources testsrc2, movegrad,
noisefx and alphaslide, which take the --fps rates in turn (default
24,30000/1001). It runs the same settings, plus a two key ramp for every float,
angle and point param (no ramps with --no-sweep), plus `no-layers` when it has
an optional layer; the layers it lists are given clips of another source. The
sweeps run on --clip-sweep-sources (default testsrc2,alphaslide). At most
--max-cases-per-effect cases are kept per temporal effect (default 400, 0 keeps
all): every all-default, all-min, all-max and no-layers case and one case for
each param's min and max always stay, then one per other setting, then a fixed
pseudo random fill. cases.json records the kind, frames, fps, layers and the
params as `vfx apply --param` takes them (keyframes included).

run applies each case through vfx and compares it with the frame or
clip named by the case's `reference` in the references directory. 8 and 16 bpc
must match exactly, 32 bpc within 1 ULP, clips frame by frame; a failing clip is
reported with its first failing frame. Cases without a reference are listed as
`no reference` and never count as a pass; --strict turns them into a failure.
--brief lists only the failures.

Every case that was compared also gets its mean error, the mean absolute difference over
all pixels and all 4 channels as a percentage of full scale (255 at 8 bpc, 65535 at 16 bpc,
1.0 at 32 bpc; a NaN or infinity against another value counts as a full scale). A clip adds
its worst frame. The `graded error` block after the case lines, and `graded` in the JSON
report, give per effect the cases compared, how many are bit exact, the mean of the mean
errors and the worst case, then the 10 worst cases. Cases without a reference are not graded.

checklist writes a markdown file, DIR/CHECKLIST.md unless --out names another, for a person
who renders the references with any program. For every case it lists the size, bit depth,
frames and rate, the input files, the settings by the names the interface shows (from
`vfx list --json`, defaults left out, keyframes as `100 at frame 0, 50 at frame 24`), the
extra layers and the reference file to produce. Each case has two ffmpeg commands: one turns
the raw input into a file a program can open (PNG at 8 and 16 bpc, float EXR at 32 bpc when
ffmpeg can write it, a clip as a numbered sequence), the other turns the program's export
into the raw reference file with the right -pix_fmt (rgba, rgba64le or gbrapf32le). An EXR
holds colour multiplied by alpha, so the 32 bpc export command divides it out again.

selfcheck needs no references. It runs a subset of the temporal cases (at most
--max-cases, default 24, spread over the effects) again and again through vfx:
twice with the default rayon thread count and once for each --threads value
(default 1,3, through RAYON_NUM_THREADS). Every output must equal the first one
byte for byte; any difference, or a failing run, exits 1.

vfx is found via --vfx, VFX_PATH, next to this executable, then PATH.

exit codes: 0 ok, 1 mismatch, 2 usage, 3 setup or io error";

#[derive(Debug, PartialEq)]
pub enum Command {
    Help,
    Gen(GenArgs),
    Run(RunArgs),
    Selfcheck(SelfcheckArgs),
    Checklist(ChecklistArgs),
}

#[derive(Debug, PartialEq)]
pub struct GenArgs {
    pub out: PathBuf,
    pub vfx: Option<PathBuf>,
    pub effects: Option<Vec<String>>,
    pub no_sweep: bool,
    pub sizes: Option<Vec<(usize, usize)>>,
    pub sources: Option<Vec<String>>,
    pub depths: Option<Vec<u32>>,
    pub sweep_sources: Option<Vec<String>>,
    pub jobs: Option<usize>,
    pub frames: Option<usize>,
    pub fps: Option<Vec<String>>,
    pub clip_sources: Option<Vec<String>>,
    pub clip_sweep_sources: Option<Vec<String>>,
    pub max_cases_per_effect: Option<usize>,
}

#[derive(Debug, PartialEq)]
pub struct RunArgs {
    pub testset: PathBuf,
    pub refs: PathBuf,
    pub effect: Option<String>,
    pub json: bool,
    pub brief: bool,
    pub strict: bool,
    pub vfx: Option<PathBuf>,
    pub jobs: Option<usize>,
}

#[derive(Debug, PartialEq)]
pub struct SelfcheckArgs {
    pub testset: PathBuf,
    pub effect: Option<String>,
    pub max_cases: Option<usize>,
    pub threads: Option<Vec<usize>>,
    pub json: bool,
    pub vfx: Option<PathBuf>,
    pub jobs: Option<usize>,
}

#[derive(Debug, PartialEq)]
pub struct ChecklistArgs {
    pub testset: PathBuf,
    pub out: Option<PathBuf>,
    pub vfx: Option<PathBuf>,
}

fn split_flag(arg: String) -> (String, Option<String>) {
    match arg.split_once('=') {
        Some((flag, value)) if flag.starts_with("--") => (flag.to_string(), Some(value.to_string())),
        _ => (arg, None),
    }
}

fn value(flag: &str, inline: Option<String>, it: &mut impl Iterator<Item = String>) -> Result<String, String> {
    inline.or_else(|| it.next()).ok_or_else(|| format!("`{flag}` needs a value"))
}

fn size(text: &str) -> Result<(usize, usize), String> {
    let bad = || format!("`--sizes` expects WxH pairs such as 320x180, got `{text}`");
    let (w, h) = text.split_once(['x', 'X']).ok_or_else(bad)?;
    match (w.trim().parse::<usize>(), h.trim().parse::<usize>()) {
        (Ok(w), Ok(h)) if w > 0 && h > 0 => Ok((w, h)),
        _ => Err(bad()),
    }
}

fn depth(text: &str) -> Result<u32, String> {
    match text.trim().parse::<u32>() {
        Ok(d @ (8 | 16 | 32)) => Ok(d),
        _ => Err(format!("`--depths` expects 8, 16 or 32, got `{text}`")),
    }
}

fn list<T>(text: &str, each: impl Fn(&str) -> Result<T, String>) -> Result<Vec<T>, String> {
    text.split(',').map(str::trim).filter(|s| !s.is_empty()).map(each).collect()
}

fn source(name: &str) -> Result<String, String> {
    crate::sources::find(name).map(|_| name.to_string()).ok_or_else(|| {
        let known: Vec<&str> = crate::sources::SOURCES.iter().map(|s| s.name).collect();
        format!("unknown source `{name}`, known: {}", known.join(", "))
    })
}

fn clip_source(name: &str) -> Result<String, String> {
    crate::sources::find_clip(name).map(|_| name.to_string()).ok_or_else(|| {
        let known: Vec<&str> = crate::sources::CLIP_SOURCES.iter().map(|s| s.name).collect();
        format!("unknown clip source `{name}`, known: {}", known.join(", "))
    })
}

/// A frame rate in the form `vfx --fps` and the cases file use: `24` or `30000/1001`.
fn rate(text: &str) -> Result<String, String> {
    let r = fx_core::Rational::parse(text).map_err(|_| format!("`--fps` expects rates such as 24, 29.97 or 30000/1001, got `{text}`"))?;
    Ok(if r.den == 1 { r.num.to_string() } else { format!("{}/{}", r.num, r.den) })
}

fn jobs(text: &str) -> Result<usize, String> {
    match text.trim().parse::<usize>() {
        Ok(n) if n > 0 => Ok(n),
        _ => Err(format!("`--jobs` expects a positive integer, got `{text}`")),
    }
}

fn count(flag: &str, text: &str, min: usize) -> Result<usize, String> {
    match text.trim().parse::<usize>() {
        Ok(n) if n >= min => Ok(n),
        _ => Err(format!("`{flag}` expects an integer of at least {min}, got `{text}`")),
    }
}

pub fn parse<I: IntoIterator<Item = String>>(args: I) -> Result<Command, String> {
    let mut it = args.into_iter();
    let cmd = it.next().ok_or("missing command, try `fx-parity --help`")?;
    match cmd.as_str() {
        "-h" | "--help" | "help" => Ok(Command::Help),
        "gen" => parse_gen(it),
        "run" => parse_run(it),
        "selfcheck" => parse_selfcheck(it),
        "checklist" => parse_checklist(it),
        other => Err(format!("unknown command `{other}`, try `fx-parity --help`")),
    }
}

fn parse_gen(mut it: impl Iterator<Item = String>) -> Result<Command, String> {
    let mut a = GenArgs {
        out: PathBuf::new(),
        vfx: None,
        effects: None,
        no_sweep: false,
        sizes: None,
        sources: None,
        depths: None,
        sweep_sources: None,
        jobs: None,
        frames: None,
        fps: None,
        clip_sources: None,
        clip_sweep_sources: None,
        max_cases_per_effect: None,
    };
    let mut out = None;
    while let Some(arg) = it.next() {
        let (flag, inline) = split_flag(arg);
        match flag.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--out" => out = Some(PathBuf::from(value(&flag, inline, &mut it)?)),
            "--vfx" => a.vfx = Some(PathBuf::from(value(&flag, inline, &mut it)?)),
            "--effects" => a.effects = Some(list(&value(&flag, inline, &mut it)?, |s| Ok(s.to_string()))?),
            "--no-sweep" => a.no_sweep = true,
            "--sizes" => a.sizes = Some(list(&value(&flag, inline, &mut it)?, size)?),
            "--depths" => a.depths = Some(list(&value(&flag, inline, &mut it)?, depth)?),
            "--sources" => a.sources = Some(list(&value(&flag, inline, &mut it)?, source)?),
            "--sweep-sources" => {
                let text = value(&flag, inline, &mut it)?;
                a.sweep_sources = Some(if text.trim() == "all" {
                    crate::sources::SOURCES.iter().map(|s| s.name.to_string()).collect()
                } else {
                    list(&text, source)?
                });
            }
            "--jobs" => a.jobs = Some(jobs(&value(&flag, inline, &mut it)?)?),
            "--frames" => a.frames = Some(count(&flag, &value(&flag, inline, &mut it)?, 1)?),
            "--fps" => a.fps = Some(list(&value(&flag, inline, &mut it)?, rate)?),
            "--clip-sources" => a.clip_sources = Some(list(&value(&flag, inline, &mut it)?, clip_source)?),
            "--clip-sweep-sources" => {
                let text = value(&flag, inline, &mut it)?;
                a.clip_sweep_sources = Some(if text.trim() == "all" {
                    crate::sources::CLIP_SOURCES.iter().map(|s| s.name.to_string()).collect()
                } else {
                    list(&text, clip_source)?
                });
            }
            "--max-cases-per-effect" => a.max_cases_per_effect = Some(count(&flag, &value(&flag, inline, &mut it)?, 0)?),
            other => return Err(format!("unexpected argument `{other}` for gen")),
        }
    }
    a.out = out.ok_or("gen needs --out DIR")?;
    for (name, empty) in [
        ("--sizes", a.sizes.as_ref().is_some_and(Vec::is_empty)),
        ("--depths", a.depths.as_ref().is_some_and(Vec::is_empty)),
        ("--sources", a.sources.as_ref().is_some_and(Vec::is_empty)),
        ("--sweep-sources", a.sweep_sources.as_ref().is_some_and(Vec::is_empty)),
        ("--effects", a.effects.as_ref().is_some_and(Vec::is_empty)),
        ("--fps", a.fps.as_ref().is_some_and(Vec::is_empty)),
        ("--clip-sources", a.clip_sources.as_ref().is_some_and(Vec::is_empty)),
        ("--clip-sweep-sources", a.clip_sweep_sources.as_ref().is_some_and(Vec::is_empty)),
    ] {
        if empty {
            return Err(format!("`{name}` is empty"));
        }
    }
    Ok(Command::Gen(a))
}

fn parse_run(mut it: impl Iterator<Item = String>) -> Result<Command, String> {
    let (mut testset, mut refs) = (None, None);
    let mut a = RunArgs {
        testset: PathBuf::new(),
        refs: PathBuf::new(),
        effect: None,
        json: false,
        brief: false,
        strict: false,
        vfx: None,
        jobs: None,
    };
    while let Some(arg) = it.next() {
        let (flag, inline) = split_flag(arg);
        match flag.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--testset" => testset = Some(PathBuf::from(value(&flag, inline, &mut it)?)),
            "--refs" => refs = Some(PathBuf::from(value(&flag, inline, &mut it)?)),
            "--effect" => a.effect = Some(value(&flag, inline, &mut it)?),
            "--vfx" => a.vfx = Some(PathBuf::from(value(&flag, inline, &mut it)?)),
            "--jobs" => a.jobs = Some(jobs(&value(&flag, inline, &mut it)?)?),
            "--json" => a.json = true,
            "--brief" => a.brief = true,
            "--strict" => a.strict = true,
            other => return Err(format!("unexpected argument `{other}` for run")),
        }
    }
    a.testset = testset.ok_or("run needs --testset DIR")?;
    a.refs = refs.ok_or("run needs --refs DIR")?;
    Ok(Command::Run(a))
}

fn parse_selfcheck(mut it: impl Iterator<Item = String>) -> Result<Command, String> {
    let mut testset = None;
    let mut a = SelfcheckArgs { testset: PathBuf::new(), effect: None, max_cases: None, threads: None, json: false, vfx: None, jobs: None };
    while let Some(arg) = it.next() {
        let (flag, inline) = split_flag(arg);
        match flag.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--testset" => testset = Some(PathBuf::from(value(&flag, inline, &mut it)?)),
            "--effect" => a.effect = Some(value(&flag, inline, &mut it)?),
            "--max-cases" => a.max_cases = Some(count(&flag, &value(&flag, inline, &mut it)?, 0)?),
            "--threads" => {
                let counts = list(&value(&flag, inline, &mut it)?, |s| count("--threads", s, 1))?;
                if counts.is_empty() {
                    return Err("`--threads` is empty".into());
                }
                a.threads = Some(counts);
            }
            "--vfx" => a.vfx = Some(PathBuf::from(value(&flag, inline, &mut it)?)),
            "--jobs" => a.jobs = Some(jobs(&value(&flag, inline, &mut it)?)?),
            "--json" => a.json = true,
            other => return Err(format!("unexpected argument `{other}` for selfcheck")),
        }
    }
    a.testset = testset.ok_or("selfcheck needs --testset DIR")?;
    Ok(Command::Selfcheck(a))
}

fn parse_checklist(mut it: impl Iterator<Item = String>) -> Result<Command, String> {
    let mut testset = None;
    let mut a = ChecklistArgs { testset: PathBuf::new(), out: None, vfx: None };
    while let Some(arg) = it.next() {
        let (flag, inline) = split_flag(arg);
        match flag.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--testset" => testset = Some(PathBuf::from(value(&flag, inline, &mut it)?)),
            "--out" => a.out = Some(PathBuf::from(value(&flag, inline, &mut it)?)),
            "--vfx" => a.vfx = Some(PathBuf::from(value(&flag, inline, &mut it)?)),
            other => return Err(format!("unexpected argument `{other}` for checklist")),
        }
    }
    a.testset = testset.ok_or("checklist needs --testset DIR")?;
    Ok(Command::Checklist(a))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(args: &[&str]) -> Result<Command, String> {
        parse(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn gen_line() {
        let Command::Gen(a) = p(&[
            "gen", "--out", "set", "--sizes", "2x3, 37X23", "--depths=8,32", "--sources", "noise,bars", "--jobs", "3",
            "--effects", "retime,retime-fgbg", "--no-sweep", "--sweep-sources", "bars",
        ])
        .unwrap() else {
            panic!("not gen")
        };
        assert_eq!(a.effects, Some(vec!["retime".to_string(), "retime-fgbg".to_string()]));
        assert!(a.no_sweep);
        assert_eq!(a.sweep_sources, Some(vec!["bars".to_string()]));
        let Command::Gen(all) = p(&["gen", "--out", "set", "--sweep-sources", "all"]).unwrap() else { panic!("not gen") };
        assert_eq!(all.sweep_sources.unwrap().len(), crate::sources::SOURCES.len());
        assert_eq!(a.out, PathBuf::from("set"));
        assert_eq!(a.sizes, Some(vec![(2, 3), (37, 23)]));
        assert_eq!(a.depths, Some(vec![8, 32]));
        assert_eq!(a.sources, Some(vec!["noise".to_string(), "bars".to_string()]));
        assert_eq!(a.jobs, Some(3));
        assert_eq!((a.frames, a.fps, a.clip_sources, a.clip_sweep_sources, a.max_cases_per_effect), (None, None, None, None, None));
    }

    #[test]
    fn gen_clip_flags() {
        let Command::Gen(a) = p(&[
            "gen", "--out", "set", "--frames", "5", "--fps", "24, 29.97,60000/2002", "--clip-sources", "noisefx,movegrad",
            "--clip-sweep-sources=alphaslide", "--max-cases-per-effect", "0",
        ])
        .unwrap() else {
            panic!("not gen")
        };
        assert_eq!(a.frames, Some(5));
        assert_eq!(a.fps, Some(vec!["24".to_string(), "2997/100".to_string(), "30000/1001".to_string()]));
        assert_eq!(a.clip_sources, Some(vec!["noisefx".to_string(), "movegrad".to_string()]));
        assert_eq!(a.clip_sweep_sources, Some(vec!["alphaslide".to_string()]));
        assert_eq!(a.max_cases_per_effect, Some(0));
        let Command::Gen(all) = p(&["gen", "--out", "set", "--clip-sweep-sources", "all"]).unwrap() else { panic!("not gen") };
        assert_eq!(all.clip_sweep_sources.unwrap().len(), crate::sources::CLIP_SOURCES.len());
    }

    #[test]
    fn run_line() {
        let Command::Run(a) = p(&["run", "--testset", "t", "--refs", "r", "--effect", "retime", "--json", "--brief", "--strict", "--vfx", "v.exe"]).unwrap() else {
            panic!("not run")
        };
        assert_eq!((a.testset, a.refs), (PathBuf::from("t"), PathBuf::from("r")));
        assert_eq!(a.effect.as_deref(), Some("retime"));
        assert!(a.json && a.brief && a.strict);
        assert_eq!(a.vfx, Some(PathBuf::from("v.exe")));
    }

    #[test]
    fn selfcheck_line() {
        let Command::Selfcheck(a) = p(&["selfcheck", "--testset", "t", "--effect", "e", "--max-cases", "7", "--threads", "1,2,8", "--json", "--vfx", "v.exe", "--jobs", "2"]).unwrap() else {
            panic!("not selfcheck")
        };
        assert_eq!(a.testset, PathBuf::from("t"));
        assert_eq!((a.effect.as_deref(), a.max_cases, a.threads.clone()), (Some("e"), Some(7), Some(vec![1, 2, 8])));
        assert!(a.json);
        assert_eq!((a.vfx, a.jobs), (Some(PathBuf::from("v.exe")), Some(2)));
        let Command::Selfcheck(plain) = p(&["selfcheck", "--testset", "t"]).unwrap() else { panic!("not selfcheck") };
        assert_eq!((plain.max_cases, plain.threads, plain.json), (None, None, false));
    }

    #[test]
    fn checklist_line() {
        let Command::Checklist(a) = p(&["checklist", "--testset", "t", "--out=notes/list.md", "--vfx", "v.exe"]).unwrap() else {
            panic!("not checklist")
        };
        assert_eq!(a.testset, PathBuf::from("t"));
        assert_eq!((a.out, a.vfx), (Some(PathBuf::from("notes/list.md")), Some(PathBuf::from("v.exe"))));
        let Command::Checklist(plain) = p(&["checklist", "--testset", "t"]).unwrap() else { panic!("not checklist") };
        assert_eq!((plain.out, plain.vfx), (None, None));
    }

    #[test]
    fn mistakes_are_reported() {
        for bad in [
            &[][..],
            &["wat"],
            &["gen"],
            &["gen", "--out"],
            &["gen", "--out", "d", "--sizes", "0x4"],
            &["gen", "--out", "d", "--sizes", "big"],
            &["gen", "--out", "d", "--depths", "12"],
            &["gen", "--out", "d", "--sources", "plaid"],
            &["gen", "--out", "d", "--sources", "testsrc2"],
            &["gen", "--out", "d", "--jobs", "0"],
            &["gen", "--out", "d", "--sizes", ","],
            &["gen", "--out", "d", "--effects", ","],
            &["gen", "--out", "d", "--sweep-sources", "plaid"],
            &["gen", "--out", "d", "extra"],
            &["gen", "--out", "d", "--frames", "0"],
            &["gen", "--out", "d", "--frames", "x"],
            &["gen", "--out", "d", "--fps", "0"],
            &["gen", "--out", "d", "--fps", "fast"],
            &["gen", "--out", "d", "--fps", ","],
            &["gen", "--out", "d", "--clip-sources", "noise"],
            &["gen", "--out", "d", "--clip-sources", ","],
            &["gen", "--out", "d", "--clip-sweep-sources", "gradient"],
            &["gen", "--out", "d", "--max-cases-per-effect", "-1"],
            &["run", "--testset", "t"],
            &["run", "--refs", "r"],
            &["run", "--testset", "t", "--refs", "r", "--nope"],
            &["selfcheck"],
            &["selfcheck", "--testset", "t", "--threads", "0"],
            &["selfcheck", "--testset", "t", "--threads", ","],
            &["selfcheck", "--testset", "t", "--max-cases", "many"],
            &["selfcheck", "--testset", "t", "--refs", "r"],
            &["checklist"],
            &["checklist", "--out", "f.md"],
            &["checklist", "--testset"],
            &["checklist", "--testset", "t", "--refs", "r"],
            &["checklist", "--testset", "t", "extra"],
        ] {
            assert!(p(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn help_is_not_an_error() {
        assert_eq!(p(&["--help"]).unwrap(), Command::Help);
        assert_eq!(p(&["gen", "--help"]).unwrap(), Command::Help);
        assert_eq!(p(&["run", "-h"]).unwrap(), Command::Help);
        assert_eq!(p(&["selfcheck", "--help"]).unwrap(), Command::Help);
        assert_eq!(p(&["checklist", "--help"]).unwrap(), Command::Help);
    }

    #[test]
    fn the_usage_text_is_plain_ascii() {
        assert!(USAGE.is_ascii());
        for word in ["selfcheck", "--max-cases-per-effect", "--clip-sources", "RAYON_NUM_THREADS", "checklist", "graded error", "CHECKLIST.md"] {
            assert!(USAGE.contains(word), "{word}");
        }
    }
}
