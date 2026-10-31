//! `fx-parity checklist`: a markdown list of every case of a test set. It holds what a person
//! needs to render the references in any program (size, depth, rate, input files, settings by
//! the names the interface shows, extra layers) and two ffmpeg commands per case: one that
//! turns the raw input into a file the program can open, one that turns the program's export
//! into the raw reference file `fx-parity run` reads.

use crate::cases::{Case, TestSet};
use crate::effects::{EffectInfo, EffectKind, Kind};
use crate::families;
use crate::run::{self, quote};
use crate::sources;
use crate::testgen;
use crate::vfx::Vfx;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The file name used when `--out` is not given.
pub const DEFAULT_NAME: &str = "CHECKLIST.md";
/// Where the export of the program is expected, relative to the test set directory.
pub const EXPORT_DIR: &str = "export";
/// Where the reference files go, relative to the test set directory.
pub const REFS_DIR: &str = "refs";
/// The ffmpeg filter that divides colour by alpha again, for the 32 bpc export.
const UNPREMULTIPLY: &str = "unpremultiply=inplace=1";

/// Whether the ffmpeg that is used can write 32 bit float EXR files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exr {
    Yes,
    No,
    /// ffmpeg could not be run, so nothing is known.
    Unknown,
}

/// True when an `ffmpeg -encoders` listing has a video encoder of that name.
pub fn has_encoder(listing: &str, name: &str) -> bool {
    listing.lines().any(|line| {
        let mut words = line.split_whitespace();
        matches!((words.next(), words.next()), (Some(flags), Some(n)) if flags.starts_with('V') && n == name)
    })
}

/// Asks the given ffmpeg whether it has the `exr` encoder.
pub fn detect_exr(ffmpeg: &Path) -> Exr {
    let out = Command::new(ffmpeg).args(["-hide_banner", "-v", "quiet", "-encoders"]).stdin(Stdio::null()).output();
    match out {
        Ok(o) if o.status.success() => {
            if has_encoder(&String::from_utf8_lossy(&o.stdout), "exr") {
                Exr::Yes
            } else {
                Exr::No
            }
        }
        _ => Exr::Unknown,
    }
}

// ---- the two ffmpeg commands --------------------------------------------------------------

/// One conversion of a raw file into a file the program can open.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportStep {
    /// What is converted: `input`, or the name of the layer.
    pub what: String,
    /// The ffmpeg arguments, without the program name.
    pub args: Vec<String>,
    /// The file it writes, relative to the test set; a clip is a numbered pattern.
    pub output: String,
}

fn words<const N: usize>(list: [&str; N]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

fn slash(path: &str) -> String {
    path.replace('\\', "/")
}

/// The path without the extension of its file name.
fn stem(path: &str) -> &str {
    let name_at = path.rfind('/').map_or(0, |i| i + 1);
    match path[name_at..].rfind('.') {
        Some(dot) if dot > 0 => &path[..name_at + dot],
        _ => path,
    }
}

fn image_ext(depth: u32) -> &'static str {
    if depth == 32 {
        "exr"
    } else {
        "png"
    }
}

fn size_text(case: &Case) -> String {
    format!("{}x{}", case.width, case.height)
}

/// The ffmpeg line for a command, ready to paste.
pub fn shell_line(args: &[String]) -> String {
    let quoted: Vec<String> = args.iter().map(|a| quote(a)).collect();
    format!("ffmpeg {}", quoted.join(" "))
}

fn import_one(case: &Case, what: &str, file: &str) -> ImportStep {
    let clip = case.is_temporal();
    let file = slash(file);
    let ext = image_ext(case.depth);
    let output = if clip { format!("{}.%04d.{ext}", stem(&file)) } else { format!("{}.{ext}", stem(&file)) };
    let mut args = words(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", sources::pix_fmt(case.depth), "-s"]);
    args.push(size_text(case));
    args.extend(words(["-i"]));
    args.push(file);
    args.extend(words(["-frames:v"]));
    args.push(case.frames.max(1).to_string());
    match case.depth {
        8 => {}
        // png keeps 16 bit samples big endian; the swap loses nothing
        16 => args.extend(words(["-pix_fmt", "rgba64be"])),
        _ => args.extend(words(["-c:v", "exr", "-format", "float"])),
    }
    if clip {
        args.extend(words(["-start_number", "0"]));
    }
    args.push(output.clone());
    ImportStep { what: what.to_string(), args, output }
}

/// The conversions that make the case's input and layer files openable, the input first.
/// None when they cannot be written: a 32 bit float file needs an ffmpeg with an EXR encoder.
/// `layer_name` gives the name the interface shows for a layer id.
pub fn import_steps(case: &Case, layer_name: &dyn Fn(&str) -> String, exr: Exr) -> Option<Vec<ImportStep>> {
    if case.depth == 32 && exr == Exr::No {
        return None;
    }
    let mut steps = vec![import_one(case, "input", &case.input)];
    for (id, file) in &case.layers {
        steps.push(import_one(case, &format!("layer {}", layer_name(id)), file));
    }
    Some(steps)
}

/// Where the program's export is expected, and the ffmpeg arguments that turn it into the
/// raw reference file. A wrong sized export is not scaled, `fx-parity run` reports its size.
/// An EXR holds colour multiplied by alpha, so at 32 bpc the colour is divided by alpha again
/// to get the straight alpha frames `vfx` writes.
pub fn export_step(case: &Case) -> (String, Vec<String>) {
    let clip = case.is_temporal();
    let ext = image_ext(case.depth);
    let file = if clip { format!("{EXPORT_DIR}/{}.%04d.{ext}", case.id) } else { format!("{EXPORT_DIR}/{}.{ext}", case.id) };
    let mut args = words(["-v", "error", "-y"]);
    if clip {
        args.extend(words(["-start_number", "0"]));
    }
    args.extend(words(["-i"]));
    args.push(file.clone());
    args.extend(words(["-frames:v"]));
    args.push(case.frames.max(1).to_string());
    args.extend(words(["-an"]));
    if clip {
        args.extend(words(["-fps_mode", "passthrough"]));
    }
    if case.depth == 32 {
        args.extend(words(["-vf", UNPREMULTIPLY]));
    }
    args.extend(words(["-pix_fmt", sources::pix_fmt(case.depth), "-f", "rawvideo"]));
    args.push(reference_file(case));
    (file, args)
}

/// The raw reference file to produce, relative to the test set directory.
pub fn reference_file(case: &Case) -> String {
    format!("{REFS_DIR}/{}", slash(&case.reference))
}

// ---- settings by the names of the interface -----------------------------------------------

/// Text safe to put in a line of markdown.
fn md(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if "\\`*[]<>|".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

fn numbers(text: &str, n: usize) -> Option<Vec<f64>> {
    let v: Option<Vec<f64>> = text.split(',').map(|t| t.trim().parse::<f64>().ok()).collect();
    v.filter(|v| v.len() == n)
}

fn flag(text: &str) -> Option<bool> {
    match text.trim() {
        "1" | "true" => Some(true),
        "0" | "false" => Some(false),
        _ => None,
    }
}

fn same(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * a.abs().max(b.abs()).max(1.0)
}

/// Whether a plain value (no keyframes) is the param's default.
fn is_default(kind: &Kind, value: &str) -> bool {
    let value = value.trim();
    match kind {
        Kind::Float { default, .. } | Kind::Angle { default } => value.parse::<f64>().is_ok_and(|x| same(x, *default)),
        Kind::Int { default, .. } | Kind::Popup { default, .. } => value.parse::<i64>().ok() == Some(*default),
        Kind::Checkbox { default } => flag(value) == Some(*default),
        Kind::Color { default } => numbers(value, 3).is_some_and(|c| c.iter().zip(default).all(|(a, b)| same(*a, f64::from(*b)))),
        Kind::Point { default } => numbers(value, 2).is_some_and(|p| p.iter().zip(default).all(|(a, b)| same(*a, *b))),
        Kind::Unknown(_) => false,
    }
}

/// A value the way the interface shows it: a popup by its choice name, a checkbox as on or off.
fn show(kind: &Kind, value: &str) -> String {
    let value = value.trim();
    match kind {
        Kind::Popup { names, .. } => match value.parse::<usize>() {
            Ok(n) if n >= 1 && n <= names.len() => md(&names[n - 1]),
            _ => format!("choice number {value}"),
        },
        Kind::Checkbox { .. } => match flag(value) {
            Some(true) => "on".to_string(),
            Some(false) => "off".to_string(),
            None => value.to_string(),
        },
        Kind::Color { .. } => match numbers(value, 3) {
            Some(c) => {
                let (r, g, b) = (c[0] as u8, c[1] as u8, c[2] as u8);
                format!("#{r:02x}{g:02x}{b:02x} ({r}, {g}, {b})")
            }
            None => value.to_string(),
        },
        Kind::Point { .. } => match numbers(value, 2) {
            Some(_) => format!("({})", value.split(',').map(str::trim).collect::<Vec<_>>().join(", ")),
            None => value.to_string(),
        },
        Kind::Angle { .. } => format!("{value} degrees"),
        _ => value.to_string(),
    }
}

/// `V@F;V@F` as `V at frame F, V at frame F`; a trailing `h` on a frame is a hold key and
/// a trailing `e` an eased one. Ease handles `iS/I` and `oS/I` ([`fx_core::split_key`]) are the
/// incoming and outgoing velocity of the Keyframe Velocity dialog, `S` per second with an
/// influence of `I` percent. A key `split_key` refuses is shown as written.
fn show_keys(kind: &Kind, value: &str) -> String {
    let keys: Vec<String> = value
        .split(';')
        .map(str::trim)
        .filter(|k| !k.is_empty())
        .map(|key| match fx_core::split_key(key) {
            Ok(k) => {
                let velocity = |side: &str, h: &fx_core::HandleText<'_>| {
                    format!("{side} velocity {} per second, influence {}%", h.speed_text, h.influence_text)
                };
                let mut how: Vec<String> = Vec::new();
                how.extend(k.ease_in.as_ref().map(|h| velocity("incoming", h)));
                how.extend(k.ease_out.as_ref().map(|h| velocity("outgoing", h)));
                if k.hold {
                    how.push("hold key".into());
                } else if k.ease {
                    how.push("eased".into());
                }
                let how = if how.is_empty() { String::new() } else { format!(" ({})", how.join("; ")) };
                format!("{} at frame {}{how}", show(kind, k.value), k.frame_text)
            }
            Err(_) => key_as_written(kind, key),
        })
        .collect();
    keys.join(", ")
}

/// A key `split_key` refuses, read as before handles existed: the last character of the frame
/// alone decides between a hold and an eased key.
fn key_as_written(kind: &Kind, key: &str) -> String {
    let (v, frame) = key.rsplit_once('@').unwrap_or((key, "0"));
    let frame = frame.trim();
    let (frame, how) = match frame.chars().last() {
        Some('h') => (&frame[..frame.len() - 1], " (hold key)"),
        Some('e') => (&frame[..frame.len() - 1], " (eased)"),
        _ => (frame, ""),
    };
    format!("{} at frame {}{how}", show(kind, v), frame.trim())
}

/// One `id=value` setting of a case in words, or None when it is the param's default.
/// `effect` is None when `vfx` does not list the effect any more.
fn setting_text(effect: Option<&EffectInfo>, spec: &str) -> Option<String> {
    let (id, value) = spec.split_once('=').unwrap_or((spec, ""));
    let Some(param) = effect.and_then(|e| e.params.iter().find(|p| p.id == id)) else {
        return Some(format!("{}: {} (not known to the effect list, shown as given)", md(id), md(value)));
    };
    if value.contains('@') {
        return Some(format!("{}: {}", md(&param.name), show_keys(&param.kind, value)));
    }
    if is_default(&param.kind, value) {
        return None;
    }
    Some(format!("{}: {}", md(&param.name), show(&param.kind, value)))
}

/// A frame rate: `24`, or `30000/1001 (29.97)`.
fn rate_text(fps: &str) -> String {
    match fps.split_once('/') {
        Some((n, d)) => match (n.trim().parse::<f64>(), d.trim().parse::<f64>()) {
            (Ok(n), Ok(d)) if d != 0.0 && d != 1.0 => {
                let exact = format!("{:.3}", n / d);
                let exact = exact.trim_end_matches('0').trim_end_matches('.');
                format!("{fps} ({exact})")
            }
            _ => fps.to_string(),
        },
        None => fps.to_string(),
    }
}

fn depth_text(depth: u32) -> String {
    match depth {
        8 => "8 bpc (export as 8 bit)".to_string(),
        16 => "16 bpc (export as 16 bit)".to_string(),
        _ => "32 bpc (export as 32 bit float)".to_string(),
    }
}

fn list_names(names: &[String]) -> String {
    match names {
        [] => String::new(),
        [one] => one.clone(),
        [a, b] => format!("{a} and {b}"),
        [init @ .., last] => format!("{}, and {last}", init.join(", ")),
    }
}

// ---- the file -------------------------------------------------------------------------------

/// A section of the file: one effect and the indexes of its cases.
struct Group<'a> {
    id: &'a str,
    cases: Vec<&'a Case>,
}

fn groups(set: &TestSet) -> Vec<Group<'_>> {
    let mut out: Vec<Group<'_>> = Vec::new();
    for c in &set.cases {
        match out.iter_mut().find(|g| g.id == c.effect) {
            Some(g) => g.cases.push(c),
            None => out.push(Group { id: &c.effect, cases: vec![c] }),
        }
    }
    out
}

fn put(out: &mut String, text: impl AsRef<str>) {
    out.push_str(text.as_ref());
    out.push('\n');
}

fn preface(out: &mut String, set: &TestSet, groups: &[Group<'_>], dir: &str, exr: Exr) {
    put(out, "# Reference checklist\n");
    put(
        out,
        format!(
            "Test set: `{dir}` ({} cases, {} effects). Every path below is relative to that folder, so run the commands from there. Create the two folders the commands use first: `mkdir {REFS_DIR} {EXPORT_DIR}`.\n",
            set.cases.len(),
            groups.len()
        ),
    );
    put(out, "For each case:\n");
    put(out, "1. Run the import command or commands. They turn the raw input files into files your program can open: PNG for 8 and 16 bpc, 32 bit float EXR for 32 bpc, and a numbered sequence for a clip.");
    put(out, "2. In your program, make a project of the stated size, frame rate and bit depth. Bring in the input, put each extra layer on the layer of that name, and set the effect as listed. A setting that is not listed is at its default.");
    put(out, "3. Render with no colour management and no other effects on top. Export 8 bpc as 8 bit, 16 bpc as 16 bit and 32 bpc as 32 bit float, with an alpha channel, at exactly the stated size and frame count. The effect works on straight alpha, which is what PNG holds. Write PNG (8 and 16 bpc) or EXR (32 bpc) to the file the export command names; a clip is a sequence numbered from 0000.");
    put(out, "4. Run the export command. It converts the export into the raw reference file the case names.\n");
    put(out, format!("When the references are in `{REFS_DIR}`, check them with `fx-parity run --testset {dir} --refs {dir}/{REFS_DIR}`.\n"));
    put(out, "Notes:\n");
    put(out, "- Keyframes are written as `value at frame N`; frames count from 0 at the first frame of the clip. A hold key keeps its value until the next key, an eased key has no slope at its frame. A key with an incoming or outgoing velocity is a Bezier key: type that velocity (units per second) and its influence into the key's Keyframe Velocity dialog; a side without one is linear, or hold for a hold key.");
    put(out, "- A menu choice is given by the name the menu shows, a checkbox as on or off, a point as (x, y) in pixels, an angle in degrees, and a colour in hex and as red, green and blue from 0 to 255.");
    put(out, "- If your program names or numbers its files differently, or writes a movie, change the `-i` argument (and `-start_number`) of the export command and leave the rest.");
    if has_families(set) {
        family_notes(out);
    }
    if set.cases.iter().any(|c| c.depth == 32) {
        put(out, format!("- 32 bpc goes through EXR, which holds colour multiplied by alpha. Import the EXR files as premultiplied alpha and export EXR as premultiplied alpha; the 32 bpc export command divides by alpha again (`-vf {UNPREMULTIPLY}`) to get straight alpha. If your program writes straight alpha EXR, delete that option. Colour under zero alpha cannot be kept this way, so pixels with zero alpha can differ for that reason alone."));
        match exr {
            Exr::Yes => {}
            Exr::No => put(out, "- The ffmpeg used to write this file has no EXR encoder, so 32 bpc cases have no import command. Use an ffmpeg that has it, or bring the raw file into your program some other way."),
            Exr::Unknown => put(out, "- ffmpeg could not be run while this file was written. The 32 bpc import commands write EXR, which needs an ffmpeg built with the `exr` encoder (see `ffmpeg -encoders`)."),
        }
    }
    put(out, "");
}

/// Whether the set holds cases of the reference case families; only then does the file speak of them.
fn has_families(set: &TestSet) -> bool {
    set.cases.iter().any(|c| c.family.is_some())
}

/// The preface notes a family case needs on top of the general ones.
fn family_notes(out: &mut String) {
    put(out, "- A case with a `settles:` line belongs to a reference case family: it settles an open point of the parity gaps register (table below), a rule `vfx` runs because the original's behaviour there is not known yet. Render it like any other case.");
    put(out, "- A matte goes on a layer of its own, hidden (video switch off), and is picked in the effect's FG Matte popup of that layer (FG1 Matte, FG2 Matte, ...); the Matte Chan setting says which channel of it is read. Two FG layers may read different channels of one matte file: put it in the project once and pick it in both popups.");
    put(out, "- Point positions are in layer pixels, with (0, 0) at the centre of the top left pixel. A keyed point needs Linear spatial interpolation (Keyframe Interpolation, Spatial Interpolation: Linear), so that it moves on the straight line between its keys.");
    put(out, "- Ease handle values go into the Keyframe Velocity dialog of the key: the incoming or outgoing velocity in units per second (frames per second for Frame Num, pixels per second along the path for a point) and the influence in percent. A key written as eased is Easy Ease.");
    put(out, "- A family is settled when `fx-parity run --json` reports it with `settled: true`: every case compared and bit exact. Then its note retires: delete the gap's entry in `plugins/retime-engine/src/gaps.rs` (the register, its effect lists and its tests) and every `record` call of it, and its Stated rules row in `docs/parity.md` goes. The observing families (G7 and K1) print no note; their references decide whether the stated rule stays as it is.");
}

/// The gaps the family cases settle, one row per family that has cases here, then the clips.
fn gaps_table(out: &mut String, set: &TestSet) {
    let mut seen: Vec<(&str, usize)> = Vec::new();
    for c in &set.cases {
        let Some(f) = c.family.as_deref() else { continue };
        match seen.iter_mut().find(|(id, _)| *id == f) {
            Some((_, n)) => *n += 1,
            None => seen.push((f, 1)),
        }
    }
    put(out, "## Parity gaps these cases settle\n");
    put(out, "| family | gap | open point | rule until a reference grades it | cases |");
    put(out, "| --- | --- | --- | --- | --- |");
    for (id, n) in &seen {
        match families::find(id) {
            Some(f) => {
                let gaps: Vec<String> = f.gaps.iter().map(|g| format!("{g} {}", md(families::title(g)))).collect();
                let open = if f.observe_only { format!("{} (observe only, no note)", md(f.settles)) } else { md(f.settles) };
                put(out, format!("| `{id}` | {} | {open} | {} | {n} |", gaps.join("; "), md(f.rule)));
            }
            None => put(out, format!("| `{}` | | not known to this build of fx-parity | | {n} |", md(id))),
        }
    }
    put(out, "");
    let mut clips: Vec<&str> = Vec::new();
    for c in set.cases.iter().filter(|c| c.family.is_some()) {
        let layer_sources = c.layers.values().filter_map(|f| set.inputs.iter().find(|i| &i.file == f).map(|i| i.source.as_str()));
        for src in std::iter::once(c.source.as_str()).chain(layer_sources) {
            if sources::find_family(src).is_some() && !clips.contains(&src) {
                clips.push(src);
            }
        }
    }
    if !clips.is_empty() {
        put(out, "The family clips:\n");
        for f in sources::FAMILY_SOURCES.iter().filter(|f| clips.contains(&f.name)) {
            put(out, format!("- `{}`: {}", f.name, md(f.about)));
        }
        put(out, "");
    }
}

/// The `settles:` line of a family case: the gaps it settles, by id and title, and its family.
fn settles_line(case: &Case, family: &str) -> String {
    let gaps: Vec<String> = families::settles(case).iter().map(|g| format!("{g} ({})", md(families::title(g)))).collect();
    match families::find(family) {
        Some(f) => {
            let observe = if f.observe_only { ", observe only" } else { "" };
            format!("- settles: {} in family `{family}`{observe}", list_names(&gaps))
        }
        None => format!("- settles: family `{}`, not known to this build of fx-parity", md(family)),
    }
}

fn summary_table(out: &mut String, groups: &[Group<'_>], effects: &[EffectInfo]) {
    put(out, "| effect | id | kind | cases |");
    put(out, "| --- | --- | --- | --- |");
    for g in groups {
        let info = effects.iter().find(|e| e.id == g.id);
        let name = info.map_or_else(|| md(g.id), |e| md(&e.name));
        let kind = match info.map(|e| e.kind) {
            Some(EffectKind::Temporal) => "clip",
            Some(EffectKind::Frame) => "still",
            None if g.cases.iter().any(|c| c.is_temporal()) => "clip",
            None => "still",
        };
        put(out, format!("| {name} | `{}` | {kind} | {} |", g.id, g.cases.len()));
    }
    put(out, "");
}

fn case_section(out: &mut String, n: usize, case: &Case, info: Option<&EffectInfo>, exr: Exr) {
    put(out, format!("### {n}. `{}`\n", case.label));
    put(out, format!("- case: `{}`", case.id));
    if let Some(family) = &case.family {
        put(out, settles_line(case, family));
    }
    let length = if case.is_temporal() {
        format!("{} frames at {} fps", case.frames, rate_text(case.fps.as_deref().unwrap_or("24")))
    } else {
        "1 frame (a still)".to_string()
    };
    put(out, format!("- size and depth: {}, {}", size_text(case), depth_text(case.depth)));
    put(out, format!("- length: {length}"));
    put(out, format!("- input: `{}`", slash(&case.input)));

    let layer_name = |id: &str| info.and_then(|e| e.layers.iter().find(|l| l.id == id)).map_or_else(|| id.to_string(), |l| l.name.clone());
    if case.is_temporal() {
        if case.layers.is_empty() {
            let optional: Vec<String> = info.map(|e| e.layers.iter().filter(|l| l.optional).map(|l| md(&l.name)).collect()).unwrap_or_default();
            if optional.is_empty() {
                put(out, "- extra layers: none");
            } else {
                put(out, format!("- extra layers: none, leave {} empty", list_names(&optional)));
            }
        } else {
            put(out, "- extra layers (put each file on the layer of that name):");
            for (id, file) in &case.layers {
                put(out, format!("  - {}: `{}`", md(&layer_name(id)), slash(file)));
            }
        }
    }

    let settings: Vec<String> = case.params.iter().filter_map(|p| setting_text(info, p)).collect();
    if settings.is_empty() {
        put(out, "- settings: every setting at its default");
    } else {
        put(out, "- settings:");
        for s in &settings {
            put(out, format!("  - {s}"));
        }
    }
    put(out, format!("- reference to produce: `{}` ({} bytes)", reference_file(case), case.frames.max(1) * case.frame_len()));
    put(out, "");

    put(out, "Import:\n");
    put(out, "```sh");
    match import_steps(case, &layer_name, exr) {
        Some(steps) => {
            for step in &steps {
                put(out, shell_line(&step.args));
            }
        }
        None => put(out, "# no import command: this ffmpeg has no EXR encoder, see the notes at the top"),
    }
    put(out, "```\n");

    let (export_file, export_args) = export_step(case);
    put(out, format!("Export from your program to `{export_file}`, then convert it:\n"));
    put(out, "```sh");
    put(out, shell_line(&export_args));
    put(out, "```\n");
}

/// The checklist for a test set. `effects` is `vfx list --json --all` and gives the names
/// the interface shows; `dir` is the test set directory as it should be printed.
pub fn render(set: &TestSet, effects: &[EffectInfo], dir: &str, exr: Exr) -> String {
    let groups = groups(set);
    let mut out = String::new();
    preface(&mut out, set, &groups, dir, exr);
    summary_table(&mut out, &groups, effects);
    if has_families(set) {
        gaps_table(&mut out, set);
    }
    for g in &groups {
        let info = effects.iter().find(|e| e.id == g.id);
        put(&mut out, format!("## {} (`{}`)\n", info.map_or_else(|| md(g.id), |e| md(&e.name)), g.id));
        let _ = writeln!(out, "{} cases.\n", g.cases.len());
        for (i, case) in g.cases.iter().enumerate() {
            case_section(&mut out, i + 1, case, info, exr);
        }
    }
    out
}

// ---- the command ----------------------------------------------------------------------------

#[derive(Debug)]
pub struct ChecklistOptions {
    pub testset: PathBuf,
    /// Where to write; the test set directory's `CHECKLIST.md` when None.
    pub out: Option<PathBuf>,
    pub vfx: Option<PathBuf>,
}

#[derive(Debug)]
pub struct ChecklistSummary {
    pub path: PathBuf,
    pub effects: usize,
    pub cases: usize,
}

/// A directory as printed in the file: absolute when that is possible, with forward slashes.
fn dir_text(dir: &Path) -> String {
    let full = std::path::absolute(dir).unwrap_or_else(|_| dir.to_path_buf());
    full.display().to_string().replace('\\', "/")
}

/// Reads the test set, asks `vfx` for the names of its effects and writes the checklist.
/// `warn` gets a line for anything odd.
pub fn generate(opts: &ChecklistOptions, warn: &dyn Fn(&str)) -> Result<ChecklistSummary, String> {
    let set = run::load_testset(&opts.testset)?;
    if set.cases.is_empty() {
        return Err(format!("{} has no cases", opts.testset.join("cases.json").display()));
    }
    let vfx = Vfx::locate(opts.vfx.as_deref())?;
    let effects = vfx.list_all()?;
    let groups = groups(&set);
    for g in &groups {
        if !effects.iter().any(|e| e.id == g.id) {
            warn(&format!("vfx does not list the effect `{}`, its settings are shown by id", g.id));
        }
    }
    let exr = detect_exr(&testgen::ffmpeg_path());
    if exr == Exr::Unknown && set.cases.iter().any(|c| c.depth == 32) {
        warn("ffmpeg could not be run, the 32 bpc import commands are written without checking for an EXR encoder");
    }
    let text = render(&set, &effects, &dir_text(&opts.testset), exr);
    let path = opts.out.clone().unwrap_or_else(|| opts.testset.join(DEFAULT_NAME));
    std::fs::write(&path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    Ok(ChecklistSummary { path, effects: groups.len(), cases: set.cases.len() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::parse_list;

    /// A hand made manifest: a still of a frame effect and clips of a temporal one.
    const CASES: &str = r#"{
      "format": 1, "ffmpeg": "ffmpeg test",
      "inputs": [],
      "cases": [
        {"id": "blur-01.all-default.noise-4x3.d8.00000001", "effect": "blur-01", "label": "all-default", "params": [],
         "depth": 8, "width": 4, "height": 3, "source": "noise", "input": "inputs/noise-4x3.d8.rgba",
         "reference": "blur-01.all-default.noise-4x3.d8.00000001.rgba"},
        {"id": "blur-01.amount-max.alpha-4x3.d16.00000002", "effect": "blur-01", "label": "amount-max",
         "params": ["amount=100", "mode=3", "keep=0", "tint=255,0,128", "center=2,1.5", "angle=45", "flip=1"],
         "depth": 16, "width": 4, "height": 3, "source": "alpha", "input": "inputs/alpha-4x3.d16.rgba64le",
         "reference": "blur-01.amount-max.alpha-4x3.d16.00000002.rgba64le"},
        {"id": "blur-01.amount-i1.range-4x3.d32.00000003", "effect": "blur-01", "label": "amount-i1", "params": ["amount=25"],
         "depth": 32, "width": 4, "height": 3, "source": "range", "input": "inputs/range-4x3.d32.gbrapf32le",
         "reference": "blur-01.amount-i1.range-4x3.d32.00000003.gbrapf32le"},
        {"id": "retime-01.speed-ramp.testsrc2-4x3.d8.00000004", "effect": "retime-01", "label": "speed-ramp",
         "params": ["speed=100@0;50@24h", "display=2", "sens=70", "frame_interp=2"],
         "depth": 8, "width": 4, "height": 3, "source": "testsrc2", "input": "inputs/testsrc2-4x3.d8.f6.rgba",
         "reference": "retime-01.speed-ramp.testsrc2-4x3.d8.00000004.rgba",
         "kind": "temporal", "frames": 6, "fps": "30000/1001",
         "layers": {"alt_a": "inputs/noisefx-4x3.d8.f6.rgba", "alt_b": "inputs/movegrad-4x3.d8.f6.rgba"}},
        {"id": "retime-01.no-layers.testsrc2-4x3.d16.00000005", "effect": "retime-01", "label": "no-layers", "params": [],
         "depth": 16, "width": 4, "height": 3, "source": "testsrc2", "input": "inputs/testsrc2-4x3.d16.f6.rgba64le",
         "reference": "retime-01.no-layers.testsrc2-4x3.d16.00000005.rgba64le",
         "kind": "temporal", "frames": 6, "fps": "24"},
        {"id": "ghost.all-default.noise-4x3.d8.00000006", "effect": "ghost", "label": "all-default", "params": ["gain=3"],
         "depth": 8, "width": 4, "height": 3, "source": "noise", "input": "inputs/noise-4x3.d8.rgba",
         "reference": "ghost.rgba"}
      ]
    }"#;

    /// What `vfx list --json` prints for those effects, with the fields the harness ignores.
    const LIST: &str = r#"[
      {"id": "blur-01", "name": "Soft Blur", "family": "fam", "kind": "frame", "hidden": false, "params": [
        {"id": "amount", "name": "Amount", "kind": "int", "min": 0, "max": 100, "default": 10},
        {"id": "mode", "name": "Blur Mode", "kind": "popup", "choices": ["Box", "Gauss", "Radial"], "default": 1},
        {"id": "keep", "name": "Keep *Alpha*", "kind": "checkbox", "default": true},
        {"id": "tint", "name": "Tint", "kind": "color", "default": [255, 255, 255]},
        {"id": "center", "name": "Center", "kind": "point", "default": [2.0, 1.5]},
        {"id": "angle", "name": "Direction", "kind": "angle", "default": 0.0},
        {"id": "flip", "name": "Flip", "kind": "checkbox", "default": false}
      ]},
      {"id": "retime-01", "name": "Retime", "family": "fam", "kind": "temporal", "hidden": false,
       "layers": [{"id": "alt_a", "name": "Alt Source A", "optional": true}, {"id": "alt_b", "name": "Alt Source B", "optional": true}],
       "params": [
        {"id": "display", "name": "Display", "kind": "popup", "choices": ["Output", "Source"], "default": 1},
        {"id": "speed", "name": "Speed %", "kind": "float", "valid_min": -100.0, "valid_max": 1000.0, "slider_min": 0.0, "slider_max": 200.0, "default": 100.0},
        {"id": "sens", "name": "Sensitivity", "kind": "float", "valid_min": 0.0, "valid_max": 100.0, "slider_min": 0.0, "slider_max": 100.0, "default": 70.0},
        {"id": "frame_interp", "name": "Frame Interp", "kind": "popup", "choices": ["Nearest", "Blend", "Weighted Blend"], "default": 2}
      ]}
    ]"#;

    fn set() -> TestSet {
        serde_json::from_str(CASES).expect("the hand made manifest parses")
    }

    fn effects() -> Vec<EffectInfo> {
        parse_list(LIST).expect("the fake list parses")
    }

    fn doc(exr: Exr) -> String {
        render(&set(), &effects(), "C:/work/set", exr)
    }

    fn line(args: &[String]) -> String {
        shell_line(args)
    }

    fn case(id_part: &str) -> Case {
        set().cases.into_iter().find(|c| c.id.contains(id_part)).unwrap_or_else(|| panic!("no case {id_part}"))
    }

    #[test]
    fn the_hand_made_manifest_and_list_parse() {
        assert_eq!(set().cases.len(), 6);
        assert_eq!(effects().len(), 2);
        assert_eq!(effects()[0].params.len(), 7);
        let Kind::Popup { names, .. } = &effects()[0].params[1].kind else { panic!("a popup") };
        assert_eq!(names, &["Box", "Gauss", "Radial"]);
    }

    #[test]
    fn a_still_gets_a_png_import_and_a_raw_export_with_the_right_pixel_format() {
        let c = case("all-default.noise-4x3.d8");
        let steps = import_steps(&c, &|id| id.to_string(), Exr::Yes).unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(
            line(&steps[0].args),
            "ffmpeg -v error -y -f rawvideo -pix_fmt rgba -s 4x3 -i inputs/noise-4x3.d8.rgba -frames:v 1 inputs/noise-4x3.d8.png"
        );
        assert_eq!(steps[0].output, "inputs/noise-4x3.d8.png");
        let (file, args) = export_step(&c);
        assert_eq!(file, "export/blur-01.all-default.noise-4x3.d8.00000001.png");
        assert_eq!(
            line(&args),
            "ffmpeg -v error -y -i export/blur-01.all-default.noise-4x3.d8.00000001.png -frames:v 1 -an -pix_fmt rgba -f rawvideo refs/blur-01.all-default.noise-4x3.d8.00000001.rgba"
        );
    }

    #[test]
    fn sixteen_bit_goes_through_big_endian_png_and_comes_back_as_rgba64le() {
        let c = case("amount-max.alpha-4x3.d16");
        let steps = import_steps(&c, &|id| id.to_string(), Exr::Yes).unwrap();
        assert_eq!(
            line(&steps[0].args),
            "ffmpeg -v error -y -f rawvideo -pix_fmt rgba64le -s 4x3 -i inputs/alpha-4x3.d16.rgba64le -frames:v 1 -pix_fmt rgba64be inputs/alpha-4x3.d16.png"
        );
        let (_, args) = export_step(&c);
        assert!(line(&args).contains("-pix_fmt rgba64le -f rawvideo refs/blur-01.amount-max.alpha-4x3.d16.00000002.rgba64le"));
    }

    #[test]
    fn thirty_two_bit_is_a_float_exr_when_the_build_has_one_and_otherwise_says_so() {
        let c = case("amount-i1.range-4x3.d32");
        let steps = import_steps(&c, &|id| id.to_string(), Exr::Yes).unwrap();
        assert_eq!(
            line(&steps[0].args),
            "ffmpeg -v error -y -f rawvideo -pix_fmt gbrapf32le -s 4x3 -i inputs/range-4x3.d32.gbrapf32le -frames:v 1 -c:v exr -format float inputs/range-4x3.d32.exr"
        );
        assert!(import_steps(&c, &|id| id.to_string(), Exr::Unknown).is_some());
        assert!(import_steps(&c, &|id| id.to_string(), Exr::No).is_none());
        let (file, args) = export_step(&c);
        assert!(file.ends_with(".d32.00000003.exr"));
        assert!(line(&args).contains("-vf unpremultiply=inplace=1 -pix_fmt gbrapf32le -f rawvideo refs/"), "{}", line(&args));
        // only the float export divides by alpha
        assert!(!line(&export_step(&case("all-default.noise-4x3.d8")).1).contains("unpremultiply"));
        assert!(!line(&export_step(&case("amount-max.alpha-4x3.d16")).1).contains("unpremultiply"));
        // the other depths do not care
        assert!(import_steps(&case("all-default.noise"), &|id| id.to_string(), Exr::No).is_some());
        let no = doc(Exr::No);
        assert!(no.contains("# no import command: this ffmpeg has no EXR encoder"), "{no}");
        assert!(no.contains("The ffmpeg used to write this file has no EXR encoder"), "{no}");
        let unknown = doc(Exr::Unknown);
        assert!(unknown.contains("ffmpeg could not be run while this file was written"), "{unknown}");
        let yes = doc(Exr::Yes);
        assert!(!yes.contains("no EXR encoder") && !yes.contains("could not be run"), "{yes}");
    }

    #[test]
    fn a_clip_is_a_numbered_sequence_with_every_layer_converted_too() {
        let c = case("speed-ramp");
        let names = |id: &str| match id {
            "alt_a" => "Alt Source A".to_string(),
            other => other.to_string(),
        };
        let steps = import_steps(&c, &names, Exr::Yes).unwrap();
        assert_eq!(steps.iter().map(|s| s.what.as_str()).collect::<Vec<_>>(), ["input", "layer Alt Source A", "layer alt_b"]);
        assert_eq!(
            line(&steps[0].args),
            "ffmpeg -v error -y -f rawvideo -pix_fmt rgba -s 4x3 -i inputs/testsrc2-4x3.d8.f6.rgba -frames:v 6 -start_number 0 inputs/testsrc2-4x3.d8.f6.%04d.png"
        );
        assert_eq!(steps[1].output, "inputs/noisefx-4x3.d8.f6.%04d.png");
        let (file, args) = export_step(&c);
        assert_eq!(file, "export/retime-01.speed-ramp.testsrc2-4x3.d8.00000004.%04d.png");
        assert_eq!(
            line(&args),
            "ffmpeg -v error -y -start_number 0 -i export/retime-01.speed-ramp.testsrc2-4x3.d8.00000004.%04d.png -frames:v 6 -an -fps_mode passthrough -pix_fmt rgba -f rawvideo refs/retime-01.speed-ramp.testsrc2-4x3.d8.00000004.rgba"
        );
    }

    #[test]
    fn paths_with_spaces_or_backslashes_are_made_safe() {
        let mut c = case("all-default.noise-4x3.d8");
        c.input = "my inputs\\noise.rgba".into();
        let steps = import_steps(&c, &|id| id.to_string(), Exr::Yes).unwrap();
        let text = line(&steps[0].args);
        assert!(text.contains("-i \"my inputs/noise.rgba\""), "{text}");
        assert_eq!(steps[0].output, "my inputs/noise.png");
        assert_eq!(stem("inputs/a.b.c"), "inputs/a.b");
        assert_eq!(stem("inputs/x"), "inputs/x");
        assert_eq!(stem("dir.v2/x"), "dir.v2/x");
        assert_eq!(stem("inputs/.hidden"), "inputs/.hidden");
    }

    #[test]
    fn settings_use_the_names_of_the_interface_and_leave_out_defaults() {
        let e = effects();
        let blur = Some(&e[0]);
        let c = case("amount-max");
        let text: Vec<String> = c.params.iter().filter_map(|p| setting_text(blur, p)).collect();
        assert_eq!(
            text,
            [
                "Amount: 100",
                "Blur Mode: Radial",
                "Keep \\*Alpha\\*: off",
                "Tint: #ff0080 (255, 0, 128)",
                // the point equals its default and is left out
                "Direction: 45 degrees",
                "Flip: on",
            ]
        );
        // a value at the default is left out in every kind
        for spec in ["amount=10", "mode=1", "keep=1", "tint=255,255,255", "center=2,1.5", "angle=0", "flip=0"] {
            assert_eq!(setting_text(blur, spec), None, "{spec}");
        }
        assert_eq!(setting_text(blur, "center=0,0").as_deref(), Some("Center: (0, 0)"));
        assert_eq!(setting_text(blur, "center=80,45").as_deref(), Some("Center: (80, 45)"));
        assert_eq!(setting_text(blur, "mode=9").as_deref(), Some("Blur Mode: choice number 9"));
    }

    #[test]
    fn keyframes_are_written_as_values_at_frames() {
        let e = effects();
        let retime = Some(&e[1]);
        assert_eq!(setting_text(retime, "speed=100@0;50@24").as_deref(), Some("Speed %: 100 at frame 0, 50 at frame 24"));
        assert_eq!(setting_text(retime, "speed=100@0;50@24h").as_deref(), Some("Speed %: 100 at frame 0, 50 at frame 24 (hold key)"));
        assert_eq!(setting_text(retime, "speed=100@0e;50@12.5e").as_deref(), Some("Speed %: 100 at frame 0 (eased), 50 at frame 12.5 (eased)"));
        // keyframes are listed even when a key holds the default
        assert!(setting_text(retime, "sens=70@0;70@5").is_some());
        let blur = Some(&e[0]);
        assert_eq!(
            setting_text(blur, "center=80,45@0;240,135@11").as_deref(),
            Some("Center: (80, 45) at frame 0, (240, 135) at frame 11")
        );
        assert_eq!(setting_text(blur, "angle=-180@0;180@11").as_deref(), Some("Direction: -180 degrees at frame 0, 180 degrees at frame 11"));
        // ease handles are velocities per second with their influence, as typed, then a hold
        assert_eq!(
            setting_text(retime, "speed=100@0o0/33.3;50@24i-12.50/50h").as_deref(),
            Some("Speed %: 100 at frame 0 (outgoing velocity 0 per second, influence 33.3%), 50 at frame 24 (incoming velocity -12.50 per second, influence 50%; hold key)")
        );
        assert_eq!(
            setting_text(retime, "speed=100@12.5i5/10o-6/0.1;70@30").as_deref(),
            Some("Speed %: 100 at frame 12.5 (incoming velocity 5 per second, influence 10%; outgoing velocity -6 per second, influence 0.1%), 70 at frame 30")
        );
        assert_eq!(
            setting_text(blur, "center=80,45@0o10/50;240,135@11").as_deref(),
            Some("Center: (80, 45) at frame 0 (outgoing velocity 10 per second, influence 50%), (240, 135) at frame 11")
        );
        // a key the grammar refuses is shown as it was before handles existed
        assert_eq!(setting_text(retime, "speed=100@0i5/0;50@24i5/50e").as_deref(), Some("Speed %: 100 at frame 0i5/0, 50 at frame 24i5/50 (eased)"));
        assert_eq!(setting_text(retime, "speed=100@0;50").as_deref(), Some("Speed %: 100 at frame 0, 50 at frame 0"));
    }

    #[test]
    fn a_param_the_list_does_not_know_is_shown_by_its_id() {
        let text = setting_text(Some(&effects()[0]), "nope=3").unwrap();
        assert!(text.starts_with("nope: 3 (not known"), "{text}");
        assert!(setting_text(None, "gain=3").unwrap().starts_with("gain: 3 (not known"));
    }

    #[test]
    fn the_file_lists_every_effect_and_case_with_what_is_needed_to_render_it() {
        let md = doc(Exr::Yes);
        // the table of effects, by name
        assert!(md.contains("| Soft Blur | `blur-01` | still | 3 |"), "{md}");
        assert!(md.contains("| Retime | `retime-01` | clip | 2 |"), "{md}");
        assert!(md.contains("| ghost | `ghost` | still | 1 |"), "{md}");
        assert!(md.contains("## Soft Blur (`blur-01`)") && md.contains("## Retime (`retime-01`)"), "{md}");
        // every case, once, with its reference
        for c in set().cases {
            assert_eq!(md.matches(&format!("- case: `{}`", c.id)).count(), 1, "{}", c.id);
            assert!(md.contains(&format!("`refs/{}`", c.reference)), "{}", c.reference);
        }
        // the clip with its layers by their names and its settings in words
        let clip = md.split("### 1. `speed-ramp`").nth(1).expect("the clip case").split("### 2.").next().unwrap();
        assert!(clip.contains("- size and depth: 4x3, 8 bpc (export as 8 bit)"), "{clip}");
        assert!(clip.contains("- length: 6 frames at 30000/1001 (29.97) fps"), "{clip}");
        assert!(clip.contains("- input: `inputs/testsrc2-4x3.d8.f6.rgba`"), "{clip}");
        assert!(clip.contains("  - Alt Source A: `inputs/noisefx-4x3.d8.f6.rgba`"), "{clip}");
        assert!(clip.contains("  - Alt Source B: `inputs/movegrad-4x3.d8.f6.rgba`"), "{clip}");
        assert!(clip.contains("  - Display: Source"), "{clip}");
        assert!(clip.contains("  - Speed %: 100 at frame 0, 50 at frame 24 (hold key)"), "{clip}");
        assert!(!clip.contains("Sensitivity") && !clip.contains("Frame Interp"), "defaults are left out: {clip}");
        assert!(clip.contains("(288 bytes)"), "6 frames of 4x3 at 8 bpc: {clip}");
        // a clip without layers says which optional ones to leave empty
        let none = md.split("### 2. `no-layers`").nth(1).unwrap().split("##").next().unwrap();
        assert!(none.contains("- extra layers: none, leave Alt Source A and Alt Source B empty"), "{none}");
        assert!(none.contains("- length: 6 frames at 24 fps"), "{none}");
        assert!(none.contains("- settings: every setting at its default"), "{none}");
        // a still has no layer line, and a still of an effect the list lacks keeps its id
        let still = md.split("### 1. `all-default`").nth(1).unwrap().split("### 2.").next().unwrap();
        assert!(still.contains("- length: 1 frame (a still)") && !still.contains("extra layers"), "{still}");
        let ghost = md.split("## ghost").nth(1).unwrap();
        assert!(ghost.contains("  - gain: 3 (not known to the effect list"), "{ghost}");
        // both command lines of a case sit right under it
        assert!(md.contains("```sh\nffmpeg -v error -y -f rawvideo -pix_fmt rgba -s 4x3 -i inputs/noise-4x3.d8.rgba"), "{md}");
        assert!(md.contains("Export from your program to `export/blur-01.all-default.noise-4x3.d8.00000001.png`, then convert it:"), "{md}");
    }

    #[test]
    fn the_preface_says_how_to_render_and_where_the_files_go() {
        let md = doc(Exr::Yes);
        let head = md.split("## Soft Blur").next().unwrap();
        for want in [
            "# Reference checklist",
            "C:/work/set",
            "mkdir refs export",
            "no colour management and no other effects",
            "8 bpc as 8 bit, 16 bpc as 16 bit and 32 bpc as 32 bit float",
            "The effect works on straight alpha, which is what PNG holds",
            "32 bpc goes through EXR, which holds colour multiplied by alpha",
            "`-vf unpremultiply=inplace=1`",
            "fx-parity run --testset C:/work/set --refs C:/work/set/refs",
            "frames count from 0",
        ] {
            assert!(head.contains(want), "{want}: {head}");
        }
        assert!(head.contains("(6 cases, 3 effects)"), "{head}");
    }

    /// The hand made set with two of its clips turned into family cases.
    fn family_set() -> TestSet {
        let mut s = set();
        s.cases[3].family = Some("gap-tracker2".into());
        s.cases[3].label = "gap-tracker2-cv-track-off-1pt-vr1".into();
        s.cases[4].family = Some("gap-keys".into());
        s.cases[4].label = "gap-keys-frame-num-easy-ease".into();
        s.cases[4].source = "fgdisc".into();
        s
    }

    #[test]
    fn family_cases_get_the_gaps_table_their_settles_line_and_the_preface_notes() {
        let md = render(&family_set(), &effects(), "C:/work/set", Exr::Yes);
        let head = md.split("## Soft Blur").next().unwrap();
        for want in [
            "picked in the effect's FG Matte popup",
            "hidden (video switch off)",
            "Point positions are in layer pixels",
            "Linear spatial interpolation",
            "Keyframe Velocity dialog of the key",
            "`settled: true`",
            "`plugins/retime-engine/src/gaps.rs`",
            "every `record` call of it",
            "its Stated rules row in `docs/parity.md` goes",
            "## Parity gaps these cases settle",
            "| `gap-tracker2` | G5 second tracker arithmetic | the flow of a layer whose Track is off",
            "| `gap-keys` | K1a a Linear end next to a Bezier end; K1b Easy Ease influence;",
            "(observe only, no note)",
            "The family clips:",
            "- `fgdisc`: a textured disc moving 2 px right",
        ] {
            assert!(head.contains(want), "{want}: {head}");
        }
        assert!(head.contains("| 1 |\n| `gap-keys`") && !head.contains("gap-tiny"), "one row per family present: {head}");
        assert!(md.contains("- case: `retime-01.speed-ramp.testsrc2-4x3.d8.00000004`\n- settles: G5 (second tracker arithmetic) in family `gap-tracker2`\n"), "{md}");
        assert!(md.contains("- settles: K1b (Easy Ease influence) in family `gap-keys`, observe only\n"), "{md}");
        assert_eq!(md.matches("- settles:").count(), 2, "only the family cases");
        assert!(md.is_ascii());
        // a family this build does not know is named, not dropped
        let mut odd = family_set();
        odd.cases[3].family = Some("gap-future".into());
        let md = render(&odd, &effects(), "C:/work/set", Exr::Yes);
        assert!(md.contains("| `gap-future` | | not known to this build of fx-parity | | 1 |"), "{md}");
        assert!(md.contains("- settles: family `gap-future`, not known to this build of fx-parity"), "{md}");
    }

    #[test]
    fn a_set_without_family_cases_says_nothing_of_them() {
        for exr in [Exr::Yes, Exr::No, Exr::Unknown] {
            let md = doc(exr);
            for word in ["settles", "family", "families", "Parity gaps", "FG Matte popup", "spatial interpolation", "gaps.rs"] {
                assert!(!md.contains(word), "{word}");
            }
        }
    }

    #[test]
    fn the_text_is_plain_ascii_without_dashes() {
        for exr in [Exr::Yes, Exr::No, Exr::Unknown] {
            let text = doc(exr);
            assert!(text.is_ascii());
            assert!(!text.contains('\u{2013}') && !text.contains('\u{2014}'));
        }
    }

    #[test]
    fn markdown_specials_in_names_are_escaped() {
        assert_eq!(md("Plain Name?"), "Plain Name?");
        assert_eq!(md("A|B *c* `d` [e] <f>"), "A\\|B \\*c\\* \\`d\\` \\[e\\] \\<f\\>");
    }

    #[test]
    fn frame_rates_read_naturally() {
        assert_eq!(rate_text("24"), "24");
        assert_eq!(rate_text("30000/1001"), "30000/1001 (29.97)");
        assert_eq!(rate_text("25/1"), "25/1");
        assert_eq!(rate_text("24000/1001"), "24000/1001 (23.976)");
    }

    #[test]
    fn the_encoder_listing_is_read_by_name() {
        let listing = " Encoders:\n V....D png                  PNG (Portable Network Graphics) image\n V....D exr                  OpenEXR image\n A....D flac                 FLAC\n V....D exrx                 not it\n";
        assert!(has_encoder(listing, "exr"));
        assert!(has_encoder(listing, "png"));
        assert!(!has_encoder(listing, "tiff"));
        assert!(!has_encoder(listing, "flac"), "an audio encoder is not a video one");
        assert!(!has_encoder("", "exr"));
        assert_eq!(detect_exr(Path::new("definitely/not/ffmpeg")), Exr::Unknown);
    }

    #[test]
    fn an_empty_set_or_a_missing_manifest_is_an_error() {
        let dir = std::env::temp_dir().join(format!("fx-parity-checklist-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let opts = ChecklistOptions { testset: dir.clone(), out: None, vfx: Some(PathBuf::from("not/there")) };
        assert!(generate(&opts, &|_| {}).unwrap_err().contains("cases.json"));
        std::fs::write(dir.join("cases.json"), r#"{"format":1,"ffmpeg":"x","inputs":[],"cases":[]}"#).unwrap();
        assert!(generate(&opts, &|_| {}).unwrap_err().contains("has no cases"));
        std::fs::write(dir.join("cases.json"), CASES).unwrap();
        assert!(generate(&opts, &|_| {}).unwrap_err().contains("vfx not found"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
