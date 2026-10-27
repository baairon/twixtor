# twixtor

An independent open source reimplementation of the four effects of the Twixtor retiming plugin. It is not affiliated with or endorsed by RE:Vision Effects, which owns the Twixtor trademark.

Slow motion, speed ramps and frame interpolation from optical flow, written in Rust, on the CPU, and paired with ffmpeg. `vfx` renders raw frames in ffmpeg's own pixel formats, `tools/apply.py` runs `vfx` between an ffmpeg decode and an ffmpeg encode, `tools/speed_curve.py` and `tools/repeat_keys.py` compute keyframes without rendering, and `fx-parity` compares `vfx` with reference renders.

Sources of truth, for agents and scripts:

| Question | Ask |
|---|---|
| Effect ids, parameter ids, kinds, ranges, defaults, layer ids | `target/release/vfx list --json`. Any document that disagrees with it is wrong. |
| What a file render will do | `python tools/apply.py ... --dry-run`: the plan and every command, nothing written. |
| What a file render did | `python tools/apply.py ... --json`: the plan and the probe of the verified output. |
| Which source frame each output frame shows | `python tools/speed_curve.py`, without rendering. |

## Contents

- [Parity](#parity)
- [The effects](#the-effects)
- [Crate map](#crate-map)
- [Build](#build)
- [Interface](#interface)
  - [Renderer](#renderer)
  - [Raw frame format](#raw-frame-format)
  - [Parameter values](#parameter-values)
  - [Exit codes](#exit-codes)
  - [Wrapper](#wrapper)
  - [Helper tools](#helper-tools)
  - [Comparison harness](#comparison-harness)
- [Smoke test](#smoke-test)
- [Comparing with reference renders](#comparing-with-reference-renders)
- [Tests](#tests)
- [Limits](#limits)
- [Documentation](#documentation)
- [Legal](#legal)
- [License](#license)

## Parity

1:1 parity with Twixtor is the first priority of this project, ahead of speed, convenience and a cleaner design. When they conflict, parity wins.

- **Output.** The same settings on the same frames give the same frames: bit for bit at 8 and 16 bits per channel, within one unit in the last place at 32-bit float. The engine's precision contracts (evaluation order, single or double precision, rounding, no fused multiply add) are written in the `//!` headers of `plugins/retime-engine/src` and are part of that target.
- **Interface.** Parameter ids map one to one onto the plugin's controls, with the same display names, choices, ranges and defaults, so settings carry over in both directions.
- **No near misses.** A control or case the engine cannot reproduce stops the render with `unsupported` and names the control to change ([Limits](#limits)). It never falls back to an approximation.
- **Measured, not assumed.** `fx-parity` builds test clips and settings, writes the checklist for rendering the same cases in the original, compares every frame and grades every miss ([Comparing with reference renders](#comparing-with-reference-renders)).

## The effects

`target/release/vfx list` shows four effects, all of family `retime` and kind `temporal`; every layer is optional.

| Id | Name in `vfx list` | Reference | Params | Layers (`--layer` ids) | What it does |
|---|---|---|---|---|---|
| `retime` | Retime | [parameters](docs/parameters.md#retime) | 17 | `alt_source_a`, `alt_source_b` | Retimes a clip by a keyframed speed in percent (`time_remap_mode=1`) or by source frame number (`time_remap_mode=2`) and builds the in-between frames from optical flow: `vectors` 1 to 5 (No Vectors to Best), `frame_interp` Nearest, Blend or Weighted Blend, `smart_blend`, `blur_compensation`. |
| `create-vectors` | Create Vectors | [parameters](docs/parameters.md#create-vectors) | 59 | `alt_source_a`, `fg1_matte`, `fg2_matte`, `fg3_matte` | Does not retime: writes the motion between each frame and its previous or next one (`track_frame`) as a vector image, red horizontal and green vertical, centred and scaled by `max_displace`, blue 0, alpha opaque. |
| `retime-fgbg` | Retime (FG/BG) | [parameters](docs/parameters.md#retime-fgbg) | 63 | `alt_source_a`, `alt_source_b`, `fg1_matte`, `fg2_matte`, `fg3_matte` | The retime of `retime` with a `warping` choice (1 Inverse, 2 Inverse w/ Smart Blend, 3 Forward, 4 Forward, no Smart Blend) in place of `smart_blend`, for single layer clips ([Limits](#limits)). |
| `vectors-in` | VectorsIn | [parameters](docs/parameters.md#vectors-in) | 20 | `source`, `vectors_prev`, `vectors_next`, `fg1_matte`, `fg2_matte`, `fg3_matte` | Retimes with the vector images of `create-vectors`, given as layers, instead of searching for motion itself; `warping` 1 Inverse or 2 Forward. |

`vfx list --all` adds the hidden family `selftest`, test effects for `vfx` and `fx-parity` that `vfx apply` also runs: `selftest-invert` (the only frame effect), `selftest-shift` (layer `overlay`), `selftest-integrate` and `selftest-threads` (deliberately thread dependent, so that `fx-parity selfcheck` can be seen to fire).

## Crate map

Workspace members of `Cargo.toml`: version 0.1.0, edition 2021, MIT, unpublished. External crates: `rayon`, `serde`, `serde_json`.

| Member | Builds | Owns | Key modules (`src/`) | Tests |
|---|---|---|---|---|
| `crates/fx-core` | library | The interface every crate shares: the `Effect` trait (frame or temporal kind), frames and the raw wire format, parameter specs and command line value parsing, keyframe curves, time bases, layers. | `lib.rs` (`Effect`, `FxError`), `frame.rs`, `params.rs`, `curves.rs`, `time.rs`, `layer.rs`, `fixed.rs` (16.16 fixed point) | in-module (`#[cfg(test)]`) |
| `crates/fx-cli` | binary `vfx` | The renderer: arguments, effect registry, listing, frame and temporal runs over files or pipes. | `main.rs`, `cli.rs` (flags, usage text), `registry.rs` (`FAMILIES`: `retime` and the hidden `selftest`), `list.rs`, `apply.rs` (frame effects, `--check`, panic guard), `temporal.rs` (time base, curves, layers), `clip.rs` (random access to raw clips, stdin spool), `selftest.rs`, `error.rs` (exit codes) | in-module; `tests/cli.rs` (black box tests of the built binary) |
| `crates/fx-parity` | library `fx_parity`, binary `fx-parity` | The comparison harness: test patterns, case plans, reference checklists, comparison and grading, determinism checks. | `args.rs`, `sources.rs` (lavfi patterns), `cases.rs` (`cases.json`), `testgen.rs` (`gen`), `checklist.rs`, `run.rs`, `compare.rs`, `selfcheck.rs`, `sweep.rs`, `effects.rs`, `vfx.rs` (finds and drives `vfx`) | in-module; `tests/selftest.rs` (needs ffmpeg, skips without) |
| `plugins/retime-engine` | library | The engine of the four effects: time model, frame selection, optical flow, warps, blends, hole filling, vector image coding ([engine](docs/engine.md)). | `timeline.rs`, `planner.rs`, `frontend.rs` (parameters to settings), `host.rs`, `compose.rs` and `compose/vector.rs`, `flow.rs` (with `flow/conv.rs`, `flow/sample.rs`), `prep.rs`, `warp.rs` (with `warp/blurcomp.rs`), `forward.rs` (with `forward/mesh.rs`, `forward/pyramid.rs`, `forward/raster.rs`), `smartblend.rs`, `holefill.rs`, `blend.rs`, `filters.rs`, `vectors.rs`, `create_vectors.rs`, `image.rs` | in-module, the larger suites in their own files (`compose/tests.rs`, `forward/tests.rs`, `warp/tests.rs`, `holefill/tests.rs`, `smartblend/tests.rs`, `vectors/tests.rs`); `tests/flow_d.rs`; model fixtures and their generators in `tests/fixtures/` |
| `plugins/retime` | library | Effect `retime`: params, layers, wiring to the engine. | `lib.rs`, `glue.rs` | `tests/`: `cli` (end to end through `vfx`), `retime` (goldens of `tests/common/gen_goldens.py`), `vector`, `dependencies`, `consistency`, `settings`, `threads` |
| `plugins/create-vectors` | library | Effect `create-vectors` and what it refuses. | `lib.rs`, `glue.rs`, `prerender.rs` | `tests/`: `cli`, `encoder`, `roundtrip`, `vector`, `display`, `dependencies`, `consistency`, `settings`, `threads`, `unsupported` |
| `plugins/retime-fgbg` | library | Effect `retime-fgbg`, single layer use. | `lib.rs`, `glue.rs`, `prerender.rs`, `timemap.rs` | `tests/`: `cli`, `retime` (goldens of `tests/common/gen_goldens.py`), `forward`, `vector`, `timemap`, `dependencies`, `consistency`, `settings`, `threads`, `unsupported` |
| `plugins/vectors-in` | library | Effect `vectors-in`: vector layers decoded into flows. | `lib.rs`, `glue.rs`, `prerender.rs`, `timemap.rs`, `vectorin.rs` | `tests/`: `cli`, `vectorin`, `vector`, `retime`, `errors`, `timemap`, `dependencies`, `consistency`, `settings`, `threads`, `unsupported` |

Outside the workspace: `tools/apply.py`, `tools/speed_curve.py` and `tools/repeat_keys.py` (Python 3 standard library only; [Wrapper](#wrapper), [Helper tools](#helper-tools)) with their `unittest` suites `tools/test_*.py`, and the technical references in `docs/` ([Documentation](#documentation)).

## Build

From the repository root, with a stable Rust toolchain, 1.82 or newer (the code uses `Option::is_none_or`). The first build fetches the external crates from crates.io.

```sh
cargo build --release -p fx-cli -p fx-parity
```

| Binary | Path (`.exe` added on Windows) | Used by |
|---|---|---|
| `vfx` | `target/release/vfx` | every render; `tools/apply.py` |
| `fx-parity` | `target/release/fx-parity` | [Comparing with reference renders](#comparing-with-reference-renders) |

`CARGO_TARGET_DIR` moves both: `tools/apply.py` looks for `vfx` there, and `fx-parity` looks next to itself. The release profile is `opt-level = 3`, `codegen-units = 1`, thin LTO. `tools/apply.py` needs ffmpeg and ffprobe; `tools/repeat_keys.py` and `fx-parity gen` need ffmpeg; regenerating test fixtures needs numpy ([Tests](#tests)).

## Interface

Every contract an agent relies on: what goes in, what comes out, how failure shows.

### Renderer

`vfx` reads raw frames, runs one effect and writes raw frames of the same size and format.

| Command | Does |
|---|---|
| `vfx --help` (also `-h`, `help`) | Prints the usage text, exit 0. `--help` after `list` or anywhere in an `apply` line does the same. |
| `vfx --version` (also `-V`) | Prints `vfx 0.1.0`. |
| `vfx list` | A table with columns `id name family kind params layers`, then per effect a `<id> params` table (`param name kind range default`) and a `<id> layers` table (`layer name optional`). |
| `vfx list --json` | The same as a JSON array. Each effect: `id`, `name`, `family`, `kind` (`frame` or `temporal`), `hidden`, `params`, `layers`. Each param: `id`, `name`, `kind` and that kind's fields: `float` `valid_min`, `valid_max` (the accepted range), `slider_min`, `slider_max` (an interface range only), `default`, `precision`; `int` `min`, `max`, `default`; `popup` `choices` (names in order) and `default` (1-based); `checkbox` `default` (`true` or `false`); `color` `default` `[r, g, b]`; `point` `default` `[x, y]`; `angle` `default`. Each layer: `id`, `name`, `optional`. |
| `vfx list --all` | Adds the hidden `selftest` family (`"hidden": true`). |
| `vfx apply <effect> ...` | Runs one effect; flags below. |

`vfx apply` flags. A valued flag also takes the form `--flag=value`; an unknown flag is a usage error.

| Flag | Default | Meaning |
|---|---|---|
| `<effect>` | required | Effect id. |
| `--width W`, `--height H` | required | Frame size in pixels, positive integers. |
| `--depth 8\|16\|32` | required | Bits per channel; picks the [raw frame format](#raw-frame-format). |
| `--fps N[/D]` | 24 (frame effects) | An integer, a decimal of at most 6 places read exactly (`29.97` is 2997/100) or a fraction (`30000/1001`). Required by a temporal effect. |
| `--param ID=VALUE` | each param at its default | Sets one parameter ([Parameter values](#parameter-values)); repeatable, each id once. |
| `--in FILE` | stdin | Input clip; `-` is stdin. A terminal is refused. |
| `--out FILE` | stdout | Output; `-` is stdout. A terminal is refused, and so is the input file or a layer file. |
| `--frames N` | the input's frame count | Temporal only. Frames to render; frames past the end of the clip see an empty input. |
| `--time-scale T` | see [time base](#parameter-values) | Temporal only. Ticks per second; a frame must last a whole number of ticks. |
| `--layer ID=FILE` | none | Temporal only, repeatable. An extra layer the effect lists: a raw clip of the same size and depth, any length. |
| `--quality draft\|high` | `high` | Temporal only. `draft` halves the motion search resolution once more than `high` does, but never below an eighth of the frame size. |
| `--field none\|upper\|lower` | `none` | Temporal only. Passed to the effect, but none of the four reads it: field order comes from the `input_fields` param. |
| `--tmp-dir DIR` | the OS temp folder | Temporal only. Where piped stdin is spooled (`vfx-spool-<pid>-...raw`); removed on every exit. |
| `--check` | off | Validates the effect, params (keyframes included), layer ids, rates and frame size, then exits 0 without reading or writing a file. It does not render, so `unsupported` refusals ([Limits](#limits)) appear only in a real run. |

**Kinds.** A *frame* effect turns each input frame into one output frame as it streams; `--fps` only sets each frame's time, and the temporal-only flags are a usage error. A *temporal* effect (all four listed effects) reads any frame of the clip and its layers and samples params over time: it needs an exact `--fps`; `--in` must be a file, or piped stdin is first copied whole to a spool file in `--tmp-dir`; frames render in order and each is written as soon as it is done.

### Raw frame format

These are ffmpeg's own raw formats, so `ffmpeg -f rawvideo -pix_fmt <fmt>` reads and writes them directly ([ffmpeg pairing](docs/ffmpeg.md#raw-frames)).

| `--depth` | ffmpeg `-pix_fmt` | Bytes a pixel | Layout | Values |
|---|---|---|---|---|
| 8 | `rgba` | 4 | packed R, G, B, A, one byte each | 0 to 255 |
| 16 | `rgba64le` | 8 | packed R, G, B, A, unsigned 16-bit little endian | 0 to 65535 |
| 32 | `gbrapf32le` | 16 | planar: the whole G plane, then B, then R, then A; 32-bit IEEE float little endian | 1.0 is full |

- A frame is W x H pixels in rows from the top, left to right, with no padding or header: W x H x bytes a pixel. One frame may be at most 4 GiB (2^32 bytes); a larger size is a usage error.
- A clip is whole frames back to back. A file that ends in part of a frame fails with exit 4 (`... ends with a partial frame (N of M bytes) after K whole frames`). An empty input is zero frames and gives an empty output; an empty layer is an error.
- Alpha is straight (not premultiplied) at every depth.
- A frame shown unchanged (100% speed) is not always returned byte for byte. The effects work on premultiplied colour, so colour under partial alpha loses precision, and under alpha 0 it is lost (measured at 8 and 32 bpc; alpha itself came back unchanged). At 16 bpc they also work on 0 to 32768, so samples can come back 1 off in 65535 (an opaque 160x96 clip: 54353 of 245760 samples, none more than 1). Opaque frames come back bit for bit at 8 and 32 bpc.

### Parameter values

`--param ID=VALUE` on `vfx apply`, and `--param K=V` on `tools/apply.py`, which passes it through unchanged. `vfx apply ... --check` tests a value without rendering. What each parameter does: [parameters](docs/parameters.md).

| Kind (in `vfx list`) | VALUE | Accepted |
|---|---|---|
| `float` | a decimal: `50`, `-12.5` | `valid_min` to `valid_max` inclusive (the `range` column) |
| `int` | an integer | `min` to `max` |
| `popup` | the 1-based number of the choice (`vectors=5` is Best); names are refused | 1 to the number of choices |
| `checkbox` | `1` or `0` (`true`/`false` and `on`/`off` also work) | |
| `color` | `r,g,b`, whole numbers 0 to 255 | |
| `point` | `x,y` in pixels | |
| `angle` | degrees | any number |

The four effects use `float`, `popup`, `checkbox`, `color` and `point`. A bad value exits 3 before a frame renders:

| Value | `vfx: error: ...` |
|---|---|
| an unknown id | ``unknown param `nope` `` |
| a number outside its range | ``bad value for `speed_percent`: 10001 outside -10000..=10000`` |
| a popup given by name | ``bad value for `vectors`: cannot parse `Best` `` |
| a key without `@frame` (`100@0;50`) | ``bad value for `speed_percent`: `50` has no @frame`` |
| a key both held and eased (`100@0he`) | ``bad value for `speed_percent`: `0he` is not a frame number (a frame may end in h to hold or e to ease, not both)`` |
| two keys on one tick (`@0` and `@0.001`) | ``bad value for `speed_percent`: two keyframes at the same time`` |

An id given twice is a usage error (exit 2). `tools/apply.py` relays each of these as `[error] vfx: <message>` and exits 1.

**Keyframes** (temporal effects; a frame effect refuses a value with `@`, exit 3):

- `VALUE` is `V@F;V@F;...`. `F` is a frame number counted from 0 on the output timeline, fractions allowed. `src_mark_segments` is read at source frame times, so its keys count source frames ([`src_mark_segments`](docs/parameters.md#src_mark_segments)).
- A trailing `h` on `F` holds that key's value until the next key; a trailing `e` eases the key (zero slope there). Not both.
- Keys may come in any order; two keys on the same tick are an error. Before the first key the curve has the first key's value, after the last key the last one's.
- `float`, `angle` and `point` values interpolate: straight between plain keys; with `u` the progress from 0 to 1, `u*u*(3-2*u)` between two eased keys, `u*u*(2-u)` after an eased key and `u*(1+u*(1-u))` before one, never overshooting. `int`, `popup`, `checkbox` and `color` hold until the next key.
- Quote the argument, because `;` ends a shell command: `--param 'speed_percent=100@0e;50@24e;100@48e'`.
- How keyed speeds and frame numbers become source positions: [timing](docs/timing.md).

**Time base.** Keys are stored in ticks, and `F` rounds to the nearest tick. By default an integer rate has 100 ticks a frame (`--fps 24`: 2400 a second) and a fractional rate N/D in lowest terms has N ticks a second and D a frame (`30000/1001`: 1001 a frame; `29.97`: 100 a frame). `--time-scale T` replaces the default when T x D / N is a whole number; an integer rate above 42949672 fps needs it. `tools/speed_curve.py` assumes the default, which is what `tools/apply.py` uses.

### Exit codes

`vfx` writes every error as one line on stderr, `vfx: error: <message>`. Frames finished before an error stay in the output.

| Code | Meaning | Messages |
|---|---|---|
| 0 | done | a render, `--check`, `list`, `--help`, `--version` |
| 2 | usage | ``unknown option `--x` ``; `` `--depth` expects 8, 16 or 32, got `12` ``; `retime is a temporal effect and needs --fps N[/D], for example 24 or 30000/1001`; `--fps 29.97002997 is not an exact rate: ...`; `--time-scale 1000 gives 41.667 ticks per frame at 24 fps, but a frame must last a whole number of ticks: use a multiple of 24`; `<id> is a frame effect, --frames only apply to temporal effects`; `--out and <file> are the same file`; `refusing to read raw frames from a terminal, pipe them in or use --in` |
| 3 | effect, param or render | ``unknown effect `nope`, `vfx list` shows the registered ones``; ``bad value for `frame_interp`: 4 outside 1..=3``; ``unknown layer `nope`: retime has the layers alt_source_a, alt_source_b``; a render failure names its frame: `frame 0: unsupported: <what, and the control to change>`, `frame N: effect panicked: ...` |
| 4 | I/O | `cannot open input <file>: ...`; `input <file> ends with a partial frame (1000 of 61440 bytes) after 0 whole frames`; `layer <id> <file> holds no frames`; read, write and spool errors |

**Error fills.** Missing data does not always fail the run. `vectors-in` writes an error fill, red at half alpha (8 bpc pixel `255,0,0,128`), at every frame where no vector layer has a frame (no `vectors_prev` or `vectors_next` given, or the vector clips end before the output does), and exits 0.

### Wrapper

`python tools/apply.py INPUT EFFECT [flags]` runs one effect over a video file: ffmpeg decodes to the raw format of `--depth`, `vfx apply` renders, ffmpeg encodes, and the result is probed against the input before it takes the output path. The commands it builds, stage by stage: [ffmpeg pairing](docs/ffmpeg.md#the-applypy-pipeline).

| Flag | Default | Meaning |
|---|---|---|
| `INPUT`, `EFFECT` | required | A video file and an effect id. |
| `--param K=V` | | Passed to `vfx apply --param` unchanged; repeatable. |
| `--layer ID=VIDEO` | | Temporal only, repeatable: a layer id the effect lists. The video is decoded to a raw file at the input's size (scaled when it differs) and frame rate (resampled when it differs), in the scratch folder. |
| `--frames N` | the frames the input decodes to | Temporal only. Output length: the output keeps the input's size and rate and lasts N / fps. |
| `--depth 8\|16\|32` | 16 | Bits per channel between ffmpeg and `vfx`. |
| `--lossless` | off | Encodes with ffv1 (level 3) in the RGB the effect wrote, alpha kept, no yuv420p and no tv range. The default output becomes `.mkv`. Refused with `--depth 32`. |
| `-o PATH`, `--output PATH` | `<input stem>_<effect>.mp4` beside the input (`.mkv` with `--lossless`), with `_1`, `_2`, ... added while the name is taken | Output path. An existing path is refused unless `--force`. |
| `--force` | off | Replaces an existing `-o` path, once the new output verifies. The input, or a layer video, is never replaced. |
| `--dry-run` | off | Prints the plan and the exact commands, writes nothing, exits 0. |
| `--json` | off | Prints the result as one JSON object on stdout. |
| `--vfx PATH` | lookup below | The `vfx` executable. |
| `--ffmpeg-path PATH`, `--ffprobe-path PATH` | lookup below | The ffmpeg and ffprobe executables. |
| `--pre-stage CMD` | | A raw frame filter of another tool, run between the decode and `vfx`; repeatable, in the order given. CMD is one command line (split like a shell would, no shell involved) that reads frames of the `--depth` format on stdin and writes the same number of frames of the same format and size on stdout. `{width}`, `{height}`, `{depth}`, `{fps}` (as N/D), `{frames}` and `{pix_fmt}` in it are filled in; `{frames}` is refused when the frame count is not known. A stage that exits non zero, stops in the middle of a frame or stops reading fails the run. With a stage, `--json` adds `contract` and `stages`, and a failure prints the error object of `contracts/error.schema.json` (exit 1 under `frames/1`). |
| `--post-stage CMD` | | Like `--pre-stage`, between `vfx` and the encode; its `{frames}` is the length of the `vfx` output (`--frames` for a temporal effect). |
| `--stage-timeout SECONDS` | 120 | Kills a `--pre-stage` or `--post-stage` that holds the pipe up this long without a byte moving on its pipes (0 never). A stage that waits for a slow neighbour is not held up. Under `frames/2` the decode and the encode are watched too; `vfx` never is. |
| `--contract frames/1\|frames/2` | `frames/1` | `frames/1` is the behaviour this section describes, unchanged. `frames/2` is the handoff contract ([handoff](docs/handoff.md)): every `--json` failure is the error object of `contracts/error.schema.json`, a text failure reads `error: <code>: <message>`, and the exit status follows the code: 1 `stage_failed`, 2 `usage`, 3 `unsupported`, 4 `bad_input`, `broken_pipe` and `partial_frame`, 5 `stage_timeout`, 6 `tool_missing`, 130 `interrupted`. A `--pre-stage` or `--post-stage` that exits 2, 3 or 4 is `usage`, `unsupported` or `bad_input`, as `vfx` is. `--json` adds `ok`, `contract`, `stages` (every process in pipe order, then the verify step, each with `role`, `index`, `name`, `command` as an argv list, `exit_code` and `stderr_tail`) and `geometry`. `--stage-timeout` also watches the decode and the encode; `vfx` is never judged. |
| `--geometry preserve\|declared` | `preserve` | `frames/2`, after a `--pre-stage` or `--post-stage`: a `declared` stage is first asked `CMD --describe [--frames N]` (`contracts/describe.schema.json`), and its width, height and frame count go to the next process, the encode and the verify. A frame rate or depth change is refused with exit 3, and so is another value. |
| `--describe` | off | `frames/2`: prints the output geometry of the run as a describe answer (`stage` is the effect id) and exits 0 without reading or writing frames. |

**Lookup order**, first hit wins:

| Program | Order | Not found |
|---|---|---|
| `vfx` | `--vfx PATH`, `VFX_PATH`, `$CARGO_TARGET_DIR/release/vfx`, this repository's `target/release/vfx`, then `vfx` on `PATH` (`vfx.exe` on Windows). A `--vfx` or `VFX_PATH` file that does not exist is an error, not a fall-through. | `[error] vfx not found at PATH`, `[error] VFX_PATH points at a missing file: PATH`, or ``[error] vfx not found. build it with `cargo build --release -p fx-cli`, or pass --vfx PATH / set VFX_PATH`` |
| ffmpeg | `--ffmpeg-path` (it sets `MEDIA_FFMPEG_PATH`), `MEDIA_FFMPEG_PATH`, `FFMPEG`, `PATH`, then on Windows `%LOCALAPPDATA%\Microsoft\WinGet\Links`, `%ProgramFiles%\ffmpeg\bin` and `%ProgramFiles(x86)%\ffmpeg\bin`. | a name or path given that cannot be found: `[error] explicit ffmpeg executable not found: <name>`; none at all: `[error] this operation requires ffmpeg and ffprobe.` with install lines |
| ffprobe | The same with `--ffprobe-path`, `MEDIA_FFPROBE_PATH`, `FFPROBE`. | as for ffmpeg |

**A run, in order:**

1. Probes the input with ffprobe and refuses with one `[error]` line and exit 1:

   | Refused | `[error] ...` |
   |---|---|
   | no video stream | `input has no video stream` |
   | an unreadable size or rate | `cannot read the input's frame size and frame rate` |
   | HDR (`color_transfer` `smpte2084` or `arib-std-b67`) | `HDR input is not supported; tone mapping is not part of this tool` |
   | `--lossless` with `--depth 32` | `--lossless needs --depth 8 or 16: ffv1 has no float pixel format` |
   | an odd width or height without `--lossless` | `the frame size 161x97 has an odd side, and yuv420p needs even sides; resize or crop the input first (...), or pass --lossless` |
   | an output path that is the input or a layer video | `the output path is the input file; choose a different --output path` |
   | an existing `-o` path without `--force` | `output already exists; choose a new --output path or pass --force` |

2. Finds `vfx`, reads the effect's kind and layers from `vfx list --json --all`, checks `--layer` and `--frames` against them, and probes every layer video (a layer gets the same no-video and HDR refusals, prefixed `layer ID:`).
3. Runs `vfx apply ... --check` with the real size, depth, rate, params and layers, so a bad id or value fails before any decoding.
4. Decodes the layers, then runs decode, `vfx` and encode as one pipe ([ffmpeg pairing](docs/ffmpeg.md#the-applypy-pipeline)).
5. Encodes with libx264 CRF 18, yuv420p, tv range, or with ffv1 under `--lossless`. Audio is copied unchanged: cut at the end of the video (`-shortest`) when the output is shorter, ending early when it is longer.
6. Writes `<output stem>.partial-<pid>.<ext>`, verifies it, then moves it onto the output path. Scratch files (layer clips, logs, the `vfx` spool) live in an `apply_*` folder in the OS temp folder, removed at exit.

A failure after step 3 ends with `[error] the pipeline failed:` and each failing stage's exit code with the last three lines of its log, or `[error] the output did not verify: ...`; the output path is left untouched.

**Verification** (a failure is `the output did not verify: ...`, exit 1): the input's displayed size; a frame rate within 0.01 fps; without `--frames`, or with `--frames` equal to the input's frame count, a duration within one frame of the input's; with another `--frames N`, a duration within max(1/fps, 0.1 s) of N / fps and, where the container reports it, exactly N frames; the input's audio still there with the same codec and channel count, and no audio added.

**Output.** Text mode prints `[plan] <summary>`, `[temp] <path> (<what>)` and one `[cmd] <command>` per stage (decode, effect, encode; runnable as printed), then `[dry-run] nothing was written` or, as the last line, `<absolute output path>: <duration>s, <W>x<H>, <fps> fps, <video codec>[, <audio codec> <N>ch]`. A plan line reads:

```
[plan] retime on clip.mp4: 160x96, 24.000 fps, 8 bpc (rgba), temporal, 95 frames, the audio ends before the video does, decode | vfx | encode -> target/demo/slow.mp4
```

The audio phrase is `audio cut at the end of the video (-shortest)` when the output is shorter than the audio by more than a frame, `the audio ends before the video does` when it is longer, and absent otherwise. `--json` prints `operation`, `input`, `effect`, `kind`, `params`, `depth`, `pix_fmt`, `size`, `fps`, `fps_exact`, `frames`, `input_frames`, `layers`, `audio` (`plan`: `shortest`, `ends-early`, `same` or null, and `shortest`), `temp_files`, `vfx`, `summary`, `dry_run`, `commands`, `outputs` (each `path` and its ffprobe facts under `probe`) and, after a real run, `seconds`. Errors print `[error] <message>` on stderr, and under `--json` also `{"ok": false, "error": "<message>"}` on stdout.

**Exit codes:** 0 done (or dry run), 1 any refusal or failure, 2 a flag argparse rejects (for example `--frames 0`), 130 interrupted.

### Helper tools

**`tools/speed_curve.py`** checks a `speed_percent` curve without rendering, with the effect's own arithmetic (half-frame sums in 32-bit floats, the default time base), so its positions agree with a render to a few thousandths of a frame. It models Speed mode (`time_remap_mode=1`) on a clip that plays at the output rate, from clip frame 0 ([what it models](docs/timing.md#what-speed_curvepy-models)).

| Flag | Default | Meaning |
|---|---|---|
| `--keys K` | required | The keys as for `--param speed_percent=...`, `h` and `e` suffixes included; a leading `speed_percent=` is accepted. Speeds -10000 to 10000. A string that starts with `-` needs the `=` form: `--keys="-100@0"`. |
| `--fps R` | required | Clip and output rate: `24`, `60`, `30000/1001`. Ticks follow the fraction, so `29.97` (100 ticks a frame) and `30000/1001` (1001) place fractional keys differently. |
| `--clip-frames N` | required | Frames in the clip the effect reads, at least 1. |
| `--frames N` | the frame of the last key, rounded | Output frames to check. |
| `--start SECONDS` | none | Where the clip starts in the recording; source and mark times become recording times. |
| `--mark NAME=SECONDS` | none | Repeatable: a source moment to find in the output. |

It prints `speed curve: <K> keys, <R> fps (<T> ticks a frame)`, `output ...` and `clip ...`, a key table (`key frame`, `time`, `speed`, `ease`, `source frame`, `source time`, plus `recording time` with `--start`), then:

| Line | Says |
|---|---|
| `last output frame N shows source frame P (T s into the clip)` | Where the curve ends. |
| `speed   A% to B%: ...` | Time below, above and at 100%; `at the slowest, one source frame spans X output frames`. |
| `ends    T s at a constant S%` | The final steady run. |
| `tail    ...` | Printed when the final speed is a whole multiple of 100% other than 0: whether that run shows whole source frames, and if not a corrected `--keys "..."` that moves the keys from the last ramp-out on by up to three frames ([exact frames](docs/timing.md#exact-frames)). |
| `frames  K of N (P%) need interpolation, about M min at 1080p` | Output frames more than 0.005 from a whole source frame, timed at 0.25 s each. |
| `change  largest speed step between frames: ...` | The sharpest speed change. |
| `marks` | One line per `--mark`: its clip frame and the output frames it shows at (`then` for a later pass, `outside the clip`, or `never reached by output frames 0 to N`). |
| `warnings: none`, or `warnings` and a list | Output frames that read outside the clip (`output frames A to B read source frames outside the clip (0 to C): positions P to Q`), plain keys that bend the speed by 1 point a frame or more (`a trailing e would round it`), a last key after the last output frame. |

Exit codes: 0 fine (bend and last-key warnings included); 1 bad `--keys` (`[error] --keys: ...`), or no `--frames` while the last key is at frame 0 (`[error] give --frames: the last key is at frame 0`); 2 a missing or malformed flag (argparse, including `--clip-frames` below 1); 3 the curve reads source frames outside the clip.

Test vector:

```sh
python tools/speed_curve.py --keys "100@0e;50@24e;100@48e" --fps 24 --clip-frames 48 --frames 60
```

Exit 0, and among its lines: `last output frame 59 shows source frame 47.0 (1.96 s into the clip)`, `frames  44 of 60 (73%) need interpolation, about 0.2 min at 1080p` and `warnings: none`.

**`tools/repeat_keys.py INPUT`** finds the frames that repeat the frame before them and prints the `frame_num` keys (for `time_remap_mode=2`) that give each repeat the picture between its neighbours, keeping the clip's length.

| Flag | Default | Meaning |
|---|---|---|
| `--threshold T` | 0.5 | A frame repeats the previous one when their mean absolute difference, on a 160x90 grey copy (0 to 255), is below T. |
| `--list` | off | Prints the repeated frame numbers instead of the keys. |

- stdout: the keys `V@F;...`, with an identity key at the first and last frame and on each side of every repeat (a run of k repeats is spread evenly towards the next new picture); under `--list`, the frame numbers separated by spaces (an empty line when there are none). Without `--list` and without repeats, stdout stays empty. A repeat on the last frame N gets the key `N.5@N`, which reads past the clip ([clip edges](docs/timing.md#clip-edges)).
- stderr: `K of N frames repeat the frame before them (P%)`, N being the frame count minus 1, and `[note] no repeats: nothing to fix` when the keys would be empty.
- ffmpeg comes from the [wrapper's lookup](#wrapper) (`MEDIA_FFMPEG_PATH`, `FFMPEG`, `PATH`, the Windows install folders); there is no `--ffmpeg-path` flag.
- A still stretch (a pause, a menu) also counts as repeats; the tool cannot tell them apart from dropped frames.
- Exit codes: 0 (also with no repeats); 1 ffmpeg missing or failing (`[error] ffmpeg was not found`), or fewer than two frames; 2 a malformed flag.

Test vector: a 20 fps source in a 24 fps file repeats every sixth frame.

```sh
ffmpeg -v error -f lavfi -i testsrc2=size=160x96:rate=20 -vf fps=24 -t 2 -c:v libx264 -crf 10 -pix_fmt yuv420p target/demo/rec.mp4
python tools/repeat_keys.py target/demo/rec.mp4 --list
python tools/repeat_keys.py target/demo/rec.mp4
```

The list is `3 9 15 21 27 33 39 45`, with `8 of 47 frames repeat the frame before them (17.0%)` on stderr; the keys start `0@0;2@2;3.5@3;4@4;8@8;9.5@9;10@10` and end `46@46;47@47`.

### Comparison harness

`target/release/fx-parity` builds a test set of clips (temporal effects) or stills (frame effects), writes a checklist for rendering references with another program, compares `vfx` with those references, and checks `vfx` for determinism. `fx-parity --help` prints the full rules.

| Subcommand | Reads | Writes | Key flags (default) |
|---|---|---|---|
| `gen --out DIR` | `vfx list --json --all`; ffmpeg (`FFMPEG`, else `ffmpeg` on `PATH`) renders the inputs | `DIR/inputs/<source>-<W>x<H>.d<depth>[.f<frames>].<pix_fmt>` and `DIR/cases.json` (format 1); prints `wrote DIR/cases.json: C cases for E effects, I inputs` | `--effects ID,...` (every listed effect; a hidden one only when named), `--sizes WxH,...` (clips 16x12,37x23, stills 1x1,2x3,37x23,255x129,320x180; the flag sets both), `--depths` (8,16,32), `--frames` (12), `--fps` (24,30000/1001, taken in turn by the clip sources), `--clip-sources` (testsrc2,movegrad,noisefx,alphaslide), `--clip-sweep-sources` (testsrc2,alphaslide; `all`), `--sources` and `--sweep-sources` (stills gradient,bars,noise,alpha,range; sweeps noise,alpha), `--no-sweep`, `--max-cases-per-effect` (400; 0 keeps all), `--jobs`, `--vfx` |
| `checklist --testset DIR` | `cases.json`, `vfx list --json` | `DIR/CHECKLIST.md` (or `--out FILE`); prints `wrote <file>: C cases for E effects` | `--out`, `--vfx` |
| `run --testset DIR --refs DIR` | `cases.json`, the inputs, the references `<case id>.<pix_fmt>` in `--refs` | nothing: a report on stdout | `--effect ID`, `--brief` (failures only), `--json`, `--strict` (a missing reference fails), `--jobs`, `--vfx` |
| `selfcheck --testset DIR` | `cases.json`, the inputs | nothing: a report on stdout | `--effect ID`, `--max-cases` (24, spread over the effects; 0 all), `--threads N,...` (1,3), `--json`, `--jobs`, `--vfx` |

- Cases: a temporal effect gets `all-default`, `all-min`, `all-max` and, when it has an optional layer, `no-layers` on every clip source, size and depth; without `--no-sweep` also one param at a time and a two key ramp per `float`, `angle` and `point` param on the sweep sources. When `--max-cases-per-effect` trims an effect, those four settings and one case for each param's min and max always stay (so a cap below their count removes none of them), then one case per other setting, then a fixed pseudo random fill. Every case except `no-layers` gets a clip of another source for every layer the effect lists. A case id reads `<effect>.<label>.<source>-<W>x<H>.d<depth>.<hash>`.
- `vfx` lookup: `--vfx`, `VFX_PATH`, next to `fx-parity` (or one folder up), then `PATH`. `--jobs` defaults to the number of cores, at most 8.
- Pass rule: 8 and 16 bpc bit exact, 32 bpc within 1 ULP, clips frame by frame with the first failing frame reported. Every compared case also gets a mean error (mean absolute difference over all pixels and all 4 channels, as a percentage of full scale), a clip its worst frame, and the report a `graded error` block per effect and the 10 worst cases.
- `selfcheck` runs each picked case twice at the default thread count and once per `--threads` value (through `RAYON_NUM_THREADS`); every output must equal the first byte for byte.
- Exit codes: 0 fine; 1 mismatch (`run`: a `FAIL`, or under `--strict` a missing reference; `selfcheck`: a difference or a failed run); 2 usage; 3 setup or I/O (for example a references folder that does not exist). Errors print `fx-parity: error: <message>`. There is no `--version`.

## Smoke test

Confirms a build end to end. Run from the repository root after [Build](#build); scratch files go to `target/demo/`, inside the git-ignored `target/`.

```sh
mkdir -p target/demo
ffmpeg -v error -f lavfi -i testsrc2=size=160x96:rate=24 -f lavfi -i sine=frequency=440:sample_rate=48000 -t 2 -c:v libx264 -crf 10 -pix_fmt yuv420p -c:a aac target/demo/clip.mp4
target/release/vfx list
python tools/apply.py target/demo/clip.mp4 retime --param speed_percent=50 --frames 95 --depth 8 -o target/demo/slow.mp4
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames,r_frame_rate,duration -of default=nw=1 target/demo/slow.mp4
ffmpeg -v error -i target/demo/clip.mp4 -f rawvideo -pix_fmt rgba - | target/release/vfx apply retime --width 160 --height 96 --depth 8 --fps 24 --param speed_percent=50 --frames 95 > target/demo/slow.rgba
```

| Step | Expected |
|---|---|
| `vfx list` | four rows, `retime`, `create-vectors`, `retime-fgbg`, `vectors-in`, all `temporal`, with 17, 59, 63 and 20 params |
| `tools/apply.py` | exit 0; the last line ends `slow.mp4: 3.96s, 160x96, 24 fps, h264, aac 1ch` |
| `ffprobe` | `r_frame_rate=24/1`, `duration=3.958333`, `nb_read_frames=95` |
| raw pipe | exit 0; `target/demo/slow.rgba` is 5836800 bytes, 95 frames of 160 x 96 x 4 |

95 frames is the longest render at 50% that stays inside the 48-frame clip ([output length](docs/timing.md#output-length)). The 2 s tone is copied, not stretched, so it ends before the video.

## Comparing with reference renders

`fx-parity` measures how far `vfx` is from another program's renders of the same cases, and whether `vfx` gives the same bytes on every run and thread count. A small set for one effect at 8 bpc:

```sh
target/release/fx-parity gen --out target/demo/pack --effects retime --depths 8 --no-sweep --sizes 160x96 --frames 12
target/release/fx-parity checklist --testset target/demo/pack
target/release/fx-parity selfcheck --testset target/demo/pack
mkdir -p target/demo/pack/refs target/demo/pack/export
target/release/fx-parity run --testset target/demo/pack --refs target/demo/pack/refs --brief
```

1. `gen` prints `wrote target/demo/pack/cases.json: 16 cases for 1 effects, 4 inputs`: 4 settings on 4 clip sources.
2. `checklist` writes `target/demo/pack/CHECKLIST.md`. Per case: the size, depth, frame count and rate, the input files, the settings by the names the interface shows (defaults left out, keys written `100 at frame 0, 50 at frame 24`), the extra layers, the reference file to produce, and two ffmpeg commands: one turns the raw inputs into PNG (8 and 16 bpc) or float EXR (32 bpc) sequences, the other turns the other program's export (in `export/`) into the raw reference (in `refs/`). The 32 bpc command divides the EXR's premultiplied colour by alpha again.
3. `selfcheck` needs no references. It ends `16 cases checked, 16 identical, 0 DIFFER` and exits 0.
4. `run` prints a line per case: `pass`, `no reference`, or `FAIL` followed by the case's settings, input and reference paths, a `repro:` line with the exact `vfx apply` command, the first failing frame and the first differing pixels (`x=20 y=10 in=[...] expected=[...] actual=[...]`). Then a `graded error` block per effect (cases compared, how many bit exact, mean error, worst case), the worst cases, and `P pass, F FAIL, N no reference, T cases`. With the empty `refs/` it ends `0 pass, 0 FAIL, 16 no reference, 16 cases` and exits 0; `--strict` makes that exit 1.

On a set of all four effects, every case but `no-layers` hands `create-vectors`, `retime-fgbg` and `vectors-in` their `fg1_matte` to `fg3_matte` layers, which they refuse ([Limits](#limits)): unless its settings only display the source, such a case fails with `vfx exited with code 3: vfx: error: frame 0: unsupported: a FG Matte layer is set ...`. A `--no-sweep` 160x96 set of all four effects gave 14 such failures among the 24 cases `selfcheck` picked (exit 1). Pass `--effect ID` to `run` or `selfcheck` to check one effect.

## Tests

From the repository root:

```sh
cargo build --release -p fx-cli
cargo test --workspace --release
python tools/test_speed_curve.py
python tools/test_apply.py
python tools/test_repeat_keys.py
python tools/test_stages.py
python tools/test_ffstage.py
python -I tools/test_handoff.py
python -I tools/test_conformance.py
python -I tools/conformance.py --quick --allow-absent-peer
```

- **Release mode** because the optical flow and thread invariance suites are far too slow unoptimised. The end to end tests (`crates/fx-cli/tests/cli.rs`, `plugins/*/tests/cli.rs`) run the `vfx` that cargo builds in the same profile.
- **ffmpeg:** the plugins' end to end tests and `crates/fx-parity/tests/selftest.rs` use ffmpeg where they need a real clip and skip those parts, with a note, without it. `tools/test_apply.py` needs ffmpeg, ffprobe and a built `vfx` (`target/release` or `VFX_PATH`) and skips when one is missing; `tools/test_repeat_keys.py` needs ffmpeg for its one end to end test; `tools/test_speed_curve.py` needs neither.
- **Evidence:** hand worked values, goldens and model fixtures from independent numpy float32 scripts, analytic ground truth (a texture moved by a known amount), and identical output across thread counts.
- **Timing probes** are `#[ignore]`d: `cargo test --release -- --ignored <name> --nocapture` with `bench_1080p`, `timing_full_hd`, `timing_probe_1080p`, `timing` or `breakdown`.

**Fixtures.** The model fixtures are hex text, 64 bytes a line, which the tests decode; the repository holds no binary files. Each generator needs numpy and writes to the path given (most default to overwriting their fixture in place). Run it from its own folder and compare, for example `python -I -B gen_warp.py /tmp/warp_cases.hex && cmp /tmp/warp_cases.hex warp_cases.hex`. Every generated fixture below came back byte for byte (numpy 2.5.1).

| Fixture | Generator | Command |
|---|---|---|
| `plugins/retime-engine/tests/fixtures/f1_cases.hex` (Smart Blend, hole fill) | `f1_gen.py` | `python -I -B f1_gen.py [OUT]` |
| `.../fixtures/f2_cases.hex` (forward warp) | `f2_gen.py` | `python -I -B f2_gen.py [OUT]` |
| `.../fixtures/blur_cases.hex` (motion blur) | `gen_blur.py` | `python -I -B gen_blur.py [OUT]` |
| `.../fixtures/vectors_cases.hex` (vector image coding) | `gen_vectors.py` | `python -I -B gen_vectors.py [OUT]` |
| `.../fixtures/warp_cases.hex` (inverse warp) | `gen_warp.py` | `python -I -B gen_warp.py [OUT]` |
| `.../fixtures/flow_a.hex` to `flow_f.hex` (optical flow, a differential model that is not bit exact) | `gen_flow_fixtures.py` with `flow_ref.py` | `python -I -B gen_flow_fixtures.py OUT_DIR` |
| `plugins/retime/tests/common/goldens.rs`, `plugins/retime-fgbg/tests/common/goldens.rs` | `gen_goldens.py` beside each | `python -I -B gen_goldens.py OUT` |

The Gaussian and binomial kernel tables are constants in the test of `flow/conv.rs`, checked against the code bit for bit.

**Continuous integration.** `.github/workflows/ci.yml` (Linux, macOS and Windows, Python 3.12) and `.forgejo/workflows/ci.yml` (Linux in the `node:24-bookworm` container, with Rust from rustup and Debian's `python3`) install ffmpeg (a static 9.0 build on Linux) and run the commands above, with the release profile's LTO off and 16 codegen units to cut compile time, on every push to `main` and every pull request, with a 60 minute limit. sapphire is not checked out there, so `--allow-absent-peer` lets through only the conformance parts that need sapphire; every failed check still fails the job. Both can also be started by hand. A new push to a pull request cancels its run in flight; each push to `main` keeps a run of its own.

## Limits

A refused feature stops the render with exit 3 and `frame N: unsupported: <what, and the control to change>`, never a silent fallback. `--check` does not catch these; they appear at the first frame that needs the feature. The exact messages are in [parameters: refusals](docs/parameters.md#refusals).

| Not supported | Where | What happens |
|---|---|---|
| Foreground matte layers `fg1_matte`, `fg2_matte`, `fg3_matte` (subject and background separation) | `create-vectors`, `retime-fgbg`, `vectors-in` | `unsupported`; remove the layer. |
| User tracking points: `ptN_use` other than 1 | `create-vectors`, `retime-fgbg` | `unsupported` once a frame needs a motion search; set `ptN_use=1`. |
| The geometry overlay: `draw_geom` other than 1 while the source is displayed and a point is in use | `create-vectors`, `retime-fgbg` | `unsupported`; set `draw_geom=1`. |
| Debug views: `display=3` | `create-vectors` | `unsupported`; set `display` to 1 or 2. |
| The second tracker: `main_bg_track=0` | `create-vectors` | `unsupported`; keep `main_bg_track=1`. |
| Wrap-around (360 degree) images | all | `video_360=1` renders, but nothing wraps at the frame edges ([`video_360`](docs/parameters.md#video_360)); `track_rgb_a=1` on `retime-fgbg`, which would need it, is `unsupported` while motion is searched. |
| Motion search on tiny frames | all but `vectors-in` | The search runs on the frame halved once (`vectors=5`; `vector_resolution=1` in `create-vectors`), twice (`vectors=4`) or three times (`vectors=2` or `3`), not at all with `vector_resolution=2`, and once more under `--quality draft` (at most three times); a working size under 2x2 is `unsupported`. |
| Bezier keyframe handles | keyframes | Keys are linear, hold (`h`) or ease (`e`) only. |
| Retimed audio | `tools/apply.py` | Audio is copied unchanged; it ends early or is cut at the end of the video. |
| New size or frame rate | `tools/apply.py` | The output keeps the input's size and frame rate; only the length changes (`--frames`). |
| HDR input; odd frame sizes on the lossy path | `tools/apply.py` | Refused before any work. |
| GPU | all | CPU only (rayon threads); the bytes do not depend on the thread count. |

## Documentation

This README specifies the programs. The files in `docs/` specify behaviour in detail; each covers one concern and fits in one read. Places are cited as `file.md#anchor`. Where any of them disagrees with `vfx list --json`, the listing is right.

| File | Specifies |
|---|---|
| [`docs/parameters.md`](docs/parameters.md) | Every control of the four effects: values, defaults, ranges, what each changes, how the effects differ, every refusal. |
| [`docs/timing.md`](docs/timing.md) | How output frames map to source frames: Speed and Frame Number modes, key shapes, output length, clip edges, exact frames, the rate model. |
| [`docs/ffmpeg.md`](docs/ffmpeg.md) | How `vfx` and `tools/apply.py` pair with ffmpeg: raw frames, the wrapper's commands, colour, rate, audio, encoding, scratch and disk, input requirements. |
| [`docs/engine.md`](docs/engine.md) | How an output frame is computed: motion search, warps, blends, hole fill, blur compensation, vector images, determinism, cost, and the module that implements each stage. |
| [`docs/handoff.md`](docs/handoff.md) | How twixtor, sapphire and ffmpeg pass raw frames to each other: contract versions, the frame contract, error codes and exits, reports, stage discovery, declared geometry, the ffmpeg filter stage, handoff packets and their commands. |

## Legal

- **Independent work.** This project is written independently and contains none of Twixtor's code, binaries, assets or documentation.
- **No affiliation.** It is not affiliated with, endorsed by or sponsored by RE:Vision Effects.
- **Trademarks.** Twixtor is a trademark of RE:Vision Effects. It appears here only to name the plugin this project reimplements.
- **Interoperability.** Parameter ids, display names and values follow the plugin's interface so that settings carry over between the two.
- **No warranty.** Everything here is provided as is under the MIT License, without warranty of any kind and without liability.
- **Your responsibility.** You need the rights to the media you process, and you stay bound by the licences of any software you use alongside this project, ffmpeg included.
- **Rights holders.** If you believe something here infringes your rights, email [hi@bairon.dev](mailto:hi@bairon.dev) or [open an issue](https://github.com/baairon/twixtor/issues). Every concern gets a prompt answer in good faith.

## License

MIT, see [LICENSE](LICENSE).
