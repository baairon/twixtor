# twixtor

An independent open source reimplementation of the four effects of the Twixtor retiming plugin. It is not affiliated with or endorsed by RE:Vision Effects, which owns the Twixtor trademark.

Slow motion, speed ramps and frame interpolation from optical flow, written in Rust and paired with ffmpeg. `vfx` renders raw frames in ffmpeg's own pixel formats, `tools/apply.py` runs `vfx` between an ffmpeg decode and an ffmpeg encode, `tools/audio_map.py` plays the audio along an effect's time map, `tools/speed_curve.py` and `tools/repeat_keys.py` compute keyframes without rendering, `fx-parity` compares `vfx` with reference renders, `tools/gen_docs.py`, `tools/check_docs.py` and `tools/parity_lock.py` keep the parity record, `tools/ffstage.py` runs an ffmpeg filter graph as a stage, and `tools/handoff.py` and `tools/conformance.py` pass frames and tasks to sapphire and test a stage against the convention.

Sources of truth, for agents and scripts:

| Question | Ask |
|---|---|
| Effect ids, parameter ids, kinds, ranges, defaults, layer ids, limits (`gaps`) | `target/release/vfx list --json`. Any document that disagrees with it is wrong. |
| What a file render will do | `python tools/apply.py ... --dry-run`: the plan and every command, nothing written. |
| What a file render did | `python tools/apply.py ... --json`: the plan and the probe of the verified output. |
| Which source frame each output frame shows | `python tools/speed_curve.py`, without rendering. |
| Every stage of twixtor and sapphire, with a ready stage command | `python -I tools/handoff.py stages --json`: each stage with the command line a wrapper takes. |
| The render devices and what each reports | `target/release/vfx devices --json`. |
| What has and has not been verified against the original | [parity](docs/parity.md), its figures generated from `docs/data/` (`python -I tools/gen_docs.py --check`). |

## Contents

- [Parity](#parity)
  - [Limits](#limits)
  - [Not offered](#not-offered)
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
- [Handoff](#handoff)
- [Smoke test](#smoke-test)
- [Comparing with reference renders](#comparing-with-reference-renders)
- [Tests](#tests)
- [Documentation](#documentation)
- [Legal](#legal)
- [License](#license)

## Parity

1:1 parity with Twixtor is the goal and the first priority of this project, ahead of speed, convenience and a cleaner design. When they conflict, parity wins.

**The port has never been verified frame by frame against the original plugin.** The only comparison made so far is of the controls in the supplied plugin builds ([interface check](docs/parity.md#interface-check)); no output of the original has been compared.

- **Output (goal).** The same settings on the same frames should give exactly the same frames, down to the last bit at 8 and 16 bits per channel and to the smallest possible step at 32-bit float. The arithmetic rules the engine follows for that are written at the top of each file in `plugins/retime-engine/src`.
- **Interface (goal).** Every control of the original should have one matching setting here, with the same name, choices, range and default, so that settings carry over in both directions. The controls the supplied plugin builds register are the source of truth for this ([interface check](docs/parity.md#interface-check)); left out are the drawn mask controls ([Not offered](#not-offered)), the graphics card settings (`--device` does that job here) and a few that only concern the host application.
- **No silent guesses.** Every control renders. Where the original's behaviour is not known, this project's own stated rule is used and `vfx` says so with a note ([Limits](#limits)). The only inputs refused are ones no setting could handle, such as a Speed mode position beyond 2^23 frames ([parameters: refusals](docs/parameters.md#refusals)). Nothing quietly falls back to a rough approximation.
- **Claimed only when measured.** `fx-parity` makes test clips and a checklist of the same cases to render in the original, then compares the two frame by frame and reports every difference ([Comparing with reference renders](#comparing-with-reference-renders)). 1:1 is claimed for a control only when the checklist cases that set it match the original's frames in `fx-parity run`; no control carries that claim yet. The figures below are generated automatically, and the automated checks fail when one goes out of date ([parity](docs/parity.md)).

These two checks only show that the code agrees with itself (with separately written models of its steps and with an earlier version of itself), not that it matches the original.

| Check | Count | Share |
|---|---|---|
| Matches separately written models exactly | <!--f:model_fixtures-->11<!--/f--> fixtures, <!--f:model_cases-->794<!--/f--> cases | - |
| Same output as an earlier version | <!--f:lock_identical-->13,073<!--/f--> of <!--f:lock_total-->13,073<!--/f--> | <!--f:lock_pct-->100.0<!--/f-->% |

The earlier version is the build of commit <!--f:lock_ref-->d74f402<!--/f-->: every setting that rendered there gives the same bytes now.

A control is one setting or input layer of the four effects. What each row counts, the test sets behind the lock and the fine print (for example, a few steps use the operating system's math library, so the output bytes are fixed per platform rather than across all of them) are in [parity](docs/parity.md).

Some features are this project's own, with nothing in the original to compare them with, so they print no note:

- the wrapper's extras in `tools/apply.py`: retimed audio, a new size or frame rate, HDR input and odd frame sizes;
- Bezier key handles (the values between keyframes are worked out here; in the original the editing application does it);
- motion on tiny frames: the search never shrinks a frame below 2x2 pixels, and a frame 1 pixel wide or tall gets no motion;
- rendering on a graphics card with `--device gpu`, which gives the same bytes as the CPU; some steps still run on the CPU, and `fx-parity devcheck` checks that the two match byte for byte ([engine: device](docs/engine.md#device)).

Their rules, and the comparisons with the original that could still test parts of them, are in [parity: rules without a note](docs/parity.md#rules-without-a-note).

### Limits

The only comparison made so far is of the controls in the supplied plugin builds ([interface check](docs/parity.md#interface-check)); no output of the original has been compared. Where the original's behaviour is not known, this project uses its own stated rule, and `vfx` prints a short `vfx: note:` line when a render uses one. Those places are mattes and matte layers, tracking points, Draw Geom, the Debug views and wrap-around; the full list and each rule are in [parity](docs/parity.md#gaps-register).

### Not offered

Not part of this project: licensing, and running inside another application (this is a command-line renderer).

Drawn masks (the Hide Mask, First Mask and Last Mask controls) belong to an editor, so they are not offered here. Regions are agent driven instead: you describe in plain words which part of the picture should be tracked on its own, and your agent makes the matte and passes it as `fg1_matte`, `fg2_matte` or `fg3_matte`. That part then gets its own motion, while the whole frame is still retimed at the same speed. `vfx` only takes the finished matte; making it is the agent's job and is not part of this repository. How a matte is read (which channel, size and timing) is in [parameters](docs/parameters.md#retime-fgbg).

## The effects

`target/release/vfx list` shows four effects, all of family `retime` and kind `temporal`; every layer is optional.

| Id | Name in `vfx list` | Reference | Params | Layers (`--layer` ids) | What it does |
|---|---|---|---|---|---|
| `retime` | Retime | [parameters](docs/parameters.md#retime) | 17 | `alt_source_a`, `alt_source_b` | Retimes a clip by a keyframed speed in percent (`time_remap_mode=1`) or by source frame number (`time_remap_mode=2`) and builds the in-between frames from optical flow: `vectors` 1 to 5 (No Vectors to Best), `frame_interp` Nearest, Blend or Weighted Blend, `smart_blend`, `blur_compensation`. |
| `create-vectors` | Create Vectors | [parameters](docs/parameters.md#create-vectors) | 59 | `alt_source_a`, `fg1_matte`, `fg2_matte`, `fg3_matte` | Does not retime: writes the motion between each frame and its previous or next one (`track_frame`) as a vector image, red horizontal and green vertical, centred and scaled by `max_displace`, blue 0, alpha opaque. FG matte layers give each layer its own motion, merged by the mattes (`layers_output`); `display=3` shows the Debug views. |
| `retime-fgbg` | Retime (FG/BG) | [parameters](docs/parameters.md#retime-fgbg) | 63 | `alt_source_a`, `alt_source_b`, `fg1_matte`, `fg2_matte`, `fg3_matte` | The retime of `retime` with a `warping` choice (1 Inverse, 2 Inverse w/ Smart Blend, 3 Forward, 4 Forward, no Smart Blend) in place of `smart_blend`, with up to three FG matte layers (one motion per layer, composed with FG1 on top, the `display` groups), tracking points, the Draw Geom overlay and `track_rgb_a` as the wrap-around flag. |
| `vectors-in` | VectorsIn | [parameters](docs/parameters.md#vectors-in) | 20 | `source`, `vectors_prev`, `vectors_next`, `fg1_matte`, `fg2_matte`, `fg3_matte` | Retimes with the vector images of `create-vectors`, given as layers, instead of searching for motion itself; `warping` 1 Inverse or 2 Forward; FG matte layers as in `retime-fgbg`. |

`vfx list --all` adds the hidden family `selftest`, test effects for `vfx` and `fx-parity` that `vfx apply` also runs: `selftest-invert` (the only frame effect), `selftest-shift` (layer `overlay`), `selftest-integrate`, `selftest-threads` (deliberately thread dependent, so that `fx-parity selfcheck` can be seen to fire) and `selftest-gap` (a copy of its input that records the test gap `T1` on every frame with the checkbox `gap` on, so that its note can be seen to print once per run).

## Crate map

Workspace members of `Cargo.toml`: version 0.1.0, edition 2021, MIT, unpublished. External crates: `rayon`, `serde`, `serde_json`.

| Member | Builds | Owns | Key modules (`src/`) | Tests |
|---|---|---|---|---|
| `crates/fx-core` | library | The interface every crate shares: the `Effect` trait (frame or temporal kind), frames and the raw wire format, parameter specs and command line value parsing, keyframe curves, time bases, layers. | `lib.rs` (`Effect`, `FxError`), `frame.rs`, `params.rs`, `curves.rs` (with the Bezier handles), `time.rs`, `layer.rs`, `fixed.rs` (16.16 fixed point), `gaps.rs` (`OpenGap`, `record`, `take_recorded`: the stated rules of the [limits](#limits) a render used), `device.rs` (`DeviceRequest`, the `--device` request a temporal context carries) | in-module (`#[cfg(test)]`); the Bezier model fixture `tests/fixtures/bezier_cases.hex` with its generator |
| `crates/fx-cli` | binary `vfx` | The renderer: arguments, effect registry, listing, frame and temporal runs over files or pipes. | `main.rs`, `cli.rs` (flags, usage text), `registry.rs` (`FAMILIES`: `retime` and the hidden `selftest`), `list.rs`, `apply.rs` (frame effects, `--check`, panic guard), `temporal.rs` (time base, curves, layers, clip rates, notes), `clip.rs` (random access to raw clips at their own rate, stdin spool), `devices.rs` (`vfx devices`, the device check of `--device`), `selftest.rs`, `error.rs` (exit codes) | in-module; `tests/cli.rs` and `tests/device.rs` (black box tests of the built binary) |
| `crates/fx-parity` | library `fx_parity`, binary `fx-parity` | The comparison harness: test patterns, case plans, the reference case families of the limits, reference checklists, comparison and grading, determinism checks. | `args.rs`, `sources.rs` (lavfi patterns, the family clips), `cases.rs` (`cases.json`), `families.rs` (the reference case families), `testgen.rs` (`gen`), `checklist.rs`, `run.rs`, `compare.rs`, `selfcheck.rs`, `devcheck.rs` (the byte gate of the GPU backend), `sweep.rs`, `effects.rs`, `vfx.rs` (finds and drives `vfx`) | in-module; `tests/selftest.rs` (needs ffmpeg, skips without) |
| `plugins/retime-engine` | library | The engine of the four effects: time model, frame selection, optical flow, warps, blends, hole filling, vector image coding, FG matte layers, tracking points, the register of limits, and the OpenCL device code ([engine](docs/engine.md)). | `timeline.rs`, `planner.rs`, `frontend.rs` (parameters to settings), `host.rs`, `compose.rs` and `compose/vector.rs`, `flow.rs` (with `flow/conv.rs`, `flow/sample.rs`, and `flow/extras.rs`: layer masks, tracking points, wrap-around), `prep.rs`, `warp.rs` (with `warp/blurcomp.rs`), `forward.rs` (with `forward/mesh.rs`, `forward/pyramid.rs`, `forward/raster.rs`), `smartblend.rs`, `holefill.rs`, `blend.rs`, `filters.rs`, `vectors.rs`, `create_vectors.rs`, `image.rs`, `matte.rs` (FG mattes), `points.rs` and `geom.rs` (tracking points, the second tracker, Draw Geom), `layers.rs` (the n-layer render), `debug.rs` (the Debug views), `gaps.rs` (the register of limits), `device.rs` and `device/` (OpenCL: `device/cl.rs` loads the runtime and holds the only FFI of the workspace, `device/probe.rs`, the kernels, the simulator, the self-check, the device stages, the executor `device/exec/` and the program binary cache `device/progcache.rs`) | in-module, the larger suites in their own files (`compose/tests.rs`, `forward/tests.rs`, `warp/tests.rs`, `holefill/tests.rs`, `smartblend/tests.rs`, `vectors/tests.rs`, `matte/tests.rs`, `layers/tests.rs`, `debug/tests.rs` and others; the device stages against their simulator, and on a device when there is one); `tests/flow_d.rs`, `tests/flow_masks.rs`, `tests/device_cl.rs` (on the machine's OpenCL devices, skipped without one), `tests/no_unsafe.rs`; model fixtures and their generators in `tests/fixtures/` |
| `plugins/retime` | library | Effect `retime`: params, layers, wiring to the engine. | `lib.rs`, `glue.rs` | `tests/`: `cli` (end to end through `vfx`), `retime` (goldens of `tests/common/gen_goldens.py`), `vector`, `dependencies`, `consistency`, `settings`, `threads`, `tiny` |
| `plugins/create-vectors` | library | Effect `create-vectors`: params, layers, wiring to the engine. | `lib.rs`, `glue.rs`, `prerender.rs` | `tests/`: `cli`, `encoder`, `roundtrip`, `vector`, `display`, `point_layer`, `dependencies`, `consistency`, `settings`, `threads`, `tiny`, `unsupported` (the renders once refused) |
| `plugins/retime-fgbg` | library | Effect `retime-fgbg`: params, layers, wiring to the engine. | `lib.rs`, `glue.rs`, `prerender.rs`, `timemap.rs` | `tests/`: `cli`, `retime` (goldens of `tests/common/gen_goldens.py`), `forward`, `vector`, `timemap`, `dependencies`, `consistency`, `settings`, `threads`, `tiny`, `unsupported` (the renders once refused) |
| `plugins/vectors-in` | library | Effect `vectors-in`: vector layers decoded into flows. | `lib.rs`, `glue.rs`, `prerender.rs`, `timemap.rs`, `vectorin.rs` | `tests/`: `cli`, `vectorin`, `vector`, `retime`, `errors`, `layers`, `timemap`, `dependencies`, `consistency`, `settings`, `threads`, `tiny`, `unsupported` (the renders once refused) |

Outside the workspace: `tools/apply.py`, `tools/audio_map.py`, `tools/speed_curve.py`, `tools/repeat_keys.py`, `tools/ffstage.py`, `tools/handoff.py` and `tools/conformance.py`, and the parity record tools `tools/gen_docs.py` (with `tools/docgen/`), `tools/check_docs.py` and `tools/parity_lock.py` (Python 3 standard library only; [Wrapper](#wrapper), [Helper tools](#helper-tools), [Handoff](#handoff)) with their `unittest` suites `tools/test_*.py`, the handoff contracts in `contracts/`, the technical references in `docs/` ([Documentation](#documentation)) and the parity snapshots in `docs/data/`.

## Build

From the repository root, with a stable Rust toolchain, 1.82 or newer (the code uses `Option::is_none_or`). The first build fetches the external crates from crates.io.

```sh
cargo build --release -p fx-cli -p fx-parity
```

| Binary | Path (`.exe` added on Windows) | Used by |
|---|---|---|
| `vfx` | `target/release/vfx` | every render; `tools/apply.py` |
| `fx-parity` | `target/release/fx-parity` | [Comparing with reference renders](#comparing-with-reference-renders) |

`CARGO_TARGET_DIR` moves both: `tools/apply.py` looks for `vfx` there, and `fx-parity` looks next to itself. The release profile is `opt-level = 3`, `codegen-units = 1`, thin LTO. `tools/apply.py` needs ffmpeg and ffprobe, and an HDR input on the lossy path an ffmpeg whose libx265 writes 10 bit; `tools/repeat_keys.py` and `fx-parity gen` need ffmpeg; regenerating test fixtures needs numpy ([Tests](#tests)). The build needs no OpenCL: `vfx` loads the system's OpenCL runtime at run time, when there is one.

## Interface

Every contract an agent relies on: what goes in, what comes out, how failure shows.

### Renderer

`vfx` reads raw frames, runs one effect and writes raw frames of the same size and format.

| Command | Does |
|---|---|
| `vfx --help` (also `-h`, `help`) | Prints the usage text, exit 0. `--help` after `list` or anywhere in an `apply` line does the same. |
| `vfx --version` (also `-V`) | Prints `vfx 0.1.0`. |
| `vfx list` | A table with columns `id name family kind params layers`, then per effect a `<id> params` table (`param name kind range default`) and a `<id> layers` table (`layer name optional`). |
| `vfx list --json` | The same as a JSON array. Each effect: `id`, `name`, `family`, `kind` (`frame` or `temporal`), `hidden`, `params`, `layers`. Each param: `id`, `name`, `kind` and that kind's fields: `float` `valid_min`, `valid_max` (the accepted range), `slider_min`, `slider_max` (an interface range only), `default`, `precision`; `int` `min`, `max`, `default`; `popup` `choices` (names in order) and `default` (1-based); `checkbox` `default` (`true` or `false`); `color` `default` `[r, g, b]`; `point` `default` `[x, y]`; `angle` `default`. Each layer: `id`, `name`, `optional`. Last, `gaps` when the effect has limits ([Limits](#limits)): each `id`, `title` and `controls` (the param and layer ids whose use reaches the rule); an effect without gaps has no `gaps` key. |
| `vfx list --all` | Adds the hidden `selftest` family (`"hidden": true`). |
| `vfx devices` | A table `id name kind default verdict` of the render devices `--device` can name: `cpu`, then every OpenCL device the system runtime reports, numbered `gpu=0`, `gpu=1`, ... platform by platform; then a block per OpenCL device with its platform, vendor, versions (OpenCL, OpenCL C, driver), device type, profile, endianness, compute units, memory (global, largest buffer, dedicated or shared with the host), fp32 and fp64 config bits, and the verdict `ok` or `refused` with one `reason` line per unmet requirement ([engine: device](docs/engine.md#device)). It runs info queries only, no kernel. Without an OpenCL runtime or device it lists `cpu` alone, without the `verdict` column, and prints one `vfx: note:` line on stderr. |
| `vfx devices --json` | The same as a JSON array. Each device: `id` (the `--device` value), `name`, `kind` (`cpu` or `gpu`), `default` (`true` for the device a run without `--device` uses); an OpenCL device adds `device_name`, `platform`, `vendor`, `version`, `driver`, `opencl_c`, `profile`, `type`, `little_endian`, `global_memory` and `max_allocation` (bytes), `unified_memory`, `compute_units`, `fp64`, `single_fp` and `double_fp` (flag names) with `single_fp_bits` and `double_fp_bits`, `extensions`, `verdict` and `reasons`. |
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
| `--frames N` | the input's frame count (with another `--clip-fps`, the output frames that start inside the clip) | Temporal only. Frames to render, counted at `--fps`; frames past the end of the clip see an empty input. |
| `--time-scale T` | see [time base](#parameter-values) | Temporal only. Ticks per second; a frame must last a whole number of ticks. |
| `--layer ID=FILE` | none | Temporal only, repeatable. An extra layer the effect lists: a raw clip of the same size and depth, any length. |
| `--clip-fps N[/D]` | `--fps` | Temporal only. The input clip's own rate, written like `--fps`: clip frame i covers i/R to (i+1)/R seconds, so footage at another rate is read by time and none of its frames is dropped or repeated beforehand (60 fps footage under `--fps 24` gives real footage frames down to 40%, [timing: rate model](docs/timing.md#rate-model)). The default `--frames` becomes the output frames that start inside the clip, ceil(C x F / R). A value equal to `--fps` changes nothing. |
| `--layer-fps ID=N[/D]` | `--fps` | Temporal only, repeatable, each id once. The same for the layer ID, which `--layer` must give. |
| `--quality draft\|high` | `high` | Temporal only. `draft` halves the motion search resolution once more than `high` does, but never below an eighth of the frame size; neither halves below 2x2 ([parameters: vectors](docs/parameters.md#vectors)). |
| `--field none\|upper\|lower` | `none` | Temporal only. Passed to the effect, but none of the four reads it: field order comes from the `input_fields` param. |
| `--device cpu\|gpu\|gpu=N` | `cpu` | Temporal only (a frame effect refuses it, exit 2). `gpu` renders on the best OpenCL device that passes the gate (the device requirements, a self-check over an adversarial corpus and the build of the kernel program; about 0.2 to 0.5 s once an earlier run has kept the program binary, several seconds when the program compiles), devices with their own memory first; when none does, the CPU renders after one `vfx: note:` line saying why. `gpu=N` is the device numbered `N` by `vfx devices`, strictly: one that is not listed or fails the gate exits 3 (`unsupported`) with the reasons, before any file is read, under `--check` too. The output bytes are the CPU's on every device: each stage the device cannot serve runs on the CPU, and the first frame that uses the device is rendered on the CPU too and compared byte for byte; a difference hands the rest of the run to the CPU with one note. `VFX_DEVICE_REPORT=1` adds a note line at the end with the device and fallback count per stage. Long clips gain the most: the device open and the cross check frame are paid once per run ([engine: device](docs/engine.md#device)). |
| `--tmp-dir DIR` | the OS temp folder | Temporal only. Where piped stdin is spooled (`vfx-spool-<pid>-...raw`); removed on every exit. |
| `--check` | off | Validates the effect, params (keyframes included), layer ids, rates (`--clip-fps` and `--layer-fps` too), frame size and `--device`, then exits 0 without reading or writing a file. It does not render, so the `unsupported` refusals of a render ([parameters: refusals](docs/parameters.md#refusals)) appear only in a real run, and it prints no `vfx: note:` line. |

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
| a malformed ease handle (`100@0i0/`) | ``bad value for `speed_percent`: `0i0/`: `i0/` is not an ease handle (i or o, then SPEED/INFLUENCE, as in i0/33.3)`` |
| a handle speed beyond 1e9 either way (`100@0o2000000000/50`) | ``bad value for `speed_percent`: handle speed 2000000000 is outside -1000000000..=1000000000`` |
| an influence outside 0.1 to 100 (`100@0i0/0`) | ``bad value for `speed_percent`: influence 0 is outside 0.1..=100`` |
| handles and `e` on one key (`100@0i0/50e`) | ``bad value for `speed_percent`: a key with ease handles cannot also end in e`` |
| an outgoing handle on a hold key (`100@0o0/50h`) | ``bad value for `speed_percent`: a held key has no outgoing handle`` |
| a handle on a `popup`, `checkbox`, `color` or `int` (`vectors=2@0i0/50;5@10`) | ``bad value for `vectors`: `2@0i0/50` has ease handles, but a popup holds its value between keys`` |

An id given twice is a usage error (exit 2). `tools/apply.py` relays each of these as `[error] vfx: <message>` and exits 1.

**Keyframes** (temporal effects; a frame effect refuses a value with `@`, exit 3):

- `VALUE` is `V@F;V@F;...`. `F` is a frame number counted from 0 on the output timeline, fractions allowed. `src_mark_segments` is read at source frame times, so its keys count source frames ([`src_mark_segments`](docs/parameters.md#src_mark_segments)).
- A trailing `h` on `F` holds that key's value until the next key; a trailing `e` eases the key (zero slope there). Not both.
- Bezier handles, as After Effects' Keyframe Velocity dialog sets them, go between `F` and the suffix: `V@F[iS/I][oS/I][h|e]`. `iS/I` shapes the segment into the key and `oS/I` the one out of it; `S` is the speed at the key in units a second (percent points a second for `speed_percent`, pixels a second along the line for a point) and `I` the influence in percent of the segment, 0.1 to 100; `S` is at most 1e9 either way, which keeps every value finite. Both numbers are written `-?[0-9]+(\.[0-9]+)?`, so `--param 'speed_percent=100@0o0/50;50@24i-30/50h'` survives a shell. A key with handles takes no `e`, and a hold key takes only an `i` handle (Bezier in, Hold out). A handle with no segment to shape (into the first key, out of the last) is kept and changes nothing. A frame number never holds an `i` or `o` followed by a digit or `-`, so a key without handles always reads as a plain, `h` or `e` key.
- Keys may come in any order; two keys on the same tick are an error. Before the first key the curve has the first key's value, after the last key the last one's.
- `float`, `angle` and `point` values interpolate: straight between plain keys; with `u` the progress from 0 to 1, `u*u*(3-2*u)` between two eased keys, `u*u*(2-u)` after an eased key and `u*(1+u*(1-u))` before one, never overshooting. A segment with a handle at either end is a cubic Bezier, where a plain end counts as a handle of the straight line's speed and an eased end as one of speed 0, both at an influence of exactly 1/3 (in exact arithmetic the four shapes above, which segments without handles keep bit for bit; [timing: key shapes](docs/timing.md#key-shapes) has the arithmetic). A handle may overshoot: a `float` is then held inside its valid range, an `angle` is not, and a `point` moves on the straight line between its keys at `S` pixels a second and stays put when its keys are equal. `int`, `popup`, `checkbox` and `color` hold until the next key and refuse handles.
- Quote the argument, because `;` ends a shell command: `--param 'speed_percent=100@0e;50@24e;100@48e'`.
- How keyed speeds and frame numbers become source positions: [timing](docs/timing.md).

**Time base.** Keys are stored in ticks, and `F` rounds to the nearest tick. By default an integer rate has 100 ticks a frame (`--fps 24`: 2400 a second) and a fractional rate N/D in lowest terms has N ticks a second and D a frame (`30000/1001`: 1001 a frame; `29.97`: 100 a frame). `--time-scale T` replaces the default when T x D / N is a whole number; an integer rate above 42949672 fps needs it. `tools/speed_curve.py` assumes the default, which is what `tools/apply.py` uses.

### Exit codes

`vfx` writes every error as one line on stderr, `vfx: error: <message>`. Frames finished before an error stay in the output.

A render that runs the stated rule of a limit ([Limits](#limits)) also prints, once per rule per run and right after the first frame that rendered with it, `vfx: note: <effect>: <id> <title>: a stated rule that no Twixtor reference render has graded yet (README: limits)`. A note never changes the frames or the exit code; a frame that fails prints none, and `--check` prints none.

| Code | Meaning | Messages |
|---|---|---|
| 0 | done | a render, `--check`, `list`, `devices`, `--help`, `--version` |
| 2 | usage | ``unknown option `--x` ``; `` `--depth` expects 8, 16 or 32, got `12` ``; `retime is a temporal effect and needs --fps N[/D], for example 24 or 30000/1001`; `--fps 29.97002997 is not an exact rate: ...`; `--time-scale 1000 gives 41.667 ticks per frame at 24 fps, but a frame must last a whole number of ticks: use a multiple of 24`; `<id> is a frame effect, --frames only apply to temporal effects`; `--out and <file> are the same file`; `refusing to read raw frames from a terminal, pipe them in or use --in`; `` `--device` expects cpu, gpu or gpu=N, got `opencl` ``; `` `--clip-fps` expects an exact positive rate such as 24, 29.97 or 30000/1001, got `x` ``; `--layer-fps overlay=25 names a layer that no --layer overlay=FILE gives` |
| 3 | effect, param or render | ``unknown effect `nope`, `vfx list` shows the registered ones``; ``bad value for `frame_interp`: 4 outside 1..=3``; ``unknown layer `nope`: retime has the layers alt_source_a, alt_source_b``; ``unknown layer `nope` in --layer-fps: selftest-shift has the layers overlay``; a `--device` this build cannot serve, before any frame (`unsupported: ...`); a render failure names its frame: `frame N: unsupported: <what>` ([parameters: refusals](docs/parameters.md#refusals)), `frame N: effect panicked: ...` |
| 4 | I/O | `cannot open input <file>: ...`; `input <file> ends with a partial frame (1000 of 61440 bytes) after 0 whole frames`; `layer <id> <file> holds no frames`; read, write and spool errors |

**Error fills.** Missing data does not always fail the run. `vectors-in` writes an error fill, red at half alpha (8 bpc pixel `255,0,0,128`), at every frame where no vector layer has a frame (no `vectors_prev` or `vectors_next` given, or the vector clips end before the output does), and exits 0.

### Wrapper

`python tools/apply.py INPUT EFFECT [flags]` runs one effect over a video file: ffmpeg decodes to the raw format of `--depth`, `vfx apply` renders, ffmpeg encodes, and the result is probed against the input before it takes the output path. The commands it builds, stage by stage: [ffmpeg pairing](docs/ffmpeg.md#the-applypy-pipeline).

| Flag | Default | Meaning |
|---|---|---|
| `INPUT`, `EFFECT` | required | A video file and an effect id. |
| `--param K=V` | | Passed to `vfx apply --param` unchanged; repeatable. |
| `--layer ID=VIDEO` | | Temporal only, repeatable: a layer id the effect lists. The video is decoded to a raw file at the output's size (scaled, stretched, when it differs), in the scratch folder. Without `--size` and `--fps` it is resampled to the input's rate when its own differs; with either it keeps its own rate (a variable one made constant at its average) and `vfx` reads it by time (`--layer-fps ID=R`). |
| `--frames N` | the frames the input decodes to; with `--fps`, the output frames that start inside the input, ceil(C x F / R) | Temporal only. Output length in output frames: the output lasts N / fps. |
| `--size WxH` | the input's displayed size | W and H 1 or more. The input and every layer are scaled to it, stretched (`scale=W:H`), before the effect, which renders at that size. The input's own size changes nothing. Matches only in full. |
| `--fps R` | the input's rate | The output rate: an integer, a decimal of at most 6 places or N/D, above 0. A temporal effect reads the input and every layer at their own rates (`vfx --clip-fps`, `--layer-fps`), so no real frame is dropped or repeated on the way in; a frame effect gets, for each output frame, the input frame that covers its start (`settb=D/N,setpts=N,fps=R:round=up` in the decode). Keys, `frame_num` and `--frames` count output frames. `retime`, `retime-fgbg` and `vectors-in` also get `in_fps_is_out_fps=0` and `input_frame_rate=<the input's rate>` (`%.10g`: 30000/1001 is written `29.97002997`) unless either id is given. The input's own rate (the average of a variable rate input) changes nothing. Matches only in full. |
| `--audio copy\|retime\|drop` | `copy` | What the output does with the input's audio; matches only in full. `copy` copies every audio stream untouched: cut at the end of the video (`-shortest`) when the output is shorter, ending early when it is longer. `retime` plays every stream along the effect's time map, varispeed (the pitch follows the speed): silent where the clip has no frame, a frame is held or the input has no sample, with a 2 ms fade at each edge, backwards where time runs backwards, as long as the video, encoded aac at the source's bit rate (else 96 kb/s a channel, at most 96 kHz) or flac with `--lossless`; stream order, channel count, language and title are kept, and the channel layout where the encoder keeps it ([timing: audio](docs/timing.md#audio)). Under `frames/2` a declared stage that changes the frame count is refused with it (exit 2). `drop` writes no audio. |
| `--device D` | | Temporal only: `cpu`, `gpu` or `gpu=N`, passed to `vfx apply --device` and to the preflight `vfx apply --check`, so a device `vfx` cannot serve fails before any decoding. A frame effect refuses it. Matches only in full: `--de` still means `--depth`. Without it the commands, plan and JSON are those of before. |
| `--depth 8\|16\|32` | 16 | Bits per channel between ffmpeg and `vfx`. |
| `--lossless` | off | Encodes with ffv1 (level 3) in the RGB the effect wrote, alpha kept, no yuv420p and no tv range, at any size. An HDR input keeps its primaries and transfer as tags. The default output becomes `.mkv`. Refused with `--depth 32`. |
| `-o PATH`, `--output PATH` | `<input stem>_<effect>.mp4` beside the input (`.mkv` with `--lossless`), with `_1`, `_2`, ... added while the name is taken | Output path. An existing path is refused unless `--force`. |
| `--force` | off | Replaces an existing `-o` path, once the new output verifies. The input, or a layer video, is never replaced. |
| `--dry-run` | off | Prints the plan and the exact commands, writes nothing, exits 0. |
| `--json` | off | Prints the result as one JSON object on stdout. |
| `--vfx PATH` | lookup below | The `vfx` executable. |
| `--ffmpeg-path PATH`, `--ffprobe-path PATH` | lookup below | The ffmpeg and ffprobe executables. |
| `--pre-stage CMD` | | A raw frame filter of another tool, run between the decode and `vfx`; repeatable, in the order given. CMD is one command line (split like a shell would, no shell involved) that reads frames of the `--depth` format on stdin and writes the same number of frames of the same format and size on stdout. `{width}`, `{height}`, `{depth}`, `{fps}` (as N/D), `{frames}` and `{pix_fmt}` in it are filled in; `{frames}` is refused when the frame count is not known. A stage that exits non zero, stops in the middle of a frame or stops reading fails the run. With a stage, `--json` adds `contract` and `stages`, and a failure prints the error object of `contracts/error.schema.json` (exit 1 under `frames/1`). With `--fps` a pre stage of a temporal effect gets the input at its own rate and count (`{fps}` and `{frames}` the input's); a post stage, and every stage of a frame effect, gets the output's. |
| `--post-stage CMD` | | Like `--pre-stage`, between `vfx` and the encode; its `{frames}` is the length of the `vfx` output (`--frames` for a temporal effect). |
| `--stage-timeout SECONDS` | 120 | Kills a `--pre-stage` or `--post-stage` that holds the pipe up this long without a byte moving on its pipes (0 never). A stage that waits for a slow neighbour is not held up. Under `frames/2` the decode and the encode are watched too; `vfx` never is. |
| `--contract frames/1\|frames/2` | `frames/1` | `frames/1` is the behaviour this section describes, unchanged. `frames/2` is the handoff contract ([Handoff](#handoff)): every `--json` failure is the error object of `contracts/error.schema.json`, a text failure reads `error: <code>: <message>`, and the exit status follows the code: 1 `stage_failed`, 2 `usage`, 3 `unsupported`, 4 `bad_input`, `broken_pipe` and `partial_frame`, 5 `stage_timeout`, 6 `tool_missing`, 130 `interrupted`. A `--pre-stage` or `--post-stage` that exits 2, 3 or 4 is `usage`, `unsupported` or `bad_input`, as `vfx` is. `--json` adds `ok`, `contract`, `stages` (every process in pipe order, then the verify step, each with `role`, `index`, `name`, `command` as an argv list, `exit_code` and `stderr_tail`) and `geometry`. `--stage-timeout` also watches the decode and the encode; `vfx` is never judged. |
| `--geometry preserve\|declared` | `preserve` | `frames/2`, after a `--pre-stage` or `--post-stage`: a `declared` stage is first asked `CMD --describe [--frames N]` (`contracts/describe.schema.json`), and its width, height and frame count go to the next process, the encode and the verify. A frame rate or depth change is refused with exit 3, and so is another value. |
| `--describe` | off | `frames/2`: prints the output geometry of the run as a describe answer (`stage` is the effect id) and exits 0 without reading or writing frames. It reports the output geometry, `--size` and `--fps` included. |

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
   | `--lossless` with `--depth 32` | `--lossless needs --depth 8 or 16: ffv1 has no float pixel format` |
   | an HDR input without `--lossless` when ffmpeg has no libx265 that writes `yuv420p10le` and `yuv444p10le` (`tool_missing`, exit 6 under `frames/2`) | `an HDR input is encoded with libx265, which <ffmpeg> does not have; use an ffmpeg built with libx265, or pass --lossless (ffv1)` |
   | `--audio retime` on an input with audio when `tools/audio_map.py` is missing (`tool_missing`) | `--audio retime needs <path>, which is missing; restore it from the repository or pass --audio copy or --audio drop` |
   | an output path that is the input or a layer video | `the output path is the input file; choose a different --output path` |
   | an existing `-o` path without `--force` | `output already exists; choose a new --output path or pass --force` |

2. Finds `vfx`, reads the effect's kind, layers and param ids from `vfx list --json --all`, checks `--layer`, `--frames` and `--device` against them, and probes every layer video (a layer gets the same no-video refusal, prefixed `layer ID:`; an HDR layer is read as stored, with its own matrix).
3. Runs `vfx apply ... --check` with the real size, depth, rate, clip rates, params, layers and `--device`, so a bad id or value, or a device `vfx` cannot serve, fails before any decoding.
4. Decodes the layers, with `--audio retime` decodes every audio stream and plays it along the time map (`tools/audio_map.py`), then runs decode, `vfx` and encode as one pipe ([ffmpeg pairing](docs/ffmpeg.md#the-applypy-pipeline)).
5. Encodes with libx264 CRF 18, yuv420p, tv range (yuv444p at the exact size when a side is odd), an HDR input with libx265 CRF 18 in 10 bit with its tags and its HDR metadata ([ffmpeg: HDR](docs/ffmpeg.md#hdr)), or with ffv1 under `--lossless`. Audio follows `--audio`: copied unchanged by default (cut at the end of the video with `-shortest` when the output is shorter, ending early when it is longer), retimed (aac, or flac with `--lossless`), or left out.
6. Writes `<output stem>.partial-<pid>.<ext>`, verifies it, then moves it onto the output path. Scratch files (layer clips, logs, the `vfx` spool) live in an `apply_*` folder in the OS temp folder, removed at exit.

A failure after step 3 ends with `[error] the pipeline failed:` and each failing stage's exit code with the last three lines of its log, `[error] audio-decode: ffmpeg exited with code N` or `[error] audio-retime: python exited with code N` with the last lines of that step's log (`stage_failed` at that stage under `frames/2`), or `[error] the output did not verify: ...`; the output path is left untouched.

**Verification** (a failure is `the output did not verify: ...`, exit 1): the requested size (`--size`, else the input's displayed size); the requested rate (`--fps`, else the input's) within 0.01 fps; without `--frames`, or with `--frames` equal to the input's frame count, a duration within one frame of the input's; with another `--frames N`, a duration within max(1/fps, 0.1 s) of N / fps and, where the container reports it, exactly N frames. With `--audio copy` the input's audio is still there with the same codec and channel count, and no audio is added; with `--audio retime` the output has as many audio streams as the input, each in the planned codec with its source's channel count and as long as the video within max(1/fps, 50 ms) (one encoder frame more under `-shortest`); with `--audio drop` it has no audio. Under `retime` and `drop` the duration is that of the video, N / fps, so an input whose audio outlasts its video passes. For an HDR input the output keeps the input's `color_transfer`, and on the lossy path an HDR input or an odd side has the planned pixel format (`yuv420p10le`, `yuv444p10le` or `yuv444p`).

**Output.** Text mode prints `[plan] <summary>`, `[temp] <path> (<what>)` and one `[cmd] <command>` per stage (the layer decodes; with `--audio retime` the audio decode and one `tools/audio_map.py` run a stream; then decode, effect, encode; runnable as printed), then `[dry-run] nothing was written` or, as the last line, `<absolute output path>: <duration>s, <W>x<H>, <fps> fps, <video codec>[, <audio codec> <N>ch]`. A plan line reads:

```
[plan] retime on clip.mp4: 160x96, 24.000 fps, 8 bpc (rgba), temporal, 95 frames, the audio ends before the video does, decode | vfx | encode -> target/demo/slow.mp4
```

After the layers the plan adds, each when it applies: `scaled from 160x96`, `converted from 24.000 fps` and `in_fps_is_out_fps=0 and input_frame_rate=24 added` (`--size`, `--fps`); `encoded yuv444p (the size has an odd side)` on the lossy path (`yuv444p10le` for HDR); for an HDR input `HDR <transfer> kept as stored (no tone mapping)`, then `libx265 10 bit`, with ` with master-display`, ` with max-cll` or ` with master-display and max-cll` when the input has that metadata, or under `--lossless` `ffv1 tagged <primaries> <transfer>`. The audio phrase comes last: `audio cut at the end of the video (-shortest)` when the output is shorter than the audio by more than a frame, `the audio ends before the video does` when it is longer, and absent otherwise; with `--audio retime` it is `audio retimed with the effect's time map (aac)` (`(flac)` with `--lossless`), followed by `audio cut at the end of the video (-shortest)` when the input does not say its frame count, and with `--audio drop` on an input with audio `audio dropped`.

`--json` prints `operation`, `input`, `effect`, `kind`, `params`, `depth`, `pix_fmt`, `size`, `fps`, `fps_exact`, `frames`, `input_frames` (all of these describe the output); with a `--size` or `--fps` other than the input's `source` (the input's `size`, `fps`, `fps_exact`, `frames` and `rounding`: `"up"` after a rate change, else null) and `auto_params` (the params the run added); `layers`; for an HDR input `hdr` (`transfer`, `primaries`, `matrix`, null in RGB under `--lossless`, and `master_display` and `max_cll`, the x265 strings written back or null); for an HDR input or an odd side on the lossy path `encode` (`codec` and `pix_fmt`); `audio` (`plan`: `shortest`, `ends-early`, `same` or null, and `shortest`; with `--audio retime` or `drop` also `mode`, `codec` (`aac`, `flac` or null) and `streams`, each with `index`, `channels`, `sample_rate`, `layout`, `bit_rate` and `offset`, the audio start minus the video start in seconds); `temp_files`, `vfx`, `summary`, `dry_run`, `commands`, `outputs` (each `path` and its ffprobe facts under `probe`); after a real run `seconds`; and `notes` only when `vfx` printed notes (each as `vfx: <effect>: <id> <title>: ...`; a run without notes and every dry run leave the key out). After a run that went through, each note is also relayed on stderr as `[note] vfx: <effect>: <id> <title>: ...`; a failed run shows them only in the log tail of its error. Errors print `[error] <message>` on stderr, and under `--json` also `{"ok": false, "error": "<message>"}` on stdout.

**Exit codes:** 0 done (or dry run), 1 any refusal or failure, 2 a flag argparse rejects (for example `--frames 0`), 130 interrupted.

### Helper tools

**`tools/speed_curve.py`** checks a `speed_percent` curve without rendering, with the effect's own arithmetic (half-frame sums in 32-bit floats, the default time base), so its positions agree with a render to a few thousandths of a frame. It models Speed mode (`time_remap_mode=1`) on a clip that plays at the output rate, from clip frame 0 ([what it models](docs/timing.md#what-speed_curvepy-models)).

| Flag | Default | Meaning |
|---|---|---|
| `--keys K` | required | The keys as for `--param speed_percent=...`, `h` and `e` suffixes and `iS/I` `oS/I` handles included (handle speeds in percent points a second, read with the time base of `--fps`); a leading `speed_percent=` is accepted. Speeds -10000 to 10000; a handle that overshoots is held inside that range, as the effect holds it. A string that starts with `-` needs the `=` form: `--keys="-100@0"`. |
| `--fps R` | required | Clip and output rate: `24`, `60`, `30000/1001`. Ticks follow the fraction, so `29.97` (100 ticks a frame) and `30000/1001` (1001) place fractional keys differently. |
| `--clip-frames N` | required | Frames in the clip the effect reads, at least 1. |
| `--frames N` | the frame of the last key, rounded | Output frames to check. |
| `--start SECONDS` | none | Where the clip starts in the recording; source and mark times become recording times. |
| `--mark NAME=SECONDS` | none | Repeatable: a source moment to find in the output. |

It prints `speed curve: <K> keys, <R> fps (<T> ticks a frame)`, `output ...` and `clip ...`, a key table (`key frame`, `time`, `speed`, `ease`, `source frame`, `source time`, plus `recording time` with `--start`; `ease` is `-`, `ease` or `hold`, or a key's handles as written, as in `i-30/50 hold`, and widens for them), then:

| Line | Says |
|---|---|
| `last output frame N shows source frame P (T s into the clip)` | Where the curve ends. |
| `speed   A% to B%: ...` | Time below, above and at 100%; `at the slowest, one source frame spans X output frames`. |
| `ends    T s at a constant S%` | The final steady run. |
| `tail    ...` | Printed when the final speed is a whole multiple of 100% other than 0: whether that run shows whole source frames, and if not a corrected `--keys "..."` that moves the keys from the last ramp-out on by up to three frames ([exact frames](docs/timing.md#exact-frames)); a segment between two keys of the final speed is part of the run only when no handle there has a speed. |
| `frames  K of N (P%) need interpolation, about M min at 1080p` | Output frames more than 0.005 from a whole source frame, timed at 0.25 s each. |
| `change  largest speed step between frames: ...` | The sharpest speed change. |
| `marks` | One line per `--mark`: its clip frame and the output frames it shows at (`then` for a later pass, `outside the clip`, or `never reached by output frames 0 to N`). |
| `warnings: none`, or `warnings` and a list | Output frames that read outside the clip (`output frames A to B read source frames outside the clip (0 to C): positions P to Q`), plain keys (no `h`, `e` or handle) that bend the speed by 1 point a frame or more (`a trailing e would round it`), a last key after the last output frame. |

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

**`tools/audio_map.py`** is the audio half of `--audio retime`, and prints where each output frame lands in the clip without rendering anything ([timing: audio](docs/timing.md#audio)).

| Command | Does |
|---|---|
| `python -I tools/audio_map.py render --effect E --fps R --frames N --clip-frames C [--clip-fps R] [--param K=V ...] --rate SR --channels CH [--offset S] --in A.f32 --out B.f32` | Plays one raw `pcm_f32le` stream along the time map of effect E with its `--param` values (ids the map does not read are passed over): N output frames at R over a clip of C frames at `--clip-fps` (default R), SR samples a second, CH channels, S the audio start minus the video start in seconds. `tools/apply.py` runs it once a stream and prints it as a `[cmd]` line. |
| `python tools/audio_map.py positions --effect E --fps R --frames N [--clip-fps R] [--param K=V ...] [--json]` | Prints one line per output frame, the frame and the clip position it shows (`47 29.375000` is the last of `--effect retime --fps 24 --frames 48 --param "speed_percent=100@0;50@24"`); `--json` adds the knots of the map. |

Exit codes: 0 fine; 1 a value the map cannot read, or a file it cannot read or write; 2 a malformed command line.

**The parity record** ([parity](docs/parity.md)) is kept by three tools:

| Command | Does |
|---|---|
| `python -I tools/gen_docs.py` | Fills every generated figure and table of `README.md` and `docs/*.md` from the snapshots in `docs/data/` and scans of the tree (hex fixtures, test counts), and writes `docs/data/facts.json`: every key with its value and source. |
| `python -I tools/gen_docs.py --check` | Regenerates in memory and exits 1 on drift or an unknown key; it runs no process. |
| `python -I tools/gen_docs.py --refresh-data` | Rebuilds `docs/data/controls.tsv` and `docs/data/gaps.tsv` from `vfx list --json --all` (`--vfx` or `VFX_PATH`), with `--testset DIR` also `docs/data/references.tsv` from a family set (every case `missing`) and with `--refs DIR` as well its grades, then fills. |
| `python -I tools/check_docs.py` | Checks the generated figures, relative links and anchors, inline repository paths, ASCII, dashes and, with `VFX_PATH` set, every flag table against `--help`; `--list NAME` and `--json` print the findings. |
| `python tools/parity_lock.py base\|corpus\|refs\|check\|refused\|record` | The regression lock: builds a baseline ref in a temporary worktree, renders its corpora, checks a new `vfx` against it byte for byte and records `docs/data/lock.json` ([parity: regression lock](docs/parity.md#regression-lock)). |

The handoff tools, `tools/handoff.py`, `tools/conformance.py` and `tools/ffstage.py`, are in [Handoff](#handoff).

### Comparison harness

`target/release/fx-parity` builds a test set of clips (temporal effects) or stills (frame effects), writes a checklist for rendering references with another program, compares `vfx` with those references, and checks `vfx` for determinism. `fx-parity --help` prints the full rules.

| Subcommand | Reads | Writes | Key flags (default) |
|---|---|---|---|
| `gen --out DIR` | `vfx list --json --all`; ffmpeg (`FFMPEG`, else `ffmpeg` on `PATH`) renders the inputs | `DIR/inputs/<source>-<W>x<H>.d<depth>[.f<frames>].<pix_fmt>` and `DIR/cases.json` (format 1); prints `wrote DIR/cases.json: C cases for E effects, I inputs` | `--effects ID,...` (every listed effect; a hidden one only when named), `--sizes WxH,...` (clips 16x12,37x23, stills 1x1,2x3,37x23,255x129,320x180; the flag sets both), `--depths` (8,16,32), `--frames` (12), `--fps` (24,30000/1001, taken in turn by the clip sources), `--clip-sources` (testsrc2,movegrad,noisefx,alphaslide), `--clip-sweep-sources` (testsrc2,alphaslide; `all`), `--sources` and `--sweep-sources` (stills gradient,bars,noise,alpha,range; sweeps noise,alpha), `--no-sweep`, `--max-cases-per-effect` (400; 0 keeps all), `--jobs`, `--vfx` |
| `gen --out DIR --families ID,...\|all` | `vfx list --json --all`; ffmpeg renders the family clips | a family test set instead of the matrix: `DIR/inputs/...` and `DIR/cases.json` (format 1, each case with its `family`); prints the `wrote` line, then `families: F with C cases (gap-shrink 10, ...)` | `--effects ID,...` (the families of those effects), `--jobs`, `--vfx`; no matrix flag is accepted with it |
| `checklist --testset DIR` | `cases.json`, `vfx list --json` | `DIR/CHECKLIST.md` (or `--out FILE`); prints `wrote <file>: C cases for E effects` | `--out`, `--vfx` |
| `run --testset DIR --refs DIR` | `cases.json`, the inputs, the references `<case id>.<pix_fmt>` in `--refs` | nothing: a report on stdout | `--effect ID`, `--brief` (failures only), `--json`, `--strict` (a missing reference fails), `--jobs`, `--vfx` |
| `selfcheck --testset DIR` | `cases.json`, the inputs | nothing: a report on stdout | `--effect ID`, `--max-cases` (24, spread over the effects; 0 all), `--threads N,...` (1,3), `--json`, `--jobs`, `--vfx` |
| `devcheck --testset DIR [--testset DIR...]` | `cases.json`, the inputs | nothing: a table on stdout | `--effect ID`, `--max-cases` (0: every temporal case; per test set), `--devices gpu=N,...` (every device `vfx devices` lists as ok), `--json`, `--vfx` |

- Cases: a temporal effect gets `all-default`, `all-min`, `all-max` and, when it has an optional layer, `no-layers` on every clip source, size and depth; without `--no-sweep` also one param at a time and a two key ramp per `float`, `angle` and `point` param on the sweep sources. When `--max-cases-per-effect` trims an effect, those four settings and one case for each param's min and max always stay (so a cap below their count removes none of them), then one case per other setting, then a fixed pseudo random fill. Every case except `no-layers` gets a clip of another source for every layer the effect lists. A case id reads `<effect>.<label>.<source>-<W>x<H>.d<depth>.<hash>`.
- Families (`gen --families`): every stated rule of the [limits](#limits) gets the cases that would settle it once they are rendered in the original: `gap-shrink` (G1), `gap-maskbm` (G1b), `gap-order` (G1c), `gap-merge` (G1d), `gap-vecmatte` (G1e), `gap-blurflow` (G1f), `gap-groups` (G1g, G1h), `gap-points` (G2), `gap-drawgeom` (G3), `gap-debug` (G4), `gap-tracker2` (G5), `gap-wrap` (G6), plus two that only observe and print no note, `gap-tiny` (frames of 1x1 to 5x5) and `gap-keys` (ease handles on Frame Num, Speed %, Sensitivity and a point). Their cases have 12 frames at 24 fps, mostly 64x48 at 8 bpc, on clips of their own made with exact arithmetic (a textured disc moving over a still background with its hard and soft mattes, two crossing discs, the disc's vector images, a texture periodic across the width, and others), kept out of the matrix's clip rotation, so a set made without `--families` is byte for byte what it was. A family case's id reads `<effect>.<family>-<label>.<source>-<W>x<H>.d<depth>.<hash>`.
- With family cases in the set, `checklist` adds a gaps table (family, gap, open point, rule, cases), the family clips, a `- settles:` line per case and notes on mattes (a hidden layer picked in the FG Matte popup), points (layer pixels, Linear spatial interpolation) and ease handles (the Keyframe Velocity dialog); `run` adds `families.<id>` with `cases`, `compared`, `exact`, `failed`, `missing` and `settled` to the JSON report and a `families` block to the text report, keeps each family case's `vfx: note:` lines (`notes` in the JSON), and warns when a family case printed no note for the gap it is there to settle.
- `vfx` lookup: `--vfx`, `VFX_PATH`, next to `fx-parity` (or one folder up), then `PATH`. `--jobs` defaults to the number of cores, at most 8.
- Pass rule: 8 and 16 bpc bit exact, 32 bpc within 1 ULP, clips frame by frame with the first failing frame reported. Every compared case also gets a mean error (mean absolute difference over all pixels and all 4 channels, as a percentage of full scale), a clip its worst frame, and the report a `graded error` block per effect and the 10 worst cases.
- `selfcheck` runs each picked case twice at the default thread count and once per `--threads` value (through `RAYON_NUM_THREADS`); every output must equal the first byte for byte.
- `devcheck` runs each case with `--device cpu` and with every device, one device process at a time, and compares the output clips byte for byte, the exit codes and stderr minus the device's own note lines; it prints per device the cases, identical, different, the cases and stages that fell back to the CPU and the refusals.
- Exit codes: 0 fine; 1 mismatch (`run`: a `FAIL`, or under `--strict` a missing reference; `selfcheck`: a difference or a failed run; `devcheck`: a difference); 2 usage; 3 setup or I/O (for example a references folder that does not exist). Errors print `fx-parity: error: <message>`. There is no `--version`.

## Handoff

twixtor runs sapphire's stages and its own around `vfx` and hands tasks to sapphire as packets, under the convention both repositories share ([handoff](docs/handoff.md)). It finds sapphire by sapphire's marker files: `--peer-dir sapphire=DIR`, else `HANDOFF_SAPPHIRE_DIR`, else the one sibling folder that holds them.

| Command | Does |
|---|---|
| `python -I tools/handoff.py stages --json` | Lists the stages of twixtor and sapphire, each with the command line a wrapper takes. |
| `python tools/apply.py INPUT EFFECT --pre-stage CMD` | Runs a listed stage before `vfx`, or after it with `--post-stage`, between an ffmpeg decode and encode ([Wrapper](#wrapper)); take CMD from `stages --json`. |
| `python tools/ffstage.py list --json` | Prints the manifest of the `ffmpeg-filter` stage in Python, which `stages` does not list. |
| `python -I tools/handoff.py new --to sapphire --task TEXT --input PATH` | Writes a task packet into sapphire's `handoff/` folder; sapphire runs `accept ID` and `done ID --output PATH --run SPEC`, and `verify ID --other` checks the result from here. |
| `python -I tools/handoff.py doctor` | Checks sapphire, ffmpeg, ffprobe, `vfx`, the contracts and the shared files. |
| `python -I tools/conformance.py --quick` | Runs the covering conformance subset in about a minute; `--full` runs the whole matrix and `--stage CMD` one stage. |
| `python -I tools/handoff.py sync --check` | Compares the shared files with sapphire's by SHA-256; `sync` copies twixtor's versions over. |

The `shared` list of `contracts/tool.json` is byte equal in both repositories: the schemas in `contracts/`, `contracts/conformance/cases.json`, `docs/handoff.md`, `docs/handoff-bench.tsv`, `tools/handoff.py`, `tools/conformance.py` and their tests. Change one here, then run `sync`, also after `tools/conformance.py --bench`, which rewrites the bench table.

twixtor's own stages are the effects of `vfx list --json --all`; under `--contract frames/2`, `tools/apply.py` speaks the convention in full.

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

95 frames is the longest render at 50% that stays inside the 48-frame clip ([output length](docs/timing.md#output-length)). The 2 s tone is copied, not stretched, so it ends before the video; `--audio retime` would play it along the 50% time map instead.

## Comparing with reference renders

Optional end to end check. Nothing else in this repository needs it: building, `vfx`, `tools/apply.py`, the tests and CI all run with no host application. Only a 1:1 claim ([Parity](#parity)) waits on it, because a reference render has to come from the original plugin.

`fx-parity` measures how far `vfx` is from another program's renders of the same cases, and whether `vfx` gives the same bytes on every run and thread count. A small set for one effect at 8 bpc:

```sh
target/release/fx-parity gen --out target/demo/pack --effects retime --depths 8 --no-sweep --sizes 160x96 --frames 12
target/release/fx-parity checklist --testset target/demo/pack
target/release/fx-parity selfcheck --testset target/demo/pack
mkdir -p target/demo/pack/refs target/demo/pack/export
target/release/fx-parity run --testset target/demo/pack --refs target/demo/pack/refs --brief
```

1. `gen` prints `wrote target/demo/pack/cases.json: 16 cases for 1 effects, 4 inputs`: 4 settings on 4 clip sources.
2. `checklist` writes `target/demo/pack/CHECKLIST.md`. Per case: the size, depth, frame count and rate, the input files, the settings by the names the interface shows (defaults left out, keys written `100 at frame 0, 50 at frame 24`, ease handles with their velocity a second and influence for the Keyframe Velocity dialog), the extra layers, the reference file to produce, and two ffmpeg commands: one turns the raw inputs into PNG (8 and 16 bpc) or float EXR (32 bpc) sequences, the other turns the other program's export (in `export/`) into the raw reference (in `refs/`). The 32 bpc command divides the EXR's premultiplied colour by alpha again.
3. `selfcheck` needs no references. It ends `16 cases checked, 16 identical, 0 DIFFER` and exits 0.
4. `run` prints a line per case: `pass`, `no reference`, or `FAIL` followed by the case's settings, input and reference paths, a `repro:` line with the exact `vfx apply` command, the first failing frame and the first differing pixels (`x=20 y=10 in=[...] expected=[...] actual=[...]`). Then a `graded error` block per effect (cases compared, how many bit exact, mean error, worst case), the worst cases, and `P pass, F FAIL, N no reference, T cases`. With the empty `refs/` it ends `0 pass, 0 FAIL, 16 no reference, 16 cases` and exits 0; `--strict` makes that exit 1.

On a set of all four effects, every case but `no-layers` hands `create-vectors`, `retime-fgbg` and `vectors-in` their `fg1_matte` to `fg3_matte` layers, which they render on the layered path, with the notes of the gaps they reach. A `--no-sweep` 160x96 set of all four effects (`gen --out DIR --depths 8 --no-sweep --sizes 160x96 --frames 12`: 64 cases) ends `selfcheck` with `24 cases checked, 24 identical, 0 DIFFER` (exit 0). Pass `--effect ID` to `run` or `selfcheck` to check one effect.

The reference case families are built to settle the [limits](#limits):

```sh
target/release/fx-parity gen --out target/demo/families --families all
target/release/fx-parity checklist --testset target/demo/families
target/release/fx-parity selfcheck --testset target/demo/families --max-cases 0
```

`gen` ends `families: 14 with 117 cases (gap-shrink 10, gap-maskbm 2, gap-order 10, gap-merge 5, gap-vecmatte 3, gap-blurflow 2, gap-groups 10, gap-points 6, gap-drawgeom 18, gap-debug 10, gap-tracker2 6, gap-wrap 8, gap-tiny 20, gap-keys 7)`, and `selfcheck` `117 cases checked, 117 identical, 0 DIFFER`. Render the checklist once in the original, convert the exports into `refs/` with its commands and run `fx-parity run --testset target/demo/families --refs target/demo/families/refs --json`: a family is settled when its `families.<id>.settled` is true, every case compared and bit exact. Its gap then retires: its entry in `plugins/retime-engine/src/gaps.rs` and its `record` calls go, so does its entry in [parity: gaps register](docs/parity.md#gaps-register), and `python -I tools/gen_docs.py --refresh-data` with `--testset` and `--refs` moves the figures ([parity: moving the numbers](docs/parity.md#moving-the-numbers)). `gap-tiny` and `gap-keys` print no note; their references decide whether the stated rule stays.

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
python -I tools/gen_docs.py --check
python -I tools/check_docs.py
python -I tools/test_docgen.py
python tools/test_parity_lock.py
python tools/test_audio_map.py
python tools/test_apply_hdr.py
python tools/test_apply_geometry.py
```

- **Release mode** because the optical flow and thread invariance suites are far too slow unoptimised. The end to end tests (`crates/fx-cli/tests/cli.rs`, `plugins/*/tests/cli.rs`) run the `vfx` that cargo builds in the same profile.
- **ffmpeg:** the plugins' end to end tests and `crates/fx-parity/tests/selftest.rs` use ffmpeg where they need a real clip and skip those parts, with a note, without it. `tools/test_apply.py` (which also runs `--audio`, `--device` and the notes end to end) and `tools/test_apply_geometry.py` need ffmpeg, ffprobe and a built `vfx` (`target/release` or `VFX_PATH`) for their runs and skip them when one is missing; `tools/test_apply_hdr.py` runs its unit tests anywhere, its end to end tests with ffmpeg, ffprobe and `vfx`, and its HDR ones only when that ffmpeg has libx265 with 10 bit (they skip with a note otherwise); `tools/test_repeat_keys.py` needs ffmpeg for its one end to end test; `tools/test_speed_curve.py` and `tools/test_audio_map.py` need neither.
- **OpenCL:** the device tests (`plugins/retime-engine/tests/device_cl.rs` and the device stage suites) run on the machine's OpenCL devices and skip without one; the line for line Rust transcription of every kernel runs on the work item simulator and needs no device, and a lint checks every kernel source. The end to end byte gate, `fx-parity devcheck`, needs a device and is run by hand, not in CI.
- **Evidence:** hand worked values, goldens and model fixtures from independent numpy float32 scripts, analytic ground truth (a texture moved by a known amount), and identical output across thread counts.
- **Counts:** <!--f:tests_rust-->1,469<!--/f--> Rust tests (`#[test]`, <!--f:tests_rust_ignored-->15<!--/f--> of them ignored) and <!--f:tests_python-->592<!--/f--> Python tests (`def test_`), counted in the source; per suite in [parity: test counts](docs/parity.md#test-counts).
- **Parity record:** `tools/gen_docs.py --check` fails when a generated figure drifts from `docs/data/` and the tree, `tools/check_docs.py` checks the documentation (its flag tables only with `VFX_PATH` set), and `tools/test_docgen.py` and `tools/test_parity_lock.py` test both tools and the regression lock with a fake `vfx` ([parity](docs/parity.md)).
- **Timing probes** are `#[ignore]`d: `cargo test --release -- --ignored <name> --nocapture` with `bench_1080p`, `timing_full_hd`, `timing_probe_1080p`, `timing` or `breakdown`.

**Fixtures.** The model fixtures are hex text, 64 bytes a line, which the tests decode; the repository holds no binary files. Each generator needs numpy (`gen_bezier.py` only the standard library) and writes to the path given (most default to overwriting their fixture in place; `gen_bezier.py` always does). Run it from its own folder and compare, for example `python -I -B gen_warp.py /tmp/warp_cases.hex && cmp /tmp/warp_cases.hex warp_cases.hex`. Every generated fixture below came back byte for byte (numpy 2.5.1).

| Fixture | Generator | Command |
|---|---|---|
| `crates/fx-core/tests/fixtures/bezier_cases.hex` (Bezier key handles, through `tools/speed_curve.py`) | `gen_bezier.py` | `python -I gen_bezier.py` |
| `plugins/retime-engine/tests/fixtures/f1_cases.hex` (Smart Blend, hole fill) | `f1_gen.py` | `python -I -B f1_gen.py [OUT]` |
| `.../fixtures/f2_cases.hex` (forward warp) | `f2_gen.py` | `python -I -B f2_gen.py [OUT]` |
| `.../fixtures/blur_cases.hex` (motion blur) | `gen_blur.py` | `python -I -B gen_blur.py [OUT]` |
| `.../fixtures/wrap_cases.hex` (wrap-around: circular blur, period keeping resample, hole fill, motion blur) | `gen_wrap.py` | `python -I -B gen_wrap.py [OUT]` |
| `.../fixtures/matte_cases.hex` (FG mattes: extract, shrink and grow, tracking mattes, OVER weights, premultiply, OVER) | `gen_matte.py` | `python -I -B gen_matte.py [OUT]` |
| `.../fixtures/mask_cases.hex` (the solver's layer masks, user displacement and wrap) | `gen_mask.py` | `python -I -B gen_mask.py OUT` |
| `.../fixtures/merge_cases.hex` (the matte weighted merge of `create-vectors`) | `gen_merge.py` | `python -I -B gen_merge.py [OUT]` |
| `.../fixtures/debug_cases.hex` (the Debug views) | `gen_debug.py` | `python -I -B gen_debug.py [OUT]` |
| `.../fixtures/vectors_cases.hex` (vector image coding) | `gen_vectors.py` | `python -I -B gen_vectors.py [OUT]` |
| `.../fixtures/warp_cases.hex` (inverse warp) | `gen_warp.py` | `python -I -B gen_warp.py [OUT]` |
| `.../fixtures/flow_a.hex` to `flow_f.hex` (optical flow, a differential model that is not bit exact) | `gen_flow_fixtures.py` with `flow_ref.py` | `python -I -B gen_flow_fixtures.py OUT_DIR` |
| `plugins/retime/tests/common/goldens.rs`, `plugins/retime-fgbg/tests/common/goldens.rs` | `gen_goldens.py` beside each | `python -I -B gen_goldens.py OUT` |

The Gaussian and binomial kernel tables are constants in the test of `flow/conv.rs`, checked against the code bit for bit.

**Continuous integration.** `.github/workflows/ci.yml` (Linux, macOS and Windows, Python 3.12) and `.forgejo/workflows/ci.yml` (Linux in the `node:24-bookworm` container, with Rust from rustup and Debian's `python3`) install ffmpeg (a static 9.0 build on Linux) and run the commands above, with the release profile's LTO off and 16 codegen units to cut compile time, on every push to `main` and every pull request, with a 60 minute limit. sapphire is not checked out there, so `--allow-absent-peer` lets through only the conformance parts that need sapphire; every failed check still fails the job. Both can also be started by hand. A new push to a pull request cancels its run in flight; each push to `main` keeps a run of its own.

## Documentation

This README specifies the programs. The files in `docs/` specify behaviour in detail; each covers one concern and fits in one read. Places are cited as `file.md#anchor`. Where any of them disagrees with `vfx list --json`, the listing is right.

| File | Specifies |
|---|---|
| [`docs/parameters.md`](docs/parameters.md) | Every control of the four effects: values, defaults, ranges, what each changes, how the effects differ, the refusals left. |
| [`docs/timing.md`](docs/timing.md) | How output frames map to source frames: Speed and Frame Number modes, key shapes and handles, output length, clip edges, exact frames, the rate model, the audio map. |
| [`docs/ffmpeg.md`](docs/ffmpeg.md) | How `vfx` and `tools/apply.py` pair with ffmpeg: raw frames, the wrapper's commands, rate, size, colour and HDR, audio, encoding, scratch and disk, input requirements. |
| [`docs/engine.md`](docs/engine.md) | How an output frame is computed: motion search, warps, blends, FG matte layers, hole fill, blur compensation, vector images, the OpenCL device, determinism, cost, and the module that implements each stage. |
| [`docs/parity.md`](docs/parity.md) | What has and has not been verified against the original, and how a future comparison would be graded: the unit counted, the tiers of evidence, the gaps register and the rules without a note, the model fixtures, the regression lock and how to reproduce it, caveats, terms, and how a reference render moves the numbers. |
| [`docs/handoff.md`](docs/handoff.md) | How twixtor, sapphire and ffmpeg pass raw frames to each other: contract versions, the frame contract, errors and exits, reports, discovery, geometry, the ffmpeg filter stage, packets, commands, platforms, known gaps. |
| [`docs/handoff-bench.tsv`](docs/handoff-bench.tsv) | The bench table of `tools/conformance.py --bench`: the wrappers timed on chained stages. |

## Legal

- **Independent work.** This project is written independently and contains none of Twixtor's code, binaries, assets or documentation.
- **No affiliation.** It is not affiliated with, endorsed by or sponsored by RE:Vision Effects.
- **Trademarks.** Twixtor is a trademark of RE:Vision Effects. It appears here only to name the plugin this project reimplements.
- **Interoperability.** Parameter ids, display names and values are meant to follow the plugin's interface, so settings can carry over.
- **No warranty.** Everything here is provided as is under the MIT License, without warranty of any kind and without liability.
- **Your responsibility.** You need the rights to the media you process, and you stay bound by the licences of any software you use alongside this project, ffmpeg included.
- **Rights holders.** If you believe something here infringes your rights, email [hi@bairon.dev](mailto:hi@bairon.dev) or [open an issue](https://github.com/baairon/twixtor/issues). Every concern gets a prompt answer in good faith.

## License

MIT, see [LICENSE](LICENSE).
