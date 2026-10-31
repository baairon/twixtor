# twixtor

An independent open source reimplementation of the four effects of the Twixtor retiming plugin. Tested against the original plugins running in an emulator, it gives identical output, bit for bit, on all <!--f:cases_compared-->2,357<!--/f--> test cases compared ([parity](#parity)). It is not affiliated with or endorsed by RE:Vision Effects, which owns the Twixtor trademark.

Slow motion, speed ramps and frame interpolation from optical flow, written in Rust and paired with ffmpeg.

## Contents

- [What it is](#what-it-is)
- [Parity](#parity)
  - [How to run the verification](#how-to-run-the-verification)
  - [Status](#status)
  - [Assumptions and stand-ins](#assumptions-and-stand-ins)
- [Limits](#limits)
  - [Not offered](#not-offered)
- [Legal](#legal)
- [License](#license)

## What it is

Slow a clip down by repeating frames and it stutters. twixtor works out how every pixel moves from one frame to the next (optical flow) and draws the frames in between that were never shot. The speed can be keyframed, so a shot can ramp into slow motion and back out.

The renderer, `vfx`, is a command line program that reads raw frames, runs one effect and writes raw frames. `tools/apply.py` puts it between an ffmpeg decode and an ffmpeg encode, so you hand it a video file and get a video file back. There are four effects: `retime` slows or speeds up a clip, `retime-fgbg` does the same with up to three separately tracked foreground layers, `create-vectors` writes the motion out as images, and `vectors-in` retimes from those images. Their settings use the original plugin's names and values.

You need a stable Rust toolchain (1.82 or newer), Python 3, and ffmpeg with ffprobe. From the repository root:

```sh
cargo build --release -p fx-cli
python tools/apply.py clip.mp4 retime --param speed_percent=50 --frames 95 --depth 8 -o slow.mp4
```

The second command plays `clip.mp4` at half speed. The output keeps the input's frame count unless `--frames` says otherwise; at 50%, a 2 second clip at 24 fps (48 frames) fills 95 output frames ([output length](docs/timing.md#output-length)). On the test clip of the [smoke test](docs/reference.md#smoke-test), the last line printed ends `slow.mp4: 3.96s, 160x96, 24 fps, h264, aac 1ch`. The audio is copied as it is; `--audio retime` plays it along the new timing instead.

A speed ramp is a keyed value, such as `--param 'speed_percent=100@0e;50@24e;100@48e'` (full speed, eased down to half at frame 24, back up by frame 48), and [`tools/speed_curve.py`](docs/reference.md#helper-tools) shows what a curve does before you render it.

Everything else is in the [reference](docs/reference.md): what each effect does, the crate map, the build, every command line with its flags, formats and exit codes, the handoff to the sibling project sapphire, the smoke test and the tests. Scripts and agents should start at its [sources of truth](docs/reference.md#sources-of-truth); `vfx list --json` is right wherever a document disagrees with it. The other docs each cover one topic:

- [parameters](docs/parameters.md): every setting of the four effects and what it changes.
- [timing](docs/timing.md): which source frame each output frame shows, keyframes, output length, audio.
- [ffmpeg](docs/ffmpeg.md): the commands `tools/apply.py` runs, frame rate, size, colour, HDR and encoding.
- [engine](docs/engine.md): how one output frame is computed, the optional GPU path, and what it costs.
- [parity](docs/parity.md): how parity is defined and measured, and what the checks found; the [parity log](docs/parity-log.md) has the dated record.
- [handoff](docs/handoff.md): how twixtor, sapphire and ffmpeg pass frames and tasks to each other.

## Parity

**<!--f:modules_match-->6<!--/f--> of <!--f:modules_total-->6<!--/f--> modules match the original plugins on the emulator run: every case compared is identical, bit for bit.** The four original plugins run unmodified in a CPU emulator, driven by a mock After Effects host. The same inputs go to them and to this rewrite, and the outputs must be identical, down to the sign of a zero and the bits of a NaN. The cases are split into six modules, each one part of one plugin's behavior, about 2,500 cases in all. The [status table](#status) has the counts.

Exceptions:

- **Cases without a reference.** Some in-contract cases have no reference output yet: frames the emulated original stops on that the tiny-frame rule's 5x5 bound does not reach, `retime-fgbg` cases that crashed in the emulator, and two cases too long to render in full.
- **13 cases out of contract.** In them the original leaves one value uninitialised, so there is nothing defined to match ([parity: out of contract](docs/parity.md#out-of-contract)).
- **One known difference outside the grid.** `retime-fgbg` draws its Draw Geom points as a disc where the original draws a cross ([Limits](#limits), G3).

What the results do not show:

- **The grid is a sample.** It covers chosen inputs, not every input.
- **The reference is the original in the emulator,** not After Effects on real hardware. Where the original depends on the host, both sides use a labelled stand-in ([below](#assumptions-and-stand-ins)).
- **Only the CPU path is compared.** `--device gpu` gives the CPU's bytes: stages whose kernels follow older rules run on the CPU.

Each module is checked three ways:

- **The grid:** at least 150 cases with edge values.
- **The call record:** what each side asks for, which frames and which settings.
- **The mutants:** deliberately broken copies of the rewrite that the grid has to catch.

The method, the root causes found and the definitions are in [parity](docs/parity.md).

### How to run the verification

1. Build the rewrite: `cargo build --release -p fx-cli -p fx-parity` ([build](docs/reference.md#build)).
2. Supply what the repository does not hold.
   - The original plugins and the Windows runtime DLLs for the emulator's root file system come from your own installation; `parity/harness/needed_dlls.txt` lists the DLLs used.
   - Adobe's After Effects SDK is not redistributable; obtain it yourself. Then write `parity/harness/sdk_layouts.json`, which the harness loads, on Linux with pyelftools. Run `gcc -c -g -D__ANDROID__ -I SDK_HEADERS -o sdk_probe.o parity/harness/sdk_probe.c`, then `python3 parity/harness/gen_layouts.py SDK_HEADERS sdk_probe.o parity/harness/sdk_layouts.json`, where SDK_HEADERS is the SDK's `Headers` folder.
   - `parity/harness/gen_aegp_suites.py` builds `parity/harness/sdk_layouts_extra.json` from the header text. It covers the suite revisions that only `AE_GeneralPlugOld.h` holds.
3. Run the original plugins under the harness on a module's test set to get the reference outputs ([harness README](parity/harness/README.md)).
4. Run the rewrite on the same cases with `VFX_CALL_RECORD=<file>` set, which logs what the render reads ([call record](docs/reference.md#call-record)).
5. Grade with `fx-parity run --max-ulp 0 --json`, or with the module's `phase_b.py`, which runs it. Without `--max-ulp 0`, 32-bit floats may differ by one step, which is looser than the bar.
6. `python -I tools/gen_docs.py` rewrites the table below from the result files; `--check` fails when the table is stale.

### Status

<!--f:modules_match-->6<!--/f--> of <!--f:modules_total-->6<!--/f--> modules match the original on the emulator run. Verified: <!--f:modules_verified-->1<!--/f--> of <!--f:modules_total-->6<!--/f-->.
- **Matches:** at least 150 cases compared, none differing, and every call record the same on both sides.
- **Verified** adds every in-contract case compared and every mutant caught.

Re-run verified then says whether an independent re-run reproduced the result.

<!--gen:modules-->
| Module | Plugin | Status | Cases compared (in contract) | Out of contract (candidates) | Values compared (bytes / floats) | Differing | Max ULP | Call records matching | Mutants killed | Re-run verified | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `retime-time` | `Twixtor_64.aex` | Matches | 460 of 462 | 12 | 11,203,140 / 3,626,220 | 0 / 0 | 0 | 460 of 460 | 21 of 26 | - | `parity/retime-time/results/summary.json` |
| `retime-flow` | `Twixtor_64.aex` | Matches | 330 of 337 | 13 | 4,977,552 / 1,174,560 | 0 / 0 | 0 | 330 of 330 | 22 of 22 | - | `parity/retime-flow/results/summary.json` |
| `create-vectors` | `CreateMotionVectors_64.aex` | Matches | 418 | 48 | 892,476 / 335,208 | 0 / 0 | 0 | 418 of 418 | 47 of 51 | - | `parity/create-vectors/results/summary.json` |
| `vectors-in` | `TwixtorProVectorsIn_64.aex` | Verified | 443 | 38 | 1,384,032 / 370,960 | 0 / 0 | 0 | 443 of 443 | 29 of 29 | - | `parity/vectors-in/results/summary.json` |
| `fgbg-core` | `TwixtorPro_64.aex` | Matches | 353 of 367 | 55 | 2,413,984 / 565,200 | 0 / 0 | 0 | 360 of 360 | 29 of 31 | - | `parity/fgbg-core/results/summary.json` |
| `fgbg-layers` | `TwixtorPro_64.aex` | Matches | 353 | 65 | 1,952,316 / 248,016 | 0 / 0 | - | 353 of 353 | 42 of 50 | - | `parity/fgbg-layers/results/summary.json` |
| Total | - | 6 of 6 match, 1 of 6 verified | 2,357 of 2,380 | 231 | 22,823,500 / 6,320,164 | 0 / 0 | 0 | 2,364 of 2,364 | 190 of 209 | 0 of 6 | - |
<!--/gen:modules-->

How to read the table:

- **Cases compared** reads "X of Y": X compared out of the module's Y in-contract cases. The rest have no reference output, and while any are left the row cannot be Verified.
- **Out of contract** counts the cases the original does not define.
- **Mutants killed** counts a mutant only when the comparison with the original catches it. A survivor also keeps the row from Verified. Each survivor is listed in the [per-module notes](docs/parity.md#per-module-notes), and the module's `results/kills.json` records what it changes.

Every column is explained in [parity: status](docs/parity.md#status).

### Assumptions and stand-ins

A stand-in is labelled data that the harness feeds to both sides where After Effects or Windows would supply something the harness cannot. A match that rests on one is called logic parity. The full list is in the [harness README](parity/harness/README.md); in short:

- **Keyframe interpolation** is done by the harness (linear, hold and ease; Bezier refused), so keyed cases reach logic parity at most.
- **FG matte layers** are handed to the plugin as ordinary layers. That matches how it reads mattes of the frame's size, unmasked and not time remapped.
- **Host answers** follow the After Effects SDK headers. Three of them matter here:
  - the host is an interactive one, not a render engine;
  - a checkout outside the clip returns an empty result at the layer's size;
  - an unset layer whose registered default is the effect's own layer gets the input clip.
- **The emulated CPU** starts with the Windows floating point state. Threads switch only when one waits, and the main thread's stack is 1 MiB.
- **A known fidelity risk:** Windows functions the emulator has no hook for, outside the C runtime DLLs, return 0.
- **NaN payloads.** The emulator picks the surviving NaN of two NaNs differently from a real x86 processor. The rewrite follows the emulator by default, because the references come from it ([parity: the NaN rule](docs/parity.md#the-nan-rule)).
- **Not graded,** because the original has nothing to compare them with: the wrapper's own features (retimed audio, a new size or frame rate, HDR input, odd frame sizes), Bezier key handles, and the `--device` choice ([rules without a note](docs/parity.md#rules-without-a-note)).

## Limits

Four points still run a rule of this project somewhere. A render that uses one prints a `vfx: note:` line on stderr ending `(README: limits)`; a note never changes the frames or the exit code. Everything else follows the original and matches it on the grid, and prints no note. Each rule in full is in [parity: stated rules](docs/parity.md#stated-rules).

| Gap | Open point | Where it stands |
|---|---|---|
| G3 | Draw Geom drawing | `retime-fgbg` draws a disc on its Source display where the original draws a cross. `create-vectors` follows the original. |
| G4 | one-layer Debug Blend view | In `create-vectors`, the Blend view of a one-layer group with matte layers is a plain blend where the original runs a step that was not read; no grid case reaches it. The other Debug views follow the original. |
| G5 | second tracker arithmetic | This project's rule, not read in the original; matches on the grid. |
| G6 | solver, hole fill and blur wrap variants | 360 Video follows the original's wrap variants and matches on the grid, but the grid has no layered or draft quality 360 case: the wrap of the layer masks and of the draft blur taps is this project's reading. |

### Not offered

Never part of this project: licensing; hosts other than After Effects; the plugins' Hide Mask paths, First/Last Mask paths and layer masks, which are parameter kinds that `fx-core` (the crate that defines the shared effect interface) cannot express. The plugins' GPU controls (Use GPU, Less GPU Mem) have no counterpart either: the rewrite picks its device with `vfx --device`, and that choice is not graded.

## Legal

- **Independent work.** This project is written independently and contains none of Twixtor's code, binaries, assets or documentation.
- **How the match was made.** To make the outputs match, the original plugins were run in an emulator on test inputs and their binaries were analysed, for interoperability. Only the resulting rules and arithmetic are written here, in this project's own code.
- **No affiliation.** It is not affiliated with, endorsed by or sponsored by RE:Vision Effects.
- **Trademarks.** Twixtor is a trademark of RE:Vision Effects. It appears here only to name the plugin this project reimplements.
- **Interoperability.** Parameter ids, display names and values follow the plugin's interface so that settings carry over between the two.
- **No warranty.** Everything here is provided as is under the MIT License, without warranty of any kind and without liability.
- **Your responsibility.** You need the rights to the media you process, and you stay bound by the licences of any software you use alongside this project, ffmpeg included.
- **Rights holders.** If you believe something here infringes your rights, email [hi@bairon.dev](mailto:hi@bairon.dev) or [open an issue](https://github.com/baairon/twixtor/issues). Every concern gets a prompt answer in good faith.

## License

MIT, see [LICENSE](LICENSE).
