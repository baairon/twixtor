# Parity

What parity means for this rewrite, how it is checked against the original Twixtor plugins, and what the checks found. The README has the [short version](../README.md#parity), and the [parity log](parity-log.md) has the dated story of how the results were reached.

## Summary

**<!--f:modules_match-->6<!--/f--> of <!--f:modules_total-->6<!--/f--> modules match the original on the emulator run.** The original plugins, run in the emulator, and this rewrite render the same test cases, and their outputs are compared bit for bit. The cases are split into six modules, each one part of one plugin's behavior, checked on its own. In all six, every case that the original defines and that could be rendered gives identical output on both sides, and every call record is the same. The [status table](#status) has the counts.

**The exceptions are known.**
- **No reference output.** Some in-contract cases have none yet: frames the emulated original stops on that the tiny-frame rule's 5x5 bound does not reach, cases that crashed in the emulator, and two cases too long to render in full ([cases without a reference](#what-the-results-do-not-cover)).
- **Out of contract.** Thirteen cases are out of contract because the original leaves one value in them uninitialised ([out of contract](#out-of-contract)).
- **One known difference outside the grid.** `retime-fgbg` draws its Draw Geom points as a disc where the original draws a cross ([G3](#stated-rules)).

**Verified is the stricter step after Matches.** It also needs every in-contract case compared and every deliberately broken copy of the rewrite caught ([status rule](#the-status-rule)). A module that is not there yet misses one of the two, as its Cases compared and Mutants killed columns show. A second, independent run of the whole check comes after; it has not started.

## Status

<!--f:modules_match-->6<!--/f--> of <!--f:modules_total-->6<!--/f--> modules match the original on the emulator run. Verified: <!--f:modules_verified-->1<!--/f--> of <!--f:modules_total-->6<!--/f-->. The table is generated from each module's result file, `parity/<id>/results/summary.json`, and from `parity/<id>/results/verify.json` when an independent re-run has written one.

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

How to read a row:

- **Cases compared** reads `compared of in-contract`. The gap between the two is the in-contract cases with no reference output ([what the results do not cover](#what-the-results-do-not-cover)).
- **Out of contract** counts cases the original does not define. They are listed apart and never counted as passes.
- **Values compared** is bytes / floats; **Differing** is cases / values; **Max ULP** is the largest distance between two floats, 0 when every float is identical.
- **Call records matching** and **Mutants killed** read `matched of compared` and `killed of total`. Every surviving mutant is listed in the [per-module notes](#per-module-notes), and the module's `results/kills.json` records what each mutant changes.
- **Re-run verified** is the answer of the independent re-run. A dash means it has not run.
- A row says Matches or Verified only when the [status rule](#the-status-rule) holds for its numbers.

## The bar

Parity is defined by one check and by nothing else on this page.

- **The setup.** The four original plugin files run unmodified in a CPU emulator (Unicorn through Qiling, on Linux) behind a mock After Effects host. They are Windows x64 After Effects plugins: `Twixtor_64`, `TwixtorPro_64`, `CreateMotionVectors_64` and `TwixtorProVectorsIn_64`.
- **The same inputs** go to the original and to this rewrite.
- **The outputs must be identical.** Bytes and integers match exactly, and floats match bit for bit (0 ULP), so -0 differs from 0 and the bits of a NaN count.
- **Three checks per module:**
  - a [grid](#the-grid-and-the-out-of-contract-rule) of at least 150 in-contract cases with edge values;
  - the [call records](#the-call-record-check);
  - the [mutants](#mutation-checks).
- **Stand-ins.** Where the original reads something the harness cannot supply the way After Effects would, both sides read the same labelled [stand-in](#stand-ins-and-logic-parity). A match that rests on one is called logic parity.

The reference is the original running in the emulator, which follows the CPU manual. Real hardware and real After Effects output are not part of the bar.

## How parity is measured

1. **Test cases.** Each module has its own test set in `parity/<id>/`: `cases.json` plus raw input frames in the rewrite's formats ([raw frame format](reference.md#raw-frame-format)), written by a deterministic generator. A module's grid has at least 150 in-contract cases with edge values, and a separate list of out-of-contract candidates.
2. **The original side.** The harness in `parity/harness/` loads the original plugin unmodified in the emulator. It drives the plugin through the After Effects command sequence: global setup, params setup, sequence setup, then a render for each output frame. It converts the case's frames and parameters in and the output back out. That output is the case's reference.
3. **The rewrite side.** `vfx` renders the same case.
4. **The output check.** `fx-parity run --max-ulp 0` compares the two outputs: bytes at 8 and 16 bits per channel, bits at 32.
5. **The call-record check.** What the original asks the host for is compared with what the rewrite reads ([below](#the-call-record-check)).
6. **Mutants.** Deliberately wrong copies of the rewrite must each change at least one compared case ([below](#mutation-checks)).
7. **Out of contract.** Cases the original does not define are judged and listed apart ([below](#the-grid-and-the-out-of-contract-rule)).
8. **Finding causes.** A difference is traced to its cause before anything changes.
   - First comes the harness, meaning a host answer or an emulator setting.
   - Then comes the rewrite, which changes only to do what the original does.
   - A difference without a shown cause stays counted.
   - Most causes were found by running the original on crafted inputs. Where that reached its limit, the original's code was read and its execution traced ([root causes](#root-causes-found)).
9. **Verification.** A module is Verified when the [status rule](#the-status-rule) holds. A separate, independent re-run on fresh copies then confirms it in the Re-run verified column.

Builds, emulation and grids run on a Linux server. The reference outputs and raw frames stay there; the repository holds the test set definitions, the tools and the small result files.

### The grid and the out-of-contract rule

A grid has at least 150 compared in-contract cases per module, with edge values in the parameters, the pixels and the times: 0, -0, denormals, inf, NaN and the extremes. It covers sizes from 1x1 to 129x71 (odd sizes included), 8, 16 and 32 bits per channel, every popup choice and real-motion clips.

A case is in contract when the original defines its behavior. A case outside that contract is reported under Out of contract. It is never counted as a pass or as a failure, and it does not count toward the 150. A case leaves the contract only by what the original does: it refuses it, it stops on it at a tiny size, the value lies outside the range the plugin registers, or the output is not defined by the original at all.

### The call-record check

The mock host records every request the original makes: which frames and layers it checks out, at which times, and which parameters it reads. `vfx` lists the rewrite's reads the same way under `VFX_CALL_RECORD` ([reference: call record](reference.md#call-record)). The module check compares the pixels each side reads, frame by frame. A stricter comparison of every recorded host call was also run for each module.

The strict comparison found a few kinds of difference, and none of them reaches an output. They are listed in each module's notes as items to accept or fix:
- the order of the reads;
- an input the original registers one more time;
- parameters the rewrite has no counterpart for (mask paths, Use GPU);
- in a few displays, a layer one side declares but neither side reads.

It matters because an output match can hide a wrong input. A rewrite that reads a different frame can give the same bytes on a grid that never separates the two.

### Mutation checks

A grid that never fails proves nothing about its own strength. A mutant is a copy of the rewrite changed on purpose in a way that is wrong, for example an operand order, a rounding, a dropped term or a swapped weight. The grid must catch it: at least one compared case must then differ from the original. Mutants are rebuilt from the current code before they are counted. A mutant the grid does not catch is a survivor. A survivor is either provably equivalent (it can never change an output), or it shows a gap in the grid, and the gap gets a supplementary case.

### Stand-ins and logic parity

Some data the original reads cannot be provided the way After Effects would. After Effects' own keyframe interpolation is the standing example: the host computes the value of a keyframed parameter at a time. For such data both sides read the same labelled stand-in, the module lists it under `stand_ins`, and a match that rests on it is called logic parity. It shows that the plugin's logic and the rewrite's logic agree on that input, not what a real host would have supplied.

### The status rule

| Status | Means |
|---|---|
| Verified | every in-contract case compared (at least 150), 0 differing (cases and values), every call record matching, every mutant killed |
| Matches | the emulator run passes: at least 150 cases compared, 0 differing (cases and values), every call record matching. In-contract cases without a reference and surviving mutants are listed, not counted |
| In progress | the module has a result file and meets neither rule yet |
| Blocked | something outside the module stops it, named in `blockers` |
| Not started | no `summary.json` |

The generator applies the rule to the numbers and counts a check with nothing in it as unmet. It reads `verify.json` separately: Verified plus a Yes in Re-run verified means a second, independent run reproduced the result.

## What the results do not cover

- **The grid is a sample.** About 2,500 chosen cases are not every input. A match is strong evidence for those inputs, not a proof for all.
- **The reference is the original in the emulator.** Where the original depends on the real After Effects host, both sides use a [stand-in](#harness-stand-ins-and-assumptions).
- **Cases without a reference.** Some in-contract cases have no reference output, so they are not compared. The per-module notes list them. They are:
  - seven `retime-flow` frames the original stops on that the tiny-frame rule does not reach: a side of 1 with the other side above 5 (1x6, 7x1), and a 7x9 frame at draft quality;
  - `fgbg-core` cases that crashed in the emulator for reasons not shown to be the original's;
  - two 6000-frame `retime-time` cases. The original rebuilds its speed curve from the first frame at every call, so a full run costs about seven hours each. 26 selected frames of each are identical.
- **The GPU path.** Only the CPU path is compared. The GPU (OpenCL) kernels were written for the earlier rules, and several no longer match the CPU path. A `vfx --device gpu` render is not covered until they are ported.

### Out of contract

- **Tiny frames.** Sizes from 1x1 to 5x5 enter a grid only if the original renders them. The original stops on frames one pixel wide and on the smallest motion search sizes, so a case it stops on goes out of contract once its module records the judgement. Every module records it. In `retime-flow` the original also stops on frames with a side of 1 and the other side above 5 (1x6, 7x1) and on a 7x9 frame at draft quality. The 5x5 bound does not reach those, so they stay in contract with no reference.
- **Values the plugin does not register.** A value outside a registered range, or a popup choice that does not exist, is never handed over by After Effects.
- **An uninitialised value.** In the Source display at 32 bits per channel without an Alt Color Source, the original fills the frame with a colour whose alpha it never writes.
  - The alpha holds whatever an earlier call left in that place on the stack, part of a return address, so it would change with the host's memory layout.
  - The rewrite writes 0. The 13 cases differ only in that one value.
  - They are listed in `parity/fgbg-core/ooc_judged.tsv` and `parity/fgbg-layers/ooc_judged.tsv`. The modules' `phase_b.py ooc` without `--judged` puts them back.

### Harness stand-ins and assumptions

The harness supplies some things that After Effects or Windows would. The full table, with the reason for each line, is in the [harness README](../parity/harness/README.md). The ones that matter for reading the results:

- **Keyframe interpolation** is done by the harness (linear, hold, ease; Bezier refused), so keyed cases reach logic parity at most.
- **FG matte layers** are handed to `TwixtorPro_64` as ordinary layers. That matches how it reads them, for mattes of the frame's size, unmasked and not time remapped.
- **Host answers** come from the SDK headers: an interactive (not render engine) host; for a checkout outside a short clip, an empty result at the layer's size; for an unset layer whose registered default is the effect's own layer, the case's input clip.
- **The emulated CPU** starts with the Windows floating point state. Threads switch only when one waits, and the main thread's stack is 1 MiB.
- **Windows functions** without a hook in the emulator (outside the C runtime DLLs) return 0. That is a fidelity risk.

### The NaN rule

When an operation gets two NaNs, the emulator keeps the one with the larger fraction, while a real x86 processor keeps the first operand. The references were made in the emulator, so the rewrite reproduces the emulator's rule by default (cargo feature `emulator-nan` of `retime-engine`). Building without default features gives the processor's rule; the cases that depend on it then differ by the bits of a NaN only. Which rule the shipped build uses is a choice, and it touches nothing but NaN payloads.

### Caveats

- **Platform math library.** A few steps take a value from a math library. The original's runs inside the emulator; the rewrite carries a port of the C runtime's `powf` where it matters and uses the host's elsewhere. A difference there would show up as a differing value. It is not hidden.
- **Threads.** The rewrite's bytes do not depend on its thread count (`fx-parity selfcheck --threads` and the `threads` tests of each plugin).
- **16 bit range.** At 16 bits per channel the effects work on 0 to 32768, so an unchanged sample can come back 1 off in 65535 ([reference: raw frame format](reference.md#raw-frame-format)). There is no tolerance for it: the output is compared with the original's as it is.
- **32 bit float.** `fx-parity run` passes floats within 1 ULP unless `--max-ulp 0` is given, so every graded run gives it.

## Root causes found

Every difference found between the original and the rewrite had a cause in one of two places.

**The harness.** Our emulator setup, not the plugin:
- the main thread's stack size and each thread's stack bounds;
- the floating point state at start;
- the answer to "is this a render engine";
- the answer for a checkout outside the clip;
- the default of an unset layer;
- the emulator's NaN rule.

**The rewrite.** Each of these was fixed to do what the original does:

| What the original does | Where it showed |
|---|---|
| Its motion path works on frames bottom row first. | every motion module |
| At the edge of a frame, the vertical pass of every separable filter adds the tap above before the tap below, so floats round differently there. | the flow, the hole fill and the blurs: the last differences of `retime-flow`, the 32 bit vectors of `create-vectors`, the fills of `vectors-in` |
| The forward warp of `TwixtorPro_64` and `TwixtorProVectorsIn_64` is a mesh of triangles shaded by their corner colours, not a textured mesh. | `fgbg-core`, `vectors-in` |
| Each FG layer is interpolated on its own from two plates (the layer over the other frame's warp) and the layers are composed over each other from the bottom up; each layer's motion is solved on its own layer images, and block matching ignores the masks. | `fgbg-layers`, `vectors-in`, `create-vectors` |
| A matte shrink uses the plugin's own choke kernel, and a shrink makes exact frames take the motion path. | `fgbg-layers` |
| Tracking points move the motion at the finest level only; the coarser levels take it from the image pyramid. | `fgbg-layers`, `create-vectors` |
| Track RGB+A switches the filters of the layered motion search to their 360 variants; 360 Video wraps the whole motion search around the sphere. | `fgbg-layers`, `fgbg-core` |
| The Debug views, and the Draw Geom glyph of `create-vectors` (an antialiased cross), have their own arithmetic. | `create-vectors` |
| FG matte channels are read as stored, and in Vectors In each layer's vector alpha is capped by its matte. | `vectors-in`, `fgbg-layers` |
| The Delinearize step of Image Prep takes `powf` from the C runtime; the time model rounds frame times and parameter samples with a bias of 0.05 where the rewrite had assumed 0.1. | `retime-flow`, `retime-time` |

The rewrite takes only the rule and the arithmetic it has to match. It contains no code, binaries or documentation of the original.

## Reproducing the results

The original plugins, the Windows runtime DLLs for the emulator's file system and Adobe's After Effects SDK are not in the repository. Take them from your own installation; the harness README says how to prepare them. Then:

1. Build the rewrite: `cargo build --release -p fx-cli -p fx-parity` ([build](reference.md#build)).
2. Render a module's test set with the original under the harness to get the references ([harness README](../parity/harness/README.md)).
3. Render the same cases with `vfx`, with `VFX_CALL_RECORD=<file>` set for the call records.
4. Grade with the module's `phase_b.py` (each module's README has the commands), which runs `fx-parity run --max-ulp 0`.
5. `python -I tools/gen_docs.py` writes the tables on this page from the result files; `--check` fails when they are stale.

## Interface

Every parameter and layer of the four effects pairs with one the plugins register, with the same id, kind, range, default and choices. The pairing tables are `parity/retime-params/pairing.tsv`, `parity/retime-fgbg-params/pairing.tsv`, `parity/create-vectors-params/pairing.tsv` and `parity/vectors-in-params/pairing.tsv`. Interface agreement is not output parity: two controls with the same name and range can still render differently, which is what the grids test.

| Plugin | Rewrite effect | Registered | Paired | Host-side | Group markers | Mask paths |
|---|---|---|---|---|---|---|
| `Twixtor_64` | `retime` | 33 | 19 | 6 | 8 | 0 |
| `TwixtorPro_64` | `retime-fgbg` | 356 | 68 | 6 | 18 | 264 |
| `CreateMotionVectors_64` | `create-vectors` | 347 | 63 | 6 | 14 | 264 |
| `TwixtorProVectorsIn_64` | `vectors-in` | 43 | 25 | 6 | 12 | 0 |

- **Host-side controls** (Activation, a forced re-render, two Use GPU popups, Less GPU Mem, SavedInPremiere) stay at their registered defaults. There is no license handling of any kind.
- **Mask paths** (256 Hide Mask paths and 8 First/Last Mask paths) are not offered ([README: not offered](../README.md#not-offered)).

## Stated rules

A render that reaches one of these points prints a `vfx: note:` line ending `(README: limits)`. The notes come from the gap register in `plugins/retime-engine/src/gaps.rs`, which `vfx list --json` lists under `gaps` and `docs/data/gaps.tsv` mirrors. The register was written when the original's behavior at these points was not known. Since then, every point except one follows the original or matches it on every compared grid case that reaches it. Retiring the notes from the code is still to do.

| Gap | Open point | What the code does now | On the grid |
|---|---|---|---|
| G1 | matte shrink tail | The original's own choke kernel. A fraction mixes the two nearest whole shrinks, and a negative shrink grows the matte. A shrink makes exact frames take the motion path. | matches (54 grid cases with a shrink) |
| G1b | mask-aware block matching cost | No gate: block matching never reads the masks, which reach the solve through each layer's own images. | matches (every FG matte case) |
| G1c | layer visibility direction | FG1 is on top. Each layer is interpolated on its own and composed from the bottom up. A layer's matte is cut by the layers above with the exact choke. | matches |
| G1d | create-vectors merge | The vectors of a group are composited from the bottom up with the raw mattes. A one-layer output is that layer's own flow, with its visible matte in blue. | matches (85 FG cases, 48 with Layers Output) |
| G1e | vectors-in per-layer matte step | Each layer's vector alpha is capped by its matte, and the matte channel is read as stored. | matches (22 FG cases) |
| G1f | blur tail flow of a layered render | The original's per-layer path: every layer's blur compensation uses that layer's own motion. This project's earlier rule is left only on a path no effect takes. | matches (22 grid cases) |
| G1g | hole fill of a single group display | The original's per-layer path: a single group display rebuilds and fills its layer from its own plates. This project's earlier rule is left only on a path no effect takes. | matches (94 group display cases) |
| G1h | fast path group display | The group is composed without warp or fill, and rebuilt where the layers above cover it. | matches (the same cases) |
| G2 | tracking point displacement | The original's rule: the points move the motion at the finest level only, and the coarser levels take it from the image pyramid. | matches (114 point cases) |
| G3 | Draw Geom drawing | `create-vectors` draws the original's antialiased cross. `retime-fgbg` still draws an opaque disc of radius 1.5 on the Source display. | `create-vectors` matches; `retime-fgbg` differs on a supplementary set (the original draws the cross), and a fix is written but not merged |
| G4 | four Debug view compositions | The original's arithmetic for all five views. | matches (60 Debug display cases) |
| G5 | second tracker arithmetic | This project's rule: a layer with Track off takes the motion of its tracking points, spread by the pull-push hole fill. | matches (13 cases with Track off); the rule was not read in the original |
| G6 | solver, hole fill and blur wrap variants | Track RGB+A switches the filters of the layered motion search to their 360 variants and adds no alpha; 360 Video wraps the whole motion search around the sphere. | matches (59 Track RGB+A cases, 16 360 Video cases) |

## Rules without a note

Some points have a stated rule but print no note. The original's behavior there is host behavior, or it belongs to the wrapper, so there is nothing in the plugin to compare.

| Point | Stated rule | How the bar treats it |
|---|---|---|
| tiny frames | the motion search never halves below 2x2; a frame with a side of 1 has no search and renders as `vectors=1` (`create-vectors`: the zero vector) | sizes 1x1 to 5x5 enter a grid only if the original renders them; otherwise out of contract |
| a plain key next to a handle | the plain end is a handle of the straight line's speed at an influence of 1/3 | keyframe interpolation is the host's work: a stand-in, logic parity at most |
| an eased key's influence | exactly 1/3 | the same |
| overshoot | a float is held inside its valid range; an angle is not | the same |
| the curve solve | 64 bisection steps in f64, in the operation order of [timing: key shapes](timing.md#key-shapes) | the same |
| a point's path | the straight line between its keys at S pixels a second; a segment of length 0 stays put | the same, together with G2 and G3 |
| retimed audio | varispeed along the effect's time map ([timing: audio](timing.md#audio)) | outside the original's contract; `tools/test_apply.py` holds the sync |
| a checkout between two composition frames | the footage frame at that exact time (`vfx --clip-fps`) | host behavior; the call record shows which frame the original asks for |
| a new size | the footage is scaled before the effect, which renders at the new size | host behavior; the call record shows what the original is given |
| the default length at another clip rate | the composition frames that start inside the layer, ceil(C x F / R) (25 frames at 24 fps give 63 at 60) | host behavior; not asked of the plugin |
| HDR input | the signal as stored, no tone mapping ([ffmpeg: HDR](ffmpeg.md#hdr)) | outside the original's contract |

The `--device` choice has no rule: the CPU path is the reference ([engine: device](engine.md#device)).

## Secondary evidence

These do not compare the rewrite with the original. They catch regressions inside the rewrite.

### Model fixtures

Some engine stages are compared bit for bit with independent models of their arithmetic (numpy float32 scripts, or the standard library for the Bezier handles), stored as hex fixtures that the Rust tests decode. A model of a stage that is itself a stated rule shows that the code follows the rule, not that the rule is right.

There are <!--f:model_fixtures-->11<!--/f--> model fixtures with <!--f:model_cases-->794<!--/f--> cases and <!--f:model_bytes-->4,322,375<!--/f--> decoded bytes. The <!--f:differential_fixtures-->6<!--/f--> flow fixtures come from a differential model that is not bit exact, so they are listed and not counted. The generators of each fixture are listed in [reference: tests](reference.md#tests).

<!--gen:fixtures-->
| Fixture | Magic | Kind | Cases | Bytes |
|---|---|---|---|---|
| `crates/fx-core/tests/fixtures/bezier_cases.hex` | `BEZ1` | model | 65 | 156,096 |
| `plugins/retime-engine/tests/fixtures/blur_cases.hex` | `BLR1` | model | 42 | 290,984 |
| `plugins/retime-engine/tests/fixtures/debug_cases.hex` | `DBG1` | model | 35 | 126,888 |
| `plugins/retime-engine/tests/fixtures/f1_cases.hex` | `F1C1` | model | 109 | 1,348,980 |
| `plugins/retime-engine/tests/fixtures/f2_cases.hex` | `F2C1` | model | 26 | 113,139 |
| `plugins/retime-engine/tests/fixtures/flow_a.hex` | `FLD2` | differential | - | 122,904 |
| `plugins/retime-engine/tests/fixtures/flow_b.hex` | `FLD2` | differential | - | 73,752 |
| `plugins/retime-engine/tests/fixtures/flow_c.hex` | `FLD2` | differential | - | 49,176 |
| `plugins/retime-engine/tests/fixtures/flow_d.hex` | `FLD2` | differential | - | 77,440 |
| `plugins/retime-engine/tests/fixtures/flow_e.hex` | `FLD2` | differential | - | 81,944 |
| `plugins/retime-engine/tests/fixtures/flow_f.hex` | `FLD2` | differential | - | 81,944 |
| `plugins/retime-engine/tests/fixtures/mask_cases.hex` | `MSK1` | model | 139 | 357,708 |
| `plugins/retime-engine/tests/fixtures/matte_cases.hex` | `MAT1` | model | 126 | 523,432 |
| `plugins/retime-engine/tests/fixtures/merge_cases.hex` | `MRG1` | model | 11 | 51,304 |
| `plugins/retime-engine/tests/fixtures/vectors_cases.hex` | `VEC1` | model | 21 | 49,564 |
| `plugins/retime-engine/tests/fixtures/warp_cases.hex` | `WRP1` | model | 60 | 693,099 |
| `plugins/retime-engine/tests/fixtures/wrap_cases.hex` | `WRA1` | model | 160 | 611,181 |
<!--/gen:fixtures-->

### Regression lock

`docs/data/lock.json`, recorded <!--f:lock_date-->2026-10-28<!--/f--> against baseline commit <!--f:lock_ref-->d74f402<!--/f-->: every setting that rendered on the baseline build renders the same bytes on the build of that day, <!--f:lock_identical-->13,073<!--/f--> of <!--f:lock_total-->13,073<!--/f--> cases. It compares two builds of this rewrite on one machine; the original is not involved. The parity fixes changed the engine since, so the lock has to be recorded again.

| Corpus | Cases |
|---|---|
| L1 | `fx-parity gen` with its defaults for the four effects and `selftest-invert`, `selftest-shift`, `selftest-integrate` and `selftest-threads` |
| L2 | `fx-parity gen` of the four effects at sizes 2x2, 3x2, 4x4, 5x4, 8x8, 9x8, 16x16 and 17x16, 6 frames, every case (`--max-cases-per-effect 0`) |
| L3 | the temporal L2 cases of `retime`, `retime-fgbg` and `create-vectors`, run with `--quality draft` |

<!--gen:lock-->
| Corpus | Rendered at the baseline | Identical now | Differ | Refused at the baseline | Of those, render now |
|---|---|---|---|---|---|
| L1 | 1,955 | 1,955 | 0 | 74 | 74 |
| L2 | 6,090 | 6,090 | 0 | 444 | 444 |
| L3 | 5,028 | 5,028 | 0 | 492 | 492 |
| all | 13,073 | 13,073 | 0 | 1,010 | 1,010 |
<!--/gen:lock-->

To reproduce it, from the repository root after [building](reference.md#build), with REF the recorded baseline (<!--f:lock_ref-->d74f402<!--/f-->):

```sh
python tools/parity_lock.py base --ref REF
python tools/parity_lock.py corpus
python tools/parity_lock.py refs
python tools/parity_lock.py check --vfx target/release/vfx
python tools/parity_lock.py refused --vfx target/release/vfx
python tools/parity_lock.py record
python -I tools/gen_docs.py
```

`base` builds both tools of the ref in a temporary worktree; `check` exits 1 on any difference and `refused` while a refused case still fails; `record` writes `docs/data/lock.json`. `python tools/parity_lock.py --help` lists the flags.

### Test counts

A scan of the source files: a `#[test]` line is one test of `cargo test`, a `def test_` line one test of its `unittest` suite, and an `#[ignore]` test runs only on request ([reference: tests](reference.md#tests)). Counting a test says nothing about whether it passes.

<!--gen:tests-->
| Suite | Counted | Tests |
|---|---|---|
| `fx-cli` | `#[test]` | 134 |
| `fx-core` | `#[test]` | 52 |
| `fx-parity` | `#[test]` | 175 |
| `create-vectors` | `#[test]` | 91 |
| `retime` | `#[test]` | 70 (2 ignored) |
| `retime-engine` | `#[test]` | 843 (36 ignored) |
| `retime-fgbg` | `#[test]` | 104 (1 ignored) |
| `vectors-in` | `#[test]` | 104 |
| `tools/test_apply.py` | `def test_` | 110 |
| `tools/test_apply_geometry.py` | `def test_` | 22 |
| `tools/test_apply_hdr.py` | `def test_` | 38 |
| `tools/test_audio_map.py` | `def test_` | 26 |
| `tools/test_conformance.py` | `def test_` | 47 |
| `tools/test_docgen.py` | `def test_` | 46 |
| `tools/test_ffstage.py` | `def test_` | 25 |
| `tools/test_handoff.py` | `def test_` | 154 |
| `tools/test_parity_lock.py` | `def test_` | 21 |
| `tools/test_repeat_keys.py` | `def test_` | 9 |
| `tools/test_speed_curve.py` | `def test_` | 32 |
| `tools/test_stages.py` | `def test_` | 70 |
<!--/gen:tests-->

## Terms

| Term | Meaning |
|---|---|
| module | one part of one plugin's behavior, checked on its own: an id, a plugin, a rewrite effect |
| in contract | a case whose behavior the original defines |
| out of contract | a case the original does not define; reported apart, never a pass or a failure |
| reference | the original's output for a case, rendered in the emulator |
| call record | the host calls an original makes, against the reads of the rewrite |
| mutant | a deliberately wrong copy of the rewrite that the grid must catch |
| survivor | a mutant the grid did not catch |
| kill | a mutant whose output differs from the original's on at least one compared case |
| stand-in | labelled data fed to both sides where the original reads something the harness cannot supply as a host would |
| logic parity | a match that rests on a stand-in |
| model fixture | the inputs and expected outputs of an independent model of one engine stage, which the Rust tests reproduce bit for bit |
| differential fixture | a fixture of a model that is not bit exact, which the tests compare within a tolerance |

## Per-module notes

What each module's result file states about blockers, stand-ins, logic parity and surviving mutants:

<!--gen:notes-->
| Module | Kind | Text |
|---|---|---|
| `retime-time` | blocker | 2 in-contract cases have no reference |
| `retime-time` | blocker | 2 cases did not render in the harness for reasons that are not the original's (see ooc.json unjudged) |
| `retime-time` | blocker | 10 out-of-contract candidates rendered in the original and need review |
| `retime-time` | stand-in | keyed curves (speed_percent, frame_num, src_mark_segments with @ keys): host side keyframe interpolation, one labelled stand-in on both sides, cases counted as logic_parity |
| `retime-time` | stand-in | 46 keyed-parameter cases are logic parity candidates; 46 compared (counted in cases.compared, logic parity only, not full parity) |
| `retime-time` | logic parity | 46 compared cases use a stand-in on both sides |
| `retime-time` | surviving mutant | m09_exact_tolerance_wider |
| `retime-time` | surviving mutant | m13_k_bias |
| `retime-time` | surviving mutant | m23_planner_second_frame |
| `retime-time` | surviving mutant | m24_blend_weight_f32 |
| `retime-time` | surviving mutant | m25_dur_operand_order |
| `retime-flow` | blocker | 7 in-contract cases have no reference |
| `retime-flow` | blocker | 7 cases have no usable outcome from the harness |
| `retime-flow` | stand-in | 3 cases with keyed speed_percent (AE keyframe interpolation) are tagged stand_in=keyframe_interpolation; 3 of them compared, counted as logic_parity (not full parity) |
| `retime-flow` | stand-in | draft quality cases (testsets/draft) need the same quality setting on the host side as vfx --quality draft |
| `retime-flow` | logic parity | 3 compared cases use a stand-in on both sides |
| `create-vectors` | blocker | mutants: 47 of 51 killed; survivors m05, m06, m29, m41 |
| `create-vectors` | blocker | 2 cases hit emulator or harness problems and are unresolved |
| `create-vectors` | stand-in | 5 cases with keyed params use the vfx keyframe syntax as the shared stand-in for AE keyframe interpolation (keyframe interpolation of the host: eased end key; keyframe interpolation of the host: linear; keyframe interpolation of the host: point, linear) |
| `create-vectors` | stand-in | 5 keyed-parameter cases are logic parity, not full parity: 5 compared, 5 identical |
| `create-vectors` | logic parity | 5 compared cases use a stand-in on both sides |
| `create-vectors` | surviving mutant | m05 |
| `create-vectors` | surviving mutant | m06 |
| `create-vectors` | surviving mutant | m29 |
| `create-vectors` | surviving mutant | m41 |
| `vectors-in` | stand-in | AE keyframe interpolation: 19 grid cases key a parameter (tag keyframe_interpolation); both sides read the same labelled interpolation of the keys, and those cases count as logic_parity (19 compared, 19 identical) |
| `vectors-in` | logic parity | 19 compared cases use a stand-in on both sides |
| `fgbg-core` | blocker | 14 of 367 in-contract cases have no reference |
| `fgbg-core` | blocker | 14 cases failed in the emulation (not the original's behaviour) and have no reference |
| `fgbg-core` | blocker | 39 cases have no complete plugin call record |
| `fgbg-core` | blocker | mutants survive: M05, M30 |
| `fgbg-core` | stand-in | 5 keyed cases (tag standin in testset/grid.tsv) use the harness's labelled keyframe interpolation on both sides; 5 of them compared, counted as logic_parity and not as full parity |
| `fgbg-core` | logic parity | 5 compared cases use a stand-in on both sides |
| `fgbg-core` | surviving mutant | M05 |
| `fgbg-core` | surviving mutant | M30 |
| `fgbg-layers` | blocker | 2 out of contract candidates were rendered by the original and are within registered ranges: review for promotion into the grid |
| `fgbg-layers` | blocker | mutant survivors: M20, M21, M22, M23, M24, M25, M29, M43 |
| `fgbg-layers` | stand-in | 33 cases (tag standin in testset/grid.tsv) use keyframed ptN_pos or ptN_use values: AE keyframe interpolation is replaced by the harness's labelled linear stand-in on both sides |
| `fgbg-layers` | stand-in | 362 cases hand matte layers (fg1_matte to fg3_matte) to the plugin as raw clips through the harness, which is a stand-in for how AE hands a hidden layer over |
| `fgbg-layers` | stand-in | logic_parity counts compared cases that depend on either stand-in (341 of 353 compared); they are not full parity |
| `fgbg-layers` | logic parity | 341 compared cases use a stand-in on both sides |
| `fgbg-layers` | surviving mutant | M20 |
| `fgbg-layers` | surviving mutant | M21 |
| `fgbg-layers` | surviving mutant | M22 |
| `fgbg-layers` | surviving mutant | M23 |
| `fgbg-layers` | surviving mutant | M24 |
| `fgbg-layers` | surviving mutant | M25 |
| `fgbg-layers` | surviving mutant | M29 |
| `fgbg-layers` | surviving mutant | M43 |
<!--/gen:notes-->

The harness's own statement, `parity/harness/results/summary.json`, shown as it stands:

<!--gen:harness-->
| Entry | Stated |
|---|---|
| `state` | all four plugins boot and render; the references of the six module grids were made with this harness |
| `plugins` | 4 entries, see the file |
| `dlls_copied` | 28 entries, see the file |
| `stand_ins` | 10 entries, see the file |
| `assumptions` | 3 entries, see the file |
| `callrecord_check` | {"ok":true,"cases":12,"all_stdout_and_stderr_equal":true} |
| `unit_tests` | the harness unit tests (test_*.py) pass: 115 run, 5 skipped |
<!--/gen:harness-->
