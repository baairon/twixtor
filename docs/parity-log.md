# Parity log

How the parity results were reached, in order. The current state, the method and the definitions are in [parity](parity.md); this page keeps the dated record, including findings that were later replaced.

## 2026-10-28: the controls

The first comparison was of control tables, not of frames. The controls of the four effects were matched against the controls the four plugin builds register, after normalizing the names: 175 of 176 had a counterpart. The one without, `vectors-in` "Output: Frame Rate", turned out not to exist in the plugin and was later removed from the rewrite.

## 2026-10-30: the harness runs the originals

**Booting.** The four plugins were loaded unmodified in the emulator and taken through global setup and params setup. They registered 33, 356, 347 and 43 parameters. Getting there took a series of fixes in the harness, none in the plugins:
- Windows functions the C runtime needs at start (`RtlPcToFileHeader`, counted semaphores, the crypto calls with a deterministic random source);
- C++ exception dispatch done the way Windows does it (106 throws on `Twixtor_64`, all handled);
- the `printf` family run by the C runtime's own code;
- suite lookups by the numeric versions of the SDK macros;
- parameter names decoded in full.

**Speed.** At first the emulator answered 5.9 Windows calls a second. A profile showed that one Python function checked every registered hook at every instruction, and a second per-instruction hook ran a thread scheduler. With the hooks narrowed to the emulator itself and the scheduler hook removed, it answered 3,413 calls a second, and a plugin booted in about 40 s.

**The first render.** It showed a fixed pattern over the plugin's output. That was not a property of the plugin: the mock host answered "is this a render engine" with an error. With the answer an interactive After Effects gives, the pattern disappeared. A flat last frame had the same kind of cause: for a checkout outside the clip the mock reported a layer of size 0, so the plugin painted its own missing-layer fill. With both answers fixed, the four smoke cases of `Twixtor_64` matched the rewrite bit for bit.

**The time model.** Probe T1 tested the frame choice and blend weights at fractional positions with motion off. Its clips alternate between the lowest and highest level, so each output value shows a weight and its rounding. All 62 cases matched in all 40 frames: Nearest 8, Blend 30 and Motion Weighted Blend 24, at speeds from 10 to 150 percent and at 8, 16 and 32 bits per channel.

**Crashes that were the harness's.**
- **Motion Weighted Blend crashed the emulator.** The plugins need a much deeper stack than the 256 KB the emulator gave the main thread, and all guest threads shared one thread information block. The fix was a 1 MiB main stack (Windows' default reserve) and each thread's own stack bounds.
- **A sensitivity of 0 raised a divide-by-zero exception.** The emulator starts with every floating point exception unmasked. The harness now sets the Windows start values.
- **Vectors In painted one flat colour.** Its Color Source defaults to the effect's own layer, which the harness had left empty. It now gets the case's input clip.

**Tiny frames.** The plugins read invalid memory on frames one pixel wide and on 2x2 to 3x3 frames. Five host answers varied one at a time changed nothing, so this is the originals' own behavior at those sizes. By the tiny-frame rule those cases leave the grid once a module records the judgement.

**Display texts.** The rewrite's control names and popup texts were changed to read as the plugins register them, for example Twixtored Output, Motion Weighted Blend, Image Prep and Alt Color Source. No id, range, default or choice order changed, so saved settings and renders stayed as they were.

## 2026-10-31: the first full grids

The six module grids were rendered with the originals for the first time and graded at 0 ULP. About 40% of the rendered cases matched. Probing the original with crafted inputs found the first causes in the rewrite:
- the bottom-up row order of the motion path in `create-vectors` and `vectors-in`;
- the Gouraud forward warp of Vectors In;
- the frame-time bias for unequal frame rates;
- the Source display fill of `retime-fgbg`;
- the rebuild of covered layers in its group displays.

The merged rewrite then matched 1,517 of the 2,364 rendered cases, 64%. Many of the remaining differences were small and evenly spread, which input/output probing could not pin down.

## 2026-10-31: root causes from the code

From this point the binaries were also analysed: their code was read and their execution traced in the emulator with an observe-only tracer. Nothing in them was patched. Their licence and activation code was excluded and not analysed. The analysis output is not part of this repository, which holds the rules and the arithmetic to match, never the original's code.

Each fix was merged and the six grids graded again, with no case that matched before differing after:

| State | Identical | Differing | Main causes added |
|---|---|---|---|
| 1 | 1,972 | 392 | the order of the taps in vertical filter passes; the emulator's NaN rule; the Gouraud forward warp in the bottom-up frame; the Debug view arithmetic; the C runtime's `powf` |
| 2 | 2,021 | 343 | the layered solve of `create-vectors`; 360 Video as a wrap of the whole motion search |
| 3 | 2,212 | 152 | each FG layer interpolated on its own from two plates and composed from the bottom up |
| 4 | 2,249 | 115 | the matte cut with the plugin's own choke; a shrink sends exact frames down the motion path |
| 5 | 2,270 | 94 | `vectors-in` on the same per-layer path, with stored mattes and capped vector alpha |
| 6 | 2,321 | 43 | the tracking point pyramid; Track RGB+A as the 360 filter mode of the layered search |
| 7 | 2,343 | 21 | the choke and OVER of float mattes; the 360 Video blur and edge taps |
| 8 | 2,345 | 19 | the Debug view OVER and its double precision luma |
| 9 | 2,351 | 13 | the NaN rule in the pull-push fill |

The 13 cases left differ only in an alpha value that the original never writes ([parity: out of contract](parity.md#out-of-contract)), and they were judged out of contract.

**Closeout.**
- The call records were compared on the final code, both the module check and a strict comparison of every host call.
- The mutants were rebuilt from the final code, with new mutants for the new rules.
- The out-of-contract candidates were judged again against what the original accepts.
- Two supplementary cases rendered by the original killed mutants the grid could not reach: a frame-number case at a different input rate in `retime-time`, and speeds of 99 and 101 percent in `retime-flow`.
- The six result files written at the end count 2,357 compared cases, all identical, with every call record matching. They are what the [status table](parity.md#status) reads.
- The tiny-frame rule was then recorded for `vectors-in` and `retime-flow` as well. The three 1x1 cases of `vectors-in`, on which the original stops, left the grid, and `vectors-in` became the first Verified module. In `retime-flow` the 12 tiny cases no earlier run had rendered went through the harness: the original stops on all of them. Ten left the grid; the frames outside the 5x5 bound stay in contract with no reference.

## Test runs, 2026-10-30

`cargo test --release` for the whole workspace and the Python suites of `tools/` on the Linux server, before the parity fixes. None of these tests compares anything with the original plugins.

| Suite | Result |
|---|---|
| Rust, whole workspace | 1,456 pass, 3 fail, 15 ignored |
| `fx-parity` | 172 pass, 3 fail |
| Python tool suites | 12 suites: 10 OK, 2 with 1 failure each; some tests skipped |

All failures came from that server's environment. Its static ffmpeg 7.0.2 cannot generate the 32 bit float test inputs, and it has no `ffprobe`. The wrapper's own tests need a machine with ffmpeg and ffprobe.
