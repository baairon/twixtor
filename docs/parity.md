# Parity

How close the output of twixtor is to the original plugin's, how every figure of the README's [Parity](../README.md#parity) table is counted, and what is not shown yet. Each figure on this page and in that table sits in a marker that `tools/gen_docs.py` fills from the snapshots in `docs/data/` and from scans of the tree; `python -I tools/gen_docs.py --check` fails when a figure drifts, and `docs/data/facts.json` lists every key with its value and source.

## The unit counted

A control: one parameter or one layer of one of the four effects, as `vfx list --json` lists them. There are <!--f:controls_total-->176<!--/f--> controls (<!--f:controls_params-->159<!--/f--> parameters and <!--f:controls_layers-->17<!--/f--> layers of <!--f:effects_total-->4<!--/f--> effects), one row each in `docs/data/controls.tsv`. The hidden `selftest` effects are not counted. A row of the Parity table that counts something else names its unit in [Standards](#standards).

## Tiers

| Tier | Means | Shown by |
|---|---|---|
| Graded | `vfx` renders a case of the checklist like the original plugin: bit for bit at 8 and 16 bpc, within one unit in the last place at 32 bpc | `fx-parity run --strict` against reference renders of the same cases made in the original ([Comparing with reference renders](../README.md#comparing-with-reference-renders)); the results in `docs/data/references.tsv` |
| Model | a stage of the engine (or the keyframe curves of `fx-core`) equals an independent model of its documented arithmetic, bit for bit | the hex model fixtures, which the Rust tests decode and compare, and the numpy or Python generators that reproduce them byte for byte ([Tests](../README.md#tests)) |
| Rule | where the original's behaviour is not established, a stated rule of this project stands in; a render that uses it prints one note, and the rule waits for a graded reference | the gaps register in `docs/data/gaps.tsv`, from the `gaps` of each effect in `vfx list --json` ([Gaps register](#gaps-register)) |
| Defined | behaviour that belongs to this tool and not to the plugin, so there is nothing in the original to grade it against | tests |

Defined covers the wrapper's own features in `tools/apply.py` (audio, a new size or frame rate, HDR input and odd frame sizes), Bezier key handles (keyframe interpolation, which the host does for the original), the motion search on tiny frames (where no result of the original is established) and the `--device` choice. Their stated rules are in [Rules without a note](#rules-without-a-note).

A model fixture is no comparison with the original. It shows that the code computes what its precision contract says; where the modelled step is a rule, it shows that the code follows the rule. Only a graded case compares with the original.

## Standards

The rows of the README's Parity table, each with its unit:

| Row | Unit | Count | Share |
|---|---|---|---|
| Graded against the original plugin (bit exact) | a checklist case of the reference families, one row of `docs/data/references.tsv` | <!--f:graded_cases-->0<!--/f--> of <!--f:family_cases-->117<!--/f--> cases | <!--f:graded_pct-->0.0<!--/f-->% of the cases |
| Rule pending a reference (a note is printed) | a gap, one row of `docs/data/gaps.tsv`, and the controls it names | <!--f:gaps_open-->13<!--/f--> gaps over <!--f:gap_controls-->57<!--/f--> controls | <!--f:gap_controls_pct-->32.4<!--/f-->% of the controls |
| Model exact (independent model fixtures) | a fixture file, and the cases its header counts | <!--f:model_fixtures-->11<!--/f--> fixtures, <!--f:model_cases-->794<!--/f--> cases | none: a fixture is no share of the controls |
| Regression lock | an invocation: one case of a lock corpus, run by `vfx` | <!--f:lock_identical-->13,073<!--/f--> of <!--f:lock_total-->13,073<!--/f--> | <!--f:lock_pct-->100.0<!--/f-->% |
| Unproven | a control | <!--f:unproven-->176<!--/f--> of <!--f:controls_total-->176<!--/f--> | <!--f:unproven_pct-->100.0<!--/f-->% of the controls |

- **Graded.** A case counts when its result is `pass`; `fail` and `missing` (no reference render yet) do not. Of the <!--f:family_cases-->117<!--/f--> cases, <!--f:graded_failed-->0<!--/f--> failed and <!--f:graded_missing-->117<!--/f--> have no reference yet.
- **Rule.** A control counts once per effect, also when several gaps name it.
- **Model.** A fixture is a model fixture when a generator beside it documents its header as `b"MAGIC", u32 ncases`; its cases are the count in that header. The model fixtures hold <!--f:model_bytes-->4,322,375<!--/f--> decoded bytes, each listed in [Model fixtures](#model-fixtures). The <!--f:differential_fixtures-->6<!--/f--> flow fixtures `flow_a.hex` to `flow_f.hex` come from a differential model that is not bit exact, so they are listed and not counted.
- **Regression lock** and **Unproven**: their own sections below.

## Unproven

A control is graded when at least one checklist case that sets it (as a param or as a layer) passed and no checklist case that sets it failed. Every other control is unproven: today <!--f:unproven-->176<!--/f--> of <!--f:controls_total-->176<!--/f-->, because no reference render of the original has been graded yet. The model fixtures and the lock do not lower the figure: they show that the code does what its contracts say and that it has not changed, not that it matches the original. A control that a failing case sets is unproven again.

## Gaps register

Where the original's behaviour is not established, each open point has one stated rule, one id and one family of reference cases that settles it. A frame that uses the rule prints, once per gap and run, `vfx: note: <effect>: <id> <open point>: a stated rule that no Twixtor reference render has graded yet (README: parity gaps)` on stderr; `--check` prints none, and the exit code does not change. `vfx list --json` lists the open gaps of each effect as `gaps`, and `docs/data/gaps.tsv` keeps them. The rule of each gap, its reference family and when its note prints are in [README: parity gaps](../README.md#parity-gaps).

<!--gen:gaps-->
| Gap | Open point | Effects | Controls | Graded reference cases |
|---|---|---|---|---|
| G1 | matte shrink tail | `create-vectors`, `retime-fgbg`, `vectors-in` | `fg1_inv_matte_shrink` to `fg3_inv_matte_shrink` | 0 of 10 |
| G1b | mask-aware block matching cost | `create-vectors`, `retime-fgbg` | `fg1_matte` to `fg3_matte` | 0 of 2 |
| G1c | layer visibility direction | `create-vectors`, `retime-fgbg`, `vectors-in` | `fg2_matte`, `fg3_matte` | 0 of 10 |
| G1d | create-vectors merge | `create-vectors` | `fg1_matte` to `fg3_matte`, `layers_output` | 0 of 5 |
| G1e | vectors-in per-layer matte step | `vectors-in` | `fg1_matte` to `fg3_matte` | 0 of 3 |
| G1f | blur tail flow of a layered render | `retime-fgbg`, `vectors-in` | `blur_compensation` | 0 of 2 |
| G1g | hole fill of a single group display | `retime-fgbg`, `vectors-in` | `display`, `warping` | 0 of 10 |
| G1h | fast path group display | `retime-fgbg`, `vectors-in` | `display` | 0 of 10 |
| G2 | tracking point displacement | `create-vectors`, `retime-fgbg` | `pt1_use` to `pt12_use` | 0 of 6 |
| G3 | Draw Geom drawing | `create-vectors`, `retime-fgbg` | `draw_geom` | 0 of 18 |
| G4 | four Debug view compositions | `create-vectors` | `debug_view_method` | 0 of 10 |
| G5 | second tracker arithmetic | `create-vectors` | `main_bg_track`, `fg1_track` to `fg3_track` | 0 of 6 |
| G6 | solver, hole fill and blur wrap variants | `retime-fgbg` | `track_rgb_a` | 0 of 8 |
<!--/gen:gaps-->

When a family is graded bit exact, its gap leaves the register: the note and the `gaps` entry go, `--refresh-data` rewrites `docs/data/gaps.tsv`, its row of the README's rules table goes, and its controls leave the Rule row.

The reference families, from `docs/data/references.tsv` (written by `--refresh-data --testset` from a set of `fx-parity gen --families all`; with no reference renders yet every case is `missing`):

<!--gen:families-->
| Family | Gap | Cases | Pass | Fail | Missing |
|---|---|---|---|---|---|
| `gap-blurflow` | G1f | 2 | 0 | 0 | 2 |
| `gap-debug` | G4 | 10 | 0 | 0 | 10 |
| `gap-drawgeom` | G3 | 18 | 0 | 0 | 18 |
| `gap-groups` | G1g,G1h | 10 | 0 | 0 | 10 |
| `gap-keys` | K1a,K1b,K1c,K1d,K1e | 7 | 0 | 0 | 7 |
| `gap-maskbm` | G1b | 2 | 0 | 0 | 2 |
| `gap-merge` | G1d | 5 | 0 | 0 | 5 |
| `gap-order` | G1c | 10 | 0 | 0 | 10 |
| `gap-points` | G2 | 6 | 0 | 0 | 6 |
| `gap-shrink` | G1 | 10 | 0 | 0 | 10 |
| `gap-tiny` | G7 | 20 | 0 | 0 | 20 |
| `gap-tracker2` | G5 | 6 | 0 | 0 | 6 |
| `gap-vecmatte` | G1e | 3 | 0 | 0 | 3 |
| `gap-wrap` | G6 | 8 | 0 | 0 | 8 |
<!--/gen:families-->

## Rules without a note

Some points have a stated rule but print no note. Tiny frames and key handles have nothing inside the plugin to grade them against (no result of the original is established at those sizes, and the host computes keyframe interpolation), and the rest belong to the wrapper or to how a host hands footage to the plugin. Each names the render that would test it.

| Point | Stated rule | What would test it |
|---|---|---|
| tiny frames | the motion search never halves below 2x2; a frame with a side of 1 has no search and renders as `vectors=1` (`create-vectors`: the zero vector) | `gap-tiny`: sizes 1x1 to 5x5, observe only |
| a plain key next to a handle | the plain end is a handle of the straight line's speed at an influence of 1/3 | `gap-keys`: keyed `frame_num` with `vectors=1` and Blend, whose blend weights show the progress |
| an eased key's influence | exactly 1/3 | `gap-keys`: the eased cases at 16 bpc |
| overshoot | a float is held inside its valid range; an angle is not | `gap-keys`: handles that overshoot `sensitivity` |
| the curve solve | 64 bisection steps in f64, in the operation order of [timing: key shapes](timing.md#key-shapes) | `gap-keys`, compared bit for bit |
| a point's path | the straight line between its keys at S pixels a second; a segment of length 0 stays put | only together with gaps G2 and G3 |
| retimed audio | varispeed along the effect's time map ([timing: audio](timing.md#audio)) | nothing in the plugin; `tools/test_apply.py` holds the sync |
| a checkout between two composition frames | the footage frame at that exact time (`vfx --clip-fps`) | 24 fps footage in a 60 fps composition, In FPS is Out FPS off, Input Frame Rate 24, No Vectors, Nearest, 100%: output frames 2 and 7 show footage frames 1 and 3 under this rule, 0 and 2 if the host quantises to its frames |
| a new size | the footage is scaled before the effect, which renders at the new size | `create-vectors` on footage at twice the composition's size: vector values in footage or in composition pixels |
| the default length at another clip rate | the composition frames that start inside the layer, ceil(C x F / R) (25 frames at 24 fps give 63 at 60) | the length a host gives such a layer |
| HDR input | the signal as stored, no tone mapping ([ffmpeg: HDR](ffmpeg.md#hdr)) | a 16 bpc render of PQ code values in an unmanaged project |

The `--device` choice has no rule: a device is used only where the gate shows it gives the CPU's bytes ([engine: device](engine.md#device)).

## Model fixtures

Every `*.hex` fixture of the tree, with the cases its header counts and its decoded size:

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

## Regression lock

The lock shows that every setting that rendered on a baseline build renders the same bytes now. A baseline `vfx` and `fx-parity`, built from commit <!--f:lock_ref-->d74f402<!--/f-->, generate the corpora and render every case; a new build must then give, for every case the baseline rendered (exit 0), exit 0, the same SHA-256 and size of stdout, and the same SHA-256 of stderr, so an empty stderr must stay empty and a new note on such a setting is a difference. The bytes are hashed because the 32 bpc pass rule of `fx-parity run` counts -0.0 and 0.0 as equal, and the lock does not. The cases the baseline refused are run on the new build as well, and those that now render are counted.

| Corpus | Cases |
|---|---|
| L1 | `fx-parity gen` with its defaults for the four effects and `selftest-invert`, `selftest-shift`, `selftest-integrate` and `selftest-threads` |
| L2 | `fx-parity gen` of the four effects at sizes 2x2, 3x2, 4x4, 5x4, 8x8, 9x8, 16x16 and 17x16, 6 frames, every case (`--max-cases-per-effect 0`) |
| L3 | the temporal L2 cases of `retime`, `retime-fgbg` and `create-vectors`, run with `--quality draft` |

Every `fg1_matte` to `fg3_matte` layer is removed from the cases, because the baseline refused them. Each case runs with the arguments `fx-parity` gives it, at the default thread count (`RAYON_NUM_THREADS` unset). The record of <!--f:lock_date-->2026-10-28<!--/f-->, `docs/data/lock.json`:

<!--gen:lock-->
| Corpus | Rendered at the baseline | Identical now | Differ | Refused at the baseline | Of those, render now |
|---|---|---|---|---|---|
| L1 | 1,955 | 1,955 | 0 | 74 | 74 |
| L2 | 6,090 | 6,090 | 0 | 444 | 444 |
| L3 | 5,028 | 5,028 | 0 | 492 | 492 |
| all | 13,073 | 13,073 | 0 | 1,010 | 1,010 |
<!--/gen:lock-->

To reproduce it, from the repository root after [Build](../README.md#build), with REF the recorded baseline (<!--f:lock_ref-->d74f402<!--/f-->); the lock folder defaults to `target/parity-lock`, which git ignores, and nothing in it is committed:

```sh
python tools/parity_lock.py base --ref REF
python tools/parity_lock.py corpus
python tools/parity_lock.py refs
python tools/parity_lock.py check --vfx target/release/vfx
python tools/parity_lock.py refused --vfx target/release/vfx
python tools/parity_lock.py record
python -I tools/gen_docs.py
```

`base` builds both tools of the ref in a temporary detached worktree, with the CI profile, and removes the worktree afterwards; `check` exits 1 on any difference and `refused` exits 1 while a refused case still fails; `record` writes `docs/data/lock.json` from a complete `check` (and the last `refused`). `python tools/parity_lock.py --help` lists the flags (`--toolchain`, `--jobs`, `--sets`, `--out`).

## Caveats

- **Platform math library.** A few steps take a value from the platform's math library: the `pow` of Delinearize and the logarithm that picks the forward warp's mip level. Those results can differ in the last bit between platforms, so the bytes of a setting that reaches them are fixed per platform, not across platforms. The lock compares two builds on one machine, and a case is graded on the platform that rendered it.
- **Threads.** The bytes do not depend on the thread count: `fx-parity selfcheck --threads` and the `threads` tests of every plugin check it. The lock runs at the default thread count.
- **16 bit range.** At 16 bpc the effects work on 0 to 32768, so a sample of a frame shown unchanged can come back 1 off in 65535 ([raw frame format](../README.md#raw-frame-format)). A graded 16 bpc case compares that output as it is; there is no tolerance at 16 bpc.
- **32 bit float.** At 32 bpc a sample passes within one unit in the last place, so a graded 32 bpc case is within one ULP of the original, not bit for bit.
- **GPU.** Every figure is of the CPU path. A device backend counts only where a byte comparison gate shows it identical to the CPU path (`fx-parity devcheck`, [engine: device](engine.md#device)), and adds no figure of its own.

## Terms

| Term | Meaning |
|---|---|
| control | one parameter or layer of one of the four effects, a row of `docs/data/controls.tsv` |
| reference render | the output of the original plugin for one case of the checklist, converted to the raw format by the checklist's commands |
| reference family | the cases `fx-parity gen --families` makes for one gap, listed with their settings in the checklist |
| checklist case | one case of a reference family, a row of `docs/data/references.tsv` |
| graded | a case whose reference render `fx-parity run --strict` found bit exact (at 32 bpc within one ULP); a control set by such a case and by no failing one |
| gap | an open point of the original's behaviour, with its stated rule and its note |
| note | the stderr line `vfx: note: ...` a render prints once per gap it used |
| model fixture | hex text of the inputs and expected outputs of an independent model of one engine stage, which the Rust tests reproduce bit for bit |
| differential fixture | a fixture of a model that is not bit exact, which the tests compare within a tolerance |
| baseline | the build of the commit the lock compares against |
| corpus | the cases of one lock set, L1, L2 or L3 |
| identical | exit 0 with the same stdout bytes and the same stderr bytes as the baseline |
| refused | a case the baseline did not render (a nonzero exit) |
| unproven | a control that no graded case has shown to match the original |

## Moving the numbers

A reference render of the original moves the Graded and Unproven rows. From the repository root:

```sh
target/release/fx-parity gen --out target/refpack --families all
target/release/fx-parity checklist --testset target/refpack
target/release/fx-parity run --testset target/refpack --refs target/refpack/refs --strict
python -I tools/gen_docs.py --refresh-data --vfx target/release/vfx --testset target/refpack --refs target/refpack/refs
python -I tools/gen_docs.py --check
```

1. `gen` writes the cases of every family, and `checklist` the instructions for rendering each in the original and converting its export into `refs/`.
2. `run` grades every case that has a reference.
3. `--refresh-data` rebuilds `docs/data/controls.tsv` and `docs/data/gaps.tsv` from `vfx list --json --all`, and `docs/data/references.tsv` from the cases and the grades, then fills every figure; `--check` confirms that nothing drifted. With `--testset` and no `--refs` every case is recorded as `missing`, which is the record today. Commit the snapshots with the change that moved them.

A change to the engine moves the lock: run the lock against the last recorded baseline, and record it again once it shows no difference. A new fixture or test moves the Model row and the test counts on the next `python -I tools/gen_docs.py`.

## Test counts

The tests the figures rest on, counted in the source:

<!--gen:tests-->
| Suite | Counted | Tests |
|---|---|---|
| `fx-cli` | `#[test]` | 133 |
| `fx-core` | `#[test]` | 52 |
| `fx-parity` | `#[test]` | 170 |
| `create-vectors` | `#[test]` | 91 |
| `retime` | `#[test]` | 67 (2 ignored) |
| `retime-engine` | `#[test]` | 755 (12 ignored) |
| `retime-fgbg` | `#[test]` | 101 (1 ignored) |
| `vectors-in` | `#[test]` | 100 |
| `tools/test_apply.py` | `def test_` | 110 |
| `tools/test_apply_geometry.py` | `def test_` | 22 |
| `tools/test_apply_hdr.py` | `def test_` | 38 |
| `tools/test_audio_map.py` | `def test_` | 26 |
| `tools/test_conformance.py` | `def test_` | 47 |
| `tools/test_docgen.py` | `def test_` | 38 |
| `tools/test_ffstage.py` | `def test_` | 25 |
| `tools/test_handoff.py` | `def test_` | 154 |
| `tools/test_parity_lock.py` | `def test_` | 21 |
| `tools/test_repeat_keys.py` | `def test_` | 9 |
| `tools/test_speed_curve.py` | `def test_` | 32 |
| `tools/test_stages.py` | `def test_` | 70 |
<!--/gen:tests-->

A `#[test]` line is one test of `cargo test`, and a `def test_` line one test of its `unittest` suite; an `#[ignore]` test runs only on request ([Tests](../README.md#tests)).
