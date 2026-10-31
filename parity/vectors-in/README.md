# vectors-in parity module

Measures the rewrite's `vectors-in` effect (`TwixtorProVectorsIn_64.aex`) against the plugin. Graded against the original: the result is `results/summary.json`, and [docs/parity.md](../../docs/parity.md#status) shows it with the other modules.

## Files

| File | What |
|---|---|
| `gen_grid.py` | Grid generator. Writes `testsets/grid` (fx-parity `cases.json` format 1 plus raw inputs), `case_meta.json` (group, flags, stand-in per case), `params_used.json` and `out_of_contract.json`. Pixel data is closed form or a fixed hash, so it is reproducible. The vector layers are hand made raw vector images. |
| `run_grid.py` | `run` renders a test set with one `vfx` binary (exit code, size, sha256, ms, first stderr line per case; `--keep` also writes `<case id>.<pix_fmt>` reference files). `check` runs `vfx apply --check` on the out-of-contract candidates. |
| `mutants/mutants.json` | The mutant list: file, exact old and new text, kind, what it changes. A copy is `tools/mutants.json` on the server. |
| `mutate.py`, `build_mutants.sh` | Apply one mutant to a scratch copy of the source, build it, render the grid. `m00` is the control (pristine source rebuilt in the scratch copy). |
| `phase_b.py`, `test_phase_b.py` | Phase B tooling and its unit tests ([Phase B](#phase-b)). |
| `compare.py` | Mutant table and `results/summary.json`; every count is computed from the TSVs. |
| `testsets/` | Small copies of `cases.json`, `case_meta.json`, `params_used.json`, `out_of_contract.json` (the inputs are regenerated, never stored here). |
| `results/` | `base.grid.tsv`, `base2.grid.tsv` (determinism re-run), `out_of_contract.tsv`, `mutants/*.grid.tsv`, `mutant_table.tsv`, `mutants.json`, `phase_a.json`, `summary.json`. |

## Re-run (everything on the server, folder `$WORK/vectors-in`; tools copied to `tools/`)

```sh
cp -a $WORK/src/base $WORK/vectors-in/src            # once
cp src/target/release/vfx bin/vfx-base
python3 tools/gen_grid.py --out testsets/grid
python3 tools/run_grid.py run --vfx bin/vfx-base --testset testsets/grid --out results/base.grid.tsv --keep refs/grid --jobs 2
python3 tools/run_grid.py check --vfx bin/vfx-base --ooc testsets/grid/out_of_contract.json --out results/out_of_contract.tsv
bash tools/build_mutants.sh m00 m01 ... m24        # long: run as a job, 2 cores
python3 tools/compare.py --results results --mutants tools/mutants.json --testset testsets/grid --date YYYY-MM-DD
```

`refs/grid` holds the base rewrite's outputs under the names `fx-parity run --refs` expects, so `fx-parity run --testset testsets/grid --refs refs/grid --vfx bin/vfx-X --max-ulp 0` grades any build against the base rewrite (a format check, not a parity result).

## What the grid is

Each case: an input clip of `frames` frames (footage), optional `source`, `vectors_prev`, `vectors_next` and `fg1_matte` to `fg3_matte` layers, 8, 16 or 32 bpc, sizes 1x1 to 64x48, rates 12 to 60 fps, and params from the rewrite's own ids. Default case: 16x12, 30 fps, 4 frames, `speed_percent=50`, `vectors_max_displace=32`, vector images written with max displace 32, textured footage moving (1.5, -0.75) pixels a frame with matching vectors (Next is the motion forward, Prev the negated motion of the frame before, with other gains per axis so a swap shows).

Groups (`group` in `case_meta.json`): size, tiny, defaults, warp, speed, framenum, inrate, mark, display, blur, vecscale, maxdisp, keyed, vecfield, layers, footage, sp_src (32 bpc special values in the footage), sp_vecR / sp_vecG / sp_vecA / sp_vecB (special values in the red, green, alpha and blue channel of the vector layers), fg (matte layers). The special values are 0, -0, the smallest and largest denormal, FLT_MIN, 1, 2.5, FLT_MAX, +inf, -inf and two NaN signs.

Flags in `case_meta.json`: `tiny` (sides of 5 or less, the plugin may refuse them), `reads_past_end`, `reads_before_start` (the speed makes output frames ask for clip frames outside the clip; what the host returns there is a harness decision, move these to out of contract if the plugin refuses), `long_source` (a `source` layer longer than the output, so a speed above 100 stays inside the clip). `stand_in: keyframe_interpolation` marks cases with keyed params.

## Out of contract

* `testsets/grid_v2/out_of_contract.json`: 31 parameter values outside the registered ranges, popup values the effect does not have and params it does not have (33 in the older `testsets/grid/out_of_contract.json`; `ooc12` and `ooc13`, the two `output_frame_rate` values, are gone because the plugin has no such control and the rewrite no longer has the param; the other ids are unchanged). Checked with `vfx apply --check` only (the rewrite's own range check); the plugin is not asked.
* Grid cases the base rewrite itself refuses (exit 3, for example NaN vector samples with `warping=1`) are listed in `results/phase_a.json` under `base_refused` and counted as out of contract, not as passes or failures.

## Plugin params and layers the grid relies on (rewrite ids; the full value lists are in `testsets/params_used.json`)

Params: `display`, `src_mark_segments`, `vectors_max_displace`, `vec_scale_x`, `vec_scale_y`, `in_fps_is_out_fps`, `input_frame_rate`, `time_remap_mode`, `speed_percent`, `frame_num`, `frame_interp`, `warping`, `blur_compensation`, `fg1_matte_chan`, `fg1_inv_matte_shrink`. Not varied: `fg2_matte_chan`, `fg3_matte_chan`, `fg2_inv_matte_shrink`, `fg3_inv_matte_shrink` (defaults only).

Layers: `input` (the footage clip), `source`, `vectors_prev`, `vectors_next`, `fg1_matte`, `fg2_matte`, `fg3_matte`.

## Notes for phase B

* Plugin params and layers the grid uses, by rewrite id: `testsets/grid/params_used.json` (copied to `testsets/params_used.json`).
* `frames` is both the output length and the footage length (fx-parity needs that). Vector layers have `frames` frames except in the `layers` group (`short-vectors`, `vectors-1-frame`, `prev-short-next-full`, `short-next-only`), where they are shorter on purpose: the error fill, or an inverted flow, for the frames past their end.
* fps strings: `30`, `24`, `25`, `12`, `60`, `30000/1001`, `24000/1001`, `23.976`.
* Kills in `results/mutant_table.tsv` are against the base rewrite and are final only once the same cases are compared with the plugin.

## Phase B

`phase_b.py` (Python 3 standard library only) grades references, judges the contract, checks call records, counts mutant kills and writes `results/summary.json`. It never runs the plugin and never touches the rewrite. On the server copy it to `tools/` beside the phase A scripts and run it from `$WORK/vectors-in` under `taskset -c N`; every result goes to `--out-dir` (default `results/`). `python3 -m unittest test_phase_b` needs no server; `phase_b.py selftest` runs the stand-in oracles and writes only under `results/selftest/`.

When the harness has rendered the test set (`run_harness.py --testset testsets/grid --refs refs/plugin --records records/plugin --summary run_summary.json`), in this order:

```sh
python3 tools/phase_b.py ooc --status run_summary.json                     # which cases the original refuses or errors on: ooc.json
python3 tools/phase_b.py grade --refs refs/plugin --vfx bin/vfx-base      # fx-parity run --max-ulp 0 --json, plus a value level check: grade.json, grade.tsv
python3 tools/phase_b.py rewrite-records --out records/rewrite            # the rewrite's VFX_CALL_RECORD logs (needs a build that writes them, see below)
python3 tools/phase_b.py calls --plugin-records records/plugin --rewrite-records records/rewrite --param-map ../harness/param_map.tsv   # calls.json
python3 tools/phase_b.py kills --refs refs/plugin                          # kills.json (add --rerender to re-render with bin/vfx-m01..m24)
python3 tools/phase_b.py summary                                           # summary.json, refuses without grade.json
python3 tools/phase_b.py triage                                            # groups the differences, writes a ladder of probe cases to probe/
```

Rules the tool applies:

* A case without a reference is `no reference`, never a pass or a failure. A case in `ooc.json`'s `moved_out` (the original refused it, errored on it or timed out; or, before `ooc` runs, the base rewrite refuses it) is `out of contract`, never a pass or a failure. A case the rewrite refuses and the original renders is a real comparison, listed in `back_in_contract`; it fails.
* `grade` counts differing values itself (bytes at 8 bpc, 16 bit samples at 16 bpc, float bit patterns at 32 bpc) beside fx-parity's pixel counts and flags any case where the two disagree. `values.max_ulp` is the largest finite ULP distance; a pair with a NaN and different bits is counted in `nan_mismatch_values`, not in the ULP. A case the rewrite cannot render counts as differing and adds no values.
* `kills`: a mutant is killed iff its output (or a failure to render) differs from the reference on at least one in-contract case where the base rewrite equals the reference. The default reads the saved phase A sha256 rows (`results/base.grid.tsv`, `results/mutants/*.grid.tsv`); `--rerender` renders with `bin/vfx-m00` (the control, which must survive) and `bin/vfx-mNN` instead. Survivors are named in `kills.json` and `summary.json`.
* `calls`: reads are compared per output frame as sets of (layer, exact time), with the frame index checked for each; `--ordered` also checks the order, `--exclude-layers input` leaves a layer out, `--ignore-outside` leaves out reads outside a clip. The plugin side is the harness's `records/<case>.jsonl`: layer reads come from `prerender_cb.checkout_layer` (smart render) and, for a layer parameter, `inter.checkout_param`, with the layer named by `param_map.tsv` (`ae_index` is the param index). Parameter sample times are recorded by the plugin side and not by the rewrite, so they are counted and not compared. Cases with no plugin record, an incomplete one, or out of contract are listed as not compared.
* `bin/vfx-base` and the mutant builds were made from a source copy that has no `VFX_CALL_RECORD`; `rewrite-records` needs a build of the current tree (default: the harness copy's `vfx`) and checks that it renders like the base rewrite.
* `summary` computes every count from the result files, keeps `killed` at 0 without `kills.json`, refuses to run without `grade.json` and refuses results made against another contract (`--allow-stale` overrides). Verified needs at least 150 in-contract cases compared, every in-contract case compared, 0 differing, every call record matching and every mutant killed. `Re-run verified` is a separate step (`results/verify.json`), never written here.
* `triage` prints differing against compared per group, tag and setting value, lists the smallest failing cases first, and for the smallest writes a ladder to `probe/` (an fx-parity test set that reuses the grid's inputs): the case itself, the base params, no params, each non-default param reverted, each alone on the base, the layer subsets, and its params on the inputs of the grid's base cases. Run it through the harness, then `grade --testset probe --refs probe/refs --out-dir probe/results`.

## Test set without `output_frame_rate`

`testsets/grid_v2/` replaces `testsets/grid/` (which is unchanged) for the rewrite that no longer has the `output_frame_rate` param: the plugin has no Output: Frame Rate control, so the rewrite refuses the param as unknown and `vfx list` shows 19 params. `gen_grid.py` dropped the three in-contract `outrate` cases (`output_frame_rate` 0.10000000149011612, 12.5 and 1000; without the param each is the case `speed50-fps24`, which the grid already has) and the candidates `ooc12` and `ooc13`. The set has 448 in-contract cases (451 before) and 31 out of contract candidates (33 before); the remaining 448 cases are the old ones with the same ids, params and layers, and all 420 input files are byte for byte the old ones. Make it with `python3 tools/gen_grid.py --out testsets/grid_v2` and pick it up with `--testset testsets/grid_v2` (`run_harness.py`, `fx-parity run`, every `phase_b.py` subcommand). Rendered with `bin/vfx-base` it gives the hashes of `results/base.grid.tsv` for all 448 cases. The phase A result files still describe the 451 case grid.
