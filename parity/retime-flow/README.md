# retime-flow parity module

Measures the rewrite's `retime` effect (motion vectors 2 to 5, smart blend, blur compensation, 360 video, sensitivity, prep, quality, 8/16/32 bpc) against `Twixtor_64.aex`. Graded against the original: the result is `results/summary.json`, and [docs/parity.md](../../docs/parity.md#status) shows it with the other modules.

## Files

| File | What |
|---|---|
| `gen.py` | Grid generator. Writes `testsets/grid` and `testsets/draft` (fx-parity `cases.json` format 1 plus raw inputs) and `testsets/out_of_contract.json`. Pixel data is closed form or fixed-seed RandomState, so it is reproducible. |
| `run_grid.py` | Renders a test set with one `vfx` binary: per case exit code, bytes, sha256, ms, first stderr line. `check` runs the out-of-contract candidates through `vfx apply --check`. |
| `mutants/mutants.json` | The mutant list: file, exact old and new text, kind, what it changes. |
| `mutate.py`, `build_mutants.sh` | Apply one mutant to a scratch copy, build it, render both test sets. `m00` is the unmodified control. |
| `compare.py` | Mutant table (`results/mutant_table.tsv`, `results/mutants.json`) and `results/summary.json`, all counts computed from the TSVs. |
| `cases/` | Small copies of `cases.json` and `out_of_contract.json` (inputs are regenerated, never stored here). |
| `results/` | `base.*.tsv`, `base2.*.tsv` (determinism rerun), `mutants/*.tsv`, `out_of_contract.tsv`, `params_used.json`, `summary.json`. |

## Re-run (everything on the server, folder `$WORK/retime-flow`)

```sh
cp -a $WORK/src/base $WORK/retime-flow/src          # once
python3 tools/gen.py --out testsets
cp src/target/release/vfx bin/vfx-base
python3 tools/run_grid.py run --vfx bin/vfx-base --testset testsets/grid  --out results/base.grid.tsv  --keep refs/grid  --jobs 2
python3 tools/run_grid.py run --vfx bin/vfx-base --testset testsets/draft --out results/base.draft.tsv --keep refs/draft --jobs 2
python3 tools/run_grid.py check --vfx bin/vfx-base --ooc testsets/out_of_contract.json --out results/out_of_contract.tsv
bash tools/build_mutants.sh m00 m01 ... m19        # long: run as a job
python3 tools/compare.py --results results --mutants tools/mutants.json --cases-dir testsets --date YYYY-MM-DD
```

`--keep` also writes each output as `<case id>.<pix_fmt>`, the reference file name fx-parity expects, so `fx-parity run --testset testsets/grid --refs refs/grid --vfx bin/vfx-X` grades any binary against the base rewrite (used here only to check the test set format).

## Notes for phase B

- Cases in `testsets/draft` carry `"quality": "draft"`. fx-parity has no quality flag; use a wrapper as `--vfx` that appends `--quality draft` (`wrapper/vfx-draft` on the server) and set the matching quality on the host side.
- Extra keys in `cases.json` (`group`, `quality`, `stand_in`, `flags`) are ignored by fx-parity.
- `flags: reads_past_end` / `reads_before_start`: the speed makes late (or early) output frames ask for clip frames outside the clip. What the host returns there is a harness decision; move these to out of contract if the plugin or host refuses.
- Output frame count equals input frame count (fx-parity has one `frames` field), so speeds above 100% always run past the clip end.
- `stand_in: keyframe_interpolation`: keyed `speed_percent`; the interpolation between keys is done by the host in After Effects.
- Plugin params used, by rewrite param id: `results/params_used.json`.

## Mutants (phase A)

`mutants/mutants.json` holds 21 mutants (`m01` to `m21`) plus the control `m00` (pristine source rebuilt in the scratch copy, changes 0 cases). `results/mutant_table.tsv` lists, per mutant, how many cases of the 345 change against the base rewrite and in which groups; `results/mutants.json` has the case ids. All 21 change at least one case. That is not a kill: a mutant is killed only when it also differs from the emulator's output, which phase B decides.

Replaced during phase A because they could not change any output: a single precision reciprocal in the flow upsample (division through f64 then f32 rounds like a direct f32 division, so it is equivalent), and a single precision logarithm in the start level (0 of 100 off-integer sensitivities differed). Both were swapped for a boundary mutant.

Weak spots to watch in phase B: `m10` (Nearest tie), `m16`, `m17` and `m04` change 3 to 7 cases each, `m13` only the large noise cases with `prep=3`.

## Phase B tooling (`phase_b.py`, `test_phase_b.py`)

Standard library Python. Ready for the moment `refs/<case id>.<pix_fmt>`, `records/<case id>.jsonl` and the harness status exist; it has only been run against stand-in oracles (`results/selftest/`), never against an original. On the server it lives in `tools/` next to the phase A scripts; run everything from `$WORK/retime-flow` with `taskset -c N`. It needs an `fx-parity` build that knows `run --max-ulp` (the harness copy, `$WORK/harness/repo/target/release/fx-parity`; pass `--fx-parity` or set `FX_PARITY`).

| Subcommand | Does | Writes (in `results/`) |
|---|---|---|
| `grade --refs DIR` | `fx-parity run --max-ulp 0 --json` on `testsets/grid` and on `testsets/draft` (through a generated wrapper that appends `--quality draft`), then a value by value analysis of every differing case (values, pixels, channels, first difference, max ULP over the whole case, NaN and -0 counts) | `grade.json`, `grade.tsv` |
| `ooc --status FILE` | moves cases the harness reports as refused / unsupported / error / timeout out of contract, with what each side did; an emulator fault or a case that never ran is listed as unjudged, not moved (`--faults-out-of-contract` overrides); the 17 candidates are judged from the same status file; `--param-map` lists cases outside registered slider ranges (`--apply-ranges` moves them) | `ooc.json` |
| `calls --plugin-records DIR --rewrite-records DIR` | per output frame, as sets (`--ordered` for sequences): frame times, clip reads (layer, time, clip index or outside the clip); first differing frame per mismatching case; plugin parameter samples are only checked against the frame time (the rewrite does not record them) | `calls.json` |
| `records --vfx BIN --out DIR` | runs the rewrite with `VFX_CALL_RECORD` over the whole test set | `DIR/<case id>.jsonl`, `DIR/index.tsv` |
| `kills --refs DIR` | mutation kills with the status rule's definition; `--mode tables` uses the phase A sha256 tables, `--mode rerender` renders with `bin/vfx-base` and `bin/vfx-mNN` | `kills.json` |
| `summary` | `summary.json` in the existing schema, every count computed from the result files; refuses without `grade.json`; `killed` is 0 without `kills.json`; status is `verified` only when the rule holds | `summary.json` |
| `triage` | groups differing cases by group, depth, size, frames, source, quality, flags and each parameter value; smallest failing cases first; writes a ladder of probe cases for the smallest one to `probe/` (an fx-parity test set) | `triage.json`, `../probe/` |
| `selftest` | the stand-in oracles, only under `results/selftest/` | `selftest/report.json` |

Real run, once references exist (`R` = the folder with the references, `S` = the harness status JSON):

```sh
cd $WORK/retime-flow
python3 tools/phase_b.py grade --refs $R
python3 tools/phase_b.py ooc --status $S --param-map $WORK/harness/param_map.tsv   # summary and kills read ooc.json, no regrade needed
python3 tools/phase_b.py records --vfx bin/vfx-base --out results/rewrite_records
python3 tools/phase_b.py calls --plugin-records $H/records --rewrite-records results/rewrite_records --param-map $WORK/harness/param_map.tsv
python3 tools/phase_b.py kills --refs $R
python3 tools/phase_b.py summary
python3 tools/phase_b.py triage          # when cases differ
```

Count conventions in `summary.json`: `in_contract` is every case of the two test sets minus the ones `ooc` moved; `compared` is the in-contract cases that have a reference; `logic_parity` is the compared cases that depend on a labelled stand-in (a subset of `compared`, not an extra group); `out_of_contract` is the moved cases plus the candidates the original did not render; `max_ulp` is the largest finite float distance over all compared 32 bpc cases (NaN mismatches are counted in `grade.json` as `nan_mismatch`). A ladder is graded with `grade --testset probe --refs PROBE_REFS --results results/probe` (a test set that mixes qualities is split per quality internally).
