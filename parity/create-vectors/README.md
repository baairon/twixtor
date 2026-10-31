# create-vectors parity

Measures the Rust `create-vectors` effect against `CreateMotionVectors_64.aex` (run under the emulator by `parity/harness/`). Status and counts: `results/summary.json`.

Graded against the original: the result is `results/summary.json`, and [docs/parity.md](../../docs/parity.md#status) shows it with the other modules.

## Files

| File | What |
|---|---|
| `gen_grid.py` | writes the test set: `cases.json` (fx-parity format 1), `inputs/`, `case_meta.json` (group, tags, stand-in label per case), `out_of_contract.json`, `params_used.json` |
| `run_grid.py` | renders a test set with one `vfx` binary, writes `id, exit, size, sha256, note` per case |
| `build_mutants.py` | builds each mutant of `mutants/mutants.json` in a scratch copy, keeps `mut/<id>/vfx` |
| `compare_mutants.py` | which cases each mutant changes against the base rewrite (`results/mutant_matrix.json`) |
| `make_summary.py` | writes `results/summary.json` from the result files |
| `mutants/mutants.json` | the mutant list: file, exact old and new text, what it changes |
| `results/` | `base_outputs.tsv` (SHA-256 and size per case), `mutant_<id>.tsv`, `mutant_matrix.json`, `out_of_contract_base.tsv`, `params_used.json`, `summary.json` |

Frames and builds stay on the server (`$WORK/create-vectors/`: `grid/`, `src/`, `msrc/`, `mut/`); only the small files above come back.

## Re-run (on the server, in `$WORK/create-vectors/`, at most 2 cores)

```sh
cp -a $WORK/src/base src            # once
python3 gen_grid.py --out grid
python3 run_grid.py --vfx src/target/release/vfx --set grid --out results/check.tsv --check
python3 run_grid.py --vfx src/target/release/vfx --set grid --out results/base_outputs.tsv
python3 run_grid.py --vfx src/target/release/vfx --set grid --cases grid/out_of_contract.json --out results/out_of_contract_base.tsv
python3 build_mutants.py --mutants mutants/mutants.json --src src --work msrc --out mut --budget 400
for d in mut/*/; do id=$(basename $d); python3 run_grid.py --vfx $d/vfx --set grid --out results/mutant_$id.tsv; done
python3 compare_mutants.py --base results/base_outputs.tsv --mutants mutants/mutants.json --dir results --out results/mutant_matrix.json --set grid
python3 make_summary.py --root . --date YYYY-MM-DD
```

`fx-parity run --testset grid --refs REFS` grades emulator references named `<case id>.<pix_fmt>` (`rgba`, `rgba64le`, `gbrapf32le`).

## The grid

Clips of 3 frames (a few of 1, 2 or 5) at 24 fps unless the case says another rate. Output frame n is the vector image between frame n and its neighbour, so frame 0 with Previous and the last frame with Next see a missing neighbour. Sources are analytic (sums of sinusoids shifted by a fixed motion, discs, boxes, zoom, rotation, noise, flat, static, alpha ramps, holes, extremes, HDR values), so the same bytes come out on every run. Special 32 bpc values (0, -0, smallest and largest denormal, FLT_MIN, 1, 2.5, FLT_MAX, +inf, -inf, NaN, negative NaN) enter as exact bit patterns, as uniform frames, as a patch in the colour planes, as a patch in the alpha plane and as a mix.

Out of contract candidates are listed in `grid/out_of_contract.json` and rendered once with the base only (`results/out_of_contract_base.tsv`): values outside the registered ranges (`vfx` refuses them with exit 3) and layers of another size (`vfx` reads the file as a partial frame, exit 4). They are not part of any pass or fail count.

Not covered: `--quality draft` (a `Case` carries no quality), `--clip-fps` and `--layer-fps`, masks (`First Mask`, `Last Mask`, not expressible), keyframed params except the 5 stand-in cases.

## Phase B tooling (`phase_b.py`, `test_phase_b.py`)

`phase_b.py` (Python 3, standard library only) grades emulator references, judges out-of-contract cases, compares call records, computes mutant kills and writes `results/summary.json` from the result files. It does not run the plugin. Its checks on stand-in references (the rewrite's own renders) are `python3 phase_b.py selftest` (outputs under `results/selftest/`, log in `results/selftest/run.log`); the unit tests are `python3 -m unittest test_phase_b`.

| Subcommand | Does | Writes (in `--results`, default `results/`) |
|---|---|---|
| `grade --refs DIR` | `fx-parity run --max-ulp 0 --json` over the grid, then its own bit comparison of every differing case (values, first difference, true max ULP over all frames) | `grade.json`, `grade.tsv` |
| `ooc --status FILE ...` | re-judges cases from the harness's per-case status (`--param-map`, `--records` optional) | `ooc.json` |
| `ooc-set --out DIR` | a test set of the 20 candidates, for `run_harness.py` | `DIR/cases.json` |
| `calls --plugin-records DIR --rewrite-records DIR` | per output frame, set comparison of the clip frames and layers read | `calls.json` |
| `record-rewrite --out DIR` | runs a `VFX_CALL_RECORD` capable `vfx` over the grid | `DIR/<case id>.jsonl` |
| `kills --refs DIR` | mutant kills from the saved phase A mutant results (`--rerun` renders base and `mut/<id>/vfx` now) | `kills.json` |
| `summary` | the summary document, every count computed; refuses without `grade.json` | `summary.json` |
| `triage` | co-occurrence table, smallest differing cases, ladder of probe cases | `probe/cases.json`, `probe/ladder.json` |

When real references exist (on the server, in `$WORK/create-vectors/`, one core, for example `taskset -c 10`):

```sh
H=$WORK/harness
python3 $H/run_harness.py --testset grid --refs refs -j1 --records records --summary run_status.json
python3 phase_b.py ooc-set --out ooc_set
python3 $H/run_harness.py --testset ooc_set --refs refs_ooc -j1 --records records --summary run_status_ooc.json
python3 phase_b.py grade --refs refs --cross-check
python3 phase_b.py ooc --status run_status.json --status run_status_ooc.json --param-map $H/param_map.tsv --records records
python3 phase_b.py calls --plugin-records records --rewrite-records rewrite_records --param-map $H/param_map.tsv --run-rewrite
python3 phase_b.py kills --refs refs
python3 phase_b.py summary
python3 phase_b.py triage
```

Candidates the plugin renders inside its registered ranges come back into contract (`ooc.json`, `candidates_back_in_contract`): grade them with `grade --testset ooc_set --refs refs_ooc --name grade_ooc` and pass `summary --extra-grade grade_ooc`. After a fix to the rewrite, rebuild the mutants (`build_mutants.py`) and rerun `grade` and `kills` (`kills` flags results from a different `vfx` build as stale). `triage` writes a probe test set; run it through the harness, then `grade --testset probe --refs probe_refs --name ladder` and `triage --ladder-result results/ladder.json` print which settings are necessary and sufficient.

Rules in the tool: a case without a reference is "no reference" (neither pass nor fail); a mutant is killed only by a case where the base rewrite already equals the reference; harness faults (emulator crashes, timeouts, unimplemented mock callbacks) never move a case out of contract, only a real plugin error (`PluginError`, a `PF_Err`), a refusal or a value outside the registered ranges does; the status becomes `verified` only when at least 150 in-contract cases are compared with no gap, none differ, every call record matches, every mutant is killed and nothing is stale or unjudged (the re-run by another agent stays a separate step).
