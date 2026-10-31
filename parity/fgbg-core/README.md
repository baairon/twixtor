# fgbg-core parity module

Measures `retime-fgbg` without FG matte layers or tracking points against `TwixtorPro_64.aex`.
Graded against the original: the result is `results/summary.json`, and [docs/parity.md](../../docs/parity.md#status) shows it with the other modules.

## Files

| File | What |
|---|---|
| `gen_grid.py` | Grid generator: raw input clips and an fx-parity style `cases.json` (format 1), `grid.tsv`, `out_of_contract.json`. Standard library only. |
| `render_grid.py` | Renders every case with one `vfx` binary, writes id, exit code, size and SHA-256 per case. |
| `compare_runs.py` | Lists the cases whose output differs between two result files. |
| `mutants.json` | The mutants: file, exact text replaced, replacement, occurrence count, what it changes. |
| `run_mutants.py` | Builds each mutant in a scratch tree, renders the grid, writes `changed.txt` per mutant. |
| `make_summary.py` | Writes `results/summary.json`, `results/params_used.tsv`, `results/mutants.tsv` from the result files. |
| `testset/cases.json`, `testset/grid.tsv` | The in-contract cases and their block and tags. Inputs are not kept locally; regenerate with `gen_grid.py`. |
| `testset/out_of_contract.json` | Out of contract candidates with vfx's verdict. Never counted as passes or failures. |
| `results/base.tsv` | Base rewrite output hash and size per case. |
| `results/mutants_summary.tsv`, `results/mutants.tsv` | Per mutant: cases changed against the base rewrite. |
| `results/params_used.tsv` | The rewrite param ids the grid relies on (phase B confirms their pairs in `parity/harness/param_map.tsv`). |

## Re-run on the server (`$WORK/fgbg-core`)

```sh
cp -a $WORK/src/base src            # once
mkdir -p bin results && cp src/target/release/vfx bin/vfx_base
python3 gen_grid.py --out testset --vfx bin/vfx_base
python3 render_grid.py --vfx bin/vfx_base --testset testset --out results/base.tsv --jobs 2
cp -a src src_mut                              # scratch tree for mutants
python3 run_mutants.py --tree src_mut --testset testset --base results/base.tsv --out mutants
python3 make_summary.py                        # after copying the small result files back
```

Mutants are only ever applied inside `src_mut`, and the file is restored after each build.
A mutant is final-killed only when it changes a case against the emulator's output (phase B);
`changed.txt` lists the cases it changes against the base rewrite.

## Grid blocks (`grid.tsv`)

`size` (16 sizes from 1x1 to 64x48 at 3 depths), `warping`, `interp`, `vectors`, `prep`, `sens`, `blur`,
`wrap` (`track_rgb_a`, `video_360`), `display`, `time` (a smaller sample of the retime controls),
`keyed` (keyframes, tag `standin`), `spec` (32 bpc 0, -0, denormals, FLT_MIN, 1, above 1, FLT_MAX,
infinities, NaN; 8 and 16 bpc extremes), `motion` (real motion on larger frames), `alpha`, `noise`,
`mix` (seeded combinations of valid values). Tags: `tiny`, `defaults`, `negzero`, `special`,
`nomatte` (display 3 to 6, which render as Output without mattes), `standin`.

## Phase A status

- 399 in-contract cases, all rendered by the base rewrite (exit 0), 23 out of contract candidates listed (values outside the registered ranges or choices, each refused by `vfx` before rendering; never counted).
- 22 mutants built in the scratch tree `src_mut` on the server and rendered on the grid; 21 change at least one case against the base rewrite (`results/mutants.tsv`). These are not final kills: that needs the emulator's output (phase B).
- M05 (first scanline of a triangle rounds down instead of up) changes no case. By reading the rasteriser it is probably equivalent (a scanline above the top vertex has no active edge, so no span is emitted); this is an argument, not a measurement, and it stays listed as a survivor.
- M20 (frame to time truncation) is only reached by the `time2` block, whose input rates make the frame duration in ticks fractional (24.024 and 24.5 change a case).

## Phase B tooling (ready, tested on stand-in references only)

`phase_b.py` (Python 3 standard library) grades the module against reference renders the moment they exist. Nothing here runs a plugin, and nothing has been compared with an original yet: the selftest uses the rewrite's own renders as stand-in references (`results/selftest/`). `test_phase_b.py` holds the unit tests that need no server (`python3 -m unittest test_phase_b`).

| Subcommand | Does | Writes (under `--results`, default `results/`) |
|---|---|---|
| `grade --refs DIR` | `fx-parity run --max-ulp 0 --json` for the test set, plus a byte audit of the same renders (differing values, max ULP over the whole clip, first differing places) | `grade.json`, `grade.tsv` |
| `ooc --status FILE...` | the harness per-case status (`run_harness.py --summary`, several files merge) moves cases out of contract; the 23 candidates stay listed with what each side did | `ooc.json` |
| `ooc-set --out DIR` | the candidates as a test set the harness can render (`ooc --status` then reads that run) | `DIR/cases.json` |
| `rewrite-records --out DIR` | renders every case with `VFX_CALL_RECORD` (the rewrite side of the call-record check) | `DIR/<case id>.jsonl` |
| `calls --plugin-records DIR --rewrite-records DIR` | per output frame, as sets: clip frames and ticks read, output frame times | `calls.json` |
| `kills --refs DIR` | mutation kills with the definition below, from the saved per-mutant hashes (`mutants/Mxx/results.tsv`) | `kills.json` |
| `summary` | `summary.json` in the existing schema, every count computed from the files above; refuses to run without `grade.json` | `summary.json` |
| `triage` | groups the differing cases, lists the smallest first, writes a ladder of probe cases to `probe/` | `probe/` |
| `selftest` | the stand-in checks | `results/selftest/` |

Rules the tools apply:

- A case with no reference is `no reference`: never a pass, never a failure, not counted as compared.
- A mutant is killed iff its output differs from the reference on at least one in-contract case where the base rewrite equals the reference. A mutant that changes no output, or only cases where the base already differs, survives and is named with the reason.
- Out of contract: the original refuses it (`refused`, `unsupported`), returns an error (`PluginError` / `PF_Err`), or a param lies outside the ranges it registers (`--param-map`, once a plugin has registered). An emulator fault, timeout or `not_run` is not the original's behaviour: it stays in contract as `not_judged` with no reference (`--move-harness-failures` moves them anyway). A candidate the original renders is flagged for review and never counted.
- Values: `bytes_compared` is the bytes of the 8 and 16 bpc references compared, `floats_compared` the float samples of the 32 bpc ones, `differing_values` the samples whose bits differ (a byte, a 16 bit value or a float; -0 against 0 and a changed NaN payload count). `max_ulp` is the largest ULP distance over the whole clip of any compared 32 bpc case, finite pairs only; NaN mismatches are counted in `grade.json` (`nan_mismatch`) and named in the summary blockers. fx-parity's own report gives the figures of a failing case's first failing frame only, which is why `grade` audits the bytes itself and flags any case where the two disagree.
- Cases tagged `standin` count in `cases.logic_parity` when compared.
- `summary` writes `verified` only when at least 150 in-contract cases are compared, all of them, 0 differ, every call record matches and every mutant is killed. It never writes `verify.json`.
- Call records: plugin side is the harness's `records/<case id>.jsonl` (`frame` lines and `cb` lines `inter.checkout_param` with `param_index` 0 or a `layer` note, and `prerender.cb.checkout_layer`); a record without `case_done` status `rendered` is `incomplete_plugin_record` and not compared. The rewrite does not record parameter sample times, so the plugin's are only counted (`param_samples_plugin`).

### When real references exist

On the server, one core (`taskset -c N`), in `$WORK/fgbg-core`:

```sh
T="taskset -c 11 python3 phase_b.py"
# 1. the harness has rendered testset/ into REFS and written records and a status summary
#    (cd $WORK/harness && run_harness.py --testset <this>/testset --refs REFS --records PLUGREC --summary STATUS.json)
$T ooc-set --out ooc_testset                     # optional: render the 23 candidates with the harness too, then pass that status
$T ooc --status STATUS.json [--param-map $WORK/harness/param_map.tsv]
$T grade --refs REFS
$T rewrite-records --out REWREC
$T calls --plugin-records PLUGREC --rewrite-records REWREC [--param-map $WORK/harness/param_map.tsv]
$T kills --refs REFS
$T summary
$T triage                                        # when anything differs; then render probe/ in the harness and
                                                 # fx-parity run ... --json > ladder.json; $T triage --ladder-report ladder.json
```

`summary` and `triage` read only small result files and the local `testset/`: they can also run on the laptop after copying `grade.json`, `ooc.json`, `kills.json` and `calls.json` back (pure Python, no render). Order matters: `ooc` before `grade`/`kills`/`summary` (kills and the summary exclude moved cases). Binaries: `bin/vfx_base` (the phase A base, the graded rewrite), `bin/fx-parity_ulp` (has `--max-ulp`; copied from `$WORK/harness/repo/target/release/`), `bin/vfx_rec` (has `VFX_CALL_RECORD`; same source; its outputs equal `results/base.tsv` hash for hash, checked by the selftest). If the rewrite changes, rerun phase A (`render_grid.py`, `run_mutants.py`) first: `grade` flags a stale `results/base.tsv`.
