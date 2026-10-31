# fgbg-layers parity module

Measures `retime-fgbg` with FG matte layers (`fg1_matte` to `fg3_matte`), the inverse matte shrink,
layer order, the `display` groups, tracking points and Draw Geom against `TwixtorPro_64.aex`.
Graded against the original: the result is `results/summary.json`, and [docs/parity.md](../../docs/parity.md#status) shows it with the other modules.

The second tracker (`main_bg_track`, `fgN_track`) is not a parameter of `retime-fgbg` (the rewrite's
`retime-fgbg` has no Track control, every layer tracks), so rule G5 has no case here and no mutant.
Of the rules G1 to G6 the grid covers G1, G1b, G1c, G1f, G1g, G1h, G2, G3 and G6 on the
`retime-fgbg` side; G1d, G1e (create-vectors, vectors-in), G4 (create-vectors Debug) and G5 do not
reach this effect.

## Files

| File | What |
|---|---|
| `gen_grid.py` | Grid generator: raw footage clips, raw matte layer clips, an fx-parity style `cases.json` (format 1, with `layers`), `grid.tsv`, `out_of_contract.json`. Standard library only. |
| `render_grid.py` | Renders every case with one `vfx` binary (`--layer` for the mattes), writes id, exit code, size and SHA-256 per case. `--cases FILE` renders another list. |
| `split_refused.py` | Moves cases the base rewrite refuses at render from `cases.json` to `out_of_contract.json` (none so far). |
| `compare_runs.py` | Lists the cases whose output differs between two result files. |
| `mutants.json` | The mutants: file, exact text replaced, replacement, occurrence count, what it changes. |
| `run_mutants.py` | Builds each mutant in a scratch tree, renders the grid, writes `changed.txt` per mutant. |
| `make_summary.py` | Writes `results/summary.json`, `results/params_used.tsv`, `results/mutants.tsv` from the result files. |
| `testset/cases.json`, `testset/grid.tsv` | The in-contract cases and their block and tags. Inputs are not kept locally; regenerate with `gen_grid.py`. |
| `testset/out_of_contract.json` | Out of contract candidates with vfx's verdict. Never counted as passes or failures. |
| `results/base.tsv` | Base rewrite output hash and size per case. |
| `results/ooc.tsv` | The base rewrite's exit code for each out of contract candidate (informational). |
| `results/mutants_summary.tsv`, `results/mutants.tsv` | Per mutant: cases changed against the base rewrite. |
| `results/params_used.tsv` | The rewrite param and layer ids the grid relies on (phase B confirms their pairs in `parity/harness/param_map.tsv`). |

## Re-run on the server (`$WORK/fgbg-layers`)

```sh
cp -a $WORK/src/base src            # once
mkdir -p bin results && cp src/target/release/vfx bin/vfx_base
python3 gen_grid.py --out testset --vfx bin/vfx_base
python3 render_grid.py --vfx bin/vfx_base --testset testset --out results/base.tsv --jobs 2
python3 split_refused.py --testset testset --base results/base.tsv
python3 render_grid.py --vfx bin/vfx_base --testset testset --cases testset/out_of_contract.json --out results/ooc.tsv --jobs 2
cp -a src src_mut                              # scratch tree for mutants
python3 run_mutants.py --tree src_mut --testset testset --base results/base.tsv --out mutants
python3 make_summary.py                        # after copying the small result files back
```

Mutants are only ever applied inside `src_mut`, and the file is restored after each build.
A mutant is final-killed only when it changes a case against the emulator's output (phase B);
`changed.txt` lists the cases it changes against the base rewrite.

## Grid blocks (`grid.tsv`)

`size` (sizes 1x1 to 64x48 at 3 depths), `layers` (1 to 3 layers x warping x depth), `chan`
(`fgN_matte_chan` 1 to 4, permutations over three layers), `shrink` (`fgN_inv_matte_shrink` 0, -0,
fractions, mid values, 10; on the background view and the output), `display` (1 to 6 with 1 to 3
layers), `fast` (exact positions, No Vectors, Mark Segments: the composed group without warp),
`interp`, `sens` (`fgN_sensitivity`, `main_bg_sensitivity`), `blur`, `order` (matte products, layer
order), `count` (a layer counts only after the one before it), `mattes` (empty, full, grey, hard,
soft, noise, ramps, values at the 0.95 threshold), `spec` (32 bpc 0, -0, denormals, FLT_MIN, 1,
above 1, FLT_MAX, infinities, NaN in footage and in mattes), `points` (`ptN_use`, `ptN_pos`),
`geom` (`draw_geom`, `ptN_color`, positions at the edges, outside and fractional), `wrap`
(`track_rgb_a` with layers), `mix` (seeded combinations of valid values).
Tags: `tiny`, `defaults`, `negzero`, `special`, `standin` (keyframed point values, logic parity).
Matte layers are raw clips named `m-<kind>-<box>-WxH.dD.fN.<pix_fmt>`; the box is the footage object
the matte follows (`hard:j` on the case list takes another box, to test layer order).

## Phase B tooling (`phase_b.py`, standard library only)

Ready and tested on stand-in references; nothing has been compared with the plugin. `test_phase_b.py`
(`python3 -m unittest test_phase_b`) covers the logic that needs no server; `python3 phase_b.py selftest` (on the server, one core)
runs the stand-in oracles and writes only under `results/selftest/`.

| Subcommand | Does | Writes |
|---|---|---|
| `grade --refs DIR` | `fx-parity run --max-ulp 0 --json` on the test set, then the exact differing values and whole-clip max ULP of each differing case (the report only gives the first failing frame) | `results/grade.json`, `results/grade.tsv`, `results/gradeset/` |
| `ooc --status FILE [--param-map FILE] [--also timeout,...]` | judges every case from the harness status and the registered ranges; moves refused, plugin error, unregistered param and out of range cases out of contract; judges the 35 candidates; `--make-testset DIR` writes the candidates as a test set for the harness | `results/ooc.json` |
| `calls --plugin-records DIR --rewrite-records DIR` | per output frame set comparison of layer reads (layer, clip frame index, time); first differing frame per case | `results/calls.json` |
| `record-rewrite --vfx BIN --out DIR` | runs the rewrite with `VFX_CALL_RECORD` over the test set and checks the output still equals `results/base.tsv` | `DIR/<case id>.jsonl` |
| `kills --refs DIR` | mutant kills: killed iff the mutant output differs from the reference on a case where the base output equals it; reads the saved `mutants2/`, `mutants/` per mutant outputs, else the changed lists; `--rebuild --tree src_mut` rebuilds first (touches files so cargo cannot reuse a build) | `results/kills.json` |
| `summary` | the `summary.json` schema, every count computed from the result files; refuses without `grade.json` | `results/summary.json` |
| `triage [--probe-refs DIR]` | groups differing cases by block, gap, tags and parameter values, lists the smallest failing cases, writes a ladder test set for the smallest into `probe/`; with `--probe-refs` reads the ladder result | `probe/` |

With real references (after the emulator renders): `python3 phase_b.py ooc --status run_summary.json`, `grade --refs REFS`,
`record-rewrite --vfx VFX --out recs_rewrite`, `calls --plugin-records RECORDS --rewrite-records recs_rewrite`,
`kills --refs REFS`, `summary`, `triage`.

Notes: `testset/cases.json` has no `inputs` list, which `fx-parity` requires, so `grade` runs on a copy in `results/gradeset/`
(same inputs through a link); `testset/` itself is not changed. `logic_parity` counts compared cases that use either stand-in
(keyed point values, matte layers handed over as clips). A case that did not render for a reason that is not the plugin's
(timeout, emulator fault, harness limit, not run) stays in contract with no reference.

Test set for the current `fx-parity`: `testset_v2/` (next to `testset/`, which is unchanged). `gen_grid.py` now writes the `inputs` list
that `fx-parity` requires, and `testset_v2/` is the same set: the same 386 cases (ids, params, layers, rates) and the same 298 input
files byte for byte; rendered with `bin/vfx_base` it gives the hashes of `results/base.tsv` for all 386 cases. Make it with
`python3 gen_grid.py --out testset_v2 --vfx bin/vfx_base` and `python3 split_refused.py --testset testset_v2 --base results/base.tsv`.
Pick it up with `--testset testset_v2` on every `phase_b.py` subcommand and on `run_harness.py`, or grade it directly with
`fx-parity run --testset testset_v2 --refs REFS --max-ulp 0 --json --vfx VFX`; no `results/gradeset/` copy is made for it.
