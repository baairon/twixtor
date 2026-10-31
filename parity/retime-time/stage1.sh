#!/bin/bash
# Phase A, step 1 on the VPS: own copy of the base build, grid, validation, base render.
# Run from ~/twixtor-parity/retime-time (the scripts of this folder sit there).
set -e
cd ~/twixtor-parity/retime-time
PY=~/twixtor-parity/venv/bin/python
[ -d src ] || cp -a ~/twixtor-parity/src/base src
mkdir -p results
VFX=$PWD/src/target/release/vfx
BASEVFX=~/twixtor-parity/src/base/target/release/vfx
sha256sum "$VFX" "$BASEVFX" | tee results/vfx_sha256.txt
grep -h Twixtor_64 ~/twixtor-parity/bin/SHA256SUMS > results/plugin_sha256.txt
cat results/plugin_sha256.txt
sha256sum src/plugins/retime-engine/src/timeline.rs src/plugins/retime-engine/src/planner.rs src/plugins/retime-engine/src/blend.rs | tee results/engine_sha256.txt
"$VFX" --version
rm -rf testset
$PY gen_grid.py testset
$PY render_grid.py --vfx "$VFX" --testset testset --tsv results/check_all.tsv --check-only
$PY check_oob.py --vfx "$VFX" --testset testset --tsv results/out_of_contract.tsv
~/twixtor-parity/src/base/target/release/fx-parity checklist --testset testset | tail -1
mkdir -p emptyrefs
~/twixtor-parity/src/base/target/release/fx-parity run --testset testset --refs emptyrefs --brief | tail -2 || true
rm -rf base-out
time timeout 500 $PY render_grid.py --vfx "$VFX" --testset testset --tsv results/base.tsv --keep base-out --jobs 2
du -sh testset base-out
