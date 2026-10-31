#!/bin/bash
# Builds mutants one at a time in a scratch copy of the source and renders the grid with each binary.
# usage (on the VPS, in ~/twixtor-parity/vectors-in): bash tools/build_mutants.sh m00 m01 ...
# m00 is the control: the pristine tree rebuilt in the scratch copy (its output must equal bin/vfx-base).
set -u
R=$HOME/twixtor-parity/vectors-in
PY=$HOME/twixtor-parity/venv/bin/python
. "$HOME/.cargo/env" 2>/dev/null
cd "$R" || exit 1
[ -d scratch ] || cp -a src scratch
mkdir -p bin results/mutants
for id in "$@"; do
  m=$id; [ "$id" = m00 ] && m=none
  $PY tools/mutate.py --pristine "$R/src" --scratch "$R/scratch" --mutants tools/mutants.json --id "$m" || { echo "$id: mutate failed"; continue; }
  rm -f scratch/target/release/vfx
  t0=$(date +%s)
  if ! (cd scratch && CARGO_BUILD_JOBS=2 cargo build --release -p fx-cli -j2 2>&1 | tail -2); then echo "$id: build failed"; continue; fi
  [ -f scratch/target/release/vfx ] || { echo "$id: no binary"; continue; }
  cp scratch/target/release/vfx "bin/vfx-$id"
  echo "$id: built in $(( $(date +%s) - t0 )) s"
  $PY tools/run_grid.py run --vfx "bin/vfx-$id" --testset testsets/grid --out "results/mutants/$id.grid.tsv" --jobs 2
done
$PY tools/mutate.py --pristine "$R/src" --scratch "$R/scratch" --mutants tools/mutants.json --id none
