#!/bin/bash
# Phase A, step 2 on the VPS: build each mutant in the scratch copy mut-src, render the grid, compare.
# A fast build profile is used (opt-level 1, no LTO); m00_control is the unmutated source built the
# same way and must reproduce the base hashes, which shows the profile changes no output.
cd ~/twixtor-parity/retime-time
source ~/.cargo/env 2>/dev/null || true
PY=~/twixtor-parity/venv/bin/python
export CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR=$PWD/mut-target CARGO_PROFILE_RELEASE_OPT_LEVEL=1 \
  CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_PROFILE_RELEASE_INCREMENTAL=true
if [ ! -d mut-src ]; then mkdir mut-src; (cd src && tar cf - --exclude=./target .) | tar xf - -C mut-src; fi
mkdir -p mutbin results/mut logs
ids="m00_control $($PY mutants.py list | cut -f1)"
for id in $ids; do
  if [ ! -x mutbin/$id.vfx ]; then
    $PY mutants.py restore mut-src src
    [ "$id" = m00_control ] || $PY mutants.py apply $id mut-src || { echo "$id: patch failed"; continue; }
    if (cd mut-src && cargo build --release -p fx-cli -j2 >../logs/build-$id.log 2>&1); then
      cp mut-target/release/vfx mutbin/$id.vfx
    else
      echo "$id: build failed, see logs/build-$id.log"
    fi
    $PY mutants.py restore mut-src src
  fi
  [ -x mutbin/$id.vfx ] && $PY render_grid.py --vfx $PWD/mutbin/$id.vfx --testset testset --tsv results/mut/$id.tsv --jobs 2
done
$PY analyze.py mutants --testset testset --base results/base.tsv --mutdir results/mut --out results/mutants_vs_base.json
