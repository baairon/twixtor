#!/bin/bash
# Build mutants one at a time in a scratch copy and render both test sets with each binary.
# usage (on the VPS, in ~/twixtor-parity/retime-flow): bash tools/build_mutants.sh m00 m01 ...
# m00 is the control: the pristine source restored into the scratch copy and rebuilt.
set -u
R=$HOME/twixtor-parity/retime-flow
. "$HOME/.cargo/env" 2>/dev/null
cd "$R" || exit 1
[ -d scratch ] || cp -a src scratch
mkdir -p bin results/mutants
for id in "$@"; do
  m=$id; [ "$id" = m00 ] && m=none
  python3 tools/mutate.py --pristine "$R/src" --scratch "$R/scratch" --mutants tools/mutants.json --id "$m" || { echo "$id: mutate failed"; continue; }
  t0=$(date +%s)
  if [ "${REUSE:-0}" = 1 ] && [ -x "bin/vfx-$id" ] && [[ " ${REBUILD:-} " != *" $id "* ]]; then
    echo "$id: reusing bin/vfx-$id"
  else
  if ! (cd scratch && CARGO_BUILD_JOBS=2 cargo build --release -p fx-cli -j2 2>&1 | tail -2); then echo "$id: build failed"; continue; fi
  cp scratch/target/release/vfx "bin/vfx-$id"
  echo "$id: built in $(( $(date +%s) - t0 )) s"
  fi
  python3 tools/run_grid.py run --vfx "bin/vfx-$id" --testset testsets/grid --out "results/mutants/$id.grid.tsv" --jobs 2
  python3 tools/run_grid.py run --vfx "bin/vfx-$id" --testset testsets/draft --out "results/mutants/$id.draft.tsv" --jobs 2
done
python3 tools/mutate.py --pristine "$R/src" --scratch "$R/scratch" --mutants tools/mutants.json --id none
