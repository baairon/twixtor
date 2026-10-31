#!/usr/bin/env python3
"""Compare two render_grid.py result files: print the ids whose exit code, size or sha256 differ.

  python3 compare_runs.py BASE.tsv OTHER.tsv
"""
import sys


def load(p):
    d = {}
    for line in open(p).read().splitlines()[1:]:
        f = line.split("\t")
        d[f[0]] = tuple(f[1:4])
    return d


def changed(base, other):
    return sorted(k for k in base if other.get(k) != base[k])


if __name__ == "__main__":
    b, o = load(sys.argv[1]), load(sys.argv[2])
    for k in changed(b, o):
        print(k)
