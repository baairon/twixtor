#!/usr/bin/env python3
"""Does any sensitivity value separate two vfx binaries?  (used to judge mutant m17)

    explore_sens.py --testsets DIR --a BIN --b BIN
Renders disc 40x30 at 32 bpc for sensitivity 0.5 + 0.9973 k, k = 1..100 (off the integer slider values), with both binaries and
prints the values whose outputs differ.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen  # noqa: E402
import run_grid  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testsets", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    a = ap.parse_args()
    b = gen.Builder()
    for k in range(1, 101):
        b.add("sweep", f"k{k}", "disc", 40, 30, 32, gen.P(40, sensitivity=f"{0.5 + 0.9973 * k:.5f}"))
    out = os.path.join(a.testsets, "explore")
    gen.write_set(b, out)
    diff = []
    for c in b.cases:
        h = []
        for vfx in (a.a, a.b):
            p = subprocess.run([vfx] + run_grid.case_args(c, out), capture_output=True, stdin=subprocess.DEVNULL)
            h.append(hashlib.sha256(p.stdout).hexdigest())
        if h[0] != h[1]:
            diff.append(c["params"][1])
    print(f"{len(diff)} of {len(b.cases)} sensitivity values differ: {diff}")


if __name__ == "__main__":
    main()
