#!/usr/bin/env python3
"""Builds mutants of the rewrite in a scratch copy and keeps one vfx binary per mutant.

usage: build_mutants.py --mutants mutants/mutants.json --src SRC --work WORK --out MUT [--ids m01,m02] [--budget SECONDS]

SRC is the untouched copy of the repo, WORK a scratch copy made once with `cp -a SRC WORK`
(created here when missing). A mutant is one exact text replacement (`old` must occur at least
`n` times; the n-th occurrence is replaced) in a file of the engine scope. The build is
`cargo build --release -p fx-cli --offline` with two jobs; the binary lands in OUT/<id>/vfx and
the file is restored from SRC afterwards. A mutant whose binary exists is skipped. --budget stops
starting new builds once that many seconds have passed.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time


def replace_nth(text, old, new, n):
    pos = -1
    for _ in range(n):
        pos = text.find(old, pos + 1)
        if pos < 0:
            return None
    return text[:pos] + new + text[pos + len(old):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutants", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids")
    ap.add_argument("--budget", type=float, default=1e9)
    a = ap.parse_args()
    muts = json.load(open(a.mutants))
    if a.ids:
        want = a.ids.split(",")
        muts = [m for m in muts if m["id"] in want]
    if not os.path.isdir(a.work):
        subprocess.check_call(["cp", "-a", a.src, a.work])
    env = dict(os.environ, CARGO_BUILD_JOBS="2")
    # restore every mutated file from SRC with a fresh mtime so cargo rebuilds it from the pristine text
    # (a restore that keeps the old mtime makes cargo reuse the previous mutant's object code)
    for f in sorted({m["file"] for m in json.load(open(a.mutants))}):
        shutil.copyfile(os.path.join(a.src, f), os.path.join(a.work, f))
        os.utime(os.path.join(a.work, f), None)
    start = time.time()
    for m in muts:
        dest = os.path.join(a.out, m["id"], "vfx")
        if os.path.exists(dest):
            print(m["id"], "already built")
            continue
        if time.time() - start > a.budget:
            print(m["id"], "not started (budget)")
            continue
        path = os.path.join(a.work, m["file"])
        pristine = os.path.join(a.src, m["file"])
        text = open(pristine, encoding="utf-8").read()
        mutated = replace_nth(text, m["old"], m["new"], m.get("n", 1))
        if mutated is None or mutated == text:
            print(m["id"], "FAILED: old text not found")
            continue
        t0 = time.time()
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(mutated)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            log = os.path.join(a.out, m["id"], "build.log")
            with open(log, "w") as lf:
                rc = subprocess.call(["cargo", "build", "--release", "-p", "fx-cli", "--offline"], cwd=a.work, env=env,
                                     stdout=lf, stderr=subprocess.STDOUT)
            if rc != 0:
                print(m["id"], "BUILD FAILED, see", log)
                continue
            shutil.copy2(os.path.join(a.work, "target", "release", "vfx"), dest)
            print(m["id"], "built in %.0fs" % (time.time() - t0))
        finally:
            shutil.copyfile(pristine, path)
            os.utime(path, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
