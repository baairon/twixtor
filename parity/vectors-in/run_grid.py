#!/usr/bin/env python3
"""Renders a test set with one vfx binary and records exit code, size and SHA-256 per case.

    run_grid.py run   --vfx BIN --testset DIR --out FILE.tsv [--keep REFDIR] [--jobs 2]
    run_grid.py check --vfx BIN --ooc FILE.json --out FILE.tsv

`run` builds the command line fx-parity builds (`vfx apply vectors-in ... --layer ID=FILE --in FILE`).
With --keep the output of each case that rendered is also written as REFDIR/<reference>, the file
name `fx-parity run --refs` expects, so any binary can be graded against it.
`check` runs `vfx apply --check` on the out-of-contract candidates (range and id checks only).

Output columns of `run`: id, exit, size, sha256, ms, note (first stderr line that is not a
`vfx: note:` line). Rows are sorted by id so two runs diff cleanly. RAYON_NUM_THREADS is 1.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor


def case_args(c, testset):
    a = ["apply", c["effect"], "--width", str(c["width"]), "--height", str(c["height"]), "--depth", str(c["depth"])]
    for p in c["params"]:
        a += ["--param", p]
    a += ["--fps", c["fps"], "--frames", str(c["frames"])]
    for lid, f in sorted(c.get("layers", {}).items()):
        a += ["--layer", "%s=%s" % (lid, os.path.join(testset, f))]
    a += ["--in", os.path.join(testset, c["input"])]
    return a


def first_note(stderr):
    for line in stderr.decode("utf8", "replace").splitlines():
        if line.startswith("vfx: note:"):
            continue
        return line.replace("\t", " ").strip()[:200]
    return ""


def run_one(vfx, testset, keep, c, env):
    t0 = time.time()
    p = subprocess.run([vfx] + case_args(c, testset), stdin=subprocess.DEVNULL, capture_output=True, env=env)
    ms = int((time.time() - t0) * 1000)
    if keep and p.returncode == 0:
        with open(os.path.join(keep, c["reference"]), "wb") as fh:
            fh.write(p.stdout)
    return [c["id"], str(p.returncode), str(len(p.stdout)), hashlib.sha256(p.stdout).hexdigest(), str(ms), first_note(p.stderr)]


def cmd_run(a):
    doc = json.load(open(os.path.join(a.testset, "cases.json")))
    if a.keep:
        os.makedirs(a.keep, exist_ok=True)
    env = dict(os.environ, RAYON_NUM_THREADS="1")
    with ThreadPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(lambda c: run_one(a.vfx, a.testset, a.keep, c, env), doc["cases"]))
    rows.sort(key=lambda r: r[0])
    with open(a.out, "w") as fh:
        fh.write("id\texit\tsize\tsha256\tms\tnote\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    print("%s: %d cases, %d with a nonzero exit" % (a.out, len(rows), sum(1 for r in rows if r[1] != "0")))


def cmd_check(a):
    rows = []
    for o in json.load(open(a.ooc)):
        args = [a.vfx, "apply", "vectors-in", "--width", str(o["width"]), "--height", str(o["height"]),
                "--depth", str(o["depth"]), "--fps", "30", "--check"]
        for p in o["params"]:
            args += ["--param", p]
        p = subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True)
        rows.append([o["id"], o["label"], o["why"], str(p.returncode), first_note(p.stderr)])
    with open(a.out, "w") as fh:
        fh.write("id\tparams\twhy\trewrite_check_exit\tmessage\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    print("%s: %d candidates, %d refused by the rewrite" % (a.out, len(rows), sum(1 for r in rows if r[3] != "0")))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--vfx", required=True)
    r.add_argument("--testset", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--keep")
    r.add_argument("--jobs", type=int, default=2)
    c = sub.add_parser("check")
    c.add_argument("--vfx", required=True)
    c.add_argument("--ooc", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    (cmd_run if a.cmd == "run" else cmd_check)(a)


if __name__ == "__main__":
    main()
