#!/usr/bin/env python3
"""Render every case of a test set with one vfx binary and record SHA-256 and size per case.

python3 render_grid.py --vfx BIN --testset DIR --tsv OUT.tsv [--keep OUTDIR] [--jobs 2] [--check-only]

TSV columns: id, exit, size, sha256, stderr_sha256. The vfx command line is the one fx-parity builds
for a temporal case (apply EFFECT --width --height --depth --param... --fps --frames --in), with
--out to a temp file. --check-only runs `--check` instead of a render (exit code and message).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor


def args_for(case, testset):
    a = ["apply", case["effect"], "--width", str(case["width"]), "--height", str(case["height"]),
         "--depth", str(case["depth"])]
    for p in case["params"]:
        a += ["--param", p]
    if case.get("fps"):
        a += ["--fps", case["fps"]]
    a += ["--frames", str(case["frames"])]
    a += ["--in", os.path.join(testset, case["input"])]
    return a


def run_case(vfx, testset, case, keep, check_only, tmp):
    a = args_for(case, testset)
    env = dict(os.environ, RAYON_NUM_THREADS="1")
    if check_only:
        a.append("--check")
        p = subprocess.run([vfx] + a, capture_output=True, env=env)
        msg = (p.stderr.decode(errors="replace").strip().splitlines() or [""])[0]
        return [case["id"], str(p.returncode), "0", "", msg]
    out = os.path.join(keep if keep else tmp, case["reference"])
    p = subprocess.run([vfx] + a + ["--out", out], capture_output=True, env=env)
    size, sha = 0, ""
    if os.path.exists(out):
        h = hashlib.sha256()
        with open(out, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        sha, size = h.hexdigest(), os.path.getsize(out)
        if not keep:
            os.remove(out)
    esha = hashlib.sha256(p.stderr).hexdigest()[:16]
    return [case["id"], str(p.returncode), str(size), sha, esha]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vfx", required=True)
    ap.add_argument("--testset", required=True)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--keep")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--only", help="comma separated group names")
    o = ap.parse_args()
    ts = json.load(open(os.path.join(o.testset, "cases.json")))
    cases = ts["cases"]
    if o.only:
        meta = json.load(open(os.path.join(o.testset, "grid.json")))
        keep = set(o.only.split(","))
        cases = [c for c in cases if meta[c["id"]]["group"] in keep]
    if o.keep:
        os.makedirs(o.keep, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(o.jobs) as ex:
        rows = list(ex.map(lambda c: run_case(o.vfx, o.testset, c, o.keep, o.check_only, tmp), cases))
    with open(o.tsv, "w") as fh:
        fh.write("id\texit\tsize\tsha256\tstderr\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    bad = sum(1 for r in rows if r[1] != "0")
    print("%d cases, %d with a nonzero exit -> %s" % (len(rows), bad, o.tsv))


if __name__ == "__main__":
    main()
