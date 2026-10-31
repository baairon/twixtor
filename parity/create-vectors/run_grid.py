#!/usr/bin/env python3
"""Renders every case of a test set with one vfx binary and records exit code, size and SHA-256.

usage: run_grid.py --vfx BIN --set DIR --out FILE.tsv [--cases FILE.json] [--jobs 2] [--threads 1]
                   [--only SUBSTR] [--check]

--cases   a cases file other than DIR/cases.json (for example out_of_contract.json, a bare list)
--threads sets RAYON_NUM_THREADS for each vfx run (default 1; at most 2 cores are used in total)
--check   run `vfx apply ... --check` instead of rendering (validates ids, params, layers)

Output: a TSV with columns id, exit, size, sha256, note. The note is the first stderr line that
is not a `vfx: note:` line, tabs removed. Rows are sorted by id so two runs diff cleanly.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor


def vfx_args(binary, case, testset, check):
    a = [binary, "apply", case["effect"], "--width", str(case["width"]), "--height", str(case["height"]),
         "--depth", str(case["depth"])]
    for p in case["params"]:
        a += ["--param", p]
    if case.get("fps"):
        a += ["--fps", case["fps"]]
    a += ["--frames", str(case["frames"])]
    for lid, f in sorted(case.get("layers", {}).items()):
        a += ["--layer", "%s=%s" % (lid, os.path.join(testset, f))]
    a += ["--in", os.path.join(testset, case["input"])]
    if check:
        a.append("--check")
    return a


def run_one(binary, case, testset, tmpdir, check, env):
    args = vfx_args(binary, case, testset, check)
    out = None
    if not check:
        out = os.path.join(tmpdir, "o-%d-%s" % (os.getpid(), hashlib.md5(case["id"].encode()).hexdigest()))
        args += ["--out", out]
    p = subprocess.run(args, capture_output=True, env=env)
    size, digest = 0, "-"
    if out and os.path.exists(out):
        h = hashlib.sha256()
        with open(out, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                size += len(chunk)
                h.update(chunk)
        digest = h.hexdigest()
        os.remove(out)
    note = ""
    for line in p.stderr.decode("utf-8", "replace").splitlines():
        if line.startswith("vfx: note:"):
            continue
        note = line.replace("\t", " ").strip()[:200]
        break
    return [case["id"], str(p.returncode), str(size), digest, note]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vfx", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--threads", default="1")
    ap.add_argument("--only")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    path = a.cases or os.path.join(a.set, "cases.json")
    with open(path) as fh:
        data = json.load(fh)
    cases = data["cases"] if isinstance(data, dict) else data
    if a.only:
        cases = [c for c in cases if a.only in c["id"]]
    env = dict(os.environ, RAYON_NUM_THREADS=a.threads)
    with tempfile.TemporaryDirectory(prefix="cvgrid-") as tmp, ThreadPoolExecutor(max_workers=a.jobs) as ex:
        rows = list(ex.map(lambda c: run_one(a.vfx, c, a.set, tmp, a.check, env), cases))
    rows.sort(key=lambda r: r[0])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("id\texit\tsize\tsha256\tnote\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    ok = sum(1 for r in rows if r[1] == "0")
    print("%s: %d cases, %d exit 0, %d other" % (a.out, len(rows), ok, len(rows) - ok))
    return 0


if __name__ == "__main__":
    sys.exit(main())
