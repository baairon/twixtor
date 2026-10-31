#!/usr/bin/env python3
"""Render a test set with one vfx binary and record sha256 and size per case.

    run_grid.py run   --vfx BIN --testset DIR --out FILE.tsv [--keep REFDIR] [--jobs 2]
    run_grid.py check --vfx BIN --ooc FILE.json --out FILE.tsv

`run` builds the same command line as fx-parity (`vfx apply retime ... --in FILE`) and adds
`--quality draft` for a case that carries "quality": "draft". With --keep, the output of each
case is also written as REFDIR/<reference> so `fx-parity run --refs REFDIR` can grade a binary
against it.
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
    if c.get("quality"):
        a += ["--quality", c["quality"]]
    a += ["--in", os.path.join(testset, c["input"])]
    return a


def run_one(vfx, testset, keep, c):
    t0 = time.time()
    p = subprocess.run([vfx] + case_args(c, testset), stdin=subprocess.DEVNULL, capture_output=True)
    ms = int((time.time() - t0) * 1000)
    if keep and p.returncode == 0:
        with open(os.path.join(keep, c["reference"]), "wb") as fh:
            fh.write(p.stdout)
    err = p.stderr.decode("utf8", "replace").strip().splitlines()
    first = err[0][:160] if err else ""
    return [c["id"], str(p.returncode), str(len(p.stdout)), hashlib.sha256(p.stdout).hexdigest(), str(ms), first.replace("\t", " ")]


def cmd_run(a):
    doc = json.load(open(os.path.join(a.testset, "cases.json")))
    if a.keep:
        os.makedirs(a.keep, exist_ok=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(lambda c: run_one(a.vfx, a.testset, a.keep, c), doc["cases"]))
    with open(a.out, "w") as fh:
        fh.write("id\texit\tbytes\tsha256\tms\tstderr\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    bad = sum(1 for r in rows if r[1] != "0")
    print(f"{a.out}: {len(rows)} cases, {bad} with a nonzero exit")


def cmd_check(a):
    ooc = json.load(open(a.ooc))
    rows = []
    for o in ooc:
        args = ["apply", "retime", "--width", str(o["width"]), "--height", str(o["height"]), "--depth", str(o["depth"]), "--fps", "24", "--check"]
        for p in o["params"]:
            args += ["--param", p]
        p = subprocess.run([a.vfx] + args, stdin=subprocess.DEVNULL, capture_output=True)
        err = p.stderr.decode("utf8", "replace").strip().splitlines()
        rows.append([o["id"], o["label"], o["why"], str(p.returncode), (err[0] if err else "").replace("\t", " ")[:160]])
    with open(a.out, "w") as fh:
        fh.write("id\tlabel\twhy\trewrite_check_exit\tmessage\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    print(f"{a.out}: {len(rows)} candidates, {sum(1 for r in rows if r[3] != '0')} refused by the rewrite")


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
