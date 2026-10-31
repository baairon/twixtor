#!/usr/bin/env python3
"""List the cases where the unmutated opt-level-1 build (m00_control) differs from the base release
build, and say for each whether every differing 32 bit word is a NaN on both sides.

python3 control_check.py --testset DIR --base base.tsv --control results/mut/m00_control.tsv \
    --base-out base-out --vfx mutbin/m00_control.vfx --out results/control_vs_base.tsv
"""
import argparse
import json
import os
import struct
import subprocess
import tempfile


def rows(path):
    d = {}
    with open(path) as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            d[p[0]] = p[1:4]
    return d


def is_nan(w):
    return (w & 0x7F800000) == 0x7F800000 and (w & 0x007FFFFF) != 0


def main():
    ap = argparse.ArgumentParser()
    for a in ("--testset", "--base", "--control", "--base-out", "--vfx", "--out"):
        ap.add_argument(a, required=True)
    o = ap.parse_args()
    b, c = rows(o.base), rows(o.control)
    cases = {x["id"]: x for x in json.load(open(os.path.join(o.testset, "cases.json")))["cases"]}
    out = ["id\tdepth\tdiffering_words\tall_differing_words_nan_on_both_sides"]
    for cid in b:
        if b[cid] == c.get(cid):
            continue
        case = cases[cid]
        with tempfile.TemporaryDirectory() as tmp:
            args = [o.vfx, "apply", case["effect"], "--width", str(case["width"]), "--height", str(case["height"]),
                    "--depth", str(case["depth"])]
            for p in case["params"]:
                args += ["--param", p]
            args += ["--fps", case["fps"], "--frames", str(case["frames"]), "--in",
                     os.path.join(o.testset, case["input"]), "--out", os.path.join(tmp, "o")]
            subprocess.run(args, check=True, capture_output=True)
            x = open(os.path.join(tmp, "o"), "rb").read()
        y = open(os.path.join(o.base_out, case["reference"]), "rb").read()
        if len(x) != len(y) or case["depth"] != 32:
            out.append("%s\t%d\t?\tfalse" % (cid, case["depth"]))
            continue
        n = len(x) // 4
        xs, ys = struct.unpack("<%dI" % n, x), struct.unpack("<%dI" % n, y)
        diff = [i for i in range(n) if xs[i] != ys[i]]
        ok = all(is_nan(xs[i]) and is_nan(ys[i]) for i in diff)
        out.append("%s\t%d\t%d\t%s" % (cid, case["depth"], len(diff), str(ok).lower()))
    open(o.out, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
