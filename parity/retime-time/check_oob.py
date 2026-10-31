#!/usr/bin/env python3
"""Run the out-of-contract candidates of out_of_contract.json through `vfx --check` and a real render.

python3 check_oob.py --vfx BIN --testset DIR --tsv OUT.tsv

The candidates are listed, never graded: the rewrite is only asked whether it refuses them, and
phase B asks the emulator the same. TSV columns: label, params, check_exit, check_message,
render_exit, render_message.
"""
import argparse
import json
import os
import subprocess
import tempfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vfx", required=True)
    ap.add_argument("--testset", required=True)
    ap.add_argument("--tsv", required=True)
    o = ap.parse_args()
    ts = json.load(open(os.path.join(o.testset, "cases.json")))
    inp = next(i for i in ts["inputs"] if i["depth"] == 8 and i["frames"] == 12 and i["width"] > 1)
    items = json.load(open(os.path.join(o.testset, "out_of_contract.json")))
    base = ["apply", "retime", "--width", str(inp["width"]), "--height", str(inp["height"]), "--depth", "8",
            "--fps", "24", "--frames", "12", "--in", os.path.join(o.testset, inp["file"])]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for it in items:
            a = list(base)
            for p in it["params"]:
                a += ["--param", p]
            chk = subprocess.run([o.vfx] + a + ["--check"], capture_output=True)
            ren = subprocess.run([o.vfx] + a + ["--out", os.path.join(tmp, "o.raw")], capture_output=True)
            m1 = (chk.stderr.decode(errors="replace").strip().splitlines() or [""])[0]
            m2 = (ren.stderr.decode(errors="replace").strip().splitlines() or [""])[0]
            rows.append([it["label"], " ".join(it["params"]), str(chk.returncode), m1, str(ren.returncode), m2])
    with open(o.tsv, "w") as fh:
        fh.write("label\tparams\tcheck_exit\tcheck_message\trender_exit\trender_message\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    print("%d candidates, %d refused by --check" % (len(rows), sum(1 for r in rows if r[2] != "0")))


if __name__ == "__main__":
    main()
