#!/usr/bin/env python3
"""Writes results/summary.json from the result files (counts are computed, never typed).

usage: make_summary.py --root DIR_OF_THIS_MODULE [--status in_progress]

Phase A reads: grid/cases.json, grid/case_meta.json, grid/out_of_contract.json,
results/base_outputs.tsv, results/mutant_matrix.json. Emulator-side counts stay 0 until phase B
adds its result files.
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--status", default="in_progress")
    ap.add_argument("--date", required=True)
    a = ap.parse_args()
    r = a.root
    cases = json.load(open(os.path.join(r, "grid", "cases.json")))["cases"]
    meta = json.load(open(os.path.join(r, "grid", "case_meta.json")))
    ooc = json.load(open(os.path.join(r, "grid", "out_of_contract.json")))
    base_rows = [l for l in open(os.path.join(r, "results", "base_outputs.tsv")).read().splitlines()[1:]]
    base_ok = sum(1 for l in base_rows if l.split("\t")[1] == "0")
    mm = json.load(open(os.path.join(r, "results", "mutant_matrix.json")))
    run = [m for m in mm["mutants"] if m["run"]]
    changed = [m for m in run if m["changed"] > 0]
    survivors = [m["id"] for m in run if m["changed"] == 0]
    standin = sorted({m["standin"] for m in meta.values() if m["standin"]})
    blockers = ["emulator side not run yet"]
    blockers.append("mutants: %d of %d built mutants change at least one case against the base rewrite; killed stays 0 until compared with the emulator"
                    % (len(changed), len(run)))
    summary = {
        "module": "create-vectors",
        "plugin": "CreateMotionVectors_64.aex",
        "plugin_sha256": "9b16c3350e4e41f10cf7fc56ead2a0e0aeea00eb76708ef973f2d10e99d57e0a",
        "effect": "create-vectors",
        "status": a.status,
        "blockers": blockers,
        "cases": {"in_contract": len(cases), "compared": 0, "identical": 0, "differing": 0,
                  "out_of_contract": len(ooc), "logic_parity": sum(1 for m in meta.values() if m["standin"])},
        "values": {"bytes_compared": 0, "floats_compared": 0, "differing_values": 0, "max_ulp": None},
        "call_records": {"compared": 0, "matching": 0},
        "mutants": {"total": len(run), "killed": 0, "survivors": survivors},
        "stand_ins": ["%d cases with keyed params use the vfx keyframe syntax as the shared stand-in for AE keyframe interpolation (%s)"
                      % (sum(1 for m in meta.values() if m["standin"]), "; ".join(standin))],
        "fixes": [],
        "commands": [
            "python3 gen_grid.py --out grid",
            "python3 run_grid.py --vfx src/target/release/vfx --set grid --out results/base_outputs.tsv",
            "python3 build_mutants.py --mutants mutants/mutants.json --src src --work msrc --out mut",
            "python3 run_grid.py --vfx mut/<id>/vfx --set grid --out results/mutant_<id>.tsv",
            "python3 compare_mutants.py --base results/base_outputs.tsv --mutants mutants/mutants.json --dir results --out results/mutant_matrix.json --set grid",
        ],
        "measured_at": a.date,
    }
    out = os.path.join(r, "results", "summary.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=1)
        fh.write("\n")
    print("wrote", out, "base rendered %d/%d exit 0" % (base_ok, len(base_rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
