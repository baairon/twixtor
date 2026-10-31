#!/usr/bin/env python3
"""Compares per-mutant grid results with the base rewrite's.

usage: compare_mutants.py --base results/base_outputs.tsv --mutants mutants/mutants.json
                          --dir results --out results/mutant_matrix.json

Reads results/mutant_<id>.tsv for every mutant in the list (missing files count as not run).
A case is changed when its exit code, size or SHA-256 differs from the base row.
Writes the matrix (per mutant: changed case count, ids, by depth and group) and prints a table.
"""
import argparse
import json
import os
import re
import sys


def load(path):
    rows = {}
    with open(path) as fh:
        next(fh)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            rows[f[0]] = tuple(f[1:4])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--mutants", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set")
    a = ap.parse_args()
    base = load(a.base)
    muts = json.load(open(a.mutants))
    meta = {}
    if a.set and os.path.exists(os.path.join(a.set, "case_meta.json")):
        meta = json.load(open(os.path.join(a.set, "case_meta.json")))
    out = {"base_cases": len(base), "mutants": []}
    print("%-5s %-16s %7s %s" % ("id", "kind", "changed", "first changed case"))
    for m in muts:
        p = os.path.join(a.dir, "mutant_%s.tsv" % m["id"])
        entry = {"id": m["id"], "kind": m["kind"], "file": m["file"], "desc": m["desc"], "run": os.path.exists(p)}
        if entry["run"]:
            rows = load(p)
            changed = sorted(c for c in base if rows.get(c) != base[c])
            entry["changed"] = len(changed)
            entry["cases"] = changed
            by_depth, by_group = {}, {}
            for c in changed:
                d = "d" + re.search(r"\.d(8|16|32)\.", c).group(1)
                by_depth[d] = by_depth.get(d, 0) + 1
                g = meta.get(c, {}).get("group", "?")
                by_group[g] = by_group.get(g, 0) + 1
            entry["by_depth"] = by_depth
            entry["by_group"] = by_group
            print("%-5s %-16s %7d %s" % (m["id"], m["kind"], len(changed), changed[0] if changed else "-"))
        else:
            print("%-5s %-16s %7s" % (m["id"], m["kind"], "not run"))
        out["mutants"].append(entry)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
