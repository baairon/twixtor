#!/usr/bin/env python3
"""Mutant table and phase A summary, every count computed from the result files.

    compare.py --results results --mutants tools/mutants.json --testset testsets/grid --date YYYY-MM-DD
               [--out-summary results/summary.json]

Reads results/base.grid.tsv (the base rewrite on the grid), results/mutants/<id>.grid.tsv,
results/out_of_contract.tsv, results/plugin_sha.txt and the testset's cases.json / case_meta.json.
A mutant changes a case when the exit code, size or SHA-256 of its output differs from the base row.
Phase A only: nothing here compares with the emulator.
"""
import argparse
import json
import os


def load(path):
    rows = {}
    with open(path) as fh:
        next(fh)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            rows[f[0]] = f
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--mutants", required=True)
    ap.add_argument("--testset", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--out-summary")
    a = ap.parse_args()
    R = a.results
    base = load(os.path.join(R, "base.grid.tsv"))
    meta = json.load(open(os.path.join(a.testset, "case_meta.json")))
    cases = json.load(open(os.path.join(a.testset, "cases.json")))["cases"]
    ooc = load(os.path.join(R, "out_of_contract.tsv"))
    key = lambda r: tuple(r[1:4])

    # determinism: the second base run, when there is one
    det = None
    p2 = os.path.join(R, "base2.grid.tsv")
    if os.path.exists(p2):
        b2 = load(p2)
        det = {"cases": len(b2), "differing": sum(1 for c in base if key(b2.get(c, ["", "", "", ""])) != key(base[c]))}
    ctrl = None
    pc = os.path.join(R, "mutants", "m00.grid.tsv")
    if os.path.exists(pc):
        c0 = load(pc)
        ctrl = {"cases": len(c0), "differing_from_base": sum(1 for c in base if key(c0.get(c, ["", "", "", ""])) != key(base[c]))}

    refused = sorted(c for c, r in base.items() if r[1] != "0")
    rendered = sorted(c for c, r in base.items() if r[1] == "0")

    muts = json.load(open(a.mutants))
    table, survivors = [], []
    for m in muts:
        p = os.path.join(R, "mutants", m["id"] + ".grid.tsv")
        e = {"id": m["id"], "kind": m["kind"], "file": m["file"], "desc": m["desc"], "run": os.path.exists(p)}
        if e["run"]:
            rows = load(p)
            ch = sorted(c for c in base if key(rows.get(c, ["", "", "", ""])) != key(base[c]))
            by_group = {}
            for c in ch:
                g = meta.get(c, {}).get("group", "?")
                by_group[g] = by_group.get(g, 0) + 1
            e.update(changed=len(ch), example=ch[:3], by_group=by_group, by_depth={
                str(d): sum(1 for c in ch if ".d%d." % d in c) for d in (8, 16, 32)})
            if not ch:
                survivors.append(m["id"])
        else:
            e.update(changed=None)
            survivors.append(m["id"])
        table.append(e)
    killed = sum(1 for e in table if e["changed"])
    with open(os.path.join(R, "mutant_table.tsv"), "w") as fh:
        fh.write("id\tkind\tfile\tchanged_cases\tby_group\tdescription\n")
        for e in table:
            bg = ",".join("%s:%d" % kv for kv in sorted(e.get("by_group", {}).items())) if e["changed"] else ""
            fh.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (e["id"], e["kind"], e["file"], "not run" if e["changed"] is None else e["changed"], bg, e["desc"]))
    with open(os.path.join(R, "mutants.json"), "w") as fh:
        json.dump({"base_cases": len(base), "control": ctrl, "mutants": table}, fh, indent=1)

    stand = sorted(c for c, m in meta.items() if m.get("stand_in"))
    flags = {}
    for c, m in meta.items():
        for f in m.get("flags", []):
            flags.setdefault(f, []).append(c)
    sha = open(os.path.join(R, "plugin_sha.txt")).read().split()[0]
    refused_ooc = sum(1 for r in ooc.values() if r[3] != "0")
    detail = {
        "grid_cases": len(cases), "base_rendered": len(rendered), "base_refused": [(c, base[c][5]) for c in refused],
        "determinism_rerun": det, "control_build": ctrl,
        "out_of_contract_candidates": len(ooc), "candidates_refused_by_rewrite_check": refused_ooc,
        "stand_in_cases": len(stand), "flagged": {k: len(v) for k, v in sorted(flags.items())},
        "groups": {g: sum(1 for m in meta.values() if m["group"] == g) for g in sorted({m["group"] for m in meta.values()})},
        "distinct_outputs": len({r[3] for r in base.values() if r[1] == "0"}),
    }
    with open(os.path.join(R, "phase_a.json"), "w") as fh:
        json.dump(detail, fh, indent=1)

    summary = {
        "module": "vectors-in", "plugin": "TwixtorProVectorsIn_64.aex", "plugin_sha256": sha, "effect": "vectors-in",
        "status": "in_progress",
        "blockers": ["emulator side not run yet"],
        "cases": {"in_contract": len(rendered), "compared": 0, "identical": 0, "differing": 0,
                  "out_of_contract": len(refused) + len(ooc), "logic_parity": 0},
        "values": {"bytes_compared": 0, "floats_compared": 0, "differing_values": 0, "max_ulp": None},
        "call_records": {"compared": 0, "matching": 0},
        "mutants": {"total": len(muts), "killed": killed, "survivors": survivors},
        "stand_ins": ["AE keyframe interpolation: %d grid cases key a parameter (tag keyframe_interpolation); both sides must read the same labelled interpolation of the keys, and those cases count as logic_parity" % len(stand)],
        "fixes": [],
        "commands": [
            "python3 tools/gen_grid.py --out testsets/grid",
            "python3 tools/run_grid.py run --vfx bin/vfx-base --testset testsets/grid --out results/base.grid.tsv --keep refs/grid --jobs 2",
            "python3 tools/run_grid.py check --vfx bin/vfx-base --ooc testsets/grid/out_of_contract.json --out results/out_of_contract.tsv",
            "bash tools/build_mutants.sh m00 m01 ... m24",
            "python3 tools/compare.py --results results --mutants tools/mutants.json --testset testsets/grid --date " + a.date,
        ],
        "measured_at": a.date,
    }
    out = a.out_summary or os.path.join(R, "summary.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=1)
    print("grid %d cases: %d rendered, %d refused by the base rewrite; %d candidates out of range (%d refused by --check)" % (
        len(cases), len(rendered), len(refused), len(ooc), refused_ooc))
    print("mutants: %d total, %d changed at least one case, survivors %s" % (len(muts), killed, survivors))
    for e in table:
        print("  %-4s %-20s %s" % (e["id"], e["kind"], "not run" if e["changed"] is None else "%4d cases changed" % e["changed"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
