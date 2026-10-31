#!/usr/bin/env python3
"""Turn result files into numbers. Nothing here is typed by hand: every count comes from a file.

python3 analyze.py mutants --testset DIR --base base.tsv --mutdir DIR --out results/mutants_vs_base.json
python3 analyze.py summary --testset DIR --results results --mutants-json results/mutants_vs_base.json \
    --plugin-sha-file results/plugin_sha256.txt --out results/summary.json [--commands-file F]

`mutants`: for each mutant TSV (named <id>.tsv) in MUTDIR, the cases whose (exit, size, sha256) differ from
the base rewrite, per group. The control (`m00_control.tsv`, the unmutated source built the same way as the
mutants) must change nothing. `summary`: phase A summary.json; emulator side fields stay zero.
"""
import argparse
import datetime
import glob
import json
import os


def read_tsv(path):
    rows = {}
    with open(path) as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            rows[p[0]] = (p[1], p[2], p[3])
    return rows


def mutants(o):
    """A mutant is judged against m00_control (same build profile); the base release build differs from
    the control only where the compiler picks another NaN payload, listed as control_vs_base."""
    meta = json.load(open(os.path.join(o.testset, "grid.json")))
    base = read_tsv(o.base)
    import mutants as mutmod
    info = {m["id"]: m for m in mutmod.M}
    runs = {}
    for path in sorted(glob.glob(os.path.join(o.mutdir, "*.tsv"))):
        runs[os.path.basename(path)[:-4]] = read_tsv(path)
    ctl = runs.get("m00_control")
    out = {"base_cases": len(base), "base_nonzero_exit": sum(1 for v in base.values() if v[0] != "0"),
           "control_vs_base": None, "mutants": []}
    if ctl is not None:
        out["control_vs_base"] = [c for c in base if ctl.get(c) != base[c]]
    ref = ctl if ctl is not None else base
    for mid, got in runs.items():
        if mid == "m00_control":
            continue
        changed = [c for c in ref if got.get(c) != ref[c]]
        vs_base = [c for c in base if got.get(c) != base[c]]
        groups = {}
        for c in changed:
            g = meta[c]["group"]
            groups[g] = groups.get(g, 0) + 1
        out["mutants"].append({
            "id": mid, "what": info[mid]["what"], "file": info[mid]["file"], "expect": info[mid]["expect"],
            "changed": len(changed), "changed_vs_base": len(vs_base), "groups": groups, "changed_ids": changed,
            "exit_changes": sum(1 for c in changed if got.get(c, ("?",))[0] != ref[c][0])})
    with open(o.out, "w") as fh:
        json.dump(out, fh, indent=1)
    tsv = os.path.splitext(o.out)[0] + ".tsv"
    with open(tsv, "w") as fh:
        fh.write("mutant\tfile\texpect\tchanged_cases\tchanged_vs_base\texit_changes\tgroups\twhat\n")
        for m in out["mutants"]:
            fh.write("%s\t%s\t%s\t%d\t%d\t%d\t%s\t%s\n" % (
                m["id"], m["file"], m["expect"], m["changed"], m["changed_vs_base"], m["exit_changes"],
                ";".join("%s:%d" % kv for kv in sorted(m["groups"].items())), m["what"]))
    cv = out["control_vs_base"]
    print("control differs from base in %s cases" % ("?" if cv is None else len(cv)))
    for m in out["mutants"]:
        print("%-30s changed %4d (vs base %4d)  %s" % (m["id"], m["changed"], m["changed_vs_base"], m["expect"]))


def summary(o):
    ts = json.load(open(os.path.join(o.testset, "cases.json")))
    meta = json.load(open(os.path.join(o.testset, "grid.json")))
    oob = json.load(open(os.path.join(o.testset, "out_of_contract.json")))
    ml = json.load(open(o.mutants_json)) if os.path.exists(o.mutants_json) else {"mutants": []}
    sha = ""
    if os.path.exists(o.plugin_sha_file):
        sha = open(o.plugin_sha_file).read().split()[0]
    cmds = []
    if o.commands_file and os.path.exists(o.commands_file):
        cmds = [l.rstrip("\n") for l in open(o.commands_file) if l.strip()]
    s = {
        "module": "retime-time", "plugin": "Twixtor_64.aex", "plugin_sha256": sha, "effect": "retime",
        "status": "in_progress",
        "blockers": ["emulator side not run yet"],
        "cases": {"in_contract": len(ts["cases"]), "compared": 0, "identical": 0, "differing": 0,
                  "out_of_contract": len(oob),
                  "logic_parity": sum(1 for m in meta.values() if m["class"] == "logic_parity")},
        "values": {"bytes_compared": 0, "floats_compared": 0, "differing_values": 0, "max_ulp": None},
        "call_records": {"compared": 0, "matching": 0},
        "mutants": {"total": len(ml["mutants"]), "killed": 0, "survivors": []},
        "stand_ins": ["keyed curves (speed_percent, frame_num, src_mark_segments with @ keys): host side "
                      "keyframe interpolation, one labelled stand-in on both sides, cases counted as logic_parity"],
        "fixes": [], "commands": cmds,
        "measured_at": datetime.date.today().isoformat(),
    }
    with open(o.out, "w") as fh:
        json.dump(s, fh, indent=1)
        fh.write("\n")
    print(json.dumps(s["cases"]), json.dumps(s["mutants"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("mutants")
    a.add_argument("--testset", required=True)
    a.add_argument("--base", required=True)
    a.add_argument("--mutdir", required=True)
    a.add_argument("--out", required=True)
    b = sub.add_parser("summary")
    b.add_argument("--testset", required=True)
    b.add_argument("--results", default="results")
    b.add_argument("--mutants-json", required=True)
    b.add_argument("--plugin-sha-file", required=True)
    b.add_argument("--commands-file")
    b.add_argument("--out", required=True)
    o = ap.parse_args()
    {"mutants": mutants, "summary": summary}[o.cmd](o)
