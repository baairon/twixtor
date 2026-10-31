#!/usr/bin/env python3
"""Compare mutant renders with the base rewrite and write the tables and summary.json.

    compare.py --results results --mutants mutants/mutants.json --cases-dir cases

Reads results/base.{grid,draft}.tsv, results/base2.{grid,draft}.tsv (determinism rerun) and
results/mutants/<id>.{grid,draft}.tsv; writes results/mutant_table.tsv, results/mutants.json
and results/summary.json. Every count is computed from those files.
"""
import argparse
import csv
import glob
import json
import os


def load(path):
    with open(path) as fh:
        return {r["id"]: r for r in csv.DictReader(fh, delimiter="\t")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--mutants", required=True)
    ap.add_argument("--cases-dir", required=True)
    ap.add_argument("--date", required=True)
    a = ap.parse_args()
    R = a.results
    cases = {}
    for s in ("grid", "draft"):
        for c in json.load(open(os.path.join(a.cases_dir, s, "cases.json")))["cases"]:
            c["set"] = s
            cases[c["id"]] = c
    base = {}
    for s in ("grid", "draft"):
        base.update(load(f"{R}/base.{s}.tsv"))
    base2 = {}
    for s in ("grid", "draft"):
        base2.update(load(f"{R}/base2.{s}.tsv"))
    assert set(base) == set(cases), "base results do not cover the case list"
    nondet = [i for i in base if base[i]["sha256"] != base2[i]["sha256"]]
    refused = sorted(i for i in base if base[i]["exit"] != "0")
    ms = json.load(open(a.mutants))
    ms = [dict(id="m00", kind="control", desc="pristine source rebuilt in the scratch copy (must change nothing)")] + ms
    out = []
    for m in ms:
        got = {}
        for s in ("grid", "draft"):
            p = f"{R}/mutants/{m['id']}.{s}.tsv"
            if os.path.exists(p):
                got.update(load(p))
        if set(got) != set(base):
            out.append(dict(m, built=False, changed=[]))
            continue
        changed = sorted(i for i in base if (got[i]["sha256"], got[i]["exit"]) != (base[i]["sha256"], base[i]["exit"]))
        groups = {}
        for i in changed:
            groups[cases[i]["group"]] = groups.get(cases[i]["group"], 0) + 1
        out.append(dict(m, built=True, changed=changed, groups=groups))
    with open(f"{R}/mutant_table.tsv", "w") as fh:
        fh.write("id\tkind\tcases_changed\tgroups\tdescription\n")
        for m in out:
            if not m["built"]:
                fh.write(f"{m['id']}\t{m['kind']}\tnot run\t\t{m['desc']}\n")
                continue
            g = ",".join(f"{k}:{v}" for k, v in sorted(m["groups"].items()))
            fh.write(f"{m['id']}\t{m['kind']}\t{len(m['changed'])}\t{g}\t{m['desc']}\n")
    used = {}
    for c in cases.values():
        for p in c["params"]:
            k, v = p.split("=", 1)
            used.setdefault(k, set()).add(v)
    pu = {"rewrite_param_ids": {k: sorted(v) for k, v in sorted(used.items())},
          "host_side_not_params": ["fps (24, 30000/1001, 60)", "frames per clip (4 or 6)", "quality (draft cases in testsets/draft)", "frame size and bit depth"],
          "left_at_default": ["time_remap_mode", "frame_num", "src_time_offset", "in_fps_is_out_fps", "input_frame_rate", "input_fields", "src_mark_segments", "display", "plenty_memory", "alt_source_a", "alt_source_b"]}
    with open(f"{R}/params_used.json", "w") as fh:
        json.dump(pu, fh, indent=1)
        fh.write("\n")
    with open(f"{R}/mutants.json", "w") as fh:
        json.dump([{k: v for k, v in m.items() if k not in ("old", "new", "file")} for m in out], fh, indent=1)
        fh.write("\n")
    real = [m for m in out if m["id"] != "m00"]
    built = [m for m in real if m["built"]]
    unchanged = [m["id"] for m in built if not m["changed"]]
    ctrl = next((m for m in out if m["id"] == "m00"), None)
    n_stand = sum(1 for c in cases.values() if c.get("stand_in"))
    n_past = sum(1 for c in cases.values() if "reads_past_end" in c.get("flags", []) or "reads_before_start" in c.get("flags", []))
    ooc_path = f"{R}/out_of_contract.tsv"
    n_ooc = max(0, sum(1 for _ in open(ooc_path)) - 1) if os.path.exists(ooc_path) else 0
    summary = {
        "module": "retime-flow", "plugin": "Twixtor_64.aex",
        "plugin_sha256": "c549c9f67545e5d490cac47394b0252635ecc80b2c718c3240cc99d4467b72b6",
        "effect": "retime", "status": "in_progress",
        "blockers": ["emulator side not run yet"],
        "cases": {"in_contract": len(cases), "compared": 0, "identical": 0, "differing": 0, "out_of_contract": n_ooc, "logic_parity": 0},
        "values": {"bytes_compared": 0, "floats_compared": 0, "differing_values": 0, "max_ulp": None},
        "call_records": {"compared": 0, "matching": 0},
        "mutants": {"total": len(real), "killed": 0, "survivors": []},
        "stand_ins": [
            f"{n_stand} cases with keyed speed_percent (AE keyframe interpolation) are tagged stand_in=keyframe_interpolation; they are logic_parity once compared",
            "draft quality cases (testsets/draft) need the same quality setting on the host side as vfx --quality draft",
        ],
        "fixes": [], "commands": json.load(open(f"{R}/commands.json")) if os.path.exists(f"{R}/commands.json") else [],
        "measured_at": a.date,
        "phase_a": {
            "grid_cases": sum(1 for c in cases.values() if c["set"] == "grid"),
            "draft_cases": sum(1 for c in cases.values() if c["set"] == "draft"),
            "base_cases_rendered": len(base) - len(refused),
            "base_cases_refused_by_rewrite": refused,
            "base_nondeterministic_cases": nondet,
            "cases_reading_outside_clip": n_past,
            "mutants_built_and_run": len(built),
            "mutants_changing_a_base_output": len(built) - len(unchanged),
            "mutants_changing_nothing_yet": unchanged,
            "control_m00_cases_changed": None if ctrl is None or not ctrl["built"] else len(ctrl["changed"]),
            "note": "mutants are not killed until compared with the emulator's outputs in phase B",
        },
    }
    with open(f"{R}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
        fh.write("\n")
    print(open(f"{R}/mutant_table.tsv").read())
    print(json.dumps(summary["phase_a"], indent=1))


if __name__ == "__main__":
    main()
