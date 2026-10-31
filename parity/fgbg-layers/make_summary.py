#!/usr/bin/env python3
"""Write results/summary.json, results/params_used.tsv and results/mutants.tsv from the result files.

Counts are computed from testset/cases.json, results/base.tsv, results/mutants_summary.tsv and
testset/out_of_contract.json; nothing is typed by hand. Phase A: nothing is compared with the emulator.

  python3 make_summary.py            (run from parity/fgbg-layers)
"""
import collections, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
PLUGIN_SHA = "58bbd99248a531acb572e5ddb13570d8102a087c32123191e2f52f477e2dcf2d"


def main():
    cases = json.load(open(os.path.join(HERE, "testset", "cases.json")))["cases"]
    oc = json.load(open(os.path.join(HERE, "testset", "out_of_contract.json")))["cases"]
    base = {l.split("\t")[0]: l.split("\t") for l in open(os.path.join(R, "base.tsv")).read().splitlines()[1:]}
    rendered = sum(1 for c in cases if base.get(c["id"], ["", "1"])[1] == "0")

    used, layers = collections.Counter(), collections.Counter()
    for c in cases:
        for p in c["params"]:
            used[p.split("=", 1)[0]] += 1
        for lid in c.get("layers", {}):
            layers[lid] += 1
    with open(os.path.join(R, "params_used.tsv"), "w") as fh:
        fh.write("kind\trewrite_id\tcases\n")
        for k, n in sorted(used.items()):
            fh.write(f"param\t{k}\t{n}\n")
        for k, n in sorted(layers.items()):
            fh.write(f"layer\t{k}\t{n}\n")

    muts = {m["id"]: m for m in json.load(open(os.path.join(HERE, "mutants.json")))}
    import glob
    by_id = {}
    for p in sorted(glob.glob(os.path.join(R, "mutants_summary*.tsv"))):
        for l in open(p).read().splitlines()[1:]:
            f = l.split("	")
            by_id[f[0]] = f
    msum = [by_id[k] for k in sorted(by_id)]
    with open(os.path.join(R, "mutants.tsv"), "w") as fh:
        fh.write("id\tfile\tstatus_vs_base\tcases_changed_vs_base\twhat\n")
        for s in msum:
            fh.write(f"{s[0]}\t{muts[s[0]]['file']}\t{s[1]}\t{s[2]}\t{s[3]}\n")
    built = [s for s in msum if s[1] in ("changed-cases", "no-change")]
    survivors = [s[0] for s in built if s[1] == "no-change"]
    unbuilt = [f"{s[0]}: {s[1]}" for s in msum if s[1] not in ("changed-cases", "no-change")]

    standin = sum(1 for l in open(os.path.join(HERE, "testset", "grid.tsv")).read().splitlines()[1:] if "standin" in l.split("\t")[2])
    summary = {
        "module": "fgbg-layers", "plugin": "TwixtorPro_64.aex", "plugin_sha256": PLUGIN_SHA, "effect": "retime-fgbg",
        "status": "in_progress",
        "blockers": ["emulator side not run yet"] + unbuilt,
        "cases": {"in_contract": len(cases), "compared": 0, "identical": 0, "differing": 0,
                  "out_of_contract": len(oc), "logic_parity": 0},
        "values": {"bytes_compared": 0, "floats_compared": 0, "differing_values": 0, "max_ulp": None},
        "call_records": {"compared": 0, "matching": 0},
        "mutants": {"total": len(built), "killed": 0, "survivors": survivors},
        "stand_ins": [
            f"{standin} cases (tag standin in testset/grid.tsv) use keyframed ptN_pos or ptN_use values: AE keyframe interpolation is replaced by the harness's labelled linear stand-in on both sides; not yet counted as logic_parity",
            "the FG matte layers are raw clips handed in as ordinary layers (how AE hands a hidden matte layer to the plugin is not provided by the harness)",
        ],
        "fixes": [],
        "commands": [
            "python3 gen_grid.py --out testset --vfx bin/vfx_base",
            "python3 render_grid.py --vfx bin/vfx_base --testset testset --out results/base.tsv --jobs 2",
            "python3 split_refused.py --testset testset --base results/base.tsv",
            "python3 render_grid.py --vfx bin/vfx_base --testset testset --cases testset/out_of_contract.json --out results/ooc.tsv --jobs 2",
            "python3 run_mutants.py --tree src_mut --testset testset --base results/base.tsv --out mutants",
            "python3 run_mutants.py --tree src_mut --testset testset --base results/base.tsv --out mutants2 --ids M05,M06   (M05 and M06 were replaced after the first run found them equivalent; results/mutants_summary_2.tsv overrides)",
            "python3 make_summary.py",
        ],
        "measured_at": "2026-10-30",
    }
    json.dump(summary, open(os.path.join(R, "summary.json"), "w"), indent=2)
    print(f"cases {len(cases)} (rendered by the base rewrite {rendered}), out of contract {len(oc)}, "
          f"mutants built {len(built)}, no change vs base {survivors}, not built {unbuilt}")


if __name__ == "__main__":
    main()
