#!/usr/bin/env python3
"""Move the cases the base rewrite refuses at render (nonzero exit) from cases.json to
out_of_contract.json, keeping the full list in cases.all.json. Never counted as pass or failure.

  python3 split_refused.py --testset DIR --base results/base.tsv
"""
import argparse, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", required=True)
    ap.add_argument("--base", required=True)
    a = ap.parse_args()
    cp = os.path.join(a.testset, "cases.json")
    allp = os.path.join(a.testset, "cases.all.json")
    src = allp if os.path.exists(allp) else cp
    ts = json.load(open(src))
    json.dump(ts, open(allp, "w"), indent=1)
    rows = [l.split("\t") for l in open(a.base).read().splitlines()[1:]]
    bad = {r[0]: r[5] for r in rows if r[1] != "0"}
    keep = [c for c in ts["cases"] if c["id"] not in bad]
    moved = []
    for c in ts["cases"]:
        if c["id"] in bad:
            c = dict(c)
            c["reason"] = "the rewrite refuses the render: " + bad[c["id"]]
            moved.append(c)
    ts["cases"] = keep
    json.dump(ts, open(cp, "w"), indent=1)
    op = os.path.join(a.testset, "out_of_contract.json")
    oc = json.load(open(op))
    oc["cases"] = [c for c in oc["cases"] if not c.get("reason", "").startswith("the rewrite refuses the render")] + moved
    json.dump(oc, open(op, "w"), indent=1)
    keep_ids = {c["id"] for c in keep}
    rows = open(os.path.join(a.testset, "grid.tsv")).read().splitlines()
    with open(os.path.join(a.testset, "grid.tsv"), "w") as fh:
        fh.write(rows[0] + "\n")
        for line in rows[1:]:
            if line.split("\t")[0] in keep_ids:
                fh.write(line + "\n")
    print(f"{len(keep)} cases kept, {len(moved)} moved to out_of_contract.json")


if __name__ == "__main__":
    main()
