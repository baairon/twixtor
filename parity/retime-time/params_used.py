#!/usr/bin/env python3
"""A4: the plugin params the grid relies on, by rewrite param id, with the values used and the case count.

python3 params_used.py results/cases.json results/params_used.tsv
"""
import json
import sys

cases = json.load(open(sys.argv[1]))["cases"]
use = {}
for c in cases:
    for p in c["params"]:
        k, v = p.split("=", 1)
        e = use.setdefault(k, {"values": set(), "cases": 0, "keyed": 0})
        e["cases"] += 1
        e["values"].add(v)
        if "@" in v:
            e["keyed"] += 1
with open(sys.argv[2], "w") as fh:
    fh.write("param_id\tcases\tkeyed_cases\tdistinct_values\tvalues\n")
    for k in sorted(use):
        e = use[k]
        vals = sorted(e["values"])
        shown = "; ".join(vals) if len(vals) <= 12 else "; ".join(vals[:12]) + "; ..."
        fh.write("%s\t%d\t%d\t%d\t%s\n" % (k, e["cases"], e["keyed"], len(vals), shown))
print(open(sys.argv[2]).read())
