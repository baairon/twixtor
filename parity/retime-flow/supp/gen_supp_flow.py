#!/usr/bin/env python3
"""gen_supp_flow.py OUTDIR GRID_TESTSET: the supplementary retime-flow set proposed by agent f5-closeout (standard library).

The exactness test of the vector path (`timeline.rs` `exact_position`): a source position whose fraction lies within
EXACT_TOLERANCE = 0.005 of a whole frame takes the fast path, any other position the vector path. No grid case of
either module has a position whose distance to the nearest whole frame lies in (0.005, 0.01], so the tolerance itself
is never tested (mutant m09 of retime-time, 0.005 -> 0.01, survives both grids). At Speed 99 % (101 %) and equal rates
the position of output frame 1 is 0.99 (1.01): 0.01 from frame 1 in single precision (0.00999999 for 99 %), so the
rewrite runs the vector path there and a tolerance of 0.01 would take the fast path. Frames 2 and 3 (1.98, 2.97;
2.02, 3.03) are 0.02 and 0.03 away and take the vector path either way. The clip is the grid's own `disc` input of
40x30 at 8 bpc, copied from GRID_TESTSET.
"""
import json
import os
import shutil
import sys

out, grid = sys.argv[1], sys.argv[2]
src = json.load(open(os.path.join(grid, "cases.json")))
inp = next(i for i in src["inputs"] if i["file"] == "inputs/disc-40x30.d8.f4.rgba")
os.makedirs(os.path.join(out, "inputs"), exist_ok=True)
shutil.copyfile(os.path.join(grid, inp["file"]), os.path.join(out, inp["file"]))
cases = []
for sp in ("99", "101"):
    cid = "retime.supp-exact-tol-s%s.disc-40x30.d8" % sp
    cases.append({"id": cid, "effect": "retime", "label": "supp-exact-tol-s%s" % sp, "params": ["speed_percent=%s" % sp],
                  "depth": 8, "width": 40, "height": 30, "source": "disc", "input": inp["file"],
                  "reference": cid + ".rgba", "kind": "temporal", "frames": 4, "fps": "24", "group": "supp",
                  "note": "output frame 1 sits 0.01 from a whole frame: the vector path in the rewrite, the fast path "
                          "with EXACT_TOLERANCE 0.01 (mutant m09 of retime-time); see the docstring of gen_supp_flow.py"})
doc = {k: v for k, v in src.items() if k not in ("inputs", "cases")}
doc.update(inputs=[inp], cases=cases)
with open(os.path.join(out, "cases.json"), "w", newline="\n") as fh:
    json.dump(doc, fh, indent=1)
    fh.write("\n")
print("wrote %d cases to %s" % (len(cases), out))
