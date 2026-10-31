#!/usr/bin/env python3
"""Supplementary case set of fgbg-core (agent C2, 2026-10-31): Draw Geom on the Source display WITH Alt Color Source.

Why: no case of the 399 case core grid (nor of the layers grid) sets `alt_source_a`. With Alt Color Source unset the
Source display is the plugin's missing layer fill and nothing is drawn on it (fgbg-core fix 01), so the Draw Geom
cases of the layers grid and the Draw Geom mutants (fgbg-layers M20 to M24) test nothing. This set takes the Draw
Geom cases of the fgbg-layers grid (`testset_v2`, the same case format) and sets `alt_source_a`, plus a few new
sizes and depths. It is reported apart from the 399; the merge owner decides whether it joins a grid.

usage: gen_supp.py --layers-testset DIR --out DIR
Writes OUT/cases.json (format 1, with the `inputs` list), OUT/grid.tsv (id, block, tags) and OUT/inputs, a link to
the layers test set's inputs (no clip is written here: every clip named is one the layers grid already has).
Standard library only, deterministic.
"""
import argparse
import copy
import hashlib
import json
import os
import re
import sys

PIX = {8: "rgba", 16: "rgba64le", 32: "gbrapf32le"}
BLOCK = "supp-geom-alt"

# (label of the layers grid case, Alt Color Source: "self" = the case's own input clip, or a clip file)
FROM_GRID = [
    ("dg1-n2", "self"), ("dg2-n2", "self"), ("dg3-n2", "self"), ("dg4-n2", "self"), ("dg5-n2", "self"), ("dg6-n2", "self"),
    ("dg2-n0", "self"), ("dg3-n0", "self"), ("dg6-n0", "self"),
    ("dg2-n3", "self"), ("dg5-n3", "self"),
    ("edge0", "self"),
    ("overlap-order", "self"), ("overlap-order-d32", "self"), ("overlap-reverse", "self"),
    ("alpha-src", "self"), ("alpha-src-d16", "self"),
    ("keyed-pos", "self"), ("keyed-pos-n2", "self"),
    ("disc-subpix", "self"), ("disc-subpix-d32", "self"),
    ("disp1-no-draw", "self"),
]
# the same drawing on another clip than the input: the points go on the Alt Color Source frame
OTHER_CLIP = [
    ("dg2-n0", "inputs/pan-32x24.d8.f3.rgba"),
    ("alpha-src", "inputs/flat-16x12.d8.f3.rgba"),
]
# new sizes (odd sides) and depths: (base label, input clip, width, height, params replacing the points)
PTS3 = ["pt1_use=5", "pt1_pos=4.5,3.25", "pt1_color=255,0,0", "pt2_use=5", "pt2_pos=9,6", "pt2_color=0,0,255",
        "pt3_use=5", "pt3_pos=0,0", "pt3_color=0,255,0"]
NEW_SIZES = [
    ("alpha-src", "inputs/box1-17x13.d8.f3.rgba", 17, 13, 8, PTS3),
    ("alpha-src-d16", "inputs/box1-23x31.d16.f3.rgba64le", 23, 31, 16, PTS3),
    ("overlap-order-d32", "inputs/box1-37x23.d32.f3.gbrapf32le", 37, 23, 32, PTS3),
    ("overlap-order-d32", "inputs/box1-24x16.d32.f3.gbrapf32le", 24, 16, 32, PTS3),
    ("alpha-src", "inputs/box1-24x16.d8.f3.rgba", 24, 16, 8, ["draw_geom=3"] + PTS3),
]

NAME = re.compile(r"^inputs/(?P<src>.+)-(?P<w>\d+)x(?P<h>\d+)\.d(?P<d>\d+)\.f(?P<f>\d+)\.(?P<pix>\w+)$")


def entry(path):
    m = NAME.match(path)
    if not m:
        raise SystemExit("gen_supp: clip name not understood: " + path)
    d = int(m["d"])
    return {"file": path, "source": m["src"], "width": int(m["w"]), "height": int(m["h"]), "depth": d,
            "pix_fmt": PIX[d], "frames": int(m["f"])}


def with_params(params, repl):
    """params with the keys of repl replaced (or added), order kept."""
    keys = {p.split("=", 1)[0] for p in repl}
    return [p for p in params if p.split("=", 1)[0] not in keys] + list(repl)


def finish(c, label, alt, tags):
    c["params"] = list(c["params"])
    lay = dict(c.get("layers") or {})
    lay["alt_source_a"] = alt
    c["layers"] = lay
    c["label"] = label
    h = hashlib.sha1(json.dumps([label, c["input"], c["params"], sorted(lay.items())], sort_keys=True).encode()).hexdigest()[:8]
    src = entry(c["input"])["source"]
    c["id"] = "retime-fgbg.{}.{}-{}x{}.d{}.{}".format(label, src, c["width"], c["height"], c["depth"], h)
    c["reference"] = c["id"] + "." + PIX[c["depth"]]
    return c, tags


def build(cases):
    by_label = {}
    for c in cases:
        by_label.setdefault(c["label"], []).append(c)
    out = []
    for label, alt in FROM_GRID:
        for c in by_label.get(label, []):
            c = copy.deepcopy(c)
            a = c["input"] if alt == "self" else alt
            out.append(finish(c, "supp-" + label + "-alt", a, ["alt-self"]))
    for label, alt in OTHER_CLIP:
        for c in by_label.get(label, [])[:1]:
            c = copy.deepcopy(c)
            out.append(finish(c, "supp-" + label + "-altother", alt, ["alt-other"]))
    for label, clip, w, h, d, repl in NEW_SIZES:
        c = copy.deepcopy(by_label[label][0])
        e = entry(clip)
        assert (e["width"], e["height"], e["depth"]) == (w, h, d), clip
        c["input"], c["width"], c["height"], c["depth"], c["frames"] = clip, w, h, d, e["frames"]
        c["params"] = with_params([p for p in c["params"] if not p.startswith("pt")], repl)
        for k in list((c.get("layers") or {})):
            c["layers"].pop(k)
        out.append(finish(c, "supp-size-{}x{}-d{}".format(w, h, d), clip, ["alt-self", "new-size"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers-testset", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = json.load(open(os.path.join(a.layers_testset, "cases.json")))
    made = build(src["cases"])
    ids = [c["id"] for c, _ in made]
    if len(ids) != len(set(ids)):
        raise SystemExit("gen_supp: duplicate ids")
    files = sorted({c["input"] for c, _ in made} | {v for c, _ in made for v in c["layers"].values()})
    inp = os.path.join(a.layers_testset, "inputs")
    missing = [f for f in files if os.path.isdir(inp) and not os.path.exists(os.path.join(a.layers_testset, f))]
    if missing:
        raise SystemExit("gen_supp: clips missing from the layers test set: " + ", ".join(missing))
    os.makedirs(a.out, exist_ok=True)
    link = os.path.join(a.out, "inputs")
    if not os.path.lexists(link):
        try:
            os.symlink(os.path.abspath(inp), link)
        except OSError as e:
            print("gen_supp: no inputs link ({}); link {} to OUT/inputs before a run".format(e, inp), file=sys.stderr)
    doc = {"format": 1, "ffmpeg": "none (clips of the fgbg-layers test set, linked)",
           "note": "fgbg-core supplementary set: Draw Geom on the Source display with Alt Color Source set (gen_supp.py)",
           "inputs": [entry(f) for f in files], "cases": [c for c, _ in made]}
    with open(os.path.join(a.out, "cases.json"), "w", newline="\n") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    with open(os.path.join(a.out, "grid.tsv"), "w", newline="\n") as fh:
        fh.write("id\tblock\ttags\n")
        for c, tags in made:
            fh.write("{}\t{}\t{}\n".format(c["id"], BLOCK, ",".join(tags)))
    with open(os.path.join(a.out, "ids.txt"), "w", newline="\n") as fh:
        fh.write("\n".join(ids) + "\n")
    print("gen_supp: {} cases, {} clips -> {}".format(len(made), len(files), a.out), file=sys.stderr)


if __name__ == "__main__":
    main()
