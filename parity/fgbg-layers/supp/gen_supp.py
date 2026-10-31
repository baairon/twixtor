#!/usr/bin/env python3
"""Supplementary Draw Geom case set of the fgbg-layers module (agent C3, 2026-10-31).

No case of the 386 case grid sets Alt Color Source, so its Source display cases show the missing layer fill and test
nothing of Draw Geom (rule G3). Every case here sets `alt_source_a`, so the original draws its points. Same case
format as the grid (fx-parity format 1 with the `inputs` list, `grid.tsv` with id, block and tags); clips are made with
the grid generator's scenes and mattes (`gen_grid.py` next to this folder). Kept apart from the grid: the merge owner
decides whether it joins it. Standard library only, deterministic.

  python3 gen_supp.py --out DIR [--gen-grid ../gen_grid.py]
"""
import argparse
import hashlib
import importlib.util
import json
import os

PIX = {8: "rgba", 16: "rgba64le", 32: "gbrapf32le"}
EFFECT = "retime-fgbg"


def load_gen(path):
    spec = importlib.util.spec_from_file_location("gen_grid", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build(g):
    cases, rows, files = [], [], {}

    def clip(kind, name, w, h, d, frames=3):
        """kind: ("scene", source) or ("matte", kind, box)."""
        path = "inputs/%s-%dx%d.d%d.f%d.%s" % (name, w, h, d, frames, PIX[d])
        files[path] = (kind, w, h, d, frames)
        return path

    def add(block, label, w, h, d, params, alt="box1", source="box1", mattes=(), tags=()):
        params = list(params)
        layers = {"alt_source_a": clip(("scene", alt), alt, w, h, d)}
        for i, m in enumerate(mattes):
            kind, _, box = m.partition(":")
            layers["fg%d_matte" % (i + 1)] = clip(("matte", kind, int(box or i)), "m-%s-%s" % (kind, box or i), w, h, d)
        inp = clip(("scene", source), source, w, h, d)
        key = json.dumps([label, w, h, d, params, alt, source, list(mattes)])
        cid = "%s.supp-%s.%s-%dx%d.d%d.%s" % (EFFECT, label, alt, w, h, d, hashlib.sha1(key.encode()).hexdigest()[:8])
        cases.append({"id": cid, "effect": EFFECT, "label": "supp-" + label, "params": params, "depth": d, "width": w,
                      "height": h, "source": source, "input": inp, "reference": cid + "." + PIX[d], "kind": "temporal",
                      "frames": 3, "fps": "24", "layers": dict(sorted(layers.items()))})
        rows.append((cid, block, ",".join(tags)))

    G = ["display=2", "draw_geom=2"]
    for d in (8, 16, 32):
        for (w, h) in ((16, 12), (23, 17), (32, 24)):
            cx, cy = w // 2, h // 2
            add("cross", "int", w, h, d, G + ["pt1_use=5", "pt1_pos=%d,%d" % (cx, cy)])
            add("cross", "sub", w, h, d, G + ["pt1_use=5", "pt1_pos=%g,%g" % (cx - 4.75, cy - 2.5), "pt1_color=255,0,0",
                                             "pt2_use=5", "pt2_pos=%g,%g" % (cx + 0.5, cy + 0.5), "pt2_color=0,255,0",
                                             "pt3_use=5", "pt3_pos=%g,%g" % (cx + 4.1, cy + 2.3), "pt3_color=0,0,255"])
            add("cross", "overlap", w, h, d, G + ["pt1_use=5", "pt1_pos=%d,%d" % (cx, cy), "pt1_color=255,0,0",
                                                 "pt2_use=5", "pt2_pos=%g,%g" % (cx + 1, cy + 0.5), "pt2_color=0,0,255"])
            add("cross", "edge", w, h, d, G + ["pt1_use=5", "pt1_pos=0,0", "pt1_color=255,0,0", "pt2_use=5",
                                              "pt2_pos=%d,%d" % (w - 1, h - 1), "pt2_color=0,255,0", "pt3_use=5",
                                              "pt3_pos=-0.5,-0.5", "pt3_color=0,0,255", "pt4_use=5",
                                              "pt4_pos=%g,3" % (w - 0.5), "pt4_color=255,255,0", "pt5_use=5",
                                              "pt5_pos=-2.6,%d" % cy, "pt5_color=255,0,255"])
        add("cross", "alpha-src", 16, 12, d, G + ["pt1_use=5", "pt1_pos=8,6", "pt2_use=5", "pt2_pos=3.4,2.6",
                                                 "pt2_color=10,100,200"], alt="box1a")
        add("cross", "colours", 32, 24, d, G + ["pt1_use=5", "pt1_pos=5,5", "pt1_color=1,2,3", "pt2_use=5",
                                               "pt2_pos=12,6", "pt2_color=254,128,127", "pt3_use=5", "pt3_pos=20,8",
                                               "pt3_color=77,200,13", "pt4_use=5", "pt4_pos=26,18",
                                               "pt4_color=255,255,255", "pt5_use=5", "pt5_pos=8,17",
                                               "pt5_color=0,0,0"], alt="pan")
        add("cross", "all12", 32, 24, d, G + sum((["pt%d_use=5" % n, "pt%d_pos=%g,%g" % (n, 2.5 * n, 3 + (n * 7) % 19)]
                                                 for n in range(1, 13)), []))
    # groups: points on FG1, FG2 and Main_BG with two FG mattes, every Draw Geom value (8 bpc)
    pts = ["pt1_use=2", "pt1_pos=6.3,5.6", "pt2_use=3", "pt2_pos=14.5,9.5", "pt3_use=5", "pt3_pos=24,12.25",
           "pt3_color=255,0,0"]
    for geom in range(1, 7):
        add("groups", "dg%d-n2" % geom, 32, 24, 8, ["display=2", "draw_geom=%d" % geom] + pts, source="box2",
            alt="box2", mattes=("hard", "hard"))
    add("groups", "dg2-n0", 32, 24, 8, G + pts)
    add("groups", "keyed", 32, 24, 8, G + ["pt1_use=5", "pt1_pos=10,8@0;14,10@2"], tags=("standin",))
    return cases, rows, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen-grid", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gen_grid.py"))
    a = ap.parse_args()
    g = load_gen(a.gen_grid)
    cases, rows, files = build(g)
    os.makedirs(os.path.join(a.out, "inputs"), exist_ok=True)
    inputs = []
    for path, (kind, w, h, d, frames) in sorted(files.items()):
        if kind[0] == "scene":
            src = g.SOURCES[kind[1]]
            fn = src
        else:
            fn = (lambda m: (lambda x, y, f, ww, hh, dd: m(x, y, f, ww, hh)))(g.matte(kind[1], kind[2], False, d))
        with open(os.path.join(a.out, path), "wb") as fh:
            fh.write(g.render_clip(fn, w, h, d, frames))
        inputs.append({"file": path, "source": kind[1], "width": w, "height": h, "depth": d, "pix_fmt": PIX[d], "frames": frames})
    doc = {"format": 1, "ffmpeg": "none (raw inputs written by gen_supp.py)", "inputs": inputs, "cases": cases}
    with open(os.path.join(a.out, "cases.json"), "w") as fh:
        json.dump(doc, fh, indent=1)
    with open(os.path.join(a.out, "grid.tsv"), "w") as fh:
        fh.write("id\tblock\ttags\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    with open(os.path.join(a.out, "ids.txt"), "w") as fh:
        fh.write("\n".join(c["id"] for c in cases) + "\n")
    print(len(cases), "cases,", len(inputs), "clips ->", a.out)


if __name__ == "__main__":
    main()
