#!/usr/bin/env python3
"""gen_supp_cv.py OUT_DIR: the proposed supplementary case set of create-vectors (f7-close-cv), standard library only.

Purpose: decide whether the original accepts frames with a side of 1 that are longer than the tiny-frame rule's 5x5.
Every grid case with a side of 1 (1x1, 1x4, 4x1) stops the original (read fault at CreateMotionVectors_64.aex+0x7ab40c),
so the rewrite's zero vector image of such a frame (`vectors::zero_vectors_image`, the rule G7 family) is never compared
and the mutants m05 (8 bpc red centred on 128) and m06 (32 bpc green one ULP above 0.5) survive. If the original renders
these cases, they kill m05 (d8) and m06 (d32) by output; if it stops on them too, every frame with a side of 1 is
outside what the original accepts at these lengths, and the two mutants mutate code that no accepted input reaches.

Cases (Vectors display, default parameters, 3 frames at 24 fps): `supp-side1.ramp-1x8.d8`, `supp-side1.ramp-8x1.d32`,
and their twins at the other depth. Inputs: a ramp that moves by one pixel a frame (8 bpc: value 32 * ((i + f) % 8),
alpha 255; float: the same / 255, alpha 1.0), written as the grid writes its inputs (`rgba` interleaved bytes,
`gbrapf32le` planar G, B, R, A floats).
"""
import json
import os
import struct
import sys

out = sys.argv[1]
os.makedirs(os.path.join(out, "inputs"), exist_ok=True)
FRAMES = 3
inputs, cases = [], []


def write_input(w, h, depth):
    n = w * h
    data = bytearray()
    for f in range(FRAMES):
        vals = [32 * ((i + f) % 8) for i in range(n)]
        if depth == 8:
            for v in vals:
                data += bytes((v, (v + 64) % 256, 255 - v, 255))
        else:
            r = [v / 255.0 for v in vals]
            g = [((v + 64) % 256) / 255.0 for v in vals]
            b = [(255 - v) / 255.0 for v in vals]
            a = [1.0] * n
            for plane in (g, b, r, a):
                data += struct.pack("<%df" % n, *plane)
    pix = {8: "rgba", 32: "gbrapf32le"}[depth]
    name = "inputs/ramp-%dx%d.d%d.f%d.%s" % (w, h, depth, FRAMES, pix)
    open(os.path.join(out, name), "wb").write(bytes(data))
    inputs.append({"file": name, "source": "ramp", "width": w, "height": h, "depth": depth, "pix_fmt": pix, "frames": FRAMES})
    return name, pix


for (w, h) in ((1, 8), (8, 1)):
    for depth in (8, 32):
        name, pix = write_input(w, h, depth)
        cid = "create-vectors.supp-side1.ramp-%dx%d.d%d" % (w, h, depth)
        cases.append({"id": cid, "effect": "create-vectors", "label": "supp-side1", "params": [], "depth": depth, "width": w,
                      "height": h, "source": "ramp", "input": name, "reference": "%s.%s" % (cid, pix), "kind": "temporal",
                      "frames": FRAMES, "fps": "24"})
json.dump({"format": 1, "ffmpeg": None, "inputs": inputs, "cases": cases}, open(os.path.join(out, "cases.json"), "w"), indent=1)
json.dump({c["id"]: {"group": "supplementary", "tags": ["side-of-1"], "standin": None} for c in cases},
          open(os.path.join(out, "case_meta.json"), "w"), indent=1)
print("wrote %d cases and %d inputs into %s" % (len(cases), len(inputs), out))
