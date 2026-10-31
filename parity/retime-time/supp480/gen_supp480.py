#!/usr/bin/env python3
"""gen_supp480.py OUTDIR: the supplementary retime-time set of `../supp/` (mutant m20) with 480 output frames instead of 2
(agent f7-supp; Python 3 standard library only).

The two cases of `../supp/` render 2 output frames from a 480 frame clip. `vfx` accepts that, but neither the harness
(before `fixes/harness_80_input_clip_length.diff`) nor `fx-parity run` does: both require the input to hold exactly
`frames` frames (fx-parity: "input ... has 1920 bytes, 2 frames of 1x1 at 8 bpc are 8"), so the module's grade cannot
compare them. Here the output has as many frames as the clip, 480. Frame Num is the constant 475, so every output frame
sits at the same source position, (29.97 / 15) * 475 = 949.05, and blends clip frames 474 and 475 exactly as frames 0
and 1 of `../supp/` do (the arithmetic is in the docstring of `../supp/gen_supp.py`); the composition is now 32 s long,
so the source time 31.67 s also lies inside it. Same inputs (written again here, byte equal to `../supp/inputs/`), same
parameters, ids with `-480f`.
"""
import json
import os
import struct
import sys

out = sys.argv[1]
os.makedirs(os.path.join(out, "inputs"), exist_ok=True)
N = 480
inputs, cases = [], []
for depth, pix in ((8, "rgba"), (32, "gbrapf32le")):
    name = "inputs/alt-1x1.d%d.f%d.%s" % (depth, N, pix)
    buf = bytearray()
    for i in range(N):
        hi = i % 2 == 1
        if depth == 8:
            buf += bytes((255 if hi else 0, (i * 37) % 256, 255 - (i % 256), 255))
        else:  # planes g, b, r, a of the one pixel
            r, g, b = (1.0 if hi else 0.0), ((i * 37) % 256) / 255.0, (255 - (i % 256)) / 255.0
            buf += struct.pack("<4f", g, b, r, 1.0)
    with open(os.path.join(out, name), "wb") as fh:
        fh.write(bytes(buf))
    inputs.append({"file": name, "source": "alt", "width": 1, "height": 1, "depth": depth, "pix_fmt": pix, "frames": N})
    cid = "retime.supp-dur-f32-fn475-at15-480f.alt-1x1.d%d" % depth
    cases.append({"id": cid, "effect": "retime", "label": "supp-dur-f32-fn475-at15-480f",
                  "params": ["vectors=1", "time_remap_mode=2", "frame_num=475", "in_fps_is_out_fps=0", "input_frame_rate=29.97"],
                  "depth": depth, "width": 1, "height": 1, "source": "alt", "input": name, "reference": "%s.%s" % (cid, pix),
                  "kind": "temporal", "frames": N, "fps": "15",
                  "note": "the case of ../supp/ with as many output frames as clip frames, so fx-parity and the harness "
                          "accept it; kills m20_dur_in_f32 by output (clip frames 474 and 475 against 475 and 475)"})
with open(os.path.join(out, "cases.json"), "w", newline="\n") as fh:
    json.dump({"format": 1, "ffmpeg": "none (inputs written by parity/retime-time/supp480/gen_supp480.py)", "inputs": inputs,
               "cases": cases}, fh, indent=1)
    fh.write("\n")
print("wrote %d cases to %s" % (len(cases), out))
