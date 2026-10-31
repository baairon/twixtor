#!/usr/bin/env python3
"""gen_supp.py OUTDIR: the supplementary retime-time set proposed by agent f5-closeout (Python 3 standard library only).

Frame Number mode at a composition rate of 15 fps with Input: Frame Rate 29.97 (snapped to the double 29.97; In FPS is
Out FPS off, so the rates are unequal) and Frame Num 475: the source position is (29.97 / 15) * 475 = 949.05, so the
output blends source frames 949 and 950 with the weight 0.95. The duration of a source frame is
dur = (float)((100 * 15) / 29.97) = 50.050048828125 in the rewrite (the original computes it in double and narrows
once); frame_time(949) = trunc((float)(949 * dur) + dur * 0.05) = 47499 ticks,
clip frame 474. A source frame duration computed in single precision (mutant m20) is 50.050052642822266 and gives
47500 ticks, clip frame 475, so the output changes. The clip alternates its colour every frame, so the clip frames 474
and 475 differ in every sample. The 32 bpc twin has the same layout. Two output frames, a clip of 480 frames.
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
    cid = "retime.supp-dur-f32-fn475-at15.alt-1x1.d%d" % depth
    cases.append({"id": cid, "effect": "retime", "label": "supp-dur-f32-fn475-at15",
                  "params": ["vectors=1", "time_remap_mode=2", "frame_num=475", "in_fps_is_out_fps=0", "input_frame_rate=29.97"],
                  "depth": depth, "width": 1, "height": 1, "source": "alt", "input": name, "reference": "%s.%s" % (cid, pix),
                  "kind": "temporal", "frames": 2, "fps": "15",
                  "note": "kills m20_dur_in_f32 when the original matches the rewrite: clip frames 474 and 475 (rewrite) "
                          "against 475 and 475 (m20); see the docstring of gen_supp.py"})
with open(os.path.join(out, "cases.json"), "w", newline="\n") as fh:
    json.dump({"format": 1, "ffmpeg": "none (inputs written by parity/retime-time/supp/gen_supp.py)", "inputs": inputs,
               "cases": cases}, fh, indent=1)
    fh.write("\n")
print("wrote %d cases to %s" % (len(cases), out))
