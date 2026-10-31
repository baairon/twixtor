#!/usr/bin/env python3
"""Builds an fx-parity style test set of probe cases for the retime effect from a list of (name, size, input, params) and
summarises a run of it.

  probe_set.py build   --out DIR --template CASES.json --spec SPEC.json    SPEC.json: [{"name": ..., "size": "64x48", "input": "grey",
                                                                            "params": ["speed_percent=50"], "frames": 4}, ...]
  probe_set.py summary --out DIR --summary SUM.json                        one line per case: status and error text

Inputs are flat black/white/grey, noise or a gradient (4 or more frames, 8 bpc, no layers). Every case of a run is a black-box
experiment: what changed is in its id and params, the result in the summary and the call record.
"""
import argparse
import json
import random
from pathlib import Path


def make_input(kind, w, h, frames, rnd):
    out = bytearray()
    for f in range(frames):
        for y in range(h):
            for x in range(w):
                if kind == "black":
                    out += bytes((0, 0, 0, 255))
                elif kind == "white":
                    out += bytes((255, 255, 255, 255))
                elif kind == "grey":
                    out += bytes((128, 128, 128, 255))
                elif kind == "noise":
                    out += bytes((rnd.randrange(256), rnd.randrange(256), rnd.randrange(256), 255))
                else:
                    out += bytes(((x * 255) // max(w - 1, 1), (y * 255) // max(h - 1, 1), (f * 80) % 256, 255))
    return bytes(out)


def build(a):
    out = Path(a.out)
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    tpl = next(c for c in json.loads(Path(a.template).read_text())["cases"] if c["effect"] == "retime" and c["label"] == "all-default")
    rnd = random.Random(11)
    cases = []
    for sp in json.loads(Path(a.spec).read_text()):
        w, h = (int(v) for v in sp["size"].split("x"))
        frames = sp.get("frames", 4)
        name = f"{sp['input']}-{w}x{h}.d8.f{frames}.rgba"
        if not (out / "inputs" / name).exists():
            (out / "inputs" / name).write_bytes(make_input(sp["input"], w, h, frames, rnd))
        cid = f"probe.{sp['name']}"
        c = dict(tpl)
        c.update(id=cid, width=w, height=h, frames=frames, input=f"inputs/{name}", layers={}, params=sp.get("params", []), label=sp["name"],
                 reference=f"{cid}.rgba", source=sp["input"])
        cases.append(c)
    (out / "cases.json").write_text(json.dumps({"cases": cases}, indent=1))
    print(len(cases), "cases in", out)


def summary(a):
    s = json.loads(Path(a.summary).read_text())
    for c in s["cases"]:
        print(f"{c['id']:44s} {c['status']:10s} {str(c.get('error') or '')[:100]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--template", required=True)
    b.add_argument("--spec", required=True)
    sm = sub.add_parser("summary")
    sm.add_argument("--out", required=True)
    sm.add_argument("--summary", required=True)
    args = ap.parse_args()
    build(args) if args.cmd == "build" else summary(args)
