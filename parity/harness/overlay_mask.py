#!/usr/bin/env python3
"""Measures the fixed pixel pattern seen in the plugin's output. Output only: nothing here sets or reads a licence,
activation or machine value, and nothing is said about what the pattern is.

  overlay_mask.py build   --out DIR --template CASES.json   writes an fx-parity style test set: flat black, white, mid grey, noise and a
                                                            gradient at 16x12, 37x23, 64x48 and 129x71, 4 frames, 8 bpc, plus a few
                                                            param settings on 64x48
  overlay_mask.py analyze --out DIR --refs REFS --vfx VFX   reads the plugin's renders (REFS/<case id>.rgba, the raw outputs) and the
                                                            rewrite's (vfx apply) and prints the mask size per frame size

The mask is the set of pixels whose plugin output is identical across the three flat-field inputs (black, white, mid grey) and
differs from the rewrite's output on a flat input. On the mask pixels the analysis reports whether the plugin value depends on the
underlying pixel (compared with the noise input), whether the pixel set is the same in every frame, and whether it is the same under
the other param settings. Render the set with run_harness.py --testset DIR --refs REFS.
"""
import argparse
import json
import random
import subprocess
from pathlib import Path

SIZES = [(16, 12), (37, 23), (64, 48), (129, 71)]
FRAMES = 4
KINDS = ("black", "white", "grey", "noise", "gradient")
SETTINGS = {"default": [], "speed50": ["speed_percent=50"], "speed200": ["speed_percent=200"],
            "remap2": ["time_remap_mode=2", "frame_num=1"], "vectors2": ["vectors=2"]}


def make_input(kind, w, h, rnd):
    frames = []
    for f in range(FRAMES):
        px = bytearray()
        for y in range(h):
            for x in range(w):
                if kind == "black":
                    px += bytes((0, 0, 0, 255))
                elif kind == "white":
                    px += bytes((255, 255, 255, 255))
                elif kind == "grey":
                    px += bytes((128, 128, 128, 255))
                elif kind == "noise":
                    px += bytes((rnd.randrange(256), rnd.randrange(256), rnd.randrange(256), 255))
                else:
                    px += bytes(((x * 255) // max(w - 1, 1), (y * 255) // max(h - 1, 1), (f * 80) % 256, 255))
        frames.append(bytes(px))
    return b"".join(frames)


def build(a):
    out = Path(a.out)
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    template = json.loads(Path(a.template).read_text())["cases"]
    tpl = next(c for c in template if c["effect"] == "retime" and c["label"] == "all-default")
    cases = []
    rnd = random.Random(7)
    for w, h in SIZES:
        for kind in KINDS:
            name = f"{kind}-{w}x{h}.d8.f{FRAMES}.rgba"
            (out / "inputs" / name).write_bytes(make_input(kind, w, h, rnd))
            for sname, params in SETTINGS.items():
                if sname != "default" and ((w, h) != (64, 48) or kind not in ("grey", "noise")):
                    continue
                cid = f"ovl.{sname}.{kind}-{w}x{h}"
                c = dict(tpl)
                c.update(id=cid, width=w, height=h, input=f"inputs/{name}", layers={}, params=params, label=sname,
                         reference=f"{cid}.rgba", source=kind)
                cases.append(c)
    (out / "cases.json").write_text(json.dumps({"cases": cases}, indent=1))
    print(len(cases), "cases in", out)


def frame(buf, w, h, f):
    n = w * h * 4
    return buf[f * n:(f + 1) * n]


def analyze(a):
    out, refs = Path(a.out), Path(a.refs)
    cases = {c["id"]: c for c in json.loads((out / "cases.json").read_text())["cases"]}
    cache = {}

    def plugin(cid):
        p = refs / f"{cid}.rgba"
        return p.read_bytes() if p.exists() else None

    def rewrite(cid):
        if cid not in cache:
            c = cases[cid]
            cmd = [a.vfx, "apply", "retime", "--width", str(c["width"]), "--height", str(c["height"]), "--depth", "8", "--fps", "24",
                   "--frames", str(FRAMES)]
            for pr in c["params"]:
                cmd += ["--param", pr]
            cmd += ["--in", str(out / c["input"])]
            cache[cid] = subprocess.run(cmd, capture_output=True).stdout
        return cache[cid]

    def mask(w, h, sname="default"):
        """{frame: set of pixel indexes}, or None when a render is missing"""
        ids = {k: f"ovl.default.{k}-{w}x{h}" for k in ("black", "white", "grey")}
        pb, pw, pg = (plugin(ids[k]) for k in ("black", "white", "grey"))
        if pb is None or pw is None or pg is None:
            return None
        rb = rewrite(ids["black"])
        m = {}
        for f in range(FRAMES):
            fb, fw, fg, fr = (frame(x, w, h, f) for x in (pb, pw, pg, rb))
            m[f] = {i // 4 for i in range(0, w * h * 4, 4) if fb[i:i + 4] == fw[i:i + 4] == fg[i:i + 4] and fb[i:i + 4] != fr[i:i + 4]}
        return m

    for w, h in SIZES:
        m = mask(w, h)
        if m is None:
            print(f"{w}x{h}: missing renders")
            continue
        sizes = [len(m[f]) for f in range(FRAMES)]
        line = f"{w}x{h}: mask size per frame {sizes} of {w * h} pixels; same pixel set in every frame: {all(m[f] == m[0] for f in m)}"
        pn, pb = plugin(f"ovl.default.noise-{w}x{h}"), plugin(f"ovl.default.black-{w}x{h}")
        if pn is not None and pb is not None:
            dep = tot = 0
            for f in range(FRAMES):
                fn, fb = frame(pn, w, h, f), frame(pb, w, h, f)
                for i in m[f]:
                    tot += 1
                    dep += fn[4 * i:4 * i + 4] != fb[4 * i:4 * i + 4]
            line += f"; mask pixels whose value differs between the noise and the black input: {dep} of {tot}"
        print(line)
    w, h = 64, 48
    base = mask(w, h)
    pb = plugin(f"ovl.default.black-{w}x{h}")
    if base and pb:
        total = sum(len(base[f]) for f in base)
        for sname in SETTINGS:
            for kind in ("grey", "noise"):
                p = plugin(f"ovl.{sname}.{kind}-{w}x{h}")
                if sname == "default" or p is None:
                    continue
                same = sum(1 for f in base for i in base[f]
                           if frame(p, w, h, f)[4 * i:4 * i + 4] == frame(pb, w, h, f)[4 * i:4 * i + 4])
                print(f"64x48 {sname}/{kind}: {same} of {total} default-mask pixels carry the same value as in the default black render")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--template", required=True)
    an = sub.add_parser("analyze")
    an.add_argument("--out", required=True)
    an.add_argument("--refs", required=True)
    an.add_argument("--vfx", required=True)
    args = ap.parse_args()
    build(args) if args.cmd == "build" else analyze(args)
