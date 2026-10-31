#!/usr/bin/env python3
"""Grid generator for the retime-flow parity module.

Writes two fx-parity style test sets (cases.json format 1 plus raw input clips):
  OUT/grid   cases rendered at the default quality
  OUT/draft  cases rendered with `vfx --quality draft`
and OUT/out_of_contract.json (candidates, never counted as passes or failures).

Deterministic: every pixel comes from closed form functions or from numpy's legacy
RandomState with fixed seeds. Run on the VPS (numpy is in the system python3):

    python3 gen.py --out ~/twixtor-parity/retime-flow/testsets
"""
import argparse
import hashlib
import json
import math
import os
import struct

import numpy as np

PIX = {8: "rgba", 16: "rgba64le", 32: "gbrapf32le"}


def f32(bits):
    return np.float32(struct.unpack("<f", struct.pack("<I", bits))[0])


SPECIALS = [
    ("zero", f32(0x00000000)),
    ("negzero", f32(0x80000000)),
    ("mindenorm", f32(0x00000001)),
    ("maxdenorm", f32(0x007FFFFF)),
    ("fltmin", f32(0x00800000)),
    ("one", f32(0x3F800000)),
    ("above1", np.float32(2.5)),
    ("fltmax", f32(0x7F7FFFFF)),
    ("posinf", f32(0x7F800000)),
    ("neginf", f32(0xFF800000)),
    ("nan", f32(0x7FC00000)),
]
SPECIAL_BY_NAME = dict(SPECIALS)


# ---------------------------------------------------------------- pixel sources


def tex(X, Y, c):
    a = [0.37, 0.29, 0.41][c]
    b = [0.23, 0.43, 0.19][c]
    return (
        0.5
        + 0.16 * np.sin(a * X + 0.7 * c)
        + 0.16 * np.sin(b * Y - 0.3 * c)
        + 0.09 * np.sin(0.21 * (X + Y) + 1.3 * c)
        + 0.05 * np.sin(0.9 * X - 0.8 * Y + 2.0 * c)
    )


def rgb(X, Y):
    return np.stack([tex(X, Y, c) for c in range(3)], -1)


PLAIN = ["shift", "static", "flat", "subpix", "bigshift", "noise", "spin", "zoom", "disc", "adisc", "hdr", "alpha"]


def base_name(src):
    if src.startswith("spec-") or src.startswith("specA-"):
        return "shift"
    if src.startswith("ext-"):
        return "shift"
    return src


def frame_f64(src, W, H, t):
    """One frame as float64 (H, W, 4), straight alpha."""
    src = base_name(src)
    X, Y = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    A = np.ones((H, W))
    dx, dy = max(0.5, W * 0.04), H * 0.02
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    if src == "shift":
        c = rgb(X - dx * t, Y - dy * t)
    elif src == "static":
        c = rgb(X, Y)
    elif src == "flat":
        c = np.empty((H, W, 3))
        c[..., 0], c[..., 1], c[..., 2] = 0.4, 0.5, 0.6
    elif src == "subpix":
        c = rgb(X - 0.3 * t, Y - 0.2 * t)
    elif src == "bigshift":
        c = rgb(X - W * 0.3 * t, Y - H * 0.1 * t)
    elif src == "noise":
        c = np.random.RandomState(1000 + t + W * 7 + H * 13).random_sample((H, W, 3))
    elif src == "spin":
        th = 0.08 * t
        xr = cx + (X - cx) * math.cos(th) + (Y - cy) * math.sin(th)
        yr = cy - (X - cx) * math.sin(th) + (Y - cy) * math.cos(th)
        c = rgb(xr, yr)
    elif src == "zoom":
        s = 1 + 0.03 * t
        c = rgb(cx + (X - cx) / s, cy + (Y - cy) / s)
    elif src in ("disc", "adisc"):
        bg = rgb(X - 0.01 * W * t, Y)
        vx, vy = W * 0.06, H * 0.03
        dcx, dcy = W * 0.3 + vx * t, H * 0.5 + vy * t
        r = max(1.0, min(W, H) * 0.25)
        dist = np.sqrt((X - dcx) ** 2 + (Y - dcy) ** 2)
        cov = np.clip(r + 0.5 - dist, 0.0, 1.0)[..., None]
        inner = rgb(X - vx * t + 5.0, Y - vy * t + 9.0)
        c = bg * (1 - cov) + inner * cov
        if src == "adisc":
            A = 1.0 - 0.5 * cov[..., 0]
            A = np.where((X < W * 0.15) & (Y < H * 0.2), 0.0, A)
    elif src == "hdr":
        c = 3.0 * (rgb(X - dx * t, Y - dy * t) - 0.25)
    elif src == "alpha":
        c = rgb(X - dx * t, Y - dy * t)
        A = np.clip((X - (W * 0.2 + dx * t)) / (W * 0.5), 0.0, 1.0)
    else:
        raise ValueError(src)
    return np.concatenate([c, A[..., None]], -1)


def patch_rect(W, H, t):
    s = max(1, W // 3)
    sh = max(1, H // 3)
    x0 = int(W * 0.2 + round(max(0.5, W * 0.04) * t))
    x0 = max(0, min(x0, W - s))
    y0 = max(0, min(H // 3, H - sh))
    return x0, y0, s, sh


def frame_raw(src, W, H, t, depth):
    """One frame as raw bytes in the vfx format of `depth`."""
    a = frame_f64(src, W, H, t)
    if depth == 8:
        arr = np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)
    elif depth == 16:
        arr = np.clip(np.rint(a * 65535.0), 0, 65535).astype("<u2")
    else:
        arr = a.astype(np.float32)
    x0, y0, s, sh = patch_rect(W, H, t)
    if src.startswith("spec-") and depth == 32:
        name = src[5:]
        if name == "all":
            for j in range(sh):
                arr[y0 + j, x0 : x0 + s, 0] = SPECIALS[(j + t) % 11][1]
                arr[y0 + j, x0 : x0 + s, 1] = SPECIALS[(j + 3 + t) % 11][1]
        else:
            arr[y0 : y0 + sh, x0 : x0 + s, 0:3] = SPECIAL_BY_NAME[name]
    elif src.startswith("specA-") and depth == 32:
        arr[y0 : y0 + sh, x0 : x0 + s, 3] = SPECIAL_BY_NAME[src[6:]]
    elif src.startswith("ext-"):
        hi = {8: 255, 16: 65535}[depth]
        for j in range(sh):
            arr[y0 + j, x0 : x0 + s, 0:3] = hi if (j + t) % 2 == 0 else 0
            arr[y0 + j, x0 : x0 + s, 3] = hi if src == "ext-hi" or (j % 3) else 0
    if depth == 32:
        planes = np.ascontiguousarray(arr[..., [1, 2, 0, 3]].transpose(2, 0, 1)).astype("<f4")
        return planes.tobytes()
    return arr.tobytes()


# ---------------------------------------------------------------- case table


class Builder:
    def __init__(self, quality=None):
        self.quality = quality
        self.cases = []
        self.inputs = {}
        self.seen = set()
        self.rot = 0

    def add(self, group, label, src, W, H, depth, params, N=4, fps="24", stand_in=None, flags=()):
        if params and params[0] == "speed_percent=ROT":
            sp = ROT[self.rot % len(ROT)]
            self.rot += 1
            params = [f"speed_percent={sp}"] + params[1:]
        fname = f"inputs/{src}-{W}x{H}.d{depth}.f{N}.{PIX[depth]}"
        key = json.dumps([group, label, src, W, H, depth, params, N, fps, self.quality])
        h = hashlib.sha1(key.encode()).hexdigest()[:8]
        cid = f"retime.{group}-{label}.{src}-{W}x{H}.d{depth}.{h}"
        assert cid not in self.seen, cid
        self.seen.add(cid)
        self.inputs[fname] = dict(file=fname, source=src, width=W, height=H, depth=depth, pix_fmt=PIX[depth], frames=N)
        fl = list(flags)
        for p in params:
            k, v = p.split("=", 1)
            if k == "speed_percent" and "@" not in v:
                sp = float(v)
                if sp > 100:
                    fl.append("reads_past_end")
                if sp < 0:
                    fl.append("reads_before_start")
        case = dict(
            id=cid, effect="retime", label=f"{group}-{label}", params=params, depth=depth, width=W, height=H,
            source=src, input=fname, reference=f"{cid}.{PIX[depth]}", kind="temporal", frames=N, fps=fps, group=group,
        )
        if self.quality:
            case["quality"] = self.quality
        if stand_in:
            case["stand_in"] = stand_in
        if fl:
            case["flags"] = sorted(set(fl))
        self.cases.append(case)


ROT = [50, 25, 75, 33.3333, 62.5, 40, 50, 20]


def P(speed="ROT", **kw):
    out = [f"speed_percent={speed}"]
    for k, v in kw.items():
        out.append(f"{k}={v}")
    return out


def build_grid(b):
    # A: sizes x depths at the default settings and a 50% slowdown
    sizes = [(1, 1), (1, 6), (7, 1), (2, 2), (3, 3), (5, 4), (7, 9), (16, 16), (17, 13), (23, 37), (32, 24), (33, 31), (48, 32), (64, 48)]
    rot = ["shift", "disc", "alpha", "noise"]
    for i, (W, H) in enumerate(sizes):
        for di, d in enumerate((8, 16, 32)):
            b.add("size", "default", rot[(i + di) % 4], W, H, d, P(), fps="30000/1001" if d == 16 else "24")
    # B: vectors x frame_interp
    for v in range(1, 6):
        for it in (1, 2, 3):
            b.add("vec", f"v{v}-i{it}", "disc", 40, 30, 8, P(vectors=v, frame_interp=it))
    for v in (2, 3, 4, 5):
        for it in (1, 3):
            b.add("vec", f"v{v}-i{it}", "shift", 24, 20, 16, P(vectors=v, frame_interp=it))
    for v in (2, 3, 4, 5):
        for it in (1, 2, 3):
            b.add("vec", f"v{v}-i{it}", "shift", 24, 20, 32, P(vectors=v, frame_interp=it))
    # C: smart_blend
    for v in (3, 5):
        for it in (1, 2, 3):
            b.add("sb", f"v{v}-i{it}", "disc", 40, 30, 8, P(vectors=v, frame_interp=it, smart_blend=1))
    for v in (3, 5):
        for it in (1, 2, 3):
            b.add("sb", f"v{v}-i{it}", "shift", 24, 20, 16, P(vectors=v, frame_interp=it, smart_blend=1))
    for v in (4, 5):
        for it in (1, 2, 3):
            b.add("sb", f"v{v}-i{it}", "shift", 24, 20, 32, P(vectors=v, frame_interp=it, smart_blend=1))
    for d in (8, 16, 32):
        b.add("sb", "alpha-i2", "alpha", 32, 24, d, P(smart_blend=1))
    for it in (1, 2, 3):
        b.add("sb", f"big-i{it}", "bigshift", 32, 24, 8, P(frame_interp=it, smart_blend=1))
    b.add("sb", "big-i2", "bigshift", 32, 24, 32, P(smart_blend=1))
    b.add("sb", "disc-25", "disc", 40, 30, 16, P(25, smart_blend=1))
    b.add("sb", "tiny", "shift", 5, 4, 8, P(smart_blend=1))
    b.add("sb", "tiny", "shift", 3, 3, 32, P(smart_blend=1))
    # D: video_360
    for sb in (0, 1):
        for it in (1, 2, 3):
            b.add("v360", f"sb{sb}-i{it}", "disc", 40, 30, 8, P(video_360=1, smart_blend=sb, frame_interp=it))
    for sb in (0, 1):
        b.add("v360", f"sb{sb}-i2", "shift", 24, 20, 16, P(video_360=1, smart_blend=sb))
    for sb in (0, 1):
        for it in (1, 2, 3):
            b.add("v360", f"sb{sb}-i{it}", "shift", 24, 20, 32, P(video_360=1, smart_blend=sb, frame_interp=it))
    for sb in (0, 1):
        b.add("v360", f"sb{sb}-v3", "disc", 40, 30, 8, P(video_360=1, smart_blend=sb, vectors=3))
    b.add("v360", "sb1-alpha", "alpha", 32, 24, 8, P(video_360=1, smart_blend=1))
    b.add("v360", "sb1-adisc", "adisc", 32, 24, 32, P(video_360=1, smart_blend=1))
    # E: blur_compensation
    for bl in ("0", "-0", "0.01", "0.5", "1", "5", "10"):
        b.add("blur", f"s50-b{bl}", "disc", 40, 30, 8, P(50, blur_compensation=bl))
    for bl in ("1", "10"):
        b.add("blur", f"s25-b{bl}", "disc", 40, 30, 8, P(25, blur_compensation=bl))
    for sp in (150, 200, 400):
        for bl in ("0.5", "5", "10"):
            b.add("blur", f"s{sp}-b{bl}", "disc", 40, 30, 8, P(sp, blur_compensation=bl), N=6)
    b.add("blur", "s100-b5", "disc", 40, 30, 8, P(100, blur_compensation=5))
    for d in (16, 32):
        b.add("blur", "s50-b3", "shift", 24, 20, d, P(50, blur_compensation=3))
        b.add("blur", "s200-b3", "shift", 24, 20, d, P(200, blur_compensation=3), N=6)
    b.add("blur", "s200-b2-i1", "disc", 40, 30, 8, P(200, blur_compensation=2, frame_interp=1), N=6)
    b.add("blur", "s200-b2-i3", "disc", 40, 30, 8, P(200, blur_compensation=2, frame_interp=3), N=6)
    b.add("blur", "s200-b2-v3", "disc", 40, 30, 8, P(200, blur_compensation=2, vectors=3), N=6)
    b.add("blur", "s200-b2-sb", "disc", 40, 30, 8, P(200, blur_compensation=2, smart_blend=1), N=6)
    b.add("blur", "s50-b2-sb", "disc", 40, 30, 8, P(50, blur_compensation=2, smart_blend=1))
    b.add("blur", "s50-b2-v360", "disc", 40, 30, 8, P(50, blur_compensation=2, video_360=1))
    b.add("blur", "s50-b2-v1", "disc", 40, 30, 8, P(50, blur_compensation=2, vectors=1))
    b.add("blur", "s50-b10-alpha", "alpha", 32, 24, 8, P(50, blur_compensation=10))
    for sp in (33.3333, 70, 120, 175):
        for bl in ("0.7", "3.3"):
            b.add("blur", f"s{sp}-b{bl}", "disc", 40, 30, 8, P(sp, blur_compensation=bl), N=6)
    for d in (16, 32):
        b.add("blur", "s400-b10", "shift", 24, 20, d, P(400, blur_compensation=10), N=6)
    for d in (16, 32):
        for sp in (33.3333, 70, 120):
            for bl in ("0.7", "3.3"):
                b.add("blur", f"s{sp}-b{bl}", "shift", 24, 20, d, P(sp, blur_compensation=bl), N=6)
    b.add("blur", "s300-b10", "disc", 40, 30, 8, P(300, blur_compensation=10), N=6)
    # F: sensitivity
    for s in ("0", "-0", "0.01", "1", "10", "25", "33.33333", "50", "69.99", "70", "99.99", "100"):
        b.add("sens", f"s{s}", "disc", 64, 48, 8, P(sensitivity=s))
    for d in (16, 32):
        for s in ("0", "25", "100"):
            b.add("sens", f"s{s}", "disc", 48, 32, d, P(sensitivity=s))
    for v in (3, 4):
        b.add("sens", f"s25-v{v}", "disc", 64, 48, 8, P(sensitivity=25, vectors=v))
    b.add("sens", "s10-noise", "noise", 32, 24, 8, P(sensitivity=10))
    for s_ in ("0.3", "7.77", "15.5", "42.42", "87.654"):
        b.add("sens", f"s{s_}", "disc", 40, 30, 8, P(sensitivity=s_))
    b.add("sens", "s0-v1", "disc", 40, 30, 8, P(sensitivity=0, vectors=1))
    b.add("sens", "s100-spin", "spin", 48, 32, 8, P(sensitivity=100))
    # G: prep
    for p in (1, 2, 3):
        b.add("prep", f"p{p}", "disc", 40, 30, 8, P(prep=p))
    for d in (16, 32):
        for p in (2, 3):
            b.add("prep", f"p{p}", "shift", 24, 20, d, P(prep=p))
    for p in (2, 3):
        b.add("prep", f"p{p}-noise", "noise", 32, 24, 8, P(prep=p))
        b.add("prep", f"p{p}-s25", "disc", 40, 30, 8, P(prep=p, sensitivity=25))
        b.add("prep", f"p{p}-alpha", "alpha", 32, 24, 8, P(prep=p))
        b.add("prep", f"p{p}-v360", "disc", 40, 30, 8, P(prep=p, video_360=1))
    b.add("prep", "p3-big", "noise", 96, 72, 32, P(prep=3))
    b.add("prep", "p3-big", "noise", 64, 48, 16, P(prep=3))
    b.add("prep", "p3-big", "disc", 96, 72, 32, P(prep=3))
    b.add("prep", "p3-hdr", "hdr", 24, 20, 32, P(prep=3))
    b.add("prep", "p2-hdr", "hdr", 24, 20, 32, P(prep=2))
    # I: speeds and one keyed ramp
    for sp in (75, 33.3333, 10):
        b.add("speed", f"s{sp}", "disc", 40, 30, 8, P(sp), N=6)
    for sp in (150, 200):
        b.add("speed", f"s{sp}", "disc", 40, 30, 8, P(sp), N=6)
    b.add("speed", "s-50", "disc", 40, 30, 8, P(-50), N=6)
    b.add("speed", "s0", "disc", 40, 30, 8, P(0), N=4)
    b.add("speed", "s-0", "disc", 40, 30, 8, P("-0"), N=4)
    b.add("speed", "ramp", "disc", 40, 30, 8, ["speed_percent=100@0;40@5"], N=6, stand_in="keyframe_interpolation")
    b.add("speed", "ramp-ease", "disc", 40, 30, 8, ["speed_percent=60@0e;30@5e"], N=6, stand_in="keyframe_interpolation")
    b.add("speed", "ramp-d32", "shift", 24, 20, 32, ["speed_percent=100@0;40@5"], N=6, stand_in="keyframe_interpolation")
    for d in (16, 32):
        b.add("speed", "fps2997", "shift", 24, 20, d, P(50), fps="30000/1001")
    b.add("speed", "s50-fps60", "disc", 40, 30, 8, P(50), fps="60")
    # J: alpha
    for d in (8, 16, 32):
        for sb in (0, 1):
            b.add("alpha", f"adisc-sb{sb}", "adisc", 32, 24, d, P(smart_blend=sb))
            b.add("alpha", f"ramp-sb{sb}", "alpha", 32, 24, d, P(smart_blend=sb))
    for d in (8, 16, 32):
        b.add("alpha", "adisc-v3-i3", "adisc", 32, 24, d, P(vectors=3, frame_interp=3, smart_blend=1))
    # K: motion types
    for src in ("static", "flat", "subpix", "shift", "bigshift", "noise", "spin", "zoom"):
        for v in (3, 5):
            b.add("motion", f"v{v}", src, 40, 30, 8, P(vectors=v))
    for src in ("spin", "zoom", "bigshift"):
        b.add("motion", "v5", src, 32, 24, 16, P())
        b.add("motion", "v5", src, 32, 24, 32, P())
    b.add("motion", "hdr", "hdr", 32, 24, 32, P())
    b.add("motion", "hdr-sb", "hdr", 32, 24, 32, P(smart_blend=1))
    # L: 32 bpc special values
    for name, _ in SPECIALS:
        b.add("spec", f"{name}", f"spec-{name}", 16, 16, 32, P())
        b.add("spec", f"{name}-sb", f"spec-{name}", 16, 16, 32, P(smart_blend=1))
    for name in ("zero", "negzero", "mindenorm", "one", "above1", "fltmax", "posinf", "nan"):
        b.add("spec", f"alpha-{name}", f"specA-{name}", 16, 16, 32, P())
    for it in (1, 2, 3):
        for sb in (0, 1):
            b.add("spec", f"all-i{it}-sb{sb}", "spec-all", 24, 24, 32, P(frame_interp=it, smart_blend=sb))
    for name in ("nan", "posinf", "fltmax"):
        b.add("spec", f"{name}-blur", f"spec-{name}", 16, 16, 32, P(200, blur_compensation=5), N=6)
        b.add("spec", f"{name}-prep3", f"spec-{name}", 16, 16, 32, P(prep=3))
    for name in ("nan", "negzero"):
        b.add("spec", f"{name}-v360", f"spec-{name}", 16, 16, 32, P(video_360=1, smart_blend=1))
    for hi in ("lo", "hi"):
        for d in (8, 16):
            for sb in (0, 1):
                b.add("spec", f"ext-{hi}-sb{sb}", f"ext-{hi}", 16, 16, d, P(smart_blend=sb))


def build_draft(b):
    for v in (2, 3, 4, 5):
        b.add("draft", f"v{v}", "disc", 64, 48, 8, P(vectors=v))
    for d in (16, 32):
        for v in (3, 5):
            b.add("draft", f"v{v}", "shift", 32, 24, d, P(vectors=v))
    b.add("draft", "tiny", "shift", 7, 9, 8, P())
    b.add("draft", "sq", "disc", 16, 16, 8, P())
    b.add("draft", "odd", "disc", 33, 31, 8, P())
    b.add("draft", "sb-blur", "disc", 40, 30, 8, P(smart_blend=1, blur_compensation=1))
    b.add("draft", "i3", "disc", 40, 30, 8, P(frame_interp=3))
    b.add("draft", "v360", "disc", 40, 30, 8, P(video_360=1))
    b.add("draft", "prep2", "disc", 40, 30, 32, P(prep=2))


OUT_OF_CONTRACT = [
    ("vectors=0", "popup below 1", ["vectors=0"]),
    ("vectors=6", "popup above 5", ["vectors=6"]),
    ("frame_interp=0", "popup below 1", ["frame_interp=0"]),
    ("frame_interp=4", "popup above 3", ["frame_interp=4"]),
    ("smart_blend=2", "checkbox not 0 or 1", ["smart_blend=2"]),
    ("video_360=2", "checkbox not 0 or 1", ["video_360=2"]),
    ("prep=0", "popup below 1", ["prep=0"]),
    ("prep=4", "popup above 3", ["prep=4"]),
    ("sensitivity=100.01", "float above 100", ["sensitivity=100.01"]),
    ("sensitivity=-0.01", "float below 0", ["sensitivity=-0.01"]),
    ("sensitivity=nan", "float NaN", ["sensitivity=nan"]),
    ("blur_compensation=10.01", "float above 10", ["blur_compensation=10.01"]),
    ("blur_compensation=-0.01", "float below 0", ["blur_compensation=-0.01"]),
    ("speed_percent=10001", "float above 10000", ["speed_percent=10001"]),
    ("speed_percent=-10001", "float below -10000", ["speed_percent=-10001"]),
    ("vectors=Best", "popup given by name", ["vectors=Best"]),
    ("speed_percent=50@", "key without a frame", ["speed_percent=50@"]),
]


def write_set(b, out):
    os.makedirs(os.path.join(out, "inputs"), exist_ok=True)
    for f, meta in b.inputs.items():
        path = os.path.join(out, f)
        W, H, d, N, src = meta["width"], meta["height"], meta["depth"], meta["frames"], meta["source"]
        with open(path, "wb") as fh:
            for t in range(N):
                fh.write(frame_raw(src, W, H, t, d))
    doc = dict(format=1, ffmpeg="none (raw frames written by gen.py)", inputs=sorted(b.inputs.values(), key=lambda m: m["file"]), cases=b.cases)
    with open(os.path.join(out, "cases.json"), "w") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g = Builder()
    build_grid(g)
    dr = Builder(quality="draft")
    build_draft(dr)
    write_set(g, os.path.join(a.out, "grid"))
    write_set(dr, os.path.join(a.out, "draft"))
    ooc = [dict(id=f"ooc.{i}", label=l, why=w, params=p, width=16, height=16, depth=8) for i, (l, w, p) in enumerate(OUT_OF_CONTRACT)]
    with open(os.path.join(a.out, "out_of_contract.json"), "w") as fh:
        json.dump(ooc, fh, indent=1)
        fh.write("\n")
    print(f"grid: {len(g.cases)} cases, {len(g.inputs)} inputs; draft: {len(dr.cases)} cases, {len(dr.inputs)} inputs; out of contract candidates: {len(ooc)}")


if __name__ == "__main__":
    main()
