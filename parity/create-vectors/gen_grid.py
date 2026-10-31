#!/usr/bin/env python3
"""Test grid generator for the create-vectors parity measurement.

Writes an fx-parity style test set: DIR/cases.json (format 1) and DIR/inputs/*.raw frames in the
raw formats of the README (rgba, rgba64le, gbrapf32le). Besides it writes
  DIR/case_meta.json          group, tags and stand-in label per case id
  DIR/out_of_contract.json    candidates the plugin or the rewrite refuses (never counted as pass/fail)
  DIR/params_used.json        every rewrite param id the grid sets, with the distinct values
Everything is deterministic: the same script gives the same bytes.

usage: gen_grid.py --out DIR
"""
import argparse
import hashlib
import json
import math
import os
import struct
import sys

FORMAT = 1
PIXFMT = {8: "rgba", 16: "rgba64le", 32: "gbrapf32le"}
EFFECT = "create-vectors"


class Bits(int):
    """A 32 bit float given by its exact bit pattern."""


SPECIALS = [
    ("zero", Bits(0x00000000)),
    ("negzero", Bits(0x80000000)),
    ("denorm_min", Bits(0x00000001)),
    ("denorm_max", Bits(0x007FFFFF)),
    ("flt_min", Bits(0x00800000)),
    ("one", Bits(0x3F800000)),
    ("above1", Bits(0x40200000)),
    ("flt_max", Bits(0x7F7FFFFF)),
    ("inf", Bits(0x7F800000)),
    ("ninf", Bits(0xFF800000)),
    ("nan", Bits(0x7FC00000)),
    ("nnan", Bits(0xFFC00000)),
]


def clamp(v, lo=0.0, hi=1.0):
    return lo if v < lo else hi if v > hi else v


# ---------------------------------------------------------------- textures and sources

def tex(x, y, c):
    v = (0.5 + 0.22 * math.sin(0.55 * x + 0.35 * y + 1.3 * c)
         + 0.18 * math.sin(-0.3 * x + 0.7 * y + 0.5 * c * c + 0.2)
         + 0.10 * math.sin(1.1 * x - 0.4 * y + 2.1 + c))
    return clamp(v)


def rgb_tex(x, y, ph=0.0):
    return tuple(tex(x + ph, y - ph * 0.5, c) for c in range(3))


def hash01(x, y, seed):
    n = (x * 73856093) ^ (y * 19349663) ^ (seed * 83492791)
    n &= 0xFFFFFFFF
    n = (n ^ (n >> 13)) * 1274126177 & 0xFFFFFFFF
    n ^= n >> 16
    return (n & 0xFFFF) / 65535.0


def px_shift(vx, vy, ph=0.0):
    def f(x, y, t, w, h):
        r, g, b = rgb_tex(x - vx * t, y - vy * t, ph)
        return (r, g, b, 1.0)
    return f


def disc_mask(x, y, cx, cy, r):
    d = math.hypot(x - cx, y - cy)
    return clamp(r - d + 0.5)


def px_disc(x, y, t, w, h):
    cx, cy, r = w * 0.35 + 1.5 * t, h * 0.5 + 0.5 * t, max(1.5, min(w, h) * 0.28)
    m = disc_mask(x, y, cx, cy, r)
    bg = rgb_tex(x, y, 0.0)
    fg = rgb_tex(x - 1.5 * t + 3.0, y - 0.5 * t, 2.0)
    bg = tuple(0.25 + 0.3 * (c - 0.5) for c in bg)
    return tuple(bg[i] * (1 - m) + fg[i] * m for i in range(3)) + (1.0,)


def multi_discs(w, h):
    r = max(1.5, min(w, h) * 0.18)
    return [(w * 0.25, h * 0.30, 2.0, 1.0, r), (w * 0.70, h * 0.35, -1.5, 1.0, r), (w * 0.50, h * 0.75, 0.0, -1.5, r)]


def px_multi(x, y, t, w, h):
    out = rgb_tex(x - 1.0 * t, y, 0.0)
    out = tuple(0.25 + 0.3 * (c - 0.5) for c in out)
    for k, (cx, cy, vx, vy, r) in reversed(list(enumerate(multi_discs(w, h)))):
        m = disc_mask(x, y, cx + vx * t, cy + vy * t, r)
        fg = rgb_tex(x - vx * t + 2.0 * k, y - vy * t, 1.0 + 1.7 * k)
        out = tuple(out[i] * (1 - m) + fg[i] * m for i in range(3))
    return out + (1.0,)


def px_matte_multi(k):
    def f(x, y, t, w, h):
        cx, cy, vx, vy, r = multi_discs(w, h)[k]
        m = disc_mask(x, y, cx + vx * t, cy + vy * t, r)
        return (m, 0.1 + 0.5 * m, 1.0 - m, 1.0)
    return f


def px_matte_multi_alpha(k):
    def f(x, y, t, w, h):
        cx, cy, vx, vy, r = multi_discs(w, h)[k]
        m = disc_mask(x, y, cx + vx * t, cy + vy * t, r)
        return (0.5, 0.5, 0.5, m)
    return f


def px_matte_disc(x, y, t, w, h):
    cx, cy, r = w * 0.35 + 1.5 * t, h * 0.5 + 0.5 * t, max(1.5, min(w, h) * 0.28)
    m = disc_mask(x, y, cx, cy, r)
    return (m, 0.1 + 0.5 * m, 1.0 - m, 1.0)


def px_matte_disc_alpha(x, y, t, w, h):
    cx, cy, r = w * 0.35 + 1.5 * t, h * 0.5 + 0.5 * t, max(1.5, min(w, h) * 0.28)
    return (0.5, 0.5, 0.5, disc_mask(x, y, cx, cy, r))


def px_zoom(x, y, t, w, h):
    s = 1.05 ** t
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r, g, b = rgb_tex(cx + (x - cx) / s, cy + (y - cy) / s)
    return (r, g, b, 1.0)


def px_rot(x, y, t, w, h):
    a = 0.06 * t
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    dx, dy = x - cx, y - cy
    r, g, b = rgb_tex(cx + dx * math.cos(a) + dy * math.sin(a), cy - dx * math.sin(a) + dy * math.cos(a))
    return (r, g, b, 1.0)


def px_box(x, y, t, w, h):
    bw, bh = max(2, w // 3), max(2, h // 3)
    x0, y0 = 1 + 2 * t, 1 + t
    inside = x0 <= x < x0 + bw and y0 <= y < y0 + bh
    return (0.9, 0.6, 0.2, 1.0) if inside else (0.1, 0.1, 0.12, 1.0)


def px_flat(x, y, t, w, h):
    return (0.4, 0.5, 0.6, 1.0)


def px_static(x, y, t, w, h):
    r, g, b = rgb_tex(x, y)
    return (r, g, b, 1.0)


def px_noise(x, y, t, w, h):
    xx = (x - t) % max(w, 1)
    return (hash01(xx, y, 1), hash01(xx, y, 2), hash01(xx, y, 3), 1.0)


def px_flicker(x, y, t, w, h):
    r, g, b = rgb_tex(x, y)
    k = 1.0 + 0.1 * t
    return (clamp(r * k), clamp(g * k), clamp(b * k), 1.0)


def px_alphaslide(x, y, t, w, h):
    r, g, b = rgb_tex(x - 1.0 * t, y - 0.5 * t)
    a = clamp(0.25 + 0.75 * ((x - t) / float(max(w, 1))))
    return (r, g, b, a)


def px_hole(x, y, t, w, h):
    r, g, b = rgb_tex(x - 1.0 * t, y - 0.5 * t)
    return (r, g, b, 0.0 if x - t < w // 3 else 1.0)


def px_black(x, y, t, w, h):
    return (0.0, 0.0, 0.0, 1.0)


def px_white(x, y, t, w, h):
    return (1.0, 1.0, 1.0, 1.0)


def px_checker(x, y, t, w, h):
    v = 1.0 if (((x - t) // 2) + (y // 2)) % 2 == 0 else 0.0
    return (v, v, v, 1.0)


def px_imp(x, y, t, w, h):
    v = 1.0 if (x, y) == (min(1 + t, w - 1), h // 2) else 0.0
    return (v, v, v, 1.0)


def px_hdr(x, y, t, w, h):
    r, g, b = rgb_tex(x - 1.0 * t, y - 0.5 * t)
    return (r * 4.0 - 1.0, g * 4.0 - 1.0, b * 4.0 - 1.0, 1.0)


def px_spec_uniform(s):
    def f(x, y, t, w, h):
        return (s, s, s, s)
    return f


def px_spec_patch(s, alpha_only=False):
    def f(x, y, t, w, h):
        r, g, b = rgb_tex(x - 1.0 * t, y - 0.5 * t)
        n = 3 if min(w, h) >= 6 else 2
        inside = (1 + t) <= x < (1 + t + n) and 1 <= y < 1 + n
        if inside and alpha_only:
            return (r, g, b, s)
        if inside:
            return (s, s, s, 1.0)
        return (r, g, b, 1.0)
    return f


def px_spec_mix(x, y, t, w, h):
    k = (x + t) + y * w
    return tuple(SPECIALS[(k + 3 * c) % len(SPECIALS)][1] for c in range(4))


SOURCES = {
    "shift": px_shift(1.5, -0.75),
    "shiftB": px_shift(-1.0, 1.0, 5.0),
    "bigshift": px_shift(5.0, 3.0),
    "sub": px_shift(0.25, 0.0),
    "disc": px_disc,
    "multi": px_multi,
    "zoom": px_zoom,
    "rot": px_rot,
    "box": px_box,
    "flat": px_flat,
    "static": px_static,
    "noise": px_noise,
    "flicker": px_flicker,
    "alphaslide": px_alphaslide,
    "hole": px_hole,
    "black": px_black,
    "white": px_white,
    "checker": px_checker,
    "imp": px_imp,
    "hdr": px_hdr,
    "matte_disc": px_matte_disc,
    "matte_disc_a": px_matte_disc_alpha,
    "specmix": px_spec_mix,
}
for _k in range(3):
    SOURCES["matte%d" % (_k + 1)] = px_matte_multi(_k)
    SOURCES["matte%d_a" % (_k + 1)] = px_matte_multi_alpha(_k)
for _n, _s in SPECIALS:
    SOURCES["spu_" + _n] = px_spec_uniform(_s)
    SOURCES["spp_" + _n] = px_spec_patch(_s)
    SOURCES["spa_" + _n] = px_spec_patch(_s, True)


# ---------------------------------------------------------------- raw frame encoding

def f2bits(v):
    if isinstance(v, Bits):
        return int(v)
    return struct.unpack("<I", struct.pack("<f", v))[0]


def encode_frame(pixels, w, h, depth):
    if depth == 8:
        return bytes(int(clamp(v) * 255.0 + 0.5) for p in pixels for v in p)
    if depth == 16:
        vals = [int(clamp(v) * 65535.0 + 0.5) for p in pixels for v in p]
        return struct.pack("<%dH" % len(vals), *vals)
    out = bytearray()
    for ch in (1, 2, 0, 3):  # planar G, B, R, A
        vals = [f2bits(p[ch]) for p in pixels]
        out += struct.pack("<%dI" % len(vals), *vals)
    return bytes(out)


def render_clip(name, w, h, depth, frames):
    fn = SOURCES[name]
    out = bytearray()
    for t in range(frames):
        pixels = [fn(x, y, t, w, h) for y in range(h) for x in range(w)]
        out += encode_frame(pixels, w, h, depth)
    return bytes(out)


# ---------------------------------------------------------------- the grid

class Grid:
    def __init__(self):
        self.cases = []
        self.meta = {}
        self.inputs = {}  # file -> InputFile dict
        self.seen = set()

    def input_file(self, name, w, h, depth, frames):
        f = "inputs/%s-%dx%d.d%d.f%d.%s" % (name, w, h, depth, frames, PIXFMT[depth])
        if f not in self.inputs:
            self.inputs[f] = {"file": f, "source": name, "width": w, "height": h, "depth": depth,
                              "pix_fmt": PIXFMT[depth], "frames": frames}
        return f

    def add(self, group, label, src, w, h, depth, params=(), frames=3, fps="24", layers=None, tags=(), standin=None):
        params = list(params)
        layers = dict(layers or {})
        key = json.dumps([label, src, w, h, depth, params, frames, fps, sorted(layers.items())])
        digest = hashlib.sha1(key.encode()).hexdigest()[:8]
        cid = "%s.%s.%s-%dx%d.d%d.%s" % (EFFECT, label, src, w, h, depth, digest)
        if cid in self.seen:
            raise SystemExit("duplicate case " + cid)
        self.seen.add(cid)
        case = {
            "id": cid, "effect": EFFECT, "label": label, "params": params, "depth": depth,
            "width": w, "height": h, "source": src,
            "input": self.input_file(src, w, h, depth, frames),
            "reference": "%s.%s" % (cid, PIXFMT[depth]),
            "kind": "temporal", "frames": frames, "fps": fps,
        }
        if layers:
            case["layers"] = {lid: self.input_file(lname, w, h, depth, frames) for lid, lname in sorted(layers.items())}
        self.cases.append(case)
        self.meta[cid] = {"group": group, "tags": list(tags), "standin": standin, "layer_sources": layers}
        return cid


def build(g):
    D = (8, 16, 32)
    # A: defaults across sizes (tiny and odd included) and depths
    sizes_a = [(1, 1), (1, 4), (4, 1), (2, 2), (2, 3), (3, 2), (3, 3), (5, 5), (6, 4), (7, 5), (8, 6), (16, 12), (17, 13), (31, 19)]
    for (w, h) in sizes_a:
        for d in D:
            g.add("defaults-sizes", "default", "shift", w, h, d, tags=["core"] if (w, h) in [(1, 1), (2, 3), (16, 12), (17, 13)] else [])
    # B: defaults across clip contents
    for src in ["bigshift", "sub", "box", "disc", "zoom", "rot", "flat", "static", "noise", "flicker", "alphaslide", "hole"]:
        for (w, h) in [(7, 5), (16, 12)]:
            for d in D:
                g.add("defaults-sources", "default", src, w, h, d, tags=["core"] if (src, w) in [("disc", 16), ("alphaslide", 16), ("box", 16)] else [])
    # C: display and Debug views
    for d in D:
        g.add("display", "display-source", "shift", 8, 6, d, ["display=1"])
    for d in D:
        for tf in (1, 2):
            for m in range(1, 6):
                g.add("debug", "debug-m%d-tf%d" % (m, tf), "disc", 16, 12, d,
                      ["display=3", "debug_view_method=%d" % m, "track_frame=%d" % tf],
                      tags=["core"] if (d == 8 and tf == 1) else [])
    # D: vector resolution and track frame
    for d in D:
        for vr, tf in [(2, 1), (2, 2), (1, 2)]:
            g.add("resolution-track", "vr%d-tf%d" % (vr, tf), "shift", 17, 13, d, ["vector_resolution=%d" % vr, "track_frame=%d" % tf])
    # E: max displace
    for md in ["1", "2", "3.5", "16", "100", "8192"]:
        for d in D:
            g.add("max-displace", "md-%s" % md, "bigshift", 16, 12, d, ["max_displace=%s" % md], tags=["core"] if md in ("1", "16") else [])
    for md in ("1", "2"):
        for d in D:
            g.add("max-displace", "md-%s-next" % md, "bigshift", 16, 12, d, ["max_displace=%s" % md, "track_frame=2"])
    for tf in (1, 2):
        g.add("max-displace", "md-1-box-tf%d" % tf, "box", 16, 12, 8, ["max_displace=1", "track_frame=%d" % tf])
        g.add("max-displace", "md-1-box-tf%d" % tf, "box", 16, 12, 16, ["max_displace=1", "track_frame=%d" % tf])
    for md in ("3", "5", "6", "7", "9", "12"):
        g.add("max-displace", "md-%s-f32" % md, "bigshift", 16, 12, 32, ["max_displace=%s" % md])
    for src in ("zoom", "rot", "sub"):
        for md in ("2", "3"):
            g.add("max-displace", "md-%s-f32" % md, src, 16, 12, 32, ["max_displace=%s" % md])
    for md in ("1", "2", "4"):
        g.add("max-displace", "md-%s-fg-f32" % md, "multi", 24, 16, 32, ["max_displace=%s" % md],
              layers={"fg1_matte": "matte1", "fg2_matte": "matte2"})
        g.add("max-displace", "md-%s-fg3-f32" % md, "multi", 24, 16, 32, ["max_displace=%s" % md, "layers_output=1"],
              layers={"fg1_matte": "matte1", "fg2_matte": "matte2", "fg3_matte": "matte3"})
        g.add("max-displace", "md-%s-fg1-f32" % md, "disc", 16, 12, 32, ["max_displace=%s" % md],
              layers={"fg1_matte": "matte_disc"})
    # F: prep
    for pv in (2, 3):
        for d in D:
            g.add("prep", "prep%d" % pv, "shift", 16, 12, d, ["prep=%d" % pv])
    g.add("prep", "prep2", "alphaslide", 16, 12, 8, ["prep=2"])
    g.add("prep", "prep3", "alphaslide", 16, 12, 8, ["prep=3"])
    g.add("prep", "prep2", "noise", 8, 6, 32, ["prep=2"])
    g.add("prep", "prep3", "noise", 8, 6, 32, ["prep=3"])
    # G: track RGB+A
    for d in D:
        g.add("track-rgba", "rgba", "alphaslide", 16, 12, d, ["track_rgb_a=1"])
        g.add("track-rgba", "rgba", "shift", 16, 12, d, ["track_rgb_a=1"])
    g.add("track-rgba", "rgba-prep2", "alphaslide", 16, 12, 8, ["track_rgb_a=1", "prep=2"])
    # H: sensitivity
    for sv in ["0", "-0", "0.01", "1", "25", "100"]:
        for d in D:
            g.add("sensitivity", "sens-%s" % sv, "shift", 16, 12, d, ["main_bg_sensitivity=%s" % sv])
    # I: tiny frames with other displays
    g.add("tiny", "tiny-debug", "shift", 1, 4, 8, ["display=3", "debug_view_method=3"])
    g.add("tiny", "tiny-debug", "shift", 2, 2, 8, ["display=3", "debug_view_method=1"])
    g.add("tiny", "tiny-debug", "shift", 3, 3, 16, ["display=3", "debug_view_method=4"])
    g.add("tiny", "tiny-source", "shift", 1, 1, 32, ["display=1"])
    g.add("tiny", "tiny-track-next", "shift", 4, 1, 8, ["track_frame=2"])
    g.add("tiny", "tiny-vr2-md1", "shift", 2, 3, 8, ["vector_resolution=2", "max_displace=1"])

    # J: tracking points and Draw Geom
    P5 = ["pt1_use=5", "pt1_pos=6.5,4.5"]
    for dg in range(1, 7):
        g.add("points", "draw-geom%d" % dg, "shift", 16, 12, 8, ["display=1", "draw_geom=%d" % dg] + P5 + ["pt2_use=2", "pt2_pos=3,8", "pt3_use=3", "pt3_pos=12,2"])
    for d in D:
        g.add("points", "debug-geom", "disc", 16, 12, d, ["display=3", "debug_view_method=2", "draw_geom=2"] + P5)
    for u in (2, 3, 4, 5):
        g.add("points", "pt-use%d-vec" % u, "disc", 16, 12, 8, ["pt1_use=%d" % u, "pt1_pos=5.25,6.75"])
    g.add("points", "pt-use5-vec", "disc", 16, 12, 16, ["pt1_use=5", "pt1_pos=5.25,6.75"])
    g.add("points", "pt-use5-vec", "disc", 16, 12, 32, ["pt1_use=5", "pt1_pos=5.25,6.75"])
    twelve = []
    for k in range(12):
        twelve += ["pt%d_use=%d" % (k + 1, 2 + (k % 4)), "pt%d_pos=%s,%s" % (k + 1, 1 + (k * 5) % 14, 1 + (k * 3) % 10),
                   "pt%d_color=%d,%d,%d" % (k + 1, (k * 40) % 256, (255 - k * 20) % 256, (k * 97) % 256)]
    for d in D:
        g.add("points", "twelve-src", "shift", 16, 12, d, ["display=1", "draw_geom=2"] + twelve)
    g.add("points", "twelve-vec", "shift", 16, 12, 8, twelve)
    g.add("points", "pt-border", "shift", 16, 12, 8, ["display=1", "pt1_use=5", "pt1_pos=0,0", "pt2_use=5", "pt2_pos=15,11", "pt3_use=5", "pt3_pos=15.5,11.5"])
    g.add("points", "pt-outside", "shift", 16, 12, 8, ["display=1", "pt1_use=5", "pt1_pos=100,100", "pt2_use=5", "pt2_pos=-3,4"])
    g.add("points", "pt-default-pos", "shift", 16, 12, 8, ["pt1_use=5"])
    g.add("points", "track-off-pts", "disc", 16, 12, 8, ["main_bg_track=0", "pt1_use=5", "pt1_pos=4,4", "pt2_use=5", "pt2_pos=12,8"])
    g.add("points", "track-off-nopt", "disc", 16, 12, 8, ["main_bg_track=0"])
    g.add("points", "track-off-nopt", "disc", 16, 12, 16, ["main_bg_track=0"])
    g.add("points", "track-off-nopt", "disc", 16, 12, 32, ["main_bg_track=0"])
    g.add("points", "track-off-debug", "disc", 16, 12, 8, ["main_bg_track=0", "display=3", "debug_view_method=2", "pt1_use=5", "pt1_pos=8,6"])
    g.add("points", "pt-color", "shift", 16, 12, 8, ["display=1", "draw_geom=2", "pt1_use=5", "pt1_pos=8,6", "pt1_color=0,0,0"])
    g.add("points", "pt-color", "shift", 16, 12, 32, ["display=1", "draw_geom=2", "pt1_use=5", "pt1_pos=8,6", "pt1_color=255,255,255"])

    # K: FG matte layers
    F1 = {"fg1_matte": "matte_disc"}
    for lo in range(1, 6):
        for d in D:
            g.add("fg-layers", "fg1-lo%d" % lo, "disc", 16, 12, d, ["layers_output=%d" % lo], layers=F1, tags=["core"] if (d == 8 and lo in (1, 2, 5)) else [])
    F2 = {"fg1_matte": "matte1", "fg2_matte": "matte2"}
    F3 = {"fg1_matte": "matte1", "fg2_matte": "matte2", "fg3_matte": "matte3"}
    for lo in range(1, 6):
        g.add("fg-layers", "fg2-lo%d" % lo, "multi", 24, 16, 8, ["layers_output=%d" % lo], layers=F2)
        g.add("fg-layers", "fg3-lo%d" % lo, "multi", 24, 16, 8, ["layers_output=%d" % lo], layers=F3, tags=["core"] if lo in (1, 3) else [])
    for d in (16, 32):
        g.add("fg-layers", "fg2-lo1", "multi", 24, 16, d, [], layers=F2)
        g.add("fg-layers", "fg3-lo1", "multi", 24, 16, d, [], layers=F3)
        g.add("fg-layers", "fg3-lo3", "multi", 24, 16, d, ["layers_output=3"], layers=F3)
    for ch in range(1, 5):
        lay = {"fg1_matte": "matte_disc_a" if ch == 4 else "matte_disc"}
        g.add("fg-matte-chan", "fg1-chan%d" % ch, "disc", 16, 12, 8, ["fg1_matte_chan=%d" % ch, "layers_output=2"], layers=lay)
    g.add("fg-matte-chan", "fg1-chan2", "disc", 16, 12, 16, ["fg1_matte_chan=2", "layers_output=2"], layers=F1)
    g.add("fg-matte-chan", "fg1-chan3", "disc", 16, 12, 32, ["fg1_matte_chan=3", "layers_output=2"], layers=F1)
    g.add("fg-matte-chan", "fg1-chan4", "disc", 16, 12, 32, ["fg1_matte_chan=4"], layers={"fg1_matte": "matte_disc_a"})
    g.add("fg-matte-chan", "fg3-chan4", "multi", 24, 16, 8, ["fg1_matte_chan=4", "fg2_matte_chan=2", "fg3_matte_chan=3"],
          layers={"fg1_matte": "matte1_a", "fg2_matte": "matte2", "fg3_matte": "matte3"})
    for sv in ("0", "100", "30"):
        g.add("fg-controls", "fg1-sens%s" % sv, "disc", 16, 12, 8, ["fg1_sensitivity=%s" % sv, "layers_output=2"], layers=F1)
    g.add("fg-controls", "fg1-track0", "disc", 16, 12, 8, ["fg1_track=0", "layers_output=2"], layers=F1)
    g.add("fg-controls", "fg1-track0-pts", "disc", 16, 12, 8, ["fg1_track=0", "pt1_use=2", "pt1_pos=5,6"], layers=F1)
    g.add("fg-controls", "main-track0-fg1", "disc", 16, 12, 8, ["main_bg_track=0", "pt1_use=5", "pt1_pos=12,3"], layers=F1)
    g.add("fg-controls", "main-sens-fg1", "disc", 16, 12, 8, ["main_bg_sensitivity=20", "layers_output=5"], layers=F1)
    for sh in ("0.5", "2.5", "10"):
        g.add("fg-controls", "fg1-shrink%s" % sh, "multi", 24, 16, 8, ["fg1_inv_matte_shrink=%s" % sh], layers=F2)
    g.add("fg-controls", "fg2-shrink-0", "multi", 24, 16, 8, ["fg2_inv_matte_shrink=-0", "fg1_inv_matte_shrink=-0"], layers=F2)
    g.add("fg-controls", "fg3-sens-track", "multi", 24, 16, 8, ["fg1_sensitivity=100", "fg2_sensitivity=10", "fg3_sensitivity=55.5", "fg2_track=0"], layers=F3)
    g.add("fg-controls", "fg3-track0-shrink", "multi", 24, 16, 8, ["fg3_track=0", "fg3_inv_matte_shrink=3", "pt1_use=4", "pt1_pos=12,12"], layers=F3)
    g.add("fg-controls", "fg3-shrink10", "multi", 24, 16, 16, ["fg3_inv_matte_shrink=10", "fg2_inv_matte_shrink=4.5"], layers=F3)
    g.add("fg-debug", "fg1-debug", "disc", 16, 12, 8, ["display=3", "debug_view_method=2"], layers=F1)
    g.add("fg-debug", "fg1-debug-warp", "disc", 16, 12, 16, ["display=3", "debug_view_method=1", "track_frame=2"], layers=F1)
    g.add("fg-debug", "fg3-debug-geom", "multi", 24, 16, 8, ["display=3", "debug_view_method=3", "draw_geom=2", "pt1_use=3", "pt1_pos=10,8"], layers=F3)
    g.add("fg-debug", "fg1-source-geom", "disc", 16, 12, 8, ["display=1", "draw_geom=3", "pt1_use=2", "pt1_pos=5,6", "pt2_use=5", "pt2_pos=10,6"], layers=F1)
    g.add("fg-debug", "fg2-geom-fg2", "multi", 24, 16, 8, ["display=1", "draw_geom=4", "pt1_use=3", "pt1_pos=8,8", "pt2_use=2", "pt2_pos=16,8"], layers=F2)
    g.add("fg-tiny", "fg1-tiny", "shift", 3, 3, 8, [], layers={"fg1_matte": "matte_disc"})
    g.add("fg-tiny", "fg1-tiny", "shift", 1, 4, 8, [], layers={"fg1_matte": "matte_disc"})
    g.add("fg-tiny", "fg1-tiny-dbg", "shift", 2, 2, 16, ["display=3"], layers={"fg1_matte": "matte_disc"})
    g.add("fg-points", "fg3-points", "multi", 24, 16, 8, ["pt1_use=2", "pt1_pos=6,5", "pt2_use=3", "pt2_pos=17,6", "pt3_use=4", "pt3_pos=12,12", "pt4_use=5", "pt4_pos=2,14"], layers=F3)

    g.add("fg-points", "fg2-pt-main", "multi", 24, 16, 8, ["pt1_use=5", "pt1_pos=10,8", "pt2_use=4", "pt2_pos=5,5"], layers=F2)
    g.add("fg-points", "fg1-pt-main", "disc", 16, 12, 8, ["pt1_use=5", "pt1_pos=12,3", "pt2_use=4", "pt2_pos=3,3"], layers=F1)
    g.add("fg-points", "fg2-pt-main-lo5", "multi", 24, 16, 16, ["pt1_use=5", "pt1_pos=10,8", "layers_output=5"], layers=F2)

    # L: Alt Src
    for d in D:
        g.add("alt-src", "alt-src", "static", 16, 12, d, [], layers={"alt_source_a": "shiftB"})
    g.add("alt-src", "alt-src-fg", "static", 16, 12, 8, [], layers={"alt_source_a": "disc", "fg1_matte": "matte_disc"})
    g.add("alt-src", "alt-src-display", "static", 16, 12, 8, ["display=1", "draw_geom=2", "pt1_use=5", "pt1_pos=7,7"], layers={"alt_source_a": "shiftB"})

    # M: rates and clip lengths
    for fps in ("30000/1001", "25", "60", "29.97", "48000/1001"):
        g.add("fps", "fps-%s" % fps.replace("/", "_"), "shift", 16, 12, 8, [], fps=fps)
    g.add("fps", "fps-30000_1001-debug", "disc", 16, 12, 16, ["display=3", "debug_view_method=1"], fps="30000/1001")
    g.add("fps", "frames5", "shift", 12, 9, 8, [], frames=5)
    g.add("fps", "frames5-next", "shift", 12, 9, 8, ["track_frame=2"], frames=5)
    g.add("fps", "frames2", "shift", 12, 9, 8, [], frames=2)
    g.add("fps", "frames1", "shift", 12, 9, 8, [], frames=1)
    g.add("fps", "frames1-next", "shift", 12, 9, 32, ["track_frame=2"], frames=1)

    # N: float specials (32 bpc) and extreme integer values
    for name, _ in SPECIALS:
        g.add("float-specials", "uniform-%s" % name, "spu_" + name, 8, 6, 32, [], tags=["core"] if name in ("negzero", "nan", "inf") else [])
        g.add("float-specials", "patch-%s" % name, "spp_" + name, 8, 6, 32, [])
        g.add("float-specials", "alphapatch-%s" % name, "spa_" + name, 8, 6, 32, [])
    g.add("float-specials", "mix", "specmix", 6, 4, 32, [])
    g.add("float-specials", "mix-dbg", "specmix", 6, 4, 32, ["display=3", "debug_view_method=2"])
    g.add("float-specials", "mix-dbg", "specmix", 6, 4, 32, ["display=3", "debug_view_method=3"])
    g.add("float-specials", "mix-dbg", "specmix", 6, 4, 32, ["display=3", "debug_view_method=1"])
    g.add("float-specials", "mix-dbg", "specmix", 6, 4, 32, ["display=3", "debug_view_method=4"])
    g.add("float-specials", "patch-nan-dbg", "spp_nan", 8, 6, 32, ["display=3", "debug_view_method=2"])
    g.add("float-specials", "patch-inf-dbg", "spp_inf", 8, 6, 32, ["display=3", "debug_view_method=3"])
    g.add("float-specials", "patch-fltmax-dbg", "spp_flt_max", 8, 6, 32, ["display=3", "debug_view_method=1"])
    g.add("float-specials", "patch-nan-prep", "spp_nan", 8, 6, 32, ["prep=2"])
    g.add("float-specials", "patch-inf-prep", "spp_inf", 8, 6, 32, ["prep=3"])
    g.add("float-specials", "patch-ninf-rgba", "spp_ninf", 8, 6, 32, ["track_rgb_a=1"])
    g.add("float-specials", "alphapatch-nan-rgba", "spa_nan", 8, 6, 32, ["track_rgb_a=1"])
    g.add("float-specials", "patch-nan-md1", "spp_nan", 8, 6, 32, ["max_displace=1"])
    g.add("float-specials", "patch-denorm-md1", "spp_denorm_min", 8, 6, 32, ["max_displace=1", "display=3", "debug_view_method=3"])
    g.add("float-specials", "mix-source", "specmix", 6, 4, 32, ["display=1"])
    for src in ("black", "white", "checker", "imp"):
        for d in (8, 16):
            g.add("extremes", "extreme", src, 8, 6, d, [])
    g.add("extremes", "extreme-dbg", "checker", 8, 6, 8, ["display=3", "debug_view_method=3"])
    g.add("extremes", "extreme-dbg", "checker", 8, 6, 16, ["display=3", "debug_view_method=4"])
    g.add("extremes", "extreme-dbg", "white", 8, 6, 8, ["display=3", "debug_view_method=3"])
    g.add("extremes", "extreme-dbg", "black", 8, 6, 16, ["display=3", "debug_view_method=1"])
    # float values above 1 and below 0
    g.add("hdr", "hdr", "hdr", 16, 12, 32, [])
    g.add("hdr", "hdr-prep2", "hdr", 16, 12, 32, ["prep=2"])
    g.add("hdr", "hdr-prep3", "hdr", 16, 12, 32, ["prep=3"])
    g.add("hdr", "hdr-debug", "hdr", 16, 12, 32, ["display=3", "debug_view_method=2"])
    g.add("hdr", "hdr-debug-diff", "hdr", 16, 12, 32, ["display=3", "debug_view_method=3"])
    g.add("hdr", "hdr-next", "hdr", 16, 12, 32, ["track_frame=2", "max_displace=4"])

    # O: larger real motion
    for d in D:
        g.add("larger", "default", "shift", 48, 32, d)
    g.add("larger", "default", "disc", 64, 48, 8)
    g.add("larger", "default", "disc", 64, 48, 32, ["vector_resolution=2"])
    g.add("larger", "fg3-all", "multi", 48, 32, 8, [], layers=F3)
    g.add("larger", "debug-edge", "multi", 48, 32, 16, ["display=3", "debug_view_method=5"])

    # P: combined settings
    mins = ["display=1", "vector_resolution=1", "debug_view_method=1", "draw_geom=1", "max_displace=1", "track_frame=1", "prep=1",
            "track_rgb_a=0", "layers_output=1", "main_bg_sensitivity=0", "main_bg_track=0"]
    maxs = ["display=3", "vector_resolution=2", "debug_view_method=5", "draw_geom=6", "max_displace=8192", "track_frame=2", "prep=3",
            "track_rgb_a=1", "layers_output=5", "main_bg_sensitivity=100", "main_bg_track=1"]
    mid = ["display=2", "vector_resolution=2", "max_displace=24.5", "track_frame=2", "prep=2", "track_rgb_a=1", "main_bg_sensitivity=45.25",
           "fg1_sensitivity=80", "fg2_sensitivity=12", "fg1_matte_chan=2", "fg1_inv_matte_shrink=1.5"]
    for d in D:
        g.add("combined", "all-min", "disc", 16, 12, d, mins, layers=F1)
        g.add("combined", "all-max", "disc", 16, 12, d, maxs, layers=F1)
        g.add("combined", "all-mid", "multi", 24, 16, d, mid, layers=F2)

    # Q: keyframed values (host side interpolation, stand-in on both sides)
    SI = "keyframe interpolation of the host: %s"
    g.add("standin-keys", "keyed-md", "bigshift", 16, 12, 8, ["max_displace=16@0;64@2"], standin=SI % "linear")
    g.add("standin-keys", "keyed-sens", "shift", 16, 12, 16, ["main_bg_sensitivity=10@0;90@2e"], standin=SI % "eased end key")
    g.add("standin-keys", "keyed-pt-vec", "disc", 16, 12, 8, ["pt1_use=5", "pt1_pos=3,3@0;9,6@2"], standin=SI % "point, linear")
    g.add("standin-keys", "keyed-pt-vec", "disc", 16, 12, 32, ["pt1_use=5", "pt1_pos=3,3@0;9,6@2"], standin=SI % "point, linear")
    g.add("standin-keys", "keyed-pt-geom", "shift", 16, 12, 8, ["display=1", "draw_geom=2", "pt1_use=5", "pt1_pos=3,3@0;9,6@2"], standin=SI % "point, linear")


def out_of_contract(g):
    """Candidates the plugin refuses or that fall outside the registered ranges."""
    base = lambda label, params, why, **kw: dict(label=label, params=params, why=why, **kw)
    items = [
        base("md-zero", ["max_displace=0"], "max_displace below its valid minimum 1"),
        base("md-low", ["max_displace=0.99"], "max_displace below its valid minimum 1"),
        base("md-neg", ["max_displace=-1"], "max_displace below its valid minimum 1"),
        base("md-high", ["max_displace=8192.5"], "max_displace above its valid maximum 8192"),
        base("sens-neg", ["main_bg_sensitivity=-0.01"], "sensitivity below 0"),
        base("sens-high", ["main_bg_sensitivity=100.01"], "sensitivity above 100"),
        base("shrink-high", ["fg1_inv_matte_shrink=10.5"], "shrink above 10"),
        base("shrink-neg", ["fg1_inv_matte_shrink=-1"], "shrink below 0"),
        base("display-0", ["display=0"], "popup below its first choice"),
        base("display-4", ["display=4"], "popup above its last choice"),
        base("dbg-6", ["debug_view_method=6"], "popup above its last choice"),
        base("tf-3", ["track_frame=3"], "popup above its last choice"),
        base("lo-6", ["layers_output=6"], "popup above its last choice"),
        base("geom-7", ["draw_geom=7"], "popup above its last choice"),
        base("pt-use-6", ["pt1_use=6"], "popup above its last choice"),
        base("pt-color-256", ["pt1_color=256,0,0"], "color channel above 255"),
        base("prep-4", ["prep=4"], "popup above its last choice"),
        base("unknown-param", ["no_such_param=1"], "unknown param id"),
        base("matte-size", [], "FG matte layer of another size than the output (plugin shows an error fill)", layer_size=(8, 6),
             layers={"fg1_matte": "matte_disc"}),
        base("alt-size", [], "Alt Src layer of another size than the output (plugin shows an error fill)", layer_size=(8, 6),
             layers={"alt_source_a": "shiftB"}),
    ]
    out = []
    for it in items:
        w, h, d = 16, 12, 8
        layers = it.get("layers")
        lsize = it.get("layer_size")
        cid = "%s.ooc-%s.shift-%dx%d.d%d" % (EFFECT, it["label"], w, h, d)
        case = {"id": cid, "effect": EFFECT, "label": "ooc-" + it["label"], "params": it["params"], "depth": d,
                "width": w, "height": h, "source": "shift", "input": g.input_file("shift", w, h, d, 3),
                "reference": "%s.%s" % (cid, PIXFMT[d]), "kind": "temporal", "frames": 3, "fps": "24", "why": it["why"]}
        if layers:
            lw, lh = lsize or (w, h)
            case["layers"] = {lid: g.input_file(lname, lw, lh, d, 3) for lid, lname in layers.items()}
        out.append(case)
    return out


def params_used(cases):
    used = {}
    for c in cases:
        for p in c["params"]:
            k, _, v = p.partition("=")
            used.setdefault(k, set()).add(v)
    layers = {}
    for c in cases:
        for lid in c.get("layers", {}):
            layers[lid] = layers.get(lid, 0) + 1
    return {"params": {k: sorted(v) for k, v in sorted(used.items())}, "layers": layers,
            "cli_flags": ["--fps", "--frames", "--layer", "--in", "--width", "--height", "--depth"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = args.out
    os.makedirs(os.path.join(out, "inputs"), exist_ok=True)
    g = Grid()
    build(g)
    ooc = out_of_contract(g)
    # write the inputs every case and candidate names
    for f, info in sorted(g.inputs.items()):
        path = os.path.join(out, f)
        if os.path.exists(path):
            continue
        data = render_clip(info["source"], info["width"], info["height"], info["depth"], info["frames"])
        want = info["width"] * info["height"] * {8: 4, 16: 8, 32: 16}[info["depth"]] * info["frames"]
        assert len(data) == want, (f, len(data), want)
        with open(path, "wb") as fh:
            fh.write(data)
    ts = {"format": FORMAT, "ffmpeg": "none (inputs written by gen_grid.py)",
          "inputs": [g.inputs[k] for k in sorted(g.inputs)], "cases": g.cases}
    with open(os.path.join(out, "cases.json"), "w") as fh:
        json.dump(ts, fh, indent=1)
    with open(os.path.join(out, "case_meta.json"), "w") as fh:
        json.dump(g.meta, fh, indent=1, sort_keys=True)
    with open(os.path.join(out, "out_of_contract.json"), "w") as fh:
        json.dump(ooc, fh, indent=1)
    with open(os.path.join(out, "params_used.json"), "w") as fh:
        json.dump(params_used(g.cases), fh, indent=1)
    standin = sum(1 for m in g.meta.values() if m["standin"])
    by_group = {}
    for m in g.meta.values():
        by_group[m["group"]] = by_group.get(m["group"], 0) + 1
    by_depth = {}
    for c in g.cases:
        by_depth[c["depth"]] = by_depth.get(c["depth"], 0) + 1
    print("wrote %s/cases.json: %d cases (%d with stand-in keyframes), %d out-of-contract candidates, %d inputs" %
          (out, len(g.cases), standin, len(ooc), len(g.inputs)))
    print("by depth:", dict(sorted(by_depth.items())))
    print("by group:", dict(sorted(by_group.items())))


if __name__ == "__main__":
    sys.exit(main())
