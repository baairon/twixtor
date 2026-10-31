#!/usr/bin/env python3
"""Test grid generator for the vectors-in parity measurement.

Writes an fx-parity style test set (format 1) into DIR:
  cases.json            the in-contract candidates; inputs are raw frames in the README formats
  inputs/               raw clips (rgba, rgba64le, gbrapf32le), shared between cases by content
  case_meta.json        group, flags and stand-in label per case id
  out_of_contract.json  parameter values outside the registered ranges (checked with --check)
  params_used.json      every rewrite param id and layer id the grid uses, with the distinct values
Pixel data is closed form or a fixed hash, so the same script gives the same bytes.

Vector layers are hand made raw vector images (no other effect is needed): red horizontal and
green vertical, centred and scaled by the max displace the image is said to be written with
(`md_img`), blue 0, alpha opaque except where a case wants holes or partial alpha. The decode in
the rewrite is 8 bpc u = (R*2/254 - 1)*md, 16 bpc with 65534, 32 bpc (R*2 - 1)*md, so the encoder
here writes R = (u/md + 1)*127, *32767 and /2 respectively.

Every case has `frames` output frames and an input clip of exactly `frames` frames (fx-parity
needs that). Layers are `frames` long unless a case says otherwise.

usage: gen_grid.py --out DIR
"""
import argparse
import hashlib
import json
import math
import os
import struct

FORMAT = 1
PIXFMT = {8: "rgba", 16: "rgba64le", 32: "gbrapf32le"}
EFFECT = "vectors-in"
VX, VY = 1.5, -0.75  # motion of the textured footage, pixels per source frame


class Raw32(int):
    """A 32 bit float given by its exact bit pattern."""


def unbits(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


SPECIALS = [
    ("zero", Raw32(0x00000000)),
    ("negzero", Raw32(0x80000000)),
    ("denorm_min", Raw32(0x00000001)),
    ("denorm_max", Raw32(0x007FFFFF)),
    ("flt_min", Raw32(0x00800000)),
    ("one", Raw32(0x3F800000)),
    ("above1", Raw32(0x40200000)),
    ("flt_max", Raw32(0x7F7FFFFF)),
    ("inf", Raw32(0x7F800000)),
    ("ninf", Raw32(0xFF800000)),
    ("nan", Raw32(0x7FC00000)),
    ("nnan", Raw32(0xFFC00000)),
]


def clamp(v, lo=0.0, hi=1.0):
    return lo if v < lo else hi if v > hi else v


def unit(v):
    """A pixel value as a float for the integer depths: Raw32 decoded, NaN to 0, clamped."""
    if isinstance(v, Raw32):
        v = unbits(int(v))
    if v != v:
        return 0.0
    return clamp(v)


def f2bits(v):
    if isinstance(v, Raw32):
        return int(v)
    return struct.unpack("<I", struct.pack("<f", v))[0]


def encode_frame(pixels, depth):
    if depth == 8:
        return bytes(int(unit(v) * 255.0 + 0.5) for p in pixels for v in p)
    if depth == 16:
        vals = [int(unit(v) * 65535.0 + 0.5) for p in pixels for v in p]
        return struct.pack("<%dH" % len(vals), *vals)
    out = bytearray()
    for ch in (1, 2, 0, 3):  # planar G, B, R, A
        out += struct.pack("<%dI" % len(pixels), *[f2bits(p[ch]) for p in pixels])
    return bytes(out)


# ---------------------------------------------------------------- footage

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


def disc_mask(x, y, cx, cy, r):
    return clamp(r - math.hypot(x - cx, y - cy) + 0.5)


def f_tex(x, y, t, w, h):
    r, g, b = rgb_tex(x - VX * t, y - VY * t)
    return (r, g, b, 1.0)


def f_disc(x, y, t, w, h):
    m = disc_mask(x, y, w * 0.35 + 1.5 * t, h * 0.5 + 0.5 * t, max(1.5, min(w, h) * 0.28))
    bg = tuple(0.25 + 0.3 * (c - 0.5) for c in rgb_tex(x, y))
    fg = rgb_tex(x - 1.5 * t + 3.0, y - 0.5 * t, 2.0)
    return tuple(bg[i] * (1 - m) + fg[i] * m for i in range(3)) + (1.0,)


def multi_discs(w, h):
    r = max(1.5, min(w, h) * 0.18)
    return [(w * 0.25, h * 0.30, 2.0, 1.0, r), (w * 0.70, h * 0.35, -1.5, 1.0, r), (w * 0.50, h * 0.75, 0.0, -1.5, r)]


def f_multi(x, y, t, w, h):
    out = tuple(0.25 + 0.3 * (c - 0.5) for c in rgb_tex(x - t, y))
    for k, (cx, cy, vx, vy, r) in reversed(list(enumerate(multi_discs(w, h)))):
        m = disc_mask(x, y, cx + vx * t, cy + vy * t, r)
        fg = rgb_tex(x - vx * t + 2.0 * k, y - vy * t, 1.0 + 1.7 * k)
        out = tuple(out[i] * (1 - m) + fg[i] * m for i in range(3))
    return out + (1.0,)


def f_noise(x, y, t, w, h):
    xx = (x - t) % max(w, 1)
    return (hash01(xx, y, 1), hash01(xx, y, 2), hash01(xx, y, 3), 1.0)


def f_alpha(x, y, t, w, h):
    r, g, b = rgb_tex(x - t, y - 0.5 * t)
    return (r, g, b, clamp(0.25 + 0.75 * ((x - t) / float(max(w, 1)))))


def f_flat(x, y, t, w, h):
    return (0.4, 0.5, 0.6, 1.0)


def f_decoy(x, y, t, w, h):
    return (0.9, 0.1, 0.3 + 0.05 * t, 1.0)


def f_checker(x, y, t, w, h):
    v = 1.0 if (((x - t) // 2) + (y // 2)) % 2 == 0 else 0.0
    return (v, v, v, 1.0)


def f_edge(x, y, t, w, h):
    inside = (2 + 2 * t) <= x < (2 + 2 * t) + max(2, w // 3)
    return (0.9, 0.6, 0.2, 1.0) if inside else (0.1, 0.1, 0.12, 1.0)


def f_hdr(x, y, t, w, h):
    r, g, b = rgb_tex(x - t, y - 0.5 * t)
    return (r * 4.0 - 1.0, g * 4.0 - 1.0, b * 4.0 - 1.0, 1.0)


def f_special(s):
    def f(x, y, t, w, h):
        px = list(f_tex(x, y, t, w, h))
        k = (x + 2 * y + t) % 4
        if k == 0:
            px[(x + y) % 4] = s
        return tuple(px)
    return f


def f_special_alpha(s):
    def f(x, y, t, w, h):
        px = list(f_tex(x, y, t, w, h))
        if (x + 2 * y + t) % 4 == 0:
            px[3] = s
        return tuple(px)
    return f


FOOT = {"tex": f_tex, "disc": f_disc, "multi": f_multi, "noise": f_noise, "alpha": f_alpha, "flat": f_flat,
        "decoy": f_decoy, "checker": f_checker, "edge": f_edge, "hdr": f_hdr}
for _n, _s in SPECIALS:
    FOOT["sp_" + _n] = f_special(_s)
    FOOT["spa_" + _n] = f_special_alpha(_s)


# ---------------------------------------------------------------- FG mattes

def matte_disc(k):
    def f(x, y, t, w, h):
        cx, cy, vx, vy, r = multi_discs(w, h)[k]
        m = disc_mask(x, y, cx + vx * t, cy + vy * t, r)
        return (m, 0.1 + 0.5 * m, 1.0 - m, 0.3 + 0.7 * m)
    return f


MATTE = {"disc": matte_disc}


# ---------------------------------------------------------------- vector fields
# A field returns (dx, dy, alpha) in pixels, or (dx, dy, alpha, override) where override maps a
# channel index (0 R, 1 G, 3 A) to a raw value that replaces the encoded one (32 bpc only).
# `ctx` is the case: it supplies md_img for the fields that depend on it.

def v_zero(x, y, t, w, h, c):
    return (0.0, 0.0, 1.0)


def v_const(vx, vy):
    def f(x, y, t, w, h, c):
        return (vx, vy, 1.0)
    return f


def v_tvar(x, y, t, w, h, c):
    return (0.4 + 0.3 * t, -0.2 - 0.15 * t, 1.0)


def v_grad(x, y, t, w, h, c):
    return (((x + 0.5) / w - 0.5) * 6.0 * (1 + 0.2 * t), ((y + 0.5) / h - 0.5) * -4.0, 1.0)


def v_rot(x, y, t, w, h, c):
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    a = 0.12 * (1 + 0.1 * t)
    return (-(y - cy) * a, (x - cx) * a, 1.0)


def v_stripes(x, y, t, w, h, c):
    return ((2.0 if x % 2 else -2.0) * (1 + 0.1 * t), (1.0 if y % 3 == 0 else -0.5), 1.0)


def v_holes(x, y, t, w, h, c):
    return (2.0, 1.0, 0.0 if (x + t) % 5 < 2 else 1.0)


def v_allhole(x, y, t, w, h, c):
    return (2.0, 1.0, 0.0)


def v_partial(x, y, t, w, h, c):
    return (2.0, 1.0, 0.5 if y < max(1, h // 2) else 1.0)


def v_noise(x, y, t, w, h, c):
    return ((hash01(x, y, 11 + t) - 0.5) * 6.0, (hash01(x, y, 29 + t) - 0.5) * 6.0, 1.0)


def v_gholes(x, y, t, w, h, c):
    g = v_grad(x, y, t, w, h, c)
    return (g[0], g[1], 0.0 if (x + t) % 5 < 2 else 1.0)


def v_gpartial(x, y, t, w, h, c):
    g = v_grad(x, y, t, w, h, c)
    return (g[0], g[1], 0.5 if y < max(1, h // 2) else (0.25 if x % 3 == 0 else 1.0))


def v_galpha(x, y, t, w, h, c):
    g = v_grad(x, y, t, w, h, c)
    return (g[0], g[1], hash01(x, y, 5 + t))


def v_gall(x, y, t, w, h, c):
    g = v_grad(x, y, t, w, h, c)
    return (g[0], g[1], 0.7)


def v_px20(x, y, t, w, h, c):
    return (20.0, -12.0, 1.0)


def v_large(x, y, t, w, h, c):
    return (1.7 * w, -1.3 * h, 1.0)


def v_big(x, y, t, w, h, c):
    return (300.0, -200.0, 1.0)


def v_clipped(x, y, t, w, h, c):
    return (2.0 * c["md_img"], -2.0 * c["md_img"], 1.0)


def v_edgeval(x, y, t, w, h, c):
    return (c["md_img"], -c["md_img"], 1.0)


def v_special(ch, s):
    def f(x, y, t, w, h, c):
        base = (VX, VY, 1.0)
        if (x + y) % 5 == 0:
            return base + ({ch: s},)
        return base
    return f


VEC = {
    "zero": v_zero, "sub": v_const(0.3, -0.2), "sub2": v_const(0.5, 0.5), "small": v_const(1.0, 0.5),
    "coh": v_const(VX, VY), "med": v_const(3.0, 2.0), "tvar": v_tvar, "grad": v_grad, "rot": v_rot,
    "stripes": v_stripes, "holes": v_holes, "allhole": v_allhole, "partial": v_partial, "noisef": v_noise,
    "gholes": v_gholes, "gpartial": v_gpartial, "galpha": v_galpha, "gall": v_gall, "px20": v_px20,
    "large": v_large, "big": v_big, "clipped": v_clipped, "edgeval": v_edgeval,
}
for _n, _s in SPECIALS:
    VEC["spR_" + _n] = v_special(0, _s)
    VEC["spG_" + _n] = v_special(1, _s)
    VEC["spA_" + _n] = v_special(3, _s)
    VEC["spB_" + _n] = v_special(2, _s)

SYMMETRIC = {"coh", "zero"}  # prev is exactly the negated next of the frame before


def qvec(u, md, depth):
    """The unit value (0 to 1) of the encoded red/green channel for the displacement u."""
    if md == 0:
        r = 0.0
    else:
        r = u / md + 1.0
    if depth == 8:
        i = int(math.floor(r * 127.0 + 0.5))
        return clamp(i, 0, 254) / 255.0
    if depth == 16:
        i = int(math.floor(r * 32767.0 + 0.5))
        return clamp(i, 0, 65534) / 65535.0
    return r / 2.0


def vec_pixel(res, md, depth):
    dx, dy, a = res[0], res[1], res[2]
    px = [qvec(dx, md, depth), qvec(dy, md, depth), 0.0, a]
    if len(res) > 3:
        for ch, val in res[3].items():
            px[ch] = val
    return tuple(px)


def make_vec_fn(name, which, c):
    """The per-frame field of Vectors Next (at frame t: the motion from t to t+1) or Vectors Prev
    (at frame t: the motion back from t to t-1, the negated next of the frame before, with a
    different gain on each axis unless the field is symmetric)."""
    base = VEC[name]
    if which == "next":
        return lambda x, y, t, w, h: base(x, y, t, w, h, c)
    gx, gy = (1.0, 1.0) if name in SYMMETRIC else (0.85, 1.1)

    def prev(x, y, t, w, h):
        r = base(x, y, max(t - 1, 0), w, h, c)
        out = (-gx * r[0], -gy * r[1], r[2])
        return out + (r[3],) if len(r) > 3 else out
    return prev


# ---------------------------------------------------------------- the grid

class Grid:
    def __init__(self, out):
        self.out = out
        self.cases = []
        self.meta = {}
        self.inputs = {}
        self.seen = set()
        self.used = {}
        self.layers_used = set()
        os.makedirs(os.path.join(out, "inputs"), exist_ok=True)

    def clip_file(self, tag, w, h, depth, frames, pixel_fn):
        data = bytearray()
        for t in range(frames):
            data += encode_frame([pixel_fn(x, y, t, w, h) for y in range(h) for x in range(w)], depth)
        digest = hashlib.sha1(bytes(data)).hexdigest()[:8]
        f = "inputs/%s-%dx%d.d%d.f%d.%s.%s" % (tag, w, h, depth, frames, digest, PIXFMT[depth])
        path = os.path.join(self.out, f)
        if f not in self.inputs:
            with open(path, "wb") as fh:
                fh.write(bytes(data))
            self.inputs[f] = {"file": f, "source": tag, "width": w, "height": h, "depth": depth,
                              "pix_fmt": PIXFMT[depth], "frames": frames}
        return f

    def add(self, group, label, w=16, h=12, depth=8, fps="30", n=4, p=None, base=True, foot="tex",
            vec="coh", prev=True, nxt=True, src=None, src_len=None, fg=0, md_img=32.0,
            prev_len=None, next_len=None, flags=(), stand_in=None, mat_len=None):
        params = {}
        if base:
            params["vectors_max_displace"] = "32"
            params["speed_percent"] = "50"
        params.update(p or {})
        plist = ["%s=%s" % kv for kv in params.items()]
        key = json.dumps([label, w, h, depth, fps, n, plist, foot, vec, prev, nxt, src, src_len, fg, md_img,
                          prev_len, next_len, mat_len])
        digest = hashlib.sha1(key.encode()).hexdigest()[:6]
        cid = "%s.%s.%dx%d.d%d.%s" % (EFFECT, label, w, h, depth, digest)
        if cid in self.seen:
            raise SystemExit("duplicate case " + cid)
        self.seen.add(cid)
        ctx = {"md_img": md_img}
        layers = {}
        if src:
            layers["source"] = self.clip_file("src-" + src, w, h, depth, src_len or n, FOOT[src])
            input_file = self.clip_file("foot-decoy", w, h, depth, n, f_decoy)
        else:
            input_file = self.clip_file("foot-" + foot, w, h, depth, n, FOOT[foot])
        for lid, on, ln, which in (("vectors_prev", prev, prev_len, "prev"), ("vectors_next", nxt, next_len, "next")):
            if on:
                fn = make_vec_fn(vec, which, ctx)
                layers[lid] = self.clip_file("%s-%s-md%g" % (which, vec, md_img), w, h, depth, ln or n,
                                             lambda x, y, t, ww, hh, fn=fn: vec_pixel(fn(x, y, t, ww, hh), md_img, depth))
        for k in range(fg):
            layers["fg%d_matte" % (k + 1)] = self.clip_file("matte%d" % (k + 1), w, h, depth, mat_len or n, MATTE["disc"](k))
        case = {
            "id": cid, "effect": EFFECT, "label": label, "params": plist, "depth": depth, "width": w, "height": h,
            "source": foot if not src else "decoy", "input": input_file, "reference": "%s.%s" % (cid, PIXFMT[depth]),
            "kind": "temporal", "frames": n, "fps": fps, "layers": layers,
        }
        self.cases.append(case)
        self.meta[cid] = {"group": group, "flags": sorted(flags), "stand_in": stand_in, "vec": vec, "foot": foot,
                          "md_img": md_img}
        for k, v in params.items():
            self.used.setdefault(k, set()).add(v)
        self.layers_used.update(layers)
        self.layers_used.add("input")

    def write(self):
        doc = {"format": FORMAT, "ffmpeg": "none (raw frames written by gen_grid.py)",
               "inputs": [self.inputs[k] for k in sorted(self.inputs)], "cases": self.cases}
        with open(os.path.join(self.out, "cases.json"), "w") as fh:
            json.dump(doc, fh, indent=1)
        with open(os.path.join(self.out, "case_meta.json"), "w") as fh:
            json.dump(self.meta, fh, indent=1, sort_keys=True)
        used = {"params": {k: sorted(v) for k, v in sorted(self.used.items())}, "layers": sorted(self.layers_used),
                "fps": sorted({c["fps"] for c in self.cases}), "depths": sorted({c["depth"] for c in self.cases}),
                "sizes": sorted({"%dx%d" % (c["width"], c["height"]) for c in self.cases})}
        with open(os.path.join(self.out, "params_used.json"), "w") as fh:
            json.dump(used, fh, indent=1)


def build(g):
    D = (8, 16, 32)
    A = g.add

    # ---- size: defaults of the effect (warping Forward, Blend) over sizes and depths, tiny ones flagged
    sizes = [(1, 1), (2, 2), (3, 2), (5, 4), (7, 6), (9, 7), (16, 12), (17, 13), (24, 16), (37, 23), (48, 32)]
    for (w, h) in sizes:
        tiny = w <= 5 and h <= 5
        for d in D:
            A("size", "size-%dx%d" % (w, h), w, h, d, flags=("tiny",) if tiny else ())
    for (w, h) in sizes[1:4]:
        for d in D:
            A("tiny", "tiny-inv-%dx%d" % (w, h), w, h, d, p={"warping": "1"}, flags=("tiny",))
    # the pure defaults of the effect (max displace 2048, speed 100, 30 fps clip model) at 24 fps
    for d in D:
        A("defaults", "pure-defaults", 64, 48, d, fps="24", n=4, base=False, vec="px20", md_img=2048.0)

    # ---- warping x frame_interp x depth
    for d in D:
        for wp in (1, 2):
            for fi in (1, 2, 3):
                if (d, wp, fi) in ((8, 2, 2), (16, 2, 2), (32, 2, 2)):
                    continue
                A("warp", "warp%d-interp%d" % (wp, fi), 16, 12, d, p={"warping": str(wp), "frame_interp": str(fi)})
    for wp in (1, 2):
        for fi in (1, 2, 3):
            A("warp", "warp%d-interp%d-irregular" % (wp, fi), 17, 13, 8, fps="24",
              p={"warping": str(wp), "frame_interp": str(fi), "speed_percent": "100"})

    # ---- speed (retime time controls)
    for fps in ("30", "24", "30000/1001", "25"):
        A("speed", "speed50-fps%s" % fps.replace("/", "_"), fps=fps)
    for fps in ("24", "25", "30000/1001", "30"):
        A("speed", "speed100-fps%s" % fps.replace("/", "_"), fps=fps, p={"speed_percent": "100"})
    for s in ("25", "75", "33.3333", "12.5", "99.99", "0", "0.5", "-0", "1", "66.6667"):
        A("speed", "speed%s" % s.replace(".", "p"), p={"speed_percent": s})
    for s in ("120", "150", "200", "300", "1000", "10000"):
        far = s in ("1000", "10000")
        A("speed", "fast%s" % s, n=4, p={"speed_percent": s}, src="tex", src_len=60 if far else 24,
          flags=("long_source", "reads_past_end") if far else ("long_source",))
    for s in ("-50", "-100", "-200", "-10000"):
        A("speed", "back%s" % s.replace("-", "m"), n=4, p={"speed_percent": s}, flags=("reads_before_start",))

    # ---- frame number mode
    for v in ("0", "0.5", "1.25", "2.75", "3", "-1", "7", "200000", "-200000", "-0"):
        fl = ()
        if v in ("-1", "-200000"):
            fl = ("reads_before_start",)
        if v in ("7", "200000"):
            fl = ("reads_past_end",)
        A("framenum", "fnum%s" % v.replace(".", "p").replace("-", "m"), p={"time_remap_mode": "2", "frame_num": v}, flags=fl)
    for d in (16, 32):
        for v in ("0.5", "2.25"):
            A("framenum", "fnum%s" % v.replace(".", "p"), 16, 12, d, p={"time_remap_mode": "2", "frame_num": v})
    for k in ("0@0;3@3", "0@0;2.5@3e", "3@0;0@3", "0@0;1@1;3@3"):
        A("framenum", "fnum-keys-" + k.replace("@", "at").replace(";", "_").replace(".", "p"),
          p={"time_remap_mode": "2", "frame_num": k}, stand_in="keyframe_interpolation")

    # ---- frame times that land just below a layer frame boundary (dur * n = 99.999..., 199.999...)
    for fps, fnv in (("10", "3.4"), ("10", "6.4"), ("15", "3.5"), ("15", "6.5"), ("7.5", "3.4")):
        for fld in ("tvar", "grad"):
            A("ftime", "ftime-fps%s-fnum%s-%s" % (fps.replace(".", "p"), fnv.replace(".", "p"), fld), fps=fps, vec=fld,
              p={"time_remap_mode": "2", "frame_num": fnv})
    A("ftime", "ftime-fps10-speed110", fps="10", n=8, vec="tvar", p={"speed_percent": "110"})

    # ---- input frame rate model
    for r in ("29.97", "29.969999313354492", "24", "23.976", "59.94", "12", "60", "1000", "0.10000000149011612",
              "29.97003", "30", "100", "15", "7.5"):
        A("inrate", "inrate%s" % r.replace(".", "p"), fps="24", p={"in_fps_is_out_fps": "0", "input_frame_rate": r})
    for fps in ("30", "30000/1001"):
        for r in ("29.97", "29.97003", "24"):
            A("inrate", "inrate%s-fps%s" % (r.replace(".", "p"), fps.replace("/", "_")), fps=fps,
              p={"in_fps_is_out_fps": "0", "input_frame_rate": r})
    A("inrate", "inrate-ignored", fps="24", p={"in_fps_is_out_fps": "1", "input_frame_rate": "12"})

    # ---- src_mark_segments, display
    for m in range(1, 6):
        A("mark", "mark%d" % m, n=5, p={"src_mark_segments": str(m)})
    for k in ("1@0;2@2", "4@0;1@3", "5@0;2@2", "2@0;3@1;1@2"):
        A("mark", "mark-keys-" + k.replace("@", "at").replace(";", "_"), n=5, p={"src_mark_segments": k},
          stand_in="keyframe_interpolation")
    for m in range(1, 7):
        A("display", "display%d" % m, p={"display": str(m)})
    A("display", "display2-source-layer", p={"display": "2"}, src="multi", n=4)
    A("display", "display2-d16", 16, 12, 16, p={"display": "2"})
    A("display", "display2-d32", 16, 12, 32, p={"display": "2"})

    # ---- blur compensation
    for b in ("0.01", "0.5", "1", "2.5", "5", "10"):
        A("blur", "blur%s" % b.replace(".", "p"), p={"blur_compensation": b})
    for b in ("1", "10"):
        A("blur", "blur%s-fast" % b, n=4, p={"blur_compensation": b, "speed_percent": "200"}, src="tex", src_len=24,
          flags=("long_source",))
    A("blur", "blur-0", p={"blur_compensation": "-0"})
    for d in (16, 32):
        A("blur", "blur2p5", 16, 12, d, p={"blur_compensation": "2.5"})
    for b in ("1", "5"):
        A("blur", "blur%s-inv" % b, p={"blur_compensation": b, "warping": "1"})

    # ---- vector scale and max displace
    for v in ("-1", "0", "-0", "0.5", "2", "10", "-10", "0.01", "3.3"):
        A("vecscale", "scalex%s" % v.replace(".", "p").replace("-", "m"), p={"vec_scale_x": v})
        A("vecscale", "scaley%s" % v.replace(".", "p").replace("-", "m"), p={"vec_scale_y": v})
    A("vecscale", "scalexy-flip", p={"vec_scale_x": "-1", "vec_scale_y": "-1"})
    A("vecscale", "scalexy-mid", p={"vec_scale_x": "0.5", "vec_scale_y": "2"})
    for d in (16, 32):
        A("vecscale", "scalex2", 16, 12, d, p={"vec_scale_x": "2", "vec_scale_y": "-0.5"})
    for v in ("0", "1", "4", "8", "16", "24", "40", "48", "64", "100", "2048", "10000", "7.505", "0.0001", "31.99999"):
        A("maxdisp", "md%s" % v.replace(".", "p"), p={"vectors_max_displace": v})
    A("maxdisp", "md-0", p={"vectors_max_displace": "-0"})
    for d in (16, 32):
        A("maxdisp", "md16", 16, 12, d, p={"vectors_max_displace": "16"})
    for d in D:
        for fld in ("big", "med"):
            A("maxdisp", "md2048-%s" % fld, 16, 12, d, p={"vectors_max_displace": "2048"}, vec=fld, md_img=2048.0)

    # ---- keyframed parameters (stand-in: the host interpolates the keys)
    keyed = [("speed_percent", "100@0;50@3"), ("speed_percent", "50@0e;100@3e"), ("vec_scale_x", "1@0;-1@3"),
             ("vec_scale_y", "2@0;0.25@3"), ("vectors_max_displace", "32@0;16@3"), ("blur_compensation", "0@0;10@3"),
             ("frame_interp", "1@0;3@2"), ("warping", "1@0;2@2"), ("display", "1@0;2@2"),
             ("input_frame_rate", "24@0;30@3"), ("time_remap_mode", "1@0;2@2")]
    for pid, k in keyed:
        extra = {"in_fps_is_out_fps": "0"} if pid == "input_frame_rate" else {}
        extra.update({pid: k})
        A("keyed", "key-%s" % pid, n=5, p=extra, stand_in="keyframe_interpolation")

    # ---- vector content, per depth
    fields = ["zero", "sub", "sub2", "small", "med", "large", "clipped", "edgeval", "grad", "rot", "stripes",
              "holes", "partial", "allhole", "noisef", "tvar", "gholes", "gpartial", "galpha", "gall"]
    for fld in fields:
        for d in D:
            A("vecfield", "vec-%s" % fld, 16, 12, d, vec=fld)
    for fld in ("coh", "grad", "noisef"):
        A("vecfield", "vec-%s-big-frame" % fld, 37, 23, 8, vec=fld)

    # ---- layer presence
    for d in D:
        A("layers", "no-vectors", 16, 12, d, prev=False, nxt=False)
        A("layers", "prev-only", 16, 12, d, nxt=False, vec="grad")
        A("layers", "next-only", 16, 12, d, prev=False, vec="grad")
        A("layers", "both-source-layer", 16, 12, d, src="tex", vec="grad")
    for wp in (1, 2):
        for fi in (1, 2, 3):
            A("layers", "prev-only-w%d-i%d" % (wp, fi), 16, 12, 8, nxt=False, vec="noisef",
              p={"warping": str(wp), "frame_interp": str(fi)})
            A("layers", "next-only-w%d-i%d" % (wp, fi), 16, 12, 8, prev=False, vec="noisef",
              p={"warping": str(wp), "frame_interp": str(fi)})
    for d in D:
        A("layers", "short-vectors", 16, 12, d, n=6, prev_len=3, next_len=3, vec="grad")
    A("layers", "short-next-only", 16, 12, 8, n=6, prev=False, next_len=3, vec="grad")
    A("layers", "vectors-1-frame", 16, 12, 8, n=4, prev_len=1, next_len=1)
    A("layers", "prev-short-next-full", 16, 12, 8, n=6, prev_len=2, vec="grad")

    # ---- footage content, per depth
    for foot in ("noise", "alpha", "flat", "checker", "disc", "multi", "edge"):
        for d in D:
            A("footage", "foot-%s" % foot, 16, 12, d, foot=foot, vec="grad")
    A("footage", "foot-hdr", 16, 12, 32, foot="hdr", vec="grad")

    # ---- 32 bit special values, 12x9
    for nm, _ in SPECIALS:
        A("sp_src", "spsrc-%s" % nm, 12, 9, 32, foot="sp_" + nm, vec="grad")
        A("sp_src", "spsrcA-%s" % nm, 12, 9, 32, foot="spa_" + nm, vec="grad")
        A("sp_vecR", "spvecR-%s" % nm, 12, 9, 32, vec="spR_" + nm)
        A("sp_vecG", "spvecG-%s" % nm, 12, 9, 32, vec="spG_" + nm)
        A("sp_vecA", "spvecA-%s" % nm, 12, 9, 32, vec="spA_" + nm)
    for nm in ("zero", "negzero", "one", "flt_max", "inf", "nan"):
        A("sp_vecB", "spvecB-%s" % nm, 12, 9, 32, vec="spB_" + nm)
    for nm in ("zero", "negzero", "denorm_min", "flt_max", "nan"):
        A("sp_vecR", "spvecR-inv-%s" % nm, 12, 9, 32, vec="spR_" + nm, p={"warping": "1"})

    # ---- FG matte layers (measurement only)
    for disp in (1, 3, 6):
        A("fg", "fg1-display%d" % disp, 16, 12, 8, fg=1, vec="grad", p={"display": str(disp)})
    for disp in (1, 4):
        A("fg", "fg2-display%d" % disp, 16, 12, 8, fg=2, vec="grad", p={"display": str(disp)})
    for disp in (1, 5, 6):
        A("fg", "fg3-display%d" % disp, 16, 12, 8, fg=3, vec="grad", p={"display": str(disp)})
    for ch in (1, 2, 3, 4):
        A("fg", "fg1-chan%d" % ch, 16, 12, 8, fg=1, vec="grad", p={"display": "3", "fg1_matte_chan": str(ch)})
    for sh in ("1.5", "10"):
        A("fg", "fg1-shrink%s" % sh.replace(".", "p"), 16, 12, 8, fg=2, vec="grad", p={"display": "1", "fg1_inv_matte_shrink": sh})
    for d in (16, 32):
        A("fg", "fg1-display3", 16, 12, d, fg=1, vec="grad", p={"display": "3"})
        A("fg", "fg2-display1", 16, 12, d, fg=2, vec="holes")
    A("fg", "fg1-prev-only", 16, 12, 8, fg=1, nxt=False, vec="grad", p={"display": "3"})
    A("fg", "fg1-warp-inverse", 16, 12, 8, fg=1, vec="grad", p={"display": "3", "warping": "1"})
    A("fg", "fg1-source-display", 16, 12, 8, fg=1, vec="grad", p={"display": "2"})
    A("fg", "fg1-short-matte", 16, 12, 8, n=6, fg=1, mat_len=2, vec="grad", p={"display": "3"})

    # ---- rates and lengths
    for fps in ("12", "23.976", "60", "24000/1001"):
        A("misc", "fps%s" % fps.replace("/", "_").replace(".", "p"), fps=fps)
    for n in (1, 2, 8):
        A("misc", "frames%d" % n, n=n)
    A("misc", "speed100-n1", n=1, p={"speed_percent": "100"})


def out_of_contract(out):
    """Parameter values outside the registered ranges, and popup values the plugin does not have.
    Checked with `vfx apply --check` (the rewrite's own range check); the plugin's behaviour is not
    asked for."""
    cands = [
        ("speed_percent=10001", "speed above 10000"), ("speed_percent=-10001", "speed below -10000"),
        ("frame_num=200001", "frame number above 200000"), ("frame_num=-200001", "frame number below -200000"),
        ("vec_scale_x=10.01", "scale x above 10"), ("vec_scale_x=-10.01", "scale x below -10"),
        ("vec_scale_y=10.01", "scale y above 10"), ("vec_scale_y=-10.01", "scale y below -10"),
        ("vectors_max_displace=-1", "max displace below 0"), ("vectors_max_displace=10001", "max displace above 10000"),
        ("input_frame_rate=0.1", "input rate below the 32 bit minimum"), ("input_frame_rate=1000.5", "input rate above 1000"),
        ("blur_compensation=10.01", "blur above 10"), ("blur_compensation=-0.01", "blur below 0"),
        ("frame_interp=0", "popup 0"), ("frame_interp=4", "popup above 3"), ("warping=0", "popup 0"),
        ("warping=3", "popup above 2"), ("display=0", "popup 0"), ("display=7", "popup above 6"),
        ("src_mark_segments=0", "popup 0"), ("src_mark_segments=6", "popup above 5"),
        ("time_remap_mode=0", "popup 0"), ("time_remap_mode=3", "popup above 2"),
        ("fg1_matte_chan=5", "popup above 4"), ("fg1_inv_matte_shrink=10.01", "shrink above 10"),
        ("fg2_inv_matte_shrink=-0.01", "shrink below 0"),
        ("src_time_offset=1", "param the effect does not have"), ("vectors=2", "param the effect does not have"),
        ("sensitivity=70", "param the effect does not have"), ("smart_blend=1", "param the effect does not have"),
    ]
    # The plugin has no Output: Frame Rate control, so the two output_frame_rate candidates (first
    # numbered ooc12 and ooc13) are gone; the others keep the ids they had.
    ids = [i for i in range(len(cands) + 2) if i not in (12, 13)]
    doc = [{"id": "ooc%02d" % i, "label": p, "why": why, "params": [p], "width": 16, "height": 12, "depth": 8}
           for i, (p, why) in zip(ids, cands)]
    with open(os.path.join(out, "out_of_contract.json"), "w") as fh:
        json.dump(doc, fh, indent=1)
    return len(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g = Grid(a.out)
    build(g)
    g.write()
    n = out_of_contract(a.out)
    groups = {}
    for m in g.meta.values():
        groups[m["group"]] = groups.get(m["group"], 0) + 1
    print("wrote %s: %d cases, %d input files, %d out-of-contract candidates" % (a.out, len(g.cases), len(g.inputs), n))
    print("groups:", ", ".join("%s %d" % kv for kv in sorted(groups.items())))


if __name__ == "__main__":
    main()
