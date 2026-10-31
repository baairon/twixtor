#!/usr/bin/env python3
"""Grid generator of the fgbg-layers parity module (retime-fgbg with FG matte layers, the inverse
matte shrink, layer order, display groups, tracking points and Draw Geom).

Writes an fx-parity style test set (format 1): DIR/cases.json (with the `inputs` list that
fx-parity requires: every clip file the kept cases name, once) plus raw input clips and matte layer
clips in DIR/inputs/, DIR/grid.tsv (id, block, tags) and DIR/out_of_contract.json. Standard library
only, deterministic.

  python3 gen_grid.py --out DIR [--vfx PATH]

With --vfx each case is checked with `vfx apply --check`; a case vfx refuses moves to
out_of_contract.json (never counted as a pass or a failure). The explicit out of contract
candidates (values outside the registered ranges or popup choices, a matte layer of another size
or shorter than the clip, a point at NaN) are always written there. The second tracker
(main_bg_track, fgN_track) is not a parameter of retime-fgbg and has no case here.
"""
import argparse, hashlib, json, math, os, random, struct, subprocess

EFFECT = "retime-fgbg"
PIX = {8: "rgba", 16: "rgba64le", 32: "gbrapf32le"}
LAYER_IDS = ["fg1_matte", "fg2_matte", "fg3_matte"]


class Bits(int):
    """A raw float32 bit pattern (32 bpc only)."""


def hsh(a, b, c, d=0):
    x = (a * 73856093) ^ (b * 19349663) ^ (c * 83492791) ^ (d * 2654435761)
    x &= 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 2246822519) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 3266489917) & 0xFFFFFFFF
    x ^= x >> 16
    return x / 4294967296.0


def clamp01(v):
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def fbits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


SPEC = [0x00000000, 0x80000000, 0x00000001, 0x007FFFFF, 0x00800000, 0x3F800000, 0x40200000,
        0x7F7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000]
NEGNAN = 0xFFC00000
LEVELS = {8: [0, 1, 2, 127, 128, 243, 254, 255], 16: [0, 1, 255, 256, 32767, 32768, 62259, 65534, 65535]}
FLOAT_LEVELS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]

# ---------------------------------------------------------------- scenes (the footage)


def box_geom(w, h, k, fast):
    m = 2 if fast else 1
    if k == 0:
        bw = bh = max(2, min(w, h) // 2 + 1)
        x0, y0, sx, sy = (w - bw) // 4, (h - bh) // 4, 1, 1
    elif k == 1:
        bw, bh = max(2, (2 * w) // 5 + 1), max(2, h // 2)
        x0, y0, sx, sy = ((w - bw) * 3) // 4, (h - bh) // 2, -1, 1
    else:
        bw, bh = max(2, w // 3), max(2, h // 3)
        x0, y0, sx, sy = (w - bw) // 2, ((h - bh) * 3) // 4, 0, -1
    return x0, y0, bw, bh, sx * m, sy * m


def box_at(w, h, k, f, fast):
    x0, y0, bw, bh, sx, sy = box_geom(w, h, k, fast)
    return x0 + sx * f, y0 + sy * f, bw, bh


def in_box(x, y, g):
    return g[0] <= x < g[0] + g[2] and g[1] <= y < g[1] + g[3]


def bg(x, y):
    return (0.5 + 0.35 * math.sin(0.8 * x + 0.5 * y) + 0.1 * (hsh(x, y, 1) - 0.5),
            0.5 + 0.35 * math.sin(0.4 * x - 0.7 * y + 1.0) + 0.1 * (hsh(x, y, 2) - 0.5),
            0.5 + 0.3 * math.cos(0.3 * x + 0.9 * y) + 0.1 * (hsh(x, y, 3) - 0.5))


BASE = [(0.9, 0.2, 0.3), (0.2, 0.8, 0.3), (0.3, 0.3, 0.9)]


def tex(k, bx, by):
    b = BASE[k]
    return (clamp01(b[0] - 0.05 * bx + 0.1 * hsh(bx, by, 5 + k)),
            clamp01(b[1] + 0.1 * hsh(bx, by, 7 + k) - 0.03 * by),
            clamp01(b[2] + 0.05 * by))


def scene(n, alpha=False, fast=False, spec=False):
    """n boxes over a background, FG1 (box 0) on top; box k is the object of FG matte k."""
    def fn(x, y, f, w, h, depth):
        hit = -1
        for k in range(n):
            if in_box(x, y, box_at(w, h, k, f, fast)):
                hit = k
                break
        if hit >= 0:
            g = box_at(w, h, hit, f, fast)
            if spec:
                v = [Bits(SPEC[(x + y + 3 * c + f + hit) % len(SPEC)]) for c in range(3)]
                a = Bits(SPEC[(x + 2 * y + f) % len(SPEC)]) if alpha else 1.0
                return (*v, a)
            r, gg, b = tex(hit, x - g[0], y - g[1])
            return (r, gg, b, 0.6 if alpha else 1.0)
        r, gg, b = bg(x, y)
        a = (0.2 + 0.8 * (x / (w - 1) if w > 1 else 0.5)) if alpha else 1.0
        return (clamp01(r), clamp01(gg), clamp01(b), a)
    return fn


def px_pan(x, y, f, w, h, depth):
    xs, ys = x - 1.3 * f, y - 0.6 * f
    return (clamp01(0.5 + 0.45 * math.sin(0.7 * xs + 0.3 * ys)),
            clamp01(0.5 + 0.45 * math.sin(0.5 * xs - 0.6 * ys + 2.0)),
            clamp01(0.5 + 0.45 * math.cos(0.9 * xs + 0.2 * ys)), 1.0)


def px_wrapcol(x, y, f, w, h, depth):
    """Periodic across the width: the content shifts by 2 columns a frame and wraps."""
    t = 2.0 * math.pi * ((x - 2.0 * f) % w) / w
    return (clamp01(0.5 + 0.4 * math.sin(2 * t + 0.4 * y)), clamp01(0.5 + 0.4 * math.sin(3 * t - 0.3 * y + 1.0)),
            clamp01(0.5 + 0.4 * math.cos(t + 0.2 * y)), 1.0)


def px_noise(x, y, f, w, h, depth):
    return (hsh(x, y, 11, f), hsh(x, y, 12, f), hsh(x, y, 13, f), 0.25 + 0.75 * hsh(x, y, 14, f))


def px_flat(x, y, f, w, h, depth):
    return (0.25, 0.5, 0.75, 1.0)


def px_edge(x, y, f, w, h, depth):
    lv = LEVELS[depth]
    top = 255.0 if depth == 8 else 65535.0
    return tuple(lv[(x + 2 * y + 3 * c + f) % len(lv)] / top for c in range(4))


def px_specall(x, y, f, w, h, depth):
    v = [Bits(SPEC[(x + 2 * y + 3 * c + f) % len(SPEC)]) for c in range(4)]
    return tuple(v)


SOURCES = {"pan": px_pan, "wrapcol": px_wrapcol, "noise": px_noise, "flat": px_flat, "edge": px_edge, "specall": px_specall}
for _n in (1, 2, 3):
    for _suf, _kw in (("", {}), ("a", {"alpha": True}), ("f", {"fast": True})):
        SOURCES[f"box{_n}{_suf}"] = scene(_n, **_kw)
    SOURCES[f"specbox{_n}"] = scene(_n, spec=True)
    SOURCES[f"specbox{_n}a"] = scene(_n, spec=True, alpha=True)
FLOAT_ONLY = {"specall", "specbox1", "specbox2", "specbox3", "specbox1a", "specbox2a", "specbox3a"}
INT_ONLY = {"edge"}

# ---------------------------------------------------------------- matte layers (the FG matte clips)


def soft_at(x, y, g):
    dx = max(g[0] - x, x - (g[0] + g[2] - 1), 0)
    dy = max(g[1] - y, y - (g[1] + g[3] - 1), 0)
    return clamp01(1.0 - math.hypot(dx, dy) / 2.0)


def matte(kind, k, fast, depth):
    """fn(x, y, f, w, h) -> (r, g, b, a) of the matte clip for the box k."""
    def hard(x, y, f, w, h, j=None):
        return 1.0 if in_box(x, y, box_at(w, h, k if j is None else j, f, fast)) else 0.0

    def soft(x, y, f, w, h):
        return soft_at(x, y, box_at(w, h, k, f, fast))

    z = Bits(0)
    one = 1.0
    if kind == "hard":
        return lambda x, y, f, w, h: (hard(x, y, f, w, h),) * 3 + (one,)
    if kind == "chan":
        return lambda x, y, f, w, h: (hard(x, y, f, w, h), soft(x, y, f, w, h), hard(x, y, f, w, h, (k + 1) % 3), one)
    if kind == "alpha":
        return lambda x, y, f, w, h: (one, one, one, hard(x, y, f, w, h))
    if kind == "soft":
        return lambda x, y, f, w, h: (soft(x, y, f, w, h),) * 3 + (one,)
    if kind == "empty":
        return lambda x, y, f, w, h: (0.0, 0.0, 0.0, 0.0)
    if kind == "emptya":
        return lambda x, y, f, w, h: (0.0, 0.0, 0.0, one)
    if kind == "full":
        return lambda x, y, f, w, h: (one, one, one, one)
    if kind == "grey":
        return lambda x, y, f, w, h: (0.5, 0.5, 0.5, one)
    if kind in ("thrA", "thrB"):
        hi = kind == "thrA"
        if depth == 8:
            lv = (243 if hi else 242) / 255.0
        elif depth == 16:
            lv = (62259 if hi else 62258) / 65535.0
        else:
            lv = 0.95 if hi else Bits(fbits(0.95) - 1)
        return lambda x, y, f, w, h: ((lv if hard(x, y, f, w, h) else 0.0),) * 3 + (one,)
    if kind == "noise":
        return lambda x, y, f, w, h: (hsh(x, y, 21 + k, f), hsh(x, y, 22 + k, f), hsh(x, y, 23 + k, f), one)
    if kind == "ramp":
        lv = [v / (255.0 if depth == 8 else 65535.0) for v in LEVELS[depth]] if depth < 32 else FLOAT_LEVELS
        return lambda x, y, f, w, h: tuple(lv[(x + 2 * y + 3 * c + f + k) % len(lv)] for c in range(3)) + (one,)
    if kind == "seam":
        return lambda x, y, f, w, h: ((1.0 if (x < 3 or x >= w - 3) else 0.0),) * 3 + (one,)
    if kind == "specm":
        return lambda x, y, f, w, h: tuple(Bits(SPEC[(x + 2 * y + 3 * c + f + k) % len(SPEC)]) for c in range(3)) + (one,)
    if kind == "specbox":
        return lambda x, y, f, w, h: (tuple(Bits(SPEC[(x + y + 3 * c + f + k) % len(SPEC)]) for c in range(3))
                                      if hard(x, y, f, w, h) else (z, z, z)) + (one,)
    consts = {"nanall": SPEC[10], "negnanall": NEGNAN, "infall": SPEC[8], "ninfall": SPEC[9], "maxall": SPEC[7],
              "denall": SPEC[2], "maxden": SPEC[3], "fltmin": SPEC[4], "negzero": SPEC[1], "over1": 0x40200000}
    if kind in consts:
        c = Bits(consts[kind])
        if kind == "over1":
            return lambda x, y, f, w, h: ((c if hard(x, y, f, w, h) else z),) * 3 + (one,)
        return lambda x, y, f, w, h: (c, c, c, one)
    if kind == "nanbox":
        c = Bits(SPEC[10])
        return lambda x, y, f, w, h: ((c if hard(x, y, f, w, h) else z),) * 3 + (one,)
    raise ValueError(kind)


FLOAT_MATTES = {"specm", "specbox", "nanall", "negnanall", "infall", "ninfall", "maxall", "denall", "maxden", "fltmin",
                "negzero", "over1", "nanbox"}

# ---------------------------------------------------------------- raw clips


def enc_int(v, top):
    return int(min(top, max(0, math.floor(v * top + 0.5))))


def render_clip(fn, w, h, depth, frames):
    out = bytearray()
    for f in range(frames):
        if depth == 32:
            planes = [[], [], [], []]  # G, B, R, A
            for y in range(h):
                for x in range(w):
                    r, g, b, a = [(v if isinstance(v, Bits) else fbits(v)) for v in fn(x, y, f, w, h, depth)]
                    planes[0].append(g); planes[1].append(b); planes[2].append(r); planes[3].append(a)
            for pl in planes:
                out += struct.pack("<%dI" % len(pl), *pl)
        else:
            top = 255 if depth == 8 else 65535
            fmt = "<B" if depth == 8 else "<H"
            for y in range(h):
                for x in range(w):
                    for v in fn(x, y, f, w, h, depth):
                        out += struct.pack(fmt, enc_int(v, top))
    return bytes(out)


# ---------------------------------------------------------------- the case grid


class Grid:
    def __init__(self):
        self.cases, self.seen, self.rows, self.files = [], set(), [], {}

    def add(self, block, label, source, w, h, depth, params=(), mattes=(), frames=3, fps="24", fast=None, tags=()):
        """mattes: kinds for FG1.. ; "kind:j" takes the box j instead of the layer's own index."""
        params = list(params)
        mattes = list(mattes)
        key = (source, w, h, depth, frames, fps, tuple(params), tuple(mattes))
        if key in self.seen:
            return
        self.seen.add(key)
        hh = hashlib.sha1("|".join(map(str, key)).encode()).hexdigest()[:8]
        cid = f"{EFFECT}.{label}.{source}-{w}x{h}.d{depth}.{hh}"
        inp = f"inputs/{source}-{w}x{h}.d{depth}.f{frames}.{PIX[depth]}"
        self.files[inp] = ("scene", source, w, h, depth, frames)
        layers = {}
        isfast = (source.startswith("box") and source.endswith("f")) if fast is None else fast
        for i, m in enumerate(mattes):
            if m == "-":
                continue
            kind, _, j = m.partition(":")
            box = int(j) if j else i
            name = f"inputs/m-{kind}-{box}{'f' if isfast else ''}-{w}x{h}.d{depth}.f{frames}.{PIX[depth]}"
            self.files[name] = ("matte", kind, box, isfast, w, h, depth, frames)
            layers[LAYER_IDS[i]] = name
        self.cases.append(dict(id=cid, effect=EFFECT, label=label, params=params, depth=depth, width=w, height=h,
                               source=source, input=inp, reference=f"{cid}.{PIX[depth]}", kind="temporal", frames=frames,
                               fps=fps, **({"layers": layers} if layers else {})))
        self.rows.append((cid, block, ",".join(tags)))


def P(**kw):
    return [f"{k}={v}" for k, v in kw.items()]


HALF = P(speed_percent=50)


def build_grid():
    g = Grid()
    add = g.add

    # S1 sizes x depths: 1 to 3 layers by turn, warping by turn, hard mattes
    SIZES = [(1, 1), (2, 2), (2, 3), (3, 2), (1, 5), (5, 1), (3, 3), (4, 4), (5, 5), (7, 9), (9, 7), (16, 12), (17, 13),
             (23, 31), (37, 23)]
    for i, (w, h) in enumerate(SIZES):
        for d in ((8, 16, 32) if (w * h < 40 or (w, h) == (16, 12)) else ((8, 16, 32)[i % 3],)):
            nl = 1 + (i + d // 8) % 3
            wp = 1 + (i + d // 16) % 4
            add("size", f"size-n{nl}-w{wp}", f"box{nl}", w, h, d, HALF + P(warping=wp), ["hard"] * nl,
                tags=("tiny",) if min(w, h) < 4 else ())
    add("size", "size-64x48", "box2", 64, 48, 8, HALF + P(warping=3), ["hard", "hard"])
    for w, h in [(1, 1), (5, 5), (17, 13)]:
        add("size", "size-defaults", "box1", w, h, 8, [], ["hard"], tags=("defaults", "tiny") if w < 4 else ("defaults",))

    # S2 layers x warping x depth
    for nl in (1, 2, 3):
        for wp in (1, 2, 3, 4):
            add("layers", f"n{nl}-w{wp}", f"box{nl}", 24, 16, 8, HALF + P(warping=wp), ["hard"] * nl)
    for nl, wp in ((2, 2), (2, 4), (3, 2), (3, 4)):
        add("layers", f"n{nl}-w{wp}", f"box{nl}", 24, 16, 16, HALF + P(warping=wp), ["hard"] * nl)
    for nl, wp in ((2, 1), (2, 3), (3, 1), (3, 3)):
        add("layers", f"n{nl}-w{wp}", f"box{nl}", 24, 16, 32, HALF + P(warping=wp), ["hard"] * nl)
    for wp in (1, 3):
        add("layers", f"alpha-n2-w{wp}", "box2a", 24, 16, 8, HALF + P(warping=wp), ["hard"] * 2)
    add("layers", "pan-n1", "pan", 24, 16, 8, HALF + P(warping=3), ["hard"])
    add("layers", "noise-n2", "noise", 24, 16, 8, HALF + P(warping=1), ["hard", "hard"])
    add("layers", "flat-n1", "flat", 24, 16, 8, HALF + P(warping=3), ["hard"])
    add("layers", "edge-n2-d8", "edge", 16, 12, 8, HALF + P(warping=3), ["hard"] * 2)
    add("layers", "edge-n2-d16", "edge", 16, 12, 16, HALF + P(warping=1), ["hard"] * 2)

    # S3 matte channels
    for c in (1, 2, 3, 4):
        add("chan", f"chan{c}-chan", "box1", 24, 16, 8, HALF + P(warping=3, fg1_matte_chan=c), ["chan"])
        add("chan", f"chan{c}-alpha", "box1", 24, 16, 8, HALF + P(warping=3, fg1_matte_chan=c), ["alpha"])
    for c, d in ((2, 16), (4, 16), (3, 32), (4, 32)):
        add("chan", f"chan{c}-chan", "box1", 24, 16, d, HALF + P(warping=3, fg1_matte_chan=c), ["chan"])
    for tag, cs in (("123", (1, 2, 3)), ("321", (3, 2, 1)), ("444", (4, 4, 4)), ("222", (2, 2, 2)), ("213", (2, 1, 3))):
        add("chan", f"perm{tag}", "box3", 24, 16, 8,
            HALF + P(warping=3, fg1_matte_chan=cs[0], fg2_matte_chan=cs[1], fg3_matte_chan=cs[2]), ["chan"] * 3)
    add("chan", "perm321-alpha", "box3", 24, 16, 8, HALF + P(warping=1, fg1_matte_chan=4, fg2_matte_chan=4, fg3_matte_chan=1),
        ["alpha", "alpha", "chan"])

    # S4 inverse matte shrink (G1): it shapes the tracking matte of the layers below, best seen on display 6
    SH = ["0", "-0", "0.01", "0.5", "1", "1.5", "2", "2.99", "3", "4.25", "6", "7.25", "9.99", "10"]
    for s in SH:
        add("shrink", f"shrink{s}-bg", "box2", 36, 24, 8, HALF + P(warping=3, display=6, fg1_inv_matte_shrink=s), ["hard", "hard"],
            tags=("negzero",) if s == "-0" else ())
    for s in ("1", "2.5", "10"):
        add("shrink", f"shrink{s}-out", "box2", 36, 24, 8, HALF + P(warping=3, fg1_inv_matte_shrink=s), ["hard", "hard"])
    for s in ("0.5", "3"):
        for d in (16, 32):
            add("shrink", f"shrink{s}-bg", "box2", 36, 24, d, HALF + P(warping=3, display=6, fg1_inv_matte_shrink=s), ["hard", "hard"])
    add("shrink", "shrink-soft", "box2", 36, 24, 8, HALF + P(warping=3, display=6, fg1_inv_matte_shrink=2), ["soft", "soft"])
    add("shrink", "shrink-fg2", "box3", 36, 24, 8, HALF + P(warping=3, display=6, fg2_inv_matte_shrink=2.5), ["hard"] * 3)
    add("shrink", "shrink-fg3", "box3", 36, 24, 8, HALF + P(warping=3, display=6, fg3_inv_matte_shrink=4), ["hard"] * 3)
    add("shrink", "shrink-fg2-fg2view", "box3", 36, 24, 8, HALF + P(warping=3, display=4, fg1_inv_matte_shrink=3), ["hard"] * 3)
    add("shrink", "shrink-all3", "box3", 36, 24, 8, HALF + P(warping=1, display=6, fg1_inv_matte_shrink=1.5, fg2_inv_matte_shrink=2,
                                                          fg3_inv_matte_shrink=0.5), ["hard"] * 3)
    add("shrink", "shrink-n1-d6", "box1", 36, 24, 8, HALF + P(warping=1, display=6, fg1_inv_matte_shrink=2), ["hard"])
    add("shrink", "shrink-thin", "box2", 12, 9, 8, HALF + P(warping=3, display=6, fg1_inv_matte_shrink=7.5), ["hard", "hard"])
    for s in ("0.5", "1", "2.5", "6"):
        add("shrink", f"shrink{s}-pan", "pan", 36, 24, 8, HALF + P(warping=3, display=6, fg1_inv_matte_shrink=s), ["hard", "hard"])
    add("shrink", "shrink-nan32", "box2", 24, 16, 32, HALF + P(warping=3, display=6, fg1_inv_matte_shrink=2.5), ["nanbox", "hard"],
        tags=("special",))

    # S5 display groups
    for dsp in (1, 2, 3, 4, 5, 6):
        for nl in (1, 2, 3):
            add("display", f"disp{dsp}-n{nl}", f"box{nl}", 24, 16, 8, HALF + P(warping=3, display=dsp), ["hard"] * nl)
    for dsp in (3, 4, 6):
        add("display", f"disp{dsp}-n2", "box2", 24, 16, 16, HALF + P(warping=2, display=dsp), ["hard"] * 2)
    for dsp in (3, 5):
        add("display", f"disp{dsp}-n3", "box3", 24, 16, 32, HALF + P(warping=1, display=dsp), ["hard"] * 3)
    add("display", "disp6-n3-w4", "box3", 24, 16, 8, HALF + P(warping=4, display=6), ["hard"] * 3)
    add("display", "disp3-n3-w2", "box3", 24, 16, 8, HALF + P(warping=2, display=3), ["hard"] * 3)

    # S6 fast paths (exact position, No Vectors, Mark Segments): the composed group, no warp (G1h)
    for dsp in (1, 3, 4, 6):
        add("fast", f"exact-d{dsp}", "box2", 24, 16, 8, P(speed_percent=100, warping=3, display=dsp), ["hard"] * 2)
    add("fast", "exact-d3-d16", "box2", 24, 16, 16, P(speed_percent=100, warping=2, display=3), ["hard"] * 2)
    add("fast", "exact-d6-d32", "box2", 24, 16, 32, P(speed_percent=100, warping=1, display=6), ["hard"] * 2)
    for dsp in (1, 3, 6):
        add("fast", f"novec-d{dsp}", "box2", 24, 16, 8, HALF + P(vectors=1, warping=3, display=dsp), ["hard"] * 2)
    add("fast", "novec-n3-w1", "box3", 24, 16, 8, HALF + P(vectors=1, warping=1), ["hard"] * 3)
    for sm in (4, 5):
        for dsp in (1, 3):
            add("fast", f"mark{sm}-d{dsp}", "box2", 24, 16, 8, HALF + P(src_mark_segments=sm, warping=3, display=dsp), ["hard"] * 2)
    add("fast", "exact-soft", "box2", 24, 16, 8, P(speed_percent=100, warping=3), ["soft", "soft"])

    # S7 frame interpolation
    for fi in (1, 2, 3):
        for wp in (1, 2, 3, 4):
            add("interp", f"interp{fi}-w{wp}", "box2", 24, 16, 8, HALF + P(warping=wp, frame_interp=fi), ["hard"] * 2)
    for fi in (1, 3):
        add("interp", f"interp{fi}-w3", "box2", 24, 16, 32, HALF + P(warping=3, frame_interp=fi), ["hard"] * 2)
    add("interp", "interp1-n1", "box1", 24, 16, 8, HALF + P(warping=1, frame_interp=1), ["hard"])

    # S8 search sensitivities and the layer motion (G1b: the search is masked by the layer matte)
    for s in ("0", "-0", "100"):
        add("sens", f"fg1sens{s}", "box2f", 32, 24, 8, HALF + P(warping=3, fg1_sensitivity=s), ["hard"] * 2,
            tags=("negzero",) if s == "-0" else ())
    for s in ("0", "100"):
        add("sens", f"fg2sens{s}", "box3f", 32, 24, 8, HALF + P(warping=3, fg2_sensitivity=s), ["hard"] * 3)
        add("sens", f"fg3sens{s}", "box3f", 32, 24, 8, HALF + P(warping=3, fg3_sensitivity=s), ["hard"] * 3)
        add("sens", f"mbsens{s}", "box2f", 32, 24, 8, HALF + P(warping=3, main_bg_sensitivity=s), ["hard"] * 2)
    add("sens", "allsens100", "box3f", 32, 24, 16, HALF + P(warping=3, fg1_sensitivity=100, fg2_sensitivity=100, fg3_sensitivity=100,
                                                         main_bg_sensitivity=100), ["hard"] * 3)
    add("sens", "allsens35-d32", "box3f", 32, 24, 32, HALF + P(warping=1, fg1_sensitivity=35, fg2_sensitivity=35, fg3_sensitivity=35,
                                                            main_bg_sensitivity=35), ["hard"] * 3)
    add("sens", "bigmotion", "box2f", 48, 32, 8, HALF + P(warping=3, fg1_sensitivity=100, main_bg_sensitivity=100), ["hard", "hard"])

    # S9 blur compensation tail of a layered render (G1f)
    for bc in ("0.01", "1", "2.5", "10"):
        for dsp in (1, 3, 6):
            add("blur", f"blur{bc}-d{dsp}", "box2", 24, 16, 8, P(speed_percent=200, warping=3, blur_compensation=bc, display=dsp),
                ["hard"] * 2)
    for wp in (1, 2, 4):
        add("blur", f"blur3-w{wp}", "box2", 24, 16, 8, P(speed_percent=200, warping=wp, blur_compensation=3), ["hard"] * 2)
    add("blur", "blur-neg0", "box2", 24, 16, 8, P(speed_percent=200, warping=3, blur_compensation="-0"), ["hard"] * 2,
        tags=("negzero",))
    add("blur", "blur2-d32", "box2", 24, 16, 32, P(speed_percent=200, warping=3, blur_compensation=2), ["hard"] * 2)

    # S10 layer order and matte products (G1c)
    for tag, ms in (("012", ["hard:0", "hard:1", "hard:2"]), ("210", ["hard:2", "hard:1", "hard:0"]),
                    ("120", ["hard:1", "hard:2", "hard:0"]), ("201", ["hard:2", "hard:0", "hard:1"])):
        add("order", f"order{tag}", "box3", 24, 16, 8, HALF + P(warping=3), ms)
    add("order", "order10", "box2", 24, 16, 8, HALF + P(warping=3), ["hard:1", "hard:0"])
    add("order", "order-fullfirst", "box2", 24, 16, 8, HALF + P(warping=3), ["full", "hard"])
    for dsp in (4, 5, 6):
        add("order", f"order-fullfirst-d{dsp}", "box3", 24, 16, 8, HALF + P(warping=3, display=dsp), ["full", "hard", "hard"])
    add("order", "order-soft-overlap", "box3", 24, 16, 8, HALF + P(warping=3, display=1), ["soft:0", "soft:0", "soft:2"])
    add("order", "order-same-box", "box2", 24, 16, 8, HALF + P(warping=1, display=4), ["hard:0", "hard:0"])
    add("order", "order-d16", "box3", 24, 16, 16, HALF + P(warping=1), ["hard:2", "hard:0", "hard:1"])

    # S11 which layers count (a layer counts only after the one before it); "-" leaves a layer out
    for label, ms in (("fg2only", ["-", "hard"]), ("fg3only", ["-", "-", "hard"]), ("fg1fg3", ["hard", "-", "hard"]),
                      ("fg2fg3", ["-", "hard", "hard"])):
        add("count", label, "box3", 24, 16, 8, HALF + P(warping=3, display=1), ms)
    add("count", "fg1fg3-d6", "box3", 24, 16, 8, HALF + P(warping=3, display=6), ["hard", "-", "hard"])
    add("count", "fg1fg2-d32", "box3", 24, 16, 32, HALF + P(warping=3, display=1), ["hard", "hard"])

    # S12 matte contents at 8, 16 and 32 bpc
    kinds_all = ["empty", "emptya", "full", "grey", "thrA", "thrB", "soft", "noise", "ramp"]
    for k in kinds_all:
        add("mattes", f"m-{k}", "box1", 24, 16, 8, HALF + P(warping=3), [k])
    for k in ("thrA", "thrB", "grey", "noise", "ramp", "empty"):
        add("mattes", f"m-{k}", "box1", 24, 16, 16, HALF + P(warping=3), [k])
    for k in ("empty", "full", "grey", "thrA", "thrB", "noise", "ramp", "soft", "emptya"):
        add("mattes", f"m-{k}", "box1", 24, 16, 32, HALF + P(warping=3), [k])
    add("mattes", "m-empty-n2", "box2", 24, 16, 8, HALF + P(warping=3), ["hard", "empty"])
    add("mattes", "m-empty-first", "box2", 24, 16, 8, HALF + P(warping=3), ["empty", "hard"])
    add("mattes", "m-empty-all3", "box3", 24, 16, 8, HALF + P(warping=3), ["empty"] * 3)
    add("mattes", "m-thr-n2", "box2", 24, 16, 8, HALF + P(warping=1), ["thrA", "thrB"])
    add("mattes", "m-thr-n2-d32", "box2", 24, 16, 32, HALF + P(warping=1), ["thrA", "thrB"])
    add("mattes", "m-full-n3", "box3", 24, 16, 8, HALF + P(warping=3, display=6), ["full", "full", "full"])
    add("mattes", "m-grey-bg", "box1", 24, 16, 8, HALF + P(warping=3, display=6, fg1_inv_matte_shrink=2), ["grey"])
    add("mattes", "m-ramp-d6", "box2", 24, 16, 16, HALF + P(warping=3, display=6), ["ramp", "ramp"])

    # S13 float special values (32 bpc), in the footage and in the mattes
    for src, wp in (("specbox2", 1), ("specbox2", 2), ("specbox2", 3), ("specbox2", 4), ("specbox2a", 3), ("specbox1", 1),
                    ("specbox3", 3), ("specall", 1), ("specall", 3)):
        add("spec", f"{src}-w{wp}", src, 16, 12, 32, HALF + P(warping=wp), ["hard"] * (3 if src.endswith("3") else 1 if src.endswith("1") or src == "specall" else 2),
            tags=("special",))
    for c in (1, 2, 3, 4):
        add("spec", f"specm-chan{c}", "box1", 16, 12, 32, HALF + P(warping=3, fg1_matte_chan=c), ["specm"], tags=("special",))
    add("spec", "specm-n2", "box2", 16, 12, 32, HALF + P(warping=1), ["specm", "specbox"], tags=("special",))
    add("spec", "specm-n2-w4", "box2", 16, 12, 32, HALF + P(warping=4), ["specbox", "specm"], tags=("special",))
    add("spec", "specbox-bg", "box2", 16, 12, 32, HALF + P(warping=3, display=6), ["specbox", "hard"], tags=("special",))
    add("spec", "specm-bg-shrink", "box2", 16, 12, 32, HALF + P(warping=3, display=6, fg1_inv_matte_shrink=1.5), ["specm", "specbox"],
        tags=("special",))
    for k in ("nanall", "negnanall", "infall", "ninfall", "maxall", "denall", "maxden", "fltmin", "negzero", "over1", "nanbox"):
        add("spec", f"m-{k}", "box1", 16, 12, 32, HALF + P(warping=3), [k], tags=("special",))
    for k in ("nanall", "infall", "negzero", "over1", "denall"):
        add("spec", f"m-{k}-n2-bg", "box2", 16, 12, 32, HALF + P(warping=3, display=6), [k, "hard"], tags=("special",))
    add("spec", "m-negzero-out", "box2", 16, 12, 32, HALF + P(warping=1), ["hard", "negzero"], tags=("special", "negzero"))
    add("spec", "specm-exact", "box2", 16, 12, 32, P(speed_percent=100, warping=3), ["specm", "specbox"], tags=("special",))

    # S14 tracking points that move the search (G2). A moving point is a keyed value (the stand-in)
    KEY = "pt1_pos=10,8@0;14,10@2"
    for u in (2, 3, 4, 5):
        add("points", f"pt-use{u}-n0", "box1", 32, 24, 8, HALF + P(warping=3, pt1_use=u) + [KEY], [], tags=("standin",))
    for u in (2, 3, 4, 5):
        add("points", f"pt-use{u}-n2", "box2", 32, 24, 8, HALF + P(warping=3, pt1_use=u) + [KEY], ["hard"] * 2, tags=("standin",))
    add("points", "pt-use1-n2", "box2", 32, 24, 8, HALF + P(warping=3, pt1_use=1) + [KEY], ["hard"] * 2, tags=("standin",))
    add("points", "pt-still", "box2", 32, 24, 8, HALF + P(warping=3, pt1_use=2, pt1_pos="10,8"), ["hard"] * 2)
    add("points", "pt-use3-n3", "box3", 32, 24, 8, HALF + P(warping=3, pt1_use=2, pt2_use=3, pt3_use=4, pt4_use=5)
        + ["pt1_pos=4,6@0;7,8@2", "pt2_pos=14,10@0;12,13@2", "pt3_pos=20,16@0;22,14@2", "pt4_pos=28,4@0;26,6@2"], ["hard"] * 3,
        tags=("standin",))
    add("points", "pt-same-pixel", "box2", 32, 24, 8, HALF + P(warping=3, pt1_use=5, pt2_use=5)
        + ["pt1_pos=10,8@0;14,10@2", "pt2_pos=10.2,8.3@0;12,14@2"], ["hard"] * 2, tags=("standin",))
    pos12 = ";".join([])
    p12 = []
    for n in range(1, 13):
        p12 += [f"pt{n}_use={2 + n % 4}", f"pt{n}_pos={2 + 2 * n},{3 + n}@0;{3 + 2 * n},{3 + n + (n % 3) - 1}@2"]
    add("points", "pt-all12", "box3", 32, 24, 8, HALF + P(warping=3) + p12, ["hard"] * 3, tags=("standin",))
    add("points", "pt12-only", "box2", 32, 24, 8, HALF + P(warping=3, pt12_use=5) + ["pt12_pos=12,9@0;15,12@2"], ["hard"] * 2,
        tags=("standin",))
    add("points", "pt-outside", "box2", 32, 24, 8, HALF + P(warping=3, pt1_use=5) + ["pt1_pos=-3,6@0;4,9@2"], ["hard"] * 2,
        tags=("standin",))
    add("points", "pt-outside-to", "box2", 32, 24, 8, HALF + P(warping=3, pt1_use=5) + ["pt1_pos=10,8@0;80,60@2"], ["hard"] * 2,
        tags=("standin",))
    add("points", "pt-default-pos", "box2", 32, 24, 8, HALF + P(warping=3, pt1_use=2), ["hard"] * 2)
    add("points", "pt-use-keyed", "box2", 32, 24, 8, HALF + P(warping=3, pt1_use="5@0;1@2") + [KEY], ["hard"] * 2, tags=("standin",))
    add("points", "pt-use-keyed2", "box2", 32, 24, 8, HALF + P(warping=3, pt1_use="1@0;5@2") + [KEY], ["hard"] * 2, tags=("standin",))
    add("points", "pt-subpixel", "box1", 32, 24, 8, HALF + P(warping=3, pt1_use=5) + ["pt1_pos=10.4,8.3@0;14.7,10.6@2"], ["hard"],
        tags=("standin",))
    add("points", "pt-w1", "box2", 32, 24, 8, HALF + P(warping=1, pt1_use=2) + [KEY], ["hard"] * 2, tags=("standin",))
    add("points", "pt-w2-n3", "box3", 32, 24, 8, HALF + P(warping=2, pt1_use=3, pt2_use=4) + ["pt1_pos=10,8@0;14,10@2", "pt2_pos=18,12@0;16,16@2"],
        ["hard"] * 3, tags=("standin",))
    for d in (16, 32):
        add("points", "pt-key", "box2", 32, 24, d, HALF + P(warping=3, pt1_use=2) + [KEY], ["hard"] * 2, tags=("standin",))
    add("points", "pt-interp1", "box2", 32, 24, 8, HALF + P(warping=3, pt1_use=2, frame_interp=1) + [KEY], ["hard"] * 2, tags=("standin",))
    add("points", "pt-interp3", "box2", 32, 24, 8, HALF + P(warping=3, pt1_use=2, frame_interp=3) + [KEY], ["hard"] * 2, tags=("standin",))
    add("points", "pt-bg-view", "box2", 32, 24, 8, HALF + P(warping=3, pt1_use=5, display=6) + [KEY], ["hard"] * 2, tags=("standin",))
    add("points", "pt-neg0", "box2", 32, 24, 8, HALF + P(warping=3, pt1_use=5) + ["pt1_pos=-0,-0@0;2,1@2"], ["hard"] * 2,
        tags=("standin", "negzero"))
    add("points", "pt-fast", "box2", 32, 24, 8, P(speed_percent=100, warping=3, pt1_use=2) + [KEY], ["hard"] * 2, tags=("standin",))
    for pos in ("-0.5,6", "-0.49,6", "31.5,6", "31.49,6", "6,-0.5", "6,23.5"):
        add("points", "pt-still-edge", "pan", 32, 24, 8, HALF + P(warping=3, pt1_use=5, pt1_pos=pos), ["hard"])
    add("points", "pt-tiny", "box1", 3, 3, 8, HALF + P(warping=3, pt1_use=5) + ["pt1_pos=1,1@0;2,1@2"], ["hard"], tags=("standin", "tiny"))
    add("points", "pt-wide", "box1", 40, 12, 8, HALF + P(warping=3, pt1_use=5) + ["pt1_pos=30,5@0;34,6@2"], ["hard"], tags=("standin",))
    add("points", "pt-tall", "box1", 12, 40, 8, HALF + P(warping=3, pt1_use=5) + ["pt1_pos=5,30@0;6,34@2"], ["hard"], tags=("standin",))

    # S15 Draw Geom on the Source display (G3). Points: four layers' choices at sub pixel positions
    DRAWN = P(pt1_use=2, pt1_pos="6.3,5.6", pt2_use=3, pt2_pos="14.5,9.5", pt3_use=4, pt3_pos="20.75,12.25", pt4_use=5,
              pt4_pos="3,18", pt5_use=5, pt5_pos="-0.5,0", pt1_color="255,0,0", pt2_color="0,255,0", pt3_color="1,2,3",
              pt4_color="255,255,255", pt5_color="0,0,0")
    for dg in (1, 2, 3, 4, 5, 6):
        add("geom", f"dg{dg}-n2", "box2", 32, 24, 8, P(display=2, draw_geom=dg) + DRAWN, ["hard"] * 2)
    for dg in (2, 3, 6):
        add("geom", f"dg{dg}-n0", "box1", 32, 24, 8, P(display=2, draw_geom=dg) + DRAWN)
    for dg in (2, 5):
        add("geom", f"dg{dg}-n3", "box3", 32, 24, 8, P(display=2, draw_geom=dg) + DRAWN, ["hard"] * 3)
    for d in (16, 32):
        for dg in (2, 4, 6):
            add("geom", f"dg{dg}-n2", "box2", 32, 24, d, P(display=2, draw_geom=dg) + DRAWN, ["hard"] * 2)
    # positions at the edges, outside, fractional, huge
    EDGE = [("0,0", "255,0,0"), ("31,23", "0,255,0"), ("-0.5,-0.5", "0,0,255"), ("-0.51,3", "255,255,0"), ("31.5,12", "255,0,255"),
            ("31.49,12", "0,255,255"), ("-1,-1", "9,9,9"), ("1000000,5", "100,100,100"), ("5,-1000000", "50,60,70"),
            ("0.4999,0.4999", "200,10,10"), ("1.5,1.5", "10,200,10"), ("2.5,2.5", "10,10,200")]
    for i0 in (0, 4, 8):
        p = []
        for j, (pos, col) in enumerate(EDGE[i0:i0 + 4]):
            n = j + 1
            p += [f"pt{n}_use=5", f"pt{n}_pos={pos}", f"pt{n}_color={col}"]
        for d in (8, 32):
            add("geom", f"edge{i0}", "box1", 32, 24, d, P(display=2, draw_geom=2) + p)
    allp = []
    for j, (pos, col) in enumerate(EDGE):
        allp += [f"pt{j + 1}_use={2 + j % 4}", f"pt{j + 1}_pos={pos}", f"pt{j + 1}_color={col}"]
    add("geom", "edge-all12", "box3", 32, 24, 16, P(display=2, draw_geom=2) + allp, ["hard"] * 3)
    add("geom", "overlap-order", "box1", 16, 12, 8, P(display=2, draw_geom=2, pt1_use=5, pt1_pos="6,5", pt1_color="255,0,0",
                                                        pt2_use=5, pt2_pos="7,5", pt2_color="0,0,255"))
    add("geom", "overlap-order-d32", "box1", 16, 12, 32, P(display=2, draw_geom=2, pt1_use=5, pt1_pos="6,5", pt1_color="255,0,0",
                                                          pt2_use=5, pt2_pos="7,5.5", pt2_color="0,0,255"))
    add("geom", "overlap-reverse", "box1", 16, 12, 8, P(display=2, draw_geom=2, pt2_use=5, pt2_pos="6,5", pt2_color="255,0,0",
                                                         pt1_use=5, pt1_pos="7,5", pt1_color="0,0,255"))
    add("geom", "alpha-src", "box1a", 16, 12, 8, P(display=2, draw_geom=2, pt1_use=5, pt1_pos="8,6"))
    add("geom", "alpha-src-d16", "box1a", 16, 12, 16, P(display=2, draw_geom=2, pt1_use=5, pt1_pos="8,6", pt1_color="1,2,254"))
    add("geom", "keyed-pos", "box1", 32, 24, 8, P(display=2, draw_geom=2, pt1_use=5) + ["pt1_pos=10,8@0;14,10@2"], [], tags=("standin",))
    add("geom", "keyed-pos-n2", "box2", 32, 24, 8, P(display=2, draw_geom=3, pt1_use=2) + ["pt1_pos=10,8@0;14,10@2"], ["hard"] * 2, tags=("standin",))
    add("geom", "disc-subpix", "flat", 16, 12, 8, P(display=2, draw_geom=2, pt1_use=5, pt1_pos="8.5,6.5", pt2_use=5, pt2_pos="3.25,2.75",
                                                      pt3_use=5, pt3_pos="12.1,9.9"))
    add("geom", "disc-subpix-d32", "specall", 16, 12, 32, P(display=2, draw_geom=2, pt1_use=5, pt1_pos="8.5,6.5", pt2_use=5, pt2_pos="3.25,2.75"),
        tags=("special",))
    add("geom", "disc-tiny", "box1", 2, 2, 8, P(display=2, draw_geom=2, pt1_use=5, pt1_pos="0.5,0.5"), tags=("tiny",))
    add("geom", "disc-1x1", "box1", 1, 1, 8, P(display=2, draw_geom=2, pt1_use=5, pt1_pos="0,0"), tags=("tiny",))
    add("geom", "disp1-no-draw", "box1", 32, 24, 8, P(display=1, draw_geom=2, pt1_use=5, pt1_pos="10,8"))

    # S16 wrap (Track RGB+A is the wrap flag) with layers and points (G6)
    for wp in (1, 2, 3, 4):
        add("wrap", f"wrap-w{wp}", "wrapcol", 24, 16, 8, HALF + P(warping=wp, track_rgb_a=1), ["seam"])
    add("wrap", "wrap-n2", "wrapcol", 24, 16, 8, HALF + P(warping=1, track_rgb_a=1), ["seam", "hard"])
    add("wrap", "wrap-box-w3", "box2", 24, 16, 8, HALF + P(warping=3, track_rgb_a=1), ["hard"] * 2)
    add("wrap", "wrap-bg", "wrapcol", 24, 16, 8, HALF + P(warping=1, track_rgb_a=1, display=6), ["seam"])
    add("wrap", "wrap-blur", "wrapcol", 24, 16, 8, P(speed_percent=200, warping=1, track_rgb_a=1, blur_compensation=2), ["seam"])
    add("wrap", "wrap-pt", "wrapcol", 24, 16, 8, HALF + P(warping=3, track_rgb_a=1, pt1_use=5) + ["pt1_pos=2,8@0;5,9@2"], ["seam"],
        tags=("standin",))
    add("wrap", "wrap-d32", "wrapcol", 24, 16, 32, HALF + P(warping=3, track_rgb_a=1), ["seam"])
    add("wrap", "wrap-d16", "wrapcol", 24, 16, 16, HALF + P(warping=2, track_rgb_a=1), ["seam"])
    add("wrap", "wrap-off", "wrapcol", 24, 16, 8, HALF + P(warping=3, track_rgb_a=0), ["seam"])

    # S17 mixes of valid values (seeded)
    rng = random.Random(20261030)
    for i in range(14):
        d = rng.choice((8, 8, 16, 32))
        w, h = rng.choice([(16, 12), (17, 13), (24, 16), (31, 21), (12, 20)])
        nl = rng.choice((1, 2, 2, 3))
        src = rng.choice([f"box{nl}", f"box{nl}", f"box{nl}f", f"box{nl}a"])
        p = P(speed_percent=rng.choice((25, 50, 66.5, 120, 200, -50)),
              warping=rng.choice((1, 2, 3, 4)), frame_interp=rng.choice((1, 2, 3)),
              display=rng.choice((1, 1, 3, 4, 5, 6)), main_bg_sensitivity=rng.choice((20, 70, 100)),
              blur_compensation=rng.choice((0, 0, 1.5)), track_rgb_a=rng.choice((0, 0, 1)))
        for k in range(1, nl + 1):
            p += P(**{f"fg{k}_matte_chan": rng.choice((1, 2, 3, 4)), f"fg{k}_sensitivity": rng.choice((10, 70, 100)),
                      f"fg{k}_inv_matte_shrink": rng.choice((0, 0, 0.75, 2, 5.5))})
        ms = [rng.choice(["chan", "alpha", "hard", "soft", "chan"]) for _ in range(nl)]
        add("mix", f"mix{i:02d}", src, w, h, d, p, ms)
    return g


# Out of contract candidates: values outside the registered ranges or popup choices, and layer
# conditions where the original is not defined or the plugin shows an error fill. vfx refuses some;
# every one is listed and none is ever counted as a pass or a failure.
OC_PARAMS = [
    ("fg1_matte_chan", "0"), ("fg1_matte_chan", "5"), ("fg2_matte_chan", "5"), ("fg3_matte_chan", "0"),
    ("fg1_sensitivity", "100.01"), ("fg1_sensitivity", "-0.01"), ("fg2_sensitivity", "101"), ("fg3_sensitivity", "-1"),
    ("fg1_inv_matte_shrink", "10.01"), ("fg1_inv_matte_shrink", "-0.01"), ("fg1_inv_matte_shrink", "-0.5"),
    ("fg1_inv_matte_shrink", "-1"), ("fg1_inv_matte_shrink", "-2.5"), ("fg2_inv_matte_shrink", "-1"),
    ("fg3_inv_matte_shrink", "11"), ("display", "0"), ("display", "7"), ("draw_geom", "0"), ("draw_geom", "7"),
    ("pt1_use", "0"), ("pt1_use", "6"), ("pt12_use", "6"), ("pt1_color", "256,0,0"), ("pt1_color", "-1,0,0"),
    ("pt1_color", "0,0"), ("pt1_pos", "nan,0"), ("pt1_pos", "0,inf"), ("main_bg_sensitivity", "100.01"),
    ("main_bg_track", "0"), ("fg1_track", "0"), ("warping", "5"), ("warping", "0"),
]


def vfx_check(vfx, c):
    args = [vfx, "apply", c["effect"], "--width", str(c["width"]), "--height", str(c["height"]), "--depth", str(c["depth"]),
            "--fps", c["fps"], "--frames", str(c["frames"]), "--check"]
    for p in c["params"]:
        args += ["--param", p]
    for lid in c.get("layers", {}):
        args += ["--layer", f"{lid}=/dev/null"]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode == 0:
        return None
    return (r.stderr.strip() or f"exit {r.returncode}").splitlines()[0]


def input_entries(cases, files):
    """The `inputs` list of the fx-parity format: every clip file the cases name (footage and matte
    layers), once, sorted by file. A matte layer file has the source name `matte`."""
    seen = {}
    for c in cases:
        for f, src in [(c["input"], c["source"])] + [(f, "matte") for f in c.get("layers", {}).values()]:
            if f not in seen:
                w, h, d, n = files[f][-4:]
                seen[f] = dict(file=f, source=src, width=w, height=h, depth=d, pix_fmt=PIX[d], frames=n)
    return [seen[k] for k in sorted(seen)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vfx")
    a = ap.parse_args()
    out = a.out
    os.makedirs(os.path.join(out, "inputs"), exist_ok=True)
    g = build_grid()

    oc = []
    base_p = ["speed_percent=50", "warping=3"]
    for k, v in OC_PARAMS:
        params = [p for p in base_p if p.split("=")[0] != k] + [f"{k}={v}"]
        if k.startswith("pt1_") and k != "pt1_use":
            params.append("pt1_use=5")
        oc.append(dict(id=f"{EFFECT}.oc-{k}-{v}", effect=EFFECT, label=f"oc-{k}", params=params, depth=8, width=24, height=16,
                       source="box2", input="inputs/box2-24x16.d8.f3.rgba", reference=f"{EFFECT}.oc-{k}-{v}.rgba",
                       kind="temporal", frames=3, fps="24",
                       layers={"fg1_matte": "inputs/m-hard-0-24x16.d8.f3.rgba", "fg2_matte": "inputs/m-hard-1-24x16.d8.f3.rgba"},
                       reason="outside the registered range or popup choices, or not a parameter of retime-fgbg"))
    g.files["inputs/box2-24x16.d8.f3.rgba"] = ("scene", "box2", 24, 16, 8, 3)
    for i in (0, 1):
        g.files[f"inputs/m-hard-{i}-24x16.d8.f3.rgba"] = ("matte", "hard", i, False, 24, 16, 8, 3)
    # layer conditions
    g.files["inputs/m-hard-0-25x16.d8.f3.rgba"] = ("matte", "hard", 0, False, 25, 16, 8, 3)
    g.files["inputs/m-hard-0-24x16.d8.f1.rgba"] = ("matte", "hard", 0, False, 24, 16, 8, 1)
    oc.append(dict(id=f"{EFFECT}.oc-matte-wrong-size", effect=EFFECT, label="oc-matte-size", params=base_p, depth=8, width=24, height=16,
                   source="box2", input="inputs/box2-24x16.d8.f3.rgba", reference=f"{EFFECT}.oc-matte-wrong-size.rgba", kind="temporal",
                   frames=3, fps="24", layers={"fg1_matte": "inputs/m-hard-0-25x16.d8.f3.rgba"},
                   reason="matte layer of another size than the footage (the plugin shows an error fill)"))
    oc.append(dict(id=f"{EFFECT}.oc-matte-short", effect=EFFECT, label="oc-matte-short", params=base_p, depth=8, width=24, height=16,
                   source="box2", input="inputs/box2-24x16.d8.f3.rgba", reference=f"{EFFECT}.oc-matte-short.rgba", kind="temporal",
                   frames=3, fps="24", layers={"fg1_matte": "inputs/m-hard-0-24x16.d8.f1.rgba"},
                   reason="matte clip shorter than the footage: frames past its end are an unprovided host read"))
    oc.append(dict(id=f"{EFFECT}.oc-matte-depth-mismatch-note", effect=EFFECT, label="oc-note", params=base_p, depth=8, width=24, height=16,
                   source="box2", input="inputs/box2-24x16.d8.f3.rgba", reference=f"{EFFECT}.oc-matte-depth-mismatch-note.rgba",
                   kind="temporal", frames=3, fps="24", layers={"fg1_matte": "inputs/m-hard-0-24x16.d8.f3.rgba"},
                   reason="control case: a valid render listed with the candidates so their verdicts can be compared"))

    # write the input files
    scene_cache = {}
    for rel, spec in sorted(g.files.items()):
        path = os.path.join(out, rel)
        if os.path.exists(path):
            continue
        if spec[0] == "scene":
            _, src, w, h, d, n = spec
            fn = SOURCES[src]
        else:
            _, kind, box, fast, w, h, d, n = spec
            mfn = matte(kind, box, fast, d)
            fn = (lambda m: (lambda x, y, f, w_, h_, depth: m(x, y, f, w_, h_)))(mfn)
        with open(path, "wb") as fh:
            fh.write(render_clip(fn, w, h, d, n))

    # a case whose source or matte kind needs another depth cannot exist: asserted here
    for c in g.cases:
        if c["source"] in FLOAT_ONLY and c["depth"] != 32:
            raise SystemExit(f"{c['id']}: float only source at {c['depth']} bpc")
        if c["source"] in INT_ONLY and c["depth"] == 32:
            raise SystemExit(f"{c['id']}: integer only source at 32 bpc")
        for lid, f in c.get("layers", {}).items():
            kind = f.split("/m-")[1].split("-")[0]
            if kind in FLOAT_MATTES and c["depth"] != 32:
                raise SystemExit(f"{c['id']}: float only matte at {c['depth']} bpc")

    refused = []
    if a.vfx:
        keep = []
        for c in g.cases:
            r = vfx_check(a.vfx, c)
            if r is None:
                keep.append(c)
            else:
                c = dict(c); c["reason"] = "vfx --check: " + r
                refused.append(c)
        g.cases = keep
    oc_checked = []
    for c in oc:
        c = dict(c)
        c["vfx_verdict"] = (vfx_check(a.vfx, c) or "accepted by --check") if a.vfx else "not checked"
        oc_checked.append(c)

    ts = dict(format=1, ffmpeg="none (raw inputs written by gen_grid.py)", inputs=input_entries(g.cases, g.files), cases=g.cases)
    with open(os.path.join(out, "cases.json"), "w") as fh:
        json.dump(ts, fh, indent=1)
    with open(os.path.join(out, "out_of_contract.json"), "w") as fh:
        json.dump(dict(format=1, note="candidates, never counted as passes or failures", cases=oc_checked + refused), fh, indent=1)
    keep_ids = {c["id"] for c in g.cases}
    with open(os.path.join(out, "grid.tsv"), "w") as fh:
        fh.write("id\tblock\ttags\n")
        for cid, block, tags in g.rows:
            if cid in keep_ids:
                fh.write(f"{cid}\t{block}\t{tags}\n")
    print(f"wrote {out}/cases.json: {len(g.cases)} cases, {len(g.files)} files; out of contract listed: {len(oc_checked) + len(refused)}")


if __name__ == "__main__":
    main()
