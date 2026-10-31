#!/usr/bin/env python3
"""Grid generator of the retime-time parity module.

Writes an fx-parity style test set (format 1): DIR/cases.json and DIR/inputs/*.raw frames, plus
DIR/grid.json (group, class and note of every case) and DIR/out_of_contract.json (candidates the
plugin or the rewrite should refuse, never counted as passes or failures).

Scope: effect `retime`, `vectors=1` (No Vectors) in every case; time_remap_mode, speed_percent,
frame_num, in_fps_is_out_fps, input_frame_rate, src_mark_segments, input_fields, frame_interp, at
8, 16 and 32 bpc. Constant parameter values are class `value`; a keyed curve depends on the host's
keyframe interpolation, so those cases are class `logic_parity` (the same labelled stand-in on both
sides). Keyed cases with Bezier handles (NOT_GRADED) are not in cases.json: the host computes the
handles, so they are not part of the bar; DIR/not_graded.json lists them (every other case keeps
its id, size and source, because the rotation counter still counts them).

Usage: python3 gen_grid.py OUT_DIR
Deterministic: the same script writes the same bytes everywhere (own PRNG, no ffmpeg).
"""
import json
import os
import struct
import sys
from array import array

PIX = {8: "rgba", 16: "rgba64le", 32: "gbrapf32le"}
M64 = (1 << 64) - 1
EFFECT = "retime"


def fnv(text):
    h = 0x811C9DC5
    for b in text.encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


class Rng:
    def __init__(self, seed):
        self.s = seed & M64

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & M64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)


def f32bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def lev(q, depth):
    """A 16 bit level q (0..65535) in the sample domain of the depth (f32 as bit pattern)."""
    if depth == 8:
        return (q * 255 + 32767) // 65535
    if depth == 16:
        return q
    return f32bits(q / 65535.0)


SPECIAL_BITS = [
    0x00000000,  # 0
    0x80000000,  # -0
    0x00000001,  # smallest denormal
    0x007FFFFF,  # largest denormal
    0x00800000,  # FLT_MIN
    0x3F800000,  # 1
    0x40000000,  # 2, above 1
    0x7F7FFFFF,  # FLT_MAX
    0x7F800000,  # +inf
    0xFF800000,  # -inf
    0x7FC00000,  # NaN
    0x3F000000,  # 0.5
    0xBF800000,  # -1
    0x3DCCCCCD,  # 0.1
    0x80000001,  # -smallest denormal
    0x807FFFFF,  # -largest denormal
    0x80800000,  # -FLT_MIN
    0xFF7FFFFF,  # -FLT_MAX
    0xBF000000,  # -0.5
]
ALPHA_BITS = [f32bits(v) for v in (1.0, 0.5, 0.25, 0.75, 0.0)]
EXT_INT = {8: [0, 1, 127, 128, 254, 255], 16: [0, 1, 32767, 32768, 65534, 65535]}
EXT_F32 = [f32bits(v) for v in (0.0, 1.0, 0.5, 0.99999994, 1.0000001, 0.001)]
ONE = {8: 255, 16: 65535, 32: f32bits(1.0)}

SOURCES_ALL = ["noise", "ident", "moving", "alphaslide", "extreme"]
SOURCES_F32 = ["special32", "special32o", "special32a"]


def planes(kind, depth, w, h, f, seed):
    """Four sample lists R, G, B, A (ints; f32 as bit patterns) of frame f, row major."""
    n = w * h
    R, G, B, A = ([0] * n for _ in range(4))
    rng = Rng(seed * 1000003 + f * 7919 + 17)
    for y in range(h):
        for x in range(w):
            i = y * w + x
            xs = x * 65535 // max(w - 1, 1)
            ys = y * 65535 // max(h - 1, 1)
            if kind == "noise":
                q = [rng.next() & 0xFFFF for _ in range(3)]
                r = rng.next() % 10
                qa = 65535 if r < 6 else (0 if r == 6 else rng.next() & 0xFFFF)
                R[i], G[i], B[i], A[i] = lev(q[0], depth), lev(q[1], depth), lev(q[2], depth), lev(qa, depth)
            elif kind == "ident":
                base = (f * 40503 + 977 * seed) & 0xFFFF
                qs = [(base + c * 21845 + x * 131 + y * 257) & 0xFFFF for c in range(3)]
                R[i], G[i], B[i], A[i] = lev(qs[0], depth), lev(qs[1], depth), lev(qs[2], depth), ONE[depth]
            elif kind == "moving":
                cx = (f * max(w, 2) // 8) % max(w, 1)
                rad = max(min(w, h) // 2, 1)
                inside = (x - cx) ** 2 + (y - h // 2) ** 2 <= rad * rad
                qr = (xs + f * 4000) & 0xFFFF
                qg = (ys + f * 2500) & 0xFFFF
                qb = 60000 if inside else (xs + ys) // 2
                R[i], G[i], B[i], A[i] = lev(qr, depth), lev(qg, depth), lev(qb, depth), ONE[depth]
            elif kind == "alphaslide":
                qa = min(((x + f * max(w // 4, 1)) * 65535) // (2 * max(w, 1)), 65535)
                if (x + f) % 7 == 0:
                    qa = 0
                R[i], G[i], B[i], A[i] = lev(xs, depth), lev(ys, depth), lev((f * 5000) & 0xFFFF, depth), lev(qa, depth)
            elif kind == "extreme":
                tab = EXT_INT[depth] if depth != 32 else EXT_F32
                vals = [tab[(x * 3 + y * 5 + f * 7 + c * 11 + seed) % 6] for c in range(4)]
                R[i], G[i], B[i], A[i] = vals
            elif kind in ("special32", "special32o", "special32a"):
                k = len(SPECIAL_BITS)
                vals = [SPECIAL_BITS[(x * 3 + y * 7 + f * 5 + c * 3 + seed) % k] for c in range(4)]
                if kind == "special32o":
                    vals[3] = ONE[32]
                elif kind == "special32a":
                    vals[3] = ALPHA_BITS[(x + y * 2 + f) % len(ALPHA_BITS)]
                R[i], G[i], B[i], A[i] = vals
            elif kind == "negzero32":
                tab = [0x80000000, 0x80000001, 0x80000002, 0x00000000, 0x00000001, 0x807FFFFF]
                vals = [tab[(x * 2 + y + f + c) % 6] for c in range(3)] + [ONE[32]]
                R[i], G[i], B[i], A[i] = vals
            else:
                raise SystemExit("unknown source " + kind)
    return R, G, B, A


def clip_bytes(kind, depth, w, h, frames, seed):
    out = bytearray()
    for f in range(frames):
        R, G, B, A = planes(kind, depth, w, h, f, seed)
        if depth == 8:
            buf = bytearray(w * h * 4)
            buf[0::4], buf[1::4], buf[2::4], buf[3::4] = bytes(R), bytes(G), bytes(B), bytes(A)
            out += buf
        elif depth == 16:
            a = array("H", [0] * (w * h * 4))
            a[0::4], a[1::4], a[2::4], a[3::4] = array("H", R), array("H", G), array("H", B), array("H", A)
            if sys.byteorder == "big":
                a.byteswap()
            out += a.tobytes()
        else:
            a = array("I", G + B + R + A)  # gbrapf32le: G plane, B plane, R plane, A plane
            if sys.byteorder == "big":
                a.byteswap()
            out += a.tobytes()
    return bytes(out)


SIZES = [(1, 1), (2, 1), (1, 3), (2, 3), (5, 7), (16, 12), (37, 23), (64, 48)]
SPEC_SIZES = [(4, 3), (5, 2), (3, 3), (7, 5)]


NOT_GRADED = {("keyed", "fn-handles"), ("keyed", "sp-handles")}
NOT_GRADED_REASON = "host-side Bezier interpolation, not part of the bar"


class Grid:
    def __init__(self):
        self.cases = []
        self.meta = {}
        self.not_graded = []
        self.n = 0  # rotation counter over settings

    def add(self, group, label, params, depths=(8, 16, 32), frames=12, fps="24", cls="value",
            source=None, size=None, note=""):
        i = self.n
        self.n += 1
        w, h = size or SIZES[i % len(SIZES)]
        src = source or SOURCES_ALL[(i * 3 + 1) % len(SOURCES_ALL)]
        full = ["vectors=1"] + list(params)
        for d in depths:
            key = "|".join([group, label, ",".join(full), str(d), str(w), str(h), src, str(frames), fps])
            cid = "%s.%s-%s.%s-%dx%d.d%d.%08x" % (
                EFFECT, group, safe(label), src, w, h, d, fnv(key))
            case = {
                "id": cid, "effect": EFFECT, "label": "%s-%s" % (group, safe(label)), "params": full,
                "depth": d, "width": w, "height": h, "source": src,
                "input": "inputs/%s-%dx%d.d%d.f%d.%s" % (src, w, h, d, frames, PIX[d]),
                "reference": "%s.%s" % (cid, PIX[d]), "kind": "temporal", "frames": frames, "fps": fps,
            }
            if (group, label) in NOT_GRADED:
                self.not_graded.append(dict(case, group=group, **{"class": cls}))
                continue
            self.cases.append(case)
            self.meta[cid] = {"group": group, "class": cls, "note": note}

    def rot(self, k):
        """Depth tuple for rotation: one depth per setting, all three over successive settings."""
        return ((8, 16, 32)[(self.n + k) % 3],)


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def snap_rate(r):
    for t in (29.97, 59.94, 23.976):
        if abs(f32(r - t)) < 2e-5:
            r = t
    return r


def mark_round_keys(ifr, fps, lo=2, hi=9, n=2):
    """Key frames (as decimal strings) that sit exactly on the tick a marker lookup rounds up to:
    marker(d) samples trunc(f32(d * dur) + 0.5), dur = f32(100 * out / in) for 100 ticks a frame.
    Returns up to n (d, tick) pairs where that differs from trunc(f32(d * dur))."""
    dur = f32((100 * snap_rate(float(fps))) / snap_rate(f32(float(ifr))))
    out = []
    for d in range(lo, hi + 1):
        prod = f32(d * dur)
        if int(prod) != int(prod + 0.5):
            out.append((d, int(prod + 0.5)))
    return out[:n]


def ratio_sensitive_frame_nums(ifr, fps, n):
    """Decimal frame numbers whose source position, ratio in / out applied in f64 as the rewrite does,
    differs from the same product done in f32, with a fraction away from 0 and 1."""
    r = snap_rate(f32(float(ifr))) / snap_rate(float(fps))
    out = []
    for k in range(10, 1100):
        x = f32(k / 100)
        p1, p2 = f32(r * x), f32(f32(r) * x)
        fr = p1 - int(p1)
        if p1 != p2 and 0.02 < fr < 0.98 and p1 < 11:
            out.append("%d.%02d" % (k // 100, k % 100))
    step = max(len(out) // n, 1)
    return out[::step][:n]


def safe(text):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def build():
    g = Grid()
    P = lambda **kw: ["%s=%s" % kv for kv in kw.items()]

    # speed mode, constant speed, default Blend (interp left at its default 2)
    for s in ["100", "50", "25", "33.3", "66.67", "150", "200", "300", "0", "-0", "-50", "-100", "10000",
              "-10000", "0.01", "99.99", "100.1", "99.9", "12.5", "40", "100.5", "99.5", "101", "75"]:
        g.add("speed", s, ["speed_percent=" + s])
    # speed x frame_interp Nearest and Weighted Blend, plus an explicit Blend
    for s in ["50", "33.3", "150", "100.1", "99.9", "-50", "25", "66.67", "0", "100"]:
        for ip, nm in ((1, "near"), (3, "wtd")):
            g.add("interp", "%s-%s" % (s, nm), ["speed_percent=" + s, "frame_interp=%d" % ip])
    g.add("interp", "50-blend", ["speed_percent=50", "frame_interp=2"])
    g.add("interp", "33.3-blend", ["speed_percent=33.3", "frame_interp=2"])

    # frame number mode, constants (positions around the 0.005 snap, clip edges, negatives)
    fn = ["0", "-0", "3", "3.5", "7.25", "11", "11.5", "0.999", "1.004", "1.006", "2.994", "2.996", "-0.5",
          "-1", "-3", "5.5", "200000", "-200000", "0.0049", "11.9", "12", "-0.25", "-0.75", "4.5", "6.995"]
    for k, v in enumerate(fn):
        extra = [] if k % 3 == 0 else (["frame_interp=1"] if k % 3 == 1 else ["frame_interp=3"])
        g.add("framenum", v, ["time_remap_mode=2", "frame_num=" + v] + extra)

    # keyed curves (host interpolation stand-in): logic_parity
    keyed = [
        ("fn-ramp-up", ["time_remap_mode=2", "frame_num=0@0;11@11"]),
        ("fn-ramp-down", ["time_remap_mode=2", "frame_num=11@0;0@11"]),
        ("fn-slow", ["time_remap_mode=2", "frame_num=0@0;5.5@11"]),
        ("fn-flat-ease", ["time_remap_mode=2", "frame_num=2@0e;2@11e"]),
        ("fn-hold", ["time_remap_mode=2", "frame_num=0@0h;6@5h;11@11"]),
        ("fn-handles", ["time_remap_mode=2", "frame_num=0@0o0/50;11@11i0/50"]),
        ("fn-ease-mix", ["time_remap_mode=2", "frame_num=0@0e;8@11", "frame_interp=1"]),
        ("fn-unordered", ["time_remap_mode=2", "frame_num=11@11;0@0;4@5", "frame_interp=3"]),
        ("sp-down", ["speed_percent=100@0;50@11"]),
        ("sp-ease", ["speed_percent=100@0e;25@11e"]),
        ("sp-hold", ["speed_percent=200@0h;30@6h;100@11"]),
        ("sp-steps", ["speed_percent=100@0;100@5;200@8"]),
        ("sp-reverse", ["speed_percent=100@0;-100@6;100@11"]),
        ("sp-handles", ["speed_percent=100@0o0/50;50@11i-30/50h"]),
        ("sp-zero", ["speed_percent=100@0;0@4;0@8;100@11", "frame_interp=1"]),
        ("sp-fract-key", ["speed_percent=100@0;50@5.5;150@11", "frame_interp=3"]),
        ("sp-up", ["speed_percent=50@0;200@11"]),
        ("sp-ntsc", ["speed_percent=100@0;40@11"]),
    ]
    for k, (lab, params) in enumerate(keyed):
        fps = "30000/1001" if lab == "sp-ntsc" else "24"
        deps = (8, 16, 32) if k < 6 else g.rot(k)
        g.add("keyed", lab, params, depths=deps, cls="logic_parity", fps=fps,
              note="keyframe interpolation is host side; same stand-in on both sides")

    # rates: In FPS is Out FPS off with a typed Input: Frame Rate
    rates = [
        ("29.969999313354492", "24"), ("12", "24"), ("15", "30"), ("24", "24"), ("25", "24"),
        ("29.97", "24"), ("29.97", "30000/1001"), ("29.97003", "30000/1001"), ("30", "24"),
        ("30", "30000/1001"), ("48", "24"), ("60", "24"), ("60", "60"), ("23.976", "24"),
        ("23.976024", "24000/1001"), ("59.94", "60"), ("59.94006", "60000/1001"), ("100", "24"),
        ("1000", "24"), ("0.10000000149011612", "24"), ("0.5", "24"), ("2", "24"), ("29.97001", "24"),
        ("29.96999", "24"), ("29.97002", "24"), ("29.97003", "24"), ("29.96998", "24"), ("24", "25"),
        ("24", "48"), ("24", "60"), ("7.5", "15"), ("30", "29.97"), ("29.97", "30"), ("60", "59.94"),
        ("24", "23.976"), ("23.976", "24"), ("25", "24.95"), ("29.95", "30"), ("29.91", "30"),
        ("29.9", "30"), ("33", "30"),
    ]
    for k, (ifr, fps) in enumerate(rates):
        sp = "100" if k % 2 == 0 else "50"
        extra = [] if k % 3 == 0 else (["frame_interp=1"] if k % 3 == 1 else ["frame_interp=3"])
        params = ["in_fps_is_out_fps=0", "input_frame_rate=" + ifr, "speed_percent=" + sp] + extra
        g.add("rate", "%s-at-%s-s%s" % (ifr, fps.replace("/", "_"), sp), params, depths=g.rot(k), fps=fps)
    for k, (ifr, fps, fnv_) in enumerate([("24", "24", "5.5"), ("12", "24", "3"), ("60", "24", "9.25"),
                                         ("30", "29.97", "6")]):
        g.add("rate", "fn%s-%s-at-%s" % (fnv_, ifr, fps.replace("/", "_")),
              ["in_fps_is_out_fps=0", "input_frame_rate=" + ifr, "time_remap_mode=2", "frame_num=" + fnv_],
              depths=g.rot(k), fps=fps)
    # In FPS is Out FPS on (the default) at other output rates
    for k, fps in enumerate(["25", "30000/1001", "60", "24000/1001", "15", "120", "29.97", "23.976"]):
        sp = "100" if k % 2 == 0 else "50"
        g.add("rate", "inout-on-%s-s%s" % (fps.replace("/", "_"), sp),
              ["in_fps_is_out_fps=1", "input_frame_rate=60", "speed_percent=" + sp], depths=g.rot(k), fps=fps)

    # input fields
    for k, (fld, sp) in enumerate([(2, "100"), (3, "100"), (2, "50"), (3, "50"), (2, "25"), (3, "33.3"),
                                    (2, "150"), (3, "-50"), (2, "66.67"), (3, "200")]):
        extra = [] if k % 3 == 0 else (["frame_interp=1"] if k % 3 == 1 else ["frame_interp=3"])
        g.add("fields", "f%d-s%s" % (fld, sp), ["input_fields=%d" % fld, "speed_percent=" + sp] + extra,
              depths=g.rot(k))
    g.add("fields", "f2-rate12", ["input_fields=2", "in_fps_is_out_fps=0", "input_frame_rate=12", "speed_percent=100"],
          depths=g.rot(1))
    g.add("fields", "f3-rate60", ["input_fields=3", "in_fps_is_out_fps=0", "input_frame_rate=60", "speed_percent=50"],
          depths=g.rot(2))
    g.add("fields", "f3-fn3.5", ["input_fields=3", "time_remap_mode=2", "frame_num=3.5"], depths=g.rot(0))
    g.add("fields", "f2-fn7.25-near", ["input_fields=2", "time_remap_mode=2", "frame_num=7.25", "frame_interp=1"],
          depths=g.rot(1))
    g.add("fields", "f1-explicit", ["input_fields=1", "speed_percent=50"], depths=g.rot(2))

    # source marks, constant and keyed (keys count source frames)
    for k, (m, sp, ip) in enumerate([(1, "50", 2), (2, "50", 2), (3, "50", 1), (4, "50", 2), (5, "50", 1),
                                      (4, "33.3", 3), (5, "33.3", 3), (5, "50", 2), (4, "50", 1), (5, "50", 3)]):
        g.add("marks", "m%d-s%s-i%d" % (m, sp, ip),
              ["src_mark_segments=%d" % m, "speed_percent=" + sp, "frame_interp=%d" % ip], depths=g.rot(k))
    marks_keyed = [
        ("cut-2-4", ["src_mark_segments=1@0;2@4", "speed_percent=50"]),
        ("cut-3way", ["src_mark_segments=1@0;2@5;3@8", "speed_percent=33.3"]),
        ("cut-down", ["src_mark_segments=4@0;1@5", "speed_percent=50"]),
        ("fb-then-1", ["src_mark_segments=5@0;1@6", "speed_percent=50"]),
        ("fb-span", ["src_mark_segments=1@0;5@3;1@6", "speed_percent=40"]),
        ("same-label", ["src_mark_segments=2@0;2@4", "speed_percent=50"]),
        ("nearest-span", ["src_mark_segments=1@0;4@3;1@7", "speed_percent=66.67", "frame_interp=3"]),
        ("fn-cut", ["src_mark_segments=1@0;2@5", "time_remap_mode=2", "frame_num=4.5"]),
        ("fn-neg-cut", ["src_mark_segments=1@0;3@1", "time_remap_mode=2", "frame_num=-0.5"]),
        ("high-to-low", ["src_mark_segments=3@0;2@4;1@8", "speed_percent=50", "frame_interp=1"]),
    ]
    for k, (lab, params) in enumerate(marks_keyed):
        g.add("marks", lab, params, depths=g.rot(k), cls="logic_parity",
              note="keyframe interpolation is host side; same stand-in on both sides")

    # marker sample time rounding: keys on the tick a marker lookup rounds up to (unequal rates)
    k = 0
    for ifr, fps in [("29.97", "24"), ("29.97", "25"), ("23.976", "25"), ("33", "24")]:
        for d, tick in mark_round_keys(ifr, fps):
            key = "%d.%02d" % (tick // 100, tick % 100)
            g.add("marks", "round-%s-at-%s-d%d" % (ifr, fps, d),
                  ["in_fps_is_out_fps=0", "input_frame_rate=" + ifr, "speed_percent=100",
                   "src_mark_segments=1@0;2@" + key], depths=g.rot(k), fps=fps, cls="logic_parity",
                  note="key on the rounded marker tick; keyframe interpolation is host side")
            k += 1
    # frame number with rate ratios that are not exact in f32: values where applying the ratio in f32
    # instead of f64 moves the position by an ulp and the position is not near a frame
    k = 0
    for ifr, fps in [("29.97", "24"), ("29.97", "25"), ("23.976", "25"), ("48", "25")]:
        for fnv_ in ratio_sensitive_frame_nums(ifr, fps, 3):
            g.add("rate", "fn%s-%s-at-%s" % (fnv_, ifr, fps), ["in_fps_is_out_fps=0", "input_frame_rate=" + ifr,
                  "time_remap_mode=2", "frame_num=" + fnv_], depths=(32,), fps=fps, source="noise", size=(5, 7))
            k += 1
        fnv_ = ratio_sensitive_frame_nums(ifr, fps, 4)[-1]
        g.add("rate", "fn%s-%s-at-%s-d16" % (fnv_, ifr, fps), ["in_fps_is_out_fps=0", "input_frame_rate=" + ifr,
              "time_remap_mode=2", "frame_num=" + fnv_], depths=(16,), fps=fps, source="noise", size=(5, 7))

    # float special values at 32 bpc (alpha varies, alpha 1, normal alphas)
    for si, src in enumerate(SOURCES_F32):
        for k, sp in enumerate(["100", "50", "33.3", "150", "25", "66.67"]):
            ip = [1, 2, 3][(si + k) % 3]
            g.add("special", "%s-s%s-i%d" % (src, sp, ip), ["speed_percent=" + sp, "frame_interp=%d" % ip],
                  depths=(32,), source=src, size=SPEC_SIZES[(si + k) % len(SPEC_SIZES)])
    for ip in (1, 2, 3):
        g.add("special", "fn3.5-i%d" % ip, ["time_remap_mode=2", "frame_num=3.5", "frame_interp=%d" % ip],
              depths=(32,), source="special32a", size=(4, 3))
    for src in ("special32", "special32a"):
        g.add("special", "%s-neg-half" % src, ["time_remap_mode=2", "frame_num=-0.5"], depths=(32,), source=src,
              size=(4, 3))
        g.add("special", "%s-past-end" % src, ["speed_percent=300"], depths=(32,), source=src, size=(5, 2))

    # -0 and negative denormals in neighbouring frames (sign of zero through the blend sum)
    for k, (sp, ip) in enumerate([("50", 2), ("33.3", 2), ("66.67", 3), ("25", 2), ("150", 3), ("50", 1)]):
        g.add("special", "negzero-s%s-i%d" % (sp, ip), ["speed_percent=" + sp, "frame_interp=%d" % ip],
              depths=(32,), source="negzero32", size=(4, 3))

    # extreme integer and unit float values, extrapolation
    for k, (sp, ip) in enumerate([("50", 2), ("33.3", 1), ("66.67", 3), ("100.1", 2), ("150", 2), ("100.5", 3),
                                    ("99.9", 1)]):
        g.add("extreme", "s%s-i%d" % (sp, ip), ["speed_percent=" + sp, "frame_interp=%d" % ip], depths=(8, 16, 32),
              source="extreme", size=SIZES[3 + k % 4])
    for v in ("-0.5", "-0.25", "-0.9"):
        g.add("extreme", "fn%s" % v, ["time_remap_mode=2", "frame_num=" + v], depths=(8, 16, 32), source="extreme",
              size=(5, 7))

    # clip length
    for lab, fr, params, size in [
        ("len1-s100", 1, ["speed_percent=100"], (3, 2)), ("len1-s50", 1, ["speed_percent=50"], (3, 2)),
        ("len2-s50", 2, ["speed_percent=50"], (3, 2)), ("len3-s33.3", 3, ["speed_percent=33.3"], (5, 3)),
        ("len24-s33.3", 24, ["speed_percent=33.3"], (7, 5)), ("len48-s66.67", 48, ["speed_percent=66.67"], (3, 2)),
        ("len24-fn", 24, ["time_remap_mode=2", "frame_num=17.3"], (5, 3)),
        ("len1-fn", 1, ["time_remap_mode=2", "frame_num=0.5"], (3, 2)),
    ]:
        g.add("length", lab, params, frames=fr, size=size)

    # long clips at tiny sizes: sums over many half frame steps
    g.add("long", "120-s33.3", ["speed_percent=33.3"], frames=120, size=(2, 2))
    g.add("long", "120-s66.67", ["speed_percent=66.67", "frame_interp=3"], frames=120, size=(3, 2))
    g.add("long", "120-s12.5", ["speed_percent=12.5"], frames=120, size=(2, 2), depths=(8, 32))
    g.add("long", "120-s0.7", ["speed_percent=0.7", "frame_interp=1"], frames=120, size=(2, 2), depths=(32,))
    g.add("long", "120-s99.7", ["speed_percent=99.7"], frames=120, size=(2, 2), depths=(8,))
    g.add("long", "120-s100.3", ["speed_percent=100.3"], frames=120, size=(2, 2), depths=(16,))
    g.add("long", "120-s7", ["speed_percent=7"], frames=120, size=(2, 2), depths=(16,))
    g.add("long", "300-s33.3", ["speed_percent=33.3"], frames=300, size=(1, 1), depths=(32,))
    g.add("long", "300-s7", ["speed_percent=7", "frame_interp=3"], frames=300, size=(1, 1), depths=(8,))
    g.add("long", "120-ramp", ["speed_percent=100@0;25@60;100@119"], frames=120, size=(2, 2), depths=(32,),
          cls="logic_parity", note="keyframe interpolation is host side; same stand-in on both sides")
    g.add("long", "120-ramp-ease", ["speed_percent=100@0e;40@60e;100@119e"], frames=120, size=(2, 2), depths=(8,),
          cls="logic_parity", note="keyframe interpolation is host side; same stand-in on both sides")
    # 6000 frames at 1x1: the documented rate snapping effect shows from output frame 5127 on
    for ifr in ("29.97", "29.97003"):
        g.add("long6000", "rate%s" % ifr, ["in_fps_is_out_fps=0", "input_frame_rate=" + ifr, "speed_percent=100"],
              frames=6000, size=(1, 1), depths=(8,), fps="30000/1001", source="ident",
              note="long: 6000 output frames at 1x1; phase B may run it separately")

    # larger odd size
    for sp in ("50", "150"):
        g.add("big", "129x71-s" + sp, ["speed_percent=" + sp], size=(129, 71), source="moving")
    return g


def out_of_contract():
    """Candidates outside the registered ranges or popup choices: listed, never graded."""
    base = ["vectors=1"]
    items = [
        ("speed above max", ["speed_percent=10001"]),
        ("speed below min", ["speed_percent=-10001"]),
        ("frame_num above max", ["time_remap_mode=2", "frame_num=200001"]),
        ("frame_num below min", ["time_remap_mode=2", "frame_num=-200001"]),
        ("input_frame_rate below min", ["in_fps_is_out_fps=0", "input_frame_rate=0.05"]),
        ("input_frame_rate double 0.1", ["in_fps_is_out_fps=0", "input_frame_rate=0.1"]),
        ("input_frame_rate above max", ["in_fps_is_out_fps=0", "input_frame_rate=1000.5"]),
        ("time_remap_mode 0", ["time_remap_mode=0"]),
        ("time_remap_mode 3", ["time_remap_mode=3"]),
        ("frame_interp 0", ["frame_interp=0"]),
        ("frame_interp 4", ["frame_interp=4"]),
        ("input_fields 0", ["input_fields=0"]),
        ("input_fields 4", ["input_fields=4"]),
        ("src_mark_segments 0", ["src_mark_segments=0"]),
        ("src_mark_segments 6", ["src_mark_segments=6"]),
        ("in_fps_is_out_fps 2", ["in_fps_is_out_fps=2"]),
        ("popup by name", ["frame_interp=Blend"]),
        ("key without frame", ["speed_percent=100@0;50"]),
        ("two keys one tick", ["speed_percent=100@0;50@0"]),
        ("held and eased key", ["speed_percent=100@0he"]),
        ("keyed popup with handle", ["frame_interp=1@0i0/50;2@5"]),
        ("keyed speed above max", ["speed_percent=100@0;10001@5"]),
    ]
    return [{"label": a, "params": base + b} for a, b in items]


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = sys.argv[1]
    g = build()
    ids = [c["id"] for c in g.cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    inputs = {}
    for c in g.cases:
        key = (c["source"], c["width"], c["height"], c["depth"], c["frames"])
        inputs[c["input"]] = key
    os.makedirs(os.path.join(out, "inputs"), exist_ok=True)
    infos = []
    for path, (src, w, h, d, fr) in sorted(inputs.items()):
        dest = os.path.join(out, path)
        data = clip_bytes(src, d, w, h, fr, fnv("%s%dx%dd%d" % (src, w, h, d)) & 0xFFFF)
        with open(dest, "wb") as fh:
            fh.write(data)
        infos.append({"file": path, "source": src, "width": w, "height": h, "depth": d,
                      "pix_fmt": PIX[d], "frames": fr})
    ts = {"format": 1, "ffmpeg": "none (inputs written by parity/retime-time/gen_grid.py)",
          "inputs": infos, "cases": g.cases}
    with open(os.path.join(out, "cases.json"), "w") as fh:
        json.dump(ts, fh, indent=1)
    with open(os.path.join(out, "grid.json"), "w") as fh:
        json.dump(g.meta, fh, indent=1, sort_keys=True)
    with open(os.path.join(out, "out_of_contract.json"), "w") as fh:
        json.dump(out_of_contract(), fh, indent=1)
    with open(os.path.join(out, "not_graded.json"), "w") as fh:
        json.dump({"reason": NOT_GRADED_REASON, "cases": g.not_graded}, fh, indent=1)
    by = {}
    for m in g.meta.values():
        by[m["group"]] = by.get(m["group"], 0) + 1
    print("wrote %s: %d cases, %d inputs; not graded (not_graded.json): %d" % (
        os.path.join(out, "cases.json"), len(g.cases), len(infos), len(g.not_graded)))
    print("groups: " + ", ".join("%s %d" % kv for kv in sorted(by.items())))
    cls = {}
    for m in g.meta.values():
        cls[m["class"]] = cls.get(m["class"], 0) + 1
    print("classes: " + ", ".join("%s %d" % kv for kv in sorted(cls.items())))


if __name__ == "__main__":
    main()
