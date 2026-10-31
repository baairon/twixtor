#!/usr/bin/env python3
"""Grid generator of the fgbg-core parity module (retime-fgbg without mattes or points).

Writes an fx-parity style test set (format 1): DIR/cases.json plus raw input clips in DIR/inputs/,
and DIR/grid.tsv (one row per case with its block and tags). Standard library only, deterministic.

  python3 gen_grid.py --out DIR [--vfx PATH]

With --vfx every case is checked with `vfx apply --check`; a case vfx refuses moves to
DIR/out_of_contract.json (never counted as a pass or a failure). The explicit out of contract
candidates (values outside the registered ranges) are always written there, with vfx's verdict.
"""
import argparse, hashlib, json, math, os, random, struct, subprocess, sys

EFFECT = "retime-fgbg"
PIX = {8: "rgba", 16: "rgba64le", 32: "gbrapf32le"}

# ---------------------------------------------------------------- pixel sources

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

def bg(x, y, f, w, h):
    return (0.5 + 0.35 * math.sin(0.8 * x + 0.5 * y) + 0.1 * (hsh(x, y, 1) - 0.5),
            0.5 + 0.35 * math.sin(0.4 * x - 0.7 * y + 1.0) + 0.1 * (hsh(x, y, 2) - 0.5),
            0.5 + 0.3 * math.cos(0.3 * x + 0.9 * y) + 0.1 * (hsh(x, y, 3) - 0.5))

def box_geom(w, h, f, speed):
    bw = max(2, min(w, h) // 2 + 1)
    bh = max(2, min(w, h) // 2 + 1)
    x0 = (w - bw) // 4 + speed[0] * f
    y0 = (h - bh) // 4 + speed[1] * f
    return x0, y0, bw, bh

def in_box(x, y, g):
    return g[0] <= x < g[0] + g[2] and g[1] <= y < g[1] + g[3]

def px_movebox(x, y, f, w, h, speed=(1, 1), alpha=False):
    g = box_geom(w, h, f, speed)
    if in_box(x, y, g):
        bx, by = x - g[0], y - g[1]
        r, gg, b = 0.9 - 0.05 * bx, 0.2 + 0.6 * hsh(bx, by, 7), 0.3 + 0.05 * by
        a = 0.6 if alpha else 1.0
        return (clamp01(r), clamp01(gg), clamp01(b), a)
    r, gg, b = bg(x, y, f, w, h)
    a = (0.2 + 0.8 * (x / (w - 1) if w > 1 else 0.5)) if alpha else 1.0
    return (clamp01(r), clamp01(gg), clamp01(b), a)

def px_pan(x, y, f, w, h):
    xs = x - 1.3 * f
    ys = y - 0.6 * f
    return (clamp01(0.5 + 0.45 * math.sin(0.7 * xs + 0.3 * ys)),
            clamp01(0.5 + 0.45 * math.sin(0.5 * xs - 0.6 * ys + 2.0)),
            clamp01(0.5 + 0.45 * math.cos(0.9 * xs + 0.2 * ys)), 1.0)

def px_noise(x, y, f, w, h):
    return (hsh(x, y, 11, f), hsh(x, y, 12, f), hsh(x, y, 13, f), 0.25 + 0.75 * hsh(x, y, 14, f))

def px_flat(x, y, f, w, h):
    return (0.25, 0.5, 0.75, 1.0)

# integer extremes: levels as a fraction of full scale
LEVELS8 = [0, 1, 2, 127, 128, 254, 255]
LEVELS16 = [0, 1, 255, 256, 32767, 32768, 65534, 65535]

def px_edge(x, y, f, w, h, depth):
    lv = LEVELS8 if depth == 8 else LEVELS16
    top = 255.0 if depth == 8 else 65535.0
    k = len(lv)
    def pick(c):
        return lv[(x + 2 * y + 3 * c + f) % k] / top
    return (pick(0), pick(1), pick(2), pick(3))

# float specials, as bit patterns: 0, -0, min denormal, max denormal, FLT_MIN, 1, above 1,
# FLT_MAX, +inf, -inf, NaN
SPEC = [0x00000000, 0x80000000, 0x00000001, 0x007FFFFF, 0x00800000, 0x3F800000, 0x40200000,
        0x7F7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000]
NEGNAN = 0xFFC00000

def fbits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]

def spec_all(x, y, f, w, h, nan=SPEC[10]):
    k = len(SPEC)
    vals = [SPEC[(x + 2 * y + 3 * c + f) % k] for c in range(4)]
    return [nan if v == SPEC[10] else v for v in vals]

def spec_rgb(x, y, f, w, h):
    v = spec_all(x, y, f, w, h)
    v[3] = fbits(1.0)
    return v

def spec_negnan(x, y, f, w, h):
    return spec_all(x, y, f, w, h, nan=NEGNAN)

def spec_box(x, y, f, w, h, alpha=False):
    g = box_geom(w, h, f, (1, 1))
    if in_box(x, y, g):
        k = len(SPEC)
        v = [SPEC[(x + y + 3 * c + f) % k] for c in range(3)]
        a = SPEC[(x + 2 * y + f) % k] if alpha else fbits(1.0)
        return v + [a]
    r, gg, b = bg(x, y, f, w, h)
    return [fbits(r), fbits(gg), fbits(b), fbits(1.0)]

SOURCES = {
    "movebox": lambda x, y, f, w, h, d: px_movebox(x, y, f, w, h),
    "fastbox": lambda x, y, f, w, h, d: px_movebox(x, y, f, w, h, speed=(3, 2)),
    "backbox": lambda x, y, f, w, h, d: px_movebox(x, y, f, w, h, speed=(-1, 1)),
    "alphabox": lambda x, y, f, w, h, d: px_movebox(x, y, f, w, h, alpha=True),
    "pan": lambda x, y, f, w, h, d: px_pan(x, y, f, w, h),
    "noise": lambda x, y, f, w, h, d: px_noise(x, y, f, w, h),
    "flat": lambda x, y, f, w, h, d: px_flat(x, y, f, w, h),
    "edge": px_edge,
    "specrgb": lambda x, y, f, w, h, d: spec_rgb(x, y, f, w, h),
    "specall": lambda x, y, f, w, h, d: spec_all(x, y, f, w, h),
    "specnan": lambda x, y, f, w, h, d: spec_negnan(x, y, f, w, h),
    "specbox": lambda x, y, f, w, h, d: spec_box(x, y, f, w, h),
    "specboxa": lambda x, y, f, w, h, d: spec_box(x, y, f, w, h, alpha=True),
}
FLOAT_ONLY = {"specrgb", "specall", "specnan", "specbox", "specboxa"}
INT_ONLY = {"edge"}

def render_clip(source, w, h, depth, frames):
    fn = SOURCES[source]
    out = bytearray()
    for f in range(frames):
        if depth == 32:
            planes = [[], [], [], []]  # G, B, R, A
            for y in range(h):
                for x in range(w):
                    p = fn(x, y, f, w, h, depth)
                    if source not in FLOAT_ONLY:
                        p = [fbits(v) for v in p]
                    r, g, b, a = p
                    planes[0].append(g); planes[1].append(b); planes[2].append(r); planes[3].append(a)
            for pl in planes:
                out += struct.pack("<%dI" % len(pl), *pl)
        else:
            top = 255 if depth == 8 else 65535
            fmt = "<B" if depth == 8 else "<H"
            for y in range(h):
                for x in range(w):
                    p = fn(x, y, f, w, h, depth)
                    for v in p:
                        out += struct.pack(fmt, int(min(top, max(0, round(v * top)))))
    return bytes(out)

# ---------------------------------------------------------------- the case grid

class Grid:
    def __init__(self):
        self.cases = []
        self.seen = set()
        self.rows = []

    def add(self, block, label, source, w, h, depth, params=(), frames=4, fps="24", tags=()):
        params = list(params)
        key = (source, w, h, depth, frames, fps, tuple(params))
        if key in self.seen:
            return
        self.seen.add(key)
        text = "|".join(map(str, key))
        hh = hashlib.sha1(text.encode()).hexdigest()[:8]
        cid = f"{EFFECT}.{label}.{source}-{w}x{h}.d{depth}.{hh}"
        self.cases.append(dict(
            id=cid, effect=EFFECT, label=label, params=params, depth=depth, width=w, height=h,
            source=source, input=f"inputs/{source}-{w}x{h}.d{depth}.f{frames}.{PIX[depth]}",
            reference=f"{cid}.{PIX[depth]}", kind="temporal", frames=frames, fps=fps))
        self.rows.append((cid, block, ",".join(tags)))

def build_grid():
    g = Grid()
    P = lambda **kw: [f"{k}={v}" for k, v in kw.items()]
    SIZES = [(1, 1), (2, 2), (2, 3), (3, 2), (1, 5), (5, 1), (3, 3), (4, 4), (5, 5), (7, 9), (9, 7),
             (16, 12), (17, 13), (23, 31), (37, 23), (64, 48)]
    SRC = ["movebox", "pan", "fastbox", "alphabox"]

    # B1 sizes x depths: forward warp at 50 percent (positions 0, 0.5, 1, 1.5), a default run too
    for i, (w, h) in enumerate(SIZES):
        for d in (8, 16, 32):
            src = SRC[(i + d // 8) % len(SRC)]
            g.add("size", "size-w3", src, w, h, d, P(speed_percent=50, warping=3),
                  tags=("tiny",) if min(w, h) < 4 else ())
    for w, h in [(1, 1), (2, 3), (5, 5), (17, 13), (64, 48)]:
        for d in (8, 16, 32):
            g.add("size", "size-default", "movebox", w, h, d, [], tags=("defaults",))
    for w, h in [(2, 2), (3, 3), (7, 9), (16, 12)]:
        for d in (8, 32):
            g.add("size", "size-w1", "pan", w, h, d, P(speed_percent=37.5, warping=1))

    # B2 warping x speed x depth
    for wp in (1, 2, 3, 4):
        for d in (8, 16, 32):
            for sp in (50, 37, 150):
                g.add("warping", f"warp{wp}", "movebox", 24, 16, d, P(speed_percent=sp, warping=wp))

    # B3 frame_interp x warping x depth
    for wp in (1, 2, 3, 4):
        for fi in (1, 2, 3):
            g.add("interp", f"interp{fi}-w{wp}", "alphabox", 17, 13, 8, P(speed_percent=50, warping=wp, frame_interp=fi))
    for fi in (1, 3):
        for d in (16, 32):
            g.add("interp", f"interp{fi}-w3", "movebox", 16, 12, d, P(speed_percent=37, warping=3, frame_interp=fi))
            g.add("interp", f"interp{fi}-w4", "pan", 16, 12, d, P(speed_percent=37, warping=4, frame_interp=fi))
            g.add("interp", f"interp{fi}-w2", "movebox", 16, 12, d, P(speed_percent=37, warping=2, frame_interp=fi))

    # B4 vectors x warping
    for v in (1, 2, 3, 4, 5):
        for wp in (1, 3):
            g.add("vectors", f"vec{v}-w{wp}", "movebox", 32, 24, 16, P(speed_percent=50, warping=wp, vectors=v))
    g.add("vectors", "vec4-w4", "fastbox", 32, 24, 8, P(speed_percent=50, warping=4, vectors=4))
    g.add("vectors", "vec3-w2", "fastbox", 32, 24, 8, P(speed_percent=50, warping=2, vectors=3))

    # B5 prep, sensitivity
    for pr in (1, 2, 3):
        for d in (8, 32):
            g.add("prep", f"prep{pr}", "pan", 20, 14, d, P(speed_percent=50, warping=3, prep=pr))
    g.add("prep", "prep2-w1", "pan", 20, 14, 16, P(speed_percent=50, warping=1, prep=2))
    g.add("prep", "prep3-w1", "pan", 20, 14, 16, P(speed_percent=50, warping=1, prep=3))
    for s in ("0", "-0", "1", "35", "70", "99.99", "100"):
        g.add("sens", f"sens{s}", "fastbox", 24, 16, 8, P(speed_percent=50, warping=3, main_bg_sensitivity=s),
              tags=("negzero",) if s == "-0" else ())
    for s in (0, 50, 100):
        g.add("sens", f"sens{s}-w1", "fastbox", 24, 16, 16, P(speed_percent=37, warping=1, main_bg_sensitivity=s))

    # B6 blur compensation
    for bc in ("0", "-0", "0.01", "1", "2.5", "5", "10"):
        for sp in (50, 200):
            g.add("blur", f"blur{bc}-s{sp}", "movebox", 20, 14, 8, P(speed_percent=sp, warping=3, blur_compensation=bc),
                  tags=("negzero",) if bc == "-0" else ())
    for wp in (1, 2, 4):
        for d in (16, 32):
            g.add("blur", f"blur3-w{wp}", "pan", 20, 14, d, P(speed_percent=200 if wp == 1 else 50, warping=wp, blur_compensation=3))
    g.add("blur", "blur1-s100", "movebox", 20, 14, 8, P(speed_percent=100, warping=3, blur_compensation=1))
    g.add("blur", "blur2-vec1", "movebox", 20, 14, 8, P(speed_percent=200, warping=3, blur_compensation=2, vectors=1))
    g.add("blur", "blur10-neg", "movebox", 20, 14, 32, P(speed_percent=-150, warping=3, blur_compensation=10))

    # B7 track_rgb_a (the wrap flag), video_360
    for wp in (1, 2, 3, 4):
        for d in (8, 16, 32):
            g.add("wrap", f"trk-w{wp}", "pan", 16, 12, d, P(speed_percent=50, warping=wp, track_rgb_a=1))
    g.add("wrap", "trk-blur", "pan", 16, 12, 8, P(speed_percent=200, warping=1, track_rgb_a=1, blur_compensation=2))
    for wp in (1, 2, 3, 4):
        g.add("wrap", f"v360-w{wp}", "alphabox", 16, 12, 8, P(speed_percent=50, warping=wp, video_360=1))
    g.add("wrap", "v360-w3-32", "alphabox", 16, 12, 32, P(speed_percent=50, warping=3, video_360=1))

    # B8 display
    for dsp in (1, 2, 3, 4, 5, 6):
        g.add("display", f"disp{dsp}", "movebox", 16, 12, 8, P(speed_percent=50, warping=3, display=dsp),
              tags=("nomatte",) if dsp >= 3 else ())
    for d in (16, 32):
        g.add("display", "disp2", "movebox", 16, 12, d, P(speed_percent=50, display=2))
    g.add("display", "disp2-speed", "movebox", 16, 12, 8, P(speed_percent=-75, time_remap_mode=1, display=2))
    g.add("display", "disp6-32", "movebox", 16, 12, 32, P(speed_percent=50, warping=3, display=6), tags=("nomatte",))
    g.add("display", "disp3-w1", "movebox", 16, 12, 8, P(speed_percent=50, warping=1, display=3), tags=("nomatte",))

    # B9 time controls (a smaller sample)
    for sp in (-10000, -100, -50, 0, "-0", 25, 100, 250, 400, 10000):
        g.add("time", f"speed{sp}", "movebox", 16, 12, 8, P(speed_percent=sp, warping=3), frames=6,
              tags=("negzero",) if sp == "-0" else ())
    for fn in (0, "-0", 1, 1.5, 2.25, 3, -1, 200000, -200000):
        g.add("time", f"fnum{fn}", "movebox", 16, 12, 16, P(time_remap_mode=2, frame_num=fn, warping=3), frames=5,
              tags=("negzero",) if fn == "-0" else ())
    g.add("time", "fnum-w1", "movebox", 16, 12, 8, P(time_remap_mode=2, frame_num=2.5, warping=1))
    for rate in ("12", "24", "29.97", "0.10000000149011612", "1000", "59.94", "23.976"):
        g.add("time", f"inrate{rate}", "movebox", 16, 12, 8,
              P(in_fps_is_out_fps=0, input_frame_rate=rate, speed_percent=50, warping=3), frames=6, fps="24")
    for fl in (2, 3):
        g.add("time", f"fields{fl}", "movebox", 16, 12, 8, P(input_fields=fl, speed_percent=50, warping=3))
    g.add("time", "plenty0", "movebox", 16, 12, 8, P(plenty_memory=0, speed_percent=50, warping=3))
    g.add("time", "ntsc-w3", "movebox", 16, 12, 16, P(speed_percent=50, warping=3), fps="30000/1001")
    g.add("time", "ntsc-w1", "movebox", 16, 12, 16, P(speed_percent=50, warping=1), fps="30000/1001")
    for sm in (1, 2, 3, 4, 5):
        g.add("time", f"mark{sm}", "movebox", 16, 12, 8, P(src_mark_segments=sm, speed_percent=50, warping=3), frames=5)
    # keyframed controls use the harness's labelled linear stand-in on both sides
    g.add("keyed", "ramp-speed", "movebox", 16, 12, 8, P(speed_percent="100@0;50@4", warping=3), frames=5, tags=("standin",))
    g.add("keyed", "ramp-fnum", "movebox", 16, 12, 16, P(time_remap_mode=2, frame_num="0@0;3@4", warping=3), frames=5, tags=("standin",))
    g.add("keyed", "ramp-blur", "movebox", 16, 12, 32, P(speed_percent=50, blur_compensation="0@0;4@4", warping=3), frames=5, tags=("standin",))
    g.add("keyed", "ramp-sens", "fastbox", 16, 12, 8, P(speed_percent=50, main_bg_sensitivity="0@0;100@4", warping=3), frames=5, tags=("standin",))
    g.add("keyed", "mark-cut", "movebox", 16, 12, 8, P(src_mark_segments="1@0;2@2", speed_percent=50, warping=3), frames=5, tags=("standin",))

    # B9b rates whose frame duration in ticks is not a whole number: the frame to time rounding
    for rate in ("29.97", "23.976", "17.3", "7.7", "41.123", "24.024", "24.5"):
        for sp in (100, 250):
            g.add("time2", f"rate{rate}-s{sp}", "movebox", 8, 6, 8,
                  P(in_fps_is_out_fps=0, input_frame_rate=rate, speed_percent=sp, warping=3), frames=12)
    g.add("time2", "rate29.97-s100-16", "pan", 8, 6, 16,
          P(in_fps_is_out_fps=0, input_frame_rate="29.97", speed_percent=100, warping=1), frames=12)
    g.add("time2", "rate7.7-fnum", "movebox", 8, 6, 8,
          P(in_fps_is_out_fps=0, input_frame_rate="7.7", time_remap_mode=2, frame_num="3.3", warping=3), frames=12)
    g.add("time2", "rate17.3-mark", "movebox", 8, 6, 8,
          P(in_fps_is_out_fps=0, input_frame_rate="17.3", src_mark_segments=5, speed_percent=100, warping=3), frames=12)

    # B10 float special values (32 bpc) and integer extremes
    for src in ("specrgb", "specall", "specnan", "specbox", "specboxa"):
        for wp in (1, 2, 3, 4):
            g.add("spec", f"{src}-w{wp}", src, 8, 6, 32, P(speed_percent=50, warping=wp), tags=("special",))
        g.add("spec", f"{src}-w3-big", src, 16, 12, 32, P(speed_percent=37, warping=3), tags=("special",))
        g.add("spec", f"{src}-disp2", src, 8, 6, 32, P(speed_percent=50, display=2), tags=("special",))
    g.add("spec", "specbox-blur", "specbox", 16, 12, 32, P(speed_percent=200, warping=3, blur_compensation=2), tags=("special",))
    g.add("spec", "specbox-v1", "specbox", 16, 12, 32, P(speed_percent=50, warping=3, vectors=1), tags=("special",))
    g.add("spec", "specbox-w1-i3", "specbox", 16, 12, 32, P(speed_percent=50, warping=1, frame_interp=3), tags=("special",))
    g.add("spec", "specbox-w3-i1", "specbox", 16, 12, 32, P(speed_percent=50, warping=3, frame_interp=1), tags=("special",))
    g.add("spec", "specbox-trk", "specbox", 16, 12, 32, P(speed_percent=50, warping=3, track_rgb_a=1), tags=("special",))
    g.add("spec", "specbox-prep3", "specbox", 16, 12, 32, P(speed_percent=50, warping=3, prep=3), tags=("special",))
    for d in (8, 16):
        for wp in (1, 2, 3, 4):
            g.add("spec", f"edge-w{wp}", "edge", 9, 7, d, P(speed_percent=50, warping=wp), tags=("special",))
        g.add("spec", "edge-w3-big", "edge", 16, 12, d, P(speed_percent=37, warping=3), tags=("special",))

    # B11 real motion on larger frames, every warping and depth
    for src in ("movebox", "fastbox", "pan", "backbox"):
        for wp in (1, 2, 3, 4):
            d = (8, 16, 32)[(len(g.cases) + wp) % 3]
            g.add("motion", f"mot-{src}-w{wp}", src, 48, 32, d, P(speed_percent=40, warping=wp))
    for d in (8, 16, 32):
        g.add("motion", "mot-64x48", "fastbox", 64, 48, d, P(speed_percent=33, warping=3), frames=3)
        g.add("motion", "mot-64x48-w4", "movebox", 64, 48, d, P(speed_percent=66, warping=4), frames=3)
    g.add("motion", "mot-back", "backbox", 32, 24, 8, P(speed_percent=-60, warping=3))
    g.add("motion", "mot-hi-sens", "fastbox", 40, 30, 16, P(speed_percent=50, warping=3, vectors=5, main_bg_sensitivity=100))
    g.add("motion", "mot-lo-vec", "fastbox", 40, 30, 16, P(speed_percent=50, warping=3, vectors=2))

    # B12 alpha, noise, flat
    for wp in (1, 2, 3, 4):
        for d in (8, 16, 32):
            g.add("alpha", f"alpha-w{wp}", "alphabox", 16, 12, d, P(speed_percent=50, warping=wp))
    for wp in (1, 3):
        for d in (8, 16, 32):
            g.add("noise", f"noise-w{wp}", "noise", 12, 9, d, P(speed_percent=50, warping=wp))
    for wp in (1, 3):
        g.add("noise", f"flat-w{wp}", "flat", 12, 9, 8, P(speed_percent=50, warping=wp))
    g.add("noise", "flat-w3-32", "flat", 12, 9, 32, P(speed_percent=50, warping=3))

    # B13 seeded combinations of valid values
    rng = random.Random(20261030)
    for i in range(36):
        d = rng.choice((8, 16, 32))
        w, h = rng.choice([(16, 12), (17, 13), (23, 15), (31, 21), (12, 20), (8, 8)])
        src = rng.choice(["movebox", "fastbox", "pan", "alphabox", "backbox"])
        p = dict(
            warping=rng.choice((1, 2, 3, 4)),
            speed_percent=rng.choice((25, 33, 50, 66.5, 75, 120, 150, 200, -50, -125)),
            frame_interp=rng.choice((1, 2, 3)),
            vectors=rng.choice((1, 2, 3, 4, 5, 5, 5)),
            prep=rng.choice((1, 1, 2, 3)),
            main_bg_sensitivity=rng.choice((0, 20, 70, 70, 100)),
            blur_compensation=rng.choice((0, 0, 0.5, 1.5, 4)),
            track_rgb_a=rng.choice((0, 0, 1)),
            video_360=rng.choice((0, 0, 1)),
            display=rng.choice((1, 1, 1, 2, 4)),
        )
        g.add("mix", f"mix{i:02d}", src, w, h, d, [f"{k}={v}" for k, v in p.items()])
    return g

# out of contract candidates: values outside the registered ranges or choices. vfx refuses these,
# the plugin refuses or clamps them; they are listed and never counted.
OUT_OF_CONTRACT = [
    ("speed_percent", "10000.5"), ("speed_percent", "-10001"), ("frame_num", "200001"), ("frame_num", "-200001"),
    ("input_frame_rate", "0.1"), ("input_frame_rate", "0.09"), ("input_frame_rate", "1000.5"),
    ("blur_compensation", "10.01"), ("blur_compensation", "-0.01"),
    ("main_bg_sensitivity", "100.01"), ("main_bg_sensitivity", "-0.01"),
    ("warping", "0"), ("warping", "5"), ("vectors", "0"), ("vectors", "6"), ("prep", "4"),
    ("frame_interp", "4"), ("display", "7"), ("display", "0"), ("time_remap_mode", "3"),
    ("src_mark_segments", "6"), ("input_fields", "4"), ("track_rgb_a", "2"),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vfx")
    a = ap.parse_args()
    out = a.out
    os.makedirs(os.path.join(out, "inputs"), exist_ok=True)
    g = build_grid()

    # explicit out of contract candidates (a small valid-size clip each)
    oc = []
    for i, (k, v) in enumerate(OUT_OF_CONTRACT):
        oc.append(dict(id=f"{EFFECT}.oc-{k}-{v}", effect=EFFECT, label=f"oc-{k}", params=[f"{k}={v}", "speed_percent=50"] if k != "speed_percent" else [f"{k}={v}"],
                       depth=8, width=16, height=12, source="movebox", input="inputs/movebox-16x12.d8.f4.rgba",
                       reference=f"{EFFECT}.oc-{k}-{v}.rgba", kind="temporal", frames=4, fps="24",
                       reason="outside the registered range or popup choices"))
    # other candidates: sizes the plugin may refuse (zero area cannot be written as a clip)
    # and a frame count of 1 (no neighbour for a position past the last frame)
    g_extra = []

    # inputs
    need = {}
    for c in g.cases + oc:
        need[(c["source"], c["width"], c["height"], c["depth"], c["frames"])] = c["input"]
    inputs = []
    for (src, w, h, d, n), rel in sorted(need.items()):
        path = os.path.join(out, rel)
        if not os.path.exists(path):
            data = render_clip(src, w, h, d, n)
            with open(path, "wb") as fh:
                fh.write(data)
        inputs.append(dict(file=rel, source=src, width=w, height=h, depth=d, pix_fmt=PIX[d], frames=n))

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
        c["vfx_verdict"] = vfx_check(a.vfx, c) if a.vfx else "not checked"
        oc_checked.append(c)

    inputs = [i for i in inputs]
    ts = dict(format=1, ffmpeg="none (raw inputs written by gen_grid.py)", inputs=inputs, cases=g.cases)
    with open(os.path.join(out, "cases.json"), "w") as fh:
        json.dump(ts, fh, indent=1)
    with open(os.path.join(out, "out_of_contract.json"), "w") as fh:
        json.dump(dict(format=1, note="candidates, never counted as passes or failures", cases=oc_checked + refused), fh, indent=1)
    with open(os.path.join(out, "grid.tsv"), "w") as fh:
        fh.write("id\tblock\ttags\n")
        keep_ids = {c["id"] for c in g.cases}
        for cid, block, tags in g.rows:
            if cid in keep_ids:
                fh.write(f"{cid}\t{block}\t{tags}\n")
    print(f"wrote {out}/cases.json: {len(g.cases)} cases, {len(inputs)} inputs; out of contract listed: {len(oc_checked) + len(refused)}")

def vfx_check(vfx, c):
    args = [vfx, "apply", c["effect"], "--width", str(c["width"]), "--height", str(c["height"]), "--depth", str(c["depth"]),
            "--fps", c["fps"], "--frames", str(c["frames"]), "--check"]
    for p in c["params"]:
        args += ["--param", p]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode == 0:
        return None
    return (r.stderr.strip() or f"exit {r.returncode}").splitlines()[0]

if __name__ == "__main__":
    main()
