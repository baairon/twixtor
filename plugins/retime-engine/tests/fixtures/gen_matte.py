"""Independent numpy float32 models of the layer primitives of the FG matte paths: the matte
channel, the choke, the tracking mattes, the OVER weights, the layer
premultiply and OVER, the tracking point splat, the second tracker (rule G5) and the Draw
Geom discs (rule G3). They produce `matte_cases.hex` for the Rust test
`matte::tests::model_fixtures_are_reproduced_bit_for_bit`.

Run:  python -I -B gen_matte.py [OUT]        (from any directory; needs numpy)

The Gaussian blur, the pull-push hole fill, OVER and its stores are the models of `f1_gen.py`,
imported from this directory. No Rust code of the engine is used. Every arithmetic step is on
np.float32 in the order of the Rust code (numpy does not fuse).

File format, little endian:
  b"MAT1", u32 ncases, then per case u32 kind, u32 depth (0 u8, 1 u16, 2 f32), u32 W, u32 H and
    kind 1 extract:      i32 chan; image (W*H*4 samples of the depth, A R G B); matte (W*H f32)
    kind 2 choke:        f32 amount; matte in (W*H f32); matte out (W*H f32)
    kind 3 effective:    u32 nfg; nfg f32 shrinks; nfg mattes; nfg + 1 mattes out
    kind 4 over weights: u32 nfg, u32 first, u32 last; nfg mattes; last - first + 1 mattes out
    kind 5 premultiply:  image; matte; image out
    kind 6 over:         dest; src; dest out
    kind 7 splat:        u32 quality, u32 hiq, u32 nlev, nlev (u32 w, u32 h), u32 npairs,
                         npairs (f32 from x, from y, to x, to y); per level d (w*h*2 f32), w (w*h f32)
    kind 8 tracker:      u32 quality, u32 hiq, u32 w0, u32 h0, u32 npairs, pairs; flow (w0*h0*2 f32)
    kind 9 draw:         f32 ratio, u32 npts, npts (f32 x, f32 y, u8 r, u8 g, u8 b, u8 0);
                         image in; image out
  W x H is the output size of kinds 7 and 8 (their depth is 0).
"""
import io
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f1_gen as F  # noqa: E402

f32 = np.float32
WHITE = F.WHITE
DTYPE = F.DTYPE
CHOKE_MAX = f32(65536.0)
ONE = f32(1.0)
ZERO = f32(0.0)


def clamp01(x):
    x = np.asarray(x, dtype=f32)
    return np.where(x > ONE, ONE, np.where(x > ZERO, x, ZERO)).astype(f32)


# --------------------------------------------------------------------------------------------
# mattes
# --------------------------------------------------------------------------------------------

def extract(img, chan, depth):
    c = {1: 1, 2: 2, 3: 3}.get(chan, 0)
    v = img.reshape(-1, 4)[:, c].astype(f32)
    with np.errstate(all="ignore"):
        return clamp01((v / WHITE[depth]).astype(f32))


def shrink(m, n):
    """m: (h, w) float32; n: a whole number as float32."""
    if not (n >= ONE) or m.size == 0:
        return m
    r = f32(np.ceil(f32(n + ONE)))
    fr = f32(r - f32(n + ONE))
    if fr < f32(0.01):
        fr = f32(fr + f32(0.01))
    radius = f32(f32(ONE - fr) + r)
    k = F.gaussian_kernel(radius)
    ni = int(n)
    t = f32(ONE - f32(k[ni] + k[ni + 1]))
    den = f32(ONE - t)
    b = F.gaussian(m, radius)
    return clamp01(((b - t) / den).astype(f32))


def choke(m, amount):
    a = f32(amount)
    if a == ZERO or not np.isfinite(a):
        return m.copy()
    grow = a < ZERO
    if grow:
        a = f32(-a)
    if a > CHOKE_MAX:
        a = CHOKE_MAX
    if grow:
        m = (ONE - m).astype(f32)
    lo = f32(np.floor(a))
    hi = f32(np.ceil(a))
    if lo == hi:
        m = shrink(m, lo)
    else:
        c = m.copy()
        m = shrink(m, lo)
        c = shrink(c, hi)
        t = f32(a - lo)
        it = f32(ONE - t)
        m = ((m * it).astype(f32) + (c * t).astype(f32)).astype(f32)
    if grow:
        m = (ONE - m).astype(f32)
    return clamp01(m)


def effective(ms, shrinks):
    vis = np.ones_like(ms[0])
    out = []
    for m, s in zip(ms, shrinks):
        out.append((m * vis).astype(f32))
        inv = choke((ONE - m).astype(f32), s)
        vis = clamp01((inv * vis).astype(f32))
    out.append(vis)
    return out


def over_weights(ms, first, last, shape):
    nfg = len(ms)
    vis = np.ones(shape, dtype=f32)
    out = []
    for k in range(first, last + 1):
        if k == nfg:
            out.append(vis)
            break
        out.append((ms[k] * vis).astype(f32))
        if k < last:
            vis = clamp01(((ONE - ms[k]).astype(f32) * vis).astype(f32))
    return out


def premultiply(img, m, depth):
    px = img.reshape(-1, 4)
    if depth == 2:
        return (px * m[:, None]).astype(f32).reshape(-1)
    x = (px.astype(f32) * m[:, None]).astype(f32)
    q = np.trunc(x.astype(np.float64) + 0.5).astype(np.int64)
    mask = 0xFF if depth == 0 else 0xFFFF
    return (q & mask).astype(DTYPE[depth]).reshape(-1)


def over(dest, src, depth):
    a = src.reshape(-1, 4).astype(f32)
    b = dest.reshape(-1, 4).astype(f32)
    with np.errstate(all="ignore"):
        v = F.over(a, b, WHITE[depth])
    return F.store(v, np.zeros(len(v), dtype=bool), depth).reshape(-1)


# --------------------------------------------------------------------------------------------
# tracking points and Draw Geom
# --------------------------------------------------------------------------------------------

def flow_resolution(q, hiq, w, h):
    r = 3 if q <= 2 else (2 if q == 3 else (1 if q == 4 else 0))
    if not hiq:
        r = {0: 1, 1: 2, 2: 3}.get(r, r)
    while r > 0 and ((w >> r) < 2 or (h >> r) < 2):
        r -= 1
    w0, h0 = w, h
    for _ in range(r):
        w0 //= 2
        h0 //= 2
    return w0, h0


def level_sizes(w0, h0):
    levels, w, h = 1, w0, h0
    while True:
        w //= 2
        if w < 2:
            break
        h //= 2
        if h < 2:
            break
        if w > 2 and h > 2 and not (w == 2 and h == 2):
            levels += 1
        else:
            break
    sizes, w, h = [], w0, h0
    for _ in range(levels):
        sizes.append((w, h))
        w, h = max(2, w // 2), max(2, h // 2)
    return sizes


def splat(pairs, wl, hl, W, H):
    d = np.zeros(wl * hl * 2, dtype=f32)
    wt = np.zeros(wl * hl, dtype=f32)
    if W < 2 or H < 2:
        return d, wt
    kx = f32(f32(wl - 1) / f32(W - 1))
    ky = f32(f32(hl - 1) / f32(H - 1))
    with np.errstate(all="ignore"):
        for p in pairs:
            fx, fy, tx, ty = (f32(v) for v in p)
            ax, ay = f32(fx + f32(0.5)), f32(fy + f32(0.5))
            if not (ax >= ZERO and ax < f32(W) and ay >= ZERO and ay < f32(H)):
                continue
            if not (np.isfinite(tx) and np.isfinite(ty)):
                continue
            ix = int(f32(f32(fx * kx) + f32(0.5)))
            iy = int(f32(f32(fy * ky) + f32(0.5)))
            if ix < 0 or iy < 0 or ix >= wl or iy >= hl:
                continue
            i = iy * wl + ix
            d[2 * i] = f32(d[2 * i] + f32(f32(fx - tx) * kx))
            d[2 * i + 1] = f32(d[2 * i + 1] + f32(f32(fy - ty) * ky))
            wt[i] = f32(wt[i] + ONE)
    return d, wt


def tracker(pairs, w0, h0, W, H):
    d, wt = splat(pairs, w0, h0, W, H)
    flow = np.zeros(w0 * h0 * 2, dtype=f32)
    known = wt != 0
    if not known.any():
        return flow
    hole = (~known).reshape(h0, w0)
    for c in range(2):
        plane = np.zeros(w0 * h0, dtype=f32)
        plane[known] = (d[c::2][known] / wt[known]).astype(f32)
        sol = F.fill_plane(plane.reshape(h0, w0), hole) if hole.any() else plane.reshape(h0, w0)
        flow[c::2] = (f32(-1.0) * sol.reshape(-1)).astype(f32)
    return flow


def draw(img, pts, ratio, depth, w, h):
    out = img.reshape(h, w, 4).copy()
    if not pts:
        return out.reshape(-1)
    r = f32(f32(1.5) * f32(ratio))
    if not (r >= ZERO) or not np.isfinite(r):
        return out.reshape(-1)
    r2 = f32(r * r)
    ys, xs = np.mgrid[0:h, 0:w]
    xs, ys = xs.astype(f32), ys.astype(f32)
    for (px, py), col in pts:
        px, py = f32(px), f32(py)
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        dx = (xs - px).astype(f32)
        dy = (ys - py).astype(f32)
        cov = ((dx * dx).astype(f32) + (dy * dy).astype(f32)).astype(f32) <= r2
        if depth == 0:
            val = np.array([255, col[0], col[1], col[2]], dtype=np.uint8)
        elif depth == 1:
            val = np.array([65535] + [c * 257 for c in col], dtype=np.uint16)
        else:
            val = np.array([1.0] + [f32(c) / f32(255.0) for c in col], dtype=f32)
        out[cov] = val
    return out.reshape(-1)


# --------------------------------------------------------------------------------------------
# test data
# --------------------------------------------------------------------------------------------

def soft_matte(rng, w, h, kind):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    if kind == "disc":
        cx, cy, rad = w * 0.45, h * 0.55, min(w, h) * 0.3
        return ((xx - cx) ** 2 + (yy - cy) ** 2 <= rad * rad).astype(f32)
    if kind == "soft":
        cx, cy, rad = w * 0.5, h * 0.4, min(w, h) * 0.35
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        return np.clip((rad - dist) / 3.0 + 0.5, 0.0, 1.0).astype(f32)
    if kind == "edge":
        return (xx < w // 2).astype(f32)
    m = rng.random((h, w))
    pick = rng.random((h, w))
    return np.where(pick < 0.25, 0.0, np.where(pick < 0.5, 1.0, m)).astype(f32)


def matte_image(rng, w, h, depth):
    img = F.random_image(rng, w, h, depth).reshape(h, w, 4)
    if depth == 2:
        img = img.copy()
        img[0, 0, 1] = np.float32(np.nan)
        img[0, 1, 2] = f32(-0.75)
        img[1 % h, 2 % w, 3] = f32(1.5)
        img[0, 3 % w, 0] = f32(np.inf)
    return img.reshape(-1)


def main(path):
    rng = np.random.default_rng(20261027)
    cases = []

    def add(kind, depth, w, h, payload):
        cases.append(struct.pack("<IIII", kind, depth, w, h) + payload)

    def arr(a):
        return np.ascontiguousarray(a).tobytes()

    # 1 extract: every depth and channel popup value
    for depth in (0, 1, 2):
        for chan in (1, 2, 3, 4, 7):
            w, h = 13, 7
            img = matte_image(rng, w, h, depth)
            add(1, depth, w, h, struct.pack("<i", chan) + arr(img) + arr(extract(img, chan, depth)))
    # 2 choke: hard, soft, edge and random mattes, the amounts of the gap-shrink family and more
    sizes = [(17, 13), (31, 23), (5, 40), (40, 3), (2, 2), (1, 9), (12, 12)]
    amounts = [0.0, 0.5, 1.0, 2.0, 3.5, -1.0, -2.25, 10.0, 0.01, 4.999, 1e9]
    kinds = ["disc", "soft", "edge", "random"]
    for si, (w, h) in enumerate(sizes):
        for ai, a in enumerate(amounts):
            if (si + ai) % 2 and a not in (1.0, 3.5):
                continue
            if a == 1e9 and (w, h) != (2, 2):
                # the CHOKE_MAX kernel has 65538 taps; one tiny matte is enough
                continue
            m = soft_matte(rng, w, h, kinds[(si + ai) % 4])
            add(2, 2, w, h, struct.pack("<f", a) + arr(m) + arr(choke(m, a)))
    # 3 effective: one to three FG mattes, with and without shrinks
    for nfg, shrinks in ((1, [0.0]), (1, [2.0]), (2, [0.0, 0.0]), (2, [1.5, 0.0]), (3, [0.0, 0.5, 3.0]),
                         (3, [2.0, -1.0, 1.0])):
        w, h = 19, 14
        ms = [soft_matte(rng, w, h, kinds[(nfg + k) % 4]) for k in range(nfg)]
        out = effective(ms, [f32(s) for s in shrinks])
        add(3, 2, w, h, struct.pack("<I", nfg) + struct.pack(f"<{nfg}f", *shrinks)
            + b"".join(arr(m) for m in ms) + b"".join(arr(o) for o in out))
    # 4 over weights: every group of up to three FG layers
    for nfg in range(4):
        groups = [(0, nfg)] + [(k, k) for k in range(nfg + 1)] + ([(1, nfg)] if nfg >= 2 else [])
        for first, last in groups:
            w, h = 11, 9
            ms = [soft_matte(rng, w, h, kinds[(k + first) % 4]) for k in range(nfg)]
            out = over_weights(ms, first, last, (h, w))
            add(4, 2, w, h, struct.pack("<III", nfg, first, last)
                + b"".join(arr(m) for m in ms) + b"".join(arr(o) for o in out))
    # 5 premultiply and 6 over: every depth
    for depth in (0, 1, 2):
        for w, h in ((23, 17), (4, 3)):
            img = F.random_pixels(rng, w * h, depth)
            m = soft_matte(rng, w, h, "random").reshape(-1)
            add(5, depth, w, h, arr(img) + arr(m) + arr(premultiply(img, m, depth)))
            dest = F.random_pixels(rng, w * h, depth)
            src = F.random_pixels(rng, w * h, depth)
            add(6, depth, w, h, arr(dest) + arr(src) + arr(over(dest, src, depth)))
    # 7 splat and 8 second tracker
    geos = [(64, 48, 5, 1), (64, 48, 4, 1), (50, 34, 3, 0), (33, 17, 5, 1), (9, 7, 4, 1)]
    for gi, (W, H, q, hiq) in enumerate(geos):
        pts = [
            (40.0, 30.0, 46.0, 32.0),
            (40.4, 30.6, 46.4, 32.6),
            (W * 0.25, H * 0.75, W * 0.25 - 3.5, H * 0.75 + 1.25),
            (W * 0.25 + 0.2, H * 0.75 - 0.2, W * 0.25 + 2.0, H * 0.75),
            (-0.6, 3.0, 0.0, 3.0),
            (W - 0.6, H - 0.6, W - 3.0, H - 2.0),
            (W - 0.4, 1.0, W - 1.0, 1.0),
            (5.0, 2.0, np.inf, 2.0),
        ]
        pts += [tuple(rng.random(4) * [W, H, W, H]) for _ in range(3)]
        sizes = level_sizes(*flow_resolution(q, hiq, W, H))
        for npairs in ((0, 1, len(pts)) if gi % 2 == 0 else (2, len(pts))):
            sel = pts[:npairs]
            payload = struct.pack("<III", q, hiq, len(sizes))
            payload += b"".join(struct.pack("<II", lw, lh) for lw, lh in sizes)
            payload += struct.pack("<I", len(sel)) + b"".join(struct.pack("<4f", *p) for p in sel)
            for lw, lh in sizes:
                d, wt = splat(sel, lw, lh, W, H)
                payload += arr(d) + arr(wt)
            add(7, 0, W, H, payload)
            w0, h0 = sizes[0]
            payload = struct.pack("<IIIII", q, hiq, w0, h0, len(sel))
            payload += b"".join(struct.pack("<4f", *p) for p in sel)
            add(8, 0, W, H, payload + arr(tracker(sel, w0, h0, W, H)))
    # 9 draw: every depth, the ratios 1, 2 and 0.5, overlapping and clipped points
    for depth in (0, 1, 2):
        for ratio in (1.0, 2.0, 0.5):
            w, h = 15, 11
            pts = [((3.0, 4.0), (0, 255, 0)), ((4.3, 4.6), (0, 0, 255)), ((0.2, 10.4), (255, 255, 255)),
                   ((14.5, 0.5), (255, 128, 64)), ((7.75, 6.25), (128, 64, 255)), ((-1.2, 5.0), (255, 0, 255))]
            img = F.random_image(rng, w, h, depth)
            payload = struct.pack("<fI", ratio, len(pts))
            payload += b"".join(struct.pack("<ff4B", x, y, c[0], c[1], c[2], 0) for (x, y), c in pts)
            add(9, depth, w, h, payload + arr(img) + arr(draw(img, pts, ratio, depth, w, h)))

    with io.BytesIO() as f:
        f.write(b"MAT1")
        f.write(struct.pack("<I", len(cases)))
        for c in cases:
            f.write(c)
        blob = f.getvalue()
    with open(path, "w", encoding="ascii", newline="\n") as out:
        out.write(F.hex_text(blob))
    print(f"wrote {len(cases)} cases to {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "matte_cases.hex")
