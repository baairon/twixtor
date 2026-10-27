"""Independent numpy float32 model of the textured forward warp.
It produces `f2_cases.hex` for the Rust test `forward::tests::model_fixture`.

Run:  python -I f2_gen.py        (from this directory; needs numpy; deterministic)

Some steps are in double: the epsilon added to the vertex positions, the neighbour sums of the mip
level and the reciprocal of the running coverage count. The model does not use the structure of the
Rust code: the triangle scan converter is a per edge description (an edge is active from the first
scanline at or below its upper vertex up to its end, its state is initialised for the previous
scanline and advanced by repeated addition), and the pyramid, the blur and the resample are written
per pixel in plain loops. Every arithmetic step is on np.float32 (numpy does not fuse); double steps
are explicit.

File format, little endian:
  b"F2C1", u32 ncases, then per case
    u32 fmt (0 u8, 1 u16, 2 f32), u32 W, u32 H, u32 FW, u32 FH, f32 amount, u32 want_cov (always 1)
    src (W*H*4 samples A R G B of the format), flow (FW*FH*2 f32), expected out (W*H*4), expected coverage (W*H bytes)
"""
import io
import math
import struct
import sys

import numpy as np

f32 = np.float32
f64 = np.float64
DT = {0: np.uint8, 1: np.uint16, 2: np.float32}


def cvtt(x):
    return int(np.trunc(x))


# ---------------------------------------------------------------- flow sampler -----------------------
def cell(x, n):
    i = cvtt(x)
    fr = x - f32(i)
    if i >= n - 1:
        i = n - 2
        fr = f32(1.0)
    i = max(i, 0)
    i = min(i, n - 2)
    return i, fr


def flow_bilerp(flow, x, y):
    fh, fw, _ = flow.shape
    ix, fx = cell(x, fw)
    iy, fy = cell(y, fh)
    out = []
    for c in range(2):
        p00 = flow[iy, ix, c]
        p10 = flow[iy, ix + 1, c]
        p01 = flow[iy + 1, ix, c]
        p11 = flow[iy + 1, ix + 1, c]
        top = (p10 - p00) * fx + p00
        bot = (p11 - p01) * fx + p01
        out.append((bot - top) * fy + top)
    return out


# ---------------------------------------------------------------- vertices ----------------------------
def grid_consts(flow, w, h, amount):
    fh, fw, _ = flow.shape
    rx = f32(fw - 1) / f32(w - 1)
    ry = f32(fh - 1) / f32(h - 1)
    return rx, ry, amount / rx, amount / ry


def positions(flow, w, h, amount):
    rx, ry, sx, sy = grid_consts(flow, w, h, amount)
    pos = np.zeros((h, w, 2), f32)
    for r in range(h):
        for c in range(w):
            fx, fy = flow_bilerp(flow, f32(c) * rx, f32(r) * ry)
            pos[r, c, 0] = fx * sx + f32(c)
            pos[r, c, 1] = fy * sy + f32(r)
        if r != 0:
            for c in range(1, w):
                for k in range(2):
                    pos[r, c, k] = f32(f64(pos[r, c, k]) + 0.0001)
    return pos


def mip(flow, w, h, amount, c, r):
    rx, ry, sx, sy = grid_consts(flow, w, h, amount)
    up = 1 if r - 1 < 0 else r - 1
    down = h - 2 if h - 1 < r + 1 else r + 1
    left = 1 if c - 1 < 0 else c - 1
    right = w - 2 if w - 1 < c + 1 else c + 1
    fx, fy = flow_bilerp(flow, f32(c) * rx, f32(r) * ry)
    xc = fx * sx + f32(c)
    yc = fy * sy + f32(r)

    def nb(ci, ri):
        gx, gy = flow_bilerp(flow, f32(ci) * rx, f32(ri) * ry)
        dx = (gx * sx + f32(ci)) - xc
        dy = (gy * sy + f32(ri)) - yc
        return np.sqrt(dy * dy + dx * dx)

    ru, lu, ld, rd = nb(right, up), nb(left, up), nb(left, down), nb(right, down)
    u, dn, l, rr = nb(c, up), nb(c, down), nb(left, r), nb(right, r)
    diag = ((ru + lu) + ld) + rd
    t = f64(diag) * 0.7071067811865476
    for s in (u, dn, l, rr):
        t = t + f64(s)
    m = f32(f64(f32(t)) * 0.125)
    inv = f32(1.0) / np.log(f32(2.0))
    with np.errstate(all="ignore"):
        d = -(np.log(m) * inv) + f32(0.0)
    return f32(d)


# ---------------------------------------------------------------- pyramid -----------------------------
TAPS = [f32(0.375), f32(0.25), f32(0.0625)]


def store(v, fmt):
    if fmt == 2:
        return f32(v)
    hi = 255 if fmt == 0 else 65535
    t = cvtt(f32(v) + f32(0.5))
    return min(max(t, 0), hi)


def conv_line(line):
    n = len(line)
    count = 3
    left = min(n, count)
    mid = max(0, n - 2 * count)
    out = []
    for i in range(n):
        w0 = TAPS[0]
        acc = line[i] * w0
        if left <= i < left + mid:
            for k in range(count - 1, 0, -1):
                acc = acc + (line[i + k] + line[i - k]) * TAPS[k]
        else:
            wsum = w0
            for k in range(count - 1, 0, -1):
                if i + k < n:
                    wr = TAPS[k]
                    rt = wr * line[i + k]
                else:
                    wr = f32(0.0)
                    rt = f32(0.0)
                wsum = wsum + wr
                if i - k >= 0:
                    wl = TAPS[k]
                    lt = line[i - k] * wl
                else:
                    wl = f32(0.0)
                    lt = f32(0.0)
                wsum = wsum + wl
                acc = (acc + rt) + lt
            acc = acc / wsum
        out.append(acc)
    return out


def blur5(img, fmt):
    h, w, _ = img.shape
    tmp = np.zeros_like(img)
    for y in range(h):
        for ch in range(4):
            line = [f32(v) for v in img[y, :, ch]]
            for x, v in enumerate(conv_line(line)):
                tmp[y, x, ch] = store(v, fmt)
    out = np.zeros_like(img)
    for x in range(w):
        for ch in range(4):
            line = [f32(v) for v in tmp[:, x, ch]]
            for y, v in enumerate(conv_line(line)):
                out[y, x, ch] = store(v, fmt)
    return out


def bilerp4(img, x, y):
    h, w, _ = img.shape
    w1, h1 = w - 1, h - 1
    ix, iy = cvtt(x), cvtt(y)
    fx, fy = x - f32(ix), y - f32(iy)
    if ix >= w1:
        ix = w1 - 1
        fx = f32(1.0)
    if iy >= h1:
        iy = h1 - 1
        fy = f32(1.0)
    out = []
    for c in range(4):
        p00, p10 = f32(img[iy, ix, c]), f32(img[iy, ix + 1, c])
        p01, p11 = f32(img[iy + 1, ix, c]), f32(img[iy + 1, ix + 1, c])
        top = (p10 - p00) * fx + p00
        bot = (p11 - p01) * fx + p01
        out.append((bot - top) * fy + top)
    return out


def resample(src, dw, dh, fmt):
    sh, sw, _ = src.shape
    sx = (f32(sw) - f32(1.0)) / f32(dw - 1)
    sy = (f32(sh) - f32(1.0)) / f32(dh - 1)
    out = np.zeros((dh, dw, 4), DT[fmt])
    for y in range(dh):
        for x in range(dw):
            v = bilerp4(src, f32(x) * sx, f32(y) * sy)
            for c in range(4):
                out[y, x, c] = store(v[c], fmt)
    return out


def level_count(w, h):
    n, a, b = 1, max(w, 2), max(h, 2)
    while True:
        a //= 2
        if a < 2:
            break
        b //= 2
        if b < 2:
            break
        n += 1
    return n


def build_pyramid(src, fmt):
    h, w, _ = src.shape
    n = level_count(w, h)
    levels = [src]
    cw, ch = max(w, 2), max(h, 2)
    for i in range(1, n):
        tmp = blur5(levels[i - 1].copy(), fmt)
        cw, ch = max(cw // 2, 2), max(ch // 2, 2)
        levels.append(resample(tmp, cw, ch, fmt))
    return levels


def sample_pyramid(levels, u, v, d):
    def tap(img):
        h, w, _ = img.shape
        return bilerp4(img, f32(w - 1) * u, f32(h - 1) * v)

    n = len(levels)
    if f32(0.0) < d:
        if d < f32(n - 1):
            i = cvtt(d)
            frac = d - f32(i)
            a, b = tap(levels[i]), tap(levels[i + 1])
            return [(b[k] - a[k]) * frac + a[k] for k in range(4)]
        return tap(levels[n - 1])
    return tap(levels[0])


# ---------------------------------------------------------------- triangle (per edge description) -----
class Edge:
    pass


def make_edge(a, b, y_act):
    e = Edge()
    inv = f32(1.0) / (b[1] - a[1])
    t = ((y_act - f32(1.0)) - a[1]) * inv
    e.ay, e.by, e.y_act = a[1], b[1], y_act
    e.st = [(b[k] - a[k]) * t + a[k] for k in (0, 2, 3, 4)]  # x u v d
    e.dl = [(b[k] - a[k]) * inv for k in (0, 2, 3, 4)]
    return e


def triangle(tri, w1, h1):
    hits = []
    w1f, h1f = f32(w1), f32(h1)
    xs = [v[0] for v in tri]
    ys = [v[1] for v in tri]
    if not (any(x >= 0 for x in xs) and any(y >= 0 for y in ys) and any(x <= w1f for x in xs) and any(y <= h1f for y in ys)):
        return hits
    top, mid, bot = sorted(tri, key=lambda v: v[1])  # stable
    ystart = f32(max(math.ceil(float(top[1])), 0))
    yend = f32(math.floor(float(bot[1])))
    if h1f < yend:
        yend = h1f
    if ystart <= yend:
        edges = []
        for a, b in ((top, mid), (top, bot), (mid, bot)):
            ya = ystart if a[1] <= ystart else f32(math.ceil(float(a[1])))
            if ya <= yend and ya < b[1] and a[1] <= ya:
                edges.append(make_edge(a, b, ya))
        y = ystart
        while y <= yend:
            active = []
            for e in edges:
                if e.y_act <= y < e.by:
                    k = int(y - e.y_act) + 1
                    st = list(e.st)
                    # repeated addition from the activation scanline
                    for _ in range(k):
                        st = [st[j] + e.dl[j] for j in range(4)]
                    active.append(st)
            if len(active) == 2:
                a, b = active
                l, r = (b, a) if b[0] < a[0] else (a, b)
                x0 = max(cvtt(f32(math.ceil(float(l[0])))), 0)
                x1 = min(cvtt(f32(math.floor(float(r[0])))), w1)
                span = r[0] - l[0]
                if x0 <= x1 and f64(span) > 1e-08:
                    yi = cvtt(y)
                    if x0 == x1:
                        t = (f32(x0) - l[0]) * (f32(1.0) / span)
                        hits.append((x0, yi) + tuple((r[j] - l[j]) * t + l[j] for j in (1, 2, 3)))
                    else:
                        inv = f32(1.0) / span
                        dd = [r[j] - l[j] for j in (1, 2, 3)]
                        t = (f32(x0) - l[0]) * inv
                        cur = [dd[j - 1] * t + l[j] for j in (1, 2, 3)]
                        for x in range(x0, x1):
                            hits.append((x, yi) + tuple(cur))
                            cur = [cur[j] + dd[j] * inv for j in range(3)]
                        hits.append((x1, yi) + tuple(cur))
            y = y + f32(1.0)
    # degenerate pass
    if mid[1] == h1f and mid[1] == bot[1]:
        l, r = (bot, mid) if bot[0] <= mid[0] else (mid, bot)
        yt = f32(cvtt(l[1]))
        if f32(0.0) <= yt <= h1f and not (f64(l[1] - yt) > 0.01):
            x0 = max(cvtt(f32(math.ceil(float(l[0])))), 0)
            x1 = min(cvtt(f32(math.floor(float(r[0])))), w1)
            span = r[0] - l[0]
            if f64(span) > 1e-05:
                inv = f32(1.0) / span
                yi = cvtt(yt)
                if x0 == x1:
                    t = (f32(x0) - l[0]) * inv
                    hits.append((x0, yi) + tuple((r[j] - l[j]) * t + l[j] for j in (2, 3, 4)))
                else:
                    dd = [r[j] - l[j] for j in (2, 3, 4)]
                    t = (f32(x0) - l[0]) * inv
                    cur = [dd[j] * t + l[j + 2] for j in range(3)]
                    x = x0
                    while x < x1:
                        hits.append((x, yi) + tuple(cur))
                        cur = [cur[j] + dd[j] * inv for j in range(3)]
                        x += 1
                    hits.append((x, yi) + tuple(cur))
    return hits


# ---------------------------------------------------------------- the warp ----------------------------
def forward_warp(src, flow, amount, fmt):
    h, w, _ = src.shape
    a = -amount if f32(0.0) > amount else amount
    if f64(1e-05) > f64(a) or all(f64(abs(v)) <= 0.0001 for v in flow.reshape(-1)):
        return src.copy(), np.full((h, w), 255, np.uint8)
    pos = positions(flow, w, h, amount)
    n = level_count(w, h)
    levels = None
    acc = np.zeros((h, w, 4), f32)
    cnt = np.zeros((h, w), np.uint8)
    w1, h1 = w - 1, h - 1
    mips = None
    if n > 1:
        mips = np.zeros((h, w), f32)
        for r in range(h):
            for c in range(w):
                mips[r, c] = mip(flow, w, h, amount, c, r)

    def vtx(r, c):
        return (pos[r, c, 0], pos[r, c, 1], f32(c) / f32(w1), f32(r) / f32(h1), f32(0.0) if mips is None else mips[r, c])

    for r in range(h - 1):
        for c in range(w - 1):
            v00, v01, v10, v11 = vtx(r, c), vtx(r, c + 1), vtx(r + 1, c), vtx(r + 1, c + 1)
            for tri in ((v00, v01, v10), (v11, v01, v10)):
                for (x, y, u, v, d) in triangle(tri, w1, h1):
                    if f32(0.0) < d:
                        if levels is None:
                            levels = build_pyramid(src, fmt)
                        col = sample_pyramid(levels, u, v, d)
                    else:
                        col = bilerp4(src, f32(w1) * u, f32(h1) * v)
                    k = int(cnt[y, x])
                    if k == 0:
                        acc[y, x] = [c_ + f32(0.0) for c_ in col]
                        cnt[y, x] = 1
                    elif k == 255:
                        acc[y, x] = [(acc[y, x, j] * f32(255.0) + col[j]) * f32(0.00390625) + f32(0.0) for j in range(4)]
                    else:
                        nf = f32(k)
                        rr = f32(f64(1.0) / (f64(nf) + 1.0))
                        acc[y, x] = [(nf * acc[y, x, j] + col[j]) * rr + f32(0.0) for j in range(4)]
                        cnt[y, x] = k + 1
    out = np.zeros((h, w, 4), DT[fmt])
    for y in range(h):
        for x in range(w):
            for j in range(4):
                v = acc[y, x, j]
                if fmt == 2:
                    out[y, x, j] = v + f32(0.0)
                else:
                    t = int(np.trunc(v + f32(0.5)))
                    out[y, x, j] = t & (0xFF if fmt == 0 else 0xFFFF)
    cov = np.where(cnt != 0, 255, 0).astype(np.uint8)
    return out, cov


# ---------------------------------------------------------------- cases -------------------------------
def make_image(rng, w, h, fmt):
    yy, xx = np.mgrid[0:h, 0:w]
    base = 0.5 + 0.25 * np.sin(xx * 0.9 + 0.3) + 0.2 * np.cos(yy * 0.7) + 0.05 * rng.random((h, w))
    img = np.zeros((h, w, 4), np.float64)
    for ch in range(4):
        img[:, :, ch] = np.clip(base * (0.6 + 0.1 * ch) + 0.1 * rng.random((h, w)), 0.0, 1.0)
    if fmt == 0:
        return (img * 255.0).astype(np.uint8)
    if fmt == 1:
        return (img * 65535.0).astype(np.uint16)
    return img.astype(np.float32)


def flow_field(kind, fw, fh, rng):
    f = np.zeros((fh, fw, 2), np.float32)
    yy, xx = np.mgrid[0:fh, 0:fw]
    if kind == "shift_int":
        f[:, :, 0], f[:, :, 1] = 2.0, 1.0
    elif kind == "shift_frac":
        f[:, :, 0], f[:, :, 1] = 1.3, -0.7
    elif kind == "shear":
        f[:, :, 0] = 0.15 * yy
        f[:, :, 1] = -0.1 * xx
    elif kind == "squeeze":
        f[:, :, 0] = -0.5 * xx
    elif kind == "stretch":
        f[:, :, 0] = 0.3 * xx
        f[:, :, 1] = 0.2 * yy
    elif kind == "swirl":
        f[:, :, 0] = 2.5 * np.sin(yy * 0.6)
        f[:, :, 1] = 2.0 * np.cos(xx * 0.5)
    elif kind == "random":
        f[:, :, :] = (rng.random((fh, fw, 2)) - 0.5) * 3.0
    elif kind == "tiny":
        f[:, :, 0], f[:, :, 1] = 0.0003, 0.0
    return f


def hex_text(blob):
    """the bytes as hex text, 64 a line: the form the Rust tests read."""
    return "".join(blob[i:i + 64].hex() + "\n" for i in range(0, len(blob), 64))


def main(path):
    rng = np.random.default_rng(20261022)
    cases = []
    spec = [
        (0, 12, 9, 12, 9, "shift_int", 1.0), (1, 12, 9, 12, 9, "shift_int", 1.0), (2, 12, 9, 12, 9, "shift_int", 1.0),
        (0, 12, 9, 12, 9, "shift_frac", 0.5), (2, 12, 9, 12, 9, "shift_frac", 0.5),
        (0, 12, 9, 12, 9, "shift_frac", -1.0), (1, 10, 10, 10, 10, "shear", 1.0), (2, 10, 10, 10, 10, "shear", 0.7),
        (0, 16, 12, 16, 12, "squeeze", 1.0), (1, 16, 12, 16, 12, "squeeze", 1.0), (2, 16, 12, 16, 12, "squeeze", 1.0),
        (2, 16, 12, 16, 12, "squeeze", 0.6), (0, 16, 16, 16, 16, "stretch", 1.0), (2, 16, 16, 16, 16, "stretch", -0.8),
        (0, 14, 11, 14, 11, "swirl", 1.0), (1, 14, 11, 14, 11, "swirl", 1.0), (2, 14, 11, 14, 11, "swirl", 1.0),
        (0, 14, 11, 7, 6, "swirl", 1.0), (2, 13, 13, 7, 7, "random", 1.0), (0, 13, 13, 13, 13, "random", 0.8),
        (1, 13, 13, 13, 13, "random", 1.0), (2, 13, 13, 13, 13, "random", 1.0),
        (0, 9, 8, 9, 8, "tiny", 1.0), (0, 8, 8, 8, 8, "random", 0.0),
        (0, 16, 16, 8, 8, "squeeze", 1.0), (2, 17, 15, 17, 15, "random", 1.0),
    ]
    for (fmt, w, h, fw, fh, kind, amount) in spec:
        src = make_image(rng, w, h, fmt)
        flow = flow_field(kind, fw, fh, rng)
        out, cov = forward_warp(src, flow, f32(amount), fmt)
        cases.append((fmt, w, h, fw, fh, f32(amount), src, flow, out, cov))
        print(kind, fmt, w, h, "cov", int((cov != 0).sum()), "of", w * h, file=sys.stderr)
    with io.BytesIO() as fh_:
        fh_.write(b"F2C1")
        fh_.write(struct.pack("<I", len(cases)))
        for (fmt, w, h, fw, fh, amount, src, flow, out, cov) in cases:
            fh_.write(struct.pack("<IIIIIfI", fmt, w, h, fw, fh, float(amount), 1))
            fh_.write(src.astype("<" + DT[fmt]().dtype.str[1:]).tobytes())
            fh_.write(flow.astype("<f4").tobytes())
            fh_.write(out.astype("<" + DT[fmt]().dtype.str[1:]).tobytes())
            fh_.write(cov.tobytes())
        blob = fh_.getvalue()
    with open(path, "w", encoding="ascii", newline="\n") as out:
        out.write(hex_text(blob))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "f2_cases.hex")
