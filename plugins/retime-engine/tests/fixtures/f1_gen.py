"""Independent numpy float32 models of Smart Blend (with its binary alpha variant) and of the
hole fill.
They produce `f1_cases.bin` for the Rust tests `smartblend::tests::model_fixtures` and
`holefill::tests::model_fixtures`.

Run:  python -I f1_gen.py        (from this directory; needs numpy)

The hole fill is a pull-push pyramid. Every arithmetic step is on np.float32 in a fixed order. No
Rust code of the engine is used. Float math is IEEE single precision per element (numpy does not
fuse); `powf(2, e)` of the Gaussian taps is evaluated in double and narrowed.

File format, little endian:
  b"F1C1", u32 ncases, then per case
    u32 kind (1 smart blend, 2 smart blend binary alpha, 3 hole fill)
    u32 depth (0 u8, 1 u16, 2 f32), u32 W, u32 H, f32 weight (kinds 1 and 2, else 0),
    u32 mask_kind (kind 3: 1 u8 mask, 2 u16 mask, 3 f32 mask, else 0)
    kind 1, 2: p1 (W*H*4 samples of the depth, A R G B), p2, expected out
    kind 3:    image (W*H*4), mask (W*H of the mask type), expected out (W*H*4)
"""
import struct
import sys

import numpy as np

f32 = np.float32
WHITE = {0: f32(255.0), 1: f32(65535.0), 2: f32(1.0)}
DTYPE = {0: np.uint8, 1: np.uint16, 2: np.float32}


# --------------------------------------------------------------------------------------------
# Smart Blend and its binary alpha variant
# --------------------------------------------------------------------------------------------

def over(a, b, white):
    a0, b0 = a[:, 0], b[:, 0]
    ia = f32(1.0) - a0 / white
    alpha = (white - b0) * (a0 / white) + b0
    rgb = b[:, 1:] * ia[:, None] + a[:, 1:]
    res = np.concatenate([alpha[:, None], rgb], axis=1).astype(f32)
    res = np.where((a0 == 0)[:, None], b, res)
    res = np.where((a0 == white)[:, None], a, res)
    return res.astype(f32)


def store(v, raw, depth):
    """v: (N, 4) float32, raw: (N,) bool."""
    if depth == 2:
        plain = (v + f32(0.0)).astype(f32)
        return np.where(raw[:, None], v, plain).astype(f32)
    white = WHITE[depth]
    y = (v + f32(0.5)).astype(f32)
    t = np.trunc(np.clip(y, f32(0.0), f32(1e9))).astype(np.int64)
    plain = np.where(y < 0, 0, np.where(y > white, int(white), t))
    rawv = np.trunc(v).astype(np.int64)
    out = np.where(raw[:, None], rawv, plain)
    return out.astype(DTYPE[depth])


def smart_model(p1, p2, w, depth):
    white = WHITE[depth]
    a = p1.reshape(-1, 4).astype(f32)
    b = p2.reshape(-1, 4).astype(f32)
    w = f32(w)
    with np.errstate(all="ignore"):
        c1 = over(a, b, white)
        c2 = over(b, a, white)
        amax = np.where(a[:, 0] > b[:, 0], a[:, 0], b[:, 0]).astype(f32)
        v = ((c2 - c1) * w + c1).astype(f32)
        eq = v[:, 0] == amax
        zero = (v[:, 0] == 0) & ~eq
        pick = np.where((c1[:, 0] > c2[:, 0])[:, None], c1, c2)
        norm = v.copy()
        norm[:, 1:] = (v[:, 1:] * amax[:, None]) / v[:, 0][:, None]
        norm[:, 0] = amax
        res = np.where(eq[:, None], v, np.where(zero[:, None], pick, norm)).astype(f32)
        return store(res, zero, depth).reshape(-1)


def binary_model(p1, p2, w, depth):
    white = WHITE[depth]
    a = p1.reshape(-1, 4).astype(f32)
    b = p2.reshape(-1, 4).astype(f32)
    w = f32(w)
    with np.errstate(all="ignore"):
        zero4 = np.zeros_like(a)
        # X: P2 opaque -> P1 over P2 (opaque result); P1 opaque -> P1; else 0
        ia1 = f32(1.0) - a[:, 0] / white
        x_over = np.concatenate([b[:, :1], b[:, 1:] * ia1[:, None] + a[:, 1:]], axis=1)
        x = np.where((b[:, 0] == white)[:, None], x_over,
                     np.where((a[:, 0] == white)[:, None], a, zero4)).astype(f32)
        ia2 = f32(1.0) - b[:, 0] / white
        y_over = np.concatenate([a[:, :1], a[:, 1:] * ia2[:, None] + b[:, 1:]], axis=1)
        y = np.where((a[:, 0] == white)[:, None], y_over,
                     np.where((b[:, 0] == white)[:, None], b, zero4)).astype(f32)
        v = ((y - x) * w + x).astype(f32)
        return store(v, np.zeros(len(v), dtype=bool), depth).reshape(-1)


def random_pixels(rng, n, depth):
    white = float(WHITE[depth])
    alpha = rng.random(n)
    pick = rng.random(n)
    alpha = np.where(pick < 0.2, 0.0, np.where(pick < 0.5, 1.0, alpha))
    col = rng.random((n, 3))
    # mostly valid premultiplied (colour <= alpha), some invalid ones
    valid = rng.random(n) < 0.85
    col = np.where(valid[:, None], col * alpha[:, None], col)
    px = np.concatenate([alpha[:, None], col], axis=1) * white
    if depth == 2:
        # a few float alphas outside 0..1 and exotic values
        k = rng.random(n)
        px[:, 0] = np.where(k < 0.03, 2.0, np.where(k < 0.05, -0.25, px[:, 0]))
        return px.astype(np.float32).reshape(-1)
    return np.floor(px + 0.5).astype(DTYPE[depth]).reshape(-1)


# --------------------------------------------------------------------------------------------
# Hole fill: pull-push pyramid
# --------------------------------------------------------------------------------------------

def exp2_f32(e):
    return f32(np.float64(2.0) ** np.float64(e))


def gaussian_kernel(r):
    r = f32(r)
    n = int(r) + 1
    s3 = r / f32(3.0)
    d = s3 * s3
    k = []
    s = f32(0.0)
    for i in range(n):
        fi = f32(i)
        t = fi * fi
        e = -(t / d)
        v = exp2_f32(e)
        k.append(v)
        s = s + v
    den = s * f32(2.0) - k[0]
    return [v / den for v in k]


BINOMIAL5 = [f32(0.375), f32(0.25), f32(0.0625)]


def conv_axis(x, k, axis):
    """1 channel line convolution along `axis` of a 2D float32 array (rows are independent)."""
    if axis == 0:
        return conv_axis(x.T.copy(), k, 1).T.copy()
    h, L = x.shape
    n = len(k)
    out = np.zeros_like(x)
    head = min(n, L)
    interior = max(L - 2 * n, 0)

    def border(i):
        num = k[0] * x[:, i]
        den = k[0]
        for j in range(n - 1, 0, -1):
            kj = k[j]
            for t in (i + j, i - j):
                if 0 <= t < L:
                    xv, wv = x[:, t], kj
                else:
                    xv, wv = np.zeros(h, dtype=f32), f32(0.0)
                den = den + wv
                num = num + xv * wv
        return (num / den).astype(f32)

    for i in range(head):
        out[:, i] = border(i)
    if interior > 0:
        a, b = head, head + interior
        acc = (x[:, a:b] * k[0]).astype(f32)
        for j in range(n - 1, 0, -1):
            acc = (acc + (x[:, a + j:b + j] + x[:, a - j:b - j]) * k[j]).astype(f32)
        out[:, a:b] = acc
    for i in range(head + interior, L):
        out[:, i] = border(i)
    return out


def convolve(x, k):
    return conv_axis(conv_axis(x, k, 1), k, 0)


def blur5(x):
    return convolve(x, BINOMIAL5)


def gaussian(x, r):
    return convolve(x, gaussian_kernel(r))


def bilinear_grid(src, xs, ys):
    h, w = src.shape
    last_x, last_y = w - 1, h - 1

    def cell(c, last):
        i = np.trunc(c).astype(np.int64)
        f = (c - i.astype(f32)).astype(f32)
        over_ = i >= last
        i = np.where(over_, last - 1, i)
        f = np.where(over_, f32(1.0), f).astype(f32)
        return i, f

    ix, fx = cell(xs, last_x)
    iy, fy = cell(ys, last_y)
    IY, IX = np.meshgrid(iy, ix, indexing="ij")
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    p00 = src[IY, IX]
    p10 = src[IY, IX + 1]
    p01 = src[IY + 1, IX]
    p11 = src[IY + 1, IX + 1]
    top = ((p10 - p00) * FX + p00).astype(f32)
    bot = ((p11 - p01) * FX + p01).astype(f32)
    return ((bot - top) * FY + top).astype(f32)


def resample(src, dw, dh):
    sh, sw = src.shape
    sx = (f32(sw) - f32(1.0)) / f32(dw - 1)
    sy = (f32(sh) - f32(1.0)) / f32(dh - 1)
    xs = (np.arange(dw).astype(f32) * sx).astype(f32)
    ys = (np.arange(dh).astype(f32) * sy).astype(f32)
    return bilinear_grid(src, xs, ys)


def geometry(w, h):
    a, b, count = w, h, 1
    while True:
        a //= 2
        b //= 2
        a = max(a, 2)
        b = max(b, 2)
        count += 1
        if not (a > 1 and b > 1 and (a != 2 or b != 2)):
            break
    dims = []
    cw, ch = w, h
    for _ in range(count + 1):
        dims.append((cw, ch))
        cw, ch = max(cw // 2, 2), max(ch // 2, 2)
    return dims, count - 1


def fill_plane(plane, hole):
    h, w = plane.shape
    dims, top = geometry(w, h)
    wts = [np.where(hole, f32(0.0), f32(1.0)).astype(f32)]
    vals = [np.where(hole, f32(0.0), plane).astype(f32)]
    for k in range(1, top + 1):
        cw, ch = dims[k]
        wts.append(resample(blur5(wts[k - 1]), cw, ch))
        vals.append(resample(blur5(vals[k - 1]), cw, ch))
    tw, th = dims[top]
    sol = np.zeros((th, tw), dtype=f32)

    def combine(sol, level):
        with np.errstate(all="ignore"):
            q = (vals[level] / wts[level]).astype(f32)
        return np.where(wts[level] != 0, q, sol).astype(f32)

    for level in range(top, -1, -1):
        sol = combine(sol, level)
        sol = gaussian(sol, 5.0)
        if level == 0:
            sol = combine(sol, 0)
        else:
            nw, nh = dims[level - 1]
            sol = resample(sol, nw, nh)
    return sol


def fill_model(img, hole, depth, w, h):
    px = img.reshape(h, w, 4)
    out = np.zeros_like(px)
    for c in range(4):
        plane = px[:, :, c].astype(f32)
        sol = fill_plane(plane, hole)
        if depth == 2:
            out[:, :, c] = sol
        else:
            white = WHITE[depth]
            y = (sol + f32(0.5)).astype(f32)
            t = np.trunc(np.clip(y, f32(0.0), f32(1e9))).astype(np.int64)
            out[:, :, c] = np.where(y < 0, 0, np.where(y > white, int(white), t)).astype(DTYPE[depth])
    return out.reshape(-1)


def blob_mask(rng, w, h, kind):
    m = np.zeros((h, w), dtype=bool)
    if kind == "none":
        return m
    if kind == "all":
        return ~m
    if kind == "single":
        m[rng.integers(0, h), rng.integers(0, w)] = True
        return m
    if kind == "border":
        m[:, 0] = True
        m[0, :] = True
        return m
    if kind == "scatter":
        return rng.random((h, w)) < 0.15
    # rectangles
    for _ in range(int(rng.integers(1, 4))):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x1, y1 = min(w, x0 + int(rng.integers(1, max(2, w // 2)))), min(h, y0 + int(rng.integers(1, max(2, h // 2))))
        m[y0:y1, x0:x1] = True
    return m


def random_image(rng, w, h, depth):
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([
        0.5 + 0.4 * np.sin(xx / 5.0 + yy / 7.0),
        xx / max(w - 1, 1),
        yy / max(h - 1, 1),
        0.3 + 0.6 * rng.random((h, w)),
    ], axis=-1)
    if depth == 2:
        return (base * 1.0).astype(np.float32).reshape(-1)
    white = float(WHITE[depth])
    return np.floor(base * white + 0.5).astype(DTYPE[depth]).reshape(-1)


def mask_arrays(rng, hole, mask_kind):
    h, w = hole.shape
    if mask_kind == 1:
        m = np.where(hole, 255, 0).astype(np.uint8)
        # some partial coverage values: any non zero is a hole
        part = hole & (rng.random((h, w)) < 0.3)
        m = np.where(part, rng.integers(1, 255, (h, w)), m).astype(np.uint8)
        return m
    if mask_kind == 2:
        m = np.where(hole, 65535, 0).astype(np.uint16)
        part = hole & (rng.random((h, w)) < 0.3)
        return np.where(part, rng.integers(1, 65535, (h, w)), m).astype(np.uint16)
    m = np.where(hole, 1.0, 0.0).astype(np.float32)
    part = hole & (rng.random((h, w)) < 0.3)
    return np.where(part, rng.random((h, w)) * 0.9 + 0.0001, m).astype(np.float32)


def hole_from_mask(m, mask_kind):
    if mask_kind == 1:
        return m.astype(np.float64) >= 0.001
    return m.astype(np.float64) >= 3.1e-5


def main(path):
    rng = np.random.default_rng(20261022)
    cases = []
    # smart blend and binary alpha: three depths, weights, many pixels
    for kind, model in ((1, smart_model), (2, binary_model)):
        for depth in (0, 1, 2):
            for w in (0.0, 0.25, 0.5, 0.7, 1.0, 0.123456):
                n_w, n_h = 37, 29
                p1 = random_pixels(rng, n_w * n_h, depth)
                p2 = random_pixels(rng, n_w * n_h, depth)
                out = model(p1, p2, w, depth)
                cases.append((kind, depth, n_w, n_h, w, 0, [p1, p2, out]))
    # hole fill
    sizes = [(12, 9), (17, 17), (33, 20), (5, 40), (2, 2), (3, 3), (64, 48), (2, 9), (31, 3), (8, 8)]
    kinds = ["rects", "scatter", "single", "border", "all", "none"]
    for si, (w, h) in enumerate(sizes):
        for depth in (0, 1, 2):
            for ki, kind in enumerate(kinds):
                if (si + depth + ki) % 3 != 0 and kind not in ("all",):
                    continue
                if w * h > 1000 and kind != "rects":
                    continue
                hole0 = blob_mask(rng, w, h, kind)
                mask_kind = 1 + (si + ki + depth) % 3
                m = mask_arrays(rng, hole0, mask_kind)
                hole = hole_from_mask(m, mask_kind)
                img = random_image(rng, w, h, depth)
                out = fill_model(img, hole, depth, w, h)
                cases.append((3, depth, w, h, 0.0, mask_kind, [img, m.reshape(-1), out]))
    with open(path, "wb") as f:
        f.write(b"F1C1")
        f.write(struct.pack("<I", len(cases)))
        for kind, depth, w, h, wt, mk, arrays in cases:
            f.write(struct.pack("<IIIIfI", kind, depth, w, h, wt, mk))
            for a in arrays:
                f.write(np.ascontiguousarray(a).tobytes())
    print(f"wrote {len(cases)} cases to {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "f1_cases.bin")
