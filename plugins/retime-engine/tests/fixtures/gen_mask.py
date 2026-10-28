"""Independent numpy float32 models of the solver hooks of the layered and the wrap-around solves:
the mask pyramid, the gradient (plain and wrapped) with gradient masking, the B vector step with
the warped mask test and the wrapped target, and the block matching sweeps with the mask gate and
the wrapped cost. Written with scalars where the Rust code works per pixel.

The wrap models are the periodic ones of rule G6: the gradient is the plain arithmetic with the
columns modulo the width; a target column is split exactly into a whole column modulo the width
and a fraction (`flow::wrap_target`); the block matching sweeps every column of a ring, the
horizontal sweeps reading the neighbours as they stood before the sweep.
It produces `mask_cases.hex` for the Rust test `flow::tests_extras::model_fixtures`.

Run:  python -I -B gen_mask.py OUT        (needs numpy; OUT defaults to mask_cases.hex)

Every arithmetic step is done on np.float32 values in a fixed order (numpy does not fuse a
multiply and an add), and no Rust code of the engine is used.

File format, little endian:
  b"MSK1", u32 ncases, then per case u32 kind and
  kind 1 (mask pyramid): u32 W, u32 H, u32 quality, u32 hiq, matte f32[W*H],
      u32 nlevels, then per level u32 w, u32 h, f32[w*h]
  kind 2 (gradient): u32 W, u32 H, u32 wrap, u32 has_mask, image f32[W*H],
      mask f32[W*H] (when has_mask), expected gradient f32[W*H*2]
  kind 3 (B vector step): u32 W, u32 H, u32 wrap, u32 has_mask, I1 f32[W*H], I0 f32[W*H],
      gradient f32[W*H*2], flow f32[W*H*2], mask of Y f32[W*H] (when has_mask),
      expected B f32[W*H*2]
  kind 4 (block matching sweeps): u32 W, u32 H, u32 wrap, u32 has_gate, A f32[W*H],
      B f32[W*H], flow f32[W*H*2], gate f32[W*H] (when has_gate), expected flow f32[W*H*2]
"""
import io
import struct
import sys

import numpy as np

f32 = np.float32
ZERO = f32(0.0)
ONE = f32(1.0)
MASK_ON = f32(0.95)
WARPED_MASK_MIN = f32(0.999)
WRAP_LIMIT = 1 << 16
BINOMIAL5 = [f32(0.375), f32(0.25), f32(0.0625)]


# --------------------------------------------------------------------------------------------
# Geometry of the solver: reduction (with the tiny frames rule), level count and sizes
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
    return r, w0, h0


def level_count(w0, h0, min_size=2):
    levels = 1
    w, h = w0, h0
    while True:
        w //= 2
        if w < min_size:
            break
        h //= 2
        if h < min_size:
            break
        if w > 2 and h > 2 and not (w == 2 and h == 2):
            levels += 1
        else:
            break
    return levels


def level_sizes(w0, h0, levels):
    sizes = []
    w, h = w0, h0
    for _ in range(levels):
        sizes.append((w, h))
        w = max(w // 2, 2)
        h = max(h // 2, 2)
    return sizes


# --------------------------------------------------------------------------------------------
# Pyramid filters: the 5 tap binomial blur (border renormalised) and the corner aligned resample
# --------------------------------------------------------------------------------------------

def conv_axis(x, k, axis):
    """1 channel line convolution along `axis` of a 2D float32 array (rows are independent)."""
    if axis == 0:
        return conv_axis(x.T.copy(), k, 1).T.copy()
    h, n_len = x.shape
    n = len(k)
    out = np.zeros_like(x)
    head = min(n, n_len)
    interior = max(n_len - 2 * n, 0)

    def border(i):
        num = k[0] * x[:, i]
        den = k[0]
        for j in range(n - 1, 0, -1):
            kj = k[j]
            for t in (i + j, i - j):
                if 0 <= t < n_len:
                    xv, wv = x[:, t], kj
                else:
                    xv, wv = np.zeros(h, dtype=f32), ZERO
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
    for i in range(head + interior, n_len):
        out[:, i] = border(i)
    return out


def blur5(x):
    return conv_axis(conv_axis(x, BINOMIAL5, 1), BINOMIAL5, 0)


def cell(c, n):
    """index and fraction of the plain sampler: at or past the last index the cell moves one
    back with the fraction 1."""
    i = int(c)
    f = f32(c - f32(i))
    if i >= n - 1:
        i, f = n - 2, ONE
    return i, f


def bilinear(img, x, y):
    h, w = img.shape
    ix, fx = cell(x, w)
    iy, fy = cell(y, h)
    p00, p10 = img[iy, ix], img[iy, ix + 1]
    p01, p11 = img[iy + 1, ix], img[iy + 1, ix + 1]
    top = (p10 - p00) * fx + p00
    bot = (p11 - p01) * fx + p01
    return (bot - top) * fy + top


def bilinear_wrap(img, x, y):
    """x already in [0, W): no edge rule, the right neighbour of the last column is column 0."""
    h, w = img.shape
    ix = int(x)
    fx = f32(x - f32(ix))
    x1 = 0 if ix + 1 > w - 1 else ix + 1
    iy, fy = cell(y, h)
    p00, p10 = img[iy, ix], img[iy, x1]
    p01, p11 = img[iy + 1, ix], img[iy + 1, x1]
    top = (p10 - p00) * fx + p00
    bot = (p11 - p01) * fx + p01
    return (bot - top) * fy + top


def resample(src, dw, dh):
    sh, sw = src.shape
    sx = (f32(sw) - ONE) / f32(dw - 1)
    sy = (f32(sh) - ONE) / f32(dh - 1)
    out = np.zeros((dh, dw), dtype=f32)
    for y in range(dh):
        fy = f32(y) * sy
        for x in range(dw):
            out[y, x] = bilinear(src, f32(x) * sx, fy)
    return out


def binarize(x):
    return np.where(x >= MASK_ON, ONE, ZERO).astype(f32)


def mask_pyramid(matte, q, hiq):
    h, w = matte.shape
    r, w0, h0 = flow_resolution(q, hiq, w, h)
    n = level_count(w0, h0)
    sizes = level_sizes(w0, h0, n)
    cur = binarize(matte)
    for _ in range(r):
        ch, cw = cur.shape
        cur = binarize(resample(blur5(cur.copy()), cw // 2, ch // 2))
    assert cur.shape == (h0, w0)
    levels = []
    for i in range(n):
        levels.append(cur)
        if i + 1 < n:
            nw, nh = sizes[i + 1]
            cur = binarize(resample(blur5(cur.copy()), nw, nh))
    return levels


# --------------------------------------------------------------------------------------------
# Gradient, gradient masking, wrap loops, B vector step
# --------------------------------------------------------------------------------------------

def gradient(img, wrap):
    h, w = img.shape
    g = np.zeros((h, w, 2), dtype=f32)
    for y in range(h):
        for x in range(w):
            c = img[y, x]
            nx, dx, ny, dy = ZERO, ZERO, ZERO, ZERO
            if x >= 1:
                nx = (c - img[y, x - 1]) + ZERO
                dx = ONE
            if y >= 1:
                ny = (c - img[y - 1, x]) + ZERO
                dy = ONE
            if x + 1 <= w - 1:
                nx = nx + (img[y, x + 1] - c)
                dx = dx + ONE
            if y + 1 <= h - 1:
                ny = ny + (img[y + 1, x] - c)
                dy = dy + ONE
            if wrap:
                nx = (c - img[y, (x - 1) % w]) + ZERO
                nx = nx + (img[y, (x + 1) % w] - c)
                g[y, x, 0] = nx / f32(2.0)
            else:
                g[y, x, 0] = nx / dx
            g[y, x, 1] = ny / dy
    return g


def mask_gradient(g, mask, wrap):
    h, w = mask.shape
    g = g.copy()
    for y in range(h):
        for x in range(w):
            near = [(x, y)]
            if y >= 1:
                near.append((x, y - 1))
            if y + 1 < h:
                near.append((x, y + 1))
            if wrap:
                near += [((x - 1) % w, y), ((x + 1) % w, y)]
            else:
                if x >= 1:
                    near.append((x - 1, y))
                if x + 1 < w:
                    near.append((x + 1, y))
            if any(mask[yy, xx] < MASK_ON for (xx, yy) in near):
                g[y, x] = ZERO
    return g


def wrap_target(x, u, w):
    """the whole column modulo the width and the fraction of the target x - u, or None past
    WRAP_LIMIT widths (or NaN)."""
    d = f32(-u)
    if not (abs(d) <= f32(WRAP_LIMIT) * f32(w)):
        return None
    whole = f32(np.floor(d))
    f = f32(d - whole)
    c = (x + int(whole)) % w
    if f >= ONE:
        c = (c + 1) % w
        f = ZERO
    return c, f


def bilinear_at(img, ix, fx, y):
    """the column ix plus the fraction fx; the right neighbour of the last column is column 0."""
    h, w = img.shape
    x1 = 0 if ix + 1 > w - 1 else ix + 1
    iy, fy = cell(y, h)
    p00, p10 = img[iy, ix], img[iy, x1]
    p01, p11 = img[iy + 1, ix], img[iy + 1, x1]
    top = (p10 - p00) * fx + p00
    bot = (p11 - p01) * fx + p01
    return (bot - top) * fy + top


def b_step(i1, i0, g, flow, mask, wrap):
    h, w = i1.shape
    fw, fh = f32(w), f32(h)
    out = np.zeros((h, w, 2), dtype=f32)
    for y in range(h):
        fy = f32(y)
        for x in range(w):
            fx = f32(x)
            sx = fx - flow[y, x, 0]
            sy = fy - flow[y, x, 1]
            if wrap:
                if sy < ZERO or fh - ONE < sy:
                    continue
                t = wrap_target(x, flow[y, x, 0], w)
                if t is None:
                    continue

                def sample(img, sx_, sy_, t=t):
                    return bilinear_at(img, t[0], t[1], sy_)
            else:
                if sx < ZERO or fw - ONE < sx or sy < ZERO or fh - ONE < sy:
                    continue
                sample = bilinear
            if mask is not None and not (sample(mask, sx, sy) >= WARPED_MASK_MIN):
                continue
            s = sample(i0, sx, sy)
            r = s - i1[y, x]
            out[y, x, 0] = r * g[y, x, 0]
            out[y, x, 1] = r * g[y, x, 1]
    return out


# --------------------------------------------------------------------------------------------
# Block matching sweeps
# --------------------------------------------------------------------------------------------

def block_sum(b, p, cols, iy, fx, fy):
    w11 = fy * fx
    w10 = fx - w11
    w01 = fy - w11
    w00 = ((ONE - fx) - fy) + w11

    def t(i, j):
        return b[iy - 1 + j, cols[i]]

    sq = []
    n = 0
    for bb in range(3):
        for aa in range(3):
            q = ((w00 * t(aa, bb) + w10 * t(aa + 1, bb)) + w01 * t(aa, bb + 1)) + w11 * t(aa + 1, bb + 1)
            d = p[n] - q
            sq.append(d * d)
            n += 1
    s = sq[0] + sq[1]
    for k in range(2, 8):
        s = s + sq[k]
    return sq[8] + s


def row_of(y2, hm2):
    if y2 == hm2:
        return int(hm2 - ONE), ONE
    iy = int(y2)
    return iy, f32(y2 - f32(iy))


def cost(b, p, x2, y2, wrap):
    """the plain cost at (x2, y2); with wrap x2 is the split target (column, fraction)."""
    h, w = b.shape
    hm2 = f32(h - 2)
    if wrap:
        if x2 is None or y2 > hm2:
            return None
        ix, fx = x2
        iy, fy = row_of(y2, hm2)
        if iy < 1:
            return None
        cols = [(ix - 1) % w, ix, (ix + 1) % w, (ix + 2) % w]
    else:
        wm2 = f32(w - 2)
        if x2 > wm2 or y2 > hm2:
            return None
        if x2 == wm2:
            ix, fx = int(wm2 - ONE), ONE
        else:
            ix = int(x2)
            fx = f32(x2 - f32(ix))
        iy, fy = row_of(y2, hm2)
        if ix < 1 or iy < 1:
            return None
        cols = [ix - 1, ix, ix + 1, ix + 2]
    return block_sum(b, p, cols, iy, fx, fy)


def block_match(a, b, flow, gate, wrap):
    h, w = a.shape
    f = flow.copy()
    if w < 3 or h < 3:
        return f
    w1, h1 = f32(w - 1), f32(h - 1)

    def step(x, y, cur, n):
        if gate is not None and gate[y, x] < MASK_ON:
            return None
        tx = f32(x) - cur[0]
        ty = f32(y) - cur[1]
        if wrap:
            ok = ZERO <= ty and ty <= h1
        else:
            ok = ZERO <= tx and tx <= w1 and ZERO <= ty and ty <= h1
        if not ok:
            return None
        if wrap:
            p = [a[y + r, (x + c) % w] for r in (-1, 0, 1) for c in (-1, 0, 1)]
            c0 = cost(b, p, wrap_target(x, cur[0], w), ty, wrap)
        else:
            p = [a[y + r, x + c] for r in (-1, 0, 1) for c in (-1, 0, 1)]
            c0 = cost(b, p, tx, ty, wrap)
        if c0 is None:
            return None
        if wrap:
            c1 = cost(b, p, wrap_target(x, n[0], w), f32(y) - n[1], wrap)
        else:
            c1 = cost(b, p, f32(x) - n[0], f32(y) - n[1], wrap)
        if c1 is None:
            return None
        return n if c1 < c0 else None

    def visit(x, y, nx, ny):
        v = step(x, y, f[y, x].copy(), f[ny, nx].copy())
        if v is not None:
            f[y, x] = v

    if wrap:
        # every row a ring, the horizontal sweeps read the row as it stood before the sweep
        for dx in (-1, 1):
            for y in range(1, h - 1):
                old = f[y].copy()
                for x in range(w):
                    v = step(x, y, old[x].copy(), old[(x + dx) % w].copy())
                    if v is not None:
                        f[y, x] = v
        for x in range(w):
            for y in range(1, h - 1):
                visit(x, y, x, y - 1)
        for x in range(w):
            for y in range(h - 2, 0, -1):
                visit(x, y, x, y + 1)
        return f
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            visit(x, y, x - 1, y)
    for y in range(1, h - 1):
        for x in range(w - 2, 0, -1):
            visit(x, y, x + 1, y)
    for x in range(1, w - 1):
        for y in range(1, h - 1):
            visit(x, y, x, y - 1)
    for x in range(1, w - 1):
        for y in range(h - 2, 0, -1):
            visit(x, y, x, y + 1)
    return f


# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------

def smooth(rng, w, h, lo, hi):
    """a smooth random field in [lo, hi] as float32."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    v = np.zeros((h, w))
    for _ in range(4):
        period = rng.uniform(3.0, 2.0 * max(w, h))
        ang = rng.uniform(0, 2 * np.pi)
        ph = rng.uniform(0, 2 * np.pi)
        v += np.sin(2 * np.pi * (xx * np.cos(ang) + yy * np.sin(ang)) / period + ph)
    v = (v - v.min()) / max(v.max() - v.min(), 1e-9)
    return (lo + (hi - lo) * v).astype(f32)


def image(rng, w, h):
    v = smooth(rng, w, h, 0.0, 255.0) + rng.normal(0.0, 6.0, (h, w)).astype(f32)
    return v.astype(f32)


def blobs(rng, w, h, kind):
    """binary masks: discs, a half plane, sparse holes, all ones or all zeros."""
    if kind == "ones":
        return np.ones((h, w), dtype=f32)
    if kind == "zeros":
        return np.zeros((h, w), dtype=f32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    if kind == "half":
        return np.where(xx < rng.uniform(0.3, 0.7) * w, ONE, ZERO).astype(f32)
    if kind == "holes":
        return np.where(rng.random((h, w)) < 0.12, ZERO, ONE).astype(f32)
    if kind == "seam":
        # a band across the left and right edges
        return np.where((xx < 0.25 * w) | (xx >= 0.8 * w), ONE, ZERO).astype(f32)
    m = np.zeros((h, w), dtype=bool)
    for _ in range(3):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(0.15, 0.4) * min(w, h)
        m |= (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
    return np.where(m, ONE, ZERO).astype(f32)


def matte(rng, w, h, kind):
    """mattes in 0..1 with soft edges and values on both sides of 0.95."""
    if kind == "flat":
        return np.full((h, w), f32(rng.uniform(0.9, 1.0)), dtype=f32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    if kind == "edge":
        v = np.clip((xx - 0.45 * w) / max(1.0, 0.1 * w) * -1.0 + 0.5, 0.0, 1.0)
    elif kind == "disc":
        cx, cy, r = 0.5 * w, 0.5 * h, 0.35 * min(w, h)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        v = np.clip((r - d) / 2.0 + 0.5, 0.0, 1.0)
    else:
        v = smooth(rng, w, h, 0.0, 1.0).astype(np.float64)
    v = v.astype(f32)
    # sprinkle the threshold itself, the float below it and exact 0 and 1
    t = rng.random((h, w))
    v[t < 0.05] = MASK_ON
    v[(t >= 0.05) & (t < 0.1)] = np.nextafter(MASK_ON, ZERO)
    v[(t >= 0.1) & (t < 0.12)] = ONE
    v[(t >= 0.12) & (t < 0.14)] = ZERO
    return v.astype(f32)


def flows(rng, w, h, wrap):
    """flows with fractional parts, targets past every edge and, with wrap, several widths
    away; some vectors place the target 2^-10 and 2^-9 next to a pixel centre."""
    f = rng.uniform(-3.0, 3.0, (h, w, 2)).astype(f32)
    pick = rng.random((h, w))
    near = rng.integers(-2, 3, (h, w, 2)).astype(f32)
    tiny = np.where(rng.random((h, w, 2)) < 0.5, f32(2.0 ** -10), f32(2.0 ** -9))
    sign = np.where(rng.random((h, w, 2)) < 0.5, f32(-1.0), ONE)
    f = np.where((pick < 0.3)[..., None], (near + tiny * sign).astype(f32), f)
    f = np.where((pick >= 0.3)[..., None] & (pick < 0.4)[..., None], near, f)
    if wrap:
        far = rng.integers(-3, 4, (h, w)).astype(f32) * f32(w)
        sel = (pick >= 0.4) & (pick < 0.55)
        f[..., 0] = np.where(sel, (f[..., 0] + far).astype(f32), f[..., 0])
    return f.astype(f32)


def hex_text(blob):
    """the bytes as hex text, 64 a line: the form the Rust tests read."""
    return "".join(blob[i:i + 64].hex() + "\n" for i in range(0, len(blob), 64))


def main(path):
    rng = np.random.default_rng(20261027)
    out = io.BytesIO()
    cases = 0

    def u32(*v):
        out.write(struct.pack("<" + "I" * len(v), *v))

    def arr(a):
        out.write(np.ascontiguousarray(a, dtype="<f4").tobytes())

    # kind 1: mask pyramids
    sizes = [(2, 2), (3, 2), (5, 3), (6, 6), (9, 4), (17, 16), (37, 29), (64, 48), (80, 60), (33, 120)]
    kinds = ["edge", "disc", "smooth", "flat"]
    for si, (w, h) in enumerate(sizes):
        for q in (1, 2, 3, 4, 5):
            for hiq in (1, 0):
                if (si + q + hiq) % 2 == 1 and w * h > 400:
                    continue
                if (si + q + hiq) % 3 != 0 and w * h > 2000:
                    continue
                m = matte(rng, w, h, kinds[(si + q + hiq) % len(kinds)])
                levels = mask_pyramid(m, q, bool(hiq))
                u32(1, w, h, q, hiq)
                arr(m)
                u32(len(levels))
                for lv in levels:
                    u32(lv.shape[1], lv.shape[0])
                    arr(lv)
                cases += 1

    # kind 2: gradients
    mask_kinds = ["discs", "half", "holes", "seam", "ones", "zeros"]
    for si, (w, h) in enumerate([(2, 2), (4, 3), (7, 5), (16, 12), (23, 9), (31, 17)]):
        for wrap in (0, 1):
            for has_mask in (0, 1):
                img = image(rng, w, h)
                mk = blobs(rng, w, h, mask_kinds[(si + wrap) % len(mask_kinds)]) if has_mask else None
                g = gradient(img, wrap)
                if has_mask:
                    g = mask_gradient(g, mk, wrap)
                u32(2, w, h, wrap, has_mask)
                arr(img)
                if has_mask:
                    arr(mk)
                arr(g)
                cases += 1

    # kind 3: B vector steps
    for si, (w, h) in enumerate([(2, 2), (3, 3), (5, 4), (12, 9), (20, 14), (27, 11)]):
        for wrap in (0, 1):
            for has_mask in (0, 1):
                i1, i0 = image(rng, w, h), image(rng, w, h)
                g = rng.normal(0.0, 20.0, (h, w, 2)).astype(f32)
                fl = flows(rng, w, h, wrap)
                mk = blobs(rng, w, h, mask_kinds[(si + 2 * wrap + has_mask) % len(mask_kinds)]) if has_mask else None
                b = b_step(i1, i0, g, fl, mk, wrap)
                u32(3, w, h, wrap, has_mask)
                arr(i1)
                arr(i0)
                arr(g)
                arr(fl)
                if has_mask:
                    arr(mk)
                arr(b)
                cases += 1

    # kind 4: block matching sweeps
    for si, (w, h) in enumerate([(3, 3), (4, 5), (9, 7), (14, 10), (18, 13)]):
        for wrap in (0, 1):
            for has_gate in (0, 1):
                a = image(rng, w, h)
                shift = rng.integers(-2, 3)
                b = (np.roll(a, shift, axis=1) + rng.normal(0.0, 2.0, (h, w))).astype(f32)
                fl = rng.integers(-3, 4, (h, w, 2)).astype(f32) * f32(0.5)
                # about 40% of the pixels hold the vector of the shift, (-shift, 0) in the
                # solver's sign
                sel = rng.random((h, w)) < 0.4
                fl[sel, 0] = f32(-shift)
                fl[sel, 1] = ZERO
                gate = blobs(rng, w, h, ["discs", "holes", "half", "seam"][(si + wrap) % 4]) if has_gate else None
                res = block_match(a, b, fl, gate, wrap)
                u32(4, w, h, wrap, has_gate)
                arr(a)
                arr(b)
                arr(fl)
                if has_gate:
                    arr(gate)
                arr(res)
                cases += 1

    blob = b"MSK1" + struct.pack("<I", cases) + out.getvalue()
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(hex_text(blob))
    print(f"wrote {cases} cases to {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mask_cases.hex")
