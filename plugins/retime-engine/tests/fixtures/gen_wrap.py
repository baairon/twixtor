"""Independent numpy float32 models of the wrap-around (360) primitives of the engine: the circular
horizontal convolution, the period keeping resample, the hole fill built on them, and the motion
blur with wrapped taps.
It produces `wrap_cases.hex` for the Rust tests `holefill::tests::wrap_fixtures`,
`warp::tests::blur::wrap_blur_fixtures` and the primitive tests in `flow::conv` and `flow::sample`.

Run:  python -I -B gen_wrap.py [OUT]        (from this directory; needs numpy)

Every arithmetic step is on np.float32 values in a fixed order (numpy does not fuse); `powf(2, e)`
of the Gaussian taps is evaluated in double and narrowed. No Rust code of the engine is used.

The rules modelled (parity gap G6, the arithmetic of the wrap modes is unknown in the original and
these are the rules this project states):
  * circular horizontal pass: every element is s = x[i] * k[0], then for j = n-1 down to 1
    s = s + (x[(i+j) mod L] + x[(i-j) mod L]) * k[j]; no border renormalisation. The vertical pass
    is the plain one (border elements renormalised, multi channel lines padded with -1.0);
  * period keeping resample: dst(x, y) = bilinear(src, x * (sw / dw), y * ((sh - 1) / (dh - 1)))
    where the right neighbour of the last column is column 0 and y keeps the plain cell rule;
  * hole fill: the pull-push pyramid of the plain fill with the two primitives above;
  * motion blur: the plain leaves with each counted tap's x wrapped into [0, W) by the loops of
    the 360 inverse warp; bilinear taps count when 0 <= y < H and sample with the 360 sampler,
    nearest taps are inside when 0 <= trunc(y) < H and read column trunc(x) of the wrapped x.

File format, little endian:
  b"WRA1", u32 ncases, then per case u32 kind and
    kind 1 (hole fill): u32 depth (0 u8, 1 u16, 2 f32), u32 W, u32 H, u32 mask_kind (1 u8, 2 u16,
      3 f32), image (W*H*4 samples of the depth, A R G B), mask (W*H of the mask type), expected
      out (W*H*4)
    kind 2 (motion blur): u32 depth, u32 W, u32 H, u32 FW, u32 FH, f32 amount, f32 shutter,
      u32 hiq, src (W*H*4 of the depth), flow (FW*FH*2 f32), expected out (W*H*4)
    kind 3 (convolution): u32 C, u32 W, u32 H, f32 radius (0: the 5 tap binomial filter),
      data (W*H*C f32), expected (W*H*C f32)
    kind 4 (resample): u32 SW, u32 SH, u32 DW, u32 DH, src (SW*SH f32), expected (DW*DH f32)
"""
import struct
import sys

import numpy as np

f32 = np.float32
WHITE = {0: f32(255.0), 1: f32(65535.0), 2: f32(1.0)}
DTYPE = {0: np.uint8, 1: np.uint16, 2: np.float32}
TINY = f32(1e-6)  # the squared length below which the bilinear leaf copies the pixel
WRAP_LIMIT = 1 << 16


# --------------------------------------------------------------------------------------------
# kernels
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


def kernel_for(radius):
    return BINOMIAL5 if radius == 0.0 else gaussian_kernel(radius)


# --------------------------------------------------------------------------------------------
# convolution: circular horizontal pass, plain vertical pass
# --------------------------------------------------------------------------------------------

def hpass_wrap(x, k):
    """x: (H, W, C) float32; the circular chain along axis 1."""
    n = len(k)
    acc = (x * k[0]).astype(f32)
    for j in range(n - 1, 0, -1):
        right = np.roll(x, -j, axis=1)   # x[(i + j) mod W]
        left = np.roll(x, j, axis=1)     # x[(i - j) mod W]
        acc = (acc + (right + left) * k[j]).astype(f32)
    return acc


def vpass_plain(x, k, multi):
    """x: (H, M) float32, every column an independent line; the plain border rule along axis 0
    (multi: the -1.0 padded multi channel rule)."""
    h, m = x.shape
    n = len(k)
    out = np.zeros_like(x)
    head = min(n, h)
    interior = max(h - 2 * n, 0)
    pad = f32(-1.0)

    def border(i):
        num = k[0] * x[i]
        den = np.full(m, k[0], dtype=f32)
        for j in range(n - 1, 0, -1):
            kj = k[j]
            for t in (i + j, i - j):
                if 0 <= t < h:
                    xv = x[t]
                    if multi:
                        wv = (np.where(xv != pad, f32(1.0), f32(0.0)) * kj).astype(f32)
                    else:
                        wv = np.full(m, kj, dtype=f32)
                else:
                    if multi:
                        xv = np.full(m, pad, dtype=f32)
                    else:
                        xv = np.zeros(m, dtype=f32)
                    wv = np.zeros(m, dtype=f32)
                den = (den + wv).astype(f32)
                num = (num + xv * wv).astype(f32)
        return (num / den).astype(f32)

    for i in range(head):
        out[i] = border(i)
    if interior > 0:
        a, b = head, head + interior
        acc = (x[a:b] * k[0]).astype(f32)
        for j in range(n - 1, 0, -1):
            acc = (acc + (x[a + j:b + j] + x[a - j:b - j]) * k[j]).astype(f32)
        out[a:b] = acc
    for i in range(head + interior, h):
        out[i] = border(i)
    return out


def convolve_wrap(x, k):
    """x: (H, W, C) float32."""
    h, w, c = x.shape
    t = hpass_wrap(x, k)
    return vpass_plain(t.reshape(h, w * c), k, c > 1).reshape(h, w, c)


def plane_blur_wrap(p, k):
    return convolve_wrap(p[:, :, None], k)[:, :, 0]


# --------------------------------------------------------------------------------------------
# period keeping resample
# --------------------------------------------------------------------------------------------

def resample_wrap(src, dw, dh):
    sh, sw = src.shape
    sx = f32(sw) / f32(dw)
    sy = (f32(sh) - f32(1.0)) / f32(dh - 1)
    xs = (np.arange(dw).astype(f32) * sx).astype(f32)
    ys = (np.arange(dh).astype(f32) * sy).astype(f32)
    ix = np.trunc(xs).astype(np.int64)
    fx = (xs - ix.astype(f32)).astype(f32)
    x1 = np.where(ix + 1 > sw - 1, 0, ix + 1)
    iy = np.trunc(ys).astype(np.int64)
    fy = (ys - iy.astype(f32)).astype(f32)
    last = iy >= sh - 1
    iy = np.where(last, sh - 2, iy)
    fy = np.where(last, f32(1.0), fy).astype(f32)
    IY, IX = np.meshgrid(iy, ix, indexing="ij")
    _, X1 = np.meshgrid(iy, x1, indexing="ij")
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    p00 = src[IY, IX]
    p10 = src[IY, X1]
    p01 = src[IY + 1, IX]
    p11 = src[IY + 1, X1]
    top = ((p10 - p00) * FX + p00).astype(f32)
    bot = ((p11 - p01) * FX + p01).astype(f32)
    return ((bot - top) * FY + top).astype(f32)


# --------------------------------------------------------------------------------------------
# hole fill with the wrapped passes
# --------------------------------------------------------------------------------------------

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


def fill_plane_wrap(plane, hole):
    h, w = plane.shape
    dims, top = geometry(w, h)
    g = gaussian_kernel(5.0)
    wts = [np.where(hole, f32(0.0), f32(1.0)).astype(f32)]
    vals = [np.where(hole, f32(0.0), plane).astype(f32)]
    for k in range(1, top + 1):
        cw, ch = dims[k]
        wts.append(resample_wrap(plane_blur_wrap(wts[k - 1], BINOMIAL5), cw, ch))
        vals.append(resample_wrap(plane_blur_wrap(vals[k - 1], BINOMIAL5), cw, ch))
    tw, th = dims[top]
    sol = np.zeros((th, tw), dtype=f32)

    def combine(sol, level):
        with np.errstate(all="ignore"):
            q = (vals[level] / wts[level]).astype(f32)
        return np.where(wts[level] != 0, q, sol).astype(f32)

    for level in range(top, -1, -1):
        sol = combine(sol, level)
        sol = plane_blur_wrap(sol, g)
        if level == 0:
            sol = combine(sol, 0)
        else:
            nw, nh = dims[level - 1]
            sol = resample_wrap(sol, nw, nh)
    return sol


def fill_model(img, hole, depth, w, h):
    px = img.reshape(h, w, 4)
    out = np.zeros_like(px)
    for c in range(4):
        plane = px[:, :, c].astype(f32)
        sol = fill_plane_wrap(plane, hole)
        if depth == 2:
            out[:, :, c] = sol
        else:
            white = WHITE[depth]
            y = (sol + f32(0.5)).astype(f32)
            t = np.trunc(np.clip(y, f32(0.0), f32(1e9))).astype(np.int64)
            out[:, :, c] = np.where(y < 0, 0, np.where(y > white, int(white), t)).astype(DTYPE[depth])
    return out.reshape(-1)


# --------------------------------------------------------------------------------------------
# motion blur with wrapped taps
# --------------------------------------------------------------------------------------------

def bilerp(img, x, y):
    H, W = img.shape[0], img.shape[1]
    w1, h1 = W - 1, H - 1
    ix, iy = int(x), int(y)
    fx = f32(x) - f32(ix)
    fy = f32(y) - f32(iy)
    if ix >= w1:
        ix, fx = w1 - 1, f32(1.0)
    if iy >= h1:
        iy, fy = h1 - 1, f32(1.0)
    p00, p10 = img[iy, ix], img[iy, ix + 1]
    p01, p11 = img[iy + 1, ix], img[iy + 1, ix + 1]
    top = (p10 - p00) * fx + p00
    bot = (p11 - p01) * fx + p01
    return (bot - top) * fy + top


def bilerp360(img, x, y):
    """The 360 sampler of the inverse warp: the right neighbour wraps to column 0, the lower taps
    come from the last row shifted by half the width when iy + 1 > h1."""
    H, W = img.shape[0], img.shape[1]
    w1, h1 = W - 1, H - 1
    ix, iy = int(x), int(y)
    fx = f32(x) - f32(ix)
    fy = f32(y) - f32(iy)
    x1 = 0 if ix + 1 > w1 else ix + 1
    if iy + 1 > h1:
        half = W // 2
        xc, xd, yc = (half + ix) % W, (half + x1) % W, h1
    else:
        xc, xd, yc = ix, x1, iy + 1
    ya = min(iy, h1)
    a, b, c, d = img[ya, ix], img[ya, x1], img[yc, xc], img[yc, xd]
    top = (b - a) * fx + a
    bot = (d - c) * fx + c
    return (bot - top) * fy + top


def wrap_x(x, W):
    wf = f32(W)
    x = f32(x)
    guard = 0
    while x < f32(0):
        x = x + wf
        guard += 1
        assert guard <= WRAP_LIMIT
    while wf <= x:
        x = x - wf
        guard += 1
        assert guard <= WRAP_LIMIT
    assert f32(0) <= x < wf
    return x


def store_blur(v, depth):
    if depth == 2:
        return v.astype(np.float32)
    hi = 255 if depth == 0 else 65535
    with np.errstate(all="ignore"):
        t = np.trunc(v + f32(0.5))
    t = np.where(np.isfinite(t), t, -2147483648.0).astype(np.int64)
    return np.clip(t, 0, hi)


def motion_blur_wrap(src, flow, amount, shutter, hiq, depth):
    H, W = src.shape[0], src.shape[1]
    FH, FW = flow.shape[0], flow.shape[1]
    amount, shutter = f32(amount), f32(shutter)
    img = src.astype(np.float32)
    fimg = flow.astype(np.float32)
    rx = f32(FW - 1) / f32(W - 1)
    ry = f32(FH - 1) / f32(H - 1)
    kx = (f32(0.5) / rx) * amount
    ky = (f32(0.5) / ry) * amount
    k = shutter * f32(2.0) - f32(1.0)
    out = src.copy()
    for y in range(H):
        yf = f32(y)
        for x in range(W):
            xf = f32(x)
            d = bilerp(fimg, xf * rx, yf * ry)
            vx = d[0] * kx
            vy = d[1] * ky
            d2 = vy * vy + vx * vx
            if hiq and TINY > d2:
                out[y, x] = src[y, x]
                continue
            dist = np.sqrt(d2)
            if shutter != f32(0.5):
                cx, cy = vx * k + xf, vy * k + yf
            else:
                cx, cy = xf, yf
            n = int(dist + f32(1.0))
            if n < 2:
                n = 2
                sx, sy = vx, vy
            else:
                sx = vx / f32(n - 1)
                sy = vy / f32(n - 1)
            total = np.zeros(4, dtype=np.float32)
            cnt = 0
            if hiq:
                px, py = cx, cy
                for _ in range(n):
                    if py >= f32(0) and py < f32(H):
                        cnt += 1
                        total = total + bilerp360(img, wrap_x(px, W), py)
                    px = px + sx
                    py = py + sy
                px, py = cx - sx, cy - sy
                for _ in range(n - 1):
                    if py >= f32(0) and py < f32(H):
                        cnt += 1
                        total = total + bilerp360(img, wrap_x(px, W), py)
                    px = px + (-sx)
                    py = py + (-sy)
            else:
                px, py = cx + f32(0.5), cy + f32(0.5)
                for _ in range(n):
                    iy = int(py)
                    if 0 <= iy < H:
                        cnt += 1
                        total = total + img[iy, int(wrap_x(px, W))]
                    px = px + sx
                    py = py + sy
                px, py = (cx + f32(0.5)) - sx, (cy + f32(0.5)) - sy
                for _ in range(n - 1):
                    iy = int(py)
                    if 0 <= iy < H:
                        total = total + img[iy, int(wrap_x(px, W))]
                    cnt += 1
                    px = px + (-sx)
                    py = py + (-sy)
            with np.errstate(all="ignore"):
                inv = f32(1.0) / f32(cnt) if cnt else f32(np.inf)
                out[y, x] = store_blur(inv * total, depth)
    return out


# --------------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------------

def blob_mask(rng, w, h, kind):
    m = np.zeros((h, w), dtype=bool)
    if kind == "none":
        return m
    if kind == "all":
        return ~m
    if kind == "single":
        m[rng.integers(0, h), rng.integers(0, w)] = True
        return m
    if kind == "seam":
        # a block across the left and right border: a hole on the seam of the periodic image
        y0 = int(rng.integers(0, h))
        y1 = min(h, y0 + max(1, h // 3))
        r = max(1, w // 5)
        m[y0:y1, :r] = True
        m[y0:y1, w - r:] = True
        return m
    if kind == "border":
        m[:, 0] = True
        m[0, :] = True
        return m
    if kind == "scatter":
        return rng.random((h, w)) < 0.15
    for _ in range(int(rng.integers(1, 4))):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x1, y1 = min(w, x0 + int(rng.integers(1, max(2, w // 2)))), min(h, y0 + int(rng.integers(1, max(2, h // 2))))
        m[y0:y1, x0:x1] = True
    return m


def random_image(rng, w, h, depth):
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([
        0.5 + 0.4 * np.sin(xx * (2.0 * np.pi / w) + yy / 7.0),
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
        part = hole & (rng.random((h, w)) < 0.3)
        return np.where(part, rng.integers(1, 255, (h, w)), m).astype(np.uint8)
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


def make_src(rng, W, H, depth):
    if depth == 0:
        return rng.integers(0, 256, (H, W, 4), dtype=np.int64).astype(np.uint8)
    if depth == 1:
        return rng.integers(0, 65536, (H, W, 4), dtype=np.int64).astype(np.uint16)
    v = rng.normal(0.4, 0.6, (H, W, 4)).astype(np.float32)
    v[rng.random((H, W, 4)) < 0.05] = f32(0.0)
    return v


def make_flow(rng, kind, FW, FH, W):
    if kind == "small":
        return rng.normal(0.0, 1.0, (FH, FW, 2)).astype(np.float32)
    if kind == "mixed":
        v = rng.normal(0.0, 3.0, (FH, FW, 2)).astype(np.float32)
        v[rng.random((FH, FW)) < 0.25] = f32(0.0)
        return v
    if kind == "large":
        return rng.uniform(-12.0, 12.0, (FH, FW, 2)).astype(np.float32)
    if kind == "sweep":
        # long horizontal vectors: most taps cross the seam, some more than once
        v = rng.uniform(-1.0, 1.0, (FH, FW, 2)).astype(np.float32)
        v[:, :, 0] = (v[:, :, 0] * f32(2.5 * W)).astype(np.float32)
        return v
    if kind == "tiny":
        return rng.normal(0.0, 0.0003, (FH, FW, 2)).astype(np.float32)
    raise ValueError(kind)


def hex_text(blob):
    """the bytes as hex text, 64 a line: the form the Rust tests read."""
    return "".join(blob[i:i + 64].hex() + "\n" for i in range(0, len(blob), 64))


def main(path):
    rng = np.random.default_rng(20261027)
    blob = bytearray()
    count = 0

    # kind 3: the circular convolution, 1 to 3 channels, kernels longer than the line included
    for c, (w, h), radius in [
        (1, (13, 7), 0.0), (1, (4, 9), 5.0), (1, (2, 6), 5.0), (1, (1, 5), 0.0), (1, (19, 3), 7.0),
        (2, (11, 8), 5.0), (2, (3, 12), 2.0), (3, (16, 6), 7.0), (3, (5, 5), 0.0),
    ]:
        data = rng.normal(0.0, 3.0, (h, w, c)).astype(np.float32)
        data[rng.random((h, w, c)) < 0.05] = f32(-1.0)  # the multi channel padding value
        out = convolve_wrap(data, kernel_for(radius))
        blob += struct.pack("<IIIIf", 3, c, w, h, float(radius))
        blob += data.astype("<f4").tobytes() + out.astype("<f4").tobytes()
        count += 1

    # kind 4: the period keeping resample, down and up
    for sw, sh, dw, dh in [
        (4, 6, 2, 3), (5, 7, 2, 3), (2, 4, 4, 8), (2, 5, 5, 9), (7, 5, 3, 2), (3, 3, 7, 5),
        (16, 9, 8, 4), (9, 4, 18, 8), (1, 4, 2, 7), (33, 2, 16, 2),
    ]:
        src = rng.normal(0.0, 10.0, (sh, sw)).astype(np.float32)
        out = resample_wrap(src, dw, dh)
        blob += struct.pack("<IIIII", 4, sw, sh, dw, dh)
        blob += src.astype("<f4").tobytes() + out.astype("<f4").tobytes()
        count += 1

    # kind 1: the hole fill with the wrapped passes
    sizes = [(12, 9), (17, 17), (33, 20), (5, 40), (2, 2), (3, 3), (64, 12), (2, 9), (31, 3), (8, 8), (16, 5)]
    kinds = ["seam", "rects", "scatter", "single", "border", "all", "none"]
    for si, (w, h) in enumerate(sizes):
        for depth in (0, 1, 2):
            for ki, kind in enumerate(kinds):
                if (si + depth + ki) % 3 != 0 and kind != "seam":
                    continue
                hole0 = blob_mask(rng, w, h, kind)
                mask_kind = 1 + (si + ki + depth) % 3
                m = mask_arrays(rng, hole0, mask_kind)
                hole = hole_from_mask(m, mask_kind)
                img = random_image(rng, w, h, depth)
                out = fill_model(img, hole, depth, w, h) if hole.any() else img
                blob += struct.pack("<IIIII", 1, depth, w, h, mask_kind)
                blob += np.ascontiguousarray(img).tobytes()
                blob += np.ascontiguousarray(m.reshape(-1)).tobytes()
                blob += np.ascontiguousarray(out).tobytes()
                count += 1

    # kind 2: the motion blur with wrapped taps
    bsizes = [(16, 11, 16, 11), (13, 9, 7, 5), (10, 12, 19, 23)]
    configs = [
        ("mixed", 1.0, 0.5),
        ("sweep", 1.0, 0.5),
        ("large", 0.7, 0.5),
        ("small", 3.5, 0.5),
        ("tiny", 1.0, 0.5),
        ("sweep", 0.6, 0.0),   # shifted centre (reachable only through the primitive)
        ("mixed", 2.0, 1.0),
    ]
    for depth in (0, 1, 2):
        for ci, (kind, amount, shutter) in enumerate(configs):
            W, H, FW, FH = bsizes[ci % len(bsizes)]
            src = make_src(rng, W, H, depth)
            flow = make_flow(rng, kind, FW, FH, W)
            for hiq in (True, False):
                out = motion_blur_wrap(src, flow, amount, shutter, hiq, depth)
                blob += struct.pack("<IIIIIIffI", 2, depth, W, H, FW, FH, float(f32(amount)), float(f32(shutter)), 1 if hiq else 0)
                blob += src.tobytes()
                blob += flow.astype("<f4").tobytes()
                blob += out.astype(src.dtype).tobytes()
                count += 1

    blob = bytearray(b"WRA1" + struct.pack("<I", count)) + blob
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write(hex_text(blob))
    print(f"wrote {count} cases, {len(blob)} bytes to {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "wrap_cases.hex")
