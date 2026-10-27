"""Independent numpy float32 model of the inverse warp, written with scalars.
It produces `warp_cases.hex` for the Rust test `warp::tests::model_fixtures`.

Run:  python -I gen_warp.py        (from this directory; needs numpy)

File format, little endian:
  b"WRP1", u32 ncases, then per case
    u32 depth (0 u8, 1 u16, 2 f32), u32 W, u32 H, u32 FW, u32 FH, f32 amount,
    u32 flags (bit0 hold, bit1 is360, bit2 coverage requested),
    src samples (W*H*4 of the depth, A R G B), flow (FW*FH*2 f32),
    expected out samples (W*H*4), expected coverage (W*H u8) when bit2 is set.

Every arithmetic step is done on np.float32 values in a fixed order; the integer store rule is
clamp(trunc(v + 0.5), 0, max), and faded integers are scaled as stored with the addition in double
precision. No Rust code of the engine is used.
"""
import struct
import sys

import numpy as np

f32 = np.float32


def bilerp(img, x, y):
    """img: (H, W, C) float32."""
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
    """Bilinear sample of a 360 degree image: the right neighbour wraps to column 0, the lower taps
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


def clip_to_rect(x, y, ix, iy, w1, h1):
    x, y, ix, iy, w1, h1 = f32(x), f32(y), f32(ix), f32(iy), f32(w1), f32(h1)
    zero = f32(0.0)
    for _ in range(1000):
        if ix < zero:
            iy = y - ((x - zero) / (x - ix)) * (y - iy)
            ix = zero
            continue
        if ix > w1:
            iy = y - ((x - w1) / (x - ix)) * (y - iy)
            ix = w1
            continue
        if iy < zero:
            ix = x - ((y - zero) / (y - iy)) * (x - ix)
            iy = zero
            continue
        if iy > h1:
            ix = x - ((y - h1) / (y - iy)) * (x - ix)
            iy = h1
            continue
        break
    return ix, iy


def store(v, depth):
    """In range store rule per depth. v: float32 vector of 4."""
    if depth == 2:
        return v.astype(np.float32)
    hi = 255 if depth == 0 else 65535
    q = np.trunc(v + f32(0.5)).astype(np.int64)
    return np.clip(q, 0, hi)


def store_faded(v, f, depth):
    if depth == 2:
        return (v * f).astype(np.float32)
    s = store(v, depth)
    scaled = (s.astype(np.float32) * f).astype(np.float64) + 0.5
    hi = 255 if depth == 0 else 65535
    return np.clip(np.trunc(scaled).astype(np.int64), 0, hi)


def wrap360(ix, iy, W, H):
    wf, hf = f32(W), f32(H)
    ix, iy = f32(ix), f32(iy)
    while ix < f32(0):
        ix = ix + wf
    while ix >= wf:
        ix = ix - wf
    n = 0
    while iy < f32(0):
        iy = iy + hf
        n += 1
    while iy >= hf:
        iy = iy - hf
        n += 1
    if n % 2 == 1:
        # the half width shift applies in all three depths
        ix = wf * f32(0.5) + ix
        if ix >= wf:
            ix = ix - wf
        if iy != f32(0):
            iy = hf - iy
    return ix, iy


def warp(src, flow, amount, hold, is360, depth):
    """src: (H, W, 4) integer or float array of the depth; flow (FH, FW, 2) float32.
    Returns (out, coverage)."""
    H, W = src.shape[0], src.shape[1]
    FH, FW = flow.shape[0], flow.shape[1]
    amount = f32(amount)
    cov = np.full((H, W), 255, np.uint8)
    out = src.copy()
    a = -amount if amount < f32(0) else amount
    if 0.005 > float(a):
        return out, cov
    r = f32(FW - 1) / f32(W - 1)
    q = f32(float(amount) / float(r))
    if q < f32(0):
        q = -q
    if 0.0001 > float(q):
        return out, cov

    img = src.astype(np.float32)
    fimg = flow.astype(np.float32)
    w1, h1 = f32(W - 1), f32(H - 1)
    wf, hf = f32(W), f32(H)
    rx = f32(FW - 1) / f32(W - 1)
    ry = f32(FH - 1) / f32(H - 1)
    sx = amount / rx
    sy = amount / ry
    for y in range(H):
        yf = f32(y)
        for x in range(W):
            xf = f32(x)
            d = bilerp(fimg, xf * rx, yf * ry)
            ix = d[0] * sx + xf
            iy = d[1] * sy + yf
            if is360:
                ix2, iy2 = wrap360(ix, iy, W, H)
                out[y, x] = store(bilerp360(img, ix2, iy2), depth)
            elif hold:
                ix, iy = clip_to_rect(xf, yf, ix, iy, w1, h1)
                out[y, x] = store(bilerp(img, ix, iy), depth)
            else:
                if not (ix > f32(-1) and ix < wf and iy > f32(-1) and iy < hf):
                    out[y, x] = 0
                    cov[y, x] = 0
                elif ix >= f32(0) and ix <= w1 and iy >= f32(0) and iy <= h1:
                    out[y, x] = store(bilerp(img, ix, iy), depth)
                    cov[y, x] = 255
                else:
                    if ix < f32(0):
                        f = ix + f32(1)
                    elif ix <= w1:
                        f = f32(1)
                    else:
                        f = wf - ix
                    if iy < f32(0):
                        f = f * (iy + f32(1))
                    elif iy > h1:
                        f = f * (hf - iy)
                    ix, iy = clip_to_rect(xf, yf, ix, iy, w1, h1)
                    out[y, x] = store_faded(bilerp(img, ix, iy), f, depth)
                    c = f * f32(255) + f32(0.5)
                    cov[y, x] = 255 if c > f32(255) else int(c)
    return out, cov


def make_src(rng, W, H, depth):
    if depth == 0:
        return rng.integers(0, 256, (H, W, 4), dtype=np.int64).astype(np.uint8)
    if depth == 1:
        return rng.integers(0, 65536, (H, W, 4), dtype=np.int64).astype(np.uint16)
    v = rng.normal(0.4, 0.6, (H, W, 4)).astype(np.float32)
    v[rng.random((H, W, 4)) < 0.05] = f32(0.0)
    return v


def make_flow(rng, kind, FW, FH, W, H):
    if kind == "gentle":
        return rng.normal(0.0, 2.0, (FH, FW, 2)).astype(np.float32)
    if kind == "strong":
        m = f32(1.5 * max(W, H))
        return rng.uniform(-m, m, (FH, FW, 2)).astype(np.float32)
    if kind == "edge":
        # pushes pixels over the borders by fractions of a pixel
        return rng.uniform(-3.0, 3.0, (FH, FW, 2)).astype(np.float32) * f32(max(W, H) / 8.0)
    if kind == "ramp":
        ys, xs = np.mgrid[0:FH, 0:FW].astype(np.float32)
        fx = (xs / f32(max(FW - 1, 1)) - f32(0.5)) * f32(1.3 * W)
        fy = (ys / f32(max(FH - 1, 1)) - f32(0.5)) * f32(0.9 * H)
        return np.stack([fx, fy], axis=-1).astype(np.float32)
    raise ValueError(kind)


def hex_text(blob):
    """the bytes as hex text, 64 a line: the form the Rust tests read."""
    return "".join(blob[i:i + 64].hex() + "\n" for i in range(0, len(blob), 64))


def main():
    rng = np.random.default_rng(20261022)
    cases = []
    sizes = [(19, 13, 19, 13), (16, 11, 31, 21), (14, 9, 7, 5), (11, 12, 11, 12)]
    configs = [
        ("gentle", 1.0),
        ("gentle", 0.37),
        ("edge", 0.8),
        ("strong", 1.0),
        ("ramp", -0.6),
        ("strong", 1.7),
    ]
    for depth in (0, 1, 2):
        for ci, (kind, amount) in enumerate(configs):
            W, H, FW, FH = sizes[ci % len(sizes)]
            src = make_src(rng, W, H, depth)
            flow = make_flow(rng, kind, FW, FH, W, H)
            # (hold, is360, coverage)
            for hold, is360, want_cov in ((False, False, True), (True, False, False), (False, True, False)):
                if is360 and ci % 2 == 1:
                    continue
                cases.append((depth, src, flow, amount, hold, is360, want_cov))
        # early outs and the flow ratio test: images 11 wide, flow 601 wide gives r = 60
        W, H = 11, 9
        src = make_src(rng, W, H, depth)
        flow601 = make_flow(rng, "gentle", 601, 9, W, H)
        cases.append((depth, src, flow601, 0.3, True, False, False))     # q = 0.005, not below 0.0001
        cases.append((depth, src, flow601, 0.006, True, False, False))    # q = 0.0001: |q| < 0.0001 false at the border
        cases.append((depth, src, flow601, 0.0059, False, False, True))   # q < 0.0001: copy path
        flow = make_flow(rng, "gentle", W, H, W, H)
        cases.append((depth, src, flow, 0.0049, False, False, True))      # below 0.005: copy path
        cases.append((depth, src, flow, 0.0051, False, False, True))      # just above: sampled

    blob = bytearray(b"WRP1" + struct.pack("<I", len(cases)))
    for depth, src, flow, amount, hold, is360, want_cov in cases:
        H, W = src.shape[0], src.shape[1]
        FH, FW = flow.shape[0], flow.shape[1]
        out, cov = warp(src, flow, amount, hold, is360, depth)
        flags = (1 if hold else 0) | (2 if is360 else 0) | (4 if want_cov else 0)
        blob += struct.pack("<IIIIIfI", depth, W, H, FW, FH, float(f32(amount)), flags)
        blob += src.tobytes()
        blob += flow.astype("<f4").tobytes()
        blob += out.astype(src.dtype).tobytes()
        if want_cov:
            blob += cov.tobytes()
    path = sys.argv[1] if len(sys.argv) > 1 else "warp_cases.hex"
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write(hex_text(blob))
    print(f"wrote {len(cases)} cases, {len(blob)} bytes to {path}")


if __name__ == "__main__":
    main()
