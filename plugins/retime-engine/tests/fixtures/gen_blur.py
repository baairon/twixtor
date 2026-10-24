"""Independent numpy float32 model of the flow directed motion blur primitive, written with scalars.
It produces `blur_cases.bin` for the Rust test `warp::tests::blur_model_fixtures`.

Run:  python -I gen_blur.py        (from this directory; needs numpy)

File format, little endian:
  b"BLR1", u32 ncases, then per case
    u32 depth (0 u8, 1 u16, 2 f32), u32 W, u32 H, u32 FW, u32 FH, f32 amount, f32 shutter, u32 hiq,
    src samples (W*H*4 of the depth, A R G B), flow (FW*FH*2 f32), expected dest samples (W*H*4).

The model has two tap modes: the bilinear leaf (copy below 1e-6 squared length, n taps forward from
the centre, n - 1 backward from centre - step, taps counted only inside) and the nearest leaf (taps
at trunc(p + 0.5), forward counted when inside, backward always counted).
"""
import struct
import sys

import numpy as np

f32 = np.float32
TINY = np.frombuffer(struct.pack("<I", 0x358637BD), dtype="<f4")[0]


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


def store(v, depth):
    if depth == 2:
        return v.astype(np.float32)
    hi = 255 if depth == 0 else 65535
    t = np.trunc(v + f32(0.5)).astype(np.int64)
    return np.clip(t, 0, hi)


def motion_blur(src, flow, amount, shutter, hiq, depth):
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
                    if px >= f32(0) and px < f32(W) and py >= f32(0) and py < f32(H):
                        cnt += 1
                        total = total + bilerp(img, px, py)
                    px = px + sx
                    py = py + sy
                px, py = cx - sx, cy - sy
                for _ in range(n - 1):
                    if px >= f32(0) and px < f32(W) and py >= f32(0) and py < f32(H):
                        cnt += 1
                        total = total + bilerp(img, px, py)
                    px = px + (-sx)
                    py = py + (-sy)
            else:
                px, py = cx + f32(0.5), cy + f32(0.5)
                for _ in range(n):
                    ix, iy = int(px), int(py)
                    if 0 <= ix < W and 0 <= iy < H:
                        cnt += 1
                        total = total + img[iy, ix]
                    px = px + sx
                    py = py + sy
                px, py = (cx + f32(0.5)) - sx, (cy + f32(0.5)) - sy
                for _ in range(n - 1):
                    ix, iy = int(px), int(py)
                    if 0 <= ix < W and 0 <= iy < H:
                        total = total + img[iy, ix]
                    cnt += 1
                    px = px + (-sx)
                    py = py + (-sy)
            inv = f32(1.0) / f32(cnt) if cnt else f32(np.inf)
            out[y, x] = store(inv * total, depth)
    return out


def make_src(rng, W, H, depth):
    if depth == 0:
        return rng.integers(0, 256, (H, W, 4), dtype=np.int64).astype(np.uint8)
    if depth == 1:
        return rng.integers(0, 65536, (H, W, 4), dtype=np.int64).astype(np.uint16)
    v = rng.normal(0.4, 0.6, (H, W, 4)).astype(np.float32)
    v[rng.random((H, W, 4)) < 0.05] = f32(0.0)
    return v


def make_flow(rng, kind, FW, FH):
    if kind == "small":
        return rng.normal(0.0, 1.0, (FH, FW, 2)).astype(np.float32)
    if kind == "mixed":
        v = rng.normal(0.0, 3.0, (FH, FW, 2)).astype(np.float32)
        v[rng.random((FH, FW)) < 0.25] = f32(0.0)
        return v
    if kind == "large":
        return rng.uniform(-12.0, 12.0, (FH, FW, 2)).astype(np.float32)
    if kind == "tiny":
        return rng.normal(0.0, 0.0003, (FH, FW, 2)).astype(np.float32)
    raise ValueError(kind)


def main():
    rng = np.random.default_rng(20261023)
    cases = []
    sizes = [(21, 15, 21, 15), (18, 13, 9, 7), (12, 11, 25, 21)]
    configs = [
        ("mixed", 1.0, 0.5),
        ("small", 3.5, 0.5),
        ("large", 0.7, 0.5),
        ("mixed", 20.0, 0.5),
        ("tiny", 1.0, 0.5),
        ("mixed", 1.0, 0.0),   # shutter shifted centre (reachable only through the primitive)
        ("small", 2.0, 1.0),
    ]
    for depth in (0, 1, 2):
        for ci, (kind, amount, shutter) in enumerate(configs):
            W, H, FW, FH = sizes[ci % len(sizes)]
            src = make_src(rng, W, H, depth)
            flow = make_flow(rng, kind, FW, FH)
            for hiq in (True, False):
                cases.append((depth, src, flow, amount, shutter, hiq))

    blob = bytearray(b"BLR1" + struct.pack("<I", len(cases)))
    for depth, src, flow, amount, shutter, hiq in cases:
        H, W = src.shape[0], src.shape[1]
        FH, FW = flow.shape[0], flow.shape[1]
        out = motion_blur(src, flow, amount, shutter, hiq, depth)
        blob += struct.pack("<IIIIIffI", depth, W, H, FW, FH, float(f32(amount)), float(f32(shutter)), 1 if hiq else 0)
        blob += src.tobytes()
        blob += flow.astype("<f4").tobytes()
        blob += out.astype(src.dtype).tobytes()
    path = sys.argv[1] if len(sys.argv) > 1 else "blur_cases.bin"
    with open(path, "wb") as fh:
        fh.write(blob)
    print(f"wrote {len(cases)} cases, {len(blob)} bytes to {path}")


if __name__ == "__main__":
    main()
