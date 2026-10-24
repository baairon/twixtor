"""Independent model of the vector image coding, written with numpy float32 / Python float64
scalars. It produces `vectors_cases.bin` for the Rust tests `vectors::tests::model_fixtures_*`.

Run:  python -I gen_vectors.py        (from this directory; needs numpy)

File format, little endian:
  b"VEC1", u32 ncases, then per case
    u32 kind (0 encode, 1 decode), u32 depth (0 u8, 1 u16, 2 f32), u32 W, u32 H, u32 FW, u32 FH, f32 scale
    encode: flow (FW*FH*2 f32), expected image (W*H*4 samples of the depth)
    decode: image (W*H*4 samples; FW = W, FH = H), expected flow (W*H*2 f32)
"""
import struct
import sys

import numpy as np

f32 = np.float32


def bilerp2(flow, x, y):
    """flow: (H, W, 2) float32."""
    H, W = flow.shape[0], flow.shape[1]
    w1, h1 = W - 1, H - 1
    ix, iy = int(x), int(y)
    fx = f32(x) - f32(ix)
    fy = f32(y) - f32(iy)
    if ix >= w1:
        ix, fx = w1 - 1, f32(1.0)
    if iy >= h1:
        iy, fy = h1 - 1, f32(1.0)
    p00, p10 = flow[iy, ix], flow[iy, ix + 1]
    p01, p11 = flow[iy + 1, ix], flow[iy + 1, ix + 1]
    top = (p10 - p00) * fx + p00
    bot = (p11 - p01) * fx + p01
    return (bot - top) * fy + top


def clamp1(v):
    if v > f32(1.0):
        return f32(1.0)
    if f32(-1.0) > v:
        return f32(-1.0)
    return v


def encode(flow, W, H, depth, scale):
    FH, FW = flow.shape[0], flow.shape[1]
    scale = f32(scale)
    dw1, dh1 = f32(W - 1), f32(H - 1)
    fw1, fh1 = f32(FW - 1), f32(FH - 1)
    sx = fw1 / dw1
    sy = fh1 / dh1
    kx = (dw1 / fw1) / scale
    ky = (dh1 / fh1) / scale
    dtype = [np.uint8, np.uint16, np.float32][depth]
    out = np.zeros((H, W, 4), dtype=dtype)
    for y in range(H):
        for x in range(W):
            d = bilerp2(flow, f32(x) * sx, f32(y) * sy)
            a = d[0] * kx
            b = d[1] * ky
            if depth == 2:
                out[y, x] = (f32(1.0), (a + f32(1.0)) * f32(0.5), (b + f32(1.0)) * f32(0.5), f32(0.0))
            else:
                a, b = clamp1(a) + f32(1.0), clamp1(b) + f32(1.0)
                mul = 127.0 if depth == 0 else 32767.0
                white = 255 if depth == 0 else 65535
                mask = 0xFF if depth == 0 else 0xFFFF
                r = int(float(a) * mul + 0.5) & mask
                g = int(float(b) * mul + 0.5) & mask
                out[y, x] = (white, r, g, 0)
    return out


def decode(img, depth, scale):
    H, W = img.shape[0], img.shape[1]
    scale64 = float(f32(scale))
    flow = np.zeros((H, W, 2), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            a, r, g = img[y, x, 0], img[y, x, 1], img[y, x, 2]
            if a == 0:
                continue
            if depth == 2:
                flow[y, x, 0] = (r * f32(2.0) - f32(1.0)) * f32(scale)
                flow[y, x, 1] = (g * f32(2.0) - f32(1.0)) * f32(scale)
            else:
                den = 254.0 if depth == 0 else 65534.0
                for k, c in ((0, r), (1, g)):
                    v = ((float(f32(c)) * 2.0) / den - 1.0) * scale64
                    flow[y, x, k] = np.float32(v)
    return flow


def random_flow(rng, FW, FH, spread):
    return rng.uniform(-spread, spread, (FH, FW, 2)).astype(np.float32)


def random_image(rng, W, H, depth):
    if depth == 0:
        img = rng.integers(0, 256, (H, W, 4), dtype=np.int64).astype(np.uint8)
        img[rng.random((H, W)) < 0.15, 0] = 0
    elif depth == 1:
        img = rng.integers(0, 65536, (H, W, 4), dtype=np.int64).astype(np.uint16)
        img[rng.random((H, W)) < 0.15, 0] = 0
    else:
        img = rng.uniform(0.0, 1.0, (H, W, 4)).astype(np.float32)
        img[rng.random((H, W)) < 0.15, 0] = f32(0.0)
    return img


def main():
    rng = np.random.default_rng(20261024)
    blob = bytearray(b"VEC1")
    cases = []
    for depth in (0, 1, 2):
        for (W, H, FW, FH, spread, scale) in (
            (13, 9, 13, 9, 4.0, 5.0),
            (16, 12, 8, 6, 3.0, 4.0),
            (10, 11, 21, 23, 30.0, 12.5),   # large vectors: clamps in the integer depths
            (17, 8, 17, 8, 0.01, 100.0),    # tiny vectors
        ):
            cases.append((0, depth, W, H, FW, FH, scale, random_flow(rng, FW, FH, spread)))
        for (W, H, scale) in ((14, 10, 7.5), (9, 9, 0.3), (12, 5, 250.0)):
            cases.append((1, depth, W, H, W, H, scale, random_image(rng, W, H, depth)))
    blob += struct.pack("<I", len(cases))
    for kind, depth, W, H, FW, FH, scale, data in cases:
        blob += struct.pack("<IIIIIIf", kind, depth, W, H, FW, FH, float(f32(scale)))
        if kind == 0:
            blob += data.astype("<f4").tobytes()
            blob += encode(data, W, H, depth, scale).tobytes()
        else:
            blob += data.tobytes()
            blob += decode(data, depth, scale).astype("<f4").tobytes()
    path = sys.argv[1] if len(sys.argv) > 1 else "vectors_cases.bin"
    with open(path, "wb") as fh:
        fh.write(blob)
    print(f"wrote {len(cases)} cases, {len(blob)} bytes to {path}")


if __name__ == "__main__":
    main()
