"""Generates the differential fixtures of tests/flow_d.rs from the numpy model of the solver
(`flow_ref.py`, float32, no block matching, not bit exact).

usage: python -I gen_flow_fixtures.py OUT_DIR
  OUT_DIR = the folder the .bin files are written to (tests/fixtures of the engine crate)

Each fixture `<name>.bin` (little endian):
  magic b"FLD2", u32 width, u32 height, f32 sensitivity, u32 quality, u32 prep
  (prep: 0 none, 1 Contrast/Edge Enhance, 2 Delinearize),
  width*height*4 bytes   X as 8 bit A,R,G,B (alpha 255, colour R,G,B of the luma),
  width*height*4 bytes   Y
  (w>>R)*(h>>R)*2 f32    expected flow of the model, (dx, dy) per pixel, X(p) ~ Y(p + out(p));
                         R = 0 for quality 5, 1 for quality 4 (high quality render)

The luma of the engine for 8 bit images is (R*0.3f + G*0.59f) + B*0.11f in single precision; the
model receives exactly that luma computed with numpy float32 scalars, so both sides start from
identical numbers.
"""
import struct
import sys

import numpy as np

import os

out_dir = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow_ref as fr  # noqa: E402

F = np.float32


def field(w, h, seed, ox=0.0, oy=0.0):
    """Smooth random field in 0..255 sampled at (x - ox, y - oy): a sum of fixed sinusoids with
    seeded phases (continuous in x, y so a subpixel shift is exact)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xx = xx - ox
    yy = yy - oy
    v = np.full((h, w), 128.0)
    for period, amp in ((41.0, 38.0), (23.0, 30.0), (13.0, 24.0), (8.3, 14.0)):
        ang = rng.uniform(0, 2 * np.pi)
        ph = rng.uniform(0, 2 * np.pi)
        v += amp * np.sin(2 * np.pi * (xx * np.cos(ang) + yy * np.sin(ang)) / period + ph)
    return np.clip(np.rint(v), 0, 255).astype(np.uint8)


def rgba(r, g, b):
    h, w = r.shape
    a = np.full((h, w), 255, np.uint8)
    return np.stack([a, r, g, b], -1)


def luma8(img):
    r = img[..., 1].astype(F)
    g = img[..., 2].astype(F)
    b = img[..., 3].astype(F)
    return ((r * F(0.3)) + (g * F(0.59))) + (b * F(0.11))


def write(name, w, h, sens, quality, prep, x, y, flow):
    with open(f"{out_dir}/{name}.bin", "wb") as f:
        f.write(struct.pack("<IIIfII", 0x32444C46, w, h, sens, quality, prep))
        f.write(np.ascontiguousarray(x, np.uint8).tobytes())
        f.write(np.ascontiguousarray(y, np.uint8).tobytes())
        f.write(np.ascontiguousarray(flow, "<f4").tobytes())


def reduce1(l):
    """one reduction stage of the pyramid builder: blur5, then corner aligned resample to half size."""
    k5 = np.array([0.375, 0.25, 0.0625], F)
    t = fr.conv_line_axis(fr.conv_line_axis(l, k5, 1), k5, 0)
    return fr.resample(t, l.shape[1] // 2, l.shape[0] // 2)


def delinearize(l):
    inv = F(1.0) / F(2.2)
    base = np.maximum(l, F(0.0)) / F(255.0)
    return (np.power(base, inv).astype(F) * F(255.0)).astype(F)


def edge_enhance(l):
    b = fr.blur(l, 2.0)
    g = fr.calc_grad(b)
    m = np.sqrt(g[..., 0] * g[..., 0] + g[..., 1] * g[..., 1]).astype(F)
    v = np.where(m <= F(20.0), m * F(0.05), F(1.0)).astype(F)
    return ((v * F(255.0) + b) * F(0.5)).astype(F)


def case(name, w, h, sens, shift, seeds, quality=5, prep=0):
    sx, sy = shift
    x = rgba(field(w, h, seeds[0]), field(w, h, seeds[1]), field(w, h, seeds[2]))
    y = rgba(field(w, h, seeds[0], sx, sy), field(w, h, seeds[1], sx, sy), field(w, h, seeds[2], sx, sy))
    lx, ly = luma8(x), luma8(y)
    if quality == 4:
        lx, ly = reduce1(lx), reduce1(ly)
    if prep == 2:
        lx, ly = delinearize(lx), delinearize(ly)
    elif prep == 1:
        lx, ly = edge_enhance(lx), edge_enhance(ly)
    out = fr.solve(lx, ly, sens)
    print(name, w, h, sens, "mean flow", out[8:-8, 8:-8].reshape(-1, 2).mean(0), "shape", out.shape)
    write(name, w, h, sens, quality, prep, x, y, out)


case("flow_a", 96, 80, 1.0, (2.5, -1.5), (1, 2, 3))
case("flow_b", 72, 64, 0.7, (-1.25, 2.0), (4, 5, 6))
case("flow_c", 64, 48, 0.85, (0.5, 0.5), (7, 8, 9))
case("flow_d", 101, 77, 0.8, (3.0, -2.0), (10, 11, 12), quality=4)
case("flow_e", 80, 64, 1.0, (1.5, 1.0), (13, 14, 15), prep=1)
case("flow_f", 80, 64, 1.0, (-1.5, 1.0), (16, 17, 18), prep=2)
