"""Independent numpy float32 model of the Debug views of create-vectors (rule G4): Warp,
Blend, Diff, Detail Compare and Edge Compare of `debug::debug_view`. It produces
`debug_cases.hex` for the Rust test `debug::tests::model_fixtures_are_reproduced_bit_for_bit`.

Run:  python -I -B gen_debug.py [OUT]        (from any directory; needs numpy)

The forward warp is the model of `f2_gen.py`, the separable convolution (the 5 tap binomial and
the Gaussian) the model of `f1_gen.py` and the gradient the model of `gen_mask.py`, imported from
this directory. No Rust code of the engine is used. Every arithmetic step is on np.float32 in the
order of the Rust code (numpy does not fuse); the luma of 16 bpc and float is double precision,
narrowed.

File format, little endian:
  b"DBG1", u32 ncases, then per case
    u32 fmt (0 u8, 1 u16, 2 f32), u32 W, u32 H, u32 FW, u32 FH, i32 code (0 Blend, 1 Diff,
    2 Detail Compare, 3 Edge Compare, 4 Warp);
    x (W*H*4 samples of the format, A R G B), y (the same), flow (FW*FH*2 f32), expected (W*H*4)
"""
import io
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f1_gen as C  # noqa: E402
import f2_gen as F  # noqa: E402
import gen_mask as GM  # noqa: E402

f32 = np.float32
f64 = np.float64
DT = {0: np.uint8, 1: np.uint16, 2: np.float32}
WHITE = {0: 255, 1: 65535, 2: f32(1.0)}


def luma(img, fmt):
    """prep::luma without the alpha term and in full range: (h, w) float32 in 0..255."""
    h, w, _ = img.shape
    out = np.zeros((h, w), dtype=f32)
    for y in range(h):
        for x in range(w):
            _, r, g, b = img[y, x]
            if fmt == 0:
                out[y, x] = f32(f32(f32(r) * f32(0.3)) + f32(f32(g) * f32(0.59))) + f32(f32(b) * f32(0.11))
            else:
                s = (f64(g) * 0.59 + f64(r) * 0.3) + f64(b) * 0.11
                out[y, x] = f32(s / 257.0) if fmt == 1 else f32(s * 255.0)
    return out


def clamp01(v):
    if v > f32(1.0):
        return f32(1.0)
    if v > f32(0.0):
        return v
    return f32(0.0)


def detail(l):
    b = C.blur5(l)
    h, w = l.shape
    out = np.zeros((h, w), dtype=f32)
    for y in range(h):
        for x in range(w):
            out[y, x] = clamp01(f32(f32(f32(l[y, x] - b[y, x]) / f32(255.0)) + f32(0.5)))
    return out


def edge(l):
    b = C.gaussian(l, f32(2.0))
    g = GM.gradient(b, False)
    h, w = l.shape
    out = np.zeros((h, w), dtype=f32)
    for y in range(h):
        for x in range(w):
            gx, gy = g[y, x, 0], g[y, x, 1]
            m = f32(np.sqrt(f32(f32(gx * gx) + f32(gy * gy))))
            out[y, x] = f32(m * f32(0.05)) if m <= f32(20.0) else f32(1.0)
    return out


def q(v):
    """trunc((x as f64) + 0.5)"""
    return int(np.trunc(f64(v) + 0.5))


def overlay(r, g, fmt):
    h, w = r.shape
    out = np.zeros((h, w, 4), dtype=DT[fmt])
    for y in range(h):
        for x in range(w):
            if fmt == 2:
                out[y, x] = (f32(1.0), r[y, x], g[y, x], f32(0.0))
            else:
                k = f32(255.0) if fmt == 0 else f32(65535.0)
                mask = 0xFF if fmt == 0 else 0xFFFF
                out[y, x] = (WHITE[fmt], q(f32(r[y, x] * k)) & mask, q(f32(g[y, x] * k)) & mask, 0)
    return out


def blend_half(a, b, fmt):
    h, w, _ = a.shape
    out = np.zeros((h, w, 4), dtype=DT[fmt])
    wt = f32(0.5)
    for y in range(h):
        for x in range(w):
            for c in range(4):
                af, bf = f32(a[y, x, c]), f32(b[y, x, c])
                v = f32(f32(f32(bf - af) * wt) + af)
                if fmt == 2:
                    out[y, x, c] = f32(v + f32(0.0))
                else:
                    out[y, x, c] = int(np.trunc(f32(v + f32(0.5)))) & (0xFF if fmt == 0 else 0xFFFF)
    return out


def diff(a, b, fmt):
    h, w, _ = a.shape
    out = np.zeros((h, w, 4), dtype=DT[fmt])
    for y in range(h):
        for x in range(w):
            out[y, x, 0] = WHITE[fmt]
            for c in (1, 2, 3):
                if fmt == 2:
                    out[y, x, c] = abs(f32(a[y, x, c] - b[y, x, c]))
                else:
                    out[y, x, c] = abs(int(a[y, x, c]) - int(b[y, x, c]))
    return out


def view(x, y, flow, code, fmt):
    warped, _ = F.forward_warp(x, flow, f32(1.0), fmt)
    if code == 4:
        return warped
    if code == 0:
        return blend_half(warped, y, fmt)
    if code == 1:
        return diff(warped, y, fmt)
    if code == 2:
        return overlay(detail(luma(warped, fmt)), detail(luma(y, fmt)), fmt)
    return overlay(edge(luma(warped, fmt)), edge(luma(y, fmt)), fmt)


def main(path):
    rng = np.random.default_rng(20261028)
    cases = []
    spec = [
        # fmt, W, H, FW, FH, flow kind
        (0, 12, 9, 12, 9, "shift_frac"),
        (1, 12, 9, 12, 9, "swirl"),
        (2, 12, 9, 12, 9, "random"),
        (0, 14, 11, 7, 6, "swirl"),
        (1, 13, 10, 7, 5, "shift_int"),
        (2, 13, 10, 7, 5, "stretch"),
        (0, 9, 8, 9, 8, "tiny"),
    ]
    for fmt, w, h, fw, fh, kind in spec:
        x = F.make_image(rng, w, h, fmt)
        yimg = F.make_image(rng, w, h, fmt)
        flow = F.flow_field(kind, fw, fh, rng)
        for code in (0, 1, 2, 3, 4):
            want = view(x, yimg, flow, code, fmt)
            payload = struct.pack("<IIIIIi", fmt, w, h, fw, fh, code)
            payload += x.tobytes() + yimg.tobytes() + flow.astype("<f4").tobytes() + want.tobytes()
            cases.append(payload)
    with io.BytesIO() as f:
        f.write(b"DBG1")
        f.write(struct.pack("<I", len(cases)))
        for c in cases:
            f.write(c)
        blob = f.getvalue()
    with open(path, "w", encoding="ascii", newline="\n") as out:
        out.write(F.hex_text(blob))
    print(f"wrote {len(cases)} cases to {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "debug_cases.hex")
