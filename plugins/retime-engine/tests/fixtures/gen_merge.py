"""Independent numpy float32 model of the matte weighted merge of create-vectors (rule G1d): the
merged vector image `vectors::encode_vectors_merged` and the merged flow `vectors::merge_flows`
of the Debug views. It produces `merge_cases.hex` for the Rust test
`vectors::merge_tests::model_fixtures_are_reproduced_bit_for_bit`.

Run:  python -I -B gen_merge.py [OUT]        (from any directory; needs numpy)

The flow sampler, the clamp and the stores are the models of `gen_vectors.py` and the OVER weights
the model of `gen_matte.py`, imported from this directory. No Rust code of the engine is used.
Every arithmetic step is on np.float32 in the order of the Rust code (numpy does not fuse): per
output pixel and per layer in order, `acc = acc + w * (u * kx)` from 0.0.

File format, little endian:
  b"MRG1", u32 ncases, then per case
    u32 depth (0 u8, 1 u16, 2 f32), u32 W, u32 H, f32 scale, u32 nlayers, then per layer
      u32 FW, u32 FH, flow (FW*FH*2 f32), weight (W*H f32);
    expected image (W*H*4 samples of the depth, A R G B), expected merged flow (W*H*2 f32)
"""
import io
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_vectors as V  # noqa: E402
import gen_matte as M  # noqa: E402

f32 = np.float32


def consts(W, H, FW, FH, scale):
    dw1, dh1 = f32(W - 1), f32(H - 1)
    fw1, fh1 = f32(FW - 1), f32(FH - 1)
    sx, sy = fw1 / dw1, fh1 / dh1
    if scale is None:
        return sx, sy, dw1 / fw1, dh1 / fh1
    return sx, sy, (dw1 / fw1) / f32(scale), (dh1 / fh1) / f32(scale)


def merged(layers, W, H, scale):
    """(H, W, 2) float32: the per pixel weighted sums of the layers."""
    out = np.zeros((H, W, 2), dtype=f32)
    cs = [consts(W, H, fl.shape[1], fl.shape[0], scale) for fl, _ in layers]
    for y in range(H):
        for x in range(W):
            ax, ay = f32(0.0), f32(0.0)
            for (fl, wt), (sx, sy, kx, ky) in zip(layers, cs):
                d = V.bilerp2(fl, f32(x) * sx, f32(y) * sy)
                wk = wt[y, x]
                ax = f32(ax + f32(wk * f32(d[0] * kx)))
                ay = f32(ay + f32(wk * f32(d[1] * ky)))
            out[y, x] = (ax, ay)
    return out


def store(acc, depth):
    H, W = acc.shape[0], acc.shape[1]
    dtype = [np.uint8, np.uint16, np.float32][depth]
    out = np.zeros((H, W, 4), dtype=dtype)
    for y in range(H):
        for x in range(W):
            a, b = acc[y, x, 0], acc[y, x, 1]
            if depth == 2:
                out[y, x] = (f32(1.0), (a + f32(1.0)) * f32(0.5), (b + f32(1.0)) * f32(0.5), f32(0.0))
            else:
                a, b = V.clamp1(a) + f32(1.0), V.clamp1(b) + f32(1.0)
                mul = 127.0 if depth == 0 else 32767.0
                white = 255 if depth == 0 else 65535
                mask = 0xFF if depth == 0 else 0xFFFF
                out[y, x] = (white, int(float(a) * mul + 0.5) & mask, int(float(b) * mul + 0.5) & mask, 0)
    return out


def soft_matte(rng, W, H):
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy, r = rng.uniform(0.2, 0.8) * W, rng.uniform(0.2, 0.8) * H, rng.uniform(0.25, 0.45) * min(W, H)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip((r - d) / 2.5 + 0.5, 0.0, 1.0).astype(f32)


def main(path):
    rng = np.random.default_rng(20261027)
    cases = []
    spec = [
        # depth, W, H, scale, flow sizes, weights ("over" from soft mattes with a group, or "random")
        (0, 13, 9, 5.0, [(13, 9)], "one"),
        (1, 16, 12, 4.0, [(8, 6), (8, 6)], ("over", 1, 0, 1)),
        (2, 16, 12, 4.0, [(8, 6), (8, 6)], ("over", 1, 0, 1)),
        (0, 17, 11, 3.0, [(8, 5), (8, 5), (8, 5)], ("over", 2, 0, 2)),
        (1, 17, 11, 3.0, [(8, 5), (8, 5)], ("over", 2, 1, 2)),
        (2, 12, 10, 2.5, [(12, 10), (6, 5), (12, 10), (6, 5)], ("over", 3, 0, 3)),
        (0, 10, 11, 1.5, [(21, 23), (5, 6)], "random"),
        (1, 14, 9, 100.0, [(7, 4), (7, 4), (7, 4)], "random"),
        (2, 9, 9, 0.75, [(9, 9)], "random"),
        (0, 12, 8, 2.0, [(6, 4)], ("over", 1, 0, 0)),
        (2, 12, 8, 2.0, [(6, 4)], ("over", 2, 2, 2)),
    ]
    for depth, W, H, scale, sizes, wkind in spec:
        flows = [V.random_flow(rng, fw, fh, rng.uniform(0.5, 8.0)) for fw, fh in sizes]
        if wkind == "one":
            weights = [np.ones((H, W), dtype=f32)]
        elif wkind == "random":
            weights = [rng.uniform(0.0, 1.0, (H, W)).astype(f32) for _ in sizes]
        else:
            _, nfg, first, last = wkind
            ms = [soft_matte(rng, W, H) for _ in range(nfg)]
            weights = [w.reshape(H, W) for w in M.over_weights(ms, first, last, (H, W))]
        assert len(weights) == len(flows)
        layers = list(zip(flows, weights))
        img = store(merged(layers, W, H, scale), depth)
        flow = merged(layers, W, H, None)
        payload = struct.pack("<IIIfI", depth, W, H, float(f32(scale)), len(layers))
        for fl, wt in layers:
            payload += struct.pack("<II", fl.shape[1], fl.shape[0]) + fl.astype("<f4").tobytes() + wt.astype("<f4").tobytes()
        payload += img.tobytes() + flow.astype("<f4").tobytes()
        cases.append(payload)
    with io.BytesIO() as f:
        f.write(b"MRG1")
        f.write(struct.pack("<I", len(cases)))
        for c in cases:
            f.write(c)
        blob = f.getvalue()
    with open(path, "w", encoding="ascii", newline="\n") as out:
        out.write(V.hex_text(blob))
    print(f"wrote {len(cases)} cases to {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "merge_cases.hex")
