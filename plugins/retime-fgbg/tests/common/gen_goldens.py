"""Independent numpy float32 model of premultiply, blend and unpremultiply for the golden tests of
plugin 03 (the pixel path is the same as in plugin 01). Writes the Rust literals to the file named
on the command line."""
import math
import numpy as np

f32 = np.float32
f64 = np.float64

def q(x):
    # q(x) = trunc((x as f64) + 0.5) as i64 (x is an f32 value widened exactly)
    return int(math.trunc(float(f64(x)) + 0.5))

K255 = f32(0.003921568859368563)
assert K255.view(np.uint32) == 0x3b808081
K16A = f32(6.1034224927425385e-05)
assert K16A.view(np.uint32) == 0x387fff00
K16B = f32(1.999969482421875)
assert K16B.view(np.uint32) == 0x3fffff00
K16C = f32(0.5000076293945312)
assert K16C.view(np.uint32) == 0x3f000080

# ---- 8 bpc: wire [r,g,b,a] <-> engine [A,R,G,B]
def pre8(px):
    r, g, b, a = px
    af = f32(a) * K255
    return [a, q(f32(r) * af) & 0xFF, q(f32(g) * af) & 0xFF, q(f32(b) * af) & 0xFF]

def unpre8(e):
    a, r, g, b = e
    if a == 0:
        return [r, g, b, a]
    s = f32(255.0) / f32(a)
    out = []
    for c in (r, g, b):
        v = f32(c) * s
        if f32(255.0) < v:
            v = f32(255.0)
        out.append(q(v) & 0xFF)
    return [out[0], out[1], out[2], a]

def blend8(ea, eb, w):
    w = f32(w)
    out = []
    for x, y in zip(ea, eb):
        t = (f32(y) - f32(x)) * w
        t = t + f32(x)
        t = t + f32(0.5)
        out.append(int(math.trunc(float(f64(t)))) & 0xFF)
    return out

# ---- 16 bpc
def u16_to_q15(v):
    return (v * 32768 + 32767) // 65535

def q15_to_u16(v):
    return (min(v, 32768) * 65535 + 16384) // 32768

def pre16(px):
    r, g, b, a = [u16_to_q15(v) for v in px]
    af = f32(a) * K16A
    return [q(f32(a) * K16B) & 0xFFFF, q(f32(r) * af) & 0xFFFF, q(f32(g) * af) & 0xFFFF, q(f32(b) * af) & 0xFFFF]

def unpre16(e):
    a, r, g, b = e
    if a == 0:
        return [q15_to_u16(r), q15_to_u16(g), q15_to_u16(b), q15_to_u16(a)]
    s = f32(32768.0) / f32(a)
    na = q(f32(a) * K16C) & 0xFFFF
    out = [q(f32(c) * s) & 0xFFFF for c in (r, g, b)]
    return [q15_to_u16(out[0]), q15_to_u16(out[1]), q15_to_u16(out[2]), q15_to_u16(na)]

def blend16(ea, eb, w):
    w = f32(w)
    out = []
    for x, y in zip(ea, eb):
        t = (f32(y) - f32(x)) * w
        t = t + f32(x)
        t = t + f32(0.5)
        out.append(int(math.trunc(float(f64(t)))) & 0xFFFF)
    return out

# ---- 32 bpc
def pre32(px):
    r, g, b, a = [f32(v) for v in px]
    return [a, a * r, a * g, a * b]

def unpre32(e):
    a, r, g, b = e
    if float(f64(a)) >= 1e-12:
        inv = f32(1.0) / a
        return [inv * r, inv * g, inv * b, a]
    return [f32(0), f32(0), f32(0), f32(0)]

def blend32(ea, eb, w):
    w = f32(w)
    return [((y - x) * w + x) + f32(0.0) for x, y in zip(ea, eb)]

def fmt32(v):
    return np.format_float_positional(f32(v), unique=True, trim='0') + 'f32'

def rust_px(px, depth):
    if depth == 32:
        return '[' + ', '.join(fmt32(v) for v in px) + ']'
    return '[' + ', '.join(str(int(v)) for v in px) + ']'

CLIPS = {
    8: [
        [(255,0,0,255), (0,255,0,255), (0,0,255,128), (200,100,50,64), (90,180,255,0), (17,34,51,255)],
        [(0,255,255,255), (255,0,255,200), (10,20,30,1), (255,255,255,254), (1,2,3,128), (100,100,100,100)],
        [(128,64,32,255), (33,66,99,33), (250,5,130,177), (0,0,0,0), (64,128,192,255), (255,0,0,3)],
        [(12,200,77,255), (0,0,0,255), (255,255,255,255), (170,85,10,90), (5,5,5,5), (240,120,60,128)],
    ],
    16: [
        [(65535,0,0,65535), (0,65535,0,40000), (12345,23456,34567,1), (50000,40000,30000,0)],
        [(1000,2000,3000,65535), (65535,65535,65535,32768), (0,0,65535,65534), (777,888,999,12345)],
        [(32768,16384,8192,65535), (40000,50000,60000,50000), (1,2,3,4), (65535,0,65535,500)],
        [(0,0,0,65535), (21845,43690,65535,60000), (65535,65535,65535,65535), (9,99,999,9999)],
    ],
    32: [
        [(1.0,0.5,0.25,1.0), (0.2,0.4,0.6,0.5), (0.9,0.8,0.7,0.0), (0.3,0.6,0.9,1e-13)],
        [(0.0,0.0,1.0,1.0), (1.0,1.0,1.0,0.25), (-0.25,0.5,1.5,0.75), (0.125,0.375,0.625,0.875)],
        [(0.5,0.5,0.5,0.5), (0.0,1.0,0.0,1.0), (2.0,3.0,4.0,1.0), (0.1,0.2,0.3,0.4)],
        [(0.75,0.25,0.125,1.0), (0.6,0.7,0.8,0.9), (1.0,1.0,1.0,1.0), (0.01,0.02,0.03,0.04)],
    ],
}
FUN = {
    8: (pre8, unpre8, blend8),
    16: (pre16, unpre16, blend16),
    32: (pre32, unpre32, blend32),
}

def roundtrip(depth, frame):
    pre, unpre, _ = FUN[depth]
    return [unpre(pre(px)) for px in frame]

def blendframes(depth, fa, fb, w):
    pre, unpre, bl = FUN[depth]
    out = []
    for pa, pb in zip(fa, fb):
        out.append(unpre(bl(pre(pa), pre(pb), w)))
    return out

def transparent(depth, n):
    return [(0,0,0,0) if depth != 32 else (0.0,0.0,0.0,0.0)] * n

def emit(name, frame, depth):
    ty = 'u8' if depth == 8 else 'u16' if depth == 16 else 'f32'
    body = ', '.join(rust_px(px, depth) for px in frame)
    return f"    pub const {name}: [[{ty}; 4]; {len(frame)}] = [{body}];\n"


HEADER = """//! Golden frames of the retime scenarios, produced by `gen_goldens.py` (next to this file): an
//! independent numpy float32 model of premultiply, blend and unpremultiply. Wire pixels are `[r, g,
//! b, a]`. `F*` is the input clip, `R*` the engine's premultiply / unpremultiply round trip of
//! `F*`, `B01` the blend of the premultiplied frames 0 and 1 with weight 0.5 (then
//! unpremultiplied), `B3T` the blend of frame 3 with a transparent frame, `B01_25` and `B01_75` the
//! weights 0.25 and 0.75.
//! Do not edit by hand: rerun `python gen_goldens.py goldens.rs`.

"""


def main(path):
    out = [HEADER]
    for depth in (8, 16, 32):
        out.append(f"pub mod d{depth} {{\n")
        frames = CLIPS[depth]
        n = len(frames[0])
        for i, fr in enumerate(frames):
            out.append(emit(f"F{i}", fr, depth))
        T = transparent(depth, n)
        for i, fr in enumerate(frames):
            out.append(emit(f"R{i}", roundtrip(depth, fr), depth))
        for i in range(3):
            out.append(emit(f"B{i}{i+1}", blendframes(depth, frames[i], frames[i+1], 0.5), depth))
        out.append(emit("B3T", blendframes(depth, frames[3], T, 0.5), depth))
        out.append(emit("B01_25", blendframes(depth, frames[0], frames[1], 0.25), depth))
        out.append(emit("B01_75", blendframes(depth, frames[0], frames[1], 0.75), depth))
        out.append("}\n\n")
    text = ''.join(out).rstrip('\n') + '\n'
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
