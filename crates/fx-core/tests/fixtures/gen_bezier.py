"""Keyframe curves with Bezier ease handles, evaluated by tools/speed_curve.py, for the Rust test
`curves::tests::matches_python_fixture` and the Python test `Handles.test_the_fixture_values` in
tools/test_speed_curve.py. speed_curve.py writes the Bezier in the operation order of
crates/fx-core/src/curves.rs, so both reading the same values proves the two bit equal.

Run:  python -I gen_bezier.py        (from any directory; stdlib only)

File format, little endian, written as hex text of 64 bytes a line (bezier_cases.hex):
  b"BEZ1", u32 ncases, then per case
    u32 ticks a frame, u32 ticks a second, u32 length, the key text (ASCII),
    f64 limit (the slider is -limit..=limit), u32 count, then count times (i64 tick, f64 value)
"""
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "tools")))
import speed_curve  # noqa: E402

LIMIT = speed_curve.SPEED_LIMIT

# (ticks a frame, ticks a second): 24 fps, 60 fps, 30000/1001, 29.97, --time-scale 24 and 48000 at 24 fps
BASES = [(100, 2400), (100, 6000), (1001, 30000), (100, 2997), (1, 24), (2000, 48000)]

WRITTEN = [
    # Easy Ease as handles, plain and eased ends next to handles, the four shapes by hand
    (0, LIMIT, "100@0o0/33.3;300@10i0/33.3"),
    (0, LIMIT, "100@0o480/50;300@10i-120/75"),
    (0, LIMIT, "0@0e;250@12i30/10;40@30o-500/90;40@48"),
    (0, LIMIT, "100@0;40@24i0/100;100@48o0/100;100@72"),
    (1, LIMIT, "100@0o2000/100;100@30"),
    (2, LIMIT, "-100@0o-50/20.5;100@17.5i50/20.5"),
    # overshoot held at the top of the range, and at a range of 150
    (0, LIMIT, "9000@0o90000/60;10000@6;9500@9i-40000/40"),
    (0, 150.0, "100@0o5000/50;150@10;0@20i-9000/80;-150@24o-20000/100;0@30"),
    # hold keys with incoming handles, influences at both ends of their range
    (0, LIMIT, "100@0i5/5o7/7;200@3.3i0.1/0.1;150@7.7o-0.5/99.9;120@9h;80@11i3/30h;60@13"),
    (2, LIMIT, "100@0;100@120e;150@168i0/33.3o0/33.3;40@204i0/33.3o0/33.3;40@400e;100@454.67i10/50;100@664.67"),
    # no handles at all: the closed forms of before
    (0, LIMIT, "100@0;50@24h;75@36e;20@48"),
    (4, LIMIT, "100@0e;300@10e;-40@13;0@15e"),
    # fractional frames on 1001 ticks a frame, numbers written in every allowed way
    (2, LIMIT, "100@0.5o60/25;-40@2.25i-3.5/66.6;12@4.75"),
    (3, LIMIT, "1@0o-0/0.1;2@1i007.50/100;3@2o-0.0/50"),
    # a long segment, segments one tick long, and keys given in any order
    (0, LIMIT, "100@0o3/40;200@500i-3/60"),
    (0, LIMIT, "0@0o100/50;10@0.01;20@0.02i-5/50;30@0.03o9/9;0@0.05"),
    (5, LIMIT, "40@6i-1000/70;10@0o250/12.5;70@3"),
    # long segments that start at 0, read a few ticks in: the value keeps every bit of the curve
    # parameter there, so these tell 64 bisection steps from 63
    (0, LIMIT, "0@0o100/50;100@200"),
    (0, LIMIT, "0@0o2400/90;1000@50i-100/5"),
    (2, LIMIT, "0@0o-7.5/100;-300@120i0/100"),
    (5, LIMIT, "0@0o12/75;50@20;0@40o-30/60;-20@80"),
    (0, LIMIT, "0@0e;100@100i50/50"),
]


class Lcg:
    """a 64 bit linear congruential generator, so the file is the same on every Python."""

    def __init__(self, seed):
        self.x = seed

    def below(self, n):
        self.x = (6364136223846793005 * self.x + 1442695040888963407) % (1 << 64)
        return (self.x >> 33) % n


def random_case(rng):
    """keys with random values, frames, handles and suffixes on a random time base."""
    base = rng.below(len(BASES))
    step, _ = BASES[base]
    limit = (LIMIT, 500.0, 2.5)[rng.below(3)]
    keys, ticks, frame = [], set(), 0.0
    for _ in range(2 + rng.below(5)):
        frame += (1 + rng.below(400)) / 20
        text = f"{frame:.2f}".rstrip("0").rstrip(".")
        tick = speed_curve.round_half_away(float(text) * step)
        if tick in ticks:
            continue
        ticks.add(tick)
        scale = int(limit * 10)
        value = f"{(rng.below(2 * scale + 1) - scale) / 10:.1f}"
        suffix = ("", "", "h", "e")[rng.below(4)]
        handles = ""
        if suffix != "e":
            for side in ("i", "o"):
                if rng.below(2) and not (side == "o" and suffix == "h"):
                    speed = (rng.below(400001) - 200000) / 100 * (limit / LIMIT if rng.below(2) else 1)
                    influence = (1 + rng.below(1000)) / 10
                    handles += f"{side}{speed:.2f}/{influence:.1f}"
        keys.append(f"{value}@{text}{handles}{suffix}")
    return base, limit, ";".join(keys)


def sample_ticks(keys, step):
    """every tick of a short curve; for a long one each key's neighbourhood and the first ticks after
    each key, a stride through the rest and the half frame times the effect reads."""
    first, last = keys[0].tick, keys[-1].tick
    span = last - first
    picked = set(range(first - 2, last + 3, max(1, span // 90)))
    for k in keys:
        picked.update(range(k.tick - 2, k.tick + 3))
        picked.update(range(k.tick + 3, k.tick + 40, 5))
    picked.update(int(step * h / 2.0 + speed_curve.SAMPLE_BIAS) for h in range(0, 24))
    return sorted(picked)


def cases():
    rng = Lcg(20261027)
    out = list(WRITTEN)
    while len(out) < 65:
        out.append(random_case(rng))
    return out


def build():
    data = bytearray(b"BEZ1")
    every = cases()
    data += struct.pack("<I", len(every))
    for base, limit, text in every:
        step, per_second = BASES[base]
        keys = speed_curve.parse_keys(text, step, per_second, limit)
        curve = speed_curve.Curve(keys, limit)
        ticks = sample_ticks(keys, step)
        raw = text.encode("ascii")
        data += struct.pack("<III", step, per_second, len(raw)) + raw + struct.pack("<d", limit)
        data += struct.pack("<I", len(ticks))
        for t in ticks:
            data += struct.pack("<qd", t, curve.percent_at(t))
    return bytes(data)


def hex_text(data):
    return "".join(data[i:i + 64].hex() + "\n" for i in range(0, len(data), 64))


if __name__ == "__main__":
    path = os.path.join(HERE, "bezier_cases.hex")
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(hex_text(build()))
    print(f"wrote {path}")
