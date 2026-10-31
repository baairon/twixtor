#!/usr/bin/env python3
"""Pixel format conversion of the emulator harness against the rewrite's wire formats.

The harness turns a case's wire frames (8 bpc `rgba`, 16 bpc `rgba64le`, 32 bpc planar `gbrapf32le`) into
After Effects worlds (ARGB; 16 bpc white 32768; float white 1.0), and the plugin's output world back into the
same wire format. Any asymmetry between that and the rewrite's own reading and writing (crates/fx-core/src/
frame.rs) would show up in `fx-parity run --max-ulp 0` as a difference that is not the rewrite's fault.

Standard library only. Nothing here boots the emulator: the conversion functions of ae_mock are imported and
called directly, and the world metadata tests run the mock host's real methods on a byte array that stands in
for guest memory.

    python3 -m unittest test_world_conversion -v

Classes whose name starts with `Gap` assert behaviour the harness does not have yet; they fail on purpose and
each failure message says what is missing.

Optional, adds a comparison with the compiled Rust (`Frame::from_bytes`, `to_bytes`, `u16_to_q15`,
`q15_to_u16`): build a small program that dumps those conversions and run
    FX_CONV_DUMP=/path/to/convdump python3 -m unittest test_world_conversion -v
"""
import array
import os
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ae_host  # noqa: E402
import ae_mock  # noqa: E402

SIZES = [(1, 1), (2, 3), (3, 1), (1, 5), (4, 4), (5, 7), (7, 3), (13, 2), (37, 23), (255, 129)]
BPP = {8: 4, 16: 8, 32: 16}
# wire channel R, G, B, A -> index inside an A, R, G, B world pixel
WORLD_OF_WIRE = (1, 2, 3, 0)


def note(text):
    sys.stderr.write("INFO " + text + "\n")


# ---------------------------------------------------------------------------------------------
# The rewrite, ported by hand from crates/fx-core/src/frame.rs (independent of the harness code)

def rs_u16_to_q15(v):
    t = v * 32768 + 32767                        # `v as u32 * 32768 + 32767`, at most 2^31 - 1
    assert t < 1 << 32
    return (t // 65535) & 0xFFFF                 # u32 division, then `as u16`


def rs_q15_to_u16(v):
    t = min(v, 32768) * 65535 + 16384            # `v.min(32768) as u32 * 65535 + 16384`
    assert t < 1 << 32
    return (t // 32768) & 0xFFFF


def rs_read_wire(depth, w, h, wire):
    """Frame::from_bytes: a list of (R, G, B, A) per pixel, raw words (u8, u16, f32 bit patterns)."""
    n = w * h
    assert len(wire) == n * BPP[depth]
    if depth == 8:
        return [tuple(wire[4 * i:4 * i + 4]) for i in range(n)]
    if depth == 16:
        words = struct.unpack("<%dH" % (4 * n), wire)
        return [tuple(words[4 * i:4 * i + 4]) for i in range(n)]
    words = struct.unpack("<%dI" % (4 * n), wire)           # integer words: no float is ever made
    plane = lambda p, i: words[p * n + i]
    return [(plane(2, i), plane(0, i), plane(1, i), plane(3, i)) for i in range(n)]      # R, G, B, A


def rs_write_wire(depth, w, h, px):
    """Frame::to_bytes."""
    n = w * h
    if depth == 8:
        return bytes(c for p in px for c in p)
    if depth == 16:
        return struct.pack("<%dH" % (4 * n), *[c for p in px for c in p])
    out = []
    for ch in (1, 2, 0, 3):                                  # planes G, B, R, A
        out.extend(p[ch] for p in px)
    return struct.pack("<%dI" % (4 * n), *out)


def world_pixels(depth, w, h, world):
    """The world bytes as a list of (A, R, G, B) raw words."""
    n = w * h
    assert len(world) == n * BPP[depth]
    if depth == 8:
        return [tuple(world[4 * i:4 * i + 4]) for i in range(n)]
    code = "H" if depth == 16 else "I"
    words = struct.unpack("<%d%s" % (4 * n, code), world)
    return [tuple(words[4 * i:4 * i + 4]) for i in range(n)]


def world_bytes(depth, w, h, pixels):
    code = {8: "B", 16: "H", 32: "I"}[depth]
    return struct.pack("<%d%s" % (4 * w * h, code), *[c for p in pixels for c in p])


def rng(tag):
    return random.Random("world-conversion-" + tag)


def random_wire(depth, w, h, seed):
    r = rng(seed)
    if depth == 8:
        return bytes(r.randrange(256) for _ in range(w * h * 4))
    if depth == 16:
        return struct.pack("<%dH" % (w * h * 4), *[r.randrange(65536) for _ in range(w * h * 4)])
    return struct.pack("<%dI" % (w * h * 4), *[r.getrandbits(32) for _ in range(w * h * 4)])


def first(items, n=8):
    return items[:n]


class TC(unittest.TestCase):
    """assertEqual that stays fast and short when two long sequences or byte strings differ (no difflib)."""

    def assertEqual(self, first, second, msg=None):
        if first == second:
            return
        seq = (bytes, bytearray, list, tuple)
        if isinstance(first, seq) and isinstance(second, seq) and (len(first) > 16 or len(second) > 16):
            head = "%s: " % msg if msg else ""
            if len(first) != len(second):
                self.fail("%slength %d != %d" % (head, len(first), len(second)))
            bad = [i for i in range(len(first)) if first[i] != second[i]]
            self.fail("%s%d of %d elements differ; first (index, got, want): %s" % (head, len(bad), len(first), [(i, first[i], second[i]) for i in bad[:6]]))
        super().assertEqual(first, second, msg)


# ---------------------------------------------------------------------------------------------
# preconditions

class Preconditions(TC):
    def test_native_layout_is_what_the_conversions_assume(self):
        # array("H") and array("f") are read and written in native order and size by ae_mock
        self.assertEqual(sys.byteorder, "little", "ae_mock reads the little endian wire with native arrays")
        self.assertEqual(array.array("H").itemsize, 2)
        self.assertEqual(array.array("f").itemsize, 4)
        self.assertEqual(struct.calcsize("f"), 4)

    def test_bytes_per_pixel_matches_the_sdk_pixel_structs_and_the_wire(self):
        self.assertEqual(ae_mock.BPP, BPP)
        lay = ae_host.Layouts()
        for name, size in (("PF_Pixel8", 4), ("PF_Pixel16", 8), ("PF_PixelFloat", 16)):
            if name in lay.L:
                self.assertEqual(lay.size(name), size, name)

    def test_pixel_struct_member_order_is_argb(self):
        # AE_Effect.h: PF_Pixel { alpha, red, green, blue }, PF_Pixel16 and PF_PixelFloat the same order
        header = HERE.parent.parent / "target" / "ae-sdk" / "AfterEffectsSDK" / "Examples" / "Headers" / "AE_Effect.h"
        if not header.exists():
            self.skipTest("SDK headers not present")
        text = header.read_text(errors="replace")
        self.assertIn("A_u_char alpha, red, green, blue;", text)
        self.assertIn("A_u_short alpha, red, green, blue;", text)
        self.assertIn("PF_FpShort alpha, red, green, blue;", text)
        self.assertIn("#define PF_MAX_CHAN16 32768", text)

    def test_wire_names_agree_with_worker_and_fx_parity(self):
        import worker
        self.assertEqual(worker.PIX_FMT, {8: "rgba", 16: "rgba64le", 32: "gbrapf32le"})
        src = HERE.parent.parent / "crates" / "fx-parity" / "src" / "sources.rs"
        if src.exists():
            text = src.read_text()
            for depth, fmt in ((8, "rgba"), (16, "rgba64le")):
                self.assertIn("%d => \"%s\"" % (depth, fmt), text)
            self.assertIn("_ => \"gbrapf32le\"", text)


# ---------------------------------------------------------------------------------------------
# 8 bpc

class Depth8(TC):
    def test_every_value_in_every_channel_round_trips_and_lands_in_argb_order(self):
        for c in range(4):
            wire = bytearray()
            for v in range(256):
                px = [(v * 3 + 17 * (k + 1)) & 255 for k in range(4)]     # distinct neighbours
                px[c] = v
                wire += bytes(px)
            wire = bytes(wire)
            world = ae_mock.wire_to_world(8, 256, 1, wire)
            self.assertEqual(len(world), 1024)
            for v in range(256):
                self.assertEqual(world[4 * v + WORLD_OF_WIRE[c]], v, "wire channel %d value %d" % (c, v))
            self.assertEqual(ae_mock.world_to_wire(8, 256, 1, world), wire, "channel %d" % c)

    def test_channel_order(self):
        wire = bytes([10, 20, 30, 40])                         # R G B A
        self.assertEqual(ae_mock.wire_to_world(8, 1, 1, wire), bytes([40, 10, 20, 30]))   # A R G B
        self.assertEqual(ae_mock.world_to_wire(8, 1, 1, bytes([40, 10, 20, 30])), wire)

    def test_sizes_round_trip_both_ways_and_match_the_rewrite_reader(self):
        for w, h in SIZES:
            wire = random_wire(8, w, h, "d8-%dx%d" % (w, h))
            world = ae_mock.wire_to_world(8, w, h, wire)
            self.assertEqual(len(world), w * h * 4)
            self.assertEqual(ae_mock.world_to_wire(8, w, h, world), wire, (w, h))
            self.assertEqual(ae_mock.wire_to_world(8, w, h, ae_mock.world_to_wire(8, w, h, world)), world, (w, h))
            # the world, pixel by pixel, from the rewrite's reading of the wire
            want = [(a, r, g, b) for (r, g, b, a) in rs_read_wire(8, w, h, wire)]
            self.assertEqual(world_pixels(8, w, h, world), want, (w, h))
            # and back through the rewrite's writer
            px = [(r, g, b, a) for (a, r, g, b) in world_pixels(8, w, h, world)]
            self.assertEqual(rs_write_wire(8, w, h, px), ae_mock.world_to_wire(8, w, h, world), (w, h))

    def test_pixels_are_row_major_with_no_padding(self):
        # a width whose row (5 * 4 = 20 bytes) is not a multiple of 16, 32 or 64: pixel (x, y) at (y * w + x) * 4
        w, h = 5, 3
        wire = bytearray()
        for y in range(h):
            for x in range(w):
                wire += bytes([x, y, 100 + x + y, 255])
        world = ae_mock.wire_to_world(8, w, h, bytes(wire))
        for y in range(h):
            for x in range(w):
                o = (y * w + x) * 4
                self.assertEqual(tuple(world[o:o + 4]), (255, x, y, 100 + x + y))


# ---------------------------------------------------------------------------------------------
# 16 bpc

def frame16(c, values):
    """A wire frame of len(values) pixels whose channel `c` runs through `values`, the others differ."""
    n = len(values)
    arr = array.array("H", bytes(8 * n))
    for k in range(4):
        if k == c:
            arr[k::4] = array.array("H", values)
        else:
            arr[k::4] = array.array("H", [(v + 0x1111 * (k + 1)) & 0xFFFF for v in values])
    return arr.tobytes()


class Depth16(TC):
    ALL = list(range(65536))

    def test_lookup_tables_are_the_rewrites_two_functions(self):
        to_q, from_q = ae_mock.luts()
        self.assertEqual(len(to_q), 65536)
        self.assertEqual(len(from_q), 32769)
        bad_in = [v for v in range(65536) if to_q[v] != rs_u16_to_q15(v) or ae_mock.q15(v) != rs_u16_to_q15(v)]
        bad_out = [v for v in range(32769) if from_q[v] != rs_q15_to_u16(v) or ae_mock.u16_of_q15(v) != rs_q15_to_u16(v)]
        self.assertEqual(bad_in, [], "wire to world differs from Frame::u16_to_q15")
        self.assertEqual(bad_out, [], "world to wire differs from Frame::q15_to_u16")

    def test_wire_to_world_exhaustive_all_65536_values_in_every_channel(self):
        expect = [rs_u16_to_q15(v) for v in self.ALL]
        bad = []
        for c in range(4):
            wire = frame16(c, self.ALL)
            world = array.array("H")
            world.frombytes(ae_mock.wire_to_world(16, 256, 256, wire))
            for v in self.ALL:
                got = world[4 * v + WORLD_OF_WIRE[c]]
                if got != expect[v]:
                    bad.append((c, v, got, expect[v]))
            # the other three channels carry (v + 0x1111 * (k + 1)): the whole world is checked, not one lane
            for v in (0, 1, 777, 32767, 32768, 40000, 65534, 65535):
                for k in range(4):
                    if k != c:
                        src = (v + 0x1111 * (k + 1)) & 0xFFFF
                        if world[4 * v + WORLD_OF_WIRE[k]] != rs_u16_to_q15(src):
                            bad.append((c, v, k, "neighbour"))
        note("16 bpc wire to world: %d values x 4 channels compared with Frame::u16_to_q15, %d mismatches" % (65536, len(bad)))
        self.assertEqual(first(bad), [], "(channel, wire value, harness world value, rewrite q15)")
        self.assertEqual(len(bad), 0)

    def test_world_to_wire_exhaustive_all_32769_in_range_values_and_the_clamp(self):
        bad = []
        for c in range(4):
            vals = array.array("H", self.ALL)                  # 0..65535 as WORLD values: 0..32768 are legal, the rest clamps
            world = array.array("H", bytes(8 * 65536))
            for k in range(4):
                world[k::4] = vals if k == c else array.array("H", [(v + 0x0101 * (k + 1)) & 0xFFFF for v in self.ALL])
            out = array.array("H")
            out.frombytes(ae_mock.world_to_wire(16, 256, 256, world.tobytes()))
            wi = (3, 0, 1, 2)[c]                              # world A, R, G, B sits at wire index 3, 0, 1, 2
            for v in self.ALL:
                got, want = out[4 * v + wi], rs_q15_to_u16(v)
                if got != want:
                    bad.append((c, v, got, want))
        note("16 bpc world to wire: 32769 in-range values + 32767 clamped values x 4 channels compared with Frame::q15_to_u16, %d mismatches" % len(bad))
        self.assertEqual(first(bad), [], "(world channel, world value, harness wire value, rewrite wire value)")
        self.assertEqual(rs_q15_to_u16(32768), 65535)
        self.assertEqual(rs_q15_to_u16(40000), 65535)
        self.assertEqual(rs_q15_to_u16(65535), 65535)

    def test_world_values_above_32768_are_clamped_like_the_rewrite(self):
        world = array.array("H", [65535, 32769, 49152, 32768]).tobytes()           # A R G B
        out = array.array("H")
        out.frombytes(ae_mock.world_to_wire(16, 1, 1, world))
        self.assertEqual(list(out), [65535, 65535, 65535, 65535])                  # R G B A

    def test_round_trip_matches_the_rewrite_for_every_value_and_its_loss_is_stated(self):
        to_q, from_q = ae_mock.luts()
        harness = [from_q[to_q[v]] for v in self.ALL]
        rewrite = [rs_q15_to_u16(rs_u16_to_q15(v)) for v in self.ALL]
        self.assertEqual(harness, rewrite)
        lossy = [v for v in self.ALL if rewrite[v] != v]
        worst = max(abs(rewrite[v] - v) for v in self.ALL)
        note("16 bpc wire to world to wire: %d of 65536 wire values do not come back (in 65535 at most %d off)" % (len(lossy), worst))
        self.assertLessEqual(worst, 1)
        # the world side is lossless: 32769 levels survive world -> wire -> world
        for q in range(32769):
            self.assertEqual(rs_u16_to_q15(rs_q15_to_u16(q)), q)
            self.assertEqual(to_q[from_q[q]], q)
        # through the real functions on a frame
        wire = frame16(0, self.ALL)
        back = ae_mock.world_to_wire(16, 256, 256, ae_mock.wire_to_world(16, 256, 256, wire))
        out = array.array("H")
        out.frombytes(back)
        self.assertEqual([out[4 * v] for v in self.ALL], rewrite)

    def test_little_endian_bytes(self):
        wire = struct.pack("<4H", 0x1234, 0x0000, 0xFFFF, 0x8000)                 # R G B A
        world = array.array("H")
        world.frombytes(ae_mock.wire_to_world(16, 1, 1, wire))
        self.assertEqual(list(world), [rs_u16_to_q15(0x8000), rs_u16_to_q15(0x1234), 0, 32768])
        self.assertEqual(ae_mock.world_to_wire(16, 1, 1, world.tobytes()), struct.pack("<4H", *[rs_q15_to_u16(rs_u16_to_q15(v)) for v in (0x1234, 0, 0xFFFF, 0x8000)]))

    def test_sizes_match_the_rewrite_reader_and_writer(self):
        for w, h in SIZES:
            wire = random_wire(16, w, h, "d16-%dx%d" % (w, h))
            world = ae_mock.wire_to_world(16, w, h, wire)
            self.assertEqual(len(world), w * h * 8)
            want = [(rs_u16_to_q15(a), rs_u16_to_q15(r), rs_u16_to_q15(g), rs_u16_to_q15(b)) for (r, g, b, a) in rs_read_wire(16, w, h, wire)]
            self.assertEqual(world_pixels(16, w, h, world), want, (w, h))
            px = [(rs_q15_to_u16(r), rs_q15_to_u16(g), rs_q15_to_u16(b), rs_q15_to_u16(a)) for (a, r, g, b) in world_pixels(16, w, h, world)]
            self.assertEqual(ae_mock.world_to_wire(16, w, h, world), rs_write_wire(16, w, h, px), (w, h))

    def test_full_scale(self):
        self.assertEqual(rs_u16_to_q15(65535), 32768)
        self.assertEqual(rs_u16_to_q15(0), 0)
        world = array.array("H")
        world.frombytes(ae_mock.wire_to_world(16, 1, 1, struct.pack("<4H", 65535, 65535, 65535, 65535)))
        self.assertEqual(list(world), [32768] * 4)                                 # white is PF_MAX_CHAN16


# ---------------------------------------------------------------------------------------------
# 32 bpc

SPECIAL_WORDS = [
    0x00000000, 0x80000000,                          # +0, -0
    0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF,  # denormals, both ends, both signs
    0x00800000, 0x80800000,                          # FLT_MIN
    0x3F800000, 0xBF800000, 0x3F7FFFFF, 0x3F800001,  # 1, -1, just below and above 1
    0x40000000, 0x41200000, 0x42C80000,              # 2, 10, 100 (above 1)
    0xC1200000,                                      # -10 (negative)
    0x7F7FFFFF, 0xFF7FFFFF,                          # +-FLT_MAX
    0x7F800000, 0xFF800000,                          # +-inf
    0x7FC00000, 0xFFC00000,                          # default quiet NaN, negative quiet NaN
    0x7F800001, 0xFF800001,                          # signalling NaN, smallest payload
    0x7FBFFFFF, 0xFFBFFFFF,                          # signalling NaN, largest payload
    0x7FC00001, 0xFFC00001, 0x7FFFFFFF, 0xFFFFFFFF,  # quiet NaNs with payloads
    0x7FA5A5A5, 0xFFA5A5A5, 0x7FC12345, 0xFFDEADBE,
    0x3F000000, 0x3DCCCCCD,                          # 0.5, 0.1
]


def structured_words():
    out = []
    for hi in range(65536):                          # sign, exponent and the top 7 mantissa bits, all of them
        for lo in (0x0000, 0x0001, 0x5555, 0x8000, 0xFFFF):
            out.append((hi << 16) | lo)
    return out


def planar_words_to_wire(words, n):
    """`words` is 4 * n integer words: planes G, B, R, A."""
    assert len(words) == 4 * n
    return struct.pack("<%dI" % (4 * n), *words)


class Depth32(TC):
    def _convert(self, w, h, words):
        n = w * h
        wire = planar_words_to_wire(words, n)
        world = ae_mock.wire_to_world(32, w, h, wire)
        self.assertEqual(len(world), n * 16)
        got = struct.unpack("<%dI" % (4 * n), world)
        # explicit index formula: pixel i is A = plane A (3), R = plane R (2), G = plane G (0), B = plane B (1)
        want = []
        for i in range(n):
            want += [words[3 * n + i], words[2 * n + i], words[0 * n + i], words[1 * n + i]]
        return wire, world, list(got), want

    def test_special_values_round_trip_bit_for_bit_in_every_plane(self):
        count = len(SPECIAL_WORDS)
        n = count
        rot = list(SPECIAL_WORDS)
        for shift in range(4):
            # every special value once in every plane across the four rotations
            words = []
            for p in range(4):
                words += rot[(shift * 5 + p * 3) % count:] + rot[:(shift * 5 + p * 3) % count]
            wire, world, got, want = self._convert(n, 1, words)
            self.assertEqual(got, want, "wire to world, rotation %d" % shift)
            back = ae_mock.world_to_wire(32, n, 1, world)
            self.assertEqual(back, wire, "world back to wire, rotation %d" % shift)

    def test_each_special_value_is_checked_by_name_in_each_plane(self):
        for word in SPECIAL_WORDS:
            for p in range(4):                                      # plane G, B, R, A
                words = [0x3F800000 + k for k in range(8)]          # 2 pixels, planes G G B B R R A A
                words[2 * p] = word
                words[2 * p + 1] = word ^ 0x80000000                # the same value with the other sign
                wire = planar_words_to_wire(words, 2)
                world = ae_mock.wire_to_world(32, 2, 1, wire)
                g, b, r, a = (words[0:2], words[2:4], words[4:6], words[6:8])
                want = (a[0], r[0], g[0], b[0], a[1], r[1], g[1], b[1])
                self.assertEqual(struct.unpack("<8I", world), want, "%s in plane %d" % (hex(word), p))
                self.assertEqual(ae_mock.world_to_wire(32, 2, 1, world), wire, "%s in plane %d" % (hex(word), p))

    def test_structured_bit_patterns_all_sign_exponent_and_top_mantissa_values(self):
        words = structured_words()
        n = (len(words) + 3) // 4
        words += words[:4 * n - len(words)]
        wire, world, got, want = self._convert(n, 1, words)
        note("32 bpc: %d bit patterns (every sign/exponent/top-mantissa value x 5 low halves) compared as integers" % len(words))
        bad = [i for i in range(len(got)) if got[i] != want[i]]
        self.assertEqual(first(bad), [])
        self.assertEqual(ae_mock.world_to_wire(32, n, 1, world), wire)

    def test_random_words_including_every_nan(self):
        r = rng("f32-random")
        words = [r.getrandbits(32) for _ in range(4 * 100003)]
        n = 100003
        wire, world, got, want = self._convert(n, 1, words)
        self.assertEqual(got, want)
        self.assertEqual(ae_mock.world_to_wire(32, n, 1, world), wire)

    def test_world_to_wire_alone_on_arbitrary_words(self):
        r = rng("f32-world")
        n = 5003
        words = [r.getrandbits(32) for _ in range(4 * n)]
        world = struct.pack("<%dI" % (4 * n), *words)                # interleaved A R G B
        out = struct.unpack("<%dI" % (4 * n), ae_mock.world_to_wire(32, n, 1, world))
        for i in range(0, n, 97):
            a, r_, g, b = words[4 * i:4 * i + 4]
            self.assertEqual((out[0 * n + i], out[1 * n + i], out[2 * n + i], out[3 * n + i]), (g, b, r_, a))

    def test_sizes_match_the_rewrite_reader_and_writer(self):
        for w, h in SIZES:
            wire = random_wire(32, w, h, "d32-%dx%d" % (w, h))
            world = ae_mock.wire_to_world(32, w, h, wire)
            self.assertEqual(len(world), w * h * 16)
            want = [(a, r, g, b) for (r, g, b, a) in rs_read_wire(32, w, h, wire)]
            self.assertEqual(world_pixels(32, w, h, world), want, (w, h))
            px = [(r, g, b, a) for (a, r, g, b) in world_pixels(32, w, h, world)]
            self.assertEqual(rs_write_wire(32, w, h, px), ae_mock.world_to_wire(32, w, h, world), (w, h))
            self.assertEqual(ae_mock.world_to_wire(32, w, h, world), wire, (w, h))

    def test_python_float_would_have_changed_these_values(self):
        # control: a float32 -> Python float -> float32 trip quiets a signalling NaN, so the checks above
        # would catch any item access of an array("f") in the conversion code
        signalling = struct.pack("<I", 0x7F800001)
        arr = array.array("f")
        arr.frombytes(signalling)
        trip = array.array("f", [arr[0]])
        self.assertNotEqual(trip.tobytes(), signalling, "this interpreter keeps sNaN through float(): the control is moot")

    def test_planar_layout_is_g_b_r_a(self):
        planes = [1, 2, 3, 4, 5, 6, 7, 8]                     # G G, B B, R R, A A
        world = ae_mock.wire_to_world(32, 2, 1, planar_words_to_wire(planes, 2))
        self.assertEqual(struct.unpack("<8I", world), (7, 5, 1, 3, 8, 6, 2, 4))


# ---------------------------------------------------------------------------------------------
# alpha

class Alpha(TC):
    def test_8bpc_colour_is_untouched_by_alpha_both_ways(self):
        for a in (0, 1, 127, 128, 254, 255):
            wire = bytes([200, 100, 50, a])
            world = ae_mock.wire_to_world(8, 1, 1, wire)
            self.assertEqual(tuple(world), (a, 200, 100, 50), "alpha %d: straight in" % a)
            self.assertEqual(ae_mock.world_to_wire(8, 1, 1, bytes([a, 255, 255, 255])), bytes([255, 255, 255, a]), "alpha %d: straight out" % a)

    def test_16bpc_colour_is_untouched_by_alpha_both_ways(self):
        for a in (0, 1, 32767, 32768, 65534, 65535):
            world = array.array("H")
            world.frombytes(ae_mock.wire_to_world(16, 1, 1, struct.pack("<4H", 40000, 20000, 60000, a)))
            self.assertEqual(list(world), [rs_u16_to_q15(a), rs_u16_to_q15(40000), rs_u16_to_q15(20000), rs_u16_to_q15(60000)], "alpha %d" % a)
            out = array.array("H")
            out.frombytes(ae_mock.world_to_wire(16, 1, 1, array.array("H", [rs_u16_to_q15(a), 32768, 16384, 5]).tobytes()))
            self.assertEqual(list(out), [65535, rs_q15_to_u16(16384), rs_q15_to_u16(5), rs_q15_to_u16(rs_u16_to_q15(a))], "alpha %d" % a)

    def test_32bpc_colour_is_untouched_by_alpha_both_ways(self):
        for a in (0x00000000, 0x80000000, 0x00000001, 0x3F000000, 0x3F800000, 0x40000000, 0x7FC00000, 0xFF800000):
            words = [0x3F800000, 0x3F000000, 0x3E800000, a]                         # planes of one pixel: G B R A
            world = struct.unpack("<4I", ae_mock.wire_to_world(32, 1, 1, planar_words_to_wire(words, 1)))
            self.assertEqual(world, (a, 0x3E800000, 0x3F800000, 0x3F000000), hex(a))
            wire = struct.unpack("<4I", ae_mock.world_to_wire(32, 1, 1, struct.pack("<4I", a, 0x3F800000, 0x3F000000, 0x3E800000)))
            self.assertEqual(wire, (0x3F000000, 0x3E800000, 0x3F800000, a), hex(a))

    def test_no_premultiplication_anywhere_in_the_conversion_functions(self):
        import inspect
        for fn in (ae_mock.wire_to_world, ae_mock.world_to_wire):
            src = inspect.getsource(fn)
            for bad in ("premul", "alpha", "/ 255", "* a", "unmul"):
                self.assertNotIn(bad, src, "%s mentions %r" % (fn.__name__, bad))


# ---------------------------------------------------------------------------------------------
# world metadata, through the mock host's own methods on a byte array in place of guest memory

class FakeMem:
    def __init__(self, base, size):
        self.base, self.buf = base, bytearray(size)

    def read(self, addr, n):
        o = addr - self.base
        assert 0 <= o and o + n <= len(self.buf), "read outside the fake arena: %#x +%d" % (addr, n)
        return bytes(self.buf[o:o + n])

    def write(self, addr, data):
        o = addr - self.base
        assert 0 <= o and o + len(data) <= len(self.buf), "write outside the fake arena: %#x +%d" % (addr, len(data))
        self.buf[o:o + len(data)] = data


class FakeQl:
    def __init__(self, base, size):
        self.mem = FakeMem(base, size)


_LAYOUTS = None


def make_host(depth, w, h, arena=16 << 20):
    global _LAYOUTS
    if _LAYOUTS is None:
        _LAYOUTS = ae_host.Layouts()
    m = ae_mock.Mock.__new__(ae_mock.Mock)                 # no emulator: Mock.__init__ would load a plugin
    base = ae_host.Emu.ARENA
    m.lay, m.ql, m.arena_top = _LAYOUTS, FakeQl(base, arena), base
    m.configure(depth, w, h, "24", 1)
    m.worlds, m.inputs, m.note, m.reg_params = {}, {}, {}, []
    m.num_params, m.utils_addr, m.effect_ref, m.basic_addr, m._inter_cache = 1, 0, 0, 0, {}
    m.cur_frame = m.cur_time = 0
    m.fn_tramp = lambda full, handler, fnsig: 0            # trampolines need the emulator
    return m


LAYER_FIELDS = ("world_flags", "data", "rowbytes", "width", "height", "extent_hint.left", "extent_hint.top",
                "extent_hint.right", "extent_hint.bottom", "pix_aspect_ratio.num", "pix_aspect_ratio.den",
                "origin_x", "origin_y", "reserved0", "reserved1", "platform_ref", "reserved_long1",
                "reserved_long2", "reserved_long3", "reserved_long4", "dephault")


def layer_fields(view):
    return {k: view.get(k) for k in LAYER_FIELDS}


class WorldMetadata(TC):
    def check_world(self, m, view, w, h, depth, label):
        f = layer_fields(view)
        deep = m.lay.e("PF_WorldFlag_DEEP")
        wr = m.lay.e("PF_WorldFlag_WRITEABLE")
        exp = {"width": w, "height": h, "rowbytes": w * BPP[depth],
               "world_flags": wr | (deep if depth == 16 else 0),
               "extent_hint.left": 0, "extent_hint.top": 0, "extent_hint.right": w, "extent_hint.bottom": h,
               "pix_aspect_ratio.num": 1, "pix_aspect_ratio.den": 1, "origin_x": 0, "origin_y": 0,
               "reserved0": 0, "reserved1": 0, "platform_ref": 0, "reserved_long1": 0, "reserved_long2": 0,
               "reserved_long3": 0, "reserved_long4": 0, "dephault": 0}
        for k, v in exp.items():
            self.assertEqual(f[k], v, "%s: %s" % (label, k))
        self.assertNotEqual(f["data"], 0, label)
        self.assertEqual(f["data"] % 64, 0, label + ": data is 64 byte aligned")
        return f

    def test_input_and_output_worlds_describe_the_case_at_every_size_and_depth(self):
        for depth in (8, 16, 32):
            for w, h in SIZES:
                m = make_host(depth, w, h)
                wire = random_wire(depth, w, h, "meta-%d-%dx%d" % (depth, w, h))
                world = ae_mock.wire_to_world(depth, w, h, wire)
                m.inputs = {"input": [world, world]}
                # params[0] of a classic render: the input layer
                arr = m.params_array(0, 1)
                d0 = m.r64(arr)
                pd = m.view(d0, "PF_ParamDef")
                ld = m.view(pd.addr("u.ld.reserved0"), "PF_LayerDef")
                f = self.check_world(m, ld, w, h, depth, "input %d bpc %dx%d" % (depth, w, h))
                self.assertEqual(m.rd(f["data"], w * h * BPP[depth]), world, "input pixels reach the plugin unchanged")
                # the output world
                out = m.output_world()
                self.check_world(m, out.view, w, h, depth, "output %d bpc %dx%d" % (depth, w, h))
                self.assertEqual(m.read_world(out), bytes(w * h * BPP[depth]), "output starts cleared")
                # what the plugin writes comes back as is
                m.wr(out.data, world)
                self.assertEqual(m.read_world(out), world)
                self.assertEqual(ae_mock.world_to_wire(depth, w, h, m.read_world(out)),
                                 ae_mock.world_to_wire(depth, w, h, world))

    def test_pixel_xy_is_at_data_plus_y_rowbytes_plus_x_bpp(self):
        for depth in (8, 16, 32):
            w, h = 5, 3
            m = make_host(depth, w, h)
            wire = bytearray()
            code = {8: "<4B", 16: "<4H", 32: "<4I"}[depth]
            if depth == 32:
                words = []
                for plane in range(4):
                    words += [(plane + 1) * 1000 + y * 10 + x for y in range(h) for x in range(w)]
                wire = struct.pack("<%dI" % (4 * w * h), *words)
            else:
                for y in range(h):
                    for x in range(w):
                        wire += struct.pack(code, x + 1, y + 1, 50 + x + y, 200)
                wire = bytes(wire)
            world = ae_mock.wire_to_world(depth, w, h, wire)
            wd = m.new_world(w, h, depth, world)
            f = layer_fields(wd.view)
            for y in range(h):
                for x in range(w):
                    o = f["data"] + y * f["rowbytes"] + x * BPP[depth]
                    px = struct.unpack(code, m.rd(o, BPP[depth]))
                    if depth == 32:
                        # A R G B from planes G B R A
                        want = (4000 + y * 10 + x, 3000 + y * 10 + x, 1000 + y * 10 + x, 2000 + y * 10 + x)
                    elif depth == 8:
                        want = (200, x + 1, y + 1, 50 + x + y)
                    else:
                        want = tuple(rs_u16_to_q15(v) for v in (200, x + 1, y + 1, 50 + x + y))
                    self.assertEqual(px, want, (depth, x, y))

    def test_no_layer_is_data_null_and_size_zero(self):
        m = make_host(8, 4, 4)
        m.inputs = {"input": None}
        pd = m.new("PF_ParamDef")
        m.fill_param(pd.a, 0, 0, 0)
        ld = m.view(pd.addr("u.ld.reserved0"), "PF_LayerDef")
        self.assertEqual((ld.get("data"), ld.get("width"), ld.get("height")), (0, 0, 0))

    def test_frame_outside_the_clip_is_a_transparent_full_size_world(self):
        m = make_host(16, 3, 2)
        world = ae_mock.wire_to_world(16, 3, 2, random_wire(16, 3, 2, "outside"))
        m.inputs = {"input": [world]}
        pd = m.new("PF_ParamDef")
        m.fill_param(pd.a, 0, 100, 1)
        ld = m.view(pd.addr("u.ld.reserved0"), "PF_LayerDef")
        self.check_world(m, ld, 3, 2, 16, "outside")
        self.assertEqual(m.rd(ld.get("data"), 3 * 2 * 8), bytes(48))

    def test_in_data_size_extent_aspect_and_origins(self):
        for depth in (8, 16, 32):
            for w, h in SIZES:
                m = make_host(depth, w, h)
                ind = m.new("PF_InData")
                m.fill_in_data(ind, 0)
                g = ind.get
                self.assertEqual((g("width"), g("height")), (w, h), (depth, w, h))
                self.assertEqual((g("extent_hint.left"), g("extent_hint.top"), g("extent_hint.right"), g("extent_hint.bottom")), (0, 0, w, h))
                self.assertEqual((g("pixel_aspect_ratio.num"), g("pixel_aspect_ratio.den")), (1, 1))
                self.assertEqual((g("downsample_x.num"), g("downsample_x.den"), g("downsample_y.num"), g("downsample_y.den")), (1, 1, 1, 1))
                self.assertEqual((g("output_origin_x"), g("output_origin_y"), g("pre_effect_source_origin_x"), g("pre_effect_source_origin_y")), (0, 0, 0, 0))
                self.assertEqual(g("quality"), m.lay.e("PF_Quality_HI"))
                self.assertEqual(g("field"), m.lay.e("PF_Field_FRAME"))
                self.assertEqual((g("shutter_angle"), g("shutter_phase"), g("in_flags")), (0, 0, 0))

    def test_in_data_and_the_input_layer_agree_with_each_other(self):
        for w, h in ((1, 1), (3, 2), (37, 23)):
            m = make_host(8, w, h)
            m.inputs = {"input": [bytes(w * h * 4)]}
            ind = m.new("PF_InData")
            m.fill_in_data(ind, 0)
            arr = m.params_array(0, 0)
            ld = m.view(m.view(m.r64(arr), "PF_ParamDef").addr("u.ld.reserved0"), "PF_LayerDef")
            self.assertEqual((ind.get("width"), ind.get("height")), (ld.get("width"), ld.get("height")))
            self.assertEqual(ld.get("rowbytes"), ind.get("width") * 4)

    def test_every_world_is_a_fresh_copy(self):
        # a plugin that writes into an input world cannot change the next checkout
        m = make_host(8, 2, 2)
        world = bytes(range(16))
        m.inputs = {"input": [world]}
        pd1, pd2 = m.new("PF_ParamDef"), m.new("PF_ParamDef")
        m.fill_param(pd1.a, 0, 0, 0)
        d1 = m.view(pd1.addr("u.ld.reserved0"), "PF_LayerDef").get("data")
        m.wr(d1, bytes(16))
        m.fill_param(pd2.a, 0, 0, 0)
        d2 = m.view(pd2.addr("u.ld.reserved0"), "PF_LayerDef").get("data")
        self.assertNotEqual(d1, d2)
        self.assertEqual(m.rd(d2, 16), world)


# ---------------------------------------------------------------------------------------------
# what the mock does not do yet (each failure is a finding)

class GapPixelFormatHandshake(TC):
    def test_pf_world_suite_get_pixel_format_has_a_handler(self):
        self.assertTrue(hasattr(ae_mock.Mock, "cb_PF_WorldSuite2_PF_GetPixelFormat"),
                        "PF World Suite 2 PF_GetPixelFormat is an unimplemented stub: it returns A_Err_GENERIC and never writes the "
                        "PF_PixelFormat, so a plugin cannot learn that its world is ARGB128 (32 bpc has no PF_WorldFlag_DEEP)")

    def test_pf_world_suite_new_world_has_a_handler(self):
        self.assertTrue(hasattr(ae_mock.Mock, "cb_PF_WorldSuite2_PF_NewWorld"),
                        "PF World Suite 2 PF_NewWorld (width, height, clear, pixel format) is an unimplemented stub")

    def test_pixel_data_callbacks_have_handlers(self):
        missing = [n for n in ("cb_utils_get_pixel_data8", "cb_utils_get_pixel_data16") if not hasattr(ae_mock.Mock, n)]
        self.assertEqual(missing, [], "unimplemented: they return A_Err_GENERIC and never write the pixel pointer")

    def test_16_and_32_bpc_are_only_offered_to_a_plugin_that_declares_them(self):
        text = (HERE / "ae_mock.py").read_text() + (HERE / "worker.py").read_text()
        self.assertTrue(any(k in text for k in ("DEEP_COLOR_AWARE", "FLOAT_COLOR_AWARE", "1 << 25", "1 << 12")),
                        "neither ae_mock nor worker tests PF_OutFlag_DEEP_COLOR_AWARE / PF_OutFlag2_FLOAT_COLOR_AWARE before "
                        "handing out a 16 or 32 bpc world; After Effects hands such a plugin an 8 bpc world instead")


class GapNewWorldFlags(TC):
    """utils->new_world takes PF_NewWorldFlags (CLEAR_PIXELS = 1, DEEP_PIXELS = 2); the mock tests PF_WorldFlag_DEEP (= 1)."""

    def depth_for(self, session_depth, flags):
        m = make_host(session_depth, 4, 4)
        wp = m.new("PF_LayerDef")
        self.assertEqual(m.cb_utils_new_world(0, 4, 4, flags, wp.a), 0)
        return m.note["depth"]

    def test_deep_pixels_flag_makes_a_16_bpc_world(self):
        for session in (8, 16):
            self.assertEqual(self.depth_for(session, 2), 16,
                             "PF_NewWorldFlag_DEEP_PIXELS (2) in a %d bpc session did not give a 16 bpc world" % session)

    def test_clear_pixels_flag_alone_does_not_make_a_16_bpc_world(self):
        for session in (8, 16):
            self.assertEqual(self.depth_for(session, 1), 8,
                             "PF_NewWorldFlag_CLEAR_PIXELS (1) in a %d bpc session gave a deep world (the mock reads it as PF_WorldFlag_DEEP)" % session)

    def test_no_flags_makes_an_8_bpc_world(self):
        for session in (8, 16):
            self.assertEqual(self.depth_for(session, 0), 8)


# ---------------------------------------------------------------------------------------------
# the compiled Rust (optional): FX_CONV_DUMP=<program built from frame.rs>

@unittest.skipUnless(os.environ.get("FX_CONV_DUMP") and os.path.exists(os.environ.get("FX_CONV_DUMP", "")), "FX_CONV_DUMP is not set")
class CompiledRust(TC):
    W, H = 37, 23

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        w, h = cls.W, cls.H
        n = w * h
        r = rng("compiled")
        cls.in8 = bytes(r.randrange(256) for _ in range(4 * n))
        cls.in16 = struct.pack("<%dH" % (4 * n), *[r.randrange(65536) for _ in range(4 * n)])
        words = SPECIAL_WORDS + [r.getrandbits(32) for _ in range(4 * n - len(SPECIAL_WORDS))]
        cls.in32 = struct.pack("<%dI" % (4 * n), *words)
        for name, data in (("8", cls.in8), ("16", cls.in16), ("32", cls.in32)):
            (d / ("in%s.bin" % name)).write_bytes(data)
        subprocess.run([os.environ["FX_CONV_DUMP"], str(d), str(w), str(h)], check=True)
        cls.dir = d

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_u16_to_q15_every_value_against_the_harness(self):
        raw = (self.dir / "u16_to_q15.bin").read_bytes()
        rust = struct.unpack("<65536H", raw)
        to_q, _ = ae_mock.luts()
        bad = [(v, to_q[v], rust[v]) for v in range(65536) if to_q[v] != rust[v]]
        note("compiled Rust u16_to_q15 vs harness table: 65536 values, %d mismatches" % len(bad))
        self.assertEqual(first(bad), [])

    def test_q15_to_u16_every_value_against_the_harness(self):
        raw = (self.dir / "q15_to_u16.bin").read_bytes()
        rust = struct.unpack("<65536H", raw)
        _, from_q = ae_mock.luts()
        bad = []
        for v in range(65536):
            h = from_q[min(v, 32768)]                       # the harness clamps before the table
            if h != rust[v]:
                bad.append((v, h, rust[v]))
        note("compiled Rust q15_to_u16 vs harness: 65536 world values (32769 legal, 32767 clamped), %d mismatches" % len(bad))
        self.assertEqual(first(bad), [])

    def test_frame_from_bytes_equals_the_harness_world(self):
        w, h = self.W, self.H
        for depth, wire, name in ((8, self.in8, "8"), (16, self.in16, "16"), (32, self.in32, "32")):
            px = (self.dir / ("px%s.bin" % name)).read_bytes()
            code = {8: "B", 16: "H", 32: "I"}[depth]
            rgba = struct.unpack("<%d%s" % (4 * w * h, code), px)            # R G B A per pixel, raw words
            if depth == 16:
                rgba = tuple(rs_u16_to_q15(v) for v in rgba)                  # the effects' range
            argb = []
            for i in range(w * h):
                r_, g, b, a = rgba[4 * i:4 * i + 4]
                argb += [a, r_, g, b]
            world = struct.unpack("<%d%s" % (4 * w * h, code), ae_mock.wire_to_world(depth, w, h, wire))
            self.assertEqual(list(world), argb, "%d bpc" % depth)

    def test_frame_to_bytes_equals_the_harness_wire(self):
        w, h = self.W, self.H
        for depth, wire, name in ((8, self.in8, "8"), (16, self.in16, "16"), (32, self.in32, "32")):
            rt = (self.dir / ("rt%s.bin" % name)).read_bytes()
            self.assertEqual(rt, wire, "Rust read then write, %d bpc" % depth)
            if depth != 16:
                world = ae_mock.wire_to_world(depth, w, h, wire)
                self.assertEqual(ae_mock.world_to_wire(depth, w, h, world), rt, "%d bpc" % depth)


if __name__ == "__main__":
    unittest.main()
