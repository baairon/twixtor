#!/usr/bin/env python3
"""tests for tools/audio_map.py: the time map of --audio retime and the varispeed render. stdlib unittest,
no ffmpeg, no vfx.

    python tools/test_audio_map.py
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from array import array
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "audio_map.py")
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import audio_map as am  # noqa: E402
import speed_curve as sc  # noqa: E402


def noise(count, seed=1):
    """`count` f32 samples in -1..1 from a fixed linear congruential sequence, so a test sees real data."""
    out, x = array("f", bytes(4 * count)), seed
    for i in range(count):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out[i] = x / 1073741824.0 - 1.0
    return out


def walked(keys, frames, fps=Fraction(24)):
    """speed_curve's own positions of output frames 0..frames-1 for the speed keys."""
    step = sc.ticks_per_frame(fps)
    curve = sc.Curve(sc.parse_keys(keys, step, sc.ticks_per_second(fps)))
    return sc.positions(curve, step, frames)[:frames]


def run(*argv):
    return subprocess.run([sys.executable, "-I", SCRIPT] + list(argv), capture_output=True, text=True)


class SampleCase(unittest.TestCase):
    """assertSamples: two runs of samples compared as the bytes of their 64 bit values, so a failure on a list of
    200000 samples reports the first index that differs at once instead of diffing the lists for minutes."""

    def assertSamples(self, got, want, msg=None):
        got_bytes, want_bytes = array("d", got).tobytes(), array("d", want).tobytes()
        if got_bytes == want_bytes:
            return
        got_bits, want_bits = memoryview(got_bytes).cast("Q"), memoryview(want_bytes).cast("Q")
        count = min(len(got_bits), len(want_bits))
        at = next((i for i in range(count) if got_bits[i] != want_bits[i]), count)
        shown = [repr(seq[at]) if at < len(seq) else "the end" for seq in (got, want)]
        self.fail(f"{msg + ': ' if msg else ''}samples differ first at index {at}: {shown[0]} != {shown[1]} "
                  f"({len(got_bits)} samples against {len(want_bits)})")


class MapTests(unittest.TestCase):
    """the knots of build_knots against the effect's arithmetic (speed_curve) and the rules of the map."""

    def test_speed_knots_equal_speed_curve_walk_at_whole_frames(self):
        for keys in ("100", "50", "-100", "250@0;25@30", "100@0e;40@20e;100@40e", "200@0h;30@20h;100@30",
                     "100@0;0@10;0@20h;-50@30", "100@0o200/50;20@24i-50/30", "33.333@0;66.667@17.5"):
            for fps in (Fraction(24), Fraction(30000, 1001), Fraction(2997, 100)):
                with self.subTest(keys=keys, fps=fps):
                    text = keys if "@" in keys else f"{keys}@0"
                    shown = am.positions(am.build_knots("retime", [f"speed_percent={keys}"], fps, 48), 48)
                    self.assertEqual(shown, walked(text, 48, fps))
                    # every whole frame is a knot, and the half frames lie between
                    knots = am.build_knots("retime-fgbg", [f"speed_percent={keys}"], fps, 48)
                    self.assertEqual([p for t, p in knots if t == int(t)][:48], walked(text, 48, fps))
                    self.assertEqual([t for t, _ in knots], [k / 2 for k in range(97)])

    def test_the_half_frame_knot_is_the_engines_half_step(self):
        # 100@0;0@1: the half step at k = 1 reads tick trunc(50 + 0.05) = 50, speed 50%
        knots = am.build_knots("retime", ["speed_percent=100@0;0@1"], 24, 1)
        self.assertEqual(knots, [(0.0, 0.0), (0.5, 0.25), (1.0, 0.25)])

    def test_constant_speeds(self):
        self.assertEqual(am.build_knots("retime", [], 24, 2), [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (1.5, 1.5), (2.0, 2.0)])
        self.assertEqual(am.positions(am.build_knots("retime", ["speed_percent=50"], 24, 5), 5), [0.0, 0.5, 1.0, 1.5, 2.0])
        self.assertEqual(am.positions(am.build_knots("retime", ["speed_percent=-100"], 24, 3), 3), [0.0, -1.0, -2.0])

    def test_frame_num_holds_jumps_and_falls(self):
        params = ["time_remap_mode=2", "frame_num=0@0;24@24;24@36h;48@48;0@96"]
        knots = am.build_knots("retime", params, 24, 96)
        shown = am.positions(knots, 96)
        self.assertEqual(shown[:25], [float(n) for n in range(25)])
        self.assertEqual(shown[24:48], [24.0] * 24, "a flat segment and a hold stand still")
        self.assertEqual(shown[48:], [48.0 - (n - 48) for n in range(48, 96)], "it jumps, then falls")
        # the jump is two knots at one time
        self.assertIn((48.0, 24.0), knots)
        self.assertEqual(knots[knots.index((48.0, 24.0)) + 1], (48.0, 48.0))
        # straight segments need no knot inside a frame
        self.assertTrue(all(t == int(t) for t, _ in knots))

    def test_frame_num_is_read_on_continuous_ticks(self):
        # eased keys bend between frames: knots at tenths of a frame, the whole frames exactly as vfx samples
        params = ["time_remap_mode=2", "frame_num=0@0e;40@10e"]
        knots = am.build_knots("vectors-in", params, 24, 12)
        step = 100
        curve = sc.Curve(sc.parse_keys("0@0e;40@10e", step, 2400, 200000.0), 200000.0)
        inside = [k for k in knots if k[0] < 10]
        self.assertEqual(len(inside), 100)
        for t, p in inside:
            self.assertAlmostEqual(p, curve.percent_at(t * step), delta=1e-9)
        self.assertEqual([p for t, p in knots if t == int(t)][:11], [curve.percent_at(n * step) for n in range(11)])
        self.assertEqual(am.positions(knots, 12)[10:], [40.0, 40.0])
        # a hold key at a fractional frame jumps inside the frame
        knots = am.build_knots("retime", ["time_remap_mode=2", "frame_num=5@0;5@2.5h;9@2.75"], 24, 4)
        self.assertEqual(knots, [(0.0, 5.0), (1.0, 5.0), (2.0, 5.0), (2.5, 5.0), (2.75, 5.0), (2.75, 9.0), (3.0, 9.0),
                                 (4.0, 9.0)])

    def test_frame_num_reaches_its_own_range(self):
        # frame_num takes values up to 200000, beyond the speed range speed_curve checks by default
        knots = am.build_knots("retime", ["time_remap_mode=2", "frame_num=150000@0;-150000@10"], 24, 10)
        self.assertEqual(am.positions(knots, 2), [150000.0, 120000.0])
        with self.assertRaises(sc.CurveError):
            sc.parse_keys("150000@0;-150000@10", 100)
        with self.assertRaises(am.MapError):
            am.build_knots("retime", ["speed_percent=15000@0;1@1"], 24, 2)

    def test_bezier_handle_keys_flow_through(self):
        keys = "100@0o300/40;30@24i0/60o-20/25;100@48i50/50"
        shown = am.positions(am.build_knots("retime", [f"speed_percent={keys}"], 24, 60), 60)
        self.assertEqual(shown, walked(keys, 60))
        fn = "0@0o48/50;30@24i24/33.3"
        knots = am.build_knots("retime", ["time_remap_mode=2", f"frame_num={fn}"], 24, 30)
        curve = sc.Curve(sc.parse_keys(fn, 100, 2400, 200000.0), 200000.0)
        self.assertEqual(am.positions(knots, 30), [curve.percent_at(n * 100) for n in range(30)])
        self.assertGreater(len(knots), 24 * 10, "a handle bends the curve between frames")
        # handles at an NTSC rate divide by its own ticks a second
        shown = am.positions(am.build_knots("retime", [f"speed_percent={keys}"], "30000/1001", 60), 60)
        self.assertEqual(shown, walked(keys, 60, Fraction(30000, 1001)))

    def test_the_offset_counts_source_frames_at_the_input_rate(self):
        def first(params):
            return am.positions(am.build_knots("retime", params, 24, 2), 2)

        self.assertEqual(first(["src_time_offset=3"]), [3.0, 4.0])
        # input_frame_rate 12 in a 24 fps clip: 3 source frames are 6 clip frames
        rates = ["in_fps_is_out_fps=0", "input_frame_rate=12"]
        self.assertEqual(first(["src_time_offset=3"] + rates), [6.0, 7.0])
        # fields double the source rate, which halves it
        self.assertEqual(first(["src_time_offset=3", "input_fields=2"] + rates), [3.0, 4.0])
        self.assertEqual(first(["src_time_offset=3", "input_fields=3"]), [1.5, 2.5], "both rates 30, then fields")
        # the rates alone move nothing, the offset is truncated, and the other effects have none
        self.assertEqual(first(rates + ["input_fields=2"]), [0.0, 1.0])
        self.assertEqual(first(["src_time_offset=3.99"]), [3.0, 4.0])
        self.assertEqual(am.positions(am.build_knots("retime-fgbg", ["input_fields=2"] + rates, 24, 2), 2), [0.0, 1.0])
        # a typed rate that snaps: 23.976 in a 24 fps clip
        snapped = first(["src_time_offset=10", "in_fps_is_out_fps=0", "input_frame_rate=23.97601"])
        self.assertEqual(snapped, [10 * 24 / 23.976, 10 * 24 / 23.976 + 1])
        # read per output frame: a keyed offset jumps at the frame it changes
        knots = am.build_knots("retime", ["src_time_offset=0@0;0@2h;5@3"], 24, 4)
        self.assertEqual(am.positions(knots, 4), [0.0, 1.0, 2.0, 8.0])
        self.assertIn((3.0, 3.0), knots)

    def test_the_source_display(self):
        self.assertEqual(am.build_knots("retime", ["display=2", "speed_percent=37", "src_time_offset=4"], 24, 2),
                         [(0.0, 4.0), (1.0, 5.0), (2.0, 6.0)])
        for effect in ("retime-fgbg", "vectors-in"):
            self.assertEqual(am.build_knots(effect, ["display=2", "speed_percent=37"], 24, 2),
                             [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])
        # the display is read per frame
        knots = am.build_knots("retime", ["display=1@0;2@2", "speed_percent=50"], 24, 4)
        self.assertEqual(am.positions(knots, 4), [0.0, 0.5, 2.0, 3.0])

    def test_identity_effects_and_mode_switches(self):
        for effect in ("create-vectors", "selftest-invert", "anything"):
            self.assertEqual(am.build_knots(effect, ["speed_percent=50", "display=2"], 24, 10), [(0.0, 0.0), (10.0, 10.0)])
        knots = am.build_knots("retime", ["time_remap_mode=1@0;2@3", "frame_num=40", "speed_percent=200"], 24, 5)
        self.assertEqual(am.positions(knots, 5), [0.0, 2.0, 4.0, 40.0, 40.0])
        self.assertEqual(knots[6:8], [(3.0, 6.0), (3.0, 40.0)])
        # ids the map does not read are passed over; a malformed value is refused
        self.assertEqual(am.build_knots("retime", ["vectors=1", "frame_interp=2"], 24, 1), am.build_knots("retime", [], 24, 1))
        for bad in (["speed_percent=fast"], ["display=x"], ["in_fps_is_out_fps=maybe"], ["display=1@0;2@0"], ["nonsense"]):
            with self.assertRaises(am.MapError, msg=bad):
                am.build_knots("retime", bad, 24, 2)

    def test_the_clip_rate_scales_clip_frames(self):
        # a clip at 60 fps under a 24 fps output: clip time is unchanged, counted in its own frames
        knots = am.build_knots("retime", ["speed_percent=50"], 24, 2, clip_fps=60)
        self.assertEqual(knots, [(0.0, 0.0), (0.5, 0.625), (1.0, 1.25), (1.5, 1.875), (2.0, 2.5)])
        self.assertEqual(am.build_knots("retime", [], 24, 1, clip_fps=Fraction(24)), am.build_knots("retime", [], 24, 1))


class RenderTests(SampleCase):
    """the varispeed render: lengths, exact copies, interpolation, silence, fades and channels."""

    def test_output_lengths(self):
        self.assertEqual(am.output_samples(24, 48000, 24), 48000)
        self.assertEqual(am.output_samples(1, 44100, Fraction(30000, 1001)), 1471)  # 1471.47
        self.assertEqual(am.output_samples(30, 44100, Fraction(30000, 1001)), 44144)  # 44144.1
        self.assertEqual(am.output_samples(7, 8000, 60), 933)  # 933.33
        self.assertEqual(am.output_samples(3, 22050, 4), 16538)  # 16537.5 rounds up
        out = am.render(am.build_knots("retime", ["speed_percent=37"], 30000, 7), noise(500), 2, 8000, 0.0,
                        30000, 7, am.output_samples(7, 8000, 30000) + 3)
        self.assertEqual(len(out), 2 * 5)
        self.assertEqual(am.fade_length(48000), 96)
        self.assertEqual(am.fade_length(44100), 88)

    def test_a_failing_comparison_names_the_first_index_at_once(self):
        got, want = array("f", range(200000)), [float(i) for i in range(200000)]
        want[123456] = -1.0
        with self.assertRaises(AssertionError) as caught:
            self.assertSamples(got, want)
        self.assertIn("samples differ first at index 123456: 123456.0 != -1.0", str(caught.exception))
        with self.assertRaises(AssertionError) as caught:
            self.assertSamples(array("f", [0.0, -0.0]), [0.0, 0.0])
        self.assertIn("index 1", str(caught.exception), "-0.0 is a sample of its own")
        with self.assertRaises(AssertionError) as caught:
            self.assertSamples(got[:3], [0.0, 1.0])
        self.assertIn("index 2: 2.0 != the end (3 samples against 2)", str(caught.exception))

    def test_100_percent_is_bit_exact(self):
        for channels in (1, 2, 6):
            pcm = noise(2000 * 24 * channels, seed=channels)
            pcm[5] = -0.0
            samples = am.output_samples(24, 48000, 24)
            for effect, params in (("retime", []), ("retime-fgbg", ["speed_percent=100"]), ("create-vectors", []),
                                   ("retime", ["time_remap_mode=2", "frame_num=0@0;24@24"])):
                knots = am.build_knots(effect, params, 24, 24)
                out = am.render(knots, pcm, channels, 48000, 0.0, 24, 24, samples)
                self.assertSamples(out, pcm, (channels, effect, params))
        # an offset shifts by whole samples, exactly: 0.021333 s as ffprobe prints 1024 samples at 48 kHz.
        # the samples before the audio starts are silent and its first ones fade in
        pcm = noise(48000)
        out = am.render(am.build_knots("retime", [], 24, 24), pcm, 1, 48000, 0.021333, 24, 24, 48000)
        self.assertSamples(out[1024 + 96:], pcm[96:48000 - 1024])
        self.assertSamples(out[:1024], [0.0] * 1024)
        self.assertEqual(am.sample_shift(0.021333, 48000), 1024)
        self.assertEqual(am.sample_shift(-0.0105, 44100), -463)

    def test_50_percent_interpolates(self):
        pcm = noise(48000, seed=7)
        out = am.render(am.build_knots("retime", ["speed_percent=50"], 24, 24), pcm, 1, 48000, 0.0, 24, 24, 48000)
        self.assertSamples(out[0::2], pcm[:24000])
        f32 = array("f", [pcm[i] + (pcm[i + 1] - pcm[i]) * 0.5 for i in range(24000)])
        self.assertSamples(out[1::2], f32)
        # a ramp source shows the position it plays: sample value = source index
        ramp = array("f", range(48000))
        out = am.render(am.build_knots("retime", ["speed_percent=50"], 24, 24), ramp, 1, 48000, 0.0, 24, 24, 48000)
        self.assertSamples(out, [i / 2 for i in range(48000)])

    def test_a_speed_ramp_follows_the_knots(self):
        ramp = array("f", range(96000))
        knots = am.build_knots("retime", ["speed_percent=100@0;25@24;100@48"], 24, 48)
        out = am.render(knots, ramp, 1, 48000, 0.0, 24, 48, am.output_samples(48, 48000, 24))
        times = [k[0] for k in knots]
        for i in range(0, 96000, 997):
            want = am.value_at(knots, i / 2000, times) * 2000
            self.assertAlmostEqual(out[i], want, delta=0.02, msg=i)
        # at every output frame the audio plays the clip position the picture shows
        shown = am.positions(knots, 48)
        for n in range(48):
            self.assertAlmostEqual(out[n * 2000] / 2000, shown[n], delta=1e-5)

    def test_minus_100_percent_plays_backwards(self):
        pcm = noise(2000 * 48, seed=3)
        # from clip frame 40 back towards 0: output sample i is input sample 80000 - i
        knots = am.build_knots("retime", ["speed_percent=-100", "src_time_offset=40"], 24, 40)
        out = am.render(knots, pcm, 1, 48000, 0.0, 24, 48, 80000)
        self.assertSamples(out[1:], [pcm[80000 - i] for i in range(1, 80000)])
        # the first sample is clip frame 40 itself, inside the clip
        self.assertEqual(out[0], pcm[80000])
        # without an offset it leaves the clip at once: one sample of sound, faded, then silence
        out = am.render(am.build_knots("retime", ["speed_percent=-100"], 24, 2), pcm, 1, 48000, 0.0, 24, 48, 4000)
        self.assertEqual(out[0], array("f", [pcm[0] / 96])[0])
        self.assertSamples(out[1:], [0.0] * 3999)

    def test_holds_are_silent_with_fades_and_jumps_cut(self):
        ramp = array("f", (1.0 + i for i in range(2000 * 120)))
        params = ["time_remap_mode=2", "frame_num=0@0;24@24;24@36h;48@48;0@96"]
        knots = am.build_knots("retime", params, 24, 96)
        out = am.render(knots, ramp, 1, 48000, 0.0, 24, 120, 2000 * 96)
        w = 96
        # frames 0..24 play straight, the last 2 ms before the hold fade out
        self.assertSamples(out[:48000 - w], [1.0 + i for i in range(48000 - w)])
        for k in range(w):
            i = 48000 - w + k
            self.assertEqual(out[i], array("f", [(1.0 + i) * ((w - k) / w)])[0])
        # the hold is silent
        self.assertSamples(out[48000:96000], [0.0] * 48000)
        # after the jump to clip frame 48 the sound fades in, then plays backwards
        for k in range(w):
            i = 96000 + k
            self.assertEqual(out[i], array("f", [(1.0 + 96000 - k) * ((k + 1) / w)])[0])
        self.assertSamples(out[96000 + w:96000 + 1000], [1.0 + 96000 - k for k in range(w, 1000)])

    def test_a_jump_between_two_sounding_places_cuts(self):
        ramp = array("f", range(2000 * 60))
        knots = am.build_knots("retime", ["src_time_offset=0@0;0@10h;20@11"], 24, 20)
        out = am.render(knots, ramp, 1, 48000, 0.0, 24, 60, 40000)
        # frames 0..10 at 0, from frame 11 on at +20 frames: no fade on either side of the cut
        self.assertEqual(out[22000 - 1], 21999.0)
        self.assertEqual(out[22000], 22000.0 + 40000)
        self.assertSamples(out[22000:], [float(i + 40000) for i in range(22000, 40000)])

    def test_clip_edges(self):
        ramp = array("f", (1.0 + i for i in range(2000 * 100)))
        # frame_num from -10 to 90 over 100 frames in a 50 frame clip: sound from p = 0 to p = 50
        knots = am.build_knots("retime", ["time_remap_mode=2", "frame_num=-10@0;90@100"], 24, 100)
        out = am.render(knots, ramp, 1, 48000, 0.0, 24, 50, 200000)
        start, end, w = 20000, 120000, 96
        self.assertSamples(out[:start], [0.0] * start)
        self.assertSamples(out[end:], [0.0] * (200000 - end))
        self.assertEqual(out[start], array("f", [1.0 / w])[0], "the first sample of sound at clip frame 0")
        self.assertSamples(out[start + w:end - w], [1.0 + i - start for i in range(start + w, end - w)])
        self.assertEqual(out[end - 1], array("f", [(1.0 + end - 1 - start) / w])[0])
        # the input's own end is an edge too: a clip longer than its audio
        short = ramp[:2000 * 30]
        out = am.render(am.build_knots("retime", [], 24, 40), short, 1, 48000, 0.0, 24, 40, 80000)
        self.assertSamples(out[:60000 - w], short[:60000 - w])
        self.assertEqual(out[59999], array("f", [short[59999] / w])[0])
        self.assertSamples(out[60000:], [0.0] * 20000)

    def test_runs_shorter_than_a_fade_fade_both_ways(self):
        # 2 fps at 8192 Hz: 4096 samples a frame, so knot times are exact; a fade is 16 samples
        ones = array("f", [1.0] * 8192)
        self.assertEqual(am.fade_length(8192), 16)
        s = 1 / 4096
        knots = [(0.0, -1.0), (16 * s, -1.0), (16 * s, 0.0), (24 * s, 8 * s), (1.0, 8 * s)]
        out = am.render(knots, ones, 1, 8192, 0.0, 2, 1, 4096)
        self.assertSamples(out[:16] + out[24:], [0.0] * 4088)
        self.assertEqual([v * 16 for v in out[16:24]], [1, 2, 3, 4, 4, 3, 2, 1])

    def test_channels_stay_interleaved(self):
        frames = 2000 * 24
        left = array("f", (float(i) for i in range(frames)))
        right = array("f", (-0.5 * i for i in range(frames)))
        pcm = array("f", bytes(8 * frames))
        pcm[0::2], pcm[1::2] = left, right
        out = am.render(am.build_knots("retime", ["speed_percent=50"], 24, 24), pcm, 2, 48000, 0.0, 24, 24, 48000)
        self.assertSamples(out[0::2], [i / 2 for i in range(48000)])
        self.assertSamples(out[1::2], [-0.25 * i for i in range(48000)])

    def test_the_audio_start_offset(self):
        ramp = array("f", range(48000))
        # the audio starts 0.25 s after the video: clip time 0.25 s is input sample 0
        out = am.render(am.build_knots("retime", [], 24, 24), ramp, 1, 48000, 0.25, 24, 24, 48000)
        self.assertSamples(out[:12000], [0.0] * 12000)
        self.assertSamples(out[12000 + 96:], [float(i) for i in range(96, 36000)])
        # a negative offset (audio before the video) reads past its first samples
        out = am.render(am.build_knots("retime", [], 24, 1), ramp, 1, 48000, -0.5, 24, 24, 2000)
        self.assertSamples(out, [float(24000 + i) for i in range(2000)])


class CommandLineTests(unittest.TestCase):
    """the two commands, run as apply.py runs them (python -I)."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="audio_map_test_")

    def tearDown(self):
        for name in os.listdir(self.dir):
            os.remove(os.path.join(self.dir, name))
        os.rmdir(self.dir)

    def test_render_writes_what_the_api_returns(self):
        pcm = noise(2 * 2000 * 30, seed=11)
        src, dst = os.path.join(self.dir, "a.f32"), os.path.join(self.dir, "b.f32")
        with open(src, "wb") as f:
            f.write(pcm.tobytes())
        params = ["speed_percent=100@0;30@10;-80@20", "src_time_offset=12", "vectors=1"]
        argv = ["render", "--effect", "retime", "--fps", "24/1", "--frames", "36", "--clip-frames", "30", "--rate",
                "48000", "--channels", "2", "--offset", "-0.01", "--in", src, "--out", dst]
        for p in params:
            argv += ["--param", p]
        proc = run(*argv)
        self.assertEqual((proc.returncode, proc.stderr), (0, ""))
        knots = am.build_knots("retime", params, 24, 36)
        want = am.render(knots, pcm, 2, 48000, -0.01, 24, 30, am.output_samples(36, 48000, 24))
        with open(dst, "rb") as f:
            self.assertEqual(f.read(), want.tobytes())
        self.assertEqual(os.path.getsize(dst), 2 * 4 * 72000)
        # an empty input renders silence of the right length
        open(src, "wb").close()
        self.assertEqual(run(*argv).returncode, 0)
        with open(dst, "rb") as f:
            self.assertEqual(f.read(), bytes(2 * 4 * 72000))

    def test_positions(self):
        proc = run("positions", "--effect", "retime", "--fps", "24", "--frames", "4", "--param", "speed_percent=50",
                   "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        doc = json.loads(proc.stdout)
        self.assertEqual(doc["positions"], [0.0, 0.5, 1.0, 1.5])
        self.assertEqual(doc["knots"][:3], [[0.0, 0.0], [0.5, 0.25], [1.0, 0.5]])
        self.assertEqual((doc["fps"], doc["clip_fps"], doc["frames"]), ("24/1", None, 4))
        text = run("positions", "--effect", "retime", "--fps", "24", "--frames", "2")
        self.assertEqual(text.stdout, "0 0.000000\n1 1.000000\n")

    def test_bad_input(self):
        proc = run("positions", "--effect", "retime", "--fps", "24", "--frames", "2", "--param", "speed_percent=x")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("[error] --param speed_percent: cannot read `x`", proc.stderr)
        for rate in ("0", "1/0", "29.97/1", "fast"):
            self.assertEqual(run("positions", "--effect", "retime", "--fps", rate, "--frames", "2").returncode, 1, rate)
        self.assertEqual(run("positions", "--effect", "retime", "--fps", "24", "--frames", "0").returncode, 2)
        missing = os.path.join(self.dir, "missing.f32")
        proc = run("render", "--effect", "retime", "--fps", "24", "--frames", "1", "--clip-frames", "1", "--rate",
                   "8000", "--channels", "1", "--in", missing, "--out", os.path.join(self.dir, "o.f32"))
        self.assertEqual(proc.returncode, 1)
        self.assertTrue(proc.stderr.startswith("[error] "), proc.stderr)


if __name__ == "__main__":
    unittest.main()
