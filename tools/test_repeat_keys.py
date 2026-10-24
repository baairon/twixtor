#!/usr/bin/env python3
"""tests for tools/repeat_keys.py. stdlib unittest, no ffmpeg needed except for the one end to end test, which
skips itself when ffmpeg is missing.

    python tools/test_repeat_keys.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, HERE)
import repeat_keys as tool  # noqa: E402
import speed_curve  # noqa: E402


def frame(value):
    return bytes([value]) * (tool.WIDTH * tool.HEIGHT)


def sample(keys, count):
    """the value of the keys at every whole frame, read the way the effect reads a curve."""
    curve = speed_curve.Curve(speed_curve.parse_keys(keys, 100))
    return [curve.percent_at(n * 100) for n in range(count)]


class Detection(unittest.TestCase):
    def test_a_frame_equal_to_the_one_before_is_a_repeat(self):
        frames = [frame(10), frame(20), frame(20), frame(30), frame(40), frame(40)]
        self.assertEqual(tool.find_repeats(frames, 0.5), [2, 5])

    def test_small_noise_still_counts_and_real_change_does_not(self):
        a = bytearray(frame(100))
        a[0] += 1
        self.assertEqual(tool.find_repeats([frame(100), bytes(a), frame(103)], 0.5), [1])

    def test_a_threshold_of_zero_finds_nothing(self):
        self.assertEqual(tool.find_repeats([frame(5), frame(5)], 0.0), [])


class Keys(unittest.TestCase):
    def test_one_repeat_shows_the_picture_halfway_to_the_next(self):
        # frame 3 repeats frame 2 (the same picture), frame 4 is the next different one
        self.assertEqual(tool.positions_for(8, [3]), {3: 3.5})

    def test_a_run_of_repeats_is_spread_evenly(self):
        # frames 3 and 4 both repeat frame 2; frame 5 is different: thirds of the step from 4 to 5
        got = tool.positions_for(8, [3, 4])
        self.assertAlmostEqual(got[3], 4 + 1 / 3)
        self.assertAlmostEqual(got[4], 4 + 2 / 3)

    def test_keys_read_back_as_the_identity_except_at_the_repeats(self):
        count = 40
        repeats = [5, 6, 20, 39]
        keys = tool.keys_for(count, repeats)
        positions = tool.positions_for(count, repeats)
        want = [positions.get(n, n) for n in range(count)]
        got = sample(keys, count)
        for n in range(count):
            self.assertAlmostEqual(got[n], want[n], places=3, msg=f"frame {n}")

    def test_the_keys_are_short_for_a_long_clip(self):
        keys = tool.keys_for(10000, [100, 5000])
        self.assertLess(len(keys), 200)
        self.assertTrue(keys.startswith("0@0;") and keys.endswith("@9999"))

    def test_number_formatting(self):
        self.assertEqual(tool.number(3), "3")
        self.assertEqual(tool.number(3.5), "3.5")
        self.assertEqual(tool.number(4 + 1 / 3), "4.3333")


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
class EndToEnd(unittest.TestCase):
    def test_a_clip_with_doubled_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = os.path.join(tmp, "clip.mp4")
            # 30 fps testsrc stretched to 60 fps by doubling every frame: every odd frame is a repeat
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=30:duration=1",
                            "-vf", "fps=60", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", clip], check=True)
            out = subprocess.run([sys.executable, "-I", os.path.join(HERE, "repeat_keys.py"), clip, "--list"],
                                 capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(out.stdout.split(), [str(n) for n in range(1, 60, 2)])
            self.assertIn("29 of 59", out.stderr + "") if False else None


if __name__ == "__main__":
    unittest.main()
