#!/usr/bin/env python3
"""tests for tools/speed_curve.py. stdlib unittest, no ffmpeg and no vfx needed.

    python tools/test_speed_curve.py

the expected positions are worked out by hand from the half frame sum in the module docstring.
"""

import contextlib
import io
import os
import sys
import unittest
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))

# no bytecode next to the tools, here or in the script the suite runs
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, HERE)
import speed_curve as tool  # noqa: E402

PLAN = "100@0e;100@120e;150@168e;40@204e;40@400e;100@454e;100@664e"


def keys_of(text, step=100):
    return tool.parse_keys(text, step)


def positions_of(text, frames, step=100):
    return tool.positions(tool.Curve(keys_of(text, step)), step, frames)


def run_main(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = tool.main(list(argv))
        except SystemExit as e:
            code = e.code
    return code, out.getvalue(), err.getvalue()


class Easing(unittest.TestCase):
    def test_values_at_the_middle_and_the_ends(self):
        for start in (False, True):
            for end in (False, True):
                self.assertEqual(tool.eased_progress(0.0, start, end), 0.0)
                self.assertEqual(tool.eased_progress(1.0, start, end), 1.0)
        self.assertEqual(tool.eased_progress(0.5, False, False), 0.5)
        self.assertEqual(tool.eased_progress(0.5, True, True), 0.5)
        self.assertEqual(tool.eased_progress(0.5, True, False), 0.375)
        self.assertEqual(tool.eased_progress(0.5, False, True), 0.625)

    def test_every_shape_is_monotonic(self):
        for start in (False, True):
            for end in (False, True):
                values = [tool.eased_progress(i / 1000, start, end) for i in range(1001)]
                self.assertEqual(values, sorted(values), (start, end))


class Keys(unittest.TestCase):
    def test_suffixes_ticks_and_order(self):
        keys = keys_of("50@10;100@0.5;25@12h;75@14e")
        self.assertEqual([k.tick for k in keys], [50, 1000, 1200, 1400])
        self.assertEqual([(k.hold, k.ease) for k in keys], [(False, False), (False, False), (True, False), (False, True)])
        self.assertEqual([k.value for k in keys], [100, 50, 25, 75])

    def test_a_leading_parameter_name_is_accepted(self):
        self.assertEqual(keys_of("speed_percent=100@0;50@24"), keys_of("100@0;50@24"))

    def test_bad_keys_are_refused(self):
        for text in ["100", "100@0;50", "100@x", "100@0he;50@10", "100@0eh;50@10", "100@e;50@10", "abc@0",
                     "100@0;50@0", "20000@0;50@10", "100@0;"]:
            with self.assertRaises(tool.CurveError, msg=text):
                keys_of(text)

    def test_the_time_base_follows_the_frame_rate(self):
        self.assertEqual(tool.ticks_per_frame(Fraction(60)), 100)
        self.assertEqual(tool.ticks_per_frame(Fraction(24)), 100)
        self.assertEqual(tool.ticks_per_frame(Fraction(30000, 1001)), 1001)
        self.assertEqual([k.tick for k in keys_of("1@1;1@2.5", 1001)], [1001, 2503])


class Positions(unittest.TestCase):
    def test_constant_speeds_land_on_whole_frames(self):
        self.assertEqual(positions_of("100@0;100@10", 10), [float(n) for n in range(11)])
        self.assertEqual(positions_of("50@0;50@10", 10), [n / 2 for n in range(11)])
        self.assertEqual(positions_of("200@0;200@10", 10), [2.0 * n for n in range(11)])
        self.assertEqual(positions_of("-100@0;-100@10", 4), [-float(n) for n in range(5)])

    def test_a_linear_ramp_by_hand(self):
        # speed 0 to 100% over 2 frames: read at 0.25, 0.5, 0.75 and 1 of the way, so 25, 50, 75, 100
        pos = positions_of("0@0;100@2", 2)
        self.assertEqual(pos, [0.0, (0.25 + 0.5) / 2, (0.25 + 0.5) / 2 + (0.75 + 1.0) / 2])

    def test_an_eased_ramp_by_hand(self):
        # the same ramp eased at both ends reads 15.625, 50, 84.375 and 100 percent
        pos = positions_of("0@0e;100@2e", 2)
        self.assertEqual(pos, [0.0, 0.328125, 1.25])

    def test_a_hold_key_keeps_its_speed_until_the_next_key(self):
        # 100% until the key at frame 4, which is read from its own tick on: the last half step of frame 4 is 200%
        self.assertEqual(positions_of("100@0h;200@4", 6), [0.0, 1.0, 2.0, 3.0, 4.5, 6.5, 8.5])

    def test_the_effect_reads_the_curve_through_the_fixed_point_percent(self):
        self.assertEqual(tool.fixed(100.0), 6553600)
        self.assertEqual(tool.fixed(33.3), 2182349)
        self.assertEqual(tool.fixed(-33.3), -2182349)

    def test_the_planned_curve(self):
        pos = positions_of(PLAN, 664)
        self.assertAlmostEqual(pos[120], 120.0, places=4)
        self.assertAlmostEqual(pos[664], 540.4, places=1)
        self.assertEqual(pos, sorted(pos))


class Analysis(unittest.TestCase):
    def analyse(self, text, frames, clip=1000, fps=Fraction(24), start=0.0, marks=()):
        step = tool.ticks_per_frame(fps)
        return tool.analyse(keys_of(text, step), fps, clip, frames, start, list(marks))

    def test_a_source_moment_shows_where_the_half_speed_puts_it(self):
        a = self.analyse("50@0;50@48", 48, marks=[("m", 0.5)])
        (name, seconds, target, found), = a["hits"]
        self.assertEqual((name, target), ("m", 12.0))
        self.assertEqual(found, [24.0])

    def test_marks_are_recording_times_after_start(self):
        a = self.analyse("100@0;100@48", 48, start=10.0, marks=[("m", 11.0)])
        self.assertEqual(a["hits"][0][2:], (24.0, [24.0]))

    def test_a_reverse_passes_a_moment_twice(self):
        a = self.analyse("100@0;-100@24", 24, marks=[("m", 3 / 24)])
        found = a["hits"][0][3]
        self.assertEqual(len(found), 2)
        self.assertLess(found[0], 12)
        self.assertGreater(found[1], 12)

    def test_a_mark_that_is_never_reached(self):
        a = self.analyse("100@0;100@10", 10, marks=[("far", 5.0)])
        self.assertEqual(a["hits"][0][3], [])

    def test_positions_outside_the_clip(self):
        a = self.analyse("200@0;200@100", 100, clip=150)
        self.assertEqual(a["outside"], [[75, 99]])
        code, out, _ = run_main("--keys", "200@0;200@100", "--fps", "24", "--clip-frames", "150")
        self.assertEqual(code, 3)
        self.assertIn("output frames 75 to 99 read source frames outside the clip", out)

    def test_only_frames_off_the_grid_need_interpolation(self):
        self.assertEqual(self.analyse("100@0;100@48", 48)["interpolated"], 0)
        self.assertEqual(self.analyse("50@0;50@48", 48)["interpolated"], 24)

    def test_a_plain_key_that_bends_the_speed_is_reported(self):
        plain = self.analyse("100@0;40@36;40@100", 100)
        self.assertEqual([k.frame for k, _, _ in plain["bends"]], [36])
        eased = self.analyse("100@0e;40@36e;40@100e", 100)
        self.assertEqual(eased["bends"], [])
        gentle = self.analyse("100@0;90@36;90@100", 100)
        self.assertEqual(gentle["bends"], [])

    def test_the_biggest_step_and_the_constant_end(self):
        a = self.analyse("100@0;50@10;50@40", 40)
        change, where = a["biggest"]
        self.assertAlmostEqual(change, 5.0)
        # frames 10 to 39 sit at 50%
        self.assertEqual(a["tail"], 30)


class Tail(unittest.TestCase):
    def test_the_planned_curve_ends_off_the_grid_and_one_move_fixes_it(self):
        keys = keys_of(PLAN)
        lock = tool.lock_tail(keys, 100)
        self.assertEqual(lock.first, 4)
        self.assertGreater(lock.base, 0.3)
        self.assertEqual(lock.shift, 67)
        self.assertLessEqual(lock.error, tool.LOCK_ERROR)
        moved = tool.moved_keys(keys, lock.first, lock.shift, 100)
        self.assertEqual(tool.key_text(moved[4]), "40@400.67e")
        self.assertEqual([tool.key_text(k) for k in moved[:4]], [tool.key_text(k) for k in keys[:4]])
        pos = tool.positions(tool.Curve(moved), 100, 665)
        for n in (500, 600, 665):
            self.assertAlmostEqual(pos[n], round(pos[n]), delta=tool.LOCK_ERROR)

    def test_a_tail_that_is_already_on_the_grid_is_left_alone(self):
        # two equal eased ramps and a plateau between them gain exactly 24 + 24 whole frames
        lock = tool.lock_tail(keys_of("100@0e;100@24e;200@48e;200@72e;100@96e;100@120e"), 100)
        self.assertEqual(lock.shift, 0)
        self.assertLessEqual(lock.base, tool.EXACT)
        code, out, _ = run_main("--keys", "100@0e;100@24e;200@48e;200@72e;100@96e;100@120e", "--fps", "24",
                                "--clip-frames", "400")
        self.assertIn("the final 100% run shows whole source frames", out)

    def test_curves_without_a_whole_frame_tail_have_nothing_to_fix(self):
        self.assertIsNone(tool.lock_tail(keys_of("100@0;50@24;50@48"), 100))
        self.assertIsNone(tool.lock_tail(keys_of("100@0;100@48"), 100))

    def test_a_curve_that_starts_slow_can_be_moved_into_place(self):
        keys = keys_of("0@0e;100@5e;100@20e")
        lock = tool.lock_tail(keys, 100)
        self.assertEqual(lock.first, 0)
        self.assertGreater(lock.base, tool.EXACT)
        self.assertIsNotNone(lock.shift)
        self.assertLessEqual(lock.error, tool.LOCK_ERROR)


class Command(unittest.TestCase):
    def test_the_planned_curve_reports_the_landmarks(self):
        code, out, _ = run_main("--keys", PLAN, "--fps", "60", "--clip-frames", "630", "--start", "13",
                                "--mark", "prompt=16.4", "--mark", "landing=18.4")
        self.assertEqual(code, 0, out)
        self.assertIn("output  664 frames, 11.07 s", out)
        self.assertIn("prompt: source 16.40 s is clip frame 204.0, shows at output frame 186.8 (3.11 s)", out)
        self.assertIn("landing: source 18.40 s is clip frame 324.0, shows at output frame 447.6 (7.46 s)", out)
        self.assertIn("541 of 664 (81%) need interpolation", out)
        self.assertIn('--keys "100@0e;100@120e;150@168e;40@204e;40@400.67e;100@454.67e;100@664.67e"', out)
        self.assertIn("warnings: none", out)

    def test_the_fixed_curve_needs_nothing_more(self):
        code, out, _ = run_main("--keys", "100@0e;100@120e;150@168e;40@204e;40@400.67e;100@454.67e;100@664.67e",
                                "--frames", "665", "--fps", "60", "--clip-frames", "630")
        self.assertEqual(code, 0, out)
        self.assertIn("the final 100% run shows whole source frames", out)
        self.assertIn("328 of 665 (49%) need interpolation", out)

    def test_bad_input_exits_with_one(self):
        code, _, err = run_main("--keys", "100@0he;50@10", "--fps", "60", "--clip-frames", "100")
        self.assertEqual(code, 1)
        self.assertIn("[error] --keys:", err)
        code, _, err = run_main("--keys", "100@0", "--fps", "60", "--clip-frames", "100")
        self.assertEqual(code, 1)
        self.assertIn("give --frames", err)

    def test_the_output_is_plain_ascii(self):
        _, out, _ = run_main("--keys", PLAN, "--fps", "60", "--clip-frames", "630")
        out.encode("ascii")


if __name__ == "__main__":
    unittest.main()
