#!/usr/bin/env python3
"""Unit tests of callrec.py (the `reads` rule of `phase_b.py calls`): python3 -m unittest test_callrec"""
import unittest
from fractions import Fraction

import callrec

NAMES = {0: "input", 11: "source", 13: "vectors_prev", 14: "vectors_next"}


def plug_lines(extra_pix=(), registered=()):
    """One output frame: probes of vectors_prev at 0 and the source at 80/2400 s (pixels read), plus registrations
    (pre-render checkouts the plugin never reads) for the layers in `registered`."""
    lines = [
        {"ev": "default_layer", "param_index": 11, "layer": "input"},
        {"ev": "frame", "out_frame": 0, "time": 0, "time_step": 100, "time_scale": 2400},
        {"ev": "cb", "fn": "prerender_cb.checkout_layer", "checkout_id": 0, "param_index": 13, "time": 0, "time_scale": 2400,
         "footage_frame": 0, "inside": True},
        {"ev": "cb", "fn": "prerender_cb.checkout_layer", "checkout_id": 1, "param_index": 11, "time": 80, "time_scale": 2400,
         "footage_frame": 0, "inside": True},
        {"ev": "cb", "fn": "smart_cb.checkout_layer_pixels", "checkout_id": 1},
    ]
    nid = 2
    for (pidx, t) in registered:
        lines.append({"ev": "cb", "fn": "prerender_cb.checkout_layer", "checkout_id": nid, "param_index": pidx, "time": t,
                      "time_scale": 2400, "footage_frame": 0, "inside": True})
        nid += 1
    for k in extra_pix:
        lines.append({"ev": "cb", "fn": "smart_cb.checkout_layer_pixels", "checkout_id": k})
    lines.append({"ev": "case_done", "status": "rendered"})
    return lines


def rew_lines(reads):
    out = [{"ev": "run", "time_scale": 3000}, {"ev": "frame", "out_frame": 0, "time": 0, "time_scale": 3000}]
    for layer, t, idx in reads:
        out.append({"ev": "read", "out_frame": 0, "layer": layer, "time": t, "index": idx, "clip_frames": 4})
    return out


class ReadsRule(unittest.TestCase):
    def test_registrations_are_not_reads(self):
        # the plugin registers vectors_next at 0 and 1/30 s and the source at 180/2400 s but never reads them
        p = plug_lines(registered=[(14, 0), (14, 80), (11, 180)])
        r = callrec.compare_case(p, rew_lines([("vectors_prev", 0, 0), ("input", 100, 0)]), NAMES)
        self.assertEqual(r["status"], "match", r)
        self.assertEqual(r["classes"], {"probe": 1, "same": 1})

    def test_a_pixel_read_the_rewrite_does_not_make_is_a_mismatch(self):
        p = plug_lines(registered=[(11, 180)], extra_pix=[2])
        r = callrec.compare_case(p, rew_lines([("vectors_prev", 0, 0), ("input", 100, 0)]), NAMES)
        self.assertEqual(r["status"], "mismatch")
        self.assertEqual(r["first_mismatch"]["items"][0]["kind"], "plugin_only")
        self.assertEqual(r["first_mismatch"]["items"][0]["read"], "input@3/40")

    def test_rewrite_only_read_and_index_and_time(self):
        p = plug_lines()
        r = callrec.compare_case(p, rew_lines([("vectors_prev", 0, 0), ("input", 100, 0), ("vectors_next", 40, 1)]), NAMES)
        self.assertEqual((r["status"], r["classes"]["rewrite_only"]), ("mismatch", 1))
        r = callrec.compare_case(p, rew_lines([("vectors_prev", 0, 0), ("input", 100, 1)]), NAMES)
        self.assertEqual((r["status"], r["classes"]["index"]), ("mismatch", 1))
        bad = rew_lines([("vectors_prev", 0, 0)])
        bad[1]["time"] = 1
        r = callrec.compare_case(p, bad, NAMES)
        self.assertEqual((r["status"], r["classes"]["frame_time"]), ("mismatch", 1))

    def test_unfinished_plugin_record(self):
        p = plug_lines()[:-1]
        self.assertEqual(callrec.compare_case(p, rew_lines([]), NAMES)["status"], "plugin_record_incomplete")

    def test_outside_the_clip_agrees_on_both_sides(self):
        self.assertTrue(callrec.index_agrees(None, 5, False, 4))
        self.assertTrue(callrec.index_agrees(None, 5, None, 4))
        self.assertFalse(callrec.index_agrees(None, 2, True, 4))
        self.assertFalse(callrec.index_agrees(1, 0, True, 4))

    def test_time_bases_compare_as_fractions(self):
        self.assertEqual(Fraction(100, 3000), Fraction(80, 2400))


if __name__ == "__main__":
    unittest.main()
