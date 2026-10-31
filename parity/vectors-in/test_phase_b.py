#!/usr/bin/env python3
"""Unit tests of phase_b.py that need no server, no vfx and no fx-parity: python3 -m unittest test_phase_b"""
import json
import struct
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import phase_b as pb


def f32(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def words(*bits):
    return b"".join(struct.pack("<I", b) for b in bits)


class DiffValues(unittest.TestCase):
    def test_identical_and_bytes(self):
        r = pb.diff_values(8, b"\x01\x02", b"\x01\x02")
        self.assertEqual((r["differing"], r["max_ulp"]), (0, None))
        r = pb.diff_values(8, b"\x01\x02\x03", b"\x01\x03\x03")
        self.assertEqual((r["differing"], r["first_index"], r["max_ulp"]), (1, 1, None))

    def test_16_bit(self):
        a = struct.pack("<3H", 1, 2, 3)
        b = struct.pack("<3H", 1, 9, 4)
        r = pb.diff_values(16, a, b)
        self.assertEqual((r["values"], r["differing"], r["first_index"]), (3, 2, 1))

    def test_float_ulp_signed_zero_and_nan(self):
        one = f32(1.0)
        r = pb.diff_values(32, words(one, 0), words(one + 3, 0))
        self.assertEqual((r["differing"], r["max_ulp"], r["nan_mismatch"]), (1, 3, 0))
        r = pb.diff_values(32, words(0x80000000), words(0))
        self.assertEqual((r["differing"], r["max_ulp"]), (1, 0), "-0 against 0 differs, ULP distance 0")
        nan = 0x7FC00000
        r = pb.diff_values(32, words(nan), words(nan | 1))
        self.assertEqual((r["differing"], r["nan_mismatch"], r["max_ulp"]), (1, 1, 0))
        r = pb.diff_values(32, words(nan), words(nan))
        self.assertEqual(r["differing"], 0)

    def test_size_mismatch(self):
        self.assertTrue(pb.diff_values(8, b"abcd", b"abc")["size_mismatch"])

    def test_locate(self):
        self.assertEqual(pb.locate(8, 2, 2, 5), {"frame": 0, "x": 1, "y": 0, "channel": "g"})
        self.assertEqual(pb.locate(32, 2, 2, 16 + 4 + 3), {"frame": 1, "x": 1, "y": 1, "channel": "b"})


class Contract(unittest.TestCase):
    cases = [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}]
    base = {"a": ["a", "0", "1", "x", "1", ""], "b": ["b", "0", "1", "x", "1", ""], "c": ["c", "3", "0", "x", "1", "refused"], "d": ["d", "0", "1", "x", "1", ""]}

    def test_phase_a_split(self):
        moved, back, words_ = pb.judge(self.cases, {}, self.base, None)
        self.assertEqual([m["id"] for m in moved], ["c"])
        self.assertEqual(back, [])

    def test_harness_status_rejudges(self):
        st = {"a": {"status": "refused", "error": "x"}, "b": {"status": "error", "error": "y", "emulator_fault": True},
              "c": {"status": "rendered"}, "d": {"status": "timeout"}}
        moved, back, words_ = pb.judge(self.cases, {}, self.base, st)
        self.assertEqual(sorted(m["id"] for m in moved), ["a", "b", "d"])
        self.assertEqual(back, ["c"], "the original renders what the rewrite refuses: a real comparison")
        self.assertIn("emulator fault", [m for m in moved if m["id"] == "b"][0]["reason"])

    def test_not_run_keeps_the_phase_a_rule(self):
        st = {"a": {"status": "not_run"}, "c": {"status": "not_run"}}
        moved, back, _ = pb.judge(self.cases, {}, self.base, st)
        self.assertEqual([m["id"] for m in moved], ["c"])

    def test_unknown_rewrite_rows_are_not_refusals(self):
        moved, _, _ = pb.judge([{"id": "probe"}], {}, {}, None)
        self.assertEqual(moved, [])

    def test_load_status_words(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "s.json"
            p.write_text(json.dumps({"cases": [{"id": "a", "status": "unsupported"}, {"id": "b", "status": "rendered"}, {"id": "c", "status": "error", "pc": "0x1"}]}))
            st = pb.load_status(p)
            self.assertEqual((st["a"]["status"], st["b"]["status"], st["c"]["status"], st["c"]["emulator_fault"]), ("refused", "rendered", "error", True))
            p.write_text(json.dumps({"a": "timeout"}))
            self.assertEqual(pb.load_status(p)["a"]["status"], "timeout")


class Kills(unittest.TestCase):
    def rows(self, d):
        return {k: [k, str(v[0]), "1", v[1], "1", ""] for k, v in d.items()}

    def test_kill_rule(self):
        ref = {"a": "R", "b": "R", "c": "R", "d": "R"}
        base = self.rows({"a": (0, "R"), "b": (0, "R"), "c": (0, "X"), "d": (0, "R")})   # c: base already differs from the reference
        m1 = self.rows({"a": (0, "R"), "b": (0, "Z"), "c": (0, "R"), "d": (0, "R")})      # changes b: killed by b
        m2 = self.rows({"a": (0, "R"), "b": (0, "R"), "c": (0, "Q"), "d": (0, "R")})      # changes only c: cannot discriminate
        m3 = self.rows({"a": (0, "R"), "b": (0, "R"), "c": (0, "X"), "d": (3, "")})       # fails on d: counts as a difference
        m4 = self.rows({k: (0, v) for k, v in {"a": "R", "b": "R", "c": "X", "d": "R"}.items()})   # identical to base
        res = pb.compute_kills(ref, base, {"m1": m1, "m2": m2, "m3": m3, "m4": m4, "m5": None}, ["a", "b", "c", "d"])
        st = {m["id"]: m["status"] for m in res["mutants"]}
        self.assertEqual(st, {"m1": "killed", "m2": "survivor", "m3": "killed", "m4": "survivor", "m5": "not_run"})
        self.assertEqual(res["discriminating"], ["a", "b", "d"])
        self.assertEqual(res["base_differs_from_reference"], ["c"])
        m = {x["id"]: x for x in res["mutants"]}
        self.assertEqual(m["m1"]["killers"], ["b"])
        self.assertEqual(m["m3"]["failed_cases"], ["d"])
        self.assertEqual(m["m2"]["changed_outside_D"], 1)

    def test_missing_reference_is_not_discriminating(self):
        base = self.rows({"a": (0, "R"), "b": (0, "R")})
        m1 = self.rows({"a": (0, "Z"), "b": (0, "R")})
        res = pb.compute_kills({"b": "R"}, base, {"m1": m1}, ["a", "b"])
        self.assertEqual(res["no_reference"], ["a"])
        self.assertEqual(res["mutants"][0]["status"], "survivor")

    def test_no_references_at_all(self):
        res = pb.compute_kills({}, self.rows({"a": (0, "R")}), {"m1": self.rows({"a": (0, "Z")})}, ["a"])
        self.assertEqual(res["mutants"][0]["status"], "not_judged")


def rewrite_lines(reads_by_frame, scale=3000, step=100, frames=None):
    out = [{"ev": "run", "effect": "vectors-in", "fps": "30", "time_scale": scale, "time_step": step, "frames": len(reads_by_frame),
            "params": ["speed_percent=50"]}]
    for n, reads in enumerate(reads_by_frame):
        out.append({"ev": "frame", "out_frame": n, "time": n * step, "time_scale": scale, "time_step": step})
        for layer, t, idx in reads:
            out.append({"ev": "read", "out_frame": n, "layer": layer, "time": t, "index": idx, "clip_frames": 4})
    return out


LAYERS = {"input": 0, "vectors_prev": 1, "vectors_next": 2, "fg1_matte": 3}


class CallRecords(unittest.TestCase):
    layer_map = {0: "input", 1: "vectors_prev", 2: "vectors_next", 3: "fg1_matte"}

    def setUp(self):
        self.rew_lines = rewrite_lines([
            [("input", 0, 0), ("vectors_prev", 0, 0), ("vectors_next", 100, 1)],
            [("input", 100, 1), ("vectors_prev", 100, 1), ("vectors_next", 5000, None)],   # next read past the end of a short clip
            [("input", 200, 2), ("vectors_prev", 233, 2)],                                    # by time, below the clip rate
        ])
        self.rew = pb.parse_rewrite_record(self.rew_lines)

    def plugin(self, style="smart", mutate=None):
        lines = pb.synth_plugin_lines(self.rew_lines, LAYERS, style=style)
        if mutate:
            lines = mutate(lines)
        return pb.parse_plugin_record(lines, self.layer_map)

    def cmp(self, plug, **kw):
        return pb.compare_records(plug, self.rew, **kw)

    def test_match_smart_and_classic(self):
        for style in ("smart", "classic"):
            r = self.cmp(self.plugin(style))
            self.assertTrue(r["match"], (style, r))
            self.assertEqual(r["frames_compared"], 3)

    def test_value_parameter_checkouts_are_not_layer_reads(self):
        plug = self.plugin("classic")
        self.assertEqual(len(plug["param_reads"]), 3)
        self.assertTrue(all(r["layer"] in LAYERS for rs in plug["reads"].values() for r in rs))

    def test_missing_read(self):
        def drop(lines):
            i = [j for j, d in enumerate(lines) if d.get("param_index") == 2 and d.get("fn") == "prerender_cb.checkout_layer"][-1]
            return [d for j, d in enumerate(lines) if j != i]
        r = self.cmp(self.plugin(mutate=drop))
        self.assertFalse(r["match"])
        self.assertEqual(r["first_mismatch"]["out_frame"], 1)
        self.assertIn("missing_in_plugin", r["first_mismatch"])
        self.assertNotIn("extra_in_plugin", r["first_mismatch"])

    def test_extra_read(self):
        def add(lines):
            d = dict([x for x in lines if x.get("fn") == "prerender_cb.checkout_layer"][0], time=700, footage_frame=7)
            lines.insert(3, d)
            return lines
        r = self.cmp(self.plugin(mutate=add))
        self.assertEqual(r["first_mismatch"]["out_frame"], 0)
        self.assertIn("extra_in_plugin", r["first_mismatch"])

    def test_wrong_time_and_wrong_layer_are_hinted(self):
        def late(lines):
            i = [j for j, d in enumerate(lines) if d.get("fn") == "prerender_cb.checkout_layer" and d["out_frame"] == 2][-1]
            lines[i] = dict(lines[i], time=lines[i]["time"] + 100)
            return lines
        r = self.cmp(self.plugin(mutate=late))
        self.assertEqual(r["first_mismatch"]["out_frame"], 2)
        self.assertTrue(any("wrong time" in h for h in r["first_mismatch"]["hints"]))

        def other(lines):
            i = [j for j, d in enumerate(lines) if d.get("fn") == "prerender_cb.checkout_layer" and d["out_frame"] == 0][-1]
            lines[i] = dict(lines[i], param_index=3)
            return lines
        r = self.cmp(self.plugin(mutate=other))
        self.assertEqual(r["first_mismatch"]["out_frame"], 0)
        self.assertTrue(any("wrong layer" in h for h in r["first_mismatch"]["hints"]))

    def test_footage_frame_disagreement(self):
        def off(lines):
            i = [j for j, d in enumerate(lines) if d.get("fn") == "prerender_cb.checkout_layer" and d["out_frame"] == 2][-1]
            lines[i] = dict(lines[i], footage_frame=3)
            return lines
        r = self.cmp(self.plugin(mutate=off))
        self.assertIn("index_differs", r["first_mismatch"])

    def test_order_only_matters_when_asked(self):
        def swap(lines):
            idx = [j for j, d in enumerate(lines) if d.get("fn") == "prerender_cb.checkout_layer" and d["out_frame"] == 0]
            lines[idx[0]], lines[idx[1]] = lines[idx[1]], lines[idx[0]]
            return lines
        plug = self.plugin(mutate=swap)
        self.assertTrue(self.cmp(plug)["match"])
        r = self.cmp(plug, ordered=True)
        self.assertFalse(r["match"])
        self.assertIn("order_differs", r["first_mismatch"])

    def test_past_the_end_read_compares_by_time(self):
        plug = self.plugin()
        outside = [r for r in plug["reads"][1] if r["layer"] == "vectors_next"][0]
        self.assertIs(outside["inside"], False)
        self.assertTrue(self.cmp(plug)["match"])
        self.assertTrue(self.cmp(plug, ignore_outside=True)["match"])

    def test_exclude_layers(self):
        def no_input(lines):
            return [d for d in lines if not (d.get("param_index") == 0 and d.get("fn") == "prerender_cb.checkout_layer")]
        plug = self.plugin(mutate=no_input)
        self.assertFalse(self.cmp(plug)["match"])
        self.assertTrue(self.cmp(plug, exclude_layers=["input"])["match"])

    def test_missing_frame_and_run_params(self):
        plug = self.plugin()
        del plug["frames"][2]
        self.assertTrue(self.cmp(plug)["first_mismatch"]["frame_missing_in_plugin"])
        r = pb.compare_records(self.plugin(), self.rew, case_params_list=["speed_percent=75"])
        self.assertFalse(r["match"])
        r = pb.compare_records(self.plugin(), self.rew, case_params_list=["speed_percent=50"])
        self.assertTrue(r["match"])

    def test_incomplete_plugin_record(self):
        plug = pb.parse_plugin_record([{"ev": "cmd", "cmd": "GLOBAL_SETUP"}], self.layer_map)
        self.assertIsNotNone(plug["incomplete"])

    def test_layer_map_from_param_map(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "param_map.tsv"
            p.write_text("plugin\teffect\tae_index\tvfx_kind\tvfx_id\nTwixtorProVectorsIn_64\tvectors-in\t4\tlayer\tvectors_prev\n"
                         "TwixtorProVectorsIn_64\tvectors-in\t5\tpopup\tdisplay\nOther\tx\t6\tlayer\tfoo\n")
            self.assertEqual(pb.load_layer_map(p), {0: "input", 4: "vectors_prev"})
            self.assertEqual(pb.load_layer_map(Path(t) / "none.tsv"), {0: "input"})

    def test_rewrite_times_are_exact_fractions(self):
        rew = pb.parse_rewrite_record(rewrite_lines([[("input", 100, 0)]], scale=30000, step=1001))
        self.assertEqual(rew["reads"][0][0]["secs"], Fraction(100, 30000))


class SummaryAndRule(unittest.TestCase):
    def grade(self, n, differ=0, floats=0):
        rows = []
        for i in range(n):
            rows.append({"id": "c%d" % i, "status": "FAIL" if i < differ else "pass", "logic_parity": i % 10 == 0, "depth": 8})
        return {"cases": {}, "values": {"bytes_compared": n * 100, "floats_compared": floats, "differing_values": differ, "max_ulp": 0 if floats else None},
                "cases_detail": rows, "ooc_sha": "phase_a"}

    def build(self, n, differ=0, calls=(160, 160), mut=(24, 24), blockers=()):
        ooc = {"moved_out": [], "candidates": [{"id": "x"}] * 3}
        kills = {"total": mut[1], "killed": mut[0], "survivors": ["m%02d" % i for i in range(mut[1] - mut[0])]}
        c = {"compared": calls[0], "matching": calls[1]}
        meta = {"c0": {"stand_in": "keyframe_interpolation"}}
        return pb.build_summary(self.grade(n, differ), ooc, kills, c, [], "sha", "2026-01-01", meta, list(blockers), ["cmd"])

    def test_verified_only_when_everything_holds(self):
        s, unmet = self.build(200)
        self.assertEqual((s["status"], unmet), ("verified", []))
        self.assertEqual(s["cases"], {"in_contract": 200, "compared": 200, "identical": 200, "differing": 0, "out_of_contract": 3, "logic_parity": 20})
        self.assertEqual(set(s), {"module", "plugin", "plugin_sha256", "effect", "status", "blockers", "cases", "values", "call_records", "mutants",
                                  "stand_ins", "fixes", "commands", "measured_at"})

    def test_each_unmet_item(self):
        self.assertIn("fewer than 150 in-contract cases compared", self.build(149)[1])
        self.assertIn("a case or value differs", self.build(200, differ=1)[1])
        self.assertIn("call records not all matching", self.build(200, calls=(160, 159))[1])
        self.assertIn("call records not all matching", self.build(200, calls=(0, 0))[1])
        s, unmet = self.build(200, mut=(23, 24))
        self.assertIn("a mutant survives or none ran", unmet)
        self.assertEqual(s["mutants"]["survivors"], ["m00"])
        self.assertEqual(s["status"], "in_progress")

    def test_blocker_makes_blocked(self):
        s, _ = self.build(10, blockers=["emulator side not run yet"])
        self.assertEqual((s["status"], s["blockers"]), ("blocked", ["emulator side not run yet"]))

    def test_no_kills_file_means_zero_killed(self):
        g = self.grade(5)
        s, unmet = pb.build_summary(g, {"moved_out": [], "candidates": []}, None, None, [], "sha", "d", {}, [], [])
        self.assertEqual((s["mutants"]["killed"], s["mutants"]["total"], s["call_records"]), (0, 0, {"compared": 0, "matching": 0}))
        self.assertEqual(s["status"], "in_progress")

    def test_summary_refuses_without_grade(self):
        import argparse
        with tempfile.TemporaryDirectory() as t:
            a = argparse.Namespace(out_dir=t, testset=t, date=None, blocker=[], allow_stale=False, out=None)
            with self.assertRaises(SystemExit):
                pb.cmd_summary(a)


class Triage(unittest.TestCase):
    def setUp(self):
        mk = lambda i, w, depth, params, layers=("vectors_prev", "vectors_next"): {
            "id": "c%d" % i, "width": w, "height": w, "depth": depth, "frames": 4, "fps": "30", "params": params,
            "input": "inputs/in%d" % i, "reference": "c%d.x" % i, "effect": "vectors-in", "source": "tex", "kind": "temporal",
            "layers": {l: "inputs/%s%d" % (l, i) for l in layers}}
        self.cases = [mk(0, 16, 8, ["vectors_max_displace=32", "speed_percent=50"]), mk(1, 16, 8, ["vectors_max_displace=32", "speed_percent=50", "warping=2"]),
                      mk(2, 8, 32, ["vectors_max_displace=32", "speed_percent=50", "warping=2", "blur_compensation=5"]),
                      mk(3, 16, 16, ["vectors_max_displace=32", "speed_percent=50"])]
        self.by_id = {c["id"]: c for c in self.cases}
        st = ["pass", "FAIL", "FAIL", "pass"]
        self.rows = [{"id": c["id"], "status": s, "group": "g", "tags": [], "logic_parity": False} for c, s in zip(self.cases, st)]

    def test_report_counts_per_value(self):
        rep = pb.triage_report(self.rows, self.by_id, {})
        settings = {k: (b, t) for k, b, t in rep["by_setting"]}
        self.assertEqual(settings["warping=2"], (2, 2))
        self.assertEqual(settings["blur_compensation=5"], (1, 1))
        self.assertEqual(rep["by_setting"][0][1], rep["by_setting"][0][2], "a setting that always co-occurs ranks first")

    def test_smallest_failing_first(self):
        small = pb.smallest_failing(self.rows, self.by_id)
        self.assertEqual([r["id"] for r in small], ["c2", "c1"])

    def test_ladder(self):
        with tempfile.TemporaryDirectory() as t:
            ts = Path(t) / "testsets" / "grid"
            ts.mkdir(parents=True)
            rungs, cases = pb.build_ladder(self.by_id["c2"], self.cases, {}, Path(t) / "probe", ts)
            names = [r["rung"] for r in rungs]
            self.assertIn("self", names)
            self.assertIn("base-params", names)
            self.assertIn("plugin-defaults", names)
            self.assertIn("rev-warping", names)
            self.assertIn("rev-blur_compensation", names)
            self.assertIn("only-warping", names)
            self.assertIn("only-blur_compensation", names)
            self.assertIn("layers-prev", names)
            by = {r["rung"]: c for r, c in zip(rungs, cases)}
            self.assertEqual(sorted(by["rev-warping"]["params"]), sorted(["vectors_max_displace=32", "speed_percent=50", "blur_compensation=5"]))
            self.assertEqual(sorted(by["only-blur_compensation"]["params"]), sorted(["vectors_max_displace=32", "speed_percent=50", "blur_compensation=5"]))
            self.assertEqual(by["plugin-defaults"]["params"], [])
            self.assertEqual(len({c["id"] for c in cases}), len(cases), "ids are unique")
            self.assertTrue(by["self"]["input"].startswith("../../testsets/grid/") or "testsets/grid" in by["self"]["input"])
            self.assertEqual(list(by["layers-prev"]["layers"]), ["vectors_prev"])
            self.assertTrue(all(c["reference"].startswith(c["id"] + ".") for c in cases))


class Locate(unittest.TestCase):
    def test_grade_rows_cross_check_and_aggregate(self):
        cases = [{"id": "a", "depth": 8, "width": 1, "height": 1, "frames": 1, "fps": "30", "params": []},
                 {"id": "b", "depth": 32, "width": 1, "height": 1, "frames": 1, "fps": "30", "params": []},
                 {"id": "c", "depth": 8, "width": 1, "height": 1, "frames": 1, "fps": "30", "params": []},
                 {"id": "d", "depth": 8, "width": 1, "height": 1, "frames": 1, "fps": "30", "params": []}]
        meta = {"b": {"group": "keyed", "flags": [], "stand_in": "keyframe_interpolation"}}
        fx = {"a": {"status": "pass", "clip_differing_pixels": 0}, "b": {"status": "FAIL", "clip_differing_pixels": 1, "failing_frames": 1, "frame": 0},
              "c": {"status": "no reference"}, "d": {"status": "pass", "clip_differing_pixels": 0}}
        scans = {"a": {"values": 4, "differing": 0, "max_ulp": None, "nan_mismatch": 0, "rewrite_failed": False},
                 "b": {"values": 4, "differing": 1, "max_ulp": 5, "nan_mismatch": 0, "rewrite_failed": False},
                 "d": {"values": 4, "differing": 2, "max_ulp": None, "nan_mismatch": 0, "rewrite_failed": False}}
        rows = pb.build_rows(cases, meta, fx, scans, set())
        by = {r["id"]: r for r in rows}
        self.assertEqual(by["c"]["status"], "no reference")
        self.assertTrue(by["b"]["logic_parity"])
        self.assertIn("crosscheck", by["d"], "fx-parity passes what the value scan says differs")
        agg = pb.aggregate(rows, {})
        self.assertEqual((agg["cases"]["compared"], agg["cases"]["identical"], agg["cases"]["differing"], agg["cases"]["no_reference"]), (3, 2, 1, 1))
        self.assertEqual((agg["values"]["floats_compared"], agg["values"]["bytes_compared"], agg["values"]["max_ulp"]), (4, 8, 5))
        rows2 = pb.build_rows(cases, meta, fx, scans, {"a"})
        self.assertEqual({r["id"]: r["status"] for r in rows2}["a"], "out of contract")


if __name__ == "__main__":
    unittest.main()
