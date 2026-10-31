#!/usr/bin/env python3
"""Tests of phase_b.py that need no server, no vfx and no fx-parity: python3 -m unittest test_phase_b"""
import json
import os
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase_b as pb  # noqa: E402


def case(depth=8, w=2, h=1, frames=1, cid="c1", params=None):
    return {"id": cid, "effect": "retime-fgbg", "depth": depth, "width": w, "height": h, "frames": frames,
            "params": params or [], "input": "inputs/x", "reference": cid + ".rgba", "source": "pan", "fps": "24",
            "kind": "temporal", "label": "l"}


def f32(*bits):
    return struct.pack("<{}I".format(len(bits)), *bits)


class ValueCompare(unittest.TestCase):
    def test_equal_clips_have_no_difference(self):
        r = pb.compare_clip_bytes(case(8), bytes(range(8)), bytes(range(8)))
        self.assertEqual((r["differing_values"], r["max_ulp"], r["bytes_compared"]), (0, None, 8))

    def test_one_bit_at_8_bpc_is_one_value_with_its_place(self):
        a = bytes(range(8))
        b = bytearray(a)
        b[5] ^= 1
        r = pb.compare_clip_bytes(case(8), a, bytes(b))
        self.assertEqual(r["differing_values"], 1)
        self.assertEqual(r["first"][0]["channel"], "G")
        self.assertEqual((r["first"][0]["x"], r["first"][0]["frame"]), (1, 0))
        self.assertIsNone(r["max_ulp"])

    def test_16_bpc_counts_values_not_bytes(self):
        a = struct.pack("<8H", *range(8))
        b = struct.pack("<8H", *[0, 1, 2, 3, 4, 5, 6, 7 ^ 0x100])
        r = pb.compare_clip_bytes(case(16), a, b)
        self.assertEqual((r["differing_values"], r["bytes_compared"]), (1, 16))
        self.assertEqual(r["first"][0]["channel"], "A")

    def test_float_planar_places(self):
        one = 0x3F800000
        a = f32(*([one] * 8))
        b = f32(*([one] * 5 + [one + 3] + [one] * 2))
        r = pb.compare_clip_bytes(case(32), a, b)
        self.assertEqual(r["max_ulp"], 3)
        self.assertEqual(r["first"][0]["channel"], "R")      # plane 2 of G, B, R, A (2 pixels per plane)
        self.assertEqual(r["floats_compared"], 8)

    def test_negative_zero_against_zero_differs_with_ulp_zero(self):
        r = pb.compare_clip_bytes(case(32, 1, 1), f32(0x80000000, 0, 0, 0), f32(0, 0, 0, 0))
        self.assertEqual((r["differing_values"], r["max_ulp"], r["nan_mismatch"]), (1, 0, 0))

    def test_nan_payload_change_is_a_nan_mismatch(self):
        r = pb.compare_clip_bytes(case(32, 1, 1), f32(0x7FC00000, 0, 0, 0), f32(0x7FC00001, 0, 0, 0))
        self.assertEqual((r["differing_values"], r["nan_mismatch"], r["max_ulp"]), (1, 1, 0))

    def test_equal_nan_bits_match(self):
        r = pb.compare_clip_bytes(case(32, 1, 1), f32(0x7FC00001, 0, 0, 0), f32(0x7FC00001, 0, 0, 0))
        self.assertEqual(r["differing_values"], 0)

    def test_ulp_across_zero_and_signs(self):
        self.assertEqual(pb.ulp_between(1, 0x80000001), 2)
        self.assertEqual(pb.ulp_between(0x80000000, 0), 0)
        self.assertEqual(pb.ulp_between(0x3F800000, 0x3F800001), 1)
        self.assertIsNone(pb.ulp_between(0x7FC00000, 0))

    def test_size_mismatch_counts_missing_values(self):
        r = pb.compare_clip_bytes(case(8), bytes(8), bytes(6))
        self.assertTrue(r["size_mismatch"])
        self.assertEqual(r["differing_values"], 2)

    def test_second_frame_is_found(self):
        c = case(8, 1, 1, frames=2)
        r = pb.compare_clip_bytes(c, bytes(8), bytes([0] * 4 + [0, 9, 0, 0]))
        self.assertEqual((r["first_frame"], r["differing_frames"]), (1, 1))


class Grade(unittest.TestCase):
    def test_merge_and_totals(self):
        cases = [case(8, cid="a"), case(32, 1, 1, cid="b"), case(8, cid="c"), case(8, cid="d")]
        grid = {"a": ("size", set()), "b": ("keyed", {"standin"}), "c": ("size", set()), "d": ("size", set())}
        report = {"cases": [{"id": "a", "status": "pass"}, {"id": "b", "status": "FAIL", "max_ulp": 0, "frame": 0},
                            {"id": "c", "status": "no reference"}, {"id": "d", "status": "pass"}]}
        audits = {"a": pb.compare_clip_bytes(cases[0], bytes(8), bytes(8)),
                  "b": pb.compare_clip_bytes(cases[1], f32(0x80000000, 0, 0, 0), f32(0, 0, 0, 0)),
                  "d": pb.compare_clip_bytes(cases[3], bytes(8), bytes([0] * 7 + [1]))}
        rows, t = pb.build_grade(cases, grid, report, audits, {})
        st = {r["id"]: r["status"] for r in rows}
        self.assertEqual(st, {"a": "pass", "b": "FAIL", "c": "no reference", "d": "FAIL"})
        self.assertEqual(t["audit_disagreements"], ["d"])            # fx-parity said pass, the audit found a byte
        self.assertEqual((t["pass"], t["fail"], t["no_reference"], t["compared"]), (1, 2, 1, 3))
        self.assertEqual(t["logic_parity_compared"], 1)
        self.assertEqual((t["bytes_compared"], t["floats_compared"], t["differing_values"], t["max_ulp"]), (16, 4, 2, 0))

    def test_no_reference_is_never_compared(self):
        rows, t = pb.build_grade([case(cid="a")], {}, {"cases": [{"id": "a", "status": "no reference"}]}, {}, {})
        self.assertEqual((t["compared"], t["pass"], t["fail"], t["no_reference"]), (0, 0, 0, 1))
        self.assertIsNone(t["max_ulp"])

    def test_stale_base_is_flagged(self):
        c = case(cid="a")
        au = pb.compare_clip_bytes(c, bytes(8), bytes(8))
        au["sha"] = "x"
        rows, t = pb.build_grade([c], {}, {"cases": [{"id": "a", "status": "pass"}]}, {"a": au}, {"a": {"sha": "y"}})
        self.assertEqual(t["base_tsv_stale"], ["a"])


class Kills(unittest.TestCase):
    base = {"a": {"exit": 0, "sha": "A", "bytes": 1, "error": ""}, "b": {"exit": 0, "sha": "B", "bytes": 1, "error": ""},
            "c": {"exit": 0, "sha": "C", "bytes": 1, "error": ""}}
    muts = [{"id": "M1"}, {"id": "M2"}, {"id": "M3"}, {"id": "M4"}]
    built = {"M1": True, "M2": True, "M3": True, "M4": False}

    def rows(self, **over):
        r = {k: dict(v) for k, v in self.base.items()}
        for k, sha in over.items():
            r[k]["sha"] = sha
        return {"rows": r}

    def test_definition(self):
        refs = {"a": "A", "b": "B-other", "c": None}          # b: base differs from the reference, c: no reference
        mres = {"M1": self.rows(a="A2"), "M2": self.rows(b="B2", c="C2"), "M3": self.rows(), "M4": None}
        out, info = pb.compute_kills(self.muts, self.built, self.base, refs, mres, set())
        st = {m["id"]: m["status"] for m in out}
        self.assertEqual(st, {"M1": "killed", "M2": "survivor", "M3": "survivor", "M4": "not_built"})
        self.assertEqual(out[0]["killing_cases"], ["a"])
        self.assertIn("none of them", out[1]["reason"])
        self.assertIn("equivalent", out[2]["reason"])
        self.assertEqual(info["discriminating_cases"], 1)

    def test_changed_list_form_and_excluded_cases(self):
        refs = {"a": "A", "b": "B", "c": "C"}
        mres = {"M1": {"changed": {"a"}}, "M2": {"changed": {"b"}}, "M3": {"changed": set()}, "M4": None}
        out, _ = pb.compute_kills(self.muts, self.built, self.base, refs, mres, {"a"})
        self.assertEqual([m["status"] for m in out], ["survivor", "killed", "survivor", "not_built"])

    def test_a_crash_of_the_mutant_kills(self):
        refs = {"a": "A", "b": "B", "c": "C"}
        r = self.rows()
        r["rows"]["c"]["exit"] = 3
        out, _ = pb.compute_kills(self.muts[:1], self.built, self.base, refs, {"M1": r}, set())
        self.assertEqual(out[0]["killing_cases"], ["c"])

    def test_grade_must_agree_with_the_hash_equality(self):
        refs = {"a": "A", "b": "B", "c": "C"}
        out, info = pb.compute_kills(self.muts[:1], self.built, self.base, refs, {"M1": {"changed": {"a", "b"}}}, set(), grade_pass={"a", "c"})
        self.assertEqual(out[0]["killing_cases"], ["a"])
        self.assertEqual(info["grade_disagreements"], [])


class OutOfContract(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(pb.classify_plugin({"status": "rendered"})[0], "rendered")
        self.assertEqual(pb.classify_plugin({"status": "refused"})[0], "refused")
        self.assertEqual(pb.classify_plugin({"status": "unsupported", "error": "x"})[0], "refused")
        self.assertEqual(pb.classify_plugin({"status": "error", "error": "PluginError: RENDER returned PF_Err 3"})[0], "plugin_error")
        self.assertEqual(pb.classify_plugin({"status": "error", "error": "GuestFault: Unhandled CPU exception"})[0], "harness_failure")
        self.assertEqual(pb.classify_plugin({"status": "timeout"})[0], "harness_failure")
        self.assertEqual(pb.classify_plugin({"status": "not_run", "error": "the plugin did not boot"})[0], "harness_failure")

    def test_moved_and_kept(self):
        cases = [case(cid=x) for x in "abcdef"]
        status = {"a": {"status": "rendered"}, "b": {"status": "refused"}, "c": {"status": "error", "error": "PluginError: x returned PF_Err 1"},
                  "d": {"status": "error", "error": "GuestFault: boom"}, "e": {"status": "timeout"}}
        cand = [{"id": "o1", "reason": "r", "vfx_verdict": "vfx: error: bad"}, {"id": "o2", "reason": "r", "vfx_verdict": "v"}]
        status["o2"] = {"status": "rendered"}
        d = pb.build_ooc(cases, cand, status, {}, None)
        self.assertEqual([m["id"] for m in d["moved"]], ["b", "c"])
        self.assertEqual([m["id"] for m in d["not_judged"]], ["d", "e"])
        self.assertEqual(d["counts"]["in_contract_after"], 4)
        self.assertEqual(d["counts"]["out_of_contract_total"], 4)
        self.assertEqual(d["candidates"][0]["plugin"]["status"], "not run")
        self.assertIn("note", d["candidates"][1])
        d2 = pb.build_ooc(cases, cand, status, {}, None, move_harness_failures=True)
        self.assertEqual(len(d2["moved"]), 4)

    def test_judged_cases_move_with_their_reason_and_only_with_the_file(self):
        cases = [case(cid=x) for x in "abcd"]
        status = {"a": {"status": "rendered"}, "b": {"status": "rendered"}, "c": {"status": "error", "error": "GuestFault: boom"}}
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "judged.tsv")
            with open(p, "w") as fh:
                fh.write("# a comment\nid\treason\tevidence\nb\tthe original leaves the word undefined\tnotes Q4\n"
                         "c\tthe original stops on it\treadfaults\nzz\tnot in the set\t-\n")
            judged = pb.load_judged([p])
            d = pb.build_ooc(cases, [], status, {}, None, judged=judged)
            self.assertEqual([m["id"] for m in d["moved"]], ["b", "c"])
            self.assertTrue(d["moved"][0]["reason"].endswith("the original leaves the word undefined"))
            self.assertEqual(d["moved"][0]["judgement"]["evidence"], "notes Q4")
            self.assertEqual(d["counts"]["judged"], 2)
            self.assertEqual(d["counts"]["not_judged"], 0)
            self.assertEqual(d["judged_unknown"], ["zz"])
            # without the file the same cases stay in contract: the judgement is reversible
            d0 = pb.build_ooc(cases, [], status, {}, None)
            self.assertEqual(d0["moved"], [])
            self.assertEqual([m["id"] for m in d0["not_judged"]], ["c"])
            # a line without a reason is refused
            with open(p, "w") as fh:
                fh.write("id\treason\tevidence\nb\t\tx\n")
            with self.assertRaises(SystemExit):
                pb.load_judged([p])

    def test_status_file_forms(self):
        with tempfile.TemporaryDirectory() as t:
            def w(name, text):
                p = os.path.join(t, name)
                open(p, "w").write(text)
                return p
            a = pb.load_status_file(w("a.json", json.dumps({"cases": [{"id": "x", "status": "rendered"}], "boots": []})))
            b = pb.load_status_file(w("b.json", json.dumps({"x": {"status": "refused"}})))
            c = pb.load_status_file(w("c.jsonl", '{"event":"boot","ok":true}\n{"event":"done","id":"x","status":"error"}\n'))
            d = pb.load_status_file(w("d.tsv", "id\tstatus\terror\nx\ttimeout\t\n"))
            self.assertEqual([a["x"]["status"], b["x"]["status"], c["x"]["status"], d["x"]["status"]], ["rendered", "refused", "error", "timeout"])

    def test_plugin_range_check(self):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "pm.tsv")
            open(p, "w").write("plugin\teffect\tae_index\tae_name\tae_type\tae_min\tae_max\tae_choices\tvfx_kind\tvfx_id\n"
                               "TwixtorPro_64\tretime-fgbg\t3\tSpeed\tFLOAT_SLIDER\t-1000\t1000\t\tfloat\tspeed_percent\n"
                               "TwixtorPro_64\tretime-fgbg\t4\tWarping\tPOPUP\t\t\tA|B|C\tpopup\twarping\n"
                               "TwixtorPro_64\tretime-fgbg\t\tgone\t\t\t\t\tfloat\tunreg\n")
            r = pb.plugin_ranges(p)
            self.assertEqual(sorted(r), ["speed_percent", "warping"])
            self.assertIn("outside", pb.out_of_range(case(params=["speed_percent=2000"]), r))
            self.assertIn("outside", pb.out_of_range(case(params=["speed_percent=100@0;-5000@4"]), r))
            self.assertIn("choices", pb.out_of_range(case(params=["warping=4"]), r))
            self.assertIsNone(pb.out_of_range(case(params=["speed_percent=-1000", "warping=3", "unreg=99"]), r))
        self.assertEqual(pb.plugin_ranges(os.path.join(HERE, "nope.tsv")), {})

    def test_value_numbers(self):
        self.assertEqual(pb.value_numbers("100@0;50@4"), [100.0, 50.0])
        self.assertEqual(pb.value_numbers("-0"), [-0.0])
        self.assertEqual(pb.value_numbers("abc"), [])


def rewrite(frames, reads):
    out = [{"ev": "run"}]
    for n in range(frames):
        out.append({"ev": "frame", "out_frame": n, "time": n * 100, "time_step": 100, "time_scale": 2400})
        for (t, i) in reads.get(n, []):
            out.append({"ev": "read", "out_frame": n, "layer": "input", "time": t, "index": i, "clip_frames": 4})
    return out


def plugin(frames, reads, smart=False, complete=True, legacy_name=False):
    """A plugin record in the harness format. smart: per read a `prerender_cb.checkout_layer` with a checkout id and the
    `smart_cb.checkout_layer_pixels` of that id (no time, no out_frame, as the harness logs it), plus per frame one
    registration of the input that is never read and one pixel read of an unset layer (param 13). legacy_name: the
    dotted pre-render name with no pixel events (the stand-in format of the first version of this tool)."""
    out = []
    for n in range(frames):
        out.append({"ev": "frame", "out_frame": n, "time": n * 100, "time_step": 100, "time_scale": 2400})
        out.append({"ev": "cb", "fn": "inter.checkout_param", "param_index": 7, "time": n * 100, "out_frame": n})     # a param sample
        if smart and not legacy_name:
            out.append({"ev": "cb", "fn": "prerender_cb.checkout_layer", "param_index": 0, "checkout_id": 90, "time": -2400,
                        "footage_frame": -24, "inside": False, "out_frame": n})                                     # a registration
            out.append({"ev": "cb", "fn": "prerender_cb.checkout_layer", "param_index": 13, "checkout_id": 91, "time": n * 100,
                        "footage_frame": 0, "inside": False, "out_frame": n})                                       # an unset layer
        for k, (t, i) in enumerate(reads.get(n, [])):
            if smart and legacy_name:
                out.append({"ev": "cb", "fn": "prerender.cb.checkout_layer", "param_index": 0, "time": t, "footage_frame": i,
                            "inside": 0 <= i < 4, "out_frame": n})
            elif smart:
                out.append({"ev": "cb", "fn": "prerender_cb.checkout_layer", "param_index": 0, "checkout_id": k, "time": t,
                            "footage_frame": i, "inside": 0 <= i < 4, "out_frame": n})
            else:
                out.append({"ev": "cb", "fn": "inter.checkout_param", "param_index": 0, "time": t, "footage_frame": i, "out_frame": n,
                            "layer": "input" if 0 <= i < 4 else "input (outside the clip: transparent)"})
        if smart and not legacy_name:
            for k in range(len(reads.get(n, []))):
                out.append({"ev": "cb", "fn": "smart_cb.checkout_layer_pixels", "checkout_id": k, "footage_frame": 0})
            out.append({"ev": "cb", "fn": "smart_cb.checkout_layer_pixels", "checkout_id": 91, "footage_frame": 0})
    if complete:
        out.append({"ev": "case_done", "status": "rendered"})
    return out


class Calls(unittest.TestCase):
    reads = {0: [(0, 0)], 1: [(100, 1), (200, 2)], 2: [(200, 2), (300, 3)]}
    SET = {"input"}

    def cmp(self, p, r, **kw):
        return pb.compare_records(pb.plugin_record(p, None, self.SET), pb.rewrite_record(r), **kw)

    def test_harness_names_resolve_pixel_reads_through_the_checkout_id(self):
        # the harness spelling `prerender_cb.checkout_layer`; the registration (id 90) has no pixels and is no read
        p = plugin(3, self.reads, smart=True)
        res = self.cmp(p, rewrite(3, self.reads))
        self.assertEqual((res["frames_compared"], res["frames_matching"], res["first_mismatch"]), (3, 3, None))
        rec = pb.plugin_record(p, None, self.SET)
        self.assertEqual(sorted(rec["reads"][1]), [("input", 100, 1, True), ("input", 200, 2, True)])

    def test_unset_layer_pixels_are_left_out_only_with_the_case_layers(self):
        p = plugin(1, {0: [(0, 0)]}, smart=True)
        self.assertEqual([r[0] for r in pb.plugin_record(p, None, self.SET)["reads"][0]], ["input"])
        self.assertEqual(sorted(r[0] for r in pb.plugin_record(p)["reads"][0]), ["input", "param13"])

    def test_a_dropped_pixel_checkout_is_a_missing_plugin_read(self):
        p = [x for x in plugin(3, self.reads, smart=True)
             if not (x.get("fn") == "smart_cb.checkout_layer_pixels" and x.get("checkout_id") == 1)]
        res = self.cmp(p, rewrite(3, self.reads))
        self.assertEqual(res["first_mismatch"]["out_frame"], 1)
        self.assertEqual([i["kind"] for i in res["first_mismatch"]["issues"]], ["extra_read"])

    def test_an_unknown_checkout_id_is_reported(self):
        p = plugin(1, {0: [(0, 0)]}, smart=True)
        p.insert(-1, {"ev": "cb", "fn": "smart_cb.checkout_layer_pixels", "checkout_id": 55, "footage_frame": 0})
        res = self.cmp(p, rewrite(1, {0: [(0, 0)]}))
        self.assertEqual([i["kind"] for i in res["first_mismatch"]["issues"]], ["missing_read"])

    def test_the_dotted_name_without_pixel_events_keeps_the_old_rule(self):
        p = plugin(3, self.reads, smart=True, legacy_name=True)
        res = self.cmp(p, rewrite(3, self.reads))
        self.assertEqual((res["frames_matching"], res["first_mismatch"]), (3, None))

    def test_match_ignores_order_param_samples_and_duplicates(self):
        p = plugin(3, {0: [(0, 0)], 1: [(200, 2), (100, 1), (100, 1)], 2: [(300, 3), (200, 2)]})
        res = self.cmp(p, rewrite(3, self.reads))
        self.assertEqual((res["frames_compared"], res["frames_matching"], res["first_mismatch"]), (3, 3, None))

    def test_ordered_mode_sees_the_order(self):
        p = plugin(3, {0: [(0, 0)], 1: [(200, 2), (100, 1)], 2: [(200, 2), (300, 3)]})
        res = self.cmp(p, rewrite(3, self.reads), ordered_mode=True)
        self.assertEqual(res["first_mismatch"]["out_frame"], 1)

    def test_missing_extra_wrong_time_wrong_index(self):
        base = dict(self.reads)
        cases = {
            "extra_read": ({1: [(100, 1)], 2: self.reads[2]}, 1),                  # the plugin never asks for it: the rewrite read is extra
            "missing_read": ({1: self.reads[1] + [(300, 3)], 2: self.reads[2]}, 1),   # the plugin asks for it: the rewrite lacks it
            "wrong_time": ({1: [(100, 1), (210, 2)], 2: self.reads[2]}, 1),
            "wrong_index": ({1: self.reads[1], 2: [(200, 2), (300, 2)]}, 2),
        }
        for kind, (reads, frame) in cases.items():
            p = plugin(3, {**base, **reads}, smart=(kind == "wrong_time"))
            res = self.cmp(p, rewrite(3, self.reads))
            self.assertEqual(res["first_mismatch"]["out_frame"], frame, kind)
            self.assertEqual([i["kind"] for i in res["first_mismatch"]["issues"]], [kind], kind)
            self.assertEqual(res["frames_matching"], 2, kind)

    def test_index_only_ignores_the_tick(self):
        p = plugin(3, {0: [(0, 0)], 1: [(150, 1), (200, 2)], 2: self.reads[2]})
        self.assertIsNotNone(self.cmp(p, rewrite(3, self.reads))["first_mismatch"])
        self.assertIsNone(self.cmp(p, rewrite(3, self.reads), index_only=True)["first_mismatch"])

    def test_outside_the_clip_on_both_sides(self):
        reads = {0: [(0, 0)], 1: [(100, 1)], 2: [(200, None)]}
        r = rewrite(3, {k: [(t, i) for t, i in v] for k, v in reads.items()})
        p = plugin(3, {0: [(0, 0)], 1: [(100, 1)], 2: [(200, 9)]})          # footage frame 9 of 4: outside
        self.assertIsNone(self.cmp(p, r)["first_mismatch"])

    def test_missing_output_frame(self):
        p = plugin(2, {0: [(0, 0)], 1: [(100, 1)]})
        res = self.cmp(p, rewrite(3, {0: [(0, 0)], 1: [(100, 1)], 2: []}))
        self.assertEqual(res["first_mismatch"]["issues"][0]["kind"], "missing_frame")

    def test_incomplete_record_is_flagged(self):
        self.assertFalse(pb.plugin_record(plugin(1, {}, complete=False))["complete"])
        self.assertTrue(pb.plugin_record(plugin(1, {}))["complete"])

    def test_layer_names_come_from_the_param_map(self):
        rec = pb.plugin_record([{"ev": "cb", "fn": "inter.checkout_param", "param_index": 9, "time": 0, "footage_frame": 0, "out_frame": 0,
                                 "layer": 9}], {9: "fg1_matte"})
        self.assertEqual(rec["reads"][0][0][0], "fg1_matte")


class Summary(unittest.TestCase):
    def grade(self, rows):
        t = {"audit_disagreements": [], "base_tsv_stale": []}
        return {"measured_at": "2026-01-01", "command": "g", "totals": t, "cases": rows}

    def row(self, cid, status, logic=False, dv=0):
        return {"id": cid, "status": status, "logic_parity": logic, "bytes_compared": 10, "floats_compared": 0,
                "differing_values": dv, "max_ulp": None, "nan_mismatch": 0}

    def test_counts_come_from_the_files_and_verified_needs_everything(self):
        cases = [case(cid=str(i)) for i in range(200)]
        rows = [self.row(str(i), "pass", logic=i < 5) for i in range(200)]
        kills = {"mutants_total": 3, "killed": 3, "survivors": [], "command": "k"}
        calls = {"compared": 4, "matching": 4, "command": "c"}
        s = pb.build_summary(cases, [{"id": "o"}] * 23, self.grade(rows), None, kills, calls, None, {}, {})
        self.assertEqual(s["status"], "verified")
        self.assertEqual(s["cases"], {"in_contract": 200, "compared": 200, "identical": 200, "differing": 0, "out_of_contract": 23, "logic_parity": 5})
        self.assertEqual(list(s), ["module", "plugin", "plugin_sha256", "effect", "status", "blockers", "cases", "values", "call_records",
                                   "mutants", "stand_ins", "fixes", "commands", "measured_at"])
        calls["matching"] = 3
        self.assertEqual(pb.build_summary(cases, [], self.grade(rows), None, kills, calls, None, {}, {})["status"], "in_progress")
        kills2 = dict(kills, killed=2, survivors=["M3"])
        self.assertEqual(pb.build_summary(cases, [], self.grade(rows), None, kills2, dict(calls, matching=4), None, {}, {})["status"], "in_progress")

    def test_fewer_than_150_or_a_missing_reference_is_not_verified(self):
        cases = [case(cid=str(i)) for i in range(100)]
        rows = [self.row(str(i), "pass") for i in range(100)]
        kills = {"mutants_total": 1, "killed": 1, "survivors": []}
        calls = {"compared": 1, "matching": 1}
        self.assertEqual(pb.build_summary(cases, [], self.grade(rows), None, kills, calls, None, {}, {})["status"], "in_progress")
        cases = [case(cid=str(i)) for i in range(200)]
        rows = [self.row(str(i), "pass" if i else "no reference") for i in range(200)]
        s = pb.build_summary(cases, [], self.grade(rows), None, kills, calls, None, {}, {})
        self.assertEqual((s["status"], s["cases"]["compared"], s["cases"]["in_contract"]), ("in_progress", 199, 200))

    def test_moved_cases_leave_the_in_contract_count(self):
        cases = [case(cid=str(i)) for i in range(10)]
        rows = [self.row(str(i), "pass") for i in range(10)]
        ooc = {"moved": [{"id": "1"}, {"id": "2"}], "candidates": [{"id": "o"}] * 3, "not_judged": [],
               "counts": {"not_judged": 0}}
        s = pb.build_summary(cases, [], self.grade(rows), ooc, None, None, None, {}, {})
        self.assertEqual((s["cases"]["in_contract"], s["cases"]["compared"], s["cases"]["out_of_contract"]), (8, 8, 5))
        self.assertEqual(s["mutants"]["killed"], 0)

    def test_killed_stays_zero_without_kills_json_and_survivors_follow_phase_a(self):
        cases = [case(cid="a")]
        s = pb.build_summary(cases, [], self.grade([self.row("a", "pass")]), None, None, None, None, {"M1": "changed-cases", "M5": "no-change"}, {})
        self.assertEqual(s["mutants"], {"total": 2, "killed": 0, "survivors": ["M5"]})

    def test_no_reference_at_all_keeps_the_phase_a_blocker(self):
        s = pb.build_summary([case(cid="a")], [], self.grade([self.row("a", "no reference")]), None, None, None, None, {}, {})
        self.assertEqual(s["blockers"][0], "emulator side not run yet")
        self.assertEqual(s["cases"]["compared"], 0)

    def test_fixes_and_plugin_fields_are_kept(self):
        prev = {"fixes": ["f1"], "plugin_sha256": "abc", "plugin": "P.aex", "commands": ["x"]}
        s = pb.build_summary([case(cid="a")], [], self.grade([self.row("a", "pass")]), None, None, None, prev, {}, {})
        self.assertEqual((s["fixes"], s["plugin_sha256"], s["commands"][0]), (["f1"], "abc", "x"))


class SummaryDefaults(unittest.TestCase):
    def test_fallbacks_and_call_blockers(self):
        g = {"measured_at": "2026-01-01", "command": "g", "totals": {"audit_disagreements": [], "base_tsv_stale": []},
             "cases": [{"id": "a", "status": "pass", "logic_parity": False, "bytes_compared": 1, "floats_compared": 0, "differing_values": 0, "max_ulp": None,
                        "nan_mismatch": 0}]}
        calls = {"compared": 1, "matching": 1, "statuses": {"match": 1, "no_plugin_record": 2, "incomplete_plugin_record": 1}}
        s = pb.build_summary([case(cid="a")], [], g, None, None, calls, None, {}, {})
        self.assertEqual(s["plugin_sha256"], pb.PLUGIN_SHA)
        self.assertEqual(s["commands"][:4], pb.PHASE_A_COMMANDS)
        self.assertIn("3 cases have no complete plugin call record", s["blockers"])

    def test_several_status_files_merge(self):
        with tempfile.TemporaryDirectory() as t:
            a, b = os.path.join(t, "a.json"), os.path.join(t, "b.json")
            json.dump({"cases": [{"id": "x", "status": "rendered"}]}, open(a, "w"))
            json.dump({"cases": [{"id": "o1", "status": "refused"}]}, open(b, "w"))
            res = os.path.join(t, "res")
            ts = os.path.join(t, "ts")
            os.makedirs(ts)
            json.dump({"cases": [case(cid="x")]}, open(os.path.join(ts, "cases.json"), "w"))
            json.dump({"cases": [{"id": "o1", "reason": "r", "vfx_verdict": "v"}]}, open(os.path.join(ts, "out_of_contract.json"), "w"))
            import argparse
            pb.cmd_ooc(argparse.Namespace(status=[a, b], results=res, testset=ts, base=os.path.join(t, "none.tsv"), param_map=None,
                                          move_harness_failures=False))
            d = json.load(open(os.path.join(res, "ooc.json")))
            self.assertEqual(d["candidates"][0]["plugin"]["status"], "refused")
            self.assertEqual(d["counts"]["rendered_by_the_original"], 1)


class Triage(unittest.TestCase):
    defaults = {"warping": {"id": "warping", "default": 3}, "track_rgb_a": {"id": "track_rgb_a", "default": False},
                "speed_percent": {"id": "speed_percent", "default": 100}}

    def test_ladder_rungs(self):
        c = case(params=["speed_percent=50", "warping=3", "track_rgb_a=1"])
        rungs = pb.ladder(c, self.defaults)
        kinds = [r[0] for r in rungs]
        self.assertEqual(kinds, ["full", "revert:speed_percent", "revert:track_rgb_a", "default", "only:speed_percent", "only:track_rgb_a"])
        self.assertEqual(rungs[1][2], [("warping", "3"), ("track_rgb_a", "1")])
        self.assertEqual(rungs[3][2], [])
        self.assertEqual(rungs[4][2], [("speed_percent", "50")])

    def test_keyed_and_negative_zero_are_not_default(self):
        self.assertFalse(pb.is_default("100@0;50@4", self.defaults["speed_percent"]))
        self.assertFalse(pb.is_default("-0", {"default": 0}))
        self.assertTrue(pb.is_default("100", self.defaults["speed_percent"]))
        self.assertTrue(pb.is_default("0", self.defaults["track_rgb_a"]))

    def test_unknown_defaults_make_every_setting_non_default(self):
        rungs = pb.ladder(case(params=["a=1", "b=2"]), {})
        self.assertEqual([r[0] for r in rungs], ["full", "revert:a", "revert:b", "default"])   # the only rungs equal the revert rungs

    def test_probe_set_is_an_fx_parity_test_set(self):
        with tempfile.TemporaryDirectory() as t:
            ts, probe = os.path.join(t, "ts"), os.path.join(t, "probe")
            os.makedirs(os.path.join(ts, "inputs"))
            open(os.path.join(ts, "inputs", "x"), "wb").write(bytes(8))
            c = case(params=["warping=2", "speed_percent=50"])
            meta = {"format": 1, "inputs": [{"file": "inputs/x", "source": "pan"}], "cases": [c]}
            lad = pb.write_probe(c, pb.ladder(c, self.defaults), ts, probe, meta)
            doc = json.load(open(os.path.join(probe, "cases.json")))
            self.assertEqual(doc["format"], 1)
            self.assertEqual(len(doc["cases"]), len(lad))
            self.assertEqual(len({x["id"] for x in doc["cases"]}), len(lad))
            self.assertTrue(os.path.exists(os.path.join(probe, "inputs", "x")))
            self.assertTrue(all(x["reference"].startswith(x["id"] + ".") for x in doc["cases"]))
            self.assertEqual(doc["cases"][3]["params"], [])

    def test_tables(self):
        rows = [{"status": "FAIL", "block": "warping", "tags": ["x"], "depth": 8, "frames": 4, "params": ["warping=2"]},
                {"status": "pass", "block": "warping", "tags": [], "depth": 8, "frames": 4, "params": ["warping=3"]},
                {"status": "no reference", "block": "warping", "tags": [], "depth": 8, "frames": 4, "params": []}]
        g = pb.triage_tables(rows, {})
        self.assertEqual(g[("param", "warping=2")], [1, 1])
        self.assertEqual(g[("block", "warping")], [1, 2])


class Helpers(unittest.TestCase):
    def test_tsv_and_grid_loading(self):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "b.tsv")
            open(p, "w").write("id\texit\tbytes\tsha256\tseconds\terror\na\t0\t4\tAA\t0.01\t\nb\t3\t0\t-\t0.0\tfail\n")
            r = pb.load_tsv(p)
            self.assertEqual((r["a"]["sha"], r["b"]["exit"], r["b"]["error"]), ("AA", 3, "fail"))
            open(os.path.join(t, "grid.tsv"), "w").write("id\tblock\ttags\na\tsize\ttiny,standin\nb\tspec\t\n")
            g = pb.load_grid(t)
            self.assertEqual(g["a"], ("size", {"tiny", "standin"}))
            self.assertEqual(g["b"], ("spec", set()))

    def test_vfx_args_match_the_phase_a_command(self):
        a = pb.vfx_args("vfx", case(params=["warping=2"]), "/ts")
        self.assertEqual(a[:3], ["vfx", "apply", "retime-fgbg"])
        self.assertIn("--param", a)
        self.assertEqual(a[-2:], ["--in", os.path.join("/ts", "inputs/x")])

    def test_no_dashes_in_the_sources(self):
        for name in ("phase_b.py", "test_phase_b.py"):
            text = open(os.path.join(HERE, name), encoding="utf-8").read()
            self.assertNotIn(chr(0x2014), text)
            self.assertNotIn(chr(0x2013), text)


if __name__ == "__main__":
    unittest.main()
