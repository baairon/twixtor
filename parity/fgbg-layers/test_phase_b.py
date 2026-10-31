#!/usr/bin/env python3
"""Tests of the logic in phase_b.py that needs neither the server nor a plugin.

  python3 -m unittest test_phase_b        (run from parity/fgbg-layers)

The parts that need vfx and fx-parity (grade, kills on real renders, record-rewrite) are covered by
`python3 phase_b.py selftest` on the server.
"""
import json
import os
import struct
import sys
import tempfile
import time
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase_b as pb  # noqa: E402


def f32(bits):
    return struct.pack("<I", bits)


def case(cid="c1", depth=8, w=2, h=2, frames=3, params=(), layers=None, source="box2"):
    c = {"id": cid, "effect": "retime-fgbg", "label": "x", "params": list(params), "depth": depth, "width": w, "height": h,
         "source": source, "input": "inputs/in.%d" % depth, "reference": cid + ".rgba", "kind": "temporal", "frames": frames, "fps": "24"}
    if layers:
        c["layers"] = dict(layers)
    return c


class ValueDiff(unittest.TestCase):
    def test_ulp(self):
        self.assertEqual(pb.ulp_distance(0x3F800000, 0x3F800001), 1)
        self.assertEqual(pb.ulp_distance(0x00000000, 0x80000000), 0)
        self.assertEqual(pb.ulp_distance(0x00000001, 0x80000001), 2)
        self.assertIsNone(pb.ulp_distance(0x7FC00000, 0x3F800000))
        self.assertIsNone(pb.ulp_distance(0x7FC00000, 0x7FC00001))

    def test_nan_detection(self):
        self.assertTrue(pb.f32_is_nan(0x7FC00001))
        self.assertTrue(pb.f32_is_nan(0xFFC00000))
        self.assertFalse(pb.f32_is_nan(0x7F800000))
        self.assertFalse(pb.f32_is_nan(0x3F800000))

    def test_8bit(self):
        a = bytes(range(16))
        b = bytearray(a)
        b[3] ^= 1
        d = pb.sample_diff(8, a, bytes(b))
        self.assertEqual((d["values"], d["differing"], d["max_ulp"], d["malformed"]), (16, 1, None, False))
        self.assertEqual(pb.sample_diff(8, a, a)["differing"], 0)

    def test_16bit_counts_values_not_bytes(self):
        a = struct.pack("<4H", 1, 2, 3, 4)
        b = struct.pack("<4H", 1, 0x0102, 3, 0x0304)
        d = pb.sample_diff(16, a, b)
        self.assertEqual((d["values"], d["differing"]), (4, 2))

    def test_32bit_bit_for_bit(self):
        ref = f32(0x3F800000) + f32(0x00000000) + f32(0x7FC00000) + f32(0x3F800000)
        out = f32(0x3F800001) + f32(0x80000000) + f32(0x7FC00001) + f32(0x3F800000)
        d = pb.sample_diff(32, ref, out)
        self.assertEqual(d["values"], 4)
        self.assertEqual(d["differing"], 3)            # 1 ULP, -0 against 0, NaN payload
        self.assertEqual(d["max_ulp"], 1)
        self.assertEqual(d["nan_mismatches"], 1)
        zero = pb.sample_diff(32, f32(0), f32(0x80000000))
        self.assertEqual((zero["differing"], zero["max_ulp"]), (1, 0))
        nan_vs_num = pb.sample_diff(32, f32(0x7FC00000), f32(0x3F800000))
        self.assertEqual((nan_vs_num["differing"], nan_vs_num["nan_mismatches"], nan_vs_num["max_ulp"]), (1, 1, None))

    def test_malformed_sizes_count_everything(self):
        d = pb.sample_diff(8, b"\0" * 8, b"\0" * 12)
        self.assertTrue(d["malformed"])
        self.assertEqual(d["differing"], 12)


class Tags(unittest.TestCase):
    def test_stand_ins(self):
        self.assertEqual(pb.stand_in_reasons(case(), []), [])
        self.assertEqual(pb.stand_in_reasons(case(layers={"fg1_matte": "m"}), []), ["matte-layers"])
        self.assertEqual(pb.stand_in_reasons(case(), ["standin"]), ["keyframes"])
        self.assertEqual(pb.stand_in_reasons(case(params=["pt1_pos=1,1@0;5,5@10"], layers={"fg1_matte": "m"}), []), ["keyframes", "matte-layers"])
        self.assertEqual(pb.stand_in_reasons(case(layers={"fg1_matte": "m"}), ["standin"], ("keyframes",)), ["keyframes"])

    def test_gaps(self):
        self.assertEqual(pb.gaps_of(case(params=["fg1_inv_matte_shrink=2"]), "shrink"), ["G1"])
        self.assertIn("G2", pb.gaps_of(case(params=["pt1_use=2"]), "points"))
        self.assertIn("G3", pb.gaps_of(case(params=["draw_geom=2"]), "geom"))
        self.assertIn("G6", pb.gaps_of(case(params=["track_rgb_a=1"]), "mix"))
        self.assertIn("G1c", pb.gaps_of(case(layers={"fg1_matte": "a", "fg2_matte": "b"}), "layers"))
        self.assertEqual(pb.gaps_of(case(params=["speed_percent=50"]), "size"), [])
        self.assertIn("G1g", pb.gaps_of(case(params=["display=4"]), "mix"))


def rows_with(ranges=None, popup=None):
    rows = [{"plugin": pb.PLUGIN_NAME, "ae_index": "1", "ae_type": "LAYER", "vfx_kind": "layer", "vfx_id": "fg1_matte", "match": "exact"}]
    for i, (pid, (lo, hi)) in enumerate((ranges or {}).items(), 10):
        rows.append({"plugin": pb.PLUGIN_NAME, "ae_index": str(i), "ae_type": "FLOAT_SLIDER", "ae_min": lo, "ae_max": hi, "vfx_kind": "float", "vfx_id": pid, "match": "exact"})
    for i, (pid, choices) in enumerate((popup or {}).items(), 40):
        rows.append({"plugin": pb.PLUGIN_NAME, "ae_index": str(i), "ae_type": "POPUP", "ae_choices": "|".join(choices), "vfx_kind": "popup", "vfx_id": pid, "match": "exact"})
    return rows


class Ranges(unittest.TestCase):
    def test_float_range(self):
        reg = pb.registered_by_id(rows_with({"fg1_inv_matte_shrink": ("0", "10")}))
        self.assertEqual(pb.range_violations(case(params=["fg1_inv_matte_shrink=-0.5"]), reg)[0][:25], "fg1_inv_matte_shrink=-0.5")
        self.assertEqual(pb.range_violations(case(params=["fg1_inv_matte_shrink=10"]), reg), [])
        self.assertEqual(pb.range_violations(case(params=["fg1_inv_matte_shrink=-0"]), reg), [])
        self.assertTrue(pb.range_violations(case(params=["fg1_inv_matte_shrink=10.01"]), reg))
        self.assertTrue(pb.range_violations(case(params=["fg1_inv_matte_shrink=nan"]), reg))
        self.assertTrue(pb.range_violations(case(params=["fg1_inv_matte_shrink=1@0;11@5"]), reg))
        self.assertEqual(pb.range_violations(case(params=["speed_percent=99999"]), reg), [])    # not registered: nothing to judge

    def test_popup_range(self):
        reg = pb.registered_by_id(rows_with(popup={"fg1_matte_chan": ["Red", "Green", "Blue", "Alpha"]}))
        self.assertEqual(pb.range_violations(case(params=["fg1_matte_chan=4"]), reg), [])
        self.assertTrue(pb.range_violations(case(params=["fg1_matte_chan=5"]), reg))
        self.assertTrue(pb.range_violations(case(params=["fg1_matte_chan=0"]), reg))

    def test_unmatched_flagged_only_with_a_plugin_side(self):
        cs = [case(params=["main_bg_track=1", "speed_percent=50"], layers={"fg1_matte": "m"})]
        self.assertIsNone(pb.unmatched_params(cs, [{"plugin": pb.PLUGIN_NAME, "vfx_id": "speed_percent", "match": "unmatched"}]))
        rows = rows_with({"speed_percent": ("0", "100")})
        rows.append({"plugin": pb.PLUGIN_NAME, "vfx_id": "main_bg_track", "match": "unmatched"})
        um = pb.unmatched_params(cs, rows)
        self.assertEqual(sorted(um), ["main_bg_track"])
        self.assertIn("unmatched", um["main_bg_track"]["why"])


class Ooc(unittest.TestCase):
    def test_classify(self):
        f = pb.classify_original
        self.assertEqual(f({"status": "rendered"}), "rendered")
        self.assertEqual(f({"status": "refused"}), "refused")
        self.assertEqual(f({"status": "unsupported", "error": "param `x` has no counterpart among the registered params of P"}), "param_unregistered")
        self.assertEqual(f({"status": "unsupported", "error": "Bezier handles are refused"}), "harness_limit")
        self.assertEqual(f({"status": "error", "error": "PluginError: RENDER returned PF_Err 3"}), "plugin_error")
        self.assertEqual(f({"status": "error", "error": "GuestFault: x"}), "emulator_fault")
        self.assertEqual(f({"status": "timeout"}), "timeout")
        self.assertEqual(f({"status": "not_run"}), "not_run")

    def test_load_status_formats(self):
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "a.json")
            json.dump({"cases": [{"id": "x", "status": "rendered"}, {"id": "y", "status": "error", "error": "PluginError: z"}]}, open(a, "w"))
            b = os.path.join(d, "b.jsonl")
            open(b, "w").write('{"event":"boot","ok":true}\n{"event":"done","id":"z","status":"timeout"}\nnot json\n')
            st = pb.load_status([a, b])
            self.assertEqual(sorted(st), ["x", "y", "z"])
            self.assertEqual(st["z"]["status"], "timeout")

    def test_judge(self):
        cs = [case("a"), case("b"), case("c"), case("d"), case("e", params=["fg1_inv_matte_shrink=-1"])]
        status = {"a": {"status": "rendered"}, "b": {"status": "error", "error": "PluginError: boom"}, "c": {"status": "timeout"},
                  "d": {"status": "refused", "error": "no"}, "e": {"status": "rendered"}}
        rows = rows_with({"fg1_inv_matte_shrink": ("0", "10")})
        cands = [case("o1"), case("o2"), case("o3")]
        cstatus = {"o1": {"status": "refused"}, "o2": {"status": "rendered"}}
        status.update(cstatus)
        doc = pb.judge(cs, {}, status, rows, {"b": {"exit": "0", "bytes": "5"}}, cands, {"o1": {"exit": "3"}}, set(pb.OOC_DEFAULT))
        moved = {m["id"]: m["category"] for m in doc["moved_out"]}
        self.assertEqual(moved, {"b": "plugin_error", "d": "refused", "e": "out_of_range"})
        self.assertEqual([u["id"] for u in doc["unresolved"]], ["c"])
        verdicts = {c["id"]: c["verdict"] for c in doc["candidates"]}
        self.assertEqual(verdicts, {"o1": "out_of_contract", "o2": "original_rendered_review", "o3": "unjudged"})
        self.assertEqual(doc["counts"]["out_of_contract_total"], 3 + 2)
        self.assertEqual(doc["counts"]["candidates_promote_review"], 1)
        self.assertEqual(doc["moved_out"][0]["rewrite"]["exit"], "0")

    def test_optional_categories(self):
        doc = pb.judge([case("c")], {}, {"c": {"status": "timeout"}}, [], {}, [], {}, set(pb.OOC_DEFAULT) | {"timeout"})
        self.assertEqual([m["id"] for m in doc["moved_out"]], ["c"])
        self.assertFalse(doc["param_map_plugin_side"])

    def test_judged_cases_move_with_their_reason_and_only_with_the_file(self):
        cs = [case("a"), case("b"), case("c"), case("e", params=["fg1_inv_matte_shrink=-1"])]
        status = {"a": {"status": "rendered"}, "b": {"status": "rendered"}, "c": {"status": "error", "error": "GuestFault: x"},
                  "e": {"status": "rendered"}}
        rows = rows_with({"fg1_inv_matte_shrink": ("0", "10")})
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "judged.tsv")
            open(p, "w").write("# comment\nid\treason\tevidence\nb\tthe original leaves the word undefined\tnotes Q4\n"
                               "c\tthe original stops on it\treadfaults\ne\tlisted too\t-\nzz\tnot in the set\t-\n")
            j = pb.load_judged([p])
            doc = pb.judge(cs, {}, status, rows, {}, [], {}, set(pb.OOC_DEFAULT), j)
            moved = {m["id"]: m["category"] for m in doc["moved_out"]}
            # the range check comes first; the judged cases move with category judged and their reason
            self.assertEqual(moved, {"b": "judged", "c": "judged", "e": "out_of_range"})
            self.assertEqual(doc["moved_out"][0]["reasons"], ["the original leaves the word undefined", "notes Q4"])
            self.assertEqual(doc["unresolved"], [])
            self.assertEqual(doc["judged_unknown"], ["zz"])
            # without the file nothing is judged: the move is reversible
            doc0 = pb.judge(cs, {}, status, rows, {}, [], {}, set(pb.OOC_DEFAULT))
            self.assertEqual({m["id"] for m in doc0["moved_out"]}, {"e"})
            self.assertEqual([u["id"] for u in doc0["unresolved"]], ["c"])
            open(p, "w").write("id\treason\tevidence\nb\t\tx\n")
            with self.assertRaises(pb.ToolError):
                pb.load_judged([p])


def plugin_lines(frames, reads, smart=True, status="rendered"):
    """reads: (out_frame, param_index, time, footage_frame, inside) tuples."""
    out = [{"seq": 1, "ev": "render_path", "smart": smart}]
    for f in range(frames):
        out.append({"ev": "frame", "out_frame": f, "time": f * 100, "time_step": 100, "time_scale": 2400})
    for of, idx, t, fr, inside in reads:
        out.append({"ev": "cb", "fn": "prerender_cb.checkout_layer", "param_index": idx, "time": t, "time_step": 100, "time_scale": 2400,
                    "footage_frame": fr, "inside": inside, "out_frame": of})
    out.append({"ev": "case_done", "status": status})
    return out


def rewrite_lines(frames, reads):
    out = [{"ev": "run", "time_scale": 2400, "time_step": 100}]
    for f in range(frames):
        out.append({"ev": "frame", "out_frame": f, "time": f * 100, "time_step": 100, "time_scale": 2400})
        for of, layer, t, idx in reads:
            if of == f:
                out.append({"ev": "read", "out_frame": f, "layer": layer, "time": t, "index": idx, "clip_frames": 3})
    return out


class Calls(unittest.TestCase):
    rows = rows_with()

    def cmp(self, plugin_reads, rewrite_reads, frames=2, ordered=False, smart=True):
        p = pb.normalize_plugin(plugin_lines(frames, plugin_reads, smart), self.rows, case(layers={"fg1_matte": "m"}), {"input": 3, "fg1_matte": 3})
        r = pb.normalize_rewrite(rewrite_lines(frames, rewrite_reads))
        return pb.compare_calls(p, r, ordered)

    def test_match_as_sets(self):
        res = self.cmp([(0, 0, 0, 0, True), (0, 1, 0, 0, True), (1, 0, 100, 1, True)],
                       [(0, "fg1_matte", 0, 0), (0, "input", 0, 0), (1, "input", 100, 1)])
        self.assertEqual(res["status"], "match")

    def test_duplicates_collapse(self):
        res = self.cmp([(0, 0, 0, 0, True), (0, 0, 0, 0, True)], [(0, "input", 0, 0)], frames=1)
        self.assertEqual(res["status"], "match")

    def test_order_matters_only_when_asked(self):
        a = [(0, 0, 0, 0, True), (0, 1, 0, 0, True)]
        b = [(0, "fg1_matte", 0, 0), (0, "input", 0, 0)]
        self.assertEqual(self.cmp(a, b, frames=1)["status"], "match")
        self.assertEqual(self.cmp(a, b, frames=1, ordered=True)["status"], "mismatch")

    def kinds(self, res):
        return [d["kind"] for d in res["first_mismatch"]["differences"]]

    def test_missing_and_extra(self):
        res = self.cmp([(0, 0, 0, 0, True), (0, 1, 0, 0, True)], [(0, "input", 0, 0)], frames=1)
        self.assertEqual(self.kinds(res), ["missing_read"])
        res = self.cmp([(0, 0, 0, 0, True)], [(0, "input", 0, 0), (0, "fg1_matte", 0, 0)], frames=1)
        self.assertEqual(self.kinds(res), ["extra_read"])

    def test_wrong_time_index_layer(self):
        self.assertEqual(self.kinds(self.cmp([(0, 0, 50, 0, True)], [(0, "input", 0, 0)], frames=1)), ["wrong_time"])
        self.assertEqual(self.kinds(self.cmp([(0, 0, 0, 1, True)], [(0, "input", 0, 0)], frames=1)), ["wrong_frame_index"])
        self.assertEqual(self.kinds(self.cmp([(0, 1, 0, 0, True)], [(0, "input", 0, 0)], frames=1)), ["wrong_layer"])

    def test_first_differing_frame_is_reported(self):
        res = self.cmp([(0, 0, 0, 0, True), (1, 0, 100, 1, True)], [(0, "input", 0, 0), (1, "input", 100, 2)])
        self.assertEqual(res["first_mismatch"]["out_frame"], 1)

    def test_outside_the_clip_is_index_none(self):
        res = self.cmp([(0, 0, 0, 7, False)], [(0, "input", 0, None)], frames=1)
        self.assertEqual(res["status"], "match")

    def test_time_base_is_compared_as_seconds(self):
        p = pb.normalize_plugin(plugin_lines(1, [(0, 0, 0, 0, True)]), self.rows)
        r = pb.normalize_rewrite([{"ev": "frame", "out_frame": 0, "time": 0, "time_step": 50, "time_scale": 1200},
                                  {"ev": "read", "out_frame": 0, "layer": "input", "time": 0, "index": 0, "clip_frames": 3}])
        self.assertEqual(pb.compare_calls(p, r)["status"], "match")
        r["frames"][0]["time"] = 50
        self.assertEqual(pb.compare_calls(p, r)["first_mismatch"]["kind"], "frame_time")

    def test_checkout_param_on_a_layer_param_is_a_layer_read(self):
        lines = plugin_lines(1, [])
        lines.insert(-1, {"ev": "cb", "fn": "inter.checkout_param", "param_index": 1, "time": 0, "time_step": 100, "time_scale": 2400, "footage_frame": 0, "out_frame": 0})
        lines.insert(-1, {"ev": "cb", "fn": "inter.checkout_param", "param_index": 10, "time": 0, "time_step": 100, "time_scale": 2400, "footage_frame": 0, "out_frame": 0})
        rows = rows_with({"speed_percent": ("0", "100")})
        n = pb.normalize_plugin(lines, rows, case(layers={"fg1_matte": "m"}), {"fg1_matte": 3})
        self.assertEqual([x[0] for x in n["reads"][0]], ["fg1_matte"])
        self.assertEqual([x[0] for x in n["params"][0]], ["speed_percent"])
        # the same read of a layer the case does not set is left out (size 0 in the harness, nothing to read)
        n = pb.normalize_plugin(lines, rows, case(), {"fg1_matte": 3})
        self.assertEqual(n["reads"].get(0, []), [])
        self.assertEqual(n["notes"]["unset_layer_checkouts"], 1)

    def test_unset_layers_are_left_out(self):
        # pre-render registrations of an unset layer (param 1, fg1_matte, not in the case) never reach the comparison
        lines = plugin_lines(1, [(0, 0, 0, 0, True), (0, 1, 0, 0, False), (0, 1, 100, 1, False)])
        p = pb.normalize_plugin(lines, self.rows, case(), {"input": 3})
        r = pb.normalize_rewrite(rewrite_lines(1, [(0, "input", 0, 0)]))
        self.assertEqual(pb.compare_calls(p, r)["status"], "match")
        self.assertEqual(p["notes"]["unset_layer_checkouts"], 2)
        # with the layer set, the same record reads it
        p = pb.normalize_plugin(lines, self.rows, case(layers={"fg1_matte": "m"}), {"input": 3, "fg1_matte": 3})
        self.assertEqual(self.kinds(pb.compare_calls(p, r)), ["missing_read", "missing_read"])

    def pixel_lines(self, prerender, pixels):
        """prerender: (checkout_id, param_index, time, footage_frame); pixels: checkout ids read in SMART_RENDER."""
        out = [{"ev": "render_path", "smart": True}, {"ev": "frame", "out_frame": 0, "time": 0, "time_step": 100, "time_scale": 2400}]
        for cid, idx, t, fr in prerender:
            out.append({"ev": "cb", "fn": "prerender_cb.checkout_layer", "param_index": idx, "checkout_id": cid, "time": t, "time_step": 100,
                        "time_scale": 2400, "footage_frame": fr, "inside": True, "out_frame": 0})
        for cid in pixels:
            out.append({"ev": "cb", "fn": "smart_cb.checkout_layer_pixels", "checkout_id": cid, "footage_frame": 0})
        out.append({"ev": "case_done", "status": "rendered"})
        return out

    def test_pixel_checkouts_are_the_reads(self):
        # the plugin checks out the input at 0 and 100 and the matte at 0 in the pre-render but reads only two of them
        lines = self.pixel_lines([(0, 0, 0, 0), (1, 0, 100, 1), (2, 1, 0, 0)], [0, 2])
        c = case(layers={"fg1_matte": "m"})
        p = pb.normalize_plugin(lines, self.rows, c, {"input": 3, "fg1_matte": 3})
        self.assertEqual(p["read_source"], "pixel checkouts")
        r = pb.normalize_rewrite(rewrite_lines(1, [(0, "input", 0, 0), (0, "fg1_matte", 0, 0)]))
        self.assertEqual(pb.compare_calls(p, r)["status"], "match")
        r = pb.normalize_rewrite(rewrite_lines(1, [(0, "input", 0, 0), (0, "input", 100, 1), (0, "fg1_matte", 0, 0)]))
        self.assertEqual(self.kinds(pb.compare_calls(p, r)), ["extra_read"])
        # a pixel read of an unset layer is counted, not compared
        p = pb.normalize_plugin(lines, self.rows, case(), {"input": 3})
        self.assertEqual(p["notes"]["unset_layer_pixel_reads"], 1)
        r = pb.normalize_rewrite(rewrite_lines(1, [(0, "input", 0, 0)]))
        self.assertEqual(pb.compare_calls(p, r)["status"], "match")

    def test_classic_path_adds_implicit_reads(self):
        n = pb.normalize_plugin(plugin_lines(2, [], smart=False), self.rows, case(layers={"fg1_matte": "m"}), {"input": 3, "fg1_matte": 3})
        self.assertEqual(sorted(x[0] for x in n["reads"][1]), ["fg1_matte", "input"])
        self.assertEqual(n["reads"][1][0][1], 1)

    def test_rewrite_param_events_are_compared_when_present(self):
        p = pb.normalize_plugin(plugin_lines(1, [(0, 0, 0, 0, True)]), self.rows)
        p["params"] = {0: [("speed_percent", Fraction(0), 0)]}
        r = pb.normalize_rewrite(rewrite_lines(1, [(0, "input", 0, 0)]))
        r["params"] = {0: [("speed_percent", Fraction(1, 24), 100)]}
        res = pb.compare_calls(p, r)
        self.assertEqual(sorted(self.kinds(res)), ["param_time_extra", "param_time_missing"])


class Kills(unittest.TestCase):
    muts = [{"id": "M1", "what": "a"}, {"id": "M2", "what": "b"}, {"id": "M3", "what": "c"}]
    cases = [case("c%d" % i) for i in range(4)]
    base = {"c%d" % i: ("0", "h%d" % i) for i in range(4)}

    def ref(self, wrong=()):
        return {"c%d" % i: ("x" if "c%d" % i in wrong else "h%d" % i) for i in range(4)}

    def run_(self, refs, data):
        return pb.kill_analysis(self.muts, self.cases, self.base, refs, data)

    def status(self, ka):
        return {m["id"]: m["status"] for m in ka["mutants"]}

    def test_perfect_oracle(self):
        data = {"M1": ("changed.txt", {"c0": None, "c1": None}), "M2": ("changed.txt", {"c3": None}), "M3": ("changed.txt", {})}
        ka = self.run_(self.ref(), data)
        self.assertEqual(ka["discriminating_cases"], 4)
        self.assertEqual(self.status(ka), {"M1": "killed", "M2": "killed", "M3": "survivor"})
        self.assertEqual(ka["mutants"][0]["killing_cases"], ["c0", "c1"])

    def test_cases_where_the_base_already_differs_cannot_discriminate(self):
        data = {"M1": ("changed.txt", {"c0": None}), "M2": ("changed.txt", {"c0": None, "c1": None}), "M3": ("changed.txt", {"c3": None})}
        ka = self.run_(self.ref(wrong=("c0", "c3")), data)
        self.assertEqual(self.status(ka), {"M1": "survivor", "M2": "killed", "M3": "survivor"})
        self.assertEqual(ka["base_differs_from_reference"], ["c0", "c3"])
        self.assertEqual(ka["mutants"][1]["killing_cases"], ["c1"])

    def test_missing_references_leave_mutants_undecided_not_killed(self):
        data = {"M1": ("changed.txt", {"c0": None}), "M2": ("changed.txt", {}), "M3": ("changed.txt", {"c1": None})}
        ka = self.run_({}, data)
        self.assertEqual(ka["discriminating_cases"], 0)
        self.assertEqual(self.status(ka), {"M1": "undecided", "M2": "survivor", "M3": "undecided"})

    def test_full_tables_compare_hashes(self):
        full = {"c%d" % i: ("0", "h%d" % i) for i in range(4)}
        full["c2"] = ("0", "mutant")
        data = {"M1": ("results.tsv", full), "M2": ("results.tsv", dict(self.base)), "M3": (None, None)}
        ka = self.run_(self.ref(), data)
        self.assertEqual(self.status(ka), {"M1": "killed", "M2": "survivor", "M3": "undecided"})
        self.assertEqual(ka["mutants"][0]["killing_cases"], ["c2"])
        self.assertEqual(ka["mutants"][0]["changed_vs_base"], 1)

    def test_a_mutant_that_only_fails_to_render_is_not_killed_by_it(self):
        full = dict(self.base)
        full["c1"] = ("3", "-")
        ka = self.run_(self.ref(), {"M1": ("results.tsv", full), "M2": ("changed.txt", {}), "M3": ("changed.txt", {})})
        self.assertEqual(self.status(ka)["M1"], "survivor")
        self.assertEqual(ka["mutants"][0]["via_render_failure"], ["c1"])

    def test_mutant_dirs_prefer_the_replacement(self):
        with tempfile.TemporaryDirectory() as d:
            for sub, ids in (("mutants", ""), ("mutants2", "c1\n")):
                os.makedirs(os.path.join(d, sub, "M05"))
                open(os.path.join(d, sub, "M05", "changed.txt"), "w").write(ids)
            src, tab = pb.mutant_hashes("M05", [os.path.join(d, "mutants2"), os.path.join(d, "mutants")], d)
            self.assertTrue(src.startswith("changed.txt:") and "mutants2" in src)
            self.assertEqual(list(tab), ["c1"])
            os.makedirs(os.path.join(d, "results", "mutants"))
            open(os.path.join(d, "results", "mutants", "M07.changed.txt"), "w").write("c0\nc2\n")
            self.assertEqual(sorted(pb.mutant_hashes("M07", [], os.path.join(d, "results"))[1]), ["c0", "c2"])
            self.assertEqual(pb.mutant_hashes("M99", [], os.path.join(d, "results")), (None, None))


class Rebuild(unittest.TestCase):
    """The cargo mtime lesson: a restored file with an old timestamp made cargo reuse the previous mutant."""

    def test_touch_newer(self):
        with tempfile.TemporaryDirectory() as d:
            ref, f = os.path.join(d, "ref"), os.path.join(d, "f")
            open(ref, "w").write("x")
            open(f, "w").write("y")
            os.utime(f, (1, 1))
            os.utime(ref, (time.time() + 100, time.time() + 100))
            pb.touch_newer(f, ref)
            self.assertGreater(os.path.getmtime(f), os.path.getmtime(ref))

    def test_rebuild_touches_both_the_mutated_and_the_restored_file(self):
        with tempfile.TemporaryDirectory() as d:
            tree = os.path.join(d, "tree")
            os.makedirs(os.path.join(tree, "src"))
            os.makedirs(os.path.join(tree, "target", "release"))
            src = os.path.join(tree, "src", "a.rs")
            open(src, "w").write("let v = 1;\n")
            vfx = os.path.join(tree, "target", "release", "vfx")
            open(vfx, "w").write("old build")
            future = time.time() + 500          # an artifact newer than the source, like a previous build
            os.utime(vfx, (future, future))
            os.utime(src, (1, 1))
            fake = os.path.join(d, "fake_cargo.py")
            open(fake, "w").write(
                "import os, sys\n"
                "src = os.path.join('src', 'a.rs'); vfx = os.path.join('target', 'release', 'vfx')\n"
                "newer = os.path.getmtime(src) > os.path.getmtime(vfx)\n"
                "open(vfx, 'w').write(('BUILT ' if newer else 'REUSED ') + open(src).read())\n")
            m = {"id": "M1", "file": "src/a.rs", "old": "let v = 1;", "new": "let v = 2;", "count": 1}
            r = pb.rebuild_mutant(tree, m, os.path.join(d, "out"), cargo=[sys.executable, fake])
            self.assertEqual(r["status"], "built")
            self.assertEqual(open(r["binary"]).read(), "BUILT let v = 2;\n")
            self.assertEqual(open(src).read(), "let v = 1;\n")         # restored
            self.assertGreaterEqual(os.path.getmtime(src), os.path.getmtime(vfx))
            r2 = pb.rebuild_mutant(tree, dict(m, id="M2", new="let v = 3;"), os.path.join(d, "out"), cargo=[sys.executable, fake])
            self.assertEqual(open(r2["binary"]).read(), "BUILT let v = 3;\n")
            self.assertNotEqual(r["binary_sha256"], r2["binary_sha256"])

    def test_anchor_mismatch_is_reported_and_nothing_changes(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "src"))
            open(os.path.join(d, "src", "a.rs"), "w").write("abc")
            r = pb.rebuild_mutant(d, {"id": "M1", "file": "src/a.rs", "old": "zzz", "new": "y", "count": 1}, os.path.join(d, "o"), cargo=[sys.executable, "-c", "pass"])
            self.assertEqual(r["status"], "anchor-mismatch")
            self.assertEqual(open(os.path.join(d, "src", "a.rs")).read(), "abc")


def graded(cid, result, depth=8, vals=None, ulp=None, nan=0, lp=False):
    return {"id": cid, "result": result, "depth": depth, "width": 2, "height": 2, "frames": 3, "differing_values": vals,
            "max_ulp": ulp, "nan_mismatches": nan, "logic_parity": lp}


class Summary(unittest.TestCase):
    template = {"plugin": "TwixtorPro_64.aex", "plugin_sha256": "abc", "commands": ["python3 gen_grid.py"]}

    def build(self, grade_cases, ooc=None, kills=None, calls=None, n=None):
        cs = [case(g["id"]) for g in grade_cases] if n is None else [case("c%d" % i) for i in range(n)]
        return pb.build_summary(self.template, cs, {}, {"cases": grade_cases, "measured_at": "2026-10-30"}, ooc, kills, calls, 27, 35, [])

    def test_schema_and_counts(self):
        g = [graded("a", "identical", vals=0, lp=True), graded("b", "differing", 32, 2, 3, 1), graded("c", "no_reference")]
        s = self.build(g)
        self.assertEqual(list(s), ["module", "plugin", "plugin_sha256", "effect", "status", "blockers", "cases", "values", "call_records",
                                   "mutants", "stand_ins", "fixes", "commands", "measured_at"])
        self.assertEqual(dict(s["cases"]), {"in_contract": 3, "compared": 2, "identical": 1, "differing": 1, "out_of_contract": 35, "logic_parity": 1})
        self.assertEqual(dict(s["values"]), {"bytes_compared": 48, "floats_compared": 48, "differing_values": 2, "max_ulp": 3})
        self.assertEqual(dict(s["mutants"]), {"total": 27, "killed": 0, "survivors": []})
        self.assertEqual(s["status"], "in_progress")
        self.assertTrue(any("1 in-contract cases have no reference" in b for b in s["blockers"]))

    def test_no_reference_is_not_compared(self):
        s = self.build([graded("a", "no_reference"), graded("b", "no_reference")])
        self.assertEqual(s["cases"]["compared"], 0)
        self.assertEqual(s["values"]["max_ulp"], None)
        self.assertTrue(s["blockers"][0].startswith("emulator side not run yet"))

    def test_out_of_contract_cases_leave_the_count(self):
        ooc = {"moved_out": [{"id": "b"}], "unresolved": [], "counts": {"out_of_contract_total": 36, "unresolved": 0, "candidates_promote_review": 0, "status_counts": {"status:rendered": 2}}}
        s = self.build([graded("a", "identical", vals=0), graded("b", "identical", vals=0), graded("c", "identical", vals=0)], ooc)
        self.assertEqual((s["cases"]["in_contract"], s["cases"]["compared"], s["cases"]["out_of_contract"]), (2, 2, 36))

    def test_blocked_when_the_harness_rendered_nothing(self):
        ooc = {"moved_out": [], "unresolved": [{"id": "a"}], "counts": {"out_of_contract_total": 35, "unresolved": 1, "candidates_promote_review": 0, "status_counts": {"status:not_run": 1}}}
        s = self.build([graded("a", "no_reference")], ooc)
        self.assertEqual(s["status"], "blocked")

    def verified_inputs(self, n=150):
        g = [graded("c%d" % i, "identical", vals=0, lp=i % 2 == 0) for i in range(n)]
        kills = {"totals": {"total": 3, "killed": 3, "survivors": [], "undecided": []}}
        calls = {"cases": [{"id": "c%d" % i, "status": "match"} for i in range(5)]}
        return g, kills, calls

    def test_verified_needs_everything(self):
        g, kills, calls = self.verified_inputs()
        s = self.build(g, None, kills, calls, n=150)
        self.assertEqual(s["status"], "verified")
        self.assertEqual(s["blockers"], [])
        self.assertEqual(pb.unmet(s), [])
        g2, k2, c2 = self.verified_inputs(149)
        self.assertNotEqual(self.build(g2, None, k2, c2, n=149)["status"], "verified")
        k3 = {"totals": {"total": 3, "killed": 2, "survivors": ["M3"], "undecided": []}}
        s3 = self.build(g, None, k3, calls, n=150)
        self.assertEqual(s3["status"], "in_progress")
        self.assertIn("mutant survivors: M3", s3["blockers"])
        c4 = {"cases": calls["cases"] + [{"id": "c9", "status": "mismatch"}]}
        self.assertEqual(self.build(g, None, kills, c4, n=150)["status"], "in_progress")
        self.assertEqual(self.build(g, None, kills, None, n=150)["status"], "in_progress")
        g5 = list(g)
        g5[0] = graded("c0", "differing", vals=1)
        self.assertEqual(self.build(g5, None, kills, calls, n=150)["status"], "in_progress")

    def test_call_records_of_moved_out_cases_do_not_count(self):
        g, kills, calls = self.verified_inputs()
        ooc = {"moved_out": [{"id": "c1"}], "counts": {"out_of_contract_total": 36, "unresolved": 0, "candidates_promote_review": 0, "status_counts": {}}}
        s = self.build(g, ooc, kills, calls, n=150)
        self.assertEqual(dict(s["call_records"]), {"compared": 4, "matching": 4})


class Ladder(unittest.TestCase):
    def test_rungs(self):
        with tempfile.TemporaryDirectory() as d:
            ts, probe = os.path.join(d, "ts"), os.path.join(d, "probe")
            os.makedirs(os.path.join(ts, "inputs"))
            for f in ("in.8", "m1", "m2"):
                open(os.path.join(ts, "inputs", f), "wb").write(b"\0" * 16)
            c = case("orig", params=["speed_percent=50", "fg1_inv_matte_shrink=2"], layers={"fg1_matte": "inputs/m1", "fg2_matte": "inputs/m2"})
            c["input"] = "inputs/in.8"
            rungs = pb.write_ladder(c, ts, probe)
            names = [r["rung"] for r in rungs]
            self.assertEqual(names, ["orig", "all-default", "no-layers", "revert-speed_percent", "revert-fg1_inv_matte_shrink",
                                     "drop-fg1_matte", "drop-fg2_matte"])
            # with two settings, reverting one is the same case as setting the other alone
            self.assertEqual([x["rung"] for x in rungs[3]["also"]], ["only-fg1_inv_matte_shrink"])
            self.assertEqual([x["rung"] for x in rungs[4]["also"]], ["only-speed_percent"])
            by = {r["rung"]: r["case"] for r in rungs}
            self.assertEqual(by["revert-speed_percent"]["params"], ["fg1_inv_matte_shrink=2"])
            self.assertEqual(by["all-default"]["params"], [])
            self.assertNotIn("layers", by["no-layers"])
            self.assertEqual(sorted(by["drop-fg1_matte"]["layers"]), ["fg2_matte"])
            self.assertEqual(len({r["case"]["id"] for r in rungs}), len(rungs))
            doc = json.load(open(os.path.join(probe, "cases.json")))
            self.assertEqual(doc["format"], 1)
            self.assertTrue(os.path.exists(os.path.join(probe, "inputs", "m2")))
            self.assertEqual(by["orig"]["reference"], by["orig"]["id"] + ".rgba")

    def test_duplicate_rungs_collapse(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "ts", "inputs"))
            open(os.path.join(d, "ts", "inputs", "in.8"), "wb").write(b"\0")
            c = case("x", params=["speed_percent=50"])
            c["input"] = "inputs/in.8"
            names = [r["rung"] for r in pb.write_ladder(c, os.path.join(d, "ts"), os.path.join(d, "p"))]
            self.assertEqual(names, ["orig", "all-default"])      # reverting the only setting is the all-default case, setting it alone is the original

    def test_interpretation(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "ts", "inputs"))
            open(os.path.join(d, "ts", "inputs", "in.8"), "wb").write(b"\0")
            c = case("x", params=["speed_percent=50", "pt1_use=2"])
            c["input"] = "inputs/in.8"
            rungs = pb.write_ladder(c, os.path.join(d, "ts"), os.path.join(d, "p"))
            st = {r["case"]["id"]: "pass" for r in rungs}
            st[next(r for r in rungs if r["rung"] == "orig")["case"]["id"]] = "FAIL"
            st[next(r for r in rungs if r["rung"] == "revert-speed_percent")["case"]["id"]] = "FAIL"      # also pt1_use alone
            lines = pb.interpret_ladder(rungs, st)
            self.assertTrue(any("pt1_use=2 alone" in x for x in lines))
            self.assertTrue(any("reverting pt1_use=2 makes it pass" in x for x in lines))


class FxParitySet(unittest.TestCase):
    def make(self, d):
        os.makedirs(os.path.join(d, "ts", "inputs"))
        for f, n in (("box2-4x2.d8.f3.rgba", 4 * 2 * 4 * 3), ("m-hard-0-5x2.d8.f3.rgba", 5 * 2 * 4 * 3), ("m-hard-0-4x2.d8.f1.rgba", 4 * 2 * 4)):
            with open(os.path.join(d, "ts", "inputs", f), "wb") as fh:
                fh.write(bytes(n))
        c = case("c", w=4, h=2, layers={"fg1_matte": "inputs/m-hard-0-5x2.d8.f3.rgba", "fg2_matte": "inputs/m-hard-0-4x2.d8.f1.rgba"})
        c["input"] = "inputs/box2-4x2.d8.f3.rgba"
        with open(os.path.join(d, "ts", "cases.json"), "w") as fh:
            json.dump({"format": 1, "ffmpeg": "none", "cases": [c]}, fh)
        return c

    def test_input_entries(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.make(d)
            ents = {e["file"]: e for e in pb.input_entries([c, c], os.path.join(d, "ts"))}
            self.assertEqual(len(ents), 3)
            m = ents["inputs/m-hard-0-5x2.d8.f3.rgba"]
            self.assertEqual((m["width"], m["height"], m["depth"], m["frames"], m["pix_fmt"], m["source"]), (5, 2, 8, 3, "rgba", "matte"))
            self.assertEqual(ents["inputs/m-hard-0-4x2.d8.f1.rgba"]["frames"], 1)
            self.assertEqual(ents["inputs/box2-4x2.d8.f3.rgba"]["source"], "box2")

    def test_the_original_cases_json_is_left_alone(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d)
            ts = os.path.join(d, "ts")
            before = pb.read_text(os.path.join(ts, "cases.json"))
            out = pb.ensure_fxparity_set(ts, os.path.join(d, "res", "gradeset"))
            self.assertEqual(pb.read_text(os.path.join(ts, "cases.json")), before)
            doc = json.load(open(os.path.join(out, "cases.json")))
            self.assertEqual(len(doc["inputs"]), 3)
            self.assertTrue(os.path.exists(os.path.join(out, "inputs", "box2-4x2.d8.f3.rgba")))
            self.assertEqual(pb.ensure_fxparity_set(out, os.path.join(d, "x")), out)       # already has inputs


if __name__ == "__main__":
    unittest.main()
