#!/usr/bin/env python3
"""Unit tests of phase_b.py: the logic that needs neither the server nor fx-parity.

usage: python3 -m unittest test_phase_b   (from this folder)
"""
import json
import os
import struct
import sys
import tempfile
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase_b as pb  # noqa: E402


def f32(bits):
    return struct.pack("<I", bits)


class RawCompare(unittest.TestCase):
    def test_ulp_and_zero_and_nan(self):
        self.assertEqual(pb.ulp_distance_bits(0x3F800000, 0x3F800001), 1)
        self.assertEqual(pb.ulp_distance_bits(0x00000000, 0x80000000), 0)
        self.assertEqual(pb.ulp_distance_bits(0x00000001, 0x80000001), 2)
        self.assertIsNone(pb.ulp_distance_bits(0x7FC00000, 0x3F800000))
        self.assertTrue(pb.is_nan_bits(0xFFC00001))
        self.assertFalse(pb.is_nan_bits(0x7F800000))

    def test_8bit_identical_and_one_value(self):
        a = bytes(range(48))
        r = pb.compare_clip(a, a, 2, 3, 8, 2)
        self.assertEqual((r["values_compared"], r["differing_values"]), (48, 0))
        b = bytearray(a)
        b[30] ^= 4          # frame 1 (24 bytes per frame), byte 6: pixel 1, channel B
        r = pb.compare_clip(a, bytes(b), 2, 3, 8, 2)
        self.assertEqual(r["differing_values"], 1)
        self.assertEqual(r["differing_frames"], [1])
        self.assertEqual(r["first"], {"frame": 1, "channel": "B", "x": 1, "y": 0, "expected": 30, "actual": 26})
        self.assertEqual(r["max_abs_diff"], 4)

    def test_16bit(self):
        a = bytes(range(32))
        b = bytearray(a)
        b[9] ^= 1           # byte 9 is 16 bit word 4: pixel 1, channel R
        r = pb.compare_clip(a, bytes(b), 2, 2, 16, 1)
        self.assertEqual((r["differing_values"], r["first"]["channel"], r["first"]["x"], r["first"]["y"]), (1, "R", 1, 0))

    def test_32bit_planar_zero_sign_nan(self):
        w, h = 2, 2
        pix = w * h
        one = 0x3F800000
        ref = b"".join(f32(one + i) for i in range(4 * pix))
        out = bytearray(ref)
        # plane 2 (R), pixel (1,1) = index 3: bump one ULP
        i = (2 * pix + 3) * 4
        out[i:i + 4] = f32(one + 2 * pix + 3 + 1)
        r = pb.compare_clip(ref, bytes(out), w, h, 32, 1)
        self.assertEqual((r["differing_values"], r["max_ulp"], r["nan_mismatches"]), (1, 1, 0))
        self.assertEqual((r["first"]["channel"], r["first"]["x"], r["first"]["y"]), ("R", 1, 1))
        # -0 against 0
        z = f32(0) * (4 * pix)
        zz = bytearray(z)
        zz[0:4] = f32(0x80000000)
        r = pb.compare_clip(z, bytes(zz), w, h, 32, 1)
        self.assertEqual((r["differing_values"], r["max_ulp"]), (1, 0))
        # NaN payload
        n = bytearray(z)
        n[0:4] = f32(0x7FC00000)
        m = bytearray(z)
        m[0:4] = f32(0x7FC00001)
        r = pb.compare_clip(bytes(n), bytes(m), w, h, 32, 1)
        self.assertEqual((r["differing_values"], r["nan_mismatches"], r["max_ulp"]), (1, 1, None))
        # the same NaN bits are equal
        self.assertEqual(pb.compare_clip(bytes(n), bytes(n), w, h, 32, 1)["differing_values"], 0)

    def test_size_error(self):
        self.assertIn("error", pb.compare_clip(b"\0" * 7, b"\0" * 8, 1, 1, 8, 2))

    def test_values_of(self):
        c = {"frames": 3, "width": 4, "height": 2, "depth": 16}
        self.assertEqual(pb.values_of(c), ("u16", 96))


class OutOfContract(unittest.TestCase):
    PM = {
        "max_displace": {"vfx_kind": "float", "ae_min": "1", "ae_max": "8192", "match": "exact"},
        "display": {"vfx_kind": "popup", "ae_choices": "Source|Motion Vectors|Debug", "match": "exact"},
        "pt1_color": {"vfx_kind": "color", "ae_min": "0", "ae_max": "255", "match": "normalized"},
        "loose": {"vfx_kind": "float", "match": "unmatched"},
    }

    def test_ranges(self):
        rv = pb.range_violation
        self.assertIsNone(rv("max_displace=1", self.PM))
        self.assertIsNone(rv("max_displace=8192", self.PM))
        self.assertIn("minimum", rv("max_displace=0.99", self.PM))
        self.assertIn("maximum", rv("max_displace=8192.5", self.PM))
        self.assertIn("minimum", rv("max_displace=64@0;0@2", self.PM))
        self.assertIsNone(rv("display=3", self.PM))
        self.assertIn("above", rv("display=4", self.PM))
        self.assertIn("below", rv("display=0", self.PM))
        self.assertIn("color", rv("pt1_color=256,0,0", self.PM))
        self.assertIsNone(rv("loose=1e9", self.PM))
        self.assertIsNone(rv("no_such=1", self.PM))

    def test_classify(self):
        c = pb.classify_plugin
        self.assertEqual(c({"status": "rendered"})[0], "rendered")
        self.assertEqual(c({"status": "error", "error": "PluginError: RENDER returned PF_Err 25"})[0], "plugin_error")
        self.assertEqual(c({"status": "error", "error": "GuestFault: Unhandled CPU exception"})[0], "harness_fault")
        self.assertEqual(c({"status": "error", "error": "PluginError: X returned PF_Err 3"}, ["inter.foo"])[0], "harness_fault")
        self.assertEqual(c({"status": "refused"})[0], "plugin_refused")
        self.assertEqual(c({"status": "timeout"})[0], "harness_timeout")
        self.assertEqual(c({"status": "unsupported", "error": "x"})[0], "harness_unsupported")
        self.assertEqual(c({"status": "not_run"})[0], "not_run")
        self.assertEqual(c(None)[0], "no_status")

    def test_judge(self):
        grid = [{"id": "g%d" % i, "params": ["max_displace=%s" % v]} for i, v in enumerate([5, 5, 5, 0, 5])]
        cands = [{"id": "c%d" % i, "params": [], "why": "w"} for i in range(4)]
        status = {
            "g0": {"status": "error", "error": "PluginError: RENDER returned PF_Err 25"},
            "g1": {"status": "error", "error": "GuestFault: x"},
            "g2": {"status": "rendered"},
            "c0": {"status": "rendered"},
            "c1": {"status": "error", "error": "PluginError: RENDER returned PF_Err 3"},
            "c2": {"status": "unsupported", "error": "no counterpart"},
        }
        rows = {i["id"]: ("0", "1", "h", "") for i in grid}
        crow = {i["id"]: ("3", "0", "-", "refused") for i in cands}
        moved, back, unresolved = pb.judge_ooc(grid, cands, status, rows, crow, self.PM)
        ids = {m["id"]: m["reason_class"] for m in moved}
        self.assertEqual(ids["g0"], "plugin_error")
        self.assertEqual(ids["g3"], "outside_registered_range")
        self.assertNotIn("g1", ids)                      # a harness fault stays in contract
        self.assertNotIn("g2", ids)
        self.assertEqual([b["id"] for b in back], ["c0"])
        self.assertEqual(ids["c1"], "plugin_error")
        self.assertEqual(ids["c2"], "candidate_harness_unsupported")
        self.assertEqual(ids["c3"], "candidate_unjudged")
        self.assertEqual({u["id"] for u in unresolved}, {"g1", "c2"})
        self.assertEqual(next(m for m in moved if m["id"] == "c1")["rewrite"]["exit"], 3)

    def test_load_status_forms(self):
        with tempfile.TemporaryDirectory() as d:
            p1, p2, p3 = (os.path.join(d, n) for n in ("a.json", "b.json", "c.json"))
            json.dump({"cases": [{"id": "x", "status": "rendered"}]}, open(p1, "w"))
            json.dump([{"id": "y", "status": "error"}], open(p2, "w"))
            json.dump({"z": "rendered", "w": {"status": "refused"}}, open(p3, "w"))
            st = pb.load_status([p1, p2, p3])
            self.assertEqual(sorted(st), ["w", "x", "y", "z"])
            self.assertEqual(st["z"]["status"], "rendered")


def plugin_lines(frames, reads, params=(), smart=False, done="rendered", scale=2400):
    """frames: [(out_frame, time)], reads: [(out_frame, param_index, time, footage_frame, outside, layer_note)]."""
    out = [{"ev": "render_path", "smart": smart}]
    for f, t in frames:
        out.append({"ev": "frame", "out_frame": f, "time": t, "time_step": 100, "time_scale": scale})
    for f, idx, t, ff, outside, note in reads:
        if smart:
            out.append({"ev": "cb", "fn": "prerender_cb.checkout_layer", "param_index": idx, "checkout_id": 1, "time": t, "time_step": 100,
                        "time_scale": scale, "footage_frame": ff, "inside": not outside, "out_frame": f})
        else:
            out.append({"ev": "cb", "fn": "inter.checkout_param", "param_index": idx, "time": t, "time_step": 100, "time_scale": scale,
                        "footage_frame": ff, "out_frame": f, "layer": note})
    for f, idx, t in params:
        out.append({"ev": "cb", "fn": "inter.checkout_param", "param_index": idx, "time": t, "time_step": 100, "time_scale": scale,
                    "footage_frame": 0, "out_frame": f})
    if done:
        out.append({"ev": "case_done", "id": "c", "status": done})
    return out


def rewrite_lines(frames, reads, scale=2400):
    out = [{"ev": "run", "effect": "create-vectors", "time_scale": scale, "time_step": 100, "frames": len(frames)}]
    for f, t in frames:
        out.append({"ev": "frame", "out_frame": f, "time": t, "time_step": 100, "time_scale": scale})
    for f, layer, t, idx in reads:
        out.append({"ev": "read", "out_frame": f, "layer": layer, "time": t, "index": idx, "clip_frames": 3})
    return out


class Calls(unittest.TestCase):
    I2L = {2: "fg1_matte"}

    def views(self, pl, rl, provided=("fg1_matte",)):
        return pb.plugin_view(pl, self.I2L, set(provided)), pb.rewrite_view(rl)

    FR = [(0, 0), (1, 100)]

    def test_match_classic_and_smart(self):
        rl = rewrite_lines(self.FR, [(0, "input", 0, 0), (0, "input", 100, 1), (1, "input", 100, 1), (1, "input", 200, 2), (1, "fg1_matte", 100, 1)])
        for smart in (False, True):
            reads = [(0, 0, 0, 0, False, "input"), (0, 0, 100, 1, False, "input"), (1, 0, 100, 1, False, "input"), (1, 0, 200, 2, False, "input"),
                     (1, 2, 100, 1, False, "fg1_matte")]
            pv, rv = self.views(plugin_lines(self.FR, reads, smart=smart), rl)
            r = pb.compare_calls(pv, rv)
            self.assertTrue(r["match"], r)
            self.assertEqual(r["frames_compared"], 2)

    def test_outside_the_clip_is_a_null_index(self):
        rl = rewrite_lines(self.FR[:1], [(0, "input", -100, None)])
        pl = plugin_lines(self.FR[:1], [(0, 0, -100, -1, True, "input (outside the clip: transparent)")])
        pv, rv = self.views(pl, rl)
        self.assertTrue(pb.compare_calls(pv, rv)["match"])
        pl = plugin_lines(self.FR[:1], [(0, 0, -100, -1, True, "x")], smart=True)
        pv, rv = self.views(pl, rl)
        self.assertTrue(pb.compare_calls(pv, rv)["match"])

    def test_mismatch_kinds_and_first_frame(self):
        rl = rewrite_lines(self.FR, [(0, "input", 0, 0), (1, "input", 100, 1), (1, "input", 200, 2)])
        base = [(0, 0, 0, 0, False, "input"), (1, 0, 100, 1, False, "input"), (1, 0, 200, 2, False, "input")]
        # plugin lacks a read the rewrite makes
        pv, rv = self.views(plugin_lines(self.FR, base[:2]), rl)
        r = pb.compare_calls(pv, rv)
        self.assertFalse(r["match"])
        self.assertEqual(r["first_diff"]["out_frame"], 1)
        self.assertEqual(r["first_diff"]["reads"]["extra_in_rewrite"], [["input", "1/12", 2]])
        # plugin makes a read the rewrite does not
        pv, rv = self.views(plugin_lines(self.FR, base + [(0, 0, 100, 1, False, "input")]), rl)
        r = pb.compare_calls(pv, rv)
        self.assertEqual(r["first_diff"]["out_frame"], 0)
        self.assertEqual(r["first_diff"]["reads"]["missing_in_rewrite"], [["input", "1/24", 1]])
        # wrong time, same clip frame
        bad = list(base)
        bad[2] = (1, 0, 210, 2, False, "input")
        pv, rv = self.views(plugin_lines(self.FR, bad), rl)
        r = pb.compare_calls(pv, rv)
        self.assertEqual(r["first_diff"]["reads"]["wrong_time"][0]["layer"], "input")
        self.assertFalse(r["first_diff"]["reads"]["missing_in_rewrite"])
        # wrong clip frame, same time
        bad = list(base)
        bad[2] = (1, 0, 200, 3, False, "input")
        pv, rv = self.views(plugin_lines(self.FR, bad), rl)
        self.assertEqual(pb.compare_calls(pv, rv)["first_diff"]["reads"]["wrong_index"][0]["plugin_index"], 3)
        # a frame the plugin never started
        pv, rv = self.views(plugin_lines(self.FR[:1], base[:1]), rl)
        self.assertIn("frame", pb.compare_calls(pv, rv)["first_diff"])

    def test_order_and_unprovided_layers(self):
        rl = rewrite_lines(self.FR[:1], [(0, "input", 0, 0), (0, "input", 100, 1)])
        rev = [(0, 0, 100, 1, False, "input"), (0, 0, 0, 0, False, "input")]
        pv, rv = self.views(plugin_lines(self.FR[:1], rev), rl)
        self.assertTrue(pb.compare_calls(pv, rv)["match"])
        pv, rv = self.views(plugin_lines(self.FR[:1], rev), rl)
        self.assertFalse(pb.compare_calls(pv, rv, ordered=True)["match"])
        # a checkout of a layer without a clip is dropped (classic: note none; smart: not provided)
        extra = rev + [(0, 2, 0, 0, False, "none")]
        pv, rv = self.views(plugin_lines(self.FR[:1], extra), rl)
        self.assertEqual(pv["dropped"], 1)
        self.assertTrue(pb.compare_calls(pv, rv)["match"])
        extra = [(0, 0, 0, 0, False, "input"), (0, 0, 100, 1, False, "input"), (0, 2, 0, 0, False, "x")]
        pv, rv = self.views(plugin_lines(self.FR[:1], extra, smart=True), rl, provided=())
        self.assertEqual(pv["dropped"], 1)
        self.assertTrue(pb.compare_calls(pv, rv)["match"])

    def test_param_times_are_reported_not_compared(self):
        rl = rewrite_lines(self.FR[:1], [(0, "input", 0, 0)])
        pl = plugin_lines(self.FR[:1], [(0, 0, 0, 0, False, "input")], params=[(0, 20, 0), (0, 20, 100)])
        pv, rv = self.views(pl, rl)
        r = pb.compare_calls(pv, rv)
        self.assertTrue(r["match"])
        self.assertFalse(r["param_times_compared"])
        self.assertEqual(pb.plugin_param_times(pv), {"0": ["0", "1/24"]})
        rl.append({"ev": "param", "out_frame": 0, "param": "param#20", "time": 0})
        pv, rv = self.views(pl, rl)
        r = pb.compare_calls(pv, rv)
        self.assertFalse(r["match"])
        self.assertTrue(r["first_diff"]["params"]["missing_in_rewrite"])

    def test_calls_check_files(self):
        with tempfile.TemporaryDirectory() as d:
            pdir, rdir = os.path.join(d, "p"), os.path.join(d, "r")
            os.makedirs(pdir)
            os.makedirs(rdir)
            cases = [{"id": n, "layers": {}} for n in ("ok", "bad", "nop", "err", "nor")]
            rl = rewrite_lines(self.FR[:1], [(0, "input", 0, 0)])
            good = plugin_lines(self.FR[:1], [(0, 0, 0, 0, False, "input")])
            bad = plugin_lines(self.FR[:1], [(0, 0, 0, 1, False, "input")])

            def wr(dirp, name, lines):
                with open(os.path.join(dirp, name + ".jsonl"), "w") as fh:
                    for l in lines:
                        fh.write(json.dumps(l) + "\n")
                    fh.write("not json\n")

            for n in ("ok", "bad", "err", "nop"):
                wr(rdir, n, rl)
            wr(pdir, "ok", good)
            wr(pdir, "bad", bad)
            wr(pdir, "err", plugin_lines(self.FR[:1], [], done="error"))
            wr(pdir, "nor", good)
            res = {r["id"]: r for r in pb.calls_check(cases, pdir, rdir, self.I2L)}
            self.assertTrue(res["ok"]["compared"] and res["ok"]["match"])
            self.assertEqual(res["ok"]["unparsable_lines"], {"plugin": 1, "rewrite": 1})
            self.assertTrue(res["bad"]["compared"] and not res["bad"]["match"])
            self.assertEqual(res["bad"]["first_diff"]["out_frame"], 0)
            self.assertFalse(res["nop"]["compared"])
            self.assertFalse(res["err"]["compared"])
            self.assertEqual(res["nor"]["reason"], "no rewrite record")


class Kills(unittest.TestCase):
    def setUp(self):
        self.base = {c: ("0", "10", "h" + c) for c in "abcdef"}
        self.refs = {"a": "ha", "b": "hb", "c": "hc", "d": "hd", "e": None, "f": "ZZ"}     # e: no reference; f: base differs
        self.mutants = [{"id": "m%d" % i, "kind": "k"} for i in range(1, 7)]
        same = dict(self.base)
        self.rows = {
            "m1": dict(same, a=("0", "10", "x")),                   # changes a (candidate): killed
            "m2": dict(same),                                       # changes nothing: survivor
            "m3": dict(same, e=("0", "10", "x"), f=("0", "10", "x")),   # changes only no-ref and base-differs cases: survivor
            "m4": dict(same, d=("1", "0", "-")),                    # exit code change on a candidate: killed
            "m5": None,                                             # not run
            "m6": dict(same, c=("0", "10", "x")),                   # changes c, which is out of contract below
        }

    def test_definition(self):
        cand, excl, res = pb.compute_kills(self.base, self.refs, self.mutants, self.rows, out_of_contract=["c"])
        self.assertEqual(cand, ["a", "b", "d"])
        self.assertEqual({k: len(v) for k, v in excl.items()}, {"out_of_contract": 1, "no_reference": 1, "base_differs": 1})
        st = {r["id"]: r["status"] for r in res}
        self.assertEqual(st, {"m1": "killed", "m2": "survivor", "m3": "survivor", "m4": "killed", "m5": "not_run", "m6": "survivor"})
        r = {r["id"]: r for r in res}
        self.assertEqual(r["m1"]["killed_by"], ["a"])
        self.assertIn("no output", r["m2"]["reason"])
        self.assertIn("only on cases", r["m3"]["reason"])
        self.assertEqual(r["m3"]["changed_vs_base"], 2)
        self.assertEqual(r["m3"]["changed_in_candidates"], 0)

    def test_without_exclusions_m6_kills(self):
        cand, _, res = pb.compute_kills(self.base, self.refs, self.mutants, self.rows)
        self.assertEqual({r["id"]: r["status"] for r in res}["m6"], "killed")


def grade_row(i, status, depth=8, ulp=None, dv=0, error=None, standin=None, nan=0, w=4, h=3, frames=3, params=(), group="g", layers=()):
    unit, n = pb.values_of({"frames": frames, "width": w, "height": h, "depth": depth})
    return {"id": i, "status": status, "error": error, "depth": depth, "width": w, "height": h, "frames": frames, "fps": "24", "source": "s",
            "params": list(params), "layers": list(layers), "group": group, "tags": [], "standin": standin, "logic_parity": bool(standin),
            "unit": unit, "values_total": n, "values_compared": 0 if status == "no reference" or error else n,
            "differing_values": dv, "differing_pixels": 0, "failing_frames": 0, "differing_frame_list": [], "max_ulp": ulp,
            "max_ulp_fxparity": ulp, "nan_mismatches": nan, "max_abs_diff": None, "first_diff": None, "ref_sha256": None}


class Summary(unittest.TestCase):
    def data(self, n=200, differ=0):
        cases = [{"id": "c%03d" % i} for i in range(n)]
        rows = []
        for i, c in enumerate(cases):
            depth = (8, 16, 32)[i % 3]
            st = "FAIL" if i < differ else "pass"
            rows.append(grade_row(c["id"], st, depth=depth, ulp=(3 if depth == 32 and st == "FAIL" else (0 if depth == 32 else None)),
                                  dv=(2 if st == "FAIL" else 0), standin=("keyed" if i in (5, 6) else None)))
        meta = {c["id"]: {"standin": "keyed" if i in (5, 6) else None} for i, c in enumerate(cases)}
        mutants = [{"id": "m1"}, {"id": "m2"}]
        return {"ident": {"module": "create-vectors", "plugin": "p", "plugin_sha256": "s", "effect": "create-vectors"},
                "cases": cases, "meta": meta, "candidates": [{"id": "o1"}, {"id": "o2"}], "mutants": mutants,
                "grade": {"engine": "fx-parity", "cases": rows, "cross_check": {"disagreements": []}},
                "ooc": None, "kills": None, "calls": None, "fixes": ["f1"]}

    def full(self, n=200, differ=0):
        d = self.data(n, differ)
        d["ooc"] = {"counts": {"candidates_unjudged": 0}, "moved_out": [{"id": "o1", "origin": "candidate"}, {"id": "o2", "origin": "candidate"}],
                    "candidates_back_in_contract": [], "unresolved": []}
        d["kills"] = {"stale": False, "warnings": [], "totals": {"killed": 2, "survivors": [], "not_run": []}}
        d["calls"] = {"cases": [{"id": c["id"], "compared": True, "match": True} for c in d["cases"]]}
        return d

    def test_counts_are_computed(self):
        s = pb.compute_summary(self.data(200, 3), date="2026-10-30")
        self.assertEqual(s["cases"], {"in_contract": 200, "compared": 200, "identical": 197, "differing": 3, "out_of_contract": 2, "logic_parity": 2})
        self.assertEqual(s["status"], "in_progress")
        self.assertEqual(s["mutants"], {"total": 2, "killed": 0, "survivors": []})
        self.assertEqual(s["call_records"], {"compared": 0, "matching": 0})
        self.assertEqual(s["values"]["differing_values"], 6)
        self.assertEqual(s["values"]["max_ulp"], 3)
        self.assertEqual(s["values"]["bytes_compared"], (67 + 67) * 36 * 1 // 1 * 0 + sum(r["values_compared"] * pb.SAMPLE[r["depth"]] for r in self.data(200, 3)["grade"]["cases"] if r["depth"] != 32))
        self.assertEqual(s["fixes"], ["f1"])
        self.assertEqual(list(s), ["module", "plugin", "plugin_sha256", "effect", "status", "blockers", "cases", "values", "call_records", "mutants",
                                   "stand_ins", "fixes", "commands", "measured_at"])
        self.assertTrue(any("differ" in b for b in s["blockers"]))

    def test_verified_gate(self):
        self.assertEqual(pb.compute_summary(self.full())["status"], "verified")
        self.assertEqual(pb.compute_summary(self.full(149))["status"], "in_progress")        # fewer than 150 compared
        self.assertEqual(pb.compute_summary(self.full(200, 1))["status"], "in_progress")     # one differs
        d = self.full()
        d["kills"]["totals"].update(killed=1, survivors=["m2"])
        s = pb.compute_summary(d)
        self.assertEqual((s["status"], s["mutants"]["survivors"]), ("in_progress", ["m2"]))
        d = self.full()
        d["calls"]["cases"][0]["match"] = False
        self.assertEqual(pb.compute_summary(d)["status"], "in_progress")
        d = self.full()
        d["kills"]["stale"] = True
        self.assertEqual(pb.compute_summary(d)["status"], "in_progress")
        d = self.full()
        d["ooc"]["counts"]["candidates_unjudged"] = 1
        self.assertEqual(pb.compute_summary(d)["status"], "in_progress")
        d = self.full()
        d["grade"]["engine"] = "python-fallback"
        self.assertEqual(pb.compute_summary(d)["status"], "in_progress")
        d = self.full()
        self.assertEqual(pb.compute_summary(d, status_override="blocked")["status"], "blocked")
        self.assertEqual(pb.compute_summary(self.data(), status_override="verified")["status"], "in_progress")

    def test_no_reference_and_ooc_moves(self):
        d = self.full()
        for r in d["grade"]["cases"][:10]:
            r.update(status="no reference", values_compared=0)
        d["ooc"]["moved_out"].append({"id": "c150", "origin": "grid"})
        d["ooc"]["moved_out"].append({"id": "c151", "origin": "grid"})
        s = pb.compute_summary(d)
        self.assertEqual(s["cases"]["in_contract"], 198)
        self.assertEqual(s["cases"]["compared"], 188)
        self.assertEqual(s["cases"]["out_of_contract"], 4)
        self.assertEqual(s["call_records"]["compared"], 198)
        self.assertEqual(s["status"], "in_progress")
        self.assertTrue(any("no reference" in b for b in s["blockers"]))

    def test_unresolved_and_back_in_block(self):
        d = self.full()
        d["ooc"]["unresolved"] = [{"id": "x"}]
        self.assertEqual(pb.compute_summary(d)["status"], "in_progress")
        d = self.full()
        d["ooc"]["candidates_back_in_contract"] = [{"id": "o9"}]
        s = pb.compute_summary(d)
        self.assertEqual(s["status"], "in_progress")
        self.assertTrue(any("not compared yet" in b for b in s["blockers"]))
        d["extra_grades"] = [{"cases": [grade_row("o9", "pass")]}]
        s = pb.compute_summary(d)
        self.assertEqual(s["cases"]["in_contract"], 201)
        self.assertEqual(s["status"], "verified")

    def test_cmd_summary_refuses_without_grade(self):
        with tempfile.TemporaryDirectory() as d:
            class A:
                root = pb.ROOT
                grid = os.path.join(pb.ROOT, "grid")
                phase_a = None
                results = d
            ctx = pb.Ctx(A)
            with self.assertRaises(pb.ToolError):
                pb.cmd_summary(ctx, type("N", (), {"grade_name": "grade", "calls_name": "calls", "extra_grade": None, "fix": None, "status": None, "date": None}))


class Triage(unittest.TestCase):
    def rows(self):
        out = []
        for i in range(8):
            st = "FAIL" if i % 2 == 0 and i < 6 else "pass"
            out.append(grade_row("r%d" % i, st, depth=(8, 32)[i % 2], params=["max_displace=64"] if i % 2 == 0 else [], w=4 + i, h=3))
        return out

    def test_cooccurrence(self):
        tot, bad = pb.cooccurrence(self.rows())
        self.assertEqual(tot[("param", "max_displace=64")], 4)
        self.assertEqual(bad[("param", "max_displace=64")], 3)
        self.assertEqual(bad[("depth", "d8")], 3)
        self.assertEqual(bad.get(("depth", "d32"), 0), 0)
        self.assertEqual(tot[("param-key", "(all defaults)")], 4)

    def test_smallest_first(self):
        fails = [r for r in self.rows() if r["status"] == "FAIL"]
        self.assertEqual(sorted(fails, key=pb.smallest_key)[0]["id"], "r0")

    def test_ladder(self):
        inputs = [{"file": "inputs/%s-%dx%d.d%d.f3.%s" % (s, 4, 3, d, pb.PIXFMT[d]), "source": s, "width": 4, "height": 3, "depth": d, "pix_fmt": pb.PIXFMT[d], "frames": 3}
                  for s in ("shift", "disc", "matte1") for d in (8, 16, 32)]
        grid = {"format": 1, "ffmpeg": "x", "inputs": inputs}
        case = {"id": "create-vectors.t.shift-4x3.d8.abcd1234", "effect": "create-vectors", "label": "t", "params": ["max_displace=64", "display=1", "pt1_use=5"],
                "depth": 8, "width": 4, "height": 3, "source": "shift", "input": "inputs/shift-4x3.d8.f3.rgba",
                "reference": "create-vectors.t.shift-4x3.d8.abcd1234.rgba", "kind": "temporal", "frames": 3, "fps": "24",
                "layers": {"fg1_matte": "inputs/matte1-4x3.d8.f3.rgba"}}
        meta = {"layer_sources": {"fg1_matte": "matte1"}}
        with tempfile.TemporaryDirectory() as d:
            probe = os.path.join(d, "probe")
            os.makedirs(probe)
            cases, rungs, ins = pb.build_ladder(case, meta, grid, probe, os.path.join(d, "grid"), ["shift", "disc"])
        kinds = [r["kind"] for r in rungs]
        names = [r["name"] for r in rungs]
        self.assertEqual(names[:2], ["base", "default"])
        self.assertEqual(kinds.count("revert"), 3)
        self.assertEqual(kinds.count("alone"), 3)
        self.assertIn("rev-layer-fg1_matte", names)
        self.assertIn("alone-layer-fg1_matte", names)
        self.assertIn("src-disc", names)
        self.assertIn("depth-16", names)
        self.assertIn("depth-32", names)
        by = {r["name"]: c for r, c in zip(rungs, cases)}
        self.assertEqual(by["rev-display"]["params"], ["max_displace=64", "pt1_use=5"])
        self.assertEqual(by["alone-display"]["params"], ["display=1"])
        self.assertNotIn("layers", by["alone-display"])
        self.assertEqual(by["default"]["params"], [])
        self.assertEqual(by["depth-32"]["reference"].rsplit(".", 1)[1], "gbrapf32le")
        self.assertTrue(by["depth-32"]["layers"]["fg1_matte"].endswith("matte1-4x3.d32.f3.gbrapf32le"))
        self.assertTrue(by["base"]["input"].startswith("../grid/"))
        self.assertEqual(len({c["id"] for c in cases}), len(cases))
        self.assertTrue(all(c["reference"].startswith(c["id"] + ".") for c in cases))
        self.assertTrue(all(i["file"].startswith("../grid/") for i in ins))

    def test_ladder_collapses_duplicates(self):
        grid = {"format": 1, "inputs": []}
        case = {"id": "create-vectors.t.shift-4x3.d8.abcd1234", "effect": "create-vectors", "label": "t", "params": ["display=1"], "depth": 8, "width": 4,
                "height": 3, "source": "shift", "input": "inputs/a", "reference": "x.rgba", "kind": "temporal", "frames": 3, "fps": "24"}
        with tempfile.TemporaryDirectory() as d:
            cases, rungs, _ = pb.build_ladder(case, None, grid, os.path.join(d, "p"), os.path.join(d, "g"))
        self.assertEqual([r["name"] for r in rungs], ["base", "default"])
        self.assertIn("rev-display", rungs[1]["also"])
        self.assertIn("alone-display", rungs[0]["also"])


class Misc(unittest.TestCase):
    def test_param_values(self):
        self.assertEqual(pb.parse_value_list("16@0;64@2"), ["16", "64"])
        self.assertEqual(pb.parse_value_list("3,3@0;9,6@2"), ["3,3", "9,6"])
        self.assertEqual(pb.parse_value_list("5"), ["5"])

    def test_vfx_args(self):
        c = {"effect": "create-vectors", "width": 4, "height": 3, "depth": 8, "params": ["a=1"], "fps": "24", "frames": 3,
             "layers": {"fg1_matte": "inputs/m"}, "input": "inputs/i"}
        a = pb.vfx_args("vfx", c, "ts")
        self.assertEqual(a[:2], ["vfx", "apply"])
        self.assertIn("--param", a)
        self.assertEqual(a[a.index("--layer") + 1], "fg1_matte=" + os.path.join("ts", "inputs/m"))
        self.assertEqual(a[-2:], ["--in", os.path.join("ts", "inputs/i")])

    def test_no_dashes_in_the_tool(self):
        for name in ("phase_b.py", "test_phase_b.py"):
            with open(os.path.join(pb.ROOT, name), encoding="utf-8") as fh:
                text = fh.read()
            self.assertNotIn(chr(0x2014), text)
            self.assertNotIn(chr(0x2013), text)
            self.assertTrue(text.isascii())


if __name__ == "__main__":
    unittest.main()
