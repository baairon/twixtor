#!/usr/bin/env python3
"""Unit tests of phase_b.py: the logic that needs neither the server nor a plugin.

python3 test_phase_b.py        (or: python3 -m unittest test_phase_b)
"""
import json
import os
import stat
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase_b as pb  # noqa: E402


def f32bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def words(*bits):
    return b"".join(struct.pack("<I", b) for b in bits)


class ValueDiff(unittest.TestCase):
    def test_identical(self):
        d = pb.value_diff(8, b"abcd", b"abcd")
        self.assertEqual((d["differing"], d["max_ulp"], d["first_byte"]), (0, None, None))

    def test_one_bit_at_8_and_16(self):
        d = pb.value_diff(8, bytes([1, 2, 3, 4]), bytes([1, 2, 3, 5]))
        self.assertEqual((d["differing"], d["values"], d["first_byte"]), (1, 4, 3))
        d = pb.value_diff(16, bytes(8), bytes([0, 0, 0, 0, 1, 0, 0, 0]))
        self.assertEqual((d["differing"], d["values"]), (1, 4))

    def test_negative_zero_is_one_differing_value_with_ulp_zero(self):
        d = pb.value_diff(32, words(0x80000000, 0x3F800000), words(0, 0x3F800000))
        self.assertEqual((d["differing"], d["max_ulp"], d["nan_mismatch"]), (1, 0, 0))

    def test_nan_payload_is_a_nan_mismatch(self):
        d = pb.value_diff(32, words(0x7FC00000), words(0x7FC00001))
        self.assertEqual((d["differing"], d["nan_mismatch"]), (1, 1))
        d = pb.value_diff(32, words(0x7FC00000), words(0x3F800000))
        self.assertEqual((d["differing"], d["nan_mismatch"]), (1, 1))

    def test_ulp_distance_across_zero_and_neighbours(self):
        one = f32bits(1.0)
        d = pb.value_diff(32, words(one), words(one + 3))
        self.assertEqual(d["max_ulp"], 3)
        d = pb.value_diff(32, words(0x00000001), words(0x80000001))
        self.assertEqual(d["max_ulp"], 2)

    def test_size_mismatch_counts_everything(self):
        d = pb.value_diff(8, b"abc", b"abcd")
        self.assertTrue(d["size_mismatch"])
        self.assertEqual(d["differing"], 4)


class Kills(unittest.TestCase):
    def setUp(self):
        self.ids = ["a", "b", "c", "d"]
        self.base = {c: ("0", "10", "h" + c) for c in self.ids}
        self.ref = {c: ("10", "h" + c) for c in self.ids}
        self.ctl = dict(self.base)

    def run_kills(self, ref=None, ctl=None, **muts):
        return pb.compute_kills(self.ids, ref if ref is not None else self.ref, self.base,
                                ctl if ctl is not None else self.ctl, muts)

    def test_a_mutant_that_changes_a_discriminating_case_is_killed(self):
        E, r = self.run_kills(m1=dict(self.base, b=("0", "10", "x")))
        self.assertEqual(E, self.ids)
        self.assertEqual(r["m1"]["killing"], ["b"])

    def test_a_mutant_that_changes_nothing_survives(self):
        _, r = self.run_kills(m1=dict(self.base))
        self.assertEqual((r["m1"]["killing"], r["m1"]["changed_total"]), ([], 0))

    def test_cases_where_the_base_differs_from_the_reference_cannot_kill(self):
        ref = dict(self.ref, b=("10", "other"))
        E, r = self.run_kills(ref=ref, m1=dict(self.base, b=("0", "10", "x")))
        self.assertNotIn("b", E)
        self.assertEqual(r["m1"]["killing"], [])
        self.assertEqual((r["m1"]["changed_total"], r["m1"]["changed_outside_discriminating"]), (1, 1))

    def test_cases_without_a_reference_cannot_kill(self):
        ref = {c: v for c, v in self.ref.items() if c != "c"}
        E, r = self.run_kills(ref=ref, m1=dict(self.base, c=("0", "10", "x")))
        self.assertEqual(E, ["a", "b", "d"])
        self.assertEqual(r["m1"]["killing"], [])

    def test_a_difference_the_build_profile_makes_alone_is_not_a_kill(self):
        ctl = dict(self.base, d=("0", "10", "profile"))
        _, r = self.run_kills(ctl=ctl, m1=dict(self.base, d=("0", "10", "profile")))
        self.assertEqual(r["m1"]["killing"], [])

    def test_a_mutant_that_equals_the_reference_is_not_killed_by_the_control_alone(self):
        ctl = dict(self.base, d=("0", "10", "profile"))
        _, r = self.run_kills(ctl=ctl, m1=dict(self.base))
        self.assertEqual(r["m1"]["killing"], [])

    def test_a_crashing_mutant_is_killed(self):
        _, r = self.run_kills(m1=dict(self.base, a=("101", "0", "")))
        self.assertEqual((r["m1"]["killing"], r["m1"]["exit_changes"]), (["a"], 1))

    def test_cases_the_mutant_did_not_run_are_not_kills(self):
        part = {c: v for c, v in self.base.items() if c != "a"}
        _, r = self.run_kills(m1=part)
        self.assertEqual((r["m1"]["killing"], r["m1"]["not_run"]), ([], 1))


class Ooc(unittest.TestCase):
    def test_classes(self):
        k = pb.classify_status
        self.assertEqual(k({"status": "rendered"}), "rendered")
        self.assertEqual(k({"status": "refused"}), "refused")
        self.assertEqual(k({"status": "error", "error": "PluginError: RENDER returned PF_Err 3"}), "plugin_error")
        self.assertEqual(k({"status": "error", "error": "GuestFault: unmapped read"}), "emulator_error")
        self.assertEqual(k({"status": "timeout"}), "timeout")
        self.assertEqual(k({"status": "unsupported"}), "unsupported")
        self.assertEqual(k({"status": "not_run"}), "not_run")

    def test_judge_moves_only_the_original_side_failures(self):
        cases = [{"id": "c%d" % i, "label": "l", "params": []} for i in range(6)]
        status = {"c0": {"status": "rendered"}, "c1": {"status": "refused"},
                  "c2": {"status": "error", "error": "PluginError: x"}, "c3": {"status": "error", "error": "GuestFault: y"},
                  "c4": {"status": "unsupported"}, "c5": {"status": "timeout"}}
        moved, unjudged, rendered = pb.judge_cases(cases, status, set(pb.DEFAULT_OOC))
        self.assertEqual([m["id"] for m in moved], ["c1", "c2", "c5"])
        self.assertEqual([u["id"] for u in unjudged], ["c3", "c4"])
        self.assertEqual(rendered, ["c0"])
        moved, _, _ = pb.judge_cases(cases, status, {"refused", "plugin_error", "timeout", "emulator_error", "unsupported"})
        self.assertEqual(len(moved), 5)

    def test_status_file_formats(self):
        with tempfile.TemporaryDirectory() as d:
            files = []
            for n, doc in enumerate([{"cases": [{"id": "a", "status": "rendered"}]}, [{"id": "b", "status": "refused"}],
                                     {"c": {"status": "error"}}, {"d": "timeout"}, {"cases": {"e": {"status": "rendered"}}}]):
                p = os.path.join(d, "s%d.json" % n)
                with open(p, "w") as fh:
                    json.dump(doc, fh)
                files.append(p)
            st = pb.load_status(files)
        self.assertEqual({k: v["status"] for k, v in st.items()},
                         {"a": "rendered", "b": "refused", "c": "error", "d": "timeout", "e": "rendered"})

    def test_candidates(self):
        cc = [{"id": "o1", "candidate": "speed above max", "params": ["vectors=1", "speed_percent=10001"]},
              {"id": "o2", "candidate": "x", "params": ["vectors=1", "frame_interp=4"]},
              {"id": "o3", "candidate": "y", "params": ["vectors=1", "speed_percent=5"]}]
        status = {"o1": {"status": "refused"}, "o2": {"status": "rendered"}, "o3": {"status": "rendered"}}
        out = pb.judge_candidates(cc, status, {}, set(pb.DEFAULT_OOC), {"speed_percent": (-10000.0, 10000.0)})
        self.assertEqual([c["verdict"] for c in out], ["out_of_contract", "review", "review"])
        out = pb.judge_candidates(cc[:1] + cc[2:], {"o3": {"status": "rendered"}}, {}, set(), {"speed_percent": (0.0, 3.0)})
        self.assertEqual(out[1]["verdict"], "out_of_contract")
        self.assertEqual(out[0]["verdict"], "out_of_contract")
        self.assertEqual(out[0]["plugin"], {"status": "not run"})


def rewrite_lines(frames=4, scale=2400, step=100):
    lines = [{"ev": "run", "effect": "retime", "width": 2, "height": 2, "depth": 8, "fps": "24", "time_scale": scale,
              "time_step": step, "frames": frames, "params": []}]
    for n in range(frames):
        lines.append({"ev": "frame", "out_frame": n, "time": n * step, "time_scale": scale, "time_step": step})
        for idx in (n, n + 1):
            lines.append({"ev": "read", "out_frame": n, "layer": "input", "time": idx * step,
                          "index": idx if idx < frames else None, "clip_frames": frames})
    return lines


class Calls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rec = os.path.join(self.tmp.name, "rec")
        self.plug = os.path.join(self.tmp.name, "plug")
        os.makedirs(self.rec)
        os.makedirs(self.plug)
        self.lines = rewrite_lines()
        pb.write_jsonl(os.path.join(self.rec, "c.jsonl"), self.lines)
        self.pl = pb.plugin_lines_from_rewrite(self.lines, "c")

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, pl=None, **kw):
        pb.write_jsonl(os.path.join(self.plug, "c.jsonl"), pl if pl is not None else self.pl)
        return pb.check_calls(["c"], self.plug, self.rec, **kw)[0]

    def reads(self):
        return [i for i, l in enumerate(self.pl) if l.get("ev") == "cb"]

    def test_matching_including_a_read_outside_the_clip(self):
        r = self.check()
        self.assertEqual(r["state"], "match")
        self.assertTrue(any(l.get("footage_frame") == 4 and not l["inside"] for l in self.pl))

    def test_missing_read(self):
        i = next(i for i in self.reads() if self.pl[i]["out_frame"] == 2)
        del self.pl[i]
        r = self.check()
        self.assertEqual((r["state"], r["first_out_frame"], r["kind"]), ("mismatch", 2, "extra_read"))

    def test_extra_read(self):
        i = next(i for i in self.reads() if self.pl[i]["out_frame"] == 1)
        self.pl.insert(i, dict(self.pl[i], footage_frame=3, inside=True))
        r = self.check()
        self.assertEqual((r["state"], r["first_out_frame"], r["kind"]), ("mismatch", 1, "missing_read"))

    def test_wrong_time(self):
        i = next(i for i in self.reads() if self.pl[i]["out_frame"] == 3)
        self.pl[i]["time"] += 1
        r = self.check()
        self.assertEqual((r["state"], r["first_out_frame"], r["kind"]), ("mismatch", 3, "wrong_time"))

    def test_wrong_frame_time_and_missing_frame(self):
        for l in self.pl:
            if l.get("ev") == "frame" and l["out_frame"] == 2:
                l["time"] += 5
        self.assertEqual(self.check()["kind"], "wrong_frame_time")
        pl = [l for l in pb.plugin_lines_from_rewrite(self.lines, "c") if not (l.get("ev") in ("frame", "cb") and l.get("out_frame") == 3)]
        r = self.check(pl)
        self.assertEqual((r["kind"], r["first_out_frame"]), ("extra_frame", 3))

    def test_sets_ignore_order_and_ordered_mode_does_not(self):
        a, b = [i for i in self.reads() if self.pl[i]["out_frame"] == 1]
        self.pl[a], self.pl[b] = self.pl[b], self.pl[a]
        self.assertEqual(self.check()["state"], "match")
        r = self.check(ordered=True)
        self.assertEqual((r["state"], r["kind"]), ("mismatch", "wrong_order"))

    def test_unfinished_or_missing_records_are_skipped(self):
        r = self.check([l for l in self.pl if l.get("ev") != "case_done"])
        self.assertEqual(r["state"], "skipped")
        r = self.check(self.pl[:-1] + [{"ev": "case_done", "id": "c", "status": "error"}])
        self.assertEqual(r["state"], "skipped")
        os.remove(os.path.join(self.plug, "c.jsonl"))
        self.assertEqual(pb.check_calls(["c"], self.plug, self.rec)[0]["reason"], "no plugin record")

    def test_checkout_param_of_the_input_is_a_frame_read_and_other_params_are_not(self):
        pl = []
        for l in self.pl:
            if l.get("fn") == "prerender_cb.checkout_layer":
                l = dict(l, fn="inter.checkout_param")
            pl.append(l)
        pl.insert(2, {"ev": "cb", "fn": "inter.checkout_param", "param_index": 5, "time": 0, "time_step": 100,
                      "time_scale": 2400, "footage_frame": 0, "out_frame": 0})
        self.assertEqual(self.check(pl)["state"], "match")

    def test_times_are_compared_as_exact_fractions_across_time_bases(self):
        pl = []
        for l in self.pl:
            if l.get("ev") == "cb":
                l = dict(l, time=l["time"] * 2, time_scale=l["time_scale"] * 2, time_step=l["time_step"] * 2)
            pl.append(l)
        self.assertEqual(self.check(pl)["state"], "match")


class Summary(unittest.TestCase):
    def grade(self, n=200):
        rows = []
        for i in range(n):
            rows.append({"id": "c%d" % i, "status": "pass", "logic_parity": i % 5 == 0, "bytes_compared": 10,
                         "floats_compared": 0 if i % 2 else 4, "differing_values": 0, "max_ulp": 0 if i % 2 == 0 else None})
        return {"cases": rows}

    def kills(self, killed=3, total=3, survivors=()):
        return {"total": total, "killed": killed, "survivors": list(survivors), "mutants": []}

    def test_schema_and_a_verified_module(self):
        s, unmet = pb.build_summary(self.grade(), None, self.kills(), {"compared": 200, "matching": 200},
                                    {"plugin_sha256": "x", "candidates_phase_a": 22, "commands": ["c"]}, "2026-10-30")
        self.assertEqual(list(s.keys()), pb.SUMMARY_KEYS)
        self.assertEqual(unmet, [])
        self.assertEqual(s["status"], "verified")
        self.assertEqual(s["cases"], {"in_contract": 200, "compared": 200, "identical": 200, "differing": 0,
                                      "out_of_contract": 22, "logic_parity": 40})
        self.assertEqual(s["values"]["bytes_compared"], 2000)
        self.assertEqual(s["values"]["floats_compared"], 400)
        self.assertEqual(s["values"]["max_ulp"], 0)

    def test_every_unmet_condition_blocks_verified(self):
        def base(**kw):
            return pb.build_summary(kw.get("grade", self.grade()), kw.get("ooc"), kw.get("kills", self.kills()),
                                    kw.get("calls", {"compared": 200, "matching": 200}), {}, "d")
        self.assertIn("a mutant survives or none ran", base(kills=self.kills(2, 3, ["m3"]))[1])
        self.assertIn("call records not all matching", base(calls={"compared": 200, "matching": 199})[1])
        self.assertIn("call records not all matching", base(calls=None)[1])
        self.assertIn("fewer than 150 cases compared", base(grade=self.grade(100))[1])
        g = self.grade()
        g["cases"][3].update(status="FAIL", differing_values=2)
        s, unmet = pb.build_summary(g, None, self.kills(), {"compared": 200, "matching": 200}, {}, "d")
        self.assertEqual((s["status"], s["cases"]["differing"], s["values"]["differing_values"]), ("in_progress", 1, 2))
        self.assertIn("a case or value differs", unmet)

    def test_no_reference_is_not_a_pass_and_not_compared(self):
        g = self.grade()
        for r in g["cases"][:10]:
            r["status"] = "no reference"
        s, unmet = pb.build_summary(g, None, None, None, {"mutants_total_phase_a": 25}, "d")
        self.assertEqual((s["cases"]["in_contract"], s["cases"]["compared"], s["cases"]["identical"]), (200, 190, 190))
        self.assertIn("not every in-contract case compared", unmet)
        self.assertEqual(s["mutants"], {"total": 25, "killed": 0, "survivors": []})
        self.assertIn("10 in-contract cases have no reference", s["blockers"])

    def test_nothing_compared_is_not_verified(self):
        g = self.grade(5)
        for r in g["cases"]:
            r["status"] = "no reference"
        s, _ = pb.build_summary(g, None, None, None, {}, "d")
        self.assertEqual((s["status"], s["cases"]["compared"], s["values"]["max_ulp"]), ("in_progress", 0, None))

    def test_moved_cases_leave_the_in_contract_count(self):
        ooc = {"moved": [{"id": "c1", "source": "testset"}, {"id": "c2", "source": "testset"}],
               "counts": {"candidates_out_of_contract": 20, "candidates_to_review": 2, "unjudged": 1}}
        s, _ = pb.build_summary(self.grade(), ooc, None, None, {}, "d")
        self.assertEqual((s["cases"]["in_contract"], s["cases"]["compared"], s["cases"]["out_of_contract"]), (198, 198, 22))
        self.assertTrue(any("need review" in b for b in s["blockers"]))

    def test_blocked_only_with_an_outside_blocker(self):
        s, _ = pb.build_summary(self.grade(), None, self.kills(), None, {}, "d", ["emulator stops in PARAMS_SETUP"])
        self.assertEqual(s["status"], "blocked")
        self.assertIn("emulator stops in PARAMS_SETUP", s["blockers"])


class Triage(unittest.TestCase):
    def test_ladder(self):
        v = pb.ladder_variants(["vectors=1", "speed_percent=50", "frame_interp=1", "frame_num=0", "input_fields=1"])
        roles = [(x["role"], x["setting"]) for x in v]
        roles += [tuple(a.split(" ", 1)) for x in v for a in x["also"]]
        self.assertEqual(roles[:2], [("case", ""), ("default", "")])
        self.assertIn(("revert", "speed_percent"), roles)
        self.assertIn(("only", "frame_interp"), roles)
        self.assertNotIn(("only", "frame_num"), roles)
        self.assertEqual(v[0]["params"], ["vectors=1", "speed_percent=50", "frame_interp=1"])
        self.assertEqual(next(x for x in v if x["role"] == "revert" and x["setting"] == "speed_percent")["params"],
                         ["vectors=1", "frame_interp=1"])
        self.assertEqual(next(x for x in v if x["role"] == "revert" and x["setting"] == "frame_interp")["params"],
                         ["vectors=1", "speed_percent=50"])
        self.assertIn("only speed_percent", next(x for x in v if x["role"] == "revert" and x["setting"] == "frame_interp")["also"])

    def test_ladder_of_a_single_setting_has_no_duplicates(self):
        v = pb.ladder_variants(["vectors=1", "speed_percent=50"])
        self.assertEqual(len({tuple(x["params"]) for x in v}), len(v))
        self.assertEqual([x["role"] for x in v], ["case", "default"])
        self.assertIn("revert speed_percent", v[1]["also"])

    def test_keyed_gets_a_constant_probe_and_defaults_compare_numerically(self):
        v = pb.ladder_variants(["vectors=1", "speed_percent=100@0;50@11", "frame_interp=2.0"])
        un = next(x for x in v if x["role"] == "unkey")
        self.assertEqual(un["params"], ["vectors=1", "speed_percent=100"])
        self.assertTrue(pb.is_default("speed_percent", "100.0"))
        self.assertFalse(pb.is_default("speed_percent", "100@0;100@5"))

    def test_defaults_cover_every_param_the_grid_uses(self):
        used = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "params_used.tsv")
        if not os.path.exists(used):
            self.skipTest("no params_used.tsv")
        ids = {l.split("\t")[0] for l in open(used).read().splitlines()[1:]}
        self.assertEqual(ids - set(pb.FIXED), set(pb.DEFAULTS))

    def test_smallest_failing_and_feature_counts(self):
        rows = [{"id": "a", "status": "FAIL", "width": 5, "height": 7, "frames": 12, "depth": 32},
                {"id": "b", "status": "FAIL", "width": 1, "height": 1, "frames": 120, "depth": 8},
                {"id": "c", "status": "FAIL", "width": 1, "height": 1, "frames": 12, "depth": 16},
                {"id": "d", "status": "pass", "width": 1, "height": 1, "frames": 1, "depth": 8}]
        self.assertEqual([r["id"] for r in pb.smallest_failing(rows)], ["c", "b", "a"])
        cases = {k: {"params": ["vectors=1", "speed_percent=50"]} for k in "abcd"}
        for r in rows:
            r["tags"] = {"group": "speed"}
        t = pb.triage_tables(rows, cases)
        self.assertEqual(t["param speed_percent=50"], [3, 4])

    def test_ladder_verdict(self):
        lad = {"probes": [{"id": "1", "role": "case", "setting": "", "params": []},
                          {"id": "2", "role": "default", "setting": "", "params": []},
                          {"id": "3", "role": "only", "setting": "frame_interp", "params": []},
                          {"id": "4", "role": "revert", "setting": "speed_percent", "params": []}]}
        rows = [{"id": "1", "status": "FAIL"}, {"id": "2", "status": "pass"}, {"id": "3", "status": "FAIL"},
                {"id": "4", "status": "pass"}]
        _, v = pb.ladder_verdict(lad, rows)
        self.assertTrue(any("frame_interp" in x and "alone" in x for x in v))
        self.assertTrue(any("speed_percent" in x and "needed" in x for x in v))


@unittest.skipUnless(os.name == "posix", "needs a posix shell script as a fake vfx")
class GradeRows(unittest.TestCase):
    def test_pass_fail_and_no_reference_with_a_fake_vfx(self):
        with tempfile.TemporaryDirectory() as d:
            refs = os.path.join(d, "refs")
            os.makedirs(refs)
            ts = os.path.join(d, "ts")
            os.makedirs(os.path.join(ts, "inputs"))
            fake = os.path.join(d, "vfx")
            with open(fake, "w") as fh:
                fh.write("#!/bin/sh\nwhile [ $# -gt 0 ]; do if [ \"$1\" = --out ]; then o=\"$2\"; fi; shift; done\n"
                         "printf '\\000\\000\\000\\001' > \"$o\"\n")
            os.chmod(fake, os.stat(fake).st_mode | stat.S_IEXEC)

            def mk(i, depth):
                return {"id": i, "effect": "retime", "label": "l", "params": ["vectors=1"], "depth": depth, "width": 1,
                        "height": 1, "source": "s", "input": "inputs/x", "reference": i + ".r", "kind": "temporal",
                        "frames": 1, "fps": "24"}
            cases = [mk("ok", 8), mk("bad", 8), mk("none", 8)]
            with open(os.path.join(refs, "ok.r"), "wb") as fh:
                fh.write(bytes([0, 0, 0, 1]))
            with open(os.path.join(refs, "bad.r"), "wb") as fh:
                fh.write(bytes([0, 0, 0, 3]))
            report = {"cases": [{"id": "ok", "status": "pass"},
                                {"id": "bad", "status": "FAIL", "frame": 0, "failing_frames": 1, "differing_pixels": 1},
                                {"id": "none", "status": "no reference"}]}
            rows = pb.grade_rows(report, cases, {}, ts, refs, fake)
            self.assertEqual([r["status"] for r in rows], ["pass", "FAIL", "no reference"])
            self.assertEqual((rows[1]["differing_values"], rows[0]["bytes_compared"], rows[2]["bytes_compared"]), (1, 4, 0))
            t = pb.grade_totals(rows)
            self.assertEqual((t["compared"], t["identical"], t["differing"], t["no_reference"]), (2, 1, 1, 1))


if __name__ == "__main__":
    unittest.main()
