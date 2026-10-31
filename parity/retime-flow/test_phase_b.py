#!/usr/bin/env python3
"""Tests of phase_b.py that need neither the server nor the emulator.

    python3 -m unittest test_phase_b        (from parity/retime-flow)

A small synthetic module (six cases, a fake `vfx` and a fake `fx-parity` written as Python scripts) stands in for
the real one, so the grading, out-of-contract, call-record, kill, summary and triage logic runs end to end.
"""
import json
import os
import struct
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase_b as pb  # noqa: E402

FAKE_VFX = r'''
import hashlib, json, os, sys
a = sys.argv[1:]
mut = None
if a[:1] == ["--mutant"]:
    mut, a = a[1], a[2:]
if a[0] == "list":
    sys.stdout.write(json.dumps([{"id": "retime", "params": [{"id": "speed_percent", "default": 100}, {"id": "vectors", "default": 5},
                                                              {"id": "prep", "default": 1}]}]))
    sys.exit(0)
def opt(k, many=False):
    out = [a[i + 1] for i, x in enumerate(a) if x == k]
    return out if many else (out[0] if out else None)
depth, frames = int(opt("--depth")), int(opt("--frames"))
params = opt("--param", True)
data = open(opt("--in"), "rb").read()
shift = sum(sum(map(ord, p)) for p in params) % 7
if opt("--quality") == "draft":
    shift += 1
if mut == "m01" and depth == 8:
    shift += 1
if mut == "m03":
    shift += 1
out = bytes((b + shift) & 255 for b in data)
rec = os.environ.get("VFX_CALL_RECORD")
if rec:
    with open(rec, "w") as fh:
        fh.write(json.dumps({"ev": "run", "effect": a[1], "frames": frames}) + "\n")
        for n in range(frames):
            fh.write(json.dumps({"ev": "frame", "out_frame": n, "time": n * 100, "time_step": 100, "time_scale": 2400}) + "\n")
            for k in {n, min(n + 1, frames - 1)}:
                fh.write(json.dumps({"ev": "read", "out_frame": n, "layer": "input", "time": k * 100, "index": k, "clip_frames": frames}) + "\n")
sys.stdout.buffer.write(out)
'''

FAKE_FXP = r'''
import json, subprocess, sys
a = sys.argv[1:]
if "--help" in a:
    print("usage: fx-parity run [--max-ulp N]")
    sys.exit(0)
def opt(k):
    return a[a.index(k) + 1]
ts, refs, vfx = opt("--testset"), opt("--refs"), opt("--vfx")
assert opt("--max-ulp") == "0" and "--json" in a
import os
cases = json.load(open(os.path.join(ts, "cases.json")))["cases"]
res, fail = [], 0
for c in cases:
    rp = os.path.join(refs, c["reference"])
    r = {"id": c["id"]}
    if not os.path.isfile(rp):
        r["status"] = "no reference"
    else:
        args = ["apply", c["effect"], "--width", str(c["width"]), "--height", str(c["height"]), "--depth", str(c["depth"])]
        for p in c["params"]:
            args += ["--param", p]
        args += ["--fps", c["fps"], "--frames", str(c["frames"]), "--in", os.path.join(ts, c["input"])]
        out = subprocess.run([vfx] + args, capture_output=True).stdout
        ref = open(rp, "rb").read()
        r["status"] = "pass" if out == ref else "FAIL"
        if out != ref:
            fail += 1
            r["clip_differing_pixels"] = sum(1 for x, y in zip(out, ref) if x != y)
    res.append(r)
print(json.dumps({"summary": {"pass": sum(1 for r in res if r["status"] == "pass"), "fail": fail, "total": len(res)}, "cases": res}))
sys.exit(1 if fail else 0)
'''


def make_exe(directory, name, script, extra=()):
    directory = Path(directory)
    script_path = directory / f"{name}.py"
    script_path.write_text(script)
    extra_text = " ".join(f'"{x}"' for x in extra)
    if os.name == "nt":
        p = directory / f"{name}.cmd"
        p.write_text(f'@"{sys.executable}" "{script_path}" {extra_text} %*\r\n')
    else:
        p = directory / name
        p.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script_path}" {extra_text} "$@"\n')
        p.chmod(0o755)
    return str(p)


class Args:
    def __init__(self, root, **kw):
        self.root, self.results, self.phase_a, self.testsets, self.testset, self.command = str(root), None, None, None, None, "test"
        self.__dict__.update(kw)


DRAFT = os.name != "nt"     # the quality wrapper is a shell script


def make_module(tmp, draft=DRAFT):
    """root/testsets/{grid,draft}, root/bin/vfx-*, root/results with fake phase A tables. -> (root, ctx, vfx)."""
    root = Path(tmp)
    (root / "bin").mkdir()
    (root / "results" / "mutants").mkdir(parents=True)
    vfx = make_exe(root / "bin", "vfx-base", FAKE_VFX)
    for m in ("m00", "m01", "m02", "m03"):
        make_exe(root / "bin", f"vfx-{m}", FAKE_VFX, extra=["--mutant", m])
    spec = {"grid": [(8, 2, 2, "50"), (8, 3, 1, "25"), (16, 2, 2, "75"), (32, 2, 2, "50"), (32, 1, 3, "50@0;40@2"), (8, 1, 1, "50")]}
    if draft:
        spec["draft"] = [(8, 2, 2, "50"), (32, 2, 2, "25")]
    k = 0
    for name, items in spec.items():
        d = root / "testsets" / name
        (d / "inputs").mkdir(parents=True)
        cases = []
        for depth, w, h, speed in items:
            k += 1
            cid = f"retime.t{k}.src-{w}x{h}.d{depth}.{k:08x}"
            fname = f"inputs/in{k}.{pb.PIX[depth]}"
            (d / fname).write_bytes(bytes((i * 7 + k) & 255 for i in range(w * h * pb.BPP[depth] * 3)))
            c = {"id": cid, "effect": "retime", "label": f"t{k}", "params": [f"speed_percent={speed}"] + (["prep=2"] if k % 2 else []),
                 "depth": depth, "width": w, "height": h, "source": "src", "input": fname, "reference": f"{cid}.{pb.PIX[depth]}",
                 "kind": "temporal", "frames": 3, "fps": "24", "group": "g" + str(k % 2)}
            if name == "draft":
                c["quality"] = "draft"
            if "@" in speed:
                c["stand_in"] = "keyframe_interpolation"
            cases.append(c)
        pb.jdump(d / "cases.json", {"format": 1, "inputs": [], "cases": cases})
    if not draft:
        (root / "testsets" / "draft").mkdir(parents=True)
        pb.jdump(root / "testsets" / "draft" / "cases.json", {"format": 1, "inputs": [], "cases": []})
    pb.jdump(root / "testsets" / "out_of_contract.json", [
        {"id": "ooc.0", "label": "vectors=0", "why": "popup below 1", "params": ["vectors=0"], "width": 16, "height": 16, "depth": 8},
        {"id": "ooc.1", "label": "prep=4", "why": "popup above 3", "params": ["prep=4"], "width": 16, "height": 16, "depth": 8}])
    (root / "results" / "out_of_contract.tsv").write_text(
        "id\tlabel\twhy\trewrite_check_exit\tmessage\nooc.0\tvectors=0\tpopup below 1\t3\tbad value\nooc.1\tprep=4\tpopup above 3\t3\tbad value\n")
    ctx = pb.Ctx(Args(root))
    cases = pb.load_cases(ctx)
    tables = {}
    for name in ("base", "m00", "m01", "m02", "m03"):
        exe = vfx if name == "base" else str(root / "bin" / (f"vfx-{name}" + (".cmd" if os.name == "nt" else "")))
        rows = {}
        for cid, c in cases.items():
            rc, out, err = pb.render(exe, c, c["_dir"])
            rows[cid] = (c["_set"], [cid, str(rc), str(len(out)), pb.sha_bytes(out), "1", err])
        tables[name] = rows
        for s in pb.SETS:
            if s not in spec:
                continue
            p = root / "results" / (f"{name}.{s}.tsv" if name == "base" else f"mutants/{name}.{s}.tsv")
            p.write_text("id\texit\tbytes\tsha256\tms\tstderr\n" + "".join("\t".join(r) + "\n" for st, r in tables[name].values() if st == s))
    changed = {m: [cid for cid in cases if tables[m][cid][1][3] != tables["base"][cid][1][3]] for m in ("m00", "m01", "m02", "m03")}
    pb.jdump(root / "results" / "mutants.json", [{"id": m, "built": True, "changed": ch} for m, ch in changed.items()])
    pb.jdump(root / "results" / "summary.json", {"module": "retime-flow", "plugin": "Twixtor_64.aex", "plugin_sha256": "abc", "effect": "retime",
                                                 "status": "in_progress", "blockers": ["emulator side not run yet"], "fixes": [], "commands": ["phase a"],
                                                 "phase_a": {"note": "kept"}})
    return root, ctx, vfx, changed


def write_refs(ctx, vfx, refs, skip=()):
    Path(refs).mkdir(parents=True, exist_ok=True)
    for cid, c in pb.load_cases(ctx).items():
        if cid not in skip:
            rc, out, err = pb.render(vfx, c, c["_dir"])
            (Path(refs) / c["reference"]).write_bytes(out)


class DiffClipTests(unittest.TestCase):
    def test_equal_clips(self):
        b = bytes(range(32))
        r = pb.diff_clip(8, 2, 2, 2, b, b)
        self.assertEqual((r["differing_values"], r["differing_pixels"], r["failing_frames"], r["first"]), (0, 0, 0, None))

    def test_8bpc_one_bit(self):
        a = bytearray(32)
        b = bytearray(32)
        b[20] ^= 1
        r = pb.diff_clip(8, 2, 2, 2, bytes(a), bytes(b))
        self.assertEqual((r["differing_values"], r["differing_pixels"], r["failing_frames"]), (1, 1, 1))
        self.assertEqual(r["first"], {"frame": 1, "x": 1, "y": 0, "channel": "r", "expected": 0, "actual": 1})
        self.assertIsNone(r["max_ulp"])

    def test_16bpc_channel_and_pixel(self):
        a = array("H", [0] * 16)
        b = array("H", a)
        b[4 * 3 + 2] = 9
        r = pb.diff_clip(16, 2, 2, 1, a.tobytes(), b.tobytes())
        self.assertEqual((r["differing_values"], r["first"]["x"], r["first"]["y"], r["first"]["channel"]), (1, 1, 1, "b"))

    def f32clip(self, words, n=4):
        a = array("I", words)
        return a.tobytes()

    def test_32bpc_planar_channels_and_ulp(self):
        n = 4
        ref = [0x3F800000] * (4 * n)
        out = list(ref)
        out[2 * n + 1] ^= 1          # plane 2 is r, pixel 1
        r = pb.diff_clip(32, 2, 2, 1, self.f32clip(ref), self.f32clip(out))
        self.assertEqual((r["differing_values"], r["max_ulp"], r["first"]["channel"], r["first"]["x"]), (1, 1, "r", 1))
        out[3 * n] = 0x3F800000 + 5   # plane 3 is a, pixel 0
        r = pb.diff_clip(32, 2, 2, 1, self.f32clip(ref), self.f32clip(out))
        self.assertEqual((r["differing_values"], r["differing_pixels"], r["max_ulp"], r["per_channel"]), (2, 2, 5, {"a": 1, "r": 1}))

    def test_negative_zero_is_a_difference_with_ulp_zero(self):
        ref = [0x80000000] + [0] * 3
        out = [0] * 4
        r = pb.diff_clip(32, 1, 1, 1, self.f32clip(ref), self.f32clip(out))
        self.assertEqual((r["differing_values"], r["max_ulp"], r["signed_zero_diffs"], r["nan_mismatch"]), (1, 0, 1, 0))

    def test_nan_payload_is_a_difference_without_a_ulp(self):
        ref = [0x7FC00000] + [0] * 3
        out = [0x7FC00001] + [0] * 3
        r = pb.diff_clip(32, 1, 1, 1, self.f32clip(ref), self.f32clip(out))
        self.assertEqual((r["differing_values"], r["nan_mismatch"], r["max_ulp"]), (1, 1, None))
        same = pb.diff_clip(32, 1, 1, 1, self.f32clip(ref), self.f32clip(ref))
        self.assertEqual(same["differing_values"], 0)

    def test_nan_against_number(self):
        r = pb.diff_clip(32, 1, 1, 1, self.f32clip([0x7FC00000, 0, 0, 0]), self.f32clip([0x3F800000, 0, 0, 0]))
        self.assertEqual((r["nan_mismatch"], r["max_ulp"]), (1, None))

    def test_size_mismatch(self):
        r = pb.diff_clip(8, 2, 2, 2, bytes(32), bytes(31))
        self.assertIn("size_mismatch", r)

    def test_ulp_across_zero(self):
        self.assertEqual(abs(pb.f32_ordered(0x00000001) - pb.f32_ordered(0x80000001)), 2)


class StatusTests(unittest.TestCase):
    def test_classes(self):
        c = pb.classify_status
        self.assertEqual(c({"status": "rendered"}), ("rendered", False))
        self.assertEqual(c({"status": "unsupported", "error": "param x has no counterpart"}), ("refused", False))
        self.assertEqual(c({"status": "error", "error": "PluginError: RENDER returned PF_Err 25"}), ("error", False))
        self.assertEqual(c({"status": "error", "error": "GuestFault: read of unmapped memory"}), ("error", True))
        self.assertEqual(c({"status": "error", "error": "the worker died"}), ("error", True))
        self.assertEqual(c({"status": "timeout"}), ("timeout", False))
        self.assertEqual(c({"status": "not_run", "error": "the plugin did not boot"}), ("not_run", True))
        self.assertEqual(c({"status": "unsupported", "error": "Bezier handles refused"}), ("refused", True))
        self.assertEqual(c({"status": "weird"})[0], "unknown")

    def test_read_formats(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            pb.jdump(t / "a.json", {"cases": [{"id": "x", "status": "rendered"}, {"id": "y", "status": "error", "error": "e"}]})
            pb.jdump(t / "b.json", {"x": {"status": "refused"}, "z": "timeout"})
            (t / "c.tsv").write_text("id\tstatus\tdetail\nq\trefused\twhy\n")
            (t / "d.tsv").write_text("p\terror\tboom\n")
            self.assertEqual(pb.read_status_file(t / "a.json")["y"]["error"], "e")
            self.assertEqual(pb.read_status_file(t / "b.json")["x"]["status"], "refused")
            self.assertEqual(pb.read_status_file(t / "b.json")["z"]["status"], "timeout")
            self.assertEqual(pb.read_status_file(t / "c.tsv")["q"]["detail"], "why")
            self.assertEqual(pb.read_status_file(t / "d.tsv")["p"]["status"], "error")

    def test_key_values_and_ranges(self):
        self.assertEqual(pb.parse_key_values("100@0;40@5e"), [100.0, 40.0])
        self.assertEqual(pb.parse_key_values("Best"), [])
        pm = [{"plugin": "P", "match": "exact", "vfx_id": "speed_percent", "ae_type": "FLOAT_SLIDER", "ae_min": "-5000", "ae_max": "5000", "ae_name": "Speed %"}]
        cases = {"a": {"params": ["speed_percent=6000"]}, "b": {"params": ["speed_percent=50@0;9000@3"]}, "c": {"params": ["speed_percent=-50"]}}
        got = pb.range_candidates(cases, pm, "P")
        self.assertEqual([g["id"] for g in got], ["a", "b"])


class CallTests(unittest.TestCase):
    def views(self, plugin_reads, rewrite_reads, pframes=None, rframes=None):
        pl = []
        for n, reads in plugin_reads.items():
            pl.append({"ev": "frame", "out_frame": n, "time": n * 100, "time_step": 100, "time_scale": 2400})
            for t, ff in reads:
                pl.append({"ev": "cb", "fn": "inter.checkout_param", "param_index": 0, "time": t, "footage_frame": ff, "out_frame": n})
        rw = []
        for n, reads in rewrite_reads.items():
            rw.append({"ev": "frame", "out_frame": n, "time": n * 100, "time_step": 100, "time_scale": 2400})
            for t, idx in reads:
                rw.append({"ev": "read", "out_frame": n, "layer": "input", "time": t, "index": idx, "clip_frames": 4})
        return pb.plugin_view(pl, {}), pb.rewrite_view(rw)

    def test_match_missing_extra_wrong_time_and_order(self):
        pv, rv = self.views({0: [(0, 0)], 1: [(100, 1), (200, 2)]}, {0: [(0, 0)], 1: [(100, 1), (200, 2)]})
        self.assertTrue(pb.compare_records(pv, rv)["match"])
        pv, rv = self.views({0: [(0, 0)], 1: [(100, 1), (200, 2)]}, {0: [(0, 0)], 1: [(100, 1)]})
        r = pb.compare_records(pv, rv)
        self.assertEqual((r["match"], r["first_mismatch"]["out_frame"], r["first_mismatch"]["problems"][0]["kind"]), (False, 1, "missing_read"))
        pv, rv = self.views({0: [(0, 0)]}, {0: [(0, 0), (100, 1)]})
        self.assertEqual(pb.compare_records(pv, rv)["first_mismatch"]["problems"][0]["kind"], "extra_read")
        pv, rv = self.views({0: [(0, 0)], 1: [(100, 1)]}, {0: [(0, 0)], 1: [(101, 1)]})
        p = pb.compare_records(pv, rv)["first_mismatch"]
        self.assertEqual((p["out_frame"], p["problems"][0]["kind"], p["problems"][0]["pairs"][0]["plugin_time"]), (1, "wrong_time", 100))
        pv, rv = self.views({0: [(0, 0)]}, {0: [(0, 0)]})
        pv[0][0] = (1, 1, 1)
        self.assertEqual(pb.compare_records(pv, rv)["first_mismatch"]["problems"][0]["kind"], "frame_time")
        pv, rv = self.views({0: [(100, 1), (200, 2)]}, {0: [(200, 2), (100, 1)]})
        self.assertTrue(pb.compare_records(pv, rv)["match"])
        self.assertEqual(pb.compare_records(pv, rv, ordered=True)["first_mismatch"]["problems"][0]["kind"], "order")

    def test_outside_the_clip_is_none_on_both_sides(self):
        pv, rv = self.views({0: [(-100, -1), (500, 5)]}, {0: [(-100, None), (500, None)]})
        self.assertTrue(pb.compare_records(pv, rv, default_clip_frames=4)["match"])
        pv, rv = self.views({0: [(300, 3)]}, {0: [(300, 2)]})
        self.assertEqual(pb.compare_records(pv, rv, default_clip_frames=4)["first_mismatch"]["problems"][0]["kind"], "wrong_index")

    def test_param_checkouts_are_not_reads(self):
        pl = [{"ev": "frame", "out_frame": 0, "time": 0, "time_step": 100, "time_scale": 2400},
              {"ev": "cb", "fn": "inter.checkout_param", "param_index": 3, "time": 7, "out_frame": 0},
              {"ev": "cb", "fn": "inter.checkout_param", "param_index": 0, "time": 0, "footage_frame": 0, "out_frame": 0}]
        frames, reads, par, inc = pb.plugin_view(pl, {})
        self.assertEqual((len(reads[0]), par[0]), (1, [(3, 7)]))
        frames, reads, par, inc = pb.plugin_view(pl, {3: "alt_source_a"})
        self.assertEqual(len(reads[0]), 2)

    def test_smart_render_checkout_layer(self):
        pl = [{"ev": "frame", "out_frame": 0, "time": 0, "time_step": 100, "time_scale": 2400},
              {"ev": "cb", "fn": "prerender_cb.checkout_layer", "param_index": 0, "time": 0, "footage_frame": 0, "inside": True, "out_frame": 0},
              {"ev": "cb", "fn": "smart_cb.checkout_layer_pixels", "out_frame": 0}]
        frames, reads, par, inc = pb.plugin_view(pl, {})
        self.assertEqual([r["time"] for r in reads[0]], [0])


class TriageTests(unittest.TestCase):
    def rows(self):
        def r(i, depth, w, status, params, **kw):
            d = {"id": i, "compared": True, "status": status, "depth": depth, "width": w, "height": 2, "frames": 4, "fps": "24",
                 "source": "disc", "group": "g", "params": params, "effect": "retime", "differing_values": 3 if status == "differs" else 0,
                 "max_ulp": None, "input": "inputs/x", "testset_dir": "/nowhere"}
            d.update(kw)
            return d
        return [r("a", 8, 40, "identical", ["speed_percent=50"]), r("b", 8, 4, "differs", ["speed_percent=150", "vectors=3", "prep=2"]),
                r("c", 32, 40, "differs", ["speed_percent=50", "prep=2"], quality="draft"), r("d", 16, 4, "identical", ["speed_percent=50", "prep=2"])]

    def test_groups_and_order(self):
        dims = pb.triage_groups(self.rows())
        self.assertEqual(dims["param prep"]["2"], [2, 3])
        self.assertEqual(dims["param speed_percent"]["150"], [1, 1])
        self.assertEqual(dims["quality"]["draft"], [1, 1])
        self.assertEqual([r["id"] for r in pb.smallest_first(self.rows())], ["b", "c"])

    def test_ladder(self):
        defaults = {"speed_percent": "100", "vectors": "5", "prep": "1"}
        row = self.rows()[1]
        rungs = pb.build_ladder(row, defaults)
        names = [n for n, _, _ in rungs]
        self.assertEqual(names[0], "original")
        self.assertIn("all-default", names[1])
        sets = [(sorted(p), q) for _, p, q in rungs]
        self.assertIn((["prep=2", "vectors=3"], None), sets)       # speed reverted
        self.assertIn((["speed_percent=150"], None), sets)         # speed alone
        self.assertIn(([], None), sets)
        self.assertEqual(sum(1 for _, _, q in rungs if q == "draft"), 1)
        self.assertEqual(len({(tuple(sorted(p)), q) for _, p, q in rungs}), len(rungs))
        row2 = dict(row, params=["speed_percent=100", "vectors=5"])
        self.assertEqual([n for n, _, _ in pb.build_ladder(row2, defaults)], ["original", "all-default", "quality-draft"])

    def test_ladder_files(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "ts" / "inputs").mkdir(parents=True)
            (t / "ts" / "inputs" / "x").write_bytes(b"123")
            row = dict(self.rows()[1], testset_dir=str(t / "ts"), source="disc")
            cases = pb.write_ladder(row, pb.build_ladder(row, {"speed_percent": "100"}), t / "probe")
            doc = pb.jload(t / "probe" / "cases.json")
            self.assertEqual(len(doc["cases"]), len(cases))
            self.assertTrue((t / "probe" / "inputs" / "x").exists())
            self.assertTrue(all(c["reference"].endswith("rgba") and c["ladder_of"] == "b" for c in doc["cases"]))


class RuleTests(unittest.TestCase):
    def base(self):
        return {"cases": {"in_contract": 200, "compared": 200, "identical": 200, "differing": 0, "out_of_contract": 3, "logic_parity": 2},
                "values": {"differing_values": 0}, "call_records": {"compared": 200, "matching": 200},
                "mutants": {"total": 5, "killed": 5, "survivors": []}}

    def test_rule(self):
        self.assertEqual(pb.unmet_rule(self.base()), [])
        s = self.base()
        s["cases"]["compared"] = 149
        s["cases"]["identical"] = 149
        self.assertEqual(len(pb.unmet_rule(s)), 2)
        s = self.base()
        s["call_records"] = {"compared": 0, "matching": 0}
        self.assertEqual(pb.unmet_rule(s), ["call records not all matching"])
        s = self.base()
        s["mutants"]["killed"], s["mutants"]["survivors"] = 4, ["m5"]
        self.assertEqual(pb.unmet_rule(s), ["a mutant survives or none ran"])
        s = self.base()
        s["values"]["differing_values"] = 1
        self.assertEqual(pb.unmet_rule(s), ["a case or value differs"])


class ModuleTests(unittest.TestCase):
    """The logic that normally runs on the server, on a six case stand-in module."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root, self.ctx, self.vfx, self.changed = make_module(self.tmp.name)
        self.cases = pb.load_cases(self.ctx)
        self.fxp = [sys.executable, str(self.root / "fxp.py")]
        (self.root / "fxp.py").write_text(FAKE_FXP)
        self.ids = list(self.cases)
        self.n = len(self.ids)

    def tearDown(self):
        self.tmp.cleanup()

    def sub(self, name):
        c = pb.Ctx(Args(self.root, results=str(self.root / name)))
        (self.root / name).mkdir(exist_ok=True)
        return c

    def grade(self, ctx, refs):
        return pb.grade(ctx, refs, self.vfx, self.fxp, jobs=1)

    def test_phase_a_fixture_is_sane(self):
        self.assertTrue(self.changed["m01"] and self.changed["m03"] and not self.changed["m02"] and not self.changed["m00"])

    def test_perfect_oracle(self):
        refs = self.root / "refs"
        write_refs_q(self.ctx, self.vfx, refs)
        c = self.sub("p")
        rows, cmds = self.grade(c, refs)
        doc = pb.write_grade(c, rows, cmds, refs)
        t = doc["totals"]
        self.assertEqual((t["compared"], t["identical"], t["differing"], t["no_reference"]), (self.n, self.n, 0, 0))
        self.assertEqual(t["logic_parity_compared"], 1)
        self.assertEqual(t["max_ulp"], 0)
        self.assertEqual(t["bytes_compared"] + t["floats_compared"] * 4 > 0, True)
        kd = pb.kills(c, refs, "tables")
        self.assertEqual((kd["totals"]["killed"], kd["totals"]["survivors"]), (2, ["m02"]))
        self.assertTrue(kd["control"]["ok"])
        kr = pb.kills(c, refs, "rerender")
        self.assertEqual({m: e["killing_cases"] for m, e in kd["mutants"].items()}, {m: e["killing_cases"] for m, e in kr["mutants"].items()})
        with self.assertRaises(pb.ToolError):
            pb.summary(self.sub("nogrades"))
        s, unmet = pb.summary(c)
        self.assertEqual((s["cases"]["in_contract"], s["cases"]["compared"], s["cases"]["out_of_contract"], s["mutants"]["survivors"]), (self.n, self.n, 2, ["m02"]))
        self.assertEqual(list(s)[:6], ["module", "plugin", "plugin_sha256", "effect", "status", "blockers"])
        self.assertEqual(s["phase_a"], {"note": "kept"})
        self.assertEqual(s["status"], "in_progress")
        self.assertIn("a mutant survives or none ran", unmet)

    def test_perturbed_missing_and_ooc(self):
        refs = self.root / "refs"
        write_refs_q(self.ctx, self.vfx, refs)
        # a flipped bit in one 8 bpc reference and -0 -> 0 in one 32 bpc reference
        a8 = next(i for i, c in self.cases.items() if c["depth"] == 8)
        p = refs / self.cases[a8]["reference"]
        b = bytearray(p.read_bytes())
        b[0] ^= 1
        p.write_bytes(bytes(b))
        a32 = next(i for i, c in self.cases.items() if c["depth"] == 32)
        p = refs / self.cases[a32]["reference"]
        w = array("I")
        w.frombytes(p.read_bytes())
        w[0] = 0x80000000 if w[0] != 0x80000000 else 0
        p.write_bytes(w.tobytes())
        c = self.sub("q")
        rows, cmds = self.grade(c, refs)
        doc = pb.write_grade(c, rows, cmds, refs)
        got = {r["id"]: r for r in rows if r["status"] == "differs"}
        self.assertEqual(set(got), {a8, a32})
        self.assertEqual(got[a8]["differing_values"], 1)
        self.assertIsNone(got[a8]["max_ulp"])
        self.assertEqual(got[a32]["differing_values"], 1)
        self.assertEqual(got[a32]["failing_frames"], 1)
        self.assertEqual(doc["totals"]["differing_values"], 2)
        kd = pb.kills(c, refs, "tables")
        for m, e in kd["mutants"].items():
            self.assertEqual(sorted(e["killing_cases"]), sorted(set(self.changed[m]) - {a8, a32}))
        self.assertEqual(sorted(kd["base_differs_from_reference"]), sorted([a8, a32]))
        # triage on the differing cases writes a ladder
        out = []
        res = pb.triage(c, vfx=self.vfx, probe_dir=self.root / "probe", out=out.append)
        self.assertEqual(res["differing"], 2)
        self.assertTrue((self.root / "probe" / "cases.json").exists())
        self.assertEqual(res["ladder"]["defaults_from"], "vfx list")
        # missing references are neither pass nor failure
        refs2 = self.root / "refs2"
        write_refs_q(self.ctx, self.vfx, refs2, skip=(self.ids[0], self.ids[1]))
        c2 = self.sub("m")
        rows, cmds = self.grade(c2, refs2)
        t = pb.write_grade(c2, rows, cmds, refs2)["totals"]
        self.assertEqual((t["no_reference"], t["compared"], t["identical"], t["differing"]), (2, self.n - 2, self.n - 2, 0))
        kd = pb.kills(c2, refs2, "tables")
        self.assertEqual(sorted(kd["no_reference"]), sorted(self.ids[:2]))
        s, unmet = pb.summary(c2)
        self.assertEqual((s["cases"]["in_contract"], s["cases"]["compared"]), (self.n, self.n - 2))
        # out of contract
        c3 = self.sub("o")
        rows, cmds = self.grade(c3, refs)
        pb.write_grade(c3, rows, cmds, refs)
        st = self.root / "status.json"
        pb.jdump(st, {"cases": [{"id": self.ids[2], "status": "unsupported", "error": "no counterpart"},
                                {"id": self.ids[3], "status": "error", "error": "GuestFault: x"},
                                {"id": self.ids[4], "status": "rendered"},
                                {"id": "ooc.0", "status": "rendered"}, {"id": "ooc.1", "status": "refused"}, {"id": "nope", "status": "rendered"}]})
        od = pb.ooc(c3, [st])
        self.assertEqual([m["id"] for m in od["moved"]], [self.ids[2]])
        self.assertEqual([m["id"] for m in od["unjudged"]], [self.ids[3]])
        self.assertEqual(od["counts"]["candidates_the_original_renders"], 1)
        self.assertEqual(od["unknown_ids"], ["nope"])
        self.assertEqual(od["moved"][0]["rewrite"]["exit"], 0)
        od2 = pb.ooc(c3, [st], faults_out_of_contract=True)
        self.assertEqual(len(od2["moved"]), 2)
        od = pb.ooc(c3, [st])
        s, unmet = pb.summary(c3)
        self.assertEqual((s["cases"]["in_contract"], s["cases"]["compared"], s["cases"]["out_of_contract"]), (self.n - 1, self.n - 1, 2))

    def test_calls_and_records(self):
        c = self.sub("r")
        rw = self.root / "rw"
        idx = pb.record_rewrite(c, self.vfx, rw, log=lambda *_: None)
        self.assertTrue(all(v[0] == 0 and v[2] for v in idx.values()))
        self.assertEqual(pb.find_record_vfx(c, self.vfx, next(iter(self.cases.values()))), [self.vfx])
        pl = self.root / "pl"
        pl.mkdir()
        for cid, case in self.cases.items():
            lines, _ = pb.load_jsonl(rw / f"{cid}.jsonl")
            pb._write_jsonl(pl / f"{cid}.jsonl", pb.synth_plugin_lines(case, lines))
        d = pb.calls(c, pl, rw)
        self.assertEqual((d["compared"], d["matching"]), (self.n, self.n))
        # drop a read from one plugin record
        cid = self.ids[0]
        lines, _ = pb.load_jsonl(pl / f"{cid}.jsonl")
        j = max(i for i, ev in enumerate(lines) if ev.get("param_index") == 0)
        del lines[j]
        pb._write_jsonl(pl / f"{cid}.jsonl", lines)
        (pl / f"{self.ids[1]}.jsonl").unlink()
        d = pb.calls(c, pl, rw)
        self.assertEqual((d["compared"], d["matching"], d["skipped"]), (self.n - 1, self.n - 2, 1))
        self.assertEqual(d["mismatches"][0]["first_mismatch"]["problems"][0]["kind"], "extra_read")
        s_file = self.root / "r" / "calls.json"
        self.assertTrue(s_file.exists())

    def cli(self, *argv):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = pb.main(list(argv))
        return code, buf.getvalue()

    def test_cli(self):
        refs = self.root / "refs"
        write_refs_q(self.ctx, self.vfx, refs)
        fxp = f'"{sys.executable}" "{self.root / "fxp.py"}"' if os.name == "nt" else f"{sys.executable} {self.root / 'fxp.py'}"
        res = str(self.root / "cli")
        common = ["--root", str(self.root), "--results", res]
        code, out = self.cli("grade", *common, "--refs", str(refs), "--vfx", self.vfx, "--fx-parity", fxp)
        self.assertEqual(code, 0, out)
        self.assertTrue((self.root / "cli" / "grade.tsv").read_text().startswith("id	status	differing_values	max_ulp	tags	logic_parity"))
        st = self.root / "st.json"
        pb.jdump(st, {"cases": [{"id": self.ids[0], "status": "refused", "error": "x"}]})
        code, out = self.cli("ooc", *common, "--status", str(st))
        self.assertEqual(code, 0, out)
        self.assertIn("out of contract", out)
        code, out = self.cli("records", *common, "--vfx", self.vfx, "--out", str(self.root / "cli" / "rw"))
        self.assertEqual(code, 0, out)
        pl = self.root / "cli" / "pl"
        pl.mkdir()
        for cid, case in self.cases.items():
            lines, _ = pb.load_jsonl(self.root / "cli" / "rw" / f"{cid}.jsonl")
            pb._write_jsonl(pl / f"{cid}.jsonl", pb.synth_plugin_lines(case, lines))
        code, out = self.cli("calls", *common, "--plugin-records", str(pl), "--rewrite-records", str(self.root / "cli" / "rw"))
        self.assertEqual(code, 0, out)
        self.assertIn(f"compared {self.n - 1}, matching {self.n - 1}", out)
        code, out = self.cli("kills", *common, "--refs", str(refs))
        self.assertEqual(code, 1, out)          # m02 changes nothing: a survivor
        self.assertIn("m02", out)
        code, out = self.cli("kills", *common, "--refs", str(refs), "--mode", "rerender", "--only", "m01,m03")
        self.assertEqual(code, 0, out)
        code, out = self.cli("summary", *common, "--date", "2026-01-01")
        self.assertEqual(code, 0, out)
        s = pb.jload(self.root / "cli" / "summary.json")
        self.assertEqual((s["measured_at"], s["cases"]["in_contract"], s["cases"]["out_of_contract"]), ("2026-01-01", self.n - 1, 3))
        self.assertEqual(s["call_records"], {"compared": self.n - 1, "matching": self.n - 1})
        code, out = self.cli("triage", *common, "--vfx", self.vfx)
        self.assertEqual(code, 0, out)
        self.assertIn("0 of", out)
        code, out = self.cli("summary", "--root", str(self.root), "--results", str(self.root / "empty"))
        self.assertEqual(code, 3)

    @unittest.skipIf(os.name == "nt", "the quality wrapper is a shell script")
    def test_probe_ladder_grades_through_testset(self):
        refs = self.root / "refs"
        write_refs_q(self.ctx, self.vfx, refs)
        a8 = next(i for i, c in self.cases.items() if c["depth"] == 8 and c["params"][0] != "speed_percent=100")
        p = refs / self.cases[a8]["reference"]
        p.write_bytes(bytes(b ^ 1 for b in p.read_bytes()))
        c = self.sub("t")
        rows, cmds = self.grade(c, refs)
        pb.write_grade(c, rows, cmds, refs)
        pb.triage(c, vfx=self.vfx, probe_dir=self.root / "probe", out=lambda *_: None)
        probe = pb.Ctx(Args(self.root, results=str(self.root / "probe_results"), testset=[str(self.root / "probe")]))
        pcases = pb.load_cases(probe)
        self.assertGreaterEqual(len(pcases), 3)
        self.assertTrue(len({c_.get("quality") for c_ in pcases.values()}) == 2)
        prefs = self.root / "probe_refs"
        prefs.mkdir()
        for cid, case in pcases.items():
            rc, out, err = pb.render(self.vfx, case, case["_dir"])
            (prefs / case["reference"]).write_bytes(out)
        probe.results.mkdir()
        rows, cmds = pb.grade(probe, prefs, self.vfx, self.fxp)
        self.assertEqual(len(cmds), 2)
        self.assertEqual(pb.grade_totals(rows)["identical"], len(pcases))


def write_refs_q(ctx, vfx, refs, skip=()):
    """References made with the case's own quality (the draft cases get --quality draft)."""
    Path(refs).mkdir(parents=True, exist_ok=True)
    for cid, c in pb.load_cases(ctx).items():
        if cid not in skip:
            rc, out, err = pb.render(vfx, c, c["_dir"])
            (Path(refs) / c["reference"]).write_bytes(out)


if __name__ == "__main__":
    unittest.main()
