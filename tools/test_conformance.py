#!/usr/bin/env python3
"""Tests of tools/conformance.py (Python 3.12 standard library; nothing is skipped).

Run: python -I tools/test_conformance.py

The fixture stages that conformance.py serves itself stand in for real providers, so these
tests need no peer repository, no built binary and no ffmpeg. tools/handoff.py (shared by
every repository of the convention) is used for schema validation and the packet level.
"""

import contextlib
import importlib.util
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SCRIPT = os.path.join(HERE, "conformance.py")

_spec = importlib.util.spec_from_file_location("conformance_under_test", SCRIPT)
conformance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conformance)
CASES = conformance.load_cases(REPO)
HANDOFF, HANDOFF_WHY = conformance.load_handoff(REPO)
REAL_STAGE_IDS = {"selftest-invert", "retime"}


def run_cli(*argv, timeout=300):
    proc = subprocess.run([sys.executable, "-I", SCRIPT, *argv], capture_output=True, timeout=timeout)
    return proc.returncode, proc.stdout.decode("utf-8", "replace"), proc.stderr.decode("utf-8", "replace")


def fixture_string(stage):
    return f'"{sys.executable}" -I "{SCRIPT}" fixture apply {stage}'


def fixture_argv(stage, width, height, depth, params=None):
    return conformance.stage_argv(conformance.fixture_command() + ["apply", stage], width, height, depth,
                                  params=params)


def statuses(doc):
    return {level["level"]: level["status"] for level in doc["levels"]}


class CasesFile(unittest.TestCase):
    def test_handoff_control_plane_is_present(self):
        self.assertIsNotNone(HANDOFF, HANDOFF_WHY)

    def test_matrix_is_the_convention(self):
        matrix = CASES["L1"]["matrix"]
        self.assertEqual(matrix["depths"], [8, 16, 32])
        self.assertEqual(matrix["sizes"], [[1, 1], [2, 2], [33, 17], [160, 96], [175, 99]])
        self.assertEqual(matrix["frames"], [1, 2, 12])
        self.assertEqual(matrix["alpha"], ["opaque", "partial", "zero"])
        self.assertRegex(matrix["fps"], r"^[0-9]+/[0-9]+$")

    def test_every_level_is_described(self):
        self.assertEqual(sorted(CASES["levels"]), list(conformance.LEVELS))

    def test_judge_cases_cover_every_injection(self):
        judge = CASES["L3"]["judge"]
        self.assertEqual({c["expect"] for c in judge},
                         {"ok", "stage_failed", "usage", "unsupported", "bad_input", "broken_pipe", "partial_frame",
                          "stage_timeout"})
        self.assertEqual(sorted(c["params"]["code"] for c in judge if c["stage"] == "fixture-exit"), [1, 2, 3, 4])
        self.assertEqual({c["stage"] for c in judge}, set(conformance.FIXTURES))

    def test_stage_ids_are_real_ids_or_fixture_placeholders(self):
        ids = [c["stage"] for c in CASES["L1"]["stages"]] + [c["stage"] for c in CASES["L3"]["judge"]]
        ids += list(CASES["L1"]["models"]) + [CASES["L4"]["packet"]["stage"], CASES["bench"]["spool_stage"]]
        for sid in ids:
            self.assertTrue(sid in REAL_STAGE_IDS or sid.startswith("fixture-"), sid)
        for sid in conformance.FIXTURES:
            self.assertTrue(sid.startswith("fixture-"), sid)

    def test_quick_rows_cover_every_value(self):
        matrix = CASES["L1"]["matrix"]
        quick = conformance.matrix_rows(matrix, "quick")
        full = conformance.matrix_rows(matrix, "full")
        self.assertEqual(len(full), 3 * 5 * 3 * 3)
        self.assertTrue(set(quick) <= set(full))
        self.assertEqual({(d, a) for d, _, _, _, a in quick},
                         {(d, a) for d in matrix["depths"] for a in matrix["alpha"]})
        self.assertEqual({(w, h) for _, w, h, _, _ in quick}, {tuple(s) for s in matrix["sizes"]})
        self.assertEqual({f for _, _, _, f, _ in quick}, set(matrix["frames"]))


class Frames(unittest.TestCase):
    def test_clip_sizes(self):
        for depth, bpp in ((8, 4), (16, 8), (32, 16)):
            self.assertEqual(conformance.frame_size(33, 17, depth), 33 * 17 * bpp)
            self.assertEqual(len(conformance.make_clip(33, 17, depth, 2, "partial", "s")), 33 * 17 * bpp * 2)

    def test_clip_is_seeded(self):
        a = conformance.make_clip(4, 3, 16, 2, "partial", "one")
        self.assertEqual(a, conformance.make_clip(4, 3, 16, 2, "partial", "one"))
        self.assertNotEqual(a, conformance.make_clip(4, 3, 16, 2, "partial", "two"))

    def test_alpha_modes_keep_straight_colour(self):
        opaque = conformance.make_clip(5, 4, 8, 1, "opaque", "a")
        zero = conformance.make_clip(5, 4, 8, 1, "zero", "a")
        partial = conformance.make_clip(5, 4, 8, 1, "partial", "a")
        self.assertEqual(set(opaque[3::4]), {255})
        self.assertEqual(set(zero[3::4]), {0})
        self.assertTrue({0, 255} <= set(partial[3::4]))
        colour = bytes(b for i, b in enumerate(zero) if i % 4 != 3)
        self.assertGreater(len(set(colour)), 1)
        self.assertEqual(conformance.make_clip(1, 1, 8, 1, "partial", "a")[3], 127)

    def test_depth_32_samples_are_multiples_of_1_256(self):
        clip = conformance.make_clip(33, 17, 32, 2, "partial", "f")
        samples = array("f")
        samples.frombytes(clip)
        if sys.byteorder == "big":
            samples.byteswap()
        units = {v * 256 for v in samples}
        self.assertTrue(all(u == int(u) and 0 <= u <= 256 for u in units))
        self.assertIn(256.0, units)
        planes = conformance.make_clip(2, 1, 32, 1, "opaque", "f")
        self.assertEqual(struct.unpack("<2f", planes[24:32]), (1.0, 1.0))

    def test_invert_known_values(self):
        self.assertEqual(list(conformance.invert(bytes([10, 20, 30, 40]), 1, 1, 8)), [245, 235, 225, 40])
        data = struct.pack("<4H", 1000, 2000, 30000, 40000)
        self.assertEqual(struct.unpack("<4H", conformance.invert(data, 1, 1, 16)), (64535, 63535, 35535, 40000))
        data = struct.pack("<4f", 0.25, 0.5, 0.75, 0.125)
        self.assertEqual(struct.unpack("<4f", conformance.invert(data, 1, 1, 32)), (0.75, 0.5, 0.25, 0.125))

    def test_invert_twice_is_identity(self):
        for depth in (8, 16, 32):
            for alpha in ("opaque", "partial", "zero"):
                clip = conformance.make_clip(7, 3, depth, 2, alpha, "twice")
                once = conformance.invert(clip, 7, 3, depth)
                self.assertNotEqual(once, clip)
                self.assertEqual(conformance.invert(once, 7, 3, depth), clip)


class Commands(unittest.TestCase):
    def test_split_plain_and_quoted(self):
        self.assertEqual(conformance.split_command("vfx apply selftest-invert"), ["vfx", "apply", "selftest-invert"])
        self.assertEqual(conformance.split_command('"C:\\Program Files\\x\\vfx.exe" apply "a b"'),
                         ["C:\\Program Files\\x\\vfx.exe", "apply", "a b"])
        self.assertEqual(conformance.split_command('a "say \\"hi\\"" ""'), ["a", 'say "hi"', ""])
        self.assertEqual(conformance.split_command("C:\\bin\\vfx.exe  apply"), ["C:\\bin\\vfx.exe", "apply"])
        with self.assertRaises(ValueError):
            conformance.split_command('"open')

    def test_stage_argv_appends_or_fills(self):
        self.assertEqual(conformance.stage_argv(["s"], 3, 2, 16, params={"b": 1, "a": "x"}),
                         ["s", "--width", "3", "--height", "2", "--depth", "16", "--fps", "24/1",
                          "--param", "a=x", "--param", "b=1"])
        base = conformance.split_command("vfx apply retime --width {width} --height {height} --depth {depth} "
                                         "--fps {fps}")
        self.assertEqual(conformance.stage_argv(base, 5, 4, 8, "30000/1001")[3:],
                         ["--width", "5", "--height", "4", "--depth", "8", "--fps", "30000/1001"])


class Judge(unittest.TestCase):
    def test_each_fixture_is_classified(self):
        width, height, depth, frames = CASES["L3"]["clip"]
        clip = conformance.make_clip(width, height, depth, frames, "partial", "judge")
        size = conformance.frame_size(width, height, depth)
        for case in CASES["L3"]["judge"]:
            with self.subTest(stage=case["stage"], params=case.get("params")):
                result = conformance.run_stage(fixture_argv(case["stage"], width, height, depth, case.get("params")),
                                               clip, size, deadline=30, stall=2)
                self.assertEqual(result["code"] or "ok", case["expect"], conformance.describe(result))
                self.assertFalse(result["stray"])
                if case["expect"] == "ok":
                    self.assertEqual(result["out"], clip)

    def test_hang_is_killed_with_its_whole_tree(self):
        clip = conformance.make_clip(2, 2, 8, 2, "opaque", "hang")
        result = conformance.run_stage(fixture_argv("fixture-hang", 2, 2, 8), clip, 16, deadline=30, stall=1)
        self.assertEqual(result["killed"], "stall")
        self.assertEqual(result["code"], "stage_timeout")
        self.assertLess(result["seconds"], 20)
        pids = conformance.child_pids(result["err"])
        self.assertEqual(len(pids), 1)
        self.assertEqual(conformance.wait_gone(pids), [])

    def test_deadline_applies_without_a_stall_limit(self):
        result = conformance.run_stage(fixture_argv("fixture-hang", 1, 1, 8), b"\0" * 4, 4, deadline=1.5)
        self.assertEqual((result["killed"], result["code"]), ("deadline", "stage_timeout"))
        self.assertEqual(conformance.wait_gone(conformance.child_pids(result["err"])), [])

    def test_an_exited_child_that_is_not_reaped_is_gone(self):
        self.assertFalse(conformance.unreaped(os.getpid()))
        if not os.path.isdir("/proc/self"):
            return
        proc = subprocess.Popen([sys.executable, "-I", "-c", "pass"])
        self.assertEqual(conformance.wait_gone([proc.pid], seconds=10), [])
        self.assertTrue(conformance.unreaped(proc.pid))
        proc.wait()

    def test_a_program_that_cannot_start_is_tool_missing(self):
        missing = os.path.join(tempfile.gettempdir(), "conformance-no-such-program.exe")
        self.assertEqual(conformance.run_stage([missing], b"", 4)["code"], "tool_missing")

    def test_a_stage_whose_reader_closes_ends_by_itself(self):
        size = conformance.frame_size(175, 99, 32)
        clip = conformance.make_clip(175, 99, 32, 12, "partial", "close")
        result = conformance.run_stage(fixture_argv("fixture-copy", 175, 99, 32), clip, size, deadline=30,
                                       read_limit=size)
        self.assertIsNone(result["killed"])
        self.assertEqual(result["exit"], 4)


class Fixtures(unittest.TestCase):
    def test_list_json_is_a_valid_stage_manifest(self):
        code, out, err = run_cli("fixture", "list", "--json")
        self.assertEqual(code, 0, err)
        manifest = json.loads(out)
        schema = HANDOFF.load_schema(os.path.join(REPO, "contracts", "stage-manifest.schema.json"))
        self.assertEqual(HANDOFF.validate(manifest, schema), [])
        self.assertEqual(manifest["contract"], "frames/2")
        self.assertEqual([s["id"] for s in manifest["stages"]], list(conformance.FIXTURES))

    def test_fixture_stage_exits(self):
        cases = [(["apply", "fixture-copy", "--width", "2", "--height", "1", "--depth", "8"], b"\1" * 12, 4),
                 (["apply", "fixture-copy", "--width", "1", "--height", "1", "--depth", "8", "--bad"], b"", 2),
                 (["apply", "fixture-copy", "--width", "1", "--height", "1", "--depth", "12"], b"", 2),
                 (["apply", "fixture-unknown", "--width", "1", "--height", "1", "--depth", "8"], b"", 3)]
        for argv, data, expected in cases:
            with self.subTest(argv=argv):
                proc = subprocess.run([sys.executable, "-I", SCRIPT, "fixture", *argv], input=data,
                                      capture_output=True, timeout=60)
                self.assertEqual(proc.returncode, expected, proc.stderr)
                self.assertRegex(proc.stderr.decode(), r"^error: [a-z_]+: ")

    def test_binary_bytes_pass_unchanged(self):
        data = bytes(range(256)) * 2
        proc = subprocess.run([sys.executable, "-I", SCRIPT, "fixture", "apply", "fixture-copy", "--width", "8",
                               "--height", "8", "--depth", "8"], input=data, capture_output=True, timeout=60)
        self.assertEqual((proc.returncode, proc.stdout), (0, data))


class Accounting(unittest.TestCase):
    def finish(self, levels, as_json=False):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = conformance.finish(levels, "quick", 0.0, as_json, REPO)
        return code, out.getvalue()

    def test_level_status(self):
        lv = conformance.Level("L1", True)
        self.assertEqual(lv.status(), "not run")
        self.assertEqual(lv.document()["not_run"], ["no check ran"])
        lv.check(True, "a")
        self.assertEqual(lv.status(), "pass")
        lv.missing("a provider")
        self.assertEqual(lv.status(), "not run")
        lv.check(False, "b", "why")
        self.assertEqual(lv.status(), "fail")
        self.assertEqual(conformance.Level("L2", False).status(), "not selected")

    def test_exit_codes(self):
        good, missing, bad = (conformance.Level(k, True) for k in ("L0", "L1", "L3"))
        good.check(True, "x")
        missing.check(True, "x")
        missing.missing("peer")
        bad.check(False, "y")
        self.assertEqual(self.finish([good])[0], 0)
        code, text = self.finish([good, missing])
        self.assertEqual(code, 6)
        self.assertIn("NOT RUN peer", text)
        self.assertEqual(self.finish([good, missing, bad])[0], 1)

    def test_json_failure_is_an_error_document(self):
        schema = HANDOFF.load_schema(os.path.join(REPO, "contracts", "error.schema.json"))
        for code_name, make in (("stage_failed", lambda lv: lv.check(False, "y")),
                                ("tool_missing", lambda lv: lv.missing("peer"))):
            lv = conformance.Level("L1", True)
            make(lv)
            code, out = self.finish([lv], as_json=True)
            doc = json.loads(out)
            self.assertEqual(HANDOFF.validate(doc, schema), [])
            self.assertEqual((doc["ok"], doc["error"]["code"]), (False, code_name))


class Cli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="conformance_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def bare_root(self):
        """a root with the cases file but without tools/handoff.py."""
        target = os.path.join(self.tmp, "contracts", "conformance")
        os.makedirs(target)
        shutil.copyfile(os.path.join(REPO, "contracts", "conformance", "cases.json"),
                        os.path.join(target, "cases.json"))
        return self.tmp

    def test_a_conforming_stage_passes(self):
        code, out, err = run_cli("--stage", fixture_string("fixture-copy"), "--json")
        doc = json.loads(out)
        self.assertEqual(code, 0, out[-2000:])
        self.assertEqual(statuses(doc), {"L0": "not selected", "L1": "pass", "L2": "not selected", "L3": "pass",
                                         "L4": "not selected", "L5": "not selected"})

    def test_a_relative_program_path_is_resolved_from_the_working_directory(self):
        here = os.path.relpath(sys.executable)
        self.assertEqual(os.path.normcase(conformance.resolve_program(here)),
                         os.path.normcase(os.path.abspath(sys.executable)))
        self.assertEqual(conformance.resolve_program("no-such-stage-program"), "no-such-stage-program")
        stage = f'"{os.path.relpath(sys.executable, self.tmp)}" -I "{SCRIPT}" fixture apply fixture-copy'
        proc = subprocess.run([sys.executable, "-I", SCRIPT, "--stage", stage, "--json"], capture_output=True,
                              timeout=300, cwd=self.tmp)
        out = proc.stdout.decode("utf-8", "replace")
        self.assertEqual(proc.returncode, 0, out[-2000:])
        self.assertEqual(statuses(json.loads(out))["L1"], "pass")

    def test_a_stage_that_drops_frames_fails(self):
        code, out, err = run_cli("--stage", fixture_string("fixture-truncate"), "--levels", "L1", "--json")
        self.assertEqual(code, 1)
        self.assertEqual(statuses(json.loads(out))["L1"], "fail")

    def test_missing_providers_are_never_a_pass(self):
        code, out, err = run_cli("--root", self.bare_root(), "--levels", "L0,L1,L3,L4")
        self.assertEqual(code, 6, out)
        self.assertNotIn(" pass ", out.replace("result: pass", ""))
        self.assertIn("NOT RUN tools/handoff.py is not present", out)
        self.assertIn("NOT RUN no stage under test was found", out)
        code, out, err = run_cli("--root", self.tmp, "--levels", "L0", "--json")
        doc = json.loads(out)
        self.assertEqual((code, doc["ok"], doc["error"]["code"]), (6, False, "tool_missing"))

    def test_output_is_ascii(self):
        code, out, err = run_cli("--root", self.bare_root(), "--levels", "L0")
        out.encode("ascii")
        for path in (SCRIPT, os.path.abspath(__file__), os.path.join(REPO, "contracts", "conformance", "cases.json")):
            with open(path, "rb") as handle:
                handle.read().decode("ascii")

    def test_usage_errors_exit_2(self):
        for argv in (["--levels", "L9"], ["--quick", "--full"], ["--stage", '"open']):
            with self.subTest(argv=argv):
                self.assertEqual(run_cli(*argv)[0], 2)

    def test_packets_level_passes(self):
        self.assertIsNotNone(HANDOFF, HANDOFF_WHY)
        code, out, err = run_cli("--levels", "L4", "--json")
        doc = json.loads(out)
        self.assertEqual(code, 0, out[-2000:])
        level = [lv for lv in doc["levels"] if lv["level"] == "L4"][0]
        self.assertEqual((level["status"], level["failed"]), ("pass", 0))
        self.assertGreaterEqual(level["passed"], 15)


class Wrappers(unittest.TestCase):
    """the pure parts of L2, L5 and the bench: no wrapper, peer or ffmpeg is needed."""

    def test_join_command_round_trips_through_split_command(self):
        argv = [sys.executable, "-I", "C:\\a dir\\ffstage.py", "apply", "ffmpeg-filter", "--width", "{width}",
                "--param", "graph=scale=16:8", "", 'say "hi"']
        self.assertEqual(conformance.split_command(conformance.join_command(argv)), argv)
        self.assertEqual(conformance.join_command(["vfx", "apply", "x"]), "vfx apply x")

    def test_hflip_mirrors_rows_and_is_an_involution(self):
        for depth in (8, 16):
            clip = conformance.make_clip(5, 3, depth, 2, "partial", "hflip")
            once = conformance.hflip(clip, 5, 3, depth)
            self.assertNotEqual(once, clip)
            self.assertEqual(conformance.hflip(once, 5, 3, depth), clip)
            bpp = conformance.BYTES_PER_PIXEL[depth]
            self.assertEqual(once[:bpp], clip[4 * bpp:5 * bpp])

    def test_first_difference(self):
        self.assertIsNone(conformance.first_difference(b"abc", b"abc"))
        self.assertEqual(conformance.first_difference(b"abc", b"abd"), 2)
        self.assertEqual(conformance.first_difference(b"ab", b"abc"), 2)
        self.assertEqual(conformance.first_difference(None, b"a"), 0)

    def test_exit_codes_come_from_the_error_schema(self):
        schema = HANDOFF.load_schema(os.path.join(REPO, "contracts", "error.schema.json"))
        exits = conformance.code_exits(schema)
        self.assertEqual({k: exits[k] for k in ("stage_failed", "usage", "unsupported", "bad_input", "broken_pipe",
                                                "partial_frame", "stage_timeout", "tool_missing")},
                         {"stage_failed": 1, "usage": 2, "unsupported": 3, "bad_input": 4, "broken_pipe": 4,
                          "partial_frame": 4, "stage_timeout": 5, "tool_missing": 6})
        for case in CASES["L5"]["errors"]:
            self.assertIn(case["code"], exits)

    def test_l5_and_bench_cases(self):
        for case in CASES["L5"]["errors"]:
            self.assertTrue(case["stage"].startswith("fixture-"), case)
        self.assertTrue(set(CASES["L2"]["depths"]["full"]) <= {8, 16})
        bench = CASES["bench"]
        self.assertEqual(len(bench["sizes"]), len(bench["frames"]))
        self.assertEqual(bench["chain"], [0, 1, 2, 3, 4])
        self.assertEqual(bench["spool_stage"], "retime")
        self.assertIn("machine", bench["columns"])

    def tools(self):
        t = object.__new__(conformance.Tools)
        t.why, t.peer = {}, {}
        t.ffmpeg, t.ffprobe, t.vfx, t.nh, t.apply_py, t.ffstage = "ff", "fp", "vfx", "nh", "apply.py", "ffstage.py"
        return t

    def test_neutral_handoff_gets_its_stages_around_the_vfx_marker(self):
        argv = self.tools().wrapper_argv("sapphire", "in.mkv", "selftest-invert", "out.mkv",
                                         [("A", None, "pre"), ("B", "declared", "post")], ["--json"])
        stages = [argv[i + 1] for i, word in enumerate(argv) if word == "--stage"]
        self.assertEqual(stages, ["A", "vfx", "B"])
        self.assertEqual(argv[argv.index("B") + 1:], ["--geometry", "declared", "--json"])
        alone = self.tools().wrapper_argv("sapphire", "in.mkv", None, "out.mkv", [("A", None, "pre")])
        self.assertNotIn("vfx", alone)
        self.assertNotIn("--vfx", alone)

    def test_apply_py_gets_pre_and_post_stages(self):
        argv = self.tools().wrapper_argv("tools/apply.py", "in.mkv", "retime", "out.mkv",
                                         [("A", None, "pre"), ("B", "declared", "post")])
        self.assertEqual(argv[1:5], ["-I", "apply.py", "in.mkv", "retime"])
        self.assertEqual(argv[-6:], ["--pre-stage", "A", "--post-stage", "B", "--geometry", "declared"])

    def test_needs_name_the_absent_peer(self):
        t = self.tools()
        t.why, t.peer = {"vfx": "twixtor vfx: not found"}, {"vfx": "twixtor"}
        self.assertEqual(t.need(["vfx", "ffmpeg"]), [("twixtor vfx: not found", "twixtor")])


class AbsentPeers(unittest.TestCase):
    def finish(self, levels, allow):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = conformance.finish(levels, "quick", 0.0, False, REPO, allow)
        return code, out.getvalue()

    def test_only_peer_reasons_are_excused(self):
        def peer_level():
            lv = conformance.Level("L2", True)
            lv.missing("needs twixtor", "twixtor")
            return lv

        self.assertEqual(self.finish([peer_level()], False)[0], 6)
        code, text = self.finish([peer_level()], True)
        self.assertEqual(code, 0)
        self.assertIn("NOT RUN needs twixtor", text)
        self.assertIn("absent peer twixtor", text)
        mixed = peer_level()
        mixed.missing("ffmpeg is missing")
        self.assertEqual(self.finish([mixed], True)[0], 6)
        empty = conformance.Level("L1", True)
        self.assertEqual(self.finish([peer_level(), empty], True)[0], 6)
        failed = peer_level()
        failed.check(False, "x")
        self.assertEqual(self.finish([failed], True)[0], 1)

    def test_missing_reasons_are_recorded_once(self):
        lv = conformance.Level("L5", True)
        lv.missing("a", "twixtor")
        lv.missing("a", "twixtor")
        self.assertEqual(lv.document()["not_run"], ["a"])


class Meter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="conformance_meter_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_meter_reports_exit_time_memory_and_disk(self):
        program = ("import os, time; p = os.path.join(os.environ['TMP'], 'spool'); "
                   "open(p, 'wb').write(bytes(3 << 20)); time.sleep(0.5); os.remove(p); "
                   "assert os.environ['TEMP'] == os.environ['TMPDIR'] == os.environ['TMP']")
        temp = os.path.join(self.tmp, "t")
        code, out, err = run_cli("meter", "--temp", temp, "--", sys.executable, "-I", "-c", program, timeout=60)
        self.assertEqual(code, 0, err)
        doc = json.loads(out.strip().splitlines()[-1])
        self.assertEqual(doc["exit"], 0, doc)
        self.assertGreaterEqual(doc["disk_mb"], 2.9)
        self.assertGreater(doc["peak_mb"], 0)
        self.assertIn(doc["peak_kind"], ("job-commit", "max-rss"))

    def test_meter_kills_a_command_past_its_deadline(self):
        code, out, err = run_cli("meter", "--temp", os.path.join(self.tmp, "t"), "--deadline", "1", "--",
                                 sys.executable, "-I", "-c", "import time; time.sleep(60)", timeout=60)
        doc = json.loads(out.strip().splitlines()[-1])
        self.assertIsNone(doc["exit"])
        self.assertLess(doc["seconds"], 30)


class WrapperCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="conformance_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        target = os.path.join(self.tmp, "contracts", "conformance")
        os.makedirs(target)
        shutil.copyfile(os.path.join(REPO, "contracts", "conformance", "cases.json"),
                        os.path.join(target, "cases.json"))

    def test_wrapper_levels_without_the_control_plane_are_not_run(self):
        code, out, err = run_cli("--root", self.tmp, "--levels", "L2,L5", "--allow-absent-peer")
        self.assertEqual(code, 6, out)
        self.assertIn("L2 wrappers           NOT RUN", out)
        self.assertIn("L5 frames/2           NOT RUN", out)

    def test_bench_without_programs_writes_nothing(self):
        code, out, err = run_cli("--root", self.tmp, "--bench")
        self.assertEqual(code, 6, out)
        self.assertIn("docs/handoff-bench.tsv was not written", out)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "docs", "handoff-bench.tsv")))


if __name__ == "__main__":
    unittest.main(verbosity=1)
