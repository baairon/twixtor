#!/usr/bin/env python3
"""Tests of tools/parity_lock.py, with a tiny fake vfx (a Python script) in place of the renderer.

Run: python tools/test_parity_lock.py   (standard library unittest; the base test needs git and skips
without it, on a throw away repository it creates in a temporary folder)
"""

import contextlib
import io
import json
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

import parity_lock as pl  # noqa: E402

# The fake renderer: stdout is a hash of its arguments (paths reduced to file names) and its input, so
# the same case gives the same bytes. FAKE_VFX_MODE changes it: `drift` alters every retime output,
# `note` adds a note line to every case that sets n=1, `fixed` renders the cases that set x=refuse.
FAKE_VFX = r'''
import hashlib, os, sys
args = sys.argv[1:]
mode = os.environ.get("FAKE_VFX_MODE", "")
if os.environ.get("RAYON_NUM_THREADS"):
    sys.stderr.write("vfx: error: RAYON_NUM_THREADS leaked\n")
    sys.exit(9)
params = [args[i + 1] for i, a in enumerate(args) if a == "--param"]
if "x=refuse" in params and mode != "fixed":
    sys.stderr.write("vfx: note: e: G1 t: a stated rule\nvfx: error: frame 0: unsupported: x\n")
    sys.exit(3)
if "n=1" in params and mode == "note":
    sys.stderr.write("vfx: note: e: G1 t: a stated rule\n")
data = b""
if "--in" in args:
    with open(args[args.index("--in") + 1], "rb") as h:
        data = h.read()
else:
    data = sys.stdin.buffer.read()
words = [os.path.basename(a) if os.sep in a or "/" in a else a for a in args]
out = hashlib.sha256(("|".join(words)).encode() + data).digest()
if mode == "drift" and args[1] == "retime":
    out = out[::-1]
sys.stdout.buffer.write(out * 3)
'''

# A fake fx-parity whose gen writes a small case file with matte layers.
FAKE_PARITY = r'''
import json, os, sys
args = sys.argv[1:]
out = args[args.index("--out") + 1]
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "clip.raw"), "wb") as h:
    h.write(b"\x01\x02\x03\x04")
cases = [
    {"id": "retime.all-default.a", "effect": "retime", "kind": "temporal", "width": 2, "height": 2, "depth": 8,
     "params": [], "fps": "24", "frames": 2, "input": "clip.raw"},
    {"id": "create-vectors.all-max.a", "effect": "create-vectors", "kind": "temporal", "width": 2, "height": 2,
     "depth": 8, "params": ["x=1"], "fps": "24", "frames": 2, "input": "clip.raw",
     "layers": {"fg1_matte": "clip.raw", "alt_source_a": "clip.raw"}},
]
with open(os.path.join(out, "cases.json"), "w") as h:
    json.dump({"format": 1, "args": args, "cases": cases}, h)
'''


def remove_tree(path):
    """Removes a folder, read only files (git objects on Windows) included."""
    for dirpath, _, files in os.walk(path):
        for f in files:
            try:
                os.chmod(os.path.join(dirpath, f), stat.S_IWRITE | stat.S_IREAD)
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)


def temporal(cid, effect, params=(), layers=None, fps="24"):
    c = {"id": cid, "effect": effect, "kind": "temporal", "width": 4, "height": 2, "depth": 8,
         "params": list(params), "fps": fps, "frames": 3, "input": "clip.raw"}
    if layers:
        c["layers"] = dict(layers)
    return c


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="lock-test-"))
        self.vfx = self.tmp / "vfx.py"
        self.vfx.write_text(FAKE_VFX, encoding="utf-8")
        self.out = self.tmp / "lock"
        self.env = dict(os.environ)
        self.addCleanup(remove_tree, self.tmp)
        self.addCleanup(self.restore_env)
        os.environ.pop("FAKE_VFX_MODE", None)

    def restore_env(self):
        os.environ.clear()
        os.environ.update(self.env)

    def corpus(self):
        """L1 with a frame case and four temporal ones, L2 with three temporal ones."""
        for name in ("L1", "L2"):
            d = self.out / name
            d.mkdir(parents=True)
            (d / "clip.raw").write_bytes(bytes(range(96)))
            (d / "still.raw").write_bytes(bytes(range(32)))
        l1 = [
            {"id": "selftest-invert.all-default.s", "effect": "selftest-invert", "kind": "frame", "width": 4,
             "height": 2, "depth": 8, "params": [], "input": "still.raw"},
            temporal("retime.all-default.c", "retime"),
            temporal("retime.n.c", "retime", ["n=1"]),
            temporal("create-vectors.x.c", "create-vectors", ["x=refuse"]),
            temporal("vectors-in.all-default.c", "vectors-in", layers={"vectors_prev": "clip.raw"}),
        ]
        l2 = [
            temporal("retime.all-min.c", "retime"),
            temporal("retime-fgbg.x.c", "retime-fgbg", ["x=refuse"]),
            temporal("vectors-in.all-min.c", "vectors-in"),
        ]
        pl.write_json(self.out / "L1" / "cases.json", {"format": 1, "cases": l1})
        pl.write_json(self.out / "L2" / "cases.json", {"format": 1, "cases": l2})

    def main(self, *words):
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            code = pl.main([str(w) for w in words])
        return code, buf_out.getvalue(), buf_err.getvalue()


class Invocation(unittest.TestCase):
    def test_temporal_case_words(self):
        case = temporal("retime.k.c", "retime", ["speed_percent=50", "vectors=5"],
                        {"fg1_matte": "m.raw", "alt_source_a": "a.raw"}, fps="30000/1001")
        args, stdin = pl.invocation(case, pathlib.Path("set"))
        self.assertIsNone(stdin)
        self.assertEqual(args, [
            "apply", "retime", "--width", "4", "--height", "2", "--depth", "8",
            "--param", "speed_percent=50", "--param", "vectors=5", "--fps", "30000/1001", "--frames", "3",
            "--layer", "alt_source_a=" + str(pathlib.Path("set") / "a.raw"),
            "--layer", "fg1_matte=" + str(pathlib.Path("set") / "m.raw"),
            "--in", str(pathlib.Path("set") / "clip.raw")])

    def test_draft_adds_the_quality_last(self):
        args, _ = pl.invocation(temporal("retime.a.c", "retime"), pathlib.Path("s"), draft=True)
        self.assertEqual(args[-2:], ["--quality", "draft"])
        self.assertNotIn("--quality", pl.invocation(temporal("retime.a.c", "retime"), pathlib.Path("s"))[0])

    def test_frame_case_reads_stdin_and_has_no_rate(self):
        tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(remove_tree, tmp)
        (tmp / "still.raw").write_bytes(b"abcd")
        case = {"id": "s", "effect": "selftest-invert", "kind": "frame", "width": 1, "height": 1, "depth": 8,
                "params": ["gain=2"], "input": "still.raw", "fps": "24"}
        args, stdin = pl.invocation(case, tmp)
        self.assertEqual(stdin, b"abcd")
        self.assertEqual(args, ["apply", "selftest-invert", "--width", "1", "--height", "1", "--depth", "8",
                                "--param", "gain=2"])

    def test_a_case_without_a_rate_gets_no_fps(self):
        args, _ = pl.invocation(temporal("r", "retime", fps=None), pathlib.Path("s"))
        self.assertNotIn("--fps", args)

    def test_program_runs_python_scripts_under_this_python(self):
        self.assertEqual(pl.program("x/vfx.py"), [sys.executable, os.path.abspath("x/vfx.py")])
        self.assertEqual(pl.program("x/vfx"), [os.path.abspath("x/vfx")])
        self.assertEqual(pl.program("vfx"), ["vfx"])  # a bare name is looked up on PATH

    def test_program_adds_exe_on_windows(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(remove_tree, tmp)
        open(os.path.join(tmp, "tool" + pl.EXE), "w").close()
        self.assertEqual(pl.program(os.path.join(tmp, "tool")), [os.path.join(os.path.abspath(tmp), "tool" + pl.EXE)])

    def test_cargo_command(self):
        self.assertEqual(pl.cargo_command("cargo", "stable-x86_64-pc-windows-gnu"),
                         ["cargo", "+stable-x86_64-pc-windows-gnu", "build", "--release", "-p", "fx-cli",
                          "-p", "fx-parity"])
        self.assertEqual(pl.cargo_command("cargo", None)[:2], ["cargo", "build"])


class Rules(unittest.TestCase):
    def test_strip_mattes(self):
        c = {"layers": {"fg1_matte": "a", "fg3_matte": "b", "alt_source_a": "c"}}
        self.assertTrue(pl.strip_mattes(c))
        self.assertEqual(c["layers"], {"alt_source_a": "c"})
        c = {"layers": {"fg2_matte": "a"}}
        self.assertTrue(pl.strip_mattes(c))
        self.assertNotIn("layers", c)
        c = {"layers": {"vectors_prev": "a"}}
        self.assertFalse(pl.strip_mattes(c))
        self.assertEqual(c, {"layers": {"vectors_prev": "a"}})
        self.assertFalse(pl.strip_mattes({}))

    def test_compare(self):
        base = {"exit": 0, "out": "a", "size": 4, "err": "", "err_line": ""}
        self.assertEqual(pl.compare(base, dict(base)), [])
        self.assertEqual(pl.compare(base, dict(base, out="b")), ["stdout differs (4 -> 4 bytes)"])
        self.assertEqual(pl.compare(base, dict(base, size=5)), ["stdout differs (4 -> 5 bytes)"])
        self.assertEqual(pl.compare(base, dict(base, err="h", err_line="vfx: note: x")),
                         ["stderr differs: vfx: note: x"])
        why = pl.compare(base, dict(base, exit=3, out="c", size=0, err="h", err_line="vfx: error: no"))
        self.assertEqual(why[0], "exit 3: vfx: error: no")
        self.assertEqual(len(why), 3)

    def test_first_error_skips_notes(self):
        self.assertEqual(pl.first_error(b"vfx: note: a\nvfx: error: frame 0: b\n"), "vfx: error: frame 0: b")
        self.assertEqual(pl.first_error(b"vfx: note: a\npanic here\n"), "panic here")
        self.assertEqual(pl.first_error(b"vfx: note: a\n"), "vfx: note: a")
        self.assertEqual(pl.first_error(b""), "")

    def test_corpus_definition_matches_the_constants(self):
        d = pl.corpus_definition()
        self.assertEqual(sorted(d), ["L1", "L2", "L3"])
        self.assertEqual(d["L1"]["gen"], ["--effects", "retime,create-vectors,retime-fgbg,vectors-in,selftest-invert,"
                                          "selftest-shift,selftest-integrate,selftest-threads"])
        self.assertIn("2x2,3x2,4x4,5x4,8x8,9x8,16x16,17x16", d["L2"]["gen"])
        self.assertEqual(d["L2"]["gen"][-2:], ["--max-cases-per-effect", "0"])
        self.assertEqual(d["L3"]["effects"], ["retime", "retime-fgbg", "create-vectors"])


class Flow(Base):
    def test_work_items_pick_l3_from_the_temporal_draft_effects(self):
        self.corpus()
        keys = [k for k, _, _, _ in pl.work_items(self.out, ["L1", "L2", "L3"])]
        self.assertEqual(keys[:5], ["L1:selftest-invert.all-default.s", "L1:retime.all-default.c", "L1:retime.n.c",
                                    "L1:create-vectors.x.c", "L1:vectors-in.all-default.c"])
        self.assertEqual([k for k in keys if k.startswith("L3:")], ["L3:retime.all-min.c", "L3:retime-fgbg.x.c"])
        drafts = [d for k, _, _, d in pl.work_items(self.out, ["L3"])]
        self.assertEqual(drafts, [True, True])

    def test_refs_check_refused_record(self):
        self.corpus()
        code, out, _ = self.main("refs", "--out", self.out, "--vfx", self.vfx, "--jobs", "2")
        self.assertEqual(code, 0)
        self.assertIn("baseline: 7 rendered, 3 refused or failed, 0 rendered with stderr output", out)
        refs = json.loads((self.out / "refs.json").read_text(encoding="utf-8"))
        refused = json.loads((self.out / "refused.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(refused), ["L1:create-vectors.x.c", "L2:retime-fgbg.x.c", "L3:retime-fgbg.x.c"])
        self.assertEqual(refused["L2:retime-fgbg.x.c"]["err_line"], "vfx: error: frame 0: unsupported: x")
        # the draft run is a different invocation, so its bytes differ from the L2 run of the same case
        self.assertNotEqual(refs["L2:retime.all-min.c"]["out"], refs["L3:retime.all-min.c"]["out"])

        os.environ["RAYON_NUM_THREADS"] = "3"  # the lock runs at the default thread count
        code, out, _ = self.main("check", "--out", self.out, "--vfx", self.vfx, "--jobs", "2")
        self.assertEqual(code, 0, out)
        self.assertIn("P2: 7 identical, 0 differ, of 7", out)
        del os.environ["RAYON_NUM_THREADS"]

        code, out, err = self.main("record", "--out", self.out, "--data", self.tmp / "lock.json")
        self.assertEqual(code, 3)  # no base.json and no --ref: setup error
        self.assertIn("parity_lock: error: base output missing", err)
        pl.write_json(self.out / "base.json", {"ref": "abc1234", "commit": "abc1234" + "0" * 33})
        code, out, _ = self.main("record", "--out", self.out, "--data", self.tmp / "lock.json", "--date", "2026-01-02")
        self.assertEqual(code, 0, out)
        rec = json.loads((self.tmp / "lock.json").read_text(encoding="utf-8"))
        self.assertEqual(rec["date"], "2026-01-02")
        self.assertEqual(rec["baseline"]["ref"], "abc1234")
        self.assertEqual(rec["corpora"]["L1"], {"rendered_at_baseline": 4, "identical": 4, "differ": 0,
                                                "refused_at_baseline": 1})
        self.assertEqual(rec["corpora"]["L2"]["rendered_at_baseline"], 2)
        self.assertEqual(rec["corpora"]["L3"], {"rendered_at_baseline": 1, "identical": 1, "differ": 0,
                                                "refused_at_baseline": 1})
        self.assertEqual(rec["definition"], json.loads(json.dumps(pl.corpus_definition())))

        os.environ["FAKE_VFX_MODE"] = "fixed"
        code, out, _ = self.main("refused", "--out", self.out, "--vfx", self.vfx)
        self.assertEqual(code, 0, out)
        self.assertIn("P3: 3 of 3 previously refused cases now render; 0 still fail", out)
        code, _, _ = self.main("record", "--out", self.out, "--data", self.tmp / "lock.json", "--date", "2026-01-02")
        rec = json.loads((self.tmp / "lock.json").read_text(encoding="utf-8"))
        self.assertEqual([rec["corpora"][s]["refused_now_render"] for s in ("L1", "L2", "L3")], [1, 1, 1])

    def test_check_finds_changed_bytes_and_new_notes(self):
        self.corpus()
        self.assertEqual(self.main("refs", "--out", self.out, "--vfx", self.vfx)[0], 0)
        os.environ["FAKE_VFX_MODE"] = "drift"
        code, out, _ = self.main("check", "--out", self.out, "--vfx", self.vfx)
        self.assertEqual(code, 1)
        diff = json.loads((self.out / "check.json").read_text(encoding="utf-8"))["differ"]
        self.assertEqual(sorted(diff), ["L1:retime.all-default.c", "L1:retime.n.c", "L2:retime.all-min.c",
                                        "L3:retime.all-min.c"])
        self.assertIn("stdout differs", diff["L1:retime.n.c"])
        os.environ["FAKE_VFX_MODE"] = "note"
        code, out, _ = self.main("check", "--out", self.out, "--vfx", self.vfx)
        self.assertEqual(code, 1)
        diff = json.loads((self.out / "check.json").read_text(encoding="utf-8"))["differ"]
        self.assertEqual(diff, {"L1:retime.n.c": "stderr differs: vfx: note: e: G1 t: a stated rule"})

    def test_refused_still_failing_exits_1(self):
        self.corpus()
        self.main("refs", "--out", self.out, "--vfx", self.vfx)
        code, out, _ = self.main("refused", "--out", self.out, "--vfx", self.vfx)
        self.assertEqual(code, 1)
        self.assertIn("P3: 0 of 3 previously refused cases now render; 3 still fail", out)
        self.assertIn("    3  vfx: error: frame 0: unsupported: x", out)

    def test_record_refuses_a_partial_check(self):
        self.corpus()
        self.main("refs", "--out", self.out, "--vfx", self.vfx)
        self.main("check", "--out", self.out, "--vfx", self.vfx, "--limit", "2")
        pl.write_json(self.out / "base.json", {"ref": "r", "commit": "c"})
        code, _, err = self.main("record", "--out", self.out, "--data", self.tmp / "lock.json")
        self.assertEqual(code, 3)
        self.assertIn("did not cover every corpus", err)
        self.main("check", "--out", self.out, "--vfx", self.vfx, "--sets", "L1")
        code, _, err = self.main("record", "--out", self.out, "--data", self.tmp / "lock.json")
        self.assertEqual(code, 3)

    def test_missing_corpus_is_a_setup_error(self):
        code, _, err = self.main("refs", "--out", self.out, "--vfx", self.vfx)
        self.assertEqual(code, 3)
        self.assertIn("run `parity_lock.py corpus` first", err)
        code, _, err = self.main("check", "--out", self.out, "--vfx", self.vfx)
        self.assertEqual(code, 3)
        code, _, err = self.main("corpus", "--out", self.out)
        self.assertEqual(code, 3)
        self.assertIn("no baseline fx-parity", err)

    def test_usage_errors_exit_2(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as e:
                pl.main(["check", "--out", str(self.out)])
        self.assertEqual(e.exception.code, 2)

    def test_corpus_strips_mattes_and_keeps_the_original(self):
        parity = self.tmp / "fx-parity.py"
        parity.write_text(FAKE_PARITY, encoding="utf-8")
        code, out, _ = self.main("corpus", "--out", self.out, "--parity", parity, "--vfx", self.vfx)
        self.assertEqual(code, 0, out)
        self.assertIn("L1: 2 cases, 1 had matte layers removed", out)
        cases = json.loads((self.out / "L2" / "cases.json").read_text(encoding="utf-8"))
        orig = json.loads((self.out / "L2" / "cases.orig.json").read_text(encoding="utf-8"))
        self.assertEqual(cases["cases"][1]["layers"], {"alt_source_a": "clip.raw"})
        self.assertIn("fg1_matte", orig["cases"][1]["layers"])
        self.assertEqual(cases["args"][cases["args"].index("--sizes") + 1], "2x2,3x2,4x4,5x4,8x8,9x8,16x16,17x16")
        code, out, _ = self.main("corpus", "--out", self.out, "--parity", parity, "--vfx", self.vfx)
        self.assertIn("L1: exists, kept", out)


# A fake cargo: checks the CI profile, then writes the two binaries where cargo would.
FAKE_CARGO = r'''
import os, sys
assert sys.argv[1:] == ["build", "--release", "-p", "fx-cli", "-p", "fx-parity"], sys.argv
for k, v in (("CARGO_INCREMENTAL", "0"), ("CARGO_PROFILE_RELEASE_LTO", "false"),
             ("CARGO_PROFILE_RELEASE_CODEGEN_UNITS", "16")):
    assert os.environ.get(k) == v, k
assert os.path.isfile("marker.txt"), "not run in the worktree"
rel = os.path.join(os.environ["CARGO_TARGET_DIR"], "release")
os.makedirs(rel, exist_ok=True)
ext = ".exe" if os.name == "nt" else ""
for name in ("vfx", "fx-parity"):
    with open(os.path.join(rel, name + ext), "w") as h:
        h.write(open("marker.txt").read())
'''


@unittest.skipIf(shutil.which("git") is None, "git is not installed")
class BaseBuild(Base):
    def git(self, repo, *words):
        return subprocess.run(["git", "-C", str(repo)] + list(words), check=True, capture_output=True,
                              text=True).stdout

    def test_base_builds_in_a_temporary_worktree_and_removes_it(self):
        repo = self.tmp / "repo"
        repo.mkdir()
        self.git(repo, "init", "-q")
        (repo / "marker.txt").write_text("first", encoding="utf-8")
        self.git(repo, "add", "marker.txt")
        self.git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "one")
        first = self.git(repo, "rev-parse", "HEAD").strip()
        (repo / "marker.txt").write_text("second", encoding="utf-8")
        self.git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-am", "two")
        cargo = self.tmp / "cargo.py"
        cargo.write_text(FAKE_CARGO, encoding="utf-8")
        code, out, err = self.main("base", "--ref", first, "--out", self.out, "--repo", repo, "--cargo", cargo)
        self.assertEqual(code, 0, out + err)
        ext = ".exe" if os.name == "nt" else ""
        self.assertEqual((self.out / "base" / ("vfx" + ext)).read_text(encoding="utf-8"), "first")
        self.assertEqual((self.out / "base" / ("fx-parity" + ext)).read_text(encoding="utf-8"), "first")
        self.assertEqual(json.loads((self.out / "base.json").read_text(encoding="utf-8")),
                         {"ref": first, "commit": first})
        self.assertEqual(len(self.git(repo, "worktree", "list").strip().splitlines()), 1)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["base", "base.json", "target"])
        self.assertEqual((repo / "marker.txt").read_text(encoding="utf-8"), "second")
        # record names the baseline by --ref when the lock folder was built elsewhere
        (self.out / "base.json").unlink()
        for name, data in (("refs.json", {"L1:a": {}}), ("refused.json", {}),
                           ("check.json", {"complete": True, "compared": {"L1": 1}, "identical": {"L1": 1}})):
            pl.write_json(self.out / name, data)
        code, out, err = self.main("record", "--out", self.out, "--ref", first[:7], "--repo", repo,
                                   "--data", self.tmp / "lock.json", "--date", "2026-01-02")
        self.assertEqual(code, 0, out + err)
        rec = json.loads((self.tmp / "lock.json").read_text(encoding="utf-8"))
        self.assertEqual(rec["baseline"], {"ref": first[:7], "commit": first})
        self.assertNotIn("refused_now_render", rec["corpora"]["L1"])

    def test_a_failed_build_still_removes_the_worktree(self):
        repo = self.tmp / "repo"
        repo.mkdir()
        self.git(repo, "init", "-q")
        (repo / "a.txt").write_text("a", encoding="utf-8")
        self.git(repo, "add", "a.txt")
        self.git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "one")
        cargo = self.tmp / "cargo.py"
        cargo.write_text("import sys\nsys.exit(101)\n", encoding="utf-8")
        code, _, err = self.main("base", "--ref", "HEAD", "--out", self.out, "--repo", repo, "--cargo", cargo)
        self.assertEqual(code, 3)
        self.assertIn("exited with code 101", err)
        self.assertEqual(len(self.git(repo, "worktree", "list").strip().splitlines()), 1)
        self.assertFalse(any(p.name.startswith("worktree-") for p in self.out.iterdir()))
        code, _, err = self.main("base", "--ref", "nope", "--out", self.out, "--repo", repo, "--cargo", cargo)
        self.assertEqual(code, 3)


if __name__ == "__main__":
    unittest.main()
