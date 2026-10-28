#!/usr/bin/env python3
"""Tests of tools/gen_docs.py, tools/docgen/ and tools/check_docs.py.

Run: python -I tools/test_docgen.py   (standard library unittest, no skips; most tests run on a small
synthetic repository in a temporary folder, a few on this tree)
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
sys.dont_write_bytecode = True

import check_docs  # noqa: E402
import gen_docs  # noqa: E402
from docgen import data as dg  # noqa: E402
from docgen import render  # noqa: E402

NL = "\n"
CONTROLS = """effect\tid\tkind
e1\ta\tfloat
e1\tb\tpopup
e1\tlay\tlayer
e2\ta\tfloat
e2\tpt1_use\tpopup
e2\tpt2_use\tpopup
e2\tpt3_use\tpopup
"""
GAPS = """id\ttitle\teffects\tcontrols
G1\tfirst open point\te1,e2\ta
G2\tsecond open point\te2\tpt1_use,pt2_use,pt3_use
"""
REFERENCES = """family\tgap\tcase\teffect\tcontrols\tresult
fam-one\tG1\tc1\te1\ta,lay\tpass
fam-one\tG1\tc2\te2\ta\tfail
fam-two\tG2\tc3\te2\tpt1_use\tmissing
"""
LOCK = {"baseline": {"ref": "abc", "commit": "abcdef0123456789"}, "date": "2026-01-02",
        "corpora": {"L1": {"rendered_at_baseline": 10, "identical": 10, "differ": 0, "refused_at_baseline": 2,
                           "refused_now_render": 1},
                    "L2": {"rendered_at_baseline": 1000, "identical": 999, "differ": 1, "refused_at_baseline": 0},
                    "L3": {"rendered_at_baseline": 5, "identical": 5, "differ": 0, "refused_at_baseline": 3,
                           "refused_now_render": 3}}}
GENERATOR = '''"""Model of one stage.

  b"ABC1", u32 ncases, then per case ...
A differential one: magic b"DIF1", u32 width, not bit exact.
"""
'''
RUST = "#[cfg(test)]\nmod tests {\n    #[test]\n    fn one() {}\n    #[test]\n    #[ignore]\n    fn two() {}\n" \
       "    // #[test] in a comment is no test\n}\n"
README = """# r

<!--f:controls_total-->0<!--/f--> controls, <!--f:gaps_open-->0<!--/f--> gaps, <!--f:unproven-->0<!--/f--> unproven.

<!--gen:gaps-->
<!--/gen:gaps-->
"""


def write(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as h:
        h.write(text)


def read(root, rel):
    with open(os.path.join(root, rel), encoding="utf-8", newline="") as h:
        return h.read()


def quiet(fn, *a):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = fn(*a)
    return code, out.getvalue(), err.getvalue()


class Synthetic(unittest.TestCase):
    """A small repository: snapshots, one crate with a fixture folder, one Python test file, two pages."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="docgen-test-")
        self.addCleanup(shutil.rmtree, self.root, True)
        write(self.root, "docs/data/controls.tsv", CONTROLS)
        write(self.root, "docs/data/gaps.tsv", GAPS)
        write(self.root, "docs/data/references.tsv", REFERENCES)
        write(self.root, "docs/data/lock.json", json.dumps(LOCK))
        write(self.root, "crates/k/Cargo.toml", '[package]\nname = "kay"\n')
        write(self.root, "crates/k/src/lib.rs", RUST)
        write(self.root, "crates/k/tests/fixtures/gen.py", GENERATOR)
        write(self.root, "crates/k/tests/fixtures/a.hex", "41424331" + "03000000" + NL + "ffffffff" + NL)
        write(self.root, "crates/k/tests/fixtures/d.hex", "4449463102000000ff" + NL)
        write(self.root, "plugins/p/Cargo.toml", '[package]\nname = "pee"\n')
        write(self.root, "plugins/p/tests/t.rs", "#[test]\nfn t() {}\n")
        write(self.root, "plugins/p/tests/fixtures/o.hex", "5a5a5a5a0100000000" + NL)
        write(self.root, "tools/test_x.py", "class T:\n    def test_a(self):\n        pass\n\n    def test_b(self):\n"
                                            "        pass\n\n    def helper(self):\n        pass\n")
        write(self.root, "README.md", README)
        write(self.root, "docs/p.md", "# P\n\n<!--f:lock_identical-->x<!--/f--> of <!--f:lock_total-->x<!--/f-->\n")

    def gen(self, *flags):
        return quiet(gen_docs.main, list(flags) + ["--root", self.root])


class Generation(Synthetic):
    def test_generate_then_check_is_clean(self):
        code, out, _ = self.gen()
        self.assertEqual(code, 0, out)
        text = read(self.root, "README.md")
        self.assertIn("<!--f:controls_total-->7<!--/f--> controls, <!--f:gaps_open-->2<!--/f--> gaps", text)
        self.assertIn("<!--f:unproven-->5<!--/f-->", text)
        self.assertIn("| G1 | first open point | `e1`, `e2` | `a` | 1 of 2 |", text)
        self.assertIn("| G2 | second open point | `e2` | `pt1_use` to `pt3_use` | 0 of 1 |", text)
        self.assertIn("<!--f:lock_identical-->1,014<!--/f--> of <!--f:lock_total-->1,015<!--/f-->",
                      read(self.root, "docs/p.md"))
        facts = json.loads(read(self.root, "docs/data/facts.json"))
        self.assertEqual(facts["generator"], "tools/gen_docs.py")
        self.assertTrue(all(f["source"] for f in facts["facts"].values()))
        self.assertEqual(sorted(facts["blocks"]), ["families", "fixtures", "gaps", "lock", "tests"])
        code, out, _ = self.gen("--check")
        self.assertEqual(code, 0, out)
        self.assertIn("0 drifted, 0 unknown keys", out)
        again, _, _ = self.gen()
        self.assertEqual(read(self.root, "README.md"), text)

    def test_check_finds_a_mutated_marker(self):
        self.gen()
        text = read(self.root, "README.md")
        write(self.root, "README.md", text.replace("<!--f:gaps_open-->2<!--/f-->", "<!--f:gaps_open-->3<!--/f-->"))
        code, out, _ = self.gen("--check")
        self.assertEqual(code, 1)
        self.assertIn("1 drifted", out)
        self.assertIn("  README.md", out)
        write(self.root, "README.md", text.replace("| 1 of 2 |", "| 2 of 2 |"))
        self.assertEqual(self.gen("--check")[0], 1)
        write(self.root, "README.md", text)
        self.assertEqual(self.gen("--check")[0], 0)

    def test_check_finds_changed_data(self):
        self.gen()
        write(self.root, "docs/data/controls.tsv", CONTROLS + "e2\tc\tcheckbox\n")
        code, out, _ = self.gen("--check")
        self.assertEqual(code, 1)
        self.assertIn("README.md", out)
        self.assertIn("docs/data/facts.json", out)
        write(self.root, "plugins/p/tests/t.rs", "#[test]\nfn t() {}\n#[test]\nfn u() {}\n")
        self.assertEqual(self.gen("--check")[0], 1)

    def test_unknown_keys_fail_and_stay_untouched(self):
        write(self.root, "docs/q.md", "<!--f:no_such_key-->7<!--/f-->\n<!--gen:nope-->\nold\n<!--/gen:nope-->\n")
        code, out, _ = self.gen("--check")
        self.assertEqual(code, 1)
        self.assertIn("2 unknown keys", out)
        self.assertIn("docs/q.md: f:no_such_key", out)
        self.assertIn("docs/q.md: gen:nope", out)
        code, out, _ = self.gen()
        self.assertEqual(code, 1)
        self.assertIn("<!--f:no_such_key-->7<!--/f-->", read(self.root, "docs/q.md"))
        self.assertIn("old", read(self.root, "docs/q.md"))

    def test_bad_snapshots_are_errors(self):
        write(self.root, "docs/data/gaps.tsv", GAPS + "G3\tthird\te1\tnope\n")
        code, _, err = self.gen("--check")
        self.assertEqual(code, 3)
        self.assertIn("G3 names nope of e1", err)
        write(self.root, "docs/data/gaps.tsv", GAPS)
        write(self.root, "docs/data/references.tsv", REFERENCES + "f\tG1\tc4\te1\ta\tmaybe\n")
        self.assertEqual(self.gen("--check")[0], 3)
        write(self.root, "docs/data/references.tsv", "family\tcase\n")
        self.assertIn("the header must be", self.gen("--check")[2])
        write(self.root, "docs/data/references.tsv", REFERENCES)
        lock = json.loads(json.dumps(LOCK))
        del lock["corpora"]["L3"]
        write(self.root, "docs/data/lock.json", json.dumps(lock))
        self.assertIn("no corpus L3", self.gen("--check")[2])
        write(self.root, "docs/data/lock.json", json.dumps(LOCK))
        write(self.root, "crates/k/tests/fixtures/bad.hex", "41zz\n")
        self.assertIn("not hex text", self.gen("--check")[2])


class Facts(Synthetic):
    def setUp(self):
        super().setUp()
        self.d = dg.Data(self.root)

    def value(self, key):
        return self.d.facts[key]["value"]

    def test_controls_gaps_and_unproven(self):
        self.assertEqual([self.value(k) for k in ("controls_total", "controls_params", "controls_layers",
                                                  "effects_total")], [7, 6, 1, 2])
        self.assertEqual(self.value("gaps_open"), 2)
        self.assertEqual(self.value("gap_controls"), 5)  # (e1, a), (e2, a), three points of e2
        self.assertEqual(self.d.facts["gap_controls_pct"]["text"], "71.4")
        # c1 passes and sets e1 a and e1 lay; c2 fails on e2 a; c3 has no reference
        self.assertEqual(sorted(self.d.graded_pairs()), [("e1", "a"), ("e1", "lay")])
        self.assertEqual(self.value("graded_controls"), 2)
        self.assertEqual(self.value("unproven"), 5)
        self.assertEqual([self.value(k) for k in ("family_cases", "families_total", "graded_cases", "graded_failed",
                                                  "graded_missing")], [3, 2, 1, 1, 1])
        self.assertEqual(self.d.facts["graded_pct"]["text"], "33.3")

    def test_a_failing_case_takes_a_control_back(self):
        write(self.root, "docs/data/references.tsv", REFERENCES + "fam-one\tG1\tc5\te1\tlay\tfail\n")
        d = dg.Data(self.root)
        self.assertEqual(sorted(d.graded_pairs()), [("e1", "a")])
        self.assertEqual(d.facts["unproven"]["value"], 6)

    def test_lock_figures(self):
        self.assertEqual(self.value("lock_ref"), "abcdef0")
        self.assertEqual([self.value(k) for k in ("lock_total", "lock_identical", "lock_differ", "lock_refused",
                                                  "lock_refused_render")], [1015, 1014, 1, 5, 4])
        self.assertEqual(self.d.facts["lock_total"]["text"], "1,015")
        self.assertEqual(self.d.facts["lock_pct"]["text"], "99.9")
        self.assertEqual(self.value("lock_l2_refused_render"), 0)

    def test_scans(self):
        kinds = {x["path"]: (x["magic"], x["kind"], x["cases"], x["bytes"]) for x in self.d.fixtures}
        self.assertEqual(kinds, {
            "crates/k/tests/fixtures/a.hex": ("ABC1", "model", 3, 12),
            "crates/k/tests/fixtures/d.hex": ("DIF1", "differential", None, 9),
            "plugins/p/tests/fixtures/o.hex": ("ZZZZ", "other", None, 9),
        })
        self.assertEqual([self.value(k) for k in ("fixtures_total", "model_fixtures", "model_cases", "model_bytes",
                                                  "differential_fixtures")], [3, 1, 3, 12, 1])
        self.assertEqual(dict(self.d.rust_tests), {"kay": (2, 1), "pee": (1, 0)})
        self.assertEqual(self.value("tests_rust_kay"), 2)
        self.assertEqual(self.value("tests_rust"), 3)
        self.assertEqual(self.value("tests_rust_ignored"), 1)
        self.assertEqual(self.value("tests_py_x"), 2)
        self.assertEqual(self.value("tests_python"), 2)

    def test_scans_skip_build_folders(self):
        write(self.root, "crates/k/target/x.rs", "#[test]\n")
        write(self.root, "crates/k/target/x.hex", "41424331ff000000\n")
        d = dg.Data(self.root)
        self.assertEqual(d.facts["tests_rust"]["value"], 3)
        self.assertEqual(d.facts["fixtures_total"]["value"], 3)


class Rendering(Synthetic):
    def test_id_list(self):
        self.assertEqual(render.id_list(["pt1_use", "pt2_use", "pt3_use"]), "`pt1_use` to `pt3_use`")
        self.assertEqual(render.id_list(["fg2_matte", "fg3_matte"]), "`fg2_matte`, `fg3_matte`")
        self.assertEqual(render.id_list(["pt1_use", "pt2_use", "pt4_use", "pt5_use", "pt6_use", "x"]),
                         "`pt1_use`, `pt2_use`, `pt4_use` to `pt6_use`, `x`")
        self.assertEqual(render.id_list(["fg1_track", "fg2_matte", "fg3_track"]),
                         "`fg1_track`, `fg2_matte`, `fg3_track`")
        self.assertEqual(render.id_list([]), "-")
        self.assertEqual(render.code("a|b"), "`a\\|b`")
        self.assertEqual(render.code("a`b"), "`` a`b ``")

    def test_tables(self):
        self.assertEqual(render.table(["A", "B"], [[1, "x"]]), ["| A | B |", "|---|---|", "| 1 | x |"])
        gen = render.Generator(dg.Data(self.root))
        self.assertEqual(gen.blocks["families"][2:], ["| `fam-one` | G1 | 2 | 1 | 1 | 0 |",
                                                      "| `fam-two` | G2 | 1 | 0 | 0 | 1 |"])
        self.assertEqual(gen.blocks["lock"][-1], "| all | 1,015 | 1,014 | 1 | 5 | 4 |")
        self.assertIn("| `crates/k/tests/fixtures/a.hex` | `ABC1` | model | 3 | 12 |", gen.blocks["fixtures"])
        self.assertIn("| `kay` | `#[test]` | 2 (1 ignored) |", gen.blocks["tests"])
        self.assertIn("| `tools/test_x.py` | `def test_` | 2 |", gen.blocks["tests"])

    def test_a_case_counts_for_every_gap_its_family_settles(self):
        write(self.root, "docs/data/references.tsv", REFERENCES + "fam-both\tG1,G2\tc4\te2\tpt1_use\tmissing\n")
        gen = render.Generator(dg.Data(self.root))
        graded = {r.split(" | ")[0].lstrip("| "): r.rsplit(" | ", 1)[1].rstrip(" |") for r in gen.blocks["gaps"][2:]}
        self.assertEqual(graded, {"G1": "1 of 3", "G2": "0 of 2"})

    def test_empty_register_and_families(self):
        write(self.root, "docs/data/gaps.tsv", dg.tsv_text(dg.GAPS_HEADER, []))
        write(self.root, "docs/data/references.tsv", dg.tsv_text(dg.REFERENCES_HEADER, []))
        gen = render.Generator(dg.Data(self.root))
        self.assertEqual(gen.blocks["gaps"][2:], ["| none | - | - | - | - |"])
        self.assertEqual(gen.blocks["families"][2:], ["| none yet | - | 0 | 0 | 0 | 0 |"])

    def test_fill_keeps_crlf_and_is_idempotent(self):
        gen = render.Generator(dg.Data(self.root))
        src = "x\r\n<!--gen:lock-->\r\nold\r\n<!--/gen:lock-->\r\n<!--f:gaps_open-->9<!--/f-->\r\n"
        text, unknown = gen.fill(src)
        self.assertEqual(unknown, [])
        self.assertNotIn("old", text)
        self.assertIn("| Corpus |", text)
        self.assertNotIn("\n", text.replace("\r\n", ""))
        self.assertIn("<!--f:gaps_open-->2<!--/f-->", text)
        self.assertEqual(gen.fill(text)[0], text)

    def test_a_block_needs_its_own_closing(self):
        gen = render.Generator(dg.Data(self.root))
        src = "<!--gen:lock-->\nold\n<!--/gen:gaps-->\n"
        self.assertEqual(gen.fill(src)[0], src)

    def test_spaced_marker_syntax_is_no_marker(self):
        gen = render.Generator(dg.Data(self.root))
        src = "a `<!--f:` KEY `-->` b `<!--gen:` KEY `-->`\n"
        self.assertEqual(gen.fill(src), (src, []))


LISTING = [
    {"id": "e1", "hidden": False, "params": [{"id": "a", "kind": "float"}], "layers": [{"id": "lay"}],
     "gaps": [{"id": "G10", "title": "ten", "controls": ["a"]}, {"id": "G1", "title": "one", "controls": ["a"]}]},
    {"id": "e2", "hidden": False, "params": [{"id": "a", "kind": "float"}, {"id": "b", "kind": "popup"}],
     "layers": [], "gaps": [{"id": "G1b", "title": "one b", "controls": ["b"]},
                            {"id": "G1", "title": "one", "controls": ["a"]}]},
    {"id": "selftest-gap", "hidden": True, "params": [{"id": "gap", "kind": "checkbox"}], "layers": [],
     "gaps": [{"id": "T1", "title": "test", "controls": ["gap"]}]},
]


class Refresh(unittest.TestCase):
    def test_controls_from_listing(self):
        rows = dg.controls_from_listing(LISTING)
        self.assertEqual([tuple(r.values()) for r in rows], [("e1", "a", "float"), ("e1", "lay", "layer"),
                                                             ("e2", "a", "float"), ("e2", "b", "popup")])

    def test_gaps_from_listing(self):
        rows = dg.gaps_from_listing(LISTING)
        self.assertEqual([tuple(r.values()) for r in rows], [("G1", "one", "e1,e2", "a"), ("G1b", "one b", "e2", "b"),
                                                             ("G10", "ten", "e1", "a")])
        bad = json.loads(json.dumps(LISTING))
        bad[1]["gaps"][1]["controls"] = ["b"]
        with self.assertRaises(dg.DataError):
            dg.gaps_from_listing(bad)

    def test_the_gap_of_a_family_case_comes_from_the_families_table(self):
        gaps = dg.family_gaps(dg.ROOT)
        self.assertEqual(gaps["gap-shrink"], ["G1"])
        self.assertTrue(all(ids for ids in gaps.values()), gaps)
        cases = {"cases": [{"id": "a", "family": "gap-shrink", "effect": "create-vectors", "params": ["x=1"]},
                           {"id": "b", "family": "gap-shrink", "gap": "G9", "effect": "retime"},
                           {"id": "c", "family": "nowhere", "effect": "retime"}]}
        rows = dg.references_from_cases(cases, None, gaps)
        self.assertEqual([r["gap"] for r in rows], ["G1", "G9", "-"])
        self.assertEqual(dg.family_gaps(os.path.join(dg.ROOT, "no-such-folder")), {})

    def test_references_from_cases(self):
        cases = {"cases": [
            {"id": "z", "effect": "e1", "family": "fam", "gap": "G1", "params": ["a=1", "b=2"],
             "layers": {"lay": "x.raw"}},
            {"id": "y", "effect": "e2", "family": "fam", "params": []},
            {"id": "plain", "effect": "e1", "params": ["a=1"]},
        ]}
        rows = dg.references_from_cases(cases, {"z": "pass"})
        self.assertEqual([tuple(r.values()) for r in rows], [("fam", "-", "y", "e2", "-", "missing"),
                                                             ("fam", "G1", "z", "e1", "a,b,lay", "pass")])
        self.assertEqual(dg.references_from_cases(cases, {"z": "FAIL", "y": "no reference"})[1]["result"], "fail")
        with self.assertRaises(dg.DataError):
            dg.references_from_cases(cases, {"z": "odd"})

    def test_program_and_sibling(self):
        self.assertEqual(gen_docs.program(os.path.join("d", "vfx.py")),
                         [sys.executable, os.path.abspath(os.path.join("d", "vfx.py"))])
        self.assertEqual(gen_docs.program("vfx"), ["vfx"])
        here = os.path.abspath("d")
        self.assertEqual(gen_docs.sibling(os.path.join("d", "vfx.py"), "fx-parity"), os.path.join(here, "fx-parity.py"))
        self.assertEqual(gen_docs.sibling(os.path.join("d", "vfx.exe"), "fx-parity"), os.path.join(here, "fx-parity.exe"))
        self.assertEqual(gen_docs.sibling(os.path.join("d", "vfx"), "fx-parity"),
                         os.path.join(here, "fx-parity" + gen_docs.EXE))

    def test_tsv_round_trip(self):
        text = dg.tsv_text(dg.CONTROLS_HEADER, dg.controls_from_listing(LISTING))
        self.assertTrue(text.startswith("effect\tid\tkind\ne1\ta\tfloat\n"))
        with self.assertRaises(dg.DataError):
            dg.tsv_text(["a"], [{"a": "x\ty"}])


FAKE_VFX = r'''
import json, sys
if sys.argv[1:] == ["--help"]:
    print("usage: vfx apply --width W --height H --depth D [--check]")
    sys.exit(0)
assert sys.argv[1:] == ["list", "--json", "--all"], sys.argv
print(%r)
''' % json.dumps(LISTING)

FAKE_PARITY = r'''
import json, sys
if sys.argv[1:] == ["--help"]:
    print("fx-parity gen --out DIR --families IDS")
    sys.exit(0)
args = sys.argv[1:]
assert args[0] == "run" and "--json" in args and "--strict" not in args, args
print(json.dumps({"summary": {}, "cases": [{"id": "z", "status": "pass"}, {"id": "y", "status": "FAIL"}]}))
sys.exit(1)
'''


class RefreshMain(Synthetic):
    def test_refresh_rewrites_the_snapshots_and_fills(self):
        tools = os.path.join(self.root, "bin")
        write(self.root, "bin/vfx.py", FAKE_VFX)
        write(self.root, "bin/fx-parity.py", FAKE_PARITY)
        write(self.root, "pack/cases.json", json.dumps({"cases": [
            {"id": "z", "effect": "e1", "family": "fam", "gap": "G1", "params": ["a=1"]},
            {"id": "y", "effect": "e2", "family": "fam", "gap": "G1", "params": ["a=2"]},
            {"id": "w", "effect": "e2", "family": "fam2", "params": []}]}))
        code, out, err = self.gen("--refresh-data", "--vfx", os.path.join(tools, "vfx.py"),
                                  "--testset", os.path.join(self.root, "pack"), "--refs", os.path.join(self.root, "pack"))
        self.assertEqual(code, 0, out + err)
        self.assertIn("gen_docs: docs/data/controls.tsv rebuilt", out)
        self.assertEqual(read(self.root, "docs/data/controls.tsv"), "effect\tid\tkind\ne1\ta\tfloat\ne1\tlay\tlayer\n"
                                                                     "e2\ta\tfloat\ne2\tb\tpopup\n")
        self.assertEqual(read(self.root, "docs/data/gaps.tsv").splitlines()[1], "G1\tone\te1,e2\ta")
        self.assertEqual(read(self.root, "docs/data/references.tsv").splitlines()[1:],
                         ["fam\tG1\ty\te2\ta\tfail", "fam\tG1\tz\te1\ta\tpass", "fam2\t-\tw\te2\t-\tmissing"])
        self.assertIn("<!--f:controls_total-->4<!--/f-->", read(self.root, "README.md"))
        self.assertEqual(self.gen("--check")[0], 0)

    def test_refresh_needs_a_vfx(self):
        env = os.environ.pop("VFX_PATH", None)
        self.addCleanup(lambda: env is not None and os.environ.__setitem__("VFX_PATH", env))
        code, _, err = self.gen("--refresh-data")
        self.assertEqual(code, 3)
        self.assertIn("needs a built vfx", err)
        code, _, err = self.gen("--refresh-data", "--vfx", os.path.join(self.root, "nope"))
        self.assertIn("vfx not found", err)
        write(self.root, "bin/vfx.py", FAKE_VFX)
        code, _, err = self.gen("--refresh-data", "--vfx", os.path.join(self.root, "bin", "vfx.py"),
                                "--refs", self.root)
        self.assertIn("--refs needs --testset", err)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as e:
                gen_docs.main(["--check", "--refresh-data"])
        self.assertEqual(e.exception.code, 2)


class CheckDocs(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="checkdocs-test-")
        self.addCleanup(shutil.rmtree, self.root, True)

    def run_checks(self, *names, vfx=None):
        return check_docs.run(self.root, list(names), vfx)[0]

    def test_links_and_anchors(self):
        write(self.root, "README.md", "# R\n\n## Some Heading, here\n\n## `vfx` flags\n\n## Again\n\n## Again\n\n"
                                      "[ok](docs/b.md#target-one) [bad](docs/b.md#nope) [gone](docs/c.md) "
                                      "[self](#some-heading-here) [code](#vfx-flags) [dup](#again-1) "
                                      "[web](https://example.com) [mail](mailto:x@example.com) `[no](link.md)`\n"
                                      "```\n[fenced](missing.md)\n```\n")
        write(self.root, "docs/b.md", "# B\n\n## Target one\n\n[up](../README.md#again) [file](../LICENSE) "
                                      "[up2](../README.md#again-2)\n")
        write(self.root, "LICENSE", "MIT\n")
        found = sorted((f[1], f[2], f[3]) for f in self.run_checks("links")["links"])
        self.assertEqual(found, [("README.md", 11, "missing anchor docs/b.md#nope"),
                                 ("README.md", 11, "missing target docs/c.md"),
                                 ("docs/b.md", 5, "missing anchor ../README.md#again-2")])

    def test_paths(self):
        write(self.root, "tools/x.py", "")
        write(self.root, "README.md", "# R\n\n`tools/x.py` `tools/x.py:12` `tools/missing.py:3-4` `crates/*/src` "
                                      "`docs/<id>.md` `.../fixtures/a.hex` `target/release/vfx` `tools/a b.py` "
                                      "`Cargo.toml`\nsapphire keeps `crates/neutral/src/x.rs`\n")
        self.assertEqual([(f[2], f[3]) for f in self.run_checks("paths")["paths"]],
                         [(3, "missing path tools/missing.py"), (3, "missing path Cargo.toml")])

    def test_ascii_and_control_characters(self):
        write(self.root, "README.md", "# R" + NL + "tab" + chr(9) + "ok" + chr(13) + NL + "bad" + chr(1) + NL
              + "caf" + chr(0xe9) + NL)
        write(self.root, "tools/t.py", "x = 1" + chr(127) + NL)
        found = [(f[1], f[2], f[3]) for f in self.run_checks("ascii")["ascii"]]
        self.assertEqual(found, [("README.md", 3, "control character 0x01"), ("README.md", 4, "non-ASCII character"),
                                 ("tools/t.py", 1, "control character 0x7f")])

    def test_dashes(self):
        write(self.root, "README.md", "a - b" + NL + "c " + chr(0x2014) + " d" + NL)
        write(self.root, "src/x.rs", "// x " + chr(0x2212) + " y" + chr(0x2013) + NL)
        write(self.root, "src/y.bin", chr(0x2014))
        found = [(f[1], f[2], f[3]) for f in self.run_checks("dashes")["dashes"]]
        self.assertEqual(found, [("README.md", 2, "dash U+2014"), ("src/x.rs", 1, "dash U+2013, U+2212")])

    def test_flags_against_help(self):
        write(self.root, "bin/vfx.py", FAKE_VFX)
        write(self.root, "bin/fx-parity.py", FAKE_PARITY)
        write(self.root, "README.md", "# R\n\n| Flag | Meaning |\n|---|---|\n| `--width W`, `--height H` | size |\n"
                                      "| `--nope` | gone |\n| `-o PATH`, `--depth 8\\|16` | x |\n\n"
                                      "| Subcommand | Key flags (default) |\n|---|---|\n| `gen` | `--families`, "
                                      "`--wrong` |\n\n<!--help:fx-parity-->\n| Flag | Meaning |\n|---|---|\n"
                                      "| `--width` | not in this help |\n")
        found, skipped = check_docs.run(self.root, ["flags"], os.path.join(self.root, "bin", "vfx.py"))
        self.assertEqual(skipped, {})
        self.assertEqual([(f[2], f[3]) for f in found["flags"]], [(6, "flag --nope not in --help"),
                                                                  (11, "flag --wrong not in --help"),
                                                                  (16, "flag --width not in --help")])

    def test_flags_skip_without_vfx(self):
        write(self.root, "README.md", "| Flag |\n|---|\n| `--nope` |\n")
        found, skipped = check_docs.run(self.root, ["flags"], None)
        self.assertEqual(found["flags"], [])
        self.assertIn("VFX_PATH", skipped["flags"])
        found, _ = check_docs.run(self.root, ["flags"], os.path.join(self.root, "missing-vfx"))
        self.assertEqual([f[3] for f in found["flags"]], ["vfx not found at " + os.path.join(self.root, "missing-vfx")])

    def test_main_counts_json_and_list(self):
        write(self.root, "README.md", "# R\n\n[a](x.md) [b](y.md) [c](z.md) [d](w.md) [e](v.md) [f](u.md)\n")
        code, out, _ = quiet(check_docs.main, ["--only", "links", "--root", self.root])
        self.assertEqual(code, 1)
        self.assertIn("check_docs: 6 findings; links 6", out)
        self.assertEqual(out.count("  links README.md:3"), 5)
        code, out, _ = quiet(check_docs.main, ["--only", "links", "--list", "links", "--root", self.root])
        self.assertEqual(out.count("  links README.md:3"), 6)
        saved = os.environ.pop("VFX_PATH", None)
        self.addCleanup(lambda: saved is None or os.environ.__setitem__("VFX_PATH", saved))
        code, out, _ = quiet(check_docs.main, ["--only", "links", "--only", "flags", "--json", "--root", self.root])
        doc = json.loads(out)
        self.assertEqual(sorted(doc), ["findings", "skipped"])
        self.assertEqual(doc["findings"]["links"][0], ["README.md", 3, "missing target x.md"])
        self.assertIn("flags", doc["skipped"])
        write(self.root, "README.md", "# R\n")
        self.assertEqual(quiet(check_docs.main, ["--only", "links", "--root", self.root])[0], 0)

    def test_regen_reports_drift_and_unknown_keys(self):
        Synthetic.setUp(self)
        self.assertEqual(quiet(gen_docs.main, ["--root", self.root])[0], 0)
        self.assertEqual(self.run_checks("regen")["regen"], [])
        write(self.root, "docs/p.md", read(self.root, "docs/p.md") + "<!--f:zzz-->1<!--/f-->\n")
        write(self.root, "docs/data/controls.tsv", CONTROLS + "e1\tq\tfloat\n")
        found = sorted((f[1], f[3]) for f in self.run_checks("regen")["regen"])
        self.assertIn(("docs/p.md", "unknown marker key f:zzz"), found)
        self.assertIn(("README.md", "differs from a fresh generation (run python -I tools/gen_docs.py)"), found)
        write(self.root, "docs/data/lock.json", "{")
        self.assertIn("cannot generate", self.run_checks("regen")["regen"][0][3])


class ThisTree(unittest.TestCase):
    """The committed snapshots and pages of this repository."""

    def test_no_drift_and_no_unknown_key(self):
        _, changed, unknown = gen_docs.drift(dg.ROOT)
        self.assertEqual((changed, unknown), ([], []))

    def test_deterministic(self):
        a, _, _ = render.build(dg.ROOT)
        b, _, _ = render.build(dg.ROOT)
        self.assertEqual(a, b)

    def test_register_covers_the_effects(self):
        d = dg.Data(dg.ROOT)
        effects = {r["effect"] for r in d.controls}
        self.assertEqual(effects, {"retime", "create-vectors", "retime-fgbg", "vectors-in"})
        for g in d.gaps:
            self.assertRegex(g["id"], r"^G\d+[a-z]?$")
            self.assertTrue(set(dg.split_list(g["effects"])) <= effects)
        self.assertEqual(d.facts["lock_differ"]["value"], 0)

    def test_parity_page_quotes_every_standard(self):
        text = read(dg.ROOT, "docs/parity.md")
        for key in ("graded_cases", "family_cases", "gaps_open", "gap_controls", "model_fixtures", "model_cases",
                    "lock_identical", "lock_total", "lock_ref", "unproven", "controls_total"):
            self.assertIn("<!--f:" + key + "-->", text)
        for block in ("gaps", "families", "fixtures", "lock", "tests"):
            self.assertIn("<!--gen:" + block + "-->", text)

    def test_documentation_checks_pass(self):
        results, _ = check_docs.run(dg.ROOT, ["links", "paths", "ascii", "dashes"])
        self.assertEqual({k: v[:3] for k, v in results.items() if v}, {})


if __name__ == "__main__":
    unittest.main()
