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

<!--f:modules_total-->0<!--/f--> modules, <!--f:modules_verified-->0<!--/f--> verified, <!--f:modules_reported-->0<!--/f--> reported.

<!--gen:modules-->
<!--/gen:modules-->
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

def verified_summary(mid):
    """A summary.json that meets the status rule."""
    return {"module": mid, "plugin": "Twixtor_64.aex", "plugin_sha256": "ab", "effect": "retime",
            "status": "verified", "blockers": [],
            "cases": {"in_contract": 150, "compared": 150, "identical": 150, "differing": 0, "out_of_contract": 4,
                      "logic_parity": 3},
            "values": {"bytes_compared": 1000, "floats_compared": 2000, "differing_values": 0, "max_ulp": 0},
            "call_records": {"compared": 12, "matching": 12},
            "mutants": {"total": 8, "killed": 8, "survivors": []},
            "stand_ins": ["keyframe values from a fixed table"], "fixes": [], "commands": ["run it"],
            "measured_at": "2026-11-02"}


def write_summary(root, mid, doc):
    write(root, "parity/{}/results/summary.json".format(mid), json.dumps(doc))


def write_verify(root, mid, doc):
    write(root, "parity/{}/results/verify.json".format(mid), json.dumps(doc))



class Synthetic(unittest.TestCase):
    """A small repository: snapshots, one crate with a fixture folder, one Python test file, two pages."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="docgen-test-")
        self.addCleanup(shutil.rmtree, self.root, True)
        write(self.root, "docs/data/controls.tsv", CONTROLS)
        write(self.root, "docs/data/gaps.tsv", GAPS)
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
        self.assertIn("<!--f:modules_total-->6<!--/f--> modules, <!--f:modules_verified-->0<!--/f--> verified, "
                      "<!--f:modules_reported-->0<!--/f--> reported", text)
        self.assertIn("| `retime-time` | `Twixtor_64.aex` | Not started | - | - | - | - | - | - | - | - | - |", text)
        self.assertIn("| Total | - | 0 of 6 match, 0 of 6 verified | 0 | 0 | 0 / 0 | 0 / 0 | - | 0 of 0 | 0 of 0 | 0 of 6 "
                      "| - |", text)
        self.assertIn("<!--f:lock_identical-->1,014<!--/f--> of <!--f:lock_total-->1,015<!--/f-->",
                      read(self.root, "docs/p.md"))
        facts = json.loads(read(self.root, "docs/data/facts.json"))
        self.assertEqual(facts["generator"], "tools/gen_docs.py")
        self.assertTrue(all(f["source"] for f in facts["facts"].values()))
        self.assertEqual(sorted(facts["blocks"]), ["fixtures", "harness", "lock", "modules", "notes", "tests"])
        code, out, _ = self.gen("--check")
        self.assertEqual(code, 0, out)
        self.assertIn("0 drifted, 0 unknown keys", out)
        again, _, _ = self.gen()
        self.assertEqual(read(self.root, "README.md"), text)

    def test_check_finds_a_mutated_marker(self):
        self.gen()
        text = read(self.root, "README.md")
        write(self.root, "README.md", text.replace("<!--f:modules_total-->6<!--/f-->", "<!--f:modules_total-->7<!--/f-->"))
        code, out, _ = self.gen("--check")
        self.assertEqual(code, 1)
        self.assertIn("1 drifted", out)
        self.assertIn("  README.md", out)
        write(self.root, "README.md", text.replace("| Not started |", "| Verified |", 1))
        self.assertEqual(self.gen("--check")[0], 1)
        write(self.root, "README.md", text)
        self.assertEqual(self.gen("--check")[0], 0)

    def test_check_finds_changed_data(self):
        self.gen()
        lock = json.loads(json.dumps(LOCK))
        lock["corpora"]["L1"]["identical"] = 9
        write(self.root, "docs/data/lock.json", json.dumps(lock))
        code, out, _ = self.gen("--check")
        self.assertEqual(code, 1)
        self.assertIn("docs/p.md", out)
        self.assertIn("docs/data/facts.json", out)
        write(self.root, "docs/data/lock.json", json.dumps(LOCK))
        write(self.root, "plugins/p/tests/t.rs", "#[test]\nfn t() {}\n#[test]\nfn u() {}\n")
        self.assertEqual(self.gen("--check")[0], 1)

    def test_a_result_file_moves_the_table(self):
        self.gen()
        write_summary(self.root, "retime-time", verified_summary("retime-time"))
        code, out, _ = self.gen("--check")
        self.assertEqual(code, 1)
        self.assertIn("README.md", out)
        self.assertEqual(self.gen()[0], 0)
        text = read(self.root, "README.md")
        self.assertIn("<!--f:modules_verified-->1<!--/f--> verified, <!--f:modules_reported-->1<!--/f--> reported", text)
        self.assertEqual(self.gen("--check")[0], 0)
        write(self.root, "parity/retime-time/results/summary.json", "{")
        code, _, err = self.gen("--check")
        self.assertEqual(code, 3)
        self.assertIn("parity/retime-time/results/summary.json", err)

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
        write(self.root, "docs/data/gaps.tsv", "id\ttitle\n")
        self.assertIn("the header must be", self.gen("--check")[2])
        write(self.root, "docs/data/gaps.tsv", GAPS)
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

    def test_the_module_facts_and_no_graded_figures(self):
        self.assertEqual([self.value(k) for k in ("modules_total", "modules_reported", "modules_verified",
                                                  "modules_rerun")], [6, 0, 0, 0])
        for gone_key in ("controls_total", "gaps_open", "graded_cases", "unproven", "family_cases"):
            self.assertNotIn(gone_key, self.d.facts)
        write_summary(self.root, "retime-time", verified_summary("retime-time"))
        write_summary(self.root, "retime-flow", dict(verified_summary("retime-flow"), status="blocked"))
        write_verify(self.root, "retime-time", {"module": "retime-time", "reran_at": "2026-11-03", "matches": True,
                                                "differences": []})
        d = dg.Data(self.root)
        self.assertEqual([d.facts[k]["value"] for k in ("modules_reported", "modules_verified", "modules_rerun")],
                         [2, 1, 1])

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
    def test_code(self):
        self.assertEqual(render.code("a|b"), "`a\\|b`")
        self.assertEqual(render.code("a`b"), "`` a`b ``")

    def test_tables(self):
        self.assertEqual(render.table(["A", "B"], [[1, "x"]]), ["| A | B |", "|---|---|", "| 1 | x |"])
        gen = render.Generator(dg.Data(self.root))
        self.assertEqual(sorted(gen.blocks), ["fixtures", "harness", "lock", "modules", "notes", "tests"])
        self.assertEqual(gen.blocks["lock"][-1], "| all | 1,015 | 1,014 | 1 | 5 | 4 |")
        self.assertIn("| `crates/k/tests/fixtures/a.hex` | `ABC1` | model | 3 | 12 |", gen.blocks["fixtures"])
        self.assertIn("| `kay` | `#[test]` | 2 (1 ignored) |", gen.blocks["tests"])
        self.assertIn("| `tools/test_x.py` | `def test_` | 2 |", gen.blocks["tests"])
        self.assertEqual(gen.blocks["notes"][2:], ["| - | - | None reported yet. |"])
        self.assertEqual(gen.blocks["harness"][2:], ["| - | No harness summary yet. |"])

    def test_the_header_names_the_twelve_columns(self):
        gen = render.Generator(dg.Data(self.root))
        self.assertEqual(gen.blocks["modules"][0],
                         "| Module | Plugin | Status | Cases compared (in contract) | Out of contract (candidates) | "
                         "Values compared (bytes / floats) | Differing | Max ULP | Call records matching | "
                         "Mutants killed | Re-run verified | Source |")
        self.assertEqual(len(gen.blocks["modules"]), 2 + 6 + 1)
        order = [r.split(" | ")[0].lstrip("| ") for r in gen.blocks["modules"][2:8]]
        self.assertEqual(order, ["`retime-time`", "`retime-flow`", "`create-vectors`", "`vectors-in`", "`fgbg-core`",
                                 "`fgbg-layers`"])

    def test_fill_keeps_crlf_and_is_idempotent(self):
        gen = render.Generator(dg.Data(self.root))
        src = "x\r\n<!--gen:lock-->\r\nold\r\n<!--/gen:lock-->\r\n<!--f:modules_total-->9<!--/f-->\r\n"
        text, unknown = gen.fill(src)
        self.assertEqual(unknown, [])
        self.assertNotIn("old", text)
        self.assertIn("| Corpus |", text)
        self.assertNotIn("\n", text.replace("\r\n", ""))
        self.assertIn("<!--f:modules_total-->6<!--/f-->", text)
        self.assertEqual(gen.fill(text)[0], text)

    def test_a_block_needs_its_own_closing(self):
        gen = render.Generator(dg.Data(self.root))
        src = "<!--gen:lock-->\nold\n<!--/gen:modules-->\n"
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
import sys
if sys.argv[1:] == ["--help"]:
    print("fx-parity gen --out DIR --families IDS")
    sys.exit(0)
sys.exit(2)
'''


class RefreshMain(Synthetic):
    def test_refresh_rewrites_the_snapshots_and_fills(self):
        tools = os.path.join(self.root, "bin")
        write(self.root, "bin/vfx.py", FAKE_VFX)
        code, out, err = self.gen("--refresh-data", "--vfx", os.path.join(tools, "vfx.py"))
        self.assertEqual(code, 0, out + err)
        self.assertIn("gen_docs: docs/data/controls.tsv rebuilt", out)
        self.assertEqual(read(self.root, "docs/data/controls.tsv"), "effect\tid\tkind\ne1\ta\tfloat\ne1\tlay\tlayer\n"
                                                                     "e2\ta\tfloat\ne2\tb\tpopup\n")
        self.assertEqual(read(self.root, "docs/data/gaps.tsv").splitlines()[1], "G1\tone\te1,e2\ta")
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
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as e:
                gen_docs.main(["--check", "--refresh-data"])
        self.assertEqual(e.exception.code, 2)


class Modules(Synthetic):
    def block(self, name="modules"):
        return render.Generator(dg.Data(self.root)).blocks[name]

    def row(self, mid):
        return [r for r in self.block() if r.startswith("| `{}` |".format(mid))][0]

    def test_a_verified_module_fills_its_row_and_the_totals(self):
        write_summary(self.root, "retime-time", verified_summary("retime-time"))
        self.assertEqual(self.row("retime-time"),
                         "| `retime-time` | `Twixtor_64.aex` | Verified | 150 | 4 | 1,000 / 2,000 | 0 / 0 | 0 | "
                         "12 of 12 | 8 of 8 | - | `parity/retime-time/results/summary.json` |")
        self.assertEqual(self.row("retime-flow").split(" | ")[2], "Not started")
        self.assertEqual(self.block()[-1], "| Total | - | 1 of 6 match, 1 of 6 verified | 150 | 4 | 1,000 / 2,000 | 0 / 0 "
                                           "| 0 | 12 of 12 | 8 of 8 | 0 of 6 | - |")
        notes = self.block("notes")[2:]
        self.assertEqual(notes, ["| `retime-time` | stand-in | keyframe values from a fixed table |",
                                 "| `retime-time` | logic parity | 3 compared cases use a stand-in on both sides |"])

    def test_the_status_rule(self):
        for field, change in (("cases", {"compared": 149, "identical": 149}), ("cases", {"differing": 1}),
                              ("values", {"differing_values": 1}), ("call_records", {"matching": 11}),
                              ("mutants", {"killed": 7}), ("mutants", {"total": 0, "killed": 0}),
                              ("call_records", {"compared": 0, "matching": 0}),
                              ("mutants", {"survivors": ["flip a sign"]})):
            doc = verified_summary("fgbg-core")
            doc[field] = dict(doc[field], **change)
            write_summary(self.root, "fgbg-core", doc)
            m = dg.load_module(self.root, "fgbg-core", "x", "y")
            self.assertTrue(m["unmet"], (field, change))
            self.assertFalse(dg.is_verified(m))
            self.assertEqual(self.row("fgbg-core").split(" | ")[2],
                             "Unconfirmed (file says verified, rule unmet)", (field, change))
        write_summary(self.root, "fgbg-core", verified_summary("fgbg-core"))
        self.assertEqual(dg.load_module(self.root, "fgbg-core", "x", "y")["unmet"], [])
        self.assertEqual(self.row("fgbg-core").split(" | ")[2], "Verified")

    def test_a_module_in_progress_that_matches_on_the_emulator_run(self):
        doc = verified_summary("vectors-in")
        doc.update(status="in_progress", blockers=["mutants survive: m07"])
        doc["cases"] = dict(doc["cases"], in_contract=160)
        doc["mutants"] = {"total": 8, "killed": 7, "survivors": ["m07"]}
        write_summary(self.root, "vectors-in", doc)
        m = dg.load_module(self.root, "vectors-in", "x", "y")
        self.assertTrue(dg.is_match(m))
        self.assertFalse(dg.is_verified(m))
        self.assertEqual(self.row("vectors-in").split(" | ")[2:4], ["Matches", "150 of 160"])
        self.assertTrue(self.block()[-1].startswith("| Total | - | 1 of 6 match, 0 of 6 verified | 150 of 160 |"))
        self.assertEqual(dg.Data(self.root).facts["modules_match"]["value"], 1)
        self.assertEqual(dg.Data(self.root).facts["cases_compared"]["value"], 150)
        for field, change in (("cases", {"differing": 1}), ("values", {"differing_values": 1}),
                              ("cases", {"compared": 149, "identical": 149}), ("call_records", {"matching": 11}),
                              ("call_records", {"compared": 0, "matching": 0})):
            bad = dict(doc, **{field: dict(doc[field], **change)})
            write_summary(self.root, "vectors-in", bad)
            self.assertFalse(dg.is_match(dg.load_module(self.root, "vectors-in", "x", "y")), (field, change))
            self.assertEqual(self.row("vectors-in").split(" | ")[2], "In progress", (field, change))
        write_summary(self.root, "vectors-in", dict(doc, status="blocked"))
        self.assertEqual(self.row("vectors-in").split(" | ")[2], "Blocked")
        self.assertEqual(dg.Data(self.root).facts["modules_match"]["value"], 0)

    def test_other_states_show_as_the_file_states_them(self):
        doc = verified_summary("create-vectors")
        doc.update(status="blocked", blockers=["no mock for call 7"])
        doc["cases"] = {"in_contract": 150, "compared": 40}
        write_summary(self.root, "create-vectors", doc)
        row = self.row("create-vectors").split(" | ")
        self.assertEqual((row[2], row[3]), ("Blocked", "40 of 150"))
        self.assertIn("| `create-vectors` | blocker | no mock for call 7 |", self.block("notes"))
        doc["status"] = "in_progress"
        write_summary(self.root, "create-vectors", doc)
        self.assertEqual(self.row("create-vectors").split(" | ")[2], "In progress")

    def test_a_re_run_is_yes_no_stale_or_a_dash(self):
        write_summary(self.root, "retime-time", verified_summary("retime-time"))
        cases = [({"module": "retime-time", "reran_at": "2026-11-03", "matches": True, "differences": []}, "Yes, 2026-11-03"),
                 ({"module": "retime-time", "reran_at": "2026-11-01", "matches": True, "differences": []}, "Stale, 2026-11-01"),
                 ({"module": "retime-time", "reran_at": "2026-11-03", "matches": False, "differences": ["a", "b"]},
                  "No, 2 differences")]
        for doc, shown in cases:
            write_verify(self.root, "retime-time", doc)
            self.assertEqual(self.row("retime-time").split(" | ")[10], shown)
        write_verify(self.root, "retime-time", cases[0][0])
        self.assertEqual(self.block()[-1].split(" | ")[10], "1 of 6")

    def test_max_ulp_total_is_the_largest(self):
        for mid, ulp in (("retime-time", 0), ("retime-flow", 3), ("vectors-in", None)):
            doc = verified_summary(mid)
            doc["values"]["max_ulp"] = ulp
            write_summary(self.root, mid, doc)
        self.assertEqual(self.row("vectors-in").split(" | ")[7], "-")
        self.assertEqual(self.block()[-1].split(" | ")[7], "3")

    def test_malformed_results_are_errors_not_rows(self):
        path = "parity/retime-time/results/summary.json"
        doc = verified_summary("retime-time")
        for bad in (dict(doc, module="retime-flow"), dict(doc, status="done"), dict(doc, cases=[1]),
                    dict(doc, cases=dict(doc["cases"], compared=True)),
                    dict(doc, cases=dict(doc["cases"], compared=-1)),
                    dict(doc, values=dict(doc["values"], max_ulp=1.5)),
                    dict(doc, blockers="one"), dict(doc, mutants=dict(doc["mutants"], survivors=[1])), [1]):
            write(self.root, path, json.dumps(bad))
            with self.assertRaises(dg.DataError, msg=str(bad)[:60]):
                dg.Data(self.root)
        write(self.root, path, json.dumps(doc))
        write_verify(self.root, "retime-time", {"module": "retime-flow", "matches": True})
        with self.assertRaises(dg.DataError):
            dg.Data(self.root)
        write_verify(self.root, "retime-time", {"module": "retime-time", "matches": "yes"})
        with self.assertRaises(dg.DataError):
            dg.Data(self.root)

    def test_result_text_is_ascii_and_dashless(self):
        doc = verified_summary("retime-time")
        doc["blockers"] = ["a " + chr(0x2014) + " b | c" + chr(0xe9)]
        doc["status"] = "blocked"
        write_summary(self.root, "retime-time", doc)
        notes = self.block("notes")
        self.assertIn("| `retime-time` | blocker | a - b \\| c? |", notes)
        self.assertEqual([l for l in notes if any(ord(c) > 127 for c in l)], [])

    def test_the_harness_summary_is_shown_as_stated(self):
        self.assertEqual(self.block("harness")[2:], ["| - | No harness summary yet. |"])
        write(self.root, "parity/harness/results/summary.json",
              json.dumps({"emulator": "qiling", "ok": True, "count": 3, "none": None, "list": [1, 2],
                          "big": list(range(100))}))
        self.assertEqual(self.block("harness")[2:], [
            "| `emulator` | qiling |", "| `ok` | true |", "| `count` | 3 |", "| `none` | null |",
            "| `list` | [1,2] |", "| `big` | 100 entries, see the file |"])
        write(self.root, "parity/harness/results/summary.json", "[]")
        with self.assertRaises(dg.DataError):
            dg.Data(self.root)


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

    def test_limits_table_must_match_the_register(self):
        write(self.root, "docs/data/gaps.tsv", GAPS)
        write(self.root, "README.md", "# R\n\n## Limits\n\n| Gap | Open point | Rule |\n|---|---|---|\n"
                                      "| G1 | first open point | a rule |\n| G3 | invented | a rule |\n\n"
                                      "## Legal\n\n| G2 | second open point | outside the section |\n")
        found = [(f[2], f[3]) for f in self.run_checks("limits")["limits"]]
        self.assertEqual(found, [(3, "gap G2 (second open point) is not in the Limits table"),
                                 (8, "gap G3 is not in docs/data/gaps.tsv")])
        write(self.root, "README.md", "# R\n\n## Limits\n\n| G1 | first open point | a |\n| G2 | other title | b |\n")
        self.assertEqual([f[3] for f in self.run_checks("limits")["limits"]],
                         ["gap G2 is titled 'other title', the register says 'second open point'"])
        write(self.root, "README.md", "# R\n\n## Limits\n\n| G1 | first open point | a |\n| G2 | second open point | b |\n")
        self.assertEqual(self.run_checks("limits")["limits"], [])
        write(self.root, "README.md", "# R\n")
        self.assertEqual([f[3] for f in self.run_checks("limits")["limits"]], ["no `## Limits` heading"])

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
        write_summary(self.root, "vectors-in", verified_summary("vectors-in"))
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
        self.assertEqual([m["id"] for m in d.modules], [m[0] for m in dg.MODULES])

    def test_limits_table_matches_the_register(self):
        self.assertEqual(check_docs.run(dg.ROOT, ["limits"])[0]["limits"], [])

    def test_parity_page_quotes_the_bar_and_not_the_old_figures(self):
        text = read(dg.ROOT, "docs/parity.md")
        for key in ("modules_total", "modules_verified", "model_fixtures", "model_cases", "lock_identical",
                    "lock_total", "lock_ref"):
            self.assertIn("<!--f:" + key + "-->", text)
        for block in ("modules", "notes", "harness", "fixtures", "lock", "tests"):
            self.assertIn("<!--gen:" + block + "-->", text)
        readme = read(dg.ROOT, "README.md")
        self.assertIn("<!--gen:modules-->", readme)
        self.assertIn("\n## Limits\n", readme)
        for old in ("Parity gaps", "Unproven", "of 117", "within one unit in the last place", "within one ULP of"):
            self.assertNotIn(old, readme)
            self.assertNotIn(old, text)
        self.assertIn("bit for bit", readme)
        self.assertIn("bit for bit", text)

    def test_documentation_checks_pass(self):
        results, _ = check_docs.run(dg.ROOT, ["links", "paths", "ascii", "dashes"])
        self.assertEqual({k: v[:3] for k, v in results.items() if v}, {})


if __name__ == "__main__":
    unittest.main()
