"""Fill the figure and table markers of the Markdown files, and write the machine readable fact index."""

import json
import os
import re
from collections import OrderedDict

from .data import (DATA, HARNESS_SUMMARY, LOCK_SETS, ROOT, Data, clean, fmt_int, is_match, is_verified, read_text,
                   rerun_state, verdict)

F_RE = re.compile(r"<!--f:([A-Za-z0-9_.-]+)-->(.*?)<!--/f-->")
G_RE = re.compile(r"(<!--gen:([A-Za-z0-9_.-]+)-->\r?\n)(.*?)(<!--/gen:\2-->)", re.S)
FACTS = DATA + "/facts.json"


def code(s):
    """An inline code span that keeps the text verbatim inside a table cell."""
    s = s.replace("|", "\\|")
    if "`" in s:
        return "`` " + s + " ``"
    return "`" + s + "`"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def md_files(root):
    """README.md and every Markdown file under docs/ except the snapshots, sorted."""
    out = ["README.md"] if os.path.isfile(os.path.join(root, "README.md")) else []
    docs = os.path.join(root, "docs")
    for dirpath, dirs, files in os.walk(docs):
        dirs[:] = sorted(d for d in dirs if d != "data")
        for f in sorted(files):
            if f.endswith(".md"):
                out.append(os.path.relpath(os.path.join(dirpath, f), root).replace(os.sep, "/"))
    return sorted(out)


class Generator:
    """The table blocks of one Data, and the marker filler."""

    def __init__(self, data):
        self.d = data
        self.blocks = OrderedDict()
        self.sources = OrderedDict()
        self._modules()
        self._notes()
        self._harness()
        self._fixtures()
        self._lock()
        self._tests()

    def add(self, name, source, lines):
        self.blocks[name] = lines
        self.sources[name] = source

    def _modules(self):
        """One row per module in module order, a totals row computed from the result files. A module with
        no summary.json is Not started and every figure is a dash."""
        mods = self.d.modules
        dash = "-"
        rows = []
        for m in mods:
            src = code(m["source"]) if m["source"] else dash
            plugin = code(m["plugin"])
            if not m["reported"]:
                rows.append([code(m["id"]), plugin, "Not started"] + [dash] * 8 + [src])
                continue
            cases = fmt_int(m["compared"])
            if m["compared"] != m["in_contract"]:
                cases += " of " + fmt_int(m["in_contract"])
            rows.append([code(m["id"]), plugin, verdict(m), cases, fmt_int(m["out_of_contract"]),
                         "{} / {}".format(fmt_int(m["bytes_compared"]), fmt_int(m["floats_compared"])),
                         "{} / {}".format(fmt_int(m["differing"]), fmt_int(m["differing_values"])),
                         fmt_int(m["max_ulp"]) if m["max_ulp"] is not None else dash,
                         "{} of {}".format(fmt_int(m["call_matching"]), fmt_int(m["call_compared"])),
                         "{} of {}".format(fmt_int(m["mutants_killed"]), fmt_int(m["mutants_total"])),
                         self._rerun(m), src])
        rep = [m for m in mods if m["reported"]]

        def total(key):
            return sum(m[key] for m in rep)

        ulps = [m["max_ulp"] for m in rep if m["max_ulp"] is not None]
        cases = fmt_int(total("compared"))
        if total("compared") != total("in_contract"):
            cases += " of " + fmt_int(total("in_contract"))
        rows.append(["Total", dash, "{} of {} match, {} of {} verified".format(
                         sum(1 for m in mods if is_match(m)), len(mods), sum(1 for m in mods if is_verified(m)), len(mods)),
                     cases, fmt_int(total("out_of_contract")),
                     "{} / {}".format(fmt_int(total("bytes_compared")), fmt_int(total("floats_compared"))),
                     "{} / {}".format(fmt_int(total("differing")), fmt_int(total("differing_values"))),
                     fmt_int(max(ulps)) if ulps else dash,
                     "{} of {}".format(fmt_int(total("call_matching")), fmt_int(total("call_compared"))),
                     "{} of {}".format(fmt_int(total("mutants_killed")), fmt_int(total("mutants_total"))),
                     "{} of {}".format(sum(1 for m in mods if rerun_state(m) == "yes"), len(mods)), dash])
        self.add("modules", "parity/<id>/results/summary.json and verify.json, in the module order of tools/docgen/data.py",
                 table(["Module", "Plugin", "Status", "Cases compared (in contract)", "Out of contract (candidates)",
                        "Values compared (bytes / floats)", "Differing", "Max ULP", "Call records matching",
                        "Mutants killed", "Re-run verified", "Source"], rows))

    @staticmethod
    def _rerun(m):
        state, v = rerun_state(m), m["verify"]
        if state == "none":
            return "-"
        if state == "no":
            return "No, {} differences".format(fmt_int(v["differences"]))
        return ("Yes, " if state == "yes" else "Stale, ") + (v["reran_at"] or "undated")

    def _notes(self):
        """Blockers, stand-ins, logic parity counts and surviving mutants, as the modules state them."""
        rows = []
        for m in self.d.modules:
            for b in m["blockers"]:
                rows.append([code(m["id"]), "blocker", b])
            for s in m["stand_ins"]:
                rows.append([code(m["id"]), "stand-in", s])
            if m["logic_parity"]:
                rows.append([code(m["id"]), "logic parity",
                             "{} compared cases use a stand-in on both sides".format(fmt_int(m["logic_parity"]))])
            for s in m["survivors"]:
                rows.append([code(m["id"]), "surviving mutant", s])
        if not rows:
            rows.append(["-", "-", "None reported yet."])
        self.add("notes", "parity/<id>/results/summary.json: blockers, stand_ins, cases.logic_parity, mutants.survivors",
                 table(["Module", "Kind", "Text"], rows))

    def _harness(self):
        """The optional harness summary, shown as it states: its top level entries, nothing inferred."""
        doc = self.d.harness
        rows = []
        for k, v in (doc or {}).items():
            if isinstance(v, (dict, list)):
                text = json.dumps(v, separators=(",", ":"))
                text = text if len(text) <= 100 else "{} entries, see the file".format(len(v))
            elif isinstance(v, bool):
                text = "true" if v else "false"
            else:
                text = "null" if v is None else str(v)
            rows.append([code(clean(k).replace("\\|", "|")), clean(text)])
        if not rows:
            rows.append(["-", "No harness summary yet." if doc is None else "The file states nothing."])
        self.add("harness", HARNESS_SUMMARY + ", top level entries as stated", table(["Entry", "Stated"], rows))

    def _fixtures(self):
        rows = []
        for x in sorted(self.d.fixtures, key=lambda x: x["path"]):
            rows.append([code(x["path"]), code(x["magic"]), x["kind"],
                         fmt_int(x["cases"]) if x["cases"] is not None else "-", fmt_int(x["bytes"])])
        if not rows:
            rows.append(["none", "-", "-", "-", "-"])
        self.add("fixtures", "scan of crates/ and plugins/ for *.hex",
                 table(["Fixture", "Magic", "Kind", "Cases", "Bytes"], rows))

    def _lock(self):
        rows = []
        tot = [0, 0, 0, 0, 0]
        for s in LOCK_SETS:
            r = self.d.lock["corpora"][s]
            vals = [r["rendered_at_baseline"], r["identical"], r["differ"], r["refused_at_baseline"],
                    r.get("refused_now_render", 0)]
            tot = [a + b for a, b in zip(tot, vals)]
            rows.append([s] + [fmt_int(v) for v in vals])
        rows.append(["all"] + [fmt_int(v) for v in tot])
        self.add("lock", DATA + "/lock.json corpora",
                 table(["Corpus", "Rendered at the baseline", "Identical now", "Differ", "Refused at the baseline",
                        "Of those, render now"], rows))

    def _tests(self):
        rows = []
        for name, (t, i) in self.d.rust_tests.items():
            rows.append([code(name), "`#[test]`", fmt_int(t) + (" ({} ignored)".format(fmt_int(i)) if i else "")])
        for name, n in self.d.py_tests.items():
            rows.append([code("tools/" + name), "`def test_`", fmt_int(n)])
        self.add("tests", "scan of the .rs files of every crate and of tools/test_*.py",
                 table(["Suite", "Counted", "Tests"], rows))

    def fill(self, text):
        """Fills every figure and table marker; returns (text, unknown keys)."""
        unknown = []
        nl = "\r\n" if "\r\n" in text else "\n"

        def fsub(m):
            fact = self.d.fact(m.group(1))
            if fact is None:
                unknown.append("f:" + m.group(1))
                return m.group(0)
            return "<!--f:{}-->{}<!--/f-->".format(m.group(1), fact["text"])

        def gsub(m):
            k = m.group(2)
            if k not in self.blocks:
                unknown.append("gen:" + k)
                return m.group(0)
            return m.group(1) + nl.join(self.blocks[k]) + nl + m.group(4)

        return G_RE.sub(gsub, F_RE.sub(fsub, text)), unknown


def facts_json(data, gen):
    doc = OrderedDict([
        ("generator", "tools/gen_docs.py"),
        ("note", "Generated from docs/data/ and scans of the tree; do not edit. Each fact: value, the text a "
                 "marker shows, and its source."),
        ("facts", data.facts),
        ("blocks", gen.sources),
    ])
    return json.dumps(doc, indent=1) + "\n"


def build(root=ROOT):
    """(outputs, unknown, data): outputs maps repository paths to their full generated text."""
    data = Data(root)
    gen = Generator(data)
    outputs = OrderedDict()
    unknown = []
    for rel in md_files(root):
        text = read_text(os.path.join(root, rel))
        new, unk = gen.fill(text)
        unknown += ["{}: {}".format(rel, u) for u in unk]
        outputs[rel] = new
    outputs[FACTS] = facts_json(data, gen)
    return outputs, unknown, data
