"""Fill the figure and table markers of the Markdown files, and write the machine readable fact index."""

import json
import os
import re
from collections import OrderedDict

from .data import DATA, LOCK_SETS, ROOT, Data, fmt_int, read_text, split_list

F_RE = re.compile(r"<!--f:([A-Za-z0-9_.-]+)-->(.*?)<!--/f-->")
G_RE = re.compile(r"(<!--gen:([A-Za-z0-9_.-]+)-->\r?\n)(.*?)(<!--/gen:\2-->)", re.S)
FACTS = DATA + "/facts.json"
RANGE_RE = re.compile(r"^([a-z]+)(\d+)(_[a-z0-9_]+)$")


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


def id_list(ids):
    """Ids in code font; three or more consecutive numbered ids (pt1_use, pt2_use, ...) as a range."""
    out, i = [], 0
    while i < len(ids):
        m = RANGE_RE.match(ids[i])
        j = i
        if m:
            while j + 1 < len(ids):
                n = RANGE_RE.match(ids[j + 1])
                if not (n and n.group(1) == m.group(1) and n.group(3) == m.group(3)
                        and int(n.group(2)) == int(m.group(2)) + (j + 1 - i)):
                    break
                j += 1
        if j - i >= 2:
            out.append("{} to {}".format(code(ids[i]), code(ids[j])))
        else:
            out.extend(code(x) for x in ids[i:j + 1])
        i = j + 1
    return ", ".join(out) if out else "-"


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
        self._gaps()
        self._families()
        self._fixtures()
        self._lock()
        self._tests()

    def add(self, name, source, lines):
        self.blocks[name] = lines
        self.sources[name] = source

    def _gaps(self):
        refs = {}
        for r in self.d.references:
            # a family that settles several gaps lists them all, and each case counts for each
            for gid in split_list(r["gap"]):
                refs.setdefault(gid, []).append(r["result"])
        rows = []
        for g in self.d.gaps:
            got = refs.get(g["id"], [])
            graded = "{} of {}".format(fmt_int(got.count("pass")), fmt_int(len(got))) if got else "none yet"
            rows.append([g["id"], g["title"], id_list(split_list(g["effects"])), id_list(split_list(g["controls"])),
                         graded])
        if not rows:
            rows.append(["none", "-", "-", "-", "-"])
        self.add("gaps", DATA + "/gaps.tsv, graded cases from " + DATA + "/references.tsv",
                 table(["Gap", "Open point", "Effects", "Controls", "Graded reference cases"], rows))

    def _families(self):
        fams = OrderedDict()
        for r in self.d.references:
            fams.setdefault((r["family"], r["gap"]), []).append(r["result"])
        rows = []
        for (fam, gap), res in sorted(fams.items()):
            rows.append([code(fam), gap, fmt_int(len(res)), fmt_int(res.count("pass")), fmt_int(res.count("fail")),
                         fmt_int(res.count("missing"))])
        if not rows:
            rows.append(["none yet", "-", "0", "0", "0", "0"])
        self.add("families", DATA + "/references.tsv by family",
                 table(["Family", "Gap", "Cases", "Pass", "Fail", "Missing"], rows))

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
