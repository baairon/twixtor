"""Load the snapshots of docs/data/, scan the tree, and derive every figure the documentation quotes.

Snapshots (committed, rebuilt by `gen_docs.py --refresh-data`, except lock.json, which
`tools/parity_lock.py record` writes):
  controls.tsv    effect, id, kind: every param and layer of the four effects (`vfx list --json --all`,
                  hidden effects left out); kind is the param kind or `layer`
  gaps.tsv        id, title, effects, controls: the parity gaps register, from the `gaps` arrays of the
                  same listing; effects and controls are comma separated
  references.tsv  family, gap, case, effect, controls, result: one row per checklist case of the reference
                  families; result is pass, fail or missing (no reference render yet)
  lock.json       the regression lock record: baseline ref and commit, date, counts per corpus

Scans (at generation time, of files only):
  hex fixtures    every *.hex under crates/ and plugins/: decoded bytes, and the case count of the header
                  when a generator beside it documents the magic as `b"MAGIC", u32 ncases`
  tests           `#[test]` and `#[ignore` lines per crate (crates/*, plugins/*); `def test_` lines per
                  tools/test_*.py
"""

import csv
import io
import json
import os
import re
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = "docs/data"
SKIP_DIRS = {".git", ".claude", "target", "handoff", "node_modules", "__pycache__"}

CONTROLS_HEADER = ["effect", "id", "kind"]
GAPS_HEADER = ["id", "title", "effects", "controls"]
REFERENCES_HEADER = ["family", "gap", "case", "effect", "controls", "result"]
RESULTS = ("pass", "fail", "missing")
LOCK_SETS = ("L1", "L2", "L3")

TEST_RE = re.compile(r"^[ \t]*#\[test\][ \t]*$", re.M)
IGNORE_RE = re.compile(r"^[ \t]*#\[ignore\b", re.M)
PY_TEST_RE = re.compile(r"^[ \t]*def test_\w*\(", re.M)
NCASES_RE = re.compile(r"b\"([A-Za-z0-9]{4})\",\s*u32 ncases")
MAGIC_RE = re.compile(r"b\"([A-Za-z0-9]{4})\"")


class DataError(Exception):
    """A snapshot that cannot be read, or that disagrees with another one."""


# ---- reading and formatting -----------------------------------------------------------------------

def read_text(path):
    with open(path, "rb") as h:
        return h.read().decode("utf-8")


def read_tsv(path, header):
    """Rows of a tab separated table as dicts; the header must be exactly `header`."""
    text = read_text(path)
    rows = list(csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE))
    if not rows or rows[0] != header:
        raise DataError("{}: the header must be {}".format(path, "\\t".join(header)))
    out = []
    for n, r in enumerate(rows[1:], 2):
        if not r:
            continue
        if len(r) != len(header):
            raise DataError("{}:{}: {} fields, expected {}".format(path, n, len(r), len(header)))
        out.append(OrderedDict(zip(header, r)))
    return out


def tsv_text(header, rows):
    """A table as text: tab separated, LF line ends, no quoting."""
    lines = ["\t".join(header)]
    for r in rows:
        cells = [str(r[h]) for h in header]
        for c in cells:
            if "\t" in c or "\n" in c:
                raise DataError("a cell holds a tab or a line end: {!r}".format(c))
        lines.append("\t".join(cells))
    return "\n".join(lines) + "\n"


def fmt_int(n):
    return "{:,}".format(n)


def fmt_pct(part, whole):
    return "{:.1f}".format(100.0 * part / whole) if whole else "0.0"


def split_list(cell):
    return [c for c in cell.split(",") if c] if cell and cell != "-" else []


def gap_key(gid):
    """Register order: G1, G1b, ..., G1h, G2, ..., G10."""
    m = re.match(r"([A-Za-z]+)(\d+)([a-z]*)$", gid)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (gid, 0, "")


def snake(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def walk(root, top):
    """Files under root/top in sorted order, as repository paths, skipping build and tool folders."""
    out = []
    base = os.path.join(root, top)
    for dirpath, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for f in sorted(files):
            out.append(os.path.relpath(os.path.join(dirpath, f), root).replace(os.sep, "/"))
    return out


# ---- scans ----------------------------------------------------------------------------------------

def crate_dirs(root):
    """(package name, repository path) of every crate under crates/ and plugins/, sorted by path."""
    out = []
    for top in ("crates", "plugins"):
        base = os.path.join(root, top)
        if not os.path.isdir(base):
            continue
        for d in sorted(os.listdir(base)):
            toml = os.path.join(base, d, "Cargo.toml")
            if os.path.isfile(toml):
                m = re.search(r"^name\s*=\s*\"([^\"]+)\"", read_text(toml), re.M)
                out.append((m.group(1) if m else d, top + "/" + d))
    return out


def scan_rust_tests(root):
    """OrderedDict package name -> (tests, ignored), counted on `#[test]` and `#[ignore` lines."""
    out = OrderedDict()
    for name, rel in crate_dirs(root):
        tests = ignored = 0
        for f in walk(root, rel):
            if f.endswith(".rs"):
                text = read_text(os.path.join(root, f))
                tests += len(TEST_RE.findall(text))
                ignored += len(IGNORE_RE.findall(text))
        out[name] = (tests, ignored)
    return out


def scan_python_tests(root):
    """OrderedDict file name -> count of `def test_` lines, for every tools/test_*.py."""
    out = OrderedDict()
    tools = os.path.join(root, "tools")
    if os.path.isdir(tools):
        for f in sorted(os.listdir(tools)):
            if f.startswith("test_") and f.endswith(".py"):
                out[f] = len(PY_TEST_RE.findall(read_text(os.path.join(tools, f))))
    return out


def fixture_magics(root, folder):
    """(magics with a case count, magics described as not bit exact) of the generators in a folder."""
    counted, differential = set(), set()
    full = os.path.join(root, folder)
    for f in sorted(os.listdir(full)):
        if not f.endswith(".py"):
            continue
        text = read_text(os.path.join(full, f))
        counted.update(NCASES_RE.findall(text))
        if "not bit exact" in text:
            for m in MAGIC_RE.findall(text):
                differential.add(m)
    return counted, differential - counted


def scan_fixtures(root):
    """One dict per *.hex fixture: path, folder, magic, kind (model, differential or other), bytes, cases."""
    out = []
    magics = {}
    for top in ("crates", "plugins"):
        for f in walk(root, top):
            if not f.endswith(".hex"):
                continue
            folder = f.rsplit("/", 1)[0]
            if folder not in magics:
                magics[folder] = fixture_magics(root, folder)
            counted, differential = magics[folder]
            digits = "".join(read_text(os.path.join(root, f)).split())
            if len(digits) % 2 or re.search(r"[^0-9a-fA-F]", digits):
                raise DataError("{}: not hex text".format(f))
            head = bytes.fromhex(digits[:16])
            magic = head[:4].decode("ascii", "replace")
            cases = None
            if magic in counted and len(head) == 8:
                kind = "model"
                cases = int.from_bytes(head[4:8], "little")
            elif magic in differential:
                kind = "differential"
            else:
                kind = "other"
            out.append(OrderedDict([("path", f), ("magic", magic), ("kind", kind), ("bytes", len(digits) // 2),
                                    ("cases", cases)]))
    return out


# ---- the snapshots from a listing (refresh) -------------------------------------------------------

def controls_from_listing(listing):
    """controls.tsv rows from `vfx list --json --all`: the visible effects, params then layers."""
    rows = []
    for e in listing:
        if e.get("hidden"):
            continue
        for p in e.get("params", []):
            rows.append(OrderedDict([("effect", e["id"]), ("id", p["id"]), ("kind", p["kind"])]))
        for l in e.get("layers", []):
            rows.append(OrderedDict([("effect", e["id"]), ("id", l["id"]), ("kind", "layer")]))
    return rows


def gaps_from_listing(listing):
    """gaps.tsv rows from the `gaps` arrays of the visible effects, in register order.

    A gap listed by several effects must have the same title and controls in each, so that one row
    holds it exactly; anything else is an error to fix in the register or in this format."""
    seen = OrderedDict()
    for e in listing:
        if e.get("hidden"):
            continue
        for g in e.get("gaps") or []:
            controls = list(g.get("controls", []))
            if g["id"] in seen:
                row = seen[g["id"]]
                if row["title"] != g["title"] or row["_controls"] != controls:
                    raise DataError("gap {} differs between {} and {}".format(g["id"], row["_effects"][0], e["id"]))
                row["_effects"].append(e["id"])
            else:
                seen[g["id"]] = {"title": g["title"], "_controls": controls, "_effects": [e["id"]]}
    rows = []
    for gid in sorted(seen, key=gap_key):
        r = seen[gid]
        rows.append(OrderedDict([("id", gid), ("title", r["title"]), ("effects", ",".join(r["_effects"])),
                                 ("controls", ",".join(r["_controls"]))]))
    return rows


FAMILY_RE = re.compile(r'FamilyInfo\s*\{\s*id:\s*"([^"]+)",\s*gaps:\s*&\[([^\]]*)\]')


def family_gaps(root=ROOT):
    """The register ids each reference case family settles, family id -> [gap ids], read from the
    FAMILIES table of crates/fx-parity/src/families.rs (a case carries its family id only)."""
    path = os.path.join(root, "crates", "fx-parity", "src", "families.rs")
    if not os.path.isfile(path):
        return {}
    found = OrderedDict()
    for fam, ids in FAMILY_RE.findall(read_text(path)):
        found[fam] = re.findall(r'"([^"]+)"', ids)
    return found


def references_from_cases(cases, statuses=None, gaps_of=None):
    """references.tsv rows from a family testset's cases.json and, when given, the statuses of
    `fx-parity run --json` (pass, FAIL, no reference) by case id. Cases without a family are left out.
    The gap column is the case's own `gap` when it has one, else the gaps of its family (`gaps_of`)."""
    word = {"pass": "pass", "FAIL": "fail", "no reference": "missing"}
    rows = []
    for c in cases.get("cases", []):
        fam = c.get("family")
        if not fam:
            continue
        ids = [p.split("=", 1)[0] for p in c.get("params", [])] + sorted(c.get("layers", {}))
        status = (statuses or {}).get(c["id"], "no reference")
        if status not in word:
            raise DataError("case {}: unknown status {!r}".format(c["id"], status))
        gap = c.get("gap") or ",".join((gaps_of or {}).get(fam, [])) or "-"
        rows.append(OrderedDict([("family", fam), ("gap", gap), ("case", c["id"]),
                                 ("effect", c["effect"]), ("controls", ",".join(ids) or "-"),
                                 ("result", word[status])]))
    rows.sort(key=lambda r: (r["family"], r["case"]))
    return rows


# ---- the facts ------------------------------------------------------------------------------------

class Data:
    """Every input, loaded once, and the facts derived from them. `root` is the repository root."""

    def __init__(self, root=ROOT):
        self.root = root
        self.d = os.path.join(root, DATA)
        self.controls = read_tsv(self.p("controls.tsv"), CONTROLS_HEADER)
        self.gaps = read_tsv(self.p("gaps.tsv"), GAPS_HEADER)
        self.references = read_tsv(self.p("references.tsv"), REFERENCES_HEADER)
        try:
            self.lock = json.loads(read_text(self.p("lock.json")))
        except ValueError as e:
            raise DataError("{}: {}".format(self.p("lock.json"), e)) from None
        self.fixtures = scan_fixtures(root)
        self.rust_tests = scan_rust_tests(root)
        self.py_tests = scan_python_tests(root)
        self._validate()
        self._facts()

    def p(self, name):
        return os.path.join(self.d, name)

    def _validate(self):
        known = {(r["effect"], r["id"]) for r in self.controls}
        if len(known) != len(self.controls):
            raise DataError("controls.tsv lists a control twice")
        for g in self.gaps:
            for e in split_list(g["effects"]):
                for c in split_list(g["controls"]):
                    if (e, c) not in known:
                        raise DataError("gaps.tsv: {} names {} of {}, which controls.tsv does not list".format(
                            g["id"], c, e))
        for r in self.references:
            if r["result"] not in RESULTS:
                raise DataError("references.tsv: case {} has result {!r} (pass, fail or missing)".format(
                    r["case"], r["result"]))
        for s in LOCK_SETS:
            row = self.lock.get("corpora", {}).get(s)
            if not isinstance(row, dict):
                raise DataError("lock.json: no corpus {}".format(s))
            for k in ("rendered_at_baseline", "identical", "differ", "refused_at_baseline"):
                if not isinstance(row.get(k), int):
                    raise DataError("lock.json: corpora.{}.{} is not a count".format(s, k))

    def gap_pairs(self):
        """The (effect, control) pairs the open gaps cover."""
        out = set()
        for g in self.gaps:
            for e in split_list(g["effects"]):
                for c in split_list(g["controls"]):
                    out.add((e, c))
        return out

    def graded_pairs(self):
        """(effect, control) pairs set by at least one passing reference case and by no failing one."""
        passed, failed = set(), set()
        for r in self.references:
            pairs = {(r["effect"], c) for c in split_list(r["controls"])}
            if r["result"] == "pass":
                passed |= pairs
            elif r["result"] == "fail":
                failed |= pairs
        known = {(r["effect"], r["id"]) for r in self.controls}
        return (passed - failed) & known

    def _facts(self):
        f = OrderedDict()

        def put(key, value, source, text=None):
            if key in f:
                raise DataError("fact {} defined twice".format(key))
            if text is None:
                text = fmt_int(value) if isinstance(value, int) else str(value)
            f[key] = OrderedDict([("value", value), ("text", text), ("source", source)])

        def pct(key, part, whole, source):
            t = fmt_pct(part, whole)
            put(key, float(t), source, t)

        C = DATA + "/controls.tsv"
        n_controls = len(self.controls)
        put("controls_total", n_controls, C + " rows")
        put("controls_params", sum(1 for r in self.controls if r["kind"] != "layer"), C + " rows whose kind is not layer")
        put("controls_layers", sum(1 for r in self.controls if r["kind"] == "layer"), C + " rows of kind layer")
        put("effects_total", len({r["effect"] for r in self.controls}), C + " distinct effect")

        G = DATA + "/gaps.tsv"
        pairs = self.gap_pairs()
        put("gaps_open", len(self.gaps), G + " rows")
        put("gap_controls", len(pairs), G + " distinct (effect, control) pairs of effects x controls")
        pct("gap_controls_pct", len(pairs), n_controls, "gap_controls over controls_total")

        R = DATA + "/references.tsv"
        res = [r["result"] for r in self.references]
        put("family_cases", len(res), R + " rows")
        put("families_total", len({r["family"] for r in self.references}), R + " distinct family")
        put("graded_cases", res.count("pass"), R + " rows with result pass")
        put("graded_failed", res.count("fail"), R + " rows with result fail")
        put("graded_missing", res.count("missing"), R + " rows with result missing")
        pct("graded_pct", res.count("pass"), len(res), "graded_cases over family_cases")
        graded = self.graded_pairs()
        put("graded_controls", len(graded), R + " (effect, control) pairs set by a pass row and by no fail row, "
            "that " + C + " lists")
        put("unproven", n_controls - len(graded), "controls_total minus graded_controls")
        pct("unproven_pct", n_controls - len(graded), n_controls, "unproven over controls_total")

        X = "scan of crates/ and plugins/ for *.hex"
        model = [x for x in self.fixtures if x["kind"] == "model"]
        put("fixtures_total", len(self.fixtures), X)
        put("model_fixtures", len(model), X + ", kind model (a generator documents b\"MAGIC\", u32 ncases)")
        put("model_cases", sum(x["cases"] for x in model), X + ", sum of the header case counts of the model fixtures")
        put("model_bytes", sum(x["bytes"] for x in model), X + ", decoded bytes of the model fixtures")
        put("differential_fixtures", sum(1 for x in self.fixtures if x["kind"] == "differential"),
            X + ", kind differential (the generator says not bit exact)")

        L = DATA + "/lock.json"
        base = self.lock.get("baseline", {})
        corpora = self.lock["corpora"]
        put("lock_ref", base.get("commit", "")[:7] or base.get("ref", ""), L + " baseline commit, 7 digits")
        put("lock_date", self.lock.get("date", ""), L + " date")
        tot = {"rendered_at_baseline": 0, "identical": 0, "differ": 0, "refused_at_baseline": 0,
               "refused_now_render": 0}
        for s in LOCK_SETS:
            row = corpora[s]
            k = "lock_" + s.lower()
            put(k + "_total", row["rendered_at_baseline"], L + " corpora." + s + ".rendered_at_baseline")
            put(k + "_identical", row["identical"], L + " corpora." + s + ".identical")
            put(k + "_refused", row["refused_at_baseline"], L + " corpora." + s + ".refused_at_baseline")
            put(k + "_refused_render", row.get("refused_now_render", 0),
                L + " corpora." + s + ".refused_now_render")
            for key in tot:
                tot[key] += row.get(key, 0)
        put("lock_total", tot["rendered_at_baseline"], L + " sum of rendered_at_baseline")
        put("lock_identical", tot["identical"], L + " sum of identical")
        put("lock_differ", tot["differ"], L + " sum of differ")
        pct("lock_pct", tot["identical"], tot["rendered_at_baseline"], "lock_identical over lock_total")
        put("lock_refused", tot["refused_at_baseline"], L + " sum of refused_at_baseline")
        put("lock_refused_render", tot["refused_now_render"], L + " sum of refused_now_render")

        T = "scan of `#[test]` lines in the .rs files of "
        put("tests_rust", sum(t for t, _ in self.rust_tests.values()), T + "every crate")
        put("tests_rust_ignored", sum(i for _, i in self.rust_tests.values()), "scan of `#[ignore` lines, every crate")
        for name, (t, _) in self.rust_tests.items():
            put("tests_rust_" + snake(name), t, T + name)
        P = "scan of `def test_` lines in "
        put("tests_python", sum(self.py_tests.values()), P + "tools/test_*.py")
        for name, n in self.py_tests.items():
            put("tests_py_" + snake(name[len("test_"):-len(".py")]), n, P + "tools/" + name)
        self.facts = f

    def fact(self, key):
        return self.facts.get(key)
