"""Load the snapshots of docs/data/, scan the tree, and derive every figure the documentation quotes.

Snapshots (committed, rebuilt by `gen_docs.py --refresh-data`, except lock.json, which
`tools/parity_lock.py record` writes):
  controls.tsv    effect, id, kind: every param and layer of the four effects (`vfx list --json --all`,
                  hidden effects left out); kind is the param kind or `layer`
  gaps.tsv        id, title, effects, controls: the register of stated rules (the open parity gaps), from
                  the `gaps` arrays of the same listing; effects and controls are comma separated. The
                  Limits table of the README is checked against it (tools/check_docs.py, check limits).
  lock.json       the regression lock record: baseline ref and commit, date, counts per corpus
  references.tsv  is no longer read: it graded reference renders of the original that were never made

Module results, written by the agents of each module and never by this tool:
  parity/<id>/results/summary.json    the measured result of one module (schema: load_module below)
  parity/<id>/results/verify.json     the re-run of a separate verification agent
  parity/harness/results/summary.json optional; shown as it states, whatever its shape

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
LOCK_SETS = ("L1", "L2", "L3")

# The modules, in table order: id, plugin file, rewrite effect.
MODULES = (
    ("retime-time", "Twixtor_64.aex", "retime"),
    ("retime-flow", "Twixtor_64.aex", "retime"),
    ("create-vectors", "CreateMotionVectors_64.aex", "create-vectors"),
    ("vectors-in", "TwixtorProVectorsIn_64.aex", "vectors-in"),
    ("fgbg-core", "TwixtorPro_64.aex", "retime-fgbg"),
    ("fgbg-layers", "TwixtorPro_64.aex", "retime-fgbg"),
)
STATUS_LABELS = OrderedDict([("verified", "Verified"), ("in_progress", "In progress"), ("blocked", "Blocked"),
                             ("not_started", "Not started")])
MIN_CASES = 150  # in-contract cases a Verified module compared
RESULTS_DIR = "parity/{}/results"
HARNESS_SUMMARY = "parity/harness/results/summary.json"

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



# ---- module results -------------------------------------------------------------------------------

DASHES = tuple(chr(c) for c in range(0x2010, 0x2016)) + (chr(0x2212),)


def clean(text):
    """Text from a result file as one line of ASCII for a table cell: dashes become hyphens, other
    non-ASCII characters become ?, line ends and tabs become spaces, a pipe is escaped."""
    out = []
    for c in str(text):
        if c in DASHES:
            c = "-"
        elif ord(c) > 127:
            c = "?"
        elif c in "\r\n\t":
            c = " "
        elif ord(c) < 32 or ord(c) == 127:
            c = "?"
        out.append(c)
    return "".join(out).strip().replace("|", "\\|")


def read_json(path, what):
    try:
        return json.loads(read_text(path))
    except (ValueError, UnicodeDecodeError) as e:
        raise DataError("{}: {}".format(what, e)) from None


def _count(obj, key, where):
    v = obj.get(key, 0)
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise DataError("{}: {} is not a count".format(where, key))
    return v


def _section(doc, key, where):
    v = doc.get(key, {})
    if not isinstance(v, dict):
        raise DataError("{}: {} is not an object".format(where, key))
    return v


def _lines(doc, key, where):
    v = doc.get(key, [])
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise DataError("{}: {} is not a list of strings".format(where, key))
    return [clean(x) for x in v]


def unmet_rule(m):
    """Why a module does not meet the Verified rule; an empty list when it does. The rule: 150 or more
    in-contract cases compared, 0 differing (cases and values), every call record matching, every mutant
    killed. A check with nothing in it (no call record, no mutant) does not count."""
    out = []
    if m["compared"] < MIN_CASES:
        out.append("fewer than {} cases compared".format(MIN_CASES))
    if m["compared"] < m["in_contract"]:
        out.append("not every in-contract case compared")
    if m["differing"] or m["differing_values"] or m["identical"] != m["compared"]:
        out.append("a case or value differs")
    if not m["call_compared"] or m["call_matching"] != m["call_compared"]:
        out.append("call records not all matching")
    if not m["mutants_total"] or m["mutants_killed"] != m["mutants_total"] or m["survivors"]:
        out.append("a mutant survives or none ran")
    return out


def load_module(root, mid, plugin, effect):
    """One module as a flat dict. A module without a summary.json is `reported` False and not started."""
    rel = RESULTS_DIR.format(mid)
    m = OrderedDict([("id", mid), ("plugin", plugin), ("effect", effect), ("reported", False),
                     ("status", "not_started"), ("source", None), ("blockers", []), ("stand_ins", []),
                     ("survivors", []), ("measured_at", ""), ("in_contract", 0), ("compared", 0),
                     ("identical", 0), ("differing", 0), ("out_of_contract", 0), ("logic_parity", 0),
                     ("bytes_compared", 0), ("floats_compared", 0), ("differing_values", 0), ("max_ulp", None),
                     ("call_compared", 0), ("call_matching", 0), ("mutants_total", 0), ("mutants_killed", 0),
                     ("verify", None), ("unmet", [])])
    path = os.path.join(root, rel, "summary.json")
    if os.path.isfile(path):
        where = rel + "/summary.json"
        doc = read_json(path, where)
        if not isinstance(doc, dict):
            raise DataError("{}: not a JSON object".format(where))
        if doc.get("module") != mid:
            raise DataError("{}: module is {!r}, expected {!r}".format(where, doc.get("module"), mid))
        status = doc.get("status")
        if status not in STATUS_LABELS:
            raise DataError("{}: status {!r} is not one of {}".format(where, status, ", ".join(STATUS_LABELS)))
        cases, values = _section(doc, "cases", where), _section(doc, "values", where)
        calls, muts = _section(doc, "call_records", where), _section(doc, "mutants", where)
        ulp = values.get("max_ulp")
        if ulp is not None and (isinstance(ulp, bool) or not isinstance(ulp, int) or ulp < 0):
            raise DataError("{}: values.max_ulp is not a count or null".format(where))
        survivors = muts.get("survivors", [])
        if not isinstance(survivors, list) or not all(isinstance(x, str) for x in survivors):
            raise DataError("{}: mutants.survivors is not a list of strings".format(where))
        if not isinstance(doc.get("plugin", plugin), str):
            raise DataError("{}: plugin is not a string".format(where))
        m.update([("reported", True), ("status", status), ("source", rel + "/summary.json"),
                  ("plugin", clean(doc.get("plugin") or plugin)), ("effect", clean(doc.get("effect") or effect)),
                  ("blockers", _lines(doc, "blockers", where)), ("stand_ins", _lines(doc, "stand_ins", where)),
                  ("survivors", [clean(x) for x in survivors]), ("measured_at", clean(doc.get("measured_at", ""))),
                  ("in_contract", _count(cases, "in_contract", where)), ("compared", _count(cases, "compared", where)),
                  ("identical", _count(cases, "identical", where)), ("differing", _count(cases, "differing", where)),
                  ("out_of_contract", _count(cases, "out_of_contract", where)),
                  ("logic_parity", _count(cases, "logic_parity", where)),
                  ("bytes_compared", _count(values, "bytes_compared", where)),
                  ("floats_compared", _count(values, "floats_compared", where)),
                  ("differing_values", _count(values, "differing_values", where)), ("max_ulp", ulp),
                  ("call_compared", _count(calls, "compared", where)), ("call_matching", _count(calls, "matching", where)),
                  ("mutants_total", _count(muts, "total", where)), ("mutants_killed", _count(muts, "killed", where))])
        m["unmet"] = unmet_rule(m)
    vpath = os.path.join(root, rel, "verify.json")
    if os.path.isfile(vpath):
        where = rel + "/verify.json"
        v = read_json(vpath, where)
        if not isinstance(v, dict) or v.get("module") != mid or not isinstance(v.get("matches"), bool):
            raise DataError("{}: needs module {!r} and a boolean matches".format(where, mid))
        diffs = v.get("differences", [])
        if not isinstance(diffs, list):
            raise DataError("{}: differences is not a list".format(where))
        m["verify"] = OrderedDict([("matches", v["matches"]), ("reran_at", clean(v.get("reran_at", ""))),
                                   ("differences", len(diffs))])
    return m


def is_match(m):
    """Whether the module matches the original on the emulator run: 150 or more cases compared, 0 differing
    (cases and values) and every call record matching. Cases without a reference and surviving mutants do not
    enter it; they are what Verified adds. A blocked module does not match."""
    return (m["reported"] and m["status"] in ("verified", "in_progress") and m["compared"] >= MIN_CASES
            and not m["differing"] and not m["differing_values"] and m["identical"] == m["compared"]
            and m["call_compared"] > 0 and m["call_matching"] == m["call_compared"])


def verdict(m):
    """The status a table shows: the file's status, except that a claim of Verified the numbers do not
    support is shown as such, and a module in progress that matches on the emulator run shows Matches."""
    if m["status"] == "verified" and m["unmet"]:
        return "Unconfirmed (file says verified, rule unmet)"
    if m["status"] == "in_progress" and is_match(m):
        return "Matches"
    return STATUS_LABELS[m["status"]]


def is_verified(m):
    return m["status"] == "verified" and not m["unmet"]


def rerun_state(m):
    """yes, no, stale or none: whether a verification agent's re-run agrees with the summary."""
    v = m["verify"]
    if v is None:
        return "none"
    if not v["matches"]:
        return "no"
    if m["measured_at"] and v["reran_at"][:10] < m["measured_at"][:10]:
        return "stale"
    return "yes"


def load_modules(root):
    return [load_module(root, mid, plugin, effect) for mid, plugin, effect in MODULES]


def load_harness(root):
    """The parity/harness/results/summary.json object, or None when it does not exist."""
    path = os.path.join(root, HARNESS_SUMMARY)
    if not os.path.isfile(path):
        return None
    doc = read_json(path, HARNESS_SUMMARY)
    if not isinstance(doc, dict):
        raise DataError("{}: not a JSON object".format(HARNESS_SUMMARY))
    return doc


# ---- the facts ------------------------------------------------------------------------------------

class Data:
    """Every input, loaded once, and the facts derived from them. `root` is the repository root."""

    def __init__(self, root=ROOT):
        self.root = root
        self.d = os.path.join(root, DATA)
        self.controls = read_tsv(self.p("controls.tsv"), CONTROLS_HEADER)
        self.gaps = read_tsv(self.p("gaps.tsv"), GAPS_HEADER)
        try:
            self.lock = json.loads(read_text(self.p("lock.json")))
        except ValueError as e:
            raise DataError("{}: {}".format(self.p("lock.json"), e)) from None
        self.modules = load_modules(root)
        self.harness = load_harness(root)
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
        for s in LOCK_SETS:
            row = self.lock.get("corpora", {}).get(s)
            if not isinstance(row, dict):
                raise DataError("lock.json: no corpus {}".format(s))
            for k in ("rendered_at_baseline", "identical", "differ", "refused_at_baseline"):
                if not isinstance(row.get(k), int):
                    raise DataError("lock.json: corpora.{}.{} is not a count".format(s, k))

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

        M = "parity/<id>/results/summary.json and verify.json of the six modules"
        put("modules_total", len(self.modules), "the module list of tools/docgen/data.py")
        put("modules_reported", sum(1 for m in self.modules if m["reported"]), M + ", modules that have a summary.json")
        put("modules_verified", sum(1 for m in self.modules if is_verified(m)),
            M + ", status verified and the status rule met")
        put("modules_match", sum(1 for m in self.modules if is_match(m)),
            M + ", 150 or more cases compared, none differing and every call record matching on the emulator run")
        put("cases_compared", sum(m["compared"] for m in self.modules if m["reported"]),
            M + ", sum of cases.compared")
        put("modules_rerun", sum(1 for m in self.modules if rerun_state(m) == "yes"),
            M + ", verify.json says matches, dated on or after measured_at")

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
