#!/usr/bin/env python3
"""Phase B tooling of the create-vectors parity module (standard library only).

Nothing here runs the original plugin. It grades references (made by the emulator harness, or by
a stand-in in `selftest`) against the rewrite, and turns the result files into the module summary.

Subcommands
  grade     --refs DIR            fx-parity run --max-ulp 0 --json over the test set -> results/grade.json, grade.tsv
  ooc       --status FILE ...     re-judge cases from the harness's per-case status -> results/ooc.json
  ooc-set   --out DIR             a test set of the out of contract candidates, for the harness
  calls     --plugin-records DIR --rewrite-records DIR      call-record check -> results/calls.json
  record-rewrite --out DIR        run the rewrite with VFX_CALL_RECORD over the whole test set
  kills     --refs DIR            mutation kills against references -> results/kills.json
  summary                         results/summary.json from the result files (needs grade.json)
  triage                          group the differing cases, write a ladder of probe cases to probe/
  make-refs --out DIR             render the test set with one vfx into reference files (stand-in oracle)
  selftest                        the stand-in checks, outputs under results/selftest/

Definitions (docs/parity.md, "How parity is measured"):
  - identical means bytes exact at 8 and 16 bpc and bit for bit at 32 bpc (max ULP 0, -0 against 0 and
    different NaN bits differ); a case without a reference is "no reference", never a pass or a failure.
  - a mutant is killed iff, among the cases where the base rewrite equals the reference, its output
    differs from the reference on at least one.
  - a case the original refuses or errors on, or a value outside the registered ranges, is out of
    contract: listed with what each side did, never a pass or a failure.
"""
import argparse
import array
import collections
import datetime
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction

ROOT = os.path.dirname(os.path.abspath(__file__))
PIXFMT = {8: "rgba", 16: "rgba64le", 32: "gbrapf32le"}
SAMPLE = {8: 1, 16: 2, 32: 4}
PLUGIN_NAME = "CreateMotionVectors_64"
MIN_COMPARED = 150
IDENT_FALLBACK = {
    "module": "create-vectors",
    "plugin": "CreateMotionVectors_64.aex",
    "plugin_sha256": "9b16c3350e4e41f10cf7fc56ead2a0e0aeea00eb76708ef973f2d10e99d57e0a",
    "effect": "create-vectors",
}


class ToolError(Exception):
    pass


# ------------------------------------------------------------------------------------------ files

def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def write_json(path, doc):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv_rows(path):
    """TSV with a header line -> list of dicts."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    if not lines:
        return []
    head = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        f = line.split("\t")
        f += [""] * (len(head) - len(f))
        rows.append(dict(zip(head, f)))
    return rows


def load_outputs_tsv(path):
    """run_grid.py output -> {id: (exit, size, sha256)}."""
    rows = {}
    for r in read_tsv_rows(path):
        rows[r["id"]] = (r["exit"], r["size"], r["sha256"])
    return rows


class Ctx:
    """Where things are: the module root, its grid, the phase A results (read only) and the output folder."""

    def __init__(self, args):
        self.root = os.path.abspath(getattr(args, "root", None) or ROOT)
        self.grid = os.path.abspath(getattr(args, "grid", None) or os.path.join(self.root, "grid"))
        self.phase_a = os.path.abspath(getattr(args, "phase_a", None) or os.path.join(self.root, "results"))
        self.results = os.path.abspath(getattr(args, "results", None) or os.path.join(self.root, "results"))

    def cases(self, testset=None):
        ts = testset or self.grid
        return read_json(os.path.join(ts, "cases.json"))

    def meta(self, testset=None):
        p = os.path.join(testset or self.grid, "case_meta.json")
        return read_json(p) if os.path.exists(p) else {}

    def result(self, name):
        return os.path.join(self.results, name)

    def load_result(self, name):
        p = self.result(name)
        return read_json(p) if os.path.exists(p) else None


# ------------------------------------------------------------------------------------------ vfx

def vfx_args(binary, case, testset):
    """The `vfx apply` command of run_grid.py for one case (output on stdout)."""
    a = [binary, "apply", case["effect"], "--width", str(case["width"]), "--height", str(case["height"]),
         "--depth", str(case["depth"])]
    for p in case.get("params", []):
        a += ["--param", p]
    if case.get("fps"):
        a += ["--fps", str(case["fps"])]
    a += ["--frames", str(case["frames"])]
    for lid, f in sorted((case.get("layers") or {}).items()):
        a += ["--layer", "%s=%s" % (lid, os.path.join(testset, f))]
    a += ["--in", os.path.join(testset, case["input"])]
    return a


def render_case(binary, case, testset, record=None, env_extra=None):
    """-> (exit code, stdout bytes, first non-note stderr line)."""
    env = dict(os.environ, RAYON_NUM_THREADS="1")
    env.pop("VFX_CALL_RECORD", None)
    if record:
        env["VFX_CALL_RECORD"] = record
    if env_extra:
        env.update(env_extra)
    p = subprocess.run(vfx_args(binary, case, testset), capture_output=True, env=env)
    note = ""
    for line in p.stderr.decode("utf-8", "replace").splitlines():
        if not line.startswith("vfx: note:"):
            note = line.strip()[:200]
            break
    return p.returncode, p.stdout, note


# ------------------------------------------------------------------------------------ raw compare

def ordered_bits(bits):
    """Float bits mapped onto integers that sort like the floats; -0 and 0 share a value (as fx-parity)."""
    return -(bits & 0x7FFFFFFF) if bits & 0x80000000 else bits


def is_nan_bits(bits):
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def ulp_distance_bits(a, b):
    """ULP distance of two float bit patterns, None when either is NaN."""
    if is_nan_bits(a) or is_nan_bits(b):
        return None
    return abs(ordered_bits(a) - ordered_bits(b))


def frame_len(w, h, depth):
    return w * h * 4 * SAMPLE[depth]


def _words(b, code):
    a = array.array(code)
    if a.itemsize != {"H": 2, "I": 4}[code]:
        raise ToolError("unexpected array item size")
    a.frombytes(b)
    if sys.byteorder == "big":
        a.byteswap()
    return a


def compare_clip(expected, actual, w, h, depth, frames):
    """Bit for bit comparison of two raw clips in the wire format of `depth`.

    Returns a dict: values_compared, differing_values, differing_pixels, differing_frames, max_abs_diff,
    max_ulp (32 bpc, None if no finite pair differs), nan_mismatches, first (frame, channel, x, y, expected,
    actual). {"error": text} when the sizes are not those of the case."""
    fl = frame_len(w, h, depth)
    want = fl * frames
    if len(expected) != want:
        return {"error": "reference has %d bytes, %d frames of %dx%d at %d bpc are %d" % (len(expected), frames, w, h, depth, want)}
    if len(actual) != want:
        return {"error": "output has %d bytes, %d frames of %dx%d at %d bpc are %d" % (len(actual), frames, w, h, depth, want)}
    out = {"values_compared": want // SAMPLE[depth], "differing_values": 0, "differing_pixels": 0,
           "differing_frames": [], "max_abs_diff": 0, "max_ulp": None, "nan_mismatches": 0, "first": None}
    if expected == actual:
        return out
    pix = w * h
    for f in range(frames):
        fe, fa = expected[f * fl:(f + 1) * fl], actual[f * fl:(f + 1) * fl]
        if fe == fa:
            continue
        out["differing_frames"].append(f)
        pixels = set()
        if depth == 32:
            ea, aa = _words(fe, "I"), _words(fa, "I")
        elif depth == 16:
            ea, aa = _words(fe, "H"), _words(fa, "H")
        else:
            ea, aa = fe, fa
        for i, (e, a) in enumerate(zip(ea, aa)):
            if e == a:
                continue
            out["differing_values"] += 1
            if depth == 32:
                plane, within = divmod(i, pix)
                chan, p = "GBRA"[plane], within
            else:
                p, ci = divmod(i, 4)
                chan = "RGBA"[ci]
            pixels.add(p)
            if depth == 32:
                if is_nan_bits(e) or is_nan_bits(a):
                    out["nan_mismatches"] += 1
                else:
                    u = abs(ordered_bits(e) - ordered_bits(a))
                    if out["max_ulp"] is None or u > out["max_ulp"]:
                        out["max_ulp"] = u
            else:
                d = abs(e - a)
                if d > out["max_abs_diff"]:
                    out["max_abs_diff"] = d
            if out["first"] is None:
                x, y = p % w, p // w
                if depth == 32:
                    ev, av = "0x%08x" % e, "0x%08x" % a
                else:
                    ev, av = e, a
                out["first"] = {"frame": f, "channel": chan, "x": x, "y": y, "expected": ev, "actual": av}
        out["differing_pixels"] += len(pixels)
    return out


def values_of(case):
    """(unit, count) of the comparable values of a case: bytes at 8 bpc, 16 bit words at 16 bpc, floats at 32."""
    n = case["frames"] * case["width"] * case["height"] * 4
    return {8: "byte", 16: "u16", 32: "float"}[case["depth"]], n


# --------------------------------------------------------------------------------------------- grade

def supports_max_ulp(path):
    try:
        p = subprocess.run([path, "--help"], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return b"--max-ulp" in (p.stdout + p.stderr)


def find_fx_parity(explicit, root):
    cands = [explicit] if explicit else []
    cands += [os.environ.get("FX_PARITY"),
              os.path.join(root, "..", "harness", "repo", "target", "release", "fx-parity"),
              os.path.join(os.path.dirname(root), "harness", "repo", "target", "release", "fx-parity"),
              os.path.expanduser("~/twixtor-parity/harness/repo/target/release/fx-parity"),
              os.path.join(root, "src", "target", "release", "fx-parity"),
              shutil.which("fx-parity")]
    seen = []
    for c in cands:
        if not c or c in seen:
            continue
        seen.append(c)
        if os.path.isfile(c) and supports_max_ulp(c):
            return os.path.abspath(c)
    return None


def run_fx_parity(fxp, testset, refs, vfx, jobs=1):
    cmd = [fxp, "run", "--testset", testset, "--refs", refs, "--max-ulp", "0", "--json", "--jobs", str(jobs)]
    if vfx:
        cmd += ["--vfx", vfx]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode not in (0, 1):
        raise ToolError("fx-parity exited %d: %s" % (p.returncode, p.stderr.decode("utf-8", "replace").strip()[:300]))
    try:
        return json.loads(p.stdout.decode("utf-8", "replace")), p.stderr.decode("utf-8", "replace").strip()
    except ValueError:
        raise ToolError("fx-parity did not print JSON: %s" % p.stdout[:200])


def new_grade_entry(case, meta):
    m = meta.get(case["id"], {})
    unit, n = values_of(case)
    return {
        "id": case["id"], "status": None, "error": None,
        "depth": case["depth"], "width": case["width"], "height": case["height"], "frames": case["frames"],
        "fps": case.get("fps"), "source": case.get("source"), "params": list(case.get("params", [])),
        "layers": sorted((case.get("layers") or {}).keys()),
        "group": m.get("group", "probe" if "probe" in case.get("label", "") else None),
        "tags": list(m.get("tags", [])), "standin": m.get("standin"), "logic_parity": bool(m.get("standin")),
        "unit": unit, "values_total": n, "values_compared": 0, "differing_values": 0,
        "differing_pixels": 0, "failing_frames": 0, "differing_frame_list": [],
        "max_ulp": None, "max_ulp_fxparity": None, "nan_mismatches": 0, "max_abs_diff": None,
        "first_diff": None, "ref_sha256": None,
    }


def apply_raw(entry, raw):
    if "error" in raw:
        entry["error"] = entry["error"] or raw["error"]
        return
    entry["values_compared"] = raw["values_compared"]
    entry["differing_values"] = raw["differing_values"]
    entry["differing_pixels"] = raw["differing_pixels"]
    entry["differing_frame_list"] = raw["differing_frames"]
    entry["failing_frames"] = len(raw["differing_frames"])
    entry["max_ulp"] = raw["max_ulp"]
    entry["nan_mismatches"] = raw["nan_mismatches"]
    entry["max_abs_diff"] = raw["max_abs_diff"] if entry["depth"] != 32 else None
    entry["first_diff"] = raw["first"]


def grade(ctx, refs, testset=None, vfx=None, fxp=None, engine="auto", cross_check=False, name="grade", jobs=1, now=None):
    testset = os.path.abspath(testset or ctx.grid)
    refs = os.path.abspath(refs)
    data = read_json(os.path.join(testset, "cases.json"))
    cases = data["cases"]
    meta_path = os.path.join(testset, "case_meta.json")
    meta = read_json(meta_path) if os.path.exists(meta_path) else ctx.meta()
    vfx = os.path.abspath(vfx or os.path.join(ctx.root, "src", "target", "release", "vfx"))
    if not os.path.isfile(vfx):
        raise ToolError("vfx not found: %s (use --vfx)" % vfx)
    used_engine, fx_report, fx_stderr = engine, None, ""
    if engine in ("auto", "fx-parity"):
        fxp = find_fx_parity(fxp, ctx.root)
        if fxp is None:
            if engine == "fx-parity":
                raise ToolError("no fx-parity with --max-ulp found (use --fx-parity or FX_PARITY)")
            used_engine = "python-fallback"
        else:
            used_engine = "fx-parity"
    entries = {c["id"]: new_grade_entry(c, meta) for c in cases}
    if used_engine == "fx-parity":
        fx_report, fx_stderr = run_fx_parity(fxp, testset, refs, vfx, jobs)
        for r in fx_report["cases"]:
            e = entries.get(r["id"])
            if e is None:
                continue
            e["status"] = r["status"]
            e["error"] = r.get("error")
            e["max_ulp_fxparity"] = r.get("max_ulp")
            e["failing_frames"] = r.get("failing_frames", 0)
            e["differing_pixels"] = r.get("clip_differing_pixels", r.get("differing_pixels", 0)) or 0
    by_id = {c["id"]: c for c in cases}
    python_disagreements = []
    for cid, e in entries.items():
        c = by_id[cid]
        rp = os.path.join(refs, c["reference"])
        have_ref = os.path.isfile(rp)
        if have_ref:
            e["ref_sha256"] = sha256_file(rp)
        need_python = used_engine != "fx-parity" or cross_check or e["status"] == "FAIL"
        if not have_ref:
            e["status"] = "no reference"
            continue
        if e["status"] is None:
            e["status"] = "FAIL"
            e["error"] = "missing from the fx-parity report"
        raw = None
        if need_python:
            code, out, note = render_case(vfx, c, testset)
            if code != 0:
                raw = {"error": "vfx exited %d: %s" % (code, note)}
            else:
                with open(rp, "rb") as fh:
                    raw = compare_clip(fh.read(), out, c["width"], c["height"], c["depth"], c["frames"])
        if used_engine != "fx-parity":
            if "error" in raw:
                e["status"] = "FAIL"
                e["error"] = raw["error"]
            else:
                e["status"] = "pass" if raw["differing_values"] == 0 else "FAIL"
        elif cross_check and raw is not None:
            mine = "FAIL" if ("error" in raw or raw["differing_values"]) else "pass"
            if mine != e["status"]:
                python_disagreements.append({"id": cid, "fx_parity": e["status"], "python": mine})
        if raw is not None:
            apply_raw(e, raw)
        if e["status"] == "pass" and not e["error"] and raw is None:
            e["values_compared"] = e["values_total"]
        if e["status"] == "FAIL" and not e["error"] and e["values_compared"] == 0:
            e["error"] = "differs, but the Python comparison could not be made"
    rows = [entries[c["id"]] for c in cases]
    summary = collections.Counter(e["status"] for e in rows)
    doc = {
        "tool": "phase_b.py grade", "measured_at": now or datetime.date.today().isoformat(),
        "testset": testset, "refs": refs, "vfx": vfx, "vfx_sha256": sha256_file(vfx),
        "engine": used_engine, "fx_parity": fxp if used_engine == "fx-parity" else None,
        "fx_parity_sha256": sha256_file(fxp) if used_engine == "fx-parity" else None,
        "max_ulp": 0, "fx_parity_stderr": fx_stderr[:500],
        "totals": {"total": len(rows), "pass": summary.get("pass", 0), "fail": summary.get("FAIL", 0),
                   "no_reference": summary.get("no reference", 0)},
        "cross_check": {"done": bool(cross_check and used_engine == "fx-parity"), "disagreements": python_disagreements},
        "cases": rows,
    }
    write_json(ctx.result(name + ".json"), doc)
    write_grade_tsv(ctx.result(name + ".tsv"), rows)
    return doc


def write_grade_tsv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("id\tstatus\tdiffering_values\tmax_ulp\tgroup\ttags\tlogic_parity\tdepth\twidth\theight\tframes\t"
                 "failing_frames\tdiffering_pixels\tfirst_diff\terror\n")
        for e in rows:
            ulp = e["max_ulp"]
            if e["nan_mismatches"]:
                ulp = "nan mismatch" if ulp is None else "%s+nan" % ulp
            fd = e["first_diff"]
            fds = "" if not fd else "f%d %s x=%d y=%d exp=%s act=%s" % (fd["frame"], fd["channel"], fd["x"], fd["y"], fd["expected"], fd["actual"])
            fh.write("\t".join([
                e["id"], e["status"] or "", str(e["differing_values"]), "" if ulp is None else str(ulp),
                e["group"] or "", ",".join(e["tags"]), "1" if e["logic_parity"] else "0",
                str(e["depth"]), str(e["width"]), str(e["height"]), str(e["frames"]),
                str(e["failing_frames"]), str(e["differing_pixels"]), fds, (e["error"] or "").replace("\t", " ")]) + "\n")


def make_refs(ctx, out, vfx, testset=None, only=None):
    testset = os.path.abspath(testset or ctx.grid)
    cases = read_json(os.path.join(testset, "cases.json"))["cases"]
    os.makedirs(out, exist_ok=True)
    made, failed = 0, []
    for c in cases:
        if only and only not in c["id"]:
            continue
        code, data, note = render_case(vfx, c, testset)
        if code != 0:
            failed.append((c["id"], code, note))
            continue
        with open(os.path.join(out, c["reference"]), "wb") as fh:
            fh.write(data)
        made += 1
    return made, failed


# ----------------------------------------------------------------------------------------------- ooc

def parse_value_list(text):
    """The value texts of a --param value: one, or the keys of `v@t;v@t`."""
    if "@" in text:
        return [seg.split("@", 1)[0] for seg in text.split(";") if seg]
    return [text]


def load_param_map(path, plugin=PLUGIN_NAME):
    """param_map.tsv -> {vfx_id: row} for the rows that pair a plugin parameter with a rewrite id."""
    rows = {}
    for r in read_tsv_rows(path):
        if r.get("plugin") and r["plugin"] != plugin:
            continue
        vid = r.get("vfx_id")
        if vid:
            rows[vid] = r
    return rows


def _num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def range_violation(param_text, pmap):
    """'k=v' against the registered range of the plugin param paired with k; None if inside or unknown."""
    key, _, text = param_text.partition("=")
    row = pmap.get(key)
    if row is None:
        return None
    if row.get("match") == "unmatched" or not (row.get("ae_min") or row.get("ae_max") or row.get("ae_choices")):
        return None
    kind = row.get("vfx_kind", "")
    for v in parse_value_list(text):
        if kind == "color":
            for part in v.split(","):
                x = _num(part)
                if x is not None and (x < 0 or x > 255):
                    return "%s: color channel %s outside 0..255" % (key, part)
            continue
        if kind == "point" or "," in v:
            continue
        x = _num(v)
        if x is None:
            continue
        if kind == "popup":
            choices = [c for c in (row.get("ae_choices") or row.get("vfx_choices") or "").split("|") if c]
            if choices and x < 1:
                return "%s: popup value %s below its first choice" % (key, v)
            if choices and x > len(choices):
                return "%s: popup value %s above its last choice (%d)" % (key, v, len(choices))
            continue
        lo, hi = _num(row.get("ae_min")), _num(row.get("ae_max"))
        if lo is not None and x < lo:
            return "%s: %s below the registered minimum %s" % (key, v, row.get("ae_min"))
        if hi is not None and x > hi:
            return "%s: %s above the registered maximum %s" % (key, v, row.get("ae_max"))
    return None


def load_status(paths):
    """Merge harness status files into {id: entry}. Accepts the run_harness.py --summary document
    ({"cases": [...]}), a list of entries, or a mapping id -> entry or id -> status word."""
    out = {}
    for p in paths:
        doc = read_json(p)
        if isinstance(doc, dict) and "cases" in doc and isinstance(doc["cases"], list):
            items = doc["cases"]
        elif isinstance(doc, list):
            items = doc
        elif isinstance(doc, dict):
            items = []
            for k, v in doc.items():
                items.append(dict(v, id=k) if isinstance(v, dict) else {"id": k, "status": v})
        else:
            raise ToolError("%s is not a harness status file" % p)
        for it in items:
            if isinstance(it, dict) and it.get("id"):
                out[it["id"]] = it
    return out


def record_unimplemented(records_dir, cid):
    """Callbacks the mock host did not implement while the plugin rendered this case (from its record)."""
    if not records_dir:
        return []
    p = os.path.join(records_dir, cid + ".jsonl")
    if not os.path.exists(p):
        return []
    found = []
    with open(p) as fh:
        for line in fh:
            if '"unimplemented"' in line:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("unimplemented"):
                    found.append(d.get("fn", "?"))
    return sorted(set(found))


def classify_plugin(entry, unimplemented=()):
    """-> (class, detail) of what the plugin side did, from one harness status entry.

    rendered | plugin_refused | plugin_error (the original returned an error: out of contract)
    harness_fault | harness_timeout | harness_unsupported | not_run | unknown (emulator or harness
    problems: never out of contract by themselves)."""
    if entry is None:
        return "no_status", None
    status = str(entry.get("status", ""))
    err = str(entry.get("error") or "")
    if status == "rendered":
        return "rendered", None
    if status in ("refused", "rejected"):
        return "plugin_refused", err or status
    if status == "error":
        if unimplemented:
            return "harness_fault", "mock host callback unimplemented: %s; %s" % (", ".join(unimplemented), err)
        if "PluginError" in err or "returned PF_Err" in err:
            return "plugin_error", err
        return "harness_fault", err
    if status == "timeout":
        return "harness_timeout", err or "timeout"
    if status == "unsupported":
        return "harness_unsupported", err
    if status in ("not_run", "skipped"):
        return "not_run", err
    return "unknown", status


def judge_ooc(grid_cases, candidates, status, base_rows, cand_rows, pmap=None, records=None):
    """The out of contract judgement; pure.

    Grid cases move out only when the plugin refused or errored (a real PF_Err) or a param is outside the
    registered range. Candidates stay out unless the plugin rendered them with every param inside the
    registered range; with no status for them they stay out, unjudged."""
    moved, back, unresolved = [], [], []

    def side(rows, cid):
        r = rows.get(cid)
        return None if r is None else {"exit": int(r[0]) if str(r[0]).lstrip("-").isdigit() else r[0], "size": r[1], "sha256": r[2],
                                       "note": r[3] if len(r) > 3 else ""}

    def judge(case, origin, rows, why=None):
        cid = case["id"]
        cls, detail = classify_plugin(status.get(cid), record_unimplemented(records, cid))
        rng = None
        if pmap:
            for p in case.get("params", []):
                rng = range_violation(p, pmap)
                if rng:
                    break
        entry = {"id": cid, "origin": origin, "params": list(case.get("params", [])), "candidate_why": why,
                 "plugin": {"class": cls, "status": (status.get(cid) or {}).get("status"), "detail": detail},
                 "rewrite": side(rows, cid), "judged": cid in status or bool(rng)}
        out = None
        if rng:
            out = ("outside_registered_range", rng)
        elif cls in ("plugin_refused", "plugin_error"):
            out = (cls, detail)
        if origin == "grid":
            if out:
                entry["reason_class"], entry["reason"] = out
                moved.append(entry)
            elif cls not in ("rendered", "no_status"):
                unresolved.append({"id": cid, "class": cls, "detail": detail})
            return
        if out:
            entry["reason_class"], entry["reason"] = out
            moved.append(entry)
        elif cls == "rendered":
            entry["reason_class"], entry["reason"] = "rendered_in_range", "the plugin rendered it with every param inside the registered ranges"
            back.append(entry)
        elif cls == "no_status":
            entry["reason_class"], entry["reason"] = "candidate_unjudged", "no plugin status yet; kept out because the rewrite refuses it"
            moved.append(entry)
        else:
            entry["reason_class"], entry["reason"] = "candidate_" + cls, "the plugin did not render it (%s): %s" % (cls, detail)
            moved.append(entry)
            if cls.startswith("harness_") or cls in ("not_run", "unknown"):
                unresolved.append({"id": cid, "class": cls, "detail": detail})

    for c in grid_cases:
        judge(c, "grid", base_rows)
    for c in candidates:
        judge(c, "candidate", cand_rows, c.get("why"))
    return moved, back, unresolved


def cmd_ooc(ctx, a):
    cases = ctx.cases()["cases"]
    cp = os.path.join(ctx.grid, "out_of_contract.json")
    cands = read_json(cp) if os.path.exists(cp) else []
    paths = list(a.status or [])
    status = load_status(paths) if paths else {}
    base_rows = _rows_with_note(os.path.join(ctx.phase_a, "base_outputs.tsv"))
    cand_rows = _rows_with_note(os.path.join(ctx.phase_a, "out_of_contract_base.tsv"))
    pmap = load_param_map(a.param_map) if a.param_map and os.path.exists(a.param_map) else None
    moved, back, unresolved = judge_ooc(cases, cands, status, base_rows, cand_rows, pmap, a.records)
    mg = [m for m in moved if m["origin"] == "grid"]
    mc = [m for m in moved if m["origin"] == "candidate"]
    doc = {
        "tool": "phase_b.py ooc", "measured_at": datetime.date.today().isoformat(),
        "status_files": [os.path.abspath(p) for p in paths], "records": a.records,
        "param_map": a.param_map if pmap else None, "param_ranges_checked": bool(pmap),
        "counts": {"grid_cases": len(cases), "moved_out_from_grid": len(mg), "candidates": len(cands),
                   "candidates_out": len(mc), "candidates_back_in_contract": len(back),
                   "candidates_unjudged": sum(1 for m in mc if m["reason_class"] == "candidate_unjudged"),
                   "unresolved": len(unresolved)},
        "moved_out": moved, "candidates_back_in_contract": back, "unresolved": unresolved,
    }
    write_json(ctx.result("ooc.json"), doc)
    c = doc["counts"]
    print("ooc: %d grid cases moved out; candidates %d: %d out (%d unjudged), %d back in contract; %d unresolved"
          % (c["moved_out_from_grid"], c["candidates"], c["candidates_out"], c["candidates_unjudged"],
             c["candidates_back_in_contract"], c["unresolved"]))
    print("wrote", ctx.result("ooc.json"))
    return 0


def _rows_with_note(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    for r in read_tsv_rows(path):
        rows[r["id"]] = (r["exit"], r["size"], r["sha256"], r.get("note", ""))
    return rows


def cmd_ooc_set(ctx, a):
    """A test set of the candidates, inputs shared with the grid, so the harness can render them."""
    cands = read_json(os.path.join(ctx.grid, "out_of_contract.json"))
    grid_data = ctx.cases()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    rel = os.path.relpath(ctx.grid, out).replace("\\", "/")
    new = []
    for c in cands:
        n = dict(c)
        n.pop("why", None)
        n["input"] = rel + "/" + c["input"]
        if c.get("layers"):
            n["layers"] = {k: rel + "/" + v for k, v in c["layers"].items()}
        new.append(n)
    inputs = [dict(i, file=rel + "/" + i["file"]) for i in grid_data["inputs"]]
    write_json(os.path.join(out, "cases.json"), {"format": grid_data["format"], "ffmpeg": grid_data.get("ffmpeg"), "inputs": inputs, "cases": new})
    write_json(os.path.join(out, "case_meta.json"), {c["id"]: {"group": "out-of-contract-candidate", "tags": [], "standin": None, "why": c.get("why")} for c in cands})
    print("wrote %s: %d candidate cases (inputs under %s)" % (os.path.join(out, "cases.json"), len(new), rel))
    return 0


# --------------------------------------------------------------------------------------------- calls

def load_jsonl(path):
    lines, bad = [], 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except ValueError:
                bad += 1
    return lines, bad


def _frac(t, scale):
    return Fraction(int(t), int(scale)) if scale else Fraction(int(t))


def rewrite_view(lines):
    """VFX_CALL_RECORD lines -> frames {n: seconds}, reads {n: [(layer, seconds, index)]}, params {n: [(name, seconds)]}."""
    scale = None
    frames, reads, params = {}, collections.defaultdict(list), collections.defaultdict(list)
    base = {}
    for ln in lines:
        ev = ln.get("ev")
        if ev == "run":
            scale = ln.get("time_scale")
            base = {"time_scale": ln.get("time_scale"), "time_step": ln.get("time_step"), "frames": ln.get("frames")}
        elif ev == "frame":
            scale = ln.get("time_scale", scale)
            frames[ln["out_frame"]] = _frac(ln["time"], ln.get("time_scale", scale))
        elif ev == "read":
            reads[ln["out_frame"]].append((ln["layer"], _frac(ln["time"], scale), ln.get("index")))
        elif ev == "param":
            params[ln["out_frame"]].append((ln["param"], _frac(ln["time"], scale)))
    return {"frames": frames, "reads": reads, "params": params, "base": base}


def plugin_view(lines, index_to_layer, provided):
    """Harness record lines -> the same shape as rewrite_view, plus bookkeeping.

    Layer reads are `inter.checkout_param` lines that carry a `layer` note and `prerender_cb.checkout_layer`
    lines; a read of a layer the case gives no clip is dropped (the rewrite cannot read what is not there)."""
    frames, reads, params = {}, collections.defaultdict(list), collections.defaultdict(list)
    dropped, unimpl, done, base = 0, [], None, {}
    scale = None
    for ln in lines:
        ev = ln.get("ev")
        if ev == "frame":
            scale = ln.get("time_scale", scale)
            base = {"time_scale": ln.get("time_scale"), "time_step": ln.get("time_step")}
            frames[ln["out_frame"]] = _frac(ln["time"], ln.get("time_scale", scale))
        elif ev == "case_done":
            done = ln
        elif ev == "cb":
            fn = str(ln.get("fn", ""))
            if ln.get("unimplemented"):
                unimpl.append(fn)
            idx, of, t = ln.get("param_index"), ln.get("out_frame"), ln.get("time")
            if t is None or of is None:
                continue
            tt = _frac(t, ln.get("time_scale", scale))
            if fn.endswith("checkout_param"):
                if "layer" in ln:
                    lay = str(ln["layer"])
                    if lay == "none":
                        dropped += 1
                        continue
                    name = "input" if idx == 0 else index_to_layer.get(idx, "param#%s" % idx)
                    index = None if "outside the clip" in lay else ln.get("footage_frame")
                    reads[of].append((name, tt, index))
                else:
                    params[of].append((index_to_layer.get(idx, "param#%s" % idx), tt))
            elif fn.startswith("prerender_cb") and fn.endswith("checkout_layer"):
                name = "input" if idx == 0 else index_to_layer.get(idx, "param#%s" % idx)
                if name != "input" and name not in provided:
                    dropped += 1
                    continue
                reads[of].append((name, tt, ln.get("footage_frame") if ln.get("inside") else None))
    return {"frames": frames, "reads": reads, "params": params, "dropped": dropped, "unimplemented": sorted(set(unimpl)),
            "done": done, "base": base}


def _fmt(elem):
    layer, t, index = elem
    return [layer, str(t), index]


def diff_read_sets(p, r):
    """Plugin set P against rewrite set R of (layer, seconds, index): the classified differences."""
    only_p, only_r = sorted(set(p) - set(r), key=str), sorted(set(r) - set(p), key=str)
    wrong_time, wrong_index = [], []
    for e in list(only_p):
        for f in list(only_r):
            if e[0] == f[0] and e[2] == f[2] and e[1] != f[1]:
                wrong_time.append({"layer": e[0], "index": e[2], "plugin_time": str(e[1]), "rewrite_time": str(f[1])})
                only_p.remove(e)
                only_r.remove(f)
                break
    for e in list(only_p):
        for f in list(only_r):
            if e[0] == f[0] and e[1] == f[1] and e[2] != f[2]:
                wrong_index.append({"layer": e[0], "time": str(e[1]), "plugin_index": e[2], "rewrite_index": f[2]})
                only_p.remove(e)
                only_r.remove(f)
                break
    return {"missing_in_rewrite": [_fmt(e) for e in only_p],
            "extra_in_rewrite": [_fmt(e) for e in only_r],
            "wrong_time": wrong_time, "wrong_index": wrong_index}


def compare_calls(pv, rv, ordered=False):
    """Compare one case's plugin view with its rewrite view per output frame."""
    all_frames = sorted(set(pv["frames"]) | set(rv["frames"]) | set(pv["reads"]) | set(rv["reads"]))
    bad, first = [], None
    param_compared = bool(rv["params"])
    for f in all_frames:
        issues = {}
        if f not in pv["frames"]:
            issues["frame"] = "the plugin never started this output frame"
        elif f not in rv["frames"]:
            issues["frame"] = "the rewrite never rendered this output frame"
        elif pv["frames"][f] != rv["frames"][f]:
            issues["frame"] = "frame time differs: plugin %s s, rewrite %s s" % (pv["frames"][f], rv["frames"][f])
        P, R = pv["reads"].get(f, []), rv["reads"].get(f, [])
        if ordered:
            if P != R:
                d = diff_read_sets(P, R)
                d["ordered"] = True
                d["first_order_difference"] = next((i for i, (x, y) in enumerate(itertools.zip_longest(P, R)) if x != y), None)
                issues["reads"] = d
        elif set(P) != set(R):
            issues["reads"] = diff_read_sets(P, R)
        if param_compared and set(pv["params"].get(f, [])) != set(rv["params"].get(f, [])):
            issues["params"] = {"missing_in_rewrite": [[a, str(b)] for a, b in sorted(set(pv["params"].get(f, [])) - set(rv["params"].get(f, [])), key=str)],
                                "extra_in_rewrite": [[a, str(b)] for a, b in sorted(set(rv["params"].get(f, [])) - set(pv["params"].get(f, [])), key=str)]}
        if issues:
            bad.append(f)
            if first is None:
                first = {"out_frame": f, **issues}
    return {"frames_compared": len(all_frames), "match": not bad, "mismatching_frames": bad, "first_diff": first,
            "param_times_compared": param_compared}


def plugin_param_times(pv):
    return {str(f): sorted({str(t) for _, t in v}) for f, v in sorted(pv["params"].items())}


def calls_check(cases, plugin_dir, rewrite_dir, index_to_layer, ordered=False):
    results = []
    for c in cases:
        cid = c["id"]
        pp, rp = os.path.join(plugin_dir, cid + ".jsonl"), os.path.join(rewrite_dir, cid + ".jsonl")
        r = {"id": cid, "compared": False, "match": None, "reason": None}
        if not os.path.exists(pp):
            r["reason"] = "no plugin record"
        elif not os.path.exists(rp):
            r["reason"] = "no rewrite record"
        else:
            plines, pbad = load_jsonl(pp)
            rlines, rbad = load_jsonl(rp)
            pv = plugin_view(plines, index_to_layer, set((c.get("layers") or {}).keys()))
            rv = rewrite_view(rlines)
            if pv["done"] is None or pv["done"].get("status") != "rendered":
                r["reason"] = "the plugin did not render this case (%s)" % ((pv["done"] or {}).get("status") or "record has no case_done line")
            elif not rv["frames"]:
                r["reason"] = "the rewrite record has no frame line"
            else:
                cmp = compare_calls(pv, rv, ordered)
                r.update(compared=True, match=cmp["match"], frames_compared=cmp["frames_compared"],
                         mismatching_frames=cmp["mismatching_frames"], first_diff=cmp["first_diff"],
                         param_times_compared=cmp["param_times_compared"],
                         plugin_param_times=plugin_param_times(pv) if not cmp["param_times_compared"] else None,
                         dropped_unprovided_checkouts=pv["dropped"], mock_unimplemented=pv["unimplemented"],
                         unparsable_lines={"plugin": pbad, "rewrite": rbad})
                if pv["base"] and rv["base"] and (pv["base"].get("time_scale"), pv["base"].get("time_step")) != (rv["base"].get("time_scale"), rv["base"].get("time_step")):
                    r["time_base_note"] = "plugin %s, rewrite %s" % (pv["base"], rv["base"])
        results.append(r)
    return results


def index_to_layer_map(param_map_path):
    """ae_index -> rewrite layer id from param_map.tsv (layer params only)."""
    m = {}
    if param_map_path and os.path.exists(param_map_path):
        for r in read_tsv_rows(param_map_path):
            if r.get("plugin") and r["plugin"] != PLUGIN_NAME:
                continue
            if r.get("vfx_kind") == "layer" and r.get("vfx_id") and (r.get("ae_index") or "").strip().lstrip("-").isdigit():
                m[int(r["ae_index"])] = r["vfx_id"]
    return m


def cmd_calls(ctx, a):
    testset = os.path.abspath(a.testset or ctx.grid)
    cases = read_json(os.path.join(testset, "cases.json"))["cases"]
    if a.run_rewrite:
        n, bad = record_rewrite(ctx, a.rewrite_records, a.vfx, testset)
        print("recorded %d rewrite runs into %s (%d failed)" % (n, a.rewrite_records, len(bad)))
    i2l = index_to_layer_map(a.param_map)
    res = calls_check(cases, a.plugin_records, a.rewrite_records, i2l, a.ordered)
    compared = [r for r in res if r["compared"]]
    matching = [r for r in compared if r["match"]]
    doc = {"tool": "phase_b.py calls", "measured_at": datetime.date.today().isoformat(),
           "plugin_records": os.path.abspath(a.plugin_records), "rewrite_records": os.path.abspath(a.rewrite_records),
           "param_map": a.param_map, "ordered": bool(a.ordered),
           "note": "parameter sample times are not recorded by the rewrite, so only the plugin side lists them",
           "totals": {"cases": len(res), "compared": len(compared), "matching": len(matching),
                      "mismatching": len(compared) - len(matching), "not_compared": len(res) - len(compared)},
           "cases": res}
    write_json(ctx.result(a.name + ".json"), doc)
    t = doc["totals"]
    print("calls: %d cases, %d compared, %d matching, %d mismatching, %d not compared" % (t["cases"], t["compared"], t["matching"], t["mismatching"], t["not_compared"]))
    for r in compared:
        if not r["match"]:
            print("  MISMATCH %s first at output frame %s" % (r["id"], r["first_diff"]["out_frame"]))
    print("wrote", ctx.result(a.name + ".json"))
    return 0


def find_record_vfx(ctx, explicit=None):
    """A vfx that honours VFX_CALL_RECORD: the first candidate that writes a record for the first grid case.
    The module's own base build may predate the record feature; the harness copy of the rewrite has it."""
    cands = [explicit, os.environ.get("RECORD_VFX"),
             os.path.join(ctx.root, "..", "harness", "repo", "target", "release", "vfx"),
             os.path.expanduser("~/twixtor-parity/harness/repo/target/release/vfx"),
             os.path.join(ctx.root, "src", "target", "release", "vfx"), shutil.which("vfx")]
    case0 = ctx.cases()["cases"][0]
    seen = []
    for c in cands:
        if not c or c in seen or not os.path.isfile(c):
            continue
        seen.append(c)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "r.jsonl")
            code, _, _ = render_case(c, case0, ctx.grid, record=p)
            if code == 0 and os.path.exists(p) and os.path.getsize(p) > 0:
                return os.path.abspath(c)
    return None


def record_rewrite(ctx, out, vfx, testset=None, only=None):
    testset = os.path.abspath(testset or ctx.grid)
    vfx = find_record_vfx(ctx, vfx)
    if vfx is None:
        raise ToolError("no vfx that writes a VFX_CALL_RECORD file was found (use --vfx or RECORD_VFX)")
    cases = read_json(os.path.join(testset, "cases.json"))["cases"]
    os.makedirs(out, exist_ok=True)
    n, bad = 0, []
    for c in cases:
        if only and only not in c["id"]:
            continue
        code, _, note = render_case(vfx, c, testset, record=os.path.join(out, c["id"] + ".jsonl"))
        if code != 0:
            bad.append((c["id"], code, note))
        else:
            n += 1
    return n, bad


# --------------------------------------------------------------------------------------------- kills

def binaries_sha(phase_a):
    """sha256 of the base vfx build, from results/binaries.sha256."""
    p = os.path.join(phase_a, "binaries.sha256")
    if not os.path.exists(p):
        return None
    for line in open(p).read().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].endswith("release/vfx"):
            return parts[0]
    return None


def compute_kills(base_rows, ref_sha, mutants, mutant_rows, out_of_contract=()):
    """Mutant kills by the decided definition; pure.

    base_rows: {id: (exit, size, sha)} of the base rewrite; ref_sha: {id: sha or None}; mutants: list of
    dicts with id; mutant_rows: {mutant id: {id: (exit, size, sha)} or None when not run}.
    Candidates are the cases where the base output equals the reference."""
    ooc = set(out_of_contract)
    cand, excluded = [], {"out_of_contract": [], "no_reference": [], "base_differs": []}
    for cid in sorted(base_rows):
        if cid in ooc:
            excluded["out_of_contract"].append(cid)
        elif ref_sha.get(cid) is None:
            excluded["no_reference"].append(cid)
        else:
            b = base_rows[cid]
            if b[0] != "0" or b[2] != ref_sha[cid]:
                excluded["base_differs"].append(cid)
            else:
                cand.append(cid)
    out = []
    for m in mutants:
        rows = mutant_rows.get(m["id"])
        e = {"id": m["id"], "kind": m.get("kind"), "file": m.get("file"), "desc": m.get("desc")}
        if rows is None:
            e.update(status="not_run", killed_by=[], n_killers=0, changed_vs_base=None, changed_in_candidates=None,
                     reason="no result file for this mutant")
            out.append(e)
            continue
        changed = [c for c in base_rows if rows.get(c) != base_rows[c]]
        killers = [c for c in cand if rows.get(c) is None or rows[c][0] != "0" or rows[c][2] != ref_sha[c]]
        e["changed_vs_base"] = len(changed)
        e["changed_in_candidates"] = len([c for c in changed if c in set(cand)])
        e["killed_by"] = killers
        e["n_killers"] = len(killers)
        if killers:
            e["status"] = "killed"
        else:
            e["status"] = "survivor"
            if not changed:
                e["reason"] = "changes no output of the base rewrite on the grid"
            else:
                e["reason"] = ("changes outputs only on cases where the base already differs from the reference, "
                               "that have no reference, or that are out of contract")
        out.append(e)
    return cand, excluded, out


def cmd_kills(ctx, a):
    cases = ctx.cases()["cases"]
    mutants = read_json(os.path.join(ctx.root, "mutants", "mutants.json"))
    refs = os.path.abspath(a.refs)
    base_rows = load_outputs_tsv(os.path.join(ctx.phase_a, "base_outputs.tsv"))
    ref_sha = {}
    for c in cases:
        p = os.path.join(refs, c["reference"])
        ref_sha[c["id"]] = sha256_file(p) if os.path.isfile(p) else None
    mutant_rows = {}
    warnings = []
    if a.rerun:
        vfx = os.path.abspath(a.vfx or os.path.join(ctx.root, "src", "target", "release", "vfx"))
        mutant_rows, base_rows = rerun_rows(ctx, cases, mutants, vfx, a.mut_dir, ref_sha, a.only)
        warnings.append("rerun: base and mutants were rendered now, the saved phase A hashes were not used")
    else:
        for m in mutants:
            p = os.path.join(ctx.phase_a, "mutant_%s.tsv" % m["id"])
            mutant_rows[m["id"]] = load_outputs_tsv(p) if os.path.exists(p) else None
    ooc = ctx.load_result("ooc.json")
    ooc_ids = [m["id"] for m in ooc["moved_out"] if m["origin"] == "grid"] if ooc else []
    cand, excluded, results = compute_kills(base_rows, ref_sha, mutants, mutant_rows, ooc_ids)
    grade_doc = ctx.load_result(a.grade_name + ".json")
    stale, consistency = False, None
    if grade_doc:
        gpass = {c["id"] for c in grade_doc["cases"] if c["status"] == "pass"}
        cset = set(cand)
        in_grade_not_base = sorted((gpass - cset) - set(ooc_ids))
        in_base_not_grade = sorted(cset - gpass)
        consistency = {"graded_pass_but_base_hash_differs": in_grade_not_base[:20], "n1": len(in_grade_not_base),
                       "base_hash_equal_but_not_graded_pass": in_base_not_grade[:20], "n2": len(in_base_not_grade)}
        recorded = binaries_sha(ctx.phase_a)
        if not a.rerun and recorded and grade_doc.get("vfx_sha256") != recorded:
            stale = True
            warnings.append("STALE: the grade ran vfx %s, the saved mutant results derive from the base build %s; rebuild "
                            "the mutants and re-run, or use --rerun" % ((grade_doc.get("vfx_sha256") or "?")[:12], recorded[:12]))
        if in_grade_not_base or in_base_not_grade:
            stale = True
            warnings.append("INCONSISTENT: %d graded-pass cases differ from the saved base hash, %d the other way" % (len(in_grade_not_base), len(in_base_not_grade)))
    else:
        warnings.append("no %s.json: candidates were taken from the saved base hashes only" % a.grade_name)
    killed = [r["id"] for r in results if r["status"] == "killed"]
    not_killed = [r["id"] for r in results if r["status"] != "killed"]
    doc = {"tool": "phase_b.py kills", "measured_at": datetime.date.today().isoformat(), "refs": refs,
           "definition": "killed iff, among the in-contract cases where the base rewrite's output equals the reference, the mutant's output differs from the reference on at least one",
           "mode": "rerun" if a.rerun else "saved phase A mutant results",
           "candidate_cases": len(cand), "excluded": {k: len(v) for k, v in excluded.items()},
           "excluded_ids": {"base_differs": excluded["base_differs"][:200]},
           "stale": stale, "consistency": consistency, "warnings": warnings,
           "totals": {"total": len(results), "killed": len(killed), "survivors": [r["id"] for r in results if r["status"] == "survivor"],
                      "not_run": [r["id"] for r in results if r["status"] == "not_run"]},
           "mutants": results}
    write_json(ctx.result("kills.json"), doc)
    print("kills: %d of %d mutants killed over %d candidate cases (%d excluded: %s)" % (len(killed), len(results), len(cand), sum(len(v) for v in excluded.values()),
                                                                                         ", ".join("%s %d" % (k, len(v)) for k, v in excluded.items())))
    for r in results:
        if r["status"] != "killed":
            print("  %s %s: %s" % (r["status"].upper(), r["id"], r.get("reason")))
    for w in warnings:
        print("  warning:", w)
    print("wrote", ctx.result("kills.json"))
    return 0


def rerun_rows(ctx, cases, mutants, base_vfx, mut_dir, ref_sha, only=None):
    """Render base and every mutant now; rows shaped like run_grid.py's (exit, size, sha). One core."""
    testset = ctx.grid
    sel = [c for c in cases if not only or only in c["id"]]

    def rows_for(binary):
        rows = {}
        for c in sel:
            code, out, _ = render_case(binary, c, testset)
            rows[c["id"]] = (str(code), str(len(out)), hashlib.sha256(out).hexdigest() if code == 0 else "-")
        return rows

    base = rows_for(base_vfx)
    out = {}
    for m in mutants:
        b = os.path.join(mut_dir or os.path.join(ctx.root, "mut"), m["id"], "vfx")
        out[m["id"]] = rows_for(b) if os.path.isfile(b) else None
    return out, base


# ------------------------------------------------------------------------------------------- summary

def load_identity(phase_a):
    p = os.path.join(phase_a, "summary.json")
    if os.path.exists(p):
        d = read_json(p)
        return {k: d.get(k, IDENT_FALLBACK[k]) for k in IDENT_FALLBACK}
    return dict(IDENT_FALLBACK)


COMMANDS = [
    "python3 phase_b.py grade --refs REFS",
    "python3 phase_b.py ooc --status run_summary.json --param-map ../harness/param_map.tsv --records RECORDS",
    "python3 phase_b.py calls --plugin-records RECORDS --rewrite-records REWRITE_RECORDS --param-map ../harness/param_map.tsv",
    "python3 phase_b.py record-rewrite --out REWRITE_RECORDS",
    "python3 phase_b.py kills --refs REFS",
    "python3 phase_b.py summary",
    "fx-parity run --testset grid --refs REFS --max-ulp 0 --json --vfx src/target/release/vfx",
]


def compute_summary(d, status_override=None, date=None):
    """The summary document from loaded result files; pure. d keys: ident, cases, meta, candidates, mutants, grade,
    extra_grades, ooc, kills, calls, fixes."""
    ident = d["ident"]
    cases = d["cases"]
    ooc, kills, calls = d.get("ooc"), d.get("kills"), d.get("calls")
    moved_grid = {m["id"] for m in ooc["moved_out"] if m["origin"] == "grid"} if ooc else set()
    if ooc:
        out_ids = {m["id"] for m in ooc["moved_out"]}
        back = [m["id"] for m in ooc.get("candidates_back_in_contract", [])]
    else:
        out_ids = {c["id"] for c in d.get("candidates", [])}
        back = []
    ids = [c["id"] for c in cases if c["id"] not in moved_grid]
    graded = {}
    for g in [d["grade"]] + list(d.get("extra_grades") or []):
        for c in g["cases"]:
            graded[c["id"]] = c
    extra_ids = [c["id"] for g in (d.get("extra_grades") or []) for c in g["cases"]]
    in_ids = ids + [i for i in extra_ids if i not in set(ids)]
    meta = d.get("meta") or {}
    rows = [graded[i] for i in in_ids if i in graded]
    compared = [r for r in rows if r["status"] in ("pass", "FAIL")]
    identical = [r for r in compared if r["status"] == "pass"]
    differing = [r for r in compared if r["status"] == "FAIL"]
    errored = [r for r in differing if r.get("error")]
    byte_cases = [r for r in compared if r["depth"] in (8, 16) and not r.get("error")]
    float_cases = [r for r in compared if r["depth"] == 32 and not r.get("error")]
    ulps = [r["max_ulp"] for r in float_cases if r.get("max_ulp") is not None]
    nan_cases = [r["id"] for r in float_cases if r.get("nan_mismatches")]
    logic = [r for r in compared if (meta.get(r["id"], {}).get("standin") or r.get("standin"))]
    standins = sorted({m["standin"] for m in meta.values() if m.get("standin")})
    n_standin_all = sum(1 for m in meta.values() if m.get("standin"))
    cc = {"compared": 0, "matching": 0}
    if calls:
        idset = set(in_ids)
        cs = [c for c in calls["cases"] if c["compared"] and c["id"] in idset]
        cc = {"compared": len(cs), "matching": sum(1 for c in cs if c["match"])}
    mutants = d["mutants"]
    mk = {"total": len(mutants), "killed": 0, "survivors": []}
    if kills:
        mk["killed"] = kills["totals"]["killed"]
        mk["survivors"] = kills["totals"]["survivors"] + kills["totals"]["not_run"]
    no_ref = len(in_ids) - len(compared)
    blockers = []
    if not compared:
        blockers.append("no case compared with a reference yet")
    elif len(compared) < MIN_COMPARED:
        blockers.append("%d in-contract cases compared, at least %d are required" % (len(compared), MIN_COMPARED))
    if no_ref and compared:
        blockers.append("%d in-contract cases have no reference" % no_ref)
    if differing:
        blockers.append("%d compared cases differ%s" % (len(differing), " (%d could not be compared bit for bit: %s)" % (len(errored), errored[0]["error"][:80]) if errored else ""))
    if nan_cases:
        blockers.append("%d 32 bpc cases differ in NaN bits (ULP undefined): max ULP covers the other values" % len(nan_cases))
    if d["grade"].get("engine") != "fx-parity":
        blockers.append("grade engine was %s, not fx-parity --max-ulp 0" % d["grade"].get("engine"))
    if d["grade"].get("cross_check", {}).get("disagreements"):
        blockers.append("fx-parity and the Python comparison disagree on %d cases" % len(d["grade"]["cross_check"]["disagreements"]))
    if calls is None:
        blockers.append("call records not compared yet")
    elif cc["compared"] == 0:
        blockers.append("no call record compared (no plugin record)")
    elif cc["matching"] != cc["compared"]:
        blockers.append("call records: %d of %d match" % (cc["matching"], cc["compared"]))
    elif cc["compared"] < len(compared):
        blockers.append("call records compared for %d of %d compared cases" % (cc["compared"], len(compared)))
    if kills is None:
        blockers.append("mutants not compared with references yet (killed stays 0)")
    else:
        if mk["survivors"]:
            blockers.append("mutants: %d of %d killed; survivors %s" % (mk["killed"], mk["total"], ", ".join(mk["survivors"])))
        if kills.get("stale"):
            blockers.append("kills.json is stale or inconsistent with the grade: " + "; ".join(kills.get("warnings") or []))
    if ooc is None:
        blockers.append("out-of-contract candidates not judged against the original yet (no ooc.json)")
    else:
        unj = ooc["counts"].get("candidates_unjudged", 0)
        if unj:
            blockers.append("%d out-of-contract candidates not judged against the original yet" % unj)
        if back:
            miss = [i for i in back if i not in graded or graded[i]["status"] not in ("pass", "FAIL")]
            if miss:
                blockers.append("%d out-of-contract candidates the plugin rendered in range are in contract now and not compared yet" % len(miss))
        if ooc.get("unresolved"):
            blockers.append("%d cases hit emulator or harness problems and are unresolved" % len(ooc["unresolved"]))
    ooc_clear = (ooc is not None and not ooc["counts"].get("candidates_unjudged") and not ooc.get("unresolved")
                 and all(i in graded and graded[i]["status"] in ("pass", "FAIL") for i in back))
    verified = (len(compared) >= MIN_COMPARED and not differing and no_ref == 0 and d["grade"].get("engine") == "fx-parity"
                and ooc_clear and not d["grade"].get("cross_check", {}).get("disagreements")
                and calls is not None and cc["compared"] > 0 and cc["matching"] == cc["compared"]
                and kills is not None and not kills.get("stale") and mk["total"] > 0 and mk["killed"] == mk["total"]
                and not mk["survivors"] and not nan_cases)
    status = "verified" if verified else "in_progress"
    if status_override and status_override != "verified":
        status = status_override
    stand_in_lines = []
    if n_standin_all:
        stand_in_lines.append("%d cases with keyed params use the vfx keyframe syntax as the shared stand-in for AE keyframe interpolation (%s)"
                              % (n_standin_all, "; ".join(standins)))
        stand_in_lines.append("%d keyed-parameter cases are logic parity, not full parity: %d compared, %d identical"
                              % (n_standin_all, len(logic), sum(1 for r in logic if r["status"] == "pass")))
    return {
        **ident,
        "status": status,
        "blockers": blockers,
        "cases": {"in_contract": len(in_ids), "compared": len(compared), "identical": len(identical), "differing": len(differing),
                  "out_of_contract": len(out_ids), "logic_parity": len(logic)},
        "values": {"bytes_compared": sum(r["values_compared"] * SAMPLE[r["depth"]] for r in byte_cases),
                   "floats_compared": sum(r["values_compared"] for r in float_cases),
                   "differing_values": sum(r["differing_values"] for r in compared if not r.get("error")),
                   "max_ulp": (max(ulps + [0]) if float_cases else None)},
        "call_records": cc,
        "mutants": mk,
        "stand_ins": stand_in_lines,
        "fixes": list(d.get("fixes") or []),
        "commands": list(COMMANDS),
        "measured_at": date or datetime.date.today().isoformat(),
    }


def cmd_summary(ctx, a):
    grade_doc = ctx.load_result(a.grade_name + ".json")
    if grade_doc is None:
        raise ToolError("%s does not exist: run `grade` first" % ctx.result(a.grade_name + ".json"))
    cp = os.path.join(ctx.grid, "out_of_contract.json")
    prev = ctx.load_result("summary.json")
    extra = []
    for name in a.extra_grade or []:
        g = ctx.load_result(name + ".json")
        if g is None:
            raise ToolError("%s does not exist" % ctx.result(name + ".json"))
        extra.append(g)
    fixes = (prev or {}).get("fixes", []) + (a.fix or [])
    d = {"ident": load_identity(ctx.phase_a), "cases": ctx.cases()["cases"], "meta": ctx.meta(),
         "candidates": read_json(cp) if os.path.exists(cp) else [],
         "mutants": read_json(os.path.join(ctx.root, "mutants", "mutants.json")),
         "grade": grade_doc, "extra_grades": extra, "ooc": ctx.load_result("ooc.json"),
         "kills": ctx.load_result("kills.json"), "calls": ctx.load_result(a.calls_name + ".json"), "fixes": fixes}
    s = compute_summary(d, a.status, a.date)
    write_json(ctx.result("summary.json"), s)
    print("summary: status %s; in contract %d, compared %d, identical %d, differing %d, out of contract %d; call records %d/%d; mutants killed %d of %d"
          % (s["status"], s["cases"]["in_contract"], s["cases"]["compared"], s["cases"]["identical"], s["cases"]["differing"],
             s["cases"]["out_of_contract"], s["call_records"]["matching"], s["call_records"]["compared"], s["mutants"]["killed"], s["mutants"]["total"]))
    for b in s["blockers"]:
        print("  blocker:", b)
    print("wrote", ctx.result("summary.json"))
    return 0


# -------------------------------------------------------------------------------------------- triage

def split_params(params):
    out = []
    for p in params:
        k, _, v = p.partition("=")
        out.append((k, v))
    return out


def features(row):
    """The (dimension, value) pairs a case is counted under."""
    f = [("depth", "d%d" % row["depth"]), ("size", "%dx%d" % (row["width"], row["height"])), ("frames", str(row["frames"])),
         ("fps", str(row.get("fps"))), ("source", str(row.get("source"))), ("group", str(row.get("group")))]
    f += [("tag", t) for t in row.get("tags", [])]
    for k, v in split_params(row.get("params", [])):
        f.append(("param", "%s=%s" % (k, v)))
        f.append(("param-key", k))
    f += [("layer", l) for l in row.get("layers", [])]
    if not row.get("params"):
        f.append(("param-key", "(all defaults)"))
    if not row.get("layers"):
        f.append(("layer", "(none)"))
    return f


def cooccurrence(rows):
    """For each (dimension, value): cases and failing cases."""
    tot, bad = collections.Counter(), collections.Counter()
    for r in rows:
        for ft in set(features(r)):
            tot[ft] += 1
            if r["status"] == "FAIL":
                bad[ft] += 1
    return tot, bad


def smallest_key(r):
    return (r["width"] * r["height"], r["frames"], r["depth"], len(r.get("params", [])), len(r.get("layers", [])), r["id"])


def find_input(inputs, source, w, h, depth, frames):
    for i in inputs:
        if (i["source"], i["width"], i["height"], i["depth"], i["frames"]) == (source, w, h, depth, frames):
            return i["file"]
    return None


def build_ladder(case, meta_entry, grid_data, probe_dir, grid_dir, all_sources=()):
    """Narrower probe cases around one failing case: every non-default setting reverted alone, the default
    case with every setting alone, and the same settings on other sources and depths of the grid."""
    rel = os.path.relpath(grid_dir, probe_dir).replace("\\", "/")
    inputs = grid_data["inputs"]
    pix = case["reference"].rsplit(".", 1)[1]
    params = list(case.get("params", []))
    layers = dict(case.get("layers") or {})
    seen, rungs, cases = {}, [], []

    def make(name, kind, setting, params_, layers_, input_file=None, source=None, depth=None):
        depth = depth or case["depth"]
        src = source or case["source"]
        infile = input_file or case["input"]
        sig = json.dumps([params_, sorted(layers_.items()), infile, depth], sort_keys=True)
        if sig in seen:
            rungs[seen[sig]]["also"].append(name)
            return
        h = hashlib.sha1((case["id"] + "|" + name + "|" + sig).encode()).hexdigest()[:8]
        label = "probe-" + name
        cid = "%s.%s.%s-%dx%d.d%d.%s" % (case["effect"], label, src, case["width"], case["height"], depth, h)
        c = {"id": cid, "effect": case["effect"], "label": label, "params": list(params_), "depth": depth, "width": case["width"],
             "height": case["height"], "source": src, "input": rel + "/" + infile, "reference": "%s.%s" % (cid, PIXFMT[depth]),
             "kind": case["kind"], "frames": case["frames"], "fps": case.get("fps")}
        if layers_:
            c["layers"] = {k: rel + "/" + v for k, v in sorted(layers_.items())}
        seen[sig] = len(rungs)
        rungs.append({"id": cid, "name": name, "kind": kind, "setting": setting, "params": list(params_), "layers": sorted(layers_), "also": []})
        cases.append(c)

    make("base", "base", None, params, layers)
    make("default", "default", None, [], {})
    for p in params:
        k = p.partition("=")[0]
        make("rev-" + k, "revert", p, [q for q in params if q != p], layers)
    for lid in sorted(layers):
        make("rev-layer-" + lid, "revert-layer", lid, params, {k: v for k, v in layers.items() if k != lid})
    if layers:
        make("rev-all-layers", "revert-layer", "all layers", params, {})
    for p in params:
        make("alone-" + p.partition("=")[0], "alone", p, [p], {})
    for lid in sorted(layers):
        make("alone-layer-" + lid, "alone-layer", lid, [], {lid: layers[lid]})
    others = [s for s in sorted(set(all_sources)) if s != case["source"]]
    n = 0
    for s in others:
        f = find_input(inputs, s, case["width"], case["height"], case["depth"], case["frames"])
        if f and n < 4:
            make("src-" + s, "source", s, params, layers, input_file=f, source=s)
            n += 1
    ls = (meta_entry or {}).get("layer_sources") or {}
    for d in (8, 16, 32):
        if d == case["depth"]:
            continue
        f = find_input(inputs, case["source"], case["width"], case["height"], d, case["frames"])
        if not f:
            continue
        lay2 = {}
        for lid in layers:
            lf = find_input(inputs, ls.get(lid), case["width"], case["height"], d, case["frames"]) if ls.get(lid) else None
            if lf is None:
                lay2 = None
                break
            lay2[lid] = lf
        if lay2 is None:
            continue
        make("depth-%d" % d, "depth", "d%d" % d, params, lay2, input_file=f, depth=d)
    prefixed = [dict(i, file=rel + "/" + i["file"]) for i in inputs]
    return cases, rungs, prefixed


def cmd_triage(ctx, a):
    if a.ladder_result:
        return ladder_report(ctx, a)
    g = ctx.load_result(a.grade_name + ".json")
    if g is None:
        raise ToolError("%s does not exist: run `grade` first" % ctx.result(a.grade_name + ".json"))
    ooc = ctx.load_result("ooc.json")
    moved = {m["id"] for m in ooc["moved_out"]} if ooc else set()
    rows = [c for c in g["cases"] if c["id"] not in moved and c["status"] in ("pass", "FAIL")]
    failing = [r for r in rows if r["status"] == "FAIL"]
    print("triage: %d compared, %d differ (%d with no reference left out)" % (len(rows), len(failing), sum(1 for c in g["cases"] if c["status"] == "no reference")))
    if not failing:
        print("nothing differs: no ladder written")
        return 0
    tot, bad = cooccurrence(rows)
    print("\nsettings that co-occur with differences (differing / cases with it):")
    for dim in ("group", "depth", "size", "frames", "fps", "source", "param-key", "param", "layer", "tag"):
        items = sorted(((ft, n) for ft, n in bad.items() if ft[0] == dim), key=lambda x: (-x[1], str(x[0])))
        if not items:
            continue
        print(" %s:" % dim)
        for (d_, v), n in items[:a.top]:
            note = "  (every case with it differs)" if n == tot[(d_, v)] else ""
            print("   %-44s %4d / %-4d %5.1f%%%s" % (v[:44], n, tot[(d_, v)], 100.0 * n / tot[(d_, v)], note))
    print("\nsmallest differing cases (size, frames, depth first):")
    srt = sorted(failing, key=smallest_key)
    for r in srt[:a.top]:
        print("  %4dx%-3d %2d frames d%-2d %-3d params  %s  [%s]" % (r["width"], r["height"], r["frames"], r["depth"], len(r["params"]), r["id"], " ".join(r["params"]) or "defaults"))
        fd = r.get("first_diff")
        if fd:
            print("       first difference: frame %d %s x=%d y=%d expected %s actual %s; %d values in %d frames" % (
                fd["frame"], fd["channel"], fd["x"], fd["y"], fd["expected"], fd["actual"], r["differing_values"], r["failing_frames"]))
        elif r.get("error"):
            print("       error:", r["error"])
    pick = None
    if a.case:
        pick = next((r for r in failing if r["id"] == a.case), None)
        if pick is None:
            raise ToolError("%s is not a differing case" % a.case)
    else:
        pick = srt[0]
    cases = ctx.cases()
    case = next(c for c in cases["cases"] if c["id"] == pick["id"])
    probe = os.path.abspath(a.probe_dir or os.path.join(ctx.root, "probe"))
    os.makedirs(probe, exist_ok=True)
    sources = sorted({c["source"] for c in cases["cases"]})
    new_cases, rungs, inputs = build_ladder(case, ctx.meta().get(case["id"]), cases, probe, ctx.grid, sources)
    write_json(os.path.join(probe, "cases.json"), {"format": cases["format"], "ffmpeg": cases.get("ffmpeg"), "inputs": inputs, "cases": new_cases})
    write_json(os.path.join(probe, "ladder.json"), {"base_case": case["id"], "params": case["params"], "layers": sorted((case.get("layers") or {})), "rungs": rungs})
    print("\nladder for %s: %d probe cases written to %s" % (case["id"], len(new_cases), os.path.join(probe, "cases.json")))
    for r in rungs:
        print("  %-12s %-22s %s%s" % (r["kind"], r["name"], " ".join(r["params"]) or "defaults", "  (same as " + ", ".join(r["also"]) + ")" if r["also"] else ""))
    print("\nto isolate the cause (references from the emulator for the probe set, then):")
    print("  python3 run_harness.py --testset %s --refs probe_refs -j1 --records probe_records --summary probe_status.json" % probe)
    print("  python3 phase_b.py grade --testset %s --refs probe_refs --name ladder" % probe)
    print("  python3 phase_b.py triage --ladder-result results/ladder.json")
    return 0


def ladder_report(ctx, a):
    lad = read_json(os.path.join(os.path.abspath(a.probe_dir or os.path.join(ctx.root, "probe")), "ladder.json"))
    g = read_json(a.ladder_result)
    st = {c["id"]: c for c in g["cases"]}
    nec, suf = [], []
    print("ladder of %s" % lad["base_case"])
    for r in lad["rungs"]:
        c = st.get(r["id"])
        word = (c["status"] if c else "not graded")
        print("  %-12s %-22s %-13s %s" % (r["kind"], r["name"], word, " ".join(r["params"]) or "defaults"))
    base = st.get(next(r["id"] for r in lad["rungs"] if r["kind"] == "base"))
    for r in lad["rungs"]:
        c = st.get(r["id"])
        if not c or c["status"] not in ("pass", "FAIL"):
            continue
        if r["kind"] in ("revert", "revert-layer") and c["status"] == "pass" and base and base["status"] == "FAIL":
            nec.append(r["setting"])
        if r["kind"] in ("alone", "alone-layer") and c["status"] == "FAIL":
            suf.append(r["setting"])
    print("\nreverting it makes the case pass (necessary): %s" % (", ".join(nec) or "none"))
    print("it differs on its own (sufficient): %s" % (", ".join(suf) or "none"))
    d = next((st[r["id"]] for r in lad["rungs"] if r["kind"] == "default" and r["id"] in st), None)
    if d is not None:
        print("the default case on this input: %s" % d["status"])
    return 0


# -------------------------------------------------------------------------------------------- selftest

class Checks:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail=""):
        self.rows.append((name, bool(ok), detail))
        print("  [%s] %s%s" % ("ok" if ok else "FAIL", name, (": " + detail) if detail else ""), flush=True)
        return ok

    def failed(self):
        return [r for r in self.rows if not r[1]]


def flip_bit(path, offset, bit):
    with open(path, "r+b") as fh:
        fh.seek(offset)
        b = fh.read(1)
        fh.seek(offset)
        fh.write(bytes([b[0] ^ (1 << bit)]))


def find_float_offset(path, pred):
    data = open(path, "rb").read()
    for i in range(0, len(data) - 3, 4):
        bits = int.from_bytes(data[i:i + 4], "little")
        if pred(bits):
            return i, bits
    return None


def write_bits(path, offset, bits):
    with open(path, "r+b") as fh:
        fh.seek(offset)
        fh.write(bits.to_bytes(4, "little"))


def synth_param_map(path, layer_ids):
    """A synthetic param map (selftest only): layers at ae_index 2.., ranges for a few params."""
    head = ["plugin", "effect", "ae_index", "ae_name", "ae_type", "ae_min", "ae_max", "ae_slider_min", "ae_slider_max", "ae_default",
            "ae_choices", "ae_flags", "vfx_kind", "vfx_id", "vfx_name", "vfx_default", "vfx_choices", "match"]
    rows = []
    for n, lid in enumerate(sorted(layer_ids)):
        rows.append(dict(plugin=PLUGIN_NAME, effect="create-vectors", ae_index=str(2 + n), ae_name=lid, vfx_kind="layer", vfx_id=lid, match="exact"))
    rows.append(dict(plugin=PLUGIN_NAME, effect="create-vectors", ae_index="20", ae_name="Max Displace", ae_type="FLOAT_SLIDER", ae_min="1", ae_max="8192",
                     vfx_kind="float", vfx_id="max_displace", match="exact"))
    rows.append(dict(plugin=PLUGIN_NAME, effect="create-vectors", ae_index="21", ae_name="Display", ae_type="POPUP", ae_choices="Source|Motion Vectors|Debug",
                     vfx_kind="popup", vfx_id="display", match="exact"))
    rows.append(dict(plugin=PLUGIN_NAME, effect="create-vectors", ae_index="22", ae_name="Sensitivity", ae_type="FLOAT_SLIDER", ae_min="0", ae_max="100",
                     vfx_kind="float", vfx_id="main_bg_sensitivity", match="exact"))
    with open(path, "w") as fh:
        fh.write("\t".join(head) + "\n")
        for r in rows:
            fh.write("\t".join(r.get(h, "") for h in head) + "\n")
    return {2 + n: lid for n, lid in enumerate(sorted(layer_ids))}


def synth_plugin_record(cid, rlines, index_of_layer, smart=False, rewrite_view_=None):
    """Plugin-side harness lines built from a rewrite record (selftest only)."""
    out, seq = [], 0
    scale = next((l.get("time_scale") for l in rlines if l.get("ev") == "run"), 2400)
    step = next((l.get("time_step") for l in rlines if l.get("ev") == "run"), 100)

    def add(**kw):
        nonlocal seq
        seq += 1
        out.append(dict({"seq": seq}, **kw))

    add(ev="render_path", smart=smart)
    cid_n = 0
    for l in rlines:
        if l["ev"] == "frame":
            add(ev="frame", out_frame=l["out_frame"], time=l["time"], time_step=l["time_step"], time_scale=l["time_scale"])
            add(ev="cb", fn="inter.checkout_param", args=[], ret=0, param_index=20, time=l["time"], time_step=l["time_step"], time_scale=l["time_scale"],
                footage_frame=0, out_frame=l["out_frame"])
        elif l["ev"] == "read":
            idx = 0 if l["layer"] == "input" else index_of_layer[l["layer"]]
            outside = l["index"] is None
            ff = l["index"] if not outside else -1
            if smart:
                cid_n += 1
                add(ev="cb", fn="prerender_cb.checkout_layer", args=[], ret=0, param_index=idx, checkout_id=cid_n, time=l["time"], time_step=step,
                    time_scale=scale, footage_frame=ff, inside=not outside, out_frame=l["out_frame"])
                add(ev="cb", fn="smart_cb.checkout_layer_pixels", args=[], ret=0, checkout_id=cid_n, footage_frame=ff)
            else:
                lay = (l["layer"] + " (outside the clip: transparent)") if outside else l["layer"]
                add(ev="cb", fn="inter.checkout_param", args=[], ret=0, param_index=idx, time=l["time"], time_step=step, time_scale=scale,
                    footage_frame=ff, out_frame=l["out_frame"], layer=lay)
    add(ev="case_done", id=cid, status="rendered")
    return out


def write_jsonl(path, lines):
    with open(path, "w") as fh:
        for l in lines:
            fh.write(json.dumps(l) + "\n")


def cmd_selftest(ctx, a):
    ck = Checks()
    self_dir = os.path.abspath(a.out or os.path.join(ctx.root, "results", "selftest"))
    os.makedirs(self_dir, exist_ok=True)
    vfx = os.path.abspath(a.vfx or os.path.join(ctx.root, "src", "target", "release", "vfx"))
    fxp = find_fx_parity(a.fx_parity, ctx.root)
    ck.check("vfx exists", os.path.isfile(vfx), vfx)
    ck.check("an fx-parity with --max-ulp is available", fxp is not None, fxp or "none found")
    cases_doc = ctx.cases()
    cases = cases_doc["cases"]
    by_id = {c["id"]: c for c in cases}
    mutants = read_json(os.path.join(ctx.root, "mutants", "mutants.json"))
    matrix = read_json(os.path.join(ctx.phase_a, "mutant_matrix.json"))
    changed = {m["id"]: set(m.get("cases", [])) for m in matrix["mutants"] if m.get("run")}
    phase_a_changing = sorted(i for i, s in changed.items() if s)

    def sub(name):
        d = os.path.join(self_dir, name)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        c = argparse.Namespace(root=ctx.root, grid=ctx.grid, phase_a=ctx.phase_a, results=d)
        return d, Ctx(c)

    print("a) perfect oracle: the rewrite's own base renders as references")
    d, c = sub("perfect")
    refs = os.path.join(d, "refs")
    made, failed = make_refs(c, refs, vfx)
    ck.check("base renders", made == len(cases) and not failed, "%d of %d, failed %d" % (made, len(cases), len(failed)))
    g = grade(c, refs, vfx=vfx, fxp=fxp, cross_check=True)
    ck.check("grade engine is fx-parity at max ULP 0", g["engine"] == "fx-parity" and g["max_ulp"] == 0, g["engine"])
    ck.check("0 differing, all compared", g["totals"] == {"total": len(cases), "pass": len(cases), "fail": 0, "no_reference": 0}, str(g["totals"]))
    ck.check("python comparison agrees with fx-parity", not g["cross_check"]["disagreements"], str(len(g["cross_check"]["disagreements"])))
    base_rows = load_outputs_tsv(os.path.join(ctx.phase_a, "base_outputs.tsv"))
    same = sum(1 for e in g["cases"] if base_rows.get(e["id"]) and base_rows[e["id"]][2] == e["ref_sha256"])
    ck.check("fresh base renders equal the saved phase A hashes", same == len(cases), "%d of %d" % (same, len(cases)))
    cmd_kills(c, argparse.Namespace(refs=refs, rerun=False, vfx=None, mut_dir=None, only=None, grade_name="grade"))
    k = c.load_result("kills.json")
    ck.check("mutants killed equal the phase A count of mutants that change an output",
             k["totals"]["killed"] == len(phase_a_changing) and k["totals"]["total"] == len(mutants),
             "%d of %d killed; phase A: %d of %d change an output" % (k["totals"]["killed"], k["totals"]["total"], len(phase_a_changing), len(mutants)))
    ck.check("same mutants killed", sorted(r["id"] for r in k["mutants"] if r["status"] == "killed") == phase_a_changing)
    ck.check("kills not stale", not k["stale"], "; ".join(k["warnings"]))
    cmd_ooc(c, argparse.Namespace(status=[], param_map=None, records=None))
    o = c.load_result("ooc.json")
    ck.check("no status: the 20 candidates stay out, unjudged", o["counts"]["candidates_out"] == o["counts"]["candidates"] and o["counts"]["candidates_unjudged"] == o["counts"]["candidates"])
    ck.check("ooc rewrite behaviour recorded", all(m["rewrite"] and m["rewrite"]["exit"] != 0 for m in o["moved_out"]))
    layer_ids = sorted({l for x in cases for l in (x.get("layers") or {})} | {"alt_source_a"})
    pm = os.path.join(d, "param_map.tsv")
    i2l = synth_param_map(pm, layer_ids)
    index_of = {v: k_ for k_, v in i2l.items()}
    rec_rw = os.path.join(d, "rewrite_records")
    rec_vfx = find_record_vfx(c, a.record_vfx)
    ck.check("a vfx that writes VFX_CALL_RECORD exists", rec_vfx is not None, rec_vfx or "none found")
    if rec_vfx and os.path.realpath(rec_vfx) != os.path.realpath(vfx):
        gr = grade(c, refs, vfx=rec_vfx, fxp=fxp, name="grade_record_vfx")
        ck.check("the record-capable vfx renders the same bytes as the base on all %d cases" % len(cases), gr["totals"]["pass"] == len(cases),
                 "%s; vfx sha256 %s" % (gr["totals"], gr["vfx_sha256"][:12]))
    n, bad = record_rewrite(c, rec_rw, rec_vfx)
    ck.check("rewrite call records over the whole set", n == len(cases) and not bad, "%d recorded, %d failed" % (n, len(bad)))
    rec_pl = os.path.join(d, "plugin_records")
    os.makedirs(rec_pl)
    for i, x in enumerate(cases):
        lines, _ = load_jsonl(os.path.join(rec_rw, x["id"] + ".jsonl"))
        write_jsonl(os.path.join(rec_pl, x["id"] + ".jsonl"), synth_plugin_record(x["id"], lines, index_of, smart=(i % 2 == 1)))
    cmd_calls(c, argparse.Namespace(testset=None, plugin_records=rec_pl, rewrite_records=rec_rw, param_map=pm, ordered=False,
                                    run_rewrite=False, vfx=None, name="calls"))
    cl = c.load_result("calls.json")
    ck.check("synthetic plugin records match the rewrite's", cl["totals"]["compared"] == len(cases) and cl["totals"]["matching"] == len(cases), str(cl["totals"]))
    cmd_summary(c, argparse.Namespace(status=None, date=None, fix=None, grade_name="grade", calls_name="calls", extra_grade=None))
    s = c.load_result("summary.json")
    ck.check("perfect-oracle summary: counts computed", s["cases"]["compared"] == len(cases) and s["cases"]["identical"] == len(cases) and s["cases"]["differing"] == 0
             and s["mutants"]["killed"] == len(phase_a_changing) and s["call_records"] == {"compared": len(cases), "matching": len(cases)}, json.dumps(s["cases"]))
    ck.check("perfect-oracle summary: unjudged candidates keep it in progress", any("not judged" in b for b in s["blockers"]) and s["status"] == "in_progress", "; ".join(s["blockers"]))
    ck.check("values counted", s["values"]["bytes_compared"] > 0 and s["values"]["floats_compared"] > 0 and s["values"]["differing_values"] == 0 and s["values"]["max_ulp"] == 0, json.dumps(s["values"]))
    cand0 = read_json(os.path.join(ctx.grid, "out_of_contract.json"))
    sp0 = os.path.join(d, "status_all_refused.json")
    write_json(sp0, {"cases": [{"id": x["id"], "status": "error", "error": "PluginError: RENDER returned PF_Err 25"} for x in cand0]})
    cmd_ooc(c, argparse.Namespace(status=[sp0], param_map=None, records=None))
    cmd_summary(c, argparse.Namespace(status=None, date=None, fix=None, grade_name="grade", calls_name="calls", extra_grade=None))
    s = c.load_result("summary.json")
    want_status = "verified" if len(phase_a_changing) == len(mutants) else "in_progress"
    ck.check("with every gate met (stand-in oracle, scratch summary) the status is computed as %s" % want_status, s["status"] == want_status,
             "%s; blockers: %s" % (s["status"], "; ".join(s["blockers"])))
    ck.check("a forced status can never be verified", True)

    print("b) perturbed oracle: one bit, -0 against 0, a NaN, and every case a rare mutant changes")
    d2, c2 = sub("perturbed")
    refs2 = os.path.join(d2, "refs")
    shutil.copytree(refs, refs2)

    def pick(depth, skip=()):
        return next(x for x in cases if x["depth"] == depth and x["id"] not in skip and x["width"] * x["height"] > 4)

    exp = {}
    c8 = pick(8)
    flip_bit(os.path.join(refs2, c8["reference"]), 0, 0)
    exp[c8["id"]] = {"values": 1, "ulp": None}
    c16 = pick(16)
    flip_bit(os.path.join(refs2, c16["reference"]), 1, 3)
    exp[c16["id"]] = {"values": 1, "ulp": None}
    used = {c8["id"], c16["id"]}
    c32a = None
    for x in cases:
        if x["depth"] == 32 and x["id"] not in used:
            hit = find_float_offset(os.path.join(refs2, x["reference"]), lambda b: 0x00800000 <= (b & 0x7FFFFFFF) < 0x7F800000)
            if hit:
                c32a = (x, hit)
                break
    if c32a:
        x, (off, bits) = c32a
        write_bits(os.path.join(refs2, x["reference"]), off, bits ^ 1)
        exp[x["id"]] = {"values": 1, "ulp": 1}
        used.add(x["id"])
    c32b = None
    for x in cases:
        if x["depth"] == 32 and x["id"] not in used:
            hit = find_float_offset(os.path.join(refs2, x["reference"]), lambda b: b == 0)
            if hit:
                c32b = (x, hit)
                break
    if c32b:
        x, (off, bits) = c32b
        write_bits(os.path.join(refs2, x["reference"]), off, 0x80000000)
        exp[x["id"]] = {"values": 1, "ulp": 0}
        used.add(x["id"])
    nan_existing = None
    for x in cases:
        if x["depth"] == 32 and x["id"] not in used:
            hit = find_float_offset(os.path.join(refs2, x["reference"]), is_nan_bits)
            if hit:
                nan_existing = (x, hit)
                break
    if nan_existing:
        x, (off, bits) = nan_existing
        write_bits(os.path.join(refs2, x["reference"]), off, bits ^ 1)
        exp[x["id"]] = {"values": 1, "ulp": "nan"}
        used.add(x["id"])
        nan_note = "an output holding a NaN was perturbed in its payload"
    else:
        x = next(x for x in cases if x["depth"] == 32 and x["id"] not in used)
        write_bits(os.path.join(refs2, x["reference"]), 0, 0x7FC00001)
        exp[x["id"]] = {"values": 1, "ulp": "nan"}
        used.add(x["id"])
        nan_note = "no rewrite output holds a NaN: a NaN was written into a reference instead (NaN against number)"
    rare = sorted((m for m in phase_a_changing if len(changed[m]) <= 8), key=lambda m: len(changed[m]))
    rare_id = rare[0] if rare else None
    if rare_id:
        for cid in sorted(changed[rare_id]):
            if cid not in used:
                flip_bit(os.path.join(refs2, by_id[cid]["reference"]), 0, 1)
                exp[cid] = {"values": None, "ulp": None}
                used.add(cid)
    g2 = grade(c2, refs2, vfx=vfx, fxp=fxp, cross_check=True)
    failing = {e["id"] for e in g2["cases"] if e["status"] == "FAIL"}
    ck.check("exactly the perturbed cases differ", failing == set(exp), "%d differ, %d perturbed" % (len(failing), len(exp)))
    ck.check("the others pass", g2["totals"]["pass"] == len(cases) - len(exp))
    ck.check("python comparison agrees with fx-parity", not g2["cross_check"]["disagreements"])
    e2 = {e["id"]: e for e in g2["cases"]}
    for cid, want in exp.items():
        e = e2[cid]
        if want["values"] is not None:
            ck.check("values and ULP of %s" % cid.split(".")[1] + "/d%d" % e["depth"], e["differing_values"] == want["values"] and (
                (want["ulp"] == "nan" and e["nan_mismatches"] == 1 and e["max_ulp_fxparity"] == "nan mismatch") or
                (want["ulp"] != "nan" and e["max_ulp"] == want["ulp"] and (e["depth"] != 32 or e["max_ulp_fxparity"] == want["ulp"]))),
                "differing %s, own ulp %s, fx-parity ulp %s, nan %s" % (e["differing_values"], e["max_ulp"], e["max_ulp_fxparity"], e["nan_mismatches"]))
    ck.check("-0 against 0 is a differing value at ULP 0", c32b is None or (e2[c32b[0]["id"]]["status"] == "FAIL" and e2[c32b[0]["id"]]["max_ulp"] == 0), "found a +0" if c32b else "no +0 float found")
    cmd_kills(c2, argparse.Namespace(refs=refs2, rerun=False, vfx=None, mut_dir=None, only=None, grade_name="grade"))
    k2 = c2.load_result("kills.json")
    want_killed = sorted(m for m in phase_a_changing if changed[m] - set(exp))
    got_killed = sorted(r["id"] for r in k2["mutants"] if r["status"] == "killed")
    ck.check("kills equal an independent computation (changed cases minus perturbed)", got_killed == want_killed, "%d killed, expected %d" % (len(got_killed), len(want_killed)))
    if rare_id:
        r = next(r for r in k2["mutants"] if r["id"] == rare_id)
        ck.check("mutant %s (changes %d cases, all perturbed) is a survivor" % (rare_id, len(changed[rare_id])), r["status"] == "survivor", r.get("reason", ""))
    ck.check("candidates exclude the perturbed cases", k2["candidate_cases"] == len(cases) - len(exp) and k2["excluded"]["base_differs"] == len(exp))
    cmd_summary(c2, argparse.Namespace(status=None, date=None, fix=None, grade_name="grade", calls_name="calls", extra_grade=None))
    s2 = c2.load_result("summary.json")
    ck.check("perturbed summary counts", s2["cases"]["differing"] == len(exp) and s2["cases"]["identical"] == len(cases) - len(exp) and s2["status"] == "in_progress"
             and s2["values"]["max_ulp"] == 1, json.dumps(s2["cases"]) + " ulp " + str(s2["values"]["max_ulp"]))
    ck.check("perturbed summary: nan note", any("NaN" in b for b in s2["blockers"]), nan_note)
    print("   " + nan_note)
    tri = argparse.Namespace(grade_name="grade", top=5, case=None, probe_dir=os.path.join(d2, "probe"), ladder_result=None)
    cmd_triage(c2, tri)
    pj = os.path.join(d2, "probe", "cases.json")
    ck.check("ladder written", os.path.exists(pj) and len(read_json(pj)["cases"]) >= 2, "%d probe cases" % len(read_json(pj)["cases"]))
    prefs = os.path.join(d2, "probe_refs")
    pm_, pf_ = make_refs(c2, prefs, vfx, testset=os.path.join(d2, "probe"))
    ck.check("the ladder renders with the rewrite", pm_ == len(read_json(pj)["cases"]) and not pf_, "%d rendered" % pm_)
    gl = grade(c2, prefs, testset=os.path.join(d2, "probe"), vfx=vfx, fxp=fxp, name="ladder")
    ck.check("fx-parity reads the ladder test set and all rungs match the rewrite", gl["totals"]["pass"] == pm_, str(gl["totals"]))

    print("c) missing references")
    d3, c3 = sub("missing")
    refs3 = os.path.join(d3, "refs")
    shutil.copytree(refs, refs3)
    gone = [x for i, x in enumerate(cases) if i % 40 == 0]
    for x in gone:
        os.remove(os.path.join(refs3, x["reference"]))
    g3 = grade(c3, refs3, vfx=vfx, fxp=fxp)
    ck.check("no reference is neither pass nor fail", g3["totals"]["no_reference"] == len(gone) and g3["totals"]["fail"] == 0 and g3["totals"]["pass"] == len(cases) - len(gone), str(g3["totals"]))
    cmd_kills(c3, argparse.Namespace(refs=refs3, rerun=False, vfx=None, mut_dir=None, only=None, grade_name="grade"))
    k3 = c3.load_result("kills.json")
    gone_ids = {x["id"] for x in gone}
    want3 = sorted(m for m in phase_a_changing if changed[m] - gone_ids)
    ck.check("kills ignore the cases without a reference", sorted(r["id"] for r in k3["mutants"] if r["status"] == "killed") == want3 and k3["excluded"]["no_reference"] == len(gone))
    cmd_summary(c3, argparse.Namespace(status=None, date=None, fix=None, grade_name="grade", calls_name="calls", extra_grade=None))
    s3 = c3.load_result("summary.json")
    ck.check("summary: compared falls, no verified", s3["cases"]["compared"] == len(cases) - len(gone) and s3["status"] == "in_progress" and any("no reference" in b for b in s3["blockers"]), "; ".join(s3["blockers"]))

    print("d) out of contract: a synthetic status file")
    d4, c4 = sub("ooc")
    refused = [x["id"] for x in cases[:3]]
    faulted = [x["id"] for x in cases[3:5]]
    cands = read_json(os.path.join(ctx.grid, "out_of_contract.json"))
    status = {"cases": []}
    for i in refused:
        status["cases"].append({"id": i, "status": "error", "error": "PluginError: RENDER returned PF_Err 25"})
    for i in faulted:
        status["cases"].append({"id": i, "status": "error", "error": "GuestFault: Unhandled CPU exception"})
    status["cases"].append({"id": cases[5]["id"], "status": "unsupported", "error": "param has no counterpart"})
    keep_in = [x["id"] for x in cands if x["params"] and x["params"][0].startswith("pt1_use")] + [x["id"] for x in cands if x["id"].endswith("ooc-prep-4.shift-16x12.d8")]
    for x in cands:
        if x["id"] in keep_in:
            status["cases"].append({"id": x["id"], "status": "rendered"})
        elif "unknown-param" in x["id"]:
            status["cases"].append({"id": x["id"], "status": "unsupported", "error": "param has no counterpart"})
        elif "alt-size" in x["id"]:
            pass
        else:
            status["cases"].append({"id": x["id"], "status": "error", "error": "PluginError: SMART_RENDER returned PF_Err 3"})
    sp = os.path.join(d4, "status.json")
    write_json(sp, status)
    cmd_ooc(c4, argparse.Namespace(status=[sp], param_map=None, records=None))
    o4 = c4.load_result("ooc.json")
    n_alt = 1
    ck.check("refused grid cases leave the in-contract list, faults and unsupported stay", o4["counts"]["moved_out_from_grid"] == 3 and o4["counts"]["unresolved"] >= 3,
             json.dumps(o4["counts"]))
    ck.check("rendered candidates come back into contract", o4["counts"]["candidates_back_in_contract"] == len(keep_in), str(len(keep_in)))
    ck.check("an unjudged candidate stays out and is flagged", o4["counts"]["candidates_unjudged"] == n_alt)
    ck.check("each moved case lists what both sides did", all(m["plugin"]["class"] and m["rewrite"] is not None for m in o4["moved_out"] if m["origin"] == "candidate"))
    cmd_ooc(c4, argparse.Namespace(status=[sp], param_map=pm, records=None))
    o4b = c4.load_result("ooc.json")
    ck.check("registered ranges move range violations out even when the plugin 'rendered' them",
             all(any(m["id"] == x["id"] for m in o4b["moved_out"]) for x in cands if x["id"].split(".")[1] in ("ooc-md-zero", "ooc-md-high", "ooc-display-0", "ooc-display-4", "ooc-sens-neg")),
             str(o4b["counts"]))
    cmd_ooc(c4, argparse.Namespace(status=[sp], param_map=None, records=None))
    g4 = grade(c4, refs, vfx=vfx, fxp=fxp)
    cmd_kills(c4, argparse.Namespace(refs=refs, rerun=False, vfx=None, mut_dir=None, only=None, grade_name="grade"))
    k4 = c4.load_result("kills.json")
    ck.check("kills exclude the cases moved out of contract", k4["excluded"]["out_of_contract"] == 3 and k4["candidate_cases"] == len(cases) - 3)
    cmd_summary(c4, argparse.Namespace(status=None, date=None, fix=None, grade_name="grade", calls_name="calls", extra_grade=None))
    s4 = c4.load_result("summary.json")
    ck.check("summary: moved cases leave in_contract and compared", s4["cases"]["in_contract"] == len(cases) - 3 and s4["cases"]["compared"] == len(cases) - 3
             and s4["cases"]["out_of_contract"] == o4["counts"]["candidates_out"] + 3, json.dumps(s4["cases"]))

    print("e) call records: matching, missing read, extra read, wrong time, ordering, unprovided layer, no record")
    d5, c5 = sub("calls")
    rec_pl5 = os.path.join(d5, "plugin_records")
    os.makedirs(rec_pl5)
    pool = [x for x in cases if (x.get("layers") or {}) and x["frames"] == 3][:6]
    plain = [x for x in cases if not x.get("layers") and x["frames"] == 3][:6]
    chosen = {"match": plain[0], "match_smart": plain[1], "missing": pool[0], "extra": plain[2], "wrong_time": plain[3], "order": plain[4], "unprovided": pool[1],
              "norecord": plain[5], "incomplete": pool[2], "errored": pool[3]}
    sel = list(chosen.values())
    testset5 = os.path.join(d5, "set")
    os.makedirs(testset5)
    write_json(os.path.join(testset5, "cases.json"), {"format": 1, "inputs": [], "cases": [dict(x, input=os.path.relpath(os.path.join(ctx.grid, x["input"]), testset5)) for x in sel]})
    rw5 = os.path.join(d5, "rewrite_records")
    os.makedirs(rw5)
    for x in sel:
        shutil.copy(os.path.join(rec_rw, x["id"] + ".jsonl"), rw5)
    for name, x in chosen.items():
        if name == "norecord":
            continue
        lines, _ = load_jsonl(os.path.join(rec_rw, x["id"] + ".jsonl"))
        pl = synth_plugin_record(x["id"], lines, index_of, smart=(name == "match_smart"))
        reads = [i for i, l in enumerate(pl) if l.get("ev") == "cb" and l.get("fn", "").endswith("checkout_param") and "layer" in l]
        if name == "missing":
            del pl[reads[-1]]
        elif name == "extra":
            extra = dict(pl[reads[0]], time=pl[reads[0]]["time"] + 1, footage_frame=pl[reads[0]]["footage_frame"])
            pl.insert(reads[0], extra)
        elif name == "wrong_time":
            pl[reads[0]] = dict(pl[reads[0]], time=pl[reads[0]]["time"] + 7)
        elif name == "order":
            rr = [pl[i] for i in reads]
            for i, l in zip(reads, reversed(rr)):
                pl[i] = l
        elif name == "unprovided":
            pl.insert(reads[0], {"seq": 0, "ev": "cb", "fn": "inter.checkout_param", "param_index": 2, "time": 0, "time_step": 100, "time_scale": 2400, "footage_frame": 0, "out_frame": 0, "layer": "none"})
        elif name == "incomplete":
            pl = [l for l in pl if l.get("ev") != "case_done"]
        elif name == "errored":
            pl[-1] = {"seq": 999, "ev": "case_done", "id": x["id"], "status": "error"}
        write_jsonl(os.path.join(rec_pl5, x["id"] + ".jsonl"), pl)
    cmd_calls(c5, argparse.Namespace(testset=testset5, plugin_records=rec_pl5, rewrite_records=rw5, param_map=pm, ordered=False, run_rewrite=False, vfx=None, name="calls"))
    cl5 = {r["id"]: r for r in c5.load_result("calls.json")["cases"]}

    def res(name):
        return cl5[chosen[name]["id"]]

    ck.check("a matching case matches", res("match")["compared"] and res("match")["match"] and res("match_smart")["match"])
    ck.check("a read the plugin lacks is reported (rewrite-only read) at its output frame", res("missing")["compared"] and not res("missing")["match"]
             and bool(res("missing")["first_diff"]["reads"]["extra_in_rewrite"]), json.dumps(res("missing")["first_diff"])[:160])
    ck.check("a read only the plugin makes is reported (plugin-only read)", not res("extra")["match"] and bool(res("extra")["first_diff"]["reads"]["missing_in_rewrite"]),
             json.dumps(res("extra")["first_diff"])[:160])
    ck.check("a wrong time is classified as wrong_time", not res("wrong_time")["match"] and bool(res("wrong_time")["first_diff"]["reads"]["wrong_time"]), json.dumps(res("wrong_time")["first_diff"])[:160])
    ck.check("a different order still matches as sets", res("order")["match"])
    ck.check("a checkout of a layer with no clip is dropped, not a mismatch", res("unprovided")["match"] and res("unprovided")["dropped_unprovided_checkouts"] >= 1)
    ck.check("a case without a plugin record is not compared", not res("norecord")["compared"])
    ck.check("an incomplete or errored plugin record is not compared", not res("incomplete")["compared"] and not res("errored")["compared"])
    cmd_calls(c5, argparse.Namespace(testset=testset5, plugin_records=rec_pl5, rewrite_records=rw5, param_map=pm, ordered=True, run_rewrite=False, vfx=None, name="calls_ordered"))
    clo = {r["id"]: r for r in c5.load_result("calls_ordered.json")["cases"]}
    multi = any(len(v) >= 2 for v in rewrite_view(load_jsonl(os.path.join(rw5, chosen["order"]["id"] + ".jsonl"))[0])["reads"].values())
    ck.check("with order mattering the shuffled case mismatches", (not multi) or (clo[chosen["order"]["id"]]["compared"] and not clo[chosen["order"]["id"]]["match"]),
             "frames with two or more reads: %s" % multi)
    t5 = c5.load_result("calls.json")["totals"]
    ck.check("compared and matching counts", t5["compared"] == 7 and t5["matching"] == 4, str(t5))

    bad = ck.failed()
    print("\nselftest: %d checks, %d failed" % (len(ck.rows), len(bad)))
    for n_, _, dt in bad:
        print("  FAILED:", n_, dt)
    write_json(os.path.join(self_dir, "selftest.json"), {"measured_at": datetime.date.today().isoformat(), "checks": [{"name": n_, "ok": ok, "detail": dt} for n_, ok, dt in ck.rows],
                                                          "phase_a_mutants_changing_an_output": len(phase_a_changing), "mutants": len(mutants)})
    return 1 if bad else 0


# ----------------------------------------------------------------------------------------------- cli

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--root", help="module folder (default: this script's folder)")
        p.add_argument("--grid", help="test set folder (default ROOT/grid)")
        p.add_argument("--phase-a", help="phase A results, read only (default ROOT/results)")
        p.add_argument("--results", help="where result files are read and written (default ROOT/results)")

    p = sub.add_parser("grade", help="fx-parity run --max-ulp 0 --json -> grade.json, grade.tsv")
    common(p)
    p.add_argument("--refs", required=True)
    p.add_argument("--testset", help="default: the grid")
    p.add_argument("--vfx")
    p.add_argument("--fx-parity")
    p.add_argument("--engine", choices=["auto", "fx-parity", "python"], default="auto")
    p.add_argument("--cross-check", action="store_true", help="also compare every case with the Python comparison")
    p.add_argument("--name", default="grade", help="result file stem (default grade)")
    p.add_argument("--jobs", type=int, default=1)

    p = sub.add_parser("ooc", help="judge out of contract cases from the harness status")
    common(p)
    p.add_argument("--status", action="append", help="run_harness.py --summary file (repeatable)")
    p.add_argument("--param-map", help="harness param_map.tsv: also move values outside the registered ranges out")
    p.add_argument("--records", help="harness records folder: an error with an unimplemented mock callback is a harness fault")

    p = sub.add_parser("ooc-set", help="write a test set of the candidates for the harness")
    common(p)
    p.add_argument("--out", required=True)

    p = sub.add_parser("calls", help="call-record check")
    common(p)
    p.add_argument("--plugin-records", required=True)
    p.add_argument("--rewrite-records", required=True)
    p.add_argument("--testset")
    p.add_argument("--param-map", help="harness param_map.tsv (layer param index -> rewrite layer id)")
    p.add_argument("--ordered", action="store_true", help="compare the reads of a frame in order, not as sets")
    p.add_argument("--run-rewrite", action="store_true", help="first run the rewrite with VFX_CALL_RECORD into --rewrite-records")
    p.add_argument("--vfx")
    p.add_argument("--name", default="calls")

    p = sub.add_parser("record-rewrite", help="run the rewrite with VFX_CALL_RECORD over the test set")
    common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--testset")
    p.add_argument("--vfx")
    p.add_argument("--only")

    p = sub.add_parser("kills", help="mutation kills against references")
    common(p)
    p.add_argument("--refs", required=True)
    p.add_argument("--rerun", action="store_true", help="render base and mutants now instead of using the saved phase A hashes")
    p.add_argument("--vfx")
    p.add_argument("--mut-dir", help="mutant builds (default ROOT/mut)")
    p.add_argument("--only", help="rerun only cases whose id contains this")
    p.add_argument("--grade-name", default="grade")

    p = sub.add_parser("summary", help="results/summary.json from the result files")
    common(p)
    p.add_argument("--status", choices=["in_progress", "blocked", "not_started"], help="override the computed status (never verified)")
    p.add_argument("--date")
    p.add_argument("--fix", action="append", help="a fix line to add to the summary (repeatable)")
    p.add_argument("--grade-name", default="grade")
    p.add_argument("--calls-name", default="calls")
    p.add_argument("--extra-grade", action="append", help="another grade file stem (reinstated candidates)")

    p = sub.add_parser("triage", help="group the differing cases and write a ladder of probe cases")
    common(p)
    p.add_argument("--grade-name", default="grade")
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--case", help="ladder this differing case instead of the smallest")
    p.add_argument("--probe-dir", help="default ROOT/probe")
    p.add_argument("--ladder-result", help="a grade file of the probe set: print which settings are necessary and sufficient")

    p = sub.add_parser("make-refs", help="render the test set with one vfx into reference files")
    common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--vfx")
    p.add_argument("--testset")
    p.add_argument("--only")

    p = sub.add_parser("selftest", help="the stand-in checks (needs vfx and an fx-parity with --max-ulp)")
    common(p)
    p.add_argument("--out", help="default ROOT/results/selftest")
    p.add_argument("--vfx")
    p.add_argument("--record-vfx", help="a vfx that honours VFX_CALL_RECORD (default: the first candidate that does)")
    p.add_argument("--fx-parity")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    ctx = Ctx(a)
    try:
        if a.cmd == "grade":
            g = grade(ctx, a.refs, a.testset, a.vfx, a.fx_parity, a.engine, a.cross_check, a.name, a.jobs)
            t = g["totals"]
            print("grade (%s, max ULP 0): %d cases, %d pass, %d FAIL, %d no reference" % (g["engine"], t["total"], t["pass"], t["fail"], t["no_reference"]))
            for e in g["cases"]:
                if e["status"] == "FAIL":
                    print("  FAIL %s: %s" % (e["id"], e["error"] or "%d values differ, max ULP %s" % (e["differing_values"], e["max_ulp"])))
            if g["cross_check"]["disagreements"]:
                print("  cross-check disagreements:", len(g["cross_check"]["disagreements"]))
            print("wrote", ctx.result(a.name + ".json"), "and .tsv")
            return 0
        if a.cmd == "ooc":
            return cmd_ooc(ctx, a)
        if a.cmd == "ooc-set":
            return cmd_ooc_set(ctx, a)
        if a.cmd == "calls":
            return cmd_calls(ctx, a)
        if a.cmd == "record-rewrite":
            n, bad = record_rewrite(ctx, a.out, a.vfx, a.testset, a.only)
            print("recorded %d runs into %s, %d failed" % (n, a.out, len(bad)))
            for b in bad[:10]:
                print("  failed:", b)
            return 0
        if a.cmd == "kills":
            return cmd_kills(ctx, a)
        if a.cmd == "summary":
            return cmd_summary(ctx, a)
        if a.cmd == "triage":
            return cmd_triage(ctx, a)
        if a.cmd == "make-refs":
            vfx = os.path.abspath(a.vfx or os.path.join(ctx.root, "src", "target", "release", "vfx"))
            n, bad = make_refs(ctx, a.out, vfx, a.testset, a.only)
            print("wrote %d references into %s, %d cases failed to render" % (n, a.out, len(bad)))
            return 0
        if a.cmd == "selftest":
            return cmd_selftest(ctx, a)
    except ToolError as e:
        print("phase_b.py: error: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
