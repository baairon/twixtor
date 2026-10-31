#!/usr/bin/env python3
"""Phase B tooling of the retime-time parity module (comparison with the original, once references exist).

Python 3 standard library only. Nothing here runs the plugin; it grades reference renders somebody else
made. Subcommands (all take --results DIR, default results/, and --phase-a DIR, default results/):

  grade           fx-parity run --max-ulp 0 --json over the test set -> grade.json, grade.tsv
  ooc-set         writes a small test set of the out-of-contract candidates, so the harness can run them
  ooc             harness per-case status -> ooc.json (cases moved out of contract, what each side did)
  rewrite-records runs the rewrite with VFX_CALL_RECORD over the test set into a directory
  calls           call-record check, plugin records against rewrite records -> calls.json, calls.tsv
  kills           mutation kills against the references -> kills.json
  summary         grade.json (+ ooc, kills, calls) -> summary.json, every count computed from files
  triage          groups the differing cases, smallest failing case first, writes a probe ladder test set
  selftest        stand-in oracles (perfect, perturbed, missing, out of contract, call records)

The bar: same inputs, identical outputs (bytes at 8 and 16 bpc, bits at 32 bpc, --max-ulp 0).
A case with no reference is "no reference", never a pass and never a failure.
Run it from this folder on the server (~/twixtor-parity/retime-time/), one core: taskset -c N python3 phase_b.py ...
"""
import argparse
import collections
import datetime
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from array import array
from fractions import Fraction

MOD = os.path.dirname(os.path.abspath(__file__))
MODULE = "retime-time"
PLUGIN = "Twixtor_64.aex"
EFFECT = "retime"
MIN_CASES = 150
PIX = {8: "rgba", 16: "rgba64le", 32: "gbrapf32le"}
WIDTH = {8: 1, 16: 2, 32: 4}
DEFAULTS = {"time_remap_mode": "1", "speed_percent": "100", "frame_num": "0", "in_fps_is_out_fps": "1",
            "input_frame_rate": "29.969999313354492", "input_fields": "1", "frame_interp": "2",
            "src_mark_segments": "1"}
FIXED = ("vectors",)
SECTION_KEYS = {"cases": ["in_contract", "compared", "identical", "differing", "out_of_contract", "logic_parity"],
                "values": ["bytes_compared", "floats_compared", "differing_values", "max_ulp"],
                "call_records": ["compared", "matching"], "mutants": ["total", "killed", "survivors"]}
SUMMARY_KEYS = ["module", "plugin", "plugin_sha256", "effect", "status", "blockers", "cases", "values",
                "call_records", "mutants", "stand_ins", "fixes", "commands", "measured_at"]
STAND_IN_TEXT = ("keyed curves (speed_percent, frame_num, src_mark_segments with @ keys): host side keyframe "
                 "interpolation, one labelled stand-in on both sides, cases counted as logic_parity")


# ---------------------------------------------------------------------------------------------
# small helpers

def jload(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as fh:
        return json.load(fh)


def jdump(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv_rows(path):
    """id -> (exit, size, sha256) of a render_grid.py table."""
    rows = {}
    with open(path) as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            rows[p[0]] = (p[1], p[2], p[3])
    return rows


def safe(text):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def fnv(text):
    h = 0x811C9DC5
    for b in text.encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


class Ctx:
    """Where things are: results (outputs and phase B inputs), phase A files, the test set."""

    def __init__(self, a):
        self.results = os.path.abspath(getattr(a, "results", None) or os.path.join(MOD, "results"))
        self.phase_a = os.path.abspath(getattr(a, "phase_a", None) or os.path.join(MOD, "results"))
        ts = getattr(a, "testset", None)
        if not ts:
            ts = os.path.join(MOD, "testset")
            if not os.path.exists(os.path.join(ts, "cases.json")):
                ts = os.path.join(MOD, "results")
        self.testset = os.path.abspath(ts)

    def cases(self):
        doc = jload(os.path.join(self.testset, "cases.json"))
        if doc is None:
            raise SystemExit("phase_b: no cases.json in %s" % self.testset)
        return doc["cases"]

    def grid(self):
        for d in (self.testset, self.phase_a):
            g = jload(os.path.join(d, "grid.json"))
            if g is not None:
                return g
        return {}

    def path(self, name):
        return os.path.join(self.results, name)

    def find(self, name):
        """A phase A file: in the test set folder (the server keeps the case files there) or in the phase A results."""
        for d in (self.testset, self.phase_a):
            if os.path.exists(os.path.join(d, name)):
                return os.path.join(d, name)
        return os.path.join(self.phase_a, name)


def log_command(ctx, argv):
    """Appends the exact command line to results/commands_b.txt (summary.json lists them)."""
    try:
        os.makedirs(ctx.results, exist_ok=True)
        line = "python3 phase_b.py " + " ".join(argv)
        path = ctx.path("commands_b.txt")
        old = open(path).read().splitlines() if os.path.exists(path) else []
        if line not in old:
            with open(path, "a") as fh:
                fh.write(line + "\n")
    except OSError:
        pass


def find_bin(name, flag=None):
    """The binary from the flag, $FX_PARITY / $VFX, or the usual places on the server. An fx-parity that does not
    know run --max-ulp (an older build) is skipped."""
    env = {"fx-parity": "FX_PARITY", "vfx": "VFX"}[name]
    home = os.path.expanduser("~/twixtor-parity/")
    where = ["harness/repo/target/release/", "src/base/target/release/"]
    cands = [flag, os.environ.get(env)]
    if name == "fx-parity":
        cands += [home + w + name for w in where] + [os.path.join(MOD, "src", "target", "release", name)]
    else:
        cands += [os.path.join(MOD, "src", "target", "release", name)] + [home + w + name for w in where]
    cands.append(shutil.which(name))
    for c in cands:
        if not (c and os.path.isfile(c) and os.access(c, os.X_OK)):
            continue
        if name == "fx-parity" and c != flag:
            h = subprocess.run([c, "--help"], capture_output=True)
            if b"--max-ulp" not in h.stdout + h.stderr:
                continue
        return os.path.abspath(c)
    raise SystemExit("phase_b: cannot find %s (use --%s or $%s)" % (name, name, env))


def case_tags(case, grid):
    m = grid.get(case["id"], {})
    return collections.OrderedDict([
        ("group", m.get("group", "")), ("class", m.get("class", "value")), ("depth", case["depth"]),
        ("size", "%dx%d" % (case["width"], case["height"])), ("frames", case["frames"]),
        ("fps", case.get("fps") or ""), ("source", case.get("source", ""))])


def parse_params(params):
    out = collections.OrderedDict()
    for p in params:
        k, _, v = p.partition("=")
        out[k] = v
    return out


def vfx_args(case, testset):
    """The vfx command line fx-parity builds for a temporal case (and render_grid.py uses)."""
    a = ["apply", case["effect"], "--width", str(case["width"]), "--height", str(case["height"]),
         "--depth", str(case["depth"])]
    for p in case.get("params") or []:
        a += ["--param", p]
    if case.get("fps"):
        a += ["--fps", case["fps"]]
    a += ["--frames", str(case["frames"])]
    for lid, f in sorted((case.get("layers") or {}).items()):
        a += ["--layer", "%s=%s" % (lid, os.path.join(testset, f))]
    a += ["--in", os.path.join(testset, case["input"])]
    return a


def render_case(vfx, testset, case, out_path, record=None):
    env = dict(os.environ, RAYON_NUM_THREADS="1")
    env.pop("VFX_CALL_RECORD", None)
    if record:
        env["VFX_CALL_RECORD"] = record
    p = subprocess.run([vfx] + vfx_args(case, testset) + ["--out", out_path], capture_output=True, env=env)
    return p.returncode


def render_hash(vfx, testset, case):
    """(exit, size, sha256) of one render, like a render_grid.py row."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "o")
        rc = render_case(vfx, testset, case, out)
        if os.path.exists(out):
            return (str(rc), str(os.path.getsize(out)), sha256_file(out))
        return (str(rc), "0", "")


# ---------------------------------------------------------------------------------------------
# value level comparison

def ordered_bits(b):
    return -(b & 0x7FFFFFFF) if b & 0x80000000 else b


def is_nan_bits(b):
    return (b & 0x7F800000) == 0x7F800000 and (b & 0x007FFFFF) != 0


def value_diff(depth, ref, out):
    """Sample level comparison of two raw clips. 8 and 16 bpc samples are bytes and 16 bit words, 32 bpc
    samples are the float bit patterns (so -0 against 0 and differing NaN bits count as different).
    Returns values, differing, max_ulp (finite pairs; None below 32 bpc), nan_mismatch, first_byte."""
    w = WIDTH[depth]
    res = {"values": max(len(ref), len(out)) // w, "differing": 0, "max_ulp": 0 if depth == 32 else None,
           "nan_mismatch": 0, "first_byte": None, "size_mismatch": len(ref) != len(out)}
    if ref == out:
        return res
    n = min(len(ref), len(out))
    first = next((i for i in range(n) if ref[i] != out[i]), n)
    res["first_byte"] = first
    if len(ref) != len(out):
        res["differing"] = res["values"]
        return res
    if depth == 8:
        res["differing"] = sum(1 for x, y in zip(ref, out) if x != y)
        return res
    code = "H" if depth == 16 else "I"
    ra, oa = array(code), array(code)
    ra.frombytes(ref)
    oa.frombytes(out)
    if sys.byteorder == "big":
        ra.byteswap()
        oa.byteswap()
    ulp = 0
    for x, y in zip(ra, oa):
        if x == y:
            continue
        res["differing"] += 1
        if depth == 32:
            if is_nan_bits(x) or is_nan_bits(y):
                res["nan_mismatch"] += 1
            else:
                ulp = max(ulp, abs(ordered_bits(x) - ordered_bits(y)))
    if depth == 32:
        res["max_ulp"] = ulp
    return res


# ---------------------------------------------------------------------------------------------
# grade

def run_fx_parity(fx, vfx, testset, refs, jobs=1, max_ulp=0):
    cmd = [fx, "run", "--testset", testset, "--refs", refs, "--max-ulp", str(max_ulp), "--json", "--vfx", vfx,
           "--jobs", str(jobs)]
    env = dict(os.environ, RAYON_NUM_THREADS="1")
    p = subprocess.run(cmd, capture_output=True, env=env)
    out = p.stdout.decode(errors="replace")
    try:
        doc = json.loads(out)
    except ValueError:
        raise SystemExit("phase_b: fx-parity run gave no JSON (exit %d): %s" % (
            p.returncode, p.stderr.decode(errors="replace")[:400]))
    if p.returncode not in (0, 1):
        raise SystemExit("phase_b: fx-parity run exit %d: %s" % (p.returncode, p.stderr.decode(errors="replace")[:400]))
    return doc, cmd


def grade_rows(report, cases, grid, testset, refs, vfx):
    """One row per case of the test set. Value level counts: every compared case counts its reference size;
    a differing case is rendered again with vfx to count the differing samples and the largest ULP."""
    by_id = {c["id"]: c for c in report["cases"]}
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for c in cases:
            r = by_id.get(c["id"])
            tags = case_tags(c, grid)
            row = collections.OrderedDict()
            row["id"] = c["id"]
            row["status"] = r["status"] if r else "no reference"
            row["tags"] = tags
            row["logic_parity"] = tags["class"] == "logic_parity"
            row["params"] = c.get("params", [])
            row["depth"] = c["depth"]
            row["width"] = c["width"]
            row["height"] = c["height"]
            row["frames"] = c["frames"]
            ref_path = os.path.join(refs, c["reference"])
            row["bytes_compared"] = 0
            row["floats_compared"] = 0
            row["differing_values"] = 0
            row["max_ulp"] = None
            row["nan_mismatch_values"] = 0
            row["first_failing_frame"] = None
            row["failing_frames"] = 0
            row["error"] = None
            if row["status"] == "no reference" or not os.path.isfile(ref_path):
                row["status"] = "no reference"
                rows.append(row)
                continue
            size = os.path.getsize(ref_path)
            if c["depth"] == 32:
                row["floats_compared"] = size // 4
                row["max_ulp"] = 0
            else:
                row["bytes_compared"] = size
            if row["status"] == "FAIL":
                row["first_failing_frame"] = r.get("frame")
                row["failing_frames"] = r.get("failing_frames", 0)
                row["error"] = r.get("error")
                row["fx_parity"] = {k: r[k] for k in ("differing_pixels", "clip_differing_pixels", "max_ulp",
                                                      "max_abs_diff", "mean_abs_diff") if k in r}
                out = os.path.join(tmp, "o")
                if os.path.exists(out):
                    os.remove(out)
                render_case(vfx, testset, c, out)
                if os.path.exists(out):
                    with open(ref_path, "rb") as fa, open(out, "rb") as fb:
                        vd = value_diff(c["depth"], fa.read(), fb.read())
                    row["differing_values"] = vd["differing"]
                    row["nan_mismatch_values"] = vd["nan_mismatch"]
                    if c["depth"] == 32:
                        row["max_ulp"] = vd["max_ulp"]
                    frame_len = c["width"] * c["height"] * 4 * WIDTH[c["depth"]]
                    if vd["first_byte"] is not None and frame_len:
                        row["first_value_frame"] = vd["first_byte"] // frame_len
                else:
                    row["error"] = (row["error"] or "") + " (vfx wrote no output to count values)"
            rows.append(row)
    return rows


def grade_totals(rows):
    t = collections.OrderedDict()
    cmp_rows = [r for r in rows if r["status"] in ("pass", "FAIL")]
    t["cases"] = len(rows)
    t["compared"] = len(cmp_rows)
    t["identical"] = sum(1 for r in rows if r["status"] == "pass")
    t["differing"] = sum(1 for r in rows if r["status"] == "FAIL")
    t["no_reference"] = sum(1 for r in rows if r["status"] == "no reference")
    t["bytes_compared"] = sum(r["bytes_compared"] for r in cmp_rows)
    t["floats_compared"] = sum(r["floats_compared"] for r in cmp_rows)
    t["differing_values"] = sum(r["differing_values"] for r in cmp_rows)
    ulps = [r["max_ulp"] for r in cmp_rows if r["max_ulp"] is not None]
    t["max_ulp"] = max(ulps) if ulps else None
    t["nan_mismatch_values"] = sum(r["nan_mismatch_values"] for r in cmp_rows)
    return t


def write_grade_tsv(path, rows):
    with open(path, "w") as fh:
        fh.write("id\tstatus\tdiffering_values\tmax_ulp\ttags\tlogic_parity\tfirst_failing_frame\tfailing_frames\terror\n")
        for r in rows:
            tags = ";".join("%s=%s" % kv for kv in r["tags"].items())
            fh.write("\t".join([
                r["id"], r["status"], str(r["differing_values"]), "" if r["max_ulp"] is None else str(r["max_ulp"]),
                tags, "yes" if r["logic_parity"] else "no",
                "" if r["first_failing_frame"] is None else str(r["first_failing_frame"]),
                str(r["failing_frames"]), (r["error"] or "").replace("\t", " ").replace("\n", " ")]) + "\n")


def cmd_grade(a):
    ctx = Ctx(a)
    t0 = time.time()
    fx, vfx = find_bin("fx-parity", a.fx_parity), find_bin("vfx", a.vfx)
    refs = os.path.abspath(a.refs)
    if not os.path.isdir(refs):
        raise SystemExit("phase_b: no references folder %s" % refs)
    cases = ctx.cases()
    report, cmd = run_fx_parity(fx, vfx, ctx.testset, refs, a.jobs)
    rows = grade_rows(report, cases, ctx.grid(), ctx.testset, refs, vfx)
    doc = collections.OrderedDict()
    doc["measured_at"] = datetime.date.today().isoformat()
    doc["testset"] = ctx.testset
    doc["refs"] = refs
    doc["command"] = " ".join(cmd)
    doc["fx_parity_summary"] = report.get("summary")
    doc["totals"] = grade_totals(rows)
    doc["cases"] = rows
    jdump(ctx.path("grade.json"), doc)
    write_grade_tsv(ctx.path("grade.tsv"), rows)
    t = doc["totals"]
    print("graded %(cases)d cases: %(compared)d compared, %(identical)d identical, %(differing)d differing, "
          "%(no_reference)d no reference; differing values %(differing_values)d, max ULP %(max_ulp)s" % t
          + " (%.0f s)" % (time.time() - t0))
    return 0


# ---------------------------------------------------------------------------------------------
# out of contract

DEFAULT_OOC = ("refused", "timeout", "plugin_error")


def load_status(paths):
    """Harness per-case status: run_harness.py --summary ({"cases": [...]}), a list of case dicts, or a dict
    id -> entry (or id -> status word)."""
    out = {}
    for p in paths:
        d = jload(p)
        items = d.get("cases") if isinstance(d, dict) and "cases" in d else d
        if isinstance(items, list):
            for e in items:
                out[e["id"]] = e
        elif isinstance(items, dict):
            for k, v in items.items():
                out[k] = v if isinstance(v, dict) else {"id": k, "status": v}
    return out


def classify_status(entry):
    st = str(entry.get("status") or "").lower()
    err = str(entry.get("error") or "")
    if st == "rendered":
        return "rendered"
    if st in ("refused", "rejected"):
        return "refused"
    if st in ("timeout", "unsupported", "not_run"):
        return st
    if st == "error":
        return "plugin_error" if err.startswith("PluginError") else "emulator_error"
    return "unknown"


OOC_REASON = {
    "refused": "the original refused the case",
    "timeout": "the original did not finish rendering the case in time",
    "plugin_error": "the original returned an error (PF_Err) for the case",
    "emulator_error": "the emulator failed on the case (not necessarily the original)",
    "unsupported": "the harness could not set the case up (stand-in limit, not the original)",
    "not_run": "the harness did not run the case",
    "unknown": "unknown harness status",
}


def judge_cases(cases, status, ooc_status):
    """-> moved (list), unjudged (list), rendered_ids. A case whose class is in ooc_status leaves the
    in-contract list; other non-rendered classes stay in contract with no reference (unjudged)."""
    moved, unjudged, rendered = [], [], []
    for c in cases:
        e = status.get(c["id"])
        if e is None:
            continue
        cls = classify_status(e)
        if cls == "rendered":
            rendered.append(c["id"])
        elif cls in ooc_status:
            moved.append({"id": c["id"], "source": "testset", "label": c.get("label"), "params": c.get("params"),
                          "reason": OOC_REASON[cls], "plugin": {"status": e.get("status"), "class": cls,
                                                                "error": e.get("error")}, "rewrite": None})
        else:
            unjudged.append({"id": c["id"], "plugin_status": e.get("status"), "class": cls, "reason": OOC_REASON[cls],
                             "error": e.get("error")})
    return moved, unjudged, rendered


def candidate_cases(ctx, out_dir=None):
    """The out-of-contract candidates as cases of a small test set (all on one tiny 8 bpc clip)."""
    cands = jload(ctx.find("out_of_contract.json"), [])
    cases = ctx.cases()
    ins = jload(os.path.join(ctx.testset, "cases.json")).get("inputs", [])
    pick = None
    for i in sorted((x for x in ins if x["depth"] == 8), key=lambda x: (x["width"] * x["height"] * x["frames"], x["file"])):
        pick = i
        break
    if pick is None:
        raise SystemExit("phase_b: the test set has no 8 bpc input to use for the candidates")
    out = []
    for n, c in enumerate(cands):
        key = "ooc|%d|%s|%s" % (n, c["label"], ",".join(c["params"]))
        cid = "%s.ooc%02d-%s.%s-%dx%d.d8.%08x" % (EFFECT, n, safe(c["label"]), pick["source"], pick["width"],
                                                   pick["height"], fnv(key))
        out.append({"id": cid, "effect": EFFECT, "label": "ooc-" + safe(c["label"]), "params": c["params"], "depth": 8,
                    "width": pick["width"], "height": pick["height"], "source": pick["source"], "input": pick["file"],
                    "reference": "%s.%s" % (cid, PIX[8]), "kind": "temporal", "frames": pick["frames"], "fps": "24",
                    "candidate": c["label"]})
    return out, pick


def cmd_ooc_set(a):
    ctx = Ctx(a)
    out_dir = os.path.abspath(a.out)
    cases, pick = candidate_cases(ctx)
    os.makedirs(os.path.join(out_dir, "inputs"), exist_ok=True)
    dest = os.path.join(out_dir, pick["file"])
    if not os.path.exists(dest):
        shutil.copyfile(os.path.join(ctx.testset, pick["file"]), dest)
    jdump(os.path.join(out_dir, "cases.json"), {"format": 1, "ffmpeg": "none (out-of-contract candidates)",
                                               "inputs": [pick], "cases": cases})
    print("wrote %s: %d candidate cases on %s" % (os.path.join(out_dir, "cases.json"), len(cases), pick["file"]))
    return 0


def plugin_ranges(path, plugin="Twixtor_64"):
    """vfx id -> (min, max) of the registered range, from param_map.tsv (empty until a plugin reached PARAMS_SETUP)."""
    out = {}
    if not os.path.exists(path):
        return out
    lines = open(path).read().splitlines()
    cols = lines[0].split("\t")
    for line in lines[1:]:
        r = dict(zip(cols, line.split("\t")))
        if r.get("plugin") != plugin or not r.get("vfx_id") or not r.get("ae_index"):
            continue
        try:
            out[r["vfx_id"]] = (float(r["ae_min"]), float(r["ae_max"]))
        except (KeyError, ValueError):
            pass
    return out


def outside_range(params, ranges):
    """True when a changed param value lies outside the plugin's registered range; None when unknown."""
    if not ranges:
        return None
    known = False
    for k, v in parse_params(params).items():
        if k not in ranges:
            continue
        for part in v.split(";"):
            try:
                x = float(part.split("@")[0])
            except ValueError:
                continue
            known = True
            if x < ranges[k][0] or x > ranges[k][1]:
                return True
    return False if known else None


def judge_candidates(cands_cases, status, rewrite_rows, ooc_status, ranges):
    out = []
    for c in cands_cases:
        e = status.get(c["id"])
        rw = rewrite_rows.get(c["candidate"], {})
        item = {"id": c["id"], "source": "candidate", "label": c["candidate"], "params": c["params"],
                "rewrite": {"check_exit": rw.get("check_exit"), "render_exit": rw.get("render_exit"),
                            "message": rw.get("render_message") or rw.get("check_message")},
                "plugin": None, "outside_registered_range": outside_range(c["params"], ranges)}
        if e is None:
            item["plugin"] = {"status": "not run"}
            item["verdict"] = "out_of_contract"
            item["reason"] = "the rewrite refuses it; the original was not run on it yet"
        else:
            cls = classify_status(e)
            item["plugin"] = {"status": e.get("status"), "class": cls, "error": e.get("error")}
            if cls == "rendered":
                if item["outside_registered_range"]:
                    item["verdict"] = "out_of_contract"
                    item["reason"] = "outside the parameter range the plugin registers"
                else:
                    item["verdict"] = "review"
                    item["reason"] = ("the original rendered it and the value is "
                                      + ("inside" if item["outside_registered_range"] is False else "not known to be outside")
                                      + " its registered range, while the rewrite refuses it: a possible difference")
            else:
                item["verdict"] = "out_of_contract"
                item["reason"] = OOC_REASON.get(cls, "")
        out.append(item)
    return out


def cmd_ooc(a):
    ctx = Ctx(a)
    status = load_status(a.status)
    ooc_status = set(a.ooc_status.split(",")) if a.ooc_status else set(DEFAULT_OOC)
    cases = ctx.cases()
    moved, unjudged, rendered = judge_cases(cases, status, ooc_status)
    base = read_tsv_rows(os.path.join(ctx.phase_a, "base.tsv")) if os.path.exists(os.path.join(ctx.phase_a, "base.tsv")) else {}
    for m in moved:
        b = base.get(m["id"])
        m["rewrite"] = {"exit": b[0], "size": b[1]} if b else None
    rows = {}
    tsv = os.path.join(ctx.phase_a, "out_of_contract.tsv")
    if os.path.exists(tsv):
        lines = open(tsv).read().splitlines()
        cols = lines[0].split("\t")
        for line in lines[1:]:
            r = dict(zip(cols, line.split("\t")))
            rows[r["label"]] = r
    ccases, _ = candidate_cases(ctx)
    cands = judge_candidates(ccases, status, rows, ooc_status, plugin_ranges(a.param_map))
    known = {c["id"] for c in cases} | {c["id"] for c in ccases}
    doc = collections.OrderedDict()
    doc["measured_at"] = datetime.date.today().isoformat()
    doc["status_files"] = [os.path.abspath(p) for p in a.status]
    doc["moved_when_status_is"] = sorted(ooc_status)
    doc["counts"] = collections.OrderedDict([
        ("testset_cases", len(cases)), ("status_entries", len(status)), ("rendered", len(rendered)),
        ("moved_out", len(moved)), ("unjudged", len(unjudged)), ("in_contract", len(cases) - len(moved)),
        ("candidates", len(cands)),
        ("candidates_out_of_contract", sum(1 for c in cands if c["verdict"] == "out_of_contract")),
        ("candidates_to_review", sum(1 for c in cands if c["verdict"] == "review"))])
    doc["moved"] = moved
    doc["candidates"] = cands
    doc["unjudged"] = unjudged
    doc["unknown_ids"] = sorted(set(status) - known)
    jdump(ctx.path("ooc.json"), doc)
    c = doc["counts"]
    print("out of contract: %d moved from the test set, %d of %d candidates stay out (%d to review), %d unjudged; "
          "%d cases remain in contract" % (c["moved_out"], c["candidates_out_of_contract"], c["candidates"],
                                          c["candidates_to_review"], c["unjudged"], c["in_contract"]))
    return 0


# ---------------------------------------------------------------------------------------------
# call records

def parse_jsonl(path):
    lines, bad = [], 0
    with open(path) as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                lines.append(json.loads(raw))
            except ValueError:
                bad += 1
    return lines, bad


def frac(t, scale):
    try:
        return Fraction(int(t), int(scale))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def norm_rewrite(lines):
    """frames: out_frame -> {time, reads [(layer, index|None, time)]}; clip_frames."""
    frames = collections.OrderedDict()
    run = next((l for l in lines if l.get("ev") == "run"), {})
    clip_frames = None
    for l in lines:
        ev = l.get("ev")
        if ev == "frame":
            frames[l["out_frame"]] = {"time": frac(l["time"], l["time_scale"]), "reads": [], "params": None}
        elif ev == "read":
            f = frames.setdefault(l["out_frame"], {"time": None, "reads": [], "params": None})
            scale = run.get("time_scale") or 1
            f["reads"].append((l["layer"], l["index"], frac(l["time"], scale)))
            clip_frames = l.get("clip_frames", clip_frames)
        elif ev == "param":
            f = frames.setdefault(l["out_frame"], {"time": None, "reads": [], "params": None})
            f["params"] = (f["params"] or []) + [(str(l.get("param")), frac(l["time"], l.get("time_scale") or run.get("time_scale") or 1))]
    return {"frames": frames, "clip_frames": clip_frames if clip_frames is not None else run.get("frames"), "run": run}


def layer_kinds(param_map, plugin="Twixtor_64"):
    """ae_index -> (vfx id, vfx kind) from param_map.tsv."""
    out = {}
    if param_map and os.path.exists(param_map):
        lines = open(param_map).read().splitlines()
        cols = lines[0].split("\t")
        for line in lines[1:]:
            r = dict(zip(cols, line.split("\t")))
            if r.get("plugin") == plugin and r.get("ae_index", "").isdigit():
                out[int(r["ae_index"])] = (r.get("vfx_id"), r.get("vfx_kind"))
    return out


def norm_plugin(lines, clip_frames, kinds=None, present=None):
    """The harness record of one case in the same shape as norm_rewrite. Frame reads are
    prerender_cb.checkout_layer and inter.checkout_param on a layer param (index 0 is the input);
    checkout_param on other params are param reads. When the record has pixel checkouts
    (smart_cb.checkout_layer_pixels), the frame reads are those pixels instead, at the time and frame of
    the pre-render checkout with the same checkout_id: the other pre-render checkouts (the probes at the
    current time and at minus the current time) read nothing. With `present` (the layer names the case
    provides, "input" always), pixels of a layer the case does not provide are not reads."""
    kinds = kinds or {}
    frames = collections.OrderedDict()
    cur = None
    done = None
    smart = None
    pixel_mode = any(l.get("fn") == "smart_cb.checkout_layer_pixels" for l in lines)
    ck = {}
    for l in lines:
        if pixel_mode and l.get("ev") == "cb" and l.get("fn") in ("prerender_cb.checkout_layer", "smart_cb.checkout_layer_pixels"):
            if l["fn"] == "prerender_cb.checkout_layer":
                pi = l.get("param_index")
                name = "input" if pi == 0 else (kinds.get(pi, (None, None))[0] or "param%s" % pi)
                ff = l.get("footage_frame")
                inside = isinstance(ff, int) and clip_frames is not None and 0 <= ff < clip_frames and l.get("inside", True)
                ck[l.get("checkout_id")] = (name, ff if inside else None, frac(l.get("time"), l.get("time_scale")))
            else:
                n = l.get("out_frame", cur)
                read = ck.get(l.get("checkout_id"))
                if n is not None and read is not None and (present is None or read[0] in present):
                    frames.setdefault(n, {"time": None, "reads": [], "params": []})["reads"].append(read)
            continue
        ev = l.get("ev")
        if ev == "frame":
            cur = l["out_frame"]
            frames[cur] = {"time": frac(l["time"], l["time_scale"]), "reads": [], "params": []}
        elif ev == "render_path":
            smart = l.get("smart")
        elif ev in ("case_done", "case_error"):
            done = l
        elif ev == "cb" and l.get("fn") in ("prerender_cb.checkout_layer", "inter.checkout_param"):
            n = l.get("out_frame", cur)
            if n is None:
                continue
            f = frames.setdefault(n, {"time": None, "reads": [], "params": []})
            pi = l.get("param_index")
            t = frac(l.get("time"), l.get("time_scale"))
            is_layer = l["fn"] == "prerender_cb.checkout_layer" or pi == 0 or kinds.get(pi, (None, None))[1] == "layer"
            if is_layer:
                name = "input" if pi == 0 else (kinds.get(pi, (None, None))[0] or "param%s" % pi)
                ff = l.get("footage_frame")
                inside = isinstance(ff, int) and clip_frames is not None and 0 <= ff < clip_frames and l.get("inside", True)
                f["reads"].append((name, ff if inside else None, t))
            else:
                name = kinds.get(pi, (None, None))[0] or "param%s" % pi
                f["params"].append((name, t))
    return {"frames": frames, "done": done, "smart": smart}


def _fmt(items):
    return sorted("%s:%s@%s" % (a, "out" if b is None else b, "?" if c is None else "%d/%d" % (c.numerator, c.denominator))
                  for a, b, c in items)


def compare_calls(plug, rew, ordered=False):
    """None when the two records agree, else the first differing output frame with what differs."""
    for n in sorted(set(plug["frames"]) | set(rew["frames"])):
        p, r = plug["frames"].get(n), rew["frames"].get(n)
        if p is None:
            return {"first_out_frame": n, "kind": "extra_frame", "detail": "the rewrite renders a frame the original did not"}
        if r is None:
            return {"first_out_frame": n, "kind": "missing_frame", "detail": "the original rendered a frame the rewrite did not"}
        if p["time"] != r["time"]:
            return {"first_out_frame": n, "kind": "wrong_frame_time", "detail": "plugin %s, rewrite %s" % (p["time"], r["time"])}
        if ordered:
            if p["reads"] != r["reads"]:
                kind = "wrong_order" if sorted(map(repr, p["reads"])) == sorted(map(repr, r["reads"])) else "different_reads"
                return {"first_out_frame": n, "kind": kind, "plugin": _fmt(p["reads"]), "rewrite": _fmt(r["reads"])}
        pi, ri = {(a, b) for a, b, _ in p["reads"]}, {(a, b) for a, b, _ in r["reads"]}
        if pi != ri:
            kinds = []
            if pi - ri:
                kinds.append("missing_read")
            if ri - pi:
                kinds.append("extra_read")
            return {"first_out_frame": n, "kind": "+".join(kinds), "plugin": _fmt(p["reads"]), "rewrite": _fmt(r["reads"])}
        pt, rt = {(a, c) for a, _, c in p["reads"]}, {(a, c) for a, _, c in r["reads"]}
        if pt != rt:
            return {"first_out_frame": n, "kind": "wrong_time", "plugin": _fmt(p["reads"]), "rewrite": _fmt(r["reads"])}
        if r["params"] is not None and p["params"] != [] and set(p["params"]) != set(r["params"]):
            return {"first_out_frame": n, "kind": "wrong_param_time", "plugin": sorted(map(str, p["params"])),
                    "rewrite": sorted(map(str, r["params"]))}
    return None


def check_calls(case_ids, plugin_dir, rewrite_dir, ordered=False, kinds=None, require_done=True, present_of=None):
    results = []
    for cid in case_ids:
        pp, rp = os.path.join(plugin_dir, cid + ".jsonl"), os.path.join(rewrite_dir, cid + ".jsonl")
        item = {"id": cid}
        if not os.path.exists(pp):
            item.update(state="skipped", reason="no plugin record")
        elif not os.path.exists(rp):
            item.update(state="skipped", reason="no rewrite record")
        else:
            pl, pbad = parse_jsonl(pp)
            rl, rbad = parse_jsonl(rp)
            rew = norm_rewrite(rl)
            plug = norm_plugin(pl, rew["clip_frames"], kinds, None if present_of is None else present_of.get(cid))
            done = plug["done"]
            if rbad or not any(l.get("ev") == "run" for l in rl):
                item.update(state="skipped", reason="rewrite record unreadable")
            elif require_done and (done is None or done.get("ev") != "case_done" or done.get("status") != "rendered"):
                item.update(state="skipped", reason="the original did not render the case (%s)" % (
                    "no case_done" if done is None else (done.get("status") or done.get("ev"))))
            elif not plug["frames"]:
                item.update(state="skipped", reason="the plugin record has no frames")
            else:
                mm = compare_calls(plug, rew, ordered)
                if mm is None:
                    item["state"] = "match"
                else:
                    item["state"] = "mismatch"
                    item.update(mm)
        results.append(item)
    return results


def cmd_rewrite_records(a):
    ctx = Ctx(a)
    vfx = find_bin("vfx", a.vfx)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    cases = ctx.cases()
    if a.every > 1:
        cases = cases[::a.every]
    if a.limit:
        cases = cases[:a.limit]
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for c in cases:
            rec = os.path.join(out, c["id"] + ".jsonl")
            rc = render_case(vfx, ctx.testset, c, os.path.join(tmp, "o"), record=rec)
            if rc != 0:
                bad += 1
            if not os.path.exists(rec):
                raise SystemExit("phase_b: %s wrote no call record for %s (exit %d); this vfx build may predate "
                                 "VFX_CALL_RECORD, use --vfx with a newer build" % (vfx, c["id"], rc))
    print("wrote %d rewrite call records to %s (%d with a nonzero exit)" % (len(cases), out, bad))
    return 0


def cmd_calls(a):
    ctx = Ctx(a)
    ooc = jload(ctx.path("ooc.json"), {})
    moved = {m["id"] for m in ooc.get("moved", []) if m.get("source") == "testset"}
    ids = [c["id"] for c in ctx.cases() if c["id"] not in moved]
    kinds = layer_kinds(a.param_map)
    present_of = {c["id"]: {"input"} | set((c.get("layers") or {}).keys()) for c in ctx.cases()}
    res = check_calls(ids, a.plugin_records, a.rewrite_records, a.ordered, kinds, not a.lenient, present_of)
    compared = [r for r in res if r["state"] in ("match", "mismatch")]
    doc = collections.OrderedDict()
    doc["measured_at"] = datetime.date.today().isoformat()
    doc["compare"] = ("per output frame, as sets of (layer, clip frame) and (layer, time): the pixels the original checks "
                      "out in SMART_RENDER (at the time of their pre-render checkout) against the rewrite's reads"
                      + (", in order" if a.ordered else ""))
    doc["compared"] = len(compared)
    doc["matching"] = sum(1 for r in compared if r["state"] == "match")
    doc["mismatching"] = len(compared) - doc["matching"]
    doc["skipped"] = sum(1 for r in res if r["state"] == "skipped")
    doc["limit"] = "parameter sample times are not recorded by the rewrite, so parameter times are not compared"
    doc["mismatches"] = [r for r in res if r["state"] == "mismatch"]
    doc["skipped_cases"] = [r for r in res if r["state"] == "skipped"]
    jdump(ctx.path("calls.json"), doc)
    with open(ctx.path("calls.tsv"), "w") as fh:
        fh.write("id\tstate\tfirst_out_frame\tkind\treason\n")
        for r in res:
            fh.write("%s\t%s\t%s\t%s\t%s\n" % (r["id"], r["state"], r.get("first_out_frame", ""), r.get("kind", ""), r.get("reason", "")))
    print("call records: %d compared, %d matching, %d mismatching, %d skipped" % (
        doc["compared"], doc["matching"], doc["mismatching"], doc["skipped"]))
    for r in doc["mismatches"][:10]:
        print("  mismatch %s: first differing output frame %s (%s)" % (r["id"], r["first_out_frame"], r["kind"]))
    return 0


# ---------------------------------------------------------------------------------------------
# mutation kills

def compute_kills(ids, ref, base, ctl, muts):
    """ids: in-contract case ids. ref: id -> (size, sha) of the reference (strings). base, ctl: id -> (exit, size,
    sha) of the base rewrite and of the unmutated control build. muts: mutant id -> same kind of table.

    E (the discriminating cases) = in-contract cases whose base output equals the reference. A mutant is killed
    iff on some case of E its output differs from the reference AND from the control build (so a difference the
    build profile alone makes does not count). Returns (E, per mutant dict)."""
    E = [c for c in ids if c in ref and c in base and base[c][0] == "0" and base[c][1:] == ref[c]]
    Es = set(E)
    judge = ctl if ctl else base
    out = {}
    for mid, got in muts.items():
        killing, changed_total, changed_outside, exit_changes, not_run = [], 0, 0, 0, 0
        for c in ids:
            g = got.get(c)
            if g is None:
                not_run += 1
                continue
            j = judge.get(c)
            if g != j:
                changed_total += 1
                if c not in Es:
                    changed_outside += 1
            if c in Es:
                differs_ref = g[0] != "0" or g[1:] != ref[c]
                if differs_ref and g != j:
                    killing.append(c)
                    if g[0] != "0":
                        exit_changes += 1
        out[mid] = {"killing": killing, "changed_total": changed_total, "changed_outside_discriminating": changed_outside,
                    "exit_changes": exit_changes, "not_run": not_run}
    return E, out


def cmd_kills(a):
    ctx = Ctx(a)
    refs = os.path.abspath(a.refs)
    cases = ctx.cases()
    ooc = jload(ctx.path("ooc.json"), {})
    moved = {m["id"] for m in ooc.get("moved", []) if m.get("source") == "testset"}
    cmap = {c["id"]: c for c in cases}
    ids = [c["id"] for c in cases if c["id"] not in moved]
    ref = {}
    for cid in ids:
        p = os.path.join(refs, cmap[cid]["reference"])
        if os.path.isfile(p):
            ref[cid] = (str(os.path.getsize(p)), sha256_file(p))
    base_path = os.path.join(ctx.phase_a, "base.tsv")
    base = read_tsv_rows(base_path)
    mut_dir = os.path.join(ctx.phase_a, "mut")
    info = jload(os.path.join(ctx.phase_a, "mutants_vs_base.json"), {"mutants": []})
    meta = {m["id"]: m for m in info["mutants"]}
    tsvs = {f[:-4] for f in os.listdir(mut_dir) if f.endswith(".tsv")} if os.path.isdir(mut_dir) else set()
    mids = [m["id"] for m in info["mutants"]] if info["mutants"] else sorted(tsvs - {"m00_control"})
    ignored = sorted(tsvs - set(mids) - {"m00_control"})
    ctl = read_tsv_rows(os.path.join(mut_dir, "m00_control.tsv")) if "m00_control" in tsvs else None
    muts = {}
    if a.rerender:
        mutbin = os.path.abspath(a.mutbin)
        E0 = [c for c in ids if c in ref and c in base and base[c][0] == "0" and base[c][1:] == ref[c]]
        tables = {}
        for mid in ["m00_control"] + mids:
            exe = os.path.join(mutbin, mid + ".vfx")
            if os.path.isfile(exe):
                tables[mid] = {c: render_hash(exe, ctx.testset, cmap[c]) for c in E0}
        ctl = tables.pop("m00_control", ctl)
        muts = {m: tables[m] for m in mids if m in tables}
        source = "re-rendered with %s on the discriminating cases only" % mutbin
    else:
        for m in mids:
            if m in tsvs:
                muts[m] = read_tsv_rows(os.path.join(mut_dir, m + ".tsv"))
        source = "the saved mutant tables %s" % mut_dir
    E, per = compute_kills(ids, ref, base, ctl, muts)
    rows = []
    for mid in mids:
        m = meta.get(mid, {})
        r = per.get(mid)
        row = collections.OrderedDict([("id", mid), ("what", m.get("what")), ("file", m.get("file")), ("expect", m.get("expect"))])
        if r is None:
            row.update(status="not_run", killing_count=0, killing_cases=[], reason="no mutant table or binary for it")
        else:
            row["status"] = "killed" if r["killing"] else "survivor"
            row["killing_count"] = len(r["killing"])
            row["killing_cases"] = r["killing"][:a.list_cases]
            row["changed_cases_vs_control"] = r["changed_total"]
            row["changed_where_base_differs_or_no_reference"] = r["changed_outside_discriminating"]
            row["exit_changes"] = r["exit_changes"]
            row["not_run_cases"] = r["not_run"]
            if not r["killing"]:
                row["reason"] = ("changes no output" if r["changed_total"] == 0 else
                                 "changes outputs only on cases where the base rewrite already differs from the reference "
                                 "or no reference exists, which cannot discriminate")
        rows.append(row)
    killed = [r["id"] for r in rows if r["status"] == "killed"]
    survivors = [r["id"] for r in rows if r["status"] != "killed"]
    grade = jload(ctx.path("grade.json"))
    cross = None
    if grade:
        passed = {r["id"] for r in grade["cases"] if r["status"] == "pass"} - moved
        cross = {"hash_equal_not_pass": sorted(set(E) - passed)[:20], "pass_not_hash_equal": sorted(passed - set(E))[:20]}
    doc = collections.OrderedDict()
    doc["measured_at"] = datetime.date.today().isoformat()
    doc["definition"] = ("among the in-contract cases where the base rewrite output equals the reference, a mutant is killed "
                         "iff its output differs from the reference (and from the unmutated control build) on at least one")
    doc["refs"] = refs
    doc["mutant_outputs"] = source
    doc["base_table"] = base_path
    doc["control_table"] = "m00_control.tsv" if ctl else None
    doc["in_contract_cases"] = len(ids)
    doc["cases_with_reference"] = len(ref)
    doc["discriminating_cases"] = len(E)
    doc["total"] = len(rows)
    doc["killed"] = len(killed)
    doc["survivors"] = survivors
    doc["ignored_tables"] = ignored
    doc["grade_crosscheck"] = cross
    doc["mutants"] = rows
    jdump(ctx.path("kills.json"), doc)
    print("mutants: %d of %d killed on %d discriminating cases; survivors: %s" % (
        len(killed), len(rows), len(E), ", ".join(survivors) or "none"))
    return 0


# ---------------------------------------------------------------------------------------------
# summary

def unmet_rule(s):
    """The Verified rule, as tools/docgen/data.py applies it."""
    c, v, cr, mu = s["cases"], s["values"], s["call_records"], s["mutants"]
    out = []
    if c["compared"] < MIN_CASES:
        out.append("fewer than %d cases compared" % MIN_CASES)
    if c["compared"] < c["in_contract"]:
        out.append("not every in-contract case compared")
    if c["differing"] or v["differing_values"] or c["identical"] != c["compared"]:
        out.append("a case or value differs")
    if not cr["compared"] or cr["matching"] != cr["compared"]:
        out.append("call records not all matching")
    if not mu["total"] or mu["killed"] != mu["total"] or mu["survivors"]:
        out.append("a mutant survives or none ran")
    return out


def build_summary(grade, ooc, kills, calls, info, today, extra_blockers=()):
    """grade/ooc/kills/calls: the parsed result files (ooc, kills, calls may be None). info: dict with
    plugin_sha256, commands, fixes, mutants_total_phase_a, candidates_phase_a."""
    moved = {m["id"] for m in (ooc or {}).get("moved", []) if m.get("source") == "testset"}
    rows = [r for r in grade["cases"] if r["id"] not in moved]
    cmp_rows = [r for r in rows if r["status"] in ("pass", "FAIL")]
    ulps = [r["max_ulp"] for r in cmp_rows if r["max_ulp"] is not None]
    if ooc:
        cand_out = ooc["counts"]["candidates_out_of_contract"]
    else:
        cand_out = info.get("candidates_phase_a", 0)
    s = collections.OrderedDict()
    s["module"], s["plugin"], s["plugin_sha256"], s["effect"] = MODULE, PLUGIN, info.get("plugin_sha256", ""), EFFECT
    s["status"] = "in_progress"
    s["blockers"] = []
    s["cases"] = collections.OrderedDict([
        ("in_contract", len(rows)), ("compared", len(cmp_rows)),
        ("identical", sum(1 for r in cmp_rows if r["status"] == "pass")),
        ("differing", sum(1 for r in cmp_rows if r["status"] == "FAIL")),
        ("out_of_contract", len(moved) + cand_out),
        ("logic_parity", sum(1 for r in cmp_rows if r["logic_parity"]))])
    s["values"] = collections.OrderedDict([
        ("bytes_compared", sum(r["bytes_compared"] for r in cmp_rows)),
        ("floats_compared", sum(r["floats_compared"] for r in cmp_rows)),
        ("differing_values", sum(r["differing_values"] for r in cmp_rows)),
        ("max_ulp", max(ulps) if ulps else None)])
    s["call_records"] = collections.OrderedDict([
        ("compared", calls["compared"] if calls else 0), ("matching", calls["matching"] if calls else 0)])
    if kills:
        s["mutants"] = collections.OrderedDict([("total", kills["total"]), ("killed", kills["killed"]),
                                                ("survivors", list(kills["survivors"]))])
    else:
        s["mutants"] = collections.OrderedDict([("total", info.get("mutants_total_phase_a", 0)), ("killed", 0), ("survivors", [])])
    keyed_all = sum(1 for r in grade["cases"] if r["logic_parity"])
    s["stand_ins"] = [STAND_IN_TEXT,
                      "%d keyed-parameter cases are logic parity candidates; %d compared%s" % (
                          keyed_all, s["cases"]["logic_parity"],
                          " (counted in cases.compared, logic parity only, not full parity)" if s["cases"]["logic_parity"] else "")]
    s["fixes"] = info.get("fixes", [])
    s["commands"] = info.get("commands", [])
    s["measured_at"] = today
    noref = sum(1 for r in rows if r["status"] == "no reference")
    blockers = []
    if not cmp_rows:
        blockers.append("emulator side not run yet: no reference compared")
    elif noref:
        blockers.append("%d in-contract cases have no reference" % noref)
    if ooc and ooc["counts"].get("unjudged"):
        blockers.append("%d cases did not render in the harness for reasons that are not the original's (see ooc.json unjudged)" % ooc["counts"]["unjudged"])
    if ooc and ooc["counts"].get("candidates_to_review"):
        blockers.append("%d out-of-contract candidates rendered in the original and need review" % ooc["counts"]["candidates_to_review"])
    blockers += list(extra_blockers)
    unmet = unmet_rule(s)
    if not unmet:
        s["status"] = "verified"
        s["blockers"] = []
    else:
        s["status"] = "blocked" if extra_blockers else "in_progress"
        s["blockers"] = blockers
    return s, unmet


def cmd_summary(a):
    ctx = Ctx(a)
    grade = jload(ctx.path("grade.json"))
    if grade is None:
        raise SystemExit("phase_b: %s does not exist; run `grade` first" % ctx.path("grade.json"))
    ooc, kills, calls = (jload(ctx.path(n)) for n in ("ooc.json", "kills.json", "calls.json"))
    sha_file = os.path.join(ctx.phase_a, "plugin_sha256.txt")
    sha = open(sha_file).read().split()[0] if os.path.exists(sha_file) else ""
    cmds = []
    for p in (os.path.join(ctx.phase_a, "commands.txt"), ctx.path("commands_b.txt")):
        if os.path.exists(p):
            cmds += [l.rstrip("\n") for l in open(p) if l.strip() and l.rstrip("\n") not in cmds]
    old = jload(ctx.path("summary.json")) or jload(os.path.join(ctx.phase_a, "summary.json")) or {}
    mv = jload(os.path.join(ctx.phase_a, "mutants_vs_base.json"), {"mutants": []})
    info = {"plugin_sha256": sha, "commands": cmds, "fixes": old.get("fixes", []),
            "mutants_total_phase_a": len(mv["mutants"]),
            "candidates_phase_a": len(jload(ctx.find("out_of_contract.json"), []))}
    s, unmet = build_summary(grade, ooc, kills, calls, info, datetime.date.today().isoformat(), a.blocker)
    out = os.path.abspath(a.out) if a.out else ctx.path("summary.json")
    jdump(out, s)
    print("%s: status %s; cases %s; values %s; call records %s; mutants %d/%d killed" % (
        out, s["status"], json.dumps(s["cases"]), json.dumps(s["values"]), json.dumps(s["call_records"]),
        s["mutants"]["killed"], s["mutants"]["total"]))
    if unmet:
        print("rule not met: " + "; ".join(unmet))
    return 0


# ---------------------------------------------------------------------------------------------
# triage and the probe ladder

def is_default(key, value):
    if "@" in value:
        return False
    d = DEFAULTS.get(key)
    if d is None:
        return False
    try:
        return float(value) == float(d)
    except ValueError:
        return value == d


def first_key_value(value):
    return value.split(";")[0].split("@")[0]


def ladder_variants(params):
    """[(role, setting, params)] for one case: the case, the default case, each non-default setting reverted to
    its default, the default case with each setting alone, and keyed settings replaced by a constant."""
    pm = parse_params(params)
    fixed = [(k, v) for k, v in pm.items() if k in FIXED]
    settings = [(k, v) for k, v in pm.items() if k not in FIXED and not is_default(k, v)]
    mk = lambda items: [("%s=%s" % kv) for kv in fixed + items]
    out = [("case", "", mk(settings)), ("default", "", mk([]))]
    for k, v in settings:
        out.append(("revert", k, mk([s for s in settings if s[0] != k])))
    for k, v in settings:
        out.append(("only", k, mk([(k, v)])))
    for k, v in settings:
        if "@" in v:
            out.append(("unkey", k, mk([(kk, first_key_value(vv) if kk == k else vv) for kk, vv in settings])))
    seen, uniq = {}, []
    for role, setting, p in out:
        key = tuple(p)
        if key in seen:
            seen[key]["also"].append("%s %s" % (role, setting) if setting else role)
            continue
        item = {"role": role, "setting": setting, "params": p, "also": []}
        seen[key] = item
        uniq.append(item)
    return uniq


def feature_set(case, tags):
    f = ["%s=%s" % kv for kv in tags.items()]
    for k, v in parse_params(case.get("params", [])).items():
        if k in FIXED:
            continue
        f.append("param %s=%s" % (k, v if "@" not in v else "<keyed>"))
        if "@" in v:
            f.append("keyed")
    return sorted(set(f))


def triage_tables(rows, cases_by_id):
    """feature -> [failing, total] over the compared cases."""
    counts = collections.OrderedDict()
    for r in rows:
        if r["status"] not in ("pass", "FAIL"):
            continue
        for f in feature_set(cases_by_id[r["id"]], r["tags"]):
            c = counts.setdefault(f, [0, 0])
            c[1] += 1
            if r["status"] == "FAIL":
                c[0] += 1
    return counts


def smallest_failing(rows, n=10):
    fails = [r for r in rows if r["status"] == "FAIL"]
    return sorted(fails, key=lambda r: (r["width"] * r["height"], r["frames"], r["depth"], r["id"]))[:n]


def write_ladder(ctx, case, out_dir, extra_depths=True):
    os.makedirs(os.path.join(out_dir, "inputs"), exist_ok=True)
    doc = jload(os.path.join(ctx.testset, "cases.json"))
    inputs = {i["file"]: i for i in doc.get("inputs", [])}
    used, cases, ladder = {}, [], []

    def add_input(rel, entry):
        dest = os.path.join(out_dir, rel)
        if not os.path.exists(dest):
            shutil.copyfile(os.path.join(ctx.testset, rel), dest)
        used[rel] = entry

    add_input(case["input"], inputs.get(case["input"]))
    variants = ladder_variants(case.get("params", []))
    for v in variants:
        key = "probe|%s|%s|%s|%s" % (v["role"], v["setting"], ",".join(v["params"]), case["id"])
        cid = "%s.probe-%s-%s.%s-%dx%d.d%d.%08x" % (EFFECT, v["role"], safe(v["setting"] or "all"), case["source"],
                                                    case["width"], case["height"], case["depth"], fnv(key))
        cases.append(dict(case, id=cid, label="probe-" + v["role"], params=v["params"], reference="%s.%s" % (cid, PIX[case["depth"]])))
        ladder.append({"id": cid, "role": v["role"], "setting": v["setting"], "params": v["params"], "also": v["also"]})
    if extra_depths:
        for d in (8, 16, 32):
            if d == case["depth"]:
                continue
            rel = "inputs/%s-%dx%d.d%d.f%d.%s" % (case["source"], case["width"], case["height"], d, case["frames"], PIX[d])
            if os.path.exists(os.path.join(ctx.testset, rel)):
                add_input(rel, inputs.get(rel))
                cid = "%s.probe-depth%d.%s-%dx%d.d%d.%08x" % (EFFECT, d, case["source"], case["width"], case["height"], d,
                                                              fnv("probe|depth|%d|%s" % (d, case["id"])))
                cases.append(dict(case, id=cid, label="probe-depth", depth=d, input=rel, reference="%s.%s" % (cid, PIX[d])))
                ladder.append({"id": cid, "role": "depth", "setting": str(d), "params": case["params"], "also": []})
    jdump(os.path.join(out_dir, "cases.json"), {"format": 1, "ffmpeg": doc.get("ffmpeg", ""),
                                               "inputs": [v for v in used.values() if v], "cases": cases})
    jdump(os.path.join(out_dir, "ladder.json"), {"from_case": case["id"], "probes": ladder})
    return ladder


def ladder_verdict(ladder, rows):
    st = {r["id"]: r["status"] for r in rows}
    lines = []
    for p in ladder["probes"]:
        lines.append("  %-8s %-18s %-12s %s" % (p["role"], p["setting"] or "-", st.get(p["id"], "?"), " ".join(p["params"])))
    fails = {p["role"] + ":" + p["setting"] for p in ladder["probes"] if st.get(p["id"]) == "FAIL"}
    out = []
    only = sorted(s.split(":", 1)[1] for s in fails if s.startswith("only:"))
    if "default:" in fails:
        out.append("the default case already differs: the difference does not depend on any setting (engine, input or depth)")
    if only:
        out.append("differs with only %s set from the default: that setting alone reproduces it" % ", ".join(only))
    nec = sorted(p["setting"] for p in ladder["probes"] if p["role"] == "revert" and st.get(p["id"]) == "pass")
    if nec:
        out.append("reverting %s makes the case pass: it is needed for the difference" % ", ".join(nec))
    if any(p["role"] == "unkey" and st.get(p["id"]) == "pass" for p in ladder["probes"]):
        out.append("the constant (unkeyed) version passes: the keyed curve path (host stand-in) is involved")
    return lines, out


def cmd_triage(a):
    ctx = Ctx(a)
    if a.ladder_report:
        lad = jload(os.path.join(a.probe_dir, "ladder.json"))
        rep = jload(a.ladder_report)
        lines, verdict = ladder_verdict(lad, rep["cases"])
        print("\n".join(lines))
        print("\n".join(verdict) or "no probe differs")
        return 0
    grade = jload(ctx.path("grade.json"))
    if grade is None:
        raise SystemExit("phase_b: %s does not exist; run `grade` first" % ctx.path("grade.json"))
    ooc = jload(ctx.path("ooc.json"), {})
    moved = {m["id"] for m in ooc.get("moved", []) if m.get("source") == "testset"}
    rows = [r for r in grade["cases"] if r["id"] not in moved]
    cases = {c["id"]: c for c in ctx.cases()}
    fails = [r for r in rows if r["status"] == "FAIL"]
    cmp_n = sum(1 for r in rows if r["status"] in ("pass", "FAIL"))
    print("%d compared, %d differing, %d no reference" % (cmp_n, len(fails), sum(1 for r in rows if r["status"] == "no reference")))
    if not fails:
        print("no differing case: nothing to triage")
        return 0
    counts = triage_tables(rows, cases)
    print("\nsettings that co-occur with differences (failing / cases with the setting, most failing first)")
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1][0], -(kv[1][0] / kv[1][1]), kv[0]))
    for f, (nf, nt) in ordered[:a.top]:
        if nf:
            print("  %-44s %4d / %-4d %5.1f%%" % (f, nf, nt, 100.0 * nf / nt))
    every = [f for f, (nf, nt) in counts.items() if nf == len(fails)]
    print("\npresent in every differing case: %s" % (", ".join(every) or "nothing"))
    always = sorted(f for f, (nf, nt) in counts.items() if nf == nt and nt >= 2)
    print("settings where every compared case with it differs: %s" % (", ".join(always[:20]) or "none"))
    small = smallest_failing(rows, a.top)
    print("\nsmallest differing cases (pixels, frames, depth)")
    for r in small:
        print("  %5d px  %4d frames  d%-2d  %s  first failing frame %s  values %d  max ULP %s" % (
            r["width"] * r["height"], r["frames"], r["depth"], r["id"], r["first_failing_frame"], r["differing_values"], r["max_ulp"]))
    case = cases[small[0]["id"]]
    ladder = write_ladder(ctx, case, os.path.abspath(a.probe_dir))
    print("\nladder for %s: %d probe cases written to %s" % (case["id"], len(ladder), os.path.abspath(a.probe_dir)))
    print("run it: python3 ~/twixtor-parity/harness/run_harness.py --testset %s --refs %s-refs -j1 --summary %s-status.json"
          % (a.probe_dir, a.probe_dir, a.probe_dir))
    print("grade it: python3 phase_b.py grade --testset %s --refs %s-refs --results %s-results" % (a.probe_dir, a.probe_dir, a.probe_dir))
    print("read it: python3 phase_b.py triage --probe-dir %s --ladder-report %s-results/grade.json" % (a.probe_dir, a.probe_dir))
    return 0


# ---------------------------------------------------------------------------------------------
# selftest (stand-in oracles, no plugin)

class Checks:
    def __init__(self):
        self.items = []

    def check(self, name, ok, detail=""):
        self.items.append({"name": name, "ok": bool(ok), "detail": detail})
        print("%s  %s%s" % ("ok  " if ok else "FAIL", name, ("  (" + detail + ")") if detail else ""), flush=True)

    @property
    def failed(self):
        return [i for i in self.items if not i["ok"]]


def link_refs(src_dir, dest, cases, skip=()):
    os.makedirs(dest, exist_ok=True)
    for c in cases:
        if c["id"] in skip:
            continue
        d = os.path.join(dest, c["reference"])
        if not os.path.exists(d):
            os.symlink(os.path.abspath(os.path.join(src_dir, c["reference"])), d)


def make_subset(ctx, ids, out_dir):
    """A test set with the given cases that shares the inputs of the full one."""
    os.makedirs(out_dir, exist_ok=True)
    doc = jload(os.path.join(ctx.testset, "cases.json"))
    keep = [c for c in doc["cases"] if c["id"] in set(ids)]
    inp = os.path.join(out_dir, "inputs")
    if not os.path.exists(inp):
        os.symlink(os.path.join(ctx.testset, "inputs"), inp)
    jdump(os.path.join(out_dir, "cases.json"), dict(doc, cases=keep, inputs=[i for i in doc["inputs"] if i["file"] in {c["input"] for c in keep}]))
    shutil.copyfile(os.path.join(ctx.testset, "grid.json"), os.path.join(out_dir, "grid.json")) if os.path.exists(os.path.join(ctx.testset, "grid.json")) else None
    return keep


def perturb_copy(ref_dir, dest_dir, case, how):
    """Replace the symlink for case with a modified real copy. how: bit | negzero | nan."""
    src = os.path.join(ref_dir, case["reference"])
    data = bytearray(open(src, "rb").read())
    detail = ""
    if how == "bit":
        data[len(data) // 2] ^= 0x01
        detail = "bit 0 of byte %d" % (len(data) // 2)
    else:
        words = array("I")
        words.frombytes(bytes(data))
        want = 0x80000000 if how == "negzero" else None
        idx = next((i for i, w in enumerate(words) if (w == 0x80000000 if how == "negzero" else is_nan_bits(w))), None)
        if idx is None:
            return None
        if how == "negzero":
            words[idx] = 0
        else:
            words[idx] ^= 0x1
        data = bytearray(words.tobytes())
        detail = "word %d" % idx
    d = os.path.join(dest_dir, case["reference"])
    if os.path.lexists(d):
        os.remove(d)
    with open(d, "wb") as fh:
        fh.write(bytes(data))
    return detail


def cmd_selftest(a):
    ctx = Ctx(a)
    base_scratch = os.path.abspath(a.scratch or os.path.join(MOD, "results", "selftest"))
    phase_a_results = os.path.abspath(os.path.join(MOD, "results"))
    if os.path.abspath(base_scratch) == phase_a_results:
        raise SystemExit("phase_b: selftest output must not be the phase A results folder")
    base_out = os.path.abspath(a.base_out)
    ck = Checks()
    fx, vfx = find_bin("fx-parity", a.fx_parity), find_bin("vfx", a.vfx)
    cases = ctx.cases()
    cmap = {c["id"]: c for c in cases}
    stages = a.stage.split(",") if a.stage != "all" else ["perfect", "perturbed", "missing", "ooc", "calls"]
    mv = jload(os.path.join(ctx.phase_a, "mutants_vs_base.json"))
    changes = [m for m in mv["mutants"] if m["changed"] > 0]
    expect_killed = {m["id"] for m in changes}
    expect_surv = {m["id"] for m in mv["mutants"]} - expect_killed
    print("phase A: %d mutants, %d change at least one case against the control, %d change nothing" % (
        len(mv["mutants"]), len(changes), len(expect_surv)))
    ref_perfect = os.path.join(base_scratch, "refs_perfect")
    link_refs(base_out, ref_perfect, cases)

    def sub(name):
        d = os.path.join(base_scratch, name)
        os.makedirs(d, exist_ok=True)
        return d

    def ns(**kw):
        d = dict(results=None, phase_a=ctx.phase_a, testset=ctx.testset, fx_parity=fx, vfx=vfx, jobs=1, refs=None)
        d.update(kw)
        return argparse.Namespace(**d)

    if "perfect" in stages:
        rd = sub("perfect")
        cmd_grade(ns(results=rd, refs=ref_perfect))
        g = jload(os.path.join(rd, "grade.json"))["totals"]
        ck.check("perfect: all cases compared", g["compared"] == len(cases), "%d of %d" % (g["compared"], len(cases)))
        ck.check("perfect: 0 differing", g["differing"] == 0 and g["differing_values"] == 0, "%d cases, %d values" % (g["differing"], g["differing_values"]))
        ck.check("perfect: 0 no reference", g["no_reference"] == 0)
        cmd_kills(ns(results=rd, refs=ref_perfect, rerender=False, mutbin=None, list_cases=10))
        k = jload(os.path.join(rd, "kills.json"))
        got_k = {m["id"] for m in k["mutants"] if m["status"] == "killed"}
        ck.check("perfect: killed mutants == phase A mutants that change an output", got_k == expect_killed,
                 "killed %d, phase A %d of %d; survivors %s" % (len(got_k), len(expect_killed), k["total"], ",".join(k["survivors"])))
        ck.check("perfect: survivors == phase A mutants that change nothing", set(k["survivors"]) == expect_surv)
        cmd_summary(ns(results=rd, out=None, blocker=[]))
        s = jload(os.path.join(rd, "summary.json"))
        sum_path = os.path.join(ctx.phase_a, "summary.json")
        old = jload(sum_path)
        old_sha = sha256_file(sum_path) if old else None
        ck.check("perfect: summary has exactly the schema keys", list(s.keys()) == SUMMARY_KEYS and (old is None or list(old.keys()) == SUMMARY_KEYS),
                 ",".join(s.keys()))
        for sec in ("cases", "values", "call_records", "mutants"):
            ck.check("perfect: summary.%s keys match the schema" % sec, list(s[sec].keys()) == SECTION_KEYS[sec]
                     and (old is None or list(old[sec].keys()) == SECTION_KEYS[sec]))
        ck.check("perfect: summary counts", s["cases"]["compared"] == s["cases"]["identical"] == len(cases) and s["cases"]["differing"] == 0
                 and s["mutants"]["killed"] == len(expect_killed), json.dumps(s["cases"]))
        ck.check("perfect: not Verified without call records and with survivors", s["status"] == "in_progress")
        ck.check("perfect: results/summary.json untouched", (sha256_file(sum_path) == old_sha) if old else not os.path.exists(sum_path))

    if "perturbed" in stages:
        rd = sub("perturbed")
        refs = os.path.join(rd, "refs")
        link_refs(base_out, refs, cases)
        by_depth = lambda d: [c for c in cases if c["depth"] == d]
        pick = {}
        # an 8 or 16 bpc case that is not the only discriminating case of some mutant
        pick["bit"] = next(c for c in cases if c["depth"] == 16 and c["width"] * c["height"] > 1 and c["frames"] <= 12)
        detail = {}
        for how, want in (("negzero", 0x80000000), ("nan", None)):
            for c in by_depth(32):
                if c["frames"] > 12 or c["id"] in {x["id"] for x in pick.values()}:
                    continue
                data = open(os.path.join(base_out, c["reference"]), "rb").read()
                ws = array("I")
                ws.frombytes(data)
                if any((w == 0x80000000) if how == "negzero" else is_nan_bits(w) for w in ws):
                    pick[how] = c
                    break
        for how, c in pick.items():
            detail[how] = perturb_copy(base_out, refs, c, how)
        ck.check("perturbed: found a -0 and a NaN to perturb", set(pick) == {"bit", "negzero", "nan"}, ",".join(sorted(pick)))
        ids = {c["id"] for c in pick.values()}
        others = [c["id"] for c in cases if c["id"] not in ids][::12]
        sdir = os.path.join(rd, "subset")
        make_subset(ctx, list(ids) + others, sdir)
        cmd_grade(ns(results=rd, refs=refs, testset=sdir))
        g = jload(os.path.join(rd, "grade.json"))
        fail = {r["id"]: r for r in g["cases"] if r["status"] == "FAIL"}
        ck.check("perturbed: exactly the perturbed cases differ", set(fail) == ids, "%d failing of %d" % (len(fail), len(g["cases"])))
        if "bit" in pick and pick["bit"]["id"] in fail:
            r = fail[pick["bit"]["id"]]
            ck.check("perturbed: bit flip is one differing value", r["differing_values"] == 1, "values %d" % r["differing_values"])
        if "negzero" in pick and pick["negzero"]["id"] in fail:
            r = fail[pick["negzero"]["id"]]
            ck.check("perturbed: -0 against 0 is one differing value with max ULP 0",
                     r["differing_values"] == 1 and r["max_ulp"] == 0 and r["fx_parity"].get("max_ulp") == 0,
                     "values %d, max ULP %s, fx-parity %s" % (r["differing_values"], r["max_ulp"], r["fx_parity"].get("max_ulp")))
        if "nan" in pick and pick["nan"]["id"] in fail:
            r = fail[pick["nan"]["id"]]
            ck.check("perturbed: NaN payload is one differing NaN mismatch",
                     r["differing_values"] == 1 and r["nan_mismatch_values"] == 1 and r["fx_parity"].get("max_ulp") == "nan mismatch",
                     "values %d, nan mismatches %d, fx-parity %s" % (r["differing_values"], r["nan_mismatch_values"], r["fx_parity"].get("max_ulp")))
        # kills with the perturbed references: brute force expectation
        cmd_kills(ns(results=rd, refs=refs, rerender=False, mutbin=None, list_cases=10))
        k = jload(os.path.join(rd, "kills.json"))
        base = read_tsv_rows(os.path.join(ctx.phase_a, "base.tsv"))
        ctl = read_tsv_rows(os.path.join(ctx.phase_a, "mut", "m00_control.tsv"))
        exp = set()
        for m in mv["mutants"]:
            tab = read_tsv_rows(os.path.join(ctx.phase_a, "mut", m["id"] + ".tsv"))
            if any(tab[c] != ctl[c] and (tab[c][0] != "0" or tab[c][1:] != base[c][1:]) for c in tab if c not in ids and base[c][0] == "0"):
                exp.add(m["id"])
        got = {m["id"] for m in k["mutants"] if m["status"] == "killed"}
        ck.check("perturbed: kills equal the brute force expectation", got == exp, "%d killed, expected %d" % (len(got), len(exp)))
        ck.check("perturbed: the 3 perturbed cases are not discriminating", k["discriminating_cases"] == len(cases) - 3, str(k["discriminating_cases"]))
        # a mutant whose only killing cases are perturbed becomes a survivor
        target = min((m for m in changes), key=lambda m: m["changed"])
        rd2 = sub("perturbed_kill")
        refs2 = os.path.join(rd2, "refs")
        link_refs(base_out, refs2, cases)
        tids = set(target["changed_ids"])
        for cid in tids:
            c = cmap[cid]
            if c["depth"] == 32:
                data = bytearray(open(os.path.join(base_out, c["reference"]), "rb").read())
                data[len(data) // 2] ^= 0x01
                d = os.path.join(refs2, c["reference"])
                os.remove(d)
                open(d, "wb").write(bytes(data))
            else:
                perturb_copy(base_out, refs2, c, "bit")
        cmd_kills(ns(results=rd2, refs=refs2, rerender=False, mutbin=None, list_cases=10))
        k2 = jload(os.path.join(rd2, "kills.json"))
        st = {m["id"]: m for m in k2["mutants"]}
        exp2 = set()
        for m in mv["mutants"]:
            tab = read_tsv_rows(os.path.join(ctx.phase_a, "mut", m["id"] + ".tsv"))
            if any(tab[c] != ctl[c] and (tab[c][0] != "0" or tab[c][1:] != base[c][1:]) for c in tab if c not in tids and base[c][0] == "0"):
                exp2.add(m["id"])
        got2 = {m["id"] for m in k2["mutants"] if m["status"] == "killed"}
        ck.check("perturbed: %s (%d killing cases, all perturbed) becomes a survivor" % (target["id"], len(tids)),
                 st[target["id"]]["status"] == "survivor" and got2 == exp2,
                 "reason: %s; killed %d, expected %d" % (st[target["id"]].get("reason"), len(got2), len(exp2)))

    if "missing" in stages:
        rd = sub("missing")
        refs = os.path.join(rd, "refs")
        drop = [c["id"] for c in cases][5:80:7][:10]
        link_refs(base_out, refs, cases, skip=set(drop))
        keep = [c["id"] for c in cases][::15] + drop
        sdir = os.path.join(rd, "subset")
        sub_cases = make_subset(ctx, keep, sdir)
        cmd_grade(ns(results=rd, refs=refs, testset=sdir))
        g = jload(os.path.join(rd, "grade.json"))
        noref = {r["id"] for r in g["cases"] if r["status"] == "no reference"}
        ck.check("missing: dropped references are 'no reference'", noref == set(drop), "%d of %d" % (len(noref), len(drop)))
        ck.check("missing: not counted as pass or failure", g["totals"]["compared"] == len(sub_cases) - len(drop)
                 and g["totals"]["identical"] == g["totals"]["compared"] and g["totals"]["differing"] == 0)
        # the summary over the full list: grade of the subset is enough to show the counts
        s, unmet = build_summary(g, None, None, None, {"candidates_phase_a": 22}, "selftest")
        ck.check("missing: in_contract keeps them, compared does not", s["cases"]["in_contract"] == len(sub_cases)
                 and s["cases"]["compared"] == len(sub_cases) - len(drop))
        ck.check("missing: rule not met (not every in-contract case compared)", "not every in-contract case compared" in unmet)

    if "ooc" in stages:
        rd = sub("ooc")
        rp = os.path.join(base_scratch, "perfect")
        if not os.path.exists(os.path.join(rp, "grade.json")):
            raise SystemExit("phase_b: run the perfect stage first (the ooc stage reuses its grade.json)")
        shutil.copyfile(os.path.join(rp, "grade.json"), os.path.join(rd, "grade.json"))
        ccases, _ = candidate_cases(ctx)
        ids = [c["id"] for c in cases]
        entries = [{"id": ids[3], "status": "refused"}, {"id": ids[10], "status": "refused"},
                   {"id": ids[20], "status": "error", "error": "PluginError: SEQUENCE_SETUP returned PF_Err 5"},
                   {"id": ids[30], "status": "timeout"},
                   {"id": ids[40], "status": "error", "error": "GuestFault: read of unmapped memory"},
                   {"id": ids[50], "status": "unsupported", "error": "Bezier handles"},
                   {"id": ids[60], "status": "not_run"}, {"id": ids[70], "status": "rendered"},
                   {"id": ccases[0]["id"], "status": "refused"}, {"id": ccases[1]["id"], "status": "rendered"},
                   {"id": "retime.unknown.id", "status": "rendered"}]
        sf = os.path.join(rd, "status.json")
        jdump(sf, {"testset": ctx.testset, "cases": entries, "boots": []})
        cmd_ooc(ns(results=rd, status=[sf], ooc_status=None, param_map=os.path.join(MOD, "..", "harness", "param_map.tsv")))
        o = jload(os.path.join(rd, "ooc.json"))
        moved = {m["id"] for m in o["moved"]}
        ck.check("ooc: refused, PluginError and timeout leave the in-contract count",
                 moved == {ids[3], ids[10], ids[20], ids[30]} and o["counts"]["in_contract"] == len(cases) - 4, str(o["counts"]))
        ck.check("ooc: emulator error, unsupported and not_run stay in contract as unjudged",
                 {u["id"] for u in o["unjudged"]} == {ids[40], ids[50], ids[60]})
        ck.check("ooc: a candidate the original rendered is flagged for review", o["counts"]["candidates_to_review"] == 1
                 and o["counts"]["candidates_out_of_contract"] == len(ccases) - 1, str(o["counts"]))
        ck.check("ooc: unknown ids are listed", o["unknown_ids"] == ["retime.unknown.id"])
        cmd_summary(ns(results=rd, out=None, blocker=[]))
        s = jload(os.path.join(rd, "summary.json"))
        ck.check("ooc: summary in_contract/compared/out_of_contract",
                 s["cases"]["in_contract"] == len(cases) - 4 and s["cases"]["compared"] == len(cases) - 4
                 and s["cases"]["out_of_contract"] == 4 + len(ccases) - 1, json.dumps(s["cases"]))
        cmd_ooc_set(ns(results=rd, out=os.path.join(rd, "ooc-testset")))

    if "calls" in stages:
        rd = sub("calls")
        rec = os.path.join(rd, "rewrite")
        cmd_rewrite_records(ns(results=rd, out=rec, every=a.every, limit=0, vfx=a.record_vfx or vfx))
        sel = [c for c in cases if os.path.exists(os.path.join(rec, c["id"] + ".jsonl"))]
        ck.check("calls: rewrite records written", len(sel) > 0, "%d records" % len(sel))
        plug = os.path.join(rd, "plugin")
        os.makedirs(plug, exist_ok=True)
        multi = None
        for c in sel:
            lines, _ = parse_jsonl(os.path.join(rec, c["id"] + ".jsonl"))
            pl = plugin_lines_from_rewrite(lines, c["id"])
            write_jsonl(os.path.join(plug, c["id"] + ".jsonl"), pl)
            if multi is None and any(sum(1 for l in lines if l.get("ev") == "read" and l["out_frame"] == n) >= 2 for n in range(c["frames"])):
                multi = c["id"]
        res = check_calls([c["id"] for c in sel], plug, rec)
        ck.check("calls: every synthetic plugin record matches the rewrite record", all(r["state"] == "match" for r in res), "%d cases" % len(res))
        mut = collect_mutations(plug, rec, sel, multi)
        for name, ok, detail in mut:
            ck.check("calls: " + name, ok, detail)
        cmd_calls(ns(results=rd, plugin_records=plug, rewrite_records=rec, ordered=False, lenient=False,
                     param_map=os.path.join(MOD, "..", "harness", "param_map.tsv")))
        c = jload(os.path.join(rd, "calls.json"))
        ck.check("calls: calls.json counts", c["compared"] == len(sel) and c["matching"] == c["compared"], "%d compared, %d matching" % (c["compared"], c["matching"]))

    jdump(os.path.join(base_scratch, "selftest.json"), {"measured_at": datetime.date.today().isoformat(), "stages": stages, "checks": ck.items,
                                                       "failed": len(ck.failed)})
    print("\nselftest: %d checks, %d failed" % (len(ck.items), len(ck.failed)))
    return 1 if ck.failed else 0


def write_jsonl(path, lines):
    with open(path, "w") as fh:
        for l in lines:
            fh.write(json.dumps(l) + "\n")


def plugin_lines_from_rewrite(lines, cid, clip_frames=None, smart=True):
    """A plugin record in the harness format that agrees with a rewrite record (used to test the checker)."""
    run = next(l for l in lines if l["ev"] == "run")
    out = [{"seq": 1, "ev": "render_path", "smart": smart}]
    seq = 1
    for l in lines:
        seq += 1
        if l["ev"] == "frame":
            out.append({"seq": seq, "ev": "frame", "out_frame": l["out_frame"], "time": l["time"], "time_step": l["time_step"],
                        "time_scale": l["time_scale"]})
        elif l["ev"] == "read":
            ff = l["index"] if l["index"] is not None else l["clip_frames"]
            out.append({"seq": seq, "ev": "cb", "fn": "prerender_cb.checkout_layer", "args": [], "ret": 0,
                        "param_index": 0 if l["layer"] == "input" else 1, "checkout_id": seq, "time": l["time"],
                        "time_step": run["time_step"], "time_scale": run["time_scale"], "footage_frame": ff,
                        "inside": l["index"] is not None, "out_frame": l["out_frame"]})
    out.append({"seq": seq + 1, "ev": "case_done", "id": cid, "status": "rendered"})
    return out


def collect_mutations(plug, rec, sel, multi):
    """Break copies of plugin records and check the checker names the right first frame and kind."""
    out = []
    tmp = tempfile.mkdtemp(prefix="calls_mut_")
    try:
        cand = None
        for c in sel:
            if c["frames"] < 6:
                continue
            lines, _ = parse_jsonl(os.path.join(plug, c["id"] + ".jsonl"))
            by_frame = collections.defaultdict(list)
            for i, l in enumerate(lines):
                if l.get("ev") == "cb":
                    by_frame[l["out_frame"]].append(i)
            for n in sorted(by_frame):
                if n < 2:
                    continue
                idx = by_frame[n]
                ffs = [lines[i]["footage_frame"] for i in idx]
                unique = [i for i in idx if ffs.count(lines[i]["footage_frame"]) == 1 and lines[i]["inside"]]
                free = [f for f in range(c["frames"]) if f not in ffs]
                if unique and free:
                    cand = (c, lines, unique[0], free[0], n)
                    break
            if cand:
                break
        if cand is None:
            return [("a case with a read to mutate exists", False, "")]
        c, lines, i, free_ff, n = cand

        def run(name, edit, kind, frame):
            ls = [dict(l) for l in lines]
            edit(ls)
            d = os.path.join(tmp, name.replace(" ", "_"))
            os.makedirs(d, exist_ok=True)
            write_jsonl(os.path.join(d, c["id"] + ".jsonl"), ls)
            r = check_calls([c["id"]], d, rec)[0]
            ok = r["state"] == "mismatch" and r["first_out_frame"] == frame and kind in r["kind"]
            out.append((name, ok, "%s, first frame %s, kind %s (expected frame %s, kind %s)" % (
                r["state"], r.get("first_out_frame"), r.get("kind"), frame, kind)))

        run("a read the rewrite does not make is reported (extra_read)", lambda ls: ls.pop(i), "extra_read", n)
        run("a read the plugin makes and the rewrite lacks is reported (missing_read)",
            lambda ls: ls.insert(i, dict(ls[i], footage_frame=free_ff)), "missing_read", n)
        run("a wrong time is reported", lambda ls: ls[i].update(time=ls[i]["time"] + 1), "wrong_time", n)
        run("a wrong clip frame is reported", lambda ls: ls[i].update(footage_frame=free_ff), "missing_read", n)
        run("a wrong frame time is reported", lambda ls: [l.update(time=l["time"] + 1) for l in ls if l.get("ev") == "frame" and l["out_frame"] == n],
            "wrong_frame_time", n)
        ls = [l for l in lines if l.get("ev") != "case_done"]
        d = os.path.join(tmp, "nodone")
        os.makedirs(d, exist_ok=True)
        write_jsonl(os.path.join(d, c["id"] + ".jsonl"), ls)
        r = check_calls([c["id"]], d, rec)[0]
        out.append(("a plugin record that never finished is skipped, not compared", r["state"] == "skipped", r.get("reason", "")))
        if multi:
            ml, _ = parse_jsonl(os.path.join(plug, multi + ".jsonl"))
            idx = [k for k, l in enumerate(ml) if l.get("ev") == "cb"]
            f0 = ml[idx[0]]["out_frame"]
            same = [k for k in idx if ml[k]["out_frame"] == f0]
            if len(same) >= 2:
                ml[same[0]], ml[same[1]] = ml[same[1]], ml[same[0]]
                d = os.path.join(tmp, "order")
                os.makedirs(d, exist_ok=True)
                write_jsonl(os.path.join(d, multi + ".jsonl"), ml)
                as_sets = check_calls([multi], d, rec)[0]["state"]
                ordered = check_calls([multi], d, rec, ordered=True)[0]
                out.append(("reordered reads match as sets", as_sets == "match", as_sets))
                out.append(("reordered reads mismatch when order matters",
                            ordered["state"] == "mismatch" and ordered["kind"] == "wrong_order",
                            "%s %s" % (ordered["state"], ordered.get("kind"))))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


# ---------------------------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, testset=True):
        p.add_argument("--results", help="outputs and phase B inputs (default results/)")
        p.add_argument("--phase-a", help="phase A files: base.tsv, mut/, mutants_vs_base.json, ... (default results/)")
        if testset:
            p.add_argument("--testset", help="the test set (default testset/, else the case files in results/)")

    def bins(p):
        p.add_argument("--fx-parity")
        p.add_argument("--vfx")

    p = sub.add_parser("grade", help="fx-parity run --max-ulp 0 --json -> grade.json, grade.tsv")
    common(p); bins(p)
    p.add_argument("--refs", required=True)
    p.add_argument("--jobs", type=int, default=1)
    p.set_defaults(fn=cmd_grade)

    p = sub.add_parser("ooc-set", help="write the candidates as a small test set the harness can run")
    common(p)
    p.add_argument("--out", default=os.path.join(MOD, "ooc-testset"))
    p.set_defaults(fn=cmd_ooc_set)

    p = sub.add_parser("ooc", help="harness per-case status -> ooc.json")
    common(p)
    p.add_argument("--status", action="append", required=True, help="run_harness.py --summary file (repeatable)")
    p.add_argument("--ooc-status", help="comma list of classes that move a case out of contract "
                   "(default refused,timeout,plugin_error; also emulator_error, unsupported, not_run)")
    p.add_argument("--param-map", default=os.path.join(MOD, "..", "harness", "param_map.tsv"))
    p.set_defaults(fn=cmd_ooc)

    p = sub.add_parser("rewrite-records", help="run the rewrite with VFX_CALL_RECORD over the test set")
    common(p); bins(p)
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=cmd_rewrite_records)

    p = sub.add_parser("calls", help="call-record check")
    common(p)
    p.add_argument("--plugin-records", required=True)
    p.add_argument("--rewrite-records", required=True)
    p.add_argument("--ordered", action="store_true", help="compare the reads of a frame in order")
    p.add_argument("--lenient", action="store_true", help="compare plugin records that have no case_done line")
    p.add_argument("--param-map", default=os.path.join(MOD, "..", "harness", "param_map.tsv"))
    p.set_defaults(fn=cmd_calls)

    p = sub.add_parser("kills", help="mutation kills -> kills.json")
    common(p)
    p.add_argument("--refs", required=True)
    p.add_argument("--rerender", action="store_true", help="re-render with mutbin/*.vfx instead of the saved tables")
    p.add_argument("--mutbin", default=os.path.join(MOD, "mutbin"))
    p.add_argument("--list-cases", type=int, default=40, help="killing case ids listed per mutant")
    p.set_defaults(fn=cmd_kills)

    p = sub.add_parser("summary", help="summary.json from the result files")
    common(p, testset=False)
    p.add_argument("--out")
    p.add_argument("--blocker", action="append", default=[], help="a blocker outside the module (status blocked)")
    p.set_defaults(fn=cmd_summary)

    p = sub.add_parser("triage", help="group the differing cases, write a probe ladder")
    common(p)
    p.add_argument("--probe-dir", default=os.path.join(MOD, "probe"))
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--ladder-report", help="grade.json of the ladder run: print which probes differ")
    p.set_defaults(fn=cmd_triage)

    p = sub.add_parser("selftest", help="stand-in oracles")
    common(p); bins(p)
    p.add_argument("--stage", default="all", help="perfect,perturbed,missing,ooc,calls or all")
    p.add_argument("--scratch", help="default results/selftest")
    p.add_argument("--base-out", default=os.path.join(MOD, "base-out"))
    p.add_argument("--every", type=int, default=6, help="calls stage: every Nth case")
    p.add_argument("--record-vfx", help="calls stage: a vfx build that writes VFX_CALL_RECORD (default --vfx)")
    p.set_defaults(fn=cmd_selftest)
    return ap


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    a = build_parser().parse_args(argv)
    if a.cmd != "selftest":
        log_command(Ctx(a), argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
