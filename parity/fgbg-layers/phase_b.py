#!/usr/bin/env python3
"""Phase B tooling of the fgbg-layers parity module: comparison, out-of-contract judging, call
records, mutation kills, summary, triage and a self test, all on stand-in references until the
emulator renders. Standard library only.

  phase_b.py grade   --refs DIR [--vfx BIN] [--fx-parity BIN] [--jobs N]
  phase_b.py ooc     --status FILE [--status FILE ...] [--param-map FILE] [--also CATEGORY,...] [--make-testset DIR]
  phase_b.py calls   --plugin-records DIR --rewrite-records DIR [--param-map FILE] [--ordered]
  phase_b.py record-rewrite --vfx BIN --out DIR [--only ID,ID]
  phase_b.py kills   --refs DIR [--mutants-dirs mutants2,mutants] [--rebuild --tree SRC_MUT --ids M01,..]
  phase_b.py summary
  phase_b.py triage  [--probe-dir probe] [--probe-refs DIR]
  phase_b.py selftest

Every subcommand takes --results DIR (default results) and --testset DIR (default testset), both
relative to this file. Nothing here runs a plugin. The bar: --max-ulp 0 (bytes exact at 8 and 16 bpc,
floats bit for bit at 32 bpc, -0 against 0 and differing NaN bits count as different).
"""
from __future__ import annotations

import argparse
import array
import collections
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = "fgbg-layers"
EFFECT = "retime-fgbg"
PLUGIN_FILE = "TwixtorPro_64.aex"
PLUGIN_NAME = "TwixtorPro_64"
PIX = {8: "rgba", 16: "rgba64le", 32: "gbrapf32le"}
BYTES = {8: 1, 16: 2, 32: 4}
MIN_COMPARED = 150
PREFIX = "python3 phase_b.py"

# the stated-rule gaps of the root README's Limits table that reach retime-fgbg (titles for display)
GAP_TITLES = {
    "G1": "matte shrink tail",
    "G1b": "mask-aware block matching cost",
    "G1c": "layer visibility direction",
    "G1f": "blur tail flow of a layered render",
    "G1g": "hole fill of a single group display",
    "G1h": "fast path group display",
    "G2": "tracking point displacement",
    "G3": "Draw Geom drawing",
    "G5": "second tracker arithmetic",
    "G6": "solver, hole fill and blur wrap variants (track_rgb_a)",
}
# grid block -> gaps its cases were built for (gen_grid.py section comments)
BLOCK_GAPS = {
    "shrink": ["G1"], "fast": ["G1h"], "sens": ["G1b"], "blur": ["G1f"], "order": ["G1c"],
    "points": ["G2"], "geom": ["G3"], "wrap": ["G6"], "display": ["G1g"], "count": ["G1c"],
}
OOC_DEFAULT = ("refused", "plugin_error", "param_unregistered", "out_of_range")
OOC_OPTIONAL = ("timeout", "emulator_fault", "harness_limit", "not_run")


class ToolError(Exception):
    pass


# ---------------------------------------------------------------------------------- small helpers

def here(p):
    return p if os.path.isabs(p) else os.path.join(HERE, p)


def read_json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(p, obj):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")


def read_text(p):
    with open(p, encoding="utf-8") as fh:
        return fh.read()


def read_tsv(p):
    with open(p, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    cols = lines[0].split("\t")
    return [dict(zip(cols, ln.split("\t"))) for ln in lines[1:] if ln]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def today():
    return datetime.date.today().isoformat()


def load_cases(testset):
    return read_json(os.path.join(testset, "cases.json"))["cases"]


def load_grid(testset):
    out = {}
    for r in read_tsv(os.path.join(testset, "grid.tsv")):
        out[r["id"]] = (r.get("block", ""), [t for t in (r.get("tags") or "").split(",") if t])
    return out


def moved_out_ids(results):
    p = os.path.join(results, "ooc.json")
    if os.path.exists(p):
        return {e["id"] for e in read_json(p).get("moved_out", [])}
    return set()


def find_tool(explicit, env, rels, names):
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get(env):
        cands.append(os.environ[env])
    cands += [here(r) for r in rels]
    for n in names:
        w = shutil.which(n)
        if w:
            cands.append(w)
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return os.path.abspath(c)
    raise ToolError("cannot find %s (tried %s); pass it on the command line" % (names[0], ", ".join(cands) or "nothing"))


def find_fx_parity(explicit=None):
    return find_tool(explicit, "FX_PARITY", ["../harness/repo/target/release/fx-parity", "src/target/release/fx-parity",
                                            "../../target/release/fx-parity", "../../target/release/fx-parity.exe"], ["fx-parity"])


def find_vfx(explicit=None):
    return find_tool(explicit, "VFX_PATH", ["bin/vfx_base"], ["vfx"])


def vfx_args(vfx, testset, c, out):
    a = [vfx, "apply", c["effect"], "--width", str(c["width"]), "--height", str(c["height"]), "--depth", str(c["depth"]),
         "--fps", c["fps"], "--frames", str(c["frames"]), "--in", os.path.join(testset, c["input"]), "--out", out]
    for p in c.get("params", []):
        a += ["--param", p]
    for lid, rel_path in sorted((c.get("layers") or {}).items()):
        a += ["--layer", "%s=%s" % (lid, os.path.join(testset, rel_path))]
    return a


def render_case(vfx, testset, c, out, record=None):
    env = dict(os.environ)
    env.pop("VFX_CALL_RECORD", None)
    if record:
        env["VFX_CALL_RECORD"] = record
    r = subprocess.run(vfx_args(vfx, testset, c, out), capture_output=True, text=True, env=env)
    err = (r.stderr.strip().splitlines() or [""])[-1][:200] if r.returncode else ""
    return r.returncode, err


def input_entries(cases, base_dir):
    """The `inputs` list fx-parity's test set format wants: every clip file the cases name, once."""
    seen = {}
    for c in cases:
        for f, src in [(c["input"], c["source"])] + [(f, "matte") for f in (c.get("layers") or {}).values()]:
            if f in seen:
                continue
            m = re.search(r"-(\d+)x(\d+)\.d(\d+)(?:\.f(\d+))?\.", os.path.basename(f))
            w, h, d = (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (c["width"], c["height"], c["depth"])
            path = os.path.join(base_dir, f)
            if os.path.exists(path):
                frames = os.path.getsize(path) // (w * h * 4 * BYTES[d])
            else:
                frames = int(m.group(4)) if m and m.group(4) else c["frames"]
            seen[f] = {"file": f, "source": src, "width": w, "height": h, "depth": d, "pix_fmt": PIX[d], "frames": frames}
    return [seen[k] for k in sorted(seen)]


def ensure_fxparity_set(testset, dest):
    """fx-parity wants an `inputs` list in cases.json. The phase A cases.json has none and must stay as
    it is, so a copy with the list and a link to the same inputs is made under `dest`."""
    doc = read_json(os.path.join(testset, "cases.json"))
    if "inputs" in doc:
        return testset
    os.makedirs(dest, exist_ok=True)
    doc = dict(doc, inputs=input_entries(doc["cases"], testset))
    write_json(os.path.join(dest, "cases.json"), doc)
    link = os.path.join(dest, "inputs")
    if not os.path.exists(link):
        try:
            os.symlink(os.path.abspath(os.path.join(testset, "inputs")), link, target_is_directory=True)
        except OSError:
            shutil.copytree(os.path.join(testset, "inputs"), link)
    return dest


def run_fx_parity(fxp, testset, refs, vfx, max_ulp=0, jobs=1):
    cmd = [fxp, "run", "--testset", testset, "--refs", refs, "--max-ulp", str(max_ulp), "--json", "--jobs", str(jobs)]
    if vfx:
        cmd += ["--vfx", vfx]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode not in (0, 1):
        raise ToolError("fx-parity run failed with exit %d: %s" % (p.returncode, (p.stderr.strip() or p.stdout.strip())[:400]))
    try:
        return json.loads(p.stdout), cmd, p.returncode
    except ValueError:
        raise ToolError("fx-parity run printed no JSON: %s" % (p.stdout[:200] or p.stderr[:200]))


# --------------------------------------------------------------------------------- float and value diff

def f32_ordered(bits):
    return -(bits & 0x7FFFFFFF) if bits & 0x80000000 else bits


def f32_is_nan(bits):
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def ulp_distance(a_bits, b_bits):
    """fx-parity's distance: -0 and 0 share a value, None when either side is NaN."""
    if f32_is_nan(a_bits) or f32_is_nan(b_bits):
        return None
    return abs(f32_ordered(a_bits) - f32_ordered(b_bits))


def _words(code, data):
    a = array.array(code)
    a.frombytes(data)
    if sys.byteorder == "big":
        a.byteswap()
    return a


def sample_diff(depth, ref, out):
    """Whole-clip comparison of two raw clips (same layout as the output), bit for bit.

    Returns values (compared), differing, max_ulp (finite distances at 32 bpc, else None),
    nan_mismatches (differing values where either side is NaN) and malformed (sizes differ)."""
    bps = BYTES[depth]
    if len(ref) != len(out):
        n = len(out) // bps
        return {"values": n, "differing": n, "max_ulp": None, "nan_mismatches": 0, "malformed": True}
    n = len(out) // bps
    if depth == 8:
        d = sum(1 for x, y in zip(ref, out) if x != y)
        return {"values": n, "differing": d, "max_ulp": None, "nan_mismatches": 0, "malformed": False}
    if depth == 16:
        ra, oa = _words("H", ref), _words("H", out)
        d = sum(1 for x, y in zip(ra, oa) if x != y)
        return {"values": n, "differing": d, "max_ulp": None, "nan_mismatches": 0, "malformed": False}
    ra, oa = _words("I", ref), _words("I", out)
    d, nan, mx = 0, 0, None
    for x, y in zip(ra, oa):
        if x == y:
            continue
        d += 1
        u = ulp_distance(x, y)
        if u is None:
            nan += 1
        else:
            mx = u if mx is None else max(mx, u)
    return {"values": n, "differing": d, "max_ulp": mx, "nan_mismatches": nan, "malformed": False}


# --------------------------------------------------------------------------------------------- tags

def stand_in_reasons(case, tags, which=("keyframes", "matte-layers")):
    r = []
    if "keyframes" in which and ("standin" in tags or any("@" in p.partition("=")[2] for p in case.get("params", []))):
        r.append("keyframes")
    if "matte-layers" in which and case.get("layers"):
        r.append("matte-layers")
    return r


def gaps_of(case, block):
    g = set(BLOCK_GAPS.get(block, []))
    params = {p.partition("=")[0]: p.partition("=")[2] for p in case.get("params", [])}

    def f(v, d=0.0):
        try:
            return float(v.split("@")[0])
        except ValueError:
            return d

    for k, v in params.items():
        if k.endswith("_inv_matte_shrink") and (f(v) != 0.0 or v.startswith("-") or "@" in v):
            g.add("G1")
        if re.fullmatch(r"pt\d+_use", k) and f(v, 1) != 1:
            g.add("G2")
        if k == "draw_geom" and f(v, 1) != 1:
            g.add("G3")
        if k == "track_rgb_a" and f(v) != 0:
            g.add("G6")
        if k.endswith("_sensitivity") and "layers" in case and case.get("layers"):
            g.add("G1b")
        if k == "blur_compensation" and f(v) != 0 and case.get("layers"):
            g.add("G1f")
        if k == "display" and f(v, 1) in (3, 4, 5):
            g.add("G1g")
    if len(case.get("layers") or {}) >= 2:
        g.add("G1c")
    return sorted(g)


# ------------------------------------------------------------------------------------------ param map

def load_param_map(path, plugin=PLUGIN_NAME):
    if not path or not os.path.exists(path):
        return []
    return [r for r in read_tsv(path) if r.get("plugin") == plugin]


def plugin_side(rows):
    return any(r.get("ae_index") for r in rows)


def registered_by_id(rows):
    return {r["vfx_id"]: r for r in rows if r.get("vfx_id") and r.get("ae_index") and r.get("match") != "unmatched"}


def unmatched_params(cases, rows):
    """Rewrite ids the grid uses that the param map does not pair with a registered plugin param.
    None when the map has no plugin side yet (nothing can be judged)."""
    if not plugin_side(rows):
        return None
    reg = registered_by_id(rows)
    out = collections.defaultdict(lambda: {"cases": 0, "why": ""})
    for c in cases:
        ids = [p.partition("=")[0] for p in c.get("params", [])] + list((c.get("layers") or {}).keys())
        for i in ids:
            if i in reg:
                continue
            rows_i = [r for r in rows if r.get("vfx_id") == i]
            out[i]["cases"] += 1
            out[i]["why"] = "no row in the param map" if not rows_i else "unmatched: the plugin registered no counterpart"
    return dict(out)


def _num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def range_violations(case, reg):
    """Parameters whose value falls outside what the plugin registers (valid min/max, popup choices)."""
    out = []
    for p in case.get("params", []):
        pid, _, text = p.partition("=")
        row = reg.get(pid)
        if row is None:
            continue
        kind = row.get("vfx_kind") or ""
        for key in text.split(";"):
            v = key.partition("@")[0]
            if kind in ("float", "int", "angle"):
                lo, hi, x = _num(row.get("ae_min")), _num(row.get("ae_max")), _num(v)
                if x is None:
                    continue
                if x != x or (lo is not None and x < lo) or (hi is not None and x > hi):
                    out.append("%s=%s outside the registered %s..%s" % (pid, v, row.get("ae_min") or "-", row.get("ae_max") or "-"))
                    break
            elif kind == "popup":
                n = len([c for c in (row.get("ae_choices") or "").split("|") if c])
                x = _num(v)
                if n and x is not None and not (1 <= x <= n):
                    out.append("%s=%s outside the registered popup 1..%d" % (pid, v, n))
                    break
    return out


# ----------------------------------------------------------------------------------------- grade

def cmd_grade(a):
    results, testset, refs = here(a.results), here(a.testset), os.path.abspath(a.refs)
    fxp, vfx = find_fx_parity(a.fx_parity), find_vfx(a.vfx)
    cases = {c["id"]: c for c in load_cases(testset)}
    grid = load_grid(testset)
    fx_set = ensure_fxparity_set(testset, os.path.join(results, "gradeset"))
    rep, cmd, code = run_fx_parity(fxp, fx_set, refs, vfx, 0, a.jobs)
    ent, tot = [], collections.Counter()
    max_ulp, tmp = None, tempfile.mkdtemp(prefix="pb-grade-")
    try:
        for r in rep["cases"]:
            c = cases.get(r["id"])
            if c is None:
                continue
            block, tags = grid.get(c["id"], ("", []))
            reasons = stand_in_reasons(c, tags)
            e = {"id": c["id"], "status": r["status"], "block": block, "tags": tags, "depth": c["depth"],
                 "width": c["width"], "height": c["height"], "frames": c["frames"], "logic_parity": bool(reasons),
                 "stand_ins": reasons, "differing_values": None, "max_ulp": None, "nan_mismatches": 0,
                 "differing_pixels": r.get("clip_differing_pixels"), "first_failing_frame": r.get("frame"),
                 "fx_parity_max_ulp": r.get("max_ulp"), "error": r.get("error")}
            if r["status"] == "no reference":
                e["result"] = "no_reference"
                tot["no_reference"] += 1
            else:
                per = c["width"] * c["height"] * 4 * c["frames"]
                tot["compared"] += 1
                if c["depth"] == 32:
                    tot["floats_compared"] += per
                else:
                    tot["bytes_compared"] += per * BYTES[c["depth"]]
                if r["status"] == "pass":
                    e["result"] = "identical"
                    e["differing_values"] = 0
                    tot["identical"] += 1
                else:
                    e["result"] = "differing"
                    tot["differing"] += 1
                    out = os.path.join(tmp, "out.raw")
                    rc, err = render_case(vfx, testset, c, out)
                    ref_path = os.path.join(refs, c["reference"])
                    if rc != 0 or not os.path.exists(out) or not os.path.exists(ref_path):
                        raise ToolError("cannot recompute the differing values of %s: vfx exit %d %s" % (c["id"], rc, err))
                    with open(ref_path, "rb") as fh:
                        ref = fh.read()
                    with open(out, "rb") as fh:
                        got = fh.read()
                    os.remove(out)
                    d = sample_diff(c["depth"], ref, got)
                    e.update(differing_values=d["differing"], max_ulp=d["max_ulp"], nan_mismatches=d["nan_mismatches"])
                    if d["malformed"]:
                        e["error"] = (e["error"] or "") + " reference has %d bytes, the output %d" % (len(ref), len(got))
                    if d["differing"] == 0:
                        e["error"] = ((e["error"] or "") + " fx-parity says FAIL but no value differs bit for bit").strip()
                    tot["differing_values"] += d["differing"]
                    tot["nan_mismatches"] += d["nan_mismatches"]
                    if d["max_ulp"] is not None:
                        max_ulp = d["max_ulp"] if max_ulp is None else max(max_ulp, d["max_ulp"])
                if reasons:
                    tot["logic_parity_compared"] += 1
            ent.append(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    totals = {"cases": len(ent), "compared": tot["compared"], "identical": tot["identical"], "differing": tot["differing"],
              "no_reference": tot["no_reference"], "bytes_compared": tot["bytes_compared"], "floats_compared": tot["floats_compared"],
              "differing_values": tot["differing_values"], "max_ulp": max_ulp, "nan_mismatches": tot["nan_mismatches"],
              "logic_parity_compared": tot["logic_parity_compared"]}
    doc = {"module": MODULE, "measured_at": today(), "max_ulp_bar": 0, "refs": refs, "testset": testset, "fx_parity_testset": fx_set, "vfx": vfx,
           "fx_parity": {"path": fxp, "exit": code, "summary": rep.get("summary")},
           "command": "%s grade --refs %s" % (PREFIX, a.refs), "totals": totals, "cases": ent}
    write_json(os.path.join(results, "grade.json"), doc)
    cols = ["id", "status", "differing_values", "max_ulp", "block", "tags", "logic_parity", "depth", "size", "frames",
            "differing_pixels", "error"]
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "grade.tsv"), "w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for e in ent:
            mu = "nan" if e["nan_mismatches"] and e["max_ulp"] is None else ("-" if e["max_ulp"] is None else str(e["max_ulp"]))
            row = [e["id"], e["status"], "-" if e["differing_values"] is None else str(e["differing_values"]), mu, e["block"],
                   ",".join(e["tags"]), "yes" if e["logic_parity"] else "no", str(e["depth"]),
                   "%dx%d" % (e["width"], e["height"]), str(e["frames"]),
                   "-" if e["differing_pixels"] is None else str(e["differing_pixels"]), (e["error"] or "").replace("\t", " ")]
            fh.write("\t".join(row) + "\n")
    print("grade: %(compared)d compared, %(identical)d identical, %(differing)d differing (%(differing_values)d values, "
          "max ULP %(max_ulp)s, %(nan_mismatches)d NaN mismatches), %(no_reference)d no reference" % totals)
    return 0


# ------------------------------------------------------------------------------------------------ ooc

def load_status(paths):
    """Per-case status from run_harness.py --summary JSON ({"cases": [...]}) or worker JSON lines."""
    out = {}
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            text = fh.read()
        try:
            doc = json.loads(text)
            items = doc.get("cases", doc) if isinstance(doc, dict) else doc
            if isinstance(items, dict):
                items = [dict(v, id=k) if isinstance(v, dict) else {"id": k, "status": v} for k, v in items.items()]
        except ValueError:
            items = []
            for ln in text.splitlines():
                try:
                    o = json.loads(ln)
                except ValueError:
                    continue
                if isinstance(o, dict) and (o.get("event") == "done" or ("id" in o and "status" in o)):
                    items.append(o)
        for it in items:
            if isinstance(it, dict) and "id" in it:
                out[it["id"]] = it
    return out


def classify_original(entry):
    st, err = entry.get("status"), entry.get("error") or ""
    if st == "rendered":
        return "rendered"
    if st == "refused":
        return "refused"
    if st == "unsupported":
        if "no counterpart among the registered" in err or "no registered layer param" in err:
            return "param_unregistered"
        return "harness_limit"
    if st == "error":
        return "plugin_error" if err.startswith("PluginError") else "emulator_fault"
    if st == "timeout":
        return "timeout"
    if st == "not_run":
        return "not_run"
    return "unknown"


def load_judged(paths):
    """The out of contract judgements of named cases: TSV files with the header `id<TAB>reason<TAB>evidence`, one case
    per line (lines starting with # are comments). Returns id -> {"reason", "evidence", "file"}; a later file wins."""
    out = {}
    for p in paths or []:
        lines = [ln for ln in read_text(p).splitlines() if ln.strip() and not ln.startswith("#")]
        if not lines or lines[0].split("\t")[0] != "id":
            raise ToolError("ooc --judged %s: the first line must be the header id<TAB>reason<TAB>evidence" % p)
        for ln in lines[1:]:
            f = ln.split("\t")
            if len(f) < 2 or not f[1].strip():
                raise ToolError("ooc --judged %s: a line without a reason: %s" % (p, ln[:80]))
            out[f[0]] = {"reason": f[1], "evidence": f[2] if len(f) > 2 else "", "file": p}
    return out


def judge(cases, grid, status, rows, base_rows, ooc_cands, ooc_rows, move_categories, judged=None):
    """Pure core of `ooc`: returns the ooc.json document body. `judged` (id -> reason, evidence) moves the named cases
    out of contract with category `judged` (after the range check, before the harness status categories)."""
    reg = registered_by_id(rows)
    has_plugin = plugin_side(rows)
    judged = judged or {}
    moved, unresolved, rendered = [], [], 0
    counts = collections.Counter()
    for c in cases:
        e = status.get(c["id"])
        reasons, cat = [], None
        orig = None
        if e is not None:
            cat = classify_original(e)
            orig = {"status": e.get("status"), "error": e.get("error"), "category": cat}
            counts["status:" + str(e.get("status"))] += 1
        rv = range_violations(c, reg) if has_plugin else []
        move_cat = None
        if rv:
            move_cat, reasons = "out_of_range", rv
        elif c["id"] in judged:
            move_cat, reasons = "judged", [judged[c["id"]]["reason"], judged[c["id"]]["evidence"]]
        elif cat in move_categories:
            move_cat, reasons = cat, [(e.get("error") or e.get("status") or "")[:200]]
        if move_cat:
            b = base_rows.get(c["id"], {})
            block, tags = grid.get(c["id"], ("", []))
            moved.append({"id": c["id"], "block": block, "tags": tags, "category": move_cat, "reasons": reasons,
                          "review": move_cat == "param_unregistered",
                          "original": orig, "rewrite": {"exit": b.get("exit"), "bytes": b.get("bytes"), "error": b.get("error")}})
        elif cat == "rendered":
            rendered += 1
        elif cat is not None:
            unresolved.append({"id": c["id"], "category": cat, "original": orig})
    cands, promote = [], []
    for c in ooc_cands:
        o = ooc_rows.get(c["id"], {})
        e = status.get(c["id"])
        cat = classify_original(e) if e is not None else None
        rv = range_violations(c, reg) if has_plugin else []
        if cat is None:
            verdict = "unjudged"
        elif rv:
            verdict = "out_of_contract"
        elif cat == "rendered":
            verdict = "original_rendered_review"
            promote.append(c["id"])
        elif cat in move_categories:
            verdict = "out_of_contract"
        else:
            verdict = "unresolved"
        cands.append({"id": c["id"], "phase_a_reason": c.get("reason"), "range_violations": rv, "verdict": verdict,
                      "original": None if e is None else {"status": e.get("status"), "error": e.get("error"), "category": cat},
                      "rewrite": {"exit": o.get("exit"), "bytes": o.get("bytes"), "error": o.get("error") or c.get("vfx_verdict")}})
    kept = [x for x in cands if x["verdict"] in ("unjudged", "out_of_contract")]
    counts_out = {"moved_out": len(moved), "unresolved": len(unresolved), "rendered_in_contract": rendered,
                  "candidates": len(cands), "candidates_out": len(kept), "candidates_promote_review": len(promote),
                  "out_of_contract_total": len(moved) + len(kept),
                  "moved_by_category": dict(collections.Counter(m["category"] for m in moved)),
                  "status_counts": {k: v for k, v in counts.items()}}
    ids = {c["id"] for c in cases}
    return {"moved_out": moved, "unresolved": unresolved, "candidates": cands, "counts": counts_out,
            "param_map_plugin_side": has_plugin, "judged_unknown": sorted(k for k in judged if k not in ids)}


def cmd_ooc(a):
    results, testset = here(a.results), here(a.testset)
    if a.make_testset:
        cands = read_json(os.path.join(testset, "out_of_contract.json"))["cases"]
        d = os.path.abspath(a.make_testset)
        os.makedirs(os.path.join(d, "inputs"), exist_ok=True)
        keep = ("id", "effect", "label", "params", "depth", "width", "height", "source", "input", "reference", "kind", "frames", "fps", "layers")
        outc, missing = [], []
        for c in cands:
            outc.append({k: c[k] for k in keep if k in c})
            for f in [c["input"]] + list((c.get("layers") or {}).values()):
                dst = os.path.join(d, f)
                if os.path.exists(dst):
                    continue
                if os.path.exists(os.path.join(testset, f)):
                    shutil.copy(os.path.join(testset, f), dst)
                elif f not in missing:
                    missing.append(f)
        if missing:
            print("ooc: warning: %d input files are not in %s: %s" % (len(missing), testset, ", ".join(missing[:5])))
        write_json(os.path.join(d, "cases.json"), {"format": 1, "ffmpeg": "none (raw inputs written by gen_grid.py)", "inputs": input_entries(outc, d),
                                                   "note": "out of contract candidates of %s, for the harness to try" % MODULE, "cases": outc})
        print("ooc: wrote %s with %d candidate cases" % (d, len(outc)))
        if not a.status:
            return 0
    if not a.status:
        raise ToolError("ooc needs --status FILE (or --make-testset DIR)")
    status = load_status([os.path.abspath(p) for p in a.status])
    pm = a.param_map or here("../harness/param_map.tsv")
    rows = load_param_map(pm)
    cases = load_cases(testset)
    grid = load_grid(testset)
    base_p = here(a.base)
    base_rows = {r["id"]: r for r in read_tsv(base_p)} if os.path.exists(base_p) else {}
    cand_p = os.path.join(testset, "out_of_contract.json")
    cands = read_json(cand_p)["cases"] if os.path.exists(cand_p) else []
    ooc_p = here(a.ooc_rewrite)
    ooc_rows = {r["id"]: r for r in read_tsv(ooc_p)} if os.path.exists(ooc_p) else {}
    cats = set(OOC_DEFAULT) | {x for x in (a.also or "").split(",") if x}
    jfiles = [os.path.abspath(p) for p in (getattr(a, "judged", None) or [])]
    doc = judge(cases, grid, status, rows, base_rows, cands, ooc_rows, cats, load_judged(jfiles))
    doc.update(module=MODULE, measured_at=today(), param_map=os.path.abspath(pm) if os.path.exists(pm) else None,
               status_files=[os.path.abspath(p) for p in a.status], categories_moved_out=sorted(cats), judged_files=jfiles,
               command="%s ooc --status %s%s%s" % (PREFIX, " --status ".join(a.status), (" --also " + a.also) if a.also else "",
                                                   "".join(" --judged " + p for p in (getattr(a, "judged", None) or []))))
    write_json(os.path.join(results, "ooc.json"), doc)
    c = doc["counts"]
    print("ooc: %d moved out of contract %s, %d unresolved (stay in contract, no reference), %d rendered; candidates: %d out/unjudged, %d to review"
          % (c["moved_out"], c["moved_by_category"], c["unresolved"], c["rendered_in_contract"], c["candidates_out"], c["candidates_promote_review"]))
    return 0


# ----------------------------------------------------------------------------------------- calls

def read_jsonl(p):
    out, bad = [], 0
    with open(p, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except ValueError:
                bad += 1
    return out, bad


def fps_of(text):
    n, _, d = str(text).partition("/")
    return Fraction(int(n), int(d or 1))


def clip_frame_counts(case, testset):
    """Frames in the input and in each layer clip, from file sizes."""
    per = case["width"] * case["height"] * 4 * BYTES[case["depth"]]
    out = {}
    for lid, f in [("input", case["input"])] + sorted((case.get("layers") or {}).items()):
        p = os.path.join(testset, f)
        out[lid] = os.path.getsize(p) // per if os.path.exists(p) else None
    return out


def _tf(time, scale):
    return Fraction(int(time), int(scale)) if scale else Fraction(int(time))


def normalize_plugin(lines, rows, case=None, clips=None):
    """Plugin record (harness JSON lines) to per output frame frames, layer reads and param reads.

    Layer reads (agent C3, 2026-10-31): the SMART_RENDER pixel checkouts (`smart_cb.checkout_layer_pixels`), each
    resolved through its checkout id to the pre-render checkout of the same output frame (layer, footage frame,
    time); the rewrite's `read` lines are pixel reads too (`frame_at`). A record without any pixel checkout (the
    classic path, synthetic records) uses the pre-render checkouts as before. `inter.checkout_param` on a layer
    param is a read on both paths. With a `case`, a layer the case does not set (neither the input nor one of
    case["layers"]) is left out: the harness answers it with size 0 (owner decision), it has no pixels, and the
    rewrite has no such clip to read; its checkouts are counted in notes (`unset_layer_checkouts`,
    `unset_layer_pixel_reads`), not compared."""
    by_index = {int(r["ae_index"]): r for r in rows if r.get("ae_index")}
    frames, reads, params = {}, collections.defaultdict(list), collections.defaultdict(list)
    pre, pix, ids, has_pixels = collections.defaultdict(list), collections.defaultdict(list), {}, False
    set_layers = None if case is None else {"input"} | set(case.get("layers") or {})
    cur, smart, status, notes = None, None, None, collections.Counter()
    for ln in lines:
        ev = ln.get("ev")
        if ev == "frame":
            cur = ln["out_frame"]
            frames[cur] = {"time": ln["time"], "time_step": ln.get("time_step"), "time_scale": ln.get("time_scale")}
            ids = {}
        elif ev == "render_path":
            smart = ln.get("smart")
        elif ev == "case_done":
            status = ln.get("status")
        elif ev == "cb":
            fn = ln.get("fn", "")
            if ln.get("unimplemented"):
                notes["unimplemented:" + fn] += 1
            if fn.endswith("checkout_layer_pixels"):
                has_pixels = True
                hit = ids.get(ln.get("checkout_id"))
                if hit is None:
                    notes["pixel_read_without_checkout"] += 1
                elif set_layers is not None and hit[1][0] not in set_layers:
                    notes["unset_layer_pixel_reads"] += 1
                else:
                    pix[cur if cur is not None else hit[0]].append(hit[1])
                continue
            if fn not in ("prerender_cb.checkout_layer", "inter.checkout_param"):
                continue
            idx = ln.get("param_index")
            of = ln.get("out_frame", cur)
            if idx is None or of is None or "time" not in ln:
                notes["setup_or_unattributed"] += 1
                continue
            row = by_index.get(int(idx))
            is_layer = fn == "prerender_cb.checkout_layer" or int(idx) == 0 or (row is not None and row.get("ae_type") == "LAYER")
            t = _tf(ln["time"], ln.get("time_scale"))
            if is_layer:
                name = "input" if int(idx) == 0 else ((row or {}).get("vfx_id") or "param#%s" % idx)
                inside = ln.get("inside")
                fr = ln.get("footage_frame")
                if inside is None:
                    n = (clips or {}).get(name)
                    inside = fr is not None and fr >= 0 and (n is None or fr < n)
                item = (name, fr if inside else None, t, ln["time"])
                if fn == "prerender_cb.checkout_layer" and ln.get("checkout_id") is not None:
                    ids[ln["checkout_id"]] = (of, item)
                if set_layers is not None and name not in set_layers:
                    notes["unset_layer_checkouts"] += 1
                    continue
                (pre if fn == "prerender_cb.checkout_layer" else reads)[of].append(item)
            else:
                params[of].append(((row or {}).get("vfx_id") or "param#%s" % idx, t, ln["time"]))
    for of, items in (pix if has_pixels else pre).items():
        reads[of].extend(items)
    if smart is False and case is not None:
        fps = fps_of(case["fps"])
        names = ["input"] + [lid for lid in sorted(case.get("layers") or {}) if lid in {r.get("vfx_id") for r in rows}]
        for f, fi in frames.items():
            t = _tf(fi["time"], fi["time_scale"])
            fr = int((t * fps).__floor__())
            for name in names:
                n = (clips or {}).get(name)
                reads[f].append((name, fr if 0 <= fr and (n is None or fr < n) else None, t, fi["time"]))
        notes["implicit_classic_reads"] = len(names)
    return {"frames": frames, "reads": dict(reads), "params": dict(params), "smart": smart, "status": status, "notes": dict(notes),
            "read_source": "pixel checkouts" if has_pixels else "pre-render checkouts"}


def normalize_rewrite(lines):
    frames, reads, params, run = {}, collections.defaultdict(list), collections.defaultdict(list), None
    for ln in lines:
        ev = ln.get("ev")
        if ev == "run":
            run = ln
        elif ev == "frame":
            frames[ln["out_frame"]] = {"time": ln["time"], "time_step": ln.get("time_step"), "time_scale": ln.get("time_scale")}
        elif ev == "read":
            of = ln["out_frame"]
            scale = (frames.get(of) or {}).get("time_scale") or (run or {}).get("time_scale")
            reads[of].append((ln["layer"], ln.get("index"), _tf(ln["time"], scale), ln["time"]))
        elif ev == "param":
            of = ln["out_frame"]
            scale = (frames.get(of) or {}).get("time_scale") or (run or {}).get("time_scale")
            params[of].append((ln["param"], _tf(ln["time"], scale), ln["time"]))
    return {"frames": frames, "reads": dict(reads), "params": dict(params), "run": run}


def _show(k):
    return {"layer": k[0], "index": k[1], "seconds": "%d/%d" % (k[2].numerator, k[2].denominator)}


def classify_diff(missing, extra):
    """Pair the plugin-only (missing) and rewrite-only (extra) reads into named differences."""
    missing, extra, out = list(missing), list(extra), []
    for kind, pred in (("wrong_time", lambda m, e: m[0] == e[0] and m[1] == e[1] and m[2] != e[2]),
                       ("wrong_frame_index", lambda m, e: m[0] == e[0] and m[2] == e[2] and m[1] != e[1]),
                       ("wrong_layer", lambda m, e: m[0] != e[0] and m[1] == e[1] and m[2] == e[2])):
        for m in list(missing):
            for e in list(extra):
                if pred(m, e):
                    out.append({"kind": kind, "plugin": _show(m), "rewrite": _show(e)})
                    missing.remove(m)
                    extra.remove(e)
                    break
    out += [{"kind": "missing_read", "plugin": _show(m)} for m in missing]
    out += [{"kind": "extra_read", "rewrite": _show(e)} for e in extra]
    return out


def compare_calls(p, r, ordered=False):
    """Per output frame; first differing frame reported. p and r come from normalize_*."""
    fr = sorted(set(p["frames"]) | set(r["frames"]))
    for f in fr:
        pf, rf = p["frames"].get(f), r["frames"].get(f)
        if pf is None or rf is None:
            return {"status": "mismatch", "frames_compared": 0,
                    "first_mismatch": {"out_frame": f, "kind": "frame_only_in_" + ("rewrite" if pf is None else "plugin")}}
        if _tf(pf["time"], pf["time_scale"]) != _tf(rf["time"], rf["time_scale"]):
            return {"status": "mismatch", "first_mismatch": {"out_frame": f, "kind": "frame_time", "plugin": pf, "rewrite": rf}}
        pr, rr = p["reads"].get(f, []), r["reads"].get(f, [])
        pk, rk = [x[:3] for x in pr], [x[:3] for x in rr]
        diffs = []
        if ordered:
            if pk != rk:
                ps, rs = set(pk), set(rk)
                diffs = classify_diff(ps - rs, rs - ps) or [{"kind": "order", "plugin": [_show(k) for k in pk[:6]], "rewrite": [_show(k) for k in rk[:6]]}]
        else:
            ps, rs = set(pk), set(rk)
            if ps != rs:
                diffs = classify_diff(ps - rs, rs - ps)
        if r["params"] or any(r["params"].values()):
            pp = {(x[0], x[1]) for x in p["params"].get(f, [])}
            rp = {(x[0], x[1]) for x in r["params"].get(f, [])}
            if pp != rp:
                diffs += [{"kind": "param_time_missing", "param": x[0], "seconds": "%d/%d" % (x[1].numerator, x[1].denominator)} for x in sorted(pp - rp)]
                diffs += [{"kind": "param_time_extra", "param": x[0], "seconds": "%d/%d" % (x[1].numerator, x[1].denominator)} for x in sorted(rp - pp)]
        if diffs:
            return {"status": "mismatch", "first_mismatch": {"out_frame": f, "differences": diffs}}
    return {"status": "match", "frames_compared": len(fr)}


def cmd_calls(a):
    results, testset = here(a.results), here(a.testset)
    pdir, rdir = os.path.abspath(a.plugin_records), os.path.abspath(a.rewrite_records)
    rows = load_param_map(a.param_map or here("../harness/param_map.tsv"))
    cases = {c["id"]: c for c in load_cases(testset)}
    skip_ids = moved_out_ids(results)
    pfiles = {f[:-6] for f in os.listdir(pdir) if f.endswith(".jsonl") and not f.startswith("_")}
    rfiles = {f[:-6] for f in os.listdir(rdir) if f.endswith(".jsonl") and not f.startswith("_")}
    out, skipped = [], []
    for cid in sorted(pfiles & rfiles):
        if cid not in cases:
            continue
        if cid in skip_ids:
            skipped.append({"id": cid, "reason": "out of contract"})
            continue
        c = cases[cid]
        pl, bad_p = read_jsonl(os.path.join(pdir, cid + ".jsonl"))
        rl, bad_r = read_jsonl(os.path.join(rdir, cid + ".jsonl"))
        pn = normalize_plugin(pl, rows, c, clip_frame_counts(c, testset))
        if pn["status"] != "rendered" or not pn["frames"]:
            skipped.append({"id": cid, "reason": "plugin record incomplete (case status %s, %d frames)" % (pn["status"], len(pn["frames"]))})
            continue
        rn = normalize_rewrite(rl)
        if not rn["frames"]:
            skipped.append({"id": cid, "reason": "rewrite record has no frame lines"})
            continue
        res = compare_calls(pn, rn, a.ordered)
        res.update(id=cid, plugin_param_reads=sum(len(v) for v in pn["params"].values()), plugin_notes=pn["notes"],
                   plugin_read_source=pn["read_source"], unreadable_lines=bad_p + bad_r)
        out.append(res)
    matching = sum(1 for x in out if x["status"] == "match")
    doc = {"module": MODULE, "measured_at": today(), "ordered": bool(a.ordered), "plugin_records": pdir, "rewrite_records": rdir,
           "compared": len(out), "matching": matching, "mismatching": [x["id"] for x in out if x["status"] != "match"],
           "skipped": skipped, "only_plugin": sorted(pfiles - rfiles), "only_rewrite": sorted(rfiles - pfiles),
           "definition": "per output frame, the sets of layer reads (layer, clip frame index or none outside the clip, time in "
                         "seconds): plugin side the SMART_RENDER pixel checkouts resolved through their pre-render checkout "
                         "(pre-render checkouts when a record has no pixel checkout), rewrite side the VFX_CALL_RECORD read lines; "
                         "layers the case does not set are left out (the harness answers them with size 0, they have no pixels)",
           "limits": ["parameter sample times are not recorded on the rewrite side: plugin param reads are counted, not compared",
                      "a classic (non smart) render path has its layers handed over in params[]; those reads are added as implicit reads at each frame time"],
           "command": "%s calls --plugin-records %s --rewrite-records %s%s" % (PREFIX, a.plugin_records, a.rewrite_records, " --ordered" if a.ordered else ""),
           "cases": out}
    write_json(os.path.join(results, "calls.json"), doc)
    print("calls: %d compared, %d matching, %d skipped" % (len(out), matching, len(skipped)))
    for x in out:
        if x["status"] != "match":
            print("  mismatch %s: %s" % (x["id"], json.dumps(x["first_mismatch"])[:300]))
    return 0


def cmd_record_rewrite(a):
    testset, out = here(a.testset), os.path.abspath(a.out)
    vfx = find_vfx(a.vfx)
    with open(vfx, "rb") as fh:
        if b"VFX_CALL_RECORD" not in fh.read():
            raise ToolError("%s does not know VFX_CALL_RECORD; pass a vfx built from the working tree" % vfx)
    cases = load_cases(testset)
    if a.only:
        keep = set(a.only.split(","))
        cases = [c for c in cases if c["id"] in keep]
    os.makedirs(out, exist_ok=True)
    base = {r["id"]: r for r in read_tsv(here(a.base))} if os.path.exists(here(a.base)) else {}
    tmp = os.path.join(out, "_out.raw")
    bad, differs, n = [], [], 0
    for c in cases:
        rec = os.path.join(out, c["id"] + ".jsonl")
        rc, err = render_case(vfx, testset, c, tmp, rec)
        if rc != 0:
            bad.append({"id": c["id"], "exit": rc, "error": err})
            continue
        n += 1
        b = base.get(c["id"])
        if b and b.get("sha256") != sha256_file(tmp):
            differs.append(c["id"])
        os.remove(tmp)
    write_json(os.path.join(out, "_record_run.json"), {"vfx": vfx, "recorded": n, "failed": bad, "output_differs_from_base": differs})
    print("record-rewrite: %d recorded, %d failed, %d with output different from results/base.tsv" % (n, len(bad), len(differs)))
    return 0 if not bad else 1


# ----------------------------------------------------------------------------------------- kills

def touch_newer(path, reference=None):
    """Give a file an mtime strictly newer than `reference` (a file) and than now."""
    t = time.time()
    if reference and os.path.exists(reference):
        t = max(t, os.path.getmtime(reference) + 1.0)
    os.utime(path, (t, t))


def rebuild_mutant(tree, m, out_dir, jobs=1, cargo="cargo"):
    """Apply one mutant in `tree`, build vfx, copy the binary, restore the file. The mutated and the
    restored file are both touched newer than the last build output, so cargo cannot reuse the
    previous mutant's code."""
    path = os.path.join(tree, m["file"])
    with open(path, "rb") as fh:
        orig = fh.read()
    old, new = m["old"].encode(), m["new"].encode()
    if orig.count(old) != m["count"]:
        return {"id": m["id"], "status": "anchor-mismatch", "found": orig.count(old), "expected": m["count"]}
    built = os.path.join(tree, "target", "release", "vfx")
    env = dict(os.environ, CARGO_BUILD_JOBS=str(jobs))
    env["PATH"] = os.path.expanduser("~/.cargo/bin") + os.pathsep + env.get("PATH", "")
    try:
        with open(path, "wb") as fh:
            fh.write(orig.replace(old, new))
        touch_newer(path, built)
        r = subprocess.run((cargo if isinstance(cargo, list) else [cargo]) + ["build", "--release", "--bin", "vfx", "-j%d" % jobs], cwd=tree, env=env, capture_output=True, text=True)
    finally:
        with open(path, "wb") as fh:
            fh.write(orig)
        touch_newer(path, built)
    if r.returncode != 0:
        return {"id": m["id"], "status": "build-failed", "log": r.stderr[-1500:]}
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, m["id"] + ".vfx")
    shutil.copy(built, dst)
    return {"id": m["id"], "status": "built", "binary": dst, "binary_sha256": sha256_file(dst)}


def mutant_hashes(mid, dirs, results):
    """(source, {case: (exit, sha)} or {case: None} for changed-only) of the freshest saved outputs."""
    for d in dirs:
        p = os.path.join(d, mid, "results.tsv")
        if os.path.exists(p):
            return "results.tsv:" + d, {r["id"]: (r.get("exit"), r.get("sha256")) for r in read_tsv(p)}
    for d in dirs:
        p = os.path.join(d, mid, "changed.txt")
        if os.path.exists(p):
            return "changed.txt:" + d, {x: None for x in read_text(p).split()}
    p = os.path.join(results, "mutants", mid + ".changed.txt")
    if os.path.exists(p):
        return "results/mutants", {x: None for x in read_text(p).split()}
    return None, None


def kill_analysis(muts, cases, base, refs_sha, mutant_data):
    """Pure kill logic. base: id -> (exit, sha); refs_sha: id -> sha of the reference or None;
    mutant_data: mid -> (source, table) where table is id -> (exit, sha) or id -> None (changed only).
    A case discriminates when the base output equals the reference. Mutant killed iff its output
    differs from the reference on at least one discriminating case."""
    ids = [c["id"] for c in cases]
    disc = [i for i in ids if refs_sha.get(i) and base.get(i) and base[i][0] == "0" and base[i][1] == refs_sha[i]]
    base_differs = [i for i in ids if refs_sha.get(i) and not (base.get(i) and base[i][0] == "0" and base[i][1] == refs_sha[i])]
    out = []
    for m in muts:
        src, table = mutant_data.get(m["id"], (None, None))
        e = {"id": m["id"], "what": m.get("what"), "file": m.get("file"), "source": src, "status": None,
             "killing_cases": [], "via_render_failure": [], "changed_vs_base": None}
        if table is None:
            e["status"] = "undecided"
            e["note"] = "no saved mutant output"
            out.append(e)
            continue
        full = any(v is not None for v in table.values())
        if full:
            changed = [i for i in ids if i in table and table[i] != base.get(i)]
        else:
            changed = [i for i in ids if i in table]
        e["changed_vs_base"] = len(changed)
        cs = set(changed)
        for i in disc:
            if i not in cs:
                continue
            t = table.get(i)
            if t is not None and t[0] != "0":
                e["via_render_failure"].append(i)
            else:
                e["killing_cases"].append(i)
        if e["killing_cases"]:
            e["status"] = "killed"
        elif not changed:
            e["status"] = "survivor"
            e["note"] = "changes no output of the base rewrite on the grid"
        elif not disc:
            e["status"] = "undecided"
            e["note"] = "no case has a reference equal to the base output yet"
        elif e["via_render_failure"]:
            e["status"] = "survivor"
            e["note"] = "differs from the references only by failing to render on %d cases" % len(e["via_render_failure"])
        else:
            e["status"] = "survivor"
            e["note"] = "changes %d cases but none of them has a reference equal to the base output" % len(changed)
        out.append(e)
    return {"discriminating_cases": len(disc), "base_differs_from_reference": base_differs, "mutants": out}


def cmd_kills(a):
    results, testset = here(a.results), here(a.testset)
    refs = os.path.abspath(a.refs)
    moved = moved_out_ids(results)
    cases = [c for c in load_cases(testset) if c["id"] not in moved]
    base = {r["id"]: (r.get("exit"), r.get("sha256")) for r in read_tsv(here(a.base))}
    muts = read_json(here(a.mutants))
    if a.ids:
        muts = [m for m in muts if m["id"] in a.ids.split(",")]
    refs_sha = {}
    for c in cases:
        p = os.path.join(refs, c["reference"])
        refs_sha[c["id"]] = sha256_file(p) if os.path.exists(p) else None
    dirs = [here(d) for d in a.mutants_dirs.split(",") if d]
    if a.rebuild:
        if not a.tree:
            raise ToolError("--rebuild needs --tree SRC_MUT")
        rb = os.path.join(results, "mutants_rebuilt")
        vfx_tmp = os.path.join(rb, "_bin")
        for m in muts:
            r = rebuild_mutant(os.path.abspath(a.tree), m, vfx_tmp)
            print("rebuild %s: %s" % (m["id"], r["status"]))
            if r["status"] != "built":
                continue
            d = os.path.join(rb, m["id"])
            os.makedirs(d, exist_ok=True)
            subprocess.run([sys.executable, here("render_grid.py"), "--vfx", r["binary"], "--testset", testset, "--out", os.path.join(d, "results.tsv"), "--jobs", "1"], check=True)
            os.remove(r["binary"])
        dirs = [rb] + dirs
    data = {m["id"]: mutant_hashes(m["id"], dirs, results) for m in muts}
    ka = kill_analysis(muts, cases, base, refs_sha, data)
    ms = ka["mutants"]
    totals = {"total": len(ms), "killed": sum(1 for m in ms if m["status"] == "killed"),
              "survivors": [m["id"] for m in ms if m["status"] == "survivor"],
              "undecided": [m["id"] for m in ms if m["status"] == "undecided"]}
    doc = {"module": MODULE, "measured_at": today(), "refs": refs, "cases_with_reference": sum(1 for v in refs_sha.values() if v),
           "discriminating_cases": ka["discriminating_cases"], "base_differs_from_reference": ka["base_differs_from_reference"],
           "totals": totals, "mutants": ms,
           "command": "%s kills --refs %s%s" % (PREFIX, a.refs, " --rebuild --tree %s" % a.tree if a.rebuild else "")}
    write_json(os.path.join(results, "kills.json"), doc)
    print("kills: %d of %d mutants killed (%d discriminating cases of %d with a reference); survivors %s; undecided %d"
          % (totals["killed"], totals["total"], ka["discriminating_cases"], doc["cases_with_reference"], totals["survivors"], len(totals["undecided"])))
    return 0


# --------------------------------------------------------------------------------------------- summary

def unmet(s):
    c, v, k, m = s["cases"], s["values"], s["call_records"], s["mutants"]
    out = []
    if c["compared"] < MIN_COMPARED:
        out.append("fewer than %d cases compared" % MIN_COMPARED)
    if c["compared"] < c["in_contract"]:
        out.append("not every in-contract case compared")
    if c["differing"] or v["differing_values"] or c["identical"] != c["compared"]:
        out.append("a case or value differs")
    if not k["compared"] or k["matching"] != k["compared"]:
        out.append("call records not all matching")
    if not m["total"] or m["killed"] != m["total"] or m["survivors"]:
        out.append("a mutant survives or none ran")
    return out


def build_summary(template, cases, grid, grade, ooc, kills, calls, mutants_built, candidates_n, fixes, which_stand_ins=("keyframes", "matte-layers")):
    moved = {e["id"] for e in ooc["moved_out"]} if ooc else set()
    ge = {e["id"]: e for e in grade["cases"]}
    inc = [c["id"] for c in cases if c["id"] not in moved]
    cmp_ = [ge[i] for i in inc if i in ge and ge[i]["result"] in ("identical", "differing")]
    ident = [e for e in cmp_ if e["result"] == "identical"]
    diff = [e for e in cmp_ if e["result"] == "differing"]
    noref = [i for i in inc if i not in ge or ge[i]["result"] == "no_reference"]
    ulps = [e["max_ulp"] for e in diff if e.get("max_ulp") is not None]
    nan_total = sum(e.get("nan_mismatches") or 0 for e in diff)
    floats = sum(e["width"] * e["height"] * 4 * e["frames"] for e in cmp_ if e["depth"] == 32)
    byts = sum(e["width"] * e["height"] * 4 * e["frames"] * BYTES[e["depth"]] for e in cmp_ if e["depth"] != 32)
    lp = sum(1 for e in cmp_ if e["logic_parity"])
    oc_n = ooc["counts"]["out_of_contract_total"] if ooc else candidates_n
    if calls:
        cin = [x for x in calls["cases"] if x["id"] not in moved]
        call = {"compared": len(cin), "matching": sum(1 for x in cin if x["status"] == "match")}
    else:
        call = {"compared": 0, "matching": 0}
    if kills:
        mut = {"total": kills["totals"]["total"], "killed": kills["totals"]["killed"], "survivors": list(kills["totals"]["survivors"])}
    else:
        mut = {"total": mutants_built, "killed": 0, "survivors": []}
    keyed = sum(1 for c in cases if stand_in_reasons(c, grid.get(c["id"], ("", []))[1], ("keyframes",)))
    layered = sum(1 for c in cases if c.get("layers"))
    s = collections.OrderedDict()
    s["module"], s["plugin"] = MODULE, template.get("plugin", PLUGIN_FILE)
    s["plugin_sha256"], s["effect"] = template.get("plugin_sha256"), EFFECT
    s["status"] = "in_progress"
    s["blockers"] = []
    s["cases"] = collections.OrderedDict([("in_contract", len(inc)), ("compared", len(cmp_)), ("identical", len(ident)),
                                          ("differing", len(diff)), ("out_of_contract", oc_n), ("logic_parity", lp)])
    s["values"] = collections.OrderedDict([("bytes_compared", byts), ("floats_compared", floats),
                                           ("differing_values", sum(e["differing_values"] or 0 for e in diff)),
                                           ("max_ulp", max(ulps) if ulps else None)])
    s["call_records"] = collections.OrderedDict(call)
    s["mutants"] = collections.OrderedDict([("total", mut["total"]), ("killed", mut["killed"]), ("survivors", mut["survivors"])])
    s["stand_ins"] = [
        "%d cases (tag standin in testset/grid.tsv) use keyframed ptN_pos or ptN_use values: AE keyframe interpolation is replaced by the harness's labelled linear stand-in on both sides" % keyed,
        "%d cases hand matte layers (fg1_matte to fg3_matte) to the plugin as raw clips through the harness, which is a stand-in for how AE hands a hidden layer over" % layered,
        "logic_parity counts compared cases that depend on either stand-in (%d of %d compared); they are not full parity" % (lp, len(cmp_)),
    ]
    s["fixes"] = list(fixes)
    s["commands"] = []
    s["measured_at"] = grade.get("measured_at", today())
    b = s["blockers"]
    if not cmp_:
        b.append("emulator side not run yet: no case has a reference to compare")
        if ooc and ooc["counts"]["status_counts"] and not ooc["counts"]["status_counts"].get("status:rendered"):
            s["status"] = "blocked"
            b.append("the harness status file shows no rendered case: %s" % json.dumps(ooc["counts"]["status_counts"], sort_keys=True))
    elif noref:
        b.append("%d in-contract cases have no reference" % len(noref))
    if diff:
        b.append("%d cases differ (%d values%s)" % (len(diff), s["values"]["differing_values"], ", %d NaN mismatches not counted in max_ulp" % nan_total if nan_total else ""))
    if ooc and ooc["counts"]["unresolved"]:
        b.append("%d cases did not render in the harness for a reason that is not the plugin's (timeout, emulator fault, harness limit): no reference" % ooc["counts"]["unresolved"])
    if ooc and ooc["counts"]["candidates_promote_review"]:
        b.append("%d out of contract candidates were rendered by the original and are within registered ranges: review for promotion into the grid" % ooc["counts"]["candidates_promote_review"])
    if ooc and any(m.get("review") for m in ooc["moved_out"]):
        b.append("%d cases left the contract because the plugin registers no counterpart of a param: review the param map" % sum(1 for m in ooc["moved_out"] if m.get("review")))
    if not kills:
        b.append("mutation kills not computed (kills.json missing)")
    else:
        if kills["totals"]["survivors"]:
            b.append("mutant survivors: %s" % ", ".join(kills["totals"]["survivors"]))
        if kills["totals"]["undecided"]:
            b.append("%d mutants undecided: no discriminating case yet" % len(kills["totals"]["undecided"]))
    if not calls:
        b.append("call records not compared (calls.json missing)")
    elif call["matching"] != call["compared"]:
        b.append("%d call records differ" % (call["compared"] - call["matching"]))
    if cmp_ and not unmet(s):
        s["status"] = "verified"
        s["blockers"] = []
    return s


def cmd_summary(a):
    results, testset = here(a.results), here(a.testset)
    gp = os.path.join(results, "grade.json")
    if not os.path.exists(gp):
        raise ToolError("summary needs %s: run grade first" % gp)
    grade = read_json(gp)
    cases = load_cases(testset)
    grid = load_grid(testset)

    def opt(name):
        p = os.path.join(results, name)
        return read_json(p) if os.path.exists(p) else None

    ooc, kills, calls = opt("ooc.json"), opt("kills.json"), opt("calls.json")
    tpl_p = here(a.template)
    template = read_json(tpl_p) if os.path.exists(tpl_p) else {}
    own_p = os.path.join(results, "summary.json")
    own = read_json(own_p) if os.path.exists(own_p) else template
    cand_p = os.path.join(testset, "out_of_contract.json")
    cand_n = len(read_json(cand_p)["cases"]) if os.path.exists(cand_p) else 0
    mt_p = here("results/mutants.tsv")
    built = len([r for r in read_tsv(mt_p) if r.get("status_vs_base") in ("changed-cases", "no-change")]) if os.path.exists(mt_p) else len(read_json(here("mutants.json")))
    fixes = list(own.get("fixes", [])) + list(a.add_fix or [])
    s = build_summary(template, cases, grid, grade, ooc, kills, calls, built, cand_n, fixes)
    keep = [c for c in template.get("commands", []) if not c.startswith(PREFIX)]
    seen = set()
    for src in (grade, ooc, calls, kills):
        if src and src.get("command") and src["command"] not in seen:
            seen.add(src["command"])
            keep.append(src["command"])
    keep.append("%s summary" % PREFIX)
    s["commands"] = keep
    write_json(os.path.join(results, "summary.json"), s)
    print("summary (%s): status %s, compared %d of %d in contract, %d identical, %d differing, %d out of contract, logic parity %d; "
          "call records %d/%d; mutants killed %d of %d" % (os.path.join(results, "summary.json"), s["status"], s["cases"]["compared"], s["cases"]["in_contract"],
                                                         s["cases"]["identical"], s["cases"]["differing"], s["cases"]["out_of_contract"], s["cases"]["logic_parity"],
                                                         s["call_records"]["matching"], s["call_records"]["compared"], s["mutants"]["killed"], s["mutants"]["total"]))
    return 0


# --------------------------------------------------------------------------------------------- triage

def tokens_of(case, block, tags):
    t = ["block:" + block, "depth:%d" % case["depth"], "layers:%d" % len(case.get("layers") or {}), "frames:%d" % case["frames"]]
    t += ["tag:" + x for x in tags]
    t += ["layer:" + k for k in sorted(case.get("layers") or {})]
    t += ["gap:" + g for g in gaps_of(case, block)]
    t += ["param:" + p for p in case.get("params", [])]
    if not case.get("params"):
        t.append("param:(all defaults)")
    return t


def write_ladder(case, testset, probe_dir):
    """A narrower test set around one failing case: reverted settings, single settings, layers dropped."""
    params, layers = list(case.get("params", [])), dict(case.get("layers") or {})
    rungs, seen = [], {}

    def add(label, kind, ps, ls, param=None):
        key = (tuple(ps), tuple(sorted(ls)))
        if key in seen:
            seen[key]["also"].append({"rung": label, "kind": kind, "param": param})
            return
        h = hashlib.sha1(("|".join([label] + ps + sorted(ls))).encode()).hexdigest()[:8]
        cid = "%s.probe-%s.%s-%dx%d.d%d.%s" % (EFFECT, label, case["source"], case["width"], case["height"], case["depth"], h)
        rc = dict(case)
        rc.update(id=cid, label="probe-" + label, params=ps, reference="%s.%s" % (cid, PIX[case["depth"]]))
        rc.pop("layers", None)
        if ls:
            rc["layers"] = dict(ls)
        rung = {"rung": label, "kind": kind, "param": param, "case": rc, "also": []}
        seen[key] = rung
        rungs.append(rung)

    add("orig", "original", params, layers)
    add("all-default", "default", [], layers)
    if layers:
        add("no-layers", "default", [], {})
    for p in params:
        pid = p.partition("=")[0]
        add("revert-" + pid, "revert", [q for q in params if q != p], layers, p)
    for p in params:
        pid = p.partition("=")[0]
        add("only-" + pid, "only", [p], layers, p)
    for lid in sorted(layers):
        add("drop-" + lid, "drop-layer", params, {k: v for k, v in layers.items() if k != lid}, lid)
    os.makedirs(os.path.join(probe_dir, "inputs"), exist_ok=True)
    for r in rungs:
        for f in [r["case"]["input"]] + list((r["case"].get("layers") or {}).values()):
            dst = os.path.join(probe_dir, f)
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(os.path.join(testset, f), dst)
    write_json(os.path.join(probe_dir, "cases.json"), {"format": 1, "ffmpeg": "none (raw inputs written by gen_grid.py)", "inputs": input_entries([r["case"] for r in rungs], probe_dir),
                                                       "note": "ladder around %s" % case["id"], "cases": [r["case"] for r in rungs]})
    write_json(os.path.join(probe_dir, "ladder.json"), {"source_case": case["id"], "rungs": [
        {"rung": r["rung"], "kind": r["kind"], "param": r["param"], "case": r["case"]["id"], "params": r["case"]["params"],
         "layers": sorted(r["case"].get("layers") or {}), "same_case_as": r["also"]} for r in rungs]})
    return rungs


def interpret_ladder(rungs, statuses):
    """statuses: case id -> 'pass' | 'FAIL' | 'no reference'. Plain lines on what isolates the cause."""
    items = []
    for r in rungs:
        status = statuses.get(r["case"]["id"], "no reference")
        items.append((r["rung"], r["kind"], r["param"], status))
        items += [(x["rung"], x["kind"], x["param"], status) for x in r["also"]]
    st = {i[0]: i[3] for i in items}
    out = ["  %-34s %s" % (k, v) for k, v in st.items()]
    if st.get("orig") != "FAIL":
        out.append("the original case does not fail here (%s): nothing to isolate" % st.get("orig"))
        return out
    if st.get("all-default") == "FAIL":
        out.append("the all-default case with the same layers already fails: the cause is in the layered path itself, not in a setting")
    if st.get("no-layers") == "FAIL":
        out.append("the case without layers fails too: not a layer problem")
    for label, kind, param, status in items:
        if kind == "only" and status == "FAIL":
            out.append("%s alone (default otherwise) fails: it is enough to cause a difference" % param)
        if kind == "revert" and status == "pass":
            out.append("reverting %s makes it pass: it is needed for the difference" % param)
        if kind == "drop-layer" and status == "pass":
            out.append("dropping layer %s makes it pass: that layer is needed for the difference" % param)
    return out


def cmd_triage(a):
    results, testset = here(a.results), here(a.testset)
    gp = os.path.join(results, "grade.json")
    if not os.path.exists(gp):
        raise ToolError("triage needs %s: run grade first" % gp)
    grade = read_json(gp)
    cases = {c["id"]: c for c in load_cases(testset)}
    grid = load_grid(testset)
    rows = load_param_map(a.param_map or here("../harness/param_map.tsv"))
    moved = moved_out_ids(results)
    cmp_ = [e for e in grade["cases"] if e["result"] in ("identical", "differing") and e["id"] not in moved]
    diff = [e for e in cmp_ if e["result"] == "differing"]
    print("triage: %d compared, %d differing" % (len(cmp_), len(diff)))
    um = unmatched_params(list(cases.values()), rows)
    if um is None:
        print("param map: no plugin side yet (no plugin reached PARAMS_SETUP): nothing can be flagged as unmatched")
    elif um:
        print("param map: grid ids the plugin does not register (unmatched or missing):")
        for k, v in sorted(um.items()):
            hit = sum(1 for e in diff if k in [p.partition("=")[0] for p in cases[e["id"]].get("params", [])] + list(cases[e["id"]].get("layers") or {}))
            print("  %-26s %3d cases, %3d differing  (%s)" % (k, v["cases"], hit, v["why"]))
    else:
        print("param map: every param and layer the grid uses is paired with a registered plugin param")
    if not diff:
        print("no differing cases: nothing to triage")
        return 0
    tot, hit = collections.Counter(), collections.Counter()
    diffids = {e["id"] for e in diff}
    for e in cmp_:
        c = cases[e["id"]]
        for t in set(tokens_of(c, e["block"], e["tags"])):
            tot[t] += 1
            if e["id"] in diffids:
                hit[t] += 1
    print("\nstated-rule gaps exercised by differing cases:")
    for g in sorted({t[4:] for t in hit if t.startswith("gap:")}):
        print("  %-4s %-48s %3d of %3d differ" % (g, GAP_TITLES.get(g, ""), hit["gap:" + g], tot["gap:" + g]))
    for title, pre in (("block", "block:"), ("depth", "depth:"), ("layers", "layers:"), ("tags", "tag:"), ("layer ids", "layer:")):
        print("\nby %s:" % title)
        for t, n in sorted(((t, n) for t, n in hit.items() if t.startswith(pre)), key=lambda x: (-x[1] / tot[x[0]], -x[1], x[0])):
            print("  %-34s %3d of %3d differ" % (t[len(pre):], n, tot[t]))
    print("\nparameter values that co-occur with differences (top 30 by share, then count):")
    ranked = sorted(((t, n) for t, n in hit.items() if t.startswith("param:")), key=lambda x: (-x[1] / tot[x[0]], -x[1], x[0]))
    for t, n in ranked[:30]:
        print("  %-44s %3d of %3d differ" % (t[6:], n, tot[t]))
    print("\nsmallest failing cases (pixels, frames, depth):")
    small = sorted(diff, key=lambda e: (e["width"] * e["height"], e["frames"], e["depth"], len(cases[e["id"]].get("params", [])), e["id"]))
    for e in small[:10]:
        print("  %-60s %dx%d f%d d%d %s" % (e["id"], e["width"], e["height"], e["frames"], e["depth"], " ".join(cases[e["id"]].get("params", [])) or "(defaults)"))
    probe = here(a.probe_dir)
    rungs = write_ladder(cases[small[0]["id"]], testset, probe)
    print("\nladder around %s: %d probe cases written to %s" % (small[0]["id"], len(rungs), probe))
    print("  run it: python3 ../harness/run_harness.py --testset %s --refs %s/refs -j1   then" % (a.probe_dir, a.probe_dir))
    print("          python3 phase_b.py triage --probe-refs %s/refs" % a.probe_dir)
    if a.probe_refs:
        fxp, vfx = find_fx_parity(a.fx_parity), find_vfx(a.vfx)
        rep, _, _ = run_fx_parity(fxp, probe, os.path.abspath(a.probe_refs), vfx, 0, 1)
        for ln in interpret_ladder(rungs, {r["id"]: r["status"] for r in rep["cases"]}):
            print(ln)
    return 0


# ------------------------------------------------------------------------------------------- selftest

class Checks:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail=""):
        self.rows.append({"check": name, "ok": bool(ok), "detail": str(detail)})
        print("  [%s] %s%s" % ("ok" if ok else "FAIL", name, (": " + str(detail)) if detail else ""))
        return ok


def _flip_bit(path, nbyte, bit=0):
    with open(path, "r+b") as fh:
        fh.seek(nbyte)
        b = fh.read(1)[0]
        fh.seek(nbyte)
        fh.write(bytes([b ^ (1 << bit)]))


def _words32(path):
    with open(path, "rb") as fh:
        data = fh.read()
    return _words("I", data)


def _find_f32(refs, cases, pred):
    """First 32 bpc reference of `cases` holding a float with pred(bits); (case, byte offset) or (None, None)."""
    for c in cases:
        if c["depth"] != 32:
            continue
        for i, bits in enumerate(_words32(os.path.join(refs, c["reference"]))):
            if pred(bits):
                return c, i * 4
    return None, None


def _put_word(path, off, bits):
    with open(path, "r+b") as fh:
        fh.seek(off)
        fh.write(bits.to_bytes(4, sys.byteorder))


def _get_word(path, off):
    with open(path, "rb") as fh:
        fh.seek(off)
        return int.from_bytes(fh.read(4), sys.byteorder)


def _finite_nonzero(b):
    return b not in (0, 0x80000000) and (b & 0x7F800000) != 0x7F800000


def phase_a_changed(muts, base):
    """Cases each mutant changes against the base rewrite, from the phase A changed lists only."""
    out = {}
    for m in muts:
        got = None
        for d in ("mutants2", "mutants"):
            p = here(os.path.join(d, m["id"], "changed.txt"))
            if os.path.exists(p):
                got = read_text(p).split()
                break
        if got is None:
            p = here(os.path.join("results", "mutants", m["id"] + ".changed.txt"))
            got = read_text(p).split() if os.path.exists(p) else []
        out[m["id"]] = sorted(i for i in got if i in base)
    return out


def cmd_selftest(a):
    sd = here(a.selftest_dir)
    testset = here(a.testset)
    if not os.path.abspath(sd).startswith(os.path.abspath(here("results")) + os.sep):
        raise ToolError("the self test writes only under results/")
    os.makedirs(sd, exist_ok=True)
    fxp, vfx = find_fx_parity(a.fx_parity), find_vfx(a.vfx)
    ck, me = Checks(), os.path.abspath(__file__)
    cases = load_cases(testset)
    muts = read_json(here("mutants.json"))
    base = {r["id"]: (r.get("exit"), r.get("sha256")) for r in read_tsv(here("results/base.tsv"))}

    def run(*args, vfx_=None):
        p = subprocess.run([sys.executable, me, *args, "--testset", testset, "--fx-parity", fxp, "--vfx", vfx_ or vfx], capture_output=True, text=True)
        if p.returncode != 0:
            raise ToolError("phase_b %s failed: %s" % (args[0], (p.stderr or p.stdout)[-600:]))
        return p.stdout.strip()

    # a) perfect oracle: the rewrite's own base renders as the references
    print("a) perfect oracle")
    pf = os.path.join(sd, "perfect")
    refs = os.path.join(pf, "refs")
    shutil.rmtree(pf, ignore_errors=True)
    os.makedirs(refs)
    t0 = time.time()
    for c in cases:
        rc, err = render_case(vfx, testset, c, os.path.join(refs, c["reference"]))
        if rc != 0:
            raise ToolError("base render of %s failed: %s" % (c["id"], err))
    ck.check("rendered the stand-in references", True, "%d cases in %.0f s" % (len(cases), time.time() - t0))
    same = sum(1 for c in cases if sha256_file(os.path.join(refs, c["reference"])) == base[c["id"]][1])
    ck.check("the stand-in references equal results/base.tsv", same == len(cases), "%d of %d" % (same, len(cases)))
    print("   " + run("grade", "--refs", refs, "--results", pf))
    g = read_json(os.path.join(pf, "grade.json"))["totals"]
    ck.check("perfect: all compared, 0 differing", g["compared"] == len(cases) and g["differing"] == 0 and g["identical"] == len(cases), json.dumps(g))
    expected_changed = phase_a_changed(muts, base)
    n_changing = sum(1 for v in expected_changed.values() if v)
    print("   " + run("kills", "--refs", refs, "--results", pf))
    k = read_json(os.path.join(pf, "kills.json"))
    srcs = collections.Counter((m["source"] or "none").split(":")[0] for m in k["mutants"])
    ck.check("perfect: mutants killed equal the phase A count of mutants that change an output", k["totals"]["killed"] == n_changing,
             "killed %d of %d, phase A changed lists say %d (sources %s)" % (k["totals"]["killed"], k["totals"]["total"], n_changing, dict(srcs)))
    ck.check("perfect: killing cases per mutant equal the phase A changed lists", all(sorted(m["killing_cases"]) == expected_changed[m["id"]] for m in k["mutants"] if m["status"] == "killed"))
    print("   " + run("ooc", "--status", write_status(sd, "perfect", cases, {}), "--results", pf, "--param-map", os.path.join(sd, "no_param_map.tsv")))
    print("   " + run("summary", "--results", pf))
    s = read_json(os.path.join(pf, "summary.json"))
    ck.check("perfect: summary computed from the files", s["cases"]["compared"] == len(cases) and s["cases"]["differing"] == 0 and s["mutants"]["killed"] == n_changing,
             json.dumps({"cases": s["cases"], "values": s["values"], "mutants": s["mutants"], "status": s["status"]}))
    ck.check("perfect: not verified without call records", s["status"] != "verified" and any("call records" in b for b in s["blockers"]), s["status"])

    # b) perturbed oracle
    print("b) perturbed oracle")
    pt = os.path.join(sd, "perturbed")
    shutil.rmtree(pt, ignore_errors=True)
    prefs = os.path.join(pt, "refs")
    shutil.copytree(refs, prefs)
    perturbed, ulp_want = {}, {}

    def rest():
        return [c for c in cases if c["id"] not in perturbed]

    m15 = expected_changed.get("M15") or []
    if len(m15) == 1:
        c = next(x for x in cases if x["id"] == m15[0])
        _flip_bit(os.path.join(prefs, c["reference"]), 4 if c["depth"] == 32 else 5, 0)
        perturbed[c["id"]] = "bit flip in the single case M15 changes"
    c8 = next(c for c in rest() if c["depth"] == 8 and c["width"] * c["height"] > 100)
    _flip_bit(os.path.join(prefs, c8["reference"]), 5, 0)
    perturbed[c8["id"]] = "8 bpc bit flip"
    c1, off1 = _find_f32(prefs, rest(), _finite_nonzero)
    if c1:
        _flip_bit(os.path.join(prefs, c1["reference"]), off1, 0)
        perturbed[c1["id"]] = "32 bpc lowest mantissa bit of a finite value"
        ulp_want[c1["id"]] = 1
    c2, off2 = _find_f32(prefs, rest(), lambda b: b == 0x80000000)
    if c2:
        _put_word(os.path.join(prefs, c2["reference"]), off2, 0)
        perturbed[c2["id"]] = "-0 in the output, +0 in the reference"
    else:
        c2, off2 = _find_f32(prefs, rest(), lambda b: b == 0)
        if c2:
            _put_word(os.path.join(prefs, c2["reference"]), off2, 0x80000000)
            perturbed[c2["id"]] = "+0 in the output, -0 in the reference (no output holds a -0)"
    if c2:
        ulp_want[c2["id"]] = 0
    c3, off3 = _find_f32(prefs, rest(), f32_is_nan)
    if c3:
        old = _get_word(os.path.join(prefs, c3["reference"]), off3)
        new = old ^ 1 if f32_is_nan(old ^ 1) else old ^ 2
        _put_word(os.path.join(prefs, c3["reference"]), off3, new)
        perturbed[c3["id"]] = "NaN payload changed"
    else:
        c3, off3 = _find_f32(prefs, rest(), _finite_nonzero)
        if c3:
            _put_word(os.path.join(prefs, c3["reference"]), off3, 0x7FC00001)
            perturbed[c3["id"]] = "NaN in the reference over a finite output (no output holds a NaN, so a payload change cannot be built from real data)"
    for cid, why in perturbed.items():
        print("   perturbed %s: %s" % (why, cid))
    print("   " + run("grade", "--refs", prefs, "--results", pt))
    differing = {e["id"]: e for e in read_json(os.path.join(pt, "grade.json"))["cases"] if e["result"] == "differing"}
    ck.check("perturbed: exactly the perturbed cases differ", set(differing) == set(perturbed), "%d differ, %d perturbed" % (len(differing), len(perturbed)))
    ck.check("perturbed: each differs in exactly one value", all(e["differing_values"] == 1 for e in differing.values()), [e["differing_values"] for e in differing.values()])
    for cid, want in ulp_want.items():
        ck.check("perturbed: max ULP for '%s'" % perturbed[cid], differing.get(cid, {}).get("max_ulp") == want, "reported %s, expected %s" % (differing.get(cid, {}).get("max_ulp"), want))
    if c3:
        ck.check("perturbed: NaN mismatch counted once", differing.get(c3["id"], {}).get("nan_mismatches") == 1, differing.get(c3["id"], {}).get("nan_mismatches"))
    ck.check("perturbed: 8 and 16 bpc cases report no ULP", all(e["max_ulp"] is None for e in differing.values() if e["depth"] != 32))
    print("   " + run("kills", "--refs", prefs, "--results", pt))
    kp = read_json(os.path.join(pt, "kills.json"))
    pset = set(perturbed)
    want_killed = sorted(mid for mid, ch in expected_changed.items() if any(i not in pset for i in ch))
    want_surv = sorted(mid for mid, ch in expected_changed.items() if not any(i not in pset for i in ch))
    got_killed = sorted(m["id"] for m in kp["mutants"] if m["status"] == "killed")
    ck.check("perturbed: killed set equals the set logic over the phase A changed lists", got_killed == want_killed, "killed %d, expected %d" % (len(got_killed), len(want_killed)))
    ck.check("perturbed: survivors are the mutants with no changed case outside the perturbed ones", sorted(kp["totals"]["survivors"]) == want_surv, "reported %s, expected %s" % (kp["totals"]["survivors"], want_surv))
    print("   " + run("summary", "--results", pt))
    sp = read_json(os.path.join(pt, "summary.json"))
    ck.check("perturbed: summary counts the perturbed cases and values", sp["cases"]["differing"] == len(perturbed) and sp["values"]["differing_values"] == len(perturbed) and sp["status"] == "in_progress",
             json.dumps({"cases": sp["cases"], "values": sp["values"], "survivors": sp["mutants"]["survivors"]}))
    out = run("triage", "--results", pt, "--probe-dir", os.path.join(pt, "probe"))
    print("\n".join("   " + x for x in out.splitlines()[:16]) + "\n   ...")
    ladder = read_json(os.path.join(pt, "probe", "ladder.json"))
    pr_refs = os.path.join(pt, "probe", "refs")
    os.makedirs(pr_refs, exist_ok=True)
    pcases = read_json(os.path.join(pt, "probe", "cases.json"))["cases"]
    for c in pcases:
        render_case(vfx, os.path.join(pt, "probe"), c, os.path.join(pr_refs, c["reference"]))
    rep, _, _ = run_fx_parity(fxp, os.path.join(pt, "probe"), pr_refs, vfx, 0, 1)
    ck.check("ladder: a valid fx-parity test set, every rung renders and grades against vfx", rep["summary"]["pass"] == len(pcases) == rep["summary"]["total"],
             "%d rungs around %s" % (len(ladder["rungs"]), ladder["source_case"]))

    # c) missing references
    print("c) missing references")
    ms_ = os.path.join(sd, "missing")
    shutil.rmtree(ms_, ignore_errors=True)
    mrefs = os.path.join(ms_, "refs")
    shutil.copytree(refs, mrefs)
    gone = cases[::38][:10]
    for c in gone:
        os.remove(os.path.join(mrefs, c["reference"]))
    print("   " + run("grade", "--refs", mrefs, "--results", ms_))
    gm = read_json(os.path.join(ms_, "grade.json"))["totals"]
    ck.check("missing: removed references are 'no reference', not passes", gm["no_reference"] == len(gone) and gm["compared"] == len(cases) - len(gone) and gm["identical"] == len(cases) - len(gone), json.dumps(gm))
    print("   " + run("kills", "--refs", mrefs, "--results", ms_))
    run("summary", "--results", ms_)
    sm = read_json(os.path.join(ms_, "summary.json"))
    ck.check("missing: summary compared excludes them, in_contract keeps them", sm["cases"]["compared"] == len(cases) - len(gone) and sm["cases"]["in_contract"] == len(cases) and any("no reference" in b for b in sm["blockers"]), sm["blockers"][:1])

    # d) out of contract from a synthetic status file and param map
    print("d) out of contract")
    od = os.path.join(sd, "ooc")
    shutil.rmtree(od, ignore_errors=True)
    os.makedirs(od)
    pick = cases[5:45:5]
    st = {pick[0]["id"]: {"status": "unsupported", "error": "param `x` has no counterpart among the registered params of TwixtorPro_64"},
          pick[1]["id"]: {"status": "error", "error": "PluginError: RENDER returned PF_Err 3"},
          pick[2]["id"]: {"status": "refused", "error": "the plugin refused the frame size"},
          pick[3]["id"]: {"status": "error", "error": "GuestFault: read of unmapped memory"},
          pick[4]["id"]: {"status": "timeout", "error": "no frame in 600 s"},
          pick[5]["id"]: {"status": "not_run", "error": "the plugin did not boot"}}
    sf = write_status(sd, "ooc", cases, st)
    print("   " + run("ooc", "--status", sf, "--results", od, "--param-map", os.path.join(sd, "no_param_map.tsv")))
    oc = read_json(os.path.join(od, "ooc.json"))
    ck.check("ooc: refused, plugin error and unregistered param leave the contract", {m["id"] for m in oc["moved_out"]} == {pick[0]["id"], pick[1]["id"], pick[2]["id"]}, oc["counts"]["moved_by_category"])
    ck.check("ooc: timeout, emulator fault and not_run stay in contract as unresolved", {u["id"] for u in oc["unresolved"]} == {pick[3]["id"], pick[4]["id"], pick[5]["id"]})
    run("grade", "--refs", refs, "--results", od)
    run("summary", "--results", od)
    so = read_json(os.path.join(od, "summary.json"))
    n_cand = len(read_json(os.path.join(testset, "out_of_contract.json"))["cases"])
    ck.check("ooc: they leave the in-contract count and join the out of contract count", so["cases"]["in_contract"] == len(cases) - 3 and so["cases"]["compared"] == len(cases) - 3 and so["cases"]["out_of_contract"] == 3 + n_cand, json.dumps(so["cases"]))
    pm = os.path.join(sd, "param_map_synthetic.tsv")
    write_synthetic_param_map(pm, ranges={"fg1_inv_matte_shrink": ("0", "1")})

    def outside(c):
        for p in c["params"]:
            if p.startswith("fg1_inv_matte_shrink="):
                for key in p.partition("=")[2].split(";"):
                    x = float(key.split("@")[0])
                    if x != x or x < 0 or x > 1:
                        return True
        return False

    want_range = sorted(c["id"] for c in cases if outside(c))
    run("ooc", "--status", sf, "--results", od, "--param-map", pm, "--also", "timeout")
    oc2 = read_json(os.path.join(od, "ooc.json"))
    got_range = sorted(m["id"] for m in oc2["moved_out"] if m["category"] == "out_of_range")
    ck.check("ooc: a registered range 0..1 moves the cases outside it out of contract", got_range == want_range and len(want_range) > 0, "%d moved, %d expected" % (len(got_range), len(want_range)))
    ck.check("ooc: --also timeout moves the timeout case", pick[4]["id"] in {m["id"] for m in oc2["moved_out"]})
    run("ooc", "--status", sf, "--results", od, "--param-map", pm, "--make-testset", os.path.join(od, "ooc_testset"))
    os.makedirs(os.path.join(od, "none"), exist_ok=True)
    rep, _, _ = run_fx_parity(fxp, os.path.join(od, "ooc_testset"), os.path.join(od, "none"), vfx, 0, 1)
    ck.check("ooc: --make-testset writes a set fx-parity reads", rep["summary"]["total"] == n_cand, rep["summary"])

    # e) call records
    print("e) call records")
    cd = os.path.join(sd, "calls")
    shutil.rmtree(cd, ignore_errors=True)
    rd, pd_ = os.path.join(cd, "rewrite"), os.path.join(cd, "plugin")
    os.makedirs(rd)
    os.makedirs(pd_)
    scen = ["match", "missing_read", "extra_read", "wrong_time", "wrong_layer", "wrong_index"]
    two = [c for c in cases if len(c.get("layers") or {}) == 2 and c["frames"] >= 3][:len(scen)]
    rec_vfx = vfx
    with open(vfx, "rb") as fh:
        has = b"VFX_CALL_RECORD" in fh.read()
    if not has:
        rec_vfx = find_tool(None, "VFX_RECORD_PATH", ["../harness/repo/target/release/vfx"], ["vfx"])
    print("   " + run("record-rewrite", "--out", rd, "--only", ",".join(c["id"] for c in two), vfx_=rec_vfx))
    info = read_json(os.path.join(rd, "_record_run.json"))
    ck.check("record-rewrite: output with VFX_CALL_RECORD equals results/base.tsv", info["recorded"] == len(two) and not info["output_differs_from_base"],
             "vfx %s%s" % (rec_vfx, "" if has else " (bin/vfx_base has no VFX_CALL_RECORD)"))
    write_synthetic_param_map(pm, ranges={})
    expect = {"missing_read": "extra_read", "extra_read": "missing_read", "wrong_time": "wrong_time", "wrong_layer": "wrong_layer", "wrong_index": "wrong_frame_index"}
    for c, sc in zip(two, scen):
        rl, _ = read_jsonl(os.path.join(rd, c["id"] + ".jsonl"))
        synth_plugin_record(os.path.join(pd_, c["id"] + ".jsonl"), rl, pm, sc)
    run("calls", "--plugin-records", pd_, "--rewrite-records", rd, "--results", cd, "--param-map", pm)
    cj = read_json(os.path.join(cd, "calls.json"))
    got = {x["id"]: x for x in cj["cases"]}
    ck.check("calls: compared and matching counts", cj["compared"] == len(scen) and cj["matching"] == 1, "%d compared, %d matching" % (cj["compared"], cj["matching"]))
    for c, sc in zip(two, scen):
        x = got.get(c["id"])
        if sc == "match":
            ck.check("calls: identical records match", bool(x) and x["status"] == "match")
        else:
            kinds = [d["kind"] for d in (x or {}).get("first_mismatch", {}).get("differences", [])]
            ck.check("calls: '%s' is reported as %s" % (sc, expect[sc]), bool(x) and x["status"] == "mismatch" and expect[sc] in kinds,
                     "first differing output frame %s, kinds %s" % ((x or {}).get("first_mismatch", {}).get("out_frame"), kinds))
    path = os.path.join(sd, "selftest.json")
    write_json(path, {"measured_at": today(), "vfx": vfx, "fx_parity": fxp, "checks": ck.rows,
                      "numbers": {"cases": len(cases), "mutants": len(muts), "mutants_changing_an_output": n_changing,
                                  "perfect": {"differing": g["differing"], "killed": k["totals"]["killed"]},
                                  "perturbed": {"cases": perturbed, "differing": len(differing), "killed": len(got_killed), "survivors": kp["totals"]["survivors"]}}})
    bad = [r for r in ck.rows if not r["ok"]]
    print("\nselftest: %d checks, %d failed -> %s" % (len(ck.rows), len(bad), path))
    return 1 if bad else 0


def write_status(sd, name, cases, entries):
    out = [dict(entries.get(c["id"], {"status": "rendered"}), id=c["id"]) for c in cases]
    p = os.path.join(sd, "status_%s.json" % name)
    write_json(p, {"testset": "synthetic", "cases": out, "boots": []})
    with open(os.path.join(sd, "no_param_map.tsv"), "w") as fh:
        fh.write("plugin\teffect\tae_index\tvfx_kind\tvfx_id\tmatch\n")
    return p


def write_synthetic_param_map(path, ranges):
    cols = ["plugin", "effect", "ae_index", "ae_name", "ae_type", "ae_min", "ae_max", "ae_slider_min", "ae_slider_max", "ae_default",
            "ae_choices", "ae_flags", "vfx_kind", "vfx_id", "vfx_name", "vfx_default", "vfx_choices", "match"]
    rows = [{"plugin": PLUGIN_NAME, "effect": EFFECT, "ae_index": str(i), "ae_name": n, "ae_type": "LAYER", "vfx_kind": "layer",
             "vfx_id": lid, "vfx_name": n, "match": "exact"} for i, (lid, n) in enumerate([("fg1_matte", "FG1 Matte"), ("fg2_matte", "FG2 Matte"), ("fg3_matte", "FG3 Matte")], 1)]
    for j, (pid, (lo, hi)) in enumerate(sorted(ranges.items()), 10):
        rows.append({"plugin": PLUGIN_NAME, "effect": EFFECT, "ae_index": str(j), "ae_name": pid, "ae_type": "FLOAT_SLIDER", "ae_min": lo, "ae_max": hi,
                     "vfx_kind": "float", "vfx_id": pid, "vfx_name": pid, "match": "exact"})
    with open(path, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(r.get(c, "") for c in cols) + "\n")


def synth_plugin_record(path, rewrite_lines, pm_path, scenario):
    """A plugin record in the harness format built from the rewrite's own record, then damaged once."""
    rows = load_param_map(pm_path)
    idx = {r["vfx_id"]: int(r["ae_index"]) for r in rows if r.get("vfx_id") and r.get("ae_index")}
    idx["input"] = 0
    reads = [ln for ln in rewrite_lines if ln["ev"] == "read"]
    seen = collections.Counter((ln["out_frame"], ln["layer"], ln["index"], ln["time"]) for ln in reads)
    unique = [i for i, ln in enumerate(reads) if seen[(ln["out_frame"], ln["layer"], ln["index"], ln["time"])] == 1]
    if scenario == "wrong_layer":
        target = next(i for i in unique if reads[i]["layer"] != "input")
    else:
        target = unique[1] if len(unique) > 1 else unique[0]
    lines, seq, n_read, scale = [], 0, -1, None

    def add(**kw):
        nonlocal seq
        seq += 1
        lines.append(dict({"seq": seq}, **kw))

    add(ev="render_path", smart=True, out_flags="0x0", out_flags2="0x0")
    for ln in rewrite_lines:
        if ln["ev"] == "frame":
            scale = ln["time_scale"]
            add(ev="frame", out_frame=ln["out_frame"], time=ln["time"], time_step=ln["time_step"], time_scale=ln["time_scale"])
        elif ln["ev"] == "read":
            n_read += 1
            layer, time_, index = ln["layer"], ln["time"], ln["index"]
            hit = n_read == target
            if hit and scenario == "missing_read":
                continue
            if hit and scenario == "wrong_time":
                time_ += 50
            if hit and scenario == "wrong_layer":
                layer = "fg3_matte"
            if hit and scenario == "wrong_index" and index is not None:
                index += 1
            add(ev="cb", fn="prerender_cb.checkout_layer", args=["0x1", idx[layer], n_read, "0x2", time_, 100, scale, "0x3"], ret=0,
                param_index=idx[layer], checkout_id=n_read, time=time_, time_step=100, time_scale=scale,
                footage_frame=index if index is not None else 99, inside=index is not None, out_frame=ln["out_frame"])
            if hit and scenario == "extra_read":
                add(ev="cb", fn="prerender_cb.checkout_layer", args=[], ret=0, param_index=0, checkout_id=9999, time=time_ + 7100,
                    time_step=100, time_scale=scale, footage_frame=(index or 0) + 71, inside=True, out_frame=ln["out_frame"])
    add(ev="case_done", id=os.path.basename(path)[:-6], status="rendered")
    with open(path, "w") as fh:
        for ln in lines:
            fh.write(json.dumps(ln) + "\n")


# ---------------------------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--results", default="results")
        p.add_argument("--testset", default="testset")
        p.add_argument("--fx-parity")
        p.add_argument("--vfx")
        p.add_argument("--base", default="results/base.tsv")
        return p

    p = common(sub.add_parser("grade"))
    p.add_argument("--refs", required=True)
    p.add_argument("--jobs", type=int, default=1)
    p = common(sub.add_parser("ooc"))
    p.add_argument("--status", action="append")
    p.add_argument("--param-map")
    p.add_argument("--also", help="comma list of %s" % ",".join(OOC_OPTIONAL))
    p.add_argument("--make-testset")
    p.add_argument("--judged", action="append",
                   help="TSV id<TAB>reason<TAB>evidence: these cases are moved out of contract (category judged) with their reason (repeatable)")
    p.add_argument("--ooc-rewrite", default="results/ooc.tsv")
    p = common(sub.add_parser("calls"))
    p.add_argument("--plugin-records", required=True)
    p.add_argument("--rewrite-records", required=True)
    p.add_argument("--param-map")
    p.add_argument("--ordered", action="store_true")
    p = common(sub.add_parser("record-rewrite"))
    p.add_argument("--out", required=True)
    p.add_argument("--only")
    p = common(sub.add_parser("kills"))
    p.add_argument("--refs", required=True)
    p.add_argument("--mutants", default="mutants.json")
    p.add_argument("--mutants-dirs", default="mutants2,mutants")
    p.add_argument("--ids")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--tree")
    p = common(sub.add_parser("summary"))
    p.add_argument("--template", default="results/summary.json")
    p.add_argument("--add-fix", action="append")
    p = common(sub.add_parser("triage"))
    p.add_argument("--probe-dir", default="probe")
    p.add_argument("--probe-refs")
    p.add_argument("--param-map")
    p = common(sub.add_parser("selftest"))
    p.add_argument("--selftest-dir", default="results/selftest")
    a = ap.parse_args(argv)
    try:
        return {"grade": cmd_grade, "ooc": cmd_ooc, "calls": cmd_calls, "record-rewrite": cmd_record_rewrite, "kills": cmd_kills,
                "summary": cmd_summary, "triage": cmd_triage, "selftest": cmd_selftest}[a.cmd](a)
    except ToolError as e:
        print("phase_b: error: %s" % e, file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError) as e:
        if os.environ.get("PHASE_B_TRACE"):
            raise
        print("phase_b: error: %s: %s (PHASE_B_TRACE=1 shows the traceback)" % (type(e).__name__, e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
