#!/usr/bin/env python3
"""Phase B tooling of the vectors-in module: grade, out of contract, call records, mutation kills,
summary, triage and a self test. Python 3 standard library only.

    phase_b.py grade   --refs DIR [--vfx BIN] [--fx-parity BIN]
    phase_b.py ooc     --status FILE
    phase_b.py rewrite-records --out DIR [--vfx BIN]
    phase_b.py calls   --plugin-records DIR --rewrite-records DIR [--rule reads|registrations]
    phase_b.py kills   --refs DIR [--rerender]
    phase_b.py summary
    phase_b.py triage  [--probe-dir DIR]
    phase_b.py selftest

Nothing here runs the plugin and nothing here changes the rewrite. Every result goes to --out-dir
(default `results/`); the self test writes only under `results/selftest/`. Run it from the module
folder on the server (`python3 tools/phase_b.py ...`, the scripts copied to `tools/` like the phase A
ones) or from the repository folder `parity/vectors-in/`.

The bar (docs/parity.md, "How parity is measured"): same inputs, identical outputs, bytes exact at 8
and 16 bpc, floats bit for bit at 32 bpc (`fx-parity run --max-ulp 0`: -0 against 0 and differing NaN
bits differ). A case with no reference is "no reference". A case the original refuses or errors on is
out of contract and counts as neither a pass nor a failure.
"""
import argparse
import array
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import date
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "results").is_dir() and (HERE / "testsets").is_dir() else HERE.parent
sys.path.insert(0, str(HERE))
import callrec  # noqa: E402  (the call-record rule of `calls`)
import run_grid  # noqa: E402  (phase A: case_args, run_one, first_note)

MODULE = "vectors-in"
PLUGIN = "TwixtorProVectorsIn_64.aex"
PLUGIN_STEM = "TwixtorProVectorsIn_64"
EFFECT = "vectors-in"
MIN_CASES = 150
PIX_FMT = {8: "rgba", 16: "rgba64le", 32: "gbrapf32le"}
SAMPLE_BYTES = {8: 1, 16: 2, 32: 4}
GRID_BASE_PARAMS = ["vectors_max_displace=32", "speed_percent=50"]
PHASE_A_COMMANDS = [
    "python3 tools/gen_grid.py --out testsets/grid",
    "python3 tools/run_grid.py run --vfx bin/vfx-base --testset testsets/grid --out results/base.grid.tsv --keep refs/grid --jobs 2",
    "python3 tools/run_grid.py check --vfx bin/vfx-base --ooc testsets/grid/out_of_contract.json --out results/out_of_contract.tsv",
    "bash tools/build_mutants.sh m00 m01 ... m24",
]


# ---------------------------------------------------------------------------------------------
# small helpers

def jload(path):
    with open(path, encoding="utf8") as fh:
        return json.load(fh)


def jsave(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")


def read_tsv(path):
    """id -> list of fields; the header line is skipped."""
    rows = {}
    with open(path, encoding="utf8") as fh:
        next(fh)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            rows[f[0]] = f
    return rows


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def today():
    return date.today().isoformat()


def load_testset(testset):
    """(cases in file order, case_meta dict). The meta file is optional (a probe set has a small one)."""
    ts = Path(testset)
    doc = jload(ts / "cases.json")
    meta_path = ts / "case_meta.json"
    meta = jload(meta_path) if meta_path.is_file() else {}
    return doc["cases"], meta


def group_of(meta, cid):
    return meta.get(cid, {}).get("group", "?")


def tags_of(meta, cid):
    m = meta.get(cid, {})
    tags = list(m.get("flags", []))
    if m.get("stand_in"):
        tags.append("stand_in:" + str(m["stand_in"]))
    return tags


def is_logic(meta, cid):
    return bool(meta.get(cid, {}).get("stand_in"))


def case_params(case):
    out = {}
    for p in case.get("params", []):
        k, _, v = p.partition("=")
        out[k] = v
    return out


def render_case(vfx, testset, case, record=None):
    """(exit code, stdout bytes, first stderr note). The same command line fx-parity builds."""
    env = dict(os.environ, RAYON_NUM_THREADS="1")
    env.pop("VFX_CALL_RECORD", None)
    if record:
        env["VFX_CALL_RECORD"] = str(record)
    p = subprocess.run([str(vfx)] + run_grid.case_args(case, str(testset)), stdin=subprocess.DEVNULL,
                       capture_output=True, env=env)
    return p.returncode, p.stdout, run_grid.first_note(p.stderr)


def find_tool(explicit, env_name, candidates, what):
    cands = [explicit, os.environ.get(env_name)] + list(candidates)
    for c in cands:
        if c and Path(c).is_file():
            return str(c)
    sys.exit("phase_b: %s not found; pass it explicitly or set %s" % (what, env_name))


def find_fxparity(explicit):
    """The first fx-parity whose `run` knows --max-ulp (the bit for bit bar needs it)."""
    cands = [explicit, os.environ.get("FX_PARITY")] + default_fxparity()
    tried = []
    for c in cands:
        if not c or not Path(c).is_file():
            continue
        try:
            p = subprocess.run([str(c), "run", "--help"], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if "--max-ulp" in (p.stdout + p.stderr):
            return str(c)
        tried.append(str(c))
    sys.exit("phase_b: no fx-parity with `run --max-ulp` found (tried %s); pass --fx-parity or set FX_PARITY" % (tried or "none"))


def default_vfx():
    return str(ROOT / "bin" / "vfx-base")


def default_fxparity():
    home = Path.home() / "twixtor-parity"
    return [home / "harness" / "repo" / "target" / "release" / "fx-parity",
            home / "closeout-1" / "target" / "release" / "fx-parity", shutil.which("fx-parity")]


# ---------------------------------------------------------------------------------------------
# value level comparison (own check beside fx-parity: counts values, locates the first one)

def f32_ordered(bits):
    return -(bits & 0x7FFFFFFF) if bits & 0x80000000 else bits


def f32_is_nan(bits):
    return (bits & 0x7FFFFFFF) > 0x7F800000


def locate(depth, w, h, idx):
    """Where value number idx of a clip is: frame, x, y, channel. 8 and 16 bpc are interleaved rgba,
    32 bpc is planar g, b, r, a per frame."""
    n = w * h
    per = n * 4
    frame, rem = divmod(idx, per)
    if depth == 32:
        plane, pix = divmod(rem, n)
        return {"frame": frame, "x": pix % w, "y": pix // w, "channel": "gbra"[plane]}
    pix, ch = divmod(rem, 4)
    return {"frame": frame, "x": pix % w, "y": pix // w, "channel": "rgba"[ch]}


def diff_values(depth, ref, out):
    """Value by value comparison of two clips of one depth: bytes at 8 bpc, 16 bit samples at 16 bpc,
    float bit patterns at 32 bpc. max_ulp is the ULP distance (as fx-parity counts it: -0 against 0 is
    0) of the largest finite pair; a pair with a NaN and different bits is counted in nan_mismatch."""
    res = {"values": len(ref) // SAMPLE_BYTES[depth], "differing": 0, "max_ulp": 0 if depth == 32 else None,
           "nan_mismatch": 0, "first_index": None, "size_mismatch": False}
    if len(ref) != len(out):
        res["size_mismatch"] = True
        return res
    if ref == out:
        return res
    if depth == 8:
        a, b = ref, out
    else:
        code = "H" if depth == 16 else "I"
        a, b = array.array(code), array.array(code)
        a.frombytes(ref)
        b.frombytes(out)
        if sys.byteorder == "big":
            a.byteswap()
            b.byteswap()
    first = None
    count = 0
    mx = 0
    nan = 0
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y:
            continue
        count += 1
        if first is None:
            first = i
        if depth == 32:
            if f32_is_nan(x) or f32_is_nan(y):
                nan += 1
            else:
                u = abs(f32_ordered(x) - f32_ordered(y))
                if u > mx:
                    mx = u
    res.update(differing=count, first_index=first, nan_mismatch=nan)
    if depth == 32:
        res["max_ulp"] = mx
    return res


# ---------------------------------------------------------------------------------------------
# the contract: which cases are in contract

STATUS_WORDS = {
    "rendered": "rendered", "ok": "rendered", "pass": "rendered", "done": "rendered",
    "unsupported": "refused", "refused": "refused", "refuse": "refused", "rejected": "refused",
    "error": "error", "fault": "error", "failed": "error", "crashed": "error",
    "timeout": "timeout", "timed_out": "timeout",
    "not_run": "not_run", "missing": "not_run", "skipped": "not_run", "unknown": "not_run",
}


def load_status(path):
    """The harness's per-case status as {id: {"status": word, "error": text, "raw": entry}}.
    Reads run_harness.py --summary ({"cases": [{"id", "status", "error", ...}]}), a list of such
    entries, or {id: status word}. Words: rendered, refused (unsupported), error, timeout, not_run."""
    doc = jload(path)
    items = doc["cases"] if isinstance(doc, dict) and "cases" in doc else doc
    out = {}
    if isinstance(items, dict):
        items = [{"id": k, "status": v} if isinstance(v, str) else dict(v, id=k) for k, v in items.items()]
    for it in items:
        raw = str(it.get("status", "not_run"))
        word = STATUS_WORDS.get(raw.strip().lower())
        if word is None:
            sys.exit("phase_b: case %s has the status %r, which is not one of %s" % (it.get("id"), raw, sorted(STATUS_WORDS)))
        out[it["id"]] = {"status": word, "raw_status": raw, "error": it.get("error"),
                         "emulator_fault": bool(it.get("faults") or it.get("pc") or it.get("last_api")),
                         "last_api": it.get("last_api")}
    return out


def rewrite_side(base_rows, cid):
    r = base_rows.get(cid)
    if r is None:
        return {"status": "unknown", "exit": None, "note": ""}
    code = r[1]
    word = "rendered" if code == "0" else ("refused" if code == "3" else "error")
    return {"status": word, "exit": int(code), "note": r[5] if len(r) > 5 else ""}


def load_candidates(root, testset):
    p = root / "results" / "out_of_contract.tsv"
    out = []
    if p.is_file():
        for r in read_tsv(p).values():
            out.append({"id": r[0], "params": r[1], "why": r[2],
                        "rewrite": {"check_exit": int(r[3]), "message": r[4] if len(r) > 4 else ""},
                        "original": {"status": "not_asked", "note": "the plugin was not asked: judged by the rewrite's own range check"}})
    else:
        pj = Path(testset) / "out_of_contract.json"
        if pj.is_file():
            for o in jload(pj):
                out.append({"id": o["id"], "params": ",".join(o["params"]), "why": o["why"], "rewrite": {"check_exit": None, "message": ""},
                            "original": {"status": "not_asked", "note": "not judged"}})
    return out


def judge(cases, meta, base_rows, status):
    """Decides every case. status is None (phase A: the rewrite's own refusals) or load_status's dict.
    Returns (moved_out entries, back_in_contract ids, counts of the original's words)."""
    moved, back = [], []
    words = {}
    for c in cases:
        cid = c["id"]
        rw = rewrite_side(base_rows, cid)
        st = status.get(cid) if status is not None else None
        orig_word = st["status"] if st else "not_judged"
        words[orig_word] = words.get(orig_word, 0) + 1
        reason = None
        if orig_word in ("refused", "error", "timeout"):
            reason = "the original %s" % {"refused": "refuses it", "error": "errors on it", "timeout": "timed out on it"}[orig_word]
            if orig_word == "error" and st.get("emulator_fault"):
                reason += " (an emulator fault: check the harness before trusting this)"
        elif orig_word == "rendered":
            if rw["status"] in ("refused", "error"):
                back.append(cid)      # the original renders it, the rewrite refuses: a real comparison
        else:
            if rw["status"] in ("refused", "error"):
                reason = "the rewrite refuses it and the original has not been asked"
        if reason:
            moved.append({"id": cid, "group": group_of(meta, cid), "reason": reason,
                          "original": {"status": orig_word, "error": (st or {}).get("error")},
                          "rewrite": rw, "params": c.get("params", [])})
    return moved, back, words


def contract_state(out_dir, root=None, testset=None):
    """The ooc.json of out_dir when it exists, else the phase A split (cases the base rewrite refuses
    are out of contract). Returns the ooc document plus its sha for staleness checks."""
    root = root or ROOT
    p = Path(out_dir) / "ooc.json"
    if p.is_file():
        return jload(p), sha256_file(p)
    cases, meta = load_testset(testset or (root / "testsets" / "grid"))
    base_rows = read_tsv(root / "results" / "base.grid.tsv")
    moved, back, words = judge(cases, meta, base_rows, None)
    doc = {"source": "phase A (no ooc.json): cases the base rewrite refuses", "moved_out": moved, "back_in_contract": back,
           "candidates": load_candidates(root, testset or (root / "testsets" / "grid")), "cases_total": len(cases)}
    return doc, "phase_a"


def cmd_ooc(a):
    ts = Path(a.testset)
    cases, meta = load_testset(ts)
    base_rows = read_tsv(Path(a.base)) if Path(a.base).is_file() else {}
    status = load_status(a.status)
    unknown = sorted(set(status) - {c["id"] for c in cases})
    moved, back, words = judge(cases, meta, base_rows, status)
    cands = load_candidates(ROOT, ts)
    in_contract = len(cases) - len(moved)
    doc = {"module": MODULE, "measured_at": a.date or today(), "command": "phase_b.py ooc --status %s" % a.status,
           "status_file": str(a.status), "cases_total": len(cases),
           "original_status_counts": words, "in_contract": in_contract, "moved_out": moved,
           "back_in_contract": back,
           "back_in_contract_note": "the original rendered these and the rewrite refuses them: real comparisons, expected to differ",
           "candidates": cands, "status_ids_not_in_testset": unknown,
           "out_of_contract_total": len(moved) + len(cands)}
    jsave(Path(a.out_dir) / "ooc.json", doc)
    print("ooc: %d cases, %d in contract, %d moved out (original: %s), %d back in contract, %d parameter candidates"
          % (len(cases), in_contract, len(moved), ", ".join("%s %d" % kv for kv in sorted(words.items())), len(back), len(cands)))
    for m in moved[:12]:
        print("  out  %-60s %s (original %s, rewrite %s)" % (m["id"], m["reason"], m["original"]["status"], m["rewrite"]["status"]))
    if len(moved) > 12:
        print("  ... %d more" % (len(moved) - 12))
    return 0


# ---------------------------------------------------------------------------------------------
# grade

def run_fxparity(fxp, testset, refs, vfx, max_ulp, jobs):
    cmd = [str(fxp), "run", "--testset", str(testset), "--refs", str(refs), "--max-ulp", str(max_ulp), "--json",
           "--vfx", str(vfx), "--jobs", str(jobs)]
    p = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if p.returncode not in (0, 1):
        sys.exit("phase_b: fx-parity exited %d: %s" % (p.returncode, p.stderr.strip()[:300]))
    i = p.stdout.find("{")
    try:
        doc = json.loads(p.stdout[i:])
    except (ValueError, json.JSONDecodeError):
        sys.exit("phase_b: fx-parity printed no JSON report: %s" % p.stdout[:200])
    return doc, " ".join(cmd)


def scan_case(vfx, testset, refs, case):
    """Own value level check of one case: render with vfx, compare with the reference file. None when
    there is no reference."""
    ref_path = Path(refs) / case["reference"]
    if not ref_path.is_file():
        return None
    ref = ref_path.read_bytes()
    code, out, note = render_case(vfx, testset, case)
    if code != 0:
        return {"rewrite_failed": True, "exit": code, "note": note, "values": len(ref) // SAMPLE_BYTES[case["depth"]]}
    d = diff_values(case["depth"], ref, out)
    d["rewrite_failed"] = False
    if d["first_index"] is not None:
        d["first_diff"] = locate(case["depth"], case["width"], case["height"], d["first_index"])
    return d


def build_rows(cases, meta, fx_by_id, scans, moved_ids):
    rows = []
    for c in cases:
        cid = c["id"]
        row = {"id": cid, "group": group_of(meta, cid), "tags": tags_of(meta, cid), "logic_parity": is_logic(meta, cid),
               "depth": c["depth"], "width": c["width"], "height": c["height"], "frames": c.get("frames", 1),
               "fps": c.get("fps"), "params": c.get("params", []), "layers": sorted(c.get("layers", {}))}
        if cid in moved_ids:
            row.update(status="out of contract", differing_pixels=None, differing_values=None, values_compared=None,
                       max_ulp=None, nan_mismatch=None)
            rows.append(row)
            continue
        fx = fx_by_id.get(cid)
        word = fx["status"] if fx else "no reference"
        sc = scans.get(cid)
        row["status"] = {"pass": "pass", "FAIL": "FAIL", "no reference": "no reference"}.get(word, word)
        row.update(differing_pixels=None, differing_values=None, values_compared=None, max_ulp=None, nan_mismatch=None)
        if fx and word != "no reference":
            row["differing_pixels"] = fx.get("clip_differing_pixels", fx.get("differing_pixels"))
            if "failing_frames" in fx:
                row["failing_frames"] = fx["failing_frames"]
                row["first_failing_frame"] = fx.get("frame")
            if fx.get("error"):
                row["error"] = fx["error"]
        if sc is not None:
            if sc["rewrite_failed"]:
                row["rewrite_failed"] = {"exit": sc["exit"], "note": sc["note"]}
            else:
                row["values_compared"] = sc["values"]
                row["differing_values"] = sc["differing"]
                row["max_ulp"] = sc["max_ulp"]
                row["nan_mismatch"] = sc["nan_mismatch"]
                if sc.get("size_mismatch"):
                    row["size_mismatch"] = True
                if sc.get("first_diff"):
                    row["first_diff"] = sc["first_diff"]
            scan_differs = sc["rewrite_failed"] or sc.get("size_mismatch") or sc["differing"] > 0
            if scan_differs != (row["status"] == "FAIL"):
                row["crosscheck"] = "fx-parity says %s, the value scan says %s" % (row["status"], "different" if scan_differs else "identical")
        rows.append(row)
    return rows


def aggregate(rows, ooc_doc):
    compared = [r for r in rows if r["status"] in ("pass", "FAIL")]
    ident = [r for r in compared if r["status"] == "pass"]
    diff = [r for r in compared if r["status"] == "FAIL"]
    floats = sum(r["values_compared"] or 0 for r in compared if r["depth"] == 32)
    byts = sum((r["values_compared"] or 0) * SAMPLE_BYTES[r["depth"]] for r in compared if r["depth"] in (8, 16))
    ulps = [r["max_ulp"] for r in compared if r["depth"] == 32 and r["max_ulp"] is not None]
    return {
        "cases": {
            "total": len(rows),
            "in_contract": sum(1 for r in rows if r["status"] != "out of contract"),
            "compared": len(compared), "identical": len(ident), "differing": len(diff),
            "no_reference": sum(1 for r in rows if r["status"] == "no reference"),
            "out_of_contract_cases": sum(1 for r in rows if r["status"] == "out of contract"),
            "logic_parity": sum(1 for r in compared if r["logic_parity"]),
            "logic_parity_identical": sum(1 for r in ident if r["logic_parity"]),
            "logic_parity_differing": sum(1 for r in diff if r["logic_parity"]),
            "rewrite_failed": sum(1 for r in rows if r.get("rewrite_failed")),
            "crosscheck_disagreements": sum(1 for r in rows if r.get("crosscheck")),
        },
        "values": {
            "bytes_compared": byts, "floats_compared": floats,
            "differing_values": sum(r["differing_values"] or 0 for r in compared),
            "max_ulp": max(ulps) if ulps else None,
            "nan_mismatch_values": sum(r["nan_mismatch"] or 0 for r in compared),
        },
    }


def write_grade_tsv(path, rows):
    cols = ["id", "status", "differing_pixels", "differing_values", "max_ulp", "nan_mismatch", "group", "tags", "logic_parity",
            "depth", "size", "frames", "note"]
    with open(path, "w", encoding="utf8") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            note = r.get("error") or ""
            if r.get("rewrite_failed"):
                note = "rewrite failed (exit %s): %s" % (r["rewrite_failed"]["exit"], r["rewrite_failed"]["note"])
            elif r.get("crosscheck"):
                note = r["crosscheck"]
            elif r.get("first_diff"):
                d = r["first_diff"]
                note = "first difference: frame %d x=%d y=%d %s" % (d["frame"], d["x"], d["y"], d["channel"])

            def cell(v):
                return "" if v is None else str(v)
            vals = [r["id"], r["status"], cell(r["differing_pixels"]), cell(r["differing_values"]), cell(r["max_ulp"]),
                    cell(r["nan_mismatch"]), r["group"], ",".join(r["tags"]), "yes" if r["logic_parity"] else "no",
                    str(r["depth"]), "%dx%d" % (r["width"], r["height"]), str(r["frames"]), note.replace("\t", " ").replace("\n", " ")]
            fh.write("\t".join(vals) + "\n")


def cmd_grade(a):
    ts = Path(a.testset)
    cases, meta = load_testset(ts)
    out_dir = Path(a.out_dir)
    vfx = a.vfx or default_vfx()
    fxp = find_fxparity(a.fx_parity)
    ooc_doc, ooc_sha = contract_state(out_dir, ROOT, ts)
    moved_ids = {m["id"] for m in ooc_doc["moved_out"]}
    if ooc_sha == "phase_a" and moved_ids:
        print("grade: %d cases the base rewrite refuses are out of contract until `ooc --status FILE` judges them" % len(moved_ids))
    fx_doc, fx_cmd = run_fxparity(fxp, ts, a.refs, vfx, a.max_ulp, a.jobs)
    fx_by_id = {c["id"]: c for c in fx_doc["cases"]}
    scans = {}
    if not a.no_scan:
        for c in cases:
            if c["id"] not in moved_ids:
                s = scan_case(vfx, ts, a.refs, c)
                if s is not None:
                    scans[c["id"]] = s
    rows = build_rows(cases, meta, fx_by_id, scans, moved_ids)
    agg = aggregate(rows, ooc_doc)
    doc = {"module": MODULE, "measured_at": a.date or today(), "testset": str(ts), "refs": str(a.refs), "vfx": str(vfx),
           "max_ulp_allowed": a.max_ulp, "ooc_sha": ooc_sha, "scan": not a.no_scan,
           "command": "fx-parity run --testset %s --refs %s --max-ulp %d --json --vfx %s" % (ts, a.refs, a.max_ulp, vfx),
           "fx_parity": fx_doc.get("summary"), **agg, "moved_out_ids": sorted(moved_ids), "cases_detail": rows}
    jsave(out_dir / "grade.json", doc)
    write_grade_tsv(out_dir / "grade.tsv", rows)
    c, v = agg["cases"], agg["values"]
    print("grade: %d cases, %d in contract, %d compared, %d identical, %d differing, %d no reference, %d out of contract"
          % (c["total"], c["in_contract"], c["compared"], c["identical"], c["differing"], c["no_reference"], c["out_of_contract_cases"]))
    print("       values: %d bytes, %d floats compared, %d differing values, max ULP %s, %d NaN mismatches; logic parity %d (%d differ)"
          % (v["bytes_compared"], v["floats_compared"], v["differing_values"], v["max_ulp"], v["nan_mismatch_values"],
             c["logic_parity"], c["logic_parity_differing"]))
    if c["crosscheck_disagreements"]:
        print("       WARNING: %d cases where fx-parity and the value scan disagree (see crosscheck in grade.json)" % c["crosscheck_disagreements"])
    for r in rows:
        if r["status"] == "FAIL":
            print("  FAIL %s  pixels=%s values=%s max_ulp=%s %s" % (r["id"], r["differing_pixels"], r["differing_values"], r["max_ulp"],
                                                               r.get("error", "")[:80] if r.get("error") else ""))
    return 0


# ---------------------------------------------------------------------------------------------
# call records

def read_lines(path):
    out = []
    with open(path, encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"ev": "unparsable", "text": line[:80]})
    return out


def load_layer_map(param_map_path, plugin=PLUGIN_STEM):
    """ae param index -> rewrite layer id, from param_map.tsv (index 0 is the input). Empty beyond 0
    when the map has no row of the plugin (no plugin has reached PARAMS_SETUP yet)."""
    layers = {0: "input"}
    p = Path(param_map_path) if param_map_path else None
    if p and p.is_file():
        with open(p, encoding="utf8") as fh:
            head = fh.readline().rstrip("\n").split("\t")
            for line in fh:
                f = dict(zip(head, line.rstrip("\n").split("\t")))
                if f.get("plugin") == plugin and f.get("vfx_kind") == "layer" and f.get("ae_index", "").strip().isdigit():
                    layers[int(f["ae_index"])] = f["vfx_id"]
    return layers


def parse_rewrite_record(lines):
    """VFX_CALL_RECORD lines -> {"run", "frames": {n: seconds}, "reads": {n: [read]}}. A read is
    {"layer", "secs", "index" (None outside the clip), "clip_frames"}; seconds are exact fractions."""
    run, frames, reads = None, {}, {}
    scale, cur = None, 0
    for d in lines:
        ev = d.get("ev")
        if ev == "run":
            run = d
            scale = d.get("time_scale")
        elif ev == "frame":
            cur = d["out_frame"]
            scale = d.get("time_scale", scale)
            frames[cur] = Fraction(d["time"], scale)
            reads.setdefault(cur, [])
        elif ev == "read":
            n = d.get("out_frame", cur)
            reads.setdefault(n, []).append({"layer": d["layer"], "secs": Fraction(d["time"], scale), "index": d.get("index"),
                                            "clip_frames": d.get("clip_frames")})
    return {"run": run, "frames": frames, "reads": reads}


def parse_plugin_record(lines, layer_map):
    """The harness's records/<case>.jsonl -> the same shape as parse_rewrite_record. Layer reads come
    from `prerender_cb.checkout_layer` (smart path) and, for a layer parameter, `inter.checkout_param`
    (classic path); a `layer` field on the line wins over the param index. Other parameter checkouts are
    collected in param_reads (the rewrite does not record parameter sample times)."""
    frames, reads, param_reads = {}, {}, []
    cur = 0
    incomplete = None
    unimplemented = set()
    saw_frame = False
    for d in lines:
        ev = d.get("ev")
        if ev == "frame":
            saw_frame = True
            cur = d["out_frame"]
            frames[cur] = Fraction(d["time"], d["time_scale"])
            reads.setdefault(cur, [])
        elif ev == "case_error":
            incomplete = "case_error: %s" % str(d.get("error", ""))[:100]
        elif ev == "cb":
            fn = d.get("fn", "")
            if d.get("unimplemented"):
                unimplemented.add(fn)
            if fn not in ("prerender_cb.checkout_layer", "inter.checkout_param"):
                continue
            args = d.get("args") or []
            pidx = d.get("param_index", args[1] if len(args) > 1 else None)
            t = d.get("time", args[2] if len(args) > 2 else None)
            scale = d.get("time_scale", args[4] if len(args) > 4 else None)
            if t is None or scale in (None, 0):
                continue
            n = d.get("out_frame", cur)
            layer = d.get("layer")
            if layer is None and pidx is not None:
                layer = layer_map.get(pidx)
            if layer is None and fn == "prerender_cb.checkout_layer":
                layer = "param#%s" % pidx
            if layer is None:
                param_reads.append({"out_frame": n, "param_index": pidx, "secs": Fraction(t, scale)})
                continue
            reads.setdefault(n, []).append({"layer": layer, "secs": Fraction(t, scale), "footage_frame": d.get("footage_frame"),
                                            "inside": d.get("inside")})
    if not saw_frame and incomplete is None:
        incomplete = "no frame event"
    return {"frames": frames, "reads": reads, "param_reads": param_reads, "incomplete": incomplete,
            "unimplemented": sorted(unimplemented)}


def _fmt_read(layer, secs, idx=None):
    s = "%s@%.6gs" % (layer, float(secs))
    return s if idx is None else "%s[%s]" % (s, idx)


def _index_matches(rew_idx, clip_frames, p):
    """Does the plugin's footage frame / inside flag agree with the rewrite's clip frame index?"""
    ff, inside = p.get("footage_frame"), p.get("inside")
    if rew_idx is None:
        if inside is False:
            return True
        if inside is None and ff is not None and clip_frames is not None:
            return ff < 0 or ff >= clip_frames
        return inside is None and ff is None
    return inside is not False and ff == rew_idx


def compare_records(plug, rew, exclude_layers=(), ordered=False, ignore_outside=False, case_params_list=None):
    """Compares one case frame by frame; reads as sets of (layer, time) unless ordered. Reads are
    keyed by exact time, so a vector layer read by time at a rate below the clip rate, or past the end
    of a short clip, compares like any other."""
    exclude = set(exclude_layers)
    clip_frames = {}
    for rs in rew["reads"].values():
        for r in rs:
            if r["clip_frames"] is not None:
                clip_frames[r["layer"]] = r["clip_frames"]
    problems_first = None
    mismatch_frames = []
    nframes = 0
    frames = sorted(set(plug["frames"]) | set(rew["frames"]))
    run_ok = True
    if case_params_list is not None and rew.get("run") is not None:
        if sorted(rew["run"].get("params", [])) != sorted(case_params_list):
            run_ok = False
            problems_first = {"out_frame": None, "what": "the rewrite record's run params %s differ from the case's %s"
                              % (rew["run"].get("params"), case_params_list)}
    for n in frames:
        nframes += 1
        prob = {}
        pt, rt = plug["frames"].get(n), rew["frames"].get(n)
        if pt is None:
            prob["frame_missing_in_plugin"] = True
        elif rt is None:
            prob["frame_missing_in_rewrite"] = True
        elif pt != rt:
            prob["frame_time"] = {"plugin": str(pt), "rewrite": str(rt)}
        prs = [r for r in plug["reads"].get(n, []) if r["layer"] not in exclude]
        rrs = [r for r in rew["reads"].get(n, []) if r["layer"] not in exclude]
        if ignore_outside:
            prs = [r for r in prs if not (r.get("inside") is False or (r.get("inside") is None and r.get("footage_frame") is not None
                                                                          and clip_frames.get(r["layer"]) is not None
                                                                          and not 0 <= r["footage_frame"] < clip_frames[r["layer"]]))]
            rrs = [r for r in rrs if r["index"] is not None]
        pk = {}
        for r in prs:
            pk.setdefault((r["layer"], r["secs"]), []).append(r)
        rk = {}
        for r in rrs:
            rk.setdefault((r["layer"], r["secs"]), []).append(r)
        missing = sorted(set(rk) - set(pk), key=lambda k: (k[0], k[1]))
        extra = sorted(set(pk) - set(rk), key=lambda k: (k[0], k[1]))
        idx_diff = []
        for k in sorted(set(pk) & set(rk), key=lambda k: (k[0], k[1])):
            ri = rk[k][0]
            if not any(_index_matches(ri["index"], clip_frames.get(k[0]), p) for p in pk[k]):
                idx_diff.append("%s: rewrite frame %s, plugin frame %s" % (_fmt_read(*k), ri["index"], pk[k][0].get("footage_frame")))
        if missing:
            prob["missing_in_plugin"] = [_fmt_read(l, t, rk[(l, t)][0]["index"]) for l, t in missing]
        if extra:
            prob["extra_in_plugin"] = [_fmt_read(l, t, pk[(l, t)][0].get("footage_frame")) for l, t in extra]
        if idx_diff:
            prob["index_differs"] = idx_diff
        hints = []
        for (ml, mt) in missing:
            for (el, et) in extra:
                if ml == el and mt != et:
                    hints.append("wrong time on %s: rewrite %.6gs, plugin %.6gs" % (ml, float(mt), float(et)))
                elif mt == et and ml != el:
                    hints.append("wrong layer at %.6gs: rewrite %s, plugin %s" % (float(mt), ml, el))
        if hints:
            prob["hints"] = hints
        if ordered and not prob:
            po = [(r["layer"], r["secs"]) for r in prs]
            ro = [(r["layer"], r["secs"]) for r in rrs]
            if po != ro:
                prob["order_differs"] = {"plugin": [_fmt_read(*k) for k in po], "rewrite": [_fmt_read(*k) for k in ro]}
        if prob:
            mismatch_frames.append(n)
            if problems_first is None:
                problems_first = dict(out_frame=n, **prob)
    return {"frames_compared": nframes, "frames_matching": nframes - len(mismatch_frames),
            "match": not mismatch_frames and run_ok, "first_mismatch": problems_first, "mismatch_frames": mismatch_frames}


def cmd_calls(a):
    ts = Path(a.testset)
    cases, meta = load_testset(ts)
    out_dir = Path(a.out_dir)
    ooc_doc, ooc_sha = contract_state(out_dir, ROOT, ts)
    moved = {m["id"] for m in ooc_doc["moved_out"]}
    layer_map = load_layer_map(a.param_map)
    reads_names = callrec.layer_names(a.param_map) if a.rule == "reads" else None
    results = []
    for c in cases:
        cid = c["id"]
        if cid in moved:
            results.append({"id": cid, "status": "out_of_contract"})
            continue
        pp, rp = Path(a.plugin_records) / (cid + ".jsonl"), Path(a.rewrite_records) / (cid + ".jsonl")
        if not pp.is_file():
            results.append({"id": cid, "status": "no_plugin_record"})
            continue
        if not rp.is_file():
            results.append({"id": cid, "status": "no_rewrite_record"})
            continue
        if a.rule == "reads":
            cmpd = callrec.compare_case(read_lines(pp), read_lines(rp), reads_names)
            if cmpd["status"] == "plugin_record_incomplete":
                results.append({"id": cid, "status": "plugin_record_incomplete", "why": "no case_done line with status rendered"})
                continue
            results.append({"id": cid, "group": group_of(meta, cid), "status": cmpd["status"],
                            "frames_compared": cmpd["frames_compared"], "frames_matching": cmpd["frames_matching"],
                            "first_mismatch": cmpd["first_mismatch"], "mismatch_frames": cmpd["mismatch_frames"],
                            "read_classes": cmpd["classes"]})
            continue
        plug = parse_plugin_record(read_lines(pp), layer_map)
        if plug["incomplete"]:
            results.append({"id": cid, "status": "plugin_record_incomplete", "why": plug["incomplete"]})
            continue
        rew = parse_rewrite_record(read_lines(rp))
        cmpd = compare_records(plug, rew, a.exclude_layers, a.ordered, a.ignore_outside, c.get("params", []))
        entry = {"id": cid, "group": group_of(meta, cid), "status": "match" if cmpd["match"] else "mismatch", **cmpd,
                 "plugin_param_reads": len(plug["param_reads"]), "plugin_unimplemented_callbacks": plug["unimplemented"]}
        results.append(entry)
    both = [r for r in results if r["status"] in ("match", "mismatch")]
    doc = {"module": MODULE, "measured_at": a.date or today(), "ooc_sha": ooc_sha,
           "rule": a.rule,
           "command": "phase_b.py calls --rule %s --plugin-records %s --rewrite-records %s%s%s%s" % (
               a.rule, a.plugin_records, a.rewrite_records, " --ordered" if a.ordered else "",
               " --ignore-outside" if a.ignore_outside else "", "".join(" --exclude-layers " + x for x in a.exclude_layers)),
           "ordered": a.ordered, "exclude_layers": list(a.exclude_layers), "ignore_outside": a.ignore_outside,
           "cases": len(results), "compared": len(both), "matching": sum(1 for r in both if r["status"] == "match"),
           "frames_compared": sum(r["frames_compared"] for r in both), "frames_matching": sum(r["frames_matching"] for r in both),
           "not_compared": {k: sum(1 for r in results if r["status"] == k) for k in
                            ("no_plugin_record", "no_rewrite_record", "plugin_record_incomplete", "out_of_contract")},
           "rule_note": ("reads: pixel checkouts of the plugin against the rewrite's reads, per frame; the registrations of the "
                         "pre-render call are declarations and are not reads (callrec.py)" if a.rule == "reads" else
                         "registrations: every checkout of the plugin, registrations included, against the rewrite's reads"),
           "param_times_note": "the rewrite does not record parameter sample times, so the plugin's parameter checkouts are counted and not compared",
           "mismatches": [{"id": r["id"], "group": r["group"], "first_mismatch": r["first_mismatch"], "mismatch_frames": r["mismatch_frames"]}
                          for r in both if r["status"] == "mismatch"],
           "per_case": results}
    jsave(out_dir / "calls.json", doc)
    print("calls: %d cases, %d compared, %d matching, %d frames compared (%d matching); not compared %s"
          % (doc["cases"], doc["compared"], doc["matching"], doc["frames_compared"], doc["frames_matching"],
             {k: v for k, v in doc["not_compared"].items() if v}))
    for m in doc["mismatches"][:15]:
        fm = m["first_mismatch"] or {}
        print("  MISMATCH %s first at output frame %s: %s" % (m["id"], fm.get("out_frame"), {k: v for k, v in fm.items() if k != "out_frame"}))
    return 0


def default_record_vfx():
    """A vfx build that writes VFX_CALL_RECORD files. bin/vfx-base and the mutant builds come from a source
    copy made before the call record existed and write none, so prefer a build of the current tree."""
    for c in (ROOT / "bin" / "vfx-record", Path.home() / "twixtor-parity" / "harness" / "repo" / "target" / "release" / "vfx", ROOT / "bin" / "vfx-base"):
        if Path(c).is_file():
            return str(c)
    sys.exit("phase_b: no vfx found for the call records; pass --vfx")


def cmd_rewrite_records(a):
    """Runs the rewrite with VFX_CALL_RECORD over the test set, one record per case. Checks that the build
    writes records at all and that its output equals the base rewrite's (results/base.grid.tsv): a record
    of a build that renders differently is not the base rewrite's record."""
    ts = Path(a.testset)
    cases, _ = load_testset(ts)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vfx = a.vfx or default_record_vfx()
    only = set(a.only.split(",")) if a.only else None
    base_path = ROOT / "results" / "base.grid.tsv"
    base_rows = read_tsv(base_path) if base_path.is_file() else {}
    n = bad = 0
    no_record, differs = [], []
    for c in cases:
        if only and c["id"] not in only:
            continue
        rec = out / (c["id"] + ".jsonl")
        if rec.exists():
            rec.unlink()
        code, stdout, note = render_case(vfx, ts, c, record=rec)
        n += 1
        if code != 0:
            bad += 1
        if not rec.is_file():
            no_record.append(c["id"])
        b = base_rows.get(c["id"])
        if b is not None and b[1] == "0" and code == 0 and sha256_bytes(stdout) != b[3]:
            differs.append(c["id"])
    jsave(out / "_index.json", {"vfx": str(vfx), "cases": n, "exit_nonzero": bad, "no_record": no_record, "output_differs_from_base": differs})
    print("rewrite-records: %s: %d cases to %s; %d exited non-zero (their records stop at the failing frame); %d wrote no record; "
          "%d render differently from the base rewrite" % (vfx, n, out, bad, len(no_record), len(differs)))
    if n and len(no_record) == n:
        print("  this vfx wrote no record at all: it was built before VFX_CALL_RECORD existed; pass --vfx with a build of the current tree")
        return 2
    if differs:
        print("  WARNING: the record build is not the base rewrite on %s ..." % differs[:3])
    return 0


# ---------------------------------------------------------------------------------------------
# mutation kills

def mutant_ids(muts, only):
    ids = [m["id"] for m in muts]
    if only:
        want = only.split(",")
        for w in want:
            if w not in ids and w != "m00":
                sys.exit("phase_b: no mutant %s in the list" % w)
        ids = [w for w in want]
    return ids


def compute_kills(ref_sha, base_rows, mutant_rows, in_contract_ids):
    """The kill rule. D = in-contract cases that have a reference and where the base rewrite's output
    equals it. A mutant is killed iff its output (or a failure to render) differs from the reference on
    at least one case of D. mutant_rows: {mutant id: {case id: [id, exit, size, sha256, ...]} or None}."""
    base_equal, base_differs, no_ref = [], [], []
    for cid in sorted(in_contract_ids):
        if cid not in ref_sha:
            no_ref.append(cid)
            continue
        b = base_rows.get(cid)
        if b is not None and b[1] == "0" and b[3] == ref_sha[cid]:
            base_equal.append(cid)
        else:
            base_differs.append(cid)
    D = set(base_equal)
    out = []
    for mid, rows in mutant_rows.items():
        if rows is None:
            out.append({"id": mid, "status": "not_run", "killers": [], "failed_cases": [], "changed_vs_base_in_D": 0})
            continue
        killers, failed, changed = [], [], 0
        for cid in base_equal:
            r = rows.get(cid)
            if r is None:
                continue
            if r[1] != "0":
                failed.append(cid)
                killers.append(cid)
            elif r[3] != ref_sha[cid]:
                killers.append(cid)
            if r[1] != "0" or r[3] != base_rows[cid][3]:
                changed += 1
        outside = sum(1 for cid in base_differs if cid in rows and (rows[cid][1] != base_rows[cid][1] or rows[cid][3] != base_rows[cid][3]))
        status = "killed" if killers else ("survivor" if D else "not_judged")
        out.append({"id": mid, "status": status, "killers": killers, "failed_cases": failed, "changed_vs_base_in_D": changed,
                    "changed_outside_D": outside})
    return {"discriminating": sorted(D), "base_differs_from_reference": base_differs, "no_reference": no_ref, "mutants": out}


def cmd_kills(a):
    ts = Path(a.testset)
    cases, meta = load_testset(ts)
    out_dir = Path(a.out_dir)
    ooc_doc, ooc_sha = contract_state(out_dir, ROOT, ts)
    moved = {m["id"] for m in ooc_doc["moved_out"]}
    in_contract = [c["id"] for c in cases if c["id"] not in moved]
    refs = Path(a.refs)
    by_id = {c["id"]: c for c in cases}
    ref_sha = {}
    for cid in in_contract:
        p = refs / by_id[cid]["reference"]
        if p.is_file():
            ref_sha[cid] = sha256_file(p)
    muts = jload(a.mutants)
    meta_of = {m["id"]: m for m in muts}
    meta_of["m00"] = {"id": "m00", "kind": "control", "desc": "the unmutated control build: must survive"}
    ids = mutant_ids(muts, a.only)
    bindir = Path(a.bin_dir)
    work = out_dir / "kills_work"
    env_jobs = max(1, a.jobs)
    if a.rerender:
        work.mkdir(parents=True, exist_ok=True)
        base_vfx = a.base_vfx or str(bindir / "vfx-m00")
        base_rows = render_rows(base_vfx, ts, [by_id[c] for c in sorted(ref_sha)], env_jobs)
        # base rows are only needed on cases with a reference
    else:
        base_rows = read_tsv(Path(a.results) / "base.grid.tsv")
    mutant_rows = {}
    for mid in ids:
        if a.rerender:
            vfx = bindir / ("vfx-%s" % mid)
            if not vfx.is_file():
                mutant_rows[mid] = None
                continue
            need = [by_id[c] for c in sorted(ref_sha) if base_rows.get(c) and base_rows[c][1] == "0" and base_rows[c][3] == ref_sha[c]]
            rows = render_rows(str(vfx), ts, need, env_jobs)
            mutant_rows[mid] = rows
            write_rows_tsv(work / (mid + ".tsv"), rows)
        else:
            p = Path(a.results) / "mutants" / (mid + ".grid.tsv")
            mutant_rows[mid] = read_tsv(p) if p.is_file() else None
    if not a.rerender:
        for mid, rows in mutant_rows.items():
            if rows is not None and set(rows) != set(by_id):
                sys.exit("phase_b: %s.grid.tsv does not cover this test set's cases; rerun with --rerender" % mid)
    res = compute_kills(ref_sha, base_rows, mutant_rows, in_contract)
    for m in res["mutants"]:
        m["kind"] = meta_of.get(m["id"], {}).get("kind", "")
        m["desc"] = meta_of.get(m["id"], {}).get("desc", "")
    real = [m for m in res["mutants"] if m["id"] != "m00"]
    killed = [m["id"] for m in real if m["status"] == "killed"]
    doc = {"module": MODULE, "measured_at": a.date or today(), "ooc_sha": ooc_sha, "refs": str(refs),
           "mode": "rerender" if a.rerender else "saved phase A shas",
           "command": "phase_b.py kills --refs %s%s%s" % (a.refs, " --rerender" if a.rerender else "", " --only " + a.only if a.only else ""),
           "discriminating_cases": len(res["discriminating"]),
           "base_differs_from_reference": len(res["base_differs_from_reference"]),
           "in_contract_without_reference": len(res["no_reference"]),
           "total": len(real), "killed": len(killed), "survivors": [m["id"] for m in real if m["status"] != "killed"],
           "control": next((m for m in res["mutants"] if m["id"] == "m00"), None),
           "mutants": real}
    jsave(out_dir / "kills.json", doc)
    print("kills: %d discriminating cases (base equals reference), %d where the base differs, %d without reference"
          % (doc["discriminating_cases"], doc["base_differs_from_reference"], doc["in_contract_without_reference"]))
    print("       mutants: %d total, %d killed, survivors %s" % (doc["total"], doc["killed"], doc["survivors"]))
    if doc["control"]:
        print("       control m00: %s" % doc["control"]["status"])
    for m in real:
        print("  %-4s %-9s killers %4d (%d failed to render), changes %d discriminating cases" % (
            m["id"], m["status"], len(m["killers"]), len(m["failed_cases"]), m["changed_vs_base_in_D"]))
    return 0


def render_rows(vfx, testset, cases, jobs):
    from concurrent.futures import ThreadPoolExecutor
    env = dict(os.environ, RAYON_NUM_THREADS="1")
    env.pop("VFX_CALL_RECORD", None)
    with ThreadPoolExecutor(jobs) as ex:
        rows = list(ex.map(lambda c: run_grid.run_one(str(vfx), str(testset), None, c, env), cases))
    return {r[0]: r for r in rows}


def write_rows_tsv(path, rows):
    with open(path, "w", encoding="utf8") as fh:
        fh.write("id\texit\tsize\tsha256\tms\tnote\n")
        for k in sorted(rows):
            fh.write("\t".join(rows[k]) + "\n")


# ---------------------------------------------------------------------------------------------
# summary

def unmet_rule(s):
    out = []
    c, v, k, m = s["cases"], s["values"], s["call_records"], s["mutants"]
    if c["compared"] < MIN_CASES:
        out.append("fewer than %d in-contract cases compared" % MIN_CASES)
    if c["compared"] < c["in_contract"]:
        out.append("not every in-contract case compared")
    if c["differing"] or v["differing_values"] or c["identical"] != c["compared"]:
        out.append("a case or value differs")
    if not k["compared"] or k["matching"] != k["compared"]:
        out.append("call records not all matching")
    if not m["total"] or m["killed"] != m["total"] or m["survivors"]:
        out.append("a mutant survives or none ran")
    return out


def build_summary(grade, ooc_doc, kills, calls, fixes, plugin_sha, date_text, meta, extra_blockers, commands):
    scan_ok = grade.get("scan", True)
    c, v = grade["cases"], grade["values"]
    moved = {m["id"] for m in ooc_doc["moved_out"]}
    rows = [r for r in grade["cases_detail"]]
    in_contract = len(rows) - len(moved)
    compared = [r for r in rows if r["status"] in ("pass", "FAIL") and r["id"] not in moved]
    cases = {"in_contract": in_contract, "compared": len(compared),
             "identical": sum(1 for r in compared if r["status"] == "pass"),
             "differing": sum(1 for r in compared if r["status"] == "FAIL"),
             "out_of_contract": len(moved) + len(ooc_doc.get("candidates", [])),
             "logic_parity": sum(1 for r in compared if r["logic_parity"])}
    values = {"bytes_compared": v["bytes_compared"], "floats_compared": v["floats_compared"],
              "differing_values": v["differing_values"], "max_ulp": v["max_ulp"]}
    calls_sec = {"compared": calls["compared"], "matching": calls["matching"]} if calls else {"compared": 0, "matching": 0}
    if kills:
        mut = {"total": kills["total"], "killed": kills["killed"], "survivors": list(kills["survivors"])}
    else:
        mut = {"total": 0, "killed": 0, "survivors": []}
    stand = sum(1 for cid in meta if meta[cid].get("stand_in"))
    lp = cases["logic_parity"]
    stand_ins = ["AE keyframe interpolation: %d grid cases key a parameter (tag keyframe_interpolation); both sides read the same labelled "
                 "interpolation of the keys, and those cases count as logic_parity (%d compared, %d identical)"
                 % (stand, lp, sum(1 for r in compared if r["logic_parity"] and r["status"] == "pass"))]
    s = {"module": MODULE, "plugin": PLUGIN, "plugin_sha256": plugin_sha, "effect": EFFECT, "status": "in_progress",
         "blockers": list(extra_blockers), "cases": cases, "values": values, "call_records": calls_sec, "mutants": mut,
         "stand_ins": stand_ins, "fixes": list(fixes), "commands": commands, "measured_at": date_text}
    unmet = unmet_rule(s)
    if not scan_ok:
        unmet.append("grade ran with --no-scan: differing values were not counted")
    if not unmet:
        s["status"] = "verified"
    elif extra_blockers:
        s["status"] = "blocked"
    return s, unmet


def cmd_summary(a):
    out_dir = Path(a.out_dir)
    gp = out_dir / "grade.json"
    if not gp.is_file():
        sys.exit("phase_b: %s does not exist; run `grade --refs DIR` first (summary reads every count from the result files)" % gp)
    grade = jload(gp)
    ooc_doc, ooc_sha = contract_state(out_dir, ROOT, a.testset)
    stale = []
    if grade.get("ooc_sha") != ooc_sha:
        stale.append("grade.json was made against another contract (ooc.json changed): run grade again")
    kills = jload(out_dir / "kills.json") if (out_dir / "kills.json").is_file() else None
    calls = jload(out_dir / "calls.json") if (out_dir / "calls.json").is_file() else None
    for name, d in (("kills.json", kills), ("calls.json", calls)):
        if d and d.get("ooc_sha") != ooc_sha:
            stale.append("%s was made against another contract (ooc.json changed): run it again" % name)
    if stale and not a.allow_stale:
        sys.exit("phase_b: " + "; ".join(stale) + " (--allow-stale overrides)")
    fixes = jload(out_dir / "fixes.json") if (out_dir / "fixes.json").is_file() else []
    sha_path = ROOT / "results" / "plugin_sha.txt"
    plugin_sha = sha_path.read_text().split()[0] if sha_path.is_file() else ""
    commands = list(PHASE_A_COMMANDS)
    for d in (ooc_doc if ooc_sha != "phase_a" else None, grade, kills, calls):
        if d and d.get("command"):
            commands.append(d["command"])
    commands.append("python3 tools/phase_b.py summary")
    _, meta = load_testset(a.testset)
    s, unmet = build_summary(grade, ooc_doc, kills, calls, fixes, plugin_sha, a.date or today(), meta, a.blocker, commands)
    out = Path(a.out) if a.out else out_dir / "summary.json"
    jsave(out, s)
    print("summary: %s status %s; compared %d of %d in contract, %d differing, %d out of contract; calls %d/%d; mutants %d/%d killed"
          % (out, s["status"], s["cases"]["compared"], s["cases"]["in_contract"], s["cases"]["differing"], s["cases"]["out_of_contract"],
             s["call_records"]["matching"], s["call_records"]["compared"], s["mutants"]["killed"], s["mutants"]["total"]))
    if unmet:
        print("         rule unmet: " + "; ".join(unmet))
    return 0


# ---------------------------------------------------------------------------------------------
# triage

def triage_report(rows, cases_by_id, meta):
    """Which settings co-occur with differences, as counts per value. rows: grade case rows."""
    compared = [r for r in rows if r["status"] in ("pass", "FAIL")]
    diff = [r for r in compared if r["status"] == "FAIL"]

    def count(keyfn):
        tot, bad = {}, {}
        for r in compared:
            for k in keyfn(r):
                tot[k] = tot.get(k, 0) + 1
                if r["status"] == "FAIL":
                    bad[k] = bad.get(k, 0) + 1
        return [(k, bad[k], tot[k]) for k in bad]

    def setting_keys(r):
        c = cases_by_id[r["id"]]
        keys = ["%s=%s" % (k, v) for k, v in sorted(case_params(c).items())]
        keys.append("depth=%d" % c["depth"])
        keys.append("size=%dx%d" % (c["width"], c["height"]))
        keys.append("frames=%d" % c.get("frames", 1))
        keys.append("fps=%s" % c.get("fps"))
        keys.append("layers=" + ("+".join(sorted(c.get("layers", {}))) or "none"))
        return keys

    def sort_rank(lst):
        return sorted(lst, key=lambda t: (-t[1] / t[2], -t[1], t[0]))

    return {"compared": len(compared), "differing": len(diff),
            "by_group": sort_rank(count(lambda r: [r["group"]])),
            "by_tag": sort_rank(count(lambda r: r["tags"])),
            "by_setting": sort_rank(count(setting_keys))}


def smallest_failing(rows, cases_by_id):
    diff = [r for r in rows if r["status"] == "FAIL"]

    def key(r):
        c = cases_by_id[r["id"]]
        nondefault = len([p for p in c.get("params", []) if p not in GRID_BASE_PARAMS])
        return (c["width"] * c["height"], c.get("frames", 1), c["depth"], nondefault, c["id"])
    return sorted(diff, key=key)


def _probe_id(base_label, c, params, layers):
    h = hashlib.sha1(json.dumps([c["input"], params, sorted(layers.items())]).encode()).hexdigest()[:6]
    return "%s.probe-%s.%dx%d.d%d.%s" % (EFFECT, base_label, c["width"], c["height"], c["depth"], h)


def build_ladder(target, cases, meta, probe_dir, testset):
    """A ladder of narrower probe cases around the smallest failing case: the case with each
    non-default setting reverted one at a time, the default case with each setting alone, the layer
    subsets, and the case's params on the inputs of the grid's default-parameter cases. Every case
    reuses input files of the grid (paths relative to the probe folder)."""
    probe_dir = Path(probe_dir)
    rel = os.path.relpath(Path(testset), probe_dir).replace("\\", "/")

    def reroot(c):
        c = dict(c)
        c["input"] = rel + "/" + c["input"]
        if c.get("layers"):
            c["layers"] = {k: rel + "/" + v for k, v in c["layers"].items()}
        return c

    tp = case_params(target)
    base = dict(p.partition("=")[::2] for p in GRID_BASE_PARAMS)
    nondef = [(k, v) for k, v in tp.items() if base.get(k) != v]
    rungs = []

    def add(label, what, params, layers=None, src=None):
        src = src or target
        lay = dict(src.get("layers", {})) if layers is None else layers
        c = dict(src)
        c["params"] = params
        c["layers"] = lay
        c["label"] = "probe-" + label
        c["id"] = _probe_id(label, src, params, lay)
        c["reference"] = "%s.%s" % (c["id"], PIX_FMT[c["depth"]])
        rungs.append({"rung": label, "what": what, "case": c, "src": src})

    plist = lambda d: ["%s=%s" % kv for kv in d.items()]
    add("self", "the failing case itself, unchanged (it must fail again)", list(target.get("params", [])))
    add("base-params", "its inputs with the grid's base params only (%s): every extra setting reverted" % ", ".join(GRID_BASE_PARAMS), list(GRID_BASE_PARAMS))
    add("plugin-defaults", "its inputs with no param given at all (the plugin's own defaults)", [])
    for k, v in nondef:
        rev = dict(tp)
        if k in base:
            rev[k] = base[k]
        else:
            rev.pop(k)
        add("rev-%s" % k, "the case with %s reverted to %s" % (k, base.get(k, "default")), plist(rev))
    for k, v in nondef:
        one = dict(base)
        one[k] = v
        add("only-%s" % k, "the base params with only %s=%s changed" % (k, v), plist(one))
    names = sorted(target.get("layers", {}))
    for sub in (["vectors_prev"], ["vectors_next"], []):
        if set(sub) != set(names) and set(sub) <= set(names):
            add("layers-%s" % ("+".join(s.replace("vectors_", "") for s in sub) or "none"),
                "the case with only the layers %s" % (sub or "none"), list(target.get("params", [])),
                layers={k: v for k, v in target["layers"].items() if k in sub})
    seen = set()
    for c in cases:
        cp = list(c.get("params", []))
        if c["id"] == target["id"] or sorted(cp) != sorted(GRID_BASE_PARAMS) and cp != []:
            continue
        if set(c.get("layers", {})) != {"vectors_prev", "vectors_next"} or c["input"] == target["input"]:
            continue
        k = (c["width"], c["height"], c["depth"])
        if k in seen or (c["width"], c["height"]) != (target["width"], target["height"]):
            continue
        seen.add(k)
        add("inputs-d%d" % c["depth"], "the case's params on the inputs of %s" % c["id"], list(target.get("params", [])),
            layers=dict(c["layers"]), src=dict(c, params=list(target.get("params", []))))
    out_cases = []
    for r in rungs:
        c = r["case"]
        # an inputs rung takes the other case's input file, size and depth
        out_cases.append(reroot(c))
    return rungs, out_cases


def cmd_triage(a):
    out_dir = Path(a.out_dir)
    gp = out_dir / "grade.json"
    if not gp.is_file():
        sys.exit("phase_b: %s does not exist; run `grade` first" % gp)
    grade = jload(gp)
    cases, meta = load_testset(a.testset)
    by_id = {c["id"]: c for c in cases}
    rows = grade["cases_detail"]
    rep = triage_report(rows, by_id, meta)
    print("triage: %d compared, %d differ" % (rep["compared"], rep["differing"]))
    if not rep["differing"]:
        print("nothing differs: no ladder to build")
        jsave(out_dir / "triage.json", rep)
        return 0
    for title, key in (("by group", "by_group"), ("by tag", "by_tag")):
        print("\ndiffering / compared %s:" % title)
        for k, b, t in rep[key][:20]:
            print("  %-40s %4d / %-4d (%3d%%)" % (k, b, t, round(100.0 * b / t)))
    print("\nsettings that co-occur with differences (differing / compared with that value):")
    for k, b, t in rep["by_setting"][:a.top]:
        print("  %-52s %4d / %-4d (%3d%%)%s" % (k, b, t, round(100.0 * b / t), "  <- every case with it differs" if b == t and t > 1 else ""))
    small = smallest_failing(rows, by_id)
    print("\nsmallest failing cases (size, frames, depth):")
    for r in small[:10]:
        c = by_id[r["id"]]
        print("  %dx%d  %d frames  d%d  %s  values=%s max_ulp=%s" % (c["width"], c["height"], c.get("frames", 1), c["depth"], r["id"],
                                                                 r["differing_values"], r["max_ulp"]))
    target = by_id[small[0]["id"]]
    probe_dir = Path(a.probe_dir) if a.probe_dir else ROOT / "probe"
    rungs, out_cases = build_ladder(target, cases, meta, probe_dir, a.testset)
    inputs = []
    seen = set()
    rel = os.path.relpath(Path(a.testset), probe_dir).replace("\\", "/")
    for c in out_cases:
        for f in [c["input"]] + list(c.get("layers", {}).values()):
            if f not in seen:
                seen.add(f)
    ts_doc = jload(Path(a.testset) / "cases.json")
    for ent in ts_doc.get("inputs", []):
        f = rel + "/" + ent["file"]
        if f in seen:
            inputs.append(dict(ent, file=f))
    doc = {"format": 1, "ffmpeg": "none (the ladder reuses the grid's raw inputs)", "inputs": inputs, "cases": out_cases}
    probe_dir.mkdir(parents=True, exist_ok=True)
    jsave(probe_dir / "cases.json", doc)
    pmeta = {}
    for r, c in zip(rungs, out_cases):
        m = dict(meta.get(r["src"]["id"], {}))
        m.update(group="probe", flags=[], ladder_of=target["id"], rung=r["rung"])
        m["stand_in"] = "keyframe_interpolation" if any("@" in v for v in case_params(c).values()) else None
        pmeta[c["id"]] = m
    jsave(probe_dir / "case_meta.json", pmeta)
    jsave(probe_dir / "ladder.json", {"target": target["id"], "target_params": target.get("params", []),
                                      "rungs": [{"id": c["id"], "rung": r["rung"], "what": r["what"], "params": c["params"],
                                                 "layers": sorted(c.get("layers", {}))} for r, c in zip(rungs, out_cases)]})
    rep["smallest_failing"] = target["id"]
    rep["ladder"] = str(probe_dir)
    jsave(out_dir / "triage.json", rep)
    print("\nladder for %s: %d probe cases in %s" % (target["id"], len(out_cases), probe_dir))
    for r, c in zip(rungs, out_cases):
        print("  %-18s %s" % (r["rung"], r["what"]))
    print("run the ladder through the harness, then grade it:\n"
          "  python run_harness.py --testset %s --refs %s/refs --records %s/records --summary %s/run_summary.json -j1\n"
          "  python3 tools/phase_b.py grade --testset %s --refs %s/refs --out-dir %s/results"
          % (probe_dir, probe_dir, probe_dir, probe_dir, probe_dir, probe_dir, probe_dir))
    return 0


# ---------------------------------------------------------------------------------------------
# self test (stand-in oracles, no plugin)

class Check:
    def __init__(self):
        self.items = []

    def ok(self, name, cond, detail=""):
        self.items.append({"check": name, "ok": bool(cond), "detail": detail})
        print("  %s %s%s" % ("ok  " if cond else "FAIL", name, ("  (" + str(detail) + ")") if detail else ""))
        return cond


def flip_bit(path, offset=0, bit=0):
    b = bytearray(Path(path).read_bytes())
    b[offset] ^= 1 << bit
    Path(path).write_bytes(bytes(b))


def find_float_word(path, pred):
    """Byte offset of the first 32 bit little endian word of the file that pred(bits) accepts, else None."""
    data = Path(path).read_bytes()
    words = array.array("I")
    words.frombytes(data[: len(data) // 4 * 4])
    if sys.byteorder == "big":
        words.byteswap()
    for i, w in enumerate(words):
        if pred(w):
            return i * 4
    return None


def write_word(path, off, bits):
    b = bytearray(Path(path).read_bytes())
    b[off:off + 4] = int(bits).to_bytes(4, "little")
    Path(path).write_bytes(bytes(b))


def synth_plugin_lines(rewrite_lines, layer_index, style="smart", fps=None, with_params=True):
    """A plugin side record in the harness's format for what the rewrite's record says was read.
    layer_index: rewrite layer id -> ae param index. style smart: prerender_cb.checkout_layer lines;
    classic: inter.checkout_param lines (plus value parameter checkouts, which the check must ignore)."""
    out = []
    seq = [0]

    def add(**kw):
        seq[0] += 1
        out.append(dict({"seq": seq[0]}, **kw))
    add(ev="cmd", cmd="SEQUENCE_SETUP", time=0)
    add(ev="render_path", smart=(style == "smart"))
    cid = 0
    scale = step = None
    for d in rewrite_lines:
        if d["ev"] == "run":
            scale, step = d["time_scale"], d["time_step"]
            fps = Fraction(d["fps"]) if fps is None else fps
        elif d["ev"] == "frame":
            add(ev="frame", out_frame=d["out_frame"], time=d["time"], time_step=d["time_step"], time_scale=d["time_scale"])
            if with_params:
                add(ev="cb", fn="inter.checkout_param", args=["0x1", 9, d["time"], d["time_step"], d["time_scale"], "0x2"], ret=0,
                    param_index=9, time=d["time"], time_step=d["time_step"], time_scale=d["time_scale"], footage_frame=d["out_frame"],
                    out_frame=d["out_frame"])
        elif d["ev"] == "read":
            idx = d["index"]
            ff = idx if idx is not None else math.floor(Fraction(d["time"], scale) * fps)
            p = layer_index[d["layer"]]
            inside = idx is not None
            cid += 1
            if style == "smart":
                add(ev="cb", fn="prerender_cb.checkout_layer", args=["0x1", p, cid, "0x2", d["time"], step, scale, "0x3"], ret=0,
                    param_index=p, checkout_id=cid, time=d["time"], time_step=step, time_scale=scale, footage_frame=ff, inside=inside,
                    out_frame=d["out_frame"])
                add(ev="cb", fn="smart_cb.checkout_layer_pixels", args=["0x1", cid, "0x4"], ret=0, checkout_id=cid, footage_frame=ff)
            else:
                add(ev="cb", fn="inter.checkout_param", args=["0x1", p, d["time"], step, scale, "0x2"], ret=0, param_index=p,
                    time=d["time"], time_step=step, time_scale=scale, footage_frame=ff, out_frame=d["out_frame"])
    add(ev="cmd_done", cmd="SEQUENCE_SETDOWN", err=0)
    return out


def write_lines(path, lines):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as fh:
        for d in lines:
            fh.write(json.dumps(d) + "\n")


def _layer_read_lines(pl):
    """Indexes of the lines of a synthetic plugin record that are layer reads (not value parameters)."""
    return [i for i, d in enumerate(pl) if d.get("ev") == "cb" and d.get("fn") in ("prerender_cb.checkout_layer", "inter.checkout_param")
            and d.get("param_index") != 9]


def perturb_plugin_lines(pl, kind, layer_index):
    """One change to a synthetic plugin record, in the last output frame that has reads. Returns
    (new lines, the output frame that changed) or None when the record cannot take this change."""
    idxs = _layer_read_lines(pl)
    if not idxs:
        return None
    last = idxs[-1]
    d_ = pl[last]
    fr = d_["out_frame"]
    if kind == "missing":
        drop = {last}
        if d_["fn"] == "prerender_cb.checkout_layer":
            drop |= {j for j, x in enumerate(pl) if x.get("fn") == "smart_cb.checkout_layer_pixels" and x.get("checkout_id") == d_["checkout_id"]}
        return [x for j, x in enumerate(pl) if j not in drop], fr
    if kind == "extra":
        dup = dict(d_, time=d_["time"] + d_["time_step"] * 7, footage_frame=(d_.get("footage_frame") or 0) + 7)
        return pl[:last + 1] + [dup] + pl[last + 1:], fr
    if kind == "time":
        return pl[:last] + [dict(d_, time=d_["time"] + 37)] + pl[last + 1:], fr
    if kind == "layer":
        inv = {v: k for k, v in layer_index.items()}
        other = "fg3_matte" if inv[d_["param_index"]] != "fg3_matte" else "source"
        return pl[:last] + [dict(d_, param_index=layer_index[other])] + pl[last + 1:], fr
    if kind == "order":
        in_frame = [j for j in idxs if pl[j]["out_frame"] == fr]
        if len(in_frame) < 2 or (pl[in_frame[0]]["param_index"], pl[in_frame[0]]["time"]) == (pl[in_frame[1]]["param_index"], pl[in_frame[1]]["time"]):
            return None
        pl2 = list(pl)
        pl2[in_frame[0]], pl2[in_frame[1]] = pl2[in_frame[1]], pl2[in_frame[0]]
        return pl2, fr
    raise ValueError(kind)


def cmd_selftest(a):
    ts = Path(a.testset)
    cases, meta = load_testset(ts)
    by_id = {c["id"]: c for c in cases}
    sdir = Path(a.out_dir) / "selftest"
    if sdir.exists():
        shutil.rmtree(sdir)
    sdir.mkdir(parents=True)
    vfx = a.vfx or default_vfx()
    bin_dir = Path(a.bin_dir)
    refs_base = Path(a.base_refs)
    res = Path(a.results)
    chk = Check()
    base_rows = read_tsv(res / "base.grid.tsv")
    muts = jload(a.mutants)
    mut_rows = {m["id"]: read_tsv(res / "mutants" / (m["id"] + ".grid.tsv")) for m in muts}
    key = lambda r: (r[1], r[2], r[3])
    changed = {mid: sorted(c for c in base_rows if key(rows[c]) != key(base_rows[c])) for mid, rows in mut_rows.items()}
    refused_base = sorted(c for c, r in base_rows.items() if r[1] != "0")
    in_base = [c["id"] for c in cases if c["id"] not in refused_base]
    pa = jload(res / "mutants.json") if (res / "mutants.json").is_file() else None
    phase_a_count = sum(1 for m in pa["mutants"] if m.get("changed")) if pa else sum(1 for v in changed.values() if v)
    n_cand = len(load_candidates(ROOT, ts))
    out = {"module": MODULE, "measured_at": a.date or today(), "scenarios": {}}
    all_ids = "m00," + ",".join(m["id"] for m in muts)

    def ns(**kw):
        d = dict(testset=str(ts), out_dir=None, date=a.date, vfx=str(vfx), fx_parity=a.fx_parity, max_ulp=0, jobs=1, no_scan=False,
                 mutants=a.mutants, bin_dir=str(bin_dir), results=str(res), base_vfx=None, only=None, rerender=False, refs=None,
                 blocker=[], allow_stale=False, out=None, top=8, probe_dir=None, record_vfx=None)
        d.update(kw)
        return argparse.Namespace(**d)

    # a) perfect oracle: the rewrite's own base renders
    print("a) perfect oracle: references are the rewrite's own base renders")
    d = sdir / "perfect"
    cmd_grade(ns(refs=str(refs_base), out_dir=str(d)))
    g = jload(d / "grade.json")
    chk.ok("perfect: no case differs", g["cases"]["differing"] == 0, "differing %d, compared %d" % (g["cases"]["differing"], g["cases"]["compared"]))
    chk.ok("perfect: every base-rendered case compared", g["cases"]["compared"] == len(in_base), "%d of %d" % (g["cases"]["compared"], len(in_base)))
    chk.ok("perfect: the cases the rewrite refuses are out of contract, not passes", g["cases"]["out_of_contract_cases"] == len(refused_base),
           "%d" % g["cases"]["out_of_contract_cases"])
    chk.ok("perfect: fx-parity and the value scan agree everywhere", g["cases"]["crosscheck_disagreements"] == 0)
    chk.ok("perfect: no differing values, max ULP 0", g["values"]["differing_values"] == 0 and g["values"]["max_ulp"] == 0,
           "%d floats, %d bytes compared" % (g["values"]["floats_compared"], g["values"]["bytes_compared"]))
    cmd_kills(ns(refs=str(refs_base), out_dir=str(d), only=all_ids))
    k = jload(d / "kills.json")
    chk.ok("perfect: mutants killed = phase A count of mutants that change an output", k["killed"] == phase_a_count == len(muts),
           "killed %d of %d, phase A %d of %d" % (k["killed"], k["total"], phase_a_count, len(muts)))
    chk.ok("perfect: the unmutated control m00 is not killed", bool(k["control"]) and k["control"]["status"] == "survivor")
    chk.ok("perfect: the killers of each mutant are the cases it changes in phase A",
           all(set(m["killers"]) == set(changed[m["id"]]) - set(refused_base) for m in k["mutants"]))
    cmd_summary(ns(out_dir=str(d)))
    s = jload(d / "summary.json")
    chk.ok("perfect: summary counts come from the files",
           s["cases"]["in_contract"] == len(in_base) and s["cases"]["compared"] == len(in_base) and s["cases"]["identical"] == len(in_base)
           and s["cases"]["differing"] == 0 and s["cases"]["out_of_contract"] == len(refused_base) + n_cand
           and s["mutants"] == {"total": len(muts), "killed": len(muts), "survivors": []}, json.dumps({x: s[x] for x in ("cases", "mutants")}))
    chk.ok("perfect: status stays in_progress (no call records compared)", s["status"] == "in_progress")
    d2 = sdir / "perfect_rerender"
    cmd_kills(ns(refs=str(refs_base), out_dir=str(d2), rerender=True, only="m00,m16,m17,m21"))
    k2 = jload(d2 / "kills.json")
    st = {m["id"]: m["status"] for m in k2["mutants"]}
    chk.ok("rerender: m16, m17, m21 each killed by one case, the control survives",
           st == {"m16": "killed", "m17": "killed", "m21": "killed"} and k2["control"]["status"] == "survivor"
           and all(len(m["killers"]) == 1 for m in k2["mutants"]), json.dumps(st))
    out["scenarios"]["perfect"] = {"differing": g["cases"]["differing"], "killed": k["killed"], "total": k["total"], "phase_a": phase_a_count}

    # b) perturbed oracle
    print("b) perturbed oracle: one flipped bit, one -0, one NaN payload")
    pr = sdir / "perturbed_refs"
    shutil.copytree(refs_base, pr)
    ids32 = [cid for cid in in_base if by_id[cid]["depth"] == 32]
    sole16 = changed["m16"][0] if len(changed["m16"]) == 1 else None
    case_flip = sole16 if sole16 and by_id[sole16]["depth"] == 8 else next(cid for cid in in_base if by_id[cid]["depth"] == 8)
    negz = nanp = None
    for cid in ids32:
        off = find_float_word(pr / by_id[cid]["reference"], lambda w: w == 0x80000000)
        if off is not None:
            negz = (cid, off, "-0 to 0")
            break
    for cid in ids32:
        if negz and cid == negz[0]:
            continue
        off = find_float_word(pr / by_id[cid]["reference"], f32_is_nan)
        if off is not None:
            nanp = (cid, off)
            break
    if negz is None:
        for cid in ids32:
            if nanp and cid == nanp[0]:
                continue
            off = find_float_word(pr / by_id[cid]["reference"], lambda w: w == 0)
            if off is not None:
                negz = (cid, off, "0 to -0 (no output holds a -0)")
                break
    flip_bit(pr / by_id[case_flip]["reference"], 5, 3)
    expected = {case_flip: ("bit flip", None, 0)}
    if negz:
        f = pr / by_id[negz[0]]["reference"]
        w = int.from_bytes(f.read_bytes()[negz[1]:negz[1] + 4], "little")
        write_word(f, negz[1], 0 if w == 0x80000000 else 0x80000000)
        expected[negz[0]] = (negz[2], 0, 0)
    if nanp:
        f = pr / by_id[nanp[0]]["reference"]
        w = int.from_bytes(f.read_bytes()[nanp[1]:nanp[1] + 4], "little")
        write_word(f, nanp[1], w ^ 1)
        expected[nanp[0]] = ("NaN payload", 0, 1)
    d = sdir / "perturbed"
    cmd_grade(ns(refs=str(pr), out_dir=str(d)))
    g = jload(d / "grade.json")
    failing = {r["id"]: r for r in g["cases_detail"] if r["status"] == "FAIL"}
    chk.ok("perturbed: exactly the perturbed cases differ", set(failing) == set(expected), "%d differ, expected %d" % (len(failing), len(expected)))
    for cid, (what, ulp, nanm) in expected.items():
        r = failing.get(cid)
        good = bool(r) and r["differing_values"] == 1 and r["max_ulp"] == ulp and r["nan_mismatch"] == nanm
        chk.ok("perturbed: %s in %s" % (what, cid), good,
               "values %s, max_ulp %s, nan_mismatch %s" % (r and r["differing_values"], r and r["max_ulp"], r and r["nan_mismatch"]))
    chk.ok("perturbed: a -0 case was found to perturb", negz is not None, "" if negz else "no 32 bpc output holds a zero")
    chk.ok("perturbed: a NaN case was found to perturb", nanp is not None, "" if nanp else "no 32 bpc output holds a NaN")
    cmd_kills(ns(refs=str(pr), out_dir=str(d), only=all_ids))
    k = jload(d / "kills.json")
    exp_killed = sorted(mid for mid, ch in changed.items() if set(ch) - set(expected) - set(refused_base))
    got_killed = sorted(m["id"] for m in k["mutants"] if m["status"] == "killed")
    chk.ok("perturbed: killed = mutants that change a case where the base equals the reference", exp_killed == got_killed,
           "killed %d of %d; survivors %s" % (len(got_killed), k["total"], k["survivors"]))
    if sole16 and sole16 in expected:
        chk.ok("perturbed: m16 (its only case perturbed) becomes a survivor", "m16" in k["survivors"])
    cmd_summary(ns(out_dir=str(d)))
    s = jload(d / "summary.json")
    chk.ok("perturbed: summary shows %d differing cases and values" % len(expected),
           s["cases"]["differing"] == len(expected) and s["values"]["differing_values"] == len(expected),
           "differing_values %d, max_ulp %s" % (s["values"]["differing_values"], s["values"]["max_ulp"]))
    out["scenarios"]["perturbed"] = {"differing": sorted(failing), "killed": len(got_killed), "survivors": k["survivors"]}
    print("   triage on the perturbed grade")
    cmd_triage(ns(out_dir=str(d), probe_dir=str(sdir / "probe")))
    pdoc = jload(sdir / "probe" / "cases.json")
    chk.ok("triage: a ladder was written", len(pdoc["cases"]) >= 4, "%d probe cases" % len(pdoc["cases"]))
    probe_refs = sdir / "probe" / "refs"
    probe_refs.mkdir()
    rendered = 0
    for c in pdoc["cases"]:
        code, o, _ = render_case(vfx, sdir / "probe", c)
        if code == 0:
            (probe_refs / c["reference"]).write_bytes(o)
            rendered += 1
    chk.ok("triage: every ladder case renders with the rewrite", rendered == len(pdoc["cases"]), "%d of %d" % (rendered, len(pdoc["cases"])))
    cmd_grade(ns(testset=str(sdir / "probe"), refs=str(probe_refs), out_dir=str(sdir / "probe" / "results")))
    pg = jload(sdir / "probe" / "results" / "grade.json")
    chk.ok("triage: the ladder grades through the same tool", pg["cases"]["compared"] == len(pdoc["cases"]) and pg["cases"]["differing"] == 0,
           "%d compared" % pg["cases"]["compared"])

    # c) missing references
    print("c) missing references")
    mr = sdir / "missing_refs"
    shutil.copytree(refs_base, mr)
    drop = in_base[::90][:5]
    for cid in drop:
        (mr / by_id[cid]["reference"]).unlink()
    d = sdir / "missing"
    cmd_grade(ns(refs=str(mr), out_dir=str(d)))
    g = jload(d / "grade.json")
    nr = sorted(r["id"] for r in g["cases_detail"] if r["status"] == "no reference")
    chk.ok("missing: exactly the dropped cases are `no reference`", nr == sorted(drop), "%d" % len(nr))
    chk.ok("missing: they are neither passes nor failures",
           g["cases"]["compared"] == len(in_base) - len(drop) and g["cases"]["identical"] == g["cases"]["compared"] and g["cases"]["differing"] == 0)
    cmd_kills(ns(refs=str(mr), out_dir=str(d)))
    k = jload(d / "kills.json")
    chk.ok("missing: the dropped cases are not discriminating", k["in_contract_without_reference"] == len(drop))
    cmd_summary(ns(out_dir=str(d)))
    s = jload(d / "summary.json")
    chk.ok("missing: summary compared is in_contract minus the dropped", s["cases"]["compared"] == s["cases"]["in_contract"] - len(drop), json.dumps(s["cases"]))
    out["scenarios"]["missing"] = {"no_reference": nr}

    # d) out of contract from a synthetic harness status
    print("d) out of contract from a synthetic status file")
    status = {"testset": str(ts), "cases": [{"id": c["id"], "status": "rendered"} for c in cases]}
    mv = {in_base[3]: "unsupported", in_base[40]: "unsupported", in_base[200]: "timeout", in_base[300]: "error"}
    for e in status["cases"]:
        if e["id"] in mv:
            e["status"] = mv[e["id"]]
            e["error"] = "synthetic"
            if mv[e["id"]] == "error":
                e["faults"] = [{"kind": "read"}]
        if refused_base and e["id"] == refused_base[-1]:
            e["status"] = "unsupported"
    jsave(sdir / "status.json", status)
    d = sdir / "ooc"
    cmd_ooc(ns(status=str(sdir / "status.json"), out_dir=str(d), base=str(res / "base.grid.tsv")))
    o = jload(d / "ooc.json")
    gone = {m["id"] for m in o["moved_out"]}
    want = set(mv) | ({refused_base[-1]} if refused_base else set())
    chk.ok("ooc: the refused, errored and timed out cases left the contract", gone == want, "%d moved out" % len(gone))
    chk.ok("ooc: a case the rewrite refuses and the original renders is back in contract", o["back_in_contract"] == refused_base[:-1], str(o["back_in_contract"]))
    chk.ok("ooc: in_contract = total - moved out", o["in_contract"] == len(cases) - len(want), str(o["in_contract"]))
    dr = sdir / "ooc_refs"
    shutil.copytree(refs_base, dr)
    for cid in o["back_in_contract"]:
        c = by_id[cid]
        (dr / c["reference"]).write_bytes(bytes(c["width"] * c["height"] * c["frames"] * 4 * SAMPLE_BYTES[c["depth"]]))
    cmd_grade(ns(refs=str(dr), out_dir=str(d)))
    g = jload(d / "grade.json")
    fails = [r["id"] for r in g["cases_detail"] if r["status"] == "FAIL"]
    chk.ok("ooc: moved out cases are `out of contract`, not graded", sum(1 for r in g["cases_detail"] if r["status"] == "out of contract") == len(gone))
    chk.ok("ooc: the case the rewrite refuses and the original renders is a real failure", sorted(fails) == sorted(o["back_in_contract"]), str(fails))
    cmd_summary(ns(out_dir=str(d)))
    s = jload(d / "summary.json")
    chk.ok("ooc: summary in_contract and out_of_contract follow ooc.json",
           s["cases"]["in_contract"] == o["in_contract"] and s["cases"]["out_of_contract"] == len(gone) + n_cand, json.dumps(s["cases"]))
    out["scenarios"]["ooc"] = {"moved_out": sorted(gone), "back_in_contract": o["back_in_contract"]}

    # e) call records
    print("e) call records: synthetic plugin records against the rewrite's")
    picks = []
    for grp in ("defaults", "layers", "ftime", "keyed", "fg", "speed", "inrate", "framenum", "vecfield", "warp"):
        for c in cases:
            if group_of(meta, c["id"]) == grp and c["id"] in in_base and c["id"] not in picks and c.get("layers"):
                picks.append(c["id"])
                break
    for needle in ("short-vectors.16x12.d8", "ftime-fps10-speed110.16x12.d8", "vectors-1-frame.16x12.d8", "prev-short-next-full", "short-next-only"):
        for c in cases:
            if needle in c["id"] and c["id"] not in picks:
                picks.append(c["id"])
                break
    rw = sdir / "calls" / "rewrite"
    rec_vfx = a.record_vfx or default_record_vfx()
    rc = cmd_rewrite_records(ns(out=str(rw), only=",".join(picks), vfx=rec_vfx))
    idx = jload(rw / "_index.json")
    chk.ok("rewrite-records: the record build writes a record for every case", rc == 0 and not idx["no_record"], "%s: %d cases" % (rec_vfx, idx["cases"]))
    chk.ok("rewrite-records: the record build renders like the base rewrite", not idx["output_differs_from_base"], str(idx["output_differs_from_base"][:2]))
    layer_ids = ["input", "vectors_prev", "vectors_next", "source", "fg1_matte", "fg2_matte", "fg3_matte"]
    layer_index = {l: i for i, l in enumerate(layer_ids)}
    pmap = sdir / "calls" / "param_map.tsv"
    with open(pmap, "w", encoding="utf8") as fh:
        fh.write("plugin\teffect\tae_index\tae_name\tvfx_kind\tvfx_id\tmatch\n")
        for l in layer_ids[1:]:
            fh.write("%s\tvectors-in\t%d\t%s\tlayer\t%s\texact\n" % (PLUGIN_STEM, layer_index[l], l, l))
    pl_dir = sdir / "calls" / "plugin"
    pl_dir.mkdir()
    synth = {}
    for i, cid in enumerate(picks):
        rl = read_lines(rw / (cid + ".jsonl"))
        pl = synth_plugin_lines(rl, layer_index, style="classic" if i % 2 else "smart")
        synth[cid] = (rl, pl)
        write_lines(pl_dir / (cid + ".jsonl"), pl)
    rich = sorted(picks, key=lambda c: -sum(1 for x in synth[c][0] if x["ev"] == "read"))
    pert = {}
    pool = [c for c in rich if sum(1 for x in synth[c][0] if x["ev"] == "read") >= 2]
    for kind in ("missing", "extra", "time", "layer", "order"):
        for cid in list(pool):
            r_ = perturb_plugin_lines(synth[cid][1], kind, layer_index)
            if r_ is not None:
                pool.remove(cid)
                write_lines(pl_dir / (cid + ".jsonl"), r_[0])
                pert[kind] = (cid, r_[1])
                break
    d = sdir / "calls"
    cmd_calls(ns(plugin_records=str(pl_dir), rewrite_records=str(rw), param_map=str(pmap), exclude_layers=[], ordered=False,
                 ignore_outside=False, out_dir=str(d)))
    cj = jload(d / "calls.json")
    per = {r["id"]: r for r in cj["per_case"]}
    touched = {v[0] for v in pert.values()}
    chk.ok("calls: unperturbed cases match (smart and classic records, layers resolved by param index)",
           all(per[c]["status"] == "match" for c in picks if c not in touched), "%d compared, %d matching" % (cj["compared"], cj["matching"]))
    for kind in ("missing", "extra", "time", "layer", "order"):
        if kind not in pert:
            chk.ok("calls: %s perturbation" % kind, False, "no record could take it")
            continue
        cid, fr = pert[kind]
        r = per[cid]
        fm = r["first_mismatch"] or {}
        if kind == "order":
            chk.ok("calls: the same reads in another order match as sets", r["status"] == "match")
            continue
        good = r["status"] == "mismatch" and fm.get("out_frame") == fr
        if kind == "missing":
            good = good and "missing_in_plugin" in fm and "extra_in_plugin" not in fm
        elif kind == "extra":
            good = good and "extra_in_plugin" in fm and "missing_in_plugin" not in fm
        else:
            good = good and "missing_in_plugin" in fm and "extra_in_plugin" in fm and bool(fm.get("hints"))
        chk.ok("calls: %s read detected at output frame %s" % (kind, fr), good, json.dumps(fm)[:230])
    if "order" in pert:
        cmd_calls(ns(plugin_records=str(pl_dir), rewrite_records=str(rw), param_map=str(pmap), exclude_layers=[], ordered=True,
                     ignore_outside=False, out_dir=str(sdir / "calls_ordered")))
        cjo = {r["id"]: r for r in jload(sdir / "calls_ordered" / "calls.json")["per_case"]}
        chk.ok("calls: that swap is a mismatch when order matters", cjo[pert["order"][0]]["status"] == "mismatch")
    past_end = [c for c in picks if any(x["ev"] == "read" and x["index"] is None for x in synth[c][0])]
    chk.ok("calls: cases that read past the end of a clip (index null) are compared by time",
           bool(past_end) and all(per[c]["status"] in ("match", "mismatch") for c in past_end), str(past_end[:3]))
    (pl_dir / (picks[0] + ".jsonl")).unlink()
    cmd_calls(ns(plugin_records=str(pl_dir), rewrite_records=str(rw), param_map=str(pmap), exclude_layers=[], ordered=False,
                 ignore_outside=False, out_dir=str(sdir / "calls_missing")))
    cm = {r["id"]: r for r in jload(sdir / "calls_missing" / "calls.json")["per_case"]}
    chk.ok("calls: a case without a plugin record is not compared, not a pass", cm[picks[0]]["status"] == "no_plugin_record")
    out["scenarios"]["calls"] = {"cases": picks, "perturbed": {k5: list(v5) for k5, v5 in pert.items()}}
    out["checks"] = chk.items
    out["passed"] = sum(1 for c in chk.items if c["ok"])
    out["failed"] = sum(1 for c in chk.items if not c["ok"])
    jsave(sdir / "selftest.json", out)
    print("\nselftest: %d checks, %d ok, %d FAIL" % (len(chk.items), out["passed"], out["failed"]))
    return 0 if not out["failed"] else 1


# ---------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, out_default=True):
        p.add_argument("--testset", default=str(ROOT / "testsets" / "grid"), help="the fx-parity test set (default testsets/grid)")
        p.add_argument("--out-dir", default=str(ROOT / "results"), help="where result files go (default results/)")
        p.add_argument("--date", help="measured_at (default today)")

    g = sub.add_parser("grade", help="fx-parity run --max-ulp 0 --json against the references; grade.json and grade.tsv")
    common(g)
    g.add_argument("--refs", required=True)
    g.add_argument("--vfx")
    g.add_argument("--fx-parity")
    g.add_argument("--max-ulp", type=int, default=0)
    g.add_argument("--jobs", type=int, default=1)
    g.add_argument("--no-scan", action="store_true", help="skip the own value level check (differing values stay unknown)")
    g.set_defaults(fn=cmd_grade)

    o = sub.add_parser("ooc", help="re-judge the contract from the harness's per-case status; ooc.json")
    common(o)
    o.add_argument("--status", required=True, help="run_harness.py --summary file")
    o.add_argument("--base", default=str(ROOT / "results" / "base.grid.tsv"))
    o.set_defaults(fn=cmd_ooc)

    r = sub.add_parser("rewrite-records", help="run the rewrite with VFX_CALL_RECORD over the test set into a folder")
    common(r)
    r.add_argument("--out", required=True)
    r.add_argument("--vfx", help="a build that writes VFX_CALL_RECORD (default: the harness copy's, bin/vfx-base has none)")
    r.add_argument("--only", help="comma separated case ids")
    r.set_defaults(fn=cmd_rewrite_records)

    c = sub.add_parser("calls", help="compare the plugin's call records with the rewrite's; calls.json")
    common(c)
    c.add_argument("--plugin-records", required=True)
    c.add_argument("--rewrite-records", required=True)
    c.add_argument("--param-map", default=str(ROOT.parent / "harness" / "param_map.tsv"), help="maps a plugin param index to a layer id")
    c.add_argument("--rule", choices=("reads", "registrations"), default="reads",
                   help="reads (default): pixel checkouts against reads, see callrec.py; registrations: the old rule, every "
                        "checkout of the plugin including the dummy registrations of its pre-render call (finds 0 of 443 for structural reasons)")
    c.add_argument("--ordered", action="store_true", help="the order of the reads inside an output frame matters")
    c.add_argument("--exclude-layers", action="append", default=[], help="leave a layer out of the comparison (repeatable)")
    c.add_argument("--ignore-outside", action="store_true", help="leave out reads that fall outside the clip, on both sides")
    c.set_defaults(fn=cmd_calls)

    k = sub.add_parser("kills", help="mutation kills against the references; kills.json")
    common(k)
    k.add_argument("--refs", required=True)
    k.add_argument("--mutants", default=str(ROOT / "mutants" / "mutants.json" if (ROOT / "mutants" / "mutants.json").is_file() else ROOT / "tools" / "mutants.json"))
    k.add_argument("--bin-dir", default=str(ROOT / "bin"))
    k.add_argument("--results", default=str(ROOT / "results"), help="phase A results (base.grid.tsv, mutants/*.grid.tsv)")
    k.add_argument("--rerender", action="store_true", help="re-render with bin/vfx-mNN instead of using the saved phase A shas")
    k.add_argument("--base-vfx", help="the base build for --rerender (default bin/vfx-m00)")
    k.add_argument("--only", help="comma separated mutant ids (m00 is the control)")
    k.add_argument("--jobs", type=int, default=1)
    k.set_defaults(fn=cmd_kills)

    s = sub.add_parser("summary", help="results/summary.json from grade.json, ooc.json, kills.json, calls.json")
    common(s)
    s.add_argument("--out", help="write here instead of <out-dir>/summary.json")
    s.add_argument("--blocker", action="append", default=[], help="a blocker outside the module (repeatable); the status becomes blocked")
    s.add_argument("--allow-stale", action="store_true")
    s.set_defaults(fn=cmd_summary)

    t = sub.add_parser("triage", help="group the differences; write a ladder of probe cases for the smallest failing case")
    common(t)
    t.add_argument("--probe-dir", help="default probe/ in the module folder")
    t.add_argument("--top", type=int, default=25)
    t.set_defaults(fn=cmd_triage)

    st = sub.add_parser("selftest", help="stand-in oracle checks (needs vfx, fx-parity, refs/grid and the saved mutants); writes results/selftest/")
    common(st)
    st.add_argument("--vfx")
    st.add_argument("--record-vfx", help="a build that writes VFX_CALL_RECORD (default: the harness copy's)")
    st.add_argument("--fx-parity")
    st.add_argument("--base-refs", default=str(ROOT / "refs" / "grid"))
    st.add_argument("--bin-dir", default=str(ROOT / "bin"))
    st.add_argument("--results", default=str(ROOT / "results"))
    st.add_argument("--mutants", default=str(ROOT / "mutants" / "mutants.json" if (ROOT / "mutants" / "mutants.json").is_file() else ROOT / "tools" / "mutants.json"))
    st.set_defaults(fn=cmd_selftest)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
