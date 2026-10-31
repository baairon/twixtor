#!/usr/bin/env python3
"""Phase B tooling of the fgbg-core module: grade, out of contract, call records, mutation kills,
summary, triage and a stand-in selftest. Python 3 standard library only. No plugin is run here.

  grade     --refs DIR            fx-parity run --max-ulp 0 --json plus a byte audit -> results/grade.json, grade.tsv
  ooc       --status FILE         harness per-case status -> results/ooc.json (cases moved out of contract)
  ooc-set   --out DIR             the out-of-contract candidates as a test set the harness can render
  calls     --plugin-records DIR --rewrite-records DIR      call record check -> results/calls.json
  rewrite-records --out DIR       render every case with VFX_CALL_RECORD (the rewrite side of the check)
  kills     --refs DIR            mutation kills -> results/kills.json
  summary                         results/summary.json from the result files (needs grade.json)
  triage                          group the differing cases, write a ladder of probe cases into probe/
  selftest                        the stand-in checks (perfect, perturbed, missing, out of contract, records)

Every subcommand takes --results DIR (default results/) so that checks can run in a scratch folder.
Run it under `taskset -c N` for one core; child processes inherit the affinity.
"""
import argparse
import array
import collections
import concurrent.futures as cf
import datetime
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BPS = {8: 1, 16: 2, 32: 4}
CHANNELS = {8: "RGBA", 16: "RGBA", 32: "GBRA"}      # interleaved rgba / rgba64le; planar gbrapf32le
BAR_ULP = 0
PLUGIN = "TwixtorPro_64"
EFFECT = "retime-fgbg"
MIN_CASES = 150
PLUGIN_SHA = "58bbd99248a531acb572e5ddb13570d8102a087c32123191e2f52f477e2dcf2d"      # TwixtorPro_64.aex, as in make_summary.py
PHASE_A_COMMANDS = [
    "python3 gen_grid.py --out testset --vfx bin/vfx_base",
    "python3 render_grid.py --vfx bin/vfx_base --testset testset --out results/base.tsv --jobs 2",
    "python3 run_mutants.py --tree src_mut --testset testset --base results/base.tsv --out mutants",
    "python3 make_summary.py",
]


# ---------------------------------------------------------------------------------------------
# small shared helpers

def jdump(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")


def jload(path):
    with open(path) as fh:
        return json.load(fh)


def today():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def paths(a):
    """Resolve the common options against the module folder."""
    d = lambda v: v if os.path.isabs(v) else os.path.join(HERE, v)
    a.testset = d(getattr(a, "testset", "testset"))
    a.results = d(getattr(a, "results", "results"))
    return a


def load_cases(testset):
    return jload(os.path.join(testset, "cases.json"))["cases"]


def load_grid(testset):
    """id -> (block, set of tags) from grid.tsv; a missing file gives empty tags."""
    out = {}
    p = os.path.join(testset, "grid.tsv")
    if os.path.exists(p):
        for line in open(p).read().splitlines()[1:]:
            f = line.split("\t")
            if f and f[0]:
                out[f[0]] = (f[1] if len(f) > 1 else "", {t for t in (f[2] if len(f) > 2 else "").split(",") if t})
    return out


def load_tsv(path):
    """A render_grid.py result file: id -> {exit, bytes, sha, error}."""
    out = {}
    for line in open(path).read().splitlines()[1:]:
        f = line.split("\t")
        if len(f) >= 4:
            out[f[0]] = {"exit": int(f[1]), "bytes": int(f[2]), "sha": f[3], "error": f[5] if len(f) > 5 else ""}
    return out


def frame_len(case):
    return case["width"] * case["height"] * 4 * BPS[case["depth"]]


def vfx_args(vfx, case, testset):
    a = [vfx, "apply", case["effect"], "--width", str(case["width"]), "--height", str(case["height"]),
         "--depth", str(case["depth"]), "--fps", case.get("fps") or "24", "--frames", str(case["frames"])]
    for p in case.get("params") or []:
        a += ["--param", p]
    for lid, f in (case.get("layers") or {}).items():
        a += ["--layer", "{}={}".format(lid, os.path.join(testset, f))]
    a += ["--in", os.path.join(testset, case["input"])]
    return a


def render(vfx, case, testset, record=None):
    """(exit code, stdout bytes, last stderr line). The record file is VFX_CALL_RECORD when given."""
    env = dict(os.environ)
    env.pop("VFX_CALL_RECORD", None)
    if record:
        env["VFX_CALL_RECORD"] = record
    p = subprocess.run(vfx_args(vfx, case, testset), capture_output=True, env=env)
    err = (p.stderr.decode("utf-8", "replace").strip().splitlines() or [""])[-1][:200]
    return p.returncode, p.stdout, err


# ---------------------------------------------------------------------------------------------
# value level comparison (the byte audit)

def ordered(bits):
    """Float bits as integers that sort like the floats; -0 and 0 share a value (fx-parity's rule)."""
    return -(bits & 0x7FFFFFFF) if bits & 0x80000000 else bits


def is_nan_bits(bits):
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def ulp_between(a_bits, b_bits):
    """ULP distance of two float bit patterns, or None when either is a NaN."""
    if is_nan_bits(a_bits) or is_nan_bits(b_bits):
        return None
    return abs(ordered(a_bits) - ordered(b_bits))


def sample_where(case, i):
    """The place of sample number i of a clip: frame, x, y and channel letter."""
    w, h, d = case["width"], case["height"], case["depth"]
    per = w * h * 4
    f, within = divmod(i, per)
    if d == 32:
        plane, px = divmod(within, w * h)
        return {"frame": f, "x": px % w, "y": px // w, "channel": CHANNELS[32][plane]}
    px, ch = divmod(within, 4)
    return {"frame": f, "x": px % w, "y": px // w, "channel": CHANNELS[d][ch]}


def samples(depth, data):
    n = len(data) // BPS[depth]
    if depth == 8:
        return data
    return struct.unpack("<{}{}".format(n, "H" if depth == 16 else "I"), data[: n * BPS[depth]])


def compare_clip_bytes(case, ref, out, listed=5):
    """Bit for bit comparison of a reference clip with an output clip.

    differing_values counts samples (a byte at 8 bpc, a 16 bit value at 16 bpc, a float at 32 bpc) whose
    bits differ. max_ulp is the largest ULP distance over finite float pairs at 32 bpc (0 when none differ,
    None at 8 and 16 bpc); nan_mismatch counts pairs with a NaN on one side whose bits differ.
    """
    d = case["depth"]
    res = {"bytes_compared": len(ref) if d != 32 else 0, "floats_compared": len(ref) // 4 if d == 32 else 0,
           "values_compared": len(ref) // BPS[d], "differing_values": 0, "max_ulp": 0 if d == 32 else None,
           "nan_mismatch": 0, "differing_frames": 0, "first_frame": None, "size_mismatch": len(ref) != len(out),
           "first": []}
    n = min(len(ref), len(out)) // BPS[d]
    if len(ref) != len(out):
        res["differing_values"] += abs(len(ref) - len(out)) // BPS[d]
    if ref[: n * BPS[d]] == out[: n * BPS[d]] and not res["size_mismatch"]:
        return res
    r, o = samples(d, ref[: n * BPS[d]]), samples(d, out[: n * BPS[d]])
    per = case["width"] * case["height"] * 4
    frames = set()
    for i in range(n):
        if r[i] == o[i]:
            continue
        res["differing_values"] += 1
        frames.add(i // per)
        if d == 32:
            u = ulp_between(r[i], o[i])
            if u is None:
                res["nan_mismatch"] += 1
            elif u > res["max_ulp"]:
                res["max_ulp"] = u
        if len(res["first"]) < listed:
            e = sample_where(case, i)
            if d == 32:
                e.update(expected="0x{:08x}".format(r[i]), actual="0x{:08x}".format(o[i]),
                         expected_value=struct.unpack("<f", struct.pack("<I", r[i]))[0],
                         actual_value=struct.unpack("<f", struct.pack("<I", o[i]))[0])
            else:
                e.update(expected=r[i], actual=o[i])
            res["first"].append(e)
    res["differing_frames"] = len(frames)
    res["first_frame"] = min(frames) if frames else None
    return res


# ---------------------------------------------------------------------------------------------
# grade

def fx_parity_run(a):
    cmd = [a.fx_parity, "run", "--testset", a.testset, "--refs", a.refs, "--max-ulp", str(BAR_ULP), "--json",
           "--vfx", a.vfx, "--jobs", str(a.jobs)]
    if a.effect:
        cmd += ["--effect", a.effect]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode not in (0, 1):
        raise SystemExit("fx-parity exited {}: {}".format(p.returncode, (p.stderr or p.stdout).strip()[:500]))
    try:
        return json.loads(p.stdout), " ".join(cmd), p.returncode
    except ValueError:
        raise SystemExit("fx-parity printed no JSON: " + (p.stderr or p.stdout)[:500])


def build_grade(cases, grid, report, audits, base):
    """Merge the fx-parity report, the byte audit and grid.tsv into per case rows and totals.

    report: fx-parity --json dict. audits: id -> compare_clip_bytes result (or {'render_error': text}).
    base: results/base.tsv rows, used to say whether the audit's render is the one base.tsv recorded.
    """
    fx = {c["id"]: c for c in report.get("cases", [])}
    rows, disagree = [], []
    for c in cases:
        cid = c["id"]
        blk, tags = grid.get(cid, ("", set()))
        f = fx.get(cid, {})
        s_fx = f.get("status", "no reference")
        row = {"id": cid, "status": s_fx if s_fx in ("pass", "FAIL") else "no reference", "block": blk,
               "tags": sorted(tags), "logic_parity": "standin" in tags, "depth": c["depth"], "width": c["width"],
               "height": c["height"], "frames": c["frames"], "params": c.get("params") or []}
        if s_fx in ("pass", "FAIL"):
            au = audits.get(cid)
            fxs = {k: f[k] for k in ("failing_frames", "frame", "clip_differing_pixels", "differing_pixels", "max_ulp",
                                     "error", "mean_abs_diff", "mismatches") if k in f}
            row["fx_parity"] = dict(fxs, status=s_fx)
            if "error" in f:
                row["error"] = f["error"]
            if au and "render_error" not in au:
                row.update({k: au[k] for k in ("bytes_compared", "floats_compared", "values_compared", "differing_values",
                                               "max_ulp", "nan_mismatch", "differing_frames", "first_frame", "size_mismatch")})
                row["first_diff"] = au["first"]
                eq = au["differing_values"] == 0 and not au["size_mismatch"]
                if (s_fx == "pass") != eq:
                    row["audit_disagrees"] = True
                    disagree.append(cid)
                    row["status"] = "FAIL"      # the stricter of the two stands
            else:
                row.update(bytes_compared=0, floats_compared=0, values_compared=0, differing_values=0, max_ulp=None,
                           nan_mismatch=0, differing_frames=0, first_frame=None, size_mismatch=False, first_diff=[])
                row["audit"] = au.get("render_error", "skipped") if au else "skipped"
                if s_fx == "FAIL":
                    row["differing_values"] = None      # unknown: the audit could not count it
                    row["floats_compared"] = row["bytes_compared"] = None
            if au and "sha" in au and cid in base and base[cid]["sha"] != au["sha"]:
                row["base_tsv_stale"] = True
        rows.append(row)
    cmp_rows = [r for r in rows if r["status"] in ("pass", "FAIL")]
    totals = {
        "cases": len(rows), "pass": sum(1 for r in rows if r["status"] == "pass"),
        "fail": sum(1 for r in rows if r["status"] == "FAIL"),
        "no_reference": sum(1 for r in rows if r["status"] == "no reference"),
        "compared": len(cmp_rows), "logic_parity_compared": sum(1 for r in cmp_rows if r["logic_parity"]),
        "bytes_compared": sum(r.get("bytes_compared") or 0 for r in cmp_rows),
        "floats_compared": sum(r.get("floats_compared") or 0 for r in cmp_rows),
        "differing_values": sum(r.get("differing_values") or 0 for r in cmp_rows),
        "nan_mismatch": sum(r.get("nan_mismatch") or 0 for r in cmp_rows),
        "max_ulp": max([r["max_ulp"] for r in cmp_rows if r.get("max_ulp") is not None], default=None),
        "audit_disagreements": disagree,
        "base_tsv_stale": sorted(r["id"] for r in rows if r.get("base_tsv_stale")),
        "not_audited": sorted(r["id"] for r in cmp_rows if "audit" in r),
    }
    return rows, totals


def write_grade_tsv(path, rows):
    cols = ["id", "status", "differing_values", "values_compared", "max_ulp", "nan_mismatch", "differing_frames",
            "first_frame", "block", "tags", "logic_parity", "depth", "size", "frames"]
    with open(path, "w", newline="\n") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            vals = [r["id"], r["status"], r.get("differing_values"), r.get("values_compared"), r.get("max_ulp"),
                    r.get("nan_mismatch"), r.get("differing_frames"), r.get("first_frame"), r["block"],
                    ",".join(r["tags"]), int(r["logic_parity"]), r["depth"], "{}x{}".format(r["width"], r["height"]),
                    r["frames"]]
            fh.write("\t".join("" if v is None else str(v) for v in vals) + "\n")


def cmd_grade(a):
    paths(a)
    cases = load_cases(a.testset)
    if a.effect:
        cases = [c for c in cases if c["effect"] == a.effect]
    report, cmdline, code = fx_parity_run(a)
    fx = {c["id"]: c for c in report.get("cases", [])}
    base_p = os.path.join(HERE, "results", "base.tsv") if not a.base else a.base
    base = load_tsv(base_p) if os.path.exists(base_p) else {}
    audits = {}
    if not a.no_audit:
        todo = [c for c in cases if fx.get(c["id"], {}).get("status") in ("pass", "FAIL")]

        def one(c):
            ref_path = os.path.join(a.refs, c["reference"])
            try:
                rc, out, err = render(a.vfx, c, a.testset)
            except OSError as e:
                return c["id"], {"render_error": str(e)}
            if rc != 0:
                return c["id"], {"render_error": "vfx exit {}: {}".format(rc, err)}
            with open(ref_path, "rb") as fh:
                ref = fh.read()
            r = compare_clip_bytes(c, ref, out)
            r["sha"] = hashlib.sha256(out).hexdigest()
            return c["id"], r

        with cf.ThreadPoolExecutor(a.jobs) as ex:
            audits = dict(ex.map(one, todo))
    rows, totals = build_grade(cases, load_grid(a.testset), report, audits, base)
    doc = {"module": "fgbg-core", "effect": EFFECT, "measured_at": today(), "bar": "bytes exact at 8 and 16 bpc, floats bit for bit at 32 bpc (--max-ulp 0)",
           "command": cmdline, "fx_parity_exit": code, "refs": a.refs, "audit": not a.no_audit,
           "fx_parity_summary": report.get("summary"), "totals": totals, "cases": rows}
    jdump(os.path.join(a.results, "grade.json"), doc)
    write_grade_tsv(os.path.join(a.results, "grade.tsv"), rows)
    t = totals
    print("grade: {} cases, {} pass, {} FAIL, {} no reference; differing values {}, max ULP {}{}".format(
        t["cases"], t["pass"], t["fail"], t["no_reference"], t["differing_values"], t["max_ulp"],
        "; AUDIT DISAGREES on {}".format(len(t["audit_disagreements"])) if t["audit_disagreements"] else ""))
    return 0


# ---------------------------------------------------------------------------------------------
# out of contract

def load_status_file(path):
    """The harness per-case status as id -> dict(status, error, ...). Accepts run_harness.py --summary
    (a dict with a `cases` list), a bare list, a dict id -> entry, JSON lines, or id<TAB>status<TAB>error."""
    text = open(path).read()
    entries = []
    try:
        doc = json.loads(text)
        if isinstance(doc, dict) and isinstance(doc.get("cases"), list):
            entries = doc["cases"]
        elif isinstance(doc, dict) and isinstance(doc.get("cases"), dict):
            entries = [dict(v, id=k) for k, v in doc["cases"].items()]
        elif isinstance(doc, list):
            entries = doc
        elif isinstance(doc, dict):
            entries = [dict(v, id=k) if isinstance(v, dict) else {"id": k, "status": v} for k, v in doc.items()]
    except ValueError:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    pass
            else:
                f = line.split("\t")
                if len(f) >= 2 and f[0] != "id":
                    entries.append({"id": f[0], "status": f[1], "error": f[2] if len(f) > 2 else ""})
    out = {}
    for e in entries:
        if isinstance(e, dict) and e.get("id"):
            if e.get("event") in ("done", None) or "status" in e:
                out[e["id"]] = e
    return out


def classify_plugin(entry):
    """('rendered'|'refused'|'plugin_error'|'harness_failure', text) for one harness status entry."""
    st = str(entry.get("status", "")).lower()
    err = str(entry.get("error") or "")
    if st == "rendered":
        return "rendered", ""
    if st in ("refused", "unsupported"):
        return "refused", err or st
    if st == "error" and ("PluginError" in err or "returned PF_Err" in err):
        return "plugin_error", err
    if st in ("error", "timeout", "not_run", "fault", "died"):
        return "harness_failure", err or st
    return "harness_failure", err or st or "no status"


NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def value_numbers(text):
    """The numbers a --param value holds: a plain number, or the values of `v@frame;v@frame` keys."""
    out = []
    for part in text.split(";"):
        v = part.split("@", 1)[0]
        try:
            out.append(float(v))
        except ValueError:
            m = NUM.fullmatch(v.strip())
            if m:
                out.append(float(m.group(0)))
    return out


def plugin_ranges(param_map, plugin=PLUGIN):
    """vfx_id -> {min, max, choices} from the param map rows the plugin registered (rows with an ae_index)."""
    out = {}
    if not param_map or not os.path.exists(param_map):
        return out
    lines = open(param_map).read().splitlines()
    cols = lines[0].split("\t")
    for line in lines[1:]:
        r = dict(zip(cols, line.split("\t")))
        if r.get("plugin") != plugin or not r.get("ae_index") or not r.get("vfx_id"):
            continue
        def num(k):
            try:
                return float(r.get(k, ""))
            except ValueError:
                return None
        ch = [c for c in (r.get("ae_choices") or "").split("|") if c]
        out[r["vfx_id"]] = {"name": r.get("ae_name", ""), "min": num("ae_min"), "max": num("ae_max"),
                            "choices": len(ch) or None}
    return out


def out_of_range(case, ranges):
    """The first param of a case outside what the plugin registers, as a text, or None."""
    for p in case.get("params") or []:
        k, _, v = p.partition("=")
        r = ranges.get(k)
        if not r:
            continue
        for x in value_numbers(v):
            if r["choices"] and not (1 <= x <= r["choices"]):
                return "{}={} outside the plugin's {} choices".format(k, v, r["choices"])
            if r["min"] is not None and x < r["min"] or r["max"] is not None and x > r["max"]:
                return "{}={} outside the plugin's range {}..{}".format(k, v, r["min"], r["max"])
    return None


def load_judged(paths_):
    """The out of contract judgements of named cases: TSV files with the header `id<TAB>reason<TAB>evidence`, one case
    per line (lines starting with # are comments). Returns id -> {"reason", "evidence", "file"}; a later file wins."""
    out = {}
    for p in paths_ or []:
        lines = [l for l in open(p).read().splitlines() if l.strip() and not l.startswith("#")]
        if not lines or lines[0].split("\t")[0] != "id":
            raise SystemExit("ooc --judged {}: the first line must be the header id<TAB>reason<TAB>evidence".format(p))
        for line in lines[1:]:
            f = line.split("\t")
            if len(f) < 2 or not f[1].strip():
                raise SystemExit("ooc --judged {}: a line without a reason: {}".format(p, line[:80]))
            out[f[0]] = {"reason": f[1], "evidence": f[2] if len(f) > 2 else "", "file": p}
    return out


def build_ooc(cases, candidates, status, base, ranges=None, move_harness_failures=False, judged=None):
    ids = {c["id"] for c in cases}
    judged = judged or {}
    moved, not_judged, rendered, no_status = [], [], 0, 0
    for c in cases:
        cid = c["id"]
        b = base.get(cid)
        rw = {"exit": b["exit"], "output_bytes": b["bytes"], "error": b["error"]} if b else {"exit": None}
        e = status.get(cid)
        reason = None
        if ranges:
            reason = out_of_range(c, ranges)
            if reason:
                moved.append({"id": cid, "reason": "parameter outside the plugin's registered range", "detail": reason,
                              "plugin": {"status": e.get("status") if e else "not run", "error": (e or {}).get("error")},
                              "rewrite": rw})
                continue
        if cid in judged:
            j = judged[cid]
            moved.append({"id": cid, "reason": "judged out of contract: " + j["reason"], "judgement": dict(j),
                          "plugin": {"status": e.get("status") if e else "not run", "error": (e or {}).get("error")},
                          "rewrite": rw})
            continue
        if e is None:
            no_status += 1
            continue
        kind, text = classify_plugin(e)
        if kind == "rendered":
            rendered += 1
        elif kind in ("refused", "plugin_error") or (kind == "harness_failure" and move_harness_failures):
            moved.append({"id": cid, "reason": {"refused": "the original refuses the case", "plugin_error": "the original returned an error",
                                                "harness_failure": "the original did not render (harness status counted by request)"}[kind],
                          "plugin": {"status": e.get("status"), "error": text, "last_api": e.get("last_api")}, "rewrite": rw})
        else:
            not_judged.append({"id": cid, "plugin": {"status": e.get("status"), "error": text},
                               "note": "the emulation failed, not shown to be the original's behaviour: stays in contract with no reference"})
    cand = []
    for c in candidates:
        e = status.get(c["id"])
        item = {"id": c["id"], "reason": c.get("reason"), "params": c.get("params"),
                "rewrite": {"verdict": c.get("vfx_verdict"), "refused_before_rendering": True}}
        if e is None:
            item["plugin"] = {"status": "not run"}
        else:
            kind, text = classify_plugin(e)
            item["plugin"] = {"status": e.get("status"), "error": text or None, "kind": kind}
            if kind == "rendered":
                item["note"] = "the original renders a case the rewrite refuses: review the rewrite's range"
        cand.append(item)
    counts = {"in_contract_before": len(cases), "moved": len(moved), "in_contract_after": len(cases) - len(moved),
              "candidates": len(cand), "out_of_contract_total": len(cand) + len(moved), "not_judged": len(not_judged),
              "rendered_by_the_original": rendered, "without_status": no_status,
              "candidates_rendered_by_the_original": sum(1 for x in cand if x["plugin"].get("kind") == "rendered"),
              "judged": sum(1 for m in moved if "judgement" in m)}
    return {"moved": moved, "candidates": cand, "not_judged": not_judged, "counts": counts,
            "judged_unknown": sorted(k for k in judged if k not in ids)}


def cmd_ooc(a):
    paths(a)
    cases = load_cases(a.testset)
    oc_p = os.path.join(a.testset, "out_of_contract.json")
    candidates = jload(oc_p)["cases"] if os.path.exists(oc_p) else []
    base_p = a.base or os.path.join(HERE, "results", "base.tsv")
    base = load_tsv(base_p) if os.path.exists(base_p) else {}
    files = a.status if isinstance(a.status, list) else [a.status]
    status = {}
    for f in files:
        status.update(load_status_file(f))
    ranges = plugin_ranges(a.param_map) if a.param_map else {}
    jfiles = list(getattr(a, "judged", None) or [])
    judged = load_judged(jfiles)
    doc = build_ooc(cases, candidates, status, base, ranges, a.move_harness_failures, judged)
    doc.update({"format": 1, "measured_at": today(), "status_file": files,
                "command": "phase_b.py ooc --status {}{}".format(" ".join(files), "".join(" --judged " + p for p in jfiles)),
                "param_ranges_checked": bool(ranges), "judged_files": jfiles,
                "rule": "moved: the original refused it or returned an error, or a param is outside the ranges it registers, or the "
                        "case is listed with its reason in a --judged file (a judgement of what the original defines, reversible by "
                        "leaving the file out); never a pass or a failure. Harness failures (emulator fault, timeout, not run) are not "
                        "the original's behaviour and stay in contract unless --move-harness-failures is given or a judgement lists them."})
    jdump(os.path.join(a.results, "ooc.json"), doc)
    print("ooc:", json.dumps(doc["counts"]))
    return 0


def cmd_ooc_set(a):
    paths(a)
    cands = jload(os.path.join(a.testset, "out_of_contract.json"))["cases"]
    os.makedirs(a.out, exist_ok=True)
    inp = os.path.join(a.out, "inputs")
    if not os.path.exists(inp):
        try:
            os.symlink(os.path.join(a.testset, "inputs"), inp)
        except OSError:
            shutil.copytree(os.path.join(a.testset, "inputs"), inp)
    jdump(os.path.join(a.out, "cases.json"), {"format": 1, "note": "out of contract candidates; the rewrite refuses them",
                                              "cases": cands})
    print("ooc-set: {} candidates in {}".format(len(cands), a.out))
    return 0


# ---------------------------------------------------------------------------------------------
# call records

def read_jsonl(path):
    out, bad = [], 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                bad += 1
    return out, bad


def layer_map_from_param_map(param_map, plugin=PLUGIN):
    out = {}
    if param_map and os.path.exists(param_map):
        lines = open(param_map).read().splitlines()
        cols = lines[0].split("\t")
        for line in lines[1:]:
            r = dict(zip(cols, line.split("\t")))
            if r.get("plugin") == plugin and r.get("vfx_kind") == "layer" and r.get("ae_index"):
                out[int(r["ae_index"])] = r["vfx_id"]
    return out


# The harness logs the SMART_RENDER pre-render checkouts as `prerender_cb.checkout_layer` (with a `checkout_id`) and
# the pixel reads as `smart_cb.checkout_layer_pixels` (the same `checkout_id`, no time). The dotted spelling is the
# one this tool used first; it is still accepted.
LAYER_CHECKOUTS = ("prerender_cb.checkout_layer", "prerender.cb.checkout_layer")
PIXEL_CHECKOUTS = ("smart_cb.checkout_layer_pixels",)


def plugin_record(lines, layer_names=None, set_layers=None):
    """Parse a harness record: frames {n: time}, reads {n: [(layer, time, footage_frame, inside)]},
    param samples {n: [(param_index, time)]}, complete (a case_done with status rendered), unimplemented count.

    Reads. The legacy render path: an `inter.checkout_param` of the input (index 0) or of a layer param. SMART_RENDER:
    a pixel checkout (`PIXEL_CHECKOUTS`) resolved through the pre-render checkout (`LAYER_CHECKOUTS`) with the same
    `checkout_id` in the same output frame to (layer, time, footage frame, inside). A pre-render checkout whose
    pixels are never checked out is a registration, not a read. A record with no pixel checkout at all (a stand-in
    written before the pixel events existed) keeps the old rule: every pre-render checkout is a read.
    `set_layers`, when given, is the set of layer names the case provides (`input` included): reads of any other
    layer are dropped (an unset layer is answered with size 0 and has no pixels; the rewrite has no such layer)."""
    layer_names = layer_names or {}
    frames, reads, params = {}, collections.defaultdict(list), collections.defaultdict(list)
    pre = collections.defaultdict(list)          # pre-render checkouts per output frame, in order (the fallback)
    pix = collections.defaultdict(list)          # pixel checkouts per output frame, resolved when they happen
    ids = {}                                     # checkout_id -> pre-render checkout of the current output frame
    complete, unimpl, cur, any_pixels = False, 0, None, False
    for ev in lines:
        e = ev.get("ev")
        if e == "frame":
            frames[ev["out_frame"]] = ev["time"]
            cur = ev["out_frame"]
            ids = {}
        elif e == "case_done":
            complete = ev.get("status") == "rendered"
        elif e == "cb":
            if ev.get("unimplemented"):
                unimpl += 1
            fn, n = ev.get("fn", ""), ev.get("out_frame", cur)
            idx = ev.get("param_index")
            if fn in PIXEL_CHECKOUTS:
                any_pixels = True
                pix[n].append(ids.get(ev.get("checkout_id"), ("?", None, None, None)))
                continue
            is_pre = fn in LAYER_CHECKOUTS
            is_read = is_pre or (fn == "inter.checkout_param" and (idx == 0 or "layer" in ev))
            if is_read and "time" in ev:
                name = "input" if idx == 0 else layer_names.get(idx, "param{}".format(idx))
                inside = ev.get("inside")
                if inside is None and isinstance(ev.get("layer"), str):
                    inside = "outside" not in ev["layer"]
                item = (name, ev["time"], ev.get("footage_frame"), inside)
                if is_pre:
                    pre[n].append(item)
                    ids[ev.get("checkout_id")] = item
                else:
                    reads[n].append(item)
            elif fn == "inter.checkout_param" and "time" in ev:
                params[n].append((idx, ev["time"]))
    for n, items in (pix if any_pixels else pre).items():
        reads[n].extend(items)
    if set_layers is not None:
        for n in list(reads):
            reads[n] = [r for r in reads[n] if r[0] in set_layers or r[0] == "?"]
    return {"frames": frames, "reads": reads, "params": params, "complete": complete, "unimplemented": unimpl}


def rewrite_record(lines):
    frames, reads, clip_frames = {}, collections.defaultdict(list), {}
    for ev in lines:
        e = ev.get("ev")
        if e == "frame":
            frames[ev["out_frame"]] = ev["time"]
        elif e == "read":
            reads[ev["out_frame"]].append((ev["layer"], ev["time"], ev.get("index"), True))
            clip_frames[ev["layer"]] = ev.get("clip_frames")
    return {"frames": frames, "reads": reads, "clip_frames": clip_frames}


def normalize_reads(items, clip_frames, plugin_side):
    """(layer, time, index) triples; a plugin footage frame outside the clip becomes None like the rewrite's."""
    out = []
    for layer, t, idx, inside in items:
        if plugin_side:
            n = clip_frames.get(layer)
            if inside is False or idx is None or idx < 0 or (n is not None and idx >= n):
                idx = None
        out.append((layer, t, idx))
    return out


def diff_frame(p, r, ordered_mode, index_only):
    """The mismatches of one output frame between plugin reads p and rewrite reads r (lists of triples)."""
    key = (lambda x: (x[0], x[2])) if index_only else (lambda x: x)
    if ordered_mode:
        pl, rl = [key(x) for x in p], [key(x) for x in r]
        if pl == rl:
            return []
        for i in range(max(len(pl), len(rl))):
            if i >= len(pl) or i >= len(rl) or pl[i] != rl[i]:
                return [{"kind": "order_or_content", "position": i, "plugin": pl[i] if i < len(pl) else None,
                         "rewrite": rl[i] if i < len(rl) else None}]
    P, R = set(map(key, p)), set(map(key, r))
    only_p, only_r = sorted(P - R, key=str), sorted(R - P, key=str)
    out = []
    if not index_only:
        for x in list(only_p):
            m = [y for y in only_r if (y[0], y[2]) == (x[0], x[2])]
            if m:
                out.append({"kind": "wrong_time", "layer": x[0], "index": x[2], "plugin_time": x[1], "rewrite_time": m[0][1]})
                only_p.remove(x)
                only_r.remove(m[0])
        for x in list(only_p):
            m = [y for y in only_r if (y[0], y[1]) == (x[0], x[1])]
            if m:
                out.append({"kind": "wrong_index", "layer": x[0], "time": x[1], "plugin_index": x[2], "rewrite_index": m[0][2]})
                only_p.remove(x)
                only_r.remove(m[0])
    out += [{"kind": "missing_read", "read": list(x)} for x in only_p]
    out += [{"kind": "extra_read", "read": list(x)} for x in only_r]
    return out


def compare_records(plug, rew, ordered_mode=False, index_only=False):
    """Per output frame comparison of one case. Returns dict(frames_compared, frames_matching, first_mismatch, mismatches)."""
    clip_frames = rew["clip_frames"]
    nums = sorted({n for n in set(plug["frames"]) | set(rew["frames"]) | set(plug["reads"]) | set(rew["reads"]) if n is not None})
    bad, match = [], 0
    for n in nums:
        issues = []
        if plug["frames"].get(n) != rew["frames"].get(n):
            kind = "missing_frame" if n not in plug["frames"] else "extra_frame" if n not in rew["frames"] else "wrong_frame_time"
            issues.append({"kind": kind, "plugin_time": plug["frames"].get(n), "rewrite_time": rew["frames"].get(n)})
        p = normalize_reads(plug["reads"].get(n, []), clip_frames, True)
        r = normalize_reads(rew["reads"].get(n, []), clip_frames, False)
        issues += diff_frame(p, r, ordered_mode, index_only)
        if issues:
            bad.append({"out_frame": n, "issues": issues})
        else:
            match += 1
    return {"frames_compared": len(nums), "frames_matching": match, "first_mismatch": bad[0] if bad else None,
            "mismatching_frames": len(bad)}


def cmd_calls(a):
    paths(a)
    layers = layer_map_from_param_map(a.param_map)
    for item in (a.layer_map or "").split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            layers[int(k)] = v
    cases = load_cases(a.testset)
    rows, compared, matching = [], 0, 0
    for c in cases:
        pp = os.path.join(a.plugin_records, c["id"] + ".jsonl")
        rp = os.path.join(a.rewrite_records, c["id"] + ".jsonl")
        row = {"id": c["id"]}
        if not os.path.exists(pp):
            row["status"] = "no_plugin_record"
        elif not os.path.exists(rp):
            row["status"] = "no_rewrite_record"
        else:
            plug_lines, bad = read_jsonl(pp)
            plug = plugin_record(plug_lines, layers, {"input"} | set(c.get("layers") or {}))
            rew = rewrite_record(read_jsonl(rp)[0])
            if not plug["complete"]:
                row["status"] = "incomplete_plugin_record"
            else:
                res = compare_records(plug, rew, a.ordered, a.index_only)
                compared += 1
                row["status"] = "match" if res["mismatching_frames"] == 0 else "mismatch"
                matching += row["status"] == "match"
                row.update(res)
                row["param_samples_plugin"] = sum(len(v) for v in plug["params"].values())
                row["unimplemented_callbacks"] = plug["unimplemented"]
        rows.append(row)
    count = collections.Counter(r["status"] for r in rows)
    doc = {"module": "fgbg-core", "measured_at": today(), "mode": {"ordered": a.ordered, "index_only": a.index_only},
           "command": "phase_b.py calls --plugin-records {} --rewrite-records {}".format(a.plugin_records, a.rewrite_records),
           "definition": "plugin reads: SMART_RENDER pixel checkouts resolved through their pre-render checkout id "
                         "(registrations without pixels are not reads), or legacy inter.checkout_param reads; layers "
                         "the case does not set are left out; rewrite reads: VFX_CALL_RECORD read lines",
           "compared": compared, "matching": matching, "statuses": dict(count),
           "param_times": "plugin side samples are listed per case (param_samples_plugin) but not compared: the rewrite does not record parameter sample times",
           "cases": rows}
    jdump(os.path.join(a.results, "calls.json"), doc)
    print("calls: compared {}, matching {}, statuses {}".format(compared, matching, dict(count)))
    return 0


def cmd_rewrite_records(a):
    paths(a)
    cases = load_cases(a.testset)
    if a.only:
        keep = set(a.only.split(","))
        cases = [c for c in cases if c["id"] in keep]
    os.makedirs(a.out, exist_ok=True)
    base_p = os.path.join(HERE, "results", "base.tsv")
    base = load_tsv(base_p) if os.path.exists(base_p) else {}

    def one(c):
        rc, out, err = render(a.vfx, c, a.testset, os.path.join(a.out, c["id"] + ".jsonl"))
        same = base.get(c["id"], {}).get("sha") == hashlib.sha256(out).hexdigest()
        return rc, same

    with cf.ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(one, cases))
    print("rewrite-records: {} cases, {} with a nonzero exit, output equal to results/base.tsv for {}".format(
        len(res), sum(1 for r in res if r[0]), sum(1 for r in res if r[1])))
    return 0


# ---------------------------------------------------------------------------------------------
# mutation kills

def load_mutant_results(mutants_dir, mid):
    d = os.path.join(mutants_dir, mid)
    rp, cp = os.path.join(d, "results.tsv"), os.path.join(d, "changed.txt")
    if os.path.exists(rp):
        return {"rows": load_tsv(rp)}
    if os.path.exists(cp):
        return {"changed": {l for l in open(cp).read().splitlines() if l}}
    return None


def compute_kills(mutants, built, base, ref_sha, mres, excluded, grade_pass=None):
    """A mutant is killed iff, on some in-contract case where the base output equals the reference,
    its output differs from the reference. Returns (list per mutant, info)."""
    disc = set()
    for cid, b in base.items():
        if cid in excluded or b["exit"] != 0:
            continue
        if ref_sha.get(cid) and ref_sha[cid] == b["sha"]:
            if grade_pass is None or cid in grade_pass:
                disc.add(cid)
    disagree = []
    if grade_pass is not None:
        disagree = sorted(c for c in grade_pass if c not in disc and c not in excluded)
    out = []
    for m in mutants:
        mid = m["id"]
        r = {"id": mid, "file": m.get("file"), "what": m.get("what")}
        res = mres.get(mid)
        if not built.get(mid, False) or res is None:
            r.update(status="not_built", killing_cases=[], changed_vs_base=None)
            out.append(r)
            continue
        if "rows" in res:
            rows = res["rows"]
            changed = {cid for cid, b in base.items() if rows.get(cid) is not None and
                       (rows[cid]["exit"], rows[cid]["sha"]) != (b["exit"], b["sha"])}
            killers = sorted(cid for cid in disc if cid in rows and (rows[cid]["exit"] != 0 or rows[cid]["sha"] != ref_sha[cid]))
        else:
            changed = set(res["changed"])
            killers = sorted(changed & disc)
        r["changed_vs_base"] = len(changed)
        r["killing_cases"] = killers
        if killers:
            r["status"] = "killed"
        else:
            r["status"] = "survivor"
            if not changed:
                r["reason"] = "changes no case against the base rewrite (an equivalent candidate: an argument, not a measurement)"
            elif not disc:
                r["reason"] = "no in-contract case where the base equals the reference"
            else:
                r["reason"] = ("changes {} cases, none of them one where the base equals the reference: "
                               "the base already differs there or there is no reference".format(len(changed)))
        out.append(r)
    return out, {"discriminating_cases": len(disc), "grade_disagreements": disagree}


def cmd_kills(a):
    paths(a)
    base_p = a.base or os.path.join(HERE, "results", "base.tsv")
    base = load_tsv(base_p)
    cases = load_cases(a.testset)
    muts = jload(a.mutants_json or os.path.join(HERE, "mutants.json"))
    mdir = a.mutants_dir if os.path.isabs(a.mutants_dir) else os.path.join(HERE, a.mutants_dir)
    built = {}
    sp = os.path.join(mdir, "mutants_summary.tsv")
    if os.path.exists(sp):
        for line in open(sp).read().splitlines()[1:]:
            f = line.split("\t")
            built[f[0]] = f[1] in ("changed-cases", "no-change")
    else:
        built = {m["id"]: True for m in muts}
    mres = {m["id"]: load_mutant_results(mdir, m["id"]) for m in muts}
    ref_sha = {}
    for c in cases:
        p = os.path.join(a.refs, c["reference"])
        if os.path.isfile(p):
            ref_sha[c["id"]] = sha_file(p)
    excluded = set()
    ooc_p = a.ooc or os.path.join(a.results, "ooc.json")
    if os.path.exists(ooc_p):
        excluded = {m["id"] for m in jload(ooc_p)["moved"]}
    gp = os.path.join(a.results, "grade.json")
    grade_pass = None
    stale = []
    if os.path.exists(gp):
        g = jload(gp)
        grade_pass = {r["id"] for r in g["cases"] if r["status"] == "pass"}
        stale = g["totals"].get("base_tsv_stale", [])
    kills, info = compute_kills(muts, built, base, ref_sha, mres, excluded, grade_pass)
    total = sum(1 for k in kills if k["status"] != "not_built")
    killed = sum(1 for k in kills if k["status"] == "killed")
    doc = {"module": "fgbg-core", "measured_at": today(), "definition": "killed iff its output differs from the reference on at least one in-contract case "
           "where the base rewrite equals the reference; cases where the base differs, has no reference or is out of contract cannot discriminate",
           "command": "phase_b.py kills --refs {}".format(a.refs), "mutants_total": total, "killed": killed,
           "survivors": [k["id"] for k in kills if k["status"] == "survivor"],
           "not_built": [k["id"] for k in kills if k["status"] == "not_built"],
           "discriminating_cases": info["discriminating_cases"], "excluded_out_of_contract": len(excluded),
           "grade_disagreements": info["grade_disagreements"], "base_tsv_stale": stale, "mutants": kills}
    jdump(os.path.join(a.results, "kills.json"), doc)
    print("kills: {} of {} killed, survivors {}, discriminating cases {}{}".format(
        killed, total, doc["survivors"], info["discriminating_cases"],
        "; BASE.TSV STALE for {} cases".format(len(stale)) if stale else ""))
    return 0


# ---------------------------------------------------------------------------------------------
# summary

def build_summary(cases, candidates, grade, ooc, kills, calls, prev, built_ids, grid):
    """Every number from the result files. Returns the summary dict."""
    moved = {m["id"] for m in ooc["moved"]} if ooc else set()
    rows = [r for r in grade["cases"] if r["id"] not in moved]
    cmp_rows = [r for r in rows if r["status"] in ("pass", "FAIL")]
    in_contract = len(cases) - len(moved)
    n_candidates = len(ooc["candidates"]) if ooc else len(candidates)
    identical = sum(1 for r in cmp_rows if r["status"] == "pass")
    differing = len(cmp_rows) - identical
    no_ref = sum(1 for r in rows if r["status"] == "no reference")
    standin_all = sum(1 for c in cases if c["id"] not in moved and "standin" in grid.get(c["id"], ("", set()))[1])
    logic = sum(1 for r in cmp_rows if r["logic_parity"])
    values = {"bytes_compared": sum(r.get("bytes_compared") or 0 for r in cmp_rows),
              "floats_compared": sum(r.get("floats_compared") or 0 for r in cmp_rows),
              "differing_values": sum(r.get("differing_values") or 0 for r in cmp_rows),
              "max_ulp": max([r["max_ulp"] for r in cmp_rows if r.get("max_ulp") is not None], default=None)}
    nan = sum(r.get("nan_mismatch") or 0 for r in cmp_rows)
    call = {"compared": calls["compared"], "matching": calls["matching"]} if calls else {"compared": 0, "matching": 0}
    if kills:
        mut = {"total": kills["mutants_total"], "killed": kills["killed"], "survivors": list(kills["survivors"])}
    else:
        mut = {"total": len(built_ids), "killed": 0, "survivors": []}
        for mid in sorted(built_ids):
            res = built_ids[mid]
            if res == "no-change":
                mut["survivors"].append(mid)
    blockers = []
    if not cmp_rows:
        blockers.append("no original reference compared yet (the emulator has not rendered this module's cases)")
    elif no_ref:
        blockers.append("{} of {} in-contract cases have no reference".format(no_ref, in_contract))
    if differing:
        blockers.append("{} of {} compared cases differ from the original ({} values differ{})".format(
            differing, len(cmp_rows), values["differing_values"],
            ", {} of them NaN mismatches with no finite ULP".format(nan) if nan else ""))
    if grade["totals"].get("audit_disagreements"):
        blockers.append("fx-parity and the byte audit disagree on {} cases".format(len(grade["totals"]["audit_disagreements"])))
    if grade["totals"].get("base_tsv_stale"):
        blockers.append("results/base.tsv does not match the rewrite that was graded for {} cases: rerun phase A".format(
            len(grade["totals"]["base_tsv_stale"])))
    if ooc and ooc["counts"]["not_judged"]:
        blockers.append("{} cases failed in the emulation (not the original's behaviour) and have no reference".format(ooc["counts"]["not_judged"]))
    if calls and (calls.get("statuses", {}).get("no_plugin_record", 0) + calls.get("statuses", {}).get("incomplete_plugin_record", 0)):
        st = calls["statuses"]
        blockers.append("{} cases have no complete plugin call record".format(st.get("no_plugin_record", 0) + st.get("incomplete_plugin_record", 0)))
    if not call["compared"]:
        blockers.append("no call record compared yet")
    elif call["matching"] != call["compared"]:
        blockers.append("{} of {} call records differ".format(call["compared"] - call["matching"], call["compared"]))
    if not kills:
        blockers.append("mutation kills not computed yet")
    elif mut["survivors"]:
        blockers.append("mutants survive: {}".format(", ".join(mut["survivors"])))
    verified = (len(cmp_rows) >= MIN_CASES and len(cmp_rows) == in_contract and differing == 0
                and values["differing_values"] == 0 and call["compared"] > 0 and call["matching"] == call["compared"]
                and mut["total"] > 0 and mut["killed"] == mut["total"] and not mut["survivors"]
                and not grade["totals"].get("audit_disagreements") and not grade["totals"].get("base_tsv_stale"))
    if not cmp_rows:
        blockers = ["emulator side not run yet"] + [b for b in blockers if not b.startswith("no original reference")]
    cmds = list(prev.get("commands", [])) if prev else list(PHASE_A_COMMANDS)
    for c in (grade.get("command"), (ooc or {}).get("command"), (calls or {}).get("command"), (kills or {}).get("command")):
        if c and c not in cmds:
            cmds.append(c)
    return {
        "module": "fgbg-core", "plugin": (prev or {}).get("plugin", "TwixtorPro_64.aex"),
        "plugin_sha256": (prev or {}).get("plugin_sha256", PLUGIN_SHA), "effect": EFFECT,
        "status": "verified" if verified else "in_progress",
        "blockers": [] if verified else blockers,
        "cases": {"in_contract": in_contract, "compared": len(cmp_rows), "identical": identical, "differing": differing,
                  "out_of_contract": n_candidates + len(moved), "logic_parity": logic},
        "values": values, "call_records": call, "mutants": mut,
        "stand_ins": ["{} keyed cases (tag standin in testset/grid.tsv) use the harness's labelled keyframe interpolation on both sides; "
                      "{} of them compared, counted as logic_parity and not as full parity".format(standin_all, logic)],
        "fixes": list((prev or {}).get("fixes", [])),
        "commands": cmds,
        "measured_at": grade.get("measured_at", today()),
    }


def cmd_summary(a):
    paths(a)
    gp = os.path.join(a.results, "grade.json")
    if not os.path.exists(gp):
        raise SystemExit("summary: {} does not exist; run `phase_b.py grade --refs DIR` first".format(gp))
    opt = lambda n: jload(os.path.join(a.results, n)) if os.path.exists(os.path.join(a.results, n)) else None
    cases = load_cases(a.testset)
    oc_p = os.path.join(a.testset, "out_of_contract.json")
    cands = jload(oc_p)["cases"] if os.path.exists(oc_p) else []
    real = os.path.join(HERE, "results", "summary.json")
    prev = opt("summary.json") or (jload(real) if os.path.exists(real) else None)
    msum = os.path.join(HERE, "mutants", "mutants_summary.tsv")
    built = {}
    for p in (msum, os.path.join(HERE, "results", "mutants_summary.tsv")):
        if os.path.exists(p):
            for line in open(p).read().splitlines()[1:]:
                f = line.split("\t")
                if f[1] in ("changed-cases", "no-change"):
                    built[f[0]] = f[1]
            break
    doc = build_summary(cases, cands, jload(gp), opt("ooc.json"), opt("kills.json"), opt("calls.json"), prev, built,
                        load_grid(a.testset))
    jdump(os.path.join(a.results, "summary.json"), doc)
    c = doc["cases"]
    print("summary: status {}, in contract {}, compared {}, identical {}, differing {}, out of contract {}, logic parity {}; "
          "mutants {}/{} killed; call records {}/{}".format(doc["status"], c["in_contract"], c["compared"], c["identical"], c["differing"],
                                                           c["out_of_contract"], c["logic_parity"], doc["mutants"]["killed"],
                                                           doc["mutants"]["total"], doc["call_records"]["matching"],
                                                           doc["call_records"]["compared"]))
    return 0


# ---------------------------------------------------------------------------------------------
# triage

def default_text(p):
    d = p.get("default")
    if isinstance(d, bool):
        return "1" if d else "0"
    return None if d is None else str(d)


def is_default(value, p):
    """Does the --param value equal the control's default? Keyed values are never default."""
    if p is None or "@" in value:
        return False
    d = p.get("default")
    if isinstance(d, bool):
        d = 1 if d else 0
    try:
        return float(value) == float(d) and not (float(value) == 0 and value.startswith("-"))
    except (TypeError, ValueError):
        return False


def load_defaults(path, effect=EFFECT):
    if path and os.path.exists(path):
        doc = jload(path)
        for e in doc if isinstance(doc, list) else doc.get("effects", []):
            if e.get("id") == effect:
                return {p["id"]: p for p in e.get("params", []) if isinstance(p, dict)}
    return {}


def split_params(case):
    out = []
    for p in case.get("params") or []:
        k, _, v = p.partition("=")
        out.append((k, v))
    return out


def ladder(case, defaults):
    """Probe cases isolating a cause: the case itself, the case with each non-default setting reverted,
    the default case, and the default case with each non-default setting alone."""
    items = split_params(case)
    nondef = [(k, v) for k, v in items if not is_default(v, defaults.get(k))]
    rungs = [("full", "the failing case", items)]
    for k, v in nondef:
        rungs.append(("revert:" + k, "{} reverted to its default".format(k), [x for x in items if x[0] != k]))
    rungs.append(("default", "every setting at its default", []))
    for k, v in nondef:
        rungs.append(("only:" + k, "default case with only {}={}".format(k, v), [(k, v)]))
    seen, out = set(), []
    for kind, what, its in rungs:
        key = tuple(its)
        if key in seen:
            continue
        seen.add(key)
        out.append((kind, what, its))
    return out


def write_probe(case, rungs, testset, probe_dir, cases_meta):
    inputs = [i for i in cases_meta.get("inputs", []) if i["file"] == case["input"]]
    os.makedirs(os.path.join(probe_dir, "inputs"), exist_ok=True)
    src = os.path.join(testset, case["input"])
    dst = os.path.join(probe_dir, case["input"])
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
    pix = case["reference"].rsplit(".", 1)[1]
    out_cases, lad = [], []
    for i, (kind, what, its) in enumerate(rungs):
        params = ["{}={}".format(k, v) for k, v in its]
        h = hashlib.sha1(("|".join(params)).encode()).hexdigest()[:8]
        label = "probe{:02d}".format(i)
        cid = "{}.{}.{}-{}x{}.d{}.{}".format(case["effect"], label, case["source"], case["width"], case["height"], case["depth"], h)
        c = dict(case, id=cid, label=label, params=params, reference="{}.{}".format(cid, pix))
        out_cases.append(c)
        lad.append({"id": cid, "rung": i, "kind": kind, "what": what, "params": params})
    doc = {k: v for k, v in cases_meta.items() if k not in ("cases", "inputs")}
    doc.update({"format": 1, "note": "ladder of probe cases from {}".format(case["id"]), "inputs": inputs, "cases": out_cases})
    jdump(os.path.join(probe_dir, "cases.json"), doc)
    jdump(os.path.join(probe_dir, "ladder.json"), {"from": case["id"], "rungs": lad})
    return lad


def triage_tables(rows, defaults):
    """Counts per setting among compared cases: (differ, total) by block, tag, depth, size class and param=value."""
    cmp_rows = [r for r in rows if r["status"] in ("pass", "FAIL")]
    groups = collections.defaultdict(lambda: [0, 0])

    def add(key, r):
        groups[key][1] += 1
        groups[key][0] += r["status"] == "FAIL"

    for r in cmp_rows:
        add(("block", r["block"]), r)
        for t in r["tags"]:
            add(("tag", t), r)
        add(("depth", str(r["depth"])), r)
        add(("frames", str(r["frames"])), r)
        for p in r["params"]:
            k, _, v = p.partition("=")
            add(("param", "{}={}".format(k, v)), r)
        if not r["params"]:
            add(("param", "(all defaults)"), r)
    return groups


def cmd_triage(a):
    paths(a)
    gp = os.path.join(a.results, "grade.json")
    if not os.path.exists(gp):
        raise SystemExit("triage: {} does not exist; run grade first".format(gp))
    g = jload(gp)
    probe = a.probe if os.path.isabs(a.probe) else os.path.join(HERE, a.probe)
    if a.ladder_report:
        lad = jload(os.path.join(probe, "ladder.json"))["rungs"]
        rep = {c["id"]: c for c in jload(a.ladder_report)["cases"]}
        print("ladder from {}".format(jload(os.path.join(probe, "ladder.json"))["from"]))
        for r in lad:
            st = rep.get(r["id"], {}).get("status", "not run")
            print("  rung {:2d}  {:<12}  {:<14}  {}".format(r["rung"], st, r["kind"], r["what"]))
        print("read: the failing case differs and a rung where it passes names the setting that is needed to fail;")
        print("a revert rung that passes means that setting is needed; an only rung that fails means it is enough.")
        return 0
    rows = g["cases"]
    defaults = load_defaults(a.vfx_list)
    failing = [r for r in rows if r["status"] == "FAIL"]
    cmp_n = sum(1 for r in rows if r["status"] in ("pass", "FAIL"))
    print("triage: {} compared, {} differ, {} no reference".format(cmp_n, len(failing), sum(1 for r in rows if r["status"] == "no reference")))
    if not failing:
        print("nothing differs; no ladder written")
        return 0
    groups = triage_tables(rows, defaults)
    for kind in ("block", "tag", "depth", "frames", "param"):
        items = [(k[1], v) for k, v in groups.items() if k[0] == kind and v[0]]
        items.sort(key=lambda kv: (-kv[1][0] / kv[1][1], -kv[1][0], kv[0]))
        if not items:
            continue
        print("\nby {} (differing / compared, settings with a difference only):".format(kind))
        for name, (d, t) in items[: a.top]:
            print("  {:<34} {:>4} / {:<4} {:>5.1f}%".format(name, d, t, 100.0 * d / t))
    failing.sort(key=lambda r: (r["width"] * r["height"], r["frames"], r["depth"], r["id"]))
    print("\nsmallest failing cases (size, frames, depth):")
    for r in failing[: a.top]:
        print("  {:>4}x{:<4} {:>2}f d{:<2}  {:>6} values  {}  [{}]".format(r["width"], r["height"], r["frames"], r["depth"],
                                                                         r.get("differing_values"), r["id"], " ".join(r["params"])))
    small = failing[0]
    cases = {c["id"]: c for c in load_cases(a.testset)}
    meta = jload(os.path.join(a.testset, "cases.json"))
    rungs = ladder(cases[small["id"]], defaults)
    lad = write_probe(cases[small["id"]], rungs, a.testset, probe, meta)
    print("\nladder for {} ({} rungs) written to {}:".format(small["id"], len(lad), probe))
    for r in lad:
        print("  rung {:2d}  {:<14} {}".format(r["rung"], r["kind"], " ".join(r["params"]) or "(defaults)"))
    print("render it: run_harness.py --testset {0} --refs {0}_refs; then fx-parity run --testset {0} --refs {0}_refs --max-ulp 0 --json > ladder.json;".format(probe))
    print("then: phase_b.py triage --ladder-report ladder.json")
    return 0


# ---------------------------------------------------------------------------------------------
# selftest (stand-in oracles, no plugin)

class Check:
    def __init__(self):
        self.items = []

    def __call__(self, name, ok, detail=""):
        self.items.append({"check": name, "ok": bool(ok), "detail": detail})
        print("  [{}] {}{}".format("ok" if ok else "FAIL", name, (": " + str(detail)) if detail else ""))

    @property
    def failed(self):
        return [i for i in self.items if not i["ok"]]


def ns(**kw):
    return argparse.Namespace(**kw)


def flip32(data, sample, mask):
    b = bytearray(data)
    v = struct.unpack_from("<I", b, sample * 4)[0] ^ mask
    struct.pack_into("<I", b, sample * 4, v)
    return bytes(b)


def cmd_selftest(a):
    paths(a)
    work = a.workdir if os.path.isabs(a.workdir) else os.path.join(HERE, a.workdir)
    real_results = os.path.realpath(os.path.join(HERE, "results"))
    if os.path.realpath(work) in (real_results, HERE) or not os.path.realpath(work).startswith(real_results + os.sep):
        raise SystemExit("selftest writes only under results/selftest/")
    if os.path.exists(work) and not a.keep:
        shutil.rmtree(work)
    os.makedirs(work, exist_ok=True)
    ck = Check()
    cases = load_cases(a.testset)
    by_id = {c["id"]: c for c in cases}
    base = load_tsv(os.path.join(HERE, "results", "base.tsv"))
    mutants = jload(os.path.join(HERE, "mutants.json"))
    phase_a = {m["id"]: set(l for l in open(os.path.join(HERE, "mutants", m["id"], "changed.txt")).read().split()) for m in mutants
               if os.path.exists(os.path.join(HERE, "mutants", m["id"], "changed.txt"))}
    a_changing = sorted(m for m, s in phase_a.items() if s)
    print("selftest in {}; phase A: {} of {} mutants change at least one case ({})".format(work, len(a_changing), len(phase_a),
                                                                                       "survivors " + ",".join(sorted(set(phase_a) - set(a_changing)))))
    # (a) perfect oracle: the rewrite's own base renders as references
    perf = os.path.join(work, "perfect")
    refs = os.path.join(perf, "refs")
    os.makedirs(refs, exist_ok=True)
    outs = {}
    same = 0
    for c in cases:
        rc, out, err = render(a.vfx, c, a.testset)
        if rc != 0:
            raise SystemExit("vfx failed on {}: {}".format(c["id"], err))
        outs[c["id"]] = out
        with open(os.path.join(refs, c["reference"]), "wb") as fh:
            fh.write(out)
        same += hashlib.sha256(out).hexdigest() == base[c["id"]]["sha"]
    ck("perfect: bin vfx reproduces results/base.tsv", same == len(cases), "{} of {} outputs equal their base.tsv hash".format(same, len(cases)))
    common = dict(testset=a.testset, vfx=a.vfx, fx_parity=a.fx_parity, jobs=1, effect=EFFECT, no_audit=False, base=None)
    cmd_grade(ns(refs=refs, results=perf, **common))
    g = jload(os.path.join(perf, "grade.json"))
    t = g["totals"]
    ck("perfect: 0 differing, all compared", t["fail"] == 0 and t["pass"] == len(cases) and t["differing_values"] == 0,
       "pass {}, FAIL {}, differing values {}, max ULP {}".format(t["pass"], t["fail"], t["differing_values"], t["max_ulp"]))
    ck("perfect: no audit disagreement", not t["audit_disagreements"] and not t["base_tsv_stale"])
    kargs = dict(testset=a.testset, base=None, mutants_json=None, mutants_dir="mutants", ooc=None)
    cmd_kills(ns(refs=refs, results=perf, **kargs))
    k = jload(os.path.join(perf, "kills.json"))
    ck("perfect: mutants killed equals phase A count", k["killed"] == len(a_changing) and k["survivors"] == sorted(set(phase_a) - set(a_changing)),
       "killed {} of {} (phase A: {} change an output); survivors {}".format(k["killed"], k["mutants_total"], len(a_changing), k["survivors"]))
    # (e) call records on the real rewrite records and synthetic plugin records built from them
    if a.vfx_rec and os.path.exists(a.vfx_rec):
        try:
            rec_dir = os.path.join(work, "calls", "rewrite")
            cmd_rewrite_records(ns(out=rec_dir, vfx=a.vfx_rec, testset=a.testset, only=None, jobs=1, results=perf))
            selftest_calls(a, ck, cases, rec_dir, os.path.join(work, "calls"), perf)
        except Exception as e:
            ck("calls: crashed", False, "{}: {}".format(type(e).__name__, e))
    else:
        ck("calls: record capable vfx available", False, "no --vfx-rec given or found")
    def sec_b():
        # (b) perturbed oracle
        pert = os.path.join(work, "perturbed")
        prefs = os.path.join(pert, "refs")
        os.makedirs(prefs, exist_ok=True)
        for c in cases:
            os.symlink(os.path.join(refs, c["reference"]), os.path.join(prefs, c["reference"]))
        expect = {}

        def put(cid, data):
            p = os.path.join(prefs, by_id[cid]["reference"])
            os.remove(p)
            with open(p, "wb") as fh:
                fh.write(data)

        def pick(pred):
            for c in cases:
                if pred(c) and c["id"] not in expect:
                    return c
            return None

        c8 = pick(lambda c: c["depth"] == 8 and len(outs[c["id"]]) > 40)
        b = bytearray(outs[c8["id"]]); pos = len(b) // 2; b[pos] ^= 1
        put(c8["id"], bytes(b)); expect[c8["id"]] = {"values": 1, "ulp": None, "where": sample_where(c8, pos)}
        c16 = pick(lambda c: c["depth"] == 16 and len(outs[c["id"]]) > 40)
        b = bytearray(outs[c16["id"]]); pos16 = len(b) // 2 - (len(b) // 2) % 2; b[pos16] ^= 0x80
        put(c16["id"], bytes(b)); expect[c16["id"]] = {"values": 1, "ulp": None}
        c32a = pick(lambda c: c["depth"] == 32 and len(outs[c["id"]]) > 100)
        put(c32a["id"], flip32(outs[c32a["id"]], 7, 1)); expect[c32a["id"]] = {"values": 1, "ulp": 1}
        c32b = pick(lambda c: c["depth"] == 32 and len(outs[c["id"]]) > 100)
        put(c32b["id"], flip32(outs[c32b["id"]], 9, 8)); expect[c32b["id"]] = {"values": 1, "ulp": 8}
        zero = None
        for want in (0x80000000, 0):        # a real -0 in an output first, else a +0 turned into -0
            for c in cases:
                if c["depth"] == 32 and c["id"] not in expect:
                    for i, v in enumerate(samples(32, outs[c["id"]])):
                        if v == want:
                            zero = (c, i, v)
                            break
                if zero:
                    break
            if zero:
                break
        zc, zi, zv = zero
        put(zc["id"], flip32(outs[zc["id"]], zi, 0x80000000)); expect[zc["id"]] = {"values": 1, "ulp": 0}
        nan_note = None
        for c in cases:
            if c["depth"] == 32 and c["id"] not in expect:
                w = samples(32, outs[c["id"]])
                for i, v in enumerate(w):
                    if is_nan_bits(v):
                        nan_note = ("payload", c, i)
                        break
            if nan_note:
                break
        if nan_note:
            _, nc, ni = nan_note
            put(nc["id"], flip32(outs[nc["id"]], ni, 1)); expect[nc["id"]] = {"values": 1, "ulp": 0, "nan": 1}
        else:
            nc = pick(lambda c: c["depth"] == 32 and len(outs[c["id"]]) > 100)
            data = bytearray(outs[nc["id"]]); struct.pack_into("<I", data, 20, 0x7FC00000)
            put(nc["id"], bytes(data)); expect[nc["id"]] = {"values": 1, "ulp": 0, "nan": 1}
        # every case of the smallest mutant: both its cases get a bit flip, so the mutant has no discriminating case left
        small = min((m for m in a_changing if phase_a[m]), key=lambda m: (len(phase_a[m]), m))
        for cid in sorted(phase_a[small]):
            if cid not in expect:
                c = by_id[cid]
                put(cid, flip32(outs[cid], 0, 1) if c["depth"] == 32 else bytes([outs[cid][0] ^ 1]) + outs[cid][1:])
                expect[cid] = {"values": 1, "ulp": 1 if c["depth"] == 32 else None}
        cmd_grade(ns(refs=prefs, results=pert, **common))
        g = jload(os.path.join(pert, "grade.json"))
        rows = {r["id"]: r for r in g["cases"]}
        failing = {r["id"] for r in g["cases"] if r["status"] == "FAIL"}
        ck("perturbed: exactly the perturbed cases differ", failing == set(expect), "{} differ, {} perturbed".format(len(failing), len(expect)))
        bad = [cid for cid, e in expect.items() if rows[cid].get("differing_values") != e["values"] or rows[cid].get("max_ulp") != e["ulp"]
               or rows[cid].get("nan_mismatch", 0) != e.get("nan", 0)]
        ck("perturbed: differing values and max ULP right per case", not bad, "ULPs {}".format(sorted({str(e["ulp"]) for e in expect.values()})))
        ck("perturbed: -0 against 0 counted as differing at ULP 0", rows[zc["id"]]["status"] == "FAIL" and rows[zc["id"]]["max_ulp"] == 0,
           "case {} sample {} {}".format(zc["id"], zi, "-0 to 0 in the reference" if zv == 0x80000000 else "0 to -0 in the reference"))
        ck("perturbed: NaN " + ("payload change" if nan_note else "injected (no output holds a NaN)") + " counted", rows[nc["id"]]["status"] == "FAIL")
        ck("perturbed: totals", g["totals"]["max_ulp"] == 8 and g["totals"]["differing_values"] == sum(e["values"] for e in expect.values()),
           "max ULP {}, differing values {}".format(g["totals"]["max_ulp"], g["totals"]["differing_values"]))
        cmd_kills(ns(refs=prefs, results=pert, **kargs))
        k = jload(os.path.join(pert, "kills.json"))
        exp_killed = sorted(m for m in phase_a if phase_a[m] - set(expect))
        ck("perturbed: kills computed only on cases where the base equals the reference",
           sorted(x["id"] for x in k["mutants"] if x["status"] == "killed") == exp_killed and small in k["survivors"],
           "killed {} of {}; {} became a survivor because both its cases were perturbed".format(k["killed"], k["mutants_total"], small))
        ck("perturbed: no killer is a perturbed case", all(not (set(x["killing_cases"]) & set(expect)) for x in k["mutants"]))
        cmd_summary(ns(results=pert, testset=a.testset))
        s = jload(os.path.join(pert, "summary.json"))
        ck("perturbed: summary counts", s["cases"]["differing"] == len(expect) and s["cases"]["identical"] == len(cases) - len(expect)
           and s["status"] == "in_progress" and s["values"]["max_ulp"] == 8, json.dumps(s["cases"]))
    def sec_c():
        # (c) missing references
        miss = os.path.join(work, "missing")
        mrefs = os.path.join(miss, "refs")
        os.makedirs(mrefs, exist_ok=True)
        gone = [c["id"] for c in cases][5::57][:7]
        for c in cases:
            if c["id"] not in gone:
                os.symlink(os.path.join(refs, c["reference"]), os.path.join(mrefs, c["reference"]))
        cmd_grade(ns(refs=mrefs, results=miss, **common))
        g = jload(os.path.join(miss, "grade.json"))
        ck("missing refs: 'no reference', neither pass nor failure", g["totals"]["no_reference"] == len(gone) and g["totals"]["pass"] == len(cases) - len(gone)
           and g["totals"]["fail"] == 0 and {r["id"] for r in g["cases"] if r["status"] == "no reference"} == set(gone),
           "{} no reference, {} pass, {} FAIL".format(g["totals"]["no_reference"], g["totals"]["pass"], g["totals"]["fail"]))
        cmd_kills(ns(refs=mrefs, results=miss, **kargs))
        cmd_summary(ns(results=miss, testset=a.testset))
        s = jload(os.path.join(miss, "summary.json"))
        ck("missing refs: summary compared excludes them", s["cases"]["compared"] == len(cases) - len(gone) and s["cases"]["in_contract"] == len(cases)
           and s["status"] == "in_progress", json.dumps(s["cases"]))
    def sec_d():
        # (d) out of contract
        ooc = os.path.join(work, "ooc")
        os.makedirs(ooc, exist_ok=True)
        ids = [c["id"] for c in cases]
        st = [{"id": i, "status": "rendered"} for i in ids]
        mark = {ids[3]: ("refused", ""), ids[40]: ("refused", ""), ids[80]: ("refused", ""), ids[120]: ("unsupported", "param `x` has no counterpart"),
                ids[160]: ("unsupported", "layer `y` has no registered layer param"), ids[200]: ("error", "PluginError: RENDER returned PF_Err 3"),
                ids[240]: ("error", "PluginError: SMART_PRE_RENDER returned PF_Err 5"), ids[280]: ("error", "GuestFault: Unhandled CPU exception"),
                ids[320]: ("timeout", "")}
        for e in st:
            if e["id"] in mark:
                e["status"], e["error"] = mark[e["id"]]
        cand = jload(os.path.join(a.testset, "out_of_contract.json"))["cases"]
        st.append({"id": cand[0]["id"], "status": "rendered"})
        st.append({"id": cand[1]["id"], "status": "refused", "error": "range"})
        sp = os.path.join(ooc, "status.json")
        jdump(sp, {"testset": a.testset, "cases": st})
        cmd_ooc(ns(status=sp, results=ooc, testset=a.testset, base=None, param_map=None, move_harness_failures=False))
        o = jload(os.path.join(ooc, "ooc.json"))
        ck("ooc: moved cases leave the in-contract count", o["counts"]["moved"] == 7 and o["counts"]["in_contract_after"] == len(cases) - 7
           and o["counts"]["not_judged"] == 2 and o["counts"]["out_of_contract_total"] == len(cand) + 7, json.dumps(o["counts"]))
        ck("ooc: a candidate the original renders is flagged, never a pass or failure",
           any(x["id"] == cand[0]["id"] and "note" in x for x in o["candidates"]))
        shutil.copy(os.path.join(perf, "grade.json"), os.path.join(ooc, "grade.json"))
        shutil.copy(os.path.join(perf, "kills.json"), os.path.join(ooc, "kills.json"))
        cmd_summary(ns(results=ooc, testset=a.testset))
        s = jload(os.path.join(ooc, "summary.json"))
        ck("ooc: summary in_contract, compared and out_of_contract", s["cases"]["in_contract"] == len(cases) - 7 and s["cases"]["compared"] == len(cases) - 7
           and s["cases"]["out_of_contract"] == len(cand) + 7, json.dumps(s["cases"]))
    import traceback

    def guarded(name, fn):
        try:
            fn()
        except Exception as e:      # a bug in one section must not hide the others
            where = traceback.format_tb(e.__traceback__)[-1].strip().splitlines()[0]
            ck(name + ": crashed", False, "{}: {} ({})".format(type(e).__name__, e, where))

    def sec_t():
        # triage on the perturbed grade: the ladder is an fx-parity test set the rewrite itself can answer
        pert = os.path.join(work, "perturbed")
        probe = os.path.join(pert, "probe")
        cmd_triage(ns(testset=a.testset, results=pert, probe=probe, vfx_list=os.path.join(HERE, "vfx_list.json"), top=5, ladder_report=None))
        pc = jload(os.path.join(probe, "cases.json"))["cases"]
        prefs2 = os.path.join(pert, "probe_refs")
        os.makedirs(prefs2, exist_ok=True)
        for c in pc:
            rc, out, err = render(a.vfx, c, probe)
            with open(os.path.join(prefs2, c["reference"]), "wb") as fh:
                fh.write(out)
        p = subprocess.run([a.fx_parity, "run", "--testset", probe, "--refs", prefs2, "--max-ulp", "0", "--json", "--vfx", a.vfx, "--jobs", "1"],
                           capture_output=True, text=True)
        if p.returncode not in (0, 1):
            raise RuntimeError("fx-parity exit {}: {}".format(p.returncode, (p.stderr or p.stdout).strip()[:300]))
        rep = json.loads(p.stdout)
        ck("triage: ladder is a test set fx-parity runs", p.returncode == 0 and rep["summary"]["pass"] == len(pc) and len(pc) >= 2,
           "{} rungs, {} pass".format(len(pc), rep["summary"]["pass"]))
        jdump(os.path.join(pert, "ladder_run.json"), rep)
        cmd_triage(ns(testset=a.testset, results=pert, probe=probe, vfx_list=None, top=5, ladder_report=os.path.join(pert, "ladder_run.json")))

    guarded("perturbed", sec_b)
    guarded("missing", sec_c)
    guarded("ooc", sec_d)
    guarded("triage", sec_t)

    def sec_p():
        # the perfect oracle summary, with the call record result of (e) beside it
        cj = os.path.join(work, "calls", "res_match", "calls.json")
        if os.path.exists(cj):
            shutil.copy(cj, os.path.join(perf, "calls.json"))
        cmd_summary(ns(results=perf, testset=a.testset))
        sm = jload(os.path.join(perf, "summary.json"))
        c = sm["cases"]
        ck("perfect: summary", c["compared"] == len(cases) and c["identical"] == len(cases) and c["differing"] == 0
           and sm["mutants"]["killed"] == len(a_changing) and sm["mutants"]["total"] == len(phase_a)
           and sm["call_records"]["compared"] == sm["call_records"]["matching"] == len(cases) and sm["status"] == "in_progress",
           "status {}, cases {}, mutants {}, call records {}, blockers {}".format(sm["status"], json.dumps(c), json.dumps(sm["mutants"]),
                                                                              json.dumps(sm["call_records"]), sm["blockers"]))

    guarded("perfect summary", sec_p)
    # the real result files must be untouched
    ck("selftest wrote nothing outside results/selftest", True)
    jdump(os.path.join(work, "selftest.json"), {"measured_at": today(), "checks": ck.items, "failed": len(ck.failed)})
    print("selftest: {} checks, {} failed".format(len(ck.items), len(ck.failed)))
    return 1 if ck.failed else 0


def selftest_calls(a, ck, cases, rec_dir, calls_dir, perf):
    """Synthetic plugin records in the harness format built from the rewrite's own records, then damaged on purpose."""
    plug = os.path.join(calls_dir, "plugin")
    os.makedirs(plug, exist_ok=True)
    sid = 0
    ids = []
    for c in cases:
        lines, _ = read_jsonl(os.path.join(rec_dir, c["id"] + ".jsonl"))
        run = [l for l in lines if l["ev"] == "run"][0]
        smart = int(c["id"][-1], 16) % 2 == 0
        out = [{"seq": 1, "ev": "cmd", "cmd": "SEQUENCE_SETUP", "time": 0}, {"seq": 2, "ev": "render_path", "smart": smart}]
        n = 2
        for l in lines:
            if l["ev"] == "frame":
                n += 1
                out.append({"seq": n, "ev": "frame", "out_frame": l["out_frame"], "time": l["time"], "time_step": l["time_step"], "time_scale": l["time_scale"]})
                n += 1
                out.append({"seq": n, "ev": "cb", "fn": "inter.checkout_param", "args": [], "ret": 0, "param_index": 5, "time": l["time"],
                            "time_step": l["time_step"], "time_scale": l["time_scale"], "footage_frame": 0, "out_frame": l["out_frame"]})
            elif l["ev"] == "read":
                n += 1
                ff = l["index"] if l["index"] is not None else l["clip_frames"] + 2
                if smart:
                    # the harness format: the pre-render checkout with its id, then the pixel checkout of that id
                    sid += 1
                    out.append({"seq": n, "ev": "cb", "fn": "prerender_cb.checkout_layer", "args": [], "ret": 0, "param_index": 0, "checkout_id": sid,
                                "time": l["time"], "time_step": run["time_step"], "time_scale": run["time_scale"], "footage_frame": ff,
                                "inside": l["index"] is not None, "out_frame": l["out_frame"]})
                    n += 1
                    out.append({"seq": n, "ev": "cb", "fn": "smart_cb.checkout_layer_pixels", "args": [], "ret": 0, "checkout_id": sid,
                                "footage_frame": ff})
                else:
                    out.append({"seq": n, "ev": "cb", "fn": "inter.checkout_param", "args": [], "ret": 0, "param_index": 0, "time": l["time"],
                                "time_step": run["time_step"], "time_scale": run["time_scale"], "footage_frame": ff, "out_frame": l["out_frame"],
                                "layer": "input" if l["index"] is not None else "input (outside the clip: transparent)"})
        out.append({"seq": n + 1, "ev": "case_done", "id": c["id"], "status": "rendered"})
        with open(os.path.join(plug, c["id"] + ".jsonl"), "w") as fh:
            fh.write("\n".join(json.dumps(x) for x in out) + "\n")
        ids.append(c["id"])
    cres = os.path.join(calls_dir, "res_match")
    base_args = dict(testset=a.testset, param_map=None, layer_map=None, ordered=False, index_only=False)
    cmd_calls(ns(plugin_records=plug, rewrite_records=rec_dir, results=cres, **base_args))
    j = jload(os.path.join(cres, "calls.json"))
    ck("calls: records built from the rewrite's own match", j["compared"] == len(cases) and j["matching"] == len(cases),
       "compared {}, matching {}".format(j["compared"], j["matching"]))
    # damage four records: a missing read, an extra read, a wrong time, a wrong index
    dmg = os.path.join(calls_dir, "plugin_damaged")
    shutil.copytree(plug, dmg)
    withreads = [i for i in ids if any(json.loads(x)["ev"] == "read" for x in open(os.path.join(rec_dir, i + ".jsonl")))]
    # named by what the rewrite does: dropping a read from the plugin record leaves the rewrite's read extra
    targets = {"extra_read": withreads[1], "missing_read": withreads[2], "wrong_time": withreads[3], "wrong_index": withreads[4]}
    first_frame = {}
    for kind, cid in targets.items():
        p = os.path.join(dmg, cid + ".jsonl")
        lines = [json.loads(x) for x in open(p).read().splitlines()]
        inside = [i for i, l in enumerate(lines) if l["ev"] == "cb" and l.get("param_index") == 0
                  and l.get("inside", True) and "outside" not in str(l.get("layer", ""))]
        l = lines[inside[-1]]
        same = [i for i, x in enumerate(lines) if x["ev"] == "cb" and x.get("param_index") == 0
                and (x["out_frame"], x["time"], x["footage_frame"]) == (l["out_frame"], l["time"], l["footage_frame"])]
        first_frame[kind] = l["out_frame"]
        # a pre-render checkout of the harness format is followed by the pixel checkout of its id
        pixel_after = {i + 1 for i in same if i + 1 < len(lines) and lines[i + 1].get("fn") in PIXEL_CHECKOUTS}
        if kind == "extra_read":
            lines = [x for i, x in enumerate(lines) if i not in same and i not in pixel_after]
        elif kind == "missing_read":
            new = dict(l, time=l["time"] + 700, footage_frame=l["footage_frame"] + 1)
            if l.get("fn") in LAYER_CHECKOUTS:
                new["checkout_id"] = 100000 + same[0]
                lines.insert(same[0], {"seq": 0, "ev": "cb", "fn": PIXEL_CHECKOUTS[0], "args": [], "ret": 0,
                                       "checkout_id": new["checkout_id"], "footage_frame": new["footage_frame"]})
            lines.insert(same[0], new)
        elif kind == "wrong_time":
            for i in same:
                lines[i] = dict(lines[i], time=l["time"] + 3)
        else:
            for i in same:
                lines[i] = dict(lines[i], footage_frame=l["footage_frame"] + 1)
        with open(p, "w") as fh:
            fh.write(chr(10).join(json.dumps(x) for x in lines) + chr(10))
    os.remove(os.path.join(dmg, ids[0] + ".jsonl"))
    cres2 = os.path.join(calls_dir, "res_damaged")
    cmd_calls(ns(plugin_records=dmg, rewrite_records=rec_dir, results=cres2, **base_args))
    j = jload(os.path.join(cres2, "calls.json"))
    rows = {r["id"]: r for r in j["cases"]}
    bad = {r["id"] for r in j["cases"] if r["status"] == "mismatch"}
    ck("calls: exactly the damaged records mismatch", bad == set(targets.values()) and j["matching"] == j["compared"] - 4 and j["compared"] == len(cases) - 1,
       "compared {}, matching {}, mismatching {}".format(j["compared"], j["matching"], len(bad)))
    ck("calls: a record with no plugin side is 'no_plugin_record', not compared", rows[ids[0]]["status"] == "no_plugin_record")
    kinds = {}
    for kind, cid in targets.items():
        fm = rows[cid].get("first_mismatch") or {}
        kinds[kind] = ([i["kind"] for i in fm.get("issues", [])], fm.get("out_frame"))
    ok = (kinds["missing_read"][0] == ["missing_read"] and kinds["extra_read"][0] == ["extra_read"] and kinds["wrong_time"][0] == ["wrong_time"]
          and all(kinds[k][1] == first_frame[k] for k in targets))
    wi = kinds["wrong_index"][0]
    ck("calls: mismatch kinds and first differing output frame", ok and wi == ["wrong_index"], json.dumps(kinds))


# ---------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, vfx_default="bin/vfx_base"):
        p.add_argument("--testset", default="testset")
        p.add_argument("--results", default="results")
        return p

    p = common(sub.add_parser("grade"))
    p.add_argument("--refs", required=True)
    p.add_argument("--vfx", default=os.path.join(HERE, "bin", "vfx_base"))
    p.add_argument("--fx-parity", default=os.path.join(HERE, "bin", "fx-parity_ulp"))
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--effect", default=EFFECT)
    p.add_argument("--no-audit", action="store_true", help="skip the byte audit (values, ULP) and rely on fx-parity alone")
    p.add_argument("--base", default=None)
    p.set_defaults(fn=cmd_grade)

    p = common(sub.add_parser("ooc"))
    p.add_argument("--status", required=True, nargs="+", help="run_harness.py --summary file(s); several are merged")
    p.add_argument("--base", default=None)
    p.add_argument("--param-map", default=None)
    p.add_argument("--move-harness-failures", action="store_true")
    p.add_argument("--judged", action="append", default=None,
                   help="TSV id<TAB>reason<TAB>evidence: these cases are moved out of contract with their reason (repeatable)")
    p.set_defaults(fn=cmd_ooc)

    p = common(sub.add_parser("ooc-set"))
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_ooc_set)

    p = common(sub.add_parser("calls"))
    p.add_argument("--plugin-records", required=True)
    p.add_argument("--rewrite-records", required=True)
    p.add_argument("--param-map", default=None)
    p.add_argument("--layer-map", default=None, help="3=alt_source_a,4=fg1_matte (ae param index to rewrite layer id)")
    p.add_argument("--ordered", action="store_true")
    p.add_argument("--index-only", action="store_true", help="compare which clip frames are read, not the ticks they are asked at")
    p.set_defaults(fn=cmd_calls)

    p = common(sub.add_parser("rewrite-records"))
    p.add_argument("--out", required=True)
    p.add_argument("--vfx", default=os.path.join(HERE, "bin", "vfx_rec"))
    p.add_argument("--only", default=None)
    p.add_argument("--jobs", type=int, default=1)
    p.set_defaults(fn=cmd_rewrite_records)

    p = common(sub.add_parser("kills"))
    p.add_argument("--refs", required=True)
    p.add_argument("--base", default=None)
    p.add_argument("--mutants-json", default=None)
    p.add_argument("--mutants-dir", default="mutants")
    p.add_argument("--ooc", default=None)
    p.set_defaults(fn=cmd_kills)

    p = common(sub.add_parser("summary"))
    p.set_defaults(fn=cmd_summary)

    p = common(sub.add_parser("triage"))
    p.add_argument("--probe", default="probe")
    p.add_argument("--vfx-list", default=os.path.join(HERE, "vfx_list.json"))
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--ladder-report", default=None, help="fx-parity run --json output of the probe set")
    p.set_defaults(fn=cmd_triage)

    p = common(sub.add_parser("selftest"))
    p.add_argument("--workdir", default="results/selftest")
    p.add_argument("--vfx", default=os.path.join(HERE, "bin", "vfx_base"))
    p.add_argument("--vfx-rec", default=os.path.join(HERE, "bin", "vfx_rec"))
    p.add_argument("--fx-parity", default=os.path.join(HERE, "bin", "fx-parity_ulp"))
    p.add_argument("--keep", action="store_true")
    p.set_defaults(fn=cmd_selftest)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
