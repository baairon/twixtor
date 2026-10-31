#!/usr/bin/env python3
"""Phase B tooling of the retime-flow parity module (Python 3 standard library only).

Everything that happens once the emulator has written reference renders (`refs/<case id>.<pix_fmt>`),
call records and per-case statuses. Nothing here runs a plugin.

    phase_b.py grade   --refs DIR                       bit exact grading of both test sets (fx-parity run --max-ulp 0)
    phase_b.py ooc     --status FILE [--status FILE]    cases the original refuses, moved out of contract
    phase_b.py calls   --plugin-records DIR --rewrite-records DIR
    phase_b.py records --vfx BIN --out DIR              rewrite side: VFX_CALL_RECORD over the whole test set
    phase_b.py kills   --refs DIR                       mutation kills against the reference renders
    phase_b.py summary                                  results/summary.json, every count computed from result files
    phase_b.py triage                                   group the differing cases, write a ladder of probe cases
    phase_b.py selftest                                 stand-in oracles, writes only under results/selftest/

Layout: the folder this file sits in (or the folder above `tools/`) is the module root; the server copy is
~/twixtor-parity/retime-flow with `tools/phase_b.py`. `--results` is where results are written (default
<root>/results); `--phase-a` is where the phase A tables are read from (default <root>/results);
`--testsets` holds `grid/` and `draft/` (default <root>/testsets, else the `cases/` copies for metadata only).
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from array import array
from collections import Counter, OrderedDict, defaultdict
from fractions import Fraction
from pathlib import Path

BPP = {8: 4, 16: 8, 32: 16}
PIX = {8: "rgba", 16: "rgba64le", 32: "gbrapf32le"}
RGBA = ("r", "g", "b", "a")
PLANES = ("g", "b", "r", "a")
SETS = ("grid", "draft")
MIN_CASES = 150
FALLBACK_DEFAULTS = {
    "speed_percent": "100", "vectors": "5", "frame_interp": "2", "sensitivity": "70", "smart_blend": "0",
    "video_360": "0", "prep": "1", "blur_compensation": "0",
}


class ToolError(Exception):
    """A setup or input problem: printed as one line, exit code 3."""


# ------------------------------------------------------------------------------------------ small helpers


def root_dir():
    here = Path(__file__).resolve().parent
    return here.parent if here.name == "tools" else here


def jload(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def jdump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, indent=1) + "\n")


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def command_text():
    return "python3 tools/phase_b.py " + " ".join(shlex.quote(x) for x in sys.argv[1:])


def as_cmd(text):
    """A program given on the command line: one path, or a command prefix with arguments."""
    if isinstance(text, (list, tuple)):
        return [str(x) for x in text]
    parts = shlex.split(text, posix=(os.name != "nt"))
    return [x[1:-1] if len(x) > 1 and x[0] == x[-1] == '"' else x for x in parts]


def today():
    return datetime.date.today().isoformat()


class Ctx:
    def __init__(self, a):
        self.root = Path(a.root).resolve() if getattr(a, "root", None) else root_dir()
        self.results = Path(a.results) if getattr(a, "results", None) else self.root / "results"
        self.phase_a = Path(a.phase_a) if getattr(a, "phase_a", None) else self.root / "results"
        self.testsets = Path(a.testsets) if getattr(a, "testsets", None) else self.root / "testsets"
        self.explicit = list(getattr(a, "testset", None) or [])
        self.command = getattr(a, "command", None) or command_text()


def set_dirs(ctx):
    if ctx.explicit:
        return [(Path(p).name, Path(p)) for p in ctx.explicit]
    return [(n, ctx.testsets / n) for n in SETS]


def load_cases(ctx, metadata_only=False):
    """Every case of the test sets in manifest order, each with `_set` and `_dir` added."""
    out = OrderedDict()
    for name, d in set_dirs(ctx):
        f = d / "cases.json"
        if not f.exists() and metadata_only and not ctx.explicit:
            alt = ctx.root / "cases" / name / "cases.json"
            if alt.exists():
                f = alt
        if not f.exists():
            raise ToolError(f"no test set at {d} (missing {f})")
        for c in jload(f)["cases"]:
            c = dict(c)
            c["_set"] = name
            c["_dir"] = str(d)
            if c["id"] in out:
                raise ToolError(f"duplicate case id {c['id']}")
            out[c["id"]] = c
    return out


def param_dict(c):
    out = OrderedDict()
    for p in c.get("params") or []:
        k, _, v = p.partition("=")
        out[k] = v
    return out


def case_tags(c):
    t = [f"group:{c.get('group', '')}", f"depth:{c['depth']}", f"size:{c['width']}x{c['height']}", f"frames:{c['frames']}",
         f"fps:{c.get('fps', '')}", f"src:{c.get('source', '')}", f"quality:{c.get('quality') or 'default'}"]
    t += [f"flag:{f}" for f in c.get("flags") or []]
    if c.get("stand_in"):
        t.append(f"stand_in:{c['stand_in']}")
    t += [f"param:{k}={v}" for k, v in param_dict(c).items()]
    return t


def case_view(c):
    """The metadata of a case that the result files carry."""
    keep = ("id", "group", "label", "params", "depth", "width", "height", "frames", "fps", "source", "quality", "flags",
            "stand_in", "input", "reference", "effect", "layers", "ladder")
    v = {k: c[k] for k in keep if k in c}
    v["set"] = c["_set"]
    v["testset_dir"] = c["_dir"]
    return v


def bytes_per_value(depth):
    return 1 if depth == 8 else 2 if depth == 16 else 4


def case_value_counts(c):
    """(bytes, floats) one case contributes when compared: every channel of every pixel of every frame."""
    n = c["width"] * c["height"] * c["frames"] * 4
    return (n * bytes_per_value(c["depth"]), 0) if c["depth"] < 32 else (0, n)


# ------------------------------------------------------------------------------------------ running vfx


def vfx_args(c, testset_dir, quality=True):
    a = ["apply", c["effect"], "--width", str(c["width"]), "--height", str(c["height"]), "--depth", str(c["depth"])]
    for p in c.get("params") or []:
        a += ["--param", p]
    a += ["--fps", str(c["fps"]), "--frames", str(c["frames"])]
    if quality and c.get("quality"):
        a += ["--quality", str(c["quality"])]
    for lid, f in sorted((c.get("layers") or {}).items()):
        a += ["--layer", f"{lid}={os.path.join(testset_dir, f)}"]
    a += ["--in", os.path.join(testset_dir, c["input"])]
    return a


def render(vfx, c, testset_dir, record=None):
    """Renders one case with the rewrite. -> (exit code, stdout bytes, first stderr line)."""
    env = dict(os.environ)
    env.pop("VFX_CALL_RECORD", None)
    if record:
        env["VFX_CALL_RECORD"] = str(record)
    p = subprocess.run(as_cmd(vfx) + vfx_args(c, testset_dir), stdin=subprocess.DEVNULL, capture_output=True, env=env)
    err = p.stderr.decode("utf-8", "replace").strip().splitlines()
    return p.returncode, p.stdout, (err[0][:200] if err else "")


def make_wrapper(path, vfx, quality):
    """A script that runs `vfx` with `--quality Q` appended (fx-parity has no quality flag)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = " ".join(shlex.quote(x) for x in as_cmd(vfx))
    path.write_text(f"#!/bin/bash\nexec {cmd} \"$@\" --quality {shlex.quote(quality)}\n", newline="\n")
    path.chmod(0o755)
    return str(path)


def find_exe(explicit, candidates, what):
    if explicit:
        return as_cmd(explicit)
    for c in candidates:
        if c and Path(c).exists():
            return [str(c)]
    found = shutil.which(Path(candidates[-1]).name) if candidates else None
    if found:
        return [found]
    raise ToolError(f"{what} not found: pass it explicitly (tried {', '.join(str(c) for c in candidates if c)})")


def fxp_supports_max_ulp(cmd):
    """True when this fx-parity build knows `run --max-ulp` (older builds refuse the flag)."""
    try:
        p = subprocess.run(as_cmd(cmd) + ["--help"], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
        return "--max-ulp" in (p.stdout + p.stderr)
    except Exception:  # noqa: BLE001
        return False


def find_fxp(ctx, explicit):
    """An fx-parity build with --max-ulp: the explicit one, $FX_PARITY, or the first known location that has it."""
    home = Path.home()
    given = explicit or os.environ.get("FX_PARITY")
    if given:
        cmd = as_cmd(given)
        if not fxp_supports_max_ulp(cmd):
            raise ToolError(f"{given} does not know `run --max-ulp` (build the current fx-parity, or pass --fx-parity)")
        return cmd
    tried = []
    for c in (ctx.root / "bin/fx-parity", ctx.root / "src/target/release/fx-parity",
              home / "twixtor-parity/harness/repo/target/release/fx-parity",
              home / "twixtor-parity/src/base/target/release/fx-parity", shutil.which("fx-parity")):
        if c and Path(c).exists():
            tried.append(str(c))
            if fxp_supports_max_ulp([str(c)]):
                return [str(c)]
    raise ToolError("no fx-parity with `run --max-ulp` found (tried: " + (", ".join(tried) or "nothing") + "); pass --fx-parity or set FX_PARITY")


def find_vfx(ctx, explicit):
    return find_exe(explicit or os.environ.get("VFX_PATH"), [ctx.root / "bin/vfx-base", ctx.root / "src/target/release/vfx", "vfx"], "vfx")


def parse_json_text(text):
    i = text.find("{")
    if i < 0:
        raise ToolError("no JSON object on fx-parity's stdout")
    return json.loads(text[i:])


# ------------------------------------------------------------------------------------------ exact comparison


def f32_ordered(bits):
    return -(bits & 0x7FFFFFFF) if bits & 0x80000000 else bits


def f32_is_nan(bits):
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def f32_text(bits):
    import struct
    return f"0x{bits:08x} ({struct.unpack('<f', struct.pack('<I', bits))[0]!r})"


def _values(depth, buf):
    if depth == 8:
        return buf
    arr = array("H" if depth == 16 else "I")
    arr.frombytes(buf)
    if sys.byteorder == "big":
        arr.byteswap()
    return arr


def diff_clip(depth, width, height, frames, ref, out):
    """Value by value comparison of two clips in the rewrite's wire format (8 bpc rgba, 16 bpc rgba64le,
    32 bpc planar gbrapf32le). -0 against 0 and differing NaN bits count as different. Returns
    differing_values, differing_pixels, failing_frames, first (the first differing value), max_ulp (the
    largest finite float distance over the differing values; None when none), nan_mismatch,
    signed_zero_diffs and per_channel counts. A length mismatch gives {"size_mismatch": [ref, out]}."""
    if len(ref) != len(out) or len(ref) != frames * width * height * BPP[depth]:
        return {"size_mismatch": [len(ref), len(out)], "expected_bytes": frames * width * height * BPP[depth]}
    n = width * height
    flen = n * BPP[depth]
    res = {"differing_values": 0, "differing_pixels": 0, "failing_frames": 0, "first": None, "max_ulp": None,
           "nan_mismatch": 0, "signed_zero_diffs": 0, "per_channel": {}}
    per = Counter()
    for f in range(frames):
        a, b = ref[f * flen:(f + 1) * flen], out[f * flen:(f + 1) * flen]
        if a == b:
            continue
        res["failing_frames"] += 1
        va, vb = _values(depth, a), _values(depth, b)
        pix = set()
        for i, (x, y) in enumerate(zip(va, vb)):
            if x == y:
                continue
            if depth == 32:
                plane, p = divmod(i, n)
                ch = PLANES[plane]
                if f32_is_nan(x) or f32_is_nan(y):
                    res["nan_mismatch"] += 1
                else:
                    if (x & 0x7FFFFFFF) == 0 and (y & 0x7FFFFFFF) == 0:
                        res["signed_zero_diffs"] += 1
                    d = abs(f32_ordered(x) - f32_ordered(y))
                    res["max_ulp"] = d if res["max_ulp"] is None else max(res["max_ulp"], d)
            else:
                p, c4 = divmod(i, 4)
                ch = RGBA[c4]
            per[ch] += 1
            pix.add(p)
            res["differing_values"] += 1
            if res["first"] is None:
                res["first"] = {"frame": f, "x": p % width, "y": p // width, "channel": ch,
                                "expected": f32_text(x) if depth == 32 else x, "actual": f32_text(y) if depth == 32 else y}
        res["differing_pixels"] += len(pix)
    res["per_channel"] = dict(sorted(per.items()))
    return res


# ------------------------------------------------------------------------------------------ grade


def split_by_quality(ts_dir, tmp):
    """[(quality, directory)]: fx-parity has no quality flag, so a test set mixing qualities is split into
    one temporary test set per quality (the inputs folder is linked)."""
    doc = jload(Path(ts_dir) / "cases.json")
    quals = OrderedDict()
    for c in doc["cases"]:
        quals.setdefault(c.get("quality"), []).append(c)
    if len(quals) <= 1:
        return [(next(iter(quals), None), Path(ts_dir))]
    out = []
    for q, cs in quals.items():
        d = Path(tmp) / f"split-{q or 'default'}-{Path(ts_dir).name}"
        d.mkdir(parents=True, exist_ok=True)
        jdump(d / "cases.json", dict(doc, cases=cs))
        link = d / "inputs"
        if not link.exists():
            try:
                os.symlink(Path(ts_dir).resolve() / "inputs", link)
            except (OSError, NotImplementedError):
                shutil.copytree(Path(ts_dir) / "inputs", link)
        out.append((q, d))
    return out


def vfx_for_quality(quality, vfx, draft_vfx, tmp):
    cmd = as_cmd(vfx)
    if not quality:
        return cmd[0] if len(cmd) == 1 else _single(vfx, tmp)
    if quality == "draft" and draft_vfx:
        return as_cmd(draft_vfx)[0]
    return make_wrapper(Path(tmp) / f"vfx-{quality}", vfx, quality)


def _single(vfx, tmp):
    """fx-parity takes one path; a command prefix is put in a script."""
    cmd = " ".join(shlex.quote(x) for x in as_cmd(vfx))
    p = Path(tmp) / "vfx-plain"
    p.write_text(f"#!/bin/bash\nexec {cmd} \"$@\"\n", newline="\n")
    p.chmod(0o755)
    return str(p)


def run_fxp(fxp, ts_dir, refs, vfx_path, jobs=1):
    cmd = as_cmd(fxp) + ["run", "--testset", str(ts_dir), "--refs", str(refs), "--max-ulp", "0", "--json", "--jobs", str(jobs),
                         "--vfx", vfx_path]
    p = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode not in (0, 1):
        raise ToolError(f"fx-parity exited {p.returncode}: {(p.stderr or p.stdout).strip()[:300]}")
    return parse_json_text(p.stdout), " ".join(shlex.quote(x) for x in cmd)


def grade_status(word):
    return {"pass": "identical", "FAIL": "differs", "no reference": "no_reference"}.get(word, "not_graded")


def analyse_case(vfx, c, ref_path):
    """The exact comparison of one differing case. -> dict of diff_clip plus rewrite exit and error."""
    rc, out, err = render(vfx, c, c["_dir"])
    if rc != 0:
        return {"rewrite_exit": rc, "rewrite_error": err}
    ref = Path(ref_path).read_bytes()
    res = diff_clip(c["depth"], c["width"], c["height"], c["frames"], ref, out)
    res["rewrite_exit"] = 0
    return res


def grade(ctx, refs, vfx=None, fxp=None, draft_vfx=None, jobs=1, exact=True, log=print):
    refs = Path(refs)
    if not refs.is_dir():
        raise ToolError(f"the references folder {refs} does not exist")
    fxp = find_fxp(ctx, fxp) if not isinstance(fxp, list) else fxp
    vfx = find_vfx(ctx, vfx) if not isinstance(vfx, list) else vfx
    cases = load_cases(ctx)
    commands, fxp_results = [], {}
    with tempfile.TemporaryDirectory(prefix="phase_b_") as tmp:
        for name, d in set_dirs(ctx):
            if not jload(Path(d) / "cases.json")["cases"]:
                continue
            for q, qdir in split_by_quality(d, tmp):
                path = vfx_for_quality(q, vfx, draft_vfx, tmp)
                report, cmd = run_fxp(fxp, qdir, refs, path, jobs)
                commands.append(cmd)
                for r in report["cases"]:
                    fxp_results[r["id"]] = r
                log(f"{name} ({q or 'default quality'}): {report['summary']}")
    rows = []
    for cid, c in cases.items():
        r = fxp_results.get(cid)
        st = grade_status(r["status"]) if r else "not_graded"
        row = case_view(c)
        row.update(status=st, tags=case_tags(c), logic_parity=bool(c.get("stand_in")) and st in ("identical", "differs"))
        compared = st in ("identical", "differs")
        row["compared"] = compared
        if compared:
            row["bytes_compared"], row["floats_compared"] = case_value_counts(c)
        if r:
            if r.get("error"):
                row["fx_parity_error"] = r["error"]
            if "clip_differing_pixels" in r:
                row["fx_parity_differing_pixels"] = r["clip_differing_pixels"]
        row.update(differing_values=0, differing_pixels=0, failing_frames=0, max_ulp=0 if (st == "identical" and c["depth"] == 32) else None,
                   nan_mismatch=0)
        if st == "differs":
            row["reference_path"] = str(refs / c["reference"])
            if exact:
                a = analyse_case(vfx, c, refs / c["reference"])
                if "differing_values" in a:
                    row.update({k: a[k] for k in ("differing_values", "differing_pixels", "failing_frames", "nan_mismatch",
                                                   "signed_zero_diffs", "per_channel", "first") if k in a})
                    row["max_ulp"] = a["max_ulp"] if c["depth"] == 32 else None
                    if "fx_parity_differing_pixels" in row and row["fx_parity_differing_pixels"] != a["differing_pixels"]:
                        row["analysis_disagrees_with_fx_parity"] = True
                else:
                    row.update({k: v for k, v in a.items()})
                    row["differing_values"] = None
            else:
                row["differing_values"] = None
        rows.append(row)
    return rows, commands


def grade_totals(rows):
    comp = [r for r in rows if r["compared"]]
    diff = [r for r in comp if r["status"] == "differs"]
    vals = [r["differing_values"] for r in diff]
    ulps = [r["max_ulp"] for r in comp if r.get("max_ulp") is not None]
    return {
        "cases": len(rows), "compared": len(comp), "identical": len(comp) - len(diff), "differing": len(diff),
        "no_reference": sum(1 for r in rows if r["status"] == "no_reference"),
        "not_graded": sum(1 for r in rows if r["status"] == "not_graded"),
        "logic_parity_compared": sum(1 for r in comp if r["logic_parity"]),
        "bytes_compared": sum(r.get("bytes_compared", 0) for r in comp),
        "floats_compared": sum(r.get("floats_compared", 0) for r in comp),
        "differing_values": sum(v for v in vals if v is not None),
        "differing_values_unknown_cases": sum(1 for v in vals if v is None),
        "differing_pixels": sum(r["differing_pixels"] for r in diff),
        "max_ulp": max(ulps) if ulps else None,
        "nan_mismatch_values": sum(r.get("nan_mismatch", 0) for r in diff),
    }


def write_grade(ctx, rows, commands, refs):
    doc = {"refs": str(refs), "measured_at": today(), "bar": "fx-parity run --max-ulp 0 (bit for bit; -0 against 0 and NaN bits count)",
           "command": ctx.command, "fx_parity_commands": commands, "totals": grade_totals(rows), "cases": rows}
    jdump(ctx.results / "grade.json", doc)
    with open(ctx.results / "grade.tsv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("id\tstatus\tdiffering_values\tmax_ulp\ttags\tlogic_parity\n")
        for r in rows:
            dv = "" if r["differing_values"] is None else r["differing_values"]
            fh.write("\t".join(str(x) for x in (r["id"], r["status"], dv, "" if r["max_ulp"] is None else r["max_ulp"],
                                                ";".join(r["tags"]), 1 if r["logic_parity"] else 0)) + "\n")
    return doc


# ------------------------------------------------------------------------------------------ out of contract

RENDERED, REFUSED, ERROR, TIMEOUT, NOT_RUN = "rendered", "refused", "error", "timeout", "not_run"
STATUS_WORDS = {
    "rendered": RENDERED, "ok": RENDERED, "pass": RENDERED, "done": RENDERED,
    "refused": REFUSED, "unsupported": REFUSED, "rejected": REFUSED, "declined": REFUSED,
    "error": ERROR, "fault": ERROR, "crash": ERROR, "failed": ERROR, "exception": ERROR,
    "timeout": TIMEOUT, "timed_out": TIMEOUT,
    "not_run": NOT_RUN, "skipped": NOT_RUN, "pending": NOT_RUN, "": NOT_RUN,
}
HARNESS_FAULT_MARKERS = ("GuestFault", "worker died", "did not boot", "too many worker restarts")
HARNESS_LIMIT_MARKERS = ("bezier", "keyframe interpol", "stand-in", "not implemented")


def read_status_file(path):
    """Per-case status of the harness: the `run_harness.py --summary` JSON ({"cases": [...]}), a JSON object
    keyed by case id, a JSON list, or a TSV with id, status and an optional detail column. -> {id: record}."""
    text = Path(path).read_text(encoding="utf-8")
    out = {}
    if text.lstrip().startswith(("{", "[")):
        doc = json.loads(text)
        if isinstance(doc, dict) and "cases" in doc:
            doc = doc["cases"]
        if isinstance(doc, dict):
            items = [dict(v, id=k) if isinstance(v, dict) else {"id": k, "status": str(v)} for k, v in doc.items()]
        else:
            items = doc
        for it in items:
            out[it["id"]] = it
        return out
    lines = [l for l in text.splitlines() if l.strip()]
    head = lines[0].split("\t")
    has_head = "id" in head and "status" in head
    for l in lines[1:] if has_head else lines:
        f = l.split("\t")
        rec = dict(zip(head, f)) if has_head else {"id": f[0], "status": f[1] if len(f) > 1 else "", "error": f[2] if len(f) > 2 else ""}
        out[rec["id"]] = rec
    return out


def classify_status(rec):
    """-> (class, harness_fault). class is one of rendered, refused, error, timeout, not_run, unknown. A
    harness fault is an emulator crash or a stand-in limit of the harness, not the plugin refusing the case."""
    word = str(rec.get("status", "")).strip().lower()
    cls = STATUS_WORDS.get(word, "unknown")
    text = " ".join(str(rec.get(k, "")) for k in ("error", "detail"))
    low = text.lower()
    fault = cls in (ERROR, NOT_RUN, REFUSED) and (any(m in text for m in HARNESS_FAULT_MARKERS)
                                                  or any(m in low for m in HARNESS_LIMIT_MARKERS))
    return cls, fault


def parse_key_values(text):
    """The numbers in a parameter value: `50`, `100@0;40@5`, `100@0h;40@5e`."""
    out = []
    for part in text.split(";"):
        v = part.split("@")[0].strip()
        try:
            out.append(float(v))
        except ValueError:
            return []
    return out


def range_candidates(cases, param_map, plugin):
    """In-contract cases with a numeric parameter value outside the [min, max] the plugin registers (sliders only)."""
    rows = [r for r in param_map if r.get("plugin") == plugin and r.get("match") in ("exact", "normalized") and r.get("vfx_id")]
    by_id = {}
    for r in rows:
        if r.get("ae_type", "").upper() in ("FLOAT_SLIDER", "SLIDER", "FIX_SLIDER", "ANGLE", "FLOAT") and r.get("ae_min", "") != "" and r.get("ae_max", "") != "":
            try:
                by_id[r["vfx_id"]] = (float(r["ae_min"]), float(r["ae_max"]), r["ae_name"])
            except ValueError:
                pass
    out = []
    for cid, c in cases.items():
        for k, v in param_dict(c).items():
            if k in by_id:
                lo, hi, name = by_id[k]
                bad = [x for x in parse_key_values(v) if not (lo <= x <= hi)]
                if bad:
                    out.append({"id": cid, "param": k, "value": v, "registered_min": lo, "registered_max": hi, "ae_name": name})
                    break
    return out


def rewrite_side(base_rows, cid):
    r = base_rows.get(cid)
    if not r:
        return {"recorded": False}
    return {"recorded": True, "exit": int(r["exit"]), "bytes": int(r["bytes"]), "stderr": r.get("stderr", "")}


def load_base_rows(phase_a, name="base"):
    rows = {}
    for s in SETS:
        p = Path(phase_a) / f"{name}.{s}.tsv"
        if p.exists():
            for r in read_tsv(p):
                rows[r["id"]] = r
    return rows


def load_candidates(ctx):
    p = ctx.root / "cases" / "out_of_contract.json"
    for q in (ctx.testsets / "out_of_contract.json", p):
        if q.exists():
            cands = jload(q)
            break
    else:
        return []
    check = {}
    t = ctx.phase_a / "out_of_contract.tsv"
    if t.exists():
        check = {r["id"]: r for r in read_tsv(t)}
    out = []
    for o in cands:
        ck = check.get(o["id"], {})
        out.append({"id": o["id"], "label": o.get("label", ""), "why": o.get("why", ""), "params": o.get("params", []),
                    "rewrite": {"check_exit": int(ck["rewrite_check_exit"]) if ck.get("rewrite_check_exit", "") != "" else None,
                                "message": ck.get("message", "")}})
    return out


def ooc(ctx, status_files, faults_out_of_contract=False, param_map=None, plugin="Twixtor_64", apply_ranges=False):
    cases = load_cases(ctx, metadata_only=True)
    base_rows = load_base_rows(ctx.phase_a)
    statuses = {}
    for f in status_files:
        statuses.update(read_status_file(f))
    cands = {c["id"]: c for c in load_candidates(ctx)}
    moved, unjudged, accepted, unknown, cand_out = [], [], [], [], []
    for cid, rec in statuses.items():
        cls, fault = classify_status(rec)
        plugin_side = {k: rec[k] for k in ("status", "error", "last_api", "last_callback", "pc", "messages", "detail") if k in rec}
        if cid in cands:
            c = cands[cid]
            verdict = ("out_of_contract_confirmed" if cls in (REFUSED, ERROR, TIMEOUT) and not fault else
                       "plugin_renders_it" if cls == RENDERED else "unjudged")
            cand_out.append(dict(c, plugin=plugin_side, verdict=verdict))
        elif cid in cases:
            c = cases[cid]
            entry = {"id": cid, "group": c.get("group"), "params": c.get("params"), "depth": c["depth"], "plugin": plugin_side,
                     "rewrite": rewrite_side(base_rows, cid)}
            if cls in (REFUSED, ERROR, TIMEOUT) and (not fault or faults_out_of_contract):
                entry["reason"] = {REFUSED: "the original refused it", ERROR: "the original errored on it",
                                   TIMEOUT: "the original timed out on it"}[cls] + (" (harness fault, moved by request)" if fault else "")
                moved.append(entry)
            elif cls == RENDERED:
                accepted.append(cid)
            else:
                entry["reason"] = ("harness fault, not judged" if fault else "no outcome" if cls == NOT_RUN else f"unknown status {rec.get('status')!r}")
                unjudged.append(entry)
        else:
            unknown.append(cid)
    seen = {c["id"] for c in cand_out}
    for cid, c in cands.items():
        if cid not in seen:
            cand_out.append(dict(c, plugin=None, verdict="unjudged"))
    ranges = []
    if param_map:
        remaining = OrderedDict((i, c) for i, c in cases.items() if i not in {m["id"] for m in moved})
        ranges = range_candidates(remaining, read_tsv(param_map), plugin)
        if apply_ranges:
            for r in ranges:
                if r["id"] not in {m["id"] for m in moved}:
                    c = cases[r["id"]]
                    moved.append({"id": r["id"], "group": c.get("group"), "params": c.get("params"), "depth": c["depth"], "plugin": None,
                                  "rewrite": rewrite_side(base_rows, r["id"]),
                                  "reason": f"{r['param']}={r['value']} outside the registered range {r['registered_min']}..{r['registered_max']}"})
    still = [c for c in cand_out if c["verdict"] != "plugin_renders_it"]
    doc = {"command": ctx.command, "measured_at": today(), "status_files": [str(f) for f in status_files],
           "counts": {"cases_in_test_sets": len(cases), "moved_out_of_contract": len(moved), "unjudged": len(unjudged),
                      "accepted_by_the_original": len(accepted), "candidates": len(cand_out), "candidates_out_of_contract": len(still),
                      "candidates_the_original_renders": sum(1 for c in cand_out if c["verdict"] == "plugin_renders_it"),
                      "range_candidates": len(ranges), "unknown_ids": len(unknown)},
           "moved": moved, "unjudged": unjudged, "candidates": cand_out, "range_candidates": ranges, "unknown_ids": unknown}
    jdump(ctx.results / "ooc.json", doc)
    return doc


# ------------------------------------------------------------------------------------------ call records


def load_jsonl(path):
    out, bad = [], 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return out, bad


def layer_map_from_param_map(path, plugin):
    """{ae param index: rewrite layer id} for the layer params the plugin registers."""
    out = {}
    if path and Path(path).exists():
        for r in read_tsv(path):
            if r.get("plugin") == plugin and r.get("vfx_kind") == "layer" and r.get("ae_index", "") != "":
                out[int(r["ae_index"])] = r["vfx_id"]
    return out


def plugin_view(lines, layer_of_index, present=None):
    """The harness's records of one case: output frames, clip reads per output frame, parameter samples per frame.
    When the record has pixel checkouts (smart_cb.checkout_layer_pixels), the clip reads are those pixels, with the
    time and frame of the pre-render checkout of the same checkout_id (the other pre-render checkouts, the probes at the
    current time and at minus the current time, read nothing); `present` (layer names the case provides, "input"
    always) drops pixels of a layer the case does not provide."""
    frames, reads, psamples, incomplete = OrderedDict(), defaultdict(list), defaultdict(list), None
    cur = None
    pixel_mode = any(str(ev.get("fn", "")) == "smart_cb.checkout_layer_pixels" for ev in lines)
    ck = {}
    for ev in lines:
        e = ev.get("ev")
        if pixel_mode and e == "cb" and ev.get("fn") in ("prerender_cb.checkout_layer", "smart_cb.checkout_layer_pixels"):
            if ev["fn"] == "prerender_cb.checkout_layer":
                idx = ev.get("param_index")
                layer = "input" if idx == 0 else layer_of_index.get(idx, f"layer{idx}")
                ck[ev.get("checkout_id")] = {"layer": layer, "time": ev.get("time"), "frame": ev.get("footage_frame"),
                                             "inside": ev.get("inside")}
            else:
                r = ck.get(ev.get("checkout_id"))
                if r is not None and (present is None or r["layer"] in present):
                    reads[ev.get("out_frame", cur)].append(dict(r))
            continue
        if e == "frame":
            frames[ev["out_frame"]] = (ev["time"], ev["time_step"], ev["time_scale"])
            cur = ev["out_frame"]
        elif e == "cb":
            fn = str(ev.get("fn", ""))
            is_param = fn.endswith("checkout_param")
            is_layer = fn.endswith("checkout_layer")
            if not (is_param or is_layer):
                continue
            args = ev.get("args") or []
            idx = ev.get("param_index", args[1] if is_param and len(args) > 1 else None)
            t = ev.get("time", args[2] if is_param and len(args) > 2 else None)
            out = ev.get("out_frame", cur)
            if idx == 0 or idx in layer_of_index or is_layer:
                layer = "input" if idx == 0 else layer_of_index.get(idx, f"layer{idx}")
                reads[out].append({"layer": layer, "time": t, "frame": ev.get("footage_frame"), "inside": ev.get("inside")})
            else:
                psamples[out].append((idx, t))
        elif e in ("fault", "timeout", "case_error", "process_exit"):
            incomplete = incomplete or e
    return frames, reads, psamples, incomplete


def rewrite_view(lines):
    frames, reads, run = OrderedDict(), defaultdict(list), None
    for ev in lines:
        e = ev.get("ev")
        if e == "run":
            run = ev
        elif e == "frame":
            frames[ev["out_frame"]] = (ev["time"], ev["time_step"], ev["time_scale"])
        elif e == "read":
            reads[ev["out_frame"]].append({"layer": ev["layer"], "time": ev["time"], "index": ev.get("index"),
                                           "clip_frames": ev.get("clip_frames")})
    return frames, reads, run


def _uniq(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def compare_records(pv, rv, ordered=False, default_clip_frames=None):
    """One case: the plugin's view against the rewrite's, per output frame. -> dict with match, frames_mismatched
    and first_mismatch (the first differing output frame with what is missing, extra or at another time)."""
    pframes, preads, ppar, incomplete = pv
    rframes, rreads, _run = rv
    clip_frames = {}
    for rs in rreads.values():
        for r in rs:
            clip_frames.setdefault(r["layer"], r.get("clip_frames"))
    mismatches = []
    for n in sorted(set(pframes) | set(rframes) | set(preads) | set(rreads)):
        problems = []
        if n not in pframes:
            problems.append({"kind": "frame_missing_in_plugin_record"})
        elif n not in rframes:
            problems.append({"kind": "frame_missing_in_rewrite_record"})
        elif tuple(pframes[n]) != tuple(rframes[n]):
            problems.append({"kind": "frame_time", "plugin": list(pframes[n]), "rewrite": list(rframes[n])})
        P, R = preads.get(n, []), rreads.get(n, [])
        pk = [(r["layer"], r["time"]) for r in P]
        rk = [(r["layer"], r["time"]) for r in R]
        ps, rs_ = set(pk), set(rk)
        missing = sorted(ps - rs_, key=repr)
        extra = sorted(rs_ - ps, key=repr)
        pairs = []
        for layer in sorted({m[0] for m in missing} & {e[0] for e in extra}):
            ms = [m for m in missing if m[0] == layer]
            es = [e for e in extra if e[0] == layer]
            for m, e in zip(ms, es):
                pairs.append({"layer": layer, "plugin_time": m[1], "rewrite_time": e[1]})
                missing.remove(m)
                extra.remove(e)
        if pairs:
            problems.append({"kind": "wrong_time", "pairs": pairs})
        if missing:
            problems.append({"kind": "missing_read", "reads": [list(m) for m in missing]})
        if extra:
            problems.append({"kind": "extra_read", "reads": [list(e) for e in extra]})
        if not (missing or extra or pairs):
            if ordered and _uniq(pk) != _uniq(rk):
                problems.append({"kind": "order", "plugin": [list(x) for x in _uniq(pk)], "rewrite": [list(x) for x in _uniq(rk)]})

            def plugin_idx(r):
                if r.get("inside") is not None:
                    inside = bool(r["inside"])
                else:
                    cf = clip_frames.get(r["layer"], default_clip_frames)
                    inside = r.get("frame") is not None and cf is not None and 0 <= r["frame"] < cf
                return (r["layer"], r["frame"] if inside else None)

            pi = {plugin_idx(r) for r in P}
            ri = {(r["layer"], r.get("index")) for r in R}
            if pi != ri:
                problems.append({"kind": "wrong_index", "plugin": sorted([list(x) for x in pi], key=repr),
                                 "rewrite": sorted([list(x) for x in ri], key=repr)})
        if problems:
            mismatches.append({"out_frame": n, "problems": problems})
    anomalies = []
    for n, samples in ppar.items():
        if n in pframes:
            for idx, t in samples:
                if t is not None and t != pframes[n][0]:
                    anomalies.append({"out_frame": n, "param_index": idx, "time": t, "frame_time": pframes[n][0]})
    return {"match": not mismatches, "frames_mismatched": len(mismatches), "first_mismatch": mismatches[0] if mismatches else None,
            "frames_plugin": len(pframes), "frames_rewrite": len(rframes),
            "reads_plugin": sum(len(v) for v in preads.values()), "reads_rewrite": sum(len(v) for v in rreads.values()),
            "param_samples_plugin": sum(len(v) for v in ppar.values()), "param_samples_off_frame_time": len(anomalies),
            "param_sample_anomalies_first": anomalies[:3]}


def calls(ctx, plugin_dir, rewrite_dir, param_map=None, plugin="Twixtor_64", ordered=False, exclude_ooc=True):
    plugin_dir, rewrite_dir = Path(plugin_dir), Path(rewrite_dir)
    cases = load_cases(ctx, metadata_only=True)
    moved = set()
    if exclude_ooc and (ctx.results / "ooc.json").exists():
        moved = {m["id"] for m in jload(ctx.results / "ooc.json")["moved"]}
    lmap = layer_map_from_param_map(param_map, plugin)
    compared, matching, mism, skipped = 0, 0, [], []
    for cid, c in cases.items():
        if cid in moved:
            skipped.append({"id": cid, "why": "moved out of contract"})
            continue
        pp, rp = plugin_dir / f"{cid}.jsonl", rewrite_dir / f"{cid}.jsonl"
        if not pp.exists():
            skipped.append({"id": cid, "why": "no plugin record"})
            continue
        if not rp.exists():
            skipped.append({"id": cid, "why": "no rewrite record"})
            continue
        plines, pbad = load_jsonl(pp)
        rlines, rbad = load_jsonl(rp)
        pv = plugin_view(plines, lmap, {"input"} | set((c.get("layers") or {}).keys()))
        if not pv[0]:
            skipped.append({"id": cid, "why": "plugin record has no frame events" + (f" ({pv[3]})" if pv[3] else "")})
            continue
        if pv[3]:
            skipped.append({"id": cid, "why": f"plugin record is incomplete ({pv[3]})"})
            continue
        rv = rewrite_view(rlines)
        res = compare_records(pv, rv, ordered=ordered, default_clip_frames=c["frames"])
        compared += 1
        if res["match"] and not pbad and not rbad:
            matching += 1
        else:
            mism.append(dict(id=cid, bad_lines=[pbad, rbad], **res))
    doc = {"command": ctx.command, "measured_at": today(), "mode": "ordered" if ordered else "sets per output frame",
           "plugin_records": str(plugin_dir), "rewrite_records": str(rewrite_dir), "compared": compared, "matching": matching,
           "mismatching": len(mism), "skipped": len(skipped), "skipped_cases": skipped, "mismatches": mism,
           "limit": "parameter sample times are not recorded by the rewrite; plugin parameter samples are only checked against the frame time"}
    jdump(ctx.results / "calls.json", doc)
    return doc


def vfx_records(vfx, case):
    """True when this vfx build writes VFX_CALL_RECORD (builds older than that feature ignore the variable)."""
    with tempfile.TemporaryDirectory(prefix="phase_b_rec_") as t:
        p = Path(t) / "r.jsonl"
        rc, out, err = render(vfx, case, case["_dir"], record=p)
        return rc == 0 and p.exists() and p.stat().st_size > 0


def find_record_vfx(ctx, explicit, case):
    """A vfx that writes call records: the explicit one, else bin/vfx-base, else the harness copy of the rewrite."""
    if explicit:
        if not vfx_records(as_cmd(explicit), case):
            raise ToolError(f"{explicit} did not write a VFX_CALL_RECORD file: this build predates the call record")
        return as_cmd(explicit)
    tried = []
    for c in (ctx.root / "bin/vfx-base", Path.home() / "twixtor-parity/harness/repo/target/release/vfx", ctx.root / "src/target/release/vfx"):
        if c.exists():
            tried.append(str(c))
            if vfx_records([str(c)], case):
                return [str(c)]
    raise ToolError("no vfx that writes VFX_CALL_RECORD found (tried: " + ", ".join(tried) + "); pass --vfx")


def record_rewrite(ctx, vfx, out_dir, log=print):
    """Runs the rewrite with VFX_CALL_RECORD over the whole test set, and without it, to show the record leaves
    the output alone. -> {id: (exit, sha256 of stdout with the record, output unchanged by the record)}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(ctx)
    idx = []
    for cid, c in cases.items():
        rc, out, err = render(vfx, c, c["_dir"], record=out_dir / f"{cid}.jsonl")
        rc0, out0, _ = render(vfx, c, c["_dir"])
        idx.append((cid, rc, len(out), sha_bytes(out), rc == rc0 and out == out0, err))
    with open(out_dir / "index.tsv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("id\texit\tbytes\tsha256\toutput_unchanged_by_record\tstderr\n")
        for r in idx:
            fh.write("\t".join(str(x).replace("\t", " ") for x in r) + "\n")
    log(f"{len(idx)} records in {out_dir}, {sum(1 for r in idx if r[1] != 0)} with a nonzero exit, "
        f"{sum(1 for r in idx if not r[4])} where the record changed the output")
    return {r[0]: (r[1], r[3], r[4]) for r in idx}


# ------------------------------------------------------------------------------------------ kills


def exe_path(bin_dir, name):
    """bin/<name>, or with a Windows suffix when that is what exists."""
    for suffix in ("", ".exe", ".cmd"):
        if (Path(bin_dir) / (name + suffix)).exists():
            return Path(bin_dir) / (name + suffix)
    return Path(bin_dir) / name


def mutant_ids(ctx):
    p = ctx.phase_a / "mutants.json"
    if p.exists():
        return [m["id"] for m in jload(p) if m["id"] != "m00"]
    d = ctx.phase_a / "mutants"
    return sorted({f.name.split(".")[0] for f in d.glob("*.grid.tsv")} - {"m00"}) if d.exists() else []


def table_rows(phase_a, name):
    rows = {}
    for s in SETS:
        p = Path(phase_a) / "mutants" / f"{name}.{s}.tsv"
        if p.exists():
            for r in read_tsv(p):
                rows[r["id"]] = r
    return rows


def kill_candidates(cases, refs, moved, base_equal):
    """In-contract cases with a reference where the base rewrite equals it. -> (candidates, no_reference, base_differs, ooc)."""
    cand, noref, bdiff, ooc_ids = [], [], [], []
    for cid, c in cases.items():
        if cid in moved:
            ooc_ids.append(cid)
        elif not (Path(refs) / c["reference"]).is_file():
            noref.append(cid)
        elif base_equal(cid, c):
            cand.append(cid)
        else:
            bdiff.append(cid)
    return cand, noref, bdiff, ooc_ids


def kills(ctx, refs, mode="tables", bin_dir=None, only=None, log=print):
    refs = Path(refs)
    cases = load_cases(ctx, metadata_only=(mode == "tables"))
    moved = set()
    if (ctx.results / "ooc.json").exists():
        moved = {m["id"] for m in jload(ctx.results / "ooc.json")["moved"]}
    ids = mutant_ids(ctx)
    if only:
        ids = [i for i in ids if i in only]
    ref_sha = {}

    def rsha(cid, c):
        if cid not in ref_sha:
            ref_sha[cid] = sha_file(refs / c["reference"])
        return ref_sha[cid]

    if mode == "tables":
        base = load_base_rows(ctx.phase_a)
        if set(base) != set(cases):
            raise ToolError("the phase A base tables do not cover the test sets")

        def base_equal(cid, c):
            return base[cid]["exit"] == "0" and base[cid]["sha256"] == rsha(cid, c)

        cand, noref, bdiff, ooc_ids = kill_candidates(cases, refs, moved, base_equal)
        rendered = None
    else:
        bin_dir = Path(bin_dir) if bin_dir else ctx.root / "bin"
        base_bin = exe_path(bin_dir, "vfx-base")

        def base_equal(cid, c):
            rc, out, err = render(str(base_bin), c, c["_dir"])
            return rc == 0 and out == (refs / c["reference"]).read_bytes()

        cand, noref, bdiff, ooc_ids = kill_candidates(cases, refs, moved, base_equal)
    per, control = OrderedDict(), None
    all_ids = ["m00"] + ids
    for mid in all_ids:
        killing, failing, missing = [], [], []
        if mode == "tables":
            rows = table_rows(ctx.phase_a, mid)
            for cid in cand:
                r = rows.get(cid)
                if r is None:
                    missing.append(cid)
                elif r["exit"] != "0":
                    killing.append(cid)
                    failing.append(cid)
                elif r["sha256"] != rsha(cid, cases[cid]):
                    killing.append(cid)
        else:
            exe = exe_path(bin_dir, f"vfx-{mid}")
            if not exe.exists():
                missing = list(cand)
            else:
                for cid in cand:
                    c = cases[cid]
                    rc, out, err = render(str(exe), c, c["_dir"])
                    if rc != 0:
                        killing.append(cid)
                        failing.append(cid)
                    elif out != (refs / c["reference"]).read_bytes():
                        killing.append(cid)
        entry = {"killed": bool(killing), "killing_cases": killing, "kill_count": len(killing),
                 "killed_by_failure": failing, "cases_without_output": len(missing)}
        if missing:
            entry["not_run"] = True
        if mid == "m00":
            control = {"id": "m00", "cases_differing_from_reference": killing, "ok": not killing and not missing,
                       "not_run": bool(missing)}
        else:
            per[mid] = entry
    killed = [m for m, e in per.items() if e["killed"]]
    not_run = [m for m, e in per.items() if e.get("not_run")]
    survivors = [m for m in per if m not in killed]
    agreement = None
    gp = ctx.results / "grade.json"
    if gp.exists():
        grows = {r["id"]: r for r in jload(gp)["cases"]}
        g_ident = {i for i in cases if i in grows and grows[i]["status"] == "identical" and i not in moved}
        base_ident = set(cand)
        agreement = {"grade_identical_not_in_kill_candidates": sorted(g_ident - base_ident)[:20],
                     "kill_candidates_not_identical_in_grade": sorted(base_ident - g_ident)[:20]}
    doc = {"command": ctx.command, "measured_at": today(), "mode": mode, "refs": str(refs),
           "definition": "among in-contract cases where the base rewrite output equals the reference, a mutant is killed iff its output differs from the reference on at least one",
           "cases": {"in_test_sets": len(cases), "kill_candidates": len(cand), "out_of_contract": len(ooc_ids),
                     "no_reference": len(noref), "base_differs_from_reference": len(bdiff)},
           "base_differs_from_reference": bdiff, "no_reference": noref, "control": control,
           "totals": {"total": len(per), "killed": len(killed), "survivors": survivors, "not_run": not_run},
           "mutants": per, "agreement_with_grade": agreement}
    jdump(ctx.results / "kills.json", doc)
    return doc


# ------------------------------------------------------------------------------------------ summary


def unmet_rule(s):
    """The status rule of docs/parity.md, applied to a summary's numbers (a mirror of tools/docgen/data.py)."""
    c, v, k, m = s["cases"], s["values"], s["call_records"], s["mutants"]
    out = []
    if c["compared"] < MIN_CASES:
        out.append(f"fewer than {MIN_CASES} cases compared")
    if c["compared"] < c["in_contract"]:
        out.append("not every in-contract case compared")
    if c["differing"] or v["differing_values"] or c["identical"] != c["compared"]:
        out.append("a case or value differs")
    if not k["compared"] or k["matching"] != k["compared"]:
        out.append("call records not all matching")
    if not m["total"] or m["killed"] != m["total"] or m["survivors"]:
        out.append("a mutant survives or none ran")
    return out


def summary(ctx, date=None, plugin_sha=None):
    gp = ctx.results / "grade.json"
    if not gp.exists():
        raise ToolError(f"{gp} does not exist: run `grade` first (summary refuses to run without it)")
    grade_doc = jload(gp)
    cases = load_cases(ctx, metadata_only=True)
    grows = {r["id"]: r for r in grade_doc["cases"]}
    if set(grows) != set(cases):
        raise ToolError("grade.json does not cover exactly the module's test sets (was it written for another test set?)")
    prev = {}
    for p in (ctx.phase_a / "summary.json", ctx.results / "summary.json"):
        if p.exists():
            prev = jload(p)
            break
    moved, ooc_doc = set(), None
    if (ctx.results / "ooc.json").exists():
        ooc_doc = jload(ctx.results / "ooc.json")
        moved = {m["id"] for m in ooc_doc["moved"]}
    in_contract = [i for i in cases if i not in moved]
    comp = [grows[i] for i in in_contract if grows[i]["compared"]]
    diff = [r for r in comp if r["status"] == "differs"]
    if ooc_doc:
        cand_ooc = ooc_doc["counts"]["candidates_out_of_contract"]
    else:
        cand = load_candidates(ctx)
        cand_ooc = len(cand)
    ulps = [r["max_ulp"] for r in comp if r.get("max_ulp") is not None]
    floats = sum(r.get("floats_compared", 0) for r in comp)
    n_stand = sum(1 for c in cases.values() if c.get("stand_in"))
    n_stand_cmp = sum(1 for r in comp if r["logic_parity"])
    if (ctx.results / "kills.json").exists():
        kd = jload(ctx.results / "kills.json")
        mutants = {"total": kd["totals"]["total"], "killed": kd["totals"]["killed"], "survivors": list(kd["totals"]["survivors"])}
    else:
        ids = mutant_ids(ctx)
        mutants = {"total": len(ids) or prev.get("mutants", {}).get("total", 0), "killed": 0, "survivors": []}
    if (ctx.results / "calls.json").exists():
        cd = jload(ctx.results / "calls.json")
        calls_ = {"compared": cd["compared"], "matching": cd["matching"]}
    else:
        calls_ = {"compared": 0, "matching": 0}
    s = OrderedDict()
    s["module"] = prev.get("module", "retime-flow")
    s["plugin"] = prev.get("plugin", "Twixtor_64.aex")
    s["plugin_sha256"] = plugin_sha or prev.get("plugin_sha256", "")
    s["effect"] = prev.get("effect", "retime")
    s["status"] = "in_progress"
    s["blockers"] = []
    s["cases"] = {"in_contract": len(in_contract), "compared": len(comp), "identical": len(comp) - len(diff), "differing": len(diff),
                  "out_of_contract": cand_ooc + len(moved), "logic_parity": n_stand_cmp}
    s["values"] = {"bytes_compared": sum(r.get("bytes_compared", 0) for r in comp), "floats_compared": floats,
                   "differing_values": sum(r["differing_values"] or 0 for r in diff),
                   "max_ulp": (max(ulps) if ulps else 0) if floats else None}
    s["call_records"] = calls_
    s["mutants"] = mutants
    s["stand_ins"] = [
        f"{n_stand} cases with keyed speed_percent (AE keyframe interpolation) are tagged stand_in=keyframe_interpolation; "
        f"{n_stand_cmp} of them compared, counted as logic_parity (not full parity)",
        "draft quality cases (testsets/draft) need the same quality setting on the host side as vfx --quality draft",
    ]
    s["fixes"] = prev.get("fixes", [])
    cmds = list(prev.get("commands", []))
    for d in (grade_doc, ooc_doc, jload(ctx.results / "kills.json") if (ctx.results / "kills.json").exists() else None,
              jload(ctx.results / "calls.json") if (ctx.results / "calls.json").exists() else None):
        if d and d.get("command") and d["command"] not in cmds:
            cmds.append(d["command"])
    s["commands"] = cmds
    s["measured_at"] = date or today()
    if "phase_a" in prev:
        s["phase_a"] = prev["phase_a"]
    unmet = unmet_rule(s)
    if s["cases"]["compared"] == 0:
        s["blockers"].append("emulator side not run yet")
    elif s["cases"]["compared"] < s["cases"]["in_contract"]:
        s["blockers"].append(f"{s['cases']['in_contract'] - s['cases']['compared']} in-contract cases have no reference")
    if ooc_doc and ooc_doc["counts"]["unjudged"]:
        s["blockers"].append(f"{ooc_doc['counts']['unjudged']} cases have no usable outcome from the harness")
    if not unmet and not s["blockers"]:
        s["status"] = "verified"
    jdump(ctx.results / "summary.json", s)
    return s, unmet


# ------------------------------------------------------------------------------------------ triage


def default_params(vfx, effect="retime"):
    try:
        p = subprocess.run(as_cmd(vfx) + ["list", "--json", "--all"], stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
        for e in json.loads(p.stdout):
            if e.get("id") == effect:
                return {x["id"]: str(x.get("default", "")) for x in e.get("params", [])}, "vfx list"
    except Exception:  # noqa: BLE001
        pass
    return dict(FALLBACK_DEFAULTS), "built-in fallback"


def same_value(a, b):
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def triage_groups(rows):
    """For each dimension and value: (differing, total) over the compared cases."""
    dims = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        if not r["compared"]:
            continue
        bad = 1 if r["status"] == "differs" else 0
        keys = [("group", r.get("group")), ("depth", r["depth"]), ("size", f"{r['width']}x{r['height']}"), ("frames", r["frames"]),
                ("fps", r.get("fps")), ("source", r.get("source")), ("quality", r.get("quality") or "default"),
                ("stand_in", r.get("stand_in") or "none")]
        keys += [("flag", f) for f in (r.get("flags") or ["none"])]
        pd = param_dict(r)
        keys += [(f"param {k}", v) for k, v in pd.items()]
        for dim, val in keys:
            cell = dims[dim][str(val)]
            cell[0] += bad
            cell[1] += 1
    return dims


def smallest_first(rows):
    bad = [r for r in rows if r["status"] == "differs"]
    return sorted(bad, key=lambda r: (r["width"] * r["height"] * r["frames"] * BPP[r["depth"]], r["frames"], r["depth"],
                                      r["width"] * r["height"], r["id"]))


def build_ladder(row, defaults):
    """Probe cases around one failing case. -> list of (rung names joined by ' = ', params list, quality): the case
    itself, all defaults, each non-default setting reverted alone, each non-default setting alone on the defaults,
    and the quality flipped. Rungs with the same settings are merged."""
    pd = param_dict(row)
    quality = row.get("quality")
    nondefault = [k for k, v in pd.items() if not (k in defaults and same_value(v, defaults[k]))]
    rungs = OrderedDict()

    def add(name, params, q):
        key = (tuple(sorted(params)), q)
        if key in rungs:
            rungs[key][0].append(name)
        else:
            rungs[key] = [[name], params, q]

    add("original", [f"{k}={v}" for k, v in pd.items()], quality)
    add("all-default", [], quality)
    for k in nondefault:
        add(f"revert-{k}", [f"{kk}={vv}" for kk, vv in pd.items() if kk != k], quality)
    for k in nondefault:
        add(f"only-{k}", [f"{k}={pd[k]}"], quality)
    add("quality-draft" if quality != "draft" else "quality-default", [f"{k}={v}" for k, v in pd.items()],
        "draft" if quality != "draft" else None)
    return [(" = ".join(v[0]), v[1], v[2]) for v in rungs.values()]


def write_ladder(row, rungs, out_dir):
    """Writes `out_dir` as an fx-parity format 1 test set: the failing case's input, one case per rung."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "inputs").mkdir(parents=True)
    src = Path(row["testset_dir"]) / row["input"]
    fname = "inputs/" + Path(row["input"]).name
    if src.exists():
        shutil.copyfile(src, out_dir / fname)
    cases = []
    for name, params, q in rungs:
        key = json.dumps([row["id"], name, params, q])
        h = hashlib.sha1(key.encode()).hexdigest()[:8]
        label = name.split(" = ")[0]
        cid = f"retime.probe-{label}.{row['source']}-{row['width']}x{row['height']}.d{row['depth']}.{h}"
        case = {"id": cid, "effect": row["effect"], "label": f"probe-{label}", "params": params, "depth": row["depth"],
                "width": row["width"], "height": row["height"], "source": row["source"], "input": fname,
                "reference": f"{cid}.{PIX[row['depth']]}", "kind": "temporal", "frames": row["frames"], "fps": row["fps"],
                "group": "probe", "ladder": name, "ladder_of": row["id"]}
        if q:
            case["quality"] = q
        cases.append(case)
    meta = {"file": fname, "source": row["source"], "width": row["width"], "height": row["height"], "depth": row["depth"],
            "pix_fmt": PIX[row["depth"]], "frames": row["frames"]}
    jdump(out_dir / "cases.json", {"format": 1, "ffmpeg": "none (probe ladder written by phase_b.py)", "inputs": [meta], "cases": cases})
    return cases


def triage(ctx, vfx=None, probe_dir=None, out=print):
    gp = ctx.results / "grade.json"
    if not gp.exists():
        raise ToolError(f"{gp} does not exist: run `grade` first")
    rows = jload(gp)["cases"]
    moved = set()
    if (ctx.results / "ooc.json").exists():
        moved = {m["id"] for m in jload(ctx.results / "ooc.json")["moved"]}
    rows = [r for r in rows if r["id"] not in moved]
    comp = [r for r in rows if r["compared"]]
    bad = smallest_first(rows)
    out(f"{len(bad)} of {len(comp)} compared in-contract cases differ ({len(rows) - len(comp)} without a reference)")
    res = {"differing": len(bad), "compared": len(comp), "dimensions": {}, "smallest": [], "ladder": None}
    if not bad:
        jdump(ctx.results / "triage.json", res)
        return res
    out("\nsettings against differences (differing/total per value; * = every case with it differs)")
    dims = triage_groups(rows)
    for dim in sorted(dims):
        cells = sorted(dims[dim].items(), key=lambda kv: (-(kv[1][0] / kv[1][1]), -kv[1][1], kv[0]))
        res["dimensions"][dim] = {v: {"differing": b, "total": t} for v, (b, t) in cells}
        if not any(b for b, _ in dims[dim].values()):
            continue
        out(f"  {dim}: " + ", ".join(f"{v} {b}/{t}{'*' if b == t else ''}" for v, (b, t) in cells if b))
    out("\nsmallest failing cases first (values = bytes of output)")
    for r in bad[:15]:
        size = r["width"] * r["height"] * r["frames"] * BPP[r["depth"]]
        out(f"  {size:>8} B  {r['width']}x{r['height']}  frames {r['frames']}  d{r['depth']}  {r['id']}  params {' '.join(r['params'])}"
            f"  values {r['differing_values']}  ulp {r['max_ulp']}")
        res["smallest"].append({"id": r["id"], "bytes": size, "width": r["width"], "height": r["height"], "frames": r["frames"],
                                "depth": r["depth"], "differing_values": r["differing_values"], "max_ulp": r["max_ulp"]})
    first = bad[0]
    defaults, how = default_params(find_vfx(ctx, vfx))
    rungs = build_ladder(first, defaults)
    pdir = Path(probe_dir) if probe_dir else ctx.root / "probe"
    cases = write_ladder(first, rungs, pdir)
    res["ladder"] = {"dir": str(pdir), "of": first["id"], "defaults_from": how, "rungs": [c["ladder"] for c in cases]}
    out(f"\nladder for {first['id']} ({len(cases)} probe cases, defaults from {how}) written to {pdir}")
    out(f"  render them with the harness, then:  python3 tools/phase_b.py grade --testset {pdir} --refs <probe refs> --results {ctx.results}/probe")
    jdump(ctx.results / "triage.json", res)
    return res


# ------------------------------------------------------------------------------------------ selftest


def _sub(ctx, name):
    import copy
    c = copy.copy(ctx)
    c.results = Path(ctx.results) / "selftest" / name
    c.results.mkdir(parents=True, exist_ok=True)
    c.command = f"python3 tools/phase_b.py selftest ({name})"
    return c


class Checks:
    def __init__(self, log=print):
        self.items, self.log = [], log

    def __call__(self, name, ok, detail=""):
        self.items.append({"check": name, "ok": bool(ok), "detail": str(detail)})
        self.log(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
        return bool(ok)

    @property
    def failed(self):
        return [i for i in self.items if not i["ok"]]


def expected_kills(changed, candidates):
    """The kill sets implied by the phase A change lists: mutant -> sorted candidate cases it changes."""
    cand = set(candidates)
    return {m: sorted(set(ids) & cand) for m, ids in changed.items() if m != "m00"}


def synth_plugin_lines(case, rewrite_lines):
    """The rewrite's call record rewritten as the harness would log it for the same case (stand-in only):
    a `frame` line per output frame, two parameter checkouts at the frame time, and a layer-0 checkout
    (`inter.checkout_param`, param_index 0) per clip read with the footage frame floor(t / time_scale * fps)."""
    fps = Fraction(str(case["fps"]))
    out, ts, step = [], 1, 1

    def add(d):
        out.append(dict({"seq": len(out) + 1}, **d))

    cur = 0
    for ev in rewrite_lines:
        if ev["ev"] == "frame":
            cur, ts, step = ev["out_frame"], ev["time_scale"], ev["time_step"]
            add({"ev": "frame", "out_frame": cur, "time": ev["time"], "time_step": step, "time_scale": ts})
            for pi in (3, 5):
                add({"ev": "cb", "fn": "inter.checkout_param", "args": ["0x1", pi, ev["time"], step, ts, "0x2"], "ret": 0,
                     "param_index": pi, "time": ev["time"], "time_step": step, "time_scale": ts,
                     "footage_frame": math.floor(Fraction(ev["time"], ts) * fps), "out_frame": cur})
        elif ev["ev"] == "read":
            add({"ev": "cb", "fn": "inter.checkout_param", "args": ["0x1", 0, ev["time"], step, ts, "0x2"], "ret": 0,
                 "param_index": 0, "time": ev["time"], "time_step": step, "time_scale": ts,
                 "footage_frame": math.floor(Fraction(ev["time"], ts) * fps), "out_frame": ev["out_frame"]})
    add({"ev": "case_done", "status": "rendered"})
    return out


def _write_jsonl(path, lines):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for l in lines:
            fh.write(json.dumps(l) + "\n")


def _words(path):
    arr = array("I")
    arr.frombytes(Path(path).read_bytes())
    return arr


def _patch_word(path, index, new):
    arr = _words(path)
    arr[index] = new
    Path(path).write_bytes(arr.tobytes())


def selftest(ctx, log=print):
    """Stand-in oracles for the tooling. Writes only under <results>/selftest/; the phase A results are read,
    never written. Needs the rewrite binaries (bin/vfx-base, bin/vfx-mNN), fx-parity and the test sets, so it
    runs on the server."""
    st = Path(ctx.results) / "selftest"
    if st.exists():
        shutil.rmtree(st)
    st.mkdir(parents=True)
    ck = Checks(log)
    report = OrderedDict()
    vfx, fxp = find_vfx(ctx, None), find_fxp(ctx, None)
    cases = load_cases(ctx)
    guard = {}
    pa_dir = Path(ctx.phase_a)
    for p in sorted(pa_dir.glob("*.tsv")) + sorted((pa_dir / "mutants").glob("*.tsv")) + [pa_dir / "summary.json", pa_dir / "mutants.json"]:
        if p.exists():
            guard[str(p)] = sha_file(p)
    phase_a_changed = {m["id"]: set(m["changed"]) for m in jload(pa_dir / "mutants.json")}
    mutants_changing = sorted(m for m, ids in phase_a_changed.items() if m != "m00" and ids)
    ids_all = list(cases)
    report["phase_a_mutants_that_change_an_output"] = len(mutants_changing)
    report["phase_a_mutants_total"] = len(phase_a_changed) - 1

    # ---- (a) perfect oracle
    log("(a) perfect oracle: the rewrite's own base renders as the references")
    pa = _sub(ctx, "perfect")
    refs_a = pa.results / "refs"
    refs_a.mkdir(parents=True, exist_ok=True)
    base_rows = load_base_rows(ctx.phase_a)
    sha_now, agree = {}, 0
    for cid, c in cases.items():
        rc, out, err = render(vfx, c, c["_dir"])
        if rc != 0:
            raise ToolError(f"the rewrite failed on {cid}: {err}")
        (refs_a / c["reference"]).write_bytes(out)
        sha_now[cid] = sha_bytes(out)
        agree += 1 if base_rows.get(cid, {}).get("sha256") == sha_now[cid] else 0
    ck("base renders equal the phase A base tables", agree == len(cases), f"{agree} of {len(cases)}")
    rows, cmds = grade(pa, refs_a, vfx, fxp)
    doc = write_grade(pa, rows, cmds, refs_a)
    t = doc["totals"]
    ck("perfect oracle: 0 differing, all compared", t["differing"] == 0 and t["compared"] == len(cases) and t["identical"] == len(cases),
       f"compared {t['compared']}, identical {t['identical']}, differing {t['differing']}, floats {t['floats_compared']}, bytes {t['bytes_compared']}, max_ulp {t['max_ulp']}")
    ck("logic parity cases counted", t["logic_parity_compared"] == sum(1 for c in cases.values() if c.get("stand_in")), t["logic_parity_compared"])
    kd = kills(pa, refs_a, "tables")
    exp = expected_kills(phase_a_changed, ids_all)
    same = all(sorted(kd["mutants"][m]["killing_cases"]) == exp[m] for m in exp)
    ck("perfect oracle: kills (tables) equal the phase A change lists exactly", same and kd["totals"]["killed"] == len(mutants_changing),
       f"killed {kd['totals']['killed']} of {kd['totals']['total']}; phase A mutants that change an output: {len(mutants_changing)}")
    ck("control m00 kills nothing", kd["control"] and kd["control"]["ok"], kd["control"])
    kr = kills(pa, refs_a, "rerender")
    same_r = all(sorted(kr["mutants"][m]["killing_cases"]) == exp[m] for m in exp)
    ck("perfect oracle: kills (rerender with bin/vfx-mNN) agree with the tables", same_r and kr["totals"]["killed"] == kd["totals"]["killed"],
       f"killed {kr['totals']['killed']}, control ok {kr['control']['ok'] if kr['control'] else None}")
    report["perfect_oracle"] = {"differing": t["differing"], "compared": t["compared"], "mutants_killed_tables": kd["totals"]["killed"],
                                "mutants_killed_rerender": kr["totals"]["killed"], "survivors": kd["totals"]["survivors"]}
    kd = kills(pa, refs_a, "tables")
    rec_rw = pa.results / "rewrite_records"
    rec_pl = pa.results / "plugin_records"
    rec_rw.mkdir(parents=True, exist_ok=True)
    rec_pl.mkdir(parents=True, exist_ok=True)
    rec_vfx = find_record_vfx(ctx, None, next(iter(cases.values())))
    idx = record_rewrite(pa, rec_vfx, rec_rw, log=lambda *_: None)
    ck("call record leaves the output bytes unchanged", all(idx[i][2] and idx[i][0] == 0 for i in cases), f"{len(idx)} cases, binary {rec_vfx[0]}")
    ck("the record binary renders what bin/vfx-base renders (information: is the graded binary the recorded one)", True,
       f"{sum(1 for i in cases if idx[i][1] == sha_now[i])} of {len(cases)} identical")
    for cid, c in cases.items():
        lines, _ = load_jsonl(rec_rw / f"{cid}.jsonl")
        _write_jsonl(rec_pl / f"{cid}.jsonl", synth_plugin_lines(c, lines))
    cd = calls(pa, rec_pl, rec_rw)
    first = (cd["mismatches"][0]["id"] + " " + json.dumps(cd["mismatches"][0]["first_mismatch"])) if cd["mismatches"] else None
    ck("synthetic plugin records equal the rewrite's records", cd["compared"] == len(cases) and cd["matching"] == len(cases),
       f"compared {cd['compared']}, matching {cd['matching']}, first mismatch {first}")
    s, unmet = summary(pa, date="selftest")
    want_cases = {"in_contract": len(cases), "compared": len(cases), "identical": len(cases), "differing": 0,
                  "out_of_contract": len(load_candidates(ctx)), "logic_parity": sum(1 for c in cases.values() if c.get("stand_in"))}
    ck("perfect oracle: summary in a scratch folder", s["cases"] == want_cases and s["mutants"]["killed"] == len(mutants_changing),
       f"status {s['status']} (scratch only), unmet {unmet}, cases {s['cases']}, mutants {s['mutants']['killed']}/{s['mutants']['total']}, calls {s['call_records']}")
    report["perfect_summary"] = {k: s[k] for k in ("status", "cases", "values", "call_records", "mutants")}

    # ---- (b) perturbed oracle
    log("(b) perturbed oracle: one bit, a -0, a NaN payload, one mantissa bit")
    pb = _sub(ctx, "perturbed")
    refs_b = pb.results / "refs"
    shutil.copytree(refs_a, refs_b)
    used, perturbed, injected = set(), {}, []
    c8 = next(i for i, c in cases.items() if c["depth"] == 8)
    p = refs_b / cases[c8]["reference"]
    b = bytearray(p.read_bytes())
    b[0] ^= 1
    p.write_bytes(bytes(b))
    used.add(c8)
    perturbed[c8] = {"what": "8 bpc bit flip", "differing_values": 1, "max_ulp": None, "nan_mismatch": 0, "signed_zero_diffs": 0}

    def find32(pred, avoid):
        for i, c in cases.items():
            if c["depth"] == 32 and i not in avoid:
                arr = _words(refs_b / c["reference"])
                for k, w in enumerate(arr):
                    if pred(w):
                        return i, k
        return None, None

    ci, k = find32(lambda w: w == 0x80000000, used)
    if ci is None:
        ci = next(i for i, c in cases.items() if c["depth"] == 32 and i not in used)
        _patch_word(refs_b / cases[ci]["reference"], 0, 0x80000000)
        injected.append("negative zero written into a reference (no case outputs -0)")
        perturbed[ci] = {"what": "injected -0 in the reference", "differing_values": 1}
    else:
        _patch_word(refs_b / cases[ci]["reference"], k, 0)
        perturbed[ci] = {"what": "-0 became 0", "differing_values": 1, "max_ulp": 0, "nan_mismatch": 0, "signed_zero_diffs": 1}
    used.add(ci)
    ci, k = find32(f32_is_nan, used)
    if ci is None:
        ci = next(i for i, c in cases.items() if c["depth"] == 32 and i not in used)
        _patch_word(refs_b / cases[ci]["reference"], 0, 0x7FC00000)
        injected.append("NaN written into a reference (no case outputs NaN)")
        perturbed[ci] = {"what": "injected NaN in the reference", "differing_values": 1, "nan_mismatch": 1, "max_ulp": None}
    else:
        w = _words(refs_b / cases[ci]["reference"])[k]
        _patch_word(refs_b / cases[ci]["reference"], k, w ^ (1 if (w & 0x7FFFFF) ^ 1 else 2))
        perturbed[ci] = {"what": "NaN payload changed", "differing_values": 1, "nan_mismatch": 1, "max_ulp": None, "signed_zero_diffs": 0}
    used.add(ci)
    ci, k = find32(lambda w: (w & 0x7F800000) != 0x7F800000, used)
    _patch_word(refs_b / cases[ci]["reference"], k, _words(refs_b / cases[ci]["reference"])[k] ^ 1)
    used.add(ci)
    perturbed[ci] = {"what": "32 bpc mantissa bit flip", "differing_values": 1, "max_ulp": 1, "nan_mismatch": 0, "signed_zero_diffs": 0}
    report["perturbed_cases"] = {i: v["what"] for i, v in perturbed.items()}
    report["perturbed_injections"] = injected
    rows, cmds = grade(pb, refs_b, vfx, fxp)
    doc = write_grade(pb, rows, cmds, refs_b)
    got = {r["id"]: r for r in rows if r["status"] == "differs"}
    ck("perturbed oracle: exactly the perturbed cases differ", set(got) == set(perturbed), f"{len(got)} differ, {len(perturbed)} perturbed")
    for cid, want in perturbed.items():
        r = got.get(cid, {})
        ok = all(r.get(k2) == v for k2, v in want.items() if k2 != "what")
        ck(f"  {want['what']}", ok, json.dumps({k2: r.get(k2) for k2 in ("differing_values", "max_ulp", "nan_mismatch", "signed_zero_diffs", "first")}))
    ck("perturbed oracle: totals", doc["totals"]["differing"] == len(perturbed) and doc["totals"]["identical"] == len(cases) - len(perturbed),
       f"differing {doc['totals']['differing']}, differing_values {doc['totals']['differing_values']}, max_ulp {doc['totals']['max_ulp']}, nan {doc['totals']['nan_mismatch_values']}")
    kd = kills(pb, refs_b, "tables")
    exp = expected_kills(phase_a_changed, [i for i in ids_all if i not in perturbed])
    ck("perturbed oracle: kills skip cases where the base already differs", all(sorted(kd["mutants"][m]["killing_cases"]) == exp[m] for m in exp)
       and sorted(kd["base_differs_from_reference"]) == sorted(perturbed),
       f"killed {kd['totals']['killed']} of {kd['totals']['total']}, survivors {kd['totals']['survivors']}")
    some = {m for m in ("m04", "m10", "m13", "m16", "m17") if m in exp}
    kr = kills(pb, refs_b, "rerender", only=some)
    ck("perturbed oracle: rerender kills agree (five of the smallest mutants)", set(kr["mutants"]) == some
       and all(sorted(kr["mutants"][m]["killing_cases"]) == exp[m] for m in some), f"killed {kr['totals']['killed']} of {kr['totals']['total']}")
    kd = kills(pb, refs_b, "tables")
    shutil.copyfile(pa.results / "calls.json", pb.results / "calls.json")
    s, unmet = summary(pb, date="selftest")
    ck("perturbed oracle: summary counts the perturbed cases", s["cases"]["differing"] == len(perturbed) and s["status"] == "in_progress"
       and s["values"]["differing_values"] == sum(v["differing_values"] for v in perturbed.values()), f"cases {s['cases']}, values {s['values']}, status {s['status']}")
    report["perturbed_oracle"] = {"differing": len(got), "mutants_killed": kd["totals"]["killed"], "survivors": kd["totals"]["survivors"]}

    # ---- (c) missing references
    log("(c) missing references")
    pc = _sub(ctx, "missing")
    refs_c = pc.results / "refs"
    shutil.copytree(refs_a, refs_c)
    removed = sorted(phase_a_changed.get("m13", set()))
    for i in ids_all:
        if len(removed) >= 10:
            break
        if i not in removed:
            removed.append(i)
    for i in removed:
        (refs_c / cases[i]["reference"]).unlink()
    rows, cmds = grade(pc, refs_c, vfx, fxp)
    doc = write_grade(pc, rows, cmds, refs_c)
    t = doc["totals"]
    ck("missing references: neither a pass nor a failure", t["no_reference"] == len(removed) and t["compared"] == len(cases) - len(removed)
       and t["identical"] == len(cases) - len(removed) and t["differing"] == 0, f"no_reference {t['no_reference']}, compared {t['compared']}, identical {t['identical']}")
    kd = kills(pc, refs_c, "tables")
    exp = expected_kills(phase_a_changed, [i for i in ids_all if i not in removed])
    ck("missing references: kills use the cases that have one", all(sorted(kd["mutants"][m]["killing_cases"]) == exp[m] for m in exp)
       and sorted(kd["no_reference"]) == sorted(removed), f"killed {kd['totals']['killed']}, survivors {kd['totals']['survivors']}")
    s, unmet = summary(pc, date="selftest")
    ck("missing references: summary does not count them as compared", s["cases"]["in_contract"] == len(cases) and s["cases"]["compared"] == len(cases) - len(removed)
       and s["status"] == "in_progress" and any("no reference" in b for b in s["blockers"]), f"cases {s['cases']}, blockers {s['blockers']}")
    report["missing_references"] = {"removed": len(removed), "compared": t["compared"], "survivors": kd["totals"]["survivors"]}

    # ---- (d) out of contract
    log("(d) out of contract")
    pd_ = _sub(ctx, "ooc")
    shutil.copyfile(pa.results / "grade.json", pd_.results / "grade.json")
    m04 = sorted(phase_a_changed.get("m04", set()))
    pool = m04 + [i for i in ids_all if i not in m04]
    refused, errored, timed, faulted, notrun = pool[0:3], pool[3:5], pool[5:6], pool[6:7], pool[7:8]
    recs = [{"id": i, "status": "unsupported", "error": "param `x` has no counterpart among the registered params of Twixtor_64"} for i in refused]
    recs += [{"id": i, "status": "error", "error": "PluginError: RENDER returned PF_Err 25"} for i in errored]
    recs += [{"id": i, "status": "timeout", "error": "no frame in 600 s"} for i in timed]
    recs += [{"id": i, "status": "error", "error": "GuestFault: read of unmapped memory"} for i in faulted]
    recs += [{"id": i, "status": "not_run", "error": "the plugin did not boot"} for i in notrun]
    recs += [{"id": i, "status": "rendered", "seconds": 1.0} for i in pool[8:12]]
    recs += [{"id": "ooc.0", "status": "unsupported", "error": "outside the registered range"}, {"id": "ooc.4", "status": "rendered"}]
    sp = pd_.results / "status.json"
    jdump(sp, {"testset": "synthetic", "cases": recs})
    od = ooc(pd_, [sp])
    moved = refused + errored + timed
    ck("out of contract: refused, errored and timed out cases are moved", sorted(m["id"] for m in od["moved"]) == sorted(moved), f"{len(od['moved'])} moved")
    ck("out of contract: an emulator fault and a missing outcome are not moved", sorted(m["id"] for m in od["unjudged"]) == sorted(faulted + notrun),
       f"{len(od['unjudged'])} unjudged")
    cv = {c["id"]: c["verdict"] for c in od["candidates"]}
    ck("out of contract: candidates judged from the status", cv.get("ooc.0") == "out_of_contract_confirmed" and cv.get("ooc.4") == "plugin_renders_it" and cv.get("ooc.1") == "unjudged", cv.get("ooc.0"))
    ck("out of contract: each moved case shows what both sides did", all(m["plugin"] and m["rewrite"].get("recorded") for m in od["moved"]), od["moved"][0]["rewrite"])
    kd = kills(pd_, refs_a, "tables")
    exp = expected_kills(phase_a_changed, [i for i in ids_all if i not in moved])
    ck("out of contract: moved cases are not kill candidates", all(sorted(kd["mutants"][m]["killing_cases"]) == exp[m] for m in exp) and kd["cases"]["out_of_contract"] == len(moved),
       f"killed {kd['totals']['killed']}, survivors {kd['totals']['survivors']}")
    s, unmet = summary(pd_, date="selftest")
    n_cand = len(load_candidates(ctx))
    ck("out of contract: they leave the in-contract count", s["cases"]["in_contract"] == len(cases) - len(moved) and s["cases"]["compared"] == len(cases) - len(moved)
       and s["cases"]["out_of_contract"] == (n_cand - 1) + len(moved), f"cases {s['cases']}")
    report["out_of_contract"] = {"moved": len(moved), "in_contract": s["cases"]["in_contract"], "out_of_contract": s["cases"]["out_of_contract"],
                                 "survivors": kd["totals"]["survivors"]}

    # ---- (e) call records
    log("(e) call records")
    pe = _sub(ctx, "calls")
    rw, pl = pe.results / "rewrite", pe.results / "plugin"
    shutil.copytree(rec_rw, rw)
    shutil.copytree(rec_pl, pl)

    def frames_with_reads(cid):
        lines, _ = load_jsonl(rw / f"{cid}.jsonl")
        per = defaultdict(list)
        for i, ev in enumerate(lines):
            if ev["ev"] == "read":
                per[ev["out_frame"]].append((i, ev))
        return lines, per

    def unique_read(per):
        for n in sorted(per):
            keys = Counter((e["layer"], e["time"]) for _, e in per[n])
            for i, e in per[n]:
                if keys[(e["layer"], e["time"])] == 1:
                    return n, i, e
        return None

    targets = {}
    usable = []
    for cid in ids_all:
        lines, per = frames_with_reads(cid)
        if unique_read(per):
            usable.append((cid, lines, per))
    for kind, (cid, lines, per) in zip(["missing_read", "extra_read", "wrong_time", "frame_time", "frame_dropped"], usable[0:5]):
        n, i, e = unique_read(per)
        targets[kind] = (cid, n)
        if kind == "missing_read":
            del lines[i]
            _write_jsonl(rw / f"{cid}.jsonl", lines)
        elif kind == "extra_read":
            pl_lines, _ = load_jsonl(pl / f"{cid}.jsonl")
            j = next(k for k, ev in enumerate(pl_lines) if ev.get("param_index") == 0 and ev.get("out_frame") == n and ev.get("time") == e["time"])
            del pl_lines[j]
            _write_jsonl(pl / f"{cid}.jsonl", pl_lines)
        elif kind == "wrong_time":
            lines[i]["time"] += 1
            _write_jsonl(rw / f"{cid}.jsonl", lines)
        elif kind == "frame_time":
            j = next(k for k, ev in enumerate(lines) if ev["ev"] == "frame" and ev["out_frame"] == n)
            lines[j]["time"] += 1
            _write_jsonl(rw / f"{cid}.jsonl", lines)
        elif kind == "frame_dropped":
            pl_lines, _ = load_jsonl(pl / f"{cid}.jsonl")
            _write_jsonl(pl / f"{cid}.jsonl", [ev for ev in pl_lines if ev.get("out_frame") != n])
    order_case = None
    for cid, lines, per in usable[5:]:
        for n in sorted(per):
            if len({(e["layer"], e["time"]) for _, e in per[n]}) >= 2:
                order_case = (cid, n)
                break
        if order_case:
            break
    if order_case:
        cid, n = order_case
        pl_lines, _ = load_jsonl(pl / f"{cid}.jsonl")
        js = [k for k, ev in enumerate(pl_lines) if ev.get("param_index") == 0 and ev.get("out_frame") == n]
        a_ = js[0]
        b_ = next(k for k in js[1:] if pl_lines[k]["time"] != pl_lines[a_]["time"])
        pl_lines[a_], pl_lines[b_] = pl_lines[b_], pl_lines[a_]
        _write_jsonl(pl / f"{cid}.jsonl", pl_lines)
    busy = {v[0] for v in targets.values()} | {(order_case or (None,))[0]}
    anomaly_case = next(c for c in ids_all if c not in busy)
    pl_lines, _ = load_jsonl(pl / f"{anomaly_case}.jsonl")
    j = next(k for k, ev in enumerate(pl_lines) if ev.get("param_index") == 3)
    pl_lines[j]["time"] += 5
    _write_jsonl(pl / f"{anomaly_case}.jsonl", pl_lines)
    cd = calls(pe, pl, rw)
    by = {m["id"]: m for m in cd["mismatches"]}
    ck("call records (sets per output frame): the edited cases are the mismatches", cd["compared"] == len(cases) and cd["matching"] == len(cases) - len(targets),
       f"compared {cd['compared']}, matching {cd['matching']} (expected {len(cases) - len(targets)})")
    for kind, (cid, n) in targets.items():
        fm = (by.get(cid) or {}).get("first_mismatch") or {}
        kinds = [p["kind"] for p in fm.get("problems", [])]
        want_kind = {"frame_dropped": "frame_missing_in_plugin_record"}.get(kind, kind)
        ck(f"  {kind}: first differing output frame is {n}", fm.get("out_frame") == n and want_kind in kinds, f"got frame {fm.get('out_frame')} kinds {kinds}")
    ck("call records: a parameter sampled off the frame time is not a mismatch", anomaly_case not in by, anomaly_case)
    ca = calls(pe, pl, rw, ordered=True)
    if order_case:
        o = {m["id"]: m for m in ca["mismatches"]}.get(order_case[0])
        ck("call records: a swapped order is a mismatch only with --ordered", o is not None and order_case[0] not in by and ca["matching"] == cd["matching"] - 1,
           f"{order_case[0]} frame {order_case[1]}, matching {ca['matching']} against {cd['matching']}")
    else:
        ck("call records: ordered check", True, "skipped: no output frame with two different reads in this module")
    report["call_records"] = {"compared": cd["compared"], "matching": cd["matching"], "ordered_matching": ca["matching"],
                              "targets": {k: list(v) for k, v in targets.items()}}

    for big in (refs_b, refs_c, rw, pl):
        shutil.rmtree(big, ignore_errors=True)
    after = {p: sha_file(p) for p in guard}
    ck("phase A files unchanged", after == guard, f"{len(guard)} files checked")
    report["checks"] = ck.items
    report["failed"] = [i["check"] for i in ck.failed]
    jdump(st / "report.json", report)
    log(f"selftest: {len(ck.items) - len(ck.failed)} of {len(ck.items)} checks ok; report in {st / 'report.json'}")
    return 0 if not ck.failed else 1


# ------------------------------------------------------------------------------------------ CLI


def common_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--root", help="module root (default: this file's folder, or the folder above tools/)")
    p.add_argument("--results", help="where results are written (default <root>/results)")
    p.add_argument("--phase-a", dest="phase_a", help="where the phase A tables are read (default <root>/results)")
    p.add_argument("--testsets", help="folder with grid/ and draft/ (default <root>/testsets)")
    p.add_argument("--testset", action="append", help="one test set directory (repeatable); replaces grid and draft")
    p.add_argument("--cpus", help="pin this process and its children to these CPUs (Linux), e.g. 3")
    return p


def build_parser():
    ap = argparse.ArgumentParser(prog="phase_b.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = common_parser()
    g = sub.add_parser("grade", parents=[c], help="fx-parity run --max-ulp 0 over both test sets")
    g.add_argument("--refs", required=True)
    g.add_argument("--vfx", help="rewrite binary (default <root>/bin/vfx-base)")
    g.add_argument("--fx-parity", dest="fxp")
    g.add_argument("--draft-vfx", dest="draft_vfx", help="wrapper that appends --quality draft (default: generated)")
    g.add_argument("--jobs", type=int, default=1)
    g.add_argument("--no-exact", action="store_true", help="skip the value by value analysis of differing cases")
    o = sub.add_parser("ooc", parents=[c], help="move cases the original refuses out of contract")
    o.add_argument("--status", action="append", required=True, help="harness status file (run_harness.py --summary JSON or TSV)")
    o.add_argument("--faults-out-of-contract", action="store_true", help="also move cases that died of an emulator fault")
    o.add_argument("--param-map", help="param_map.tsv: list cases outside the registered slider ranges")
    o.add_argument("--plugin", default="Twixtor_64")
    o.add_argument("--apply-ranges", action="store_true", help="move the range candidates out of contract too")
    k = sub.add_parser("calls", parents=[c], help="compare plugin call records with the rewrite's")
    k.add_argument("--plugin-records", required=True)
    k.add_argument("--rewrite-records", required=True)
    k.add_argument("--param-map", help="param_map.tsv (to tell layer params from other params)")
    k.add_argument("--plugin", default="Twixtor_64")
    k.add_argument("--ordered", action="store_true", help="order of the reads matters")
    k.add_argument("--include-ooc", action="store_true")
    r = sub.add_parser("records", parents=[c], help="run the rewrite with VFX_CALL_RECORD over the whole test set")
    r.add_argument("--vfx")
    r.add_argument("--out", required=True)
    m = sub.add_parser("kills", parents=[c], help="mutation kills against the reference renders")
    m.add_argument("--refs", required=True)
    m.add_argument("--mode", choices=("tables", "rerender"), default="tables",
                   help="tables: the saved phase A sha256 tables; rerender: render with bin/vfx-base and bin/vfx-mNN")
    m.add_argument("--bin-dir")
    m.add_argument("--only", help="comma separated mutant ids")
    s = sub.add_parser("summary", parents=[c], help="write summary.json from the result files")
    s.add_argument("--date")
    s.add_argument("--plugin-sha256")
    t = sub.add_parser("triage", parents=[c], help="group the differing cases and write a ladder of probe cases")
    t.add_argument("--vfx")
    t.add_argument("--probe-dir")
    sub.add_parser("selftest", parents=[c], help="stand-in oracle checks, writes under results/selftest/")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.command = command_text() if argv is None else "python3 tools/phase_b.py " + " ".join(shlex.quote(x) for x in argv)
    if a.cpus and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {int(x) for x in a.cpus.split(",")})
    ctx = Ctx(a)
    try:
        if a.cmd == "grade":
            rows, cmds = grade(ctx, a.refs, a.vfx, a.fxp, a.draft_vfx, a.jobs, exact=not a.no_exact)
            doc = write_grade(ctx, rows, cmds, a.refs)
            print(json.dumps(doc["totals"], indent=1))
            print(f"wrote {ctx.results / 'grade.json'} and grade.tsv")
            return 1 if doc["totals"]["differing"] else 0
        if a.cmd == "ooc":
            doc = ooc(ctx, a.status, a.faults_out_of_contract, a.param_map, a.plugin, a.apply_ranges)
            print(json.dumps(doc["counts"], indent=1))
            for m in doc["moved"]:
                print(f"  out of contract: {m['id']}: {m['reason']}")
            for c in doc["candidates"]:
                if c["verdict"] == "plugin_renders_it":
                    print(f"  ATTENTION: the original renders candidate {c['id']} ({c['label']}) that the rewrite refuses")
            return 0
        if a.cmd == "calls":
            doc = calls(ctx, a.plugin_records, a.rewrite_records, a.param_map, a.plugin, a.ordered, not a.include_ooc)
            print(f"call records compared {doc['compared']}, matching {doc['matching']}, skipped {doc['skipped']}")
            for m in doc["mismatches"][:20]:
                fm = m["first_mismatch"]
                print(f"  {m['id']}: first differing output frame {fm['out_frame'] if fm else '?'}: "
                      + "; ".join(p["kind"] for p in (fm["problems"] if fm else [])))
            return 0 if doc["compared"] and doc["matching"] == doc["compared"] else 1
        if a.cmd == "records":
            first = next(iter(load_cases(ctx).values()))
            record_rewrite(ctx, find_record_vfx(ctx, a.vfx, first), a.out)
            return 0
        if a.cmd == "kills":
            doc = kills(ctx, a.refs, a.mode, a.bin_dir, set(a.only.split(",")) if a.only else None)
            t = doc["totals"]
            print(f"mutants {t['total']}, killed {t['killed']}, survivors {t['survivors']}, not run {t['not_run']}; "
                  f"kill candidates {doc['cases']['kill_candidates']} of {doc['cases']['in_test_sets']}")
            return 0 if not t["survivors"] else 1
        if a.cmd == "summary":
            s, unmet = summary(ctx, a.date, a.plugin_sha256)
            print(json.dumps({k: s[k] for k in ("status", "blockers", "cases", "values", "call_records", "mutants")}, indent=1))
            print("rule not met: " + "; ".join(unmet) if unmet else "status rule met")
            return 0
        if a.cmd == "triage":
            triage(ctx, a.vfx, a.probe_dir)
            return 0
        if a.cmd == "selftest":
            return selftest(ctx)
    except ToolError as e:
        print(f"phase_b: error: {e}", file=sys.stderr)
        return 3
    return 2


if __name__ == "__main__":
    sys.exit(main())
