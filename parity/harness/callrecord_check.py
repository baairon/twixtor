#!/usr/bin/env python3
"""Renders cases of an fx-parity test set with and without VFX_CALL_RECORD and compares the SHA-256 of
the output (and of stderr). The record must not change a single byte.

usage: callrecord_check.py --vfx PATH --testset DIR [--per-effect N] [--out FILE] [--records DIR]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def args_of(case, testset: Path) -> list[str]:
    a = ["apply", case["effect"], "--width", str(case["width"]), "--height", str(case["height"]), "--depth", str(case["depth"])]
    for p in case["params"]:
        a += ["--param", p]
    a += ["--fps", case["fps"], "--frames", str(case["frames"])]
    for lid, f in (case.get("layers") or {}).items():
        a += ["--layer", f"{lid}={testset / f}"]
    a += ["--in", str(testset / case["input"])]
    return a


def run(vfx, args, record=None):
    env = dict(os.environ)
    env.pop("VFX_CALL_RECORD", None)
    if record:
        env["VFX_CALL_RECORD"] = str(record)
    p = subprocess.run([vfx, *args], capture_output=True, env=env)
    return p.returncode, hashlib.sha256(p.stdout).hexdigest(), hashlib.sha256(p.stderr).hexdigest(), len(p.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vfx", required=True)
    ap.add_argument("--testset", required=True)
    ap.add_argument("--per-effect", type=int, default=3)
    ap.add_argument("--out")
    ap.add_argument("--records", default="callrecord_logs")
    a = ap.parse_args()
    ts = Path(a.testset)
    cases = json.loads((ts / "cases.json").read_text())["cases"]
    rec_dir = Path(a.records)
    rec_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, int] = {}
    rows = []
    for c in cases:
        if c.get("kind") != "temporal" or seen.get(c["effect"], 0) >= a.per_effect:
            continue
        seen[c["effect"]] = seen.get(c["effect"], 0) + 1
        args = args_of(c, ts)
        log = rec_dir / f"{c['id']}.jsonl"
        plain = run(a.vfx, args)
        logged = run(a.vfx, args, log)
        lines = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
        rows.append({
            "id": c["id"], "effect": c["effect"], "depth": c["depth"], "layers": sorted((c.get("layers") or {})),
            "exit": [plain[0], logged[0]], "stdout_sha256": plain[1], "stdout_sha256_with_record": logged[1],
            "stdout_equal": plain[1] == logged[1], "stderr_equal": plain[2] == logged[2], "bytes": plain[3],
            "record_lines": len(lines), "record_reads": sum(1 for l in lines if l["ev"] == "read"),
            "record_layers": sorted({l["layer"] for l in lines if l["ev"] == "read"}),
        })
    ok = all(r["stdout_equal"] and r["stderr_equal"] and r["exit"][0] == r["exit"][1] for r in rows)
    doc = {"ok": ok, "cases": rows}
    text = json.dumps(doc, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
