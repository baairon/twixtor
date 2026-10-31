#!/usr/bin/env python3
"""The smoke test: one small case per plugin (default params, 8 bpc, 64x48, a few frames) rendered under
emulation and graded with `fx-parity run --max-ulp 0 --json`. Writes results/smoke.json.

usage: smoke.py --testset DIR --fx-parity PATH --vfx PATH [--out results/smoke.json] [--jobs 4]
The test set comes from: fx-parity gen --out DIR --effects retime,retime-fgbg,create-vectors,vectors-in
  --depths 8 --no-sweep --sizes 64x48 --frames 4 --fps 24 --clip-sources testsrc2 --max-cases-per-effect 4
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLUGIN_OF = {"retime": "Twixtor_64", "retime-fgbg": "TwixtorPro_64", "create-vectors": "CreateMotionVectors_64",
             "vectors-in": "TwixtorProVectorsIn_64"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", required=True)
    ap.add_argument("--fx-parity", required=True)
    ap.add_argument("--vfx", required=True)
    ap.add_argument("--out", default=str(HERE / "results" / "smoke.json"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--work", default="smoke_work")
    a = ap.parse_args()
    ts = Path(a.testset)
    cases = json.loads((ts / "cases.json").read_text())["cases"]
    picked = {}
    for c in cases:
        if c["effect"] in PLUGIN_OF and c["effect"] not in picked and c["label"] == "all-default" and c["depth"] == 8:
            picked[c["effect"]] = c
    work = Path(a.work)
    refs = work / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    for f in refs.iterdir():
        f.unlink()
    summary = work / "run_summary.json"
    cmd = [sys.executable, str(HERE / "run_harness.py"), "--testset", a.testset, "--refs", str(refs), "-j", str(a.jobs),
           "--records", str(work / "records"), "--summary", str(summary)]
    for c in picked.values():
        cmd += ["--case", c["id"]]
    subprocess.run(cmd, check=False)
    run = json.loads(summary.read_text()) if summary.exists() else {"cases": [], "boots": []}
    grade = subprocess.run([a.fx_parity, "run", "--testset", a.testset, "--refs", str(refs), "--vfx", a.vfx,
                            "--max-ulp", "0", "--json"], capture_output=True, text=True)
    report = json.loads(grade.stdout) if grade.stdout.strip() else {}
    by_id = {c["id"]: c for c in report.get("cases", [])}
    out = {"max_ulp": 0, "fx_parity_exit": grade.returncode, "fx_parity_summary": report.get("summary"), "plugins": {}}
    for eff, c in picked.items():
        r = next((x for x in run["cases"] if x.get("id") == c["id"]), {})
        g = by_id.get(c["id"], {})
        boot = next((b for b in run["boots"] if b.get("plugin") == PLUGIN_OF[eff]), {})
        out["plugins"][PLUGIN_OF[eff]] = {
            "effect": eff, "case": c["id"], "size": f"{c['width']}x{c['height']}", "depth": c["depth"], "frames": c["frames"],
            "params": "defaults",
            "rendered": r.get("status") == "rendered",
            "boot": {k: boot.get(k) for k in ("ok", "error", "last_api", "pc", "faults", "messages", "seconds", "init_errors")},
            "error": None if r.get("status") == "rendered" else (r.get("error") or boot.get("error")),
            "seconds_per_frame": r.get("seconds_per_frame"),
            "grade": {"status": g.get("status"), "differing_pixels": g.get("clip_differing_pixels", g.get("differing_pixels")),
                      "failing_frames": g.get("failing_frames"), "max_ulp": g.get("max_ulp")},
        }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    main()
