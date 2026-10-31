#!/usr/bin/env python3
"""Renders an fx-parity test set under emulation and writes the references that `fx-parity run` grades.

usage: run_harness.py --testset DIR --refs DIR [-j N] [--effect ID] [--max-cases N] [--case ID ...]
                      [--records DIR] [--summary FILE] [--rootfs DIR]

Each case of cases.json whose effect is one of the four plugins is rendered by a worker process
(worker.py) that boots the plugin once and renders its share of the cases one after the other;
`-j N` runs N workers at a time (about 1.5 GB each). A worker that dies (a fault in the emulation)
is replaced and the cases after the one that killed it carry on. The result of every case goes to
the summary JSON; a case that did not render leaves no file in refs/, so `fx-parity run` reports it
as `no reference`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

PLUGIN_OF = {
    "retime": "Twixtor_64",
    "retime-fgbg": "TwixtorPro_64",
    "create-vectors": "CreateMotionVectors_64",
    "vectors-in": "TwixtorProVectorsIn_64",
}


# every restart drops the case that killed the worker, so the loop always progresses; the limit only stops a runaway
MAX_RESTARTS = 500


def param_map_for(plugin: str) -> str:
    own = HERE / f"param_map_{plugin}.tsv"
    return str(own if own.exists() else HERE / "param_map.tsv")


def partial(args, ev):
    """append every finished case to SUMMARY.partial.jsonl at once, so a killed run keeps its status."""
    if args.summary:
        with open(args.summary + ".partial.jsonl", "a") as pf:
            pf.write(json.dumps(ev) + "\n")


def drive(plugin: str, case_ids: list[str], args, log, results: dict, lock: threading.Lock, sem: threading.Semaphore):
    with sem:
        _drive(plugin, case_ids, args, log, results, lock)


def _drive(plugin: str, case_ids: list[str], args, log, results: dict, lock: threading.Lock):
    """Runs the cases of one plugin on one worker, restarting it after a death."""
    todo = list(case_ids)
    restarts = 0
    while todo:
        cmd = [sys.executable, "-I", str(HERE / "worker.py"), "--plugin", plugin, "--rootfs", args.rootfs,
               "--testset", args.testset, "--refs", args.refs, "--records", args.records,
               "--param-map", args.param_map or param_map_for(plugin), "--cases", ",".join(todo)]
        if args.plugins_dir:
            cmd += ["--plugins-dir", args.plugins_dir]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        current = None
        booted = False
        timer = None
        timed_out = []

        def kill_slow(cid, p=proc):            # a case that runs past --case-timeout ends the worker
            timed_out.append(cid)
            p.kill()

        for line in proc.stdout:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev["event"] == "boot":
                booted = ev.get("ok", False)
                with lock:
                    results.setdefault("boots", []).append(ev)
                log(f"{plugin}: boot {'ok' if ev.get('ok') else 'FAILED: ' + str(ev.get('error'))} ({ev.get('seconds')} s)")
            elif ev["event"] == "start":
                current = ev["id"]
                if args.case_timeout:
                    timer = threading.Timer(args.case_timeout, kill_slow, args=(current,))
                    timer.start()
            elif ev["event"] == "done":
                if timer:
                    timer.cancel()
                    timer = None
                with lock:
                    results.setdefault("cases", {})[ev["id"]] = ev
                    partial(args, ev)
                if ev["id"] in todo:
                    todo.remove(ev["id"])
                log(f"{plugin}: {ev['id']} {ev['status']}" + (f" {ev.get('seconds_per_frame')} s/frame" if ev["status"] == "rendered" else f": {ev.get('error')}"))
                current = None
        proc.wait()
        if timer:
            timer.cancel()
        if timed_out and current == timed_out[-1] and current in todo:
            todo.remove(current)
            ev = {"id": current, "status": "timeout", "error": f"no result within {args.case_timeout} s (--case-timeout)"}
            with lock:
                results.setdefault("cases", {})[current] = ev
                partial(args, ev)
            log(f"{plugin}: {current} timeout")
            current = None
        if not booted:
            for cid in todo:
                with lock:
                    results.setdefault("cases", {})[cid] = {"id": cid, "status": "not_run", "error": "the plugin did not boot"}
            return
        if todo:
            restarts += 1
            if current in todo:
                todo.remove(current)
                with lock:
                    results.setdefault("cases", {}).setdefault(current, {"id": current, "status": "error", "error": "the worker died"})
                    partial(args, results["cases"][current])
            if restarts > MAX_RESTARTS:
                for cid in todo:
                    with lock:
                        results.setdefault("cases", {})[cid] = {"id": cid, "status": "not_run", "error": "too many worker restarts"}
                log(f"{plugin}: {len(todo)} cases not run: too many worker restarts")
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("-j", "--jobs", type=int, default=1)
    ap.add_argument("--effect")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--case", action="append", default=[])
    ap.add_argument("--records", default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--rootfs", default=str(HERE / "rootfs" / "x8664_windows"))
    ap.add_argument("--param-map", default=None, help="one map for every plugin; default: param_map_<plugin>.tsv next to this file, else param_map.tsv")
    ap.add_argument("--plugins-dir")
    ap.add_argument("--cases-file", help="case ids, blank separated; run in this order")
    ap.add_argument("--case-timeout", type=float, default=0, help="seconds per case before the worker is killed (0: off)")
    args = ap.parse_args()
    if args.cases_file:
        args.case += Path(args.cases_file).read_text().split()
    args.records = args.records or str(Path(args.refs).parent / "records")
    ts = json.loads((Path(args.testset) / "cases.json").read_text())
    cases = [c for c in ts["cases"] if c["effect"] in PLUGIN_OF and c.get("kind", "frame") == "temporal"]
    if args.effect:
        cases = [c for c in cases if c["effect"] == args.effect]
    if args.case:
        order = {cid: i for i, cid in enumerate(args.case)}
        cases = sorted((c for c in cases if c["id"] in order), key=lambda c: order[c["id"]])
    if args.max_cases:
        cases = cases[: args.max_cases]
    Path(args.refs).mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    results: dict = {}
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    # split each effect's cases over the workers: -j N workers in all, at least one per effect
    by_effect: dict[str, list[str]] = {}
    for c in cases:
        by_effect.setdefault(c["effect"], []).append(c["id"])
    sem = threading.Semaphore(max(1, args.jobs))
    shares = max(1, args.jobs // max(len(by_effect), 1))
    threads = []
    for eff, ids in by_effect.items():
        parts = [ids[i::shares] for i in range(shares) if ids[i::shares]]
        for part in parts:
            t = threading.Thread(target=drive, args=(PLUGIN_OF[eff], part, args, log, results, lock, sem))
            t.start()
            threads.append(t)
    for t in threads:
        t.join()
    summary = {"testset": args.testset, "refs": args.refs, "seconds": round(time.time() - t0, 1), "jobs": args.jobs,
               "cases": [results.get("cases", {}).get(c["id"], {"id": c["id"], "status": "not_run"}) for c in cases],
               "boots": results.get("boots", [])}
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=1))
    ok = sum(1 for c in summary["cases"] if c["status"] == "rendered")
    log(f"{ok} of {len(cases)} cases rendered")
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
