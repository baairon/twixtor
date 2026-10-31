#!/usr/bin/env python3
"""How fast does the emulator run GLOBAL_SETUP, with and without the capture script's tracing?

Boots Twixtor_64 the way worker.py does (Mock, boot()), then runs GLOBAL_SETUP for a fixed number of seconds
and reports what it got through. Modes:
  --mode plain     the worker path, no tracing
  --mode capture   the capture script's hooks (capture_msvcr120.py: hook_winapi replacement, Rtl and API record, range hooks)
  --blocks         also count executed basic blocks and their bytes with one block hook (a coarse counter:
                   Unicorn has no instruction counter; instructions are estimated as bytes / 4.3)
Progress is compared by the number of host API calls the harness answered (Emu.api_counts), which is the same
sequence in every mode. Nothing of the plugin is read or logged.

usage: speed_probe.py --mode plain|capture [--blocks] --seconds N --rootfs DIR --out FILE.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ae_host import GuestFault, Layouts, Record  # noqa: E402
from ae_mock import Mock  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("plain", "capture"), required=True)
    ap.add_argument("--blocks", action="store_true")
    ap.add_argument("--seconds", type=float, default=240)
    ap.add_argument("--rootfs", default=str(HERE / "rootfs" / "x8664_windows"))
    ap.add_argument("--plugin", default="Twixtor_64")
    ap.add_argument("--profile", action="store_true", help="cProfile the GLOBAL_SETUP run and write the top functions next to --out")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plugin = str(Path(a.rootfs) / "plugins" / f"{a.plugin}.aex")
    res = {"mode": a.mode, "blocks_counted": a.blocks, "seconds_allowed": a.seconds, "pid": os.getpid()}
    t0 = time.perf_counter()
    if a.mode == "capture":
        import capture_msvcr120 as cap

        cap.install_qiling_patches()
        cap.CMD_TIMEOUT = a.seconds
        m = cap.CapMock(plugin, a.rootfs, cap.MemRecord(), Layouts())
        cap.ST.mock = m
        cap.ST.phase = "harness-boot"
    else:
        class PlainMock(Mock):
            def call(self, addr, args, label="", timeout=0):
                if len(args) == 3 and args[1] == 1:
                    return super().call(addr, args, label, timeout)
                return super().call(addr, args, label, a.seconds)

        m = PlainMock(plugin, a.rootfs, Record(None), Layouts())
    m.plugin_name = a.plugin
    m.boot()
    res["boot_seconds"] = round(time.perf_counter() - t0, 1)
    res["api_calls_at_boot_end"] = sum(m.api_counts.values())

    state = {"blocks": 0, "bytes": 0, "next": 1 << 16, "t": None}
    checkpoints = []

    def snap():
        checkpoints.append({"t": round(time.perf_counter() - state["t"], 1), "blocks": state["blocks"], "bytes": state["bytes"],
                            "api_calls": sum(m.api_counts.values()) - res["api_calls_at_boot_end"]})

    if a.blocks:
        def on_block(ql, address, size):
            state["blocks"] += 1
            state["bytes"] += size
            if state["blocks"] >= state["next"]:
                state["next"] += 1 << 16
                now = time.perf_counter() - state["t"]
                if not checkpoints or now - checkpoints[-1]["t"] >= 20:
                    snap()

        m.ql.hook_block(on_block)
    state["t"] = time.perf_counter()
    cpu0 = time.process_time()
    prof = None
    if a.profile:
        import cProfile

        prof = cProfile.Profile()
        prof.enable()
    try:
        err = m.cmd("GLOBAL_SETUP")
        res["global_setup"] = {"finished": True, "PF_Err": err}
    except GuestFault as ex:
        res["global_setup"] = {"finished": False, "stopped_by": str(ex)[:80]}
    if prof:
        prof.disable()
        import io
        import pstats

        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(22)
        Path(a.out).with_suffix(".profile.txt").write_text(buf.getvalue(), encoding="utf-8")
    res["global_setup_seconds"] = round(time.perf_counter() - state["t"], 1)
    res["global_setup_cpu_seconds"] = round(time.process_time() - cpu0, 1)
    res["api_calls_per_cpu_second"] = round((sum(m.api_counts.values()) - res["api_calls_at_boot_end"]) / max(time.process_time() - cpu0, 0.001), 2)
    res["api_calls_in_global_setup"] = sum(m.api_counts.values()) - res["api_calls_at_boot_end"]
    res["api_calls_per_second"] = round(res["api_calls_in_global_setup"] / max(res["global_setup_seconds"], 0.001), 2)
    if a.blocks:
        snap()
        res["blocks"] = state["blocks"]
        res["block_bytes"] = state["bytes"]
        res["blocks_per_second"] = round(state["blocks"] / res["global_setup_seconds"])
        res["blocks_per_cpu_second"] = round(state["blocks"] / max(res["global_setup_cpu_seconds"], 0.001))
        res["estimated_instructions"] = round(state["bytes"] / 4.3)
        res["estimated_instructions_per_second"] = round(state["bytes"] / 4.3 / res["global_setup_seconds"])
        res["checkpoints"] = checkpoints
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "checkpoints"}))


if __name__ == "__main__":
    main()
