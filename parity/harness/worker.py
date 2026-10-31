#!/usr/bin/env python3
"""One emulator process: boots one plugin, then renders the cases it is given, one after the other.

Prints one JSON object per line on stdout: {"event":"boot",...}, then for each case {"event":"start"}
and {"event":"done"} with the outcome. The controller (run_harness.py) restarts a worker that dies.

usage: worker.py --plugin Twixtor_64 --rootfs DIR --testset DIR --refs DIR --records DIR
                 --param-map param_map.tsv --cases ID[,ID...] [--cases-file FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ae_host import GuestFault, Layouts, Record  # noqa: E402
from ae_mock import Mock, PluginError, BPP, wire_to_world  # noqa: E402
from keys import Unsupported, make_value  # noqa: E402

PIX_FMT = {8: "rgba", 16: "rgba64le", 32: "gbrapf32le"}


def out(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def load_param_map(path: Path, plugin: str) -> dict:
    rows = {}
    lines = path.read_text().splitlines()
    cols = lines[0].split("\t")
    for line in lines[1:]:
        r = dict(zip(cols, line.split("\t")))
        if r.get("plugin") == plugin and r.get("vfx_id") and r.get("ae_index"):
            rows[r["vfx_id"]] = r
    return rows


def split_clip(path: Path, frame_len: int, count: int) -> list[bytes]:
    data = path.read_bytes()
    if len(data) != frame_len * count:
        raise ValueError(f"{path} has {len(data)} bytes, {count} frames of {frame_len} are {frame_len * count}")
    return [data[i * frame_len:(i + 1) * frame_len] for i in range(count)]


def run_case(m: Mock, case: dict, testset: Path, refs: Path, pmap: dict) -> dict:
    depth, w, h, n = case["depth"], case["width"], case["height"], case["frames"]
    res = {"id": case["id"], "effect": case["effect"], "frames": n}
    # After Effects hands a plugin a 16 or 32 bpc world only when GLOBAL_SETUP declared it aware (kept in global_out_flags*)
    if depth == 16 and not m.global_out_flags & m.lay.e("PF_OutFlag_DEEP_COLOR_AWARE"):
        raise Unsupported(f"{m.plugin_name} does not declare PF_OutFlag_DEEP_COLOR_AWARE: no 16 bpc world")
    if depth == 32 and not m.global_out_flags2 & m.lay.e("PF_OutFlag2_FLOAT_COLOR_AWARE"):
        raise Unsupported(f"{m.plugin_name} does not declare PF_OutFlag2_FLOAT_COLOR_AWARE: no 32 bpc world")
    frame_len = w * h * BPP[depth]
    fps = case.get("fps") or "24"
    m.configure(depth, w, h, fps, n)
    m.case_id = case["id"]
    m.quality_name = case.get("quality") or "high"       # "draft" is PF_Quality_LO in PF_InData.quality, anything else PF_Quality_HI
    # the input clip and the layers, converted to AE worlds
    clip = split_clip(testset / case["input"], frame_len, n)
    m.inputs = {"input": [wire_to_world(depth, w, h, f) for f in clip]}
    for lid, file in (case.get("layers") or {}).items():
        row = pmap.get(lid)
        if row is None or row.get("vfx_kind") != "layer":
            raise Unsupported(f"layer `{lid}` has no registered layer param in {m.plugin_name}")
        data = (testset / file).read_bytes()
        count = len(data) // frame_len
        m.inputs[int(row["ae_index"])] = [wire_to_world(depth, w, h, f) for f in split_clip(testset / file, frame_len, count)]
    # an unset layer param whose registered default is PF_LayerDefault_MYSELF (AE_Effect.h) is the layer the effect is
    # applied to, which is the case's input clip; a registered default of PF_LayerDefault_NONE stays unset (no layer)
    myself = m.lay.e("PF_LayerDefault_MYSELF")
    for r in m.reg_params:
        if r["type_name"] == "LAYER" and r.get("default") == myself and r["reg_index"] not in m.inputs:
            m.inputs[r["reg_index"]] = m.inputs["input"]
            m.rec.log("default_layer", param_index=r["reg_index"], name=r["name"], layer="input")
    # param values
    for r in m.reg_params:
        m.param_values[r["reg_index"]] = None
    for item in case.get("params") or []:
        pid, _, text = item.partition("=")
        row = pmap.get(pid)
        if row is None:
            raise Unsupported(f"param `{pid}` has no counterpart among the registered params of {m.plugin_name}")
        m.param_values[int(row["ae_index"])] = make_value(row["vfx_kind"], text, m.tb.time_step)
    t0 = time.perf_counter()
    frames = []
    for i, wire in enumerate(m.sequence()):
        frames.append(wire)
        res.setdefault("frame_seconds", []).append(round(time.perf_counter() - t0, 3))
    seconds = time.perf_counter() - t0
    path = refs / case["reference"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(frames))
    res.update(status="rendered", seconds=round(seconds, 3), seconds_per_frame=round(seconds / max(n, 1), 3), reference=str(path))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", required=True)
    ap.add_argument("--rootfs", required=True)
    ap.add_argument("--testset", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--param-map", default=None)
    ap.add_argument("--cases", default="")
    ap.add_argument("--cases-file")
    ap.add_argument("--plugins-dir")
    a = ap.parse_args()
    testset, refs, records = Path(a.testset), Path(a.refs), Path(a.records)
    records.mkdir(parents=True, exist_ok=True)
    ts = json.loads((testset / "cases.json").read_text())
    ids = [x for x in a.cases.split(",") if x]
    if a.cases_file:
        ids += [x for x in Path(a.cases_file).read_text().split() if x]
    cases = {c["id"]: c for c in ts["cases"]}
    rootfs = Path(a.rootfs)
    plugin_path = str((Path(a.plugins_dir) if a.plugins_dir else rootfs / "plugins") / f"{a.plugin}.aex")
    lay = Layouts()
    rec = Record(str(records / f"_boot_{a.plugin}.jsonl"))
    t0 = time.perf_counter()
    try:
        m = Mock(plugin_path, str(rootfs), rec, lay)
        m.plugin_name = a.plugin
        m.boot()
        load_s = time.perf_counter() - t0
        m.setup_plugin()
    except (GuestFault, PluginError, Exception) as e:  # noqa: BLE001
        out({"event": "boot", "ok": False, "plugin": a.plugin, "error": f"{type(e).__name__}: {e}",
             "last_api": getattr(e, "last", None), "pc": hex(e.pc) if getattr(e, "pc", None) else None,
             "faults": [{k: (hex(v) if isinstance(v, int) and k != "size" else v) for k, v in f.items()} for f in getattr(locals().get("m"), "faults", [])][:5],
             "messages": getattr(locals().get("m"), "messages", []), "seconds": round(time.perf_counter() - t0, 1),
             "trace": traceback.format_exc().splitlines()[-6:]})
        rec.close()
        return 2
    rec.close()
    out({"event": "boot", "ok": True, "plugin": a.plugin, "seconds": round(time.perf_counter() - t0, 1), "load_seconds": round(load_s, 1),
         "entry": m.entry_name, "params": len(m.reg_params), "smart_render": m.is_smart(),
         "out_flags": hex(m.global_out_flags & 0xFFFFFFFF), "out_flags2": hex(m.global_out_flags2 & 0xFFFFFFFF),
         "init_errors": m.init_errors, "messages": m.messages})
    pm_path = Path(a.param_map) if a.param_map else (HERE / f"param_map_{a.plugin}.tsv" if (HERE / f"param_map_{a.plugin}.tsv").exists() else HERE / "param_map.tsv")
    pmap = load_param_map(pm_path, a.plugin) if pm_path.exists() else {}
    for cid in ids:
        case = cases[cid]
        out({"event": "start", "id": cid})
        rec = Record(str(records / f"{cid}.jsonl"))
        m.rec = rec
        m.faults.clear()
        m.messages.clear()
        try:
            res = run_case(m, case, testset, refs, pmap)
        except Unsupported as e:
            res = {"id": cid, "status": "unsupported", "error": str(e)}
        except PluginError as e:
            # a returned PF_Err is the plugin's answer, not a crash: the sequence was torn down (SEQUENCE_SETDOWN) and the
            # worker goes on with the next case; see README, "PluginError does not restart the worker"
            res = {"id": cid, "status": "error", "error": f"{type(e).__name__}: {e}", "last_api": None, "last_callback": m.last_cb,
                   "pc": None, "faults": [], "messages": list(m.messages)}
            rec.log("case_error", **res)
            rec.close()
            out({"event": "done", **res})
            continue
        except GuestFault as e:
            res = {"id": cid, "status": "error", "error": f"{type(e).__name__}: {e}", "last_api": getattr(e, "last", None) or m.last_api,
                   "last_callback": m.last_cb, "pc": hex(e.pc) if getattr(e, "pc", None) else None,
                   "faults": [{k: (hex(v) if isinstance(v, int) and k != "size" else v) for k, v in f.items()} for f in m.faults][:5],
                   "messages": list(m.messages)}
            rec.log("case_error", **res)
            rec.close()
            out({"event": "done", **res})
            return 3            # the emulator state after a fault is not trusted: the controller restarts
        except Exception as e:  # noqa: BLE001
            res = {"id": cid, "status": "error", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc().splitlines()[-6:]}
        res["messages"] = list(m.messages)
        rec.log("case_done", **{k: v for k, v in res.items() if k != "frame_seconds"})
        rec.close()
        out({"event": "done", **res})
    return 0


if __name__ == "__main__":
    sys.exit(main())
