#!/usr/bin/env python3
"""Boots one plugin and writes what it registers in PARAMS_SETUP (and its flags) as JSON.

usage: dump_params.py PLUGIN_NAME ROOTFS OUT.json [CALLS.jsonl]
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ae_host import GuestFault, Layouts, Record  # noqa: E402
from ae_mock import Mock, PluginError  # noqa: E402

name, rootfs, out = sys.argv[1], sys.argv[2], sys.argv[3]
rec = Record(sys.argv[4] if len(sys.argv) > 4 else None)
t0 = time.time()
m = Mock(f"{rootfs}/plugins/{name}.aex", rootfs, rec, Layouts())
m.plugin_name = name
result = {"plugin": name, "ok": False}
try:
    m.boot()
    m.setup_plugin()
    result.update(ok=True, entry=m.entry_name, out_flags=m.global_out_flags & 0xFFFFFFFF, out_flags2=m.global_out_flags2 & 0xFFFFFFFF,
                  smart_render=m.is_smart(), num_params=m.out_num_params,
                  params=[{k: v for k, v in d.items() if k != "guest"} for d in m.reg_params])
except (GuestFault, PluginError, Exception) as e:  # noqa: BLE001
    result.update(error=f"{type(e).__name__}: {e}", last_api=getattr(e, "last", None), last_cb=m.last_cb,
                  faults=m.faults[:5], messages=m.messages, init_errors=m.init_errors)
result["seconds"] = round(time.time() - t0, 1)
result["unimplemented"] = sorted(m.unimplemented)
result["api_top"] = m.api_counts.most_common(25)
result["api_ring"] = list(m.api_ring)
Path(out).write_text(json.dumps(result, indent=1, default=str))
rec.close()
print(json.dumps({k: v for k, v in result.items() if k not in ("params", "unimplemented")}, default=str)[:3500])
print(len(result.get("params", [])), "params")
