#!/usr/bin/env python3
"""State capture for the msvcr120.dll start-up fault under the harness. Capture only, no fixes.

Boots one plugin through the existing Mock (ae_host.py / ae_mock.py, unchanged), adds tracing from the
outside by wrapping Qiling and Mock methods, and writes a JSON file and a short Markdown file.
It records state and metadata only: registers, stack words, the memory map, API calls and their
arguments, DllMain calls. It never reads, dumps or disassembles code bytes, and it never looks into the
plugin. The return-address chain is unwound from the images' exception tables (.pdata / .xdata, data).

usage (on the VPS, from a copy of the harness folder):
  python capture_msvcr120.py --plugin Twixtor_64 --rootfs rootfs/x8664_windows --out results/msvcr120_capture.json
"""
from __future__ import annotations

import argparse
import bisect
import collections
import functools
import hashlib
import json
import os
import platform
import struct
import sys
import time
import traceback
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ae_host import GuestFault, Layouts, NATIVE_DLLS, NATIVE_APIS, Record, WIN_STUBS  # noqa: E402
from ae_mock import Mock  # noqa: E402

T0 = time.perf_counter()

ST = types.SimpleNamespace(
    ql=None, phase="qiling-load", total=0, ring=collections.deque(maxlen=100), counts=collections.Counter(),
    cur=None, special=[], dm=[], dm_stack=[], loads=[], load_depth=0, maps=[], starts=[], unmapped=[], pdata={}, mock=None,
)

STOP_ON_RAISE = False
CMD_TIMEOUT = 0
DISABLED = {x for x in os.environ.get("CAP_DISABLE", "").split(",") if x}

ALLOC_STUBS = {"HeapAlloc", "HeapReAlloc", "HeapSize", "HeapFree", "malloc", "calloc", "realloc", "free", "_msize"}
SEMA_STUBS = {"CreateSemaphoreA", "CreateSemaphoreW", "ReleaseSemaphore", "WaitForSingleObject", "WaitForSingleObjectEx"}
EXIT_STUBS = {"ExitProcess", "exit", "_exit", "abort", "TerminateProcess", "FatalExit"}


def now():
    return round(time.perf_counter() - T0, 3)


def hx(v):
    return None if v is None else hex(v)


# ---------------------------------------------------------------------------------------------
# small readers

def rd(a, n):
    try:
        return bytes(ST.ql.mem.read(a, n))
    except Exception:  # noqa: BLE001
        return None


def rd64(a):
    b = rd(a, 8)
    return None if b is None else struct.unpack("<Q", b)[0]


def rd32(a):
    b = rd(a, 4)
    return None if b is None else struct.unpack("<I", b)[0]


def rd16(a):
    b = rd(a, 2)
    return None if b is None else struct.unpack("<H", b)[0]


def rd8(a):
    b = rd(a, 1)
    return None if b is None else b[0]


def where(a):
    """module+offset when the address is inside a loaded image, else region+offset, else None."""
    ql = ST.ql
    if ql is None or a is None or a == 0:
        return None
    img = ql.loader.find_containing_image(a)
    if img:
        return f"{os.path.basename(img.path)}+{a - img.base:#x}"
    for ent in ql.mem.map_info:
        s, e, lab = ent[0], ent[1], ent[3]
        if s <= a < e:
            return f"{lab}+{a - s:#x}"
    return None


def ascii_str(b):
    n = b.find(b"\0")
    if n < 0:
        n = len(b)
    s = b[:n]
    if len(s) >= 3 and all(0x20 <= c < 0x7F or c in (9, 10, 13) for c in s):
        return s.decode("ascii")
    return None


def utf16_str(b):
    n = 0
    while n + 1 < len(b) and not (b[n] == 0 and b[n + 1] == 0):
        n += 2
    s = b[:n]
    if len(s) >= 6 and all(0x20 <= s[i] < 0x7F and s[i + 1] == 0 for i in range(0, len(s), 2)):
        return s.decode("utf-16le")
    return None


def decode_value(v):
    """Value and where it points (module+offset or region). Strings are not read: the memory behind an
    argument can hold plugin data, which this capture never copies."""
    if v is None:
        return None
    d = {"hex": hex(v)}
    w = where(v) if v >= 0x10000 else None
    if w:
        d["at"] = w
    return d


def jsonable(o, depth=0):
    if isinstance(o, bool) or o is None or isinstance(o, str):
        return o
    if isinstance(o, int):
        return hex(o) if abs(o) > 0xFFFF else o
    if isinstance(o, float):
        return o
    if isinstance(o, (bytes, bytearray)):
        return bytes(o[:32]).hex()
    if depth > 3:
        return str(o)
    if isinstance(o, dict):
        return {str(k): jsonable(v, depth + 1) for k, v in list(o.items())[:16]}
    if isinstance(o, (list, tuple)):
        return [jsonable(v, depth + 1) for v in list(o)[:16]]
    return str(o)[:80]


# ---------------------------------------------------------------------------------------------
# API trace

def dm_top():
    return ST.dm_stack[-1] if ST.dm_stack else None


def begin_api(name, dll, addr):
    ql = ST.ql
    regs = ql.arch.regs
    sp = regs.read("rsp")
    ST.total += 1
    ctx = dm_top()
    ra = rd64(sp)
    raw = [regs.read(r) for r in ("rcx", "rdx", "r8", "r9")] + [rd64(sp + 0x28 + 8 * i) for i in range(4)]
    return {
        "n": ST.total, "phase": ST.phase, "t": now(), "dllmain_ctx": ctx["id"] if ctx else None,
        "api": name, "dll": dll, "api_addr": hex(addr), "return_addr": hx(ra), "called_from": where(ra),
        "args_raw": dict(zip(("rcx", "rdx", "r8", "r9", "[rsp+0x28]", "[rsp+0x30]", "[rsp+0x38]", "[rsp+0x40]"), [hx(x) for x in raw])),
        "_raw": raw,
    }


ARG_LABELS = ("rcx", "rdx", "r8", "r9", "[rsp+0x28]", "[rsp+0x30]", "[rsp+0x38]", "[rsp+0x40]")


def decode_entry(e):
    raw = e.get("_raw")
    if raw is None:
        return
    e["args_decoded"] = {ARG_LABELS[i]: decode_value(v) for i, v in enumerate(raw) if v is not None and v >= 0x10000 and where(v)}
    e["decoded_at"] = "record time" if e["phase"] != "qiling-load" else "capture time (the pointed-to memory may have changed since the call)"
    del e["_raw"]


def finish_api(e, ret, via):
    e["via"] = via
    e["ret"] = hx(ret)
    ST.counts[(e["phase"], e["api"])] += 1
    ctx = next((c for c in ST.dm if c["id"] == e["dllmain_ctx"]), None)
    if ctx is not None:
        ctx["api_calls"] += 1
        if len(ctx["api_names"]) < 400:
            ctx["api_names"].append(e["api"])
        ctx["api_tail"] = (ctx.get("api_tail", []) + [f"{e['n']}:{e['api']}:{e['ret']}:{via.split(' (')[0][:40]}"])[-15:]
    if e["phase"] != "qiling-load":
        decode_entry(e)
        if (e["api"] == "RtlPcToFileHeader" or via.startswith("harness patch") or e["api"] == "RaiseException") and len(ST.special) < 1000:
            ST.special.append({k: e.get(k) for k in ("n", "phase", "dllmain_ctx", "api", "dll", "args_raw", "ret", "via", "called_from", "args_decoded")})
    e.pop("_pc", None)
    ST.ring.append(e)


def install_qiling_patches():
    from unicorn import UcError
    from qiling import Qiling
    from qiling.const import QL_INTERCEPT, QL_STATE
    from qiling.exception import QlErrorSyscallError, QlErrorSyscallNotFound
    from qiling.loader.pe import Process, QlLoaderPE
    from qiling.os.memory import QlMemoryManager
    from qiling.os.os import QlOs
    from qiling.os.windows import const
    from qiling.os.windows.windows import QlOsWindows
    import qiling.os.windows.dlls as api

    # -- Qiling's per instruction API hook: the same logic, with a record of each API call
    def hook_winapi(self, ql, address, size):
        entry = ql.loader.import_symbols.get(address)
        if entry is None:
            return
        name = entry["name"]
        name = const.Mapper[entry["dll"]][entry["ordinal"]] if name is None else name.decode()
        user = self.user_defined_api[QL_INTERCEPT.CALL].get(name)
        api_func = user or getattr(api, f"hook_{name}", None)
        e = begin_api(name, entry["dll"], address)
        if api_func:
            ST.cur = e
            e["_pc"] = address
            try:
                api_func(ql, address, name)
            except Exception as ex:  # noqa: BLE001
                e["exception"] = f"{type(ex).__name__}: {ex}"
                finish_api(e, None, "Qiling Python hook (raised)")
                ql.log.exception(ex)
                ql.log.debug("%s Exception Found" % name)
                raise QlErrorSyscallError("Windows API Implementation Error")
            finally:
                ST.cur = None
            kind = "user_defined_api" if user else "Qiling Python hook"
            finish_api(e, ql.arch.regs.read("rax"), f"{kind} {e.get('qiling_func', getattr(api_func, '__name__', name))}")
        else:
            ql.log.warning(f'api {name} ({entry["dll"]}) is not implemented')
            finish_api(e, None, "no Qiling hook (the DLL's own code at that address runs; return value not captured here)")
            if ql.debug_stop:
                raise QlErrorSyscallNotFound("Windows API implementation not found")

    if "winapi" not in DISABLED:
        QlOsWindows.hook_winapi = hook_winapi

    # -- Qiling's fcall wrapper: decoded parameter names and values of the Python hooks
    orig_call = QlOs.call

    def os_call(self, pc, func, proto, onenter, onexit, passthru=False):
        cur = ST.cur
        if cur is None or cur.get("_pc") != pc:
            return orig_call(self, pc, func, proto, onenter, onexit, passthru)
        seen = {}

        @functools.wraps(func)
        def f(*a, **k):
            seen["params"] = next((x for x in reversed(a) if isinstance(x, dict)), None)
            return func(*a, **k)

        r = orig_call(self, pc, f, proto, onenter, onexit, passthru)
        cur["qiling_func"] = func.__name__
        if seen.get("params") is not None:
            cur["qiling_params"] = jsonable(seen["params"])
        cur["qiling_retval"] = jsonable(r)
        return r

    if "oscall" not in DISABLED:
        QlOs.call = os_call

    # -- loader: DllMain calls made by Qiling
    orig_cde = Process.call_dll_entrypoint

    def cde(self, dll, dll_base, dll_len, dll_name):
        ep = dll.OPTIONAL_HEADER.AddressOfEntryPoint
        skip = None
        if dll.get_section_by_rva(ep) is None:
            skip = "no section holds the entry point RVA"
        elif dll_name in ("kernelbase.dll", "kernel32.dll"):
            skip = "entry point ignored by Qiling for this name"
        elif dll_name == "gdi32.dll":
            skip = "entry point blacklisted by Qiling"
        ctx = dm_begin("Qiling loader (Process.call_dll_entrypoint, inside load_dll)", dll_name, dll_base + ep, dll_base)
        ctx["entry_rva"] = hex(ep)
        n0 = len(ST.starts)
        try:
            orig_cde(self, dll, dll_base, dll_len, dll_name)
        except BaseException as ex:  # noqa: BLE001
            dm_end(ctx, "raised in the loader: " + f"{type(ex).__name__}: {ex}")
            raise
        starts = ST.starts[n0:]
        if not starts:
            dm_end(ctx, "skipped by Qiling" + (f": {skip}" if skip else ""))
        else:
            s = starts[0]
            if s.get("error"):
                dm_end(ctx, "faulted (Qiling logged 'bailing' and restored the registers)", error=s["error"], fault=s.get("snapshot"))
            else:
                rax = ST.ql.arch.regs.read("rax") & 0xFFFFFFFF
                dm_end(ctx, "finished", ret=rax, reached_exit_point=s.get("reached_end"))

    Process.call_dll_entrypoint = cde

    # -- loader: every load_dll request
    orig_ld = Process.load_dll

    def load_dll(self, name, is_driver=False):
        ql = self.ql
        before = {os.path.basename(i.path).lower() for i in ql.loader.images}
        rec = {"n": len(ST.loads) + 1, "phase": ST.phase, "depth": ST.load_depth, "requested": name, "t": now(),
               "emu_state": str(ql.emu_state), "dllmain_ctx": dm_top()["id"] if dm_top() else None}
        ST.loads.append(rec)
        ST.load_depth += 1
        try:
            base = orig_ld(self, name, is_driver)
        finally:
            ST.load_depth -= 1
        rec["base"] = hx(base)
        rec["already_loaded"] = os.path.basename(str(name)).lower() in before
        rec["emu_started"] = ql.emu_state is QL_STATE.STARTED
        return base

    Process.load_dll = load_dll

    # -- loader: hook for unmapped accesses from the very start of the load
    orig_run = QlLoaderPE.run

    def run(self):
        ST.ql = self.ql
        if "unmapped" not in DISABLED:
            self.ql.hook_mem_unmapped(unmapped_cb)
        return orig_run(self)

    QlLoaderPE.run = run

    # -- every guest run started through Qiling.emu_start
    orig_es = Qiling.emu_start

    def emu_start(self, *a, **k):
        begin = a[0] if a else k.get("begin")
        end = a[1] if len(a) > 1 else k.get("end")
        ctx = dm_top()
        rec = {"n": len(ST.starts) + 1, "phase": ST.phase, "t": now(), "dllmain_ctx": ctx["id"] if ctx else None,
               "begin": where(begin) or hex(begin), "end": where(end) or hex(end)}
        ST.starts.append(rec)
        try:
            r = orig_es(self, *a, **k)
        except UcError as ex:
            rec["error"] = str(ex)
            rec["errno"] = ex.errno
            rec["snapshot"] = light_snapshot()
            raise
        pc = self.arch.regs.arch_pc
        rec["pc_after"] = where(pc) or hex(pc)
        rec["reached_end"] = (pc == end)
        return r

    Qiling.emu_start = emu_start

    # -- order in which memory is mapped
    orig_map = QlMemoryManager.map

    def mm_map(self, addr, size, *a, **k):
        info = k.get("info")
        if info is None:
            info = next((x for x in a if isinstance(x, str)), None)
        if len(ST.maps) < 4000:
            ST.maps.append({"n": len(ST.maps) + 1, "phase": ST.phase, "start": hex(addr), "size": hex(size), "info": info})
        return orig_map(self, addr, size, *a, **k)

    QlMemoryManager.map = mm_map


def unmapped_cb(ql, access, address, size, value):
    kind = {16: "read", 17: "write", 18: "fetch", 19: "read of unmapped memory", 20: "write to unmapped memory", 21: "fetch from unmapped memory",
            22: "write to protected memory", 23: "read of protected memory", 24: "fetch from protected memory"}.get(access, str(access))
    pc = ql.arch.regs.arch_pc
    ctx = dm_top()
    last = ST.ring[-1]["api"] if ST.ring else None
    if len(ST.unmapped) < 200:
        ST.unmapped.append({"phase": ST.phase, "dllmain_ctx": ctx["id"] if ctx else None, "access": kind, "address": hex(address),
                            "size": size, "pc": hex(pc), "pc_where": where(pc), "last_api": last, "t": now()})
    return False


# ---------------------------------------------------------------------------------------------
# DllMain history

def dm_begin(caller, dll, entry, hinst, reason=1, reserved=0):
    ctx = {"id": len(ST.dm) + 1, "caller": caller, "dll": dll, "entry": hex(entry), "entry_where": where(entry), "reason": reason,
           "hInstance": hex(hinst), "lpReserved": hex(reserved), "phase": ST.phase, "nesting_depth": len(ST.dm_stack),
           "t_start": now(), "api_calls": 0, "api_names": [], "result": "running"}
    ST.dm.append(ctx)
    ST.dm_stack.append(ctx)
    return ctx


def dm_end(ctx, result, **kw):
    ctx["result"] = result
    ctx["t_end"] = now()
    for k, v in kw.items():
        ctx[k] = v
    if ST.dm_stack and ST.dm_stack[-1] is ctx:
        ST.dm_stack.pop()


# ---------------------------------------------------------------------------------------------
# Mock subclass: tracing added from outside

class MemRecord(Record):
    def __init__(self):
        super().__init__(None)
        self.events = []

    def log(self, ev, **kw):
        self.seq += 1
        if len(self.events) < 20000:
            self.events.append({"seq": self.seq, "ev": ev, **jsonable(kw)})


class CapMock(Mock):
    def hook_apis(self):
        """Traces the imports the harness patches to return 0 (xor eax,eax; ret). One code hook per image
        range, not one per patched address: thousands of single address hooks make translation too slow
        and changed the run (a 20 s timeout instead of the fault)."""
        super().hook_apis()
        if "zero" in DISABLED:
            return
        from qiling.os.windows import const
        import qiling.os.windows.dlls as api
        from qiling.const import QL_INTERCEPT

        ql = self.ql
        user = ql.os.user_defined_api[QL_INTERCEPT.CALL]
        zero = ST.pdata.setdefault("zero_addrs", {})
        ranged = ST.pdata.setdefault("zero_ranged", {})
        ST.pdata.setdefault("zero_other", 0)
        for a, entry in list(ql.loader.import_symbols.items()):
            if a in zero:
                continue
            name = self._api_name(const, entry)
            if name in NATIVE_APIS or name in WIN_STUBS or name in user or hasattr(api, f"hook_{name}") or entry["dll"] in NATIVE_DLLS:
                continue
            zero[a] = (name, entry["dll"])
            img = ql.loader.find_containing_image(a)
            if img is not None and img.base not in ranged:
                ranged[img.base] = os.path.basename(img.path)
                ql.hook_code(self._zero_range_cb, begin=img.base, end=img.end - 1)

    def _zero_range_cb(self, ql, address, size):
        z = ST.pdata["zero_addrs"].get(address)
        if z is None:
            ST.pdata["zero_other"] += 1
            w = where(address)
            c = ST.pdata.setdefault("zero_other_where", collections.Counter())
            if w in c or len(c) < 300:
                c[w] += 1
            return
        e = begin_api(z[0], z[1], address)
        finish_api(e, 0, "harness patch (xor eax,eax; ret written at the export; no Python version existed)")

    def _api_cb(self, ql, address, size):
        if "apicb" in DISABLED:
            return super()._api_cb(ql, address, size)
        from qiling.os.windows import const

        entry = ql.loader.import_symbols[address]
        name = self._api_name(const, entry)
        e = begin_api(name, entry["dll"], address)
        e["_pc"] = address
        if STOP_ON_RAISE and name == "RaiseException" and ST.phase != "qiling-load":
            regs = ql.arch.regs
            code, flags, n, ptr = (regs.read(r) for r in ("rcx", "rdx", "r8", "r9"))
            params = [rd64(ptr + 8 * i) for i in range(min(n & 0xFFFFFFFF, 15))] if ptr else []
            ST.raise_info = {"ExceptionCode": hex(code & 0xFFFFFFFF), "ExceptionFlags": hex(flags & 0xFFFFFFFF), "NumberParameters": n & 0xFFFFFFFF,
                             "lpArguments": hex(ptr), "ExceptionInformation": [{"hex": hx(v), "at": where(v)} for v in params],
                             "called_from": e["called_from"], "return_address": e["return_addr"],
                             "dllmain_ctx": e["dllmain_ctx"], "api_call_n": e["n"]}
            self.exit_reason = f"capture stopped at RaiseException {code & 0xFFFFFFFF:#x}"
            finish_api(e, None, "capture stop: RaiseException not executed (--stop-on-raise)")
            ql.emu_stop()
            return
        ST.cur = e
        try:
            super()._api_cb(ql, address, size)
        except BaseException as ex:  # noqa: BLE001
            e["exception"] = f"{type(ex).__name__}: {ex}"
            finish_api(e, None, "raised")
            raise
        finally:
            ST.cur = None
        rax = ql.arch.regs.read("rax")
        from qiling.const import QL_INTERCEPT
        import qiling.os.windows.dlls as api

        fn = ql.os.user_defined_api[QL_INTERCEPT.CALL].get(name) or getattr(api, f"hook_{name}", None)
        if fn is not None and not e.get("qiling_func"):
            e["qiling_func"] = getattr(fn, "__name__", name)
        if name in WIN_STUBS:
            if name in ALLOC_STUBS:
                via = "harness allocator (GuestHeap)"
            elif name in SEMA_STUBS:
                via = "harness win_semaphore shim" + (f" -> Qiling hook {e['qiling_func']}" if e.get("qiling_func") else "")
            elif name in EXIT_STUBS:
                via = "harness exit stub"
            else:
                via = "harness Python stub (WIN_STUBS)"
            finish_api(e, rax, via)
        elif e.get("qiling_func"):
            finish_api(e, rax, f"Qiling Python hook {e['qiling_func']}")
        elif entry["dll"] not in NATIVE_DLLS:
            finish_api(e, 0, "harness fallback (no Python version; win_ret 0)")
        else:
            finish_api(e, None, "no hook answered: the native DLL's own code at that address runs (return value not captured here)")

    def call(self, addr, args, label="", timeout=0):
        if not (len(args) == 3 and args[1] == 1 and (label.endswith(".dll") or label == "DllMain")):
            return super().call(addr, args, label, timeout or CMD_TIMEOUT)
        dll = label if label.endswith(".dll") else os.path.basename(self.plugin)
        ctx = dm_begin("harness (Emu.call)", dll.lower() if dll.endswith(".dll") else dll, addr, args[0], args[1], args[2])
        n_unmapped = len(ST.unmapped)
        try:
            r = super().call(addr, args, label, timeout)
        except GuestFault as ex:
            dm_end(ctx, "faulted", error=str(ex), fault_pc=hx(ex.pc), fault_pc_where=where(ex.pc), last_api=ex.last,
                   unmapped=ST.unmapped[n_unmapped:], snapshot=light_snapshot())
            raise
        except BaseException as ex:  # noqa: BLE001
            dm_end(ctx, f"raised {type(ex).__name__}: {ex}")
            raise
        dm_end(ctx, "finished", ret=r & 0xFFFFFFFF)
        return r


# ---------------------------------------------------------------------------------------------
# state capture

def uc_consts():
    from unicorn import x86_const as c

    gp = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "rip"]
    ids = {n: getattr(c, "UC_X86_REG_" + n.upper()) for n in gp}
    ids["eflags"] = c.UC_X86_REG_EFLAGS
    ids["mxcsr"] = c.UC_X86_REG_MXCSR
    for s in ("cs", "ds", "es", "fs", "gs", "ss"):
        ids[s] = getattr(c, "UC_X86_REG_" + s.upper())
    return gp, ids, c


def light_snapshot():
    try:
        gp, ids, _ = uc_consts()
        uc = ST.ql.uc
        regs = {n: uc.reg_read(ids[n]) for n in gp}
        sp = regs["rsp"]
        return {"rip": hex(regs["rip"]), "rip_where": where(regs["rip"]), "regs": {n: hex(v) for n, v in regs.items()},
                "stack": [{"off": 8 * i, "value": hx(rd64(sp + 8 * i)), "where": where(rd64(sp + 8 * i))} for i in range(8)]}
    except Exception as ex:  # noqa: BLE001
        return {"error": str(ex)}


def mem_regions():
    ql = ST.ql
    info = [(ent[0], ent[1], ent[3]) for ent in ql.mem.map_info]
    out = []
    for b, e, p in ql.uc.mem_regions():
        end = e + 1
        labels = sorted({lab for s, ee, lab in info if s < end and ee > b})
        img = ql.loader.find_containing_image(b)
        out.append({"start": hex(b), "end": hex(end), "size": hex(end - b), "perms": "".join(c if p & f else "-" for c, f in (("r", 1), ("w", 2), ("x", 4))),
                    "qiling_label": ", ".join(labels) if labels else None, "image": os.path.basename(img.path) if img else None, "_b": b, "_e": end})
    out.sort(key=lambda r: r["_b"])
    return out


def fault_address_analysis(addr, regions):
    inside = [r for r in regions if r["_b"] <= addr < r["_e"]]
    below = [r for r in regions if r["_e"] <= addr]
    above = [r for r in regions if r["_b"] > addr]
    nb = max(below, key=lambda r: r["_e"]) if below else None
    na = min(above, key=lambda r: r["_b"]) if above else None

    def d(r, which):
        if r is None:
            return None
        return {"start": r["start"], "end": r["end"], "perms": r["perms"], "label": r["qiling_label"],
                "gap_bytes": hex(addr - r["_e"] if which == "below" else r["_b"] - addr),
                "gap_pages": round((addr - r["_e"] if which == "below" else r["_b"] - addr) / 4096, 2)}

    if inside:
        verdict = "inside a mapping: " + ", ".join(f"{r['start']}-{r['end']} {r['perms']} {r['qiling_label']}" for r in inside)
    else:
        gaps = [(addr - nb["_e"]) if nb else None, (na["_b"] - addr) if na else None]
        g = min(x for x in gaps if x is not None) if any(x is not None for x in gaps) else None
        if g is not None and g < 0x1000:
            verdict = f"not mapped, adjacent (nearest mapping {g:#x} bytes away, under one page)"
        elif g is not None and g < 0x100000:
            verdict = f"not mapped, near (nearest mapping {g:#x} bytes away, under 1 MiB)"
        else:
            verdict = f"not mapped, far from every mapping (nearest is {g:#x} bytes away)" if g is not None else "not mapped, no mapping at all"
    return {"address": hex(addr), "verdict": verdict, "inside": [r["start"] for r in inside], "nearest_below": d(nb, "below"), "nearest_above": d(na, "above")}


def pdata_of(img):
    key = img.base
    if key in ST.pdata:
        return ST.pdata[key]
    out = None
    nt = rd32(img.base + 0x3C)
    if nt is not None and rd16(img.base + nt + 0x18) == 0x20B:
        rva = rd32(img.base + nt + 0x18 + 0x88)
        size = rd32(img.base + nt + 0x18 + 0x8C)
        if rva and size:
            blob = rd(img.base + rva, size)
            if blob:
                rows = [struct.unpack_from("<III", blob, i) for i in range(0, len(blob) - 11, 12)]
                out = (rows, [r[0] for r in rows])
    ST.pdata[key] = out
    return out


def lookup_rf(img, rva):
    pd = pdata_of(img)
    if not pd:
        return None
    rows, begins = pd
    i = bisect.bisect_right(begins, rva) - 1
    if i >= 0 and rows[i][0] <= rva < rows[i][1]:
        return rows[i]
    return None


REG64 = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"]


def apply_unwind(img, unwind, regs, off_in_func, notes):
    """Applies an UNWIND_INFO (and its chain) to regs. off_in_func None means the prolog is complete."""
    base = img.base
    guard = 0
    while unwind is not None and guard < 8:
        guard += 1
        ui = base + (unwind & ~3)
        b0, prolog, ncodes, fr = rd8(ui), rd8(ui + 1), rd8(ui + 2), rd8(ui + 3)
        if None in (b0, prolog, ncodes, fr):
            notes.append("unwind info unreadable")
            return False
        flags = b0 >> 3
        frame_reg, frame_off = fr & 0xF, (fr >> 4) * 16
        codes = rd(ui + 4, ncodes * 2) or b""
        i = 0
        while i < ncodes and 2 * i + 1 < len(codes):
            co, b = codes[2 * i], codes[2 * i + 1]
            op, info = b & 0xF, b >> 4
            slots = {1: 2 if info == 0 else 3, 4: 2, 5: 3, 8: 2, 9: 3}.get(op, 1)
            applies = off_in_func is None or off_in_func >= co
            if applies:
                if op == 0:
                    regs[REG64[info]] = rd64(regs["rsp"])
                    regs["rsp"] += 8
                elif op == 1:
                    size = (codes[2 * (i + 1)] | codes[2 * (i + 1) + 1] << 8) * 8 if info == 0 else int.from_bytes(codes[2 * (i + 1):2 * (i + 1) + 4], "little")
                    regs["rsp"] += size
                elif op == 2:
                    regs["rsp"] += info * 8 + 8
                elif op == 3:
                    regs["rsp"] = regs[REG64[frame_reg]] - frame_off
                elif op == 4:
                    off = (codes[2 * (i + 1)] | codes[2 * (i + 1) + 1] << 8) * 8
                    regs[REG64[info]] = rd64(regs["rsp"] + off)
                elif op == 5:
                    off = int.from_bytes(codes[2 * (i + 1):2 * (i + 1) + 4], "little")
                    regs[REG64[info]] = rd64(regs["rsp"] + off)
                elif op == 10:
                    notes.append("machine frame op present (not followed)")
            i += slots
        if flags & 4:
            off = 4 + ((ncodes + 1) & ~1) * 2
            b, e, u = rd32(ui + off), rd32(ui + off + 4), rd32(ui + off + 8)
            if None in (b, e, u):
                notes.append("chained unwind info unreadable")
                return False
            unwind = u
            off_in_func = None
        else:
            unwind = None
    return True


def unwind_chain(regs0, maxn=24):
    ql = ST.ql
    regs = {n: regs0[n] for n in REG64}
    rip = regs0["rip"]
    frames = []
    for depth in range(maxn):
        f = {"depth": depth, "rip": hex(rip), "where": where(rip), "rsp": hex(regs["rsp"])}
        img = ql.loader.find_containing_image(rip)
        if img is None:
            f["note"] = "rip is not inside a loaded image; chain stops"
            frames.append(f)
            break
        if ST.mock is not None and img.base == ST.mock.base:
            f["note"] = "inside the plugin: not unwound (the plugin's own tables are not read); this address is the single return address into it"
            frames.append(f)
            break
        rf = lookup_rf(img, rip - img.base)
        notes = []
        if rf is None:
            f["unwind"] = "no RUNTIME_FUNCTION covers this address (treated as a leaf: return address at [rsp])"
        else:
            f["function_rva"] = f"{os.path.basename(img.path)}+{rf[0]:#x}..{rf[1]:#x}"
            off = (rip - img.base - rf[0]) if depth == 0 else None
            f["unwind"] = f"from .pdata / .xdata (offset in function {off:#x})" if off is not None else "from .pdata / .xdata (return address, prolog complete)"
            if not apply_unwind(img, rf[2], regs, off, notes):
                f["note"] = "; ".join(notes)
                frames.append(f)
                break
        if notes:
            f["note"] = "; ".join(notes)
        ra = rd64(regs["rsp"])
        f["return_address_slot"] = hex(regs["rsp"])
        f["return_address"] = hx(ra)
        f["return_where"] = where(ra)
        frames.append(f)
        if ra is None or ra == 0:
            break
        regs["rsp"] += 8
        if ra == ST.mock.STUB:
            f["note"] = (f.get("note", "") + "; the return address is the harness return stub (Emu.STUB): end of the chain").lstrip("; ")
            break
        rip = ra
    return frames


def capture_fault_state(m):
    ql = ST.ql
    uc = ql.uc
    gp, ids, c = uc_consts()
    regs = {n: uc.reg_read(ids[n]) for n in gp}
    fl = uc.reg_read(ids["eflags"])
    flag_names = {0: "CF", 2: "PF", 4: "AF", 6: "ZF", 7: "SF", 8: "TF", 9: "IF", 10: "DF", 11: "OF"}
    out = {"registers": {n: {"hex": hex(v), "at": where(v)} for n, v in regs.items()},
           "rflags": {"hex": hex(fl), "set": [nm for b, nm in flag_names.items() if fl >> b & 1]},
           "mxcsr": hex(uc.reg_read(ids["mxcsr"])),
           "segment_selectors": {s: hex(uc.reg_read(ids[s])) for s in ("cs", "ds", "es", "fs", "gs", "ss")}}
    out["rip_where"] = where(regs["rip"])
    bases = {}
    for nm in ("FS_BASE", "GS_BASE"):
        try:
            bases["uc_reg_" + nm.lower()] = hex(uc.reg_read(getattr(c, "UC_X86_REG_" + nm)))
        except Exception as ex:  # noqa: BLE001
            bases["uc_reg_" + nm.lower()] = f"unavailable ({ex})"
    for nm, msr in (("IA32_FS_BASE", 0xC0000100), ("IA32_GS_BASE", 0xC0000101), ("IA32_KERNEL_GS_BASE", 0xC0000102)):
        try:
            bases["msr_" + nm.lower()] = hex(uc.msr_read(msr))
        except Exception as ex:  # noqa: BLE001
            bases["msr_" + nm.lower()] = f"unavailable ({ex})"
    out["segment_bases"] = bases
    gs = None
    for k in ("msr_ia32_gs_base", "uc_reg_gs_base"):
        v = bases.get(k)
        if isinstance(v, str) and v.startswith("0x") and int(v, 16):
            gs = int(v, 16)
            break
    teb = {}
    if gs:
        teb = {"gs_base": hex(gs), "TEB.Self (gs:[0x30])": hx(rd64(gs + 0x30)), "PEB (gs:[0x60])": hx(rd64(gs + 0x60)),
               "TEB.StackBase (gs:[0x8])": hx(rd64(gs + 0x8)), "TEB.StackLimit (gs:[0x10])": hx(rd64(gs + 0x10)),
               "TEB.ExceptionList/first qword (gs:[0x0])": hx(rd64(gs))}
        peb = rd64(gs + 0x60)
        teb["PEB_where"] = where(peb)
        teb["TEB_where"] = where(gs)
        if peb:
            teb["PEB.ImageBaseAddress (peb+0x10)"] = hx(rd64(peb + 0x10))
            teb["PEB.Ldr (peb+0x18)"] = hx(rd64(peb + 0x18))
            teb["PEB.ProcessHeap (peb+0x30)"] = hx(rd64(peb + 0x30))
    teb["loader_stack_address"] = hex(ql.loader.stack_address)
    teb["loader_stack_size"] = hex(ql.loader.stack_size)
    teb["harness_stack_top (Emu.stack_top)"] = hex(m.stack_top)
    out["teb_peb"] = teb
    sp = regs["rsp"]
    stack = []
    for i in range(128):
        v = rd64(sp + 8 * i)
        if v is None:
            stack.append({"slot": i, "addr": hex(sp + 8 * i), "value": None, "note": "unreadable (unmapped)"})
            break
        stack.append({"slot": i, "addr": hex(sp + 8 * i), "value": hex(v), "where": where(v)})
    out["stack_dump"] = stack
    try:
        out["return_chain"] = unwind_chain(regs)
    except Exception as ex:  # noqa: BLE001
        out["return_chain"] = [{"depth": 0, "rip": hex(regs["rip"]), "where": where(regs["rip"]), "note": f"unwind failed: {type(ex).__name__}: {ex}"}]
    regions = mem_regions()
    fa = None
    ev = [u for u in ST.unmapped if u["phase"] != "qiling-load"]
    if ev:
        fa = int(ev[-1]["address"], 16)
    out["unmapped_events"] = ST.unmapped
    if fa is not None:
        out["fault_address"] = fault_address_analysis(fa, regions)
        out["registers_near_fault_address"] = [n for n, v in regs.items() if abs(v - fa) <= 0x10000 and n not in ("rip",)]
        out["stack_slots_near_fault_address"] = [s["slot"] for s in stack if s.get("value") and abs(int(s["value"], 16) - fa) <= 0x10000]
    out["memory_map"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in regions]
    out["images"] = images()
    return out


def images():
    ql = ST.ql
    return [{"order": i + 1, "name": os.path.basename(im.path), "base": hex(im.base), "end": hex(im.end), "size": hex(im.end - im.base)}
            for i, im in enumerate(ql.loader.images)]


# ---------------------------------------------------------------------------------------------
# files

def file_facts(path):
    import pefile

    p = Path(path)
    out = {"path": str(path)}
    if not p.exists():
        out["error"] = "missing"
        return out
    data = p.read_bytes()
    out["sha256"] = hashlib.sha256(data).hexdigest()
    out["size"] = len(data)
    pe = pefile.PE(data=data)
    out["TimeDateStamp"] = hex(pe.FILE_HEADER.TimeDateStamp)
    out["SizeOfImage"] = hex(pe.OPTIONAL_HEADER.SizeOfImage)
    out["AddressOfEntryPoint"] = hex(pe.OPTIONAL_HEADER.AddressOfEntryPoint)
    out["ImageBase"] = hex(pe.OPTIONAL_HEADER.ImageBase)
    strings = {}

    def walk(o):
        if isinstance(o, (list, tuple)):
            for x in o:
                walk(x)
            return
        if hasattr(o, "StringTable"):
            for st in o.StringTable:
                for k, v in st.entries.items():
                    strings[k.decode("latin-1")] = v.decode("latin-1")

    walk(getattr(pe, "FileInfo", []))
    fixed = getattr(pe, "VS_FIXEDFILEINFO", None)
    if fixed:
        f = fixed[0] if isinstance(fixed, (list, tuple)) else fixed
        out["FileVersion_fixed"] = f"{f.FileVersionMS >> 16}.{f.FileVersionMS & 0xFFFF}.{f.FileVersionLS >> 16}.{f.FileVersionLS & 0xFFFF}"
        out["ProductVersion_fixed"] = f"{f.ProductVersionMS >> 16}.{f.ProductVersionMS & 0xFFFF}.{f.ProductVersionLS >> 16}.{f.ProductVersionLS & 0xFFFF}"
    for k in ("FileVersion", "ProductVersion", "FileDescription", "OriginalFilename", "CompanyName"):
        if k in strings:
            out[k] = strings[k]
    pe.close()
    return out


# ---------------------------------------------------------------------------------------------
# Markdown

def write_params(m, jpath, tpath):
    """Every parameter the plugin registered in PARAMS_SETUP, as the harness decoded it from the host call."""
    rows = [{k: v for k, v in d.items() if k != "guest"} for d in m.reg_params]
    jpath.write_text(json.dumps({"plugin": os.path.splitext(os.path.basename(m.plugin))[0], "count": len(rows), "params": rows}, indent=1, default=str), encoding="utf-8")
    tab, nl = chr(9), chr(10)
    lines = [tab.join(["index", "name", "type", "min", "max", "default", "choices", "precision", "display_flags"])]
    for d in rows:
        vals = [d.get("reg_index"), d.get("name"), d.get("type_name"), d.get("min"), d.get("max"), d.get("default"),
                "|".join(d.get("choices") or []), d.get("precision"), d.get("display_flags")]
        lines.append(tab.join("" if v is None else str(v).replace(tab, " ") for v in vals))
    tpath.write_text(nl.join(lines) + nl, encoding="utf-8")


def md_cell(s):
    return str(s).replace("|", "/").replace("\n", " ")


def write_md(out, path):
    L = []
    a = L.append
    meta = out["meta"]
    a("# msvcr120.dll capture, Twixtor_64.aex")
    a("")
    a(f"State capture only, no fixes. Qiling {meta['qiling']}, Unicorn {meta['unicorn']}, run {meta['date']} on {meta['host']}. "
      "Full data (stack of 128 qwords, complete memory map, 100 API calls with decoded arguments) is in `msvcr120_capture.json`.")
    a("")
    a("## 1. The DLLs")
    a("")
    a("| file | sha256 | size | FileVersion | ProductVersion |")
    a("|---|---|---|---|---|")
    for k, v in out["files"]["rootfs"].items():
        a(f"| rootfs {k} | `{v.get('sha256')}` | {v.get('size')} | {v.get('FileVersion', v.get('FileVersion_fixed'))} | {v.get('ProductVersion', v.get('ProductVersion_fixed'))} |")
    lp = out["files"].get("laptop_msvcr120")
    if lp:
        a(f"| laptop C:\\Windows\\System32\\msvcr120.dll | `{lp.get('sha256')}` | {lp.get('size')} | {lp.get('FileVersion')} | {lp.get('ProductVersion')} |")
        s = out["files"]["same_file"]
        a("")
        a(f"Rootfs msvcr120.dll vs the laptop's: sha256 equal = {s['sha256']}, size equal = {s['size']}, version equal = {s['version']}.")
    a("")
    fs = out.get("fault_state")
    oc = out["outcome"]
    a("## 2. Fault-time state")
    a("")
    a(f"Outcome: `{oc.get('error')}` (boot stopped after {oc.get('seconds')} s).")
    a("")
    if fs:
        a(f"rip = {fs['registers']['rip']['hex']} ({fs['rip_where']}); rflags = {fs['rflags']['hex']} {fs['rflags']['set']}; mxcsr = {fs['mxcsr']}")
        a("")
        a("| reg | value | module+offset or region |")
        a("|---|---|---|")
        for n, v in fs["registers"].items():
            a(f"| {n} | {v['hex']} | {v['at'] or ''} |")
        a("")
        a("Segment selectors: " + ", ".join(f"{k}={v}" for k, v in fs["segment_selectors"].items()))
        a("")
        a("Segment bases and TEB/PEB:")
        a("")
        for k, v in {**fs["segment_bases"], **fs["teb_peb"]}.items():
            a(f"- {k}: {v}")
        a("")
        fa = fs.get("fault_address")
        if fa:
            a(f"Fault address {fa['address']}: {fa['verdict']}.")
            for w in ("nearest_below", "nearest_above"):
                r = fa[w]
                a(f"- {w}: " + (f"{r['start']}-{r['end']} {r['perms']} {r['label']} (gap {r['gap_bytes']} = {r['gap_pages']} pages)" if r else "none"))
            a(f"- registers within 0x10000 of it: {fs.get('registers_near_fault_address')}; stack slots within 0x10000 of it: {fs.get('stack_slots_near_fault_address')}")
            a("")
        a("Unmapped-access events seen by the capture hook:")
        a("")
        for u in fs["unmapped_events"]:
            a(f"- [{u['phase']}] {u['access']} {u['address']} size {u['size']} at pc {u['pc_where'] or u['pc']}, last API {u['last_api']}, DllMain ctx {u['dllmain_ctx']}")
        a("")
        a("Return-address chain (unwound from .pdata/.xdata; metadata only):")
        a("")
        a("| depth | rip | function | return address | return where |")
        a("|---|---|---|---|---|")
        for f in fs["return_chain"]:
            a(f"| {f['depth']} | {f['where'] or f['rip']} | {f.get('function_rva', '')} | {f.get('return_address', '')} | {f.get('return_where') or ''} {md_cell(f.get('note', ''))} |")
        a("")
        a("Stack from rsp (first 64 of 128 qwords; the rest is in the JSON):")
        a("")
        a("| slot | addr | value | module+offset or region |")
        a("|---|---|---|---|")
        for s in fs["stack_dump"][:64]:
            a(f"| {s['slot']} | {s['addr']} | {s.get('value')} | {s.get('where') or ''} |")
        a("")
        a("Complete Unicorn memory map (sorted):")
        a("")
        a("| start | end | perms | Qiling label | image |")
        a("|---|---|---|---|---|")
        for r in fs["memory_map"]:
            a(f"| {r['start']} | {r['end']} | {r['perms']} | {md_cell(r['qiling_label'])} | {r['image'] or ''} |")
        a("")
    if out.get("raise_exception"):
        r = out["raise_exception"]
        a("## RaiseException stop (--stop-on-raise)")
        a("")
        a(f"The boot was stopped at API call n={r['api_call_n']} before RaiseException ran: code {r['ExceptionCode']}, flags {r['ExceptionFlags']}, NumberParameters {r['NumberParameters']}, lpArguments {r['lpArguments']}, called from {r['called_from']} (DllMain ctx {r['dllmain_ctx']}).")
        a("")
        for i, v in enumerate(r["ExceptionInformation"]):
            a(f"- ExceptionInformation[{i}] = {v['hex']} {v['at'] or ''}")
        a("")
    tr = out["api_trace"]
    a("## 3. Last API calls before the fault")
    a("")
    a(f"{tr['total_calls']} API calls were recorded in total; the last {len(tr['last_100'])} are listed (oldest first). 'via' says what answered. "
      "Calls into native DLL code that no hook intercepts (msvcr120 and friends calling each other) are not API calls and are not listed.")
    a("")
    a("| n | phase | DllMain ctx | api (dll) | rcx | rdx | r8 | r9 | ret | via | called from |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for e in tr["last_100"]:
        r = e["args_raw"]
        a(f"| {e['n']} | {e['phase']} | {e['dllmain_ctx']} | {e['api']} ({e['dll']}) | {r['rcx']} | {r['rdx']} | {r['r8']} | {r['r9']} | {e.get('ret')} | {md_cell(e['via'])} | {e.get('called_from') or ''} |")
    a("")
    if tr.get("memset_calls"):
        a("memset calls in that list:")
        a("")
        for e in tr["memset_calls"]:
            dd = e.get("args_decoded", {})
            a(f"- n={e['n']} [{e['phase']}] dest={e['args_raw']['rcx']} ({where_text(e)}), c={e['args_raw']['rdx']}, count={e['args_raw']['r8']}, ret={e.get('ret')}, via {e['via']}")
        a("")
    dmh = out["dllmain"]
    a("## 4. DllMain history")
    a("")
    a("| id | caller | dll | entry | reason | hInstance | lpReserved | result | return | API calls | t start-end (s) |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in dmh["history"]:
        a(f"| {c['id']} | {md_cell(c['caller'])} | {c['dll']} | {c.get('entry_where') or c['entry']} | {c['reason']} | {c['hInstance']} | {c['lpReserved']} | {md_cell(c['result'])} | {c.get('ret', '')} | {c['api_calls']} | {c['t_start']}-{c.get('t_end', '')} |")
    a("")
    for line in dmh["summary_lines"]:
        a(f"- {line}")
    a("")
    a("## 5. Order of loading")
    a("")
    a("Images in the order Qiling recorded them (ql.loader.images):")
    a("")
    a("| order | image | base | end | size |")
    a("|---|---|---|---|---|")
    for i in out["load_order"]["images"]:
        a(f"| {i['order']} | {i['name']} | {i['base']} | {i['end']} | {i['size']} |")
    a("")
    a("load_dll requests (depth shows nesting; requests for an already loaded image are marked):")
    a("")
    a("| n | depth | requested | base | already loaded | phase |")
    a("|---|---|---|---|---|---|")
    for r in out["load_order"]["load_dll_calls"]:
        a(f"| {r['n']} | {r['depth']} | {r['requested']} | {r.get('base')} | {r.get('already_loaded')} | {r['phase']} |")
    a("")
    a("## Not captured")
    a("")
    for n in out["notes"]:
        a(f"- {n}")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def where_text(e):
    d = (e.get("args_decoded") or {}).get("rcx")
    return d.get("at") if d else "not inside any mapping or unreadable"


# ---------------------------------------------------------------------------------------------

def build_dllmain_summary(m):
    hist = ST.dm
    ran = ("finished", "faulted")
    summary = []
    by = collections.defaultdict(list)
    for c in hist:
        by[c["dll"]].append(c)
    names = {c["dll"] for c in hist}
    plugin = os.path.basename(m.plugin)
    if plugin not in names:
        names.add(plugin)
    for dll in ("msvcr120.dll", "msvcp120.dll", plugin):
        cs = by.get(dll, [])
        q = [c for c in cs if c["caller"].startswith("Qiling")]
        hcs = [c for c in cs if c["caller"].startswith("harness")]
        qran = [c for c in q if c["result"].startswith(ran)]
        hran = [c for c in hcs if c["result"].startswith(ran)]
        summary.append(f"{dll}: Qiling loader calls {len(q)} ({', '.join(c['result'] for c in q) or 'none, the loader only sets up the call frame for the main image'}); "
                       f"harness calls {len(hcs)} ({', '.join(c['result'] for c in hcs) or 'never ran, boot stopped earlier'}); "
                       f"times DllMain code started: {len(qran) + len(hran)}"
                       + ("; RAN MORE THAN ONCE" if len(qran) + len(hran) > 1 else ""))
    others = [c for c in hist if c["dll"] not in ("msvcr120.dll", "msvcp120.dll", plugin) and c["caller"].startswith("Qiling")]
    summary.append("Other DLLs (Qiling loader): " + "; ".join(f"{c['dll']} #{c['id']} {c['result']}" + (f" ret={c['ret']}" if "ret" in c else "") for c in others))
    twice = [d for d, cs in by.items() if sum(1 for c in cs if c["result"].startswith(ran)) > 1]
    summary.append("DLLs whose DllMain code ran more than once: " + (", ".join(twice) if twice else "none"))
    return summary


def plain_boot(a):
    from importlib.metadata import version as pkg_version

    rec = Record(None)
    plugin = str(Path(a.rootfs) / "plugins" / f"{a.plugin}.aex")
    t0 = time.perf_counter()
    out = {"mode": "plain Mock boot, no tracing", "plugin": a.plugin, "qiling": pkg_version("qiling"), "unicorn": pkg_version("unicorn"),
           "python": platform.python_version(), "argv": sys.argv, "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"), "isolated": sys.flags.isolated}
    m = None
    try:
        m = Mock(plugin, a.rootfs, rec, Layouts())
        m.plugin_name = a.plugin
        out["load_seconds"] = round(time.perf_counter() - t0, 1)
        m.boot()
        out["error"] = None
    except BaseException as ex:  # noqa: BLE001
        out["error"] = f"{type(ex).__name__}: {ex}"
        out["last_api"] = getattr(ex, "last", None)
        out["pc"] = hx(getattr(ex, "pc", None))
    out["seconds"] = round(time.perf_counter() - t0, 1)
    if m is not None:
        out["faults"] = [{k: (hex(v) if isinstance(v, int) and k != "size" else v) for k, v in f.items()} for f in m.faults[:5]]
        out["last_api_seen"] = m.last_api
        out["api_ring_last_40"] = list(m.api_ring)
        out["api_calls_by_harness_hooks"] = sum(m.api_counts.values())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out.get(k) for k in ("error", "last_api", "pc", "faults", "seconds", "api_calls_by_harness_hooks")}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", default="Twixtor_64")
    ap.add_argument("--rootfs", default=str(HERE / "rootfs" / "x8664_windows"))
    ap.add_argument("--out", default=str(HERE / "results" / "msvcr120_capture.json"))
    ap.add_argument("--md", default=None)
    ap.add_argument("--laptop-path")
    ap.add_argument("--laptop-sha256")
    ap.add_argument("--laptop-size", type=int)
    ap.add_argument("--laptop-fileversion")
    ap.add_argument("--laptop-productversion")
    ap.add_argument("--command", default="")
    ap.add_argument("--stop-on-raise", action="store_true", help="stop the boot at the first RaiseException made after the Qiling load and record its exception record")
    ap.add_argument("--setup", action="store_true", help="after boot() also run setup_plugin() (GLOBAL_SETUP and PARAMS_SETUP), as the worker does")
    ap.add_argument("--cmd-timeout", type=float, default=0, help="seconds of emulation allowed to each plugin command (0 = unlimited, the harness default)")
    ap.add_argument("--case", help="after setup, render this fx-parity case id the way worker.py does (needs --testset)")
    ap.add_argument("--testset")
    ap.add_argument("--refs", default="capture_refs")
    ap.add_argument("--param-map", default=str(HERE / "param_map.tsv"))
    ap.add_argument("--plain", action="store_true", help="no tracing at all: the plain Mock boot, outcome only (reproduction check)")
    a = ap.parse_args()
    global STOP_ON_RAISE, CMD_TIMEOUT
    STOP_ON_RAISE = a.stop_on_raise
    CMD_TIMEOUT = a.cmd_timeout
    if a.plain:
        return plain_boot(a)
    out_path = Path(a.out)
    md_path = Path(a.md) if a.md else out_path.with_suffix(".md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from importlib.metadata import version as pkg_version

    out = {"meta": {"date": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "host": platform.node(), "python": platform.python_version(),
                    "qiling": pkg_version("qiling"), "unicorn": pkg_version("unicorn"),
                    "plugin": a.plugin, "rootfs": a.rootfs, "command": a.command},
           "notes": []}
    sysdir = Path(a.rootfs) / "Windows" / "System32"
    files = {}
    for nm in ("msvcr120.dll", "msvcp120.dll", "ucrtbase.dll"):
        cand = [p for p in sysdir.iterdir() if p.name.lower() == nm] if sysdir.exists() else []
        files[nm] = file_facts(cand[0]) if cand else {"error": "missing"}
        if cand:
            files[nm]["file_name_in_rootfs"] = cand[0].name
    out["files"] = {"rootfs": files}
    if a.laptop_sha256:
        lp = {"path": a.laptop_path, "sha256": a.laptop_sha256.lower(), "size": a.laptop_size, "FileVersion": a.laptop_fileversion,
              "ProductVersion": a.laptop_productversion}
        out["files"]["laptop_msvcr120"] = lp
        r = files["msvcr120.dll"]
        out["files"]["same_file"] = {"sha256": r.get("sha256") == lp["sha256"], "size": r.get("size") == lp["size"],
                                     "version": (r.get("FileVersion") == lp["FileVersion"] and r.get("ProductVersion") == lp["ProductVersion"])}

    install_qiling_patches()
    rec = MemRecord()
    lay = Layouts()
    plugin = str(Path(a.rootfs) / "plugins" / f"{a.plugin}.aex")
    t0 = time.perf_counter()
    m = None
    outcome = {}
    try:
        ST.phase = "qiling-load"
        m = CapMock(plugin, a.rootfs, rec, lay)
        ST.mock = m
        m.plugin_name = a.plugin
        outcome["load_seconds"] = round(time.perf_counter() - t0, 1)
        ST.phase = "harness-boot"
        try:
            m.boot()
            outcome["error"] = None
            outcome["note"] = "boot() returned without a fault"
            if a.setup:
                outcome["boot_seconds"] = round(time.perf_counter() - t0, 1)
                ST.phase = "harness-setup"
                m.setup_plugin()
                outcome["note"] += "; setup_plugin() returned without a fault"
                if a.case:
                    import worker as W

                    ST.phase = "harness-render"
                    ts = Path(a.testset)
                    case = next(c for c in json.loads((ts / "cases.json").read_text())["cases"] if c["id"] == a.case)
                    pm = W.load_param_map(Path(a.param_map), a.plugin) if Path(a.param_map).exists() else {}
                    outcome["render_result"] = W.run_case(m, case, ts, Path(a.refs), pm)
                    outcome["note"] += "; the case rendered"
        except (GuestFault, Exception) as ex:  # noqa: BLE001
            outcome["error"] = f"{type(ex).__name__}: {ex}"
            outcome["last_api"] = getattr(ex, "last", None)
            outcome["pc"] = hx(getattr(ex, "pc", None))
            outcome["pc_where"] = where(getattr(ex, "pc", None))
            outcome["trace"] = traceback.format_exc().splitlines()[-4:]
    except BaseException as ex:  # noqa: BLE001
        outcome["error"] = f"{type(ex).__name__}: {ex} (during the Qiling load, before the harness took over)"
        outcome["trace"] = traceback.format_exc().splitlines()[-6:]
    outcome["seconds"] = round(time.perf_counter() - t0, 1)
    out["outcome"] = outcome
    if m is not None:
        out["commands"] = [{k: ev.get(k) for k in ("cmd", "err", "out_flags", "out_flags2", "num_params", "msg", "seconds")}
                           for ev in rec.events if ev.get("ev") == "cmd_done"]
        out["crypt_events"] = {"note": "CryptoAPI calls as logged by win_crypt: names, flags, sizes and return values only; CryptGenRandom is a deterministic stand-in",
                               "events": [ev for ev in rec.events if ev.get("ev") == "crypt"]}
        out["seh_events"] = [ev for ev in rec.events if str(ev.get("ev", "")).startswith("seh_")]
        keep = ("fn", "args", "ret", "suite", "version", "struct", "acquired", "unimplemented", "runtime")
        out["host_callbacks"] = {"note": "every host callback and suite call the plugin made: names and numbers only",
                                 "events": [{k: ev[k] for k in keep if k in ev} for ev in rec.events if ev.get("ev") == "cb"][-400:]}
        sched = getattr(m, "_sched", None)
        if sched is not None:
            def where_t(a):
                img = ST.ql.loader.find_containing_image(a) if a else None
                if img is None:
                    return None
                return os.path.basename(img.path) if img.base == m.base else f"{os.path.basename(img.path)}+{a - img.base:#x}"
            out["guest_threads"] = {"current": sched.cur.id, "threads": [
                {"id": t.id, "state": t.state, "suspended": t.suspended,
                 "waits_on": [type(w).__name__ for w in (t.wait if isinstance(t.wait, list) else [t.wait])] if t.wait is not None else None,
                 "rsp": hex(t.ctx["rsp"]) if t.ctx else None, "rsp_in": ST.ql.mem.map_info and next((e[3] for e in ST.ql.mem.map_info if t.ctx and e[0] <= t.ctx["rsp"] < e[1]), None),
                 "rip_in": where_t(t.ctx["rip"]) if t.ctx else None} for t in sched.threads]}
        out["api_errors"] = [ev for ev in rec.events if ev.get("ev") == "api_error"]
        out["commands_started"] = [ev.get("cmd") for ev in rec.events if ev.get("ev") == "cmd"]
        out["out_flags_now"] = {"out_flags": hex(m.out_flags & 0xFFFFFFFF), "out_flags2": hex(m.out_flags2 & 0xFFFFFFFF), "note": "of the last command"}
        out["out_flags_global_setup"] = {"out_flags": hex(m.global_out_flags & 0xFFFFFFFF), "out_flags2": hex(m.global_out_flags2 & 0xFFFFFFFF)}
        if getattr(m, "reg_params", None):
            write_params(m, out_path.with_name(out_path.stem + "_params.json"), out_path.with_name(out_path.stem + "_params.tsv"))
            out["params_files"] = [out_path.stem + "_params.json", out_path.stem + "_params.tsv"]
    if getattr(ST, "raise_info", None):
        out["raise_exception"] = ST.raise_info

    try:
        if ST.ql is not None:
            out["fault_state"] = capture_fault_state(m if m is not None else types.SimpleNamespace(stack_top=0, STUB=0))
    except BaseException as ex:  # noqa: BLE001
        out["fault_state_error"] = f"{type(ex).__name__}: {ex}"
        out["fault_state_trace"] = traceback.format_exc().splitlines()[-6:]

    ring = list(ST.ring)
    for e in ring:
        decode_entry(e)
    out["api_trace"] = {"total_calls": ST.total, "ring_size": 100, "last_100": ring,
                        "memset_calls": [e for e in ring if e["api"] == "memset"],
                        "call_counts": {f"{ph}:{nm}": n for (ph, nm), n in sorted(ST.counts.items(), key=lambda kv: -kv[1])[:500]},
                        "special_calls_note": "every RtlPcToFileHeader call, every call answered by a harness zero patch and every RaiseException call after the Qiling load, in order",
                        "special_calls": ST.special}
    if m is not None:
        pl = os.path.basename(m.plugin)
        for dll in (pl,):
            if not any(c["dll"] == dll for c in ST.dm):
                ST.dm.append({"id": len(ST.dm) + 1, "caller": "harness (Emu.call)", "dll": dll, "entry": hex(m.ql.loader.entry_point),
                              "entry_where": where(m.ql.loader.entry_point), "reason": 1, "hInstance": hex(m.base), "lpReserved": "0x0",
                              "phase": "harness-boot", "nesting_depth": 0, "t_start": None, "api_calls": 0, "api_names": [],
                              "result": "never ran (boot stopped earlier)" if outcome.get("error") else "not recorded"})
        if not any(c["dll"] == "msvcp120.dll" and c["caller"].startswith("harness") for c in ST.dm):
            ST.dm.append({"id": len(ST.dm) + 1, "caller": "harness (Emu.call)", "dll": "msvcp120.dll", "entry": "?", "entry_where": None,
                          "reason": 1, "hInstance": "?", "lpReserved": "0x0", "phase": "harness-boot", "nesting_depth": 0, "t_start": None,
                          "api_calls": 0, "api_names": [], "result": "never ran (init_libraries stopped at msvcr120.dll)" if outcome.get("error") else "not recorded"})
    out["dllmain"] = {"history": ST.dm, "summary_lines": build_dllmain_summary(m) if m is not None else [], "emu_start_calls": ST.starts}
    out["load_order"] = {"images": images() if ST.ql is not None else [], "load_dll_calls": ST.loads, "map_calls": ST.maps}
    tail = rec.events[-80:] if rec else []
    for ev in tail:
        if ev.get("ev") == "api_args" and ev.get("name") == "RaiseException" and isinstance(ev.get("args"), list):
            ev["args"] = ev["args"][:3]
    out["harness_events_tail"] = tail
    out["qiling_log_lines"] = [l for l in (getattr(m, "qlog", []) or []) if "DllMain" in l or "bailing" in l or l.startswith("ERROR")][:100] if m is not None else []
    out["qiling_log_counts"] = dict(collections.Counter(l.split(":")[0] for l in (getattr(m, "qlog", []) or []))) if m is not None else {}
    out["notes"] += [
        "Calls into native DLL code (msvcr120, msvcp120, ucrtbase calling each other through their own tables) are not API calls here and are not traced; only imports that Qiling or the harness hook, and imports the harness patches to return 0, are.",
        "The register and stack snapshot is taken after the exception has unwound to the harness; Unicorn leaves the registers as they were at the faulting instruction.",
        "The return-address chain is unwound from the exception tables of the images (.pdata/.xdata); the fault happens inside a function body, not in an epilogue, so the table walk applies.",
        f"Instructions executed inside the images that hold harness patched imports, other than the first instruction of a patched stub (the ret of each stub counts here too): {ST.pdata.get('zero_other')}; by address: "
        + ", ".join(f"{w} x{n}" for w, n in sorted(ST.pdata.get("zero_other_where", {}).items(), key=lambda kv: -kv[1])[:25]),
        "Disabled tracing pieces for this run (CAP_DISABLE): " + (",".join(sorted(DISABLED)) or "none"),
        "No code bytes of any image were read for this capture, and nothing was disassembled.",
    ]
    out_path.write_text(json.dumps(out, indent=1, default=lambda o: hex(o) if isinstance(o, int) else str(o)), encoding="utf-8")
    write_md(out, md_path)
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes) and {md_path}")
    print("outcome:", json.dumps(outcome))


if __name__ == "__main__":
    main()
