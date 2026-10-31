#!/usr/bin/env python3
"""A mock After Effects host for one .aex plugin under Qiling (x8664 Windows).

Black box: the plugin is only driven through the host interface (entry point commands, callbacks,
suites). Struct layouts come from sdk_layouts.json, generated from the SDK headers by
gen_layouts.py. See README.md for the stand-ins and assumptions.
"""
from __future__ import annotations

import collections
import json
import math
import os
import re
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------------------------
# SDK layouts

U32 = struct.Struct("<I")
U64 = struct.Struct("<Q")


class Layouts:
    def __init__(self, path: Path = HERE / "sdk_layouts.json"):
        d = json.loads(Path(path).read_text())
        self.L = d["layouts"]
        self.enums = d["enums"]
        self.suites = d["suites"]
        extra = Path(path).with_name("sdk_layouts_extra.json")        # suite revisions from AE_GeneralPlugOld.h, see gen_aegp_suites.py
        if extra.exists():
            e = json.loads(extra.read_text())
            self.L.update(e["layouts"])
            self.suites = self.suites + e["suites"]
        self._idx = {}

    def idx(self, struct_name: str) -> dict:
        if struct_name not in self._idx:
            self._idx[struct_name] = {m["name"]: m for m in self.L[struct_name]["members"]}
        return self._idx[struct_name]

    def off(self, struct_name: str, field: str) -> int:
        return self.idx(struct_name)[field]["offset"]

    def size(self, struct_name: str) -> int:
        return self.L[struct_name]["size"]

    def e(self, name: str) -> int:
        return self.enums[name]

    def suite_struct(self, name: str, version: int):
        """The struct name of the suite `name` at `version`, or None."""
        for s in self.suites:
            if s["name"] != name or s["struct"] not in self.L:
                continue
            # the SDK header gives each revision of a suite its numeric version in a macro, which is not always
            # the revision's number (kPFHandleSuiteVersion1 is 2, kPFAppSuiteVersion4 is 6): that macro decides
            m = re.match(r"^(.*?)(\d+)$", s["struct"])
            macro = "k" + m.group(1).replace("_", "") + "Version" + m.group(2) if m else None
            numeric = s.get("versions", {}).get(macro)
            if (numeric == version) if numeric is not None else s["struct"].endswith(str(version)):
                return s["struct"]
        return None


def _scalar(mtype: str, size: int):
    """(struct format, is float) for a member type."""
    if "FpShort" in mtype:
        return "<f"
    if "FpLong" in mtype:
        return "<d"
    if mtype.endswith("*"):
        return "<Q"
    unsigned = "u_" in mtype or mtype.startswith("unsigned") or mtype in ("PF_UFixed", "A_u_long", "size_t", "PF_PixLong", "PF_WorldFlags")
    return "<" + {1: "B" if unsigned else "b", 2: "H" if unsigned else "h", 4: "I" if unsigned else "i", 8: "Q" if unsigned else "q"}[size]


class View:
    """A struct in guest memory, read and written by member name."""

    def __init__(self, host: "Host", addr: int, name: str):
        self.h, self.a, self.name = host, addr, name
        self.idx = host.lay.idx(name)

    def addr(self, field: str) -> int:
        return self.a + self.idx[field]["offset"]

    def get(self, field: str):
        m = self.idx[field]
        if m["type"].endswith("]"):
            return bytes(self.h.ql.mem.read(self.a + m["offset"], m["size"]))
        return struct.unpack(_scalar(m["type"], m["size"]), bytes(self.h.ql.mem.read(self.a + m["offset"], m["size"])))[0]

    def set(self, field: str, value):
        m = self.idx[field]
        if m["type"].endswith("]"):
            data = bytes(value)[: m["size"]].ljust(m["size"], b"\0")
        else:
            data = struct.pack(_scalar(m["type"], m["size"]), value)
        self.h.ql.mem.write(self.a + m["offset"], data)


# ---------------------------------------------------------------------------------------------
# The call record

class Record:
    """JSON lines of every host callback and suite call the plugin makes."""

    def __init__(self, path: str | None):
        self.f = open(path, "w") if path else None
        self.seq = 0
        self.t0 = time.time()

    def log(self, ev: str, **kw):
        if self.f is None:
            return
        self.seq += 1
        kw = {"seq": self.seq, "ev": ev, **kw}
        self.f.write(json.dumps(kw, default=lambda o: hex(o) if isinstance(o, int) else str(o)) + "\n")
        self.f.flush()

    def close(self):
        if self.f:
            self.f.flush()
            self.f.close()
            self.f = None


class GuestFault(Exception):
    def __init__(self, msg: str, pc: int | None = None, last: str | None = None):
        super().__init__(msg)
        self.pc, self.last = pc, last


# ---------------------------------------------------------------------------------------------
# Qiling bootstrap

# Libraries whose own code runs in the emulator when Qiling has no Python version of an export: the
# C and C++ runtimes. Every other library's exports are Python only, and one Qiling does not
# implement returns 0.
NATIVE_DLLS = {"msvcr120", "msvcp120", "ucrtbase"}

# Qiling answers these with a Python stub that does nothing; the runtime code in the DLL runs instead.
TRACED_APIS = {"LoadLibraryA", "LoadLibraryW", "LoadLibraryExW", "GetModuleHandleA", "GetModuleHandleW", "GetProcAddress", "RaiseException"}
NATIVE_APIS = {"_initterm", "_initterm_e"}
# Qiling's sprintf, printf and wprintf read the variadic arguments through a pointer in r8 (they treat them as
# vsprintf-style); under the x64 ABI the arguments are in r8, r9 and on the stack, so the C runtime's own code runs.
NATIVE_APIS.update({"sprintf", "printf", "wprintf"})

# Windows functions Qiling does not implement and the plugin's runtime reaches. Each is a minimal answer
# for a process with no desktop; see README.md (stand-ins).
WIN_STUBS: dict = {}


def win_stub(name):
    def deco(f):
        WIN_STUBS[name] = f
        return f

    return deco


def win_ret(ql, value, nargs: int = 0):
    """Return from a Win64 function called at its first instruction."""
    regs = ql.arch.regs
    regs.write("rax", value & 0xFFFFFFFFFFFFFFFF)
    sp = regs.arch_sp
    ret = int.from_bytes(bytes(ql.mem.read(sp, 8)), "little")
    regs.write("rsp", sp + 8)
    regs.arch_pc = ret


def win_arg(ql, i: int) -> int:
    regs = ql.arch.regs
    if i < 4:
        return regs.read(ARG_REGS[i])
    return int.from_bytes(bytes(ql.mem.read(regs.arch_sp + STACK_ARG0 + 8 * (i - 4), 8)), "little")


def wstr(ql, addr: int, limit: int = 2048) -> str:
    out = bytearray()
    while len(out) < limit:
        c = bytes(ql.mem.read(addr + len(out), 2))
        if c == bytes(2):
            break
        out += c
    return out.decode("utf-16le", "replace")


@win_stub("GetProcessWindowStation")
def _gpws(emu, ql):
    win_ret(ql, 0)


@win_stub("GetUserObjectInformationW")
def _guoi(emu, ql):
    win_ret(ql, 0)


@win_stub("GetActiveWindow")
def _gaw(emu, ql):
    win_ret(ql, 0)


@win_stub("GetLastActivePopup")
def _glap(emu, ql):
    win_ret(ql, 0)


@win_stub("MessageBoxW")
def _mbw(emu, ql):
    text, cap = wstr(ql, win_arg(ql, 1)), wstr(ql, win_arg(ql, 2))
    emu.rec.log("messagebox", text_chars=len(text), caption_chars=len(cap))
    emu.messages.append(f"<message box, {len(text)} characters>")
    win_ret(ql, 1)


@win_stub("MessageBoxA")
def _mba(emu, ql):
    text = emu.cstr(win_arg(ql, 1))
    emu.rec.log("messagebox", text_chars=len(text), caption_chars=len(emu.cstr(win_arg(ql, 2))))
    emu.messages.append(f"<message box, {len(text)} characters>")
    win_ret(ql, 1)


def _exit_stub(kind, code_arg=0):
    def f(emu, ql):
        code = win_arg(ql, code_arg) & 0xFFFFFFFF if code_arg is not None else None
        emu.rec.log("process_exit", via=kind, code=code)
        emu.exit_reason = f"the plugin called {kind}" + (f"({code})" if code is not None else "")
        ql.emu_stop()

    return f


for _n, _a in (("ExitProcess", 0), ("exit", 0), ("_exit", 0), ("abort", None), ("TerminateProcess", 1), ("FatalExit", 0)):
    WIN_STUBS[_n] = _exit_stub(_n, _a)

ARG_REGS = ("rcx", "rdx", "r8", "r9")

STACK_ARG0 = 0x28           # offset of the fifth argument from rsp at function entry


def _make_unimplemented_safe():
    """Qiling's own crash report disassembles the code at the faulting address; this build
    never does that, so the report is replaced by the address and registers only."""
    from qiling.os.windows.windows import QlOsWindows

    def emu_error(self):
        pc = self.ql.arch.regs.arch_pc
        img = self.ql.loader.find_containing_image(pc)
        where = f" ({os.path.basename(img.path)} + {pc - img.base:#x})" if img else ""
        self.ql.log.error(f"emulation error at {pc:#x}{where}")

    QlOsWindows.emu_error = emu_error


class Emu:
    """Qiling plus the helpers the mock host needs: guest memory, calls into the guest and
    host functions behind trampolines."""

    ARENA = 0x300000000
    ARENA_SIZE = 0x40000000
    TRAMP = 0x2F0000000
    TRAMP_SIZE = 0x100000
    STUB = 0x2F8000000
    MAIN_STACK = 0x100000       # bytes of stack of the main guest thread, see _main_stack
    HEAP = 0x400000000
    HEAP_SIZE = 0x80000000
    # floating point control state at the start of program execution, from the x64 calling convention (Microsoft Learn,
    # "x64 Calling Convention", sections MXCSR and FPCSR): every exception masked, round to nearest, no denormals-are-zero,
    # no flush to zero; x87 precision control 10B (double)
    MXCSR_DEFAULT = 0x1F80
    FPCW_DEFAULT = 0x27F

    def __init__(self, plugin: str, rootfs: str, rec: Record, lay: Layouts):
        from qiling import Qiling
        from qiling.const import QL_VERBOSE

        self.rec, self.lay = rec, lay
        _make_unimplemented_safe()
        # Qiling hooks every instruction to find API entries; the same hook is put on the API
        # addresses only (hook_apis below), which is the same behaviour at native speed.
        orig = Qiling.hook_code

        winapi_hook = []

        def hook_code(ql_self, callback, *a, **kw):
            h = orig(ql_self, callback, *a, **kw)
            if getattr(callback, "__name__", "") == "hook_winapi":
                winapi_hook.append(h)
            return h

        Qiling.hook_code = hook_code
        import logging

        self.qlog: list[str] = []
        outer = self

        class Collect(logging.Handler):
            def emit(self, r):
                msg = r.getMessage()
                if r.levelno >= logging.INFO or msg.startswith(("Error in loading function", "Requesting imports")):
                    outer.qlog.append(f"{r.levelname}: {msg}")

        root = logging.getLogger()
        handler = Collect(level=logging.DEBUG)
        root.addHandler(handler)
        try:
            import contextlib

            verbose = QL_VERBOSE.DEBUG if os.environ.get("AE_QL_DEBUG") else QL_VERBOSE.DEFAULT
            # Qiling's own log handler writes to the stdout it finds when it is built: keep that off
            # the worker's event channel
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                self.ql = Qiling([plugin], rootfs, verbose=verbose, libcache=False)
            for h in list(self.ql.log.handlers):
                self.ql.log.removeHandler(h)
            self.ql.log.addHandler(handler)
            self.ql.log.propagate = False
            # the libraries are loaded with Qiling's own per instruction API hook (their DllMain runs
            # with it); from here on only the API addresses are hooked, which is the same behaviour
            for h in winapi_hook:
                self.ql.hook_del(h)
        finally:
            Qiling.hook_code = orig
            root.removeHandler(handler)
        ql = self.ql
        self._fp_defaults()
        self._narrow_hooks_to_unicorn()
        self.plugin = plugin
        self.base = ql.loader.pe_image_address
        self.arena_top = self.ARENA
        self._main_stack()
        ql.mem.map(self.ARENA, self.ARENA_SIZE, info="[ae host arena]")
        ql.mem.map(self.TRAMP, self.TRAMP_SIZE, info="[ae host trampolines]")
        ql.mem.write(self.TRAMP, b"\xc3" * self.TRAMP_SIZE)
        ql.mem.map(self.STUB, 0x1000, info="[ae host return stub]")
        ql.mem.write(self.STUB, b"\xc3")
        self.tramp_top = self.TRAMP
        self.heap = None
        self.tramp_names: dict[int, str] = {}
        self.last_cb = None
        self.last_api = None
        self.unimplemented: set[str] = set()
        self.hooked: set[int] = set()
        self.zeroed: set[str] = set()
        self.api_trace = bool(os.environ.get("AE_TRACE_API"))
        self.stack_top = ql.loader.stack_address + ql.loader.stack_size - 0x2000
        self.faults: list[dict] = []
        self.messages: list[str] = []
        self.api_ring = collections.deque(maxlen=40)
        self.api_counts = collections.Counter()
        self.init_errors: list[tuple[str, str]] = []
        self.exit_reason = None
        ql.hook_mem_unmapped(self._unmapped_cb)
        self._wrap_load_dll()
        self.hook_apis()

    def _fp_defaults(self):
        """Unicorn's CPU reset leaves MXCSR and the x87 control word at 0: every floating point exception unmasked (and x87
        precision 24 bit). Windows starts a program with the standard values of the x64 calling convention, and a caller that
        changed them must restore them before it invokes a callee, so the host calls the plugin with MXCSR_DEFAULT and
        FPCW_DEFAULT. With 0, msvcr120's math error path found the zero-divide exception unmasked and raised
        STATUS_FLOAT_DIVIDE_BY_ZERO, which no handler catches (CreateMotionVectors_64 at a sensitivity of 0).
        Qiling's own DllMain calls while it loads the libraries run before this. Guest threads
        share the one MXCSR: the scheduler switches the registers of Qiling's register save, which has no MXCSR."""
        from unicorn.x86_const import UC_X86_REG_FPCW, UC_X86_REG_MXCSR

        self.ql.uc.reg_write(UC_X86_REG_MXCSR, self.MXCSR_DEFAULT)
        self.ql.uc.reg_write(UC_X86_REG_FPCW, self.FPCW_DEFAULT)

    def _main_stack(self):
        """Qiling's Windows x86-64 profile maps 256 KB of stack for the main thread (0x40000 bytes at 0x7ffffffde000). A Windows
        thread gets the stack reserve of the program's header, 1 MB with the linker's default. The renders of three plugins
        need up to 0x59000 bytes on the main thread (measured), so 256 KB ran out. The stack
        is extended downwards to MAIN_STACK bytes and NT_TIB.StackLimit in the TEB (winnt.h NT_TIB, TEB+0x10) is moved to the
        new low end: the whole stack is committed memory. `main_stack` keeps (StackBase, StackLimit) for win_threads."""
        ql = self.ql
        high = ql.loader.stack_address + ql.loader.stack_size
        low = ql.loader.stack_address
        if self.MAIN_STACK > ql.loader.stack_size:
            low = high - self.MAIN_STACK
            ql.mem.map(low, ql.loader.stack_address - low, info="[stack]")
            ql.loader.TEB.StackLimit = low
        self.main_stack = (high, low)

    def _narrow_hooks_to_unicorn(self):
        """Qiling runs all its code hooks through one Python callback that checks every registered hook's range
        at each executed instruction. With the API address hooks (about 900) that is some 200 million Python calls
        for 235 thousand instructions: about 1.5 thousand instructions per second (measured in speed_probe.py).
        A code hook with a finite range is given to Unicorn directly instead, which filters by address in C;
        the callbacks and what they do are unchanged."""
        from unicorn import UC_HOOK_CODE

        ql = self.ql
        original = ql.hook_code

        def hook_code(callback, user_data=None, begin=1, end=0, *args, **kw):
            if end < begin:
                return original(callback, user_data, begin, end, *args, **kw)

            def raw(uc, address, size, _ud):
                if user_data is None:
                    callback(ql, address, size)
                else:
                    callback(ql, address, size, user_data)

            return ql.uc.hook_add(UC_HOOK_CODE, raw, None, begin, end)

        ql.hook_code = hook_code

        # Qiling's Windows thread scheduler is a code hook on every instruction (Python, ~35 microseconds each).
        # This harness runs the guest serially (win_semaphore.py): guest threads are not scheduled, so it goes.
        # Measured: with it 1.74 million instructions in 120 s under cProfile, 51 s of that in the scheduler.
        hooks = ql._hook.get(UC_HOOK_CODE, [])
        def is_scheduler(cb):
            if getattr(cb, "__name__", "") == "__thread_scheduler":
                return True
            return any(callable(c.cell_contents) and getattr(c.cell_contents, "__name__", "") == "__thread_scheduler"
                       for c in (getattr(cb, "__closure__", None) or ()))

        hooks[:] = [h for h in hooks if not is_scheduler(h.callback)]
        if not hooks and UC_HOOK_CODE in ql._hook_fuc:
            ql._h_uc.hook_del(ql._hook_fuc.pop(UC_HOOK_CODE))

    # -- library start -------------------------------------------------------------------------

    INIT_DLLS = ("msvcr120.dll", "msvcp120.dll")
    # harness limit, not a stand-in: emulation time allowed for one library's DllMain (msvcp120 needs 16 to 17 s alone)
    INIT_TIMEOUT = 180

    def init_libraries(self):
        """DllMain(DLL_PROCESS_ATTACH) of the C and C++ runtime libraries, whose code runs in the
        emulator. Qiling runs DllMain while it loads, but stops doing so after the first one that fails."""
        images = {os.path.basename(i.path).lower(): i for i in self.ql.loader.images}
        for name in self.INIT_DLLS:
            img = images.get(name)
            if img is None:
                continue
            nt = self.r32(img.base + 0x3C)
            entry = self.r32(img.base + nt + 0x18 + 0x10)
            if entry == 0:
                continue
            t0 = time.perf_counter()
            try:
                r = self.call(img.base + entry, [img.base, 1, 0], name, timeout=self.INIT_TIMEOUT)
                self.rec.log("dllmain", dll=name, ret=r & 0xFFFFFFFF, seconds=round(time.perf_counter() - t0, 3))
                if not (r & 0xFFFFFFFF):
                    raise GuestFault(f"{name} rejected DLL_PROCESS_ATTACH", img.base + entry, name)
            except GuestFault as e:
                self.rec.log("dllmain", dll=name, error=str(e), last=e.last)
                self.init_errors.append((name, str(e)))
                raise GuestFault(f"runtime initialization failed: {name}: {e}", e.pc, e.last) from e

    # -- memory -------------------------------------------------------------------------------

    def alloc(self, size: int, align: int = 16) -> int:
        a = (self.arena_top + align - 1) & ~(align - 1)
        self.arena_top = a + max(size, 1)
        if self.arena_top > self.ARENA + self.ARENA_SIZE:
            raise MemoryError("host arena exhausted")
        return a

    def rd(self, addr: int, n: int) -> bytes:
        return bytes(self.ql.mem.read(addr, n))

    def wr(self, addr: int, data: bytes):
        self.ql.mem.write(addr, bytes(data))

    def r32(self, a):
        return U32.unpack(self.rd(a, 4))[0]

    def ri32(self, a):
        return struct.unpack("<i", self.rd(a, 4))[0]

    def r64(self, a):
        return U64.unpack(self.rd(a, 8))[0]

    def w32(self, a, v):
        self.wr(a, struct.pack("<I", v & 0xFFFFFFFF))

    def w64(self, a, v):
        self.wr(a, struct.pack("<Q", v & 0xFFFFFFFFFFFFFFFF))

    def cstr(self, a: int, limit: int = 4096) -> str:
        out = bytearray()
        while len(out) < limit:
            b = self.rd(a + len(out), 1)
            if b == b"\0":
                break
            out += b
        return out.decode("latin-1")

    def view(self, addr: int, name: str) -> View:
        return View(self, addr, name)

    def new(self, name: str) -> View:
        a = self.alloc(self.lay.size(name))
        self.wr(a, b"\0" * self.lay.size(name))
        return View(self, a, name)

    def _unmapped_cb(self, ql, access, address, size, value):
        """Records the first unmapped access (the address, not what the code there is)."""
        kind = {16: "read", 17: "write", 18: "fetch", 19: "read", 20: "write", 21: "fetch"}.get(access, str(access))
        pc = ql.arch.regs.arch_pc
        self.faults.append({"access": kind, "address": address, "size": size, "pc": pc})
        self.rec.log("fault", access=kind, address=address, size=size, pc=pc, last_api=self.last_api, last_cb=self.last_cb)
        return False

    # -- API hooks (Windows) ------------------------------------------------------------------

    def _wrap_load_dll(self):
        from qiling.loader.pe import Process

        orig = Process.load_dll
        emu = self

        def load_dll(proc, name, is_driver=False):
            r = orig(proc, name, is_driver)
            if emu.rec:
                emu.rec.log("load_dll", dll=name, base=r)
            if emu.hooked:
                emu.hook_apis()
            return r

        Process.load_dll = load_dll

    def _trace_args(self, name, ql):
        """Argument values as numbers. No string is read from guest memory: names the plugin passes to the loader
        (LoadLibrary, GetProcAddress) and exception messages are plugin data and are not logged."""
        a = [win_arg(ql, i) for i in range(4)]
        if name == "RaiseException":
            return [hex(a[0]), a[1], a[2]]
        return [hex(x) for x in a]

    @staticmethod
    def _api_name(const, entry) -> str:
        if entry["name"] is not None:
            return entry["name"].decode()
        return const.Mapper.get(entry["dll"], {}).get(entry["ordinal"], f"ord{entry['ordinal']}")

    def hook_apis(self):
        """Python hooks on the exports that have a Python version (Qiling's or one of ours). Exports of
        a library that runs no code here and has none return 0 straight from the emulator's memory
        (xor eax, eax; ret); exports of the C runtimes without a Python version run natively. Hooking
        every export would make every translated instruction pay for tens of thousands of hooks."""
        from qiling.os.windows import const
        import qiling.os.windows.dlls as api
        from qiling.const import QL_INTERCEPT

        ql = self.ql
        user = ql.os.user_defined_api[QL_INTERCEPT.CALL]
        for a, entry in list(ql.loader.import_symbols.items()):
            if a in self.hooked:
                continue
            self.hooked.add(a)
            name = entry["name"]
            name = self._api_name(const, entry)
            if name in NATIVE_APIS:
                continue
            if name in WIN_STUBS or name in user or hasattr(api, f"hook_{name}"):
                ql.hook_code(self._api_cb, begin=a, end=a)
            elif entry["dll"] not in NATIVE_DLLS:
                ql.mem.write(a, bytes([0x31, 0xC0, 0xC3]))
                self.zeroed.add(name)

    def _api_cb(self, ql, address, size):
        from qiling.os.windows import const
        from qiling.const import QL_INTERCEPT
        import qiling.os.windows.dlls as api

        entry = ql.loader.import_symbols[address]
        name = entry["name"]
        name = self._api_name(const, entry)
        self.last_api = name
        if name in TRACED_APIS:
            self.rec.log("api_args", name=name, args=self._trace_args(name, ql))
        self.api_ring.append(name)
        self.api_counts[name] += 1
        if name in NATIVE_APIS:
            return
        stub = WIN_STUBS.get(name)
        if stub is not None:
            self.rec.log("api_stub", name=name)
            stub(self, ql)
            return
        fn = ql.os.user_defined_api[QL_INTERCEPT.CALL].get(name) or getattr(api, f"hook_{name}", None)
        if self.api_trace:
            self.rec.log("api", name=name, dll=entry["dll"], impl=bool(fn))
        if fn:
            try:
                fn(ql, address, name)
            except Exception as ex:  # noqa: BLE001
                self.rec.log("api_error", name=name, error=f"{type(ex).__name__}: {ex}")
                raise
        else:
            if name not in self.unimplemented:
                self.unimplemented.add(name)
                self.rec.log("api_unimplemented", name=name, dll=entry["dll"], native=entry["dll"] in NATIVE_DLLS)
            if entry["dll"] not in NATIVE_DLLS:
                # no Windows code runs in the emulator except the C runtime: the call returns 0
                win_ret(ql, 0)

    # -- host functions behind trampolines ---------------------------------------------------

    def argv(self, i: int, kind: str):
        regs = self.ql.arch.regs
        if kind in "df" and i < 4:
            raw = regs.read(f"xmm{i}") & 0xFFFFFFFFFFFFFFFF
        elif i < 4:
            raw = regs.read(ARG_REGS[i])
        else:
            raw = self.r64(regs.arch_sp + STACK_ARG0 + 8 * (i - 4))
        if kind == "d":
            return struct.unpack("<d", U64.pack(raw))[0]
        if kind == "f":
            return struct.unpack("<f", U32.pack(raw & 0xFFFFFFFF))[0]
        if kind == "i":
            v = raw & 0xFFFFFFFF
            return v - (1 << 32) if v >> 31 else v
        if kind == "u":
            return raw & 0xFFFFFFFF
        if kind == "h":
            v = raw & 0xFFFF
            return v - (1 << 16) if v >> 15 else v
        if kind == "H":
            return raw & 0xFFFF
        if kind == "c":
            v = raw & 0xFF
            return v - 256 if v >> 7 else v
        if kind == "b":
            return raw & 0xFF
        if kind == "q":
            return raw - (1 << 64) if raw >> 63 else raw
        return raw

    def ret(self, kind: str, v):
        regs = self.ql.arch.regs
        if kind == "d":
            regs.write("xmm0", U64.unpack(struct.pack("<d", float(v)))[0])
        elif kind == "f":
            regs.write("xmm0", U32.unpack(struct.pack("<f", float(v)))[0])
        elif kind != "v":
            regs.write("rax", int(v) & 0xFFFFFFFFFFFFFFFF)

    def tramp(self, name: str, handler, sig: str, ret: str = "i") -> int:
        """A guest address that runs `handler(*args)` when the guest calls it."""
        addr = self.tramp_top
        self.tramp_top += 16
        if self.tramp_top > self.TRAMP + self.TRAMP_SIZE:
            raise MemoryError("trampoline area exhausted")
        self.tramp_names[addr] = name

        def cb(ql, address, size):
            self.last_cb = name
            args = [self.argv(i, k) for i, k in enumerate(sig)]
            result = handler(*args)
            if ret != "v" and result is not None:
                self.ret(ret, result)

        self.ql.hook_code(cb, begin=addr, end=addr)
        return addr

    # -- calling the guest ---------------------------------------------------------------------

    def call(self, addr: int, args: list[int], label: str = "", timeout: float = 0) -> int:
        """Calls a guest function (Win64 ABI, integer arguments) and returns rax."""
        from unicorn import UcError

        regs = self.ql.arch.regs
        extra = max(0, len(args) - 4)
        frame = 0x20 + 8 * extra
        sp = self.stack_top - frame - 8
        sp -= (sp - 8) % 16
        self.w64(sp, self.STUB)
        for i, a in enumerate(args):
            v = a & 0xFFFFFFFFFFFFFFFF
            if i < 4:
                regs.write(ARG_REGS[i], v)
            else:
                self.w64(sp + STACK_ARG0 + 8 * (i - 4), v)
        regs.write("rsp", sp)
        self.last_cb = None
        self.exit_reason = None
        try:
            self.ql.emu_start(addr, self.STUB, timeout=int(timeout * 1e6))
            if self.exit_reason:
                raise GuestFault(self.exit_reason, regs.arch_pc, self.last_api)
            if regs.arch_pc != self.STUB:
                img = self.ql.loader.find_containing_image(regs.arch_pc)
                where = f"{os.path.basename(img.path)}+{regs.arch_pc - img.base:#x}" if img else f"{regs.arch_pc:#x}"
                snap = {r: hex(regs.read(r)) for r in ("rax", "rcx", "rdx", "rbx", "rsi", "rdi", "r8", "r9", "rsp")}
                self.rec.log("timeout", where=where, regs=snap, last_api=self.last_api)
                raise GuestFault(f"timed out after {timeout:g} s at {where} {snap}", regs.arch_pc, self.last_api)
        except UcError as e:
            pc = regs.arch_pc
            img = self.ql.loader.find_containing_image(pc)
            where = f"{os.path.basename(img.path)}+{pc - img.base:#x}" if img else f"{pc:#x}"
            raise GuestFault(f"{e} at {where}", pc, self.last_cb or self.last_api) from None
        return regs.read("rax")

    def call_nested(self, addr: int, args: list[int]) -> int:
        """A guest call made from inside a host function (the plugin called the host, the host
        calls a function pointer of the plugin)."""
        from unicorn import UcError

        regs = self.ql.arch.regs
        saved_sp = regs.arch_sp
        ctx = self.ql.uc.context_save()
        extra = max(0, len(args) - 4)
        sp = saved_sp - 0x2000 - 0x20 - 8 * extra - 8
        sp -= (sp - 8) % 16
        self.w64(sp, self.STUB)
        for i, a in enumerate(args):
            v = a & 0xFFFFFFFFFFFFFFFF
            if i < 4:
                regs.write(ARG_REGS[i], v)
            else:
                self.w64(sp + STACK_ARG0 + 8 * (i - 4), v)
        regs.write("rsp", sp)
        try:
            self.ql.uc.emu_start(addr, self.STUB)
        except UcError as e:
            pc = regs.arch_pc
            raise GuestFault(f"{e} in a nested call at {pc:#x}", pc, self.last_cb or self.last_api) from None
        rax = regs.read("rax")
        self.ql.uc.context_restore(ctx)
        return rax


# ---------------------------------------------------------------------------------------------
# The process heap. Qiling's HeapSize/HeapReAlloc/realloc do not agree with each other, which sends
# the C runtime's __dllonexit into a loop, so the heap and C allocation functions are answered here
# with one allocator that remembers the size of every block.

class GuestHeap:
    ALIGN = 16

    def __init__(self, emu, base: int, size: int):
        self.emu, self.base, self.top, self.end = emu, base, base, base + size
        self.sizes: dict[int, int] = {}
        self.free_lists: dict[int, list[int]] = {}

    def alloc(self, size: int, zero: bool = False) -> int:
        n = max(16, (size + self.ALIGN - 1) & ~(self.ALIGN - 1))
        lst = self.free_lists.get(n)
        if lst:
            p = lst.pop()
        else:
            p = self.top
            self.top += n + 16
            if self.top > self.end:
                return 0
        self.sizes[p] = size
        if zero or lst is not None:
            self.emu.wr(p, bytes(n))
        return p

    def free(self, p: int):
        size = self.sizes.pop(p, None)
        if size is not None:
            n = max(16, (size + self.ALIGN - 1) & ~(self.ALIGN - 1))
            self.free_lists.setdefault(n, []).append(p)

    def realloc(self, p: int, size: int, zero: bool = False) -> int:
        if p == 0:
            return self.alloc(size, zero)
        old = self.sizes.get(p)
        if old is None:
            return 0
        if size <= old:
            self.sizes[p] = size
            return p
        q = self.alloc(size, zero)
        if q:
            self.emu.wr(q, self.emu.rd(p, old))
            if zero:
                self.emu.wr(q + old, bytes(size - old))
            self.free(p)
        return q


def _heap(emu) -> GuestHeap:
    if emu.heap is None:
        emu.ql.mem.map(emu.HEAP, emu.HEAP_SIZE, info="[ae host heap]")
        emu.heap = GuestHeap(emu, emu.HEAP, emu.HEAP_SIZE)
    return emu.heap


@win_stub("HeapAlloc")
def _heap_alloc(emu, ql):
    win_ret(ql, _heap(emu).alloc(win_arg(ql, 2), bool(win_arg(ql, 1) & 8)))


@win_stub("HeapReAlloc")
def _heap_realloc(emu, ql):
    win_ret(ql, _heap(emu).realloc(win_arg(ql, 2), win_arg(ql, 3), bool(win_arg(ql, 1) & 8)))


@win_stub("HeapSize")
def _heap_size(emu, ql):
    win_ret(ql, _heap(emu).sizes.get(win_arg(ql, 2), 0xFFFFFFFFFFFFFFFF))


@win_stub("HeapFree")
def _heap_free(emu, ql):
    _heap(emu).free(win_arg(ql, 2))
    win_ret(ql, 1)


@win_stub("malloc")
def _malloc(emu, ql):
    win_ret(ql, _heap(emu).alloc(win_arg(ql, 0)))


@win_stub("calloc")
def _calloc(emu, ql):
    win_ret(ql, _heap(emu).alloc(win_arg(ql, 0) * win_arg(ql, 1), True))


@win_stub("realloc")
def _realloc(emu, ql):
    win_ret(ql, _heap(emu).realloc(win_arg(ql, 0), win_arg(ql, 1)))


@win_stub("free")
def _free(emu, ql):
    _heap(emu).free(win_arg(ql, 0))
    win_ret(ql, 0)


@win_stub("_msize")
def _msize(emu, ql):
    win_ret(ql, _heap(emu).sizes.get(win_arg(ql, 0), 0xFFFFFFFFFFFFFFFF))


# Install before any Emu instance scans the Windows export table.
from win_semaphore import install as _install_semaphores

_install_semaphores(WIN_STUBS, win_arg, win_ret, GuestFault)

from win_rtl import install as _install_rtl

_install_rtl(WIN_STUBS, win_arg, win_ret, GuestFault)

from win_crypt import install as _install_crypt

_install_crypt(WIN_STUBS, win_arg, win_ret, GuestFault)

from win_seh import install as _install_seh

_install_seh(WIN_STUBS, win_arg, win_ret, GuestFault)

from win_threads import install as _install_threads

_install_threads(WIN_STUBS, win_arg, win_ret, GuestFault)

from win_io import install as _install_io

_install_io(WIN_STUBS, win_arg, win_ret, GuestFault)

from win_path import install as _install_path

_install_path(WIN_STUBS, win_arg, win_ret, GuestFault)
