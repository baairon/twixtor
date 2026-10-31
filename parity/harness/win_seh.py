"""x64 structured exception dispatch for guest throws, from the OS's documented model.

What the OS does when a throw is raised in guest code, done here at run time:
  RaiseException builds an EXCEPTION_RECORD and a CONTEXT, then the first pass (RtlDispatchException)
  walks the frames with the loaded images' exception directories (RtlLookupFunctionEntry,
  RtlVirtualUnwind) and calls each frame's language handler with the x64 ABI
  (handler(EXCEPTION_RECORD*, EstablisherFrame, CONTEXT*, DISPATCHER_CONTEXT*)). A handler that finds a
  catch calls RtlUnwindEx, which is the second pass: it walks the frames again, calls the handlers with
  EXCEPTION_UNWINDING, and resumes at the target. The language handlers (__CxxFrameHandler3 and the funclets
  it runs) are the real guest code of msvcr120.dll; only the dispatch loop and the Rtl* functions are Python.
  A throw nobody handles stops the run (an unhandled exception); it is never answered by returning.

Parts that are Python: RaiseException, RtlLookupFunctionEntry, RtlVirtualUnwind, RtlUnwindEx,
RtlCaptureContext, RtlRestoreContext and the dispatch loop. Limits: no epilogue detection in
RtlVirtualUnwind (a control address is treated as inside the function body, which holds for call return
addresses), no dynamic function tables, no CPU faults, and no __C_specific_handler (added only if a throw hits it).

The exception tables are read only to unwind; nothing from them is logged. The log has the exception
code and flags, the raising address (module and offset for Microsoft DLLs, module name only for the plugin),
the handler module that ran, handled or unhandled, and the module of the frame that resumed.
"""
import bisect
import os
import struct

UNW_FLAG_EHANDLER, UNW_FLAG_UHANDLER, UNW_FLAG_CHAININFO = 1, 2, 4

EH_NONCONTINUABLE, EH_UNWINDING, EH_EXIT_UNWIND = 1, 2, 4
EH_NESTED_CALL, EH_TARGET_UNWIND, EH_COLLIDED_UNWIND = 0x10, 0x20, 0x40

ExceptionContinueExecution, ExceptionContinueSearch, ExceptionNestedException, ExceptionCollidedUnwind = 0, 1, 2, 3

STATUS_UNWIND = 0x80000026
STATUS_UNWIND_CONSOLIDATE = 0x80000029
INFO_CONTEXT = 10        # ExceptionInformation[10] points at the context during a consolidate callback
STATUS_NONCONTINUABLE_EXCEPTION = 0xC0000025
STATUS_INVALID_DISPOSITION = 0xC0000026
STATUS_INVALID_UNWIND_TARGET = 0xC0000029

CONTEXT_SIZE = 0x4D0
CONTEXT_FULL = 0x10000B
GPR = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"]
CTX_OFF = {n: 0x78 + 8 * i for i, n in enumerate(GPR)}
CTX_OFF["rip"] = 0xF8
XMM_OFF = 0x1A0
MAX_FRAMES = 600


class UnwindError(Exception):
    pass


def new_ctx():
    c = {n: 0 for n in GPR}
    c.update(rip=0, eflags=0x202, mxcsr=0x1F80, xmm={})
    return c


def copy_ctx(c):
    d = dict(c)
    d["xmm"] = dict(c.get("xmm", {}))
    return d


def ctx_to_bytes(c):
    b = bytearray(CONTEXT_SIZE)
    struct.pack_into("<I", b, 0x30, CONTEXT_FULL)
    struct.pack_into("<I", b, 0x34, c.get("mxcsr", 0x1F80))
    struct.pack_into("<I", b, 0x118, c.get("mxcsr", 0x1F80))
    struct.pack_into("<I", b, 0x44, c.get("eflags", 0x202))
    for n, off in CTX_OFF.items():
        struct.pack_into("<Q", b, off, c[n] & 0xFFFFFFFFFFFFFFFF)
    for i, v in c.get("xmm", {}).items():
        b[XMM_OFF + 16 * i:XMM_OFF + 16 * i + 16] = (v & ((1 << 128) - 1)).to_bytes(16, "little")
    return bytes(b)


def ctx_from_bytes(b):
    c = new_ctx()
    c["mxcsr"] = struct.unpack_from("<I", b, 0x34)[0]
    c["eflags"] = struct.unpack_from("<I", b, 0x44)[0]
    for n, off in CTX_OFF.items():
        c[n] = struct.unpack_from("<Q", b, off)[0]
    for i in range(16):
        c["xmm"][i] = int.from_bytes(b[XMM_OFF + 16 * i:XMM_OFF + 16 * i + 16], "little")
    return c


class Mem:
    """Reads with errors; `rd(addr, n)` returns bytes or None."""

    def __init__(self, rd):
        self.rd = rd

    def get(self, a, n):
        b = self.rd(a, n)
        if b is None or len(b) != n:
            raise UnwindError(f"unreadable {a:#x}")
        return b

    def u8(self, a):
        return self.get(a, 1)[0]

    def u16(self, a):
        return struct.unpack("<H", self.get(a, 2))[0]

    def u32(self, a):
        return struct.unpack("<I", self.get(a, 4))[0]

    def u64(self, a):
        return struct.unpack("<Q", self.get(a, 8))[0]

    def u128(self, a):
        return int.from_bytes(self.get(a, 16), "little")


def virtual_unwind(m, htype, base, pc, rf, ctx):
    """RtlVirtualUnwind: unwinds `ctx` one frame in place. rf = (begin, end, unwind) RVAs.
    Returns (handler address or 0, handler data address or 0, establisher frame)."""
    begin, _end, unwind = rf
    off = pc - (base + begin)
    ui = base + (unwind & ~3)
    first = True
    frame = ctx["rsp"]
    machframe = False
    for _ in range(16):
        b0, prolog, ncodes, fr = m.u8(ui), m.u8(ui + 1), m.u8(ui + 2), m.u8(ui + 3)
        flags = b0 >> 3
        frame_reg, frame_off = fr & 15, (fr >> 4) * 16
        codes = m.get(ui + 4, ncodes * 2) if ncodes else b""
        done = off if first else None       # None: the prolog is complete
        if first and frame_reg and (done >= prolog or any(codes[2 * k + 1] & 15 == 3 and codes[2 * k] <= done for k in range(ncodes))):
            frame = ctx[GPR[frame_reg]] - frame_off
        i = 0
        while i < ncodes:
            co, bb = codes[2 * i], codes[2 * i + 1]
            op, info = bb & 15, bb >> 4
            slots = {1: 2 if info == 0 else 3, 4: 2, 5: 3, 8: 2, 9: 3}.get(op, 1)
            if done is None or done >= co:
                if op == 0:
                    ctx[GPR[info]] = m.u64(ctx["rsp"])
                    ctx["rsp"] += 8
                elif op == 1:
                    n = codes[2 * (i + 1)] | codes[2 * (i + 1) + 1] << 8
                    ctx["rsp"] += n * 8 if info == 0 else int.from_bytes(codes[2 * (i + 1):2 * (i + 1) + 4], "little")
                elif op == 2:
                    ctx["rsp"] += info * 8 + 8
                elif op == 3:
                    ctx["rsp"] = ctx[GPR[frame_reg]] - frame_off
                elif op == 4:
                    ctx[GPR[info]] = m.u64(ctx["rsp"] + (codes[2 * (i + 1)] | codes[2 * (i + 1) + 1] << 8) * 8)
                elif op == 5:
                    ctx[GPR[info]] = m.u64(ctx["rsp"] + int.from_bytes(codes[2 * (i + 1):2 * (i + 1) + 4], "little"))
                elif op == 8:
                    ctx["xmm"][info] = m.u128(ctx["rsp"] + (codes[2 * (i + 1)] | codes[2 * (i + 1) + 1] << 8) * 16)
                elif op == 9:
                    ctx["xmm"][info] = m.u128(ctx["rsp"] + int.from_bytes(codes[2 * (i + 1):2 * (i + 1) + 4], "little"))
                elif op == 10:
                    sp = ctx["rsp"] + (8 if info else 0)
                    ctx["rip"] = m.u64(sp)
                    ctx["rsp"] = m.u64(sp + 24)
                    machframe = True
            i += slots
        tail = ui + 4 + ((ncodes + 1) & ~1) * 2
        if flags & UNW_FLAG_CHAININFO:
            b2, _e2, u2 = m.u32(tail), m.u32(tail + 4), m.u32(tail + 8)
            ui = base + (u2 & ~3)
            first = False
            continue
        handler = hdata = 0
        if flags & htype & 3:
            handler = base + m.u32(tail)
            hdata = tail + 4
        if not machframe:
            ctx["rip"] = m.u64(ctx["rsp"])
            ctx["rsp"] += 8
        return handler, hdata, frame
    raise UnwindError("unwind chain too long")


class Task:
    def __init__(self, kind, gen, sp, orig_ctx=None):
        self.kind, self.gen, self.sp, self.orig_ctx = kind, gen, sp, orig_ctx


class Seh:
    """The dispatcher. `host` gives rd, wr, alloc, images, plugin_base, stack_base, log; `cpu` gives
    get_ctx, set_ctx and start_call (see RealHost and RealCpu below, and the fakes in test_win_seh.py)."""

    def __init__(self, host, cpu):
        self.host, self.cpu = host, cpu
        self.m = Mem(host.rd)
        self.tasks = []
        self.pdata = {}
        self.marker = None
        self.marker_ctx = {}    # stack slot of a return-to-marker address -> the context unwinding continues in
        self.result = None      # what the last run ended with, for tests

    # -- modules -------------------------------------------------------------------------------

    def image_at(self, a):
        for img in self.host.images():
            if img.base <= a < img.end:
                return img
        return None

    def module(self, a, offset=True):
        """module name, plus +offset for Microsoft DLLs; the plugin is named without an offset."""
        img = self.image_at(a)
        if img is None:
            return "unmapped" if not a else "outside every image"
        name = os.path.basename(img.path)
        if img.base == self.host.plugin_base or not offset:
            return name
        return f"{name}+{a - img.base:#x}"

    def table(self, img):
        if img.base in self.pdata:
            return self.pdata[img.base]
        out = None
        try:
            nt = self.m.u32(img.base + 0x3C)
            if self.m.u16(img.base + nt + 0x18) == 0x20B:
                rva = self.m.u32(img.base + nt + 0x18 + 0x88)
                size = self.m.u32(img.base + nt + 0x18 + 0x8C)
                if rva and size:
                    blob = self.m.get(img.base + rva, size)
                    rows = [struct.unpack_from("<III", blob, i) for i in range(0, len(blob) - 11, 12)]
                    out = (rva, rows, [r[0] for r in rows])
        except UnwindError:
            out = None
        self.pdata[img.base] = out
        return out

    def lookup(self, pc):
        """RtlLookupFunctionEntry: (image, row, entry address) or None."""
        img = self.image_at(pc)
        if img is None:
            return None
        t = self.table(img)
        if not t:
            return None
        rva, rows, begins = t
        k = bisect.bisect_right(begins, pc - img.base) - 1
        if k >= 0 and rows[k][0] <= pc - img.base < rows[k][1]:
            return img, rows[k], img.base + rva + 12 * k
        return None

    def cross_marker(self, ctx):
        """Unwinding reached the return address of a call the dispatcher made: the frames above it are the
        dispatcher's, and the walk continues in the context that call stands in for (None: nothing above)."""
        cont = self.marker_ctx.get(ctx["rsp"] - 8)
        return copy_ctx(cont) if cont else None

    def unwind_one(self, ctx, htype):
        """One frame: returns (caller context, handler, handler data, establisher frame, function entry address, image base)."""
        new = copy_ctx(ctx)
        pc = ctx["rip"]
        hit = self.lookup(pc)
        if hit is None:
            frame = new["rsp"]
            new["rip"] = self.m.u64(new["rsp"])
            new["rsp"] += 8
            return new, 0, 0, frame, 0, 0
        img, rf, entry = hit
        handler, hdata, frame = virtual_unwind(self.m, htype, img.base, pc, rf, new)
        return new, handler, hdata, frame, entry, img.base

    # -- records in guest memory ---------------------------------------------------------------

    def put_ctx(self, ctx):
        a = self.host.alloc(CONTEXT_SIZE)
        self.host.wr(a, ctx_to_bytes(ctx))
        return a

    def put_dc(self, ctx, ctx_addr, base, entry, est, handler, hdata, target_ip):
        a = self.host.alloc(0x50)
        self.host.wr(a, struct.pack("<QQQQQQQQQII", ctx["rip"], base, entry, est, target_ip, ctx_addr, handler, hdata, 0, 0, 0))
        return a

    def rec_flags(self, rec):
        return self.m.u32(rec + 4)

    def set_rec_flags(self, rec, flags):
        self.host.wr(rec + 4, struct.pack("<I", flags & 0xFFFFFFFF))

    def new_record(self, code, flags, params, address):
        a = self.host.alloc(0x98)
        b = bytearray(0x98)
        struct.pack_into("<IIQQI", b, 0, code, flags, 0, address, len(params))
        for i, p in enumerate(params[:15]):
            struct.pack_into("<Q", b, 0x20 + 8 * i, p)
        self.host.wr(a, bytes(b))
        return a

    # -- the two passes ------------------------------------------------------------------------

    def dispatch_gen(self, rec, ctx0):
        code = self.m.u32(rec)
        ctx = copy_ctx(ctx0)
        for _ in range(MAX_FRAMES):
            if ctx["rip"] == self.marker:
                ctx = self.cross_marker(ctx)
                if ctx is None:
                    break
            if ctx["rip"] == 0 or ctx["rsp"] >= self.host.stack_base:
                break
            try:
                new, handler, hdata, est, entry, base = self.unwind_one(ctx, UNW_FLAG_EHANDLER)
            except UnwindError as ex:
                yield ("unhandled", f"unwind failed ({ex})")
            if handler:
                ca = self.put_ctx(ctx)
                dc = self.put_dc(ctx, ca, base, entry, est, handler, hdata, 0)
                disp = (yield ("call", handler, [rec, est, ca, dc], ctx0)) & 0xFFFFFFFF
                self.host.log("seh_handler", pass_="first", code=hex(code), handler_module=self.module(handler, False), disposition=disp)
                if disp == ExceptionContinueExecution:
                    if self.rec_flags(rec) & EH_NONCONTINUABLE:
                        yield ("unhandled", "a handler continued a non-continuable exception (STATUS_NONCONTINUABLE_EXCEPTION)")
                    yield ("jump", ctx0, "continue execution")
                elif disp != ExceptionContinueSearch:
                    yield ("unhandled", f"invalid disposition {disp} (STATUS_INVALID_DISPOSITION)")
            ctx = new
        yield ("unhandled", "no handler found")

    def unwind_gen(self, target_frame, target_ip, rec, retval, start_ctx, ctx_out):
        flags = self.rec_flags(rec) | EH_UNWINDING | (0 if target_frame else EH_EXIT_UNWIND)
        self.set_rec_flags(rec, flags)
        ctx = copy_ctx(start_ctx)
        final = None
        for _ in range(MAX_FRAMES):
            if ctx["rip"] == self.marker:
                ctx = self.cross_marker(ctx)
                if ctx is None:
                    break
            if ctx["rip"] == 0 or ctx["rsp"] >= self.host.stack_base:
                break
            try:
                new, handler, hdata, est, entry, base = self.unwind_one(ctx, UNW_FLAG_UHANDLER)
            except UnwindError as ex:
                yield ("unhandled", f"unwind failed ({ex})")
            if target_frame and est > target_frame:
                break
            if handler:
                if est == target_frame:
                    flags |= EH_TARGET_UNWIND
                self.set_rec_flags(rec, flags)
                ca = self.put_ctx(ctx)
                dc = self.put_dc(ctx, ca, base, entry, est, handler, hdata, target_ip)
                disp = (yield ("call", handler, [rec, est, ca, dc], start_ctx)) & 0xFFFFFFFF
                flags &= ~(EH_TARGET_UNWIND | EH_COLLIDED_UNWIND)
                self.set_rec_flags(rec, flags)
                self.host.log("seh_handler", pass_="unwind", code=hex(self.m.u32(rec)), handler_module=self.module(handler, False), disposition=disp)
                if disp not in (ExceptionContinueSearch, ExceptionCollidedUnwind):
                    yield ("unhandled", f"invalid disposition {disp} in the unwind pass (STATUS_INVALID_DISPOSITION)")
            if est == target_frame:
                final = ctx
                break
            ctx = new
        if final is None:
            yield ("unhandled", "unwind target frame not found (STATUS_INVALID_UNWIND_TARGET)")
        final = copy_ctx(final)
        final["rax"] = retval
        consolidate = self.m.u32(rec) == STATUS_UNWIND_CONSOLIDATE and self.m.u32(rec + 0x18) >= 1
        if consolidate:
            # the unwind lands at the target frame, then the callback in ExceptionInformation[0] runs (for C++
            # it runs the catch block) and returns the address to continue at; it sees the context through
            # ExceptionInformation[10]
            ca = self.put_ctx(final)
            self.host.wr(rec + 0x20 + 8 * INFO_CONTEXT, struct.pack("<Q", ca))
            callback = self.m.u64(rec + 0x20)
            cont = yield ("call", callback, [rec], final)
            final = ctx_from_bytes(self.m.get(ca, CONTEXT_SIZE))
            final["rip"] = cont
        elif target_ip:
            final["rip"] = target_ip
        if ctx_out:
            self.host.wr(ctx_out, ctx_to_bytes(final))
        yield ("jump", final, "unwind complete")

    # -- the driver -----------------------------------------------------------------------------

    def start(self, kind, gen, sp, orig_ctx=None):
        self.tasks.append(Task(kind, gen, sp, orig_ctx))
        self.drive(None)

    def drive(self, value):
        try:
            self._drive(value)
        except Exception as ex:  # noqa: BLE001
            self.tasks.clear()
            self.finish_unhandled(f"internal error in the dispatcher ({type(ex).__name__})")

    def _drive(self, value):
        while self.tasks:
            t = self.tasks[-1]
            try:
                req = t.gen.send(value)
            except StopIteration:
                self.tasks.clear()
                return self.finish_unhandled("dispatcher ended without a result")
            if req[0] == "call":
                _, addr, args, cont = req
                self.marker_ctx[t.sp] = copy_ctx(cont)
                self.cpu.start_call(addr, args, t.sp, self.marker)
                return
            if req[0] == "jump":
                _, ctx, why = req
                self.discard_below(ctx["rsp"])
                self.result = ("jump", ctx)
                self.host.log("seh_resume", resumed_module=self.module(ctx["rip"], False), why=why)
                self.cpu.set_ctx(ctx)
                return
            self.tasks.clear()
            self.marker_ctx.clear()
            return self.finish_unhandled(req[1])

    def discard_below(self, rsp):
        """A resume at `rsp` kills the frames below it: dispatches started there are over, while the ones
        whose calls sit above (a throw inside a catch block, resumed within it) stay live."""
        self.tasks = [t for t in self.tasks if t.sp > rsp]
        self.marker_ctx = {k: v for k, v in self.marker_ctx.items() if k > rsp}

    def finish_unhandled(self, reason):
        self.result = ("unhandled", reason)
        self.host.log("seh_unhandled", reason=reason)
        self.cpu.stop(f"unhandled exception: {reason}")

    def on_return(self):
        if self.tasks:
            self.drive(self.cpu.get_rax())

    # -- entry points used by the stubs ---------------------------------------------------------

    def sp_below(self, rsp):
        return ((rsp - 0x2000) & ~0xF) - 8

    def raise_exception(self, code, flags, params, ctx0, api_addr):
        rec = self.new_record(code & 0xFFFFFFFF, flags & 1, params, api_addr)
        self.host.log("seh_raise", code=hex(code & 0xFFFFFFFF), flags=hex(flags & 1), nparams=len(params),
                      raised_at=self.module(ctx0["rip"]))
        self.start("dispatch", self.dispatch_gen(rec, ctx0), self.sp_below(ctx0["rsp"]), orig_ctx=ctx0)

    def unwind_ex(self, target_frame, target_ip, rec, retval, ctx_out, start_ctx):
        if not rec:
            rec = self.new_record(STATUS_UNWIND, 0, [], 0)
        self.start("unwind", self.unwind_gen(target_frame, target_ip, rec, retval, start_ctx, ctx_out), self.sp_below(start_ctx["rsp"]))

    def virtual_unwind_api(self, htype, base, pc, rf_addr, ctx_addr, hdata_out, est_out):
        ctx = ctx_from_bytes(self.m.get(ctx_addr, CONTEXT_SIZE))
        rf = struct.unpack("<III", self.m.get(rf_addr, 12)) if rf_addr else None
        if rf is None:
            est = ctx["rsp"]
            ctx["rip"] = self.m.u64(ctx["rsp"])
            ctx["rsp"] += 8
            handler = hdata = 0
        else:
            handler, hdata, est = virtual_unwind(self.m, htype, base, pc, rf, ctx)
        old = bytearray(self.m.get(ctx_addr, CONTEXT_SIZE))
        new = ctx_to_bytes(ctx)
        old[0x78:0x100] = new[0x78:0x100]
        for i in ctx["xmm"]:
            old[XMM_OFF + 16 * i:XMM_OFF + 16 * i + 16] = new[XMM_OFF + 16 * i:XMM_OFF + 16 * i + 16]
        self.host.wr(ctx_addr, bytes(old))
        if hdata_out:
            self.host.wr(hdata_out, struct.pack("<Q", hdata))
        if est_out:
            self.host.wr(est_out, struct.pack("<Q", est))
        return handler

    def lookup_api(self, pc, base_out):
        hit = self.lookup(pc)
        if base_out:
            self.host.wr(base_out, struct.pack("<Q", hit[0].base if hit else 0))
        return hit[2] if hit else 0


# ---------------------------------------------------------------------------------------------
# the real host and CPU

class RealHost:
    def __init__(self, emu):
        self.emu = emu
        ql = emu.ql
        self.plugin_base = emu.base
        self.stack_base = ql.loader.stack_address + ql.loader.stack_size

    def rd(self, a, n):
        try:
            return bytes(self.emu.ql.mem.read(a, n))
        except Exception:  # noqa: BLE001
            return None

    def wr(self, a, data):
        self.emu.ql.mem.write(a, bytes(data))

    def alloc(self, n):
        return self.emu.alloc(n)

    def images(self):
        return self.emu.ql.loader.images

    def log(self, ev, **kw):
        if "pass_" in kw:
            kw["pass"] = kw.pop("pass_")
        self.emu.rec.log(ev, **kw)


class RealCpu:
    def __init__(self, emu, host):
        self.emu, self.host = emu, host
        from unicorn import x86_const as c

        self.c = c
        self.ids = {n: getattr(c, "UC_X86_REG_" + n.upper()) for n in GPR + ["rip"]}

    @property
    def uc(self):
        return self.emu.ql.uc

    def get_ctx(self):
        c = new_ctx()
        for n, rid in self.ids.items():
            c[n] = self.uc.reg_read(rid)
        c["eflags"] = self.uc.reg_read(self.c.UC_X86_REG_EFLAGS)
        try:
            c["mxcsr"] = self.uc.reg_read(self.c.UC_X86_REG_MXCSR)
            for i in range(6, 16):
                c["xmm"][i] = self.uc.reg_read(getattr(self.c, f"UC_X86_REG_XMM{i}"))
        except Exception:  # noqa: BLE001
            pass
        return c

    def set_ctx(self, c):
        for n, rid in self.ids.items():
            self.uc.reg_write(rid, c[n] & 0xFFFFFFFFFFFFFFFF)
        self.uc.reg_write(self.c.UC_X86_REG_EFLAGS, c.get("eflags", 0x202) & 0xFFFFFFFF)
        try:
            self.uc.reg_write(self.c.UC_X86_REG_MXCSR, c.get("mxcsr", 0x1F80))
            for i, v in c.get("xmm", {}).items():
                if i >= 6:
                    self.uc.reg_write(getattr(self.c, f"UC_X86_REG_XMM{i}"), v)
        except Exception:  # noqa: BLE001
            pass

    def start_call(self, addr, args, sp, marker):
        self.host.wr(sp, struct.pack("<Q", marker))
        for rid, v in zip(("rcx", "rdx", "r8", "r9"), args):
            self.uc.reg_write(self.ids[rid], v & 0xFFFFFFFFFFFFFFFF)
        self.uc.reg_write(self.ids["rsp"], sp)
        self.uc.reg_write(self.ids["rip"], addr)

    def get_rax(self):
        return self.uc.reg_read(self.ids["rax"])

    def stop(self, reason):
        self.emu.exit_reason = reason
        self.emu.ql.emu_stop()


def get_seh(emu):
    s = getattr(emu, "_seh", None)
    if s is None:
        host = RealHost(emu)
        s = emu._seh = Seh(host, RealCpu(emu, host))
        s.marker = emu.tramp_top
        emu.tramp_top += 16
        emu.ql.hook_code(lambda ql, a, size: s.on_return(), begin=s.marker, end=s.marker)
    return s


def install(stubs, arg, ret, fault_type=None):
    """Register the handlers before Emu.hook_apis scans exports."""
    def caller_ctx(emu, ql):
        """The context of the caller of the hooked function: rip is the return address, rsp is past it."""
        s = get_seh(emu)
        c = s.cpu.get_ctx()
        c["rip"] = s.m.u64(c["rsp"])
        c["rsp"] += 8
        return s, c

    def raise_exception(emu, ql):
        s = get_seh(emu)
        api = ql.arch.regs.arch_pc
        code, flags, n, ptr = arg(ql, 0), arg(ql, 1), arg(ql, 2) & 0xFFFFFFFF, arg(ql, 3)
        params = []
        if ptr and n:
            for i in range(min(n, 15)):
                params.append(s.m.u64(ptr + 8 * i))
        _, c = caller_ctx(emu, ql)
        s.raise_exception(code, flags, params, c, api)

    def lookup_function_entry(emu, ql):
        s = get_seh(emu)
        ret(ql, s.lookup_api(arg(ql, 0), arg(ql, 1)))

    def virtual_unwind_stub(emu, ql):
        s = get_seh(emu)
        ret(ql, s.virtual_unwind_api(arg(ql, 0) & 0xFFFFFFFF, arg(ql, 1), arg(ql, 2), arg(ql, 3), arg(ql, 4), arg(ql, 5), arg(ql, 6)))

    def unwind_ex(emu, ql):
        s, c = caller_ctx(emu, ql)
        s.unwind_ex(arg(ql, 0), arg(ql, 1), arg(ql, 2), arg(ql, 3), arg(ql, 4), c)

    def capture_context(emu, ql):
        s, c = caller_ctx(emu, ql)
        ctx_addr = arg(ql, 0)
        s.host.wr(ctx_addr, ctx_to_bytes(c))
        ret(ql, c["rax"])

    def restore_context(emu, ql):
        s = get_seh(emu)
        c = ctx_from_bytes(s.m.get(arg(ql, 0), CONTEXT_SIZE))
        s.discard_below(c["rsp"])
        s.host.log("seh_resume", resumed_module=s.module(c["rip"], False), why="RtlRestoreContext")
        s.cpu.set_ctx(c)

    stubs["RaiseException"] = raise_exception
    stubs["RtlLookupFunctionEntry"] = lookup_function_entry
    stubs["RtlVirtualUnwind"] = virtual_unwind_stub
    stubs["RtlUnwindEx"] = unwind_ex
    stubs["RtlCaptureContext"] = capture_context
    stubs["RtlRestoreContext"] = restore_context
    # the harness logs the message text of a C++ throw it reads from the thrown object: the plugin's own data
    try:
        import sys

        host = sys.modules.get("ae_host")
        if host is not None:
            host.TRACED_APIS.discard("RaiseException")
    except Exception:  # noqa: BLE001
        pass
