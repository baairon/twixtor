"""Offline checks of the x64 exception dispatch: unwinding a function through its own tables, the
disposition handling and the unhandled path. The image is a small fake PE built in memory."""
import struct
import types
import unittest

import ae_host
import win_seh as seh

BASE = 0x10000
SIZE = 0x2000
STUB = 0x2F8000000


def build_image(mem):
    """f1 (0x100..0x180): push rbp; sub rsp,0x20, no handler.  f2 (0x200..0x280): no prolog, handlers E+U."""
    b = BASE
    struct.pack_into("<I", mem, b + 0x3C, 0x80)
    struct.pack_into("<H", mem, b + 0x80 + 0x18, 0x20B)
    struct.pack_into("<II", mem, b + 0x80 + 0x18 + 0x88, 0x1000, 24)
    struct.pack_into("<III", mem, b + 0x1000, 0x100, 0x180, 0x1800)
    struct.pack_into("<III", mem, b + 0x100C, 0x200, 0x280, 0x1840)
    # unwind info of f1: version 1, prolog size 5, two codes (reverse order): alloc 0x20 at 5, push rbp at 1
    mem[b + 0x1800:b + 0x1808] = bytes([1, 5, 2, 0, 5, 0x32, 1, 0x50])
    # unwind info of f2: flags E+U, no codes, handler RVA 0x300 and one data dword
    mem[b + 0x1840:b + 0x1844] = bytes([1 | (3 << 3), 0, 0, 0])
    struct.pack_into("<II", mem, b + 0x1844, 0x300, 0xDA7A)


class FakeHost:
    def __init__(self):
        self.mem = bytearray(0x20000)
        build_image(self.mem)
        self.top = 0x14000
        self.events = []
        self.plugin_base = BASE
        self.stack_base = 0x9000
        self.img = types.SimpleNamespace(base=BASE, end=BASE + SIZE, path="C:/x/msvcr.dll")

    def rd(self, a, n):
        return bytes(self.mem[a:a + n]) if 0 <= a and a + n <= len(self.mem) else None

    def wr(self, a, data):
        self.mem[a:a + len(data)] = data

    def alloc(self, n):
        a, self.top = self.top, self.top + ((n + 15) & ~15)
        return a

    def images(self):
        return [self.img]

    def log(self, ev, **kw):
        self.events.append((ev, kw))


class FakeCpu:
    def __init__(self):
        self.calls, self.ctx, self.stopped, self.rax = [], None, None, 0

    def start_call(self, addr, args, sp, marker):
        self.calls.append((addr, list(args), sp, marker))

    def set_ctx(self, c):
        self.ctx = c

    def get_rax(self):
        return self.rax

    def stop(self, reason):
        self.stopped = reason


def frame_ctx():
    c = seh.new_ctx()
    c.update(rip=BASE + 0x120, rsp=0x8000, rbp=0x1111)
    return c


class SehTests(unittest.TestCase):
    def setUp(self):
        self.host, self.cpu = FakeHost(), FakeCpu()
        self.s = seh.Seh(self.host, self.cpu)
        self.s.marker = 0x2F0000100
        # stack: saved rbp and the return address into f2, then f2's own return address 0 (end of stack)
        self.host.wr(0x8020, struct.pack("<QQQ", 0xAAAA, BASE + 0x220, 0))

    def raise_it(self, flags=0):
        self.s.raise_exception(0xE06D7363, flags, [0x19930520, 1, 2, BASE], frame_ctx(), 0xABC)

    def test_unwind_through_its_own_tables(self):
        c = frame_ctx()
        hit = self.s.lookup(c["rip"])
        handler, hdata, frame = seh.virtual_unwind(self.s.m, 1, BASE, c["rip"], hit[1], c)
        self.assertEqual((handler, hdata, frame), (0, 0, 0x8000))
        self.assertEqual((c["rip"], c["rsp"], c["rbp"]), (BASE + 0x220, 0x8030, 0xAAAA))

    def test_unwind_inside_the_prolog_applies_only_the_executed_codes(self):
        c = seh.new_ctx()
        c.update(rip=BASE + 0x102, rsp=0x8020, rbp=0x2222)
        self.host.wr(0x8020, struct.pack("<QQ", 0xBBBB, BASE + 0x220))
        hit = self.s.lookup(c["rip"])
        seh.virtual_unwind(self.s.m, 1, BASE, c["rip"], hit[1], c)
        self.assertEqual((c["rip"], c["rsp"], c["rbp"]), (BASE + 0x220, 0x8030, 0xBBBB))

    def test_handler_is_returned_for_the_asked_type_with_its_data(self):
        c = seh.new_ctx()
        c.update(rip=BASE + 0x220, rsp=0x8030)
        hit = self.s.lookup(c["rip"])
        handler, hdata, frame = seh.virtual_unwind(self.s.m, seh.UNW_FLAG_EHANDLER, BASE, c["rip"], hit[1], c)
        self.assertEqual(handler, BASE + 0x300)
        self.assertEqual(struct.unpack("<I", self.host.rd(hdata, 4))[0], 0xDA7A)
        self.assertEqual(frame, 0x8030)
        c2 = seh.new_ctx()
        c2.update(rip=BASE + 0x220, rsp=0x8030)
        self.assertEqual(seh.virtual_unwind(self.s.m, 0, BASE, c2["rip"], hit[1], c2)[0], 0)

    def test_lookup_misses_outside_a_function_and_an_image(self):
        self.assertIsNone(self.s.lookup(BASE + 0x190))
        self.assertIsNone(self.s.lookup(0x5000))

    def test_handler_gets_the_x64_arguments(self):
        self.raise_it()
        addr, args, sp, marker = self.cpu.calls[0]
        self.assertEqual(addr, BASE + 0x300)
        rec, est, ctx_addr, dc = args
        self.assertEqual(est, 0x8030)
        self.assertEqual(struct.unpack("<II", self.host.rd(rec, 8)), (0xE06D7363, 0))
        self.assertEqual(struct.unpack("<Q", self.host.rd(rec + 0x38, 8))[0], BASE)
        self.assertEqual(struct.unpack("<Q", self.host.rd(ctx_addr + 0xF8, 8))[0], BASE + 0x220)
        control_pc, base, entry, frame = struct.unpack("<QQQQ", self.host.rd(dc, 32))
        self.assertEqual((control_pc, base, frame), (BASE + 0x220, BASE, 0x8030))
        self.assertEqual(entry, BASE + 0x1000 + 12)
        self.assertEqual(marker, 0x2F0000100)
        self.assertEqual(sp % 16, 8)

    def test_continue_search_to_the_end_of_the_stack_is_unhandled(self):
        self.raise_it()
        self.cpu.rax = seh.ExceptionContinueSearch
        self.s.on_return()
        self.assertIn("unhandled exception", self.cpu.stopped)
        self.assertEqual(self.s.result[0], "unhandled")
        events = [e for e, _ in self.host.events]
        self.assertIn("seh_raise", events)
        self.assertIn("seh_unhandled", events)
        self.assertIsNone(self.cpu.ctx)

    def test_continue_execution_resumes_a_continuable_exception_only(self):
        self.raise_it()
        self.cpu.rax = seh.ExceptionContinueExecution
        self.s.on_return()
        self.assertEqual(self.cpu.ctx["rip"], BASE + 0x120)
        self.assertIsNone(self.cpu.stopped)
        self.setUp()
        self.raise_it(flags=seh.EH_NONCONTINUABLE)
        self.cpu.rax = seh.ExceptionContinueExecution
        self.s.on_return()
        self.assertIn("non-continuable", self.cpu.stopped)

    def test_invalid_disposition_is_unhandled(self):
        self.raise_it()
        self.cpu.rax = 7
        self.s.on_return()
        self.assertIn("invalid disposition", self.cpu.stopped)

    def test_unwind_runs_the_handlers_then_resumes_at_the_target(self):
        self.raise_it()
        rec = self.cpu.calls[0][1][0]
        # the first pass handler calls RtlUnwindEx from a frame whose return address is the dispatcher marker
        start = seh.new_ctx()
        start.update(rip=self.s.marker, rsp=self.cpu.calls[0][2] + 8)
        self.s.unwind_ex(0x8030, BASE + 0x260, rec, 5, 0, start)
        addr, args, sp, _ = self.cpu.calls[-1]
        self.assertEqual(addr, BASE + 0x300)
        flags = struct.unpack("<I", self.host.rd(rec + 4, 4))[0]
        self.assertEqual(flags & (seh.EH_UNWINDING | seh.EH_TARGET_UNWIND), seh.EH_UNWINDING | seh.EH_TARGET_UNWIND)
        dc = args[3]
        self.assertEqual(struct.unpack("<Q", self.host.rd(dc + 0x20, 8))[0], BASE + 0x260)
        self.cpu.rax = seh.ExceptionContinueSearch
        self.s.on_return()
        self.assertEqual(self.cpu.ctx["rip"], BASE + 0x260)
        self.assertEqual(self.cpu.ctx["rax"], 5)
        self.assertEqual(self.cpu.ctx["rsp"], 0x8030)
        self.assertEqual(self.s.tasks, [])
        self.assertEqual([e for e, _ in self.host.events][-1], "seh_resume")

    def test_consolidate_unwind_runs_the_callback_and_resumes_where_it_says(self):
        self.raise_it()
        rec = self.s.new_record(seh.STATUS_UNWIND_CONSOLIDATE, 0, [BASE + 0x1F0], 0)
        start = seh.new_ctx()
        start.update(rip=self.s.marker, rsp=self.cpu.calls[0][2] + 8)
        self.s.unwind_ex(0x8030, BASE + 0x260, rec, 0, 0, start)
        self.cpu.rax = seh.ExceptionContinueSearch
        self.s.on_return()
        addr, args, _, _ = self.cpu.calls[-1]
        self.assertEqual((addr, args), (BASE + 0x1F0, [rec]))
        ca = struct.unpack("<Q", self.host.rd(rec + 0x20 + 8 * seh.INFO_CONTEXT, 8))[0]
        self.assertEqual(struct.unpack("<Q", self.host.rd(ca + 0x98, 8))[0], 0x8030)
        self.host.wr(ca + 0x90, struct.pack("<Q", 0x77))       # the callback changes rbx in the context it was given
        self.cpu.rax = BASE + 0x270
        self.s.on_return()
        self.assertEqual(self.cpu.ctx["rip"], BASE + 0x270)
        self.assertEqual(self.cpu.ctx["rbx"], 0x77)
        self.assertEqual(self.cpu.ctx["rsp"], 0x8030)

    def test_a_resume_inside_a_live_call_keeps_the_outer_dispatch(self):
        self.raise_it()
        outer = self.s.tasks[0]
        self.assertEqual(len(self.s.tasks), 1)
        inner_rsp = outer.sp - 0x100          # a throw inside the handler's frames, resumed within them
        ctx = seh.new_ctx()
        ctx.update(rip=BASE + 0x120, rsp=inner_rsp)
        self.s.discard_below(inner_rsp - 0x40)
        self.assertEqual(self.s.tasks, [outer])
        self.s.discard_below(outer.sp + 0x1000)
        self.assertEqual(self.s.tasks, [])
        self.assertEqual(self.s.marker_ctx, {})

    def test_unwind_to_a_frame_that_is_not_on_the_stack_is_unhandled(self):
        self.raise_it()
        rec = self.cpu.calls[0][1][0]
        start = seh.new_ctx()
        start.update(rip=self.s.marker, rsp=self.cpu.calls[0][2] + 8)
        self.s.unwind_ex(0x8100, BASE + 0x260, rec, 0, 0, start)
        self.cpu.rax = seh.ExceptionContinueSearch
        self.s.on_return()
        self.assertIn("unhandled exception", self.cpu.stopped)
        self.assertIn("target frame not found", self.cpu.stopped)

    def test_log_names_modules_without_plugin_offsets(self):
        self.s.host.plugin_base = BASE
        self.assertEqual(self.s.module(BASE + 0x220), "msvcr.dll")
        self.s.host.plugin_base = 0
        self.assertEqual(self.s.module(BASE + 0x220), "msvcr.dll+0x220")
        self.assertEqual(self.s.module(BASE + 0x220, offset=False), "msvcr.dll")

    def test_context_round_trip(self):
        c = frame_ctx()
        c["xmm"][7] = 0x1234
        back = seh.ctx_from_bytes(seh.ctx_to_bytes(c))
        self.assertEqual((back["rip"], back["rsp"], back["rbp"], back["xmm"][7]), (c["rip"], c["rsp"], c["rbp"], 0x1234))

    def test_installed_in_the_host(self):
        for name in ("RaiseException", "RtlLookupFunctionEntry", "RtlVirtualUnwind", "RtlUnwindEx", "RtlCaptureContext", "RtlRestoreContext"):
            self.assertIn(name, ae_host.WIN_STUBS)
        self.assertNotIn("RaiseException", ae_host.TRACED_APIS)


if __name__ == "__main__":
    unittest.main()
