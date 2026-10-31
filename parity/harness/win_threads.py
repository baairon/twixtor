"""Guest threads for the serial harness: CreateThread, ResumeThread, ExitThread, GetCurrentThreadId and waits on
threads, with cooperative scheduling.

A thread runs until it blocks (a wait on a semaphore with no token, or on a thread that has not ended) or ends
(returns from its start routine or calls ExitThread); only then another runnable thread runs, in creation order.
The guest is serial and deterministic: no preemption, so one run gives one answer. Plugins whose threads share data
without synchronisation, or poll a variable written by another thread, would need preemption and are not served.

Qiling's own scheduler (a Python callback on every instruction, switching every 10 instructions) is not used: it made
the emulator about 500 times slower (see README, harness speed changes), and Qiling has no ResumeThread.
A deadlock (every thread blocked) stops the run with a message, it is never answered by returning.
"""
import struct

from win_semaphore import Semaphore, WOULD_BLOCK, WAIT_OBJECT_0, WAIT_TIMEOUT, INFINITE

CREATE_SUSPENDED = 4
MIN_STACK = 0x100000


def is_ready(w):
    return (w.count > 0) if isinstance(w, Semaphore) else (w.state == "done")


class Thread:
    def __init__(self, tid, ctx=None, suspended=False, stack=None):
        self.id, self.ctx, self.suspended = tid, ctx, suspended
        self.stack = stack            # (StackBase, StackLimit) of the thread's own stack, written to the TEB while it runs
        self.state = "ready"          # ready | blocked | done
        self.wait = None              # what a blocked thread waits for: a Semaphore or a Thread
        self.exit_code = 0


class Scheduler:
    def __init__(self, cpu, alloc, main_id=1):
        """cpu: save() -> dict of registers, restore(dict), stop(reason). alloc(n) -> address."""
        self.cpu, self.alloc = cpu, alloc
        self.main = Thread(main_id)
        self.threads = [self.main]
        self.cur = self.main
        self.next_id = main_id + 1
        self.end_marker = None

    # -- state -------------------------------------------------------------------------------------

    def runnable(self, t):
        if t.state == "done" or t.suspended:
            return False
        if t.state == "blocked":
            if not any(is_ready(w) for w in (t.wait if isinstance(t.wait, list) else [t.wait])):
                return False
            t.state, t.wait = "ready", None
        return True

    def pick(self, after):
        n = len(self.threads)
        i = self.threads.index(after)
        for k in range(1, n + 1):
            t = self.threads[(i + k) % n]
            if t is not after and self.runnable(t):
                return t
        return None

    def switch_to(self, t):
        self.cur = t
        self.cpu.restore(t.ctx)
        if t.stack is not None:
            self.cpu.set_stack(*t.stack)

    # -- operations --------------------------------------------------------------------------------

    def create(self, start, param, stack_size, flags, regs_now):
        size = max(stack_size or 0, MIN_STACK)
        base = self.alloc(size)
        sp = ((base + size - 0x40) & ~0xF) - 8                 # [sp] holds the return address, rsp % 16 == 8 at entry
        ctx = dict(regs_now)
        ctx.update(rip=start, rcx=param, rsp=sp, rbp=0)
        t = Thread(self.next_id, ctx, suspended=bool(flags & CREATE_SUSPENDED), stack=(base + size, base))
        self.next_id += 1
        self.cpu.write_u64(sp, self.end_marker)
        self.threads.append(t)
        return t

    def resume(self, t):
        was = t.suspended
        t.suspended = False
        return 1 if was else 0

    def block(self, waiting_for):
        """The current thread waits at the API entry it is on (it will re-run the call when it wakes). False when
        no other thread can run, which is a deadlock."""
        other = self.pick(self.cur)
        if other is None:
            return False
        self.cur.ctx = self.cpu.save()
        self.cur.state, self.cur.wait = "blocked", waiting_for
        self.switch_to(other)
        return True

    def exit_current(self, code=0):
        self.cur.state, self.cur.exit_code = "done", code
        other = self.pick(self.cur)
        if other is None:
            self.cpu.stop("deadlock: the last runnable guest thread ended or every other thread is blocked")
            return False
        self.switch_to(other)
        return True


# ---------------------------------------------------------------------------------------------
# the real CPU and the stubs

class RealCpu:
    def __init__(self, emu):
        self.emu = emu

    def save(self):
        return self.emu.ql.arch.regs.save()

    def restore(self, ctx):
        self.emu.ql.arch.regs.restore(ctx)

    def write_u64(self, addr, value):
        self.emu.ql.mem.write(addr, struct.pack("<Q", value))

    def set_stack(self, base, limit):
        """NT_TIB.StackBase and StackLimit (winnt.h NT_TIB; TEB+0x08 and TEB+0x10 on x64) of the thread that runs. Windows
        keeps them per thread (every thread has its own TEB). With the main thread's values left in place while a worker ran
        on its stack in the host arena, far below the main stack, the worker's stack probe walked down from the main stack's
        limit into unmapped memory."""
        teb = self.emu.ql.loader.TEB
        teb.StackBase, teb.StackLimit = base, limit

    def stop(self, reason):
        self.emu.exit_reason = reason
        self.emu.ql.emu_stop()


def get_scheduler(emu):
    s = getattr(emu, "_sched", None)
    if s is None:
        s = emu._sched = Scheduler(RealCpu(emu), emu.alloc)
        s.main.stack = getattr(emu, "main_stack", None)
        s.end_marker = emu.tramp_top
        emu.tramp_top += 16
        emu.ql.hook_code(lambda ql, a, size: s.exit_current(0), begin=s.end_marker, end=s.end_marker)
    return s


def install(stubs, arg, ret, fault_type=None):
    """After win_semaphore.install: wraps its waits and adds the thread functions."""
    def new_handle(ql, obj):
        from qiling.os.windows.handle import Handle

        h = Handle(obj=obj)
        ql.os.handle_manager.append(h)
        return h.id

    def create_thread(emu, ql):
        s = get_scheduler(emu)
        stack, start, param, flags, tid_p = arg(ql, 1), arg(ql, 2), arg(ql, 3), arg(ql, 4) & 0xFFFFFFFF, arg(ql, 5)
        t = s.create(start, param, stack, flags, ql.arch.regs.save())
        if tid_p:
            ql.mem.write(tid_p, struct.pack("<I", t.id))
        emu.rec.log("thread_create", thread=t.id, suspended=t.suspended)
        ret(ql, new_handle(ql, t))

    def resume_thread(emu, ql):
        s = get_scheduler(emu)
        h = ql.os.handle_manager.get(arg(ql, 0))
        if h is None or not isinstance(h.obj, Thread):
            ret(ql, 0xFFFFFFFF)
            return
        emu.rec.log("thread_resume", thread=h.obj.id)
        ret(ql, s.resume(h.obj))

    def exit_thread(emu, ql):
        s = get_scheduler(emu)
        emu.rec.log("thread_exit", thread=s.cur.id)
        s.exit_current(arg(ql, 0) & 0xFFFFFFFF)

    def current_thread_id(emu, ql):
        ret(ql, get_scheduler(emu).cur.id)

    def wrap_wait(name):
        orig = stubs[name]

        def wait(emu, ql):
            s = get_scheduler(emu)
            h = ql.os.handle_manager.get(arg(ql, 0))
            timeout = arg(ql, 1) & 0xFFFFFFFF
            if h is not None and isinstance(h.obj, Thread):
                if h.obj.state == "done":
                    ret(ql, WAIT_OBJECT_0)
                elif timeout != INFINITE:
                    ret(ql, WAIT_TIMEOUT)
                elif not s.block(h.obj):
                    s.cpu.stop(f"deadlock: wait on a thread that cannot end (thread {s.cur.id})")
                return
            try:
                orig(emu, ql)
            except Exception as ex:  # noqa: BLE001
                if WOULD_BLOCK not in str(ex):
                    raise
                tid = s.cur.id
                if not s.block(h.obj):
                    raise
                emu.rec.log("thread_block", thread=tid, on="semaphore")

        return wait

    def wait_multiple(emu, ql):
        """WaitForMultipleObjects over semaphores and threads, in handle order."""
        s = get_scheduler(emu)
        n, array, wait_all, timeout = arg(ql, 0) & 0xFFFFFFFF, arg(ql, 1), arg(ql, 2) & 0xFFFFFFFF, arg(ql, 3) & 0xFFFFFFFF
        objs = []
        for i in range(n):
            h = ql.os.handle_manager.get(struct.unpack("<Q", bytes(ql.mem.read(array + 8 * i, 8)))[0])
            if h is None or not isinstance(h.obj, (Semaphore, Thread)):
                emu.rec.log("wait_multiple_unsupported_object", index=i)
                ret(ql, 0xFFFFFFFF)                                  # WAIT_FAILED
                return
            objs.append(h.obj)
        ready = [is_ready(o) for o in objs]
        if (all(ready) if wait_all else any(ready)):
            picks = range(n) if wait_all else [ready.index(True)]
            for i in picks:
                if isinstance(objs[i], Semaphore):
                    objs[i].count -= 1
            ret(ql, WAIT_OBJECT_0 + (0 if wait_all else picks[0]))
        elif timeout != INFINITE:
            ret(ql, WAIT_TIMEOUT)
        else:
            tid = s.cur.id
            if not s.block(objs):
                s.cpu.stop("deadlock: WaitForMultipleObjects that nothing can satisfy")
            else:
                emu.rec.log("thread_block", thread=tid, on="multiple objects")

    stubs["WaitForMultipleObjects"] = wait_multiple
    stubs["CreateThread"] = create_thread
    stubs["ResumeThread"] = resume_thread
    stubs["ExitThread"] = exit_thread
    stubs["GetCurrentThreadId"] = current_thread_id
    for n in ("WaitForSingleObject", "WaitForSingleObjectEx"):
        stubs[n] = wrap_wait(n)
