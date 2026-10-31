"""Counted, unnamed Win32 semaphores for the serial harness execution model.

An infinite wait with no token needs guest scheduling. Stop in that case rather
than inventing a successful acquisition and accepting an invalid reference frame.
"""
from dataclasses import dataclass
import struct

WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 258
WAIT_FAILED = 0xFFFFFFFF
INFINITE = 0xFFFFFFFF
ERROR_INVALID_HANDLE = 6
ERROR_INVALID_PARAMETER = 87
ERROR_TOO_MANY_POSTS = 298
WOULD_BLOCK = "an unsignalled semaphore requires guest thread scheduling"


def signed_long(value):
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


@dataclass
class Semaphore:
    count: int
    maximum: int

    def release(self, amount):
        if amount <= 0:
            return None, ERROR_INVALID_PARAMETER
        if self.count + amount > self.maximum:
            return None, ERROR_TOO_MANY_POSTS
        previous = self.count
        self.count += amount
        return previous, 0

    def acquire(self, timeout):
        if self.count:
            self.count -= 1
            return WAIT_OBJECT_0
        if timeout == INFINITE:
            raise RuntimeError(WOULD_BLOCK)
        return WAIT_TIMEOUT


def install(stubs, arg, ret, fault_type, handle_factory=None):
    """Register address-hook handlers before Emu.hook_apis scans exports."""
    def new_handle(ql, obj):
        if handle_factory is None:
            from qiling.os.windows.handle import Handle
            handle = Handle(obj=obj)
        else:
            handle = handle_factory(obj)
        ql.os.handle_manager.append(handle)
        return handle.id

    def create(emu, ql):
        initial, maximum = signed_long(arg(ql, 1)), signed_long(arg(ql, 2))
        if maximum <= 0 or initial < 0 or initial > maximum:
            ql.os.last_error = ERROR_INVALID_PARAMETER
            ret(ql, 0)
            return
        if arg(ql, 3):
            raise fault_type("named semaphores are not implemented by this serial shim",
                             ql.arch.regs.arch_pc, "CreateSemaphore")
        handle = new_handle(ql, Semaphore(initial, maximum))
        emu.rec.log("semaphore_create", handle=handle, initial=initial, maximum=maximum)
        ret(ql, handle)

    def release(emu, ql):
        handle_id, amount, previous_ptr = arg(ql, 0), signed_long(arg(ql, 1)), arg(ql, 2)
        handle = ql.os.handle_manager.get(handle_id)
        if handle is None or not isinstance(handle.obj, Semaphore):
            ql.os.last_error = ERROR_INVALID_HANDLE
            ret(ql, 0)
            return
        previous, error = handle.obj.release(amount)
        if error:
            ql.os.last_error = error
            ret(ql, 0)
            return
        if previous_ptr:
            ql.mem.write(previous_ptr, struct.pack('<i', previous))
        emu.rec.log("semaphore_release", handle=handle_id, amount=amount, previous=previous)
        ret(ql, 1)

    def wait(name):
        def handler(emu, ql):
            handle_id, timeout = arg(ql, 0), arg(ql, 1) & 0xFFFFFFFF
            handle = ql.os.handle_manager.get(handle_id)
            if handle is None:
                ql.os.last_error = ERROR_INVALID_HANDLE
                ret(ql, WAIT_FAILED)
                return
            if not isinstance(handle.obj, Semaphore):
                import qiling.os.windows.dlls as api
                getattr(api, 'hook_' + name)(ql, ql.arch.regs.arch_pc, name)
                return
            try:
                result = handle.obj.acquire(timeout)
            except RuntimeError as ex:
                raise fault_type(str(ex), ql.arch.regs.arch_pc, name) from None
            emu.rec.log("semaphore_wait", handle=handle_id, timeout=timeout, result=result)
            ret(ql, result)
        return handler

    stubs['CreateSemaphoreA'] = create
    stubs['CreateSemaphoreW'] = create
    stubs['ReleaseSemaphore'] = release
    stubs['WaitForSingleObject'] = wait('WaitForSingleObject')
    stubs['WaitForSingleObjectEx'] = wait('WaitForSingleObjectEx')
