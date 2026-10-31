"""Console output of the guest: WriteFile to the standard output and error handles is counted and dropped.

The text a plugin prints belongs to the plugin; the harness neither shows nor stores it (Qiling's own WriteFile
prints it to the process's stdout, which ended up in logs). Only the number of bytes is logged. Other handles keep
Qiling's behaviour.
"""
import struct

STD_OUTPUT_HANDLE = 0xFFFFFFF5          # GetStdHandle(-11) as Qiling hands it out
STD_ERROR_HANDLE = 0xFFFFFFF4           # GetStdHandle(-12)


def install(stubs, arg, ret, fault_type=None):
    def write_file(emu, ql):
        handle, n, written_p = arg(ql, 0) & 0xFFFFFFFF, arg(ql, 2) & 0xFFFFFFFF, arg(ql, 3)
        if handle not in (STD_OUTPUT_HANDLE, STD_ERROR_HANDLE):
            import qiling.os.windows.dlls as api
            return getattr(api, "hook_WriteFile")(ql, ql.arch.regs.arch_pc, "WriteFile")
        if written_p:
            ql.mem.write(written_p, struct.pack("<I", n))
        emu.rec.log("console_write", stream="stdout" if handle == STD_OUTPUT_HANDLE else "stderr", bytes=n)
        ret(ql, 1)

    stubs["WriteFile"] = write_file
