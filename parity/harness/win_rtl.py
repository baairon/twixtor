"""RtlPcToFileHeader from the loader's image list.

PVOID RtlPcToFileHeader(PVOID PcValue, PVOID *BaseOfImage): the base of the loaded image that contains
PcValue, written to *BaseOfImage (when that pointer is non-null) and returned; NULL in both when no
loaded image contains it. Before this, the export was a generic fallback that returned 0 and wrote nothing.
"""
import struct


def image_base(images, pc):
    """Base of the image with base <= pc < end, or 0."""
    for img in images:
        if img.base <= pc < img.end:
            return img.base
    return 0


def install(stubs, arg, ret, fault_type=None):
    """Register the handler before Emu.hook_apis scans exports (a name in `stubs` is hooked, not zero patched)."""
    def rtl_pc_to_file_header(emu, ql):
        pc, out = arg(ql, 0), arg(ql, 1)
        base = image_base(ql.loader.images, pc)
        if out:
            ql.mem.write(out, struct.pack('<Q', base))
        emu.rec.log("rtl_pc_to_file_header", pc=pc, out=out, base=base)
        ret(ql, base)

    stubs['RtlPcToFileHeader'] = rtl_pc_to_file_header
