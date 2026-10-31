#!/usr/bin/env python3
"""Writes the six empty registry hives Qiling opens at start (a stand-in: no keys, no values).

Each file is a regf header and one hbin holding a single root key with no subkeys and no values,
which python-registry (what Qiling uses) opens. usage: make_hive.py REGISTRY_DIR
"""
import struct
import sys
from pathlib import Path


def hive(name: str) -> bytes:
    root = b"ROOT"
    nk = bytearray(0x4C + len(root))
    nk[0:2] = b"nk"
    struct.pack_into("<H", nk, 2, 0x24)             # root key, ascii name
    struct.pack_into("<I", nk, 0x10, 0xFFFFFFFF)    # parent
    struct.pack_into("<I", nk, 0x1C, 0xFFFFFFFF)    # no subkey list
    struct.pack_into("<I", nk, 0x20, 0xFFFFFFFF)
    struct.pack_into("<I", nk, 0x28, 0xFFFFFFFF)    # no value list
    struct.pack_into("<I", nk, 0x2C, 0xFFFFFFFF)    # no security
    struct.pack_into("<I", nk, 0x30, 0xFFFFFFFF)    # no class name
    struct.pack_into("<H", nk, 0x48, len(root))
    nk[0x4C:] = root
    cell_size = (4 + len(nk) + 7) & ~7
    cell = struct.pack("<i", -cell_size) + bytes(nk) + b"\0" * (cell_size - 4 - len(nk))
    hbin = bytearray(0x1000)
    hbin[0:4] = b"hbin"
    struct.pack_into("<I", hbin, 0x4, 0)
    struct.pack_into("<I", hbin, 0x8, 0x1000)
    hbin[0x20:0x20 + len(cell)] = cell
    free = 0x1000 - 0x20 - len(cell)
    struct.pack_into("<i", hbin, 0x20 + len(cell), free)
    regf = bytearray(0x1000)
    regf[0:4] = b"regf"
    struct.pack_into("<II", regf, 4, 1, 1)
    struct.pack_into("<I", regf, 0x14, 1)           # major
    struct.pack_into("<I", regf, 0x18, 3)           # minor
    struct.pack_into("<I", regf, 0x20, 1)           # file format
    struct.pack_into("<I", regf, 0x24, 0x20)        # root cell, relative to the first hbin
    struct.pack_into("<I", regf, 0x28, 0x1000)      # size of the hbins
    regf[0x30:0x30 + 2 * len(name)] = name.encode("utf-16le")
    xsum = 0
    for i in range(0, 0x1FC, 4):
        xsum ^= struct.unpack_from("<I", regf, i)[0]
    xsum = 1 if xsum == 0 else 0xFFFFFFFE if xsum == 0xFFFFFFFF else xsum
    struct.pack_into("<I", regf, 0x1FC, xsum)
    return bytes(regf) + bytes(hbin)


def main(out: str) -> None:
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    for n in ("SECURITY", "SAM", "SOFTWARE", "SYSTEM", "HARDWARE", "NTUSER.DAT"):
        (d / n).write_bytes(hive(n))


if __name__ == "__main__":
    main(sys.argv[1])
