#!/usr/bin/env python3
"""Reads struct layouts of the After Effects SDK headers from the debug info of sdk_probe.o, and the
suite names and versions from the header text. Writes sdk_layouts.json.

usage: gen_layouts.py HEADERS_DIR PROBE_O OUT_JSON
"""
import json
import re
import sys
from pathlib import Path

from elftools.elf.elffile import ELFFile

headers, probe, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]


def type_name(die, cu):
    """A readable name for the type a DIE refers to."""
    t = die.get_DIE_from_attribute("DW_AT_type") if "DW_AT_type" in die.attributes else None
    if t is None:
        return "void"
    tag = t.tag
    if tag == "DW_TAG_pointer_type":
        return type_name(t, cu) + "*"
    if tag in ("DW_TAG_const_type", "DW_TAG_volatile_type"):
        return type_name(t, cu)
    if tag == "DW_TAG_array_type":
        return type_name(t, cu) + "[]"
    nm = t.attributes.get("DW_AT_name")
    return nm.value.decode() if nm else tag


def size_of(die):
    if "DW_AT_byte_size" in die.attributes:
        return die.attributes["DW_AT_byte_size"].value
    if "DW_AT_type" in die.attributes:
        return size_of(die.get_DIE_from_attribute("DW_AT_type"))
    return 8


def real(die):
    while die.tag in ("DW_TAG_typedef", "DW_TAG_const_type", "DW_TAG_volatile_type") and "DW_AT_type" in die.attributes:
        die = die.get_DIE_from_attribute("DW_AT_type")
    return die


def kind_of(tdie):
    """One letter for an argument or return type: p pointer, i/u 32 bit signed/unsigned, c/b 8 or 16 bit
    (read as i/u of that size: h/H), q/Q 64 bit, f float, d double, v void, S struct by value."""
    if tdie is None:
        return "v"
    t = real(tdie)
    if t.tag == "DW_TAG_pointer_type":
        return "p"
    if t.tag == "DW_TAG_enumeration_type":
        return "i"
    if t.tag in ("DW_TAG_structure_type", "DW_TAG_union_type"):
        return "S"
    if t.tag == "DW_TAG_base_type":
        enc, size = t.attributes["DW_AT_encoding"].value, t.attributes["DW_AT_byte_size"].value
        if enc == 4:
            return "f" if size == 4 else "d"
        signed = enc in (5, 6, 13)
        return {1: "c" if signed else "b", 2: "h" if signed else "H", 4: "i" if signed else "u", 8: "q" if signed else "Q"}.get(size, "q")
    return "p"


def func_sig(pdie):
    """The signature of a pointer-to-function member, or None."""
    t = real(pdie.get_DIE_from_attribute("DW_AT_type")) if "DW_AT_type" in pdie.attributes else None
    if t is None or t.tag != "DW_TAG_pointer_type" or "DW_AT_type" not in t.attributes:
        return None
    sub = real(t.get_DIE_from_attribute("DW_AT_type"))
    if sub.tag != "DW_TAG_subroutine_type":
        return None
    args = []
    variadic = False
    for c in sub.iter_children():
        if c.tag == "DW_TAG_formal_parameter":
            args.append(kind_of(c.get_DIE_from_attribute("DW_AT_type")))
        elif c.tag == "DW_TAG_unspecified_parameters":
            variadic = True
    ret = kind_of(sub.get_DIE_from_attribute("DW_AT_type")) if "DW_AT_type" in sub.attributes else "v"
    return {"args": args, "ret": ret, "variadic": variadic}


def members(sdie, base=0, prefix=""):
    res = []
    for ch in sdie.iter_children():
        if ch.tag != "DW_TAG_member":
            continue
        off = base + (ch.attributes["DW_AT_data_member_location"].value if "DW_AT_data_member_location" in ch.attributes else 0)
        nm = ch.attributes.get("DW_AT_name")
        nm = nm.value.decode() if nm else None
        tdie = ch.get_DIE_from_attribute("DW_AT_type")
        rt = real(tdie)
        if nm is None and rt.tag in ("DW_TAG_structure_type", "DW_TAG_union_type"):
            res += members(rt, off, prefix)
            continue
        entry = {"name": prefix + nm, "offset": off, "size": size_of(ch), "type": type_name(ch, None)}
        fs = func_sig(ch)
        if fs:
            entry["fn"] = fs
        res.append(entry)
        if rt.tag in ("DW_TAG_structure_type", "DW_TAG_union_type") and nm:
            res += members(rt, off, prefix + nm + ".")
    return res


layouts = {}
enums = {}
with open(probe, "rb") as f:
    elf = ELFFile(f)
    dwarf = elf.get_dwarf_info()
    for cu in dwarf.iter_CUs():
        for die in cu.iter_DIEs():
            if die.tag == "DW_TAG_enumeration_type":
                for e in die.iter_children():
                    if e.tag == "DW_TAG_enumerator":
                        v = e.attributes["DW_AT_const_value"].value
                        enums[e.attributes["DW_AT_name"].value.decode()] = v if v < (1 << 63) else v - (1 << 64)
            if die.tag == "DW_TAG_typedef" and "DW_AT_name" in die.attributes:
                name = die.attributes["DW_AT_name"].value.decode()
                if "DW_AT_type" not in die.attributes:
                    continue
                target = real(die.get_DIE_from_attribute("DW_AT_type"))
                if target.tag in ("DW_TAG_structure_type", "DW_TAG_union_type") and "DW_AT_byte_size" in target.attributes:
                    layouts[name] = {
                        "size": target.attributes["DW_AT_byte_size"].value,
                        "kind": "union" if target.tag == "DW_TAG_union_type" else "struct",
                        "members": members(target),
                    }
            elif die.tag == "DW_TAG_structure_type" and "DW_AT_name" in die.attributes and "DW_AT_byte_size" in die.attributes:
                name = die.attributes["DW_AT_name"].value.decode()
                layouts.setdefault(name, {"size": die.attributes["DW_AT_byte_size"].value, "kind": "struct", "members": members(die)})

# suite names: "#define kXSuite "name"", then the struct that follows with a version digit
suites = []
for path in sorted(headers.glob("*.h")):
    text = path.read_text(encoding="latin-1")
    pending = {}
    for m in re.finditer(r'#define\s+(k\w+Suite\w*)\s+"([^"]+)"|#define\s+(k\w+Version\w*)\s+(\d+)|typedef\s+struct\s+(\w+)?\s*\{|\}\s*(\w+Suite\w*)\s*;', text):
        if m.group(1):
            pending["name"] = m.group(2)
            pending["macro"] = m.group(1)
        elif m.group(3):
            pending.setdefault("versions", {})[m.group(3)] = int(m.group(4))
        elif m.group(6):
            suites.append({"struct": m.group(6), "name": pending.get("name"), "macro": pending.get("macro"),
                           "versions": pending.get("versions", {}), "file": path.name})
print(f"{len(layouts)} layouts, {len(suites)} suite structs, {len(enums)} enumerators", file=sys.stderr)
Path(out).write_text(json.dumps({"layouts": layouts, "suites": suites, "enums": enums}, indent=1))
