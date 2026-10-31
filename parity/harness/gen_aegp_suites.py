#!/usr/bin/env python3
"""Adds suite revisions that the compiled layouts of gen_layouts.py do not cover (the AEGP suites, and old
revisions kept in AE_GeneralPlugOld.h) to sdk_layouts_extra.json: member names, offsets (8 byte function
pointers in declaration order) and argument kinds, read from the header text.

usage: gen_aegp_suites.py HEADERS_DIR "AEGP Layer Suite" NUMERIC_VERSION MACRO_PREFIX STRUCT_PREFIX OUT.json
  e.g. gen_aegp_suites.py sdk/Headers "AEGP Layer Suite" 5 kAEGPLayerSuite AEGP_LayerSuite sdk_layouts_extra.json
The header defines `#define <MACRO_PREFIX>Version<N> <NUMERIC_VERSION>` and `typedef struct <STRUCT_PREFIX><N> {...}`.
Argument kinds: p pointer (also the SDK's pointer typedefs and AEGP_*H handles), d A_FpLong, f A_FpShort,
u unsigned, i other. Return kind is always i (A_Err).
"""
import json
import re
import sys
from pathlib import Path

POINTER_TYPES = ("AEGP_GlobalRefcon", "AEGP_MemHandle", "AEGP_RefcountedRef", "PF_ProgPtr", "PF_ContextH")
HANDLE = re.compile(r"^(const\s+)?AEGP_\w+H\b")           # AEGP_LayerH, AEGP_ItemH, AEGP_EffectRefH ... are pointers


def kinds_of(args):
    args = " ".join(args.split())
    params = [] if args in ("", "void") else [x.strip() for x in args.split(",")]
    kinds = []
    for x in params:
        if "*" in x or "[" in x or HANDLE.match(x) or any(x.startswith(p) for p in POINTER_TYPES):
            kinds.append("p")
        elif "FpLong" in x:
            kinds.append("d")
        elif "FpShort" in x:
            kinds.append("f")
        elif "u_long" in x:
            kinds.append("u")
        else:
            kinds.append("i")
    return kinds


def main():
    headers, name, version, macro_prefix, struct_prefix, out = sys.argv[1:7]
    texts = {p.name: p.read_text(encoding="latin-1") for p in sorted(Path(headers).glob("*.h"))}
    found = None
    for fname, t in texts.items():
        for m in re.finditer(r"#define\s+%sVersion(\d+)\s+(\d+)" % re.escape(macro_prefix), t):
            if int(m.group(2)) == int(version):
                found = (int(m.group(1)), fname)
    if not found:
        sys.exit(f"no {macro_prefix}Version<N> with value {version}")
    n, _ = found
    struct = f"{struct_prefix}{n}"
    body = None
    for fname, t in texts.items():
        t = re.sub(r"/\*.*?\*/", "", t, flags=re.S)
        t = re.sub(r"//[^\n]*", "", t)
        m = re.search(r"typedef\s+struct\s+%s\s*\{(.*?)\}\s*%s\s*;" % (struct, struct), t, flags=re.S)
        if m:
            body, src = m.group(1), fname
            break
    if body is None:
        sys.exit(f"no struct {struct}")
    members = []
    for i, (fn, args) in enumerate(re.findall(r"SPAPI\s+A_Err\s*\(\s*\*\s*(\w+)\s*\)\s*\((.*?)\)\s*;", body, flags=re.S)):
        members.append({"name": fn, "offset": 8 * i, "size": 8, "type": "DW_TAG_subroutine_type*",
                        "fn": {"args": kinds_of(args), "ret": "i", "variadic": False}})
    p = Path(out)
    data = json.loads(p.read_text()) if p.exists() else {"layouts": {}, "suites": []}
    data["layouts"][struct] = {"size": 8 * len(members), "kind": "struct", "members": members}
    data["suites"] = [s for s in data["suites"] if s["struct"] != struct] + [
        {"struct": struct, "name": name, "macro": macro_prefix, "versions": {f"{macro_prefix}Version{n}": int(version)}, "file": src}]
    p.write_text(json.dumps(data, indent=1))
    print(struct, len(members), "members from", src)


if __name__ == "__main__":
    main()
