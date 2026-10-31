"""shlwapi PathAppendW: BOOL PathAppendW(LPWSTR pszPath, LPCWSTR pszMore).

Joins two path parts with one backslash, as shlwapi does (a leading backslash of pszMore is dropped, an existing
trailing backslash of pszPath is reused, an empty pszMore leaves pszPath alone, the result must fit MAX_PATH (260
characters) or the call fails and pszPath is unchanged). Before this it was a generic fallback that returned 0.
Strings are read to compute the result and are never logged.
"""
MAX_PATH = 260


def append(path: str, more: str):
    """The joined path, or None when it would not fit in MAX_PATH."""
    if not more:
        return path
    more = more.lstrip("\\")
    if path and not path.endswith("\\"):
        path += "\\"
    out = path + more
    return out if len(out) < MAX_PATH else None


def install(stubs, arg, ret, fault_type=None):
    def read_w(ql, addr):
        out = bytearray()
        while len(out) < 2 * MAX_PATH * 2:
            c = bytes(ql.mem.read(addr + len(out), 2))
            if c == bytes(2):
                break
            out += c
        return out.decode("utf-16le", "replace")

    def path_append_w(emu, ql):
        dst, src = arg(ql, 0), arg(ql, 1)
        if not dst or not src:
            return ret(ql, 0)
        joined = append(read_w(ql, dst), read_w(ql, src))
        if joined is None:
            emu.rec.log("path_append", ok=False)
            return ret(ql, 0)
        ql.mem.write(dst, joined.encode("utf-16le") + bytes(2))
        emu.rec.log("path_append", ok=True, chars=len(joined))
        ret(ql, 1)

    stubs["PathAppendW"] = path_append_w
