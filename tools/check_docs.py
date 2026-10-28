#!/usr/bin/env python3
"""check_docs.py: check the documentation of the twixtor repository.

  python -I tools/check_docs.py              run every check, print counts and the first findings, exit 1 on any
  python -I tools/check_docs.py --list NAME  print every finding of one check
  python -I tools/check_docs.py --json       print every finding as JSON

Checks:
  regen   the generated figures and tables equal a fresh `gen_docs.py` generation, no unknown key
  links   relative links and #anchors in README.md and docs/*.md resolve (GitHub heading anchors)
  paths   inline code that looks like a repository path (crates/, plugins/, tools/, docs/, contracts/,
          .github/, .forgejo/ or a root file) exists; a `:LINE` suffix is ignored, and so are spans with a
          placeholder (* < > { } $ |, ..., NNN) or a space, the paths of OPTIONAL_PATHS, and every line
          that names sapphire (the shared handoff pages cite the peer repository's paths)
  ascii   README.md, docs/, tools/, contracts/ and the workflows hold ASCII only, and no control character
          but tab, LF and CR
  dashes  no U+2010 to U+2015 or U+2212 in any text file of the repository
  flags   every flag in the first column of a README or docs table headed Flag, and in a column headed
          Key flags, appears in the --help of vfx, fx-parity, tools/apply.py, tools/speed_curve.py,
          tools/repeat_keys.py, tools/gen_docs.py or tools/parity_lock.py (a table preceded by
          `<!--help:NAME-->` is checked against that one help only). It needs a built vfx: --vfx PATH or
          VFX_PATH; fx-parity is taken from beside it. Without one the check is skipped with a note.

Exit codes: 0 no findings; 1 findings; 2 usage. Python 3 standard library only.
"""

import argparse
import json
import os
import re
import subprocess
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)

from docgen.data import ROOT, SKIP_DIRS  # noqa: E402
from docgen.render import md_files  # noqa: E402

CHECKS = ["regen", "links", "paths", "ascii", "dashes", "flags"]
TEXT_EXT = (".md", ".py", ".rs", ".toml", ".json", ".tsv", ".yml", ".yaml", ".txt", ".lock")
ROOT_FILES = {"README.md", "Cargo.toml", "Cargo.lock", "LICENSE", ".gitignore", ".gitattributes"}
PATH_HEAD = re.compile(r"^(crates|plugins|tools|docs|contracts|\.github|\.forgejo)/")
# Paths a run creates (not in a fresh checkout); naming them is not a finding.
OPTIONAL_PATHS = set()
DASHES = tuple(chr(c) for c in range(0x2010, 0x2016)) + (chr(0x2212),)
LINK = re.compile(r"!?\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
SPAN = re.compile(r"``\s?(.+?)\s?``|`([^`\n]+)`")
COMMENT = re.compile(r"<!--.*?-->")
HELP_MARK = re.compile(r"<!--help:([A-Za-z0-9_.-]+)-->")
PY_TOOLS = ["apply.py", "speed_curve.py", "repeat_keys.py", "gen_docs.py", "parity_lock.py"]


class Ctx:
    def __init__(self, root, vfx=None):
        self.root = root
        self.vfx = vfx
        self._text = {}
        self.md = md_files(root)
        self.skipped = {}

    def text(self, rel):
        if rel not in self._text:
            with open(os.path.join(self.root, rel), "rb") as h:
                self._text[rel] = h.read().decode("utf-8", errors="replace")
        return self._text[rel]

    def text_files(self):
        out = []
        for dirpath, dirs, files in os.walk(self.root):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for f in sorted(files):
                rel = os.path.relpath(os.path.join(dirpath, f), self.root).replace(os.sep, "/")
                if f.endswith(TEXT_EXT) or rel in ROOT_FILES:
                    out.append(rel)
        return out


def lines_outside_fences(text):
    """(line number, line) of every line outside fenced code blocks."""
    fence = None
    for i, line in enumerate(text.split("\n"), 1):
        s = line.strip()
        m = re.match(r"(```|~~~)", s)
        if m:
            if fence is None:
                fence = m.group(1)
            elif s.startswith(fence):
                fence = None
            continue
        if fence is None:
            yield i, line


def slug(heading):
    """The anchor GitHub gives a heading: lower case, punctuation dropped, spaces to hyphens."""
    h = COMMENT.sub("", heading)
    h = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", h).strip().lower()
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")


def anchors(text):
    out, seen = set(), {}
    for _, line in lines_outside_fences(text):
        m = re.match(r"#{1,6}\s+(.*?)\s*#*\s*$", line)
        if m:
            s = slug(m.group(1))
            n = seen.get(s, 0)
            out.add(s if n == 0 else "{}-{}".format(s, n))
            seen[s] = n + 1
        for a in re.finditer(r"<a\s+(?:id|name)=\"([^\"]+)\"", line):
            out.add(a.group(1))
    return out


def tables(text):
    """(line of the header, the line before the table, header cells, [(line, cells)]) of every table."""
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1]):
            head = [c.strip().strip("`").lower() for c in re.split(r"(?<!\\)\|", lines[i].strip().strip("|"))]
            prev = lines[i - 1].strip() if i > 0 else ""
            if not prev and i > 1:
                prev = lines[i - 2].strip()
            rows, j = [], i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append((j + 1, [c.strip() for c in re.split(r"(?<!\\)\|", lines[j].strip().strip("|"))]))
                j += 1
            yield i + 1, prev, head, rows
            i = j
        else:
            i += 1


# ---- checks ---------------------------------------------------------------------------------------

def check_regen(ctx):
    import gen_docs
    from docgen.data import DataError
    try:
        _, changed, unknown = gen_docs.drift(ctx.root)
    except (DataError, OSError) as e:
        return [("regen", "docs/data", 0, "cannot generate: {}".format(e))]
    out = [("regen", p, 0, "differs from a fresh generation (run python -I tools/gen_docs.py)") for p in changed]
    out += [("regen", u.split(": ")[0], 0, "unknown marker key " + u.split(": ", 1)[1]) for u in unknown]
    return out


def check_links(ctx):
    out, cache = [], {}
    for rel in ctx.md:
        base = os.path.dirname(rel)
        for i, line in lines_outside_fences(ctx.text(rel)):
            for m in LINK.finditer(SPAN.sub(" ", line)):
                target = m.group(2)
                if re.match(r"(?i)[a-z][a-z0-9+.-]*:", target):
                    continue
                path, _, anchor = target.partition("#")
                full = rel if not path else os.path.normpath(os.path.join(base, path)).replace(os.sep, "/")
                if not os.path.exists(os.path.join(ctx.root, full)):
                    out.append(("links", rel, i, "missing target " + target))
                    continue
                if anchor and full.endswith(".md"):
                    if full not in cache:
                        cache[full] = anchors(ctx.text(full))
                    if anchor not in cache[full]:
                        out.append(("links", rel, i, "missing anchor " + target))
    return out


def code_paths(line):
    """The repository paths named in the inline code of one line."""
    out = []
    for m in SPAN.finditer(line):
        s = (m.group(1) or m.group(2)).strip().replace("\\", "/")
        s = re.sub(r":\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*$", "", s.split("#")[0]).rstrip(":")
        if " " in s or not (PATH_HEAD.match(s) or s in ROOT_FILES):
            continue
        if any(c in s for c in "*<>{}$|") or "..." in s or "NNN" in s or s in OPTIONAL_PATHS:
            continue
        out.append(s)
    return out


def check_paths(ctx):
    out, seen = [], {}
    for rel in ctx.md:
        for i, line in lines_outside_fences(ctx.text(rel)):
            if "sapphire" in line.lower():
                continue
            for s in code_paths(line):
                if s not in seen:
                    seen[s] = os.path.exists(os.path.join(ctx.root, s.rstrip("/")))
                if not seen[s]:
                    out.append(("paths", rel, i, "missing path " + s))
    return out


def check_ascii(ctx):
    out = []
    heads = ("tools/", "docs/", "contracts/", ".github/", ".forgejo/")
    files = sorted(set(ctx.md + [p for p in ctx.text_files() if p.startswith(heads)]))
    for rel in files:
        for i, line in enumerate(ctx.text(rel).split("\n"), 1):
            if any(ord(c) > 127 for c in line):
                out.append(("ascii", rel, i, "non-ASCII character"))
            ctrl = sorted({ord(c) for c in line if (ord(c) < 32 and ord(c) not in (9, 13)) or ord(c) == 127})
            if ctrl:
                out.append(("ascii", rel, i, "control character " + ", ".join("0x%02x" % o for o in ctrl)))
    return out


def check_dashes(ctx):
    out = []
    for rel in ctx.text_files():
        text = ctx.text(rel)
        if not any(d in text for d in DASHES):
            continue
        for i, line in enumerate(text.split("\n"), 1):
            hit = sorted({"U+%04X" % ord(c) for c in line if c in DASHES})
            if hit:
                out.append(("dashes", rel, i, "dash " + ", ".join(hit)))
    return out


def help_texts(root, vfx):
    """The --help output of each tool, keyed by name; an empty text when it cannot run."""
    out = {}

    def run(key, cmd):
        try:
            p = subprocess.run(cmd, cwd=root, capture_output=True, timeout=60)
            out[key] = (p.stdout + p.stderr).decode("utf-8", "replace")
        except (OSError, subprocess.TimeoutExpired):
            out[key] = ""

    import gen_docs
    run("vfx", gen_docs.program(vfx) + ["--help"])
    run("fx-parity", gen_docs.program(gen_docs.sibling(vfx, "fx-parity")) + ["--help"])
    for tool in PY_TOOLS:
        path = os.path.join(root, "tools", tool)
        if os.path.exists(path):
            key = tool[:-3]
            run(key, [sys.executable, "-I", path, "--help"])
            m = re.search(r"^usage: \S+ (?:\[-h\] )?\{([a-z0-9_,-]+)\}", out[key], re.M)
            for sub in (m.group(1).split(",") if m else []):
                run(key + " " + sub, [sys.executable, "-I", path, sub, "--help"])
    return out


def flag_cells(head, cells):
    """The cells of one row that name flags: the first column of a Flag table, any Key flags column."""
    out = []
    if head and head[0] in ("flag", "flags", "option", "options"):
        out.append(cells[0] if cells else "")
    for n, h in enumerate(head):
        if h.startswith("key flags") and n < len(cells):
            out.append(cells[n])
    return out


def check_flags(ctx):
    if not ctx.vfx:
        ctx.skipped["flags"] = "no vfx: pass --vfx PATH or set VFX_PATH to check the flag tables against --help"
        return []
    import gen_docs
    vfx = gen_docs.program(ctx.vfx)[-1]
    if not os.path.isfile(vfx):
        return [("flags", "VFX_PATH", 0, "vfx not found at " + ctx.vfx)]
    helps = help_texts(ctx.root, vfx)
    out = [("flags", "--help", 0, "no help output from " + k) for k in ("vfx", "fx-parity") if not helps.get(k)]
    union = "\n".join(helps.values())
    for rel in ctx.md:
        for ln, prev, head, rows in tables(ctx.text(rel)):
            if not any(flag_cells(head, ["x"] * len(head))):
                continue
            text = union
            m = HELP_MARK.search(prev)
            if m:
                text = helps.get(m.group(1), "")
                if not text:
                    out.append(("flags", rel, ln, "no help output for " + m.group(1)))
                    continue
            for row_ln, cells in rows:
                for cell in flag_cells(head, cells):
                    for flag in re.findall(r"(?<![\w-])--[a-z0-9][a-z0-9-]*", SPAN.sub(lambda s: " " + (
                            s.group(1) or s.group(2)) + " ", cell)):
                        if not re.search(re.escape(flag) + r"(?![a-z0-9-])", text):
                            out.append(("flags", rel, row_ln, "flag {} not in --help".format(flag)))
    return out


def run(root=ROOT, names=None, vfx=None):
    """{check: [(check, path, line, message)]} and the {check: reason} of the skipped ones."""
    ctx = Ctx(root, vfx)
    results = {}
    for name in names or CHECKS:
        results[name] = globals()["check_" + name](ctx)
    return results, ctx.skipped


def main(argv=None):
    ap = argparse.ArgumentParser(prog="check_docs.py", description="Check the twixtor documentation.")
    ap.add_argument("--list", metavar="NAME", choices=CHECKS, help="print every finding of one check")
    ap.add_argument("--json", action="store_true", help="print every finding as JSON")
    ap.add_argument("--only", metavar="NAME", action="append", choices=CHECKS, help="run only these checks")
    ap.add_argument("--vfx", default=os.environ.get("VFX_PATH") or None, help="vfx for the flags check (default VFX_PATH)")
    ap.add_argument("--root", default=ROOT, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    results, skipped = run(args.root, args.only, args.vfx)
    total = sum(len(v) for v in results.values())
    if args.json:
        print(json.dumps({"findings": {k: [list(f[1:]) for f in v] for k, v in results.items()},
                          "skipped": skipped}, indent=1))
        return 1 if total else 0
    print("check_docs: {} findings; ".format(total) + ", ".join("{} {}".format(k, len(v)) for k, v in results.items()))
    for name, why in skipped.items():
        print("  {} skipped: {}".format(name, why))
    for name, found in results.items():
        shown = found if args.list == name else (found[:5] if not args.list else [])
        for _, path, line, msg in shown:
            print("  {} {}:{} {}".format(name, path, line, msg))
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
