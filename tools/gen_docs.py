#!/usr/bin/env python3
"""gen_docs.py: fill the generated figures and tables of the documentation from docs/data/ and the tree.

  python -I tools/gen_docs.py                 fill every marker of README.md and docs/*.md, write docs/data/facts.json
  python -I tools/gen_docs.py --check         regenerate in memory; exit 1 on drift or an unknown key
  python -I tools/gen_docs.py --refresh-data  rebuild the snapshots from a built vfx, then fill as above

A figure is an HTML comment pair around its text, `<!--f:` KEY `-->` text `<!--/f-->`, and a table sits
between `<!--gen:` KEY `-->` and `<!--/gen:` KEY `-->` (spaced out here so this text is no marker).
Every key and its source is listed in docs/data/facts.json; the snapshots and scans are described in
tools/docgen/data.py.

--check is hermetic: it reads the committed snapshots and scans files, and runs no process.

--refresh-data runs `vfx list --json --all` (vfx from --vfx, else VFX_PATH) and rewrites
docs/data/controls.tsv and docs/data/gaps.tsv. With --testset DIR (a set of `fx-parity gen --families`)
it also rewrites docs/data/references.tsv from DIR/cases.json, every case `missing`; with --refs DIR as
well it grades them with `fx-parity run --testset T --refs R --json` (fx-parity from --parity, else next
to vfx). docs/data/lock.json is written by `tools/parity_lock.py record`, not here.

Exit codes: 0 done (or no drift); 1 drift or an unknown key; 2 usage; 3 a snapshot that cannot be read,
or a tool that cannot be run. Python 3 standard library only.
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from docgen import data as dg  # noqa: E402
from docgen.render import build  # noqa: E402

EXE = ".exe" if os.name == "nt" else ""


def disk(root, rel):
    path = os.path.join(root, rel)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as h:
        return h.read().decode("utf-8", errors="replace")


def drift(root=dg.ROOT):
    """(outputs, changed paths, unknown keys) for the tree at root."""
    outputs, unknown, _ = build(root)
    changed = [rel for rel, text in outputs.items() if disk(root, rel) != text]
    return outputs, changed, unknown


def write(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as h:
        h.write(text.encode("utf-8"))


def program(path):
    """The argv prefix that runs a tool: a .py file runs under this Python (for tests). A path with a
    folder in it is made absolute, and gets .exe on Windows when it lacks it."""
    path = str(path)
    if os.sep in path or "/" in path:
        path = os.path.abspath(path)
        if EXE and not os.path.exists(path) and os.path.exists(path + EXE):
            path += EXE
    return [sys.executable, path] if path.endswith(".py") else [path]


def run_json(cmd):
    cmd = program(cmd[0]) + list(cmd[1:])
    try:
        p = subprocess.run(cmd, capture_output=True)
    except OSError as e:
        raise dg.DataError("cannot run {}: {}".format(cmd[0], e)) from None
    if p.returncode not in (0, 1) or not p.stdout.strip():
        raise dg.DataError("{} exited with code {}: {}".format(
            " ".join(cmd[:2]), p.returncode, p.stderr.decode("utf-8", "replace").strip()[:300]))
    try:
        return json.loads(p.stdout.decode("utf-8"))
    except ValueError as e:
        raise dg.DataError("{} printed no JSON: {}".format(" ".join(cmd[:2]), e)) from None


def sibling(vfx, name):
    """The tool `name` beside vfx, with vfx's extension (.exe, .py or none)."""
    ext = os.path.splitext(vfx)[1] if os.path.splitext(vfx)[1] in (".exe", ".py") else EXE
    return os.path.join(os.path.dirname(os.path.abspath(vfx)), name + ext)


def refresh(root, vfx, parity=None, testset=None, refs=None):
    """Rewrites the snapshots; returns the paths it wrote."""
    if not vfx:
        raise dg.DataError("--refresh-data needs a built vfx: pass --vfx PATH or set VFX_PATH")
    if not os.path.isfile(program(vfx)[-1]):
        raise dg.DataError("vfx not found at {}".format(vfx))
    listing = run_json([vfx, "list", "--json", "--all"])
    wrote = OrderedPaths()
    wrote.put(root, dg.DATA + "/controls.tsv", dg.tsv_text(dg.CONTROLS_HEADER, dg.controls_from_listing(listing)))
    wrote.put(root, dg.DATA + "/gaps.tsv", dg.tsv_text(dg.GAPS_HEADER, dg.gaps_from_listing(listing)))
    if testset:
        cases_path = os.path.join(testset, "cases.json")
        if not os.path.isfile(cases_path):
            raise dg.DataError("no cases.json in {}".format(testset))
        cases = json.loads(dg.read_text(cases_path))
        statuses = None
        if refs:
            parity = parity or sibling(vfx, "fx-parity")
            if not os.path.isfile(parity):
                raise dg.DataError("fx-parity not found at {}: pass --parity PATH".format(parity))
            report = run_json([parity, "run", "--testset", testset, "--refs", refs, "--json", "--vfx", vfx])
            statuses = {c["id"]: c["status"] for c in report.get("cases", [])}
        rows = dg.references_from_cases(cases, statuses, dg.family_gaps(root))
        wrote.put(root, dg.DATA + "/references.tsv", dg.tsv_text(dg.REFERENCES_HEADER, rows))
    elif refs:
        raise dg.DataError("--refs needs --testset")
    return wrote.paths


class OrderedPaths:
    def __init__(self):
        self.paths = []

    def put(self, root, rel, text):
        if disk(root, rel) != text:
            write(root, rel, text)
            self.paths.append(rel)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gen_docs.py",
                                 description="Fill the generated figures and tables of the twixtor documentation.")
    ap.add_argument("--check", action="store_true", help="regenerate in memory and exit 1 on drift")
    ap.add_argument("--refresh-data", action="store_true", help="rebuild the snapshots of docs/data/ from a built vfx")
    ap.add_argument("--vfx", default=os.environ.get("VFX_PATH") or None, help="vfx for --refresh-data (default VFX_PATH)")
    ap.add_argument("--parity", help="fx-parity for --refs (default: next to vfx)")
    ap.add_argument("--testset", metavar="DIR", help="a reference family set (fx-parity gen --families) for references.tsv")
    ap.add_argument("--refs", metavar="DIR", help="reference renders of --testset, graded by fx-parity run")
    ap.add_argument("--root", default=dg.ROOT, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.check and args.refresh_data:
        ap.error("--check and --refresh-data exclude each other")
    root = args.root
    try:
        if args.refresh_data:
            for rel in refresh(root, args.vfx, args.parity, args.testset, args.refs):
                print("gen_docs: {} rebuilt".format(rel))
        outputs, changed, unknown = drift(root)
    except (dg.DataError, OSError) as e:
        print("gen_docs: error: {}".format(e), file=sys.stderr)
        return 3
    if args.check:
        print("gen_docs --check: {} outputs, {} facts, {} drifted, {} unknown keys".format(
            len(outputs), len(_facts(outputs)), len(changed), len(unknown)))
        for item in (changed + unknown)[:20]:
            print("  " + item)
        return 1 if (changed or unknown) else 0
    for rel in changed:
        write(root, rel, outputs[rel])
    print("gen_docs: {} outputs, {} written, {} unknown keys".format(len(outputs), len(changed), len(unknown)))
    for item in unknown[:20]:
        print("  " + item)
    return 1 if unknown else 0


def _facts(outputs):
    return json.loads(outputs[dg.DATA + "/facts.json"])["facts"]


if __name__ == "__main__":
    sys.exit(main())
