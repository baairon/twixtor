#!/usr/bin/env python3
"""parity_lock.py: the regression lock. Every setting that rendered on a baseline build must render the
same bytes, with the same (empty) stderr, on a new build.

  python tools/parity_lock.py base --ref REF [--out DIR]     build vfx and fx-parity of REF
  python tools/parity_lock.py corpus [--out DIR]             generate the corpora with the baseline tools
  python tools/parity_lock.py refs [--out DIR]               run the baseline vfx on every case, store hashes
  python tools/parity_lock.py check --vfx NEW [--out DIR]    P2: every case the baseline rendered renders the same
  python tools/parity_lock.py refused --vfx NEW [--out DIR]  P3: the cases the baseline refused, on the new build
  python tools/parity_lock.py record [--out DIR]             write docs/data/lock.json from the last check

DIR defaults to target/parity-lock in this repository; nothing in it is ever committed.

base: makes a temporary detached git worktree of REF in DIR, builds `vfx` and `fx-parity` there with the
CI profile (CARGO_INCREMENTAL=0, CARGO_PROFILE_RELEASE_LTO=false, CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16,
--release, target folder DIR/target), copies both into DIR/base and removes the worktree again. It
writes DIR/base.json (the ref and its commit). corpus and refs use DIR/base unless --parity and --vfx
name other tools.

Corpora (the case files of `fx-parity gen`, with every fgN_matte layer removed from every case, because
the baseline refused them; the original case file is kept as cases.orig.json):
  L1  gen --effects of the four effects and selftest-invert, -shift, -integrate and -threads, defaults
  L2  gen of the four effects at --sizes 2x2,3x2,4x4,5x4,8x8,9x8,16x16,17x16, --frames 6, every case
  L3  the temporal L2 cases of retime, retime-fgbg and create-vectors, run with --quality draft

Each case runs exactly as fx-parity runs it (the arguments of crates/fx-parity/src/vfx.rs), with
RAYON_NUM_THREADS removed. A case the baseline renders (exit 0) goes to refs.json with the SHA-256 and
size of its stdout and the SHA-256 of its stderr; any other exit goes to refused.json. check passes a
case only when the new build exits 0 with the same stdout hash and size and the same stderr hash (so an
empty stderr stays empty: a new note on a setting that rendered before is a difference). refused runs
the refused cases on the new build and counts those that now exit 0.

Exit codes: 0 pass; 1 a lock failure (check: a difference; refused: a case that still fails); 2 usage;
3 setup (a missing corpus, a failed build or generation). A `vfx`, `fx-parity` or `cargo` path that
ends in .py runs under this Python (for tests). Python 3 standard library only.
"""

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_OUT = ROOT / "target" / "parity-lock"
EXE = ".exe" if os.name == "nt" else ""

REAL_EFFECTS = ["retime", "create-vectors", "retime-fgbg", "vectors-in"]
L1_EFFECTS = REAL_EFFECTS + ["selftest-invert", "selftest-shift", "selftest-integrate", "selftest-threads"]
L2_SIZES = ["2x2", "3x2", "4x4", "5x4", "8x8", "9x8", "16x16", "17x16"]
DRAFT_EFFECTS = ["retime", "retime-fgbg", "create-vectors"]
SETS = ["L1", "L2", "L3"]
CI_PROFILE = {
    "CARGO_INCREMENTAL": "0",
    "CARGO_PROFILE_RELEASE_LTO": "false",
    "CARGO_PROFILE_RELEASE_CODEGEN_UNITS": "16",
}


class Setup(Exception):
    """A setup or I/O failure: exit 3."""


def corpus_definition():
    """The corpora, as the `fx-parity gen` arguments and the rules applied to them."""
    return {
        "L1": {"gen": ["--effects", ",".join(L1_EFFECTS)], "strip": "fgN_matte layers", "quality": "high"},
        "L2": {"gen": ["--effects", ",".join(REAL_EFFECTS), "--sizes", ",".join(L2_SIZES), "--frames", "6",
                       "--max-cases-per-effect", "0"], "strip": "fgN_matte layers", "quality": "high"},
        "L3": {"from": "L2", "effects": DRAFT_EFFECTS, "kind": "temporal", "quality": "draft"},
    }


def program(path):
    """The argv prefix that runs a tool: a .py file runs under this Python. A path with a folder in it
    is made absolute (Windows does not run a relative one), and gets .exe on Windows when it lacks it."""
    path = str(path)
    if os.sep in path or "/" in path:
        path = os.path.abspath(path)
        if EXE and not os.path.exists(path) and os.path.exists(path + EXE):
            path += EXE
    return [sys.executable, path] if path.endswith(".py") else [path]


def run_tool(cmd, **kw):
    """Runs a setup command (no shell) and raises Setup when it fails."""
    words = [str(c) for c in cmd]
    print("+ " + " ".join(words), flush=True)
    try:
        p = subprocess.run(words, **kw)
    except OSError as e:
        raise Setup("cannot run {}: {}".format(words[0], e)) from None
    if p.returncode != 0:
        raise Setup("{} exited with code {}".format(" ".join(words[:3]), p.returncode))
    return p


def load_cases(dirp):
    path = pathlib.Path(dirp) / "cases.json"
    if not path.is_file():
        raise Setup("no corpus at {}: run `parity_lock.py corpus` first".format(path))
    return json.loads(path.read_text(encoding="utf-8"))


def read_json(path, what):
    path = pathlib.Path(path)
    if not path.is_file():
        raise Setup("{} missing: {} (run `parity_lock.py {}` first)".format(what, path, what.split()[0]))
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    pathlib.Path(path).write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def strip_mattes(case):
    """Removes the fgN_matte layers of a case (the baseline refused them); True when any was removed."""
    layers = case.get("layers", {})
    kept = {k: v for k, v in layers.items() if not (k.startswith("fg") and k.endswith("_matte"))}
    if kept == layers:
        return False
    if kept:
        case["layers"] = kept
    else:
        case.pop("layers", None)
    return True


def invocation(case, testset, draft=False):
    """(argv words after the program, stdin bytes or None) of one case, as fx-parity builds them."""
    testset = pathlib.Path(testset)
    args = ["apply", case["effect"], "--width", str(case["width"]), "--height", str(case["height"]),
            "--depth", str(case["depth"])]
    for p in case.get("params", []):
        args += ["--param", p]
    if case.get("kind") == "temporal":
        if case.get("fps"):
            args += ["--fps", case["fps"]]
        args += ["--frames", str(case.get("frames", 1))]
        for lid, f in sorted(case.get("layers", {}).items()):
            args += ["--layer", "{}={}".format(lid, testset / f)]
        args += ["--in", str(testset / case["input"])]
        if draft:
            args += ["--quality", "draft"]
        return args, None
    return args, (testset / case["input"]).read_bytes()


def first_error(stderr):
    """The line a failure is reported by: the `vfx: error:` line, else the first that is not a note."""
    lines = [l.strip() for l in stderr.decode("utf-8", "replace").splitlines() if l.strip()]
    for l in lines:
        if l.startswith("vfx: error:"):
            return l
    for l in lines:
        if not l.startswith("vfx: note:"):
            return l
    return lines[0] if lines else ""


def run_case(vfx, case, testset, draft):
    """Runs one case and returns its exit code and the hashes of what it wrote."""
    args, stdin = invocation(case, testset, draft)
    env = dict(os.environ)
    env.pop("RAYON_NUM_THREADS", None)
    p = subprocess.run(program(vfx) + args, input=stdin,
                       stdin=None if stdin is not None else subprocess.DEVNULL, capture_output=True, env=env)
    return {
        "exit": p.returncode,
        "out": hashlib.sha256(p.stdout).hexdigest(),
        "size": len(p.stdout),
        "err": hashlib.sha256(p.stderr).hexdigest() if p.stderr else "",
        "err_line": first_error(p.stderr) if p.stderr else "",
    }


def work_items(out, sets):
    """(key, case, testset folder, draft) for every invocation of the named corpora, in a fixed order."""
    out = pathlib.Path(out)
    items = []
    for name in sets:
        if name in ("L1", "L2"):
            d = out / name
            for c in load_cases(d)["cases"]:
                items.append(("{}:{}".format(name, c["id"]), c, d, False))
        elif name == "L3":
            d = out / "L2"
            for c in load_cases(d)["cases"]:
                if c.get("kind") == "temporal" and c["effect"] in DRAFT_EFFECTS:
                    items.append(("L3:{}".format(c["id"]), c, d, True))
        else:
            raise Setup("unknown corpus {} (L1, L2 or L3)".format(name))
    return items


def run_all(vfx, items, jobs, limit=None):
    if limit:
        items = items[:limit]
    res = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futs = {ex.submit(run_case, vfx, c, d, dr): key for key, c, d, dr in items}
        done = 0
        for f in concurrent.futures.as_completed(futs):
            res[futs[f]] = f.result()
            done += 1
            if done % 1000 == 0:
                print("  {} of {}".format(done, len(items)), flush=True)
    return res


def compare(base, new):
    """The reasons a new result differs from its baseline result; empty when it is identical."""
    why = []
    if new["exit"] != 0:
        why.append("exit {}: {}".format(new["exit"], new["err_line"]))
    if new["out"] != base["out"] or new["size"] != base["size"]:
        why.append("stdout differs ({} -> {} bytes)".format(base["size"], new["size"]))
    if new["err"] != base["err"]:
        why.append("stderr differs: {}".format(new["err_line"] or "(empty)"))
    return why


def set_of(key):
    return key.split(":", 1)[0]


def per_set(keys):
    counts = {s: 0 for s in SETS}
    for k in keys:
        counts[set_of(k)] = counts.get(set_of(k), 0) + 1
    return counts


# ---- subcommands ----------------------------------------------------------------------------------

def git(repo, *words, capture=False):
    cmd = ["git", "-C", str(repo)] + list(words)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as e:
        raise Setup("cannot run git: {}".format(e)) from None
    if p.returncode != 0:
        raise Setup("{} failed: {}".format(" ".join(cmd[3:]), (p.stderr or p.stdout).strip()))
    return p.stdout.strip() if capture else None


def cargo_command(cargo, toolchain):
    cmd = program(cargo)
    if toolchain:
        cmd.append("+" + toolchain)
    return cmd + ["build", "--release", "-p", "fx-cli", "-p", "fx-parity"]


def cmd_base(a):
    out = pathlib.Path(a.out)
    repo = pathlib.Path(a.repo)
    commit = git(repo, "rev-parse", "--verify", a.ref + "^{commit}", capture=True)
    out.mkdir(parents=True, exist_ok=True)
    wt = out / ("worktree-" + commit[:12])
    if wt.exists():
        raise Setup("{} exists: remove it (a previous base run was interrupted)".format(wt))
    git(repo, "worktree", "add", "--detach", str(wt), commit)
    try:
        env = dict(os.environ)
        env.update(CI_PROFILE)
        env["CARGO_TARGET_DIR"] = str((out / "target").resolve())
        run_tool(cargo_command(a.cargo, a.toolchain), cwd=str(wt), env=env)
        dest = out / "base"
        dest.mkdir(exist_ok=True)
        for name in ("vfx", "fx-parity"):
            src = out / "target" / "release" / (name + EXE)
            if not src.is_file():
                raise Setup("the build left no {}".format(src))
            shutil.copy2(src, dest / (name + EXE))
    finally:
        try:
            git(repo, "worktree", "remove", "--force", str(wt))
        except Setup:
            shutil.rmtree(wt, ignore_errors=True)
            git(repo, "worktree", "prune")
    write_json(out / "base.json", {"ref": a.ref, "commit": commit})
    print("base: {} ({}) built into {}".format(a.ref, commit[:12], out / "base"))
    return 0


def baseline_tool(a, name, given):
    if given:
        return given
    path = pathlib.Path(a.out) / "base" / (name + EXE)
    if not path.is_file():
        raise Setup("no baseline {} at {}: run `parity_lock.py base` or pass --{}".format(
            name, path, "vfx" if name == "vfx" else "parity"))
    return str(path)


def cmd_corpus(a):
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    parity = baseline_tool(a, "fx-parity", a.parity)
    vfx = baseline_tool(a, "vfx", a.vfx)
    for name in ("L1", "L2"):
        d = out / name
        if (d / "cases.json").exists() and not a.force:
            print("{}: exists, kept (--force regenerates)".format(name))
        else:
            run_tool(program(parity) + ["gen", "--out", str(d), "--vfx", vfx] + corpus_definition()[name]["gen"])
            data = load_cases(d)
            write_json(d / "cases.orig.json", data)
            stripped = sum(strip_mattes(c) for c in data["cases"])
            write_json(d / "cases.json", data)
            print("{}: {} cases, {} had matte layers removed".format(name, len(data["cases"]), stripped))
    return 0


def cmd_refs(a):
    out = pathlib.Path(a.out)
    vfx = baseline_tool(a, "vfx", a.vfx)
    items = work_items(out, a.sets)
    print("baseline: {} invocations, {} jobs".format(len(items), a.jobs), flush=True)
    res = run_all(vfx, items, a.jobs, a.limit)
    ok = {k: v for k, v in sorted(res.items()) if v["exit"] == 0}
    bad = {k: v for k, v in sorted(res.items()) if v["exit"] != 0}
    write_json(out / "refs.json", ok)
    write_json(out / "refused.json", bad)
    noisy = [k for k, v in ok.items() if v["err"]]
    print("baseline: {} rendered, {} refused or failed, {} rendered with stderr output".format(
        len(ok), len(bad), len(noisy)))
    return 0


def cmd_check(a):
    out = pathlib.Path(a.out)
    base = read_json(out / "refs.json", "refs output")
    items = [it for it in work_items(out, a.sets) if it[0] in base]
    print("check: {} invocations, {} jobs".format(len(items), a.jobs), flush=True)
    res = run_all(a.vfx, items, a.jobs, a.limit)
    diffs = {}
    for key in sorted(res):
        why = compare(base[key], res[key])
        if why:
            diffs[key] = "; ".join(why)
    write_json(out / "check.json", {
        "vfx": str(a.vfx), "sets": a.sets, "complete": not a.limit and a.sets == SETS,
        "compared": per_set(res), "identical": per_set(k for k in res if k not in diffs), "differ": diffs})
    print("P2: {} identical, {} differ, of {}".format(len(res) - len(diffs), len(diffs), len(res)))
    for key in list(diffs)[:25]:
        print("  DIFF {} - {}".format(key, diffs[key]))
    if len(diffs) > 25:
        print("  ... and {} more (check.json)".format(len(diffs) - 25))
    return 1 if diffs else 0


def cmd_refused(a):
    out = pathlib.Path(a.out)
    bad = read_json(out / "refused.json", "refs output")
    items = [it for it in work_items(out, a.sets) if it[0] in bad]
    res = run_all(a.vfx, items, a.jobs, a.limit)
    still = {k: v["err_line"] for k, v in sorted(res.items()) if v["exit"] != 0}
    write_json(out / "refused-now.json", {
        "vfx": str(a.vfx), "sets": a.sets, "complete": not a.limit and a.sets == SETS,
        "refused": per_set(res), "render": per_set(k for k in res if k not in still), "still": still})
    print("P3: {} of {} previously refused cases now render; {} still fail".format(
        len(res) - len(still), len(res), len(still)))
    heads = {}
    for line in still.values():
        heads[line] = heads.get(line, 0) + 1
    for line, n in sorted(heads.items(), key=lambda x: (-x[1], x[0]))[:15]:
        print("  {:5d}  {}".format(n, line))
    return 1 if still else 0


def lock_record(base_info, refs, refused, check, now, date):
    """The lock.json record of one complete check (and, when given, one complete refused run)."""
    if not check.get("complete"):
        raise Setup("the last check did not cover every corpus (it ran with --sets or --limit)")
    if now is not None and not now.get("complete"):
        raise Setup("the last refused run did not cover every corpus (it ran with --sets or --limit)")
    rendered = per_set(refs)
    refused_n = per_set(refused)
    corpora = {}
    for s in SETS:
        row = {
            "rendered_at_baseline": rendered.get(s, 0),
            "identical": check["identical"].get(s, 0),
            "differ": check["compared"].get(s, 0) - check["identical"].get(s, 0),
            "refused_at_baseline": refused_n.get(s, 0),
        }
        if now is not None:
            row["refused_now_render"] = now["render"].get(s, 0)
        corpora[s] = row
    for s, row in corpora.items():
        if check["compared"].get(s, 0) != row["rendered_at_baseline"]:
            raise Setup("the last check compared {} of the {} cases of {} in refs.json".format(
                check["compared"].get(s, 0), row["rendered_at_baseline"], s))
    return {
        "baseline": {"ref": base_info.get("ref", ""), "commit": base_info.get("commit", "")},
        "date": date,
        "corpora": corpora,
        "definition": corpus_definition(),
        "rules": {
            "rendered": "the baseline vfx exits 0 on the case",
            "identical": "the new vfx exits 0 with the same stdout SHA-256 and size and the same stderr SHA-256",
            "threads": "RAYON_NUM_THREADS removed: the default thread count",
        },
    }


def cmd_record(a):
    out = pathlib.Path(a.out)
    if a.ref:
        base_info = {"ref": a.ref, "commit": git(a.repo, "rev-parse", "--verify", a.ref + "^{commit}", capture=True)}
    else:
        base_info = read_json(out / "base.json", "base output")
    refs = read_json(out / "refs.json", "refs output")
    refused = read_json(out / "refused.json", "refs output")
    check = read_json(out / "check.json", "check output")
    now_path = out / "refused-now.json"
    now = json.loads(now_path.read_text(encoding="utf-8")) if now_path.is_file() else None
    date = a.date or datetime.date.today().isoformat()
    rec = lock_record(base_info, refs, refused, check, now, date)
    target = pathlib.Path(a.data)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, rec)
    total = sum(r["rendered_at_baseline"] for r in rec["corpora"].values())
    same = sum(r["identical"] for r in rec["corpora"].values())
    print("record: {} of {} identical against {}, written to {}".format(
        same, total, rec["baseline"]["commit"][:7] or rec["baseline"]["ref"], target))
    return 0


def jobs_default():
    return max(1, min(8, os.cpu_count() or 1))


def parse(argv):
    ap = argparse.ArgumentParser(prog="parity_lock.py",
                                 description="The regression lock of twixtor: a baseline build against a new one.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, vfx_required=False):
        p.add_argument("--out", default=str(DEFAULT_OUT), help="the lock folder (default target/parity-lock)")
        if vfx_required:
            p.add_argument("--vfx", required=True, help="the new vfx")
        p.add_argument("--jobs", type=int, default=jobs_default(), help="parallel runs (default: cores, at most 8)")
        p.add_argument("--sets", type=lambda s: s.split(","), default=list(SETS), help="corpora (default L1,L2,L3)")
        p.add_argument("--limit", type=int, help="run only the first N invocations (a quick look; never recorded)")

    p = sub.add_parser("base", help="build vfx and fx-parity of a ref in a temporary worktree")
    p.add_argument("--ref", required=True, help="the baseline commit, tag or branch")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="the lock folder (default target/parity-lock)")
    p.add_argument("--repo", default=str(ROOT), help=argparse.SUPPRESS)
    p.add_argument("--cargo", default=os.environ.get("CARGO", "cargo"), help="cargo (default $CARGO or cargo)")
    p.add_argument("--toolchain", help="a rustup toolchain, passed as +NAME")

    p = sub.add_parser("corpus", help="generate the corpora L1 and L2 with the baseline tools")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="the lock folder (default target/parity-lock)")
    p.add_argument("--parity", help="the baseline fx-parity (default DIR/base)")
    p.add_argument("--vfx", help="the baseline vfx (default DIR/base)")
    p.add_argument("--force", action="store_true", help="regenerate corpora that exist")

    p = sub.add_parser("refs", help="run the baseline vfx on every case")
    common(p)
    p.add_argument("--vfx", help="the baseline vfx (default DIR/base)")

    p = sub.add_parser("check", help="P2: the cases the baseline rendered, on a new vfx")
    common(p, vfx_required=True)

    p = sub.add_parser("refused", help="P3: the cases the baseline refused, on a new vfx")
    common(p, vfx_required=True)

    p = sub.add_parser("record", help="write docs/data/lock.json from the last complete check")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="the lock folder (default target/parity-lock)")
    p.add_argument("--data", default=str(ROOT / "docs" / "data" / "lock.json"), help="the record to write")
    p.add_argument("--ref", help="the baseline ref when DIR has no base.json (built elsewhere)")
    p.add_argument("--repo", default=str(ROOT), help=argparse.SUPPRESS)
    p.add_argument("--date", help="the date to record (default today, YYYY-MM-DD)")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    handlers = {"base": cmd_base, "corpus": cmd_corpus, "refs": cmd_refs, "check": cmd_check,
                "refused": cmd_refused, "record": cmd_record}
    try:
        return handlers[a.cmd](a)
    except Setup as e:
        print("parity_lock: error: {}".format(e), file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
