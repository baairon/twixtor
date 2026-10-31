#!/usr/bin/env python3
"""Build and grid-render mutants in a scratch tree on the VPS, never in the repo.

  python3 run_mutants.py --tree SCRATCH_SRC --testset DIR --base BASE.tsv --out DIR [--ids M01,M02]

Per mutant: apply the text replacement in SCRATCH_SRC (occurrence count checked), cargo build
vfx with 2 jobs, copy the binary to OUT/<id>/vfx, restore the file, render the grid, write
OUT/<id>/results.tsv and OUT/<id>/changed.txt (ids that differ from BASE). A table is written
to OUT/mutants_summary.tsv. A mutant that does not build is reported as `build-failed`.
"""
import argparse, json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare_runs import load, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--testset", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids")
    ap.add_argument("--mutants", default=os.path.join(HERE, "mutants.json"))
    a = ap.parse_args()
    muts = json.load(open(a.mutants))
    if a.ids:
        keep = a.ids.split(",")
        muts = [m for m in muts if m["id"] in keep]
    base = load(a.base)
    os.makedirs(a.out, exist_ok=True)
    env = dict(os.environ, CARGO_BUILD_JOBS="2")
    env["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + env["PATH"]
    summary = []

    def flush():
        with open(os.path.join(a.out, "mutants_summary.tsv"), "w") as fh:
            fh.write("id\tstatus\tchanged_cases\twhat\n")
            for s in summary:
                fh.write("\t".join(s) + "\n")

    for m in muts:
        d = os.path.join(a.out, m["id"])
        os.makedirs(d, exist_ok=True)
        path = os.path.join(a.tree, m["file"])
        orig = open(path, newline="").read()
        n = orig.count(m["old"])
        if n != m["count"]:
            summary.append((m["id"], "anchor-mismatch", f"found {n}, expected {m['count']}", m["what"]))
            flush()
            continue
        open(path, "w", newline="").write(orig.replace(m["old"], m["new"]))
        try:
            r = subprocess.run(["cargo", "build", "--release", "--bin", "vfx", "-j2"], cwd=a.tree, env=env,
                               capture_output=True, text=True)
        finally:
            open(path, "w", newline="").write(orig)
        if r.returncode != 0:
            open(os.path.join(d, "build.log"), "w").write(r.stderr[-4000:])
            summary.append((m["id"], "build-failed", "", m["what"]))
            flush()
            continue
        binp = os.path.join(d, "vfx")
        shutil.copy(os.path.join(a.tree, "target/release/vfx"), binp)
        res = os.path.join(d, "results.tsv")
        subprocess.run([sys.executable, os.path.join(HERE, "render_grid.py"), "--vfx", binp, "--testset", a.testset,
                        "--out", res], check=True)
        ch = changed(base, load(res))
        open(os.path.join(d, "changed.txt"), "w").write("\n".join(ch) + ("\n" if ch else ""))
        summary.append((m["id"], "changed-cases" if ch else "no-change", str(len(ch)), m["what"]))
        os.remove(binp)
        flush()
    for s in summary:
        print("\t".join(s))


if __name__ == "__main__":
    main()
