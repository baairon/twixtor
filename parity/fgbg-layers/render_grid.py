#!/usr/bin/env python3
"""Render every case of a test set with one vfx binary; write id, exit code, size and sha256.

  python3 render_grid.py --vfx BIN --testset DIR --out FILE.tsv [--jobs 2] [--only ID,ID] [--cases FILE]

The command per case is the one `fx-parity run` uses: vfx apply EFFECT --width --height --depth
--param ... --fps --frames --in, plus --layer ID=FILE for each matte layer. Output goes to a temp
file, is hashed and removed. --cases names another case list (for example out_of_contract.json).
"""
import argparse, concurrent.futures as cf, hashlib, json, os, subprocess, tempfile, time


def one(vfx, ts, c, tmpdir):
    out = os.path.join(tmpdir, c["id"] + ".out")
    args = [vfx, "apply", c["effect"], "--width", str(c["width"]), "--height", str(c["height"]),
            "--depth", str(c["depth"]), "--fps", c["fps"], "--frames", str(c["frames"]),
            "--in", os.path.join(ts, c["input"]), "--out", out]
    for p in c["params"]:
        args += ["--param", p]
    for lid, rel in sorted(c.get("layers", {}).items()):
        args += ["--layer", f"{lid}={os.path.join(ts, rel)}"]
    t = time.time()
    r = subprocess.run(args, capture_output=True, text=True)
    dt = time.time() - t
    size, sha = 0, "-"
    if os.path.exists(out):
        h = hashlib.sha256()
        with open(out, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
                size += len(chunk)
        sha = h.hexdigest()
        os.remove(out)
    err = (r.stderr.strip().splitlines() or [""])[-1].replace("\t", " ")[:160] if r.returncode else ""
    return (c["id"], r.returncode, size, sha, f"{dt:.2f}", err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vfx", required=True)
    ap.add_argument("--testset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only")
    ap.add_argument("--cases")
    a = ap.parse_args()
    cases = json.load(open(a.cases or os.path.join(a.testset, "cases.json")))["cases"]
    if a.only:
        keep = set(a.only.split(","))
        cases = [c for c in cases if c["id"] in keep]
    with tempfile.TemporaryDirectory() as tmp, cf.ThreadPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(lambda c: one(a.vfx, a.testset, c, tmp), cases))
    with open(a.out, "w") as fh:
        fh.write("id\texit\tbytes\tsha256\tseconds\terror\n")
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
    bad = sum(1 for r in rows if r[1] != 0)
    print(f"{a.out}: {len(rows)} cases, {bad} with a nonzero exit")


if __name__ == "__main__":
    main()
