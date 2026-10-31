#!/usr/bin/env python3
"""Applies one mutant of mutants/mutants.json to a scratch copy of the source tree.

    mutate.py --pristine SRC --scratch DIR --mutants FILE --id m01      (id "none" only restores)

Every file that any mutant touches is first restored from the pristine tree, then the chosen
mutation is applied; the old text must occur exactly once. Only the scratch copy is changed.
"""
import argparse
import json
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pristine", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--mutants", required=True)
    ap.add_argument("--id", required=True)
    a = ap.parse_args()
    ms = json.load(open(a.mutants))
    for f in sorted({m["file"] for m in ms}):
        shutil.copyfile("%s/%s" % (a.pristine, f), "%s/%s" % (a.scratch, f))
    if a.id == "none":
        print("restored")
        return
    m = next(m for m in ms if m["id"] == a.id)
    path = "%s/%s" % (a.scratch, m["file"])
    text = open(path, newline="").read()
    nl = "\r\n" if "\r\n" in text else "\n"
    old, new = m["old"].replace("\n", nl), m["new"].replace("\n", nl)
    n = text.count(old)
    assert n == 1, "%s: old text found %d times in %s" % (a.id, n, m["file"])
    open(path, "w", newline="").write(text.replace(old, new))
    print("%s: applied to %s" % (a.id, m["file"]))


if __name__ == "__main__":
    main()
