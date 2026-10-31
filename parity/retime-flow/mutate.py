#!/usr/bin/env python3
"""Apply one mutant from mutants/mutants.json to a scratch copy.

    mutate.py --pristine SRC --scratch DIR --mutants FILE --id m01     (id "none" restores only)

Every file any mutant touches is first restored from the pristine tree, then the chosen
mutation is applied, requiring exactly one occurrence of the old text.
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
        shutil.copyfile(f"{a.pristine}/{f}", f"{a.scratch}/{f}")
    if a.id == "none":
        print("restored")
        return
    m = next(m for m in ms if m["id"] == a.id)
    path = f"{a.scratch}/{m['file']}"
    text = open(path, newline="").read()
    nl = "\r\n" if "\r\n" in text else "\n"
    old, new = m["old"].replace("\n", nl), m["new"].replace("\n", nl)
    n = text.count(old)
    assert n == 1, f"{a.id}: old text found {n} times in {m['file']}"
    open(path, "w", newline="").write(text.replace(old, new))
    print(f"{a.id}: applied to {m['file']}")


if __name__ == "__main__":
    main()
