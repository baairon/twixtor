#!/usr/bin/env python3
"""The PRE_RENDER checkout list of the original against the rewrite's checkout list (retime-time; standard library).

`phase_b.py calls` compares what the original's SMART_RENDER checks out (pixels) with the rewrite's clip reads. It cannot
see the list of layer checkouts the original makes in SMART_PRE_RENDER (what it tells the host to provide), because the
rewrite's VFX_CALL_RECORD did not log the rewrite's counterpart, `Effect::dependencies`. With the proposed rewrite change
`fixes/80_call_record_checkouts.diff` the record has one `checkouts` line per output frame (the list in order, with
repeats); this tool compares it with the plugin record's `prerender_cb.checkout_layer` lines of the same frame.

  check    per case and output frame: the original's (layer, time) checkouts against the rewrite's list. A case matches
           when the two are equal as SETS in every frame the original rendered (what the host must provide). Multisets
           and order are reported as classes, not counted: the original lists the input at minus the current time
           once more than the rewrite in every frame (f5-closeout.md, strict check class 2), a duplicate request for
           pixels the host already provides.
  kills    renders the cases with each mutant binary (and the control) under VFX_CALL_RECORD and counts a mutant killed
           when on some case where the control's checkouts match the original's, the mutant's do not (m23 of this
           module changes only the checkout list, so no output and no clip read can show it).
  selftest synthetic records: the check finds a missing, an extra and a moved checkout, ignores a duplicate, and skips
           a rewrite record without `checkouts` lines.

Layer names: AE index 0 is `input`, a layer param is its `vfx_id` from the param map (layer params only; as
`phase_b.py layer_kinds`), anything else `param<i>`. Times are compared as exact fractions of a second.

  python3 checkouts.py check --testset TS --plugin-records DIR --rewrite-records DIR --param-map TSV --out FILE
  python3 checkouts.py kills --testset TS --plugin-records DIR --param-map TSV --mutbin DIR --mutant ID ... --out FILE
  python3 checkouts.py selftest
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import tempfile
from fractions import Fraction

MOD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MOD)
import phase_b as pb  # noqa: E402  (the module's own helpers: records, param map, the vfx command line)


def plugin_checkouts(lines, names):
    """out_frame -> [(layer, time)] of the SMART_PRE_RENDER layer checkouts, in order; the case's end line."""
    frames, cur, done = collections.OrderedDict(), None, None
    for l in lines:
        ev = l.get("ev")
        if ev == "frame":
            cur = l["out_frame"]
            frames.setdefault(cur, [])
        elif ev in ("case_done", "case_error"):
            done = l
        elif ev == "cb" and l.get("fn") == "prerender_cb.checkout_layer":
            n = l.get("out_frame", cur)
            if n is None:
                continue
            pi = l.get("param_index")
            name = "input" if pi == 0 else names.get(pi, "param%s" % pi)
            frames.setdefault(n, []).append((name, pb.frac(l.get("time"), l.get("time_scale"))))
    return frames, done


def rewrite_checkouts(lines):
    """out_frame -> [(layer, time)] from the `checkouts` lines, or None when the record has none (an older build);
    frames whose list could not be made map to the error text."""
    run = next((l for l in lines if l.get("ev") == "run"), {})
    scale = run.get("time_scale") or 1
    frames, seen = collections.OrderedDict(), False
    for l in lines:
        if l.get("ev") != "checkouts":
            continue
        seen = True
        if "error" in l:
            frames[l["out_frame"]] = str(l["error"])
        else:
            frames[l["out_frame"]] = [(c["layer"], pb.frac(c["time"], scale)) for c in l.get("list", [])]
    return frames if seen else None


def _fmt(items):
    return sorted("%s@%s" % (a, "?" if t is None else "%d/%d" % (t.numerator, t.denominator)) for a, t in items)


def compare_case(plug, rew):
    """None when every frame the original rendered has the same SET of checkouts, else the first differing frame.
    Also returns the classes (multiplicity, order) seen in matching frames."""
    classes = collections.Counter()
    first = None
    for n in plug:
        p, r = plug[n], rew.get(n)
        if r is None:
            first = first or {"first_out_frame": n, "kind": "missing_frame"}
            continue
        if isinstance(r, str):
            first = first or {"first_out_frame": n, "kind": "rewrite_error", "detail": r}
            continue
        ps, rs = set(p), set(r)
        if ps != rs:
            kind = "+".join(k for k, v in (("missing_checkout", ps - rs), ("extra_checkout", rs - ps)) if v)
            first = first or {"first_out_frame": n, "kind": kind, "missing": _fmt(ps - rs), "extra": _fmt(rs - ps),
                              "plugin": _fmt(p), "rewrite": _fmt(r)}
            continue
        if collections.Counter(p) != collections.Counter(r):
            cp, cr = collections.Counter(p), collections.Counter(r)
            for k in ps:
                if cp[k] != cr[k]:
                    classes["multiplicity %s: original %d, rewrite %d" % (k[0], cp[k], cr[k])] += 1
        elif p != r:
            classes["order"] += 1
    return first, classes


def load_names(param_map):
    return {i: v[0] for i, v in pb.layer_kinds(param_map).items() if v[1] == "layer" and v[0]}


def check(ids, plugin_dir, rewrite_dir, names, frames_only=False):
    rows, classes = [], collections.Counter()
    for cid in ids:
        pp, rp = os.path.join(plugin_dir, cid + ".jsonl"), os.path.join(rewrite_dir, cid + ".jsonl")
        row = {"id": cid}
        if not os.path.exists(pp):
            row.update(state="skipped", reason="no plugin record")
        elif not os.path.exists(rp):
            row.update(state="skipped", reason="no rewrite record")
        else:
            plug, done = plugin_checkouts(pb.parse_jsonl(pp)[0], names)
            rew = rewrite_checkouts(pb.parse_jsonl(rp)[0])
            if done is None or done.get("ev") != "case_done" or done.get("status") != "rendered":
                row.update(state="skipped", reason="the original did not render the case")
            elif rew is None:
                row.update(state="skipped", reason="the rewrite record has no checkouts lines (a build without fix 80)")
            elif not plug:
                row.update(state="skipped", reason="the plugin record has no frames")
            else:
                if frames_only:
                    row["frames_compared"] = len(plug)
                first, cl = compare_case(plug, rew)
                classes.update(cl)
                row["state"] = "match" if first is None else "mismatch"
                if first:
                    row.update(first)
        rows.append(row)
    return rows, classes


def summarize(rows, classes, what):
    doc = collections.OrderedDict()
    cmp_rows = [r for r in rows if r["state"] in ("match", "mismatch")]
    doc["compare"] = what
    doc["compared"] = len(cmp_rows)
    doc["matching"] = sum(1 for r in cmp_rows if r["state"] == "match")
    doc["mismatching"] = len(cmp_rows) - doc["matching"]
    doc["skipped"] = len(rows) - len(cmp_rows)
    doc["classes_in_matching_frames"] = [[k, v] for k, v in classes.most_common()]
    doc["mismatches"] = [r for r in rows if r["state"] == "mismatch"]
    doc["skipped_cases"] = [r for r in rows if r["state"] == "skipped"]
    return doc


WHAT = ("per output frame the original rendered, the SMART_PRE_RENDER layer checkouts (layer, time) of the original "
        "against the rewrite's checkout list (VFX_CALL_RECORD `checkouts`), as sets")


def case_ids(testset):
    return [c["id"] for c in json.load(open(os.path.join(testset, "cases.json")))["cases"]]


def cmd_check(a):
    rows, classes = check(case_ids(a.testset), a.plugin_records, a.rewrite_records, load_names(a.param_map), a.frames_only)
    doc = summarize(rows, classes, WHAT)
    pb.jdump(a.out, doc)
    print("checkouts: %d compared, %d matching, %d mismatching, %d skipped" % (
        doc["compared"], doc["matching"], doc["mismatching"], doc["skipped"]))
    for r in doc["mismatches"][:5]:
        print("  mismatch %s: frame %s %s missing %s extra %s" % (r["id"], r.get("first_out_frame"), r.get("kind"),
                                                                 r.get("missing", [])[:4], r.get("extra", [])[:4]))
    return 0


def render_records(exe, testset, cases, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for c in cases:
            pb.render_case(exe, testset, c, os.path.join(tmp, "o"), record=os.path.join(out_dir, c["id"] + ".jsonl"))


def cmd_kills(a):
    names = load_names(a.param_map)
    cases = json.load(open(os.path.join(a.testset, "cases.json")))["cases"]
    if a.cases:
        want = set(open(a.cases).read().split())
        cases = [c for c in cases if c["id"] in want]
    have = [c for c in cases if os.path.exists(os.path.join(a.plugin_records, c["id"] + ".jsonl"))]
    work = a.work or tempfile.mkdtemp(prefix="checkouts-kills-")
    ids = [c["id"] for c in have]
    results = {}
    for mid in ["m00_control"] + a.mutant:
        d = os.path.join(work, mid)
        if not (os.path.isdir(d) and len(os.listdir(d)) >= len(have)):
            render_records(os.path.join(a.mutbin, mid + ".vfx"), a.testset, have, d)
        results[mid] = {r["id"]: r for r in check(ids, a.plugin_records, d, names)[0]}
    ctl = results["m00_control"]
    disc = [i for i in ids if ctl[i]["state"] == "match"]
    doc = collections.OrderedDict()
    doc["definition"] = ("a mutant is killed iff on some case where the control build's checkout list matches the original's "
                         "(as sets, every frame) the mutant's does not")
    doc["compare"] = WHAT
    doc["cases_with_plugin_record"] = len(ids)
    doc["control_matching"] = len(disc)
    doc["control_skipped"] = sum(1 for i in ids if ctl[i]["state"] == "skipped")
    rows = []
    for mid in a.mutant:
        killing = [i for i in disc if results[mid][i]["state"] == "mismatch"]
        rows.append({"id": mid, "status": "killed" if killing else "survivor", "killing_count": len(killing),
                     "killing_cases": killing[:20],
                     "first": {k: results[mid][killing[0]].get(k) for k in ("first_out_frame", "kind", "missing", "extra")} if killing else None})
    doc["mutants"] = rows
    doc["killed"] = [r["id"] for r in rows if r["status"] == "killed"]
    doc["survivors"] = [r["id"] for r in rows if r["status"] != "killed"]
    pb.jdump(a.out, doc)
    print("checkout kills: control matches %d of %d cases; killed %s; survivors %s" % (
        len(disc), len(ids), ", ".join(doc["killed"]) or "none", ", ".join(doc["survivors"]) or "none"))
    return 0


def cmd_selftest(_a):
    def plugin(frames, done=True):
        out = []
        for n, items in frames.items():
            out.append({"ev": "frame", "out_frame": n, "time": n * 100, "time_scale": 2400})
            for pi, t in items:
                out.append({"ev": "cb", "fn": "prerender_cb.checkout_layer", "param_index": pi, "time": t, "time_scale": 2400,
                            "out_frame": n})
        out.append({"ev": "case_done", "status": "rendered"} if done else {"ev": "case_error", "status": "error"})
        return out

    def rewrite(frames, with_lines=True):
        out = [{"ev": "run", "time_scale": 2400}]
        for n, items in frames.items():
            out.append({"ev": "frame", "out_frame": n, "time": n * 100})
            if with_lines:
                out.append({"ev": "checkouts", "out_frame": n, "list": [{"layer": l, "time": t} for l, t in items]})
        return out

    names = {13: "alt_motion_source"}
    # the original: input at -t three times, Alt Motion Source at -t, then f and f + 1 (the shape of the grid's records)
    orig = {1: [(0, -100), (0, -100), (0, -100), (13, -100), (0, 100), (0, 200)]}
    same = {1: [("input", -100), ("input", -100), ("alt_motion_source", -100), ("input", 100), ("input", 200)]}
    moved = {1: [("input", -100), ("alt_motion_source", -100), ("input", 100), ("input", 300)]}
    lost = {1: [("input", -100), ("input", 100), ("input", 200)]}
    extra = same.copy()
    extra[1] = same[1] + [("input", 400)]
    reordered = {1: list(reversed([("input", -100)] * 3 + [("alt_motion_source", -100), ("input", 100), ("input", 200)]))}
    pl, _ = plugin_checkouts(plugin(orig), names)
    ok = []

    def expect(label, rew, state, kind=None, cls=None):
        first, classes = compare_case(pl, rewrite_checkouts(rewrite(rew)))
        got = "match" if first is None else "mismatch"
        good = got == state and (kind is None or (first or {}).get("kind") == kind) and (cls is None or cls in classes)
        ok.append(good)
        print("%-40s %s (%s %s)" % (label, "ok" if good else "FAILED", got, (first or {}).get("kind") or dict(classes)))

    expect("duplicate input at -t ignored", same, "match", cls="multiplicity input: original 3, rewrite 2")
    expect("f + 2 instead of f + 1 (m23)", moved, "mismatch", kind="missing_checkout+extra_checkout")
    expect("Alt Motion Source dropped", lost, "mismatch", kind="missing_checkout")
    expect("an extra checkout", extra, "mismatch", kind="extra_checkout")
    expect("same multiset, other order", reordered, "match", cls="order")
    with tempfile.TemporaryDirectory() as t:
        pd, rd = os.path.join(t, "p"), os.path.join(t, "r")
        os.makedirs(pd)
        os.makedirs(rd)
        for cid, prec, rrec in (("a", plugin(orig), rewrite(same)), ("b", plugin(orig), rewrite(same, with_lines=False)),
                                ("c", plugin(orig, done=False), rewrite(same)), ("d", plugin(orig), rewrite(moved))):
            for d, lines in ((pd, prec), (rd, rrec)):
                with open(os.path.join(d, cid + ".jsonl"), "w") as fh:
                    fh.write("".join(json.dumps(x) + "\n" for x in lines))
        rows, _ = check(["a", "b", "c", "d", "e"], pd, rd, names)
        states = [(r["id"], r["state"]) for r in rows]
        good = states == [("a", "match"), ("b", "skipped"), ("c", "skipped"), ("d", "mismatch"), ("e", "skipped")]
        ok.append(good)
        print("%-40s %s %s" % ("check over a folder", "ok" if good else "FAILED", states))
    print("selftest: %d of %d ok" % (sum(ok), len(ok)))
    return 0 if all(ok) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--testset", required=True)
    p.add_argument("--plugin-records", required=True)
    p.add_argument("--rewrite-records", required=True)
    p.add_argument("--param-map", required=True)
    p.add_argument("--frames-only", action="store_true", help="a plugin record of selected output frames (HARNESS_FRAMES)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("kills")
    p.add_argument("--testset", required=True)
    p.add_argument("--plugin-records", required=True)
    p.add_argument("--param-map", required=True)
    p.add_argument("--mutbin", required=True, help="m00_control.vfx and <id>.vfx, builds with fix 80")
    p.add_argument("--mutant", action="append", required=True)
    p.add_argument("--cases", help="a file of case ids (default: every case with a plugin record)")
    p.add_argument("--work", help="where the records go (kept; a complete folder is reused)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_kills)
    p = sub.add_parser("selftest")
    p.set_defaults(fn=cmd_selftest)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
