#!/usr/bin/env python3
"""Call-record rule of vectors-in (`phase_b.py calls`, rule `reads`): what the original asks the host for against what
the rewrite reads, per output frame, as sets. Python 3 standard library only.

How the original asks (read from `TwixtorProVectorsIn_64`, re-cmv notes): its PRE_RENDER lists, per plan entry, every layer
at `X + off` and at `-X` (registrations, the "dummy" pairs of unset layers included) and probes the vector layers
at the current time with size-only checkouts; its SMART_RENDER then checks out the PIXELS of the probes it needs and of
the first registration after the probes, in every output frame. Registrations are declarations for the host's cache and
dependency tracking; only pixel checkouts are reads. The rewrite records every `frame_at` it makes (`VFX_CALL_RECORD`)
and lists its registrations in `Effect::dependencies`, which the recorder does not see. So the check compares reads with
reads and keeps the registrations apart (class `pre`, never counted as a match or a miss).

What is compared (the rule):
  plugin side   every SMART_RENDER pixel checkout (`smart_cb.checkout_layer_pixels`), resolved through its pre-render
                checkout id of the same output frame to (layer, time, footage frame); and, kept apart, every pre-render
                checkout (`prerender_cb.checkout_layer`) of the frame;
  rewrite side  every `read` line of VFX_CALL_RECORD (each `frame_at` call: layer, time, clip index).
  Times are compared as exact fractions of a second (ticks / time_scale), so two time bases compare correctly.

A case matches when, in every output frame,
  (1) every pixel read of the plugin is a read of the rewrite: the same layer at the same time, and the rewrite's clip
      index agrees with the plugin's footage frame (both outside the clip, or the same frame);
  (2) every read of the rewrite is either a pixel read of the plugin (as in 1) or the same layer at the same time as
      one of the plugin's pre-render checkouts of that frame (class `probe`, see below);
  (3) the frame times agree.

The differences that are treated as not substantive, and why:
  name     the plugin's Color Source (param 11) left unset gets the case's input clip (harness default-layer answer,
           owner decision, `default_layer` line of the record); the plugin's reads of it are named `input`, which is
           what the rewrite calls the same clip. Same clip, same times; only the name differs.
  probe    a rewrite read that matches a plugin pre-render checkout of the same layer at the same time in the same
           frame, without a plugin pixel read: the plugin asked the host for that layer at that time (the presence
           probes of the vector layers at the current time), the rewrite has no size-only call (its `Layer` trait
           returns the frame), so the same request shows up as a read. The request and its arguments agree.
Nothing else is normalised. A pixel read of the plugin that the rewrite does not make is a mismatch (class
`plugin_only`), a rewrite read with no plugin request is a mismatch (`rewrite_only`), an index or a time that differs is
a mismatch. Reads of layers the case does not set are kept on both sides.
"""
import collections
import json
from fractions import Fraction

PLUGIN = "TwixtorProVectorsIn_64"


def layer_names(path):
    out = {0: "input"}
    with open(path, encoding="utf8") as fh:
        head = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            r = dict(zip(head, line.rstrip("\n").split("\t")))
            if r.get("plugin") == PLUGIN and r.get("vfx_kind") == "layer" and r.get("ae_index", "").strip().isdigit():
                out[int(r["ae_index"])] = r["vfx_id"]
    return out


def plugin_side(lines, names):
    """{'frames': {n: secs}, 'pix': {n: [(layer, secs, ff, inside, rel)]}, 'pre': {n: set((layer, secs))},
    'done': bool, 'renamed': param index or None}"""
    names = dict(names)
    frames, pix, pre = {}, collections.defaultdict(list), collections.defaultdict(set)
    ids, cur, done, renamed = {}, None, False, None
    for d in lines:
        ev = d.get("ev")
        if ev == "default_layer":
            # the harness gave this unset layer param the case's input clip
            if d.get("layer") == "input" and d.get("param_index") is not None:
                names[int(d["param_index"])] = "input"
                renamed = int(d["param_index"])
        elif ev == "frame":
            cur = d["out_frame"]
            frames[cur] = Fraction(d["time"], d["time_scale"])
            ids = {}
        elif ev == "case_done":
            done = d.get("status") == "rendered"
        elif ev == "cb":
            fn = d.get("fn", "")
            if fn == "prerender_cb.checkout_layer" and "checkout_id" in d:
                lay = names.get(d.get("param_index"), "param%s" % d.get("param_index"))
                secs = Fraction(d["time"], d["time_scale"])
                ids[d["checkout_id"]] = (lay, secs, d.get("footage_frame"), d.get("inside"))
                pre[cur].add((lay, secs))
            elif fn == "smart_cb.checkout_layer_pixels":
                pix[cur].append(ids.get(d.get("checkout_id"), ("?id%s" % d.get("checkout_id"), None, None, None)))
    return {"frames": frames, "pix": pix, "pre": pre, "done": done, "renamed": renamed}


def rewrite_side(lines):
    frames, reads = {}, collections.defaultdict(list)
    scale = None
    for d in lines:
        ev = d.get("ev")
        if ev == "run":
            scale = d.get("time_scale")
        elif ev == "frame":
            scale = d.get("time_scale", scale)
            frames[d["out_frame"]] = Fraction(d["time"], scale)
        elif ev == "read":
            reads[d["out_frame"]].append((d["layer"], Fraction(d["time"], scale), d.get("index"), d.get("clip_frames")))
    return {"frames": frames, "reads": reads}


def index_agrees(rw_index, ff, inside, clip_frames):
    """the rewrite's clip index against the plugin's footage frame and inside flag"""
    if rw_index is None:
        if inside is False:
            return True
        return ff is not None and clip_frames is not None and not 0 <= ff < clip_frames
    if inside is False:
        return False
    return ff == rw_index


def compare(plug, rew):
    """per frame problems and per class counts for one case"""
    problems, classes = [], collections.Counter()
    for n in sorted(set(plug["frames"]) | set(rew["frames"])):
        pt, rt = plug["frames"].get(n), rew["frames"].get(n)
        if pt != rt:
            problems.append({"out_frame": n, "kind": "frame_time", "plugin": str(pt), "rewrite": str(rt)})
            classes["frame_time"] += 1
            continue
        P = {}
        for lay, secs, ff, inside in plug["pix"].get(n, []):
            P.setdefault((lay, secs), (ff, inside))
        R = {}
        for lay, secs, idx, cf in rew["reads"].get(n, []):
            R.setdefault((lay, secs), (idx, cf))
        pre = plug["pre"].get(n, set())
        frame_probs = []
        for key in sorted(P, key=str):
            if key not in R:
                classes["plugin_only"] += 1
                frame_probs.append({"kind": "plugin_only", "read": "%s@%s" % (key[0], key[1]), "rel_to_X": str(key[1] - pt)})
            else:
                ff, inside = P[key]
                idx, cf = R[key]
                if not index_agrees(idx, ff, inside, cf):
                    classes["index"] += 1
                    frame_probs.append({"kind": "index", "read": "%s@%s" % key, "plugin_frame": ff, "rewrite_index": idx})
                else:
                    classes["same"] += 1
        for key in sorted(R, key=str):
            if key in P:
                continue
            if key in pre:
                classes["probe"] += 1
            else:
                classes["rewrite_only"] += 1
                frame_probs.append({"kind": "rewrite_only", "read": "%s@%s" % key})
        if frame_probs:
            problems.append({"out_frame": n, "kind": "reads", "items": frame_probs})
    return problems, classes


def read_lines(path):
    out = []
    with open(path, encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def compare_case(plugin_lines, rewrite_lines, names):
    """One case: {'status': 'match' | 'mismatch' | 'plugin_record_incomplete', 'frames_compared', 'frames_matching',
    'mismatch_frames', 'first_mismatch', 'classes', 'renamed'}."""
    plug = plugin_side(plugin_lines, names)
    if not plug["done"]:
        return {"status": "plugin_record_incomplete"}
    rew = rewrite_side(rewrite_lines)
    probs, cls = compare(plug, rew)
    frames = sorted(set(plug["frames"]) | set(rew["frames"]))
    bad = sorted({p["out_frame"] for p in probs})
    return {"status": "mismatch" if probs else "match", "frames_compared": len(frames),
            "frames_matching": len(frames) - len(bad), "mismatch_frames": bad,
            "first_mismatch": probs[0] if probs else None, "classes": dict(cls), "renamed": plug["renamed"]}
