#!/usr/bin/env python3
"""Builds param_map.tsv: every parameter a plugin registers in PARAMS_SETUP paired with the rewrite's
param (or layer) id from `vfx list --json --all`.

usage: make_param_map.py [--split] [--overrides FILE.tsv] VFX_LIST_JSON OUT_TSV DUMP.json...
  DUMP.json is written by dump_params.py or by capture_msvcr120.py (<out>_params.json), one per plugin.
  --split: also writes param_map_<plugin>.tsv for every plugin in the dumps, next to OUT_TSV; run_harness.py and worker.py pick
  the map of the case's plugin from there. A registered param that is neither paired nor listed with a status in a pairing file
  counts as `unpaired`; column `pairing_status` is the pairing file's status (exact, by-name-and-evidence, group-marker, ...).
  --overrides: a tab separated pairing that wins over the name matching. Columns (header row, lines starting with
  # are ignored): `ae_index` or `ae_name` (the registered param), `vfx_id` (the rewrite's param or layer id) and
  optionally `plugin`; the pairing agents' spelling `plugin_index`, `plugin_name`, `rewrite_id` is accepted too (rows
  without a rewrite_id are ignored). Whatever the flag, parity/<effect>-params/pairing.tsv (effect: retime, retime-fgbg,
  create-vectors, vectors-in) is read for its plugin when it exists, after --overrides, so it wins. A paired row gets
  match `override`; a row that held that vfx_id by name matching becomes unmatched.
match column: exact (same display name), normalized (same after lower casing and dropping every
character that is not a letter or digit), unmatched (a registered parameter without a counterpart,
a group marker, or a rewrite parameter that no registered one claimed).
"""
import json
import re
import sys
from pathlib import Path

EFFECT_OF = {
    "Twixtor_64": "retime",
    "TwixtorPro_64": "retime-fgbg",
    "CreateMotionVectors_64": "create-vectors",
    "TwixtorProVectorsIn_64": "vectors-in",
}

COLUMNS = ["plugin", "effect", "ae_index", "ae_name", "ae_type", "ae_min", "ae_max", "ae_slider_min",
           "ae_slider_max", "ae_default", "ae_choices", "ae_flags", "vfx_kind", "vfx_id", "vfx_name",
           "vfx_default", "vfx_choices", "match", "ae_kind", "pairing_status"]

# ae_index is the param index the host gives the plugin (0 is the input layer; registered params count from 1), the
# number the call record lines carry. ae_kind is the registered param's kind in the rewrite's vocabulary.
AE_KIND = {"POPUP": "popup", "CHECKBOX": "bool", "LAYER": "layer", "SLIDER": "int", "FIX_SLIDER": "float",
           "FLOAT_SLIDER": "float", "ANGLE": "float", "POINT": "point", "POINT_3D": "point3d", "COLOR": "color",
           "BUTTON": "button", "GROUP_START": "group_start", "GROUP_END": "group_end"}

PARITY_DIR = Path(__file__).resolve().parent.parent      # holds <effect>-params/pairing.tsv

DATA_TYPES_NO = {"GROUP_START", "GROUP_END", "NO_DATA", "BUTTON"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return repr(round(v, 6))
    if isinstance(v, (list, tuple)):
        return ",".join(fmt(x) for x in v)
    return str(v)


def vfx_entries(eff: dict) -> list[dict]:
    out = []
    for p in eff["params"]:
        out.append({"kind": p["kind"], "id": p["id"], "name": p["name"], "default": fmt(p.get("default")),
                    "choices": "|".join(p.get("choices", []))})
    for l in eff["layers"]:
        out.append({"kind": "layer", "id": l["id"], "name": l["name"], "default": "", "choices": ""})
    return out


def read_overrides(path):
    import csv
    if not path or not Path(path).exists():
        return []
    lines = [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip() and not l.lstrip().startswith("#")]
    out = []
    for r in csv.DictReader(lines, delimiter="	"):
        # both spellings: ae_index / ae_name / vfx_id and plugin_index / plugin_name / rewrite_id
        out.append({"plugin": (r.get("plugin") or "").strip(),
                    "ae_index": (r.get("ae_index") or r.get("plugin_index") or "").strip(),
                    "ae_name": (r.get("ae_name") or r.get("plugin_name") or "").strip(),
                    "vfx_id": (r.get("vfx_id") or r.get("rewrite_id") or "").strip(),
                    "status": (r.get("status") or "").strip()})
    return out


def apply_overrides(rows, overrides):
    for o in overrides:
        vid = (o.get("vfx_id") or "").strip()
        if not vid:
            for r in rows:
                if r.get("ae_index") not in ("", None) and (not o.get("plugin") or r["plugin"] == o["plugin"]) and (
                        (o.get("ae_index") and str(r["ae_index"]) == o["ae_index"].strip()) or (not o.get("ae_index") and o.get("ae_name") and r["ae_name"] == o["ae_name"].strip())):
                    r["pairing_status"] = o.get("status") or "listed"
            continue
        reg = [r for r in rows if r.get("ae_index") != "" and r.get("ae_index") is not None
               and (not o.get("plugin") or r["plugin"] == o["plugin"])
               and ((o.get("ae_index") and str(r["ae_index"]) == o["ae_index"].strip()) or (not o.get("ae_index") and o.get("ae_name") and r["ae_name"] == o["ae_name"].strip()))]
        for target in reg:
            holders = [r for r in rows if r["plugin"] == target["plugin"] and r.get("vfx_id") == vid and r is not target]
            info = next((h for h in holders), None)
            if info is None:
                continue
            fields = {k: info.get(k, "") for k in ("vfx_kind", "vfx_id", "vfx_name", "vfx_default", "vfx_choices")}
            for h in holders:
                if h.get("ae_index"):
                    for k in fields:
                        h[k] = ""
                    h["match"] = "unmatched"
                else:
                    rows.remove(h)
            target.update(fields)
            target["match"] = "override"
            target["pairing_status"] = o.get("status") or "override"
    return rows


def main(argv):
    argv = list(argv)
    split = "--split" in argv
    if split:
        argv.remove("--split")
    overrides_path = None
    if "--overrides" in argv:
        i = argv.index("--overrides")
        overrides_path = argv[i + 1]
        del argv[i:i + 2]
    parity = PARITY_DIR
    listing = {e["id"]: e for e in json.load(open(argv[1]))}
    rows = []
    for dump_path in argv[3:]:
        dump = json.load(open(dump_path))
        plugin = dump["plugin"]
        eff = EFFECT_OF[plugin]
        pool = vfx_entries(listing[eff])
        used = set()

        def take(pred):
            for i, e in enumerate(pool):
                if i not in used and pred(e):
                    used.add(i)
                    return e
            return None

        for p in dump.get("params", []):
            idx = p["reg_index"]
            base = {"plugin": plugin, "effect": eff, "ae_index": idx, "ae_name": p["name"], "ae_type": p["type_name"],
                    "ae_min": fmt(p.get("min")), "ae_max": fmt(p.get("max")), "ae_slider_min": fmt(p.get("slider_min")),
                    "ae_slider_max": fmt(p.get("slider_max")), "ae_default": fmt(p.get("default")),
                    "ae_choices": "|".join(p.get("choices", [])), "ae_flags": hex(p.get("flags", 0) & 0xFFFFFFFF),
                    "ae_kind": AE_KIND.get(p["type_name"], p["type_name"].lower())}
            if p["type_name"] in DATA_TYPES_NO:
                rows.append({**base, "match": "unmatched"})
                continue
            e = take(lambda e: e["name"] == p["name"])
            how = "exact"
            if e is None:
                e = take(lambda e: norm(e["name"]) == norm(p["name"]))
                how = "normalized"
            if e is None:
                rows.append({**base, "match": "unmatched"})
            else:
                rows.append({**base, "vfx_kind": e["kind"], "vfx_id": e["id"], "vfx_name": e["name"],
                             "vfx_default": e["default"], "vfx_choices": e["choices"], "match": how})
        for i, e in enumerate(pool):
            if i not in used:
                rows.append({"plugin": plugin, "effect": eff, "vfx_kind": e["kind"], "vfx_id": e["id"],
                             "vfx_name": e["name"], "vfx_default": e["default"], "vfx_choices": e["choices"],
                             "match": "unmatched"})
    overrides = read_overrides(overrides_path)
    for plugin, eff in EFFECT_OF.items():           # parity/<effect>-params/pairing.tsv and pairing_overrides.tsv, read when they exist; they win over --overrides
        for name in ("pairing.tsv", "pairing_overrides.tsv"):
            overrides += [{**o, "plugin": plugin} for o in read_overrides(str(parity / (eff + "-params") / name))]
    rows = apply_overrides(rows, overrides)
    tab, nl = chr(9), chr(10)

    def write(path, subset):
        with open(path, "w", newline="") as f:
            f.write(tab.join(COLUMNS) + nl)
            for r in subset:
                f.write(tab.join(str(r.get(c, "")).replace(tab, " ").replace(nl, " ") for c in COLUMNS) + nl)

    write(argv[2], rows)
    if split:       # one map per plugin next to the output: param_map_<plugin>.tsv
        for plugin in EFFECT_OF:
            subset = [r for r in rows if r["plugin"] == plugin]
            if subset:
                write(str(Path(argv[2]).with_name("param_map_" + plugin + ".tsv")), subset)
    counts = {}
    for r in rows:
        k = (r["plugin"], "registered" if r.get("ae_index") else "rewrite only")
        counts.setdefault(k, {"exact": 0, "normalized": 0, "override": 0, "unmatched": 0, "unpaired": 0})[r["match"]] += 1
        if r.get("ae_index") and not r.get("vfx_id") and not r.get("pairing_status"):
            counts[k]["unpaired"] += 1          # a registered param with neither a rewrite id nor a status in a pairing file
    for k, v in sorted(counts.items()):
        print(k, v, file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv)
