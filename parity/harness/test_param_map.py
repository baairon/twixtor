"""Offline checks of make_param_map.py: ae_index and ae_kind on every registered param, and the overrides file."""
import json
import tempfile
import unittest
from pathlib import Path

import make_param_map as mpm

LISTING = [{"id": "retime", "params": [{"kind": "popup", "id": "vectors", "name": "vectors", "default": 1, "choices": ["a", "b"]},
                                         {"kind": "float", "id": "speed", "name": "Speed"}],
            "layers": [{"id": "alt_source_a", "name": "alt_source_a"}]}]
DUMP = {"plugin": "Twixtor_64", "params": [
    {"reg_index": 1, "name": "Speed", "type_name": "FLOAT_SLIDER", "flags": 0},
    {"reg_index": 2, "name": "Motion Vectors", "type_name": "POPUP", "flags": 0, "choices": ["a", "b"]},
    {"reg_index": 3, "name": "Alt Color Source", "type_name": "LAYER", "flags": 0},
    {"reg_index": 4, "name": "Group", "type_name": "GROUP_START", "flags": 0}]}


def run(overrides=None, pairing=None):
    with tempfile.TemporaryDirectory() as d:
        mpm.PARITY_DIR = Path(d) / "parity"
        if pairing is not None:
            (Path(d) / "parity" / "retime-params").mkdir(parents=True)
            (Path(d) / "parity" / "retime-params" / "pairing.tsv").write_text(pairing)
        (Path(d) / "list.json").write_text(json.dumps(LISTING))
        (Path(d) / "dump.json").write_text(json.dumps(DUMP))
        argv = ["x"]
        if overrides is not None:
            (Path(d) / "o.tsv").write_text(overrides)
            argv += ["--overrides", str(Path(d) / "o.tsv")]
        argv += [str(Path(d) / "list.json"), str(Path(d) / "out.tsv"), str(Path(d) / "dump.json")]
        mpm.main(argv)
        lines = (Path(d) / "out.tsv").read_text().splitlines()
        return [dict(zip(lines[0].split("\t"), l.split("\t"))) for l in lines[1:]]


class ParamMap(unittest.TestCase):
    def test_ae_index_and_kind_on_every_registered_param(self):
        rows = [r for r in run() if r["ae_index"]]
        self.assertEqual([r["ae_index"] for r in rows], ["1", "2", "3", "4"])
        self.assertEqual([r["ae_kind"] for r in rows], ["float", "popup", "layer", "group_start"])
        self.assertEqual(rows[0]["match"], "exact")
        self.assertEqual(rows[1]["match"], "unmatched")

    def test_overrides_pair_by_name_or_index_and_win(self):
        rows = run(chr(10).join(["# comment", "ae_name" + chr(9) + "vfx_id", "Motion Vectors" + chr(9) + "vectors", ""]))
        by = {r["ae_index"]: r for r in rows if r["ae_index"]}
        self.assertEqual((by["2"]["vfx_id"], by["2"]["match"]), ("vectors", "override"))
        self.assertFalse([r for r in rows if not r["ae_index"] and r["vfx_id"] == "vectors"])

    def test_overrides_by_index(self):
        rows = run("ae_index\tvfx_id\n3\talt_source_a\n")
        by = {r["ae_index"]: r for r in rows if r["ae_index"]}
        self.assertEqual((by["3"]["vfx_id"], by["3"]["vfx_kind"], by["3"]["match"]), ("alt_source_a", "layer", "override"))

    def test_an_override_takes_a_pairing_from_the_name_match(self):
        rows = run("ae_index\tvfx_id\n2\tspeed\n")
        by = {r["ae_index"]: r for r in rows if r["ae_index"]}
        self.assertEqual((by["1"]["vfx_id"], by["1"]["match"]), ("", "unmatched"))
        self.assertEqual(by["2"]["vfx_id"], "speed")

    def test_pairing_file_of_the_effect_uses_its_own_column_names_and_wins(self):
        tab, nl = chr(9), chr(10)
        pairing = nl.join([tab.join(["plugin_index", "plugin_name", "rewrite_id", "status"]),
                           tab.join(["2", "Motion Vectors", "vectors", "by-name-and-evidence"]),
                           tab.join(["1", "Speed", "", "plugin-only"]), ""])
        rows = run(pairing=pairing)
        by = {r["ae_index"]: r for r in rows if r["ae_index"]}
        self.assertEqual((by["2"]["vfx_id"], by["2"]["match"]), ("vectors", "override"))
        self.assertEqual(by["1"]["match"], "exact")        # a row without rewrite_id changes nothing


if __name__ == "__main__":
    unittest.main()
