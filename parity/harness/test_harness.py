#!/usr/bin/env python3
"""Tests of the parts of the harness that need no emulator: python3 -m unittest test_harness"""
import array
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ae_mock  # noqa: E402
import keys  # noqa: E402
import make_hive  # noqa: E402
import make_param_map  # noqa: E402


class Conversions(unittest.TestCase):
    def test_8bit_is_argb_straight(self):
        wire = bytes([10, 20, 30, 40, 50, 60, 70, 80])           # R G B A, twice
        world = ae_mock.wire_to_world(8, 2, 1, wire)
        self.assertEqual(world, bytes([40, 10, 20, 30, 80, 50, 60, 70]))
        self.assertEqual(ae_mock.world_to_wire(8, 2, 1, world), wire)

    def test_16bit_uses_the_rewrites_q15_rounding(self):
        vals = [0, 1, 32767, 32768, 65534, 65535, 12345, 40000]
        wire = array.array("H", vals).tobytes()
        world = array.array("H")
        world.frombytes(ae_mock.wire_to_world(16, 2, 1, wire))
        # A R G B order: the second wire value is G, the fourth is A
        q = [(v * 32768 + 32767) // 65535 for v in vals]
        self.assertEqual(list(world), [q[3], q[0], q[1], q[2], q[7], q[4], q[5], q[6]])
        self.assertEqual(max(world), 32768)
        for v in range(0, 65536, 257):
            one = ae_mock.world_to_wire(16, 1, 1, array.array("H", [q15 := (v * 32768 + 32767) // 65535, q15, q15, q15]).tobytes())
            back = array.array("H")
            back.frombytes(one)
            self.assertEqual(back[0], (q15 * 65535 + 16384) // 32768)

    def test_float_is_planar_gbra_on_the_wire_and_argb_in_the_world(self):
        # two pixels: planes G, B, R, A
        planes = array.array("f", [1, 2, 3, 4, 5, 6, 7, 8])
        world = array.array("f")
        world.frombytes(ae_mock.wire_to_world(32, 2, 1, planes.tobytes()))
        self.assertEqual(list(world), [7, 5, 1, 3, 8, 6, 2, 4])  # A R G B per pixel
        back = array.array("f")
        back.frombytes(ae_mock.world_to_wire(32, 2, 1, world.tobytes()))
        self.assertEqual(list(back), list(planes))

    def test_float_bits_survive(self):
        nan = array.array("I", [0x7FC00001, 0x80000000, 0x3F800000, 0xFFC00000, 1, 2, 3, 4])
        wire = nan.tobytes()
        back = ae_mock.world_to_wire(32, 2, 1, ae_mock.wire_to_world(32, 2, 1, wire))
        self.assertEqual(back, wire)


class TimeBase(unittest.TestCase):
    def test_default_time_base(self):
        a = ae_mock.TimeBase("24")
        self.assertEqual((a.time_scale, a.time_step), (2400, 100))
        b = ae_mock.TimeBase("30000/1001")
        self.assertEqual((b.time_scale, b.time_step), (30000, 1001))

    def test_footage_frame_of_a_tick(self):
        a = ae_mock.TimeBase("24")
        self.assertEqual(a.frame_of(250, 100, 2400), 2)
        self.assertEqual(a.frame_of(-1, 100, 2400), -1)
        self.assertEqual(a.frame_of(1, 1, 24), 1)            # a checkout on another clock
        b = ae_mock.TimeBase("30000/1001")
        self.assertEqual(b.frame_of(1001, 1001, 30000), 1)
        self.assertEqual(b.frame_of(1000, 1001, 30000), 0)


class Keys(unittest.TestCase):
    def test_constants_and_kinds(self):
        self.assertEqual(keys.make_value("float", "12.5", 100), 12.5)
        self.assertEqual(keys.make_value("popup", "3", 100), 3)
        self.assertEqual(keys.make_value("color", "1,2,3", 100), (1, 2, 3))
        self.assertEqual(keys.make_value("point", "1.5,2", 100), (1.5, 2.0))

    def test_linear_hold_ease(self):
        lin = keys.make_value("float", "0@0;10@4", 100)
        self.assertEqual([lin(t) for t in (-5, 0, 200, 400, 900)], [0, 0, 5, 10, 10])
        hold = keys.make_value("float", "0@0h;10@4", 100)
        self.assertEqual(hold(399), 0)
        ease = keys.make_value("float", "0@0e;10@4e", 100)
        self.assertEqual(ease(200), 5)
        self.assertEqual(ease(100), 10 * (0.25 * 0.25 * (3 - 0.5)))

    def test_discrete_kinds_hold(self):
        v = keys.make_value("popup", "1@0;3@2", 100)
        self.assertEqual([v(0), v(199), v(200)], [1, 1, 3])

    def test_bezier_handles_are_refused(self):
        with self.assertRaises(keys.Unsupported):
            keys.make_value("float", "100@0o0/50;50@24i-30/50h", 100)


class Hive(unittest.TestCase):
    def test_hive_opens(self):
        try:
            from Registry import Registry
        except ImportError:
            self.skipTest("python-registry is not installed")
        with tempfile.TemporaryDirectory() as d:
            make_hive.main(d)
            r = Registry.Registry(str(Path(d) / "SOFTWARE"))
            self.assertEqual(len(r.root().subkeys()), 0)
            with self.assertRaises(Registry.RegistryKeyNotFoundException):
                r.open("Microsoft")


class ParamMap(unittest.TestCase):
    def test_matching(self):
        listing = [{"id": "retime", "params": [
            {"id": "speed_percent", "name": "Speed %", "kind": "float", "default": 100},
            {"id": "vectors", "name": "Vectors", "kind": "popup", "choices": ["A", "B"], "default": 1},
            {"id": "never", "name": "Never Registered", "kind": "checkbox", "default": False}],
            "layers": [{"id": "alt_source_a", "name": "Alt Source A", "optional": True}]}]
        dump = {"plugin": "Twixtor_64", "params": [
            {"reg_index": 1, "name": "Speed %", "type_name": "FLOAT_SLIDER", "min": 0.0, "max": 1.0, "default": 100.0, "flags": 0},
            {"reg_index": 2, "name": "VECTORS", "type_name": "POPUP", "choices": ["A", "B"], "default": 1, "flags": 0},
            {"reg_index": 3, "name": "Alt. Source-A", "type_name": "LAYER", "flags": 0},
            {"reg_index": 4, "name": "Group", "type_name": "GROUP_START", "flags": 0},
            {"reg_index": 5, "name": "Mystery", "type_name": "SLIDER", "flags": 0}]}
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "list.json").write_text(json.dumps(listing))
            (Path(d) / "dump.json").write_text(json.dumps(dump))
            make_param_map.main(["x", str(Path(d) / "list.json"), str(Path(d) / "out.tsv"), str(Path(d) / "dump.json")])
            rows = [dict(zip(make_param_map.COLUMNS, l.split("\t"))) for l in (Path(d) / "out.tsv").read_text().splitlines()[1:]]
        by = {r["ae_name"] or r["vfx_name"]: r for r in rows}
        self.assertEqual(by["Speed %"]["match"], "exact")
        self.assertEqual(by["VECTORS"]["match"], "normalized")
        self.assertEqual(by["VECTORS"]["vfx_id"], "vectors")
        self.assertEqual(by["Alt. Source-A"]["vfx_id"], "alt_source_a")
        self.assertEqual(by["Alt. Source-A"]["vfx_kind"], "layer")
        self.assertEqual(by["Group"]["match"], "unmatched")
        self.assertEqual(by["Mystery"]["match"], "unmatched")
        self.assertEqual(by["Never Registered"]["match"], "unmatched")
        self.assertEqual(by["Never Registered"]["ae_index"], "")


class FpDefaults(unittest.TestCase):
    def test_windows_x64_start_values_replace_unicorns_reset(self):
        from unicorn import UC_ARCH_X86, UC_MODE_64, Uc
        from unicorn.x86_const import UC_X86_REG_FPCW, UC_X86_REG_MXCSR

        import ae_host

        uc = Uc(UC_ARCH_X86, UC_MODE_64)               # Unicorn 2.1.4 resets both to 0
        emu = ae_host.Emu.__new__(ae_host.Emu)
        emu.ql = type("Ql", (), {"uc": uc})()
        emu._fp_defaults()
        mxcsr, fpcw = uc.reg_read(UC_X86_REG_MXCSR), uc.reg_read(UC_X86_REG_FPCW)
        self.assertEqual(mxcsr, 0x1F80)
        self.assertEqual((mxcsr >> 7) & 0x3F, 0x3F)         # MXCSR[7:12] exception masks all 1s
        self.assertEqual(mxcsr & 0xE040, 0)                  # MXCSR[6] DAZ, [13:14] round to nearest, [15] FTZ: 0
        self.assertEqual(fpcw, 0x27F)
        self.assertEqual(fpcw & 0x3F, 0x3F)                  # FPCSR exception masks all 1s
        self.assertEqual((fpcw >> 8) & 3, 2)                 # FPCSR[8:9] precision control 10B (double)
        self.assertEqual((fpcw >> 10) & 7, 0)                # FPCSR[10:11] round to nearest, [12] infinity control 0


if __name__ == "__main__":
    unittest.main()
