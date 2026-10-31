"""Offline checks of RtlPcToFileHeader."""
import struct
import types
import unittest

import win_rtl
import ae_host


class RtlPcToFileHeader(unittest.TestCase):
    def setUp(self):
        self.args = []
        self.result = None
        self.writes = {}
        self.events = []
        images = [types.SimpleNamespace(base=0x180000000, end=0x1810f7000, path='plugin.aex'),
                  types.SimpleNamespace(base=0x183010000, end=0x1830ff000, path='msvcr120.dll')]
        self.ql = types.SimpleNamespace(
            loader=types.SimpleNamespace(images=images),
            mem=types.SimpleNamespace(write=lambda p, b: self.writes.__setitem__(p, bytes(b))))
        self.emu = types.SimpleNamespace(rec=types.SimpleNamespace(
            log=lambda event, **kw: self.events.append((event, kw))))
        self.stubs = {}
        win_rtl.install(self.stubs, lambda q, i: self.args[i], lambda q, v: setattr(self, 'result', v))

    def call(self, pc, out):
        self.args = [pc, out]
        self.stubs['RtlPcToFileHeader'](self.emu, self.ql)
        return self.result

    def test_inside_an_image(self):
        self.assertEqual(self.call(0x1830dd258, 0x2000), 0x183010000)
        self.assertEqual(self.writes[0x2000], struct.pack('<Q', 0x183010000))

    def test_edges_of_an_image(self):
        self.assertEqual(self.call(0x183010000, 0x2000), 0x183010000)
        self.assertEqual(self.call(0x1830fefff, 0x2000), 0x183010000)
        self.assertEqual(self.call(0x1830ff000, 0x2000), 0)
        self.assertEqual(self.writes[0x2000], struct.pack('<Q', 0))
        self.assertEqual(self.call(0x18300ffff, 0x2000), 0)

    def test_outside_every_image_writes_null(self):
        self.assertEqual(self.call(0xcd200, 0x2000), 0)
        self.assertEqual(self.writes[0x2000], struct.pack('<Q', 0))

    def test_null_out_pointer_writes_nothing(self):
        self.assertEqual(self.call(0x1830dd258, 0), 0x183010000)
        self.assertEqual(self.writes, {})

    def test_installed_in_the_host_and_not_a_fallback(self):
        self.assertIn('RtlPcToFileHeader', ae_host.WIN_STUBS)


if __name__ == '__main__':
    unittest.main()
