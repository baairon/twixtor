"""Offline checks of PathAppendW."""
import types
import unittest

import win_path as wp

B = chr(92)


class PathAppend(unittest.TestCase):
    def test_join_rules(self):
        self.assertEqual(wp.append("C:" + B + "a", "b"), "C:" + B + "a" + B + "b")
        self.assertEqual(wp.append("C:" + B + "a" + B, "b"), "C:" + B + "a" + B + "b")
        self.assertEqual(wp.append("C:" + B + "a", B + "b"), "C:" + B + "a" + B + "b")
        self.assertEqual(wp.append("C:" + B + "a", ""), "C:" + B + "a")
        self.assertEqual(wp.append("", "b"), "b")

    def test_too_long_fails(self):
        self.assertIsNone(wp.append("C:" + B + "a" * 250, "b" * 20))

    def test_stub_writes_the_result_and_returns_true(self):
        mem = {}
        def write(a, d):
            for i, b in enumerate(bytes(d)):
                mem[a + i] = b
        def read(a, n):
            return bytes(mem.get(a + i, 0) for i in range(n))
        def put(a, s):
            write(a, s.encode("utf-16le") + bytes(2))
        put(0x100, "C:" + B + "dir")
        put(0x400, "file.dat")
        stubs, args, res = {}, [0x100, 0x400], []
        wp.install(stubs, lambda q, i: args[i], lambda q, v: res.append(v))
        ql = types.SimpleNamespace(mem=types.SimpleNamespace(read=read, write=write))
        emu = types.SimpleNamespace(rec=types.SimpleNamespace(log=lambda *a, **k: None))
        stubs["PathAppendW"](emu, ql)
        self.assertEqual(res, [1])
        self.assertEqual(read(0x100, 36), ("C:" + B + "dir" + B + "file.dat").encode("utf-16le") + bytes(36 - 30))


if __name__ == "__main__":
    unittest.main()
