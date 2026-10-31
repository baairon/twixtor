"""Offline checks of the CryptoAPI context functions."""
import struct
import types
import unittest

import ae_host
import win_crypt as wc


class CryptContexts(unittest.TestCase):
    def setUp(self):
        self.args = []
        self.result = None
        self.memory = {}
        self.events = []
        self.ql = types.SimpleNamespace(
            os=types.SimpleNamespace(last_error=0),
            mem=types.SimpleNamespace(write=self.write, read=self.read))
        self.emu = types.SimpleNamespace(rec=types.SimpleNamespace(
            log=lambda event, **kw: self.events.append((event, kw))))
        self.stubs = {}
        self.state = wc.install(self.stubs, lambda q, i: self.args[i], lambda q, v: setattr(self, "result", v))

    def write(self, addr, data):
        for i, b in enumerate(bytes(data)):
            self.memory[addr + i] = b

    def read(self, addr, n):
        return bytes(self.memory.get(addr + i, 0) for i in range(n))

    def put(self, addr, text, wide=False):
        self.write(addr, text.encode("utf-16le" if wide else "latin-1") + bytes(2 if wide else 1))

    def call(self, name, *args):
        self.args = list(args)
        self.ql.os.last_error = 0
        self.stubs[name](self.emu, self.ql)
        return self.result

    def handle(self, addr=0x2000):
        return struct.unpack("<Q", self.read(addr, 8))[0]

    def test_verify_context_gives_a_handle_and_release_works(self):
        self.assertEqual(self.call("CryptAcquireContextA", 0x2000, 0, 0, 1, wc.CRYPT_VERIFYCONTEXT), 1)
        h = self.handle()
        self.assertNotEqual(h, 0)
        self.assertEqual(self.call("CryptReleaseContext", h, 0), 1)
        self.assertEqual(self.call("CryptReleaseContext", h, 0), 0)
        self.assertEqual(self.ql.os.last_error, wc.NTE_BAD_UID)

    def test_missing_container_then_newkeyset_then_exists(self):
        self.assertEqual(self.call("CryptAcquireContextA", 0x2000, 0, 0, 1, 0), 0)
        self.assertEqual(self.ql.os.last_error, wc.NTE_BAD_KEYSET)
        self.assertEqual(self.call("CryptAcquireContextA", 0x2000, 0, 0, 1, wc.CRYPT_NEWKEYSET), 1)
        self.assertEqual(self.call("CryptAcquireContextA", 0x2000, 0, 0, 1, wc.CRYPT_NEWKEYSET), 0)
        self.assertEqual(self.ql.os.last_error, wc.NTE_EXISTS)
        self.assertEqual(self.call("CryptAcquireContextA", 0x2000, 0, 0, 1, 0), 1)
        self.assertEqual(self.call("CryptAcquireContextA", 0x2000, 0, 0, 1, wc.CRYPT_DELETEKEYSET), 1)
        self.assertEqual(self.call("CryptAcquireContextA", 0x2000, 0, 0, 1, wc.CRYPT_DELETEKEYSET), 0)
        self.assertEqual(self.ql.os.last_error, wc.NTE_BAD_KEYSET)

    def test_named_container_and_wide_strings(self):
        self.put(0x3000, "box", wide=True)
        self.put(0x3100, "Microsoft Enhanced RSA and AES Cryptographic Provider", wide=True)
        self.assertEqual(self.call("CryptAcquireContextW", 0x2000, 0x3000, 0x3100, 24, wc.CRYPT_NEWKEYSET), 1)
        self.assertEqual(self.call("CryptAcquireContextW", 0x2000, 0x3000, 0x3100, 24, 0), 1)
        self.put(0x3200, "other", wide=True)
        self.assertEqual(self.call("CryptAcquireContextW", 0x2000, 0x3200, 0x3100, 24, 0), 0)
        self.assertEqual(self.ql.os.last_error, wc.NTE_BAD_KEYSET)

    def test_bad_arguments(self):
        self.assertEqual(self.call("CryptAcquireContextA", 0, 0, 0, 1, 0), 0)
        self.assertEqual(self.ql.os.last_error, wc.ERROR_INVALID_PARAMETER)
        self.assertEqual(self.call("CryptAcquireContextA", 0x2000, 0, 0, 1, 0x1), 0)
        self.assertEqual(self.ql.os.last_error, wc.NTE_BAD_FLAGS)
        self.assertEqual(self.call("CryptAcquireContextA", 0x2000, 0, 0, 0, wc.CRYPT_VERIFYCONTEXT), 0)
        self.assertEqual(self.ql.os.last_error, wc.NTE_BAD_PROV_TYPE)
        self.assertEqual(self.call("CryptAcquireContextA", 0x2000, 0, 0, 1000, wc.CRYPT_VERIFYCONTEXT), 0)
        self.assertEqual(self.ql.os.last_error, wc.NTE_BAD_PROV_TYPE)
        self.assertEqual(self.call("CryptAcquireContextA", 0x2000, 0, 0, 5, wc.CRYPT_VERIFYCONTEXT), 0)
        self.assertEqual(self.ql.os.last_error, wc.NTE_PROV_TYPE_NOT_DEF)
        self.put(0x3000, "box")
        self.assertEqual(self.call("CryptAcquireContextA", 0x2000, 0x3000, 0, 1, wc.CRYPT_VERIFYCONTEXT), 0)
        self.assertEqual(self.ql.os.last_error, wc.NTE_BAD_FLAGS)

    def test_provider_name_checks(self):
        self.put(0x3000, "No Such Provider")
        self.assertEqual(self.call("CryptAcquireContextA", 0x2000, 0, 0x3000, 1, wc.CRYPT_VERIFYCONTEXT), 0)
        self.assertEqual(self.ql.os.last_error, wc.NTE_KEYSET_NOT_DEF)
        self.put(0x3000, "microsoft base cryptographic provider v1.0")
        self.assertEqual(self.call("CryptAcquireContextA", 0x2000, 0, 0x3000, 24, wc.CRYPT_VERIFYCONTEXT), 0)
        self.assertEqual(self.ql.os.last_error, wc.NTE_PROV_TYPE_NO_MATCH)
        self.assertEqual(self.call("CryptAcquireContextA", 0x2000, 0, 0x3000, 1, wc.CRYPT_VERIFYCONTEXT), 1)

    def test_release_flags(self):
        self.call("CryptAcquireContextA", 0x2000, 0, 0, 1, wc.CRYPT_VERIFYCONTEXT)
        self.assertEqual(self.call("CryptReleaseContext", self.handle(), 1), 0)
        self.assertEqual(self.ql.os.last_error, wc.NTE_BAD_FLAGS)

    def test_gen_random_is_a_deterministic_stream_and_checks_the_handle(self):
        self.call("CryptAcquireContextA", 0x2000, 0, 0, 1, wc.CRYPT_VERIFYCONTEXT)
        h = self.handle()
        self.assertEqual(self.call("CryptGenRandom", h, 40, 0x4000), 1)
        first = self.read(0x4000, 40)
        self.assertEqual(self.call("CryptGenRandom", h, 8, 0x5000), 1)
        again = wc.Stream()
        self.assertEqual(first, again.read(40))
        self.assertEqual(self.read(0x5000, 8), again.read(8))
        self.assertEqual(self.call("CryptGenRandom", h, 0, 0), 1)
        self.assertEqual(self.call("CryptGenRandom", h, 4, 0), 0)
        self.assertEqual(self.ql.os.last_error, wc.ERROR_NOACCESS)
        self.assertEqual(self.call("CryptGenRandom", 0xDEAD, 4, 0x6000), 0)
        self.assertEqual(self.ql.os.last_error, wc.NTE_BAD_UID)

    def test_log_has_no_buffers_or_strings(self):
        self.put(0x3000, "secret-container")
        self.call("CryptAcquireContextA", 0x2000, 0x3000, 0, 1, wc.CRYPT_NEWKEYSET)
        self.call("CryptGenRandom", self.handle(), 16, 0x4000)
        text = repr(self.events)
        self.assertNotIn("secret-container", text)
        self.assertNotIn(self.read(0x4000, 16).hex(), text)

    def test_installed_in_the_host(self):
        for name in ("CryptAcquireContextA", "CryptAcquireContextW", "CryptReleaseContext", "CryptGenRandom"):
            self.assertIn(name, ae_host.WIN_STUBS)


if __name__ == "__main__":
    unittest.main()
