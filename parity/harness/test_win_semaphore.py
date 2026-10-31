"""Offline checks of the startup semaphore path and Win64 hook arguments."""
import struct
import types
import unittest

import win_semaphore as sync
from ae_host import Emu, GuestFault


class Semaphores(unittest.TestCase):
    def setUp(self):
        self.handles = {}
        self.args = []
        self.result = None
        self.writes = {}
        self.events = []
        self.next_handle = 0xA000
        manager = types.SimpleNamespace(get=self.handles.get,
                                        append=lambda h: self.handles.__setitem__(h.id, h))
        self.ql = types.SimpleNamespace(
            os=types.SimpleNamespace(handle_manager=manager, last_error=0),
            mem=types.SimpleNamespace(write=lambda p, b: self.writes.__setitem__(p, bytes(b))),
            arch=types.SimpleNamespace(regs=types.SimpleNamespace(arch_pc=0x1000)))
        self.emu = types.SimpleNamespace(rec=types.SimpleNamespace(
            log=lambda event, **kw: self.events.append((event, kw))))
        self.stubs = {}
        sync.install(self.stubs, lambda q, i: self.args[i], self.ret, RuntimeError, self.handle)

    def handle(self, obj):
        self.next_handle += 4
        return types.SimpleNamespace(id=self.next_handle, obj=obj)

    def ret(self, ql, value):
        self.result = value

    def call(self, name, *args):
        self.args = list(args)
        self.stubs[name](self.emu, self.ql)
        return self.result

    def test_observed_startup_sequence_and_consumption(self):
        h = self.call('CreateSemaphoreA', 0, 0, 1, 0)
        self.assertNotEqual(h, 0)
        self.assertEqual(self.call('ReleaseSemaphore', h, 1, 0x2000), 1)
        self.assertEqual(self.writes[0x2000], struct.pack('<i', 0))
        self.assertEqual(self.call('WaitForSingleObject', h, sync.INFINITE), sync.WAIT_OBJECT_0)
        self.assertEqual(self.call('WaitForSingleObjectEx', h, 0, 0), sync.WAIT_TIMEOUT)

    def test_release_overflow_preserves_count(self):
        h = self.call('CreateSemaphoreW', 0, 1, 1, 0)
        self.assertEqual(self.call('ReleaseSemaphore', h, 1, 0), 0)
        self.assertEqual(self.ql.os.last_error, sync.ERROR_TOO_MANY_POSTS)
        self.assertEqual(self.handles[h].obj.count, 1)

    def test_signed_long_arguments_and_invalid_counts(self):
        for initial, maximum in ((0xFFFFFFFF, 1), (2, 1), (0, 0), (0, 0xFFFFFFFF)):
            self.assertEqual(self.call('CreateSemaphoreA', 0, initial, maximum, 0), 0)
            self.assertEqual(self.ql.os.last_error, sync.ERROR_INVALID_PARAMETER)
        h = self.call('CreateSemaphoreA', 0, 0, 1, 0)
        self.assertEqual(self.call('ReleaseSemaphore', h, 0xFFFFFFFF, 0), 0)
        self.assertEqual(self.handles[h].obj.count, 0)

    def test_invalid_and_closed_handles(self):
        h = self.call('CreateSemaphoreA', 0, 0, 1, 0)
        del self.handles[h]
        self.assertEqual(self.call('WaitForSingleObject', h, 0), sync.WAIT_FAILED)
        self.assertEqual(self.call('ReleaseSemaphore', h, 1, 0), 0)
        self.assertEqual(self.ql.os.last_error, sync.ERROR_INVALID_HANDLE)

    def test_blocking_wait_does_not_fabricate_success(self):
        h = self.call('CreateSemaphoreA', 0, 0, 1, 0)
        with self.assertRaisesRegex(RuntimeError, 'thread scheduling'):
            self.call('WaitForSingleObject', h, sync.INFINITE)
        self.assertEqual(self.handles[h].obj.count, 0)


class RuntimeInitialization(unittest.TestCase):
    def host(self, call):
        host = Emu.__new__(Emu)
        host.ql = types.SimpleNamespace(loader=types.SimpleNamespace(images=[
            types.SimpleNamespace(path='msvcr120.dll', base=0x100000),
            types.SimpleNamespace(path='msvcp120.dll', base=0x200000)]))
        host.r32 = lambda addr: 0x100
        host.call = call
        host.rec = types.SimpleNamespace(log=lambda *a, **kw: None)
        host.init_errors = []
        return host

    def test_crt_fault_stops_before_next_library_or_plugin(self):
        calls = []

        def call(*args, **kw):
            calls.append(args)
            raise GuestFault('unmapped read', 0x1234, 'memset')

        host = self.host(call)
        with self.assertRaisesRegex(GuestFault, 'runtime initialization failed') as ex:
            host.init_libraries()
        self.assertEqual(ex.exception.pc, 0x1234)
        self.assertEqual(ex.exception.last, 'memset')
        self.assertEqual(len(calls), 1)
        self.assertEqual(host.init_errors[0][0], 'msvcr120.dll')

    def test_crt_false_return_is_a_boot_failure(self):
        host = self.host(lambda *a, **kw: 0)
        with self.assertRaisesRegex(GuestFault, 'rejected DLL_PROCESS_ATTACH'):
            host.init_libraries()


if __name__ == '__main__':
    unittest.main()
