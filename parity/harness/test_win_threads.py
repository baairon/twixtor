"""Offline checks of the cooperative guest scheduler."""
import unittest

import win_threads as wt
from win_semaphore import Semaphore


class FakeCpu:
    def __init__(self):
        self.regs = {"rip": 0x1000, "rsp": 0x9000, "rcx": 0, "rbp": 0}
        self.mem = {}
        self.stopped = None

    def save(self):
        return dict(self.regs)

    def restore(self, ctx):
        self.regs = dict(ctx)

    def write_u64(self, a, v):
        self.mem[a] = v

    def set_stack(self, base, limit):
        self.tib = (base, limit)

    def stop(self, reason):
        self.stopped = reason


class Sched(unittest.TestCase):
    def setUp(self):
        self.cpu = FakeCpu()
        self.top = 0x100000
        self.s = wt.Scheduler(self.cpu, self.alloc)
        self.s.end_marker = 0xE0D

    def alloc(self, n):
        a, self.top = self.top, self.top + n
        return a

    def spawn(self, flags=0):
        return self.s.create(0x2000, 7, 0, flags, self.cpu.save())

    def test_created_thread_starts_at_its_routine_with_a_return_to_the_marker(self):
        t = self.spawn()
        self.assertEqual((t.ctx["rip"], t.ctx["rcx"]), (0x2000, 7))
        self.assertEqual(t.ctx["rsp"] % 16, 8)
        self.assertEqual(self.cpu.mem[t.ctx["rsp"]], 0xE0D)
        self.assertEqual(self.cpu.regs["rip"], 0x1000)           # the creator keeps running

    def test_a_suspended_thread_does_not_run_until_resumed(self):
        t = self.spawn(wt.CREATE_SUSPENDED)
        sem = Semaphore(0, 1)
        self.assertFalse(self.s.block(sem))
        self.assertEqual(self.s.resume(t), 1)
        self.assertEqual(self.s.resume(t), 0)
        self.assertTrue(self.s.block(sem))
        self.assertIs(self.s.cur, t)
        self.assertEqual(self.cpu.regs["rip"], 0x2000)

    def test_block_saves_the_blocked_thread_and_wakes_it_when_the_token_arrives(self):
        t = self.spawn()
        sem = Semaphore(0, 4)
        self.assertTrue(self.s.block(sem))
        self.assertIs(self.s.cur, t)
        self.assertEqual(self.s.main.state, "blocked")
        self.assertEqual(self.s.main.ctx["rip"], 0x1000)
        sem.release(1)
        self.assertTrue(self.s.exit_current(0))                  # the worker ends: main runs again
        self.assertIs(self.s.cur, self.s.main)
        self.assertEqual(self.cpu.regs["rip"], 0x1000)
        self.assertEqual(self.s.main.state, "ready")

    def test_waiting_on_a_thread_wakes_when_it_ends(self):
        t = self.spawn()
        self.assertTrue(self.s.block(t))
        self.assertIs(self.s.cur, t)
        self.assertTrue(self.s.exit_current(3))
        self.assertIs(self.s.cur, self.s.main)
        self.assertEqual(t.exit_code, 3)

    def test_deadlock_stops_the_run(self):
        t = self.spawn()
        sem = Semaphore(0, 4)
        self.s.block(sem)                                        # main blocks, t runs
        self.assertFalse(self.s.block(Semaphore(0, 4)))          # t blocks on another empty semaphore: nobody can run
        self.assertTrue(self.s.exit_current(0) is False)
        self.assertIn("deadlock", self.cpu.stopped)

    def test_round_robin_in_creation_order(self):
        a, b = self.spawn(), self.spawn()
        sem = Semaphore(0, 4)
        self.s.block(sem)
        self.assertIs(self.s.cur, a)
        self.s.exit_current(0)
        self.assertIs(self.s.cur, b)

    def test_the_running_thread_has_its_own_stack_bounds_in_the_tib(self):
        self.s.main.stack = (0x80000, 0x40000)
        t = self.spawn()
        self.assertEqual(t.stack, (0x100000 + wt.MIN_STACK, 0x100000))   # the whole allocation, high end first
        self.assertTrue(t.stack[1] <= t.ctx["rsp"] < t.stack[0])
        sem = Semaphore(0, 4)
        self.assertTrue(self.s.block(sem))                              # main blocks, t runs
        self.assertEqual(self.cpu.tib, t.stack)
        sem.release(1)
        self.assertTrue(self.s.exit_current(0))                         # t ends, main runs again
        self.assertIs(self.s.cur, self.s.main)
        self.assertEqual(self.cpu.tib, (0x80000, 0x40000))

    def test_wait_on_several_objects_wakes_when_any_is_ready(self):
        t = self.spawn()
        a, b = Semaphore(0, 4), Semaphore(0, 4)
        self.assertTrue(self.s.block([a, b]))
        self.assertIs(self.s.cur, t)
        b.release(1)
        self.assertTrue(self.s.exit_current(0))
        self.assertIs(self.s.cur, self.s.main)


if __name__ == "__main__":
    unittest.main()
