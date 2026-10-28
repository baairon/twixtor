#!/usr/bin/env python3
"""tests for the --pre-stage, --post-stage and --stage-timeout flags of tools/apply.py. stdlib
unittest, no network.

    python tools/test_stages.py

the unit tests and the relay tests need nothing but python. the pipeline tests run apply.py over
a small clip with ffmpeg and vfx (cargo build --release -p fx-cli, or VFX_PATH) and skip themselves
with a printed reason when either is missing. the stages are fake ones written in python: one that
inverts every byte, and others that exit with a code, stop in the middle of a frame, stop reading,
hang or go quiet.

every test holds for any frame effect, except where it names selftest-invert, the hidden effect
that inverts the colour channels (and alpha with invert_alpha=1), or selftest-shift.
"""

import argparse
import contextlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
APPLY = os.path.join(HERE, "apply.py")

# no bytecode next to the tools, here or in the script the suite runs
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, HERE)
import apply as tool  # noqa: E402

FPS = 24
FRAMES = 12
WIDTH, HEIGHT = 160, 96
# bytes in one frame of the clip at each depth
FRAME_BYTES = {8: WIDTH * HEIGHT * 4, 16: WIDTH * HEIGHT * 8}
TMP = None
CLIP = None
VFX = None
FAKE = None

# the fake stage: python, so it runs wherever the tests do. `python fake_stage.py MODE key=value ...`
FAKE_STAGE = r'''
import json
import os
import shutil
import sys
import threading
import time

mode = sys.argv[1]
opts = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
fb = int(opts.get("fb", 0))
inp, out = sys.stdin.buffer, sys.stdout.buffer
INVERT = bytes(255 - i for i in range(256))
INC = bytes((i + 1) % 256 for i in range(256))
DBL = bytes((i * 2) % 256 for i in range(256))


def frames():
    while True:
        data = inp.read(fb)
        if not data:
            return
        yield data


def put(data):
    out.write(data)
    out.flush()


beat = opts.get("beat")
if beat:
    def beating():
        while True:
            with open(beat, "ab") as handle:
                handle.write(b".")
            time.sleep(0.05)
    threading.Thread(target=beating, daemon=True).start()

if mode in ("invert", "inc", "dbl"):
    table = {"invert": INVERT, "inc": INC, "dbl": DBL}[mode]
    for data in frames():
        put(data.translate(table))
elif mode == "pass":
    for data in frames():
        put(data)
elif mode == "record":
    # record PATH A B C ...: write the arguments to PATH, then pass everything through
    with open(sys.argv[2], "w") as handle:
        handle.write(json.dumps(sys.argv[3:]))
    shutil.copyfileobj(inp, out, 65536)
    out.flush()
elif mode == "fail":
    inp.read(int(opts.get("after", 100)))
    sys.stderr.write("boom: cannot continue\nsecond line\n")
    sys.exit(int(opts["code"]))
elif mode == "failend":
    for data in frames():
        put(data)
    sys.stderr.write("late failure\n")
    sys.exit(int(opts["code"]))
elif mode == "cut":
    data = inp.read()
    put(data[:len(data) - int(opts["bytes"])])
elif mode == "extra":
    for data in frames():
        put(data)
    put(b"x" * int(opts["bytes"]))
elif mode == "hang":
    time.sleep(600)
elif mode == "hold":
    inp.read()
    time.sleep(600)
elif mode == "spawn":
    # a launcher: the hanging program is a child that shares the pipes
    import subprocess
    subprocess.Popen([sys.executable, sys.argv[0], "hang", "beat=" + opts["beat"]])
    time.sleep(600)
elif mode == "hangafter":
    put(inp.read(fb))
    time.sleep(600)
elif mode == "early":
    put(inp.read(fb))
    os.close(1)
    os._exit(0)
elif mode == "early-drain":
    put(inp.read(fb))
    os.close(1)
    inp.read()
elif mode == "slow":
    for data in frames():
        time.sleep(float(opts["delay"]))
        put(data)
elif mode == "late":
    time.sleep(float(opts["delay"]))
    for data in frames():
        put(data)
else:
    sys.exit("unknown mode " + mode)
'''

PASS_THROUGH = "import shutil, sys; shutil.copyfileobj(sys.stdin.buffer, sys.stdout.buffer, 65536); sys.stdout.buffer.flush()"


def ff(*args):
    cmd = [tool.ffmpeg_path(), "-v", "error", "-y"] + list(args)
    subprocess.run(cmd, shell=False, check=True, capture_output=True)


def make_clip(path, size=f"{WIDTH}x{HEIGHT}", frames=FRAMES, audio=True):
    inputs = ["-f", "lavfi", "-i", f"testsrc=size={size}:rate={FPS}:duration=1"]
    if audio:
        inputs += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1"]
    cmd = inputs + ["-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac", "-shortest"] if audio else []
    ff(*(cmd + [path]))


def cmdline(*argv):
    """one stage command line for apply.py, which splits it with shlex: double quotes on windows
    (posix=False keeps the backslashes), shell quoting elsewhere."""
    if os.name == "nt":
        return " ".join('"%s"' % a for a in map(str, argv))
    return " ".join(shlex.quote(str(a)) for a in argv)


def stage(mode, *plain, **opts):
    """the command line of the fake stage in `mode` with key=value options (and plain arguments first)."""
    return cmdline(sys.executable, FAKE, mode, *plain, *[f"{k}={v}" for k, v in opts.items()])


def apply(*args, tmp=None):
    """run apply.py; `tmp` points the temp directory of apply.py, ffmpeg and vfx at a directory of the test."""
    env = dict(os.environ, VFX_PATH=VFX, PYTHONDONTWRITEBYTECODE="1")
    if tmp:
        env.update(TMP=tmp, TEMP=tmp, TMPDIR=tmp)
    return subprocess.run([sys.executable, APPLY] + list(args), shell=False, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", env=env)


def slurp(path):
    with open(path, "rb") as f:
        return f.read()


def raw_output(path, depth):
    """the frames of a file as the raw bytes vfx works with."""
    cmd = [tool.ffmpeg_path(), "-v", "error", "-i", path, "-an", "-pix_fmt", tool.PIX_FMT[depth], "-f", "rawvideo", "-"]
    return subprocess.run(cmd, shell=False, check=True, capture_output=True).stdout


def raw_input(src, depth, tmp):
    """the frames apply.py feeds the pipe: its own decode command, taken from a dry run."""
    dry = apply(src, "selftest-invert", "--depth", str(depth), "--lossless", "--dry-run", "--json", tmp=tmp)
    decode = shlex.split(json.loads(dry.stdout)["commands"][0])
    return subprocess.run(decode, shell=False, check=True, capture_output=True).stdout


def facts(**over):
    base = {"width": WIDTH, "height": HEIGHT, "rotation": 0, "rate": Fraction(FPS), "nb_frames": FRAMES, "vfr": False}
    base.update(over)
    return base


# ----------------------------------------------------------------------------
# no tools needed
# ----------------------------------------------------------------------------

class ArgumentTests(unittest.TestCase):
    def test_a_command_line_is_split_like_a_shell_would(self):
        self.assertEqual(tool.split_command('prog --a "b c" d', windows=False), ["prog", "--a", "b c", "d"])
        self.assertEqual(tool.split_command("prog 'b c' \"d e\"", windows=False), ["prog", "b c", "d e"])
        self.assertEqual(tool.split_command(r'"C:\Program Files\tool\tool.exe" --size 160x96 "D:\clips\a b.raw"',
                                            windows=True),
                         [r"C:\Program Files\tool\tool.exe", "--size", "160x96", r"D:\clips\a b.raw"])
        self.assertEqual(tool.split_command(r"C:\tools\stage.exe -x", windows=True), [r"C:\tools\stage.exe", "-x"])
        self.assertEqual(tool.split_command("prog '' x", windows=True), ["prog", "", "x"])
        self.assertEqual(tool.split_command("   ", windows=False), [])
        for windows in (False, True):
            with self.assertRaises(ValueError):
                tool.split_command('prog "unclosed', windows=windows)

    def test_stage_and_timeout_arguments(self):
        self.assertEqual(tool.stage_arg("sapphire stage apply x"), "sapphire stage apply x")
        for bad in ("", "   ", 'prog "open'):
            with self.assertRaises(argparse.ArgumentTypeError):
                tool.stage_arg(bad)
        self.assertEqual([tool.timeout_arg(t) for t in ("0", "1.5", "120")], [0.0, 1.5, 120.0])
        for bad in ("-1", "abc", "nan", "inf", ""):
            with self.assertRaises(argparse.ArgumentTypeError):
                tool.timeout_arg(bad)

    def test_the_stage_and_frames_2_flags_are_never_abbreviated(self):
        # a prefix of a frames/1 flag means what it meant before the stage flags existed, and a prefix
        # of a new flag is refused as it was then
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "missing.mp4")

            def parse(*flags):
                proc = subprocess.run([sys.executable, APPLY, missing, "retime", *flags], shell=False,
                                      capture_output=True, text=True, encoding="utf-8", errors="replace",
                                      env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
                return proc.returncode, (proc.stderr.strip().splitlines() or [""])[-1]

            for flags in (["--de", "8"], ["--p", "k=1"], ["--describe"]):
                code, last = parse(*flags)
                self.assertNotEqual(code, 2, last)
            self.assertEqual(parse("--d"), (2, "apply.py: error: ambiguous option: --d could match --depth, --dry-run"))
            for flags in (["--desc"], ["--cont", "frames/2"], ["--pre-st", "x"], ["--stage-t", "5"], ["--geo", "x"]):
                self.assertEqual(parse(*flags), (2, "apply.py: error: unrecognized arguments: " + " ".join(flags)))

    def test_the_size_rate_and_audio_flags_are_never_abbreviated(self):
        # --size, --fps and --audio came after the stage flags: their prefixes are refused as they were before
        # them, and the prefixes they share with older flags mean what they meant
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "missing.mp4")

            def parse(*flags):
                proc = subprocess.run([sys.executable, APPLY, missing, "retime", *flags], shell=False,
                                      capture_output=True, text=True, encoding="utf-8", errors="replace",
                                      env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
                return proc.returncode, (proc.stderr.strip().splitlines() or [""])[-1]

            for flags in (["--si", "8x8"], ["--s", "8x8"], ["--fp", "24"], ["--au", "copy"], ["--a", "copy"]):
                self.assertEqual(parse(*flags), (2, "apply.py: error: unrecognized arguments: " + " ".join(flags)))
            self.assertEqual(parse("--f", "24"), (2, "apply.py: error: ambiguous option: --f could match --frames, "
                                                     "--force, --ffmpeg-path, --ffprobe-path"))
            for flags in (["--fr", "5"], ["--fo"], ["--size", "8x8"], ["--fps", "24"], ["--audio", "copy"]):
                code, last = parse(*flags)
                self.assertEqual((code, last), (1, f"[error] no such file: {missing}"), flags)


class StageBuildTests(unittest.TestCase):
    """the placeholders and the order of the stages, on facts alone."""

    def args(self, pre=(), post=(), depth=16, frames=None):
        return argparse.Namespace(pre_stage=list(pre), post_stage=list(post), depth=depth, frames=frames)

    def test_every_placeholder_is_filled_in(self):
        text = cmdline(sys.executable, "x", "{width}", "{height}", "{depth}", "{fps}", "{frames}", "{pix_fmt}",
                       "{width}x{height}@{fps}")
        for depth, fmt in ((8, "rgba"), (16, "rgba64le"), (32, "gbrapf32le")):
            stages = tool.build_stages(self.args(pre=[text], depth=depth), facts(rate=Fraction(30000, 1001)))
            self.assertEqual(stages[0]["argv"][2:], ["160", "96", str(depth), "30000/1001", "12", fmt,
                                                     "160x96@30000/1001"])
            self.assertEqual(stages[0]["command"], tool.fmt_cmd(stages[0]["argv"]))

    def test_the_size_is_the_displayed_size(self):
        stages = tool.build_stages(self.args(pre=[cmdline(sys.executable, "{width}", "{height}")]),
                                   facts(rotation=90))
        self.assertEqual(stages[0]["argv"][1:], ["96", "160"])

    def test_other_braces_are_left_alone(self):
        text = cmdline(sys.executable, "-c", "print({1: 2}, {name}, {width})")
        stages = tool.build_stages(self.args(pre=[text]), facts())
        self.assertEqual(stages[0]["argv"][2], "print({1: 2}, {name}, 160)")

    def test_frames_is_the_length_the_stage_sees(self):
        text = cmdline(sys.executable, "{frames}")
        # a pre stage reads what the input decodes to, a post stage what vfx writes
        both = tool.build_stages(self.args(pre=[text], post=[text], frames=20), facts())
        self.assertEqual([(s["role"], s["argv"][1]) for s in both], [("pre", "12"), ("post", "20")])
        same = tool.build_stages(self.args(pre=[text], post=[text]), facts(nb_frames=30))
        self.assertEqual([s["argv"][1] for s in same], ["30", "30"])

    def test_frames_is_refused_when_it_is_not_known(self):
        text = cmdline(sys.executable, "{frames}")
        for unknown in (facts(nb_frames=None), facts(vfr=True)):
            for role in ("pre", "post"):
                err = io.StringIO()
                with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as caught:
                    tool.build_stages(self.args(**{role: [text]}), unknown)
                self.assertEqual(caught.exception.code, 1)
                self.assertIn("{frames}", err.getvalue())
                self.assertIn("not known", err.getvalue())
                self.assertIn(f"{role}-stage 1", err.getvalue())
        # --frames makes it known after vfx, and only there
        known = tool.build_stages(self.args(post=[text], frames=7), facts(nb_frames=None))
        self.assertEqual(known[0]["argv"][1], "7")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            tool.build_stages(self.args(pre=[text], frames=7), facts(nb_frames=None))

    def test_the_chain_starts_at_the_work_geometry(self):
        # the decode writes frames of the work geometry (work_geometry), so the first stage reads those; vfx
        # writes the same size, and --frames sets the count after it
        text = cmdline(sys.executable, "{width}", "{height}", "{fps}", "{frames}")
        work = {"width": 64, "height": 36, "rate": Fraction(60), "frames": 25, "source": None, "auto_params": [],
                "rounding": None}
        chain = {}
        stages = tool.build_stages(self.args(pre=[text], post=[text], frames=50), facts(), chain, work)
        self.assertEqual([s["argv"][1:] for s in stages], [["64", "36", "60/1", "25"], ["64", "36", "60/1", "50"]])
        self.assertEqual(chain["vfx"], {"width": 64, "height": 36, "depth": 16, "fps": Fraction(60), "frames": 25})
        self.assertEqual([s["frame_bytes"] for s in stages], [64 * 36 * 8] * 2)
        # without one the chain starts at the input's, as before, and so do the relays of a run
        self.assertEqual(tool.build_stages(self.args(pre=[text]), facts())[0]["argv"][1:], ["160", "96", "24/1", "12"])
        self.assertEqual(tool.build_stages(self.args(pre=[text]), facts(), None, tool.work_geometry(self.args(), facts())),
                         tool.build_stages(self.args(pre=[text]), facts()))
        self.assertEqual(tool.frame_size(self.args(depth=8), facts(), work), 64 * 36 * 4)
        self.assertEqual(tool.frame_size(self.args(depth=8), facts()), WIDTH * HEIGHT * 4)

    def test_a_program_that_cannot_be_found_is_refused(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as caught:
            tool.build_stages(self.args(post=["no-such-program-anywhere --x"]), facts())
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("post-stage 1: cannot find the program `no-such-program-anywhere`", err.getvalue())

    def test_the_stages_keep_the_order_given_and_sit_around_vfx(self):
        def line(tag):
            return cmdline(sys.executable, tag)

        stages = tool.build_stages(self.args(pre=[line("a"), line("b")], post=[line("c"), line("d")]), facts())
        self.assertEqual([(s["name"], s["label"], s["argv"][1]) for s in stages],
                         [("pre1", "pre-stage 1", "a"), ("pre2", "pre-stage 2", "b"),
                          ("post1", "post-stage 1", "c"), ("post2", "post-stage 2", "d")])
        chain = tool.pipe_chain(["dec"], ["vfx"], ["enc"], stages)
        self.assertEqual([e["name"] for e in chain], ["decode", "pre1", "pre2", "vfx", "post1", "post2", "encode"])
        self.assertEqual([e["argv"][-1] for e in chain],
                         ["dec", "a", "b", "vfx", "c", "d", "enc"])
        plain = tool.pipe_chain(["dec"], ["vfx"], ["enc"], [])
        self.assertEqual([(e["name"], e["argv"]) for e in plain], [("decode", ["dec"]), ("vfx", ["vfx"]), ("encode", ["enc"])])
        self.assertEqual(tool.build_stages(self.args(), facts()), [])


class Hop:
    """what diagnose reads of a relay."""

    def __init__(self, read=0, written=None, eof=True, broken=False, state="done"):
        self.read, self.written = read, read if written is None else written
        self.eof, self.broken, self.state, self.moved = eof, broken, state, 0.0


def entry(name, role=None):
    return {"name": name, "label": name, "role": role or ("pre" if name.startswith("pre") else
                                                         "post" if name.startswith("post") else name)}


FB = 100


class DiagnoseTests(unittest.TestCase):
    """the failure of a run with stages, judged from exit codes and what the relays saw."""

    CHAIN = [entry("decode"), entry("pre1"), entry("vfx"), entry("post1"), entry("encode")]

    def codes(self, **over):
        base = {"decode": 0, "pre1": 0, "vfx": 0, "post1": 0, "encode": 0}
        base.update(over)
        return base

    def hops(self, *overrides):
        hops = [Hop(read=10 * FB) for _ in range(4)]
        for index, hop in overrides:
            hops[index] = hop
        return hops

    def judge(self, codes, hops, timed_out=()):
        return tool.diagnose(self.CHAIN, codes, hops, set(timed_out), 2, FB)

    def test_a_clean_run_has_no_problems(self):
        self.assertEqual(self.judge(self.codes(), self.hops()), [])

    def test_an_exit_code_is_the_stage_failed_and_the_neighbours_that_only_suffered_come_after(self):
        # pre1 exits 3 early: decode's write to it broke, decode died of the broken pipe
        problems = self.judge(self.codes(pre1=3, decode=4294967264),
                              self.hops((0, Hop(read=5 * FB, written=3 * FB, eof=False, broken=True))))
        self.assertEqual([(p["name"], p["kind"], p["exit_code"], p["root"]) for p in problems],
                         [("pre1", "stage_failed", 3, True), ("decode", "stage_failed", 4294967264, False)])
        self.assertEqual(problems[0]["message"], "pre1 exited with code 3")

    def test_a_stage_that_stops_in_the_middle_of_a_frame(self):
        cut = Hop(read=12 * FB + 30)
        problems = self.judge(self.codes(vfx=4), self.hops((1, cut)))
        self.assertEqual([(p["name"], p["kind"], p["root"]) for p in problems],
                         [("pre1", "partial_frame", True), ("vfx", "bad_input", False)])
        self.assertIn("30 bytes after 12 whole frames", problems[0]["message"])
        # the same with a clean exit and no other process complaining
        self.assertEqual(self.judge(self.codes(), self.hops((3, Hop(read=2 * FB + 1))))[0]["kind"], "partial_frame")
        # a stage that fails and also stopped mid frame is a failed stage
        self.assertEqual(self.judge(self.codes(pre1=5), self.hops((1, cut)))[0]["kind"], "stage_failed")

    def test_a_stage_that_stops_reading_or_writes_too_few_frames(self):
        stopped = self.hops((0, Hop(read=8 * FB, written=3 * FB, eof=False, broken=True)))
        problems = self.judge(self.codes(), stopped)
        self.assertEqual((problems[0]["name"], problems[0]["kind"]), ("pre1", "broken_pipe"))
        self.assertIn("after 3 whole frames", problems[0]["message"])
        short = self.hops((1, Hop(read=1 * FB)))
        problems = self.judge(self.codes(), short)
        self.assertEqual((problems[0]["name"], problems[0]["kind"]), ("pre1", "broken_pipe"))
        self.assertIn("1 frame, it was sent 10", problems[0]["message"])
        # vfx and the encode are not held to a frame count, only the stages are
        self.assertEqual(self.judge(self.codes(), self.hops((2, Hop(read=2 * FB)))), [])

    def test_a_killed_stage_is_a_timeout_whatever_its_exit_code(self):
        problems = self.judge(self.codes(post1=1, encode=0),
                              self.hops((2, Hop(read=4 * FB, written=4 * FB, eof=False, broken=True))), ["post1"])
        self.assertEqual((problems[0]["name"], problems[0]["kind"], problems[0]["exit_code"]), ("post1", "stage_timeout", None))
        self.assertIn("2 s", problems[0]["message"])

    def test_vfx_exit_codes_keep_their_meaning(self):
        for code, kind in ((2, "usage"), (3, "unsupported"), (4, "bad_input"), (9, "stage_failed")):
            self.assertEqual(self.judge(self.codes(vfx=code), self.hops())[0]["kind"], kind)
        self.assertEqual(self.judge(self.codes(encode=1), self.hops())[0]["kind"], "stage_failed")

    def test_roots_are_listed_before_the_victims(self):
        # post1 exits 7, the write of vfx to it broke and vfx died of that: vfx comes first in the pipe
        problems = self.judge(self.codes(vfx=1, post1=7), self.hops((2, Hop(read=FB, eof=False, broken=True))))
        self.assertEqual([(p["name"], p["root"]) for p in problems], [("post1", True), ("vfx", False)])


class FakeProc:
    def __init__(self, alive=True):
        self.alive = alive

    def poll(self):
        return None if self.alive else 0


class StallTests(unittest.TestCase):
    """which process, if any, holds the pipe up."""

    CHAIN = DiagnoseTests.CHAIN

    def stalled(self, hops, timeout=10, now=100.0, alive=True):
        procs = [FakeProc(alive) for _ in self.CHAIN]
        return tool.stalled_stage(self.CHAIN, procs, hops, timeout, now)

    def hops(self, **states):
        hops = [Hop(state="reading") for _ in range(4)]
        for index, (state, moved) in states.items():
            hops[int(index[1:])].state, hops[int(index[1:])].moved = state, moved
        return hops

    def test_a_stage_that_is_not_taking_its_input_is_stalled(self):
        # decode's relay has been blocked writing to pre1 for 30 s, and pre1 writes nothing
        self.assertEqual(self.stalled(self.hops(h0=("writing", 70.0), h1=("reading", 70.0))), 1)
        # everything was handed over and nothing came back
        self.assertEqual(self.stalled(self.hops(h0=("done", 70.0), h1=("reading", 70.0))), 1)

    def test_waiting_for_a_slow_neighbour_is_not_a_stall(self):
        # pre1 sits idle because decode has nothing to offer
        self.assertIsNone(self.stalled(self.hops(h0=("reading", 0.0), h1=("reading", 0.0))))
        # post1 cannot write because the encode is slow to read
        self.assertIsNone(self.stalled(self.hops(h2=("writing", 0.0), h3=("writing", 0.0))))
        # the next process has not been given the time yet
        self.assertIsNone(self.stalled(self.hops(h0=("writing", 95.0), h1=("reading", 95.0))))

    def test_decode_vfx_and_encode_are_never_judged_and_a_dead_stage_is_not_stalled(self):
        # pre1 and post1 wait for their producers while vfx does not take what pre1 sends: only vfx is stuck
        self.assertIsNone(self.stalled(self.hops(h1=("writing", 0.0))))
        # a stage that is already gone cannot be stalled
        self.assertIsNone(self.stalled(self.hops(h0=("writing", 0.0), h1=("reading", 0.0)), alive=False))


# ----------------------------------------------------------------------------
# the relay, with python processes in place of ffmpeg and vfx
# ----------------------------------------------------------------------------

class RelayTests(unittest.TestCase):
    """run_staged_pipeline over fake processes: frames in, frames out, and no hang whatever dies."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="stages_relay_")
        with open(os.path.join(self.dir, "fake_stage.py"), "w") as f:
            f.write(FAKE_STAGE)
        self.fake = os.path.join(self.dir, "fake_stage.py")
        self.sink = os.path.join(self.dir, "sink.txt")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def py(self, code):
        return [sys.executable, "-c", code]

    def stage_argv(self, mode, **opts):
        return [sys.executable, self.fake, mode] + [f"{k}={v}" for k, v in opts.items()]

    def source(self, frames, size=FB, delay=0.0):
        return self.py("import sys, time\n"
                       f"for n in range({frames}):\n"
                       f"    sys.stdout.buffer.write(bytes([n % 256]) * {size}); sys.stdout.buffer.flush()\n"
                       f"    time.sleep({delay})")

    def counter(self):
        return self.py(f"import sys; n = len(sys.stdin.buffer.read()); open({self.sink!r}, 'w').write(str(n))")

    def chain(self, decode, pre=(), post=(), encode=None, vfx=None):
        out = [{"name": "decode", "label": "decode", "role": "decode", "argv": decode}]
        out += [{"name": f"pre{i}", "label": f"pre-stage {i}", "role": "pre", "argv": a} for i, a in enumerate(pre, 1)]
        out.append({"name": "vfx", "label": "vfx", "role": "vfx", "argv": vfx or self.py(PASS_THROUGH)})
        out += [{"name": f"post{i}", "label": f"post-stage {i}", "role": "post", "argv": a} for i, a in enumerate(post, 1)]
        out.append({"name": "encode", "label": "encode", "role": "encode", "argv": encode or self.counter()})
        return out

    def run_chain(self, chain, timeout=0, limit=90):
        """run it on a thread so that a hang fails the test instead of the suite."""
        box = {}
        worker = threading.Thread(target=lambda: box.update(result=tool.run_staged_pipeline(chain, self.dir, timeout)))
        started = time.time()
        worker.start()
        worker.join(limit)
        self.assertFalse(worker.is_alive(), "the pipeline hung")
        codes, logs, hops, timed_out = box["result"]
        return codes, logs, hops, timed_out, time.time() - started

    def sunk(self):
        with open(self.sink) as f:
            return int(f.read())

    def test_every_byte_goes_through_every_stage_and_the_counts_add_up(self):
        chain = self.chain(self.source(40), pre=[self.stage_argv("pass", fb=FB)],
                           post=[self.stage_argv("invert", fb=FB), self.stage_argv("pass", fb=FB)])
        codes, logs, hops, timed_out, _ = self.run_chain(chain)
        self.assertEqual(set(codes.values()), {0})
        self.assertEqual(timed_out, set())
        self.assertEqual(self.sunk(), 40 * FB)
        self.assertEqual([h.read for h in hops], [40 * FB] * 5)
        self.assertEqual([h.written for h in hops], [40 * FB] * 5)
        self.assertTrue(all(h.eof and not h.broken and h.state == "done" for h in hops))
        self.assertEqual(tool.diagnose(chain, codes, hops, timed_out, 0, FB), [])
        self.assertEqual(set(logs), {e["name"] for e in chain})

    def test_a_large_stream_goes_through_without_a_deadlock(self):
        chain = self.chain(self.source(300, size=20000), pre=[self.stage_argv("pass", fb=20000)])
        codes, _, hops, _, _ = self.run_chain(chain)
        self.assertEqual(set(codes.values()), {0})
        self.assertEqual(self.sunk(), 300 * 20000)

    def test_a_stage_that_dies_mid_frame_cannot_hang_the_pipe(self):
        decode = self.py("import sys; sys.stdout.buffer.write(b'x' * 8000000)")
        chain = self.chain(decode, pre=[self.stage_argv("fail", code=3, after=10)])
        codes, logs, hops, timed_out, _ = self.run_chain(chain)
        self.assertEqual(codes["pre1"], 3)
        self.assertNotEqual(codes["decode"], 0)
        problems = tool.diagnose(chain, codes, hops, timed_out, 0, FB)
        self.assertEqual((problems[0]["name"], problems[0]["kind"], problems[0]["exit_code"]), ("pre1", "stage_failed", 3))
        with open(logs["pre1"]) as f:
            self.assertIn("boom: cannot continue", f.read())

    def test_a_downstream_stage_that_dies_stops_the_producers_too(self):
        decode = self.py("import sys; sys.stdout.buffer.write(b'x' * 8000000)")
        chain = self.chain(decode, post=[self.stage_argv("fail", code=9, after=10)])
        codes, _, hops, timed_out, _ = self.run_chain(chain)
        self.assertEqual(codes["post1"], 9)
        problems = tool.diagnose(chain, codes, hops, timed_out, 0, FB)
        self.assertEqual((problems[0]["name"], problems[0]["kind"]), ("post1", "stage_failed"))

    def test_a_stage_that_never_reads_is_killed_after_the_timeout(self):
        beat = os.path.join(self.dir, "beat")
        decode = self.py("import sys; sys.stdout.buffer.write(b'x' * 8000000)")
        chain = self.chain(decode, pre=[self.stage_argv("hang", beat=beat)])
        codes, _, hops, timed_out, seconds = self.run_chain(chain, timeout=1)
        self.assertEqual(timed_out, {"pre1"})
        self.assertLess(seconds, 30)
        problems = tool.diagnose(chain, codes, hops, timed_out, 1, FB)
        self.assertEqual((problems[0]["name"], problems[0]["kind"], problems[0]["exit_code"]), ("pre1", "stage_timeout", None))
        self.assertEqual(self.silent_after(beat), 0, "the hung stage was not killed")

    def test_a_killed_stage_takes_the_children_it_started_with_it(self):
        """a launcher that is killed alone would leave its child holding the pipes, and the run would never end."""
        beat = os.path.join(self.dir, "beat")
        decode = self.py("import sys; sys.stdout.buffer.write(b'x' * 8000000)")
        chain = self.chain(decode, pre=[self.stage_argv("spawn", beat=beat)])
        _, _, _, timed_out, seconds = self.run_chain(chain, timeout=1)
        self.assertEqual(timed_out, {"pre1"})
        self.assertLess(seconds, 30)
        self.assertEqual(self.silent_after(beat), 0, "the child of the killed stage is still running")

    def test_a_stage_that_took_everything_and_went_quiet_is_killed(self):
        chain = self.chain(self.source(3), post=[self.stage_argv("hold", beat=os.path.join(self.dir, "beat"))])
        codes, _, hops, timed_out, seconds = self.run_chain(chain, timeout=1)
        self.assertEqual(timed_out, {"post1"})
        self.assertLess(seconds, 30)

    def test_a_slow_neighbour_is_not_blamed_and_a_steady_stage_survives(self):
        # the producer needs 0.5 s a frame, 4 s in all: pre1 waits for it for longer than the 2 s timeout
        chain = self.chain(self.source(8, delay=0.5), pre=[self.stage_argv("pass", fb=FB)])
        codes, _, hops, timed_out, _ = self.run_chain(chain, timeout=2)
        self.assertEqual((set(codes.values()), timed_out), ({0}, set()))
        # and a stage that takes 0.5 s a frame, 4 s in all, keeps moving bytes
        chain = self.chain(self.source(8), pre=[self.stage_argv("slow", fb=FB, delay=0.5)])
        codes, _, hops, timed_out, _ = self.run_chain(chain, timeout=2)
        self.assertEqual((set(codes.values()), timed_out), ({0}, set()))
        self.assertEqual(self.sunk(), 8 * FB)

    def test_a_timeout_of_zero_never_kills(self):
        chain = self.chain(self.source(4), pre=[self.stage_argv("late", fb=FB, delay=2.5)])
        codes, _, _, timed_out, _ = self.run_chain(chain, timeout=0)
        self.assertEqual((set(codes.values()), timed_out), ({0}, set()))
        self.assertEqual(self.sunk(), 4 * FB)

    def test_a_program_that_will_not_start_is_reported(self):
        chain = self.chain(self.source(1), pre=[[os.path.join(self.dir, "not-a-program.exe")]])
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as caught:
            tool.run_staged_pipeline(chain, self.dir, 0)
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("cannot start pre-stage 1", err.getvalue())

    def silent_after(self, beat):
        """bytes the heartbeat of a stage added after the run ended: 0 when the process is gone."""
        size = os.path.getsize(beat)
        time.sleep(0.5)
        return os.path.getsize(beat) - size


# ----------------------------------------------------------------------------
# apply.py end to end
# ----------------------------------------------------------------------------

class PipelineTests(unittest.TestCase):
    """apply.py with stages, over a real clip, ffmpeg and vfx."""

    @classmethod
    def setUpClass(cls):
        global TMP, CLIP, VFX, FAKE
        try:
            tool.require_ffmpeg()
            VFX = tool.find_vfx(None)
        except SystemExit:
            reason = "needs ffmpeg, ffprobe and a built vfx (cargo build --release -p fx-cli, or VFX_PATH)"
            print(f"test_stages: skipping the pipeline tests: {reason}", file=sys.stderr)
            raise unittest.SkipTest(reason)
        TMP = tempfile.mkdtemp(prefix="stages_test_")
        FAKE = os.path.join(TMP, "fake_stage.py")
        with open(FAKE, "w") as f:
            f.write(FAKE_STAGE)
        CLIP = os.path.join(TMP, "clip.mp4")
        make_clip(CLIP)
        listing = json.loads(subprocess.run([VFX, "list", "--json", "--all"], shell=False, capture_output=True,
                                            text=True).stdout)
        cls.effects = {e["id"]: e["kind"] for e in listing}
        if "selftest-invert" not in cls.effects:
            raise unittest.SkipTest("this vfx has no selftest-invert effect (build it from the current workspace)")

    @classmethod
    def tearDownClass(cls):
        if TMP:
            shutil.rmtree(TMP, ignore_errors=True)

    def setUp(self):
        self.dir = tempfile.mkdtemp(dir=TMP)
        self.src = os.path.join(self.dir, "in.mp4")
        shutil.copy(CLIP, self.src)
        self.src_bytes = slurp(self.src)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.dir, name)

    def run_apply(self, *args, effect="selftest-invert"):
        return apply(self.src, effect, "--depth", "8", "--lossless", "--param", "invert_alpha=1", *args, tmp=self.dir)

    def assertClean(self):
        """no partial file and no scratch folder left behind, and the input is as it was."""
        leftovers = [n for n in os.listdir(self.dir) if n.startswith(("apply_", "vfx-spool-"))]
        leftovers += [n for _, _, fs in os.walk(self.dir) for n in fs if ".partial-" in n]
        self.assertEqual(leftovers, [])
        self.assertEqual(slurp(self.src), self.src_bytes)

    def assertFailed(self, proc, kind, stage, text, exit_code=None):
        """a staged --json run failed with this error object; the exit code of apply.py stays 1."""
        self.assertEqual(proc.returncode, 1, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertFalse(data["ok"])
        self.assertEqual(data["contract"], "frames/1")
        error = data["error"]
        self.assertEqual(set(error), {"code", "stage", "exit_code", "message", "stderr_tail"})
        self.assertEqual((error["code"], error["stage"], error["exit_code"]), (kind, stage, exit_code), error)
        self.assertIn(text, error["message"])
        self.assertNotIn("\n", error["message"])
        self.assertIn(text, proc.stderr)
        return error

    # ---- the plan ------------------------------------------------------------------

    def test_a_dry_run_lists_the_stages_in_pipe_order_and_writes_nothing(self):
        before = sorted(os.listdir(self.dir))
        a, b, c = (stage("pass", fb=n) for n in (1, 2, 3))
        proc = apply(self.src, "selftest-invert", "--dry-run", "--depth", "8", "--pre-stage", a, "--post-stage", c,
                     "--pre-stage", b, tmp=self.dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(sorted(os.listdir(self.dir)), before)
        cmds = [line for line in proc.stdout.splitlines() if line.startswith("[cmd]")]
        self.assertEqual(len(cmds), 6, proc.stdout)
        # decode, pre 1, pre 2, vfx, post 1, encode
        self.assertIn("-pix_fmt rgba -f rawvideo -", cmds[0])
        self.assertIn("fb=1", cmds[1])
        self.assertIn("fb=2", cmds[2])
        self.assertIn(" apply selftest-invert", cmds[3])
        self.assertIn("fb=3", cmds[4])
        self.assertIn("-i -", cmds[5])
        self.assertIn("decode | pre-stage 1 | pre-stage 2 | vfx | post-stage 1 | encode", proc.stdout)
        self.assertIn("[dry-run] nothing was written", proc.stdout)

    def test_the_json_result_names_the_contract_and_the_stages(self):
        stages = [stage("pass", fb="{width}"), stage("invert", fb="{frames}")]
        proc = apply(self.src, "selftest-invert", "--dry-run", "--json", "--depth", "16", "--pre-stage", stages[0],
                     "--post-stage", stages[1], tmp=self.dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertEqual(data["contract"], "frames/1")
        self.assertEqual([(s["role"], s["name"], s["exit_code"], s["stderr_tail"]) for s in data["stages"]],
                         [("pre", "pre1", None, []), ("post", "post1", None, [])])
        self.assertTrue(data["stages"][0]["command"].endswith(" fb=160"), data["stages"][0]["command"])
        self.assertTrue(data["stages"][1]["command"].endswith(" fb=12"), data["stages"][1]["command"])
        self.assertEqual(len(data["commands"]), 5)
        self.assertEqual(data["commands"][1], data["stages"][0]["command"])
        self.assertEqual(data["commands"][3], data["stages"][1]["command"])
        self.assertEqual(data["outputs"], [])

    def test_without_stages_the_result_has_no_contract_and_no_stages(self):
        for flags in ([], ["--stage-timeout", "5"]):
            proc = apply(self.src, "selftest-invert", "--dry-run", "--json", *flags, tmp=self.dir)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            data = json.loads(proc.stdout)
            self.assertNotIn("contract", data)
            self.assertNotIn("stages", data)
            self.assertEqual(len(data["commands"]), 3)
        text = apply(self.src, "selftest-invert", "--dry-run", tmp=self.dir).stdout
        self.assertIn("decode | vfx | encode", text)
        self.assertEqual(text.count("[cmd]"), 3)

    def test_the_placeholders_reach_the_stage(self):
        pre, post = self.path("pre.json"), self.path("post.json")
        args = ("{width}", "{height}", "{depth}", "{fps}", "{frames}", "{pix_fmt}")
        out = self.path("out.mkv")
        proc = apply(self.src, "selftest-invert", "--depth", "16", "--lossless", "-o", out,
                     "--pre-stage", stage("record", pre, *args), "--post-stage", stage("record", post, *args),
                     tmp=self.dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        want = ["160", "96", "16", "24/1", "12", "rgba64le"]
        for record in (pre, post):
            with open(record) as f:
                self.assertEqual(json.load(f), want)
        self.assertClean()

    def test_after_a_temporal_effect_frames_is_the_length_of_its_output(self):
        if self.effects.get("selftest-shift") != "temporal":
            self.skipTest("this vfx has no selftest-shift effect")
        pre, post = self.path("pre.json"), self.path("post.json")
        out = self.path("long.mkv")
        proc = apply(self.src, "selftest-shift", "--depth", "8", "--lossless", "--frames", "20", "-o", out,
                     "--pre-stage", stage("record", pre, "{frames}"), "--post-stage", stage("record", post, "{frames}"),
                     tmp=self.dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(pre) as f:
            self.assertEqual(json.load(f), ["12"])
        with open(post) as f:
            self.assertEqual(json.load(f), ["20"])
        self.assertEqual(len(raw_output(out, 8)), 20 * FRAME_BYTES[8])

    # ---- frames through the stages -----------------------------------------------------

    def test_vfx_invert_and_a_fake_invert_stage_give_the_input_back(self):
        """the conformance case: running selftest-invert twice, once as vfx and once as a stage in front
        of it or behind it, returns the input frames bit for bit."""
        for depth in (8, 16):
            want = raw_input(self.src, depth, self.dir)
            self.assertEqual(len(want), FRAMES * FRAME_BYTES[depth])
            for flag in ("--pre-stage", "--post-stage"):
                with self.subTest(depth=depth, flag=flag):
                    out = self.path(f"back{depth}{flag[2:6]}.mkv")
                    proc = apply(self.src, "selftest-invert", "--param", "invert_alpha=1", "--depth", str(depth),
                                 "--lossless", flag, stage("invert", fb=FRAME_BYTES[depth]), "-o", out, "--json",
                                 tmp=self.dir)
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    self.assertEqual(raw_output(out, depth), want, "the frames did not come back")
                    self.assertEqual(json.loads(proc.stdout)["stages"][0]["exit_code"], 0)
                    self.assertClean()
            # and one stage on each side of vfx inverts three times
            out = self.path(f"three{depth}.mkv")
            proc = apply(self.src, "selftest-invert", "--param", "invert_alpha=1", "--depth", str(depth), "--lossless",
                         "--pre-stage", stage("invert", fb=FRAME_BYTES[depth]),
                         "--post-stage", stage("invert", fb=FRAME_BYTES[depth]), "-o", out, tmp=self.dir)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(raw_output(out, depth), want.translate(bytes(255 - i for i in range(256))))

    def test_the_stages_run_in_the_order_given_around_vfx(self):
        """inc and dbl do not commute, and neither does either with the invert of vfx."""
        invert = bytes(255 - i for i in range(256))
        inc = bytes((i + 1) % 256 for i in range(256))
        dbl = bytes((i * 2) % 256 for i in range(256))
        want = raw_input(self.src, 8, self.dir)
        expected = want.translate(inc).translate(dbl).translate(invert).translate(inc)
        swapped = want.translate(dbl).translate(inc).translate(invert).translate(inc)
        moved = want.translate(inc).translate(dbl).translate(inc).translate(invert)
        self.assertEqual(len({expected, swapped, moved}), 3, "the stages would not tell the orders apart")
        fb = FRAME_BYTES[8]
        out = self.path("order.mkv")
        proc = apply(self.src, "selftest-invert", "--param", "invert_alpha=1", "--depth", "8", "--lossless",
                     "--pre-stage", stage("inc", fb=fb), "--pre-stage", stage("dbl", fb=fb),
                     "--post-stage", stage("inc", fb=fb), "-o", out, tmp=self.dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(raw_output(out, 8), expected)

    def test_the_lossy_default_output_works_with_a_stage_too(self):
        out = self.path("lossy.mp4")
        proc = apply(self.src, "selftest-invert", "--depth", "8", "--pre-stage", stage("pass", fb=FRAME_BYTES[8]),
                     "-o", out, tmp=self.dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        info = tool.probe(out)
        self.assertEqual((tool.display_size(info), info["vcodec"], info["pix_fmt"], info["has_audio"]),
                         ((WIDTH, HEIGHT), "h264", "yuv420p", True))
        self.assertClean()

    # ---- a stage that fails ----------------------------------------------------------------

    def test_a_stage_that_exits_with_a_code_fails_the_render_the_way_vfx_does(self):
        old = self.path("out.mkv")
        with open(old, "wb") as f:
            f.write(b"keep me")
        for flag, name in (("--pre-stage", "pre-stage 1"), ("--post-stage", "post-stage 1")):
            with self.subTest(flag=flag):
                proc = self.run_apply(flag, stage("fail", code=3, fb=FRAME_BYTES[8]), "-o", old, "--force")
                self.assertEqual(proc.returncode, 1)
                self.assertIn("[error] the pipeline failed:", proc.stderr)
                self.assertIn(f"{name} exited with code 3", proc.stderr)
                self.assertIn("  boom: cannot continue", proc.stderr)
                self.assertIn("  second line", proc.stderr)
                self.assertEqual(slurp(old), b"keep me")
                self.assertClean()
        fresh = self.path("fresh.mkv")
        self.assertEqual(self.run_apply("--pre-stage", stage("fail", code=3), "-o", fresh).returncode, 1)
        self.assertFalse(os.path.exists(fresh))
        self.assertClean()

    def test_the_json_error_object_names_the_stage_and_its_exit_code(self):
        for flag, name in (("--pre-stage", "pre1"), ("--post-stage", "post1")):
            with self.subTest(flag=flag):
                proc = self.run_apply(flag, stage("fail", code=7), "--json", "-o", self.path("x.mkv"))
                error = self.assertFailed(proc, "stage_failed", name, "exited with code 7", exit_code=7)
                self.assertEqual(error["stderr_tail"], ["boom: cannot continue", "second line"])
                self.assertFalse(os.path.exists(self.path("x.mkv")))
                self.assertClean()

    def test_a_stage_that_fails_after_writing_every_frame_still_fails(self):
        proc = self.run_apply("--post-stage", stage("failend", code=5, fb=FRAME_BYTES[8]), "--json", "-o", self.path("x.mkv"))
        error = self.assertFailed(proc, "stage_failed", "post1", "exited with code 5", exit_code=5)
        self.assertEqual(error["stderr_tail"], ["late failure"])
        self.assertClean()

    def test_the_second_stage_of_a_kind_is_the_one_named(self):
        proc = self.run_apply("--pre-stage", stage("pass", fb=FRAME_BYTES[8]), "--pre-stage", stage("fail", code=4),
                              "--json", "-o", self.path("x.mkv"))
        self.assertFailed(proc, "stage_failed", "pre2", "pre-stage 2 exited with code 4", exit_code=4)

    # ---- a stage that stops mid frame, stops reading, or hangs ---------------------------

    def test_a_stage_that_ends_in_the_middle_of_a_frame_is_detected(self):
        fb = FRAME_BYTES[8]
        for flag, name in (("--pre-stage", "pre1"), ("--post-stage", "post1")):
            for mode, opts, cut in (("extra", {"bytes": 100}, "100 bytes after 12 whole frames"),
                                    ("cut", {"bytes": 1000}, f"{fb - 1000} bytes after 11 whole frames")):
                with self.subTest(flag=flag, mode=mode):
                    proc = self.run_apply(flag, stage(mode, fb=fb, **opts), "--json", "-o", self.path("x.mkv"))
                    error = self.assertFailed(proc, "partial_frame", name, "ended in the middle of a frame", exit_code=0)
                    self.assertIn(cut, error["message"])
                    self.assertFalse(os.path.exists(self.path("x.mkv")))
                    self.assertClean()

    def test_a_stage_that_closes_its_output_early_is_a_broken_pipe(self):
        fb = FRAME_BYTES[8]
        for flag, name, label in (("--pre-stage", "pre1", "pre-stage 1"), ("--post-stage", "post1", "post-stage 1")):
            for mode in ("early", "early-drain"):
                with self.subTest(flag=flag, mode=mode):
                    proc = self.run_apply(flag, stage(mode, fb=fb), "--json", "-o", self.path("x.mkv"))
                    self.assertFailed(proc, "broken_pipe", name, label, exit_code=0)
                    self.assertFalse(os.path.exists(self.path("x.mkv")))
                    self.assertClean()

    def test_a_small_clip_that_fits_in_the_pipe_is_caught_by_the_frame_count(self):
        tiny = self.path("tiny.mp4")
        make_clip(tiny, size="16x8", frames=3, audio=False)
        for mode in ("early", "early-drain"):
            with self.subTest(mode=mode):
                proc = apply(tiny, "selftest-invert", "--depth", "8", "--lossless", "--json", "-o", self.path("t.mkv"),
                             "--pre-stage", stage(mode, fb=16 * 8 * 4), tmp=self.dir)
                self.assertFailed(proc, "broken_pipe", "pre1", "pre-stage 1", exit_code=0)
                self.assertClean()

    def test_a_stage_that_hangs_is_killed_and_reported(self):
        beat = self.path("beat")
        fb = FRAME_BYTES[8]
        cases = (("--pre-stage", "pre1", "hang", stage("hang", beat=beat)),
                 ("--post-stage", "post1", "hold", stage("hold", beat=beat)),
                 ("--pre-stage", "pre1", "hangafter", stage("hangafter", fb=fb, beat=beat)))
        for flag, name, mode, command in cases:
            with self.subTest(flag=flag, mode=mode):
                if os.path.exists(beat):
                    os.remove(beat)
                started = time.time()
                proc = self.run_apply(flag, command, "--stage-timeout", "1", "--json", "-o", self.path("x.mkv"))
                error = self.assertFailed(proc, "stage_timeout", name, "--stage-timeout")
                self.assertLess(time.time() - started, 60)
                self.assertFalse(os.path.exists(self.path("x.mkv")))
                size = os.path.getsize(beat)
                time.sleep(0.5)
                self.assertEqual(os.path.getsize(beat), size, "the hung stage is still running")
                self.assertClean()

    def test_the_timeout_is_read_in_seconds_and_zero_turns_it_off(self):
        fb = FRAME_BYTES[8]
        quiet = stage("late", fb=fb, delay=2.5)
        cut = self.run_apply("--pre-stage", quiet, "--stage-timeout", "1", "--json", "-o", self.path("a.mkv"))
        self.assertFailed(cut, "stage_timeout", "pre1", "1 s")
        for flags in (["--stage-timeout", "0"], ["--stage-timeout", "30"], []):
            with self.subTest(flags=flags):
                proc = self.run_apply("--pre-stage", quiet, *flags, "-o", self.path("b.mkv"), "--force")
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertClean()
        # the quiet stage passes the frames on, vfx inverts them
        self.assertEqual(raw_output(self.path("b.mkv"), 8),
                         raw_input(self.src, 8, self.dir).translate(bytes(255 - i for i in range(256))))

    def test_a_slow_but_steady_stage_survives_a_short_timeout(self):
        proc = self.run_apply("--pre-stage", stage("slow", fb=FRAME_BYTES[8], delay=0.5), "--stage-timeout", "2",
                              "-o", self.path("slow.mkv"))
        self.assertEqual(proc.returncode, 0, proc.stderr)

    # ---- failures before the pipe starts ------------------------------------------------

    def test_failures_before_the_pipe_also_carry_the_error_object_when_stages_are_given(self):
        ok = stage("pass", fb=FRAME_BYTES[8])
        out = self.path("never.mkv")
        # vfx's own refusal keeps vfx's exit code (3: unsupported)
        cases = [
            ((self.src, "no-such-effect"), "unsupported", "vfx", None, "unknown effect"),
            ((self.src, "selftest-invert", "--param", "no_such_param=1"), "unsupported", "vfx", 3, "no_such_param"),
            ((self.path("missing.mp4"), "selftest-invert"), "bad_input", None, None, "no such file"),
            ((self.src, "selftest-invert", "--depth", "32", "--lossless"), "usage", None, None, "--lossless needs"),
            ((self.src, "selftest-invert", "--frames", "5"), "usage", None, None, "frame effect"),
            ((self.src, "selftest-invert", "--post-stage", "no-such-program-anywhere"), "tool_missing", "post1", None,
             "cannot find the program"),
        ]
        for args, kind, name, code, text in cases:
            with self.subTest(args=args[1:]):
                proc = apply(*args, "--pre-stage", ok, "--json", "-o", out, tmp=self.dir)
                self.assertFailed(proc, kind, name, text, exit_code=code)
                self.assertFalse(os.path.exists(out))
        self.assertClean()

    def test_an_unknown_frame_count_is_refused_before_any_work(self):
        mkv = self.path("clip.mkv")
        make_clip(mkv)
        if tool.probe(mkv)["nb_frames"] is not None:
            self.skipTest("this ffmpeg writes a frame count into mkv")
        before = sorted(os.listdir(self.dir))
        proc = apply(mkv, "selftest-invert", "--depth", "8", "--json", "-o", self.path("x.mp4"),
                     "--pre-stage", stage("pass", "{frames}"), tmp=self.dir)
        self.assertFailed(proc, "usage", "pre1", "{frames}")
        self.assertEqual(sorted(os.listdir(self.dir)), before)
        # a stage that does not ask for it is fine
        proc = apply(mkv, "selftest-invert", "--depth", "8", "-o", self.path("x.mp4"),
                     "--pre-stage", stage("pass", fb=FRAME_BYTES[8]), tmp=self.dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_without_stages_a_json_failure_is_still_the_plain_message(self):
        proc = apply(self.src, "no-such-effect", "--json", tmp=self.dir)
        self.assertEqual(proc.returncode, 1)
        data = json.loads(proc.stdout)
        self.assertEqual(set(data), {"ok", "error"})
        self.assertIsInstance(data["error"], str)

    def test_usage_errors_exit_2_and_carry_the_object_only_with_stages_in_json_mode(self):
        proc = apply(self.src, "selftest-invert", "--json", "--pre-stage", stage("pass"), "--stage-timeout", "soon",
                     tmp=self.dir)
        self.assertEqual(proc.returncode, 2)
        data = json.loads(proc.stdout)
        self.assertEqual((data["ok"], data["contract"], data["error"]["code"]), (False, "frames/1", "usage"))
        self.assertIn("--stage-timeout", data["error"]["message"])
        self.assertIn("usage:", proc.stderr)
        for args in (("--pre-stage", ""), ("--post-stage", 'prog "open')):
            proc = apply(self.src, "selftest-invert", "--json", *args, tmp=self.dir)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stdout)["error"]["code"], "usage")
        plain = apply(self.src, "selftest-invert", "--json", "--stage-timeout", "-3", tmp=self.dir)
        self.assertEqual((plain.returncode, plain.stdout), (2, ""))


# ----------------------------------------------------------------------------
# --contract frames/2
# ----------------------------------------------------------------------------

CONTRACTS = os.path.join(os.path.dirname(HERE), "contracts")
FFSTAGE = os.path.join(HERE, "ffstage.py")
ANSWER = "import sys; sys.stdout.write(open(sys.argv[1]).read())"


def contract_schema(name):
    with open(os.path.join(CONTRACTS, name), encoding="utf-8") as f:
        return json.load(f)


def validate(document, name):
    import handoff
    return handoff.validate(document, contract_schema(name))


class FramedUnitTests(unittest.TestCase):
    """the frames/2 tables and judgements, without running anything."""

    def test_the_exit_table_is_the_one_of_the_error_schema(self):
        errors = contract_schema("error.schema.json")
        self.assertEqual({row["code"]: row["exit"]["frames/2"] for row in errors["x-codes"]}, tool.EXIT_CODE)
        self.assertTrue(all(row["exit"]["frames/1"] in (1, 130) for row in errors["x-codes"]))
        self.assertEqual({row["exit"]: row["code"] for row in errors["x-stage-exits"] if row["exit"] in (2, 3, 4)},
                         tool.STAGE_EXIT_KIND)

    def test_the_contract_is_read_from_the_command_line_before_argparse(self):
        self.assertEqual(tool.argv_contract(["a", "--contract", "frames/2"]), "frames/2")
        self.assertEqual(tool.argv_contract(["--contract=frames/2"]), "frames/2")
        self.assertEqual(tool.argv_contract(["--contract", "frames/9"]), "frames/1")
        self.assertEqual(tool.argv_contract([]), "frames/1")

    def test_geometry_belongs_to_the_stage_just_before_it(self):
        parser = tool.ArgParser(prog="x")
        for flag in ("--pre-stage", "--post-stage"):
            parser.add_argument(flag, action=tool.StageAction, default=[])
        parser.add_argument("--geometry", action=tool.GeometryAction)
        args = parser.parse_args(["--pre-stage", "a", "--post-stage", "b", "--geometry", "declared", "--pre-stage", "c",
                                  "--geometry", "grow"])
        self.assertEqual((args.pre_stage, args.post_stage), (["a", "c"], ["b"]))
        self.assertEqual([getattr(s, "geometry", "preserve") for s in args.pre_stage + args.post_stage],
                         ["preserve", "grow", "declared"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            parser.parse_args(["--geometry", "declared"])
        self.assertEqual(caught.exception.code, 2)

    def test_every_stage_exit_is_mapped_under_frames_2_and_only_vfx_under_frames_1(self):
        judge = DiagnoseTests()
        for code, kind in ((2, "usage"), (3, "unsupported"), (4, "bad_input"), (9, "stage_failed")):
            for name in ("pre1", "post1"):
                problems = tool.diagnose(DiagnoseTests.CHAIN, judge.codes(**{name: code}), judge.hops(), set(), 2, FB,
                                         stage_exits=True)
                self.assertEqual((problems[0]["name"], problems[0]["kind"]), (name, kind))
                problems = tool.diagnose(DiagnoseTests.CHAIN, judge.codes(**{name: code}), judge.hops(), set(), 2, FB,
                                         stage_exits=False)
                self.assertEqual(problems[0]["kind"], "stage_failed")
        # ffmpeg is not a stage: its exits keep stage_failed
        problems = tool.diagnose(DiagnoseTests.CHAIN, judge.codes(encode=3), judge.hops(), set(), 2, FB, stage_exits=True)
        self.assertEqual(problems[0]["kind"], "stage_failed")

    def test_a_declared_stage_owes_the_frames_of_its_answer_on_relays_of_its_size(self):
        chain = [dict(e) for e in DiagnoseTests.CHAIN]
        chain[3].update(geometry="declared", answer={"frames": 5})
        sizes = [FB, FB, FB, 40]
        clean = [Hop(read=10 * FB), Hop(read=10 * FB), Hop(read=10 * FB), Hop(read=5 * 40)]
        codes = DiagnoseTests().codes()
        self.assertEqual(tool.diagnose(chain, codes, clean, set(), 2, sizes), [])
        short = clean[:3] + [Hop(read=4 * 40)]
        problems = tool.diagnose(chain, codes, short, set(), 2, sizes)
        self.assertEqual((problems[0]["name"], problems[0]["kind"]), ("post1", "broken_pipe"))
        self.assertIn("declared 5", problems[0]["message"])
        cut = clean[:3] + [Hop(read=5 * 40 + 3)]
        self.assertEqual(tool.diagnose(chain, codes, cut, set(), 2, sizes)[0]["kind"], "partial_frame")

    def test_decode_and_encode_join_the_stall_watch_only_under_frames_2(self):
        stall = StallTests()
        procs = [FakeProc() for _ in DiagnoseTests.CHAIN]

        def judge(hops, ends):
            return tool.stalled_stage(DiagnoseTests.CHAIN, procs, hops, 10, 100.0, ends=ends)
        # the decode's relay has waited 100 s for its output
        quiet_decode = stall.hops(h0=("reading", 0.0), h1=("reading", 100.0), h2=("reading", 100.0), h3=("reading", 100.0))
        self.assertEqual(judge(quiet_decode, True), 0)
        self.assertIsNone(judge(quiet_decode, False))
        # the encode has refused what its relay offers for 100 s
        full_encode = stall.hops(h0=("writing", 100.0), h1=("reading", 100.0), h2=("reading", 100.0), h3=("writing", 0.0))
        self.assertEqual(judge(full_encode, True), 4)
        # an encode that took everything and is finishing its file, and vfx that does not take its input, are not judged
        self.assertIsNone(judge(stall.hops(h0=("done", 100.0), h1=("done", 100.0), h2=("done", 100.0), h3=("done", 0.0)), True))
        self.assertIsNone(judge(stall.hops(h0=("done", 100.0), h1=("writing", 0.0), h2=("reading", 100.0),
                                           h3=("reading", 100.0)), True))

    def test_the_report_has_every_process_with_role_and_index(self):
        stages = [{"role": r, "index": i, "name": f"{r}{i}", "label": f"{r}-stage {i}", "argv": [f"{r}{i}"]}
                  for r, i in (("pre", 1), ("pre", 2), ("post", 1))]
        stages[2]["answer"] = {"width": 8, "height": 4, "depth": 8, "fps": Fraction(24), "frames": None}
        report = tool.chain_report(tool.pipe_chain(["d"], ["v"], ["e"], stages), ["ffprobe", "out"])
        self.assertEqual([(s["role"], s["index"], s["name"]) for s in report],
                         [("decode", 1, "decode"), ("pre", 1, "pre1"), ("pre", 2, "pre2"), ("effect", 1, "vfx"),
                          ("post", 1, "post1"), ("encode", 1, "encode"), ("verify", 1, "verify")])
        self.assertEqual(report[4]["geometry"], {"width": 8, "height": 4, "depth": 8, "fps": "24/1", "frames": None})
        self.assertEqual(validate({"ok": True, "contract": "frames/2", "stages": report}, "run-report.schema.json"), [])


class FramedPipelineTests(unittest.TestCase):
    """apply.py --contract frames/2 over a real clip, ffmpeg, vfx and tools/ffstage.py."""

    setUpClass = PipelineTests.__dict__["setUpClass"]
    tearDownClass = PipelineTests.__dict__["tearDownClass"]
    setUp, tearDown, path, assertClean = PipelineTests.setUp, PipelineTests.tearDown, PipelineTests.path, \
        PipelineTests.assertClean

    def run_apply(self, *args, effect="selftest-invert"):
        return apply(self.src, effect, "--contract", "frames/2", "--depth", "8", "--lossless", "--param",
                     "invert_alpha=1", *args, tmp=self.dir)

    def assertFramed(self, proc, kind, stage, text, exit_code=None):
        """a frames/2 --json run failed with this error object and the exit status of its code."""
        self.assertEqual(proc.returncode, tool.EXIT_CODE[kind], proc.stderr)
        data = json.loads(proc.stdout)
        self.assertEqual(validate(data, "error.schema.json"), [])
        self.assertEqual(data["contract"], "frames/2")
        error = data["error"]
        self.assertEqual((error["code"], error["stage"], error["exit_code"]), (kind, stage, exit_code), error)
        self.assertIn(text, error["message"])
        self.assertIn(f"error: {kind}: ", proc.stderr)
        return error

    def ffstage(self, graph):
        return cmdline(sys.executable, FFSTAGE, "apply", "ffmpeg-filter", "--width", "{width}", "--height", "{height}",
                       "--depth", "{depth}", "--fps", "{fps}", "--param", "graph=" + graph,
                       "--param", "expect=declared")

    def answer(self, **over):
        doc = {"ok": True, "contract": "frames/2", "stage": "fixture-a", "width": WIDTH, "height": HEIGHT,
               "depth": 8, "fps": f"{FPS}/1", "frames": FRAMES}
        doc.update(over)
        path = self.path("answer.json")
        with open(path, "w") as f:
            f.write(json.dumps({k: v for k, v in doc.items() if v != "drop"}))
        return cmdline(sys.executable, "-c", ANSWER, path)

    def test_a_dry_run_reports_every_process_with_role_index_and_geometry(self):
        for flags, roles in (([], ["decode", "effect", "encode", "verify"]),
                             (["--pre-stage", stage("pass", fb=1), "--post-stage", stage("pass", fb=2)],
                              ["decode", "pre", "effect", "post", "encode", "verify"])):
            proc = self.run_apply("--dry-run", "--json", *flags)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            data = json.loads(proc.stdout)
            self.assertEqual(validate(data, "run-report.schema.json"), [])
            self.assertEqual((data["ok"], data["contract"]), (True, "frames/2"))
            self.assertEqual([s["role"] for s in data["stages"]], roles)
            self.assertTrue(all(s["index"] == 1 and s["exit_code"] is None for s in data["stages"]))
            self.assertIsInstance(data["stages"][0]["command"], list)
            self.assertEqual(data["geometry"], {"width": WIDTH, "height": HEIGHT, "depth": 8, "fps": "24/1",
                                                "frames": FRAMES})
        self.assertClean()

    def test_invert_twice_gives_the_input_back_with_and_without_a_stage(self):
        want = raw_input(self.src, 8, self.dir)
        inverted = want.translate(bytes(255 - i for i in range(256)))
        for flags, expected in (([], inverted), (["--pre-stage", stage("invert", fb=FRAME_BYTES[8])], want)):
            out = self.path("out.mkv")
            proc = self.run_apply("--json", "-o", out, "--force", *flags)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            data = json.loads(proc.stdout)
            self.assertEqual(validate(data, "run-report.schema.json"), [])
            self.assertEqual([s["exit_code"] for s in data["stages"]], [0] * len(data["stages"]))
            self.assertEqual(raw_output(out, 8), expected)
            self.assertClean()

    def test_refusals_exit_with_the_code_of_their_kind(self):
        taken = self.path("taken.mkv")
        with open(taken, "w") as f:
            f.write("keep")
        cases = [((self.path("missing.mp4"), "selftest-invert"), "bad_input", None, "no such file"),
                 ((self.src, "no-such-effect"), "unsupported", "vfx", "unknown effect"),
                 ((self.src, "selftest-invert", "-o", taken), "usage", None, "already exists"),
                 ((self.src, "selftest-invert", "--depth", "32", "--lossless"), "usage", None, "--lossless needs"),
                 ((self.src, "selftest-invert", "--frames", "5"), "usage", None, "frame effect"),
                 ((self.src, "selftest-invert", "--post-stage", "no-such-program-anywhere"), "tool_missing", "post1",
                  "cannot find the program"),
                 ((self.src, "selftest-invert", "--post-stage", stage("pass"), "--geometry", "grow"), "unsupported", None,
                  "unknown stage geometry")]
        for args, kind, name, text in cases:
            with self.subTest(args=args[1:]):
                proc = apply(*args, "--contract", "frames/2", "--json", tmp=self.dir)
                self.assertFramed(proc, kind, name, text)
        # the new flags need frames/2; frames/1 keeps its exit 1
        for flags in (("--describe",), ("--post-stage", stage("pass"), "--geometry", "declared")):
            proc = apply(self.src, "selftest-invert", *flags, tmp=self.dir)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertIn("needs --contract frames/2", proc.stderr)
        usage = apply(self.src, "selftest-invert", "--contract", "frames/2", "--json", "--stage-timeout", "soon",
                      tmp=self.dir)
        self.assertEqual(usage.returncode, 2)
        self.assertEqual(json.loads(usage.stdout)["contract"], "frames/2")
        self.assertEqual(slurp(taken), b"keep")

    def test_every_stage_exit_maps_to_its_code(self):
        for code, kind in ((2, "usage"), (3, "unsupported"), (4, "bad_input"), (7, "stage_failed")):
            with self.subTest(code=code):
                proc = self.run_apply("--pre-stage", stage("fail", code=code), "--json", "-o", self.path("x.mkv"))
                self.assertFramed(proc, kind, "pre1", "pre-stage 1 exited with code", exit_code=code)
                self.assertClean()
        old = apply(self.src, "selftest-invert", "--depth", "8", "--lossless", "--pre-stage", stage("fail", code=3),
                    "--json", "-o", self.path("y.mkv"), tmp=self.dir)
        self.assertEqual((old.returncode, json.loads(old.stdout)["error"]["code"]), (1, "stage_failed"))

    def test_a_partial_frame_and_a_broken_pipe_exit_4_and_a_hang_exits_5(self):
        fb = FRAME_BYTES[8]
        self.assertFramed(self.run_apply("--pre-stage", stage("cut", bytes=10), "--json", "-o", self.path("a.mkv")),
                          "partial_frame", "pre1", "middle of a frame", exit_code=0)
        self.assertFramed(self.run_apply("--post-stage", stage("early", fb=fb), "--json", "-o", self.path("b.mkv")),
                          "broken_pipe", "post1", "post-stage 1", exit_code=0)
        started = time.time()
        self.assertFramed(self.run_apply("--pre-stage", stage("late", fb=fb, delay=30), "--stage-timeout", "1", "--json",
                                         "-o", self.path("c.mkv")), "stage_timeout", "pre1", "--stage-timeout")
        self.assertLess(time.time() - started, 25)
        self.assertClean()

    def test_a_quiet_decode_or_a_full_encode_is_killed_under_frames_2(self):
        py = sys.executable
        producer = [py, "-c", "import sys; sys.stdout.buffer.write(bytes(1 << 24))"]
        quiet = [py, FAKE, "late", "delay=30", "fb=4096"]
        cases = (("decode", [quiet, [py, FAKE, "pass", "fb=4096"], [py, FAKE, "pass", "fb=4096"]]),
                 ("encode", [producer, [py, FAKE, "pass", "fb=4096"], quiet]))
        saved = tool.CONTRACT
        tool.CONTRACT = "frames/2"
        try:
            for stuck, argvs in cases:
                with self.subTest(stuck=stuck):
                    chain = [{"name": n, "label": n, "role": n, "argv": a} for n, a in zip(("decode", "vfx", "encode"), argvs)]
                    started = time.time()
                    codes, logs, hops, timed_out = tool.run_staged_pipeline(chain, self.dir, 1)
                    self.assertEqual(timed_out, {stuck})
                    self.assertLess(time.time() - started, 25)
        finally:
            tool.CONTRACT = saved

    def test_declared_geometry_reaches_vfx_the_encode_and_the_verify(self):
        crop = self.ffstage("crop=80:48:0:0")
        for flag in ("--pre-stage", "--post-stage"):
            with self.subTest(flag=flag):
                out = self.path("crop.mkv")
                proc = self.run_apply(flag, crop, "--geometry", "declared", "--json", "-o", out, "--force")
                self.assertEqual(proc.returncode, 0, proc.stderr)
                data = json.loads(proc.stdout)
                self.assertEqual(validate(data, "run-report.schema.json"), [])
                self.assertEqual(data["geometry"], {"width": 80, "height": 48, "depth": 8, "fps": "24/1", "frames": FRAMES})
                effect = next(s for s in data["stages"] if s["role"] == "effect")["command"]
                self.assertEqual(effect[effect.index("--width") + 1], "80" if flag == "--pre-stage" else str(WIDTH))
                facts_out = tool.probe(out)
                self.assertEqual((facts_out["width"], facts_out["height"]), (80, 48))
                self.assertEqual(len(raw_output(out, 8)), 80 * 48 * 4 * FRAMES)
                self.assertClean()
        proc = self.run_apply("--post-stage", crop, "--geometry", "declared", "--describe")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        answer = json.loads(proc.stdout)
        self.assertEqual(validate(answer, "describe.schema.json"), [])
        self.assertEqual((answer["stage"], answer["width"], answer["height"], answer["frames"]),
                         ("selftest-invert", 80, 48, FRAMES))
        # a declared size with an odd side encodes yuv444p at that exact size on the lossy path
        odd_out = self.path("odd.mp4")
        odd = apply(self.src, "selftest-invert", "--contract", "frames/2", "--json", "--post-stage",
                    self.ffstage("crop=81:48"), "--geometry", "declared", "-o", odd_out, tmp=self.dir)
        self.assertEqual(odd.returncode, 0, odd.stderr)
        data = json.loads(odd.stdout)
        self.assertEqual(validate(data, "run-report.schema.json"), [])
        self.assertEqual(data["geometry"], {"width": 81, "height": 48, "depth": 16, "fps": "24/1", "frames": FRAMES})
        self.assertEqual(data["encode"], {"codec": "libx264", "pix_fmt": "yuv444p"})
        self.assertIn("encoded yuv444p (the size has an odd side)", data["summary"])
        facts_out = tool.probe(odd_out)
        self.assertEqual((facts_out["width"], facts_out["height"], facts_out["pix_fmt"]), (81, 48, "yuv444p"))
        self.assertClean()

    def test_an_interrupt_is_reported_as_interrupted_with_exit_130(self):
        # no console signal: apply.py's main runs in this process and a KeyboardInterrupt is raised where
        # the pipe would start, after the scratch folder exists
        names = ("CONTRACT", "JSON_OUTPUT", "FORCE", "STAGED", "run_staged", "run_pipeline")
        saved = {name: getattr(tool, name) for name in names}
        saved_argv, saved_tmp, saved_vfx = sys.argv, tempfile.tempdir, os.environ.get("VFX_PATH")

        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt

        out_path = self.path("never.mkv")
        try:
            tool.run_staged = tool.run_pipeline = interrupt
            tempfile.tempdir = self.dir
            os.environ["VFX_PATH"] = VFX
            for flags, framed in ((["--contract", "frames/2", "--json"], True), (["--contract", "frames/2"], True),
                                  (["--json"], False)):
                with self.subTest(flags=flags):
                    sys.argv = ["apply.py", self.src, "selftest-invert", "--depth", "8", "--lossless", "-o", out_path,
                                *flags]
                    out, err = io.StringIO(), io.StringIO()
                    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                            self.assertRaises(SystemExit) as caught:
                        tool.main()
                    self.assertEqual(caught.exception.code, 130, err.getvalue())
                    if framed and "--json" in flags:
                        data = json.loads(out.getvalue())
                        self.assertEqual(validate(data, "error.schema.json"), [])
                        self.assertEqual((data["contract"], data["error"]["code"], data["error"]["stage"],
                                          data["error"]["exit_code"]), ("frames/2", "interrupted", None, None))
                    else:
                        self.assertEqual(out.getvalue(), "")
                    self.assertIn("error: interrupted: " if framed else "[interrupted]", err.getvalue())
                    self.assertFalse(os.path.exists(out_path))
                    self.assertClean()
        finally:
            for name, value in saved.items():
                setattr(tool, name, value)
            sys.argv, tempfile.tempdir = saved_argv, saved_tmp
            if saved_vfx is None:
                os.environ.pop("VFX_PATH", None)
            else:
                os.environ["VFX_PATH"] = saved_vfx

    def test_a_declared_answer_is_checked(self):
        cases = (({"fps": "12/1"}, "unsupported", "frame rate"),
                 ({"depth": 16}, "unsupported", "depth"),
                 ({"frames": "drop"}, "stage_failed", "invalid answer"),
                 ({"width": 0}, "stage_failed", "invalid answer"))
        for over, kind, text in cases:
            with self.subTest(over=over):
                proc = self.run_apply("--post-stage", self.answer(**over), "--geometry", "declared", "--json", "--dry-run")
                self.assertFramed(proc, kind, "post1", text, exit_code=0 if kind == "stage_failed" else None)
        refused = self.run_apply("--post-stage", self.ffstage("fps=12"), "--geometry", "declared", "--json", "--dry-run")
        self.assertFramed(refused, "unsupported", "post1", "--describe exited with code 3", exit_code=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
