#!/usr/bin/env python3
"""Conformance suite of the handoff convention (docs/handoff.md).

Reads contracts/conformance/cases.json and checks, level by level:

  L0  manifests: every registered provider's stage manifest (the native `stage list --json`
      and the vfx-list adapter of `vfx list --json`) and *.stage.json files, through
      tools/handoff.py
  L1  stage matrix: depths, sizes, frame counts and alpha modes through every provider of a
      modelled stage, alone and chained in both orders
  L2  wrappers end to end (--lossless, both orders, an ffmpeg filter between)
  L3  failure injection, each case under a deadline: the reference judge on the fixture
      stages, and the error exits of every stage under test
  L4  packets through tools/handoff.py in a scratch folder
  L5  frames/2 wrapper reports and error documents

A level that cannot run (a provider, peer or wrapper is missing) prints NOT RUN and makes
the exit 6. It never passes silently. --allow-absent-peer (for a CI job that checks out one
repository) makes the exit 0 only when every part that did not run waited for a peer
repository that is absent; the NOT RUN lines and the absent peers are still printed.

--bench runs the wrappers on chained stages (contracts/conformance/cases.json, bench) and
writes docs/handoff-bench.tsv; peak memory and disk come from `conformance.py meter`.

Usage:
  python -I tools/conformance.py [--quick | --full] [--levels L0,L1,...] [--json]
  python -I tools/conformance.py --stage "<command>" [--quick | --full] [--json]
  python -I tools/conformance.py --bench
  python -I tools/conformance.py meter --temp DIR [--deadline S] -- <command>
  python -I tools/conformance.py fixture list --json
  python -I tools/conformance.py fixture apply <fixture stage> --width W --height H --depth D

--stage takes the command of one stage up to and including its stage id (for example
`vfx apply selftest-invert`), or a ready string of `handoff.py stages --json` with its
{width}, {height}, {depth} and {fps} placeholders; it runs L1 and L3 on that stage only.

Exit: 0 every selected level ran and passed, 1 a check failed, 2 usage, 6 nothing failed
but something did not run. Python 3.12 standard library only; run with `python -I`.
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import platform
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
SELF = os.path.abspath(__file__)
DEFAULT_ROOT = os.path.dirname(HERE)
CASES_REL = "contracts/conformance/cases.json"
CONTRACT = "frames/2"
VERSION = "1"
IS_WINDOWS = os.name == "nt"
BIG_ENDIAN = sys.byteorder == "big"
CHUNK = 1 << 16

BYTES_PER_PIXEL = {8: 4, 16: 8, 32: 16}
FULL_SCALE = {8: 255, 16: 65535, 32: 256}
STAGE_EXIT_CODES = {2: "usage", 3: "unsupported", 4: "bad_input"}
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_RUN = 0, 1, 2, 6
LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5")
LEVEL_NAMES = {"L0": "manifests", "L1": "stage matrix", "L2": "wrappers", "L3": "failure injection",
               "L4": "packets", "L5": "frames/2"}
STAGE_LEVELS = ("L1", "L3")
MAX_TEXT_FAILURES = 12
MAX_JSON_FAILURES = 60

INV8 = bytes(255 - i for i in range(256))
UNIT32 = [k / 256.0 for k in range(255)] + [1.0]


# -- frames -------------------------------------------------------------------------------------

def frame_size(width, height, depth):
    return width * height * BYTES_PER_PIXEL[depth]


def _alpha_units(rng, count, mode, top):
    if mode == "opaque":
        return [top] * count
    if mode == "zero":
        return [0] * count
    values = [rng.randrange(top + 1) for _ in range(count)]
    if count == 1:
        values[0] = top // 2
    else:
        values[0], values[1] = 0, top
    return values


def _float_plane(units):
    plane = array("f", [u / 256.0 for u in units])
    if BIG_ENDIAN:
        plane.byteswap()
    return plane.tobytes()


def make_clip(width, height, depth, frames, alpha, seed):
    """whole frames in the contract layout with a seeded pattern. alpha is opaque, partial or
    zero; colour stays random under any alpha (straight alpha). depth 32 samples are multiples
    of 1/256 in [0, 1], where invert twice is exact."""
    rng = random.Random(seed)
    count = width * height
    top = FULL_SCALE[depth]
    out = bytearray()
    for _ in range(frames):
        if depth == 8:
            frame = bytearray(rng.randbytes(count * 4))
            frame[3::4] = bytes(_alpha_units(rng, count, alpha, top))
            out += frame
        elif depth == 16:
            frame = array("H")
            frame.frombytes(rng.randbytes(count * 8))
            frame[3::4] = array("H", _alpha_units(rng, count, alpha, top))
            if BIG_ENDIAN:
                frame.byteswap()
            out += frame.tobytes()
        else:
            for _plane in range(3):
                out += _float_plane([b if b < 255 else 256 for b in rng.randbytes(count)])
            out += _float_plane(_alpha_units(rng, count, alpha, top))
    return bytes(out)


def invert(data, width, height, depth):
    """the model of `selftest-invert`: every colour sample becomes full scale minus the sample,
    alpha is unchanged (8 and 16 bit packed R, G, B, A; 32 bit planes G, B, R, A)."""
    if depth == 8:
        buf = bytearray(data)
        for c in range(3):
            buf[c::4] = buf[c::4].translate(INV8)
        return bytes(buf)
    if depth == 16:
        samples = array("H")
        samples.frombytes(data)
        if BIG_ENDIAN:
            samples.byteswap()
        for c in range(3):
            samples[c::4] = array("H", [65535 - v for v in samples[c::4]])
        if BIG_ENDIAN:
            samples.byteswap()
        return samples.tobytes()
    samples = array("f")
    samples.frombytes(data)
    if BIG_ENDIAN:
        samples.byteswap()
    count = width * height
    for start in range(0, len(samples), 4 * count):
        for plane in range(3):
            lo = start + plane * count
            samples[lo:lo + count] = array("f", [1.0 - v for v in samples[lo:lo + count]])
    if BIG_ENDIAN:
        samples.byteswap()
    return samples.tobytes()


MODELS = {"invert": invert}


def matrix_rows(matrix, mode):
    """(depth, width, height, frames, alpha) rows: the whole product for full, a covering set for
    quick in which every depth meets every alpha mode and every size and frame count appears."""
    depths, sizes, frames, alphas = matrix["depths"], matrix["sizes"], matrix["frames"], matrix["alpha"]
    if mode == "full":
        return [(d, w, h, f, a) for d in depths for (w, h) in sizes for f in frames for a in alphas]
    rows = []
    for i, (d, a) in enumerate(itertools.product(depths, alphas)):
        w, h = sizes[i % len(sizes)]
        rows.append((d, w, h, frames[i % len(frames)], a))
    return rows


# -- processes ----------------------------------------------------------------------------------

def split_command(text):
    """argv of a command string: words split on blanks, double quotes group words and \\" is a
    quote inside them; other backslashes are kept, so Windows paths survive."""
    words, cur, quoted, have, i = [], [], False, False, 0
    while i < len(text):
        ch = text[i]
        if quoted:
            if ch == "\\" and text[i + 1:i + 2] == '"':
                cur.append('"')
                i += 2
                continue
            if ch == '"':
                quoted = False
            else:
                cur.append(ch)
        elif ch == '"':
            quoted, have = True, True
        elif ch.isspace():
            if cur or have:
                words.append("".join(cur))
                cur, have = [], False
        else:
            cur.append(ch)
        i += 1
    if quoted:
        raise ValueError("unbalanced double quote in the stage command")
    if cur or have:
        words.append("".join(cur))
    return words


def stage_argv(base, width, height, depth, fps="24/1", params=None, extra=()):
    values = {"{width}": str(width), "{height}": str(height), "{depth}": str(depth), "{fps}": fps}
    if any(key in word for word in base for key in values):
        argv = []
        for word in base:
            for key, value in values.items():
                word = word.replace(key, value)
            argv.append(word)
    else:
        argv = list(base) + ["--width", str(width), "--height", str(height), "--depth", str(depth),
                             "--fps", fps]
    for key, value in sorted((params or {}).items()):
        argv += ["--param", f"{key}={value}"]
    return argv + list(extra)


def group_kwargs():
    return {} if IS_WINDOWS else {"start_new_session": True}


def kill_tree(proc):
    """end a stage and every process it started: taskkill /T on Windows, the process group on
    POSIX (each stage starts in its own session)."""
    if IS_WINDOWS:
        taskkill = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "taskkill.exe")
        try:
            subprocess.run([taskkill, "/T", "/F", "/PID", str(proc.pid)], stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
    try:
        proc.kill()
    except OSError:
        pass


def pid_alive(pid):
    if IS_WINDOWS:
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        handle = kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            status = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(status))
            return bool(ok) and status.value == 259
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return not unreaped(pid)


def unreaped(pid):
    """whether a pid that still answers has exited and only waits to be reaped: a killed child whose
    parent is gone, under a new parent that never reaps (Linux /proc; False elsewhere)."""
    try:
        with open(f"/proc/{int(pid)}/stat", "rb") as f:
            stat = f.read()
    except OSError:
        return False
    return stat[stat.rfind(b")") + 2:][:1] in (b"Z", b"X")


def wait_gone(pids, seconds=5.0):
    """the pids still alive after waiting up to `seconds`."""
    end = time.monotonic() + seconds
    alive = [p for p in pids if pid_alive(p)]
    while alive and time.monotonic() < end:
        time.sleep(0.05)
        alive = [p for p in alive if pid_alive(p)]
    return alive


def run_stage(argv, data, frame_bytes, deadline=60.0, stall=None, read_limit=None, env=None):
    """run one stage process on `data` and judge it like a wrapper does. code is None when the
    stage wrote every frame it was sent and exited 0; otherwise stage_timeout (killed after
    `stall` seconds without a byte moving or after `deadline`), usage, unsupported, bad_input
    (stage exits 2, 3, 4), stage_failed (other exits), partial_frame (exit 0, output ends inside
    a frame), broken_pipe (exit 0, fewer whole frames than sent, or input refused) or
    tool_missing. read_limit closes the stage's output after that many bytes."""
    start = time.monotonic()
    state = {"moved": start, "pipe_closed": False}
    out, err = [], []
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=env, **group_kwargs())
    except OSError as exc:
        return {"code": "tool_missing", "exit": None, "out": b"", "err": str(exc).encode(), "seconds": 0.0,
                "killed": None, "pipe_closed": False, "stray": False}

    def feed():
        try:
            view = memoryview(data)
            for i in range(0, len(view), CHUNK):
                proc.stdin.write(view[i:i + CHUNK])
                state["moved"] = time.monotonic()
            proc.stdin.close()
        except (OSError, ValueError):
            state["pipe_closed"] = True
            try:
                proc.stdin.close()
            except (OSError, ValueError):
                pass

    def drain(stream, sink, limit, track):
        got = 0
        try:
            while True:
                chunk = stream.read1(CHUNK)
                if not chunk:
                    break
                sink.append(chunk)
                got += len(chunk)
                if track:
                    state["moved"] = time.monotonic()
                if limit is not None and got >= limit:
                    break
        except (OSError, ValueError):
            pass
        finally:
            try:
                stream.close()
            except OSError:
                pass

    threads = [threading.Thread(target=feed, daemon=True),
               threading.Thread(target=drain, args=(proc.stdout, out, read_limit, True), daemon=True),
               threading.Thread(target=drain, args=(proc.stderr, err, None, False), daemon=True)]
    for thread in threads:
        thread.start()
    killed = None
    while proc.poll() is None:
        now = time.monotonic()
        if now - start > deadline:
            killed = "deadline"
        elif stall is not None and now - state["moved"] > stall:
            killed = "stall"
        if killed:
            kill_tree(proc)
            break
        time.sleep(0.01)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        killed = killed or "deadline"
    for thread in threads:
        thread.join(timeout=10)
    stray = any(thread.is_alive() for thread in threads)
    output = b"".join(out)
    rc = proc.returncode
    if killed:
        code = "stage_timeout"
    elif rc != 0:
        code = STAGE_EXIT_CODES.get(rc, "stage_failed")
    elif frame_bytes and len(output) % frame_bytes:
        code = "partial_frame"
    elif read_limit is None and frame_bytes and (state["pipe_closed"] or
                                                 len(output) // frame_bytes < len(data) // frame_bytes):
        code = "broken_pipe"
    else:
        code = None
    return {"code": code, "exit": rc, "out": output, "err": b"".join(err), "seconds": time.monotonic() - start,
            "killed": killed, "pipe_closed": state["pipe_closed"], "stray": stray}


def run_capture(argv, timeout=120.0, env=None):
    """(exit code, stdout text, stderr text) of a short command under a deadline; the tree is
    killed when it runs past it (exit None)."""
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=env, **group_kwargs())
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        out, err = proc.communicate(timeout=15)
        return None, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
    return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


def child_pids(stderr):
    return [int(m) for m in re.findall(rb"child pid ([0-9]+)", stderr)]


def describe(result):
    tail = result["err"].decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
    return (f"code {result['code'] or 'ok'}, exit {result['exit']}, {len(result['out'])} bytes out"
            + (f", killed ({result['killed']})" if result["killed"] else "") + (f", stderr: {tail[0][:120]}"
                                                                               if tail[0] else ""))


# -- fixture stages -----------------------------------------------------------------------------

FIXTURES = {
    "fixture-copy": "copies every whole frame",
    "fixture-exit": "copies every whole frame, then exits with the status in param code",
    "fixture-truncate": "reads every frame, writes all but the last and exits 0",
    "fixture-partial": "copies every frame, then writes half a frame and exits 0",
    "fixture-close": "reads one frame, closes its input, writes that frame and exits 0",
    "fixture-hang": "starts a sleeping child that holds its output, reads its input and never exits",
}


def fixture_command():
    return [sys.executable, "-I", SELF, "fixture"]


def fixture_manifest():
    stages = []
    for sid, text in FIXTURES.items():
        params = [{"id": "code", "kind": "integer", "default": 1}] if sid == "fixture-exit" else []
        stages.append({"id": sid, "name": text, "kind": "frame", "spool": False, "geometry": "preserve",
                       "depths": [8, 16, 32], "params": params, "layers": []})
    return {"contract": CONTRACT, "provider": {"id": "fixture", "version": VERSION, "command": fixture_command()},
            "stages": stages}


def _fixture_fail(code, message, status):
    sys.stderr.write(f"error: {code}: {message}\n")
    sys.stderr.flush()
    return status


def _read_exact(count):
    parts, got = [], 0
    while got < count:
        chunk = os.read(0, min(count - got, CHUNK))
        if not chunk:
            break
        parts.append(chunk)
        got += len(chunk)
    return b"".join(parts)


def fixture_main(argv):
    """a stage provider of placeholder stages that misbehave on purpose, for L0 and L3."""
    if IS_WINDOWS:
        import msvcrt
        for fd in (0, 1):
            msvcrt.setmode(fd, os.O_BINARY)
    if argv[:1] == ["list"]:
        sys.stdout.write(json.dumps(fixture_manifest(), indent=2) + "\n")
        return 0
    if argv[:1] != ["apply"] or len(argv) < 2:
        return _fixture_fail("usage", "expected `list --json` or `apply STAGE --width W --height H --depth D`", 2)
    stage, opts, params, i = argv[1], {}, {}, 2
    if stage not in FIXTURES:
        return _fixture_fail("unsupported", f"unknown fixture stage `{stage}`", 3)
    while i < len(argv):
        word = argv[i]
        if word in ("--width", "--height", "--depth", "--fps", "--frames") and i + 1 < len(argv):
            opts[word[2:]] = argv[i + 1]
        elif word == "--param" and i + 1 < len(argv) and "=" in argv[i + 1]:
            key, value = argv[i + 1].split("=", 1)
            params[key] = value
        else:
            return _fixture_fail("usage", f"unknown option `{word}`", 2)
        i += 2
    try:
        width, height, depth = int(opts["width"]), int(opts["height"]), int(opts["depth"])
    except (KeyError, ValueError):
        return _fixture_fail("usage", "--width, --height and --depth take integers and are required", 2)
    if width < 1 or height < 1 or depth not in BYTES_PER_PIXEL:
        return _fixture_fail("usage", "--depth expects 8, 16 or 32 and the sides must be positive", 2)
    size = frame_size(width, height, depth)
    out = sys.stdout.buffer
    try:
        if stage == "fixture-hang":
            child = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(300)"],
                                     stdin=subprocess.DEVNULL, stdout=out.fileno(), stderr=subprocess.DEVNULL)
            sys.stderr.write(f"fixture-hang: child pid {child.pid}\n")
            sys.stderr.flush()
            while _read_exact(CHUNK):
                pass
            time.sleep(600)
            return 0
        if stage == "fixture-close":
            frame = _read_exact(size)
            os.close(0)
            out.write(frame)
            out.flush()
            return 0
        if stage == "fixture-truncate":
            frames = []
            while True:
                frame = _read_exact(size)
                if not frame:
                    break
                frames.append(frame)
            out.write(b"".join(frames[:-1]))
            out.flush()
            return 0
        count = 0
        while True:
            frame = _read_exact(size)
            if not frame:
                break
            if len(frame) < size:
                return _fixture_fail("bad_input", f"input ends with a partial frame ({len(frame)} of {size} bytes) "
                                     f"after {count} whole frames", 4)
            out.write(frame)
            count += 1
        if stage == "fixture-partial":
            out.write(b"\0" * max(1, size // 2))
        out.flush()
        if stage == "fixture-exit":
            try:
                return int(params.get("code", "1"))
            except ValueError:
                return _fixture_fail("usage", "param code takes an integer", 2)
        return 0
    except OSError as exc:
        # the reader is gone: point stdout at the null device so the flush at exit cannot fail
        # and turn the status into 120
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
        except OSError:
            pass
        return _fixture_fail("bad_input", f"output closed: {exc.strerror or exc}", 4)


# -- levels -------------------------------------------------------------------------------------

class Level:
    def __init__(self, key, selected):
        self.key, self.name, self.selected = key, LEVEL_NAMES[key], selected
        self.passed, self.failures, self.not_run = 0, [], []
        self.peers, self.untagged = set(), 0

    def check(self, ok, what, detail=""):
        if ok:
            self.passed += 1
        else:
            self.failures.append(f"{what}: {detail}" if detail else what)
        return ok

    def missing(self, what, peer=None):
        """record a part that did not run; `peer` names the absent peer repository it needed."""
        if what in self.not_run:
            return
        self.not_run.append(what)
        if peer:
            self.peers.add(peer)
        else:
            self.untagged += 1

    def only_absent_peers(self):
        """True when everything that did not run waited for an absent peer repository."""
        return bool(self.not_run) and self.untagged == 0 and not self.failures

    def status(self):
        if not self.selected:
            return "not selected"
        if self.failures:
            return "fail"
        if self.not_run or not self.passed:
            return "not run"
        return "pass"

    def document(self):
        reasons = list(self.not_run)
        if self.selected and not self.failures and not self.not_run and not self.passed:
            reasons.append("no check ran")
        return {"level": self.key, "name": self.name, "status": self.status(), "passed": self.passed,
                "failed": len(self.failures), "not_run": reasons, "failures": self.failures[:MAX_JSON_FAILURES]}


class Context:
    def __init__(self, root, cases, mode, deadline):
        self.root, self.cases, self.mode, self.deadline = root, cases, mode, deadline
        self.tmp = tempfile.mkdtemp(prefix="conformance_")
        self.handoff, self.handoff_why = load_handoff(root)
        self._providers = None
        self._tools = None

    def tools(self):
        if self._tools is None:
            self._tools = Tools(self)
        return self._tools

    def providers(self):
        """{tool id: {"tool", "provider", "binary", "program", "why"}} of the registry."""
        if self._providers is None:
            self._providers = find_providers(self)
        return self._providers


def load_handoff(root):
    path = os.path.join(root, "tools", "handoff.py")
    if not os.path.isfile(path):
        return None, "tools/handoff.py is not present"
    try:
        spec = importlib.util.spec_from_file_location("handoff_for_conformance", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:  # a broken control plane is reported, not raised
        return None, f"tools/handoff.py did not load: {exc}"
    return module, ""


def find_providers(ctx):
    h, found = ctx.handoff, {}
    if h is None:
        return found
    try:
        registry = h.load_registry()
        me = h.detect_self(ctx.root)
    except h.HandoffError as exc:
        return {"": {"why": f"registry: {exc.message}", "program": None}}
    for tool in registry["tools"]:
        binaries = {b["name"]: b for b in tool["binaries"]}
        for provider in tool["providers"]:
            binary = binaries[provider["binary"]]
            entry = {"tool": tool, "provider": provider, "binary": binary, "program": None, "why": "",
                     "root": None, "absent_peer": None}
            tool_root = None
            try:
                if tool["id"] == me:
                    tool_root, why = ctx.root, ""
                else:
                    tool_root, why = h.find_peer(tool["id"], ctx.root)
                program, reason = h.tool_binary(tool, binary, tool_root)
            except h.HandoffError as exc:
                program, reason, why = None, exc.message, ""
            entry["program"], entry["root"] = program, tool_root
            entry["why"] = reason + (f" ({why})" if why else "") if program is None else ""
            if program is None and tool_root is None and tool["id"] != me:
                entry["absent_peer"] = tool["id"]
            found[tool["id"]] = entry
    return found


def level0(ctx, lv):
    h = ctx.handoff
    if h is None:
        lv.missing(ctx.handoff_why)
        return
    script = os.path.join(ctx.root, "tools", "handoff.py")
    rc, out, err = run_capture([sys.executable, "-I", script, "contracts", "--json"], timeout=ctx.deadline)
    lv.check(rc == 0, "handoff.py contracts", f"exit {rc}: {err.strip()[-200:]}")
    modelled = ctx.cases["L1"]["stages"]
    for tool_id, entry in sorted(ctx.providers().items()):
        if entry["program"] is None:
            lv.missing(f"{tool_id or 'registry'} provider: {entry['why']}", peer=entry.get("absent_peer"))
            continue
        adapter = entry["provider"]["adapter"]
        what = f"{tool_id} manifest ({adapter})"
        try:
            manifest = h.provider_manifest(entry["tool"], entry["provider"], entry["binary"], entry["program"])
        except h.HandoffError as exc:
            lv.check(False, what, exc.message[:300])
            continue
        lv.check(True, what)
        if adapter == "native":
            lv.check(manifest["contract"] == CONTRACT, f"{what} contract", f"{manifest['contract']}, expected {CONTRACT}")
        else:
            lv.check(manifest["contract"] == "frames/1", f"{what} contract", manifest["contract"])
        lv.check(manifest["provider"]["id"] == tool_id, f"{what} provider id", manifest["provider"]["id"])
        stages = {s["id"]: s for s in manifest["stages"]}
        for case in modelled:
            if tool_id not in case["providers"]:
                continue
            st = stages.get(case["stage"])
            lv.check(st is not None and st["kind"] == "frame" and st["geometry"] == "preserve"
                     and not st["spool"] and set(ctx.cases["L1"]["matrix"]["depths"]) <= set(st["depths"]),
                     f"{what} lists {case['stage']} as a frame stage that preserves geometry at every depth",
                     json.dumps(st)[:200] if st else "absent")
    folder = os.path.join(ctx.tmp, "providers")
    os.makedirs(folder)
    good = os.path.join(folder, "fixture.stage.json")
    with open(good, "w", encoding="utf-8") as handle:
        json.dump(fixture_manifest(), handle, indent=2)
    lv.check(h.validate(fixture_manifest(), h.load_schema(h.MANIFEST_SCHEMA_PATH)) == [],
             "the fixture manifest validates against contracts/stage-manifest.schema.json")
    rc, out, err = run_capture([sys.executable, "-I", script, "stages", "--json", "--no-registry",
                                "--provider", good], timeout=ctx.deadline)
    rows = []
    try:
        rows = json.loads(out)["stages"] if rc == 0 else []
    except ValueError:
        pass
    listed = {r["id"]: r for r in rows if r.get("provider") == "fixture"}
    lv.check(sorted(listed) == sorted(FIXTURES), "stages --provider FILE lists every fixture stage",
             f"exit {rc}, listed {sorted(listed)}")
    if "fixture-copy" in listed:
        clip = make_clip(2, 2, 8, 2, "partial", "L0/ready")
        try:
            argv = stage_argv(split_command(listed["fixture-copy"]["stage"]), 2, 2, 8)
            result = run_stage(argv, clip, frame_size(2, 2, 8), deadline=ctx.deadline)
            lv.check(result["code"] is None and result["out"] == clip,
                     "the ready --stage string of stages --json runs as listed", describe(result))
        except ValueError as exc:
            lv.check(False, "the ready --stage string of stages --json splits", str(exc))
    odd = dict(fixture_manifest(), stages=[dict(fixture_manifest()["stages"][0], id="fixture-odd-geometry",
                                                geometry="stretch")])
    with open(os.path.join(folder, "odd.stage.json"), "w", encoding="utf-8") as handle:
        json.dump(odd, handle)
    future = dict(fixture_manifest(), contract="frames/9")
    future["provider"] = dict(future["provider"], id="fixture_future")
    with open(os.path.join(folder, "future.stage.json"), "w", encoding="utf-8") as handle:
        json.dump(future, handle)
    env = dict(os.environ, HANDOFF_PROVIDERS=folder)
    rc, out, err = run_capture([sys.executable, "-I", script, "stages", "--json", "--no-registry"],
                               timeout=ctx.deadline, env=env)
    try:
        doc = json.loads(out)
    except ValueError:
        doc = {"stages": [], "problems": []}
    ids = {r["id"] for r in doc.get("stages", [])}
    problems = " ".join(doc.get("problems", []))
    lv.check(rc == 0 and set(FIXTURES) <= ids, "HANDOFF_PROVIDERS folders are read", f"exit {rc}, {sorted(ids)}")
    lv.check("fixture-odd-geometry" not in ids and "stretch" in problems,
             "a stage with an unknown geometry value is refused and reported", problems[:200])
    lv.check("frames/9" in problems, "a manifest with an unknown contract major is refused and reported",
             problems[:200])


def l1_targets(ctx, lv, stage_cmd):
    """[(label, base argv, model name or None, params)] of the stages under test."""
    cfg = ctx.cases["L1"]
    if stage_cmd is not None:
        base = split_command(stage_cmd)
        model = next((cfg["models"][w] for w in base if w in cfg["models"]), None)
        return [("--stage", base, model, {})]
    targets = []
    if ctx.handoff is None:
        lv.missing(f"providers of the registry: {ctx.handoff_why}")
        return targets
    providers = ctx.providers()
    for case in cfg["stages"]:
        for tool_id in case["providers"]:
            entry = providers.get(tool_id)
            if entry is None or entry["program"] is None:
                lv.missing(f"{case['stage']} through {tool_id}: "
                           + (entry["why"] if entry else "not in contracts/tool.json"),
                           peer=entry.get("absent_peer") if entry else None)
                continue
            base = [entry["program"]] + entry["provider"]["prefix"] + ["apply", case["stage"]]
            targets.append((f"{tool_id} {case['stage']}", base, case["model"], case.get("params", {})))
    return targets


def level1(ctx, lv, targets, chain_pairs):
    cfg = ctx.cases["L1"]
    deadline = cfg.get("deadline", ctx.deadline)
    for depth, width, height, frames, alpha in matrix_rows(cfg["matrix"], ctx.mode):
        size = frame_size(width, height, depth)
        clip = make_clip(width, height, depth, frames, alpha, f"L1/{depth}/{width}x{height}/{frames}/{alpha}")
        row = f"{depth} bit {width}x{height} x{frames} {alpha}"
        outputs = {}
        for label, base, model, params in targets:
            argv = stage_argv(base, width, height, depth, cfg["matrix"]["fps"], params)
            result = run_stage(argv, clip, size, deadline=deadline)
            if not lv.check(result["code"] is None and not result["stray"], f"{label} {row}", describe(result)):
                continue
            outputs[label] = result["out"]
            if model:
                lv.check(result["out"] == MODELS[model](clip, width, height, depth),
                         f"{label} {row} equals the {model} model", "output differs")
            else:
                again = run_stage(argv, clip, size, deadline=deadline)
                lv.check(len(result["out"]) == len(clip) and again["out"] == result["out"],
                         f"{label} {row} keeps the byte count and is deterministic", describe(again))
        for first, second in chain_pairs:
            a = next(t for t in targets if t[0] == first)
            b = next(t for t in targets if t[0] == second)
            if first not in outputs:
                continue
            argv = stage_argv(b[1], width, height, depth, cfg["matrix"]["fps"], b[3])
            result = run_stage(argv, outputs[first], size, deadline=deadline)
            lv.check(result["code"] is None and result["out"] == clip,
                     f"{first} then {second} {row} returns the input", describe(result))


def chain_pairs_for(ctx, lv, targets, stage_cmd):
    """ordered pairs of involution stages to chain: every provider pair in both orders, or the
    one --stage target with itself."""
    involutions = set(ctx.cases["L1"].get("involutions", []))
    labels = [t[0] for t in targets if t[2] in involutions]
    if stage_cmd is not None:
        return [(labels[0], labels[0])] if labels else []
    pairs = [(a, b) for a in labels for b in labels if a != b]
    for case in ctx.cases["L1"]["stages"]:
        if case["model"] in involutions:
            present = [t for t in labels if t.endswith(" " + case["stage"])]
            if len(present) < len(case["providers"]):
                peers = [(ctx.providers().get(p) or {}).get("absent_peer") for p in case["providers"]]
                lv.missing(f"{case['stage']} in both orders: needs every provider of {case['providers']}, "
                           f"found {len(present)}", next((p for p in peers if p), None))
    return pairs


def level3(ctx, lv, targets):
    cfg = ctx.cases["L3"]
    deadline, stall = cfg["deadline"], cfg["stall"]
    width, height, depth, frames = cfg["clip"]
    size = frame_size(width, height, depth)
    clip = make_clip(width, height, depth, frames, "partial", "L3/judge")
    for case in cfg["judge"]:
        argv = stage_argv(fixture_command() + ["apply", case["stage"]], width, height, depth,
                          params=case.get("params"))
        what = f"judge {case['stage']}" + "".join(f" {k}={v}" for k, v in sorted(case.get("params", {}).items()))
        result = run_stage(argv, clip, size, deadline=deadline, stall=stall)
        lv.check((result["code"] or "ok") == case["expect"], what,
                 f"expected {case['expect']}, got {describe(result)}")
        lv.check(result["seconds"] <= deadline + 20 and not result["stray"], f"{what} ends within its deadline",
                 f"{result['seconds']:.1f} s, stray pipe holder {result['stray']}")
        left = wait_gone(child_pids(result["err"]))
        lv.check(not left, f"{what} leaves no process behind", f"alive: {left}")
    big = cfg["big_clip"]
    big_size = frame_size(big[0], big[1], big[2])
    big_data = make_clip(big[0], big[1], big[2], big[3], "partial", "L3/big")
    for label, base, _model, params in targets:
        for check in cfg["stage_checks"]:
            depth_arg = check.get("depth", depth)
            what = f"{label} {check['name']}"
            if check["input"] == "partial":
                data, limit, fb = clip + clip[:size // 2], None, size
            elif check["input"] == "empty":
                data, limit, fb = b"", None, size
            elif check["input"] == "big":
                data, limit, fb = big_data, big_size, big_size
            else:
                data, limit, fb = clip, None, size
            if check["input"] == "big":
                argv = stage_argv(base, big[0], big[1], big[2], params=params, extra=check.get("extra", ()))
            else:
                argv = stage_argv(base, width, height, depth_arg, params=params, extra=check.get("extra", ()))
            result = run_stage(argv, data, fb, deadline=deadline, read_limit=limit)
            if "exits" in check:
                lv.check(result["exit"] in check["exits"] and not result["killed"], what,
                         f"expected exit {check['exits']}, got {describe(result)}")
            if "out_bytes" in check:
                lv.check(len(result["out"]) == check["out_bytes"], f"{what} output size", describe(result))
            if check.get("terminates"):
                lv.check(not result["killed"] and not result["stray"], f"{what}: the stage ends by itself",
                         describe(result))


INVERT_8_CODE = ("import sys; d = sys.stdin.buffer.read(); "
                 "sys.stdout.buffer.write(bytes(b if i % 4 == 3 else 255 - b for i, b in enumerate(d)))")


def level4(ctx, lv):
    h = ctx.handoff
    if h is None:
        lv.missing(ctx.handoff_why)
        return
    script = os.path.join(ctx.root, "tools", "handoff.py")
    base = os.path.join(ctx.tmp, "packets")
    recv = os.path.join(base, "recv")
    os.makedirs(os.path.join(recv, "scratch"))
    raw = os.path.join(base, "in.raw")
    with open(raw, "wb") as handle:
        handle.write(make_clip(4, 2, 8, 3, "partial", "L4/in"))
    env = {k: v for k, v in os.environ.items() if k not in ("HANDOFF_SELF", "HANDOFF_PROVIDERS")}
    error_schema = h.load_schema(os.path.join(ctx.root, "contracts", "error.schema.json"))
    packet_schema = h.load_schema(os.path.join(ctx.root, "contracts", "handoff-packet.schema.json"))
    documents = []

    def cli(*argv):
        rc, out, err = run_capture([sys.executable, "-I", script, *argv, "--root", recv, "--json"],
                                   timeout=ctx.deadline, env=env)
        try:
            doc = json.loads(out)
        except ValueError:
            doc = None
        if isinstance(doc, dict) and doc.get("ok") is False:
            documents.append(doc)
        return rc, doc

    def path_of(pid):
        return os.path.join(recv, "handoff", pid + ".json")

    def load(pid):
        with open(path_of(pid), encoding="utf-8") as handle:
            return json.load(handle)

    def save(packet):
        with open(path_of(packet["id"]), "w", encoding="utf-8") as handle:
            json.dump(packet, handle, indent=2)

    def new(task):
        rc, doc = cli("new", "--from", "sapphire", "--to", "twixtor", "--task", task, "--input", raw,
                      "--width", "4", "--height", "2", "--depth", "8", "--fps", "24", "--stage", "selftest-invert")
        return rc, (doc or {}).get("id")

    def paths(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("path", "stdin", "stdout", "cwd", "scratch") and isinstance(item, str):
                    yield item
                else:
                    yield from paths(item)
        elif isinstance(value, list):
            for item in value:
                yield from paths(item)

    rc, pid = new("conformance packet")
    if not lv.check(rc == 0 and pid and h.ID_RE.match(pid), "new writes a packet", f"exit {rc}"):
        return
    packet = load(pid)
    lv.check(h.validate(packet, packet_schema) == [] and packet["schema_version"] == 2 and packet["revision"] == 1
             and packet["status"] == "new", "the new packet validates as schema_version 2, revision 1")
    bad = [p for p in paths(packet) if not h.REL_RE.match(p)]
    lv.check(not bad, "every packet path is relative POSIX without ..", str(bad[:3]))
    rc, _ = cli("accept", pid)
    rc2, _ = cli("accept", pid)
    packet = load(pid)
    lv.check(rc == 0 and rc2 == 0 and packet["status"] == "accepted" and packet["revision"] == 2,
             "accept moves to accepted at revision 2 and a second accept is a no-op",
             f"exits {rc}, {rc2}, revision {packet['revision']}")
    output = os.path.join(recv, "out.raw")
    spec = os.path.join(base, "run.json")
    with open(spec, "w", encoding="utf-8") as handle:
        json.dump({"argv": [sys.executable, "-I", "-c", INVERT_8_CODE],
                   "stdin": h.resolve(packet["inputs"][0]["path"], recv), "stdout": output}, handle)
    rc, _ = cli("done", pid, "--output", output, "--run", "@" + spec, "--scratch", os.path.join(recv, "scratch"))
    packet = load(pid)
    lv.check(rc == 0 and packet["status"] == "done" and packet["revision"] == 3 and packet["result"]["commands"],
             "done runs and records the command", f"exit {rc}, status {packet['status']}")
    rc, _ = cli("verify", pid, "--replay", "--no-write")
    lv.check(rc == 3 and documents and documents[-1]["error"]["code"] == "unsupported",
             "replay of an unregistered program without --trust is unsupported (exit 3)", f"exit {rc}")
    rc, _ = cli("verify", pid, "--replay", "--trust")
    lv.check(rc == 0 and load(pid)["status"] == "verified", "verify --replay --trust passes and marks verified",
             f"exit {rc}")
    rc, second = new("revision conflict")
    rc, _ = cli("accept", second, "--revision", "7")
    lv.check(rc == 4 and documents and documents[-1]["error"]["code"] == "bad_input",
             "a revision conflict is bad_input (exit 4)", f"exit {rc}")
    lock = os.path.join(recv, "handoff", second + ".lock")
    with open(lock, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "host": socket.gethostname(), "at": h.now_iso()}, handle)
    rc, _ = cli("accept", second, "--lock-wait", "1")
    lv.check(rc == 5 and documents and documents[-1]["error"]["code"] == "stage_timeout",
             "a lock held by a live process is stage_timeout (exit 5)", f"exit {rc}")
    gone = subprocess.Popen([sys.executable, "-I", "-c", "pass"])
    gone.wait()
    with open(lock, "w", encoding="utf-8") as handle:
        json.dump({"pid": gone.pid, "host": socket.gethostname(), "at": h.now_iso()}, handle)
    rc, _ = cli("accept", second)
    lv.check(rc == 0 and load(second)["status"] == "accepted", "a lock of a process that is gone is broken",
             f"exit {rc}")
    rc, third = new("path rule")
    packet = load(third)
    packet["inputs"][0]["path"] = "../outside.raw"
    save(packet)
    rc, _ = cli("verify", third, "--no-write")
    lv.check(rc == 4, "a packet path with .. is bad_input (exit 4)", f"exit {rc}")
    rc, fourth = new("size cap")
    packet = load(fourth)
    packet["notes"].append({"at": packet["created"], "by": "sapphire", "text": "x" * (h.MAX_PACKET_BYTES + 16)})
    save(packet)
    rc, _ = cli("show", fourth)
    lv.check(rc == 4, "a packet over 1 MiB is bad_input (exit 4)", f"exit {rc}")
    old_id = "20261026T120000Z-sapphire-twixtor-0c0f0e"
    inside = os.path.join(recv, "old.raw")
    shutil.copyfile(raw, inside)
    old = {"schema_version": 1, "id": old_id, "created": "2026-10-26T12:00:00Z", "updated": "2026-10-26T12:00:00Z",
           "from": "sapphire", "to": "twixtor", "task": "old", "status": "new",
           "inputs": [{"path": inside, "sha256": sha256_file(inside), "width": 4, "height": 2, "depth": 8,
                       "pix_fmt": "rgba", "fps": "24/1", "frames": 3}],
           "request": {"stages": [{"stage": "selftest-invert", "params": {}, "layers": {}}]},
           "expect": {"frames": 3, "size": 96, "determinism": "exact",
                      "tolerance": {"max_abs_diff": 0, "max_diff_fraction": 0}},
           "result": None, "notes": [{"at": "2026-10-26T12:00:00Z", "by": "sapphire", "text": "created"}]}
    save(old)
    rc, _ = cli("show", old_id)
    rc2, _ = cli("accept", old_id)
    lv.check(rc == 0 and rc2 != 0, "a version 1 packet is readable and refuses writes before migrate",
             f"show exit {rc}, accept exit {rc2}: " + (documents[-1]["error"]["message"] if documents else ""))
    rc, _ = cli("migrate", old_id)
    migrated = load(old_id)
    lv.check(rc == 0 and migrated["schema_version"] == 2 and h.validate(migrated, packet_schema) == []
             and all(h.REL_RE.match(p) for p in paths(migrated)), "migrate rewrites version 1 as version 2",
             f"exit {rc}")
    bad_docs = [d for d in documents if h.validate(d, error_schema)]
    lv.check(documents and not bad_docs, "every refusal printed a valid error document",
             f"{len(documents)} documents, {len(bad_docs)} invalid")


def sha256_file(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


# -- wrappers: L2, L5 and the bench --------------------------------------------------------------

PLACEHOLDERS = ["--width", "{width}", "--height", "{height}", "--depth", "{depth}", "--fps", "{fps}"]
RAW_PIX_FMT = {8: "rgba", 16: "rgba64le", 32: "gbrapf32le"}
BENCH_SOURCE_PIX_FMT = {8: "bgr0", 16: "gbrp16le"}


def join_command(argv):
    """one command string that split_command and the wrappers turn back into `argv`."""
    words = []
    for word in map(str, argv):
        if word and not any(ch.isspace() or ch == '"' for ch in word):
            words.append(word)
        else:
            words.append('"' + word.replace('"', '\\"') + '"')
    return " ".join(words)


def resolve_program(word):
    """the program of a --stage command as an absolute path, so the stage runs the same in every
    child: what shutil.which finds (PATH for a bare name; a path with a folder part relative to the
    caller's working directory, PATHEXT tried on Windows), else an existing file relative to the
    caller's working directory; unchanged when neither finds it (the run reports tool_missing)."""
    found = shutil.which(word) or (word if os.path.isfile(word) else None)
    return os.path.abspath(found) if found else word


def _program(value):
    if not value:
        return None
    if os.path.isfile(value):
        return os.path.abspath(value)
    return shutil.which(value)


class Tools:
    """the programs L2, L5 and the bench drive; `why` holds the reason each missing one is
    missing and `peer` the absent peer repository it waits for."""

    def __init__(self, ctx):
        self.why, self.peer = {}, {}
        self.ffmpeg = _program(os.environ.get("MEDIA_FFMPEG_PATH") or os.environ.get("FFMPEG") or "ffmpeg")
        self.ffprobe = _program(os.environ.get("MEDIA_FFPROBE_PATH") or os.environ.get("FFPROBE") or "ffprobe")
        if not (self.ffmpeg and self.ffprobe):
            self.why["ffmpeg"] = ("ffmpeg and ffprobe: not found on PATH or through MEDIA_FFMPEG_PATH, FFMPEG, "
                                  "MEDIA_FFPROBE_PATH, FFPROBE")
        providers = ctx.providers() if ctx.handoff is not None else {}
        found = {}
        for key, tool_id in (("vfx", "twixtor"), ("sapphire", "sapphire")):
            entry = providers.get(tool_id) or {}
            found[tool_id] = entry
            if not entry:
                self.why[key] = f"{tool_id} {key}: " + (ctx.handoff_why or "not in contracts/tool.json")
            elif entry["program"] is None:
                self.why[key] = f"{tool_id} {key}: {entry['why']}"
                self.peer[key] = entry.get("absent_peer")
        self.vfx, self.nh = found["twixtor"].get("program"), found["sapphire"].get("program")
        twixtor_root = found["twixtor"].get("root")
        self.apply_py = self.ffstage = None
        for key, attr in (("apply.py", "apply_py"), ("ffstage.py", "ffstage")):
            path = os.path.join(twixtor_root, "tools", key) if twixtor_root else None
            if path and os.path.isfile(path):
                setattr(self, attr, path)
            elif twixtor_root:
                self.why[key] = f"twixtor tools/{key}: not present in {twixtor_root}"
            else:
                self.why[key] = (f"twixtor tools/{key}: the twixtor checkout was not found ("
                                 + (found["twixtor"].get("why") or ctx.handoff_why or "twixtor is not registered") + ")")
                self.peer[key] = "twixtor" if found["twixtor"] else None

    def need(self, keys):
        """[(reason, absent peer or None)] of the programs in `keys` that are missing."""
        return [(self.why[k], self.peer.get(k)) for k in keys if k in self.why]

    def env(self):
        env = dict(os.environ)
        if self.ffmpeg and self.ffprobe:
            env.update(MEDIA_FFMPEG_PATH=self.ffmpeg, MEDIA_FFPROBE_PATH=self.ffprobe)
        return env

    def sapphire_stage(self, stage, params=(), extra=()):
        argv = [self.nh, "stage", "apply", stage] + PLACEHOLDERS
        for key, value in params:
            argv += ["--param", f"{key}={value}"]
        return join_command(argv + list(extra))

    def sapphire_filter(self, graph, expect="preserve"):
        return self.sapphire_stage("ffmpeg-filter", [("graph", graph), ("expect", expect)],
                                   ["--ffmpeg-path", self.ffmpeg])

    def twixtor_filter(self, graph, expect="preserve"):
        return join_command([sys.executable, "-I", self.ffstage, "apply", "ffmpeg-filter"] + PLACEHOLDERS
                            + ["--param", f"graph={graph}", "--param", f"expect={expect}"])

    @staticmethod
    def fixture_stage(stage, params=None):
        argv = fixture_command() + ["apply", stage] + PLACEHOLDERS
        for key, value in sorted((params or {}).items()):
            argv += ["--param", f"{key}={value}"]
        return join_command(argv)

    def wrapper_argv(self, wrapper, src, effect, out, stages=(), extra=(), ffmpeg=None):
        """argv of one wrapper run. `stages` are (command, geometry or None, "pre" or "post");
        sapphire gets them around its `vfx` marker in the same pipe order. A None
        `effect` runs sapphire on its stages alone."""
        tail = ["--ffmpeg-path", ffmpeg or self.ffmpeg, "--ffprobe-path", self.ffprobe, "-o", out]
        if wrapper == "tools/apply.py":
            argv = [sys.executable, "-I", self.apply_py, src, effect, "--vfx", self.vfx] + tail
            for command, geometry, where in stages:
                argv += ["--pre-stage" if where == "pre" else "--post-stage", command]
                argv += ["--geometry", geometry] if geometry else []
            return argv + list(extra)
        argv = [self.nh, src] + ([effect, "--vfx", self.vfx] if effect else []) + tail
        order = [s for s in stages if s[2] == "pre"]
        if effect and stages:
            order.append(("vfx", None, "effect"))
        order += [s for s in stages if s[2] == "post"]
        for command, geometry, where in order:
            argv += ["--stage", command]
            argv += ["--geometry", geometry] if geometry else []
        return argv + list(extra)


def run_feed(argv, data, timeout, env=None):
    """(exit or None past the deadline, stdout bytes, stderr bytes) with `data` on stdin."""
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=env, **group_kwargs())
    try:
        out, err = proc.communicate(data, timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        out, err = proc.communicate(timeout=15)
        return None, out, err
    return proc.returncode, out, err


def encode_source(tools, clip, width, height, depth, path, timeout):
    """write raw frames as a lossless ffv1 file at 24 fps; (exit, stderr tail)."""
    rc, _, err = run_feed([tools.ffmpeg, "-v", "error", "-f", "rawvideo", "-pix_fmt", RAW_PIX_FMT[depth],
                           "-s", f"{width}x{height}", "-r", "24", "-i", "-", "-c:v", "ffv1", "-level", "3",
                           "-y", path], clip, timeout)
    return rc, err.decode("utf-8", "replace").strip()[-200:]


def decode_frames(tools, path, depth, timeout):
    """the raw frames of a file at `depth`, or None when ffmpeg cannot read it."""
    if not os.path.isfile(path):
        return None
    rc, out, _ = run_feed([tools.ffmpeg, "-v", "error", "-nostdin", "-i", path, "-map", "0:v:0", "-f", "rawvideo",
                           "-pix_fmt", RAW_PIX_FMT[depth], "-"], b"", timeout)
    return out if rc == 0 else None


def hflip(data, width, height, depth):
    """frames mirrored left to right (packed formats: depth 8 and 16)."""
    bpp, out = BYTES_PER_PIXEL[depth], bytearray()
    row = width * bpp
    for start in range(0, len(data), row):
        line = data[start:start + row]
        for x in range(width - 1, -1, -1):
            out += line[x * bpp:(x + 1) * bpp]
    return bytes(out)


def first_difference(a, b):
    a, b = a or b"", b or b""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def tail_text(*texts):
    lines = []
    for text in texts:
        lines += [line.strip() for line in (text or "").splitlines() if line.strip()]
    return " | ".join(lines[-3:])[:300]


def frames_detail(got, expected):
    return (f"{len(got or b'')} bytes, expected {len(expected)}, first difference at byte "
            f"{first_difference(got, expected)}")


def code_exits(schema, contract=CONTRACT):
    """{error code: wrapper exit} of the x-codes table of contracts/error.schema.json."""
    return {row["code"]: row["exit"][contract] for row in schema.get("x-codes", [])}


def missing_all(lv, missing, prefix):
    for reason, peer in missing:
        lv.missing(f"{prefix}: {reason}", peer)
    return bool(missing)


def lossless_source(ctx, lv, t, clip, width, height, depth, name, deadline):
    """the path of a lossless source of `clip`, checked to decode back to it, or None."""
    src = os.path.join(ctx.tmp, name)
    rc, why = encode_source(t, clip, width, height, depth, src, deadline)
    if not lv.check(rc == 0, f"depth {depth}: the lossless source encodes", f"exit {rc}: {why}"):
        return None
    back = decode_frames(t, src, depth, deadline)
    if not lv.check(back == clip, f"depth {depth}: the lossless source decodes to its frames",
                    frames_detail(back, clip)):
        return None
    return src


def level2(ctx, lv):
    cfg, t = ctx.cases["L2"], ctx.tools()
    width, height, frames = cfg["size"][0], cfg["size"][1], cfg["frames"]
    graph, deadline = cfg["graph"], cfg.get("deadline", ctx.deadline)
    if missing_all(lv, t.need(["ffmpeg"]), "every wrapper run"):
        return
    runs = [
        ("tools/apply.py: vfx selftest-invert, twixtor ffmpeg-filter, sapphire selftest-invert", "tools/apply.py",
         ("vfx", "apply.py", "ffstage.py", "sapphire"),
         lambda: [(t.twixtor_filter(graph), None, "post"), (t.sapphire_stage("selftest-invert"), None, "post")]),
        ("tools/apply.py: sapphire selftest-invert, sapphire ffmpeg-filter, vfx selftest-invert", "tools/apply.py",
         ("vfx", "apply.py", "sapphire"),
         lambda: [(t.sapphire_stage("selftest-invert"), None, "pre"), (t.sapphire_filter(graph), None, "pre")]),
        ("sapphire: vfx selftest-invert, @ffmpeg-filter, @selftest-invert", "sapphire",
         ("vfx", "sapphire"),
         lambda: [(f"@ffmpeg-filter graph={graph} expect=preserve", None, "post"), ("@selftest-invert", None, "post")]),
        ("sapphire: @selftest-invert, twixtor ffmpeg-filter, vfx selftest-invert", "sapphire",
         ("vfx", "sapphire", "ffstage.py"),
         lambda: [("@selftest-invert", None, "pre"), (t.twixtor_filter(graph), None, "pre")]),
    ]
    runnable = [run for run in runs if not missing_all(lv, t.need(run[2]), run[0])]
    if not runnable:
        return
    for depth in cfg["depths"][ctx.mode]:
        clip = make_clip(width, height, depth, frames, cfg["alpha"], f"L2/{depth}")
        src = lossless_source(ctx, lv, t, clip, width, height, depth, f"l2_{depth}.mkv", deadline)
        if src is None:
            continue
        # selftest-invert is an involution and commutes with a mirror: the chain is the mirror
        expected = hflip(clip, width, height, depth)
        for n, (label, wrapper, _, stages) in enumerate(runnable):
            out = os.path.join(ctx.tmp, f"l2_{depth}_{n}.mkv")
            argv = t.wrapper_argv(wrapper, src, "selftest-invert", out, stages(), ["--lossless", "--depth", str(depth)])
            rc, so, se = run_capture(argv, timeout=deadline, env=t.env())
            if not lv.check(rc == 0, f"{label} at depth {depth}", f"exit {rc}: {tail_text(so, se)}"):
                continue
            got = decode_frames(t, out, depth, deadline)
            lv.check(got == expected, f"{label} at depth {depth} returns the input mirrored by {graph}",
                     frames_detail(got, expected))


def _json_or_none(text):
    try:
        doc = json.loads(text)
    except ValueError:
        return None
    return doc if isinstance(doc, dict) else None


def level5(ctx, lv):
    cfg, t, h = ctx.cases["L5"], ctx.tools(), ctx.handoff
    if h is None:
        lv.missing(ctx.handoff_why)
        return
    schemas = {name: h.load_schema(os.path.join(ctx.root, "contracts", f"{name}.schema.json"))
               for name in ("run-report", "error", "describe")}
    exits = code_exits(schemas["error"])
    if missing_all(lv, t.need(["ffmpeg"]), "every frames/2 wrapper run"):
        return
    width, height, depth, frames = cfg["clip"]
    deadline = cfg.get("deadline", ctx.deadline)
    wrappers = []
    if not missing_all(lv, t.need(["vfx", "apply.py", "ffstage.py"]), "tools/apply.py under frames/2"):
        wrappers.append(("tools/apply.py", "selftest-invert", lambda g, e: t.twixtor_filter(g, e)))
    if not missing_all(lv, t.need(["sapphire"]), "sapphire under frames/2"):
        effect = "selftest-invert" if t.vfx else None
        if effect is None:
            missing_all(lv, t.need(["vfx"]), "sapphire frames/2 runs with an effect (they ran on stages alone)")
        wrappers.append(("sapphire", effect, lambda g, e: t.sapphire_filter(g, e)))
    if not wrappers:
        return
    clip = make_clip(width, height, depth, frames, "partial", "L5")
    src = lossless_source(ctx, lv, t, clip, width, height, depth, "l5.mkv", deadline)
    if src is None:
        return
    flags = ["--contract", "frames/2", "--json", "--lossless", "--depth", str(depth)]
    for n, (wrapper, effect, make_filter) in enumerate(wrappers):
        def run(tag, stages=(), extra=(), ffmpeg=None, base=flags):
            out = os.path.join(ctx.tmp, f"l5_{n}_{tag}.mkv")
            argv = t.wrapper_argv(wrapper, src, effect, out, stages, list(base) + list(extra), ffmpeg=ffmpeg)
            rc, so, se = run_capture(argv, timeout=deadline, env=t.env())
            return rc, _json_or_none(so), so, se, out

        what = f"{wrapper} frames/2 report"
        rc, doc, so, se, out = run("ok", [(t.fixture_stage("fixture-copy"), None, "pre")])
        if lv.check(rc == 0 and doc is not None, f"{what}: the run succeeds and prints JSON",
                    f"exit {rc}: {tail_text(so, se)}"):
            problems = h.validate(doc, schemas["run-report"])
            lv.check(not problems, f"{what} validates contracts/run-report.schema.json", "; ".join(problems[:3]))
            lv.check(doc.get("contract") == "frames/2", f"{what} names contract frames/2", str(doc.get("contract")))
            steps = [s for s in doc.get("stages") or [] if isinstance(s, dict)]
            roles = [s.get("role") for s in steps]
            want = ["decode", "pre", "encode"] + (["effect"] if effect else [])
            lv.check(all(r in roles for r in want), f"{what} lists the {', '.join(want)} steps", str(roles))
            counts, ordered = {}, True
            for step in steps:
                counts[step.get("role")] = counts.get(step.get("role"), 0) + 1
                ordered = ordered and step.get("index") == counts[step.get("role")]
            lv.check(ordered, f"{what}: index counts 1, 2, ... per role in pipe order",
                     json.dumps([[s.get("role"), s.get("index")] for s in steps]))
            geometry = doc.get("geometry") or {}
            lv.check([geometry.get(k) for k in ("width", "height", "depth", "fps")] == [width, height, depth, "24/1"],
                     f"{what} carries the output geometry", json.dumps(geometry))
            expected = invert(clip, width, height, depth) if effect else clip
            got = decode_frames(t, out, depth, deadline)
            lv.check(got == expected, f"{wrapper} frames/2 run writes the frames of its steps",
                     frames_detail(got, expected))
        for case in cfg["errors"]:
            stages = []
            if case.get("stage"):
                stages = [(t.fixture_stage(case["stage"], case.get("params")), case.get("geometry"), "pre")]
            extra = list(case.get("flags", []))
            if case["code"] == "stage_timeout":
                extra += ["--stage-timeout", str(cfg["stage_timeout"])]
            ffmpeg = None
            if case.get("absent") == "ffmpeg":
                ffmpeg = os.path.join(ctx.tmp, "absent", "ffmpeg.exe" if IS_WINDOWS else "ffmpeg")
            label = f"{wrapper} frames/2, {case['name']}"
            rc, doc, so, se, out = run(case["code"], stages, extra, ffmpeg)
            lv.check(rc == exits[case["code"]], f"{label}: exit {exits[case['code']]}", f"exit {rc}: {tail_text(so, se)}")
            problems = h.validate(doc, schemas["error"]) if doc is not None else ["stdout is not one JSON object"]
            lv.check(not problems, f"{label}: the error object validates contracts/error.schema.json",
                     "; ".join(problems[:3]))
            code = ((doc or {}).get("error") or {}).get("code")
            lv.check(code == case["code"], f"{label}: error code {case['code']}", str(code))
            if case["code"] == "stage_timeout":
                pids = child_pids((so + se).encode("utf-8", "replace"))
                lv.check(not wait_gone(pids), f"{label}: the stage and the child it started are gone", f"pids {pids}")
        dw, dh = cfg["describe"]["width"], cfg["describe"]["height"]
        # a declared stage after the effect: sapphire refuses one before it (the effect
        # runs at the input size), tools/apply.py accepts both places
        declared = [(make_filter(cfg["describe"]["graph"], "declared"), "declared", "post")]
        rc, answer, so, se, out = run("describe", declared, ["--describe"],
                                      base=["--contract", "frames/2", "--lossless", "--depth", str(depth)])
        problems = h.validate(answer, schemas["describe"]) if answer is not None else ["stdout is not one JSON object"]
        lv.check(rc == 0 and not problems, f"{wrapper} --describe prints a describe answer",
                 f"exit {rc}: {'; '.join(problems[:3])} {tail_text(so, se)}")
        lv.check([(answer or {}).get(k) for k in ("width", "height", "depth", "fps")] == [dw, dh, depth, "24/1"],
                 f"{wrapper} --describe answers the geometry after a declared stage", json.dumps(answer)[:200])
        lv.check(not os.path.exists(out), f"{wrapper} --describe writes no output")
        rc, doc, so, se, out = run("declared", declared)
        if lv.check(rc == 0 and doc is not None, f"{wrapper} frames/2 run through a declared stage",
                    f"exit {rc}: {tail_text(so, se)}"):
            geometry = doc.get("geometry") or {}
            lv.check([geometry.get("width"), geometry.get("height")] == [dw, dh],
                     f"{wrapper} report geometry follows the declared stage", json.dumps(geometry))
            got = decode_frames(t, out, depth, deadline)
            lv.check(got is not None and len(got) == frames * frame_size(dw, dh, depth),
                     f"{wrapper} encodes {frames} frames of the declared {dw}x{dh}", f"{len(got or b'')} bytes")


# -- bench --------------------------------------------------------------------------------------

def folder_bytes(path):
    total = 0
    for base, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(base, name))
            except OSError:
                pass
    return total


class WindowsJob:
    """a job object around one process tree: the peak committed memory of the whole tree and a
    kill of every process in it. The process starts suspended and resumes once it is assigned,
    so no child escapes the job."""

    def __init__(self, proc):
        import ctypes
        from ctypes import wintypes
        self.ctypes, self.k32 = ctypes, ctypes.WinDLL("kernel32", use_last_error=True)
        k32 = self.k32
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                                  ctypes.c_void_p]
        k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = wintypes.HANDLE(int(proc._handle))
        self.job = k32.CreateJobObjectW(None, None)
        self.assigned = bool(self.job) and bool(k32.AssignProcessToJobObject(self.job, handle))
        ntdll = ctypes.WinDLL("ntdll")
        ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
        ntdll.NtResumeProcess(handle)

    def peak_bytes(self):
        ctypes = self.ctypes
        size_t, u64, dword = ctypes.c_size_t, ctypes.c_ulonglong, ctypes.c_uint32

        class Basic(ctypes.Structure):
            _fields_ = [("a", ctypes.c_int64), ("b", ctypes.c_int64), ("flags", dword), ("wmin", size_t),
                        ("wmax", size_t), ("active", dword), ("affinity", size_t), ("priority", dword),
                        ("scheduling", dword)]

        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", u64 * 6), ("process_limit", size_t), ("job_limit", size_t),
                        ("peak_process", size_t), ("peak_job", size_t)]

        info = Extended()
        ok = self.assigned and self.k32.QueryInformationJobObject(self.job, 9, ctypes.byref(info),
                                                                  ctypes.sizeof(info), None)
        return int(info.peak_job) if ok else None

    def terminate(self):
        if self.job:
            self.k32.TerminateJobObject(self.job, 1)

    def close(self):
        if self.job:
            self.k32.CloseHandle(self.job)
            self.job = None


def meter_main(argv):
    """`conformance.py meter --temp DIR [--deadline S] -- CMD...`: run CMD with TMP, TEMP and
    TMPDIR at DIR; print one JSON line: exit, seconds, peak_mb (Windows: peak committed memory
    of the whole tree, job-commit; POSIX: the largest process's resident set, max-rss),
    disk_mb (the peak size of DIR) and stderr_tail."""
    parser = argparse.ArgumentParser(prog="conformance.py meter", allow_abbrev=False)
    parser.add_argument("--temp", required=True)
    parser.add_argument("--deadline", type=float, default=900.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is needed after --")
    os.makedirs(args.temp, exist_ok=True)
    env = dict(os.environ, TMP=args.temp, TEMP=args.temp, TMPDIR=args.temp)
    stop, peak_disk, tail = threading.Event(), [0], []

    def watch():
        while not stop.is_set():
            peak_disk[0] = max(peak_disk[0], folder_bytes(args.temp))
            stop.wait(0.1)

    start, job = time.monotonic(), None
    kwargs = {"creationflags": 0x4} if IS_WINDOWS else group_kwargs()  # 0x4: CREATE_SUSPENDED
    proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            env=env, **kwargs)
    if IS_WINDOWS:
        job = WindowsJob(proc)
    reader = threading.Thread(target=lambda: tail.extend(proc.stderr.read().decode("utf-8", "replace").splitlines()),
                              daemon=True)
    watcher = threading.Thread(target=watch, daemon=True)
    reader.start()
    watcher.start()
    try:
        code = proc.wait(timeout=args.deadline)
    except subprocess.TimeoutExpired:
        job.terminate() if job else kill_tree(proc)
        proc.wait(timeout=15)
        code = None
    seconds = time.monotonic() - start
    stop.set()
    watcher.join()
    reader.join(timeout=15)
    peak_disk[0] = max(peak_disk[0], folder_bytes(args.temp))
    if job:
        peak, kind = job.peak_bytes(), "job-commit"
        job.close()
    else:
        import resource
        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        peak, kind = (rss if sys.platform == "darwin" else rss * 1024), "max-rss"
    print(json.dumps({"exit": code, "seconds": round(seconds, 3),
                      "peak_mb": None if peak is None else round(peak / 2 ** 20, 1), "peak_kind": kind,
                      "disk_mb": round(peak_disk[0] / 2 ** 20, 1), "stderr_tail": tail[-3:]}))
    return 0


def machine_text(t):
    rc, out, _ = run_capture([t.ffmpeg, "-version"], timeout=30)
    found = re.search(r"ffmpeg version (\S+)", out or "")
    version = found.group(1).split("-")[0] if found else "unknown"
    return (f"{platform.system()} {platform.release()} {platform.machine()}, {os.cpu_count()} logical CPUs, "
            f"Python {platform.python_version()}, ffmpeg {version}")


def run_bench(ctx, lv):
    """the rows of docs/handoff-bench.tsv, or None when a program is missing."""
    cfg, t = ctx.cases["bench"], ctx.tools()
    if missing_all(lv, t.need(["ffmpeg", "vfx", "sapphire", "apply.py"]), "bench"):
        return None
    machine, deadline, rows = machine_text(t), cfg.get("deadline", 900), []
    stage = t.sapphire_stage(cfg["chain_stage"])
    plan = [(cfg["effect"], n) for n in cfg["chain"]] + [(cfg["spool_stage"], 0)]
    for (width, height), frames in zip(cfg["sizes"], cfg["frames"]):
        for depth in cfg["depths"]:
            src = os.path.join(ctx.tmp, f"bench_{width}x{height}_{depth}.mkv")
            rc, so, se = run_capture([t.ffmpeg, "-v", "error", "-nostdin", "-f", "lavfi", "-i",
                                      f"testsrc2=size={width}x{height}:rate=24", "-frames:v", str(frames),
                                      "-c:v", "ffv1", "-level", "3", "-pix_fmt", BENCH_SOURCE_PIX_FMT[depth],
                                      "-y", src], timeout=deadline)
            if not lv.check(rc == 0, f"bench source {width}x{height} depth {depth}", tail_text(so, se)):
                continue
            for wrapper in cfg["wrappers"]:
                for effect, count in plan:
                    out, spool = os.path.join(ctx.tmp, "bench_out.mkv"), os.path.join(ctx.tmp, "bench_temp")
                    shutil.rmtree(spool, ignore_errors=True)
                    if os.path.exists(out):
                        os.remove(out)
                    argv = t.wrapper_argv(wrapper, src, effect, out, [(stage, None, "post")] * count,
                                          ["--contract", "frames/2", "--lossless", "--depth", str(depth)])
                    rc, so, se = run_capture([sys.executable, "-I", SELF, "meter", "--temp", spool, "--deadline",
                                              str(deadline), "--"] + argv, timeout=deadline + 60, env=t.env())
                    meter = _json_or_none((so.strip().splitlines() or [""])[-1])
                    label = f"bench {wrapper} {effect} with {count} chained stages at {width}x{height} depth {depth}"
                    if not lv.check(rc == 0 and meter is not None and meter["exit"] == 0, label,
                                    f"exit {rc}: {tail_text(' | '.join((meter or {}).get('stderr_tail') or []), se)}"):
                        continue
                    seconds = meter["seconds"]
                    out_mb = os.path.getsize(out) / 2 ** 20 if os.path.exists(out) else 0.0
                    rows.append([wrapper, "frames/2", effect, str(count), str(width), str(height), str(depth),
                                 str(frames), f"{seconds:.3f}", f"{frames / seconds:.2f}" if seconds > 0 else "0",
                                 "" if meter["peak_mb"] is None else f"{meter['peak_mb']:.1f}", meter["peak_kind"],
                                 f"{meter['disk_mb']:.1f}", f"{out_mb:.2f}", machine])
    shutil.rmtree(os.path.join(ctx.tmp, "bench_temp"), ignore_errors=True)
    return rows


def write_bench(root, columns, rows):
    path = os.path.join(root, "docs", "handoff-bench.tsv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="ascii", newline="\n") as handle:
        for row in [columns] + rows:
            handle.write("\t".join(row) + "\n")
    return path


# -- main ---------------------------------------------------------------------------------------

def parse_levels(text):
    levels = [w.strip().upper() for w in text.split(",") if w.strip()]
    unknown = [w for w in levels if w not in LEVELS]
    if unknown or not levels:
        raise argparse.ArgumentTypeError(f"levels are a comma list of {', '.join(LEVELS)}")
    return levels


def build_parser():
    parser = argparse.ArgumentParser(prog="conformance.py", allow_abbrev=False,
                                     description="conformance suite of the handoff convention (docs/handoff.md)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="a covering subset, about a minute (the default)")
    mode.add_argument("--full", action="store_true", help="the whole matrix of every level")
    parser.add_argument("--levels", type=parse_levels, metavar="L0,L1,...", help="only these levels")
    parser.add_argument("--stage", metavar="CMD", help="run L1 and L3 on this one stage command")
    parser.add_argument("--bench", action="store_true", help="write docs/handoff-bench.tsv")
    parser.add_argument("--root", metavar="DIR", help="repository root (default: the folder above this script's)")
    parser.add_argument("--deadline", type=float, default=120.0, metavar="SECONDS",
                        help="limit of one helper command (default 120); stage cases use cases.json")
    parser.add_argument("--json", action="store_true", dest="as_json", help="print the result as JSON")
    parser.add_argument("--allow-absent-peer", action="store_true",
                        help="exit 0 when every part that did not run needs an absent peer repository (CI)")
    return parser


def load_cases(root):
    with open(os.path.join(root, *CASES_REL.split("/")), encoding="utf-8") as handle:
        return json.load(handle)


def run_levels(ctx, selected, stage_cmd):
    levels = {key: Level(key, key in selected) for key in LEVELS}
    if levels["L0"].selected:
        level0(ctx, levels["L0"])
    targets = []
    if levels["L1"].selected or levels["L3"].selected:
        holder = levels["L1"] if levels["L1"].selected else levels["L3"]
        targets = l1_targets(ctx, holder, stage_cmd)
    if levels["L1"].selected:
        pairs = chain_pairs_for(ctx, levels["L1"], targets, stage_cmd)
        level1(ctx, levels["L1"], targets, pairs)
    if levels["L2"].selected:
        level2(ctx, levels["L2"])
    if levels["L3"].selected:
        if not targets:
            levels["L3"].missing("no stage under test was found; the error exits of real stages did not run")
        level3(ctx, levels["L3"], targets)
    if levels["L4"].selected:
        level4(ctx, levels["L4"])
    if levels["L5"].selected:
        level5(ctx, levels["L5"])
    return [levels[key] for key in LEVELS]


def finish(levels, mode, seconds, as_json, root, allow_absent_peer=False):
    docs = [lv.document() for lv in levels]
    failed = [d for d in docs if d["status"] == "fail"]
    not_run = [d for d in docs if d["status"] == "not run"]
    passed = [d for d in docs if d["status"] == "pass"]
    peers = sorted(set().union(*(lv.peers for lv in levels)))
    excused = (allow_absent_peer and not failed and bool(not_run)
               and all(lv.only_absent_peers() for lv in levels if lv.status() == "not run"))
    exit_code = EXIT_FAIL if failed else (EXIT_NOT_RUN if not_run and not excused else EXIT_OK)
    verdict = {EXIT_OK: "pass", EXIT_FAIL: "fail", EXIT_NOT_RUN: "not run"}[exit_code]
    if excused:
        verdict = f"pass with absent peer {', '.join(peers)} (every NOT RUN part needs it)"
    summary = f"{len(passed)} passed, {len(failed)} failed, {len(not_run)} not run"
    if as_json:
        doc = {"ok": exit_code == EXIT_OK, "contract": CONTRACT, "mode": mode, "levels": docs,
               "seconds": round(seconds, 1), "absent_peers": peers}
        if exit_code != EXIT_OK:
            first = (failed or not_run)[0]
            detail = (first["failures"] or first["not_run"] or ["no check ran"])[0]
            doc["error"] = {"code": "stage_failed" if failed else "tool_missing", "stage": first["level"],
                            "exit_code": None, "message": f"{summary}; first: {first['level']} {detail}"[:300],
                            "stderr_tail": []}
        print(json.dumps(doc, indent=2))
        return exit_code
    print(f"conformance {mode}: {root}")
    for d in docs:
        counts = f"{d['passed']} passed" + (f", {d['failed']} failed" if d["failed"] else "")
        status = d["status"].upper() if d["status"] in ("fail", "not run") else d["status"]
        print(f"{d['level']} {d['name']:<18} {status:<13} {counts if d['status'] != 'not selected' else ''}".rstrip())
        for line in d["failures"][:MAX_TEXT_FAILURES]:
            print(f"   FAIL {line}")
        if d["failed"] > MAX_TEXT_FAILURES:
            print(f"   ... {d['failed'] - MAX_TEXT_FAILURES} more failures (--json lists up to {MAX_JSON_FAILURES})")
        for line in d["not_run"]:
            print(f"   NOT RUN {line}")
    print(f"result: {verdict} (exit {exit_code}): {summary}")
    return exit_code


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["fixture"]:
        return fixture_main(argv[1:])
    if argv[:1] == ["meter"]:
        return meter_main(argv[1:])
    args = build_parser().parse_args(argv)
    root = os.path.abspath(args.root or DEFAULT_ROOT)
    mode = "full" if args.full else "quick"
    try:
        cases = load_cases(root)
    except (OSError, ValueError) as exc:
        print(f"error: bad_input: {CASES_REL} could not be read: {exc}", file=sys.stderr)
        return 4
    if args.bench:
        start = time.monotonic()
        ctx = Context(root, cases, "bench", args.deadline)
        lv = Level("L2", True)
        lv.key, lv.name = "bench", "wrappers on chains"
        try:
            rows = run_bench(ctx, lv)
        finally:
            shutil.rmtree(ctx.tmp, ignore_errors=True)
        if rows and lv.status() == "pass":
            path = write_bench(root, cases["bench"]["columns"], rows)
            print(f"wrote {os.path.relpath(path, root)}: {len(rows)} rows", file=sys.stderr)
        else:
            lv.missing("docs/handoff-bench.tsv was not written: the bench did not run every row")
        return finish([lv], "bench", time.monotonic() - start, args.as_json, root)
    selected = set(args.levels or LEVELS)
    if args.stage is not None:
        try:
            words = split_command(args.stage)
        except ValueError as exc:
            print(f"error: usage: {exc}", file=sys.stderr)
            return EXIT_USAGE
        if words:
            args.stage = join_command([resolve_program(words[0])] + words[1:])
        selected &= set(STAGE_LEVELS)
    start = time.monotonic()
    ctx = Context(root, cases, mode, args.deadline)
    try:
        levels = run_levels(ctx, selected, args.stage)
    finally:
        shutil.rmtree(ctx.tmp, ignore_errors=True)
    return finish(levels, mode, time.monotonic() - start, args.as_json, root, args.allow_absent_peer)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    sys.exit(main())
