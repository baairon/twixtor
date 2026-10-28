#!/usr/bin/env python3
"""tests for tools/apply.py. stdlib unittest, no network.

    python tools/test_apply.py

needs ffmpeg and ffprobe, and a built vfx (cargo build --release -p fx-cli, or
VFX_PATH). it generates its own short clip with ffmpeg in a temp dir and skips
itself with a note when a tool is missing.

every test holds for any effect, so they keep passing as the stubs gain real math.
"""

import argparse
import contextlib
import hashlib
import inspect
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from fractions import Fraction
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
APPLY = os.path.join(HERE, "apply.py")

# no bytecode next to the tools, here or in the script the suite runs
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, HERE)
import apply as tool  # noqa: E402

FPS = 30
TMP = None
CLIP = None
VFX = None
EFFECT = None
# the hidden self-test effects of vfx: an identity-like temporal effect (output frame n is input
# frame n + offset, with an optional overlay layer) and one that integrates a parameter curve
TEMPORAL = None
INTEGRATE = None


def ff(*args):
    cmd = [tool.ffmpeg_path(), "-v", "error", "-y"] + list(args)
    subprocess.run(cmd, shell=False, check=True, capture_output=True)


def make_clip(path, size="160x120", seconds=1, audio=True, pix_fmt="yuv420p", vf=None, extra=()):
    inputs = ["-f", "lavfi", "-i", f"testsrc=size={size}:rate={FPS}:duration={seconds}"]
    if audio:
        inputs += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    cmd = inputs + (["-vf", vf, "-fps_mode", "passthrough"] if vf else [])
    cmd += ["-c:v", "libx264", "-pix_fmt", pix_fmt]
    cmd += ["-c:a", "aac", "-shortest"] if audio else []
    ff(*(cmd + list(extra) + [path]))


def listed(*extra):
    out = subprocess.run([VFX, "list", "--json", *extra], shell=False, capture_output=True, text=True)
    return json.loads(out.stdout) if out.returncode == 0 else []


def setUpModule():
    global TMP, CLIP, VFX, EFFECT, TEMPORAL, INTEGRATE
    try:
        tool.require_ffmpeg()
        VFX = tool.find_vfx(None)
    except SystemExit:
        raise unittest.SkipTest("needs ffmpeg, ffprobe and a built vfx (cargo build --release -p fx-cli, or VFX_PATH)")
    # the first listed frame effect, so the generic tests follow the real plugins; the hidden
    # frame effect when the plugins have all become temporal
    frame_effects = [e["id"] for e in listed() if e.get("kind", "frame") == "frame"]
    everything = {e["id"] for e in listed("--all")}
    EFFECT = frame_effects[0] if frame_effects else "selftest-invert"
    TEMPORAL = "selftest-shift" if "selftest-shift" in everything else None
    INTEGRATE = "selftest-integrate" if "selftest-integrate" in everything else None
    TMP = tempfile.mkdtemp(prefix="apply_test_")
    CLIP = os.path.join(TMP, "clip.mp4")
    make_clip(CLIP)


def tearDownModule():
    if TMP:
        shutil.rmtree(TMP, ignore_errors=True)


def slurp(path):
    with open(path, "rb") as f:
        return f.read()


def sha(path):
    return hashlib.sha256(slurp(path)).hexdigest()


def snapshot(root):
    """every file and directory under root with its size and mtime."""
    seen = []
    for base, dirs, files in os.walk(root):
        for name in dirs + files:
            full = os.path.join(base, name)
            st = os.stat(full)
            seen.append((os.path.relpath(full, root), st.st_size, st.st_mtime_ns))
    return sorted(seen)


def apply(*args, tmp=None):
    """run apply.py; `tmp` points the temp directory of apply.py, ffmpeg and vfx at a directory of the test."""
    env = dict(os.environ, VFX_PATH=VFX, PYTHONDONTWRITEBYTECODE="1")
    if tmp:
        env.update(TMP=tmp, TEMP=tmp, TMPDIR=tmp)
    return subprocess.run([sys.executable, APPLY] + list(args), shell=False, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", env=env)


def make_changing_clip(path, seconds=1):
    """every frame a flat colour of its own, all bright (each channel 100..249), plus a tone."""
    expr = ":".join(f"{c}='{base}+mod(N*{step}+{off},150)'" for c, base, step, off in
                    (("r", 100, 37, 0), ("g", 100, 61, 20), ("b", 100, 83, 40)))
    graph = f"color=c=black:s=160x120:r={FPS}:d={seconds},format=gbrp,geq={expr}"
    ff("-f", "lavfi", "-i", graph, "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path)


def psnr(a, b, frame_a=None, frame_b=None):
    """average PSNR of two videos, or of frame `frame_a` of the first and `frame_b` of the second."""
    if frame_a is None:
        graph = "[0:v][1:v]psnr"
    else:
        graph = (f"[0:v]select='eq(n,{frame_a})',setpts=PTS-STARTPTS[a];"
                 f"[1:v]select='eq(n,{frame_b})',setpts=PTS-STARTPTS[b];[a][b]psnr")
    cmd = [tool.ffmpeg_path(), "-hide_banner", "-v", "info", "-i", a, "-i", b, "-lavfi", graph, "-f", "null", "-"]
    log = subprocess.run(cmd, shell=False, capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
    match = re.search(r"average:([0-9.]+|inf)", log)
    assert match, log
    return float(match.group(1))


def mean_luma(path, frame):
    """mean of the luma plane of one frame, 0..255."""
    cmd = [tool.ffmpeg_path(), "-v", "error", "-i", path, "-vf", f"select='eq(n,{frame})',format=gray",
           "-frames:v", "1", "-f", "rawvideo", "-"]
    data = subprocess.run(cmd, shell=False, capture_output=True).stdout
    assert data, f"no frame {frame} in {path}"
    return sum(data) / len(data)


class Workspace(unittest.TestCase):
    """a scratch directory with a copy of the clip as the input."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(dir=TMP)
        self.src = os.path.join(self.dir, "in.mp4")
        shutil.copy(CLIP, self.src)
        self.src_hash = sha(self.src)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.dir, name)

    def assertClean(self):
        """no partial files left behind, and the input is as it was."""
        leftovers = [n for _, _, fs in os.walk(self.dir) for n in fs if ".partial-" in n]
        self.assertEqual(leftovers, [])
        self.assertEqual(sha(self.src), self.src_hash)

    def assertSameFacts(self, src, out, rate=FPS):
        a, b = tool.probe(src), tool.probe(out)
        self.assertIsNotNone(b, "the output cannot be probed")
        self.assertEqual(tool.display_size(a), tool.display_size(b))
        self.assertAlmostEqual(a["fps"], b["fps"], delta=0.01)
        self.assertLessEqual(abs(a["duration"] - b["duration"]), 1.0 / rate + 1e-3)
        self.assertEqual(a["has_audio"], b["has_audio"])
        self.assertEqual((a["acodec"], a["channels"]), (b["acodec"], b["channels"]))
        self.assertEqual(b["vcodec"], "h264")
        self.assertEqual(b["pix_fmt"], "yuv420p")


class ApplyTests(Workspace):
    # ---- dry run -----------------------------------------------------------------

    def test_dry_run_writes_nothing(self):
        before = snapshot(self.dir)
        proc = apply(self.src, EFFECT, "--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(snapshot(self.dir), before)
        self.assertEqual(proc.stdout.count("[cmd]"), 3, proc.stdout)
        self.assertIn("[dry-run] nothing was written", proc.stdout)
        for want in ("libx264", "-crf 18", "yuv420p", "-c:a copy", "rgba64le", " apply " + EFFECT):
            self.assertIn(want, proc.stdout)

        nested = self.path(os.path.join("not", "yet", "out.mp4"))
        proc = apply(self.src, EFFECT, "--dry-run", "-o", nested, "--depth", "32")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("gbrapf32le", proc.stdout)
        self.assertFalse(os.path.exists(self.path("not")), "a dry run must not create directories")
        self.assertEqual(snapshot(self.dir), before)

    def test_dry_run_still_checks_the_effect_and_params(self):
        bad = apply(self.src, EFFECT, "--dry-run", "--param", "no_such_param=1")
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("no_such_param", bad.stderr)

    def test_dry_run_json_is_one_machine_readable_result(self):
        proc = apply(self.src, EFFECT, "--dry-run", "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertTrue(data["dry_run"])
        self.assertEqual((data["effect"], data["depth"], data["pix_fmt"]), (EFFECT, 16, "rgba64le"))
        self.assertEqual(data["size"], "160x120")
        self.assertEqual(len(data["commands"]), 3)
        self.assertEqual(data["outputs"], [])

    # ---- a real run --------------------------------------------------------------

    def test_real_run_keeps_duration_size_fps_and_audio_at_every_depth(self):
        for depth in (8, 16, 32):
            with self.subTest(depth=depth):
                out = self.path(f"out{depth}.mp4")
                proc = apply(self.src, EFFECT, "--depth", str(depth), "-o", out)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertTrue(proc.stdout.rstrip().splitlines()[-1].startswith(os.path.abspath(out)), proc.stdout)
                self.assertSameFacts(self.src, out)
                self.assertClean()

    def test_json_result_reports_the_probed_output(self):
        out = self.path("out.mp4")
        proc = apply(self.src, EFFECT, "--json", "-o", out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        probe = data["outputs"][0]["probe"]
        self.assertEqual((probe["width"], probe["height"]), (160, 120))
        self.assertTrue(probe["has_audio"])
        self.assertEqual(data["outputs"][0]["path"], os.path.abspath(out))
        self.assertFalse(data["dry_run"])

    def test_default_output_sits_next_to_the_input_and_never_replaces_itself(self):
        first = apply(self.src, EFFECT)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertTrue(os.path.isfile(self.path(f"in_{EFFECT}.mp4")))
        second = apply(self.src, EFFECT)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertTrue(os.path.isfile(self.path(f"in_{EFFECT}_1.mp4")))
        self.assertClean()

    def test_the_picture_survives_when_the_effect_is_an_identity(self):
        probe = bytes(range(256)) * 16  # 16x16 rgba, every byte value
        run = subprocess.run([VFX, "apply", EFFECT, "--width", "16", "--height", "16", "--depth", "8"],
                             shell=False, input=probe, capture_output=True)
        if run.returncode != 0 or run.stdout != probe:
            self.skipTest(f"{EFFECT} is not an identity, so there is nothing to compare against")
        out = self.path("out.mp4")
        self.assertEqual(apply(self.src, EFFECT, "-o", out).returncode, 0)
        cmd = [tool.ffmpeg_path(), "-hide_banner", "-v", "info", "-i", self.src, "-i", out,
               "-lavfi", "[0:v][1:v]psnr", "-f", "null", "-"]
        log = subprocess.run(cmd, shell=False, capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
        match = re.search(r"average:([0-9.]+|inf)", log)
        self.assertIsNotNone(match, log)
        self.assertGreater(float(match.group(1)), 30.0, "frames came out misaligned or recoloured")

    # ---- never overwrite ---------------------------------------------------------

    def test_input_is_never_overwritten(self):
        for flags in ([], ["--force"]):
            proc = apply(self.src, EFFECT, "-o", self.src, *flags)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("input", proc.stderr)
            self.assertEqual(sha(self.src), self.src_hash)
        self.assertEqual(apply(self.src, EFFECT, "-o", self.path("x.mp4"), "--force").returncode, 0)
        self.assertClean()

    def test_force_semantics(self):
        out = self.path("out.mp4")
        with open(out, "wb") as f:
            f.write(b"keep me")

        refused = apply(self.src, EFFECT, "-o", out)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("--force", refused.stderr)
        self.assertEqual(slurp(out), b"keep me")

        failed = apply(self.src, EFFECT, "-o", out, "--force", "--param", "no_such_param=1")
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(slurp(out), b"keep me")

        replaced = apply(self.src, EFFECT, "-o", out, "--force")
        self.assertEqual(replaced.returncode, 0, replaced.stderr)
        self.assertNotEqual(slurp(out), b"keep me")
        self.assertSameFacts(self.src, out)
        self.assertClean()

    def run_main_with_a_failing_effect(self, out, *flags):
        """the real main(), with the effect stage swapped for one that dies after reading a little."""
        dies = [sys.executable, "-c", "import sys; sys.stdin.buffer.read(100); sys.exit(3)"]
        real = tool.build_commands

        def swapped(*a, **k):
            decode, _, encode = real(*a, **k)
            return decode, dies, encode

        argv = ["apply.py", self.src, EFFECT, "-o", out, *flags]
        stderr = io.StringIO()
        with mock.patch.object(sys, "argv", argv), mock.patch.object(tool, "build_commands", swapped), \
                contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                tool.main()
        return caught.exception.code, stderr.getvalue()

    def test_a_pipeline_that_fails_midway_leaves_the_old_output_and_no_partial(self):
        out = self.path("out.mp4")
        with open(out, "wb") as f:
            f.write(b"keep me")
        code, err = self.run_main_with_a_failing_effect(out, "--force")
        self.assertNotEqual(code, 0)
        self.assertIn("vfx exited with code 3", err)
        self.assertEqual(slurp(out), b"keep me")
        self.assertClean()

        fresh = self.path("fresh.mp4")
        code, _ = self.run_main_with_a_failing_effect(fresh)
        self.assertNotEqual(code, 0)
        self.assertFalse(os.path.exists(fresh))
        self.assertClean()

    def test_a_dying_stage_cannot_hang_the_pipeline(self):
        """each stage's exit code is reported and no stage is left waiting on a dead neighbour."""
        python = sys.executable
        decode = [python, "-c", "import sys; sys.stdout.buffer.write(b'x' * 5000000)"]
        effect = [python, "-c", "import sys; sys.stdin.buffer.read(10); sys.exit(3)"]
        encode = [python, "-c", "import sys; sys.stdin.buffer.read()"]
        scratch = tempfile.mkdtemp(dir=self.dir)
        box = {}
        worker = threading.Thread(target=lambda: box.update(result=tool.run_pipeline(decode, effect, encode, scratch)))
        worker.start()
        worker.join(60)
        self.assertFalse(worker.is_alive(), "the pipeline hung")
        codes, logs = box["result"]
        self.assertEqual(codes["vfx"], 3)
        self.assertNotEqual(codes["decode"], 0)
        self.assertEqual(codes["encode"], 0)
        self.assertEqual(set(logs), {"decode", "vfx", "encode"})

    # ---- refusing early ----------------------------------------------------------

    def test_an_odd_frame_size_encodes_yuv444p_at_its_exact_size(self):
        """yuv420p needs even sides; of the chroma layouts only 4:4:4 has a crop unit of one pixel, so a size with
        an odd side encodes yuv444p at that size, and the plan says so."""
        odd = self.path("odd.mp4")
        make_clip(odd, size="101x61", pix_fmt="yuv444p")
        out = self.path("odd_out.mp4")
        proc = apply(odd, EFFECT, "-o", out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("101x61, 30.000 fps, 16 bpc (rgba64le), encoded yuv444p (the size has an odd side), decode",
                      proc.stdout.splitlines()[0])
        got = tool.probe(out)
        self.assertEqual((got["width"], got["height"], got["vcodec"], got["pix_fmt"]), (101, 61, "h264", "yuv444p"))
        self.assertEqual((got["acodec"], got["channels"], got["color_space"]), ("aac", 1, "smpte170m"))
        self.assertLessEqual(abs(got["duration"] - tool.probe(odd)["duration"]), 1.0 / FPS + 1e-3)
        dry = json.loads(apply(odd, EFFECT, "--dry-run", "--json").stdout)
        self.assertEqual(dry["encode"], {"codec": "libx264", "pix_fmt": "yuv444p"})
        self.assertNotIn("hdr", dry)
        self.assertClean()

    def test_lossless_keeps_the_rgb_the_effect_wrote_and_allows_odd_sizes(self):
        """--lossless writes ffv1 in RGB: what vfx wrote comes back bit for bit, which libx264 yuv420p cannot do."""
        rgb = self.path("rgb_src.mkv")
        make_clip(rgb, size="101x61", audio=False, pix_fmt="rgb24", extra=("-c:v", "ffv1"))
        for depth, wire in ((8, "rgba"), (16, "rgba64le")):
            with self.subTest(depth=depth):
                out = self.path(f"lossless{depth}.mkv")
                proc = apply(rgb, "selftest-invert", "--depth", str(depth), "--lossless", "-o", out)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                probe = subprocess.run([tool.ffprobe_path(), "-v", "error", "-select_streams", "v:0", "-show_entries",
                                        "stream=codec_name,pix_fmt,width,height", "-of", "csv=p=0", out],
                                       shell=False, capture_output=True, text=True).stdout.strip()
                codec, width, height, pix_fmt = probe.split(",")
                self.assertEqual((codec, width, height), ("ffv1", "101", "61"), probe)
                self.assertTrue(pix_fmt.startswith(("bgra", "gbrap")), probe)
                raw_in, raw_expect, raw_back = (self.path(n) for n in ("in.raw", "expect.raw", "back.raw"))
                ff("-i", rgb, "-an", "-pix_fmt", wire, "-f", "rawvideo", raw_in)
                run = subprocess.run([VFX, "apply", "selftest-invert", "--width", "101", "--height", "61",
                                      "--depth", str(depth), "--in", raw_in, "--out", raw_expect],
                                     shell=False, capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                ff("-i", out, "-an", "-pix_fmt", wire, "-f", "rawvideo", raw_back)
                with open(raw_expect, "rb") as a, open(raw_back, "rb") as b:
                    self.assertTrue(a.read() == b.read(), "the lossless file does not hold what vfx wrote")
                self.assertClean()

    def test_lossless_refuses_float_and_defaults_to_mkv(self):
        proc = apply(self.src, EFFECT, "--depth", "32", "--lossless", "-o", self.path("f.mkv"))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("ffv1", proc.stderr)
        self.assertFalse(os.path.exists(self.path("f.mkv")))
        dry = apply(self.src, EFFECT, "--depth", "8", "--lossless", "--dry-run", "--json")
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertIn(".mkv", dry.stdout)

    def test_unknown_effect_and_param_are_rejected_before_any_work(self):
        before = snapshot(self.dir)
        for args, word in ((("no-such-effect",), "unknown effect"),
                           ((EFFECT, "--param", "no_such_param=1"), "no_such_param")):
            proc = apply(self.src, *args)
            self.assertNotEqual(proc.returncode, 0, args)
            self.assertIn(word, proc.stderr)
            self.assertEqual(snapshot(self.dir), before)
        # a malformed --param is argparse's own usage error
        self.assertEqual(apply(self.src, EFFECT, "--param", "novalue").returncode, 2)

    def test_missing_and_non_video_inputs(self):
        self.assertNotEqual(apply(self.path("nope.mp4"), EFFECT).returncode, 0)
        wav = self.path("tone.wav")
        ff("-f", "lavfi", "-i", "sine=frequency=440:duration=1", wav)
        proc = apply(wav, EFFECT, "-o", self.path("x.mp4"))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("no video", proc.stderr)
        self.assertFalse(os.path.exists(self.path("x.mp4")))

    # ---- inputs of other shapes --------------------------------------------------

    def test_a_clip_without_audio_stays_without_audio(self):
        silent = self.path("silent.mp4")
        make_clip(silent, audio=False)
        out = self.path("silent_out.mp4")
        proc = apply(silent, EFFECT, "-o", out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse(tool.probe(out)["has_audio"])
        self.assertSameFacts(silent, out)

    def test_a_rotated_clip_comes_out_upright_with_the_swapped_size(self):
        rotated = self.path("rotated.mp4")
        try:
            ff("-display_rotation", "90", "-i", CLIP, "-c", "copy", rotated)
        except subprocess.CalledProcessError:
            self.skipTest("this ffmpeg cannot write a display rotation")
        if tool.probe(rotated)["rotation"] != 90:
            self.skipTest("the display rotation was not kept")
        out = self.path("rotated_out.mp4")
        proc = apply(rotated, EFFECT, "-o", out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        info = tool.probe(out)
        self.assertEqual((info["width"], info["height"], info["rotation"]), (120, 160, 0))
        self.assertSameFacts(rotated, out)

    def test_a_variable_frame_rate_clip_keeps_its_duration(self):
        vfr = self.path("vfr.mp4")
        make_clip(vfr, seconds=2, vf="select='lt(mod(n,5),3)'")
        info = tool.probe(vfr)
        if not info["vfr"]:
            self.skipTest("could not build a variable frame rate clip")
        out = self.path("vfr_out.mp4")
        proc = apply(vfr, EFFECT, "-o", out)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertSameFacts(vfr, out, rate=float(info["rate"]))

    def test_color_tags_follow_the_source_or_the_size_convention(self):
        hd = self.path("hd.mp4")
        make_clip(hd, size="1280x720", seconds=1, audio=False)
        out = self.path("hd_out.mp4")
        self.assertEqual(apply(hd, EFFECT, "-o", out, "--depth", "8").returncode, 0)
        self.assertEqual(tool.probe(out)["color_space"], "bt709")

        sd_out = self.path("sd_out.mp4")
        self.assertEqual(apply(self.src, EFFECT, "-o", sd_out).returncode, 0)
        self.assertEqual(tool.probe(sd_out)["color_space"], "smpte170m")

        tagged = self.path("tagged.mp4")
        make_clip(tagged, audio=False, extra=["-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709"])
        tagged_out = self.path("tagged_out.mp4")
        self.assertEqual(apply(tagged, EFFECT, "-o", tagged_out).returncode, 0)
        got = tool.probe(tagged_out)
        self.assertEqual((got["color_space"], got["color_primaries"], got["color_transfer"]), ("bt709",) * 3)


class DeviceTests(Workspace):
    """--device: checked like vfx checks it, passed through to the vfx of a temporal effect, refused by
    a frame effect; a run without it is the run of before."""

    def setUp(self):
        super().setUp()
        if not TEMPORAL:
            self.skipTest("this vfx has no temporal self-test effects (build it from the current workspace)")

    def test_the_device_values_are_the_ones_vfx_takes(self):
        for text, want in (("cpu", "cpu"), ("gpu", "gpu"), ("gpu=0", "gpu=0"), ("gpu=12", "gpu=12"), (" gpu=3 ", "gpu=3"),
                           ("gpu=4294967295", "gpu=4294967295")):
            self.assertEqual(tool.device_arg(text), want)
        for bad in ("", "opencl", "GPU", "Cpu", "gpu=", "gpu=-1", "gpu=+1", "gpu=1.5", "gpu=x", "gpu=\u0663",
                    "gpu=4294967296", "cpu=0", "gpu 1"):
            with self.assertRaises(argparse.ArgumentTypeError, msg=repr(bad)):
                tool.device_arg(bad)

    def test_only_the_vfx_of_a_temporal_effect_gets_the_device(self):
        info = facts(nb_frames=30, audio_duration=1.0, video_index=0, vfr=False, color_space=None,
                     color_primaries=None, color_transfer=None)
        temporal = {"kind": "temporal", "layers": []}
        for device in ("cpu", "gpu=1"):
            args = argparse.Namespace(input="in.mp4", effect="fx", depth=8, param=["a=1"], frames=None, layer=[],
                                      device=device)
            plan = tool.make_plan(args, info, temporal, [], "/tmp/s")
            effect = tool.build_commands(args, info, "vfx", "out.mp4", plan)[1]
            self.assertEqual(effect[effect.index("--device") + 1], device)
            self.assertEqual(effect.count("--device"), 1)
            # a frame effect never gets it (the run refuses the flag before building commands)
            self.assertNotIn("--device", tool.build_commands(args, info, "vfx", "out.mp4")[1])
        # without the flag, or from a caller that predates it, the command is the one of before
        for args in (argparse.Namespace(input="in.mp4", effect="fx", depth=8, param=[], frames=None, layer=[], device=None),
                     argparse.Namespace(input="in.mp4", effect="fx", depth=8, param=[], frames=None, layer=[])):
            plan = tool.make_plan(args, info, temporal, [], "/tmp/s")
            self.assertNotIn("--device", tool.build_commands(args, info, "vfx", "out.mp4", plan)[1])

    def test_a_default_dry_run_is_unchanged_and_cpu_passes_through(self):
        plain = apply(self.src, TEMPORAL, "--param", "offset=1", "--dry-run", "--json", tmp=self.dir)
        cpu = apply(self.src, TEMPORAL, "--param", "offset=1", "--device", "cpu", "--dry-run", "--json", tmp=self.dir)
        self.assertEqual(plain.returncode, 0, plain.stderr)
        self.assertEqual(cpu.returncode, 0, cpu.stderr)
        a, b = json.loads(plain.stdout), json.loads(cpu.stdout)
        self.assertNotIn("--device", " ".join(a["commands"]))
        vfx_cmds = [c for c in b["commands"] if " apply " + TEMPORAL in c]
        self.assertEqual(len(vfx_cmds), 1)
        self.assertIn(" --device cpu", vfx_cmds[0])
        # the device is the only difference: same plan, same JSON, the other commands untouched (the
        # partial output path carries the process id)
        for data in (a, b):
            data["commands"] = [re.sub(r"\.partial-[0-9]+\.", ".partial-PID.", c) for c in data["commands"]]
        b["commands"] = [c.replace(" --device cpu", "") for c in b["commands"]]
        self.assertEqual(a, b)
        text = apply(self.src, TEMPORAL, "--param", "offset=1", "--device", "cpu", "--dry-run", tmp=self.dir)
        self.assertEqual(text.returncode, 0, text.stderr)
        self.assertEqual(text.stdout.count("--device cpu"), 1)

    def test_a_real_run_on_the_cpu_device_writes_the_frames_of_a_run_without_it(self):
        outs = []
        for extra in ((), ("--device", "cpu")):
            out = self.path(f"out{len(outs)}.mkv")
            proc = apply(self.src, TEMPORAL, "--param", "offset=1", "--lossless", "--depth", "8", *extra, "-o", out,
                         tmp=self.dir)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            raw = self.path(f"out{len(outs)}.rgba")
            ff("-i", out, "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgba", raw)
            outs.append(slurp(raw))
        self.assertTrue(outs[0])
        self.assertEqual(outs[0], outs[1])
        self.assertClean()

    def test_a_device_vfx_cannot_serve_is_refused_before_any_work(self):
        # gpu=N is strict: a number no machine lists is refused by the preflight, nothing is written
        before = snapshot(self.dir)
        device = "gpu=4000"
        proc = apply(self.src, TEMPORAL, "--device", device, "-o", self.path("never.mp4"), tmp=self.dir)
        self.assertNotEqual(proc.returncode, 0, device)
        self.assertIn("unsupported: --device gpu=4000: ", proc.stderr)
        self.assertEqual(snapshot(self.dir), before, device)
        # under frames/2 it is the unsupported error of the vfx stage
        proc = apply(self.src, TEMPORAL, "--device", device, "--contract", "frames/2", "--json", "--dry-run",
                     tmp=self.dir)
        self.assertEqual(proc.returncode, 3, proc.stderr)
        error = json.loads(proc.stdout)["error"]
        self.assertEqual((error["code"], error["stage"], error["exit_code"]), ("unsupported", "vfx", 3))
        self.assertTrue(error["message"].startswith("vfx: unsupported: --device gpu=4000: "), error["message"])
        self.assertEqual(snapshot(self.dir), before, device)

    def test_gpu_renders_the_frames_of_the_cpu(self):
        # gpu: a device that passes the gate, or the cpu after one note; the frames are the same
        outs = []
        for extra in (("--device", "cpu"), ("--device", "gpu")):
            out = self.path(f"dev{len(outs)}.mkv")
            proc = apply(self.src, TEMPORAL, "--param", "offset=1", "--lossless", "--depth", "8", *extra, "-o", out,
                         tmp=self.dir)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            raw = self.path(f"dev{len(outs)}.rgba")
            ff("-i", out, "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgba", raw)
            outs.append(slurp(raw))
        self.assertTrue(outs[0])
        self.assertEqual(outs[0], outs[1])
        self.assertClean()

    def test_a_frame_effect_refuses_the_device_flag(self):
        proc = apply(self.src, EFFECT, "--device", "cpu", "--dry-run", tmp=self.dir)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(f"{EFFECT} is a frame effect, --device only apply to temporal effects", proc.stderr)

    def test_bad_values_and_abbreviations_are_usage_errors(self):
        missing = self.path("missing.mp4")
        for flags in (["--device", "opencl"], ["--device", "gpu="], ["--device"]):
            proc = apply(missing, TEMPORAL, *flags, tmp=self.dir)
            self.assertEqual(proc.returncode, 2, flags)
        # the flag matches only in full, so its prefixes mean what they meant before it existed
        for flags in (["--dev", "cpu"], ["--devi", "cpu"], ["--devic", "gpu"]):
            proc = apply(missing, TEMPORAL, *flags, tmp=self.dir)
            self.assertEqual((proc.returncode, proc.stderr.strip().splitlines()[-1]),
                             (2, "apply.py: error: unrecognized arguments: " + " ".join(flags)))
        for flags in (["--de", "8"], ["--device", "cpu"]):
            proc = apply(missing, TEMPORAL, *flags, tmp=self.dir)
            self.assertNotEqual(proc.returncode, 2, proc.stderr)


def facts(**over):
    base = {"has_video": True, "has_audio": True, "width": 160, "height": 120, "rotation": 0, "fps": 30.0,
            "rate": Fraction(30), "duration": 1.0, "acodec": "aac", "channels": 2,
            "nb_frames": None, "video_duration": None, "audio_duration": None}
    base.update(over)
    return base


class TemporalTests(Workspace):
    """the temporal path: --fps as a fraction, --frames, --layer, the spool and the temp files."""

    def setUp(self):
        super().setUp()
        if not TEMPORAL:
            self.skipTest("this vfx has no temporal self-test effects (build it from the current workspace)")
        self.other = self.path("other.mp4")
        make_clip(self.other, seconds=2, audio=False, vf="negate")
        # every frame a flat colour of its own, all of them bright: shifts and black frames are easy to see
        self.changing = self.path("changing.mp4")
        make_changing_clip(self.changing)

    def assertNoTempLeft(self):
        """the scratch directories and spool files all live in self.dir (the tests point TMP there)."""
        names = [n for n in os.listdir(self.dir) if n.startswith(("apply_", "vfx-spool-"))]
        self.assertEqual(names, [])

    def assertDuration(self, out, seconds, window=0.1):
        got = tool.probe(out)["duration"]
        self.assertIsNotNone(got)
        self.assertLess(abs(got - seconds), window + 1.0 / FPS, f"{got}s, expected {seconds}s")

    # ---- the plan --------------------------------------------------------------

    def test_dry_run_plans_the_layer_decode_and_the_temp_files_and_writes_nothing(self):
        before = snapshot(self.dir)
        proc = apply(self.src, TEMPORAL, "--layer", f"overlay={self.other}", "--frames", "20", "--dry-run",
                     "--param", "offset=2", tmp=self.dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(snapshot(self.dir), before, "a dry run writes nothing, not even a scratch directory")
        out = proc.stdout
        self.assertEqual(out.count("[cmd]"), 4, out)
        self.assertIn("[dry-run] nothing was written", out)
        self.assertGreaterEqual(out.count("[temp]"), 2, out)
        self.assertIn("layer_overlay.rgba64le", out)
        for want in (" apply " + TEMPORAL, "--fps 30/1", "--frames 20", "--tmp-dir", "--layer", "overlay=",
                     "--param offset=2", "rgba64le", "libx264", "-c:a copy"):
            self.assertIn(want, out)
        # 20 frames of 30 are shorter than the 1 s of audio, so the audio copy stops with the video
        self.assertIn("-shortest", out)

    def test_dry_run_json_carries_the_new_fields(self):
        proc = apply(self.src, TEMPORAL, "--layer", f"overlay={self.other}", "--frames", "45", "--dry-run", "--json",
                     tmp=self.dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertEqual((data["kind"], data["frames"], data["fps_exact"]), ("temporal", 45, "30/1"))
        self.assertEqual(data["input_frames"], 30)
        self.assertEqual([layer["id"] for layer in data["layers"]], ["overlay"])
        self.assertTrue(data["layers"][0]["temp"].endswith("layer_overlay.rgba64le"))
        self.assertEqual(len(data["temp_files"]), 2)
        self.assertEqual(len(data["commands"]), 4)
        # 45 frames outlast the 1 s of audio, which just ends early
        self.assertEqual(data["audio"], {"plan": "ends-early", "shortest": False})
        self.assertNotIn("-shortest", " ".join(data["commands"]))
        self.assertTrue(data["dry_run"])
        self.assertEqual(data["outputs"], [])
        frame = json.loads(apply(self.src, EFFECT, "--dry-run", "--json", tmp=self.dir).stdout)
        self.assertEqual((frame["kind"], frame["layers"], frame["temp_files"], frame["fps_exact"]), ("frame", [], [], "30/1"))

    def test_the_default_frame_count_is_left_to_vfx_and_an_explicit_one_is_passed(self):
        plain = json.loads(apply(self.src, TEMPORAL, "--dry-run", "--json", tmp=self.dir).stdout)
        self.assertEqual(plain["input_frames"], 30)
        self.assertEqual(plain["frames"], 30, "the container says 30 and the rate is constant")
        self.assertIn("--frames 30", " ".join(plain["commands"]))
        self.assertEqual(plain["audio"], {"plan": None, "shortest": False})
        # a variable frame rate input is resampled while decoding: its count is not known up front
        vfr = self.path("vfr.mp4")
        make_clip(vfr, seconds=2, vf="select='lt(mod(n,5),3)'")
        if not tool.probe(vfr)["vfr"]:
            self.skipTest("could not build a variable frame rate clip")
        data = json.loads(apply(vfr, TEMPORAL, "--dry-run", "--json", tmp=self.dir).stdout)
        self.assertIsNone(data["frames"])
        self.assertNotIn("--frames", " ".join(data["commands"]))
        data = json.loads(apply(vfr, TEMPORAL, "--frames", "10", "--dry-run", "--json", tmp=self.dir).stdout)
        self.assertEqual(data["frames"], 10)

    # ---- a real run ------------------------------------------------------------

    def test_real_run_keeps_size_fps_duration_and_audio_and_leaves_no_temp_file(self):
        for depth in (8, 16, 32):
            with self.subTest(depth=depth):
                out = self.path(f"t{depth}.mp4")
                proc = apply(self.src, TEMPORAL, "--depth", str(depth), "-o", out, tmp=self.dir)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertTrue(proc.stdout.rstrip().splitlines()[-1].startswith(os.path.abspath(out)), proc.stdout)
                self.assertSameFacts(self.src, out)
                self.assertClean()
                self.assertNoTempLeft()

    def test_the_picture_survives_an_identity_and_a_shift_moves_it_by_frames(self):
        ident = self.path("ident.mp4")
        self.assertEqual(apply(self.changing, TEMPORAL, "--param", "offset=0", "-o", ident, tmp=self.dir).returncode, 0)
        self.assertGreater(psnr(self.changing, ident), 30.0, "an identity must not move or recolour the picture")
        shifted = self.path("shifted.mp4")
        proc = apply(self.changing, TEMPORAL, "--param", "offset=2", "-o", shifted, tmp=self.dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertSameFacts(self.changing, shifted)
        # output frame n is input frame n + 2
        self.assertGreater(psnr(shifted, self.changing, 0, 2), 35.0)
        self.assertLess(psnr(shifted, self.changing, 0, 0), 30.0)
        self.assertGreater(psnr(shifted, self.changing, 27, 29), 35.0)
        # and the last two frames have nothing to show: transparent black, which the encode makes black
        self.assertGreater(mean_luma(shifted, 27), 60)
        self.assertLess(mean_luma(shifted, 28), 24)
        self.assertLess(mean_luma(shifted, 29), 24)

    def test_a_shorter_output_cuts_video_and_audio_at_the_same_place(self):
        out = self.path("short.mp4")
        proc = apply(self.changing, TEMPORAL, "--frames", "15", "-o", out, "--json", tmp=self.dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertEqual(data["audio"], {"plan": "shortest", "shortest": True})
        info = tool.probe(out)
        self.assertEqual(tool.display_size(info), (160, 120))
        self.assertAlmostEqual(info["fps"], FPS, delta=0.01)
        self.assertTrue(info["has_audio"])
        src = tool.probe(self.changing)
        self.assertEqual((info["acodec"], info["channels"]), (src["acodec"], src["channels"]))
        self.assertDuration(out, 0.5)
        self.assertLessEqual(info["audio_duration"], 0.5 + 0.1)
        self.assertEqual(info["nb_frames"], 15)
        self.assertClean()
        self.assertNoTempLeft()

    def test_a_longer_output_extends_the_video_and_the_audio_just_ends_early(self):
        out = self.path("long.mp4")
        proc = apply(self.changing, TEMPORAL, "--frames", "45", "-o", out, tmp=self.dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        info = tool.probe(out)
        self.assertEqual((info["nb_frames"], tool.display_size(info)), (45, (160, 120)))
        self.assertAlmostEqual(info["fps"], FPS, delta=0.01)
        self.assertDuration(out, 1.5)
        self.assertTrue(info["has_audio"])
        src = tool.probe(self.changing)
        self.assertEqual((info["acodec"], info["channels"]), (src["acodec"], src["channels"]))
        self.assertLess(info["audio_duration"], 1.2, "the audio was copied, not padded or stretched")
        # past the input there is nothing to show
        self.assertLess(mean_luma(out, 44), 24)
        self.assertGreater(mean_luma(out, 10), 60)
        self.assertNoTempLeft()

    def test_frames_equal_to_the_input_count_is_the_default(self):
        out = self.path("same.mp4")
        proc = apply(self.src, TEMPORAL, "--frames", "30", "-o", out, "--json", "--dry-run", tmp=self.dir)
        self.assertEqual(json.loads(proc.stdout)["audio"], {"plan": None, "shortest": False})
        self.assertEqual(apply(self.src, TEMPORAL, "--frames", "30", "-o", out, tmp=self.dir).returncode, 0)
        self.assertSameFacts(self.src, out)

    def test_a_clip_without_audio_has_no_audio_to_trim(self):
        silent = self.path("silent.mp4")
        make_clip(silent, audio=False)
        out = self.path("silent_out.mp4")
        proc = apply(silent, TEMPORAL, "--frames", "10", "-o", out, "--json", tmp=self.dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["audio"], {"plan": None, "shortest": False})
        self.assertFalse(tool.probe(out)["has_audio"])
        self.assertDuration(out, 10 / FPS)

    # ---- layers ----------------------------------------------------------------

    def test_a_layer_is_decoded_to_a_temp_file_mixed_in_and_removed(self):
        plain, layered = self.path("plain.mp4"), self.path("layered.mp4")
        self.assertEqual(apply(self.changing, TEMPORAL, "-o", plain, tmp=self.dir).returncode, 0)
        proc = apply(self.changing, TEMPORAL, "--layer", f"overlay={self.other}", "-o", layered, "--json", tmp=self.dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertSameFacts(self.changing, layered)
        self.assertNotEqual(sha(plain), sha(layered), "the layer changed the picture")
        # half the layer, which is the negative of testsrc, is mixed into a flat colour
        self.assertLess(psnr(layered, plain, 5, 5), 25.0)
        data = json.loads(proc.stdout)
        self.assertTrue(data["layers"][0]["temp"].endswith("layer_overlay.rgba64le"))
        self.assertFalse(os.path.exists(data["layers"][0]["temp"]), "the temp file is gone")
        self.assertNoTempLeft()

    def test_a_layer_of_another_size_is_scaled_to_the_picture(self):
        small = self.path("small.mp4")
        make_clip(small, size="80x60", audio=False)
        out = self.path("scaled.mp4")
        proc = apply(self.src, TEMPORAL, "--layer", f"overlay={small}", "-o", out, tmp=self.dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertSameFacts(self.src, out)
        self.assertNoTempLeft()

    def test_layer_mistakes_are_refused_before_any_work(self):
        before = snapshot(self.dir)
        cases = [
            ((TEMPORAL, "--layer", f"nope={self.other}"), "unknown layer `nope`"),
            ((TEMPORAL, "--layer", f"overlay={self.other}", "--layer", f"overlay={self.other}"), "more than once"),
            ((TEMPORAL, "--layer", f"overlay={self.path('missing.mp4')}"), "no such file"),
            ((INTEGRATE, "--layer", f"overlay={self.other}"), "has no layers"),
            ((EFFECT, "--layer", f"overlay={self.other}"), "frame effect"),
            ((EFFECT, "--frames", "10"), "frame effect"),
        ]
        for args, word in cases:
            with self.subTest(args=args):
                proc = apply(self.src, *args, "-o", self.path("never.mp4"), tmp=self.dir)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn(word, proc.stderr)
                self.assertEqual(snapshot(self.dir), before)
        # usage errors are argparse's
        self.assertEqual(apply(self.src, TEMPORAL, "--layer", "novalue").returncode, 2)
        self.assertEqual(apply(self.src, TEMPORAL, "--frames", "0").returncode, 2)
        self.assertEqual(apply(self.src, TEMPORAL, "--frames", "many").returncode, 2)

    def test_a_layer_that_is_not_a_video_is_refused(self):
        wav = self.path("tone.wav")
        ff("-f", "lavfi", "-i", "sine=frequency=440:duration=1", wav)
        proc = apply(self.src, TEMPORAL, "--layer", f"overlay={wav}", "-o", self.path("x.mp4"), tmp=self.dir)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("no video stream", proc.stderr)
        self.assertFalse(os.path.exists(self.path("x.mp4")))

    def test_keyframes_and_bad_parameters_are_checked_before_decoding(self):
        out = self.path("keyed.mp4")
        proc = apply(self.src, INTEGRATE, "--param", "gain=0@0;4@10", "-o", out, tmp=self.dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertSameFacts(self.src, out)
        before = snapshot(self.dir)
        for bad, word in (("gain=999", "gain"), ("gain=1@0;2@0", "same time"), ("nope=1", "nope")):
            proc = apply(self.src, INTEGRATE, "--param", bad, "-o", self.path("never.mp4"), tmp=self.dir)
            self.assertNotEqual(proc.returncode, 0, bad)
            self.assertIn(word, proc.stderr)
            self.assertEqual(snapshot(self.dir), before)

    def test_bezier_handles_pass_through_to_vfx(self):
        import shlex
        keys = "gain=0@0o0/100;4@10i-2.5/33.3h"
        dry = apply(self.src, INTEGRATE, "--param", keys, "--dry-run", "--json", tmp=self.dir)
        self.assertEqual(dry.returncode, 0, dry.stderr)
        effect = next(shlex.split(c) for c in json.loads(dry.stdout)["commands"] if "--param" in c)
        self.assertEqual(effect[effect.index("--param") + 1], keys)
        out = self.path("handles.mp4")
        proc = apply(self.src, INTEGRATE, "--param", keys, "-o", out, tmp=self.dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertSameFacts(self.src, out)
        before = snapshot(self.dir)
        for bad, word in (("gain=0@0i0/0;4@10", "influence 0 is outside 0.1..=100"),
                          ("gain=0@0;4@10o1/50h", "a held key has no outgoing handle")):
            proc = apply(self.src, INTEGRATE, "--param", bad, "-o", self.path("never.mp4"), tmp=self.dir)
            self.assertNotEqual(proc.returncode, 0, bad)
            self.assertIn(word, proc.stderr)
            self.assertEqual(snapshot(self.dir), before)

    # ---- failure -----------------------------------------------------------------

    def run_main_with_a_failing_effect(self, out, *flags):
        """the real main(), temp files in self.dir, with the effect stage swapped for one that dies early."""
        dies = [sys.executable, "-c", "import sys; sys.stdin.buffer.read(100); sys.exit(3)"]
        real = tool.build_commands

        def swapped(*a, **k):
            decode, _, encode = real(*a, **k)
            return decode, dies, encode

        argv = ["apply.py", self.src, TEMPORAL, "-o", out, *flags]
        stderr = io.StringIO()
        with mock.patch.object(sys, "argv", argv), mock.patch.object(tool, "build_commands", swapped), \
                mock.patch.object(tempfile, "tempdir", self.dir), \
                contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                tool.main()
        return caught.exception.code, stderr.getvalue()

    def test_a_pipeline_that_fails_midway_removes_the_layer_files_and_the_scratch(self):
        out = self.path("out.mp4")
        with open(out, "wb") as f:
            f.write(b"keep me")
        code, err = self.run_main_with_a_failing_effect(out, "--force", "--layer", f"overlay={self.other}")
        self.assertNotEqual(code, 0)
        self.assertIn("vfx exited with code 3", err)
        self.assertEqual(slurp(out), b"keep me")
        self.assertClean()
        self.assertNoTempLeft()

    def test_a_layer_that_cannot_be_decoded_fails_and_cleans_up(self):
        broken = [sys.executable, "-c", "import sys; sys.stderr.write('no picture in here'); sys.exit(5)"]
        out = self.path("out.mp4")
        argv = ["apply.py", self.src, TEMPORAL, "-o", out, "--layer", f"overlay={self.other}"]
        stderr = io.StringIO()
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(tool, "build_layer_commands", lambda *a, **k: [broken]), \
                mock.patch.object(tempfile, "tempdir", self.dir), \
                contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                tool.main()
        self.assertNotEqual(caught.exception.code, 0)
        self.assertIn("layer overlay: decoding", stderr.getvalue())
        self.assertIn("no picture in here", stderr.getvalue())
        self.assertFalse(os.path.exists(out))
        self.assertNoTempLeft()
        self.assertClean()


class PlanTests(unittest.TestCase):
    """the decisions about frames and audio, on facts alone."""

    def test_the_audio_follows_the_length_of_the_video(self):
        rate = Fraction(30)
        info = facts(audio_duration=1.0)
        self.assertEqual(tool.audio_plan(info, 15, rate), "shortest")
        self.assertEqual(tool.audio_plan(info, 45, rate), "ends-early")
        self.assertEqual(tool.audio_plan(info, 30, rate), "same")
        self.assertEqual(tool.audio_plan(info, 31, rate), "same", "within a frame")
        self.assertEqual(tool.audio_plan(info, 28, rate), "shortest")
        self.assertIsNone(tool.audio_plan(facts(has_audio=False, acodec=None, channels=None), 15, rate))
        # without a stream duration the container's stands in
        self.assertEqual(tool.audio_plan(facts(audio_duration=None, duration=2.0), 30, rate), "shortest")

    def test_the_frame_count_is_known_only_for_a_constant_rate_that_says_so(self):
        self.assertEqual(tool.exact_frames(facts(nb_frames=30, vfr=False)), 30)
        self.assertIsNone(tool.exact_frames(facts(nb_frames=None, vfr=False)))
        self.assertIsNone(tool.exact_frames(facts(nb_frames=30, vfr=True)))

    def test_ffprobe_counts_and_seconds_are_read_defensively(self):
        self.assertEqual((tool._count("30"), tool._count("N/A"), tool._count(None), tool._count("0")), (30, None, None, None))
        self.assertEqual((tool._seconds("1.5"), tool._seconds("N/A"), tool._seconds(None), tool._seconds("-1")), (1.5, None, None, None))

    def test_layer_and_frames_arguments(self):
        self.assertEqual(tool.layer_arg("overlay=C:/clips/o.mp4"), ("overlay", "C:/clips/o.mp4"))
        self.assertEqual(tool.layer_arg("a=b=c"), ("a", "b=c"))
        for bad in ("overlay", "=x.mp4", "overlay="):
            with self.assertRaises(argparse.ArgumentTypeError):
                tool.layer_arg(bad)
        self.assertEqual(tool.frames_arg("12"), 12)
        for bad in ("0", "-3", "x", ""):
            with self.assertRaises(argparse.ArgumentTypeError):
                tool.frames_arg(bad)

    def test_a_plan_names_its_temp_files_without_creating_any(self):
        args = argparse.Namespace(depth=8, frames=15, layer=[])
        info = facts(nb_frames=30, vfr=False, audio_duration=1.0)
        einfo = {"kind": "temporal", "layers": []}
        layer = {"id": "overlay", "video": "o.mp4", "info": facts()}
        scratch = os.path.join(tempfile.gettempdir(), "apply_never_created")
        plan = tool.make_plan(args, info, einfo, [layer], scratch)
        self.assertEqual(plan["layers"][0]["temp"], os.path.join(scratch, "layer_overlay.rgba"))
        self.assertEqual([t["path"] for t in plan["temp_files"]], [plan["layers"][0]["temp"], scratch])
        self.assertEqual((plan["frames"], plan["audio"], plan["shortest"]), (15, "shortest", True))
        self.assertFalse(os.path.exists(scratch))
        frame = tool.make_plan(args, info, {"kind": "frame", "layers": []}, [], scratch)
        self.assertEqual((frame["kind"], frame["layers"], frame["frames"], frame["temp_files"]), ("frame", [], None, []))

    def test_the_stage_commands_carry_the_temporal_options(self):
        args = argparse.Namespace(input="in.mp4", effect="fx", depth=16, param=["a=1@0;2@5"], frames=20, layer=[("overlay", "o.mp4")])
        untagged = dict(video_index=0, vfr=False, color_space=None, color_primaries=None, color_transfer=None)
        info = facts(nb_frames=30, audio_duration=1.0, **untagged)
        layer = {"id": "overlay", "video": "o.mp4", "info": facts(**untagged)}
        plan = tool.make_plan(args, info, {"kind": "temporal", "layers": []}, [layer], "/tmp/s")
        decode, effect, encode = tool.build_commands(args, info, "vfx", "out.mp4", plan)
        self.assertEqual(effect[:3], ["vfx", "apply", "fx"])
        self.assertEqual(effect[effect.index("--fps") + 1], "30/1")
        self.assertEqual(effect[effect.index("--frames") + 1], "20")
        self.assertEqual(effect[effect.index("--tmp-dir") + 1], "/tmp/s")
        self.assertEqual(effect[effect.index("--layer") + 1], "overlay=" + os.path.join("/tmp/s", "layer_overlay.rgba64le"))
        self.assertEqual(effect[effect.index("--param") + 1], "a=1@0;2@5")
        self.assertIn("-shortest", encode)
        self.assertLess(encode.index("-shortest"), encode.index("-movflags"))
        # a frame effect gets none of it
        plain = tool.build_commands(args, info, "vfx", "out.mp4")
        for word in ("--frames", "--tmp-dir", "--layer"):
            self.assertNotIn(word, plain[1])
        self.assertNotIn("-shortest", plain[2])
        layer_cmd = tool.build_layer_commands(args, info, plan)[0]
        self.assertEqual(layer_cmd[layer_cmd.index("-i") + 1], "o.mp4")
        self.assertEqual(layer_cmd[-1], os.path.join("/tmp/s", "layer_overlay.rgba64le"))
        self.assertIn("rgba64le", layer_cmd)
        # same size and rate: no fps or scale to a new size
        vf = layer_cmd[layer_cmd.index("-vf") + 1]
        self.assertTrue(vf.startswith("scale=in_color_matrix=") and "fps=" not in vf, vf)


class NoteTests(Workspace):
    """the notes vfx prints for the open parity gaps a render used, relayed by apply.py. the hidden
    effect selftest-gap copies its input and records the test gap T1 while its checkbox gap is on."""

    GAP = "selftest-gap"
    NOTE = ("vfx: selftest-gap: T1 notes plumbing test: a stated rule that no Twixtor reference render "
            "has graded yet (README: limits)")

    def setUp(self):
        super().setUp()
        if self.GAP not in {e["id"] for e in listed("--all")}:
            self.skipTest("this vfx has no selftest-gap effect (build it from the current workspace)")

    def notes_on(self, stderr):
        return [line for line in stderr.splitlines() if line.startswith("[note]")]

    def test_vfx_notes_are_relayed(self):
        out = self.path("gap_on.mp4")
        proc = apply(self.src, self.GAP, "--param", "gap=1", "--depth", "8", "-o", out, tmp=self.dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.notes_on(proc.stderr), ["[note] " + self.NOTE])
        self.assertNotIn("[note]", proc.stdout)
        self.assertTrue(proc.stdout.rstrip().splitlines()[-1].startswith(os.path.abspath(out)), proc.stdout)
        self.assertSameFacts(self.src, out)
        # --json: the same line under notes, and every other key as a run without notes has it
        noted = json.loads(apply(self.src, self.GAP, "--param", "gap=1", "--depth", "8", "--json",
                                 "-o", self.path("a.mp4"), tmp=self.dir).stdout)
        plain = apply(self.src, self.GAP, "--param", "gap=0", "--depth", "8", "--json", "-o", self.path("b.mp4"),
                      tmp=self.dir)
        self.assertEqual(plain.returncode, 0, plain.stderr)
        self.assertEqual(self.notes_on(plain.stderr), [])
        plain = json.loads(plain.stdout)
        self.assertNotIn("notes", plain)
        self.assertEqual(noted["notes"], [self.NOTE])
        self.assertEqual(list(noted), [k for k in plain if k != "seconds"] + ["notes", "seconds"])
        # a dry run renders nothing, so it has no notes and no key
        dry = apply(self.src, self.GAP, "--param", "gap=1", "--dry-run", "--json", tmp=self.dir)
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertNotIn("notes", json.loads(dry.stdout))
        self.assertEqual(self.notes_on(dry.stderr), [])
        # the frames/2 report of a run that went through carries them too
        framed = apply(self.src, self.GAP, "--param", "gap=1", "--depth", "8", "--contract", "frames/2", "--json",
                       "-o", self.path("c.mp4"), tmp=self.dir)
        self.assertEqual(framed.returncode, 0, framed.stderr)
        report = json.loads(framed.stdout)
        self.assertTrue(report["ok"])
        self.assertEqual(report["notes"], [self.NOTE])
        self.assertEqual(self.notes_on(framed.stderr), ["[note] " + self.NOTE])

    def test_only_note_lines_of_the_vfx_log_are_relayed(self):
        log = self.path("vfx.log")
        with open(log, "w", encoding="utf-8") as f:
            f.write("something else\n"
                    "vfx: note: retime-fgbg: G1 matte shrink tail: a stated rule that no Twixtor reference render "
                    "has graded yet (README: limits)\n"
                    "  vfx: note: indented, so not one\n"
                    "vfx: note: retime-fgbg: G6 solver, hole fill and blur wrap variants: x\r\n"
                    "vfx: error: not a note\n")
        self.assertEqual(tool.vfx_notes(log), [
            "vfx: retime-fgbg: G1 matte shrink tail: a stated rule that no Twixtor reference render has graded yet "
            "(README: limits)",
            "vfx: retime-fgbg: G6 solver, hole fill and blur wrap variants: x"])
        self.assertEqual(tool.vfx_notes(self.path("missing.log")), [])
        result = {"outputs": []}
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            tool.relay_notes(result, {"vfx": self.path("missing.log")})
            tool.relay_notes(result, None)
        self.assertEqual((result, stderr.getvalue()), ({"outputs": []}, ""))
        with contextlib.redirect_stderr(stderr):
            tool.relay_notes(result, {"vfx": log, "decode": log})
        self.assertEqual(len(result["notes"]), 2)
        self.assertEqual(stderr.getvalue().splitlines(), ["[note] " + n for n in result["notes"]])

    def test_a_run_that_fails_relays_no_notes(self):
        """a note before the failure stays in the log tail of the error and is not relayed on its own."""
        dies = [sys.executable, "-c", "import sys; sys.stdin.buffer.read(100); "
                "sys.stderr.write('vfx: note: selftest-gap: T1 notes plumbing test: x\\n'); sys.exit(3)"]
        real = tool.build_commands

        def swapped(*a, **k):
            decode, _, encode = real(*a, **k)
            return decode, dies, encode

        argv = ["apply.py", self.src, self.GAP, "-o", self.path("never.mp4"), "--param", "gap=1"]
        stderr = io.StringIO()
        with mock.patch.object(sys, "argv", argv), mock.patch.object(tool, "build_commands", swapped), \
                mock.patch.object(tempfile, "tempdir", self.dir), \
                contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                tool.main()
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("vfx exited with code 3", stderr.getvalue())
        self.assertIn("vfx: note: selftest-gap: T1", stderr.getvalue())
        self.assertEqual(self.notes_on(stderr.getvalue()), [])
        self.assertFalse(os.path.exists(self.path("never.mp4")))


class VerifyTests(unittest.TestCase):
    """the output check, on facts alone."""

    def test_a_matching_output_has_no_problems(self):
        self.assertEqual(tool.verify(facts(), facts(duration=1.03), Fraction(30)), [])

    def test_each_kind_of_mismatch_is_named(self):
        rate = Fraction(30)
        cases = [
            (facts(duration=1.2), "duration"),
            (facts(width=162), "size"),
            (facts(fps=24.0), "frame rate"),
            (facts(has_audio=False, acodec=None, channels=None), "audio stream was lost"),
            (facts(acodec="mp3"), "audio is mp3"),
            (facts(duration=None), "no readable duration"),
            (facts(has_video=False), "no readable video"),
        ]
        for out, word in cases:
            with self.subTest(word=word):
                problems = tool.verify(facts(), out, rate)
                self.assertTrue(any(word in p for p in problems), problems)
        self.assertTrue(any("no readable" in p for p in tool.verify(facts(), None, rate)))

    def test_audio_that_appears_from_nowhere_is_a_problem(self):
        silent = facts(has_audio=False, acodec=None, channels=None)
        self.assertEqual(tool.verify(silent, silent, Fraction(30)), [])
        self.assertTrue(tool.verify(silent, facts(), Fraction(30)))

    def test_a_rotated_source_is_compared_by_its_displayed_size(self):
        self.assertEqual(tool.verify(facts(rotation=90), facts(width=120, height=160), Fraction(30)), [])
        self.assertTrue(tool.verify(facts(rotation=90), facts(), Fraction(30)))

    def test_an_explicit_frame_count_sets_the_expected_duration_and_frame_count(self):
        rate = Fraction(30)
        src = facts()
        cut = facts(duration=0.5, nb_frames=15)
        self.assertEqual(tool.verify(src, cut, rate, frames=15, source_frames=30), [])
        # a trimmed audio copy may run a packet past the video
        self.assertEqual(tool.verify(src, facts(duration=0.58, nb_frames=15), rate, frames=15, source_frames=30), [])
        too_long = tool.verify(src, facts(duration=1.0, nb_frames=15), rate, frames=15, source_frames=30)
        self.assertTrue(any("duration" in p for p in too_long), too_long)
        wrong_count = tool.verify(src, facts(duration=0.5, nb_frames=14), rate, frames=15, source_frames=30)
        self.assertTrue(any("14 frames, expected 15" in p for p in wrong_count), wrong_count)
        longer = facts(duration=1.5, nb_frames=45)
        self.assertEqual(tool.verify(src, longer, rate, frames=45, source_frames=30), [])
        self.assertTrue(tool.verify(src, longer, rate, frames=30, source_frames=30),
                        "the input's own count keeps the input's duration")
        # a missing frame count in the output cannot be checked, and is not an error
        self.assertEqual(tool.verify(src, facts(duration=0.5), rate, frames=15, source_frames=30), [])
        # without frames the old rule stands
        self.assertEqual(tool.verify(src, facts(duration=1.03), rate), [])
        # the audio checks still apply
        silent = facts(duration=0.5, nb_frames=15, has_audio=False, acodec=None, channels=None)
        self.assertTrue(any("audio stream" in p for p in tool.verify(src, silent, rate, frames=15, source_frames=30)))

    def test_the_duration_window_is_one_frame(self):
        self.assertEqual(tool.verify(facts(), facts(duration=1.0 + 1 / 30), Fraction(30)), [])
        self.assertTrue(tool.verify(facts(), facts(duration=1.0 + 2 / 30), Fraction(30)))


class ColorTests(unittest.TestCase):
    def test_tagged_sources_keep_their_tags_and_untagged_ones_follow_size(self):
        tagged = facts(color_space="bt709", color_primaries="bt709", color_transfer="bt709")
        self.assertEqual(tool.color_settings(tagged), ("bt709", "bt709", "bt709", "bt709"))
        pal = facts(color_space="bt470bg", color_primaries=None, color_transfer=None)
        self.assertEqual(tool.color_settings(pal), ("bt601", "bt470bg", None, None))
        untagged = dict(color_space=None, color_primaries=None, color_transfer=None)
        self.assertEqual(tool.color_settings(facts(width=1280, height=720, **untagged))[0], "bt709")
        self.assertEqual(tool.color_settings(facts(width=1920, height=1080, **untagged))[1:], ("bt709",) * 3)
        self.assertEqual(tool.color_settings(facts(**untagged))[1:], ("smpte170m",) * 3)

    def test_display_size_swaps_on_quarter_turns_only(self):
        self.assertEqual(tool.display_size(facts(rotation=90)), (120, 160))
        self.assertEqual(tool.display_size(facts(rotation=270)), (120, 160))
        self.assertEqual(tool.display_size(facts(rotation=180)), (160, 120))
        self.assertEqual(tool.display_size(facts()), (160, 120))


def stage_line(*argv):
    """one --pre-stage or --post-stage command line, quoted the way apply.py splits it."""
    if os.name == "nt":
        return " ".join('"%s"' % a for a in map(str, argv))
    return " ".join(shlex.quote(str(a)) for a in argv)


def no_pid(text):
    """the partial output path carries the process id; everything else of a plan is the same run to run."""
    return re.sub(r"\.partial-[0-9]+\.", ".partial-PID.", text)


def stream_hashes(path):
    """the sha256 of every packet of every stream of a file, as ffmpeg's streamhash muxer prints it."""
    cmd = [tool.ffmpeg_path(), "-v", "error", "-i", path, "-map", "0", "-c", "copy", "-f", "streamhash",
           "-hash", "sha256", "-"]
    return subprocess.run(cmd, shell=False, check=True, capture_output=True, text=True).stdout


class UnbuiltFlagTests(Workspace):
    """--size, --fps and --audio: their grammar, their help and their full-only names, and --audio copy
    being the run of before. The behaviour of --size and --fps: test_apply_geometry.py; of the
    audio modes: test_audio_map.py and AudioRunTests."""

    def setUp(self):
        super().setUp()
        if not TEMPORAL:
            self.skipTest("this vfx has no temporal self-test effects (build it from the current workspace)")

    def test_audio_copy_is_the_run_of_before(self):
        for effect, flags in ((EFFECT, []), (TEMPORAL, ["--frames", "15"]), (TEMPORAL, ["--frames", "45"]),
                              (TEMPORAL, ["--contract", "frames/2"])):
            with self.subTest(effect=effect, flags=flags):
                plain = apply(self.src, effect, *flags, "--dry-run", "--json", tmp=self.dir)
                copy = apply(self.src, effect, *flags, "--audio", "copy", "--dry-run", "--json", tmp=self.dir)
                self.assertEqual(plain.returncode, 0, plain.stderr)
                self.assertEqual((copy.returncode, no_pid(copy.stdout), copy.stderr),
                                 (0, no_pid(plain.stdout), plain.stderr))
                text = apply(self.src, effect, *flags, "--audio", "copy", "--dry-run", tmp=self.dir)
                self.assertEqual(no_pid(text.stdout), no_pid(apply(self.src, effect, *flags, "--dry-run", tmp=self.dir).stdout))
        # a real run writes the same packets, the copied audio included
        outs = []
        for extra in ((), ("--audio", "copy")):
            out = self.path(f"copy{len(outs)}.mkv")
            proc = apply(self.src, TEMPORAL, "--frames", "20", "--lossless", "--depth", "8", *extra, "-o", out,
                         tmp=self.dir)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            outs.append(stream_hashes(out))
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[0].count(",a,"), 1, outs[0])
        self.assertClean()

    def test_the_flags_match_only_in_full_and_check_their_values(self):
        missing = self.path("missing.mp4")

        def parse(*flags):
            proc = apply(missing, TEMPORAL, *flags, tmp=self.dir)
            return proc.returncode, (proc.stderr.strip().splitlines() or [""])[-1]

        for flags in (["--s", "64x48"], ["--si", "64x48"], ["--siz", "64x48"], ["--fp", "24"], ["--a", "copy"],
                      ["--au", "copy"], ["--audi", "drop"]):
            self.assertEqual(parse(*flags), (2, "apply.py: error: unrecognized arguments: " + " ".join(flags)))
        self.assertEqual(parse("--f", "24"),
                         (2, "apply.py: error: ambiguous option: --f could match --frames, --force, --ffmpeg-path, "
                             "--ffprobe-path"))
        for flags, words in ((["--size", "0x48"], "argument --size: expected WxH"),
                             (["--size", "64X48"], "argument --size: expected WxH"),
                             (["--fps", "0"], "argument --fps: expected a frame rate above 0"),
                             (["--fps", "29.9700001"], "argument --fps: expected a frame rate above 0"),
                             (["--audio", "loud"], "argument --audio: invalid choice: 'loud'"),
                             (["--size"], "argument --size: expected one argument")):
            code, last = parse(*flags)
            self.assertEqual(code, 2, last)
            self.assertIn(words, last)
        # a value they take passes the parser and meets the input checks first
        for flags in (["--size", "64x48"], ["--fps", "30000/1001"], ["--audio", "drop"], ["--audio", "copy"],
                      ["--size=1x1"], ["--fps=24"], ["--audio=retime"]):
            self.assertEqual(parse(*flags), (1, f"[error] no such file: {missing}"), flags)
        self.assertLessEqual({"--size", "--fps", "--audio"}, tool.ArgParser.FULL_ONLY)
        helped = apply("--help", tmp=self.dir).stdout
        for flag in ("--size WxH", "--fps R", "--audio {copy,retime,drop}"):
            self.assertIn(flag, helped)

    def test_the_rate_grammar_is_the_one_of_vfx(self):
        good = {"24": Fraction(24), "29.97": Fraction(2997, 100), "30000/1001": Fraction(30000, 1001),
                "60000/2002": Fraction(30000, 1001), " 25 ": Fraction(25), "24 / 1": Fraction(24), "+24/1": Fraction(24),
                "0.000001": Fraction(1, 1000000), "29.970000": Fraction(2997, 100), "00024": Fraction(24),
                "24\u3000": Fraction(24), "4294967295/4294967294": Fraction(4294967295, 4294967294)}
        bad = ["0", "0/1", "1/0", "-24", "nan", "inf", "x", "", "24.", ".5", "29.9700000", "1.2.3", "+24", "1_000",
               "\u0663", "\x1c24", "4294967296", "1/4294967296", "24/1/1", "1e3", "0x18", "24 1"]
        for text, rate in good.items():
            self.assertEqual(tool.fps_arg(text), rate, repr(text))
            self.assertIsInstance(tool.fps_arg(text), Fraction)
        for text in bad:
            with self.assertRaises(argparse.ArgumentTypeError, msg=repr(text)):
                tool.fps_arg(text)
        # vfx itself takes exactly these as an exact rate for a temporal effect
        for text in list(good) + bad:
            with self.subTest(text=text):
                check = subprocess.run([VFX, "apply", TEMPORAL, "--width", "2", "--height", "2", "--depth", "8",
                                        "--fps", text, "--check"], shell=False, capture_output=True, text=True,
                                       encoding="utf-8", errors="replace")
                self.assertEqual(check.returncode == 0, text in good, check.stderr)

    def test_the_size_grammar_is_w_x_h_of_one_or_more(self):
        for text, size in (("1280x720", (1280, 720)), ("1x1", (1, 1)), (" 64x48 ", (64, 48)), ("0064x048", (64, 48))):
            self.assertEqual(tool.size_arg(text), size)
        for text in ("0x48", "64x0", "64X48", "64", "x48", "64x", "-1x5", "1.5x2", "64x48x2", "\u0661x1", "", "64 x 48",
                     "64*48", "+64x48"):
            with self.assertRaises(argparse.ArgumentTypeError, msg=repr(text)):
                tool.size_arg(text)


class SeamTests(unittest.TestCase):
    """the seams the audio, size and rate, and HDR work fills, on facts alone: what each gives in this
    build, and the order the encode, the summary and the report keep around them."""

    def args(self, **over):
        base = dict(input="in.mp4", effect="fx", depth=16, param=["a=1"], frames=None, layer=[])
        base.update(over)
        return argparse.Namespace(**base)

    def info(self, **over):
        base = dict(nb_frames=30, vfr=False, video_index=0, color_space=None, color_primaries=None,
                    color_transfer=None, audio_duration=1.0)
        base.update(over)
        return facts(**base)

    def refused(self, call, *args):
        """the [error] line and the exit code of a call that refuses, as frames/1 text prints them."""
        err = io.StringIO()
        with mock.patch.object(tool, "JSON_OUTPUT", False), mock.patch.object(tool, "CONTRACT", "frames/1"), \
                contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as caught:
            call(*args)
        return caught.exception.code, err.getvalue()

    def test_the_work_geometry_is_the_input_geometry(self):
        geo = tool.work_geometry(self.args(), self.info())
        self.assertEqual(geo, {"width": 160, "height": 120, "rate": Fraction(30), "frames": 30, "source": None,
                               "auto_params": [], "rounding": None, "clip_rate": Fraction(30), "out_frames": 30})
        self.assertEqual(list(geo), ["width", "height", "rate", "frames", "source", "auto_params", "rounding",
                                     "clip_rate", "out_frames"])
        self.assertEqual(tool.work_geometry(self.args(), self.info(rotation=90))["width"], 120, "the displayed size")
        self.assertIsNone(tool.work_geometry(self.args(), self.info(vfr=True))["frames"])
        listed_ids = {"kind": "temporal", "layers": [], "params": ["in_fps_is_out_fps", "input_frame_rate"]}
        self.assertEqual(tool.work_geometry(self.args(), self.info(), listed_ids), geo)
        # a caller from before the flags, without size, fps or audio, gets the same
        self.assertEqual(tool.work_geometry(argparse.Namespace(), self.info()), geo)
        self.assertEqual(tool.work_geometry(self.args(size=None, fps=None, audio="copy"), self.info()), geo)

    def test_the_audio_modes_and_a_size_and_rate_are_all_taken(self):
        # no size, rate or audio mode is refused any more (test_apply_geometry.py, test_audio_map.py)
        for mode in ("copy", "retime", "drop"):
            self.assertEqual(tool.audio_mode(self.args(audio=mode)), mode)
        self.assertEqual([tool.audio_mode(self.args(audio="copy")), tool.audio_mode(self.args())], ["copy", "copy"])
        self.assertEqual(tool.AUDIO_MODES, ("copy", "retime", "drop"))

    def test_the_decode_and_the_layers_are_read_with_their_own_matrix_at_the_work_geometry(self):
        info = self.info()
        geo = tool.work_geometry(self.args(), info)
        self.assertEqual(tool.decode_filters(self.args(), info, geo), ["scale=in_color_matrix=smpte170m"])
        vfr = self.info(vfr=True, rate=Fraction(30000, 1001))
        self.assertEqual(tool.decode_filters(self.args(), vfr, tool.work_geometry(self.args(), vfr)),
                         ["fps=30000/1001", "scale=in_color_matrix=smpte170m"])
        hd = self.info(width=1280, height=720)
        self.assertEqual(tool.decode_filters(self.args(), hd, tool.work_geometry(self.args(), hd)),
                         ["scale=in_color_matrix=bt709"])
        # a work size other than the input's stretches the decode to it
        self.assertEqual(tool.decode_filters(self.args(), info, dict(geo, width=320, height=180)),
                         ["scale=320:180:in_color_matrix=smpte170m"])
        layer = self.info()
        self.assertEqual(tool.layer_filters(self.args(), layer, geo), ["scale=in_color_matrix=smpte170m"])
        self.assertEqual(tool.layer_filters(self.args(), self.info(rate=Fraction(24)), geo),
                         ["fps=30/1", "scale=in_color_matrix=smpte170m"])
        self.assertEqual(tool.layer_filters(self.args(), self.info(width=80, height=60, color_space="bt709"), geo),
                         ["scale=160:120:in_color_matrix=bt709"])
        self.assertEqual(tool.layer_filters(self.args(), layer, geo, (80, 48)), ["scale=80:48:in_color_matrix=smpte170m"],
                         "the size vfx reads after a declared pre-stage")
        self.assertEqual(tool.layer_filters(self.args(), layer, dict(geo, rate=Fraction(60))),
                         ["fps=60/1", "scale=in_color_matrix=smpte170m"], "the work rate")
        plan = tool.make_plan(self.args(), info, {"kind": "temporal", "layers": []},
                              [{"id": "overlay", "video": "o.mp4", "info": self.info(rate=Fraction(24))}], "/tmp/s")
        cmd = tool.build_layer_commands(self.args(), info, plan, None, dict(geo, width=64, height=48))[0]
        self.assertEqual(cmd[cmd.index("-vf") + 1], "fps=30/1,scale=64:48:in_color_matrix=smpte170m")

    def test_the_added_params_follow_the_users_in_the_preflight_and_the_run(self):
        args, info = self.args(param=["a=1", "b=2@0;3@5"]), self.info()
        plain = tool.work_geometry(args, info)
        geo = dict(plain, auto_params=["in_fps_is_out_fps=0", "input_frame_rate=24"])
        self.assertEqual(tool.effect_params(args), ["a=1", "b=2@0;3@5"])
        self.assertEqual(tool.effect_params(args, plain), ["a=1", "b=2@0;3@5"])
        want = ["a=1", "b=2@0;3@5", "in_fps_is_out_fps=0", "input_frame_rate=24"]
        self.assertEqual(tool.effect_params(args, geo), want)
        self.assertEqual(args.param, ["a=1", "b=2@0;3@5"], "the user's list is left as it was")
        effect = tool.build_commands(args, info, "vfx", "out.mp4", None, None, geo)[1]
        self.assertEqual([effect[i + 1] for i, word in enumerate(effect) if word == "--param"], want)
        seen = []
        with mock.patch.object(tool, "run", lambda cmd, **k: seen.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", "")):
            tool.preflight(args, info, "vfx", {"kind": "frame", "layers": []}, dict(geo, width=64, height=48))
        cmd = seen[0]
        self.assertEqual([cmd[i + 1] for i, word in enumerate(cmd) if word == "--param"], want)
        self.assertEqual(cmd[:12], ["vfx", "apply", "fx", "--width", "64", "--height", "48", "--depth", "16", "--fps",
                                    "30/1", "--check"])

    def test_the_encode_keeps_its_order_around_the_audio_and_video_seams(self):
        args, info = self.args(frames=15), self.info()
        plan = tool.make_plan(args, info, {"kind": "temporal", "layers": []}, [], "/tmp/s")
        real = tool.video_encode_args
        calls = []

        def video(*a, **k):
            calls.append(inspect.signature(real).bind(*a, **k).arguments)
            return ["VIDEO", "ARGS"]

        parts = (["-i", "A.f32"], ["-map", "1:a:0"], ["-c:a", "aac"])
        with mock.patch.object(tool, "video_encode_args", video), mock.patch.object(tool, "audio_parts", lambda *a: parts):
            encode = tool.build_commands(args, info, "vfx", "out.mp4", plan, ((160, 120), (80, 48)))[2]
        head = [tool.ffmpeg_path(), "-v", "error", "-nostdin", "-n", "-f", "rawvideo", "-pix_fmt", "rgba64le",
                "-video_size", "80x48", "-framerate", "30/1", "-i", "-"]
        self.assertEqual(encode, head + ["-i", "A.f32", "-map", "0:v:0", "-map", "1:a:0", "VIDEO", "ARGS", "-c:a", "aac",
                                         "-shortest", "-movflags", "+faststart", tool.partial_path("out.mp4")])
        call, = calls
        self.assertEqual((call["out_size"], call["destination"], call["geo"]["width"]), ((80, 48), "out.mp4", 160))
        # the parts of this build
        self.assertEqual(tool.audio_parts(args, info, plan), (["-i", "in.mp4"], ["-map", "1:a?"], ["-c:a", "copy"]))
        self.assertEqual(tool.audio_parts(args, info, None), (["-i", "in.mp4"], ["-map", "1:a?"], ["-c:a", "copy"]))
        self.assertEqual(tool.audio_commands(args, info, plan), [])
        geo = tool.work_geometry(args, info)
        self.assertEqual(real(args, info, geo, (160, 120)),
                         ["-vf", "scale=out_color_matrix=smpte170m:out_range=tv,setparams=colorspace=smpte170m:"
                                 "color_primaries=smpte170m:color_trc=smpte170m:range=tv",
                          "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"])
        self.assertEqual(real(self.args(lossless=True, depth=8), info, geo, (101, 61), "out.mkv"),
                         ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgra"])
        self.assertEqual(real(self.args(lossless=True), info, geo, (160, 120)),
                         ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "rgba64le"])

    def test_the_summary_and_the_report_keep_their_order_around_the_report_seams(self):
        args, info = self.args(frames=15, dry_run=True), self.info()
        layer = {"id": "overlay", "video": "o.mp4", "info": self.info()}
        plan = tool.make_plan(args, info, {"kind": "temporal", "layers": []}, [layer], "/tmp/s")
        seen = {}

        def hook(name, phrases, keys):
            real = getattr(tool, name)

            def report(*a, **k):
                seen[name] = inspect.signature(real).bind(*a, **k).arguments
                return phrases, keys
            return report

        with mock.patch.object(tool, "report_geometry", hook("report_geometry", ["GEO"], {"source": {"size": "80x60"}})), \
                mock.patch.object(tool, "report_encode", hook("report_encode", ["ENC"], {"encode": {"pix_fmt": "x"}})), \
                mock.patch.object(tool, "report_audio", hook("report_audio", ["AUD"], {"audio": {"mode": "m"}, "z": 1})):
            result = tool.make_result(args, info, {"kind": "temporal"}, "vfx", "out.mp4", plan, [], (), None, (81, 48))
        self.assertEqual(result["summary"], "fx on in.mp4: 160x120, 30.000 fps, 16 bpc (rgba64le), temporal, 15 frames, "
                                            "layers overlay, GEO, ENC, AUD, decode | vfx | encode -> out.mp4")
        keys = list(result)
        self.assertEqual(keys[keys.index("pix_fmt"):keys.index("vfx") + 1],
                         ["pix_fmt", "size", "fps", "fps_exact", "frames", "input_frames", "source", "layers", "encode",
                          "audio", "z", "temp_files", "vfx"])
        self.assertEqual((result["audio"], seen["report_encode"]["out_size"]), ({"mode": "m"}, (81, 48)))
        self.assertIs(seen["report_audio"]["plan"], plan)
        # the reports of this build: nothing about the geometry and the encode, the copy's decision about the audio
        geo = tool.work_geometry(args, info)
        self.assertEqual(tool.report_geometry(args, info, geo), ([], {}))
        self.assertEqual(tool.report_encode(args, info, geo, (160, 120)), ([], {}))
        self.assertEqual(tool.report_audio(args, info, plan),
                         (["audio cut at the end of the video (-shortest)"], {"audio": {"plan": "shortest", "shortest": True}}))
        longer = tool.make_plan(self.args(frames=45), info, {"kind": "temporal", "layers": []}, [], "/tmp/s")
        self.assertEqual(tool.report_audio(args, info, longer)[0], ["the audio ends before the video does"])
        # a frame plan never names the audio in the summary, even when a declared stage changed its length
        frame = tool.plan_audio(args, info, tool.make_plan(args, info, {"kind": "frame", "layers": []}, [], "/tmp/s"), 15)
        self.assertEqual(tool.report_audio(args, info, frame), ([], {"audio": {"plan": "shortest", "shortest": True}}))
        plain = tool.make_result(args, info, {"kind": "temporal"}, "vfx", "out.mp4", plan, [], ())
        self.assertEqual(plain["summary"], "fx on in.mp4: 160x120, 30.000 fps, 16 bpc (rgba64le), temporal, 15 frames, "
                                           "layers overlay, audio cut at the end of the video (-shortest), "
                                           "decode | vfx | encode -> out.mp4")
        self.assertNotIn("source", plain)

    def test_the_audio_decision_waits_for_a_count_that_differs_and_uses_the_work_rate(self):
        args, info = self.args(), self.info()

        def fresh():
            return {"input_frames": 30, "audio": None, "shortest": False}

        self.assertEqual(tool.plan_audio(args, info, fresh(), None), fresh())
        self.assertEqual(tool.plan_audio(args, info, fresh(), 30), fresh())
        self.assertEqual(tool.plan_audio(args, info, fresh(), 15), dict(fresh(), audio="shortest", shortest=True))
        self.assertEqual(tool.plan_audio(args, info, fresh(), 45)["audio"], "ends-early")
        # 45 frames at a work rate of 60 last 0.75 s, shorter than the 1 s of audio
        geo = dict(tool.work_geometry(args, info), rate=Fraction(60))
        self.assertEqual(tool.plan_audio(args, info, fresh(), 45, geo)["audio"], "shortest")
        # a decision made stands when a later count is the input's own or unknown
        decided = tool.plan_audio(args, info, fresh(), 15)
        for later in (30, None):
            self.assertEqual(tool.plan_audio(args, info, dict(decided), later)["audio"], "shortest")
        silent = tool.plan_audio(args, self.info(has_audio=False, acodec=None, channels=None), fresh(), 15)
        self.assertEqual((silent["audio"], silent["shortest"]), (None, False))

    def test_verify_hands_the_audio_and_the_color_to_checks_of_their_own(self):
        src, rate = facts(), Fraction(30)
        got = {}

        def color(s, o, c=None):
            got["color"] = c
            return ["COLOR"]

        def audio(s, o, a=None):
            got["audio"] = a
            return ["AUDIO"]

        with mock.patch.object(tool, "verify_color", color), mock.patch.object(tool, "verify_audio", audio):
            problems = tool.verify(src, facts(width=162), rate, audio={"kind": "frame"}, color="PLAN")
            self.assertEqual(problems, ["size is 162x120, expected 160x120", "COLOR", "AUDIO"])
            self.assertEqual(got, {"color": "PLAN", "audio": {"kind": "frame"}})
            self.assertEqual(tool.verify(src, None, rate), ["the output has no readable video stream"])
        # the checks of this build: no color check without a color plan (the encode of before), the pixel format of
        # a 4:4:4 encode for an odd side (test_apply_hdr.py has the HDR ones), the audio of the copy
        self.assertEqual(tool.verify_color(src, facts(color_transfer="smpte2084", pix_fmt="gray")), [])
        self.assertEqual(tool.verify_color(src, facts(color_transfer="smpte2084", pix_fmt="gray"), None), [])
        odd = tool.encode_plan(argparse.Namespace(input="in.mp4", depth=16), src, (101, 61))
        self.assertEqual(tool.verify_color(src, facts(pix_fmt="yuv444p"), odd), [])
        self.assertEqual(tool.verify_color(src, facts(pix_fmt="yuv420p"), odd), ["pixel format is yuv420p, expected yuv444p"])
        silent = facts(has_audio=False, acodec=None, channels=None)
        self.assertEqual(tool.verify_audio(src, facts()), [])
        self.assertEqual(tool.verify_audio(src, silent), ["the audio stream was lost"])
        self.assertEqual(tool.verify_audio(src, facts(channels=1)), ["audio is aac 1ch, expected aac 2ch"])
        self.assertEqual(tool.verify_audio(silent, facts()), ["the output has audio the input did not"])
        self.assertEqual(tool.verify_audio(silent, silent, {"kind": "temporal"}), [])

    def test_the_audio_steps_run_in_order_and_a_failing_one_ends_the_run(self):
        scratch = tempfile.mkdtemp(dir=TMP)
        marker = os.path.join(scratch, "ran.txt")
        ok = [sys.executable, "-c", f"open({marker!r}, 'a').write('x')"]
        bad = [sys.executable, "-c", "import sys; sys.stderr.write('no samples here\\n'); sys.exit(4)"]
        tool.run_audio([], scratch)
        tool.run_audio([("audio-decode", ok), ("audio-retime", ok)], scratch)
        self.assertEqual(slurp(marker), b"xx")
        code, err = self.refused(tool.run_audio, [("audio-decode", ok), ("audio-retime", bad), ("audio-decode", ok)],
                                 scratch)
        self.assertEqual(code, 1)
        self.assertEqual(err.splitlines(), [f"[error] audio-retime: {os.path.basename(sys.executable)} exited with code 4",
                                            "  no samples here"])
        self.assertEqual(slurp(marker), b"xxx", "nothing after the failing step ran")
        out = io.StringIO()
        with mock.patch.object(tool, "CONTRACT", "frames/2"), mock.patch.object(tool, "JSON_OUTPUT", True), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as caught:
            tool.run_audio([("audio-retime", bad)], scratch)
        self.assertEqual(caught.exception.code, 1)
        error = json.loads(out.getvalue())["error"]
        self.assertEqual((error["code"], error["stage"], error["exit_code"], error["stderr_tail"]),
                         ("stage_failed", "audio-retime", 4, ["no samples here"]))


class SeamRunTests(Workspace):
    """the seams in a real process: the raw probe json, the param ids, and main() handing one work
    geometry and the plan to every seam."""

    SEAMS = ("preflight", "build_stages", "make_plan", "build_commands", "build_layer_commands", "decode_layers",
             "run_staged", "make_result")

    def setUp(self):
        super().setUp()
        if not TEMPORAL:
            self.skipTest("this vfx has no temporal self-test effects (build it from the current workspace)")

    def run_main(self, *argv):
        """the real main() in this process: (exit code, stdout, stderr), the module's globals put back."""
        names = ("CONTRACT", "JSON_OUTPUT", "FORCE", "STAGED")
        saved = {name: getattr(tool, name) for name in names}
        out, err = io.StringIO(), io.StringIO()
        try:
            with mock.patch.object(sys, "argv", ["apply.py", *argv]), mock.patch.dict(os.environ, {"VFX_PATH": VFX}), \
                    mock.patch.object(tempfile, "tempdir", self.dir), contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as caught:
                tool.main()
        finally:
            for name, value in saved.items():
                setattr(tool, name, value)
        return caught.exception.code, out.getvalue(), err.getvalue()

    def test_probe_keeps_the_raw_json_beside_the_facts_of_before(self):
        info = tool.probe(self.src)
        self.assertEqual(set(info), {"path", "has_video", "has_audio", "vcodec", "acodec", "pix_fmt", "video_index",
                                     "width", "height", "rotation", "fps", "rate", "vfr", "duration", "channels", "hdr",
                                     "size", "container", "nb_frames", "video_duration", "audio_duration",
                                     "color_space", "color_primaries", "color_transfer"})
        data = tool._PROBE_DATA[info["path"]]
        self.assertEqual([s["codec_type"] for s in data["streams"]], ["video", "audio"])
        self.assertEqual((data["streams"][1]["channels"], data["format"]["format_name"]), (info["channels"], info["container"]))
        # a file that cannot be probed has no entry, and loses the one of an earlier good probe
        other = self.path("other.mp4")
        shutil.copy(self.src, other)
        self.assertIsNotNone(tool.probe(other))
        self.assertIn(os.path.abspath(other), tool._PROBE_DATA)
        with open(other, "w") as f:
            f.write("not a video")
        self.assertIsNone(tool.probe(other))
        self.assertNotIn(os.path.abspath(other), tool._PROBE_DATA)

    def test_the_effect_listing_gives_the_param_ids(self):
        for entry in listed("--all"):
            with self.subTest(effect=entry["id"]):
                einfo = tool.effect_info(VFX, entry["id"])
                self.assertEqual(einfo["params"], [p["id"] for p in entry["params"]])
                self.assertEqual((einfo["kind"], einfo["layers"]), (entry.get("kind") or "frame", entry["layers"]))
        self.assertEqual(tool.effect_info(VFX, TEMPORAL)["params"], ["offset"])

    def test_main_hands_one_work_geometry_and_the_plan_to_every_seam(self):
        other = self.path("other.mp4")
        make_clip(other, audio=False, vf="negate")
        real = {name: getattr(tool, name) for name in self.SEAMS + ("work_geometry", "verify")}
        made, seen, verified = [], {}, []

        def making(*a, **k):
            made.append(real["work_geometry"](*a, **k))
            return made[-1]

        def watching(name):
            signature = inspect.signature(real[name])

            def wrapper(*a, **k):
                bound = signature.bind(*a, **k).arguments
                seen.setdefault(name, []).append(bound.get("work" if name == "build_stages" else "geo"))
                return real[name](*a, **k)
            return wrapper

        def verify(*a, **k):
            verified.append(inspect.signature(real["verify"]).bind(*a, **k).arguments)
            return real["verify"](*a, **k)

        runs = ((["--dry-run", "--json"], ()), (["--frames", "15", "--lossless", "--depth", "8", "-o", self.path("a.mkv")],
                                                 ("decode_layers", "verify")),
                (["--frames", "15", "--lossless", "--depth", "8", "-o", self.path("b.mkv"), "--post-stage",
                  stage_line(sys.executable, "-c", "import shutil, sys; shutil.copyfileobj(sys.stdin.buffer, sys.stdout.buffer)")],
                 ("decode_layers", "run_staged", "verify")))
        for flags, also in runs:
            with self.subTest(flags=flags):
                made.clear()
                seen.clear()
                verified.clear()
                with contextlib.ExitStack() as stack:
                    stack.enter_context(mock.patch.object(tool, "work_geometry", making))
                    stack.enter_context(mock.patch.object(tool, "verify", verify))
                    for name in self.SEAMS:
                        stack.enter_context(mock.patch.object(tool, name, watching(name)))
                    code, out, err = self.run_main(self.src, TEMPORAL, "--layer", f"overlay={other}", *flags)
                self.assertEqual(code, 0, err)
                self.assertEqual(len(made), 1, "main computes the work geometry once")
                expected = {"preflight", "build_stages", "make_plan", "build_commands", "build_layer_commands",
                            "make_result"} | (set(also) - {"verify"})
                self.assertEqual(set(seen), expected)
                for name, geos in seen.items():
                    self.assertTrue(all(g is made[0] for g in geos), name)
                if "verify" in also:
                    call, = verified
                    self.assertEqual((call["rate"], call["frames"], call["source_frames"], call["size"]),
                                     (Fraction(FPS), 15, 30, (160, 120)))
                    self.assertEqual((call["audio"]["kind"], call["audio"]["audio"]), ("temporal", "shortest"))
                    self.assertIsNone(call["color"], "an SDR run at even sides has no color plan to check")
                else:
                    self.assertEqual(verified, [])
        self.assertClean()



# ----------------------------------------------------------------------------
# --audio retime and drop
# ----------------------------------------------------------------------------

def raw_probe(path, streams, duration=None):
    """put a raw ffprobe answer for `path` into apply's cache, as probe() leaves it for a file it read."""
    tool._PROBE_DATA[os.path.abspath(path)] = {"streams": streams, "format": {"duration": duration}}


# a video stream and two audio streams: 44.1 kHz mono with tags and the aac start of 1024 samples, 48 kHz stereo
TWO_STREAMS = [
    {"index": 0, "codec_type": "video", "codec_name": "h264", "start_time": "0.000000"},
    {"index": 1, "codec_type": "audio", "codec_name": "aac", "channels": 1, "sample_rate": "44100",
     "channel_layout": "mono", "bit_rate": "64000", "start_time": "0.023220", "duration": "1.000000",
     "tags": {"language": "eng", "title": "Commentary"}},
    {"index": 2, "codec_type": "audio", "codec_name": "aac", "channels": 2, "sample_rate": "48000",
     "channel_layout": "stereo", "start_time": "0.000000", "tags": {"LANGUAGE": "fra", "DURATION": "00:00:01.010000000"}},
]


class AudioPlanTests(unittest.TestCase):
    """--audio retime and drop on facts alone: the plan, the steps, the parts of the encode, the report and the
    verify. --audio copy (and no flag) plans what it planned before."""

    SRC = os.path.abspath("in.mp4")

    def setUp(self):
        raw_probe(self.SRC, TWO_STREAMS)
        self.addCleanup(tool._PROBE_DATA.pop, self.SRC, None)

    def args(self, **over):
        base = dict(input="in.mp4", effect="retime", depth=16, param=["speed_percent=50"], frames=None, layer=[],
                    audio="retime", lossless=False, pre_stage=[], post_stage=[])
        base.update(over)
        return argparse.Namespace(**base)

    def info(self, **over):
        base = dict(path=self.SRC, nb_frames=30, vfr=False, video_index=0, video_duration=1.0, audio_duration=1.0,
                    color_space=None, color_primaries=None, color_transfer=None)
        base.update(over)
        return facts(**base)

    def plan(self, args=None, info=None, kind="temporal"):
        args, info = args or self.args(), info or self.info()
        return tool.make_plan(args, info, {"kind": kind, "layers": []}, [], os.path.join("S", "x"))

    def test_the_streams_of_the_raw_probe(self):
        streams = tool.audio_streams(self.info())
        self.assertEqual([(s["index"], s["channels"], s["sample_rate"], s["layout"], s["language"], s["title"])
                          for s in streams], [(1, 1, 44100, "mono", "eng", "Commentary"), (2, 2, 48000, "stereo", "fra", None)])
        self.assertEqual([(s["bit_rate"], s["start_time"], s["duration"]) for s in streams],
                         [(64000, 0.02322, 1.0), (None, 0.0, 1.01)], "the matroska DURATION tag counts")
        self.assertEqual(tool.audio_streams(facts(path=os.path.abspath("never.mp4"))), [])
        self.assertEqual(tool.audio_mode(self.args()), "retime")
        self.assertEqual(tool.audio_mode(self.args(audio="drop")), "drop")

    def test_a_clip_read_at_its_own_rate_is_told_to_the_time_map(self):
        args = self.args(fps=Fraction(60))
        plan = self.plan(args)
        steps = tool.audio_commands(args, self.info(), plan)
        cmd = steps[1][1]
        self.assertEqual(cmd[cmd.index("--clip-fps") + 1], tool.rate_text(Fraction(30)))
        self.assertEqual(cmd[cmd.index("--fps") + 1], tool.rate_text(Fraction(60)))
        self.assertEqual(plan["audio_clip_frames"], 30, "the frames the decode reads, not the output's")
        self.assertNotIn("--clip-fps", tool.audio_commands(self.args(), self.info(), self.plan())[1][1])

    def test_retime_plans_every_stream_in_order(self):
        plan = self.plan()
        a, b = plan["audio_streams"]
        scratch = os.path.join("S", "x")
        self.assertEqual((plan["audio_mode"], plan["audio_codec"], plan["audio_frames"], plan["audio_clip_frames"],
                          plan["audio"], plan["shortest"]), ("retime", "aac", 30, 30, "same", False))
        self.assertEqual((a["pcm"], a["retimed"], b["pcm"]), (os.path.join(scratch, "audio_1.f32"),
                                                              os.path.join(scratch, "audio_1_retimed.f32"),
                                                              os.path.join(scratch, "audio_2.f32")))
        self.assertEqual((a["offset"], b["offset"]), (0.02322, 0.0))
        self.assertEqual((a["bit_rate"], b["bit_rate"]), (64000, 192000), "the source's rate, else 96 kb/s a channel")
        self.assertEqual([t["what"] for t in plan["temp_files"][1:]],
                         ["audio stream 1, decoded to raw pcm_f32le",
                          "audio stream 1, retimed to the effect's time map (tools/audio_map.py)",
                          "audio stream 2, decoded to raw pcm_f32le",
                          "audio stream 2, retimed to the effect's time map (tools/audio_map.py)"])
        inputs, maps, codec = tool.audio_parts(self.args(), self.info(), plan)
        self.assertEqual(inputs, ["-f", "f32le", "-ar", "44100", "-ch_layout", "mono", "-i", a["retimed"],
                                  "-f", "f32le", "-ar", "48000", "-ch_layout", "stereo", "-i", b["retimed"]])
        self.assertEqual(maps, ["-map", "1:a:0", "-map", "2:a:0"])
        self.assertEqual(codec, ["-c:a", "aac", "-b:a:0", "64000", "-metadata:s:a:0", "language=eng",
                                 "-metadata:s:a:0", "title=Commentary", "-b:a:1", "192000", "-metadata:s:a:1",
                                 "language=fra"])
        steps = tool.audio_commands(self.args(), self.info(), plan)
        self.assertEqual([stage for stage, _ in steps], ["audio-decode", "audio-retime", "audio-retime"])
        self.assertEqual(steps[0][1], [tool.ffmpeg_path(), "-v", "error", "-nostdin", "-y", "-i", "in.mp4",
                                       "-map", "0:1", "-c:a", "pcm_f32le", "-f", "f32le", a["pcm"],
                                       "-map", "0:2", "-c:a", "pcm_f32le", "-f", "f32le", b["pcm"]])
        self.assertEqual(steps[1][1], [sys.executable, "-I", tool.AUDIO_MAP, "render", "--effect", "retime", "--fps",
                                       "30/1", "--frames", "30", "--clip-frames", "30", "--param", "speed_percent=50",
                                       "--rate", "44100", "--channels", "1", "--offset", "0.02322", "--in", a["pcm"],
                                       "--out", a["retimed"]])
        self.assertEqual(steps[2][1][-10:], ["--rate", "48000", "--channels", "2", "--offset", "0.0", "--in", b["pcm"],
                                             "--out", b["retimed"]])
        # --frames sets the length the map renders; the encode reads the raw video, then the two streams
        plan = self.plan(self.args(frames=45))
        self.assertEqual((plan["audio_frames"], plan["shortest"]), (45, False))
        encode = tool.build_commands(self.args(frames=45), self.info(), "vfx", "out.mp4", plan)[2]
        at = encode.index("-i")
        self.assertEqual(encode[at:at + 2], ["-i", "-"])
        self.assertEqual(encode[encode.index("-map"):encode.index("-map") + 6],
                         ["-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0"])
        self.assertNotIn("-shortest", encode)
        self.assertEqual(encode[encode.index("-c:a"):encode.index("-c:a") + 2], ["-c:a", "aac"])

    def test_lossless_is_flac_and_aac_takes_at_most_96_khz(self):
        plan = self.plan(self.args(lossless=True))
        self.assertEqual(plan["audio_codec"], "flac")
        self.assertEqual([s["bit_rate"] for s in plan["audio_streams"]], [None, None])
        self.assertEqual(tool.audio_parts(self.args(), self.info(), plan)[2],
                         ["-c:a", "flac", "-metadata:s:a:0", "language=eng", "-metadata:s:a:0", "title=Commentary",
                          "-metadata:s:a:1", "language=fra"])
        high = [TWO_STREAMS[0], dict(TWO_STREAMS[2], sample_rate="192000", channel_layout=None, tags={})]
        raw_probe(self.SRC, high)
        plan = self.plan()
        self.assertEqual(tool.audio_parts(self.args(), self.info(), plan),
                         (["-f", "f32le", "-ar", "192000", "-ac", "2", "-i", plan["audio_streams"][0]["retimed"]],
                          ["-map", "1:a:0"], ["-c:a", "aac", "-b:a:0", "192000", "-ar:a:0", "96000"]))
        flac = self.plan(self.args(lossless=True))
        self.assertEqual(tool.audio_parts(self.args(), self.info(), flac)[2], ["-c:a", "flac"])
        # a stream whose rate ffprobe cannot read is refused before any work
        raw_probe(self.SRC, [dict(TWO_STREAMS[2], sample_rate="N/A")])
        err = io.StringIO()
        with mock.patch.object(tool, "JSON_OUTPUT", False), mock.patch.object(tool, "CONTRACT", "frames/1"), \
                contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            self.plan()
        self.assertEqual(err.getvalue(), "[error] --audio retime: cannot read the sample rate of audio stream 2\n")

    def test_an_unknown_length_renders_past_the_end_and_is_cut_by_shortest(self):
        info = self.info(nb_frames=None, video_duration=1.0)
        plan = self.plan(info=info)
        self.assertEqual((plan["frames"], plan["audio_frames"], plan["audio_clip_frames"], plan["shortest"]),
                         (None, 31, 30, True))
        self.assertIn("-shortest", tool.build_commands(self.args(), info, "vfx", "out.mp4", plan)[2])
        self.assertEqual(tool.report_audio(self.args(), info, plan)[0],
                         ["audio retimed with the effect's time map (aac)", "audio cut at the end of the video (-shortest)"])
        # a frame effect renders the input's own frames
        frame = self.plan(self.args(effect="fx"), self.info(), kind="frame")
        self.assertEqual((frame["audio_frames"], frame["shortest"]), (30, False))
        self.assertEqual(tool.audio_commands(self.args(effect="fx"), self.info(), frame)[1][1][4:6], ["--effect", "fx"])

    def test_drop_and_an_input_without_audio_add_no_audio(self):
        drop = self.plan(self.args(audio="drop"))
        self.assertEqual((drop["audio_mode"], drop["audio_streams"], drop["audio"], drop["shortest"]),
                         ("drop", [], None, False))
        self.assertEqual(tool.audio_parts(self.args(audio="drop"), self.info(), drop), ([], [], []))
        self.assertEqual(tool.audio_commands(self.args(audio="drop"), self.info(), drop), [])
        self.assertEqual(len(drop["temp_files"]), 1, "the scratch folder only")
        self.assertEqual(tool.report_audio(self.args(), self.info(), drop),
                         (["audio dropped"], {"audio": {"plan": None, "shortest": False, "mode": "drop", "codec": None,
                                                        "streams": []}}))
        silent = self.info(has_audio=False, acodec=None, channels=None)
        for mode in ("drop", "retime"):
            plan = self.plan(self.args(audio=mode), silent)
            self.assertEqual(tool.audio_parts(self.args(audio=mode), silent, plan), ([], [], []))
            self.assertEqual(tool.report_audio(self.args(), silent, plan)[0], [])
        encode = tool.build_commands(self.args(audio="drop"), self.info(), "vfx", "out.mkv", drop)[2]
        self.assertEqual(encode.count("-i"), 1, "drop leaves out the audio input")
        self.assertEqual(encode.count("-map"), 1)

    def test_the_report_names_the_mode_the_codec_and_the_streams(self):
        plan = self.plan()
        phrases, keys = tool.report_audio(self.args(), self.info(), plan)
        self.assertEqual(phrases, ["audio retimed with the effect's time map (aac)"])
        self.assertEqual(keys, {"audio": {"plan": "same", "shortest": False, "mode": "retime", "codec": "aac", "streams": [
            {"index": 1, "channels": 1, "sample_rate": 44100, "layout": "mono", "bit_rate": 64000, "offset": 0.02322},
            {"index": 2, "channels": 2, "sample_rate": 48000, "layout": "stereo", "bit_rate": 192000, "offset": 0.0}]}})
        result = tool.make_result(self.args(dry_run=True), self.info(), {"kind": "temporal"}, "vfx", "out.mp4", plan, [])
        self.assertTrue(result["summary"].endswith(", temporal, 30 frames, audio retimed with the effect's time map "
                                                   "(aac), decode | vfx | encode -> out.mp4"), result["summary"])
        flac = tool.report_audio(self.args(), self.info(), self.plan(self.args(lossless=True)))[0]
        self.assertEqual(flac, ["audio retimed with the effect's time map (flac)"])

    def test_copy_plans_and_reports_what_it_did_before(self):
        for args in (self.args(audio="copy"), self.args(audio=None), argparse.Namespace(
                **{k: v for k, v in vars(self.args()).items() if k != "audio"})):
            plan = self.plan(args)
            self.assertFalse([k for k in plan if k.startswith("audio_")])
            self.assertEqual(tool.audio_parts(args, self.info(), plan), (["-i", "in.mp4"], ["-map", "1:a?"], ["-c:a", "copy"]))
            self.assertEqual(tool.audio_commands(args, self.info(), plan), [])
            self.assertEqual(tool.report_audio(args, self.info(), plan), ([], {"audio": {"plan": None, "shortest": False}}))

    def test_the_verify_of_a_retimed_output(self):
        plan = self.plan()
        out_path = os.path.abspath("out.mp4")
        self.addCleanup(tool._PROBE_DATA.pop, out_path, None)

        def problems(*streams, video=None, mode_plan=None):
            raw_probe(out_path, [{"index": 0, "codec_type": "video", "duration": video}] + list(streams))
            out = facts(path=out_path, video_index=0, has_audio=bool(streams))
            return tool.verify_audio(facts(), out, mode_plan or plan)

        mono = {"index": 1, "codec_type": "audio", "codec_name": "aac", "channels": 1, "duration": "1.000000"}
        stereo = {"index": 2, "codec_type": "audio", "codec_name": "aac", "channels": 2,
                  "tags": {"DURATION": "00:00:01.021000000"}}
        self.assertEqual(problems(mono, stereo), [])
        self.assertEqual(problems(mono), ["the output has 1 audio streams, expected 2"])
        self.assertEqual(problems(dict(mono, codec_name="mp3"), stereo), ["audio stream 1 is mp3, expected aac"])
        self.assertEqual(problems(mono, dict(stereo, channels=1)), ["audio stream 2 has 1 channels, expected 2"])
        # within the larger of a frame and 50 ms of the 30 frames at 30 fps
        self.assertEqual(problems(dict(mono, duration="1.049"), stereo), [])
        self.assertEqual(problems(dict(mono, duration="1.051"), stereo),
                         ["audio stream 1 lasts 1.051s, expected 1.000s within 0.050s"])
        self.assertEqual(problems(dict(mono, duration=None), stereo), ["audio stream 1 has no readable duration"])
        # a run cut by -shortest compares the audio with the video it got
        cut = self.plan(info=self.info(nb_frames=None))
        self.assertEqual(problems(dict(mono, duration="1.200"), dict(stereo, tags={}, duration="1.230"), video="1.2",
                                  mode_plan=cut), [])
        # drop, and retime of an input without audio, leave none
        drop = self.plan(self.args(audio="drop"))
        self.assertEqual(problems(mode_plan=drop), [])
        self.assertEqual(problems(mono, mode_plan=drop), ["the output has audio, which --audio drop leaves out"])
        silent = self.plan(info=self.info(has_audio=False, acodec=None, channels=None))
        self.assertEqual(problems(mono, mode_plan=silent), ["the output has audio the input did not"])

    def test_an_uncounted_input_takes_the_video_streams_own_duration(self):
        # matroska: no frame count, the video's DURATION tag says 1 s, the file lasts as long as its 2.02 s audio
        video = dict(TWO_STREAMS[0], tags={"DURATION": "00:00:01.000000000"})
        raw_probe(self.SRC, [video, dict(TWO_STREAMS[2], tags={"DURATION": "00:00:02.020000000"})], duration="2.02")
        info = self.info(nb_frames=None, video_duration=None, duration=2.02, rate=Fraction(24), fps=24.0)
        plan = self.plan(self.args(), info)
        self.assertEqual((plan["audio_frames"], plan["audio_clip_frames"], plan["shortest"]), (25, 24, True))
        self.assertEqual(tool.video_seconds(info), 1.0)

    def test_the_chained_geometry_of_the_stages_is_not_asked_again(self):
        args = self.args(post_stage=[tool.StageCommand("cmd", "declared")])
        chain = {"vfx": {"frames": 30}, "out": {"frames": 30}}
        with mock.patch.object(tool, "build_stages", side_effect=AssertionError("asked again")):
            plan = tool.make_plan(args, self.info(), {"kind": "temporal", "layers": []}, [], "S", chain=chain)
        self.assertEqual(plan["audio_frames"], 30)
        err = io.StringIO()
        with mock.patch.object(tool, "JSON_OUTPUT", False), mock.patch.object(tool, "CONTRACT", "frames/2"), \
                contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            tool.make_plan(args, self.info(), {"kind": "temporal", "layers": []}, [], "S",
                           chain={"vfx": {"frames": 30}, "out": {"frames": 20}})
        self.assertIn("post-stages change the frames of the output from 30 to 20", err.getvalue())

    def test_the_length_of_a_retimed_or_dropped_output_is_its_video(self):
        # 30 frames of video in a file whose audio lasts 2 s: the output lasts 1 s whatever the input
        src = self.info(duration=2.0)
        out_path = os.path.abspath("long.mp4")
        self.addCleanup(tool._PROBE_DATA.pop, out_path, None)
        raw_probe(out_path, [{"index": 0, "codec_type": "video"}])
        out = facts(path=out_path, video_index=0, has_audio=False, duration=1.0, fps=30.0, width=160, height=120)
        drop = self.plan(self.args(audio="drop"), src)
        self.assertEqual(drop["audio_length"], 1.0)
        self.assertEqual(tool.verify(src, out, Fraction(30), source_frames=30, audio=drop), [])
        self.assertEqual(tool.verify(src, dict(out, duration=1.05), Fraction(30), source_frames=30, audio=drop),
                         ["duration is 1.050s, expected 1.000s within one frame (0.033s)"])
        # copy keeps the input's length
        self.assertEqual(tool.verify(dict(src, has_audio=False), out, Fraction(30), source_frames=30),
                         ["duration is 1.000s, expected 2.000s within one frame (0.033s)"])
        # a frame effect writes every frame the decode gives
        frame = self.plan(self.args(audio="drop", effect="fx"), src, kind="frame")
        self.assertEqual(frame["audio_length"], 1.0)

    def test_a_stream_cut_by_shortest_may_end_a_frame_of_its_encoder_early(self):
        cut = self.plan(self.args(lossless=True), self.info(nb_frames=None))
        self.assertTrue(cut["shortest"])
        out_path = os.path.abspath("cut.mkv")
        self.addCleanup(tool._PROBE_DATA.pop, out_path, None)

        def problems(seconds, plan=cut):
            raw_probe(out_path, [{"index": 0, "codec_type": "video", "duration": "1.000"}] + [
                {"index": k, "codec_type": "audio", "codec_name": plan["audio_codec"], "channels": c,
                 "duration": seconds} for k, c in ((1, 1), (2, 2))])
            return tool.verify_audio(facts(), facts(path=out_path, video_index=0), plan)

        # flac ends on its 4608 sample frames (104.5 ms at 44.1 kHz)
        self.assertEqual(problems("0.929"), [])
        self.assertEqual(problems("0.890"), ["audio stream 1 lasts 0.890s, expected 1.000s within 0.105s",
                                             "audio stream 2 lasts 0.890s, expected 1.000s within 0.105s"])
        # an exact count is not cut: the window stays the larger of a frame and 50 ms
        exact = self.plan(self.args(lossless=True))
        self.assertEqual(problems("0.929", exact)[0], "audio stream 1 lasts 0.929s, expected 1.000s within 0.050s")
        self.assertAlmostEqual(tool.codec_frame_seconds("aac", 48000), 1024 / 48000)

    def test_a_relabelled_layout_keeps_its_channel_count(self):
        # the aac encoder writes 5.1(side) as 5.1: the verify compares the channel count
        raw_probe(self.SRC, [TWO_STREAMS[0], dict(TWO_STREAMS[2], channels=6, channel_layout="5.1(side)")])
        plan = self.plan()
        self.assertEqual(plan["audio_streams"][0]["layout"], "5.1(side)")
        out_path = os.path.abspath("six.mp4")
        self.addCleanup(tool._PROBE_DATA.pop, out_path, None)
        raw_probe(out_path, [{"index": 0, "codec_type": "video"},
                             {"index": 1, "codec_type": "audio", "codec_name": "aac", "channels": 6,
                              "channel_layout": "5.1", "duration": "1.000"}])
        self.assertEqual(tool.verify_audio(facts(), facts(path=out_path, video_index=0), plan), [])

    def test_a_missing_audio_map_is_refused_up_front(self):
        missing = os.path.abspath("no_such_audio_map.py")
        err = io.StringIO()
        with mock.patch.object(tool, "AUDIO_MAP", missing), mock.patch.object(tool, "JSON_OUTPUT", False), \
                mock.patch.object(tool, "CONTRACT", "frames/2"), contextlib.redirect_stderr(err), \
                self.assertRaises(SystemExit) as caught:
            tool.require_audio_map(self.args(), self.info())
        self.assertEqual(caught.exception.code, tool.EXIT_CODE["tool_missing"])
        self.assertTrue(err.getvalue().startswith("error: tool_missing: --audio retime needs "), err.getvalue())
        self.assertIn("no_such_audio_map.py, which is missing", err.getvalue())
        with mock.patch.object(tool, "AUDIO_MAP", missing):
            for args, info in ((self.args(audio="drop"), self.info()), (self.args(audio="copy"), self.info()),
                               (self.args(), self.info(has_audio=False))):
                tool.require_audio_map(args, info)


class DecodeLeadTests(Workspace):
    """the decode of the clip as cfr_decode and decoded_clip work it out, against real decodes: a video stream
    that starts after another stream gets copies of its first frame in front, by the arithmetic of ffmpeg's
    video sync, and the frame count is the one the decode writes."""

    # h264 with two b-frames every time, at a quality that keeps flat frames within a level or two
    H264 = ("-c:v", "libx264", "-crf", "4", "-bf", "2", "-x264-params", "b-adapt=0", "-pix_fmt", "yuv420p", "-c:a", "aac")

    @classmethod
    def setUpClass(cls):
        cls.fixtures = tempfile.mkdtemp(dir=TMP)
        # 8 bit frame index (level 8 x the frame number) with b-frames, and a tone
        cls.base = os.path.join(cls.fixtures, "base.mp4")
        ff("-f", "lavfi", "-i", "color=black:s=64x48:r=24:d=1,format=gray,geq=lum=N*8", "-f", "lavfi", "-i",
           "sine=r=48000:d=1", *cls.H264, "-shortest", cls.base)
        # the same without the edit list of its b-frames: the video starts 2 frames after the audio
        cls.noedit = os.path.join(cls.fixtures, "noedit.mp4")
        ff("-f", "lavfi", "-i", "color=black:s=64x48:r=24:d=1,format=gray,geq=lum=N*8", "-f", "lavfi", "-i",
           "sine=r=48000:d=1", *cls.H264, "-use_editlist", "0", cls.noedit)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.fixtures, ignore_errors=True)

    def delayed(self, delay, ext):
        """the base clip with its video `delay` seconds late, streams copied (so the delay stays fractional)."""
        path = self.path(f"d{delay}.{ext}")
        ff("-itsoffset", repr(delay), "-i", self.base, "-i", self.base, "-map", "0:v", "-map", "1:a", "-c", "copy", path)
        return path

    def real_decode(self, path, info, geo):
        """the frame index each frame of the run's decode shows."""
        graph = ",".join(tool.decode_filters(argparse.Namespace(), info, geo))
        data = subprocess.run([tool.ffmpeg_path(), "-v", "error", "-nostdin", "-i", path, "-map",
                               f"0:{info['video_index']}", "-an", "-sn", "-dn", "-vf", graph, "-pix_fmt", "gray",
                               "-f", "rawvideo", "-"], shell=False, capture_output=True, check=True).stdout
        size = 64 * 48
        return [round(sum(data[n * size:(n + 1) * size]) / size / 8) for n in range(len(data) // size)]

    def test_the_lead_and_the_count_are_those_of_a_real_decode(self):
        # the edit list of the b-frames left out (2 frames late), and mpeg-ts, whose start ffmpeg takes again
        # from the streams the decode reads
        cases = [(self.base, 0), (self.noedit, 2), (self.delayed(0.07, "ts"), 0)]
        # 0.48 and 0.504 frames late in mp4 (12288 ticks a second); 0.504 in matroska, whose frames of a whole
        # 41 ms sum to under 1.1 frames; 1.5 exactly (a midpoint); 2.4; 1.512 in matroska, whose ms rounding
        # can add a copy after the first frame; 8 in mov and nut. the lead a decode shows is the shift most of
        # its frames have
        for delay, ext, lead in ((0.02, "mp4", 0), (0.021, "mp4", 1), (0.021, "mkv", 0), (0.0625, "mp4", 2),
                                 (0.1, "mp4", 2), (0.1, "mkv", 2), (0.0625, "mkv", None), (0.3333, "mov", 8),
                                 (0.3333, "nut", None)):
            cases.append((self.delayed(delay, ext), lead))
        for path, lead in cases:
            with self.subTest(path=os.path.basename(path)):
                info = tool.probe(path)
                geo = tool.work_geometry(argparse.Namespace(size=None, fps=None, param=[]), info)
                frames, got = tool.decoded_clip(info, geo)
                shown = self.real_decode(path, info, geo)
                shifts = [p - s for p, s in enumerate(shown)]
                seen = max(sorted(set(shifts)), key=shifts.count)
                self.assertEqual((frames, got), (len(shown), seen), shown)
                if lead is not None:
                    self.assertEqual(got, lead)
                # clip frame p shows the input's frame p - lead for all of the clip past the lead but a frame
                self.assertGreaterEqual(shifts.count(got), len(shown) - got - 2, shown)

    def test_a_delayed_video_moves_the_audio_by_the_lead(self):
        for effect, frames in ((TEMPORAL or "retime", 24), (EFFECT, 26)):
            with self.subTest(effect=effect):
                proc = apply(self.noedit, effect, "--audio", "retime", "--dry-run", "--json", tmp=self.dir)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                report = json.loads(proc.stdout)
                self.assertLess(abs(report["audio"]["streams"][0]["offset"]), 1e-6, "2 frames of lead cancel the delay")
                self.assertIn(f" --frames {frames} --clip-frames 26 ", report["commands"][1])


def index_clip(path, seconds, rate=24, sample_rate=48000, video_delay=0.0):
    """the frame index clip: gray16 frames whose level is 512 x the frame number, and a flac tone whose sample at
    time t is t / 8, so a picture and a sample each tell the clip time they come from. `video_delay` starts the
    video that many seconds after the audio."""
    delay = ["-itsoffset", repr(video_delay)] if video_delay else []
    ff(*delay, "-f", "lavfi", "-i", f"color=black:s=64x48:r={rate}:d={seconds},format=gray16le,geq=lum=N*512",
       "-f", "lavfi", "-i", f"aevalsrc=t/8:s={sample_rate}:d={seconds}", "-map", "0", "-map", "1", "-c:v", "ffv1",
       "-c:a", "flac", "-sample_fmt", "s32", path)


def stream_starts(path):
    """(the video stream's start_time, the first audio stream's) of a file, from ffprobe."""
    tool.probe(path)
    streams = tool._PROBE_DATA[os.path.abspath(path)]["streams"]
    return tuple(float(next(s for s in streams if s["codec_type"] == kind).get("start_time", 0))
                 for kind in ("video", "audio"))


def decoded_audio(path, stream=0):
    from array import array
    data = subprocess.run([tool.ffmpeg_path(), "-v", "error", "-i", path, "-map", f"0:a:{stream}", "-f", "f32le", "-"],
                          shell=False, capture_output=True, check=True).stdout
    samples = array("f")
    samples.frombytes(data)
    return samples


def clip_sync(path, rate, sample_rate, clip_rate=None, offset=0.0):
    """(the largest gap in seconds between the clip time a frame of the output shows and the clip time its audio
    plays at the frame's start, frames compared) for an output of the index clip at `rate` made from an index
    clip at `clip_rate` (default `rate`) whose audio starts `offset` seconds after its video (the source's audio
    start_time minus its video's). frames that show no whole picture or sit within 4 ms of silence (a hold, a
    clip edge, a fade) are left out."""
    clip_rate = clip_rate or rate
    import struct
    video = subprocess.run([tool.ffmpeg_path(), "-v", "error", "-i", path, "-map", "0:v:0", "-f", "rawvideo",
                            "-pix_fmt", "rgba64le", "-"], shell=False, capture_output=True, check=True).stdout
    audio = decoded_audio(path)
    size = 64 * 48 * 8
    near = int(0.004 * sample_rate)
    worst, compared = 0.0, 0
    for n in range(len(video) // size):
        level, _, _, alpha = struct.unpack_from("<HHHH", video, n * size)
        i = n * sample_rate // rate
        if alpha != 65535 or level == 0 or i >= len(audio) or not all(audio[max(0, i - near):i + near]):
            continue
        worst = max(worst, abs(8 * audio[i] + offset - level / 512 / clip_rate))
        compared += 1
    return worst, compared


class AudioRunTests(Workspace):
    """--audio retime, drop and copy in real runs: the audio plays the clip time the picture shows within 1 ms at
    every frame, aac clicks land within 2 ms, every stream keeps its place, rate, layout and tags, drop writes no
    audio, a declared stage that changes the frame count is a usage error under frames/2, and a failing audio
    step is stage_failed at its stage."""

    @classmethod
    def setUpClass(cls):
        cls.fixtures = tempfile.mkdtemp(dir=TMP)
        cls.index = os.path.join(cls.fixtures, "index.mkv")
        index_clip(cls.index, 3)
        # a click of 2 samples every 0.25 s over a quiet tone, h264 and aac in mp4
        cls.clicks = os.path.join(cls.fixtures, "clicks.mp4")
        ff("-f", "lavfi", "-i", "testsrc=size=64x48:rate=24:duration=2", "-f", "lavfi", "-i",
           "aevalsrc='0.01*sin(2*PI*300*t)+if(lt(mod(n\\,12000)\\,2)\\,0.9\\,0)':s=48000:d=2", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", cls.clicks)
        # 44.1 kHz mono with a language and a title, then 48 kHz stereo
        cls.two = os.path.join(cls.fixtures, "two.mp4")
        ff("-f", "lavfi", "-i", "testsrc=size=64x48:rate=24:duration=1", "-f", "lavfi", "-i",
           "sine=frequency=440:sample_rate=44100:duration=1", "-f", "lavfi", "-i",
           "aevalsrc='sin(2*PI*660*t)|0.5*sin(2*PI*880*t)':s=48000:d=1", "-map", "0", "-map", "1", "-map", "2",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-metadata:s:a:0", "language=eng",
           "-metadata:s:a:0", "title=Commentary", "-metadata:s:a:1", "language=fra", "-shortest", cls.two)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.fixtures, ignore_errors=True)

    def retimed(self, src, effect, *flags, name="out.mkv"):
        out = self.path(name)
        proc = apply(src, effect, "--audio", "retime", *flags, "-o", out, "--json", tmp=self.dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return out, json.loads(proc.stdout)

    def test_the_audio_plays_the_clip_time_the_picture_shows(self):
        cases = {
            "ramp": ("retime", "speed_percent=100@0;25@24;100@60"),
            "frame_num": ("retime", "time_remap_mode=2", "frame_num=0@0;30@20;30@30h;60@40;10@72"),
            "offset": ("retime", "src_time_offset=10", "speed_percent=80"),
            "rates": ("retime", "in_fps_is_out_fps=0", "input_frame_rate=12", "src_time_offset=3",
                      "speed_percent=60"),
            "retime-fgbg": ("retime-fgbg", "speed_percent=100@0;40@30;150@60"),
        }
        for name, (effect, *params) in cases.items():
            with self.subTest(case=name):
                flags = ["--lossless", "--depth", "16", "--param", "vectors=1", "--param", "frame_interp=2",
                         "--frames", "72"]
                for p in params:
                    flags += ["--param", p]
                out, report = self.retimed(self.index, effect, *flags, name=f"{name}.mkv")
                self.assertEqual((report["audio"]["mode"], report["audio"]["codec"]), ("retime", "flac"))
                worst, compared = clip_sync(out, 24, 48000)
                self.assertLess(worst, 0.001)
                self.assertGreater(compared, 40)
                self.assertEqual(len(decoded_audio(out)), 72 * 2000, "floor(N * SR / F + 1/2) samples")
        self.assertClean()

    def test_aac_clicks_at_half_speed_land_within_2_ms(self):
        out, report = self.retimed(self.clicks, "retime", "--param", "speed_percent=50", "--param", "vectors=1",
                                   name="slow.mp4")
        self.assertEqual(report["audio"]["streams"][0]["offset"], 0.0)
        audio = decoded_audio(out)
        self.assertEqual(len(audio), 96000)
        # the click at source time 0.25 k shows at output time 0.5 k
        for k in range(4):
            at = k * 24000
            peak = max(range(max(0, at - 480), at + 480), key=lambda i: abs(audio[i]))
            self.assertLessEqual(abs(peak - at), 96, f"click {k} at sample {peak}, expected {at}")
        self.assertClean()

    def test_every_stream_keeps_its_place_rate_layout_and_tags(self):
        out, report = self.retimed(self.two, "retime", "--param", "speed_percent=200", "--param", "vectors=1",
                                   name="two.mp4")
        self.assertEqual([(s["index"], s["channels"], s["sample_rate"]) for s in report["audio"]["streams"]],
                         [(1, 1, 44100), (2, 2, 48000)])
        streams = tool.audio_streams(tool.probe(out))
        self.assertEqual([(s["codec"], s["channels"], s["sample_rate"], s["layout"], s["language"], s["title"])
                          for s in streams], [("aac", 1, 44100, "mono", "eng", "Commentary"),
                                              ("aac", 2, 48000, "stereo", "fra", None)])
        for s in streams:
            self.assertLessEqual(abs(s["duration"] - 1.0), 0.05)
        self.assertEqual((len(decoded_audio(out, 0)), len(decoded_audio(out, 1))), (44100, 2 * 48000), "interleaved")
        self.assertClean()

    def test_a_frame_effect_at_its_identity_map_keeps_every_sample(self):
        # h264 and 24 bit flac in mp4: the container counts the frames, so the length is exact
        src = self.path("flac.mp4")
        ff("-f", "lavfi", "-i", "testsrc=size=64x48:rate=24:duration=1", "-f", "lavfi", "-i", "aevalsrc=t/8:s=48000:d=1",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "flac", "-sample_fmt", "s32", src)
        out, report = self.retimed(src, EFFECT, "--lossless", name="same.mkv")
        self.assertEqual((report["audio"]["codec"], report["audio"]["shortest"]), ("flac", False))
        self.assertEqual(decoded_audio(out).tobytes(), decoded_audio(src).tobytes())
        # matroska gives no frame count: its packets give it, so every sample comes back as it was here too
        out, report = self.retimed(self.index, EFFECT, "--lossless", name="counted.mkv")
        self.assertFalse(report["audio"]["shortest"])
        self.assertEqual(decoded_audio(out).tobytes(), decoded_audio(self.index).tobytes())

    def test_an_input_without_a_frame_count_is_counted_from_its_packets(self):
        # matroska gives no frame count: the packets of the video tell the frames vfx renders, so no -shortest
        out, report = self.retimed(self.index, "retime", "--lossless", "--param", "speed_percent=50",
                                   "--param", "vectors=1", name="counted.mkv")
        self.assertEqual((report["frames"], report["audio"]["shortest"]), (None, False))
        self.assertNotIn("-shortest", report["commands"][-1])
        self.assertIn(" --frames 72 --clip-frames 72 ", report["commands"][1])
        self.assertEqual(len(decoded_audio(out)), 72 * 2000)
        self.assertClean()

    def test_drop_writes_no_audio(self):
        out = self.path("dropped.mp4")
        proc = apply(self.src, TEMPORAL or EFFECT, "--audio", "drop", "-o", out, "--json", tmp=self.dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["audio"], {"plan": None, "shortest": False, "mode": "drop", "codec": None, "streams": []})
        self.assertIn(", audio dropped, ", report["summary"])
        self.assertFalse(tool.probe(out)["has_audio"])
        self.assertTrue(all("audio_map" not in c and "1:a" not in c for c in report["commands"]))
        self.assertClean()

    def test_the_default_dry_run_is_unchanged(self):
        for effect in (EFFECT, "retime"):
            plain = apply(self.src, effect, "--dry-run", "--json", tmp=self.dir)
            report = json.loads(plain.stdout)
            self.assertEqual(set(report["audio"]), {"plan", "shortest"})
            self.assertTrue(all("audio_map" not in c for c in report["commands"]))
            self.assertIn("-c:a copy", report["commands"][-1])
            retimed = json.loads(apply(self.src, effect, "--dry-run", "--json", "--audio", "retime", tmp=self.dir).stdout)
            # retime adds the two audio steps before the pipe and changes only the encode of the pipe
            self.assertEqual(len(retimed["commands"]), len(report["commands"]) + 2)
            self.assertEqual(retimed["commands"][2:-1], report["commands"][:-1])
            self.assertIn("audio_map.py", retimed["commands"][1])

    def describe_stage(self, frames):
        """a declared stage that copies its frames and answers a describe of the clip with `frames` frames."""
        script, answer = self.path("declared.py"), self.path(f"answer{frames}.json")
        with open(script, "w") as f:
            f.write("import json, shutil, sys\nif '--describe' in sys.argv:\n    print(open(sys.argv[1]).read())\n"
                    "    sys.exit(0)\nshutil.copyfileobj(sys.stdin.buffer, sys.stdout.buffer)\n")
        with open(answer, "w") as f:
            json.dump({"ok": True, "contract": "frames/2", "stage": "fixture", "width": 160, "height": 120, "depth": 16,
                       "fps": "30/1", "frames": frames}, f)
        return [stage_line(sys.executable, script, answer), "--geometry", "declared"]

    def test_a_declared_stage_that_changes_the_frame_count_is_a_usage_error(self):
        framed = ["--contract", "frames/2", "--json", "--dry-run"]
        effect = TEMPORAL or "retime"
        for where, extra, frames, words in (("--post-stage", [], 20, "post-stages change the frames of the output "
                                                                     "from 30 to 20"),
                                            ("--post-stage", [], None, "from 30 to an unknown count"),
                                            ("--pre-stage", ["--frames", "30"], 20, "pre-stages change the frames vfx "
                                                                                    "reads from 30 to 20")):
            with self.subTest(where=where, frames=frames):
                proc = apply(self.src, effect, "--audio", "retime", *extra, where, *self.describe_stage(frames), *framed,
                             tmp=self.dir)
                self.assertEqual(proc.returncode, 2, proc.stderr)
                error = json.loads(proc.stdout)["error"]
                self.assertEqual(error["code"], "usage")
                self.assertIn(words, error["message"])
                # copy and drop do not mind
                for mode in ("copy", "drop"):
                    self.assertEqual(apply(self.src, effect, "--audio", mode, *extra, where, *self.describe_stage(frames),
                                           *framed, tmp=self.dir).returncode, 0)
        kept = apply(self.src, effect, "--audio", "retime", "--post-stage", *self.describe_stage(30), *framed, tmp=self.dir)
        self.assertEqual(kept.returncode, 0, kept.stderr)
        self.assertEqual(json.loads(kept.stdout)["audio"]["mode"], "retime")

    def test_a_failing_audio_step_is_stage_failed_at_its_stage(self):
        if not TEMPORAL:
            self.skipTest("this vfx has no temporal self-test effects")
        failing = self.path("failing.py")
        with open(failing, "w") as f:
            f.write("import sys\nsys.stderr.write('cannot map this\\n')\nsys.exit(3)\n")
        missing = [{"index": 9, "codec": "aac", "channels": 2, "sample_rate": 48000, "layout": "stereo",
                    "bit_rate": None, "start_time": 0.0, "duration": 1.0, "language": None, "title": None}]
        for stage, patch in (("audio-decode", mock.patch.object(tool, "audio_streams", lambda info: list(missing))),
                             ("audio-retime", mock.patch.object(tool, "AUDIO_MAP", failing))):
            with self.subTest(stage=stage):
                with patch:
                    code, out, err = SeamRunTests.run_main(self, self.src, TEMPORAL, "--audio", "retime", "--lossless",
                                                           "--depth", "8", "-o", self.path(f"{stage}.mkv"), "--json",
                                                           "--contract", "frames/2")
                self.assertEqual(code, 1, err)
                error = json.loads(out)["error"]
                self.assertEqual((error["code"], error["stage"]), ("stage_failed", stage))
                if stage == "audio-retime":
                    self.assertEqual((error["exit_code"], error["stderr_tail"]), (3, ["cannot map this"]))
                self.assertFalse(os.path.exists(self.path(f"{stage}.mkv")))
        self.assertClean()

    def test_a_missing_audio_map_is_tool_missing_before_any_work(self):
        if not TEMPORAL:
            self.skipTest("this vfx has no temporal self-test effects")
        with mock.patch.object(tool, "AUDIO_MAP", self.path("no_audio_map.py")), \
                mock.patch.object(tool, "audio_commands", side_effect=AssertionError("planned the audio steps")):
            code, out, err = SeamRunTests.run_main(self, self.src, TEMPORAL, "--audio", "retime", "-o",
                                                   self.path("none.mp4"), "--json", "--contract", "frames/2")
        self.assertEqual(code, tool.EXIT_CODE["tool_missing"], err)
        error = json.loads(out)["error"]
        self.assertEqual((error["code"], error["stage"]), ("tool_missing", "audio-retime"))
        self.assertFalse(os.path.exists(self.path("none.mp4")))

    def test_a_video_that_starts_after_the_audio_stays_in_sync(self):
        # the video 0.1 s late in matroska (0.083 s on the frame grid): the decode puts 2 copies of its first frame
        # in front, and the audio follows the frames as they come out of the decode
        src = self.path("late.mkv")
        index_clip(src, 3, video_delay=0.1)
        video, audio = stream_starts(src)
        self.assertGreater(video - audio, 0.05)
        for name, params in (("ramp", ["speed_percent=100@0;25@24;100@60"]), ("same", [])):
            with self.subTest(case=name):
                flags = ["--lossless", "--depth", "16", "--param", "vectors=1", "--param", "frame_interp=2",
                         "--frames", "72"]
                for p in params:
                    flags += ["--param", p]
                out, report = self.retimed(src, "retime", *flags, name=f"late_{name}.mkv")
                worst, compared = clip_sync(out, 24, 48000, offset=audio - video)
                self.assertLess(worst, 0.001)
                self.assertGreater(compared, 40)
        self.assertClean()

    def test_audio_that_outlasts_the_video_verifies_in_every_mode(self):
        # 1 s of video and 2 s of aac in mp4: the output lasts as long as its video
        src = self.path("long.mp4")
        ff("-f", "lavfi", "-i", "testsrc=size=64x48:rate=24:duration=1", "-f", "lavfi", "-i",
           "sine=frequency=440:sample_rate=48000:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
           src)
        effects = [EFFECT] + ([TEMPORAL] if TEMPORAL else [])
        for effect in effects:
            for mode in ("retime", "drop", "copy"):
                with self.subTest(effect=effect, mode=mode):
                    out = self.path(f"long_{effect}_{mode}.mp4")
                    proc = apply(src, effect, "--audio", mode, "-o", out, "--json", tmp=self.dir)
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    if mode == "retime":
                        streams = tool.audio_streams(tool.probe(out))
                        self.assertLessEqual(abs(streams[0]["duration"] - 1.0), 0.05)
        self.assertClean()

    def test_lossless_audio_of_an_uncounted_input_has_every_sample(self):
        # h264 at 25 fps and flac mono in matroska: no frame count, so the packets give it and nothing is cut
        for rate in (44100, 48000):
            with self.subTest(sample_rate=rate):
                src = self.path(f"flac{rate}.mkv")
                ff("-f", "lavfi", "-i", "testsrc=size=64x48:rate=25:duration=1", "-f", "lavfi", "-i",
                   f"sine=frequency=440:sample_rate={rate}:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                   "-c:a", "flac", src)
                out, report = self.retimed(src, "retime", "--lossless", "--param", "vectors=1", name=f"flac{rate}_out.mkv")
                self.assertFalse(report["audio"]["shortest"])
                self.assertEqual(decoded_audio(out).tobytes(), decoded_audio(src).tobytes())
        self.assertClean()

    def test_a_rate_change_plays_the_audio_at_the_clip_time(self):
        # the 24 fps index clip rendered at 48 fps: the effect reads the clip at its own rate (--clip-fps)
        flags = ["--lossless", "--depth", "16", "--fps", "48", "--param", "vectors=1", "--param", "frame_interp=2",
                 "--frames", "144"]
        for name, params in (("half", ["speed_percent=50"]), ("ramp", ["speed_percent=100@0;25@48;100@120"])):
            with self.subTest(case=name):
                flags_case = flags + [word for p in params for word in ("--param", p)]
                out, report = self.retimed(self.index, "retime", *flags_case, name=f"fps48_{name}.mkv")
                self.assertIn(" --fps 48/1 ", report["commands"][1])
                self.assertIn(" --clip-fps 24/1 ", report["commands"][1])
                worst, compared = clip_sync(out, 48, 48000, clip_rate=24)
                self.assertLess(worst, 0.001)
                self.assertGreater(compared, 80)
                self.assertEqual(len(decoded_audio(out)), 144 * 1000)
        self.assertClean()

    def test_whole_clip_frame_steps_stay_within_one_clip_frame(self):
        # with input_fields, or an input_frame_rate above the output's, the picture steps in whole clip frames
        # while the audio slides along the continuous time map: the gap stays under one clip frame
        for name, effect, params in (("fields", "retime", ["input_fields=3"]),
                                     ("rate48", "retime", ["in_fps_is_out_fps=0", "input_frame_rate=48"]),
                                     ("fgbg48", "retime-fgbg", ["in_fps_is_out_fps=0", "input_frame_rate=48",
                                                                "input_fields=3"])):
            with self.subTest(case=name):
                flags = ["--lossless", "--depth", "16", "--param", "vectors=1", "--param", "frame_interp=2",
                         "--param", "speed_percent=50", "--frames", "60"]
                for p in params:
                    flags += ["--param", p]
                out, report = self.retimed(self.index, effect, *flags, name=f"steps_{name}.mkv")
                worst, compared = clip_sync(out, 24, 48000)
                self.assertLess(worst, 1 / 24 + 0.001)
                self.assertGreater(compared, 30)
        self.assertClean()


if __name__ == "__main__":
    unittest.main(verbosity=2)
