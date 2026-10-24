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
import io
import json
import os
import re
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

    def test_odd_frame_sizes_fail_early_with_a_clear_message(self):
        odd = self.path("odd.mp4")
        make_clip(odd, size="101x61", pix_fmt="yuv444p")
        before = snapshot(self.dir)
        proc = apply(odd, EFFECT, "-o", self.path("odd_out.mp4"))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("odd", proc.stderr)
        self.assertIn("101x61", proc.stderr)
        self.assertEqual(snapshot(self.dir), before)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
