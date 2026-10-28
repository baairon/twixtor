#!/usr/bin/env python3
"""tests for --size and --fps of tools/apply.py. stdlib unittest, no network.

    python tools/test_apply_geometry.py

the facts tests need nothing; the runs need ffmpeg, ffprobe and a built vfx (cargo build --release
-p fx-cli, or VFX_PATH) and skip without them. the runs use frame index clips: 16-bit gray frames whose
value is 512 times the frame number, stored lossless, so that the value of an output frame read back
says which source frame (or which position between two, under Blend) it shows.
"""

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
APPLY = os.path.join(HERE, "apply.py")

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, HERE)
import apply as tool  # noqa: E402

TMP = None
VFX = None
TOOLS = False
TEMPORAL = {"kind": "temporal", "layers": [], "params": ["speed_percent", "in_fps_is_out_fps", "input_frame_rate"]}
PLAIN_TEMPORAL = {"kind": "temporal", "layers": [], "params": ["offset"]}
FRAME = {"kind": "frame", "layers": [], "params": ["invert_alpha"]}


def facts(**over):
    base = {"has_video": True, "has_audio": True, "width": 160, "height": 96, "rotation": 0, "fps": 24.0,
            "rate": Fraction(24), "vfr": False, "duration": 1.0, "acodec": "aac", "channels": 2,
            "nb_frames": 24, "video_duration": None, "audio_duration": 1.0, "video_index": 0,
            "color_space": None, "color_primaries": None, "color_transfer": None}
    base.update(over)
    return base


def args(**over):
    base = dict(input="in.mp4", effect="fx", depth=16, param=[], frames=None, layer=[], size=None, fps=None,
                pre_stage=[], post_stage=[])
    base.update(over)
    return argparse.Namespace(**base)


def ff(*argv):
    subprocess.run([tool.ffmpeg_path(), "-v", "error", "-y", *argv], shell=False, check=True, capture_output=True)


def index_clip(path, rate, frames, size="64x48"):
    """a lossless clip whose frame n is gray 512 * n at 16 bits (png in mov, so the container counts its frames)."""
    ff("-f", "lavfi", "-i", f"color=c=black:s={size}:r={rate}:d={frames},format=gray16le,geq=lum='N*512'",
       "-frames:v", str(frames), "-c:v", "png", path)


def setUpModule():
    global TMP, VFX, TOOLS
    TMP = tempfile.mkdtemp(prefix="apply_geometry_test_")
    try:
        tool.require_ffmpeg()
        VFX = tool.find_vfx(None)
    except SystemExit:
        return
    TOOLS = True
    fix = lambda name: os.path.join(TMP, name)
    ff("-f", "lavfi", "-i", "testsrc=size=160x96:rate=24:duration=1", "-f", "lavfi", "-i",
       "sine=frequency=440:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
       fix("clip.mp4"))
    ff("-f", "lavfi", "-i", "testsrc=size=80x60:rate=30:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
       fix("layer.mp4"))
    ff("-f", "lavfi", "-i", "testsrc=size=160x96:rate=30000/1001:duration=0.5", "-c:v", "libx264", "-pix_fmt",
       "yuv420p", fix("ntsc.mp4"))
    ff("-f", "lavfi", "-i", "testsrc=size=160x96:rate=30:duration=2", "-vf", "select='lt(mod(n,5),3)'", "-fps_mode",
       "passthrough", "-c:v", "libx264", "-pix_fmt", "yuv420p", fix("vfr.mp4"))
    index_clip(fix("idx24.mov"), 24, 25)
    index_clip(fix("idx60.mov"), 60, 60)


def tearDownModule():
    if TMP:
        shutil.rmtree(TMP, ignore_errors=True)


def fixture(name):
    return os.path.join(TMP, name)


def apply(*argv, tmp=None):
    env = dict(os.environ, VFX_PATH=VFX or "", PYTHONDONTWRITEBYTECODE="1")
    if tmp:
        env.update(TMP=tmp, TEMP=tmp, TMPDIR=tmp)
    return subprocess.run([sys.executable, APPLY] + list(argv), shell=False, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env)


def no_pid(text):
    return re.sub(r"\.partial-[0-9]+\.", ".partial-PID.", text)


def red16(path):
    """the red channel of the first pixel of every frame of a file, decoded to rgba64le."""
    raw = subprocess.run([tool.ffmpeg_path(), "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgba64le", "-"],
                         shell=False, check=True, capture_output=True).stdout
    info = tool.probe(path)
    size = info["width"] * info["height"] * 8
    return [struct.unpack_from("<H", raw, i * size)[0] for i in range(len(raw) // size)]


def words_after(cmd, flag):
    return [cmd[i + 1] for i, word in enumerate(cmd) if word == flag]


# ---- facts alone ----------------------------------------------------------------------------------


class GeometryFactTests(unittest.TestCase):
    """work_geometry, the filters, the vfx flags, the stages and the report, on facts alone."""

    def test_values_equal_to_the_inputs_change_nothing(self):
        plain = tool.work_geometry(args(), facts(), TEMPORAL)
        self.assertEqual(plain, {"width": 160, "height": 96, "rate": Fraction(24), "frames": 24, "source": None,
                                 "auto_params": [], "rounding": None, "clip_rate": Fraction(24), "out_frames": 24})
        for over in (dict(size=(160, 96)), dict(fps=Fraction(24)), dict(size=(160, 96), fps=Fraction(48, 2))):
            self.assertEqual(tool.work_geometry(args(**over), facts(), TEMPORAL), plain, over)
        # a rotated input is compared by its displayed size, a variable rate one by its average rate
        self.assertIsNone(tool.work_geometry(args(size=(96, 160)), facts(rotation=90), TEMPORAL)["source"])
        vfr = facts(vfr=True, rate=Fraction(7200, 137))
        self.assertIsNone(tool.work_geometry(args(fps=Fraction(7200, 137)), vfr, TEMPORAL)["source"])

    def test_a_size_and_a_rate_set_the_output_and_keep_the_inputs_own(self):
        geo = tool.work_geometry(args(size=(320, 192), fps=Fraction(60)), facts(nb_frames=25), TEMPORAL)
        # rate is the output's; frames and clip_rate are the clip vfx reads (the input at its own rate), and
        # out_frames the output frames that start inside it
        self.assertEqual(geo, {"width": 320, "height": 192, "rate": Fraction(60), "frames": 25,
                               "source": {"width": 160, "height": 96, "fps": Fraction(24), "frames": 25},
                               "auto_params": ["in_fps_is_out_fps=0", "input_frame_rate=24"], "rounding": "up",
                               "clip_rate": Fraction(24), "out_frames": 63})
        self.assertEqual(tool.output_length(geo), 63)
        self.assertEqual(tool.output_length({"frames": 7}), 7, "a geometry from before out_frames")
        # the size alone: no rounding, no params, the rate the input's
        geo = tool.work_geometry(args(size=(320, 192)), facts(), TEMPORAL)
        self.assertEqual((geo["rate"], geo["frames"], geo["out_frames"], geo["rounding"], geo["auto_params"]),
                         (Fraction(24), 24, 24, None, []))
        self.assertEqual(geo["clip_rate"], Fraction(24))
        # a frame effect gets the input converted in the decode: its clip is at the output rate
        geo = tool.work_geometry(args(fps=Fraction(60)), facts(nb_frames=25), FRAME)
        self.assertEqual((geo["frames"], geo["out_frames"], geo["clip_rate"], geo["auto_params"]),
                         (63, 63, Fraction(60), []))
        # no effect facts: read like a frame effect
        self.assertEqual(tool.work_geometry(args(fps=Fraction(60)), facts())["clip_rate"], Fraction(60))
        # an unknown count stays unknown
        for info in (facts(nb_frames=None), facts(vfr=True)):
            geo = tool.work_geometry(args(fps=Fraction(60)), info, TEMPORAL)
            self.assertEqual((geo["frames"], geo["out_frames"]), (None, None))

    def test_the_rate_params_are_added_only_when_the_user_gave_neither(self):
        ntsc = facts(rate=Fraction(30000, 1001), fps=29.97)
        geo = tool.work_geometry(args(fps=Fraction(60)), ntsc, TEMPORAL)
        self.assertEqual(geo["auto_params"], ["in_fps_is_out_fps=0", "input_frame_rate=29.97002997"])
        for rate, text in ((Fraction(25), "25"), (Fraction(2997, 100), "29.97"), (Fraction(24000, 1001), "23.97602398"),
                           (Fraction(7200, 137), "52.55474453")):
            geo = tool.work_geometry(args(fps=Fraction(60)), facts(rate=rate), TEMPORAL)
            self.assertEqual(geo["auto_params"][1], "input_frame_rate=" + text)
        for given in (["in_fps_is_out_fps=1"], ["input_frame_rate=30"], ["speed_percent=40", "in_fps_is_out_fps=0"]):
            geo = tool.work_geometry(args(fps=Fraction(60), param=given), facts(), TEMPORAL)
            self.assertEqual(geo["auto_params"], [], given)
            self.assertEqual(tool.effect_params(args(param=given), geo), given)
        # an effect without both ids, a frame effect and a size alone add nothing
        self.assertEqual(tool.work_geometry(args(fps=Fraction(60)), facts(), PLAIN_TEMPORAL)["auto_params"], [])
        partial = dict(TEMPORAL, params=["input_frame_rate"])
        self.assertEqual(tool.work_geometry(args(fps=Fraction(60)), facts(), partial)["auto_params"], [])
        self.assertEqual(tool.work_geometry(args(fps=Fraction(60)), facts(), FRAME)["auto_params"], [])
        self.assertEqual(tool.work_geometry(args(size=(64, 48)), facts(), TEMPORAL)["auto_params"], [])

    def test_the_default_length_is_the_output_frames_that_start_inside_the_input(self):
        out = tool.output_frames
        self.assertEqual(out(25, Fraction(24), Fraction(60)), 63)
        self.assertEqual(out(24, Fraction(24), Fraction(60)), 60)
        self.assertEqual(out(60, Fraction(60), Fraction(24)), 24)
        self.assertEqual(out(61, Fraction(60), Fraction(24)), 25)
        self.assertEqual(out(50, Fraction(25), Fraction(24)), 48)
        self.assertEqual(out(1001, Fraction(30000, 1001), Fraction(24)), 802)
        for n in (0, 1, 37, 10 ** 12):
            self.assertEqual(out(n, Fraction(30000, 1001), Fraction(30000, 1001)), n)

    def test_the_decode_keeps_the_inputs_frames_for_a_temporal_effect_and_converts_for_a_frame_effect(self):
        info = facts()
        temporal = tool.work_geometry(args(size=(320, 192), fps=Fraction(60)), info, TEMPORAL)
        self.assertEqual(tool.decode_filters(args(), info, temporal), ["scale=320:192:in_color_matrix=smpte170m"])
        frame = tool.work_geometry(args(fps=Fraction(60)), info, FRAME)
        self.assertEqual(tool.decode_filters(args(), info, frame),
                         ["settb=1/24", "setpts=N", "fps=60/1:round=up", "scale=in_color_matrix=smpte170m"])
        ntsc = facts(rate=Fraction(30000, 1001))
        self.assertEqual(tool.decode_filters(args(), ntsc, tool.work_geometry(args(fps=Fraction(24)), ntsc, FRAME))[:3],
                         ["settb=1001/30000", "setpts=N", "fps=24/1:round=up"])
        # a variable rate input is made constant at its average rate first, as without the flags
        vfr = facts(vfr=True, rate=Fraction(7200, 137))
        self.assertEqual(tool.decode_filters(args(), vfr, tool.work_geometry(args(fps=Fraction(24)), vfr, TEMPORAL)),
                         ["fps=7200/137", "scale=in_color_matrix=smpte170m"])
        self.assertEqual(tool.decode_filters(args(), vfr, tool.work_geometry(args(fps=Fraction(24)), vfr, FRAME))[:2],
                         ["fps=7200/137", "settb=137/7200"])
        # a size alone scales, nothing else
        size = tool.work_geometry(args(size=(64, 48)), info, FRAME)
        self.assertEqual(tool.decode_filters(args(), info, size), ["scale=64:48:in_color_matrix=smpte170m"])

    def test_layers_keep_their_own_rate_and_take_the_output_size(self):
        info = facts()
        geo = tool.work_geometry(args(size=(320, 192), fps=Fraction(60)), info, TEMPORAL)
        layer = facts(width=80, height=60, rate=Fraction(30), color_space="bt709")
        self.assertEqual(tool.layer_filters(args(), layer, geo), ["scale=320:192:in_color_matrix=bt709"])
        vfr = facts(vfr=True, rate=Fraction(7200, 137))
        self.assertEqual(tool.layer_filters(args(), vfr, geo), ["fps=7200/137", "scale=320:192:in_color_matrix=smpte170m"])
        self.assertEqual(tool.layer_filters(args(), layer, geo, (80, 48)), ["scale=80:48:in_color_matrix=bt709"])
        # without the flags a layer of another rate is resampled to the work rate, as before
        plain = tool.work_geometry(args(), info, TEMPORAL)
        self.assertEqual(tool.layer_filters(args(), layer, plain), ["fps=24/1", "scale=160:96:in_color_matrix=bt709"])

    def test_vfx_reads_the_input_and_the_layers_at_their_own_rates(self):
        layers = [{"id": "alt_source_a", "info": facts(rate=Fraction(30))}, {"id": "alt_source_b", "info": facts()},
                  {"id": "fg1_matte", "info": facts(rate=Fraction(60))}]
        geo = tool.work_geometry(args(fps=Fraction(60)), facts(), TEMPORAL)
        self.assertEqual(tool.vfx_rate_flags(args(), geo, layers),
                         ["--clip-fps", "24/1", "--layer-fps", "alt_source_a=30/1", "--layer-fps", "alt_source_b=24/1"])
        size = tool.work_geometry(args(size=(64, 48)), facts(), TEMPORAL)
        self.assertEqual(tool.vfx_rate_flags(args(), size, layers),
                         ["--layer-fps", "alt_source_a=30/1", "--layer-fps", "fg1_matte=60/1"])
        self.assertEqual(tool.vfx_rate_flags(args(), tool.work_geometry(args(), facts(), TEMPORAL), layers), [])
        plan = tool.make_plan(args(), facts(), TEMPORAL, [dict(layer, video="v.mp4") for layer in layers], "/tmp/s", geo)
        self.assertEqual((plan["frames"], plan["input_frames"]), (60, 60), "the output's length")
        a = args(param=["speed_percent=40"], layer=[(layer["id"], "v.mp4") for layer in layers])
        effect = tool.build_commands(a, facts(), "vfx", "out.mp4", plan, None, geo)[1]
        self.assertEqual(effect[:10], ["vfx", "apply", "fx", "--width", "160", "--height", "96", "--depth", "16", "--fps"])
        self.assertEqual(words_after(effect, "--fps"), ["60/1"])
        self.assertEqual(words_after(effect, "--frames"), ["60"])
        self.assertEqual(words_after(effect, "--clip-fps"), ["24/1"])
        self.assertEqual(words_after(effect, "--layer-fps"), ["alt_source_a=30/1", "alt_source_b=24/1"])
        self.assertEqual(words_after(effect, "--param"), ["speed_percent=40", "in_fps_is_out_fps=0", "input_frame_rate=24"])
        self.assertLess(effect.index("--layer"), effect.index("--clip-fps"))
        self.assertLess(effect.index("--layer-fps"), effect.index("--param"))
        encode = tool.build_commands(a, facts(), "vfx", "out.mp4", plan, None, geo)[2]
        self.assertEqual(words_after(encode, "-framerate"), ["60/1"])
        # the preflight asks vfx about the same rate flags, from a probe of each layer video
        rates = {"v.mp4": facts(rate=Fraction(30))}
        seen = []
        with mock.patch.object(tool, "probe", lambda path: rates[path]),                 mock.patch.object(tool, "run", lambda cmd, **k: seen.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", "")):
            tool.preflight(a, facts(), "vfx", TEMPORAL, geo)
        self.assertEqual(words_after(seen[0], "--clip-fps"), ["24/1"])
        self.assertEqual(words_after(seen[0], "--layer-fps"), ["alt_source_a=30/1", "alt_source_b=30/1", "fg1_matte=30/1"])
        self.assertEqual(words_after(seen[0], "--param"), words_after(effect, "--param"))
        seen.clear()
        with mock.patch.object(tool, "probe", lambda path: self.fail("no probe without a geometry of its own")),                 mock.patch.object(tool, "run", lambda cmd, **k: seen.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", "")):
            tool.preflight(a, facts(), "vfx", TEMPORAL, tool.work_geometry(a, facts(), TEMPORAL))
        self.assertNotIn("--layer-fps", seen[0])

    def test_pre_stages_get_what_the_decode_writes_and_post_stages_the_output(self):
        line = "{width} {height} {fps} {frames}"
        stage = [sys.executable + " -c pass " + line]
        if os.name == "nt":
            stage = ['"%s" -c pass %s' % (sys.executable, line)]
        a = args(size=(320, 192), fps=Fraction(60), pre_stage=stage, post_stage=stage)
        info = facts(nb_frames=25)
        for einfo, pre in ((TEMPORAL, ["320", "192", "24/1", "25"]), (FRAME, ["320", "192", "60/1", "63"])):
            geo = tool.work_geometry(a, info, einfo)
            chain = {}
            stages = tool.build_stages(a, info, chain, geo)
            self.assertEqual([s["argv"][-4:] for s in stages], [pre, ["320", "192", "60/1", "63"]], einfo["kind"])
            self.assertEqual(tool.geometry_json(chain["out"]),
                             {"width": 320, "height": 192, "depth": 16, "fps": "60/1", "frames": 63})
            self.assertEqual((chain["vfx"]["width"], chain["vfx"]["fps"]), (320, Fraction(pre[2])))
        # --frames counts output frames
        a.frames = 10
        stages = tool.build_stages(a, info, None, tool.work_geometry(a, info, TEMPORAL))
        self.assertEqual(stages[1]["argv"][-4:], ["320", "192", "60/1", "10"])

    def test_the_report_names_the_source_and_the_added_params(self):
        info = facts(nb_frames=25)
        self.assertEqual(tool.report_geometry(args(), info, tool.work_geometry(args(), info, TEMPORAL)), ([], {}))
        geo = tool.work_geometry(args(size=(320, 192), fps=Fraction(60)), info, TEMPORAL)
        phrases, keys = tool.report_geometry(args(), info, geo)
        self.assertEqual(phrases, ["scaled from 160x96", "converted from 24.000 fps",
                                   "in_fps_is_out_fps=0 and input_frame_rate=24 added"])
        self.assertEqual(keys, {"source": {"size": "160x96", "fps": 24.0, "fps_exact": "24/1", "frames": 25,
                                           "rounding": "up"},
                                "auto_params": ["in_fps_is_out_fps=0", "input_frame_rate=24"]})
        phrases, keys = tool.report_geometry(args(), info, tool.work_geometry(args(size=(64, 48)), info, FRAME))
        self.assertEqual(phrases, ["scaled from 160x96"])
        self.assertEqual(keys["source"]["rounding"], None)
        self.assertEqual(keys["auto_params"], [])
        ntsc = facts(rate=Fraction(30000, 1001))
        phrases, keys = tool.report_geometry(args(), ntsc, tool.work_geometry(args(fps=Fraction(60)), ntsc, TEMPORAL))
        self.assertEqual(phrases, ["converted from 29.970 fps", "in_fps_is_out_fps=0 and input_frame_rate=29.97002997 added"])
        self.assertEqual((keys["source"]["fps"], keys["source"]["fps_exact"]), (30000 / 1001, "30000/1001"))

    def test_verify_names_the_requested_size_and_rate(self):
        src, out = facts(), facts(width=160, height=96, fps=24.0)
        problems = tool.verify(src, out, Fraction(60), size=(320, 192))
        self.assertIn("size is 160x96, expected 320x192", problems)
        self.assertIn("frame rate is 24.0, expected 60.000", problems)
        self.assertEqual(tool.verify(src, facts(width=320, height=192, fps=60.0), Fraction(60),
                                     size=(320, 192)), [])


# ---- runs -----------------------------------------------------------------------------------------


class GeometryRunTests(unittest.TestCase):
    def setUp(self):
        if not TOOLS:
            self.skipTest("needs ffmpeg, ffprobe and a built vfx (cargo build --release -p fx-cli, or VFX_PATH)")
        self.dir = tempfile.mkdtemp(dir=TMP)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def run_ok(self, *argv):
        proc = apply(*argv, tmp=self.dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc

    def dry(self, *argv):
        return json.loads(self.run_ok(*argv, "--dry-run", "--json").stdout)

    def out(self, name):
        return os.path.join(self.dir, name)

    def test_values_equal_to_the_inputs_give_the_same_plan(self):
        clip = fixture("clip.mp4")
        runs = ((("selftest-shift", ["--layer", "overlay=" + fixture("layer.mp4")]), (["--json"], [])),
                (("retime", ["--param", "speed_percent=40"]), (["--json", "--contract", "frames/2"],)),
                (("selftest-invert", []), (["--json"],)))
        for (effect, extra), modes in runs:
            for mode in modes:
                base = self.run_ok(clip, effect, *extra, "--dry-run", *mode).stdout
                for flags in (["--size", "160x96"], ["--fps", "48/2"], ["--size", "160x96", "--fps", "24.000"]):
                    same = self.run_ok(clip, effect, *extra, *flags, "--dry-run", *mode).stdout
                    self.assertEqual(no_pid(same), no_pid(base), (effect, flags, mode))

    def test_a_size_scales_the_input_and_the_layers_dry_and_real(self):
        clip, layer = fixture("clip.mp4"), fixture("layer.mp4")
        dst = self.out("big.mp4")
        argv = [clip, "selftest-shift", "--size", "320x192", "--layer", "overlay=" + layer, "-o", dst]
        plan = self.dry(*argv)
        self.assertEqual((plan["size"], plan["fps_exact"], plan["frames"], plan["input_frames"]), ("320x192", "24/1", 24, 24))
        self.assertEqual(plan["source"], {"size": "160x96", "fps": 24.0, "fps_exact": "24/1", "frames": 24, "rounding": None})
        self.assertEqual(plan["auto_params"], [])
        self.assertIn(", layers overlay, scaled from 160x96, ", plan["summary"])
        layer_cmd, decode, effect = plan["commands"][0], plan["commands"][1], plan["commands"][2]
        self.assertIn("-vf scale=320:192:in_color_matrix=smpte170m -pix_fmt", layer_cmd, "the layer keeps its 30 fps")
        self.assertIn("-vf scale=320:192:in_color_matrix=smpte170m -pix_fmt", decode)
        self.assertIn("--width 320 --height 192", effect)
        self.assertIn("--layer-fps overlay=30/1", effect)
        self.assertNotIn("--clip-fps", effect)
        proc = self.run_ok(*argv[:-2], "--json", "-o", dst)
        result = json.loads(proc.stdout)
        self.assertEqual(result["source"]["size"], "160x96")
        out = tool.probe(dst)
        self.assertEqual((out["width"], out["height"], out["fps"], out["has_audio"]), (320, 192, 24.0, True))
        self.assertEqual(tool.display_size(tool.probe(clip)), (160, 96), "the input is as it was")

    def test_24_fps_footage_in_a_60_fps_output_plays_at_positions_0_4_n_with_63_frames(self):
        dst = self.out("slow.mkv")
        argv = [fixture("idx24.mov"), "retime", "--fps", "60", "--lossless", "--depth", "16", "--param", "vectors=1",
                "--param", "frame_interp=2", "-o", dst]
        plan = self.dry(*argv)
        self.assertEqual((plan["fps_exact"], plan["frames"], plan["input_frames"]), ("60/1", 63, 63))
        self.assertEqual(plan["source"], {"size": "64x48", "fps": 24.0, "fps_exact": "24/1", "frames": 25, "rounding": "up"})
        self.assertEqual(plan["auto_params"], ["in_fps_is_out_fps=0", "input_frame_rate=24"])
        self.assertIn("converted from 24.000 fps, in_fps_is_out_fps=0 and input_frame_rate=24 added", plan["summary"])
        effect = plan["commands"][1]
        self.assertIn("--fps 60/1 --frames 63 ", effect)
        self.assertIn("--clip-fps 24/1 --param vectors=1 --param frame_interp=2 --param in_fps_is_out_fps=0 "
                      "--param input_frame_rate=24", effect)
        self.assertNotIn("fps=", plan["commands"][0], "no frame is resampled on the way in")
        self.run_ok(*argv)
        values = red16(dst)
        self.assertEqual(len(values), 63)
        for n in range(61):
            # Blend between source frames trunc(p) and trunc(p) + 1 gives 512 p, to within one step of the
            # 15-bit range the blend runs at (2 in 16 bits); positions 0.4 n repeat the same blend every 5 frames
            self.assertLess(abs(values[n] - 204.8 * n), 2, f"output frame {n}: {values[n]}")
            if n + 5 <= 60:
                self.assertEqual(values[n + 5] - values[n], 1024, f"output frames {n} and {n + 5}")
            if n % 5 == 0:
                self.assertEqual(values[n], 512 * (2 * n // 5), f"output frame {n} is a copy of a footage frame")

    def test_60_fps_footage_in_a_24_fps_output_shows_its_real_frames(self):
        dst = self.out("real.mkv")
        argv = [fixture("idx60.mov"), "retime", "--fps", "24", "--lossless", "--depth", "16", "--param", "vectors=1",
                "--param", "frame_interp=2", "--param", "speed_percent=40", "-o", dst]
        self.run_ok(*argv)
        self.assertEqual(red16(dst), [512 * n for n in range(24)], "output frame n is footage frame n")
        # at 100% the footage keeps its pace: every second output frame is footage frame 2.5 n
        dst = self.out("pace.mkv")
        self.run_ok(fixture("idx60.mov"), "retime", "--fps", "24", "--lossless", "--depth", "16", "--param", "vectors=1",
                    "--param", "frame_interp=1", "-o", dst)
        values = red16(dst)
        self.assertEqual(len(values), 24)
        self.assertEqual(values[::2], [512 * (5 * n) for n in range(12)])

    def test_a_frame_effect_gets_the_input_frame_that_covers_each_output_frame(self):
        dst = self.out("frame.mkv")
        plan = self.dry(fixture("idx24.mov"), "selftest-invert", "--fps", "60", "--lossless", "--depth", "16", "-o", dst)
        self.assertEqual((plan["input_frames"], plan["auto_params"]), (63, []))
        self.assertIn("-vf settb=1/24,setpts=N,fps=60/1:round=up,scale=in_color_matrix=", plan["commands"][0])
        self.run_ok(fixture("idx24.mov"), "selftest-invert", "--fps", "60", "--lossless", "--depth", "16", "-o", dst)
        self.assertEqual(red16(dst), [65535 - 512 * (n * 24 // 60) for n in range(63)])

    def test_layers_are_read_at_their_own_rates_in_a_run(self):
        dst = self.out("layers.mkv")
        argv = [fixture("idx24.mov"), "selftest-shift", "--fps", "48", "--layer", "overlay=" + fixture("idx60.mov"),
                "--lossless", "--depth", "16", "-o", dst]
        plan = self.dry(*argv)
        self.assertIn("--clip-fps 24/1 --layer-fps overlay=60/1", plan["commands"][2])
        self.assertNotIn("fps=", plan["commands"][0])
        self.run_ok(*argv)
        # output frame n at n/48 s mixes input frame n / 2 with overlay frame 1.25 n (none past its 60 frames)
        want = []
        for n in range(50):
            over = 512 * (n * 60 // 48) if n * 60 // 48 < 60 else 0
            want.append((512 * (n // 2) + over + 1) >> 1)
        self.assertEqual(red16(dst), want)

    def test_ntsc_footage_writes_its_true_rate_and_the_users_values_are_kept(self):
        ntsc = fixture("ntsc.mp4")
        plan = self.dry(ntsc, "retime", "--fps", "60")
        self.assertEqual(plan["auto_params"], ["in_fps_is_out_fps=0", "input_frame_rate=29.97002997"])
        self.assertEqual(plan["source"]["fps_exact"], "30000/1001")
        self.assertEqual(plan["frames"], tool.output_frames(plan["source"]["frames"], Fraction(30000, 1001), Fraction(60)))
        self.assertIn("--clip-fps 30000/1001 --param in_fps_is_out_fps=0 --param input_frame_rate=29.97002997",
                      plan["commands"][1])
        for given in (["in_fps_is_out_fps=1"], ["input_frame_rate=30"], ["in_fps_is_out_fps=0", "input_frame_rate=29.97"]):
            extra = [w for p in given for w in ("--param", p)]
            plan = self.dry(ntsc, "retime", "--fps", "60", *extra)
            self.assertEqual((plan["auto_params"], plan["params"]), ([], given))
            self.assertEqual(words_after(plan["commands"][1].split(), "--param"), given)
            self.assertNotIn(" added", plan["summary"])
        # an effect without the two params reads by time alone
        plan = self.dry(ntsc, "create-vectors", "--fps", "60")
        self.assertEqual(plan["auto_params"], [])
        self.assertIn("--clip-fps 30000/1001", plan["commands"][1])

    def test_a_variable_rate_input_gives_null_frames(self):
        plan = self.dry(fixture("vfr.mp4"), "retime", "--fps", "24")
        info = tool.probe(fixture("vfr.mp4"))
        self.assertTrue(info["vfr"])
        self.assertEqual((plan["frames"], plan["input_frames"], plan["source"]["frames"]), (None, None, None))
        self.assertIn("-vf fps=" + tool.rate_text(info["rate"]) + ",scale=", plan["commands"][0])
        self.assertIn("--clip-fps " + tool.rate_text(info["rate"]), plan["commands"][1])
        self.assertNotIn("--frames", plan["commands"][1])
        self.assertIn("temporal, as many frames as the input decodes to", plan["summary"])

    def test_frames_counts_output_frames(self):
        dst = self.out("ten.mkv")
        argv = [fixture("idx24.mov"), "selftest-shift", "--fps", "60", "--frames", "10", "--lossless", "--depth", "16",
                "-o", dst]
        self.assertIn("--fps 60/1 --frames 10 ", self.dry(*argv)["commands"][1])
        self.run_ok(*argv)
        self.assertEqual(red16(dst), [512 * (n * 24 // 60) for n in range(10)])
        self.assertAlmostEqual(tool.probe(dst)["duration"], 10 / 60, delta=0.02)

    def test_stage_placeholders_and_the_frames2_describe_answer(self):
        clip = fixture("clip.mp4")
        line = '"%s" -c pass {width} {height} {fps} {frames}' % sys.executable
        plan = self.dry(clip, "selftest-shift", "--size", "320x192", "--fps", "60", "--pre-stage", line,
                        "--post-stage", line, "--depth", "8")
        pre, post = plan["stages"][0]["command"], plan["stages"][1]["command"]
        self.assertTrue(pre.endswith(" 320 192 24/1 24"), pre)
        self.assertTrue(post.endswith(" 320 192 60/1 60"), post)
        plan = self.dry(clip, "selftest-invert", "--fps", "60", "--pre-stage", line, "--depth", "8")
        self.assertTrue(plan["stages"][0]["command"].endswith(" 160 96 60/1 60"), plan["stages"][0]["command"])
        f2 = ["--contract", "frames/2", "--lossless", "--depth", "8"]
        for effect in ("selftest-shift", "selftest-invert", "retime"):
            proc = self.run_ok(clip, effect, "--size", "320x192", "--fps", "60", *f2, "--describe")
            self.assertEqual(json.loads(proc.stdout), {"ok": True, "contract": "frames/2", "stage": effect, "width": 320,
                                                       "height": 192, "depth": 8, "fps": "60/1", "frames": 60})
            plan = self.dry(clip, effect, "--size", "320x192", "--fps", "60", *f2)
            self.assertEqual(plan["geometry"], {"width": 320, "height": 192, "depth": 8, "fps": "60/1", "frames": 60})
            self.assertEqual((plan["size"], plan["fps_exact"], plan["input_frames"]), ("320x192", "60/1", 60))

    def test_the_flags_match_only_in_full(self):
        for flags in (["--fp", "24"], ["--si", "64x48"], ["--au", "copy"]):
            proc = apply(fixture("clip.mp4"), "retime", *flags, "--dry-run", tmp=self.dir)
            self.assertEqual(proc.returncode, 2, flags)
            self.assertIn("unrecognized arguments: " + " ".join(flags), proc.stderr)
        self.assertLessEqual({"--size", "--fps"}, tool.ArgParser.FULL_ONLY)

    def test_verify_compares_the_output_with_the_requested_geometry(self):
        dst = self.out("v.mkv")
        self.run_ok(fixture("idx24.mov"), "selftest-shift", "--size", "32x24", "--fps", "60", "--lossless", "--depth", "8",
                    "-o", dst)
        src, out = tool.probe(fixture("idx24.mov")), tool.probe(dst)
        self.assertEqual((out["width"], out["height"], out["fps"]), (32, 24, 60.0))
        self.assertEqual(tool.verify(src, out, Fraction(60), size=(32, 24)), [])
        problems = tool.verify(src, out, Fraction(24), size=(64, 48))
        self.assertIn("size is 32x24, expected 64x48", problems)
        self.assertIn("frame rate is 60.0, expected 24.000", problems)


if __name__ == "__main__":
    unittest.main()
