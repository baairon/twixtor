#!/usr/bin/env python3
"""tests for the HDR and odd size encodes of tools/apply.py. stdlib unittest, no network.

    python tools/test_apply_hdr.py

an HDR input (color_transfer smpte2084 or arib-std-b67) goes through as stored, with no tone mapping, and
is encoded libx265 CRF 18 in 10 bit with its tags and its HDR metadata (ffv1 RGB tagged with its primaries
and its transfer under --lossless); a size with an odd side encodes yuv444p (yuv444p10le for HDR) at that
exact size. the unit tests need nothing but python. the end to end tests need ffmpeg, ffprobe and a built
vfx (cargo build --release -p fx-cli, or VFX_PATH) and skip themselves with a note when one is missing;
the HDR ones also skip when that ffmpeg has no libx265 with 10 bit, which makes their fixtures.
"""

import argparse
import contextlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
APPLY = os.path.join(HERE, "apply.py")
FFSTAGE = os.path.join(HERE, "ffstage.py")

# no bytecode next to the tools, here or in the script the suite runs
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, HERE)
import apply as tool  # noqa: E402

FPS = 24
FRAMES = 12
TMP = None
VFX = None
X265 = False

# the HDR metadata of the PQ fixture as x265 takes it, and as ffprobe reads it from a frame and a container
MASTER = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"
CLL = "1000,400"
MASTER_SIDE = {"side_data_type": "Mastering display metadata", "red_x": "34000/50000", "red_y": "16000/50000",
               "green_x": "13250/50000", "green_y": "34500/50000", "blue_x": "7500/50000", "blue_y": "3000/50000",
               "white_point_x": "15635/50000", "white_point_y": "16450/50000", "min_luminance": "1/10000",
               "max_luminance": "10000000/10000"}
MKV_SIDE = dict(MASTER_SIDE, red_x="17/25", red_y="8/25", green_x="53/200", green_y="69/100", blue_x="3/20",
                blue_y="3/50", white_point_x="3127/10000", white_point_y="329/1000", max_luminance="1000/1")
CLL_SIDE = {"side_data_type": "Content light level metadata", "max_content": 1000, "max_average": 400}
# other values: what the bitstream of a file says when its container keeps MASTER and CLL
OTHER_MASTER = "G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(40000000,50)"
OTHER_CLL = "500,200"
OTHER_MASTER_SIDE = dict(MASTER_SIDE, red_x="35400/50000", red_y="14600/50000", green_x="8500/50000",
                         green_y="39850/50000", blue_x="6550/50000", blue_y="2300/50000", min_luminance="50/10000",
                         max_luminance="40000000/10000")
OTHER_CLL_SIDE = dict(CLL_SIDE, max_content=500, max_average=200)
PQ_TAGS = "colorspace=bt2020nc:color_primaries=bt2020:color_trc=smpte2084:range=tv"
HLG_TAGS = "colorspace=bt2020nc:color_primaries=bt2020:color_trc=arib-std-b67:range=tv"
COLOR_KEYS = ("color_space", "color_primaries", "color_transfer", "color_range")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def facts(**over):
    """the facts probe() gives for a 64x48 clip without audio, SDR and untagged unless told otherwise."""
    base = {"path": os.path.abspath("in.mp4"), "has_video": True, "has_audio": False, "vcodec": "hevc",
            "acodec": None, "pix_fmt": "yuv420p10le", "video_index": 0, "width": 64, "height": 48, "rotation": 0,
            "fps": 24.0, "rate": Fraction(24), "vfr": False, "duration": 0.5, "channels": None, "hdr": False,
            "nb_frames": FRAMES, "video_duration": None, "audio_duration": None, "color_space": None,
            "color_primaries": None, "color_transfer": None}
    base.update(over)
    return base


PQ = facts(hdr=True, color_space="bt2020nc", color_primaries="bt2020", color_transfer="smpte2084")
HLG = facts(hdr=True, color_space="bt2020nc", color_primaries="bt2020", color_transfer="arib-std-b67")


def args(**over):
    base = dict(input="in.mp4", effect="fx", depth=16, param=[], frames=None, layer=[], lossless=False)
    base.update(over)
    return argparse.Namespace(**base)


def ff(*argv):
    subprocess.run([tool.ffmpeg_path(), "-v", "error", "-y", *argv], shell=False, check=True, capture_output=True)


def x265_has_10_bit():
    """whether this ffmpeg's libx265 writes 10 bit 4:2:0 and 4:4:4 (it makes the HDR fixtures)."""
    proc = subprocess.run([tool.ffmpeg_path(), "-hide_banner", "-h", "encoder=libx265"], shell=False,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout.splitlines():
        label, _, rest = line.strip().partition(":")
        if label == "Supported pixel formats":
            return {"yuv420p10le", "yuv444p10le"} <= set(rest.split())
    return False


def source(size, audio, seconds=FRAMES / FPS):
    # testsrc keeps an odd size, testsrc2 rounds it down to even sides
    inputs = ["-f", "lavfi", "-i", f"testsrc=size={size}:rate={FPS}:duration={seconds}"]
    if audio:
        inputs += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    return inputs, (["-c:a", "aac", "-shortest"] if audio else [])


def sdr_clip(path, size="160x96", pix_fmt="yuv420p", audio=True, extra=()):
    inputs, sound = source(size, audio)
    ff(*inputs, "-c:v", "libx264", "-pix_fmt", pix_fmt, *extra, *sound, path)


def hdr_clip(path, size="64x48", tags=PQ_TAGS, pix_fmt="yuv420p10le", params="", audio=False):
    """a testsrc clip encoded libx265 in 10 bit with the color tags `tags` and the x265 params `params`."""
    inputs, sound = source(size, audio)
    tag = ["-tag:v", "hvc1"] if path.endswith(".mp4") else []
    ff(*inputs, "-vf", f"format={pix_fmt},setparams={tags}", "-c:v", "libx265", "-crf", "18",
       "-x265-params", "log-level=error" + params, *tag, *sound, path)


MAKERS = {
    "clip.mp4": lambda p: sdr_clip(p),
    "odd.mp4": lambda p: sdr_clip(p, size="101x61", pix_fmt="yuv444p"),
    # HDR tags on an 8 bit H.264 stream: an HDR input for a test without libx265
    "pq264.mp4": lambda p: sdr_clip(p, size="64x48", audio=False, extra=(
        "-bsf:v", "h264_metadata=colour_primaries=9:transfer_characteristics=16:matrix_coefficients=9")),
    "pq.mp4": lambda p: hdr_clip(p, params=f":master-display={MASTER}:max-cll={CLL}", audio=True),
    "hlg.mp4": lambda p: hdr_clip(p, tags=HLG_TAGS),
    "pq65.mp4": lambda p: hdr_clip(p, size="65x49", pix_fmt="yuv444p10le", params=f":master-display={MASTER}"),
    # an ffmpeg transcode keeps the decoded metadata in the container, x265 writes the other values in the SEI
    "two.mkv": lambda p: ff("-i", fixture("pq.mp4"), "-map", "0:v", "-c:v", "libx265", "-x265-params",
                            f"log-level=error:master-display={OTHER_MASTER}:max-cll={OTHER_CLL}", p),
    # 10 bit streams without color tags, and one tagged SDR: HDR that apply.py cannot see
    "plain265.mp4": lambda p: ff(*source("64x48", False)[0], "-vf", "format=yuv420p10le", "-c:v", "libx265",
                                 "-x265-params", "log-level=error", "-tag:v", "hvc1", p),
    "plain265.mkv": lambda p: ff(*source("64x48", False)[0], "-vf", "format=yuv420p10le", "-c:v", "libx265",
                                 "-x265-params", "log-level=error", p),
    "plain264.mp4": lambda p: ff(*source("64x48", False)[0], "-vf", "format=yuv420p10le", "-c:v", "libx264", p),
    "sdr265.mp4": lambda p: hdr_clip(p, tags="colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv"),
}


def fixture(name):
    """a clip of the suite, made in TMP on first use."""
    path = os.path.join(TMP, name)
    if not os.path.exists(path):
        MAKERS[name](path)
    return path


def apply(*argv, tmp=None):
    """run apply.py; `tmp` points the temp directory of apply.py, ffmpeg and vfx at a directory of the test."""
    env = dict(os.environ, VFX_PATH=VFX, PYTHONDONTWRITEBYTECODE="1")
    if tmp:
        env.update(TMP=tmp, TEMP=tmp, TMPDIR=tmp)
    return subprocess.run([sys.executable, APPLY, *argv], shell=False, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env)


def ffprobe_json(*argv):
    proc = subprocess.run([tool.ffprobe_path(), "-v", "quiet", "-print_format", "json", *argv], shell=False,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    return json.loads(proc.stdout)


def stream_facts(path):
    """codec, codec tag, pixel format, size and the four color fields of the first video stream."""
    return ffprobe_json("-select_streams", "v:0", "-show_entries", "stream=codec_name,codec_tag_string,pix_fmt,"
                        "width,height,color_space,color_primaries,color_transfer,color_range", path)["streams"][0]


def frame_side_data(path):
    """the mastering display and content light level side data of the first frame, by type, without the type."""
    frames = ffprobe_json("-select_streams", "v:0", "-show_frames", "-read_intervals", "%+#1", "-show_entries",
                          "frame=side_data_list", path)["frames"]
    found = {}
    for side in frames[0].get("side_data_list") or []:
        if side.get("side_data_type") in ("Mastering display metadata", "Content light level metadata"):
            found[side["side_data_type"]] = {k: v for k, v in side.items() if k != "side_data_type"}
    return found


def raw_frames(path, depth):
    """the frames of a file as the raw bytes vfx works with at `depth`."""
    cmd = [tool.ffmpeg_path(), "-v", "error", "-i", path, "-an", "-pix_fmt", tool.PIX_FMT[depth], "-f", "rawvideo",
           "-"]
    return subprocess.run(cmd, shell=False, check=True, capture_output=True).stdout


def psnr(a, b, plane="average"):
    """the PSNR of two videos over all their frames: the average of the planes, or the one of `plane` (y)."""
    cmd = [tool.ffmpeg_path(), "-hide_banner", "-v", "info", "-i", a, "-i", b, "-lavfi", "[0:v][1:v]psnr", "-f",
           "null", "-"]
    log = subprocess.run(cmd, shell=False, capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
    match = re.search(rf"\b{plane}:([0-9.]+|inf)", log)
    assert match, log
    return float(match.group(1))


def mean_luma10(path):
    """the mean of the 10 bit luma code values of every frame of a file."""
    cmd = [tool.ffmpeg_path(), "-v", "error", "-i", path, "-vf", "extractplanes=y", "-pix_fmt", "gray10le", "-f",
           "rawvideo", "-"]
    data = subprocess.run(cmd, shell=False, check=True, capture_output=True).stdout
    values = memoryview(data).cast("H")
    return sum(values) / len(values)


def cmdline(*argv):
    """one stage command line for apply.py, which splits it with shlex: double quotes on windows."""
    if os.name == "nt":
        return " ".join('"%s"' % a for a in map(str, argv))
    return " ".join(shlex.quote(str(a)) for a in argv)


def ffstage(graph):
    return cmdline(sys.executable, FFSTAGE, "apply", "ffmpeg-filter", "--width", "{width}", "--height", "{height}",
                   "--depth", "{depth}", "--fps", "{fps}", "--param", "graph=" + graph, "--param", "expect=declared")


def options(command, flag):
    """every value after `flag` in a command line of a report, in order."""
    words = shlex.split(command)
    return [words[i + 1] for i, word in enumerate(words[:-1]) if word == flag]


def option(command, flag):
    """the one value after `flag` in a command line of a report."""
    values = options(command, flag)
    assert len(values) == 1, (flag, command)
    return values[0]


def setUpModule():
    global TMP, VFX, X265
    try:
        tool.require_ffmpeg()
        VFX = tool.find_vfx(None)
    except SystemExit:
        VFX = None
        return
    X265 = x265_has_10_bit()
    TMP = tempfile.mkdtemp(prefix="apply_hdr_test_")


def tearDownModule():
    if TMP:
        shutil.rmtree(TMP, ignore_errors=True)


# ----------------------------------------------------------------------------
# no tools needed
# ----------------------------------------------------------------------------

class ColorSettingsTests(unittest.TestCase):
    def test_an_hdr_source_keeps_its_transfer_and_its_tags_or_bt2020(self):
        self.assertEqual(tool.color_settings(PQ), ("bt2020", "bt2020nc", "bt2020", "smpte2084"))
        bare = facts(hdr=True, color_transfer="arib-std-b67")
        self.assertEqual(tool.color_settings(bare), ("bt2020", "bt2020nc", "bt2020", "arib-std-b67"))
        # HD or not, an HDR source never follows the size convention of an untagged SDR one
        self.assertEqual(tool.color_settings(dict(bare, width=1920, height=1080)),
                         ("bt2020", "bt2020nc", "bt2020", "arib-std-b67"))
        # a matrix and primaries as tagged
        self.assertEqual(tool.color_settings(dict(PQ, color_space="bt709", color_primaries="smpte432")),
                         ("bt709", "bt709", "smpte432", "smpte2084"))
        # a matrix swscale has no conversion for (constant luminance, ICtCp, RGB) is bt2020nc
        for space in ("bt2020c", "ictcp", "gbr"):
            self.assertEqual(tool.color_settings(dict(PQ, color_space=space)), ("bt2020", "bt2020nc", "bt2020", "smpte2084"))

    def test_an_sdr_source_is_read_as_before(self):
        for flag in ({}, {"hdr": False}):
            untagged = {k: v for k, v in facts(**flag).items() if flag or k != "hdr"}
            self.assertEqual(tool.color_settings(untagged), ("smpte170m",) * 4)
            self.assertEqual(tool.color_settings(dict(untagged, width=1280, height=720)), ("bt709",) * 4)
            self.assertEqual(tool.color_settings(dict(untagged, color_space="bt470bg")), ("bt601", "bt470bg", None, None))
            self.assertEqual(tool.color_settings(dict(untagged, color_space="bt2020nc", color_primaries="bt2020",
                                                      color_transfer="bt2020-10")),
                             ("bt2020", "bt2020nc", "bt2020", "bt2020-10"))


class MetadataTests(unittest.TestCase):
    def test_the_master_display_is_green_blue_red_white_point_and_luminance_in_x265_units(self):
        self.assertEqual(tool.master_display(MASTER_SIDE), MASTER)
        self.assertEqual(tool.master_display(MKV_SIDE), MASTER, "the reduced fractions of a container")
        self.assertEqual(tool.master_display(OTHER_MASTER_SIDE), OTHER_MASTER)

    def test_a_partial_or_out_of_range_master_display_is_left_out(self):
        for key in ("red_x", "white_point_y", "min_luminance", "max_luminance"):
            self.assertIsNone(tool.master_display({k: v for k, v in MASTER_SIDE.items() if k != key}), key)
        for key, value in (("green_x", "-1/50000"), ("green_x", "65536/50000"), ("max_luminance", "0/0"),
                           ("blue_y", "N/A"), ("red_x", None), ("min_luminance", "nan")):
            self.assertIsNone(tool.master_display(dict(MASTER_SIDE, **{key: value})), (key, value))

    def test_a_value_rounds_half_up_to_whole_units(self):
        self.assertEqual([tool._units(v, 50000, 65535) for v in ("1/200000", "1/100000", "3/100000", "0.708", "0")],
                         [0, 1, 2, 35400, 0])
        self.assertEqual(tool._units("65535/50000", 50000, 65535), 65535)
        self.assertIsNone(tool._units("65536/50000", 50000, 65535))
        self.assertEqual(tool._units("429496.7295", 10000, 0xFFFFFFFF), 0xFFFFFFFF)

    def test_the_content_light_level_is_two_16_bit_numbers(self):
        self.assertEqual(tool.max_cll(CLL_SIDE), CLL)
        self.assertEqual(tool.max_cll({"max_content": "0", "max_average": 0}), "0,0")
        self.assertEqual(tool.max_cll({"max_content": 65535, "max_average": 65535}), "65535,65535")
        for bad in ({"max_content": 1000}, {"max_content": 65536, "max_average": 1}, {"max_content": -1, "max_average": 1},
                    {"max_content": "1.5", "max_average": 1}, {"max_content": True, "max_average": 1}):
            self.assertIsNone(tool.max_cll(bad), bad)

    def frame_probe(self, calls, side_data=(OTHER_MASTER_SIDE, OTHER_CLL_SIDE), code=0):
        def run(cmd, **kwargs):
            calls.append(cmd)
            frames = {"frames": [{"side_data_list": list(side_data)}, {"side_data_list": [MASTER_SIDE]}]}
            return subprocess.CompletedProcess(cmd, code, json.dumps(frames) if code == 0 else "", "")
        return run

    def test_the_stream_side_data_comes_first_then_the_first_frame(self):
        path = os.path.abspath("meta_case.mkv")
        info = dict(PQ, path=path, video_index=1)
        cases = (([MASTER_SIDE, CLL_SIDE], (MASTER, CLL), 0), ([CLL_SIDE, MASTER_SIDE], (MASTER, CLL), 0),
                 ([MASTER_SIDE], (MASTER, OTHER_CLL), 1), ([], (OTHER_MASTER, OTHER_CLL), 1),
                 (None, (OTHER_MASTER, OTHER_CLL), 1), ([dict(MASTER_SIDE, red_x="N/A"), CLL_SIDE], (OTHER_MASTER, CLL), 1),
                 ([{"side_data_type": "Display Matrix", "rotation": 0}, "junk", MKV_SIDE], (MASTER, OTHER_CLL), 1))
        for side, want, runs in cases:
            with self.subTest(side=side):
                calls = []
                video = {"index": 1, "codec_type": "video"}
                if side is not None:
                    video["side_data_list"] = side
                # the side data of another stream (cover art at index 0) is not the video's
                data = {"streams": [{"index": 0, "codec_type": "video", "side_data_list": [OTHER_MASTER_SIDE]}, video]}
                with mock.patch.dict(tool._PROBE_DATA, {path: data}), mock.patch.dict(tool._HDR_METADATA, clear=True), \
                        mock.patch.object(tool, "run", self.frame_probe(calls)):
                    got = tool.hdr_metadata(path, info)
                self.assertEqual((got["master_display"], got["max_cll"]), want)
                self.assertEqual(len(calls), runs)
                if runs:
                    cmd = calls[0]
                    self.assertEqual(cmd[1:], ["-v", "quiet", "-print_format", "json", "-select_streams", "1",
                                               "-show_frames", "-read_intervals", "%+#1", "-show_entries",
                                               "frame=side_data_list", path])

    def test_the_metadata_is_read_once_per_probe_of_the_file(self):
        path = os.path.abspath("meta_cache.mp4")
        info = dict(PQ, path=path)
        calls = []
        data = {"streams": [{"index": 0, "codec_type": "video"}]}
        with mock.patch.dict(tool._PROBE_DATA, {path: data}), mock.patch.dict(tool._HDR_METADATA, clear=True), \
                mock.patch.object(tool, "run", self.frame_probe(calls)):
            first = tool.hdr_metadata(path, info)
            again = tool.hdr_metadata("meta_cache.mp4", info)
            self.assertEqual((first, again, len(calls)), ({"master_display": OTHER_MASTER, "max_cll": OTHER_CLL},) * 2 + (1,))
            again["max_cll"] = "changed"
            self.assertEqual(tool.hdr_metadata(path, info)["max_cll"], OTHER_CLL, "a copy each time")
            # a new probe of the file is a new look
            tool._PROBE_DATA[path] = {"streams": [{"index": 0, "codec_type": "video", "side_data_list": [CLL_SIDE]}]}
            self.assertEqual(tool.hdr_metadata(path, info), {"master_display": OTHER_MASTER, "max_cll": CLL})
            self.assertEqual(len(calls), 2)

    def test_a_frame_probe_that_fails_leaves_the_metadata_out(self):
        path = os.path.abspath("meta_fail.mp4")
        info = dict(PQ, path=path)
        for run in (self.frame_probe([], code=1), self.frame_probe([], side_data=()),
                    lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, "not json", ""),
                    lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, "[1, 2]", ""),
                    lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, '{"frames": []}', "")):
            with mock.patch.dict(tool._PROBE_DATA, {path: {"streams": []}}), mock.patch.dict(tool._HDR_METADATA, clear=True), \
                    mock.patch.object(tool, "run", run):
                self.assertEqual(tool.hdr_metadata(path, info), {"master_display": None, "max_cll": None})
        # a file probe() never read: the frame alone, and nothing kept
        calls = []
        with mock.patch.dict(tool._HDR_METADATA, clear=True), mock.patch.object(tool, "run", self.frame_probe(calls)):
            tool._PROBE_DATA.pop(path, None)
            self.assertEqual(tool.hdr_metadata(path, dict(info, video_index=None))["max_cll"], OTHER_CLL)
            self.assertEqual(calls[0][calls[0].index("-select_streams") + 1], "v:0")
            self.assertEqual(tool._HDR_METADATA, {})


class EncodeTests(unittest.TestCase):
    """encode_plan, video_encode_args and report_encode, with the HDR metadata of the PQ fixture."""

    def setUp(self):
        patcher = mock.patch.object(tool, "hdr_metadata", lambda path, info: {"master_display": MASTER, "max_cll": CLL})
        patcher.start()
        self.addCleanup(patcher.stop)

    def encode(self, info, size=(64, 48), destination="out.mp4", **over):
        a = args(**over)
        return tool.video_encode_args(a, info, tool.work_geometry(a, info), size, destination)

    def test_an_sdr_input_with_even_sides_encodes_as_before(self):
        sdr = facts()
        self.assertIsNone(tool.encode_plan(args(), sdr, (64, 48), "out.mp4"))
        self.assertEqual(self.encode(sdr), [
            "-vf", "scale=out_color_matrix=smpte170m:out_range=tv,setparams=colorspace=smpte170m:"
                   "color_primaries=smpte170m:color_trc=smpte170m:range=tv",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"])
        # --lossless takes any size as it is
        for size in ((64, 48), (65, 49)):
            self.assertIsNone(tool.encode_plan(args(lossless=True), sdr, size))
            self.assertEqual(self.encode(sdr, size, "out.mkv", lossless=True, depth=8),
                             ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgra"])
            self.assertEqual(tool.report_encode(args(lossless=True), sdr, None, size), ([], {}))
        self.assertEqual(tool.report_encode(args(), sdr, None, (64, 48)), ([], {}))
        # a facts dict from before the hdr flag reads as SDR
        self.assertIsNone(tool.encode_plan(args(), {k: v for k, v in sdr.items() if k != "hdr"}, (64, 48)))

    def test_an_odd_side_encodes_yuv444p_at_the_exact_size(self):
        sdr = facts(width=101, height=61)
        for size in ((101, 61), (101, 60), (100, 61)):
            with self.subTest(size=size):
                self.assertEqual(tool.encode_plan(args(), sdr, size, "out.mp4"),
                                 {"codec": "libx264", "pix_fmt": "yuv444p", "odd": True, "tag": None, "hdr": False})
                encode = self.encode(sdr, size)
                self.assertEqual(encode[2:], ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv444p"])
                self.assertEqual(encode[:2], self.encode(facts())[:2], "the matrix and the tags of before")
                self.assertEqual(tool.report_encode(args(), sdr, None, size),
                                 (["encoded yuv444p (the size has an odd side)"],
                                  {"encode": {"codec": "libx264", "pix_fmt": "yuv444p"}}))
        # the size the encode reads decides, not the input's (a declared stage, later --size)
        self.assertIsNone(tool.encode_plan(args(), sdr, (100, 60)))
        self.assertEqual(tool.encode_plan(args(), facts(), (81, 48))["pix_fmt"], "yuv444p")

    def test_an_hdr_input_encodes_libx265_10_bit_with_its_tags_and_metadata(self):
        encode = self.encode(PQ)
        self.assertEqual(encode, [
            "-vf", "scale=out_color_matrix=bt2020:out_range=tv,setparams=colorspace=bt2020nc:color_primaries=bt2020:"
                   "color_trc=smpte2084:range=tv",
            "-c:v", "libx265", "-crf", "18", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1",
            "-x265-params", f"log-level=error:master-display={MASTER}:max-cll={CLL}"])
        self.assertEqual(tool.encode_plan(args(), PQ, (64, 48), "out.mp4"), {
            "codec": "libx265", "pix_fmt": "yuv420p10le", "odd": False, "tag": "hvc1", "hdr": True,
            "transfer": "smpte2084", "primaries": "bt2020", "matrix": "bt2020nc", "master_display": MASTER,
            "max_cll": CLL})
        # hvc1 in mp4, m4v and mov
        for destination, tagged in (("out.mov", True), ("OUT.M4V", True), ("a.b.mp4", True), ("out.mkv", False),
                                    ("out", False), (None, False)):
            self.assertEqual("-tag:v" in self.encode(PQ, destination=destination), tagged, destination)
        # an odd side: 4:4:4 in 10 bit
        odd = self.encode(PQ, (65, 49))
        self.assertEqual(odd, encode[:7] + ["yuv444p10le"] + encode[8:])
        # any depth gives the same encode
        for depth in (8, 32):
            self.assertEqual(self.encode(PQ, depth=depth), encode)
        # an HLG input without metadata
        with mock.patch.object(tool, "hdr_metadata", lambda path, info: {"master_display": None, "max_cll": None}):
            hlg = self.encode(HLG, destination="out.mkv")
        self.assertEqual(hlg[1].split(",")[1], "setparams=colorspace=bt2020nc:color_primaries=bt2020:"
                                               "color_trc=arib-std-b67:range=tv")
        self.assertEqual(hlg[-2:], ["-x265-params", "log-level=error"])
        self.assertNotIn("-tag:v", hlg)
        with mock.patch.object(tool, "hdr_metadata", lambda path, info: {"master_display": None, "max_cll": CLL}):
            self.assertEqual(self.encode(PQ)[-1], f"log-level=error:max-cll={CLL}")

    def test_lossless_hdr_stays_rgb_in_ffv1_tagged_with_its_primaries_and_transfer(self):
        for depth, fmt in ((8, "bgra"), (16, "rgba64le")):
            for size in ((64, 48), (65, 49)):
                self.assertEqual(self.encode(PQ, size, "out.mkv", lossless=True, depth=depth),
                                 ["-vf", "setparams=color_primaries=bt2020:color_trc=smpte2084", "-c:v", "ffv1",
                                  "-level", "3", "-pix_fmt", fmt])
        self.assertEqual(self.encode(dict(HLG, color_primaries=None), lossless=True)[1],
                         "setparams=color_primaries=bt2020:color_trc=arib-std-b67")
        with mock.patch.object(tool, "hdr_metadata", side_effect=AssertionError("ffv1 carries no metadata")):
            plan = tool.encode_plan(args(lossless=True), PQ, (65, 49), "out.mkv")
        self.assertEqual((plan["codec"], plan["odd"], plan["matrix"], plan["master_display"], plan["max_cll"]),
                         ("ffv1", False, None, None, None))

    def test_the_report_names_the_hdr_signal_and_the_encode(self):
        phrases, keys = tool.report_encode(args(), PQ, None, (64, 48))
        self.assertEqual(phrases, ["HDR smpte2084 kept as stored (no tone mapping)",
                                   "libx265 10 bit with master-display and max-cll"])
        self.assertEqual(keys, {"hdr": {"transfer": "smpte2084", "primaries": "bt2020", "matrix": "bt2020nc",
                                        "master_display": MASTER, "max_cll": CLL},
                                "encode": {"codec": "libx265", "pix_fmt": "yuv420p10le"}})
        self.assertEqual(list(keys), ["hdr", "encode"])
        with mock.patch.object(tool, "hdr_metadata", lambda path, info: {"master_display": MASTER, "max_cll": None}):
            phrases, keys = tool.report_encode(args(), HLG, None, (65, 49))
        self.assertEqual(phrases, ["HDR arib-std-b67 kept as stored (no tone mapping)", "libx265 10 bit with master-display",
                                   "encoded yuv444p10le (the size has an odd side)"])
        self.assertEqual(keys["encode"], {"codec": "libx265", "pix_fmt": "yuv444p10le"})
        with mock.patch.object(tool, "hdr_metadata", lambda path, info: {"master_display": None, "max_cll": None}):
            self.assertEqual(tool.report_encode(args(), PQ, None, (64, 48))[0][1], "libx265 10 bit")
        phrases, keys = tool.report_encode(args(lossless=True, depth=8), PQ, None, (65, 49))
        self.assertEqual(phrases, ["HDR smpte2084 kept as stored (no tone mapping)", "ffv1 tagged bt2020 smpte2084"])
        self.assertEqual(keys, {"hdr": {"transfer": "smpte2084", "primaries": "bt2020", "matrix": None,
                                        "master_display": None, "max_cll": None},
                                "encode": {"codec": "ffv1", "pix_fmt": "bgra"}})

    def test_the_summary_and_the_report_carry_the_encode_after_the_layers(self):
        a = args(dry_run=True)
        plan = tool.make_plan(a, PQ, {"kind": "frame", "layers": []}, [], "/tmp/s")
        result = tool.make_result(a, PQ, {"kind": "frame"}, "vfx", "out.mp4", plan, [], ())
        self.assertEqual(result["summary"], "fx on in.mp4: 64x48, 24.000 fps, 16 bpc (rgba64le), HDR smpte2084 kept as "
                                            "stored (no tone mapping), libx265 10 bit with master-display and max-cll, "
                                            "decode | vfx | encode -> out.mp4")
        keys = list(result)
        self.assertEqual(keys[keys.index("layers"):keys.index("audio") + 1], ["layers", "hdr", "encode", "audio"])
        odd = tool.make_result(a, facts(width=101, height=61), {"kind": "frame"}, "vfx", "out.mp4", plan, [], ())
        self.assertIn("101x61, 24.000 fps, 16 bpc (rgba64le), encoded yuv444p (the size has an odd side), decode",
                      odd["summary"])
        self.assertNotIn("hdr", odd)
        # the size the encode reads (a declared stage) decides
        crop = tool.make_result(a, facts(), {"kind": "frame"}, "vfx", "out.mp4", plan, [], (), None, (81, 48))
        self.assertEqual(crop["encode"], {"codec": "libx264", "pix_fmt": "yuv444p"})


class VerifyColorTests(unittest.TestCase):
    def plan(self, info, size=(64, 48), **over):
        with mock.patch.object(tool, "hdr_metadata", lambda path, i: {"master_display": None, "max_cll": None}):
            return tool.encode_plan(args(**over), info, size, "out.mp4")

    def test_without_a_plan_nothing_is_checked(self):
        wrong = facts(color_transfer="bt709", pix_fmt="gray")
        self.assertEqual(tool.verify_color(PQ, wrong), [])
        self.assertEqual(tool.verify_color(PQ, wrong, None), [])

    def test_an_hdr_output_keeps_the_transfer_of_its_input_in_10_bit(self):
        plan = self.plan(PQ)
        good = facts(color_transfer="smpte2084", pix_fmt="yuv420p10le")
        self.assertEqual(tool.verify_color(PQ, good, plan), [])
        self.assertEqual(tool.verify_color(PQ, dict(good, color_transfer=None), plan), ["transfer is None, expected smpte2084"])
        self.assertEqual(tool.verify_color(PQ, dict(good, color_transfer="bt709", pix_fmt="yuv420p"), plan),
                         ["transfer is bt709, expected smpte2084", "pixel format is yuv420p, expected yuv420p10le"])
        # the transfer of the input, whatever it is
        self.assertEqual(tool.verify_color(HLG, good, self.plan(HLG)), ["transfer is smpte2084, expected arib-std-b67"])
        odd = self.plan(PQ, (65, 49))
        self.assertEqual(tool.verify_color(PQ, good, odd), ["pixel format is yuv420p10le, expected yuv444p10le"])
        self.assertEqual(tool.verify_color(PQ, dict(good, pix_fmt="yuv444p10le"), odd), [])

    def test_an_ffv1_output_keeps_the_transfer_in_the_format_ffv1_stores(self):
        plan = self.plan(PQ, lossless=True)
        self.assertEqual(tool.verify_color(PQ, facts(color_transfer="smpte2084", pix_fmt="gbrap16le"), plan), [])
        self.assertEqual(tool.verify_color(PQ, facts(color_transfer=None, pix_fmt="gbrap16le"), plan),
                         ["transfer is None, expected smpte2084"])

    def test_an_odd_side_output_is_444(self):
        plan = self.plan(facts(), (81, 48))
        self.assertEqual(tool.verify_color(facts(), facts(pix_fmt="yuv444p"), plan), [])
        self.assertEqual(tool.verify_color(facts(), facts(pix_fmt="yuv420p"), plan), ["pixel format is yuv420p, expected yuv444p"])
        # no transfer check on SDR
        self.assertEqual(tool.verify_color(facts(), facts(pix_fmt="yuv444p", color_transfer="bt709"), plan), [])

    def test_verify_reports_the_color_problems_after_the_others(self):
        plan = self.plan(PQ)
        out = facts(width=66, color_transfer="bt709", pix_fmt="yuv420p10le")
        self.assertEqual(tool.verify(PQ, out, Fraction(24), size=(64, 48), color=plan),
                         ["size is 66x48, expected 64x48", "transfer is bt709, expected smpte2084"])
        self.assertEqual(tool.verify(PQ, out, Fraction(24), size=(64, 48)), ["size is 66x48, expected 64x48"])


class EncoderCheckTests(unittest.TestCase):
    HELP = ("Encoder libx265 [libx265 H.265 / HEVC]:\n    General capabilities: dr1 delay threads\n"
            "    Supported pixel formats: yuv420p yuvj420p yuv422p yuv444p {extra}\nlibx265 AVOptions:\n")
    ABSENT = "Codec 'libx265' is not recognized by FFmpeg.\n\nExiting with exit code 0\n"

    def check(self, stdout, code=0, contract="frames/1", as_json=False):
        """(exit code or None, stdout, stderr, the commands run) of require_hdr_encoder over a canned answer."""
        out, err, calls = io.StringIO(), io.StringIO(), []

        def run(cmd, **kwargs):
            calls.append((cmd, kwargs.get("stdin")))
            return subprocess.CompletedProcess(cmd, code, stdout, "")
        with mock.patch.object(tool, "run", run), mock.patch.object(tool, "ffmpeg_path", lambda: "ffmpeg"), \
                mock.patch.object(tool, "CONTRACT", contract), mock.patch.object(tool, "JSON_OUTPUT", as_json), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                tool.require_hdr_encoder()
                status = None
            except SystemExit as exc:
                status = exc.code
        return status, out.getvalue(), err.getvalue(), calls

    def test_a_libx265_with_10_bit_passes(self):
        status, out, err, calls = self.check(self.HELP.format(extra="gbrp yuv420p10le yuv422p10le yuv444p10le"))
        self.assertEqual((status, out, err), (None, "", ""))
        self.assertEqual(calls, [(["ffmpeg", "-hide_banner", "-h", "encoder=libx265"], subprocess.DEVNULL)])

    def test_no_libx265_is_tool_missing_at_the_encode(self):
        message = ("an HDR input is encoded with libx265, which ffmpeg does not have; use an ffmpeg built with libx265, "
                   "or pass --lossless (ffv1)")
        self.assertEqual(self.check(self.ABSENT)[:3], (1, "", f"[error] {message}\n"))
        self.assertEqual(self.check("", code=127)[:3], (1, "", f"[error] {message}\n"))
        status, out, err, _ = self.check(self.ABSENT, contract="frames/2", as_json=True)
        self.assertEqual(status, 6)
        self.assertEqual(json.loads(out), {"ok": False, "contract": "frames/2", "error": {
            "code": "tool_missing", "stage": "encode", "exit_code": None, "message": message, "stderr_tail": []}})
        self.assertEqual(err, f"error: tool_missing: {message}\n")

    def test_a_libx265_without_10_bit_is_tool_missing(self):
        status, _, err, _ = self.check(self.HELP.format(extra="gbrp gray"))
        self.assertEqual(status, 1)
        self.assertEqual(err, "[error] an HDR input is encoded with libx265 in 10 bit, and the libx265 of ffmpeg cannot "
                              "write yuv420p10le or yuv444p10le; use an ffmpeg whose libx265 has 10 bit, or pass "
                              "--lossless (ffv1)\n")
        self.assertIn("cannot write yuv444p10le;", self.check(self.HELP.format(extra="yuv420p10le"))[2])


# ----------------------------------------------------------------------------
# end to end: ffmpeg, ffprobe and vfx (libx265 for the HDR ones)
# ----------------------------------------------------------------------------

class Pipeline(unittest.TestCase):
    NEEDS_X265 = False

    @classmethod
    def setUpClass(cls):
        if VFX is None:
            raise unittest.SkipTest("needs ffmpeg, ffprobe and a built vfx (cargo build --release -p fx-cli, or VFX_PATH)")
        if cls.NEEDS_X265 and not X265:
            raise unittest.SkipTest("needs libx265 with 10 bit in ffmpeg (the HDR fixtures and the HDR encode)")

    def setUp(self):
        self.dir = tempfile.mkdtemp(dir=TMP)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.dir, name)

    def run_ok(self, *argv):
        proc = apply(*argv, tmp=self.dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc

    def report(self, *argv):
        return json.loads(self.run_ok(*argv, "--json").stdout)

    def assertClean(self):
        """no partial file and no scratch folder left behind."""
        leftovers = [n for n in os.listdir(self.dir) if n.startswith(("apply_", "vfx-spool-"))]
        leftovers += [n for _, _, fs in os.walk(self.dir) for n in fs if ".partial-" in n]
        self.assertEqual(leftovers, [])

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


class OddSizeTests(Pipeline):
    def test_an_odd_input_encodes_yuv444p_at_its_exact_size_with_the_picture_of_the_effect(self):
        src = fixture("odd.mp4")
        out, ref = self.path("odd.mp4"), self.path("ref.mkv")
        data = self.report(src, "selftest-invert", "-o", out)
        self.assertIn("101x61, 24.000 fps, 16 bpc (rgba64le), encoded yuv444p (the size has an odd side), decode",
                      data["summary"])
        self.assertEqual(data["encode"], {"codec": "libx264", "pix_fmt": "yuv444p"})
        self.assertNotIn("hdr", data)
        self.assertEqual(options(data["commands"][-1], "-pix_fmt"), ["rgba64le", "yuv444p"])
        got = stream_facts(out)
        self.assertEqual((got["codec_name"], got["pix_fmt"], got["width"], got["height"]), ("h264", "yuv444p", 101, 61))
        self.assertEqual([got[k] for k in COLOR_KEYS], ["smpte170m", "smpte170m", "smpte170m", "tv"])
        self.assertEqual((data["outputs"][0]["probe"]["acodec"], data["outputs"][0]["probe"]["nb_frames"]), ("aac", FRAMES))
        # the lossless run holds what vfx wrote, and the 4:4:4 encode is that picture within CRF 18, not the input's
        self.run_ok(src, "selftest-invert", "--lossless", "--depth", "8", "-o", ref)
        self.assertGreater(psnr(out, ref), 35.0)
        self.assertLess(psnr(out, src), 20.0)
        text = self.run_ok(src, "selftest-invert", "--dry-run").stdout
        self.assertIn("encoded yuv444p (the size has an odd side)", text.splitlines()[0])
        self.assertClean()

    def test_a_declared_stage_that_leaves_an_odd_side_encodes_yuv444p(self):
        out = self.path("crop.mp4")
        data = self.report(fixture("clip.mp4"), "selftest-invert", "--contract", "frames/2", "--post-stage",
                           ffstage("crop=81:48"), "--geometry", "declared", "-o", out)
        self.assertEqual(data["geometry"], {"width": 81, "height": 48, "depth": 16, "fps": f"{FPS}/1", "frames": FRAMES})
        self.assertEqual(data["encode"], {"codec": "libx264", "pix_fmt": "yuv444p"})
        self.assertIn("encoded yuv444p (the size has an odd side)", data["summary"])
        effect = next(s for s in data["stages"] if s["role"] == "effect")["command"]
        self.assertEqual(effect[effect.index("--width") + 1], "160")
        got = stream_facts(out)
        self.assertEqual((got["pix_fmt"], got["width"], got["height"]), ("yuv444p", 81, 48))
        self.assertEqual(len(raw_frames(out, 8)), 81 * 48 * 4 * FRAMES)
        # an odd input that a declared pre-stage crops to even sides is the encode of before
        even = self.path("even.mp4")
        data = self.report(fixture("odd.mp4"), "selftest-invert", "--contract", "frames/2", "--pre-stage",
                           ffstage("crop=100:60:0:0"), "--geometry", "declared", "-o", even)
        self.assertFalse({"hdr", "encode"} & set(data))
        self.assertEqual((stream_facts(even)["pix_fmt"], data["geometry"]["width"]), ("yuv420p", 100))
        described = self.run_ok(fixture("clip.mp4"), "selftest-invert", "--contract", "frames/2", "--post-stage",
                                ffstage("crop=81:48"), "--geometry", "declared", "--describe")
        self.assertEqual((json.loads(described.stdout)["width"], json.loads(described.stdout)["height"]), (81, 48))
        self.assertClean()

    def test_lossless_and_even_runs_report_as_before(self):
        """the hdr and encode keys and the phrases come only with HDR or an odd side on the lossy path."""
        odd, clip = fixture("odd.mp4"), fixture("clip.mp4")
        for argv in ((odd, "--lossless"), (odd, "--lossless", "--depth", "8"), (clip,), (clip, "--depth", "8"),
                     (clip, "--lossless", "--contract", "frames/2")):
            with self.subTest(argv=argv[1:]):
                data = self.report(argv[0], "selftest-invert", *argv[1:], "--dry-run")
                self.assertFalse({"hdr", "encode"} & set(data))
                self.assertNotIn("odd side", data["summary"])
                self.assertNotIn("HDR", data["summary"])

    def test_an_hdr_layer_is_read_as_stored_in_an_sdr_run(self):
        out = self.path("layered.mp4")
        data = self.report(fixture("clip.mp4"), "selftest-shift", "--layer", f"overlay={fixture('pq264.mp4')}", "-o", out)
        self.assertFalse({"hdr", "encode"} & set(data))
        self.assertEqual(option(data["commands"][0], "-vf"), "scale=160:96:in_color_matrix=bt2020")
        self.assertEqual(stream_facts(out)["pix_fmt"], "yuv420p")
        self.assertClean()


class MissingEncoderTests(Pipeline):
    """an HDR input without libx265 in ffmpeg, by a run() that answers for the encoder query alone."""

    def test_without_libx265_an_hdr_input_is_tool_missing_and_lossless_still_runs(self):
        src = fixture("pq264.mp4")
        real = tool.run

        def no_x265(cmd, **kwargs):
            if "encoder=libx265" in cmd:
                return subprocess.CompletedProcess(cmd, 0, EncoderCheckTests.ABSENT, "")
            return real(cmd, **kwargs)

        with mock.patch.object(tool, "run", no_x265):
            code, out, err = self.run_main(src, "selftest-invert", "--dry-run", "--json", "--contract", "frames/2")
            self.assertEqual(code, 6, err)
            error = json.loads(out)["error"]
            self.assertEqual((error["code"], error["stage"]), ("tool_missing", "encode"))
            code, out, err = self.run_main(src, "selftest-shift", "--frames", "3", "-o", self.path("never.mp4"))
            self.assertEqual((code, out), (1, ""))
            self.assertIn("[error] an HDR input is encoded with libx265", err)
            self.assertFalse(os.path.exists(self.path("never.mp4")))
            # --lossless keeps the HDR signal in ffv1 and asks for no libx265, nor does an SDR input
            code, out, err = self.run_main(src, "selftest-invert", "--lossless", "--dry-run", "--json")
            self.assertEqual(code, 0, err)
            self.assertEqual(json.loads(out)["encode"], {"codec": "ffv1", "pix_fmt": "rgba64le"})
            code, out, err = self.run_main(fixture("clip.mp4"), "selftest-invert", "--dry-run")
            self.assertEqual(code, 0, err)
        self.assertClean()


class HdrTests(Pipeline):
    NEEDS_X265 = True

    def test_pq_keeps_its_tags_and_its_metadata_and_the_picture_goes_through_as_stored(self):
        src = fixture("pq.mp4")
        out = self.path("pq.mp4")
        data = self.report(src, "selftest-shift", "-o", out)
        self.assertEqual(data["hdr"], {"transfer": "smpte2084", "primaries": "bt2020", "matrix": "bt2020nc",
                                       "master_display": MASTER, "max_cll": CLL})
        self.assertEqual(data["encode"], {"codec": "libx265", "pix_fmt": "yuv420p10le"})
        self.assertIn("temporal, 12 frames, HDR smpte2084 kept as stored (no tone mapping), libx265 10 bit with "
                      "master-display and max-cll, decode", data["summary"])
        encode = data["commands"][-1]
        self.assertEqual(option(encode, "-x265-params"), f"log-level=error:master-display={MASTER}:max-cll={CLL}")
        self.assertEqual((option(encode, "-c:v"), option(encode, "-crf"), option(encode, "-tag:v")), ("libx265", "18", "hvc1"))
        self.assertEqual(option(data["commands"][0], "-vf"), "scale=in_color_matrix=bt2020")
        for word in ("tonemap", "zscale", "libplacebo", "colorspace="):
            self.assertNotIn(word, data["commands"][0] + data["commands"][1])
        a, b = stream_facts(src), stream_facts(out)
        self.assertEqual([b[k] for k in COLOR_KEYS], [a[k] for k in COLOR_KEYS])
        self.assertEqual([a[k] for k in COLOR_KEYS], ["bt2020nc", "bt2020", "smpte2084", "tv"])
        self.assertEqual((b["codec_name"], b["codec_tag_string"], b["pix_fmt"]), ("hevc", "hvc1", "yuv420p10le"))
        self.assertEqual(set(frame_side_data(src)), {"Mastering display metadata", "Content light level metadata"})
        self.assertEqual(frame_side_data(out), frame_side_data(src))
        self.assertEqual(data["outputs"][0]["probe"]["acodec"], "aac")
        # an identity keeps the code values: no tone mapping moved them (one to SDR doubles this mean)
        self.assertLess(abs(mean_luma10(out) - mean_luma10(src)), 2.0)
        self.assertGreater(psnr(out, src, "y"), 35.0)
        self.assertClean()

    def test_the_side_data_of_the_container_comes_before_the_first_frame(self):
        src = fixture("two.mkv")
        tool.probe(src)
        stream = next(s for s in tool._PROBE_DATA[os.path.abspath(src)]["streams"] if s["codec_type"] == "video")
        if not stream.get("side_data_list"):
            self.skipTest("this ffmpeg keeps no HDR side data in the container of a transcode")
        self.assertEqual(frame_side_data(src)["Content light level metadata"], {"max_content": 500, "max_average": 200})
        data = self.report(src, "selftest-invert", "--dry-run", "-o", self.path("two.mp4"))
        self.assertEqual((data["hdr"]["master_display"], data["hdr"]["max_cll"]), (MASTER, CLL))
        self.assertEqual(option(data["commands"][-1], "-x265-params"),
                         f"log-level=error:master-display={MASTER}:max-cll={CLL}")

    def test_hlg_keeps_its_transfer_without_metadata(self):
        out = self.path("hlg.mkv")
        data = self.report(fixture("hlg.mp4"), "selftest-invert", "-o", out)
        self.assertEqual(data["hdr"], {"transfer": "arib-std-b67", "primaries": "bt2020", "matrix": "bt2020nc",
                                       "master_display": None, "max_cll": None})
        self.assertIn("HDR arib-std-b67 kept as stored (no tone mapping), libx265 10 bit, decode", data["summary"])
        encode = data["commands"][-1]
        self.assertEqual(option(encode, "-x265-params"), "log-level=error")
        self.assertNotIn("-tag:v", shlex.split(encode), "no hvc1 in mkv")
        got = stream_facts(out)
        self.assertEqual([got[k] for k in COLOR_KEYS] + [got["pix_fmt"]],
                         ["bt2020nc", "bt2020", "arib-std-b67", "tv", "yuv420p10le"])
        self.assertEqual(frame_side_data(out), {})
        self.assertClean()

    def test_lossless_hdr_is_ffv1_rgb_tagged_and_holds_the_bits_of_the_effect(self):
        src = fixture("pq.mp4")
        for depth in (16, 8):
            with self.subTest(depth=depth):
                out = self.path(f"ll{depth}.mkv")
                data = self.report(src, "selftest-invert", "--lossless", "--depth", str(depth), "-o", out)
                self.assertEqual(data["hdr"], {"transfer": "smpte2084", "primaries": "bt2020", "matrix": None,
                                               "master_display": None, "max_cll": None})
                self.assertEqual(data["encode"], {"codec": "ffv1", "pix_fmt": tool.LOSSLESS_PIX_FMT[depth]})
                self.assertIn("HDR smpte2084 kept as stored (no tone mapping), ffv1 tagged bt2020 smpte2084", data["summary"])
                got = stream_facts(out)
                self.assertEqual((got["codec_name"], got["color_primaries"], got["color_transfer"]),
                                 ("ffv1", "bt2020", "smpte2084"))
                # the decode of apply.py, vfx by hand and the file read back: bit for bit
                decode = shlex.split(self.report(src, "selftest-invert", "--lossless", "--depth", str(depth),
                                                 "--dry-run")["commands"][0])
                raw_in, raw_want = self.path(f"in{depth}.raw"), self.path(f"want{depth}.raw")
                with open(raw_in, "wb") as handle:
                    handle.write(subprocess.run(decode, shell=False, check=True, capture_output=True).stdout)
                vfx = subprocess.run([VFX, "apply", "selftest-invert", "--width", "64", "--height", "48", "--depth",
                                      str(depth), "--in", raw_in, "--out", raw_want], shell=False, capture_output=True,
                                     text=True)
                self.assertEqual(vfx.returncode, 0, vfx.stderr)
                with open(raw_want, "rb") as handle:
                    want = handle.read()
                self.assertEqual(len(want), 64 * 48 * tool.BYTES_PER_PIXEL[depth] * FRAMES)
                self.assertTrue(raw_frames(out, depth) == want, "the lossless file does not hold what vfx wrote")
                self.assertClean()

    def test_an_odd_hdr_size_encodes_yuv444p10le(self):
        out = self.path("pq65.mp4")
        data = self.report(fixture("pq65.mp4"), "selftest-invert", "-o", out)
        self.assertEqual(data["encode"], {"codec": "libx265", "pix_fmt": "yuv444p10le"})
        # x265 writes a content light level of 0,0 beside a master display; it goes back as stored
        self.assertEqual((data["hdr"]["master_display"], data["hdr"]["max_cll"]), (MASTER, "0,0"))
        self.assertIn("65x49, 24.000 fps, 16 bpc (rgba64le), HDR smpte2084 kept as stored (no tone mapping), libx265 10 "
                      "bit with master-display and max-cll, encoded yuv444p10le (the size has an odd side), decode",
                      data["summary"])
        got = stream_facts(out)
        self.assertEqual((got["width"], got["height"], got["pix_fmt"], got["color_transfer"], got["codec_tag_string"]),
                         (65, 49, "yuv444p10le", "smpte2084", "hvc1"))
        self.assertEqual(frame_side_data(out), frame_side_data(fixture("pq65.mp4")))
        self.assertClean()

    def test_an_hdr_run_with_an_hdr_layer(self):
        out = self.path("both.mp4")
        data = self.report(fixture("pq.mp4"), "selftest-shift", "--layer", f"overlay={fixture('hlg.mp4')}", "-o", out)
        self.assertEqual(data["hdr"]["transfer"], "smpte2084")
        self.assertEqual(option(data["commands"][0], "-vf"), "scale=in_color_matrix=bt2020")
        got = stream_facts(out)
        self.assertEqual((got["color_transfer"], got["pix_fmt"]), ("smpte2084", "yuv420p10le"))
        self.assertClean()

    def test_any_depth_gives_the_same_hdr_output(self):
        src = fixture("pq.mp4")
        for depth in (8, 32):
            with self.subTest(depth=depth):
                out = self.path(f"d{depth}.mp4")
                data = self.report(src, "selftest-invert", "--depth", str(depth), "-o", out)
                self.assertEqual((data["pix_fmt"], data["encode"]["pix_fmt"]), (tool.PIX_FMT[depth], "yuv420p10le"))
                got = stream_facts(out)
                self.assertEqual([got[k] for k in COLOR_KEYS] + [got["pix_fmt"]],
                                 ["bt2020nc", "bt2020", "smpte2084", "tv", "yuv420p10le"])
                self.assertEqual(frame_side_data(out), frame_side_data(src))
        self.assertClean()

    def test_a_frames_2_hdr_run_reports_the_hdr_keys_beside_the_contract(self):
        data = self.report(fixture("pq.mp4"), "selftest-invert", "--contract", "frames/2", "--dry-run", "--lossless")
        self.assertEqual((data["ok"], data["contract"], data["geometry"]["width"]), (True, "frames/2", 64))
        self.assertEqual(data["encode"]["codec"], "ffv1")
        self.assertEqual(data["hdr"]["transfer"], "smpte2084")


class RetagRecipeTests(Pipeline):
    """an HDR file whose tags are missing or wrong is not detected: it renders as SDR, as it always has. the
    recipe of docs/ffmpeg.md retags it, the stream with a metadata bitstream filter and the container with
    the three color options of ffmpeg, and then it is an HDR input."""
    NEEDS_X265 = True

    RECIPES = (("plain265.mp4", "hevc_metadata", 16, "smpte2084", "hevc"),
               ("plain265.mkv", "hevc_metadata", 16, "smpte2084", "hevc"),
               ("sdr265.mp4", "hevc_metadata", 16, "smpte2084", "hevc"),
               ("plain264.mp4", "h264_metadata", 18, "arib-std-b67", "h264"))

    def test_the_recipe_makes_an_untagged_or_mistagged_file_an_hdr_input(self):
        for name, bsf, code, trc, raw in self.RECIPES:
            with self.subTest(name=name):
                src = fixture(name)
                before = self.report(src, "selftest-invert", "--dry-run")
                self.assertFalse({"hdr", "encode"} & set(before), "not detected: it renders as it always has")
                self.assertEqual((option(before["commands"][-1], "-c:v"), options(before["commands"][-1], "-pix_fmt")),
                                 ("libx264", ["rgba64le", "yuv420p"]))
                fixed = self.path("retagged" + os.path.splitext(name)[1])
                ff("-i", src, "-map", "0", "-c", "copy", "-bsf:v",
                   f"{bsf}=colour_primaries=9:transfer_characteristics={code}:matrix_coefficients=9",
                   "-color_primaries", "bt2020", "-color_trc", trc, "-colorspace", "bt2020nc", fixed)
                after = self.report(fixed, "selftest-invert", "--dry-run")
                self.assertEqual((after["hdr"]["transfer"], after["hdr"]["primaries"], after["hdr"]["matrix"]),
                                 (trc, "bt2020", "bt2020nc"))
                # the stream carries the tags too: a raw copy, which has no container, reads them back
                stream = self.path("stream." + raw)
                ff("-i", fixed, "-map", "0:v", "-c", "copy", "-f", raw, stream)
                got = stream_facts(stream)
                self.assertEqual((got["color_transfer"], got["color_primaries"], got["color_space"]), (trc, "bt2020", "bt2020nc"))
                for leftover in (fixed, stream):
                    os.remove(leftover)

    def test_the_bitstream_filter_alone_leaves_a_container_tag_standing(self):
        fixed = self.path("half.mp4")
        ff("-i", fixture("sdr265.mp4"), "-map", "0", "-c", "copy", "-bsf:v",
           "hevc_metadata=colour_primaries=9:transfer_characteristics=16:matrix_coefficients=9", fixed)
        got = stream_facts(fixed)
        if got["color_transfer"] == "smpte2084":
            self.skipTest("this ffmpeg retags the container from the bitstream filter alone")
        self.assertEqual(got["color_transfer"], "bt709")
        self.assertNotIn("hdr", self.report(fixed, "selftest-invert", "--dry-run"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
