#!/usr/bin/env python3
"""tests for tools/ffstage.py, the stage ffmpeg-filter. stdlib unittest, no network.

    python tools/test_ffstage.py

the parser tests need nothing but python. the stage tests run ffstage.py over raw frames with the
ffmpeg tools/apply.py finds (MEDIA_FFMPEG_PATH, FFMPEG or PATH); they fail rather than skip without
it. every child runs under a deadline and is killed with its process tree when it passes it. the
manifest and the describe answers are checked against contracts/ with the validator of
tools/handoff.py.
"""

import json
import os
import random
import struct
import subprocess
import sys
import time
import unittest
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
FFSTAGE = os.path.join(HERE, "ffstage.py")
CONTRACTS = os.path.join(os.path.dirname(HERE), "contracts")

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, HERE)
import ffstage  # noqa: E402
import handoff  # noqa: E402

ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
DEADLINE = 60


def schema(name):
    with open(os.path.join(CONTRACTS, name), encoding="utf-8") as f:
        return json.load(f)


def run_stage(*args, data=b"", deadline=DEADLINE):
    """ffstage.py with `args` and `data` on stdin: (exit code, stdout bytes, stderr text)."""
    proc = subprocess.Popen([sys.executable, FFSTAGE] + list(map(str, args)), shell=False, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ENV,
                            start_new_session=os.name != "nt")
    try:
        out, err = proc.communicate(data, timeout=deadline)
    except subprocess.TimeoutExpired:
        ffstage.kill_tree(proc)
        proc.communicate()
        raise AssertionError(f"ffstage.py {' '.join(map(str, args))} ran past its {deadline} s deadline")
    return proc.returncode, out, err.decode("utf-8", "replace")


def geometry(width, height, depth, fps="24/1", frames=None):
    args = ["--width", width, "--height", height, "--depth", depth, "--fps", fps]
    return args + (["--frames", frames] if frames is not None else [])


def stage(graph, *more, expect=None):
    params = ["--param", f"graph={graph}"] + (["--param", f"expect={expect}"] if expect else [])
    return ["apply", ffstage.STAGE_ID] + params + list(more)


def frames_of(width, height, depth, count, seed=7):
    """`count` random frames; at depth 32 floats that are multiples of 1/256 in 0..1."""
    rng = random.Random(seed)
    if depth == 32:
        n = width * height * 4 * count
        return struct.pack(f"<{n}f", *(rng.randrange(257) / 256 for _ in range(n)))
    return bytes(rng.randrange(256) for _ in range(width * height * ffstage.BYTES_PER_PIXEL[depth] * count))


# ----------------------------------------------------------------------------
# no ffmpeg needed
# ----------------------------------------------------------------------------

class ParseTests(unittest.TestCase):
    def test_the_version_line_and_the_measured_version(self):
        self.assertEqual(ffstage.version_of("ffmpeg version 9.0.2-full_build-www.gyan.dev Copyright"),
                         ("9.0.2-full_build-www.gyan.dev", "9.0.2"))
        self.assertEqual(ffstage.version_of("ffmpeg version n7.1 Copyright")[1], "7.1")
        self.assertEqual(ffstage.version_of("nothing here"), ("unknown", None))

    def test_pixel_formats_and_filters_are_read_from_the_listings(self):
        fmts = ffstage.pix_fmts_of("Pixel formats:\n-----\nIO... rgba     4  32  8-8-8-8\n"
                                   "I.... yuvj411p 3 12 8-8-8\n..H.. vdpau 0 0 0\n")
        self.assertEqual(fmts, {"rgba": (True, True), "yuvj411p": (True, False), "vdpau": (False, False)})
        names = ffstage.filters_of("Filters:\n  ------\n TS aap  AA->A  Apply.\n .. crop  V->V  Crop.\n"
                                   " .. movie  |->N  Read.\n")
        self.assertEqual(names, {"aap", "crop", "movie"})

    def test_a_graph_is_split_into_filters_outside_quotes_and_labels(self):
        graph = "[in]crop=80:48,hflip@a;[x]select='eq(n\\,0)',scale=w=1:h=1[out]"
        self.assertEqual(ffstage.filter_names(graph), ["crop", "hflip", "select", "scale"])
        self.assertEqual(ffstage.filter_names("drawtext=text='a,b;c'"), ["drawtext"])

    def test_filters_outside_the_safe_list_are_found(self):
        self.assertEqual(ffstage.unsafe_filters("movie=x.mp4[a];[in][a]overlay"), ["movie", "overlay"])
        self.assertEqual(ffstage.unsafe_filters("hflip,lut3d=a.cube,psnr@p,lut3d"), ["lut3d", "psnr"])
        self.assertEqual(ffstage.unsafe_filters("drawtext=text='a'"), ["drawtext"])
        for graph in ("hflip", "scale=2:2", "scale=w=2:h=2", "crop=80:48,negate@n", "transpose=dir=clock",
                      "select='eq(n,0)',setpts=PTS-STARTPTS"):
            self.assertEqual(ffstage.unsafe_filters(graph), [], graph)

    def test_the_safe_list_holds_frame_filters_only(self):
        self.assertTrue({"null", "scale", "crop", "pad", "eq", "negate", "hflip", "fps", "setpts"} <= ffstage.SAFE_FILTERS)
        self.assertTrue(ffstage.SAFE_FILTERS.isdisjoint({"movie", "amovie", "sendcmd", "zmq", "lut3d", "subtitles",
                                                         "drawtext", "curves", "overlay"}))

    def test_the_colour_matrix_rule_of_apply(self):
        for (w, h), m in (((1280, 720), "bt709"), ((1920, 1080), "bt709"), ((1280, 16), "bt709"),
                          ((16, 720), "bt709"), ((1279, 719), "smpte170m"), ((160, 96), "smpte170m")):
            self.assertEqual(ffstage.matrix_for(w, h), m, (w, h))
        self.assertTrue(ffstage.full_graph("hflip", 160, 96).startswith("setparams=colorspace=smpte170m:"))
        self.assertTrue(ffstage.full_graph("hflip", 160, 96).endswith(",hflip"))

    def test_rates_are_exact_fractions(self):
        self.assertEqual(ffstage.fps_arg("30000/1001"), Fraction(30000, 1001))
        self.assertEqual(ffstage.fps_arg("24"), Fraction(24))
        for bad in ("29.97", "0/1", "24/0", "-24", "x"):
            with self.assertRaises(Exception, msg=bad):
                ffstage.fps_arg(bad)

    def test_the_dry_run_is_bounded_by_60_frames_and_2_seconds(self):
        self.assertEqual(ffstage.dry_count(Fraction(24), None), 48)
        self.assertEqual(ffstage.dry_count(Fraction(60), None), 60)
        self.assertEqual(ffstage.dry_count(Fraction(120), None), 60)
        self.assertEqual(ffstage.dry_count(Fraction(30000, 1001), None), 59)
        self.assertEqual(ffstage.dry_count(Fraction(24), 5), 5)
        self.assertEqual(ffstage.dry_count(Fraction(24), 0), 1)
        self.assertEqual(ffstage.dry_count(Fraction(1, 2), None), 1)

    def test_framemd5_gives_time_base_size_and_timestamps(self):
        text = ("#tb 0: 1001/30000\n#dimensions 0: 32x16\n#stream#, dts, pts, duration, size, hash\n"
                "0,          0,          0,        1,     4096, ab\n0,          1,          2,        1,     4096, cd\n")
        self.assertEqual(ffstage.parse_framemd5(text), (Fraction(1001, 30000), (32, 16), [0, 2], [1, 1]))

    def test_params_are_checked(self):
        self.assertEqual(ffstage.read_params([("graph", "hflip")]), ("hflip", "preserve", False))
        self.assertEqual(ffstage.read_params([("graph", "a=b=c"), ("expect", "declared"), ("unsafe", "1")]),
                         ("a=b=c", "declared", True))
        for pairs, kind in (([("graph", "x"), ("speed", "1")], "usage"), ([("expect", "declared")], "usage"),
                            ([("graph", "x"), ("graph", "y")], "usage"), ([("graph", "x"), ("expect", "grow")], "unsupported"),
                            ([("graph", "x"), ("unsafe", "yes")], "unsupported")):
            with self.assertRaises(ffstage.StageError) as caught:
                ffstage.read_params(pairs)
            self.assertEqual(caught.exception.kind, kind, pairs)

    def test_another_ffmpeg_version_is_labelled_unverified(self):
        fake = "fixture-ffmpeg"
        ffstage._CAPS[fake] = {"version": "7.1", "verified": False, "filters": set(),
                               "pix_fmts": {"rgba": (True, True), "rgba64le": (True, True)}}
        try:
            data = ffstage.manifest(fake)
        finally:
            del ffstage._CAPS[fake]
        self.assertIn("unverified", data["provider"]["version"])
        self.assertEqual(data["provider"]["ffmpeg"], {"version": "7.1", "verified": False})
        self.assertEqual(data["stages"][0]["depths"], [8, 16])
        self.assertEqual(handoff.validate(data, schema("stage-manifest.schema.json")), [])


# ----------------------------------------------------------------------------
# the stage over real frames
# ----------------------------------------------------------------------------

class StageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ffmpeg = ffstage.ffmpeg_path()
        cls.caps = ffstage.capabilities(cls.ffmpeg)

    def describe(self, graph, *geo, expect="declared"):
        code, out, err = run_stage(*stage(graph, "--describe", *geometry(*geo), expect=expect))
        return code, json.loads(out.decode("utf-8")), err

    def test_probing_finds_the_formats_and_filters_of_this_ffmpeg(self):
        for fmt in ffstage.PIX_FMT.values():
            self.assertEqual(self.caps["pix_fmts"].get(fmt), (True, True), fmt)
        self.assertTrue({"crop", "hflip", "movie", "sendcmd"} <= self.caps["filters"])
        self.assertEqual(self.caps["verified"],
                         self.caps["version"].removeprefix("n").split("-")[0] == ffstage.MEASURED_VERSION)

    def test_list_prints_a_manifest_that_validates(self):
        code, out, err = run_stage("list", "--json")
        self.assertEqual(code, 0, err)
        data = json.loads(out.decode("utf-8"))
        self.assertEqual(handoff.validate(data, schema("stage-manifest.schema.json")), [])
        self.assertEqual((data["contract"], data["provider"]["id"]), ("frames/2", "ffstage"))
        (st,) = data["stages"]
        self.assertEqual((st["id"], st["kind"], st["spool"], st["geometry"], st["depths"]),
                         ("ffmpeg-filter", "frame", False, "declared", [8, 16, 32]))
        self.assertEqual([p["id"] for p in st["params"]], ["graph", "expect", "unsafe"])
        self.assertEqual(data["provider"]["command"][-1], FFSTAGE)

    def test_preserve_gives_bytes_out_equal_to_bytes_in_at_every_depth(self):
        for depth in (8, 16, 32):
            for w, h, n in ((33, 17, 3), (1, 1, 2), (2, 2, 1)):
                data = frames_of(w, h, depth, n)
                code, out, err = run_stage(*stage("hflip,hflip", *geometry(w, h, depth, frames=n)), data=data)
                self.assertEqual(code, 0, err)
                self.assertEqual(out, data, (depth, w, h))
                code, out, err = run_stage(*stage("hflip", *geometry(w, h, depth)), data=data)
                self.assertEqual((code, len(out)), (0, len(data)), err)

    def test_describe_measures_declared_geometry_and_keeps_the_exact_rate(self):
        code, answer, err = self.describe("crop=80:48", 160, 96, 8, "30000/1001", 12)
        self.assertEqual(code, 0, err)
        self.assertEqual(handoff.validate(answer, schema("describe.schema.json")), [])
        self.assertEqual(answer, {"ok": True, "contract": "frames/2", "stage": "ffmpeg-filter", "width": 80,
                                  "height": 48, "depth": 8, "fps": "30000/1001", "frames": 12})
        # the count: exact when the dry run saw the whole input, kept when the graph keeps it, else unknown
        self.assertEqual(self.describe("trim=end_frame=5", 32, 16, 16, "24/1", 12)[1]["frames"], 5)
        self.assertEqual(self.describe("hflip", 32, 16, 16, "24/1", 500)[1]["frames"], 500)
        self.assertIsNone(self.describe("trim=end_frame=5", 32, 16, 16, "24/1", 500)[1]["frames"])
        self.assertIsNone(self.describe("crop=16:8", 32, 16, 32, "24/1")[1]["frames"])
        code, answer, err = self.describe("hflip", 32, 16, 32, "24/1", 7, expect="preserve")
        self.assertEqual((code, answer["width"], answer["frames"]), (0, 32, 7), err)

    def test_a_declared_run_writes_the_declared_frames(self):
        data = frames_of(32, 16, 8, 4)
        code, out, err = run_stage(*stage("crop=16:8:0:0", *geometry(32, 16, 8, frames=4), expect="declared"),
                                   data=data)
        self.assertEqual(code, 0, err)
        self.assertEqual(len(out), 16 * 8 * 4 * 4)
        self.assertEqual(out[:64], data[:64])

    def test_refusals_exit_3_with_an_error_document(self):
        cases = (("crop=80:48", "preserve", "pass --param expect=declared"),
                 ("fps=12", "declared", "frame rate"),
                 ("setpts=2*PTS", "declared", "frame rate"),
                 ("lut3d=a.cube", "declared", "unsafe=1"),
                 ("movie=a.mp4[m];[in][m]overlay", "declared", "unsafe=1"),
                 ("nosuchfilter", "declared", "no filter `nosuchfilter`"))
        for graph, expect, text in cases:
            code, answer, err = self.describe(graph, 160, 96, 8, "30000/1001", 12, expect=expect)
            self.assertEqual(code, 3, (graph, err))
            self.assertEqual(handoff.validate(answer, schema("error.schema.json")), [], graph)
            self.assertEqual((answer["error"]["code"], answer["error"]["stage"]), ("unsupported", "ffmpeg-filter"))
            self.assertIn(text, answer["error"]["message"], graph)
        code, out, err = run_stage("apply", "nosuch-stage", *geometry(8, 8, 8), "--param", "graph=null", "--describe")
        self.assertEqual(code, 3, err)

    def test_filters_outside_the_safe_list_are_refused_with_exit_3_unless_unsafe(self):
        for graph in ("lut3d=look.cube", "psnr", "sendcmd=f=c.txt,hflip", "signature", "movie=a.mp4"):
            with self.assertRaises(ffstage.StageError) as caught:
                ffstage.check_graph(graph, False, self.caps)
            self.assertEqual(caught.exception.kind, "unsupported", graph)
            self.assertIn("unsafe=1", caught.exception.message)
        for graph in ("hflip", "scale=2:2", "transpose=dir=clock"):
            ffstage.check_graph(graph, False, self.caps)
        ffstage.check_graph("lut3d=look.cube", True, self.caps)
        code, out, err = run_stage(*stage("lut3d=look.cube", *geometry(8, 8, 8)))
        self.assertEqual(code, 3, err)
        self.assertEqual(out, b"")
        self.assertIn("unsafe=1", err)

    def test_a_graph_ffmpeg_refuses_exits_3(self):
        code, out, err = run_stage(*stage("hflip=nosuchoption=1", *geometry(8, 8, 8)))
        self.assertEqual(code, 3, err)
        self.assertEqual(out, b"")
        self.assertIn("ffmpeg cannot run the graph", err)

    def test_unsafe_lets_a_file_reading_graph_through_the_check(self):
        with self.assertRaises(ffstage.StageError):
            ffstage.check_graph("sendcmd=f=a.txt,hflip", False, self.caps)
        ffstage.check_graph("sendcmd=f=a.txt,hflip", True, self.caps)

    def test_usage_errors_exit_2(self):
        for args in (stage("null", "--height", 8, "--depth", 8, "--fps", "24/1"),
                     stage("null", *geometry(8, 8, 8, fps="29.97")),
                     stage("null", *geometry(8, 8, 12)),
                     ["apply", "ffmpeg-filter", *geometry(8, 8, 8)]):
            code, out, err = run_stage(*args)
            self.assertEqual(code, 2, (args, err))
            self.assertEqual(out, b"")

    def test_truncated_input_is_bad_input_although_ffmpeg_exits_0(self):
        fb = 32 * 16 * 4
        code, out, err = run_stage(*stage("null", *geometry(32, 16, 8)), data=bytes(fb * 3 + 7))
        self.assertEqual(code, 4, err)
        self.assertIn("7 bytes after 3 frames", err)
        self.assertEqual(len(out), fb * 3)
        code, out, err = run_stage(*stage("null", *geometry(32, 16, 8, frames=5)), data=bytes(fb * 3))
        self.assertEqual(code, 4, err)
        self.assertIn("--frames said 5", err)

    def test_a_graph_that_drops_frames_is_refused_under_preserve_and_counted_under_declared(self):
        data = frames_of(16, 16, 8, 3)
        code, out, err = run_stage(*stage("select='lt(n\\,1)'", *geometry(16, 16, 8)), data=data)
        self.assertEqual(code, 3, err)
        code, out, err = run_stage(*stage("select='lt(n\\,1)'", *geometry(16, 16, 8, frames=3), expect="declared"),
                                   data=data)
        self.assertEqual((code, len(out)), (0, 16 * 16 * 4), err)

    def test_the_colour_matrix_of_the_rule_reaches_a_yuv_conversion(self):
        # one pure red frame converted to yuv444p: luma 63 under bt709 (tv range), 81 under smpte170m
        for (w, h), luma in (((1280, 720), 63), ((160, 96), 81)):
            red = b"\xff\x00\x00\xff" * (w * h)
            cmd = [self.ffmpeg, "-v", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", "rgba", "-video_size",
                   f"{w}x{h}", "-framerate", "24", "-i", "-", "-vf", ffstage.full_graph("format=yuv444p", w, h),
                   "-f", "rawvideo", "-"]
            proc = subprocess.run(cmd, shell=False, input=red, capture_output=True, timeout=DEADLINE)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout[0], luma, (w, h))

    def test_a_stage_that_waits_on_an_open_input_is_killed_with_its_tree_by_the_deadline(self):
        argv = [sys.executable, FFSTAGE] + [str(a) for a in stage("null", *geometry(32, 16, 8))]
        proc = subprocess.Popen(argv, shell=False,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ENV,
                                start_new_session=os.name != "nt")
        try:
            proc.stdin.write(bytes(32 * 16 * 4))
            proc.stdin.flush()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and proc.poll() is None:
                time.sleep(0.1)
            self.assertIsNone(proc.poll(), "the stage should still wait for the rest of its input")
        finally:
            started = time.monotonic()
            ffstage.kill_tree(proc)
            proc.wait(timeout=DEADLINE)
            for end in (proc.stdin, proc.stdout, proc.stderr):
                end.close()
        self.assertLess(time.monotonic() - started, DEADLINE)
        self.assertIsNotNone(proc.returncode)


if __name__ == "__main__":
    unittest.main(verbosity=2)
