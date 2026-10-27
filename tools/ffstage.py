#!/usr/bin/env python3
"""ffstage.py: the stage `ffmpeg-filter`, one ffmpeg filter graph over raw frames.

the python twin of sapphire's ffstage.rs. it follows the stage convention of docs/handoff.md
(frames/2): a stage manifest from `list --json`, frames on stdin and stdout with `apply`, the stage
exits 0, 2 usage, 3 unsupported, 4 bad_input and 1 for anything else, and `--describe` for its
declared geometry. ffmpeg stays upstream and unmodified; it runs as a child process.

  python tools/ffstage.py list --json
  python tools/ffstage.py apply ffmpeg-filter --width 160 --height 96 --depth 8 --fps 24/1 \\
      --param graph=hflip < in.rgba > out.rgba
  python tools/ffstage.py apply ffmpeg-filter --width 160 --height 96 --depth 8 --fps 24/1 \\
      --frames 12 --param graph=crop=80:48 --param expect=declared --describe

  --param graph=G    the filter graph, one input and one output (a -vf graph)
  --param expect=E   preserve (default): the output has the size, rate and frame count of the
                     input, so bytes out equal bytes in; declared: --describe measures the
                     output geometry with a bounded dry run (at most 60 frames and 2 seconds
                     of video), and only the size and the frame count may change
  --param unsafe=1   allow filters outside SAFE_FILTERS. without it a graph may only use the
                     frame filters of that list (scale, crop, pad, eq, negate and the rest),
                     which read and write nothing but the frames

frames are in the ffmpeg pixel format of the depth (8 rgba, 16 rgba64le, 32 gbrapf32le), whole
frames back to back. rates are exact fractions (30000/1001, never 29.97). a graph that converts to
YUV inside uses the colour matrix rule of tools/apply.py: bt709 when the frame is 1280 wide or 720
high and up, smpte170m below. ffmpeg exits 0 on an input that ends inside a frame, so the stage
counts bytes and frames itself and exits 4. before use it runs `ffmpeg -version`, `-pix_fmts` and
`-filters`; 9.0.2 is the measured version, any other is reported as unverified.

stdlib only; every child runs with list args, shell=False.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
from fractions import Fraction

STAGE_ID = "ffmpeg-filter"
PROVIDER_ID = "ffstage"
VERSION = "ffstage 1"
MEASURED_VERSION = "9.0.2"
CONTRACT = "frames/2"
PIX_FMT = {8: "rgba", 16: "rgba64le", 32: "gbrapf32le"}
BYTES_PER_PIXEL = {8: 4, 16: 8, 32: 16}
EXPECT = ("preserve", "declared")
# the dry run that measures a graph: at most this many frames and this many seconds of video
DRY_FRAMES = 60
DRY_SECONDS = 2
# seconds any one probe or dry run may take before it is killed
PROBE_TIMEOUT = 30
RELAY_CHUNK = 1 << 20
TAIL = 5
# the frame filters a graph may use without --param unsafe=1: they read and write nothing but the
# frames. any other filter needs it. the same list as the Rust twin (sapphire
# crates/neutral-handoff/src/ffstage.rs).
SAFE_FILTERS = {"boxblur", "colorbalance", "colorchannelmixer", "colorlevels", "copy", "crop", "deflicker", "drawbox",
                "drawgrid", "edgedetect", "eq", "fade", "format", "fps", "framerate", "gblur", "geq", "hflip", "hue",
                "lut", "lutrgb", "lutyuv", "minterpolate", "negate", "noise", "null", "pad", "reverse", "rotate",
                "scale", "select", "setdar", "setparams", "setpts", "setsar", "smartblur", "sobel", "tblend", "tmix",
                "transpose", "trim", "unsharp", "vflip", "vignette"}
# x-stage-exits of contracts/error.schema.json
EXIT = {"usage": 2, "unsupported": 3, "bad_input": 4, "stage_failed": 1}


class StageError(Exception):
    """a refusal or failure: the error code of the contract, a one line message and the child's stderr tail."""

    def __init__(self, kind, message, tail=None, exit_code=None):
        super().__init__(message)
        self.kind, self.message, self.tail, self.exit_code = kind, message, list(tail or []), exit_code


class ArgParser(argparse.ArgumentParser):
    def error(self, message):
        raise StageError("usage", message)


# ----------------------------------------------------------------------------
# ffmpeg: where it is and what it can do
# ----------------------------------------------------------------------------

def ffmpeg_path():
    """MEDIA_FFMPEG_PATH or FFMPEG, then PATH, then the places a windows install lands (as tools/apply.py)."""
    for var in ("MEDIA_FFMPEG_PATH", "FFMPEG"):
        override = os.environ.get(var)
        if override:
            found = shutil.which(override)
            if not found:
                raise StageError("stage_failed", f"{var} names a program that cannot be found: {override}")
            return found
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidates = []
    if os.environ.get("LOCALAPPDATA"):
        candidates.append(os.path.join(os.environ["LOCALAPPDATA"], "Microsoft", "WinGet", "Links", "ffmpeg.exe"))
    for var in ("ProgramFiles", "ProgramFiles(x86)"):
        if os.environ.get(var):
            candidates.append(os.path.join(os.environ[var], "ffmpeg", "bin", "ffmpeg.exe"))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise StageError("stage_failed", "ffmpeg not found (set MEDIA_FFMPEG_PATH or put ffmpeg on PATH)")


def kill_tree(proc):
    """kill a child and what it started: taskkill /T on windows, the process group on posix (unproven
    until a CI run), then the child itself."""
    if os.name == "nt":
        if proc.poll() is None:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], shell=False, stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=PROBE_TIMEOUT)
    else:
        # a child of capture() leads its own session, so its group is signalled even when the leader has
        # exited (the members may still run); the ffmpeg of run_graph shares this group and is killed alone
        try:
            if getattr(proc, "leads_group", False) or (proc.poll() is None and os.getpgid(proc.pid) == proc.pid):
                os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass


def capture(cmd, timeout=PROBE_TIMEOUT):
    """run a child with a deadline; (exit code, stdout, stderr lines). a child past the deadline is killed."""
    try:
        proc = subprocess.Popen(cmd, shell=False, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, start_new_session=os.name != "nt")
        proc.leads_group = os.name != "nt"
    except OSError as exc:
        raise StageError("stage_failed", f"cannot start {os.path.basename(cmd[0])}: {exc}")
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        proc.communicate()
        raise StageError("stage_failed", f"{os.path.basename(cmd[0])} {cmd[1]} ran past {timeout} s and was killed")
    return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace").strip().splitlines()


def version_of(text):
    """the version in the first line of `ffmpeg -version`: (full text, numeric part or None)."""
    match = re.search(r"ffmpeg version (\S+)", text)
    full = match.group(1) if match else "unknown"
    number = re.match(r"n?(\d+(?:\.\d+)*)", full)
    return full, number.group(1) if number else None


def pix_fmts_of(text):
    """{name: (can read, can write)} from `ffmpeg -pix_fmts`."""
    found = {}
    for line in text.splitlines():
        match = re.match(r"^([I.])([O.])[A-Z.]{3}\s+(\S+)\s+\d+\s+\d+", line)
        if match:
            found[match.group(3)] = (match.group(1) == "I", match.group(2) == "O")
    return found


def filters_of(text):
    """the filter names of `ffmpeg -filters`."""
    return {m.group(1) for m in re.finditer(r"(?m)^\s*[A-Z.]{2,3}\s+(\S+)\s+\S*->\S*\s", text)}


_CAPS = {}


def capabilities(ffmpeg):
    """probe ffmpeg once: version, whether it is the measured one, pixel formats and filters."""
    if ffmpeg not in _CAPS:
        caps = {}
        for flag in ("-version", "-pix_fmts", "-filters"):
            code, out, err = capture([ffmpeg, "-hide_banner", flag] if flag != "-version" else [ffmpeg, flag])
            if code != 0:
                raise StageError("stage_failed", f"`ffmpeg {flag}` exited with code {code}", err[-TAIL:], code)
            caps[flag] = out
        full, number = version_of(caps["-version"])
        _CAPS[ffmpeg] = {"version": full, "verified": number == MEASURED_VERSION,
                         "pix_fmts": pix_fmts_of(caps["-pix_fmts"]), "filters": filters_of(caps["-filters"])}
    return _CAPS[ffmpeg]


# ----------------------------------------------------------------------------
# the graph
# ----------------------------------------------------------------------------

def split_graph(graph):
    """the filters of a graph, link labels removed: split on , and ; outside quotes and escapes."""
    parts, word, quote, escape = [], [], False, False
    for ch in re.sub(r"\[[^\]]*\]", " ", graph):
        if escape:
            word.append(ch)
            escape = False
        elif ch == "\\":
            word.append(ch)
            escape = True
        elif ch == "'":
            word.append(ch)
            quote = not quote
        elif ch in ",;" and not quote:
            parts.append("".join(word).strip())
            word = []
        else:
            word.append(ch)
    parts.append("".join(word).strip())
    return [p for p in parts if p]


def filter_names(graph):
    """the filter name of every filter in a graph (an @instance suffix dropped)."""
    return [re.split(r"[=@]", part, maxsplit=1)[0].strip() for part in split_graph(graph)]


def unsafe_filters(graph):
    """the filters of a graph outside SAFE_FILTERS, each once in the order they appear: what needs
    unsafe=1."""
    found = []
    for name in filter_names(graph):
        if name and name not in SAFE_FILTERS and name not in found:
            found.append(name)
    return found


def matrix_for(width, height):
    """the colour matrix rule of tools/apply.py for an untagged frame: HD is bt709, smaller smpte170m."""
    return "bt709" if max(width, height) >= 1280 or height >= 720 else "smpte170m"


def full_graph(graph, width, height):
    """the user's graph behind a setparams that tags the RGB frames with the matrix of the rule, so a
    conversion to YUV and back inside the graph uses it (measured with ffmpeg 9.0.2: red goes to
    luma 63 of 255 under bt709 and 81 under smpte170m, tv range)."""
    m = matrix_for(width, height)
    return f"setparams=colorspace={m}:color_primaries={m}:color_trc={m}," + graph


def check_graph(graph, unsafe, caps):
    if not graph.strip():
        raise StageError("usage", "graph is empty: pass --param graph=GRAPH")
    missing = [name for name in filter_names(graph) if name and name not in caps["filters"]]
    if missing:
        raise StageError("unsupported", "this ffmpeg has no filter " + ", ".join(f"`{n}`" for n in missing))
    outside = unsafe_filters(graph)
    if outside and not unsafe:
        raise StageError("unsupported", "the graph uses filters outside the safe list (" + ", ".join(outside) + "); "
                                        "pass --param unsafe=1 to allow it")


# ----------------------------------------------------------------------------
# the dry run: what a graph does to the geometry
# ----------------------------------------------------------------------------

def dry_count(fps, frames):
    """frames fed to the dry run: at most DRY_FRAMES and DRY_SECONDS of video, and at most the input."""
    n = max(1, min(DRY_FRAMES, int(DRY_SECONDS * fps)))
    return n if frames is None else max(1, min(n, frames))


def parse_framemd5(text):
    """(time base, (width, height) or None, [pts of every frame], [duration of every frame])."""
    tb = re.search(r"(?m)^#tb 0: (\d+)/(\d+)", text)
    dims = re.search(r"(?m)^#dimensions 0: (\d+)x(\d+)", text)
    pts, durations = [], []
    for line in text.splitlines():
        if line and not line.startswith("#"):
            fields = [f.strip() for f in line.split(",")]
            if len(fields) >= 6:
                pts.append(int(fields[2]))
                durations.append(int(fields[3]))
    return (Fraction(int(tb.group(1)), int(tb.group(2))) if tb else None,
            (int(dims.group(1)), int(dims.group(2))) if dims else None, pts, durations)


def measure(ffmpeg, graph, width, height, depth, fps, frames):
    """run the graph over a bounded synthetic clip of the input geometry and read what comes out:
    {width, height, fps (Fraction or None when the rate varies), fed, out}."""
    fed = dry_count(fps, frames)
    fmt = PIX_FMT[depth]
    source = f"color=c=gray:s={width}x{height}:r={fps.numerator}/{fps.denominator},format={fmt},trim=end_frame={fed}"
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-f", "lavfi", "-i", source, "-vf", full_graph(graph, width, height),
           "-fps_mode", "passthrough", "-pix_fmt", fmt, "-f", "framemd5", "-"]
    code, out, err = capture(cmd)
    if code != 0:
        raise StageError("unsupported", "ffmpeg cannot run the graph: " + (err[-1] if err else f"exit code {code}"),
                         err[-TAIL:], code)
    tb, dims, pts, durations = parse_framemd5(out)
    rate = None
    if tb and len(pts) >= 2:
        steps = {b - a for a, b in zip(pts, pts[1:])}
        if len(steps) == 1 and min(steps) > 0:
            rate = 1 / (tb * steps.pop())
    elif tb and durations and durations[0] > 0:
        rate = 1 / (tb * durations[0])
    elif tb and not pts:
        rate = fps
    size = dims or (width, height)
    return {"width": size[0], "height": size[1], "fps": rate, "fed": fed, "out": len(pts)}


def geometry(ffmpeg, graph, expect, width, height, depth, fps, frames):
    """the output geometry of the stage for this input, or a StageError: preserve refuses any change,
    declared refuses a rate change. frames is None when the input count is unknown, or when the dry
    run did not see the whole input and the graph changed the count."""
    seen = measure(ffmpeg, graph, width, height, depth, fps, frames)
    if seen["fps"] != fps:
        got = f"{seen['fps'].numerator}/{seen['fps'].denominator}" if seen["fps"] else "a variable rate"
        raise StageError("unsupported", f"the graph changes the frame rate from {fps.numerator}/{fps.denominator} "
                                        f"to {got}; a frame rate change is not supported")
    if frames is None:
        count = None
    elif frames <= seen["fed"]:
        count = seen["out"] if frames else 0
    else:
        count = frames if seen["out"] == seen["fed"] else None
    changed = (seen["width"], seen["height"]) != (width, height) or seen["out"] != seen["fed"]
    if expect == "preserve" and changed:
        raise StageError("unsupported", f"expect=preserve, but the graph turns {seen['fed']} frames of {width}x{height} "
                                        f"into {seen['out']} of {seen['width']}x{seen['height']}; pass "
                                        "--param expect=declared")
    return {"width": seen["width"], "height": seen["height"], "depth": depth, "fps": fps, "frames": count}


# ----------------------------------------------------------------------------
# the command line
# ----------------------------------------------------------------------------

def fps_arg(text):
    match = re.fullmatch(r"([1-9][0-9]*)(?:/([1-9][0-9]*))?", text)
    if not match:
        raise argparse.ArgumentTypeError("expected an exact rate N/D such as 30000/1001 (never 29.97)")
    return Fraction(int(match.group(1)), int(match.group(2) or 1))


def positive(text):
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError("expected a whole number of 1 or more")
    return value


def count_arg(text):
    try:
        value = int(text)
    except ValueError:
        value = -1
    if value < 0:
        raise argparse.ArgumentTypeError("expected a whole number of frames, 0 or more")
    return value


def param_arg(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError("expected K=V")
    return tuple(text.split("=", 1))


def manifest(ffmpeg):
    """the stage manifest of `list --json` (contracts/stage-manifest.schema.json)."""
    caps = capabilities(ffmpeg)
    depths = [d for d, fmt in sorted(PIX_FMT.items()) if caps["pix_fmts"].get(fmt) == (True, True)]
    label = caps["version"] + ("" if caps["verified"] else f", unverified (measured: {MEASURED_VERSION})")
    return {
        "contract": CONTRACT,
        "provider": {"id": PROVIDER_ID, "version": f"{VERSION}, ffmpeg {label}",
                     "command": [sys.executable, os.path.abspath(__file__)],
                     "ffmpeg": {"version": caps["version"], "verified": caps["verified"]}},
        "stages": [{
            "id": STAGE_ID, "kind": "frame", "spool": False, "geometry": "declared", "depths": depths,
            "params": [{"id": "graph", "kind": "text"},
                       {"id": "expect", "kind": "popup", "choices": list(EXPECT), "default": "preserve"},
                       {"id": "unsafe", "kind": "checkbox", "default": 0}],
            "layers": [],
        }],
    }


def build_parser():
    parser = ArgParser(prog="ffstage.py", description="the stage ffmpeg-filter: one ffmpeg filter graph over raw frames")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=ArgParser)
    listing = sub.add_parser("list", help="print the stage manifest")
    listing.add_argument("--json", action="store_true", required=True, help="the manifest as json (the only form)")
    apply = sub.add_parser("apply", help="run the stage: frames on stdin, frames on stdout")
    apply.add_argument("stage", help=f"the stage id, {STAGE_ID}")
    apply.add_argument("--width", type=positive, required=True)
    apply.add_argument("--height", type=positive, required=True)
    apply.add_argument("--depth", type=int, choices=sorted(PIX_FMT), required=True)
    apply.add_argument("--fps", type=fps_arg, required=True, help="exact rate, N/D")
    apply.add_argument("--frames", type=count_arg, help="frames of the input, when known")
    apply.add_argument("--param", type=param_arg, action="append", default=[], metavar="K=V",
                       help="graph=G (required), expect=preserve|declared, unsafe=0|1")
    apply.add_argument("--describe", action="store_true",
                       help="print the output geometry (contracts/describe.schema.json), read no frames")
    return parser


def read_params(pairs):
    params = {}
    for key, value in pairs:
        if key not in ("graph", "expect", "unsafe"):
            raise StageError("usage", f"unknown param `{key}`: {STAGE_ID} takes graph, expect and unsafe")
        if key in params:
            raise StageError("usage", f"param `{key}` given more than once")
        params[key] = value
    if "graph" not in params:
        raise StageError("usage", "pass the filter graph as --param graph=GRAPH")
    expect = params.get("expect", "preserve")
    if expect not in EXPECT:
        raise StageError("unsupported", f"expect must be preserve or declared, not `{expect}`")
    unsafe = params.get("unsafe", "0")
    if unsafe not in ("0", "1"):
        raise StageError("unsupported", f"unsafe must be 0 or 1, not `{unsafe}`")
    return params["graph"], expect, unsafe == "1"


# ----------------------------------------------------------------------------
# the run
# ----------------------------------------------------------------------------

def relay(source, sink, counter, key, on_error=None):
    """copy source to sink in chunks, counting bytes in counter[key]; close the sink at the end."""
    try:
        while True:
            chunk = source.read(RELAY_CHUNK)
            if not chunk:
                break
            counter[key] += len(chunk)
            try:
                sink.write(chunk)
                sink.flush()
            except (OSError, ValueError):
                counter[key + "_broken"] = True
                if on_error:
                    on_error()
                break
    except (OSError, ValueError):
        pass
    finally:
        try:
            sink.close()
        except (OSError, ValueError):
            pass


def frames_text(count):
    return f"{count} frame" + ("" if count == 1 else "s")


def run_graph(ffmpeg, args, graph, expect, out_geo, stdin, stdout):
    """feed stdin through ffmpeg into stdout and check the counts afterwards."""
    fmt = PIX_FMT[args.depth]
    in_fb = args.width * args.height * BYTES_PER_PIXEL[args.depth]
    out_fb = out_geo["width"] * out_geo["height"] * BYTES_PER_PIXEL[args.depth]
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", fmt,
           "-video_size", f"{args.width}x{args.height}",
           "-framerate", f"{args.fps.numerator}/{args.fps.denominator}", "-i", "-",
           "-vf", full_graph(graph, args.width, args.height), "-fps_mode", "passthrough",
           "-pix_fmt", fmt, "-f", "rawvideo", "-"]
    try:
        proc = subprocess.Popen(cmd, shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, bufsize=0)
    except OSError as exc:
        raise StageError("stage_failed", f"cannot start ffmpeg: {exc}")
    counts = {"in": 0, "out": 0}
    errors = []
    feeder = threading.Thread(target=relay, args=(stdin, proc.stdin, counts, "in"), daemon=True)
    reader = threading.Thread(target=relay, args=(proc.stdout, stdout, counts, "out", lambda: kill_tree(proc)),
                              daemon=True)
    drain = threading.Thread(target=lambda: errors.extend(proc.stderr.read().decode("utf-8", "replace")
                                                          .strip().splitlines()), daemon=True)
    for thread in (feeder, reader, drain):
        thread.start()
    try:
        reader.join()
        code = proc.wait()
        drain.join(PROBE_TIMEOUT)
        if counts.get("out_broken"):
            raise StageError("stage_failed", "the next process stopped reading the output")
        if code != 0:
            raise StageError("stage_failed", "ffmpeg exited with code " + str(code)
                             + (": " + errors[-1] if errors else ""), errors[-TAIL:], code)
        feeder.join(PROBE_TIMEOUT)
    except BaseException:
        kill_tree(proc)
        raise
    whole, cut = divmod(counts["in"], in_fb)
    if cut:
        raise StageError("bad_input", f"the input ended inside a frame ({cut} bytes after {frames_text(whole)} "
                                      f"of {in_fb} bytes)")
    if args.frames is not None and whole != args.frames:
        raise StageError("bad_input", f"the input has {frames_text(whole)}, --frames said {args.frames}")
    made, rest = divmod(counts["out"], out_fb)
    if rest:
        raise StageError("stage_failed", f"ffmpeg ended in the middle of a frame ({rest} bytes after {frames_text(made)})")
    # preserve: bytes out equal bytes in; declared: the count of the answer, when it is known
    owed = whole if expect == "preserve" else out_geo["frames"]
    if owed is not None and made != owed:
        raise StageError("stage_failed", f"the graph wrote {frames_text(made)} for {frames_text(whole)}, "
                                         f"expected {owed}")


def error_document(exc):
    return {"ok": False, "contract": CONTRACT,
            "error": {"code": exc.kind, "stage": STAGE_ID, "exit_code": exc.exit_code,
                      "message": exc.message.replace("\r", " ").replace("\n", " "), "stderr_tail": exc.tail}}


def main(argv=None, stdin=None, stdout=None):
    stdin = stdin or sys.stdin.buffer
    stdout = stdout or sys.stdout.buffer
    describing = "--describe" in (sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(argv)
        if args.command == "list":
            stdout.write((json.dumps(manifest(ffmpeg_path()), indent=2) + "\n").encode("utf-8"))
            stdout.flush()
            return 0
        describing = args.describe
        if args.stage != STAGE_ID:
            raise StageError("unsupported", f"unknown stage `{args.stage}`: this provider has {STAGE_ID}")
        graph, expect, unsafe = read_params(args.param)
        ffmpeg = ffmpeg_path()
        caps = capabilities(ffmpeg)
        if not caps["verified"]:
            sys.stderr.write(f"ffstage: ffmpeg {caps['version']} is not the measured version {MEASURED_VERSION}; "
                             "results are unverified\n")
        if caps["pix_fmts"].get(PIX_FMT[args.depth]) != (True, True):
            raise StageError("unsupported", f"this ffmpeg cannot read and write {PIX_FMT[args.depth]}")
        check_graph(graph, unsafe, caps)
        geo = geometry(ffmpeg, graph, expect, args.width, args.height, args.depth, args.fps, args.frames)
        if describing:
            answer = {"ok": True, "contract": CONTRACT, "stage": STAGE_ID, "width": geo["width"],
                      "height": geo["height"], "depth": geo["depth"],
                      "fps": f"{geo['fps'].numerator}/{geo['fps'].denominator}", "frames": geo["frames"]}
            stdout.write((json.dumps(answer) + "\n").encode("utf-8"))
            stdout.flush()
            return 0
        run_graph(ffmpeg, args, graph, expect, geo, stdin, stdout)
        return 0
    except StageError as exc:
        if describing:
            try:
                stdout.write((json.dumps(error_document(exc)) + "\n").encode("utf-8"))
                stdout.flush()
            except (OSError, ValueError):
                pass
        sys.stderr.write(f"error: {exc.kind}: {exc.message}\n")
        for line in exc.tail:
            sys.stderr.write(f"  {line}\n")
        return EXIT[exc.kind]
    except KeyboardInterrupt:
        sys.stderr.write("error: interrupted\n")
        return 130


if __name__ == "__main__":
    sys.exit(main())
