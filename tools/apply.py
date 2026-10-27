#!/usr/bin/env python3
"""apply.py: run one of the effects over a video, through ffmpeg and vfx.

ffmpeg decodes the input to raw frames, `vfx apply` runs the effect on them, and a
second ffmpeg encodes the result with libx264 (CRF 18, yuv420p) while copying the
source audio untouched. the output is written to a partial file, probed against the
input, and only then moved onto the real path.

  python tools/apply.py clip.mp4 retime
  python tools/apply.py clip.mp4 retime --param speed_percent=40 --depth 32 -o out.mp4
  python tools/apply.py clip.mp4 retime --dry-run
  python tools/apply.py clip.mp4 retime --param "speed_percent=100@0;50@24" --frames 90
  python tools/apply.py clip.mp4 retime-fgbg --param warping=3 --layer alt_source_b=track.mp4

an effect is a frame effect (one frame in, one out) or a temporal effect (it reads
other frames of the clip, extra layers and parameters sampled over time); `vfx list
--json` says which and lists the layers. a temporal effect needs random access to
the clip, so vfx spools the decoded frames to a temporary file (in the scratch
directory this tool deletes) and gets the exact frame rate as a fraction.

  --param K=V        a value, or for a temporal effect keyframes V@F;V@F with F the
                     frame number (a trailing h on F holds the key, a trailing e
                     eases it). quote it in a shell, the ; ends a command there.
  --layer ID=VIDEO   an extra input layer the effect lists. the video is decoded to a
                     temporary raw file in the same pixel format and size (and frame
                     rate), then removed. repeatable.
  --frames N         output length in frames for a temporal effect (default: the
                     frames of the input). the output has the input's size and frame
                     rate and lasts N / fps. the audio is copied untouched; when the
                     output is shorter than the audio it is cut at the end of the
                     video (ffmpeg -shortest), when longer the audio just ends early.
  --pre-stage CMD    a raw frame filter of another tool that runs between the decode
  --post-stage CMD   and vfx (pre) or between vfx and the encode (post). repeatable,
                     run in the order given. CMD is one command line; it reads frames
                     of the --depth format on stdin and writes the same number of
                     frames of the same format and size on stdout. {width} {height}
                     {depth} {fps} (N/D) {frames} and {pix_fmt} in it are filled in.
  --stage-timeout S  kill a stage that holds the pipe up for S seconds without moving
                     a byte (default 120, 0 turns it off).
  --contract C       frames/1 (the default, unchanged) or frames/2: the error object on
                     every --json failure, the exit codes of contracts/error.schema.json,
                     role and index for every process in the report, the decode and
                     the encode watched by --stage-timeout too, and declared geometry.
  --geometry G       after a --pre-stage or --post-stage (frames/2): preserve (the
                     default) or declared. a declared stage is asked `CMD --describe`
                     first and its answer sets the size and frame count downstream.
  --describe         (frames/2) print the output geometry of the run as a describe
                     answer and stop, reading and writing no frames.

stdlib only: probe first, act, verify, with --dry-run, --json, -o and --force on
the one entry point. every external tool runs via subprocess with list args,
shell=False. use sys.executable, never python3.
"""

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(HERE)

PROBE_TIMEOUT = 30
JSON_OUTPUT = False
FORCE = False
# true when --pre-stage or --post-stage was given: --json failures then carry the error object
# of the stage contract instead of a bare message
STAGED = False
CONTRACT = "frames/1"
STAGE_TIMEOUT = 120
# stderr lines of a stage kept in a --json report; the text messages keep the last three
STAGE_TAIL = 5
# the error code vfx's own exit code stands for
VFX_EXIT_KIND = {2: "usage", 3: "unsupported", 4: "bad_input"}
# under frames/2 the same three exits hold for every process stage (contracts/error.schema.json, x-stage-exits)
STAGE_EXIT_KIND = VFX_EXIT_KIND
CONTRACTS = ("frames/1", "frames/2")
# the exit status of apply.py for each error code under frames/2 (contracts/error.schema.json, x-codes);
# frames/1 keeps 0, 1, 2 (argparse) and 130
EXIT_CODE = {"stage_failed": 1, "usage": 2, "unsupported": 3, "bad_input": 4, "broken_pipe": 4,
             "partial_frame": 4, "stage_timeout": 5, "tool_missing": 6, "interrupted": 130}
# the geometry values a stage can have; any other value is refused (unsupported)
GEOMETRY_VALUES = ("preserve", "declared")
# the role of a pipe process in a frames/2 report, where it differs from the internal one
REPORT_ROLE = {"vfx": "effect"}

# the raw frame format vfx reads and writes at each depth
PIX_FMT = {8: "rgba", 16: "rgba64le", 32: "gbrapf32le"}
# the -pix_fmt asked of ffv1 at each depth; at 16 ffmpeg stores gbrap16le, the
# nearest format ffv1 has. float (32) has no ffv1 pixel format
LOSSLESS_PIX_FMT = {8: "bgra", 16: "rgba64le"}

# ffprobe color_space names, as swscale's out_color_matrix spells them
MATRIX = {"bt709": "bt709", "bt470bg": "bt601", "smpte170m": "smpte170m",
          "bt2020nc": "bt2020", "smpte240m": "smpte240m", "fcc": "fcc"}


def eprint(*args):
    print(*args, file=sys.stderr)


# a pipe gets the ansi code page, and a path it cannot spell would crash the
# report line after the work is done; stderr already escapes those characters
try:
    sys.stdout.reconfigure(errors="backslashreplace")
except (AttributeError, ValueError):
    pass


def fail(message, code=1, kind="usage", stage=None, exit_code=None, tail=None, summary=None):
    """print the failure and exit. `kind` is the error code of the stage contract (usage,
    unsupported, bad_input, stage_failed, stage_timeout, broken_pipe, partial_frame,
    tool_missing); it is only reported in --json mode of a run with stages, together with the
    stage (decode, pre1, pre2, ..., vfx, post1, ..., encode, layer-decode or verify), its exit
    code and the end of its stderr. `summary` is the one line message for a `message` that
    spans several lines. under --contract frames/2 every --json failure carries the error object,
    the text line is `error: <kind>: <message>` and the exit status follows from `kind` (EXIT_CODE)."""
    framed = CONTRACT == "frames/2"
    if JSON_OUTPUT:
        if STAGED or framed:
            # frames/2 keeps the message to one line, as the error schema asks, even without a summary
            line = summary or (message.splitlines()[0] if framed else message)
            error = {"code": kind, "stage": stage, "exit_code": exit_code,
                     "message": line, "stderr_tail": list(tail or [])}
            print(json.dumps({"ok": False, "contract": CONTRACT, "error": error}))
        else:
            print(json.dumps({"ok": False, "error": message}))
    if framed:
        eprint(f"error: {kind}: {message}")
        sys.exit(EXIT_CODE[kind])
    eprint(f"[error] {message}")
    sys.exit(code)


# ----------------------------------------------------------------------------
# tool resolution
# ----------------------------------------------------------------------------

def _resolve_tool(name, env_names):
    """an explicit override, then PATH, then the places a fresh windows install lands."""
    for var in env_names:
        override = os.environ.get(var)
        if override:
            resolved = shutil.which(override)
            if not resolved:
                fail(f"explicit {name} executable not found: {override}", kind="tool_missing")
            return resolved
    exe = shutil.which(name)
    if exe:
        return exe
    candidates = []
    localappdata = os.environ.get("LOCALAPPDATA")
    if localappdata:
        candidates.append(os.path.join(localappdata, "Microsoft", "WinGet", "Links", name + ".exe"))
    for var in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(var)
        if root:
            candidates.append(os.path.join(root, "ffmpeg", "bin", name + ".exe"))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


_TOOLS = {}


def ffmpeg_path():
    if "ffmpeg" not in _TOOLS:
        _TOOLS["ffmpeg"] = _resolve_tool("ffmpeg", ("MEDIA_FFMPEG_PATH", "FFMPEG"))
    return _TOOLS["ffmpeg"]


def ffprobe_path():
    if "ffprobe" not in _TOOLS:
        _TOOLS["ffprobe"] = _resolve_tool("ffprobe", ("MEDIA_FFPROBE_PATH", "FFPROBE"))
    return _TOOLS["ffprobe"]


def require_ffmpeg():
    if not ffmpeg_path() or not ffprobe_path():
        fail("this operation requires ffmpeg and ffprobe.\n"
             "  winget install Gyan.FFmpeg        (windows)\n"
             "  brew install ffmpeg               (macos)\n"
             "  sudo apt install ffmpeg           (debian/ubuntu)", kind="tool_missing")


def exe_name():
    return "vfx.exe" if os.name == "nt" else "vfx"


def find_vfx(explicit):
    """--vfx, then VFX_PATH, then the workspace release build, then PATH."""
    if explicit:
        if not os.path.isfile(explicit):
            fail(f"vfx not found at {explicit}", kind="tool_missing", stage="vfx")
        return os.path.abspath(explicit)
    env = os.environ.get("VFX_PATH")
    if env:
        if not os.path.isfile(env):
            fail(f"VFX_PATH points at a missing file: {env}", kind="tool_missing", stage="vfx")
        return os.path.abspath(env)
    target = os.environ.get("CARGO_TARGET_DIR")
    roots = ([target] if target else []) + [os.path.join(WORKSPACE, "target")]
    for root in roots:
        candidate = os.path.join(root, "release", exe_name())
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    found = shutil.which("vfx")
    if found:
        return found
    fail("vfx not found. build it with `cargo build --release -p fx-cli`, "
         "or pass --vfx PATH / set VFX_PATH", kind="tool_missing", stage="vfx")


# ----------------------------------------------------------------------------
# probe: what a file really is
# ----------------------------------------------------------------------------

def run(cmd, **kwargs):
    """run a command with list args, never a shell."""
    if kwargs.get("text") and cmd and cmd[0] in (_TOOLS.get("ffmpeg"), _TOOLS.get("ffprobe")):
        # ffmpeg and ffprobe write utf-8 into a pipe whatever the code page
        kwargs.setdefault("encoding", "utf-8")
        kwargs.setdefault("errors", "replace")
    try:
        return subprocess.run(cmd, shell=False, **kwargs)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, stdout="", stderr="process timed out")
    except OSError:
        return subprocess.CompletedProcess(cmd, 127, stdout="", stderr="executable unavailable")


def _fraction(value):
    """'30000/1001' -> Fraction, tolerating '0/0' and junk."""
    try:
        f = Fraction(str(value))
    except (ValueError, ZeroDivisionError, TypeError):
        return None
    return f if f > 0 else None


def _count(value):
    """a whole number from ffprobe's text, None for 'N/A' and junk."""
    try:
        n = int(str(value))
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _seconds(value):
    try:
        t = float(value)
    except (TypeError, ValueError):
        return None
    return t if t >= 0 else None


def _rotation(stream):
    """display rotation in degrees (0, 90, 180, 270) from side data or the old rotate tag."""
    raw = None
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            raw = side["rotation"]
    if raw is None:
        raw = (stream.get("tags") or {}).get("rotate")
    try:
        return int(round(float(raw))) % 360 if raw is not None else 0
    except (TypeError, ValueError):
        return 0


def probe_command(path):
    """the ffprobe argv of `probe`, which is also the verify step of a frames/2 report."""
    return [ffprobe_path(), "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", path]


def probe(path):
    """one ffprobe json call. returns a facts dict, or None if it cannot be read.

    keys: has_video, has_audio, vcodec, acodec, pix_fmt, width, height, rotation,
    fps, rate (exact Fraction), vfr, duration, channels, hdr, size, container,
    nb_frames (the container's frame count, None when it does not say),
    video_duration and audio_duration (each stream's own, None when not given) and
    the color tags. ffprobe omits fields it cannot find, so every read is guarded.
    """
    cmd = probe_command(path)
    proc = run(cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT)
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        data = json.loads(proc.stdout)
    except ValueError:
        return None

    info = {
        "path": os.path.abspath(path),
        "has_video": False, "has_audio": False,
        "vcodec": None, "acodec": None, "pix_fmt": None, "video_index": None,
        "width": None, "height": None, "rotation": 0,
        "fps": None, "rate": None, "vfr": False, "duration": None,
        "channels": None, "hdr": False, "size": None, "container": None,
        "nb_frames": None, "video_duration": None, "audio_duration": None,
        "color_space": None, "color_primaries": None, "color_transfer": None,
    }
    for st in data.get("streams", []):
        ctype = st.get("codec_type")
        if ctype == "video" and not info["has_video"]:
            # cover art is a video stream with a single frame
            if (st.get("disposition") or {}).get("attached_pic"):
                continue
            info["has_video"] = True
            info["video_index"] = st.get("index")
            info["vcodec"] = st.get("codec_name")
            info["pix_fmt"] = st.get("pix_fmt")
            info["width"] = st.get("width")
            info["height"] = st.get("height")
            info["rotation"] = _rotation(st)
            avg, base = _fraction(st.get("avg_frame_rate")), _fraction(st.get("r_frame_rate"))
            # a base rate that disagrees with the average means variable frame rate
            info["vfr"] = bool(avg and base and abs(avg - base) / base > Fraction(1, 1000))
            info["rate"] = avg if info["vfr"] else (base or avg)
            info["fps"] = round(float(info["rate"]), 3) if info["rate"] else None
            for key in ("color_space", "color_primaries", "color_transfer"):
                value = st.get(key)
                info[key] = value if value not in (None, "unknown", "unspecified") else None
            info["hdr"] = (info["color_transfer"] or "").lower() in ("smpte2084", "arib-std-b67")
            info["nb_frames"] = _count(st.get("nb_frames"))
            info["video_duration"] = _seconds(st.get("duration"))
        elif ctype == "audio" and not info["has_audio"]:
            info["has_audio"] = True
            info["acodec"] = st.get("codec_name")
            info["channels"] = st.get("channels")
            info["audio_duration"] = _seconds(st.get("duration"))

    fmt = data.get("format") or {}
    info["container"] = fmt.get("format_name")
    for key, src, cast in (("duration", "duration", float), ("size", "size", lambda v: int(float(v)))):
        raw = fmt.get(src)
        if raw is not None:
            try:
                info[key] = cast(raw)
            except (ValueError, TypeError):
                pass
    return info


def display_size(info):
    """the frame size ffmpeg decodes to: a 90 or 270 degree rotation swaps the sides."""
    w, h = info["width"], info["height"]
    return (h, w) if info["rotation"] in (90, 270) else (w, h)


def describe(info):
    if not info:
        return "unreadable"
    bits = []
    if info.get("duration") is not None:
        bits.append(f"{info['duration']:.2f}s")
    if info.get("width") and info.get("height"):
        w, h = display_size(info)
        bits.append(f"{w}x{h}")
    if info.get("fps"):
        bits.append(f"{info['fps']:g} fps")
    if info.get("vcodec"):
        bits.append(info["vcodec"])
    if info.get("acodec"):
        bits.append(f"{info['acodec']} {info['channels']}ch" if info.get("channels") else info["acodec"])
    return ", ".join(bits) or "no streams"


def public(info):
    """the probe without the exact Fraction, which json cannot hold."""
    if info is None:
        return None
    out = dict(info)
    out["rate"] = str(info["rate"]) if info.get("rate") else None
    return out


def require_input(path):
    if not os.path.isfile(path):
        fail(f"no such file: {path}", kind="bad_input")
    info = probe(path)
    if info is None:
        fail(f"cannot probe {path}; is it a media file?", kind="bad_input")
    return info


# ----------------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------------

def unique_path(path):
    """avoid clobbering an existing target by appending _1, _2, ..."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"


def default_output(src, operation, ext):
    base, _ = os.path.splitext(src)
    return unique_path(f"{base}_{operation}.{ext.lstrip('.')}")


def same_file(a, b):
    try:
        return os.path.samefile(a, b)
    except (OSError, TypeError, ValueError):
        return False


def partial_path(destination):
    """a sibling scratch name for an output in progress. the extension stays
    last because ffmpeg picks the container from it."""
    root, ext = os.path.splitext(destination)
    return f"{root}.partial-{os.getpid()}{ext}"


def discard(path):
    try:
        os.remove(path)
    except OSError:
        pass


def fmt_cmd(cmd):
    """render an argv list as a copy-pasteable command line."""
    return " ".join(shlex.quote(str(c)) for c in cmd)


# ----------------------------------------------------------------------------
# argparse + reporting
# ----------------------------------------------------------------------------

def param_arg(text):
    key, sep, _ = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError("expected K=V, for example speed_percent=40")
    return text


def layer_arg(text):
    """ID=VIDEO, split at the first '=' (a Windows path has none before the drive colon)."""
    layer_id, sep, video = text.partition("=")
    if not sep or not layer_id or not video:
        raise argparse.ArgumentTypeError("expected ID=VIDEO, for example overlay=overlay.mp4")
    return (layer_id, video)


def frames_arg(text):
    try:
        n = int(text)
    except ValueError:
        n = 0
    if n < 1:
        raise argparse.ArgumentTypeError("expected a positive number of frames")
    return n


def split_command(text, windows=None):
    """a stage command line as an argv list: shlex, with posix=False on windows so that a backslash
    stays a backslash. that mode leaves the quotes on a quoted word, so they are taken off here."""
    windows = os.name == "nt" if windows is None else windows
    words = shlex.split(text, posix=not windows)
    if windows:
        words = [w[1:-1] if len(w) >= 2 and w[0] == w[-1] and w[0] in "\"'" else w for w in words]
    return words


def stage_arg(text):
    """a stage command line. the placeholders are filled in later, when the size and rate are known."""
    try:
        words = split_command(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"cannot read the command line: {exc}")
    if not words:
        raise argparse.ArgumentTypeError("expected a command line, for example \"python filter.py {width} {height}\"")
    return text


def timeout_arg(text):
    try:
        seconds = float(text)
    except ValueError:
        seconds = -1.0
    if not 0 <= seconds < float("inf"):
        raise argparse.ArgumentTypeError("expected a number of seconds, 0 or more (0 turns the timeout off)")
    return seconds


def argv_contract(argv):
    """the --contract a command line asks for, read before argparse has run (frames/1 when absent or unknown)."""
    value = None
    for i, word in enumerate(argv):
        if word == "--contract" and i + 1 < len(argv):
            value = argv[i + 1]
        elif word.startswith("--contract="):
            value = word.split("=", 1)[1]
    return value if value in CONTRACTS else "frames/1"


class ArgParser(argparse.ArgumentParser):
    """argparse, except that a usage error in --json mode of a run with stages, or of any run under
    --contract frames/2, also prints the error object. the exit status stays 2 in both contracts.
    the stage and frames/2 flags match only in full, so an abbreviation means the flag it meant
    before they were added, and an abbreviation of one of them is refused as it was then."""

    FULL_ONLY = frozenset({"--pre-stage", "--post-stage", "--stage-timeout", "--geometry", "--contract",
                           "--describe"})

    def _get_option_tuples(self, option_string):
        return [match for match in super()._get_option_tuples(option_string) if match[1] not in self.FULL_ONLY]

    def error(self, message):
        flags = {a.split("=", 1)[0] for a in sys.argv[1:]}
        contract = argv_contract(sys.argv[1:])
        if "--json" in flags and (flags & {"--pre-stage", "--post-stage"} or contract == "frames/2"):
            error = {"code": "usage", "stage": None, "exit_code": None, "message": message, "stderr_tail": []}
            print(json.dumps({"ok": False, "contract": contract, "error": error}))
        super().error(message)


class StageCommand(str):
    """a --pre-stage or --post-stage command line with the --geometry value given after it."""

    def __new__(cls, text, geometry):
        obj = super().__new__(cls, text)
        obj.geometry = geometry
        return obj


class StageAction(argparse.Action):
    """append a stage command line and remember which list took it, for a --geometry that follows."""

    def __call__(self, parser, namespace, values, option_string=None):
        items = list(getattr(namespace, self.dest, None) or [])
        items.append(values)
        setattr(namespace, self.dest, items)
        namespace.last_stage = self.dest


class GeometryAction(argparse.Action):
    """--geometry VALUE: the geometry of the --pre-stage or --post-stage given just before it. the value
    is checked later, so that an unknown one is refused as unsupported rather than as a usage error."""

    def __call__(self, parser, namespace, values, option_string=None):
        dest = getattr(namespace, "last_stage", None)
        if dest is None:
            parser.error("--geometry describes the --pre-stage or --post-stage given just before it")
        items = getattr(namespace, dest)
        items[-1] = StageCommand(items[-1], values)
        namespace.geometry_given = True


def add_common_args(parser):
    """the shared flags: output, overwrite, dry run, json and the ffmpeg paths."""
    parser.add_argument("-o", "--output", metavar="PATH",
                        help="output path (default: <input>_<effect>.mp4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and commands, write nothing")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="print a structured result on stdout")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing -o path once the new output "
                             "verifies (the input is never overwritten)")
    parser.add_argument("--ffmpeg-path", help="ffmpeg executable (or MEDIA_FFMPEG_PATH, FFMPEG)")
    parser.add_argument("--ffprobe-path", help="ffprobe executable (or MEDIA_FFPROBE_PATH, FFPROBE)")
    return parser


def configure(args):
    global JSON_OUTPUT, FORCE, STAGED, CONTRACT
    CONTRACT = getattr(args, "contract", None) or "frames/1"
    for name in ("ffmpeg", "ffprobe"):
        value = getattr(args, name + "_path", None)
        if value:
            os.environ["MEDIA_" + name.upper() + "_PATH"] = value
    _TOOLS.clear()
    JSON_OUTPUT = bool(args.as_json)
    FORCE = bool(args.force)
    STAGED = bool(getattr(args, "pre_stage", None) or getattr(args, "post_stage", None))
    output = args.output
    if output and os.path.lexists(output):
        if same_file(output, args.input):
            fail("the output path is the input file; choose a different --output path")
        if not FORCE:
            fail("output already exists; choose a new --output path or pass --force")


def verified_output(path, plan_only=False):
    """probe a written output so the result reports what is really there."""
    if plan_only:
        return {"path": os.path.abspath(path), "probe": None}
    info = probe(path)
    if info is None:
        fail("output verification failed", kind="stage_failed", stage="verify")
    return {"path": os.path.abspath(path), "probe": public(info)}


def emit(result, as_json):
    """final report: json on stdout, or the plan and a line per output with its facts.
    an output path is the last thing printed in human mode."""
    if as_json:
        print(json.dumps(result, indent=2))
        return
    if result.get("summary"):
        print(f"[plan] {result['summary']}")
    for temp in result.get("temp_files") or []:
        print(f"[temp] {temp['path']} ({temp['what']})")
    for cmd in result.get("commands") or []:
        print(f"[cmd] {cmd}")
    if result.get("dry_run"):
        print("[dry-run] nothing was written")
        return
    for out in result.get("outputs") or []:
        print(f"{out['path']}: {describe(out.get('probe'))}")


# ----------------------------------------------------------------------------
# the pipeline
# ----------------------------------------------------------------------------

def color_settings(info):
    """the matrix to decode and encode with, and the tags to write.

    a tagged source keeps its own. an untagged one follows the size convention
    players use, HD is bt709 and anything smaller bt601, and is tagged to match.
    returns (swscale matrix, colorspace tag, primaries tag, transfer tag), where
    a tag the source never had stays None."""
    space = info["color_space"]
    primaries, trc = info["color_primaries"], info["color_transfer"]
    if space not in MATRIX:
        w, h = display_size(info)
        space = "bt709" if max(w, h) >= 1280 or h >= 720 else "smpte170m"
        primaries = trc = space
    return MATRIX[space], space, primaries, trc


# ----------------------------------------------------------------------------
# frame and temporal effects
# ----------------------------------------------------------------------------

def effect_info(vfx, effect_id):
    """kind and layers of an effect, from `vfx list --json --all` (the hidden self-test
    effects included). a vfx that predates kinds gives frame effects without layers."""
    proc = None
    for extra in (["--all"], []):
        proc = run([vfx, "list", "--json"] + extra, stdin=subprocess.DEVNULL,
                   capture_output=True, text=True, timeout=PROBE_TIMEOUT)
        if proc.returncode == 127:
            fail(f"cannot run vfx at {vfx}", kind="tool_missing", stage="vfx")
        if proc.returncode == 0:
            break
    else:
        lines = (proc.stderr or "").strip().splitlines()
        fail("`vfx list --json` failed: " + (lines[-1] if lines else f"exit code {proc.returncode}"),
             kind="stage_failed", stage="vfx", exit_code=proc.returncode, tail=lines[-STAGE_TAIL:])
    try:
        listing = json.loads(proc.stdout)
    except ValueError:
        fail("`vfx list --json` did not print JSON", kind="stage_failed", stage="vfx", exit_code=proc.returncode)
    for entry in listing:
        if entry.get("id") == effect_id:
            return {"id": effect_id, "kind": entry.get("kind") or "frame",
                    "layers": list(entry.get("layers") or [])}
    fail(f"unknown effect `{effect_id}`, `vfx list` shows the registered ones", kind="unsupported", stage="vfx")


def check_layer_options(args, einfo):
    """--layer and --frames against the kind of the effect, with a clear message when they do not fit."""
    if einfo["kind"] != "temporal":
        used = [flag for flag, given in (("--layer", args.layer), ("--frames", args.frames)) if given]
        if used:
            fail(f"{args.effect} is a frame effect, {' and '.join(used)} only apply to temporal effects")
        return
    ids = [layer["id"] for layer in einfo["layers"]]
    seen = set()
    for layer_id, _ in args.layer:
        if layer_id not in ids:
            have = "the layers " + ", ".join(ids) if ids else "no layers"
            fail(f"unknown layer `{layer_id}`: {args.effect} has {have}")
        if layer_id in seen:
            fail(f"layer `{layer_id}` given more than once")
        seen.add(layer_id)
    for layer in einfo["layers"]:
        if not layer.get("optional") and layer["id"] not in seen:
            fail(f"{args.effect} needs the layer `{layer['id']}`, pass --layer {layer['id']}=VIDEO")


def probe_layers(args):
    """probe every --layer video up front, so a missing file or a file without picture fails before any work."""
    layers = []
    for layer_id, path in args.layer:
        li = require_input(path)
        if not li["has_video"]:
            fail(f"layer {layer_id}: {path} has no video stream", kind="bad_input")
        if not li["rate"] or not li["width"] or not li["height"]:
            fail(f"layer {layer_id}: cannot read the frame size and frame rate of {path}", kind="bad_input")
        if li["hdr"]:
            fail(f"layer {layer_id}: HDR video is not supported; tone mapping is not part of this tool",
                 kind="unsupported")
        layers.append({"id": layer_id, "video": os.path.abspath(path), "info": li})
    return layers


def exact_frames(info):
    """the frames the input decodes to when the container says so for a constant frame rate, else None.
    a variable frame rate input is resampled while decoding, so its count is only an estimate."""
    if info["nb_frames"] and not info["vfr"]:
        return info["nb_frames"]
    return None


def audio_plan(info, frames, rate):
    """what the output does to the audio when it has `frames` frames, judged by lengths alone:
    'shortest' cuts the copy at the end of the video, 'ends-early' leaves a shorter track,
    'same' keeps it as it is. None when there is no audio."""
    if not info["has_audio"]:
        return None
    audio = info["audio_duration"] or info["duration"]
    if audio is None:
        return "same"
    video, slack = frames / float(rate), 1.0 / float(rate)
    if video + slack < audio:
        return "shortest"
    if video > audio + slack:
        return "ends-early"
    return "same"


def make_plan(args, info, einfo, layers, scratch):
    """what the temporal path adds to the frame path: frames, layer temp files and the audio decision.
    `scratch` only names paths here, nothing is created."""
    plan = {"kind": einfo["kind"], "scratch": scratch, "layers": [], "frames": None,
            "input_frames": exact_frames(info), "audio": None, "shortest": False, "temp_files": []}
    if einfo["kind"] != "temporal":
        return plan
    fmt = PIX_FMT[args.depth]
    for layer in layers:
        temp = os.path.join(scratch, f"layer_{layer['id']}.{fmt}")
        plan["layers"].append({"id": layer["id"], "video": layer["video"], "temp": temp, "info": layer["info"]})
        plan["temp_files"].append({"path": temp, "what": f"layer {layer['id']}, decoded to raw {fmt}"})
    plan["temp_files"].append({"path": scratch, "what": "scratch directory: the layer clips, the logs and the "
                                                        "spool file vfx makes of the piped input"})
    # without --frames vfx renders the frames it was fed, so the output is as long as the decode
    # whatever the container said; with --frames the length is the user's
    plan["frames"] = args.frames if args.frames is not None else plan["input_frames"]
    if args.frames is not None and args.frames != plan["input_frames"]:
        plan["audio"] = audio_plan(info, args.frames, info["rate"])
        plan["shortest"] = plan["audio"] == "shortest"
    return plan


def build_layer_commands(args, info, plan, size=None):
    """one ffmpeg per layer: decode it to a raw file of the effect's pixel format, size and frame rate.
    `size` is the frame size vfx gets when a declared pre-stage changed it (frames/2)."""
    w, h = size or display_size(info)
    fmt = PIX_FMT[args.depth]
    rate = info["rate"]
    commands = []
    for layer in plan["layers"]:
        li = layer["info"]
        matrix = color_settings(li)[0]
        graph = []
        if li["vfr"] or li["rate"] != rate:
            graph.append(f"fps={rate_text(rate)}")
        if display_size(li) != (w, h):
            graph.append(f"scale={w}:{h}:in_color_matrix={matrix}")
        else:
            graph.append(f"scale=in_color_matrix={matrix}")
        commands.append([ffmpeg_path(), "-v", "error", "-nostdin", "-y", "-i", layer["video"],
                         "-map", f"0:{li['video_index']}", "-an", "-sn", "-dn",
                         "-vf", ",".join(graph), "-pix_fmt", fmt, "-f", "rawvideo", layer["temp"]])
    return commands


def decode_layers(commands, plan, args, info, scratch, size=None):
    """run the layer decodes one after the other; each leaves a raw file of whole frames or fails with ffmpeg's last lines."""
    w, h = size or display_size(info)
    frame_bytes = w * h * {8: 4, 16: 8, 32: 16}[args.depth]
    for layer, cmd in zip(plan["layers"], commands):
        log = os.path.join(scratch, f"layer_{layer['id']}.log")
        with open(log, "w", encoding="utf-8") as handle:
            proc = run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=handle)
        if proc.returncode != 0:
            lines = "\n".join("  " + t for t in tail(log))
            fail(f"layer {layer['id']}: decoding {layer['video']} failed (ffmpeg exited with code {proc.returncode})"
                 + ("\n" + lines if lines else ""), kind="stage_failed", stage="layer-decode",
                 exit_code=proc.returncode, tail=tail(log, STAGE_TAIL),
                 summary=f"layer {layer['id']}: decoding {layer['video']} failed (ffmpeg exited with code {proc.returncode})")
        try:
            size = os.path.getsize(layer["temp"])
        except OSError:
            size = 0
        if size == 0 or size % frame_bytes:
            fail(f"layer {layer['id']}: {layer['video']} decoded to {size} bytes, "
                 f"which is not a whole number of {w}x{h} frames", kind="bad_input", stage="layer-decode")


def rate_text(rate):
    return f"{rate.numerator}/{rate.denominator}"


def build_commands(args, info, vfx, destination, plan=None, sizes=None):
    """the three argv lists: decode, effect, encode.

    the decode and the encode both name the color matrix, the tagged one or the
    size convention for an untagged source, so a color that goes out as RGB comes
    back as the same YUV. ffmpeg alone would read an untagged HD file as bt601 on
    the way in and a player would read it as bt709 on the way out.

    a temporal effect (plan["kind"]) also gets its frame count, the layers' raw
    files and the scratch directory to spool the piped input in; an output that
    must be cut at the end of the video makes the audio copy stop there. `sizes` is the frame size
    vfx gets and the one the encode gets, when a declared stage changed them (frames/2)."""
    w, h = display_size(info)
    (ew, eh), (ow, oh) = sizes or ((w, h), (w, h))
    fmt = PIX_FMT[args.depth]
    rate = info["rate"]
    matrix, space, primaries, trc = color_settings(info)
    base = [ffmpeg_path(), "-v", "error", "-nostdin"]

    graph = []
    if info["vfr"]:
        # variable frame rate in, constant out: keep the duration instead of the timestamps
        graph.append(f"fps={rate_text(rate)}")
    graph.append(f"scale=in_color_matrix={matrix}")
    decode = base + ["-i", args.input, "-map", f"0:{info['video_index']}", "-an", "-sn", "-dn",
                     "-vf", ",".join(graph), "-pix_fmt", fmt, "-f", "rawvideo", "-"]

    effect = [vfx, "apply", args.effect, "--width", str(ew), "--height", str(eh),
              "--depth", str(args.depth), "--fps", rate_text(rate)]
    if plan and plan["kind"] == "temporal":
        if plan["frames"] is not None:
            effect += ["--frames", str(plan["frames"])]
        effect += ["--tmp-dir", plan["scratch"]]
        for layer in plan["layers"]:
            effect += ["--layer", f"{layer['id']}={layer['temp']}"]
    for p in args.param:
        effect += ["--param", p]

    tags = [f"colorspace={space}"]
    if primaries:
        tags.append(f"color_primaries={primaries}")
    if trc:
        tags.append(f"color_trc={trc}")
    graph = [f"scale=out_color_matrix={matrix}:out_range=tv", "setparams=" + ":".join(tags + ["range=tv"])]
    if getattr(args, "lossless", False):
        # RGB goes out as it came out of the effect: no matrix, no tv range, no chroma subsampling.
        # this is the way to carry data frames such as the vector images of create-vectors
        encode = base + ["-n", "-f", "rawvideo", "-pix_fmt", fmt, "-video_size", f"{ow}x{oh}",
                         "-framerate", rate_text(rate), "-i", "-", "-i", args.input,
                         "-map", "0:v:0", "-map", "1:a?",
                         "-c:v", "ffv1", "-level", "3", "-pix_fmt", LOSSLESS_PIX_FMT[args.depth], "-c:a", "copy"]
    else:
        encode = base + ["-n", "-f", "rawvideo", "-pix_fmt", fmt, "-video_size", f"{ow}x{oh}",
                         "-framerate", rate_text(rate), "-i", "-", "-i", args.input,
                         "-map", "0:v:0", "-map", "1:a?", "-vf", ",".join(graph),
                         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy"]
    if plan and plan["shortest"]:
        encode += ["-shortest"]
    if os.path.splitext(destination)[1].lower() in (".mp4", ".m4v", ".mov"):
        encode += ["-movflags", "+faststart"]
    encode += [partial_path(destination)]
    return decode, effect, encode


def preflight(args, info, vfx, einfo=None):
    """ask vfx to check the effect id, every param (keyframes too), the layer ids and the
    rate with `apply --check`, which reads and writes no file, before any decoding happens.
    it also proves the binary runs."""
    w, h = display_size(info)
    cmd = [vfx, "apply", args.effect, "--width", str(w), "--height", str(h),
           "--depth", str(args.depth), "--fps", rate_text(info["rate"]), "--check"]
    if einfo and einfo["kind"] == "temporal":
        for layer_id, video in args.layer:
            cmd += ["--layer", f"{layer_id}={video}"]
    for p in args.param:
        cmd += ["--param", p]
    proc = run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if proc.returncode == 127:
        fail(f"cannot run vfx at {vfx}", kind="tool_missing", stage="vfx")
    if proc.returncode != 0:
        lines = (proc.stderr or "").strip().splitlines()
        fail(lines[-1].replace("vfx: error: ", "vfx: ") if lines else f"vfx exited with code {proc.returncode}",
             kind=VFX_EXIT_KIND.get(proc.returncode, "stage_failed"), stage="vfx",
             exit_code=proc.returncode, tail=lines[-STAGE_TAIL:])


def tail(path, count=3):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return [line.rstrip() for line in f.read().strip().splitlines()[-count:]]
    except OSError:
        return []


def run_pipeline(decode, effect, encode, scratch):
    """decode | vfx | encode with every stderr in a log file, so no pipe can fill up
    and stall a stage. returns each stage's exit code and log path."""
    stages = [("decode", decode), ("vfx", effect), ("encode", encode)]
    logs = {name: os.path.join(scratch, name + ".log") for name, _ in stages}
    handles = {name: open(path, "w", encoding="utf-8") for name, path in logs.items()}
    procs = []
    try:
        p1 = subprocess.Popen(decode, shell=False, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=handles["decode"])
        procs.append(p1)
        p2 = subprocess.Popen(effect, shell=False, stdin=p1.stdout,
                              stdout=subprocess.PIPE, stderr=handles["vfx"])
        procs.append(p2)
        p1.stdout.close()
        p3 = subprocess.Popen(encode, shell=False, stdin=p2.stdout,
                              stdout=subprocess.DEVNULL, stderr=handles["encode"])
        procs.append(p3)
        p2.stdout.close()
        codes = {"encode": p3.wait(), "vfx": p2.wait(), "decode": p1.wait()}
    except BaseException:
        for p in procs:
            try:
                p.kill()
            except OSError:
                pass
        raise
    finally:
        for h in handles.values():
            h.close()
    return codes, logs


# ----------------------------------------------------------------------------
# external stages: raw frame filters of other tools, before and after vfx
# ----------------------------------------------------------------------------

PLACEHOLDER = re.compile(r"\{(width|height|depth|fps|frames|pix_fmt)\}")
# bytes moved per read, and the seconds the rest of a pipe gets to end after one process of it
# failed before the stragglers are killed
RELAY_CHUNK = 1 << 20
GRACE = 20


def frame_size(args, info):
    """bytes in one raw frame of the --depth format."""
    w, h = display_size(info)
    return w * h * {8: 4, 16: 8, 32: 16}[args.depth]


def fill_placeholders(word, values, stage, what):
    """the six placeholders of a stage command in one word. {frames} is only known for a constant
    frame rate input whose container counts its frames (and, after vfx, for --frames); anything else
    in braces is left alone, so a command can carry code or json."""
    def pick(match):
        value = values[match.group(1)]
        if value is None:
            fail(f"{stage['label']}: {{frames}} needs the number of frames {what}, which is not known for "
                 "this input (the container does not say, or the frame rate varies and ffmpeg resamples "
                 "it while decoding)", kind="usage", stage=stage["name"])
        return value
    return PLACEHOLDER.sub(pick, word)


def build_stages(args, info, chain=None):
    """the --pre-stage and --post-stage commands with their placeholders filled in: dicts with role,
    index, name (pre1, post1, ...), label (pre-stage 1, ...), argv and command, the pre stages first,
    each group in the order given. a pre stage reads the frames the input decodes to, a post stage the
    frames vfx writes (--frames for a temporal effect, else the input's count). a stage is expected
    to write as many frames as it reads. a program that cannot be found is refused here, before any work.

    with `chain` (a dict, frames/2) the geometry is chained in pipe order: every stage's placeholders
    hold the geometry it receives, a stage with --geometry declared is asked `CMD --describe` and its
    answer is what the next process gets. chain["vfx"] is then the geometry vfx reads and chain["out"]
    the one the encode reads; a stage dict also gets frame_bytes (of its output) and, when declared, answer."""
    w, h = display_size(info)
    decoded = exact_frames(info)
    geo = {"width": w, "height": h, "depth": args.depth, "fps": info["rate"], "frames": decoded}
    groups = (("pre", args.pre_stage, "the input decodes to"), ("post", args.post_stage, "the output has"))
    stages = []
    for role, templates, what in groups:
        if role == "post":
            if chain is not None:
                chain["vfx"] = dict(geo)
            if args.frames is not None:
                geo["frames"] = args.frames
        for index, template in enumerate(templates, 1):
            values = {"width": str(geo["width"]), "height": str(geo["height"]), "depth": str(args.depth),
                      "fps": rate_text(geo["fps"]), "pix_fmt": PIX_FMT[args.depth],
                      "frames": None if geo["frames"] is None else str(geo["frames"])}
            stage = {"role": role, "index": index, "name": f"{role}{index}", "label": f"{role}-stage {index}"}
            argv = [fill_placeholders(word, values, stage, what) for word in split_command(template)]
            if not shutil.which(argv[0]):
                fail(f"{stage['label']}: cannot find the program `{argv[0]}`", kind="tool_missing", stage=stage["name"])
            stage.update(argv=argv, command=fmt_cmd(argv))
            if chain is not None:
                stage["geometry"] = getattr(template, "geometry", "preserve")
                if stage["geometry"] == "declared":
                    geo = describe_stage(stage, geo)
                    stage["answer"] = dict(geo)
                stage["frame_bytes"] = geo["width"] * geo["height"] * BYTES_PER_PIXEL[args.depth]
            stages.append(stage)
    if chain is not None:
        chain.setdefault("vfx", dict(geo))
        chain["out"] = dict(geo)
    return stages


BYTES_PER_PIXEL = {8: 4, 16: 8, 32: 16}
DESCRIBE_KEYS = ("ok", "contract", "stage", "width", "height", "depth", "fps", "frames")


def describe_stage(stage, geo):
    """ask a declared stage for its output geometry: its command line plus --describe (and --frames
    when the count is known and the line has none), no stdin. the answer must follow
    contracts/describe.schema.json; only width, height and frames may change, and a stage that would
    change the frame rate or the depth is refused as unsupported. returns the geometry after the stage."""
    argv = stage["argv"] + ["--describe"]
    if geo["frames"] is not None and "--frames" not in stage["argv"]:
        argv += ["--frames", str(geo["frames"])]
    label, name = stage["label"], stage["name"]
    proc = run(argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
               errors="replace", timeout=PROBE_TIMEOUT)
    lines = (proc.stderr or "").strip().splitlines()
    if proc.returncode == 127:
        fail(f"{label}: cannot run `{argv[0]}`", kind="tool_missing", stage=name)
    if proc.returncode == 124 and proc.stderr == "process timed out":
        fail(f"{label}: --describe gave no answer within {PROBE_TIMEOUT} s", kind="stage_timeout", stage=name)
    if proc.returncode != 0:
        fail(f"{label}: --describe exited with code {proc.returncode}" + (": " + lines[-1] if lines else ""),
             kind=STAGE_EXIT_KIND.get(proc.returncode, "stage_failed"), stage=name,
             exit_code=proc.returncode, tail=lines[-STAGE_TAIL:])
    try:
        answer = json.loads(proc.stdout)
    except ValueError:
        answer = None
    problem = None
    if not isinstance(answer, dict) or set(answer) != set(DESCRIBE_KEYS):
        problem = "it is not a describe answer (contracts/describe.schema.json)"
    elif answer["ok"] is not True or answer["contract"] != "frames/2":
        problem = "ok must be true and contract frames/2"
    elif not all(type(answer[k]) is int and answer[k] >= 1 for k in ("width", "height")):
        problem = "width and height must be whole numbers of 1 or more"
    elif answer["frames"] is not None and not (type(answer["frames"]) is int and answer["frames"] >= 0):
        problem = "frames must be a whole number or null"
    elif not isinstance(answer["fps"], str) or not re.fullmatch(r"[1-9][0-9]*/[1-9][0-9]*", answer["fps"]):
        problem = "fps must be an exact N/D"
    if problem:
        fail(f"{label}: --describe gave an invalid answer: {problem}", kind="stage_failed", stage=name,
             exit_code=0, tail=lines[-STAGE_TAIL:])
    if Fraction(answer["fps"]) != geo["fps"]:
        fail(f"{label}: the stage would change the frame rate from {rate_text(geo['fps'])} to {answer['fps']}; "
             "a frame rate change is not supported", kind="unsupported", stage=name)
    if answer["depth"] != geo["depth"]:
        fail(f"{label}: the stage would change the depth from {geo['depth']} to {answer['depth']}",
             kind="unsupported", stage=name)
    return dict(geo, width=answer["width"], height=answer["height"], frames=answer["frames"])


def pipe_chain(decode, effect, encode, stages):
    """every process of the pipe in the order frames flow through it: decode, the pre stages, vfx,
    the post stages, encode. without stages it is the three commands of the plain pipe."""
    chain = [{"name": "decode", "label": "decode", "role": "decode", "argv": decode}]
    chain += [s for s in stages if s["role"] == "pre"]
    chain.append({"name": "vfx", "label": "vfx", "role": "vfx", "argv": effect})
    chain += [s for s in stages if s["role"] == "post"]
    chain.append({"name": "encode", "label": "encode", "role": "encode", "argv": encode})
    return chain


def frames_text(count, kind=""):
    return f"{count} {kind}frame" + ("" if count == 1 else "s")


def stage_report(stage, code=None, log=None):
    """the json entry of a stage: its place, the command after substitution and, once it ran, how it ended."""
    return {"role": stage["role"], "name": stage["name"], "command": stage["command"],
            "exit_code": code, "stderr_tail": tail(log, STAGE_TAIL) if log else []}


def geometry_json(geo):
    """a chained geometry as contracts/run-report.schema.json writes it (fps as an exact N/D)."""
    return {"width": geo["width"], "height": geo["height"], "depth": geo["depth"],
            "fps": rate_text(geo["fps"]), "frames": geo["frames"]}


def chain_report(chain, verify_argv, codes=None, logs=None, verified=None):
    """the frames/2 `stages` array: every process of the pipe in pipe order, then the verify step,
    each with role (vfx is the effect), index within its role, name, argv, exit code and stderr tail
    (contracts/run-report.schema.json). a declared stage also carries the geometry it answered."""
    counts, report = {}, []
    for entry in chain:
        role = REPORT_ROLE.get(entry["role"], entry["role"])
        counts[role] = counts.get(role, 0) + 1
        item = {"role": role, "index": counts[role], "name": entry["name"], "command": list(entry["argv"]),
                "exit_code": codes.get(entry["name"]) if codes else None,
                "stderr_tail": tail(logs[entry["name"]], STAGE_TAIL) if logs else []}
        if entry.get("answer"):
            item["geometry"] = geometry_json(entry["answer"])
        report.append(item)
    report.append({"role": "verify", "index": 1, "name": "verify", "command": list(verify_argv),
                   "exit_code": verified, "stderr_tail": []})
    return report


class Hop:
    """what flows from one process of the pipe to the next, and the thread that carries it. between
    two processes of a run with stages vfx's data goes through here instead of a direct pipe, which
    lets the run see what an exit code cannot: bytes that stop moving, a consumer that stops reading
    and a producer that stops in the middle of a frame."""

    def __init__(self, source, sink):
        self.source, self.sink = source, sink
        self.moved = time.monotonic()  # when a byte last went in or came out
        self.state = "reading"         # reading: waits for the producer, writing: waits for the consumer, done
        self.read = self.written = 0   # bytes taken from the producer, bytes the consumer accepted
        self.eof = False               # the producer closed its output
        self.broken = False            # the consumer closed its input while the producer still had data
        self.thread = threading.Thread(target=self.carry, daemon=True)

    def carry(self):
        try:
            while True:
                self.state = "reading"
                try:
                    chunk = self.source.read(RELAY_CHUNK)
                except (OSError, ValueError):
                    chunk = b""
                if not chunk:
                    self.eof = True
                    return
                self.read += len(chunk)
                self.moved = time.monotonic()
                self.state = "writing"
                view = memoryview(chunk)
                try:
                    while view:
                        sent = self.sink.write(view)
                        view = view[sent:]
                        self.written += sent
                        self.moved = time.monotonic()
                except (OSError, ValueError):
                    self.broken = True
                    return
        finally:
            # closing the sink is the producer's end of file for the consumer; closing the source
            # is a broken pipe for a producer whose consumer is gone
            self.state = "done"
            for end in (self.sink, self.source):
                try:
                    end.close()
                except (OSError, ValueError):
                    pass


def stalled_stage(chain, procs, hops, timeout, now, ends=False):
    """the index of a pre or post stage that holds the pipe up, or None. it is alive, no byte has
    moved on either of its pipes for `timeout` seconds, it is not taking what is offered (the offer is
    blocked, or everything was handed over) and the next process is not the one that is slow. a stage
    that waits for a slow producer is not stalled. vfx is never judged, because a temporal effect can
    be silent for a long time while it renders. with `ends` (frames/2) the decode is judged when its
    relay has waited `timeout` seconds for its output, and the encode when the relay has offered it
    data for `timeout` seconds that it does not take (an encode finishing its file after the last
    frame is not judged); under frames/1 they are not judged."""
    last = len(chain) - 1
    for i, entry in enumerate(chain):
        role = entry["role"]
        if role in ("pre", "post"):
            if procs[i].poll() is not None:
                continue
            before, after = hops[i - 1], hops[i]
            if now - max(before.moved, after.moved) >= timeout and before.state != "reading" and after.state != "writing":
                return i
        elif ends and role == "decode" and i < last and procs[i].poll() is None:
            if now - hops[i].moved >= timeout and hops[i].state == "reading":
                return i
        elif ends and role == "encode" and i > 0 and procs[i].poll() is None:
            if now - hops[i - 1].moved >= timeout and hops[i - 1].state == "writing":
                return i
    return None


def kill_tree(proc):
    """kill a process and what it started. a launcher (a windows venv python, a .cmd wrapper) runs the
    real program as a child, and killing only the launcher would leave that child holding the pipes.
    on posix the child leads its own session, so its group is signalled even when the leader has
    already exited: the members it started may still run."""
    if os.name == "nt":
        if proc.poll() is None:
            run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=PROBE_TIMEOUT)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass


def run_staged_pipeline(chain, scratch, timeout):
    """chain[0] | chain[1] | ... with a Hop between every two neighbours and every stderr in a log
    file in scratch, so no pipe can fill up and stall a process. a pre or post stage that stalls for
    `timeout` seconds (0: never) is killed; once a process has failed the others get GRACE seconds to
    end before they are killed too, so the run always ends. returns the exit code and log path of every
    process by name, the hops (hops[i] carries chain[i] to chain[i + 1]) and the names killed for stalling."""
    names = [entry["name"] for entry in chain]
    logs = {name: os.path.join(scratch, name + ".log") for name in names}
    handles = {name: open(path, "w", encoding="utf-8") for name, path in logs.items()}
    procs, hops, timed_out = [], [], set()
    last = len(chain) - 1
    try:
        for i, entry in enumerate(chain):
            try:
                procs.append(subprocess.Popen(entry["argv"], shell=False, bufsize=0,
                                              stdin=subprocess.DEVNULL if i == 0 else subprocess.PIPE,
                                              stdout=subprocess.DEVNULL if i == last else subprocess.PIPE,
                                              stderr=handles[entry["name"]],
                                              start_new_session=os.name != "nt"))
            except OSError as exc:
                fail(f"cannot start {entry['label']}: {exc}", kind="tool_missing", stage=entry["name"])
        hops = [Hop(procs[i].stdout, procs[i + 1].stdin) for i in range(last)]
        for hop in hops:
            hop.thread.start()
        failed_at = None
        while any(p.poll() is None for p in procs):
            now = time.monotonic()
            if failed_at is None and (any(p.poll() not in (None, 0) for p in procs) or any(h.broken for h in hops)):
                failed_at = now
            if failed_at is not None and now - failed_at > GRACE:
                for p in procs:
                    kill_tree(p)
            if timeout:
                stalled = stalled_stage(chain, procs, hops, timeout, now, ends=CONTRACT == "frames/2")
                if stalled is not None and names[stalled] not in timed_out:
                    timed_out.add(names[stalled])
                    kill_tree(procs[stalled])
            time.sleep(0.05)
        for hop in hops:
            hop.thread.join(GRACE)
        codes = {name: p.returncode for name, p in zip(names, procs)}
    except BaseException:
        for p in procs:
            kill_tree(p)
        raise
    finally:
        for h in handles.values():
            h.close()
    return codes, logs, hops, timed_out


def diagnose(chain, codes, hops, timed_out, timeout, frame_bytes, stage_exits=None):
    """what went wrong in a run with stages: a list of problems, the root causes first, empty when
    nothing did. a problem is a dict with kind (an error code of the contract), name, label,
    exit_code, message and root. the exit code of a process that only suffered from its neighbour
    (a broken pipe to a consumer that went away, a partial frame from a producer that stopped
    mid-frame) is listed after the cause. the hops add three failures an exit code does not show:
    a stage that stopped in the middle of a frame, one that stopped reading its input, and one that
    wrote fewer whole frames than it was sent.

    `frame_bytes` is one size for every relay, or a list with the size on each relay (hops[i]) when a
    declared stage changed it; a declared stage owes the frame count of its answer, not what it was
    sent. `stage_exits` (default: under frames/2) maps the exits 2, 3 and 4 of every pre and post
    stage to usage, unsupported and bad_input, as they are for vfx in both contracts."""
    if stage_exits is None:
        stage_exits = CONTRACT == "frames/2"

    def size(k):
        return frame_bytes[k] if isinstance(frame_bytes, (list, tuple)) else frame_bytes

    problems = []
    last = len(chain) - 1
    for i, entry in enumerate(chain):
        name, label, code = entry["name"], entry["label"], codes[entry["name"]]
        before = hops[i - 1] if i > 0 else None
        after = hops[i] if i < last else None
        cut = after.read % size(i) if after is not None and after.eof else 0
        found = None
        if name in timed_out:
            found = ("stage_timeout", None, f"{label} held the pipe up for {timeout:g} s without moving a byte "
                                            "and was killed (--stage-timeout)", True)
        elif code:
            victim = (after is not None and after.broken) or (before is not None and before.eof
                                                              and before.read % size(i - 1) > 0)
            mapped = name == "vfx" or (stage_exits and entry["role"] in ("pre", "post"))
            kind = STAGE_EXIT_KIND.get(code, "stage_failed") if mapped else "stage_failed"
            found = (kind, code, f"{label} exited with code {code}", not victim)
        elif cut:
            whole = frames_text(after.read // size(i), "whole ")
            found = ("partial_frame", code, f"{label} ended in the middle of a frame ({cut} bytes after {whole}, "
                                            f"a frame is {size(i)} bytes)", True)
        elif entry["role"] in ("pre", "post"):
            sent, wrote = before.written // size(i - 1), after.read // size(i)
            owed = entry["answer"]["frames"] if entry.get("geometry") == "declared" else None
            if before.broken:
                found = ("broken_pipe", code, f"{label} stopped reading its input after "
                                              f"{frames_text(sent, 'whole ')}", True)
            elif entry.get("geometry") == "declared":
                if owed is not None and wrote != owed:
                    found = ("broken_pipe", code, f"{label} wrote {frames_text(wrote)}, its --describe answer "
                                                  f"declared {owed}", True)
            elif wrote < sent:
                found = ("broken_pipe", code, f"{label} stopped after writing {frames_text(wrote)}, "
                                              f"it was sent {sent}", True)
        if found:
            kind, exit_code, message, root = found
            problems.append({"kind": kind, "name": name, "label": label, "exit_code": exit_code,
                             "message": message, "root": root})
    problems.sort(key=lambda p: not p["root"])
    return problems


def run_staged(chain, scratch, args, info):
    """run the pipe with stages and fail, in the style of a vfx failure, when it did not go through:
    the failing process, its exit code and the last lines of its own stderr. returns the codes and logs."""
    codes, logs, hops, timed_out = run_staged_pipeline(chain, scratch, args.stage_timeout)
    sizes = frame_size(args, info)
    if any("frame_bytes" in entry for entry in chain):
        sizes = [entry.get("frame_bytes", sizes) for entry in chain[:-1]]
    problems = diagnose(chain, codes, hops, timed_out, args.stage_timeout, sizes)
    if problems:
        lines = []
        for p in problems:
            lines.append(p["message"])
            if p["root"]:
                lines += ["  " + t for t in tail(logs[p["name"]])]
        first = problems[0]
        fail("the pipeline failed:\n" + "\n".join(lines), kind=first["kind"], stage=first["name"],
             exit_code=first["exit_code"], tail=tail(logs[first["name"]], STAGE_TAIL), summary=first["message"])
    return codes, logs


# an output cut at the end of the video keeps the audio up to the packet that crosses the end
TRIM_TOLERANCE = 0.1


def verify(src, out, rate, frames=None, source_frames=None, size=None, durations=True):
    """problems found comparing the output to the input; empty means it checks out.

    with `frames` (a --frames value) the output must have that many frames and last
    frames / rate; when that is the input's own count (`source_frames`) the output is
    compared with the input's duration as usual. `size` is the size a declared stage gave the output
    (frames/2); `durations` False skips the length checks when a declared stage left the count unknown."""
    if out is None or not out["has_video"]:
        return ["the output has no readable video stream"]
    problems = []
    want = size or display_size(src)
    if (out["width"], out["height"]) != want:
        problems.append(f"size is {out['width']}x{out['height']}, expected {want[0]}x{want[1]}")
    if out["fps"] is None or abs(out["fps"] - float(rate)) > 0.01:
        problems.append(f"frame rate is {out['fps']}, expected {float(rate):.3f}")
    if frames is not None and frames != source_frames:
        expected, window = frames / float(rate), max(1.0 / float(rate), TRIM_TOLERANCE) + 1e-3
        if out["duration"] is None:
            problems.append("the output has no readable duration")
        elif abs(out["duration"] - expected) > window:
            problems.append(f"duration is {out['duration']:.3f}s, expected {expected:.3f}s "
                            f"({frames} frames at {float(rate):.3f} fps) within {window:.3f}s")
    elif src["duration"] is not None and durations:
        if out["duration"] is None:
            problems.append("the output has no readable duration")
        elif abs(out["duration"] - src["duration"]) > 1.0 / float(rate) + 1e-3:
            problems.append(f"duration is {out['duration']:.3f}s, expected {src['duration']:.3f}s "
                            f"within one frame ({1.0 / float(rate):.3f}s)")
    if frames is not None and out.get("nb_frames") is not None and out["nb_frames"] != frames:
        problems.append(f"the output has {out['nb_frames']} frames, expected {frames}")
    if src["has_audio"]:
        if not out["has_audio"]:
            problems.append("the audio stream was lost")
        elif (out["acodec"], out["channels"]) != (src["acodec"], src["channels"]):
            problems.append(f"audio is {out['acodec']} {out['channels']}ch, "
                            f"expected {src['acodec']} {src['channels']}ch")
    elif out["has_audio"]:
        problems.append("the output has audio the input did not")
    return problems


def make_result(args, info, einfo, vfx, dst, plan, commands, stages=()):
    """the report for a plan: the facts a caller needs before anything runs, outputs filled in later.
    a run with stages also names the contract and lists the stages (exit codes filled in after the run)."""
    w, h = display_size(info)
    rate = info["rate"]
    parts = [f"{args.effect} on {os.path.basename(args.input)}: {w}x{h}, {float(rate):.3f} fps, "
             f"{args.depth} bpc ({PIX_FMT[args.depth]})"]
    if plan["kind"] == "temporal":
        parts.append(f"temporal, {plan['frames']} frames" if plan["frames"] is not None
                     else "temporal, as many frames as the input decodes to")
        if plan["layers"]:
            parts.append("layers " + ", ".join(layer["id"] for layer in plan["layers"]))
        if plan["audio"] == "shortest":
            parts.append("audio cut at the end of the video (-shortest)")
        elif plan["audio"] == "ends-early":
            parts.append("the audio ends before the video does")
    pipe = [e["label"] for e in pipe_chain(None, None, None, stages)]
    parts.append(f"{' | '.join(pipe)} -> {dst}")
    return {
        "operation": "apply",
        **({"contract": CONTRACT} if stages else {}),
        "input": os.path.abspath(args.input),
        "effect": args.effect,
        "kind": plan["kind"],
        "params": args.param,
        "depth": args.depth,
        "pix_fmt": PIX_FMT[args.depth],
        "size": f"{w}x{h}",
        "fps": float(rate),
        "fps_exact": rate_text(rate),
        "frames": plan["frames"],
        "input_frames": plan["input_frames"],
        "layers": [{"id": layer["id"], "video": layer["video"], "temp": layer["temp"]} for layer in plan["layers"]],
        "audio": {"plan": plan["audio"], "shortest": plan["shortest"]},
        "temp_files": plan["temp_files"],
        "vfx": vfx,
        "summary": ", ".join(parts),
        "dry_run": args.dry_run,
        "commands": [fmt_cmd(c) for c in commands],
        **({"stages": [stage_report(s) for s in stages]} if stages else {}),
        "outputs": [],
    }


def main():
    parser = ArgParser(
        prog="apply.py",
        description="run an effect over a video via ffmpeg and vfx.")
    parser.add_argument("input")
    parser.add_argument("effect", help="effect id as `vfx list` shows it, for example retime")
    parser.add_argument("--param", action="append", default=[], type=param_arg, metavar="K=V",
                        help="effect param, repeatable (see `vfx list` for ids and ranges). a temporal "
                             "effect also takes keyframes, V@F;V@F with F a frame number, a trailing h "
                             "for a hold key and a trailing e to ease a key; quote the argument, the ; "
                             "ends a command in a shell")
    parser.add_argument("--layer", action="append", default=[], type=layer_arg, metavar="ID=VIDEO",
                        help="an extra input layer of a temporal effect (the ids `vfx list` shows), "
                             "repeatable. the video is decoded to a temporary raw file in the same "
                             "pixel format, size and frame rate, and removed afterwards")
    parser.add_argument("--frames", type=frames_arg, metavar="N",
                        help="frames of the output of a temporal effect (default: the frames of the "
                             "input). the output keeps the input's size and frame rate and lasts "
                             "N / fps. the audio is copied untouched: when the output is shorter than "
                             "the audio it is cut at the end of the video (ffmpeg -shortest), when "
                             "longer the audio just ends early")
    parser.add_argument("--depth", type=int, choices=sorted(PIX_FMT), default=16,
                        help="bits per channel the effect runs at (default: 16)")
    parser.add_argument("--lossless", action="store_true",
                        help="encode with ffv1 in the RGB the effect produced (no yuv420p, no tv range) "
                             "instead of libx264 CRF 18, and write .mkv by default. use it for effects "
                             "whose output is data, such as the vector images of create-vectors, that "
                             "another run reads back with --layer. depth 8 or 16 only (float has no "
                             "ffv1 format)")
    parser.add_argument("--pre-stage", action=StageAction, default=[], type=stage_arg, metavar="CMD",
                        help="a raw frame filter that runs between the decode and vfx, repeatable, in the "
                             "order given. CMD is one command line (split like a shell would, no shell "
                             "involved) that reads frames of the --depth format on stdin and writes the "
                             "same number of frames of the same format and size on stdout. in it {width} "
                             "{height} {depth} {fps} (as N/D) {frames} and {pix_fmt} are filled in; "
                             "{frames} is refused when the frame count is not known. a stage that exits "
                             "non-zero, stops mid-frame or stops reading fails the run")
    parser.add_argument("--post-stage", action=StageAction, default=[], type=stage_arg, metavar="CMD",
                        help="like --pre-stage, between vfx and the encode; its {frames} is the length "
                             "of vfx's output (--frames for a temporal effect)")
    parser.add_argument("--stage-timeout", type=timeout_arg, default=STAGE_TIMEOUT, metavar="SECONDS",
                        help="kill a --pre-stage or --post-stage that holds the pipe up for this long "
                             "without a byte moving on its pipes (default: 120, 0 never). a stage that "
                             "waits for a slow neighbour is not held up. under frames/2 the decode and the "
                             "encode are watched too; vfx never is")
    parser.add_argument("--geometry", action=GeometryAction, metavar="VALUE",
                        help="the geometry of the --pre-stage or --post-stage given just before it (frames/2): "
                             "preserve (the default) or declared. a declared stage is asked `CMD --describe` "
                             "(contracts/describe.schema.json) and its width, height and frame count are what "
                             "the next process, the encode and the verify get; a frame rate or depth change is refused "
                             "(exit 3). "
                             "another value is refused as unsupported")
    parser.add_argument("--contract", choices=CONTRACTS, default="frames/1",
                        help="frames/1 (default, unchanged) or frames/2: the error object on every --json "
                             "failure, the exit codes of contracts/error.schema.json (2 usage, 3 unsupported, "
                             "4 bad_input, broken_pipe and partial_frame, 5 stage_timeout, 6 tool_missing), "
                             "the exits 2, 3 and 4 of every stage mapped, role and index for every process "
                             "in the report, and declared geometry")
    parser.add_argument("--describe", action="store_true",
                        help="frames/2: print the output geometry of the run (after every declared stage) as "
                             "a describe answer and stop, reading and writing no frames")
    parser.add_argument("--vfx", metavar="PATH",
                        help="vfx executable (default: VFX_PATH, then the workspace release build "
                             "under CARGO_TARGET_DIR or target/, then PATH)")
    add_common_args(parser)
    args = parser.parse_args()
    configure(args)
    framed = CONTRACT == "frames/2"
    if not framed and (getattr(args, "geometry_given", False) or args.describe):
        fail(("--describe" if args.describe else "--geometry") + " needs --contract frames/2")
    for template in args.pre_stage + args.post_stage:
        value = getattr(template, "geometry", "preserve")
        if value not in GEOMETRY_VALUES:
            fail(f"unknown stage geometry `{value}`: this tool knows " + " and ".join(GEOMETRY_VALUES),
                 kind="unsupported")

    require_ffmpeg()
    info = require_input(args.input)
    if not info["has_video"]:
        fail("input has no video stream", kind="bad_input")
    if not info["rate"] or not info["width"] or not info["height"]:
        fail("cannot read the input's frame size and frame rate", kind="bad_input")
    if info["hdr"]:
        fail("HDR input is not supported; tone mapping is not part of this tool", kind="unsupported")
    w, h = display_size(info)
    if args.lossless and args.depth == 32:
        fail("--lossless needs --depth 8 or 16: ffv1 has no float pixel format")
    if (w % 2 or h % 2) and not args.lossless:
        fail(f"the frame size {w}x{h} has an odd side, and yuv420p needs even sides; "
             "resize or crop the input first (ffmpeg -vf crop=trunc(iw/2)*2:trunc(ih/2)*2), or pass --lossless",
             kind="unsupported")

    vfx = find_vfx(args.vfx)
    einfo = effect_info(vfx, args.effect)
    check_layer_options(args, einfo)
    layers = probe_layers(args) if einfo["kind"] == "temporal" else []
    preflight(args, info, vfx, einfo)
    chain_geo = {} if framed else None
    stages = build_stages(args, info, chain_geo)
    declared = any(s.get("geometry") == "declared" for s in stages)
    sizes = None
    if framed:
        vgeo, ogeo = chain_geo["vfx"], chain_geo["out"]
        sizes = ((vgeo["width"], vgeo["height"]), (ogeo["width"], ogeo["height"]))
        if declared and (ogeo["width"] % 2 or ogeo["height"] % 2) and not args.lossless:
            fail(f"the declared stages give a frame size of {ogeo['width']}x{ogeo['height']}, which has an odd "
                 "side, and yuv420p needs even sides; pass --lossless", kind="unsupported")
        if args.describe:
            print(json.dumps({"ok": True, "contract": "frames/2", "stage": args.effect, **geometry_json(ogeo)}))
            sys.exit(0)

    dst = args.output or default_output(args.input, args.effect, "mkv" if args.lossless else "mp4")
    if same_file(dst, args.input):
        fail("the output path is the input file; choose a different --output path")
    for layer in layers:
        if same_file(dst, layer["video"]):
            fail(f"the output path is the video of layer {layer['id']}; choose a different --output path")

    def assemble(scratch):
        """plan, layer decodes and the three stages for a scratch directory (named, not created)."""
        plan = make_plan(args, info, einfo, layers, scratch)
        if declared and chain_geo["out"]["frames"] not in (None, plan["input_frames"]):
            plan["audio"] = audio_plan(info, chain_geo["out"]["frames"], info["rate"])
            plan["shortest"] = plan["audio"] == "shortest"
        decode, effect, encode = build_commands(args, info, vfx, dst, plan, sizes)
        layer_commands = build_layer_commands(args, info, plan, sizes[0] if sizes else None)
        return plan, layer_commands, (decode, effect, encode)

    def framed_result(result, chain):
        """the frames/2 report: ok and contract first, every process with role and index, the geometry."""
        out = {"ok": True, "contract": "frames/2"}
        out.update((k, v) for k, v in result.items() if k not in ("contract", "stages"))
        out["stages"] = chain_report(chain, probe_command(partial_path(dst)))
        out["geometry"] = geometry_json(chain_geo["out"])
        return out

    if args.dry_run:
        plan, layer_commands, triple = assemble(os.path.join(tempfile.gettempdir(), "apply_XXXXXXXX"))
        chain = pipe_chain(*triple, stages)
        pipe = [entry["argv"] for entry in chain]
        result = make_result(args, info, einfo, vfx, dst, plan, layer_commands + pipe, stages)
        emit(framed_result(result, chain) if framed else result, args.as_json)
        sys.exit(0)

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    partial = partial_path(dst)
    discard(partial)
    scratch = tempfile.mkdtemp(prefix="apply_")
    started = time.time()
    try:
        plan, layer_commands, (decode, effect, encode) = assemble(scratch)
        chain = pipe_chain(decode, effect, encode, stages)
        result = make_result(args, info, einfo, vfx, dst, plan, layer_commands + [e["argv"] for e in chain], stages)
        if framed:
            for entry in chain:
                if entry["role"] == "vfx":
                    entry["frame_bytes"] = vgeo["width"] * vgeo["height"] * BYTES_PER_PIXEL[args.depth]
            result = framed_result(result, chain)
        decode_layers(layer_commands, plan, args, info, scratch, sizes[0] if sizes else None)
        if framed:
            codes, logs = run_staged(chain, scratch, args, info)
            result["stages"] = chain_report(chain, probe_command(partial), codes, logs)
        elif stages:
            codes, logs = run_staged(chain, scratch, args, info)
            result["stages"] = [stage_report(s, codes[s["name"]], logs[s["name"]]) for s in stages]
        else:
            codes, logs = run_pipeline(decode, effect, encode, scratch)
            if any(codes.values()):
                lines = []
                for name in ("vfx", "decode", "encode"):
                    if codes[name]:
                        lines.append(f"{name} exited with code {codes[name]}")
                        lines += ["  " + t for t in tail(logs[name])]
                fail("the pipeline failed:\n" + "\n".join(lines))
        if declared:
            final = chain_geo["out"]
            problems = verify(info, probe(partial), info["rate"], frames=final["frames"],
                              source_frames=plan["input_frames"], size=(final["width"], final["height"]),
                              durations=final["frames"] is not None)
        else:
            problems = verify(info, probe(partial), info["rate"], frames=args.frames,
                              source_frames=plan["input_frames"])
        if problems:
            fail("the output did not verify: " + "; ".join(problems), kind="stage_failed", stage="verify")
        if framed:
            result["stages"][-1]["exit_code"] = 0
        try:
            os.replace(partial, dst)
        except OSError:
            fail("could not move the finished output into place; is the destination open in another program?",
                 kind="stage_failed", stage="verify")
    except KeyboardInterrupt:
        if framed:
            fail("the run was interrupted", kind="interrupted")
        eprint("[interrupted]")
        sys.exit(130)
    finally:
        discard(partial)
        shutil.rmtree(scratch, ignore_errors=True)

    result["seconds"] = round(time.time() - started, 2)
    result["outputs"] = [verified_output(dst)]
    emit(result, args.as_json)
    sys.exit(0)


if __name__ == "__main__":
    main()
