#!/usr/bin/env python3
"""apply.py: run one of the effects over a video, through ffmpeg and vfx.

ffmpeg decodes the input to raw frames, `vfx apply` runs the effect on them, and a
second ffmpeg encodes the result with libx264 (CRF 18, yuv420p) while copying the
source audio untouched. the output is written to a partial file, probed against the
input, and only then moved onto the real path. a size with an odd side encodes
yuv444p at that exact size; an HDR input (transfer smpte2084 or arib-std-b67) goes
through as stored, with no tone mapping, and is encoded with libx265 in 10 bit with
its tags and its HDR metadata.

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
                     eases it, iS/I and oS/I after F are incoming and outgoing
                     Bezier handles of S units a second and I percent influence).
                     quote it in a shell, the ; ends a command there.
  --layer ID=VIDEO   an extra input layer the effect lists. the video is decoded to a
                     temporary raw file at the output's size (scaled when it
                     differs), then removed. repeatable.
  --frames N         output length in frames for a temporal effect (default: the
                     frames of the input). the output lasts N / fps. with the default
                     --audio copy the audio is copied untouched; when the output is
                     shorter than the audio it is cut at the end of the video (ffmpeg
                     -shortest), when longer the audio just ends early.
  --device D         the device vfx renders a temporal effect on: cpu (the default),
                     gpu or gpu=N, passed through to vfx apply --device (`vfx
                     devices` lists them). gpu falls back to the cpu with a note
                     when no device passes the gate; gpu=N that cannot be used is
                     refused before any work. the frames are the cpu's either way.
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
  --size WxH         the output frame size: the input and every layer are scaled to
                     it, stretched, before the effect.
  --fps R            the output frame rate. a temporal effect reads the input and
                     every layer at their own rates (vfx --clip-fps, --layer-fps), so
                     no real frame is dropped or repeated on the way in; a frame
                     effect gets, for each output frame, the input frame that covers
                     its start. --frames, keys and frame numbers count output frames,
                     and the default length is the output frames that start inside
                     the input. the input's own size or rate changes nothing.

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
import struct
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


# the raw ffprobe json of the last good probe() of each file, by absolute path (the "path" of its
# facts). the facts stay what they are; a reader that needs more of a file (every audio stream, the
# side data of the video stream) takes it from here instead of running ffprobe again
_PROBE_DATA = {}


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
    _PROBE_DATA.pop(os.path.abspath(path), None)
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        data = json.loads(proc.stdout)
    except ValueError:
        return None
    _PROBE_DATA[os.path.abspath(path)] = data

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


def device_arg(text):
    """cpu, gpu or gpu=N (N in ascii digits, at most 2^32 - 1), the values vfx apply --device takes."""
    text = text.strip()
    if not re.fullmatch(r"cpu|gpu|gpu=[0-9]+", text) or (text[:4] == "gpu=" and int(text[4:]) > 0xFFFFFFFF):
        raise argparse.ArgumentTypeError("expected cpu, gpu or gpu=N, for example gpu=0")
    return text


# the whitespace Rust's str::trim takes off a value (Unicode White_Space): --fps reads a rate as vfx does
RUST_WHITESPACE = ("\t\n\x0b\x0c\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007"
                   "\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000")
U32_MAX = 0xFFFFFFFF


def exact_rate(text):
    """a frame rate read the way vfx reads --fps into an exact rate: an integer, a decimal of at most 6
    places (29.97 is 2997/100) or N/D with N and D whole numbers, each at most 2^32 - 1 and the rate above
    0, in ascii digits with surrounding whitespace allowed. a Fraction in lowest terms, or None."""
    s = text.strip(RUST_WHITESPACE)
    if "/" in s:
        parts = [part.strip(RUST_WHITESPACE) for part in s.split("/", 1)]
        if not all(re.fullmatch(r"\+?[0-9]+", part) for part in parts):
            return None
        num, den = (int(part) for part in parts)
    else:
        whole, point, decimals = s.partition(".")
        digits = whole + decimals
        if not whole or (point and not decimals) or len(decimals) > 6 or not re.fullmatch(r"[0-9]+", digits):
            return None
        num, den = int(digits), 10 ** len(decimals)
    if not (0 < num <= U32_MAX and 0 < den <= U32_MAX):
        return None
    return Fraction(num, den)


def fps_arg(text):
    """--fps R: the output frame rate, as exact_rate reads it."""
    rate = exact_rate(text)
    if rate is None:
        raise argparse.ArgumentTypeError("expected a frame rate above 0: an integer, a decimal of at most 6 "
                                         "places or N/D, for example 24, 29.97 or 30000/1001")
    return rate


def size_arg(text):
    """--size WxH: the output frame size, W and H whole numbers of 1 or more in ascii digits."""
    match = re.fullmatch(r"([0-9]+)x([0-9]+)", text.strip())
    if not match or int(match.group(1)) < 1 or int(match.group(2)) < 1:
        raise argparse.ArgumentTypeError("expected WxH with W and H of 1 or more, for example 1280x720")
    return int(match.group(1)), int(match.group(2))


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
                           "--describe", "--device", "--audio", "--size", "--fps"})

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
    for step in result.get("prepare") or []:
        print(f"[step] {step['stream']}: decoded as {step['decoded']}, given to vfx as {step['given']}, the samples "
              f"{step['samples']}, the alpha {step['alpha']}")
    if result.get("dry_run"):
        print("[dry-run] nothing was written")
        return
    for out in result.get("outputs") or []:
        print(f"{out['path']}: {describe(out.get('probe'))}")


# ----------------------------------------------------------------------------
# the pipeline
# ----------------------------------------------------------------------------

# the tags of an HDR input (probe's hdr: transfer smpte2084 or arib-std-b67) that has no matrix swscale
# knows, or no primaries
HDR_MATRIX, HDR_PRIMARIES = "bt2020nc", "bt2020"


def color_settings(info):
    """the matrix to decode and encode with, and the tags to write.

    a tagged source keeps its own. an untagged one follows the size convention
    players use, HD is bt709 and anything smaller bt601, and is tagged to match.
    an HDR source keeps its transfer, its matrix as tagged (bt2020nc when swscale has
    no such matrix) and its primaries as tagged (bt2020 when it has none): the signal
    goes through as stored, with no tone mapping, so a blend of the effect acts on the
    PQ or HLG code values. returns (swscale matrix, colorspace tag, primaries tag,
    transfer tag), where a tag the source never had stays None."""
    space = info["color_space"]
    primaries, trc = info["color_primaries"], info["color_transfer"]
    if info.get("hdr"):
        space = space if space in MATRIX else HDR_MATRIX
        return MATRIX[space], space, primaries or HDR_PRIMARIES, trc
    if space not in MATRIX:
        w, h = display_size(info)
        space = "bt709" if max(w, h) >= 1280 or h >= 720 else "smpte170m"
        primaries = trc = space
    return MATRIX[space], space, primaries, trc


# the units of the master-display string of x265: chromaticity in 0.00002, luminance in 0.0001 cd/m2
CHROMA_UNIT, LUMA_UNIT = 50000, 10000
U16_MAX = 0xFFFF


def _units(value, unit, limit):
    """an ffprobe rational ("34000/50000", "17/25") as a whole number of 1/unit, rounded half up; None
    when it is missing or malformed, or the number is below 0 or above limit."""
    try:
        number = Fraction(str(value))
    except (ValueError, ZeroDivisionError, TypeError):
        return None
    whole = (2 * number * unit + 1) // 2
    return whole if 0 <= whole <= limit else None


def master_display(side):
    """the x265 master-display string G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min) of an ffprobe "Mastering
    display metadata" entry (whose primaries are in red, green, blue order); None unless the entry
    carries all ten values, each in range."""
    chroma = [_units(side.get(f"{color}_{axis}"), CHROMA_UNIT, U16_MAX)
              for color in ("green", "blue", "red", "white_point") for axis in ("x", "y")]
    luma = [_units(side.get(key), LUMA_UNIT, U32_MAX) for key in ("max_luminance", "min_luminance")]
    if any(v is None for v in chroma + luma):
        return None
    return "G({},{})B({},{})R({},{})WP({},{})L({},{})".format(*(chroma + luma))


def max_cll(side):
    """the x265 max-cll string MaxCLL,MaxFALL of an ffprobe "Content light level metadata" entry; None
    unless both are whole numbers of 0 to 65535."""
    values = []
    for key in ("max_content", "max_average"):
        text = str(side.get(key))
        if not re.fullmatch(r"[0-9]+", text) or int(text) > U16_MAX:
            return None
        values.append(int(text))
    return f"{values[0]},{values[1]}"


# the side data an HDR output gets back: the ffprobe side_data_type, the key and the reader of each
HDR_SIDE_DATA = {"Mastering display metadata": ("master_display", master_display),
                 "Content light level metadata": ("max_cll", max_cll)}
# hdr_metadata of each probed file, by absolute path: (the raw probe json it was read with, the result)
_HDR_METADATA = {}


def _hdr_side_data(entries, found):
    """fill the keys of `found` still None from a side_data_list, the first usable entry of each kind."""
    for side in entries if isinstance(entries, list) else []:
        kind = HDR_SIDE_DATA.get(side.get("side_data_type")) if isinstance(side, dict) else None
        if kind and found[kind[0]] is None:
            found[kind[0]] = kind[1](side)


def hdr_metadata(path, info):
    """the HDR metadata of a file that x265 can write back, {"master_display": ..., "max_cll": ...}, each
    the x265 string or None. each comes from the side data of the video stream (ffprobe side_data_list,
    what the container holds, read from the json of probe()), else from the side data of its first
    frame (one ffprobe of one packet: the SEI of HEVC, the metadata of AV1). the result is kept until
    the file is probed again."""
    key = os.path.abspath(path)
    data = _PROBE_DATA.get(key)
    if data is not None and key in _HDR_METADATA and _HDR_METADATA[key][0] is data:
        return dict(_HDR_METADATA[key][1])
    found = {"master_display": None, "max_cll": None}
    index = info.get("video_index")
    streams = data.get("streams") if isinstance(data, dict) else None
    for stream in streams if isinstance(streams, list) else []:
        if isinstance(stream, dict) and stream.get("index") == index:
            _hdr_side_data(stream.get("side_data_list"), found)
    if None in found.values():
        cmd = [ffprobe_path(), "-v", "quiet", "-print_format", "json", "-select_streams",
               "v:0" if index is None else str(index), "-show_frames", "-read_intervals", "%+#1",
               "-show_entries", "frame=side_data_list", path]
        proc = run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=PROBE_TIMEOUT)
        try:
            frames = json.loads(proc.stdout).get("frames") if proc.returncode == 0 else None
        except (ValueError, AttributeError, TypeError):
            frames = None
        if isinstance(frames, list) and frames and isinstance(frames[0], dict):
            _hdr_side_data(frames[0].get("side_data_list"), found)
    if data is not None:
        _HDR_METADATA[key] = (data, dict(found))
    return found


# ----------------------------------------------------------------------------
# frame and temporal effects
# ----------------------------------------------------------------------------

def effect_info(vfx, effect_id):
    """kind, layers and param ids of an effect, from `vfx list --json --all` (the hidden self-test
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
                    "layers": list(entry.get("layers") or []),
                    "params": [p["id"] for p in entry.get("params") or [] if isinstance(p, dict) and "id" in p]}
    fail(f"unknown effect `{effect_id}`, `vfx list` shows the registered ones", kind="unsupported", stage="vfx")


def check_layer_options(args, einfo):
    """--layer and --frames against the kind of the effect, with a clear message when they do not fit."""
    if einfo["kind"] != "temporal":
        used = [flag for flag, given in (("--layer", args.layer), ("--frames", args.frames),
                                         ("--device", getattr(args, "device", None))) if given]
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
    """probe every --layer video up front, so a missing file or a file without picture fails before any work.
    an HDR layer is read as stored, with its own matrix (color_settings), like any other."""
    layers = []
    for layer_id, path in args.layer:
        li = require_input(path)
        if not li["has_video"]:
            fail(f"layer {layer_id}: {path} has no video stream", kind="bad_input")
        if not li["rate"] or not li["width"] or not li["height"]:
            fail(f"layer {layer_id}: cannot read the frame size and frame rate of {path}", kind="bad_input")
        layers.append({"id": layer_id, "video": os.path.abspath(path), "info": li})
    return layers


def exact_frames(info):
    """the frames the input decodes to when the container says so for a constant frame rate, else None.
    a variable frame rate input is resampled while decoding, so its count is only an estimate."""
    if info["nb_frames"] and not info["vfr"]:
        return info["nb_frames"]
    return None


# ----------------------------------------------------------------------------
# the work geometry: the frame size and rate the run renders at
# ----------------------------------------------------------------------------

# the two params through which an effect reads its clip at a rate of its own (retime, retime-fgbg, vectors-in)
RATE_PARAMS = ("in_fps_is_out_fps", "input_frame_rate")


def output_frames(frames, source_rate, rate):
    """the output frames at `rate` that start inside `frames` frames at `source_rate`: ceil(frames * rate /
    source_rate) in exact integers, as vfx counts its default --frames under --clip-fps. each of them shows a
    frame of the input, and at the input's own rate the count is `frames` itself."""
    num = frames * rate.numerator * source_rate.denominator
    return -(-num // (rate.denominator * source_rate.numerator))


def work_geometry(args, info, einfo=None):
    """the geometry the run works at, as a dict: width and height, the frame size the decode gives vfx;
    rate, the exact output rate; frames, the count the input decodes to (the clip vfx reads, None when it is
    not known); source, the input's own geometry when the output's differs from it (None when it does not);
    auto_params, the --param values the run adds for the effect, whose param ids `einfo` lists; rounding,
    how a rate change counts frames (None without one); clip_rate, the rate of the decoded clip, which vfx
    reads it at; out_frames, the default length of the output in output frames (None when not known).

    --size and --fps set the output's size and rate; a value equal to the input's (the average rate of a
    variable rate input) changes nothing, and the geometry is then the input's: source None, clip_rate the
    rate and out_frames the frames. otherwise source holds the input's width, height, fps (its exact rate,
    the one the footage plays at) and frames (its own count, None when not known). a temporal effect (einfo)
    gets the input decoded at its own rate, read by time through vfx --clip-fps, so nothing is resampled
    away on the way in: clip_rate is the input's rate and frames its count. a frame effect cannot read by
    time, so the decode converts it, each output frame taking the input frame that covers its start:
    clip_rate is the output rate and frames the output frames. after a rate change out_frames is the output
    frames that start inside the input (output_frames, rounding "up"), and an effect that lists both
    RATE_PARAMS gets in_fps_is_out_fps=0 and input_frame_rate=<the input's rate> (%.10g, so 30000/1001 is
    29.97002997) when the user gave neither: vfx refuses an id given twice."""
    w, h = display_size(info)
    rate, frames = info["rate"], exact_frames(info)
    size = tuple(getattr(args, "size", None) or (w, h))
    fps = getattr(args, "fps", None) or rate
    if size == (w, h) and fps == rate:
        return {"width": w, "height": h, "rate": rate, "frames": frames, "source": None, "auto_params": [],
                "rounding": None, "clip_rate": rate, "out_frames": frames}
    temporal = bool(einfo) and einfo.get("kind") == "temporal"
    converted = fps != rate
    given = {p.partition("=")[0] for p in getattr(args, "param", None) or []}
    auto = []
    if temporal and converted and set(RATE_PARAMS) <= set(einfo.get("params") or []) and not given & set(RATE_PARAMS):
        auto = ["in_fps_is_out_fps=0", f"input_frame_rate={float(rate):.10g}"]
    out = frames if frames is None or not converted else output_frames(frames, rate, fps)
    clip_rate = rate if temporal else fps
    return {"width": size[0], "height": size[1], "rate": fps, "frames": frames if temporal else out,
            "source": {"width": w, "height": h, "fps": rate, "frames": frames}, "auto_params": auto,
            "rounding": "up" if converted else None, "clip_rate": clip_rate, "out_frames": out}


def output_length(geo):
    """the default length of the output in output frames: out_frames, or for a geometry without it (a
    caller from before it) the frames."""
    return geo.get("out_frames", geo["frames"])


def layer_rates(args, layers=None):
    """(id, rate) of every --layer, from the plan layers' probe facts (`layers`) or, without them, from a
    probe of each layer video (the preflight runs before the plan exists)."""
    if layers is not None:
        return [(layer["id"], layer["info"]["rate"]) for layer in layers]
    return [(layer_id, (probe(video) or {}).get("rate")) for layer_id, video in args.layer]


def vfx_rate_flags(args, geo, layers=None):
    """the vfx flags that read the input and the layers at their own rates (vfx apply --clip-fps and
    --layer-fps), the same for the preflight and the run: in a run whose geometry is not the input's,
    --clip-fps when the decoded clip's rate differs from the output's and --layer-fps ID=R for each layer
    whose own rate does (layer_rates). none for a run at the input's geometry, whose layers are resampled
    to it as before."""
    if geo.get("source") is None:
        return []
    words = []
    if geo["clip_rate"] != geo["rate"]:
        words += ["--clip-fps", rate_text(geo["clip_rate"])]
    for layer_id, rate in layer_rates(args, layers):
        if rate and rate != geo["rate"]:
            words += ["--layer-fps", f"{layer_id}={rate_text(rate)}"]
    return words


def decode_filters(args, info, geo):
    """the -vf of the decode, as a list: the input in RGB at the work size, read with its own matrix
    (color_settings) so that the encode can take it back the same way."""
    matrix = color_settings(info)[0]
    graph = []
    if info["vfr"]:
        # variable frame rate in, constant out: keep the duration instead of the timestamps
        graph.append(f"fps={rate_text(info['rate'])}")
    source = geo.get("source")
    if source is not None and geo["clip_rate"] != source["fps"]:
        # a frame effect at another rate: the input's frames renumbered at its exact rate, then each output
        # frame takes the one that covers its start (fps rounding the input's times up), as a host shows
        # footage of another rate in a composition
        r = source["fps"]
        graph += [f"settb={r.denominator}/{r.numerator}", "setpts=N", f"fps={rate_text(geo['rate'])}:round=up"]
    if display_size(info) != (geo["width"], geo["height"]):
        graph.append(f"scale={geo['width']}:{geo['height']}:in_color_matrix={matrix}")
    else:
        graph.append(f"scale=in_color_matrix={matrix}")
    return graph


def layer_filters(args, layer_info, geo, size=None):
    """the -vf of one layer decode, as a list: the layer in RGB at the work rate and at the frame size vfx
    reads (`size` when a declared pre-stage changed it, else the work size), stretched when its own size
    differs, read with the layer's own matrix. in a run whose geometry is not the input's (geo["source"])
    the layer keeps its own rate, a variable one made constant at its average, and vfx reads it by time
    (vfx_rate_flags)."""
    w, h = size or (geo["width"], geo["height"])
    rate = geo["rate"]
    matrix = color_settings(layer_info)[0]
    graph = []
    if geo.get("source") is not None:
        if layer_info["vfr"]:
            graph.append(f"fps={rate_text(layer_info['rate'])}")
    elif layer_info["vfr"] or layer_info["rate"] != rate:
        graph.append(f"fps={rate_text(rate)}")
    if display_size(layer_info) != (w, h):
        graph.append(f"scale={w}:{h}:in_color_matrix={matrix}")
    else:
        graph.append(f"scale=in_color_matrix={matrix}")
    return graph


def effect_params(args, geo=None):
    """the --param values of vfx, for its preflight and its run: the user's in the order given, then the
    ones the work geometry adds (auto_params, never an id the user gave: vfx refuses an id twice)."""
    return list(args.param) + list((geo or {}).get("auto_params") or [])


# ----------------------------------------------------------------------------
# the audio: what the output carries of the input's sound
# ----------------------------------------------------------------------------

AUDIO_MODES = ("copy", "retime", "drop")
# --audio retime: the script that plays each stream along the effect's time map, run as `python -I`
AUDIO_MAP = os.path.join(HERE, "audio_map.py")
# the aac encode of a retimed stream: the source's bit rate, else this much a channel, and at most this
# sample rate (the encoder's highest)
AAC_BITS_PER_CHANNEL = 96000
AAC_MAX_RATE = 96000
# a retimed stream lasts as long as the video within the larger of one frame and this many seconds
AUDIO_LENGTH_TOLERANCE = 0.05


def audio_mode(args):
    """the --audio mode of a run: copy (the default, and for a caller that predates the flag), retime or
    drop."""
    return getattr(args, "audio", None) or "copy"


def require_audio_map(args, info):
    """--audio retime of an input with audio runs tools/audio_map.py: without it the run is refused before
    any work, as a missing ffmpeg is (tool_missing)."""
    if audio_mode(args) == "retime" and info.get("has_audio") and not os.path.isfile(AUDIO_MAP):
        fail(f"--audio retime needs {AUDIO_MAP}, which is missing; restore it from the repository or pass "
             "--audio copy or --audio drop", kind="tool_missing", stage="audio-retime")


def _start_time(value):
    """a stream's start_time from ffprobe's text in seconds (it can be negative), None for 'N/A' and junk."""
    try:
        t = float(value)
    except (TypeError, ValueError):
        return None
    return t if t == t and abs(t) != float("inf") else None


def _tag(stream, name):
    """a stream tag whatever the case its container writes it in (language, LANGUAGE)."""
    for key, value in (stream.get("tags") or {}).items():
        if key.lower() == name:
            return value
    return None


def stream_duration(stream):
    """a stream's own duration in seconds: ffprobe's duration, else the DURATION tag matroska writes
    (HH:MM:SS.fraction); None when neither says."""
    duration = _seconds(stream.get("duration"))
    if duration is not None:
        return duration
    match = re.fullmatch(r"([0-9]+):([0-9]{2}):([0-9]{2}(\.[0-9]+)?)", (_tag(stream, "duration") or "").strip())
    if match:
        return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))
    return None


def audio_streams(info):
    """every audio stream of a probed file, in order, from the raw ffprobe json of its probe (_PROBE_DATA):
    dicts with index, codec, channels, sample_rate, layout, bit_rate, start_time, duration, language and
    title (None where ffprobe says nothing). [] for a file that was not probed."""
    data = _PROBE_DATA.get((info or {}).get("path"))
    streams = []
    for st in (data or {}).get("streams", []):
        if st.get("codec_type") != "audio":
            continue
        layout = st.get("channel_layout")
        streams.append({
            "index": st.get("index"), "codec": st.get("codec_name"), "channels": _count(st.get("channels")),
            "sample_rate": _count(st.get("sample_rate")),
            "layout": layout if layout and layout != "unknown" and "channels" not in layout else None,
            "bit_rate": _count(st.get("bit_rate")), "start_time": _start_time(st.get("start_time")),
            "duration": stream_duration(st), "language": _tag(st, "language"),
            # mp4 keeps a stream's title as its name
            "title": _tag(st, "title") or _tag(st, "name"),
        })
    return streams


def _video_start(info):
    """the start_time of the video stream the decode reads, 0 when ffprobe gives none."""
    for st in (_PROBE_DATA.get(info.get("path")) or {}).get("streams", []):
        if st.get("index") == info.get("video_index"):
            return _start_time(st.get("start_time")) or 0.0
    return 0.0


# the demuxers whose start time ffmpeg takes again from the streams a run reads (AVFMT_TS_DISCONT): the
# decode reads the video alone, so it starts at the video's own start_time and puts no frame in front of it
TS_DISCONT_FORMATS = ("mpegts", "mpegtsraw", "mpeg", "hls")


def _video_stream(info):
    """the raw ffprobe json of the video stream the decode reads ({} when there is none)."""
    for st in (_PROBE_DATA.get((info or {}).get("path")) or {}).get("streams", []):
        if st.get("index") == info.get("video_index"):
            return st
    return {}


def video_seconds(info):
    """how long the decode of the video lasts, in seconds from the file's start time (where the raw frames
    start, decoded_clip): the video stream's own duration plus its start after the file's, or the matroska
    DURATION tag, which is already where the stream ends; else the probe's video duration, else the
    file's. None when nothing says. never the file's when the stream says: the file lasts as long as its
    longest stream, which can be the audio."""
    stream = _video_stream(info)
    own = _seconds(stream.get("duration"))
    start = _start_time((((_PROBE_DATA.get(info.get("path")) or {}).get("format")) or {}).get("start_time")) or 0.0
    if own is not None:
        return own + max(0.0, (_start_time(stream.get("start_time")) or 0.0) - start)
    tagged = stream_duration(stream)
    if tagged is not None:
        return max(0.0, tagged - start)
    return info.get("video_duration") or info.get("duration")


def video_packets(info):
    """(pts, duration) of every packet of the video stream in its time base, in file order, from one
    ffprobe pass that reads the packets and decodes nothing (a duration it does not give is 0). None when
    ffprobe cannot read them or a packet has no pts."""
    cmd = [ffprobe_path(), "-v", "error", "-select_streams", str(info["video_index"]), "-show_entries",
           "packet=pts,duration", "-of", "csv=p=0", info["path"]]
    proc = run(cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT * 20)
    if proc.returncode != 0:
        return None
    packets = []
    for line in proc.stdout.split():
        pts, _, duration = line.strip(",").partition(",")
        try:
            packets.append((int(pts), int(duration) if duration.lstrip("-").isdigit() else 0))
        except ValueError:
            return None
    return packets or None


def _rescale(a, b, c):
    """a * b / c rounded half away from zero, in integers (ffmpeg's av_rescale_rnd with AV_ROUND_NEAR_INF)."""
    q, r = divmod(abs(a * b), c)
    q += 2 * r >= c
    return q if a * b >= 0 else -q


def _f32(x):
    """x rounded to a 32 bit float, as C does when it hands a double to llrintf."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def cfr_decode(packets, time_base, start_us, rate):
    """(frames, lead) of ffmpeg's constant rate decode of a video stream to raw frames at `rate`: how many
    frames it writes and how many it puts in front of the clip's first one, worked out from the packets
    ((pts, duration) in `time_base`) with the arithmetic of ffmpeg's video sync (fps_mode cfr, the default
    of a raw output): the timestamps start at the file's start time (`start_us`, microseconds), each frame's
    is taken to the output time base with 2^extra bits more precision and moved off an exact midpoint,
    a frame is written llrintf(delta) times when its end lies more than 1.1 frames past the next output
    frame, never when it lies more than 1.1 frames before it, and the end of the stream repeats the last
    frame the median of the last three times a previous frame was repeated. the decode then shows clip frame
    s at output frame s + lead for the run of frames this holds for most often, so the lead is that shift."""
    out_num, out_den = rate.denominator, rate.numerator
    extra = max(0, min(16, 29 - (out_den.bit_length() - 1)))
    offset = _rescale(-start_us, time_base.denominator, 1000000 * time_base.numerator)
    seconds = time_base.numerator / time_base.denominator
    written, shifts, history = 0, {}, [0, 0, 0]
    for index, (pts, duration) in enumerate(sorted((p + offset, d) for p, d in packets)):
        at = _rescale(pts, time_base.numerator * (out_den << extra), out_num * time_base.denominator) / (1 << extra)
        if at != round(at):
            at += (1.0 if at > 0 else -1.0) / (1 << 17)
        length = duration * seconds / (out_num / out_den)
        delta0 = at - written
        delta = delta0 + length
        if delta0 < 0 < delta:
            delta0 = 0.0
        copies, before = 1, 0
        if delta < -1.1:
            copies = 0
        elif delta > 1.1:
            copies = round(_f32(delta))
            if delta0 > 1.1:
                before = round(delta0 - 0.6)
        if index == 0:
            # nothing came before the first frame: every copy is the frame, it stands at the last one
            shift = copies - 1
        else:
            shift = written + before - index
        if copies > (before if index else 0):
            shifts[shift] = shifts.get(shift, 0) + 1
        history = [before] + history[:2]
        written += copies
    written += sorted(history)[1]
    lead = max(sorted(shifts.items()), key=lambda kv: kv[1])[0] if shifts else 0
    return written, max(0, lead)


def decoded_clip(info, geo):
    """(frames, lead) of the clip the decode writes for --audio retime and drop: the frames it writes (None
    when that is not known) and how many of them come before the input's first frame, at the rate of the
    decoded clip. ffmpeg starts the raw frames at the file's start time, so a video stream that starts after
    another stream (an edit list, a delay) gets copies of its first frame in front (cfr_decode, from the
    packets the container holds). a frame effect at another rate has its frames renumbered from 0 and none in
    front; a variable rate input is resampled by the fps filter first, which rounds the first frame to the
    nearest output frame and leaves the count unknown."""
    rate = geo.get("clip_rate", geo["rate"])
    source = geo.get("source")
    stream = _video_stream(info)
    if source is not None and geo["clip_rate"] != source["fps"]:
        frames = geo["frames"]
        if frames is None and not info["vfr"]:
            packets = video_packets(info)
            frames = output_frames(len(packets), source["fps"], geo["rate"]) if packets else None
        return frames, 0
    formats = (info.get("container") or "").split(",")
    start = _start_time((((_PROBE_DATA.get(info.get("path")) or {}).get("format")) or {}).get("start_time")) or 0.0
    first = _start_time(stream.get("start_time"))
    if any(f in TS_DISCONT_FORMATS for f in formats):
        start = first if first is not None else start
    if info["vfr"]:
        x = ((first or 0.0) - start) * float(rate)
        return None, max(0, int(x + 0.5))
    base = _fraction(stream.get("time_base"))
    packets = video_packets(info) if base else None
    if not packets:
        # the first frame alone, as long as a frame: the container's count, when it gives one, plus the lead
        x = ((first or 0.0) - start) * float(rate)
        lead = round(_f32(x + 1.0)) - 1 if x + 1.0 > 1.1 else 0
        return (None if geo["frames"] is None else geo["frames"] + lead), lead
    return cfr_decode(packets, base, round(start * 1000000), rate)


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


def plan_audio(args, info, plan, frames=None, geo=None, chain=None):
    """the audio part of a plan for an output of `frames` frames at the work rate: plan["audio"], what the
    copy does to the audio by lengths alone (audio_plan), and plan["shortest"], whether the encode cuts it
    at the end of the video. a count of None or of the input's own frames keeps what the plan says so far
    (at first: nothing, the output lasts as long as the input). returns the plan.

    --audio retime and drop plan once (retime_plan, with the chained stage geometry `chain` when the
    caller has it) and add the audio_* keys; copy adds none."""
    mode = audio_mode(args)
    if mode != "copy":
        if "audio_mode" not in plan:
            retime_plan(args, info, plan, mode, geo or work_geometry(args, info), chain)
        return plan
    if frames not in (None, plan["input_frames"]):
        plan["audio"] = audio_plan(info, frames, geo["rate"] if geo else info["rate"])
        plan["shortest"] = plan["audio"] == "shortest"
    return plan


def _count_text(count):
    return "an unknown count" if count is None else str(count)


def retime_plan(args, info, plan, mode, geo, chain=None):
    """the plan of --audio retime or drop. drop and an input without audio plan no stream. retime
    plays every audio stream of the input, in order, along the effect's time map over the output's frames N,
    reading the clip of C frames the decode writes (decoded_clip): per stream the decode to raw pcm_f32le and
    the retimed file in the scratch folder, and the encode, aac at the source's bit rate (else 96 kb/s a
    channel, at most 96 kHz) or flac with --lossless. N is --frames, else the frames vfx renders: the input's
    count when the container gives it (a temporal effect is told it), else the frames the decode writes, which
    the packets of the input tell (an unknown count, a variable rate: ceil(duration * fps) + 1 frames, cut by
    -shortest). a video stream that starts after another stream gets `lead` copies of its first frame in front
    of the clip, so clip frame p shows the input at p - lead and the offset of a stream is its start_time
    minus the video's plus lead / the clip rate. under frames/2 a declared stage that changes the frames vfx
    reads or the output's is a usage error: the map would no longer match the picture; `chain` is the chained
    geometry build_stages already worked out (None: it is worked out here). plan keys: audio_mode,
    audio_codec, audio_streams (dicts), audio_frames (N), audio_clip_frames (C), audio_fps (the output rate),
    audio_clip_fps (the rate of the decoded clip, --clip-fps of audio_map when it differs), audio_params
    (vfx's --param values), audio_length (the seconds the output's video lasts, for the verify; None without
    audio in the input); audio is "same", and shortest is True for an unknown N, when there is a stream to
    retime."""
    plan.update(audio_mode=mode, audio_codec=None, audio_streams=[], audio_frames=None, audio_clip_frames=None,
                audio_fps=geo["rate"], audio_clip_fps=geo.get("clip_rate", geo["rate"]),
                audio_params=effect_params(args, geo), audio=None, shortest=False, audio_length=None)
    if not info["has_audio"]:
        return
    sources = audio_streams(info) if mode == "retime" else []
    rate = geo["rate"]
    clip_rate = geo.get("clip_rate", rate)
    temporal = plan["kind"] == "temporal"
    expected = plan["frames"] if temporal else plan["input_frames"]
    templates = list(getattr(args, "pre_stage", None) or []) + list(getattr(args, "post_stage", None) or [])
    if sources and any(getattr(t, "geometry", "preserve") == "declared" for t in templates):
        if chain is None or "out" not in chain:
            chain = {}
            build_stages(args, info, chain, geo)
        if chain["vfx"]["frames"] != geo["frames"]:
            fail(f"--audio retime: the declared pre-stages change the frames vfx reads from "
                 f"{_count_text(geo['frames'])} to {_count_text(chain['vfx']['frames'])}, so the clip no longer "
                 "lines up with the input's audio; pass --audio copy or --audio drop", kind="usage")
        if chain["out"]["frames"] != expected:
            fail(f"--audio retime: the declared post-stages change the frames of the output from "
                 f"{_count_text(expected)} to {_count_text(chain['out']['frames'])}, so the output no longer "
                 "follows the effect's time map; pass --audio copy or --audio drop", kind="usage")
    decoded, lead = decoded_clip(info, geo)
    if args.frames is not None:
        frames = args.frames
    elif temporal:
        # vfx renders --frames when the plan knows the count, else the output frames that start in the clip
        frames = plan["frames"] if plan["frames"] is not None else (
            None if decoded is None else output_frames(decoded, clip_rate, rate))
    else:
        # a frame effect writes every frame the decode gives it
        frames = decoded
    duration = video_seconds(info)
    plan["audio_length"] = frames / float(rate) if frames is not None else duration
    if not sources:
        return
    if (frames is None or decoded is None) and duration is None:
        fail("--audio retime needs the length of the input, which ffprobe does not give", kind="bad_input")
    shortest = frames is None
    if frames is None:
        frames = int(-(-(duration * float(rate)) // 1)) + 1
    clip = decoded if decoded is not None else max(1, int(duration * float(clip_rate) + 0.5))
    codec = "flac" if getattr(args, "lossless", False) else "aac"
    start = _video_start(info)
    for number, st in enumerate(sources, 1):
        if not st["sample_rate"]:
            fail(f"--audio retime: cannot read the sample rate of audio stream {st['index']}", kind="bad_input")
        pcm = os.path.join(plan["scratch"], f"audio_{number}.f32")
        retimed = os.path.join(plan["scratch"], f"audio_{number}_retimed.f32")
        channels = st["channels"] or 2
        plan["audio_streams"].append(dict(
            st, channels=channels, number=number, pcm=pcm, retimed=retimed,
            offset=(st["start_time"] or 0.0) - start + lead / float(clip_rate),
            bit_rate=(st["bit_rate"] or AAC_BITS_PER_CHANNEL * channels) if codec == "aac" else None,
            encode_rate=min(st["sample_rate"], AAC_MAX_RATE) if codec == "aac" else st["sample_rate"]))
        plan["temp_files"].append({"path": pcm, "what": f"audio stream {st['index']}, decoded to raw pcm_f32le"})
        plan["temp_files"].append({"path": retimed, "what": f"audio stream {st['index']}, retimed to the effect's "
                                                            "time map (tools/audio_map.py)"})
    plan.update(audio_codec=codec, audio_frames=frames, audio_clip_frames=clip, audio="same", shortest=shortest)


def audio_parts(args, info, plan):
    """the audio of the encode as (inputs, maps, codec): the inputs come after the raw video (input 0),
    the maps after its map and the codec after its options. --audio copy copies every audio stream of the
    input untouched, and `1:a?` maps none of an input without audio. `plan` is None for a caller that
    predates it. --audio retime reads each retimed stream as raw f32le at its rate and layout (inputs 1, 2,
    ...), maps them in the input's order and encodes them aac at the planned bit rate or flac, with the
    language and title of the source stream; drop and an input without audio have no part."""
    if not plan or plan.get("audio_mode") is None:
        return ["-i", args.input], ["-map", "1:a?"], ["-c:a", "copy"]
    streams = plan["audio_streams"]
    if not streams:
        return [], [], []
    inputs, maps, codec = [], [], ["-c:a", plan["audio_codec"]]
    for k, st in enumerate(streams):
        layout = ["-ch_layout", st["layout"]] if st["layout"] else ["-ac", str(st["channels"])]
        inputs += ["-f", "f32le", "-ar", str(st["sample_rate"])] + layout + ["-i", st["retimed"]]
        maps += ["-map", f"{k + 1}:a:0"]
        if st["bit_rate"]:
            codec += [f"-b:a:{k}", str(st["bit_rate"])]
        if st["encode_rate"] != st["sample_rate"]:
            codec += [f"-ar:a:{k}", str(st["encode_rate"])]
        for tag in ("language", "title"):
            if st[tag]:
                codec += [f"-metadata:s:a:{k}", f"{tag}={st[tag]}"]
    return inputs, maps, codec


def audio_commands(args, info, plan):
    """the audio steps of a plan in the order they run, each (stage, argv) with stage the name a failure
    reports (audio-decode, audio-retime): after the layer decodes and before the pipe, whose encode reads
    what they write. --audio copy has none, its encode reads the input itself. --audio retime decodes every
    stream to raw pcm_f32le in one ffmpeg, then runs tools/audio_map.py render once a stream."""
    streams = (plan or {}).get("audio_streams")
    if not streams:
        return []
    decode = [ffmpeg_path(), "-v", "error", "-nostdin", "-y", "-i", args.input]
    for st in streams:
        decode += ["-map", f"0:{st['index']}", "-c:a", "pcm_f32le", "-f", "f32le", st["pcm"]]
    steps = [("audio-decode", decode)]
    for st in streams:
        cmd = [sys.executable, "-I", AUDIO_MAP, "render", "--effect", args.effect, "--fps", rate_text(plan["audio_fps"]),
               "--frames", str(plan["audio_frames"]), "--clip-frames", str(plan["audio_clip_frames"])]
        if plan.get("audio_clip_fps") not in (None, plan["audio_fps"]):
            cmd += ["--clip-fps", rate_text(plan["audio_clip_fps"])]
        for p in plan["audio_params"]:
            cmd += ["--param", p]
        cmd += ["--rate", str(st["sample_rate"]), "--channels", str(st["channels"]), "--offset", repr(st["offset"]),
                "--in", st["pcm"], "--out", st["retimed"]]
        steps.append(("audio-retime", cmd))
    return steps


def run_audio(steps, scratch):
    """run the steps of audio_commands one after the other, each with its stderr in a log in scratch; a
    step that fails ends the run as stage_failed at its stage, with the last lines of its log."""
    for index, (stage, cmd) in enumerate(steps, 1):
        log = os.path.join(scratch, f"audio{index}.log")
        with open(log, "w", encoding="utf-8") as handle:
            proc = run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=handle)
        if proc.returncode != 0:
            message = f"{stage}: {os.path.basename(str(cmd[0]))} exited with code {proc.returncode}"
            lines = "\n".join("  " + t for t in tail(log))
            fail(message + ("\n" + lines if lines else ""), kind="stage_failed", stage=stage,
                 exit_code=proc.returncode, tail=tail(log, STAGE_TAIL), summary=message)


def make_plan(args, info, einfo, layers, scratch, geo=None, chain=None):
    """what the temporal path adds to the frame path: frames, layer temp files and the audio decision.
    `scratch` only names paths here, nothing is created. `geo` is the work geometry (work_geometry) and
    `chain` the stage geometry build_stages chained (frames/2), which the audio plan reads."""
    geo = geo or work_geometry(args, info, einfo)
    plan = {"kind": einfo["kind"], "scratch": scratch, "layers": [], "frames": None,
            "input_frames": output_length(geo), "audio": None, "shortest": False, "temp_files": []}
    if einfo["kind"] != "temporal":
        return plan_audio(args, info, plan, None, geo, chain)
    fmt = PIX_FMT[args.depth]
    for layer in layers:
        temp = os.path.join(scratch, f"layer_{layer['id']}.{fmt}")
        entry = {"id": layer["id"], "video": layer["video"], "temp": temp, "info": layer["info"]}
        what = f"layer {layer['id']}, decoded to raw {fmt}"
        if expands(args, layer["info"]):
            # an 8 bit layer at 16 or 32 bpc: decoded to rgba, then expanded exactly into the raw file vfx reads
            entry["decoded"] = os.path.join(scratch, f"layer_{layer['id']}.rgba")
            plan["temp_files"].append({"path": entry["decoded"], "what": f"layer {layer['id']}, decoded to raw rgba, "
                                                                          f"expanded into raw {fmt}, then removed"})
            what = f"layer {layer['id']}, raw {fmt} expanded exactly from its rgba decode"
        plan["layers"].append(entry)
        plan["temp_files"].append({"path": temp, "what": what})
    plan["temp_files"].append({"path": scratch, "what": "scratch directory: the layer clips, the logs and the "
                                                        "spool file vfx makes of the piped input"})
    # without --frames vfx renders the frames it was fed, so the output is as long as the decode
    # whatever the container said; with --frames the length is the user's
    plan["frames"] = args.frames if args.frames is not None else plan["input_frames"]
    return plan_audio(args, info, plan, args.frames, geo, chain)


def build_layer_commands(args, info, plan, size=None, geo=None):
    """one ffmpeg per layer: decode it to a raw file of the effect's pixel format, size and frame rate
    (layer_filters); an 8 bit layer at 16 or 32 bpc to an rgba file that decode_layers expands. `size` is the
    frame size vfx gets when a declared pre-stage changed it (frames/2), `geo` the work geometry."""
    geo = geo or work_geometry(args, info)
    commands = []
    for layer in plan["layers"]:
        li = layer["info"]
        fmt = "rgba" if layer.get("decoded") else PIX_FMT[args.depth]
        commands.append([ffmpeg_path(), "-v", "error", "-nostdin", "-y", "-i", layer["video"],
                         "-map", f"0:{li['video_index']}", "-an", "-sn", "-dn",
                         "-vf", ",".join(layer_filters(args, li, geo, size)), "-pix_fmt", fmt, "-f", "rawvideo",
                         layer.get("decoded") or layer["temp"]])
    return commands


def decode_layers(commands, plan, args, info, scratch, size=None, geo=None):
    """run the layer decodes one after the other; each leaves a raw file of whole frames or fails with ffmpeg's last lines.
    the frames are then prepared as the input's are (stream_wire): an 8 bit layer at 16 or 32 bpc is expanded from its
    rgba file into the raw file vfx reads, and a layer without an alpha channel is made opaque."""
    geo = geo or work_geometry(args, info)
    w, h = size or (geo["width"], geo["height"])
    for layer, cmd in zip(plan["layers"], commands):
        wire = stream_wire(args, layer["info"], w, h)
        decoded = layer.get("decoded") or layer["temp"]
        frame_bytes = wire.in_bytes if wire else w * h * BYTES_PER_PIXEL[args.depth]
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
            size = os.path.getsize(decoded)
        except OSError:
            size = 0
        if size == 0 or size % frame_bytes:
            fail(f"layer {layer['id']}: {layer['video']} decoded to {size} bytes, "
                 f"which is not a whole number of {w}x{h} frames", kind="bad_input", stage="layer-decode")
        if wire:
            wire_file(decoded, wire, layer["temp"] if layer.get("decoded") else None)


def rate_text(rate):
    return f"{rate.numerator}/{rate.denominator}"


# ----------------------------------------------------------------------------
# the frames vfx is given: 8 bit sources expanded exactly, opaque footage white
# ----------------------------------------------------------------------------

# the float 1.0 of a gbrapf32le alpha sample
ONE_F32 = struct.pack("<f", 1.0)
# byte k (little endian) of the f32 of v / 255 for every 8 bit value v: the translate tables of the 32 bpc expansion.
# in binary v / 255 repeats the 8 bits of v without end, so it never lies on a tie of f32 rounding, and rounding the
# double quotient to f32 gives the f32 quotient itself
F32_OF_BYTE = tuple(bytes(struct.pack("<f", v / 255)[k] for v in range(256)) for k in range(4))
# {pixel format name: (carries alpha, bits of its deepest sample or None)} from ffprobe, read once a run
_PIX_FORMATS = {}


def pixel_formats():
    """{pixel format name: (carries alpha, bits of its deepest sample)}, from ffprobe's list of pixel formats: its
    alpha flag (a palette format is flagged, since its entries may be transparent) and the bit depths of its
    components (None when it lists none). empty when ffprobe cannot list them."""
    if "table" not in _PIX_FORMATS:
        proc = run([ffprobe_path(), "-v", "error", "-show_pixel_formats", "-print_format", "json"],
                   stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=PROBE_TIMEOUT)
        table = {}
        try:
            for entry in json.loads(proc.stdout or "{}").get("pixel_formats", []):
                depths = [c.get("bit_depth") for c in entry.get("components") or []]
                depths = [d for d in depths if isinstance(d, int)]
                table[entry["name"]] = (bool((entry.get("flags") or {}).get("alpha")), max(depths) if depths else None)
        except (ValueError, KeyError, TypeError, AttributeError):
            table = {}
        _PIX_FORMATS["table"] = table
    return _PIX_FORMATS["table"]


def opaque_source(info):
    """true when a probed video stream has no alpha channel: ffprobe lists its pixel format without the alpha flag.
    vfx reads a pixel below white as partly transparent, so its frames are given with the alpha of every pixel at
    white (Wire), whatever ffmpeg's conversion left there (an 8 bit bgr0 source decodes to 65532 at 16 bpc). a real
    alpha channel, and a format ffprobe does not list, keep their own."""
    fmt = info.get("pix_fmt") if info else None
    return bool(fmt) and pixel_formats().get(fmt, (None, None))[0] is False


def expands(args, info):
    """true when a decode at --depth 16 or 32 reads a probed video stream as 8 bit rgba and expands it exactly (Wire):
    its pixel format has at most 8 bits a sample. ffmpeg's own conversion is not exact: an 8 bit white arrives as
    65280 or 65283 at 16 bpc and as 0.9961 at 32 bpc, so a white matte would read as partly transparent. a deeper
    source, and a format ffprobe does not list, are decoded to the raw format of --depth."""
    fmt = info.get("pix_fmt") if info else None
    bits = pixel_formats().get(fmt, (None, None))[1] if fmt else None
    return args.depth in (16, 32) and bits is not None and bits <= 8


def decode_format(args, info):
    """the -pix_fmt of the decode of a probed video stream (the input or a layer): rgba when it expands, else the
    raw format of --depth."""
    return "rgba" if expands(args, info) else PIX_FMT[args.depth]


class Wire:
    """turns whole decoded frames into the frames vfx is given, at the size and in the order they came. with `expand`
    each 8 bit rgba frame becomes a frame of the --depth format with every sample exact: 16 bpc v * 257 (each byte
    twice, low and high, so 255 is 65535), 32 bpc v / 255 as f32 in the planes G, B, R, A of gbrapf32le, built with a
    translate table per float byte and extended slices. with `opaque` the alpha of every pixel is white (255, 65535
    or 1.0), else a real alpha expands like the colour. `in_bytes` is a decoded frame, `frame_bytes` a given one."""

    def __init__(self, depth, width, height, expand=False, opaque=False):
        self.depth, self.pixels, self.expand, self.opaque = depth, width * height, expand, opaque
        self.frame_bytes = self.pixels * BYTES_PER_PIXEL[depth]
        self.in_bytes = self.pixels * 4 if expand else self.frame_bytes
        self._out = bytearray()

    def output(self, size):
        """the buffer an expansion writes into, kept from one call to the next of the same size: what a call
        returned is only valid until the next one."""
        if len(self._out) != size:
            self._out = bytearray(size)
        return self._out

    def __call__(self, block):
        """`block` is a bytearray of whole decoded frames; returns the frames as given: the Wire's output buffer when
        they expand (valid until the next call), else `block` itself."""
        if self.opaque and (self.expand or self.depth == 8):
            block[3::4] = b"\xff" * (len(block) // 4)
        if not self.expand:
            if self.opaque and self.depth == 16:
                white = b"\xff" * (len(block) // 8)
                block[6::8] = white
                block[7::8] = white
            elif self.opaque and self.depth == 32:
                plane = self.pixels * 4
                one = ONE_F32 * self.pixels
                for start in range(0, len(block), self.frame_bytes):
                    block[start + 3 * plane:start + self.frame_bytes] = one
            return block
        if self.depth == 16:
            out = self.output(2 * len(block))
            out[0::2] = block
            out[1::2] = block
            return out
        n = self.pixels
        out = self.output(4 * len(block))
        for start in range(0, len(block), self.in_bytes):
            frame = block[start:start + self.in_bytes]
            at = 4 * start
            for channel in (1, 2, 0, 3):
                plane = frame[channel::4]
                for k in range(4):
                    out[at + k:at + 4 * n:4] = plane.translate(F32_OF_BYTE[k])
                at += 4 * n
        return out


def wire_file(path, wire, out=None):
    """Wire over a raw file of whole decoded frames, a few frames at a time: in place, or into the file `out` when the
    frames expand (the decoded file is removed afterwards)."""
    step = wire.in_bytes * max(1, (16 << 20) // wire.in_bytes)
    if out is None:
        with open(path, "r+b") as handle:
            at = 0
            while True:
                handle.seek(at)
                block = bytearray(handle.read(step))
                whole = len(block) - len(block) % wire.in_bytes
                if not whole:
                    return
                del block[whole:]
                handle.seek(at)
                handle.write(wire(block))
                at += whole
    with open(path, "rb") as source, open(out, "wb") as sink:
        while True:
            block = bytearray(source.read(step))
            whole = len(block) - len(block) % wire.in_bytes
            if not whole:
                break
            del block[whole:]
            sink.write(wire(block))
    os.remove(path)


def stream_wire(args, info, width, height):
    """the Wire of the frames decoded from a probed video stream at width x height, or None when vfx is given them
    as decoded."""
    expand, opaque = expands(args, info), opaque_source(info)
    return Wire(args.depth, width, height, expand, opaque) if expand or opaque else None


def decode_wire(args, info, geo=None):
    """the Wire of the decode's frames (at the work size), or None."""
    w, h = (geo["width"], geo["height"]) if geo else display_size(info)
    return stream_wire(args, info, w, h)


def prepare_report(args, info, plan):
    """what apply.py does to the decoded frames before vfx reads them, one entry for each stream it changes (the
    input, then the layers): the format decoded, the format given, the samples and the alpha."""
    steps = []
    streams = [("input", info)] + [("layer " + layer["id"], layer["info"]) for layer in plan.get("layers") or []]
    for name, facts in streams:
        expand, opaque = expands(args, facts), opaque_source(facts)
        if not (expand or opaque):
            continue
        rule = "v * 257" if args.depth == 16 else "v / 255 as f32"
        steps.append({"stream": name, "decoded": decode_format(args, facts), "given": PIX_FMT[args.depth],
                      "samples": rule if expand else "as decoded",
                      "alpha": "white (the source has none)" if opaque else rule})
    return steps


# the pixel formats of the libx265 encode of an HDR input: 4:2:0, and 4:4:4 for a size with an odd side
HDR_PIX_FMT, HDR_ODD_PIX_FMT = "yuv420p10le", "yuv444p10le"
# the containers whose HEVC is tagged hvc1, the tag players of those files look for
HVC1_EXTENSIONS = (".mp4", ".m4v", ".mov")


def encode_plan(args, info, out_size, destination=None):
    """the encode of a run where it differs from the one of before, as a dict; None where it does not:
    an SDR input at a size whose sides are even (libx264 CRF 18 yuv420p, or ffv1 with --lossless), so
    that nothing is added to the encode, the report or the verify.

    an HDR input (probe's hdr) encodes libx265 CRF 18 in 10 bit, tagged hvc1 in mp4, m4v and mov, with
    the tags of color_settings and its HDR metadata (hdr_metadata) written back by x265; with --lossless
    it stays RGB in ffv1, tagged with its primaries and its transfer. a size with an odd side (`out_size`,
    the size the encode reads) encodes yuv444p (yuv444p10le for HDR) at that exact size: of the chroma
    layouts only 4:4:4 has a crop unit of one pixel. --lossless takes any size as it is.
    keys: codec, pix_fmt, odd (the lossy encode is 4:4:4 for an odd side), tag (None for none), hdr and,
    for an HDR input, transfer, primaries, matrix (None in RGB), master_display and max_cll."""
    lossless = bool(getattr(args, "lossless", False))
    odd = not lossless and bool(out_size[0] % 2 or out_size[1] % 2)
    if not info.get("hdr"):
        if not odd:
            return None
        return {"codec": "libx264", "pix_fmt": "yuv444p", "odd": True, "tag": None, "hdr": False}
    matrix, space, primaries, trc = color_settings(info)
    plan = {"odd": odd, "hdr": True, "transfer": trc, "primaries": primaries}
    if lossless:
        plan.update(codec="ffv1", pix_fmt=LOSSLESS_PIX_FMT.get(args.depth), tag=None, matrix=None,
                    master_display=None, max_cll=None)
    else:
        tagged = destination and os.path.splitext(destination)[1].lower() in HVC1_EXTENSIONS
        plan.update(codec="libx265", pix_fmt=HDR_ODD_PIX_FMT if odd else HDR_PIX_FMT,
                    tag="hvc1" if tagged else None, matrix=space, **hdr_metadata(args.input, info))
    return plan


def x265_params(plan):
    """the -x265-params of an HDR encode: x265's own log quiet but for errors, and the HDR metadata the
    input has (the master display, the content light level) written back as SEI."""
    params = ["log-level=error"]
    if plan.get("master_display"):
        params.append("master-display=" + plan["master_display"])
    if plan.get("max_cll"):
        params.append("max-cll=" + plan["max_cll"])
    return ":".join(params)


def require_hdr_encoder():
    """the lossy encode of an HDR input needs libx265 with 10 bit output, 4:2:0 and 4:4:4, in this ffmpeg;
    without it the run is refused before any work (tool_missing)."""
    ffmpeg = ffmpeg_path()
    proc = run([ffmpeg, "-hide_banner", "-h", "encoder=libx265"], stdin=subprocess.DEVNULL, capture_output=True,
               text=True, timeout=PROBE_TIMEOUT)
    formats = None
    for line in (proc.stdout or "").splitlines():
        label, sep, rest = line.strip().partition(":")
        if sep and label == "Supported pixel formats":
            formats = rest.split()
    if proc.returncode != 0 or formats is None:
        fail(f"an HDR input is encoded with libx265, which {ffmpeg} does not have; use an ffmpeg built with "
             "libx265, or pass --lossless (ffv1)", kind="tool_missing", stage="encode")
    missing = [fmt for fmt in (HDR_PIX_FMT, HDR_ODD_PIX_FMT) if fmt not in formats]
    if missing:
        fail(f"an HDR input is encoded with libx265 in 10 bit, and the libx265 of {ffmpeg} cannot write "
             + " or ".join(missing) + "; use an ffmpeg whose libx265 has 10 bit, or pass --lossless (ffv1)",
             kind="tool_missing", stage="encode")


def video_encode_args(args, info, geo, out_size, destination=None):
    """the video options of the encode, after its maps: libx264 CRF 18 yuv420p in tv range, and the filter
    that takes the effect's RGB back to YUV with the matrix and tags of color_settings; --lossless keeps the
    RGB in ffv1. `out_size` is the frame size the encode reads, `destination` the output path (its container
    can call for a codec tag). an HDR input and a size with an odd side change the encode (encode_plan):
    libx265 10 bit with its tag and its HDR metadata, or yuv444p at the exact size; under --lossless an HDR
    input only gains its primaries and transfer as tags, which setparams sets without changing a pixel."""
    plan = encode_plan(args, info, out_size, destination)
    if getattr(args, "lossless", False):
        # RGB goes out as it came out of the effect: no matrix, no tv range, no chroma subsampling.
        # this is the way to carry data frames such as the vector images of create-vectors
        lossless = ["-c:v", "ffv1", "-level", "3", "-pix_fmt", LOSSLESS_PIX_FMT[args.depth]]
        if plan is None:
            return lossless
        return ["-vf", f"setparams=color_primaries={plan['primaries']}:color_trc={plan['transfer']}"] + lossless
    matrix, space, primaries, trc = color_settings(info)
    tags = [f"colorspace={space}"]
    if primaries:
        tags.append(f"color_primaries={primaries}")
    if trc:
        tags.append(f"color_trc={trc}")
    graph = [f"scale=out_color_matrix={matrix}:out_range=tv", "setparams=" + ":".join(tags + ["range=tv"])]
    if plan is None:
        return ["-vf", ",".join(graph), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]
    encode = ["-vf", ",".join(graph), "-c:v", plan["codec"], "-crf", "18", "-pix_fmt", plan["pix_fmt"]]
    if plan["tag"]:
        encode += ["-tag:v", plan["tag"]]
    if plan["hdr"]:
        encode += ["-x265-params", x265_params(plan)]
    return encode


def build_commands(args, info, vfx, destination, plan=None, sizes=None, geo=None):
    """the three argv lists: decode, effect, encode.

    the decode and the encode both name the color matrix, the tagged one or the
    size convention for an untagged source, so a color that goes out as RGB comes
    back as the same YUV. ffmpeg alone would read an untagged HD file as bt601 on
    the way in and a player would read it as bt709 on the way out.

    a temporal effect (plan["kind"]) also gets its frame count, the layers' raw
    files and the scratch directory to spool the piped input in; an output that
    must be cut at the end of the video makes the audio copy stop there. `sizes` is the frame size
    vfx gets and the one the encode gets, when a declared stage changed them (frames/2), and `geo` the
    work geometry (work_geometry). the encode reads the raw frames, then the audio inputs, maps the
    video and then the audio, and ends with the video options, the audio codec, -shortest and the mp4
    flags (decode_filters, effect_params, audio_parts, video_encode_args)."""
    geo = geo or work_geometry(args, info)
    w, h = geo["width"], geo["height"]
    (ew, eh), (ow, oh) = sizes or ((w, h), (w, h))
    fmt = PIX_FMT[args.depth]
    rate = geo["rate"]
    base = [ffmpeg_path(), "-v", "error", "-nostdin"]

    decode = base + ["-i", args.input, "-map", f"0:{info['video_index']}", "-an", "-sn", "-dn",
                     "-vf", ",".join(decode_filters(args, info, geo)), "-pix_fmt", decode_format(args, info),
                     "-f", "rawvideo", "-"]

    effect = [vfx, "apply", args.effect, "--width", str(ew), "--height", str(eh),
              "--depth", str(args.depth), "--fps", rate_text(rate)]
    if plan and plan["kind"] == "temporal":
        if plan["frames"] is not None:
            effect += ["--frames", str(plan["frames"])]
        effect += ["--tmp-dir", plan["scratch"]]
        for layer in plan["layers"]:
            effect += ["--layer", f"{layer['id']}={layer['temp']}"]
        effect += vfx_rate_flags(args, geo, plan["layers"])
        if getattr(args, "device", None) is not None:
            effect += ["--device", args.device]
    for p in effect_params(args, geo):
        effect += ["--param", p]

    inputs, maps, codec = audio_parts(args, info, plan)
    encode = base + ["-n", "-f", "rawvideo", "-pix_fmt", fmt, "-video_size", f"{ow}x{oh}",
                     "-framerate", rate_text(rate), "-i", "-"] + inputs + ["-map", "0:v:0"] + maps
    encode += video_encode_args(args, info, geo, (ow, oh), destination) + codec
    if plan and plan["shortest"]:
        encode += ["-shortest"]
    if os.path.splitext(destination)[1].lower() in (".mp4", ".m4v", ".mov"):
        encode += ["-movflags", "+faststart"]
    encode += [partial_path(destination)]
    return decode, effect, encode


def preflight(args, info, vfx, einfo=None, geo=None):
    """ask vfx to check the effect id, every param (keyframes too), the layer ids and the
    rate with `apply --check`, which reads and writes no file, before any decoding happens.
    it also proves the binary runs. `geo` is the work geometry the run renders at."""
    geo = geo or work_geometry(args, info, einfo)
    cmd = [vfx, "apply", args.effect, "--width", str(geo["width"]), "--height", str(geo["height"]),
           "--depth", str(args.depth), "--fps", rate_text(geo["rate"]), "--check"]
    if einfo and einfo["kind"] == "temporal":
        for layer_id, video in args.layer:
            cmd += ["--layer", f"{layer_id}={video}"]
        cmd += vfx_rate_flags(args, geo)
        if getattr(args, "device", None) is not None:
            cmd += ["--device", args.device]
    for p in effect_params(args, geo):
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


# how vfx starts the stderr line of an open parity gap that a render used (README: limits)
VFX_NOTE = "vfx: note: "


def vfx_notes(log):
    """the notes vfx printed into its log, in order, each as `vfx: <effect>: <id> <title>: ...`."""
    try:
        with open(log, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except (OSError, TypeError):
        return []
    return ["vfx: " + line[len(VFX_NOTE):].rstrip() for line in lines if line.startswith(VFX_NOTE)]


def relay_notes(result, logs):
    """after a run that went through, each note of vfx as `[note] vfx: ...` on stderr and the list
    under `notes` in the report. a run without notes keeps its report and its output as they were."""
    notes = vfx_notes(logs.get("vfx")) if logs else []
    for line in notes:
        eprint(f"[note] {line}")
    if notes:
        result["notes"] = notes


def run_pipeline(decode, effect, encode, scratch, wire=None):
    """decode | vfx | encode with every stderr in a log file, so no pipe can fill up
    and stall a stage. with `wire` (a Wire) the decoded frames go to vfx through a relay
    that prepares them. returns each stage's exit code and log path."""
    stages = [("decode", decode), ("vfx", effect), ("encode", encode)]
    logs = {name: os.path.join(scratch, name + ".log") for name, _ in stages}
    handles = {name: open(path, "w", encoding="utf-8") for name, path in logs.items()}
    procs = []
    buffering = 0 if wire else -1
    try:
        p1 = subprocess.Popen(decode, shell=False, stdin=subprocess.DEVNULL, bufsize=buffering,
                              stdout=subprocess.PIPE, stderr=handles["decode"])
        procs.append(p1)
        p2 = subprocess.Popen(effect, shell=False, stdin=subprocess.PIPE if wire else p1.stdout, bufsize=buffering,
                              stdout=subprocess.PIPE, stderr=handles["vfx"])
        procs.append(p2)
        hop = Hop(p1.stdout, p2.stdin, wire) if wire else None
        if hop:
            hop.thread.start()
        else:
            p1.stdout.close()
        p3 = subprocess.Popen(encode, shell=False, stdin=p2.stdout,
                              stdout=subprocess.DEVNULL, stderr=handles["encode"])
        procs.append(p3)
        p2.stdout.close()
        codes = {"encode": p3.wait(), "vfx": p2.wait(), "decode": p1.wait()}
        if hop:
            hop.thread.join(GRACE)
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


def frame_size(args, info, geo=None):
    """bytes in one raw frame of the --depth format at the work size (`geo`, else the input's)."""
    w, h = (geo["width"], geo["height"]) if geo else display_size(info)
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


def build_stages(args, info, chain=None, work=None):
    """the --pre-stage and --post-stage commands with their placeholders filled in: dicts with role,
    index, name (pre1, post1, ...), label (pre-stage 1, ...), argv and command, the pre stages first,
    each group in the order given. a pre stage reads the frames the input decodes to, a post stage the
    frames vfx writes (--frames for a temporal effect, else the input's count). a stage is expected
    to write as many frames as it reads. a program that cannot be found is refused here, before any work.

    with `chain` (a dict, frames/2) the geometry is chained in pipe order: every stage's placeholders
    hold the geometry it receives, a stage with --geometry declared is asked `CMD --describe` and its
    answer is what the next process gets. chain["vfx"] is then the geometry vfx reads and chain["out"]
    the one the encode reads; a stage dict also gets frame_bytes (of its output) and, when declared, answer.
    the chain starts at what the decode writes, the work geometry `work` (work_geometry): the work size, the
    rate of the decoded clip (clip_rate: the input's own when a temporal effect reads it at that rate under
    --fps) and its count; vfx writes the output rate, and its frames that start inside the clip
    (output_frames) unless --frames says otherwise."""
    work = work or work_geometry(args, info)
    geo = {"width": work["width"], "height": work["height"], "depth": args.depth,
           "fps": work.get("clip_rate", work["rate"]), "frames": work["frames"]}
    groups = (("pre", args.pre_stage, "the input decodes to"), ("post", args.post_stage, "the output has"))
    stages = []
    for role, templates, what in groups:
        if role == "post":
            if chain is not None:
                chain["vfx"] = dict(geo)
            if args.frames is not None:
                geo["frames"] = args.frames
            elif geo["fps"] != work["rate"] and geo["frames"] is not None:
                geo["frames"] = output_frames(geo["frames"], geo["fps"], work["rate"])
            geo["fps"] = work["rate"]
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
    and a producer that stops in the middle of a frame. with `wire` (a Wire) it passes on whole frames
    prepared and a partial last frame as it came, and counts what it read in prepared bytes."""

    def __init__(self, source, sink, wire=None):
        self.source, self.sink, self.wire = source, sink, wire
        self.scale = wire.frame_bytes // wire.in_bytes if wire else 1
        self.moved = time.monotonic()  # when a byte last went in or came out
        self.state = "reading"         # reading: waits for the producer, writing: waits for the consumer, done
        self.read = self.written = 0   # bytes taken from the producer, bytes the consumer accepted
        self.eof = False               # the producer closed its output
        self.broken = False            # the consumer closed its input while the producer still had data
        self.thread = threading.Thread(target=self.carry, daemon=True)

    def carry_frames(self):
        """the relay with a Wire: each decoded frame read whole into one buffer, prepared and passed on; a partial last
        frame goes on as it came."""
        frame = bytearray(self.wire.in_bytes)
        view = memoryview(frame)
        while True:
            self.state = "reading"
            got = 0
            while got < len(frame):
                try:
                    n = self.source.readinto(view[got:])
                except (OSError, ValueError):
                    n = 0
                if not n:
                    break
                got += n
                self.read += n * self.scale
                self.moved = time.monotonic()
            if got < len(frame):
                if got:
                    self.give(view[:got])
                self.eof = True
                return
            if not self.give(self.wire(frame)):
                return

    def give(self, data):
        """write `data` to the consumer; False when it closed its input."""
        self.state = "writing"
        view = memoryview(data)
        try:
            while view:
                sent = self.sink.write(view)
                view = view[sent:]
                self.written += sent
                self.moved = time.monotonic()
        except (OSError, ValueError):
            self.broken = True
            return False
        return True

    def carry(self):
        try:
            if self.wire is not None:
                self.carry_frames()
                return
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
                if not self.give(chunk):
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


def run_staged_pipeline(chain, scratch, timeout, wire=None):
    """chain[0] | chain[1] | ... with a Hop between every two neighbours and every stderr in a log
    file in scratch, so no pipe can fill up and stall a process. a pre or post stage that stalls for
    `timeout` seconds (0: never) is killed; once a process has failed the others get GRACE seconds to
    end before they are killed too, so the run always ends. returns the exit code and log path of every
    process by name, the hops (hops[i] carries chain[i] to chain[i + 1]) and the names killed for stalling. with
    `wire` (a Wire) the hop after the decode prepares its frames."""
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
        hops = [Hop(procs[i].stdout, procs[i + 1].stdin, wire if i == 0 and chain[0]["role"] == "decode" else None)
                for i in range(last)]
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


def run_staged(chain, scratch, args, info, geo=None):
    """run the pipe with stages and fail, in the style of a vfx failure, when it did not go through:
    the failing process, its exit code and the last lines of its own stderr. returns the codes and logs.
    `geo` is the work geometry, the size of the frames on every relay a declared stage did not change."""
    codes, logs, hops, timed_out = run_staged_pipeline(chain, scratch, args.stage_timeout, decode_wire(args, info, geo))
    sizes = frame_size(args, info, geo)
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


def verify(src, out, rate, frames=None, source_frames=None, size=None, durations=True, audio=None, color=None):
    """problems found comparing the output to the input; empty means it checks out.

    with `frames` (a --frames value) the output must have that many frames and last
    frames / rate; when that is the input's own count (`source_frames`) the output is
    compared with the input's duration as usual. `size` is the size the output must have (default: the
    input's; a declared stage can change it, frames/2); `durations` False skips the length checks when a
    declared stage left the count unknown. the color and the audio checks are verify_color's (with the
    color plan `color`) and verify_audio's (with the plan of the run, `audio`). under --audio retime and drop
    the output lasts as long as its video (the plan's audio_length), not as the input, whose audio may
    outlast its video."""
    if out is None or not out["has_video"]:
        return ["the output has no readable video stream"]
    problems = []
    reference = src["duration"]
    if (audio or {}).get("audio_mode") is not None and audio.get("audio_length") is not None:
        reference = audio["audio_length"]
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
    elif reference is not None and durations:
        if out["duration"] is None:
            problems.append("the output has no readable duration")
        elif abs(out["duration"] - reference) > 1.0 / float(rate) + 1e-3:
            problems.append(f"duration is {out['duration']:.3f}s, expected {reference:.3f}s "
                            f"within one frame ({1.0 / float(rate):.3f}s)")
    if frames is not None and out.get("nb_frames") is not None and out["nb_frames"] != frames:
        problems.append(f"the output has {out['nb_frames']} frames, expected {frames}")
    return problems + verify_color(src, out, color) + verify_audio(src, out, audio)


def verify_color(src, out, color=None):
    """the color problems of an output. `color` is the color plan of a run whose encode differs from the
    one of before (encode_plan; None: there is none, and nothing is checked, as before): the output of an
    HDR input keeps the input's transfer, and a lossy output has the pixel format of the plan, 10 bit for
    HDR and 4:4:4 for a size with an odd side (the size itself is verify's)."""
    if not color:
        return []
    problems = []
    if color.get("hdr") and out.get("color_transfer") != src.get("color_transfer"):
        problems.append(f"transfer is {out.get('color_transfer')}, expected {src.get('color_transfer')}")
    if color.get("codec") != "ffv1" and out.get("pix_fmt") != color.get("pix_fmt"):
        problems.append(f"pixel format is {out.get('pix_fmt')}, expected {color.get('pix_fmt')}")
    return problems


def verify_audio(src, out, audio=None):
    """the audio problems of an output. `audio` is the plan of the run (None for a caller that predates
    it); --audio copy must keep the input's first audio stream with its codec and its channels, and adds
    no audio to an input without any. --audio retime must give as many audio streams as it planned, each in
    the planned codec with its source's channel count (an encoder may relabel a layout as an equivalent one,
    aac writes 5.1(side) as 5.1) and as long as the video within the larger of one frame and 50 ms, or of one
    frame of its encoder when -shortest cuts it; drop (and retime of an input without audio) leaves no audio."""
    if (audio or {}).get("audio_mode") is not None:
        return verify_retimed(out, audio)
    problems = []
    if src["has_audio"]:
        if not out["has_audio"]:
            problems.append("the audio stream was lost")
        elif (out["acodec"], out["channels"]) != (src["acodec"], src["channels"]):
            problems.append(f"audio is {out['acodec']} {out['channels']}ch, "
                            f"expected {src['acodec']} {src['channels']}ch")
    elif out["has_audio"]:
        problems.append("the output has audio the input did not")
    return problems


# the samples of one frame of an encoder: aac's, and the most flac takes (its frames last at most 105 ms)
AAC_FRAME = 1024
FLAC_FRAME_SECONDS = 0.105


def codec_frame_seconds(codec, rate):
    """the seconds of the longest frame `codec` encodes at `rate` samples a second."""
    return FLAC_FRAME_SECONDS if codec == "flac" else AAC_FRAME / float(rate)


def verify_retimed(out, plan):
    """verify_audio for --audio retime and drop (see there)."""
    planned = plan["audio_streams"]
    if not planned:
        if not out["has_audio"]:
            return []
        return ["the output has audio, which --audio drop leaves out" if plan["audio_mode"] == "drop"
                else "the output has audio the input did not"]
    streams = audio_streams(out)
    problems = []
    if len(streams) != len(planned):
        problems.append(f"the output has {len(streams)} audio streams, expected {len(planned)}")
    rate = float(plan["audio_fps"])
    expected = plan["audio_frames"] / rate if not plan["shortest"] else None
    if expected is None:
        video = [stream_duration(st) for st in (_PROBE_DATA.get(out["path"]) or {}).get("streams", [])
                 if st.get("index") == out.get("video_index")]
        expected = (video[0] if video else None) or out["duration"]
    window = max(1.0 / rate, AUDIO_LENGTH_TOLERANCE)
    if plan["shortest"]:
        # -shortest ends a stream on a whole frame of its encoder, which can be longer than the window
        window = max([window] + [codec_frame_seconds(plan["audio_codec"], want["encode_rate"]) for want in planned])
    for number, (st, want) in enumerate(zip(streams, planned), 1):
        if st["codec"] != plan["audio_codec"]:
            problems.append(f"audio stream {number} is {st['codec']}, expected {plan['audio_codec']}")
        if st["channels"] != want["channels"]:
            problems.append(f"audio stream {number} has {st['channels']} channels, expected {want['channels']}")
        if st["duration"] is None or expected is None:
            problems.append(f"audio stream {number} has no readable duration")
        elif abs(st["duration"] - expected) > window + 1e-6:
            problems.append(f"audio stream {number} lasts {st['duration']:.3f}s, expected {expected:.3f}s "
                            f"within {window:.3f}s")
    return problems


def report_geometry(args, info, geo):
    """what the summary and the report say about the work geometry, as (phrases, keys): the phrases follow
    the layers in the summary, the keys follow input_frames in the report. size, fps, fps_exact, frames and
    input_frames always describe the output (geo); an output with the input's geometry adds nothing. any
    other adds the phrases `scaled from WxH`, `converted from R fps` and `K=V and K=V added` (each when it
    applies) and the keys source (the input's size, fps, fps_exact, frames and the rounding of the frame
    count) and auto_params (the params the run added)."""
    source = geo.get("source")
    if source is None:
        return [], {}
    phrases = []
    if (geo["width"], geo["height"]) != (source["width"], source["height"]):
        phrases.append(f"scaled from {source['width']}x{source['height']}")
    if geo["rate"] != source["fps"]:
        phrases.append(f"converted from {float(source['fps']):.3f} fps")
    auto = list(geo.get("auto_params") or [])
    if auto:
        phrases.append(" and ".join(auto) + " added")
    keys = {"source": {"size": f"{source['width']}x{source['height']}", "fps": float(source["fps"]),
                       "fps_exact": rate_text(source["fps"]), "frames": source["frames"],
                       "rounding": geo.get("rounding")},
            "auto_params": auto}
    return phrases, keys


def report_encode(args, info, geo, out_size):
    """what the summary and the report say about the encode, as (phrases, keys): the phrases follow the
    geometry's, the keys follow the layers. `out_size` is the frame size the encode reads; the encode of
    before (libx264 yuv420p, or ffv1 with --lossless) adds nothing. an HDR input adds a phrase that its
    signal is kept as stored and one about its encode, and the key hdr (its tags and the HDR metadata the
    output gets, None for none); an odd side on the lossy path adds the phrase `encoded yuv444p (the size
    has an odd side)` (yuv444p10le for HDR); both add the key encode (codec and pixel format)."""
    plan = encode_plan(args, info, out_size)
    if plan is None:
        return [], {}
    phrases, keys = [], {}
    if plan["hdr"]:
        phrases.append(f"HDR {plan['transfer']} kept as stored (no tone mapping)")
        if plan["codec"] == "ffv1":
            phrases.append(f"ffv1 tagged {plan['primaries']} {plan['transfer']}")
        else:
            written = [name for name, key in (("master-display", "master_display"), ("max-cll", "max_cll"))
                       if plan[key]]
            phrases.append("libx265 10 bit" + (" with " + " and ".join(written) if written else ""))
        keys["hdr"] = {key: plan[key] for key in ("transfer", "primaries", "matrix", "master_display", "max_cll")}
    if plan["odd"]:
        phrases.append(f"encoded {plan['pix_fmt']} (the size has an odd side)")
    keys["encode"] = {"codec": plan["codec"], "pix_fmt": plan["pix_fmt"]}
    return phrases, keys


def report_audio(args, info, plan):
    """what the summary and the report say about the audio, as (phrases, keys): the phrases follow the
    encode's, and the audio key follows the encode's keys. the summary of a temporal plan says what the
    copy does to the audio when the length changes it; the audio key gives the decision (audio_plan) and
    whether -shortest cuts the copy."""
    if plan.get("audio_mode") is not None:
        return report_retimed(info, plan)
    phrases = []
    if plan["kind"] == "temporal":
        if plan["audio"] == "shortest":
            phrases.append("audio cut at the end of the video (-shortest)")
        elif plan["audio"] == "ends-early":
            phrases.append("the audio ends before the video does")
    return phrases, {"audio": {"plan": plan["audio"], "shortest": plan["shortest"]}}


def report_retimed(info, plan):
    """report_audio for --audio retime and drop: the phrase of the mode (when the input has audio), and the
    audio key with mode, codec and the streams after plan and shortest."""
    phrases = []
    streams = plan["audio_streams"]
    if streams:
        phrases.append(f"audio retimed with the effect's time map ({plan['audio_codec']})")
        if plan["shortest"]:
            phrases.append("audio cut at the end of the video (-shortest)")
    elif plan["audio_mode"] == "drop" and info["has_audio"]:
        phrases.append("audio dropped")
    shown = [{"index": st["index"], "channels": st["channels"], "sample_rate": st["encode_rate"],
              "layout": st["layout"], "bit_rate": st["bit_rate"], "offset": st["offset"]} for st in streams]
    return phrases, {"audio": {"plan": plan["audio"], "shortest": plan["shortest"], "mode": plan["audio_mode"],
                               "codec": plan["audio_codec"], "streams": shown}}


def make_result(args, info, einfo, vfx, dst, plan, commands, stages=(), geo=None, out_size=None):
    """the report for a plan: the facts a caller needs before anything runs, outputs filled in later.
    a run with stages also names the contract and lists the stages (exit codes filled in after the run).
    `geo` is the work geometry and `out_size` the frame size the encode reads (default: the work size).
    the summary reads: head, kind and frames, layers, geometry, encode, audio, pipe."""
    geo = geo or work_geometry(args, info, einfo)
    w, h = geo["width"], geo["height"]
    rate = geo["rate"]
    geometry_phrases, geometry_keys = report_geometry(args, info, geo)
    encode_phrases, encode_keys = report_encode(args, info, geo, out_size or (w, h))
    audio_phrases, audio_keys = report_audio(args, info, plan)
    parts = [f"{args.effect} on {os.path.basename(args.input)}: {w}x{h}, {float(rate):.3f} fps, "
             f"{args.depth} bpc ({PIX_FMT[args.depth]})"]
    if plan["kind"] == "temporal":
        parts.append(f"temporal, {plan['frames']} frames" if plan["frames"] is not None
                     else "temporal, as many frames as the input decodes to")
        if plan["layers"]:
            parts.append("layers " + ", ".join(layer["id"] for layer in plan["layers"]))
    parts += geometry_phrases + encode_phrases + audio_phrases
    pipe = [e["label"] for e in pipe_chain(None, None, None, stages)]
    parts.append(f"{' | '.join(pipe)} -> {dst}")
    prepared = prepare_report(args, info, plan)
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
        **geometry_keys,
        "layers": [{"id": layer["id"], "video": layer["video"], "temp": layer["temp"]} for layer in plan["layers"]],
        **encode_keys,
        **audio_keys,
        "temp_files": plan["temp_files"],
        "vfx": vfx,
        "summary": ", ".join(parts),
        "dry_run": args.dry_run,
        "commands": [fmt_cmd(c) for c in commands],
        **({"prepare": prepared} if prepared else {}),
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
                             "for a hold key and a trailing e to ease a key, iS/I and oS/I after F for "
                             "incoming and outgoing Bezier handles (S units a second, I percent "
                             "influence); quote the argument, the ; ends a command in a shell")
    parser.add_argument("--layer", action="append", default=[], type=layer_arg, metavar="ID=VIDEO",
                        help="an extra input layer of a temporal effect (the ids `vfx list` shows), "
                             "repeatable. the video is decoded to a temporary raw file at the output's "
                             "size (scaled when it differs) and removed afterwards")
    parser.add_argument("--frames", type=frames_arg, metavar="N",
                        help="frames of the output of a temporal effect (default: the frames of the "
                             "input). the output lasts N / fps. with the default --audio copy the "
                             "audio is copied untouched: when the output is shorter than the audio it "
                             "is cut at the end of the video (ffmpeg -shortest), when longer the audio "
                             "just ends early")
    parser.add_argument("--size", type=size_arg, metavar="WxH",
                        help="the frame size of the output, W and H whole numbers of 1 or more (default: the "
                             "input's). the input and every layer are scaled to it, stretched, before the "
                             "effect, which renders at that size. the input's own size changes nothing")
    parser.add_argument("--fps", type=fps_arg, metavar="R",
                        help="the frame rate of the output: an integer, a decimal of at most 6 places or N/D, "
                             "above 0 (default: the input's). the effect renders at this rate and reads the "
                             "input and every layer at their own rates, so a slower output still shows real "
                             "frames of faster footage; keyframes, frame numbers and --frames count output "
                             "frames, and the output lasts as long as the input unless --frames says "
                             "otherwise. an effect with the in_fps_is_out_fps and input_frame_rate params gets "
                             "in_fps_is_out_fps=0 and input_frame_rate=<the input's rate> unless either one "
                             "is given. the input's own rate changes nothing")
    parser.add_argument("--audio", choices=AUDIO_MODES, default="copy",
                        help="what the output does with the input's audio. copy (the default) copies every "
                             "audio stream untouched: an output shorter than the audio cuts it at the end of "
                             "the video (ffmpeg -shortest), a longer one lets it end early. retime plays every "
                             "stream along the effect's time map, varispeed, so the pitch follows the speed: "
                             "silent where the clip has no frame or a frame is held, backwards where time "
                             "runs backwards, as long as the video, encoded aac at the source bit rate (flac "
                             "with --lossless). drop writes no audio")
    parser.add_argument("--device", type=device_arg, metavar="DEVICE",
                        help="the device vfx renders a temporal effect on: cpu (the default), gpu or gpu=N "
                             "(`vfx devices` lists them), passed through to vfx apply --device: gpu falls "
                             "back to the cpu with a note, gpu=N that cannot be used is refused before any "
                             "work; the frames are the cpu's either way. a frame effect refuses the flag")
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
    if args.lossless and args.depth == 32:
        fail("--lossless needs --depth 8 or 16: ffv1 has no float pixel format")
    if info["hdr"] and not args.lossless:
        require_hdr_encoder()

    vfx = find_vfx(args.vfx)
    einfo = effect_info(vfx, args.effect)
    check_layer_options(args, einfo)
    geo = work_geometry(args, info, einfo)
    require_audio_map(args, info)
    layers = probe_layers(args) if einfo["kind"] == "temporal" else []
    preflight(args, info, vfx, einfo, geo)
    chain_geo = {} if framed else None
    stages = build_stages(args, info, chain_geo, geo)
    declared = any(s.get("geometry") == "declared" for s in stages)
    sizes = None
    if framed:
        vgeo, ogeo = chain_geo["vfx"], chain_geo["out"]
        sizes = ((vgeo["width"], vgeo["height"]), (ogeo["width"], ogeo["height"]))
        if args.describe:
            print(json.dumps({"ok": True, "contract": "frames/2", "stage": args.effect, **geometry_json(ogeo)}))
            sys.exit(0)

    dst = args.output or default_output(args.input, args.effect, "mkv" if args.lossless else "mp4")
    if same_file(dst, args.input):
        fail("the output path is the input file; choose a different --output path")
    for layer in layers:
        if same_file(dst, layer["video"]):
            fail(f"the output path is the video of layer {layer['id']}; choose a different --output path")

    out_size = sizes[1] if sizes else None

    def assemble(scratch):
        """plan, layer decodes, audio steps and the three stages for a scratch directory (named, not created)."""
        plan = make_plan(args, info, einfo, layers, scratch, geo, chain_geo)
        if declared:
            plan_audio(args, info, plan, chain_geo["out"]["frames"], geo)
        decode, effect, encode = build_commands(args, info, vfx, dst, plan, sizes, geo)
        layer_commands = build_layer_commands(args, info, plan, sizes[0] if sizes else None, geo)
        return plan, layer_commands, audio_commands(args, info, plan), (decode, effect, encode)

    def framed_result(result, chain):
        """the frames/2 report: ok and contract first, every process with role and index, the geometry."""
        out = {"ok": True, "contract": "frames/2"}
        out.update((k, v) for k, v in result.items() if k not in ("contract", "stages"))
        out["stages"] = chain_report(chain, probe_command(partial_path(dst)))
        out["geometry"] = geometry_json(chain_geo["out"])
        return out

    if args.dry_run:
        plan, layer_commands, audio, triple = assemble(os.path.join(tempfile.gettempdir(), "apply_XXXXXXXX"))
        chain = pipe_chain(*triple, stages)
        pipe = [entry["argv"] for entry in chain]
        result = make_result(args, info, einfo, vfx, dst, plan, layer_commands + [argv for _, argv in audio] + pipe,
                             stages, geo, out_size)
        emit(framed_result(result, chain) if framed else result, args.as_json)
        sys.exit(0)

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    partial = partial_path(dst)
    discard(partial)
    scratch = tempfile.mkdtemp(prefix="apply_")
    started = time.time()
    try:
        plan, layer_commands, audio, (decode, effect, encode) = assemble(scratch)
        chain = pipe_chain(decode, effect, encode, stages)
        result = make_result(args, info, einfo, vfx, dst, plan,
                             layer_commands + [argv for _, argv in audio] + [e["argv"] for e in chain], stages, geo,
                             out_size)
        if framed:
            for entry in chain:
                if entry["role"] == "vfx":
                    entry["frame_bytes"] = vgeo["width"] * vgeo["height"] * BYTES_PER_PIXEL[args.depth]
            result = framed_result(result, chain)
        decode_layers(layer_commands, plan, args, info, scratch, sizes[0] if sizes else None, geo)
        run_audio(audio, scratch)
        if framed:
            codes, logs = run_staged(chain, scratch, args, info, geo)
            result["stages"] = chain_report(chain, probe_command(partial), codes, logs)
        elif stages:
            codes, logs = run_staged(chain, scratch, args, info, geo)
            result["stages"] = [stage_report(s, codes[s["name"]], logs[s["name"]]) for s in stages]
        else:
            codes, logs = run_pipeline(decode, effect, encode, scratch, decode_wire(args, info, geo))
            if any(codes.values()):
                lines = []
                for name in ("vfx", "decode", "encode"):
                    if codes[name]:
                        lines.append(f"{name} exited with code {codes[name]}")
                        lines += ["  " + t for t in tail(logs[name])]
                fail("the pipeline failed:\n" + "\n".join(lines))
        if declared:
            final = chain_geo["out"]
            problems = verify(info, probe(partial), geo["rate"], frames=final["frames"],
                              source_frames=plan["input_frames"], size=(final["width"], final["height"]),
                              durations=final["frames"] is not None, audio=plan,
                              color=encode_plan(args, info, (final["width"], final["height"]), dst))
        else:
            problems = verify(info, probe(partial), geo["rate"], frames=args.frames,
                              source_frames=plan["input_frames"], size=(geo["width"], geo["height"]), audio=plan,
                              color=encode_plan(args, info, (geo["width"], geo["height"]), dst))
        if problems:
            fail("the output did not verify: " + "; ".join(problems), kind="stage_failed", stage="verify")
        if framed:
            result["stages"][-1]["exit_code"] = 0
        try:
            os.replace(partial, dst)
        except OSError:
            fail("could not move the finished output into place; is the destination open in another program?",
                 kind="stage_failed", stage="verify")
        relay_notes(result, logs)
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
