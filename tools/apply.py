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

stdlib only: probe first, act, verify, with --dry-run, --json, -o and --force on
the one entry point. every external tool runs via subprocess with list args,
shell=False. use sys.executable, never python3.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(HERE)

PROBE_TIMEOUT = 30
JSON_OUTPUT = False
FORCE = False

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


def fail(message, code=1):
    if JSON_OUTPUT:
        print(json.dumps({"ok": False, "error": message}))
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
                fail(f"explicit {name} executable not found: {override}")
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
             "  sudo apt install ffmpeg           (debian/ubuntu)")


def exe_name():
    return "vfx.exe" if os.name == "nt" else "vfx"


def find_vfx(explicit):
    """--vfx, then VFX_PATH, then the workspace release build, then PATH."""
    if explicit:
        if not os.path.isfile(explicit):
            fail(f"vfx not found at {explicit}")
        return os.path.abspath(explicit)
    env = os.environ.get("VFX_PATH")
    if env:
        if not os.path.isfile(env):
            fail(f"VFX_PATH points at a missing file: {env}")
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
         "or pass --vfx PATH / set VFX_PATH")


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


def probe(path):
    """one ffprobe json call. returns a facts dict, or None if it cannot be read.

    keys: has_video, has_audio, vcodec, acodec, pix_fmt, width, height, rotation,
    fps, rate (exact Fraction), vfr, duration, channels, hdr, size, container,
    nb_frames (the container's frame count, None when it does not say),
    video_duration and audio_duration (each stream's own, None when not given) and
    the color tags. ffprobe omits fields it cannot find, so every read is guarded.
    """
    cmd = [ffprobe_path(), "-v", "quiet", "-print_format", "json",
           "-show_streams", "-show_format", path]
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
        fail(f"no such file: {path}")
    info = probe(path)
    if info is None:
        fail(f"cannot probe {path}; is it a media file?")
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
    global JSON_OUTPUT, FORCE
    for name in ("ffmpeg", "ffprobe"):
        value = getattr(args, name + "_path", None)
        if value:
            os.environ["MEDIA_" + name.upper() + "_PATH"] = value
    _TOOLS.clear()
    JSON_OUTPUT = bool(args.as_json)
    FORCE = bool(args.force)
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
        fail("output verification failed")
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
            fail(f"cannot run vfx at {vfx}")
        if proc.returncode == 0:
            break
    else:
        lines = (proc.stderr or "").strip().splitlines()
        fail("`vfx list --json` failed: " + (lines[-1] if lines else f"exit code {proc.returncode}"))
    try:
        listing = json.loads(proc.stdout)
    except ValueError:
        fail("`vfx list --json` did not print JSON")
    for entry in listing:
        if entry.get("id") == effect_id:
            return {"id": effect_id, "kind": entry.get("kind") or "frame",
                    "layers": list(entry.get("layers") or [])}
    fail(f"unknown effect `{effect_id}`, `vfx list` shows the registered ones")


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
            fail(f"layer {layer_id}: {path} has no video stream")
        if not li["rate"] or not li["width"] or not li["height"]:
            fail(f"layer {layer_id}: cannot read the frame size and frame rate of {path}")
        if li["hdr"]:
            fail(f"layer {layer_id}: HDR video is not supported; tone mapping is not part of this tool")
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


def build_layer_commands(args, info, plan):
    """one ffmpeg per layer: decode it to a raw file of the effect's pixel format, size and frame rate."""
    w, h = display_size(info)
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


def decode_layers(commands, plan, args, info, scratch):
    """run the layer decodes one after the other; each leaves a raw file of whole frames or fails with ffmpeg's last lines."""
    w, h = display_size(info)
    frame_bytes = w * h * {8: 4, 16: 8, 32: 16}[args.depth]
    for layer, cmd in zip(plan["layers"], commands):
        log = os.path.join(scratch, f"layer_{layer['id']}.log")
        with open(log, "w", encoding="utf-8") as handle:
            proc = run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=handle)
        if proc.returncode != 0:
            lines = "\n".join("  " + t for t in tail(log))
            fail(f"layer {layer['id']}: decoding {layer['video']} failed (ffmpeg exited with code {proc.returncode})"
                 + ("\n" + lines if lines else ""))
        try:
            size = os.path.getsize(layer["temp"])
        except OSError:
            size = 0
        if size == 0 or size % frame_bytes:
            fail(f"layer {layer['id']}: {layer['video']} decoded to {size} bytes, "
                 f"which is not a whole number of {w}x{h} frames")


def rate_text(rate):
    return f"{rate.numerator}/{rate.denominator}"


def build_commands(args, info, vfx, destination, plan=None):
    """the three argv lists: decode, effect, encode.

    the decode and the encode both name the color matrix, the tagged one or the
    size convention for an untagged source, so a color that goes out as RGB comes
    back as the same YUV. ffmpeg alone would read an untagged HD file as bt601 on
    the way in and a player would read it as bt709 on the way out.

    a temporal effect (plan["kind"]) also gets its frame count, the layers' raw
    files and the scratch directory to spool the piped input in; an output that
    must be cut at the end of the video makes the audio copy stop there."""
    w, h = display_size(info)
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

    effect = [vfx, "apply", args.effect, "--width", str(w), "--height", str(h),
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
        encode = base + ["-n", "-f", "rawvideo", "-pix_fmt", fmt, "-video_size", f"{w}x{h}",
                         "-framerate", rate_text(rate), "-i", "-", "-i", args.input,
                         "-map", "0:v:0", "-map", "1:a?",
                         "-c:v", "ffv1", "-level", "3", "-pix_fmt", LOSSLESS_PIX_FMT[args.depth], "-c:a", "copy"]
    else:
        encode = base + ["-n", "-f", "rawvideo", "-pix_fmt", fmt, "-video_size", f"{w}x{h}",
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
        fail(f"cannot run vfx at {vfx}")
    if proc.returncode != 0:
        lines = (proc.stderr or "").strip().splitlines()
        fail(lines[-1].replace("vfx: error: ", "vfx: ") if lines else f"vfx exited with code {proc.returncode}")


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


# an output cut at the end of the video keeps the audio up to the packet that crosses the end
TRIM_TOLERANCE = 0.1


def verify(src, out, rate, frames=None, source_frames=None):
    """problems found comparing the output to the input; empty means it checks out.

    with `frames` (a --frames value) the output must have that many frames and last
    frames / rate; when that is the input's own count (`source_frames`) the output is
    compared with the input's duration as usual."""
    if out is None or not out["has_video"]:
        return ["the output has no readable video stream"]
    problems = []
    want = display_size(src)
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
    elif src["duration"] is not None:
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


def make_result(args, info, einfo, vfx, dst, plan, commands):
    """the report for a plan: the facts a caller needs before anything runs, outputs filled in later."""
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
    parts.append(f"decode | vfx | encode -> {dst}")
    return {
        "operation": "apply",
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
        "outputs": [],
    }


def main():
    parser = argparse.ArgumentParser(
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
    parser.add_argument("--vfx", metavar="PATH",
                        help="vfx executable (default: VFX_PATH, then the workspace release build "
                             "under CARGO_TARGET_DIR or target/, then PATH)")
    add_common_args(parser)
    args = parser.parse_args()
    configure(args)

    require_ffmpeg()
    info = require_input(args.input)
    if not info["has_video"]:
        fail("input has no video stream")
    if not info["rate"] or not info["width"] or not info["height"]:
        fail("cannot read the input's frame size and frame rate")
    if info["hdr"]:
        fail("HDR input is not supported; tone mapping is not part of this tool")
    w, h = display_size(info)
    if args.lossless and args.depth == 32:
        fail("--lossless needs --depth 8 or 16: ffv1 has no float pixel format")
    if (w % 2 or h % 2) and not args.lossless:
        fail(f"the frame size {w}x{h} has an odd side, and yuv420p needs even sides; "
             "resize or crop the input first (ffmpeg -vf crop=trunc(iw/2)*2:trunc(ih/2)*2), or pass --lossless")

    vfx = find_vfx(args.vfx)
    einfo = effect_info(vfx, args.effect)
    check_layer_options(args, einfo)
    layers = probe_layers(args) if einfo["kind"] == "temporal" else []
    preflight(args, info, vfx, einfo)

    dst = args.output or default_output(args.input, args.effect, "mkv" if args.lossless else "mp4")
    if same_file(dst, args.input):
        fail("the output path is the input file; choose a different --output path")
    for layer in layers:
        if same_file(dst, layer["video"]):
            fail(f"the output path is the video of layer {layer['id']}; choose a different --output path")

    def assemble(scratch):
        """plan, layer decodes and the three stages for a scratch directory (named, not created)."""
        plan = make_plan(args, info, einfo, layers, scratch)
        decode, effect, encode = build_commands(args, info, vfx, dst, plan)
        layer_commands = build_layer_commands(args, info, plan)
        return plan, layer_commands, (decode, effect, encode)

    if args.dry_run:
        plan, layer_commands, stages = assemble(os.path.join(tempfile.gettempdir(), "apply_XXXXXXXX"))
        emit(make_result(args, info, einfo, vfx, dst, plan, layer_commands + list(stages)), args.as_json)
        sys.exit(0)

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    partial = partial_path(dst)
    discard(partial)
    scratch = tempfile.mkdtemp(prefix="apply_")
    started = time.time()
    try:
        plan, layer_commands, (decode, effect, encode) = assemble(scratch)
        result = make_result(args, info, einfo, vfx, dst, plan, layer_commands + [decode, effect, encode])
        decode_layers(layer_commands, plan, args, info, scratch)
        codes, logs = run_pipeline(decode, effect, encode, scratch)
        if any(codes.values()):
            lines = []
            for name in ("vfx", "decode", "encode"):
                if codes[name]:
                    lines.append(f"{name} exited with code {codes[name]}")
                    lines += ["  " + t for t in tail(logs[name])]
            fail("the pipeline failed:\n" + "\n".join(lines))
        problems = verify(info, probe(partial), info["rate"], frames=args.frames,
                          source_frames=plan["input_frames"])
        if problems:
            fail("the output did not verify: " + "; ".join(problems))
        try:
            os.replace(partial, dst)
        except OSError:
            fail("could not move the finished output into place; is the destination open in another program?")
    except KeyboardInterrupt:
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
