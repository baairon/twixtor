#!/usr/bin/env python3
"""audio_map.py: play the audio of a clip along the time map of a retiming effect (varispeed).

`apply.py --audio retime` runs it once for each audio stream, between the ffmpeg that decodes the stream to
raw pcm_f32le and the encode. it can also print where each output frame lands in the clip.

  python -I tools/audio_map.py render --effect retime --fps 24/1 --frames 96 --clip-frames 96 \\
      --param speed_percent=50 --rate 48000 --channels 2 --offset 0 --in a.f32 --out b.f32
  python tools/audio_map.py positions --effect retime --fps 24 --frames 48 --param "speed_percent=100@0;50@24" --json

the map: output position t (output frames, continuous) shows clip position p(t) (clip frames).
- retime, retime-fgbg and vectors-in in Speed mode (time_remap_mode=1): p is linear between the knots of
  the effect's own half frame sum (32 bit floats, the speed read at tick trunc(L*k/2 + 0.1), L the ticks a
  frame of vfx's time base: 100 at a whole rate, D at N/D), plus trunc(src_time_offset) * out_fps / in_fps
  in retime only. the rate settings (in_fps_is_out_fps, input_frame_rate, input_fields) act only through
  that offset: the rate ratio and the duration of a source frame cancel everywhere else.
- Frame Number mode (2): p is frame_num on continuous ticks t*L, with its keys, eases and handles as vfx
  reads them; a hold stays put and jumps at the next key.
- display=2 (Source): t + trunc(src_time_offset) in retime, t in the other two.
- create-vectors, the frame effects and any other effect: t.
mode, display, offset and rates are read at the start of each output frame (tick n*L), as the effect reads
them for frame n, and hold over that frame.

the audio: output sample i (time i/SR) plays the input at clip time p/R (R the clip rate, the output rate
unless --clip-fps says otherwise), input sample (p/R - offset)*SR with offset = audio start - video start
(ffprobe start_time, which ffprobe prints to the microsecond: offset*SR is taken to the nearest whole
sample, since an audio stream starts on a sample), linearly interpolated, on every channel. it is silent where p < 0, p >= the clip's
frames, the sample lies outside the input, or p stands still (two consecutive samples at one position: a
hold); every edge between sound and silence gets a 2 ms linear fade, a jump of p cuts straight to the new
place, a falling p plays backwards. the output has floor(frames*SR/fps + 1/2) samples.

precision: the knots are the effect's own numbers (f32 sums widened to double, frame_num and the offset
as vfx reads them). each piece between two knots becomes the source position A + B*i of output sample i,
with A and B worked out in exact fractions and rounded to double once; per sample one product and one sum
in double, then a + (b - a)*f in double, stored as f32. a piece that only shifts by whole samples (B = 1, A
whole) copies the input bytes, so 100% with audio and video starting together is bit exact.

stdlib only, python 3.11. exit codes: 0 ok, 1 a value the map cannot read or a file it cannot read or
write, 2 a malformed command line.
"""

import argparse
import json
import math
import mmap
import os
import re
import sys
from array import array
from bisect import bisect_left, bisect_right
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
# apply.py runs this script with python -I, which leaves the script's folder off sys.path
sys.path.insert(0, HERE)
import speed_curve as sc  # noqa: E402

# the linear fade at an edge between sound and silence
FADE_SECONDS = 0.002
# points a frame where frame_num is not a straight line between two keys (an ease or a handle)
FRAME_NUM_STEPS = 10
# output samples rendered at a time
BLOCK = 1 << 16

# the parameters the map reads, with their kind, default and valid range as `vfx list --json` gives them,
# so a value or a key reads here as vfx reads it. any other --param is passed over
PARAMS = {
    "time_remap_mode": ("popup", 1, None),
    "speed_percent": ("float", 100.0, (-10000.0, 10000.0)),
    "frame_num": ("float", 0.0, (-200000.0, 200000.0)),
    "src_time_offset": ("float", 0.0, (0.0, 30000.0)),
    "display": ("popup", 1, None),
    "in_fps_is_out_fps": ("checkbox", True, None),
    "input_frame_rate": ("float", 29.97, (sc.f32(0.1), 1000.0)),
    "input_fields": ("popup", 1, None),
}
# the effects that follow a time map, each with the rate and offset settings it has
TIMED = {"retime": ("src_time_offset", "input_fields"), "retime-fgbg": ("input_fields",), "vectors-in": ()}
# the engine's rate model: typed rates within 2e-5 of these snap to them (timeline.rs)
SNAP_RATES = (29.97, 59.94, 23.976)
SNAP_TOLERANCE = 2e-5


class MapError(Exception):
    pass


def to_rate(value):
    """a frame rate as an exact positive Fraction: a Fraction, an int, or text N/D, an integer or a decimal."""
    if isinstance(value, Fraction):
        rate = value
    elif isinstance(value, int):
        rate = Fraction(value)
    else:
        text = str(value).strip()
        if not re.fullmatch(r"[0-9]+(\.[0-9]+)?(/[0-9]+)?", text):
            raise MapError(f"`{value}` is not a frame rate (N/D, an integer or a decimal)")
        try:
            rate = Fraction(text)
        except (ValueError, ZeroDivisionError):
            raise MapError(f"`{value}` is not a frame rate (N/D, an integer or a decimal)") from None
    if rate <= 0:
        raise MapError(f"the frame rate {value} is not above 0")
    return rate


# ----------------------------------------------------------------------------
# parameters as vfx reads them
# ----------------------------------------------------------------------------

class Const:
    """a parameter without keys."""

    def __init__(self, value):
        self.value = value

    def at(self, tick):
        return self.value


class Steps:
    """a keyed popup or checkbox: the value of the last key at or before the tick, the first key's before it."""

    def __init__(self, ticks, values):
        self.ticks, self.values = ticks, values

    def at(self, tick):
        return self.values[max(bisect_right(self.ticks, tick) - 1, 0)]


class Floats:
    """a keyed float: speed_curve's Curve (the operations of fx-core's curves.rs), kept inside the valid range as
    vfx keeps an overshooting handle, at integer or fractional ticks."""

    def __init__(self, curve, lo, hi):
        self.curve, self.lo, self.hi = curve, lo, hi

    def at(self, tick):
        v = self.curve.percent_at(tick)
        return self.lo if v < self.lo else self.hi if v > self.hi else v

    def straight_from(self, tick):
        """whether the curve is a straight line (or flat) from `tick` to the next key."""
        keys = self.curve.keys
        i = bisect_right(self.curve.ticks, tick) - 1
        if i < 0 or i >= len(keys) - 1:
            return True
        a, b = keys[i], keys[i + 1]
        return a.hold or (a.out is None and b.inc is None and not a.ease and not b.ease)

    def before(self, tick):
        """the value just before `tick`: a held key's own value at the key that ends the hold."""
        keys = self.curve.keys
        j = bisect_left(self.curve.ticks, tick)
        if 0 < j < len(keys) and keys[j].tick == tick and keys[j - 1].hold:
            return keys[j - 1].value
        return self.at(tick)


def plain_value(pid, kind, text):
    """one plain value of a parameter, as fx-core's ParamSpec::parse reads it."""
    s = text.strip()
    try:
        if kind == "float":
            value = float(s)
            if not math.isfinite(value):
                raise ValueError
            return value
        if kind == "popup":
            return int(s)
    except ValueError:
        raise MapError(f"--param {pid}: cannot read `{text}`") from None
    if s in ("1", "true", "on"):
        return True
    if s in ("0", "false", "off"):
        return False
    raise MapError(f"--param {pid}: cannot read `{text}`")


def track(pid, text, step, per_second):
    """the curve of one parameter from its --param text (None: the default)."""
    kind, default, valid = PARAMS[pid]
    if text is None:
        return Const(default)
    if "@" not in text:
        return Const(plain_value(pid, kind, text))
    if kind == "float":
        lo, hi = valid
        limit = max(abs(lo), abs(hi))
        try:
            keys = sc.parse_keys(text, step, per_second, limit)
        except sc.CurveError as e:
            raise MapError(f"--param {pid}: {e}") from None
        return Floats(sc.Curve(keys, limit), lo, hi)
    keyed = []
    for piece in text.split(";"):
        value, at, frame = piece.rpartition("@")
        frame = frame.strip()
        if frame[-1:] in ("h", "e"):
            frame = frame[:-1]
        try:
            number = float(frame)
        except ValueError:
            raise MapError(f"--param {pid}: `{piece}` is not value@frame") from None
        if not at or not math.isfinite(number):
            raise MapError(f"--param {pid}: `{piece}` is not value@frame")
        keyed.append((sc.round_half_away(number * step), plain_value(pid, kind, value)))
    keyed.sort(key=lambda k: k[0])
    if any(a[0] == b[0] for a, b in zip(keyed, keyed[1:])):
        raise MapError(f"--param {pid}: two keyframes at the same time")
    return Steps([k[0] for k in keyed], [k[1] for k in keyed])


def read_params(params, fps):
    """the K=V texts of --param as a curve for every id the map reads, at vfx's default time base."""
    given = {}
    for text in params:
        key, sep, value = text.partition("=")
        if not sep or not key:
            raise MapError(f"--param {text}: expected K=V")
        given[key] = value
    step, per_second = sc.ticks_per_frame(fps), sc.ticks_per_second(fps)
    return {pid: track(pid, given.get(pid), step, per_second) for pid in PARAMS}


# ----------------------------------------------------------------------------
# the time map
# ----------------------------------------------------------------------------

def snap(rate):
    for target in SNAP_RATES:
        if abs(sc.f32(rate - target)) < SNAP_TOLERANCE:
            rate = target
    return rate


def rate_ratio(input_rate, comp_rate, in_is_out, fields):
    """out_fps / in_fps of the engine's rate model (timeline.rs Rates::new): the source rate is the typed rate
    narrowed to f32, In FPS is Out FPS makes both 30, both snap, and fields double the source rate."""
    in_fps, out_fps = sc.f32(input_rate), comp_rate
    if in_is_out:
        in_fps = out_fps = 30.0
    in_fps, out_fps = snap(in_fps), snap(out_fps)
    if fields in (2, 3):
        in_fps *= 2.0
    return out_fps / in_fps


def whole_frames(offset):
    """trunc(src_time_offset) as the engine takes it: the 16.16 fixed value narrowed to f32, times 2^-16."""
    return int(sc.f32(float(sc.fixed(offset))) / 65536.0)


class HalfSums:
    """the Speed mode sum after k half frames, as speed_curve.walk runs it (f32), extended on demand."""

    def __init__(self, speed, step):
        self.at, self.step, self.sums = speed.at, step, [0.0]

    def __getitem__(self, k):
        sums, f32, fixed = self.sums, sc.f32, sc.fixed
        while len(sums) <= k:
            tick = int(math.floor(self.step * len(sums) / 2.0 + sc.SAMPLE_BIAS))
            sp = f32(fixed(self.at(tick)) / 6553600.0)
            sums.append(f32(f32(sp / 2.0) + sums[-1]))
        return sums[k]


def frame_num_points(frame_num, n, step):
    """the knots of Frame Number mode over output frame n, in ticks: frame_num at the frame's ends and at every
    key inside it, FRAME_NUM_STEPS points a frame where the curve bends, the value before a key where a hold
    ends (so the jump is a pair of knots at one time)."""
    t0, t1 = n * step, (n + 1) * step
    if not isinstance(frame_num, Floats):
        v = frame_num.at(t0)
        return [(t0, v), (t1, v)]
    ticks = frame_num.curve.ticks
    cuts = [t0] + ticks[bisect_right(ticks, t0):bisect_left(ticks, t1)] + [t1]
    points = []
    for a, b in zip(cuts, cuts[1:]):
        points.append((a, frame_num.at(a)))
        if not frame_num.straight_from(a):
            m = -(-(b - a) * FRAME_NUM_STEPS // step)
            for j in range(1, m):
                tick = a + (b - a) * j / m
                points.append((tick, frame_num.at(tick)))
        points.append((b, frame_num.before(b)))
    return points


def add(knots, point):
    if not knots or knots[-1] != point:
        knots.append(point)


def build_knots(effect, params, fps, frames, clip_fps=None):
    """the time map of `effect` with the --param texts `params` over output frames 0..frames at the output rate
    `fps`, as knots [(t, p)]: t in output frames, increasing; p in clip frames at the clip rate `clip_fps`
    (default: the output rate), linear between knots. two knots at one t are a jump."""
    fps = to_rate(fps)
    frames = int(frames)
    if frames < 1:
        raise MapError("the output needs at least one frame")
    if effect not in TIMED:
        knots = [(0.0, 0.0), (float(frames), float(frames))]
    else:
        step = sc.ticks_per_frame(fps)
        p = read_params(params, fps)
        offset, fields = "src_time_offset" in TIMED[effect], "input_fields" in TIMED[effect]
        sums = HalfSums(p["speed_percent"], step)
        comp = float(fps)
        knots = []
        for n in range(frames):
            tick = n * step
            if p["display"].at(tick) == 2:
                w = whole_frames(p["src_time_offset"].at(tick)) if offset else 0
                points = [(float(n), float(n + w)), (float(n + 1), float(n + 1 + w))]
            elif p["time_remap_mode"].at(tick) == 2:
                points = [(t / step, v) for t, v in frame_num_points(p["frame_num"], n, step)]
            else:
                extra = 0.0
                w = whole_frames(p["src_time_offset"].at(tick)) if offset else 0
                if w:
                    extra = w * rate_ratio(p["input_frame_rate"].at(tick), comp, p["in_fps_is_out_fps"].at(tick),
                                           p["input_fields"].at(tick) if fields else 1)
                points = [(float(n), sums[2 * n] + extra), (n + 0.5, sums[2 * n + 1] + extra),
                          (float(n + 1), sums[2 * n + 2] + extra)]
            for point in points:
                add(knots, point)
    if clip_fps is not None:
        scale = to_rate(clip_fps) / fps
        if scale != 1:
            knots = [(t, v * float(scale)) for t, v in knots]
    return knots


def value_at(knots, t, times=None):
    """p at output position t: the later knot of a jump at t, linear between knots, the end values outside."""
    times = times if times is not None else [k[0] for k in knots]
    i = bisect_right(times, t) - 1
    if i < 0:
        return knots[0][1]
    ta, pa = knots[i]
    if ta == t or i == len(knots) - 1:
        return pa
    tb, pb = knots[i + 1]
    return pa + (pb - pa) * (t - ta) / (tb - ta)


def positions(knots, frames):
    """the clip position each output frame 0..frames-1 shows."""
    times = [k[0] for k in knots]
    return [value_at(knots, float(n), times) for n in range(frames)]


# ----------------------------------------------------------------------------
# the audio
# ----------------------------------------------------------------------------

def output_samples(frames, rate, fps):
    """floor(frames * rate / fps + 1/2): the samples a channel of `frames` output frames has."""
    return math.floor(Fraction(int(frames)) * int(rate) / to_rate(fps) + Fraction(1, 2))


def fade_length(rate):
    """the samples of a 2 ms fade, at least one."""
    return max(1, int(round(FADE_SECONDS * int(rate))))


def _ceil(x):
    return -((-x.numerator) // x.denominator)


def sample_shift(offset, rate):
    """offset seconds (audio start - video start) as whole samples at `rate`, the nearest one."""
    return Fraction(round(Fraction(offset) * int(rate)))


def pieces(knots, fps, rate, offset, samples, clip_fps=None):
    """the output samples 0..samples-1 as linear pieces (i0, i1, A, B): output sample i of [i0, i1) plays input
    sample position A + B*i, A and B exact Fractions. pieces in a row with the same A and B are one."""
    fps = to_rate(fps)
    clip = to_rate(clip_fps) if clip_fps is not None else fps
    sr = Fraction(int(rate))
    shift = sample_shift(offset, rate)
    out = []
    for (ta, pa), (tb, pb) in zip(knots, knots[1:]):
        if not tb > ta:
            continue
        fa, fb = Fraction(ta), Fraction(tb)
        i0, i1 = max(0, _ceil(fa * sr / fps)), min(samples, _ceil(fb * sr / fps))
        if i1 <= i0:
            continue
        slope = (Fraction(pb) - Fraction(pa)) / (fb - fa)
        b = slope * fps / clip
        a = (Fraction(pa) - slope * fa) * sr / clip - shift
        if out and out[-1][1] == i0 and out[-1][2] == a and out[-1][3] == b:
            out[-1] = (out[-1][0], i1, a, b)
        else:
            out.append((i0, i1, a, b))
    return out


def _first(pred, lo, hi):
    """the smallest i in [lo, hi) with pred(i), hi when there is none; pred is False then True."""
    while lo < hi:
        mid = (lo + hi) // 2
        if pred(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def sound_spans(parts, low, high, last):
    """the part of each piece that sounds, as (k0, k1, a, b, whole): the source position s = a + b*i in double
    must satisfy low <= s < high and s <= last; a piece that stands still (B = 0) is a hold and stays silent.
    whole is the whole sample shift of a piece with B = 1 and a whole A (copied), else None."""
    spans = []
    for i0, i1, fa, fb in parts:
        if fb == 0:
            continue
        a, b = float(fa), float(fb)

        def above(i):
            return a + b * i >= low

        def below(i):
            s = a + b * i
            return s < high and s <= last

        if b > 0:
            k0 = _first(above, i0, i1)
            k1 = _first(lambda i: not below(i), k0, i1)
        else:
            k0 = _first(below, i0, i1)
            k1 = _first(lambda i: not above(i), k0, i1)
        if k1 > k0:
            whole = int(fa) if fb == 1 and fa.denominator == 1 else None
            spans.append((k0, k1, a, b, whole))
    return spans


def _chunk(src, channels, frames_in, a, b, whole, c0, c1):
    """output samples c0..c1-1 of one span, interleaved, before fades."""
    n = c1 - c0
    if whole is not None:
        start = (whole + c0) * channels
        out = array("f")
        out.frombytes(src[start:start + n * channels].tobytes())
        return out
    vals = [a + b * i for i in range(c0, c1)]
    js = [int(s) for s in vals]
    lo = min(js[0], js[-1])
    hi = min(max(js[0], js[-1]) + 1, frames_in - 1)
    rel = [j - lo for j in js]
    fs = [s - j for s, j in zip(vals, js)]
    if channels == 1:
        w = src[lo:hi + 1].tolist()
        w.append(0.0)
        return array("f", [w[j] + (w[j + 1] - w[j]) * f for j, f in zip(rel, fs)])
    out = array("f", bytes(4 * n * channels))
    for c in range(channels):
        w = src[lo * channels + c:(hi + 1) * channels:channels].tolist()
        w.append(0.0)
        out[c::channels] = array("f", [w[j] + (w[j + 1] - w[j]) * f for j, f in zip(rel, fs)])
    return out


def blocks(knots, src, channels, rate, offset, fps, clip_frames, samples, clip_fps=None):
    """the sounding output samples as (first sample, interleaved f32 array) in order; every sample not given
    is silence. src is the input as a 1-D float memoryview (or array), channels interleaved."""
    channels = int(channels)
    frames_in = len(src) // channels
    fps = to_rate(fps)
    clip = to_rate(clip_fps) if clip_fps is not None else fps
    sr = int(rate)
    shift = sample_shift(offset, sr)
    low = max(0.0, float(-shift))
    high = float(Fraction(int(clip_frames)) / clip * sr - shift)
    spans = sound_spans(pieces(knots, fps, sr, offset, samples, clip_fps), low, high, frames_in - 1)
    # runs of sound: spans that touch (a jump between them cuts without a fade)
    runs, owner = [], []
    for k0, k1, *_ in spans:
        if runs and runs[-1][1] == k0:
            runs[-1][1] = k1
        else:
            runs.append([k0, k1])
        owner.append(len(runs) - 1)
    width = fade_length(sr)
    for (k0, k1, a, b, whole), r in zip(spans, owner):
        start, end = runs[r]
        fade_in, fade_out = start > 0, end < samples
        for c0 in range(k0, k1, BLOCK):
            c1 = min(k1, c0 + BLOCK)
            out = _chunk(src, channels, frames_in, a, b, whole, c0, c1)
            edges = []
            if fade_in and c0 < start + width:
                edges.append(range(c0, min(c1, start + width)))
            if fade_out and c1 > end - width:
                edges.append(range(max(c0, end - width), c1))
            for i in sorted(set(i for e in edges for i in e)):
                gain = 1.0
                if fade_in:
                    gain = min(gain, (i - start + 1) / width)
                if fade_out:
                    gain = min(gain, (end - i) / width)
                for c in range(channels):
                    k = (i - c0) * channels + c
                    out[k] = out[k] * gain
            yield c0, out


def as_floats(pcm):
    """a buffer of f32 samples as a 1-D float memoryview."""
    view = memoryview(pcm)
    if view.format != "f":
        view = view.cast("B")
        view = view[:len(view) - len(view) % 4].cast("f")
    return view


def render(knots, pcm, channels, rate, offset, fps, clip_frames, samples, clip_fps=None):
    """the input `pcm` (interleaved f32 samples of `channels` channels at `rate`) played along `knots`
    (build_knots): an array('f') of `samples` samples per channel. offset is the audio start minus the video
    start in seconds, clip_frames the frames of the clip, clip_fps its rate (default: the output rate fps)."""
    channels = int(channels)
    out = array("f", bytes(4 * samples * channels))
    for start, block in blocks(knots, as_floats(pcm), channels, rate, offset, fps, clip_frames, samples, clip_fps):
        out[start * channels:start * channels + len(block)] = block
    return out


def render_file(knots, path_in, path_out, channels, rate, offset, fps, clip_frames, samples, clip_fps=None):
    """render() from a raw pcm_f32le file to another, reading the input mapped and writing in blocks."""
    channels = int(channels)
    zero = bytes(4 * channels * BLOCK)

    def silence(handle, count):
        while count > 0:
            step = min(count, BLOCK)
            handle.write(zero[:4 * channels * step])
            count -= step

    with open(path_in, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        swap = sys.byteorder != "little"
        mapped = None
        if size < 4 or swap:
            data = array("f")
            data.frombytes(f.read(size - size % 4))
            if swap:
                data.byteswap()
            src = memoryview(data)
        else:
            mapped = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            src = memoryview(mapped)[:size - size % 4].cast("f")
        try:
            with open(path_out, "wb") as out:
                at = 0
                for start, block in blocks(knots, src, channels, rate, offset, fps, clip_frames, samples, clip_fps):
                    silence(out, start - at)
                    if swap:
                        block.byteswap()
                    out.write(block.tobytes())
                    at = start + len(block) // channels
                silence(out, samples - at)
        finally:
            src.release()
            if mapped is not None:
                mapped.close()


# ----------------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------------

def whole_number(least):
    def parse(text):
        try:
            n = int(text)
        except ValueError:
            n = least - 1
        if n < least:
            raise argparse.ArgumentTypeError(f"expected a whole number of {least} or more")
        return n
    return parse


def seconds(text):
    try:
        value = float(text)
    except ValueError:
        value = float("nan")
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("expected a number of seconds")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="audio_map.py",
        description="play the audio of a clip along the time map of a retiming effect (varispeed), or print "
                    "the clip position of every output frame.")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--effect", required=True, help="the effect id, as for apply.py")
    common.add_argument("--fps", required=True, help="the output frame rate, N/D or a number")
    common.add_argument("--frames", required=True, type=whole_number(1), metavar="N", help="output frames")
    common.add_argument("--param", action="append", default=[], metavar="K=V",
                        help="the effect's --param values, as vfx gets them (ids the map does not read are passed "
                             "over), repeatable")
    common.add_argument("--clip-fps", metavar="R", help="the rate the clip runs at (default: --fps)")
    render_cmd = sub.add_parser("render", parents=[common],
                                help="write the retimed audio of one stream as raw pcm_f32le")
    render_cmd.add_argument("--clip-frames", required=True, type=whole_number(0), metavar="C",
                            help="frames of the clip the effect reads")
    render_cmd.add_argument("--rate", required=True, type=whole_number(1), metavar="SR", help="sample rate")
    render_cmd.add_argument("--channels", required=True, type=whole_number(1), metavar="CH", help="channels")
    render_cmd.add_argument("--offset", type=seconds, default=0.0, metavar="S",
                            help="the audio stream's start minus the video stream's start, in seconds (default 0)")
    render_cmd.add_argument("--in", dest="path_in", required=True, metavar="A.f32", help="input, raw pcm_f32le")
    render_cmd.add_argument("--out", dest="path_out", required=True, metavar="B.f32", help="output, raw pcm_f32le")
    show = sub.add_parser("positions", parents=[common], help="print the clip position of every output frame")
    show.add_argument("--json", action="store_true", help="print the positions and the knots as JSON")
    args = parser.parse_args(argv)
    try:
        fps = to_rate(args.fps)
        clip_fps = to_rate(args.clip_fps) if args.clip_fps is not None else None
        knots = build_knots(args.effect, args.param, fps, args.frames, clip_fps)
    except MapError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    if args.command == "positions":
        shown = positions(knots, args.frames)
        if args.json:
            print(json.dumps({"effect": args.effect, "fps": f"{fps.numerator}/{fps.denominator}",
                              "clip_fps": None if clip_fps is None else f"{clip_fps.numerator}/{clip_fps.denominator}",
                              "frames": args.frames, "positions": shown, "knots": [list(k) for k in knots]}))
        else:
            for n, p in enumerate(shown):
                print(f"{n} {p:.6f}")
        return 0
    samples = output_samples(args.frames, args.rate, fps)
    try:
        render_file(knots, args.path_in, args.path_out, args.channels, args.rate, args.offset, fps,
                    args.clip_frames, samples, clip_fps)
    except OSError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
