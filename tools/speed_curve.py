#!/usr/bin/env python3
"""speed_curve.py: check a speed curve before a long render.

the retime effect shows, at output frame n, the source frame P(n) = sum for j = 1..n of
(S(j - 1/2) + S(j)) / 2, with S the speed percent divided by 100. this tool runs the same sum over the
keys you would pass as --param speed_percent=..., so you can see before rendering which source moments
land where, whether the curve leaves the clip, how much of the output needs interpolated frames and
where the speed bends or jumps.

  python tools/speed_curve.py --keys "100@0e;100@120e;150@168e;40@204e;40@400e;100@454e;100@664e" --fps 60 --clip-frames 630
  python tools/speed_curve.py --keys "100@0;40@60;100@120" --fps 24 --clip-frames 240 --start 13 --mark hit=14.2

  --keys K            the keyframes V@F;V@F as for --param speed_percent: F is the output frame (a
                      trailing h holds the key, a trailing e eases it; iS/I and oS/I after F are the
                      key's incoming and outgoing Bezier handles, S percent points a second and I the
                      influence in percent). quote it in a shell, the ; ends a command there. a
                      leading speed_percent= is accepted.
  --fps R             the frame rate of the clip and of the output, 60, 24 or a fraction such as
                      30000/1001. the clip is assumed to play at the output rate.
  --clip-frames N     frames in the clip the effect reads (the trimmed clip, not the whole recording).
  --frames N          output frames to check (default: the frame of the last key).
  --start SECONDS     where the clip begins in the recording, so source times and --mark times are
                      recording times (default: the clip's own times).
  --mark NAME=SECONDS repeatable. a source moment (recording time with --start) and the output time at
                      which it shows.

exit codes: 0 ok, 1 bad --keys or a curve with no length, 2 a malformed command line, 3 the curve
reads outside the clip.
stdlib only. the numbers follow the effect's own arithmetic (half frame steps, 32 bit sums), so the
positions agree with a render to a few thousandths of a frame.
"""

import argparse
import math
import re
import struct
import sys
from bisect import bisect_right
from collections import namedtuple
from fractions import Fraction
from itertools import islice

# the effect reads the curve 0.1 tick after each half frame, then drops the fraction of a tick
SAMPLE_BIAS = 0.1
# a position this close to a whole frame shows that frame as it is, with no interpolation
EXACT = 0.005
# the tail search accepts a position this close to a whole frame, inside the tolerance above
LOCK_ERROR = 0.002
SPEED_LIMIT = 10000.0
# the largest handle speed either way (vfx's bound): a handle's reach stays finite on any segment
HANDLE_SPEED_LIMIT = 1e9
# a plain key that changes the slope of the speed by more than this many points a frame is reported
BEND = 1.0
# a rough cost of one interpolated frame at 1080p, for the time estimate only
SECONDS_PER_INTERPOLATED_FRAME = 0.25

# inc and out are the key's ease handles (Handle or None): inc shapes the segment into the key, out the one
# after it
Key = namedtuple("Key", "frame tick value hold ease inc out", defaults=(None, None))
# speed in percent points a second and influence in percent as read, per_tick the speed divided once by the
# ticks a second (what the curve uses), text the S/I as written
Handle = namedtuple("Handle", "speed influence per_tick text")
HANDLE_CHARS = "0123456789./-"


class CurveError(Exception):
    pass


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def round_half_away(x):
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


def fixed(percent):
    """the 16.16 integer the effect works with, rounded half away from zero."""
    return int(percent * 65536.0 + (-0.5 if percent < 0 else 0.5))


def eased_progress(u, ease_start, ease_end):
    """progress along a segment reshaped by the easing of its two keys (cubic with zero slope at an
    eased key and the straight line's slope at a plain one)."""
    if ease_start and ease_end:
        return u * u * (3.0 - 2.0 * u)
    if ease_start:
        return u * u * (2.0 - u)
    if ease_end:
        return u * (1.0 + u * (1.0 - u))
    return u


def ticks_per_frame(fps):
    """the layer time base the effect samples in: 100 ticks a frame at whole rates, the rate's own
    denominator at fractional ones (1001 at 30000/1001)."""
    return 100 if fps.denominator == 1 else fps.denominator


def ticks_per_second(fps):
    """the ticks a second of that time base: 2400 at 24, 30000 at 30000/1001."""
    return ticks_per_frame(fps) * fps.numerator // fps.denominator


def handle_start(frame):
    """the index of the first i or o followed by a digit or -, where the handles begin, or None."""
    for i, c in enumerate(frame[:-1]):
        if c in ("i", "o") and frame[i + 1] in "0123456789-":
            return i
    return None


def is_handle_number(text):
    return re.fullmatch(r"-?[0-9]+(\.[0-9]+)?", text) is not None


def not_a_handle(frame, text):
    return f"`{frame}`: `{text}` is not an ease handle (i or o, then SPEED/INFLUENCE, as in i0/33.3)"


def parse_handle(frame, text, per_second):
    """the handle at the start of text (i or o, then every character a handle can hold) and the text
    after it, as vfx reads it."""
    n = 1
    while n < len(text) and text[n] in HANDLE_CHARS:
        n += 1
    token, rest = text[:n], text[n:]
    speed, slash, influence = token[1:].partition("/")
    if not slash or not is_handle_number(speed) or not is_handle_number(influence):
        raise CurveError(not_a_handle(frame, token))
    if not -HANDLE_SPEED_LIMIT <= float(speed) <= HANDLE_SPEED_LIMIT:
        raise CurveError(f"handle speed {speed} is outside -{HANDLE_SPEED_LIMIT:.0f}..={HANDLE_SPEED_LIMIT:.0f}")
    if not 0.1 <= float(influence) <= 100:
        raise CurveError(f"influence {influence} is outside 0.1..=100")
    if per_second is None:
        raise CurveError(f"`{frame}` has an ease handle, which needs the ticks a second (per_second)")
    return Handle(float(speed), float(influence), float(speed) / per_second, token[1:]), rest


def split_frame(frame, per_second):
    """F[iS/I][oS/I][h|e] as (number text, hold, ease, inc, out); None when the frame has no handle, so
    the suffix rules of before apply."""
    start = handle_start(frame)
    if start is None:
        return None
    rest, inc, out = frame[start:], None, None
    if rest.startswith("i"):
        inc, rest = parse_handle(frame, rest, per_second)
    if rest.startswith("o") and handle_start(rest) == 0:
        out, rest = parse_handle(frame, rest, per_second)
    if rest == "e":
        raise CurveError("a key with ease handles cannot also end in e")
    if rest not in ("", "h"):
        raise CurveError(not_a_handle(frame, rest))
    if rest == "h" and out is not None:
        raise CurveError("a held key has no outgoing handle")
    return frame[:start], rest == "h", False, inc, out


def parse_keys(text, step, per_second=None, limit=SPEED_LIMIT):
    """the keys of text at `step` ticks a frame. per_second, the ticks a second, is needed only by keys
    with ease handles; limit bounds the values (the speed_percent range by default)."""
    text = text.strip()
    if re.match(r"^[A-Za-z_]\w*=", text):
        text = text.split("=", 1)[1]
    keys = []
    for piece in text.split(";"):
        value, at, frame = piece.rpartition("@")
        if not at:
            raise CurveError(f"`{piece}` has no @frame")
        frame = frame.strip()
        hold = ease = False
        inc = out = None
        handles = split_frame(frame, per_second)
        if handles is not None:
            number, hold, ease, inc, out = handles
        elif frame.endswith("h"):
            hold, number = True, frame[:-1]
        elif frame.endswith("e"):
            ease, number = True, frame[:-1]
        else:
            number = frame
        try:
            number = float(number)
            percent = float(value)
        except ValueError:
            raise CurveError(f"`{piece}` is not value@frame (a frame may end in h to hold or e to ease, not both)") from None
        if not math.isfinite(number) or not math.isfinite(percent):
            raise CurveError(f"`{piece}` is not a finite number")
        if abs(percent) > limit:
            raise CurveError(f"speed {percent:g} is outside -{limit:g}..{limit:g}")
        keys.append(Key(number, round_half_away(number * step), percent, hold, ease, inc, out))
    keys.sort(key=lambda k: k.tick)
    if any(a.tick == b.tick for a, b in zip(keys, keys[1:])):
        raise CurveError("two keyframes at the same time")
    return keys


def bezier(u, n, va, d3, a, b):
    """the value at progress u of a segment n ticks long from va to va + d3 with a handle at either end,
    in the operation order of fx-core's curves.rs. a handle (per tick speed S, influence I) gives the
    control point (I/100, (S*(I/100))*n) out of a and (1 - I/100, d3 - (S*(I/100))*n) into b, an eased
    end (1/3, 0) or (1 - 1/3, d3), a plain one (1/3, d3/3) or (1 - 1/3, d3 - d3/3); 64 bisection steps
    find the curve parameter."""
    if a.out is not None:
        x1, d1 = a.out.influence / 100, (a.out.per_tick * (a.out.influence / 100)) * n
    elif a.ease:
        x1, d1 = 1 / 3, 0.0
    else:
        x1, d1 = 1 / 3, d3 / 3
    if b.inc is not None:
        x2, d2 = 1 - b.inc.influence / 100, d3 - (b.inc.per_tick * (b.inc.influence / 100)) * n
    elif b.ease:
        x2, d2 = 1 - 1 / 3, d3
    else:
        x2, d2 = 1 - 1 / 3, d3 - d3 / 3
    lo, hi = 0.0, 1.0
    for _ in range(64):
        m = (lo + hi) * 0.5
        r = 1 - m
        x = ((3 * r * r * m) * x1 + (3 * r * m * m) * x2) + m * m * m
        if x < u:
            lo = m
        else:
            hi = m
    s = (lo + hi) * 0.5
    r = 1 - s
    return va + (((3 * r * r * s) * d1 + (3 * r * s * s) * d2) + (s * s * s) * d3)


class Curve:
    def __init__(self, keys, limit=SPEED_LIMIT):
        self.keys = keys
        self.ticks = [k.tick for k in keys]
        self.limit = limit

    def percent_at(self, tick):
        keys = self.keys
        if tick <= keys[0].tick:
            return keys[0].value
        if tick >= keys[-1].tick:
            return keys[-1].value
        i = bisect_right(self.ticks, tick) - 1
        a, b = keys[i], keys[i + 1]
        if a.hold:
            return a.value
        if a.out is None and b.inc is None:
            u = eased_progress((tick - a.tick) / (b.tick - a.tick), a.ease, b.ease)
            return a.value + (b.value - a.value) * u
        if tick == a.tick:
            return a.value
        n = b.tick - a.tick
        y = bezier((tick - a.tick) / n, n, a.value, b.value - a.value, a, b)
        # a handle may overshoot; the slider stays inside its range, as the effect clamps it
        if y < -self.limit:
            return -self.limit
        if y > self.limit:
            return self.limit
        return y


def walk(curve, step, first=0, out=0.0):
    """the source position after each output frame from first + 1 on: the effect's half frame sum in 32 bit
    floats, resumed from `out` at frame `first`."""
    n = first
    while True:
        n += 1
        for k in (2 * n - 1, 2 * n):
            tick = int(math.floor(step * k / 2.0 + SAMPLE_BIAS))
            sp = f32(fixed(curve.percent_at(tick)) / 6553600.0)
            out = f32(f32(sp / 2.0) + out)
        yield out


def positions(curve, step, frames):
    """the source frame shown by output frames 0..frames."""
    return [0.0] + list(islice(walk(curve, step), frames))


def whole_frame_error(p):
    return abs(p - round(p))


Lock = namedtuple("Lock", "first base shift error")


def lock_tail(keys, step):
    """how far to move the final ramp-out so the constant run after it shows whole source frames.

    the run needs a speed that is a multiple of 100% (one whole source frame per output frame at 100%,
    two at 200%), and it sits off the grid by whatever fraction of a frame the slow or fast part before
    it left over. moving the keys from the ramp-out on by a fraction of a frame changes that fraction.
    returns None when the curve has no such tail, else a Lock: the index of the first key to move, how
    far off the grid the run is now (in frames), the smallest move that fixes it in ticks (0 when it
    is fine already, None when no move of up to three frames does) and the error that move leaves."""
    final = keys[-1].value
    t = len(keys) - 1
    while t > 0 and keys[t - 1].value == final and not keys[t - 1].hold and flat(keys[t - 1], keys[t]):
        t -= 1
    if t == 0 or final == 0 or final % 100:
        return None
    j = t - 1
    before = keys[j - 1] if j > 0 else None
    # half steps up to here read only keys that do not move
    resume = max(0, before.tick // step - 1) if before else 0
    out = ([0.0] + list(islice(walk(Curve(keys), step), resume)))[resume]

    def error(dt):
        moved = moved_keys(keys, j, dt, step)
        n = max(resume + 1, -(-moved[t].tick // step))
        return whole_frame_error(next(islice(walk(Curve(moved), step, resume, out), n - resume - 1, None)))

    base = error(0)
    if base <= EXACT:
        return Lock(j, base, 0, base)
    for size in range(1, 3 * step):
        for dt in (size, -size):
            e = error(dt)
            if e <= LOCK_ERROR:
                return Lock(j, base, dt, e)
    return Lock(j, base, None, base)


def flat(a, b):
    """whether the segment from a to b, two keys of one value, stays at that value: a handle with a
    speed bends it."""
    return all(h is None or h.speed == 0 for h in (a.out, b.inc))


def handles_text(k):
    return (f"i{k.inc.text}" if k.inc else "") + (f"o{k.out.text}" if k.out else "")


def mode_text(k):
    """the ease column of the key table: hold, ease or -, or the handles as written (and hold)."""
    if k.inc or k.out:
        parts = (f"i{k.inc.text}" if k.inc else "", f"o{k.out.text}" if k.out else "", "hold" if k.hold else "")
        return " ".join(part for part in parts if part)
    return "hold" if k.hold else "ease" if k.ease else "-"


def key_text(k):
    frame = f"{k.frame:.6f}".rstrip("0").rstrip(".")
    return f"{k.value:g}@{frame}{handles_text(k)}{'h' if k.hold else 'e' if k.ease else ''}"


def moved_keys(keys, first, dt, step):
    return [k if i < first else k._replace(tick=k.tick + dt, frame=k.frame + dt / step) for i, k in enumerate(keys)]


def runs(frames):
    """consecutive output frame numbers as (first, last) ranges."""
    ranges = []
    for n in frames:
        if ranges and ranges[-1][1] == n - 1:
            ranges[-1][1] = n
        else:
            ranges.append([n, n])
    return ranges


def crossings(pos, target):
    """the output frames (fractional) at which the source position first reaches `target`, once for
    each pass: more than one when the curve reverses."""
    found = [0.0] if pos[0] == target else []
    for n in range(len(pos) - 1):
        a, b = pos[n], pos[n + 1]
        if a < target <= b or b <= target < a:
            found.append(n + (target - a) / (b - a))
    return found


def parse_mark(text):
    name, eq, seconds = text.rpartition("=")
    if not eq or not name:
        raise argparse.ArgumentTypeError("expected NAME=SECONDS")
    try:
        return name, float(seconds)
    except ValueError:
        raise argparse.ArgumentTypeError(f"`{seconds}` is not a number of seconds") from None


def parse_fps(text):
    try:
        fps = Fraction(text)
    except (ValueError, ZeroDivisionError):
        raise argparse.ArgumentTypeError(f"`{text}` is not a frame rate") from None
    if fps <= 0:
        raise argparse.ArgumentTypeError("the frame rate must be positive")
    return fps


def analyse(keys, fps, clip_frames, frames, start, marks):
    """everything the report prints, as plain data."""
    step = ticks_per_frame(fps)
    rate = float(fps)
    curve = Curve(keys)
    pos = positions(curve, step, frames)
    shown = pos[:frames]
    speed = [curve.percent_at(n * step) for n in range(frames)]
    last_frame = clip_frames - 1
    outside = [n for n, p in enumerate(shown) if p < -EXACT or p > last_frame + EXACT]
    interpolated = sum(1 for p in shown if min(p - math.floor(p), math.ceil(p) - p) > EXACT)
    deltas = [(abs(speed[n] - speed[n - 1]), n) for n in range(1, frames)]
    biggest = max(deltas) if deltas else (0.0, 0)
    tail = frames - 1
    while tail > 0 and abs(speed[tail - 1] - speed[-1]) < 0.5:
        tail -= 1
    bends = []
    for i in range(1, len(keys) - 1):
        k = keys[i]
        before, after = keys[i - 1], keys[i + 1]
        if k.hold or k.ease or before.hold or k.inc or k.out:
            continue
        left = curve.percent_at(k.tick) - curve.percent_at(k.tick - step)
        right = curve.percent_at(k.tick + step) - curve.percent_at(k.tick)
        if abs(right - left) >= BEND:
            bends.append((k, left, right))
    hits = []
    for name, seconds in marks:
        target = (seconds - start) * rate
        hits.append((name, seconds, target, crossings(pos[:frames], target)))
    return {
        "pos": pos, "speed": speed, "outside": runs(outside), "interpolated": interpolated,
        "biggest": biggest, "tail": frames - tail, "bends": bends, "hits": hits, "step": step,
        "lock": lock_tail(keys, step),
    }


def clock(seconds):
    return f"{seconds:.2f} s"


def report(keys, fps, clip_frames, frames, start, marks, shown_start):
    a = analyse(keys, fps, clip_frames, frames, start, marks)
    rate = float(fps)
    pos, speed = a["pos"], a["speed"]
    lines = []
    out = lines.append
    out(f"speed curve: {len(keys)} keys, {rate:g} fps ({a['step']} ticks a frame)")
    out(f"output  {frames} frames, {clock(frames / rate)}")
    clip = f"clip    {clip_frames} frames, {clock(clip_frames / rate)}"
    if shown_start:
        clip += f" (recording {clock(start)} to {clock(start + clip_frames / rate)})"
    out(clip)
    out("")
    # the ease column widens for keys with handles, which it shows as written
    modes = [mode_text(k) for k in keys]
    wide = max([5] + [len(m) for m in modes])
    head = "key frame    time   speed  ease" + " " * (wide - 5) + "   source frame  source time"
    if shown_start:
        head += "  recording time"
    out(head)
    key_pos = positions(Curve(keys), a["step"], max(1, int(math.ceil(keys[-1].frame))))
    for k, mode in zip(keys, modes):
        n = k.frame
        whole = int(math.floor(n))
        p = key_pos[whole] if whole < len(key_pos) else key_pos[-1]
        if n != whole and whole + 1 < len(key_pos):
            p += (key_pos[whole + 1] - key_pos[whole]) * (n - whole)
        row = f"{n:9g} {clock(n / rate):>7} {k.value:6g}%  {mode:<{wide}} {p:12.1f}  {clock(p / rate):>11}"
        if shown_start:
            row += f"  {clock(start + p / rate):>14}"
        out(row)
    out("")
    end = pos[frames - 1]
    line = f"last output frame {frames - 1} shows source frame {end:.1f} ({clock(end / rate)} into the clip"
    line += f", recording {clock(start + end / rate)})" if shown_start else ")"
    out(line)
    slow = sum(1 for s in speed if s < 99.5)
    fast = sum(1 for s in speed if s > 100.5)
    out(f"speed   {min(speed):g}% to {max(speed):g}%: {clock(slow / rate)} slower than 100%, "
        f"{clock(fast / rate)} faster, {clock((frames - slow - fast) / rate)} at 100%")
    low = min(speed)
    if 0 < abs(low) < 100:
        out(f"        at the slowest, one source frame spans {100.0 / abs(low):.1f} output frames")
    out(f"ends    {clock(a['tail'] / rate)} at a constant {speed[-1]:g}%")
    lock = a["lock"]
    if lock is not None:
        if lock.base <= EXACT:
            out(f"tail    the final {speed[-1]:g}% run shows whole source frames, so it needs no interpolation")
        else:
            out(f"tail    the final {speed[-1]:g}% run sits {lock.base:.2f} of a frame off the grid of source frames, "
                f"so all {a['tail']} of its frames are interpolated")
            if lock.shift is None:
                out("        no move of up to three frames of the keys from the last ramp-out on puts it on the grid")
            else:
                fixed_keys = moved_keys(keys, lock.first, lock.shift, a["step"])
                out(f"        moving the keys from frame {keys[lock.first].frame:g} on by "
                    f"{lock.shift / a['step']:+g} frames puts it on whole frames ({lock.error:.3f} off):")
                out('        --keys "' + ";".join(key_text(k) for k in fixed_keys) + '"')
    share = 100.0 * a["interpolated"] / frames
    out(f"frames  {a['interpolated']} of {frames} ({share:.0f}%) need interpolation, about "
        f"{a['interpolated'] * SECONDS_PER_INTERPOLATED_FRAME / 60:.1f} min at 1080p")
    change, where = a["biggest"]
    if frames > 1:
        out(f"change  largest speed step between frames: {change:.2f} points at frame {where} "
            f"({speed[where - 1]:.1f}% to {speed[where]:.1f}%)")
    if a["hits"]:
        out("")
        out("marks")
        for name, seconds, target, found in a["hits"]:
            note = f"{name}: source {clock(seconds)} is clip frame {target:.1f}"
            if target < -EXACT or target > clip_frames - 1 + EXACT:
                out(f"  {note}, outside the clip")
            elif not found:
                out(f"  {note}, never reached by output frames 0 to {frames - 1} "
                    f"(they show source frames {min(pos[:frames]):.1f} to {max(pos[:frames]):.1f})")
            else:
                at = ", then ".join(f"output frame {n:.1f} ({clock(n / rate)})" for n in found)
                out(f"  {note}, shows at {at}")
    warnings = []
    for first, last in a["outside"]:
        warnings.append(f"output frames {first} to {last} read source frames outside the clip "
                        f"(0 to {clip_frames - 1}): positions {pos[first]:.1f} to {pos[last]:.1f}")
    for k, left, right in a["bends"]:
        warnings.append(f"the plain key at frame {k.frame:g} ({k.value:g}%) bends the speed from "
                        f"{left:+.2f} to {right:+.2f} points a frame; a trailing e would round it")
    if keys[-1].frame > frames:
        warnings.append(f"the last key (frame {keys[-1].frame:g}) is after the last output frame ({frames - 1})")
    out("")
    out("warnings" if warnings else "warnings: none")
    for w in warnings:
        out(f"  {w}")
    return lines, bool(a["outside"])


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="speed_curve.py",
        description="check a speed curve before a long render: where source moments land, whether the "
                    "curve leaves the clip, how much needs interpolation and where the speed bends.")
    parser.add_argument("--keys", required=True, metavar="K",
                        help="keyframes V@F;V@F as for --param speed_percent, a trailing h on F holds the "
                             "key and a trailing e eases it, iS/I and oS/I after F are incoming and outgoing "
                             "Bezier handles (S points a second, I percent influence); quote the argument, "
                             "the ; ends a command in a shell")
    parser.add_argument("--fps", required=True, type=parse_fps, metavar="R",
                        help="frame rate of the clip and the output: 60, 24 or a fraction such as 30000/1001")
    parser.add_argument("--clip-frames", required=True, type=int, metavar="N",
                        help="frames in the clip the effect reads (the trimmed clip)")
    parser.add_argument("--frames", type=int, metavar="N",
                        help="output frames to check (default: the frame of the last key)")
    parser.add_argument("--start", type=float, metavar="SECONDS",
                        help="where the clip begins in the recording; source times and --mark times are "
                             "then recording times")
    parser.add_argument("--mark", action="append", default=[], type=parse_mark, metavar="NAME=SECONDS",
                        help="a source moment to locate in the output, repeatable")
    args = parser.parse_args(argv)
    if args.clip_frames < 1:
        parser.error("--clip-frames must be at least 1")
    try:
        keys = parse_keys(args.keys, ticks_per_frame(args.fps), ticks_per_second(args.fps))
    except CurveError as e:
        print(f"[error] --keys: {e}", file=sys.stderr)
        return 1
    frames = args.frames if args.frames is not None else int(round_half_away(keys[-1].frame))
    if frames < 1:
        print("[error] give --frames: the last key is at frame 0", file=sys.stderr)
        return 1
    lines, outside = report(keys, args.fps, args.clip_frames, frames, args.start or 0.0, args.mark,
                            args.start is not None)
    print("\n".join(lines))
    return 3 if outside else 0


if __name__ == "__main__":
    sys.exit(main())
