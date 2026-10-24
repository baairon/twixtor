#!/usr/bin/env python3
"""repeat_keys.py: find the frames of a clip that repeat the frame before them, and print the keys that
replace each repeat by an in-between frame so the clip keeps its length and plays evenly.

a recording of a game or an app only stores the frames it drew. at a fixed output rate a late frame is
shown twice, and when the clip is slowed down each repeat becomes a hold followed by a jump. the retime
effect can show a position between two frames in Frame Number mode, so a repeat can be given the picture
halfway between its neighbours instead.

  python tools/repeat_keys.py work.mp4
  python tools/apply.py work.mp4 retime --param time_remap_mode=2 --param "frame_num=$(python tools/repeat_keys.py work.mp4)" --lossless -o fixed.mkv

  --threshold T   mean absolute difference (0..255, on a 160x90 grey copy) under which a frame counts as
                  a repeat of the one before (default 0.5; real motion is usually above 1)
  --list          print the repeated frame numbers instead of the keys

the keys go to stdout (the --param value for frame_num), a summary to stderr. a clip with a whole second of
still picture (a pause, a menu) has repeats that are meant: look at the list before using the keys.
"""

import argparse
import os
import subprocess
import sys
from operator import sub

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

WIDTH, HEIGHT = 160, 90


def grey_frames(path, ffmpeg):
    """the frames of a video as 160x90 grey byte strings, decoded by ffmpeg."""
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-i", path, "-an", "-sn", "-dn", "-vf",
           f"scale={WIDTH}:{HEIGHT}:flags=area,format=gray", "-f", "rawvideo", "-"]
    proc = subprocess.run(cmd, shell=False, capture_output=True)
    if proc.returncode:
        raise SystemExit(f"[error] ffmpeg failed: {proc.stderr.decode(errors='replace').strip()[-300:]}")
    size = WIDTH * HEIGHT
    data = proc.stdout
    return [data[i:i + size] for i in range(0, len(data) - size + 1, size)]


def mean_difference(a, b):
    return sum(map(abs, map(sub, a, b))) / len(a)


def find_repeats(frames, threshold):
    """the numbers of the frames that repeat the frame before them."""
    return [n for n in range(1, len(frames)) if mean_difference(frames[n - 1], frames[n]) < threshold]


def positions_for(count, repeats):
    """{frame: position} for the frames whose position is not their own number. a run of k repeats after
    frame i is spread evenly over the step from the last repeated picture to the next different one."""
    marked = set(repeats)
    out = {}
    n = 0
    while n < count:
        if n in marked:
            start = n
            while n + 1 in marked:
                n += 1
            run = n - start + 1
            for j in range(1, run + 1):
                if start - 1 + j < count:
                    out[start - 1 + j] = (start - 1 + run) + j / (run + 1)
        n += 1
    return out


def number(x):
    text = f"{x:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def keys_for(count, repeats):
    """the frame_num keys: the identity everywhere except at the repeats, with an identity key on each side of
    them so the line between keys stays straight."""
    moved = positions_for(count, repeats)
    frames = {0, count - 1}
    for n in moved:
        frames.update(f for f in (n - 1, n, n + 1) if 0 <= f < count)
    return ";".join(f"{number(moved.get(n, n))}@{n}" for n in sorted(frames))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="repeat_keys.py",
        description="find the frames that repeat the previous frame and print the frame_num keys that "
                    "replace each by the picture halfway to the next different frame.")
    parser.add_argument("input")
    parser.add_argument("--threshold", type=float, default=0.5, metavar="T",
                        help="mean absolute difference under which a frame is a repeat (default 0.5)")
    parser.add_argument("--list", action="store_true", help="print the repeated frame numbers instead of the keys")
    args = parser.parse_args(argv)

    import apply as tool
    ffmpeg = tool.ffmpeg_path()
    if not ffmpeg:
        print("[error] ffmpeg was not found", file=sys.stderr)
        return 1
    frames = grey_frames(args.input, ffmpeg)
    if len(frames) < 2:
        print("[error] the clip has fewer than two frames", file=sys.stderr)
        return 1
    repeats = find_repeats(frames, args.threshold)
    print(f"{len(repeats)} of {len(frames) - 1} frames repeat the frame before them "
          f"({100.0 * len(repeats) / (len(frames) - 1):.1f}%)", file=sys.stderr)
    if args.list:
        print(" ".join(map(str, repeats)))
    elif repeats:
        print(keys_for(len(frames), repeats))
    else:
        print("[note] no repeats: nothing to fix", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
