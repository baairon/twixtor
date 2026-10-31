#!/usr/bin/env python3
"""Param values of a case as the mock host gives them to the plugin: constants, or keyframes
`V@F;V@F;...` interpolated by the host. Keyframe interpolation is computed by After Effects, not by
the plugin, so this module is a stand-in; it follows the shapes the rewrite documents (README,
Parameter values): straight between plain keys, hold keys, ease keys with the cubic shapes, and
nothing else. Bezier handles (`iS/I`, `oS/I`) are refused."""
from __future__ import annotations

import math
import re

KEY = re.compile(r"^(?P<v>[^@]*)@(?P<f>-?[0-9]+(?:\.[0-9]+)?)(?P<s>.*)$")



def half_away(x: float) -> int:
    """Round half away from zero, the rule of the rewrite's f64::round (fx-core/src/time.rs). Which rule real After
    Effects uses at an exact tie is not known: this is a stand-in, the same on both sides."""
    a = abs(x)
    t = math.floor(a)
    if a - t >= 0.5:
        t += 1
    return int(math.copysign(t, x))


class Unsupported(Exception):
    pass


def parse_scalar(kind: str, text: str):
    text = text.strip()
    if kind in ("float", "angle"):
        return float(text)
    if kind in ("int", "popup"):
        return int(float(text))
    if kind == "checkbox":
        return 1 if text.lower() in ("1", "true", "on") else 0
    if kind == "color":
        r, g, b = (int(x) for x in text.split(","))
        return (r, g, b)
    if kind == "point":
        x, y = (float(x) for x in text.split(","))
        return (x, y)
    raise Unsupported(f"param kind {kind}")


def _lerp(a, b, u):
    if isinstance(a, tuple):
        return tuple(x + (y - x) * u for x, y in zip(a, b))
    return a + (b - a) * u


def make_value(kind: str, text: str, time_step: int):
    """A constant, or a function of the tick."""
    if "@" not in text:
        return parse_scalar(kind, text)
    keys = []
    for part in text.split(";"):
        m = KEY.match(part)
        if not m:
            raise Unsupported(f"key `{part}`")
        suffix = m.group("s")
        if re.search(r"[io]-?[0-9]", suffix):
            raise Unsupported(f"Bezier handles in `{part}`")
        if suffix not in ("", "h", "e"):
            raise Unsupported(f"key suffix `{suffix}`")
        keys.append((half_away(float(m.group("f")) * time_step), parse_scalar(kind, m.group("v")), suffix))
    keys.sort(key=lambda k: k[0])
    if len({k[0] for k in keys}) != len(keys):
        raise Unsupported("two keys on one tick")
    interpolates = kind in ("float", "angle", "point")

    def at(t: int):
        if t <= keys[0][0]:
            return keys[0][1]
        if t >= keys[-1][0]:
            return keys[-1][1]
        for (ta, va, sa), (tb, vb, sb) in zip(keys, keys[1:]):
            if ta <= t < tb:
                if sa == "h" or not interpolates:
                    return va
                u = (t - ta) / (tb - ta)
                ea, eb = sa == "e", sb == "e"
                if ea and eb:
                    u = u * u * (3 - 2 * u)
                elif ea:
                    u = u * u * (2 - u)
                elif eb:
                    u = u * (1 + u * (1 - u))
                return _lerp(va, vb, u)
        return keys[-1][1]

    return at
