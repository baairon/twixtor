#!/usr/bin/env python3
"""The After Effects side of the mock host: in_data/out_data, callbacks, suites, params, worlds and
the command sequence. Built on ae_host.Emu."""
from __future__ import annotations

import array
import math
import os
import re
import struct
import time
from fractions import Fraction

from ae_host import GuestFault, Layouts, Record, View, Emu

A_Err_NONE = 0
A_Err_GENERIC = 1
A_Err_MISSING_SUITE = 13
kSPSuiteNotFoundError = int.from_bytes(b"S!Fd", "big")


def fixed(v: float) -> int:
    """16.16 from a double, half away from zero like the rewrite's float_to_fixed (fx-core/src/fixed.rs:
    (v * 65536.0 +- 0.5) as i32, truncating). Which rule After Effects uses at a tie is not known: a stand-in."""
    return int(v * 65536.0 + (-0.5 if v < 0 else 0.5))


def q15(v: int) -> int:
    return (v * 32768 + 32767) // 65535


def u16_of_q15(v: int) -> int:
    return (min(v, 32768) * 65535 + 16384) // 32768


_LUT_TO_Q15 = None
_LUT_FROM_Q15 = None


def luts():
    global _LUT_TO_Q15, _LUT_FROM_Q15
    if _LUT_TO_Q15 is None:
        _LUT_TO_Q15 = array.array("H", (q15(v) for v in range(65536)))
        _LUT_FROM_Q15 = array.array("H", (u16_of_q15(v) for v in range(32769)))
    return _LUT_TO_Q15, _LUT_FROM_Q15


class TimeBase:
    """The rewrite's default time base: an integer fps F is 100 ticks a frame (time_scale F*100); a
    fraction N/D is N ticks a second and D a frame."""

    def __init__(self, fps: str):
        f = Fraction(fps) if "/" in fps else Fraction(fps)
        self.fps = f
        if f.denominator == 1:
            unit = int(os.environ.get("HARNESS_TIME_UNIT", "100"))     # ticks a frame for an integer rate; 100 is the rewrite's default
            self.time_scale, self.time_step = int(f) * unit, unit
        else:
            self.time_scale, self.time_step = f.numerator, f.denominator

    def frame_of(self, t: int, time_step: int, time_scale: int) -> int:
        """The footage frame holding the tick t of a clock `time_scale` ticks a second."""
        return math.floor(Fraction(t, time_scale) * self.fps)


# ---------------------------------------------------------------------------------------------
# frames: wire format <-> AE worlds

def wire_to_world(depth: int, w: int, h: int, wire: bytes) -> bytes:
    """A raw frame in the rewrite's wire format as the bytes of an AE world (ARGB, straight)."""
    n = w * h
    if depth == 8:
        out = bytearray(n * 4)
        out[0::4], out[1::4], out[2::4], out[3::4] = wire[3::4], wire[0::4], wire[1::4], wire[2::4]
        return bytes(out)
    if depth == 16:
        to_q, _ = luts()
        vals = array.array("H")
        vals.frombytes(wire)
        q = array.array("H", map(to_q.__getitem__, vals))
        out = array.array("H", bytes(n * 8))
        out[0::4], out[1::4], out[2::4], out[3::4] = q[3::4], q[0::4], q[1::4], q[2::4]
        return out.tobytes()
    planes = array.array("f")
    planes.frombytes(wire)          # planar G, B, R, A
    g, b, r, a = (planes[i * n:(i + 1) * n] for i in range(4))
    out = array.array("f", bytes(n * 16))
    out[0::4], out[1::4], out[2::4], out[3::4] = a, r, g, b
    return out.tobytes()


def world_to_wire(depth: int, w: int, h: int, world: bytes) -> bytes:
    n = w * h
    if depth == 8:
        out = bytearray(n * 4)
        out[0::4], out[1::4], out[2::4], out[3::4] = world[1::4], world[2::4], world[3::4], world[0::4]
        return bytes(out)
    if depth == 16:
        _, from_q = luts()
        vals = array.array("H")
        vals.frombytes(world)
        vals = array.array("H", (min(v, 32768) for v in vals))
        u = array.array("H", map(from_q.__getitem__, vals))
        out = array.array("H", bytes(n * 8))
        out[0::4], out[1::4], out[2::4], out[3::4] = u[1::4], u[2::4], u[3::4], u[0::4]
        return out.tobytes()
    px = array.array("f")
    px.frombytes(world)             # interleaved A, R, G, B
    out = array.array("f", bytes(n * 16))
    out[0:n], out[n:2 * n], out[2 * n:3 * n], out[3 * n:4 * n] = px[2::4], px[3::4], px[1::4], px[0::4]
    return out.tobytes()


BPP = {8: 4, 16: 8, 32: 16}


class World:
    def __init__(self, view: View, data: int, w: int, h: int, depth: int):
        self.view, self.data, self.w, self.h, self.depth = view, data, w, h, depth

    @property
    def nbytes(self) -> int:
        return self.w * self.h * BPP[self.depth]


class Mock(Emu):
    """The mock host."""

    def __init__(self, plugin, rootfs, rec: Record, lay: Layouts, depth: int = 8, width: int = 64, height: int = 48,
                 fps: str = "24", frames: int = 1):
        super().__init__(plugin, rootfs, rec, lay)
        self.configure(depth, width, height, fps, frames)
        self.reg_params: list[dict] = []        # registered by add_param, in order (index 1..)
        self.inputs: dict[str, list[bytes] | None] = {}   # layer key ("input" or a param id) -> AE world bytes per frame
        self.note: dict = {}
        self.cur_frame = 0
        self.cur_time = 0
        self.worlds: dict[int, World] = {}
        self.checkouts: dict[int, dict] = {}
        self.suites: dict[tuple[str, int], int] = {}
        self.suite_log: list[tuple[str, int]] = []
        self.param_values: dict[int, object] = {}    # reg index -> current python value (or None for default)
        self.layer_clips: dict[int, list[bytes] | None] = {}   # reg index of a layer param -> frames (AE world bytes)
        self.lay_view = {}
        self.entry = None
        self.out_flags = 0
        self.out_flags2 = 0
        # the plugin's flags are what GLOBAL_SETUP returned; PARAMS_SETUP and the later commands overwrite out_data
        # and zero them, so they are kept apart (out_flags and out_flags2 stay the last command's, as the cmd_done lines show)
        self.global_out_flags = 0
        self.global_out_flags2 = 0
        self.num_params = 1
        self.effect_ref = self.alloc(16)
        self.build_basic_suite()
        self.utils_addr = self.build_table("PF_UtilCallbacks", "utils")
        self._inter_cache = {}

    def configure(self, depth: int, width: int, height: int, fps: str, frames: int):
        """The sequence a case renders: size, depth, rate and length."""
        self.depth, self.width, self.height, self.nframes = depth, width, height, frames
        self.tb = TimeBase(fps)
        self.fps = fps

    # -- trampoline tables ---------------------------------------------------------------------

    def _handler(self, prefix: str, member: str, struct_name: str):
        """A host function: explicit `cb_<prefix>_<member>` when it exists, else a logging stub."""
        key = f"{prefix}_{member}".replace(".", "_")
        h = getattr(self, f"cb_{key}", None)
        return h

    def fn_tramp(self, full: str, handler, fnsig: dict) -> int:
        sig = "".join(k if k != "S" else "p" for k in fnsig["args"])
        ret = fnsig["ret"]
        implemented = handler is not None

        def run(*args):
            self.note = {}
            t0 = time.perf_counter()
            if implemented:
                r = handler(*args)
            else:
                r = None
                self.note["unimplemented"] = True
                if "unimpl:" + full not in self.unimplemented:
                    self.unimplemented.add("unimpl:" + full)
                r = A_Err_GENERIC if ret in "iu" else 0
            self.rec.log("cb", fn=full, args=[hex(a) if isinstance(a, int) and k in "pqQ" else a for a, k in zip(args, sig)],
                         ret=r if r is None or not isinstance(r, int) else r, **self.note)
            return r

        return self.tramp(full, run, sig, ret if ret in "diufq" else ("i" if ret != "v" else "v"))

    def build_table(self, struct_name: str, prefix: str, only_prefix: str | None = None, addr: int | None = None) -> int:
        lay = self.lay.L[struct_name]
        if addr is None:
            addr = self.alloc(lay["size"])
            self.wr(addr, b"\0" * lay["size"])
        for m in lay["members"]:
            fs = m.get("fn")
            if not fs or (only_prefix and not m["name"].startswith(only_prefix)):
                continue
            full = f"{prefix}.{m['name']}"
            real = self.crt_for(full, m["name"], fs)
            if real:
                self.w64(addr + m["offset"], real)
                continue
            h = self._handler(prefix, m["name"], struct_name)
            self.w64(addr + m["offset"], self.fn_tramp(full, h, fs))
        return addr

    CRT = {"acos", "asin", "atan", "atan2", "ceil", "cos", "exp", "fabs", "floor", "fmod", "hypot", "log", "log10",
           "pow", "sin", "sqrt", "tan", "strcpy"}

    def crt_for(self, full: str, name: str, fs: dict):
        """The host's math callbacks run the C runtime inside the emulator (ucrtbase.dll) where it
        exports the function; a code hook only records the call."""
        if not full.startswith("utils.ansi.") or name.split(".")[-1] not in self.CRT:
            return None
        fname = name.split(".")[-1]
        tab = self.ql.loader.import_address_table.get("ucrtbase.dll", {})
        addr = tab.get(fname.encode())
        if not addr:
            return None
        sig = "".join(fs["args"])

        def log(ql, address, size):
            args = [self.argv(i, k) for i, k in enumerate(sig)]
            self.last_cb = full
            self.rec.log("cb", fn=full, args=args, ret="(ucrtbase.dll)", runtime="ucrtbase.dll")

        self.ql.hook_code(log, begin=addr, end=addr)
        return addr

    # -- SPBasicSuite and suites ---------------------------------------------------------------

    def build_basic_suite(self):
        lay = self.lay.L["SPBasicSuite"]
        self.basic_addr = self.alloc(lay["size"])
        self.wr(self.basic_addr, b"\0" * lay["size"])
        for m in lay["members"]:
            fs = m.get("fn")
            if not fs:
                continue
            h = getattr(self, f"sp_{m['name']}", None)
            self.w64(self.basic_addr + m["offset"], self.fn_tramp(f"SPBasic.{m['name']}", h, fs))

    def sp_AcquireSuite(self, name_p, version, out_pp):
        name = self.cstr(name_p)
        st = self.lay.suite_struct(name, version)
        self.note.update(suite=name, version=version, struct=st)
        if st is None or name in self.REFUSED_SUITES:
            self.w64(out_pp, 0)
            self.note["acquired"] = False
            return kSPSuiteNotFoundError
        key = (st, version)
        if key not in self.suites:
            self.suites[key] = self.build_table(st, st)
        self.w64(out_pp, self.suites[key])
        self.note["acquired"] = True
        return 0

    # suites answered with "not found": the mock offers no GPU
    REFUSED_SUITES = {"PF GPU Device Suite", "PF GPU Suite", "Premiere GPU Device Suite"}

    def sp_ReleaseSuite(self, name_p, version):
        self.note.update(suite=self.cstr(name_p), version=version)
        return 0

    def sp_AllocateBlock(self, size, out_pp):
        a = self.alloc(size)
        self.w64(out_pp, a)
        return 0

    def sp_FreeBlock(self, block):
        return 0

    def sp_ReallocateBlock(self, block, new_size, out_pp):
        a = self.alloc(new_size)
        self.w64(out_pp, a)
        return 0

    def sp_IsEqual(self, a_p, b_p):
        return int(self.cstr(a_p) == self.cstr(b_p))

    # -- worlds ----------------------------------------------------------------------------------

    def new_world(self, w: int, h: int, depth: int, data: bytes | None = None) -> World:
        v = self.new("PF_LayerDef")
        buf = self.alloc(max(w * h * BPP[depth], 16), 64)
        if data is not None:
            self.wr(buf, data)
        return self._init_world(v, buf, w, h, depth)

    def _init_world(self, v: View, buf: int, w: int, h: int, depth: int) -> World:
        v.set("data", buf)
        v.set("rowbytes", w * BPP[depth])
        v.set("width", w)
        v.set("height", h)
        v.set("world_flags", self.lay.e("PF_WorldFlag_WRITEABLE") | (self.lay.e("PF_WorldFlag_DEEP") if depth == 16 else 0))
        v.set("extent_hint.left", 0)
        v.set("extent_hint.top", 0)
        v.set("extent_hint.right", w)
        v.set("extent_hint.bottom", h)
        v.set("pix_aspect_ratio.num", 1)
        v.set("pix_aspect_ratio.den", 1)
        wd = World(v, buf, w, h, depth)
        self.worlds[v.a] = wd
        return wd

    def read_world(self, wd: World) -> bytes:
        return self.rd(wd.data, wd.nbytes)

    # -- handlers: inter ---------------------------------------------------------------------

    def cb_inter_add_param(self, effect_ref, index, def_p):
        d = self.decode_param(def_p)
        d["reg_index"] = len(self.reg_params) + 1
        d["guest"] = self.alloc(self.lay.size("PF_ParamDef"))
        self.wr(d["guest"], self.rd(def_p, self.lay.size("PF_ParamDef")))
        self.reg_params.append(d)
        self.note.update(param=d["name"], param_type=d["type_name"])
        return 0

    def cb_inter_abort(self, effect_ref):
        return 0

    def cb_inter_progress(self, effect_ref, current, total):
        return 0

    def cb_inter_register_ui(self, effect_ref, cust):
        return 0

    def cb_inter_checkin_param(self, effect_ref, def_p):
        return 0

    def cb_inter_checkout_param(self, effect_ref, index, what_time, time_step, time_scale, def_p):
        frame = self.tb.frame_of(what_time, time_step, time_scale)
        t = what_time
        if time_scale and time_scale != self.tb.time_scale:
            # the plugin asked with a clock other than the composition's: read the value at the same instant
            from fractions import Fraction
            exact = Fraction(what_time * self.tb.time_scale, time_scale)
            t = int(exact + Fraction(1, 2)) if exact >= 0 else -int(-exact + Fraction(1, 2))
            self.note.update(clock_differs=True, time_in_comp_ticks=t)
        self.note.update(param_index=index, time=what_time, time_step=time_step, time_scale=time_scale,
                         footage_frame=frame, out_frame=self.cur_frame)
        self.fill_param(def_p, index, t, frame)
        return 0

    # -- params ----------------------------------------------------------------------------------

    PARAM_TYPES = None

    def type_names(self) -> dict:
        if self.PARAM_TYPES is None:
            Mock.PARAM_TYPES = {v: k[len("PF_Param_"):] for k, v in self.lay.enums.items() if k.startswith("PF_Param_")}
        return self.PARAM_TYPES

    def decode_param(self, p: int) -> dict:
        v = self.view(p, "PF_ParamDef")
        ty = v.get("param_type")
        tn = self.type_names().get(ty, str(ty))
        d = {
            "name": self.rd(v.addr("name_do_not_use_directly"), 32).split(bytes(1))[0].decode("latin-1"),      # the member is declared A_char[PF_MAX_EFFECT_PARAM_NAME_LEN + 1]: 32 bytes, flags follows at 48
            "name_bytes": self.rd(v.addr("name_do_not_use_directly"), 32).split(bytes(1))[0].hex(),
            "type": ty,
            "type_name": tn,
            "flags": v.get("flags"),
            "ui_flags": v.get("ui_flags"),
            "id": v.get("uu.id"),
        }
        g = v.get
        if tn == "SLIDER":
            d.update(value=g("u.sd.value"), min=g("u.sd.valid_min"), max=g("u.sd.valid_max"),
                     slider_min=g("u.sd.slider_min"), slider_max=g("u.sd.slider_max"), default=g("u.sd.dephault"))
        elif tn == "FIX_SLIDER":
            f = lambda k: g(k) / 65536.0
            d.update(value=f("u.fd.value"), min=f("u.fd.valid_min"), max=f("u.fd.valid_max"),
                     slider_min=f("u.fd.slider_min"), slider_max=f("u.fd.slider_max"), default=f("u.fd.dephault"),
                     precision=g("u.fd.precision"), display_flags=g("u.fd.display_flags"))
        elif tn == "FLOAT_SLIDER":
            d.update(value=g("u.fs_d.value"), min=g("u.fs_d.valid_min"), max=g("u.fs_d.valid_max"),
                     slider_min=g("u.fs_d.slider_min"), slider_max=g("u.fs_d.slider_max"), default=g("u.fs_d.dephault"),
                     precision=g("u.fs_d.precision"), display_flags=g("u.fs_d.display_flags"))
        elif tn == "ANGLE":
            d.update(value=g("u.ad.value") / 65536.0, default=g("u.ad.dephault") / 65536.0)
        elif tn == "CHECKBOX":
            np_ = g("u.bd.u.nameptr")
            d.update(value=g("u.bd.value"), default=g("u.bd.dephault"), label=self.cstr(np_) if np_ else "")
        elif tn == "COLOR":
            px = lambda k: tuple(self.rd(v.addr(k), 4))
            d.update(value=px("u.cd.value"), default=px("u.cd.dephault"))   # A, R, G, B
        elif tn == "POINT":
            d.update(value=(g("u.td.x_value") / 65536.0, g("u.td.y_value") / 65536.0),
                     default=(g("u.td.x_dephault") / 65536.0, g("u.td.y_dephault") / 65536.0),
                     restrict_bounds=g("u.td.restrict_bounds"))
        elif tn == "POPUP":
            np_ = g("u.pd.u.namesptr")
            names = self.cstr(np_) if np_ else ""
            d.update(value=g("u.pd.value"), default=g("u.pd.dephault"), num_choices=g("u.pd.num_choices"),
                     choices=names.split("|") if names else [])
        elif tn == "LAYER":
            d.update(default=g("u.ld.dephault"))
        elif tn == "POINT_3D":
            d.update(value=(g("u.point3d_d.x_value"), g("u.point3d_d.y_value"), g("u.point3d_d.z_value")))
        elif tn == "BUTTON":
            np_ = g("u.button_d.u.namesptr")
            d.update(label=self.cstr(np_) if np_ else "")
        return d

    def set_default_values(self, d: dict):
        """The param's value is its default until a case sets it."""
        self.param_values[d["reg_index"]] = None

    def fill_param(self, def_p: int, index: int, what_time: int, frame: int):
        """A checked-out param: the registered def with its value, or for a layer the world."""
        size = self.lay.size("PF_ParamDef")
        if index == 0:
            self.wr(def_p, b"\0" * size)
            v = self.view(def_p, "PF_ParamDef")
            v.set("param_type", self.lay.e("PF_Param_LAYER"))
            self.checkout_layer_into(v, "input", frame)
            return
        d = self.reg_params[index - 1]
        self.wr(def_p, self.rd(d["guest"], size))
        v = self.view(def_p, "PF_ParamDef")
        if d["type_name"] == "LAYER":
            self.checkout_layer_into(v, d["reg_index"], frame)
        else:
            self.apply_value(v, d, what_time)

    def checkout_layer_into(self, v: View, key, frame: int):
        """Fills the PF_LayerDef inside a ParamDef with the frame of a layer; no clip: data is NULL."""
        clip = self.inputs.get(key)
        base = v.addr("u.ld.reserved0")
        ld = self.view(base, "PF_LayerDef")
        if clip is None:
            ld.set("data", 0)
            ld.set("width", 0)
            ld.set("height", 0)
            self.note["layer"] = "none"
            return
        data = clip[frame] if 0 <= frame < len(clip) else bytes(self.width * self.height * BPP[self.depth])
        buf = self.alloc(len(data), 64)
        self.wr(buf, data)
        self._init_world(ld, buf, self.width, self.height, self.depth)
        self.note["layer"] = key if frame < len(clip) and frame >= 0 else f"{key} (outside the clip: transparent)"

    def apply_value(self, v: View, d: dict, t: int):
        """Writes the current value of the param at tick t into a ParamDef copy."""
        val = self.param_values.get(d["reg_index"])
        if val is None:
            return
        val = val(t) if callable(val) else val
        tn = d["type_name"]
        if tn == "SLIDER":
            v.set("u.sd.value", int(round(val)))
        elif tn == "FIX_SLIDER":
            v.set("u.fd.value", fixed(val))
        elif tn == "FLOAT_SLIDER":
            v.set("u.fs_d.value", float(val))
        elif tn == "ANGLE":
            v.set("u.ad.value", fixed(val))
        elif tn == "CHECKBOX":
            v.set("u.bd.value", 1 if val else 0)
        elif tn == "POPUP":
            v.set("u.pd.value", int(val))
        elif tn == "COLOR":
            r, g_, b = val
            self.wr(v.addr("u.cd.value"), bytes([255, r, g_, b]))
        elif tn == "POINT":
            v.set("u.td.x_value", fixed(val[0]))
            v.set("u.td.y_value", fixed(val[1]))

    # -- handlers: utils ----------------------------------------------------------------------

    def cb_utils_host_new_handle(self, size):
        h = self.alloc(8)
        data = self.alloc(max(size, 8), 64)
        self.w64(h, data)
        self.handle_sizes[h] = size
        self.note.update(size=size, handle=hex(h))
        return h

    @property
    def handle_sizes(self):
        if not hasattr(self, "_hs"):
            self._hs = {}
        return self._hs

    def cb_utils_host_lock_handle(self, h):
        return self.r64(h) if h else 0

    def cb_utils_host_unlock_handle(self, h):
        return 0

    def cb_utils_host_dispose_handle(self, h):
        return 0

    def cb_utils_host_get_handle_size(self, h):
        return self.handle_sizes.get(h, 0)

    def cb_utils_host_resize_handle(self, new_size, hp):
        h = self.r64(hp)
        old = self.r64(h)
        data = self.alloc(max(new_size, 8), 64)
        n = min(new_size, self.handle_sizes.get(h, 0))
        if old and n:
            self.wr(data, self.rd(old, n))
        self.w64(h, data)
        self.handle_sizes[h] = new_size
        return 0

    # PF AE App Suite: an interactive After Effects is not a render engine
    def cb_PFAppSuite4_PF_IsRenderEngine(self, out_p):
        if os.environ.get("HARNESS_ISRENDERENGINE_STUB"):
            return 1                  # the earlier behaviour (an unimplemented stub: error, nothing written), kept to reproduce it
        if out_p:
            self.wr(out_p, bytes(1))
        return 0

    # AEGP Utility Suite (numeric version 11, struct AEGP_UtilitySuite5). The host side of the AE host: ids are handed
    # out in order from 1, and the host is activated. Other members stay logged stubs returning A_Err_NONE.
    def cb_AEGP_UtilitySuite5_AEGP_RegisterWithAEGP(self, global_refcon, name_p, id_p):
        self._aegp_ids = getattr(self, "_aegp_ids", 0) + 1
        if id_p:
            self.w32(id_p, self._aegp_ids)
        self.note.update(plugin_id=self._aegp_ids)
        return 0

    def cb_AEGP_UtilitySuite5_AEGP_HostIsActivated(self, out_p):
        if out_p:
            self.wr(out_p, b"")
        return 0

    # AEGP PF Interface Suite 1: the layer an effect is applied to is one opaque handle per sequence
    def cb_AEGP_PFInterfaceSuite1_AEGP_GetEffectLayer(self, effect_ref, out_p):
        if not getattr(self, "_layer_h", 0):
            self._layer_h = self.alloc(64)
        if out_p:
            self.w64(out_p, self._layer_h)
        return 0

    # AEGP Layer Suite (numeric version 5, struct AEGP_LayerSuite1): the layer sits in one composition
    def cb_AEGP_LayerSuite1_AEGP_GetLayerParentComp(self, layer_h, out_p):
        if not getattr(self, "_comp_h", 0):
            self._comp_h = self.alloc(64)
        if out_p:
            self.w64(out_p, self._comp_h)
        return 0

    # AEGP Comp Suite (numeric version 4, struct AEGP_CompSuite1): the composition runs at the case's frame rate
    def _a_time(self, out_p, ticks):
        """An A_Time {A_long value, A_u_long scale} in the composition's clock."""
        if out_p:
            self.wr(out_p, struct.pack("<iI", ticks, self.tb.time_scale))
        return 0

    def cb_AEGP_CompSuite1_AEGP_GetCompWorkAreaStart(self, comp_h, out_p):
        return self._a_time(out_p, 0)

    def cb_AEGP_CompSuite1_AEGP_GetCompWorkAreaDuration(self, comp_h, out_p):
        return self._a_time(out_p, self.tb.time_step * self.nframes)

    def cb_AEGP_CompSuite1_AEGP_GetCompDownsampleFactor(self, comp_h, out_p):
        if out_p:
            self.wr(out_p, struct.pack("<ii", 1, 1))              # PF_Point: full resolution
        return 0

    def cb_AEGP_CompSuite1_AEGP_GetCompFramerate(self, comp_h, out_p):
        if out_p:
            self.wr(out_p, struct.pack("<d", self.tb.time_scale / self.tb.time_step))
        return 0

    # PF Path Query Suite 1: the layers of these cases carry no masks
    def cb_PF_PathQuerySuite1_PF_NumPaths(self, effect_ref, out_p):
        stub = os.environ.get("HARNESS_PATHQUERY_STUB")     # "1": always; a case id: while that case renders
        if stub and (stub == "1" or stub == getattr(self, "case_id", None)):
            return A_Err_GENERIC          # the earlier behaviour (an unimplemented stub), kept to reproduce it
        if out_p:
            self.w32(out_p, 0)
        return 0

    def cb_PF_PathQuerySuite1_PF_PathInfo(self, effect_ref, index, id_p):
        return 513                        # PF_Err_INVALID_INDEX: there is no path to index

    def cb_PF_PathQuerySuite1_PF_CheckoutPath(self, effect_ref, unique_id, what_time, time_step, time_scale, path_pp):
        if path_pp:
            self.w64(path_pp, 0)          # "can return NULL ptr if path doesn't exist"
        return 0

    def cb_PF_PathQuerySuite1_PF_CheckinPath(self, effect_ref, unique_id, changed, path_p):
        return 0

    # PF Handle Suite (numeric version 2, struct PF_HandleSuite1): the same six functions as the utils callbacks
    def cb_PF_HandleSuite1_host_new_handle(self, size):
        return self.cb_utils_host_new_handle(size)

    def cb_PF_HandleSuite1_host_lock_handle(self, h):
        return self.cb_utils_host_lock_handle(h)

    def cb_PF_HandleSuite1_host_unlock_handle(self, h):
        return self.cb_utils_host_unlock_handle(h)

    def cb_PF_HandleSuite1_host_dispose_handle(self, h):
        return self.cb_utils_host_dispose_handle(h)

    def cb_PF_HandleSuite1_host_get_handle_size(self, h):
        return self.cb_utils_host_get_handle_size(h)

    def cb_PF_HandleSuite1_host_resize_handle(self, new_size, hp):
        return self.cb_utils_host_resize_handle(new_size, hp)

    def cb_utils_new_world(self, effect_ref, w, h, flags, world_p):
        # flags are PF_NewWorldFlags (CLEAR_PIXELS = 1, DEEP_PIXELS = 2), not PF_WorldFlags
        deep = bool(flags & self.lay.e("PF_NewWorldFlag_DEEP_PIXELS"))
        depth = 16 if deep else (32 if self.depth == 32 else 8)
        v = self.view(world_p, "PF_LayerDef")
        buf = self.alloc(max(w * h * BPP[depth], 16), 64)
        self._init_world(v, buf, w, h, depth)
        self.note.update(width=w, height=h, depth=depth)
        return 0

    def cb_utils_dispose_world(self, effect_ref, world_p):
        return 0

    PIXEL_FORMAT_DEPTH = {"PF_PixelFormat_ARGB32": 8, "PF_PixelFormat_ARGB64": 16, "PF_PixelFormat_ARGB128": 32}

    def cb_PF_WorldSuite2_PF_NewWorld(self, effect_ref, w, h, clear, pixel_format, world_p):
        depth = {self.lay.e(k): d for k, d in self.PIXEL_FORMAT_DEPTH.items()}.get(pixel_format & 0xFFFFFFFF)
        if depth is None:
            return A_Err_GENERIC
        v = self.view(world_p, "PF_LayerDef")
        buf = self.alloc(max(w * h * BPP[depth], 16), 64)
        if clear:
            self.wr(buf, bytes(w * h * BPP[depth]))
        self._init_world(v, buf, w, h, depth)
        self.note.update(width=w, height=h, depth=depth)
        return 0

    def cb_PF_WorldSuite2_PF_DisposeWorld(self, effect_ref, world_p):
        return 0

    def cb_PF_WorldSuite2_PF_GetPixelFormat(self, world_p, fmt_p):
        wd = self.worlds.get(world_p)
        if wd is None or not fmt_p:
            return A_Err_GENERIC
        names = {8: "PF_PixelFormat_ARGB32", 16: "PF_PixelFormat_ARGB64", 32: "PF_PixelFormat_ARGB128"}
        self.w32(fmt_p, self.lay.e(names[wd.depth]))
        self.note.update(depth=wd.depth)
        return 0

    def _pixel_data(self, world_p, pixels0, out_p, depth):
        wd = self.worlds.get(world_p)
        if not out_p:
            return A_Err_GENERIC
        self.w64(out_p, (pixels0 or (wd.data if wd is not None else 0)) if wd is not None and wd.depth == depth else 0)
        return 0

    def cb_utils_get_pixel_data8(self, world_p, pixels0, out_pp):
        return self._pixel_data(world_p, pixels0, out_pp, 8)

    def cb_utils_get_pixel_data16(self, world_p, pixels0, out_pp):
        return self._pixel_data(world_p, pixels0, out_pp, 16)

    def cb_utils_get_platform_data(self, effect_ref, which, data_p):
        return A_Err_GENERIC

    # -- command sequence ------------------------------------------------------------------------

    def find_entry(self) -> int:
        exp = {(e["name"] or b"").decode(): a for a, e in self.ql.loader.export_symbols.items()}
        for n in ("EffectMain", "EntryPointFunc", "mainB"):
            if n in exp:
                self.entry_name = n
                return exp[n]
        if len(exp) == 1:
            (self.entry_name, addr), = exp.items()
            return addr
        raise GuestFault(f"no entry point among the loader's exports {sorted(exp)}")

    def boot(self):
        """DllMain with DLL_PROCESS_ATTACH, then the entry point address."""
        self.init_libraries()
        ep = self.ql.loader.entry_point
        r = self.call(ep, [self.base, 1, 0], "DllMain", timeout=120)
        self.rec.log("dllmain", ret=r)
        self.entry = self.find_entry()

    def fill_in_data(self, v: View, frame_time: int):
        L = self.lay
        v.set("utils", self.utils_addr)
        v.set("effect_ref", self.effect_ref)
        v.set("quality", L.e("PF_Quality_LO") if getattr(self, "quality_name", "high") == "draft" else L.e("PF_Quality_HI"))
        v.set("version.major", 13)
        v.set("version.minor", 28)
        v.set("serial_num", 0)
        v.set("appl_id", int.from_bytes(b"FXTC", "big"))
        v.set("num_params", self.num_params)
        v.set("what_cpu", 0)
        v.set("what_fpu", 0)
        v.set("current_time", frame_time)
        v.set("time_step", self.tb.time_step)
        v.set("total_time", self.tb.time_step * self.nframes)
        v.set("local_time_step", self.tb.time_step)
        v.set("time_scale", self.tb.time_scale)
        v.set("field", L.e("PF_Field_FRAME"))
        v.set("shutter_angle", 0)
        v.set("width", self.width)
        v.set("height", self.height)
        v.set("extent_hint.left", 0)
        v.set("extent_hint.top", 0)
        v.set("extent_hint.right", self.width)
        v.set("extent_hint.bottom", self.height)
        v.set("downsample_x.num", 1)
        v.set("downsample_x.den", 1)
        v.set("downsample_y.num", 1)
        v.set("downsample_y.den", 1)
        v.set("pixel_aspect_ratio.num", 1)
        v.set("pixel_aspect_ratio.den", 1)
        v.set("pica_basicP", self.basic_addr)
        for f in ("global_data", "sequence_data", "frame_data"):
            v.set(f, getattr(self, "_" + f, 0))
        # callbacks
        for m in L.L["PF_InData"]["members"]:
            fs = m.get("fn")
            if fs and m["name"].startswith("inter."):
                key = ("inter", m["name"])
                if key not in self._inter_cache:
                    h = self._handler("inter", m["name"][len("inter."):], "PF_InData")
                    self._inter_cache[key] = self.fn_tramp(f"inter.{m['name'][len('inter.'):]}", h, fs)
                self.w64(v.a + m["offset"], self._inter_cache[key])

    def dummy_params(self) -> int:
        """The params array of a command that has none yet: one zeroed ParamDef."""
        arr = self.alloc(8)
        self.w64(arr, self.new("PF_ParamDef").a)
        return arr

    def cmd(self, name: str, params_p: int = 0, output_p: int = 0, extra_p: int = 0, frame_time: int = 0,
            inter_frame: int | None = None) -> int:
        """One call of the entry point. Returns the PF_Err."""
        cmd = self.lay.e(f"PF_Cmd_{name}")
        ind = self.new("PF_InData")
        outd = self.new("PF_OutData")
        self.fill_in_data(ind, frame_time)
        for f in ("global_data", "sequence_data", "frame_data"):
            outd.set(f, getattr(self, "_" + f, 0))
        outd.set("num_params", self.num_params)
        if not params_p:
            params_p = self.dummy_params()
        if not output_p:
            output_p = self.new("PF_LayerDef").a
        if not extra_p:
            extra_p = self.alloc(0x400)
            self.wr(extra_p, b"\0" * 0x400)
        t0 = time.perf_counter()
        self.rec.log("cmd", cmd=name, time=frame_time)
        err = self.call(self.entry, [cmd, ind.a, outd.a, params_p, output_p, extra_p], name) & 0xFFFFFFFF
        err = err - (1 << 32) if err >> 31 else err
        for f in ("global_data", "sequence_data", "frame_data"):
            setattr(self, "_" + f, outd.get(f))
        self.out_flags, self.out_flags2 = outd.get("out_flags"), outd.get("out_flags2")
        if name == "GLOBAL_SETUP":
            self.global_out_flags, self.global_out_flags2 = self.out_flags, self.out_flags2
        self.out_num_params = outd.get("num_params")
        self.out_return_msg = f"return_msg of {len(bytes(outd.get('return_msg')).split(bytes(1))[0])} bytes"      # the text is plugin data and is not read
        self.rec.log("cmd_done", cmd=name, err=err, out_flags=hex(self.out_flags & 0xFFFFFFFF), out_flags2=hex(self.out_flags2 & 0xFFFFFFFF),
                     num_params=self.out_num_params, msg=self.out_return_msg, seconds=round(time.perf_counter() - t0, 3))
        return err

    # -- render path -------------------------------------------------------------------------------

    def params_array(self, frame_time: int, frame: int) -> int:
        """The params[] a classic render gets: the input layer at this frame, then every registered param
        with its value at this time."""
        n = 1 + len(self.reg_params)
        arr = self.alloc(8 * n)
        d0 = self.new("PF_ParamDef")
        self.fill_param(d0.a, 0, frame_time, frame)
        self.w64(arr, d0.a)
        size = self.lay.size("PF_ParamDef")
        for i, d in enumerate(self.reg_params):
            copy = self.alloc(size)
            self.fill_param(copy, i + 1, frame_time, frame)
            self.w64(arr + 8 * (i + 1), copy)
        return arr

    def is_smart(self) -> bool:
        return bool(self.global_out_flags2 & self.lay.e("PF_OutFlag2_SUPPORTS_SMART_RENDER"))

    def output_world(self) -> World:
        w = self.new_world(self.width, self.height, self.depth)
        self.wr(w.data, bytes(w.nbytes))
        return w

    def render_classic(self, n: int, t: int) -> bytes:
        out = self.output_world()
        params = self.params_array(t, n)
        self.cur_world = out
        for cmd in ("FRAME_SETUP", "RENDER", "FRAME_SETDOWN"):
            err = self.cmd(cmd, params, out.view.a, 0, t)
            if err:
                raise PluginError(cmd, err, self.out_return_msg)
        return self.read_world(out)

    def render_smart(self, n: int, t: int) -> bytes:
        out = self.output_world()
        self.cur_world = out
        self.checkouts = {}
        if not hasattr(self, "pre_cb"):
            self.pre_cb = self.build_table("PF_PreRenderCallbacks", "prerender_cb")
            self.smart_cb = self.build_table("PF_SmartRenderCallbacks", "smart_cb")
        e = self.lay.e
        request = (("output_request.rect.left", 0), ("output_request.rect.top", 0), ("output_request.rect.right", self.width),
                   ("output_request.rect.bottom", self.height), ("output_request.channel_mask", e("PF_ChannelMask_ARGB")),
                   ("output_request.preserve_rgb_of_zero_alpha", 1), ("bitdepth", self.depth),
                   ("what_gpu", e("PF_GPU_Framework_NONE")), ("device_index", 0))
        pin = self.new("PF_PreRenderInput")
        pout = self.new("PF_PreRenderOutput")
        for f, v in request:
            pin.set(f, v)
        pre = self.new("PF_PreRenderExtra")
        pre.set("input", pin.a)
        pre.set("output", pout.a)
        pre.set("cb", self.pre_cb)
        err = self.cmd("SMART_PRE_RENDER", 0, 0, pre.a, t)
        if err:
            raise PluginError("SMART_PRE_RENDER", err, self.out_return_msg)
        keys = ("result_rect.left", "result_rect.top", "result_rect.right", "result_rect.bottom", "solid", "flags")
        self.rec.log("pre_render_output", **{k: pout.get(k) for k in keys})
        sin = self.new("PF_SmartRenderInput")
        for f, v in request:
            sin.set(f, v)
        sin.set("pre_render_data", pout.get("pre_render_data"))
        sm = self.new("PF_SmartRenderExtra")
        sm.set("input", sin.a)
        sm.set("cb", self.smart_cb)
        err = self.cmd("SMART_RENDER", 0, 0, sm.a, t)
        if err:
            raise PluginError("SMART_RENDER", err, self.out_return_msg)
        # the plugin frees its pre render data through the function it set
        dele = pout.get("delete_pre_render_data_func")
        if dele and pout.get("pre_render_data"):
            try:
                self.call(dele, [pout.get("pre_render_data")], "delete_pre_render_data")
            except GuestFault as ex:
                self.rec.log("delete_pre_render_data_failed", error=str(ex))
        return self.read_world(out)

    def cb_prerender_cb_checkout_layer(self, effect_ref, index, checkout_id, req_p, what_time, time_step, time_scale, result_p):
        frame = self.tb.frame_of(what_time, time_step, time_scale)
        key = "input" if index == 0 else self.reg_params[index - 1]["reg_index"]
        clip = self.inputs.get(key)
        inside = clip is not None and 0 <= frame < len(clip)
        self.checkouts[checkout_id] = {"key": key, "frame": frame, "time": what_time}
        r = self.view(result_p, "PF_CheckoutResult")
        w, h = (self.width, self.height) if inside else (0, 0)
        # the layer keeps its size at every time: outside the clip result_rect is empty (nothing to draw) and max_result_rect
        # stays the layer's (AE_Effect.h: result_rect "can be empty", max_result_rect "the maximum size the output could possibly be")
        mw, mh = (self.width, self.height) if clip is not None else (0, 0)
        for f, v in (("result_rect.left", 0), ("result_rect.top", 0), ("result_rect.right", w), ("result_rect.bottom", h),
                     ("max_result_rect.left", 0), ("max_result_rect.top", 0), ("max_result_rect.right", mw),
                     ("max_result_rect.bottom", mh), ("par.num", 1), ("par.den", 1), ("solid", 0),
                     ("ref_width", self.width), ("ref_height", self.height)):
            r.set(f, v)
        self.note.update(param_index=index, checkout_id=checkout_id, time=what_time, time_step=time_step,
                         time_scale=time_scale, footage_frame=frame, inside=inside, out_frame=self.cur_frame)
        return 0

    def cb_prerender_cb_GuidMixInPtr(self, effect_ref, size, buf):
        self.note.update(size=size)
        return 0

    def cb_smart_cb_checkout_layer_pixels(self, effect_ref, checkout_id, pixels_pp):
        c = self.checkouts.get(checkout_id)
        if c is None:
            return A_Err_GENERIC
        clip = self.inputs.get(c["key"])
        if clip is not None and 0 <= c["frame"] < len(clip):
            w = self.new_world(self.width, self.height, self.depth, clip[c["frame"]])
        else:
            w = self.new_world(self.width, self.height, self.depth, bytes(self.width * self.height * BPP[self.depth]))
        self.w64(pixels_pp, w.view.a)
        self.note.update(checkout_id=checkout_id, footage_frame=c["frame"])
        return 0

    def cb_smart_cb_checkin_layer_pixels(self, effect_ref, checkout_id):
        self.note.update(checkout_id=checkout_id)
        return 0

    def cb_smart_cb_checkout_output(self, effect_ref, output_pp):
        self.w64(output_pp, self.cur_world.view.a)
        return 0

    # -- a sequence --------------------------------------------------------------------------------

    def setup_plugin(self):
        """GLOBAL_SETUP and PARAMS_SETUP, once per process."""
        self.num_params = 1
        err = self.cmd("GLOBAL_SETUP")
        if err:
            raise PluginError("GLOBAL_SETUP", err, self.out_return_msg)
        self.reg_params = []
        err = self.cmd("PARAMS_SETUP")
        if err:
            raise PluginError("PARAMS_SETUP", err, self.out_return_msg)
        self.num_params = 1 + len(self.reg_params)
        for d in self.reg_params:
            self.param_values[d["reg_index"]] = None

    def sequence(self):
        """SEQUENCE_SETUP, the frames, SEQUENCE_SETDOWN; yields the wire bytes of each output frame."""
        self.num_params = 1 + len(self.reg_params)
        params = self.params_array(0, 0)
        err = self.cmd("SEQUENCE_SETUP", params, 0, 0, 0)
        if err:
            raise PluginError("SEQUENCE_SETUP", err, self.out_return_msg)
        smart = self.is_smart()
        self.rec.log("render_path", smart=smart, out_flags=hex(self.global_out_flags & 0xFFFFFFFF), out_flags2=hex(self.global_out_flags2 & 0xFFFFFFFF))
        try:
            for n in range(self.nframes):
                self.cur_frame = n
                t = n * self.tb.time_step
                self.cur_time = t
                self.rec.log("frame", out_frame=n, time=t, time_step=self.tb.time_step, time_scale=self.tb.time_scale)
                world = self.render_smart(n, t) if smart else self.render_classic(n, t)
                yield world_to_wire(self.depth, self.width, self.height, world)
        finally:
            try:
                self.cmd("SEQUENCE_SETDOWN", self.params_array(0, 0), 0, 0, 0)
            except (GuestFault, PluginError) as ex:
                self.rec.log("sequence_setdown_failed", error=str(ex))


class PluginError(Exception):
    def __init__(self, cmd: str, err: int, msg: str = ""):
        super().__init__(f"{cmd} returned PF_Err {err}" + (f" ({msg})" if msg else ""))
        self.cmd, self.err = cmd, err
