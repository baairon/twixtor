#!/usr/bin/env python3
"""Mutant table of the retime-time module, and the patcher that applies one to a scratch copy.

python3 mutants.py list
python3 mutants.py apply ID ROOT      # patch ROOT/plugins/retime-engine/src/<file>; every `old` must occur once
python3 mutants.py restore ROOT SRC   # copy the three engine files back from the pristine copy SRC
python3 mutants.py json               # the table as JSON

Only ever run against a scratch copy on the VPS (mut-src), never against the repository. A mutant
here is a deliberate defect in code of the module's scope (timeline.rs, planner.rs, blend.rs).
`expect` records what was predicted before running: "kill" or "equivalent" (a mutant that should
not change any output; it checks that the grid does not report phantom kills).
"""
import json
import os
import shutil
import sys

ENG = "plugins/retime-engine/src"
FILES = ["timeline.rs", "planner.rs", "blend.rs"]

M = []


def mut(mid, file, edits, what, expect="kill"):
    M.append({"id": mid, "file": file, "edits": edits, "what": what, "expect": expect})


mut("m01_u8_round_dropped", "blend.rs",
    [("row[i] = cvtt((bf - af) * w + af + 0.5f32) as u8;", "row[i] = cvtt((bf - af) * w + af + 0.0f32) as u8;")],
    "8 bpc blend truncates instead of rounding (+0.5 dropped)")
mut("m02_u16_operands_swapped", "blend.rs",
    [("row[i] = cvtt((bf - af) * w + af + 0.5f32) as u16;", "row[i] = cvtt((af - bf) * w + bf + 0.5f32) as u16;")],
    "16 bpc blend with the operands swapped (weight applies to the other frame)")
mut("m03_f32_blend_in_f64", "blend.rs",
    [("row[i] = (br[i] - ar[i]) * w + ar[i] + 0.0f32;",
      "row[i] = ((br[i] as f64 - ar[i] as f64) * (w as f64) + ar[i] as f64) as f32 + 0.0f32;")],
    "32 bpc blend sum computed in f64 and rounded once")
mut("m04_blend_snap_wider", "blend.rs", [("const SNAP: f64 = 0.005;", "const SNAP: f64 = 0.01;")],
    "blend snap-to-copy tolerance 0.005 -> 0.01")
mut("m05_extrapolation_weight", "blend.rs",
    [("lerp(out, b, (1.0f64 - f64::from(w)) as f32);", "lerp(out, b, w);")],
    "extrapolation branch passes w instead of 1 - w to lerp (swapped weight)")
mut("m06_lerp_clamp_dropped", "blend.rs",
    [("let (r, g, b) = (clamp(r), clamp(g), clamp(b));", "let (r, g, b) = (r, g, b);\n    let _ = clamp(0.0);")],
    "integer lerp no longer clamps colours to 0..=alpha")
mut("m07_weighted_weight_swapped", "blend.rs",
    [("let t = 1.0f32 - wa;\n    match (&a.samples, &b.samples) {", "let t = wa;\n    match (&a.samples, &b.samples) {")],
    "Weighted Blend uses wa instead of 1 - wa as the blend weight")
mut("m08_f32_plus_zero_dropped", "blend.rs",
    [("row[i] = (br[i] - ar[i]) * w + ar[i] + 0.0f32;", "row[i] = (br[i] - ar[i]) * w + ar[i];")],
    "32 bpc blend no longer adds 0.0 (turns -0 into +0)", expect="maybe equivalent")
mut("m09_exact_tolerance_wider", "timeline.rs",
    [("pub const EXACT_TOLERANCE: f64 = 0.005;", "pub const EXACT_TOLERANCE: f64 = 0.01;")],
    "exact-frame tolerance 0.005 -> 0.01 (the exact flag only gates the vector path, so none expected at vectors=1)",
    expect="equivalent in scope")
mut("m10_nearest_tie_to_b", "timeline.rs", [("self.wa >= 0.5", "self.wa > 0.5")],
    "Nearest at exactly half way takes the later frame")
mut("m11_trunc_vs_floor", "timeline.rs",
    [("let a = cvtt_i32_f32(s0);\n    let frac: f32 = s0 - a as f32;", "let a = s0.floor() as i32;\n    let frac: f32 = s0 - a as f32;")],
    "frame choice floors the source position instead of truncating (differs below 0)")
mut("m12_b_frame_off_by_one", "timeline.rs", [("b: a.wrapping_add(1),", "b: a.wrapping_add(2),")],
    "second source frame is a + 2")
mut("m13_k_bias", "timeline.rs", [("pub const K: f64 = 0.1;", "pub const K: f64 = 0.05;")],
    "tick bias and rate-equality tolerance K 0.1 -> 0.05")
mut("m14_snap_tolerance", "timeline.rs", [("pub const SNAP_TOLERANCE: f64 = 2e-5;", "pub const SNAP_TOLERANCE: f64 = 4e-5;")],
    "typed frame rate snapping tolerance 2e-5 -> 4e-5")
mut("m15_speed_sum_in_f64", "timeline.rs",
    [("let mut out = 0.0f32;", "let mut out = 0.0f32;\n                let mut acc64: f64 = 0.0;"),
     ("out = (sp / 2.0) + out;", "acc64 += (sp as f64) / 2.0;\n                        out = acc64 as f32;")],
    "Speed half-frame sum accumulated in f64 instead of f32")
mut("m16_marker_cut_one_way", "timeline.rs", [("if ma != mb {", "if ma > mb {")],
    "a label change only cuts when the label falls (ma > mb)")
mut("m17_marker_time_unrounded", "timeline.rs",
    [("let time = cvtt_i32(((d * self.rates.dur) as f64) + 0.5);", "let time = cvtt_i32(((d * self.rates.dur) as f64) + 0.0);")],
    "marker sample time loses its +0.5 rounding")
mut("m18_frame_time_round_dropped", "timeline.rs",
    [("cvtt_i32((d as f64) * (dur as f64) + 0.5)", "cvtt_i32((d as f64) * (dur as f64) + 0.0)")],
    "frame time for near-equal rates loses its +0.5 rounding")
mut("m19_fields_not_doubled", "timeline.rs", [("in_fps *= 2.0;", "in_fps *= 1.0;")],
    "input fields no longer double the source rate")
mut("m20_dur_in_f32", "timeline.rs",
    [("let dur = ((local_time_step as f64 * out_fps) / in_fps) as f32;", "let dur = (local_time_step as f32 * out_fps as f32) / in_fps as f32;")],
    "source frame duration computed in f32 instead of f64 (a search over input rates 0.1 to 1000 in steps of 0.01 against 13 output rates and 14 frames found no frame time that moves)",
    expect="equivalent in practice")
mut("m21_framenum_ratio_in_f32", "timeline.rs",
    [("Ok(((self.rates.in_fps / self.rates.out_fps) * n as f64) as f32)", "Ok((self.rates.in_fps / self.rates.out_fps) as f32 * n)")],
    "Frame Num rate ratio applied in f32 instead of f64")
mut("m22_speed_ratio_in_f32", "timeline.rs",
    [("Ok(((self.rates.in_fps / self.rates.out_fps) * out as f64) as f32)", "Ok((self.rates.in_fps / self.rates.out_fps) as f32 * out)")],
    "Speed rate ratio applied in f32 instead of f64")
mut("m23_planner_second_frame", "planner.rs", [("let g = f.wrapping_add(1);", "let g = f.wrapping_add(2);")],
    "planner asks the host for frames f and f + 2 (changes the call record only; the fast path reads select's frames)",
    expect="call record only")
mut("m24_blend_weight_f32", "timeline.rs", [("(1.0f64 - self.wa as f64) as f32", "1.0f32 - self.wa")],
    "blend weight 1 - wa in f32 instead of f64 then narrowed (should be identical)", expect="equivalent")
mut("m25_dur_operand_order", "timeline.rs",
    [("let dur = ((local_time_step as f64 * out_fps) / in_fps) as f32;", "let dur = (local_time_step as f64 * (out_fps / in_fps)) as f32;")],
    "source frame duration computed as lts * (out / in) (no input found where the f32 result differs)",
    expect="equivalent in practice")


def apply(mid, root):
    m = next(x for x in M if x["id"] == mid)
    path = os.path.join(root, ENG, m["file"])
    text = open(path).read()
    for old, new in m["edits"]:
        if text.count(old) != 1:
            raise SystemExit("%s: pattern occurs %d times: %r" % (mid, text.count(old), old))
        text = text.replace(old, new)
    open(path, "w").write(text)


def restore(root, src):
    for f in FILES:
        shutil.copyfile(os.path.join(src, ENG, f), os.path.join(root, ENG, f))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        for m in M:
            print("%s\t%s\t%s\t%s" % (m["id"], m["file"], m["expect"], m["what"]))
    elif cmd == "json":
        print(json.dumps(M, indent=1))
    elif cmd == "apply":
        apply(sys.argv[2], sys.argv[3])
    elif cmd == "restore":
        restore(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
