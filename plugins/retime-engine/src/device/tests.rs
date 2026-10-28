//! Device-free tests of the backend foundation: the kernel source lint, the prelude and kernel
//! transcriptions against the CPU code in shuffled work item orders, the simulator itself, the
//! corpus, the probe requirements over synthetic devices and the loader's error paths.

use super::cl::{self, ClError, Range};
use super::kernels::{self, prelude, starter, BUILD_OPTIONS, SOURCES};
use super::probe::{self, DeviceInfo};
use super::selfcheck::{self, Bits};
use super::sim::{self, Global};
use std::path::{Path, PathBuf};

// ---- source lint ------------------------------------------------------------------------------

/// `source` without its `/* */` and `//` comments (kernel sources have no string literals).
fn strip_comments(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let b = source.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
            let end = source[i + 2..].find("*/").map(|e| i + 2 + e + 2).unwrap_or(b.len());
            out.push(' ');
            i = end;
        } else if b[i] == b'/' && b.get(i + 1) == Some(&b'/') {
            let end = source[i..].find('\n').map(|e| i + e).unwrap_or(b.len());
            i = end;
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// The identifiers of `code`.
fn identifiers(code: &str) -> Vec<&str> {
    let b = code.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() {
            // a number: skip the whole preprocessing number
            while i < b.len() && (is_ident(b[i]) || b[i] == b'.' || ((b[i] == b'+' || b[i] == b'-') && matches!(b[i - 1], b'e' | b'E' | b'p' | b'P'))) {
                i += 1;
            }
        } else if is_ident(b[i]) {
            let start = i;
            while i < b.len() && is_ident(b[i]) {
                i += 1;
            }
            out.push(&code[start..i]);
        } else {
            i += 1;
        }
    }
    out
}

/// The numeric literals of `code` (C preprocessing numbers, including `.5`).
fn numbers(code: &str) -> Vec<&str> {
    let b = code.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let starts = b[i].is_ascii_digit() || (b[i] == b'.' && b.get(i + 1).is_some_and(u8::is_ascii_digit));
        let after_ident = i > 0 && is_ident(b[i - 1]);
        if starts && !after_ident {
            let start = i;
            i += 1;
            while i < b.len() && (is_ident(b[i]) || b[i] == b'.' || ((b[i] == b'+' || b[i] == b'-') && matches!(b[i - 1], b'e' | b'E' | b'p' | b'P'))) {
                i += 1;
            }
            out.push(&code[start..i]);
        } else {
            i += 1;
        }
    }
    out
}

/// Whether a numeric literal is a decimal float (forbidden): not hexadecimal, and with a point,
/// a decimal exponent or a float suffix.
fn is_decimal_float(n: &str) -> bool {
    let lower = n.to_ascii_lowercase();
    if lower.starts_with("0x") {
        return false;
    }
    lower.contains('.') || lower.contains('e') || lower.ends_with('f') || lower.ends_with('h')
}

const FORBIDDEN: &[&str] = &[
    "mad", "mad24", "mad_hi", "mad_sat", "fma", "pow", "pown", "powr", "log", "log2", "log10", "log1p", "logb", "exp",
    "exp2", "exp10", "expm1", "fmin", "fmax", "mix",
];
const FORBIDDEN_PREFIXES: &[&str] = &["native_", "half_"];

/// Every lint finding of one source.
fn lint(source: &str) -> Vec<String> {
    let code = strip_comments(source);
    let mut out = Vec::new();
    for id in identifiers(&code) {
        if FORBIDDEN.contains(&id) || FORBIDDEN_PREFIXES.iter().any(|p| id.starts_with(p)) {
            out.push(format!("forbidden `{id}`"));
        }
    }
    for n in numbers(&code) {
        if is_decimal_float(n) {
            out.push(format!("decimal float literal `{n}`"));
        }
    }
    if code.contains("while") {
        out.push("a while loop (unbounded)".into());
    }
    // every for loop compares its counter against an integer literal
    let mut rest = code.as_str();
    while let Some(p) = rest.find("for (") {
        let head = &rest[p + 5..];
        let close = head.find(')').unwrap_or(head.len());
        let parts: Vec<&str> = head[..close].split(';').collect();
        let bounded = parts.len() == 3 && {
            let cond = parts[1].trim();
            let bound = cond.rsplit(['<', '>']).next().unwrap_or("").trim_start_matches('=').trim();
            !bound.is_empty() && bound.bytes().all(|c| c.is_ascii_digit())
        };
        if !bounded {
            out.push(format!("a loop without a constant bound: `for ({}`", &head[..close.min(60)]));
        }
        rest = &head[close.min(head.len())..];
    }
    if code.contains("FP_CONTRACT") && !code.starts_with("#pragma OPENCL FP_CONTRACT OFF") {
        out.push("FP_CONTRACT outside the prelude's first line".into());
    }
    out
}

#[test]
fn the_prelude_starts_with_fp_contract_off() {
    assert_eq!(prelude::PRELUDE.lines().next(), Some("#pragma OPENCL FP_CONTRACT OFF"));
    assert!(kernels::starter_program().starts_with("#pragma OPENCL FP_CONTRACT OFF\n"));
    assert!(kernels::program(&[]).starts_with("#pragma OPENCL FP_CONTRACT OFF\n"));
    assert_eq!(prelude::PRELUDE.matches("FP_CONTRACT").count(), 1);
    for (name, src) in SOURCES.iter().filter(|(n, _)| *n != "prelude::PRELUDE") {
        assert!(!src.contains("FP_CONTRACT"), "{name} must not touch FP_CONTRACT");
        assert!(!src.contains("#pragma"), "{name}: pragmas belong to the prelude");
    }
}

#[test]
fn every_kernel_source_passes_the_lint() {
    assert!(!SOURCES.is_empty());
    for (name, src) in SOURCES {
        let findings = lint(src);
        assert!(findings.is_empty(), "{name}: {findings:?}");
    }
    assert!(lint(&kernels::starter_program()).is_empty());
}

#[test]
fn the_lint_catches_what_it_forbids() {
    let bad = [
        ("float y = mad(a, b, c);", "forbidden `mad`"),
        ("float y = fma(a, b, c);", "forbidden `fma`"),
        ("float y = native_exp(a);", "forbidden `native_exp`"),
        ("float y = half_sqrt(a);", "forbidden `half_sqrt`"),
        ("float y = pow(a, b);", "forbidden `pow`"),
        ("float y = log2(a);", "forbidden `log2`"),
        ("float y = exp(a);", "forbidden `exp`"),
        ("float y = fmin(a, b);", "forbidden `fmin`"),
        ("float y = fmax(a, b);", "forbidden `fmax`"),
        ("float y = mix(a, b, t);", "forbidden `mix`"),
        ("float y = x * 0.5f;", "decimal float literal `0.5f`"),
        ("float y = x * .5;", "decimal float literal `.5`"),
        ("float y = x * 1e3f;", "decimal float literal `1e3f`"),
        ("float y = x * 2.0;", "decimal float literal `2.0`"),
        ("float y = x * 1f;", "decimal float literal `1f`"),
        ("while (x) { }", "a while loop (unbounded)"),
        ("for (int i = 0; i < n; i++) { }", "a loop without a constant bound: `for (int i = 0; i < n; i++`"),
        ("#pragma OPENCL FP_CONTRACT ON", "FP_CONTRACT outside the prelude's first line"),
    ];
    for (src, want) in bad {
        assert!(lint(src).iter().any(|f| f == want), "`{src}` gave {:?}", lint(src));
    }
    // what is allowed: hex floats, integers, identifiers containing a forbidden word, comments
    let good = "/* 0.5 mad pow */ // fma 1.0\nint expected = 0x1p-1f > 0 ? 128 : 255; float mixed_up = 0x1.8p+1f * 0x0p+0f; \
                for (int s = 0; s < 64; s++) { } for (int s = 3; s >= 0; s--) { }";
    assert_eq!(lint(good), Vec::<String>::new());
    assert_eq!(lint("for (int s = 3; s >= 0; s--) { }"), Vec::<String>::new());
}

#[test]
fn the_build_options_are_exactly_the_strict_ones() {
    assert_eq!(BUILD_OPTIONS, "-cl-std=CL1.2 -cl-fp32-correctly-rounded-divide-sqrt");
    for relaxing in ["fast-relaxed-math", "mad-enable", concat!("un", "safe-math"), "finite-math", "no-signed-zeros", "denorms-are-zero", "-w"] {
        assert!(!BUILD_OPTIONS.contains(relaxing), "{relaxing}");
    }
}

fn engine_src() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            rs_files(&p, out);
        } else if p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
}

#[test]
fn every_kernel_source_of_the_tree_is_in_the_registry() {
    let dir = engine_src().join("device").join("kernels");
    let mut files = Vec::new();
    rs_files(&dir, &mut files);
    let mut found = Vec::new();
    for f in &files {
        let stem = f.file_stem().unwrap().to_string_lossy().to_string();
        let text = std::fs::read_to_string(f).unwrap();
        for line in text.lines() {
            let Some(rest) = line.trim_start().strip_prefix("pub const ") else { continue };
            let Some((name, ty)) = rest.split_once(':') else { continue };
            if ty.trim_start().starts_with("&str = r#\"") {
                found.push(format!("{stem}::{name}"));
            }
        }
    }
    let registered: Vec<String> = SOURCES.iter().map(|(n, _)| n.to_string()).collect();
    found.sort();
    let mut want = registered.clone();
    want.sort();
    assert_eq!(found, want, "kernel sources under device/kernels and kernels::SOURCES differ");
    // no kernel source anywhere else in the engine
    let mut all = Vec::new();
    rs_files(&engine_src(), &mut all);
    for f in all {
        if f.starts_with(&dir) || f.ends_with("tests.rs") && f.parent().is_some_and(|p| p.ends_with("device")) {
            continue;
        }
        let text = std::fs::read_to_string(&f).unwrap();
        assert!(!text.contains("__kernel"), "{} holds a kernel source outside device/kernels", f.display());
    }
}

#[test]
fn starter_constants_match_their_sources() {
    assert!(starter::CHAIN.contains(&format!("s < {}", starter::CHAIN_STEPS)));
    assert!(starter::CONVERT.contains(&format!("i * {}", starter::CONVERT_F32_OUTPUTS)));
    assert!(starter::CONVERT.contains(&format!("i * {}", starter::CONVERT_F64_OUTPUTS)));
    assert_eq!(selfcheck::CONVERT_F32_NAMES.len(), starter::CONVERT_F32_OUTPUTS);
    for name in selfcheck::CONVERT_F32_NAMES {
        assert!(starter::CONVERT.contains(&format!("{name}(v)")), "{name}");
        assert!(prelude::PRELUDE.contains(&format!(" {name}(")), "{name} is a prelude function");
    }
    for k in ["copy_u8", "copy_u16", "copy_u32", "blend_u8", "blend_u16", "blend_f32", "convert_f32", "convert_f64", "index_2d", "chain_f32"] {
        assert!(kernels::starter_program().contains(&format!("__kernel void {k}(")), "{k}");
    }
}

// ---- prelude transcription against the CPU helpers ----------------------------------------------

#[test]
fn the_prelude_transcription_equals_every_cpu_conversion() {
    for x in selfcheck::convert_f32_inputs() {
        let s = x.show();
        assert_eq!(prelude::rs::cvtt_i32_f(x), crate::warp::cvtt32(x), "cvtt_i32_f {s}");
        assert_eq!(prelude::rs::cvtt_i32_f(x), crate::flow::sample::cvtt(x), "cvtt_i32_f / sample {s}");
        assert_eq!(prelude::rs::cvtt_i32_f(x), crate::timeline::cvtt_i32_f32(x), "cvtt_i32_f / timeline {s}");
        assert_eq!(prelude::rs::cvtt_i64_f(x), crate::blend::cvtt(x), "cvtt_i64_f {s}");
        assert_eq!(prelude::rs::q_round(x), crate::image::q(x), "q_round {s}");
        assert_eq!(prelude::rs::store_q_u8(x), crate::image::q(x) as u8, "store_q_u8 {s}");
        assert_eq!(prelude::rs::store_q_u16(x), crate::image::q(x) as u16, "store_q_u16 {s}");
        assert_eq!(prelude::rs::trunc_u8_f(x), crate::blend::cvtt(x) as u8, "trunc_u8_f {s}");
        assert_eq!(prelude::rs::trunc_u16_f(x), crate::blend::cvtt(x) as u16, "trunc_u16_f {s}");
        assert_eq!(prelude::rs::sat_u8(x), crate::warp::sat_u8(x), "sat_u8 {s}");
        assert_eq!(prelude::rs::sat_u16(x), crate::warp::sat_u16(x), "sat_u16 {s}");
        // the double cast restated in selfcheck is the one inside image::q
        assert_eq!(selfcheck::cvtt_i64_d(x as f64 + 0.5), crate::image::q(x), "cvtt_i64_d / q {s}");
    }
    for x in selfcheck::convert_f64_inputs() {
        assert_eq!(prelude::rs::cvtt_i32_d(x), crate::timeline::cvtt_i32(x), "cvtt_i32_d {}", x.show());
        assert_eq!(prelude::rs::cvtt_i64_d(x), selfcheck::cvtt_i64_d(x), "cvtt_i64_d {}", x.show());
    }
}

#[test]
fn the_conversions_follow_x86_at_the_edges() {
    use prelude::rs::*;
    assert_eq!(cvtt_i32_f(2147483648.0), i32::MIN);
    assert_eq!(cvtt_i32_f(-2147483648.0), i32::MIN);
    assert_eq!(cvtt_i32_f(f32::from_bits(2147483648.0f32.to_bits() - 1)), 2147483520);
    assert_eq!(cvtt_i32_f(f32::NAN), i32::MIN);
    assert_eq!(cvtt_i32_f(-1.9), -1);
    assert_eq!(cvtt_i64_f(f32::INFINITY), i64::MIN);
    assert_eq!(cvtt_i32_d(-2147483648.9999998), -2147483648);
    assert_eq!(cvtt_i32_d(-2147483649.0), i32::MIN);
    assert_eq!(cvtt_i64_d(9223372036854775808.0), i64::MIN);
    assert_eq!(cvtt_i64_d(-9223372036854775808.0), i64::MIN);
    assert_eq!(sat_u8(f32::NAN), 255);
    assert_eq!(sat_u8(-3.0), 0);
    assert_eq!(sat_u8(300.0), 255);
    assert_eq!(sat_u16(f32::NAN), 65535);
    assert_eq!(sat_u16(70000.0), 65535);
    assert_eq!(trunc_u8_f(256.5), 0, "low byte, no clamp");
    assert_eq!(trunc_u8_f(0.49999997 + 0.5), 1, "the f32 add rounds the near tie up");
    assert_eq!(store_q_u8(0.49999997), 0, "the double add does not");
}

// ---- kernels in the simulator against the CPU ----------------------------------------------------

#[test]
fn copy_transcriptions_equal_blend_copy_in_every_order() {
    let u8s: Vec<u8> = (0..1024).map(|i| (i * 37 % 256) as u8).collect();
    let u16s: Vec<u16> = (0..1024).map(|i| (i * 4099 % 65536) as u16).collect();
    let mut f32s = selfcheck::f32_specials();
    while f32s.len() % 4 != 0 {
        f32s.push(-1.0);
    }
    let n8 = u8s.len();
    let got = sim::same_for_every_order(|seed| {
        let mut dst = Global::new(n8, 0u8);
        sim::launch_1d(n8, seed, |i| starter::copy_item(i, &u8s, &mut dst, n8 as u32));
        assert_eq!(dst.written(), n8);
        dst.into_vec()
    });
    assert_eq!(got, selfcheck::cpu_copy(&u8s, selfcheck::u8_samples, selfcheck::samples_u8));
    let n16 = u16s.len();
    let got = sim::same_for_every_order(|seed| {
        let mut dst = Global::new(n16, 0u16);
        sim::launch_1d(n16, seed, |i| starter::copy_item(i, &u16s, &mut dst, n16 as u32));
        dst.into_vec()
    });
    assert_eq!(got, selfcheck::cpu_copy(&u16s, selfcheck::u16_samples, selfcheck::samples_u16));
    let bits: Vec<u32> = f32s.iter().map(|v| v.to_bits()).collect();
    let n = bits.len();
    let got = sim::same_for_every_order(|seed| {
        let mut dst = Global::new(n, 0u32);
        sim::launch_1d(n, seed, |i| starter::copy_item(i, &bits, &mut dst, n as u32));
        dst.into_vec()
    });
    let want: Vec<u32> = selfcheck::cpu_copy(&f32s, selfcheck::f32_samples, selfcheck::samples_f32).iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, want, "NaN payloads, -0.0 and subnormals survive the copy");
}

#[test]
fn blend_transcriptions_equal_blend_in_every_order() {
    let (a8, b8) = selfcheck::blend_u8_pairs();
    let (a16, b16) = selfcheck::blend_u16_pairs();
    for w in selfcheck::blend_weights() {
        let n = a8.len();
        let got = sim::same_for_every_order(|seed| {
            let mut out = Global::new(n, 0u8);
            sim::launch_1d(n, seed, |i| starter::blend_u8_item(i, &a8, &b8, &mut out, w, n as u32));
            out.into_vec()
        });
        let want = selfcheck::cpu_blend(&a8, &b8, w, selfcheck::u8_samples, selfcheck::samples_u8);
        assert_eq!(selfcheck::first_difference(&want, &got), None, "u8 w={}", w.show());
        let n = a16.len();
        let got = sim::same_for_every_order(|seed| {
            let mut out = Global::new(n, 0u16);
            sim::launch_1d(n, seed, |i| starter::blend_u16_item(i, &a16, &b16, &mut out, w, n as u32));
            out.into_vec()
        });
        let want = selfcheck::cpu_blend(&a16, &b16, w, selfcheck::u16_samples, selfcheck::samples_u16);
        assert_eq!(selfcheck::first_difference(&want, &got), None, "u16 w={}", w.show());
        let (a, b) = selfcheck::blend_f32_pairs(w);
        let n = a.len();
        let got = sim::same_for_every_order(|seed| {
            let mut out = Global::new(n, 0f32);
            sim::launch_1d(n, seed, |i| starter::blend_f32_item(i, &a, &b, &mut out, w, n as u32));
            out.into_vec().iter().map(|v| v.to_bits()).collect::<Vec<u32>>()
        });
        let got: Vec<f32> = got.into_iter().map(f32::from_bits).collect();
        let want = selfcheck::cpu_blend(&a, &b, w, selfcheck::f32_samples, selfcheck::samples_f32);
        // as in the device gate (`selfcheck::check_blend_f32`): NaN must land where the CPU has one, but with two NaN
        // operands the compiler may commute the multiply and keep the other NaN's sign and payload
        let (diff, _) = selfcheck::compare_floats(&want, &got);
        assert_eq!(diff, None, "f32 w={}: differs beyond NaN payloads", w.show());
    }
}

#[test]
fn convert_and_chain_transcriptions_equal_the_cpu_in_every_order() {
    let x = selfcheck::convert_f32_inputs();
    let n = x.len();
    let got = sim::same_for_every_order(|seed| {
        let mut out = Global::new(n * starter::CONVERT_F32_OUTPUTS, 0i64);
        sim::launch_1d(n, seed, |i| starter::convert_f32_item(i, &x, &mut out, n as u32));
        assert_eq!(out.written(), out.len());
        out.into_vec()
    });
    let want: Vec<i64> = x.iter().flat_map(|v| selfcheck::cpu_convert_f32(*v)).collect();
    assert_eq!(got, want);
    let d = selfcheck::convert_f64_inputs();
    let n = d.len();
    let got = sim::same_for_every_order(|seed| {
        let mut out = Global::new(n * starter::CONVERT_F64_OUTPUTS, 0i64);
        sim::launch_1d(n, seed, |i| starter::convert_f64_item(i, &d, &mut out, n as u32));
        out.into_vec()
    });
    assert_eq!(got, d.iter().flat_map(|v| selfcheck::cpu_convert_f64(*v)).collect::<Vec<i64>>());
    let (cx, ck) = selfcheck::chain_inputs();
    let n = cx.len();
    let got = sim::same_for_every_order(|seed| {
        let mut out = Global::new(n, 0f32);
        sim::launch_1d(n, seed, |i| starter::chain_f32_item(i, &cx, &ck, &mut out, n as u32));
        out.into_vec().iter().map(|v| v.to_bits()).collect::<Vec<u32>>()
    });
    let want: Vec<u32> = cx.iter().zip(&ck).map(|(x, k)| selfcheck::cpu_chain(*x, *k).to_bits()).collect();
    assert_eq!(got, want);
}

#[test]
fn the_index_transcription_lays_out_rows() {
    let (w, h) = selfcheck::INDEX_SIZE;
    let got = sim::same_for_every_order(|seed| {
        let mut out = Global::new(w * h, u32::MAX);
        // one item more on each axis, as a rounded up launch would have: the guard drops them
        sim::launch_2d(w + 1, h + 1, seed, |x, y| starter::index_2d_item(x, y, &mut out, w as u32, h as u32));
        assert_eq!(out.written(), w * h);
        out.into_vec()
    });
    assert_eq!(got, (0..(w * h) as u32).collect::<Vec<_>>());
}

#[test]
fn a_transcription_that_fuses_would_be_caught() {
    // the corpus is what makes the device check bite: a fused blend differs on it
    // a power of two weight makes the product exact, so only the others can fuse differently
    for w in selfcheck::blend_weights().into_iter().filter(|w| w.is_finite() && w.to_bits() & 0x007f_ffff != 0) {
        let (a, b) = selfcheck::blend_f32_pairs(w);
        let want = selfcheck::cpu_blend(&a, &b, w, selfcheck::f32_samples, selfcheck::samples_f32);
        let fused: Vec<f32> = a.iter().zip(&b).map(|(a, b)| (b - a).mul_add(w, *a) + 0.0).collect();
        let diffs = want.iter().zip(&fused).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        assert!(diffs >= 64, "w={}: only {diffs} FMA sensitive pairs", w.show());
    }
    let (x, k) = selfcheck::chain_inputs();
    let fused = x
        .iter()
        .zip(&k)
        .filter(|(x, k)| {
            let mut v = **x;
            for _ in 0..starter::CHAIN_STEPS {
                v = v.mul_add(**k, **x);
            }
            v.to_bits() != selfcheck::cpu_chain(**x, **k).to_bits()
        })
        .count();
    assert!(fused >= 256, "only {fused} chain inputs are FMA sensitive");
    // a denormal flush would be caught too
    let flushed = x.iter().filter(|v| v.is_subnormal()).count();
    assert!(flushed > 0);
}

// ---- simulator ----------------------------------------------------------------------------------

#[test]
fn orders_are_permutations_and_differ_by_seed() {
    for n in [0usize, 1, 2, 7, 1000] {
        for seed in sim::SEEDS {
            let mut o = sim::order(n, seed);
            o.sort_unstable();
            assert_eq!(o, (0..n).collect::<Vec<_>>(), "n={n} seed={seed}");
        }
    }
    assert_eq!(sim::order(10, 0), (0..10).collect::<Vec<_>>());
    assert_ne!(sim::order(1000, 1), sim::order(1000, 2));
    assert_ne!(sim::order(1000, 1), (0..1000).collect::<Vec<_>>());
    assert_eq!(sim::order(1000, 7), sim::order(1000, 7), "seeded, so repeatable");
}

#[test]
fn launch_2d_visits_every_item_once() {
    let (w, h) = (13, 7);
    for seed in sim::SEEDS {
        let mut seen = Global::new(w * h, (0usize, 0usize));
        sim::launch_2d(w, h, seed, |x, y| seen.set(y * w + x, (x, y)));
        assert_eq!(seen.written(), w * h);
        for (i, (x, y)) in seen.into_vec().into_iter().enumerate() {
            assert_eq!((x, y), (i % w, i / w));
        }
    }
    sim::launch_2d(0, 5, 1, |_, _| panic!("no item"));
}

#[test]
#[should_panic(expected = "written by two work items")]
fn a_double_write_is_a_race() {
    let mut out = Global::new(4, 0u8);
    sim::launch_1d(4, 1, |i| out.set(i / 2, 1));
}

#[test]
#[should_panic(expected = "depends on the work item order")]
fn an_order_dependent_transcription_is_caught() {
    sim::same_for_every_order(|seed| {
        let mut acc = 0f32;
        let vals = [1.0e8f32, 1.0, -1.0e8, 0.5, 3.0];
        sim::launch_1d(vals.len(), seed, |i| acc += vals[i]);
        acc.to_bits()
    });
}

#[test]
fn the_rng_is_splitmix64() {
    let mut r = sim::Rng::new(0);
    // SplitMix64 reference values for seed 0
    assert_eq!(r.next_u64(), 0xE220_A839_7B1D_CDAF);
    assert_eq!(r.next_u64(), 0x6E78_9E6A_A1B9_65F4);
    let mut r = sim::Rng::new(5);
    for _ in 0..1000 {
        assert!(r.below(3) < 3);
    }
}

// ---- corpus -------------------------------------------------------------------------------------

#[test]
fn the_corpus_holds_every_adversarial_class() {
    let s = selfcheck::f32_specials();
    let has = |f: &dyn Fn(f32) -> bool| s.iter().any(|v| f(*v));
    assert!(has(&|v| v.is_subnormal() && v > 0.0) && has(&|v| v.is_subnormal() && v < 0.0), "subnormals");
    assert!(has(&|v| v == 0.0 && v.is_sign_negative()), "-0.0");
    assert!(has(&|v| v.is_nan() && v.to_bits() & 0x003f_ffff != 0), "a NaN payload");
    assert!(has(&|v| v == f32::INFINITY) && has(&|v| v == f32::NEG_INFINITY), "infinities");
    assert!(has(&|v| v == -1.0), "exact -1.0");
    let c = selfcheck::convert_f32_inputs();
    assert!(c.contains(&2147483648.0) && c.contains(&-2147483648.0) && c.contains(&9223372036854775808.0));
    assert!(c.iter().any(|v| v.fract() == 0.5 || v.fract() == -0.5), ".5 ties");
    assert!(c.iter().any(|v| *v > 65535.0 && v.is_finite()) && c.iter().any(|v| *v < -128.0), "out of range casts");
    let d = selfcheck::convert_f64_inputs();
    assert!(d.contains(&-2147483649.0) && d.iter().any(|v| v.is_nan()) && d.iter().any(|v| v.is_subnormal()));
    for w in selfcheck::blend_weights() {
        assert!(selfcheck::in_blend_branch4(w), "{}", w.show());
    }
    assert!(selfcheck::blend_weights().iter().any(|w| w.is_nan()));
    assert!(!selfcheck::in_blend_branch4(0.005) && !selfcheck::in_blend_branch4(1.0) && !selfcheck::in_blend_branch4(-0.5));
    assert_eq!(selfcheck::blend_u8_pairs().0.len(), 65536);
    assert_eq!(selfcheck::blend_u16_pairs().0.len() % 4, 0);
    for w in selfcheck::blend_weights() {
        assert_eq!(selfcheck::blend_f32_pairs(w).0.len() % 4, 0);
    }
}

#[test]
fn nan_payloads_are_told_apart_from_differences() {
    let q = f32::from_bits(0x7fc0_0000);
    let canon = f32::from_bits(0x7fff_ffff);
    assert_eq!(selfcheck::compare_floats(&[1.0f32, q, 2.0], &[1.0, canon, 2.0]), (None, Some(1)));
    assert_eq!(selfcheck::compare_floats(&[1.0f32, q, 2.0], &[1.0, canon, 3.0]), (Some(2), Some(1)));
    assert_eq!(selfcheck::compare_floats(&[q], &[1.0f32]), (Some(0), None), "NaN against a number is a difference");
    assert_eq!(selfcheck::compare_floats(&[1.0f32], &[q]), (Some(0), None));
    assert_eq!(selfcheck::compare_floats(&[0.0f32], &[-0.0]), (Some(0), None));
    assert_eq!(selfcheck::compare_floats(&[q, q], &[q, q]), (None, None));
    let mut d = good_device();
    assert_eq!(d.display_name(), "Test GPU");
    d.board = "Test Board".into();
    assert_eq!(d.display_name(), "Test Board");
}

#[test]
fn first_difference_compares_bits() {
    assert_eq!(selfcheck::first_difference(&[0.0f32, 1.0], &[0.0, 1.0]), None);
    assert_eq!(selfcheck::first_difference(&[0.0f32], &[-0.0]), Some(0));
    assert_eq!(selfcheck::first_difference(&[f32::NAN], &[f32::NAN]), None);
    assert_eq!(selfcheck::first_difference(&[f32::from_bits(0x7fc0_0000)], &[f32::from_bits(0x7fff_ffff)]), Some(0));
    assert_eq!(selfcheck::first_difference(&[1u8, 2, 3], &[1, 2]), Some(2));
}

// ---- probe over synthetic devices ---------------------------------------------------------------

fn good_device() -> DeviceInfo {
    DeviceInfo {
        index: 0,
        platform: "Test Platform".into(),
        platform_version: "OpenCL 3.0".into(),
        name: "Test GPU".into(),
        board: String::new(),
        vendor: "Test Vendor".into(),
        version: "OpenCL 3.0 test".into(),
        driver: "1.0".into(),
        c_version: "OpenCL C 1.2 ".into(),
        profile: "FULL_PROFILE".into(),
        device_type: cl::CL_DEVICE_TYPE_GPU,
        little_endian: true,
        available: true,
        compiler_available: true,
        global_memory: 8 << 30,
        max_allocation: 2 << 30,
        unified_memory: false,
        compute_units: 24,
        extensions: vec!["cl_khr_global_int32_base_atomics".into(), "cl_khr_fp64".into()],
        single_fp: 0xbf,
        double_fp: 0x3f,
        errors: Vec::new(),
    }
}

#[test]
fn a_device_meeting_every_requirement_is_usable() {
    assert_eq!(probe::requirements(&good_device()), Vec::<String>::new());
    assert!(probe::usable(&good_device()));
    // the minimum: exactly the required bits
    let mut d = good_device();
    d.single_fp = cl::CL_FP_DENORM | cl::CL_FP_INF_NAN | cl::CL_FP_ROUND_TO_NEAREST | cl::CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT;
    d.double_fp = cl::CL_FP_DENORM | cl::CL_FP_INF_NAN | cl::CL_FP_ROUND_TO_NEAREST;
    d.c_version = "OpenCL C 3.0 ".into();
    assert!(probe::usable(&d));
}

#[test]
fn every_requirement_failure_is_named() {
    type Edit = fn(&mut DeviceInfo);
    let cases: Vec<(Edit, &str)> = vec![
        (|d| d.profile = "EMBEDDED_PROFILE".into(), "EMBEDDED_PROFILE, needs FULL_PROFILE"),
        (|d| d.profile = String::new(), "no profile, needs FULL_PROFILE"),
        (|d| d.little_endian = false, "big endian"),
        (|d| d.extensions.retain(|e| e != "cl_khr_fp64"), "no cl_khr_fp64 (double precision)"),
        (|d| d.single_fp &= !cl::CL_FP_DENORM, "single precision without DENORM"),
        (|d| d.single_fp &= !cl::CL_FP_INF_NAN, "single precision without INF_NAN"),
        (|d| d.single_fp &= !cl::CL_FP_ROUND_TO_NEAREST, "single precision without ROUND_TO_NEAREST"),
        (
            |d| d.single_fp &= !cl::CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT,
            "single precision without CORRECTLY_ROUNDED_DIVIDE_SQRT",
        ),
        (|d| d.double_fp &= !cl::CL_FP_DENORM, "double precision without DENORM"),
        (|d| d.double_fp &= !cl::CL_FP_INF_NAN, "double precision without INF_NAN"),
        (|d| d.double_fp &= !cl::CL_FP_ROUND_TO_NEAREST, "double precision without ROUND_TO_NEAREST"),
        (|d| d.c_version = "OpenCL C 1.1 ".into(), "OpenCL C 1.1, needs 1.2 or later"),
        (|d| d.c_version = "garbage".into(), "unreadable OpenCL C version `garbage`"),
        (|d| d.available = false, "not available"),
        (|d| d.compiler_available = false, "no compiler"),
        (|d| d.errors.push("CL_DEVICE_NAME: x".into()), "probe failed: CL_DEVICE_NAME: x"),
    ];
    for (edit, want) in cases {
        let mut d = good_device();
        edit(&mut d);
        assert_eq!(probe::requirements(&d), vec![want.to_string()]);
        assert!(!probe::usable(&d));
    }
    // bits that are not required change nothing
    let mut d = good_device();
    d.single_fp &= !(cl::CL_FP_FMA | cl::CL_FP_ROUND_TO_ZERO | cl::CL_FP_ROUND_TO_INF);
    d.double_fp &= !(cl::CL_FP_FMA | cl::CL_FP_ROUND_TO_ZERO);
    assert!(probe::usable(&d));
    // a device without double precision at all collects every double reason
    let mut d = good_device();
    d.extensions.clear();
    d.double_fp = 0;
    assert_eq!(probe::requirements(&d).len(), 4);
}

#[test]
fn versions_and_flags_parse() {
    assert_eq!(probe::parse_version("OpenCL 3.0 CUDA", "OpenCL "), Some((3, 0)));
    assert_eq!(probe::parse_version("OpenCL 2.0 AMD-APP (3628.0)", "OpenCL "), Some((2, 0)));
    assert_eq!(probe::parse_version("OpenCL C 1.2 ", "OpenCL C "), Some((1, 2)));
    assert_eq!(probe::parse_version("  OpenCL C 2.0", "OpenCL C "), Some((2, 0)));
    for bad in ["", "OpenCL", "OpenCL x.y", "OpenCL 3", "OpenCL 3.", "OpenCL .1", "OCL 1.2", "OpenCL -1.2"] {
        assert_eq!(probe::parse_version(bad, "OpenCL "), None, "{bad:?}");
    }
    assert_eq!(probe::fp_flags(0), Vec::<String>::new());
    assert_eq!(probe::fp_flags(0xbf), ["DENORM", "INF_NAN", "ROUND_TO_NEAREST", "ROUND_TO_ZERO", "ROUND_TO_INF", "FMA", "CORRECTLY_ROUNDED_DIVIDE_SQRT"]);
    assert_eq!(probe::fp_flags(0x40 | 0x300), ["SOFT_FLOAT", "0x300"]);
    let mut d = good_device();
    assert_eq!(d.type_name(), "gpu");
    d.device_type = cl::CL_DEVICE_TYPE_CPU;
    assert_eq!(d.type_name(), "cpu");
    d.device_type = cl::CL_DEVICE_TYPE_ACCELERATOR | cl::CL_DEVICE_TYPE_DEFAULT;
    assert_eq!(d.type_name(), "accelerator");
    d.device_type = 0;
    assert_eq!(d.type_name(), "other");
}

#[test]
fn gpu_prefers_dedicated_memory_and_budgets_three_quarters() {
    let mut igpu = good_device();
    igpu.unified_memory = true;
    let dgpu = good_device();
    let mut bad = good_device();
    bad.little_endian = false;
    assert_eq!(probe::preferred(&[igpu.clone(), dgpu.clone()]), Some(1));
    assert_eq!(probe::preferred(&[bad.clone(), igpu.clone()]), Some(1));
    assert_eq!(probe::preferred(&[bad.clone()]), None);
    assert_eq!(probe::preferred(&[]), None);
    assert_eq!(probe::memory_budget(&dgpu), 6 << 30);
}

// ---- loader and errors --------------------------------------------------------------------------

#[test]
fn a_missing_runtime_is_an_error_not_a_panic() {
    let err = cl::Api::load("twixtor-no-such-opencl-runtime-1").unwrap_err();
    match &err {
        ClError::NoRuntime { library, reason } => {
            assert_eq!(library, "twixtor-no-such-opencl-runtime-1");
            assert!(!reason.is_empty());
        }
        other => panic!("{other:?}"),
    }
    assert!(err.to_string().starts_with("no OpenCL runtime (twixtor-no-such-opencl-runtime-1: "), "{err}");
}

#[cfg(windows)]
#[test]
fn windows_loads_names_only_and_reports_a_missing_entry_point() {
    for path in ["C:\\Windows\\System32\\OpenCL.dll", ".\\OpenCL.dll", "sub/OpenCL.dll"] {
        match cl::Api::load(path) {
            Err(ClError::NoRuntime { reason, .. }) => assert!(reason.contains("never a path"), "{reason}"),
            other => panic!("{path}: {other:?}"),
        }
    }
    // kernel32 loads from System32 but exports no OpenCL entry point
    assert_eq!(
        cl::Api::load("kernel32.dll").unwrap_err(),
        ClError::MissingEntry { library: "kernel32.dll".into(), name: "clGetPlatformIDs" }
    );
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[test]
fn unix_reports_a_missing_entry_point() {
    assert_eq!(
        cl::Api::load("libc.so.6").unwrap_err(),
        ClError::MissingEntry { library: "libc.so.6".into(), name: "clGetPlatformIDs" }
    );
}

#[test]
fn the_entry_points_and_error_codes_are_well_formed() {
    assert!(cl::ENTRY_POINTS.len() >= 25, "{}", cl::ENTRY_POINTS.len());
    for name in cl::ENTRY_POINTS {
        assert!(name.starts_with("cl") && name[2..].starts_with(|c: char| c.is_ascii_uppercase()), "{name}");
    }
    let mut names: Vec<&str> = cl::ENTRY_POINTS.to_vec();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), cl::ENTRY_POINTS.len(), "no entry point twice");
    let mut codes: Vec<i32> = cl::ERRORS.iter().map(|(c, _)| *c).collect();
    codes.sort_unstable();
    codes.dedup();
    assert_eq!(codes.len(), cl::ERRORS.len(), "no code twice");
    assert_eq!(cl::error_name(-30), Some("CL_INVALID_VALUE"));
    assert_eq!(cl::error_name(-11), Some("CL_BUILD_PROGRAM_FAILURE"));
    assert_eq!(cl::error_name(-46), Some("CL_INVALID_KERNEL_NAME"));
    assert_eq!(cl::error_name(-9999), None);
    assert_eq!(cl::code_text(-5), "CL_OUT_OF_RESOURCES (-5)");
    assert_eq!(cl::code_text(-9999), "status -9999");
    assert_eq!(ClError::Api { call: "clFinish", code: -36 }.to_string(), "clFinish: CL_INVALID_COMMAND_QUEUE (-36)");
    assert_eq!(ClError::Build { code: -11, log: " line 3: error \n".into() }.to_string(), "program build failed: CL_BUILD_PROGRAM_FAILURE (-11): line 3: error");
    assert_eq!(ClError::Build { code: -11, log: String::new() }.to_string(), "program build failed: CL_BUILD_PROGRAM_FAILURE (-11)");
    // the fp bits are the eight of OpenCL 1.2, in order
    let bits: Vec<u64> = probe::FP_FLAGS.iter().map(|(b, _)| *b).collect();
    assert_eq!(bits, (0..8).map(|i| 1u64 << i).collect::<Vec<_>>());
}

#[test]
fn ranges_count_their_items() {
    assert_eq!(Range::D1(5).items(), Some(5));
    assert_eq!(Range::D2(4, 3).items(), Some(12));
    assert_eq!(Range::D2(usize::MAX, 2).items(), None);
    assert!(cl::MAX_WORK_ITEMS >= 4096 * 2160 * 4);
}
