//! Device-free tests of the solver stages: every `sv_` kernel's Rust transcription run through the
//! simulator (shuffled work item orders, write-once outputs, an ownership check on the in place
//! sweeps) and compared bit for bit with the CPU function on random and adversarial inputs, then
//! whole solves (the very orchestration the device runs) compared with `solve_pyramids_with`,
//! `solve` and `solve_pair` level by level.

use super::cases::{self, Case};
use super::sim::backend;
use super::*;
use crate::device::convstage::corpus;
use crate::device::kernels::solver as ks;
use crate::device::sim::{Rng, SEEDS};
use crate::flow::extras::{self, UserLevel};
use crate::flow::{self, MASK_ON};

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Panics with the first differing index. NaN payloads are not compared (the compiler may
/// commute an operation of the CPU code, which picks another payload, and a device result holding
/// a NaN is refused by the wrappers anyway); every other value must have the CPU's bits.
fn assert_bits(cpu: &[f32], dev: &[f32], what: &str) {
    assert_eq!(cpu.len(), dev.len(), "{what}: lengths");
    if let Some(i) = (0..cpu.len()).find(|&i| !cases::same(cpu[i], dev[i])) {
        panic!("{what}: first difference at {i}: cpu {:e} ({:#010x}) sim {:e} ({:#010x})", cpu[i], cpu[i].to_bits(), dev[i], dev[i].to_bits());
    }
}

/// A plane of `n` values: smooth values with the adversarial specials at `rate`.
fn plane(rng: &mut Rng, n: usize, rate: f64) -> Vec<f32> {
    corpus::plane(rng, n, rate)
}

/// A flow of `n` vectors for a `w x h` level: small, integer, half, large and special values.
fn flow_values(rng: &mut Rng, w: usize, h: usize, special: bool) -> Vec<f32> {
    let n = w * h * 2;
    (0..n)
        .map(|i| {
            let side = if i % 2 == 0 { w } else { h } as f32;
            match rng.below(if special { 12 } else { 9 }) {
                0 => 0.0,
                1 => -0.0,
                2 => (corpus::uniform(rng, -3.0, 3.0)).round(),
                3 => (corpus::uniform(rng, -3.0, 3.0)).floor() + 0.5,
                4 => corpus::uniform(rng, -side * 1.5, side * 1.5),
                5 => -f32::from_bits(1),
                6 => corpus::uniform(rng, -side * 70000.0, side * 70000.0),
                9 => f32::NAN,
                10 => f32::INFINITY,
                11 => f32::NEG_INFINITY,
                _ => corpus::uniform(rng, -2.0, 2.0),
            }
        })
        .collect()
}

/// A binary level mask with a share of zeros (and a few values next to the thresholds).
fn mask(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| match rng.below(10) {
            0..=2 => 0.0,
            3 => MASK_ON,
            4 => f32::from_bits(MASK_ON.to_bits() - 1),
            5 => ks::SV_WARPED_MASK_MIN,
            _ => 1.0,
        })
        .collect()
}

const SIZES: [(usize, usize); 9] = [(2, 2), (3, 3), (2, 5), (5, 2), (4, 4), (7, 3), (9, 8), (16, 5), (13, 11)];

#[test]
fn literals_match_the_cpu_constants() {
    assert_eq!(ks::SV_MASK_ON.to_bits(), MASK_ON.to_bits());
    assert_eq!(ks::SV_WARPED_MASK_MIN.to_bits(), 0.999f32.to_bits());
    assert_eq!(ks::SV_UPDATE_MIN.to_bits(), 40.0f32.to_bits());
    // the hexadecimal literals of the source are those values
    assert!(ks::SOLVER.contains("0x1.e66666p-1f") && 0x3f73_3333 == MASK_ON.to_bits());
    assert!(ks::SOLVER.contains("0x1.ff7ceep-1f") && 0x3f7f_be77 == 0.999f32.to_bits());
    assert!(ks::SOLVER.contains("0x1.4p+5f") && 40.0f32.to_bits() == 0x4220_0000);
    assert!(ks::SOLVER.contains("0x1p+16f") && 65536.0f32.to_bits() == 0x4780_0000);
    assert!(ks::SOLVER.contains("k < 512") && ks::SV_SWEEP_CHUNK == 512);
    assert!(ks::SOLVER.contains("< 16384") && ks::SV_LINE_MAX == 16384);
}

#[test]
fn every_solver_identifier_is_prefixed_and_the_parts_hold_the_samplers() {
    let mut count = 0;
    // the kernel qualifier spelled in two pieces: no kernel source outside device/kernels
    let kernel_void = format!("{}kernel void ", "__");
    for line in ks::SOLVER.lines() {
        let t = line.trim_start();
        let decl = [kernel_void.as_str(), "int ", "float ", "float2 ", "void "].iter().find_map(|p| t.strip_prefix(p).filter(|_| !line.starts_with(' ')));
        if let Some(rest) = decl {
            if rest.contains('(') && t.ends_with('{') {
                count += 1;
                assert!(rest.starts_with("sv_"), "unprefixed function: {t}");
            }
        }
    }
    assert!(count >= 20, "{count} functions");
    let parts = program_parts();
    assert_eq!(parts.last(), Some(&ks::SOLVER));
    for p in [crate::device::kernels::resample::SAMPLERS, crate::device::kernels::conv::CONV, crate::device::kernels::pyramid::PYRAMID] {
        assert!(parts.contains(&p));
    }
    assert!(program().starts_with("#pragma OPENCL FP_CONTRACT OFF\n"));
}

#[test]
fn wrap_target_matches_the_cpu() {
    let mut rng = Rng::new(11);
    for &w in &[1usize, 2, 3, 7, 64, 641] {
        let mut us = corpus::specials();
        us.extend([-0.5, 0.5, -1.0e-10, 1.0e-10, -(w as f32), w as f32 * 3.25, -(w as f32) * 65536.0, w as f32 * 65536.5]);
        for _ in 0..200 {
            us.push(corpus::uniform(&mut rng, -(w as f32) * 4.0, w as f32 * 4.0));
        }
        for x in 0..w.min(9) {
            for &u in &us {
                let cpu = flow::wrap_target(x, u, w);
                let dev = ks::sv_wrap_target(x as i32, u, w as i32).map(|(c, f)| (c as usize, f.to_bits()));
                assert_eq!(cpu.map(|(c, f)| (c, f.to_bits())), dev, "x {x} u {u:e} w {w}");
            }
        }
    }
}

#[test]
#[ignore = "the kernel follows the rule before the parity work; port pending (docs/engine.md, Device)"]
fn gradient_masking_and_tensor_match_the_cpu() {
    let mut rng = Rng::new(21);
    for &(w, h) in &SIZES {
        for k in 0..6 {
            let wrap = k % 2 == 1;
            let with_mask = k >= 2;
            let img = plane(&mut rng, w * h, if k >= 4 { 0.2 } else { 0.0 });
            let m = mask(&mut rng, w * h);
            let mut grad = if wrap { flow::calc_grad_wrap(&img, w, h) } else { flow::calc_grad(&img, w, h) };
            if with_mask {
                flow::mask_gradient(&mut grad, &m, w, h, wrap);
            }
            let tensor = flow::calc_a(&grad, w, h);
            for &seed in &SEEDS {
                let b = backend(seed);
                let (ib, mb) = (b.upload(&img).unwrap(), b.upload(&m).unwrap());
                let (gb, tb) = (b.alloc::<f32>(w * h * 2).unwrap(), b.alloc::<f32>(w * h * 3).unwrap());
                b.sv_launch(SvLaunch::Grad { img: &ib, mask: with_mask.then_some(&mb), wrap, grad: &gb, tensor: &tb, w, h }).unwrap();
                assert_bits(&grad, &b.download(&gb).unwrap(), &format!("grad {w}x{h} k {k}"));
                assert_bits(&tensor, &b.download(&tb).unwrap(), &format!("tensor {w}x{h} k {k}"));
            }
        }
    }
}

#[test]
#[ignore = "the kernel follows the rule before the parity work; port pending (docs/engine.md, Device)"]
fn right_hand_side_matches_the_cpu() {
    let mut rng = Rng::new(31);
    for &(w, h) in &SIZES {
        for k in 0..8 {
            let wrap = k % 2 == 1;
            let with_mask = k % 4 >= 2;
            let special = k >= 4;
            let i1 = plane(&mut rng, w * h, if special { 0.1 } else { 0.0 });
            let i0 = plane(&mut rng, w * h, if special { 0.1 } else { 0.0 });
            let grad = plane(&mut rng, w * h * 2, if special { 0.1 } else { 0.0 });
            let fl = flow_values(&mut rng, w, h, special);
            let my = mask(&mut rng, w * h);
            let cpu = if !with_mask && !wrap {
                flow::calc_b(&i1, &i0, &grad, &fl, w, h)
            } else {
                flow::calc_b_with(&i1, &i0, &grad, &fl, w, h, with_mask.then_some(&my[..]), wrap)
            };
            for &seed in &SEEDS {
                let b = backend(seed);
                let up = |v: &[f32]| b.upload(v).unwrap();
                let (b1, b0, bg, bf, bm) = (up(&i1), up(&i0), up(&grad), up(&fl), up(&my));
                let out = b.alloc::<f32>(w * h * 2).unwrap();
                b.sv_launch(SvLaunch::CalcB { i1: &b1, i0: &b0, grad: &bg, flow: &bf, mask: with_mask.then_some(&bm), wrap, b: &out, w, h }).unwrap();
                assert_bits(&cpu, &b.download(&out).unwrap(), &format!("calc_b {w}x{h} k {k}"));
            }
        }
    }
}

#[test]
fn update_user_displacement_scale_and_range_check_match_the_cpu() {
    let mut rng = Rng::new(41);
    for &(w, h) in &SIZES {
        let n = w * h;
        for k in 0..4 {
            let special = k >= 2;
            let fl = flow_values(&mut rng, w, h, special);
            // tensor values around the |t| > 40 test: t = b*b - dd*a
            let a: Vec<f32> = (0..n * 3).map(|i| if i % 7 == 0 { 6.5 } else { corpus::uniform(&mut rng, -30.0, 30.0) }).collect();
            let bv = plane(&mut rng, n * 2, if special { 0.1 } else { 0.0 });
            let mut cpu = fl.clone();
            flow::new_disp(&mut cpu, &a, &bv);
            let d = plane(&mut rng, n * 2, 0.1);
            let wt: Vec<f32> = (0..n).map(|_| [0.0, -0.0, 1.0, 3.0, f32::NAN, 0.5][rng.below(6) as usize]).collect();
            let mut cpu_user = fl.clone();
            extras::apply_user(&mut cpu_user, &UserLevel { d: d.clone(), w: wt.clone() }, w);
            let f = [0.9709f32, -1.0, 1.0, 0.0][k];
            let mut cpu_scale = fl.clone();
            flow::scale_values(&mut cpu_scale, f);
            if f == 1.0 {
                // the CPU skips a scale by 1.0; the driver never launches one
                cpu_scale = fl.iter().map(|v| 1.0 * v).collect();
            }
            for &seed in &SEEDS {
                let b = backend(seed);
                let bf = b.upload(&fl).unwrap();
                let (ba, bb) = (b.upload(&a).unwrap(), b.upload(&bv).unwrap());
                b.sv_launch(SvLaunch::NewDisp { flow: &bf, a: &ba, b: &bb, n }).unwrap();
                assert_bits(&cpu, &b.download(&bf).unwrap(), "new_disp");
                let bf = b.upload(&fl).unwrap();
                let (bd, bw) = (b.upload(&d).unwrap(), b.upload(&wt).unwrap());
                b.sv_launch(SvLaunch::ApplyUser { flow: &bf, d: &bd, wt: &bw, n }).unwrap();
                assert_bits(&cpu_user, &b.download(&bf).unwrap(), "apply_user");
                let bf = b.upload(&fl).unwrap();
                b.sv_launch(SvLaunch::Scale { data: &bf, f, n: n * 2 }).unwrap();
                assert_bits(&cpu_scale, &b.download(&bf).unwrap(), "scale");
            }
        }
        // the range check and its state over three levels: out, out, in
        for wrap in [false, true] {
            let mut flows = vec![[1.0e6f32, 1.0e6].repeat(n), flow_values(&mut rng, w, h, true), [0.25f32, -0.25].repeat(n)];
            flows[1][3] = f32::NAN;
            let mut check = true;
            for &seed in &SEEDS[..2] {
                let b = backend(seed);
                let state = b.upload(&[1i32, 0]).unwrap();
                check = true;
                for fl in &flows {
                    let mut cpu = fl.clone();
                    if check {
                        let out = if wrap { flow::all_flow_out_of_range_y(&cpu, w, h) } else { flow::all_flow_out_of_range(&cpu, w, h) };
                        if out {
                            cpu.iter_mut().for_each(|v| *v = 0.0);
                        } else {
                            check = false;
                        }
                    }
                    let bf = b.upload(fl).unwrap();
                    let rows = b.alloc::<i32>(h).unwrap();
                    b.sv_launch(SvLaunch::RangeRows { flow: &bf, state: &state, rows: &rows, w, h, wrap }).unwrap();
                    b.sv_launch(SvLaunch::RangeDecide { state: &state, rows: &rows, h }).unwrap();
                    b.sv_launch(SvLaunch::RangeApply { flow: &bf, state: &state, n: n * 2 }).unwrap();
                    assert_bits(&cpu, &b.download(&bf).unwrap(), &format!("range {w}x{h} wrap {wrap}"));
                    assert_eq!(b.download(&state).unwrap()[0], check as i32);
                }
            }
            assert!(!check);
        }
    }
}

#[test]
#[ignore = "the kernel follows the rule before the parity work; port pending (docs/engine.md, Device)"]
fn block_matching_sweeps_match_the_cpu_in_every_order() {
    let mut rng = Rng::new(51);
    let mut sizes = SIZES.to_vec();
    sizes.extend([(20, 15), (31, 9), (1100, 4), (5, 1030)]);
    let (mut total, mut changed) = (0, 0);
    for &(w, h) in &sizes {
        let n = w * h;
        let reps = if n > 2000 { 2 } else { 6 };
        for k in 0..reps {
            let wrap = k % 2 == 1;
            let gated = k % 4 >= 2;
            let img_a = plane(&mut rng, n, if k >= 4 { 0.05 } else { 0.0 });
            // B: A moved by one pixel plus noise, so the sweeps take neighbours
            let img_b: Vec<f32> = (0..n).map(|i| img_a[(i + 1) % n] + corpus::uniform(&mut rng, -1.0, 1.0)).collect();
            let fl = flow_values(&mut rng, w, h, k >= 4);
            let gate = mask(&mut rng, n);
            let mut cpu = fl.clone();
            if !gated && !wrap {
                flow::block_match(&img_a, &img_b, &mut cpu, w, h);
            } else {
                flow::block_match_with(&img_a, &img_b, &mut cpu, w, h, gated.then_some(&gate[..]), wrap);
            }
            let seeds: &[u64] = if n > 2000 { &SEEDS[..2] } else { &SEEDS };
            for &seed in seeds {
                let b = backend(seed);
                let (ba, bb, bf, bg) = (b.upload(&img_a).unwrap(), b.upload(&img_b).unwrap(), b.upload(&fl).unwrap(), b.upload(&gate).unwrap());
                refine_on(&b, &ba, &bb, &bf, w, h, gated.then_some(&bg), wrap).unwrap();
                assert_bits(&cpu, &b.download(&bf).unwrap(), &format!("block match {w}x{h} k {k} seed {seed:#x}"));
            }
            total += 1;
            if bits(&cpu) != bits(&fl) {
                changed += 1;
            }
        }
    }
    assert!(changed * 2 >= total, "the sweeps changed only {changed} of {total} fields (weak cases)");
}

#[test]
#[ignore = "the kernel follows the rule before the parity work; port pending (docs/engine.md, Device)"]
fn wrap_upsample_matches_the_cpu() {
    let mut rng = Rng::new(61);
    for &((sw, sh), (dw, dh)) in &[((2, 2), (4, 5)), ((3, 3), (6, 6)), ((5, 4), (11, 9)), ((8, 6), (16, 12)), ((7, 2), (14, 3))] {
        let src = flow_values(&mut rng, sw, sh, true);
        let cpu = flow::upsample_flow_wrapx(&src, sw, sh, dw, dh);
        for &seed in &SEEDS {
            let b = backend(seed);
            let s = b.upload(&src).unwrap();
            let d = b.alloc::<f32>(dw * dh * 2).unwrap();
            b.sv_launch(SvLaunch::UpsampleWrapx { src: &s, sw, sh, dst: &d, dw, dh, scale: upsample_wrapx_scales(sw, sh, dw, dh) }).unwrap();
            assert_bits(&cpu, &b.download(&d).unwrap(), "upsample wrapx");
        }
    }
}

/// A whole solve on the simulator against the CPU, with the level trace on a difference. Returns
/// whether the CPU result was compared (false when the CPU refuses the case).
fn check_case(c: &Case, seed: u64) -> bool {
    let Some(r) = cases::reference(c) else { return false };
    let b = backend(seed);
    let mut dev_trace = Vec::new();
    let got = solve_pyramids_with_traced(&b, &r.px, &r.py, &c.s, r.extras(), &mut |l, buf| {
        dev_trace.push((l, b.download(buf)?));
        Ok(())
    });
    let cpu_nan = r.flow.data.iter().any(|v| v.is_nan());
    let fail = match &got {
        Ok(f) if !cpu_nan && bits(&f.data) == bits(&r.flow.data) => None,
        Err(StageError::NanOutput) if cpu_nan => None,
        Ok(_) => Some("the flows differ".to_string()),
        Err(e) => Some(format!("refused: {e}")),
    };
    if let Some(why) = fail {
        let cpu_trace = cases::cpu_trace(&r, &c.s);
        let at = cases::first_trace_difference(&cpu_trace, &dev_trace, &r.px.geometry().sizes).unwrap_or_else(|| "the traces agree".into());
        panic!("{} (seed {seed:#x}): {why}; {at}", c.name);
    }
    true
}

/// A case of the plain solve (`solve`, `solve_pair`): no 360 flag, no masks, no user points.
fn plain(c: &Case) -> bool {
    !c.s.wrap && c.mattes.is_none() && c.points.is_empty()
}

#[test]
fn whole_solves_on_the_simulator_match_the_cpu() {
    let cs = cases::corpus(0xD5, 96, 33);
    let mut compared = 0;
    for (k, c) in cs.iter().enumerate().filter(|(_, c)| plain(c)) {
        let seed = SEEDS[k % SEEDS.len()];
        if check_case(c, seed) {
            compared += 1;
        }
    }
    assert!(compared >= 30, "{compared} solves compared");
}

#[test]
#[ignore = "the kernel follows the rule before the parity work; port pending (docs/engine.md, Device)"]
fn layered_and_360_solves_on_the_simulator_match_the_cpu() {
    let cs = cases::corpus(0xD5, 96, 33);
    let mut compared = 0;
    for (k, c) in cs.iter().enumerate().filter(|(_, c)| !plain(c)) {
        let seed = SEEDS[k % SEEDS.len()];
        if check_case(c, seed) {
            compared += 1;
        }
    }
    assert!(compared >= 20, "{compared} solves compared");
}

#[test]
fn the_cpu_trace_ends_in_the_cpu_flow() {
    for c in cases::corpus(0xE1, 20, 24) {
        let Some(r) = cases::reference(&c) else { continue };
        let t = cases::cpu_trace(&r, &c.s);
        let mut last = t.last().unwrap().1.clone();
        flow::scale_values(&mut last, -1.0);
        assert!(last.iter().zip(&r.flow.data).all(|(a, b)| cases::same(*a, *b)), "{}", c.name);
    }
}

#[test]
fn every_corpus_class_is_exercised_and_changes_the_flow() {
    let cs = cases::corpus(0xD5, 60, 33);
    let names: Vec<&str> = cs.iter().map(|c| c.name.as_str()).collect();
    for class in ["translate", "rotate", "still", "large", "mask", "points", "wrap", "spike"] {
        assert!(names.iter().any(|n| n.contains(class)), "no {class} case");
    }
    // the corpus is not trivial: most solves give a nonzero flow
    // (the tiny sizes of this corpus often give the exact zero flow: one level, no update)
    let nonzero = cs.iter().filter_map(cases::reference).filter(|r| r.flow.data.iter().any(|v| *v != 0.0 && !v.is_nan())).count();
    assert!(nonzero >= 18, "only {nonzero} nonzero flows");

}

#[test]
fn solve_and_solve_pair_on_the_simulator_match_the_cpu() {
    let mut n = 0;
    for (k, c) in cases::corpus(0xF00, 24, 24).iter().enumerate() {
        let seed = SEEDS[k % SEEDS.len()];
        let b = backend(seed);
        if c.s.wrap {
            // the 360 solve has no kernels yet: refused, the CPU solves
            assert!(matches!(solve_pair_host(&b, &c.a, &c.b, &c.s), Err(StageError::Unsupported(_))), "{}", c.name);
            assert!(matches!(solve_host(&b, &c.a, &c.b, &c.s), Err(StageError::Unsupported(_))), "{}", c.name);
            continue;
        }
        let cpu = flow::solve_pair(&c.a, &c.b, &c.s);
        let dev = solve_pair_host(&b, &c.a, &c.b, &c.s);
        match (cpu, dev) {
            (Ok(p), Ok(d)) => {
                assert_bits(&p.f_ab.data, &d.f_ab.data, &format!("{} f_ab", c.name));
                assert_bits(&p.f_ba.data, &d.f_ba.data, &format!("{} f_ba", c.name));
                n += 1;
            }
            (Ok(p), Err(StageError::NanOutput)) => assert!(p.f_ab.data.iter().chain(&p.f_ba.data).any(|v| v.is_nan()), "{}", c.name),
            (Err(_), Err(StageError::Unsupported(_))) => {}
            (cpu, dev) => panic!("{}: cpu {:?} device {:?}", c.name, cpu.is_ok(), dev.err()),
        }
        let cpu = flow::solve(&c.a, &c.b, &c.s);
        let dev = solve_host(&b, &c.a, &c.b, &c.s);
        match (cpu, dev) {
            (Ok(f), Ok(d)) => assert_bits(&f.data, &d.data, &format!("{} solve", c.name)),
            (Ok(f), Err(StageError::NanOutput)) => assert!(f.data.iter().any(|v| v.is_nan())),
            (Err(_), Err(StageError::Unsupported(_))) => {}
            (cpu, dev) => panic!("{}: cpu {:?} device {:?}", c.name, cpu.is_ok(), dev.err()),
        }
    }
    assert!(n >= 12, "{n} pairs compared");
}

#[test]
fn refusals() {
    let b = backend(0);
    // a case without the 360 flag (a 360 solve is refused: its rules have no kernels yet)
    let cs = cases::corpus(1, 12, 16);
    let c = cs.iter().skip(10).chain(cs.iter()).find(|c| !c.s.wrap && cases::reference(c).is_some()).unwrap();
    let r = cases::reference(c).unwrap();
    let other = flow::build_pyramid(&cases::texture(8, 8, 8, 3, 0.0, 0.0, 0.0, false), &c.s).unwrap();
    assert!(matches!(solve_pyramids_with_host(&b, &r.px, &other, &c.s, SolveExtras::default()), Err(StageError::Unsupported(_))));
    // a user displacement of another level count
    let bad = UserDisplacement { levels: vec![UserLevel::default()] };
    if r.px.geometry().levels != 1 {
        let e = SolveExtras { masks: None, user: Some(&bad) };
        assert!(matches!(solve_pyramids_with_host(&b, &r.px, &r.py, &c.s, e), Err(StageError::Unsupported(_))));
    }
    // an empty user displacement is none
    let empty = UserDisplacement::default();
    let e = SolveExtras { masks: None, user: Some(&empty) };
    let f = solve_pyramids_with_host(&b, &r.px, &r.py, &c.s, e);
    let cpu = extras::solve_pyramids_with(&r.px, &r.py, &c.s, e).unwrap();
    match f {
        Ok(f) => assert_bits(&cpu.data, &f.data, "empty user"),
        Err(StageError::NanOutput) => assert!(cpu.data.iter().any(|v| v.is_nan())),
        Err(e) => panic!("{e}"),
    }
    // images of other sizes
    let x = cases::texture(8, 8, 8, 1, 0.0, 0.0, 0.0, false);
    let y = cases::texture(9, 8, 8, 1, 0.0, 0.0, 0.0, false);
    assert!(matches!(solve_pair_host(&b, &x, &y, &c.s), Err(StageError::Unsupported(_))));
    // a frame the CPU cannot search
    let t = cases::texture(1, 5, 8, 1, 0.0, 0.0, 0.0, false);
    assert!(matches!(solve_host(&b, &t, &t, &c.s), Err(StageError::Unsupported(_))));
}
