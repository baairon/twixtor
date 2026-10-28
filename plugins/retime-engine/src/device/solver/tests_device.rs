//! Device tests of the solver on the OpenCL devices of the machine: whole solves compared with
//! the CPU bit for bit over a seeded corpus (translating, rotating and still textures at 8, 16
//! and 32 bit, Sloppy to full resolution, draft, sides 2 to 640, masks, points, wrap, large
//! motion, NaN and infinite pixels), at least 200 solves per device. Where the CPU flow holds a
//! NaN the device must refuse with `StageError::NanOutput`; anywhere else it must give the CPU's
//! bits, and a difference is reported with the first failing level and pixel of the per level
//! traces. With no OpenCL runtime or no usable device every test prints a note and passes. The
//! tests take the devices one at a time. `solver_timings_at_1080p` (ignored) prints `solve_pair`
//! at 1920x1080 per quality, device (resident and with transfers) against the CPU.

use super::cases::{self, Case};
use crate::flow::extras;
use super::*;
use crate::device::cl;
use crate::device::probe;
use crate::device::sim::Rng;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());
static SESSIONS: OnceLock<Vec<(String, Session)>> = OnceLock::new();

/// One session per usable device, the solver program built once.
fn sessions() -> &'static [(String, Session)] {
    SESSIONS.get_or_init(|| {
        let api = match cl::runtime() {
            Ok(api) => api,
            Err(e) => {
                println!("device solver: {e}");
                return Vec::new();
            }
        };
        let list = match probe::devices(api) {
            Ok((list, _)) => list,
            Err(e) => {
                println!("device solver: {e}");
                return Vec::new();
            }
        };
        let source = program();
        let mut out = Vec::new();
        for (device, info) in list {
            let name = format!("gpu={} {}", info.index, info.display_name());
            if !probe::usable(&info) {
                println!("{name}: not usable: {:?}", probe::requirements(&info));
                continue;
            }
            match Session::open(&device, &source) {
                Ok(s) => out.push((name, s)),
                Err(e) => panic!("{name}: the solver program does not build: {e}"),
            }
        }
        out
    })
}

fn on_devices(test: &str, f: impl Fn(&str, &Session) -> usize, min: usize) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let list = sessions();
    if list.is_empty() {
        println!("{test}: skipped, no usable OpenCL device");
        return;
    }
    for (name, s) in list {
        let t = Instant::now();
        let n = f(name, s);
        println!("{test}: {name}: {n} solves compared bit for bit in {:.2} s", t.elapsed().as_secs_f64());
        assert!(n >= min, "{test}: {name}: only {n} solves compared");
    }
}

/// What one case gave on the device.
enum Outcome {
    /// The CPU refuses the case.
    Skipped,
    /// The CPU's bits.
    Same,
    /// The CPU flow holds a NaN and the device refused with `NanOutput`.
    NanRefused,
}

/// One case on the device against the CPU: the outcome, or the difference with the first failing
/// level and pixel of the per level traces.
fn run_case(s: &Session, c: &Case) -> Result<Outcome, String> {
    let Some(r) = cases::reference(c) else { return Ok(Outcome::Skipped) };
    let got = solve_pyramids_with_gpu(s, &r.px, &r.py, &c.s, r.extras());
    let cpu_nan = r.flow.data.iter().any(|v| v.is_nan());
    let fail = match &got {
        Ok(f) if !cpu_nan && f.data.iter().zip(&r.flow.data).all(|(a, b)| a.to_bits() == b.to_bits()) => return Ok(Outcome::Same),
        Err(StageError::NanOutput) if cpu_nan => return Ok(Outcome::NanRefused),
        Ok(_) if cpu_nan => "the CPU flow holds a NaN and the device returned a flow without one".to_string(),
        Ok(_) => "the flows differ".to_string(),
        Err(e) => format!("refused: {e}"),
    };
    let mut dev_trace = Vec::new();
    let _ = solve_pyramids_with_traced(s, &r.px, &r.py, &c.s, r.extras(), &mut |l, buf| {
        dev_trace.push((l, s.download(buf)?));
        Ok(())
    });
    let cpu_trace = cases::cpu_trace(&r, &c.s);
    let at = cases::first_trace_difference(&cpu_trace, &dev_trace, &r.px.geometry().sizes).unwrap_or_else(|| "the traces agree".into());
    Err(format!("{}: {fail}; {at}", c.name))
}

/// [`run_case`] that panics on a difference; true when compared.
fn check_case(name: &str, s: &Session, c: &Case) -> bool {
    match run_case(s, c) {
        Ok(Outcome::Skipped) => false,
        Ok(_) => true,
        Err(e) => panic!("{name}: {e}"),
    }
}

#[test]
fn device_solves_match_the_cpu_over_the_corpus() {
    on_devices(
        "device_solves_match_the_cpu_over_the_corpus",
        |name, s| {
            let (mut n, mut moving) = (0, 0);
            for c in cases::corpus(0x50_1E, 230, 640) {
                if check_case(name, s, &c) {
                    n += 1;
                    if cases::reference(&c).is_some_and(|r| r.flow.data.iter().any(|v| *v != 0.0 && !v.is_nan())) {
                        moving += 1;
                    }
                }
            }
            let nan = cases::corpus(0x50_1E, 230, 640).iter().filter(|c| matches!(run_case(s, c), Ok(Outcome::NanRefused))).count();
            println!("{name}: {moving} of {n} compared flows are not zero, {nan} refused for a NaN as the CPU flow holds one");
            assert!(moving * 2 >= n, "{name}: a weak corpus ({moving} of {n} flows not zero)");
            n
        },
        200,
    );
}

#[test]
fn device_solve_and_solve_pair_match_the_cpu() {
    on_devices(
        "device_solve_and_solve_pair_match_the_cpu",
        |name, s| {
            let mut n = 0;
            for c in cases::corpus(0x9A1, 40, 160) {
                let cpu = flow::solve_pair(&c.a, &c.b, &c.s);
                let dev = solve_pair_gpu(s, &c.a, &c.b, &c.s);
                match (cpu, dev) {
                    (Ok(p), Ok(d)) => {
                        assert!(
                            p.f_ab.data.iter().zip(&d.f_ab.data).all(|(a, b)| a.to_bits() == b.to_bits())
                                && p.f_ba.data.iter().zip(&d.f_ba.data).all(|(a, b)| a.to_bits() == b.to_bits()),
                            "{name}: {}: solve_pair differs",
                            c.name
                        );
                        n += 1;
                    }
                    (Ok(p), Err(StageError::NanOutput)) => {
                        assert!(p.f_ab.data.iter().chain(&p.f_ba.data).any(|v| v.is_nan()), "{name}: {}: a NaN refusal without a NaN", c.name)
                    }
                    (Err(_), Err(StageError::Unsupported(_))) => {}
                    (cpu, dev) => panic!("{name}: {}: cpu ok {} device {:?}", c.name, cpu.is_ok(), dev.err()),
                }
                if let (Ok(f), Ok(d)) = (flow::solve(&c.a, &c.b, &c.s), solve_gpu(s, &c.a, &c.b, &c.s)) {
                    assert!(f.data.iter().zip(&d.data).all(|(a, b)| a.to_bits() == b.to_bits()), "{name}: {}: solve differs", c.name);
                    n += 1;
                }
            }
            n
        },
        40,
    );
}

#[test]
fn device_resident_pyramids_from_device_images_match_the_cpu() {
    on_devices(
        "device_resident_pyramids_from_device_images_match_the_cpu",
        |name, s| {
            let mut n = 0;
            for c in cases::corpus(0x7E5, 20, 96) {
                let Ok(cpu) = flow::solve(&c.a, &c.b, &c.s) else { continue };
                let taps5 = Taps::binomial5(s).unwrap();
                let (da, db) = (DeviceImage::upload(s, &c.a).unwrap(), DeviceImage::upload(s, &c.b).unwrap());
                let pa = build_pyramid_enqueue(s, &da, &c.s, &taps5).unwrap();
                let pb = build_pyramid_enqueue(s, &db, &c.s, &taps5).unwrap();
                let buf = solve_pyramids_with_enqueue(s, &pa, &pb, &c.s, &DeviceExtras::default()).unwrap();
                match download_flow(s, &buf, &pa.geometry) {
                    Ok(f) => assert!(f.data.iter().zip(&cpu.data).all(|(a, b)| a.to_bits() == b.to_bits()), "{name}: {}", c.name),
                    Err(StageError::NanOutput) => assert!(cpu.data.iter().any(|v| v.is_nan())),
                    Err(e) => panic!("{name}: {}: {e}", c.name),
                }
                n += 1;
            }
            n
        },
        15,
    );
}

#[test]
fn a_nan_flow_is_refused_and_a_finite_spot_is_exact() {
    on_devices(
        "a_nan_flow_is_refused_and_a_finite_spot_is_exact",
        |name, s| {
            let mut n = 0;
            for c in cases::corpus(0x4A4, 24, 160) {
                let Some(r) = cases::reference(&c) else { continue };
                for v in [f32::NAN, 2.5f32, -1.0e30] {
                    let u = cases::spot_user(r.px.geometry(), v);
                    let e = SolveExtras { masks: r.masks.as_ref().map(|(a, b)| (a, b)), user: Some(&u) };
                    let cpu = extras::solve_pyramids_with(&r.px, &r.py, &c.s, e).unwrap();
                    let dev = solve_pyramids_with_gpu(s, &r.px, &r.py, &c.s, e);
                    if cpu.data.iter().any(|x| x.is_nan()) {
                        assert!(matches!(dev, Err(StageError::NanOutput)), "{name}: {} v {v}: a NaN flow not refused", c.name);
                    } else {
                        let d = dev.unwrap_or_else(|e| panic!("{name}: {} v {v}: {e}", c.name));
                        assert!(d.data.iter().zip(&cpu.data).all(|(a, b)| a.to_bits() == b.to_bits()), "{name}: {} v {v}: differs", c.name);
                    }
                    n += 1;
                }
            }
            n
        },
        60,
    );
}

/// The corpus check must catch a solver program that differs from the CPU in one constant or one
/// comparison: each mutation of the source gives at least one reported difference.
#[test]
fn a_mutated_solver_program_is_caught() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let Ok(api) = cl::runtime() else {
        println!("a_mutated_solver_program_is_caught: skipped, no OpenCL runtime");
        return;
    };
    let Ok((list, _)) = probe::devices(api) else { return };
    let Some((device, _)) = list.into_iter().find(|(_, info)| probe::usable(info)) else {
        println!("a_mutated_solver_program_is_caught: skipped, no usable OpenCL device");
        return;
    };
    let mutations = [
        ("update threshold 40 to 41", "fabs(t) > 0x1.4p+5f", "fabs(t) > 0x1.48p+5f"),
        ("warped mask 0.999 to 0.99", ">= 0x1.ff7ceep-1f) {
                s = rs_bilinear1", ">= 0x1.fae148p-1f) {
                s = rs_bilinear1"),
        ("sweep 1 neighbour", "int j = dir == 0 ? i - 1 : i + 1;", "int j = dir == 0 ? i + 1 : i + 1;"),
        ("gradient rounding", "float gy = ny / dy;", "float gy = ny / dy * 0x1.000002p+0f;"),
    ];
    let cs = cases::corpus(0x50_1E, 120, 160);
    for (what, from, to) in mutations {
        let src = program();
        assert!(src.contains(from), "{what}: the source no longer holds the mutated text");
        let s = Session::open(&device, &src.replacen(from, to, 1)).unwrap();
        let first = cs.iter().find_map(|c| run_case(&s, c).err());
        match first {
            Some(e) => println!("mutation {what}: caught: {e}"),
            None => panic!("mutation {what}: no case of the corpus differs"),
        }
    }
}

/// Median of `k` timed runs of `f`, in milliseconds.
fn time_ms(k: usize, mut f: impl FnMut()) -> f64 {
    let mut v: Vec<f64> = (0..k)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[k / 2]
}

#[test]
#[ignore]
fn solver_timings_at_1080p() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let list = sessions();
    if list.is_empty() {
        println!("solver timings: skipped, no usable OpenCL device");
        return;
    }
    let (w, h) = (1920, 1080);
    let a = cases::texture(w, h, 8, 77, 0.0, 0.0, 0.0, false);
    let b = cases::texture(w, h, 8, 77, 2.5, -1.25, 0.01, false);
    let qualities = [("best", (4u8, true)), ("high", (3, true)), ("draft", (4, false)), ("full", (5, true))];
    for (qn, q) in qualities {
        let st = cases::settings(q, 0.7, false);
        let cpu_pair = flow::solve_pair(&a, &b, &st).unwrap();
        let cpu = time_ms(3, || {
            flow::solve_pair(&a, &b, &st).unwrap();
        });
        for (name, s) in list {
            // warm up and check
            let d = solve_pair_gpu(s, &a, &b, &st).unwrap();
            assert!(d.f_ab.data.iter().zip(&cpu_pair.f_ab.data).all(|(x, y)| x.to_bits() == y.to_bits()), "{name}: {qn}: differs");
            let with = time_ms(3, || {
                solve_pair_gpu(s, &a, &b, &st).unwrap();
            });
            let taps5 = Taps::binomial5(s).unwrap();
            let (da, db) = (DeviceImage::upload(s, &a).unwrap(), DeviceImage::upload(s, &b).unwrap());
            let pa = build_pyramid_enqueue(s, &da, &st, &taps5).unwrap();
            let pb = build_pyramid_enqueue(s, &db, &st, &taps5).unwrap();
            s.queue.finish().unwrap();
            let none = DeviceExtras::default();
            let resident = time_ms(3, || {
                let f1 = solve_pyramids_with_enqueue(s, &pb, &pa, &st, &none).unwrap();
                let f2 = solve_pyramids_with_enqueue(s, &pa, &pb, &st, &none).unwrap();
                s.queue.finish().unwrap();
                drop((f1, f2));
            });
            let build = time_ms(3, || {
                let p1 = build_pyramid_enqueue(s, &da, &st, &taps5).unwrap();
                let p2 = build_pyramid_enqueue(s, &db, &st, &taps5).unwrap();
                s.queue.finish().unwrap();
                drop((p1, p2));
            });
            println!(
                "solve_pair 1920x1080 {qn} (working {}x{}): {name}: resident pyramids {resident:.1} ms (+ pyramid build {build:.1} ms), with transfers {with:.1} ms; CPU {cpu:.1} ms",
                pa.geometry.width, pa.geometry.height
            );
        }
    }
    let _ = Rng::new(0);
}
