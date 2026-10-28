//! `fx-parity`: build the test set, write the checklist for rendering its references, check
//! `vfx` against reference frames and clips, and check the temporal cases for determinism.

use fx_parity::args::{self, ChecklistArgs, Command, DevcheckArgs, GenArgs, RunArgs, SelfcheckArgs};
use fx_parity::cases::Plan;
use fx_parity::{checklist, devcheck, families, run, selfcheck, sources, testgen};
use std::io::Write;
use std::process::ExitCode;

const MISMATCH: u8 = 1;
const USAGE: u8 = 2;
const ERROR: u8 = 3;

/// Cases picked by `selfcheck` when --max-cases is not given.
const SELFCHECK_CASES: usize = 24;
/// Thread counts compared with the default one when --threads is not given.
const SELFCHECK_THREADS: [usize; 2] = [1, 3];

fn default_jobs() -> usize {
    std::thread::available_parallelism().map_or(2, |n| n.get()).min(8)
}

fn cmd_gen(a: GenArgs) -> Result<u8, String> {
    let mut plan = Plan { sweep: !a.no_sweep, ..Plan::default() };
    if let Some(sizes) = a.sizes {
        plan.temporal.sizes = sizes.clone();
        plan.sizes = sizes;
    }
    if let Some(depths) = a.depths {
        plan.depths = depths;
    }
    if let Some(names) = a.sources {
        plan.sources = names.iter().filter_map(|n| sources::find(n)).collect();
    }
    if let Some(names) = a.sweep_sources {
        plan.sweep_sources = names.iter().filter_map(|n| sources::find(n)).collect();
    }
    if let Some(frames) = a.frames {
        plan.temporal.frames = frames;
    }
    if let Some(rates) = a.fps {
        plan.temporal.fps = rates;
    }
    if let Some(names) = a.clip_sources {
        plan.temporal.sources = names.iter().filter_map(|n| sources::find_clip(n)).collect();
    }
    if let Some(names) = a.clip_sweep_sources {
        plan.temporal.sweep_sources = names.iter().filter_map(|n| sources::find_clip(n)).collect();
    }
    if let Some(cap) = a.max_cases_per_effect {
        plan.temporal.max_cases_per_effect = cap;
    }
    let opts = testgen::GenOptions {
        out: a.out,
        vfx: a.vfx,
        effects: a.effects,
        plan,
        jobs: a.jobs.unwrap_or_else(default_jobs),
        families: a.families,
    };
    let s = testgen::generate(&opts, &|m| eprintln!("fx-parity: {m}"))?;
    println!("wrote {}: {} cases for {} effects, {} inputs", s.manifest.display(), s.cases, s.effects, s.inputs);
    if let Some(counts) = &s.families {
        println!("{}", families::summary(counts));
    }
    Ok(0)
}

fn cmd_run(a: RunArgs) -> Result<u8, String> {
    let opts = run::RunOptions {
        testset: a.testset,
        refs: a.refs,
        effect: a.effect,
        vfx: a.vfx,
        jobs: a.jobs.unwrap_or_else(default_jobs),
    };
    let results = run::run(&opts, &|m| eprintln!("fx-parity: warning: {m}"))?;
    let mut out = std::io::stdout().lock();
    let written = if a.json {
        let v = run::to_json(&results, a.brief);
        writeln!(out, "{}", serde_json::to_string_pretty(&v).expect("report serializes"))
    } else {
        run::write_human(&results, a.brief, &mut out)
    };
    written.map_err(|e| format!("cannot write the report: {e}"))?;
    let t = run::totals(&results);
    let bad = t.fail > 0 || (a.strict && t.no_reference > 0);
    Ok(if bad { MISMATCH } else { 0 })
}

fn cmd_selfcheck(a: SelfcheckArgs) -> Result<u8, String> {
    let opts = selfcheck::SelfcheckOptions {
        testset: a.testset,
        effect: a.effect,
        max_cases: a.max_cases.unwrap_or(SELFCHECK_CASES),
        threads: a.threads.unwrap_or_else(|| SELFCHECK_THREADS.to_vec()),
        vfx: a.vfx,
        jobs: a.jobs.unwrap_or_else(default_jobs),
    };
    let checks = selfcheck::run(&opts)?;
    let mut out = std::io::stdout().lock();
    let written = if a.json {
        writeln!(out, "{}", serde_json::to_string_pretty(&selfcheck::to_json(&checks)).expect("report serializes"))
    } else {
        selfcheck::write_human(&checks, &mut out)
    };
    written.map_err(|e| format!("cannot write the report: {e}"))?;
    Ok(if selfcheck::all_ok(&checks) { 0 } else { MISMATCH })
}

fn cmd_devcheck(a: DevcheckArgs) -> Result<u8, String> {
    let opts = devcheck::DevcheckOptions { testsets: a.testsets, effect: a.effect, max_cases: a.max_cases.unwrap_or(0), devices: a.devices, vfx: a.vfx };
    let (devices, results) = devcheck::run(&opts, &|m| eprintln!("fx-parity: {m}"))?;
    let mut out = std::io::stdout().lock();
    let written = if a.json {
        writeln!(out, "{}", serde_json::to_string_pretty(&devcheck::to_json(&devices, &results)).expect("report serializes"))
    } else {
        devcheck::write_human(&devices, &results, &mut out)
    };
    written.map_err(|e| format!("cannot write the report: {e}"))?;
    Ok(if devcheck::all_identical(&results) { 0 } else { MISMATCH })
}

fn cmd_checklist(a: ChecklistArgs) -> Result<u8, String> {
    let opts = checklist::ChecklistOptions { testset: a.testset, out: a.out, vfx: a.vfx };
    let s = checklist::generate(&opts, &|m| eprintln!("fx-parity: warning: {m}"))?;
    println!("wrote {}: {} cases for {} effects", s.path.display(), s.cases, s.effects);
    Ok(0)
}

fn main() -> ExitCode {
    let command = match args::parse(std::env::args().skip(1)) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("fx-parity: error: {msg}");
            return ExitCode::from(USAGE);
        }
    };
    let result = match command {
        Command::Help => {
            println!("{}", args::USAGE);
            Ok(0)
        }
        Command::Gen(a) => cmd_gen(a),
        Command::Run(a) => cmd_run(a),
        Command::Selfcheck(a) => cmd_selfcheck(a),
        Command::Checklist(a) => cmd_checklist(a),
        Command::Devcheck(a) => cmd_devcheck(a),
    };
    match result {
        Ok(code) => ExitCode::from(code),
        Err(msg) => {
            eprintln!("fx-parity: error: {}", msg.split_whitespace().collect::<Vec<_>>().join(" "));
            ExitCode::from(ERROR)
        }
    }
}
