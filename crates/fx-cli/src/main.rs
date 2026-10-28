//! `vfx`: run the effects over raw frames, the same way ffmpeg pipes feed them.

mod apply;
mod cli;
mod clip;
mod devices;
mod error;
mod list;
mod registry;
mod selftest;
mod temporal;

use cli::Command;
use error::CliError;
use std::io::{self, Write};

fn print(text: &str) -> Result<(), CliError> {
    let mut out = io::stdout().lock();
    writeln!(out, "{text}")
        .and_then(|_| out.flush())
        .map_err(|e| CliError::io("write error", e))
}

fn run() -> Result<(), CliError> {
    match cli::parse(std::env::args().skip(1))? {
        Command::Help => print(cli::USAGE),
        Command::Version => print(&format!("vfx {}", env!("CARGO_PKG_VERSION"))),
        Command::List { json, all } => list::run(json, all, &mut io::stdout().lock()),
        Command::Devices { json } => devices::run(json, &mut io::stdout().lock()),
        Command::Apply(args) => apply::run(args),
    }
}

fn main() {
    apply::install_panic_hook();
    let code = match run() {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("vfx: error: {e}");
            e.code()
        }
    };
    std::process::exit(code);
}
