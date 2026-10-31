//! `VFX_CALL_RECORD=<file>`: a JSON lines log of what a temporal render asked of its clips.
//!
//! One line per event, written as the render runs and never read back, so the output bytes are
//! the ones of a run without it:
//! - `run`: once at the start, the effect, size, depth, rate, time base, frame count and the
//!   `--param` values as given.
//! - `frame`: before each output frame, `out_frame` and its `time` (`out_frame * time_step`) in
//!   ticks, with `time_step` and `time_scale`.
//! - `read`: each call of `frame_at` on the input or a layer: `layer` (`input` or the layer id),
//!   the `time` asked in ticks, `index` (the clip frame that holds it, `null` outside the clip),
//!   `clip_frames` and the `out_frame` that asked.
//!
//! A frame the device cross check renders twice (`--device gpu`, the first frame that uses the
//! device) logs its reads twice.
//!
//! Parameters are sampled inside the effect (`ParamCurves`), not through `vfx`, so their sample
//! times are not recorded: a host compares the `frame` times and the `run` params instead.

use crate::cli::ApplyArgs;
use crate::error::CliError;
use crate::temporal::rate_text;
use fx_core::{Rational, TimeBase};
use serde_json::{json, Value};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// The environment variable that turns the log on.
pub const ENV: &str = "VFX_CALL_RECORD";

pub struct CallRecord {
    out: Mutex<BufWriter<File>>,
    frame: AtomicU64,
    failed: AtomicBool,
}

impl CallRecord {
    /// The log named by `VFX_CALL_RECORD`, created (or emptied) now; `None` when it is unset or empty.
    pub fn from_env() -> Result<Option<Arc<CallRecord>>, CliError> {
        let Some(path) = std::env::var_os(ENV).filter(|p| !p.is_empty()) else { return Ok(None) };
        let file = File::create(&path).map_err(|e| CliError::io(format!("cannot create {ENV} file {}", path.to_string_lossy()), e))?;
        Ok(Some(Arc::new(CallRecord { out: Mutex::new(BufWriter::new(file)), frame: AtomicU64::new(0), failed: AtomicBool::new(false) })))
    }

    fn line(&self, v: Value) {
        let mut out = self.out.lock().unwrap_or_else(|p| p.into_inner());
        if writeln!(out, "{v}").is_err() {
            self.failed.store(true, Ordering::SeqCst);
        }
    }

    /// The start of the run.
    pub fn run(&self, args: &ApplyArgs, fps: Rational, time: TimeBase, frames: u64) {
        self.line(json!({
            "ev": "run",
            "effect": args.effect,
            "width": args.width,
            "height": args.height,
            "depth": args.depth.bits(),
            "fps": rate_text(fps),
            "time_scale": time.time_scale,
            "time_step": time.time_step,
            "frames": frames,
            "params": args.params.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>(),
        }));
    }

    /// The output frame `n` is about to render; the reads that follow are its own.
    pub fn begin_frame(&self, n: u64, time: TimeBase) {
        self.frame.store(n, Ordering::SeqCst);
        self.line(json!({
            "ev": "frame",
            "out_frame": n,
            "time": n as i64 * time.time_step as i64,
            "time_step": time.time_step,
            "time_scale": time.time_scale,
        }));
    }

    /// One `frame_at` call on `layer` at `time` ticks, which landed on clip frame `index`.
    pub fn read(&self, layer: &str, time: i64, index: Option<u64>, clip_frames: u64) {
        self.line(json!({
            "ev": "read",
            "out_frame": self.frame.load(Ordering::SeqCst),
            "layer": layer,
            "time": time,
            "index": index,
            "clip_frames": clip_frames,
        }));
    }

    /// Flushes the log; a line that could not be written makes the run an io error, because a
    /// log with holes is worse than none.
    pub fn finish(&self) -> Result<(), CliError> {
        let flushed = self.out.lock().unwrap_or_else(|p| p.into_inner()).flush();
        match flushed {
            Ok(()) if !self.failed.load(Ordering::SeqCst) => Ok(()),
            Ok(()) => Err(CliError::Io(format!("cannot write the {ENV} file"))),
            Err(e) => Err(CliError::io(format!("cannot write the {ENV} file"), e)),
        }
    }
}
