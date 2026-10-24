//! Reference comparison harness: render a test set of stills (frame effects) and clips (temporal
//! effects), write the checklist a person follows to render the references with any program,
//! check `vfx` against reference frames and clips, and check the temporal cases for determinism
//! and thread count invariance.

pub mod args;
pub mod cases;
pub mod checklist;
pub mod compare;
pub mod effects;
pub mod run;
pub mod selfcheck;
pub mod sources;
pub mod sweep;
pub mod testgen;
pub mod vfx;
