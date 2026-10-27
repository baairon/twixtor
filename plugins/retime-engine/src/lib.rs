//! The engine shared by the four effects: time model, frame selection, optical flow,
//! warps and compositions, and the glue the front-ends share (see the modules below).
//!
//! Float expressions are evaluated in the order written and in the precision noted on each
//! function, with the rounding of that precision; another order changes the last bits.

pub mod blend;
pub mod compose;
pub mod create_vectors;
pub mod filters;
#[cfg(test)]
mod fixture;
pub mod flow;
pub mod forward;
pub mod frontend;
pub mod holefill;
pub mod host;
pub mod image;
pub mod matte;
pub mod planner;
pub mod prep;
pub mod smartblend;
pub mod timeline;
pub mod vectors;
pub mod warp;

pub use image::{Fmt, Image, Samples, CHANNELS};
