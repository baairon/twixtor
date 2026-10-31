//! The engine shared by the four effects: time model, frame selection, optical flow,
//! warps and compositions, and the glue the front-ends share (see the modules below).
//!
//! Float expressions are evaluated in the order written and in the precision noted on each
//! function, with the rounding of that precision; another order changes the last bits.
//!
//! Raw FFI lives in `device::cl` alone, the hand written OpenCL bindings (the only module the
//! lint below allows).

#![deny(unsafe_code)]

pub mod blend;
pub mod compose;
pub mod create_vectors;
pub mod crt_powf;
pub mod debug;
pub mod device;
pub mod filters;
#[cfg(test)]
mod fixture;
pub mod flow;
pub mod forward;
pub mod frontend;
pub mod gaps;
pub mod geom;
pub mod holefill;
pub mod host;
pub mod image;
pub mod layers;
pub mod matte;
pub mod nanrule;
pub mod planner;
pub mod points;
pub mod prep;
pub mod smartblend;
pub mod stages;
pub mod timeline;
pub mod vectors;
pub mod warp;

pub use image::{Fmt, Image, Samples, CHANNELS};
