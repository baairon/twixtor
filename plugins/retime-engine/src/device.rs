//! The opt-in OpenCL device backend: its foundation. The CPU code stays the untouched reference;
//! everything here is additive, and a device is used only when it gives the CPU's bytes.
//!
//! * [`cl`]: the runtime loaded at run time, hand written OpenCL 1.2 bindings and owning wrappers
//!   (context, in-order queue, program, kernel, buffer). The only module allowed raw FFI.
//! * [`probe`]: what each device reports, read with info queries only, and the requirements a
//!   device must meet (`FULL_PROFILE`, little endian, `cl_khr_fp64`, the fp config bits).
//! * [`image`]: engine images on the device, upload and download, and the stage error type.
//! * [`kernels`]: OpenCL C sources as string constants, the shared prelude (`FP_CONTRACT OFF`,
//!   range tested truncating conversions, hex float literals) and the build options.
//! * [`sim`]: a work item simulator that runs Rust transcriptions of the kernels in shuffled orders.
//! * [`selfcheck`]: the gate, starter kernels on the device over an adversarial corpus, compared
//!   bit for bit with the CPU functions.
//! * [`progcache`]: the device binaries of built programs, kept per user between runs and keyed
//!   by the device, the driver, the build options and the whole program text.
//!
//! * [`exec`]: the executor that renders `--device gpu`: device selection and the gate at open, the
//!   placement of the stages, per stage CPU fallback, the device frame cache and the run level
//!   cross check.
//!
//! Device bytes equal CPU bytes by construction and by the gates, so nothing here is a parity gap.

#[allow(unsafe_code)]
pub mod cl;
pub mod convstage;
pub mod exec;
pub mod fwdstage;
pub mod image;
pub mod kernels;
pub mod probe;
pub mod progcache;
pub mod selfcheck;
pub mod sim;
pub mod solver;
pub mod vecstage;
pub mod warpstage;
pub mod wbstage;

#[cfg(test)]
mod tests;
