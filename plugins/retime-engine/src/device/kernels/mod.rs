//! Device kernel sources: OpenCL C 1.2 as Rust string constants, the shared [`prelude`], the
//! build options and the registry the source lint walks.
//!
//! Rules for every source (checked by the lint in `device/tests.rs` over [`SOURCES`]): the
//! program starts with the prelude, whose first line is `#pragma OPENCL FP_CONTRACT OFF`, and no
//! other source mentions `FP_CONTRACT`; none of `mad fma native_ half_ pow log exp fmin fmax mix`
//! is used (their results are not specified to the bit, or they fuse); float literals are
//! hexadecimal only; every loop has a constant bound. A kernel reproduces the operation order and
//! precision of the CPU function it ports, and the module that defines it states that contract
//! with V / I / U tags. Every kernel source under `device/kernels/` is listed in [`SOURCES`].

pub mod conv;
pub mod fwdstage;
pub mod holefill;
pub mod prelude;
pub mod pyramid;
pub mod resample;
pub mod solver;
pub mod starter;
pub mod vecstage;
pub mod warpstage;
pub mod wbstage;

/// The build options of every program: exactly these, never `-cl-fast-relaxed-math`,
/// `-cl-mad-enable` or any other option that relaxes IEEE 754 behaviour.
pub const BUILD_OPTIONS: &str = "-cl-std=CL1.2 -cl-fp32-correctly-rounded-divide-sqrt";

/// Every kernel source of the tree as `(module::CONSTANT, text)`. The lint checks each one, and a
/// test fails when a `pub const ...: &str` under `device/kernels/` is missing here.
pub const SOURCES: &[(&str, &str)] = &[
    ("prelude::PRELUDE", prelude::PRELUDE),
    ("starter::COPY", starter::COPY),
    ("starter::BLEND", starter::BLEND),
    ("starter::CONVERT", starter::CONVERT),
    ("starter::INDEX", starter::INDEX),
    ("starter::CHAIN", starter::CHAIN),
    ("conv::CV_WRAPX_TAP", conv::CV_WRAPX_TAP),
    ("conv::CONV", conv::CONV),
    ("conv::CHOKE", conv::CHOKE),
    ("resample::SAMPLERS", resample::SAMPLERS),
    ("resample::RESAMPLE", resample::RESAMPLE),
    ("pyramid::PYRAMID", pyramid::PYRAMID),
    ("holefill::HOLEFILL", holefill::HOLEFILL),
    ("solver::SOLVER", solver::SOLVER),
    ("vecstage::VECTORS", vecstage::VECTORS),
    ("vecstage::MATTE", vecstage::MATTE),
    ("vecstage::PREP", vecstage::PREP),
    ("vecstage::OVERLAY", vecstage::OVERLAY),
    ("vecstage::GEOM", vecstage::GEOM),
    ("warpstage::BLEND", warpstage::BLEND),
    ("warpstage::WARP", warpstage::WARP),
    ("warpstage::BLURCOMP", warpstage::BLURCOMP),
    ("warpstage::SMARTBLEND", warpstage::SMARTBLEND),
    ("wbstage::WB", wbstage::WB),
    ("fwdstage::FORWARD", fwdstage::FORWARD),
];

/// One program text: the prelude, then `parts` in order.
pub fn program(parts: &[&str]) -> String {
    let mut text = String::with_capacity(prelude::PRELUDE.len() + parts.iter().map(|p| p.len() + 1).sum::<usize>());
    text.push_str(prelude::PRELUDE);
    for p in parts {
        text.push('\n');
        text.push_str(p);
    }
    text
}

/// The program of the self-check: the prelude and every starter kernel.
pub fn starter_program() -> String {
    program(&[starter::COPY, starter::BLEND, starter::CONVERT, starter::INDEX, starter::CHAIN])
}
