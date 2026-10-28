//! The render device a host asks a temporal effect to use (`vfx apply --device`).
//!
//! A request only travels: the host parses it, refuses what it cannot serve and passes the rest in
//! [`crate::TemporalContext::device`], and an effect hands it on to its engine. Pixels never depend
//! on it: a device other than the CPU must give the CPU's bytes or not be used at all. Integer
//! text only, no arithmetic.

use std::fmt;

/// Which device a temporal render runs on. The text form is `cpu`, `gpu` or `gpu=N`
/// ([`DeviceRequest::parse`], [`fmt::Display`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DeviceRequest {
    /// The CPU, the reference path. The default.
    #[default]
    Cpu,
    /// The first usable GPU, whichever the host picks.
    Gpu,
    /// GPU number `N` as the host lists its devices (0 is the first).
    GpuIndex(u32),
}

impl DeviceRequest {
    /// Reads `cpu`, `gpu` or `gpu=N` with `N` written in decimal digits only (no sign), at most
    /// `u32::MAX`. Blanks around the text are ignored; anything else is `None`.
    pub fn parse(text: &str) -> Option<DeviceRequest> {
        match text.trim() {
            "cpu" => Some(DeviceRequest::Cpu),
            "gpu" => Some(DeviceRequest::Gpu),
            other => {
                let n = other.strip_prefix("gpu=")?;
                if n.is_empty() || !n.bytes().all(|b| b.is_ascii_digit()) {
                    return None;
                }
                n.parse().ok().map(DeviceRequest::GpuIndex)
            }
        }
    }
}

impl fmt::Display for DeviceRequest {
    /// `cpu`, `gpu` or `gpu=N`: the form [`DeviceRequest::parse`] reads back.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceRequest::Cpu => write!(f, "cpu"),
            DeviceRequest::Gpu => write!(f, "gpu"),
            DeviceRequest::GpuIndex(n) => write!(f, "gpu={n}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_the_cpu() {
        assert_eq!(DeviceRequest::default(), DeviceRequest::Cpu);
    }

    #[test]
    fn parses_the_three_forms() {
        assert_eq!(DeviceRequest::parse("cpu"), Some(DeviceRequest::Cpu));
        assert_eq!(DeviceRequest::parse("gpu"), Some(DeviceRequest::Gpu));
        assert_eq!(DeviceRequest::parse("gpu=0"), Some(DeviceRequest::GpuIndex(0)));
        assert_eq!(DeviceRequest::parse(" gpu=12 "), Some(DeviceRequest::GpuIndex(12)));
        assert_eq!(DeviceRequest::parse("gpu=007"), Some(DeviceRequest::GpuIndex(7)));
        assert_eq!(DeviceRequest::parse("gpu=4294967295"), Some(DeviceRequest::GpuIndex(u32::MAX)));
        for bad in ["", "CPU", "Gpu", "gpu=", "gpu=-1", "gpu=+1", "gpu=1.5", "gpu=x", "gpu=4294967296", "gpu 1", "gpu==1", "opencl", "cpu=0"] {
            assert_eq!(DeviceRequest::parse(bad), None, "`{bad}` should not parse");
        }
    }

    #[test]
    fn display_is_the_form_parse_reads() {
        for d in [DeviceRequest::Cpu, DeviceRequest::Gpu, DeviceRequest::GpuIndex(0), DeviceRequest::GpuIndex(3), DeviceRequest::GpuIndex(u32::MAX)] {
            assert_eq!(DeviceRequest::parse(&d.to_string()), Some(d));
        }
        assert_eq!(DeviceRequest::GpuIndex(2).to_string(), "gpu=2");
    }
}
