//! Device probing: what the OpenCL runtime reports about each platform and device, read with
//! info queries only (no program is built and no kernel runs), and the requirements a device must
//! meet before the self-check may run on it.
//!
//! Devices are numbered across platforms in the runtime's order, platform by platform and device
//! by device: that number is the `N` of `--device gpu=N`. No arithmetic beyond integer
//! comparisons.

use crate::device::cl::{self, ClError};

/// What one device reports. Built by [`read`] from a live device or written by hand in tests.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Position across every platform: the `N` of `gpu=N`.
    pub index: usize,
    pub platform: String,
    pub platform_version: String,
    pub name: String,
    /// The board name an AMD device reports (`cl_amd_device_attribute_query`), else empty.
    pub board: String,
    pub vendor: String,
    /// `CL_DEVICE_VERSION`, for example `OpenCL 3.0 CUDA`.
    pub version: String,
    /// `CL_DRIVER_VERSION`.
    pub driver: String,
    /// `CL_DEVICE_OPENCL_C_VERSION`, for example `OpenCL C 1.2 `.
    pub c_version: String,
    /// `FULL_PROFILE` or `EMBEDDED_PROFILE`.
    pub profile: String,
    /// `CL_DEVICE_TYPE` bits.
    pub device_type: u64,
    pub little_endian: bool,
    pub available: bool,
    pub compiler_available: bool,
    /// Bytes.
    pub global_memory: u64,
    /// Bytes, the largest single buffer.
    pub max_allocation: u64,
    /// Memory shared with the host (an integrated GPU).
    pub unified_memory: bool,
    pub compute_units: u32,
    pub extensions: Vec<String>,
    /// `CL_DEVICE_SINGLE_FP_CONFIG` bits.
    pub single_fp: u64,
    /// `CL_DEVICE_DOUBLE_FP_CONFIG` bits (0 without double precision).
    pub double_fp: u64,
    /// Queries that failed, as text; any entry refuses the device.
    pub errors: Vec<String>,
}

impl DeviceInfo {
    /// The board name when the device has one, else its name.
    pub fn display_name(&self) -> &str {
        if self.board.is_empty() {
            &self.name
        } else {
            &self.board
        }
    }

    pub fn has_extension(&self, name: &str) -> bool {
        self.extensions.iter().any(|e| e == name)
    }

    /// `gpu`, `cpu`, `accelerator`, `custom` or `other` from the type bits.
    pub fn type_name(&self) -> &'static str {
        if self.device_type & cl::CL_DEVICE_TYPE_GPU != 0 {
            "gpu"
        } else if self.device_type & cl::CL_DEVICE_TYPE_CPU != 0 {
            "cpu"
        } else if self.device_type & cl::CL_DEVICE_TYPE_ACCELERATOR != 0 {
            "accelerator"
        } else if self.device_type & cl::CL_DEVICE_TYPE_CUSTOM != 0 {
            "custom"
        } else {
            "other"
        }
    }
}

/// Every device of the runtime with what it reports, plus platforms whose device list failed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Probe {
    pub devices: Vec<DeviceInfo>,
    /// One line per platform that could not be listed.
    pub errors: Vec<String>,
}

/// The fp config bits by name, in bit order.
pub const FP_FLAGS: &[(u64, &str)] = &[
    (cl::CL_FP_DENORM, "DENORM"),
    (cl::CL_FP_INF_NAN, "INF_NAN"),
    (cl::CL_FP_ROUND_TO_NEAREST, "ROUND_TO_NEAREST"),
    (cl::CL_FP_ROUND_TO_ZERO, "ROUND_TO_ZERO"),
    (cl::CL_FP_ROUND_TO_INF, "ROUND_TO_INF"),
    (cl::CL_FP_FMA, "FMA"),
    (cl::CL_FP_SOFT_FLOAT, "SOFT_FLOAT"),
    (cl::CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT, "CORRECTLY_ROUNDED_DIVIDE_SQRT"),
];

/// The names of the bits set in `bits`, plus `0x...` for bits no name covers.
pub fn fp_flags(bits: u64) -> Vec<String> {
    let mut out: Vec<String> = FP_FLAGS.iter().filter(|(b, _)| bits & b != 0).map(|(_, n)| n.to_string()).collect();
    let known = FP_FLAGS.iter().fold(0u64, |acc, (b, _)| acc | b);
    if bits & !known != 0 {
        out.push(format!("0x{:x}", bits & !known));
    }
    out
}

/// Single precision bits a device must report.
pub const SINGLE_REQUIRED: &[(u64, &str)] = &[
    (cl::CL_FP_DENORM, "DENORM"),
    (cl::CL_FP_INF_NAN, "INF_NAN"),
    (cl::CL_FP_ROUND_TO_NEAREST, "ROUND_TO_NEAREST"),
    (cl::CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT, "CORRECTLY_ROUNDED_DIVIDE_SQRT"),
];

/// Double precision bits a device must report.
pub const DOUBLE_REQUIRED: &[(u64, &str)] =
    &[(cl::CL_FP_DENORM, "DENORM"), (cl::CL_FP_INF_NAN, "INF_NAN"), (cl::CL_FP_ROUND_TO_NEAREST, "ROUND_TO_NEAREST")];

/// `major.minor` after `prefix` at the start of `text` (`OpenCL 3.0 CUDA` with prefix `OpenCL `
/// gives `(3, 0)`); `None` when the text has another shape.
pub fn parse_version(text: &str, prefix: &str) -> Option<(u32, u32)> {
    let rest = text.trim_start().strip_prefix(prefix)?;
    let word = rest.split_whitespace().next()?;
    let (major, minor) = word.split_once('.')?;
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if !digits(major) || !digits(minor) {
        return None;
    }
    Some((major.parse().ok()?, minor.parse().ok()?))
}

/// The reasons `info` cannot be used, empty when it can: `FULL_PROFILE`, little
/// endian, `cl_khr_fp64`, single precision DENORM + INF_NAN + ROUND_TO_NEAREST +
/// CORRECTLY_ROUNDED_DIVIDE_SQRT, double precision DENORM + INF_NAN + ROUND_TO_NEAREST. The build
/// options add OpenCL C 1.2 or later, an available device and an available compiler; a failed
/// query refuses the device too.
pub fn requirements(info: &DeviceInfo) -> Vec<String> {
    let mut out = Vec::new();
    for e in &info.errors {
        out.push(format!("probe failed: {e}"));
    }
    if !info.available {
        out.push("not available".into());
    }
    if !info.compiler_available {
        out.push("no compiler".into());
    }
    if info.profile != "FULL_PROFILE" {
        let shown = if info.profile.is_empty() { "no profile" } else { info.profile.as_str() };
        out.push(format!("{shown}, needs FULL_PROFILE"));
    }
    if !info.little_endian {
        out.push("big endian".into());
    }
    match parse_version(&info.c_version, "OpenCL C ") {
        Some(v) if v >= (1, 2) => {}
        Some((major, minor)) => out.push(format!("OpenCL C {major}.{minor}, needs 1.2 or later")),
        None => out.push(format!("unreadable OpenCL C version `{}`", info.c_version)),
    }
    if !info.has_extension("cl_khr_fp64") {
        out.push("no cl_khr_fp64 (double precision)".into());
    }
    for (bit, name) in SINGLE_REQUIRED {
        if info.single_fp & bit == 0 {
            out.push(format!("single precision without {name}"));
        }
    }
    for (bit, name) in DOUBLE_REQUIRED {
        if info.double_fp & bit == 0 {
            out.push(format!("double precision without {name}"));
        }
    }
    out
}

/// Whether `info` meets every requirement.
pub fn usable(info: &DeviceInfo) -> bool {
    requirements(info).is_empty()
}

/// The device `gpu` picks among `devices`: the first usable one with its own
/// memory, else the first usable one with memory shared with the host. `None` when none is usable.
pub fn preferred(devices: &[DeviceInfo]) -> Option<usize> {
    let pick = |unified: bool| devices.iter().position(|d| d.unified_memory == unified && usable(d));
    pick(false).or_else(|| pick(true))
}

/// Bytes a run may place on the device: three quarters of its global memory.
pub fn memory_budget(info: &DeviceInfo) -> u64 {
    info.global_memory / 4 * 3
}

/// Reads what `device` reports. A failed query leaves its field at the default and adds a line
/// to `errors`; the double precision config of a device without `cl_khr_fp64` may fail and
/// counts as 0.
pub fn read(device: &cl::Device, index: usize, platform: &str, platform_version: &str) -> DeviceInfo {
    let mut info = DeviceInfo { index, platform: platform.to_string(), platform_version: platform_version.to_string(), ..Default::default() };
    let mut errors = Vec::new();
    macro_rules! get {
        ($method:ident, $param:expr, $name:literal) => {
            match device.$method($param) {
                Ok(v) => v,
                Err(e) => {
                    errors.push(format!("{}: {e}", $name));
                    Default::default()
                }
            }
        };
    }
    info.name = get!(info_string, cl::CL_DEVICE_NAME, "CL_DEVICE_NAME");
    info.vendor = get!(info_string, cl::CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
    info.version = get!(info_string, cl::CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
    info.driver = get!(info_string, cl::CL_DRIVER_VERSION, "CL_DRIVER_VERSION");
    info.c_version = get!(info_string, cl::CL_DEVICE_OPENCL_C_VERSION, "CL_DEVICE_OPENCL_C_VERSION");
    info.profile = get!(info_string, cl::CL_DEVICE_PROFILE, "CL_DEVICE_PROFILE");
    info.device_type = get!(info_u64, cl::CL_DEVICE_TYPE, "CL_DEVICE_TYPE");
    info.little_endian = get!(info_bool, cl::CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
    info.available = get!(info_bool, cl::CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
    info.compiler_available = get!(info_bool, cl::CL_DEVICE_COMPILER_AVAILABLE, "CL_DEVICE_COMPILER_AVAILABLE");
    info.global_memory = get!(info_u64, cl::CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
    info.max_allocation = get!(info_u64, cl::CL_DEVICE_MAX_MEM_ALLOC_SIZE, "CL_DEVICE_MAX_MEM_ALLOC_SIZE");
    info.unified_memory = get!(info_bool, cl::CL_DEVICE_HOST_UNIFIED_MEMORY, "CL_DEVICE_HOST_UNIFIED_MEMORY");
    info.compute_units = get!(info_u32, cl::CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
    let extensions: String = get!(info_string, cl::CL_DEVICE_EXTENSIONS, "CL_DEVICE_EXTENSIONS");
    info.extensions = extensions.split_whitespace().map(str::to_string).collect();
    if info.has_extension("cl_amd_device_attribute_query") {
        info.board = device.info_string(cl::CL_DEVICE_BOARD_NAME_AMD).unwrap_or_default();
    }
    info.single_fp = get!(info_u64, cl::CL_DEVICE_SINGLE_FP_CONFIG, "CL_DEVICE_SINGLE_FP_CONFIG");
    info.double_fp = match device.info_u64(cl::CL_DEVICE_DOUBLE_FP_CONFIG) {
        Ok(v) => v,
        Err(_) if !info.has_extension("cl_khr_fp64") => 0,
        Err(e) => {
            errors.push(format!("CL_DEVICE_DOUBLE_FP_CONFIG: {e}"));
            0
        }
    };
    info.errors = errors;
    info
}

/// Every device of `api` with its handle, numbered as [`probe`] numbers them.
pub fn devices(api: &'static cl::Api) -> Result<(Vec<(cl::Device, DeviceInfo)>, Vec<String>), ClError> {
    let mut out = Vec::new();
    let mut errors = Vec::new();
    for (p, platform) in cl::platforms(api)?.into_iter().enumerate() {
        let name = platform.info_string(cl::CL_PLATFORM_NAME).unwrap_or_else(|e| format!("platform {p} ({e})"));
        let version = platform.info_string(cl::CL_PLATFORM_VERSION).unwrap_or_default();
        match platform.devices() {
            Ok(list) => {
                for device in list {
                    let info = read(&device, out.len(), &name, &version);
                    out.push((device, info));
                }
            }
            Err(e) => errors.push(format!("{name}: {e}")),
        }
    }
    Ok((out, errors))
}

/// Probes the system runtime. `Err` only when no OpenCL runtime can be loaded or the platform list
/// itself fails; a runtime with no platform gives an empty probe.
pub fn probe() -> Result<Probe, ClError> {
    probe_api(cl::runtime()?)
}

/// [`probe`] over a given runtime.
pub fn probe_api(api: &'static cl::Api) -> Result<Probe, ClError> {
    let (list, errors) = devices(api)?;
    Ok(Probe { devices: list.into_iter().map(|(_, info)| info).collect(), errors })
}

/// The live device numbered `index` and what it reports.
pub fn open(index: usize) -> Result<(cl::Device, DeviceInfo), ClError> {
    let (mut list, _) = devices(cl::runtime()?)?;
    if index >= list.len() {
        return Err(ClError::Usage(format!("no OpenCL device {index} ({} listed)", list.len())));
    }
    Ok(list.swap_remove(index))
}
