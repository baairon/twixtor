//! `vfx devices` and the device check of `vfx apply --device`: the render devices of this build.
//!
//! The listing shows the CPU and every OpenCL device the system runtime reports, numbered as
//! `gpu=N` takes them, with what each reports and whether it meets the device requirements. It
//! probes with info queries only: no program is built and no kernel runs. Without an OpenCL
//! runtime it lists the CPU and prints one `vfx: note:` line on stderr.
//!
//! The device of a temporal run (`retime_engine::device::exec::open`), settled
//! before it reads a frame, under `--check` too: `cpu` (the default) is the CPU; `gpu` is the best
//! device that passes the gate (the requirements, the self-check and the program build), else the
//! CPU with one `vfx: note:` line saying why; `gpu=N` is the device numbered `N` here, strictly:
//! one that is not listed or fails the gate is `unsupported` (exit code 3) with the reasons. A
//! frame effect refuses `--device` itself as a usage error (`cli`).

use crate::error::CliError;
use fx_core::{DeviceRequest, FxError};
use retime_engine::device::cl::ClError;
use retime_engine::device::exec::{self, Executor, Opened};
use std::sync::Arc;
use retime_engine::device::probe::{self, DeviceInfo, Probe};
use serde::Serialize;
use std::io::Write;

/// One render device of the listing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Device {
    /// The `--device` value that selects it.
    pub id: String,
    pub name: String,
    /// `cpu` or `gpu` (every OpenCL device is a `gpu=N`, whatever its OpenCL type).
    pub kind: &'static str,
    /// The device a run without `--device` uses.
    pub default: bool,
    /// What an OpenCL device reports; absent for the CPU.
    #[serde(flatten)]
    pub opencl: Option<OpenCl>,
}

/// The probe of one OpenCL device.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct OpenCl {
    /// `CL_DEVICE_NAME` (the board name is the listing's `name` when the device reports one).
    pub device_name: String,
    pub platform: String,
    pub vendor: String,
    /// `CL_DEVICE_VERSION`.
    pub version: String,
    pub driver: String,
    pub opencl_c: String,
    pub profile: String,
    /// The OpenCL device type: `gpu`, `cpu`, `accelerator`, `custom` or `other`.
    #[serde(rename = "type")]
    pub device_type: &'static str,
    pub little_endian: bool,
    /// Bytes.
    pub global_memory: u64,
    /// Bytes, the largest single buffer.
    pub max_allocation: u64,
    pub unified_memory: bool,
    pub compute_units: u32,
    pub fp64: bool,
    pub single_fp: Vec<String>,
    pub single_fp_bits: u64,
    pub double_fp: Vec<String>,
    pub double_fp_bits: u64,
    pub extensions: Vec<String>,
    /// `ok`, or `refused` with `reasons`.
    pub verdict: &'static str,
    pub reasons: Vec<String>,
}

fn cpu() -> Device {
    Device { id: DeviceRequest::Cpu.to_string(), name: "CPU".into(), kind: "cpu", default: true, opencl: None }
}

fn opencl_device(info: &DeviceInfo) -> Device {
    let reasons = probe::requirements(info);
    Device {
        id: DeviceRequest::GpuIndex(info.index as u32).to_string(),
        name: info.display_name().to_string(),
        kind: "gpu",
        default: false,
        opencl: Some(OpenCl {
            device_name: info.name.clone(),
            platform: info.platform.clone(),
            vendor: info.vendor.clone(),
            version: info.version.clone(),
            driver: info.driver.clone(),
            opencl_c: info.c_version.clone(),
            profile: info.profile.clone(),
            device_type: info.type_name(),
            little_endian: info.little_endian,
            global_memory: info.global_memory,
            max_allocation: info.max_allocation,
            unified_memory: info.unified_memory,
            compute_units: info.compute_units,
            fp64: info.has_extension("cl_khr_fp64"),
            single_fp: probe::fp_flags(info.single_fp),
            single_fp_bits: info.single_fp,
            double_fp: probe::fp_flags(info.double_fp),
            double_fp_bits: info.double_fp,
            extensions: info.extensions.clone(),
            verdict: if reasons.is_empty() { "ok" } else { "refused" },
            reasons,
        }),
    }
}

/// The devices of this build from a probe result, and the note to print when OpenCL is absent
/// or a platform could not be listed.
pub fn devices_from(probe: &Result<Probe, ClError>) -> (Vec<Device>, Vec<String>) {
    let mut list = vec![cpu()];
    let mut notes = Vec::new();
    match probe {
        Ok(p) => {
            list.extend(p.devices.iter().map(opencl_device));
            if p.devices.is_empty() {
                notes.push("no OpenCL device found, listing the CPU only".to_string());
            }
            for e in &p.errors {
                notes.push(format!("OpenCL platform skipped: {e}"));
            }
        }
        Err(e) => notes.push(format!("{e}, listing the CPU only")),
    }
    (list, notes)
}

/// The devices of this build: the CPU, then every OpenCL device of the system runtime.
pub fn devices() -> (Vec<Device>, Vec<String>) {
    devices_from(&probe::probe())
}

/// The device a run renders on: `None` for the CPU, with the note to print when `gpu` falls back
/// to the CPU.
pub struct RunDevice {
    pub executor: Option<Arc<Executor>>,
    pub note: Option<String>,
}

/// Opens what `request` asks for ([`exec::open`]): the CPU, a device that passed the gate, or for
/// `gpu` the CPU and a note; `gpu=N` that cannot be served is `unsupported` with the reasons.
pub fn open(request: DeviceRequest) -> Result<RunDevice, CliError> {
    match exec::open(request) {
        Ok(Opened::Cpu) => Ok(RunDevice { executor: None, note: None }),
        Ok(Opened::Device(e)) => Ok(RunDevice { executor: Some(e), note: None }),
        Ok(Opened::Fallback(why)) => Ok(RunDevice { executor: None, note: Some(why) }),
        Err(why) => Err(FxError::Unsupported(why).into()),
    }
}

/// Writes the listing (a table, or with `json` a JSON array) to `out` and its notes to stderr.
pub fn run(json: bool, out: &mut impl Write) -> Result<(), CliError> {
    let (list, notes) = devices();
    for n in &notes {
        eprintln!("vfx: note: {n}");
    }
    write_listing(&list, json, out)
}

pub fn write_listing(list: &[Device], json: bool, out: &mut impl Write) -> Result<(), CliError> {
    let text = if json { to_json(list) } else { to_table(list) };
    out.write_all(text.as_bytes())
        .and_then(|_| out.flush())
        .map_err(|e| CliError::io("write error", e))
}

pub fn to_json(list: &[Device]) -> String {
    let mut text = serde_json::to_string_pretty(list).expect("device list serializes");
    text.push('\n');
    text
}

fn mib(bytes: u64) -> String {
    format!("{} MiB", bytes >> 20)
}

/// The table: one row per device, then a block per OpenCL device with what it reports.
pub fn to_table(list: &[Device]) -> String {
    let gpus = list.iter().any(|d| d.opencl.is_some());
    let mut header = vec!["id", "name", "kind", "default"];
    if gpus {
        header.push("verdict");
    }
    let mut rows = vec![header.into_iter().map(String::from).collect::<Vec<_>>()];
    for d in list {
        let mut row = vec![d.id.clone(), d.name.clone(), d.kind.to_string(), if d.default { "yes" } else { "no" }.to_string()];
        if gpus {
            row.push(d.opencl.as_ref().map_or("ok", |o| o.verdict).to_string());
        }
        rows.push(row);
    }
    let mut text = crate::list::render(&rows);
    for d in list {
        let Some(o) = &d.opencl else { continue };
        let fp = |names: &[String], bits: u64| {
            if names.is_empty() {
                format!("0x{bits:02x}")
            } else {
                format!("0x{bits:02x} {}", names.join(" "))
            }
        };
        let name = if o.device_name == d.name { d.name.clone() } else { format!("{} ({})", d.name, o.device_name) };
        let mut lines = vec![
            vec!["platform".to_string(), o.platform.clone()],
            vec!["vendor".to_string(), o.vendor.clone()],
            vec!["version".to_string(), format!("{}, {}, driver {}", o.version, o.opencl_c, o.driver)],
            vec![
                "device".to_string(),
                format!(
                    "{}, {}, {} endian, {} compute units",
                    o.device_type,
                    o.profile,
                    if o.little_endian { "little" } else { "big" },
                    o.compute_units
                ),
            ],
            vec![
                "memory".to_string(),
                format!(
                    "{} global, {} largest buffer, {}",
                    mib(o.global_memory),
                    mib(o.max_allocation),
                    if o.unified_memory { "shared with the host" } else { "dedicated" }
                ),
            ],
            vec!["fp32".to_string(), fp(&o.single_fp, o.single_fp_bits)],
            vec!["fp64".to_string(), if o.fp64 { fp(&o.double_fp, o.double_fp_bits) } else { "none (no cl_khr_fp64)".to_string() }],
            vec!["verdict".to_string(), o.verdict.to_string()],
        ];
        for r in &o.reasons {
            lines.push(vec!["reason".to_string(), r.clone()]);
        }
        text.push('\n');
        text.push_str(&format!("{}  {name}\n", d.id));
        for line in crate::list::render(&lines).lines() {
            text.push_str("  ");
            text.push_str(line);
            text.push('\n');
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(index: usize, name: &str) -> DeviceInfo {
        DeviceInfo {
            index,
            platform: "Test Platform".into(),
            platform_version: "OpenCL 3.0".into(),
            name: name.into(),
            board: String::new(),
            vendor: "Test Vendor".into(),
            version: "OpenCL 3.0 test".into(),
            driver: "1.2.3".into(),
            c_version: "OpenCL C 1.2".into(),
            profile: "FULL_PROFILE".into(),
            device_type: 4,
            little_endian: true,
            available: true,
            compiler_available: true,
            global_memory: 8 << 30,
            max_allocation: 2 << 30,
            unified_memory: false,
            compute_units: 24,
            extensions: vec!["cl_khr_fp64".into(), "cl_khr_icd".into()],
            single_fp: 0xbf,
            double_fp: 0x3f,
            errors: Vec::new(),
        }
    }

    #[test]
    fn without_opencl_the_cpu_is_listed_with_a_note() {
        let missing: Result<Probe, ClError> =
            Err(ClError::NoRuntime { library: "OpenCL.dll".into(), reason: "The specified module could not be found.".into() });
        let (list, notes) = devices_from(&missing);
        assert_eq!(list, vec![cpu()]);
        assert_eq!(notes, ["no OpenCL runtime (OpenCL.dll: The specified module could not be found.), listing the CPU only"]);
        assert_eq!(to_table(&list), "id   name  kind  default\ncpu  CPU   cpu   yes\n");
        let v: serde_json::Value = serde_json::from_str(&to_json(&list)).unwrap();
        assert_eq!(v, serde_json::json!([{"id": "cpu", "name": "CPU", "kind": "cpu", "default": true}]));
        // a runtime with no device
        let (list, notes) = devices_from(&Ok(Probe::default()));
        assert_eq!(list, vec![cpu()]);
        assert_eq!(notes, ["no OpenCL device found, listing the CPU only"]);
        let (_, notes) = devices_from(&Ok(Probe { devices: vec![gpu(0, "G")], errors: vec!["P: clGetDeviceIDs: CL_OUT_OF_HOST_MEMORY (-6)".into()] }));
        assert_eq!(notes, ["OpenCL platform skipped: P: clGetDeviceIDs: CL_OUT_OF_HOST_MEMORY (-6)"]);
    }

    #[test]
    fn opencl_devices_are_listed_with_their_probe_and_verdict() {
        let mut bad = gpu(1, "gfx0000");
        bad.board = "Test Board".into();
        bad.unified_memory = true;
        bad.single_fp = 0x3e;
        bad.extensions.clear();
        bad.double_fp = 0;
        let probe = Ok(Probe { devices: vec![gpu(0, "Test GPU"), bad], errors: Vec::new() });
        let (list, notes) = devices_from(&probe);
        assert!(notes.is_empty());
        assert_eq!(list.iter().map(|d| d.id.as_str()).collect::<Vec<_>>(), ["cpu", "gpu=0", "gpu=1"]);
        assert_eq!(list[2].name, "Test Board", "the board name is shown when there is one");
        let text = to_table(&list);
        let want = "\
id     name        kind  default  verdict
cpu    CPU         cpu   yes      ok
gpu=0  Test GPU    gpu   no       ok
gpu=1  Test Board  gpu   no       refused

gpu=0  Test GPU
  platform  Test Platform
  vendor    Test Vendor
  version   OpenCL 3.0 test, OpenCL C 1.2, driver 1.2.3
  device    gpu, FULL_PROFILE, little endian, 24 compute units
  memory    8192 MiB global, 2048 MiB largest buffer, dedicated
  fp32      0xbf DENORM INF_NAN ROUND_TO_NEAREST ROUND_TO_ZERO ROUND_TO_INF FMA CORRECTLY_ROUNDED_DIVIDE_SQRT
  fp64      0x3f DENORM INF_NAN ROUND_TO_NEAREST ROUND_TO_ZERO ROUND_TO_INF FMA
  verdict   ok

gpu=1  Test Board (gfx0000)
  platform  Test Platform
  vendor    Test Vendor
  version   OpenCL 3.0 test, OpenCL C 1.2, driver 1.2.3
  device    gpu, FULL_PROFILE, little endian, 24 compute units
  memory    8192 MiB global, 2048 MiB largest buffer, shared with the host
  fp32      0x3e INF_NAN ROUND_TO_NEAREST ROUND_TO_ZERO ROUND_TO_INF FMA
  fp64      none (no cl_khr_fp64)
  verdict   refused
  reason    no cl_khr_fp64 (double precision)
  reason    single precision without DENORM
  reason    single precision without CORRECTLY_ROUNDED_DIVIDE_SQRT
  reason    double precision without DENORM
  reason    double precision without INF_NAN
  reason    double precision without ROUND_TO_NEAREST
";
        assert_eq!(text, want);
        let v: serde_json::Value = serde_json::from_str(&to_json(&list)).unwrap();
        assert_eq!(v[0], serde_json::json!({"id": "cpu", "name": "CPU", "kind": "cpu", "default": true}));
        let g = &v[1];
        assert_eq!(g["id"], "gpu=0");
        assert_eq!(g["kind"], "gpu");
        assert_eq!(g["type"], "gpu");
        assert_eq!(g["default"], false);
        assert_eq!(g["global_memory"], 8u64 << 30);
        assert_eq!(g["single_fp_bits"], 0xbf);
        assert_eq!(g["double_fp"][0], "DENORM");
        assert_eq!(g["fp64"], true);
        assert_eq!(g["verdict"], "ok");
        assert_eq!(g["reasons"], serde_json::json!([]));
        assert_eq!(v[2]["verdict"], "refused");
        assert_eq!(v[2]["device_name"], "gfx0000");
        assert_eq!(v[2]["reasons"].as_array().unwrap().len(), 6);
        let keys: Vec<&str> = g.as_object().unwrap().keys().map(String::as_str).collect();
        for k in ["platform", "vendor", "version", "driver", "opencl_c", "profile", "little_endian", "max_allocation", "unified_memory", "compute_units", "double_fp_bits", "extensions"] {
            assert!(keys.contains(&k), "{k}");
        }
    }

    #[test]
    fn every_listed_id_parses_as_a_device_request() {
        let probe = Ok(Probe { devices: vec![gpu(0, "A"), gpu(1, "B"), gpu(2, "C")], errors: Vec::new() });
        for d in devices_from(&probe).0 {
            assert!(DeviceRequest::parse(&d.id).is_some(), "{}", d.id);
        }
    }

    #[test]
    fn the_cpu_opens_nothing_gpu_falls_back_and_a_missing_gpu_n_is_unsupported() {
        let cpu = open(DeviceRequest::Cpu).unwrap();
        assert!(cpu.executor.is_none() && cpu.note.is_none());
        // gpu: a device that passes the gate, or the CPU with one note; never an error
        let any = open(DeviceRequest::Gpu).unwrap();
        match (&any.executor, &any.note) {
            (Some(e), None) => assert!(e.label().starts_with("gpu=")),
            (None, Some(n)) => assert!(n.starts_with("--device gpu: ") && n.ends_with("rendering on the CPU"), "{n}"),
            _ => panic!("gpu opens a device or notes the fallback"),
        }
        // a number past the listing is refused with the reason, wherever this runs
        let err = open(DeviceRequest::GpuIndex(4000)).err().expect("gpu=4000 is refused");
        assert_eq!(err.code(), 3);
        assert!(err.to_string().starts_with("unsupported: --device gpu=4000: "), "{err}");
    }
}
