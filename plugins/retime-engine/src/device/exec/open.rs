//! Opening the device a run asks for: `cpu` opens nothing; `gpu` takes the
//! best device that passes the gate, else the run renders on the CPU with one note saying why;
//! `gpu=N` is strict, the device numbered `N` as `vfx devices` lists it, or a refusal with the
//! reasons (the host exits `unsupported`).
//!
//! The gate at open: the probe requirements (`probe::requirements`), the self-check
//! (`selfcheck::selfcheck`, the starter kernels over the adversarial corpus) and the build of the
//! executor's whole program. No arithmetic beyond integer comparisons.

use super::backend::ClBackend;
use super::{Exec, Executor, OpenTimes, Placement};
use crate::device::cl;
use crate::device::probe::{self, DeviceInfo};
use crate::device::selfcheck;
use fx_core::DeviceRequest;
use std::sync::Arc;
use std::time::Instant;

/// What [`open`] gives for a request.
pub enum Opened {
    /// The CPU was asked for.
    Cpu,
    /// A device passed the gate.
    Device(Arc<Executor>),
    /// `gpu` found no device that passes the gate: the CPU renders and the text says why.
    Fallback(String),
}

/// The order in which `gpu` tries the devices: those that meet the requirements, first the ones
/// with their own memory, then those sharing memory with the host, each group in
/// listing order. `probe::preferred` is the first of this list.
pub fn candidates(devices: &[DeviceInfo]) -> Vec<usize> {
    let pick = |unified: bool| devices.iter().filter(move |d| d.unified_memory == unified && probe::usable(d)).map(|d| d.index);
    pick(false).chain(pick(true)).collect()
}

/// `gpu=N (name)`.
pub fn label(info: &DeviceInfo) -> String {
    format!("gpu={} ({})", info.index, info.display_name())
}

/// The self-check and the program build on one device that meets the requirements. The two are
/// independent (each has its own context), so the program builds on a second thread while the
/// self-check runs; a device the self-check refuses drops its program unused.
fn gate(device: &cl::Device, info: DeviceInfo) -> Result<Arc<Executor>, String> {
    let name = label(&info);
    let start = Instant::now();
    let ((verdict, check_time), built) = std::thread::scope(|sc| {
        let build = sc.spawn(|| {
            let t = Instant::now();
            let b = ClBackend::open(device, info.clone());
            (b, t.elapsed())
        });
        let t = Instant::now();
        let verdict = selfcheck::selfcheck(device);
        ((verdict, t.elapsed()), build.join())
    });
    if !verdict.is_usable() {
        return Err(format!("{name} failed the self-check: {verdict}"));
    }
    let (backend, build_time) = match built {
        Ok((Ok(b), t)) => (b, t),
        Ok((Err(e), _)) => return Err(format!("{name}: the program does not build: {e}")),
        Err(_) => return Err(format!("{name}: the program build stopped")),
    };
    let exec = Exec::new(backend, name, Placement::default());
    exec.set_open_times(OpenTimes { total: start.elapsed(), selfcheck: check_time, build: build_time });
    Ok(Arc::new(exec))
}

/// Opens what `request` asks for. `Err` only for `gpu=N`, with the reasons it is refused.
pub fn open(request: DeviceRequest) -> Result<Opened, String> {
    match request {
        DeviceRequest::Cpu => Ok(Opened::Cpu),
        DeviceRequest::Gpu => Ok(open_any()),
        DeviceRequest::GpuIndex(n) => open_index(n as usize).map(Opened::Device),
    }
}

fn open_any() -> Opened {
    let api = match cl::runtime() {
        Ok(api) => api,
        Err(e) => return Opened::Fallback(format!("--device gpu: {e}, rendering on the CPU")),
    };
    let (list, _) = match probe::devices(api) {
        Ok(l) => l,
        Err(e) => return Opened::Fallback(format!("--device gpu: {e}, rendering on the CPU")),
    };
    if list.is_empty() {
        return Opened::Fallback("--device gpu: no OpenCL device found, rendering on the CPU".into());
    }
    let infos: Vec<DeviceInfo> = list.iter().map(|(_, i)| i.clone()).collect();
    let mut reasons: Vec<String> = infos
        .iter()
        .filter(|i| !probe::usable(i))
        .map(|i| format!("{} does not meet the requirements ({})", label(i), probe::requirements(i).join(", ")))
        .collect();
    let mut list: Vec<Option<(cl::Device, DeviceInfo)>> = list.into_iter().map(Some).collect();
    for index in candidates(&infos) {
        let Some((device, info)) = list[index].take() else { continue };
        match gate(&device, info) {
            Ok(exec) => return Opened::Device(exec),
            Err(why) => reasons.push(why),
        }
    }
    Opened::Fallback(format!("--device gpu: no OpenCL device passes the gate ({}), rendering on the CPU", reasons.join("; ")))
}

fn open_index(n: usize) -> Result<Arc<Executor>, String> {
    let (device, info) = probe::open(n).map_err(|e| format!("--device gpu={n}: {e}"))?;
    let reasons = probe::requirements(&info);
    if !reasons.is_empty() {
        return Err(format!("--device gpu={n}: {} does not meet the requirements: {}", info.display_name(), reasons.join(", ")));
    }
    gate(&device, info).map_err(|why| format!("--device gpu={n}: {why}"))
}
