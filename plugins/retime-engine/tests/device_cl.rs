//! Tests on the OpenCL devices of the machine: the hand written constants and signatures checked
//! against real drivers, the wrappers' error paths, the self-check and the contraction test. With
//! no OpenCL runtime or no device every test prints a note and passes (CI runners have none).
//! Run with `--nocapture` to see the probe and the verdicts. Workloads are small and every kernel
//! loop is bounded; the tests take the devices one at a time.

use retime_engine::device::cl::{self, Access, Arg, Buffer, ClError, Context, Kernel, Program, Queue, Range};
use retime_engine::device::kernels::{self, BUILD_OPTIONS};
use retime_engine::device::probe::{self, DeviceInfo};
use retime_engine::device::selfcheck::{self, Refusal, Verdict};
use std::sync::Mutex;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// Every device with what it reports, or `None` (with a printed note) when there is none.
fn devices(test: &str) -> Option<Vec<(cl::Device, DeviceInfo)>> {
    let api = match cl::runtime() {
        Ok(api) => api,
        Err(e) => {
            println!("{test}: skipped, {e}");
            return None;
        }
    };
    match probe::devices(api) {
        Ok((list, _)) if !list.is_empty() => Some(list),
        Ok(_) => {
            println!("{test}: skipped, the OpenCL runtime lists no device");
            None
        }
        Err(e) => {
            println!("{test}: skipped, {e}");
            None
        }
    }
}

fn show(info: &DeviceInfo) -> String {
    let mib = |b: u64| b / (1 << 20);
    format!(
        "gpu={} {} ({}) | vendor {} | platform {} ({}) | {} | driver {} | {} | {} | type {} | little endian {} | \
         global {} MiB | max alloc {} MiB | unified {} | units {} | fp32 0x{:x} {:?} | fp64 0x{:x} {:?} | fp64 ext {}",
        info.index,
        info.display_name(),
        info.name,
        info.vendor,
        info.platform,
        info.platform_version,
        info.version,
        info.driver,
        info.c_version,
        info.profile,
        info.type_name(),
        info.little_endian,
        mib(info.global_memory),
        mib(info.max_allocation),
        info.unified_memory,
        info.compute_units,
        info.single_fp,
        probe::fp_flags(info.single_fp),
        info.double_fp,
        probe::fp_flags(info.double_fp),
        info.has_extension("cl_khr_fp64"),
    )
}

#[test]
fn the_probe_reads_every_device_with_the_sizes_the_spec_gives() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let Some(list) = devices("probe") else { return };
    let api = cl::runtime().unwrap();
    for platform in cl::platforms(api).unwrap() {
        let profile = platform.info_string(cl::CL_PLATFORM_PROFILE).unwrap();
        let version = platform.info_string(cl::CL_PLATFORM_VERSION).unwrap();
        let name = platform.info_string(cl::CL_PLATFORM_NAME).unwrap();
        let vendor = platform.info_string(cl::CL_PLATFORM_VENDOR).unwrap();
        let ext = platform.info_string(cl::CL_PLATFORM_EXTENSIONS).unwrap();
        println!("platform {name} | {vendor} | {version} | {profile} | {ext}");
        assert!(profile == "FULL_PROFILE" || profile == "EMBEDDED_PROFILE", "{profile}");
        assert!(probe::parse_version(&version, "OpenCL ").is_some(), "{version}");
        assert!(!name.is_empty() && !vendor.is_empty());
        // an unknown query is refused with CL_INVALID_VALUE (-30)
        assert_eq!(platform.info_bytes(0x0905_0000).unwrap_err(), ClError::Api { call: "clGetPlatformInfo", code: -30 });
    }
    for (device, info) in &list {
        println!("{}", show(info));
        println!("  extensions: {}", info.extensions.join(" "));
        println!("  requirements: {:?}", probe::requirements(info));
        assert!(info.errors.is_empty(), "{:?}", info.errors);
        assert!(probe::parse_version(&info.version, "OpenCL ").is_some(), "{}", info.version);
        assert!(probe::parse_version(&info.c_version, "OpenCL C ").is_some(), "{}", info.c_version);
        assert!(info.profile == "FULL_PROFILE" || info.profile == "EMBEDDED_PROFILE");
        assert!(info.device_type & 0x1f != 0, "a known type bit");
        assert!(device.reports_its_platform().unwrap(), "CL_DEVICE_PLATFORM names the platform");
        // scalar queries of every width return exactly the size of their type
        device.info_u32(cl::CL_DEVICE_VENDOR_ID).unwrap();
        assert!(device.info_u32(cl::CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS).unwrap() >= 3);
        assert!(device.info_usize(cl::CL_DEVICE_MAX_WORK_GROUP_SIZE).unwrap() >= 1);
        assert!(device.info_u32(cl::CL_DEVICE_MAX_CLOCK_FREQUENCY).unwrap() > 0);
        assert!(matches!(device.info_u32(cl::CL_DEVICE_ADDRESS_BITS).unwrap(), 32 | 64));
        assert!(device.info_u32(cl::CL_DEVICE_MEM_BASE_ADDR_ALIGN).unwrap() >= 8);
        device.info_bool(cl::CL_DEVICE_IMAGE_SUPPORT).unwrap();
        device.info_bool(cl::CL_DEVICE_ERROR_CORRECTION_SUPPORT).unwrap();
        device.info_bool(cl::CL_DEVICE_LINKER_AVAILABLE).unwrap();
        assert!(device.info_u64(cl::CL_DEVICE_LOCAL_MEM_SIZE).unwrap() > 0);
        let queue_props = device.info_u64(cl::CL_DEVICE_QUEUE_PROPERTIES).unwrap();
        assert_eq!(queue_props & !(cl::CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | cl::CL_QUEUE_PROFILING_ENABLE), 0);
        assert!(queue_props & cl::CL_QUEUE_PROFILING_ENABLE != 0, "every device supports profiling");
        // OpenCL 1.2 minimums: the largest allocation is at least a quarter of global memory, single
        // precision has INF_NAN and ROUND_TO_NEAREST, double precision (when present) has FMA,
        // all three rounding modes, INF_NAN and DENORM. A wrong constant or bit breaks these.
        assert!(info.max_allocation >= info.global_memory / 4 || info.max_allocation >= 128 << 20);
        assert_eq!(info.single_fp & 0x6, 0x6, "fp32 INF_NAN and ROUND_TO_NEAREST");
        if info.has_extension("cl_khr_fp64") {
            assert_eq!(info.double_fp & 0x3f, 0x3f, "the fp64 minimum of OpenCL 1.2");
        }
        assert!(info.little_endian, "x86 hosts pair with little endian devices");
        assert_eq!(device.info_bytes(0x7fff).unwrap_err(), ClError::Api { call: "clGetDeviceInfo", code: -30 });
        // a query asked with the wrong width is refused by its size, before any value is read
        assert!(matches!(device.info_u32(cl::CL_DEVICE_GLOBAL_MEM_SIZE), Err(ClError::BadInfo { .. })));
        assert!(matches!(device.info_u64(cl::CL_DEVICE_MAX_COMPUTE_UNITS), Err(ClError::BadInfo { .. })));
        if info.has_extension("cl_amd_device_attribute_query") {
            assert!(!info.board.is_empty(), "CL_DEVICE_BOARD_NAME_AMD answers on an AMD device");
        }
        assert_eq!(probe::read(device, info.index, &info.platform, &info.platform_version), *info);
    }
    let preferred = probe::preferred(&list.iter().map(|(_, i)| i.clone()).collect::<Vec<_>>());
    println!("gpu picks: {preferred:?}");
}

#[test]
fn the_wrappers_drive_a_device_and_report_driver_errors() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let Some(list) = devices("wrappers") else { return };
    for (device, info) in &list {
        let context = Context::new(device).unwrap();
        assert_eq!(context.device_count().unwrap(), 1);
        let queue = Queue::new(&context).unwrap();
        assert_eq!(queue.properties().unwrap(), 0, "in order, no profiling");
        // buffers and their flags as the driver reports them
        let data: Vec<u32> = (0..1000).map(|i| i * 7 + 1).collect();
        let src = Buffer::from_slice(&context, &data, Access::ReadOnly).unwrap();
        assert_eq!(src.mem().flags().unwrap(), cl::CL_MEM_READ_ONLY | cl::CL_MEM_COPY_HOST_PTR);
        assert_eq!(src.mem().size().unwrap(), 4000);
        assert_eq!(src.mem().kind().unwrap(), cl::CL_MEM_OBJECT_BUFFER);
        let dst = Buffer::<u32>::new(&context, 1000, Access::WriteOnly).unwrap();
        assert_eq!(dst.mem().flags().unwrap(), cl::CL_MEM_WRITE_ONLY);
        let rw = Buffer::<u32>::new(&context, 1000, Access::ReadWrite).unwrap();
        assert_eq!(rw.mem().flags().unwrap(), cl::CL_MEM_READ_WRITE);
        assert_eq!(queue.read(&src).unwrap(), data);
        queue.fill(&rw, 0xdead_beef).unwrap();
        assert_eq!(queue.read(&rw).unwrap(), vec![0xdead_beef; 1000]);
        queue.write_at(&rw, 10, &[1, 2, 3]).unwrap();
        let mut three = [0u32; 3];
        queue.read_at(&rw, 10, &mut three).unwrap();
        assert_eq!(three, [1, 2, 3]);
        queue.copy(&src, &rw).unwrap();
        assert_eq!(queue.read(&rw).unwrap(), data);
        assert!(matches!(queue.read_at(&rw, 999, &mut three), Err(ClError::Usage(_))));
        assert!(matches!(queue.write(&rw, &[1, 2]), Err(ClError::Usage(_))));
        assert!(matches!(queue.copy(&rw, &rw), Err(ClError::Usage(_))));
        assert!(matches!(Buffer::<u8>::new(&context, 0, Access::ReadWrite), Err(ClError::Usage(_))));
        // the program and its kernels
        let program = Program::build(&context, &kernels::starter_program(), BUILD_OPTIONS).unwrap();
        assert_eq!(program.build_status().unwrap(), cl::CL_BUILD_SUCCESS);
        let options = program.build_options().unwrap();
        assert!(options.contains("-cl-fp32-correctly-rounded-divide-sqrt"), "{options}");
        println!("gpu={} build log: {:?} | options: {options:?}", info.index, program.build_log().unwrap());
        let mut copy = Kernel::new(&program, "copy_u32").unwrap();
        assert_eq!(copy.arg_count(), 3);
        assert_eq!(copy.function_name().unwrap(), "copy_u32");
        assert!(copy.work_group_size().unwrap() >= 1);
        queue.run(&mut copy, Range::D1(1000), &[src.arg(), dst.arg(), Arg::U32(1000)]).unwrap();
        queue.finish().unwrap();
        assert_eq!(queue.read(&dst).unwrap(), data);
        // an empty range queues nothing, an oversized one is refused before the driver
        queue.run(&mut copy, Range::D1(0), &[src.arg(), dst.arg(), Arg::U32(0)]).unwrap();
        assert!(matches!(queue.run(&mut copy, Range::D1(cl::MAX_WORK_ITEMS + 1), &[src.arg(), dst.arg(), Arg::U32(0)]), Err(ClError::Usage(_))));
        assert!(matches!(queue.run(&mut copy, Range::D1(4), &[src.arg(), dst.arg()]), Err(ClError::Usage(_))));
        // driver errors with their OpenCL codes
        assert_eq!(copy.set_arg(99, Arg::U32(0)).unwrap_err(), ClError::Api { call: "clSetKernelArg", code: -49 });
        assert_eq!(copy.set_arg(2, Arg::U64(0)).unwrap_err(), ClError::Api { call: "clSetKernelArg", code: -51 });
        assert_eq!(Kernel::new(&program, "no_such_kernel").unwrap_err(), ClError::Api { call: "clCreateKernel", code: -46 });
        match Program::build(&context, "__kernel void broken(__global int* x) { x[0] = undefined_name; }", BUILD_OPTIONS) {
            Err(ClError::Build { code, log }) => {
                assert_eq!(code, -11, "CL_BUILD_PROGRAM_FAILURE");
                assert!(log.contains("undefined_name"), "the build log names the error: {log}");
            }
            other => panic!("a broken source must not build: {:?}", other.err()),
        }
        match Program::build(&context, &kernels::starter_program(), "-cl-no-such-option") {
            Err(ClError::Build { code, .. }) => {
                println!("gpu={} unknown build option: {}", info.index, cl::code_text(code));
                assert!(matches!(code, -43 | -11 | -66), "CL_INVALID_BUILD_OPTIONS, CL_BUILD_PROGRAM_FAILURE or CL_INVALID_COMPILER_OPTIONS");
            }
            other => panic!("an unknown option must not build: {:?}", other.err()),
        }
        // a buffer of another context is refused by the wrapper
        let other = Context::new(device).unwrap();
        let foreign = Buffer::<u32>::new(&other, 4, Access::ReadWrite).unwrap();
        assert!(matches!(queue.read(&foreign), Err(ClError::Usage(_))));
        assert!(matches!(queue.run(&mut copy, Range::D1(4), &[foreign.arg(), dst.arg(), Arg::U32(4)]), Err(ClError::Usage(_))));
    }
}

#[test]
fn the_self_check_runs_on_every_device() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let Some(list) = devices("selfcheck") else { return };
    for (device, info) in &list {
        let verdict = selfcheck::selfcheck(device);
        println!("gpu={} {}: selfcheck {verdict}", info.index, info.name);
        // the kernels themselves, also on a device the requirements refuse
        let raw = selfcheck::check_source(device, &kernels::starter_program());
        match &raw {
            Ok(Ok(t)) => println!("gpu={} starter kernels: {} outputs equal to the CPU, NaN payloads {:?}", info.index, t.outputs, t.nan_payloads),
            Ok(Err(m)) => println!("gpu={} starter kernels: first difference: {m}", info.index),
            Err(e) => println!("gpu={} starter kernels: {e}", info.index),
        }
        if probe::usable(info) {
            assert!(raw.is_ok(), "a device meeting the requirements builds and runs the starter program: {raw:?}");
            match (&verdict, &raw) {
                (Verdict::Usable(a), Ok(Ok(b))) => assert_eq!(a, b),
                (Verdict::Refused(Refusal::Mismatch(a)), Ok(Err(b))) => assert_eq!(a, b),
                other => panic!("the verdict follows the checks: {other:?}"),
            }
        } else {
            assert!(matches!(verdict, Verdict::Refused(Refusal::Requirements(_))));
        }
    }
}

#[test]
fn a_contracting_or_fused_build_is_caught() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|p| p.into_inner());
    let Some(list) = devices("contraction") else { return };
    let strict = kernels::starter_program();
    let on = strict.replacen("#pragma OPENCL FP_CONTRACT OFF", "#pragma OPENCL FP_CONTRACT ON", 1);
    assert_ne!(on, strict);
    let blend_line = "out[i] = (b[i] - a[i]) * w + a[i] + 0x0p+0f;";
    assert!(strict.contains(blend_line));
    let fused = strict.replace(blend_line, "out[i] = fma(b[i] - a[i], w, a[i]) + 0x0p+0f;");
    for (device, info) in &list {
        if !probe::usable(info) {
            println!("gpu={} {}: contraction test skipped, {:?}", info.index, info.name, probe::requirements(info));
            continue;
        }
        // FP_CONTRACT ON: a contracting compiler fuses `(b - a) * w + a`; the corpus then differs
        match selfcheck::check_source(device, &on).unwrap() {
            Ok(t) => println!("gpu={} {}: FP_CONTRACT ON passes ({} outputs): this compiler does not contract", info.index, info.name, t.outputs),
            Err(m) => {
                println!("gpu={} {}: FP_CONTRACT ON caught: {m}", info.index, info.name);
                assert!(m.kernel.starts_with("blend") || m.kernel == "chain_f32", "{m}");
            }
        }
        // an explicit fma is always fused, so the check must refuse it
        match selfcheck::check_source(device, &fused).unwrap() {
            Ok(t) => panic!("gpu={}: an fma kernel passed {} outputs", info.index, t.outputs),
            Err(m) => {
                println!("gpu={} {}: explicit fma caught: {m}", info.index, info.name);
                assert_eq!(m.kernel, "blend_f32");
            }
        }
        // relaxed options
        for options in ["-cl-std=CL1.2 -cl-mad-enable", "-cl-std=CL1.2 -cl-fast-relaxed-math", "-cl-std=CL1.2 -cl-denorms-are-zero"] {
            match selfcheck::check_source_with(device, &strict, options) {
                Ok(Ok(t)) => println!("gpu={} `{options}`: passes ({} outputs)", info.index, t.outputs),
                Ok(Err(m)) => println!("gpu={} `{options}`: caught: {m}", info.index),
                Err(e) => println!("gpu={} `{options}`: {e}", info.index),
            }
        }
    }
}
