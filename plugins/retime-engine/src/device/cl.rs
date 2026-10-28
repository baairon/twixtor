//! Hand written OpenCL 1.2 bindings: the runtime is loaded at run time, the entry points are
//! declared here by hand, and owning wrappers release what they create.
//!
//! This is the only module of the workspace allowed to contain `unsafe` code: the crate root
//! denies `unsafe_code` and allows it on this module alone, and `tests/no_unsafe.rs` scans every
//! `.rs` file of the workspace for the keyword outside this file. Everything unsafe stays behind
//! the safe wrappers below; no other module sees a raw handle.
//!
//! Loading. Windows: `LoadLibraryExW("OpenCL.dll", LOAD_LIBRARY_SEARCH_SYSTEM32)`, so only the
//! system ICD loader is ever loaded, never a DLL from the working or the executable directory;
//! Linux and the other Unix systems: `dlopen("libOpenCL.so.1")`; macOS: the OpenCL framework at its
//! fixed system path. The library is loaded once per process and stays loaded (ICD loaders do not
//! expect to be unloaded); a missing library or entry point is a [`ClError`], never a panic.
//!
//! Constants, types and signatures are written from the Khronos OpenCL 1.2 specification (no header
//! file); `tests/device_cl.rs` cross-checks them against the devices of the machine it runs on
//! (returned sizes of every scalar query, strings that must start with `OpenCL`, the device's own
//! platform handle, error codes provoked on purpose, buffer flags read back).
//!
//! Rules of the wrappers: every driver status becomes a [`ClError`]; reads and writes are blocking,
//! so no host memory is referenced after a call returns; a launch sets every argument of the
//! kernel each time and is bounded by [`MAX_WORK_ITEMS`]; one queue is in order. A released object
//! that a queued command still uses is deleted by the driver when that command finishes (OpenCL 1.2
//! `clRelease*` semantics), so dropping a wrapper is always sound. No arithmetic happens here.

use std::ffi::{c_char, c_void, CString};
use std::fmt;
use std::marker::PhantomData;
use std::ptr;
use std::sync::OnceLock;

/// An opaque OpenCL object handle (`cl_platform_id`, `cl_device_id`, `cl_context`, ...).
type Handle = *mut c_void;

// ---- constants (OpenCL 1.2) -------------------------------------------------------------------

pub const CL_SUCCESS: i32 = 0;
pub const CL_TRUE: u32 = 1;
pub const CL_FALSE: u32 = 0;

/// `cl_platform_info`.
pub const CL_PLATFORM_PROFILE: u32 = 0x0900;
pub const CL_PLATFORM_VERSION: u32 = 0x0901;
pub const CL_PLATFORM_NAME: u32 = 0x0902;
pub const CL_PLATFORM_VENDOR: u32 = 0x0903;
pub const CL_PLATFORM_EXTENSIONS: u32 = 0x0904;

/// `cl_device_type` bits.
pub const CL_DEVICE_TYPE_DEFAULT: u64 = 1 << 0;
pub const CL_DEVICE_TYPE_CPU: u64 = 1 << 1;
pub const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
pub const CL_DEVICE_TYPE_ACCELERATOR: u64 = 1 << 3;
pub const CL_DEVICE_TYPE_CUSTOM: u64 = 1 << 4;
pub const CL_DEVICE_TYPE_ALL: u64 = 0xFFFF_FFFF;

/// `cl_device_info`.
pub const CL_DEVICE_TYPE: u32 = 0x1000;
pub const CL_DEVICE_VENDOR_ID: u32 = 0x1001;
pub const CL_DEVICE_MAX_COMPUTE_UNITS: u32 = 0x1002;
pub const CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: u32 = 0x1003;
pub const CL_DEVICE_MAX_WORK_GROUP_SIZE: u32 = 0x1004;
pub const CL_DEVICE_MAX_CLOCK_FREQUENCY: u32 = 0x100C;
pub const CL_DEVICE_ADDRESS_BITS: u32 = 0x100D;
pub const CL_DEVICE_MAX_MEM_ALLOC_SIZE: u32 = 0x1010;
pub const CL_DEVICE_IMAGE_SUPPORT: u32 = 0x1016;
pub const CL_DEVICE_MEM_BASE_ADDR_ALIGN: u32 = 0x1019;
pub const CL_DEVICE_SINGLE_FP_CONFIG: u32 = 0x101B;
pub const CL_DEVICE_GLOBAL_MEM_SIZE: u32 = 0x101F;
pub const CL_DEVICE_LOCAL_MEM_SIZE: u32 = 0x1023;
pub const CL_DEVICE_ERROR_CORRECTION_SUPPORT: u32 = 0x1024;
pub const CL_DEVICE_ENDIAN_LITTLE: u32 = 0x1026;
pub const CL_DEVICE_AVAILABLE: u32 = 0x1027;
pub const CL_DEVICE_COMPILER_AVAILABLE: u32 = 0x1028;
pub const CL_DEVICE_QUEUE_PROPERTIES: u32 = 0x102A;
pub const CL_DEVICE_NAME: u32 = 0x102B;
pub const CL_DEVICE_VENDOR: u32 = 0x102C;
pub const CL_DRIVER_VERSION: u32 = 0x102D;
pub const CL_DEVICE_PROFILE: u32 = 0x102E;
pub const CL_DEVICE_VERSION: u32 = 0x102F;
pub const CL_DEVICE_EXTENSIONS: u32 = 0x1030;
pub const CL_DEVICE_PLATFORM: u32 = 0x1031;
pub const CL_DEVICE_DOUBLE_FP_CONFIG: u32 = 0x1032;
pub const CL_DEVICE_HOST_UNIFIED_MEMORY: u32 = 0x1035;
pub const CL_DEVICE_OPENCL_C_VERSION: u32 = 0x103D;
pub const CL_DEVICE_LINKER_AVAILABLE: u32 = 0x103E;
/// `cl_amd_device_attribute_query`: the marketing name of an AMD device (its `CL_DEVICE_NAME` is
/// the chip, for example `gfx1103`).
pub const CL_DEVICE_BOARD_NAME_AMD: u32 = 0x4038;

/// `cl_device_fp_config` bits.
pub const CL_FP_DENORM: u64 = 1 << 0;
pub const CL_FP_INF_NAN: u64 = 1 << 1;
pub const CL_FP_ROUND_TO_NEAREST: u64 = 1 << 2;
pub const CL_FP_ROUND_TO_ZERO: u64 = 1 << 3;
pub const CL_FP_ROUND_TO_INF: u64 = 1 << 4;
pub const CL_FP_FMA: u64 = 1 << 5;
pub const CL_FP_SOFT_FLOAT: u64 = 1 << 6;
pub const CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT: u64 = 1 << 7;

/// `cl_context_properties` and `cl_context_info`.
pub const CL_CONTEXT_REFERENCE_COUNT: u32 = 0x1080;
pub const CL_CONTEXT_DEVICES: u32 = 0x1081;
pub const CL_CONTEXT_PROPERTIES: u32 = 0x1082;
pub const CL_CONTEXT_NUM_DEVICES: u32 = 0x1083;
pub const CL_CONTEXT_PLATFORM: u32 = 0x1084;

/// `cl_command_queue_properties` bits and `cl_command_queue_info`.
pub const CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE: u64 = 1 << 0;
pub const CL_QUEUE_PROFILING_ENABLE: u64 = 1 << 1;
pub const CL_QUEUE_CONTEXT: u32 = 0x1090;
pub const CL_QUEUE_DEVICE: u32 = 0x1091;
pub const CL_QUEUE_REFERENCE_COUNT: u32 = 0x1092;
pub const CL_QUEUE_PROPERTIES: u32 = 0x1093;

/// `cl_mem_flags` bits.
pub const CL_MEM_READ_WRITE: u64 = 1 << 0;
pub const CL_MEM_WRITE_ONLY: u64 = 1 << 1;
pub const CL_MEM_READ_ONLY: u64 = 1 << 2;
pub const CL_MEM_USE_HOST_PTR: u64 = 1 << 3;
pub const CL_MEM_ALLOC_HOST_PTR: u64 = 1 << 4;
pub const CL_MEM_COPY_HOST_PTR: u64 = 1 << 5;
pub const CL_MEM_HOST_WRITE_ONLY: u64 = 1 << 7;
pub const CL_MEM_HOST_READ_ONLY: u64 = 1 << 8;
pub const CL_MEM_HOST_NO_ACCESS: u64 = 1 << 9;

/// `cl_mem_info`.
pub const CL_MEM_TYPE: u32 = 0x1100;
pub const CL_MEM_FLAGS: u32 = 0x1101;
pub const CL_MEM_SIZE: u32 = 0x1102;
pub const CL_MEM_OBJECT_BUFFER: u32 = 0x10F0;

/// `cl_program_build_info` and `cl_build_status`.
pub const CL_PROGRAM_BINARY_SIZES: u32 = 0x1165;
pub const CL_PROGRAM_BINARIES: u32 = 0x1166;
pub const CL_PROGRAM_BUILD_STATUS: u32 = 0x1181;
pub const CL_PROGRAM_BUILD_OPTIONS: u32 = 0x1182;
pub const CL_PROGRAM_BUILD_LOG: u32 = 0x1183;
pub const CL_BUILD_SUCCESS: i32 = 0;
pub const CL_BUILD_NONE: i32 = -1;
pub const CL_BUILD_ERROR: i32 = -2;
pub const CL_BUILD_IN_PROGRESS: i32 = -3;

/// `cl_kernel_info` and `cl_kernel_work_group_info`.
pub const CL_KERNEL_FUNCTION_NAME: u32 = 0x1190;
pub const CL_KERNEL_NUM_ARGS: u32 = 0x1191;
pub const CL_KERNEL_WORK_GROUP_SIZE: u32 = 0x11B0;

/// Status codes, by value: `(code, name)`.
pub const ERRORS: &[(i32, &str)] = &[
    (0, "CL_SUCCESS"),
    (-1, "CL_DEVICE_NOT_FOUND"),
    (-2, "CL_DEVICE_NOT_AVAILABLE"),
    (-3, "CL_COMPILER_NOT_AVAILABLE"),
    (-4, "CL_MEM_OBJECT_ALLOCATION_FAILURE"),
    (-5, "CL_OUT_OF_RESOURCES"),
    (-6, "CL_OUT_OF_HOST_MEMORY"),
    (-7, "CL_PROFILING_INFO_NOT_AVAILABLE"),
    (-8, "CL_MEM_COPY_OVERLAP"),
    (-9, "CL_IMAGE_FORMAT_MISMATCH"),
    (-10, "CL_IMAGE_FORMAT_NOT_SUPPORTED"),
    (-11, "CL_BUILD_PROGRAM_FAILURE"),
    (-12, "CL_MAP_FAILURE"),
    (-13, "CL_MISALIGNED_SUB_BUFFER_OFFSET"),
    (-14, "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"),
    (-15, "CL_COMPILE_PROGRAM_FAILURE"),
    (-16, "CL_LINKER_NOT_AVAILABLE"),
    (-17, "CL_LINK_PROGRAM_FAILURE"),
    (-18, "CL_DEVICE_PARTITION_FAILED"),
    (-19, "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"),
    (-30, "CL_INVALID_VALUE"),
    (-31, "CL_INVALID_DEVICE_TYPE"),
    (-32, "CL_INVALID_PLATFORM"),
    (-33, "CL_INVALID_DEVICE"),
    (-34, "CL_INVALID_CONTEXT"),
    (-35, "CL_INVALID_QUEUE_PROPERTIES"),
    (-36, "CL_INVALID_COMMAND_QUEUE"),
    (-37, "CL_INVALID_HOST_PTR"),
    (-38, "CL_INVALID_MEM_OBJECT"),
    (-39, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"),
    (-40, "CL_INVALID_IMAGE_SIZE"),
    (-41, "CL_INVALID_SAMPLER"),
    (-42, "CL_INVALID_BINARY"),
    (-43, "CL_INVALID_BUILD_OPTIONS"),
    (-44, "CL_INVALID_PROGRAM"),
    (-45, "CL_INVALID_PROGRAM_EXECUTABLE"),
    (-46, "CL_INVALID_KERNEL_NAME"),
    (-47, "CL_INVALID_KERNEL_DEFINITION"),
    (-48, "CL_INVALID_KERNEL"),
    (-49, "CL_INVALID_ARG_INDEX"),
    (-50, "CL_INVALID_ARG_VALUE"),
    (-51, "CL_INVALID_ARG_SIZE"),
    (-52, "CL_INVALID_KERNEL_ARGS"),
    (-53, "CL_INVALID_WORK_DIMENSION"),
    (-54, "CL_INVALID_WORK_GROUP_SIZE"),
    (-55, "CL_INVALID_WORK_ITEM_SIZE"),
    (-56, "CL_INVALID_GLOBAL_OFFSET"),
    (-57, "CL_INVALID_EVENT_WAIT_LIST"),
    (-58, "CL_INVALID_EVENT"),
    (-59, "CL_INVALID_OPERATION"),
    (-60, "CL_INVALID_GL_OBJECT"),
    (-61, "CL_INVALID_BUFFER_SIZE"),
    (-62, "CL_INVALID_MIP_LEVEL"),
    (-63, "CL_INVALID_GLOBAL_WORK_SIZE"),
    (-64, "CL_INVALID_PROPERTY"),
    (-65, "CL_INVALID_IMAGE_DESCRIPTOR"),
    (-66, "CL_INVALID_COMPILER_OPTIONS"),
    (-67, "CL_INVALID_LINKER_OPTIONS"),
    (-68, "CL_INVALID_DEVICE_PARTITION_COUNT"),
    (-1001, "CL_PLATFORM_NOT_FOUND_KHR"),
];

/// The name of a status code, or `None` for a code this table does not know.
pub fn error_name(code: i32) -> Option<&'static str> {
    ERRORS.iter().find(|(c, _)| *c == code).map(|(_, n)| *n)
}

/// `clGetPlatformIDs` returns this when no ICD is installed (`cl_khr_icd`).
const CL_PLATFORM_NOT_FOUND_KHR: i32 = -1001;
const CL_DEVICE_NOT_FOUND: i32 = -1;

/// The most work items one launch may have. Kernels in this crate do a bounded amount of work per
/// item, so this bounds a launch; it covers one 4096x2160 four channel image.
pub const MAX_WORK_ITEMS: usize = 1 << 26;
/// Largest string or log a query may return before it is treated as a broken driver.
const MAX_INFO_BYTES: usize = 16 << 20;

// ---- errors -----------------------------------------------------------------------------------

/// Every failure of this module. None of them is a panic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClError {
    /// No OpenCL runtime library could be loaded.
    NoRuntime { library: String, reason: String },
    /// The library loaded but lacks an entry point.
    MissingEntry { library: String, name: &'static str },
    /// An API call returned a status other than `CL_SUCCESS`.
    Api { call: &'static str, code: i32 },
    /// The program did not build; `log` is the device's build log.
    Build { code: i32, log: String },
    /// A query returned data of an unexpected size or shape.
    BadInfo { call: &'static str, param: u32, detail: String },
    /// The wrappers refused a call before reaching the driver.
    Usage(String),
}

impl fmt::Display for ClError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClError::NoRuntime { library, reason } => write!(f, "no OpenCL runtime ({library}: {reason})"),
            ClError::MissingEntry { library, name } => write!(f, "{library} has no entry point {name}"),
            ClError::Api { call, code } => write!(f, "{call}: {}", code_text(*code)),
            ClError::Build { code, log } => {
                let log = log.trim();
                if log.is_empty() {
                    write!(f, "program build failed: {}", code_text(*code))
                } else {
                    write!(f, "program build failed: {}: {log}", code_text(*code))
                }
            }
            ClError::BadInfo { call, param, detail } => write!(f, "{call}(0x{param:04x}): {detail}"),
            ClError::Usage(text) => write!(f, "{text}"),
        }
    }
}

impl std::error::Error for ClError {}

/// `NAME (code)`, or `status code` for an unknown code.
pub fn code_text(code: i32) -> String {
    match error_name(code) {
        Some(name) => format!("{name} ({code})"),
        None => format!("status {code}"),
    }
}

fn check(call: &'static str, code: i32) -> Result<(), ClError> {
    if code == CL_SUCCESS {
        Ok(())
    } else {
        Err(ClError::Api { call, code })
    }
}

// ---- the operating system loader ---------------------------------------------------------------

#[cfg(windows)]
mod os {
    use std::ffi::c_void;

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
        fn GetLastError() -> u32;
    }

    /// Search only `%windir%\System32` for the library and its dependencies.
    const LOAD_LIBRARY_SEARCH_SYSTEM32: u32 = 0x0000_0800;
    pub const DEFAULT_LIBRARY: &str = "OpenCL.dll";

    pub struct Lib(*mut c_void);

    pub fn open(name: &str) -> Result<Lib, String> {
        if name.contains(['/', '\\', ':']) {
            return Err("only a library name is loaded, never a path".into());
        }
        let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
        // SAFETY: `wide` is a NUL terminated UTF-16 string that outlives the call.
        let handle = unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), LOAD_LIBRARY_SEARCH_SYSTEM32) };
        if handle.is_null() {
            // SAFETY: reads the calling thread's last error, no pointers involved.
            let code = unsafe { GetLastError() };
            Err(std::io::Error::from_raw_os_error(code as i32).to_string())
        } else {
            Ok(Lib(handle))
        }
    }

    /// The address of `name` (NUL terminated), or null.
    pub fn symbol(lib: &Lib, name: &str) -> *mut c_void {
        debug_assert!(name.ends_with('\0'));
        // SAFETY: `lib.0` is a module handle from `LoadLibraryExW` and `name` is NUL terminated.
        unsafe { GetProcAddress(lib.0, name.as_ptr()) }
    }

    pub fn close(lib: Lib) {
        // SAFETY: the handle came from `LoadLibraryExW` and no symbol of it is used afterwards.
        unsafe {
            FreeLibrary(lib.0);
        }
    }
}

#[cfg(unix)]
mod os {
    use std::ffi::{c_char, c_int, c_void, CStr, CString};

    #[cfg_attr(all(target_os = "linux", target_env = "gnu"), link(name = "dl"))]
    extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
        fn dlerror() -> *mut c_char;
    }

    const RTLD_NOW: c_int = 2;
    #[cfg(target_os = "macos")]
    pub const DEFAULT_LIBRARY: &str = "/System/Library/Frameworks/OpenCL.framework/OpenCL";
    #[cfg(not(target_os = "macos"))]
    pub const DEFAULT_LIBRARY: &str = "libOpenCL.so.1";

    pub struct Lib(*mut c_void);

    pub fn open(name: &str) -> Result<Lib, String> {
        let c = CString::new(name).map_err(|_| "the library name contains NUL".to_string())?;
        // SAFETY: `c` is a NUL terminated string that outlives the call.
        let handle = unsafe { dlopen(c.as_ptr(), RTLD_NOW) };
        if handle.is_null() {
            // SAFETY: `dlerror` returns null or a NUL terminated string valid until the next call.
            let text = unsafe {
                let e = dlerror();
                if e.is_null() {
                    "dlopen failed".to_string()
                } else {
                    CStr::from_ptr(e).to_string_lossy().into_owned()
                }
            };
            Err(text)
        } else {
            Ok(Lib(handle))
        }
    }

    pub fn symbol(lib: &Lib, name: &str) -> *mut c_void {
        debug_assert!(name.ends_with('\0'));
        // SAFETY: `lib.0` is a handle from `dlopen` and `name` is NUL terminated.
        unsafe { dlsym(lib.0, name.as_ptr() as *const c_char) }
    }

    pub fn close(lib: Lib) {
        // SAFETY: the handle came from `dlopen` and no symbol of it is used afterwards.
        unsafe {
            dlclose(lib.0);
        }
    }
}

#[cfg(not(any(windows, unix)))]
mod os {
    use std::ffi::c_void;
    pub const DEFAULT_LIBRARY: &str = "OpenCL";
    pub struct Lib;
    pub fn open(_: &str) -> Result<Lib, String> {
        Err("no dynamic loader on this platform".into())
    }
    pub fn symbol(_: &Lib, _: &str) -> *mut c_void {
        std::ptr::null_mut()
    }
    pub fn close(_: Lib) {}
}

/// The library name the runtime is loaded from on this platform.
pub const DEFAULT_LIBRARY: &str = os::DEFAULT_LIBRARY;

// ---- entry points -----------------------------------------------------------------------------

type Notify = Option<unsafe extern "system" fn(*const c_char, *const c_void, usize, *mut c_void)>;
type BuildNotify = Option<unsafe extern "system" fn(Handle, *mut c_void)>;
type InfoFn = unsafe extern "system" fn(Handle, u32, usize, *mut c_void, *mut usize) -> i32;
type Info2Fn = unsafe extern "system" fn(Handle, Handle, u32, usize, *mut c_void, *mut usize) -> i32;
type ReleaseFn = unsafe extern "system" fn(Handle) -> i32;

macro_rules! entry_points {
    ($($field:ident : $ty:ty = $name:literal;)*) => {
        /// The loaded runtime: one function pointer per entry point.
        pub struct Api {
            library: String,
            $($field: $ty,)*
        }

        /// The entry points this module declares, in declaration order.
        pub const ENTRY_POINTS: &[&str] = &[$($name),*];

        impl Api {
            fn resolve(lib: &os::Lib, library: &str) -> Result<Api, ClError> {
                $(
                    let p = os::symbol(lib, concat!($name, "\0"));
                    if p.is_null() {
                        return Err(ClError::MissingEntry { library: library.to_string(), name: $name });
                    }
                    // SAFETY: `p` is the non null address of the exported function `$name`, whose
                    // C signature is the one `$ty` declares (OpenCL 1.2); both are pointer sized.
                    let $field = unsafe { std::mem::transmute::<*mut c_void, $ty>(p) };
                )*
                Ok(Api { library: library.to_string(), $($field,)* })
            }
        }
    };
}

entry_points! {
    get_platform_ids: unsafe extern "system" fn(u32, *mut Handle, *mut u32) -> i32 = "clGetPlatformIDs";
    get_platform_info: InfoFn = "clGetPlatformInfo";
    get_device_ids: unsafe extern "system" fn(Handle, u64, u32, *mut Handle, *mut u32) -> i32 = "clGetDeviceIDs";
    get_device_info: InfoFn = "clGetDeviceInfo";
    create_context: unsafe extern "system" fn(*const isize, u32, *const Handle, Notify, *mut c_void, *mut i32) -> Handle = "clCreateContext";
    get_context_info: InfoFn = "clGetContextInfo";
    release_context: ReleaseFn = "clReleaseContext";
    create_command_queue: unsafe extern "system" fn(Handle, Handle, u64, *mut i32) -> Handle = "clCreateCommandQueue";
    get_command_queue_info: InfoFn = "clGetCommandQueueInfo";
    release_command_queue: ReleaseFn = "clReleaseCommandQueue";
    create_buffer: unsafe extern "system" fn(Handle, u64, usize, *mut c_void, *mut i32) -> Handle = "clCreateBuffer";
    get_mem_object_info: InfoFn = "clGetMemObjectInfo";
    release_mem_object: ReleaseFn = "clReleaseMemObject";
    create_program_with_source: unsafe extern "system" fn(Handle, u32, *const *const c_char, *const usize, *mut i32) -> Handle = "clCreateProgramWithSource";
    build_program: unsafe extern "system" fn(Handle, u32, *const Handle, *const c_char, BuildNotify, *mut c_void) -> i32 = "clBuildProgram";
    get_program_build_info: Info2Fn = "clGetProgramBuildInfo";
    get_program_info: InfoFn = "clGetProgramInfo";
    create_program_with_binary: unsafe extern "system" fn(Handle, u32, *const Handle, *const usize, *const *const u8, *mut i32, *mut i32) -> Handle = "clCreateProgramWithBinary";
    release_program: ReleaseFn = "clReleaseProgram";
    create_kernel: unsafe extern "system" fn(Handle, *const c_char, *mut i32) -> Handle = "clCreateKernel";
    set_kernel_arg: unsafe extern "system" fn(Handle, u32, usize, *const c_void) -> i32 = "clSetKernelArg";
    get_kernel_info: InfoFn = "clGetKernelInfo";
    get_kernel_work_group_info: Info2Fn = "clGetKernelWorkGroupInfo";
    release_kernel: ReleaseFn = "clReleaseKernel";
    enqueue_read_buffer: unsafe extern "system" fn(Handle, Handle, u32, usize, usize, *mut c_void, u32, *const Handle, *mut Handle) -> i32 = "clEnqueueReadBuffer";
    enqueue_write_buffer: unsafe extern "system" fn(Handle, Handle, u32, usize, usize, *const c_void, u32, *const Handle, *mut Handle) -> i32 = "clEnqueueWriteBuffer";
    enqueue_copy_buffer: unsafe extern "system" fn(Handle, Handle, Handle, usize, usize, usize, u32, *const Handle, *mut Handle) -> i32 = "clEnqueueCopyBuffer";
    enqueue_fill_buffer: unsafe extern "system" fn(Handle, Handle, *const c_void, usize, usize, usize, u32, *const Handle, *mut Handle) -> i32 = "clEnqueueFillBuffer";
    enqueue_nd_range_kernel: unsafe extern "system" fn(Handle, Handle, u32, *const usize, *const usize, *const usize, u32, *const Handle, *mut Handle) -> i32 = "clEnqueueNDRangeKernel";
    flush: ReleaseFn = "clFlush";
    finish: ReleaseFn = "clFinish";
}

// SAFETY: the table holds plain function pointers into a library that is never unloaded; OpenCL 1.2
// entry points are thread safe except `clSetKernelArg` on one kernel, which `Kernel` guards.
unsafe impl Send for Api {}
// SAFETY: as above, the table is read only after it is built.
unsafe impl Sync for Api {}

impl fmt::Debug for Api {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Api").field("library", &self.library).finish_non_exhaustive()
    }
}

impl Api {
    /// Loads `library` (a name only, never a path, on Windows) and resolves every entry point. The
    /// library stays loaded for the rest of the process. Fails cleanly when the library or an
    /// entry point is missing.
    pub fn load(library: &str) -> Result<&'static Api, ClError> {
        let lib = os::open(library).map_err(|reason| ClError::NoRuntime { library: library.to_string(), reason })?;
        match Api::resolve(&lib, library) {
            Ok(api) => {
                // The library is never closed: the function pointers live as long as the process.
                std::mem::forget(lib);
                Ok(Box::leak(Box::new(api)))
            }
            Err(e) => {
                os::close(lib);
                Err(e)
            }
        }
    }

    /// The library the table was loaded from.
    pub fn library(&self) -> &str {
        &self.library
    }
}

static RUNTIME: OnceLock<Result<&'static Api, ClError>> = OnceLock::new();

/// The system OpenCL runtime ([`DEFAULT_LIBRARY`]), loaded on first use.
pub fn runtime() -> Result<&'static Api, ClError> {
    RUNTIME.get_or_init(|| Api::load(DEFAULT_LIBRARY)).clone()
}

// ---- info queries -----------------------------------------------------------------------------

/// Runs a two step info query (`size` first, then the data) through `call`.
fn query_bytes(
    name: &'static str,
    param: u32,
    call: impl Fn(usize, *mut c_void, *mut usize) -> i32,
) -> Result<Vec<u8>, ClError> {
    let mut size = 0usize;
    check(name, call(0, ptr::null_mut(), &mut size))?;
    if size > MAX_INFO_BYTES {
        return Err(ClError::BadInfo { call: name, param, detail: format!("{size} bytes") });
    }
    let mut buf = vec![0u8; size];
    if size > 0 {
        check(name, call(size, buf.as_mut_ptr() as *mut c_void, ptr::null_mut()))?;
    }
    Ok(buf)
}

/// A scalar query whose returned size must be exactly `N` bytes: a wrong constant shows up here.
fn query_scalar<const N: usize>(
    name: &'static str,
    param: u32,
    call: impl Fn(usize, *mut c_void, *mut usize) -> i32,
) -> Result<[u8; N], ClError> {
    let mut size = 0usize;
    check(name, call(0, ptr::null_mut(), &mut size))?;
    if size != N {
        return Err(ClError::BadInfo { call: name, param, detail: format!("{size} bytes, expected {N}") });
    }
    let mut out = [0u8; N];
    check(name, call(N, out.as_mut_ptr() as *mut c_void, ptr::null_mut()))?;
    Ok(out)
}

/// The text of a `char[]` query: up to the first NUL, invalid UTF-8 replaced, blanks trimmed.
fn text(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).trim().to_string()
}

/// Typed queries over one info entry point.
trait Query {
    fn raw(&self, name: &'static str, param: u32) -> Result<Vec<u8>, ClError>;
    fn scalar<const N: usize>(&self, name: &'static str, param: u32) -> Result<[u8; N], ClError>;
}

struct InfoOf {
    f: InfoFn,
    h: Handle,
}

impl Query for InfoOf {
    fn raw(&self, name: &'static str, param: u32) -> Result<Vec<u8>, ClError> {
        // SAFETY: `h` is a live handle of the kind `f` takes; the buffer and size pointers are valid
        // for the sizes passed (see `query_bytes`).
        query_bytes(name, param, |size, value, ret| unsafe { (self.f)(self.h, param, size, value, ret) })
    }
    fn scalar<const N: usize>(&self, name: &'static str, param: u32) -> Result<[u8; N], ClError> {
        // SAFETY: as in `raw`.
        query_scalar(name, param, |size, value, ret| unsafe { (self.f)(self.h, param, size, value, ret) })
    }
}

macro_rules! typed_queries {
    () => {
        /// A `char[]` query as text.
        pub fn info_string(&self, param: u32) -> Result<String, ClError> {
            self.query().raw(Self::CALL, param).map(|b| text(&b))
        }
        /// A `cl_ulong` or bitfield query; the driver must return exactly 8 bytes.
        pub fn info_u64(&self, param: u32) -> Result<u64, ClError> {
            self.query().scalar::<8>(Self::CALL, param).map(u64::from_ne_bytes)
        }
        /// A `cl_uint` or `cl_int` query; exactly 4 bytes.
        pub fn info_u32(&self, param: u32) -> Result<u32, ClError> {
            self.query().scalar::<4>(Self::CALL, param).map(u32::from_ne_bytes)
        }
        /// A `cl_bool` query (a `cl_uint`); exactly 4 bytes, and 0 or 1.
        pub fn info_bool(&self, param: u32) -> Result<bool, ClError> {
            match self.info_u32(param)? {
                0 => Ok(false),
                1 => Ok(true),
                v => Err(ClError::BadInfo { call: Self::CALL, param, detail: format!("cl_bool {v}") }),
            }
        }
        /// A `size_t` query; exactly the pointer size.
        pub fn info_usize(&self, param: u32) -> Result<usize, ClError> {
            self.query().scalar::<{ std::mem::size_of::<usize>() }>(Self::CALL, param).map(usize::from_ne_bytes)
        }
        /// The raw bytes of a query, for the constant checks.
        pub fn info_bytes(&self, param: u32) -> Result<Vec<u8>, ClError> {
            self.query().raw(Self::CALL, param)
        }
    };
}

// ---- platforms and devices --------------------------------------------------------------------

/// An OpenCL platform (one ICD).
#[derive(Clone, Copy)]
pub struct Platform {
    api: &'static Api,
    id: Handle,
}

// SAFETY: platform ids are process wide identifiers, valid on every thread.
unsafe impl Send for Platform {}
// SAFETY: as above; queries on them are thread safe.
unsafe impl Sync for Platform {}

impl fmt::Debug for Platform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Platform({:p})", self.id)
    }
}

impl PartialEq for Platform {
    fn eq(&self, other: &Platform) -> bool {
        self.id == other.id
    }
}

impl Platform {
    const CALL: &'static str = "clGetPlatformInfo";

    fn query(&self) -> InfoOf {
        InfoOf { f: self.api.get_platform_info, h: self.id }
    }

    typed_queries!();

    /// Every device of this platform (`CL_DEVICE_TYPE_ALL`), in the driver's order; a platform with
    /// no device gives an empty list.
    pub fn devices(&self) -> Result<Vec<Device>, ClError> {
        let api = self.api;
        let mut n = 0u32;
        // SAFETY: `id` is a platform id from `clGetPlatformIDs`; `n` is a valid out pointer.
        let code = unsafe { (api.get_device_ids)(self.id, CL_DEVICE_TYPE_ALL, 0, ptr::null_mut(), &mut n) };
        if code == CL_DEVICE_NOT_FOUND || (code == CL_SUCCESS && n == 0) {
            return Ok(Vec::new());
        }
        check("clGetDeviceIDs", code)?;
        let mut ids = vec![ptr::null_mut(); n as usize];
        // SAFETY: `ids` has room for `n` handles.
        check("clGetDeviceIDs", unsafe { (api.get_device_ids)(self.id, CL_DEVICE_TYPE_ALL, n, ids.as_mut_ptr(), ptr::null_mut()) })?;
        Ok(ids.into_iter().map(|id| Device { api, id, platform: self.id }).collect())
    }
}

/// Every platform of the runtime, in the ICD loader's order; none installed gives an empty list.
pub fn platforms(api: &'static Api) -> Result<Vec<Platform>, ClError> {
    let mut n = 0u32;
    // SAFETY: `n` is a valid out pointer; no array is passed.
    let code = unsafe { (api.get_platform_ids)(0, ptr::null_mut(), &mut n) };
    if code == CL_PLATFORM_NOT_FOUND_KHR || (code == CL_SUCCESS && n == 0) {
        return Ok(Vec::new());
    }
    check("clGetPlatformIDs", code)?;
    let mut ids = vec![ptr::null_mut(); n as usize];
    // SAFETY: `ids` has room for `n` handles.
    check("clGetPlatformIDs", unsafe { (api.get_platform_ids)(n, ids.as_mut_ptr(), ptr::null_mut()) })?;
    Ok(ids.into_iter().map(|id| Platform { api, id }).collect())
}

/// One OpenCL device.
#[derive(Clone, Copy)]
pub struct Device {
    api: &'static Api,
    id: Handle,
    platform: Handle,
}

// SAFETY: root device ids are process wide identifiers, valid on every thread.
unsafe impl Send for Device {}
// SAFETY: as above; queries on them are thread safe.
unsafe impl Sync for Device {}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Device({:p})", self.id)
    }
}

impl Device {
    const CALL: &'static str = "clGetDeviceInfo";

    fn query(&self) -> InfoOf {
        InfoOf { f: self.api.get_device_info, h: self.id }
    }

    typed_queries!();

    /// The platform this device was listed under.
    pub fn platform(&self) -> Platform {
        Platform { api: self.api, id: self.platform }
    }

    /// Whether `CL_DEVICE_PLATFORM` names the platform the device was listed under (a check of
    /// that constant and of the handle size).
    pub fn reports_its_platform(&self) -> Result<bool, ClError> {
        Ok(self.info_usize(CL_DEVICE_PLATFORM)? == self.platform as usize)
    }
}

// ---- context and queue ------------------------------------------------------------------------

/// A context over one device.
pub struct Context {
    api: &'static Api,
    handle: Handle,
    device: Device,
}

// SAFETY: OpenCL contexts may be used from any thread (all calls on them are thread safe).
unsafe impl Send for Context {}
// SAFETY: as above.
unsafe impl Sync for Context {}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: the handle is ours; objects created from it hold their own reference.
        unsafe {
            (self.api.release_context)(self.handle);
        }
    }
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Context({:p})", self.handle)
    }
}

impl Context {
    /// A context for `device` on its own platform (`CL_CONTEXT_PLATFORM`).
    pub fn new(device: &Device) -> Result<Context, ClError> {
        let api = device.api;
        let props: [isize; 3] = [CL_CONTEXT_PLATFORM as isize, device.platform as isize, 0];
        let mut err = 0i32;
        // SAFETY: `props` is a zero terminated property list, one valid device id is passed, no
        // callback; `err` is a valid out pointer.
        let handle = unsafe { (api.create_context)(props.as_ptr(), 1, &device.id, None, ptr::null_mut(), &mut err) };
        if err != CL_SUCCESS || handle.is_null() {
            return Err(ClError::Api { call: "clCreateContext", code: if err == CL_SUCCESS { -6 } else { err } });
        }
        Ok(Context { api, handle, device: *device })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn query(&self) -> InfoOf {
        InfoOf { f: self.api.get_context_info, h: self.handle }
    }

    /// `CL_CONTEXT_NUM_DEVICES` (a `cl_uint`): 1 for a context of this module.
    pub fn device_count(&self) -> Result<u32, ClError> {
        self.query().scalar::<4>("clGetContextInfo", CL_CONTEXT_NUM_DEVICES).map(u32::from_ne_bytes)
    }
}

/// An in-order command queue.
pub struct Queue {
    api: &'static Api,
    handle: Handle,
    context: Handle,
}

// SAFETY: every queue call of OpenCL 1.2 is thread safe; the executor still serialises on one
// queue.
unsafe impl Send for Queue {}
// SAFETY: as above.
unsafe impl Sync for Queue {}

impl Drop for Queue {
    fn drop(&mut self) {
        // SAFETY: the handle is ours. A queue release implies a flush of its commands.
        unsafe {
            (self.api.release_command_queue)(self.handle);
        }
    }
}

/// The work items of one launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Range {
    /// `n` items, `get_global_id(0)` in `0..n`.
    D1(usize),
    /// `w x h` items, `get_global_id(0)` in `0..w` and `get_global_id(1)` in `0..h`.
    D2(usize, usize),
}

impl Range {
    /// The number of work items, `None` on overflow.
    pub fn items(self) -> Option<usize> {
        match self {
            Range::D1(n) => Some(n),
            Range::D2(w, h) => w.checked_mul(h),
        }
    }
}

impl Queue {
    /// An in-order queue (no properties) on the context's device.
    pub fn new(context: &Context) -> Result<Queue, ClError> {
        let api = context.api;
        let mut err = 0i32;
        // SAFETY: a live context and its own device; `err` is a valid out pointer.
        let handle = unsafe { (api.create_command_queue)(context.handle, context.device.id, 0, &mut err) };
        if err != CL_SUCCESS || handle.is_null() {
            return Err(ClError::Api { call: "clCreateCommandQueue", code: if err == CL_SUCCESS { -6 } else { err } });
        }
        Ok(Queue { api, handle, context: context.handle })
    }

    /// `CL_QUEUE_PROPERTIES`: 0 for the in-order queue of this module.
    pub fn properties(&self) -> Result<u64, ClError> {
        InfoOf { f: self.api.get_command_queue_info, h: self.handle }
            .scalar::<8>("clGetCommandQueueInfo", CL_QUEUE_PROPERTIES)
            .map(u64::from_ne_bytes)
    }

    fn same_context(&self, mem: &Mem) -> Result<(), ClError> {
        if mem.context != self.context {
            return Err(ClError::Usage("the buffer belongs to another context".into()));
        }
        Ok(())
    }

    /// Writes `data` into `buf` from element `offset` on, blocking until the data are copied.
    pub fn write_at<T: Pod>(&self, buf: &Buffer<T>, offset: usize, data: &[T]) -> Result<(), ClError> {
        self.same_context(&buf.mem)?;
        if offset.checked_add(data.len()).is_none_or(|end| end > buf.len) {
            return Err(ClError::Usage(format!("write of {} at {offset} into a buffer of {}", data.len(), buf.len)));
        }
        if data.is_empty() {
            return Ok(());
        }
        let size = std::mem::size_of::<T>();
        // SAFETY: a blocking write of `data.len() * size` bytes from a live slice into a range the
        // check above keeps inside the buffer; the slice is not referenced after the call returns.
        check("clEnqueueWriteBuffer", unsafe {
            (self.api.enqueue_write_buffer)(
                self.handle,
                buf.mem.handle,
                CL_TRUE,
                offset * size,
                data.len() * size,
                data.as_ptr() as *const c_void,
                0,
                ptr::null(),
                ptr::null_mut(),
            )
        })
    }

    /// Writes the whole buffer; `data.len()` must equal `buf.len()`.
    pub fn write<T: Pod>(&self, buf: &Buffer<T>, data: &[T]) -> Result<(), ClError> {
        if data.len() != buf.len {
            return Err(ClError::Usage(format!("write of {} into a buffer of {}", data.len(), buf.len)));
        }
        self.write_at(buf, 0, data)
    }

    /// Reads `out.len()` elements of `buf` from element `offset` on, blocking until they arrive
    /// (after every command queued before it, the queue being in order).
    pub fn read_at<T: Pod>(&self, buf: &Buffer<T>, offset: usize, out: &mut [T]) -> Result<(), ClError> {
        self.same_context(&buf.mem)?;
        if offset.checked_add(out.len()).is_none_or(|end| end > buf.len) {
            return Err(ClError::Usage(format!("read of {} at {offset} from a buffer of {}", out.len(), buf.len)));
        }
        if out.is_empty() {
            return Ok(());
        }
        let size = std::mem::size_of::<T>();
        // SAFETY: a blocking read into a live, exclusively borrowed slice of exactly the requested
        // size, from a range inside the buffer; every bit pattern is a valid `T` (`Pod`).
        check("clEnqueueReadBuffer", unsafe {
            (self.api.enqueue_read_buffer)(
                self.handle,
                buf.mem.handle,
                CL_TRUE,
                offset * size,
                out.len() * size,
                out.as_mut_ptr() as *mut c_void,
                0,
                ptr::null(),
                ptr::null_mut(),
            )
        })
    }

    /// Reads the whole buffer into a new vector.
    pub fn read<T: Pod>(&self, buf: &Buffer<T>) -> Result<Vec<T>, ClError> {
        let mut out = vec![T::default(); buf.len];
        self.read_at(buf, 0, &mut out)?;
        Ok(out)
    }

    /// Sets every element of `buf` to `value` (queued; ordered with the commands around it).
    pub fn fill<T: Pod>(&self, buf: &Buffer<T>, value: T) -> Result<(), ClError> {
        self.same_context(&buf.mem)?;
        let size = std::mem::size_of::<T>();
        // SAFETY: the pattern is a live value of `size` bytes (1, 2, 4 or 8, all valid pattern
        // sizes); the driver copies it before returning. The range is the whole buffer.
        check("clEnqueueFillBuffer", unsafe {
            (self.api.enqueue_fill_buffer)(
                self.handle,
                buf.mem.handle,
                &value as *const T as *const c_void,
                size,
                0,
                buf.len * size,
                0,
                ptr::null(),
                ptr::null_mut(),
            )
        })
    }

    /// Copies all of `src` into `dst` (equal lengths; queued).
    pub fn copy<T: Pod>(&self, src: &Buffer<T>, dst: &Buffer<T>) -> Result<(), ClError> {
        self.same_context(&src.mem)?;
        self.same_context(&dst.mem)?;
        if src.len != dst.len {
            return Err(ClError::Usage(format!("copy of {} into a buffer of {}", src.len, dst.len)));
        }
        if src.mem.handle == dst.mem.handle {
            return Err(ClError::Usage("copy of a buffer onto itself".into()));
        }
        // SAFETY: two distinct live buffers of the same byte size, whole range.
        check("clEnqueueCopyBuffer", unsafe {
            (self.api.enqueue_copy_buffer)(
                self.handle,
                src.mem.handle,
                dst.mem.handle,
                0,
                0,
                src.mem.bytes,
                0,
                ptr::null(),
                ptr::null_mut(),
            )
        })
    }

    /// Sets every argument of `kernel` from `args` (in order, all of them) and queues one launch
    /// over `range` with a local size the driver picks. An empty range queues nothing. A range of
    /// more than [`MAX_WORK_ITEMS`] items is refused.
    pub fn run(&self, kernel: &mut Kernel, range: Range, args: &[Arg<'_>]) -> Result<(), ClError> {
        if args.len() != kernel.args as usize {
            return Err(ClError::Usage(format!(
                "kernel {} takes {} arguments, {} given",
                kernel.name,
                kernel.args,
                args.len()
            )));
        }
        let items = range.items().ok_or_else(|| ClError::Usage("launch size overflows".into()))?;
        if items > MAX_WORK_ITEMS {
            return Err(ClError::Usage(format!("launch of {items} work items, at most {MAX_WORK_ITEMS}")));
        }
        for (i, arg) in args.iter().enumerate() {
            if let Arg::Mem(m) = arg {
                self.same_context(m)?;
            }
            kernel.set(i as u32, arg)?;
        }
        if items == 0 {
            return Ok(());
        }
        let (dims, global): (u32, [usize; 2]) = match range {
            Range::D1(n) => (1, [n, 1]),
            Range::D2(w, h) => (2, [w, h]),
        };
        // SAFETY: `global` holds `dims` sizes; no offset, no local size, no events. Every argument
        // was set just above from live objects.
        check("clEnqueueNDRangeKernel", unsafe {
            (self.api.enqueue_nd_range_kernel)(
                self.handle,
                kernel.handle,
                dims,
                ptr::null(),
                global.as_ptr(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null_mut(),
            )
        })
    }

    /// Submits the queued commands to the device without waiting.
    pub fn flush(&self) -> Result<(), ClError> {
        // SAFETY: a live queue.
        check("clFlush", unsafe { (self.api.flush)(self.handle) })
    }

    /// Waits until every queued command has finished.
    pub fn finish(&self) -> Result<(), ClError> {
        // SAFETY: a live queue.
        check("clFinish", unsafe { (self.api.finish)(self.handle) })
    }
}

// ---- buffers ----------------------------------------------------------------------------------

mod sealed {
    pub trait Sealed {}
}

/// Element types a buffer may hold: plain numbers for which every bit pattern is a value.
pub trait Pod: Copy + Default + Send + Sync + 'static + sealed::Sealed {}

macro_rules! pod {
    ($($t:ty),*) => {$(
        impl sealed::Sealed for $t {}
        impl Pod for $t {}
    )*};
}
pod!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);

/// How kernels use a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    ReadWrite,
    ReadOnly,
    WriteOnly,
}

impl Access {
    fn flags(self) -> u64 {
        match self {
            Access::ReadWrite => CL_MEM_READ_WRITE,
            Access::ReadOnly => CL_MEM_READ_ONLY,
            Access::WriteOnly => CL_MEM_WRITE_ONLY,
        }
    }
}

/// A device memory object (untyped); kernels take it through [`Arg::Mem`].
pub struct Mem {
    api: &'static Api,
    handle: Handle,
    context: Handle,
    bytes: usize,
}

// SAFETY: memory objects may be used from any thread; reads and writes go through a queue.
unsafe impl Send for Mem {}
// SAFETY: as above.
unsafe impl Sync for Mem {}

impl Drop for Mem {
    fn drop(&mut self) {
        // SAFETY: the handle is ours; a command still using it keeps it alive until it finishes.
        unsafe {
            (self.api.release_mem_object)(self.handle);
        }
    }
}

impl Mem {
    /// `CL_MEM_FLAGS` as the driver reports them.
    pub fn flags(&self) -> Result<u64, ClError> {
        InfoOf { f: self.api.get_mem_object_info, h: self.handle }
            .scalar::<8>("clGetMemObjectInfo", CL_MEM_FLAGS)
            .map(u64::from_ne_bytes)
    }

    /// `CL_MEM_SIZE` in bytes as the driver reports it.
    pub fn size(&self) -> Result<usize, ClError> {
        InfoOf { f: self.api.get_mem_object_info, h: self.handle }
            .scalar::<{ std::mem::size_of::<usize>() }>("clGetMemObjectInfo", CL_MEM_SIZE)
            .map(usize::from_ne_bytes)
    }

    /// `CL_MEM_TYPE`: `CL_MEM_OBJECT_BUFFER` for every object of this module.
    pub fn kind(&self) -> Result<u32, ClError> {
        InfoOf { f: self.api.get_mem_object_info, h: self.handle }
            .scalar::<4>("clGetMemObjectInfo", CL_MEM_TYPE)
            .map(u32::from_ne_bytes)
    }
}

/// A typed device buffer of `len` elements.
pub struct Buffer<T: Pod> {
    mem: Mem,
    len: usize,
    _t: PhantomData<T>,
}

impl<T: Pod> fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Buffer({:p}, {} x {})", self.mem.handle, self.len, std::any::type_name::<T>())
    }
}

impl<T: Pod> Buffer<T> {
    fn create(context: &Context, len: usize, flags: u64, host: *const T) -> Result<Buffer<T>, ClError> {
        if len == 0 {
            return Err(ClError::Usage("a buffer needs at least one element".into()));
        }
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| ClError::Usage(format!("a buffer of {len} elements overflows")))?;
        let mut err = 0i32;
        // SAFETY: a live context; `host` is null or points to `bytes` readable bytes that the
        // driver copies before returning (`CL_MEM_COPY_HOST_PTR`).
        let handle = unsafe { (context.api.create_buffer)(context.handle, flags, bytes, host as *mut c_void, &mut err) };
        if err != CL_SUCCESS || handle.is_null() {
            return Err(ClError::Api { call: "clCreateBuffer", code: if err == CL_SUCCESS { -4 } else { err } });
        }
        Ok(Buffer { mem: Mem { api: context.api, handle, context: context.handle, bytes }, len, _t: PhantomData })
    }

    /// An uninitialised buffer of `len` elements (at least one).
    pub fn new(context: &Context, len: usize, access: Access) -> Result<Buffer<T>, ClError> {
        Buffer::create(context, len, access.flags(), ptr::null())
    }

    /// A buffer holding a copy of `data` (at least one element).
    pub fn from_slice(context: &Context, data: &[T], access: Access) -> Result<Buffer<T>, ClError> {
        Buffer::create(context, data.len(), access.flags() | CL_MEM_COPY_HOST_PTR, data.as_ptr())
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The buffer as a kernel argument.
    pub fn arg(&self) -> Arg<'_> {
        Arg::Mem(&self.mem)
    }

    /// The untyped memory object (driver queries).
    pub fn mem(&self) -> &Mem {
        &self.mem
    }
}

// ---- programs and kernels ---------------------------------------------------------------------

/// A program built for the context's device.
pub struct Program {
    api: &'static Api,
    handle: Handle,
    device: Handle,
}

// SAFETY: built programs may be used from any thread.
unsafe impl Send for Program {}
// SAFETY: as above.
unsafe impl Sync for Program {}

impl Drop for Program {
    fn drop(&mut self) {
        // SAFETY: the handle is ours; kernels hold their own reference.
        unsafe {
            (self.api.release_program)(self.handle);
        }
    }
}

impl Program {
    /// Compiles `source` with `options` for the context's device. A failed build returns
    /// [`ClError::Build`] with the device's build log.
    pub fn build(context: &Context, source: &str, options: &str) -> Result<Program, ClError> {
        let api = context.api;
        let src = CString::new(source).map_err(|_| ClError::Usage("the program source contains NUL".into()))?;
        let opts = CString::new(options).map_err(|_| ClError::Usage("the build options contain NUL".into()))?;
        let strings = [src.as_ptr()];
        let lengths = [source.len()];
        let mut err = 0i32;
        // SAFETY: one string with its exact length, both arrays live for the call.
        let handle =
            unsafe { (api.create_program_with_source)(context.handle, 1, strings.as_ptr(), lengths.as_ptr(), &mut err) };
        if err != CL_SUCCESS || handle.is_null() {
            return Err(ClError::Api {
                call: "clCreateProgramWithSource",
                code: if err == CL_SUCCESS { -6 } else { err },
            });
        }
        let program = Program { api, handle, device: context.device.id };
        // SAFETY: a live program, one device of its context, NUL terminated options, no callback
        // (so the call blocks until the build is done).
        let code = unsafe { (api.build_program)(handle, 1, &context.device.id, opts.as_ptr(), None, ptr::null_mut()) };
        if code != CL_SUCCESS {
            let log = program.build_log().unwrap_or_default();
            return Err(ClError::Build { code, log });
        }
        Ok(program)
    }

    /// Loads a device binary that [`Program::binary`] returned for this device and driver, and
    /// builds it with `options` (the call OpenCL requires before kernels are created). A binary
    /// the driver does not accept is an error; the caller then builds from source.
    pub fn from_binary(context: &Context, binary: &[u8], options: &str) -> Result<Program, ClError> {
        let api = context.api;
        if binary.is_empty() {
            return Err(ClError::Usage("an empty program binary".into()));
        }
        let opts = CString::new(options).map_err(|_| ClError::Usage("the build options contain NUL".into()))?;
        let lengths = [binary.len()];
        let binaries = [binary.as_ptr()];
        let (mut status, mut err) = (0i32, 0i32);
        // SAFETY: one device of the context, one binary with its exact length; the arrays and the
        // binary live for the call, which copies the binary.
        let handle = unsafe {
            (api.create_program_with_binary)(context.handle, 1, &context.device.id, lengths.as_ptr(), binaries.as_ptr(), &mut status, &mut err)
        };
        if handle.is_null() {
            return Err(ClError::Api { call: "clCreateProgramWithBinary", code: if err == CL_SUCCESS { -6 } else { err } });
        }
        let program = Program { api, handle, device: context.device.id };
        if err != CL_SUCCESS || status != CL_SUCCESS {
            return Err(ClError::Api { call: "clCreateProgramWithBinary", code: if err == CL_SUCCESS { status } else { err } });
        }
        // SAFETY: as in `build`.
        let code = unsafe { (api.build_program)(handle, 1, &context.device.id, opts.as_ptr(), None, ptr::null_mut()) };
        if code != CL_SUCCESS {
            let log = program.build_log().unwrap_or_default();
            return Err(ClError::Build { code, log });
        }
        Ok(program)
    }

    /// The device binary of the built program (`CL_PROGRAM_BINARIES` of its one device).
    pub fn binary(&self) -> Result<Vec<u8>, ClError> {
        const N: usize = std::mem::size_of::<usize>();
        let sizes = |size, value, ret| {
            // SAFETY: a live program; pointers valid for `size`.
            unsafe { (self.api.get_program_info)(self.handle, CL_PROGRAM_BINARY_SIZES, size, value, ret) }
        };
        let len = usize::from_ne_bytes(query_scalar::<N>("clGetProgramInfo", CL_PROGRAM_BINARY_SIZES, sizes)?);
        if len == 0 {
            return Err(ClError::Usage("the driver reports no program binary".into()));
        }
        let mut bin = vec![0u8; len];
        let mut ptrs = [bin.as_mut_ptr()];
        // SAFETY: the program has one device, so the driver writes one binary of `len` bytes
        // through the one pointer of `ptrs`, whose size is passed.
        let code = unsafe {
            (self.api.get_program_info)(self.handle, CL_PROGRAM_BINARIES, std::mem::size_of_val(&ptrs), ptrs.as_mut_ptr() as *mut c_void, ptr::null_mut())
        };
        if code != CL_SUCCESS {
            return Err(ClError::Api { call: "clGetProgramInfo", code });
        }
        Ok(bin)
    }

    /// The build log for the device (warnings of a successful build, errors of a failed one).
    pub fn build_log(&self) -> Result<String, ClError> {
        self.build_text(CL_PROGRAM_BUILD_LOG)
    }

    /// The options the program was built with, as the driver reports them.
    pub fn build_options(&self) -> Result<String, ClError> {
        self.build_text(CL_PROGRAM_BUILD_OPTIONS)
    }

    fn build_text(&self, param: u32) -> Result<String, ClError> {
        let f = |size, value, ret| {
            // SAFETY: live program and device handles; pointers valid for `size`.
            unsafe { (self.api.get_program_build_info)(self.handle, self.device, param, size, value, ret) }
        };
        query_bytes("clGetProgramBuildInfo", param, f).map(|b| text(&b))
    }

    /// `CL_PROGRAM_BUILD_STATUS` (`CL_BUILD_SUCCESS` after a successful build).
    pub fn build_status(&self) -> Result<i32, ClError> {
        let param = CL_PROGRAM_BUILD_STATUS;
        let f = |size, value, ret| {
            // SAFETY: as in `build_text`.
            unsafe { (self.api.get_program_build_info)(self.handle, self.device, param, size, value, ret) }
        };
        query_scalar::<4>("clGetProgramBuildInfo", param, f).map(i32::from_ne_bytes)
    }
}

/// One kernel of a built program. Not `Sync`: setting arguments needs exclusive access.
pub struct Kernel {
    api: &'static Api,
    handle: Handle,
    device: Handle,
    name: String,
    args: u32,
}

// SAFETY: a kernel may move between threads; `&mut self` on `Queue::run` keeps one user at a time.
unsafe impl Send for Kernel {}

impl Drop for Kernel {
    fn drop(&mut self) {
        // SAFETY: the handle is ours; a queued launch keeps what it needs alive.
        unsafe {
            (self.api.release_kernel)(self.handle);
        }
    }
}

impl fmt::Debug for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Kernel({}, {} args)", self.name, self.args)
    }
}

impl fmt::Debug for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Program({:p})", self.handle)
    }
}

impl fmt::Debug for Queue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Queue({:p})", self.handle)
    }
}

/// One kernel argument.
#[derive(Clone, Copy)]
pub enum Arg<'a> {
    Mem(&'a Mem),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    F32(f32),
    F64(f64),
}

impl Kernel {
    /// The kernel `name` of `program`.
    pub fn new(program: &Program, name: &str) -> Result<Kernel, ClError> {
        let api = program.api;
        let c = CString::new(name).map_err(|_| ClError::Usage("the kernel name contains NUL".into()))?;
        let mut err = 0i32;
        // SAFETY: a live built program, a NUL terminated name, a valid out pointer.
        let handle = unsafe { (api.create_kernel)(program.handle, c.as_ptr(), &mut err) };
        if err != CL_SUCCESS || handle.is_null() {
            return Err(ClError::Api { call: "clCreateKernel", code: if err == CL_SUCCESS { -6 } else { err } });
        }
        let mut kernel = Kernel { api, handle, device: program.device, name: name.to_string(), args: 0 };
        kernel.args = InfoOf { f: api.get_kernel_info, h: handle }
            .scalar::<4>("clGetKernelInfo", CL_KERNEL_NUM_ARGS)
            .map(u32::from_ne_bytes)?;
        Ok(kernel)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// `CL_KERNEL_NUM_ARGS`.
    pub fn arg_count(&self) -> u32 {
        self.args
    }

    /// `CL_KERNEL_FUNCTION_NAME` as the driver reports it.
    pub fn function_name(&self) -> Result<String, ClError> {
        InfoOf { f: self.api.get_kernel_info, h: self.handle }.raw("clGetKernelInfo", CL_KERNEL_FUNCTION_NAME).map(|b| text(&b))
    }

    /// `CL_KERNEL_WORK_GROUP_SIZE` on the program's device.
    pub fn work_group_size(&self) -> Result<usize, ClError> {
        let param = CL_KERNEL_WORK_GROUP_SIZE;
        let f = |size, value, ret| {
            // SAFETY: live kernel and device handles; pointers valid for `size`.
            unsafe { (self.api.get_kernel_work_group_info)(self.handle, self.device, param, size, value, ret) }
        };
        query_scalar::<{ std::mem::size_of::<usize>() }>("clGetKernelWorkGroupInfo", param, f).map(usize::from_ne_bytes)
    }

    fn set(&mut self, index: u32, arg: &Arg<'_>) -> Result<(), ClError> {
        let k = self.handle;
        let f = self.api.set_kernel_arg;
        // SAFETY: each arm passes a pointer to a live value of exactly the size it states; the
        // driver copies the value before returning. A memory object is passed as its handle.
        let code = unsafe {
            match arg {
                Arg::Mem(m) => f(k, index, std::mem::size_of::<Handle>(), &m.handle as *const Handle as *const c_void),
                Arg::I32(v) => f(k, index, 4, v as *const i32 as *const c_void),
                Arg::U32(v) => f(k, index, 4, v as *const u32 as *const c_void),
                Arg::I64(v) => f(k, index, 8, v as *const i64 as *const c_void),
                Arg::U64(v) => f(k, index, 8, v as *const u64 as *const c_void),
                Arg::F32(v) => f(k, index, 4, v as *const f32 as *const c_void),
                Arg::F64(v) => f(k, index, 8, v as *const f64 as *const c_void),
            }
        };
        check("clSetKernelArg", code)
    }

    /// Sets one argument by hand (the constant checks provoke size errors with it).
    pub fn set_arg(&mut self, index: u32, arg: Arg<'_>) -> Result<(), ClError> {
        self.set(index, &arg)
    }
}
