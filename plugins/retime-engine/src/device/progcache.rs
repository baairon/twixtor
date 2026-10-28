//! The program cache: device binaries of built programs kept in one file per program in a folder
//! of the user's temporary directory (`vfx-opencl-programs`), so a run loads the binary the driver
//! built in an earlier run instead of compiling the source again.
//!
//! An entry holds its whole key, compared exactly when it is read: the device (platform name and
//! version, device name, vendor and version, driver version), the build options and the full
//! program text, then the binary and an FNV-1a digest of it. Anything that does not match, a
//! binary the driver does not accept and any read or write error mean a plain build from source
//! (the entry is then rewritten); the cache never changes what a program computes, only where its
//! binary comes from. Entries are written to a unique temporary name and renamed into place.
//! `VFX_DEVICE_PROGRAM_CACHE=0` turns the cache off.
//!
//! The gates stay as they are: the self-check runs its kernels from whatever binary was loaded,
//! and the first frame of a run is cross checked against the CPU.

use super::cl::{self, ClError, Context, Program};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const MAGIC: &[u8] = b"vfx opencl program 1\n";

/// FNV-1a over `bytes`.
pub fn fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// What identifies the compiler of `device`: platform, device and driver names and versions.
pub fn device_key(device: &cl::Device) -> String {
    let p = device.platform();
    let q = |r: Result<String, ClError>| r.unwrap_or_default();
    [
        q(p.info_string(cl::CL_PLATFORM_NAME)),
        q(p.info_string(cl::CL_PLATFORM_VERSION)),
        q(device.info_string(cl::CL_DEVICE_NAME)),
        q(device.info_string(cl::CL_DEVICE_VENDOR)),
        q(device.info_string(cl::CL_DEVICE_VERSION)),
        q(device.info_string(cl::CL_DRIVER_VERSION)),
    ]
    .join("\n")
}

fn enabled() -> bool {
    std::env::var_os("VFX_DEVICE_PROGRAM_CACHE").is_none_or(|v| v != "0")
}

/// The folder of the cache.
pub fn folder() -> PathBuf {
    std::env::temp_dir().join("vfx-opencl-programs")
}

fn push_field(out: &mut Vec<u8>, field: &[u8]) {
    out.extend_from_slice(&(field.len() as u64).to_le_bytes());
    out.extend_from_slice(field);
}

/// The bytes of an entry.
pub fn encode(key: &str, options: &str, source: &str, binary: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(MAGIC.len() + key.len() + options.len() + source.len() + binary.len() + 48);
    out.extend_from_slice(MAGIC);
    push_field(&mut out, key.as_bytes());
    push_field(&mut out, options.as_bytes());
    push_field(&mut out, source.as_bytes());
    push_field(&mut out, binary);
    out.extend_from_slice(&fnv(binary).to_le_bytes());
    out
}

fn take_field<'a>(rest: &mut &'a [u8]) -> Option<&'a [u8]> {
    let len = u64::from_le_bytes(rest.get(..8)?.try_into().ok()?) as usize;
    let field = rest.get(8..8usize.checked_add(len)?)?;
    *rest = &rest[8 + len..];
    Some(field)
}

/// The binary of an entry whose key, options and source are exactly these and whose digest
/// matches; `None` for anything else.
pub fn decode<'a>(bytes: &'a [u8], key: &str, options: &str, source: &str) -> Option<&'a [u8]> {
    let mut rest = bytes.strip_prefix(MAGIC)?;
    if take_field(&mut rest)? != key.as_bytes() || take_field(&mut rest)? != options.as_bytes() || take_field(&mut rest)? != source.as_bytes() {
        return None;
    }
    let binary = take_field(&mut rest)?;
    let digest = u64::from_le_bytes(rest.try_into().ok()?);
    (!binary.is_empty() && digest == fnv(binary)).then_some(binary)
}

/// The file of the entry for this key, options and source.
pub fn entry_path(dir: &Path, key: &str, options: &str, source: &str) -> PathBuf {
    let mut id = Vec::with_capacity(key.len() + options.len() + source.len() + 2);
    id.extend_from_slice(key.as_bytes());
    id.push(0);
    id.extend_from_slice(options.as_bytes());
    id.push(0);
    id.extend_from_slice(source.as_bytes());
    dir.join(format!("{:016x}.bin", fnv(&id)))
}

/// Writes an entry under a unique temporary name and renames it into place; errors are ignored
/// (the next run builds from source again).
pub fn store(path: &Path, bytes: &[u8]) {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let Some(dir) = path.parent() else { return };
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let tmp = path.with_extension(format!("{}.{}.tmp", std::process::id(), SEQ.fetch_add(1, Ordering::SeqCst)));
    if std::fs::write(&tmp, bytes).is_ok() && std::fs::rename(&tmp, path).is_ok() {
        return;
    }
    let _ = std::fs::remove_file(&tmp);
}

/// `Program::build(context, source, options)`, from the cached binary when there is a matching
/// one (see the module documentation).
pub fn build(context: &Context, device: &cl::Device, source: &str, options: &str) -> Result<Program, ClError> {
    if !enabled() {
        return Program::build(context, source, options);
    }
    let key = device_key(device);
    let path = entry_path(&folder(), &key, options, source);
    if let Ok(bytes) = std::fs::read(&path) {
        if let Some(binary) = decode(&bytes, &key, options, source) {
            if let Ok(p) = Program::from_binary(context, binary, options) {
                return Ok(p);
            }
        }
    }
    let program = Program::build(context, source, options)?;
    if let Ok(binary) = program.binary() {
        store(&path, &encode(&key, options, source, &binary));
    }
    Ok(program)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_entry_reads_back_only_for_its_exact_key_options_and_source() {
        let bin = [1u8, 2, 3, 250, 0, 7];
        let e = encode("dev\ndriver 1", "-cl-std=CL1.2", "kernel void k() {}", &bin);
        assert_eq!(decode(&e, "dev\ndriver 1", "-cl-std=CL1.2", "kernel void k() {}"), Some(&bin[..]));
        assert_eq!(decode(&e, "dev\ndriver 2", "-cl-std=CL1.2", "kernel void k() {}"), None, "another driver");
        assert_eq!(decode(&e, "dev\ndriver 1", "-cl-std=CL1.2 -cl-mad-enable", "kernel void k() {}"), None, "other options");
        assert_eq!(decode(&e, "dev\ndriver 1", "-cl-std=CL1.2", "kernel void k() { }"), None, "another source");
        // a truncated or corrupted entry is a miss
        assert_eq!(decode(&e[..e.len() - 1], "dev\ndriver 1", "-cl-std=CL1.2", "kernel void k() {}"), None);
        let mut bad = e.clone();
        let at = bad.len() - 9;
        bad[at] ^= 1;
        assert_eq!(decode(&bad, "dev\ndriver 1", "-cl-std=CL1.2", "kernel void k() {}"), None);
        assert_eq!(decode(b"garbage", "dev", "", ""), None);
        assert_eq!(decode(&encode("k", "", "s", &[]), "k", "", "s"), None, "an empty binary is never used");
    }

    #[test]
    fn entries_are_named_by_their_whole_key_and_written_in_place() {
        let dir = std::env::temp_dir().join(format!("vfx-progcache-test-{}", std::process::id()));
        let a = entry_path(&dir, "k", "o", "s");
        assert_ne!(a, entry_path(&dir, "k", "o", "s2"));
        assert_ne!(a, entry_path(&dir, "k", "o2", "s"));
        assert_ne!(entry_path(&dir, "ko", "", "s"), entry_path(&dir, "k", "o", "s"), "fields are separated");
        let bytes = encode("k", "o", "s", &[9, 9, 9]);
        store(&a, &bytes);
        assert_eq!(std::fs::read(&a).unwrap(), bytes);
        store(&a, &encode("k", "o", "s", &[8]));
        assert_eq!(decode(&std::fs::read(&a).unwrap(), "k", "o", "s"), Some(&[8u8][..]), "a rewrite replaces the entry");
        let left: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(left.len(), 1, "no temporary file is left: {left:?}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
