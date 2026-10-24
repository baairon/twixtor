//! File backed clips for temporal effects: random access to raw frames, and the
//! spool file that gives piped input a seekable form.

use crate::error::CliError;
use fx_core::{Depth, Frame, FxError, Layer, TimeBase};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

/// The frames one clip keeps cached are bounded by this many bytes (but never fewer than two)...
const CACHE_BYTES: u64 = 128 << 20;
/// ...and by this many frames.
const CACHE_MAX_FRAMES: usize = 16;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// Where a clip leaves the first read error it met. The `Layer` interface can only
/// say "an effect error", so the host looks here to report an io error instead.
#[derive(Clone, Default)]
pub struct IoNote(Arc<Mutex<Option<String>>>);

impl IoNote {
    /// Keeps the first message; later ones are consequences of it.
    pub fn set(&self, message: String) {
        lock(&self.0).get_or_insert(message);
    }

    pub fn take(&self) -> Option<String> {
        lock(&self.0).take()
    }
}

struct Inner {
    file: File,
    /// Least recently used first.
    cache: Vec<(u64, Arc<Frame>)>,
    capacity: usize,
}

/// A clip of whole frames back to back in a file, in one of the wire formats.
/// Frame `i` covers the ticks `i * time_step .. (i + 1) * time_step`, like `MemLayer`.
pub struct RawClip {
    /// How messages name the clip, for example `layer overlay C:\clips\o.raw`.
    what: String,
    width: usize,
    height: usize,
    depth: Depth,
    frame_len: u64,
    frames: u64,
    time: TimeBase,
    note: IoNote,
    reads: AtomicU64,
    inner: Mutex<Inner>,
}

impl RawClip {
    /// Wraps an open file. The length must be a whole number of frames.
    pub fn new(
        file: File,
        what: impl Into<String>,
        (width, height, depth): (usize, usize, Depth),
        time: TimeBase,
        note: IoNote,
    ) -> Result<RawClip, CliError> {
        let what = what.into();
        let frame_len = (width * height * depth.bytes_per_pixel()) as u64;
        let len = file.metadata().map_err(|e| CliError::io(format!("cannot stat {what}"), e))?.len();
        let (frames, rest) = (len / frame_len, len % frame_len);
        if rest != 0 {
            return Err(CliError::Io(format!(
                "{what} ends with a partial frame ({rest} of {frame_len} bytes) after {frames} whole frames"
            )));
        }
        let capacity = ((CACHE_BYTES / frame_len) as usize).clamp(2, CACHE_MAX_FRAMES);
        Ok(RawClip {
            what,
            width,
            height,
            depth,
            frame_len,
            frames,
            time,
            note,
            reads: AtomicU64::new(0),
            inner: Mutex::new(Inner { file, cache: Vec::new(), capacity }),
        })
    }

    /// Number of whole frames in the file.
    pub fn frame_count(&self) -> u64 {
        self.frames
    }

    /// How many frames were read from the file so far (cache hits do not count).
    #[cfg(test)]
    pub fn reads(&self) -> u64 {
        self.reads.load(Ordering::SeqCst)
    }

    /// Indices of the frames currently cached, least recently used first.
    #[cfg(test)]
    pub fn cached(&self) -> Vec<u64> {
        lock(&self.inner).cache.iter().map(|(i, _)| *i).collect()
    }

    fn read_failed(&self, index: u64, err: io::Error) -> FxError {
        let message = format!("cannot read frame {index} of {}: {err}", self.what);
        self.note.set(message.clone());
        FxError::Mismatch(message)
    }
}

impl Layer for RawClip {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }

    fn depth(&self) -> Depth {
        self.depth
    }

    fn frame_at(&self, time: i64) -> fx_core::Result<Option<Arc<Frame>>> {
        let Ok(index) = u64::try_from(self.time.frame_of(time)) else { return Ok(None) };
        if index >= self.frames {
            return Ok(None);
        }
        let mut inner = lock(&self.inner);
        if let Some(pos) = inner.cache.iter().position(|(i, _)| *i == index) {
            let entry = inner.cache.remove(pos);
            let frame = entry.1.clone();
            inner.cache.push(entry);
            return Ok(Some(frame));
        }
        let mut bytes = vec![0u8; self.frame_len as usize];
        inner
            .file
            .seek(SeekFrom::Start(index * self.frame_len))
            .and_then(|_| inner.file.read_exact(&mut bytes))
            .map_err(|e| self.read_failed(index, e))?;
        self.reads.fetch_add(1, Ordering::SeqCst);
        let frame = Arc::new(Frame::from_bytes(self.width, self.height, self.depth, &bytes));
        if inner.cache.len() >= inner.capacity {
            inner.cache.remove(0);
        }
        inner.cache.push((index, frame.clone()));
        Ok(Some(frame))
    }
}

/// Removes the spool file when dropped, whichever way the run ends: success, an
/// error, or a panic unwinding through the caller.
pub struct SpoolGuard {
    path: PathBuf,
}

impl SpoolGuard {
    #[cfg(test)]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for SpoolGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

static SPOOL_SERIAL: AtomicU64 = AtomicU64::new(0);

/// Chunk of one read from the pipe and one write to the spool file.
const SPOOL_CHUNK: usize = 1 << 20;

/// `io::copy` with a megabyte buffer, and an error that says which side failed.
fn copy_all(input: &mut dyn Read, file: &mut File) -> Result<(), CliError> {
    let mut buf = vec![0u8; SPOOL_CHUNK];
    loop {
        match input.read(&mut buf) {
            Ok(0) => return Ok(()),
            Ok(n) => file.write_all(&buf[..n]).map_err(|e| CliError::io("cannot write the spool file", e))?,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(CliError::io("read error while spooling the input", e)),
        }
    }
}

/// Copies `input` into a new file in `dir` and returns it with its guard. The name
/// holds the process id, the clock and a counter, and the file is created with
/// `create_new`, so parallel runs never share one.
pub fn spool(input: &mut dyn Read, dir: &Path) -> Result<(SpoolGuard, File), CliError> {
    let mut last = None;
    for _ in 0..16 {
        let serial = SPOOL_SERIAL.fetch_add(1, Ordering::SeqCst);
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        let path = dir.join(format!("vfx-spool-{}-{nanos:x}-{serial}.raw", std::process::id()));
        match OpenOptions::new().read(true).write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                let guard = SpoolGuard { path };
                copy_all(input, &mut file)?;
                return Ok((guard, file));
            }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => last = Some(e),
            Err(e) => return Err(CliError::io(format!("cannot create a spool file in {}", dir.display()), e)),
        }
    }
    Err(CliError::io(
        format!("cannot find a free spool file name in {}", dir.display()),
        last.expect("the loop ran at least once"),
    ))
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use fx_core::Rational;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    /// A scratch directory removed when it goes out of scope.
    pub struct TempDir(pub PathBuf);

    impl TempDir {
        pub fn new(tag: &str) -> TempDir {
            static N: AtomicU64 = AtomicU64::new(0);
            let dir = std::env::temp_dir().join(format!("vfx-unit-{tag}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::SeqCst)));
            std::fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }

        pub fn file(&self, name: &str, bytes: &[u8]) -> PathBuf {
            let path = self.0.join(name);
            std::fs::write(&path, bytes).unwrap();
            path
        }

        pub fn names(&self) -> Vec<String> {
            let mut names: Vec<String> =
                std::fs::read_dir(&self.0).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
            names.sort();
            names
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const SHAPE: (usize, usize, Depth) = (2, 1, Depth::U8);

    /// Frame `i` is two pixels `[i, 0, 0, 255]` and `[i, 1, 0, 255]`: 8 bytes.
    fn clip_bytes(frames: u8) -> Vec<u8> {
        (0..frames).flat_map(|i| [i, 0, 0, 255, i, 1, 0, 255]).collect()
    }

    fn open(path: &Path, time: TimeBase) -> Result<RawClip, CliError> {
        RawClip::new(File::open(path).unwrap(), "input x.raw", SHAPE, time, IoNote::default())
    }

    fn marker(clip: &RawClip, t: i64) -> Option<u8> {
        clip.frame_at(t).unwrap().map(|f| f.to_bytes()[0])
    }

    fn base() -> TimeBase {
        TimeBase::default_for_rate(Rational::new(24, 1).unwrap())
    }

    #[test]
    fn the_length_must_be_a_whole_number_of_frames() {
        let dir = TempDir::new("len");
        let ok = dir.file("ok.raw", &clip_bytes(3));
        assert_eq!(open(&ok, base()).unwrap().frame_count(), 3);
        let mut ragged = clip_bytes(3);
        ragged.extend_from_slice(&[1, 2, 3]);
        let bad = dir.file("bad.raw", &ragged);
        let err = open(&bad, base()).err().expect("a partial frame");
        assert_eq!(err.code(), 4);
        assert_eq!(err.to_string(), "input x.raw ends with a partial frame (3 of 8 bytes) after 3 whole frames");
        let empty = dir.file("empty.raw", &[]);
        let clip = open(&empty, base()).unwrap();
        assert_eq!(clip.frame_count(), 0);
        assert_eq!(marker(&clip, 0), None);
    }

    #[test]
    fn frames_follow_the_time_base_and_none_is_outside() {
        let dir = TempDir::new("time");
        let path = dir.file("c.raw", &clip_bytes(3));
        let ntsc = TimeBase::from_fps(Rational::new(30000, 1001).unwrap());
        let clip = open(&path, ntsc).unwrap();
        assert_eq!((clip.width(), clip.height(), clip.depth()), (2, 1, Depth::U8));
        assert_eq!([marker(&clip, 0), marker(&clip, 1000), marker(&clip, 1001), marker(&clip, 3002)], [Some(0), Some(0), Some(1), Some(2)]);
        assert_eq!([marker(&clip, -1), marker(&clip, 3003), marker(&clip, i64::MIN), marker(&clip, i64::MAX)], [None; 4]);
        let f = clip.frame_at(1500).unwrap().unwrap();
        assert_eq!(f.to_bytes(), [1, 0, 0, 255, 1, 1, 0, 255]);
    }

    #[test]
    fn the_cache_keeps_recent_frames_and_evicts_the_least_recently_used() {
        let dir = TempDir::new("lru");
        let path = dir.file("c.raw", &clip_bytes(40));
        let clip = open(&path, base()).unwrap();
        let at = |i: i64| marker(&clip, i * 100);
        for i in 0..16 {
            at(i);
        }
        assert_eq!(clip.reads(), 16);
        // all cached: reading them again touches the file no more
        for i in 0..16 {
            assert_eq!(at(i), Some(i as u8));
        }
        assert_eq!(clip.reads(), 16);
        // touch frame 0 so that frame 1 is the oldest, then a new frame pushes out 1
        at(0);
        assert_eq!(clip.reads(), 16);
        at(16);
        assert_eq!(clip.reads(), 17);
        assert_eq!(clip.cached().len(), 16);
        assert!(clip.cached().contains(&0) && !clip.cached().contains(&1));
        at(0);
        assert_eq!(clip.reads(), 17, "frame 0 stayed cached");
        at(1);
        assert_eq!(clip.reads(), 18, "frame 1 had to be read again");
        assert_eq!(*clip.cached().last().unwrap(), 1);
    }

    #[test]
    fn big_frames_get_a_small_cache() {
        // 4096x4096 float is 256 MiB a frame: the cache still holds two
        let frame_len = 4096u64 * 4096 * 16;
        assert_eq!(((CACHE_BYTES / frame_len) as usize).clamp(2, CACHE_MAX_FRAMES), 2);
        assert_eq!(((CACHE_BYTES / 8) as usize).clamp(2, CACHE_MAX_FRAMES), CACHE_MAX_FRAMES);
    }

    #[test]
    fn a_read_error_is_noted_for_the_host_and_reported_to_the_effect() {
        let dir = TempDir::new("err");
        let path = dir.file("c.raw", &clip_bytes(4));
        let note = IoNote::default();
        let clip = RawClip::new(File::open(&path).unwrap(), "input x.raw", SHAPE, base(), note.clone()).unwrap();
        // the file shrinks after it was opened and measured
        OpenOptions::new().write(true).open(&path).unwrap().set_len(10).unwrap();
        assert!(clip.frame_at(0).unwrap().is_some());
        let err = clip.frame_at(300).unwrap_err();
        assert!(err.to_string().contains("cannot read frame 3 of input x.raw"), "{err}");
        let noted = note.take().expect("the host is told");
        assert!(noted.contains("frame 3"), "{noted}");
        assert_eq!(note.take(), None);
        // only the first error is kept
        note.set("first".into());
        note.set("second".into());
        assert_eq!(note.take().as_deref(), Some("first"));
    }

    #[test]
    fn concurrent_readers_see_the_right_frames() {
        let dir = TempDir::new("par");
        let path = dir.file("c.raw", &clip_bytes(30));
        let clip = open(&path, base()).unwrap();
        std::thread::scope(|s| {
            for t in 0..4i64 {
                let clip = &clip;
                s.spawn(move || {
                    for round in 0..50i64 {
                        let i = (round * 7 + t * 3) % 30;
                        assert_eq!(marker(clip, i * 100), Some(i as u8));
                    }
                });
            }
        });
    }

    #[test]
    fn spooling_copies_the_bytes_and_drop_removes_the_file() {
        let dir = TempDir::new("spool");
        let data = clip_bytes(5);
        let (guard, mut file) = spool(&mut &data[..], &dir.0).unwrap();
        let name = guard.path().file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with(&format!("vfx-spool-{}-", std::process::id())) && name.ends_with(".raw"), "{name}");
        assert_eq!(dir.names(), [name]);
        let mut back = Vec::new();
        file.seek(SeekFrom::Start(0)).unwrap();
        file.read_to_end(&mut back).unwrap();
        assert_eq!(back, data);
        drop(file);
        drop(guard);
        assert!(dir.names().is_empty());
    }

    #[test]
    fn spool_names_never_collide() {
        let dir = TempDir::new("names");
        let guards: Vec<_> = (0..20).map(|_| spool(&mut io::empty(), &dir.0).unwrap()).collect();
        assert_eq!(dir.names().len(), 20);
        drop(guards);
        assert!(dir.names().is_empty());
    }

    #[test]
    fn a_failed_copy_and_a_missing_directory_leave_nothing_behind() {
        struct Dies(usize);
        impl Read for Dies {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                if self.0 == 0 {
                    return Err(io::Error::new(io::ErrorKind::BrokenPipe, "pipe closed"));
                }
                let n = self.0.min(buf.len());
                buf[..n].fill(7);
                self.0 -= n;
                Ok(n)
            }
        }
        let dir = TempDir::new("fail");
        let err = spool(&mut Dies(100), &dir.0).err().expect("the copy fails");
        assert_eq!(err.code(), 4);
        assert!(dir.names().is_empty());
        let err = spool(&mut io::empty(), &dir.0.join("missing")).err().expect("no such directory");
        assert_eq!(err.code(), 4);
        assert!(err.to_string().contains("cannot create a spool file"), "{err}");
    }

    #[test]
    fn a_panic_unwinding_through_the_caller_still_removes_the_spool() {
        let dir = TempDir::new("panic");
        let result = catch_unwind(AssertUnwindSafe(|| {
            let (_guard, mut file) = spool(&mut &clip_bytes(2)[..], &dir.0).unwrap();
            file.write_all(b"more").unwrap();
            assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 1);
            panic!("the render blew up");
        }));
        assert!(result.is_err());
        assert!(dir.names().is_empty(), "{:?}", dir.names());
    }
}
