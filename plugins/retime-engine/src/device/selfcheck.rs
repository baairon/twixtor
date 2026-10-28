//! The device gate: before a device renders anything, the starter kernels run on it over an
//! adversarial corpus and every output is compared bit for bit with the CPU code; any difference
//! refuses the device and names the first failing input (the first half of the gate; the second
//! half, the first vector frame of a run rendered on both, belongs to the executor).
//!
//! The corpus, per kernel and depth: every u8 pair; u16 pairs at the edges of the range plus a
//! seeded sample; f32 values with subnormals, -0.0, NaN (quiet and with payloads), +-inf, the
//! largest finite values, exact .5 ties after the `+ 0.5f`, exact -1.0, out of range casts on both
//! sides of +-2^31 and +-2^63, and products where a fused multiply add rounds differently from a
//! multiply and an add (found with `f32::mul_add` when the corpus is built); doubles at the cast
//! boundaries; and a dependent loop of 64 steps. Weights stay in the domain of branch 4 of
//! `blend::blend` (the branch the device kernel ports), NaN included.
//!
//! The CPU references are the engine functions themselves (`blend::copy`, `blend::blend`,
//! `blend::cvtt`, `warp::cvtt32`, `warp::sat_u8`, `warp::sat_u16`, `image::q`,
//! `timeline::cvtt_i32`); the double to 64 bit cast, private in the engine (`image.rs`, `warp.rs`,
//! `vectors.rs`), is restated in [`cvtt_i64_d`] and tested against `image::q`. Every launch is a
//! few hundred thousand work items at most.

use crate::blend;
use crate::device::cl::{self, Access, Arg, Buffer, ClError, Context, Kernel, Program, Queue, Range};
use crate::device::kernels::{self, starter, BUILD_OPTIONS};
use crate::device::probe;
use crate::device::sim::Rng;
use crate::image::{Image, Samples};
use std::fmt;

/// The first output that differs from the CPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mismatch {
    /// The kernel function.
    pub kernel: &'static str,
    /// The launch within the kernel's corpus (for example the weight).
    pub case: String,
    /// The output element.
    pub index: usize,
    pub input: String,
    pub expected: String,
    pub got: String,
}

impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {}: output {} for {}: expected {}, got {}",
            self.kernel, self.case, self.index, self.input, self.expected, self.got
        )
    }
}

/// Why a device is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The probe requirements do not hold; nothing ran.
    Requirements(Vec<String>),
    /// The driver failed: no context, a build error, a failed launch or read.
    Driver(ClError),
    /// A kernel gave other bits than the CPU.
    Mismatch(Mismatch),
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::Requirements(r) => write!(f, "{}", r.join("; ")),
            Refusal::Driver(e) => write!(f, "driver: {e}"),
            Refusal::Mismatch(m) => write!(f, "differs from the CPU: {m}"),
        }
    }
}

/// How the device's f32 NaN results compare with the CPU's. IEEE 754 fixes which operations give
/// NaN but not the payload: x86 keeps the first NaN operand's payload and gives `0xffc00000` for an
/// invalid operation, other hardware may not. A NaN never changes a comparison or a conversion, so
/// payloads only show where a NaN is stored as an f32 output. The self-check requires NaN in exactly
/// the places the CPU has one and reports the payloads here; with [`NanPayloads::Differ`] the
/// executor must recompute on the CPU every f32 output that holds a NaN.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NanPayloads {
    Same,
    /// The first output where both are NaN with other bits.
    Differ(Mismatch),
}

/// What the checks of one source found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tally {
    /// Outputs compared.
    pub outputs: usize,
    pub nan_payloads: NanPayloads,
}

/// The result of [`selfcheck`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Every output of every check matched, NaN payloads aside (see [`NanPayloads`]).
    Usable(Tally),
    Refused(Refusal),
}

impl Verdict {
    pub fn is_usable(&self) -> bool {
        matches!(self, Verdict::Usable(_))
    }

    /// Usable with the CPU's bits everywhere, NaN payloads included.
    pub fn is_bit_exact(&self) -> bool {
        matches!(self, Verdict::Usable(Tally { nan_payloads: NanPayloads::Same, .. }))
    }
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Verdict::Usable(Tally { outputs, nan_payloads: NanPayloads::Same }) => {
                write!(f, "ok ({outputs} outputs equal to the CPU)")
            }
            Verdict::Usable(Tally { outputs, nan_payloads: NanPayloads::Differ(m) }) => {
                write!(f, "ok ({outputs} outputs equal to the CPU; NaN payloads differ, first: {m})")
            }
            Verdict::Refused(r) => write!(f, "refused: {r}"),
        }
    }
}

/// The gate: the probe requirements, then the starter program built with [`BUILD_OPTIONS`] and
/// every check of [`check_source`].
pub fn selfcheck(device: &cl::Device) -> Verdict {
    let info = probe::read(device, 0, "", "");
    let reasons = probe::requirements(&info);
    if !reasons.is_empty() {
        return Verdict::Refused(Refusal::Requirements(reasons));
    }
    match check_source(device, &kernels::starter_program()) {
        Ok(Ok(tally)) => Verdict::Usable(tally),
        Ok(Err(m)) => Verdict::Refused(Refusal::Mismatch(m)),
        Err(e) => Verdict::Refused(Refusal::Driver(e)),
    }
}

/// Builds `source` (which must define every starter kernel) on `device` without the probe gate and
/// runs every check: `Ok(Ok(tally))` when every output matches (NaN payloads aside), `Ok(Err(m))`
/// at the first mismatch.
/// The test that a contracting build is caught passes a source with `FP_CONTRACT ON` here.
pub fn check_source(device: &cl::Device, source: &str) -> Result<Result<Tally, Mismatch>, ClError> {
    check_source_with(device, source, BUILD_OPTIONS)
}

/// [`check_source`] with other build options (the tests show that relaxing options are caught).
pub fn check_source_with(device: &cl::Device, source: &str, options: &str) -> Result<Result<Tally, Mismatch>, ClError> {
    let session = Session::open_with(device, source, options)?;
    let mut outputs = 0usize;
    let mut nan_payloads = NanPayloads::Same;
    for check in [check_copy, check_blend_u8, check_blend_u16, check_blend_f32, check_convert_f32, check_convert_f64, check_index_2d, check_chain] {
        match check(&session)? {
            Ok(t) => {
                outputs += t.outputs;
                if let (NanPayloads::Same, NanPayloads::Differ(m)) = (&nan_payloads, t.nan_payloads) {
                    nan_payloads = NanPayloads::Differ(m);
                }
            }
            Err(m) => return Ok(Err(m)),
        }
    }
    Ok(Ok(Tally { outputs, nan_payloads }))
}

/// A context, an in-order queue and one built program on a device.
pub struct Session {
    pub context: Context,
    pub queue: Queue,
    pub program: Program,
}

impl Session {
    /// Builds `source` with [`BUILD_OPTIONS`].
    pub fn open(device: &cl::Device, source: &str) -> Result<Session, ClError> {
        Session::open_with(device, source, BUILD_OPTIONS)
    }

    /// Builds `source` with `options`.
    pub fn open_with(device: &cl::Device, source: &str, options: &str) -> Result<Session, ClError> {
        let context = Context::new(device)?;
        let queue = Queue::new(&context)?;
        let program = super::progcache::build(&context, device, source, options)?;
        Ok(Session { context, queue, program })
    }

    pub fn kernel(&self, name: &str) -> Result<Kernel, ClError> {
        Kernel::new(&self.program, name)
    }

    /// Uploads `data` read only.
    pub fn input<T: cl::Pod>(&self, data: &[T]) -> Result<Buffer<T>, ClError> {
        Buffer::from_slice(&self.context, data, Access::ReadOnly)
    }

    pub fn output<T: cl::Pod>(&self, len: usize) -> Result<Buffer<T>, ClError> {
        Buffer::new(&self.context, len, Access::WriteOnly)
    }
}

/// Values compared by their bits.
pub trait Bits: Copy {
    fn bits(self) -> u64;
    fn show(self) -> String;
    fn is_nan(self) -> bool {
        false
    }
}

impl Bits for u8 {
    fn bits(self) -> u64 {
        self as u64
    }
    fn show(self) -> String {
        self.to_string()
    }
}

impl Bits for u16 {
    fn bits(self) -> u64 {
        self as u64
    }
    fn show(self) -> String {
        self.to_string()
    }
}

impl Bits for u32 {
    fn bits(self) -> u64 {
        self as u64
    }
    fn show(self) -> String {
        format!("0x{self:08x}")
    }
}

impl Bits for i64 {
    fn bits(self) -> u64 {
        self as u64
    }
    fn show(self) -> String {
        self.to_string()
    }
}

impl Bits for f32 {
    fn bits(self) -> u64 {
        self.to_bits() as u64
    }
    fn show(self) -> String {
        format!("0x{:08x} ({self:e})", self.to_bits())
    }
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
}

impl Bits for f64 {
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn show(self) -> String {
        format!("0x{:016x} ({self:e})", self.to_bits())
    }
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
}

/// The first index where `got` differs from `expected` in its bits (or in length), and apart from
/// that the first index where both are NaN with other payloads.
pub fn compare_floats<T: Bits>(expected: &[T], got: &[T]) -> (Option<usize>, Option<usize>) {
    if expected.len() != got.len() {
        return (Some(expected.len().min(got.len())), None);
    }
    let mut payload = None;
    for (i, (e, g)) in expected.iter().zip(got).enumerate() {
        if e.bits() == g.bits() {
            continue;
        }
        if e.is_nan() && g.is_nan() {
            payload.get_or_insert(i);
            continue;
        }
        return (Some(i), payload);
    }
    (None, payload)
}

/// The first index where `got` differs from `expected` in its bits (or in length).
pub fn first_difference<T: Bits>(expected: &[T], got: &[T]) -> Option<usize> {
    if expected.len() != got.len() {
        return Some(expected.len().min(got.len()));
    }
    expected.iter().zip(got).position(|(e, g)| e.bits() != g.bits())
}

fn mismatch<T: Bits>(kernel: &'static str, case: String, index: usize, input: String, expected: &[T], got: &[T]) -> Mismatch {
    let show = |v: &[T]| v.get(index).map(|x| x.show()).unwrap_or_else(|| "nothing".into());
    Mismatch { kernel, case, index, input, expected: show(expected), got: show(got) }
}

/// Runs a 1D kernel whose last argument is the item count `n` and reads `out` back.
fn run_1d<T: cl::Pod>(s: &Session, kernel: &mut Kernel, n: usize, args: &[Arg<'_>], out: &Buffer<T>) -> Result<Vec<T>, ClError> {
    let mut all: Vec<Arg<'_>> = args.to_vec();
    all.push(Arg::U32(n as u32));
    s.queue.run(kernel, Range::D1(n), &all)?;
    s.queue.read(out)
}

// ---- corpus -----------------------------------------------------------------------------------

/// f32 values every float check includes.
pub fn f32_specials() -> Vec<f32> {
    let mut v = vec![
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        0.1,
        1.0 / 3.0,
        f32::from_bits(1),           // smallest subnormal
        f32::from_bits(0x8000_0001), // its negative
        f32::from_bits(0x007f_ffff), // largest subnormal
        f32::from_bits(0x0040_0000),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc0_0000), // quiet NaN
        f32::from_bits(0xffc0_0000), // negative quiet NaN
        f32::from_bits(0x7fc0_1234), // quiet NaN with a payload
        f32::from_bits(0x7fa0_0001), // signalling NaN pattern
        255.0,
        65535.0,
        1.0e30,
        -1.0e30,
    ];
    v.extend([f32::from_bits(0x3f7f_ffff), f32::from_bits(0x3f80_0001)]); // 1.0 -+ 1 ulp
    v
}

/// Inputs of the conversion checks.
pub fn convert_f32_inputs() -> Vec<f32> {
    let mut v = f32_specials();
    let edges = [2147483648.0f32, 9223372036854775808.0f32, 65536.0, 256.0, 128.0, 32768.0];
    for e in edges {
        for x in [e, -e] {
            v.extend([x, f32::from_bits(x.to_bits() - 1), f32::from_bits(x.to_bits() + 1)]);
        }
    }
    // exact ties and near ties around the + 0.5 of every store
    for t in [0.5f32, 1.5, 2.5, 127.5, 254.5, 255.5, 256.5, 32767.5, 65534.5, 65535.5, 8388607.5, -0.5, -1.5, -127.5, -128.5] {
        v.extend([t, f32::from_bits(t.to_bits() - 1), f32::from_bits(t.to_bits() + 1)]);
    }
    v.extend([0.49999997f32, -0.49999997, 16777216.0, 16777215.0, -16777216.0, -1.0, -2.0, 4294967296.0]);
    for i in -2048i32..=2048 {
        v.push(i as f32 / 4.0);
    }
    for i in -300i32..=70_000 {
        if i % 7 == 0 || !(-2..65_538).contains(&i) {
            v.push(i as f32);
        }
    }
    let mut rng = Rng::new(0xC0DE_0001);
    for _ in 0..4096 {
        let bits = rng.next_u32();
        v.push(f32::from_bits(bits));
    }
    for _ in 0..2048 {
        // magnitudes from 2^-10 to 2^70, both signs
        let exp = 117 + rng.below(80) as u32;
        let bits = (rng.next_u32() & 0x8000_0000) | (exp << 23) | (rng.next_u32() & 0x007f_ffff);
        v.push(f32::from_bits(bits));
    }
    v
}

/// Inputs of the double conversions.
pub fn convert_f64_inputs() -> Vec<f64> {
    let mut v = vec![
        0.0f64,
        -0.0,
        0.5,
        -0.5,
        0.9999999999999999,
        -0.9999999999999999,
        2147483647.0,
        2147483647.9999998,
        2147483648.0,
        -2147483648.0,
        -2147483648.9999998,
        -2147483649.0,
        -2147483649.0000005,
        9223372036854775808.0,
        -9223372036854775808.0,
        f64::from_bits(9223372036854775808.0f64.to_bits() - 1),
        f64::from_bits((-9223372036854775808.0f64).to_bits() - 1),
        f64::from_bits((-9223372036854775808.0f64).to_bits() + 1),
        f64::MAX,
        f64::MIN,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::from_bits(0x8000_0000_0000_0001),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::from_bits(0x7ff0_0000_0000_0001),
        -1.0,
    ];
    let mut rng = Rng::new(0xC0DE_0002);
    for _ in 0..2048 {
        let exp = 1000 + rng.below(80);
        let bits = (rng.next_u64() & (1 << 63)) | (exp << 52) | (rng.next_u64() & ((1 << 52) - 1));
        v.push(f64::from_bits(bits));
    }
    for _ in 0..1024 {
        v.push(f64::from_bits(rng.next_u64()));
    }
    v
}

/// Whether `w` reaches branch 4 of `blend::blend` (no snap copy, no extrapolation).
pub fn in_blend_branch4(w: f32) -> bool {
    let snap = 0.005f64;
    !(snap > f64::from(w.abs())) && !(snap > (f64::from(w) - 1.0).abs()) && !(w < 0.0 || w > 1.0)
}

/// The weights of the blend checks, all in branch 4 (NaN included, it fails every comparison).
pub fn blend_weights() -> Vec<f32> {
    let w = vec![
        f32::from_bits(0.005f32.to_bits() + 1),
        0.0625,
        0.1,
        0.25,
        1.0 / 3.0,
        0.5,
        0.7,
        0.9,
        f32::from_bits(0x3f7e_b852), // about 0.995, just inside
        f32::NAN,
    ];
    w.into_iter().filter(|w| in_blend_branch4(*w)).collect()
}

/// Every 8 bit pair `(a, b)`.
pub fn blend_u8_pairs() -> (Vec<u8>, Vec<u8>) {
    let mut a = Vec::with_capacity(65536);
    let mut b = Vec::with_capacity(65536);
    for x in 0..=255u8 {
        for y in 0..=255u8 {
            a.push(x);
            b.push(y);
        }
    }
    (a, b)
}

/// 16 bit pairs: every pair of edge values plus a seeded sample; a multiple of 4 long.
pub fn blend_u16_pairs() -> (Vec<u16>, Vec<u16>) {
    let edges = [0u16, 1, 2, 127, 128, 255, 256, 4095, 16384, 32767, 32768, 32769, 49151, 65278, 65534, 65535];
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for x in edges {
        for y in edges {
            a.push(x);
            b.push(y);
        }
    }
    let mut rng = Rng::new(0xB1E0_0016);
    while a.len() < 16384 {
        a.push(rng.next_u32() as u16);
        b.push(rng.next_u32() as u16);
    }
    (a, b)
}

/// f32 pairs: every pair of specials, pairs where a fused multiply add would round `(b - a) * w + a`
/// differently for the weight `w` (`fma_for`), and a seeded sample; a multiple of 4 long.
pub fn blend_f32_pairs(fma_for: f32) -> (Vec<f32>, Vec<f32>) {
    let s = f32_specials();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for x in &s {
        for y in &s {
            a.push(*x);
            b.push(*y);
        }
    }
    let mut rng = Rng::new(0xF3A0_0032 ^ fma_for.to_bits() as u64);
    let mut fused = 0;
    let mut tries = 0;
    // a power of two weight makes the product exact: nothing to find
    let inexact = fma_for.is_finite() && fma_for.to_bits() & 0x007f_ffff != 0;
    while inexact && fused < 256 && tries < 1_000_000 {
        tries += 1;
        let x = (rng.next_u32() >> 8) as f32 / 16777216.0;
        let y = (rng.next_u32() >> 8) as f32 / 16777216.0;
        let d = y - x;
        if d.mul_add(fma_for, x) != d * fma_for + x {
            a.push(x);
            b.push(y);
            fused += 1;
        }
    }
    for _ in 0..1024 {
        let x = f32::from_bits(rng.next_u32() & 0x7fff_ffff);
        let y = f32::from_bits(rng.next_u32());
        a.push(if x.is_finite() { x } else { 1.0 });
        b.push(y);
    }
    while a.len() % 4 != 0 {
        a.push(-1.0);
        b.push(-1.0);
    }
    (a, b)
}

/// Inputs of the dependent loop `v = v * k + x`: subnormal, tiny, ordinary and FMA sensitive
/// values, with `k` below 1 in magnitude so the loop stays finite.
pub fn chain_inputs() -> (Vec<f32>, Vec<f32>) {
    let (mut x, mut k) = (Vec::new(), Vec::new());
    for xi in [f32::from_bits(1), f32::from_bits(0x0000_1234), f32::from_bits(0x007f_ffff), f32::MIN_POSITIVE, -1.0, 0.0, -0.0, 1.0e-30, 0.1, 1.0 / 3.0] {
        for ki in [0.5f32, -0.5, 0.75, 0.999, 1.0 / 3.0, 0.1, f32::from_bits(0x3f7f_ffff), 0.0] {
            x.push(xi);
            k.push(ki);
        }
    }
    let mut rng = Rng::new(0xC4A1_0064);
    while x.len() < 4096 {
        x.push((rng.next_u32() >> 8) as f32 / 16777216.0 - 0.5);
        k.push((rng.next_u32() >> 8) as f32 / 16777216.0 * 1.8 - 0.9);
    }
    (x, k)
}

// ---- CPU references ---------------------------------------------------------------------------

/// Truncation of a double toward zero, `i64::MIN` out of range and for NaN: the rule of the
/// private `cvtt_i64` of `image.rs` (and `cvtt64` of `warp.rs` and `vectors.rs`).
pub fn cvtt_i64_d(x: f64) -> i64 {
    if x.is_nan() || x >= 9.223372036854775808e18 || x < -9.223372036854775808e18 {
        i64::MIN
    } else {
        x as i64
    }
}

/// `blend::blend` over a 1 x n/4 image of the samples.
pub fn cpu_blend<T: Copy>(a: &[T], b: &[T], w: f32, wrap: fn(Vec<T>) -> Samples, unwrap: fn(Samples) -> Vec<T>) -> Vec<T> {
    let n = a.len();
    assert_eq!(n % 4, 0);
    let img = |v: &[T]| Image { width: n / 4, height: 1, samples: wrap(v.to_vec()) };
    let (ia, ib) = (img(a), img(b));
    let mut out = img(a);
    blend::blend(&mut out, &ia, &ib, w);
    unwrap(out.samples)
}

pub fn u8_samples(v: Vec<u8>) -> Samples {
    Samples::U8(v)
}
pub fn u16_samples(v: Vec<u16>) -> Samples {
    Samples::U16(v)
}
pub fn f32_samples(v: Vec<f32>) -> Samples {
    Samples::F32(v)
}
pub fn samples_u8(s: Samples) -> Vec<u8> {
    match s {
        Samples::U8(v) => v,
        _ => unreachable!("an 8 bit image"),
    }
}
pub fn samples_u16(s: Samples) -> Vec<u16> {
    match s {
        Samples::U16(v) => v,
        _ => unreachable!("a 16 bit image"),
    }
}
pub fn samples_f32(s: Samples) -> Vec<f32> {
    match s {
        Samples::F32(v) => v,
        _ => unreachable!("a float image"),
    }
}

/// The nine outputs of `convert_f32` for `x`, from the engine's own helpers.
pub fn cpu_convert_f32(x: f32) -> [i64; starter::CONVERT_F32_OUTPUTS] {
    [
        crate::warp::cvtt32(x) as i64,
        blend::cvtt(x),
        crate::image::q(x),
        crate::image::q(x) as u8 as i64,
        crate::image::q(x) as u16 as i64,
        blend::cvtt(x) as u8 as i64,
        blend::cvtt(x) as u16 as i64,
        crate::warp::sat_u8(x) as i64,
        crate::warp::sat_u16(x) as i64,
    ]
}

/// The two outputs of `convert_f64` for `x`.
pub fn cpu_convert_f64(x: f64) -> [i64; starter::CONVERT_F64_OUTPUTS] {
    [crate::timeline::cvtt_i32(x) as i64, cvtt_i64_d(x)]
}

/// The dependent loop in f32, in the order written.
pub fn cpu_chain(x: f32, k: f32) -> f32 {
    let mut v = x;
    for _ in 0..starter::CHAIN_STEPS {
        v = v * k + x;
    }
    v
}

/// A bit copy through `blend::copy`.
pub fn cpu_copy<T: Copy>(src: &[T], wrap: fn(Vec<T>) -> Samples, unwrap: fn(Samples) -> Vec<T>) -> Vec<T> {
    let n = src.len();
    let img = |v: Vec<T>| Image { width: n / 4, height: 1, samples: wrap(v) };
    let s = img(src.to_vec());
    let mut out = img(src.to_vec());
    out.clear();
    blend::copy(&mut out, &s);
    unwrap(out.samples)
}

// ---- checks -----------------------------------------------------------------------------------

type Outcome = Result<Result<Tally, Mismatch>, ClError>;

fn exact(outputs: usize) -> Outcome {
    Ok(Ok(Tally { outputs, nan_payloads: NanPayloads::Same }))
}

fn check_copy(s: &Session) -> Outcome {
    let mut rng = Rng::new(0xC097_0001);
    let n = 4096;
    let u8s: Vec<u8> = (0..n).map(|i| if i < 256 { i as u8 } else { rng.next_u32() as u8 }).collect();
    let u16s: Vec<u16> = (0..n).map(|i| if i < 1024 { (i * 64) as u16 } else { rng.next_u32() as u16 }).collect();
    let mut f32s: Vec<f32> = f32_specials();
    while f32s.len() < n {
        f32s.push(f32::from_bits(rng.next_u32()));
    }
    let mut total = 0;
    {
        let want = cpu_copy(&u8s, u8_samples, samples_u8);
        let (src, dst) = (s.input(&u8s)?, s.output::<u8>(n)?);
        let got = run_1d(s, &mut s.kernel("copy_u8")?, n, &[src.arg(), dst.arg()], &dst)?;
        if let Some(i) = first_difference(&want, &got) {
            return Ok(Err(mismatch("copy_u8", "bytes".into(), i, u8s[i].to_string(), &want, &got)));
        }
        total += n;
    }
    {
        let want = cpu_copy(&u16s, u16_samples, samples_u16);
        let (src, dst) = (s.input(&u16s)?, s.output::<u16>(n)?);
        let got = run_1d(s, &mut s.kernel("copy_u16")?, n, &[src.arg(), dst.arg()], &dst)?;
        if let Some(i) = first_difference(&want, &got) {
            return Ok(Err(mismatch("copy_u16", "words".into(), i, u16s[i].to_string(), &want, &got)));
        }
        total += n;
    }
    {
        let want: Vec<u32> = cpu_copy(&f32s, f32_samples, samples_f32).into_iter().map(f32::to_bits).collect();
        let bits: Vec<u32> = f32s.iter().map(|v| v.to_bits()).collect();
        let (src, dst) = (s.input(&bits)?, s.output::<u32>(n)?);
        let got = run_1d(s, &mut s.kernel("copy_u32")?, n, &[src.arg(), dst.arg()], &dst)?;
        if let Some(i) = first_difference(&want, &got) {
            return Ok(Err(mismatch("copy_u32", "float bits".into(), i, f32s[i].show(), &want, &got)));
        }
        total += n;
    }
    exact(total)
}

fn check_blend_u8(s: &Session) -> Outcome {
    let (a, b) = blend_u8_pairs();
    let n = a.len();
    let (ba, bb, out) = (s.input(&a)?, s.input(&b)?, s.output::<u8>(n)?);
    let mut k = s.kernel("blend_u8")?;
    let mut total = 0;
    for w in blend_weights() {
        let want = cpu_blend(&a, &b, w, u8_samples, samples_u8);
        let got = run_1d(s, &mut k, n, &[ba.arg(), bb.arg(), out.arg(), Arg::F32(w)], &out)?;
        if let Some(i) = first_difference(&want, &got) {
            return Ok(Err(mismatch("blend_u8", format!("w={}", w.show()), i, format!("a={} b={}", a[i], b[i]), &want, &got)));
        }
        total += n;
    }
    exact(total)
}

fn check_blend_u16(s: &Session) -> Outcome {
    let (a, b) = blend_u16_pairs();
    let n = a.len();
    let (ba, bb, out) = (s.input(&a)?, s.input(&b)?, s.output::<u16>(n)?);
    let mut k = s.kernel("blend_u16")?;
    let mut total = 0;
    for w in blend_weights() {
        let want = cpu_blend(&a, &b, w, u16_samples, samples_u16);
        let got = run_1d(s, &mut k, n, &[ba.arg(), bb.arg(), out.arg(), Arg::F32(w)], &out)?;
        if let Some(i) = first_difference(&want, &got) {
            return Ok(Err(mismatch("blend_u16", format!("w={}", w.show()), i, format!("a={} b={}", a[i], b[i]), &want, &got)));
        }
        total += n;
    }
    exact(total)
}

fn check_blend_f32(s: &Session) -> Outcome {
    let mut k = s.kernel("blend_f32")?;
    let mut total = 0;
    let mut nan_payloads = NanPayloads::Same;
    for w in blend_weights() {
        let (a, b) = blend_f32_pairs(w);
        let n = a.len();
        let (ba, bb, out) = (s.input(&a)?, s.input(&b)?, s.output::<f32>(n)?);
        let want = cpu_blend(&a, &b, w, f32_samples, samples_f32);
        let got = run_1d(s, &mut k, n, &[ba.arg(), bb.arg(), out.arg(), Arg::F32(w)], &out)?;
        let (differ, payload) = compare_floats(&want, &got);
        let at = |i: usize| (format!("w={}", w.show()), i, format!("a={} b={}", a[i].show(), b[i].show()));
        if let Some(i) = differ {
            let (case, i, input) = at(i);
            return Ok(Err(mismatch("blend_f32", case, i, input, &want, &got)));
        }
        if let (Some(i), NanPayloads::Same) = (payload, &nan_payloads) {
            let (case, i, input) = at(i);
            nan_payloads = NanPayloads::Differ(mismatch("blend_f32", case, i, input, &want, &got));
        }
        total += n;
    }
    Ok(Ok(Tally { outputs: total, nan_payloads }))
}

fn check_convert_f32(s: &Session) -> Outcome {
    let x = convert_f32_inputs();
    let n = x.len();
    let want: Vec<i64> = x.iter().flat_map(|v| cpu_convert_f32(*v)).collect();
    let (bx, out) = (s.input(&x)?, s.output::<i64>(n * starter::CONVERT_F32_OUTPUTS)?);
    let got = run_1d(s, &mut s.kernel("convert_f32")?, n, &[bx.arg(), out.arg()], &out)?;
    if let Some(i) = first_difference(&want, &got) {
        let which = CONVERT_F32_NAMES[i % starter::CONVERT_F32_OUTPUTS];
        return Ok(Err(mismatch("convert_f32", which.into(), i, x[i / starter::CONVERT_F32_OUTPUTS].show(), &want, &got)));
    }
    exact(want.len())
}

/// The helper behind each output of `convert_f32`, in order.
pub const CONVERT_F32_NAMES: [&str; starter::CONVERT_F32_OUTPUTS] =
    ["cvtt_i32_f", "cvtt_i64_f", "q_round", "store_q_u8", "store_q_u16", "trunc_u8_f", "trunc_u16_f", "sat_u8", "sat_u16"];

fn check_convert_f64(s: &Session) -> Outcome {
    let x = convert_f64_inputs();
    let n = x.len();
    let want: Vec<i64> = x.iter().flat_map(|v| cpu_convert_f64(*v)).collect();
    let (bx, out) = (s.input(&x)?, s.output::<i64>(n * starter::CONVERT_F64_OUTPUTS)?);
    let got = run_1d(s, &mut s.kernel("convert_f64")?, n, &[bx.arg(), out.arg()], &out)?;
    if let Some(i) = first_difference(&want, &got) {
        let which = ["cvtt_i32_d", "cvtt_i64_d"][i % 2];
        return Ok(Err(mismatch("convert_f64", which.into(), i, x[i / 2].show(), &want, &got)));
    }
    exact(want.len())
}

/// The 2D layout: `get_global_id(0)` runs along a row.
pub const INDEX_SIZE: (usize, usize) = (37, 19);

fn check_index_2d(s: &Session) -> Outcome {
    let (w, h) = INDEX_SIZE;
    let want: Vec<u32> = (0..(w * h) as u32).collect();
    let out = s.output::<u32>(w * h)?;
    s.queue.fill(&out, u32::MAX)?;
    s.queue.run(&mut s.kernel("index_2d")?, Range::D2(w, h), &[out.arg(), Arg::U32(w as u32), Arg::U32(h as u32)])?;
    let got = s.queue.read(&out)?;
    if let Some(i) = first_difference(&want, &got) {
        return Ok(Err(mismatch("index_2d", format!("{w}x{h}"), i, format!("x={} y={}", i % w, i / w), &want, &got)));
    }
    exact(w * h)
}

fn check_chain(s: &Session) -> Outcome {
    let (x, k) = chain_inputs();
    let n = x.len();
    let want: Vec<f32> = x.iter().zip(&k).map(|(x, k)| cpu_chain(*x, *k)).collect();
    let (bx, bk, out) = (s.input(&x)?, s.input(&k)?, s.output::<f32>(n)?);
    let got = run_1d(s, &mut s.kernel("chain_f32")?, n, &[bx.arg(), bk.arg(), out.arg()], &out)?;
    let (differ, payload) = compare_floats(&want, &got);
    let input = |i: usize| format!("x={} k={}", x[i].show(), k[i].show());
    if let Some(i) = differ {
        return Ok(Err(mismatch("chain_f32", "64 steps".into(), i, input(i), &want, &got)));
    }
    let nan_payloads = match payload {
        Some(i) => NanPayloads::Differ(mismatch("chain_f32", "64 steps".into(), i, input(i), &want, &got)),
        None => NanPayloads::Same,
    };
    Ok(Ok(Tally { outputs: n, nan_payloads }))
}
