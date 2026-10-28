//! A work item simulator: runs a Rust closure once per work item of a 1D or 2D range, in a
//! shuffled order drawn from a seed, so a kernel's Rust transcription can be checked against the
//! CPU function without a device.
//!
//! A device runs work items in no particular order and in parallel, so a correct kernel gives the
//! same bytes for every order. The simulator makes the order explicit: [`launch_1d`] and
//! [`launch_2d`] visit every item exactly once in the order of [`order`] (identity for seed 0, a
//! Fisher-Yates shuffle driven by SplitMix64 otherwise), and [`Global`] stands for a `__global`
//! output buffer that refuses a second write to one element, since two work items writing one
//! element is a race on a device. Integer arithmetic only.

/// SplitMix64: a small, fixed pseudo random sequence (also used to build test corpora).
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// A value in `0..n` (`n > 0`), by rejection so every value is equally likely.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "empty range");
        let zone = u64::MAX - u64::MAX % n;
        loop {
            let v = self.next_u64();
            if v < zone {
                return v % n;
            }
        }
    }
}

/// The visiting order of `n` work items for `seed`: `0..n` for seed 0, else a seeded shuffle.
pub fn order(n: usize, seed: u64) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    if seed == 0 {
        return v;
    }
    let mut rng = Rng::new(seed);
    for i in (1..n).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        v.swap(i, j);
    }
    v
}

/// Runs `item(gid)` for every `gid` in `0..n`, in the order of `seed`.
pub fn launch_1d(n: usize, seed: u64, mut item: impl FnMut(usize)) {
    for gid in order(n, seed) {
        item(gid);
    }
}

/// Runs `item(x, y)` for every `x` in `0..w` and `y` in `0..h`, in the order of `seed` over the
/// `w * h` items.
pub fn launch_2d(w: usize, h: usize, seed: u64, mut item: impl FnMut(usize, usize)) {
    if w == 0 {
        return;
    }
    for g in order(w * h, seed) {
        item(g % w, g / w);
    }
}

/// The seeds the order independence checks use: identity first, then shuffles.
pub const SEEDS: [u64; 4] = [0, 1, 0x5EED, 0xC0FF_EE00_D15E_A5E5];

/// Runs `run(seed)` for every seed of [`SEEDS`] and returns the common result; panics with the
/// seed when one order gives another result (the transcription depends on the order).
pub fn same_for_every_order<T: PartialEq + std::fmt::Debug>(run: impl Fn(u64) -> T) -> T {
    let first = run(SEEDS[0]);
    for seed in &SEEDS[1..] {
        let other = run(*seed);
        assert!(other == first, "the result depends on the work item order (seed {seed:#x})");
    }
    first
}

/// A simulated `__global` output buffer: reads see what was written, a second write to one
/// element panics (a race between work items), an index out of range panics.
#[derive(Clone, Debug)]
pub struct Global<T> {
    data: Vec<T>,
    written: Vec<bool>,
}

impl<T: Copy> Global<T> {
    /// `len` elements, all `fill`, none written yet.
    pub fn new(len: usize, fill: T) -> Global<T> {
        Global { data: vec![fill; len], written: vec![false; len] }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get(&self, i: usize) -> T {
        assert!(i < self.data.len(), "read of element {i} of a buffer of {}", self.data.len());
        self.data[i]
    }

    /// Writes element `i`; a second write to it panics.
    pub fn set(&mut self, i: usize, v: T) {
        assert!(i < self.data.len(), "write of element {i} of a buffer of {}", self.data.len());
        assert!(!self.written[i], "element {i} written by two work items");
        self.written[i] = true;
        self.data[i] = v;
    }

    /// How many elements were written.
    pub fn written(&self) -> usize {
        self.written.iter().filter(|w| **w).count()
    }

    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}
