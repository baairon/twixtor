//! The device frame cache: resident device data keyed by the exact host data it was made from,
//! least recently used first out, inside a byte budget.
//!
//! A key is compared bit for bit ([`ImageKey`]: size, format and every sample's bits, so `-0.0`
//! and `0.0` are different keys and a NaN matches only the same NaN), never by a hash alone: a
//! hit is always data made from these very samples. No arithmetic on samples.

use crate::flow::FlowSettings;
use crate::image::{Image, Samples};

/// What a cached entry was made from: the host image (kept for the exact comparison) and the
/// settings that shaped the entry.
pub struct ImageKey {
    pub image: Image,
    pub settings: FlowSettings,
    /// A quick filter over a few samples; equal keys always have equal digests.
    digest: u64,
}

impl ImageKey {
    pub fn new(image: &Image, settings: &FlowSettings) -> ImageKey {
        ImageKey { image: image.clone(), settings: *settings, digest: digest(image) }
    }

    /// True when `image` and `settings` are bit for bit the ones of this key.
    pub fn matches(&self, image: &Image, settings: &FlowSettings, digest_of_image: u64) -> bool {
        self.digest == digest_of_image && same_settings(&self.settings, settings) && same_bits(&self.image, image)
    }
}

/// The settings compared field by field, floats by their bits.
fn same_settings(a: &FlowSettings, b: &FlowSettings) -> bool {
    a.quality == b.quality
        && a.hiq == b.hiq
        && a.sensitivity.to_bits() == b.sensitivity.to_bits()
        && a.prep_contrast_edge == b.prep_contrast_edge
        && a.prep_delinearize == b.prep_delinearize
        && a.track_alpha == b.track_alpha
        && a.video_range_luma == b.video_range_luma
        && a.skip_refine == b.skip_refine
        && a.wrap == b.wrap
}

/// Two images with the same size, format and sample bits.
pub fn same_bits(a: &Image, b: &Image) -> bool {
    if a.width != b.width || a.height != b.height {
        return false;
    }
    match (&a.samples, &b.samples) {
        (Samples::U8(x), Samples::U8(y)) => x == y,
        (Samples::U16(x), Samples::U16(y)) => x == y,
        (Samples::F32(x), Samples::F32(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()),
        _ => false,
    }
}

/// A digest of the size, the format and up to 4096 evenly spaced samples (FNV-1a over their bits).
pub fn digest(img: &Image) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |v: u64| {
        for b in v.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    eat(img.width as u64);
    eat(img.height as u64);
    let n = img.sample_count();
    let step = (n / 4096).max(1);
    match &img.samples {
        Samples::U8(v) => {
            eat(1);
            v.iter().step_by(step).for_each(|&x| eat(u64::from(x)));
        }
        Samples::U16(v) => {
            eat(2);
            v.iter().step_by(step).for_each(|&x| eat(u64::from(x)));
        }
        Samples::F32(v) => {
            eat(3);
            v.iter().step_by(step).for_each(|&x| eat(u64::from(x.to_bits())));
        }
    }
    h
}

/// Entries of `V` with their keys and sizes, most recently used last.
pub struct Lru<K, V> {
    entries: Vec<(K, V, u64)>,
    budget: u64,
    used: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl<K, V> Lru<K, V> {
    /// An empty cache that holds at most `budget` bytes.
    pub fn new(budget: u64) -> Lru<K, V> {
        Lru { entries: Vec::new(), budget, used: 0, hits: 0, misses: 0, evictions: 0 }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Bytes held.
    pub fn used(&self) -> u64 {
        self.used
    }

    pub fn budget(&self) -> u64 {
        self.budget
    }

    /// The entry whose key passes `find`, made the most recent; counts a hit or a miss.
    pub fn get(&mut self, find: impl Fn(&K) -> bool) -> Option<&V> {
        match self.entries.iter().position(|(k, _, _)| find(k)) {
            Some(i) => {
                self.hits += 1;
                let e = self.entries.remove(i);
                self.entries.push(e);
                self.entries.last().map(|(_, v, _)| v)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Adds an entry of `bytes`, evicting the least recently used ones until it fits. An entry
    /// larger than the whole budget is not kept (`false`).
    pub fn insert(&mut self, key: K, value: V, bytes: u64) -> bool {
        if bytes > self.budget {
            return false;
        }
        while self.used + bytes > self.budget {
            let (_, _, b) = self.entries.remove(0);
            self.used -= b;
            self.evictions += 1;
        }
        self.entries.push((key, value, bytes));
        self.used += bytes;
        true
    }

    /// The most recent entry.
    pub fn last(&self) -> Option<&V> {
        self.entries.last().map(|(_, v, _)| v)
    }

    /// The entry whose key passes `find`, without counting or reordering (two entries can stay
    /// borrowed at once).
    pub fn peek(&self, find: impl Fn(&K) -> bool) -> Option<&V> {
        self.entries.iter().find(|(k, _, _)| find(k)).map(|(_, v, _)| v)
    }

    /// Drops every entry.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.used = 0;
    }
}
