//! Tests of the hole fill. Hand goldens, properties, an independent f64 reference of the
//! pull-push pyramid, the numpy float32 model fixtures (`tests/fixtures/f1_gen.py`), thread
//! count invariance. The helpers at the top are shared with `smartblend::tests`.

use super::*;
use crate::image::Fmt;

// ------------------------------------------------------------------------------------------------
// shared helpers
// ------------------------------------------------------------------------------------------------

pub(crate) struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        (self.next() >> 11) % n
    }
    pub fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub(crate) fn in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(f)
}

pub(crate) fn same_bits(a: &Image, b: &Image) -> bool {
    a.width == b.width
        && a.height == b.height
        && match (&a.samples, &b.samples) {
            (Samples::U8(x), Samples::U8(y)) => x == y,
            (Samples::U16(x), Samples::U16(y)) => x == y,
            (Samples::F32(x), Samples::F32(y)) => x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()) && x.len() == y.len(),
            _ => false,
        }
}

/// First sample index where two images differ bit for bit.
pub(crate) fn first_diff(a: &Image, b: &Image) -> Option<usize> {
    match (&a.samples, &b.samples) {
        (Samples::U8(x), Samples::U8(y)) => (0..x.len().max(y.len())).find(|&i| x.get(i) != y.get(i)),
        (Samples::U16(x), Samples::U16(y)) => (0..x.len().max(y.len())).find(|&i| x.get(i) != y.get(i)),
        (Samples::F32(x), Samples::F32(y)) => {
            (0..x.len().max(y.len())).find(|&i| x.get(i).map(|v| v.to_bits()) != y.get(i).map(|v| v.to_bits()))
        }
        _ => Some(0),
    }
}

pub(crate) fn image_from(fmt: Fmt, w: usize, h: usize, rng: &mut Rng) -> Image {
    let mut img = Image::new(w, h, fmt);
    match &mut img.samples {
        Samples::U8(v) => v.iter_mut().for_each(|s| *s = rng.below(256) as u8),
        Samples::U16(v) => v.iter_mut().for_each(|s| *s = rng.below(65536) as u16),
        Samples::F32(v) => v.iter_mut().for_each(|s| *s = rng.unit()),
    }
    img
}

pub(crate) fn as_f64(img: &Image) -> Vec<f64> {
    match &img.samples {
        Samples::U8(v) => v.iter().map(|&s| f64::from(s)).collect(),
        Samples::U16(v) => v.iter().map(|&s| f64::from(s)).collect(),
        Samples::F32(v) => v.iter().map(|&s| f64::from(s)).collect(),
    }
}

/// One case of `tests/fixtures/f1_cases.hex` (format in `f1_gen.py`).
pub(crate) struct Case {
    pub kind: u32,
    pub depth: u32,
    pub w: usize,
    pub h: usize,
    pub weight: f32,
    pub mask_kind: u32,
    pub p1: Image,
    pub p2: Option<Image>,
    pub mask: Vec<u8>,
    pub out: Image,
}

pub(crate) fn load_cases() -> Vec<Case> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/f1_cases.hex");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let b = crate::fixture::hex(&text);
    assert_eq!(&b[0..4], b"F1C1");
    let u32_at = |p: usize| u32::from_le_bytes(b[p..p + 4].try_into().unwrap());
    let n = u32_at(4) as usize;
    let mut pos = 8;
    let mut cases = Vec::new();
    for _ in 0..n {
        let kind = u32_at(pos);
        let depth = u32_at(pos + 4);
        let w = u32_at(pos + 8) as usize;
        let h = u32_at(pos + 12) as usize;
        let weight = f32::from_bits(u32_at(pos + 16));
        let mask_kind = u32_at(pos + 20);
        pos += 24;
        let fmt = match depth {
            0 => Fmt::U8,
            1 => Fmt::U16,
            _ => Fmt::F32,
        };
        let count = w * h * 4;
        let read_img = |pos: &mut usize| -> Image {
            let mut img = Image::new(w, h, fmt);
            match &mut img.samples {
                Samples::U8(v) => {
                    v.copy_from_slice(&b[*pos..*pos + count]);
                    *pos += count;
                }
                Samples::U16(v) => {
                    for (i, s) in v.iter_mut().enumerate() {
                        *s = u16::from_le_bytes(b[*pos + 2 * i..*pos + 2 * i + 2].try_into().unwrap());
                    }
                    *pos += 2 * count;
                }
                Samples::F32(v) => {
                    for (i, s) in v.iter_mut().enumerate() {
                        *s = f32::from_bits(u32::from_le_bytes(b[*pos + 4 * i..*pos + 4 * i + 4].try_into().unwrap()));
                    }
                    *pos += 4 * count;
                }
            }
            img
        };
        let p1 = read_img(&mut pos);
        let (p2, mask);
        if kind == 3 {
            let msz = match mask_kind {
                1 => 1,
                2 => 2,
                _ => 4,
            };
            mask = b[pos..pos + w * h * msz].to_vec();
            pos += w * h * msz;
            p2 = None;
        } else {
            p2 = Some(read_img(&mut pos));
            mask = Vec::new();
        }
        let out = read_img(&mut pos);
        cases.push(Case { kind, depth, w, h, weight, mask_kind, p1, p2, mask, out });
    }
    assert_eq!(pos, b.len(), "trailing bytes in the fixture");
    cases
}

/// A mask in its native type from bytes.
enum OwnedMask {
    U8(Vec<u8>),
    U16(Vec<u16>),
    F32(Vec<f32>),
}

impl OwnedMask {
    fn from_case(c: &Case) -> OwnedMask {
        match c.mask_kind {
            1 => OwnedMask::U8(c.mask.clone()),
            2 => OwnedMask::U16(c.mask.chunks(2).map(|p| u16::from_le_bytes([p[0], p[1]])).collect()),
            _ => OwnedMask::F32(c.mask.chunks(4).map(|p| f32::from_bits(u32::from_le_bytes([p[0], p[1], p[2], p[3]]))).collect()),
        }
    }
    fn as_ref(&self) -> HoleMaskRef<'_> {
        match self {
            OwnedMask::U8(m) => HoleMaskRef::U8(m),
            OwnedMask::U16(m) => HoleMaskRef::U16(m),
            OwnedMask::F32(m) => HoleMaskRef::F32(m),
        }
    }
}

fn u8_mask(w: usize, h: usize, holes: &[(usize, usize)]) -> Vec<u8> {
    let mut m = vec![0u8; w * h];
    for &(x, y) in holes {
        m[y * w + x] = 255;
    }
    m
}

fn rect(m: &mut [u8], w: usize, x0: usize, y0: usize, x1: usize, y1: usize) {
    for y in y0..y1 {
        for x in x0..x1 {
            m[y * w + x] = 255;
        }
    }
}

fn samples_u8(img: &Image) -> &[u8] {
    match &img.samples {
        Samples::U8(v) => v,
        _ => panic!("not u8"),
    }
}

fn samples_f32(img: &Image) -> &[f32] {
    match &img.samples {
        Samples::F32(v) => v,
        _ => panic!("not f32"),
    }
}

// ------------------------------------------------------------------------------------------------
// geometry of the pyramid
// ------------------------------------------------------------------------------------------------

#[test]
fn geometry_goldens_worked_out_by_hand() {
    // 2x2: one loop pass gives (2, 2), count = 2, levels 0..=2 all 2x2, push starts at level 1
    let g = geometry(2, 2);
    assert_eq!(g.dims, vec![(2, 2), (2, 2), (2, 2)]);
    assert_eq!(g.top, 1);
    // 8x8: (4,4) count 2; (2,2) count 3 stop; levels 8, 4, 2 and the unused 2
    let g = geometry(8, 8);
    assert_eq!(g.dims, vec![(8, 8), (4, 4), (2, 2), (2, 2)]);
    assert_eq!(g.top, 2);
    // 16x4: (8,2) c2; (4,1->2) c3; (2,2) c4 stop
    let g = geometry(16, 4);
    assert_eq!(g.dims, vec![(16, 4), (8, 2), (4, 2), (2, 2), (2, 2)]);
    assert_eq!(g.top, 3);
    // 3x5: (1->2, 2) c2: a = 2 and b = 2 stop (the clamp makes both 2)
    let g = geometry(3, 5);
    assert_eq!(g.dims, vec![(3, 5), (2, 2), (2, 2)]);
    assert_eq!(g.top, 1);
    // 1920x1080: 960x540, 480x270, 240x135, 120x67, 60x33, 30x16, 15x8, 7x4, 3x2, 2x2 (count 11)
    let g = geometry(1920, 1080);
    assert_eq!(g.dims[1], (960, 540));
    assert_eq!(g.dims[4], (120, 67));
    assert_eq!(g.dims[8], (7, 4));
    assert_eq!(g.dims[9], (3, 2));
    assert_eq!(g.dims[10], (2, 2));
    assert_eq!(g.top, 10);
    assert_eq!(g.dims.len(), 12);
}

#[test]
fn geometry_matches_an_independent_halving_loop_for_all_small_sizes() {
    for w in 2..70usize {
        for h in 2..70usize {
            // independent: halve (clamped to 2) until both are 2; `top` is the number of halvings, at least 1
            let mut dims = vec![(w, h)];
            loop {
                let (a, b) = *dims.last().unwrap();
                let next = ((a / 2).max(2), (b / 2).max(2));
                dims.push(next);
                if next == (2, 2) {
                    break;
                }
            }
            let top = dims.len() - 1;
            // the pyramid object holds one more level than it uses
            dims.push((2, 2));
            let g = geometry(w, h);
            assert_eq!(g.top, top, "{w}x{h}");
            assert_eq!(g.dims, dims, "{w}x{h}");
        }
    }
}

// ------------------------------------------------------------------------------------------------
// hand goldens and properties
// ------------------------------------------------------------------------------------------------

#[test]
fn without_holes_the_image_is_returned_unchanged_at_every_depth() {
    let mut rng = Rng(7);
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let img = image_from(fmt, 13, 9, &mut rng);
        let mut out = img.clone();
        fill_holes(&mut out, &Coverage::new(13, 9, 0), false).unwrap();
        assert!(same_bits(&img, &out), "{fmt:?}");
        // and with the 360 flag too (no hole, no work)
        fill_holes(&mut out, &Coverage::new(13, 9, 0), true).unwrap();
        assert!(same_bits(&img, &out), "{fmt:?} 360");
    }
}

#[test]
fn a_pixel_that_is_not_a_hole_comes_back_bit_exact_in_every_depth() {
    // weight 1.0 everywhere known: value / 1.0 is exact in f32, and u8 and u16 store x + 0.5 truncated
    let mut rng = Rng(11);
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        for &(w, h) in &[(2usize, 2usize), (5, 7), (16, 16), (33, 20)] {
            let img = image_from(fmt, w, h, &mut rng);
            let mut mask = vec![0u8; w * h];
            for m in mask.iter_mut() {
                if rng.below(4) == 0 {
                    *m = 255;
                }
            }
            mask[0] = 255; // at least one hole
            let mut out = img.clone();
            fill_holes_masked(&mut out, HoleMaskRef::U8(&mask), false).unwrap();
            let (a, b) = (as_f64(&img), as_f64(&out));
            for i in 0..w * h {
                if mask[i] == 0 {
                    for c in 0..4 {
                        assert_eq!(a[i * 4 + c].to_bits(), b[i * 4 + c].to_bits(), "{fmt:?} {w}x{h} pixel {i} channel {c}");
                    }
                }
            }
        }
    }
}

#[test]
fn all_holes_give_a_zero_image() {
    // every weight is 0: the cleared top level stays 0, blurs and resamples of zeros are zeros
    let mut rng = Rng(3);
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let mut img = image_from(fmt, 9, 6, &mut rng);
        fill_holes(&mut img, &Coverage::new(9, 6, 255), false).unwrap();
        assert!(as_f64(&img).iter().all(|&v| v == 0.0), "{fmt:?}");
    }
}

#[test]
fn a_hole_in_a_constant_area_is_filled_with_the_constant() {
    // u8: weights and values blur with the same kernels, v / w is the constant to far better than half a step
    for c in [1u8, 100, 200, 255] {
        let mut img = Image::new(16, 16, Fmt::U8);
        if let Samples::U8(v) = &mut img.samples {
            v.fill(c);
        }
        let mask = u8_mask(16, 16, &[(8, 8), (9, 8), (8, 9), (0, 0), (15, 15), (0, 15)]);
        fill_holes_masked(&mut img, HoleMaskRef::U8(&mask), false).unwrap();
        assert!(samples_u8(&img).iter().all(|&s| s == c), "constant {c}");
    }
    // f32: within a few ulps
    let mut img = Image::new(20, 12, Fmt::F32);
    if let Samples::F32(v) = &mut img.samples {
        v.fill(0.3712);
    }
    let mut mask = vec![0u8; 20 * 12];
    rect(&mut mask, 20, 5, 3, 11, 9);
    fill_holes_masked(&mut img, HoleMaskRef::U8(&mask), false).unwrap();
    for &v in samples_f32(&img) {
        assert!((v - 0.3712).abs() < 1e-5, "{v}");
    }
}

#[test]
fn a_hole_between_two_flat_areas_blends_between_them() {
    // columns 0..8 hold 40, columns 12..20 hold 200, the 4 columns between are holes
    let (w, h) = (20usize, 6usize);
    let mut img = Image::new(w, h, Fmt::U8);
    if let Samples::U8(v) = &mut img.samples {
        for y in 0..h {
            for x in 0..w {
                let s = if x < 8 { 40 } else { 200 };
                v[(y * w + x) * 4..(y * w + x) * 4 + 4].fill(s);
            }
        }
    }
    let mut mask = vec![0u8; w * h];
    rect(&mut mask, w, 8, 0, 12, h);
    fill_holes_masked(&mut img, HoleMaskRef::U8(&mask), false).unwrap();
    let v = samples_u8(&img);
    let mut last = 0u8;
    for x in 8..12 {
        let s = v[(3 * w + x) * 4];
        assert!((40..=200).contains(&s), "x {x}: {s}");
        assert!(s >= last, "monotone ramp at x {x}: {last} then {s}");
        last = s;
    }
    // symmetric geometry: the first hole column is nearer to 40 than the last
    assert!(v[(3 * w + 8) * 4] < v[(3 * w + 11) * 4]);
    // known pixels unchanged
    assert_eq!(v[(3 * w + 7) * 4], 40);
    assert_eq!(v[(3 * w + 12) * 4], 200);
}

#[test]
fn filled_values_stay_inside_the_range_of_the_known_values() {
    // every level is a convex combination of known pixels (weights from non negative blurs, bilinear resamples)
    let mut rng = Rng(99);
    for round in 0..30 {
        let (w, h) = (3 + rng.below(30) as usize, 3 + rng.below(30) as usize);
        let mut img = image_from(Fmt::F32, w, h, &mut rng);
        let mut mask = vec![0u8; w * h];
        for m in mask.iter_mut() {
            if rng.below(3) == 0 {
                *m = 255;
            }
        }
        if mask.iter().all(|&m| m != 0) {
            mask[0] = 0;
        }
        let before = samples_f32(&img).to_vec();
        fill_holes_masked(&mut img, HoleMaskRef::U8(&mask), false).unwrap();
        let after = samples_f32(&img);
        for c in 0..4 {
            let known: Vec<f32> = (0..w * h).filter(|&i| mask[i] == 0).map(|i| before[i * 4 + c]).collect();
            let (lo, hi) = (known.iter().cloned().fold(f32::MAX, f32::min), known.iter().cloned().fold(f32::MIN, f32::max));
            for i in 0..w * h {
                let v = after[i * 4 + c];
                assert!(v >= lo - 1e-4 && v <= hi + 1e-4, "round {round} ch {c} pixel {i}: {v} outside [{lo}, {hi}]");
            }
        }
    }
}

#[test]
fn a_transparent_hole_next_to_opaque_pixels_becomes_opaque() {
    // colour and alpha are filled independently: alpha 0 inside the hole turns into the alpha around it
    let (w, h) = (12usize, 12usize);
    let mut img = Image::new(w, h, Fmt::U8);
    if let Samples::U8(v) = &mut img.samples {
        for i in 0..w * h {
            v[i * 4..i * 4 + 4].copy_from_slice(&[255, 90, 60, 30]);
        }
    }
    let mut mask = vec![0u8; w * h];
    rect(&mut mask, w, 4, 4, 8, 8);
    if let Samples::U8(v) = &mut img.samples {
        for y in 4..8 {
            for x in 4..8 {
                v[(y * w + x) * 4..(y * w + x) * 4 + 4].fill(0);
            }
        }
    }
    fill_holes_masked(&mut img, HoleMaskRef::U8(&mask), false).unwrap();
    for px in samples_u8(&img).chunks(4) {
        assert_eq!(px, &[255, 90, 60, 30]);
    }
}

#[test]
fn mask_thresholds_per_type() {
    // u8: any non zero value is a hole (eps is tiny), u16 likewise, f32: 3.1e-5 and above
    let m8 = [0u8, 1, 128, 255];
    let hole8: Vec<bool> = (0..4).map(|i| HoleMaskRef::U8(&m8).is_hole(i)).collect();
    assert_eq!(hole8, vec![false, true, true, true]);
    let m16 = [0u16, 1, 300, 65535];
    let hole16: Vec<bool> = (0..4).map(|i| HoleMaskRef::U16(&m16).is_hole(i)).collect();
    assert_eq!(hole16, vec![false, true, true, true]);
    let mf = [0.0f32, 1.0e-5, 3.0e-5, 3.2e-5, 0.5, 1.0, f32::NAN, f32::INFINITY, -0.0, -1.0];
    let holef: Vec<bool> = (0..mf.len()).map(|i| HoleMaskRef::F32(&mf).is_hole(i)).collect();
    // `eps > (double)m` is "not a hole"; NaN is unordered and therefore a hole; negative values are not holes
    assert_eq!(holef, vec![false, false, false, true, true, true, true, true, false, false]);
    assert_eq!(WIDE_EPS, f64::from_bits(0x3f00_40bf_e3b0_3e21));
    assert!((WIDE_EPS - 3.1e-5).abs() < 2e-7);
}

#[test]
fn the_three_mask_types_fill_identically() {
    let mut rng = Rng(5);
    let (w, h) = (21usize, 14usize);
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let img = image_from(fmt, w, h, &mut rng);
        let mut m8 = vec![0u8; w * h];
        rect(&mut m8, w, 3, 2, 9, 7);
        m8[w * 10 + 15] = 255;
        let m16: Vec<u16> = m8.iter().map(|&m| if m != 0 { 65535 } else { 0 }).collect();
        let mf: Vec<f32> = m8.iter().map(|&m| if m != 0 { 1.0 } else { 0.0 }).collect();
        let mut a = img.clone();
        let mut b = img.clone();
        let mut c = img.clone();
        fill_holes_masked(&mut a, HoleMaskRef::U8(&m8), false).unwrap();
        fill_holes_masked(&mut b, HoleMaskRef::U16(&m16), false).unwrap();
        fill_holes_masked(&mut c, HoleMaskRef::F32(&mf), false).unwrap();
        assert!(same_bits(&a, &b) && same_bits(&a, &c), "{fmt:?}");
    }
}

#[test]
fn fill_holes_uses_the_coverage_bytes_as_the_mask() {
    // `holes` is the inverted coverage, 255 where nothing was warped; partial values count as holes
    let mut rng = Rng(8);
    let img = image_from(Fmt::U8, 10, 10, &mut rng);
    let mut holes = Coverage::new(10, 10, 0);
    holes.data[3 * 10 + 4] = 255;
    holes.data[5 * 10 + 5] = 1;
    let mut a = img.clone();
    fill_holes(&mut a, &holes, false).unwrap();
    let mut b = img.clone();
    fill_holes_masked(&mut b, HoleMaskRef::U8(&holes.data), false).unwrap();
    assert!(same_bits(&a, &b));
    assert!(!same_bits(&a, &img));
    // the pixel with mask 1 was replaced as a whole
    let (va, vi) = (samples_u8(&a), samples_u8(&img));
    let d: Vec<usize> = (0..100).filter(|&p| va[p * 4..p * 4 + 4] != vi[p * 4..p * 4 + 4]).collect();
    assert!(d.iter().all(|&p| p == 3 * 10 + 4 || p == 5 * 10 + 5), "{d:?}");
}

#[test]
fn errors_and_refusals() {
    let mut img = Image::new(8, 8, Fmt::U8);
    // a mask of the wrong size is an error
    assert!(matches!(fill_holes(&mut img, &Coverage::new(8, 7, 255), false), Err(FxError::Mismatch(_))));
    assert!(matches!(fill_holes_masked(&mut img, HoleMaskRef::U8(&[0u8; 3]), false), Err(FxError::Mismatch(_))));
    // the 360 flag with a hole fills (the wrapped passes, G6) instead of refusing
    let mut rng = Rng(9);
    let src = image_from(Fmt::U8, 8, 8, &mut rng);
    let mut cov = Coverage::new(8, 8, 0);
    cov.data[10] = 255;
    let mut wrapped = src.clone();
    fill_holes(&mut wrapped, &cov, true).unwrap();
    assert!(!same_bits(&wrapped, &src), "the hole pixel was filled");
    let changed: Vec<usize> = (0..64).filter(|&p| samples_u8(&wrapped)[p * 4..p * 4 + 4] != samples_u8(&src)[p * 4..p * 4 + 4]).collect();
    assert_eq!(changed, vec![10]);
    // an image with a dimension of 1, with either flag
    for is360 in [false, true] {
        let mut thin = Image::new(1, 8, Fmt::F32);
        let mut cov = Coverage::new(1, 8, 0);
        cov.data[2] = 255;
        assert!(matches!(fill_holes(&mut thin, &cov, is360), Err(FxError::Unsupported(_))), "{is360}");
        let mut wide = Image::new(8, 1, Fmt::U8);
        let mut cov = Coverage::new(8, 1, 0);
        cov.data[2] = 255;
        assert!(matches!(fill_holes(&mut wide, &cov, is360), Err(FxError::Unsupported(_))), "{is360}");
        // ... but no hole is fine
        fill_holes(&mut thin, &Coverage::new(1, 8, 0), is360).unwrap();
        // and a mask of the wrong size is an error with the flag too
        assert!(matches!(fill_holes_masked(&mut img, HoleMaskRef::U8(&[0u8; 3]), is360), Err(FxError::Mismatch(_))));
    }
    // flow data length
    let mut flow = FlowField { width: 4, height: 4, data: vec![0.0; 5] };
    assert!(matches!(fill_flow_holes(&mut flow, HoleMaskRef::U8(&[0u8; 16]), false), Err(FxError::Mismatch(_))));
}

// ------------------------------------------------------------------------------------------------
// flow fill (Vectors In)
// ------------------------------------------------------------------------------------------------

#[test]
fn flow_holes_are_filled_per_component_like_a_float_plane() {
    let mut rng = Rng(21);
    let (w, h) = (24usize, 15usize);
    let mut flow = FlowField::new(w, h);
    for v in flow.data.iter_mut() {
        *v = rng.unit() * 8.0 - 4.0;
    }
    let mut mask = vec![0u8; w * h];
    rect(&mut mask, w, 6, 4, 14, 10);
    mask[0] = 255;
    let mut filled = flow.clone();
    fill_flow_holes(&mut filled, HoleMaskRef::U8(&mask), false).unwrap();
    for c in 0..2 {
        let mut plane: Vec<f32> = (0..w * h).map(|i| flow.data[i * 2 + c]).collect();
        fill_plane(&mut plane, w, h, HoleMaskRef::U8(&mask), false).unwrap();
        for i in 0..w * h {
            assert_eq!(plane[i].to_bits(), filled.data[i * 2 + c].to_bits(), "component {c} pixel {i}");
        }
    }
    // known vectors untouched
    for i in 0..w * h {
        if mask[i] == 0 {
            assert_eq!(flow.data[i * 2], filled.data[i * 2]);
            assert_eq!(flow.data[i * 2 + 1], filled.data[i * 2 + 1]);
        }
    }
}

#[test]
fn a_constant_flow_survives_its_holes() {
    let (w, h) = (18usize, 18usize);
    let mut flow = FlowField::new(w, h);
    for i in 0..w * h {
        flow.data[i * 2] = 3.0;
        flow.data[i * 2 + 1] = -2.0;
    }
    let mut mask = vec![0u8; w * h];
    rect(&mut mask, w, 5, 5, 12, 12);
    let mf: Vec<f32> = mask.iter().map(|&m| f32::from(m) / 255.0).collect();
    fill_flow_holes(&mut flow, HoleMaskRef::F32(&mf), false).unwrap();
    for i in 0..w * h {
        assert!((flow.data[i * 2] - 3.0).abs() < 1e-5 && (flow.data[i * 2 + 1] + 2.0).abs() < 1e-5, "pixel {i}");
    }
}

// ------------------------------------------------------------------------------------------------
// the vertical tap order (VerticalTaps)
// ------------------------------------------------------------------------------------------------

/// The G plane of output frame 1 of the probe case `vectors-in.hf1-12x9-4-6`: the frame pushed by
/// (2, 3) pixels, 12 x 9, in
/// the bottom-up rows the fill of `TwixtorProVectorsIn_64` sees, holes zeroed (`HF1_MASK`: the
/// three rows 0 to 2 and columns 0 and 1).
const HF1_W: usize = 12;
const HF1_H: usize = 9;
const HF1_INPUT: [u32; 108] = [
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x428bb2d1, 0x42df23fc,
        0x447853a8, 0x446f63a0, 0x434e6364, 0x440877ce, 0x440fc25f, 0x43cfbf20, 0x44032d4b, 0x4418b309,
        0x00000000, 0x00000000, 0x41c80000, 0x442c8825, 0x43788fdc, 0x4388c972, 0x43d35188, 0x43d783cd,
        0x44757f9a, 0x4400e6ea, 0x44168b78, 0x44006c2d, 0x00000000, 0x00000000, 0x42fb3400, 0x43050d2b,
        0x445afc81, 0x42549c00, 0x44787e58, 0x433533d0, 0x4459db77, 0x444eea43, 0x439e716d, 0x440fa8cc,
        0x00000000, 0x00000000, 0x443e4000, 0x43880e3d, 0x44401029, 0x42def5e0, 0x4475851d, 0x422e5a00,
        0x4411fe65, 0x44379306, 0x436fec80, 0x44324f0e, 0x00000000, 0x00000000, 0x43cc2000, 0x4354188a,
        0x430995f1, 0x4411cda6, 0x43bee412, 0x41bc4710, 0x43bab9c8, 0x43a13380, 0x443145cc, 0x438d0515,
        0x00000000, 0x00000000, 0x42640000, 0x446919ff, 0x442b4161, 0x440ac68b, 0x43f13b92, 0x438e8cb8,
        0x43b0357e, 0x445e803a, 0x42ad5b20, 0x447554d0,
];
const HF1_MASK: &str = "111111111111111111111111111111111111110000000000110000000000110000000000110000000000110000000000110000000000";
/// The 48 hole values the original wrote (MEASURED under the emulator, `refs/` of probe hf1),
/// in row order.
const HF1_PLUGIN: [u32; 48] = [
        0x43b473d7, 0x43b8e616, 0x43c0ae12, 0x43cc571f, 0x43d92ea2, 0x43e34296, 0x43ea2f59, 0x43f01a31,
        0x43f58273, 0x43f9ab57, 0x43fc96f1, 0x43fea27e, 0x43b08f0b, 0x43b39371, 0x43bbda7b, 0x43cba56c,
        0x43dd8ebf, 0x43e8f9bb, 0x43ee0f61, 0x43f2cfae, 0x43f84c41, 0x43fc54cd, 0x43ff0646, 0x4400a1a5,
        0x43aa0ee3, 0x43aa7471, 0x43b3541c, 0x43c97612, 0x43e2f652, 0x43f092a0, 0x43f3ff0b, 0x43f8c207,
        0x43ffa81c, 0x4401aed5, 0x44024da7, 0x440311f6, 0x43a39b2a, 0x43a1597c, 0x43a28e41, 0x439ffd84,
        0x43a8777d, 0x43a8b9e6, 0x43af90d9, 0x43b3918a, 0x43b1ccc7, 0x43b83410, 0x43afb395, 0x43b76508,
];
/// The same holes filled with the engine's other tap order, as the rewrite filled them before.
const HF1_BELOW_FIRST: [u32; 48] = [
        0x43b473d7, 0x43b8e616, 0x43c0ae12, 0x43cc571f, 0x43d92ea2, 0x43e34296, 0x43ea2f59, 0x43f01a31,
        0x43f58273, 0x43f9ab57, 0x43fc96f1, 0x43fea27e, 0x43b08f0b, 0x43b39371, 0x43bbda7b, 0x43cba56c,
        0x43dd8ec0, 0x43e8f9bc, 0x43ee0f61, 0x43f2cfae, 0x43f84c41, 0x43fc54cd, 0x43ff0647, 0x4400a1a5,
        0x43aa0ee3, 0x43aa7472, 0x43b3541c, 0x43c97612, 0x43e2f652, 0x43f092a0, 0x43f3ff0d, 0x43f8c208,
        0x43ffa81c, 0x4401aed5, 0x44024da7, 0x440311f6, 0x43a39b2a, 0x43a1597d, 0x43a28e41, 0x439ffd84,
        0x43a8777d, 0x43a8b9e6, 0x43af90d9, 0x43b3918a, 0x43b1ccc7, 0x43b8340f, 0x43afb395, 0x43b76508,
];

fn hf1_fill(taps: VerticalTaps) -> Vec<u32> {
    let mut plane: Vec<f32> = HF1_INPUT.iter().map(|&b| f32::from_bits(b)).collect();
    let mask: Vec<u8> = HF1_MASK.bytes().map(|b| if b == b'1' { 255 } else { 0 }).collect();
    fill_plane_with(&mut plane, HF1_W, HF1_H, HoleMaskRef::U8(&mask), false, taps).unwrap();
    plane.iter().zip(&mask).filter(|(_, &m)| m != 0).map(|(v, _)| v.to_bits()).collect()
}

#[test]
fn above_first_reproduces_the_holes_the_vectors_in_plugin_filled() {
    assert_eq!(HF1_MASK.len(), HF1_W * HF1_H);
    assert_eq!(hf1_fill(VerticalTaps::AboveFirst), HF1_PLUGIN.to_vec());
    // the engine's order is the default, and it differs from the plugin in 8 of the 48 values
    let below = hf1_fill(VerticalTaps::default());
    assert_eq!(below, HF1_BELOW_FIRST.to_vec());
    assert_eq!(below.iter().zip(HF1_PLUGIN.iter()).filter(|(a, b)| a != b).count(), 8);
    let mut plane: Vec<f32> = HF1_INPUT.iter().map(|&b| f32::from_bits(b)).collect();
    let mask: Vec<u8> = HF1_MASK.bytes().map(|b| if b == b'1' { 255 } else { 0 }).collect();
    fill_plane(&mut plane, HF1_W, HF1_H, HoleMaskRef::U8(&mask), false).unwrap();
    let plain: Vec<u32> = plane.iter().zip(&mask).filter(|(_, &m)| m != 0).map(|(v, _)| v.to_bits()).collect();
    assert_eq!(plain, below, "fill_plane is the BelowFirst fill");
}

#[test]
fn the_tap_order_reaches_every_entry_point_and_keeps_the_known_pixels() {
    let (w, h) = (17usize, 13usize);
    let mut rng = Rng(0x5eed_0004);
    let mut mask = vec![0u8; w * h];
    rect(&mut mask, w, 0, 0, w, 2);
    rect(&mut mask, w, 0, 0, 3, h);
    let img = image_from(Fmt::F32, w, h, &mut rng);
    let cov = crate::warp::Coverage { width: w, height: h, data: mask.clone() };
    let (mut a, mut b, mut c) = (img.clone(), img.clone(), img.clone());
    fill_holes_with(&mut a, &cov, false, VerticalTaps::AboveFirst).unwrap();
    fill_holes_masked_with(&mut b, HoleMaskRef::U8(&mask), false, VerticalTaps::AboveFirst).unwrap();
    fill_holes(&mut c, &cov, false).unwrap();
    assert!(same_bits(&a, &b));
    assert!(!same_bits(&a, &c), "the two orders differ somewhere in the holes");
    let (sa, si) = (samples_f32(&a), samples_f32(&img));
    for i in 0..w * h {
        if mask[i] == 0 {
            for ch in 0..CHANNELS {
                assert_eq!(sa[i * CHANNELS + ch].to_bits(), si[i * CHANNELS + ch].to_bits(), "known pixel {i}");
            }
        }
    }
    // the flow fill: each component is the plane fill with the same order
    let mut flow = FlowField::new(w, h);
    for (i, v) in flow.data.iter_mut().enumerate() {
        *v = (i % 23) as f32 * 0.37 - 3.0;
    }
    let mut want0: Vec<f32> = flow.data.iter().step_by(2).copied().collect();
    fill_plane_with(&mut want0, w, h, HoleMaskRef::U8(&mask), false, VerticalTaps::AboveFirst).unwrap();
    fill_flow_holes_with(&mut flow, HoleMaskRef::U8(&mask), false, VerticalTaps::AboveFirst).unwrap();
    let got0: Vec<u32> = flow.data.iter().step_by(2).map(|v| v.to_bits()).collect();
    assert_eq!(got0, want0.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    // without a hole either order is the identity
    let mut d = img.clone();
    fill_holes_masked_with(&mut d, HoleMaskRef::U8(&vec![0u8; w * h]), false, VerticalTaps::AboveFirst).unwrap();
    assert!(same_bits(&d, &img));
}

// ------------------------------------------------------------------------------------------------
// independent f64 reference of the pull-push fill
// ------------------------------------------------------------------------------------------------

mod reference {
    //! A straightforward double precision version of the pull-push fill: no shared code with the
    //! engine (kernels, border rule and resample are re-derived).

    pub fn gaussian_taps(r: f64) -> Vec<f64> {
        let n = r as usize + 1;
        let d = (r / 3.0) * (r / 3.0);
        let raw: Vec<f64> = (0..n).map(|i| 2.0f64.powf(-((i * i) as f64) / d)).collect();
        let sum: f64 = raw.iter().sum();
        let den = 2.0 * sum - raw[0];
        raw.iter().map(|v| v / den).collect()
    }

    pub const BINOMIAL: [f64; 3] = [0.375, 0.25, 0.0625];

    /// Separable blur with taps `k[|j|]`, renormalised over the taps that exist (the border rule).
    pub fn blur(img: &[f64], w: usize, h: usize, k: &[f64]) -> Vec<f64> {
        let n = k.len() as i64;
        let mut tmp = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let (mut num, mut den) = (0.0, 0.0);
                for j in -(n - 1)..n {
                    let xx = x as i64 + j;
                    if xx >= 0 && xx < w as i64 {
                        let wt = k[j.unsigned_abs() as usize];
                        num += wt * img[y * w + xx as usize];
                        den += wt;
                    }
                }
                tmp[y * w + x] = num / den;
            }
        }
        let mut out = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let (mut num, mut den) = (0.0, 0.0);
                for j in -(n - 1)..n {
                    let yy = y as i64 + j;
                    if yy >= 0 && yy < h as i64 {
                        let wt = k[j.unsigned_abs() as usize];
                        num += wt * tmp[yy as usize * w + x];
                        den += wt;
                    }
                }
                out[y * w + x] = num / den;
            }
        }
        out
    }

    /// Corner aligned bilinear resample.
    pub fn resample(src: &[f64], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f64> {
        let mut out = vec![0.0; dw * dh];
        for y in 0..dh {
            for x in 0..dw {
                let fx = x as f64 * (sw as f64 - 1.0) / (dw as f64 - 1.0);
                let fy = y as f64 * (sh as f64 - 1.0) / (dh as f64 - 1.0);
                let ix = (fx.floor() as usize).min(sw - 2);
                let iy = (fy.floor() as usize).min(sh - 2);
                let (tx, ty) = (fx - ix as f64, fy - iy as f64);
                let p = |xx: usize, yy: usize| src[yy * sw + xx];
                let top = p(ix, iy) * (1.0 - tx) + p(ix + 1, iy) * tx;
                let bot = p(ix, iy + 1) * (1.0 - tx) + p(ix + 1, iy + 1) * tx;
                out[y * dw + x] = top * (1.0 - ty) + bot * ty;
            }
        }
        out
    }

    pub fn fill(plane: &[f64], hole: &[bool], w: usize, h: usize) -> Vec<f64> {
        let mut dims = vec![(w, h)];
        loop {
            let (a, b) = *dims.last().unwrap();
            let next = ((a / 2).max(2), (b / 2).max(2));
            dims.push(next);
            if next == (2, 2) {
                break;
            }
        }
        let top = dims.len() - 1;
        let mut vals = vec![plane.iter().zip(hole).map(|(&v, &hl)| if hl { 0.0 } else { v }).collect::<Vec<f64>>()];
        let mut wts = vec![hole.iter().map(|&hl| if hl { 0.0 } else { 1.0 }).collect::<Vec<f64>>()];
        for k in 1..=top {
            let (pw, ph) = dims[k - 1];
            let (cw, ch) = dims[k];
            let v = resample(&blur(&vals[k - 1], pw, ph, &BINOMIAL), pw, ph, cw, ch);
            let wt = resample(&blur(&wts[k - 1], pw, ph, &BINOMIAL), pw, ph, cw, ch);
            vals.push(v);
            wts.push(wt);
        }
        let g = gaussian_taps(5.0);
        let (tw, th) = dims[top];
        let mut sol = vec![0.0; tw * th];
        for level in (0..=top).rev() {
            let (lw, lh) = dims[level];
            for i in 0..lw * lh {
                if wts[level][i] != 0.0 {
                    sol[i] = vals[level][i] / wts[level][i];
                }
            }
            sol = blur(&sol, lw, lh, &g);
            if level == 0 {
                for i in 0..lw * lh {
                    if wts[0][i] != 0.0 {
                        sol[i] = vals[0][i] / wts[0][i];
                    }
                }
            } else {
                let (nw, nh) = dims[level - 1];
                sol = resample(&sol, lw, lh, nw, nh);
            }
        }
        sol
    }
}

#[test]
fn the_engine_fill_agrees_with_an_independent_double_precision_reference() {
    let mut rng = Rng(1234);
    let sizes = [(2usize, 2usize), (3, 3), (7, 5), (12, 9), (17, 17), (33, 20), (5, 40), (64, 48)];
    for &(w, h) in &sizes {
        for round in 0..3 {
            let mut mask = vec![0u8; w * h];
            match round {
                0 => rect(&mut mask, w, w / 4, h / 4, (w / 4 + w / 2).max(w / 4 + 1).min(w), (h / 4 + h / 2).max(h / 4 + 1).min(h)),
                1 => mask.iter_mut().for_each(|m| {
                    if rng.below(5) == 0 {
                        *m = 255
                    }
                }),
                _ => mask[rng.below((w * h) as u64) as usize] = 255,
            }
            if mask.iter().all(|&m| m != 0) {
                mask[0] = 0;
            }
            if mask.iter().all(|&m| m == 0) {
                continue;
            }
            let img = image_from(Fmt::F32, w, h, &mut rng);
            let mut out = img.clone();
            fill_holes_masked(&mut out, HoleMaskRef::U8(&mask), false).unwrap();
            let hole: Vec<bool> = mask.iter().map(|&m| m != 0).collect();
            let a = samples_f32(&img);
            let b = samples_f32(&out);
            let mut worst = 0.0f64;
            for c in 0..4 {
                let plane: Vec<f64> = (0..w * h).map(|i| f64::from(a[i * 4 + c])).collect();
                let want = reference::fill(&plane, &hole, w, h);
                for i in 0..w * h {
                    let d = (want[i] - f64::from(b[i * 4 + c])).abs();
                    worst = worst.max(d);
                    assert!(d < 2e-4, "{w}x{h} round {round} ch {c} pixel {i}: engine {} reference {}", b[i * 4 + c], want[i]);
                }
            }
            let _ = worst;
        }
    }
}

#[test]
fn u8_and_u16_fills_are_the_float_fill_stored_with_the_integer_rule() {
    // the integer depths run the same float pipeline in native units, then `x + 0.5` truncated and clamped
    let mut rng = Rng(77);
    for fmt in [Fmt::U8, Fmt::U16] {
        let (w, h) = (19usize, 13usize);
        let img = image_from(fmt, w, h, &mut rng);
        let mut mask = vec![0u8; w * h];
        rect(&mut mask, w, 4, 3, 12, 8);
        mask[w * 11 + 17] = 255;
        let mut out = img.clone();
        fill_holes_masked(&mut out, HoleMaskRef::U8(&mask), false).unwrap();
        for c in 0..4 {
            let mut plane: Vec<f32> = (0..w * h).map(|i| as_f64(&img)[i * 4 + c] as f32).collect();
            fill_plane(&mut plane, w, h, HoleMaskRef::U8(&mask), false).unwrap();
            for i in 0..w * h {
                let want = match fmt {
                    Fmt::U8 => f64::from(store_u8(plane[i])),
                    _ => f64::from(store_u16(plane[i])),
                };
                assert_eq!(as_f64(&out)[i * 4 + c], want, "{fmt:?} pixel {i} channel {c}");
            }
        }
    }
}

#[test]
fn integer_write_back_rules() {
    // `x + 0.5f`, then `0 > y` gives 0, `y > white` gives white, else the truncation
    assert_eq!(store_u8(0.0), 0);
    assert_eq!(store_u8(0.49), 0);
    assert_eq!(store_u8(0.5), 1);
    assert_eq!(store_u8(254.49), 254);
    assert_eq!(store_u8(254.5), 255);
    assert_eq!(store_u8(255.0), 255);
    assert_eq!(store_u8(255.4), 255);
    assert_eq!(store_u8(300.0), 255);
    assert_eq!(store_u8(-0.4), 0);
    assert_eq!(store_u8(-3.0), 0);
    assert_eq!(store_u8(f32::NAN), 0); // NaN truncates to i32::MIN, whose low byte is 0
    assert_eq!(store_u16(65534.5), 65535);
    assert_eq!(store_u16(65535.4), 65535);
    assert_eq!(store_u16(70000.0), 65535);
    assert_eq!(store_u16(1234.4), 1234);
    assert_eq!(store_u16(-1.0), 0);
}

// ------------------------------------------------------------------------------------------------
// numpy float32 model fixtures
// ------------------------------------------------------------------------------------------------

#[test]
fn model_fixtures_hole_fill_is_reproduced_bit_for_bit() {
    let cases = load_cases();
    let mut n = 0;
    let mut depths = [0usize; 3];
    for (i, c) in cases.iter().enumerate().filter(|(_, c)| c.kind == 3) {
        let mask = OwnedMask::from_case(c);
        let mut img = c.p1.clone();
        fill_holes_masked(&mut img, mask.as_ref(), false).unwrap();
        if let Some(pos) = first_diff(&img, &c.out) {
            panic!(
                "case {i} (depth {}, {}x{}, mask kind {}): first difference at sample {pos}: got {:?} want {:?}",
                c.depth,
                c.w,
                c.h,
                c.mask_kind,
                as_f64(&img)[pos],
                as_f64(&c.out)[pos]
            );
        }
        n += 1;
        depths[c.depth as usize] += 1;
    }
    assert!(n >= 30 && depths.iter().all(|&d| d >= 8), "fixture mix {n} {depths:?}");
}

// ------------------------------------------------------------------------------------------------
// thread count invariance
// ------------------------------------------------------------------------------------------------

#[test]
fn fill_is_identical_for_one_and_four_threads() {
    let mut rng = Rng(31);
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let (w, h) = (97usize, 61usize);
        let img = image_from(fmt, w, h, &mut rng);
        let mut mask = vec![0u8; w * h];
        rect(&mut mask, w, 10, 10, 50, 40);
        rect(&mut mask, w, 70, 5, 75, 58);
        let run = |threads: usize| {
            let mut a = img.clone();
            in_pool(threads, || fill_holes_masked(&mut a, HoleMaskRef::U8(&mask), false).unwrap());
            a
        };
        let one = run(1);
        let four = run(4);
        assert!(same_bits(&one, &four), "{fmt:?}");
        assert!(!same_bits(&one, &img));
    }
    // and a flow
    let mut flow = FlowField::new(70, 50);
    for v in flow.data.iter_mut() {
        *v = rng.unit() * 6.0 - 3.0;
    }
    let mut mask = vec![0u8; 70 * 50];
    rect(&mut mask, 70, 20, 10, 40, 30);
    let mut a = flow.clone();
    let mut b = flow.clone();
    in_pool(1, || fill_flow_holes(&mut a, HoleMaskRef::U8(&mask), false).unwrap());
    in_pool(4, || fill_flow_holes(&mut b, HoleMaskRef::U8(&mask), false).unwrap());
    assert!(a.data.iter().zip(&b.data).all(|(x, y)| x.to_bits() == y.to_bits()));
}

// ------------------------------------------------------------------------------------------------
// the wrap-around (360) fill: fixtures of `tests/fixtures/gen_wrap.py`, rotation, identities
// ------------------------------------------------------------------------------------------------

/// One case of `tests/fixtures/wrap_cases.hex` (format in `gen_wrap.py`); shared with the tests of
/// `flow::conv`, `flow::sample` and the motion blur.
pub(crate) enum WrapCase {
    Fill { w: usize, h: usize, mask_kind: u32, img: Image, mask: Vec<u8>, out: Image },
    Blur { src: Image, flow: FlowField, amount: f32, shutter: f32, hiq: bool, out: Image },
    Conv { c: usize, w: usize, h: usize, radius: f32, data: Vec<f32>, out: Vec<f32> },
    Resample { sw: usize, sh: usize, dw: usize, dh: usize, src: Vec<f32>, out: Vec<f32> },
}

pub(crate) fn load_wrap_cases() -> Vec<WrapCase> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wrap_cases.hex");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let b = crate::fixture::hex(&text);
    assert_eq!(&b[0..4], b"WRA1");
    let mut pos = 4usize;
    let u32_next = |pos: &mut usize| -> u32 {
        let v = u32::from_le_bytes(b[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        v
    };
    let n = u32_next(&mut pos) as usize;
    let f32s = |pos: &mut usize, count: usize| -> Vec<f32> {
        let v = (0..count).map(|i| f32::from_bits(u32::from_le_bytes(b[*pos + 4 * i..*pos + 4 * i + 4].try_into().unwrap()))).collect();
        *pos += 4 * count;
        v
    };
    let image = |pos: &mut usize, depth: u32, w: usize, h: usize| -> Image {
        let fmt = match depth {
            0 => Fmt::U8,
            1 => Fmt::U16,
            _ => Fmt::F32,
        };
        let count = w * h * 4;
        let mut img = Image::new(w, h, fmt);
        match &mut img.samples {
            Samples::U8(v) => {
                v.copy_from_slice(&b[*pos..*pos + count]);
                *pos += count;
            }
            Samples::U16(v) => {
                for (i, s) in v.iter_mut().enumerate() {
                    *s = u16::from_le_bytes(b[*pos + 2 * i..*pos + 2 * i + 2].try_into().unwrap());
                }
                *pos += 2 * count;
            }
            Samples::F32(v) => {
                for (i, s) in v.iter_mut().enumerate() {
                    *s = f32::from_bits(u32::from_le_bytes(b[*pos + 4 * i..*pos + 4 * i + 4].try_into().unwrap()));
                }
                *pos += 4 * count;
            }
        }
        img
    };
    let mut cases = Vec::with_capacity(n);
    for _ in 0..n {
        match u32_next(&mut pos) {
            1 => {
                let depth = u32_next(&mut pos);
                let w = u32_next(&mut pos) as usize;
                let h = u32_next(&mut pos) as usize;
                let mask_kind = u32_next(&mut pos);
                let img = image(&mut pos, depth, w, h);
                let msz = [0, 1, 2, 4][mask_kind as usize];
                let mask = b[pos..pos + w * h * msz].to_vec();
                pos += w * h * msz;
                let out = image(&mut pos, depth, w, h);
                cases.push(WrapCase::Fill { w, h, mask_kind, img, mask, out });
            }
            2 => {
                let depth = u32_next(&mut pos);
                let (w, h) = (u32_next(&mut pos) as usize, u32_next(&mut pos) as usize);
                let (fw, fh) = (u32_next(&mut pos) as usize, u32_next(&mut pos) as usize);
                let amount = f32::from_bits(u32_next(&mut pos));
                let shutter = f32::from_bits(u32_next(&mut pos));
                let hiq = u32_next(&mut pos) != 0;
                let src = image(&mut pos, depth, w, h);
                let flow = FlowField { width: fw, height: fh, data: f32s(&mut pos, fw * fh * 2) };
                let out = image(&mut pos, depth, w, h);
                cases.push(WrapCase::Blur { src, flow, amount, shutter, hiq, out });
            }
            3 => {
                let c = u32_next(&mut pos) as usize;
                let (w, h) = (u32_next(&mut pos) as usize, u32_next(&mut pos) as usize);
                let radius = f32::from_bits(u32_next(&mut pos));
                let data = f32s(&mut pos, w * h * c);
                let out = f32s(&mut pos, w * h * c);
                cases.push(WrapCase::Conv { c, w, h, radius, data, out });
            }
            4 => {
                let (sw, sh) = (u32_next(&mut pos) as usize, u32_next(&mut pos) as usize);
                let (dw, dh) = (u32_next(&mut pos) as usize, u32_next(&mut pos) as usize);
                let src = f32s(&mut pos, sw * sh);
                let out = f32s(&mut pos, dw * dh);
                cases.push(WrapCase::Resample { sw, sh, dw, dh, src, out });
            }
            k => panic!("unknown wrap case kind {k}"),
        }
    }
    assert_eq!(pos, b.len(), "trailing bytes in the wrap fixture");
    cases
}

fn mask_ref(kind: u32, bytes: &[u8]) -> OwnedMask {
    match kind {
        1 => OwnedMask::U8(bytes.to_vec()),
        2 => OwnedMask::U16(bytes.chunks(2).map(|p| u16::from_le_bytes([p[0], p[1]])).collect()),
        _ => OwnedMask::F32(bytes.chunks(4).map(|p| f32::from_bits(u32::from_le_bytes([p[0], p[1], p[2], p[3]]))).collect()),
    }
}

#[test]
#[ignore = "the G6 wrap rule (U) this test encodes is superseded by the 360 rules read from TwixtorPro_64 (parity/lab/re/op-residual.md, finding 6)"]
fn wrap_model_fixtures_hole_fill_is_reproduced_bit_for_bit() {
    let mut n = 0;
    let mut depths = [0usize; 3];
    let mut mask_kinds = [0usize; 4];
    for (i, c) in load_wrap_cases().iter().enumerate() {
        let WrapCase::Fill { w, h, mask_kind, img, mask, out } = c else { continue };
        let m = mask_ref(*mask_kind, mask);
        let mut got = img.clone();
        fill_holes_masked(&mut got, m.as_ref(), true).unwrap();
        if let Some(pos) = first_diff(&got, out) {
            panic!(
                "wrap case {i} ({:?}, {w}x{h}, mask kind {mask_kind}): first difference at sample {pos}: got {:?} want {:?}",
                img.fmt(),
                as_f64(&got)[pos],
                as_f64(out)[pos]
            );
        }
        n += 1;
        depths[match img.fmt() {
            Fmt::U8 => 0,
            Fmt::U16 => 1,
            Fmt::F32 => 2,
        }] += 1;
        mask_kinds[*mask_kind as usize] += 1;
    }
    assert!(n >= 60 && depths.iter().all(|&d| d >= 20) && mask_kinds[1..].iter().all(|&k| k >= 10), "fixture mix {n} {depths:?} {mask_kinds:?}");
}

/// `img` rotated left by `s` columns: `out(x, y) = img((x + s) mod w, y)`.
fn rotate_image(img: &Image, s: usize) -> Image {
    let (w, h) = (img.width, img.height);
    let mut out = img.clone();
    let map = |x: usize, y: usize| ((y * w + (x + s) % w) * 4, (y * w + x) * 4);
    match (&mut out.samples, &img.samples) {
        (Samples::U8(o), Samples::U8(i)) => (0..h).for_each(|y| (0..w).for_each(|x| {
            let (a, b) = map(x, y);
            o[b..b + 4].copy_from_slice(&i[a..a + 4]);
        })),
        (Samples::U16(o), Samples::U16(i)) => (0..h).for_each(|y| (0..w).for_each(|x| {
            let (a, b) = map(x, y);
            o[b..b + 4].copy_from_slice(&i[a..a + 4]);
        })),
        (Samples::F32(o), Samples::F32(i)) => (0..h).for_each(|y| (0..w).for_each(|x| {
            let (a, b) = map(x, y);
            o[b..b + 4].copy_from_slice(&i[a..a + 4]);
        })),
        _ => unreachable!(),
    }
    out
}

fn rotate_plane<T: Copy>(p: &[T], w: usize, h: usize, s: usize) -> Vec<T> {
    let mut out = p.to_vec();
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = p[y * w + (x + s) % w];
        }
    }
    out
}

#[test]
fn a_seam_hole_is_filled_like_the_same_hole_in_the_middle_of_the_rotated_image() {
    // With power of two widths every level halves exactly, so a rotation by half the width is a
    // rotation of every level and the wrapped fill commutes with it bit for bit: the fill of a
    // hole across the left and right border equals the fill of the rotated image, whose hole is
    // in the middle, rotated back.
    let mut rng = Rng(4242);
    for &(w, h) in &[(64usize, 40usize), (32, 9), (16, 2), (8, 21), (2, 6)] {
        for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
            let img = image_from(fmt, w, h, &mut rng);
            let mut mask = vec![0u8; w * h];
            let r = (w / 8).max(1);
            rect(&mut mask, w, 0, h / 4, r, (h / 4 + h / 2).max(h / 4 + 1));
            rect(&mut mask, w, w - r, h / 4, w, (h / 4 + h / 2).max(h / 4 + 1));
            mask[(h - 1) * w + w / 2] = 255; // and a hole far from the seam
            let mut seam = img.clone();
            fill_holes_masked(&mut seam, HoleMaskRef::U8(&mask), true).unwrap();
            let half = w / 2;
            let mut middle = rotate_image(&img, half);
            let rmask = rotate_plane(&mask, w, h, half);
            fill_holes_masked(&mut middle, HoleMaskRef::U8(&rmask), true).unwrap();
            let back = rotate_image(&middle, w - half);
            assert!(first_diff(&seam, &back).is_none(), "{w}x{h} {fmt:?}: first difference at {:?}", first_diff(&seam, &back));
            assert!(!same_bits(&seam, &img));
            // the plain fill has no such symmetry: its borders are renormalised, not continued
            if w >= 8 && h >= 8 {
                let mut plain = img.clone();
                fill_holes_masked(&mut plain, HoleMaskRef::U8(&mask), false).unwrap();
                assert!(!same_bits(&plain, &seam), "{w}x{h} {fmt:?}");
            }
        }
    }
    // the flow fill (Vectors In) is the same plane fill
    let (w, h) = (32usize, 16usize);
    let mut flow = FlowField::new(w, h);
    for v in flow.data.iter_mut() {
        *v = rng.unit() * 8.0 - 4.0;
    }
    let mut mask = vec![0u8; w * h];
    rect(&mut mask, w, 0, 4, 3, 12);
    rect(&mut mask, w, 29, 4, 32, 12);
    let mut seam = flow.clone();
    fill_flow_holes(&mut seam, HoleMaskRef::U8(&mask), true).unwrap();
    let rot = |f: &FlowField, s: usize| -> FlowField {
        let mut o = f.clone();
        for y in 0..h {
            for x in 0..w {
                for c in 0..2 {
                    o.data[(y * w + x) * 2 + c] = f.data[(y * w + (x + s) % w) * 2 + c];
                }
            }
        }
        o
    };
    let mut middle = rot(&flow, w / 2);
    fill_flow_holes(&mut middle, HoleMaskRef::U8(&rotate_plane(&mask, w, h, w / 2)), true).unwrap();
    let back = rot(&middle, w / 2);
    assert!(seam.data.iter().zip(&back.data).all(|(a, b)| a.to_bits() == b.to_bits()));
}

#[test]
fn a_seam_hole_takes_the_colours_of_both_sides() {
    // left half 40, right half 200 (the seam lies between the last and the first column): a hole
    // on the first columns is filled from both sides with the flag, from the right only without
    let (w, h) = (64usize, 32usize);
    let mut img = Image::new(w, h, Fmt::F32);
    if let Samples::F32(v) = &mut img.samples {
        for y in 0..h {
            for x in 0..w {
                let c = if x < w / 2 { 40.0 } else { 200.0 };
                v[(y * w + x) * 4..(y * w + x) * 4 + 4].copy_from_slice(&[c, c, c, c]);
            }
        }
    }
    let mut mask = vec![0u8; w * h];
    rect(&mut mask, w, 0, 10, 4, 22);
    let mut wrap = img.clone();
    fill_holes_masked(&mut wrap, HoleMaskRef::U8(&mask), true).unwrap();
    let mut plain = img.clone();
    fill_holes_masked(&mut plain, HoleMaskRef::U8(&mask), false).unwrap();
    let at = |im: &Image, x: usize| samples_f32(im)[(16 * w + x) * 4];
    // the hole's centre lies 2 columns from the 40 side and 2 from the 200 side across the seam
    for x in 1..3 {
        let (cw, cp) = (at(&wrap, x), at(&plain, x));
        assert!((cw - 120.0).abs() < 30.0, "wrapped fill at column {x}: {cw}");
        assert!(cp < 70.0, "plain fill at column {x}: {cp}");
    }
    // rotated by half a width the hole lies in the middle, where the plain fill sees both sides
    // too: the two agree closely (not bit for bit: the coarse levels differ)
    let mut middle = rotate_image(&img, w / 2);
    fill_holes_masked(&mut middle, HoleMaskRef::U8(&rotate_plane(&mask, w, h, w / 2)), false).unwrap();
    let back = rotate_image(&middle, w / 2);
    for y in 10..22 {
        for x in 0..4 {
            let (a, b) = (samples_f32(&wrap)[(y * w + x) * 4], samples_f32(&back)[(y * w + x) * 4]);
            assert!((a - b).abs() < 8.0, "({x}, {y}): wrapped {a} against the rotated plain fill {b}");
        }
    }
}

#[test]
fn the_wrap_fill_without_holes_is_the_identity() {
    let mut rng = Rng(77);
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        for &(w, h) in &[(2usize, 2usize), (13, 9), (1, 7), (7, 1), (64, 3)] {
            let img = image_from(fmt, w, h, &mut rng);
            let mut out = img.clone();
            fill_holes_masked(&mut out, HoleMaskRef::U8(&vec![0u8; w * h]), true).unwrap();
            assert!(same_bits(&img, &out), "{fmt:?} {w}x{h} u8");
            // values below the thresholds are not holes
            fill_holes_masked(&mut out, HoleMaskRef::U16(&vec![0u16; w * h]), true).unwrap();
            fill_holes_masked(&mut out, HoleMaskRef::F32(&vec![3.0e-5f32; w * h]), true).unwrap();
            assert!(same_bits(&img, &out), "{fmt:?} {w}x{h} wide masks");
        }
    }
    let mut plane: Vec<f32> = (0..30).map(|i| i as f32 * 0.5 - 3.0).collect();
    let keep = plane.clone();
    fill_plane(&mut plane, 6, 5, HoleMaskRef::U8(&[0u8; 30]), true).unwrap();
    assert_eq!(plane, keep);
    let mut flow = FlowField::new(6, 5);
    flow.data.iter_mut().enumerate().for_each(|(i, v)| *v = i as f32);
    let keep = flow.clone();
    fill_flow_holes(&mut flow, HoleMaskRef::U8(&[0u8; 30]), true).unwrap();
    assert_eq!(flow, keep);
}

#[test]
fn the_wrap_fill_keeps_every_known_pixel_bit_exact() {
    let mut rng = Rng(12);
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        for &(w, h) in &[(2usize, 2usize), (5, 7), (16, 16), (33, 20)] {
            let img = image_from(fmt, w, h, &mut rng);
            let mut mask = vec![0u8; w * h];
            for m in mask.iter_mut() {
                if rng.below(4) == 0 {
                    *m = 255;
                }
            }
            mask[0] = 255;
            let mut out = img.clone();
            fill_holes_masked(&mut out, HoleMaskRef::U8(&mask), true).unwrap();
            let (a, b) = (as_f64(&img), as_f64(&out));
            for i in (0..w * h).filter(|&i| mask[i] == 0) {
                for c in 0..4 {
                    assert_eq!(a[i * 4 + c].to_bits(), b[i * 4 + c].to_bits(), "{fmt:?} {w}x{h} pixel {i} channel {c}");
                }
            }
        }
    }
}

mod frozen {
    //! The plain fill as it was before the wrap-around passes existed, kept verbatim on the
    //! unchanged primitives: the flag off must stay equal to it bit for bit.
    use crate::flow::conv::{blur5, blur_gaussian};
    use crate::flow::sample::resample1;
    use crate::holefill::{geometry, GAUSS_RADIUS};

    pub fn fill(plane: &mut [f32], hole: &[bool], w: usize, h: usize) {
        let geo = geometry(w, h);
        let mut wts: Vec<Vec<f32>> = vec![hole.iter().map(|&b| if b { 0.0f32 } else { 1.0f32 }).collect()];
        let mut v0 = plane.to_vec();
        for (v, &hl) in v0.iter_mut().zip(hole) {
            if hl {
                *v = 0.0;
            }
        }
        let mut vals = vec![v0];
        for k in 1..=geo.top {
            let (pw, ph) = geo.dims[k - 1];
            let (cw, ch) = geo.dims[k];
            let mut t = wts[k - 1].clone();
            blur5(&mut t, pw, ph);
            wts.push(resample1(&t, pw, ph, cw, ch));
            let mut t = vals[k - 1].clone();
            blur5(&mut t, pw, ph);
            vals.push(resample1(&t, pw, ph, cw, ch));
        }
        let combine = |sol: &mut [f32], v: &[f32], wt: &[f32]| {
            for ((s, &v), &wt) in sol.iter_mut().zip(v).zip(wt) {
                if wt != 0.0 {
                    *s = v / wt;
                }
            }
        };
        let (tw, th) = geo.dims[geo.top];
        let mut sol = vec![0.0f32; tw * th];
        for level in (0..=geo.top).rev() {
            let (lw, lh) = geo.dims[level];
            combine(&mut sol, &vals[level], &wts[level]);
            blur_gaussian(&mut sol, lw, lh, 1, GAUSS_RADIUS);
            if level == 0 {
                combine(&mut sol, &vals[0], &wts[0]);
            } else {
                let (nw, nh) = geo.dims[level - 1];
                sol = resample1(&sol, lw, lh, nw, nh);
            }
        }
        plane.copy_from_slice(&sol);
    }
}

#[test]
fn the_flag_off_equals_the_plain_fill_of_before_bit_for_bit() {
    let mut rng = Rng(2027);
    let sizes = [(2usize, 2usize), (3, 3), (2, 9), (31, 3), (7, 5), (16, 16), (33, 20), (64, 48), (5, 40)];
    for &(w, h) in &sizes {
        for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
            let img = image_from(fmt, w, h, &mut rng);
            let mut mask = vec![0u8; w * h];
            rect(&mut mask, w, 0, 0, (w / 3).max(1), (h / 2).max(1));
            for m in mask.iter_mut() {
                if rng.below(9) == 0 {
                    *m = 255;
                }
            }
            if mask.iter().all(|&m| m != 0) {
                mask[w * h - 1] = 0;
            }
            let hole: Vec<bool> = mask.iter().map(|&m| m != 0).collect();
            let mut got = img.clone();
            fill_holes_masked(&mut got, HoleMaskRef::U8(&mask), false).unwrap();
            let src = as_f64(&img);
            let dst = as_f64(&got);
            for c in 0..4 {
                let mut plane: Vec<f32> = (0..w * h).map(|i| src[i * 4 + c] as f32).collect();
                frozen::fill(&mut plane, &hole, w, h);
                for i in 0..w * h {
                    let want = match fmt {
                        Fmt::U8 => f64::from(store_u8(plane[i])),
                        Fmt::U16 => f64::from(store_u16(plane[i])),
                        Fmt::F32 => f64::from(plane[i]),
                    };
                    assert_eq!(dst[i * 4 + c].to_bits(), want.to_bits(), "{w}x{h} {fmt:?} pixel {i} channel {c}");
                }
            }
        }
    }
}

mod reference_wrap {
    //! A double precision version of the wrapped fill, re-derived without engine code: a circular
    //! blur along x (no renormalisation), the renormalised blur along y, a periodic resample in x.
    use super::reference::{gaussian_taps, BINOMIAL};

    fn blur(img: &[f64], w: usize, h: usize, k: &[f64]) -> Vec<f64> {
        let n = k.len() as i64;
        let mut tmp = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut s = 0.0;
                for j in -(n - 1)..n {
                    let xx = (x as i64 + j).rem_euclid(w as i64) as usize;
                    s += k[j.unsigned_abs() as usize] * img[y * w + xx];
                }
                tmp[y * w + x] = s;
            }
        }
        let mut out = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let (mut num, mut den) = (0.0, 0.0);
                for j in -(n - 1)..n {
                    let yy = y as i64 + j;
                    if yy >= 0 && yy < h as i64 {
                        let wt = k[j.unsigned_abs() as usize];
                        num += wt * tmp[yy as usize * w + x];
                        den += wt;
                    }
                }
                out[y * w + x] = num / den;
            }
        }
        out
    }

    fn resample(src: &[f64], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f64> {
        let mut out = vec![0.0; dw * dh];
        for y in 0..dh {
            for x in 0..dw {
                let fx = x as f64 * sw as f64 / dw as f64;
                let fy = y as f64 * (sh as f64 - 1.0) / (dh as f64 - 1.0);
                let ix = fx.floor() as usize;
                let x1 = (ix + 1) % sw;
                let iy = (fy.floor() as usize).min(sh - 2);
                let (tx, ty) = (fx - ix as f64, fy - iy as f64);
                let p = |xx: usize, yy: usize| src[yy * sw + xx];
                let top = p(ix, iy) * (1.0 - tx) + p(x1, iy) * tx;
                let bot = p(ix, iy + 1) * (1.0 - tx) + p(x1, iy + 1) * tx;
                out[y * dw + x] = top * (1.0 - ty) + bot * ty;
            }
        }
        out
    }

    pub fn fill(plane: &[f64], hole: &[bool], w: usize, h: usize) -> Vec<f64> {
        let mut dims = vec![(w, h)];
        loop {
            let (a, b) = *dims.last().unwrap();
            let next = ((a / 2).max(2), (b / 2).max(2));
            dims.push(next);
            if next == (2, 2) {
                break;
            }
        }
        let top = dims.len() - 1;
        let mut vals = vec![plane.iter().zip(hole).map(|(&v, &hl)| if hl { 0.0 } else { v }).collect::<Vec<f64>>()];
        let mut wts = vec![hole.iter().map(|&hl| if hl { 0.0 } else { 1.0 }).collect::<Vec<f64>>()];
        for k in 1..=top {
            let (pw, ph) = dims[k - 1];
            let (cw, ch) = dims[k];
            vals.push(resample(&blur(&vals[k - 1], pw, ph, &BINOMIAL), pw, ph, cw, ch));
            wts.push(resample(&blur(&wts[k - 1], pw, ph, &BINOMIAL), pw, ph, cw, ch));
        }
        let g = gaussian_taps(5.0);
        let (tw, th) = dims[top];
        let mut sol = vec![0.0; tw * th];
        for level in (0..=top).rev() {
            let (lw, lh) = dims[level];
            for i in 0..lw * lh {
                if wts[level][i] != 0.0 {
                    sol[i] = vals[level][i] / wts[level][i];
                }
            }
            sol = blur(&sol, lw, lh, &g);
            if level == 0 {
                for i in 0..lw * lh {
                    if wts[0][i] != 0.0 {
                        sol[i] = vals[0][i] / wts[0][i];
                    }
                }
            } else {
                let (nw, nh) = dims[level - 1];
                sol = resample(&sol, lw, lh, nw, nh);
            }
        }
        sol
    }
}

#[test]
#[ignore = "the G6 wrap rule (U) this test encodes is superseded by the 360 rules read from TwixtorPro_64 (parity/lab/re/op-residual.md, finding 6)"]
fn the_wrap_fill_agrees_with_an_independent_double_precision_reference() {
    let mut rng = Rng(5678);
    let sizes = [(2usize, 2usize), (3, 3), (7, 5), (12, 9), (17, 17), (33, 20), (5, 40), (64, 12)];
    for &(w, h) in &sizes {
        for round in 0..3 {
            let mut mask = vec![0u8; w * h];
            match round {
                0 => {
                    rect(&mut mask, w, 0, h / 4, (w / 5).max(1), (h / 4 + h / 2).max(h / 4 + 1));
                    rect(&mut mask, w, w - (w / 5).max(1), h / 4, w, (h / 4 + h / 2).max(h / 4 + 1));
                }
                1 => mask.iter_mut().for_each(|m| {
                    if rng.below(5) == 0 {
                        *m = 255
                    }
                }),
                _ => mask[rng.below((w * h) as u64) as usize] = 255,
            }
            if mask.iter().all(|&m| m != 0) {
                mask[0] = 0;
            }
            if mask.iter().all(|&m| m == 0) {
                continue;
            }
            let img = image_from(Fmt::F32, w, h, &mut rng);
            let mut out = img.clone();
            fill_holes_masked(&mut out, HoleMaskRef::U8(&mask), true).unwrap();
            let hole: Vec<bool> = mask.iter().map(|&m| m != 0).collect();
            let (a, b) = (samples_f32(&img), samples_f32(&out));
            for c in 0..4 {
                let plane: Vec<f64> = (0..w * h).map(|i| f64::from(a[i * 4 + c])).collect();
                let want = reference_wrap::fill(&plane, &hole, w, h);
                for i in 0..w * h {
                    let d = (want[i] - f64::from(b[i * 4 + c])).abs();
                    assert!(d < 2e-4, "{w}x{h} round {round} ch {c} pixel {i}: engine {} reference {}", b[i * 4 + c], want[i]);
                }
            }
        }
    }
}

#[test]
fn the_wrap_fill_is_identical_for_one_and_four_threads() {
    let mut rng = Rng(32);
    for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
        let (w, h) = (97usize, 61usize);
        let img = image_from(fmt, w, h, &mut rng);
        let mut mask = vec![0u8; w * h];
        rect(&mut mask, w, 0, 10, 12, 40);
        rect(&mut mask, w, 90, 5, 97, 58);
        rect(&mut mask, w, 40, 20, 60, 30);
        let run = |threads: usize| {
            let mut a = img.clone();
            in_pool(threads, || fill_holes_masked(&mut a, HoleMaskRef::U8(&mask), true).unwrap());
            a
        };
        let one = run(1);
        let four = run(4);
        assert!(same_bits(&one, &four), "{fmt:?}");
        assert!(!same_bits(&one, &img));
    }
}

/// Timing probe (run with `--release -- --ignored --nocapture`): 1920x1080 u8, a large block and scattered holes.
#[test]
#[ignore]
fn timing_probe_1080p() {
    let mut rng = Rng(5);
    let (w, h) = (1920usize, 1080usize);
    let mut img = image_from(Fmt::U8, w, h, &mut rng);
    let mut mask = vec![0u8; w * h];
    rect(&mut mask, w, 400, 300, 900, 700);
    for i in 0..2000 {
        mask[(i * 977) % (w * h)] = 255;
    }
    let t = std::time::Instant::now();
    fill_holes_masked(&mut img, HoleMaskRef::U8(&mask), false).unwrap();
    println!("[f1] hole fill 1920x1080 u8, 4 channels: {:?}", t.elapsed());
    let mut fl = FlowField::new(w / 2, h / 2);
    let m2 = vec![255u8; (w / 2) * (h / 2)].iter().enumerate().map(|(i, _)| if i % 7 == 0 { 255 } else { 0 }).collect::<Vec<u8>>();
    let t = std::time::Instant::now();
    fill_flow_holes(&mut fl, HoleMaskRef::U8(&m2), false).unwrap();
    println!("[f1] flow fill 960x540: {:?}", t.elapsed());
}

// ------------------------------------------------------------------------------------------------
// NaN colours: the fill is repeated with the NaN rule (crate::nanrule) when its result holds a NaN
// ------------------------------------------------------------------------------------------------

#[test]
fn the_fill_with_the_nan_rule_gives_the_plain_numbers_in_every_mode() {
    let (w, h) = (19usize, 13usize);
    let mut rng = Rng(0x5eed_0f6f);
    let mut mask = vec![0u8; w * h];
    rect(&mut mask, w, 0, 0, w, 3);
    rect(&mut mask, w, 0, 0, 4, h);
    rect(&mut mask, w, 9, 5, 14, 9);
    let num: Vec<f32> = (0..w * h).map(|_| rng.unit() * 300.0 - 100.0).collect();
    let mut special = num.clone();
    special[6 * w + 6] = f32::INFINITY;
    special[6 * w + 7] = f32::NEG_INFINITY;
    special[10 * w + 16] = f32::from_bits(0x7fc0_0000);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    for is360 in [false, true] {
        for taps in [VerticalTaps::BelowFirst, VerticalTaps::AboveFirst] {
            let f = Filler::new(w, h, HoleMaskRef::U8(&mask), is360, taps).unwrap().unwrap();
            assert_eq!(bits(&f.solve(&num, true)), bits(&f.solve(&num, false)), "numbers {is360} {taps:?}");
            let (r, p) = (f.solve(&special, true), f.solve(&special, false));
            assert!(r.iter().any(|v| v.is_nan()), "the infinities make NaNs");
            for (i, (a, b)) in r.iter().zip(&p).enumerate() {
                assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()), "sample {i} {is360} {taps:?}");
            }
            // a plane without NaN in its result is filled once, with the plain operations
            let mut once = num.clone();
            f.fill(&mut once);
            assert_eq!(bits(&once), bits(&f.solve(&num, false)));
        }
    }
}

#[cfg(feature = "emulator-nan")]
#[test]
fn where_two_nans_meet_in_the_fill_the_nan_rule_picks_the_result() {
    // a positive and a negative quiet NaN side by side, and an infinity next to its negative (their sums make the
    // default NaN 0xffc00000): with the emulator's rule every NaN the fill makes is the positive one, whatever the
    // order of the operands; the known negative NaN keeps its own bits
    let (w, h) = (16usize, 12usize);
    let mut mask = vec![0u8; w * h];
    rect(&mut mask, w, 0, 0, w, 4);
    let (pos, neg) = (0x7fc0_0000u32, 0xffc0_0000u32);
    let mut plane: Vec<f32> = (0..w * h).map(|i| (i % 7) as f32 * 0.25).collect();
    plane[6 * w + 5] = f32::from_bits(neg);
    plane[6 * w + 6] = f32::from_bits(pos);
    plane[9 * w + 10] = f32::INFINITY;
    plane[9 * w + 11] = f32::NEG_INFINITY;
    for taps in [VerticalTaps::BelowFirst, VerticalTaps::AboveFirst] {
        let mut p = plane.clone();
        fill_plane_with(&mut p, w, h, HoleMaskRef::U8(&mask), false, taps).unwrap();
        assert!(p[..4 * w].iter().all(|v| v.is_nan()), "every hole sees the NaNs");
        for (i, v) in p.iter().enumerate() {
            if i == 6 * w + 5 {
                assert_eq!(v.to_bits(), neg, "a known pixel keeps its NaN");
            } else if v.is_nan() {
                assert_eq!(v.to_bits(), pos, "pixel ({}, {}) {taps:?}", i % w, i / w);
            }
        }
        assert_eq!(p[9 * w + 10], f32::INFINITY, "a known infinity comes back");
    }
}

// A traced fill of the original: `specbox1-w1` (fgbg-layers, 32 bpc,
// 16x12) frame 1, FG1's plate of side A, colour channel 2, the plane handed to the plugin's single channel fill
// (equal to the rewrite's plate bit for bit) and the plane the plugin wrote back (the image at the channel copy
// right after the fill), both in the rewrite's rows, with the holes of the plate's mask (`1` a hole).
#[cfg(feature = "emulator-nan")]
const TP_HOLES: &str = "000000000000000000000000000000000000000000000000000000000000000000011111100000000001111110000000000111111000000000011111100000000001111110000000000111111000000000000000000000000000000000000000";
#[cfg(feature = "emulator-nan")]
const TP_INPUT: [u32; 192] = [
    0x3ecde0c5, 0x3ef352c6, 0x3f1eb156, 0x3f462d47, 0x3f46d54d, 0x3f5fbbba, 0x3f4590bb, 0x3f40bc70, 0x3f1baa70, 0x3ef2e040, 0x3ebee233, 0x3e85ba8a, 0x3e57caef, 0x3e16e878, 0x3e500531, 0x3e736ad7,
    0x3f218c02, 0x3f452857, 0x3f4a8c43, 0x3f5f2e72, 0x3f56c10a, 0x3f4bcb44, 0x3f257d4e, 0x3f0b4be2, 0x3ebee1d9, 0x3e9479bf, 0x3e5ce4f1, 0x3e245343, 0x3e603bf2, 0x3e52e5c6, 0x3ead5dc5, 0x3f077c41,
    0x3f4de3ee, 0x3f4dc934, 0x3f631b52, 0x3f4eaace, 0x3f384c7f, 0x3f17d766, 0x3ed8e6b4, 0x3ea9f465, 0x3e58ab71, 0x3e19f57b, 0x3e35ce2c, 0x3e678ad8, 0x3eba96c2, 0x3eea41c9, 0x3f0e0012, 0x3f3c7b0a,
    0x3f605fe1, 0x3f51095e, 0x3f33f85a, 0x3f167dca, 0x3efc88f4, 0x3eba0bb4, 0x3e68bd28, 0x3e020587, 0x3e478d6c, 0x3e3e7472, 0x3e818b81, 0x3ecb30a0, 0x3f09a0a9, 0x3f36caeb, 0x3f52a028, 0x3f615e15,
    0x3f3d1279, 0x3f2bfe05, 0x3eff455a, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xffc00000, 0xffc00000, 0x3ed40f97, 0x3efa0985, 0x3f1f6f7c, 0x3f4d10f0, 0x3f580087, 0x3f5dc2d8, 0x3f4b6e6d,
    0x3f0c0225, 0x3eac1151, 0x3e5c1a7a, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xffc00000, 0x3f16eb2f, 0x3f3e3af7, 0x3f5b0575, 0x3f5fc5db, 0x3f4ca8fc, 0x3f35babf, 0x3f16e7a3,
    0x3e781816, 0x3e6b1795, 0x3e2b8ce5, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f58098b, 0x3f573cce, 0x3f603c23, 0x3f42eb1b, 0x3f1bd49e, 0x3f0c6865, 0x3ece9a6d,
    0x3e529a08, 0x3e1c4d8c, 0x7fc00000, 0x7fc00000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f548183, 0x3f506469, 0x3f235ce1, 0x3f0be1da, 0x3ed74d0b, 0x3e8bb7cb, 0x3e223a18,
    0x3e1f86eb, 0x3e8b6129, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f31711a, 0x3f0c6da4, 0x3ee06dfe, 0x3e92e4f9, 0x3e7ed582, 0x3e1e2598, 0x3de2f479,
    0x3ebaf385, 0x3ef8cd07, 0xffc00000, 0x7fc00000, 0x7fc00000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3ef22caf, 0x3ea82aed, 0x3e43824d, 0x3e41dfa8, 0x3e049226, 0x3e487ca4, 0x3e98d66d,
    0x3f1ff308, 0x3f3d4ba9, 0x3f54094b, 0x3f539b1a, 0x3f59cd73, 0x3f406a01, 0x3f1cf8ab, 0x3f0af8b5, 0x3eb0b0bb, 0x3e98f11f, 0x3e10ea3b, 0x3e25cba5, 0x3e3b2b2b, 0x3e84c412, 0x3eb21cb5, 0x3f09ff20,
    0x3f55278f, 0x3f52df3d, 0x3f4b92ef, 0x3f52fa78, 0x3f3318ee, 0x3f10cd3d, 0x3ecb7e2d, 0x3e8aa715, 0x3e66b6f5, 0x3e174a14, 0x3debf9ed, 0x3e82e055, 0x3e9ed9fe, 0x3ef29d27, 0x3f25caaf, 0x3f31693a,
];
#[cfg(feature = "emulator-nan")]
const TP_PLUGIN: [u32; 192] = [
    0x3ecde0c5, 0x3ef352c6, 0x3f1eb156, 0x3f462d47, 0x3f46d54d, 0x3f5fbbba, 0x3f4590bb, 0x3f40bc70, 0x3f1baa70, 0x3ef2e040, 0x3ebee233, 0x3e85ba8a, 0x3e57caef, 0x3e16e878, 0x3e500531, 0x3e736ad7,
    0x3f218c02, 0x3f452857, 0x3f4a8c43, 0x3f5f2e72, 0x3f56c10a, 0x3f4bcb44, 0x3f257d4e, 0x3f0b4be2, 0x3ebee1d9, 0x3e9479bf, 0x3e5ce4f1, 0x3e245343, 0x3e603bf2, 0x3e52e5c6, 0x3ead5dc5, 0x3f077c41,
    0x3f4de3ee, 0x3f4dc934, 0x3f631b52, 0x3f4eaace, 0x3f384c7f, 0x3f17d766, 0x3ed8e6b4, 0x3ea9f465, 0x3e58ab71, 0x3e19f57b, 0x3e35ce2c, 0x3e678ad8, 0x3eba96c2, 0x3eea41c9, 0x3f0e0012, 0x3f3c7b0a,
    0x3f605fe1, 0x3f51095e, 0x3f33f85a, 0x3f167dca, 0x3efc88f4, 0x3eba0bb4, 0x3e68bd28, 0x3e020587, 0x3e478d6c, 0x3e3e7472, 0x3e818b81, 0x3ecb30a0, 0x3f09a0a9, 0x3f36caeb, 0x3f52a028, 0x3f615e15,
    0x3f3d1279, 0x3f2bfe05, 0x3eff455a, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x3ed40f97, 0x3efa0985, 0x3f1f6f7c, 0x3f4d10f0, 0x3f580087, 0x3f5dc2d8, 0x3f4b6e6d,
    0x3f0c0225, 0x3eac1151, 0x3e5c1a7a, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x3f16eb2f, 0x3f3e3af7, 0x3f5b0575, 0x3f5fc5db, 0x3f4ca8fc, 0x3f35babf, 0x3f16e7a3,
    0x3e781816, 0x3e6b1795, 0x3e2b8ce5, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x3f58098b, 0x3f573cce, 0x3f603c23, 0x3f42eb1b, 0x3f1bd49e, 0x3f0c6865, 0x3ece9a6d,
    0x3e529a08, 0x3e1c4d8c, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x3f548183, 0x3f506469, 0x3f235ce1, 0x3f0be1da, 0x3ed74d0b, 0x3e8bb7cb, 0x3e223a18,
    0x3e1f86eb, 0x3e8b6129, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x3f31711a, 0x3f0c6da4, 0x3ee06dfe, 0x3e92e4f9, 0x3e7ed582, 0x3e1e2598, 0x3de2f479,
    0x3ebaf385, 0x3ef8cd07, 0xffc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x3ef22caf, 0x3ea82aed, 0x3e43824d, 0x3e41dfa8, 0x3e049226, 0x3e487ca4, 0x3e98d66d,
    0x3f1ff308, 0x3f3d4ba9, 0x3f54094b, 0x3f539b1a, 0x3f59cd73, 0x3f406a01, 0x3f1cf8ab, 0x3f0af8b5, 0x3eb0b0bb, 0x3e98f11f, 0x3e10ea3b, 0x3e25cba5, 0x3e3b2b2b, 0x3e84c412, 0x3eb21cb5, 0x3f09ff20,
    0x3f55278f, 0x3f52df3d, 0x3f4b92ef, 0x3f52fa78, 0x3f3318ee, 0x3f10cd3d, 0x3ecb7e2d, 0x3e8aa715, 0x3e66b6f5, 0x3e174a14, 0x3debf9ed, 0x3e82e055, 0x3e9ed9fe, 0x3ef29d27, 0x3f25caaf, 0x3f31693a,
];

#[cfg(feature = "emulator-nan")]
#[test]
fn a_traced_fill_of_nan_colours_is_the_plugins_bit_for_bit() {
    let (w, h) = (16usize, 12usize);
    let mask: Vec<u8> = TP_HOLES.bytes().map(|b| if b == b'1' { 255 } else { 0 }).collect();
    assert_eq!(mask.len(), w * h);
    let mut plane: Vec<f32> = TP_INPUT.iter().map(|&b| f32::from_bits(b)).collect();
    fill_plane_with(&mut plane, w, h, HoleMaskRef::U8(&mask), false, VerticalTaps::AboveFirst).unwrap();
    let got: Vec<u32> = plane.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, TP_PLUGIN.to_vec());
    // the plane holds NaNs of both signs, so the fill went through the NaN rule
    assert!(TP_INPUT.contains(&0x7fc0_0000) && TP_INPUT.contains(&0xffc0_0000));
}
