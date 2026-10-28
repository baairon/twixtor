//! Tests of the matte primitives: the channel map, identities (a choke of 0, a premultiply by
//! ones, OVER onto and of transparent pixels at every depth), hand goldens of every rule (the
//! shrink of a straight edge, its growth and fractional mix, the layer order, the OVER weights,
//! the premultiply rounding, OVER), an exhaustive u8 OVER property test against an f64 reference,
//! the numpy float32 model fixtures (`tests/fixtures/gen_matte.py`) and thread count invariance.

use super::*;
use crate::image::Fmt;

const FMTS: [Fmt; 3] = [Fmt::U8, Fmt::U16, Fmt::F32];

pub(crate) fn in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(f)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Small deterministic generator (xorshift64*), for test data only.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }
    pub(crate) fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    pub(crate) fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// A value in 0..1 with 24 random bits.
    pub(crate) fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / 16_777_216.0
    }
}

/// An image whose sample `(x, y, c)` is `f(x, y, c)` in units of white (0..1), rounded to the
/// integer depths.
fn image_of(w: usize, h: usize, fmt: Fmt, f: impl Fn(usize, usize, usize) -> f64) -> Image {
    let mut img = Image::new(w, h, fmt);
    let white = f64::from(fmt.white());
    for y in 0..h {
        for x in 0..w {
            for c in 0..CHANNELS {
                let i = (y * w + x) * CHANNELS + c;
                let v = f(x, y, c);
                match &mut img.samples {
                    Samples::U8(s) => s[i] = (v * white + 0.5).floor() as u8,
                    Samples::U16(s) => s[i] = (v * white + 0.5).floor() as u16,
                    Samples::F32(s) => s[i] = v as f32,
                }
            }
        }
    }
    img
}

/// A random valid premultiplied image (colour at most alpha), some pixels transparent or opaque.
fn random_image(rng: &mut Rng, w: usize, h: usize, fmt: Fmt) -> Image {
    let mut img = Image::new(w, h, fmt);
    let white = f64::from(fmt.white());
    for p in 0..w * h {
        let pick = rng.below(10);
        let a = if pick < 2 { 0.0 } else if pick < 5 { 1.0 } else { f64::from(rng.unit()) };
        let px = [a, a * f64::from(rng.unit()), a * f64::from(rng.unit()), a * f64::from(rng.unit())];
        for c in 0..CHANNELS {
            let i = p * CHANNELS + c;
            match &mut img.samples {
                Samples::U8(s) => s[i] = (px[c] * white + 0.5).floor() as u8,
                Samples::U16(s) => s[i] = (px[c] * white + 0.5).floor() as u16,
                Samples::F32(s) => s[i] = px[c] as f32,
            }
        }
    }
    img
}

fn random_matte(rng: &mut Rng, w: usize, h: usize) -> Matte {
    let data = (0..w * h)
        .map(|_| match rng.below(4) {
            0 => 0.0,
            1 => 1.0,
            _ => rng.unit(),
        })
        .collect();
    Matte { width: w, height: h, data }
}

fn same_image(a: &Image, b: &Image) -> bool {
    let same = match (&a.samples, &b.samples) {
        (Samples::U8(x), Samples::U8(y)) => x == y,
        (Samples::U16(x), Samples::U16(y)) => x == y,
        (Samples::F32(x), Samples::F32(y)) => bits(x) == bits(y),
        _ => false,
    };
    same && a.width == b.width && a.height == b.height
}

fn sample(img: &Image, i: usize) -> f64 {
    match &img.samples {
        Samples::U8(s) => f64::from(s[i]),
        Samples::U16(s) => f64::from(s[i]),
        Samples::F32(s) => f64::from(s[i]),
    }
}

/// A vertical straight edge: 1.0 left of column `edge`, 0.0 from it on.
fn edge_matte(w: usize, h: usize, edge: usize) -> Matte {
    let data = (0..w * h).map(|i| if i % w < edge { 1.0 } else { 0.0 }).collect();
    Matte { width: w, height: h, data }
}

// ------------------------------------------------------------------------------------------------
// extract
// ------------------------------------------------------------------------------------------------

#[test]
fn the_channel_popup_names_r_g_b_and_alpha() {
    assert_eq!([1, 2, 3, 4].map(matte_channel), [1, 2, 3, 0]);
    assert_eq!([R, G, B, A], [1, 2, 3, 0]);
    // values the popup never gives read alpha, the last entry
    assert_eq!([0, 5, -1, i32::MAX].map(matte_channel), [0, 0, 0, 0]);
}

#[test]
fn extract_takes_the_chosen_channel_over_white_at_every_depth() {
    let (w, h) = (5, 3);
    for fmt in FMTS {
        // channel c of pixel p holds a distinct fraction
        let img = image_of(w, h, fmt, |x, y, c| ((y * w + x) * 4 + c) as f64 / 64.0);
        for (chan, c) in [(1, 1), (2, 2), (3, 3), (4, 0)] {
            let m = extract(&img, chan);
            assert_eq!((m.width, m.height, m.data.len()), (w, h, w * h));
            for p in 0..w * h {
                let v = sample(&img, p * 4 + c) as f32;
                let want = v / fmt.white();
                assert_eq!(m.data[p].to_bits(), want.to_bits(), "{fmt:?} chan {chan} pixel {p}");
            }
        }
    }
}

#[test]
fn extract_hand_values_and_the_float_clamp() {
    // 8 bpc: 51 / 255 = 0.2 (f32), 255 is 1; 16 bpc: 32768 / 65535
    let mut img = Image::new(2, 1, Fmt::U8);
    if let Samples::U8(s) = &mut img.samples {
        s.copy_from_slice(&[255, 51, 0, 7, 128, 255, 1, 0]);
    }
    assert_eq!(extract(&img, 1).data, [51.0f32 / 255.0, 1.0]);
    assert_eq!(extract(&img, 4).data, [1.0, 128.0f32 / 255.0]);
    assert_eq!(extract(&img, 1).data[0], 0.2f32);
    let mut img = Image::new(1, 1, Fmt::U16);
    if let Samples::U16(s) = &mut img.samples {
        s.copy_from_slice(&[65535, 32768, 0, 1]);
    }
    assert_eq!(extract(&img, 1).data, [32768.0f32 / 65535.0]);
    assert_eq!(extract(&img, 2).data, [0.0]);
    assert_eq!(extract(&img, 3).data, [1.0f32 / 65535.0]);
    assert_eq!(extract(&img, 4).data, [1.0]);
    // float: outside 0..1 is clamped, NaN and -0.0 read 0.0, +inf 1.0
    let vals = [-0.5f32, 1.5, f32::NAN, -0.0, f32::INFINITY, 0.25, f32::NEG_INFINITY, 1.0];
    let mut img = Image::new(vals.len(), 1, Fmt::F32);
    if let Samples::F32(s) = &mut img.samples {
        for (p, &v) in vals.iter().enumerate() {
            s[p * 4 + 3] = v;
        }
    }
    let m = extract(&img, 3);
    assert_eq!(bits(&m.data), bits(&[0.0, 1.0, 0.0, 0.0, 1.0, 0.25, 0.0, 1.0]));
}

#[test]
fn an_empty_frame_gives_an_empty_matte() {
    for fmt in FMTS {
        let m = extract(&Image::new(0, 4, fmt), 1);
        assert_eq!((m.width, m.height, m.data.len()), (0, 4, 0));
    }
}

// ------------------------------------------------------------------------------------------------
// choke
// ------------------------------------------------------------------------------------------------

#[test]
fn a_choke_of_zero_changes_nothing() {
    let mut rng = Rng::new(3);
    let mut m = random_matte(&mut rng, 13, 9);
    // values a clamp would change are kept by the identity too
    m.data[0] = -0.0;
    m.data[1] = 1.5;
    let before = bits(&m.data);
    for a in [0.0f32, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        choke(&mut m, a).unwrap();
        assert_eq!(bits(&m.data), before, "amount {a}");
    }
    // a shrink by 0 is the identity as well
    let mut d = m.data.clone();
    shrink(&mut d, 13, 9, 0.0);
    assert_eq!(bits(&d), before);
}

#[test]
fn the_threshold_of_one_pixel_is_one_minus_the_two_outer_taps() {
    // n = 1: r = 2, fr = 0.01, R = 0.99 + 2 = 2.99; three normalised taps
    let k = gaussian_kernel(2.99);
    assert_eq!(k.len(), 3);
    assert_eq!([k[0], k[1], k[2]], [0.472_128_87f32, 0.234_970_64, 0.028_964_948]);
    let t = 1.0f32 - (k[1] + k[2]);
    assert_eq!(t, 0.736_064_43f32);
    // the radius of every whole n is n + 1.99 in f32
    for (n, want) in [(1u32, 2.99f32), (2, 3.99), (3, 4.99), (10, 11.99)] {
        let r = (n as f32 + 1.0).ceil();
        assert_eq!((1.0f32 - 0.01f32) + r, want, "n = {n}");
    }
}

#[test]
fn a_shrink_of_n_erodes_a_straight_edge_by_n_pixels() {
    // 24x16, inside for x < 12; columns 6 to 15 of the rows 5 to 10, which the vertical pass of
    // every kernel here reaches as interior rows (the border rows differ by a few ulps)
    let (w, h, edge) = (24, 16, 12);
    let cases: [(f32, [f32; 10]); 3] = [
        (1.0, [1.0, 1.0, 1.0, 1.0, 0.890_257_775_783_538_8, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (2.0, [1.0, 1.0, 1.0, 0.876_453_101_634_979_2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (3.0, [1.0, 1.0, 0.852_424_740_791_320_8, 1.713_282_244_963_920_7e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ];
    for (n, want) in cases {
        let mut m = edge_matte(w, h, edge);
        choke(&mut m, n).unwrap();
        for y in 5..=10 {
            assert_eq!(&m.data[y * w + 6..y * w + 16], &want, "n = {n}, row {y}");
        }
        // every row: the n pixels next to the edge are gone (up to a rounding residue), the next
        // one is k[n] / (k[n] + k[n + 1]), the rest keeps its side
        let ni = n as usize;
        let k = gaussian_kernel(n + 1.99);
        let partial = k[ni] / (k[ni] + k[ni + 1]);
        for y in 0..h {
            let row = &m.data[y * w..(y + 1) * w];
            for x in edge - ni..edge {
                assert!(row[x] < 1e-5, "n = {n}, row {y}, x = {x}: {}", row[x]);
            }
            assert!((row[edge - ni - 1] - partial).abs() < 1e-5, "n = {n}, row {y}: {} against {partial}", row[edge - ni - 1]);
            assert!(row[..edge - ni - 1].iter().all(|&v| v == 1.0), "n = {n}, row {y}");
            assert!(row[edge..].iter().all(|&v| v == 0.0), "n = {n}, row {y}");
        }
    }
}

#[test]
fn a_negative_amount_grows_the_matte_through_its_inverse() {
    let (w, h) = (24, 16);
    let mut grown = edge_matte(w, h, 12);
    choke(&mut grown, -1.0).unwrap();
    let row = &grown.data[8 * w..9 * w];
    // the first outside pixel is now inside, the next partial (1 minus the shrink's partial)
    assert!(row[..13].iter().all(|&v| v == 1.0));
    assert_eq!(row[13], 0.109_742_224_216_461_18);
    assert_eq!(row[13], 1.0f32 - 0.890_257_775_783_538_8);
    assert!(row[14..].iter().all(|&v| v == 0.0));
    // bit for bit: grow(m, a) = 1 - shrink(1 - m, a), clamped
    let mut rng = Rng::new(9);
    let m = random_matte(&mut rng, 17, 11);
    for a in [0.5f32, 1.0, 2.0, 3.5] {
        let mut g = m.clone();
        choke(&mut g, -a).unwrap();
        let mut inv = m.clone();
        inv.data.iter_mut().for_each(|v| *v = 1.0 - *v);
        choke(&mut inv, a).unwrap();
        let want: Vec<f32> = inv.data.iter().map(|&v| clamp01(1.0 - v)).collect();
        assert_eq!(bits(&g.data), bits(&want), "amount -{a}");
    }
}

#[test]
fn a_fractional_amount_mixes_the_floor_and_the_ceiling() {
    let mut rng = Rng::new(11);
    let m = random_matte(&mut rng, 19, 13);
    for (a, lo, hi) in [(1.5f32, 1.0f32, 2.0f32), (0.25, 0.0, 1.0), (3.75, 3.0, 4.0), (2.125, 2.0, 3.0)] {
        let mut got = m.clone();
        choke(&mut got, a).unwrap();
        let (mut l, mut c) = (m.data.clone(), m.data.clone());
        shrink(&mut l, 19, 13, lo);
        shrink(&mut c, 19, 13, hi);
        let t = a - lo;
        let want: Vec<f32> = l.iter().zip(&c).map(|(&x, &y)| clamp01(x * (1.0 - t) + y * t)).collect();
        assert_eq!(bits(&got.data), bits(&want), "amount {a}");
    }
    // half a pixel on the edge: half way between the matte and its one pixel shrink
    let (w, h) = (24, 16);
    let mut half = edge_matte(w, h, 12);
    choke(&mut half, 0.5).unwrap();
    let row = &half.data[8 * w..9 * w];
    assert_eq!(&row[8..14], &[1.0, 1.0, 0.945_128_917_694_091_8, 0.5, 0.0, 0.0]);
    assert_eq!(row[10], 1.0f32 * 0.5 + 0.890_257_775_783_538_8 * 0.5);
}

#[test]
fn constant_mattes_survive_a_shrink() {
    for n in [1.0f32, 2.0, 4.0, 2.5] {
        let mut zeros = Matte::filled(9, 7, 0.0);
        choke(&mut zeros, n).unwrap();
        assert!(zeros.data.iter().all(|&v| v.to_bits() == 0), "zeros, n = {n}");
        let mut ones = Matte::filled(9, 7, 1.0);
        choke(&mut ones, n).unwrap();
        assert!(ones.data.iter().all(|&v| v > 1.0 - 1e-6 && v <= 1.0), "ones, n = {n}");
    }
}

#[test]
fn tiny_and_degenerate_mattes_shrink_without_panicking() {
    for (w, h) in [(1, 1), (1, 7), (7, 1), (2, 2), (0, 0), (0, 3)] {
        let mut m = Matte::filled(w, h, 1.0);
        choke(&mut m, 2.5).unwrap();
        choke(&mut m, -1.5).unwrap();
        assert_eq!(m.data.len(), w * h);
        assert!(m.data.iter().all(|v| (0.0..=1.0).contains(v)));
    }
    // a magnitude above CHOKE_MAX acts as CHOKE_MAX (no control reaches either)
    let mut a = Matte::filled(3, 2, 0.5);
    let mut b = a.clone();
    choke(&mut a, 1e30).unwrap();
    choke(&mut b, CHOKE_MAX).unwrap();
    assert_eq!(bits(&a.data), bits(&b.data));
}

// ------------------------------------------------------------------------------------------------
// effective and over weights
// ------------------------------------------------------------------------------------------------

#[test]
fn fg1_is_on_top_and_main_bg_gets_the_product_of_the_inverses() {
    // exact binary fractions: m0 = 0.5, m1 = 0.25, m2 = 0.75
    let ms = [Matte::filled(3, 2, 0.5), Matte::filled(3, 2, 0.25), Matte::filled(3, 2, 0.75)];
    let e = effective(&ms, &[0.0, 0.0, 0.0], 3, 2).unwrap();
    assert_eq!(e.len(), 4);
    let want = [0.5f32, 0.125, 0.281_25, 0.093_75];
    for (k, &v) in want.iter().enumerate() {
        assert!(e[k].data.iter().all(|&x| x == v), "layer {k}: {:?}", e[k].data);
        assert_eq!((e[k].width, e[k].height), (3, 2));
    }
    assert_eq!(want.iter().sum::<f32>(), 1.0);
    // FG1 over FG2 where both are opaque: FG2 sees nothing there, Main_BG neither
    let a = edge_matte(8, 2, 5);
    let b = edge_matte(8, 2, 3);
    let e = effective(&[a.clone(), b], &[0.0, 0.0], 8, 2).unwrap();
    assert_eq!(&e[0].data[..8], &a.data[..8]);
    assert_eq!(&e[1].data[..8], &[0.0f32; 8]);
    assert_eq!(&e[2].data[..8], &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
}

#[test]
fn one_fg_layer_leaves_main_bg_its_inverse() {
    let mut rng = Rng::new(5);
    let m = random_matte(&mut rng, 11, 7);
    let e = effective(std::slice::from_ref(&m), &[0.0], 11, 7).unwrap();
    assert_eq!(e.len(), 2);
    assert_eq!(bits(&e[0].data), bits(&m.data));
    let inv: Vec<f32> = m.data.iter().map(|&v| 1.0 - v).collect();
    assert_eq!(bits(&e[1].data), bits(&inv));
}

#[test]
fn the_shrink_chokes_only_the_inverse_matte() {
    let mut rng = Rng::new(6);
    let (m0, m1) = (random_matte(&mut rng, 15, 9), random_matte(&mut rng, 15, 9));
    let e = effective(&[m0.clone(), m1.clone()], &[2.0, 1.5], 15, 9).unwrap();
    // each layer tracks with its raw matte times the (choked) visibility from above
    assert_eq!(bits(&e[0].data), bits(&m0.data));
    let mut inv0 = Matte { width: 15, height: 9, data: m0.data.iter().map(|&v| 1.0 - v).collect() };
    choke(&mut inv0, 2.0).unwrap();
    let vis1: Vec<f32> = inv0.data.iter().map(|&i| clamp01(i * 1.0)).collect();
    let e1: Vec<f32> = m1.data.iter().zip(&vis1).map(|(&m, &v)| m * v).collect();
    assert_eq!(bits(&e[1].data), bits(&e1));
    let mut inv1 = Matte { width: 15, height: 9, data: m1.data.iter().map(|&v| 1.0 - v).collect() };
    choke(&mut inv1, 1.5).unwrap();
    let main: Vec<f32> = inv1.data.iter().zip(&vis1).map(|(&i, &v)| clamp01(i * v)).collect();
    assert_eq!(bits(&e[2].data), bits(&main));
    // the shrink takes area from the layers below: Main_BG of an edge loses n pixels
    let e = effective(&[edge_matte(24, 16, 12)], &[1.0], 24, 16).unwrap();
    let row = &e[1].data[8 * 24..9 * 24];
    assert!(row[..13].iter().all(|&v| v == 0.0));
    assert_eq!(row[13], 0.890_257_775_783_538_8);
    assert!(row[14..].iter().all(|&v| v == 1.0));
    assert_eq!(&e[0].data, &edge_matte(24, 16, 12).data);
}

#[test]
fn without_fg_mattes_main_bg_is_the_constant_one_of_the_layer_size() {
    let e = effective(&[], &[], 64, 48).unwrap();
    assert_eq!(e, vec![Matte::filled(64, 48, 1.0)]);
    // so a premultiply by it is the identity (it was a 0x0 matte, which the premultiply refused)
    let mut rng = Rng::new(21);
    for fmt in FMTS {
        let img = random_image(&mut rng, 64, 48, fmt);
        assert!(same_image(&premultiply(&img, &e[0]).unwrap(), &img), "{fmt:?}");
    }
    assert_eq!(effective(&[], &[], 0, 0).unwrap(), vec![Matte::filled(0, 0, 1.0)]);
}

fn is_mismatch<T: std::fmt::Debug>(r: Result<T>) -> bool {
    matches!(r, Err(FxError::Mismatch(_)))
}

#[test]
fn effective_refuses_mattes_of_another_size_and_a_wrong_shrink_count() {
    let shrinks = [0.0, 0.0];
    assert!(is_mismatch(effective(&[Matte::filled(3, 2, 1.0), Matte::filled(2, 3, 1.0)], &shrinks, 3, 2)));
    assert!(is_mismatch(effective(&[Matte::filled(3, 2, 1.0), Matte::filled(3, 2, 1.0)], &shrinks, 4, 2)));
    assert!(is_mismatch(effective(&[Matte::filled(4, 4, 1.0)], &[], 4, 4)));
    assert!(is_mismatch(effective(&[Matte::filled(4, 4, 1.0)], &[0.0, 1.0], 4, 4)));
    assert!(is_mismatch(effective(&[Matte { width: 4, height: 4, data: vec![0.5; 15] }], &[0.0], 4, 4)));
    assert!(is_mismatch(effective(&[Matte { width: 4, height: 4, data: vec![0.5; 15] }], &[1.0], 4, 4)));
    // the working size of a solve against mattes of the output size
    assert!(is_mismatch(effective(&[Matte::filled(64, 48, 0.5)], &[0.0], 32, 24)));
}

#[test]
fn the_over_weights_of_every_layer_are_the_unshrunk_tracking_mattes() {
    let mut rng = Rng::new(8);
    for nfg in 1..=3usize {
        let ms: Vec<Matte> = (0..nfg).map(|_| random_matte(&mut rng, 10, 6)).collect();
        let e = effective(&ms, &vec![0.0; nfg], 10, 6).unwrap();
        let wts = over_weights(&ms, 0, nfg, 10, 6).unwrap();
        assert_eq!(wts.len(), nfg + 1);
        for k in 0..=nfg {
            assert_eq!(bits(&wts[k].data), bits(&e[k].data), "nfg {nfg} layer {k}");
        }
    }
}

#[test]
fn a_group_is_weighted_by_itself_alone() {
    let ms = [Matte::filled(2, 2, 0.5), Matte::filled(2, 2, 0.25), Matte::filled(2, 2, 0.75)];
    // single layers: the raw matte, Main_BG the constant 1
    for k in 0..3 {
        let w = over_weights(&ms, k, k, 2, 2).unwrap();
        assert_eq!(w.len(), 1);
        assert_eq!(bits(&w[0].data), bits(&ms[k].data));
    }
    assert_eq!(over_weights(&ms, 3, 3, 2, 2).unwrap(), vec![Matte::filled(2, 2, 1.0)]);
    // FG2 and FG3: FG2 is not occluded by FG1 outside the group
    let w = over_weights(&ms, 1, 2, 2, 2).unwrap();
    assert_eq!(w.len(), 2);
    assert!(w[0].data.iter().all(|&v| v == 0.25));
    assert!(w[1].data.iter().all(|&v| v == 0.75 * 0.75));
    // FG3 and Main_BG
    let w = over_weights(&ms, 2, 3, 2, 2).unwrap();
    assert!(w[0].data.iter().all(|&v| v == 0.75) && w[1].data.iter().all(|&v| v == 0.25));
    // no FG layer at all: one constant matte of the given size
    assert_eq!(over_weights(&[], 0, 0, 4, 3).unwrap(), vec![Matte::filled(4, 3, 1.0)]);
}

#[test]
fn over_weights_refuse_a_group_that_does_not_exist_and_mattes_of_another_size() {
    let one = [Matte::filled(4, 4, 0.5)];
    assert!(is_mismatch(over_weights(&one, 1, 2, 4, 4)));
    assert!(is_mismatch(over_weights(&one, 1, 0, 4, 4)));
    assert!(is_mismatch(over_weights(&one, 0, 2, 4, 4)));
    // weights at the working size against mattes of the output size
    assert!(is_mismatch(over_weights(&[Matte::filled(64, 48, 0.5)], 0, 1, 32, 24)));
    assert!(is_mismatch(over_weights(&[Matte { width: 4, height: 4, data: vec![0.5; 15] }], 0, 1, 4, 4)));
    assert_eq!(over_weights(&one, 1, 1, 4, 4).unwrap(), vec![Matte::filled(4, 4, 1.0)]);
}

// ------------------------------------------------------------------------------------------------
// premultiply, OVER, holes
// ------------------------------------------------------------------------------------------------

#[test]
fn a_premultiply_by_ones_is_the_identity_at_every_depth() {
    let mut rng = Rng::new(12);
    for fmt in FMTS {
        let img = random_image(&mut rng, 9, 5, fmt);
        let out = premultiply(&img, &Matte::filled(9, 5, 1.0)).unwrap();
        assert!(same_image(&out, &img), "{fmt:?}");
        let clear = premultiply(&img, &Matte::filled(9, 5, 0.0)).unwrap();
        assert!(same_image(&clear, &Image::new(9, 5, fmt)), "{fmt:?}");
    }
}

#[test]
fn the_premultiply_rounds_every_channel_like_the_engine() {
    let m = Matte { width: 2, height: 1, data: vec![0.5, 0.2] };
    let mut img = Image::new(2, 1, Fmt::U8);
    if let Samples::U8(s) = &mut img.samples {
        s.copy_from_slice(&[255, 1, 3, 200, 255, 254, 5, 100]);
    }
    // q(255 * 0.5) = trunc(127.5 + 0.5) = 128, q(1 * 0.5) = 1, q(3 * 0.5) = 2; alpha too
    let out = premultiply(&img, &m).unwrap();
    assert_eq!(out.samples, Samples::U8(vec![128, 1, 2, 100, 51, 51, 1, 20]));
    let mut img = Image::new(1, 1, Fmt::U16);
    if let Samples::U16(s) = &mut img.samples {
        s.copy_from_slice(&[65535, 1, 3, 40000]);
    }
    let out = premultiply(&img, &Matte { width: 1, height: 1, data: vec![0.5] }).unwrap();
    assert_eq!(out.samples, Samples::U16(vec![32768, 1, 2, 20000]));
    let mut img = Image::new(1, 1, Fmt::F32);
    if let Samples::F32(s) = &mut img.samples {
        s.copy_from_slice(&[1.0, 0.8, 2.5, -0.25]);
    }
    let out = premultiply(&img, &Matte { width: 1, height: 1, data: vec![0.5] }).unwrap();
    assert_eq!(out.samples, Samples::F32(vec![0.5, 0.8f32 * 0.5, 1.25, -0.125]));
}

#[test]
fn a_premultiply_needs_a_matte_of_the_image_size() {
    assert!(is_mismatch(premultiply(&Image::new(3, 2, Fmt::U8), &Matte::filled(2, 3, 1.0))));
    assert!(is_mismatch(premultiply(&Image::new(64, 48, Fmt::U16), &Matte::filled(32, 24, 1.0))));
    assert!(is_mismatch(premultiply(&Image::new(2, 2, Fmt::F32), &Matte { width: 2, height: 2, data: vec![1.0; 3] })));
    assert!(is_mismatch(premultiply(&Image::new(64, 48, Fmt::U8), &Matte::filled(0, 0, 1.0))));
}

#[test]
fn a_choke_refuses_a_matte_whose_data_does_not_fit_its_size() {
    for amount in [0.0f32, 1.0, -2.5, f32::NAN] {
        let mut m = Matte { width: 4, height: 4, data: vec![0.5; 15] };
        assert!(is_mismatch(choke(&mut m, amount)), "amount {amount}");
        assert_eq!(m.data, vec![0.5; 15], "left as it is");
        let mut m = Matte { width: 0, height: 4, data: vec![0.5; 4] };
        assert!(is_mismatch(choke(&mut m, amount)), "amount {amount}");
    }
    let mut empty = Matte::filled(0, 4, 0.5);
    choke(&mut empty, 1.0).unwrap();
    assert!(empty.data.is_empty());
}

#[test]
fn out_of_range_mattes_are_clamped_and_in_range_ones_are_used_as_they_are() {
    assert_eq!(clamp_matte(1.5), 1.0);
    assert_eq!(clamp_matte(-0.5), 0.0);
    assert_eq!(clamp_matte(f32::NAN), 0.0);
    assert_eq!(clamp_matte(f32::INFINITY), 1.0);
    assert_eq!(clamp_matte(-0.0).to_bits(), (-0.0f32).to_bits(), "an in range value passes bit for bit");
    assert_eq!(clamp_matte(0.3), 0.3);
    // premultiply: 1.5 of 200 was q(300) = 300 & 0xff = 44; now the clamped 1.0 keeps 200
    let mut im = Image::new(3, 1, Fmt::U8);
    if let Samples::U8(s) = &mut im.samples {
        s.iter_mut().for_each(|v| *v = 200);
    }
    let m = Matte { width: 3, height: 1, data: vec![1.5, -0.5, f32::NAN] };
    let mut want = vec![200u8; 4];
    want.extend([0u8; 8]);
    assert_eq!(premultiply(&im, &m).unwrap().samples, Samples::U8(want));
    // each depth: an out of range matte gives the premultiply of its clamp
    let mut rng = Rng::new(22);
    for fmt in FMTS {
        let img = random_image(&mut rng, 7, 5, fmt);
        let raw = Matte { width: 7, height: 5, data: (0..35).map(|i| (i as f32 - 10.0) * 0.07).collect() };
        let clamped = Matte { width: 7, height: 5, data: raw.data.iter().map(|&v| clamp_matte(v)).collect() };
        assert!(same_image(&premultiply(&img, &raw).unwrap(), &premultiply(&img, &clamped).unwrap()), "{fmt:?}");
    }
    // effective and the over weights read the clamped mattes
    let raw = [Matte { width: 2, height: 1, data: vec![1.5, -0.5] }, Matte { width: 2, height: 1, data: vec![f32::NAN, 0.25] }];
    let clamped = [Matte { width: 2, height: 1, data: vec![1.0, 0.0] }, Matte { width: 2, height: 1, data: vec![0.0, 0.25] }];
    for shrinks in [[0.0f32, 0.0], [1.0, 0.0]] {
        assert_eq!(effective(&raw, &shrinks, 2, 1).unwrap(), effective(&clamped, &shrinks, 2, 1).unwrap(), "{shrinks:?}");
    }
    let e = effective(&raw, &[0.0, 0.0], 2, 1).unwrap();
    assert_eq!(e.iter().map(|m| m.data.clone()).collect::<Vec<_>>(), vec![vec![1.0, 0.0], vec![0.0, 0.25], vec![0.0, 0.75]]);
    assert_eq!(over_weights(&raw, 0, 2, 2, 1).unwrap(), over_weights(&clamped, 0, 2, 2, 1).unwrap());
    // an in range matte holding -0.0 keeps it (the float premultiply of -0.0 is -0.0)
    let mut im = Image::new(1, 1, Fmt::F32);
    if let Samples::F32(s) = &mut im.samples {
        s.copy_from_slice(&[1.0, 0.5, 0.25, 2.0]);
    }
    let p = premultiply(&im, &Matte { width: 1, height: 1, data: vec![-0.0] }).unwrap();
    let Samples::F32(ps) = &p.samples else { unreachable!() };
    assert!(ps.iter().all(|v| v.to_bits() == (-0.0f32).to_bits()), "{ps:?}");
}

#[test]
fn over_onto_a_transparent_image_gives_the_source_at_every_depth() {
    let mut rng = Rng::new(13);
    for fmt in FMTS {
        let src = random_image(&mut rng, 11, 7, fmt);
        let mut dest = Image::new(11, 7, fmt);
        over(&mut dest, &src).unwrap();
        assert!(same_image(&dest, &src), "{fmt:?}");
    }
}

#[test]
fn over_of_a_transparent_image_changes_nothing_at_every_depth() {
    let mut rng = Rng::new(14);
    for fmt in FMTS {
        let dest0 = random_image(&mut rng, 11, 7, fmt);
        let mut dest = dest0.clone();
        over(&mut dest, &Image::new(11, 7, fmt)).unwrap();
        assert!(same_image(&dest, &dest0), "{fmt:?}");
        // and an opaque source replaces the destination
        let src = image_of(11, 7, fmt, |x, y, c| if c == 0 { 1.0 } else { (x + y + c) as f64 / 32.0 });
        over(&mut dest, &src).unwrap();
        assert!(same_image(&dest, &src), "{fmt:?}");
    }
}

#[test]
fn over_hand_values() {
    // u8: a half transparent red over opaque green
    let mut src = Image::new(1, 1, Fmt::U8);
    let mut dest = Image::new(1, 1, Fmt::U8);
    if let (Samples::U8(s), Samples::U8(d)) = (&mut src.samples, &mut dest.samples) {
        s.copy_from_slice(&[128, 64, 0, 0]);
        d.copy_from_slice(&[255, 0, 200, 0]);
    }
    over(&mut dest, &src).unwrap();
    // ia = 1 - 128/255; g = 200 * ia = 99.6078 -> 100; alpha (255 - 255) * a + 255 = 255
    assert_eq!(dest.samples, Samples::U8(vec![255, 64, 100, 0]));
    // float: alpha 0.25 over alpha 0.5: alpha 0.5 + 0.5 * 0.25 = 0.625, rgb b * 0.75 + a
    let mut src = Image::new(1, 1, Fmt::F32);
    let mut dest = Image::new(1, 1, Fmt::F32);
    if let (Samples::F32(s), Samples::F32(d)) = (&mut src.samples, &mut dest.samples) {
        s.copy_from_slice(&[0.25, 0.125, 0.0, 0.25]);
        d.copy_from_slice(&[0.5, 0.5, 0.25, 0.0]);
    }
    over(&mut dest, &src).unwrap();
    assert_eq!(dest.samples, Samples::F32(vec![0.625, 0.5, 0.1875, 0.25]));
}

#[test]
fn over_matches_an_f64_reference_for_every_pair_of_u8_alphas() {
    // pixel (sa, da) of a 256x256 image: source alpha sa over destination alpha da, with colours
    // 0, about a half or a third, and the alpha itself (valid premultiplied pixels)
    let n = 256usize;
    let parts = [0.0f64, 0.5, 1.0];
    let mut checked = 0usize;
    for &sp in &parts {
        for &dp in &[0.0f64, 1.0 / 3.0, 1.0] {
            let mut src = Image::new(n, n, Fmt::U8);
            let mut dest = Image::new(n, n, Fmt::U8);
            if let (Samples::U8(s), Samples::U8(d)) = (&mut src.samples, &mut dest.samples) {
                for da in 0..n {
                    for sa in 0..n {
                        let i = (da * n + sa) * 4;
                        let sc = (sa as f64 * sp).floor() as u8;
                        let dc = (da as f64 * dp).floor() as u8;
                        s[i..i + 4].copy_from_slice(&[sa as u8, sc, sc / 2, 0]);
                        d[i..i + 4].copy_from_slice(&[da as u8, dc, 0, dc / 3]);
                    }
                }
            }
            let before = dest.clone();
            over(&mut dest, &src).unwrap();
            let (Samples::U8(s), Samples::U8(b), Samples::U8(o)) = (&src.samples, &before.samples, &dest.samples) else {
                unreachable!()
            };
            for p in 0..n * n {
                let i = p * 4;
                let a = f64::from(s[i]) / 255.0;
                let want = [
                    f64::from(b[i]) + (255.0 - f64::from(b[i])) * a,
                    f64::from(b[i + 1]) * (1.0 - a) + f64::from(s[i + 1]),
                    f64::from(b[i + 2]) * (1.0 - a) + f64::from(s[i + 2]),
                    f64::from(b[i + 3]) * (1.0 - a) + f64::from(s[i + 3]),
                ];
                for c in 0..4 {
                    let got = f64::from(o[i + c]);
                    let v = want[c];
                    assert!((got - v).abs() <= 0.5 + 1e-4, "src {:?} dest {:?}: channel {c} {got} against {v}", &s[i..i + 4], &b[i..i + 4]);
                    // away from a rounding tie the result is the nearest integer exactly
                    let frac = v - v.floor();
                    if (frac - 0.5).abs() > 1e-3 {
                        assert_eq!(got, (v + 0.5).floor(), "src {:?} dest {:?}: channel {c}", &s[i..i + 4], &b[i..i + 4]);
                    }
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 9 * 256 * 256 * 4);
}

#[test]
fn over_matches_an_f64_reference_at_16_bpc_and_in_float() {
    let mut rng = Rng::new(15);
    for fmt in [Fmt::U16, Fmt::F32] {
        let src = random_image(&mut rng, 64, 32, fmt);
        let dest0 = random_image(&mut rng, 64, 32, fmt);
        let mut dest = dest0.clone();
        over(&mut dest, &src).unwrap();
        let white = f64::from(fmt.white());
        for p in 0..64 * 32 {
            let i = p * 4;
            let a = sample(&src, i) / white;
            let want = [
                sample(&dest0, i) + (white - sample(&dest0, i)) * a,
                sample(&dest0, i + 1) * (1.0 - a) + sample(&src, i + 1),
                sample(&dest0, i + 2) * (1.0 - a) + sample(&src, i + 2),
                sample(&dest0, i + 3) * (1.0 - a) + sample(&src, i + 3),
            ];
            for c in 0..4 {
                let got = sample(&dest, i + c);
                let tol = if fmt == Fmt::U16 { 0.5 + 0.02 } else { 1e-6 };
                assert!((got - want[c]).abs() <= tol, "{fmt:?} pixel {p} channel {c}: {got} against {}", want[c]);
            }
        }
    }
}

#[test]
fn over_refuses_images_of_another_shape() {
    let mut dest = Image::new(3, 2, Fmt::U8);
    assert!(matches!(over(&mut dest, &Image::new(2, 3, Fmt::U8)), Err(FxError::Mismatch(_))));
    assert!(matches!(over(&mut dest, &Image::new(3, 2, Fmt::U16)), Err(FxError::Mismatch(_))));
    assert!(over(&mut Image::new(0, 0, Fmt::F32), &Image::new(0, 0, Fmt::F32)).is_ok());
}

#[test]
fn the_holes_of_a_composite_are_white_minus_alpha() {
    let mut img = Image::new(3, 1, Fmt::U8);
    if let Samples::U8(s) = &mut img.samples {
        s.copy_from_slice(&[0, 9, 9, 9, 255, 1, 2, 3, 100, 0, 0, 0]);
    }
    assert_eq!(alpha_holes(&img), HoleMask::U8(vec![255, 0, 155]));
    let mut img = Image::new(2, 1, Fmt::U16);
    if let Samples::U16(s) = &mut img.samples {
        s.copy_from_slice(&[65535, 0, 0, 0, 1, 0, 0, 0]);
    }
    assert_eq!(alpha_holes(&img), HoleMask::U16(vec![0, 65534]));
    let mut img = Image::new(2, 1, Fmt::F32);
    if let Samples::F32(s) = &mut img.samples {
        s.copy_from_slice(&[0.25, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    }
    assert_eq!(alpha_holes(&img), HoleMask::F32(vec![0.75, 0.0]));
}

#[test]
fn the_layer_primitives_do_not_depend_on_the_thread_count() {
    let mut rng = Rng::new(16);
    let (w, h) = (37, 29);
    let ms: Vec<Matte> = (0..3).map(|_| random_matte(&mut rng, w, h)).collect();
    let run = |threads: usize| {
        in_pool(threads, || {
            let mut c = ms[0].clone();
            choke(&mut c, 2.5).unwrap();
            let mut g = ms[1].clone();
            choke(&mut g, -1.25).unwrap();
            let e = effective(&ms, &[1.5, 0.0, 3.0], w, h).unwrap();
            let wts = over_weights(&ms, 0, 3, w, h).unwrap();
            (c, g, e, wts)
        })
    };
    let (a, b) = (run(1), run(4));
    assert_eq!(bits(&a.0.data), bits(&b.0.data));
    assert_eq!(bits(&a.1.data), bits(&b.1.data));
    for k in 0..4 {
        assert_eq!(bits(&a.2[k].data), bits(&b.2[k].data), "effective {k}");
        assert_eq!(bits(&a.3[k].data), bits(&b.3[k].data), "over weight {k}");
    }
    for fmt in FMTS {
        let src = random_image(&mut rng, w, h, fmt);
        let dest0 = random_image(&mut rng, w, h, fmt);
        let run = |threads: usize| {
            in_pool(threads, || {
                let p = premultiply(&src, &ms[2]).unwrap();
                let mut d = dest0.clone();
                over(&mut d, &p).unwrap();
                (extract(&src, 2), p, d)
            })
        };
        let (a, b) = (run(1), run(3));
        assert_eq!(bits(&a.0.data), bits(&b.0.data), "{fmt:?}");
        assert!(same_image(&a.1, &b.1) && same_image(&a.2, &b.2), "{fmt:?}");
    }
}

// ------------------------------------------------------------------------------------------------
// the numpy float32 model fixtures (tests/fixtures/gen_matte.py)
// ------------------------------------------------------------------------------------------------

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl Reader<'_> {
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn f32(&mut self) -> f32 {
        f32::from_bits(self.u32())
    }
    fn floats(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f32()).collect()
    }
    fn matte(&mut self, w: usize, h: usize) -> Matte {
        Matte { width: w, height: h, data: self.floats(w * h) }
    }
    fn image(&mut self, fmt: Fmt, w: usize, h: usize) -> Image {
        let n = w * h * CHANNELS;
        let samples = match fmt {
            Fmt::U8 => {
                let v = self.b[self.p..self.p + n].to_vec();
                self.p += n;
                Samples::U8(v)
            }
            Fmt::U16 => {
                let v = (0..n).map(|i| u16::from_le_bytes([self.b[self.p + 2 * i], self.b[self.p + 2 * i + 1]])).collect();
                self.p += 2 * n;
                Samples::U16(v)
            }
            Fmt::F32 => Samples::F32(self.floats(n)),
        };
        Image { width: w, height: h, samples }
    }
    fn pairs(&mut self) -> Vec<([f32; 2], [f32; 2])> {
        let n = self.u32() as usize;
        (0..n).map(|_| ([self.f32(), self.f32()], [self.f32(), self.f32()])).collect()
    }
}

#[test]
fn model_fixtures_are_reproduced_bit_for_bit() {
    use crate::flow::{FlowGeometry, FlowSettings};
    use crate::{geom, points};
    let blob = crate::fixture::hex(include_str!("../../tests/fixtures/matte_cases.hex"));
    assert_eq!(&blob[..4], b"MAT1");
    let mut r = Reader { b: &blob, p: 4 };
    let n = r.u32();
    let mut kinds = [0usize; 10];
    for case in 0..n {
        let (kind, depth) = (r.u32(), r.u32());
        let (w, h) = (r.u32() as usize, r.u32() as usize);
        let fmt = FMTS[depth as usize];
        kinds[kind as usize] += 1;
        let at = format!("case {case} (kind {kind}, {fmt:?} {w}x{h})");
        match kind {
            1 => {
                let chan = r.u32() as i32;
                let img = r.image(fmt, w, h);
                let want = r.matte(w, h);
                assert_eq!(bits(&extract(&img, chan).data), bits(&want.data), "{at} chan {chan}");
            }
            2 => {
                let amount = r.f32();
                let mut m = r.matte(w, h);
                let want = r.matte(w, h);
                choke(&mut m, amount).unwrap();
                assert_eq!(bits(&m.data), bits(&want.data), "{at} amount {amount}");
            }
            3 => {
                let nfg = r.u32() as usize;
                let shrinks = r.floats(nfg);
                let ms: Vec<Matte> = (0..nfg).map(|_| r.matte(w, h)).collect();
                let got = effective(&ms, &shrinks, w, h).unwrap();
                assert_eq!(got.len(), nfg + 1, "{at}");
                for (k, g) in got.iter().enumerate() {
                    assert_eq!(bits(&g.data), bits(&r.matte(w, h).data), "{at} layer {k} shrinks {shrinks:?}");
                }
            }
            4 => {
                let (nfg, first, last) = (r.u32() as usize, r.u32() as usize, r.u32() as usize);
                let ms: Vec<Matte> = (0..nfg).map(|_| r.matte(w, h)).collect();
                let got = over_weights(&ms, first, last, w, h).unwrap();
                assert_eq!(got.len(), last - first + 1, "{at}");
                for (k, g) in got.iter().enumerate() {
                    assert_eq!(bits(&g.data), bits(&r.matte(w, h).data), "{at} group {first}..={last} entry {k}");
                }
            }
            5 => {
                let img = r.image(fmt, w, h);
                let m = r.matte(w, h);
                let want = r.image(fmt, w, h);
                assert!(same_image(&premultiply(&img, &m).unwrap(), &want), "{at}");
            }
            6 => {
                let mut dest = r.image(fmt, w, h);
                let src = r.image(fmt, w, h);
                let want = r.image(fmt, w, h);
                over(&mut dest, &src).unwrap();
                assert!(same_image(&dest, &want), "{at}");
            }
            7 => {
                let (q, hiq) = (r.u32() as u8, r.u32() != 0);
                let nlev = r.u32() as usize;
                let sizes: Vec<(usize, usize)> = (0..nlev).map(|_| (r.u32() as usize, r.u32() as usize)).collect();
                let geo = FlowSettings::new(q, hiq, 1.0).geometry(w, h).unwrap();
                assert_eq!(geo.sizes, sizes, "{at}: the geometry of the generator");
                let pairs = r.pairs();
                let ud = points::user_displacement(&pairs, &geo, (w, h));
                assert_eq!(ud.levels.len(), nlev, "{at}");
                for (l, &(lw, lh)) in sizes.iter().enumerate() {
                    let (d, wt) = (r.floats(lw * lh * 2), r.floats(lw * lh));
                    assert_eq!(bits(&ud.levels[l].d), bits(&d), "{at} level {l}, {} pairs", pairs.len());
                    assert_eq!(bits(&ud.levels[l].w), bits(&wt), "{at} level {l}, {} pairs", pairs.len());
                }
            }
            8 => {
                let (q, hiq) = (r.u32() as u8, r.u32() != 0);
                let (w0, h0) = (r.u32() as usize, r.u32() as usize);
                let geo = FlowSettings::new(q, hiq, 1.0).geometry(w, h).unwrap();
                assert_eq!((geo.width, geo.height), (w0, h0), "{at}");
                let manual = FlowGeometry { reduction: 0, width: w0, height: h0, levels: 1, sizes: vec![(w0, h0)] };
                let pairs = r.pairs();
                let want = r.floats(w0 * h0 * 2);
                let got = points::second_tracker(&pairs, &geo, (w, h)).unwrap();
                assert_eq!((got.width, got.height), (w0, h0), "{at}");
                assert_eq!(bits(&got.data), bits(&want), "{at}, {} pairs", pairs.len());
                // only the working size of the geometry matters
                assert_eq!(bits(&points::second_tracker(&pairs, &manual, (w, h)).unwrap().data), bits(&want), "{at}");
            }
            9 => {
                let ratio = r.f32();
                let npts = r.u32() as usize;
                let pts: Vec<([f32; 2], [u8; 3])> = (0..npts)
                    .map(|_| {
                        let p = [r.f32(), r.f32()];
                        let c = [r.b[r.p], r.b[r.p + 1], r.b[r.p + 2]];
                        r.p += 4;
                        (p, c)
                    })
                    .collect();
                let mut img = r.image(fmt, w, h);
                let want = r.image(fmt, w, h);
                geom::draw_points(&mut img, &pts, ratio);
                assert!(same_image(&img, &want), "{at} ratio {ratio}");
            }
            _ => panic!("{at}: unknown kind"),
        }
    }
    assert_eq!(r.p, blob.len());
    assert_eq!(kinds, [0, 15, 42, 6, 16, 6, 6, 13, 13, 9]);
}
