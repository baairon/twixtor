//! Engine images: interleaved 4 channel buffers in channel order A, R, G, B.
//!
//! The data live in the engine domain: colour is PREMULTIPLIED by alpha, and 16 bpc uses the full
//! range 0..=65535 (the range 0..=32768 only exists at the boundary conversions). 32 bpc white is
//! 1.0. Rows are tightly packed, top row first.

use fx_core::{par_rows, Depth, Frame, Pixels};

pub const CHANNELS: usize = 4;
/// Channel indices inside a pixel.
pub const A: usize = 0;
pub const R: usize = 1;
pub const G: usize = 2;
pub const B: usize = 3;

/// Pixel format of an image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fmt {
    U8,
    U16,
    F32,
}

impl Fmt {
    pub fn from_depth(depth: Depth) -> Fmt {
        match depth {
            Depth::U8 => Fmt::U8,
            Depth::U16 => Fmt::U16,
            Depth::F32 => Fmt::F32,
        }
    }

    pub fn depth(self) -> Depth {
        match self {
            Fmt::U8 => Depth::U8,
            Fmt::U16 => Depth::U16,
            Fmt::F32 => Depth::F32,
        }
    }

    /// The format index: 0 = 8 bpc, 1 = 16 bpc, 2 = float.
    pub fn code(self) -> u8 {
        match self {
            Fmt::U8 => 0,
            Fmt::U16 => 1,
            Fmt::F32 => 2,
        }
    }

    /// White of the engine domain: 255, 65535, 1.0.
    pub fn white(self) -> f32 {
        match self {
            Fmt::U8 => 255.0,
            Fmt::U16 => 65535.0,
            Fmt::F32 => 1.0,
        }
    }
}

/// The samples of an image, `width * height * 4` values in A, R, G, B order.
#[derive(Clone, Debug, PartialEq)]
pub enum Samples {
    U8(Vec<u8>),
    U16(Vec<u16>),
    F32(Vec<f32>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub samples: Samples,
}

impl Image {
    /// A cleared (all zero) image.
    pub fn new(width: usize, height: usize, fmt: Fmt) -> Image {
        let n = width * height * CHANNELS;
        let samples = match fmt {
            Fmt::U8 => Samples::U8(vec![0; n]),
            Fmt::U16 => Samples::U16(vec![0; n]),
            Fmt::F32 => Samples::F32(vec![0.0; n]),
        };
        Image { width, height, samples }
    }

    pub fn fmt(&self) -> Fmt {
        match self.samples {
            Samples::U8(_) => Fmt::U8,
            Samples::U16(_) => Fmt::U16,
            Samples::F32(_) => Fmt::F32,
        }
    }

    /// The same image with its rows in reverse order (the last row first).
    pub fn rows_reversed(&self) -> Image {
        fn reverse<T: Copy>(out: &mut [T], src: &[T], row_len: usize) {
            for (o, s) in out.chunks_exact_mut(row_len).zip(src.chunks_exact(row_len).rev()) {
                o.copy_from_slice(s);
            }
        }
        let mut out = self.clone();
        let row_len = self.row_len();
        if row_len == 0 {
            return out;
        }
        match (&mut out.samples, &self.samples) {
            (Samples::U8(o), Samples::U8(s)) => reverse(o, s, row_len),
            (Samples::U16(o), Samples::U16(s)) => reverse(o, s, row_len),
            (Samples::F32(o), Samples::F32(s)) => reverse(o, s, row_len),
            _ => unreachable!("a clone has the format of its source"),
        }
        out
    }

    /// Samples per row.
    pub fn row_len(&self) -> usize {
        self.width * CHANNELS
    }

    pub fn sample_count(&self) -> usize {
        self.width * self.height * CHANNELS
    }

    /// Same size and format, the precondition of every two image primitive.
    pub fn same_shape(&self, other: &Image) -> bool {
        self.width == other.width && self.height == other.height && self.fmt() == other.fmt()
    }

    pub fn clear(&mut self) {
        match &mut self.samples {
            Samples::U8(v) => v.fill(0),
            Samples::U16(v) => v.fill(0),
            Samples::F32(v) => v.fill(0.0),
        }
    }

    /// Engine image from a wire frame read STRAIGHT: the channels as stored, without the premultiply of
    /// [`Image::from_frame`], in the engine order A, R, G, B; 16 bpc takes every channel from the wire range to
    /// 0..=32768 (`Frame::u16_to_q15`) and then to the engine range the way [`Image::from_frame`] scales the alpha. The
    /// FG matte frames of Vectors In are read this way (`host::EnvHost::with_straight_mattes`).
    pub fn from_frame_straight(frame: &Frame) -> Image {
        let (w, h) = (frame.width, frame.height);
        let mut img = Image::new(w, h, Fmt::from_depth(frame.depth()));
        if w == 0 || h == 0 {
            return img;
        }
        match (&frame.pixels, &mut img.samples) {
            (Pixels::U8(px), Samples::U8(out)) => par_rows(out, w * CHANNELS, |y, row| {
                for (x, o) in row.chunks_exact_mut(CHANNELS).enumerate() {
                    let [r, g, b, a] = px[y * w + x];
                    o.copy_from_slice(&[a, r, g, b]);
                }
            }),
            (Pixels::U16(px), Samples::U16(out)) => par_rows(out, w * CHANNELS, |y, row| {
                for (x, o) in row.chunks_exact_mut(CHANNELS).enumerate() {
                    let [r, g, b, a] = px[y * w + x].map(Frame::u16_to_q15);
                    let s = |v: u16| q(v as f32 * PREMUL_16_ALPHA) as u16;
                    o.copy_from_slice(&[s(a), s(r), s(g), s(b)]);
                }
            }),
            (Pixels::F32(px), Samples::F32(out)) => par_rows(out, w * CHANNELS, |y, row| {
                for (x, o) in row.chunks_exact_mut(CHANNELS).enumerate() {
                    let [r, g, b, a] = px[y * w + x];
                    o.copy_from_slice(&[a, r, g, b]);
                }
            }),
            _ => unreachable!("Image::new follows the depth of the frame"),
        }
        img
    }

    /// Engine image from a straight wire frame: wire range 0..=65535 to the range 0..=32768 (16
    /// bpc, `Frame::u16_to_q15`), channel reorder to A, R, G, B, then the engine's premultiply
    /// (`premultiply_u8`, `premultiply_u16`, `premultiply_f32`). 16 bpc ends in the engine range
    /// 0..=65535 (alpha 32768 becomes 65535). The conversion is lossy: pixels with alpha below
    /// full do not survive `from_frame` followed by `to_frame` bit exactly.
    pub fn from_frame(frame: &Frame) -> Image {
        let (w, h) = (frame.width, frame.height);
        let mut img = Image::new(w, h, Fmt::from_depth(frame.depth()));
        if w == 0 || h == 0 {
            return img;
        }
        match (&frame.pixels, &mut img.samples) {
            (Pixels::U8(px), Samples::U8(out)) => par_rows(out, w * CHANNELS, |y, row| {
                for (x, o) in row.chunks_exact_mut(CHANNELS).enumerate() {
                    let [r, g, b, a] = px[y * w + x];
                    o.copy_from_slice(&premultiply_u8([a, r, g, b]));
                }
            }),
            (Pixels::U16(px), Samples::U16(out)) => par_rows(out, w * CHANNELS, |y, row| {
                for (x, o) in row.chunks_exact_mut(CHANNELS).enumerate() {
                    let [r, g, b, a] = px[y * w + x].map(Frame::u16_to_q15);
                    o.copy_from_slice(&premultiply_u16([a, r, g, b]));
                }
            }),
            (Pixels::F32(px), Samples::F32(out)) => par_rows(out, w * CHANNELS, |y, row| {
                for (x, o) in row.chunks_exact_mut(CHANNELS).enumerate() {
                    let [r, g, b, a] = px[y * w + x];
                    o.copy_from_slice(&premultiply_f32([a, r, g, b]));
                }
            }),
            _ => unreachable!("Image::new follows the depth of the frame"),
        }
        img
    }

    /// Wire frame from an engine image: the engine's unpremultiply (`unpremultiply_u8`,
    /// `unpremultiply_u16`, `unpremultiply_f32`), for 16 bpc the result in the range 0..=32768
    /// that `Frame::q15_to_u16` maps to the wire range, then the channel order R, G, B, A. Pixels
    /// with alpha 0 are left as they are in the engine image (8 and 16 bpc), float pixels with
    /// alpha below 1e-12 become all zero.
    pub fn to_frame(&self) -> Frame {
        let (w, h) = (self.width, self.height);
        let mut frame = Frame::new(w, h, self.fmt().depth());
        if w == 0 || h == 0 {
            return frame;
        }
        match (&self.samples, &mut frame.pixels) {
            (Samples::U8(src), Pixels::U8(out)) => par_rows(out, w, |y, row| {
                for (x, o) in row.iter_mut().enumerate() {
                    let i = (y * w + x) * CHANNELS;
                    let [a, r, g, b] = unpremultiply_u8([src[i], src[i + 1], src[i + 2], src[i + 3]]);
                    *o = [r, g, b, a];
                }
            }),
            (Samples::U16(src), Pixels::U16(out)) => par_rows(out, w, |y, row| {
                for (x, o) in row.iter_mut().enumerate() {
                    let i = (y * w + x) * CHANNELS;
                    let [a, r, g, b] =
                        unpremultiply_u16([src[i], src[i + 1], src[i + 2], src[i + 3]]).map(Frame::q15_to_u16);
                    *o = [r, g, b, a];
                }
            }),
            (Samples::F32(src), Pixels::F32(out)) => par_rows(out, w, |y, row| {
                for (x, o) in row.iter_mut().enumerate() {
                    let i = (y * w + x) * CHANNELS;
                    let [a, r, g, b] = unpremultiply_f32([src[i], src[i + 1], src[i + 2], src[i + 3]]);
                    *o = [r, g, b, a];
                }
            }),
            _ => unreachable!("Frame::new follows the format of the image"),
        }
        frame
    }
}

/// Truncation toward zero; out of range or NaN gives `i64::MIN`.
fn cvtt_i64(x: f64) -> i64 {
    if x >= -9.223372036854775808e18 && x < 9.223372036854775808e18 {
        x as i64
    } else {
        i64::MIN
    }
}

/// The rounding `q(x) = trunc((x as f64) + 0.5)`; callers keep the low byte or word of the
/// result.
pub(crate) fn q(x: f32) -> i64 {
    cvtt_i64(x as f64 + 0.5)
}

/// The f32 nearest to 1/255.
const INV_255: f32 = f32::from_bits(0x3b80_8081);
/// About 65535 / 2^30: the 16 bpc colour scale after dividing by 32768 and rescaling to 65535.
const PREMUL_16_COLOR: f32 = f32::from_bits(0x387f_ff00);
/// About 65535 / 32768: the 16 bpc alpha rescale.
const PREMUL_16_ALPHA: f32 = f32::from_bits(0x3fff_ff00);
/// About 32768 / 65535: back to the alpha range 0..=32768.
const UNPREMUL_16_ALPHA: f32 = f32::from_bits(0x3f00_0080);
/// Below this alpha a float pixel is cleared by the unpremultiply.
const UNPREMUL_F32_MIN_ALPHA: f64 = 1e-12;

/// One pixel `[A, R, G, B]`: `a = A * (1/255 as f32)`, colour `q(c * a)` (low byte), alpha
/// unchanged.
pub(crate) fn premultiply_u8(p: [u8; 4]) -> [u8; 4] {
    let a = p[0] as f32 * INV_255;
    [p[0], q(p[1] as f32 * a) as u8, q(p[2] as f32 * a) as u8, q(p[3] as f32 * a) as u8]
}

/// One pixel `[A, R, G, B]` in the range 0..=32768 to the engine range 0..=65535.
pub(crate) fn premultiply_u16(p: [u16; 4]) -> [u16; 4] {
    let a = p[0] as f32 * PREMUL_16_COLOR;
    [
        q(p[0] as f32 * PREMUL_16_ALPHA) as u16,
        q(p[1] as f32 * a) as u16,
        q(p[2] as f32 * a) as u16,
        q(p[3] as f32 * a) as u16,
    ]
}

/// One pixel `[A, R, G, B]`: colour times alpha in f32.
pub(crate) fn premultiply_f32(p: [f32; 4]) -> [f32; 4] {
    let r = [p[0], p[0] * p[1], p[0] * p[2], p[0] * p[3]];
    // a NaN in the result: the products with the NaN rule (`crate::nanrule`); numbers are the same bits
    if crate::nanrule::any_nan(&r) {
        use crate::nanrule::mul;
        [p[0], mul(p[0], p[1]), mul(p[0], p[2]), mul(p[0], p[3])]
    } else {
        r
    }
}

/// One pixel `[A, R, G, B]`: `s = 255 / A`, colour `q(min(c * s, 255))` (low byte); a pixel with
/// alpha 0 is left untouched.
pub(crate) fn unpremultiply_u8(p: [u8; 4]) -> [u8; 4] {
    if p[0] == 0 {
        return p;
    }
    let s = 255.0f32 / p[0] as f32;
    let f = |c: u8| {
        let mut v = c as f32 * s;
        if v > 255.0 {
            v = 255.0;
        }
        q(v) as u8
    };
    [p[0], f(p[1]), f(p[2]), f(p[3])]
}

/// One pixel `[A, R, G, B]` from the engine range back to the range 0..=32768: `s = 32768 / A`,
/// alpha `q(A * (32768/65535 as f32))`, colour `q(c * s)`, no clamp, low 16 bits stored; a pixel
/// with alpha 0 is left untouched.
pub(crate) fn unpremultiply_u16(p: [u16; 4]) -> [u16; 4] {
    if p[0] == 0 {
        return p;
    }
    let a = p[0] as f32;
    let s = 32768.0f32 / a;
    [q(a * UNPREMUL_16_ALPHA) as u16, q(p[1] as f32 * s) as u16, q(p[2] as f32 * s) as u16, q(p[3] as f32 * s) as u16]
}

/// One pixel `[A, R, G, B]`: `inv = 1 / A` and colour `c * inv` when `A as f64 >= 1e-12`, else
/// all four channels 0. The clear branch is taken when `A as f64 < 1e-12`, so a NaN alpha takes
/// the divide path. The colour is the first operand of the product: when both the colour and
/// the alpha are NaN the colour's payload
/// is kept (MEASURED against the original).
pub(crate) fn unpremultiply_f32(p: [f32; 4]) -> [f32; 4] {
    if (p[0] as f64) < UNPREMUL_F32_MIN_ALPHA {
        return [0.0; 4];
    }
    let inv = 1.0f32 / p[0];
    let r = [p[0], p[1] * inv, p[2] * inv, p[3] * inv];
    if crate::nanrule::any_nan(&r) {
        // `c * inv` with the colour as the first operand of the product (only the processor rule looks at the order)
        let m = |c: f32| crate::nanrule::mul(c, inv);
        [p[0], m(p[1]), m(p[2]), m(p[3])]
    } else {
        r
    }
}

/// Which of the two plugin error images is shown. Both are written straight into the output in
/// its own range (16 bpc: 0..=32768, 0x8000 is full) and the finish step is skipped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// A checked out layer has another size than the output: 50 percent red, identical in all
    /// four plugins.
    SizeMismatch,
    /// A layer is missing (width 0); the colour depends on the plugin.
    MissingLayer(MissingLayerStyle),
}

/// The missing layer fill of each plugin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissingLayerStyle {
    /// Plugin 01: 50 percent blue at every depth.
    HalfBlue,
    /// Plugin 02: grey. 8 bpc `(0xff, 0x7f, 0x7f, 0x7f)`; 16 bpc alpha is 1; the 32 bpc alpha is
    /// never written (0.0 here, see [`error_fill`]).
    Grey,
    /// Plugins 03 and 04: blue as plugin 01 at 8 bpc, but the 16 bpc alpha is 0 and the 32 bpc
    /// alpha is never written (0.0 here, see [`error_fill`]).
    BlueAlphaSlip,
}

/// The values the plugin writes as its output, per channel `[A, R, G, B]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ErrorFill {
    /// 8 bpc, 0..=255.
    U8([u8; 4]),
    /// 16 bpc, the range 0..=32768.
    U16([u16; 4]),
    /// 32 bpc, 1.0 is full.
    F32([f32; 4]),
}

/// The error fill of `kind` at `depth`. The 32 bpc alpha of plugins 02 to 04 is never written;
/// 0.0, the value of a cleared buffer, is an assumed value for it (plugin 01 clears the output
/// first and writes 0.5).
pub fn error_fill(kind: ErrorKind, depth: Depth) -> ErrorFill {
    use ErrorKind::*;
    use MissingLayerStyle::*;
    match (kind, depth) {
        (SizeMismatch, Depth::U8) => ErrorFill::U8([0x80, 0xff, 0, 0]),
        (SizeMismatch, Depth::U16) => ErrorFill::U16([0x4000, 0x8000, 0, 0]),
        (SizeMismatch, Depth::F32) => ErrorFill::F32([0.5, 1.0, 0.0, 0.0]),
        (MissingLayer(HalfBlue), Depth::U8) => ErrorFill::U8([0x80, 0, 0, 0xff]),
        (MissingLayer(HalfBlue), Depth::U16) => ErrorFill::U16([0x4000, 0, 0, 0x8000]),
        (MissingLayer(HalfBlue), Depth::F32) => ErrorFill::F32([0.5, 0.0, 0.0, 1.0]),
        (MissingLayer(Grey), Depth::U8) => ErrorFill::U8([0xff, 0x7f, 0x7f, 0x7f]),
        (MissingLayer(Grey), Depth::U16) => ErrorFill::U16([1, 0x4000, 0x4000, 0x4000]),
        (MissingLayer(Grey), Depth::F32) => ErrorFill::F32([0.0, 0.5, 0.5, 0.5]),
        (MissingLayer(BlueAlphaSlip), Depth::U8) => ErrorFill::U8([0x80, 0, 0, 0xff]),
        (MissingLayer(BlueAlphaSlip), Depth::U16) => ErrorFill::U16([0, 0, 0, 0x8000]),
        (MissingLayer(BlueAlphaSlip), Depth::F32) => ErrorFill::F32([0.0, 0.0, 0.0, 1.0]),
    }
}

/// The error image as a WIRE frame: [`error_fill`] converted to the wire (16 bpc through
/// `Frame::q15_to_u16`, channel order R, G, B, A). The output of the plugin in these cases skips
/// the unpremultiply, so nothing else is applied.
pub fn error_frame(kind: ErrorKind, width: usize, height: usize, depth: Depth) -> Frame {
    let mut frame = Frame::new(width, height, depth);
    match (error_fill(kind, depth), &mut frame.pixels) {
        (ErrorFill::U8([a, r, g, b]), Pixels::U8(px)) => px.fill([r, g, b, a]),
        (ErrorFill::U16(v), Pixels::U16(px)) => {
            let [a, r, g, b] = v.map(Frame::q15_to_u16);
            px.fill([r, g, b, a]);
        }
        (ErrorFill::F32([a, r, g, b]), Pixels::F32(px)) => px.fill([r, g, b, a]),
        _ => unreachable!("error_fill follows the depth"),
    }
    frame
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_images_are_cleared_and_shaped() {
        for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
            let img = Image::new(3, 2, fmt);
            assert_eq!((img.fmt(), img.row_len(), img.sample_count()), (fmt, 12, 24));
            assert!(img.same_shape(&Image::new(3, 2, fmt)));
            assert!(!img.same_shape(&Image::new(2, 3, fmt)));
        }
        assert!(!Image::new(1, 1, Fmt::U8).same_shape(&Image::new(1, 1, Fmt::U16)));
    }

    #[test]
    fn format_codes_are_stable() {
        assert_eq!([Fmt::U8, Fmt::U16, Fmt::F32].map(Fmt::code), [0, 1, 2]);
        assert_eq!([Fmt::U8, Fmt::U16, Fmt::F32].map(Fmt::white), [255.0, 65535.0, 1.0]);
        for d in [Depth::U8, Depth::U16, Depth::F32] {
            assert_eq!(Fmt::from_depth(d).depth(), d);
        }
    }

    #[test]
    fn clear_zeroes_every_sample() {
        let mut img = Image { width: 1, height: 1, samples: Samples::F32(vec![0.5; 4]) };
        img.clear();
        assert_eq!(img.samples, Samples::F32(vec![0.0; 4]));
    }

    // ---- boundary conversions ----

    /// FNV-1a over bytes, used to pin whole tables computed independently with numpy float32
    /// arithmetic from the document formulas (scratch script, not part of the repo).
    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in bytes {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    fn le16(vals: &[u16], out: &mut Vec<u8>) {
        for v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }

    #[test]
    fn constants_are_the_binary_bit_patterns() {
        assert_eq!(INV_255.to_bits(), 0x3b80_8081);
        assert_eq!(INV_255, 0.003921568859368563f32);
        assert_eq!(PREMUL_16_COLOR, 6.1034224927425385e-05f32);
        assert_eq!(PREMUL_16_ALPHA, 1.999969482421875f32);
        assert_eq!(UNPREMUL_16_ALPHA, 0.5000076293945312f32);
        assert_eq!(UNPREMUL_F32_MIN_ALPHA.to_bits(), 0x3d71_9799_812d_ea11);
        // 65535 / 32768 is exactly representable, so the alpha rescale maps full to full.
        assert_eq!(PREMUL_16_ALPHA as f64, 65535.0 / 32768.0);
    }

    #[test]
    fn premultiply_8bpc_every_colour_and_alpha() {
        let mut table = Vec::with_capacity(65536);
        for a in 0..=255u8 {
            for c in 0..=255u8 {
                let p = premultiply_u8([a, c, c, c]);
                assert_eq!(p[0], a, "alpha is unchanged");
                assert!(p[1] == p[2] && p[2] == p[3]);
                // Integer reference: exact rounding half up of c * a / 255. (2ca + 255) / 510
                // never meets a tie because 255 is odd, so the float formula must agree with it
                // for every pair.
                let exact = (2 * c as u32 * a as u32 + 255) / 510;
                assert_eq!(p[1] as u32, exact, "c {c} a {a}");
                table.push(p[1]);
            }
        }
        assert_eq!(fnv1a(&table), 0x1cc0_9b6f_9711_c88b, "numpy float32 table");
    }

    #[test]
    fn premultiply_8bpc_opaque_and_clear_alpha() {
        for c in 0..=255u8 {
            assert_eq!(premultiply_u8([255, c, c, c]), [255, c, c, c], "alpha 255 is the identity");
            assert_eq!(premultiply_u8([0, c, c, c]), [0, 0, 0, 0], "alpha 0 clears the colour");
        }
        // Hand worked: a = 128 * 0.0039215689 = 0.50196081; 200 * a = 100.39 -> 100; 100 * a = 50.196 -> 50; 50 * a = 25.098 -> 25.
        assert_eq!(premultiply_u8([128, 200, 100, 50]), [128, 100, 50, 25]);
    }

    #[test]
    fn unpremultiply_8bpc_every_colour_and_alpha() {
        let mut table = Vec::with_capacity(65536);
        let mut tie_down = 0;
        for a in 0..=255u8 {
            for c in 0..=255u8 {
                let p = unpremultiply_u8([a, c, c, c]);
                table.push(if a == 0 { c } else { p[1] });
                if a == 0 {
                    assert_eq!(p, [0, c, c, c], "alpha 0 is left untouched");
                    continue;
                }
                assert_eq!(p[0], a);
                // Integer reference: exact c * 255 / a rounded half up, capped at 255. The f32
                // `s = 255 / a` resolves exact ties (c * 255 / a == n + 0.5) downwards in some
                // cases; everything else must be exact.
                let num = 2 * c as u32 * 255;
                let exact = ((num + a as u32) / (2 * a as u32)).min(255);
                let is_tie = num % (2 * a as u32) == a as u32;
                if p[1] as u32 == exact {
                    continue;
                }
                assert!(is_tie && p[1] as u32 + 1 == exact, "a {a} c {c}: got {} exact {exact}", p[1]);
                if c <= a {
                    tie_down += 1;
                }
            }
        }
        // 17 ties with consistent input (c <= a), for example a = 14, c = 7: 127.5 -> 127.
        assert_eq!(tie_down, 17);
        assert_eq!(unpremultiply_u8([14, 7, 7, 7]), [14, 127, 127, 127]);
        assert_eq!(unpremultiply_u8([2, 1, 1, 1]), [2, 128, 128, 128], "127.5 is exact in f32 and rounds up");
        assert_eq!(fnv1a(&table), 0xfbdd_7857_08c6_d817, "numpy float32 table");
    }

    #[test]
    fn unpremultiply_8bpc_saturates_at_255_and_stores_low_byte() {
        // Inconsistent input (colour above alpha): v = c * 255 / a is capped at 255.0 before the +0.5.
        assert_eq!(unpremultiply_u8([100, 255, 200, 101]), [100, 255, 255, 255]);
        assert_eq!(unpremultiply_u8([255, 255, 254, 1]), [255, 255, 254, 1]);
    }

    #[test]
    fn premultiply_16bpc_every_alpha() {
        let mut table = Vec::new();
        for a in 0..=32768u16 {
            let colours = [0, 1, a / 2, a.saturating_sub(1), a, (a + 1).min(32768), 32768, 16384, 12345];
            let mut row = vec![premultiply_u16([a, 0, 0, 0])[0]];
            for &c in &colours {
                row.push(premultiply_u16([a, c, 0, 0])[1]);
            }
            le16(&row, &mut table);
        }
        assert_eq!(fnv1a(&table), 0xab09_baae_b5a3_a60b, "numpy float32 table");
        // End points and the alpha rescale 32768 -> 65535.
        assert_eq!(premultiply_u16([32768, 32768, 16384, 1]), [65535, 65535, 32768, 2]);
        assert_eq!(premultiply_u16([0, 32768, 5, 1]), [0, 0, 0, 0]);
        assert_eq!(premultiply_u16([16384, 16384, 16384, 16384])[0], 32768);
        // Colour times alpha, both scaled to 65535: 16384 * (16384/32768) * (65535/32768) = 16383.75 -> 16384.
        assert_eq!(premultiply_u16([16384, 32768, 16384, 0]), [32768, 32768, 16384, 0]);
    }

    #[test]
    fn unpremultiply_16bpc_every_alpha() {
        let mut table = Vec::new();
        for a in 0..=65535u16 {
            if a == 0 {
                le16(&[0; 9], &mut table);
                assert_eq!(unpremultiply_u16([0, 7, 8, 9]), [0, 7, 8, 9], "alpha 0 is left untouched");
                continue;
            }
            let colours = [0, 1, a / 2, a - 1, a, a.saturating_add(1), 65535, 32768];
            let mut row = vec![unpremultiply_u16([a, 0, 0, 0])[0]];
            for &c in &colours {
                row.push(unpremultiply_u16([a, c, 0, 0])[1]);
            }
            le16(&row, &mut table);
        }
        assert_eq!(fnv1a(&table), 0x0be1_d68e_5553_6dcf, "numpy float32 table");
        assert_eq!(unpremultiply_u16([65535, 65535, 32768, 1]), [32768, 32768, 16384, 1]);
        // No clamp: colour above alpha keeps growing (low 16 bits stored).
        assert_eq!(unpremultiply_u16([1000, 65535, 0, 0])[1], q(65535.0f32 * (32768.0f32 / 1000.0f32)) as u16);
    }

    #[test]
    fn sixteen_bit_opaque_chain_is_exact() {
        // Range 0..=32768: premultiply then unpremultiply of an opaque pixel returns every level
        // 0..=32768.
        for ae in 0..=32768u16 {
            let p = premultiply_u16([32768, ae, ae, ae]);
            assert_eq!(p[0], 65535);
            let back = unpremultiply_u16(p);
            assert_eq!(back, [32768, ae, ae, ae], "ae {ae}");
        }
        // Wire domain: the whole chain wire -> 0..=32768 -> engine -> 0..=32768 -> wire is what a
        // frame goes through. All 65536 wire levels at full alpha in one 256 x 256 frame.
        let all: Vec<u16> = (0..=65535u16).collect();
        let f = Frame { width: 256, height: 256, pixels: Pixels::U16(all.iter().map(|&v| [v, v, v, 65535]).collect()) };
        let want: Vec<[u16; 4]> = all
            .iter()
            .map(|&v| {
                let w = Frame::q15_to_u16(Frame::u16_to_q15(v));
                [w, w, w, 65535]
            })
            .collect();
        assert_eq!(Image::from_frame(&f).to_frame().pixels, Pixels::U16(want));
    }

    /// The NaN a float conversion returns when two NaNs meet follows `crate::nanrule` (the references carry the emulator's
    /// rule: the positive NaN of equal fractions wins, whatever the operand order). MEASURED in the original's
    /// premultiply and unpremultiply loops.
    #[cfg(feature = "emulator-nan")]
    #[test]
    fn float_conversions_keep_the_positive_nan() {
        let (pos, neg) = (f32::from_bits(0x7fc0_0000), f32::from_bits(0xffc0_0000));
        // alpha NaN of one sign, colours NaN of both signs and a number: the positive NaN wins, the number takes the alpha's
        let out = unpremultiply_f32([neg, pos, 1.0, neg]);
        assert_eq!(out.map(f32::to_bits), [0xffc0_0000, 0x7fc0_0000, 0xffc0_0000, 0xffc0_0000]);
        let out = unpremultiply_f32([pos, neg, 1.0, pos]);
        assert_eq!(out.map(f32::to_bits), [0x7fc0_0000, 0x7fc0_0000, 0x7fc0_0000, 0x7fc0_0000]);
        // premultiply: alpha 0 with an infinite colour is the default NaN; NaN beside NaN keeps the positive one
        let out = premultiply_f32([0.0, f32::INFINITY, pos, 0.5]);
        assert_eq!(out.map(f32::to_bits), [0, 0xffc0_0000, 0x7fc0_0000, 0]);
        let out = premultiply_f32([pos, neg, 1.0, f32::from_bits(0x7fc0_0001)]);
        assert_eq!(out.map(f32::to_bits), [0x7fc0_0000, 0x7fc0_0000, 0x7fc0_0000, 0x7fc0_0001]);
    }

    #[test]
    fn float_premultiply_and_the_unpremultiply_threshold() {
        assert_eq!(premultiply_f32([0.5, 0.8, 0.4, 1.0]), [0.5, 0.5f32 * 0.8, 0.5f32 * 0.4, 0.5]);
        assert_eq!(unpremultiply_f32([0.5, 0.25, 0.5, 0.125]), [0.5, 0.5, 1.0, 0.25]);
        // Alpha at or above 1e-12 (compared as f64) divides, below it clears all four channels.
        let nearest = 1e-12f32; // 9.99999982e-13, which is BELOW the f64 1e-12
        assert!((nearest as f64) < 1e-12);
        assert_eq!(unpremultiply_f32([nearest, 1e-13, 2e-13, 3e-13]), [0.0; 4]);
        let above = f32::from_bits(nearest.to_bits() + 1);
        assert!(above as f64 >= 1e-12);
        let out = unpremultiply_f32([above, 1e-13, 2e-13, 3e-13]);
        assert_eq!(out[0], above);
        assert_eq!(out[1], (1.0f32 / above) * 1e-13);
        assert_eq!(unpremultiply_f32([5e-13, 1.0, 1.0, 1.0]), [0.0; 4]);
        assert_eq!(unpremultiply_f32([2e-12, 1e-12, 0.0, 0.0])[1], (1.0f32 / 2e-12f32) * 1e-12f32);
        assert_eq!(unpremultiply_f32([0.0, 1.0, 1.0, 1.0]), [0.0; 4], "alpha 0");
        assert_eq!(unpremultiply_f32([-0.5, 1.0, 1.0, 1.0]), [0.0; 4], "negative alpha");
        // A NaN alpha (the default NaN of an invalid operation) and a NaN colour of another payload: the colour's
        // payload survives; a finite colour takes the alpha's.
        let out = unpremultiply_f32([f32::from_bits(0xffc0_0000), f32::from_bits(0x7fc0_0000), 1.0, f32::from_bits(0x7fc0_0000)]);
        assert_eq!(out.map(f32::to_bits), [0xffc0_0000, 0x7fc0_0000, 0xffc0_0000, 0x7fc0_0000]);
        // Values above 1 pass through (no clamp), NaN alpha takes the divide path.
        assert_eq!(unpremultiply_f32([2.0, 3.0, 4.0, 5.0]), [2.0, 1.5, 2.0, 2.5]);
        assert!(unpremultiply_f32([f32::NAN, 1.0, 1.0, 1.0])[0].is_nan());
        assert_eq!(premultiply_f32([1.0, 0.1, 0.2, 0.3]), [1.0, 0.1, 0.2, 0.3]);
    }

    fn frame8(px: Vec<[u8; 4]>, w: usize, h: usize) -> Frame {
        Frame { width: w, height: h, pixels: Pixels::U8(px) }
    }

    #[test]
    fn from_frame_orders_channels_and_positions() {
        // 3x2, opaque so the premultiply is the identity and only the layout is visible.
        let px: Vec<[u8; 4]> = (0..6u8).map(|i| [10 + i, 20 + i, 30 + i, 255]).collect();
        let img = Image::from_frame(&frame8(px.clone(), 3, 2));
        assert_eq!((img.width, img.height, img.fmt()), (3, 2, Fmt::U8));
        let Samples::U8(s) = &img.samples else { panic!("u8") };
        assert_eq!(&s[..8], &[255, 10, 20, 30, 255, 11, 21, 31], "A, R, G, B per pixel, left to right");
        assert_eq!(&s[12..16], &[255, 13, 23, 33], "second row starts at pixel 3");
        assert_eq!(img.to_frame(), frame8(px, 3, 2));
    }

    #[test]
    fn float_frames_keep_straight_wire_values_through_the_round_trip() {
        let px = vec![[0.25f32, 0.5, 0.75, 1.0], [2.0, -1.0, 0.5, 1.0], [0.1, 0.2, 0.3, 0.5], [0.4, 0.5, 0.6, 0.0]];
        let f = Frame { width: 2, height: 2, pixels: Pixels::F32(px) };
        let img = Image::from_frame(&f);
        let Samples::F32(s) = &img.samples else { panic!("f32") };
        assert_eq!(&s[0..4], &[1.0, 0.25, 0.5, 0.75]);
        assert_eq!(&s[4..8], &[1.0, 2.0, -1.0, 0.5], "out of range values are not clamped");
        assert_eq!(&s[8..12], &[0.5, 0.5f32 * 0.1, 0.5f32 * 0.2, 0.5f32 * 0.3]);
        assert_eq!(&s[12..16], &[0.0, 0.0, 0.0, 0.0], "alpha 0 premultiplies to zero");
        let Pixels::F32(back) = Image::from_frame(&f).to_frame().pixels else { panic!("f32") };
        assert_eq!(back[0], [0.25, 0.5, 0.75, 1.0]);
        assert_eq!(back[1], [2.0, -1.0, 0.5, 1.0]);
        // Alpha below one: (a * c) / a in f32 is not always c; here it is the documented expression.
        let inv = 1.0f32 / 0.5;
        assert_eq!(back[2], [inv * (0.5f32 * 0.1), inv * (0.5f32 * 0.2), inv * (0.5f32 * 0.3), 0.5]);
        assert_eq!(back[3], [0.0, 0.0, 0.0, 0.0], "alpha 0 loses its colour");
    }

    #[test]
    fn eight_bit_round_trip_is_lossy_below_full_alpha() {
        // Hand worked in the premultiply test: [200, 100, 50, 128] -> [128, 100, 50, 25];
        // back: s = 255 / 128 = 1.9921875, 100 * s = 199.22 -> 199, 50 * s = 99.61 -> 100, 25 * s = 49.80 -> 50.
        let f = frame8(vec![[200, 100, 50, 128]], 1, 1);
        assert_eq!(Image::from_frame(&f).to_frame(), frame8(vec![[199, 100, 50, 128]], 1, 1));
        // Alpha 0 loses the colour although the checkouts ask for it to be preserved.
        let f = frame8(vec![[200, 100, 50, 0]], 1, 1);
        assert_eq!(Image::from_frame(&f).to_frame(), frame8(vec![[0, 0, 0, 0]], 1, 1));
        // Opaque pixels survive for every value.
        let px: Vec<[u8; 4]> = (0..=255u8).map(|c| [c, 255 - c, c / 2, 255]).collect();
        assert_eq!(Image::from_frame(&frame8(px.clone(), 256, 1)).to_frame(), frame8(px, 256, 1));
        // Not every semi transparent pixel is changed: count how many of all (colour, alpha) pairs come back different.
        let mut different = 0;
        for a in 1..=254u8 {
            for c in 0..=255u8 {
                let f = frame8(vec![[c, c, c, a]], 1, 1);
                if Image::from_frame(&f).to_frame() != f {
                    different += 1;
                }
            }
        }
        assert!(different > 10_000, "{different}");
    }

    #[test]
    fn sixteen_bit_wire_chain_golden_values() {
        // wire (R, G, B, A) = (32768, 0, 65535, 65535): in the range 0..=32768 (16384, 0, 32768,
        // 32768); engine alpha 65535, red 16384 * 1.99997 = 32767.5 -> 32768, blue 32768 *
        // 1.99997 = 65535; back: red 32768 * 0.50000763 = 16384.25 -> 16384, blue 65535 *
        // 0.50000763 = 32768.0 -> 32768, alpha 65535 * 0.50000763 -> 32768.
        let f = Frame { width: 1, height: 1, pixels: Pixels::U16(vec![[32768, 0, 65535, 65535]]) };
        let img = Image::from_frame(&f);
        assert_eq!(img.samples, Samples::U16(vec![65535, 32768, 0, 65535]));
        assert_eq!(img.to_frame(), f);
        // Half alpha: wire alpha 32768 -> 16384 in the range 0..=32768 -> engine 32768 (exactly
        // half of 65535 plus one level).
        let f = Frame { width: 1, height: 1, pixels: Pixels::U16(vec![[65535, 65535, 65535, 32768]]) };
        let img = Image::from_frame(&f);
        assert_eq!(img.samples, Samples::U16(vec![32768, 32768, 32768, 32768]));
        // Back: s = 32768 / 32768 = 1, so the colour 32768 (full) comes back as 32768 and maps to
        // wire 65535; the alpha 16384 stays below full.
        let back = img.to_frame();
        assert_eq!(back.pixels, Pixels::U16(vec![[65535, 65535, 65535, 32768]]));
        // Alpha 0 loses the colour.
        let f = Frame { width: 1, height: 1, pixels: Pixels::U16(vec![[65535, 1, 2, 0]]) };
        assert_eq!(Image::from_frame(&f).to_frame().pixels, Pixels::U16(vec![[0, 0, 0, 0]]));
    }

    #[test]
    fn empty_frames_convert() {
        for depth in [Depth::U8, Depth::U16, Depth::F32] {
            let f = Frame::new(0, 5, depth);
            let img = Image::from_frame(&f);
            assert_eq!((img.width, img.height, img.sample_count()), (0, 5, 0));
            assert_eq!(img.to_frame(), f);
        }
    }

    // ---- error fills ----

    #[test]
    fn error_fills_match_the_specifications() {
        use ErrorKind::*;
        use MissingLayerStyle::*;
        let fill = |k, d| error_fill(k, d);
        // Size mismatch: 50 percent red, identical in every plugin.
        assert_eq!(fill(SizeMismatch, Depth::U8), ErrorFill::U8([0x80, 0xff, 0, 0]));
        assert_eq!(fill(SizeMismatch, Depth::U16), ErrorFill::U16([0x4000, 0x8000, 0, 0]));
        assert_eq!(fill(SizeMismatch, Depth::F32), ErrorFill::F32([0.5, 1.0, 0.0, 0.0]));
        // Plugin 01: 50 percent blue.
        assert_eq!(fill(MissingLayer(HalfBlue), Depth::U8), ErrorFill::U8([0x80, 0, 0, 0xff]));
        assert_eq!(fill(MissingLayer(HalfBlue), Depth::U16), ErrorFill::U16([0x4000, 0, 0, 0x8000]));
        assert_eq!(fill(MissingLayer(HalfBlue), Depth::F32), ErrorFill::F32([0.5, 0.0, 0.0, 1.0]));
        // Plugin 02: grey, with its alpha slips.
        assert_eq!(fill(MissingLayer(Grey), Depth::U8), ErrorFill::U8([0xff, 0x7f, 0x7f, 0x7f]));
        assert_eq!(fill(MissingLayer(Grey), Depth::U16), ErrorFill::U16([1, 0x4000, 0x4000, 0x4000]));
        assert_eq!(fill(MissingLayer(Grey), Depth::F32), ErrorFill::F32([0.0, 0.5, 0.5, 0.5]));
        // Plugins 03 and 04: blue, 16 bpc alpha 0, 32 bpc alpha never written.
        assert_eq!(fill(MissingLayer(BlueAlphaSlip), Depth::U8), ErrorFill::U8([0x80, 0, 0, 0xff]));
        assert_eq!(fill(MissingLayer(BlueAlphaSlip), Depth::U16), ErrorFill::U16([0, 0, 0, 0x8000]));
        assert_eq!(fill(MissingLayer(BlueAlphaSlip), Depth::F32), ErrorFill::F32([0.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn error_frames_are_wire_frames() {
        use ErrorKind::*;
        use MissingLayerStyle::*;
        let px = |f: &Frame| match &f.pixels {
            Pixels::U8(p) => p.iter().map(|v| v.map(|c| c as f64)).collect::<Vec<_>>(),
            Pixels::U16(p) => p.iter().map(|v| v.map(|c| c as f64)).collect(),
            Pixels::F32(p) => p.iter().map(|v| v.map(|c| c as f64)).collect(),
        };
        // 8 bpc: wire order R, G, B, A.
        let f = error_frame(SizeMismatch, 2, 3, Depth::U8);
        assert_eq!((f.width, f.height, f.depth()), (2, 3, Depth::U8));
        assert_eq!(px(&f), vec![[255.0, 0.0, 0.0, 128.0]; 6]);
        let f = error_frame(MissingLayer(HalfBlue), 1, 1, Depth::U8);
        assert_eq!(px(&f), vec![[0.0, 0.0, 255.0, 128.0]]);
        // 16 bpc: the values 0x4000 and 0x8000 of the range 0..=32768 map to 32768 and 65535 on
        // the wire, alpha 1 to 2.
        let f = error_frame(SizeMismatch, 1, 1, Depth::U16);
        assert_eq!(px(&f), vec![[65535.0, 0.0, 0.0, 32768.0]]);
        let f = error_frame(MissingLayer(Grey), 1, 1, Depth::U16);
        assert_eq!(px(&f), vec![[32768.0, 32768.0, 32768.0, 2.0]]);
        let f = error_frame(MissingLayer(BlueAlphaSlip), 1, 1, Depth::U16);
        assert_eq!(px(&f), vec![[0.0, 0.0, 65535.0, 0.0]]);
        // 32 bpc passes through.
        let f = error_frame(SizeMismatch, 1, 2, Depth::F32);
        assert_eq!(px(&f), vec![[1.0, 0.0, 0.0, 0.5]; 2]);
        let f = error_frame(MissingLayer(Grey), 1, 1, Depth::F32);
        assert_eq!(px(&f), vec![[0.5, 0.5, 0.5, 0.0]]);
        // An empty frame stays empty.
        assert_eq!(error_frame(SizeMismatch, 0, 0, Depth::U8).byte_len(), 0);
    }
}
