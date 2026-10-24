/// Channel depth of a frame.
///
/// 16 bpc values use the range 0..=65535 on the wire. Effects whose math runs on the range
/// 0..=32768 convert with [`Frame::u16_to_q15`] and [`Frame::q15_to_u16`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Depth {
    U8,
    U16,
    F32,
}

impl Depth {
    pub fn from_bits(bits: u32) -> Option<Depth> {
        match bits {
            8 => Some(Depth::U8),
            16 => Some(Depth::U16),
            32 => Some(Depth::F32),
            _ => None,
        }
    }

    pub fn bits(self) -> u32 {
        match self {
            Depth::U8 => 8,
            Depth::U16 => 16,
            Depth::F32 => 32,
        }
    }

    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Depth::U8 => 4,
            Depth::U16 => 8,
            Depth::F32 => 16,
        }
    }
}

/// Straight (not premultiplied) RGBA pixels, row major, no padding.
#[derive(Clone, Debug, PartialEq)]
pub enum Pixels {
    U8(Vec<[u8; 4]>),
    U16(Vec<[u16; 4]>),
    F32(Vec<[f32; 4]>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub pixels: Pixels,
}

impl Frame {
    pub fn new(width: usize, height: usize, depth: Depth) -> Frame {
        let n = width * height;
        let pixels = match depth {
            Depth::U8 => Pixels::U8(vec![[0; 4]; n]),
            Depth::U16 => Pixels::U16(vec![[0; 4]; n]),
            Depth::F32 => Pixels::F32(vec![[0.0; 4]; n]),
        };
        Frame { width, height, pixels }
    }

    pub fn depth(&self) -> Depth {
        match self.pixels {
            Pixels::U8(_) => Depth::U8,
            Pixels::U16(_) => Depth::U16,
            Pixels::F32(_) => Depth::F32,
        }
    }

    pub fn byte_len(&self) -> usize {
        self.width * self.height * self.depth().bytes_per_pixel()
    }

    /// Reads one frame of packed little endian RGBA (`rgba`, `rgba64le`) or
    /// planar float (`gbrapf32le`, planes G, B, R, A) as ffmpeg writes it.
    pub fn from_bytes(width: usize, height: usize, depth: Depth, bytes: &[u8]) -> Frame {
        let n = width * height;
        assert_eq!(bytes.len(), n * depth.bytes_per_pixel());
        let pixels = match depth {
            Depth::U8 => Pixels::U8(
                bytes
                    .chunks_exact(4)
                    .map(|c| [c[0], c[1], c[2], c[3]])
                    .collect(),
            ),
            Depth::U16 => Pixels::U16(
                bytes
                    .chunks_exact(8)
                    .map(|c| {
                        let w = |i: usize| u16::from_le_bytes([c[i], c[i + 1]]);
                        [w(0), w(2), w(4), w(6)]
                    })
                    .collect(),
            ),
            Depth::F32 => {
                let plane = |p: usize, i: usize| {
                    let o = (p * n + i) * 4;
                    f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
                };
                Pixels::F32(
                    (0..n)
                        .map(|i| [plane(2, i), plane(0, i), plane(1, i), plane(3, i)])
                        .collect(),
                )
            }
        };
        Frame { width, height, pixels }
    }

    /// Inverse of [`Frame::from_bytes`].
    pub fn to_bytes(&self) -> Vec<u8> {
        let n = self.width * self.height;
        let mut out = Vec::with_capacity(self.byte_len());
        match &self.pixels {
            Pixels::U8(px) => px.iter().for_each(|p| out.extend_from_slice(p)),
            Pixels::U16(px) => px
                .iter()
                .for_each(|p| p.iter().for_each(|c| out.extend_from_slice(&c.to_le_bytes()))),
            Pixels::F32(px) => {
                for ch in [1usize, 2, 0, 3] {
                    for p in px.iter().take(n) {
                        out.extend_from_slice(&p[ch].to_le_bytes());
                    }
                }
            }
        }
        out
    }

    /// Wire range 0..=65535 to the range 0..=32768, rounding half up.
    pub fn u16_to_q15(v: u16) -> u16 {
        ((v as u32 * 32768 + 32767) / 65535) as u16
    }

    /// Range 0..=32768 to the wire range 0..=65535, rounding half up.
    pub fn q15_to_u16(v: u16) -> u16 {
        ((v.min(32768) as u32 * 65535 + 16384) / 32768) as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_round_trip_every_depth() {
        for depth in [Depth::U8, Depth::U16, Depth::F32] {
            let bytes: Vec<u8> = (0..3 * 2 * depth.bytes_per_pixel())
                .map(|i| (i * 7 % 61) as u8)
                .collect();
            let f = Frame::from_bytes(3, 2, depth, &bytes);
            assert_eq!(f.to_bytes(), bytes);
        }
    }

    #[test]
    fn q15_range_end_points() {
        assert_eq!(Frame::u16_to_q15(0), 0);
        assert_eq!(Frame::u16_to_q15(65535), 32768);
        assert_eq!(Frame::q15_to_u16(32768), 65535);
        for v in 0..=32768u16 {
            assert_eq!(Frame::u16_to_q15(Frame::q15_to_u16(v)), v);
        }
    }
}
