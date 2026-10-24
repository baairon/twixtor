//! Create Vectors pass: estimates the optical flow between two frames of the source layer and
//! writes it into the output as an encoded vector image ([`crate::vectors`]). This module is the
//! engine side of the single layer case (no FG matte layers, no user geometry, `Main_BG: Track`
//! checked):
//!
//! * [`frame_index`], [`frame_pair`], [`frame_time`]: the frame arithmetic;
//! * [`CreateVectorsSettings`] and [`create_vectors`]: solve the flow of one layer and encode it.
//!
//! Resolution of the result: the solver returns its flow at the working resolution (`W >> R`, R = 1
//! for Normal (half res), 0 for Full res, one more halving at draft quality). It is not resampled
//! to the output size in a separate step. The encoder does it: for every pixel of the output image
//! it samples the flow bilinearly at `(x * (fw - 1) / (dw - 1), y * (fh - 1) / (dh - 1))` and
//! multiplies the displacement by `(dw - 1) / (fw - 1)` before dividing by `Max Displace`, so the
//! encoded image is always at the output size and in pixels of the output.
//! [`vectors::encode_vectors`] is exactly that.

use crate::flow::{self, FlowField, FlowSettings};
use crate::image::Image;
use crate::timeline::cvtt_i32_f32;
use crate::vectors;
use fx_core::Result;

/// Flow quality code for Vector Resolution "Normal (half res)" (the popup value 1).
pub const QUALITY_NORMAL: u8 = 4;
/// Vector Resolution "Full res" as the engine argument a14 (the popup value 2).
pub const QUALITY_FULL: u8 = 5;

/// Vector Resolution popup value (1 Normal (half res), 2 Full res) as the flow quality code; any
/// other value gives the Normal code, 4.
pub fn quality_code(popup: i32) -> u8 {
    if popup == 2 {
        QUALITY_FULL
    } else {
        QUALITY_NORMAL
    }
}

/// The current frame number of the pre-render and the render:
/// `(u32)((float)current_time / (float)time_step + 0.5)`, in single precision. The conversion
/// truncates toward zero and the low 32 bits are kept.
pub fn frame_index(current_time: i32, time_step: i32) -> u32 {
    let x = (current_time as f32) / (time_step as f32) + 0.5f32;
    (x as i64) as u32
}

/// The two frame numbers `(a, b)` of one Create Vectors render: `a` is the image `X` and `b` the
/// image `Y` of the solver (`X(p) ~ Y(p + flow(p))`).
///
/// `track_frame` is the popup value (1 Previous, 2 Next; anything else leaves the pair at
/// `(n, n)`), `debug` is `Display == Debug`. Not debug: Previous gives `b = n - 1`, Next
/// `b = n + 1`. Debug swaps the roles: Previous gives `a = n + 1`, Next `a = n - 1`. The
/// arithmetic is in single precision (`(float)n`, `+- 1.0f`, truncating back).
pub fn frame_pair(frame: u32, track_frame: i32, debug: bool) -> (i32, i32) {
    let n = (frame as i32) as f32;
    let (mut a, mut b) = (n, n);
    if !debug {
        match track_frame {
            1 => b = n - 1.0f32,
            2 => b = n + 1.0f32,
            _ => {}
        }
    } else {
        match track_frame {
            1 => a = n + 1.0f32,
            2 => a = n - 1.0f32,
            _ => {}
        }
    }
    (cvtt_i32_f32(a), cvtt_i32_f32(b))
}

/// The layer time of frame number `n`: the frame number times `(float)time_step` plus one half, in
/// single precision, rounded half away from zero (`n < 0` negates around the truncation).
pub fn frame_time(n: i32, time_step: i32) -> i32 {
    let ts = time_step as f32;
    let x = n as f32;
    let t = if x < 0.0 { -((-x) * ts + 0.5f32) } else { ts * x + 0.5f32 };
    cvtt_i32_f32(t)
}

/// Everything the Create Vectors pass reads besides the two images.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CreateVectorsSettings {
    /// a14, [`QUALITY_NORMAL`] or [`QUALITY_FULL`].
    pub quality: u8,
    /// High quality render (not draft); draft quality halves the working resolution once more.
    pub hiq: bool,
    /// `Main_BG: Sensitivity` already divided by 100.
    pub sensitivity: f32,
    /// a10: `Prep == Contrast/Edge Enhance`.
    pub prep_contrast_edge: bool,
    /// a15: `Prep == Delinearize`.
    pub prep_delinearize: bool,
    /// a16: `Track RGB+A`, the `rgba` flag of the luma conversion (alpha added to the luma).
    pub track_rgb_a: bool,
    /// `Max Displace` times the horizontal downsample factor of the render, in pixels of the output
    /// resolution.
    pub max_displace: f32,
}

impl CreateVectorsSettings {
    /// The solver settings of the pass: the video range luma and `skip_refine` are off.
    pub fn flow_settings(&self) -> FlowSettings {
        FlowSettings {
            quality: self.quality,
            hiq: self.hiq,
            sensitivity: self.sensitivity,
            prep_contrast_edge: self.prep_contrast_edge,
            prep_delinearize: self.prep_delinearize,
            track_alpha: self.track_rgb_a,
            video_range_luma: false,
            skip_refine: false,
        }
    }
}

/// The flow of the pass at the solver's working resolution: `X(p) ~ Y(p + flow(p))`. `x` and `y`
/// are the premultiplied engine images of frames `a` and `b`, same size and format.
pub fn solve_flow(x: &Image, y: &Image, settings: &CreateVectorsSettings) -> Result<FlowField> {
    flow::solve(x, y, &settings.flow_settings())
}

/// The Create Vectors pass for one layer: solves the flow of `x` towards `y` and encodes it into a
/// new image of the size and format of `x`: red `dx`, green `dy` centred and scaled by
/// `max_displace`, blue 0, alpha white.
///
/// Errors are the solver's (`Mismatch` for images of different shapes, `Unsupported` for a working
/// size below 2x2) and the encoder's.
pub fn create_vectors(x: &Image, y: &Image, settings: &CreateVectorsSettings) -> Result<Image> {
    let flow = solve_flow(x, y, settings)?;
    vectors::encode_vectors_image(x.width, x.height, x.fmt(), &flow, settings.max_displace)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_index_rounds_to_the_nearest_frame() {
        assert_eq!(frame_index(0, 100), 0);
        assert_eq!(frame_index(700, 100), 7);
        // half a frame rounds up, just under rounds down
        assert_eq!(frame_index(50, 100), 1);
        assert_eq!(frame_index(49, 100), 0);
        assert_eq!(frame_index(749, 100), 7);
        assert_eq!(frame_index(750, 100), 8);
    }

    #[test]
    fn frame_pair_follows_track_frame_and_debug() {
        assert_eq!(frame_pair(7, 1, false), (7, 6));
        assert_eq!(frame_pair(7, 2, false), (7, 8));
        assert_eq!(frame_pair(7, 1, true), (8, 7));
        assert_eq!(frame_pair(7, 2, true), (6, 7));
        assert_eq!(frame_pair(0, 1, false), (0, -1));
    }

    #[test]
    fn frame_time_of_whole_frames() {
        assert_eq!(frame_time(7, 100), 700);
        assert_eq!(frame_time(-1, 100), -100);
        // time steps that are not integers per frame in float: 1001 ticks per frame
        assert_eq!(frame_time(3, 1001), 3003);
        assert_eq!(frame_time(0, 1), 0);
        assert_eq!(frame_time(-5, 1), -5);
    }

    #[test]
    fn frame_time_is_single_precision() {
        // the product of two integers is exact in f32 only below 2^24: frame 16777 at 1001 ticks
        // per frame is 16793777, which f32 stores as the even neighbour 16793776 (a tie), so that
        // tick is addressed, not 16793777
        assert_eq!(frame_time(16777, 1001), 16_793_776);
        assert_eq!(frame_time(-16777, 1001), -16_793_776);
        // below 2^24 the product is exact
        assert_eq!(frame_time(16000, 1001), 16_016_000);
    }

    #[test]
    fn quality_codes() {
        assert_eq!(quality_code(1), 4);
        assert_eq!(quality_code(2), 5);
    }

    #[test]
    fn flow_settings_carry_the_prep_and_alpha_flags() {
        let s = CreateVectorsSettings {
            quality: 5,
            hiq: false,
            sensitivity: 0.25,
            prep_contrast_edge: true,
            prep_delinearize: false,
            track_rgb_a: true,
            max_displace: 100.0,
        };
        let f = s.flow_settings();
        assert_eq!((f.quality, f.hiq, f.sensitivity), (5, false, 0.25));
        assert!(f.prep_contrast_edge && !f.prep_delinearize && f.track_alpha);
        assert!(!f.video_range_luma && !f.skip_refine);
    }
}
