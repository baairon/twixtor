//! Engine settings and the host interface of the engine core.
//!
//! [`EngineSettings`] is everything the engine derives from the parameters and the time context
//! before it calls the planner (pre-render) or the core (render). [`EngineHost`] is the engine's
//! way to get source frames: it turns an integer frame position into layer time with the time
//! map, fetches the frame and premultiplies it, for the retimed layer and for the tracking layer.
//! [`EnvHost`] is the ready implementation over `fx_core::Env`.

use crate::image::{ErrorKind, Fmt, Image, MissingLayerStyle};
use crate::planner::{LAYER_ALT_A, LAYER_ALT_B};
use crate::timeline::{Rates, TimeMap};
use fx_core::{Env, FieldOrder, FxError, Layer, Result};
use std::cell::Cell;

/// Everything the engine reads from the parameters and the time context for one output frame.
/// Built by `frontend::read_settings`. Popup indices are stored as the engine uses them (popup
/// value minus one).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EngineSettings {
    /// Current time of the output frame, in ticks of layer time.
    pub current_time: i32,
    /// Time step in ticks: the next output frame is at `current_time + time_step`.
    pub time_step: i32,
    /// Ticks per layer frame, the divisor of the frame positions.
    pub local_time_step: i32,
    /// Time scale of the ticks; only passed along with every checkout.
    pub time_scale: i32,
    /// High render quality: selects the solver resolution and the motion blur tap mode of Blur
    /// Compensation. Unused by the fast path.
    pub quality_high: bool,
    /// Output size, the result rectangle of the input layer.
    pub width: usize,
    pub height: usize,
    /// Pixel format of the output and of every source.
    pub fmt: Fmt,
    /// The rate model: source and composition frame rates and the duration of one source frame.
    pub rates: Rates,
    /// `Input: Fields` popup: any value but `Progressive` doubles `rates.in_fps` and makes the
    /// pre-render fetch every plan entry a second time at +0.5.
    pub input_fields: FieldOrder,
    /// `Display == Source` (popup value 2) (row S10).
    pub display_source: bool,
    /// `Src Time Offset` as the raw 16.16 integer, read at `current_time`; only the `Display == Source`
    /// branch uses this reading (the time map samples the parameter itself).
    pub src_time_offset_fixed: i32,
    /// `Vectors` popup minus one (0 = No Vectors .. 4 = Best), before the segment override.
    pub vectors_quality: i32,
    /// `Frame Interp` popup minus one (0 Nearest, 1 Blend, 2 Weighted Blend), before the segment
    /// override.
    pub interp: i32,
    /// `Blur Compensation`, the fixed point reader with scale 1.0f.
    pub blur: f32,
    /// `Sensitivity` slider divided by 100.0f.
    pub sensitivity: f32,
    /// `Prep` popup == 2, Contrast/Edge Enhance.
    pub prep_contrast_edge: bool,
    /// `Prep` popup == 3, Delinearize.
    pub prep_delinearize: bool,
    /// `Smart Blend` checkbox.
    pub smart_blend: bool,
    /// The warping code the core reads. Plugin 01: the `Smart Blend` checkbox, so 0 or 1; plugins
    /// 03 and 04: their `Warping` popup (0 to 3).
    pub warp_mode: i32,
    /// `360 Video` (row S11).
    pub video_360: bool,
    /// `Plenty Memory?` / `Cache Last?`: caching only, no effect on values. The raw checkbox is
    /// kept as it is; nothing reads it.
    pub cache_last: bool,
    /// The retimed layer is Alt Source A (it has a non empty size at `current_time`); otherwise
    /// it is the input layer.
    pub alt_a_present: bool,
    /// `noB`: Alt Source B (the tracking layer) is not set or has no pixels.
    pub no_tracking_layer: bool,
    /// Which fill the missing layer error shows (plugin 01: [`MissingLayerStyle::HalfBlue`]).
    pub missing_layer_style: MissingLayerStyle,
}

/// The two error flags of the pre-render data. They are sticky: a probe or a fetch raises them
/// and the render shows the error fill instead of the picture.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LayerIssues {
    /// A checked out layer has a size other than the output.
    pub size_mismatch: bool,
    /// A layer has width or height 0.
    pub missing_layer: bool,
}

impl LayerIssues {
    pub fn any(self) -> bool {
        self.size_mismatch || self.missing_layer
    }

    /// The error fill to show. The size mismatch flag is tested before the missing layer flag, so
    /// red wins when both are set.
    pub fn error_kind(self, missing: MissingLayerStyle) -> Option<ErrorKind> {
        if self.size_mismatch {
            Some(ErrorKind::SizeMismatch)
        } else if self.missing_layer {
            Some(ErrorKind::MissingLayer(missing))
        } else {
            None
        }
    }
}

/// The fetch interface of the engine, as the core uses it. Images come back in the engine domain:
/// premultiplied, 16 bpc in 0..=65535 (`Image::from_frame`), and always at the output size and
/// format.
///
/// A frame without pixels (`Layer::frame_at` gives `None`, a time outside the clip) is a
/// transparent image. A frame whose size is not the output size comes back as a transparent image
/// too and raises the matching flag in [`EngineHost::issues`]; the caller shows the error fill
/// (`compose::render` does).
pub trait EngineHost {
    /// The plugin's time map (`src_pos`, `frame_time`, `marker`). The selection and the fetches
    /// must use this one object: `RemapTimeMap` keeps `base`, which `src_pos` writes and
    /// `frame_time` reads.
    fn time_map(&self) -> &dyn TimeMap;

    /// The frame of the retimed layer shown at `time` ticks of layer time (Alt Source A when it
    /// has pixels, else the input layer). Also the only fetch of `Display == Source`.
    fn main_frame_at(&self, time: i32) -> Result<Image>;

    /// The same for the tracking layer (Alt Source B). An error when there is none
    /// (`EngineSettings::no_tracking_layer`).
    fn tracking_frame_at(&self, time: i32) -> Result<Image>;

    /// The flags set so far: by the probes made when the host was built and by every fetch since.
    fn issues(&self) -> LayerIssues;

    /// Frame number `position` of the retimed layer at layer time
    /// `time_map().frame_time(position as f32)`.
    fn main_frame(&self, position: i32) -> Result<Image> {
        self.main_frame_at(self.time_map().frame_time(position as f32))
    }

    /// The same for the tracking layer.
    fn tracking_frame(&self, position: i32) -> Result<Image> {
        self.tracking_frame_at(self.time_map().frame_time(position as f32))
    }
}

/// The ids of the two optional layers of a front-end. The defaults are plugin 01's
/// (`planner::LAYER_ALT_A`, `planner::LAYER_ALT_B`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerIds {
    /// The retimed layer replacement ("Alt Source A").
    pub alt_a: &'static str,
    /// The tracking layer ("Alt Source B").
    pub alt_b: &'static str,
}

impl Default for LayerIds {
    fn default() -> LayerIds {
        LayerIds { alt_a: LAYER_ALT_A, alt_b: LAYER_ALT_B }
    }
}

/// Probe of a layer parameter: set means the layer exists and has a non empty size.
pub fn layer_is_set(layer: Option<&dyn Layer>) -> bool {
    layer.is_some_and(|l| l.width() > 0 && l.height() > 0)
}

/// [`EngineHost`] over `fx_core::Env`: the retimed layer is the layer `alt_a` when
/// `EngineSettings::alt_a_present`, else `env.input`; the tracking layer is `alt_b` unless
/// `no_tracking_layer`.
///
/// Building the host makes the pre-render probes: a main layer of size 0 raises the missing layer
/// flag, a size other than the output raises the size mismatch flag; the tracking layer with
/// another size raises the size mismatch flag too, which shows the red fill. Every fetched frame
/// is checked again.
pub struct EnvHost<'a> {
    settings: &'a EngineSettings,
    map: &'a dyn TimeMap,
    main: &'a dyn Layer,
    tracking: Option<&'a dyn Layer>,
    size_mismatch: Cell<bool>,
    missing_layer: Cell<bool>,
}

impl<'a> EnvHost<'a> {
    /// `map` is the time map the caller also uses for the selection. An error when the settings
    /// ask for a layer the environment does not have, or a layer has another bit depth than the
    /// output (depths are not converted).
    pub fn new(env: &'a Env<'_>, settings: &'a EngineSettings, map: &'a dyn TimeMap, ids: LayerIds) -> Result<EnvHost<'a>> {
        let main: &'a dyn Layer = if settings.alt_a_present {
            env.layer(ids.alt_a).ok_or_else(|| {
                FxError::Mismatch(format!("the settings use the layer `{}` but the environment has none", ids.alt_a))
            })?
        } else {
            env.input
        };
        let tracking = if settings.no_tracking_layer {
            None
        } else {
            Some(env.layer(ids.alt_b).ok_or_else(|| {
                FxError::Mismatch(format!("the settings use the layer `{}` but the environment has none", ids.alt_b))
            })?)
        };
        for layer in std::iter::once(main).chain(tracking) {
            if layer.depth() != settings.fmt.depth() {
                return Err(FxError::Mismatch(format!(
                    "a layer is {} bpc but the output is {} bpc",
                    layer.depth().bits(),
                    settings.fmt.depth().bits()
                )));
            }
        }
        let host = EnvHost { settings, map, main, tracking, size_mismatch: Cell::new(false), missing_layer: Cell::new(false) };
        host.check_size(main.width(), main.height());
        if let Some(t) = tracking {
            host.check_size(t.width(), t.height());
        }
        Ok(host)
    }

    /// Width or height 0 is a missing layer, another size a size mismatch.
    fn check_size(&self, width: usize, height: usize) -> bool {
        if width == 0 || height == 0 {
            self.missing_layer.set(true);
            false
        } else if (width, height) != (self.settings.width, self.settings.height) {
            self.size_mismatch.set(true);
            false
        } else {
            true
        }
    }

    fn blank(&self) -> Image {
        Image::new(self.settings.width, self.settings.height, self.settings.fmt)
    }

    fn fetch(&self, layer: &dyn Layer, time: i32) -> Result<Image> {
        let Some(frame) = layer.frame_at(i64::from(time))? else {
            return Ok(self.blank());
        };
        if frame.depth() != self.settings.fmt.depth() {
            return Err(FxError::Mismatch(format!(
                "a frame is {} bpc but the output is {} bpc",
                frame.depth().bits(),
                self.settings.fmt.depth().bits()
            )));
        }
        if !self.check_size(frame.width, frame.height) {
            return Ok(self.blank());
        }
        Ok(Image::from_frame(&frame))
    }
}

impl EngineHost for EnvHost<'_> {
    fn time_map(&self) -> &dyn TimeMap {
        self.map
    }

    fn main_frame_at(&self, time: i32) -> Result<Image> {
        self.fetch(self.main, time)
    }

    fn tracking_frame_at(&self, time: i32) -> Result<Image> {
        match self.tracking {
            Some(layer) => self.fetch(layer, time),
            None => Err(FxError::Mismatch("there is no tracking layer (Alt Source B) to fetch from".into())),
        }
    }

    fn issues(&self) -> LayerIssues {
        LayerIssues { size_mismatch: self.size_mismatch.get(), missing_layer: self.missing_layer.get() }
    }
}

/// Settings for the tests of the engine: 30 fps both ways, 100 ticks per frame, output frame 0,
/// No Vectors, Blend, no alternate layers.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::timeline::test_support::plain_rates;

    pub fn settings(width: usize, height: usize, fmt: Fmt) -> EngineSettings {
        EngineSettings {
            current_time: 0,
            time_step: 100,
            local_time_step: 100,
            time_scale: 3000,
            quality_high: true,
            width,
            height,
            fmt,
            rates: plain_rates(),
            input_fields: FieldOrder::Progressive,
            display_source: false,
            src_time_offset_fixed: 0,
            vectors_quality: 0,
            interp: 1,
            blur: 0.0,
            sensitivity: 0.7,
            prep_contrast_edge: false,
            prep_delinearize: false,
            smart_blend: false,
            warp_mode: 0,
            video_360: false,
            cache_last: true,
            alt_a_present: false,
            no_tracking_layer: true,
            missing_layer_style: MissingLayerStyle::HalfBlue,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::settings;
    use super::*;
    use crate::timeline::test_support::Mock;
    use crate::timeline::RemapTimeMap;
    use fx_core::{Depth, Frame, LayerSet, MemLayer, ParamCurves, Pixels, Rational, TimeBase};
    use std::sync::Arc;

    fn base() -> TimeBase {
        TimeBase::default_for_rate(Rational::new(30, 1).unwrap())
    }

    /// A clip whose frame `i` is filled with the opaque grey `10 * (i + 1)` (8 bpc).
    fn grey_clip(n: usize, w: usize, h: usize) -> MemLayer {
        let frames = (0..n)
            .map(|i| {
                let mut f = Frame::new(w, h, Depth::U8);
                if let Pixels::U8(px) = &mut f.pixels {
                    px.fill([10 * (i as u8 + 1), 0, 0, 255]);
                }
                f
            })
            .collect();
        MemLayer::new(base(), frames).unwrap()
    }

    fn first_red(img: &Image) -> u8 {
        match &img.samples {
            crate::image::Samples::U8(v) => v[1],
            _ => unreachable!(),
        }
    }

    #[test]
    fn main_frames_come_from_the_time_map_and_missing_frames_are_transparent() {
        let clip = grey_clip(3, 2, 2);
        let params = ParamCurves::new(Vec::new());
        let layers = LayerSet::new();
        let env = Env { params: &params, input: &clip, layers: &layers };
        let s = settings(2, 2, Fmt::U8);
        let mock = Mock::speed(100.0);
        let map = RemapTimeMap::new(&mock, s.rates, 0);
        let host = EnvHost::new(&env, &s, &map, LayerIds::default()).unwrap();
        // frame_time(f) = f * 100 for equal rates: position 1 is layer time 100, the second clip frame.
        assert_eq!(first_red(&host.main_frame(0).unwrap()), 10);
        assert_eq!(first_red(&host.main_frame(1).unwrap()), 20);
        assert_eq!(first_red(&host.main_frame(2).unwrap()), 30);
        assert_eq!(host.main_frame_at(250).unwrap(), host.main_frame(2).unwrap());
        for outside in [host.main_frame(3).unwrap(), host.main_frame(-1).unwrap(), host.main_frame_at(i32::MIN).unwrap()] {
            assert_eq!(outside, Image::new(2, 2, Fmt::U8));
        }
        assert_eq!(host.issues(), LayerIssues::default());
        assert!(host.tracking_frame(0).is_err());
    }

    #[test]
    fn alt_source_a_replaces_the_input_and_the_tracking_layer_is_separate() {
        let input = grey_clip(2, 2, 2);
        let mut layers = LayerSet::new();
        let alt_a = MemLayer::new(base(), vec![Frame::new(2, 2, Depth::U8)]).unwrap();
        layers.insert(LAYER_ALT_A, Arc::new(alt_a));
        layers.insert(LAYER_ALT_B, Arc::new(grey_clip(2, 2, 2)));
        let params = ParamCurves::new(Vec::new());
        let env = Env { params: &params, input: &input, layers: &layers };
        let mut s = settings(2, 2, Fmt::U8);
        s.alt_a_present = true;
        s.no_tracking_layer = false;
        let mock = Mock::speed(100.0);
        let map = RemapTimeMap::new(&mock, s.rates, 0);
        let host = EnvHost::new(&env, &s, &map, LayerIds::default()).unwrap();
        assert_eq!(host.main_frame(0).unwrap(), Image::from_frame(&Frame::new(2, 2, Depth::U8)));
        assert_eq!(first_red(&host.tracking_frame(1).unwrap()), 20);
        // asking for a layer the environment lacks is a configuration error
        let empty = LayerSet::new();
        let env = Env { params: &params, input: &input, layers: &empty };
        assert!(EnvHost::new(&env, &s, &map, LayerIds::default()).is_err());
    }

    #[test]
    fn sizes_set_the_flags_and_red_wins() {
        let params = ParamCurves::new(Vec::new());
        let mock = Mock::speed(100.0);
        let layers = LayerSet::new();
        // the input is 3x2 but the output is 2x2: size mismatch at the probe
        let wide = grey_clip(2, 3, 2);
        let env = Env { params: &params, input: &wide, layers: &layers };
        let s = settings(2, 2, Fmt::U8);
        let map = RemapTimeMap::new(&mock, s.rates, 0);
        let host = EnvHost::new(&env, &s, &map, LayerIds::default()).unwrap();
        assert_eq!(host.issues(), LayerIssues { size_mismatch: true, missing_layer: false });
        assert_eq!(host.main_frame(0).unwrap(), Image::new(2, 2, Fmt::U8));
        // a zero sized input is a missing layer
        let none = MemLayer::new(base(), vec![Frame::new(0, 0, Depth::U8)]).unwrap();
        let env = Env { params: &params, input: &none, layers: &layers };
        let host = EnvHost::new(&env, &s, &map, LayerIds::default()).unwrap();
        assert_eq!(host.issues(), LayerIssues { size_mismatch: false, missing_layer: true });
        let both = LayerIssues { size_mismatch: true, missing_layer: true };
        assert_eq!(both.error_kind(MissingLayerStyle::HalfBlue), Some(ErrorKind::SizeMismatch));
        assert_eq!(
            LayerIssues { size_mismatch: false, missing_layer: true }.error_kind(MissingLayerStyle::Grey),
            Some(ErrorKind::MissingLayer(MissingLayerStyle::Grey))
        );
        assert_eq!(LayerIssues::default().error_kind(MissingLayerStyle::HalfBlue), None);
    }

    #[test]
    fn a_layer_of_another_depth_is_a_configuration_error() {
        let params = ParamCurves::new(Vec::new());
        let layers = LayerSet::new();
        let clip = grey_clip(1, 2, 2);
        let env = Env { params: &params, input: &clip, layers: &layers };
        let s = settings(2, 2, Fmt::U16);
        let mock = Mock::speed(100.0);
        let map = RemapTimeMap::new(&mock, s.rates, 0);
        assert!(matches!(EnvHost::new(&env, &s, &map, LayerIds::default()), Err(FxError::Mismatch(_))));
    }
}
