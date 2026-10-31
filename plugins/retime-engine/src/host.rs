//! Engine settings and the host interface of the engine core.
//!
//! [`EngineSettings`] is everything the engine derives from the parameters and the time context
//! before it calls the planner (pre-render) or the core (render). [`EngineHost`] is the engine's
//! way to get source frames: it turns an integer frame position into layer time with the time
//! map, fetches the frame and premultiplies it, for the retimed layer, the tracking layer and the
//! FG matte layers. [`EnvHost`] is the ready implementation over `fx_core::Env`.

use crate::image::{ErrorKind, Fmt, Image, MissingLayerStyle};
use crate::planner::{LAYER_ALT_A, LAYER_ALT_B};
use crate::stages::StageSet;
use crate::timeline::{Rates, TimeMap};
use fx_core::{DeviceRequest, Env, FieldOrder, Frame, FxError, Layer, Result};
use std::cell::Cell;

/// Everything the engine reads from the parameters and the time context for one output frame.
/// Built by `frontend::read_settings`. Popup indices are stored as the engine uses them (popup
/// value minus one).
#[derive(Clone, Debug, PartialEq)]
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
    /// The engine's wrap-around (360) flag: set from retime-fgbg's `Track RGB+A` checkbox,
    /// as the wrap flag only (never alpha in the luma; the `360 Video` checkbox is `video_360`, a
    /// different control). It reaches `FlowSettings::wrap`, the `is360` argument of the inverse
    /// warps, Smart Blend and the hole fill, and `warp::blur_compensation_wrap` (the
    /// vector path's module documentation in `compose` lists the sites). Only retime-fgbg sets it;
    /// off, every stage is the plain one, bit for bit.
    pub wrap_360: bool,
    /// The vector path (the solves, the warps and the composition of `compose::vector`) runs on the
    /// frames with their rows in reverse order, bottom row first, and its output is reversed back.
    /// MEASURED for plugin 01. Off, every stage
    /// sees the frames top row first. The other paths are per pixel and do not depend on it.
    pub motion_rows_reversed: bool,
    /// The solver's luma adds alpha (`FlowSettings::track_alpha`, `prep::luma`): plugin 03's
    /// `Track RGB+A`. MEASURED for plugin 03.
    /// Off for the other plugins; off, the luma is the plain one, bit for bit.
    pub track_alpha: bool,
    /// The render device the host asked for (`TemporalContext::device`). Carried only; the stages
    /// a render runs on are [`EngineSettings::stages`], and no output depends on the value.
    pub device: DeviceRequest,
    /// The stages of the render (`stages::StageSet::from_context`): the CPU, or the run's device
    /// executor, which gives the CPU's bytes stage by stage or runs the CPU function.
    pub stages: StageSet,
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
/// format. An FG matte layer is fetched exactly like a source frame (the matte goes through
/// the same input preparation before its channel is taken).
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

    /// The frame of FG matte layer `k` (0 is FG1, as the engine numbers the layers) shown at `time`
    /// ticks of layer time, with the same premultiply, 16 bpc range and error flags as
    /// [`EngineHost::main_frame_at`]. The default is a host without matte layers: an error for
    /// every `k`.
    fn matte_frame_at(&self, k: usize, _time: i32) -> Result<Image> {
        Err(FxError::Mismatch(format!("there is no FG matte layer {k} (0 is FG1) to fetch from: this host has no matte layers")))
    }

    /// Frame number `position` of FG matte layer `k`, at `time_map().frame_time(position as f32)`
    /// like the other layers (the mattes are fetched at the layer times of the main frames
    /// of the pair).
    fn matte_frame(&self, k: usize, position: i32) -> Result<Image> {
        self.matte_frame_at(k, self.time_map().frame_time(position as f32))
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
/// is checked again. FG matte layers are added with [`EnvHost::with_mattes`], which probes them the
/// same way.
pub struct EnvHost<'a> {
    settings: &'a EngineSettings,
    map: &'a dyn TimeMap,
    main: &'a dyn Layer,
    tracking: Option<&'a dyn Layer>,
    /// The FG matte layers of [`EnvHost::with_mattes`], FG1 first; empty without.
    mattes: Vec<&'a dyn Layer>,
    /// The FG matte frames are read straight ([`EnvHost::with_straight_mattes`]).
    straight_mattes: bool,
    size_mismatch: Cell<bool>,
    missing_layer: Cell<bool>,
}

/// Layers are not converted between depths: one of another depth than the output is an error.
fn check_depth(layer: &dyn Layer, settings: &EngineSettings) -> Result<()> {
    if layer.depth() != settings.fmt.depth() {
        return Err(FxError::Mismatch(format!(
            "a layer is {} bpc but the output is {} bpc",
            layer.depth().bits(),
            settings.fmt.depth().bits()
        )));
    }
    Ok(())
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
            check_depth(layer, settings)?;
        }
        let host = EnvHost {
            settings,
            map,
            main,
            tracking,
            mattes: Vec::new(),
            straight_mattes: false,
            size_mismatch: Cell::new(false),
            missing_layer: Cell::new(false),
        };
        host.check_size(main.width(), main.height());
        if let Some(t) = tracking {
            host.check_size(t.width(), t.height());
        }
        Ok(host)
    }

    /// The host with the FG matte layers `ids` of `env`, FG1 first: matte `k` of
    /// [`EngineHost::matte_frame`] is the layer `ids[k]`. The list replaces the one of an earlier
    /// call (the flags are sticky: what its probes raised stays raised). The caller passes the
    /// layers its matte probe counted (the consecutive layers with pixels from FG1 on, so
    /// `ids.len()` is `nfg`); an empty list leaves the host without mattes.
    ///
    /// The probes are those of [`EnvHost::new`] for every listed layer, made after the checks: a
    /// matte of another size than the output raises the size mismatch flag (the red fill, as the
    /// FG matte probes of the pre-render do), and a matte of size 0 the missing layer flag (U: the
    /// probe loop stops at an empty layer, so the original never fetches one; the rule of a
    /// source frame is applied). Every fetched matte frame is checked again and prepared like a
    /// source frame. An error, with no flag raised, when `env` lacks one of the layers or
    /// one has another bit depth than the output.
    pub fn with_mattes(mut self, env: &'a Env<'_>, ids: &[&'static str]) -> Result<EnvHost<'a>> {
        let mut mattes: Vec<&'a dyn Layer> = Vec::with_capacity(ids.len());
        for &id in ids {
            let layer: &'a dyn Layer = env.layer(id).ok_or_else(|| {
                FxError::Mismatch(format!("the matte layer `{id}` was asked for but the environment has none"))
            })?;
            check_depth(layer, self.settings)?;
            mattes.push(layer);
        }
        for layer in &mattes {
            self.check_size(layer.width(), layer.height());
        }
        self.mattes = mattes;
        Ok(self)
    }

    /// The host whose FG matte frames are read STRAIGHT ([`Image::from_frame_straight`]: the channels as stored, not
    /// premultiplied by the matte layer's own alpha); the retimed and tracking layers are still prepared. Vectors In
    /// replaces the alpha of an FG layer's plate by the layer's raw matte, and on an exact frame of a single group
    /// display its output alpha is the STORED matte channel (reference `fg1-display3.16x12.d8` frame 0: matte pixel R 15
    /// under alpha 87 gives alpha 15, not 15 x 87 / 255 = 5). Set by the vectors-in
    /// front-end only.
    pub fn with_straight_mattes(mut self) -> EnvHost<'a> {
        self.straight_mattes = true;
        self
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
        self.fetch_with(layer, time, Image::from_frame)
    }

    /// [`Self::fetch`] with the conversion `convert` of the wire frame.
    fn fetch_with(&self, layer: &dyn Layer, time: i32, convert: fn(&Frame) -> Image) -> Result<Image> {
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
        Ok(convert(&frame))
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

    fn matte_frame_at(&self, k: usize, time: i32) -> Result<Image> {
        match self.mattes.get(k) {
            Some(&layer) if self.straight_mattes => self.fetch_with(layer, time, Image::from_frame_straight),
            Some(&layer) => self.fetch(layer, time),
            None => Err(FxError::Mismatch(format!(
                "there is no FG matte layer {k} (0 is FG1) to fetch from: the host has {} matte layers",
                self.mattes.len()
            ))),
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
            wrap_360: false,
            motion_rows_reversed: false,
            track_alpha: false,
            device: DeviceRequest::Cpu,
            stages: StageSet::cpu(),
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

    /// Matte layer ids for the tests; the host takes whatever ids it is given.
    const FG1: &str = "fg1_matte";
    const FG2: &str = "fg2_matte";

    /// A clip of `n` frames of `w x h` at `depth`. Pixel 0 of every frame is opaque white; the others
    /// differ per frame and pixel and have partial alpha, so the premultiply changes them.
    fn matte_clip(depth: Depth, n: usize, w: usize, h: usize, salt: usize) -> MemLayer {
        let frames = (0..n)
            .map(|i| {
                let mut f = Frame::new(w, h, depth);
                match &mut f.pixels {
                    Pixels::U8(px) => {
                        for (p, v) in px.iter_mut().enumerate() {
                            *v = if p == 0 { [255; 4] } else { [(40 * i + 3 * p + salt) as u8, 200, 17, (60 + 50 * p) as u8] };
                        }
                    }
                    Pixels::U16(px) => {
                        for (p, v) in px.iter_mut().enumerate() {
                            *v = if p == 0 { [65535; 4] } else { [(9000 * i + 700 * p + salt) as u16, 65535, 4097, (20000 + 15000 * p) as u16] };
                        }
                    }
                    Pixels::F32(px) => {
                        for (p, v) in px.iter_mut().enumerate() {
                            *v = if p == 0 { [1.0; 4] } else { [0.125 * (i + 1 + salt) as f32, 0.75, 0.3, 0.25 * p as f32] };
                        }
                    }
                }
                f
            })
            .collect();
        MemLayer::new(base(), frames).unwrap()
    }

    fn samples(img: &Image) -> Vec<f64> {
        match &img.samples {
            crate::image::Samples::U8(v) => v.iter().map(|&x| f64::from(x)).collect(),
            crate::image::Samples::U16(v) => v.iter().map(|&x| f64::from(x)).collect(),
            crate::image::Samples::F32(v) => v.iter().map(|&x| f64::from(x)).collect(),
        }
    }

    #[test]
    fn a_matte_layer_is_fetched_exactly_like_a_source_frame_at_every_depth() {
        for depth in [Depth::U8, Depth::U16, Depth::F32] {
            let fmt = Fmt::from_depth(depth);
            // the input and the matte hold the same pictures, so the two fetches must agree
            let input = matte_clip(depth, 3, 2, 2, 0);
            let mut layers = LayerSet::new();
            layers.insert(FG1, Arc::new(matte_clip(depth, 3, 2, 2, 0)));
            let params = ParamCurves::new(Vec::new());
            let env = Env { params: &params, input: &input, layers: &layers };
            let s = settings(2, 2, fmt);
            let mock = Mock::speed(100.0);
            let map = RemapTimeMap::new(&mock, s.rates, 0);
            let host = EnvHost::new(&env, &s, &map, LayerIds::default()).unwrap().with_mattes(&env, &[FG1]).unwrap();
            for position in 0..3 {
                let matte = host.matte_frame(0, position).unwrap();
                assert_eq!(matte, host.main_frame(position).unwrap(), "{depth:?} position {position}");
                let wire = input.frame_at(i64::from(position) * 100).unwrap().unwrap();
                assert_eq!(matte, Image::from_frame(&wire), "{depth:?} position {position}");
                // opaque white is the engine white: 255, 65535 (the 16 bpc wire range is rescaled), 1.0
                assert_eq!(samples(&matte)[..4], [f64::from(fmt.white()); 4], "{depth:?}");
            }
            // frame_time(f) = f * 100 for equal rates, and a time inside frame 2 is frame 2
            assert_eq!(host.matte_frame_at(0, 250).unwrap(), host.matte_frame(0, 2).unwrap());
            for outside in [host.matte_frame(0, 3).unwrap(), host.matte_frame(0, -1).unwrap(), host.matte_frame_at(0, i32::MIN).unwrap()] {
                assert_eq!(outside, Image::new(2, 2, fmt), "{depth:?}: a time outside the clip is transparent");
            }
            assert_eq!(host.issues(), LayerIssues::default());
        }
    }

    #[test]
    fn the_premultiply_of_a_matte_frame_is_the_engine_one() {
        // 8 bpc: [R, G, B, A] = [0, 200, 17, 60] gives [A, R, G, B] = [60, 0, 47, 4]; 16 bpc: wire
        // alpha 32768 is 16384 in the range 0..=32768 and 32768 in the engine range
        let mut f8 = Frame::new(1, 1, Depth::U8);
        f8.pixels = Pixels::U8(vec![[0, 200, 17, 60]]);
        let mut f16 = Frame::new(1, 1, Depth::U16);
        f16.pixels = Pixels::U16(vec![[65535, 65535, 65535, 32768]]);
        let mut f32_ = Frame::new(1, 1, Depth::F32);
        f32_.pixels = Pixels::F32(vec![[0.5, 0.25, 1.0, 0.5]]);
        for (frame, want) in [(f8, vec![60.0, 0.0, 47.0, 4.0]), (f16, vec![32768.0; 4]), (f32_, vec![0.5, 0.25, 0.125, 0.5])] {
            let depth = frame.depth();
            let input = MemLayer::new(base(), vec![Frame::new(1, 1, depth)]).unwrap();
            let mut layers = LayerSet::new();
            layers.insert(FG1, Arc::new(MemLayer::new(base(), vec![frame]).unwrap()));
            let params = ParamCurves::new(Vec::new());
            let env = Env { params: &params, input: &input, layers: &layers };
            let s = settings(1, 1, Fmt::from_depth(depth));
            let mock = Mock::speed(100.0);
            let map = RemapTimeMap::new(&mock, s.rates, 0);
            let host = EnvHost::new(&env, &s, &map, LayerIds::default()).unwrap().with_mattes(&env, &[FG1]).unwrap();
            assert_eq!(samples(&host.matte_frame(0, 0).unwrap()), want, "{depth:?}");
        }
    }

    #[test]
    fn straight_mattes_keep_the_stored_channels() {
        // the frames of the premultiply test above, read straight: the channels as stored in the engine order A, R, G,
        // B (16 bpc in the engine range, scaled as the alpha is); the retimed layer is still prepared
        let mut f8 = Frame::new(1, 1, Depth::U8);
        f8.pixels = Pixels::U8(vec![[0, 200, 17, 60]]);
        let mut f16 = Frame::new(1, 1, Depth::U16);
        f16.pixels = Pixels::U16(vec![[65535, 0, 65535, 32768]]);
        let mut f32_ = Frame::new(1, 1, Depth::F32);
        f32_.pixels = Pixels::F32(vec![[0.5, 0.25, 1.0, 0.5]]);
        for (frame, want) in [(f8, vec![60.0, 0.0, 200.0, 17.0]), (f16, vec![32768.0, 65535.0, 0.0, 65535.0]), (f32_, vec![0.5, 0.5, 0.25, 1.0])] {
            let depth = frame.depth();
            let input = MemLayer::new(base(), vec![frame.clone()]).unwrap();
            let mut layers = LayerSet::new();
            layers.insert(FG1, Arc::new(MemLayer::new(base(), vec![frame.clone()]).unwrap()));
            let params = ParamCurves::new(Vec::new());
            let env = Env { params: &params, input: &input, layers: &layers };
            let s = settings(1, 1, Fmt::from_depth(depth));
            let mock = Mock::speed(100.0);
            let map = RemapTimeMap::new(&mock, s.rates, 0);
            let host = EnvHost::new(&env, &s, &map, LayerIds::default()).unwrap().with_mattes(&env, &[FG1]).unwrap().with_straight_mattes();
            assert_eq!(samples(&host.matte_frame(0, 0).unwrap()), want, "{depth:?}");
            assert_eq!(host.matte_frame(0, 0).unwrap(), Image::from_frame_straight(&frame), "{depth:?}");
            assert_eq!(host.main_frame(0).unwrap(), Image::from_frame(&frame), "{depth:?}: the retimed layer is prepared");
            // outside the clip still transparent
            assert_eq!(host.matte_frame(0, 5).unwrap(), Image::new(1, 1, Fmt::from_depth(depth)), "{depth:?}");
        }
    }

    #[test]
    fn the_matte_probes_raise_the_flags_of_the_other_layers() {
        let params = ParamCurves::new(Vec::new());
        let input = grey_clip(2, 2, 2);
        let s = settings(2, 2, Fmt::U8);
        let mock = Mock::speed(100.0);
        let map = RemapTimeMap::new(&mock, s.rates, 0);
        // a matte of another size: the size mismatch flag at the probe, and every fetch of it is
        // transparent; the main layer is still served
        let mut layers = LayerSet::new();
        layers.insert(FG1, Arc::new(grey_clip(2, 3, 2)));
        let env = Env { params: &params, input: &input, layers: &layers };
        let host = EnvHost::new(&env, &s, &map, LayerIds::default()).unwrap();
        assert_eq!(host.issues(), LayerIssues::default());
        let host = host.with_mattes(&env, &[FG1]).unwrap();
        assert_eq!(host.issues(), LayerIssues { size_mismatch: true, missing_layer: false });
        assert_eq!(host.matte_frame(0, 0).unwrap(), Image::new(2, 2, Fmt::U8));
        assert_eq!(first_red(&host.main_frame(1).unwrap()), 20);
        assert_eq!(host.issues().error_kind(MissingLayerStyle::HalfBlue), Some(ErrorKind::SizeMismatch));
        // a matte of size 0 is a missing layer, like a main layer of size 0
        let mut layers = LayerSet::new();
        layers.insert(FG1, Arc::new(MemLayer::new(base(), vec![Frame::new(0, 0, Depth::U8)]).unwrap()));
        let env = Env { params: &params, input: &input, layers: &layers };
        let host = EnvHost::new(&env, &s, &map, LayerIds::default()).unwrap().with_mattes(&env, &[FG1]).unwrap();
        assert_eq!(host.issues(), LayerIssues { size_mismatch: false, missing_layer: true });
        assert_eq!(host.matte_frame(0, 0).unwrap(), Image::new(2, 2, Fmt::U8));
        // both: red wins, as for the other layers
        let mut layers = LayerSet::new();
        layers.insert(FG1, Arc::new(MemLayer::new(base(), vec![Frame::new(0, 0, Depth::U8)]).unwrap()));
        layers.insert(FG2, Arc::new(grey_clip(1, 2, 3)));
        let env = Env { params: &params, input: &input, layers: &layers };
        let host = EnvHost::new(&env, &s, &map, LayerIds::default()).unwrap().with_mattes(&env, &[FG1, FG2]).unwrap();
        assert_eq!(host.issues(), LayerIssues { size_mismatch: true, missing_layer: true });
        assert_eq!(host.issues().error_kind(MissingLayerStyle::Grey), Some(ErrorKind::SizeMismatch));
    }

    #[test]
    fn missing_matte_layers_and_other_depths_are_errors() {
        let params = ParamCurves::new(Vec::new());
        let input = grey_clip(2, 2, 2);
        let s = settings(2, 2, Fmt::U8);
        let mock = Mock::speed(100.0);
        let map = RemapTimeMap::new(&mock, s.rates, 0);
        let mut layers = LayerSet::new();
        layers.insert(FG1, Arc::new(grey_clip(2, 2, 2)));
        let env = Env { params: &params, input: &input, layers: &layers };
        // the environment has no FG2
        let host = EnvHost::new(&env, &s, &map, LayerIds::default()).unwrap();
        assert!(matches!(host.with_mattes(&env, &[FG1, FG2]), Err(FxError::Mismatch(m)) if m.contains(FG2)));
        // a host without mattes, and an index past the list
        let host = EnvHost::new(&env, &s, &map, LayerIds::default()).unwrap();
        assert!(matches!(host.matte_frame(0, 0), Err(FxError::Mismatch(_))));
        let host = host.with_mattes(&env, &[FG1]).unwrap();
        assert!(host.matte_frame(0, 0).is_ok());
        assert!(matches!(host.matte_frame(1, 0), Err(FxError::Mismatch(m)) if m.contains("1 matte layers")));
        // a matte of another depth than the output
        let mut layers = LayerSet::new();
        layers.insert(FG1, Arc::new(MemLayer::new(base(), vec![Frame::new(2, 2, Depth::U16)]).unwrap()));
        let env = Env { params: &params, input: &input, layers: &layers };
        let host = EnvHost::new(&env, &s, &map, LayerIds::default()).unwrap();
        assert!(matches!(host.with_mattes(&env, &[FG1]), Err(FxError::Mismatch(m)) if m.contains("16 bpc")));
    }

    #[test]
    fn a_host_without_the_matte_method_has_no_mattes() {
        struct Bare<'m>(RemapTimeMap<&'m Mock>);
        impl EngineHost for Bare<'_> {
            fn time_map(&self) -> &dyn TimeMap {
                &self.0
            }
            fn main_frame_at(&self, _time: i32) -> Result<Image> {
                Ok(Image::new(1, 1, Fmt::U8))
            }
            fn tracking_frame_at(&self, _time: i32) -> Result<Image> {
                Err(FxError::Mismatch("none".into()))
            }
            fn issues(&self) -> LayerIssues {
                LayerIssues::default()
            }
        }
        let mock = Mock::speed(100.0);
        let host = Bare(RemapTimeMap::new(&mock, settings(1, 1, Fmt::U8).rates, 0));
        assert!(matches!(host.matte_frame(0, 0), Err(FxError::Mismatch(_))));
        assert!(matches!(host.matte_frame_at(2, 100), Err(FxError::Mismatch(_))));
    }

    #[test]
    fn with_mattes_replaces_the_list_and_keeps_the_other_layers() {
        let params = ParamCurves::new(Vec::new());
        let input = grey_clip(2, 2, 2);
        let mut layers = LayerSet::new();
        layers.insert(FG1, Arc::new(matte_clip(Depth::U8, 2, 2, 2, 1)));
        layers.insert(FG2, Arc::new(matte_clip(Depth::U8, 2, 2, 2, 7)));
        let env = Env { params: &params, input: &input, layers: &layers };
        let s = settings(2, 2, Fmt::U8);
        let mock = Mock::speed(100.0);
        let map = RemapTimeMap::new(&mock, s.rates, 0);
        let fg = |id: &str, f: i64| Image::from_frame(&env.layer(id).unwrap().frame_at(f * 100).unwrap().unwrap());
        let host = EnvHost::new(&env, &s, &map, LayerIds::default()).unwrap().with_mattes(&env, &[FG1, FG2]).unwrap();
        assert_eq!(host.matte_frame(0, 1).unwrap(), fg(FG1, 1));
        assert_eq!(host.matte_frame(1, 1).unwrap(), fg(FG2, 1));
        assert_ne!(fg(FG1, 1), fg(FG2, 1));
        assert_eq!(first_red(&host.main_frame(1).unwrap()), 20);
        let host = host.with_mattes(&env, &[FG2]).unwrap();
        assert_eq!(host.matte_frame(0, 0).unwrap(), fg(FG2, 0));
        assert!(host.matte_frame(1, 0).is_err());
        let host = host.with_mattes(&env, &[]).unwrap();
        assert!(host.matte_frame(0, 0).is_err());
        assert_eq!(host.issues(), LayerIssues::default());
    }

    #[test]
    fn the_new_settings_fields_default_to_off_and_the_cpu() {
        let s = settings(2, 2, Fmt::U8);
        assert!(!s.wrap_360);
        assert_eq!(s.device, DeviceRequest::Cpu);
    }

    /// Output frame 1 of a 4x4 three frame clip, Best and Blend, at `speed` percent.
    fn render_at(speed: f64, wrap: bool, device: DeviceRequest) -> Result<crate::compose::Rendered> {
        let clip = grey_clip(3, 4, 4);
        let params = ParamCurves::new(Vec::new());
        let layers = LayerSet::new();
        let env = Env { params: &params, input: &clip, layers: &layers };
        let mut s = settings(4, 4, Fmt::U8);
        s.vectors_quality = 4;
        s.current_time = 100;
        s.wrap_360 = wrap;
        s.device = device;
        let mock = Mock::speed(speed);
        let map = RemapTimeMap::new(&mock, s.rates, s.current_time);
        let host = EnvHost::new(&env, &s, &map, LayerIds::default())?;
        crate::compose::render(&s, &host)
    }

    #[test]
    fn the_wrap_flag_renders_where_motion_is_searched_and_the_device_changes_nothing() {
        // half speed puts output frame 1 between two source frames: the vector path, which serves
        // the wrap flag (solver, warps, hole fill and blur wrap). The frames of the grey clip are
        // flat, so the flow is zero and a wrapped warp samples the pixel it warps: the same bytes
        let plain = render_at(50.0, false, DeviceRequest::Cpu).unwrap();
        let wrapped = render_at(50.0, true, DeviceRequest::Cpu).unwrap();
        assert_eq!(wrapped, plain);
        // full speed lands on a frame: the fast path has no motion to wrap
        assert_eq!(render_at(100.0, true, DeviceRequest::Cpu).unwrap(), render_at(100.0, false, DeviceRequest::Cpu).unwrap());
        // the device is carried, never read
        for device in [DeviceRequest::Gpu, DeviceRequest::GpuIndex(1)] {
            assert_eq!(render_at(50.0, false, device).unwrap(), plain, "{device}");
        }
    }
}
