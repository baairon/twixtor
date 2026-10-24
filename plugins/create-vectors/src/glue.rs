//! Plugin 02's temporal front-end: settings, error fills, the Source display, the Create Vectors
//! pass on the shared engine (`retime_engine::create_vectors`) and the checks that turn every
//! unsupported feature into an explicit `FxError::Unsupported`.
//!
//! Only the single layer vector path is rendered. Not supported:
//!
//! * a FG Matte layer with pixels (`nfg > 0`): the engine layer count becomes `nfg + 1`, every
//!   layer gets its own flow and the per layer results are merged;
//! * `Display == Debug`: the debug views (Warp and four compositions of the two frames), none of
//!   which exists in the engine;
//! * `Main_BG: Track` unchecked: the layer then uses a displacement interpolator driven by user
//!   geometry instead of the optical flow;
//! * user geometry reaching the solver: a `Pt k: Use` point that is in use at the layer times of
//!   both frames (with `nfg == 0` every value 2 to 5 belongs to the only layer) becomes user
//!   displacement; masks (`First Mask`, `Last Mask`) feed the same slot but are parameter kinds
//!   `fx-core` cannot express, so they cannot be detected and do not exist for this crate;
//! * the Source display with `Draw Geom` not None and geometry at the frame time: the overlay.

use crate::prerender::{PreRender02, LAYER_ALT_SRC, LAYER_FG_MATTE};
use fx_core::{Checkout, Depth, Env, Frame, FxError, Layer, ParamCurves, Quality, Result, TemporalContext};
use retime_engine::create_vectors::{self, frame_index, frame_pair, frame_time, quality_code, CreateVectorsSettings};
use retime_engine::frontend::{fixed_reader, ticks_i32};
use retime_engine::host::{layer_is_set, LayerIssues};
use retime_engine::image::{error_frame, Fmt, Image, MissingLayerStyle};
use std::cell::Cell;

/// The fill of the missing layer error of plugin 02: grey.
pub const MISSING_LAYER_STYLE: MissingLayerStyle = MissingLayerStyle::Grey;

/// The downsample ratio as a fraction (`Max Displace` is scaled by `num / den`, the shrink of the
/// FG layers by the y fraction). Renders here are always at full resolution.
pub const DOWNSAMPLE: (u32, u32) = (1, 1);

/// Everything one render reads from the parameters and the context, all read at `current_time`.
/// Popups are the raw values (1 based).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings02 {
    /// Current time in ticks since the start of the layer.
    pub current_time: i32,
    /// Ticks per frame.
    pub time_step: i32,
    /// Ticks per second, carried along only.
    pub time_scale: i32,
    /// Output size and pixel format.
    pub width: usize,
    pub height: usize,
    pub fmt: Fmt,
    /// Full quality render (`Quality::High`).
    pub hiq: bool,
    /// `Display`: 1 Source, 2 Vectors, 3 Debug.
    pub display: i32,
    /// `Vector Resolution`: 1 Normal (half res), 2 Full res.
    pub vector_resolution: i32,
    /// `Debug View Method`: 1 Warp, 2 Blend, 3 Diff, 4 Detail Compare, 5 Edge Compare.
    pub debug_view_method: i32,
    /// `Draw Geom`: 1 None, 2 All, 3 FG1, 4 FG2, 5 FG3, 6 Main_BG.
    pub draw_geom: i32,
    /// `Max Displace` through the 16.16 reader, before the downsample ratio.
    pub max_displace: f32,
    /// `Track Frame`: 1 Previous, 2 Next.
    pub track_frame: i32,
    /// `Prep`: 1 None, 2 Contrast/Edge Enhance, 3 Delinearize.
    pub prep: i32,
    /// `Track RGB+A`.
    pub track_rgb_a: bool,
    /// `Main_BG: Sensitivity` through the 16.16 reader, divided by `100.0f`.
    pub main_bg_sensitivity: f32,
    /// `Main_BG: Track`.
    pub main_bg_track: bool,
}

impl Settings02 {
    /// `Display == Source`.
    pub fn display_source(&self) -> bool {
        self.display == 1
    }

    /// `Display == Debug`.
    pub fn debug(&self) -> bool {
        self.display == 3
    }

    /// The engine view code of `Debug View Method`: Warp 4, Blend 0, Diff 1, Detail Compare 2, Edge
    /// Compare 3.
    pub fn debug_view_code(&self) -> i32 {
        match self.debug_view_method {
            1 => 4,
            2 => 0,
            3 => 1,
            4 => 2,
            _ => 3,
        }
    }

    /// The arguments of the engine call that reach the solver and the encoder.
    pub fn vector_settings(&self) -> CreateVectorsSettings {
        CreateVectorsSettings {
            quality: quality_code(self.vector_resolution),
            hiq: self.hiq,
            sensitivity: self.main_bg_sensitivity,
            prep_contrast_edge: self.prep == 2,
            prep_delinearize: self.prep == 3,
            track_rgb_a: self.track_rgb_a,
            // a4 = ((float)downsample_x.num / (float)downsample_x.den) * Max Displace
            max_displace: (DOWNSAMPLE.0 as f32 / DOWNSAMPLE.1 as f32) * self.max_displace,
        }
    }

    /// The pre-render inputs for `fg_count` active FG matte layers and whether `Alt Src` has pixels.
    pub fn pre_render(&self, alt_present: bool, fg_count: usize) -> PreRender02 {
        PreRender02 {
            current_time: self.current_time,
            time_step: self.time_step,
            display_source: self.display_source(),
            debug: self.debug(),
            track_frame: self.track_frame,
            alt_present,
            fg_count,
        }
    }
}

/// Reads the settings of one output frame at `cx.current_time`. Errors: a time that does not fit 32
/// bit ticks.
pub fn read_settings(cx: &TemporalContext, env: &Env<'_>) -> Result<Settings02> {
    let c: &ParamCurves = env.params;
    let t = cx.current_time;
    Ok(Settings02 {
        current_time: ticks_i32(cx.current_time, "current_time")?,
        time_step: ticks_i32(cx.time_step(), "time_step")?,
        time_scale: ticks_i32(cx.time_scale(), "time_scale")?,
        width: cx.width,
        height: cx.height,
        fmt: Fmt::from_depth(cx.depth),
        hiq: cx.quality == Quality::High,
        display: c.int_at("display", t) as i32,
        vector_resolution: c.int_at("vector_resolution", t) as i32,
        debug_view_method: c.int_at("debug_view_method", t) as i32,
        draw_geom: c.int_at("draw_geom", t) as i32,
        max_displace: fixed_reader(c.fixed_at("max_displace", t), 1.0f32),
        track_frame: c.int_at("track_frame", t) as i32,
        prep: c.int_at("prep", t) as i32,
        track_rgb_a: c.bool_at("track_rgb_a", t),
        main_bg_sensitivity: fixed_reader(c.fixed_at("main_bg_sensitivity", t), 1.0f32) / 100.0f32,
        main_bg_track: c.bool_at("main_bg_track", t),
    })
}

/// The FG matte probes: the count of consecutive present layers starting at FG1 (the loop stops at
/// the first empty one) and whether one of them has another size than the output.
pub fn fg_probe(env: &Env<'_>, width: usize, height: usize) -> (usize, bool) {
    let mut count = 0;
    let mut mismatch = false;
    for id in LAYER_FG_MATTE {
        let layer = env.layer(id);
        if !layer_is_set(layer) {
            break;
        }
        count += 1;
        if let Some(l) = layer {
            mismatch |= (l.width(), l.height()) != (width, height);
        }
    }
    (count, mismatch)
}

/// Whether the track point `Pt k: Use` with popup value `v` belongs to `layer` of `layer_count`
/// layers: value 2 to 4 are FG1 to FG3, 5 is Main_BG, which is the last layer; a value whose layer
/// does not exist (`layer_count <= v - 2`) belongs to the last layer.
pub fn point_belongs(v: i64, layer: usize, layer_count: usize) -> bool {
    let target = v - 2;
    layer as i64 == target || (layer_count as i64 <= target && layer + 1 == layer_count)
}

/// The track points (1 based) that give geometry to one of the layers `first..=last` between two
/// layer times: `Pt k: Use` is not "Don't Use" (1) at the first time and has the same value at the
/// second, and the value maps to the layer.
pub fn points_in_use(
    curves: &ParamCurves,
    time1: i32,
    time2: i32,
    layer_count: usize,
    first: usize,
    last: usize,
) -> Vec<usize> {
    let mut out = Vec::new();
    for (i, id) in PT_USE_IDS.iter().enumerate() {
        let v1 = curves.int_at(id, i64::from(time1));
        if v1 == 1 || curves.int_at(id, i64::from(time2)) != v1 {
            continue;
        }
        if (first..=last).any(|l| point_belongs(v1, l, layer_count)) {
            out.push(i + 1);
        }
    }
    out
}

const PT_USE_IDS: [&str; 12] = [
    "pt1_use", "pt2_use", "pt3_use", "pt4_use", "pt5_use", "pt6_use", "pt7_use", "pt8_use", "pt9_use", "pt10_use",
    "pt11_use", "pt12_use",
];

/// The layers `(first, last)` the Draw Geom overlay draws: `None` draws nothing, `All` every layer,
/// FG1 to FG3 and Main_BG one layer (`value - 3`, clamped to the last layer).
pub fn overlay_layers(draw_geom: i32, layer_count: usize) -> Option<(usize, usize)> {
    match draw_geom {
        1 => None,
        2 => Some((0, layer_count - 1)),
        v => {
            let l = ((v - 3).max(0) as usize).min(layer_count - 1);
            Some((l, l))
        }
    }
}

/// The message of a FG Matte layer being set.
pub const FG_UNSUPPORTED: &str = "a FG Matte layer is set: the multi layer path (one flow per \
     layer, matte weighted merge of the vector images) is not implemented; \
     remove the FG matte layers (Create Vectors handles the single layer case only)";

/// The message of `Main_BG: Track` being unchecked.
pub const MAIN_BG_TRACK_UNSUPPORTED: &str = "Main_BG: Track is unchecked: the second tracker \
     (a displacement interpolator driven by user geometry, used instead of the optical flow) is not \
     implemented; check Main_BG: Track";

/// The message of user tracking points reaching the solver.
pub fn points_unsupported(points: &[usize]) -> String {
    let list: Vec<String> = points.iter().map(|n| format!("Pt {n}: Use")).collect();
    format!(
        "user tracking points feed the flow solver as user displacement, which is not \
         implemented; set {} to Don't Use",
        list.join(", ")
    )
}

/// The message of the Draw Geom overlay.
pub const DRAW_GEOM_UNSUPPORTED: &str = "Draw Geom would draw the geometry of the tracking points over the source view, \
     which is not implemented; set Draw Geom to None or Pt n: Use to Don't Use";

/// The message of a Debug view (`Display == Debug`).
pub fn debug_unsupported(view_popup: i32) -> String {
    let (name, what) = match view_popup {
        1 => ("Warp", "the forward warp of the first frame with the flow"),
        2 => ("Blend", "the blend view of the two frames"),
        3 => ("Diff", "the difference view of the two frames"),
        4 => ("Detail Compare", "the detail comparison view"),
        _ => ("Edge Compare", "the edge comparison view"),
    };
    format!(
        "Display Debug with Debug View Method \"{name}\" draws {what}, which is not implemented; \
         set Display to Vectors or Source"
    )
}

/// The layer the render reads and the two error flags (size mismatch, missing layer).
/// A fetched frame of another size raises the mismatch flag and comes back transparent.
struct Source<'a> {
    settings: &'a Settings02,
    layer: &'a dyn Layer,
    size_mismatch: Cell<bool>,
    missing_layer: Cell<bool>,
}

impl<'a> Source<'a> {
    /// `layer` is `Alt Src` when it has pixels, else the input layer. Its size is probed first:
    /// width or height 0 is a missing layer, another size a size mismatch.
    fn new(settings: &'a Settings02, layer: &'a dyn Layer) -> Result<Source<'a>> {
        if layer.depth() != settings.fmt.depth() {
            return Err(FxError::Mismatch(format!(
                "the source layer is {} bpc but the output is {} bpc",
                layer.depth().bits(),
                settings.fmt.depth().bits()
            )));
        }
        let src = Source { settings, layer, size_mismatch: Cell::new(false), missing_layer: Cell::new(false) };
        src.check_size(layer.width(), layer.height());
        Ok(src)
    }

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

    /// The frame of the source at layer time `time` (ticks), in the engine domain: premultiplied,
    /// 16 bpc in 0..=65535 (`Image::from_frame`). Outside the clip it is transparent; a frame of
    /// another size is transparent and sets the flag.
    fn fetch(&self, time: i32) -> Result<Image> {
        let blank = || Image::new(self.settings.width, self.settings.height, self.settings.fmt);
        let Some(frame) = self.layer.frame_at(i64::from(time))? else {
            return Ok(blank());
        };
        if frame.depth() != self.settings.fmt.depth() {
            return Err(FxError::Mismatch(format!(
                "a frame is {} bpc but the output is {} bpc",
                frame.depth().bits(),
                self.settings.fmt.depth().bits()
            )));
        }
        if !self.check_size(frame.width, frame.height) {
            return Ok(blank());
        }
        Ok(Image::from_frame(&frame))
    }

    fn issues(&self) -> LayerIssues {
        LayerIssues { size_mismatch: self.size_mismatch.get(), missing_layer: self.missing_layer.get() }
    }
}

/// The error image for the flags set so far (red before grey), a wire frame that skips the finish.
fn error_image(issues: LayerIssues, s: &Settings02) -> Option<Frame> {
    issues
        .error_kind(MISSING_LAYER_STYLE)
        .map(|kind| error_frame(kind, s.width, s.height, s.fmt.depth()))
}

fn require_output(cx: &TemporalContext, output: &Frame) -> Result<()> {
    if (output.width, output.height, output.depth()) != (cx.width, cx.height, cx.depth) {
        let bits = |d: Depth| d.bits();
        return Err(FxError::Mismatch(format!(
            "the output frame is {}x{} at {} bpc but the context says {}x{} at {} bpc",
            output.width,
            output.height,
            bits(output.depth()),
            cx.width,
            cx.height,
            bits(cx.depth)
        )));
    }
    Ok(())
}

/// `Effect::dependencies`: the checkout list (`prerender::checkouts`).
pub(crate) fn dependencies(cx: &TemporalContext, env: &Env<'_>) -> Result<Vec<Checkout>> {
    let s = read_settings(cx, env)?;
    let (fg_count, _) = fg_probe(env, s.width, s.height);
    let alt_present = layer_is_set(env.layer(LAYER_ALT_SRC));
    Ok(crate::prerender::checkouts(&s.pre_render(alt_present, fg_count)))
}

/// `Effect::render_temporal`: settings, error flags, the Source display or the Create Vectors pass, the finish.
pub(crate) fn render_temporal(cx: &TemporalContext, env: &Env<'_>, output: &mut Frame) -> Result<()> {
    require_output(cx, output)?;
    let s = read_settings(cx, env)?;
    *output = render(&s, env)?;
    Ok(())
}

/// The render of one frame: error flags first (they are tested before both branches), then the
/// Source display, then the refusals of unsupported features, then the Create Vectors pass.
fn render(s: &Settings02, env: &Env<'_>) -> Result<Frame> {
    let alt = env.layer(LAYER_ALT_SRC);
    let layer: &dyn Layer = match alt {
        Some(l) if layer_is_set(Some(l)) => l,
        _ => env.input,
    };
    let src = Source::new(s, layer)?;
    let (fg_count, fg_mismatch) = fg_probe(env, s.width, s.height);
    if fg_mismatch {
        src.size_mismatch.set(true);
    }
    if let Some(f) = error_image(src.issues(), s) {
        return Ok(f);
    }
    let curves = env.params;
    let layer_count = fg_count + 1;
    if s.display_source() {
        // 6.3: the overlay runs after the copy of the source and draws only geometry at the frame time.
        if let Some((first, last)) = overlay_layers(s.draw_geom, layer_count) {
            let t = frame_time(frame_index(s.current_time, s.time_step) as i32, s.time_step);
            if !points_in_use(curves, t, t, layer_count, first, last).is_empty() {
                return Err(FxError::Unsupported(DRAW_GEOM_UNSUPPORTED.into()));
            }
        }
        let img = src.fetch(s.current_time)?;
        if let Some(f) = error_image(src.issues(), s) {
            return Ok(f);
        }
        return Ok(img.to_frame());
    }
    if fg_count > 0 {
        return Err(FxError::Unsupported(FG_UNSUPPORTED.into()));
    }
    if s.debug() {
        return Err(FxError::Unsupported(debug_unsupported(s.debug_view_method)));
    }
    if !s.main_bg_track {
        return Err(FxError::Unsupported(MAIN_BG_TRACK_UNSUPPORTED.into()));
    }
    let frame = frame_index(s.current_time, s.time_step);
    let (a, b) = frame_pair(frame, s.track_frame, s.debug());
    let (ta, tb) = (frame_time(a, s.time_step), frame_time(b, s.time_step));
    let points = points_in_use(curves, ta, tb, layer_count, 0, layer_count - 1);
    if !points.is_empty() {
        return Err(FxError::Unsupported(points_unsupported(&points)));
    }
    let x = src.fetch(ta)?;
    let y = src.fetch(tb)?;
    if let Some(f) = error_image(src.issues(), s) {
        return Ok(f);
    }
    let encoded = create_vectors::create_vectors(&x, &y, &s.vector_settings())?;
    Ok(encoded.to_frame())
}
