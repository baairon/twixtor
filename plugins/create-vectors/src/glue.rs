//! Plugin 02's temporal front-end: settings, error fills, the Source display with the Draw Geom
//! overlay, and the Create Vectors pass on the shared engine (`retime_engine::create_vectors`)
//! with its Vectors and Debug displays.
//!
//! What one render does:
//!
//! * the FG matte layers with pixels (`nfg`, FG1 first, stopping at the first empty one) are
//!   fetched at the layer times of the pair and their `fgk_matte_chan` channel is taken
//!   (`matte::extract`); every layer (FG1 to FGn, then Main_BG) gets its own flow with its
//!   sensitivity, Track and tracking matte, and the layers of `layers_output` are merged into the
//!   vector image (`create_vectors::create_vectors_job`, rule G1d);
//! * a `Pt k: Use` point in use at the layer times of both frames (not Don't Use and the same
//!   value at both; with `nfg == 0` every value 2 to 5 belongs to the only layer) becomes user
//!   displacement of its layer (rule G2), or drives the second tracker of a layer whose Track is
//!   off (rule G5); masks (`First Mask`, `Last Mask`) feed the same slot but are parameter
//!   kinds `fx-core` cannot express, so they do not exist for this crate;
//! * `Display == Debug` shows the debug view of `Debug View Method` (rule G4) of the pair's
//!   flow (with FG layers the merged flow);
//! * the Source display and the Debug display draw the points in use at the frame time of the
//!   layers `Draw Geom` names over the image (rule G3).
//!
//! Tiny frames (U, ours, G7 family): a frame with a side of 1 has no motion search, so the
//! Vectors and the Debug displays are the zero vector image whatever the FG layers, points,
//! Track, Layers Output and Debug settings; the Source display is the plain copy with its overlay.
//!
//! The single layer case (no FG matte, no point in use, `Main_BG: Track` checked, Vectors
//! display) is today's `create_vectors::create_vectors` call, byte for byte.

use crate::prerender::{PreRender02, LAYER_ALT_SRC, LAYER_FG_MATTE};
use fx_core::{Checkout, Depth, Env, Frame, FxError, Layer, ParamCurves, Quality, Result, TemporalContext};
use retime_engine::create_vectors::{
    self, frame_index, frame_pair, frame_time, output_group, quality_code, CreateVectorsJob, CreateVectorsSettings, LayerParams,
};
use retime_engine::frontend::{fixed_reader, ticks_i32};
use retime_engine::host::{layer_is_set, LayerIssues};
use retime_engine::image::{error_frame, Fmt, Image, MissingLayerStyle};
use retime_engine::matte::{self, Matte};
use retime_engine::points::{point_layer, read_pos, UserPoint};
use retime_engine::stages::StageSet;
use retime_engine::{debug, geom, prep, vectors};
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
    /// `Layers Output`: 1 All, 2 FG1, 3 FG2, 4 FG3, 5 MAIN_BG.
    pub layers_output: i32,
    /// The controls of FG1 to FG3 (whether the layer exists is the matte probe's): `fgk_matte_chan`,
    /// `fgk_sensitivity` through the 16.16 reader divided by `100.0f`, `fgk_inv_matte_shrink`
    /// through the 16.16 reader times the downsample ratio (pixels) and `fgk_track`.
    pub fg: [LayerParams; 3],
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
            // the max-displace scale: ((float)downsample_x.num / (float)downsample_x.den) * Max Displace
            max_displace: (DOWNSAMPLE.0 as f32 / DOWNSAMPLE.1 as f32) * self.max_displace,
        }
    }

    /// The `nfg + 1` layers of a render with `nfg` FG matte layers: FG1 to FGn, then Main_BG
    /// (its sensitivity and Track; no matte channel and no shrink).
    pub fn layers(&self, nfg: usize) -> Vec<LayerParams> {
        let mut out: Vec<LayerParams> = self.fg[..nfg.min(3)].to_vec();
        out.push(LayerParams { matte_chan: 0, sensitivity: self.main_bg_sensitivity, shrink: 0.0, track: self.main_bg_track });
        out
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
        layers_output: c.int_at("layers_output", t) as i32,
        fg: [0, 1, 2].map(|k| LayerParams {
            matte_chan: c.int_at(FG_IDS[k][0], t) as i32,
            sensitivity: fixed_reader(c.fixed_at(FG_IDS[k][1], t), 1.0f32) / 100.0f32,
            shrink: (DOWNSAMPLE.0 as f32 / DOWNSAMPLE.1 as f32) * fixed_reader(c.fixed_at(FG_IDS[k][2], t), 1.0f32),
            track: c.bool_at(FG_IDS[k][3], t),
        }),
    })
}

/// The ids of FG1 to FG3's controls: matte channel, sensitivity, inverse matte shrink, track.
const FG_IDS: [[&str; 4]; 3] = [
    ["fg1_matte_chan", "fg1_sensitivity", "fg1_inv_matte_shrink", "fg1_track"],
    ["fg2_matte_chan", "fg2_sensitivity", "fg2_inv_matte_shrink", "fg2_track"],
    ["fg3_matte_chan", "fg3_sensitivity", "fg3_inv_matte_shrink", "fg3_track"],
];

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

const PT_POS_IDS: [&str; 12] = [
    "pt1_pos", "pt2_pos", "pt3_pos", "pt4_pos", "pt5_pos", "pt6_pos", "pt7_pos", "pt8_pos", "pt9_pos", "pt10_pos",
    "pt11_pos", "pt12_pos",
];

const PT_COLOR_IDS: [&str; 12] = [
    "pt1_color", "pt2_color", "pt3_color", "pt4_color", "pt5_color", "pt6_color", "pt7_color", "pt8_color", "pt9_color",
    "pt10_color", "pt11_color", "pt12_color",
];

/// The tracking points of every layer of a pair: `nfg + 1` lists, FG1 first and Main_BG
/// last. A point is in use when `Pt k: Use` is not Don't Use (1) at `time_a` and has the same
/// value at `time_b`; it goes to the layer `points::point_layer` names (with `nfg == 0` every
/// value 2 to 5 is the only layer), in point order, with its position read at both times
/// (`points::read_pos`, the 16.16 reader).
pub fn layer_points(curves: &ParamCurves, time_a: i32, time_b: i32, nfg: usize) -> Vec<Vec<UserPoint>> {
    let mut out = vec![Vec::new(); nfg + 1];
    for (i, id) in PT_USE_IDS.iter().enumerate() {
        let v1 = curves.int_at(id, i64::from(time_a));
        if v1 == 1 || curves.int_at(id, i64::from(time_b)) != v1 {
            continue;
        }
        if let Some(k) = point_layer(v1, nfg) {
            out[k].push(UserPoint { at_a: read_pos(curves, PT_POS_IDS[i], time_a), at_b: read_pos(curves, PT_POS_IDS[i], time_b) });
        }
    }
    out
}

/// The Draw Geom overlay (rule G3) over `img`, the Source display after
/// the source copy or the Debug display after its view: the points in use at the frame time
/// (`frame_time` of the current frame number, the rounded frame time) whose layer is in the
/// group `Draw Geom` names ([`overlay_layers`]), at their position at that time and in their
/// `Pt k:` colour, through `geom::draw_points` with the downsample ratio (1 here). Draws nothing
/// for None or without such a point.
fn draw_geom(img: &mut Image, s: &Settings02, curves: &ParamCurves, layer_count: usize) {
    let Some((first, last)) = overlay_layers(s.draw_geom, layer_count) else {
        return;
    };
    let t = frame_time(frame_index(s.current_time, s.time_step) as i32, s.time_step);
    let used = points_in_use(curves, t, t, layer_count, first, last);
    if used.is_empty() {
        return;
    }
    let pts: Vec<([f32; 2], [u8; 3])> =
        used.iter().map(|&n| (read_pos(curves, PT_POS_IDS[n - 1], t), curves.color_at(PT_COLOR_IDS[n - 1], i64::from(t)))).collect();
    geom::draw_points(img, &pts, DOWNSAMPLE.0 as f32 / DOWNSAMPLE.1 as f32);
}

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
        self.fetch_from(self.layer, time)
    }

    /// The same for another layer of the render, an FG matte (a matte goes through the same
    /// input preparation as the source), with the same flags.
    fn fetch_from(&self, layer: &dyn Layer, time: i32) -> Result<Image> {
        let blank = || Image::new(self.settings.width, self.settings.height, self.settings.fmt);
        let Some(frame) = layer.frame_at(i64::from(time))? else {
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
    *output = render(&s, env, &StageSet::from_context(cx))?;
    Ok(())
}

/// The render of one frame: error flags first (they are tested before both branches), then the
/// Source display with its overlay, else the fetches of the pair and the mattes, the tiny frame
/// rule, and the Debug view or the Create Vectors pass. The solves run on `stages` (the CPU unless
/// the host passed a device executor).
fn render(s: &Settings02, env: &Env<'_>, stages: &StageSet) -> Result<Frame> {
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
        // the overlay runs after the copy of the source and draws only geometry at the frame time
        let mut img = src.fetch(s.current_time)?;
        if let Some(f) = error_image(src.issues(), s) {
            return Ok(f);
        }
        draw_geom(&mut img, s, curves, layer_count);
        return Ok(img.to_frame());
    }
    let frame = frame_index(s.current_time, s.time_step);
    let (a, b) = frame_pair(frame, s.track_frame, s.debug());
    let (ta, tb) = (frame_time(a, s.time_step), frame_time(b, s.time_step));
    let x = src.fetch(ta)?;
    let y = src.fetch(tb)?;
    // the mattes at the layer times of the pair, FG1 first
    let mut mattes_x: Vec<Matte> = Vec::with_capacity(fg_count);
    let mut mattes_y: Vec<Matte> = Vec::with_capacity(fg_count);
    for (k, id) in LAYER_FG_MATTE.iter().enumerate().take(fg_count) {
        let Some(m) = env.layer(id) else {
            return Err(FxError::Mismatch(format!("the matte layer `{id}` was probed but the environment has none")));
        };
        let chan = s.fg[k].matte_chan;
        mattes_x.push(matte::extract(&src.fetch_from(m, ta)?, chan));
        mattes_y.push(matte::extract(&src.fetch_from(m, tb)?, chan));
    }
    if let Some(f) = error_image(src.issues(), s) {
        return Ok(f);
    }
    // U, ours (G7 family): a side of 1 has no search, Vectors and Debug show the zero vector
    if !prep::searchable(s.width, s.height) {
        return Ok(vectors::zero_vectors_image(s.width, s.height, s.fmt).to_frame());
    }
    let (first, last) = output_group(s.layers_output, fg_count);
    let job = CreateVectorsJob {
        settings: s.vector_settings(),
        layers: s.layers(fg_count),
        mattes_x,
        mattes_y,
        points: layer_points(curves, ta, tb, fg_count),
        first,
        last,
    };
    if s.debug() {
        let flow = create_vectors::job_flow_on(stages.get(), &x, &y, &job)?;
        let mut img = debug::debug_view_on(stages.get(), &x, &y, &flow, s.debug_view_code())?;
        draw_geom(&mut img, s, curves, layer_count);
        return Ok(img.to_frame());
    }
    let encoded = create_vectors::create_vectors_job_on(stages.get(), &x, &y, &job)?;
    Ok(encoded.to_frame())
}
