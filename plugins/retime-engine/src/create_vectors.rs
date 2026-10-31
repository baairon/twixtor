//! Create Vectors pass: estimates the optical flow between two frames of the source layer and
//! writes it into the output as an encoded vector image ([`crate::vectors`]). This module is the
//! engine side of the effect:
//!
//! * [`frame_index`], [`frame_pair`], [`frame_time`]: the frame arithmetic;
//! * [`CreateVectorsSettings`] and [`create_vectors`]: solve the flow of one layer and encode it
//!   (the single layer case: no FG matte layer, no point in use, `Main_BG: Image Track` checked);
//! * [`CreateVectorsJob`], [`layer_flows`], [`job_flow`] and [`create_vectors_job`]: the layered
//!   case: one flow per layer of the Layers Output group, with the masks of
//!   the tracking mattes, the user displacement of the tracking points or the second tracker,
//!   merged by the matte weighted encoder (rule G1d). The single layer case of a job calls
//!   [`create_vectors`], so it keeps its bytes.
//!
//! Claim tags: V established, I inferred, U not established (a rule of this project, pinned in
//! the parity gaps register until a reference render grades it).
//!
//! Precision contract of the layered case: the contracts of the stages it calls (the solver, the
//! matte primitives, the second tracker, the merged encoder), with the layers in a fixed order
//! (FG1 first, Main_BG last) and no float summed across threads, so the output is the same for
//! any thread count.
//!
//! Resolution of the result: the solver returns its flow at the working resolution (`W >> R`, R = 1
//! for Normal (half res), 0 for Full res, one more halving at draft quality). It is not resampled
//! to the output size in a separate step. The encoder does it: for every pixel of the output image
//! it samples the flow bilinearly at `(x * (fw - 1) / (dw - 1), y * (fh - 1) / (dh - 1))` and
//! multiplies the displacement by `(dw - 1) / (fw - 1)` before dividing by `Max Displace`, so the
//! encoded image is always at the output size and in pixels of the output.
//! [`vectors::encode_vectors`] is exactly that.

use crate::flow::extras::{MaskPyramid, SolveExtras};
use crate::flow::{self, FlowField, FlowSettings};
use crate::image::Image;
pub use crate::layers::LayerParams;
use crate::matte::{self, Matte};
use crate::points::{self, UserPoint};
use crate::stages::{Cpu, Stages};
use crate::timeline::cvtt_i32_f32;
use crate::vectors;
use fx_core::Result;

/// Flow quality code for Vector Resolution "Normal (half res)" (the popup value 1).
pub const QUALITY_NORMAL: u8 = 4;
/// Vector Resolution "Full res" as the flow quality code (the popup value 2).
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
    /// The flow quality code, [`QUALITY_NORMAL`] or [`QUALITY_FULL`].
    pub quality: u8,
    /// High quality render (not draft); draft quality halves the working resolution once more.
    pub hiq: bool,
    /// `Main_BG: Sensitivity` already divided by 100.
    pub sensitivity: f32,
    /// The prep edge enhance flag: `Prep == Contrast/Edge Enhance`.
    pub prep_contrast_edge: bool,
    /// The prep delinearize flag: `Prep == Delinearize`.
    pub prep_delinearize: bool,
    /// `Track RGB+A`, the `rgba` flag of the luma conversion (alpha added to the luma).
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
            wrap: false,
        }
    }
}

/// The flow of the pass at the solver's working resolution: `X(p) ~ Y(p + flow(p))`. `x` and `y`
/// are the premultiplied engine images of frames `a` and `b`, same size and format.
pub fn solve_flow(x: &Image, y: &Image, settings: &CreateVectorsSettings) -> Result<FlowField> {
    solve_flow_on(&Cpu, x, y, settings)
}

/// [`solve_flow`] on `stages` (`stages::Cpu` is [`solve_flow`]).
pub fn solve_flow_on(stages: &dyn Stages, x: &Image, y: &Image, settings: &CreateVectorsSettings) -> Result<FlowField> {
    stages.solve(x, y, &settings.flow_settings())
}

/// The Create Vectors pass for one layer: solves the flow of `x` towards `y` and encodes it into a
/// new image of the size and format of `x`: red `dx`, green `dy` centred and scaled by
/// `max_displace`, blue 0, alpha white.
///
/// Tiny frames (U, ours, parity gap G7): a frame with a side of 1 is not
/// [`crate::prep::searchable`], so the solver does not run and the result is the zero vector
/// image ([`vectors::zero_vectors_image`]), the image a zero flow encodes to. Any other size is
/// searched (the reduction never halves below 2x2, [`crate::prep::flow_resolution`]).
///
/// Errors are the solver's (`Mismatch` for images of different shapes) and the encoder's.
pub fn create_vectors(x: &Image, y: &Image, settings: &CreateVectorsSettings) -> Result<Image> {
    create_vectors_on(&Cpu, x, y, settings)
}

/// [`create_vectors`] on `stages`.
pub fn create_vectors_on(stages: &dyn Stages, x: &Image, y: &Image, settings: &CreateVectorsSettings) -> Result<Image> {
    if !crate::prep::searchable(x.width, x.height) {
        if !x.same_shape(y) {
            return Err(fx_core::FxError::Mismatch(format!(
                "create vectors: incompatible inputs ({}x{} {:?} against {}x{} {:?})",
                x.width,
                x.height,
                x.fmt(),
                y.width,
                y.height,
                y.fmt()
            )));
        }
        return Ok(vectors::zero_vectors_image(x.width, x.height, x.fmt()));
    }
    let flow = solve_flow_on(stages, x, y, settings)?;
    vectors::encode_vectors_image(x.width, x.height, x.fmt(), &flow, settings.max_displace)
}

/// The layers `(first, last)` a `Layers Output` popup value outputs: 1 All gives
/// `(0, nfg)`, 2 to 5 (FG1, FG2, FG3, MAIN_BG) the single layer `min(value - 2, nfg)`, so a value
/// naming a layer that does not exist outputs Main_BG (index `nfg`). A value below 1 (the popup
/// has none) is All. It is the arithmetic of `layers::LayerJob::output_group`.
pub fn output_group(layers_output: i32, nfg: usize) -> (usize, usize) {
    if layers_output >= 2 {
        let l = ((layers_output - 2) as usize).min(nfg);
        (l, l)
    } else {
        (0, nfg)
    }
}

/// One output frame of Create Vectors with everything the layered paths add to the single
/// layer case. `layers`, `points` and the solve frames follow the engine's
/// layer numbering: index `k < nfg` is FG `k + 1`, index `nfg` is Main_BG.
#[derive(Clone, Debug, PartialEq)]
pub struct CreateVectorsJob {
    /// Resolution, render quality, Prep, Track RGB+A and `Max Displace` of the pass. Its
    /// `sensitivity` is not read: every layer brings its own (`layers[k].sensitivity`).
    pub settings: CreateVectorsSettings,
    /// `nfg + 1` entries, Main_BG last: sensitivity (divided by 100), Track, and for FG k the
    /// shrink of its inverse matte in pixels (`matte_chan` is the glue's, the mattes below are
    /// already extracted).
    pub layers: Vec<LayerParams>,
    /// The `nfg` raw FG mattes at the layer time of frame `X` (frame `a` of the pair), FG1
    /// first, each at the output size (`matte::extract`).
    pub mattes_x: Vec<Matte>,
    /// The same at the layer time of frame `Y` (frame `b`).
    pub mattes_y: Vec<Matte>,
    /// The tracking points in use per layer (`nfg + 1` lists, or empty for none), each at the
    /// layer times of frame `A = X` and `B = Y` (`UserPoint::pair(true)` is `(from X, to Y)`).
    pub points: Vec<Vec<UserPoint>>,
    /// The layers that reach the output, `first..=last` ([`output_group`]).
    pub first: usize,
    pub last: usize,
}

impl CreateVectorsJob {
    /// The single layer job of today's path: Main_BG only, no matte, no point, Track on, the
    /// sensitivity of `settings`.
    pub fn single(settings: CreateVectorsSettings) -> CreateVectorsJob {
        CreateVectorsJob {
            settings,
            layers: vec![LayerParams { matte_chan: 0, sensitivity: settings.sensitivity, shrink: 0.0, track: true }],
            mattes_x: Vec::new(),
            mattes_y: Vec::new(),
            points: Vec::new(),
            first: 0,
            last: 0,
        }
    }

    /// The number of FG matte layers.
    pub fn nfg(&self) -> usize {
        self.mattes_x.len()
    }

    /// The points of layer `k` as `(from X, to Y)` pairs, in point order.
    fn pairs(&self, k: usize) -> Vec<([f32; 2], [f32; 2])> {
        self.points.get(k).map(|v| v.iter().map(|p| p.pair(true)).collect()).unwrap_or_default()
    }

    /// The solver settings of layer `k`: the pass settings with that layer's sensitivity.
    fn flow_settings(&self, k: usize) -> FlowSettings {
        FlowSettings { sensitivity: self.layers[k].sensitivity, ..self.settings.flow_settings() }
    }

    /// Today's single layer case: no FG matte, no point in use and Main_BG's Track on.
    pub fn is_single_layer(&self) -> bool {
        self.nfg() == 0 && self.pairs(0).is_empty() && self.layers[0].track
    }

    /// The consistency of the job with frames of `w x h`: one matte per FG layer at each frame,
    /// `nfg + 1` layers, `nfg + 1` point lists or none, and `first <= last <= nfg`.
    fn check(&self, w: usize, h: usize) -> Result<()> {
        let nfg = self.nfg();
        let bad = |m: String| Err(fx_core::FxError::Mismatch(format!("create vectors job: {m}")));
        if self.mattes_y.len() != nfg {
            return bad(format!("{nfg} mattes at X and {} at Y", self.mattes_y.len()));
        }
        if self.layers.len() != nfg + 1 {
            return bad(format!("{} layers for {nfg} FG mattes (Main_BG is the last)", self.layers.len()));
        }
        if !self.points.is_empty() && self.points.len() != nfg + 1 {
            return bad(format!("{} point lists for {} layers", self.points.len(), nfg + 1));
        }
        if !(self.first <= self.last && self.last <= nfg) {
            return bad(format!("output layers {}..={} of {}", self.first, self.last, nfg + 1));
        }
        for m in self.mattes_x.iter().chain(&self.mattes_y) {
            if m.width != w || m.height != h || m.data.len() != w * h {
                return bad(format!("a {}x{} matte for frames of {w}x{h}", m.width, m.height));
            }
        }
        Ok(())
    }
}

fn check_pair(x: &Image, y: &Image) -> Result<()> {
    if x.same_shape(y) {
        return Ok(());
    }
    Err(fx_core::FxError::Mismatch(format!(
        "create vectors: incompatible inputs ({}x{} {:?} against {}x{} {:?})",
        x.width,
        x.height,
        x.fmt(),
        y.width,
        y.height,
        y.fmt()
    )))
}

/// The flows of the layers `job.first..=job.last` (entry `i` is layer `first + i`) for the solve
/// `X(p) ~ Y(p + flow(p))` with `x` the frame `a` and `y` the frame `b` of the pair, each at the
/// working resolution (one flow per layer, V).
///
/// - The luma pyramids of `x` and `y` are built once and shared by the layers; only the
///   sensitivity, the masks and the user displacement differ per layer.
/// - With FG layers (`nfg >= 1`, the "multi layer" flag) every solve gets the mask pyramids
///   (`flow::extras::MaskPyramid`) of the layer's tracking matte (`matte::effective` of the raw
///   mattes with the shrinks, rule G1c and G1) at `X` and at `Y`. With `nfg = 0` there is no mask
///   (the single layer solve), and a group whose layers all have Track off computes no tracking
///   matte.
/// - A layer with points in use gets their user displacement (`points::user_displacement`,
///   rule G2) at every level.
/// - A layer whose Track is off does not solve: its flow is the second tracker of its
///   points (`points::second_tracker`, rule G5), the exact zero field without a point.
///
/// With `nfg = 0`, no point and Track on this is `solve_flow` bit for bit (the plain driver of
/// `flow::extras::solve_pyramids_with`). Tiny frames (U, ours; the rule of
/// [`create_vectors_job`]): a frame that is not [`crate::prep::searchable`] has no search, and
/// every entry is the zero field of the frame size. Records gap G1 with FG layers (a layered
/// frame is built); the primitives record G1b, G1c, G2 and G5 where their rules run.
///
/// Errors: `Mismatch` for frames of different shapes or a job that does not fit them; the
/// solver's.
pub fn layer_flows(x: &Image, y: &Image, job: &CreateVectorsJob) -> Result<Vec<FlowField>> {
    layer_flows_on(&Cpu, x, y, job)
}

/// [`layer_flows`] on `stages`.
pub fn layer_flows_on(stages: &dyn Stages, x: &Image, y: &Image, job: &CreateVectorsJob) -> Result<Vec<FlowField>> {
    check_pair(x, y)?;
    let (w, h) = (x.width, x.height);
    job.check(w, h)?;
    let (first, last) = (job.first, job.last);
    if !crate::prep::searchable(w, h) {
        return Ok((first..=last).map(|_| FlowField::new(w, h)).collect());
    }
    let nfg = job.nfg();
    if nfg >= 1 {
        fx_core::gaps::record(&crate::gaps::G1);
    }
    let tracks = (first..=last).any(|k| job.layers[k].track);
    let shrinks: Vec<f32> = job.layers[..nfg].iter().map(|l| l.shrink).collect();
    let base = job.flow_settings(first);
    let geo = base.geometry(w, h)?;
    // the single layer solve reads the frames themselves; a layer of a stack with FG mattes solves on
    // its own layer images (below), so only nfg = 0 builds these
    let pyramids = if tracks && nfg == 0 {
        let (px, py) = rayon::join(|| flow::build_pyramid(x, &base), || flow::build_pyramid(y, &base));
        Some((px?, py?))
    } else {
        None
    };
    let mut out = Vec::with_capacity(last - first + 1);
    for k in first..=last {
        let pairs = job.pairs(k);
        let (layer, fs) = (&job.layers[k], job.flow_settings(k));
        let f = match &pyramids {
            Some((px, py)) if layer.track => {
                let user = (!pairs.is_empty()).then(|| points::user_displacement(&pairs, px.geometry(), (w, h)));
                let extras = SolveExtras { masks: None, user: user.as_ref() };
                stages.solve_pyramids_with(px, py, &fs, extras)?
            }
            None if layer.track && nfg >= 1 => {
                // read from the code (`matte::layer_solve_mattes`): the frames times the layer's visible
                // matte at X and at Y, and the mask pyramids of `white - (own - visible)`
                let (vis_x, mask_x) = matte::layer_solve_mattes(&job.mattes_x, &shrinks, k, x.fmt(), w, h)?;
                let (vis_y, mask_y) = matte::layer_solve_mattes(&job.mattes_y, &shrinks, k, y.fmt(), w, h)?;
                let (lx, ly) = (matte::premultiply_pl(x, &vis_x)?, matte::premultiply_pl(y, &vis_y)?);
                let (px, py) = rayon::join(|| flow::build_pyramid(&lx, &fs), || flow::build_pyramid(&ly, &fs));
                let (px, py) = (px?, py?);
                let masks = (MaskPyramid::build_ungated(&mask_x, &fs)?, MaskPyramid::build_ungated(&mask_y, &fs)?);
                let user = (!pairs.is_empty()).then(|| points::user_displacement(&pairs, px.geometry(), (w, h)));
                let extras = SolveExtras { masks: Some((&masks.0, &masks.1)), user: user.as_ref() };
                stages.solve_pyramids_with(&px, &py, &fs, extras)?
            }
            _ => points::second_tracker(&pairs, &geo, (w, h))?,
        };
        out.push(f);
    }
    Ok(out)
}

/// The flow the Debug views of a job use (rule G4: with `nfg >= 1` the merged flow):
/// with `nfg = 0` the flow of the only layer as [`layer_flows`] gives it (at the working
/// resolution); with FG layers the merged flow of `vectors::merge_flows` at the output size,
/// over the layers `first..=last` with the OVER weights of the raw mattes at `X`
/// (`matte::over_weights`, U, ours: the Layers Output group applies to the Debug display as it
/// does to Vectors). Records gap G1d with FG layers. A frame that is not searchable gives the
/// zero field. Errors as [`layer_flows`].
pub fn job_flow(x: &Image, y: &Image, job: &CreateVectorsJob) -> Result<FlowField> {
    job_flow_on(&Cpu, x, y, job)
}

/// [`job_flow`] on `stages`.
pub fn job_flow_on(stages: &dyn Stages, x: &Image, y: &Image, job: &CreateVectorsJob) -> Result<FlowField> {
    let mut flows = layer_flows_on(stages, x, y, job)?;
    let (w, h) = (x.width, x.height);
    if job.nfg() == 0 || !crate::prep::searchable(w, h) {
        return Ok(flows.swap_remove(0));
    }
    fx_core::gaps::record(&crate::gaps::G1D);
    let weights = matte::over_weights(&job.mattes_x, job.first, job.last, w, h)?;
    let layers: Vec<(&FlowField, &Matte)> = flows.iter().zip(weights.iter()).collect();
    vectors::merge_flows(w, h, &layers)
}

/// The Debug display of a job, view `code` (the engine codes of `crate::debug`), with the solves on
/// `stages` and the Debug forward warp on `warp`. Without FG mattes it is [`crate::debug::debug_view_on`]
/// with the flow of [`job_flow_on`]. With FG mattes (READ in `CreateMotionVectors_64`, the Debug branch of
/// the per frame driver): the output starts transparent
/// and the layers of the group are visited from the bottom one up; per layer `k` its layer images
/// `X_k = premultiply_pl(x, visible_k at X)` and `Y_k` (the solve's, [`matte::layer_solve_mattes`]) and
/// the forward warp `W_k` of `X_k` by the layer's own flow ([`layer_flows_on`]); the output takes `Y_k`
/// OVER it, then `W_k` OVER it ([`matte::over_pl`], the image OVER of the plugin). The Warp view of a one
/// layer group is `W_k` alone. The other views compare the result (as `W'`) with `y` for a group of
/// several layers, with `Y_k` for a one layer group (whose Diff, Detail and Edge first run
/// [`crate::debug::match_alpha`] on the two; its Blend view runs a step that was not read, U: the plain
/// `lerp` here). Errors as [`layer_flows`] and the forward warp's.
pub fn debug_job_view_on(stages: &dyn Stages, warp: &dyn Stages, x: &Image, y: &Image, job: &CreateVectorsJob, code: i32) -> Result<Image> {
    use crate::debug::{self, VIEW_WARP};
    check_pair(x, y)?;
    let (w, h, fmt) = (x.width, x.height, x.fmt());
    job.check(w, h)?;
    if job.nfg() == 0 || !crate::prep::searchable(w, h) {
        let flow = job_flow_on(stages, x, y, job)?;
        return debug::debug_view_on(warp, x, y, &flow, code);
    }
    let flows = layer_flows_on(stages, x, y, job)?;
    let shrinks: Vec<f32> = job.layers[..job.nfg()].iter().map(|l| l.shrink).collect();
    let single = job.first == job.last;
    let mut out = Image::new(w, h, fmt);
    let mut y_layer = None;
    for k in (job.first..=job.last).rev() {
        let flow = &flows[k - job.first];
        let (vx, _) = matte::layer_solve_mattes(&job.mattes_x, &shrinks, k, fmt, w, h)?;
        let (vy, _) = matte::layer_solve_mattes(&job.mattes_y, &shrinks, k, fmt, w, h)?;
        let (lx, ly) = (matte::premultiply_pl(x, &vx)?, matte::premultiply_pl(y, &vy)?);
        if code == VIEW_WARP && single {
            warp.forward_warp(&mut out, &lx, flow, 1.0f32)?;
        } else {
            matte::over_pl(&mut out, &ly)?;
            let mut wk = Image::new(w, h, fmt);
            warp.forward_warp(&mut wk, &lx, flow, 1.0f32)?;
            matte::over_pl(&mut out, &wk)?;
        }
        y_layer = Some(ly);
    }
    if code == VIEW_WARP {
        return Ok(out);
    }
    fx_core::gaps::record(&crate::gaps::G4);
    match (single, y_layer) {
        (true, Some(mut yk)) => {
            if code != debug::VIEW_BLEND {
                debug::match_alpha(&mut yk, &mut out);
            }
            Ok(debug::views_of(&out, &yk, code))
        }
        _ => Ok(debug::views_of(&out, y, code)),
    }
}

/// The Create Vectors pass of one frame with every layered input: the
/// vector image of the size and format of `x`.
///
/// - The single layer case (`nfg = 0`, no point in use, Main_BG's Track on) IS today's
///   [`create_vectors`] with Main_BG's sensitivity, so it keeps its bytes.
/// - `nfg = 0` with points or Track off: the encoding ([`vectors::encode_vectors_image`]) of
///   the only layer's flow from [`layer_flows`].
/// - With FG layers: the flows of the layers `first..=last` ([`layer_flows`]) merged by
///   [`vectors::encode_vectors_merged`] (rule G1d) with the OVER weights of the raw mattes at
///   `X`, the frame whose grid the flows live on (`matte::over_weights`, U, ours), Main_BG's
///   weight the visibility left by the FG layers of the group.
///
/// Tiny frames (U, ours, G7 family): a frame that is not [`crate::prep::searchable`] (a side of
/// 1) is the zero vector image ([`vectors::zero_vectors_image`]: 127, 32767 or 0.5 in R and G,
/// B 0, alpha white) whatever the FG layers, points, Track and Layers Output: neither the
/// solver, the second tracker nor the merge runs there, and no gap is recorded.
///
/// Errors: `Mismatch` for frames of different shapes or a job that does not fit them; the
/// solver's and the encoder's.
pub fn create_vectors_job(x: &Image, y: &Image, job: &CreateVectorsJob) -> Result<Image> {
    create_vectors_job_on(&Cpu, x, y, job)
}

/// [`create_vectors_job`] on `stages`.
pub fn create_vectors_job_on(stages: &dyn Stages, x: &Image, y: &Image, job: &CreateVectorsJob) -> Result<Image> {
    check_pair(x, y)?;
    let (w, h, fmt) = (x.width, x.height, x.fmt());
    job.check(w, h)?;
    if !crate::prep::searchable(w, h) {
        return Ok(vectors::zero_vectors_image(w, h, fmt));
    }
    if job.is_single_layer() {
        let settings = CreateVectorsSettings { sensitivity: job.layers[0].sensitivity, ..job.settings };
        return create_vectors_on(stages, x, y, &settings);
    }
    let flows = layer_flows_on(stages, x, y, job)?;
    let scale = job.settings.max_displace;
    if job.nfg() == 0 {
        return vectors::encode_vectors_image(w, h, fmt, &flows[0], scale);
    }
    // the weights inside the whole stack (layers above the group occlude it), and a single layer
    // output carries its weight in the blue channel (MEASURED, `matte::stack_weights`,
    // `vectors::encode_vectors_merged_blue`)
    let mut img = Image::new(w, h, fmt);
    if job.first == job.last {
        // READ: a group of one layer is that layer's own flow, not weighted, and the blue channel
        // its visible matte plane at X (`vectors::encode_vectors_blue`, `matte::layer_solve_mattes`)
        let shrinks: Vec<f32> = job.layers[..job.nfg()].iter().map(|l| l.shrink).collect();
        let (visible, _) = matte::layer_solve_mattes(&job.mattes_x, &shrinks, job.first, fmt, w, h)?;
        vectors::encode_vectors_blue(&mut img, &flows[0], scale, &visible)?;
    } else {
        // READ: the group is composited from its bottom layer up: the bottom layer's flow encoded
        // plainly, every layer above it laid over with its raw matte at X (`vectors::composite_vectors`)
        let n = flows.len();
        vectors::encode_vectors(&mut img, &flows[n - 1], scale)?;
        for i in (0..n - 1).rev() {
            let m = matte::integer_plane(&job.mattes_x[job.first + i], fmt);
            vectors::composite_vectors(&mut img, &flows[i], scale, &m)?;
        }
    }
    Ok(img)
}

#[cfg(test)]
mod job_tests;

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

    /// A frame of pseudo random premultiplied pixels (NaN and infinities included at float).
    fn noise(w: usize, h: usize, fmt: crate::image::Fmt, seed: u32) -> Image {
        use crate::image::Samples;
        let mut img = Image::new(w, h, fmt);
        let mut s = seed;
        let mut next = move || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            s >> 8
        };
        match &mut img.samples {
            Samples::U8(v) => v.iter_mut().for_each(|c| *c = next() as u8),
            Samples::U16(v) => v.iter_mut().for_each(|c| *c = next() as u16),
            Samples::F32(v) => v.iter_mut().enumerate().for_each(|(i, c)| {
                *c = match i % 7 {
                    3 => f32::NAN,
                    5 => f32::INFINITY,
                    _ => (next() % 1000) as f32 / 999.0,
                }
            }),
        }
        img
    }

    #[test]
    fn side_of_one_is_the_zero_vector() {
        use crate::image::Fmt;
        for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
            // the zero vector image is what the encoder writes for a zero flow
            for (w, h) in [(2usize, 2usize), (3, 2), (2, 5), (7, 4)] {
                let zero = crate::vectors::encode_vectors_image(w, h, fmt, &FlowField::new(w, h), 32.0).unwrap();
                assert_eq!(vectors::zero_vectors_image(w, h, fmt), zero, "{fmt:?} {w}x{h}");
            }
            for (w, h) in [(1usize, 1usize), (1, 2), (2, 1), (1, 17), (17, 1)] {
                for quality in [QUALITY_NORMAL, QUALITY_FULL] {
                    for hiq in [true, false] {
                        let s = CreateVectorsSettings {
                            quality,
                            hiq,
                            sensitivity: 1.0,
                            prep_contrast_edge: true,
                            prep_delinearize: false,
                            track_rgb_a: true,
                            max_displace: 16.0,
                        };
                        let got = create_vectors(&noise(w, h, fmt, 1), &noise(w, h, fmt, 2), &s).unwrap();
                        assert_eq!(got, vectors::zero_vectors_image(w, h, fmt), "{fmt:?} {w}x{h}");
                    }
                }
                // the literal pixels: A white, R and G the centre, B 0
                let z = vectors::zero_vectors_image(w, h, fmt);
                match &z.samples {
                    crate::image::Samples::U8(v) => assert!(v.chunks(4).all(|p| p == [255, 127, 127, 0])),
                    crate::image::Samples::U16(v) => assert!(v.chunks(4).all(|p| p == [65535, 32767, 32767, 0])),
                    crate::image::Samples::F32(v) => assert!(v.chunks(4).all(|p| p == [1.0, 0.5, 0.5, 0.0])),
                }
                // frames of different shapes are still a mismatch
                let other = noise(w + 1, h, fmt, 3);
                assert!(create_vectors(&noise(w, h, fmt, 1), &other, &CreateVectorsSettings {
                    quality: QUALITY_NORMAL,
                    hiq: true,
                    sensitivity: 0.7,
                    prep_contrast_edge: false,
                    prep_delinearize: false,
                    track_rgb_a: false,
                    max_displace: 16.0,
                })
                .is_err());
            }
        }
    }

    #[test]
    fn tiny_searchable_frames_are_searched() {
        use crate::image::Fmt;
        // 2x2 to 6x6 and 2x33 at Normal: the reduction stops at 2x2, so the solver runs and the encode
        // of its flow is the result (NaN and infinite floats included)
        for fmt in [Fmt::U8, Fmt::U16, Fmt::F32] {
            for (w, h) in [(2usize, 2usize), (3, 3), (4, 4), (5, 5), (6, 6), (2, 33), (33, 2)] {
                for quality in [QUALITY_NORMAL, QUALITY_FULL] {
                    for hiq in [true, false] {
                        let s = CreateVectorsSettings {
                            quality,
                            hiq,
                            sensitivity: 1.0,
                            prep_contrast_edge: false,
                            prep_delinearize: false,
                            track_rgb_a: false,
                            max_displace: 16.0,
                        };
                        let (x, y) = (noise(w, h, fmt, 5), noise(w, h, fmt, 6));
                        let flow = solve_flow(&x, &y, &s).unwrap();
                        let want = crate::vectors::encode_vectors_image(w, h, fmt, &flow, 16.0).unwrap();
                        assert_eq!(create_vectors(&x, &y, &s).unwrap(), want, "{fmt:?} {w}x{h}");
                    }
                }
            }
        }
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
