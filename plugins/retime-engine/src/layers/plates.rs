//! The layered composition of plugin 03 as the original builds it (TwixtorPro_64, read from the code: the per layer
//! loop of the render function, its kernels confirmed bit for bit on the traced layer images, plates
//! and blends of `size-n1-w1.box1-5x5.d8`). Every layer of the group is interpolated
//! ON ITS OWN, bottom layer first, and the interpolated layers are composed with the plugin's OVER:
//!
//! 1. the composition matte of layer `k` at each side ([`comp_plane`]): FG k's raw matte in the units of the depth
//!    (Main_BG: white), cut by every FG layer above it, the nearest first, each cut quantised to the unit
//!    (`trunc(clamp((1.0 - v) * c, 0, white) + 0.5)`, the product in double);
//! 2. the layer images `LA`, `LB`: the frames times that matte ([`matte::premultiply_pl`]'s arithmetic);
//! 3. with two sides (Blend, Weighted Blend) the PLATES: `PA = LA OVER warp(LB to time A)` and
//!    `PB = LB OVER warp(LA to time B)`, each warp by the layer's own flow with amount 1 (inverse warps with hold
//!    edges for warping codes 0 and 1, the forward warp without coverage for codes 2 and 3); Nearest has the one
//!    layer image of its side;
//! 4. every plate [`refill`]ed: unpremultiplied, its colour filled where it is not opaque (the pull-push fill with
//!    the hole mask `white - alpha`), its alpha REPLACED by the layer's own raw matte (Main_BG: the frame's alpha),
//!    premultiplied again; so a layer shows, under the layers above it and where it was only visible in the other
//!    frame, its own colour rebuilt from the other frame and from its surroundings;
//! 5. the two plates go through the single layer composition of the warping code (`Stages::compose_vectors`: the
//!    warps by frac and wA, the blend or Smart Blend, the fills, the blur compensation tail with this layer's flow);
//! 6. Main_BG's interpolated layer is copied into the result, every FG layer is composed OVER it ([`matte::over_pl`]).
//!
//! The flows of a layer ([`solve_and_compose_plates`]) are solved on the luma of ITS OWN layer images (step 2 on the
//! tracking frames), masked where the layers above cover it ([`solver_mask`]); FG1's solve has no mask.
//!
//! Precision contract: the integer units of the depth for the mattes and the 8 and 16 bpc images, f32 in the order
//! written, no fused multiply add; the stages called keep their own contracts.

use super::*;
use crate::image::Fmt;

/// One matte plane in the units of the image depth: whole numbers 0..=255 or 0..=65535 (held as f32), or the float
/// value itself.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Plane {
    pub white: f32,
    pub v: Vec<f32>,
}

fn white_of(fmt: Fmt) -> f32 {
    match fmt {
        Fmt::U8 => 255.0,
        Fmt::U16 => 65535.0,
        Fmt::F32 => 1.0,
    }
}

/// The raw matte `m` (from `matte::extract_raw`, 0..1 at 8 and 16 bpc) in the units of `fmt`: the integer the engine
/// matte came from at 8 and 16 bpc (`trunc(m * white + 0.5)`), the value itself in float (NaN, infinities and values
/// outside 0..1 included: the plugin's plane is the matte channel copied as it is).
pub(crate) fn raw_plane(m: &Matte, fmt: Fmt) -> Plane {
    let white = white_of(fmt);
    let v = match fmt {
        Fmt::F32 => m.data.clone(),
        _ => matte::in_range(&m.data).iter().map(|&x| ((x * white + 0.5f32) as i64) as f32).collect(),
    };
    Plane { white, v }
}

/// One cut of a composition matte `c` (units) by the matte `v` (0..1) of a layer above it (READ: the matte cut's
/// row kernels at 8 bpc and in float): `x = (float)((1.0 - (double)v) * (double)c)`, below 0 gives
/// 0, above `white` gives `white`, a NaN stays NaN (both compares are false for it); in the integer units
/// `trunc(x + 0.5f)` (a NaN cannot reach the integer planes), in float `x + 0.0` (a `-0` product is stored as `+0`).
#[inline]
fn cut(c: f32, v: f32, white: f32, float: bool) -> f32 {
    let x = ((1.0f64 - f64::from(v)) * f64::from(c)) as f32;
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    let x = if 0.0 > x {
        0.0
    } else if x > white {
        white
    } else {
        x
    };
    if float {
        x + 0.0f32
    } else {
        ((x + 0.5f32) as i64) as f32
    }
}

/// The matte `v` of FG layer `j` as the cut reads it: the raw unit value times the f32 `1 / white` (1.0 in float),
/// then the plugin's choke by MINUS the layer's inverse matte shrink (`matte::cut_view`: invert, shrink, invert back,
/// and the clamp of a float plane, which also runs for a zero
/// shrink and keeps a NaN; READ and TRACED), on the rows of
/// the motion path the plates run in.
fn cut_matte(raw: &Plane, shrink: f32, w: usize, h: usize) -> Result<Vec<f32>> {
    let k = 1.0f32 / raw.white;
    let v: Vec<f32> = raw.v.iter().map(|&x| x * k).collect();
    Ok(matte::cut_view(&Matte { width: w, height: h, data: v }, shrink)?.data)
}

/// The composition matte of layer `k` (FG `k`, or Main_BG for `k == nfg`) at one side: its raw matte (Main_BG white)
/// cut by every FG layer above it, `k - 1` first and FG1 last ([`cut`]). `raws` are the raw planes of the FG layers at
/// that side, FG1 first.
pub(crate) fn comp_plane(raws: &[Plane], shrinks: &[f32], k: usize, fmt: Fmt, w: usize, h: usize) -> Result<Plane> {
    let white = white_of(fmt);
    let float = fmt == Fmt::F32;
    let mut c = if k == raws.len() { Plane { white, v: vec![white; w * h] } } else { raws[k].clone() };
    for j in (0..k).rev() {
        let v = cut_matte(&raws[j], shrinks[j], w, h)?;
        for (x, &vj) in c.v.iter_mut().zip(&v) {
            *x = cut(*x, vj, white, float);
        }
    }
    Ok(c)
}

/// The composition mattes of every layer at one side, FG1 first and Main_BG last, as mattes: [`comp_plane`] of the
/// raw mattes `mattes` (0..1, FG1 first) divided by its white, in the format of `like` (8 and 16 bpc: whole units over
/// 255 or 65535, which `trunc(m * white + 0.5)` turns back into the plane exactly; float: the plane itself). The
/// per layer vector loads of Vectors In cap the vector alpha with them (READ in TwixtorProVectorsIn_64, with the
/// layer's two cut planes).
pub(crate) fn comp_mattes(mattes: &[Matte], shrinks: &[f32], like: &Image) -> Result<Vec<Matte>> {
    let (w, h, fmt) = (like.width, like.height, like.fmt());
    let raws: Vec<Plane> = mattes.iter().map(|m| raw_plane(m, fmt)).collect();
    (0..=mattes.len())
        .map(|k| {
            let c = comp_plane(&raws, shrinks, k, fmt, w, h)?;
            let data = if c.white == 1.0 { c.v } else { c.v.iter().map(|&x| x / c.white).collect() };
            Ok(Matte { width: w, height: h, data })
        })
        .collect()
}

/// A layer image: every channel of `src` (alpha included) times the plane `p` (units): 8 and 16 bpc
/// `trunc(c * (m * K) + 0.5)` with `K` the f32 `1/255` or `1/65535`, float `m * c` (the matte multiply kernels
/// at 8 bpc and their 16 bpc and float twins).
pub(crate) fn layer_times(src: &Image, p: &Plane) -> Image {
    let mut out = Image::new(src.width, src.height, src.fmt());
    match (&src.samples, &mut out.samples) {
        (Samples::U8(s), Samples::U8(o)) => {
            for (i, (sp, op)) in s.chunks_exact(CHANNELS).zip(o.chunks_exact_mut(CHANNELS)).enumerate() {
                let f = p.v[i] * matte::PL_K8;
                for c in 0..CHANNELS {
                    op[c] = (f32::from(sp[c]) * f + 0.5f32) as i64 as u8;
                }
            }
        }
        (Samples::U16(s), Samples::U16(o)) => {
            for (i, (sp, op)) in s.chunks_exact(CHANNELS).zip(o.chunks_exact_mut(CHANNELS)).enumerate() {
                let f = p.v[i] * matte::PL_K16;
                for c in 0..CHANNELS {
                    op[c] = (f32::from(sp[c]) * f + 0.5f32) as i64 as u16;
                }
            }
        }
        (Samples::F32(s), Samples::F32(o)) => {
            for (i, (sp, op)) in s.chunks_exact(CHANNELS).zip(o.chunks_exact_mut(CHANNELS)).enumerate() {
                for c in 0..CHANNELS {
                    op[c] = crate::nanrule::mul(p.v[i], sp[c]);
                }
            }
        }
        _ => unreachable!("Image::new follows the format of the source"),
    }
    out
}

/// The unpremultiply of the plugin's image kernel in place: 8 and 16 bpc a pixel with alpha 0 is kept, else
/// `inv = white / a`, colour `min(c * inv, white)` rounded `trunc((double)x + 0.5)`; float a pixel whose alpha is
/// below 1e-7 (as a double) is kept, else `inv = 1 / a` and colour `inv * c`; alpha is never written.
pub(crate) fn unpremultiply_pl(img: &mut Image) {
    match &mut img.samples {
        Samples::U8(s) => s.chunks_exact_mut(CHANNELS).for_each(|p| {
            let q = crate::image::unpremultiply_u8([p[0], p[1], p[2], p[3]]);
            p.copy_from_slice(&q);
        }),
        Samples::U16(s) => s.chunks_exact_mut(CHANNELS).for_each(|p| {
            if p[0] == 0 {
                return;
            }
            let inv = 65535.0f32 / f32::from(p[0]);
            for c in 1..CHANNELS {
                let mut v = f32::from(p[c]) * inv;
                if 65535.0f32 < v {
                    v = 65535.0;
                }
                p[c] = crate::image::q(v) as u16;
            }
        }),
        Samples::F32(s) => s.chunks_exact_mut(CHANNELS).for_each(|p| {
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if 1e-7f64 > f64::from(p[0]) {
                return;
            }
            let inv = 1.0f32 / p[0];
            for c in 1..CHANNELS {
                p[c] = crate::nanrule::mul(inv, p[c]);
            }
        }),
    }
}

/// The premultiply of the plugin's image kernel in place: 8 and 16 bpc `f = a * K` (`K` the f32 `1/255` or
/// `1/65535`), colour `trunc((double)(c * f) + 0.5)`; float `a * c`; alpha is never written.
pub(crate) fn premultiply_pl_inplace(img: &mut Image) {
    match &mut img.samples {
        Samples::U8(s) => s.chunks_exact_mut(CHANNELS).for_each(|p| {
            let q = crate::image::premultiply_u8([p[0], p[1], p[2], p[3]]);
            p.copy_from_slice(&q);
        }),
        Samples::U16(s) => s.chunks_exact_mut(CHANNELS).for_each(|p| {
            let f = f32::from(p[0]) * matte::PL_K16;
            for c in 1..CHANNELS {
                p[c] = crate::image::q(f32::from(p[c]) * f) as u16;
            }
        }),
        Samples::F32(s) => s.chunks_exact_mut(CHANNELS).for_each(|p| {
            let q = crate::image::premultiply_f32([p[0], p[1], p[2], p[3]]);
            p.copy_from_slice(&q);
        }),
    }
}

/// The alpha of `img` replaced by the plane `p` (units), or by the alpha of `frame` (Main_BG).
fn set_alpha(img: &mut Image, alpha: Alpha<'_>) {
    let n = img.width * img.height;
    for i in 0..n {
        match (&mut img.samples, alpha) {
            (Samples::U8(s), Alpha::Plane(p)) => s[i * CHANNELS + A] = p.v[i] as u8,
            (Samples::U16(s), Alpha::Plane(p)) => s[i * CHANNELS + A] = p.v[i] as u16,
            (Samples::F32(s), Alpha::Plane(p)) => s[i * CHANNELS + A] = p.v[i],
            (Samples::U8(s), Alpha::Frame(f)) => {
                if let Samples::U8(fs) = &f.samples {
                    s[i * CHANNELS + A] = fs[i * CHANNELS + A];
                }
            }
            (Samples::U16(s), Alpha::Frame(f)) => {
                if let Samples::U16(fs) = &f.samples {
                    s[i * CHANNELS + A] = fs[i * CHANNELS + A];
                }
            }
            (Samples::F32(s), Alpha::Frame(f)) => {
                if let Samples::F32(fs) = &f.samples {
                    s[i * CHANNELS + A] = fs[i * CHANNELS + A];
                }
            }
        }
    }
}

/// Where the alpha of a refilled layer comes from: the layer's raw matte plane, or the frame (Main_BG).
#[derive(Clone, Copy)]
pub(crate) enum Alpha<'a> {
    Plane(&'a Plane),
    Frame(&'a Image),
}

/// The refill of a plate (the steps of the render function after the plate OVER): [`unpremultiply_pl`], the pull-push
/// fill of every channel with the hole mask `white - alpha` (the plugin's single channel fill, the fill of the Smart
/// Blend compositions: `holefill::fill_holes_masked_with`), the alpha replaced ([`Alpha`]), [`premultiply_pl_inplace`].
pub(crate) fn refill(img: &mut Image, alpha: Alpha<'_>, is360: bool, taps: crate::holefill::VerticalTaps) -> Result<()> {
    unpremultiply_pl(img);
    let holes = matte::alpha_holes(img);
    let mask = match &holes {
        HoleMask::U8(m) => HoleMaskRef::U8(m),
        HoleMask::U16(m) => HoleMaskRef::U16(m),
        HoleMask::F32(m) => HoleMaskRef::F32(m),
    };
    crate::holefill::fill_holes_masked_with(img, mask, is360, taps)?;
    set_alpha(img, alpha);
    premultiply_pl_inplace(img);
    Ok(())
}

/// A warp of a layer image all the way to the other frame (amount 1) for its plate: inverse warp with hold edges for
/// warping codes 0 and 1, the forward warp without coverage for codes 2 and 3.
fn warp_full(settings: &EngineSettings, src: &Image, flow: &FlowField) -> Result<Image> {
    let mut t = Image::new(src.width, src.height, src.fmt());
    if settings.warp_mode >= 2 {
        settings.stages.get().forward_warp(&mut t, src, flow, 1.0)?;
    } else {
        inv_warp(&mut t, src, flow, 1.0, WarpOpts { is360: settings.wrap_360, ..WarpOpts::default() })?;
    }
    Ok(t)
}

/// The flow that brings side `from_a` (A when true) to the other frame: inverse warps read `F_AB` for B's image at A
/// and `F_BA` for A's image at B, the forward warps push each frame by its own flow (A by `F_AB`).
fn plate_flow<'f>(flows: &'f Flows, code: i32, from_a: bool, k: usize) -> Result<&'f FlowField> {
    let ab = if code >= 2 { from_a } else { !from_a };
    if ab {
        flows.f_ab.as_ref().ok_or_else(|| missing_flow("F_AB", k))
    } else {
        flows.f_ba.as_ref().ok_or_else(|| missing_flow("F_BA", k))
    }
}

/// The solver mask of layer `k` at one side (READ, the render function before the solve of a layer: a one channel
/// subtraction, then the inversion, handed to the solver): `white - (raw - comp)` with
/// `raw` the layer's raw matte (Main_BG: white) and `comp` its composition matte ([`comp_plane`]), as a matte in 0..1
/// (`/ white`): 1 where no layer above covers layer `k`. `None` when it is 1 everywhere (FG1, or nothing above covers the
/// layer at this side).
pub(crate) fn solver_mask(raws: &[Plane], comp: &Plane, k: usize, w: usize, h: usize) -> Option<Matte> {
    let white = comp.white;
    let data: Vec<f32> = (0..comp.v.len())
        .map(|i| {
            let r = if k == raws.len() { white } else { raws[k].v[i] };
            (white - (r - comp.v[i])) / white
        })
        .collect();
    if data.iter().all(|&v| v == 1.0) {
        None
    } else {
        Some(Matte { width: w, height: h, data })
    }
}

/// The settings of a layer solve with plates from the render's `FlowSettings`: `(solve, pyramid)`, the first for the
/// solver and the mask pyramids, the second for the luma pyramids. In the layered solves Track RGB+A
/// (`FlowSettings::track_alpha`) adds no alpha to the luma:
/// the render function hands that flag to the pyramid prepare
/// as its filter mode argument (READ in TwixtorPro_64: the argument that switches the blur, the resamples and the band
/// blur from mode 3 to the 360 mode 6; the luma gets the solver's alpha flag, 0 here). TRACED on
/// `wrap-w1` with Track RGB+A on and off: the only functions that differ are the 360 variants of the separable
/// filters and of the sampler of the pyramid resample, the argument is 1 and 0, the luma is the plain kernel in both.
/// So the luma pyramids are built in the 360 mode of the filters (`FlowSettings::wrap` for the pyramid only) and the
/// solver, the mask pyramids and the user displacement keep the render's own 360 flag.
pub(crate) fn layer_solve_settings(base: FlowSettings) -> (FlowSettings, FlowSettings) {
    let solve = FlowSettings { track_alpha: false, ..base };
    let pyramid = FlowSettings { wrap: base.wrap || base.track_alpha, ..solve };
    (solve, pyramid)
}

/// The solves and the composition of a layered frame for a job with `plates` (READ, the render function's layer loop):
/// every layer `k` of the group solves the luma pyramids of ITS OWN layer images (the tracking frames times its
/// composition matte, [`layer_times`]; Track RGB+A switches the filters of their pyramids, [`layer_solve_settings`]), masked
/// by [`solver_mask`] at both sides (through UNGATED mask pyramids, `MaskPyramid::build_ungated`: the CPU solver never
/// gates the block matching with the masks, READ; a side without a mask takes
/// the constant 1), with its own sensitivity and tracking points; then [`compose_layers_plates`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_and_compose_plates(
    settings: &EngineSettings,
    sel: &Selection,
    job: &LayerJob,
    a: &Image,
    b: &Image,
    xa: &Image,
    xb: &Image,
    mattes: &(Vec<Matte>, Vec<Matte>),
    pts: &[Vec<UserPoint>],
) -> Result<Image> {
    let shrinks = job.shrinks();
    let (base, pyramid) = layer_solve_settings(flow_settings(settings, sel));
    let (w, h, fmt) = (xa.width, xa.height, xa.fmt());
    let raws_a: Vec<Plane> = mattes.0.iter().map(|m| raw_plane(m, fmt)).collect();
    let raws_b: Vec<Plane> = mattes.1.iter().map(|m| raw_plane(m, fmt)).collect();
    let mut flows = Vec::with_capacity(job.last - job.first + 1);
    for k in job.first..=job.last {
        let layer = &job.layers[k];
        let fs = FlowSettings { sensitivity: layer.sensitivity, ..base };
        let ca = comp_plane(&raws_a, &shrinks, k, fmt, w, h)?;
        let cb = comp_plane(&raws_b, &shrinks, k, fmt, w, h)?;
        let (pa, pb) = pyramids(&layer_times(xa, &ca), &layer_times(xb, &cb), &FlowSettings { sensitivity: layer.sensitivity, ..pyramid })?;
        let masks = match (solver_mask(&raws_a, &ca, k, w, h), solver_mask(&raws_b, &cb, k, w, h)) {
            (None, None) => None,
            (ma, mb) => {
                let one = || Matte::filled(w, h, 1.0);
                let (ma, mb) = (ma.unwrap_or_else(one), mb.unwrap_or_else(one));
                Some((MaskPyramid::build_ungated(&ma, &fs)?, MaskPyramid::build_ungated(&mb, &fs)?))
            }
        };
        flows.push(solve_layer(settings, sel, &fs, layer.track, &pa, &pb, masks.as_ref().map(|(x, y)| (x, y)), &pts[k])?);
    }
    compose_layers(settings, sel, job, a, b, mattes, &flows)
}

/// [`compose_layers`] for a job with `plates` (module documentation). Same inputs and checks.
pub(crate) fn compose_layers_plates(
    settings: &EngineSettings,
    sel: &Selection,
    job: &LayerJob,
    a: &Image,
    b: &Image,
    mattes: &(Vec<Matte>, Vec<Matte>),
    flows: &[Flows],
) -> Result<Image> {
    let (w, h, fmt) = (a.width, a.height, a.fmt());
    let shrinks = job.shrinks();
    let raws_a: Vec<Plane> = mattes.0.iter().map(|m| raw_plane(m, fmt)).collect();
    let raws_b: Vec<Plane> = mattes.1.iter().map(|m| raw_plane(m, fmt)).collect();
    let taps = settings.stages.get().vertical_taps();
    let is360 = settings.wrap_360;
    let code = settings.warp_mode;
    let mut out = Image::new(w, h, fmt);
    for k in (job.first..=job.last).rev() {
        let f = &flows[k - job.first];
        let alpha_a = if k == job.nfg { Alpha::Frame(a) } else { Alpha::Plane(&raws_a[k]) };
        let alpha_b = if k == job.nfg { Alpha::Frame(b) } else { Alpha::Plane(&raws_b[k]) };
        let la = layer_times(a, &comp_plane(&raws_a, &shrinks, k, fmt, w, h)?);
        let lb = layer_times(b, &comp_plane(&raws_b, &shrinks, k, fmt, w, h)?);
        let r = if sel.interp == 0 {
            let takes_a = nearest_takes_a(code, sel.wa);
            let mut p = if takes_a { la } else { lb };
            refill(&mut p, if takes_a { alpha_a } else { alpha_b }, is360, taps)?;
            settings.stages.get().compose_vectors(settings, sel, &p, &p, f)?
        } else {
            // the plate of a side: its own layer image OVER the other side's brought to its time
            let mut pa = warp_full(settings, &lb, plate_flow(f, code, false, k)?)?;
            matte::over_pl(&mut pa, &la)?;
            let mut pb = warp_full(settings, &la, plate_flow(f, code, true, k)?)?;
            matte::over_pl(&mut pb, &lb)?;
            refill(&mut pa, alpha_a, is360, taps)?;
            refill(&mut pb, alpha_b, is360, taps)?;
            settings.stages.get().compose_vectors(settings, sel, &pa, &pb, f)?
        };
        if k == job.nfg {
            out = r;
        } else {
            matte::over_pl(&mut out, &r)?;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::test_support::settings;
    use crate::stages::AboveFirstTaps;

    fn img_u8(w: usize, h: usize, px: &[[u8; 4]]) -> Image {
        let mut im = Image::new(w, h, Fmt::U8);
        if let Samples::U8(s) = &mut im.samples {
            for (o, p) in s.chunks_exact_mut(CHANNELS).zip(px) {
                o.copy_from_slice(p);
            }
        }
        im
    }

    fn px_u8(im: &Image, x: usize, y: usize) -> [u8; 4] {
        match &im.samples {
            Samples::U8(s) => {
                let i = (y * im.width + x) * CHANNELS;
                [s[i], s[i + 1], s[i + 2], s[i + 3]]
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn track_rgb_a_switches_the_layer_pyramids_to_the_360_filters_and_keeps_the_luma_plain() {
        let base = FlowSettings::new(4, true, 0.7);
        // off: both settings are the render's own
        let (solve, pyramid) = layer_solve_settings(base);
        assert_eq!((solve, pyramid), (base, base));
        // on: no alpha anywhere, the 360 mode for the luma pyramids only
        let (solve, pyramid) = layer_solve_settings(FlowSettings { track_alpha: true, ..base });
        assert_eq!(solve, base);
        assert_eq!(pyramid, FlowSettings { wrap: true, ..base });
        // with 360 Video the solve keeps its own wrap
        let v360 = FlowSettings { wrap: true, ..base };
        let (solve, pyramid) = layer_solve_settings(FlowSettings { track_alpha: true, ..v360 });
        assert_eq!((solve, pyramid), (v360, v360));
    }

    #[test]
    fn the_cut_of_a_composition_matte_is_quantised_per_layer_above_the_nearest_first() {
        // 8 bpc, one pixel, FG1 128 and FG2 200: Main_BG 255 -> (1 - 200/255) 255 = 54.999996 -> 55, then
        // (1 - 128/255) 55 = 27.39 -> 27; FG2 200 -> (1 - 128/255) 200 = 99.61 -> 100; FG1 keeps 128
        let raws = vec![Plane { white: 255.0, v: vec![128.0] }, Plane { white: 255.0, v: vec![200.0] }];
        let c = |k| comp_plane(&raws, &[0.0, 0.0], k, Fmt::U8, 1, 1).unwrap().v[0];
        assert_eq!(c(2), 27.0);
        assert_eq!(c(1), 100.0);
        assert_eq!(c(0), 128.0);
        // the product is formed in double: (1 - v) * c of 0.2156862616539001 and 255 rounds to 54.999996, not 55.5 - 0.5
        assert_eq!(cut(255.0, 200.0f32 * matte::PL_K8, 255.0, false), 55.0);
        // float: the clamped product, no rounding
        assert_eq!(cut(1.0, 0.25, 1.0, true), 0.75);
        assert_eq!(cut(f32::NAN, 0.25, 255.0, false), 0.0);
    }

    #[test]
    fn the_composition_mattes_give_the_integer_planes_back() {
        // 8 bpc, the planes of the cut test above as mattes (FG1, FG2, Main_BG): trunc(m * 255 + 0.5) gives them back
        let m = |v: f32| Matte { width: 1, height: 1, data: vec![v] };
        let cm = comp_mattes(&[m(128.0 / 255.0), m(200.0 / 255.0)], &[0.0, 0.0], &Image::new(1, 1, Fmt::U8)).unwrap();
        let units: Vec<f32> = cm.iter().map(|c| ((c.data[0] * 255.0f32 + 0.5f32) as i64) as f32).collect();
        assert_eq!(units, vec![128.0, 100.0, 27.0]);
        // 16 bpc: every unit of a ramp survives the division
        let ramp = Matte { width: 65536, height: 1, data: (0..65536).map(|i| i as f32 / 65535.0).collect() };
        let c16 = comp_mattes(&[ramp], &[0.0], &Image::new(65536, 1, Fmt::U16)).unwrap();
        for (i, &v) in c16[0].data.iter().enumerate() {
            assert_eq!((v * 65535.0f32 + 0.5f32) as i64, i as i64);
        }
        // float: the values themselves, Main_BG the cut of white
        let cf = comp_mattes(&[m(0.25)], &[0.0], &Image::new(1, 1, Fmt::F32)).unwrap();
        assert_eq!((cf[0].data[0], cf[1].data[0]), (0.25, 0.75));
    }

    /// READ (the matte cut's float row kernel): below 0 gives 0, above white
    /// gives white, a NaN stays NaN, and the store adds `+ 0.0`; TRACED on `m-nanall-n2-bg` (an all NaN FG1 matte makes
    /// Main_BG's composition matte NaN in every pixel, which the plates refill as holes).
    #[test]
    fn the_float_cut_keeps_a_nan_and_stores_plus_zero() {
        assert!(cut(1.0, f32::NAN, 1.0, true).is_nan());
        assert!(cut(0.0, f32::NAN, 1.0, true).is_nan(), "NaN times 0 is NaN");
        assert_eq!(cut(-0.0, 0.5, 1.0, true).to_bits(), 0.0f32.to_bits(), "a -0 product is stored as +0");
        assert_eq!(cut(1.0, 2.0, 1.0, true), 0.0);
        assert_eq!(cut(1.0, -1.0, 1.0, true), 1.0);
        // a raw float matte reaches the cut through the clamp of the plugin's choke, which runs for a zero shrink too:
        // NaN kept, 2.0 read as 1.0, -inf as 0, -0 kept
        let raws = vec![Plane { white: 1.0, v: vec![f32::NAN, 2.0, f32::NEG_INFINITY, -0.0, 0.25] }];
        let c = comp_plane(&raws, &[0.0], 1, Fmt::F32, 5, 1).unwrap().v;
        assert!(c[0].is_nan());
        assert_eq!(c[1].to_bits(), 0.0f32.to_bits());
        assert_eq!(c[2], 1.0);
        assert_eq!(c[3], 1.0);
        assert_eq!(c[4], 0.75);
        // the raw plane of a float matte is the matte itself
        let m = Matte { width: 2, height: 1, data: vec![f32::NAN, 1.5] };
        let p = raw_plane(&m, Fmt::F32);
        assert!(p.v[0].is_nan());
        assert_eq!(p.v[1], 1.5);
    }

    #[test]
    fn the_solver_mask_of_a_layer_is_white_where_no_layer_above_covers_it() {
        // 8 bpc, three pixels, FG1 255, 128, 0 and FG2 200: FG2's composition matte is 0, 100, 200, its mask
        // (255 - (200 - comp)) / 255; Main_BG's is 0, 27, 55 (255 cut by FG2 to 55, then by FG1), its mask comp / 255; FG1
        // has nothing above it: no mask
        let raws = vec![Plane { white: 255.0, v: vec![255.0, 128.0, 0.0] }, Plane { white: 255.0, v: vec![200.0; 3] }];
        let comp = |k| comp_plane(&raws, &[0.0, 0.0], k, Fmt::U8, 3, 1).unwrap();
        assert_eq!(comp(1).v, vec![0.0, 100.0, 200.0]);
        assert_eq!(comp(2).v, vec![0.0, 27.0, 55.0]);
        assert!(solver_mask(&raws, &comp(0), 0, 3, 1).is_none());
        assert_eq!(solver_mask(&raws, &comp(1), 1, 3, 1).unwrap().data, vec![55.0 / 255.0, 155.0 / 255.0, 1.0]);
        assert_eq!(solver_mask(&raws, &comp(2), 2, 3, 1).unwrap().data, vec![0.0, 27.0 / 255.0, 55.0 / 255.0]);
    }

    #[test]
    fn a_refill_unpremultiplies_fills_what_is_not_opaque_takes_the_layer_alpha_and_premultiplies() {
        // 3x2 at 8 bpc: a half transparent pixel and a transparent one in the top row, opaque (10, 20, 30) elsewhere
        let o = [255, 10, 20, 30];
        let mut p = img_u8(3, 2, &[[128, 100, 50, 0], [0, 0, 0, 0], o, o, o, o]);
        let alpha = Plane { white: 255.0, v: vec![200.0, 255.0, 255.0, 255.0, 255.0, 0.0] };
        refill(&mut p, Alpha::Plane(&alpha), false, crate::holefill::VerticalTaps::AboveFirst).unwrap();
        // both pixels below full alpha are holes of the fill (255 - a), filled from the opaque ones, then the alpha of
        // the plane: 200 gives trunc(c * (200 * 1/255) + 0.5) = 8, 16, 24
        assert_eq!(px_u8(&p, 0, 0), [200, 8, 16, 24]);
        assert_eq!(px_u8(&p, 1, 0), [255, 10, 20, 30]);
        assert_eq!(px_u8(&p, 2, 1), [0, 0, 0, 0]);
        // the unpremultiply alone: 255 / 128 = 1.9921875, 100 -> 199.22 -> 199, 50 -> 99.61 -> 100; alpha 0 untouched
        let mut q = img_u8(2, 1, &[[128, 100, 50, 0], [0, 7, 7, 7]]);
        unpremultiply_pl(&mut q);
        assert_eq!(px_u8(&q, 0, 0), [128, 199, 100, 0]);
        assert_eq!(px_u8(&q, 1, 0), [0, 7, 7, 7]);
    }

    #[test]
    fn the_layered_frame_interpolates_every_layer_on_its_own_and_composes_them() {
        // the traced case size-n1-w1.box1-5x5.d8: one FG layer whose hard
        // matte box moves by one pixel, zero flows, warping code 0, Blend at wA = 0.5; the expected frame is the
        // original's reference frame 1 (host rows)
        let (w, h) = (5usize, 5usize);
        let rows_a: [[[u8; 3]; 5]; 5] = [
            [[235, 53, 77], [232, 60, 77], [209, 66, 77], [200, 210, 169], [113, 178, 144]],
            [[235, 64, 89], [227, 47, 89], [211, 50, 89], [139, 211, 107], [74, 217, 99]],
            [[231, 50, 102], [239, 37, 102], [214, 48, 102], [111, 181, 61], [50, 205, 52]],
            [[221, 40, 62], [190, 69, 59], [132, 108, 53], [68, 129, 64], [26, 161, 74]],
            [[219, 52, 62], [168, 39, 68], [100, 51, 78], [32, 75, 118], [60, 99, 128]],
        ];
        let rows_b: [[[u8; 3]; 5]; 5] = [
            [[118, 212, 200], [202, 210, 201], [216, 203, 182], [200, 210, 169], [113, 178, 144]],
            [[183, 159, 170], [235, 53, 77], [232, 60, 77], [209, 66, 77], [74, 217, 99]],
            [[215, 93, 117], [235, 64, 89], [227, 47, 89], [211, 50, 89], [50, 205, 52]],
            [[221, 40, 62], [231, 50, 102], [239, 37, 102], [214, 48, 102], [26, 161, 74]],
            [[219, 52, 62], [168, 39, 68], [100, 51, 78], [32, 75, 118], [60, 99, 128]],
        ];
        let expect: [[[u8; 3]; 5]; 5] = [
            [[177, 133, 139], [217, 135, 139], [213, 134, 130], [200, 210, 169], [113, 178, 144]],
            [[209, 111, 130], [231, 50, 83], [222, 55, 83], [174, 138, 92], [74, 217, 99]],
            [[223, 71, 109], [237, 51, 96], [221, 48, 96], [161, 115, 75], [50, 205, 52]],
            [[221, 40, 62], [211, 59, 80], [186, 73, 77], [141, 88, 83], [26, 161, 74]],
            [[219, 52, 62], [168, 39, 68], [100, 51, 78], [32, 75, 118], [60, 99, 128]],
        ];
        let to_img = |rows: &[[[u8; 3]; 5]; 5]| {
            let px: Vec<[u8; 4]> = rows.iter().flat_map(|r| r.iter().map(|c| [255, c[0], c[1], c[2]])).collect();
            img_u8(w, h, &px)
        };
        let boxm = |x0: usize, y0: usize| Matte {
            width: w,
            height: h,
            data: (0..w * h).map(|i| if (x0..x0 + 3).contains(&(i % w)) && (y0..y0 + 3).contains(&(i / w)) { 1.0 } else { 0.0 }).collect(),
        };
        // the vector path runs on bottom-up rows (fgbg-layers fix 20)
        let (a, b) = (to_img(&rows_a).rows_reversed(), to_img(&rows_b).rows_reversed());
        let mattes = (vec![boxm(0, 0).rows_reversed()], vec![boxm(1, 1).rows_reversed()]);
        let zero = FlowField::new(w, h);
        let fl = Flows { f_ab: Some(zero.clone()), f_ba: Some(zero) };
        let flows = vec![fl.clone(), fl];
        let sel = Selection { s0: 0.5, s1: 1.0, a: 0, b: 1, wa: 0.5, exact: false, vectors_quality: 4, interp: 1 };
        let mut s = settings(w, h, Fmt::U8);
        s.stages = AboveFirstTaps::wrap(s.stages.clone());
        let lp = LayerParams { matte_chan: 1, sensitivity: 0.7, shrink: 0.0, track: true };
        let mut job = LayerJob::for_display(&[lp], lp, 1);
        job.pl_kernels = true;
        job.plates = true;
        let got = compose_layers(&s, &sel, &job, &a, &b, &mattes, &flows).unwrap().rows_reversed();
        for y in 0..h {
            for x in 0..w {
                let e = expect[y][x];
                assert_eq!(px_u8(&got, x, y), [255, e[0], e[1], e[2]], "({x}, {y})");
            }
        }
        // composing each side first and blending the sides (plates off) rounds the OVER otherwise: (0, 1) G is 112
        job.plates = false;
        let old = compose_layers(&s, &sel, &job, &a, &b, &mattes, &flows).unwrap().rows_reversed();
        assert_eq!(px_u8(&old, 0, 1)[2], 112);
    }
}
