//! vectors-in: a time remap and frame interpolation effect that takes its motion vectors from
//! layers.
//!
//! It does not estimate motion itself: the vectors come from two layers ("Vectors: Prev", "Vectors:
//! Next") and are applied to a source layer, once per layer when FG Matte layers are set (the
//! n-layer path of the engine). This crate is its temporal front-end: `dependencies()` is the
//! pre-render ([`prerender`]), `render_temporal()` the render on the shared engine with the vectors
//! taken from the two layers ([`glue`], [`vectorin`]). `params()` lists the visible user
//! parameters that can affect the CPU render output; the layers are the `layers()` of the effect.

use fx_core::{
    Checkout, Effect, EffectKind, Env, Frame, LayerSpec, ParamKind, ParamSpec, Params, RenderContext, Result, TemporalContext,
};

pub mod glue;
pub mod prerender;
pub mod timemap;
pub mod vectorin;

pub use prerender::{LAYER_FG_MATTE, LAYER_SOURCE, LAYER_VECTORS_NEXT, LAYER_VECTORS_PREV};

pub struct VectorsIn;

/// Default of "Input: Frame Rate": the 32 bit float 29.97 widened to a double, not the double 29.97.
pub const FRAME_RATE_DEFAULT: f64 = 29.97_f32 as f64;

/// Valid and slider minimum of the frame rate slider: the 32 bit float 0.1 widened to a double.
pub const FRAME_RATE_MIN: f64 = 0.1_f32 as f64;

fn popup(id: &'static str, name: &'static str, choices: &[&str], default: u32) -> ParamSpec {
    ParamSpec {
        id,
        name,
        kind: ParamKind::Popup { choices: choices.iter().map(|c| c.to_string()).collect(), default },
    }
}

fn checkbox(id: &'static str, name: &'static str, default: bool) -> ParamSpec {
    ParamSpec { id, name, kind: ParamKind::Checkbox { default } }
}

fn slider(
    id: &'static str,
    name: &'static str,
    (valid_min, valid_max): (f64, f64),
    (slider_min, slider_max): (f64, f64),
    default: f64,
    precision: u8,
) -> ParamSpec {
    ParamSpec {
        id,
        name,
        kind: ParamKind::Float { valid_min, valid_max, slider_min, slider_max, default, precision },
    }
}

/// One FG block: "FGn Matte Chan" popup and "FGn Inv Matte Shrink" slider. The "FGn Matte" layer is
/// listed by `layers()`.
fn fg_params(n: usize) -> [ParamSpec; 2] {
    let (chan_id, chan_name, shrink_id, shrink_name): (&'static str, &'static str, &'static str, &'static str) =
        match n {
            1 => ("fg1_matte_chan", "FG1 Matte Chan", "fg1_inv_matte_shrink", "FG1 Inv Matte Shrink"),
            2 => ("fg2_matte_chan", "FG2 Matte Chan", "fg2_inv_matte_shrink", "FG2 Inv Matte Shrink"),
            3 => ("fg3_matte_chan", "FG3 Matte Chan", "fg3_inv_matte_shrink", "FG3 Inv Matte Shrink"),
            _ => unreachable!("there are exactly three FG blocks"),
        };
    [
        popup(chan_id, chan_name, &["Red", "Green", "Blue", "Alpha"], 1),
        slider(shrink_id, shrink_name, (0.0, 10.0), (0.0, 10.0), 0.0, 2),
    ]
}

impl Effect for VectorsIn {
    fn id(&self) -> &'static str {
        "vectors-in"
    }

    fn name(&self) -> &'static str {
        "VectorsIn"
    }

    fn params(&self) -> Vec<ParamSpec> {
        let mut v = vec![
            // Value 2 (Source) shows the source layer only.
            popup(
                "display",
                "Display",
                &["Twixtored Output", "Source", "FG1 Twixtored", "FG2 Twixtored", "FG3 Twixtored", "Background Twixtored"],
                1,
            ),
            // Enabled only while Display is "Source".
            popup(
                "src_mark_segments",
                "Src: Mark Segments",
                &["Cut A", "Cut B", "Cut C", "No MV, Use Nearest", "No MV, Frame Blend"],
                1,
            ),
            // Fixed point slider, pixels, scaled by the horizontal downsample ratio before use.
            slider("vectors_max_displace", "Vectors: Max Displace", (0.0, 10000.0), (0.0, 10000.0), 2048.0, 2),
            // Fixed point sliders.
            slider("vec_scale_x", "Vec Scale X", (-10.0, 10.0), (-10.0, 10.0), 1.0, 2),
            slider("vec_scale_y", "Vec Scale Y", (-10.0, 10.0), (-10.0, 10.0), 1.0, 2),
            checkbox("in_fps_is_out_fps", "In FPS is Out FPS", true),
            // Float slider, enabled only while "In FPS is Out FPS" is off. The minimum is a 32 bit
            // float widened to a double.
            slider(
                "input_frame_rate",
                "Input: Frame Rate",
                (FRAME_RATE_MIN, 1000.0),
                (FRAME_RATE_MIN, 1000.0),
                FRAME_RATE_DEFAULT,
                3,
            ),
            popup("time_remap_mode", "Time Remap Mode", &["Speed", "Frame Number"], 1),
            // Fixed point slider, percent, enabled only in Speed mode.
            slider("speed_percent", "Speed %", (-10000.0, 10000.0), (-1000.0, 1000.0), 100.0, 3),
            // Float slider, enabled only in Frame Number mode.
            slider("frame_num", "Frame Num", (-200000.0, 200000.0), (0.0, 2000.0), 0.0, 2),
            // Default is Blend.
            popup("frame_interp", "Frame Interp", &["Nearest", "Blend", "Motion Weighted Blend"], 2),
            // Default is Forward.
            popup("warping", "Warping", &["Inverse", "Forward"], 2),
            // Fixed point slider.
            slider("blur_compensation", "Motion Blur Compensation", (0.0, 10.0), (0.0, 10.0), 0.0, 2),
        ];
        for n in 1..=3 {
            v.extend(fg_params(n));
        }
        v
    }

    fn render(&self, _ctx: &RenderContext, _params: &Params, input: &Frame, output: &mut Frame) -> Result<()> {
        output.clone_from(input);
        Ok(())
    }

    /// The effect needs other frames of its layers: hosts call `render_temporal`.
    fn kind(&self) -> EffectKind {
        EffectKind::Temporal
    }

    /// The layer parameters. `Color Source` (id `source`) is the retimed footage; when it is left out
    /// the input layer plays its part. `Vectors: Prev` is fetched at frame `B` and holds the
    /// displacement from `B` back to `A`, `Vectors: Next` is fetched at frame `A` and holds the
    /// displacement from `A` forward to `B`; at least one of them is needed (with none the render shows the red error
    /// fill), with one the other flow is derived by inversion. The three FG Matte layers feed the
    /// n-layer path (the consecutive layers with pixels from FG1 on; their sizes are probed for the
    /// error fills).
    fn layers(&self) -> Vec<LayerSpec> {
        vec![
            LayerSpec { id: LAYER_SOURCE, name: "Color Source", optional: true },
            LayerSpec { id: LAYER_VECTORS_PREV, name: "Vectors: Prev", optional: true },
            LayerSpec { id: LAYER_VECTORS_NEXT, name: "Vectors: Next", optional: true },
            LayerSpec { id: LAYER_FG_MATTE[0], name: "FG1 Matte", optional: true },
            LayerSpec { id: LAYER_FG_MATTE[1], name: "FG2 Matte", optional: true },
            LayerSpec { id: LAYER_FG_MATTE[2], name: "FG3 Matte", optional: true },
        ]
    }

    /// The open parity gaps of the layered retime path over loaded vectors (`retime_engine::gaps`).
    fn gaps(&self) -> &'static [&'static fx_core::OpenGap] {
        retime_engine::gaps::VECTORS_IN
    }

    /// The pre-render checkouts of output frame `cx.frame_index`, in issue order.
    fn dependencies(&self, cx: &TemporalContext, env: &Env<'_>) -> Result<Vec<Checkout>> {
        glue::dependencies(cx, env)
    }

    /// Time Remap Mode Speed and Frame Number, Nearest, Blend and Motion Weighted Blend, Warping
    /// "Inverse" and "Forward", Motion Blur Compensation, Max Displace and Vec Scale, one or two
    /// vector layers, the FG Matte layers with the Display groups, and the Source display.
    fn render_temporal(&self, cx: &TemporalContext, env: &Env<'_>, output: &mut Frame) -> Result<()> {
        glue::render_temporal(cx, env, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fx_core::{resolve_params, ParamValue};

    fn find(id: &str) -> ParamSpec {
        VectorsIn.params().into_iter().find(|p| p.id == id).unwrap_or_else(|| panic!("missing {id}"))
    }

    fn float(id: &str) -> (f64, f64, f64, f64, f64, u8) {
        match find(id).kind {
            ParamKind::Float { valid_min, valid_max, slider_min, slider_max, default, precision } => {
                (valid_min, valid_max, slider_min, slider_max, default, precision)
            }
            other => panic!("{id} is {other:?}"),
        }
    }

    #[test]
    fn identity() {
        assert_eq!(VectorsIn.id(), "vectors-in");
        assert_eq!(VectorsIn.name(), "VectorsIn");
    }

    #[test]
    fn order_and_unique_ids() {
        let ids: Vec<&str> = VectorsIn.params().iter().map(|p| p.id).collect();
        assert_eq!(
            ids,
            [
                "display",
                "src_mark_segments",
                "vectors_max_displace",
                "vec_scale_x",
                "vec_scale_y",
                "in_fps_is_out_fps",
                "input_frame_rate",
                "time_remap_mode",
                "speed_percent",
                "frame_num",
                "frame_interp",
                "warping",
                "blur_compensation",
                "fg1_matte_chan",
                "fg1_inv_matte_shrink",
                "fg2_matte_chan",
                "fg2_inv_matte_shrink",
                "fg3_matte_chan",
                "fg3_inv_matte_shrink",
            ]
        );
        // the plugin registers no output rate control, so there is none
        assert_eq!(ids.len(), 19);
        // the display names as the plugin registers them
        let name = |id: &str| find(id).name;
        assert_eq!(name("blur_compensation"), "Motion Blur Compensation");
        assert_eq!(name("frame_interp"), "Frame Interp");
        let layers: Vec<(&str, &str)> = VectorsIn.layers().iter().map(|l| (l.id, l.name)).collect();
        assert_eq!(layers[0], ("source", "Color Source"));
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len());
        assert!(ids.iter().all(|id| id.is_ascii() && !id.contains(' ') && *id == id.to_lowercase()));
    }

    #[test]
    fn popups_match_the_binary() {
        let check = |id: &str, n: usize, default: u32, first: &str, last: &str| match find(id).kind {
            ParamKind::Popup { choices, default: d } => {
                assert_eq!(choices.len(), n, "{id}");
                assert_eq!(d, default, "{id}");
                assert_eq!(choices.first().map(String::as_str), Some(first), "{id}");
                assert_eq!(choices.last().map(String::as_str), Some(last), "{id}");
            }
            other => panic!("{id} is {other:?}"),
        };
        check("display", 6, 1, "Twixtored Output", "Background Twixtored");
        let texts = |id: &str| match find(id).kind {
            ParamKind::Popup { choices, .. } => choices,
            other => panic!("{id} is {other:?}"),
        };
        assert_eq!(
            texts("display"),
            ["Twixtored Output", "Source", "FG1 Twixtored", "FG2 Twixtored", "FG3 Twixtored", "Background Twixtored"]
        );
        assert_eq!(texts("frame_interp"), ["Nearest", "Blend", "Motion Weighted Blend"]);
        check("src_mark_segments", 5, 1, "Cut A", "No MV, Frame Blend");
        check("time_remap_mode", 2, 1, "Speed", "Frame Number");
        check("frame_interp", 3, 2, "Nearest", "Motion Weighted Blend");
        check("warping", 2, 2, "Inverse", "Forward");
        for n in 1..=3 {
            check(&format!("fg{n}_matte_chan"), 4, 1, "Red", "Alpha");
        }
    }

    #[test]
    fn checkbox_defaults() {
        assert_eq!(find("in_fps_is_out_fps").kind, ParamKind::Checkbox { default: true });
    }

    #[test]
    fn slider_ranges_and_defaults() {
        let fr_min = 0.1_f32 as f64;
        let fr_def = 29.969999313354492;
        assert_eq!(float("vectors_max_displace"), (0.0, 10000.0, 0.0, 10000.0, 2048.0, 2));
        assert_eq!(float("vec_scale_x"), (-10.0, 10.0, -10.0, 10.0, 1.0, 2));
        assert_eq!(float("vec_scale_y"), (-10.0, 10.0, -10.0, 10.0, 1.0, 2));
        assert_eq!(float("input_frame_rate"), (fr_min, 1000.0, fr_min, 1000.0, fr_def, 3));
        assert_eq!(float("speed_percent"), (-10000.0, 10000.0, -1000.0, 1000.0, 100.0, 3));
        assert_eq!(float("frame_num"), (-200000.0, 200000.0, 0.0, 2000.0, 0.0, 2));
        assert_eq!(float("blur_compensation"), (0.0, 10.0, 0.0, 10.0, 0.0, 2));
        for n in 1..=3 {
            assert_eq!(float(&format!("fg{n}_inv_matte_shrink")), (0.0, 10.0, 0.0, 10.0, 0.0, 2));
        }
    }

    #[test]
    fn frame_rate_default_is_the_widened_float() {
        assert_eq!(FRAME_RATE_DEFAULT.to_bits(), 0x403d_f851_e000_0000);
        assert_eq!(FRAME_RATE_MIN.to_bits(), (f32::from_bits(0x3dcc_cccd) as f64).to_bits());
    }

    #[test]
    fn defaults_resolve_and_ranges_are_consistent() {
        let resolved = resolve_params(&VectorsIn, &Params::default()).expect("defaults must validate");
        assert_eq!(resolved.int("frame_interp"), 2);
        assert_eq!(resolved.int("warping"), 2);
        assert_eq!(resolved.int("display"), 1);
        assert_eq!(resolved.int("time_remap_mode"), 1);
        assert!(resolved.bool("in_fps_is_out_fps"));
        assert_eq!(resolved.float("speed_percent"), 100.0);
        assert_eq!(resolved.float("vectors_max_displace"), 2048.0);
        for p in VectorsIn.params() {
            if let ParamKind::Float { valid_min, valid_max, slider_min, slider_max, default, .. } = p.kind {
                assert!(valid_min <= slider_min && slider_max <= valid_max, "{}", p.id);
                assert!(valid_min <= default && default <= valid_max, "{}", p.id);
            }
            assert!(matches!(p.default_value(), ParamValue::Float(_) | ParamValue::Int(_) | ParamValue::Bool(_)));
        }
    }
}
