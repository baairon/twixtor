//! retime: a time remap and frame interpolation effect.
//!
//! It retimes a layer by a speed curve or by an absolute frame number and synthesizes the
//! in-between frames from motion vectors ("Frame Interp", "Speed %", "Time Remap Mode").
//!
//! `params()` lists the parameters of the effect and `layers()` its two optional layers.

use fx_core::{
    Checkout, Effect, EffectKind, Env, Frame, LayerSpec, ParamKind, ParamSpec, Params, RenderContext, Result, TemporalContext,
};
use retime_engine::planner::{LAYER_ALT_A, LAYER_ALT_B};

mod glue;

pub use glue::WIRING;

pub struct Retime;

/// Default of "Input: Frame Rate": the 32 bit float 29.97 widened to a double, not the double
/// 29.97.
pub const INPUT_FRAME_RATE_DEFAULT: f64 = 29.97_f32 as f64;

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

impl Effect for Retime {
    fn id(&self) -> &'static str {
        "retime"
    }

    fn name(&self) -> &'static str {
        "Retime"
    }

    fn params(&self) -> Vec<ParamSpec> {
        vec![
            popup("display", "Display", &["Output", "Source"], 1),
            checkbox("video_360", "360 Video", false),
            // Enabled only while Display is "Source".
            popup(
                "src_mark_segments",
                "Src: Mark Segments",
                &["Cut A", "Cut B", "Cut C", "No MV, Use Nearest", "No MV, Frame Blend"],
                1,
            ),
            popup("input_fields", "Input: Fields", &["None", "Upper Field First", "Lower Field First"], 1),
            checkbox("in_fps_is_out_fps", "In FPS is Out FPS", true),
            // Float slider, enabled only while "In FPS is Out FPS" is off. The bounds are the 32
            // bit floats 0.1 and 1000 widened to doubles.
            slider(
                "input_frame_rate",
                "Input: Frame Rate",
                (0.1_f32 as f64, 1000.0),
                (0.1_f32 as f64, 1000.0),
                INPUT_FRAME_RATE_DEFAULT,
                3,
            ),
            // Fixed point slider, in frames, enabled only in Speed mode.
            slider("src_time_offset", "Src Time Offset", (0.0, 30000.0), (0.0, 30000.0), 0.0, 0),
            popup("vectors", "Vectors", &["No Vectors", "Sloppy", "Medium", "High", "Best"], 5),
            popup("prep", "Prep", &["None", "Contrast/Edge Enhance", "Delinearize"], 1),
            // Fixed point slider, percent.
            slider("sensitivity", "Sensitivity", (0.0, 100.0), (0.0, 100.0), 70.0, 2),
            checkbox("plenty_memory", "Plenty Memory?", true),
            popup("time_remap_mode", "Time Remap Mode", &["Speed", "Frame Number"], 1),
            // Fixed point slider, percent, enabled only in Speed mode.
            slider("speed_percent", "Speed %", (-10000.0, 10000.0), (-1000.0, 1000.0), 100.0, 3),
            // Float slider, enabled only in Frame Number mode.
            slider("frame_num", "Frame Num", (-200000.0, 200000.0), (0.0, 2000.0), 0.0, 2),
            popup("frame_interp", "Frame Interp", &["Nearest", "Blend", "Weighted Blend"], 2),
            checkbox("smart_blend", "Smart Blend", false),
            // Fixed point slider.
            slider("blur_compensation", "Blur Compensation", (0.0, 10.0), (0.0, 10.0), 0.0, 2),
        ]
    }

    fn render(&self, _ctx: &RenderContext, _params: &Params, input: &Frame, output: &mut Frame) -> Result<()> {
        output.clone_from(input);
        Ok(())
    }

    /// The effect needs other frames of its input: hosts call `render_temporal`.
    fn kind(&self) -> EffectKind {
        EffectKind::Temporal
    }

    /// The two optional layers: layer A replaces the input as the retimed footage, layer B is the
    /// separate source for the vector estimation.
    fn layers(&self) -> Vec<LayerSpec> {
        vec![
            LayerSpec { id: LAYER_ALT_A, name: "Alt Source A", optional: true },
            LayerSpec { id: LAYER_ALT_B, name: "Alt Source B", optional: true },
        ]
    }

    /// The layer checkouts of output frame `cx.frame_index`, in issue order.
    fn dependencies(&self, cx: &TemporalContext, env: &Env<'_>) -> Result<Vec<Checkout>> {
        glue::dependencies(cx, env)
    }

    /// Time Remap Mode Speed and Frame Number with Nearest, Blend and Weighted Blend. Exact frame
    /// positions and Vectors set to No Vectors take the frame blend path; a position between frames
    /// with Vectors on (the default, Best) takes the optical flow path (flow solver, inverse warps,
    /// Smart Blend when its checkbox is on, Blur Compensation). Frames below the solver's 2 x 2
    /// working size return `FxError::Unsupported` with the reason; there is no silent fallback to
    /// the frame blend.
    fn render_temporal(&self, cx: &TemporalContext, env: &Env<'_>, output: &mut Frame) -> Result<()> {
        glue::render_temporal(cx, env, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fx_core::{resolve_params, ParamValue};

    fn find(id: &str) -> ParamSpec {
        Retime.params().into_iter().find(|p| p.id == id).unwrap_or_else(|| panic!("missing {id}"))
    }

    #[test]
    fn identity() {
        assert_eq!(Retime.id(), "retime");
        assert!(!Retime.name().is_empty());
    }

    #[test]
    fn order_and_unique_ids() {
        let ids: Vec<&str> = Retime.params().iter().map(|p| p.id).collect();
        assert_eq!(
            ids,
            [
                "display",
                "video_360",
                "src_mark_segments",
                "input_fields",
                "in_fps_is_out_fps",
                "input_frame_rate",
                "src_time_offset",
                "vectors",
                "prep",
                "sensitivity",
                "plenty_memory",
                "time_remap_mode",
                "speed_percent",
                "frame_num",
                "frame_interp",
                "smart_blend",
                "blur_compensation",
            ]
        );
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len());
        assert!(ids.iter().all(|id| id.is_ascii() && !id.contains(' ') && *id == id.to_lowercase()));
    }

    #[test]
    fn popups_match_the_binary() {
        let check = |id: &str, n: usize, default: u32, first: &str, last: &str| {
            match find(id).kind {
                ParamKind::Popup { choices, default: d } => {
                    assert_eq!(choices.len(), n, "{id}");
                    assert_eq!(d, default, "{id}");
                    assert_eq!(choices.first().map(String::as_str), Some(first), "{id}");
                    assert_eq!(choices.last().map(String::as_str), Some(last), "{id}");
                }
                other => panic!("{id} is {other:?}"),
            }
        };
        check("display", 2, 1, "Output", "Source");
        check("src_mark_segments", 5, 1, "Cut A", "No MV, Frame Blend");
        check("input_fields", 3, 1, "None", "Lower Field First");
        check("vectors", 5, 5, "No Vectors", "Best");
        check("prep", 3, 1, "None", "Delinearize");
        check("time_remap_mode", 2, 1, "Speed", "Frame Number");
        check("frame_interp", 3, 2, "Nearest", "Weighted Blend");
    }

    #[test]
    fn checkbox_defaults() {
        for (id, default) in [
            ("video_360", false),
            ("in_fps_is_out_fps", true),
            ("plenty_memory", true),
            ("smart_blend", false),
        ] {
            assert_eq!(find(id).kind, ParamKind::Checkbox { default }, "{id}");
        }
    }

    #[test]
    fn slider_ranges_and_defaults() {
        let float = |id: &str| match find(id).kind {
            ParamKind::Float { valid_min, valid_max, slider_min, slider_max, default, precision } => {
                (valid_min, valid_max, slider_min, slider_max, default, precision)
            }
            other => panic!("{id} is {other:?}"),
        };
        assert_eq!(float("input_frame_rate"), (0.1_f32 as f64, 1000.0, 0.1_f32 as f64, 1000.0, 29.969999313354492, 3));
        assert_eq!(float("src_time_offset"), (0.0, 30000.0, 0.0, 30000.0, 0.0, 0));
        assert_eq!(float("sensitivity"), (0.0, 100.0, 0.0, 100.0, 70.0, 2));
        assert_eq!(float("speed_percent"), (-10000.0, 10000.0, -1000.0, 1000.0, 100.0, 3));
        assert_eq!(float("frame_num"), (-200000.0, 200000.0, 0.0, 2000.0, 0.0, 2));
        assert_eq!(float("blur_compensation"), (0.0, 10.0, 0.0, 10.0, 0.0, 2));
    }

    #[test]
    fn frame_rate_default_is_the_widened_float() {
        assert_eq!(INPUT_FRAME_RATE_DEFAULT.to_bits(), 0x403d_f851_e000_0000);
    }

    #[test]
    fn defaults_resolve_and_ranges_are_consistent() {
        let resolved = resolve_params(&Retime, &Params::default()).expect("defaults must validate");
        assert_eq!(resolved.int("frame_interp"), 2);
        assert_eq!(resolved.int("vectors"), 5);
        assert!(resolved.bool("in_fps_is_out_fps"));
        assert_eq!(resolved.float("speed_percent"), 100.0);
        for p in Retime.params() {
            if let ParamKind::Float { valid_min, valid_max, slider_min, slider_max, default, .. } = p.kind {
                assert!(valid_min <= slider_min && slider_max <= valid_max, "{}", p.id);
                assert!(valid_min <= default && default <= valid_max, "{}", p.id);
            }
            assert!(matches!(p.default_value(), ParamValue::Float(_) | ParamValue::Int(_) | ParamValue::Bool(_)));
        }
    }
}
