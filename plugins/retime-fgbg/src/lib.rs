//! retime-fgbg: a time remap and frame interpolation effect with foreground and background handling.
//!
//! It retimes a layer by a speed curve or by an absolute frame number and synthesizes the
//! in-between frames from motion vectors. This crate is its temporal front-end for the SINGLE LAYER
//! use (no FG Matte layer, no user tracking points): `dependencies()` is the pre-render,
//! `render_temporal()` the render on the shared engine, and everything the engine or this front-end
//! cannot do is an `FxError::Unsupported` with a precise message (see `glue.rs`). `params()` lists
//! the visible user parameters that can affect the CPU render output, restricted to the kinds
//! `fx-core` can express; the layers are the `layers()` of the effect.

use fx_core::{
    Checkout, Effect, EffectKind, Env, Frame, LayerSpec, ParamKind, ParamSpec, Params, RenderContext, Result, TemporalContext,
};
use retime_engine::planner::{LAYER_ALT_A, LAYER_ALT_B};

pub mod glue;
pub mod prerender;
pub mod timemap;

pub use glue::WIRING;
pub use prerender::LAYER_FG_MATTE;

pub struct RetimeFgBg;

/// Number of tracked foreground groups (FG1 to FG3).
pub const FG_COUNT: usize = 3;

/// Number of user placed tracking points.
pub const POINT_COUNT: usize = 12;

const FG_CHAN_IDS: [&str; FG_COUNT] = ["fg1_matte_chan", "fg2_matte_chan", "fg3_matte_chan"];
const FG_CHAN_NAMES: [&str; FG_COUNT] = ["FG1 Matte Chan", "FG2 Matte Chan", "FG3 Matte Chan"];
const FG_SENS_IDS: [&str; FG_COUNT] = ["fg1_sensitivity", "fg2_sensitivity", "fg3_sensitivity"];
const FG_SENS_NAMES: [&str; FG_COUNT] = ["FG1 Sensitivity", "FG2 Sensitivity", "FG3 Sensitivity"];
const FG_SHRINK_IDS: [&str; FG_COUNT] =
    ["fg1_inv_matte_shrink", "fg2_inv_matte_shrink", "fg3_inv_matte_shrink"];
const FG_SHRINK_NAMES: [&str; FG_COUNT] =
    ["FG1 Inv Matte Shrink", "FG2 Inv Matte Shrink", "FG3 Inv Matte Shrink"];

pub(crate) const PT_USE_IDS: [&str; POINT_COUNT] = [
    "pt1_use", "pt2_use", "pt3_use", "pt4_use", "pt5_use", "pt6_use", "pt7_use", "pt8_use",
    "pt9_use", "pt10_use", "pt11_use", "pt12_use",
];
const PT_USE_NAMES: [&str; POINT_COUNT] = [
    "Pt 1: Use", "Pt 2: Use", "Pt 3: Use", "Pt 4: Use", "Pt 5: Use", "Pt 6: Use", "Pt 7: Use",
    "Pt 8: Use", "Pt 9: Use", "Pt 10: Use", "Pt 11: Use", "Pt 12: Use",
];
const PT_POS_IDS: [&str; POINT_COUNT] = [
    "pt1_pos", "pt2_pos", "pt3_pos", "pt4_pos", "pt5_pos", "pt6_pos", "pt7_pos", "pt8_pos",
    "pt9_pos", "pt10_pos", "pt11_pos", "pt12_pos",
];
const PT_POS_NAMES: [&str; POINT_COUNT] = [
    "Pt 1: Pos", "Pt 2: Pos", "Pt 3: Pos", "Pt 4: Pos", "Pt 5: Pos", "Pt 6: Pos", "Pt 7: Pos",
    "Pt 8: Pos", "Pt 9: Pos", "Pt 10: Pos", "Pt 11: Pos", "Pt 12: Pos",
];
const PT_COLOR_IDS: [&str; POINT_COUNT] = [
    "pt1_color", "pt2_color", "pt3_color", "pt4_color", "pt5_color", "pt6_color", "pt7_color",
    "pt8_color", "pt9_color", "pt10_color", "pt11_color", "pt12_color",
];
const PT_COLOR_NAMES: [&str; POINT_COUNT] = [
    "Pt 1:", "Pt 2:", "Pt 3:", "Pt 4:", "Pt 5:", "Pt 6:", "Pt 7:", "Pt 8:", "Pt 9:", "Pt 10:",
    "Pt 11:", "Pt 12:",
];

/// Default color of each tracking point.
pub const PT_DEFAULT_COLORS: [[u8; 3]; POINT_COUNT] = [
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [255, 128, 128],
    [128, 255, 128],
    [128, 128, 255],
    [255, 128, 64],
    [128, 255, 64],
    [128, 64, 255],
];

fn popup(id: &'static str, name: &'static str, choices: &[&str], default: u32) -> ParamSpec {
    ParamSpec {
        id,
        name,
        kind: ParamKind::Popup { choices: choices.iter().map(|s| s.to_string()).collect(), default },
    }
}

fn checkbox(id: &'static str, name: &'static str, default: bool) -> ParamSpec {
    ParamSpec { id, name, kind: ParamKind::Checkbox { default } }
}

/// Slider with separate valid and slider ranges.
fn slider(
    id: &'static str,
    name: &'static str,
    valid: (f64, f64),
    slider: (f64, f64),
    default: f64,
    precision: u8,
) -> ParamSpec {
    ParamSpec {
        id,
        name,
        kind: ParamKind::Float {
            valid_min: valid.0,
            valid_max: valid.1,
            slider_min: slider.0,
            slider_max: slider.1,
            default,
            precision,
        },
    }
}

impl Effect for RetimeFgBg {
    fn id(&self) -> &'static str {
        "retime-fgbg"
    }

    fn name(&self) -> &'static str {
        "Retime (FG/BG)"
    }

    fn params(&self) -> Vec<ParamSpec> {
        let mut p = Vec::with_capacity(63);

        p.push(popup(
            "display",
            "Display",
            &["Output", "Source", "FG1", "FG2", "FG3", "Background"],
            1,
        ));
        // Default is All.
        p.push(popup("draw_geom", "Draw Geom", &["None", "All", "FG1", "FG2", "FG3", "Main_BG"], 2));
        p.push(checkbox("video_360", "360 Video", false));
        p.push(popup(
            "src_mark_segments",
            "Src: Mark Segments",
            &["Cut A", "Cut B", "Cut C", "No MV, Use Nearest", "No MV, Frame Blend"],
            1,
        ));
        p.push(popup(
            "input_fields",
            "Input: Fields",
            &["None", "Upper Field First", "Lower Field First"],
            1,
        ));
        p.push(checkbox("in_fps_is_out_fps", "In FPS is Out FPS", true));
        // Float slider; the minimum and the default are f32 values widened to f64.
        p.push(slider(
            "input_frame_rate",
            "Input: Frame Rate",
            (0.1f32 as f64, 1000.0),
            (0.1f32 as f64, 1000.0),
            29.97f32 as f64,
            3,
        ));
        p.push(popup(
            "vectors",
            "Vectors",
            &["No Vectors", "Sloppy", "Medium", "High", "Best"],
            5,
        ));
        p.push(popup(
            "prep",
            "Prep",
            &["None", "Contrast/Edge Enhance", "Delinearize"],
            1,
        ));
        p.push(checkbox("track_rgb_a", "Track RGB+A", false));
        p.push(checkbox("plenty_memory", "Plenty Memory?", true));
        p.push(popup("time_remap_mode", "Time Remap Mode", &["Speed", "Frame Number"], 1));
        // Percent, read as a 16.16 fixed point value.
        p.push(slider("speed_percent", "Speed %", (-10000.0, 10000.0), (-1000.0, 1000.0), 100.0, 3));
        p.push(slider("frame_num", "Frame Num", (-200000.0, 200000.0), (0.0, 2000.0), 0.0, 2));
        // Default is Blend.
        p.push(popup(
            "frame_interp",
            "Frame Interp",
            &["Nearest", "Blend", "Weighted Blend"],
            2,
        ));
        p.push(popup(
            "warping",
            "Warping",
            &["Inverse", "Inverse w/ Smart Blend", "Forward", "Forward, no Smart Blend"],
            1,
        ));
        p.push(slider("blur_compensation", "Blur Compensation", (0.0, 10.0), (0.0, 10.0), 0.0, 2));
        p.push(slider("main_bg_sensitivity", "Main_BG Sensitivity", (0.0, 100.0), (0.0, 100.0), 70.0, 2));

        // FG1 to FG3 groups: the channel, sensitivity and shrink parameters of each.
        for k in 0..FG_COUNT {
            p.push(popup(FG_CHAN_IDS[k], FG_CHAN_NAMES[k], &["Red", "Green", "Blue", "Alpha"], 1));
            p.push(slider(FG_SENS_IDS[k], FG_SENS_NAMES[k], (0.0, 100.0), (0.0, 100.0), 70.0, 2));
            p.push(slider(FG_SHRINK_IDS[k], FG_SHRINK_NAMES[k], (0.0, 10.0), (0.0, 10.0), 0.0, 2));
        }

        // Track Points group: 12 points of (Use, Pos, color).
        for n in 0..POINT_COUNT {
            p.push(popup(
                PT_USE_IDS[n],
                PT_USE_NAMES[n],
                &["Don't Use", "FG 1", "FG 2", "FG3", "Main_BG Layer"],
                1,
            ));
            p.push(ParamSpec {
                id: PT_POS_IDS[n],
                name: PT_POS_NAMES[n],
                kind: ParamKind::Point { default: [-5.0, -5.0] },
            });
            p.push(ParamSpec {
                id: PT_COLOR_IDS[n],
                name: PT_COLOR_NAMES[n],
                kind: ParamKind::Color { default: PT_DEFAULT_COLORS[n] },
            });
        }
        p
    }

    fn render(&self, _ctx: &RenderContext, _params: &Params, input: &Frame, output: &mut Frame) -> Result<()> {
        output.clone_from(input);
        Ok(())
    }

    /// The effect needs other frames of its input: hosts call `render_temporal`.
    fn kind(&self) -> EffectKind {
        EffectKind::Temporal
    }

    /// The five optional layer parameters: Alt Source A replaces the input as the retimed footage,
    /// Alt Source B is the separate source for the vector estimation (the same ids as plugin 01),
    /// and the three FG Matte layers feed the multi layer path, which this front-end does not run
    /// (a set matte layer gives `FxError::Unsupported`, the matte sizes are still probed for the
    /// error fills).
    fn layers(&self) -> Vec<LayerSpec> {
        vec![
            LayerSpec { id: LAYER_ALT_A, name: "Alt Source A", optional: true },
            LayerSpec { id: LAYER_ALT_B, name: "Alt Source B", optional: true },
            LayerSpec { id: LAYER_FG_MATTE[0], name: "FG1 Matte", optional: true },
            LayerSpec { id: LAYER_FG_MATTE[1], name: "FG2 Matte", optional: true },
            LayerSpec { id: LAYER_FG_MATTE[2], name: "FG3 Matte", optional: true },
        ]
    }

    /// The pre-render checkouts of output frame `cx.frame_index`, in issue order.
    fn dependencies(&self, cx: &TemporalContext, env: &Env<'_>) -> Result<Vec<Checkout>> {
        glue::dependencies(cx, env)
    }

    /// Single layer use: Time Remap Mode Speed and Frame Number, Nearest, Blend and Weighted Blend, the
    /// inverse warp (Warping "Inverse") with every Vectors quality, Blur Compensation, Prep and the
    /// Source display. Warping codes 1 to 3, FG Matte layers, user tracking points and Track RGB+A (when
    /// vectors are computed) return `FxError::Unsupported`.
    fn render_temporal(&self, cx: &TemporalContext, env: &Env<'_>, output: &mut Frame) -> Result<()> {
        glue::render_temporal(cx, env, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn spec(id: &str) -> ParamSpec {
        RetimeFgBg.params().into_iter().find(|s| s.id == id).unwrap_or_else(|| panic!("no param {id}"))
    }

    fn popup_of(id: &str) -> (Vec<String>, u32) {
        match spec(id).kind {
            ParamKind::Popup { choices, default } => (choices, default),
            k => panic!("{id} is not a popup: {k:?}"),
        }
    }

    fn float_of(id: &str) -> (f64, f64, f64, f64, f64, u8) {
        match spec(id).kind {
            ParamKind::Float { valid_min, valid_max, slider_min, slider_max, default, precision } => {
                (valid_min, valid_max, slider_min, slider_max, default, precision)
            }
            k => panic!("{id} is not a float: {k:?}"),
        }
    }

    #[test]
    fn identity() {
        assert_eq!(RetimeFgBg.id(), "retime-fgbg");
        assert!(!RetimeFgBg.name().is_empty());
    }

    #[test]
    fn ids_are_unique_snake_case_ascii() {
        let params = RetimeFgBg.params();
        assert_eq!(params.len(), 18 + 3 * FG_COUNT + 3 * POINT_COUNT);
        let mut seen = HashSet::new();
        for s in &params {
            assert!(seen.insert(s.id), "duplicate id {}", s.id);
            assert!(!s.id.is_empty());
            assert!(s.id.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'), "{}", s.id);
            assert!(s.id.is_ascii() && s.name.is_ascii(), "{}", s.id);
            assert!(!s.name.contains('\u{2013}') && !s.name.contains('\u{2014}'));
        }
    }

    #[test]
    fn defaults_validate() {
        for s in RetimeFgBg.params() {
            let v = s.default_value();
            s.check(&v).unwrap_or_else(|e| panic!("{}: default rejected: {e}", s.id));
        }
    }

    #[test]
    fn display_and_draw_geom() {
        let (c, d) = popup_of("display");
        assert_eq!(c.len(), 6);
        assert_eq!(d, 1);
        let (c, d) = popup_of("draw_geom");
        assert_eq!(c, ["None", "All", "FG1", "FG2", "FG3", "Main_BG"]);
        assert_eq!(d, 2);
    }

    #[test]
    fn source_control_group() {
        assert_eq!(spec("video_360").kind, ParamKind::Checkbox { default: false });
        let (c, d) = popup_of("src_mark_segments");
        assert_eq!(c.len(), 5);
        assert_eq!(c[3], "No MV, Use Nearest");
        assert_eq!(d, 1);
        let (c, d) = popup_of("input_fields");
        assert_eq!(c, ["None", "Upper Field First", "Lower Field First"]);
        assert_eq!(d, 1);
        assert_eq!(spec("in_fps_is_out_fps").kind, ParamKind::Checkbox { default: true });
        let (vmin, vmax, smin, smax, def, prec) = float_of("input_frame_rate");
        assert_eq!((vmin, vmax, smin, smax), (0.1f32 as f64, 1000.0, 0.1f32 as f64, 1000.0));
        assert_eq!(def, 29.97f32 as f64);
        assert_eq!(prec, 3);
    }

    #[test]
    fn track_control_group() {
        let (c, d) = popup_of("vectors");
        assert_eq!(c.len(), 5);
        assert_eq!(c[0], "No Vectors");
        assert_eq!(d, 5, "default is Best");
        let (c, d) = popup_of("prep");
        assert_eq!(c, ["None", "Contrast/Edge Enhance", "Delinearize"]);
        assert_eq!(d, 1);
        assert_eq!(spec("track_rgb_a").kind, ParamKind::Checkbox { default: false });
        assert_eq!(spec("plenty_memory").kind, ParamKind::Checkbox { default: true });
    }

    #[test]
    fn output_control_group() {
        let (c, d) = popup_of("time_remap_mode");
        assert_eq!(c, ["Speed", "Frame Number"]);
        assert_eq!(d, 1);
        assert_eq!(float_of("speed_percent"), (-10000.0, 10000.0, -1000.0, 1000.0, 100.0, 3));
        assert_eq!(float_of("frame_num"), (-200000.0, 200000.0, 0.0, 2000.0, 0.0, 2));
        let (c, d) = popup_of("frame_interp");
        assert_eq!(c, ["Nearest", "Blend", "Weighted Blend"]);
        assert_eq!(d, 2, "default is Blend");
        let (c, d) = popup_of("warping");
        assert_eq!(c, ["Inverse", "Inverse w/ Smart Blend", "Forward", "Forward, no Smart Blend"]);
        assert_eq!(d, 1);
        assert_eq!(float_of("blur_compensation"), (0.0, 10.0, 0.0, 10.0, 0.0, 2));
    }

    #[test]
    fn main_bg_and_fg_groups() {
        assert_eq!(float_of("main_bg_sensitivity"), (0.0, 100.0, 0.0, 100.0, 70.0, 2));
        for k in 1..=FG_COUNT {
            let (c, d) = popup_of(&format!("fg{k}_matte_chan"));
            assert_eq!(c, ["Red", "Green", "Blue", "Alpha"]);
            assert_eq!(d, 1);
            assert_eq!(float_of(&format!("fg{k}_sensitivity")), (0.0, 100.0, 0.0, 100.0, 70.0, 2));
            assert_eq!(float_of(&format!("fg{k}_inv_matte_shrink")), (0.0, 10.0, 0.0, 10.0, 0.0, 2));
        }
    }

    #[test]
    fn track_points() {
        for n in 1..=POINT_COUNT {
            let (c, d) = popup_of(&format!("pt{n}_use"));
            assert_eq!(c, ["Don't Use", "FG 1", "FG 2", "FG3", "Main_BG Layer"]);
            assert_eq!(d, 1);
            assert_eq!(spec(&format!("pt{n}_pos")).kind, ParamKind::Point { default: [-5.0, -5.0] });
            assert_eq!(
                spec(&format!("pt{n}_color")).kind,
                ParamKind::Color { default: PT_DEFAULT_COLORS[n - 1] }
            );
        }
        assert_eq!(PT_DEFAULT_COLORS[0], [0, 255, 0]);
        assert_eq!(PT_DEFAULT_COLORS[11], [128, 64, 255]);
    }

    #[test]
    fn parameter_order_is_stable() {
        let ids: Vec<&str> = RetimeFgBg.params().iter().map(|s| s.id).collect();
        let pos = |id: &str| ids.iter().position(|x| *x == id).unwrap();
        assert!(pos("display") < pos("draw_geom"));
        assert!(pos("draw_geom") < pos("input_frame_rate"));
        assert!(pos("input_frame_rate") < pos("vectors"));
        assert!(pos("plenty_memory") < pos("time_remap_mode"));
        assert!(pos("time_remap_mode") < pos("speed_percent"));
        assert!(pos("speed_percent") < pos("frame_num"));
        assert!(pos("blur_compensation") < pos("main_bg_sensitivity"));
        assert!(pos("main_bg_sensitivity") < pos("fg1_matte_chan"));
        assert!(pos("fg3_inv_matte_shrink") < pos("pt1_use"));
        assert!(pos("pt12_color") == ids.len() - 1);
    }

    #[test]
    fn range_checks_follow_valid_range() {
        let s = spec("speed_percent");
        assert!(s.check(&fx_core::ParamValue::Float(-10000.0)).is_ok());
        assert!(s.check(&fx_core::ParamValue::Float(10000.0)).is_ok());
        assert!(s.check(&fx_core::ParamValue::Float(10000.5)).is_err());
        let s = spec("frame_num");
        assert!(s.check(&fx_core::ParamValue::Float(-200000.0)).is_ok());
        assert!(s.check(&fx_core::ParamValue::Float(200001.0)).is_err());
        let s = spec("input_frame_rate");
        assert!(s.check(&fx_core::ParamValue::Float(0.05)).is_err());
        assert!(s.check(&fx_core::ParamValue::Float(23.976)).is_ok());
    }
}
