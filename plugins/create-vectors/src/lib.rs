//! create-vectors: the vector creation effect ("Create Vectors"). `params()` lists every parameter
//! whose kind `fx-core` can express and `layers()` lists its optional layers.
//!
//! The effect is temporal: it reads the source (the input layer or `Alt Motion Src`) at the frame pair of
//! `Track Frame`, runs the optical flow solver of the shared engine and OUTPUTS the vector field as
//! an encoded RGBA image (`retime_engine::vectors`: red `dx`, green `dy`, centred and scaled by
//! `Max Displace`, blue 0, alpha opaque), the frame plugins 03 and 04 read as a vector layer. With
//! FG matte layers every layer gets its own flow and `Layers Output` picks the layers merged into
//! the image; tracking points, the second tracker, the Debug views and the Draw Geom overlay are
//! in `glue.rs`.

use fx_core::{
    Checkout, Effect, EffectKind, Env, Frame, LayerSpec, ParamKind, ParamSpec, Params, RenderContext, Result,
    TemporalContext,
};

pub mod glue;
pub mod prerender;

pub use prerender::{LAYER_ALT_SRC, LAYER_FG_MATTE};

pub struct CreateVectors;

/// Ids and names of the three foreground layer blocks, in order FG1 to FG3.
const FG_SENSITIVITY: [(&str, &str); 3] = [
    ("fg1_sensitivity", "FG1: Sensitivity"),
    ("fg2_sensitivity", "FG2: Sensitivity"),
    ("fg3_sensitivity", "FG3: Sensitivity"),
];
const FG_TRACK: [(&str, &str); 3] = [
    ("fg1_track", "FG1: Image Track"),
    ("fg2_track", "FG2: Image Track"),
    ("fg3_track", "FG3: Image Track"),
];
const FG_MATTE_CHAN: [(&str, &str); 3] = [
    ("fg1_matte_chan", "FG1: Matte Chan"),
    ("fg2_matte_chan", "FG2: Matte Chan"),
    ("fg3_matte_chan", "FG3: Matte Chan"),
];
const FG_INV_MATTE_SHRINK: [(&str, &str); 3] = [
    ("fg1_inv_matte_shrink", "FG1: Inv Matte Shrink"),
    ("fg2_inv_matte_shrink", "FG2: Inv Matte Shrink"),
    ("fg3_inv_matte_shrink", "FG3: Inv Matte Shrink"),
];

/// Ids and names of the twelve track points.
const PT_USE: [(&str, &str); 12] = [
    ("pt1_use", "Pt 1: Use"),
    ("pt2_use", "Pt 2: Use"),
    ("pt3_use", "Pt 3: Use"),
    ("pt4_use", "Pt 4: Use"),
    ("pt5_use", "Pt 5: Use"),
    ("pt6_use", "Pt 6: Use"),
    ("pt7_use", "Pt 7: Use"),
    ("pt8_use", "Pt 8: Use"),
    ("pt9_use", "Pt 9: Use"),
    ("pt10_use", "Pt 10: Use"),
    ("pt11_use", "Pt 11: Use"),
    ("pt12_use", "Pt 12: Use"),
];
const PT_POS: [(&str, &str); 12] = [
    ("pt1_pos", "Pt 1: Pos"),
    ("pt2_pos", "Pt 2: Pos"),
    ("pt3_pos", "Pt 3: Pos"),
    ("pt4_pos", "Pt 4: Pos"),
    ("pt5_pos", "Pt 5: Pos"),
    ("pt6_pos", "Pt 6: Pos"),
    ("pt7_pos", "Pt 7: Pos"),
    ("pt8_pos", "Pt 8: Pos"),
    ("pt9_pos", "Pt 9: Pos"),
    ("pt10_pos", "Pt 10: Pos"),
    ("pt11_pos", "Pt 11: Pos"),
    ("pt12_pos", "Pt 12: Pos"),
];
/// The color of track point N is named "Pt N: Color", as the plugin registers it.
const PT_COLOR: [(&str, &str); 12] = [
    ("pt1_color", "Pt 1: Color"),
    ("pt2_color", "Pt 2: Color"),
    ("pt3_color", "Pt 3: Color"),
    ("pt4_color", "Pt 4: Color"),
    ("pt5_color", "Pt 5: Color"),
    ("pt6_color", "Pt 6: Color"),
    ("pt7_color", "Pt 7: Color"),
    ("pt8_color", "Pt 8: Color"),
    ("pt9_color", "Pt 9: Color"),
    ("pt10_color", "Pt 10: Color"),
    ("pt11_color", "Pt 11: Color"),
    ("pt12_color", "Pt 12: Color"),
];

/// Default point colors (R, G, B).
const PT_DEFAULT_COLORS: [[u8; 3]; 12] = [
    [0x00, 0xff, 0x00],
    [0x00, 0x00, 0xff],
    [0xff, 0xff, 0xff],
    [0xff, 0xff, 0x00],
    [0xff, 0x00, 0xff],
    [0x00, 0xff, 0xff],
    [0xff, 0x80, 0x80],
    [0x80, 0xff, 0x80],
    [0x80, 0x80, 0xff],
    [0xff, 0x80, 0x40],
    [0x80, 0xff, 0x40],
    [0x80, 0x40, 0xff],
];

/// Track point position default: -5.0 on both axes (-327680 in 16.16 fixed point).
const PT_DEFAULT_POS: [f64; 2] = [-5.0, -5.0];

fn popup(id: &'static str, name: &'static str, choices: &[&str], default: u32) -> ParamSpec {
    ParamSpec {
        id,
        name,
        kind: ParamKind::Popup {
            choices: choices.iter().map(|c| c.to_string()).collect(),
            default,
        },
    }
}

fn checkbox(id: &'static str, name: &'static str, default: bool) -> ParamSpec {
    ParamSpec {
        id,
        name,
        kind: ParamKind::Checkbox { default },
    }
}

/// Fixed point slider (16.16) with equal valid and slider ranges and two decimals.
fn fixed_slider(
    id: &'static str,
    name: &'static str,
    min: f64,
    max: f64,
    default: f64,
) -> ParamSpec {
    ParamSpec {
        id,
        name,
        kind: ParamKind::Float {
            valid_min: min,
            valid_max: max,
            slider_min: min,
            slider_max: max,
            default,
            precision: 2,
        },
    }
}

impl Effect for CreateVectors {
    fn id(&self) -> &'static str {
        "create-vectors"
    }

    fn name(&self) -> &'static str {
        "Create Vectors"
    }

    fn params(&self) -> Vec<ParamSpec> {
        let mut p = Vec::with_capacity(11 + 4 * 3 + 3 * 12);

        // Index 3.
        p.push(popup(
            "display",
            "Display",
            &["Source", "Motion Vectors", "Debug"],
            2,
        ));
        // Index 10.
        p.push(popup(
            "vector_resolution",
            "Vector Resolution",
            &["Normal (half res)", "Full res"],
            1,
        ));
        // Index 12, disabled in the UI unless Display is Debug.
        p.push(popup(
            "debug_view_method",
            "Debug View Method",
            &["Warp", "Blend", "Diff", "Detail Compare", "Edge Compare"],
            2,
        ));
        // Index 13.
        p.push(popup(
            "draw_geom",
            "Draw Geom",
            &["None", "All", "FG1", "FG2", "FG3", "Main_BG"],
            2,
        ));
        // Index 15, first parameter of the Track and Output Control group.
        p.push(fixed_slider(
            "max_displace",
            "Max Displace",
            1.0,
            8192.0,
            2048.0,
        ));
        // Index 16.
        p.push(popup(
            "track_frame",
            "Track Frame",
            &["Previous", "Next"],
            1,
        ));
        // Index 17.
        p.push(popup(
            "prep",
            "Image Prep",
            &["None", "Contrast/Edge Enhance", "Delinearize"],
            1,
        ));
        // Index 18.
        p.push(checkbox("track_rgb_a", "Track RGB+A", false));
        // Index 19.
        p.push(popup(
            "layers_output",
            "Layers Output",
            &["All", "FG1", "FG2", "FG3", "MAIN_BG"],
            1,
        ));
        // Index 22 and 23, inside the Main_BG Settings group.
        p.push(fixed_slider(
            "main_bg_sensitivity",
            "Main_BG: Sensitivity",
            0.0,
            100.0,
            70.0,
        ));
        p.push(checkbox("main_bg_track", "Main_BG: Image Track", true));
        // Indices 28 to 32, 37 to 41 and 46 to 50, inside the FG groups.
        for i in 0..3 {
            p.push(fixed_slider(
                FG_SENSITIVITY[i].0,
                FG_SENSITIVITY[i].1,
                0.0,
                100.0,
                70.0,
            ));
            p.push(checkbox(FG_TRACK[i].0, FG_TRACK[i].1, true));
            p.push(popup(
                FG_MATTE_CHAN[i].0,
                FG_MATTE_CHAN[i].1,
                &["Red", "Green", "Blue", "Alpha"],
                1,
            ));
            p.push(fixed_slider(
                FG_INV_MATTE_SHRINK[i].0,
                FG_INV_MATTE_SHRINK[i].1,
                0.0,
                10.0,
                0.0,
            ));
        }
        // Indices 55 to 90, inside the Track Points group.
        for k in 0..12 {
            p.push(popup(
                PT_USE[k].0,
                PT_USE[k].1,
                &["Don't Use", "FG1", "FG2", "FG3", "Main_BG Layer"],
                1,
            ));
            p.push(ParamSpec {
                id: PT_POS[k].0,
                name: PT_POS[k].1,
                kind: ParamKind::Point {
                    default: PT_DEFAULT_POS,
                },
            });
            p.push(ParamSpec {
                id: PT_COLOR[k].0,
                name: PT_COLOR[k].1,
                kind: ParamKind::Color {
                    default: PT_DEFAULT_COLORS[k],
                },
            });
        }
        p
    }

    fn render(
        &self,
        _ctx: &RenderContext,
        _params: &Params,
        input: &Frame,
        output: &mut Frame,
    ) -> Result<()> {
        output.clone_from(input);
        Ok(())
    }

    /// The effect needs other frames of its input: hosts call `render_temporal`.
    fn kind(&self) -> EffectKind {
        EffectKind::Temporal
    }

    /// The four optional layers: `Alt Motion Src` replaces the input as the source (the role of plugin
    /// 01's Alt Source A, so the same id) and the three FG matte layers of the multi layer path
    /// (FG1 first; the matte sizes are probed for the error fills).
    fn layers(&self) -> Vec<LayerSpec> {
        vec![
            LayerSpec { id: LAYER_ALT_SRC, name: "Alt Motion Src", optional: true },
            LayerSpec { id: LAYER_FG_MATTE[0], name: "FG1: Matte", optional: true },
            LayerSpec { id: LAYER_FG_MATTE[1], name: "FG2: Matte", optional: true },
            LayerSpec { id: LAYER_FG_MATTE[2], name: "FG3: Matte", optional: true },
        ]
    }

    /// The open parity gaps of the layer flows, their merge, the points, the overlay, the Debug
    /// views and the second tracker (`retime_engine::gaps`).
    fn gaps(&self) -> &'static [&'static fx_core::OpenGap] {
        retime_engine::gaps::CREATE_VECTORS
    }

    /// The checkouts of output frame `cx.frame_index`, in issue order.
    fn dependencies(&self, cx: &TemporalContext, env: &Env<'_>) -> Result<Vec<Checkout>> {
        glue::dependencies(cx, env)
    }

    /// Display Source (a copy of the source with the Draw Geom overlay), Display Motion Vectors (the encoded vector
    /// image: one flow per layer with the FG matte layers, merged over `Layers Output`, tracking points, the
    /// second tracker of a layer whose Track is off) and Display Debug (the five debug views).
    fn render_temporal(&self, cx: &TemporalContext, env: &Env<'_>, output: &mut Frame) -> Result<()> {
        glue::render_temporal(cx, env, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fx_core::{resolve_params, ParamValue};

    fn find(id: &str) -> ParamSpec {
        CreateVectors
            .params()
            .into_iter()
            .find(|s| s.id == id)
            .unwrap_or_else(|| panic!("missing {id}"))
    }

    #[test]
    fn ids_are_unique_ascii_snake_case() {
        let specs = CreateVectors.params();
        for (i, s) in specs.iter().enumerate() {
            assert!(
                s.id.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "{}",
                s.id
            );
            assert!(
                specs[..i].iter().all(|o| o.id != s.id),
                "duplicate id {}",
                s.id
            );
        }
    }

    #[test]
    fn param_count_and_order() {
        let specs = CreateVectors.params();
        // 11 singles + 3 x 4 FG entries + 12 x 3 points.
        assert_eq!(specs.len(), 11 + 12 + 36);
        let ids: Vec<&str> = specs.iter().map(|s| s.id).collect();
        assert_eq!(
            &ids[..11],
            &[
                "display",
                "vector_resolution",
                "debug_view_method",
                "draw_geom",
                "max_displace",
                "track_frame",
                "prep",
                "track_rgb_a",
                "layers_output",
                "main_bg_sensitivity",
                "main_bg_track",
            ]
        );
        assert_eq!(ids[11], "fg1_sensitivity");
        assert_eq!(ids[14], "fg1_inv_matte_shrink");
        assert_eq!(ids[15], "fg2_sensitivity");
        assert_eq!(ids[19], "fg3_sensitivity");
        assert_eq!(ids[22], "fg3_inv_matte_shrink");
        assert_eq!(ids[23], "pt1_use");
        assert_eq!(ids[24], "pt1_pos");
        assert_eq!(ids[25], "pt1_color");
        assert_eq!(*ids.last().unwrap(), "pt12_color");
    }

    #[test]
    fn display_and_view_popups() {
        assert_eq!(
            find("display").kind,
            ParamKind::Popup {
                choices: vec!["Source".into(), "Motion Vectors".into(), "Debug".into()],
                default: 2
            }
        );
        assert_eq!(
            find("debug_view_method").kind,
            ParamKind::Popup {
                choices: ["Warp", "Blend", "Diff", "Detail Compare", "Edge Compare"]
                    .map(String::from)
                    .to_vec(),
                default: 2
            }
        );
        assert_eq!(
            find("draw_geom").kind,
            ParamKind::Popup {
                choices: ["None", "All", "FG1", "FG2", "FG3", "Main_BG"]
                    .map(String::from)
                    .to_vec(),
                default: 2
            }
        );
        assert_eq!(
            find("vector_resolution").kind,
            ParamKind::Popup {
                choices: vec!["Normal (half res)".into(), "Full res".into()],
                default: 1
            }
        );
    }

    #[test]
    fn tracking_controls() {
        assert_eq!(
            find("max_displace").kind,
            ParamKind::Float {
                valid_min: 1.0,
                valid_max: 8192.0,
                slider_min: 1.0,
                slider_max: 8192.0,
                default: 2048.0,
                precision: 2
            }
        );
        assert_eq!(
            find("track_frame").kind,
            ParamKind::Popup {
                choices: vec!["Previous".into(), "Next".into()],
                default: 1
            }
        );
        assert_eq!(
            find("prep").kind,
            ParamKind::Popup {
                choices: vec![
                    "None".into(),
                    "Contrast/Edge Enhance".into(),
                    "Delinearize".into()
                ],
                default: 1
            }
        );
        assert_eq!(
            find("track_rgb_a").kind,
            ParamKind::Checkbox { default: false }
        );
        assert_eq!(
            find("layers_output").kind,
            ParamKind::Popup {
                choices: ["All", "FG1", "FG2", "FG3", "MAIN_BG"]
                    .map(String::from)
                    .to_vec(),
                default: 1
            }
        );
    }

    #[test]
    fn layer_blocks() {
        let sens = ParamKind::Float {
            valid_min: 0.0,
            valid_max: 100.0,
            slider_min: 0.0,
            slider_max: 100.0,
            default: 70.0,
            precision: 2,
        };
        assert_eq!(find("main_bg_sensitivity").kind, sens);
        assert_eq!(
            find("main_bg_track").kind,
            ParamKind::Checkbox { default: true }
        );
        for n in 1..=3 {
            assert_eq!(find(&format!("fg{n}_sensitivity")).kind, sens);
            assert_eq!(
                find(&format!("fg{n}_track")).kind,
                ParamKind::Checkbox { default: true }
            );
            assert_eq!(
                find(&format!("fg{n}_matte_chan")).kind,
                ParamKind::Popup {
                    choices: ["Red", "Green", "Blue", "Alpha"].map(String::from).to_vec(),
                    default: 1
                }
            );
            assert_eq!(
                find(&format!("fg{n}_inv_matte_shrink")).kind,
                ParamKind::Float {
                    valid_min: 0.0,
                    valid_max: 10.0,
                    slider_min: 0.0,
                    slider_max: 10.0,
                    default: 0.0,
                    precision: 2
                }
            );
        }
    }

    #[test]
    fn track_points() {
        let colors = [
            [0x00, 0xff, 0x00],
            [0x00, 0x00, 0xff],
            [0xff, 0xff, 0xff],
            [0xff, 0xff, 0x00],
            [0xff, 0x00, 0xff],
            [0x00, 0xff, 0xff],
            [0xff, 0x80, 0x80],
            [0x80, 0xff, 0x80],
            [0x80, 0x80, 0xff],
            [0xff, 0x80, 0x40],
            [0x80, 0xff, 0x40],
            [0x80, 0x40, 0xff],
        ];
        for (k, c) in colors.iter().enumerate() {
            let n = k + 1;
            assert_eq!(
                find(&format!("pt{n}_use")).kind,
                ParamKind::Popup {
                    choices: ["Don't Use", "FG1", "FG2", "FG3", "Main_BG Layer"]
                        .map(String::from)
                        .to_vec(),
                    default: 1
                }
            );
            assert_eq!(
                find(&format!("pt{n}_pos")).kind,
                ParamKind::Point {
                    default: [-5.0, -5.0]
                }
            );
            assert_eq!(
                find(&format!("pt{n}_color")).kind,
                ParamKind::Color { default: *c }
            );
        }
    }

    #[test]
    fn defaults_resolve_and_validate() {
        let resolved =
            resolve_params(&CreateVectors, &Params::default()).expect("defaults must be valid");
        assert_eq!(resolved.float("max_displace"), 2048.0);
        assert_eq!(resolved.int("display"), 2);
        assert_eq!(resolved.point("pt3_pos"), [-5.0, -5.0]);
        assert_eq!(resolved.color("pt12_color"), [0x80, 0x40, 0xff]);
        assert!(!resolved.bool("track_rgb_a"));
        assert_eq!(CreateVectors.params()[0].default_value(), ParamValue::Int(2));
    }

    #[test]
    fn identity_stub() {
        assert_eq!(CreateVectors.id(), "create-vectors");
        assert_eq!(CreateVectors.name(), "Create Vectors");
    }
}
