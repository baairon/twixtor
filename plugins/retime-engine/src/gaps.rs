//! The parity gaps register: every open point of the effects, each pinned to one stated rule
//! until a reference render of the original grades it (README: limits).
//!
//! Precision contract: no arithmetic. The entries are static data; the formulas in the doc
//! comments are the rules the engine modules implement, with their own precision contracts.
//!
//! An entry names the open point (the title of the `vfx: note:` line), the controls whose use
//! reaches it, the rule this project runs there (tagged V established, I inferred, U not
//! established) and the reference case family that
//! settles it. The code that runs a rule calls `fx_core::gaps::record(&gaps::G1)` at the point
//! of use; `vfx` prints the note once per gap per run. Each effect lists the gaps it can note
//! ([`CREATE_VECTORS`], [`RETIME_FGBG`], [`VECTORS_IN`]); `retime` has none.
//!
//! Retiring a gap: once its family is graded bit exact, its entry, its list memberships and its
//! `record` calls go, and its Stated rules row in `docs/parity.md` goes.
//!
//! G7 (tiny frames) and the key handle points K1 have no entry: there is nothing inside the
//! plugin to grade them against, so they print no note.

use fx_core::OpenGap;

/// The `fgN_inv_matte_shrink` sliders of the three FG layers.
const SHRINKS: &[&str] = &["fg1_inv_matte_shrink", "fg2_inv_matte_shrink", "fg3_inv_matte_shrink"];
/// The three FG matte layers.
const MATTES: &[&str] = &["fg1_matte", "fg2_matte", "fg3_matte"];

/// G1 (structure V, tail U). Rule: a choke of 0 is the identity; `shrink(m, n >= 1)` blurs
/// with radius `(1 - fr) + r`, then `clamp((b - T) / (1 - T))` with `T = 1 - (k[n] + k[n+1])`,
/// so a straight edge erodes by exactly n pixels; it chokes the inverse matte only. Noted
/// on every frame rendered with an FG matte layer present, on any display but Source, because
/// the matte structure is shared by all layered paths. Family `gap-shrink`.
pub static G1: OpenGap = OpenGap::new("G1", "matte shrink tail", SHRINKS);

/// G1b. Rule: the block matching sweep is gated by the mask of frame X; the cost itself
/// is unmasked. Noted when an FG matte layer is present on a frame that searches motion.
/// Family `gap-maskbm`.
pub static G1B: OpenGap = OpenGap::new("G1b", "mask-aware block matching cost", MATTES);

/// G1c (intent V, order I/U). Rule: FG1 on top; the tracking matte of layer k is
/// `m_k * prod(1 - m_j, j < k)`, Main_BG's the product of all inverses. Noted with two or more
/// FG matte layers (`nfg >= 2`). Family `gap-order`.
pub static G1C: OpenGap = OpenGap::new("G1c", "layer visibility direction", &["fg2_matte", "fg3_matte"]);

/// G1d (flow V, merge U). Rule: the vector image is the sum of `w_k * v_k` with the over
/// weights (raw mattes, Main_BG 1) over the layers `first..=last` of Layers Output; alpha white,
/// blue 0. Noted by create-vectors with `nfg >= 1`. Family `gap-merge`.
pub static G1D: OpenGap = OpenGap::new("G1d", "create-vectors merge", &["fg1_matte", "fg2_matte", "fg3_matte", "layers_output"]);

/// G1e (flow V, step U). Rule: each per layer load sets alpha to 0 where the layer's
/// effective matte is below 0.95, then decodes as the single layer path does. Noted by
/// vectors-in with `nfg >= 1`. Family `gap-vecmatte`.
pub static G1E: OpenGap = OpenGap::new("G1e", "vectors-in per-layer matte step", MATTES);

/// G1f (not established). Rule: the blur compensation tail of a layered render uses the flow of
/// the last layer of the displayed group. U: no note trigger is established for it; the reading
/// taken here is `nfg >= 1` with Blur Compensation above 0 on a frame that warps. Family
/// `gap-blurflow`.
pub static G1F: OpenGap = OpenGap::new("G1f", "blur tail flow of a layered render", &["blur_compensation"]);

/// G1g (not established, a literal reading of the composition). Rule: the hole fill runs whatever the
/// displayed group, so an FG only view gets flooded by Smart Blend and the forward warps. U: no
/// trigger stated; read as a single group display (3 to 6) with `nfg >= 1` on a warping code
/// that fills. Family `gap-groups`.
pub static G1G: OpenGap = OpenGap::new("G1g", "hole fill of a single group display", &["display", "warping"]);

/// G1h (not established). Rule: the fast path (exact positions, `vectors=1`, segment rows)
/// composes the displayed group too, with no warp and no fill. U: no trigger stated; read as a
/// single group display (3 to 6) with `nfg >= 1` on a fast path frame. Family `gap-groups` at
/// 100% speed.
pub static G1H: OpenGap = OpenGap::new("G1h", "fast path group display", &["display"]);

/// G2 (consumer V, producer U). Rule: a nearest pixel splat per pyramid level,
/// `d += (from - to) * k_l` and `w += 1` in point order; points outside the frame are skipped.
/// Noted when a point in use reaches a solve. Family `gap-points`.
pub static G2: OpenGap = OpenGap::new(
    "G2",
    "tracking point displacement",
    &[
        "pt1_use", "pt2_use", "pt3_use", "pt4_use", "pt5_use", "pt6_use", "pt7_use", "pt8_use", "pt9_use", "pt10_use",
        "pt11_use", "pt12_use",
    ],
);

/// G3 (trigger V, drawing U). Rule: an opaque disc of radius 1.5 times the downsample
/// ratio in the point colour, in point order, later points on top. Noted when Draw Geom draws a
/// point. Family `gap-drawgeom`.
pub static G3: OpenGap = OpenGap::new("G3", "Draw Geom drawing", &["draw_geom"]);

/// G4 (mapping V, four bodies U). Rule: Blend is `blend(W, Y, 0.5)`; Diff `abs(W - Y)`,
/// opaque; Detail Compare an R/G overlay of `clamp((L - blur5(L)) / 255 + 0.5)`; Edge Compare an
/// R/G overlay of `min(abs(grad(blur2(L))) * 0.05, 1)`; with `nfg >= 1` they use the merged
/// flow. Warp (the forward warp at amount 1, no fill) has no open point and notes nothing. Noted
/// for `display=3` with `debug_view_method` other than 1. Family `gap-debug`.
pub static G4: OpenGap = OpenGap::new("G4", "four Debug view compositions", &["debug_view_method"]);

/// G5 (role V, arithmetic U). Rule: a pull-push fill (`fill_flow_holes`) of the level 0
/// splat of the points, with no extra blur, replaces the flow of a layer whose Track is off.
/// Noted when a layer's Track is off on the Vectors or Debug display. Family `gap-tracker2`.
pub static G5: OpenGap = OpenGap::new("G5", "second tracker arithmetic", &["main_bg_track", "fg1_track", "fg2_track", "fg3_track"]);

/// G6 (warp V, rest U). Rule: the solver, hole fill and motion blur wrap horizontally as
/// the inverse warp does (indices taken modulo the width; the range check tests y only); the
/// forward warp itself does not wrap; `track_rgb_a` does not add alpha to the luma. Noted for
/// `track_rgb_a=1` on a frame that searches motion. Family `gap-wrap`.
pub static G6: OpenGap = OpenGap::new("G6", "solver, hole fill and blur wrap variants", &["track_rgb_a"]);

/// Every gap of the register, in id order.
pub static ALL: &[&OpenGap] = &[&G1, &G1B, &G1C, &G1D, &G1E, &G1F, &G1G, &G1H, &G2, &G3, &G4, &G5, &G6];

/// The gaps `create-vectors` can note: one flow per layer, its merge, points, the overlay, the
/// Debug views and the second tracker. No wrap (its Track RGB+A adds alpha to the luma).
pub static CREATE_VECTORS: &[&OpenGap] = &[&G1, &G1B, &G1C, &G1D, &G2, &G3, &G4, &G5];

/// The gaps `retime-fgbg` can note: the layered retime path, points, the overlay and the wrap.
pub static RETIME_FGBG: &[&OpenGap] = &[&G1, &G1B, &G1C, &G1F, &G1G, &G1H, &G2, &G3, &G6];

/// The gaps `vectors-in` can note: the layered retime path over loaded vectors (no solve, so
/// no block matching, points or wrap).
pub static VECTORS_IN: &[&OpenGap] = &[&G1, &G1C, &G1E, &G1F, &G1G, &G1H];

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(list: &[&OpenGap]) -> Vec<&'static str> {
        list.iter().map(|g| g.id).collect()
    }

    #[test]
    fn the_register_holds_g1_to_g6_with_unique_ids_in_order() {
        assert_eq!(ids(ALL), ["G1", "G1b", "G1c", "G1d", "G1e", "G1f", "G1g", "G1h", "G2", "G3", "G4", "G5", "G6"]);
        assert!(ALL.windows(2).all(|w| w[0].id < w[1].id), "ids sort as the register reads");
    }

    #[test]
    fn every_entry_has_a_one_line_title_and_snake_case_controls() {
        for g in ALL {
            let t = g.title;
            assert!(!t.is_empty() && t.is_ascii() && t.trim() == t, "{}: {t:?}", g.id);
            assert!(!t.contains(": ") && !t.contains('\n') && !t.ends_with('.'), "{}: {t:?}", g.id);
            assert!(!g.controls.is_empty(), "{} names no control", g.id);
            for (i, c) in g.controls.iter().enumerate() {
                assert!(!c.is_empty() && c.chars().all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_'), "{}: {c}", g.id);
                assert!(!g.controls[..i].contains(c), "{}: {c} twice", g.id);
            }
        }
    }

    #[test]
    fn the_effect_lists_are_ordered_subsets_that_cover_the_register() {
        let lists = [("create-vectors", CREATE_VECTORS), ("retime-fgbg", RETIME_FGBG), ("vectors-in", VECTORS_IN)];
        for (effect, list) in lists {
            assert!(list.windows(2).all(|w| w[0].id < w[1].id), "{effect}: {:?}", ids(list));
            for g in list.iter() {
                assert!(ALL.iter().any(|r| std::ptr::eq(*r, *g)), "{effect}: {} is not the register's static", g.id);
            }
        }
        for g in ALL {
            assert!(lists.iter().any(|(_, list)| list.iter().any(|x| std::ptr::eq(*x, *g))), "{} belongs to no effect", g.id);
        }
        // the effect specific steps stay with their effect
        assert_eq!(ids(CREATE_VECTORS), ["G1", "G1b", "G1c", "G1d", "G2", "G3", "G4", "G5"]);
        assert_eq!(ids(RETIME_FGBG), ["G1", "G1b", "G1c", "G1f", "G1g", "G1h", "G2", "G3", "G6"]);
        assert_eq!(ids(VECTORS_IN), ["G1", "G1c", "G1e", "G1f", "G1g", "G1h"]);
    }
}
