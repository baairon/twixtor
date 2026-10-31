//! The parity gaps register: every open point of the effects, each pinned to one stated rule
//! until a reference render of the original grades it (README: limits).
//!
//! Precision contract: no arithmetic. The entries are static data; the formulas in the doc
//! comments are the rules the engine modules implement, with their own precision contracts.
//!
//! An entry names the open point (the title of the `vfx: note:` line), the controls whose use
//! reaches it, the rule this project runs there (tagged V established, I inferred, U not
//! established) and the reference case family that
//! settles it. The code that runs a rule calls `fx_core::gaps::record(&gaps::G3)` at the point
//! of use; `vfx` prints the note once per gap per run. Each effect lists the gaps it can note
//! ([`CREATE_VECTORS`], [`RETIME_FGBG`], [`VECTORS_IN`]); `retime` has none.
//!
//! Retiring a gap: once every path that runs its rule follows the original (read from its code,
//! or graded bit exact on every reference case that reaches it), its entry, its list memberships,
//! its `record` calls, its reference case family and its Stated rules row in `docs/parity.md` go.
//! G1, G1b to G1h and G2 are retired that way: the layered paths, the vector merge, the vector
//! matte step and the point displacement follow the original.
//!
//! G7 (tiny frames) and the key handle points K1 have no entry: there is nothing inside the
//! plugin to grade them against, so they print no note.

use fx_core::OpenGap;

/// G3 (trigger V, drawing U). Rule: an opaque disc of radius 1.5 times the downsample ratio in
/// the point colour, in point order, later points on top (`geom::draw_points`), where the
/// original draws its antialiased cross. Noted when retime-fgbg's Draw Geom draws a point (the
/// Source display with Alt Color Source set); create-vectors draws the original's cross and
/// notes nothing. Family `gap-drawgeom`.
pub static G3: OpenGap = OpenGap::new("G3", "Draw Geom drawing", &["draw_geom"]);

/// G4 (U). Rule: the Blend view of a one layer group of an FG stack (`display=3`,
/// `debug_view_method=2`, Layers Output FG1 to MAIN_BG with an FG matte) is the plain
/// `lerp(W', Y_k, 0.5)`, with no alpha match; the original runs a step there that was not read.
/// The other views and groups follow the original. Family `gap-debug`.
pub static G4: OpenGap = OpenGap::new("G4", "one-layer Debug Blend view", &["debug_view_method", "layers_output"]);

/// G5 (role V, arithmetic U). Rule: a pull-push fill (`fill_flow_holes`) of the level 0
/// splat of the points, with no extra blur, replaces the flow of a layer whose Track is off.
/// Noted when a layer's Track is off on the Vectors or Debug display. Family `gap-tracker2`.
pub static G5: OpenGap = OpenGap::new("G5", "second tracker arithmetic", &["main_bg_track", "fg1_track", "fg2_track", "fg3_track"]);

/// G6 (structure V, parts U). Rule: with 360 Video the solver, the hole fill and the motion blur
/// wrap horizontally (indices taken modulo the width; the range check tests y only; the blur's
/// bilinear taps on the sphere, READ); the forward warp itself does not wrap. The wrap of the
/// mask pyramids of a layered render and the nearest taps of a draft quality blur are this
/// project's reading (U). Noted for `video_360=1` where a wrapped solve, fill or blur runs.
/// Family `gap-wrap`.
pub static G6: OpenGap = OpenGap::new("G6", "solver, hole fill and blur wrap variants", &["video_360"]);

/// Every gap of the register, in id order.
pub static ALL: &[&OpenGap] = &[&G3, &G4, &G5, &G6];

/// The gaps `create-vectors` can note: the Debug Blend view of a one layer group and the second
/// tracker.
pub static CREATE_VECTORS: &[&OpenGap] = &[&G4, &G5];

/// The gaps `retime-fgbg` can note: the Draw Geom disc and the 360 wrap.
pub static RETIME_FGBG: &[&OpenGap] = &[&G3, &G6];

/// The gaps `vectors-in` can note: none (its layered path follows the original).
pub static VECTORS_IN: &[&OpenGap] = &[];

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(list: &[&OpenGap]) -> Vec<&'static str> {
        list.iter().map(|g| g.id).collect()
    }

    #[test]
    fn the_register_holds_the_open_gaps_with_unique_ids_in_order() {
        assert_eq!(ids(ALL), ["G3", "G4", "G5", "G6"]);
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
        assert_eq!(ids(CREATE_VECTORS), ["G4", "G5"]);
        assert_eq!(ids(RETIME_FGBG), ["G3", "G6"]);
        assert!(VECTORS_IN.is_empty());
    }
}
