//! The frame plan and the pre-render checkout list of plugin 01.
//!
//! The plan names the source frames a render fetches ([`plan`]); the checkout list adds the layer
//! probes and the other checkouts, in the order the pre-render issues them ([`checkouts`]).

use crate::timeline::{
    cvtt_i32_f32, exact_position, frame_positions, marker_override, source_display_time, TimeMap,
};
use fx_core::{Checkout, Result};

/// Layer id of "Alt Source A". The input layer is `None`.
pub const LAYER_ALT_A: &str = "alt_source_a";
/// Layer id of "Alt Source B", the separate tracking source.
pub const LAYER_ALT_B: &str = "alt_source_b";

/// The `type` of a plan entry. The engine also knows type 2 ("reuse the previous layer"); plugin
/// 01 never emits it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryKind {
    /// 0: the tracking layer (Alt Source B).
    LayerB,
    /// 1: the retimed layer (Alt Source A, or the input layer when A is not set).
    Main,
}

impl EntryKind {
    /// The integer of the kind: 0 for `LayerB`, 1 for `Main`.
    pub fn code(self) -> i32 {
        match self {
            EntryKind::LayerB => 0,
            EntryKind::Main => 1,
        }
    }
}

/// One source frame the render will fetch: its layer kind and integer frame number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanEntry {
    pub kind: EntryKind,
    pub position: i32,
}

/// The frame plan: the entries to fetch and the values derived while making it.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    /// In the order the planner writes them: 2 or 4 entries.
    pub entries: Vec<PlanEntry>,
    /// The planner's `simple` flag: the position is on a frame or blur compensation is off.
    pub simple: bool,
    /// Vectors quality after the segment override.
    pub vectors_quality: i32,
    /// Frame interpolation after the segment override.
    pub interp: i32,
    /// `src_pos(t0)`.
    pub s0: f32,
    /// `src_pos(t1)`.
    pub s1: f32,
}

/// The plan for the output positions `t0` and `t1`: the source positions `x = src_pos(t0)`,
/// `y = src_pos(t1)` (in this order), `f = trunc(x)`; the segment override with the planner's
/// form of the second frame (see [`marker_override`]); `simple` as in [`exact_position`].
/// Entries: `(Main, f), (Main, f+1)` when the overridden vectors quality is 0 or `simple` or
/// there is no layer B (`no_b`); otherwise `(LayerB, f), (LayerB, f+1), (Main, f), (Main, f+1)`.
/// `vectors_quality` and `interp` are the raw values minus one.
pub fn plan(
    map: &dyn TimeMap,
    t0: f32,
    t1: f32,
    vectors_quality: i32,
    interp: i32,
    blur: f32,
    no_b: bool,
) -> Result<Plan> {
    let s0 = map.src_pos(t0)?;
    let s1 = map.src_pos(t1)?;
    let (vq, ip) = marker_override(map, s0, vectors_quality, interp, true)?;
    let simple = exact_position(s0, s1, blur);
    let f = cvtt_i32_f32(s0);
    let g = f.wrapping_add(1);
    let entries = if vq == 0 || simple || no_b {
        vec![PlanEntry { kind: EntryKind::Main, position: f }, PlanEntry { kind: EntryKind::Main, position: g }]
    } else {
        vec![
            PlanEntry { kind: EntryKind::LayerB, position: f },
            PlanEntry { kind: EntryKind::LayerB, position: g },
            PlanEntry { kind: EntryKind::Main, position: f },
            PlanEntry { kind: EntryKind::Main, position: g },
        ]
    };
    Ok(Plan { entries, simple, vectors_quality: vq, interp: ip, s0, s1 })
}

/// Everything the pre-render of plugin 01 reads before it issues checkouts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreRender {
    /// Current time of the output frame (`cur`).
    pub current_time: i32,
    /// Time step in ticks: the next output frame is at `current_time + time_step`.
    pub time_step: i32,
    /// Ticks per layer frame (`lts`, the divisor of the frame positions).
    pub local_time_step: i32,
    /// `Display == Source` (parameter value 2).
    pub display_source: bool,
    /// `Src Time Offset` as 16.16 fixed point, read at `current_time`. Only used for `Display == Source`.
    pub src_time_offset_fixed: i32,
    /// Alt Source A is set and has pixels (the probe of layer 12 gives a non empty size).
    pub alt_a_present: bool,
    /// Alt Source B is set and has pixels (`noB` is false).
    pub alt_b_present: bool,
    /// `Input: Fields` is not None: every plan entry is fetched twice.
    pub input_fields: bool,
    /// Raw vectors quality, 0..=4.
    pub vectors_quality: i32,
    /// Raw frame interpolation, 0..=2.
    pub interp: i32,
    /// Blur Compensation.
    pub blur: f32,
    /// Lists three extra checkouts at `-cur`: the input layer, the source layer, Alt Source B.
    pub register_layers: bool,
}

impl PreRender {
    /// The layer the retimed frames come from: Alt Source A when it has pixels, else the input
    /// layer.
    pub fn src_layer(&self) -> Option<&'static str> {
        if self.alt_a_present {
            Some(LAYER_ALT_A)
        } else {
            None
        }
    }
}

fn checkout(layer: Option<&'static str>, time: i32) -> Checkout {
    Checkout { layer, time: time as i64 }
}

/// The checkouts that follow the plan: one per entry at `frame_time(position)`, and for input
/// fields a second one at `frame_time(position + 0.5)` where the half frame position is
/// `(position as f64 + 0.5) as f32`. Layer: `Main` is the source layer, `LayerB` is Alt Source B.
pub fn plan_checkouts(req: &PreRender, map: &dyn TimeMap, plan: &Plan) -> Vec<Checkout> {
    let src = req.src_layer();
    let mut out = Vec::with_capacity(plan.entries.len() * 2);
    for entry in &plan.entries {
        let layer = match entry.kind {
            EntryKind::Main => src,
            EntryKind::LayerB => Some(LAYER_ALT_B),
        };
        out.push(checkout(layer, map.frame_time(entry.position as f32)));
        if req.input_fields {
            let half = ((entry.position as f64) + 0.5) as f32;
            out.push(checkout(layer, map.frame_time(half)));
        }
    }
    out
}

/// The complete list of layer fetches of plugin 01 for one output frame, in the order the
/// pre-render issues them:
///
/// 1. layer 0 (the input) at `cur`;
/// 2. a size probe of Alt Source A at `cur`;
/// 3. when A has no pixels, a probe of the input layer at `cur` (the source layer is then the
///    input);
/// 4. a size probe of Alt Source B at `cur`;
/// 5. `Display == Source`: one checkout of the source layer at [`source_display_time`], and
///    nothing else;
/// 6. otherwise the plan ([`plan`] with `t0`, `t1` from [`frame_positions`]);
/// 7. when `register_layers` is set, three checkouts at `-cur`: the input layer, the source layer, Alt
///    Source B;
/// 8. the plan checkouts ([`plan_checkouts`]).
///
/// Probes of absent layers are listed too: they resolve to "empty" when the layer is absent.
/// Every fetch is listed once.
pub fn checkouts(req: &PreRender, map: &dyn TimeMap) -> Result<Vec<Checkout>> {
    let mut out = Vec::new();
    let cur = req.current_time;
    let src = req.src_layer();
    out.push(checkout(None, cur));
    out.push(checkout(Some(LAYER_ALT_A), cur));
    if !req.alt_a_present {
        out.push(checkout(None, cur));
    }
    out.push(checkout(Some(LAYER_ALT_B), cur));
    if req.display_source {
        let t = source_display_time(cur, req.time_step, req.src_time_offset_fixed);
        out.push(checkout(src, t));
        return Ok(out);
    }
    let (t0, t1) = frame_positions(cur, req.time_step, req.local_time_step);
    let p = plan(map, t0, t1, req.vectors_quality, req.interp, req.blur, !req.alt_b_present)?;
    if req.register_layers {
        let neg = cur.wrapping_neg();
        out.push(checkout(None, neg));
        out.push(checkout(src, neg));
        out.push(checkout(Some(LAYER_ALT_B), neg));
    }
    out.extend(plan_checkouts(req, map, &p));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timeline::test_support::{fixed, plain_rates, rates, Mock};
    use crate::timeline::{select, RemapTimeMap};

    /// The state of the worked example: output frame 7 of a 30 fps comp (100 ticks per frame),
    /// Speed 100 %, no offset, no fields, no layer A or B, with the extra checkouts at `-cur`.
    fn req() -> PreRender {
        PreRender {
            current_time: 700,
            time_step: 100,
            local_time_step: 100,
            display_source: false,
            src_time_offset_fixed: 0,
            alt_a_present: false,
            alt_b_present: false,
            input_fields: false,
            vectors_quality: 4,
            interp: 1,
            blur: 0.0,
            register_layers: true,
        }
    }

    fn cp(layer: Option<&'static str>, time: i64) -> Checkout {
        Checkout { layer, time }
    }

    const A: Option<&str> = Some(LAYER_ALT_A);
    const B: Option<&str> = Some(LAYER_ALT_B);

    fn list(req: &PreRender, mock: &Mock, r: crate::timeline::Rates) -> Vec<Checkout> {
        checkouts(req, &RemapTimeMap::new(mock, r, req.current_time)).unwrap()
    }

    #[test]
    fn layer_ids_are_the_ones_the_front_end_declares() {
        assert_eq!(LAYER_ALT_A, "alt_source_a");
        assert_eq!(LAYER_ALT_B, "alt_source_b");
        assert_eq!((EntryKind::LayerB.code(), EntryKind::Main.code()), (0, 1));
    }

    #[test]
    fn worked_planner_example() {
        // x = n, f = n, simple: plan (1, 7), (1, 8); pos_to_time(p) = p * 100 (equal rates).
        let got = list(&req(), &Mock::speed(100.0), plain_rates());
        assert_eq!(
            got,
            vec![
                cp(None, 700),  // 1: layer 0 at cur
                cp(A, 700),     // 2: probe of layer 12
                cp(None, 700),  // 3: layer 12 is empty, probe of layer 0
                cp(B, 700),     // 4: probe of layer 13
                cp(None, -700), // 5 to 7: at -cur: layer 0, src (= 0), layer 13
                cp(None, -700),
                cp(B, -700),
                cp(None, 700), // 8 on: plan entries (1, 7) and (1, 8) through pos_to_time
                cp(None, 800),
            ]
        );
    }

    #[test]
    fn alt_source_a_replaces_the_input_and_skips_the_second_probe() {
        let r = PreRender { alt_a_present: true, ..req() };
        assert_eq!(r.src_layer(), A);
        let got = list(&r, &Mock::speed(100.0), plain_rates());
        assert_eq!(
            got,
            vec![
                cp(None, 700),
                cp(A, 700),
                cp(B, 700),
                cp(None, -700),
                cp(A, -700),
                cp(B, -700),
                cp(A, 700),
                cp(A, 800),
            ]
        );
    }

    #[test]
    fn older_hosts_skip_the_three_checkouts_at_minus_cur() {
        let r = PreRender { register_layers: false, ..req() };
        let got = list(&r, &Mock::speed(100.0), plain_rates());
        assert_eq!(got, vec![cp(None, 700), cp(A, 700), cp(None, 700), cp(B, 700), cp(None, 700), cp(None, 800)]);
    }

    #[test]
    fn half_speed_asks_for_the_frames_around_the_position() {
        // Output frame 7 at 50 % is source position 3.5: frames 3 and 4.
        let got = list(&req(), &Mock::speed(50.0), plain_rates());
        assert_eq!(&got[got.len() - 2..], &[cp(None, 300), cp(None, 400)]);
        assert_eq!(got.len(), 9, "layer B is absent, so no tracking layer entries");
    }

    #[test]
    fn layer_b_adds_its_own_entries_when_vectors_will_run() {
        let r = PreRender { alt_b_present: true, ..req() };
        let got = list(&r, &Mock::speed(50.0), plain_rates());
        assert_eq!(
            got,
            vec![
                cp(None, 700),
                cp(A, 700),
                cp(None, 700),
                cp(B, 700),
                cp(None, -700),
                cp(None, -700),
                cp(B, -700),
                cp(B, 300), // (0, 3), (0, 4): layer B first
                cp(B, 400),
                cp(None, 300), // (1, 3), (1, 4)
                cp(None, 400),
            ]
        );
        // On a frame (100 %), with Vectors off, or when a marker disables vectors only layer 1 is fetched.
        assert_eq!(list(&r, &Mock::speed(100.0), plain_rates()).len(), 9);
        let r0 = PreRender { vectors_quality: 0, ..r };
        assert_eq!(list(&r0, &Mock::speed(50.0), plain_rates()).len(), 9);
    }

    #[test]
    fn input_fields_double_every_plan_entry_with_a_half_frame_position() {
        // 24 fps source in a 30 fps comp: x = 0.8 * 7 = 5.6, frames 5 and 6, dur = 125, unequal rates (bias 12.5):
        // 5 -> 637, 5.5 -> 700, 6 -> 762, 6.5 -> 825.
        let r = rates(24.0, 30.0, 100);
        let plain = list(&req(), &Mock::speed(100.0), r);
        assert_eq!(&plain[plain.len() - 2..], &[cp(None, 637), cp(None, 762)]);
        let fields = PreRender { input_fields: true, ..req() };
        let got = list(&fields, &Mock::speed(100.0), r);
        assert_eq!(&got[got.len() - 4..], &[cp(None, 637), cp(None, 700), cp(None, 762), cp(None, 825)]);
        assert_eq!(got.len(), 11);
        // The half position is (p as f64 + 0.5) as f32 handed to the same frame_time.
        let map = RemapTimeMap::new(Mock::speed(100.0), r, 700);
        assert_eq!(map.frame_time(((5i32 as f64) + 0.5) as f32), 700);
    }

    #[test]
    fn display_source_has_the_probes_and_one_checkout() {
        let m = Mock::speed(100.0);
        let r = PreRender { display_source: true, src_time_offset_fixed: fixed(2.5), ..req() };
        let got = checkouts(&r, &RemapTimeMap::new(&m, plain_rates(), 700)).unwrap();
        assert_eq!(got, vec![cp(None, 700), cp(A, 700), cp(None, 700), cp(B, 700), cp(None, 900)]);
        assert!(m.reads().is_empty(), "no time map call in the Source branch");
        let r = PreRender { alt_a_present: true, display_source: true, src_time_offset_fixed: fixed(-1.5), ..req() };
        let got = checkouts(&r, &RemapTimeMap::new(&m, plain_rates(), 700)).unwrap();
        assert_eq!(got, vec![cp(None, 700), cp(A, 700), cp(B, 700), cp(A, 600)]);
    }

    #[test]
    fn plan_entries_follow_the_three_cases() {
        fn map(m: &Mock) -> RemapTimeMap<&Mock> {
            RemapTimeMap::new(m, plain_rates(), 0)
        }
        let m = Mock::speed(50.0);
        let entry = |kind, position| PlanEntry { kind, position };
        // Fractional position, vectors on, layer B present: four entries, layer B first.
        let p = plan(&map(&m), 7.0, 8.0, 4, 1, 0.0, false).unwrap();
        assert_eq!(
            p.entries,
            vec![
                entry(EntryKind::LayerB, 3),
                entry(EntryKind::LayerB, 4),
                entry(EntryKind::Main, 3),
                entry(EntryKind::Main, 4)
            ]
        );
        assert_eq!((p.simple, p.vectors_quality, p.interp, p.s0, p.s1), (false, 4, 1, 3.5, 4.0));
        // No layer B: two entries.
        let p = plan(&map(&m), 7.0, 8.0, 4, 1, 0.0, true).unwrap();
        assert_eq!(p.entries, vec![entry(EntryKind::Main, 3), entry(EntryKind::Main, 4)]);
        // Vectors off: two entries even with layer B.
        let p = plan(&map(&m), 7.0, 8.0, 0, 1, 0.0, false).unwrap();
        assert_eq!(p.entries.len(), 2);
        // On a frame: simple.
        let p = plan(&map(&m), 6.0, 7.0, 4, 1, 0.0, false).unwrap();
        assert!(p.simple);
        assert_eq!(p.entries, vec![entry(EntryKind::Main, 3), entry(EntryKind::Main, 4)]);
        // Blur compensation makes an exact position take the vector plan (s0 = 3, s1 = 3.5).
        let p = plan(&map(&m), 6.0, 7.0, 4, 1, 1.0, false).unwrap();
        assert!(!p.simple);
        assert_eq!(p.entries.len(), 4);
    }

    #[test]
    fn markers_can_switch_the_vectors_off() {
        // Marker value 4 ("No MV, Use Nearest" is popup value 4 -> marker 3) at frame 3.
        let nearest = Mock::speed(50.0).with_marks(|t| if t == 300 { 4 } else { 1 });
        let map = RemapTimeMap::new(&nearest, plain_rates(), 0);
        let p = plan(&map, 7.0, 8.0, 4, 2, 0.0, false).unwrap();
        assert_eq!((p.vectors_quality, p.interp, p.entries.len()), (0, 0, 2));
        // Popup value 5 on both frames: "No MV, Frame Blend".
        let blend = Mock::speed(50.0).with_marks(|_| 5);
        let map = RemapTimeMap::new(&blend, plain_rates(), 0);
        let p = plan(&map, 7.0, 8.0, 4, 0, 0.0, false).unwrap();
        assert_eq!((p.vectors_quality, p.interp, p.entries.len()), (0, 1, 2));
        // A cut between frames 3 and 4 (marker 0 against 1).
        let cut = Mock::speed(50.0).with_marks(|t| if t >= 400 { 2 } else { 1 });
        let map = RemapTimeMap::new(&cut, plain_rates(), 0);
        let p = plan(&map, 7.0, 8.0, 4, 2, 0.0, false).unwrap();
        assert_eq!((p.vectors_quality, p.interp, p.entries.len()), (0, 0, 2));
    }

    #[test]
    fn negative_positions_use_the_planner_form_of_the_second_marker_frame() {
        // Frame Number -0.5: a = 0, the core compares frame 0 with itself, the planner with frame 1.
        let m = Mock::frame_number(-0.5).with_marks(|t| if t >= 100 { 3 } else { 1 });
        let map = RemapTimeMap::new(&m, plain_rates(), 0);
        let p = plan(&map, 1.0, 2.0, 4, 2, 0.0, false).unwrap();
        assert_eq!(p.s0, -0.5);
        assert_eq!((p.vectors_quality, p.interp), (0, 0), "marker(0) = 0 against marker(1) = 2");
        let sel = select(&map, 1.0, 2.0, 4, 2, 0.0).unwrap();
        assert_eq!((sel.a, sel.vectors_quality, sel.interp), (0, 4, 2), "the core sees no cut");
    }

    #[test]
    fn negative_frame_numbers_plan_toward_zero() {
        let m = Mock::frame_number(-2.5);
        let map = RemapTimeMap::new(&m, plain_rates(), 0);
        let p = plan(&map, 1.0, 2.0, 4, 1, 0.0, false).unwrap();
        let positions: Vec<i32> = p.entries.iter().map(|e| e.position).collect();
        assert_eq!(positions, vec![-2, -1, -2, -1]);
        // frame_time truncates toward zero: frame -2 is at -199.5 -> -199 ... in ticks (equal rates, base 0).
        assert_eq!(map.frame_time(-2.0), -199);
        assert_eq!(map.frame_time(-1.0), -99);
    }

    #[test]
    fn time_map_errors_surface_and_the_source_branch_does_not_need_the_map() {
        let m = Mock::speed(100.0);
        m.mode.set(7);
        let map = RemapTimeMap::new(&m, plain_rates(), 0);
        assert!(checkouts(&req(), &map).is_err());
        let r = PreRender { display_source: true, ..req() };
        assert!(checkouts(&r, &map).is_ok());
    }

    #[test]
    fn a_whole_clip_of_checkouts_is_consistent_with_the_selection() {
        // Speed 75 %, 24 fps source in a 25 fps comp (1 tick unit = 1/2500 s, 100 ticks per frame): for each output
        // frame the last two checkouts are the frames the render selects, at the times frame_time gives them.
        let r = rates(24.0, 25.0, 100);
        let m = Mock::speed(75.0);
        let map = RemapTimeMap::new(&m, r, 0);
        for n in 0..60i32 {
            let pre = PreRender { current_time: n * 100, ..req() };
            let got = checkouts(&pre, &map).unwrap();
            let (t0, t1) = crate::timeline::frame_positions(n * 100, 100, 100);
            let sel = select(&map, t0, t1, 4, 1, 0.0).unwrap();
            let tail = &got[got.len() - 2..];
            assert_eq!(tail[0].time, map.frame_time(sel.a as f32) as i64, "frame {n}");
            assert_eq!(tail[1].time, map.frame_time(sel.b as f32) as i64, "frame {n}");
        }
    }
}
