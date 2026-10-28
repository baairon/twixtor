//! Open parity gaps: the points where an effect runs a rule of this project because the
//! original's behaviour there is not known, and the record of which of them a render used.
//!
//! Precision contract: no arithmetic. A gap is static data (an id, a one line title and the ids
//! of the controls that reach it); the record is a set of references to such statics, and
//! nothing here touches a pixel or a parameter value.
//!
//! The register of the effects' gaps lives with the engine (`retime_engine::gaps`). The code that
//! runs an open rule calls [`record`] at the point of use, from any thread, as often as it likes;
//! a host collects with [`take_recorded`] after each frame and tells the user about the new ones
//! (`vfx` prints one `vfx: note:` line per gap per run on stderr). A library never prints.
//!
//! The record is process wide. Only this module's own tests may assert on its contents: tests
//! of other crates run in parallel threads of one process, and any of them may record or take
//! at any moment, so they check notes through the `vfx` binary (one process per run) instead.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// One open point of the parity register. Build it with [`OpenGap::new`] as a `static`, and pass
/// that static to [`record`]: the record keeps a reference to it.
pub struct OpenGap {
    /// The register id, `G1`, `G1b`, ... as its Stated rules row in `docs/parity.md` names it.
    pub id: &'static str,
    /// One line naming the open point; it holds no `": "`, so a note line splits cleanly.
    pub title: &'static str,
    /// The param and layer ids, as `vfx list --json` names them, whose use reaches the rule.
    pub controls: &'static [&'static str],
    /// True from the first [`record`] of the gap until the [`take_recorded`] that returns it;
    /// it is only written with the record's lock held.
    pending: AtomicBool,
}

impl OpenGap {
    pub const fn new(id: &'static str, title: &'static str, controls: &'static [&'static str]) -> OpenGap {
        OpenGap { id, title, controls, pending: AtomicBool::new(false) }
    }
}

impl fmt::Debug for OpenGap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenGap").field("id", &self.id).field("title", &self.title).field("controls", &self.controls).finish()
    }
}

/// Two gaps are equal when their id, title and controls are; the record state does not count.
impl PartialEq for OpenGap {
    fn eq(&self, other: &OpenGap) -> bool {
        (self.id, self.title, self.controls) == (other.id, other.title, other.controls)
    }
}

impl Eq for OpenGap {}

/// The gaps recorded since the last [`take_recorded`], each once, in the order first recorded.
static RECORDED: Mutex<Vec<&'static OpenGap>> = Mutex::new(Vec::new());

/// Notes that the code running `gap`'s rule was reached. Cheap once the gap is recorded (one
/// atomic load), so a caller may record per row or per pixel, from any rayon thread.
pub fn record(gap: &'static OpenGap) {
    if gap.pending.load(Ordering::Acquire) {
        return;
    }
    let mut recorded = RECORDED.lock().unwrap_or_else(|p| p.into_inner());
    // checked again under the lock: a racing record of the same gap may have pushed it meanwhile
    if !gap.pending.load(Ordering::Acquire) {
        gap.pending.store(true, Ordering::Release);
        recorded.push(gap);
    }
}

/// The gaps recorded since the last call, deduplicated by id and sorted by id, and an empty
/// record afterwards: a gap recorded again from now on is returned again by the next call.
pub fn take_recorded() -> Vec<&'static OpenGap> {
    let mut taken = {
        let mut recorded = RECORDED.lock().unwrap_or_else(|p| p.into_inner());
        // cleared with the lock held, so that a record racing this call either finds its gap
        // still pending (and this call returns it) or pushes it anew for the next call
        for gap in recorded.iter() {
            gap.pending.store(false, Ordering::Release);
        }
        std::mem::take(&mut *recorded)
    };
    taken.sort_by_key(|g| g.id);
    taken.dedup_by_key(|g| g.id);
    taken
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;
    use std::sync::MutexGuard;

    // ids no other test of this crate records
    static A: OpenGap = OpenGap::new("X1", "first test gap", &["a"]);
    static B: OpenGap = OpenGap::new("X1b", "second test gap", &["b", "c"]);
    static C: OpenGap = OpenGap::new("X2", "third test gap", &[]);
    // a second static under A's id: the record still returns the id once
    static A_AGAIN: OpenGap = OpenGap::new("X1", "first test gap", &["a"]);

    /// The tests below read the process wide record, so they take turns.
    fn alone() -> MutexGuard<'static, ()> {
        static TURN: Mutex<()> = Mutex::new(());
        let guard = TURN.lock().unwrap_or_else(|p| p.into_inner());
        take_recorded();
        guard
    }

    fn ids(gaps: &[&'static OpenGap]) -> Vec<&'static str> {
        gaps.iter().map(|g| g.id).collect()
    }

    #[test]
    fn record_dedups_across_threads() {
        let _turn = alone();
        // plain threads and rayon rows, every gap recorded thousands of times in interleaved order
        std::thread::scope(|s| {
            for t in 0..8 {
                s.spawn(move || {
                    for i in 0..2000 {
                        record(if (i + t) % 3 == 0 { &C } else if i % 2 == 0 { &B } else { &A });
                    }
                });
            }
        });
        (0..4096).into_par_iter().for_each(|i| record(if i % 2 == 0 { &A } else { &B }));
        assert_eq!(ids(&take_recorded()), ["X1", "X1b", "X2"]);
        assert!(take_recorded().is_empty(), "a take empties the record");
        // a gap recorded after a take is returned again
        (0..64).into_par_iter().for_each(|_| record(&B));
        assert_eq!(take_recorded(), [&B]);
        assert!(take_recorded().is_empty());
    }

    #[test]
    fn take_sorts_by_id_and_returns_an_id_once() {
        let _turn = alone();
        for g in [&C, &B, &A_AGAIN, &A, &C] {
            record(g);
        }
        let taken = take_recorded();
        assert_eq!(ids(&taken), ["X1", "X1b", "X2"]);
        assert_eq!(taken[0], &A, "either static of the id, they are equal");
        assert!(take_recorded().is_empty());
        // both statics are clear again
        record(&A);
        record(&A_AGAIN);
        assert_eq!(ids(&take_recorded()), ["X1"]);
    }

    #[test]
    fn a_take_racing_the_records_loses_none() {
        let _turn = alone();
        for _ in 0..300 {
            let (during, ()) = rayon::join(take_recorded, || (0..64).into_par_iter().for_each(|i| record(if i % 2 == 0 { &A } else { &B })));
            let after = take_recorded();
            for list in [&during, &after] {
                let mut seen = ids(list);
                seen.dedup();
                assert_eq!(seen.len(), list.len(), "no id twice in one take: {list:?}");
            }
            // every record finished before the last take: each gap is in one of the two takes
            for g in [&A, &B] {
                assert!(during.contains(&g) || after.contains(&g), "{} was lost", g.id);
            }
        }
    }

    #[test]
    fn a_gap_compares_by_its_text_and_prints_without_its_state() {
        assert_eq!(A, A_AGAIN);
        assert_ne!(A, B);
        let text = format!("{A:?}");
        assert_eq!(text, r#"OpenGap { id: "X1", title: "first test gap", controls: ["a"] }"#);
    }
}
