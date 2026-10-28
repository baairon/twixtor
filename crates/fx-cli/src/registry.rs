//! Plugin families known to `vfx`.
//!
//! Adding a family is one dependency line in Cargo.toml plus one line in
//! [`FAMILIES`]. A family with a single crate can point `effects` straight at
//! a `pub fn effects() -> Vec<Box<dyn Effect>>` exported by that crate.

use fx_core::Effect;

pub struct Family {
    pub name: &'static str,
    /// A hidden family stays out of `vfx list` unless `--all` is passed. Its
    /// effects still run with `vfx apply`.
    pub hidden: bool,
    pub effects: fn() -> Vec<Box<dyn Effect>>,
}

pub const FAMILIES: &[Family] = &[
    Family { name: "retime", hidden: false, effects: retime_family },
    Family { name: "selftest", hidden: true, effects: crate::selftest::effects },
];

fn retime_family() -> Vec<Box<dyn Effect>> {
    vec![
        Box::new(retime::Retime),
        Box::new(create_vectors::CreateVectors),
        Box::new(retime_fgbg::RetimeFgBg),
        Box::new(vectors_in::VectorsIn),
    ]
}

pub struct Entry {
    pub family: &'static str,
    pub hidden: bool,
    pub effect: Box<dyn Effect>,
}

/// Every registered effect, hidden ones included, in family order, then registration order.
pub fn all() -> Vec<Entry> {
    FAMILIES
        .iter()
        .flat_map(|f| (f.effects)().into_iter().map(|effect| Entry { family: f.name, hidden: f.hidden, effect }))
        .collect()
}

/// The effects `vfx list` shows by default.
pub fn listed() -> Vec<Entry> {
    all().into_iter().filter(|e| !e.hidden).collect()
}

/// Looks an effect up by id, hidden ones included.
pub fn find(id: &str) -> Option<Entry> {
    all().into_iter().find(|e| e.effect.id() == id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fx_core::EffectKind;
    use std::collections::HashSet;

    #[test]
    fn ids_are_unique_and_cli_safe() {
        let entries = all();
        assert!(!entries.is_empty());
        let mut seen = HashSet::new();
        for e in &entries {
            let id = e.effect.id();
            assert!(seen.insert(id), "duplicate effect id `{id}`");
            assert!(
                id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
                "effect id `{id}` is not safe on a command line or in a file name"
            );
        }
    }

    #[test]
    fn param_ids_are_unique_per_effect() {
        for e in all() {
            let mut seen = HashSet::new();
            for p in e.effect.params() {
                assert!(seen.insert(p.id), "{}: duplicate param id `{}`", e.effect.id(), p.id);
            }
        }
    }

    #[test]
    fn layer_ids_are_unique_and_snake_case() {
        for e in all() {
            let mut seen = HashSet::new();
            for l in e.effect.layers() {
                assert!(seen.insert(l.id), "{}: duplicate layer id `{}`", e.effect.id(), l.id);
                assert!(
                    !l.id.is_empty() && l.id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
                    "{}: layer id `{}` is not snake_case",
                    e.effect.id(),
                    l.id
                );
            }
            if e.effect.kind() == EffectKind::Frame {
                assert!(e.effect.layers().is_empty(), "{}: a frame effect has no layers", e.effect.id());
            }
        }
    }

    #[test]
    fn find_matches_by_id() {
        let first = all().remove(0);
        let id = first.effect.id();
        assert_eq!(find(id).map(|e| e.effect.id()), Some(id));
        assert!(find("no-such-effect").is_none());
    }

    #[test]
    fn hidden_effects_run_but_are_not_listed() {
        let hidden: Vec<_> = all().into_iter().filter(|e| e.hidden).map(|e| e.effect.id()).collect();
        for want in ["selftest-invert", "selftest-shift", "selftest-integrate", "selftest-threads", "selftest-gap"] {
            assert!(hidden.contains(&want), "{want} should be hidden");
            assert!(find(want).is_some());
            assert!(listed().iter().all(|e| e.effect.id() != want));
        }
        assert!(listed().iter().any(|e| e.effect.id() == "retime"));
        assert!(all().iter().filter(|e| e.hidden).all(|e| e.family == "selftest"));
    }
}
