//! The workspace holds code that needs the FFI keyword (see `keyword`) in one file only: the
//! hand written OpenCL bindings of the engine (`plugins/retime-engine/src/device/cl.rs`). This test reads every `.rs` file of the
//! workspace (build output and editor folders aside) and fails on the keyword anywhere else, in
//! code or comments; the crate root's `#![deny(unsafe_code)]` holds the compiler to the same rule.

use std::path::{Path, PathBuf};

/// The keyword, spelled so that this file does not contain it.
fn keyword() -> String {
    ["un", "safe"].concat()
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").canonicalize().unwrap()
}

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
    entries.sort();
    for p in entries {
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if p.is_dir() {
            if name == "target" || name.starts_with('.') {
                continue;
            }
            rs_files(&p, out);
        } else if name.ends_with(".rs") {
            out.push(p);
        }
    }
}

/// Byte offsets of the keyword as a whole word (`unsafe_code` and the like do not count).
fn whole_word_hits(text: &str, word: &str) -> Vec<usize> {
    let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(p) = text[from..].find(word) {
        let at = from + p;
        let end = at + word.len();
        let before = at == 0 || !ident(b[at - 1]);
        let after = end >= b.len() || !ident(b[end]);
        if before && after {
            out.push(at);
        }
        from = end;
    }
    out
}

#[test]
fn only_the_opencl_bindings_contain_the_keyword() {
    let root = workspace_root();
    let allowed = root.join("plugins").join("retime-engine").join("src").join("device").join("cl.rs");
    let mut files = Vec::new();
    for top in ["crates", "plugins", "tools"] {
        let dir = root.join(top);
        if dir.is_dir() {
            rs_files(&dir, &mut files);
        }
    }
    assert!(files.len() > 50, "the walk found the workspace sources ({} files)", files.len());
    assert!(files.contains(&allowed), "the bindings are where this test expects them");
    let word = keyword();
    let mut offenders = Vec::new();
    for f in &files {
        if *f == allowed {
            continue;
        }
        let text = std::fs::read_to_string(f).unwrap();
        for at in whole_word_hits(&text, &word) {
            let line = text[..at].matches('\n').count() + 1;
            offenders.push(format!("{}:{line}", f.strip_prefix(&root).unwrap_or(f).display()));
        }
    }
    assert!(offenders.is_empty(), "`{word}` outside device/cl.rs: {offenders:?}");
    // and the bindings do use it, so the scan is looking at real text
    let bindings = std::fs::read_to_string(&allowed).unwrap();
    assert!(whole_word_hits(&bindings, &word).len() > 10);
}

#[test]
fn the_engine_root_denies_the_lint_and_allows_it_on_the_bindings_only() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let lib = std::fs::read_to_string(src.join("lib.rs")).unwrap();
    assert!(lib.contains(&format!("#![deny({}_code)]", keyword())));
    let device = std::fs::read_to_string(src.join("device.rs")).unwrap();
    let allow = format!("#[allow({}_code)]", keyword());
    assert_eq!(device.matches(&allow).count(), 1);
    assert!(device.contains(&format!("{allow}\npub mod cl;")));
}

#[test]
fn the_word_scan_ignores_longer_identifiers() {
    let w = keyword();
    assert_eq!(whole_word_hits(&format!("{w}_code {w} x{w} {w}2 ({w})"), &w), vec![12, 36]);
}
