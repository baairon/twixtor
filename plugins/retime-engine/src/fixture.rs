//! Test fixtures are hex text, 64 bytes a line as the generators in `tests/fixtures` write them,
//! so the repository holds no binary files.

/// The bytes of a hex fixture; line breaks between the digits are skipped.
pub(crate) fn hex(text: &str) -> Vec<u8> {
    let digits: Vec<u8> = text.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    assert!(digits.len() % 2 == 0, "a hex fixture holds whole bytes");
    digits.chunks(2).map(|p| nibble(p[0]) << 4 | nibble(p[1])).collect()
}

fn nibble(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        _ => panic!("not a lowercase hex digit: {:?}", c as char),
    }
}
