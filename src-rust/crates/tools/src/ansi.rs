//! Robust ANSI / VT escape-sequence stripping for terminal command output.
//!
//! Tool output (especially from a PTY) carries color codes, cursor movement,
//! OSC hyperlinks, and charset designators that are pure noise once the text
//! reaches the model. This is the single canonical stripper shared by the
//! output reducers and the PTY bash tool.

use std::borrow::Cow;

/// Remove ANSI/VT escape sequences from `input`, returning the cleaned text.
///
/// Fast path: if `input` contains no ESC byte (`0x1B`), the original slice is
/// returned borrowed with no allocation. Otherwise these sequence forms are
/// stripped, and all other text (including multi-byte Unicode) is preserved:
///
/// - **CSI** — `ESC [`, parameter/intermediate bytes, a final byte in
///   `0x40..=0x7E` (colors, cursor moves, screen clears).
/// - **OSC** — `ESC ]` … terminated by `BEL` (`0x07`) or `ST` (`ESC \`)
///   (e.g. hyperlinks).
/// - **Charset designator** — `ESC` followed by `(`, `)`, `*`, or `+`, then one
///   charset byte.
/// - **Other two-character escapes** — `ESC` plus a single byte.
/// - A lone trailing `ESC` is dropped.
pub fn strip_ansi(input: &str) -> Cow<'_, str> {
    if !input.as_bytes().contains(&0x1b) {
        return Cow::Borrowed(input);
    }

    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\u{1b}' {
            out.push(ch);
            continue;
        }
        // An ESC was found; classify the sequence from the next char.
        match chars.peek() {
            Some('[') => {
                // CSI: consume params/intermediates up to and including the
                // final byte (0x40..=0x7E).
                chars.next();
                for c in chars.by_ref() {
                    if ('\u{40}'..='\u{7e}').contains(&c) {
                        break;
                    }
                }
            }
            Some(']') => {
                // OSC: consume until BEL or ST (ESC \).
                chars.next();
                let mut prev = '\0';
                for c in chars.by_ref() {
                    if c == '\u{07}' || (prev == '\u{1b}' && c == '\\') {
                        break;
                    }
                    prev = c;
                }
            }
            Some('(') | Some(')') | Some('*') | Some('+') => {
                // Charset designator: introducer plus one charset byte.
                chars.next();
                chars.next();
            }
            Some(_) => {
                // Other two-character escape: drop the following byte.
                chars.next();
            }
            None => {
                // Lone trailing ESC: drop it.
            }
        }
    }
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    #[test]
    fn plain_text_is_borrowed_unchanged() {
        let s = "no escapes here";
        let out = strip_ansi(s);
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(out, "no escapes here");
    }

    #[test]
    fn sgr_color_is_removed() {
        assert_eq!(strip_ansi("\u{1b}[31mred\u{1b}[0m"), "red");
    }

    #[test]
    fn multi_param_sgr_is_removed() {
        assert_eq!(strip_ansi("\u{1b}[1;31;40mx\u{1b}[0m"), "x");
    }

    #[test]
    fn cursor_and_clear_sequences_are_removed() {
        assert_eq!(strip_ansi("\u{1b}[2J\u{1b}[Hhi"), "hi");
    }

    #[test]
    fn osc_hyperlink_bel_terminated_is_removed() {
        let input = "\u{1b}]8;;http://example.com\u{07}label\u{1b}]8;;\u{07}";
        assert_eq!(strip_ansi(input), "label");
    }

    #[test]
    fn osc_st_terminated_is_removed() {
        // ESC \ (ST) terminates the OSC title sequence.
        assert_eq!(strip_ansi("\u{1b}]0;window title\u{1b}\\done"), "done");
    }

    #[test]
    fn charset_designator_is_removed() {
        assert_eq!(strip_ansi("\u{1b}(Bok"), "ok");
    }

    #[test]
    fn generic_two_char_escape_is_removed() {
        // ESC M (reverse index) is a two-character escape.
        assert_eq!(strip_ansi("\u{1b}Mnext"), "next");
    }

    #[test]
    fn empty_string() {
        assert_eq!(strip_ansi(""), "");
    }

    #[test]
    fn lone_trailing_escape_is_dropped() {
        assert_eq!(strip_ansi("ab\u{1b}"), "ab");
    }

    #[test]
    fn unicode_text_is_preserved() {
        assert_eq!(strip_ansi("café\u{1b}[0m 日本語"), "café 日本語");
    }
}
