//! Diagnostics for failed `old_string` matches in edit tools.
//!
//! When an exact match fails, the model usually mis-copied whitespace, line
//! endings, or indentation. Pointing at the near-miss saves a re-read
//! round-trip and a retry.

/// Diagnose why `old_string` failed to match `content` exactly, returning a
/// hint suitable for appending to the tool error message.
pub fn not_found_hint(content: &str, old_string: &str) -> Option<String> {
    // Line-ending mismatch: the text is present once CRLF/LF differences
    // are ignored.
    let content_lf = content.replace("\r\n", "\n");
    let old_lf = old_string.replace("\r\n", "\n");
    if content_lf.contains(&old_lf) {
        return Some(
            "The text DOES appear in the file but with different line endings \
             (CRLF vs LF). Re-read the file and copy the text exactly."
                .to_string(),
        );
    }

    // Whitespace/indentation mismatch: the same lines exist but differ in
    // leading/trailing/internal whitespace.
    if let Some(line) = whitespace_insensitive_location(&content_lf, &old_lf) {
        return Some(format!(
            "Found a near match at line {} that differs only in whitespace or \
             indentation. Re-read the file and copy the exact text, including \
             all spaces and tabs.",
            line
        ));
    }

    // Pasted Read output: every line starts with a line-number prefix like
    // "  42\t" that is not part of the file.
    if looks_like_line_numbered_paste(&old_lf) {
        return Some(
            "old_string looks like it includes line-number prefixes from Read \
             tool output (e.g. \"42\\t...\"). Strip the numbers and tabs and \
             pass only the file content."
                .to_string(),
        );
    }

    None
}

/// Align an edit's `old`/`new` strings to the file's line endings when a
/// literal match would otherwise fail. Returns the strings to use for matching
/// and replacement. Prefers a literal match so a file with mixed endings (or an
/// intentionally bare-LF region or replacement) is matched and written exactly;
/// only when `old` is absent literally AND the file is CRLF while `old` is
/// bare-LF does it convert both strings to CRLF. Shared by FileEdit and
/// BatchEdit so the two tools agree on what counts as a match.
pub fn align_line_endings(content: &str, old: &str, new: &str) -> (String, String) {
    if content.contains(old) {
        (old.to_string(), new.to_string())
    } else if content.contains("\r\n") && !old.contains("\r\n") && old.contains('\n') {
        (old.replace('\n', "\r\n"), new.replace('\n', "\r\n"))
    } else {
        (old.to_string(), new.to_string())
    }
}

/// True when every non-empty line starts with `<spaces><digits>\t`, the
/// format the Read tool uses for line numbers.
fn looks_like_line_numbered_paste(old: &str) -> bool {
    let mut saw_line = false;
    for line in old.lines() {
        if line.trim().is_empty() {
            continue;
        }
        saw_line = true;
        let rest = line.trim_start_matches(' ');
        let digits = rest.chars().take_while(char::is_ascii_digit).count();
        if digits == 0 || !rest[digits..].starts_with('\t') {
            return false;
        }
    }
    saw_line
}

/// Find the 1-based line where `old` matches `content` when every line is
/// compared with whitespace runs collapsed.
fn whitespace_insensitive_location(content: &str, old: &str) -> Option<usize> {
    fn collapse(s: &str) -> String {
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    let old_lines: Vec<String> = old.lines().map(collapse).collect();
    if old_lines.is_empty() || old_lines.iter().all(String::is_empty) {
        return None;
    }
    let content_lines: Vec<String> = content.lines().map(collapse).collect();
    if content_lines.len() < old_lines.len() {
        return None;
    }
    (0..=content_lines.len() - old_lines.len())
        .find(|&i| content_lines[i..i + old_lines.len()] == old_lines[..])
        .map(|i| i + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_line_ending_mismatch() {
        let content = "fn main() {\r\n    println!(\"hi\");\r\n}\r\n";
        let old = "fn main() {\n    println!(\"hi\");\n}";
        let hint = not_found_hint(content, old).unwrap();
        assert!(hint.contains("line endings"));
    }

    #[test]
    fn detects_indentation_mismatch_with_line_number() {
        let content = "alpha\nbeta\n    gamma delta\nepsilon\n";
        let old = "gamma  delta";
        let hint = not_found_hint(content, old).unwrap();
        assert!(hint.contains("line 3"), "hint was: {hint}");
    }

    #[test]
    fn detects_multiline_whitespace_mismatch() {
        let content = "one\n  two three\n  four\nfive\n";
        let old = "two three\nfour";
        let hint = not_found_hint(content, old).unwrap();
        assert!(hint.contains("line 2"), "hint was: {hint}");
    }

    #[test]
    fn no_hint_when_text_genuinely_absent() {
        assert!(not_found_hint("alpha\nbeta\n", "missing").is_none());
    }

    #[test]
    fn detects_line_numbered_paste() {
        let content = "fn alpha() {}\nfn beta() {}\n";
        let old = "    12\tfn alpha() {}\n    13\tfn beta() {}";
        let hint = not_found_hint(content, old).unwrap();
        assert!(hint.contains("line-number prefixes"), "hint was: {hint}");
    }

    #[test]
    fn plain_code_not_mistaken_for_numbered_paste() {
        // Numeric-looking code lines without the tab are not flagged.
        assert!(not_found_hint("alpha\n", "42 * x\n7 + y").is_none());
    }

    #[test]
    fn no_hint_for_whitespace_only_old_string() {
        assert!(not_found_hint("alpha\nbeta\n", "   \n  ").is_none());
    }
}
