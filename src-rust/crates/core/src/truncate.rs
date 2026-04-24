//! Text truncation utilities.

/// Return a prefix of `text` with at most `max_bytes` UTF-8 bytes, never splitting a `char` boundary.
pub fn truncate_bytes_prefix<'a>(text: &'a str, max_bytes: usize) -> &'a str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Truncate to a UTF-8-safe prefix of at most `max_bytes` bytes, appending `…` when shortened.
pub fn truncate_bytes_with_ellipsis(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    format!("{}…", truncate_bytes_prefix(text, max_bytes))
}

/// Shrink `s` to at most `max_bytes` UTF-8 bytes without splitting a codepoint (for in-place truncation).
pub fn truncate_string_to_max_bytes(s: &mut String, max_bytes: usize) {
    if s.len() <= max_bytes {
        return;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s.truncate(end);
}

/// Truncate `text` to at most `max_chars` characters.
/// If truncated, appends `… (truncated)`.
pub fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }
    // Find a safe char boundary
    let mut end = max_chars;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}… (truncated)", &text[..end])
}

/// Truncate a list of lines to at most `max_lines`.
/// If truncated, appends a `"… N more lines"` indicator.
pub fn truncate_lines(lines: &[String], max_lines: usize) -> (Vec<String>, bool) {
    if lines.len() <= max_lines {
        return (lines.to_vec(), false);
    }
    let mut out = lines[..max_lines].to_vec();
    let remaining = lines.len() - max_lines;
    out.push(format!(
        "… {} more line{}",
        remaining,
        if remaining == 1 { "" } else { "s" }
    ));
    (out, true)
}

/// Truncate tool output to a safe display length.
/// Returns `(truncated_text, was_truncated)`.
pub fn truncate_tool_output(text: &str, max_chars: usize) -> (String, bool) {
    if text.len() <= max_chars {
        return (text.to_string(), false);
    }
    let mut end = max_chars;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    (
        format!("{}… [{} chars truncated]", &text[..end], text.len() - end),
        true,
    )
}

/// Truncate a file path for display, keeping the filename and shortening the directory.
pub fn truncate_path(path: &str, max_chars: usize) -> String {
    if path.len() <= max_chars {
        return path.to_string();
    }
    let filename = std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path);
    if filename.len() >= max_chars {
        return filename.to_string();
    }
    let prefix_len = max_chars - filename.len() - 4; // 4 for "…/"
    let dir = std::path::Path::new(path)
        .parent()
        .and_then(|p| p.to_str())
        .unwrap_or("");
    if dir.len() <= prefix_len {
        return path.to_string();
    }
    format!("…/{}/{}", &dir[dir.len() - prefix_len..], filename)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_text_short() {
        assert_eq!(truncate_text("hello", 10), "hello");
    }

    #[test]
    fn truncate_text_long() {
        let t = truncate_text("hello world", 5);
        assert!(t.starts_with("hello"));
        assert!(t.contains("truncated"));
    }

    #[test]
    fn truncate_lines_over_limit() {
        let lines: Vec<String> = (0..10).map(|i| format!("line {}", i)).collect();
        let (out, truncated) = truncate_lines(&lines, 5);
        assert!(truncated);
        assert_eq!(out.len(), 6); // 5 lines + 1 indicator
        assert!(out[5].contains("5 more"));
    }

    #[test]
    fn truncate_bytes_prefix_keycap_emoji() {
        // Digit + VS16 + U+20E3 is multiple UTF-8 bytes; slicing at a fixed byte index must not panic.
        let s = "\"0\", \"0\u{fe0f}\u{20e3}\", \"1\"";
        let p = truncate_bytes_prefix(s, s.len());
        assert_eq!(p, s);
        // Force a cut inside the keycap sequence (U+20E3 is 3 bytes in UTF-8).
        for cut in 1..s.len() {
            let _ = truncate_bytes_prefix(s, cut);
        }
    }

    #[test]
    fn truncate_string_to_max_bytes_in_place() {
        let mut s = "hi\u{fe0f}\u{20e3}".to_string();
        truncate_string_to_max_bytes(&mut s, 3);
        assert_eq!(s, "hi");
    }
}
