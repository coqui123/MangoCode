//! Shared parsing of a leading `---`-delimited frontmatter block.
//!
//! This logic was previously duplicated across several files.
//! Use [`split_frontmatter`] to extract the frontmatter and body,
//! or [`strip_frontmatter`] to get just the body.

/// Split leading `---`-delimited frontmatter from the body.
///
/// Returns `Some((frontmatter, body))` when `content` starts with `---`
/// and a closing `\n---` marker follows. Both slices are borrowed from
/// `content`; no allocation is performed.
///
/// - `frontmatter` is the text between the opening `---` and the closing
///   `\n---` (exclusive of both delimiters).
/// - `body` is everything after the closing `\n---`, with leading
///   carriage-returns and newlines trimmed.
pub fn split_frontmatter(content: &str) -> Option<(&str, &str)> {
    let after_open = content.strip_prefix("---")?;
    let close = after_open.find("\n---")?;
    let frontmatter = &after_open[..close];
    let body = after_open[close + 4..].trim_start_matches(['\r', '\n']);
    Some((frontmatter, body))
}

/// Return the body with any leading frontmatter removed.
///
/// If `content` contains valid frontmatter the body portion is returned;
/// otherwise the original `content` is returned unchanged.
pub fn strip_frontmatter(content: &str) -> &str {
    split_frontmatter(content)
        .map(|(_, body)| body)
        .unwrap_or(content)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_frontmatter() {
        let input = "---\nname: x\n---\nBODY";
        let (fm, body) = split_frontmatter(input).expect("should parse");
        assert_eq!(fm, "\nname: x");
        assert_eq!(body, "BODY");
    }

    #[test]
    fn no_frontmatter() {
        let input = "just body";
        assert!(split_frontmatter(input).is_none());
        assert_eq!(strip_frontmatter(input), "just body");
    }

    #[test]
    fn opening_but_no_closing() {
        let input = "---\nname: x\nno closing marker";
        assert!(split_frontmatter(input).is_none());
    }

    #[test]
    fn crlf_body() {
        let input = "---\nkey: val\n---\r\nBODY";
        let (fm, body) = split_frontmatter(input).expect("should parse");
        assert_eq!(fm, "\nkey: val");
        assert_eq!(body, "BODY");
    }

    #[test]
    fn strip_returns_body() {
        let input = "---\ntitle: hello\n---\ncontent here";
        assert_eq!(strip_frontmatter(input), "content here");
    }

    #[test]
    fn empty_string() {
        assert!(split_frontmatter("").is_none());
    }
}
