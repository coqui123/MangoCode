pub fn parse_git_diff_new_path(rest: &str) -> Option<String> {
    let rest = rest.trim();
    if rest.is_empty() {
        return None;
    }

    if let Some((_old, new)) = parse_quoted_git_diff_paths(rest) {
        return Some(strip_single_git_diff_prefix(&new).to_string());
    }

    parse_unquoted_git_diff_new_path(rest)
}

pub fn strip_single_git_diff_prefix(path: &str) -> &str {
    path.strip_prefix("a/")
        .or_else(|| path.strip_prefix("b/"))
        .unwrap_or(path)
}

pub fn parse_unified_diff_marker_path(raw: &str) -> Option<String> {
    let path = raw
        .trim()
        .split_once('\t')
        .map_or_else(|| raw.trim(), |(path, _timestamp)| path.trim());
    if path.is_empty() || path == "/dev/null" {
        return None;
    }

    let unquoted = unquote_diff_path(path);
    if unquoted.is_empty() {
        return None;
    }

    Some(strip_single_git_diff_prefix(&unquoted).to_string())
}

fn parse_unquoted_git_diff_new_path(rest: &str) -> Option<String> {
    let mut fallback: Option<String> = None;

    for (index, _) in rest.match_indices(" b/") {
        let old = strip_single_git_diff_prefix(rest[..index].trim());
        let new = strip_single_git_diff_prefix(rest[index + 1..].trim());
        if new.is_empty() {
            continue;
        }

        fallback = Some(new.to_string());

        if old == new {
            return Some(new.to_string());
        }
    }

    fallback
}

fn unquote_diff_path(path: &str) -> String {
    let Some(stripped) = path.strip_prefix('"').and_then(|p| p.strip_suffix('"')) else {
        return path.to_string();
    };

    decode_git_quoted_path_body(stripped)
}

fn decode_git_quoted_path_body(body: &str) -> String {
    let mut bytes = Vec::with_capacity(body.len());
    let mut chars = body.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch != '\\' {
            let mut buf = [0u8; 4];
            bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            continue;
        }

        let Some(escaped) = chars.next() else {
            bytes.push(b'\\');
            break;
        };

        match escaped {
            '0'..='7' => {
                let mut value = escaped as u32 - '0' as u32;
                for _ in 0..2 {
                    let Some(next @ '0'..='7') = chars.peek().copied() else {
                        break;
                    };
                    chars.next();
                    value = value * 8 + (next as u32 - '0' as u32);
                }
                bytes.push((value & 0xff) as u8);
            }
            'a' => bytes.push(0x07),
            'b' => bytes.push(0x08),
            'f' => bytes.push(0x0c),
            'n' => bytes.push(b'\n'),
            'r' => bytes.push(b'\r'),
            't' => bytes.push(b'\t'),
            'v' => bytes.push(0x0b),
            '\\' => bytes.push(b'\\'),
            '"' => bytes.push(b'"'),
            other => {
                let mut buf = [0u8; 4];
                bytes.extend_from_slice(other.encode_utf8(&mut buf).as_bytes());
            }
        }
    }

    String::from_utf8_lossy(&bytes).into_owned()
}

fn parse_quoted_git_diff_paths(rest: &str) -> Option<(String, String)> {
    let (old, remaining) = parse_quoted_token(rest.trim_start())?;
    let (new, remaining) = parse_quoted_token(remaining.trim_start())?;
    if remaining.trim().is_empty() {
        Some((old, new))
    } else {
        None
    }
}

fn parse_quoted_token(input: &str) -> Option<(String, &str)> {
    let body = input.strip_prefix('"')?;
    let mut chars = body.char_indices();
    let mut token = String::new();
    let mut escaped = false;

    for (offset, ch) in chars.by_ref() {
        if escaped {
            token.push(ch);
            escaped = false;
        } else if ch == '\\' {
            token.push(ch);
            escaped = true;
        } else if ch == '"' {
            let remaining = &body[offset + ch.len_utf8()..];
            return Some((decode_git_quoted_path_body(&token), remaining));
        } else {
            token.push(ch);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_git_diff_new_path_handles_simple_paths() {
        assert_eq!(
            parse_git_diff_new_path("a/src/foo.rs b/src/foo.rs").as_deref(),
            Some("src/foo.rs")
        );
    }

    #[test]
    fn parse_git_diff_new_path_preserves_paths_with_spaces() {
        assert_eq!(
            parse_git_diff_new_path("a/docs/my file.md b/docs/my file.md").as_deref(),
            Some("docs/my file.md")
        );
    }

    #[test]
    fn parse_git_diff_new_path_preserves_paths_containing_b_prefix_marker() {
        assert_eq!(
            parse_git_diff_new_path("a/docs/foo b/bar.md b/docs/foo b/bar.md").as_deref(),
            Some("docs/foo b/bar.md")
        );
    }

    #[test]
    fn parse_git_diff_new_path_uses_new_path_for_renames() {
        assert_eq!(
            parse_git_diff_new_path("a/old name.md b/new name.md").as_deref(),
            Some("new name.md")
        );
    }

    #[test]
    fn parse_git_diff_new_path_uses_last_candidate_for_ambiguous_renames() {
        assert_eq!(
            parse_git_diff_new_path("a/docs/foo b/bar.md b/docs/new name.md").as_deref(),
            Some("docs/new name.md")
        );
    }

    #[test]
    fn parse_git_diff_new_path_preserves_quoted_paths_with_spaces() {
        assert_eq!(
            parse_git_diff_new_path(r#""a/docs/my file.md" "b/docs/my file.md""#).as_deref(),
            Some("docs/my file.md")
        );
    }

    #[test]
    fn parse_git_diff_new_path_decodes_git_quoted_escapes() {
        assert_eq!(
            parse_git_diff_new_path(r#""a/docs/quote\"tab\t.md" "b/docs/quote\"tab\t.md""#),
            Some("docs/quote\"tab\t.md".to_string())
        );
    }

    #[test]
    fn strip_single_git_diff_prefix_removes_only_one_prefix() {
        assert_eq!(strip_single_git_diff_prefix("b/src/foo.rs"), "src/foo.rs");
        assert_eq!(
            strip_single_git_diff_prefix("b/b/src/foo.rs"),
            "b/src/foo.rs"
        );
    }

    #[test]
    fn parse_unified_diff_marker_path_preserves_paths_with_spaces() {
        assert_eq!(
            parse_unified_diff_marker_path("b/docs/my file.md").as_deref(),
            Some("docs/my file.md")
        );
    }

    #[test]
    fn parse_unified_diff_marker_path_handles_tab_timestamps() {
        assert_eq!(
            parse_unified_diff_marker_path("b/docs/my file.md\t2026-05-24").as_deref(),
            Some("docs/my file.md")
        );
    }

    #[test]
    fn parse_unified_diff_marker_path_preserves_quoted_paths_with_spaces() {
        assert_eq!(
            parse_unified_diff_marker_path(r#""b/docs/my file.md""#).as_deref(),
            Some("docs/my file.md")
        );
    }

    #[test]
    fn parse_unified_diff_marker_path_decodes_git_quoted_escapes() {
        assert_eq!(
            parse_unified_diff_marker_path(r#""b/docs/a\tb.md""#),
            Some("docs/a\tb.md".to_string())
        );
        assert_eq!(
            parse_unified_diff_marker_path(r#""b/docs/caf\303\251.md""#),
            Some(format!("docs/caf{}.md", '\u{e9}'))
        );
    }

    #[test]
    fn parse_unified_diff_marker_path_ignores_dev_null() {
        assert_eq!(parse_unified_diff_marker_path("/dev/null"), None);
    }
}
