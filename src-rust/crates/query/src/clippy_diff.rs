use serde_json::Value;
use std::collections::HashMap;

/// A content-based fingerprint for a single clippy diagnostic.
///
/// The `(code, message, snippet)` tuple deliberately omits line numbers so the
/// fingerprint survives edits that shift lines around.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClippyFinding {
    /// The lint name, e.g. `"clippy::redundant_clone"`.
    pub code: String,
    /// First line of the rendered message, trimmed.
    pub message: String,
    /// Trimmed source text of the primary span.
    pub snippet: String,
}

/// Parse newline-delimited JSON produced by
/// `cargo clippy --message-format=json` into [`ClippyFinding`]s.
///
/// Lines that fail to parse, that are not `compiler-message` records, that lack
/// a lint code, or whose level is neither `"warning"` nor `"error"` are silently
/// skipped.
pub fn parse_clippy_findings(output: &str) -> Vec<ClippyFinding> {
    let mut findings = Vec::new();

    for line in output.lines() {
        let value: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Only compiler-message records carry lint diagnostics.
        if value["reason"].as_str() != Some("compiler-message") {
            continue;
        }

        let msg = &value["message"];

        // A lint code is required — plain notes have null/absent code.
        let code = match msg["code"]["code"].as_str() {
            Some(c) => c.to_owned(),
            None => continue,
        };

        // Keep only warnings and errors.
        match msg["level"].as_str() {
            Some("warning") | Some("error") => {}
            _ => continue,
        }

        // First line of the human-readable message, trimmed.
        let message = msg["message"]
            .as_str()
            .unwrap_or_default()
            .lines()
            .next()
            .unwrap_or_default()
            .trim()
            .to_owned();

        // Source text from the first primary span.
        let snippet = msg["spans"]
            .as_array()
            .and_then(|spans| {
                spans
                    .iter()
                    .find(|s| s["is_primary"].as_bool() == Some(true))
            })
            .and_then(|span| span["text"].as_array())
            .and_then(|texts| texts.first())
            .and_then(|t| t["text"].as_str())
            .unwrap_or_default()
            .trim()
            .to_owned();

        findings.push(ClippyFinding {
            code,
            message,
            snippet,
        });
    }

    findings
}

/// Compute the multiset difference `current \ baseline`.
///
/// A finding in `current` is "introduced" only if it is not accounted for by a
/// matching baseline entry.  Duplicate counts are respected: if the baseline
/// contains 2 copies of a fingerprint and current contains 3, exactly 1 is
/// returned.
pub fn introduced_findings(
    baseline: &[ClippyFinding],
    current: &[ClippyFinding],
) -> Vec<ClippyFinding> {
    let mut baseline_counts: HashMap<&ClippyFinding, usize> = HashMap::new();
    for finding in baseline {
        *baseline_counts.entry(finding).or_insert(0) += 1;
    }

    let mut introduced = Vec::new();
    for finding in current {
        if let Some(count) = baseline_counts.get_mut(finding) {
            if *count > 0 {
                *count -= 1;
                continue;
            }
        }
        introduced.push(finding.clone());
    }

    introduced
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a minimal `compiler-message` JSON line. Serialized via
    /// serde_json so quotes/escapes in the inputs are handled correctly.
    fn make_line(
        code: Option<&str>,
        level: &str,
        message: &str,
        primary_text: Option<&str>,
    ) -> String {
        let code_json = match code {
            Some(c) => serde_json::json!({ "code": c, "explanation": null }),
            None => Value::Null,
        };
        let spans_json = match primary_text {
            Some(txt) => serde_json::json!([{ "is_primary": true, "text": [{ "text": txt }] }]),
            None => serde_json::json!([]),
        };
        serde_json::json!({
            "reason": "compiler-message",
            "message": {
                "code": code_json,
                "level": level,
                "message": message,
                "spans": spans_json,
                "rendered": "",
            },
        })
        .to_string()
    }

    #[test]
    fn parse_warning_with_primary_span() {
        let line = make_line(
            Some("clippy::redundant_clone"),
            "warning",
            "redundant clone",
            Some("    foo.clone()"),
        );
        let findings = parse_clippy_findings(&line);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].code, "clippy::redundant_clone");
        assert_eq!(findings[0].message, "redundant clone");
        assert_eq!(findings[0].snippet, "foo.clone()");
    }

    #[test]
    fn ignore_non_compiler_message() {
        let line = r#"{"reason":"build-script-executed","message":null}"#;
        let findings = parse_clippy_findings(line);
        assert!(findings.is_empty());
    }

    #[test]
    fn ignore_null_code() {
        let line = make_line(None, "warning", "some note", Some("x"));
        let findings = parse_clippy_findings(&line);
        assert!(findings.is_empty());
    }

    #[test]
    fn ignore_unparseable_line() {
        let input = "this is not json\n{{{bad";
        let findings = parse_clippy_findings(input);
        assert!(findings.is_empty());
    }

    #[test]
    fn introduced_new_finding() {
        let baseline = vec![];
        let current = vec![ClippyFinding {
            code: "clippy::needless_return".into(),
            message: "needless return".into(),
            snippet: "return x;".into(),
        }];
        let result = introduced_findings(&baseline, &current);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].code, "clippy::needless_return");
    }

    #[test]
    fn pre_existing_finding_not_introduced() {
        let finding = ClippyFinding {
            code: "clippy::redundant_clone".into(),
            message: "redundant clone".into(),
            snippet: "foo.clone()".into(),
        };
        let baseline = vec![finding.clone()];
        let current = vec![finding];
        let result = introduced_findings(&baseline, &current);
        assert!(result.is_empty());
    }

    #[test]
    fn multiset_difference_respects_counts() {
        let finding = ClippyFinding {
            code: "clippy::redundant_clone".into(),
            message: "redundant clone".into(),
            snippet: "foo.clone()".into(),
        };
        // baseline has 1, current has 2 → 1 introduced
        let baseline = vec![finding.clone()];
        let current = vec![finding.clone(), finding.clone()];
        let result = introduced_findings(&baseline, &current);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], finding);
    }

    #[test]
    fn ignore_info_level() {
        let line = make_line(
            Some("clippy::some_lint"),
            "note",
            "informational",
            Some("x"),
        );
        let findings = parse_clippy_findings(&line);
        assert!(findings.is_empty());
    }

    #[test]
    fn parse_error_level_finding() {
        let line = make_line(
            Some("clippy::invalid_regex"),
            "error",
            "invalid regex",
            Some("  Regex::new(\"[\")  "),
        );
        let findings = parse_clippy_findings(&line);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].code, "clippy::invalid_regex");
        assert_eq!(findings[0].message, "invalid regex");
        // leading/trailing whitespace trimmed
        assert_eq!(findings[0].snippet, r#"Regex::new("[")"#);
    }

    #[test]
    fn missing_primary_span_gives_empty_snippet() {
        // Spans array with no primary span.
        let line = r#"{"reason":"compiler-message","message":{"code":{"code":"clippy::todo","explanation":null},"level":"warning","message":"todo found","spans":[{"is_primary":false,"text":[{"text":"todo!()"}]}],"rendered":""}}"#;
        let findings = parse_clippy_findings(line);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].snippet, "");
    }
}
