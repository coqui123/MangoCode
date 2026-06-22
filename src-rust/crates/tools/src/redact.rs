//! Redacts secrets and credentials from arbitrary text before logging or model submission.

use std::borrow::Cow;
use std::sync::LazyLock;

use regex::Regex;

/// Combined pattern matching high-confidence single-token secret formats.
static SECRET_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?x)
          sk-ant-[A-Za-z0-9_-]{20,}                                  # Anthropic API key
        | sk-[A-Za-z0-9]{20,}                                        # OpenAI-style key
        | gh[pousr]_[A-Za-z0-9]{36,}                                 # GitHub tokens
        | AKIA[0-9A-Z]{16}                                           # AWS access key id
        | AIza[0-9A-Za-z_-]{35}                                      # Google API key
        | xox[baprs]-[A-Za-z0-9-]{10,}                               # Slack token
        | eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+       # JWT (header.payload.sig)
        ",
    )
    .expect("redact: secret regex must compile")
});

/// PEM private-key blocks span multiple lines, so they need their own
/// dot-matches-newline pattern.
static PRIVATE_KEY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----")
        .expect("redact: private-key regex must compile")
});

/// Replace every occurrence of a known secret pattern with `[REDACTED]`.
///
/// Returns `Cow::Borrowed` when the input is empty or contains no matches,
/// avoiding an allocation in the common case.
pub fn redact_secrets(input: &str) -> Cow<'_, str> {
    if input.is_empty() {
        return Cow::Borrowed(input);
    }
    // Redact multi-line private-key blocks first, then single-token secrets.
    match PRIVATE_KEY_RE.replace_all(input, "[REDACTED]") {
        Cow::Borrowed(_) => SECRET_RE.replace_all(input, "[REDACTED]"),
        Cow::Owned(stripped) => {
            Cow::Owned(SECRET_RE.replace_all(&stripped, "[REDACTED]").into_owned())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    #[test]
    fn anthropic_key_redacted() {
        let input = "key: sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAA";
        let result = redact_secrets(input);
        assert_eq!(result, "key: [REDACTED]");
        assert!(matches!(result, Cow::Owned(_)));
    }

    #[test]
    fn openai_key_redacted() {
        let input = "Authorization: Bearer sk-abc123XYZ456def789ghi012";
        let result = redact_secrets(input);
        assert_eq!(result, "Authorization: Bearer [REDACTED]");
    }

    #[test]
    fn github_ghp_token_redacted() {
        // ghp_ + 40 alphanumeric chars
        let token = format!("ghp_{}", "A".repeat(40));
        let input = format!("token: {token}");
        let result = redact_secrets(&input);
        assert_eq!(result, "token: [REDACTED]");
        assert!(matches!(result, Cow::Owned(_)));
    }

    #[test]
    fn github_gho_token_redacted() {
        let token = format!("gho_{}", "B".repeat(36));
        let result = redact_secrets(&token);
        assert_eq!(result, "[REDACTED]");
    }

    #[test]
    fn aws_access_key_redacted() {
        let input = "aws_access_key_id = AKIAIOSFODNN7EXAMPLE";
        let result = redact_secrets(input);
        assert_eq!(result, "aws_access_key_id = [REDACTED]");
    }

    #[test]
    fn google_api_key_redacted() {
        // AIza + 35 chars
        let key = format!("AIza{}", "x".repeat(35));
        let input = format!("key={key}");
        let result = redact_secrets(&input);
        assert_eq!(result, "key=[REDACTED]");
    }

    #[test]
    fn slack_token_redacted() {
        let input = "SLACK_TOKEN=xoxb-1234567890-abcdefghij";
        let result = redact_secrets(input);
        assert_eq!(result, "SLACK_TOKEN=[REDACTED]");
    }

    #[test]
    fn plain_text_unchanged_and_borrowed() {
        let input = "Hello, this is perfectly normal text with no secrets.";
        let result = redact_secrets(input);
        assert_eq!(result, input);
        assert!(
            matches!(result, Cow::Borrowed(_)),
            "plain text should return Cow::Borrowed"
        );
    }

    #[test]
    fn empty_string_returns_borrowed() {
        let result = redact_secrets("");
        assert_eq!(result, "");
        assert!(
            matches!(result, Cow::Borrowed(_)),
            "empty input should return Cow::Borrowed"
        );
    }

    #[test]
    fn multiple_secrets_all_redacted() {
        let input = "first=sk-ant-api03-BBBBBBBBBBBBBBBBBBBBBBBB second=AKIAIOSFODNN7EXAMPLE";
        let result = redact_secrets(input);
        assert_eq!(result, "first=[REDACTED] second=[REDACTED]");
    }

    #[test]
    fn jwt_is_redacted() {
        let input = "token=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U end";
        let result = redact_secrets(input);
        assert!(!result.contains("eyJ"), "JWT leaked: {result}");
        assert!(result.contains("[REDACTED]"));
        assert!(result.contains("end"));
    }

    #[test]
    fn pem_private_key_block_is_redacted() {
        let input = "before\n-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEAabc123\nXYZ987\n-----END RSA PRIVATE KEY-----\nafter";
        let result = redact_secrets(input);
        assert!(!result.contains("PRIVATE KEY"), "key leaked: {result}");
        assert!(!result.contains("MIIEpA"), "key body leaked: {result}");
        assert!(result.contains("before") && result.contains("after"));
        assert!(result.contains("[REDACTED]"));
    }
}
