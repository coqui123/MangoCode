//! OpenAI Codex OAuth configuration and constants.
//!

/// OpenAI Codex OAuth client ID shared by the browser flow.
pub const CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

/// OpenAI OAuth issuer base URL.
pub const CODEX_ISSUER: &str = "https://auth.openai.com";

/// OpenAI OAuth authorization endpoint
pub const CODEX_AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";

/// OpenAI OAuth token endpoint
pub const CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";

/// Codex Responses API endpoint.
pub const CODEX_API_ENDPOINT: &str = "https://chatgpt.com/backend-api/codex/responses";

/// Local redirect URI for OAuth callback
pub const CODEX_REDIRECT_URI: &str = "http://localhost:1455/auth/callback";

/// OAuth callback port.
pub const CODEX_OAUTH_PORT: u16 = 1455;

/// OAuth scopes requested from OpenAI
pub const CODEX_SCOPES: &str = "openid profile email offline_access";

/// Available Codex models
pub const CODEX_MODELS: &[(&str, &str)] = &[
    ("gpt-5.3-codex", "GPT-5.3 Codex"),
    ("gpt-5.2-codex", "GPT-5.2 Codex (default)"),
    ("gpt-5.1-codex", "GPT-5.1 Codex"),
    ("gpt-5.1-codex-mini", "GPT-5.1 Codex Mini"),
    ("gpt-5.1-codex-max", "GPT-5.1 Codex Max"),
    ("gpt-5.4", "GPT-5.4"),
    ("gpt-5.2", "GPT-5.2"),
];

/// Default Codex model to use
pub const DEFAULT_CODEX_MODEL: &str = "gpt-5.2-codex";

/// Heuristic for environments where `http://localhost:…` OAuth callbacks are
/// often unreachable (SSH, Codespaces, Dev Containers, VS Code integrated terminal).
pub fn likely_headless_or_remote() -> bool {
    std::env::var("SSH_CONNECTION").is_ok()
        || std::env::var("SSH_CLIENT").is_ok()
        || std::env::var("CODESPACES").is_ok()
        || std::env::var("GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN").is_ok()
        || std::env::var("REMOTE_CONTAINERS").is_ok()
        || std::env::var("VSCODE_IPC_HOOK_CLI").is_ok()
        || std::env::var("TERM_PROGRAM")
            .map(|v| v.eq_ignore_ascii_case("vscode"))
            .unwrap_or(false)
}

/// User-facing guidance when browser+localhost OAuth is unlikely to work.
pub const HEADLESS_CODEX_OAUTH_HINT: &str = "OpenAI Codex browser login uses a localhost callback, which often fails over SSH, in GitHub Codespaces, or in some remote containers.\n\
Authenticate on your local machine with MangoCode and `/connect`, or use OpenAI API key mode (`/connect` → OpenAI) for usage-based access.\n\
Device-code login for Codex will be added when OpenAI documents a supported device authorization endpoint for this client.";

// ---------------------------------------------------------------------------
// JWT helpers (Codex OAuth)
// ---------------------------------------------------------------------------

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde_json::Value;

/// Decode the JWT payload (2nd segment) into JSON.
///
/// Codex OAuth access tokens are JWTs. We avoid validating signatures here; we
/// only need selected claims to build Codex backend headers.
pub fn decode_jwt_payload(token: &str) -> Option<Value> {
    let mut parts = token.splitn(3, '.');
    let _header_b64 = parts.next()?;
    let payload_b64 = parts.next()?;
    let _sig_b64 = parts.next()?;

    let payload_bytes = URL_SAFE_NO_PAD.decode(payload_b64).ok()?;
    serde_json::from_slice::<Value>(&payload_bytes).ok()
}

/// Extract a ChatGPT/Codex account id from a Codex OAuth access token.
///
/// Different clients have historically used different claim paths. We try the
/// plugin-documented keys first, then fall back to older MangoCode-derived ones.
pub fn extract_chatgpt_account_id(token: &str) -> Option<String> {
    let payload = decode_jwt_payload(token)?;

    // Plugin-described location often appears nested under a key that itself
    // contains slashes (cannot be accessed via JSON Pointer safely).
    payload
        .get("https://api.openai.com/auth")
        .and_then(|v| v.get("chatgpt_account_id"))
        .and_then(|v| v.as_str())
        .map(str::to_string)
        // Sometimes flattened.
        .or_else(|| {
            payload
                .get("chatgpt_account_id")
                .and_then(|v| v.as_str())
                .map(str::to_string)
        })
        // Older MangoCode extraction (account_id).
        .or_else(|| {
            payload
                .get("https://api.openai.com/auth")
                .and_then(|v| v.get("account_id"))
                .and_then(|v| v.as_str())
                .map(str::to_string)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;

    #[test]
    fn test_codex_constants_not_empty() {
        assert!(!CODEX_CLIENT_ID.is_empty());
        assert!(!CODEX_AUTHORIZE_URL.is_empty());
        assert!(!CODEX_TOKEN_URL.is_empty());
        assert!(!CODEX_API_ENDPOINT.is_empty());
        assert!(!CODEX_REDIRECT_URI.is_empty());
        assert_eq!(CODEX_OAUTH_PORT, 1455);
        assert!(!CODEX_SCOPES.is_empty());
        assert!(!CODEX_MODELS.is_empty());
        assert!(!DEFAULT_CODEX_MODEL.is_empty());
    }

    #[test]
    fn test_codex_models_contains_default() {
        let default_found = CODEX_MODELS
            .iter()
            .any(|(model, _)| model == &DEFAULT_CODEX_MODEL);
        assert!(
            default_found,
            "DEFAULT_CODEX_MODEL must be in CODEX_MODELS list"
        );
    }

    #[test]
    fn test_redirect_uri_is_localhost() {
        assert!(CODEX_REDIRECT_URI.contains("localhost:1455"));
    }

    #[test]
    fn likely_headless_or_remote_is_boolean() {
        // Smoke: must not panic regardless of test runner environment.
        let _ = likely_headless_or_remote();
    }

    #[test]
    fn decode_jwt_payload_rejects_non_jwt() {
        assert!(decode_jwt_payload("not.a.jwt").is_none());
    }

    #[test]
    fn extract_chatgpt_account_id_none_for_invalid() {
        assert!(extract_chatgpt_account_id("not.a.jwt").is_none());
    }

    #[test]
    fn extract_chatgpt_account_id_from_nested_claim() {
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(
            br#"{"https://api.openai.com/auth":{"chatgpt_account_id":"acct_123"}}"#,
        );
        let token = format!("{}.{}.{}", "hdr", payload, "sig");
        assert_eq!(extract_chatgpt_account_id(&token).as_deref(), Some("acct_123"));
    }
}
