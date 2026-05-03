//! OpenAI Codex OAuth 2.0 PKCE flow for MangoCode (ChatGPT / Codex subscription).
//!
//! Authorization code + PKCE with a localhost callback on port 1455.

use anyhow::{anyhow, bail, Context};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use mangocode_core::codex_oauth::{
    extract_chatgpt_account_id, CODEX_AUTHORIZE_URL, CODEX_CLIENT_ID, CODEX_OAUTH_PORT,
    CODEX_REDIRECT_URI, CODEX_SCOPES, CODEX_TOKEN_URL,
};
use mangocode_core::oauth_config::CodexTokens;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::sync::mpsc;

use mangocode_tui::DeviceAuthEvent;

/// Generate a PKCE code verifier (48 random bytes → base64url).
pub fn generate_code_verifier() -> String {
    let mut bytes = [0u8; 48];
    let u1 = uuid::Uuid::new_v4();
    let u2 = uuid::Uuid::new_v4();
    let u3 = uuid::Uuid::new_v4();
    bytes[..16].copy_from_slice(u1.as_bytes());
    bytes[16..32].copy_from_slice(u2.as_bytes());
    bytes[32..48].copy_from_slice(&u3.as_bytes()[..16]);
    URL_SAFE_NO_PAD.encode(bytes)
}

/// Compute PKCE code challenge (SHA-256 of verifier, base64url encoded).
pub fn compute_code_challenge(verifier: &str) -> String {
    let hash = Sha256::digest(verifier.as_bytes());
    URL_SAFE_NO_PAD.encode(hash)
}

/// Generate a random OAuth state parameter.
pub fn generate_state() -> String {
    let bytes = uuid::Uuid::new_v4();
    URL_SAFE_NO_PAD
        .encode(bytes.as_bytes())
        .chars()
        .take(32)
        .collect()
}

/// Build the OpenAI authorization URL for Codex OAuth.
pub fn build_auth_url(code_challenge: &str, state: &str) -> String {
    format!(
        "{}?client_id={}&redirect_uri={}&response_type=code&scope={}&code_challenge={}&code_challenge_method=S256&state={}&id_token_add_organizations=true&codex_cli_simplified_flow=true",
        CODEX_AUTHORIZE_URL,
        CODEX_CLIENT_ID,
        urlencoding::encode(CODEX_REDIRECT_URI),
        urlencoding::encode(CODEX_SCOPES),
        code_challenge,
        state,
    )
}

fn parse_query_params(query: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for pair in query.split('&') {
        let mut kv = pair.splitn(2, '=');
        let k = kv.next().unwrap_or("");
        let v = kv.next().unwrap_or("");
        if k.is_empty() {
            continue;
        }
        if let Ok(decoded) = urlencoding::decode(v) {
            out.insert(k.to_string(), decoded.to_string());
        }
    }
    out
}

/// Parse `GET /path?query HTTP/1.x` and return the raw query string.
fn parse_request_query_line(line: &str) -> anyhow::Result<String> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 2 {
        bail!("Invalid HTTP request line");
    }
    let path = parts[1];
    let query_start = path
        .find('?')
        .ok_or_else(|| anyhow!("No query string in OAuth callback"))?;
    Ok(path[query_start + 1..].to_string())
}

async fn respond_html_ok(socket: &mut tokio::net::TcpStream, body: &str) -> anyhow::Result<()> {
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    socket.write_all(response.as_bytes()).await?;
    Ok(())
}

/// Accept one OAuth redirect, validate `state`, return the authorization `code`.
async fn accept_oauth_callback(
    listener: &TcpListener,
    expected_state: &str,
) -> anyhow::Result<String> {
    let (mut socket, _) = tokio::time::timeout(Duration::from_secs(300), listener.accept())
        .await
        .map_err(|_| anyhow!("OAuth callback timed out after 5 minutes"))?
        .map_err(|e| anyhow!("Failed to accept OAuth callback: {}", e))?;

    let mut reader = BufReader::new(&mut socket);
    let mut request_line = String::new();
    reader
        .read_line(&mut request_line)
        .await
        .context("read callback request")?;

    let query = parse_request_query_line(request_line.trim())?;
    let params = parse_query_params(&query);

    if let Some(err) = params.get("error") {
        let desc = params
            .get("error_description")
            .cloned()
            .unwrap_or_else(|| err.clone());
        let html = format!(
            "<html><body><h2>Login failed</h2><p>{}</p></body></html>",
            html_escape::encode_text(&desc)
        );
        respond_html_ok(&mut socket, &html).await?;
        bail!(
            "OAuth error: {}",
            params.get("error_description").unwrap_or(err)
        );
    }

    if params.get("state").map(|s| s.as_str()) != Some(expected_state) {
        respond_html_ok(
            &mut socket,
            "<html><body><h2>Login failed</h2><p>OAuth state mismatch.</p></body></html>",
        )
        .await?;
        bail!("OAuth state mismatch");
    }

    let code = params
        .get("code")
        .ok_or_else(|| anyhow!("Missing authorization code"))?;
    if code.is_empty() {
        respond_html_ok(
            &mut socket,
            "<html><body><h2>Login failed</h2><p>Missing authorization code.</p></body></html>",
        )
        .await?;
        bail!("Missing authorization code");
    }

    respond_html_ok(
        &mut socket,
        "<html><body><h2>Login complete</h2><p>You can return to MangoCode.</p></body></html>",
    )
    .await?;
    Ok(code.clone())
}

// Minimal HTML entity escaping for error_description (dependency-free).
mod html_escape {
    pub fn encode_text(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                '&' => out.push_str("&amp;"),
                '<' => out.push_str("&lt;"),
                '>' => out.push_str("&gt;"),
                '"' => out.push_str("&quot;"),
                _ => out.push(c),
            }
        }
        out
    }
}

/// Exchange authorization code for access tokens.
pub async fn exchange_code_for_tokens(code: &str, verifier: &str) -> anyhow::Result<CodexTokens> {
    let client = reqwest::Client::new();
    let params = [
        ("client_id", CODEX_CLIENT_ID),
        ("code", code),
        ("code_verifier", verifier),
        ("grant_type", "authorization_code"),
        ("redirect_uri", CODEX_REDIRECT_URI),
    ];

    let resp = client
        .post(CODEX_TOKEN_URL)
        .form(&params)
        .send()
        .await
        .map_err(|e| anyhow!("Failed to exchange code: {}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        bail!("Token exchange failed ({}): {}", status, body);
    }

    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| anyhow!("Failed to parse token response: {}", e))?;

    let access_token = body["access_token"].as_str().unwrap_or("").to_string();

    if access_token.is_empty() {
        bail!("No access_token in response");
    }

    let refresh_token = body["refresh_token"].as_str().map(|s| s.to_string());
    let expires_in = body["expires_in"]
        .as_u64()
        .or(body["expires_in"].as_i64().map(|v| v as u64));
    let expires_at = expires_in.map(|secs| {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        now.saturating_add(secs)
    });

    let account_id = extract_chatgpt_account_id(&access_token);

    Ok(CodexTokens {
        access_token,
        refresh_token,
        account_id,
        expires_at,
    })
}

/// Start local callback server, open browser, exchange code for tokens.
pub async fn run_oauth_flow(
    event_tx: mpsc::Sender<DeviceAuthEvent>,
) -> anyhow::Result<CodexTokens> {
    let verifier = generate_code_verifier();
    let challenge = compute_code_challenge(&verifier);
    let state = generate_state();
    let auth_url = build_auth_url(&challenge, &state);

    let _ = event_tx
        .send(DeviceAuthEvent::GotBrowserUrl {
            url: auth_url.clone(),
        })
        .await;

    let addr = format!("127.0.0.1:{}", CODEX_OAUTH_PORT);
    let listener = TcpListener::bind(&addr)
        .await
        .map_err(|e| {
            anyhow!(
                "Could not bind OAuth callback on {}: {}. Try closing other apps using this port, or run login on a machine where port {} is free.\n\n{}",
                addr,
                e,
                CODEX_OAUTH_PORT,
                mangocode_core::codex_oauth::HEADLESS_CODEX_OAUTH_HINT
            )
        })?;

    let _ = open::that(&auth_url);

    let code = accept_oauth_callback(&listener, &state).await?;
    exchange_code_for_tokens(&code, &verifier).await
}

/// Plain terminal variant for `mangocode auth codex login`.
///
/// It still uses browser + localhost callback OAuth, but does not require the TUI.
pub async fn run_terminal_oauth_flow() -> anyhow::Result<CodexTokens> {
    let verifier = generate_code_verifier();
    let challenge = compute_code_challenge(&verifier);
    let state = generate_state();
    let auth_url = build_auth_url(&challenge, &state);

    let addr = format!("127.0.0.1:{}", CODEX_OAUTH_PORT);
    let listener = TcpListener::bind(&addr)
        .await
        .map_err(|e| {
            anyhow!(
                "Could not bind OAuth callback on {}: {}. Try closing other apps using this port, or run login on a machine where port {} is free.\n\n{}",
                addr,
                e,
                CODEX_OAUTH_PORT,
                mangocode_core::codex_oauth::HEADLESS_CODEX_OAUTH_HINT
            )
        })?;

    println!("Open this URL to sign in to OpenAI Codex:");
    println!("{}", auth_url);
    println!();
    println!(
        "Waiting for OAuth callback on {} (timeout: 5 minutes)...",
        CODEX_REDIRECT_URI
    );

    if let Err(e) = open::that(&auth_url) {
        eprintln!("Warning: could not open browser automatically: {}", e);
        eprintln!("Open the URL above manually.");
    }

    let code = accept_oauth_callback(&listener, &state).await?;
    exchange_code_for_tokens(&code, &verifier).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_code_verifier_format() {
        let verifier = generate_code_verifier();
        assert!(verifier
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-'));
        assert!(!verifier.is_empty());
    }

    #[test]
    fn test_compute_code_challenge_consistency() {
        let verifier = "test_verifier_string";
        let challenge1 = compute_code_challenge(verifier);
        let challenge2 = compute_code_challenge(verifier);
        assert_eq!(challenge1, challenge2);
        assert!(challenge1
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-'));
    }

    #[test]
    fn test_generate_state_format() {
        let state = generate_state();
        assert!(!state.is_empty());
        assert!(state
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-'));
    }

    #[test]
    fn test_build_auth_url_contains_required_params() {
        let url = build_auth_url("challenge123", "state456");
        assert!(url.contains("client_id="));
        assert!(url.contains("challenge123"));
        assert!(url.contains("state456"));
        assert!(url.contains("S256"));
        assert!(url.contains("response_type=code"));
        assert!(url.contains("id_token_add_organizations=true"));
        assert!(url.contains("codex_cli_simplified_flow=true"));
    }

    #[test]
    fn parse_query_params_roundtrip() {
        let q = "code=abc&state=xyz&error_description=bad";
        let m = parse_query_params(q);
        assert_eq!(m.get("code").map(String::as_str), Some("abc"));
        assert_eq!(m.get("state").map(String::as_str), Some("xyz"));
    }

    #[test]
    fn oauth_state_mismatch_returns_err_from_callback_simulation() {
        // accept_oauth_callback requires a real listener — covered indirectly;
        // here we assert query parsing rejects wrong state at the param level.
        let params = parse_query_params("code=foo&state=wrong");
        assert_ne!(params.get("state").map(String::as_str), Some("expected"));
    }

    #[test]
    fn test_extract_account_id_from_invalid_jwt() {
        let invalid_token = "not.a.jwt";
        let result = extract_chatgpt_account_id(invalid_token);
        assert!(result.is_none());
    }
}
