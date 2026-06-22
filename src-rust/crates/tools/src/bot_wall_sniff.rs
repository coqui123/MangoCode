//! Heuristics for HTML/Markdown that often indicates a CDN / bot challenge page.
//!
//! The module is gated in `lib.rs` so these helpers compile only when a consumer
//! feature enables them.

/// The kind of bot wall or challenge detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BotWallKind {
    /// Cloudflare JS challenge / Turnstile / managed challenge.
    CloudflareChallenge,
    /// Cloudflare "sorry, you have been blocked" interstitial.
    CloudflareBlock,
    /// Generic "access denied" (may or may not be Cloudflare).
    AccessDenied,
    /// Generic "checking your browser" / "please wait" interstitial.
    BrowserCheck,
}

impl BotWallKind {
    #[allow(dead_code)]
    pub(crate) fn is_cloudflare(self) -> bool {
        matches!(
            self,
            BotWallKind::CloudflareChallenge | BotWallKind::CloudflareBlock
        )
    }
}

const CF_CHALLENGE_MARKERS: &[&str] = &[
    "__cf_chl",
    "cf-challenge",
    "challenge-platform",
    "cf-browser-verification",
    "turnstile",
    "\\/cdn-cgi\\/challenge",
];

const BROWSER_CHECK_MARKERS: &[&str] = &[
    "just a moment",
    "please wait while we verify",
    "checking your browser",
    "enable javascript and cookies to continue",
    "attention required!",
];

/// Detect the specific kind of bot wall present in an HTML/text snippet.
pub(crate) fn detect_bot_wall(snippet: &str) -> Option<BotWallKind> {
    let lower = snippet.to_ascii_lowercase();

    if CF_CHALLENGE_MARKERS.iter().any(|m| lower.contains(m)) {
        return Some(BotWallKind::CloudflareChallenge);
    }
    if lower.contains("sorry, you have been blocked") {
        return Some(BotWallKind::CloudflareBlock);
    }
    if lower.contains("cloudflare") && lower.contains("ray id") && snippet.len() < 64 * 1024 {
        return Some(BotWallKind::CloudflareChallenge);
    }
    if lower.contains("access denied") {
        return Some(BotWallKind::AccessDenied);
    }
    if BROWSER_CHECK_MARKERS.iter().any(|m| lower.contains(m)) {
        return Some(BotWallKind::BrowserCheck);
    }
    None
}

/// Returns `true` if the snippet looks like a bot-wall page.
pub(crate) fn text_suggests_bot_wall(snippet: &str) -> bool {
    detect_bot_wall(snippet).is_some()
}

/// Structured result from HTTP-level challenge detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChallengeDetection {
    pub kind: BotWallKind,
    pub has_cf_header: bool,
    pub status: u16,
}

/// Detect whether an HTTP failure response is likely a bot challenge.
pub(crate) fn detect_http_challenge(
    status: reqwest::StatusCode,
    headers: &reqwest::header::HeaderMap,
    body: &str,
) -> Option<ChallengeDetection> {
    let sts = status.as_u16();
    let has_cf_header = headers.get("cf-ray").is_some()
        || headers
            .get(reqwest::header::SERVER)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .eq_ignore_ascii_case("cloudflare");

    let body_kind = detect_bot_wall(body);

    let kind = match sts {
        403 | 429 | 503 if has_cf_header => body_kind.unwrap_or(BotWallKind::CloudflareChallenge),
        403 | 429 | 503 if body_kind.is_some() => body_kind.unwrap(),
        _ if body_kind.is_some() && (has_cf_header || sts == 429) => body_kind.unwrap(),
        _ => return None,
    };

    Some(ChallengeDetection {
        kind,
        has_cf_header,
        status: sts,
    })
}

/// Backward-compatible wrapper: returns `true` if the response looks like a challenge.
pub(crate) fn http_failure_might_be_challenge(
    status: reqwest::StatusCode,
    headers: &reqwest::header::HeaderMap,
    body: &str,
) -> bool {
    detect_http_challenge(status, headers, body).is_some()
}
