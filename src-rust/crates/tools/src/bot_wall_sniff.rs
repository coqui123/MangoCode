//! Heuristics for HTML/Markdown that often indicates a CDN / bot challenge page (Cloudflare-ish).
//!
//! The module is gated in `lib.rs` so these helpers compile only when a consumer feature enables them.

pub(crate) fn text_suggests_bot_wall(snippet: &str) -> bool {
    let lower = snippet.to_ascii_lowercase();
    const MARKERS: &[&str] = &[
        "just a moment",
        "please wait while we verify",
        "__cf_chl",
        "cf-challenge",
        "challenge-platform",
        "checking your browser",
        "attention required!",
        "cf-browser-verification",
        "enable javascript and cookies to continue",
        "turnstile",
        "\\/cdn-cgi\\/challenge",
        "sorry, you have been blocked",
        "access denied",
    ];
    if MARKERS.iter().any(|m| lower.contains(*m)) {
        return true;
    }
    if lower.contains("cloudflare") && lower.contains("ray id") && snippet.len() < 64 * 1024 {
        return true;
    }
    false
}

pub(crate) fn http_failure_might_be_challenge(
    status: reqwest::StatusCode,
    headers: &reqwest::header::HeaderMap,
    body: &str,
) -> bool {
    let sts = status.as_u16();
    let cf_hdr = headers.get("cf-ray").is_some()
        || headers
            .get(reqwest::header::SERVER)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .eq_ignore_ascii_case("cloudflare");
    match sts {
        403 | 429 | 503 if cf_hdr || text_suggests_bot_wall(body) => true,
        _ => text_suggests_bot_wall(body) && (cf_hdr || sts == 429),
    }
}
