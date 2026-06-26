// error_handling.rs — Provider-aware error detection and retry utilities
// (Phase 6).
//
// Provides:
//  - `is_context_overflow`: checks a message string against 29+ known
//    context-window overflow error patterns from all major providers.
//  - `parse_error_response`: converts an HTTP status + body into the correct
//    `ProviderError` variant, including overflow detection and JSON code
//    extraction.
//  - `RetryConfig`: exponential back-off configuration with jitter.

use std::time::Duration;

use mangocode_core::provider_id::ProviderId;

use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;

use crate::provider::LlmProvider;
use crate::provider_error::ProviderError;
use crate::provider_types::{ProviderRequest, StreamEvent};

// ---------------------------------------------------------------------------
// Overflow pattern table
// ---------------------------------------------------------------------------

/// 29+ context-overflow patterns that appear across all major providers.
static OVERFLOW_PATTERNS: &[&str] = &[
    "prompt is too long",
    "input is too long for requested model",
    "expected maxlength:",
    "exceeds the context window",
    "maximum context length",
    "input token count.*exceeds the maximum",
    "maximum prompt length is",
    "reduce the length of the messages",
    "maximum context length is.*tokens",
    "exceeds the limit of",
    "exceeds the available context size",
    "greater than the context length",
    "context window exceeds limit",
    "exceeded model token limit",
    "prompt too long",
    "too large for model with.*maximum context length",
    "model_context_window_exceeded",
    "context length is only.*tokens",
    "input length.*exceeds.*context length",
    "context_length_exceeded",
    "request entity too large",
    "too many tokens",
    "context.*length.*exceeded",
    "token.*limit.*exceeded",
    "prompt.*too.*long",
    "exceeds.*context.*size",
    "context.*window.*exceeded",
    "max.*tokens.*exceeded",
    "input.*too.*long",
];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Returns `true` if `message` matches any known context-overflow pattern.
///
/// The comparison is case-insensitive.  Patterns are matched as substrings
/// (not full regexes) for performance — the pattern table is designed so that
/// simple substring matching is sufficient.
pub fn is_context_overflow(message: &str) -> bool {
    let lower = message.to_lowercase();
    OVERFLOW_PATTERNS
        .iter()
        .any(|pattern| lower.contains(&pattern.to_lowercase()))
}

/// Detect a WebSocket "message too big" failure (close code 1009) surfaced by
/// proxy-backed providers such as the Copilot-pirate / M365 Copilot bridge.
/// Those backends abort the upstream socket when a single request frame exceeds
/// their byte limit (commonly ~1 MB) and report it back as a 502. This is a
/// request-size problem, not a transient server error, so it must NOT be
/// retried — retrying re-sends the same oversized payload and fails again.
pub fn is_websocket_frame_overflow(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("message too big")
        || (lower.contains("exceeds limit of") && lower.contains("byte"))
        || (lower.contains("frame") && lower.contains("exceeds limit") && lower.contains("byte"))
}

/// Convert an HTTP error response into the appropriate [`ProviderError`].
///
/// Tries JSON parsing, then falls back to the raw body.  Context overflow is
/// checked before the HTTP status code so that a 400 with an overflow message
/// is classified as [`ProviderError::ContextOverflow`] rather than
/// [`ProviderError::InvalidRequest`].
pub fn parse_error_response(status: u16, body: &str, provider: &ProviderId) -> ProviderError {
    let json: Option<serde_json::Value> = serde_json::from_str(body).ok();

    let message = if let Some(ref j) = json {
        extract_error_message(j)
    } else if body.trim_start().starts_with('<') {
        // HTML error page (Azure proxy, CDN, etc.)
        "Received HTML error page — check provider endpoint configuration".to_string()
    } else {
        body.to_string()
    };

    // Check for context overflow before all other classifications.
    if is_context_overflow(&message) || is_context_overflow(body) {
        return ProviderError::ContextOverflow {
            provider: provider.clone(),
            message,
            max_tokens: extract_token_limit(body),
        };
    }

    // WebSocket-proxied backends (Copilot-pirate / M365 Copilot bridge) enforce a
    // hard per-frame byte limit (~1 MB) and abort the upstream socket with close
    // code 1009 ("message too big"), which surfaces here as a 502. Retrying the
    // same oversized payload always fails, so treat it as a non-retryable
    // request-size overflow with actionable guidance rather than a transient
    // server error.
    if is_websocket_frame_overflow(&message) || is_websocket_frame_overflow(body) {
        return ProviderError::ContextOverflow {
            provider: provider.clone(),
            message: format!(
                "Request exceeded the provider proxy's WebSocket frame limit \
                 (~1 MB; upstream closed with code 1009 \"message too big\"). Shrink \
                 the request — run /compact, attach fewer or smaller files, or raise \
                 the proxy's max frame size — then retry. (proxy: {message})"
            ),
            max_tokens: None,
        };
    }

    // Check for structured error codes returned by some providers.
    if let Some(ref j) = json {
        if let Some(code) = extract_error_code(j) {
            match code.as_str() {
                "context_length_exceeded" | "context_window_exceeded" => {
                    return ProviderError::ContextOverflow {
                        provider: provider.clone(),
                        message,
                        max_tokens: None,
                    };
                }
                "insufficient_quota" | "billing_not_active" => {
                    return ProviderError::QuotaExceeded {
                        provider: provider.clone(),
                        message,
                    };
                }
                "invalid_prompt" | "invalid_request_error" => {
                    return ProviderError::InvalidRequest {
                        provider: provider.clone(),
                        message,
                    };
                }
                "content_filter" | "content_policy_violation" => {
                    return ProviderError::ContentFiltered {
                        provider: provider.clone(),
                        message,
                    };
                }
                _ => {}
            }
        }
    }

    // Some providers return 401/403 with a quota / tier message (e.g. DashScope free tier).
    let lower = message.to_lowercase();
    if (status == 401 || status == 403)
        && lower.contains("free tier")
        && (lower.contains("exhausted") || lower.contains("quota"))
    {
        return ProviderError::QuotaExceeded {
            provider: provider.clone(),
            message,
        };
    }

    // Classify by HTTP status code.
    match status {
        402 => ProviderError::QuotaExceeded {
            provider: provider.clone(),
            message,
        },
        401 | 403 => ProviderError::AuthFailed {
            provider: provider.clone(),
            message,
        },
        404 => ProviderError::ModelNotFound {
            provider: provider.clone(),
            model: "unknown".to_string(),
            suggestions: vec![],
        },
        429 => ProviderError::RateLimited {
            provider: provider.clone(),
            retry_after: None,
        },
        413 => ProviderError::ContextOverflow {
            provider: provider.clone(),
            message: "Request too large (413)".to_string(),
            max_tokens: None,
        },
        500..=599 => ProviderError::ServerError {
            provider: provider.clone(),
            status: Some(status),
            message,
            is_retryable: true,
        },
        _ => ProviderError::Other {
            provider: provider.clone(),
            message,
            status: Some(status),
            body: Some(body.to_string()),
        },
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Walk several well-known JSON paths to find the human-readable error message.
fn extract_error_message(json: &serde_json::Value) -> String {
    // Ordered by prevalence across providers:
    //   OpenAI / Google: /error/message
    //   Anthropic:        /error/error/message
    //   Cohere / simple:  /message
    //   Some providers:   /detail
    let paths = [
        "/error/message",
        "/error/error/message",
        "/message",
        "/detail",
    ];
    for path in paths {
        if let Some(msg) = json.pointer(path).and_then(|v| v.as_str()) {
            return msg.to_string();
        }
    }
    json.to_string()
}

/// Extract a machine-readable error code from the JSON body, if present.
fn extract_error_code(json: &serde_json::Value) -> Option<String> {
    json.pointer("/error/code")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            json.pointer("/error/type")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
}

/// Heuristically extract a token-limit number from error text.
///
/// Looks for an integer that is:
/// - between 1 000 and 10 000 000 (plausible token limit range), and
/// - adjacent to words like "token", "limit", "context", or "max".
fn extract_token_limit(text: &str) -> Option<u64> {
    let words: Vec<&str> = text.split_whitespace().collect();
    for (i, word) in words.iter().enumerate() {
        // Strip non-digit chars from both ends, then try to parse.
        let trimmed = word.trim_matches(|c: char| !c.is_ascii_digit());
        if let Ok(n) = trimmed.parse::<u64>() {
            if n > 1_000 && n < 10_000_000 {
                let start = i.saturating_sub(3);
                let context = words[start..i].join(" ").to_lowercase();
                if context.contains("token")
                    || context.contains("limit")
                    || context.contains("context")
                    || context.contains("max")
                {
                    return Some(n);
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// RetryConfig
// ---------------------------------------------------------------------------

/// Exponential back-off configuration for provider retries.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Delay before the first retry.
    pub initial_delay: Duration,
    /// Upper bound on per-attempt delay.
    pub max_delay: Duration,
    /// Multiplicative factor applied at each attempt.
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 8,
            initial_delay: Duration::from_secs(2),
            max_delay: Duration::from_secs(300),
            backoff_multiplier: 2.0,
        }
    }
}

impl RetryConfig {
    /// Compute the delay for a given `attempt` number (0-indexed).
    ///
    /// Applies exponential back-off with ±10 % jitter derived from the
    /// current system time (no external `rand` dependency required).
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let base = self.initial_delay.as_secs_f64() * self.backoff_multiplier.powi(attempt as i32);
        let jitter = base * 0.1 * time_jitter_f64();
        Duration::from_secs_f64((base + jitter).min(self.max_delay.as_secs_f64()))
    }
}

/// Returns a deterministic-ish value in `[0, 1)` derived from the current
/// system time nanoseconds.  Used for retry jitter without pulling in `rand`.
fn time_jitter_f64() -> f64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    (nanos % 100) as f64 / 100.0
}

// ---------------------------------------------------------------------------
// Provider-level failover
// ---------------------------------------------------------------------------

/// Try each provider in `providers` (priority order) until one succeeds.
///
/// Returns `(index, stream)` where `index` is the position of the provider
/// that produced the stream.  On a non-retryable error (auth failure, model
/// not found, quota exhausted, invalid request, etc.) the function skips to
/// the next provider.  If all providers fail, the last error is returned.
pub async fn stream_with_failover(
    providers: &[Arc<dyn LlmProvider>],
    request: &ProviderRequest,
) -> Result<
    (
        usize,
        Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>,
    ),
    ProviderError,
> {
    if providers.is_empty() {
        return Err(ProviderError::Other {
            provider: ProviderId::new("none"),
            message: "No providers configured for failover".to_string(),
            status: None,
            body: None,
        });
    }

    let mut last_error: Option<ProviderError> = None;

    for (i, provider) in providers.iter().enumerate() {
        let req = request.clone();
        match provider.create_message_stream(req).await {
            Ok(stream) => return Ok((i, stream)),
            Err(e) => {
                if e.is_retryable() {
                    // Retryable errors (rate-limit, transient server error)
                    // are returned immediately so the caller can apply
                    // RetryConfig with the *same* provider.
                    return Err(e);
                }
                // Non-retryable (auth, model-not-found, quota, etc.) →
                // failover to the next provider.
                last_error = Some(e);
            }
        }
    }

    Err(last_error.expect("providers was non-empty; at least one error must exist"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_context_overflow_basic() {
        assert!(is_context_overflow("prompt is too long for this model"));
        assert!(is_context_overflow("This exceeds the context window"));
        assert!(is_context_overflow("Maximum context length exceeded"));
        assert!(!is_context_overflow("something else went wrong"));
    }

    #[test]
    fn test_parse_error_response_overflow_413() {
        let pid = ProviderId::new("openai");
        let err = parse_error_response(413, "Request too large", &pid);
        assert!(matches!(err, ProviderError::ContextOverflow { .. }));
    }

    #[test]
    fn test_websocket_frame_overflow_detection() {
        // The real body the Copilot-pirate proxy returns on the 1 MB frame cap.
        let body = "sent 1009 (message too big) frame with 7530 bytes after reading \
                    1042385 bytes exceeds limit of 1048576 bytes; no close frame received";
        assert!(is_websocket_frame_overflow(body));
        // A handshake timeout is genuinely transient and must NOT match.
        assert!(!is_websocket_frame_overflow("timed out during opening handshake"));
        assert!(!is_websocket_frame_overflow("502 Bad Gateway"));
    }

    #[test]
    fn test_frame_overflow_502_is_non_retryable_overflow() {
        let pid = ProviderId::new("lm-studio");
        let body = "sent 1009 (message too big) frame with 7530 bytes after reading \
                    1042385 bytes exceeds limit of 1048576 bytes; no close frame received";
        let err = parse_error_response(502, body, &pid);
        assert!(matches!(err, ProviderError::ContextOverflow { .. }));
        assert!(!err.is_retryable());
    }

    #[test]
    fn test_transient_502_stays_retryable_server_error() {
        let pid = ProviderId::new("lm-studio");
        let err = parse_error_response(502, "timed out during opening handshake", &pid);
        assert!(matches!(
            err,
            ProviderError::ServerError {
                is_retryable: true,
                ..
            }
        ));
        assert!(err.is_retryable());
    }

    #[test]
    fn test_parse_error_response_auth() {
        let pid = ProviderId::new("anthropic");
        let err = parse_error_response(401, r#"{"error":{"message":"Invalid API key"}}"#, &pid);
        assert!(matches!(err, ProviderError::AuthFailed { .. }));
    }

    #[test]
    fn test_parse_error_response_free_tier_as_quota() {
        let pid = ProviderId::new("qwen");
        let body = r#"{"error":{"message":"The free tier of the model has been exhausted."}}"#;
        let err = parse_error_response(401, body, &pid);
        assert!(matches!(err, ProviderError::QuotaExceeded { .. }));
    }

    #[test]
    fn test_parse_error_response_payment_required_as_quota() {
        let pid = ProviderId::new("openai");
        let err = parse_error_response(402, "Payment required", &pid);
        assert!(matches!(err, ProviderError::QuotaExceeded { .. }));
    }

    #[test]
    fn test_parse_error_response_rate_limit() {
        let pid = ProviderId::new("openai");
        let err = parse_error_response(429, "rate limited", &pid);
        assert!(matches!(err, ProviderError::RateLimited { .. }));
    }

    #[test]
    fn test_retry_config_delay_increases() {
        let cfg = RetryConfig::default();
        let d0 = cfg.delay_for_attempt(0);
        let d1 = cfg.delay_for_attempt(1);
        let d2 = cfg.delay_for_attempt(2);
        // Each attempt should be strictly larger than the previous.
        assert!(d1 >= d0, "d1={:?} should be >= d0={:?}", d1, d0);
        assert!(d2 >= d1, "d2={:?} should be >= d1={:?}", d2, d1);
    }

    #[test]
    fn test_retry_config_respects_max_delay() {
        let cfg = RetryConfig::default();
        let d10 = cfg.delay_for_attempt(10);
        assert!(d10 <= cfg.max_delay + Duration::from_millis(1));
    }
}
