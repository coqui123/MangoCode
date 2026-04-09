//! Shared HTTP retry middleware for all LLM providers.
//!
//! Retries requests on 429 (rate limit) and 529 (overloaded) responses with
//! exponential backoff.  Reads the `Retry-After` header when present and falls
//! back to the configured backoff schedule otherwise.
//!
//! An optional status callback notifies the UI/caller about retry waits so
//! the user sees "Rate limited — waiting 30s (attempt 2/8)" instead of a
//! silent stall.

use std::future::Future;
use std::time::Duration;
use tracing::warn;

use crate::error_handling::RetryConfig;

/// Execute an HTTP request with automatic retry on 429/529.
///
/// `make_request` is called on each attempt and must produce a fresh
/// [`reqwest::Response`].  The closure receives the 0-based attempt index so
/// callers can recompute auth signatures if needed.
///
/// `on_retry` is called before each wait with a human-readable status message.
/// Pass `|_| {}` if no UI notification is needed.
///
/// On success (any non-429/529 status) the response is returned immediately.
/// On 429/529, the middleware honours the `Retry-After` header and falls back
/// to exponential backoff from `config`.  After all retries are exhausted the
/// **last** response is returned so callers can still inspect the body.
pub async fn retry_request<F, Fut, S>(
    config: &RetryConfig,
    provider_name: &str,
    make_request: F,
    on_retry: S,
) -> Result<reqwest::Response, reqwest::Error>
where
    F: Fn(u32) -> Fut,
    Fut: Future<Output = Result<reqwest::Response, reqwest::Error>>,
    S: Fn(String),
{
    for attempt in 0..=config.max_retries {
        let resp = make_request(attempt).await?;
        let status = resp.status().as_u16();

        if status != 429 && status != 529 {
            return Ok(resp);
        }

        // Last attempt — don't sleep, just return the error response.
        if attempt == config.max_retries {
            return Ok(resp);
        }

        // Read Retry-After header before consuming the response.
        let retry_after = extract_retry_after(resp.headers());
        let wait = retry_after.unwrap_or_else(|| config.delay_for_attempt(attempt));
        let wait_secs = wait.as_secs();

        let msg = if wait_secs >= 60 {
            format!(
                "Rate limited by {} — waiting {}m {}s before retry (attempt {}/{})",
                provider_name,
                wait_secs / 60,
                wait_secs % 60,
                attempt + 1,
                config.max_retries,
            )
        } else {
            format!(
                "Rate limited by {} — waiting {}s before retry (attempt {}/{})",
                provider_name,
                wait_secs,
                attempt + 1,
                config.max_retries,
            )
        };

        warn!(
            provider = provider_name,
            status,
            attempt = attempt + 1,
            max_retries = config.max_retries,
            wait_secs = wait.as_secs_f64(),
            "Rate limited, retrying after backoff"
        );

        on_retry(msg);
        tokio::time::sleep(wait).await;
    }

    unreachable!("retry loop always returns from within the loop")
}

/// Parse the `Retry-After` response header (integer seconds).
fn extract_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    headers
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_retry_after() {
        let mut headers = reqwest::header::HeaderMap::new();
        assert!(extract_retry_after(&headers).is_none());

        headers.insert("retry-after", "5".parse().unwrap());
        assert_eq!(extract_retry_after(&headers), Some(Duration::from_secs(5)));
    }
}
