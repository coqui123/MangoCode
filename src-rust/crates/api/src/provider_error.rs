// provider_error.rs — Unified error type for all provider adapters.
//
// Every provider implementation maps its own error representation onto
// `ProviderError` so that the application-layer code can handle errors
// generically without knowing which provider was involved.

use mangocode_core::error::ClaudeError;
use mangocode_core::provider_id::ProviderId;
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// ProviderError
// ---------------------------------------------------------------------------

/// A structured error produced by any provider adapter.
#[derive(Debug, Clone)]
pub enum ProviderError {
    /// The request exceeded the model's context window.
    ContextOverflow {
        provider: ProviderId,
        message: String,
        /// The provider's advertised context limit in tokens, if known.
        max_tokens: Option<u64>,
    },

    /// The provider returned HTTP 429 or an equivalent rate-limit signal.
    RateLimited {
        provider: ProviderId,
        /// How long to wait before retrying, in seconds (if provided).
        retry_after: Option<u64>,
    },

    /// The API key or credentials were rejected by the provider.
    AuthFailed {
        provider: ProviderId,
        message: String,
    },

    /// The account's usage quota has been exhausted.
    QuotaExceeded {
        provider: ProviderId,
        message: String,
    },

    /// The requested model does not exist or is not accessible.
    ModelNotFound {
        provider: ProviderId,
        model: String,
        /// Alternative model IDs the caller might try instead.
        suggestions: Vec<String>,
    },

    /// The provider returned a 5xx or equivalent server-side error.
    ServerError {
        provider: ProviderId,
        /// HTTP status code, if applicable.
        status: Option<u16>,
        message: String,
        /// Whether the caller should retry this request.
        is_retryable: bool,
    },

    /// The request itself was malformed or contained invalid parameters.
    InvalidRequest {
        provider: ProviderId,
        message: String,
    },

    /// The response was blocked by the provider's content-safety system.
    ContentFiltered {
        provider: ProviderId,
        message: String,
    },

    /// An error occurred during streaming after the response had already begun.
    StreamError {
        provider: ProviderId,
        message: String,
        /// Any content blocks that had been received before the error, if any.
        partial_response: Option<String>,
    },

    /// A catch-all variant for errors that do not fit any of the above.
    Other {
        provider: ProviderId,
        message: String,
        /// HTTP status code, if applicable.
        status: Option<u16>,
        /// Raw response body, if available.
        body: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderDiagnosticKind {
    Auth,
    ModelUnavailable,
    RateLimit,
    Quota,
    Context,
    Network,
    Safety,
    InvalidRequest,
    Server,
    Stream,
    Unknown,
}

impl ProviderDiagnosticKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auth => "auth",
            Self::ModelUnavailable => "model_unavailable",
            Self::RateLimit => "rate_limit",
            Self::Quota => "quota",
            Self::Context => "context",
            Self::Network => "network",
            Self::Safety => "safety",
            Self::InvalidRequest => "invalid_request",
            Self::Server => "server",
            Self::Stream => "stream",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderDiagnostic {
    pub provider: ProviderId,
    pub kind: ProviderDiagnosticKind,
    pub retryable: bool,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<String>,
}

// ---------------------------------------------------------------------------
// impl ProviderError
// ---------------------------------------------------------------------------

impl ProviderError {
    /// Returns `true` if the caller should retry the request after a delay.
    pub fn is_retryable(&self) -> bool {
        match self {
            ProviderError::RateLimited { .. } => true,
            ProviderError::ServerError { is_retryable, .. } => *is_retryable,
            ProviderError::StreamError { .. } => true,
            _ => false,
        }
    }

    /// Returns the `ProviderId` of the provider that produced this error.
    pub fn provider_id(&self) -> &ProviderId {
        match self {
            ProviderError::ContextOverflow { provider, .. } => provider,
            ProviderError::RateLimited { provider, .. } => provider,
            ProviderError::AuthFailed { provider, .. } => provider,
            ProviderError::QuotaExceeded { provider, .. } => provider,
            ProviderError::ModelNotFound { provider, .. } => provider,
            ProviderError::ServerError { provider, .. } => provider,
            ProviderError::InvalidRequest { provider, .. } => provider,
            ProviderError::ContentFiltered { provider, .. } => provider,
            ProviderError::StreamError { provider, .. } => provider,
            ProviderError::Other { provider, .. } => provider,
        }
    }

    pub fn diagnostic(&self) -> ProviderDiagnostic {
        match self {
            ProviderError::ContextOverflow {
                provider, message, ..
            } => self.base_diagnostic(provider, ProviderDiagnosticKind::Context, message, None),
            ProviderError::RateLimited {
                provider,
                retry_after,
            } => ProviderDiagnostic {
                provider: provider.clone(),
                kind: ProviderDiagnosticKind::RateLimit,
                retryable: true,
                message: match retry_after {
                    Some(secs) => format!("Rate limited; retry after {secs}s"),
                    None => "Rate limited".to_string(),
                },
                status: Some(429),
                retry_after: *retry_after,
                model: None,
                suggestions: Vec::new(),
            },
            ProviderError::AuthFailed { provider, message } => {
                self.base_diagnostic(provider, ProviderDiagnosticKind::Auth, message, None)
            }
            ProviderError::QuotaExceeded { provider, message } => {
                self.base_diagnostic(provider, ProviderDiagnosticKind::Quota, message, None)
            }
            ProviderError::ModelNotFound {
                provider,
                model,
                suggestions,
            } => ProviderDiagnostic {
                provider: provider.clone(),
                kind: ProviderDiagnosticKind::ModelUnavailable,
                retryable: false,
                message: format!("Model not found or inaccessible: {model}"),
                status: Some(404),
                retry_after: None,
                model: Some(model.clone()),
                suggestions: suggestions.clone(),
            },
            ProviderError::ServerError {
                provider,
                status,
                message,
                is_retryable,
            } => ProviderDiagnostic {
                provider: provider.clone(),
                kind: ProviderDiagnosticKind::Server,
                retryable: *is_retryable,
                message: message.clone(),
                status: *status,
                retry_after: None,
                model: None,
                suggestions: Vec::new(),
            },
            ProviderError::InvalidRequest { provider, message } => self.base_diagnostic(
                provider,
                ProviderDiagnosticKind::InvalidRequest,
                message,
                None,
            ),
            ProviderError::ContentFiltered { provider, message } => {
                self.base_diagnostic(provider, ProviderDiagnosticKind::Safety, message, None)
            }
            ProviderError::StreamError {
                provider, message, ..
            } => self.base_diagnostic(provider, ProviderDiagnosticKind::Stream, message, None),
            ProviderError::Other {
                provider,
                message,
                status,
                ..
            } => {
                let kind = if let Some(status) = status {
                    diagnostic_kind_for_status(*status, message)
                } else if let Some(kind) = diagnostic_kind_for_message(message) {
                    kind
                } else if message_looks_like_network_failure(message) {
                    ProviderDiagnosticKind::Network
                } else {
                    ProviderDiagnosticKind::Unknown
                };
                let mut diagnostic = self.base_diagnostic(provider, kind, message, *status);
                diagnostic.retryable =
                    diagnostic.retryable || status.map(status_is_retryable).unwrap_or(false);
                diagnostic
            }
        }
    }

    fn base_diagnostic(
        &self,
        provider: &ProviderId,
        kind: ProviderDiagnosticKind,
        message: &str,
        status: Option<u16>,
    ) -> ProviderDiagnostic {
        ProviderDiagnostic {
            provider: provider.clone(),
            kind,
            retryable: self.is_retryable(),
            message: message.to_string(),
            status,
            retry_after: None,
            model: None,
            suggestions: Vec::new(),
        }
    }
}

fn message_looks_like_network_failure(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    [
        "network",
        "connect",
        "connection",
        "dns",
        "tls",
        "certificate",
        "timed out",
        "timeout",
        "error sending request",
        "request failed",
        "tcp",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn message_looks_like_quota_failure(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    [
        "insufficient_quota",
        "quota",
        "billing",
        "payment required",
        "payment_required",
        "credits",
        "credit balance",
        "usage limit",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn message_looks_like_safety_failure(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    [
        "content_filter",
        "content filter",
        "content_policy",
        "content policy",
        "safety",
        "blocked by policy",
        "policy violation",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn message_looks_like_invalid_request(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    [
        "invalid_request",
        "invalid request",
        "invalid parameter",
        "bad request",
        "malformed",
        "unsupported parameter",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn message_looks_like_context_failure(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    [
        "context_length_exceeded",
        "context window",
        "context length",
        "too many tokens",
        "prompt is too long",
        "request too large",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn diagnostic_kind_for_message(message: &str) -> Option<ProviderDiagnosticKind> {
    if message_looks_like_quota_failure(message) {
        Some(ProviderDiagnosticKind::Quota)
    } else if message_looks_like_safety_failure(message) {
        Some(ProviderDiagnosticKind::Safety)
    } else if message_looks_like_context_failure(message) {
        Some(ProviderDiagnosticKind::Context)
    } else if message_looks_like_invalid_request(message) {
        Some(ProviderDiagnosticKind::InvalidRequest)
    } else if message_looks_like_network_failure(message) {
        Some(ProviderDiagnosticKind::Network)
    } else {
        None
    }
}

fn diagnostic_kind_for_status(status: u16, message: &str) -> ProviderDiagnosticKind {
    if message_looks_like_quota_failure(message) || status == 402 {
        return ProviderDiagnosticKind::Quota;
    }
    if message_looks_like_safety_failure(message) {
        return ProviderDiagnosticKind::Safety;
    }
    if message_looks_like_context_failure(message) || status == 413 {
        return ProviderDiagnosticKind::Context;
    }
    match status {
        400 | 422 => ProviderDiagnosticKind::InvalidRequest,
        401 | 403 => ProviderDiagnosticKind::Auth,
        404 => ProviderDiagnosticKind::ModelUnavailable,
        408 => ProviderDiagnosticKind::Network,
        429 => ProviderDiagnosticKind::RateLimit,
        500..=599 => ProviderDiagnosticKind::Server,
        _ => ProviderDiagnosticKind::Unknown,
    }
}

fn status_is_retryable(status: u16) -> bool {
    matches!(status, 408 | 429 | 500..=599)
}

pub fn diagnostic_from_claude_error(
    provider: Option<&str>,
    model: Option<&str>,
    err: &ClaudeError,
) -> ProviderDiagnostic {
    let provider = ProviderId::new(provider.unwrap_or("lmstudio"));
    let mut diagnostic = match err {
        ClaudeError::Auth(message) => ProviderDiagnostic {
            provider,
            kind: ProviderDiagnosticKind::Auth,
            retryable: false,
            message: message.clone(),
            status: Some(401),
            retry_after: None,
            model: model.map(str::to_string),
            suggestions: Vec::new(),
        },
        ClaudeError::RateLimit => ProviderDiagnostic {
            provider,
            kind: ProviderDiagnosticKind::RateLimit,
            retryable: true,
            message: "Rate limited".to_string(),
            status: Some(429),
            retry_after: None,
            model: model.map(str::to_string),
            suggestions: Vec::new(),
        },
        ClaudeError::ContextWindowExceeded => ProviderDiagnostic {
            provider,
            kind: ProviderDiagnosticKind::Context,
            retryable: false,
            message: "Context window exceeded".to_string(),
            status: None,
            retry_after: None,
            model: model.map(str::to_string),
            suggestions: Vec::new(),
        },
        ClaudeError::ApiStatus { status, message } => ProviderDiagnostic {
            provider,
            kind: diagnostic_kind_for_status(*status, message),
            retryable: err.is_retryable() || status_is_retryable(*status),
            message: message.clone(),
            status: Some(*status),
            retry_after: None,
            model: model.map(str::to_string),
            suggestions: Vec::new(),
        },
        ClaudeError::Http(http_err) => ProviderDiagnostic {
            provider,
            kind: ProviderDiagnosticKind::Network,
            retryable: http_err.is_timeout() || http_err.is_connect(),
            message: http_err.to_string(),
            status: http_err.status().map(|status| status.as_u16()),
            retry_after: None,
            model: model.map(str::to_string),
            suggestions: Vec::new(),
        },
        other => {
            let message = other.to_string();
            ProviderDiagnostic {
                provider,
                kind: diagnostic_kind_for_message(&message)
                    .unwrap_or(ProviderDiagnosticKind::Unknown),
                retryable: other.is_retryable(),
                message,
                status: None,
                retry_after: None,
                model: model.map(str::to_string),
                suggestions: Vec::new(),
            }
        }
    };
    if matches!(diagnostic.kind, ProviderDiagnosticKind::ModelUnavailable)
        && diagnostic.model.is_none()
    {
        diagnostic.model = model.map(str::to_string);
    }
    diagnostic
}

pub fn format_provider_diagnostic(diagnostic: &ProviderDiagnostic) -> String {
    let mut parts = Vec::new();
    let suggestions = recovery_suggestions(diagnostic);
    parts.push(format!(
        "Provider diagnostic: provider={}, kind={}",
        diagnostic.provider,
        diagnostic.kind.as_str()
    ));
    if let Some(model) = &diagnostic.model {
        parts.push(format!("model={model}"));
    }
    if let Some(status) = diagnostic.status {
        parts.push(format!("status={status}"));
    }
    if let Some(retry_after) = diagnostic.retry_after {
        parts.push(format!("retry_after={retry_after}s"));
    }
    parts.push(if diagnostic.retryable {
        "retryable=true".to_string()
    } else {
        "retryable=false".to_string()
    });
    if !suggestions.is_empty() {
        parts.push(format!("suggestions={}", suggestions.join(", ")));
    }
    parts.push(format!("message={}", diagnostic.message));
    parts.join("; ")
}

fn recovery_suggestions(diagnostic: &ProviderDiagnostic) -> Vec<String> {
    let mut suggestions = diagnostic.suggestions.clone();
    let provider = diagnostic.provider.as_ref();

    match diagnostic.kind {
        ProviderDiagnosticKind::Auth => {
            suggestions.push(auth_recovery_suggestion(provider).to_string());
        }
        ProviderDiagnosticKind::ModelUnavailable => {
            suggestions
                .push("Run /model to choose a model visible to the active provider.".to_string());
            if matches!(provider, "openai-codex" | "codex") {
                suggestions.push(
                    "Run /connect and choose OpenAI Codex if the model list is stale.".to_string(),
                );
            }
        }
        ProviderDiagnosticKind::RateLimit => {
            if let Some(secs) = diagnostic.retry_after {
                suggestions.push(format!("Wait at least {secs}s before retrying."));
            } else {
                suggestions.push("Wait briefly and retry the request.".to_string());
            }
            suggestions.push(
                "Use /model to switch providers or configure a fallback model for repeated limits."
                    .to_string(),
            );
        }
        ProviderDiagnosticKind::Quota => {
            suggestions.push(
                "Check provider quota or switch to another configured provider with /model."
                    .to_string(),
            );
        }
        ProviderDiagnosticKind::Context => {
            suggestions.push("Run /compact or narrow the task scope, then retry.".to_string());
        }
        ProviderDiagnosticKind::Network => {
            suggestions.push(
                "Check network connectivity, proxy/base URL settings, and provider status."
                    .to_string(),
            );
        }
        ProviderDiagnosticKind::Server | ProviderDiagnosticKind::Stream if diagnostic.retryable => {
            suggestions.push(
                "Retry the request; if it repeats, switch models or providers with /model."
                    .to_string(),
            );
        }
        ProviderDiagnosticKind::InvalidRequest => {
            suggestions.push(
                "Check model/provider compatibility, tool support, and request size.".to_string(),
            );
        }
        ProviderDiagnosticKind::Safety => {
            suggestions.push(
                "Revise the prompt or remove content blocked by the provider safety filter."
                    .to_string(),
            );
        }
        ProviderDiagnosticKind::Server
        | ProviderDiagnosticKind::Stream
        | ProviderDiagnosticKind::Unknown => {}
    }

    dedupe_suggestions(suggestions)
}

fn auth_recovery_suggestion(provider: &str) -> &'static str {
    match provider {
        "openai-codex" | "codex" => {
            "Run /connect and choose OpenAI Codex (OAuth); this is separate from OPENAI_API_KEY."
        }
        "openai" => "Set OPENAI_API_KEY or run /connect and choose OpenAI.",
        "anthropic" => "Run /login or configure an Anthropic API key.",
        "anthropic-max" => "Run /connect and choose Claude Max (OAuth).",
        "google" => "Set GOOGLE_API_KEY or run /connect and choose Google.",
        "github-copilot" => "Reconnect GitHub Copilot with /connect or set GITHUB_TOKEN.",
        "ollama" => "Start the local Ollama server and confirm OLLAMA_HOST if configured.",
        _ => "Set the provider API key or reconnect the provider with /connect.",
    }
}

fn dedupe_suggestions(suggestions: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    for suggestion in suggestions {
        let suggestion = suggestion.trim();
        if suggestion.is_empty() {
            continue;
        }
        if !out.iter().any(|existing: &String| existing == suggestion) {
            out.push(suggestion.to_string());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProviderError::ContextOverflow {
                provider,
                message,
                max_tokens,
            } => {
                write!(f, "[{}] Context overflow: {}", provider, message)?;
                if let Some(max) = max_tokens {
                    write!(f, " (max {} tokens)", max)?;
                }
                Ok(())
            }
            ProviderError::RateLimited {
                provider,
                retry_after,
            } => {
                write!(f, "[{}] Rate limited", provider)?;
                if let Some(secs) = retry_after {
                    write!(f, "; retry after {}s", secs)?;
                }
                Ok(())
            }
            ProviderError::AuthFailed { provider, message } => {
                write!(f, "[{}] Authentication failed: {}", provider, message)
            }
            ProviderError::QuotaExceeded { provider, message } => {
                write!(f, "[{}] Quota exceeded: {}", provider, message)
            }
            ProviderError::ModelNotFound {
                provider,
                model,
                suggestions,
            } => {
                write!(f, "[{}] Model not found: {}", provider, model)?;
                if !suggestions.is_empty() {
                    write!(f, " (suggestions: {})", suggestions.join(", "))?;
                }
                Ok(())
            }
            ProviderError::ServerError {
                provider,
                status,
                message,
                ..
            } => match status {
                Some(s) => write!(f, "[{}] Server error {}: {}", provider, s, message),
                None => write!(f, "[{}] Server error: {}", provider, message),
            },
            ProviderError::InvalidRequest { provider, message } => {
                write!(f, "[{}] Invalid request: {}", provider, message)
            }
            ProviderError::ContentFiltered { provider, message } => {
                write!(f, "[{}] Content filtered: {}", provider, message)
            }
            ProviderError::StreamError {
                provider, message, ..
            } => {
                write!(f, "[{}] Stream error: {}", provider, message)
            }
            ProviderError::Other {
                provider,
                message,
                status,
                ..
            } => match status {
                Some(s) => write!(f, "[{}] Error {}: {}", provider, s, message),
                None => write!(f, "[{}] Error: {}", provider, message),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// std::error::Error
// ---------------------------------------------------------------------------

impl std::error::Error for ProviderError {}

// ---------------------------------------------------------------------------
// From<ProviderError> for ClaudeError
// ---------------------------------------------------------------------------

impl From<ProviderError> for ClaudeError {
    fn from(err: ProviderError) -> Self {
        match &err {
            ProviderError::ContextOverflow { .. } => ClaudeError::ContextWindowExceeded,
            ProviderError::RateLimited { .. } => ClaudeError::RateLimit,
            ProviderError::AuthFailed { message, .. } => ClaudeError::Auth(message.clone()),
            ProviderError::QuotaExceeded { message, .. } => ClaudeError::ApiStatus {
                status: 402,
                message: message.clone(),
            },
            ProviderError::ModelNotFound {
                model, suggestions, ..
            } => {
                let message = if suggestions.is_empty() {
                    format!("Model not found or inaccessible: {model}")
                } else {
                    format!(
                        "Model not found or inaccessible: {model}; suggestions: {}",
                        suggestions.join(", ")
                    )
                };
                ClaudeError::ApiStatus {
                    status: 404,
                    message,
                }
            }
            ProviderError::ServerError {
                status: Some(s),
                message,
                ..
            } => ClaudeError::ApiStatus {
                status: *s,
                message: message.clone(),
            },
            ProviderError::InvalidRequest { message, .. } => ClaudeError::ApiStatus {
                status: 400,
                message: message.clone(),
            },
            ProviderError::ContentFiltered { message, .. } => ClaudeError::ApiStatus {
                status: 400,
                message: message.clone(),
            },
            ProviderError::Other {
                status: Some(status),
                message,
                ..
            } => ClaudeError::ApiStatus {
                status: *status,
                message: message.clone(),
            },
            _ => ClaudeError::Api(err.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_classifies_actionable_provider_errors() {
        let provider = ProviderId::new("openai");
        let err = ProviderError::ModelNotFound {
            provider: provider.clone(),
            model: "missing-model".to_string(),
            suggestions: vec!["gpt-5.1".to_string()],
        };

        let diagnostic = err.diagnostic();

        assert_eq!(diagnostic.provider, provider);
        assert_eq!(diagnostic.kind, ProviderDiagnosticKind::ModelUnavailable);
        assert_eq!(diagnostic.status, Some(404));
        assert_eq!(diagnostic.model.as_deref(), Some("missing-model"));
        assert_eq!(diagnostic.suggestions, vec!["gpt-5.1".to_string()]);
        assert!(!diagnostic.retryable);
    }

    #[test]
    fn diagnostic_classifies_likely_network_failures() {
        let err = ProviderError::Other {
            provider: ProviderId::new("anthropic"),
            message: "error sending request: dns lookup failed".to_string(),
            status: None,
            body: None,
        };

        let diagnostic = err.diagnostic();

        assert_eq!(diagnostic.kind, ProviderDiagnosticKind::Network);
        assert!(diagnostic.status.is_none());
        assert!(!diagnostic.retryable);
    }

    #[test]
    fn diagnostic_preserves_rate_limit_retry_after() {
        let err = ProviderError::RateLimited {
            provider: ProviderId::new("openai-codex"),
            retry_after: Some(30),
        };

        let diagnostic = err.diagnostic();

        assert_eq!(diagnostic.kind, ProviderDiagnosticKind::RateLimit);
        assert_eq!(diagnostic.status, Some(429));
        assert_eq!(diagnostic.retry_after, Some(30));
        assert!(diagnostic.retryable);
    }

    #[test]
    fn diagnostic_from_claude_error_classifies_payment_required_as_quota() {
        let err = ClaudeError::ApiStatus {
            status: 402,
            message: "Payment required: billing quota exhausted".to_string(),
        };

        let diagnostic = diagnostic_from_claude_error(Some("openai"), Some("gpt-5.1"), &err);

        assert_eq!(diagnostic.kind, ProviderDiagnosticKind::Quota);
        assert_eq!(diagnostic.status, Some(402));
        assert_eq!(diagnostic.model.as_deref(), Some("gpt-5.1"));
    }

    #[test]
    fn provider_error_conversion_preserves_quota_diagnostic_kind() {
        let provider = ProviderId::new("openai");
        let err = ProviderError::QuotaExceeded {
            provider: provider.clone(),
            message: "billing quota exhausted".to_string(),
        };
        let claude_error: ClaudeError = err.into();

        let diagnostic =
            diagnostic_from_claude_error(Some(provider.as_ref()), Some("gpt-5.1"), &claude_error);

        assert_eq!(diagnostic.kind, ProviderDiagnosticKind::Quota);
        assert_eq!(diagnostic.status, Some(402));
        assert_eq!(diagnostic.provider, provider);
    }

    #[test]
    fn provider_error_conversion_preserves_model_unavailable_kind() {
        let err = ProviderError::ModelNotFound {
            provider: ProviderId::new("openai"),
            model: "missing-model".to_string(),
            suggestions: vec!["gpt-5.1".to_string()],
        };
        let claude_error: ClaudeError = err.into();

        let diagnostic =
            diagnostic_from_claude_error(Some("openai"), Some("missing-model"), &claude_error);

        assert_eq!(diagnostic.kind, ProviderDiagnosticKind::ModelUnavailable);
        assert_eq!(diagnostic.status, Some(404));
        assert_eq!(diagnostic.model.as_deref(), Some("missing-model"));
    }

    #[test]
    fn provider_error_conversion_preserves_safety_kind() {
        let err = ProviderError::ContentFiltered {
            provider: ProviderId::new("openai"),
            message: "content policy violation".to_string(),
        };
        let claude_error: ClaudeError = err.into();

        let diagnostic =
            diagnostic_from_claude_error(Some("openai"), Some("gpt-5.1"), &claude_error);

        assert_eq!(diagnostic.kind, ProviderDiagnosticKind::Safety);
        assert_eq!(diagnostic.status, Some(400));
    }

    #[test]
    fn formats_claude_error_diagnostic_for_user_recovery() {
        let err = ClaudeError::ApiStatus {
            status: 529,
            message: "overloaded".to_string(),
        };
        let diagnostic =
            diagnostic_from_claude_error(Some("anthropic"), Some("claude-sonnet-4-5"), &err);

        assert_eq!(diagnostic.kind, ProviderDiagnosticKind::Server);
        assert_eq!(diagnostic.model.as_deref(), Some("claude-sonnet-4-5"));
        assert!(diagnostic.retryable);

        let rendered = format_provider_diagnostic(&diagnostic);
        assert!(rendered.contains("provider=anthropic"));
        assert!(rendered.contains("kind=server"));
        assert!(rendered.contains("retryable=true"));
        assert!(rendered.contains("Retry the request"));
    }

    #[test]
    fn formats_codex_auth_diagnostic_with_connect_recovery() {
        let diagnostic = ProviderDiagnostic {
            provider: ProviderId::new("openai-codex"),
            kind: ProviderDiagnosticKind::Auth,
            retryable: false,
            message: "missing credential".to_string(),
            status: Some(401),
            retry_after: None,
            model: Some("gpt-5.1-codex".to_string()),
            suggestions: Vec::new(),
        };

        let rendered = format_provider_diagnostic(&diagnostic);

        assert!(rendered.contains("provider=openai-codex"));
        assert!(rendered.contains("kind=auth"));
        assert!(rendered.contains("Run /connect and choose OpenAI Codex"));
    }
}
