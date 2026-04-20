// providers/anthropic_oauth.rs — AnthropicMaxProvider: Claude Max (OAuth)
//
// Wraps the standard AnthropicProvider but authenticates using a Bearer token
// obtained via Anthropic's Claude.ai OAuth flow instead of an API key.
// This enables Claude Max subscription users to use their subscription
// without needing a separate API key from console.anthropic.com.
//
// The inner client is held behind a mutex so we can swap in a fresh access token
// after OAuth refresh (see `ensure_inner_fresh`) without rebuilding the registry.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;
use mangocode_core::auth_store::{AuthStore, StoredCredential};
use mangocode_core::oauth::{self as core_oauth, OAuthTokens};
use mangocode_core::provider_id::ProviderId;

use crate::client::ClientConfig;
use crate::provider::{LlmProvider, ModelInfo};
use crate::provider_error::ProviderError;
use crate::provider_types::{
    ProviderCapabilities, ProviderRequest, ProviderResponse, ProviderStatus, StreamEvent,
    SystemPromptStyle,
};

use super::anthropic::AnthropicProvider;

// ---------------------------------------------------------------------------
// Beta headers required for Claude Max OAuth
// ---------------------------------------------------------------------------

/// Canonical Claude Max beta list from reference implementations (Dario, Meridian, ccproxy).
/// This is the exact set of betas used by Claude Code for OAuth requests.
const MAX_BETAS: &str = "claude-code-20250219,oauth-2025-04-20,\
context-1m-2025-08-07,interleaved-thinking-2025-05-14,\
context-management-2025-06-27,prompt-caching-scope-2026-01-05,\
advisor-tool-2026-03-01,effort-2025-11-24";

/// Build a [`ClientConfig`] tuned for Claude Max OAuth:
/// - `use_bearer_auth = true` so the caller sends `Authorization: Bearer …`
/// - `beta_features` set to the canonical Claude Max beta list
///   so Anthropic actually accepts the Bearer token.
fn max_client_config(bearer_token: String) -> ClientConfig {
    ClientConfig {
        api_key: bearer_token,
        use_bearer_auth: true,
        beta_features: MAX_BETAS.to_string(),
        // Claude Max OAuth appears to have stricter burst limits than API keys.
        // Use a slightly more patient retry strategy so the first request after
        // OAuth doesn't "fail fast" on transient 429s.
        max_retries: 8,
        initial_retry_delay: std::time::Duration::from_secs(2),
        max_retry_delay: std::time::Duration::from_secs(300),
        ..ClientConfig::default()
    }
}

// ---------------------------------------------------------------------------
// AnthropicMaxProvider
// ---------------------------------------------------------------------------

/// Claude Max OAuth provider — wraps [`AnthropicProvider`] with Bearer token
/// authentication from the Anthropic Claude.ai OAuth flow.
///
/// The key differences from the standard `AnthropicProvider`:
/// - Uses `Authorization: Bearer <token>` instead of `x-api-key`
/// - The token comes from the auth store (`~/.mangocode/auth.json`) under
///   the `"anthropic-max"` key, stored as `StoredCredential::OAuthToken`
/// - Before each API call, if `~/.mangocode/oauth_tokens.json` holds a Bearer
///   token that is expired or expiring soon, we refresh, persist, sync
///   `auth.json`, and replace the inner provider so long sessions stay valid.
pub struct AnthropicMaxProvider {
    inner: Arc<tokio::sync::Mutex<AnthropicProvider>>,
    id: ProviderId,
}

impl AnthropicMaxProvider {
    fn from_inner(inner: AnthropicProvider) -> Self {
        Self {
            inner: Arc::new(tokio::sync::Mutex::new(inner)),
            id: ProviderId::new(ProviderId::ANTHROPIC_MAX),
        }
    }

    /// Create a new `AnthropicMaxProvider` from a Bearer token.
    ///
    /// The token is the `access` field from the stored `OAuthToken` credential.
    /// The inner `AnthropicProvider` is configured to use this token as the
    /// API key with Bearer auth mode **and** with the `oauth-2025-04-20` beta
    /// header — without that beta, Anthropic returns `401 "OAuth
    /// authentication is currently not supported."`.
    pub fn new(bearer_token: String) -> Self {
        Self::from_inner(AnthropicProvider::from_config(max_client_config(
            bearer_token,
        )))
    }

    /// Try to create from the auth store. Returns `None` if no valid
    /// `anthropic-max` credential is stored.
    pub fn from_auth_store() -> Option<Self> {
        let store = AuthStore::load();
        match store.get("anthropic-max") {
            Some(StoredCredential::OAuthToken { access, .. }) if !access.is_empty() => {
                Some(Self::new(access.clone()))
            }
            _ => None,
        }
    }

    /// Try to create from Claude CLI credentials as a fallback.
    /// This reads from ~/.claude/.credentials.json if MangoCode has no stored token.
    pub async fn from_claude_cli_fallback() -> Option<Self> {
        // First check if we already have a stored credential
        if Self::from_auth_store().is_some() {
            return None;
        }

        // Try to load from Claude CLI credentials
        let tokens = core_oauth::OAuthTokens::load_from_claude_cli().await?;
        if tokens.uses_bearer_auth() {
            let access = tokens.effective_credential()?;
            Some(Self::new(access.to_string()))
        } else {
            None
        }
    }

    /// If OAuth tokens on disk are Claude Max (Bearer) and near expiry, refresh,
    /// persist, sync `auth.json`, and rebuild the inner Anthropic client.
    async fn ensure_inner_fresh(&self) {
        let Some(tokens) = OAuthTokens::load().await else {
            return;
        };
        if !tokens.uses_bearer_auth() {
            return;
        }

        if !tokens.is_expired_or_expiring_soon() {
            return;
        }
        if tokens
            .refresh_token
            .as_ref()
            .map(|s| s.is_empty())
            .unwrap_or(true)
        {
            return;
        }

        match core_oauth::refresh_oauth_tokens_from_refresh(&tokens).await {
            Ok(updated) => {
                if updated.persist_to_disk_with_auth_sync().await.is_err() {
                    tracing::warn!("Claude Max: refreshed tokens but failed to persist to disk");
                }
                let new_inner = AnthropicProvider::from_config(max_client_config(
                    updated.access_token.clone(),
                ));
                *self.inner.lock().await = new_inner;
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "Claude Max OAuth refresh failed; falling back to auth.json access token if present"
                );
                if let Some(StoredCredential::OAuthToken { access, .. }) =
                    AuthStore::load().get(ProviderId::ANTHROPIC_MAX)
                {
                    if !access.is_empty() {
                        *self.inner.lock().await =
                            AnthropicProvider::from_config(max_client_config(access.clone()));
                    }
                }
            }
        }
    }
}

#[async_trait]
impl LlmProvider for AnthropicMaxProvider {
    fn id(&self) -> &ProviderId {
        &self.id
    }

    fn name(&self) -> &str {
        "Anthropic (Claude Max)"
    }

    async fn create_message(
        &self,
        request: ProviderRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        self.ensure_inner_fresh().await;
        let inner = self.inner.lock().await.clone();
        inner.create_message(request).await
    }

    async fn create_message_stream(
        &self,
        request: ProviderRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        self.ensure_inner_fresh().await;
        let inner = self.inner.lock().await.clone();
        inner.create_message_stream(request).await
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        self.ensure_inner_fresh().await;
        let inner = self.inner.lock().await.clone();
        inner.list_models().await
    }

    async fn health_check(&self) -> Result<ProviderStatus, ProviderError> {
        self.ensure_inner_fresh().await;
        let inner = self.inner.lock().await.clone();
        inner.health_check().await
    }

    fn capabilities(&self) -> ProviderCapabilities {
        // Same as [`AnthropicProvider`]; independent of the current access token.
        ProviderCapabilities {
            streaming: true,
            tool_calling: true,
            thinking: true,
            image_input: true,
            pdf_input: true,
            audio_input: false,
            video_input: false,
            caching: true,
            structured_output: true,
            system_prompt_style: SystemPromptStyle::TopLevel,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_betas_contains_required_features() {
        // Verify MAX_BETAS contains the essential OAuth and Claude Code betas
        assert!(MAX_BETAS.contains("oauth-2025-04-20"));
        assert!(MAX_BETAS.contains("claude-code-20250219"));
    }

    #[test]
    fn test_max_client_config_uses_canonical_betas() {
        let config = max_client_config("test_token".to_string());
        assert_eq!(config.beta_features, MAX_BETAS);
        assert!(config.use_bearer_auth);
    }
}
