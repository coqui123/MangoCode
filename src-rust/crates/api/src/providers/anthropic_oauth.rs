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

/// Enables Bearer OAuth on `api.anthropic.com` routes. Without this header,
/// Anthropic rejects OAuth tokens with:
///     "Authentication error: OAuth authentication is currently not supported."
const OAUTH_BETA: &str = "oauth-2025-04-20";

/// Anthropic expects this beta flag on non-Haiku `/v1/messages` calls so Bearer
/// OAuth bills against a Claude Max subscription (not console API-key quota)
/// and server-side Max features stay enabled.
const CLAUDE_CODE_BETA: &str = "claude-code-20250219";

/// Build a [`ClientConfig`] tuned for Claude Max OAuth:
/// - `use_bearer_auth = true` so the caller sends `Authorization: Bearer …`
/// - `beta_features` prepended with `oauth-2025-04-20,claude-code-20250219`
///   so Anthropic actually accepts the Bearer token.
fn max_client_config(bearer_token: String) -> ClientConfig {
    let defaults = ClientConfig::default();
    let combined_betas = format!(
        "{},{},{}",
        OAUTH_BETA, CLAUDE_CODE_BETA, defaults.beta_features
    );
    ClientConfig {
        api_key: bearer_token,
        use_bearer_auth: true,
        beta_features: combined_betas,
        ..defaults
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
