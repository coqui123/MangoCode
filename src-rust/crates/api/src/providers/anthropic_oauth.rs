// providers/anthropic_oauth.rs — AnthropicMaxProvider: Claude Max (OAuth)
//
// Wraps the standard AnthropicProvider but authenticates using a Bearer token
// obtained via Anthropic's Claude.ai OAuth flow instead of an API key.
// This enables Claude Max subscription users to use their subscription
// without needing a separate API key from console.anthropic.com.

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use mangocode_core::provider_id::ProviderId;

use crate::client::ClientConfig;
use crate::provider::{LlmProvider, ModelInfo};
use crate::provider_error::ProviderError;
use crate::provider_types::{
    ProviderCapabilities, ProviderRequest, ProviderResponse, ProviderStatus, StreamEvent,
};

use super::anthropic::AnthropicProvider;

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
/// - Token refresh is handled by the CLI layer before requests are made
pub struct AnthropicMaxProvider {
    inner: AnthropicProvider,
    id: ProviderId,
}

impl AnthropicMaxProvider {
    /// Create a new `AnthropicMaxProvider` from a Bearer token.
    ///
    /// The token is the `access` field from the stored `OAuthToken` credential.
    /// The inner `AnthropicProvider` is configured to use this token as the
    /// API key with Bearer auth mode.
    pub fn new(bearer_token: String) -> Self {
        let config = ClientConfig {
            api_key: bearer_token,
            base_url: Some("https://api.anthropic.com".to_string()),
            use_bearer_auth: true,
            ..Default::default()
        };
        Self {
            inner: AnthropicProvider::from_config(config),
            id: ProviderId::new(ProviderId::ANTHROPIC_MAX),
        }
    }

    /// Try to create from the auth store. Returns `None` if no valid
    /// `anthropic-max` credential is stored.
    pub fn from_auth_store() -> Option<Self> {
        use mangocode_core::auth_store::{AuthStore, StoredCredential};
        let store = AuthStore::load();
        match store.get("anthropic-max") {
            Some(StoredCredential::OAuthToken { access, .. }) if !access.is_empty() => {
                Some(Self::new(access.clone()))
            }
            _ => None,
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
        self.inner.create_message(request).await
    }

    async fn create_message_stream(
        &self,
        request: ProviderRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        self.inner.create_message_stream(request).await
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        self.inner.list_models().await
    }

    async fn health_check(&self) -> Result<ProviderStatus, ProviderError> {
        self.inner.health_check().await
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.inner.capabilities()
    }
}
