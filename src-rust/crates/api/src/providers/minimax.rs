// providers/minimax.rs — MiniMax provider wired through Anthropic protocol compatibility.

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use mangocode_core::provider_id::{ModelId, ProviderId};

use crate::client::ClientConfig;
use crate::provider::{LlmProvider, ModelInfo};
use crate::provider_error::ProviderError;
use crate::provider_types::{
    ProviderCapabilities, ProviderRequest, ProviderResponse, ProviderStatus, StreamEvent,
};

use super::anthropic::AnthropicProvider;

pub struct MinimaxProvider {
    inner: AnthropicProvider,
    id: ProviderId,
}

impl MinimaxProvider {
    pub fn new(api_key: String) -> Self {
        let api_base = std::env::var("MINIMAX_BASE_URL")
            .unwrap_or_else(|_| "https://api.minimax.io/anthropic".to_string());

        let inner = AnthropicProvider::from_config(ClientConfig {
            api_key,
            api_base,
            use_bearer_auth: true,
            ..Default::default()
        });

        Self {
            inner,
            id: ProviderId::new(ProviderId::MINIMAX),
        }
    }
}

#[async_trait]
impl LlmProvider for MinimaxProvider {
    fn id(&self) -> &ProviderId {
        &self.id
    }

    fn name(&self) -> &str {
        "MiniMax"
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
        let minimax_id = ProviderId::new(ProviderId::MINIMAX);
        Ok(vec![ModelInfo {
            id: ModelId::new("MiniMax-M2.7"),
            provider_id: minimax_id,
            name: "MiniMax M2.7".to_string(),
            context_window: 128_000,
            max_output_tokens: 8_192,
        }])
    }

    async fn health_check(&self) -> Result<ProviderStatus, ProviderError> {
        self.inner.health_check().await
    }

    fn capabilities(&self) -> ProviderCapabilities {
        let mut caps = self.inner.capabilities();
        caps.thinking = false;
        caps
    }
}
