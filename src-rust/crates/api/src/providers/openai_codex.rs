// providers/openai_codex.rs — OpenAI Codex via ChatGPT OAuth (Bearer token).
//
// Distinct from [`super::openai::OpenAiProvider`] (usage-based API key to api.openai.com).

use async_stream::stream;
use async_trait::async_trait;
use futures::Stream;
use mangocode_core::auth_store::{AuthStore, StoredCredential};
use mangocode_core::codex_oauth::{CODEX_API_ENDPOINT, CODEX_CLIENT_ID, CODEX_TOKEN_URL};
use mangocode_core::oauth_config::{get_codex_tokens, save_codex_tokens, CodexTokens};
use mangocode_core::provider_id::{ModelId, ProviderId};
use mangocode_core::types::ContentBlock;
use serde_json::Value;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::codex_adapter;
use crate::provider::{LlmProvider, ModelInfo};
use crate::provider_error::ProviderError;
use crate::provider_types::{
    ProviderCapabilities, ProviderRequest, ProviderResponse, ProviderStatus, StopReason,
    StreamEvent, SystemPromptStyle,
};
use crate::providers::anthropic::AnthropicProvider;
use crate::types::CreateMessageResponse;

const REFRESH_SKEW_SECS: i64 = 120;
const DISK_CHECK_INTERVAL_MS: i64 = 60_000;

/// OpenAI Codex using a ChatGPT-plan OAuth access token against the Codex
/// HTTP API (`chatgpt.com/backend-api/codex/...`).
pub struct OpenAiCodexProvider {
    id: ProviderId,
    http: reqwest::Client,
    /// Latest Bearer access token used for `Authorization` headers.
    bearer: Arc<Mutex<String>>,
    cached_expiry_ms: Arc<Mutex<Option<i64>>>,
    last_disk_check_ms: Arc<Mutex<Option<i64>>>,
}

impl OpenAiCodexProvider {
    pub fn new(access_token: String) -> Self {
        Self {
            id: ProviderId::new(ProviderId::OPENAI_CODEX),
            http: mangocode_core::vault::reqwest_client_builder()
                .timeout(std::time::Duration::from_secs(600))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            bearer: Arc::new(Mutex::new(access_token)),
            cached_expiry_ms: Arc::new(Mutex::new(None)),
            last_disk_check_ms: Arc::new(Mutex::new(None)),
        }
    }

    /// Build from `~/.mangocode/auth.json` OAuth entry for [`ProviderId::OPENAI_CODEX`].
    ///
    /// If `auth.json` has no Codex entry but `~/.mangocode/codex_tokens.json` exists from a
    /// prior MangoCode login, that file is merged into `auth.json` once so refresh metadata
    /// stays consistent (this is not the separate `~/.codex/auth.json` CLI file).
    pub fn from_auth_store() -> Option<Self> {
        let mut store = AuthStore::load();
        if let Some(StoredCredential::OAuthToken { access, .. }) =
            store.get(ProviderId::OPENAI_CODEX)
        {
            if !access.is_empty() {
                return Some(Self::new(access.clone()));
            }
        }
        let tokens = get_codex_tokens()?;
        if tokens.access_token.is_empty() {
            return None;
        }
        let expires_ms = tokens
            .expires_at
            .map(|s| s.saturating_mul(1000))
            .unwrap_or(0);
        store.set(
            ProviderId::OPENAI_CODEX,
            StoredCredential::OAuthToken {
                access: tokens.access_token.clone(),
                refresh: tokens.refresh_token.clone().unwrap_or_default(),
                expires: expires_ms,
            },
        );
        Some(Self::new(tokens.access_token))
    }

    fn map_stop_reason(s: &str) -> StopReason {
        match s {
            "end_turn" => StopReason::EndTurn,
            "stop_sequence" => StopReason::StopSequence,
            "max_tokens" => StopReason::MaxTokens,
            "tool_use" => StopReason::ToolUse,
            other => StopReason::Other(other.to_string()),
        }
    }

    fn cmr_to_provider_response(
        cmr: CreateMessageResponse,
        provider_id: &ProviderId,
    ) -> Result<ProviderResponse, ProviderError> {
        let mut content: Vec<ContentBlock> = Vec::new();
        for block in cmr.content {
            match serde_json::from_value::<ContentBlock>(block) {
                Ok(b) => content.push(b),
                Err(e) => {
                    return Err(ProviderError::Other {
                        provider: provider_id.clone(),
                        message: format!("Codex response content parse error: {}", e),
                        status: None,
                        body: None,
                    });
                }
            }
        }
        let stop_reason = cmr
            .stop_reason
            .as_deref()
            .map(Self::map_stop_reason)
            .unwrap_or(StopReason::EndTurn);
        Ok(ProviderResponse {
            id: cmr.id,
            content,
            stop_reason,
            usage: cmr.usage,
            model: cmr.model,
        })
    }

    async fn persist_oauth(access: &str, refresh: &str, expires_ms: u64) {
        let mut store = AuthStore::load_async().await;
        store.set(
            ProviderId::OPENAI_CODEX,
            StoredCredential::OAuthToken {
                access: access.to_string(),
                refresh: refresh.to_string(),
                expires: expires_ms,
            },
        );
        let tokens = CodexTokens {
            access_token: access.to_string(),
            refresh_token: if refresh.is_empty() {
                None
            } else {
                Some(refresh.to_string())
            },
            account_id: None,
            expires_at: if expires_ms > 0 {
                Some(expires_ms / 1000)
            } else {
                None
            },
        };
        let _ = save_codex_tokens(&tokens);
    }

    async fn refresh_tokens(refresh_token: &str) -> anyhow::Result<(String, Option<String>, u64)> {
        let client = reqwest::Client::new();
        let params = [
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("client_id", CODEX_CLIENT_ID),
        ];
        let resp = client
            .post(CODEX_TOKEN_URL)
            .form(&params)
            .send()
            .await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Codex token refresh failed ({}): {}", status, body);
        }
        let body: Value = resp.json().await?;
        if let Some(err) = body.get("error").and_then(|e| e.as_str()) {
            anyhow::bail!("Codex token refresh error: {}", err);
        }
        let access_token = body["access_token"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("refresh: missing access_token"))?
            .to_string();
        let new_refresh = body["refresh_token"].as_str().map(|s| s.to_string());
        let expires_in = body["expires_in"].as_i64().unwrap_or(3600);
        let now = chrono::Utc::now().timestamp();
        let expires_ms = (now + expires_in) * 1000;
        Ok((access_token, new_refresh, expires_ms as u64))
    }

    /// Refresh the access token when near expiry; updates disk and in-memory bearer.
    async fn ensure_fresh(&self) {
        let now_ms = chrono::Utc::now().timestamp_millis();
        let should_check_disk = {
            let last = self.last_disk_check_ms.lock().await;
            match *last {
                Some(t) => now_ms - t > DISK_CHECK_INTERVAL_MS,
                None => true,
            }
        };

        if !should_check_disk {
            let exp = *self.cached_expiry_ms.lock().await;
            if let Some(expiry) = exp {
                if now_ms + (REFRESH_SKEW_SECS * 1000) < expiry {
                    return;
                }
            }
        }

        *self.last_disk_check_ms.lock().await = Some(now_ms);

        let store = AuthStore::load_async().await;
        let Some(StoredCredential::OAuthToken {
            access,
            refresh,
            expires,
        }) = store.get(ProviderId::OPENAI_CODEX)
        else {
            return;
        };

        if *expires > 0 {
            *self.cached_expiry_ms.lock().await = Some(*expires as i64);
        }

        let exp_ms = *expires as i64;
        let needs_refresh = exp_ms > 0 && now_ms + (REFRESH_SKEW_SECS * 1000) >= exp_ms;
        if !needs_refresh {
            *self.bearer.lock().await = access.clone();
            return;
        }

        if refresh.is_empty() {
            tracing::warn!("OpenAI Codex OAuth: access token expired and no refresh token stored");
            *self.bearer.lock().await = access.clone();
            return;
        }

        match Self::refresh_tokens(refresh).await {
            Ok((new_access, new_refresh, expires_ms)) => {
                let rt = new_refresh
                    .as_deref()
                    .filter(|s| !s.is_empty())
                    .unwrap_or(refresh);
                Self::persist_oauth(&new_access, rt, expires_ms).await;
                *self.cached_expiry_ms.lock().await = Some(expires_ms as i64);
                *self.bearer.lock().await = new_access;
            }
            Err(e) => {
                tracing::warn!(error = %e, "OpenAI Codex OAuth refresh failed");
                let mut s = AuthStore::load_async().await;
                s.remove(ProviderId::OPENAI_CODEX);
                let _ = mangocode_core::oauth_config::clear_codex_tokens();
                *self.bearer.lock().await = String::new();
            }
        }
    }
}

#[async_trait]
impl LlmProvider for OpenAiCodexProvider {
    fn id(&self) -> &ProviderId {
        &self.id
    }

    fn name(&self) -> &str {
        "OpenAI Codex (OAuth)"
    }

    async fn create_message(
        &self,
        request: ProviderRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        self.ensure_fresh().await;
        let token = self.bearer.lock().await.clone();
        if token.is_empty() {
            return Err(ProviderError::AuthFailed {
                provider: self.id.clone(),
                message: "OpenAI Codex OAuth is not connected. Run /connect and choose OpenAI Codex (OAuth)."
                    .to_string(),
            });
        }

        let anthropic_req = AnthropicProvider::build_request(&request);
        let openai_body = codex_adapter::anthropic_to_openai_request(&anthropic_req);

        let resp = self
            .http
            .post(CODEX_API_ENDPOINT)
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .json(&openai_body)
            .send()
            .await
            .map_err(|e| ProviderError::Other {
                provider: self.id.clone(),
                message: format!("Codex request failed: {}", e),
                status: None,
                body: None,
            })?;

        let status = resp.status();
        let text = resp.text().await.map_err(|e| ProviderError::Other {
            provider: self.id.clone(),
            message: e.to_string(),
            status: None,
            body: None,
        })?;

        if !status.is_success() {
            if status.as_u16() == 401 {
                return Err(ProviderError::AuthFailed {
                    provider: self.id.clone(),
                    message: "OpenAI Codex OAuth token was rejected. Run /connect → OpenAI Codex (OAuth) to sign in again.".to_string(),
                });
            }
            return Err(ProviderError::Other {
                provider: self.id.clone(),
                message: format!("Codex API error ({})", status),
                status: Some(status.as_u16()),
                body: Some(text),
            });
        }

        let openai_resp: Value = serde_json::from_str(&text).map_err(|e| ProviderError::Other {
            provider: self.id.clone(),
            message: format!("Codex JSON parse error: {}", e),
            status: None,
            body: None,
        })?;

        let (content, stop_reason, input_tokens, output_tokens) =
            codex_adapter::parse_openai_response(&openai_resp);
        let cmr = codex_adapter::build_anthropic_response(
            &content,
            &stop_reason,
            input_tokens,
            output_tokens,
            &request.model,
        );
        Self::cmr_to_provider_response(cmr, &self.id)
    }

    async fn create_message_stream(
        &self,
        request: ProviderRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        // Query loop always uses the streaming path; Codex OAuth is non-streaming over HTTP,
        // so we synthesize a single-turn event sequence from `create_message`.
        let resp = self.create_message(request).await?;
        let id = resp.id.clone();
        let model = resp.model.clone();
        let usage = resp.usage.clone();
        let stop = resp.stop_reason.clone();

        let mut aggregate_text = String::new();
        for block in &resp.content {
            if let ContentBlock::Text { text } = block {
                if !aggregate_text.is_empty() {
                    aggregate_text.push('\n');
                }
                aggregate_text.push_str(text);
            }
        }

        let s = stream! {
            yield Ok(StreamEvent::MessageStart {
                id: id.clone(),
                model: model.clone(),
                usage: usage.clone(),
            });
            yield Ok(StreamEvent::ContentBlockStart {
                index: 0,
                content_block: ContentBlock::Text {
                    text: String::new(),
                },
            });
            if !aggregate_text.is_empty() {
                yield Ok(StreamEvent::TextDelta {
                    index: 0,
                    text: aggregate_text,
                });
            }
            yield Ok(StreamEvent::ContentBlockStop { index: 0 });
            yield Ok(StreamEvent::MessageDelta {
                stop_reason: Some(stop.clone()),
                usage: Some(usage.clone()),
            });
            yield Ok(StreamEvent::MessageStop);
        };

        Ok(Box::pin(s))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        let pid = self.id.clone();
        Ok(mangocode_core::codex_oauth::CODEX_MODELS
            .iter()
            .map(|(id, name)| ModelInfo {
                id: ModelId::new(*id),
                provider_id: pid.clone(),
                name: (*name).to_string(),
                context_window: 256_000,
                max_output_tokens: 32_000,
            })
            .collect())
    }

    async fn health_check(&self) -> Result<ProviderStatus, ProviderError> {
        self.ensure_fresh().await;
        if self.bearer.lock().await.is_empty() {
            return Ok(ProviderStatus::Unavailable {
                reason: "OpenAI Codex OAuth not configured".to_string(),
            });
        }
        Ok(ProviderStatus::Healthy)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            // True SSE from Codex is not implemented; the stream is synthesized in
            // `create_message_stream` so the query engine receives deltas.
            streaming: true,
            tool_calling: false,
            thinking: false,
            image_input: false,
            pdf_input: false,
            audio_input: false,
            video_input: false,
            caching: false,
            structured_output: false,
            system_prompt_style: SystemPromptStyle::TopLevel,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_codex_provider_uses_openai_codex_id() {
        let p = OpenAiCodexProvider::new("test-token".into());
        assert_eq!(format!("{}", p.id()), mangocode_core::ProviderId::OPENAI_CODEX);
    }
}
