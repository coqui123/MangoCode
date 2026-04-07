// providers/vertex_openai.rs — Google Cloud Vertex AI via the OpenAI-compatible
// Chat Completions endpoint.
//
// Vertex AI exposes an OpenAI-compatible API at:
//   https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/openapi
//
// Authentication uses Google Cloud Bearer tokens (not static API keys).
// Three modes are supported:
//   - Adc / GcloudCommand: shell out to `gcloud auth application-default print-access-token`
//   - AccessToken: use a statically provided token (useful for CI/testing)
//
// The provider reuses all OpenAI-compat message/request helpers and SSE streaming
// from OpenAiProvider, only swapping the base URL and auth layer.

use std::pin::Pin;

use async_stream::stream;
use async_trait::async_trait;
use futures::Stream;
use mangocode_core::provider_id::{ModelId, ProviderId};
use mangocode_core::types::{ContentBlock, UsageInfo};
use serde_json::{json, Value};
use tracing::debug;

use super::openai::OpenAiProvider;
use super::request_options::merge_openai_compatible_options;
use crate::error_handling::parse_error_response;
use crate::provider::{LlmProvider, ModelInfo};
use crate::provider_error::ProviderError;
use crate::provider_types::{
    ProviderCapabilities, ProviderRequest, ProviderResponse, ProviderStatus, StreamEvent,
    SystemPromptStyle,
};

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// How to obtain a Google Cloud Bearer token for Vertex AI.
#[derive(Debug, Clone)]
pub enum VertexAuthMode {
    /// Use Application Default Credentials via `gcloud auth application-default
    /// print-access-token`.  This is the recommended mode for local development.
    Adc,
    /// Same as Adc — shells out to `gcloud auth application-default print-access-token`.
    GcloudCommand,
    /// Use a statically provided token.  Useful for CI/CD pipelines where the
    /// token is injected as an environment variable.
    AccessToken,
}

/// Configuration for the Vertex AI OpenAI-compatible provider.
#[derive(Debug, Clone)]
pub struct VertexConfig {
    pub project_id: String,
    pub location: String,
    pub model: String,
    pub auth_mode: VertexAuthMode,
    /// Required when `auth_mode == AccessToken`.
    pub access_token: Option<String>,
    /// Override the base URL (omit the `/chat/completions` suffix).
    pub base_url_override: Option<String>,
}

impl VertexConfig {
    /// Load configuration from environment variables.
    ///
    /// Returns `None` if `VERTEX_PROJECT_ID` is not set.
    /// `VERTEX_ENABLED` is accepted for backwards-compatibility but
    /// is no longer required — having a project ID is sufficient.
    pub fn from_env() -> Option<Self> {
        let project_id = std::env::var("VERTEX_PROJECT_ID").ok()?;
        if project_id.is_empty() {
            return None;
        }

        let location =
            std::env::var("VERTEX_LOCATION").unwrap_or_else(|_| "us-central1".to_string());

        let model =
            std::env::var("VERTEX_MODEL").unwrap_or_else(|_| "google/gemini-2.5-flash".to_string());

        let auth_mode = match std::env::var("VERTEX_AUTH_MODE")
            .as_deref()
            .unwrap_or("adc")
        {
            "access_token" | "AccessToken" => VertexAuthMode::AccessToken,
            "gcloud" | "GcloudCommand" => VertexAuthMode::GcloudCommand,
            _ => VertexAuthMode::Adc,
        };

        let access_token = std::env::var("VERTEX_ACCESS_TOKEN").ok();
        let base_url_override = std::env::var("VERTEX_BASE_URL").ok();

        Some(Self {
            project_id,
            location,
            model,
            auth_mode,
            access_token,
            base_url_override,
        })
    }
}

// ---------------------------------------------------------------------------
// URL builder
// ---------------------------------------------------------------------------

/// Build the Vertex AI OpenAI-compatible base URL.
///
/// Pattern: `https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi`
///
/// Append `/chat/completions` for inference requests.
pub fn vertex_openai_base_url(project_id: &str, location: &str) -> String {
    format!(
        "https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi",
        location = location,
        project_id = project_id,
    )
}

// ---------------------------------------------------------------------------
// Bearer token provider
// ---------------------------------------------------------------------------

/// Fetch a Google Cloud Bearer token according to the configured auth mode.
async fn fetch_bearer_token(config: &VertexConfig) -> Result<String, String> {
    match &config.auth_mode {
        VertexAuthMode::AccessToken => config
            .access_token
            .clone()
            .filter(|t| !t.is_empty())
            .ok_or_else(|| "VERTEX_ACCESS_TOKEN is not set or empty".to_string()),
        VertexAuthMode::Adc | VertexAuthMode::GcloudCommand => fetch_gcloud_token().await,
    }
}

/// Locate the `gcloud` executable.  Checks PATH first, then falls back to
/// well-known Windows and macOS/Linux Cloud SDK install paths.
fn find_gcloud() -> Option<std::path::PathBuf> {
    // 1. Try plain "gcloud" on PATH (works on Linux/macOS and Windows if SDK is in PATH).
    if which_gcloud("gcloud") {
        return Some("gcloud".into());
    }
    // 2. Windows: Cloud SDK default install locations.
    #[cfg(target_os = "windows")]
    {
        let candidates = [
            // Per-user install (most common)
            std::env::var("LOCALAPPDATA")
                .map(|p| format!("{}/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd", p))
                .unwrap_or_default(),
            // System-wide install
            "C:/Program Files (x86)/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd".to_string(),
            "C:/Program Files/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd".to_string(),
        ];
        for c in &candidates {
            if !c.is_empty() && std::path::Path::new(c).exists() {
                return Some(c.into());
            }
        }
    }
    // 3. macOS: homebrew / default SDK path.
    #[cfg(target_os = "macos")]
    {
        let candidates = [
            "/usr/local/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/bin/gcloud",
            "/opt/homebrew/bin/gcloud",
            "/usr/local/bin/gcloud",
        ];
        for c in &candidates {
            if std::path::Path::new(c).exists() {
                return Some((*c).into());
            }
        }
    }
    None
}

/// Returns true if `name` resolves successfully via a quick `--version` call.
fn which_gcloud(name: &str) -> bool {
    std::process::Command::new(name)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Shell out to `gcloud auth application-default print-access-token` and
/// return the trimmed token string.
async fn fetch_gcloud_token() -> Result<String, String> {
    let gcloud = find_gcloud().ok_or_else(|| {
        "gcloud CLI not found. Install the Google Cloud SDK: https://cloud.google.com/sdk/docs/install".to_string()
    })?;

    let output = tokio::process::Command::new(&gcloud)
        .args(["auth", "application-default", "print-access-token"])
        .output()
        .await
        .map_err(|e| format!("Failed to run gcloud: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "gcloud auth application-default print-access-token failed: {}. \
             Run `gcloud auth application-default login` to authenticate.",
            stderr.trim()
        ));
    }

    let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if token.is_empty() {
        return Err("gcloud returned an empty access token. \
             Run `gcloud auth application-default login` to authenticate."
            .to_string());
    }
    Ok(token)
}

// ---------------------------------------------------------------------------
// Provider struct
// ---------------------------------------------------------------------------

pub struct VertexOpenAiProvider {
    id: ProviderId,
    config: VertexConfig,
    base_url: String,
    http_client: reqwest::Client,
}

impl VertexOpenAiProvider {
    /// Resolve the model to use for a request. Vertex requires `publisher/model`
    /// format (e.g. `google/gemini-2.5-flash`). If the caller passes a bare
    /// model name (e.g. `claude-opus-4-6` left over from a previous provider),
    /// fall back to the configured default.
    fn resolve_model<'a>(&'a self, requested: &'a str) -> &'a str {
        if requested.contains('/') {
            requested
        } else {
            &self.config.model
        }
    }
}

impl VertexOpenAiProvider {
    pub fn new(config: VertexConfig) -> Self {
        let base_url = config
            .base_url_override
            .clone()
            .unwrap_or_else(|| vertex_openai_base_url(&config.project_id, &config.location));

        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            id: ProviderId::new(ProviderId::GOOGLE_VERTEX),
            config,
            base_url,
            http_client,
        }
    }

    /// Build from environment variables. Returns `None` if Vertex is not
    /// configured.
    pub fn from_env() -> Option<Self> {
        VertexConfig::from_env().map(Self::new)
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn map_http_error(&self, status: u16, body: &str) -> ProviderError {
        // Rewrite 401/403 to a clear, actionable message.
        if status == 401 || status == 403 {
            return ProviderError::AuthFailed {
                provider: self.id.clone(),
                message: format!(
                    "Vertex auth failed (HTTP {}). Run `gcloud auth application-default login` \
                     to authenticate, or set VERTEX_ACCESS_TOKEN.",
                    status
                ),
            };
        }
        parse_error_response(status, body, &self.id)
    }

    async fn bearer_token(&self) -> Result<String, ProviderError> {
        fetch_bearer_token(&self.config)
            .await
            .map_err(|msg| ProviderError::AuthFailed {
                provider: self.id.clone(),
                message: msg,
            })
    }

    // -----------------------------------------------------------------------
    // Non-streaming
    // -----------------------------------------------------------------------

    async fn create_message_non_streaming(
        &self,
        request: &ProviderRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        let token = self.bearer_token().await?;
        let model = self.resolve_model(&request.model).to_string();

        let messages = OpenAiProvider::to_openai_messages_pub(
            &request.messages,
            request.system_prompt.as_ref(),
        );
        let tools = OpenAiProvider::to_openai_tools_pub(&request.tools);

        let mut body = json!({
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": messages,
            "stream": false,
        });

        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
        if let Some(t) = request.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(p) = request.top_p {
            body["top_p"] = json!(p);
        }
        if !request.stop_sequences.is_empty() {
            body["stop"] = json!(request.stop_sequences);
        }

        // Vertex-specific overrides via provider_options.google
        apply_vertex_extras(&mut body, &request.provider_options);
        merge_openai_compatible_options(&mut body, &request.provider_options);

        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        let resp = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::Other {
                provider: self.id.clone(),
                message: format!("HTTP request failed: {}", e),
                status: None,
                body: None,
            })?;

        let status = resp.status().as_u16();
        let text = resp.text().await.map_err(|e| ProviderError::Other {
            provider: self.id.clone(),
            message: format!("Failed to read response body: {}", e),
            status: Some(status),
            body: None,
        })?;

        if !(200..300).contains(&(status as usize)) {
            return Err(self.map_http_error(status, &text));
        }

        let json: Value = serde_json::from_str(&text).map_err(|e| ProviderError::Other {
            provider: self.id.clone(),
            message: format!("Failed to parse response JSON: {}", e),
            status: Some(status),
            body: Some(text.clone()),
        })?;

        OpenAiProvider::parse_non_streaming_response_pub(&json, &self.id)
    }

    // -----------------------------------------------------------------------
    // Streaming
    // -----------------------------------------------------------------------

    async fn do_streaming(
        &self,
        request: &ProviderRequest,
    ) -> Result<reqwest::Response, ProviderError> {
        let token = self.bearer_token().await?;
        let model = self.resolve_model(&request.model).to_string();

        let messages = OpenAiProvider::to_openai_messages_pub(
            &request.messages,
            request.system_prompt.as_ref(),
        );
        let tools = OpenAiProvider::to_openai_tools_pub(&request.tools);

        let mut body = json!({
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": messages,
            "stream": true,
            "stream_options": { "include_usage": true },
        });

        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
        if let Some(t) = request.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(p) = request.top_p {
            body["top_p"] = json!(p);
        }
        if !request.stop_sequences.is_empty() {
            body["stop"] = json!(request.stop_sequences);
        }

        apply_vertex_extras(&mut body, &request.provider_options);
        merge_openai_compatible_options(&mut body, &request.provider_options);

        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        let resp = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::Other {
                provider: self.id.clone(),
                message: format!("HTTP request failed: {}", e),
                status: None,
                body: None,
            })?;

        let status = resp.status().as_u16();
        if !(200..300).contains(&(status as usize)) {
            let text = resp.text().await.unwrap_or_default();
            return Err(self.map_http_error(status, &text));
        }

        Ok(resp)
    }
}

// ---------------------------------------------------------------------------
// Vertex-specific request extras
// ---------------------------------------------------------------------------

/// Build a human-readable display name from a raw Gemini model ID.
///
/// Examples:
///   `gemini-2.5-flash`          → `"Gemini 2.5 Flash"`
///   `gemini-2.0-flash-lite-001` → `"Gemini 2.0 Flash Lite"`
///   `gemini-3-pro-preview`      → `"Gemini 3 Pro (Preview)"`
fn gemini_display_name(id: &str) -> String {
    // Strip trailing "-NNN" version suffixes like "-001", "-002"
    let without_ver = {
        let parts: Vec<&str> = id.split('-').collect();
        if parts
            .last()
            .map(|p| p.chars().all(|c| c.is_ascii_digit()))
            .unwrap_or(false)
        {
            parts[..parts.len() - 1].join("-")
        } else {
            id.to_string()
        }
    };

    // Capitalise known tokens; leave others title-cased
    let mut words: Vec<String> = Vec::new();
    let mut preview = false;
    for part in without_ver.split('-') {
        match part {
            "gemini" => words.push("Gemini".into()),
            "flash" => words.push("Flash".into()),
            "pro" => words.push("Pro".into()),
            "lite" => words.push("Lite".into()),
            "ultra" => words.push("Ultra".into()),
            "nano" => words.push("Nano".into()),
            "preview" => {
                preview = true;
            }
            other => words.push(other.to_string()),
        }
    }
    let mut name = words.join(" ");
    if preview {
        name.push_str(" (Preview)");
    }
    name
}

/// Infer the context window size from the model ID.
fn gemini_context_window(id: &str) -> u32 {
    if id.contains("1.5-pro") {
        2_000_000
    } else {
        1_000_000
    }
}

/// Build a human-readable display name for any Vertex publisher model.
fn vertex_model_display_name(publisher: &str, model_id: &str) -> String {
    match publisher {
        "google" => gemini_display_name(model_id),
        "anthropic" => {
            // "claude-sonnet-4-6" → "Claude Sonnet 4.6 (Vertex)"
            let pretty = model_id
                .split('-')
                .map(|p| {
                    let mut c = p.chars();
                    match c.next() {
                        None => String::new(),
                        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");
            format!("{} (Vertex)", pretty)
        }
        "mistralai" => {
            let pretty = model_id
                .split('-')
                .map(|p| {
                    let mut c = p.chars();
                    match c.next() {
                        None => String::new(),
                        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");
            format!("{} (Vertex)", pretty)
        }
        "meta" => {
            let pretty = model_id.replace(['-', '_'], " ");
            format!("{} (Vertex)", pretty)
        }
        "nvidia" => {
            let pretty = model_id
                .split('-')
                .map(|p| {
                    let mut c = p.chars();
                    c.next()
                        .map(|f| f.to_uppercase().collect::<String>() + c.as_str())
                        .unwrap_or_default()
                })
                .collect::<Vec<_>>()
                .join(" ");
            format!("{} (Vertex)", pretty)
        }
        "ai21" => {
            let pretty = model_id
                .split('-')
                .map(|p| {
                    let mut c = p.chars();
                    c.next()
                        .map(|f| f.to_uppercase().collect::<String>() + c.as_str())
                        .unwrap_or_default()
                })
                .collect::<Vec<_>>()
                .join(" ");
            format!("{} (Vertex)", pretty)
        }
        "writer" => {
            let pretty = model_id
                .split('-')
                .map(|p| {
                    let mut c = p.chars();
                    c.next()
                        .map(|f| f.to_uppercase().collect::<String>() + c.as_str())
                        .unwrap_or_default()
                })
                .collect::<Vec<_>>()
                .join(" ");
            format!("{} (Vertex)", pretty)
        }
        _ => format!("{}/{}", publisher, model_id),
    }
}

/// Merge `provider_options.google` into `extra_body.google` on the request body
/// (e.g. safety_settings, thinking_config).
fn apply_vertex_extras(body: &mut Value, provider_options: &Value) {
    if let Some(google_opts) = provider_options.get("google") {
        if !google_opts.is_null() {
            body["extra_body"] = json!({ "google": google_opts });
        }
    }
}

// ---------------------------------------------------------------------------
// Inherent helpers (not part of the trait)
// ---------------------------------------------------------------------------

impl VertexOpenAiProvider {
    /// Static fallback model list used when the Publisher Models API is
    /// unreachable or returns no results.
    fn fallback_models(&self) -> Vec<ModelInfo> {
        let pid = self.id.clone();
        vec![
            ModelInfo {
                id: ModelId::new("google/gemini-2.5-flash"),
                provider_id: pid.clone(),
                name: "Gemini 2.5 Flash".into(),
                context_window: 1_000_000,
                max_output_tokens: 8_192,
            },
            ModelInfo {
                id: ModelId::new("google/gemini-2.5-pro"),
                provider_id: pid.clone(),
                name: "Gemini 2.5 Pro".into(),
                context_window: 1_000_000,
                max_output_tokens: 8_192,
            },
            ModelInfo {
                id: ModelId::new("google/gemini-2.0-flash-001"),
                provider_id: pid.clone(),
                name: "Gemini 2.0 Flash".into(),
                context_window: 1_000_000,
                max_output_tokens: 8_192,
            },
            ModelInfo {
                id: ModelId::new("google/gemini-1.5-pro-002"),
                provider_id: pid.clone(),
                name: "Gemini 1.5 Pro".into(),
                context_window: 2_000_000,
                max_output_tokens: 8_192,
            },
        ]
    }
}

// ---------------------------------------------------------------------------
// LlmProvider impl
// ---------------------------------------------------------------------------

#[async_trait]
impl LlmProvider for VertexOpenAiProvider {
    fn id(&self) -> &ProviderId {
        &self.id
    }

    fn name(&self) -> &str {
        "Google Vertex AI"
    }

    async fn create_message(
        &self,
        request: ProviderRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        self.create_message_non_streaming(&request).await
    }

    async fn create_message_stream(
        &self,
        request: ProviderRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        let resp = self.do_streaming(&request).await?;
        let provider_id = self.id.clone();

        let s = stream! {
            use futures::StreamExt;

            let mut byte_stream = resp.bytes_stream();
            let mut leftover = String::new();

            let mut message_started = false;
            let mut message_id = String::from("unknown");
            let mut model_name = String::new();
            let mut tool_call_buffers: std::collections::HashMap<
                usize,
                (String, String, String),
            > = std::collections::HashMap::new();

            while let Some(chunk_result) = byte_stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        yield Err(ProviderError::StreamError {
                            provider: provider_id.clone(),
                            message: format!("Stream read error: {}", e),
                            partial_response: None,
                        });
                        return;
                    }
                };

                let text = String::from_utf8_lossy(&chunk);
                let combined = if leftover.is_empty() {
                    text.to_string()
                } else {
                    let mut s = std::mem::take(&mut leftover);
                    s.push_str(&text);
                    s
                };

                let mut lines: Vec<&str> = combined.split('\n').collect();
                if !combined.ends_with('\n') {
                    leftover = lines.pop().unwrap_or("").to_string();
                }

                for line in lines {
                    let line = line.trim_end_matches('\r').trim();

                    if line.is_empty() || line.starts_with(':') {
                        continue;
                    }

                    let data = if let Some(rest) = line.strip_prefix("data:") {
                        rest.trim()
                    } else {
                        continue;
                    };

                    if data == "[DONE]" {
                        yield Ok(StreamEvent::MessageStop);
                        return;
                    }

                    let chunk_json: Value = match serde_json::from_str(data) {
                        Ok(v) => v,
                        Err(e) => {
                            debug!("Failed to parse Vertex SSE chunk: {}: {}", e, data);
                            continue;
                        }
                    };

                    if !message_started {
                        if let Some(id) = chunk_json.get("id").and_then(|v| v.as_str()) {
                            message_id = id.to_string();
                        }
                        if let Some(m) = chunk_json.get("model").and_then(|v| v.as_str()) {
                            model_name = m.to_string();
                        }
                        yield Ok(StreamEvent::MessageStart {
                            id: message_id.clone(),
                            model: model_name.clone(),
                            usage: UsageInfo::default(),
                        });
                        yield Ok(StreamEvent::ContentBlockStart {
                            index: 0,
                            content_block: ContentBlock::Text { text: String::new() },
                        });
                        message_started = true;
                    }

                    let choices = match chunk_json.get("choices").and_then(|c| c.as_array()) {
                        Some(c) => c,
                        None => {
                            if let Some(usage_val) = chunk_json.get("usage") {
                                let usage = OpenAiProvider::parse_usage_pub(Some(usage_val));
                                yield Ok(StreamEvent::MessageDelta {
                                    stop_reason: None,
                                    usage: Some(usage),
                                });
                            }
                            continue;
                        }
                    };

                    let choice = match choices.first() {
                        Some(c) => c,
                        None => continue,
                    };

                    let delta = match choice.get("delta") {
                        Some(d) => d,
                        None => continue,
                    };

                    // Text content delta
                    if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                        if !content.is_empty() {
                            yield Ok(StreamEvent::TextDelta {
                                index: 0,
                                text: content.to_string(),
                            });
                        }
                    }

                    // Tool call deltas
                    if let Some(tool_calls) =
                        delta.get("tool_calls").and_then(|t| t.as_array())
                    {
                        for tc in tool_calls {
                            let tc_index = tc
                                .get("index")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0) as usize;
                            if let Some(tc_id) = tc.get("id").and_then(|v| v.as_str()) {
                                let name = tc
                                    .get("function")
                                    .and_then(|f| f.get("name"))
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let block_index = 1 + tc_index;
                                tool_call_buffers.insert(
                                    block_index,
                                    (tc_id.to_string(), name.clone(), String::new()),
                                );
                                yield Ok(StreamEvent::ContentBlockStart {
                                    index: block_index,
                                    content_block: ContentBlock::ToolUse {
                                        id: tc_id.to_string(),
                                        name,
                                        input: json!({}),
                                    },
                                });
                            }
                            if let Some(args_frag) = tc
                                .get("function")
                                .and_then(|f| f.get("arguments"))
                                .and_then(|v| v.as_str())
                            {
                                if !args_frag.is_empty() {
                                    let block_index = 1 + tc_index;
                                    if let Some((_, _, buf)) =
                                        tool_call_buffers.get_mut(&block_index)
                                    {
                                        buf.push_str(args_frag);
                                    }
                                    yield Ok(StreamEvent::InputJsonDelta {
                                        index: block_index,
                                        partial_json: args_frag.to_string(),
                                    });
                                }
                            }
                        }
                    }

                    // finish_reason
                    if let Some(finish_reason) =
                        choice.get("finish_reason").and_then(|v| v.as_str())
                    {
                        if !finish_reason.is_empty() && finish_reason != "null" {
                            yield Ok(StreamEvent::ContentBlockStop { index: 0 });
                            let mut tc_indices: Vec<usize> =
                                tool_call_buffers.keys().cloned().collect();
                            tc_indices.sort();
                            for idx in tc_indices {
                                yield Ok(StreamEvent::ContentBlockStop { index: idx });
                            }

                            let stop_reason =
                                OpenAiProvider::map_finish_reason_pub(finish_reason);

                            let usage_val = chunk_json.get("usage");
                            let usage =
                                usage_val.map(|u| OpenAiProvider::parse_usage_pub(Some(u)));

                            yield Ok(StreamEvent::MessageDelta {
                                stop_reason: Some(stop_reason),
                                usage,
                            });
                        }
                    }
                }
            }

            if message_started {
                yield Ok(StreamEvent::MessageStop);
            }
        };

        Ok(Box::pin(s))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        // Vertex AI does not expose a /models route on its OpenAI-compat
        // endpoint. Instead, query the Publisher Models API (v1beta1) for each
        // publisher in parallel, then combine and filter to chat-capable models.
        let token = self.bearer_token().await?;

        // Publishers available on Vertex Model Garden that support the
        // OpenAI-compatible chat completions endpoint.
        //
        // Inclusion criteria:
        //  - "openGenerationAiStudio" action  → works immediately (Google, Anthropic)
        //  - "requestAccess" action            → serverless MaaS; works once access
        //                                        is approved in Model Garden
        //
        // Publishers with only "deploy"/"multiDeployVertex" require a custom
        // endpoint deployment and are excluded.
        let publishers = [
            "google",    // Gemini (immediate)
            "anthropic", // Claude (immediate, requires Model Garden enable)
            "mistralai", // Mistral / Codestral (requestAccess MaaS)
            "meta",      // Llama MaaS variants (requestAccess)
            "ai21",      // Jamba (requestAccess)
            "writer",    // Palmyra (requestAccess)
            "nvidia",    // Nemotron (requestAccess)
        ];

        let futures: Vec<_> = publishers
            .iter()
            .map(|publisher| {
                let url = format!(
                    "https://{location}-aiplatform.googleapis.com/v1beta1/publishers/{publisher}/models",
                    location = self.config.location,
                    publisher = publisher,
                );
                let client = self.http_client.clone();
                let token = token.clone();
                let project_id = self.config.project_id.clone();
                let publisher = publisher.to_string();
                async move {
                    let resp = client
                        .get(&url)
                        .header("Authorization", format!("Bearer {}", token))
                        .header("x-goog-user-project", &project_id)
                        .send()
                        .await;
                    match resp {
                        Ok(r) if r.status().is_success() => {
                            let text = r.text().await.unwrap_or_default();
                            Some((publisher, text))
                        }
                        Ok(r) => {
                            debug!("Vertex publisher/{} returned HTTP {}", publisher, r.status());
                            None
                        }
                        Err(e) => {
                            debug!("Vertex publisher/{} request failed: {}", publisher, e);
                            None
                        }
                    }
                }
            })
            .collect();

        let results = futures::future::join_all(futures).await;

        let pid = self.id.clone();
        let mut models: Vec<ModelInfo> = Vec::new();

        for result in results.into_iter().flatten() {
            let (publisher, text) = result;
            let json: Value = match serde_json::from_str(&text) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let prefix = format!("publishers/{}/models/", publisher);
            if let Some(arr) = json.get("publisherModels").and_then(|v| v.as_array()) {
                for m in arr {
                    let Some(full_name) = m.get("name").and_then(|v| v.as_str()) else {
                        continue;
                    };
                    let Some(model_id) = full_name.strip_prefix(&prefix) else {
                        continue;
                    };

                    // Include models with either:
                    //   "openGenerationAiStudio" → works immediately
                    //   "requestAccess"          → serverless MaaS (works once approved)
                    //
                    // Exclude models that only have "deploy"/"multiDeployVertex"
                    // since those require a custom endpoint, not the openapi route.
                    let actions = m
                        .get("supportedActions")
                        .and_then(|a| a.as_object())
                        .map(|obj| obj.keys().cloned().collect::<Vec<_>>())
                        .unwrap_or_default();

                    let supports_chat = actions
                        .iter()
                        .any(|a| a == "openGenerationAiStudio" || a == "requestAccess");
                    if !supports_chat {
                        continue;
                    }

                    // Publisher-specific filtering — drop non-chat model families
                    let skip = match publisher.as_str() {
                        "google" => {
                            !model_id.starts_with("gemini")
                                || [
                                    "embedding",
                                    "-tts",
                                    "-audio",
                                    "-image",
                                    "computer-use",
                                    "live-",
                                ]
                                .iter()
                                .any(|s| model_id.contains(s))
                        }
                        "anthropic" => !model_id.starts_with("claude"),
                        "mistralai" => {
                            // Self-deploy variant and OCR are not chat models
                            model_id.contains("self-deploy") || model_id.contains("ocr")
                        }
                        "meta" => {
                            // Only the -maas suffixed models use the serverless API
                            !model_id.ends_with("-maas")
                        }
                        "nvidia" => {
                            // Only Nemotron/Llama-based language models; drop vision/cosmos
                            model_id.contains("cosmos")
                                || model_id.contains("vl")
                                || (!model_id.contains("nemotron") && !model_id.contains("llama"))
                        }
                        "ai21" => !model_id.contains("jamba"),
                        "writer" => !model_id.contains("palmyra"),
                        _ => false,
                    };
                    if skip {
                        continue;
                    }

                    // Format the model ID as "publisher/model-id" which is
                    // what the Vertex OpenAI-compat endpoint expects.
                    let api_id = format!("{}/{}", publisher, model_id);
                    let display = vertex_model_display_name(&publisher, model_id);
                    let ctx = match publisher.as_str() {
                        "google" => gemini_context_window(model_id),
                        "anthropic" => 200_000,
                        "mistralai" => {
                            if model_id.contains("codestral") {
                                256_000
                            } else {
                                128_000
                            }
                        }
                        "meta" => 128_000,
                        "nvidia" => 128_000,
                        "ai21" => 256_000,   // Jamba has 256K context
                        "writer" => 512_000, // Palmyra X4 has 512K context
                        _ => 128_000,
                    };

                    models.push(ModelInfo {
                        id: ModelId::new(&api_id),
                        provider_id: pid.clone(),
                        name: display,
                        context_window: ctx,
                        max_output_tokens: 8_192,
                    });
                }
            }
        }

        if models.is_empty() {
            return Ok(self.fallback_models());
        }

        // For key publishers where the discovery API may be blocked (e.g.
        // Anthropic's v1beta1 endpoint can return 403 even when the models
        // themselves work via the openapi route), inject known static models
        // if that publisher returned nothing from the API.
        let has_anthropic = models
            .iter()
            .any(|m| m.id.to_string().starts_with("anthropic/"));
        if !has_anthropic {
            let pid = self.id.clone();
            for (id, name) in &[
                ("anthropic/claude-opus-4-6", "Claude Opus 4.6"),
                ("anthropic/claude-sonnet-4-6", "Claude Sonnet 4.6"),
                ("anthropic/claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
                ("anthropic/claude-3-5-sonnet-v2", "Claude 3.5 Sonnet v2"),
                ("anthropic/claude-3-5-haiku", "Claude 3.5 Haiku"),
                ("anthropic/claude-3-opus", "Claude 3 Opus"),
            ] {
                models.push(ModelInfo {
                    id: ModelId::new(*id),
                    provider_id: pid.clone(),
                    name: name.to_string(),
                    context_window: 200_000,
                    max_output_tokens: 8_192,
                });
            }
        }

        // Group by publisher in a logical order; within each group sort
        // descending so newer/larger models appear at the top.
        models.sort_by(|a, b| {
            let publisher_order = |id: &str| -> u8 {
                if id.starts_with("google/") {
                    0
                } else if id.starts_with("anthropic/") {
                    1
                } else if id.starts_with("mistralai/") {
                    2
                } else if id.starts_with("meta/") {
                    3
                } else if id.starts_with("nvidia/") {
                    4
                } else if id.starts_with("ai21/") {
                    5
                } else if id.starts_with("writer/") {
                    6
                } else {
                    7
                }
            };
            let pa = publisher_order(&a.id.to_string());
            let pb = publisher_order(&b.id.to_string());
            pa.cmp(&pb)
                .then_with(|| b.id.to_string().cmp(&a.id.to_string()))
        });

        Ok(models)
    }

    async fn health_check(&self) -> Result<ProviderStatus, ProviderError> {
        // A quick token fetch validates that auth is working.
        match fetch_bearer_token(&self.config).await {
            Err(msg) => Ok(ProviderStatus::Unavailable { reason: msg }),
            Ok(token) => {
                // Make a minimal non-streaming request to verify the project/location.
                let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
                let body = json!({
                    "model": self.config.model,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": false,
                });
                match self
                    .http_client
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", token))
                    .header("Content-Type", "application/json")
                    .json(&body)
                    .send()
                    .await
                {
                    Ok(r) if r.status().is_success() => Ok(ProviderStatus::Healthy),
                    Ok(r) => {
                        let status = r.status().as_u16();
                        let body_text = r.text().await.unwrap_or_default();
                        if status == 401 || status == 403 {
                            Ok(ProviderStatus::Unavailable {
                                reason: format!(
                                    "Vertex auth failed (HTTP {}). Run `gcloud auth application-default login`.",
                                    status
                                ),
                            })
                        } else {
                            Ok(ProviderStatus::Unavailable {
                                reason: format!(
                                    "HTTP {}: {}",
                                    status,
                                    body_text.chars().take(120).collect::<String>()
                                ),
                            })
                        }
                    }
                    Err(e) => Ok(ProviderStatus::Unavailable {
                        reason: e.to_string(),
                    }),
                }
            }
        }
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            tool_calling: true,
            // Vertex supports JSON Schema structured output but not fully
            // recursive schemas — flag conservatively.
            thinking: false,
            image_input: true,
            pdf_input: false,
            audio_input: false,
            video_input: false,
            caching: false,
            structured_output: true,
            system_prompt_style: SystemPromptStyle::SystemMessage,
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
    fn base_url_format() {
        let url = vertex_openai_base_url("my-project-123", "us-central1");
        assert_eq!(
            url,
            "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project-123/locations/us-central1/endpoints/openapi"
        );
    }

    #[test]
    fn base_url_europe() {
        let url = vertex_openai_base_url("proj", "europe-west4");
        assert!(url.starts_with("https://europe-west4-aiplatform.googleapis.com"));
        assert!(url.contains("/projects/proj/"));
        assert!(url.contains("/locations/europe-west4/"));
    }

    #[test]
    fn apply_vertex_extras_sets_google_key() {
        let mut body = json!({"model": "test"});
        let opts = json!({"google": {"safety_settings": []}});
        apply_vertex_extras(&mut body, &opts);
        assert!(body.get("extra_body").is_some());
        assert!(body["extra_body"]["google"]["safety_settings"].is_array());
    }

    #[test]
    fn apply_vertex_extras_skips_when_absent() {
        let mut body = json!({"model": "test"});
        apply_vertex_extras(&mut body, &json!({}));
        assert!(body.get("extra_body").is_none());
    }
}
