// providers/openai_compat.rs — OpenAI-Compatible generic provider adapter.
//
// A configurable OpenAI Chat Completions adapter that can target any
// provider exposing an OpenAI-compatible API.  Configure base URL, auth,
// extra headers, and per-provider behavioural quirks via the builder API.

use std::pin::Pin;

use async_stream::stream;
use async_trait::async_trait;
use futures::Stream;
use futures::StreamExt;
use mangocode_core::provider_id::{ModelId, ProviderId};
use mangocode_core::types::{ContentBlock, UsageInfo};
use serde_json::{json, Value};
use tracing::{debug, trace};

use crate::error_handling::parse_error_response;
use crate::provider::{LlmProvider, ModelInfo};
use crate::provider_error::ProviderError;
use crate::provider_types::{
    ProviderCapabilities, ProviderRequest, ProviderResponse, ProviderStatus, StreamEvent,
    SystemPromptStyle,
};

// Re-use the message transformation helpers from openai.rs.
use super::openai::OpenAiProvider;
use super::request_options::merge_openai_compatible_options;

// ---------------------------------------------------------------------------
// ProviderQuirks
// ---------------------------------------------------------------------------

/// Provider-specific behavioural quirks that alter how the generic adapter
/// builds and interprets requests/responses.
#[derive(Debug, Clone, Default)]
pub struct ProviderQuirks {
    /// Truncate tool call IDs to at most this many characters before sending.
    /// For example, Mistral requires tool IDs of at most 9 characters.
    pub tool_id_max_len: Option<usize>,

    /// If `true`, strip all non-alphanumeric characters from tool IDs.
    pub tool_id_alphanumeric_only: bool,

    /// Extra error-message substrings (or regex-like patterns) that indicate
    /// the request exceeded the model's context window.
    pub overflow_patterns: Vec<String>,

    /// Whether to send `{"stream_options": {"include_usage": true}}` when
    /// streaming.  Required by some providers to receive token counts.
    pub include_usage_in_stream: bool,

    /// Override the sampling temperature when the request does not specify one.
    pub default_temperature: Option<f64>,

    /// Some providers (e.g. older Mistral releases) reject a message sequence
    /// that goes …tool_result → user… without an intervening assistant turn.
    /// When `true`, an `{"role":"assistant","content":"Done."}` message is
    /// inserted between any `role: tool` message and a following `role: user`
    /// message.
    pub fix_tool_user_sequence: bool,

    /// Name of the JSON field in the assistant message that carries extended
    /// reasoning / thinking text.  `None` means the provider does not expose
    /// reasoning output.  Example: `Some("reasoning_content")` for DeepSeek.
    pub reasoning_field: Option<String>,

    /// Optional hard cap for `max_tokens` on providers/models with lower
    /// output ceilings than the default request budget.
    pub max_tokens_cap: Option<u32>,

    /// When `true`, inject `preserve_thinking: true` into the request body
    /// for models that support it (Qwen3.6-Plus agentic mode).
    /// Retains reasoning traces across turns, improving decision consistency
    /// and reducing redundant computation in long tool-heavy sessions.
    /// Per Alibaba docs: recommended for agentic scenarios, default false.
    pub preserve_thinking: bool,

    /// When `Some(true)`, explicitly enable parallel tool calls for providers
    /// that support it. When `Some(false)`, disable. `None` omits the param.
    /// For Qwen3.6-Plus, parallel_tool_calls is supported via OpenAI compat.
    pub parallel_tool_calls: Option<bool>,
}

// ---------------------------------------------------------------------------
// OpenAiCompatProvider
// ---------------------------------------------------------------------------

pub struct OpenAiCompatProvider {
    id: ProviderId,
    name: String,
    base_url: String,
    api_key: Option<String>,
    extra_headers: Vec<(String, String)>,
    quirks: ProviderQuirks,
    http_client: reqwest::Client,
}

impl OpenAiCompatProvider {
    /// Create a new compat provider.  `base_url` should already include any
    /// path prefix (e.g. `"https://api.groq.com/openai/v1"`).
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        let http_client = mangocode_core::vault::reqwest_client_builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            id: ProviderId::new(id),
            name: name.into(),
            base_url: base_url.into(),
            api_key: None,
            extra_headers: Vec::new(),
            quirks: ProviderQuirks::default(),
            http_client,
        }
    }

    /// Set an API key that will be sent as `Authorization: Bearer <key>`.
    pub fn with_api_key(mut self, key: String) -> Self {
        self.api_key = if key.is_empty() { None } else { Some(key) };
        self
    }

    /// Override the provider base URL.
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    /// Append a custom header sent on every request.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.push((name.into(), value.into()));
        self
    }

    /// Apply provider-specific quirks.
    pub fn with_quirks(mut self, quirks: ProviderQuirks) -> Self {
        self.quirks = quirks;
        self
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Returns `true` when the provider has no usable API key.
    fn has_no_key(&self) -> bool {
        self.api_key.is_none()
    }

    /// Returns `true` when the provider is expected to work without a key.
    ///
    /// This is based primarily on the provider ID so custom local hostnames
    /// such as `host.docker.internal` still count as valid keyless backends.
    fn allows_missing_api_key(&self) -> bool {
        matches!(
            &*self.id,
            ProviderId::OLLAMA
                | ProviderId::LM_STUDIO
                | ProviderId::LLAMA_CPP
                | "vllm"
                | "llama-server"
        ) || self.base_url.contains("localhost")
            || self.base_url.contains("127.0.0.1")
            || self.base_url.contains("::1")
    }

    /// Scrub a tool-call ID according to the configured quirks.
    fn scrub_tool_id(&self, id: &str) -> String {
        let mut s = id.to_string();
        if self.quirks.tool_id_alphanumeric_only {
            s = s.chars().filter(|c| c.is_alphanumeric()).collect();
        }
        if let Some(max_len) = self.quirks.tool_id_max_len {
            let truncated: String = s.chars().take(max_len).collect();
            s = format!("{:0<width$}", truncated, width = max_len);
        }
        s
    }

    /// Apply `scrub_tool_id` to every tool-call id/tool_call_id in a messages
    /// array that was already built by `OpenAiProvider::to_openai_messages`.
    fn apply_tool_id_quirks(&self, messages: &mut [Value]) {
        if self.quirks.tool_id_max_len.is_none() && !self.quirks.tool_id_alphanumeric_only {
            return;
        }
        for msg in messages.iter_mut() {
            // assistant message tool_calls[].id
            if let Some(tcs) = msg.get_mut("tool_calls").and_then(|v| v.as_array_mut()) {
                for tc in tcs.iter_mut() {
                    if let Some(id_val) = tc.get("id").and_then(|v| v.as_str()) {
                        let scrubbed = self.scrub_tool_id(id_val);
                        if let Some(obj) = tc.as_object_mut() {
                            obj.insert("id".to_string(), json!(scrubbed));
                        }
                    }
                }
            }
            // tool message tool_call_id
            if let Some(id_val) = msg.get("tool_call_id").and_then(|v| v.as_str()) {
                let scrubbed = self.scrub_tool_id(id_val);
                if let Some(obj) = msg.as_object_mut() {
                    obj.insert("tool_call_id".to_string(), json!(scrubbed));
                }
            }
        }
    }

    /// Insert `{"role":"assistant","content":"Done."}` between any
    /// `role: tool` message that is immediately followed by a `role: user`
    /// message.
    fn apply_fix_tool_user_sequence(messages: &mut Vec<Value>) {
        let mut i = 0;
        while i + 1 < messages.len() {
            let current_is_tool = messages[i].get("role").and_then(|v| v.as_str()) == Some("tool");
            let next_is_user = messages[i + 1].get("role").and_then(|v| v.as_str()) == Some("user");

            if current_is_tool && next_is_user {
                messages.insert(i + 1, json!({ "role": "assistant", "content": "Done." }));
                i += 2; // skip past the inserted message and the user message
            } else {
                i += 1;
            }
        }
    }

    /// Build the full messages array, applying all quirks.
    fn build_messages(&self, request: &ProviderRequest) -> Vec<Value> {
        let mut messages = OpenAiProvider::to_openai_messages_pub(
            &request.messages,
            request.system_prompt.as_ref(),
        );

        self.apply_tool_id_quirks(&mut messages);

        if self.quirks.fix_tool_user_sequence {
            Self::apply_fix_tool_user_sequence(&mut messages);
        }

        messages
    }

    /// Resolve the temperature to use: request value takes priority, then
    /// the quirk default, then nothing (let the API default apply).
    fn resolve_temperature(&self, request: &ProviderRequest) -> Option<f64> {
        request.temperature.or(self.quirks.default_temperature)
    }

    fn ollama_reasoning_effort_from_budget(budget_tokens: u32) -> &'static str {
        match budget_tokens {
            0 => "none",
            1..=1024 => "low",
            1025..=4096 => "medium",
            _ => "high",
        }
    }

    fn ollama_model_supports_reasoning(model: &str) -> bool {
        let lower = model.to_ascii_lowercase();
        lower.contains("thinking") || lower.contains("qwen3") || lower.contains("gpt-oss")
    }

    fn ollama_model_uses_qwen_thinking(model: &str) -> bool {
        model.to_ascii_lowercase().contains("qwen3")
    }

    fn extract_delta_text(delta: &Value) -> Option<String> {
        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
            return Some(content.to_string());
        }

        if let Some(message) = delta.get("message") {
            if let Some(content) = message.get("content") {
                if let Some(text) = content.as_str() {
                    return Some(text.to_string());
                }
                if let Some(parts) = content.as_array() {
                    let mut result = String::new();
                    for part in parts {
                        if let Some(text) = part.as_str() {
                            result.push_str(text);
                        } else if let Some(text) = part
                            .get("text")
                            .and_then(|t| t.as_str())
                        {
                            result.push_str(text);
                        } else if let Some(subparts) = part.get("parts").and_then(|p| p.as_array()) {
                            for subpart in subparts {
                                if let Some(text) = subpart.as_str() {
                                    result.push_str(text);
                                } else if let Some(text) = subpart
                                    .get("text")
                                    .and_then(|t| t.as_str())
                                {
                                    result.push_str(text);
                                }
                            }
                        }
                    }
                    if !result.is_empty() {
                        return Some(result);
                    }
                }
                if let Some(text) = content
                    .get("parts")
                    .and_then(|parts| parts.as_array())
                    .and_then(|parts| {
                        let mut buf = String::new();
                        for part in parts {
                            if let Some(text) = part.as_str() {
                                buf.push_str(text);
                            }
                        }
                        if buf.is_empty() {
                            None
                        } else {
                            Some(buf)
                        }
                    })
                {
                    return Some(text);
                }
            }
        }

        None
    }

    fn apply_thinking_config(&self, body: &mut Value, request: &ProviderRequest) {
        let Some(thinking) = &request.thinking else {
            return;
        };

        if self.id == ProviderId::OLLAMA && Self::ollama_model_supports_reasoning(&request.model) {
            if Self::ollama_model_uses_qwen_thinking(&request.model) {
                body["enable_thinking"] = json!(true);
                body["thinking_budget"] = json!(thinking.budget_tokens);
            } else {
                let effort = Self::ollama_reasoning_effort_from_budget(thinking.budget_tokens);
                body["reasoning_effort"] = json!(effort);
                body["reasoning"] = json!({ "effort": effort });
            }
        }
    }

    /// Attach the authorization header if an API key is configured.
    fn apply_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(key) = &self.api_key {
            builder.header("Authorization", format!("Bearer {}", key))
        } else {
            builder
        }
    }

    /// Attach all configured extra headers.
    fn apply_extra_headers(&self, mut builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        for (name, value) in &self.extra_headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        builder
    }

    fn map_http_error(&self, status: u16, body: &str) -> ProviderError {
        parse_error_response(status, body, &self.id)
    }

    // -----------------------------------------------------------------------
    // Non-streaming
    // -----------------------------------------------------------------------

    async fn create_message_non_streaming(
        &self,
        request: &ProviderRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        let dump_http = std::env::var("MANGOCODE_DUMP_OPENAI_COMPAT_HTTP")
            .ok()
            .map(|v| {
                matches!(
                    v.as_str(),
                    "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
                )
            })
            .unwrap_or(false);

        let messages = self.build_messages(request);
        let tools = OpenAiProvider::to_openai_tools_pub(&request.tools);
        let max_tokens = self
            .quirks
            .max_tokens_cap
            .map(|cap| request.max_tokens.min(cap))
            .unwrap_or(request.max_tokens);

        let mut body = json!({
            "model": request.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "stream": false,
        });

        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
        if let Some(t) = self.resolve_temperature(request) {
            body["temperature"] = json!(t);
        }
        if let Some(p) = request.top_p {
            body["top_p"] = json!(p);
        }
        if !request.stop_sequences.is_empty() {
            body["stop"] = json!(request.stop_sequences);
        }
        self.apply_thinking_config(&mut body, request);
        // Qwen-specific: preserve_thinking keeps reasoning traces across turns.
        // Only inject when the quirk is explicitly enabled (set by query layer
        // via build_provider_options based on session heuristics).
        if self.quirks.preserve_thinking {
            body["preserve_thinking"] = json!(true);
        }
        // parallel_tool_calls: let providers that support it run tools concurrently.
        if let Some(parallel) = self.quirks.parallel_tool_calls {
            if !tools.is_empty() {
                body["parallel_tool_calls"] = json!(parallel);
            }
        }
        merge_openai_compatible_options(&mut body, &request.provider_options);

        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let body_str = serde_json::to_string(&body).map_err(|e| ProviderError::Other {
            provider: self.id.clone(),
            message: format!("Failed to serialize request: {}", e),
            status: None,
            body: None,
        })?;

        if dump_http {
            const MAX: usize = 8192;
            let truncated = if body_str.len() > MAX {
                &body_str[..MAX]
            } else {
                &body_str
            };
            // Use stderr directly to avoid any logger filtering/compilation issues.
            eprintln!("[openai_compat wire] request_json={}", truncated);
            trace!(target: "mangocode_api::providers::openai_compat::wire", request_json = %truncated);
        }

        let retry_cfg = crate::error_handling::RetryConfig::default();
        let provider_name = self.name.clone();
        let resp = crate::retry::retry_request(
            &retry_cfg,
            &provider_name,
            |_attempt| {
                let mut builder = self
                    .http_client
                    .post(&url)
                    .header("Content-Type", "application/json");
                builder = self.apply_auth(builder);
                builder = self.apply_extra_headers(builder);
                let b = body_str.clone();
                async move { builder.body(b).send().await }
            },
            |msg| eprintln!("{}", msg),
        )
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

        if dump_http {
            const MAX: usize = 16384;
            let truncated = if text.len() > MAX {
                &text[..MAX]
            } else {
                &text
            };
            // Use stderr directly to avoid any logger filtering/compilation issues.
            eprintln!(
                "[openai_compat wire] status={} response_json={}",
                status, truncated
            );
            trace!(target: "mangocode_api::providers::openai_compat::wire", status = status, response_json = %truncated);
        }

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
        let messages = self.build_messages(request);
        let tools = OpenAiProvider::to_openai_tools_pub(&request.tools);
        let max_tokens = self
            .quirks
            .max_tokens_cap
            .map(|cap| request.max_tokens.min(cap))
            .unwrap_or(request.max_tokens);

        let mut body = json!({
            "model": request.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "stream": true,
        });

        if self.quirks.include_usage_in_stream {
            body["stream_options"] = json!({ "include_usage": true });
        }

        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
        if let Some(t) = self.resolve_temperature(request) {
            body["temperature"] = json!(t);
        }
        if let Some(p) = request.top_p {
            body["top_p"] = json!(p);
        }
        if !request.stop_sequences.is_empty() {
            body["stop"] = json!(request.stop_sequences);
        }
        self.apply_thinking_config(&mut body, request);
        // Qwen-specific: preserve_thinking keeps reasoning traces across turns.
        if self.quirks.preserve_thinking {
            body["preserve_thinking"] = json!(true);
        }
        // parallel_tool_calls for providers that support concurrent tool execution.
        if let Some(parallel) = self.quirks.parallel_tool_calls {
            if !tools.is_empty() {
                body["parallel_tool_calls"] = json!(parallel);
            }
        }
        merge_openai_compatible_options(&mut body, &request.provider_options);

        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let body_str = serde_json::to_string(&body).map_err(|e| ProviderError::Other {
            provider: self.id.clone(),
            message: format!("Failed to serialize request: {}", e),
            status: None,
            body: None,
        })?;
        let retry_cfg = crate::error_handling::RetryConfig::default();
        let provider_name = self.name.clone();
        let resp = crate::retry::retry_request(
            &retry_cfg,
            &provider_name,
            |_attempt| {
                let mut builder = self
                    .http_client
                    .post(&url)
                    .header("Content-Type", "application/json")
                    .header("Accept", "text/event-stream");
                builder = self.apply_auth(builder);
                builder = self.apply_extra_headers(builder);
                let b = body_str.clone();
                async move { builder.body(b).send().await }
            },
            |msg| eprintln!("{}", msg),
        )
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
// LlmProvider impl
// ---------------------------------------------------------------------------

#[async_trait]
impl LlmProvider for OpenAiCompatProvider {
    fn id(&self) -> &ProviderId {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn create_message(
        &self,
        request: ProviderRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        if self.has_no_key() {
            // Providers that have no key set are considered unconfigured.
            // We allow the call to proceed in case the provider genuinely needs
            // no auth (e.g. Ollama), but callers that gate on health_check()
            // will see Unavailable first.
        }
        self.create_message_non_streaming(&request).await
    }

    async fn create_message_stream(
        &self,
        request: ProviderRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>, ProviderError>
    {
        let resp = self.do_streaming(&request).await?;
        let provider_id = self.id.clone();
        let reasoning_field = self.quirks.reasoning_field.clone();
        let dump_sse = std::env::var("MANGOCODE_DUMP_OPENAI_COMPAT_SSE")
            .ok()
            .map(|v| {
                matches!(
                    v.as_str(),
                    "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
                )
            })
            .unwrap_or(false);

        let s = stream! {
            let mut byte_stream = resp.bytes_stream();
            let mut leftover = String::new();

            let mut message_started = false;
            let mut message_id = String::from("unknown");
            let mut model_name = String::new();
            #[derive(Debug, Clone)]
            struct ToolCallBuf {
                id: String,
                name: String,
                args: String,
                started: bool,
            }
            let mut tool_call_buffers: std::collections::HashMap<usize, ToolCallBuf> =
                std::collections::HashMap::new();

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

                    if dump_sse {
                        // Log raw provider frames to stderr (truncated) for debugging tool-call wire formats.
                        // This is intentionally opt-in because it can include sensitive content.
                        const MAX: usize = 8192;
                        let truncated = if data.len() > MAX { &data[..MAX] } else { data };
                        trace!(target: "mangocode_api::providers::openai_compat::wire", sse_data = %truncated);
                    }

                    let chunk_json: Value = match serde_json::from_str(data) {
                        Ok(v) => v,
                        Err(e) => {
                            debug!("Failed to parse SSE chunk: {}: {}", e, data);
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

                    // Reasoning / thinking extraction.
                    // Check provider-specific field first, then common aliases.
                    {
                        const COMMON_REASONING_FIELDS: &[&str] =
                            &["reasoning_content", "reasoning_text", "reasoning", "thinking"];

                        let fields_to_check: Vec<&str> = if let Some(ref f) = reasoning_field {
                            let mut v = vec![f.as_str()];
                            for common in COMMON_REASONING_FIELDS {
                                if *common != f.as_str() {
                                    v.push(common);
                                }
                            }
                            v
                        } else {
                            COMMON_REASONING_FIELDS.to_vec()
                        };

                        for field in &fields_to_check {
                            if let Some(reasoning) = delta.get(*field).and_then(|v| v.as_str()) {
                                if !reasoning.is_empty() {
                                    yield Ok(StreamEvent::ReasoningDelta {
                                        index: 0,
                                        reasoning: reasoning.to_string(),
                                    });
                                    break;
                                }
                            }
                        }
                    }

                    // Text content delta
                    if let Some(content) = Self::extract_delta_text(delta) {
                        if !content.is_empty() {
                            yield Ok(StreamEvent::TextDelta {
                                index: 0,
                                text: content,
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
                            let block_index = 1 + tc_index;

                            let buf = tool_call_buffers.entry(block_index).or_insert_with(|| ToolCallBuf {
                                id: String::new(),
                                name: String::new(),
                                args: String::new(),
                                started: false,
                            });

                            // Qwen/DashScope and other OpenAI-compatible providers may stream tool calls in fragments:
                            // id/name can arrive in separate deltas and may not be repeated on each chunk.
                            if let Some(tc_id) = tc.get("id").and_then(|v| v.as_str()) {
                                if !tc_id.is_empty() {
                                    buf.id = tc_id.to_string();
                                }
                            }

                            // Tool name may appear either at tool_calls[].function.name (OpenAI) or tool_calls[].name (some compat providers).
                            if let Some(name) = tc
                                .get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(|v| v.as_str())
                                .or_else(|| tc.get("name").and_then(|v| v.as_str()))
                            {
                                if !name.is_empty() {
                                    buf.name = name.to_string();
                                }
                            }

                            // Emit ContentBlockStart only once we have a tool name. If the provider didn't send an id,
                            // generate a stable one so tool_result can reference it.
                            if !buf.started && !buf.name.is_empty() {
                                if buf.id.is_empty() {
                                    buf.id = format!("call_{}_{}", message_id, tc_index);
                                }
                                buf.started = true;
                                yield Ok(StreamEvent::ContentBlockStart {
                                    index: block_index,
                                    content_block: ContentBlock::ToolUse {
                                        id: buf.id.clone(),
                                        name: buf.name.clone(),
                                        input: json!({}),
                                    },
                                });
                            }

                            // Argument fragment(s): usually function.arguments is a JSON string fragment; some providers may emit an object.
                            if let Some(args_frag) = tc
                                .get("function")
                                .and_then(|f| f.get("arguments"))
                                .and_then(|v| v.as_str().map(|s| s.to_string()).or_else(|| Some(v.to_string())))
                            {
                                if !args_frag.is_empty() {
                                    buf.args.push_str(&args_frag);
                                    yield Ok(StreamEvent::InputJsonDelta {
                                        index: block_index,
                                        partial_json: args_frag,
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
                            let usage = usage_val.map(|u| OpenAiProvider::parse_usage_pub(Some(u)));

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
        let url = format!("{}/models", self.base_url.trim_end_matches('/'));
        let builder = self.http_client.get(&url);
        let builder = self.apply_auth(builder);
        let builder = self.apply_extra_headers(builder);

        let resp = builder.send().await.map_err(|e| ProviderError::Other {
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
            message: format!("Failed to parse models JSON: {}", e),
            status: Some(status),
            body: Some(text),
        })?;

        let data = match json.get("data").and_then(|d| d.as_array()) {
            Some(d) => d,
            None => return Ok(vec![]),
        };

        let provider_id = self.id.clone();
        let models: Vec<ModelInfo> = data
            .iter()
            .filter_map(|m| {
                let id = m.get("id").and_then(|v| v.as_str())?;
                Some(ModelInfo {
                    id: ModelId::new(id),
                    provider_id: provider_id.clone(),
                    name: id.to_string(),
                    context_window: 128_000,
                    max_output_tokens: 16_384,
                })
            })
            .collect();

        Ok(models)
    }

    async fn health_check(&self) -> Result<ProviderStatus, ProviderError> {
        // Providers that need an API key but have none configured are
        // immediately unavailable without making a network call.
        if self.has_no_key() && !self.allows_missing_api_key() {
            return Ok(ProviderStatus::Unavailable {
                reason: "No API key configured".to_string(),
            });
        }

        let url = format!("{}/models", self.base_url.trim_end_matches('/'));
        let builder = self.http_client.get(&url);
        let builder = self.apply_auth(builder);
        let builder = self.apply_extra_headers(builder);

        match builder.send().await {
            Ok(r) if r.status().is_success() => Ok(ProviderStatus::Healthy),
            Ok(r) => Ok(ProviderStatus::Unavailable {
                reason: format!("models endpoint returned {}", r.status()),
            }),
            Err(e) => Ok(ProviderStatus::Unavailable {
                reason: e.to_string(),
            }),
        }
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            tool_calling: true,
            thinking: self.quirks.reasoning_field.is_some() || self.id == ProviderId::OLLAMA,
            // OpenAI-compatible gateways host a mix of text-only and
            // multimodal models. Provider-level capability is conservative;
            // Smart Attachments uses provider+model heuristics for routing.
            image_input: false,
            pdf_input: false,
            audio_input: false,
            video_input: false,
            caching: false,
            structured_output: true,
            system_prompt_style: SystemPromptStyle::SystemMessage,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn mistral_tool_ids_match_opencode_style() {
        let provider = OpenAiCompatProvider::new("mistral", "Mistral", "https://example.com")
            .with_quirks(ProviderQuirks {
                tool_id_max_len: Some(9),
                tool_id_alphanumeric_only: true,
                ..Default::default()
            });

        assert_eq!(provider.scrub_tool_id("call-123456789abc"), "call12345");
        assert_eq!(provider.scrub_tool_id("x"), "x00000000");
    }

    #[test]
    fn fix_tool_user_sequence_inserts_done_between_tool_and_user() {
        let mut messages = vec![
            json!({"role": "tool", "tool_call_id": "call_1", "content": "ok"}),
            json!({"role": "user", "content": "continue"}),
        ];

        OpenAiCompatProvider::apply_fix_tool_user_sequence(&mut messages);

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["role"], json!("assistant"));
        assert_eq!(messages[1]["content"], json!("Done."));
    }

    #[test]
    fn local_provider_ids_allow_missing_api_key_even_with_custom_hostnames() {
        let provider = OpenAiCompatProvider::new(
            ProviderId::OLLAMA,
            "Ollama",
            "http://host.docker.internal:11434/v1",
        );

        assert!(provider.has_no_key());
        assert!(provider.allows_missing_api_key());
    }

    #[test]
    fn ollama_reasoning_effort_tracks_thinking_budget() {
        assert_eq!(
            OpenAiCompatProvider::ollama_reasoning_effort_from_budget(0),
            "none"
        );
        assert_eq!(
            OpenAiCompatProvider::ollama_reasoning_effort_from_budget(128),
            "low"
        );
        assert_eq!(
            OpenAiCompatProvider::ollama_reasoning_effort_from_budget(4096),
            "medium"
        );
        assert_eq!(
            OpenAiCompatProvider::ollama_reasoning_effort_from_budget(8192),
            "high"
        );
    }

    #[test]
    fn ollama_reasoning_controls_only_target_thinking_models() {
        assert!(OpenAiCompatProvider::ollama_model_supports_reasoning(
            "SimonPu/qwen3:30B-Thinking-2507-Q4_K_XL"
        ));
        assert!(OpenAiCompatProvider::ollama_model_supports_reasoning(
            "batiai/qwen3.5-9b"
        ));
        assert!(OpenAiCompatProvider::ollama_model_supports_reasoning(
            "gpt-oss:20b"
        ));
        assert!(!OpenAiCompatProvider::ollama_model_supports_reasoning(
            "llama3.2"
        ));
        assert!(!OpenAiCompatProvider::ollama_model_supports_reasoning(
            "mistral"
        ));
    }

    #[test]
    fn ollama_reports_thinking_capability() {
        let provider =
            OpenAiCompatProvider::new(ProviderId::OLLAMA, "Ollama", "http://localhost:11434/v1");

        assert!(provider.capabilities().thinking);
    }

    #[test]
    fn unknown_remote_provider_without_key_requires_auth() {
        let provider = OpenAiCompatProvider::new(
            "custom-compat",
            "Custom Compat",
            "https://api.example.com/v1",
        );

        assert!(provider.has_no_key());
        assert!(!provider.allows_missing_api_key());
    }
}
