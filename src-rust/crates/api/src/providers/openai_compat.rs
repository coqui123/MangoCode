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
use super::openai_compat_providers::{
    discover_installed_ollama_models, ollama_native_base_from_env,
};
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

    /// When `true`, the provider may emit reasoning text wrapped in inline
    /// `<think>...</think>` tags inside `delta.content`/`message.content`
    /// rather than in a structured `reasoning_content` field. Common for
    /// local Ollama / Qwen / DeepSeek-R1 distills served via the OpenAI
    /// chat-completions adapter. When set, the streaming parser strips the
    /// tags from visible text and re-emits the inner text as
    /// `ReasoningDelta` events.
    pub inline_think_tags: bool,
}

fn env_flag_truthy(v: &str) -> bool {
    matches!(v, "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
}

/// Strip `<think>...</think>` blocks from a complete (non-streaming) text
/// response. Returns `(visible_text, reasoning_text)`. Used to handle Ollama /
/// Qwen-style providers that embed reasoning inline in `message.content`.
pub(crate) fn strip_think_tags_complete(text: &str) -> (String, String) {
    let mut splitter = ThinkTagSplitter::new();
    let mut split = splitter.push(text);
    let tail = splitter.flush();
    split.visible.push_str(&tail.visible);
    split.reasoning.push_str(&tail.reasoning);
    (split.visible, split.reasoning)
}

// ---------------------------------------------------------------------------
// Inline `<think>` tag extraction
// ---------------------------------------------------------------------------

/// Streaming-safe parser that extracts `<think>...</think>` reasoning blocks
/// from a sequence of text fragments. Returns visible text and reasoning text
/// separately so the caller can emit them as `TextDelta` and `ReasoningDelta`.
///
/// Handles tags split arbitrarily across chunks (e.g. `<thi` + `nk>`),
/// nested-thinking is treated as flat (re-entry on `<think>` while already
/// inside is folded back into reasoning), and partial tag prefixes at the end
/// of a chunk are buffered until the next chunk arrives.
#[derive(Debug, Default)]
pub(crate) struct ThinkTagSplitter {
    /// `true` while inside a `<think>...</think>` block.
    inside: bool,
    /// Pending bytes that might be the start of a tag — held until we either
    /// see the rest of the tag or confirm it's literal content.
    pending: String,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct ThinkSplit {
    pub visible: String,
    pub reasoning: String,
}

impl ThinkTagSplitter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a chunk of text and produce visible+reasoning slices for it.
    pub fn push(&mut self, fragment: &str) -> ThinkSplit {
        let mut buf = std::mem::take(&mut self.pending);
        buf.push_str(fragment);

        let mut out = ThinkSplit::default();
        let mut i = 0;
        let bytes = buf.as_bytes();

        while i < bytes.len() {
            let rest = &buf[i..];
            if !self.inside {
                // Look for the next '<' that might start a tag.
                if let Some(lt_off) = rest.find('<') {
                    out.visible.push_str(&rest[..lt_off]);
                    let tag_start = i + lt_off;
                    let tag_rest = &buf[tag_start..];
                    if let Some(rel_end) = tag_rest.find('>') {
                        let full_tag = &tag_rest[..=rel_end];
                        let trimmed = full_tag
                            .trim_start_matches('<')
                            .trim_end_matches('>')
                            .trim();
                        let lower = trimmed.to_ascii_lowercase();
                        if lower == "think" || lower.starts_with("think ") {
                            self.inside = true;
                            i = tag_start + rel_end + 1;
                            continue;
                        } else {
                            // Not a think open tag — pass through verbatim.
                            out.visible.push_str(full_tag);
                            i = tag_start + rel_end + 1;
                            continue;
                        }
                    } else {
                        // Tag may be split across chunks: only buffer if the
                        // partial could plausibly become a `<think...>` open.
                        if Self::could_be_think_open(tag_rest) {
                            self.pending = tag_rest.to_string();
                            return out;
                        } else {
                            out.visible.push_str(tag_rest);
                            return out;
                        }
                    }
                } else {
                    out.visible.push_str(rest);
                    return out;
                }
            } else {
                // Inside a thinking block — consume up to `</think>`.
                if let Some(close_off) = rest.to_ascii_lowercase().find("</think>") {
                    out.reasoning.push_str(&rest[..close_off]);
                    self.inside = false;
                    i += close_off + "</think>".len();
                    continue;
                } else {
                    // No full `</think>` in `rest`. Hold back the longest
                    // suffix of `rest` that is a prefix of `</think>` so that
                    // when more data arrives we can still match the closing
                    // tag across the boundary. Anything before that suffix is
                    // safe to emit as reasoning text.
                    let hold_len = Self::longest_close_prefix_suffix(rest);
                    let split_at = rest.len() - hold_len;
                    let (emit, tail) = rest.split_at(split_at);
                    out.reasoning.push_str(emit);
                    if !tail.is_empty() {
                        self.pending = tail.to_string();
                    }
                    return out;
                }
            }
        }

        out
    }

    /// Flush any pending bytes as visible/reasoning content (called on stream end).
    pub fn flush(&mut self) -> ThinkSplit {
        let mut out = ThinkSplit::default();
        let pending = std::mem::take(&mut self.pending);
        if !pending.is_empty() {
            if self.inside {
                out.reasoning.push_str(&pending);
            } else {
                out.visible.push_str(&pending);
            }
        }
        out
    }

    /// Could `s` be the prefix of an opening `<think...>` tag?
    fn could_be_think_open(s: &str) -> bool {
        // Accept `<`, `<t`, `<th`, ... `<think`, `<think `.
        let lower = s.to_ascii_lowercase();
        let tgt = "<think";
        if lower.len() <= tgt.len() {
            tgt.starts_with(&lower)
        } else {
            // `<think` followed by space/attr characters before `>`.
            lower.starts_with(tgt)
                && !lower
                    .chars()
                    .nth(tgt.len())
                    .map(|c| c.is_alphanumeric())
                    .unwrap_or(false)
        }
    }

    /// Could `s` be the prefix of a closing `</think>` tag?
    fn could_be_think_close(s: &str) -> bool {
        let lower = s.to_ascii_lowercase();
        let tgt = "</think>";
        if lower.len() <= tgt.len() {
            tgt.starts_with(&lower)
        } else {
            false
        }
    }

    /// Length of the longest suffix of `s` that is a prefix of `</think>`.
    /// Used to know how many trailing bytes to hold back when we don't yet
    /// have a full closing tag.
    fn longest_close_prefix_suffix(s: &str) -> usize {
        let close = "</think>";
        let s_lower = s.to_ascii_lowercase();
        let max = std::cmp::min(s.len(), close.len() - 1);
        for n in (1..=max).rev() {
            // n is the candidate suffix length in bytes; ensure it lands on a
            // char boundary in the lowercased string and the original.
            if !s.is_char_boundary(s.len() - n) {
                continue;
            }
            let suf = &s_lower[s_lower.len() - n..];
            if Self::could_be_think_close(suf) {
                return n;
            }
        }
        0
    }
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

    /// Returns `true` when an Ollama model is known to handle native
    /// OpenAI-style `tools` arrays correctly. Models outside this list
    /// frequently stall, hallucinate XML/plain-text "tool calls" inside
    /// `<think>` blocks, or simply return empty content. We default to
    /// gating tools off for unknown models so the TUI gets a usable reply
    /// instead of an indefinite "Calling model..." spinner.
    ///
    /// The list intentionally covers only models that Ollama itself
    /// documents as tool-capable. Set
    /// `MANGOCODE_OLLAMA_FORCE_TOOLS=1` to bypass the gate.
    fn ollama_model_supports_tools(model: &str) -> bool {
        let lower = model.to_ascii_lowercase();
        const KNOWN_TOOL_FAMILIES: &[&str] = &[
            "llama3.1",
            "llama3.2",
            "llama3.3",
            "llama4",
            "qwen2.5",
            "qwen3",
            "mistral-nemo",
            "mistral-large",
            "mixtral",
            "command-r",
            "firefunction",
            "hermes3",
            "granite",
            "smollm2",
            "gpt-oss",
            "gemma4",
        ];
        KNOWN_TOOL_FAMILIES
            .iter()
            .any(|family| lower.contains(family))
    }

    /// Apply Ollama-specific tool-gating rules. Returns the (possibly empty)
    /// tools array that should actually be sent to the provider, plus a flag
    /// indicating whether tools were dropped.
    fn gated_tools_for_request(&self, request: &ProviderRequest, tools: Vec<Value>) -> Vec<Value> {
        if tools.is_empty() {
            return tools;
        }
        if self.id != ProviderId::OLLAMA {
            return tools;
        }
        let force = std::env::var("MANGOCODE_OLLAMA_FORCE_TOOLS")
            .ok()
            .map(|v| env_flag_truthy(&v))
            .unwrap_or(false);
        if force {
            return tools;
        }
        if Self::ollama_model_supports_tools(&request.model) {
            return tools;
        }
        // Drop tools rather than send a request the model is likely to stall on.
        // The query layer can still observe the warning via stderr.
        eprintln!(
            "[mangocode] warn: Ollama model '{}' is not in the known tool-capable list; \
             dropping tools from request to avoid stalls. Override with \
             MANGOCODE_OLLAMA_FORCE_TOOLS=1 if your custom Modelfile supports tools.",
            request.model
        );
        Vec::new()
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
                        } else if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                            result.push_str(text);
                        } else if let Some(subparts) = part.get("parts").and_then(|p| p.as_array())
                        {
                            for subpart in subparts {
                                if let Some(text) = subpart.as_str() {
                                    result.push_str(text);
                                } else if let Some(text) =
                                    subpart.get("text").and_then(|t| t.as_str())
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
        let tools = self.gated_tools_for_request(request, tools);
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

        let mut response = OpenAiProvider::parse_non_streaming_response_pub(&json, &self.id)?;

        // Strip inline `<think>...</think>` tags from non-streaming text.
        // We do not currently expose a separate reasoning field on
        // ProviderResponse; dropping the wrapper is enough to keep the visible
        // assistant text clean. The streaming path emits the inner text as
        // ReasoningDelta events for the TUI.
        let inline_think_tags = self.quirks.inline_think_tags || self.id == ProviderId::OLLAMA;
        if inline_think_tags {
            for block in response.content.iter_mut() {
                if let ContentBlock::Text { text } = block {
                    if text.contains("<think") {
                        let (visible, _reasoning) = strip_think_tags_complete(text);
                        *text = visible.trim_start_matches('\n').to_string();
                    }
                }
            }
            response
                .content
                .retain(|b| !matches!(b, ContentBlock::Text { text } if text.is_empty()));
        }

        Ok(response)
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
        let tools = self.gated_tools_for_request(request, tools);
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
        let request_model = request.model.clone();
        let reasoning_field = self.quirks.reasoning_field.clone();
        let inline_think_tags = self.quirks.inline_think_tags || self.id == ProviderId::OLLAMA;
        let dump_sse = std::env::var("MANGOCODE_DUMP_OPENAI_COMPAT_SSE")
            .ok()
            .map(|v| env_flag_truthy(&v))
            .unwrap_or(false);
        let debug_ollama = std::env::var("MANGOCODE_DEBUG_OLLAMA_STREAM")
            .ok()
            .map(|v| env_flag_truthy(&v))
            .unwrap_or(false);
        let idle_timeout_ms: u64 = std::env::var("MANGOCODE_OLLAMA_IDLE_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(120_000);

        let s = stream! {
            let mut byte_stream = resp.bytes_stream();
            let mut leftover = String::new();

            let mut message_id = String::from("unknown");
            let mut think_splitter = ThinkTagSplitter::new();
            #[derive(Debug, Clone)]
            struct ToolCallBuf {
                id: String,
                name: String,
                args: String,
                started: bool,
            }
            let mut tool_call_buffers: std::collections::HashMap<usize, ToolCallBuf> =
                std::collections::HashMap::new();

            // Emit an early MessageStart so the UI can show "model is alive"
            // even when the provider buffers all reasoning in `<think>` tags
            // before any visible content. We use placeholder id/model values;
            // the TUI shows progress immediately and does not need to wait
            // for the first JSON chunk to arrive.
            yield Ok(StreamEvent::MessageStart {
                id: message_id.clone(),
                model: request_model.clone(),
                usage: UsageInfo::default(),
            });
            // Open the default text content block up front so subsequent
            // TextDelta / ReasoningDelta events have a parent.
            yield Ok(StreamEvent::ContentBlockStart {
                index: 0,
                content_block: ContentBlock::Text { text: String::new() },
            });

            loop {
                let next = tokio::time::timeout(
                    std::time::Duration::from_millis(idle_timeout_ms),
                    byte_stream.next(),
                )
                .await;

                let chunk_result = match next {
                    Ok(Some(c)) => c,
                    Ok(None) => break,
                    Err(_) => {
                        yield Err(ProviderError::StreamError {
                            provider: provider_id.clone(),
                            message: format!(
                                "Stream idle for {}ms with no data — provider may be stalled. \
                                 If using Ollama with a thinking model, try a larger num_ctx \
                                 or set MANGOCODE_OLLAMA_IDLE_TIMEOUT_MS.",
                                idle_timeout_ms
                            ),
                            partial_response: None,
                        });
                        return;
                    }
                };

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
                        if inline_think_tags {
                            let tail = think_splitter.flush();
                            if !tail.reasoning.is_empty() {
                                yield Ok(StreamEvent::ReasoningDelta {
                                    index: 0,
                                    reasoning: tail.reasoning,
                                });
                            }
                            if !tail.visible.is_empty() {
                                yield Ok(StreamEvent::TextDelta {
                                    index: 0,
                                    text: tail.visible,
                                });
                            }
                        }
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
                    if debug_ollama {
                        const MAX: usize = 4096;
                        let truncated = if data.len() > MAX { &data[..MAX] } else { data };
                        eprintln!("[ollama stream] raw_chunk={}", truncated);
                    }

                    let chunk_json: Value = match serde_json::from_str(data) {
                        Ok(v) => v,
                        Err(e) => {
                            debug!("Failed to parse SSE chunk: {}: {}", e, data);
                            continue;
                        }
                    };

                    // First parsed chunk: capture provider-supplied id so any
                    // synthetic tool-call ids reference it. We do not re-emit
                    // MessageStart because we already emitted one before the
                    // first chunk.
                    if let Some(id) = chunk_json.get("id").and_then(|v| v.as_str()) {
                        if id != message_id {
                            message_id = id.to_string();
                        }
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
                            if inline_think_tags {
                                let split = think_splitter.push(&content);
                                if !split.reasoning.is_empty() {
                                    if debug_ollama {
                                        eprintln!(
                                            "[ollama stream] classified=reasoning bytes={}",
                                            split.reasoning.len()
                                        );
                                    }
                                    yield Ok(StreamEvent::ReasoningDelta {
                                        index: 0,
                                        reasoning: split.reasoning,
                                    });
                                }
                                if !split.visible.is_empty() {
                                    if debug_ollama {
                                        eprintln!(
                                            "[ollama stream] classified=visible bytes={}",
                                            split.visible.len()
                                        );
                                    }
                                    yield Ok(StreamEvent::TextDelta {
                                        index: 0,
                                        text: split.visible,
                                    });
                                }
                            } else {
                                yield Ok(StreamEvent::TextDelta {
                                    index: 0,
                                    text: content,
                                });
                            }
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

            if inline_think_tags {
                let tail = think_splitter.flush();
                if !tail.reasoning.is_empty() {
                    yield Ok(StreamEvent::ReasoningDelta {
                        index: 0,
                        reasoning: tail.reasoning,
                    });
                }
                if !tail.visible.is_empty() {
                    yield Ok(StreamEvent::TextDelta {
                        index: 0,
                        text: tail.visible,
                    });
                }
            }
            // We always emit MessageStart up front, so always close.
            yield Ok(StreamEvent::MessageStop);
        };

        Ok(Box::pin(s))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        if self.id == ProviderId::OLLAMA {
            let base = ollama_native_base_from_env();
            let installed =
                discover_installed_ollama_models(&base, std::time::Duration::from_secs(3))
                    .await
                    .map_err(|e| ProviderError::Other {
                        provider: self.id.clone(),
                        message: format!("Failed to discover local Ollama models: {}", e),
                        status: None,
                        body: None,
                    })?;
            let provider_id = self.id.clone();
            return Ok(installed
                .into_iter()
                .map(|m| ModelInfo {
                    id: ModelId::new(m.name.clone()),
                    provider_id: provider_id.clone(),
                    name: m.name.clone(),
                    context_window: 128_000,
                    max_output_tokens: 16_384,
                })
                .collect());
        }

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

    // -----------------------------------------------------------------------
    // ThinkTagSplitter
    // -----------------------------------------------------------------------

    #[test]
    fn think_splitter_extracts_complete_block_in_one_chunk() {
        let mut s = ThinkTagSplitter::new();
        let r = s.push("<think>weighing options</think>here is the answer");
        assert_eq!(r.reasoning, "weighing options");
        assert_eq!(r.visible, "here is the answer");
        let tail = s.flush();
        assert!(tail.reasoning.is_empty());
        assert!(tail.visible.is_empty());
    }

    #[test]
    fn think_splitter_handles_open_tag_split_across_chunks() {
        let mut s = ThinkTagSplitter::new();
        let a = s.push("Hi <thi");
        assert_eq!(a.visible, "Hi ");
        assert!(a.reasoning.is_empty());
        let b = s.push("nk>secret</think>visible");
        assert_eq!(b.reasoning, "secret");
        assert_eq!(b.visible, "visible");
    }

    #[test]
    fn think_splitter_handles_close_tag_split_across_chunks() {
        let mut s = ThinkTagSplitter::new();
        let a = s.push("<think>thinking part</thi");
        assert_eq!(a.reasoning, "thinking part");
        assert!(a.visible.is_empty());
        let b = s.push("nk>final answer");
        assert_eq!(b.visible, "final answer");
        assert!(b.reasoning.is_empty());
    }

    #[test]
    fn think_splitter_close_prefix_helper_matches_split_boundary() {
        assert!(ThinkTagSplitter::could_be_think_close("</thi"));
        assert!(ThinkTagSplitter::could_be_think_close("</THI"));
        assert!(!ThinkTagSplitter::could_be_think_close("</thought"));

        let mut s = ThinkTagSplitter::new();
        let a = s.push("<think>reasoning</THI");
        assert_eq!(a.reasoning, "reasoning");
        assert!(a.visible.is_empty());

        let b = s.push("NK>visible");
        assert_eq!(b.visible, "visible");
        assert!(b.reasoning.is_empty());
    }

    #[test]
    fn think_splitter_emits_visible_after_close_tag() {
        let mut s = ThinkTagSplitter::new();
        let r = s.push("<think>r</think>visible-only");
        assert_eq!(r.visible, "visible-only");
    }

    #[test]
    fn think_splitter_passes_through_text_with_no_tags() {
        let mut s = ThinkTagSplitter::new();
        let r = s.push("plain assistant text");
        assert_eq!(r.visible, "plain assistant text");
        assert!(r.reasoning.is_empty());
    }

    #[test]
    fn think_splitter_does_not_swallow_unrelated_lt() {
        let mut s = ThinkTagSplitter::new();
        let r = s.push("if x < y then\n");
        assert_eq!(r.visible, "if x < y then\n");
        assert!(r.reasoning.is_empty());
    }

    #[test]
    fn think_splitter_keeps_non_think_xml_tags() {
        let mut s = ThinkTagSplitter::new();
        let r = s.push("<b>bold</b>");
        // Non-think tags are preserved as visible text.
        assert!(r.visible.contains("<b>"));
        assert!(r.visible.contains("</b>"));
        assert_eq!(r.reasoning, "");
    }

    #[test]
    fn think_splitter_byte_by_byte_open_split() {
        let mut s = ThinkTagSplitter::new();
        let mut visible = String::new();
        let mut reasoning = String::new();
        for ch in "<think>r</think>v".chars() {
            let r = s.push(&ch.to_string());
            visible.push_str(&r.visible);
            reasoning.push_str(&r.reasoning);
        }
        let tail = s.flush();
        visible.push_str(&tail.visible);
        reasoning.push_str(&tail.reasoning);
        assert_eq!(reasoning, "r");
        assert_eq!(visible, "v");
    }

    #[test]
    fn think_splitter_flushes_unterminated_block_as_reasoning() {
        let mut s = ThinkTagSplitter::new();
        // Force the splitter into a pending state with a partial close tag,
        // then flush — the held-back bytes should surface as reasoning.
        let r1 = s.push("<think>more </thi");
        assert_eq!(r1.reasoning, "more ");
        let tail = s.flush();
        assert_eq!(tail.reasoning, "</thi");
        assert!(tail.visible.is_empty());
    }

    #[test]
    fn strip_think_tags_complete_strips_inline_block() {
        let (visible, reasoning) =
            strip_think_tags_complete("<think>scratch</think>\n\nfinal answer");
        assert_eq!(reasoning, "scratch");
        assert!(visible.contains("final answer"));
    }

    // -----------------------------------------------------------------------
    // Ollama tool gating
    // -----------------------------------------------------------------------

    #[test]
    fn ollama_tool_gate_recognizes_known_models() {
        assert!(OpenAiCompatProvider::ollama_model_supports_tools(
            "llama3.2"
        ));
        assert!(OpenAiCompatProvider::ollama_model_supports_tools(
            "qwen2.5:14b"
        ));
        assert!(OpenAiCompatProvider::ollama_model_supports_tools(
            "SimonPu/qwen3:30B-Thinking-2507-Q4_K_XL"
        ));
        assert!(OpenAiCompatProvider::ollama_model_supports_tools(
            "gpt-oss:20b"
        ));
    }

    #[test]
    fn ollama_tool_gate_rejects_unknown_models() {
        assert!(!OpenAiCompatProvider::ollama_model_supports_tools("phi3"));
        assert!(!OpenAiCompatProvider::ollama_model_supports_tools(
            "tinyllama"
        ));
    }

    // Gemma 4 documents native function-calling, so the tool gate should let
    // tools through for every published tag.
    #[test]
    fn ollama_tool_gate_accepts_gemma4_tags() {
        for tag in [
            "gemma4",
            "gemma4:latest",
            "gemma4:e2b",
            "gemma4:e4b",
            "gemma4:26b",
            "gemma4:31b",
            "gemma4:31b-cloud",
        ] {
            assert!(
                OpenAiCompatProvider::ollama_model_supports_tools(tag),
                "expected {tag} to be tool-capable"
            );
        }
    }

    // Gemma 4 emits its chain-of-thought via OpenAI-Harmony channel tokens
    // (`<|channel>thought ... <channel|>`), not the Qwen-style
    // `<think>...</think>` wrapper that the streaming parser handles. Until
    // a channel-token parser exists we deliberately leave gemma4 off the
    // reasoning path so we don't surface raw markup as user-visible text.
    #[test]
    fn ollama_reasoning_does_not_assume_gemma4_uses_qwen_think_tags() {
        assert!(!OpenAiCompatProvider::ollama_model_supports_reasoning(
            "gemma4"
        ));
        assert!(!OpenAiCompatProvider::ollama_model_supports_reasoning(
            "gemma4:e4b"
        ));
        assert!(!OpenAiCompatProvider::ollama_model_uses_qwen_thinking(
            "gemma4:31b"
        ));
    }

    #[test]
    fn env_flag_truthy_recognises_common_truthy_strings() {
        for v in ["1", "true", "TRUE", "on", "yes"] {
            assert!(env_flag_truthy(v), "{} should be truthy", v);
        }
        for v in ["0", "false", "no", "off", "", "maybe"] {
            assert!(!env_flag_truthy(v), "{} should be falsy", v);
        }
    }
}
