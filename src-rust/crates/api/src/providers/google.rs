// providers/google.rs — GoogleProvider: implements LlmProvider for the
// Google Gemini API (generativelanguage.googleapis.com).
//
// Supports:
// - Non-streaming: POST .../generateContent?key={api_key}
// - Streaming SSE: POST .../streamGenerateContent?alt=sse&key={api_key}
// - Tool/function calling via functionDeclarations
// - System prompts via systemInstruction field
// - Thinking config for Gemini 2.5+ and 3.0+ models
// - Image/video inputs via inlineData parts
// - list_models via GET /v1beta/models

use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use mangocode_core::provider_id::{ModelId, ProviderId};
use mangocode_core::types::{ContentBlock, Message, MessageContent, Role, ToolResultContent, UsageInfo};
use futures::{Stream, StreamExt};
use serde_json::{json, Value};
use tracing::{debug, warn};

use crate::error_handling::parse_error_response as parse_http_error;
use crate::provider::{LlmProvider, ModelInfo};
use crate::provider_error::ProviderError;
use crate::provider_types::{
    ProviderCapabilities, ProviderRequest, ProviderResponse, ProviderStatus, StopReason,
    StreamEvent, SystemPrompt, SystemPromptStyle,
};

use super::request_options::merge_google_options;

// ---------------------------------------------------------------------------
// GoogleProvider
// ---------------------------------------------------------------------------

pub struct GoogleProvider {
    id: ProviderId,
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
}

impl GoogleProvider {
    pub fn new(api_key: String) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .connect_timeout(std::time::Duration::from_secs(30))
            .read_timeout(std::time::Duration::from_secs(300))
            .build()
            .expect("failed to build reqwest client");
        Self {
            id: ProviderId::new(ProviderId::GOOGLE),
            api_key,
            base_url: "https://generativelanguage.googleapis.com".to_string(),
            http_client,
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Returns true if the model supports thinking config (Gemini 2.5+ / 3.0+).
    fn supports_thinking(model: &str) -> bool {
        model.contains("2.5") || model.contains("3.0") || model.contains("3.1") || model.contains("gemini-3")
    }

    /// Build the full generateContent URL for non-streaming.
    fn generate_url(&self, model: &str) -> String {
        format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            self.base_url, model, self.api_key
        )
    }

    /// Build the full streamGenerateContent URL for streaming.
    fn stream_url(&self, model: &str) -> String {
        format!(
            "{}/v1beta/models/{}:streamGenerateContent?alt=sse&key={}",
            self.base_url, model, self.api_key
        )
    }

    fn sse_data_payload(line: &str) -> Option<&str> {
        line.strip_prefix("data: ")
            .or_else(|| line.strip_prefix("data:"))
    }

    fn tool_use_part(name: &str, input: &Value, thought_signature: Option<&str>) -> Value {
        let mut part = json!({
            "functionCall": {
                "name": name,
                "args": input
            }
        });
        if let Some(sig) = thought_signature {
            part["thoughtSignature"] = Value::String(sig.to_string());
        }
        part
    }

    fn map_finish_reason(finish_reason: &str) -> StopReason {
        match finish_reason {
            "STOP" => StopReason::EndTurn,
            "MAX_TOKENS" => StopReason::MaxTokens,
            "SAFETY" => StopReason::ContentFiltered,
            "RECITATION" => StopReason::ContentFiltered,
            "TOOL_CODE" | "FUNCTION_CALL" => StopReason::ToolUse,
            other => StopReason::Other(other.to_string()),
        }
    }

    fn normalized_stop_reason(finish_reason: &str, saw_tool_call: bool) -> StopReason {
        let mapped = Self::map_finish_reason(finish_reason);
        if matches!(mapped, StopReason::EndTurn) && saw_tool_call {
            StopReason::ToolUse
        } else {
            mapped
        }
    }

    fn tool_use_id_for_name(name: &str, occurrence: usize) -> String {
        let sanitized: String = name
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                    ch
                } else {
                    '_'
                }
            })
            .collect();
        let base = if sanitized.is_empty() { "tool" } else { sanitized.as_str() };
        if occurrence == 0 {
            format!("call_{}", base)
        } else {
            format!("call_{}_{}", base, occurrence + 1)
        }
    }

    fn tool_name_by_id(messages: &[Message]) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        for message in messages {
            let MessageContent::Blocks(blocks) = &message.content else {
                continue;
            };
            for block in blocks {
                if let ContentBlock::ToolUse { id, name, .. } = block {
                    map.insert(id.clone(), name.clone());
                }
            }
        }
        map
    }

    fn infer_tool_name_from_id(tool_use_id: &str) -> Option<String> {
        let raw = tool_use_id.strip_prefix("call_")?;
        let trimmed = if let Some((candidate, suffix)) = raw.rsplit_once('_') {
            if !candidate.is_empty() && suffix.chars().all(|ch| ch.is_ascii_digit()) {
                candidate
            } else {
                raw
            }
        } else {
            raw
        };

        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }

    /// Convert a single ContentBlock to a Gemini "part" Value.
    /// Returns None for blocks that should be dropped (e.g. Thinking).
    fn content_block_to_part(block: &ContentBlock) -> Option<Value> {
        match block {
            ContentBlock::Text { text } => Some(json!({ "text": text })),

            ContentBlock::Image { source } => {
                // Prefer base64 inline data; fall back to URL if available.
                if let (Some(data), Some(mime)) = (&source.data, &source.media_type) {
                    Some(json!({
                        "inlineData": {
                            "data": data,
                            "mimeType": mime
                        }
                    }))
                } else {
                    source.url.as_ref().map(|url| {
                        json!({
                            "fileData": {
                                "fileUri": url,
                                "mimeType": source.media_type.as_deref().unwrap_or("image/jpeg")
                            }
                        })
                    })
                }
            }

            ContentBlock::ToolUse { name, input, .. } => {
                Some(Self::tool_use_part(name, input, None))
            }

            // Thinking blocks are not supported by Gemini — drop silently.
            ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => None,

            // Document blocks: treat as file data when URL is available,
            // otherwise as inline base64.
            ContentBlock::Document { source, .. } => {
                if let (Some(data), Some(mime)) = (&source.data, &source.media_type) {
                    Some(json!({
                        "inlineData": {
                            "data": data,
                            "mimeType": mime
                        }
                    }))
                } else {
                    source.url.as_ref().map(|url| {
                        json!({
                            "fileData": {
                                "fileUri": url,
                                "mimeType": source.media_type.as_deref().unwrap_or("application/pdf")
                            }
                        })
                    })
                }
            }

            // Render UI-only / metadata blocks as text so context is not lost.
            ContentBlock::UserLocalCommandOutput { command, output } => Some(json!({
                "text": format!("$ {}\n{}", command, output)
            })),
            ContentBlock::UserCommand { name, args } => Some(json!({
                "text": format!("/{} {}", name, args)
            })),
            ContentBlock::UserMemoryInput { key, value } => Some(json!({
                "text": format!("[memory] {}: {}", key, value)
            })),
            ContentBlock::SystemAPIError { message, .. } => Some(json!({
                "text": format!("[error] {}", message)
            })),
            ContentBlock::CollapsedReadSearch { tool_name, paths, .. } => Some(json!({
                "text": format!("[{}] {}", tool_name, paths.join(", "))
            })),
            ContentBlock::TaskAssignment { id, subject, description } => Some(json!({
                "text": format!("[task:{}] {}: {}", id, subject, description)
            })),

            // ToolResult is handled specially in message conversion.
            ContentBlock::ToolResult { .. } => None,
        }
    }

    /// Convert a ToolResult block to a "functionResponse" part Value.
    fn tool_result_to_part(tool_name: &str, content: &ToolResultContent) -> Value {
        let response_content = match content {
            ToolResultContent::Text(t) => json!({ "content": t }),
            ToolResultContent::Blocks(blocks) => {
                // Concatenate all text blocks for the response payload.
                let text: String = blocks
                    .iter()
                    .filter_map(|b| {
                        if let ContentBlock::Text { text } = b {
                            Some(text.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                json!({ "content": text })
            }
        };
        json!({
            "functionResponse": {
                "name": tool_name,
                "response": response_content
            }
        })
    }

    /// Sanitize a JSON Schema object for Google's stricter requirements:
    /// - Remove unsupported `additionalProperties`
    /// - Integer enums → string enums
    /// - `required` must only list fields actually in `properties`
    /// - Non-object types must not have `properties`/`required`
    /// - Array `items` must have a `type` field
    fn sanitize_schema(schema: Value) -> Value {
        match schema {
            Value::Object(mut map) => {
                // Gemini function declaration schemas reject this JSON Schema key.
                map.remove("additionalProperties");

                // Recurse into nested schemas first.
                let schema_type = map
                    .get("type")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                // Convert integer enums to string enums.
                if let Some(Value::Array(enum_vals)) = map.get("enum") {
                    if enum_vals.iter().any(|v| v.is_number()) {
                        let string_enums: Vec<Value> = enum_vals
                            .iter()
                            .map(|v| Value::String(v.to_string()))
                            .collect();
                        map.insert("enum".to_string(), Value::Array(string_enums));
                        // Upgrade type to string when converting number enums.
                        map.insert("type".to_string(), Value::String("string".to_string()));
                    }
                }

                // For object types: sanitize properties recursively and fix required.
                if schema_type.as_deref() == Some("object") {
                    if let Some(Value::Object(props)) = map.get_mut("properties") {
                        let sanitized_props: serde_json::Map<String, Value> = props
                            .iter()
                            .map(|(k, v)| (k.clone(), Self::sanitize_schema(v.clone())))
                            .collect();
                        *props = sanitized_props;
                    }

                    // Filter required to only include keys present in properties.
                    if let Some(Value::Array(req_arr)) = map.get("required").cloned() {
                        let prop_keys: std::collections::HashSet<String> = map
                            .get("properties")
                            .and_then(|p| p.as_object())
                            .map(|o| o.keys().cloned().collect())
                            .unwrap_or_default();

                        let filtered: Vec<Value> = req_arr
                            .into_iter()
                            .filter(|v| {
                                v.as_str()
                                    .map(|s| prop_keys.contains(s))
                                    .unwrap_or(false)
                            })
                            .collect();
                        map.insert("required".to_string(), Value::Array(filtered));
                    }
                } else {
                    // Non-object types must not carry properties/required.
                    map.remove("properties");
                    map.remove("required");
                }

                // Array items: ensure a type field is present.
                if schema_type.as_deref() == Some("array") {
                    if let Some(items) = map.get_mut("items") {
                        if let Value::Object(ref mut items_map) = items {
                            if !items_map.contains_key("type") {
                                items_map
                                    .insert("type".to_string(), Value::String("string".to_string()));
                            }
                            // Recurse sanitize into items.
                            let sanitized = Self::sanitize_schema(Value::Object(items_map.clone()));
                            *items = sanitized;
                        }
                    }
                }

                Value::Object(map)
            }
            other => other,
        }
    }

    /// Build the full request body JSON for the Gemini API.
    fn build_request_body(&self, request: &ProviderRequest) -> Value {
        // ---- Convert messages ----
        // Google requires a flat list of content objects.
        // ToolResult blocks must become separate user-role messages.
        let mut contents: Vec<Value> = Vec::new();
        let tool_name_by_id = Self::tool_name_by_id(&request.messages);
        let use_gemini3_signature_fallback = request.model.contains("gemini-3");

        for msg in &request.messages {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "model",
            };

            let blocks = msg.content_blocks();
            let mut saw_first_model_tool_call = false;

            let mut regular_parts: Vec<Value> = Vec::new();
            let mut tool_result_parts: Vec<Value> = Vec::new();
            let flush_regular_parts = |contents: &mut Vec<Value>, parts: &mut Vec<Value>| {
                if !parts.is_empty() {
                    contents.push(json!({
                        "role": role,
                        "parts": std::mem::take(parts)
                    }));
                }
            };
            let flush_tool_result_parts = |contents: &mut Vec<Value>, parts: &mut Vec<Value>| {
                if !parts.is_empty() {
                    contents.push(json!({
                        "role": "user",
                        "parts": std::mem::take(parts)
                    }));
                }
            };

            let mut block_idx = 0usize;
            while block_idx < blocks.len() {
                match &blocks[block_idx] {
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } => {
                        flush_regular_parts(&mut contents, &mut regular_parts);
                        let tool_name = tool_name_by_id
                            .get(tool_use_id)
                            .cloned()
                            .or_else(|| Self::infer_tool_name_from_id(tool_use_id))
                            .unwrap_or_else(|| tool_use_id.clone());
                        tool_result_parts.push(Self::tool_result_to_part(&tool_name, content));
                        block_idx += 1;
                    }
                    ContentBlock::Thinking { signature, .. } => {
                        if let Some(ContentBlock::ToolUse { name, input, .. }) =
                            blocks.get(block_idx + 1)
                        {
                            // Gemini 3 validates functionCall thoughtSignature values.
                            flush_tool_result_parts(&mut contents, &mut tool_result_parts);
                            regular_parts.push(Self::tool_use_part(
                                name,
                                input,
                                Some(signature.as_str()),
                            ));
                            if role == "model" {
                                saw_first_model_tool_call = true;
                            }
                            block_idx += 2;
                        } else {
                            // Keep existing behavior for standalone thinking: drop.
                            block_idx += 1;
                        }
                    }
                    ContentBlock::ToolUse { name, input, .. } => {
                        flush_tool_result_parts(&mut contents, &mut tool_result_parts);
                        let fallback_signature = if role == "model"
                            && use_gemini3_signature_fallback
                            && !saw_first_model_tool_call
                        {
                            // Compatibility fallback for legacy histories that
                            // did not preserve Gemini 3 thought signatures.
                            Some("skip_thought_signature_validator")
                        } else {
                            None
                        };
                        regular_parts.push(Self::tool_use_part(name, input, fallback_signature));
                        if role == "model" {
                            saw_first_model_tool_call = true;
                        }
                        block_idx += 1;
                    }
                    other => {
                        if let Some(part) = Self::content_block_to_part(other) {
                            flush_tool_result_parts(&mut contents, &mut tool_result_parts);
                            regular_parts.push(part);
                        }
                        block_idx += 1;
                    }
                }
            }

            flush_regular_parts(&mut contents, &mut regular_parts);
            flush_tool_result_parts(&mut contents, &mut tool_result_parts);
        }

        // ---- System instruction ----
        let system_instruction: Option<Value> = request.system_prompt.as_ref().map(|sp| {
            let text = match sp {
                SystemPrompt::Text(t) => t.clone(),
                SystemPrompt::Blocks(blocks) => blocks
                    .iter()
                    .map(|b| b.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
            };
            json!({ "parts": [{ "text": text }] })
        });

        // ---- Tool declarations ----
        let tools_value: Option<Value> = if request.tools.is_empty() {
            None
        } else {
            let declarations: Vec<Value> = request
                .tools
                .iter()
                .map(|td| {
                    json!({
                        "name": td.name,
                        "description": td.description,
                        "parameters": Self::sanitize_schema(td.input_schema.clone())
                    })
                })
                .collect();
            Some(json!([{ "functionDeclarations": declarations }]))
        };

        // ---- Generation config ----
        let mut gen_config = serde_json::Map::new();
        gen_config.insert(
            "maxOutputTokens".to_string(),
            json!(request.max_tokens),
        );
        if let Some(temp) = request.temperature {
            gen_config.insert("temperature".to_string(), json!(temp));
        }
        if !request.stop_sequences.is_empty() {
            gen_config.insert(
                "stopSequences".to_string(),
                json!(request.stop_sequences),
            );
        }
        if let Some(top_p) = request.top_p {
            gen_config.insert("topP".to_string(), json!(top_p));
        }
        if let Some(top_k) = request.top_k {
            gen_config.insert("topK".to_string(), json!(top_k));
        }

        // Thinking config for supported models.
        if Self::supports_thinking(&request.model) && request.thinking.is_some() {
            let budget = request
                .thinking
                .as_ref()
                .map(|t| t.budget_tokens)
                .unwrap_or(8192);
            gen_config.insert(
                "thinkingConfig".to_string(),
                json!({
                    "includeThoughts": true,
                    "thinkingBudget": budget
                }),
            );
        }

        // ---- Assemble body ----
        let mut body = serde_json::Map::new();
        body.insert("contents".to_string(), Value::Array(contents));
        body.insert(
            "generationConfig".to_string(),
            Value::Object(gen_config),
        );
        if let Some(si) = system_instruction {
            body.insert("systemInstruction".to_string(), si);
        }
        if let Some(tools) = tools_value {
            body.insert("tools".to_string(), tools);
        }

        let mut value = Value::Object(body);
        merge_google_options(&mut value, &request.provider_options);
        value
    }

    /// Parse a Google error JSON body and return the appropriate ProviderError.
    fn parse_error_response(&self, status: u16, body: &str) -> ProviderError {
        parse_http_error(status, body, &self.id)
    }

    /// Extract content blocks and usage from a completed Gemini response body.
    fn parse_response_body(
        &self,
        body: &Value,
        model: &str,
    ) -> Result<ProviderResponse, ProviderError> {
        let candidates = body
            .get("candidates")
            .and_then(|c| c.as_array())
            .ok_or_else(|| ProviderError::Other {
                provider: self.id.clone(),
                message: "Missing 'candidates' in response".to_string(),
                status: None,
                body: Some(body.to_string()),
            })?;

        let candidate = candidates.first().ok_or_else(|| ProviderError::Other {
            provider: self.id.clone(),
            message: "Empty 'candidates' array in response".to_string(),
            status: None,
            body: Some(body.to_string()),
        })?;

        let finish_reason = candidate
            .get("finishReason")
            .and_then(|r| r.as_str())
            .unwrap_or("STOP");

        let parts = candidate
            .get("content")
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array());

        let mut content_blocks: Vec<ContentBlock> = Vec::new();
        let mut tool_name_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut saw_tool_call = false;

        if let Some(parts) = parts {
            for part in parts {
                if let Some(fc) = part.get("functionCall") {
                    saw_tool_call = true;

                    if let Some(signature) = part
                        .get("thoughtSignature")
                        .and_then(|s| s.as_str())
                    {
                        content_blocks.push(ContentBlock::Thinking {
                            thinking: String::new(),
                            signature: signature.to_string(),
                        });
                    }

                    let name = fc
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string();
                    let args = fc.get("args").cloned().unwrap_or(json!({}));
                    let occurrence = tool_name_counts
                        .entry(name.clone())
                        .and_modify(|count| *count += 1)
                        .or_insert(0);
                    let id = Self::tool_use_id_for_name(&name, *occurrence);
                    content_blocks.push(ContentBlock::ToolUse {
                        id,
                        name,
                        input: args,
                    });
                    continue;
                }

                if part
                    .get("thought")
                    .and_then(|t| t.as_bool())
                    .unwrap_or(false)
                {
                    continue;
                }

                if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                    content_blocks.push(ContentBlock::Text {
                        text: text.to_string(),
                    });
                }
            }
        }

        let stop_reason = Self::normalized_stop_reason(finish_reason, saw_tool_call);

        // Extract usage metadata.
        let usage = self.extract_usage(body);

        Ok(ProviderResponse {
            id: format!("gemini-{}", uuid_v4_simple()),
            content: content_blocks,
            stop_reason,
            usage,
            model: model.to_string(),
        })
    }

    /// Extract UsageInfo from a response body's usageMetadata field.
    fn extract_usage(&self, body: &Value) -> UsageInfo {
        let meta = body.get("usageMetadata");
        UsageInfo {
            input_tokens: meta
                .and_then(|m| m.get("promptTokenCount"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            output_tokens: meta
                .and_then(|m| m.get("candidatesTokenCount"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        }
    }

}

// ---------------------------------------------------------------------------
// LlmProvider impl
// ---------------------------------------------------------------------------

#[async_trait]
impl LlmProvider for GoogleProvider {
    fn id(&self) -> &ProviderId {
        &self.id
    }

    fn name(&self) -> &str {
        "Google"
    }

    async fn create_message(
        &self,
        request: ProviderRequest,
    ) -> Result<ProviderResponse, ProviderError> {
        let url = self.generate_url(&request.model);
        let model = request.model.clone();
        let body = self.build_request_body(&request);

        debug!("Google create_message: POST {}", url);

        let resp = self
            .http_client
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::ServerError {
                provider: self.id.clone(),
                status: None,
                message: e.to_string(),
                is_retryable: true,
            })?;

        let status = resp.status().as_u16();
        let resp_body = resp.text().await.map_err(|e| ProviderError::ServerError {
            provider: self.id.clone(),
            status: Some(status),
            message: e.to_string(),
            is_retryable: true,
        })?;

        if status >= 400 {
            return Err(self.parse_error_response(status, &resp_body));
        }

        let json_body: Value =
            serde_json::from_str(&resp_body).map_err(|e| ProviderError::Other {
                provider: self.id.clone(),
                message: format!("Failed to parse response JSON: {}", e),
                status: Some(status),
                body: Some(resp_body.clone()),
            })?;

        self.parse_response_body(&json_body, &model)
    }

    async fn create_message_stream(
        &self,
        request: ProviderRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>>,
        ProviderError,
    > {
        let url = self.stream_url(&request.model);
        let model = request.model.clone();
        let body = self.build_request_body(&request);

        debug!("Google create_message_stream: POST {}", url);

        let resp = self
            .http_client
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::ServerError {
                provider: self.id.clone(),
                status: None,
                message: e.to_string(),
                is_retryable: true,
            })?;

        let status = resp.status().as_u16();
        if status >= 400 {
            let resp_body =
                resp.text()
                    .await
                    .unwrap_or_else(|_| "<unreadable>".to_string());
            return Err(self.parse_error_response(status, &resp_body));
        }

        // Wrap the byte stream in a line-based SSE parser.
        let provider_id_for_stream = self.id.clone();
        let model_clone = model.clone();
        let byte_stream = resp.bytes_stream();

        let stream = async_stream::stream! {
            let mut byte_stream = byte_stream;
            let text_block_index: usize = 0;
            let mut tool_block_index: usize = 1000;
            let mut open_tool_calls: std::collections::HashMap<usize, (usize, String, String)> =
                std::collections::HashMap::new();
            let mut emitted_message_start = false;
            let mut emitted_text_block_start = false;
            let mut emitted_message_stop = false;
            let message_id = format!("gemini-{}", uuid_v4_simple());
            let mut line_buf = String::new();
            let mut tool_name_counts: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();
            let mut had_stream_error = false;

            // Logical inactivity watchdog: track when we last received a
            // *meaningful* event (candidate parsed, text delta, tool delta,
            // finish reason, promptFeedback).  If the message has started
            // and we go N seconds without meaningful progress, we synthesise
            // terminal events and close the stream so the UI never hangs.
            let inactivity_limit = std::time::Duration::from_secs(20);
            let chunk_read_timeout = std::time::Duration::from_secs(60);
            let mut last_meaningful_event = tokio::time::Instant::now();

            // Helper closure extracted as a macro-like block below since
            // async_stream! doesn't support closures that yield.

            loop {
                // Read the next byte chunk with a hard timeout so that we
                // don't block forever when the server keeps the connection
                // open but sends nothing at all.
                let chunk_result = match tokio::time::timeout(
                    chunk_read_timeout,
                    byte_stream.next(),
                ).await {
                    Ok(Some(result)) => result,
                    Ok(None) => break,  // stream exhausted
                    Err(_) => {
                        // Hard timeout — no bytes at all for chunk_read_timeout.
                        warn!("Google SSE: chunk read timeout ({}s)", chunk_read_timeout.as_secs());
                        had_stream_error = true;
                        break;
                    }
                };

                // Check logical inactivity watchdog *before* processing the chunk.
                // If we started a message but haven't had meaningful progress,
                // break out so the cleanup code below emits MessageStop.
                if emitted_message_start && !emitted_message_stop {
                    if last_meaningful_event.elapsed() >= inactivity_limit {
                        warn!(
                            "Google SSE: logical inactivity timeout ({}s with no meaningful event)",
                            inactivity_limit.as_secs()
                        );
                        break;
                    }
                }
                let chunk: Bytes = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        warn!("Google SSE: byte stream error: {}", e);
                        had_stream_error = true;
                        break;
                    }
                };

                // Empty chunk (keepalive) — don't update last_meaningful_event.
                if chunk.is_empty() {
                    continue;
                }

                let chunk_str = match std::str::from_utf8(&chunk) {
                    Ok(s) => s,
                    Err(_) => {
                        warn!("Google SSE: non-UTF8 chunk, skipping");
                        continue;
                    }
                };

                line_buf.push_str(chunk_str);

                // Process complete lines.
                while let Some(newline_pos) = line_buf.find('\n') {
                    let line = line_buf[..newline_pos].trim_end_matches('\r').to_string();
                    line_buf = line_buf[newline_pos + 1..].to_string();

                    if let Some(data) = Self::sse_data_payload(&line) {
                        let data = data.trim();
                        if data.is_empty() {
                            continue;
                        }

                        if data == "[DONE]" {
                            if emitted_message_start && !emitted_message_stop {
                                if emitted_text_block_start {
                                    yield Ok(StreamEvent::ContentBlockStop {
                                        index: text_block_index,
                                    });
                                }

                                let had_open_tool_calls = !open_tool_calls.is_empty();
                                let mut tool_indices: Vec<usize> =
                                    open_tool_calls
                                        .values()
                                        .map(|(idx, _, _)| *idx)
                                        .collect();
                                tool_indices.sort_unstable();
                                for idx in tool_indices {
                                    yield Ok(StreamEvent::ContentBlockStop { index: idx });
                                }
                                open_tool_calls.clear();

                                yield Ok(StreamEvent::MessageDelta {
                                    stop_reason: Some(if had_open_tool_calls {
                                        StopReason::ToolUse
                                    } else {
                                        StopReason::EndTurn
                                    }),
                                    usage: Some(UsageInfo::default()),
                                });
                                yield Ok(StreamEvent::MessageStop);
                            }
                            return;
                        }

                        // Parse the JSON payload and emit events.
                        let parsed: Value = match serde_json::from_str(data) {
                            Ok(v) => v,
                            Err(e) => {
                                warn!("Google SSE: JSON parse error: {}: {}", e, data);
                                continue;
                            }
                        };

                        // Check for stream-level error.
                        if let Some(err) = parsed.get("error") {
                            let msg = err
                                .get("message")
                                .and_then(|m| m.as_str())
                                .unwrap_or("unknown error")
                                .to_string();
                            yield Err(ProviderError::StreamError {
                                provider: provider_id_for_stream.clone(),
                                message: msg,
                                partial_response: None,
                            });
                            return;
                        }

                        // Emit MessageStart on first chunk.
                        if !emitted_message_start {
                            emitted_message_start = true;
                            let meta = parsed.get("usageMetadata");
                            let usage = UsageInfo {
                                input_tokens: meta
                                    .and_then(|m| m.get("promptTokenCount"))
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0),
                                output_tokens: meta
                                    .and_then(|m| m.get("candidatesTokenCount"))
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0),
                                cache_creation_input_tokens: 0,
                                cache_read_input_tokens: 0,
                            };
                            yield Ok(StreamEvent::MessageStart {
                                id: message_id.clone(),
                                model: model_clone.clone(),
                                usage,
                            });
                        }

                        // Handle promptFeedback-only / blocked responses.
                        // Google may return a response with promptFeedback
                        // but no candidates when the request is blocked.
                        if let Some(feedback) = parsed.get("promptFeedback") {
                            last_meaningful_event = tokio::time::Instant::now();
                            let block_reason = feedback
                                .get("blockReason")
                                .and_then(|r| r.as_str())
                                .unwrap_or("");

                            if !block_reason.is_empty() {
                                // This is a blocked/refused request. Emit
                                // a text block with the reason and close.
                                let reason_text = format!(
                                    "Request blocked by Google: {}",
                                    block_reason,
                                );
                                if !emitted_text_block_start {
                                    yield Ok(StreamEvent::ContentBlockStart {
                                        index: text_block_index,
                                        content_block: ContentBlock::Text {
                                            text: String::new(),
                                        },
                                    });
                                }
                                yield Ok(StreamEvent::TextDelta {
                                    index: text_block_index,
                                    text: reason_text,
                                });
                                yield Ok(StreamEvent::ContentBlockStop {
                                    index: text_block_index,
                                });
                                yield Ok(StreamEvent::MessageDelta {
                                    stop_reason: Some(StopReason::EndTurn),
                                    usage: Some(UsageInfo::default()),
                                });
                                yield Ok(StreamEvent::MessageStop);
                                return;
                            }
                        }

                        let candidates = parsed
                            .get("candidates")
                            .and_then(|c| c.as_array());

                        let Some(candidates) = candidates else { continue };

                        for candidate in candidates {
                            let mut saw_tool_call_in_candidate = false;
                            let mut emitted_meaningful_candidate_progress = false;
                            let parts = candidate
                                .get("content")
                                .and_then(|c| c.get("parts"))
                                .and_then(|p| p.as_array());

                            if let Some(parts) = parts {
                                for (part_idx, part) in parts.iter().enumerate() {
                                    if part
                                        .get("thought")
                                        .and_then(|t| t.as_bool())
                                        .unwrap_or(false)
                                        && part.get("functionCall").is_none()
                                    {
                                        continue;
                                    }

                                    if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                        if !emitted_text_block_start {
                                            emitted_text_block_start = true;
                                            yield Ok(StreamEvent::ContentBlockStart {
                                                index: text_block_index,
                                                content_block: ContentBlock::Text { text: String::new() },
                                            });
                                        }
                                        yield Ok(StreamEvent::TextDelta {
                                            index: text_block_index,
                                            text: text.to_string(),
                                        });
                                        emitted_meaningful_candidate_progress = true;
                                    } else if let Some(fc) = part.get("functionCall") {
                                        saw_tool_call_in_candidate = true;
                                        let name = fc
                                            .get("name")
                                            .and_then(|n| n.as_str())
                                            .unwrap_or("")
                                            .to_string();
                                        let args_str = fc
                                            .get("args")
                                            .map(|a| a.to_string())
                                            .unwrap_or_else(|| "{}".to_string());

                                        let idx = if let Some((existing_idx, _, _)) = open_tool_calls.get(&part_idx) {
                                            *existing_idx
                                        } else {
                                            let occurrence = tool_name_counts
                                                .entry(name.clone())
                                                .and_modify(|count| *count += 1)
                                                .or_insert(0);
                                            let id = Self::tool_use_id_for_name(&name, *occurrence);
                                            let idx = tool_block_index;
                                            tool_block_index += 1;
                                            open_tool_calls.insert(part_idx, (idx, id.clone(), name.clone()));
                                            yield Ok(StreamEvent::ContentBlockStart {
                                                index: idx,
                                                content_block: ContentBlock::ToolUse {
                                                    id,
                                                    name: name.clone(),
                                                    input: json!({}),
                                                },
                                            });
                                            idx
                                        };

                                        yield Ok(StreamEvent::InputJsonDelta {
                                            index: idx,
                                            partial_json: args_str,
                                        });
                                        emitted_meaningful_candidate_progress = true;

                                        if let Some(signature) = part
                                            .get("thoughtSignature")
                                            .and_then(|s| s.as_str())
                                        {
                                            yield Ok(StreamEvent::SignatureDelta {
                                                index: idx,
                                                signature: signature.to_string(),
                                            });
                                            emitted_meaningful_candidate_progress = true;
                                        }
                                    }
                                }
                            }

                            // Handle finish reason.
                            let finish_reason = candidate
                                .get("finishReason")
                                .and_then(|r| r.as_str())
                                .unwrap_or("");

                            if !finish_reason.is_empty()
                                && finish_reason != "FINISH_REASON_UNSPECIFIED"
                            {
                                emitted_meaningful_candidate_progress = true;
                                let saw_tool_call_for_stop =
                                    saw_tool_call_in_candidate || !open_tool_calls.is_empty();

                                // Close text block.
                                if emitted_text_block_start {
                                    yield Ok(StreamEvent::ContentBlockStop {
                                        index: text_block_index,
                                    });
                                }

                                // Close tool call blocks.
                                let mut tool_indices: Vec<usize> =
                                    open_tool_calls
                                        .values()
                                        .map(|(idx, _, _)| *idx)
                                        .collect();
                                tool_indices.sort_unstable();
                                for idx in tool_indices {
                                    yield Ok(StreamEvent::ContentBlockStop { index: idx });
                                }
                                open_tool_calls.clear();

                                let stop_reason = Some(Self::normalized_stop_reason(
                                    finish_reason,
                                    saw_tool_call_for_stop,
                                ));

                                let meta = parsed.get("usageMetadata");
                                let final_usage = UsageInfo {
                                    input_tokens: meta
                                        .and_then(|m| m.get("promptTokenCount"))
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0),
                                    output_tokens: meta
                                        .and_then(|m| m.get("candidatesTokenCount"))
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0),
                                    cache_creation_input_tokens: 0,
                                    cache_read_input_tokens: 0,
                                };

                                yield Ok(StreamEvent::MessageDelta {
                                    stop_reason,
                                    usage: Some(final_usage),
                                });
                                yield Ok(StreamEvent::MessageStop);
                                emitted_message_stop = true;
                            }

                            // Only refresh logical inactivity when we actually
                            // emitted meaningful output or a terminal marker.
                            if emitted_meaningful_candidate_progress {
                                last_meaningful_event = tokio::time::Instant::now();
                            }
                        }
                    }
                    // SSE comment lines (": ...") and blank lines are ignored.
                }
            }

            // ---------------------------------------------------------------
            // Stream ended (byte_stream exhausted or errored). Process any
            // remaining data in line_buf, then guarantee MessageStop is
            // emitted so the query loop never hangs.
            // ---------------------------------------------------------------

            // Process any trailing data left in line_buf (final chunk may
            // lack a trailing newline).
            if !emitted_message_stop {
                let remaining = line_buf.trim().to_string();
                if let Some(data) = Self::sse_data_payload(&remaining) {
                    let data = data.trim();
                    if !data.is_empty() && data != "[DONE]" {
                        if let Ok(parsed) = serde_json::from_str::<Value>(data) {
                            if let Some(candidates) = parsed.get("candidates").and_then(|c| c.as_array()) {
                                for candidate in candidates {
                                    let mut saw_tool_call_in_candidate = false;
                                    // Check for a finish reason in the trailing data.
                                    let finish_reason = candidate
                                        .get("finishReason")
                                        .and_then(|r| r.as_str())
                                        .unwrap_or("");
                                    if !finish_reason.is_empty()
                                        && finish_reason != "FINISH_REASON_UNSPECIFIED"
                                    {
                                        // Extract any final parts.
                                        if let Some(parts) = candidate
                                            .get("content")
                                            .and_then(|c| c.get("parts"))
                                            .and_then(|p| p.as_array())
                                        {
                                            for (part_idx, part) in parts.iter().enumerate() {
                                                if part
                                                    .get("thought")
                                                    .and_then(|t| t.as_bool())
                                                    .unwrap_or(false)
                                                    && part.get("functionCall").is_none()
                                                {
                                                    continue;
                                                }

                                                if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                                    if !emitted_text_block_start {
                                                        emitted_text_block_start = true;
                                                        yield Ok(StreamEvent::ContentBlockStart {
                                                            index: text_block_index,
                                                            content_block: ContentBlock::Text { text: String::new() },
                                                        });
                                                    }
                                                    yield Ok(StreamEvent::TextDelta {
                                                        index: text_block_index,
                                                        text: text.to_string(),
                                                    });
                                                } else if let Some(fc) = part.get("functionCall") {
                                                    saw_tool_call_in_candidate = true;
                                                    let name = fc
                                                        .get("name")
                                                        .and_then(|n| n.as_str())
                                                        .unwrap_or("")
                                                        .to_string();
                                                    let args_str = fc
                                                        .get("args")
                                                        .map(|a| a.to_string())
                                                        .unwrap_or_else(|| "{}".to_string());

                                                    let idx = if let Some((existing_idx, _, _)) = open_tool_calls.get(&part_idx) {
                                                        *existing_idx
                                                    } else {
                                                        let occurrence = tool_name_counts
                                                            .entry(name.clone())
                                                            .and_modify(|count| *count += 1)
                                                            .or_insert(0);
                                                        let id = Self::tool_use_id_for_name(&name, *occurrence);
                                                        let idx = tool_block_index;
                                                        tool_block_index += 1;
                                                        open_tool_calls.insert(part_idx, (idx, id.clone(), name.clone()));
                                                        yield Ok(StreamEvent::ContentBlockStart {
                                                            index: idx,
                                                            content_block: ContentBlock::ToolUse {
                                                                id,
                                                                name: name.clone(),
                                                                input: json!({}),
                                                            },
                                                        });
                                                        idx
                                                    };

                                                    yield Ok(StreamEvent::InputJsonDelta {
                                                        index: idx,
                                                        partial_json: args_str,
                                                    });

                                                    if let Some(signature) = part
                                                        .get("thoughtSignature")
                                                        .and_then(|s| s.as_str())
                                                    {
                                                        yield Ok(StreamEvent::SignatureDelta {
                                                            index: idx,
                                                            signature: signature.to_string(),
                                                        });
                                                    }
                                                }
                                            }
                                        }

                                        let saw_tool_call_for_stop =
                                            saw_tool_call_in_candidate || !open_tool_calls.is_empty();
                                        let stop_reason = Some(Self::normalized_stop_reason(
                                            finish_reason,
                                            saw_tool_call_for_stop,
                                        ));

                                        if emitted_text_block_start {
                                            yield Ok(StreamEvent::ContentBlockStop {
                                                index: text_block_index,
                                            });
                                        }

                                        let mut tool_indices: Vec<usize> =
                                            open_tool_calls.values().map(|(idx, _, _)| *idx).collect();
                                        tool_indices.sort_unstable();
                                        for idx in tool_indices {
                                            yield Ok(StreamEvent::ContentBlockStop { index: idx });
                                        }
                                        open_tool_calls.clear();

                                        let meta = parsed.get("usageMetadata");
                                        let final_usage = UsageInfo {
                                            input_tokens: meta
                                                .and_then(|m| m.get("promptTokenCount"))
                                                .and_then(|v| v.as_u64())
                                                .unwrap_or(0),
                                            output_tokens: meta
                                                .and_then(|m| m.get("candidatesTokenCount"))
                                                .and_then(|v| v.as_u64())
                                                .unwrap_or(0),
                                            cache_creation_input_tokens: 0,
                                            cache_read_input_tokens: 0,
                                        };

                                        yield Ok(StreamEvent::MessageDelta {
                                            stop_reason,
                                            usage: Some(final_usage),
                                        });
                                        yield Ok(StreamEvent::MessageStop);
                                        emitted_message_stop = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Final safety net: if we started a message but never emitted
            // MessageStop (connection dropped, malformed response, etc.),
            // close all open blocks and emit MessageStop so the query loop
            // can proceed instead of hanging.
            if emitted_message_start && !emitted_message_stop {
                warn!("Google SSE: stream ended without finishReason — forcing MessageStop");

                if emitted_text_block_start {
                    yield Ok(StreamEvent::ContentBlockStop {
                        index: text_block_index,
                    });
                }

                let mut tool_indices: Vec<usize> =
                    open_tool_calls.values().map(|(idx, _, _)| *idx).collect();
                tool_indices.sort_unstable();
                for idx in tool_indices {
                    yield Ok(StreamEvent::ContentBlockStop { index: idx });
                }

                // Determine stop reason: if we have open tool calls, treat
                // as tool_use so the query loop still executes them.
                let stop_reason = if !open_tool_calls.is_empty() {
                    Some(StopReason::ToolUse)
                } else {
                    Some(StopReason::EndTurn)
                };

                yield Ok(StreamEvent::MessageDelta {
                    stop_reason,
                    usage: Some(UsageInfo::default()),
                });
                yield Ok(StreamEvent::MessageStop);
            }

            // If we never even started (no data received), emit an error.
            if !emitted_message_start {
                if had_stream_error {
                    yield Err(ProviderError::StreamError {
                        provider: provider_id_for_stream.clone(),
                        message: "Stream failed before any data was received".to_string(),
                        partial_response: None,
                    });
                } else {
                    yield Err(ProviderError::StreamError {
                        provider: provider_id_for_stream.clone(),
                        message: "Stream ended without any data".to_string(),
                        partial_response: None,
                    });
                }
            }
        };

        Ok(Box::pin(stream))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderError> {
        let url = format!(
            "{}/v1beta/models?key={}",
            self.base_url, self.api_key
        );

        let resp = self
            .http_client
            .get(&url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await
            .map_err(|e| ProviderError::ServerError {
                provider: self.id.clone(),
                status: None,
                message: e.to_string(),
                is_retryable: true,
            })?;

        let status = resp.status().as_u16();
        let body_text = resp.text().await.map_err(|e| ProviderError::ServerError {
            provider: self.id.clone(),
            status: Some(status),
            message: e.to_string(),
            is_retryable: true,
        })?;

        if status >= 400 {
            return Err(self.parse_error_response(status, &body_text));
        }

        let body: Value =
            serde_json::from_str(&body_text).map_err(|e| ProviderError::Other {
                provider: self.id.clone(),
                message: format!("Failed to parse models list JSON: {}", e),
                status: Some(status),
                body: Some(body_text.clone()),
            })?;

        let models_array = body
            .get("models")
            .and_then(|m| m.as_array())
            .cloned()
            .unwrap_or_default();

        let provider_id = self.id.clone();
        let models: Vec<ModelInfo> = models_array
            .iter()
            .filter_map(|m| {
                let name = m.get("name").and_then(|n| n.as_str())?;
                // Only include Gemini models (filter out palm, embedding, etc.)
                if !name.starts_with("models/gemini-") {
                    return None;
                }
                // Strip the "models/" prefix for the model ID.
                let model_id = name.strip_prefix("models/").unwrap_or(name);
                let display = m
                    .get("displayName")
                    .and_then(|d| d.as_str())
                    .unwrap_or(model_id)
                    .to_string();
                let input_limit = m
                    .get("inputTokenLimit")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(32_768) as u32;
                let output_limit = m
                    .get("outputTokenLimit")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(8_192) as u32;

                Some(ModelInfo {
                    id: ModelId::new(model_id),
                    provider_id: provider_id.clone(),
                    name: display,
                    context_window: input_limit,
                    max_output_tokens: output_limit,
                })
            })
            .collect();

        Ok(models)
    }

    async fn health_check(&self) -> Result<ProviderStatus, ProviderError> {
        // Use list_models as a lightweight liveness check.
        match self.list_models().await {
            Ok(models) if !models.is_empty() => Ok(ProviderStatus::Healthy),
            Ok(_) => Ok(ProviderStatus::Degraded {
                reason: "No Gemini models returned".to_string(),
            }),
            Err(ProviderError::AuthFailed { message, .. }) => {
                Err(ProviderError::AuthFailed {
                    provider: self.id.clone(),
                    message,
                })
            }
            Err(e) => Ok(ProviderStatus::Unavailable {
                reason: e.to_string(),
            }),
        }
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            tool_calling: true,
            thinking: true,
            image_input: true,
            pdf_input: true,
            audio_input: false,
            video_input: true,
            caching: false,
            structured_output: true,
            system_prompt_style: SystemPromptStyle::SystemInstruction,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a simple pseudo-random hex ID without pulling in the uuid crate.
/// Uses a combination of the current time and a thread-local counter.
fn uuid_v4_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    // Simple hash mix to spread bits.
    let a = t ^ (t >> 17) ^ (t << 13);
    let b = a.wrapping_mul(0x517cc1b727220a95);
    format!("{:032x}", b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mangocode_core::types::Message;
    use serde_json::json;

    fn test_request(messages: Vec<Message>) -> ProviderRequest {
        ProviderRequest {
            model: "gemini-3-flash-preview".to_string(),
            messages,
            system_prompt: None,
            tools: vec![],
            max_tokens: 512,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: vec![],
            thinking: None,
            provider_options: json!({}),
        }
    }

    #[test]
    fn build_request_body_uses_function_names_for_tool_results() {
        let provider = GoogleProvider::new("test".to_string());
        let request = test_request(vec![
            Message::assistant_blocks(vec![ContentBlock::ToolUse {
                id: "call_search_2".to_string(),
                name: "search".to_string(),
                input: json!({"q": "cats"}),
            }]),
            Message::user_blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "call_search_2".to_string(),
                content: ToolResultContent::Text("ok".to_string()),
                is_error: Some(false),
            }]),
        ]);

        let body = provider.build_request_body(&request);
        let contents = body["contents"].as_array().expect("contents array");
        assert_eq!(contents.len(), 2);
        assert_eq!(
            contents[1]["parts"][0]["functionResponse"]["name"],
            json!("search")
        );
    }

    #[test]
    fn build_request_body_preserves_tool_result_order() {
        let provider = GoogleProvider::new("test".to_string());
        let request = test_request(vec![Message::user_blocks(vec![
            ContentBlock::Text {
                text: "before".to_string(),
            },
            ContentBlock::ToolResult {
                tool_use_id: "call_search".to_string(),
                content: ToolResultContent::Text("done".to_string()),
                is_error: Some(false),
            },
            ContentBlock::Text {
                text: "after".to_string(),
            },
        ])]);

        let body = provider.build_request_body(&request);
        let contents = body["contents"].as_array().expect("contents array");
        assert_eq!(contents.len(), 3);
        assert_eq!(contents[0]["role"], json!("user"));
        assert_eq!(contents[0]["parts"][0]["text"], json!("before"));
        assert_eq!(contents[1]["parts"][0]["functionResponse"]["name"], json!("search"));
        assert_eq!(contents[2]["parts"][0]["text"], json!("after"));
    }

    #[test]
    fn parse_response_body_assigns_unique_ids_for_duplicate_tool_names() {
        let provider = GoogleProvider::new("test".to_string());
        let response = json!({
            "candidates": [{
                "finishReason": "FUNCTION_CALL",
                "content": {
                    "parts": [
                        { "functionCall": { "name": "search", "args": { "q": "a" } } },
                        { "functionCall": { "name": "search", "args": { "q": "b" } } }
                    ]
                }
            }],
            "usageMetadata": {}
        });

        let parsed = provider
            .parse_response_body(&response, "gemini-3-flash-preview")
            .expect("parsed response");

        assert!(matches!(
            &parsed.content[0],
            ContentBlock::ToolUse { id, .. } if id == "call_search"
        ));
        assert!(matches!(
            &parsed.content[1],
            ContentBlock::ToolUse { id, .. } if id == "call_search_2"
        ));
    }

    #[test]
    fn parse_response_body_preserves_function_call_thought_signature() {
        let provider = GoogleProvider::new("test".to_string());
        let response = json!({
            "candidates": [{
                "finishReason": "FUNCTION_CALL",
                "content": {
                    "parts": [
                        {
                            "functionCall": { "name": "mkdir", "args": { "path": "x" } },
                            "thoughtSignature": "sig-a"
                        }
                    ]
                }
            }],
            "usageMetadata": {}
        });

        let parsed = provider
            .parse_response_body(&response, "gemini-3.1-pro-preview")
            .expect("parsed response");

        assert_eq!(parsed.content.len(), 2);
        assert!(matches!(
            &parsed.content[0],
            ContentBlock::Thinking { signature, .. } if signature == "sig-a"
        ));
        assert!(matches!(
            &parsed.content[1],
            ContentBlock::ToolUse { name, .. } if name == "mkdir"
        ));
    }

    #[test]
    fn build_request_body_attaches_signature_to_function_call_part() {
        let provider = GoogleProvider::new("test".to_string());
        let request = test_request(vec![Message::assistant_blocks(vec![
            ContentBlock::Thinking {
                thinking: String::new(),
                signature: "sig-a".to_string(),
            },
            ContentBlock::ToolUse {
                id: "call_mkdir".to_string(),
                name: "mkdir".to_string(),
                input: json!({ "path": "hello_world_python" }),
            },
        ])]);

        let body = provider.build_request_body(&request);
        let part = &body["contents"][0]["parts"][0];

        assert_eq!(part["functionCall"]["name"], json!("mkdir"));
        assert_eq!(part["thoughtSignature"], json!("sig-a"));
    }

    #[test]
    fn build_request_body_injects_fallback_signature_for_first_model_tool_call() {
        let provider = GoogleProvider::new("test".to_string());
        let request = test_request(vec![Message::assistant_blocks(vec![
            ContentBlock::ToolUse {
                id: "call_one".to_string(),
                name: "tool_one".to_string(),
                input: json!({ "a": 1 }),
            },
            ContentBlock::ToolUse {
                id: "call_two".to_string(),
                name: "tool_two".to_string(),
                input: json!({ "b": 2 }),
            },
        ])]);

        let body = provider.build_request_body(&request);
        let parts = body["contents"][0]["parts"].as_array().expect("parts array");

        assert_eq!(parts[0]["thoughtSignature"], json!("skip_thought_signature_validator"));
        assert!(parts[1].get("thoughtSignature").is_none());
    }

    #[test]
    fn parse_response_body_ignores_thought_text_and_prefers_tool_use() {
        let provider = GoogleProvider::new("test".to_string());
        let response = json!({
            "candidates": [{
                "finishReason": "STOP",
                "content": {
                    "parts": [
                        {
                            "text": "internal planning",
                            "thought": true,
                            "thoughtSignature": "sig-1"
                        },
                        {
                            "functionCall": {
                                "name": "mkdir",
                                "args": { "path": "hello_world_python" }
                            }
                        }
                    ]
                }
            }],
            "usageMetadata": {}
        });

        let parsed = provider
            .parse_response_body(&response, "gemini-3-flash-preview")
            .expect("parsed response");

        assert!(matches!(parsed.stop_reason, StopReason::ToolUse));
        assert_eq!(parsed.content.len(), 1);
        assert!(matches!(
            &parsed.content[0],
            ContentBlock::ToolUse { name, .. } if name == "mkdir"
        ));
    }

    #[test]
    fn sse_data_payload_accepts_both_data_prefix_forms() {
        assert_eq!(GoogleProvider::sse_data_payload("data: {\"a\":1}"), Some("{\"a\":1}"));
        assert_eq!(GoogleProvider::sse_data_payload("data:{\"a\":1}"), Some("{\"a\":1}"));
        assert_eq!(GoogleProvider::sse_data_payload("event: ping"), None);
    }

    #[test]
    fn normalized_stop_reason_promotes_stop_to_tool_use_when_tool_call_seen() {
        assert!(matches!(
            GoogleProvider::normalized_stop_reason("STOP", false),
            StopReason::EndTurn
        ));
        assert!(matches!(
            GoogleProvider::normalized_stop_reason("STOP", true),
            StopReason::ToolUse
        ));
        assert!(matches!(
            GoogleProvider::normalized_stop_reason("FUNCTION_CALL", false),
            StopReason::ToolUse
        ));
    }

    #[test]
    fn sanitize_schema_removes_additional_properties_recursively() {
        let schema = json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "value": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "inner": {
                            "type": "object",
                            "additionalProperties": false,
                            "properties": {
                                "name": { "type": "string" }
                            },
                            "required": ["name", "missing"]
                        }
                    },
                    "required": ["inner", "ghost"]
                }
            },
            "required": ["value", "absent"]
        });

        let sanitized = GoogleProvider::sanitize_schema(schema);

        assert!(sanitized.get("additionalProperties").is_none());
        assert_eq!(sanitized["required"], json!(["value"]));

        let value = &sanitized["properties"]["value"];
        assert!(value.get("additionalProperties").is_none());
        assert_eq!(value["required"], json!(["inner"]));

        let inner = &value["properties"]["inner"];
        assert!(inner.get("additionalProperties").is_none());
        assert_eq!(inner["required"], json!(["name"]));
    }

    // -----------------------------------------------------------------------
    // Integration-style tests: mocked SSE byte streams
    // -----------------------------------------------------------------------
    //
    // These tests feed raw SSE bytes directly into the stream-parsing logic
    // by building a GoogleProvider that points at a local mock HTTP server.
    // They assert that the stream always terminates and emits MessageStop.

    use std::pin::Pin;
    use std::time::Duration;
    use futures::Stream;

    /// Helper: collect all stream events from a GoogleProvider stream response.
    /// Returns the collected events or panics on timeout.
    async fn collect_stream_events(
        body: &str,
        timeout: Duration,
    ) -> Vec<Result<StreamEvent, ProviderError>> {
        use futures::StreamExt;

        // Spin up a tiny HTTP server that returns `body` as chunked SSE.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let body_owned = body.to_string();

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            use tokio::io::AsyncWriteExt;
            // Read the full request (consume headers).
            let mut buf = vec![0u8; 8192];
            let _ = tokio::io::AsyncReadExt::read(&mut socket, &mut buf).await;
            // Write HTTP response with chunked transfer encoding.
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
            );
            socket.write_all(header.as_bytes()).await.unwrap();

            // Write body in a single chunk then close.
            let chunk = format!("{:x}\r\n{}\r\n0\r\n\r\n", body_owned.len(), body_owned);
            socket.write_all(chunk.as_bytes()).await.unwrap();
            socket.flush().await.unwrap();
            // Drop socket to close connection.
        });

        let provider = GoogleProvider {
            id: ProviderId::new(ProviderId::GOOGLE),
            api_key: "test-key".to_string(),
            base_url: format!("http://{}", addr),
            http_client: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
        };

        let request = test_request(vec![Message::user("hello".to_string())]);
        let stream_result: Pin<Box<dyn Stream<Item = Result<StreamEvent, ProviderError>> + Send>> =
            provider
                .create_message_stream(request)
                .await
                .expect("stream creation should succeed");

        let events: Vec<Result<StreamEvent, ProviderError>> =
            tokio::time::timeout(timeout, stream_result.collect::<Vec<_>>())
                .await
                .expect("stream should complete within timeout");

        server.abort();
        events
    }

    fn has_message_stop(events: &[Result<StreamEvent, ProviderError>]) -> bool {
        events.iter().any(|e| matches!(e, Ok(StreamEvent::MessageStop)))
    }

    fn has_message_start(events: &[Result<StreamEvent, ProviderError>]) -> bool {
        events.iter().any(|e| matches!(e, Ok(StreamEvent::MessageStart { .. })))
    }

    /// Case A: Final content arrives, then the stream ends with no [DONE].
    /// The safety-net must emit MessageStop.
    #[tokio::test]
    async fn stream_completes_without_done_marker() {
        let body = concat!(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello!\"}]},",
            "\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":10,",
            "\"candidatesTokenCount\":5}}\n\n",
        );
        let events = collect_stream_events(body, Duration::from_secs(5)).await;

        assert!(has_message_start(&events), "should have MessageStart");
        assert!(has_message_stop(&events), "should have MessageStop");
        // Should have text delta with "Hello!"
        let has_text = events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::TextDelta { text, .. }) if text == "Hello!"
        ));
        assert!(has_text, "should have text delta");
    }

    /// Case B: Refusal / promptFeedback-only stream with no candidates.
    #[tokio::test]
    async fn stream_completes_on_prompt_feedback_block() {
        let body = concat!(
            "data: {\"promptFeedback\":{\"blockReason\":\"SAFETY\"},",
            "\"usageMetadata\":{\"promptTokenCount\":10}}\n\n",
        );
        let events = collect_stream_events(body, Duration::from_secs(5)).await;

        assert!(has_message_start(&events), "should have MessageStart");
        assert!(has_message_stop(&events), "should have MessageStop");
        // Should contain the block reason in a text delta.
        let has_block_text = events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::TextDelta { text, .. }) if text.contains("SAFETY")
        ));
        assert!(has_block_text, "should surface block reason");
    }

    /// Case C: [DONE] arrives without any finishReason in candidates.
    #[tokio::test]
    async fn stream_completes_on_done_without_finish_reason() {
        let body = concat!(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]}}],",
            "\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":2}}\n\n",
            "data: [DONE]\n\n",
        );
        let events = collect_stream_events(body, Duration::from_secs(5)).await;

        assert!(has_message_start(&events), "should have MessageStart");
        assert!(has_message_stop(&events), "should have MessageStop");
        let has_text = events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::TextDelta { text, .. }) if text == "partial"
        ));
        assert!(has_text, "should have partial text");
    }
}
