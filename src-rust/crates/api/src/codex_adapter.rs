//! Codex schema adapter — translates between Anthropic Messages API and OpenAI API formats.
//!
//! When using OpenAI Codex provider, requests are translated from Anthropic's
//! CreateMessageRequest format to OpenAI's ChatCompletion API format, and responses
//! are translated back to Anthropic's CreateMessageResponse format.

use super::types::{CreateMessageRequest, CreateMessageResponse, SystemPrompt};
use mangocode_core::codex_oauth::CODEX_API_ENDPOINT;
use mangocode_core::types::UsageInfo;
use serde_json::{json, Value};

/// OpenAI Codex API endpoint for responses (ChatGPT plan / Codex OAuth).
pub const CODEX_RESPONSES_ENDPOINT: &str = CODEX_API_ENDPOINT;

/// Strip fields that are specific to SDKs / client-side orchestration and may be
/// rejected by ChatGPT/Codex backends.
///
/// Mirrors the concept described in OpenCode's Codex auth plugin docs: keep the
/// request payload "AI SDK compatible" by removing keys that are not part of the
/// backend contract.
pub fn strip_sdk_only_fields(mut value: serde_json::Value) -> serde_json::Value {
    fn recurse(v: &mut serde_json::Value) {
        match v {
            serde_json::Value::Object(map) => {
                map.remove("item_reference");
                map.remove("parallel_tool_calls");
                map.remove("previous_response_id");
                map.remove("response_id");
                for child in map.values_mut() {
                    recurse(child);
                }
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    recurse(item);
                }
            }
            _ => {}
        }
    }

    recurse(&mut value);
    value
}

/// Parse a `text/event-stream` response body and return the last JSON `data:` payload.
///
/// The ChatGPT/Codex backend frequently returns SSE even when the client expects a single JSON.
pub fn sse_to_last_json_value(sse_body: &str) -> anyhow::Result<serde_json::Value> {
    let mut last: Option<serde_json::Value> = None;
    for raw_line in sse_body.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("data:") {
            let data = rest.trim();
            if data == "[DONE]" {
                break;
            }
            if data.is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                last = Some(v);
            }
        }
    }
    last.ok_or_else(|| anyhow::anyhow!("no JSON data frames found in SSE body"))
}

fn content_to_input_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Convert an Anthropic CreateMessageRequest into a Codex "responses" payload.
///
/// MangoCode previously sent ChatCompletions-shaped payloads to the Codex
/// responses endpoint, which can produce HTTP 400. The Codex backend expects a
/// Responses-style body with `input` instead of `messages`.
pub fn anthropic_to_codex_responses_request(request: &CreateMessageRequest) -> Value {
    let mut input_items: Vec<Value> = Vec::new();

    if let Some(system) = &request.system {
        let system_text = match system {
            SystemPrompt::Text(text) => text.clone(),
            SystemPrompt::Blocks(blocks) => blocks
                .iter()
                .map(|b| b.text.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        };
        input_items.push(json!({
            "role": "system",
            "content": [{ "type": "input_text", "text": system_text }],
        }));
    }

    for msg in &request.messages {
        let role = msg.role.to_lowercase();
        let text = content_to_input_text(&msg.content);
        input_items.push(json!({
            "role": role,
            "content": [{ "type": "input_text", "text": text }],
        }));
    }

    let mut body = json!({
        "model": request.model,
        "input": input_items,
        // Codex/ChatGPT backend requires stateless operation.
        "store": false,
        // MangoCode uses a synthesized stream; keep Codex non-streaming for now.
        "stream": false,
        // Responses API naming.
        "max_output_tokens": request.max_tokens,
    });

    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(top_p) = request.top_p {
        body["top_p"] = json!(top_p);
    }

    strip_sdk_only_fields(body)
}

fn parse_responses_output_text(resp: &Value) -> Option<String> {
    let out = resp.get("output")?.as_array()?;
    let mut text = String::new();
    for item in out {
        let Some(parts) = item.get("content").and_then(|v| v.as_array()) else {
            continue;
        };
        for part in parts {
            let part_type = part.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if part_type == "output_text" || part_type == "text" {
                if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                    text.push_str(t);
                }
            }
        }
    }
    if text.is_empty() { None } else { Some(text) }
}

fn parse_chat_completions_text(resp: &Value) -> Option<String> {
    resp.get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .map(|s| s.to_string())
}

/// Parse either a Responses-shaped Codex response (preferred) or a legacy
/// ChatCompletions-shaped response.
///
/// Returns (content_text, stop_reason-ish, input_tokens, output_tokens)
pub fn parse_codex_response(response: &Value) -> (String, String, u64, u64) {
    let content = parse_responses_output_text(response)
        .or_else(|| parse_chat_completions_text(response))
        .unwrap_or_default();

    let stop_reason = response
        .get("finish_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("end_turn")
        .to_string();

    let usage = response.get("usage");
    let input_tokens = usage
        .and_then(|u| u.get("input_tokens").or_else(|| u.get("prompt_tokens")))
        .and_then(|t| t.as_u64())
        .unwrap_or(0);
    let output_tokens = usage
        .and_then(|u| u.get("output_tokens").or_else(|| u.get("completion_tokens")))
        .and_then(|t| t.as_u64())
        .unwrap_or(0);

    (content, stop_reason, input_tokens, output_tokens)
}

/// Build an Anthropic CreateMessageResponse from parsed OpenAI data.
pub fn build_anthropic_response(
    content: &str,
    stop_reason: &str,
    input_tokens: u64,
    output_tokens: u64,
    model: &str,
) -> CreateMessageResponse {
    // Generate a simple message ID
    let id = format!(
        "msg_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| format!("{:x}", d.as_nanos()))
            .unwrap_or_else(|_| "unknown".to_string())
    );

    CreateMessageResponse {
        id,
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content: vec![json!({
            "type": "text",
            "text": content,
        })],
        model: model.to_string(),
        stop_reason: Some(stop_reason.to_string()),
        stop_sequence: None,
        usage: UsageInfo {
            input_tokens,
            output_tokens,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ApiMessage, SystemPrompt};

    #[test]
    fn test_anthropic_to_codex_request_basic() {
        let request = CreateMessageRequest {
            model: "gpt-5.2-codex".to_string(),
            max_tokens: 1024,
            messages: vec![ApiMessage {
                role: "user".to_string(),
                content: json!("Hello"),
            }],
            system: Some(SystemPrompt::Text("You are helpful".to_string())),
            tools: None,
            temperature: Some(0.7),
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            thinking: None,
        };

        let codex_req = anthropic_to_codex_responses_request(&request);

        // Verify structure
        assert_eq!(codex_req["model"], "gpt-5.2-codex");
        assert_eq!(codex_req["max_output_tokens"], 1024);
        let temp = codex_req["temperature"]
            .as_f64()
            .expect("temperature should be numeric");
        assert!((temp - 0.7_f64).abs() < 1e-6_f64);
        assert!(codex_req["input"].is_array());

        let input = codex_req["input"].as_array().unwrap();
        assert_eq!(input.len(), 2); // system + user
        assert_eq!(input[0]["role"], "system");
        assert_eq!(input[1]["role"], "user");

        // Codex requests must be stateless.
        assert_eq!(codex_req["store"], json!(false));
    }

    #[test]
    fn sdk_only_fields_are_removed() {
        let input = serde_json::json!({
            "messages": [],
            "item_reference": "abc",
            "nested": {
                "parallel_tool_calls": true,
                "previous_response_id": "prev"
            }
        });

        let out = strip_sdk_only_fields(input);
        assert!(out.get("item_reference").is_none());
        assert!(out["nested"].get("parallel_tool_calls").is_none());
        assert!(out["nested"].get("previous_response_id").is_none());
    }

    #[test]
    fn test_parse_openai_response_basic() {
        let openai_resp = json!({
            "choices": [{
                "message": {
                    "content": "Hello, world!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        });

        let (content, stop_reason, input_tokens, output_tokens) = parse_codex_response(&openai_resp);

        assert_eq!(content, "Hello, world!");
        assert_eq!(stop_reason, "end_turn");
        assert_eq!(input_tokens, 10);
        assert_eq!(output_tokens, 5);
    }

    #[test]
    fn test_build_anthropic_response() {
        let response =
            build_anthropic_response("Test response", "end_turn", 100, 50, "gpt-5.2-codex");

        assert_eq!(response.response_type, "message");
        assert_eq!(response.role, "assistant");
        assert_eq!(response.model, "gpt-5.2-codex");
        assert_eq!(response.stop_reason, Some("end_turn".to_string()));
        assert_eq!(response.usage.input_tokens, 100);
        assert_eq!(response.usage.output_tokens, 50);
    }
}
