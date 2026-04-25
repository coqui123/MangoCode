//! Codex schema adapter — translates between Anthropic Messages API and OpenAI API formats.
//!
//! When using OpenAI Codex provider, requests are translated from Anthropic's
//! CreateMessageRequest format to OpenAI's ChatCompletion API format, and responses
//! are translated back to Anthropic's CreateMessageResponse format.

use super::types::{ApiToolDefinition, CreateMessageRequest, CreateMessageResponse, SystemPrompt};
use mangocode_core::codex_oauth::{normalize_codex_model, CODEX_API_ENDPOINT};
use mangocode_core::types::UsageInfo;
use serde_json::{json, Value};
use std::collections::BTreeMap;

/// OpenAI Codex API endpoint for responses (ChatGPT plan / Codex OAuth).
pub const CODEX_RESPONSES_ENDPOINT: &str = CODEX_API_ENDPOINT;
const CODEX_DEBUG_ENV: &str = "MANGOCODE_CODEX_DEBUG";

#[derive(Debug, Clone)]
struct PendingToolCall {
    call_id: String,
    name: String,
    arguments_json: String,
}

fn codex_debug_enabled() -> bool {
    std::env::var(CODEX_DEBUG_ENV)
        .ok()
        .map(|v| {
            let t = v.trim();
            t == "1" || t.eq_ignore_ascii_case("true") || t.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(false)
}

fn debug_codex_event(event_type: &str, tool_call: Option<&PendingToolCall>, accumulated: bool) {
    if !codex_debug_enabled() {
        return;
    }
    tracing::debug!(
        event_type = %event_type,
        has_tool_call = tool_call.is_some(),
        tool_call_id = tool_call.map(|tc| tc.call_id.as_str()).unwrap_or("<none>"),
        tool_name = tool_call.map(|tc| tc.name.as_str()).unwrap_or("<none>"),
        arguments_accumulated = accumulated,
        "parsed Codex SSE event"
    );
}

fn tool_call_key(v: &Value) -> Option<String> {
    v.get("call_id")
        .or_else(|| v.get("item").and_then(|i| i.get("call_id")))
        .or_else(|| v.get("id"))
        .or_else(|| v.get("item_id"))
        .or_else(|| v.get("output_item_id"))
        .and_then(|value| value.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            v.get("output_index")
                .and_then(|value| value.as_u64())
                .map(|idx| format!("output_index:{}", idx))
        })
}

fn output_item_tool_call_key(item: &Value) -> Option<String> {
    item.get("id")
        .or_else(|| item.get("item_id"))
        .or_else(|| item.get("output_item_id"))
        .or_else(|| item.get("call_id"))
        .and_then(|value| value.as_str())
        .map(|s| s.to_string())
}

fn pending_tool_call_from_item(item: &Value) -> Option<PendingToolCall> {
    let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if !matches!(item_type, "function_call" | "tool_call" | "function") {
        return None;
    }
    let name = item
        .get("name")
        .or_else(|| item.get("function").and_then(|f| f.get("name")))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if name.is_empty() {
        return None;
    }
    let call_id = item
        .get("call_id")
        .or_else(|| item.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let arguments_json = item
        .get("arguments")
        .or_else(|| item.get("function").and_then(|f| f.get("arguments")))
        .map(|v| match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        })
        .unwrap_or_default();
    Some(PendingToolCall {
        call_id,
        name,
        arguments_json,
    })
}

fn pending_tool_call_to_output_item(call: &PendingToolCall) -> Value {
    json!({
        "type": "function_call",
        "call_id": call.call_id,
        "name": call.name,
        "arguments": call.arguments_json,
    })
}

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

/// Parse a `text/event-stream` response body and return the most useful JSON payload.
///
/// The ChatGPT/Codex backend frequently returns SSE even when the client expects a single JSON.
pub fn sse_to_last_json_value(sse_body: &str) -> anyhow::Result<serde_json::Value> {
    let mut last: Option<serde_json::Value> = None;
    let mut last_with_output: Option<serde_json::Value> = None;
    let mut completed: Option<serde_json::Value> = None;
    let mut output_text = String::new();
    let mut output_items: Vec<Value> = Vec::new();
    let mut pending_tool_calls: BTreeMap<String, PendingToolCall> = BTreeMap::new();

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
                let event_type = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
                let mut event_tool_call: Option<PendingToolCall> = None;
                let mut arguments_accumulated = false;
                match event_type {
                    "response.output_text.delta" => {
                        if let Some(delta) = v.get("delta").and_then(|d| d.as_str()) {
                            output_text.push_str(delta);
                        }
                    }
                    "response.output_text.done" => {
                        if let Some(text) = v.get("text").and_then(|d| d.as_str()) {
                            output_text.clear();
                            output_text.push_str(text);
                        }
                    }
                    "response.content_part.done" => {
                        if let Some(text) = collect_content_text(v.get("part").unwrap_or(&v)) {
                            if output_text.is_empty() {
                                output_text.push_str(&text);
                            }
                        }
                    }
                    "response.output_item.added" => {
                        if let Some(item) = v.get("item") {
                            if let Some(call) = pending_tool_call_from_item(item) {
                                let key = output_item_tool_call_key(item)
                                    .or_else(|| tool_call_key(&v))
                                    .unwrap_or_else(|| call.call_id.clone());
                                pending_tool_calls.insert(key, call.clone());
                                event_tool_call = Some(call);
                            }
                        }
                    }
                    "response.function_call_arguments.delta" => {
                        if let Some(delta) = v
                            .get("delta")
                            .or_else(|| v.get("arguments_delta"))
                            .and_then(|d| d.as_str())
                        {
                            if let Some(key) = tool_call_key(&v) {
                                let call =
                                    pending_tool_calls.entry(key.clone()).or_insert_with(|| {
                                        PendingToolCall {
                                            call_id: v
                                                .get("call_id")
                                                .and_then(|id| id.as_str())
                                                .unwrap_or(&key)
                                                .to_string(),
                                            name: v
                                                .get("name")
                                                .and_then(|name| name.as_str())
                                                .unwrap_or("")
                                                .to_string(),
                                            arguments_json: String::new(),
                                        }
                                    });
                                call.arguments_json.push_str(delta);
                                arguments_accumulated = true;
                                event_tool_call = Some(call.clone());
                            }
                        }
                    }
                    "response.function_call_arguments.done" => {
                        if let Some(key) = tool_call_key(&v) {
                            let call = pending_tool_calls.entry(key.clone()).or_insert_with(|| {
                                PendingToolCall {
                                    call_id: v
                                        .get("call_id")
                                        .and_then(|id| id.as_str())
                                        .unwrap_or(&key)
                                        .to_string(),
                                    name: v
                                        .get("name")
                                        .and_then(|name| name.as_str())
                                        .unwrap_or("")
                                        .to_string(),
                                    arguments_json: String::new(),
                                }
                            });
                            if let Some(name) = v.get("name").and_then(|name| name.as_str()) {
                                if !name.is_empty() {
                                    call.name = name.to_string();
                                }
                            }
                            if let Some(arguments) = v
                                .get("arguments")
                                .or_else(|| v.get("arguments_json"))
                                .and_then(|args| args.as_str())
                            {
                                call.arguments_json.clear();
                                call.arguments_json.push_str(arguments);
                            }
                            arguments_accumulated = !call.arguments_json.is_empty();
                            event_tool_call = Some(call.clone());
                        }
                    }
                    "response.output_item.done" => {
                        if let Some(item) = v.get("item") {
                            if let Some(call) = pending_tool_call_from_item(item) {
                                let key = output_item_tool_call_key(item)
                                    .or_else(|| tool_call_key(&v))
                                    .unwrap_or_else(|| call.call_id.clone());
                                pending_tool_calls.insert(key, call.clone());
                                output_items.push(item.clone());
                                event_tool_call = Some(call);
                            } else {
                                if let Some(text) = collect_output_item_text(item) {
                                    if output_text.is_empty() {
                                        output_text.push_str(&text);
                                    }
                                }
                                output_items.push(item.clone());
                            }
                        }
                    }
                    "response.completed" => {
                        completed = Some(v.clone());
                    }
                    _ => {}
                }
                debug_codex_event(event_type, event_tool_call.as_ref(), arguments_accumulated);
                let response_body = v.get("response").unwrap_or(&v);
                if response_body.get("output").is_some() || response_body.get("choices").is_some() {
                    last_with_output = Some(v.clone());
                }
                last = Some(v);
            }
        }
    }

    if let Some(v) = &completed {
        if response_content_blocks_from_value(v)
            .iter()
            .any(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_use"))
            || response_text_from_value(v).is_some()
        {
            return Ok(v.clone());
        }
    }
    if let Some(v) = &last_with_output {
        if response_content_blocks_from_value(v)
            .iter()
            .any(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_use"))
            || response_text_from_value(v).is_some()
        {
            return Ok(v.clone());
        }
    }
    let mut pending_output_items = pending_tool_calls
        .values()
        .filter(|call| !call.name.is_empty())
        .map(pending_tool_call_to_output_item)
        .collect::<Vec<_>>();
    if !pending_output_items.is_empty() {
        let mut output = Vec::new();
        if !output_text.is_empty() {
            output.push(json!({
                "type": "message",
                "content": [{ "type": "output_text", "text": output_text }]
            }));
        }
        output.append(&mut pending_output_items);
        return Ok(json!({ "output": output }));
    }
    if output_items
        .iter()
        .any(|item| collect_output_item_tool_use(item).is_some())
    {
        return Ok(json!({ "output": output_items }));
    }
    if !output_text.is_empty() {
        return Ok(json!({
            "output": [{
                "type": "message",
                "content": [{ "type": "output_text", "text": output_text }]
            }]
        }));
    }
    if let Some(v) = completed {
        return Ok(v);
    }
    if let Some(v) = last_with_output {
        return Ok(v);
    }
    last.ok_or_else(|| anyhow::anyhow!("no JSON data frames found in SSE body"))
}

fn normalize_system_prompt_text(text: &str) -> String {
    // Conservative normalization: trim and collapse repeated blank lines.
    let trimmed = text.trim();
    let mut out = String::new();
    let mut last_blank = false;
    for line in trimmed.lines() {
        let is_blank = line.trim().is_empty();
        if is_blank {
            if last_blank {
                continue;
            }
            last_blank = true;
            out.push('\n');
            continue;
        }
        last_blank = false;
        out.push_str(line.trim_end());
        out.push('\n');
    }
    out.trim().to_string()
}

fn blocks_value_to_plain_text(blocks: &[Value]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for b in blocks {
        let t = b.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match t {
            "text" => {
                if let Some(s) = b.get("text").and_then(|v| v.as_str()) {
                    if !s.trim().is_empty() {
                        parts.push(s.trim().to_string());
                    }
                }
            }
            // Tool/result blocks: codex backend may reject malformed sequences; coerce to text.
            "tool_use" => {
                let name = b.get("name").and_then(|v| v.as_str()).unwrap_or("tool");
                parts.push(format!("[tool_use: {}]", name));
            }
            "tool_result" => {
                let content = b.get("content");
                let summary = match content {
                    Some(Value::String(s)) => s.clone(),
                    Some(v) => v.to_string(),
                    None => String::new(),
                };
                if summary.trim().is_empty() {
                    parts.push("[tool_result]".to_string());
                } else {
                    parts.push(format!("[tool_result]\n{}", summary));
                }
            }
            // Drop thinking/redacted_thinking/etc.
            _ => {}
        }
    }
    parts.join("\n")
}

fn content_to_input_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(items) => blocks_value_to_plain_text(items),
        other => other.to_string(),
    }
}

fn tool_result_content_to_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => blocks_value_to_plain_text(blocks),
        Some(v) => v.to_string(),
        None => String::new(),
    }
}

fn codex_content_type_for_role(role: &str) -> &'static str {
    if role.eq_ignore_ascii_case("assistant") {
        "output_text"
    } else {
        "input_text"
    }
}

fn codex_tool_definition(tool: &ApiToolDefinition) -> Value {
    json!({
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema,
    })
}

fn append_codex_input_items_for_message(input_items: &mut Vec<Value>, role: &str, content: &Value) {
    let role = role.to_lowercase();
    let Some(blocks) = content.as_array() else {
        let text = content_to_input_text(content);
        let content_type = codex_content_type_for_role(&role);
        input_items.push(json!({
            "role": role,
            "content": [{ "type": content_type, "text": text }],
        }));
        return;
    };

    let mut text_parts: Vec<Value> = Vec::new();
    let flush_text = |input_items: &mut Vec<Value>, text_parts: &mut Vec<Value>| {
        if !text_parts.is_empty() {
            input_items.push(json!({
                "role": role,
                "content": std::mem::take(text_parts),
            }));
        }
    };

    for block in blocks {
        match block.get("type").and_then(|v| v.as_str()).unwrap_or("") {
            "text" => {
                if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        text_parts.push(json!({
                            "type": codex_content_type_for_role(&role),
                            "text": text,
                        }));
                    }
                }
            }
            "tool_use" if role == "assistant" => {
                flush_text(input_items, &mut text_parts);
                let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let arguments = block
                    .get("input")
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "{}".to_string());
                input_items.push(json!({
                    "type": "function_call",
                    "call_id": id,
                    "name": name,
                    "arguments": arguments,
                }));
            }
            "tool_result" => {
                flush_text(input_items, &mut text_parts);
                let call_id = block
                    .get("tool_use_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                input_items.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": tool_result_content_to_text(block.get("content")),
                }));
            }
            _ => {}
        }
    }

    flush_text(input_items, &mut text_parts);
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
            SystemPrompt::Text(text) => normalize_system_prompt_text(text),
            SystemPrompt::Blocks(blocks) => blocks
                .iter()
                .map(|b| normalize_system_prompt_text(&b.text))
                .collect::<Vec<_>>()
                .join("\n"),
        };
        // Plugin-style: prefer top-level `instructions` over a system message item.
        // We still keep `input` purely conversational.
        // (Set below on `body`.)
        input_items.push(json!({
            "role": "system",
            "content": [{ "type": "input_text", "text": system_text }],
        }));
    }

    for msg in &request.messages {
        append_codex_input_items_for_message(&mut input_items, &msg.role, &msg.content);
    }

    // If the first item is a system message, lift it into `instructions` and drop it from input
    // to better match Codex Responses conventions.
    let mut instructions: Option<String> = None;
    if let Some(first) = input_items.first() {
        if first.get("role").and_then(|v| v.as_str()) == Some("system") {
            let text = first
                .get("content")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("text"))
                .and_then(|t| t.as_str())
                .map(|s| s.to_string());
            instructions = text;
            input_items.remove(0);
        }
    }

    let mut body = json!({
        "model": normalize_codex_model(&request.model),
        "input": input_items,
        // Codex/ChatGPT backend requires stateless operation.
        "store": false,
        // The ChatGPT Codex backend requires streaming; the provider collects
        // the final SSE payload and synthesizes MangoCode stream events.
        "stream": true,
    });
    if let Some(instr) = instructions {
        body["instructions"] = json!(instr);
    }
    if let Some(tools) = &request.tools {
        if !tools.is_empty() {
            body["tools"] = json!(tools.iter().map(codex_tool_definition).collect::<Vec<_>>());
        }
    }

    // The ChatGPT Codex backend is not the public /v1/responses endpoint and
    // rejects several public Responses fields, including max_output_tokens,
    // temperature, and top_p. Let the backend/model defaults apply.
    // If thinking is enabled, request encrypted reasoning continuity (plugin style).
    if request.thinking.is_some() {
        body["include"] = json!(["reasoning.encrypted_content"]);
    }

    strip_sdk_only_fields(body)
}

fn collect_content_text(part: &Value) -> Option<String> {
    let part_type = part.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if matches!(part_type, "output_text" | "text" | "input_text") || part_type.is_empty() {
        if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
        if let Some(t) = part.get("output_text").and_then(|v| v.as_str()) {
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
        if let Some(t) = part.get("delta").and_then(|v| v.as_str()) {
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
    }
    None
}

fn collect_output_item_text(item: &Value) -> Option<String> {
    let mut text = String::new();
    if let Some(parts) = item.get("content").and_then(|v| v.as_array()) {
        for part in parts {
            if let Some(t) = collect_content_text(part) {
                text.push_str(&t);
            }
        }
    }
    if text.is_empty() {
        item.get("text")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
    } else {
        Some(text)
    }
}

fn collect_output_item_tool_use(item: &Value) -> Option<Value> {
    let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if !matches!(item_type, "function_call" | "tool_call" | "function") {
        return None;
    }

    let id = item
        .get("call_id")
        .or_else(|| item.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let name = item
        .get("name")
        .or_else(|| item.get("function").and_then(|f| f.get("name")))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if name.is_empty() {
        return None;
    }

    let args_value = item
        .get("arguments")
        .or_else(|| item.get("function").and_then(|f| f.get("arguments")));
    let input = match args_value {
        Some(Value::String(s)) => serde_json::from_str::<Value>(s).unwrap_or_else(|_| json!({})),
        Some(v) => v.clone(),
        None => json!({}),
    };

    Some(json!({
        "type": "tool_use",
        "id": id,
        "name": name,
        "input": input,
    }))
}

fn response_text_from_value(resp: &Value) -> Option<String> {
    parse_responses_output_text(resp).or_else(|| parse_chat_completions_text(resp))
}

fn response_content_blocks_from_value(resp: &Value) -> Vec<Value> {
    let resp = resp.get("response").unwrap_or(resp);
    let mut blocks: Vec<Value> = Vec::new();

    if let Some(out) = resp.get("output").and_then(|v| v.as_array()) {
        for item in out {
            if let Some(tool_use) = collect_output_item_tool_use(item) {
                blocks.push(tool_use);
            } else if let Some(text) = collect_output_item_text(item) {
                if !text.is_empty() {
                    blocks.push(json!({ "type": "text", "text": text }));
                }
            }
        }
    }

    if blocks.is_empty() {
        if let Some(message) = resp
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
        {
            if let Some(text) = message.get("content").and_then(|c| c.as_str()) {
                if !text.is_empty() {
                    blocks.push(json!({ "type": "text", "text": text }));
                }
            }
            if let Some(tool_calls) = message.get("tool_calls").and_then(|t| t.as_array()) {
                for tc in tool_calls {
                    if let Some(tool_use) = collect_output_item_tool_use(tc) {
                        blocks.push(tool_use);
                    }
                }
            }
        }
    }

    if blocks.is_empty() {
        if let Some(text) = response_text_from_value(resp) {
            if !text.is_empty() {
                blocks.push(json!({ "type": "text", "text": text }));
            }
        }
    }

    blocks
}

fn parse_responses_output_text(resp: &Value) -> Option<String> {
    let resp = resp.get("response").unwrap_or(resp);
    if let Some(t) = resp.get("output_text").and_then(|v| v.as_str()) {
        if !t.is_empty() {
            return Some(t.to_string());
        }
    }
    if let Some(parts) = resp.get("output_text").and_then(|v| v.as_array()) {
        let mut text = String::new();
        for part in parts {
            match part {
                Value::String(s) => text.push_str(s),
                Value::Object(_) => {
                    if let Some(t) = collect_content_text(part) {
                        text.push_str(&t);
                    }
                }
                _ => {}
            }
        }
        if !text.is_empty() {
            return Some(text);
        }
    }
    let out = resp.get("output")?.as_array()?;
    let mut text = String::new();
    for item in out {
        if let Some(t) = collect_output_item_text(item) {
            text.push_str(&t);
        }
    }
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
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
    let response_body = response.get("response").unwrap_or(response);
    let content = response_text_from_value(response_body).unwrap_or_default();

    let stop_reason = response_body
        .get("finish_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("end_turn")
        .to_string();

    let usage = response_body.get("usage");
    let input_tokens = usage
        .and_then(|u| u.get("input_tokens").or_else(|| u.get("prompt_tokens")))
        .and_then(|t| t.as_u64())
        .unwrap_or(0);
    let output_tokens = usage
        .and_then(|u| {
            u.get("output_tokens")
                .or_else(|| u.get("completion_tokens"))
        })
        .and_then(|t| t.as_u64())
        .unwrap_or(0);

    (content, stop_reason, input_tokens, output_tokens)
}

/// Parse Codex output into Anthropic-compatible content blocks plus stop/usage metadata.
pub fn parse_codex_response_blocks(response: &Value) -> (Vec<Value>, String, u64, u64) {
    let response_body = response.get("response").unwrap_or(response);
    let blocks = response_content_blocks_from_value(response_body);
    let (_, mut stop_reason, input_tokens, output_tokens) = parse_codex_response(response_body);

    if blocks
        .iter()
        .any(|b| b.get("type").and_then(|v| v.as_str()) == Some("tool_use"))
    {
        stop_reason = "tool_use".to_string();
    }

    (blocks, stop_reason, input_tokens, output_tokens)
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

pub fn build_anthropic_response_from_blocks(
    content: Vec<Value>,
    stop_reason: &str,
    input_tokens: u64,
    output_tokens: u64,
    model: &str,
) -> CreateMessageResponse {
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
        content,
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
        assert!(codex_req["input"].is_array());
        assert!(codex_req.get("max_output_tokens").is_none());
        assert!(codex_req.get("temperature").is_none());
        assert!(codex_req.get("top_p").is_none());

        let input = codex_req["input"].as_array().unwrap();
        // System prompt is lifted to top-level `instructions`, so input is user-only.
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["role"], "user");
        assert_eq!(codex_req["instructions"], "You are helpful");

        // Codex requests must be stateless.
        assert_eq!(codex_req["store"], json!(false));
        assert_eq!(codex_req["stream"], json!(true));
    }

    #[test]
    fn codex_request_uses_output_text_for_assistant_history() {
        let request = CreateMessageRequest {
            model: "gpt-5.5".to_string(),
            max_tokens: 1024,
            messages: vec![
                ApiMessage {
                    role: "user".to_string(),
                    content: json!("hi"),
                },
                ApiMessage {
                    role: "assistant".to_string(),
                    content: json!("Hello!"),
                },
                ApiMessage {
                    role: "user".to_string(),
                    content: json!("again"),
                },
            ],
            system: None,
            tools: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            thinking: None,
        };

        let codex_req = anthropic_to_codex_responses_request(&request);
        let input = codex_req["input"].as_array().unwrap();

        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[1]["role"], "assistant");
        assert_eq!(input[1]["content"][0]["type"], "output_text");
        assert_eq!(input[2]["content"][0]["type"], "input_text");
    }

    #[test]
    fn codex_request_includes_function_tools() {
        let request = CreateMessageRequest {
            model: "gpt-5.5".to_string(),
            max_tokens: 1024,
            messages: vec![ApiMessage {
                role: "user".to_string(),
                content: json!("read the file"),
            }],
            system: None,
            tools: Some(vec![ApiToolDefinition {
                name: "Read".to_string(),
                description: "Read a file".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": { "file_path": { "type": "string" } },
                    "required": ["file_path"]
                }),
                cache_control: None,
            }]),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            thinking: None,
        };

        let codex_req = anthropic_to_codex_responses_request(&request);
        let tools = codex_req["tools"].as_array().expect("tools present");

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["name"], "Read");
        assert_eq!(tools[0]["parameters"]["required"][0], "file_path");
    }

    #[test]
    fn codex_request_preserves_tool_result_history() {
        let request = CreateMessageRequest {
            model: "gpt-5.5".to_string(),
            max_tokens: 1024,
            messages: vec![
                ApiMessage {
                    role: "assistant".to_string(),
                    content: json!([{
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "Read",
                        "input": { "file_path": "Cargo.toml" }
                    }]),
                },
                ApiMessage {
                    role: "user".to_string(),
                    content: json!([{
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "file contents"
                    }]),
                },
            ],
            system: None,
            tools: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            thinking: None,
        };

        let codex_req = anthropic_to_codex_responses_request(&request);
        let input = codex_req["input"].as_array().unwrap();

        assert_eq!(input[0]["type"], "function_call");
        assert_eq!(input[0]["call_id"], "call_1");
        assert_eq!(input[0]["name"], "Read");
        assert_eq!(input[1]["type"], "function_call_output");
        assert_eq!(input[1]["call_id"], "call_1");
        assert_eq!(input[1]["output"], "file contents");
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

        let (content, stop_reason, input_tokens, output_tokens) =
            parse_codex_response(&openai_resp);

        assert_eq!(content, "Hello, world!");
        assert_eq!(stop_reason, "end_turn");
        assert_eq!(input_tokens, 10);
        assert_eq!(output_tokens, 5);
    }

    #[test]
    fn parses_codex_response_completed_envelope() {
        let event = json!({
            "type": "response.completed",
            "response": {
                "output": [{
                    "content": [{ "type": "output_text", "text": "Hello from Codex" }]
                }],
                "usage": { "input_tokens": 3, "output_tokens": 4 }
            }
        });

        let (content, stop_reason, input_tokens, output_tokens) = parse_codex_response(&event);

        assert_eq!(content, "Hello from Codex");
        assert_eq!(stop_reason, "end_turn");
        assert_eq!(input_tokens, 3);
        assert_eq!(output_tokens, 4);
    }

    #[test]
    fn parses_codex_function_call_as_tool_use_block() {
        let event = json!({
            "type": "response.completed",
            "response": {
                "output": [{
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "Read",
                    "arguments": "{\"file_path\":\"Cargo.toml\"}"
                }],
                "usage": { "input_tokens": 3, "output_tokens": 4 }
            }
        });

        let (blocks, stop_reason, input_tokens, output_tokens) =
            parse_codex_response_blocks(&event);

        assert_eq!(stop_reason, "tool_use");
        assert_eq!(input_tokens, 3);
        assert_eq!(output_tokens, 4);
        assert_eq!(blocks[0]["type"], "tool_use");
        assert_eq!(blocks[0]["id"], "call_abc");
        assert_eq!(blocks[0]["name"], "Read");
        assert_eq!(blocks[0]["input"]["file_path"], "Cargo.toml");
    }

    #[test]
    fn parses_codex_function_call_output_item() {
        let response = json!({
            "output": [{
                "type": "function_call",
                "call_id": "call_ls",
                "name": "Bash",
                "arguments": "{\"command\":\"ls\"}"
            }]
        });

        let (blocks, stop_reason, _, _) = parse_codex_response_blocks(&response);

        assert_eq!(stop_reason, "tool_use");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "tool_use");
        assert_eq!(blocks[0]["id"], "call_ls");
        assert_eq!(blocks[0]["name"], "Bash");
        assert_eq!(blocks[0]["input"]["command"], "ls");
    }

    #[test]
    fn sse_completed_tool_call_without_text_is_not_downgraded() {
        let sse = r#"data: {"type":"response.completed","response":{"status":"completed","output":[{"type":"function_call","call_id":"call_1","name":"Bash","arguments":"{\"command\":\"ls\"}"}]}}
"#;

        let value = sse_to_last_json_value(sse).expect("sse parsed");
        let (blocks, stop_reason, _, _) = parse_codex_response_blocks(&value);

        assert_eq!(stop_reason, "tool_use");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "tool_use");
        assert_eq!(blocks[0]["name"], "Bash");
        assert_eq!(blocks[0]["input"]["command"], "ls");
    }

    #[test]
    fn tool_call_beats_plain_text_when_both_exist() {
        let response = json!({
            "output": [
                {
                    "type": "message",
                    "content": [{ "type": "output_text", "text": "I will check." }]
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Read",
                    "arguments": "{\"file_path\":\"Cargo.toml\"}"
                }
            ]
        });

        let (blocks, stop_reason, _, _) = parse_codex_response_blocks(&response);

        assert_eq!(stop_reason, "tool_use");
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "tool_use");
        assert_eq!(blocks[1]["name"], "Read");
    }

    #[test]
    fn accumulates_function_call_argument_deltas() {
        let sse = r#"data: {"type":"response.output_item.added","item":{"type":"function_call","id":"item_1","call_id":"call_1","name":"Bash","arguments":""}}
data: {"type":"response.function_call_arguments.delta","item_id":"item_1","delta":"{\"command\""}
data: {"type":"response.function_call_arguments.delta","item_id":"item_1","delta":":\"ls\"}"}
data: {"type":"response.function_call_arguments.done","item_id":"item_1","arguments":"{\"command\":\"ls\"}"}
data: {"type":"response.completed","response":{"status":"completed","output":[]}}
"#;

        let value = sse_to_last_json_value(sse).expect("sse parsed");
        let (blocks, stop_reason, _, _) = parse_codex_response_blocks(&value);

        assert_eq!(stop_reason, "tool_use");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "tool_use");
        assert_eq!(blocks[0]["id"], "call_1");
        assert_eq!(blocks[0]["name"], "Bash");
        assert_eq!(blocks[0]["input"]["command"], "ls");
    }

    #[test]
    fn plain_text_without_tool_call_stays_text() {
        let response = json!({
            "output": [{
                "type": "message",
                "content": [{ "type": "output_text", "text": "Nothing to call." }]
            }]
        });

        let (blocks, stop_reason, _, _) = parse_codex_response_blocks(&response);

        assert_eq!(stop_reason, "end_turn");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[0]["text"], "Nothing to call.");
    }

    #[test]
    fn conversational_text_regression_does_not_fake_tool_call() {
        let sse = r#"data: {"type":"response.output_text.delta","delta":"can you just ls or whatever the directory youre in to test you can utilize the tools"}
data: {"type":"response.completed","response":{"status":"completed","output":[]}}
"#;

        let value = sse_to_last_json_value(sse).expect("sse parsed");
        let (blocks, stop_reason, _, _) = parse_codex_response_blocks(&value);

        assert_eq!(stop_reason, "end_turn");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "text");
        assert!(blocks[0]["text"]
            .as_str()
            .unwrap()
            .contains("can you just ls"));
    }

    #[test]
    fn tool_call_response_emits_tool_use_stop_reason() {
        let response = json!({
            "output": [{
                "type": "function_call",
                "call_id": "call_1",
                "name": "Glob",
                "arguments": "{\"pattern\":\"*.rs\"}"
            }],
            "finish_reason": "stop"
        });

        let (_, stop_reason, _, _) = parse_codex_response_blocks(&response);

        assert_eq!(stop_reason, "tool_use");
    }

    #[test]
    fn sse_parser_prefers_completed_response_over_trailing_metadata() {
        let sse = r#"event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"Hi"}

event: response.completed
data: {"type":"response.completed","response":{"output":[{"content":[{"type":"output_text","text":"Hi there"}]}],"usage":{"input_tokens":1,"output_tokens":2}}}

event: response.done
data: {"type":"response.done"}

"#;

        let value = sse_to_last_json_value(sse).expect("sse parsed");
        let (content, _, input_tokens, output_tokens) = parse_codex_response(&value);

        assert_eq!(content, "Hi there");
        assert_eq!(input_tokens, 1);
        assert_eq!(output_tokens, 2);
    }

    #[test]
    fn sse_parser_accumulates_delta_text_without_completed_response() {
        let sse = r#"data: {"type":"response.output_text.delta","delta":"Hel"}
data: {"type":"response.output_text.delta","delta":"lo"}
data: {"type":"response.done"}
"#;

        let value = sse_to_last_json_value(sse).expect("sse parsed");
        let (content, _, _, _) = parse_codex_response(&value);

        assert_eq!(content, "Hello");
    }

    #[test]
    fn sse_parser_uses_deltas_when_completed_has_no_text() {
        let sse = r#"data: {"type":"response.output_text.delta","delta":"Hel"}
data: {"type":"response.output_text.delta","delta":"lo"}
data: {"type":"response.completed","response":{"status":"completed","output":[]}}
"#;

        let value = sse_to_last_json_value(sse).expect("sse parsed");
        let (content, _, _, _) = parse_codex_response(&value);

        assert_eq!(content, "Hello");
    }

    #[test]
    fn parses_codex_output_text_done_event() {
        let sse = r#"data: {"type":"response.output_text.delta","delta":"partial"}
data: {"type":"response.output_text.done","text":"final text"}
data: {"type":"response.completed","response":{"status":"completed","output":[]}}
"#;

        let value = sse_to_last_json_value(sse).expect("sse parsed");
        let (content, _, _, _) = parse_codex_response(&value);

        assert_eq!(content, "final text");
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
