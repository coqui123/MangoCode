//! Provider serialization/deserialization integration tests.

use base64::Engine;
use futures::StreamExt;
use mangocode_api::provider::LlmProvider;
use mangocode_api::provider_types::{ProviderRequest, StopReason, SystemPrompt};
use mangocode_api::providers::openai_compat_providers;
use mangocode_api::providers::OpenAiCodexProvider;
use mangocode_api::{GoogleProvider, OpenAiProvider, ThinkingConfig};
use mangocode_core::types::{ContentBlock, ImageSource, Message};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

async fn spawn_json_server(response_body: Value) -> (String, oneshot::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test server");
    let addr = listener.local_addr().expect("local addr");
    let (tx, rx) = oneshot::channel::<String>();

    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.expect("accept connection");
        let mut buf = vec![0u8; 1024 * 1024];
        let mut n = 0usize;

        loop {
            let read = stream
                .read(&mut buf[n..])
                .await
                .expect("read request bytes");
            if read == 0 {
                break;
            }
            n += read;
            if n >= 4 && buf[..n].windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }

        let request = String::from_utf8_lossy(&buf[..n]).to_string();
        let header_end = request.find("\r\n\r\n").expect("header terminator");
        let headers = &request[..header_end];
        let body_start = header_end + 4;
        let content_length = headers
            .lines()
            .find_map(|line| {
                let lower = line.to_ascii_lowercase();
                if lower.starts_with("content-length:") {
                    line.split(':')
                        .nth(1)
                        .and_then(|v| v.trim().parse::<usize>().ok())
                } else {
                    None
                }
            })
            .unwrap_or(0);

        while n < body_start + content_length {
            let read = stream.read(&mut buf[n..]).await.expect("read request body");
            if read == 0 {
                break;
            }
            n += read;
        }

        let full_request = String::from_utf8_lossy(&buf[..n]).to_string();
        let _ = tx.send(full_request);

        let body = response_body.to_string();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );

        stream
            .write_all(response.as_bytes())
            .await
            .expect("write response");
    });

    (format!("http://{}", addr), rx)
}

async fn spawn_sse_server(response_body: String) -> (String, oneshot::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test server");
    let addr = listener.local_addr().expect("local addr");
    let (tx, rx) = oneshot::channel::<String>();

    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.expect("accept connection");
        let mut buf = vec![0u8; 1024 * 1024];
        let mut n = 0usize;

        loop {
            let read = stream
                .read(&mut buf[n..])
                .await
                .expect("read request bytes");
            if read == 0 {
                break;
            }
            n += read;
            if n >= 4 && buf[..n].windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }

        let full_request = String::from_utf8_lossy(&buf[..n]).to_string();
        let _ = tx.send(full_request);

        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response_body.len(),
            response_body
        );

        stream
            .write_all(response.as_bytes())
            .await
            .expect("write response");
    });

    (format!("http://{}", addr), rx)
}

fn base_request(model: &str) -> ProviderRequest {
    ProviderRequest {
        model: model.to_string(),
        messages: vec![Message::user("Say hello")],
        system_prompt: Some(SystemPrompt::Text("You are a test assistant".to_string())),
        tools: vec![],
        max_tokens: 64,
        temperature: Some(0.1),
        top_p: None,
        top_k: None,
        stop_sequences: vec![],
        thinking: Some(ThinkingConfig::enabled(128)),
        provider_options: json!({}),
    }
}

#[tokio::test]
async fn openai_provider_serializes_and_deserializes() {
    let response = json!({
        "id": "chatcmpl-test-1",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Hello from OpenAI mock"
            }
        }],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "total_tokens": 15
        }
    });

    let (base_url, req_rx) = spawn_json_server(response).await;
    let provider = OpenAiProvider::new("test-key".to_string()).with_base_url(base_url);

    let parsed = provider
        .create_message(base_request("gpt-4o-mini"))
        .await
        .expect("openai response parsed");

    let raw = req_rx.await.expect("captured request");
    let body = raw.split("\r\n\r\n").nth(1).expect("request body present");
    let body_json: Value = serde_json::from_str(body).expect("json request body");

    assert_eq!(body_json["model"], json!("gpt-4o-mini"));
    assert_eq!(body_json["messages"][0]["role"], json!("system"));
    assert_eq!(body_json["messages"][1]["role"], json!("user"));
    assert_eq!(parsed.stop_reason, StopReason::EndTurn);
    assert!(matches!(
        &parsed.content[0],
        ContentBlock::Text { text } if text.contains("OpenAI mock")
    ));
}

#[tokio::test]
async fn google_provider_serializes_and_deserializes() {
    let response = json!({
        "candidates": [{
            "finishReason": "STOP",
            "content": {
                "parts": [{ "text": "Hello from Google mock" }]
            }
        }],
        "usageMetadata": {
            "promptTokenCount": 9,
            "candidatesTokenCount": 4
        }
    });

    let (base_url, req_rx) = spawn_json_server(response).await;
    let provider = GoogleProvider::new("google-test-key".to_string()).with_base_url(base_url);

    let parsed = provider
        .create_message(base_request("gemini-2.5-flash"))
        .await
        .expect("google response parsed");

    let raw = req_rx.await.expect("captured request");
    let request_line = raw.lines().next().expect("request line");
    assert!(request_line
        .contains("/v1beta/models/gemini-2.5-flash:generateContent?key=google-test-key"));

    let body = raw.split("\r\n\r\n").nth(1).expect("request body present");
    let body_json: Value = serde_json::from_str(body).expect("json request body");

    assert_eq!(body_json["contents"][0]["role"], json!("user"));
    assert_eq!(
        body_json["contents"][0]["parts"][0]["text"],
        json!("Say hello")
    );
    assert_eq!(parsed.stop_reason, StopReason::EndTurn);
    assert!(matches!(
        &parsed.content[0],
        ContentBlock::Text { text } if text.contains("Google mock")
    ));
}

#[tokio::test]
async fn ollama_provider_serializes_without_auth_header() {
    let response = json!({
        "id": "chatcmpl-ollama-test-1",
        "model": "SimonPu/qwen3:30B-Thinking-2507-Q4_K_XL",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Hello from Ollama mock"
            }
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14
        }
    });

    let (base_url, req_rx) = spawn_json_server(response).await;
    let provider = openai_compat_providers::ollama().with_base_url(format!("{}/v1", base_url));

    let parsed = provider
        .create_message(base_request("SimonPu/qwen3:30B-Thinking-2507-Q4_K_XL"))
        .await
        .expect("ollama response parsed");

    let raw = req_rx.await.expect("captured request");
    let request_line = raw.lines().next().expect("request line");
    assert!(request_line.contains("/v1/chat/completions"));
    assert!(
        !raw.to_ascii_lowercase().contains("authorization: bearer"),
        "ollama requests should not include bearer auth by default"
    );

    let body = raw.split("\r\n\r\n").nth(1).expect("request body present");
    let body_json: Value = serde_json::from_str(body).expect("json request body");

    assert_eq!(
        body_json["model"],
        json!("SimonPu/qwen3:30B-Thinking-2507-Q4_K_XL")
    );
    assert_eq!(body_json["messages"][0]["role"], json!("system"));
    assert_eq!(body_json["messages"][1]["role"], json!("user"));
    assert_eq!(body_json["enable_thinking"], json!(true));
    assert_eq!(body_json["thinking_budget"], json!(128));
    assert!(body_json.get("reasoning_effort").is_none());
    assert!(body_json.get("reasoning").is_none());
    assert_eq!(parsed.stop_reason, StopReason::EndTurn);
    assert!(matches!(
        &parsed.content[0],
        ContentBlock::Text { text } if text.contains("Ollama mock")
    ));
}

#[tokio::test]
async fn ollama_stream_parses_nested_message_content() {
    let response_body = concat!(
        "data: {\"choices\":[{\"delta\":{\"message\":{\"content\":\"Hello\"}}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"message\":{\"content\":\" world\"}}}]}\n\n",
        "data: [DONE]\n\n"
    );

    let (base_url, _req_rx) = spawn_sse_server(response_body.to_string()).await;
    let provider = openai_compat_providers::ollama().with_base_url(format!("{}/v1", base_url));
    let mut stream = provider
        .create_message_stream(base_request("SimonPu/qwen3:30B-Thinking-2507-Q4_K_XL"))
        .await
        .expect("create stream");

    let mut text = String::new();
    while let Some(event) = stream.next().await {
        match event.expect("stream item") {
            mangocode_api::StreamEvent::TextDelta { text: chunk, .. } => text.push_str(&chunk),
            mangocode_api::StreamEvent::MessageStop => break,
            _ => {}
        }
    }

    assert_eq!(text, "Hello world");
}

#[tokio::test]
async fn ollama_provider_omits_reasoning_for_non_thinking_models() {
    let response = json!({
        "id": "chatcmpl-ollama-test-2",
        "model": "llama3.2",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Hello from non-thinking Ollama mock"
            }
        }]
    });

    let (base_url, req_rx) = spawn_json_server(response).await;
    let provider = openai_compat_providers::ollama().with_base_url(format!("{}/v1", base_url));

    provider
        .create_message(base_request("llama3.2"))
        .await
        .expect("ollama response parsed");

    let raw = req_rx.await.expect("captured request");
    let body = raw.split("\r\n\r\n").nth(1).expect("request body present");
    let body_json: Value = serde_json::from_str(body).expect("json request body");

    assert_eq!(body_json["model"], json!("llama3.2"));
    assert!(body_json.get("reasoning_effort").is_none());
    assert!(body_json.get("reasoning").is_none());
}

#[tokio::test]
async fn openai_codex_oauth_is_stateless_and_does_not_use_api_key_format() {
    // NOTE: Codex provider talks to a different backend; we only verify request shaping here.
    let response = json!({
        "choices": [{
            "message": { "content": "Hello from Codex mock" },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2
        }
    });

    let (endpoint, req_rx) = spawn_json_server(response).await;
    let provider = OpenAiCodexProvider::new("test-oauth-token".to_string())
        .with_endpoint(endpoint)
        .with_skip_disk(true);

    provider
        .create_message(base_request("gpt-5.2-codex"))
        .await
        .expect("codex response parsed");

    let raw = req_rx.await.expect("captured request");
    assert!(
        raw.to_ascii_lowercase()
            .contains("content-type: application/json"),
        "expected content-type header"
    );

    // Header should be present even if account id is not extractable from a fake token.
    assert!(
        raw.to_ascii_lowercase().contains("authorization: bearer"),
        "expected bearer auth header"
    );
    assert!(
        raw.to_ascii_lowercase()
            .contains("accept: text/event-stream"),
        "expected accept: text/event-stream"
    );
    assert!(
        raw.to_ascii_lowercase()
            .contains("openai-beta: responses=experimental"),
        "expected OpenAI-Beta header for responses"
    );
    assert!(
        raw.to_ascii_lowercase()
            .contains("originator: codex_cli_rs"),
        "expected originator header"
    );
    assert!(
        !raw.to_ascii_lowercase().contains("session_id:"),
        "did not expect session_id header without promptCacheKey"
    );
    assert!(
        !raw.to_ascii_lowercase().contains("conversation_id:"),
        "did not expect conversation_id header without promptCacheKey"
    );

    let body = raw.split("\r\n\r\n").nth(1).expect("request body present");
    let body_json: Value = serde_json::from_str(body).expect("json request body");

    // Enforce stateless operation (`store:false`) for the ChatGPT/Codex backend.
    assert_eq!(body_json["store"], json!(false));

    // Codex backend expects a Responses-style payload with `input`, not chat-completions `messages`.
    assert!(body_json.get("messages").is_none());
    assert!(body_json["input"].is_array());
}

#[tokio::test]
async fn openai_codex_includes_conversation_and_session_headers_with_prompt_cache_key() {
    let response = json!({
        "output": [{
            "content": [{ "type": "output_text", "text": "ok" }]
        }],
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    });

    let (endpoint, req_rx) = spawn_json_server(response).await;
    let mut req = base_request("gpt-5.2-codex");
    req.provider_options = json!({ "promptCacheKey": "conv_abc123" });

    let provider = OpenAiCodexProvider::new("test-oauth-token".to_string())
        .with_endpoint(endpoint)
        .with_skip_disk(true);

    provider
        .create_message(req)
        .await
        .expect("codex response parsed");

    let raw = req_rx.await.expect("captured request");
    let lower = raw.to_ascii_lowercase();
    assert!(lower.contains("conversation_id: conv_abc123"));
    assert!(lower.contains("session_id:"));
}

#[tokio::test]
async fn openai_codex_includes_chatgpt_account_id_header_when_present() {
    let response = json!({
        "output": [{
            "content": [{ "type": "output_text", "text": "ok" }]
        }],
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    });

    let (endpoint, req_rx) = spawn_json_server(response).await;

    // Minimal JWT with the plugin-style claim:
    // payload: {"https://api.openai.com/auth":{"chatgpt_account_id":"acct_123"}}
    let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(br#"{"https://api.openai.com/auth":{"chatgpt_account_id":"acct_123"}}"#);
    let fake_token = format!("{}.{}.{}", "hdr", payload, "sig");

    let provider = OpenAiCodexProvider::new(fake_token)
        .with_endpoint(endpoint)
        .with_skip_disk(true);
    provider
        .create_message(base_request("gpt-5.2-codex"))
        .await
        .expect("codex response parsed");

    let raw = req_rx.await.expect("captured request");
    let lower = raw.to_ascii_lowercase();
    assert!(
        lower.contains("chatgpt-account-id: acct_123"),
        "expected chatgpt-account-id header with extracted account id"
    );
}

#[tokio::test]
async fn openai_codex_oauth_serializes_image_inputs() {
    let response = json!({
        "output": [{
            "content": [{ "type": "output_text", "text": "ok" }]
        }],
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    });

    let (endpoint, req_rx) = spawn_json_server(response).await;
    let mut req = base_request("gpt-5.2-codex");
    req.messages = vec![Message::user_blocks(vec![
        ContentBlock::Image {
            source: ImageSource {
                source_type: "base64".to_string(),
                media_type: Some("image/jpeg".to_string()),
                data: Some(base64::engine::general_purpose::STANDARD.encode(b"hello-image")),
                url: None,
            },
        },
        ContentBlock::Text {
            text: "Describe this image".to_string(),
        },
    ])];

    let provider = OpenAiCodexProvider::new("test-oauth-token".to_string())
        .with_endpoint(endpoint)
        .with_skip_disk(true);

    provider
        .create_message(req)
        .await
        .expect("codex response parsed");

    let raw = req_rx.await.expect("captured request");
    let body = raw.split("\r\n\r\n").nth(1).expect("request body present");
    let body_json: Value = serde_json::from_str(body).expect("json request body");

    assert_eq!(body_json["input"][0]["role"], json!("user"));
    assert_eq!(
        body_json["input"][0]["content"][0]["type"],
        json!("input_image")
    );
    assert_eq!(
        body_json["input"][0]["content"][0]["image_url"],
        json!("data:image/jpeg;base64,aGVsbG8taW1hZ2U=")
    );
    assert_eq!(
        body_json["input"][0]["content"][1]["type"],
        json!("input_text")
    );
    assert_eq!(
        body_json["input"][0]["content"][1]["text"],
        json!("Describe this image")
    );
}

#[tokio::test]
async fn ollama_stream_extracts_inline_think_tags_as_reasoning() {
    let response_body = concat!(
        "data: {\"id\":\"o1\",\"model\":\"qwen3\",\"choices\":[{\"delta\":{\"content\":\"<think>secret\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\" reasoning</think>visible \"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let (base_url, _req_rx) = spawn_sse_server(response_body.to_string()).await;
    let provider = openai_compat_providers::ollama().with_base_url(format!("{}/v1", base_url));
    let mut stream = provider
        .create_message_stream(base_request("qwen3:8b"))
        .await
        .expect("create stream");

    let mut visible = String::new();
    let mut reasoning = String::new();
    while let Some(event) = stream.next().await {
        match event.expect("stream item") {
            mangocode_api::StreamEvent::TextDelta { text, .. } => visible.push_str(&text),
            mangocode_api::StreamEvent::ReasoningDelta {
                reasoning: r, ..
            } => reasoning.push_str(&r),
            mangocode_api::StreamEvent::MessageStop => break,
            _ => {}
        }
    }

    assert!(
        reasoning.contains("secret reasoning"),
        "reasoning should contain stripped <think> contents, got {:?}",
        reasoning
    );
    assert!(
        visible.contains("visible") && visible.contains("answer"),
        "visible text should contain post-</think> content, got {:?}",
        visible
    );
    assert!(
        !visible.contains("<think>") && !visible.contains("</think>"),
        "raw think tags must not be forwarded to visible text, got {:?}",
        visible
    );
}

#[tokio::test]
async fn ollama_stream_emits_message_start_before_first_chunk_parsed() {
    // A stream that delivers only an SSE comment then a normal chunk; we
    // assert that MessageStart is the very first event the consumer observes.
    let response_body = concat!(
        ": keep-alive\n\n",
        "data: {\"id\":\"o1\",\"model\":\"qwen3\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let (base_url, _req_rx) = spawn_sse_server(response_body.to_string()).await;
    let provider = openai_compat_providers::ollama().with_base_url(format!("{}/v1", base_url));
    let mut stream = provider
        .create_message_stream(base_request("qwen3:8b"))
        .await
        .expect("create stream");

    let first = stream.next().await.expect("first event").expect("first ok");
    assert!(
        matches!(first, mangocode_api::StreamEvent::MessageStart { .. }),
        "expected MessageStart as first event, got {:?}",
        first
    );
}

#[tokio::test]
async fn ollama_stream_split_think_tag_across_chunks_is_handled() {
    // `<think>` is split across SSE chunks at byte level — exercises the
    // splitter's pending-buffer behaviour.
    let response_body = concat!(
        "data: {\"id\":\"o1\",\"choices\":[{\"delta\":{\"content\":\"<thi\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"nk>r1</think>v1\"}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let (base_url, _req_rx) = spawn_sse_server(response_body.to_string()).await;
    let provider = openai_compat_providers::ollama().with_base_url(format!("{}/v1", base_url));
    let mut stream = provider
        .create_message_stream(base_request("qwen3:8b"))
        .await
        .expect("create stream");

    let mut visible = String::new();
    let mut reasoning = String::new();
    while let Some(event) = stream.next().await {
        match event.expect("stream item") {
            mangocode_api::StreamEvent::TextDelta { text, .. } => visible.push_str(&text),
            mangocode_api::StreamEvent::ReasoningDelta { reasoning: r, .. } => {
                reasoning.push_str(&r)
            }
            mangocode_api::StreamEvent::MessageStop => break,
            _ => {}
        }
    }
    assert_eq!(reasoning, "r1");
    assert_eq!(visible, "v1");
}

#[tokio::test]
async fn ollama_drops_tools_for_unknown_model_to_avoid_stall() {
    let response = json!({
        "id": "chatcmpl-tool-gate",
        "model": "tinyllama",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": { "role": "assistant", "content": "ok" }
        }]
    });
    let (base_url, req_rx) = spawn_json_server(response).await;
    let provider = openai_compat_providers::ollama().with_base_url(format!("{}/v1", base_url));

    let mut req = base_request("tinyllama");
    req.tools = vec![mangocode_core::types::ToolDefinition {
        name: "get_weather".to_string(),
        description: "fetch weather".to_string(),
        input_schema: json!({"type":"object","properties":{}}),
    }];

    provider
        .create_message(req)
        .await
        .expect("ollama response parsed");

    let raw = req_rx.await.expect("captured request");
    let body = raw.split("\r\n\r\n").nth(1).expect("request body present");
    let body_json: Value = serde_json::from_str(body).expect("json request body");
    assert!(
        body_json.get("tools").is_none(),
        "tools should be dropped for unknown ollama model: {:?}",
        body_json.get("tools")
    );
}

#[tokio::test]
async fn ollama_keeps_tools_for_known_model() {
    let response = json!({
        "id": "chatcmpl-tool-gate",
        "model": "llama3.2",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": { "role": "assistant", "content": "ok" }
        }]
    });
    let (base_url, req_rx) = spawn_json_server(response).await;
    let provider = openai_compat_providers::ollama().with_base_url(format!("{}/v1", base_url));

    let mut req = base_request("llama3.2");
    req.tools = vec![mangocode_core::types::ToolDefinition {
        name: "get_weather".to_string(),
        description: "fetch weather".to_string(),
        input_schema: json!({"type":"object","properties":{}}),
    }];

    provider
        .create_message(req)
        .await
        .expect("ollama response parsed");

    let raw = req_rx.await.expect("captured request");
    let body = raw.split("\r\n\r\n").nth(1).expect("request body present");
    let body_json: Value = serde_json::from_str(body).expect("json request body");
    assert!(
        body_json.get("tools").is_some(),
        "tools must still be sent to known tool-capable ollama model"
    );
}
