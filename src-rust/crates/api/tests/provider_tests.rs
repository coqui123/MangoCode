//! Provider serialization/deserialization integration tests.

use mangocode_api::provider::LlmProvider;
use mangocode_api::provider_types::{ProviderRequest, StopReason, SystemPrompt};
use mangocode_api::{GoogleProvider, OpenAiProvider, ThinkingConfig};
use mangocode_api::providers::OpenAiCodexProvider;
use mangocode_core::types::{ContentBlock, Message};
use base64::Engine;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

async fn spawn_json_server(response_body: Value) -> (String, oneshot::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind test server");
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
        let header_end = request
            .find("\r\n\r\n")
            .expect("header terminator");
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
            let read = stream
                .read(&mut buf[n..])
                .await
                .expect("read request body");
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
    let body = raw
        .split("\r\n\r\n")
        .nth(1)
        .expect("request body present");
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
    assert!(request_line.contains("/v1beta/models/gemini-2.5-flash:generateContent?key=google-test-key"));

    let body = raw
        .split("\r\n\r\n")
        .nth(1)
        .expect("request body present");
    let body_json: Value = serde_json::from_str(body).expect("json request body");

    assert_eq!(body_json["contents"][0]["role"], json!("user"));
    assert_eq!(body_json["contents"][0]["parts"][0]["text"], json!("Say hello"));
    assert_eq!(parsed.stop_reason, StopReason::EndTurn);
    assert!(matches!(
        &parsed.content[0],
        ContentBlock::Text { text } if text.contains("Google mock")
    ));
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
        raw.to_ascii_lowercase().contains("content-type: application/json"),
        "expected content-type header"
    );

    // Header should be present even if account id is not extractable from a fake token.
    assert!(
        raw.to_ascii_lowercase().contains("authorization: bearer"),
        "expected bearer auth header"
    );
    assert!(
        raw.to_ascii_lowercase().contains("accept: text/event-stream"),
        "expected accept: text/event-stream"
    );
    assert!(
        raw.to_ascii_lowercase().contains("openai-beta: responses=experimental"),
        "expected OpenAI-Beta header for responses"
    );
    assert!(
        raw.to_ascii_lowercase().contains("originator: codex_cli_rs"),
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

    let body = raw
        .split("\r\n\r\n")
        .nth(1)
        .expect("request body present");
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

    provider.create_message(req).await.expect("codex response parsed");

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
    let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(
        br#"{"https://api.openai.com/auth":{"chatgpt_account_id":"acct_123"}}"#,
    );
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
