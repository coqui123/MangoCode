//! Provider serialization/deserialization integration tests.

use mangocode_api::provider::LlmProvider;
use mangocode_api::provider_types::{ProviderRequest, StopReason, SystemPrompt};
use mangocode_api::{GoogleProvider, OpenAiProvider, ThinkingConfig};
use mangocode_core::types::{ContentBlock, Message};
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
