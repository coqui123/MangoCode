//! Cloud session API.
//!
//! Converts between internal Message types and the cloud API format.
//! Provides CRUD operations for cloud-hosted sessions.

use crate::types::{ContentBlock, Message, MessageContent, Role};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::warn;

async fn response_error_body(response: reqwest::Response, context: &str) -> String {
    response
        .text()
        .await
        .unwrap_or_else(|err| format!("<failed to read {context} error response body: {err}>"))
}

fn format_http_error(prefix: &str, status: reqwest::StatusCode, body: String) -> String {
    let body = body.trim();
    if body.is_empty() {
        format!("{prefix} {status}")
    } else {
        format!("{prefix} {status}: {body}")
    }
}

// ---------------------------------------------------------------------------
// Cloud session API types
// ---------------------------------------------------------------------------

/// Options for creating a new cloud session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudSessionCreateOpts {
    pub project_root: Option<String>,
    pub model: String,
    pub title: Option<String>,
}

/// A cloud session detail (with full message list).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudSessionDetail {
    pub id: String,
    pub title: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
    pub messages: Vec<CloudMessage>,
}

/// A message in the cloud API format.
///
/// `content` is a JSON array of Anthropic-style content-block objects plus
/// MangoCode transcript extensions so that structured blocks (tool_use,
/// tool_result, image, rich transcript metadata, …) survive a round-trip
/// through the cloud without being collapsed to plain text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudMessage {
    pub id: String,
    pub role: String,        // "user" | "assistant"
    pub content: Vec<Value>, // Array of Anthropic-schema content block objects
    pub created_at: u64,
    pub session_id: String,
}

// ---------------------------------------------------------------------------
// SDK message adapter
// ---------------------------------------------------------------------------

/// Normalise a `MessageContent` into a flat `Vec<ContentBlock>`.
///
/// A `MessageContent::Text` shorthand is lifted into a single
/// `ContentBlock::Text` so every path produces the same block list.
fn content_to_blocks(content: &MessageContent) -> Vec<ContentBlock> {
    match content {
        MessageContent::Text(t) => vec![ContentBlock::Text { text: t.clone() }],
        MessageContent::Blocks(blocks) => blocks.clone(),
    }
}

/// Convert an internal `Message` to a `CloudMessage`.
///
/// Every `ContentBlock` is serialised to its transcript JSON representation;
/// no information is discarded. Use `message_to_external_value` for provider
/// or share payloads where transcript-only metadata must be stripped.
pub fn message_to_cloud(msg: &Message, session_id: &str, msg_id: &str, ts: u64) -> CloudMessage {
    let role = match msg.role {
        Role::User => "user".to_string(),
        Role::Assistant => "assistant".to_string(),
    };

    let content: Vec<Value> = content_to_blocks(&msg.content)
        .into_iter()
        .map(|block| serde_json::to_value(&block).unwrap_or(Value::Null))
        .collect();

    CloudMessage {
        id: msg_id.to_string(),
        role,
        content,
        created_at: ts,
        session_id: session_id.to_string(),
    }
}

/// Convert a `CloudMessage` back to an internal `Message`.
///
/// Each element of `content` is deserialised as a `ContentBlock`. Elements
/// that cannot be parsed are preserved as text so unknown future block types do
/// not crash older clients or disappear from the transcript.
pub fn cloud_to_message(cloud: &CloudMessage) -> Message {
    let role = if cloud.role == "assistant" {
        Role::Assistant
    } else {
        Role::User
    };

    let blocks: Vec<ContentBlock> = cloud
        .content
        .iter()
        .enumerate()
        .map(
            |(index, value)| match serde_json::from_value::<ContentBlock>(value.clone()) {
                Ok(block) => block,
                Err(err) => unsupported_cloud_content_block(index, value, &err),
            },
        )
        .collect();

    // Use the compact Text shorthand when there is exactly one plain-text block.
    let content = if blocks.len() == 1 {
        if let ContentBlock::Text { text } = &blocks[0] {
            MessageContent::Text(text.clone())
        } else {
            MessageContent::Blocks(blocks)
        }
    } else {
        MessageContent::Blocks(blocks)
    };

    Message {
        role,
        content,
        uuid: None,
        cost: None,
    }
}

fn unsupported_cloud_content_block(
    index: usize,
    value: &Value,
    err: &serde_json::Error,
) -> ContentBlock {
    warn!(
        index,
        error = %err,
        "failed to parse cloud session content block; preserving raw JSON as text"
    );
    ContentBlock::Text {
        text: format!("[Unsupported cloud content block {index}: {value}]"),
    }
}

// ---------------------------------------------------------------------------
// Cloud session API client
// ---------------------------------------------------------------------------

/// Thin client for the cloud session REST API.
pub struct CloudSessionClient {
    base_url: String,
    access_token: String,
    http: reqwest::Client,
}

impl CloudSessionClient {
    pub fn new(access_token: String) -> Self {
        Self {
            base_url: "https://api.claude.ai".to_string(),
            access_token,
            http: reqwest::Client::new(),
        }
    }

    /// List all cloud sessions.
    pub async fn list(&self) -> Result<Vec<crate::remote_session::CloudSession>, String> {
        let resp = self
            .http
            .get(format!("{}/api/sessions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.access_token))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        resp.json().await.map_err(|e| e.to_string())
    }

    /// Fetch full session details including messages.
    pub async fn fetch(&self, session_id: &str) -> Result<CloudSessionDetail, String> {
        let resp = self
            .http
            .get(format!("{}/api/sessions/{}", self.base_url, session_id))
            .header("Authorization", format!("Bearer {}", self.access_token))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        resp.json().await.map_err(|e| e.to_string())
    }

    /// Push new messages to a cloud session.
    pub async fn push_messages(
        &self,
        session_id: &str,
        messages: &[CloudMessage],
    ) -> Result<(), String> {
        let resp = self
            .http
            .post(format!(
                "{}/api/sessions/{}/messages",
                self.base_url, session_id
            ))
            .header("Authorization", format!("Bearer {}", self.access_token))
            .json(messages)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = response_error_body(resp, "cloud session push").await;
            return Err(format_http_error("HTTP", status, body));
        }
        Ok(())
    }

    /// Create a new cloud session.
    pub async fn create(
        &self,
        opts: CloudSessionCreateOpts,
    ) -> Result<crate::remote_session::CloudSession, String> {
        let resp = self
            .http
            .post(format!("{}/api/sessions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.access_token))
            .json(&opts)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = response_error_body(resp, "cloud session create").await;
            return Err(format_http_error("HTTP", status, body));
        }
        resp.json().await.map_err(|e| e.to_string())
    }

    /// Delete a cloud session.
    pub async fn delete(&self, session_id: &str) -> Result<(), String> {
        let resp = self
            .http
            .delete(format!("{}/api/sessions/{}", self.base_url, session_id))
            .header("Authorization", format!("Bearer {}", self.access_token))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = response_error_body(resp, "cloud session delete").await;
            return Err(format_http_error("HTTP", status, body));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cloud_message(content: Vec<Value>) -> CloudMessage {
        CloudMessage {
            id: "msg-1".to_string(),
            role: "assistant".to_string(),
            content,
            created_at: 1,
            session_id: "sess-1".to_string(),
        }
    }

    #[test]
    fn cloud_to_message_preserves_unsupported_single_block_as_text() {
        let msg = cloud_to_message(&cloud_message(vec![json!({
            "type": "future_block",
            "payload": { "answer": 42 }
        })]));

        match msg.content {
            MessageContent::Text(text) => {
                assert!(text.contains("Unsupported cloud content block 0"));
                assert!(text.contains("future_block"));
                assert!(text.contains("\"answer\":42"));
            }
            other => panic!("expected text fallback, got {other:?}"),
        }
    }

    #[test]
    fn cloud_to_message_keeps_supported_blocks_when_one_block_is_unsupported() {
        let msg = cloud_to_message(&cloud_message(vec![
            json!({ "type": "text", "text": "hello" }),
            json!({ "type": "future_block", "payload": "kept" }),
        ]));

        match msg.content {
            MessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert!(matches!(blocks[0], ContentBlock::Text { .. }));
                match &blocks[1] {
                    ContentBlock::Text { text } => {
                        assert!(text.contains("future_block"));
                        assert!(text.contains("kept"));
                    }
                    other => panic!("expected text fallback, got {other:?}"),
                }
            }
            other => panic!("expected block content, got {other:?}"),
        }
    }
}
