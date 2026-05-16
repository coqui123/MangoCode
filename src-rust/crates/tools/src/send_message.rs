// SendMessageTool: compatibility wrapper for agent/session messaging.
//
// SQLite coordination is the canonical cross-process delivery path. The
// in-process inbox below remains only for older local callers that directly use
// drain_inbox() / peek_inbox().

use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// In-process inbox
// ---------------------------------------------------------------------------

/// A single message in the inbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    pub from: String,
    pub to: String,
    pub content: String,
    pub timestamp: u64,
}

/// Global inbox: recipient_id -> queued messages.
static INBOX: Lazy<DashMap<String, Vec<AgentMessage>>> = Lazy::new(DashMap::new);

/// Remove and return all messages queued for `recipient`.
pub fn drain_inbox(recipient: &str) -> Vec<AgentMessage> {
    INBOX.remove(recipient).map(|(_, v)| v).unwrap_or_default()
}

/// Read (without removing) all messages queued for `recipient`.
pub fn peek_inbox(recipient: &str) -> Vec<AgentMessage> {
    INBOX.get(recipient).map(|v| v.clone()).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

pub struct SendMessageTool;

#[derive(Debug, Deserialize)]
struct SendMessageInput {
    /// Recipient actor target, or "*" for broadcast.
    to: String,
    /// Message body.
    message: String,
    /// Short preview text shown in the UI.
    #[serde(default)]
    summary: Option<String>,
}

#[async_trait]
impl Tool for SendMessageTool {
    fn name(&self) -> &str {
        "SendMessage"
    }

    fn description(&self) -> &str {
        "Compatibility wrapper for sending a local coordination message. Use to=\"*\" to broadcast \
         to active MangoCode actors in this repo, or provide an exact/unique actor id or title for \
         a direct message. Recipients read messages through CoordinationInbox."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::None
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient actor id, unique actor-id prefix, or title. Use \"*\" for a repo broadcast to active local MangoCode actors."
                },
                "message": {
                    "type": "string",
                    "description": "Message content"
                },
                "summary": {
                    "type": "string",
                    "description": "5-10 word preview for the UI (optional)"
                }
            },
            "required": ["to", "message"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: SendMessageInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        if params.message.is_empty() {
            return ToolResult::error("Message cannot be empty.".to_string());
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let msg = AgentMessage {
            from: ctx.coordination_actor_id(),
            to: params.to.clone(),
            content: params.message.clone(),
            timestamp: now,
        };

        let preview = params.summary.as_deref().unwrap_or_else(|| {
            let s = params.message.as_str();
            mangocode_core::truncate::truncate_bytes_prefix(s, 60)
        });

        if params.to == "*" {
            if let Err(result) =
                crate::coordination::send_coordination_message(ctx, &params.message, None, true)
            {
                return result;
            }
            // Broadcast: deliver to every existing inbox key
            let recipients: Vec<String> = INBOX.iter().map(|e| e.key().clone()).collect();

            if recipients.is_empty() {
                return ToolResult::success(
                    "Broadcast queued for active local MangoCode actors in this repo.".to_string(),
                );
            }

            for key in &recipients {
                INBOX.entry(key.clone()).or_default().push(msg.clone());
            }

            return ToolResult::success(format!(
                "Broadcast queued for active local MangoCode actors; also delivered to {} legacy in-process recipient(s): {}",
                recipients.len(),
                preview
            ));
        }

        let message = match crate::coordination::send_coordination_message(
            ctx,
            &params.message,
            Some(&params.to),
            false,
        ) {
            Ok(message) => message,
            Err(result) => return result,
        };
        // Directed message compatibility path after canonical delivery accepts it.
        let legacy_recipient = message.to_session_id.unwrap_or_else(|| params.to.clone());
        let mut legacy_msg = msg;
        legacy_msg.to = legacy_recipient.clone();
        INBOX.entry(legacy_recipient).or_default().push(legacy_msg);

        ToolResult::success(format!("Message sent to '{}': {}", params.to, preview))
    }
}
