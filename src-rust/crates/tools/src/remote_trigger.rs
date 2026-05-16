//! RemoteTriggerTool — cross-session event dispatch.

use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use mangocode_core::truncate::truncate_bytes_prefix;
use serde::Deserialize;
use serde_json::{json, Value};

/// Input schema for RemoteTriggerTool.
#[derive(Debug, Deserialize)]
struct RemoteTriggerInput {
    /// Target session ID to send the event to.
    session_id: String,
    /// Event name (arbitrary string).
    event_name: String,
    /// JSON payload to deliver.
    #[serde(default)]
    payload: Value,
}

/// Delivers cross-session trigger events via the Claude.ai API.
pub struct RemoteTriggerTool;

#[async_trait]
impl Tool for RemoteTriggerTool {
    fn name(&self) -> &str {
        "RemoteTrigger"
    }

    fn description(&self) -> &str {
        "Send a named event to another active MangoCode session. \
         Use this to coordinate across parallel sessions or notify a parent session of results."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Network
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The target session ID to trigger"
                },
                "event_name": {
                    "type": "string",
                    "description": "Name of the event to send (e.g., 'task_complete', 'result_ready')"
                },
                "payload": {
                    "type": "object",
                    "description": "Optional JSON payload to deliver with the event",
                    "additionalProperties": true
                }
            },
            "required": ["session_id", "event_name"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: RemoteTriggerInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {e}")),
        };
        let target_session_id = params.session_id.trim().to_string();
        if target_session_id.is_empty() {
            return ToolResult::error("session_id cannot be empty.");
        }

        // Auth token is not available via a sync helper; pass empty string.
        // A future implementation can wire in a proper OAuth token retrieval.
        let token = String::new();

        let client = reqwest::Client::new();
        let url = match remote_trigger_url(&target_session_id) {
            Ok(url) => url,
            Err(e) => return ToolResult::error(e),
        };

        if let Err(e) = ctx.check_permission(self.name(), &format!("POST {}", url), false) {
            return ToolResult::error(e.to_string());
        }

        let body = json!({
            "event_name": params.event_name,
            "payload": params.payload,
            "source_session_id": ctx.session_id,
        });

        let mut request = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body);

        if !token.is_empty() {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let resp = match request.send().await {
            Ok(r) => r,
            Err(e) => return ToolResult::error(format!("HTTP error: {e}")),
        };

        let target_prefix = short_session_id(&target_session_id);

        if resp.status().is_success() {
            match resp.json::<Value>().await {
                Ok(data) => {
                    let delivered = data["delivered"].as_bool().unwrap_or(false);
                    let status = data["session_status"].as_str().unwrap_or("unknown");
                    ToolResult::success(format!(
                        "Event '{}' {} to session {} (status: {})",
                        params.event_name,
                        if delivered { "delivered" } else { "queued" },
                        target_prefix,
                        status,
                    ))
                }
                Err(_) => ToolResult::success(format!(
                    "Event '{}' sent to session {}",
                    params.event_name, target_prefix,
                )),
            }
        } else {
            ToolResult::error(format!(
                "Trigger failed: HTTP {} — is session {} active?",
                resp.status(),
                target_prefix,
            ))
        }
    }
}

fn remote_trigger_url(session_id: &str) -> Result<String, String> {
    let mut url = reqwest::Url::parse("https://api.claude.ai/")
        .map_err(|e| format!("Failed to build trigger URL: {e}"))?;
    url.path_segments_mut()
        .map_err(|_| "Failed to build trigger URL path.".to_string())?
        .push("api")
        .push("sessions")
        .push(session_id)
        .push("trigger");
    Ok(url.to_string())
}

fn short_session_id(session_id: &str) -> &str {
    truncate_bytes_prefix(session_id, 8)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_denying_context() -> ToolContext {
        ToolContext {
            working_dir: std::path::PathBuf::from("/workspace"),
            permission_mode: mangocode_core::config::PermissionMode::Default,
            permission_handler: std::sync::Arc::new(
                mangocode_core::permissions::AutoPermissionHandler {
                    mode: mangocode_core::config::PermissionMode::Default,
                },
            ),
            cost_tracker: mangocode_core::cost::CostTracker::new(),
            session_metrics: None,
            session_id: "remote-trigger-test".to_string(),
            coordination_actor_id: None,
            coordination_parent_actor_id: None,
            inject_coordination_inbox: true,
            file_history: std::sync::Arc::new(parking_lot::Mutex::new(
                mangocode_core::file_history::FileHistory::new(),
            )),
            current_turn: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            non_interactive: true,
            mcp_manager: None,
            config: mangocode_core::config::Config::default(),
        }
    }

    #[tokio::test]
    async fn remote_trigger_requires_network_permission_before_http() {
        let result = RemoteTriggerTool
            .execute(
                json!({
                    "session_id": "session-123",
                    "event_name": "result_ready",
                    "payload": {},
                }),
                &default_denying_context(),
            )
            .await;

        assert!(result.is_error);
        assert!(result.content.contains("Permission denied"));
    }

    #[test]
    fn remote_trigger_url_encodes_session_id_path_segment() {
        let url = remote_trigger_url("session/with?control#chars").unwrap();

        assert_eq!(
            url,
            "https://api.claude.ai/api/sessions/session%2Fwith%3Fcontrol%23chars/trigger"
        );
    }

    #[test]
    fn short_session_id_handles_multibyte_boundaries() {
        assert_eq!(short_session_id("☃☃☃session"), "☃☃");
    }
}
