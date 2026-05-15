// update_plan tool: transcript-facing checklist display.

#![cfg_attr(not(feature = "tool-update-plan"), allow(dead_code, unused_imports))]

use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::debug;

#[cfg(feature = "tool-update-plan")]
pub struct UpdatePlanTool;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlanStatus {
    Pending,
    InProgress,
    Completed,
}

impl PlanStatus {
    fn from_str_ci(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "pending" => Ok(Self::Pending),
            "in_progress" => Ok(Self::InProgress),
            "completed" => Ok(Self::Completed),
            other => Err(format!(
                "Invalid status {:?}: must be one of \"pending\", \"in_progress\", or \"completed\".",
                other
            )),
        }
    }
}

impl<'de> Deserialize<'de> for PlanStatus {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Self::from_str_ci(&s).map_err(serde::de::Error::custom)
    }
}

impl std::fmt::Display for PlanStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => f.write_str("pending"),
            Self::InProgress => f.write_str("in_progress"),
            Self::Completed => f.write_str("completed"),
        }
    }
}

#[derive(Debug, Deserialize)]
struct UpdatePlanInput {
    #[serde(default)]
    explanation: Option<String>,
    plan: Vec<PlanItem>,
}

#[derive(Debug, Clone, Deserialize)]
struct PlanItem {
    step: String,
    status: PlanStatus,
}

#[cfg(feature = "tool-update-plan")]
#[async_trait]
impl Tool for UpdatePlanTool {
    fn name(&self) -> &str {
        mangocode_core::constants::TOOL_NAME_UPDATE_PLAN
    }

    fn description(&self) -> &str {
        "Update the session plan checklist. Provide the full current plan with \
         each step status set to pending, in_progress, or completed. At most \
         one step may be in_progress. This is a lightweight progress display, \
         not plan mode."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::None
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": "Optional one- or two-sentence note explaining this plan update"
                },
                "plan": {
                    "type": "array",
                    "description": "The complete current plan checklist. At most one item may be in_progress.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": { "type": "string" },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"]
                            }
                        },
                        "required": ["step", "status"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["plan"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: UpdatePlanInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        debug!(count = params.plan.len(), "Updating session plan");

        let in_progress_count = params
            .plan
            .iter()
            .filter(|item| item.status == PlanStatus::InProgress)
            .count();
        if in_progress_count > 1 {
            return ToolResult::error(
                "Invalid plan: at most one step can be in_progress at a time.".to_string(),
            );
        }
        if params.plan.iter().any(|item| item.step.trim().is_empty()) {
            return ToolResult::error("Invalid plan: step text cannot be empty.".to_string());
        }

        let explanation = params
            .explanation
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);

        let normalized_plan: Vec<(String, PlanStatus)> = params
            .plan
            .iter()
            .map(|item| (item.step.trim().to_string(), item.status))
            .collect();

        let plan_json: Vec<Value> = normalized_plan
            .iter()
            .map(|(step, status)| {
                json!({
                    "step": step,
                    "status": status.to_string(),
                })
            })
            .collect();

        // Keep the existing todo/progress surface in sync without requiring the
        // model to use TodoWrite's separate id/priority schema.
        let todos_json: Vec<Value> = normalized_plan
            .iter()
            .enumerate()
            .map(|(idx, (step, status))| {
                json!({
                    "id": format!("plan-{}", idx + 1),
                    "content": step,
                    "status": status.to_string(),
                })
            })
            .collect();
        crate::todo_write::save_todos(&ctx.session_id, &todos_json);

        ToolResult::success("Plan updated").with_metadata(json!({
            "transcript_display": {
                "kind": "updated_plan",
                "explanation": explanation,
                "plan": plan_json,
            },
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_parsing_is_case_insensitive() {
        assert_eq!(
            PlanStatus::from_str_ci("IN_PROGRESS").unwrap(),
            PlanStatus::InProgress
        );
    }

    #[cfg(feature = "tool-update-plan")]
    #[tokio::test]
    async fn execute_returns_plan_updated_metadata_and_syncs_todos() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::Arc;

        let dir = tempfile::tempdir().unwrap();
        let session_id = format!(
            "update-plan-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        let ctx = ToolContext {
            working_dir: dir.path().to_path_buf(),
            permission_mode: mangocode_core::config::PermissionMode::BypassPermissions,
            permission_handler: Arc::new(mangocode_core::permissions::AutoPermissionHandler {
                mode: mangocode_core::config::PermissionMode::BypassPermissions,
            }),
            cost_tracker: mangocode_core::cost::CostTracker::new(),
            session_metrics: None,
            session_id: session_id.clone(),
            file_history: Arc::new(parking_lot::Mutex::new(
                mangocode_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(AtomicUsize::new(1)),
            non_interactive: true,
            mcp_manager: None,
            config: mangocode_core::config::Config::default(),
        };

        let result = UpdatePlanTool
            .execute(
                json!({
                    "explanation": "Testing the plan display.",
                    "plan": [
                        { "step": "  One  ", "status": "completed" },
                        { "step": "Two", "status": "pending" }
                    ]
                }),
                &ctx,
            )
            .await;

        assert!(!result.is_error, "{}", result.content);
        assert_eq!(result.content, "Plan updated");
        let display = &result.metadata.as_ref().unwrap()["transcript_display"];
        assert_eq!(display["kind"].as_str(), Some("updated_plan"));
        assert_eq!(display["plan"].as_array().unwrap().len(), 2);
        assert_eq!(display["plan"][0]["step"].as_str(), Some("One"));

        let todos = crate::todo_write::load_todos(&session_id);
        assert_eq!(todos.len(), 2);
        assert_eq!(todos[0]["content"].as_str(), Some("One"));
        let _ = std::fs::remove_file(crate::todo_write::todos_path(&session_id));
    }

    #[cfg(feature = "tool-update-plan")]
    #[tokio::test]
    async fn execute_rejects_multiple_in_progress_steps() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::Arc;

        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolContext {
            working_dir: dir.path().to_path_buf(),
            permission_mode: mangocode_core::config::PermissionMode::BypassPermissions,
            permission_handler: Arc::new(mangocode_core::permissions::AutoPermissionHandler {
                mode: mangocode_core::config::PermissionMode::BypassPermissions,
            }),
            cost_tracker: mangocode_core::cost::CostTracker::new(),
            session_metrics: None,
            session_id: "update-plan-multiple-in-progress-test".to_string(),
            file_history: Arc::new(parking_lot::Mutex::new(
                mangocode_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(AtomicUsize::new(1)),
            non_interactive: true,
            mcp_manager: None,
            config: mangocode_core::config::Config::default(),
        };

        let result = UpdatePlanTool
            .execute(
                json!({
                    "plan": [
                        { "step": "One", "status": "in_progress" },
                        { "step": "Two", "status": "in_progress" }
                    ]
                }),
                &ctx,
            )
            .await;

        assert!(result.is_error);
        assert!(result.content.contains("at most one"));
    }
}
