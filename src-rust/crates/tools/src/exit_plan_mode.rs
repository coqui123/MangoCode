// ExitPlanMode tool: leave planning mode and return to normal execution.

use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::debug;

pub struct ExitPlanModeTool;

#[derive(Debug, Deserialize)]
struct ExitPlanModeInput {
    #[serde(default)]
    summary: Option<String>,
}

#[async_trait]
impl Tool for ExitPlanModeTool {
    fn name(&self) -> &str {
        mangocode_core::constants::TOOL_NAME_EXIT_PLAN_MODE
    }

    fn description(&self) -> &str {
        "Exit plan mode and return to normal execution mode where runtime-visible \
         tools are available subject to feature flags, session config, and permissions. \
         Optionally provide a summary of the plan."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::None
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Summary of the plan you developed"
                }
            },
            "required": []
        })
    }

    async fn execute(&self, input: Value, _ctx: &ToolContext) -> ToolResult {
        let params = match parse_input(input) {
            Ok(params) => params,
            Err(err) => return ToolResult::error(err),
        };

        debug!(summary = ?params.summary, "Exiting plan mode");

        let msg = if let Some(summary) = &params.summary {
            format!("Exited plan mode. Plan summary: {}", summary)
        } else {
            "Exited plan mode. Runtime-visible tools are now available subject to permissions."
                .to_string()
        };

        ToolResult::success(msg).with_metadata(json!({
            "type": "exit_plan_mode",
            "summary": params.summary,
        }))
    }
}

fn parse_input(input: Value) -> Result<ExitPlanModeInput, String> {
    serde_json::from_value(input).map_err(|err| format!("Invalid input: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_input_rejects_malformed_summary_type() {
        let err = parse_input(json!({ "summary": 42 })).unwrap_err();

        assert!(err.contains("Invalid input"));
    }

    #[test]
    fn parse_input_accepts_empty_object() {
        let parsed = parse_input(json!({})).unwrap();

        assert_eq!(parsed.summary, None);
    }
}
