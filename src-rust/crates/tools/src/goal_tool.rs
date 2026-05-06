#![cfg_attr(
    not(all(
        feature = "tool-get-goal",
        feature = "tool-create-goal",
        feature = "tool-update-goal"
    )),
    allow(dead_code)
)]

use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use mangocode_core::goals::ThreadGoalStatus;
use mangocode_core::sqlite_storage::SqliteSessionStore;
use mangocode_tool_runtime::ToolCapabilities;
use serde::Deserialize;
use serde_json::{json, Value};

#[cfg(feature = "tool-get-goal")]
pub struct GetGoalTool;
#[cfg(feature = "tool-create-goal")]
pub struct CreateGoalTool;
#[cfg(feature = "tool-update-goal")]
pub struct UpdateGoalTool;

#[derive(Debug, Deserialize)]
struct CreateGoalInput {
    objective: String,
    token_budget: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct UpdateGoalInput {
    status: String,
}

#[cfg(feature = "tool-get-goal")]
#[async_trait]
impl Tool for GetGoalTool {
    fn name(&self) -> &str {
        "get_goal"
    }

    fn description(&self) -> &str {
        "Return the persistent local goal for the current MangoCode session, if one is set."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::None
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        })
    }

    fn capabilities(&self, _input: &Value) -> ToolCapabilities {
        serial_goal_capabilities(false)
    }

    async fn execute(&self, _input: Value, ctx: &ToolContext) -> ToolResult {
        let store = match mangocode_core::goals::open_default_goal_store() {
            Ok(store) => store,
            Err(e) => return ToolResult::error(format!("Failed to open goal store: {}", e)),
        };
        get_goal_with_store(&store, &ctx.session_id)
    }
}

#[cfg(feature = "tool-create-goal")]
#[async_trait]
impl Tool for CreateGoalTool {
    fn name(&self) -> &str {
        "create_goal"
    }

    fn description(&self) -> &str {
        "Create a persistent local goal for this MangoCode session. Use only when the user explicitly asks to create or pursue a goal. Do not infer goals from ordinary tasks. Fails if a goal already exists."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::None
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": "The explicit user-requested goal objective."
                },
                "token_budget": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional positive token budget. Use only if the user explicitly provided one."
                }
            },
            "required": ["objective"],
            "additionalProperties": false
        })
    }

    fn capabilities(&self, _input: &Value) -> ToolCapabilities {
        serial_goal_capabilities(true)
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: CreateGoalInput = match serde_json::from_value(input) {
            Ok(params) => params,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };
        let store = match mangocode_core::goals::open_default_goal_store() {
            Ok(store) => store,
            Err(e) => return ToolResult::error(format!("Failed to open goal store: {}", e)),
        };
        create_goal_with_store(
            &store,
            &ctx.session_id,
            &params.objective,
            params.token_budget,
        )
    }
}

#[cfg(feature = "tool-update-goal")]
#[async_trait]
impl Tool for UpdateGoalTool {
    fn name(&self) -> &str {
        "update_goal"
    }

    fn description(&self) -> &str {
        "Mark the persistent local goal complete. Only use when the objective is fully achieved and no required work remains. Do not call this just because a budget was reached or the user stopped the task."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::None
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["complete"],
                    "description": "The only model-updatable goal status."
                }
            },
            "required": ["status"],
            "additionalProperties": false
        })
    }

    fn capabilities(&self, _input: &Value) -> ToolCapabilities {
        serial_goal_capabilities(true)
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: UpdateGoalInput = match serde_json::from_value(input) {
            Ok(params) => params,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };
        let store = match mangocode_core::goals::open_default_goal_store() {
            Ok(store) => store,
            Err(e) => return ToolResult::error(format!("Failed to open goal store: {}", e)),
        };
        update_goal_with_store(&store, &ctx.session_id, &params.status)
    }
}

fn get_goal_with_store(store: &SqliteSessionStore, session_id: &str) -> ToolResult {
    match store.get_thread_goal(session_id) {
        Ok(goal) => json_tool_result(json!({ "goal": goal })),
        Err(e) => ToolResult::error(format!("Failed to read goal: {}", e)),
    }
}

fn create_goal_with_store(
    store: &SqliteSessionStore,
    session_id: &str,
    objective: &str,
    token_budget: Option<i64>,
) -> ToolResult {
    match store.insert_thread_goal(session_id, objective, token_budget) {
        Ok(Some(goal)) => json_tool_result(json!({ "goal": goal })),
        Ok(None) => ToolResult::error(
            "A goal already exists for this session. Use get_goal to inspect it.".to_string(),
        ),
        Err(e) => ToolResult::error(format!("Failed to create goal: {}", e)),
    }
}

fn update_goal_with_store(
    store: &SqliteSessionStore,
    session_id: &str,
    status: &str,
) -> ToolResult {
    if status != "complete" {
        return ToolResult::error(
            "update_goal only supports status=\"complete\". Pause, resume, budget, and clear are user-controlled.".to_string(),
        );
    }
    match store.update_thread_goal(session_id, Some(ThreadGoalStatus::Complete), None) {
        Ok(Some(goal)) => json_tool_result(json!({ "goal": goal })),
        Ok(None) => ToolResult::error("No goal is set for this session.".to_string()),
        Err(e) => ToolResult::error(format!("Failed to update goal: {}", e)),
    }
}

fn serial_goal_capabilities(mutating: bool) -> ToolCapabilities {
    let mut capabilities = if mutating {
        ToolCapabilities::mutating()
    } else {
        ToolCapabilities::read_only()
    };
    capabilities.parallel_safe = false;
    capabilities
}

fn json_tool_result(value: Value) -> ToolResult {
    match serde_json::to_string_pretty(&value) {
        Ok(text) => ToolResult::success(text).with_metadata(value),
        Err(e) => ToolResult::error(format!("Failed to serialize goal result: {}", e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_goal_store() -> (tempfile::TempDir, SqliteSessionStore) {
        let dir = tempfile::tempdir().expect("temp dir");
        let db = dir.path().join("sessions.db");
        let store = SqliteSessionStore::open(&db).expect("open sqlite store");
        (dir, store)
    }

    #[test]
    fn create_and_get_goal_use_same_session_store() {
        let (_dir, store) = temp_goal_store();

        let created = create_goal_with_store(&store, "session-1", "ship tool goals", Some(50));
        assert!(!created.is_error, "{}", created.content);
        let created_meta = created.metadata.as_ref().expect("created metadata");
        assert_eq!(created_meta["goal"]["objective"], "ship tool goals");
        assert_eq!(created_meta["goal"]["status"], "active");
        assert_eq!(created_meta["goal"]["token_budget"], 50);

        let fetched = get_goal_with_store(&store, "session-1");
        assert!(!fetched.is_error, "{}", fetched.content);
        let fetched_meta = fetched.metadata.as_ref().expect("fetched metadata");
        assert_eq!(fetched_meta["goal"]["objective"], "ship tool goals");
        assert_eq!(
            fetched_meta["goal"]["goal_id"],
            created_meta["goal"]["goal_id"]
        );
    }

    #[test]
    fn create_goal_rejects_duplicates_without_replacing_existing_goal() {
        let (_dir, store) = temp_goal_store();

        let first = create_goal_with_store(&store, "session-1", "first objective", None);
        assert!(!first.is_error, "{}", first.content);
        let first_goal_id = first.metadata.unwrap()["goal"]["goal_id"]
            .as_str()
            .unwrap()
            .to_string();

        let duplicate = create_goal_with_store(&store, "session-1", "second objective", None);
        assert!(duplicate.is_error);

        let saved = store
            .get_thread_goal("session-1")
            .unwrap()
            .expect("saved goal");
        assert_eq!(saved.goal_id, first_goal_id);
        assert_eq!(saved.objective, "first objective");
    }

    #[test]
    fn update_goal_only_allows_model_to_complete_existing_goal() {
        let (_dir, store) = temp_goal_store();

        let missing = update_goal_with_store(&store, "session-1", "complete");
        assert!(missing.is_error);

        create_goal_with_store(&store, "session-1", "finish objective", None);
        let invalid = update_goal_with_store(&store, "session-1", "paused");
        assert!(invalid.is_error);
        let still_active = store
            .get_thread_goal("session-1")
            .unwrap()
            .expect("saved goal");
        assert_eq!(still_active.status, ThreadGoalStatus::Active);

        let completed = update_goal_with_store(&store, "session-1", "complete");
        assert!(!completed.is_error, "{}", completed.content);
        let completed_meta = completed.metadata.as_ref().expect("completed metadata");
        assert_eq!(completed_meta["goal"]["status"], "complete");
    }
}
