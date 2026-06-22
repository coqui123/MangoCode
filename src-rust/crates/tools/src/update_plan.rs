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
    #[serde(default)]
    plan: Vec<PlanItem>,
    /// Optional durable plan id. When set, the full plan (steps, statuses, and
    /// dependencies) is checkpointed to `<project>/.mangocode/plans/<id>.json`
    /// so it survives across sessions and can be resumed later.
    #[serde(default)]
    plan_id: Option<String>,
    /// "update" (default) writes the provided plan; "resume" loads a previously
    /// checkpointed plan by `plan_id` and reports done/ready/blocked steps.
    #[serde(default)]
    action: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct PlanItem {
    step: String,
    status: PlanStatus,
    /// Stable identifier for dependency references. Defaults to the 1-based
    /// position when omitted (so a flat plan needs no ids).
    #[serde(default)]
    id: Option<String>,
    /// Ids of steps that must be `completed` before this step is `ready`.
    /// Turning the checklist into a dependency DAG.
    #[serde(default)]
    depends_on: Vec<String>,
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
         one step may be in_progress. Steps may declare `depends_on` (ids of \
         steps that must complete first), turning the plan into a dependency \
         DAG; the tool reports which steps are ready vs blocked. Pass a `plan_id` \
         to checkpoint the plan durably under .mangocode/plans/ so it survives \
         across sessions; call with action=\"resume\" and that `plan_id` to reload \
         a checkpointed plan and see done/ready/blocked steps."
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
                    "description": "The complete current plan checklist. At most one item may be in_progress. Required for action=update.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": { "type": "string" },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"]
                            },
                            "id": {
                                "type": "string",
                                "description": "Optional stable id for dependency references. Defaults to the 1-based position."
                            },
                            "depends_on": {
                                "type": "array",
                                "items": { "type": "string" },
                                "description": "Ids of steps that must be completed before this step is ready."
                            }
                        },
                        "required": ["step", "status"],
                        "additionalProperties": false
                    }
                },
                "plan_id": {
                    "type": "string",
                    "description": "Optional durable id. When set, checkpoints the plan under .mangocode/plans/<id>.json for cross-session resume."
                },
                "action": {
                    "type": "string",
                    "enum": ["update", "resume"],
                    "description": "update (default) writes the plan; resume loads a checkpointed plan by plan_id and reports done/ready/blocked."
                }
            },
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: UpdatePlanInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        let action = params
            .action
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("update")
            .to_ascii_lowercase();
        let plan_id = params
            .plan_id
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);

        // ----- resume: reload a checkpointed plan and report its status -----
        if action == "resume" {
            let Some(plan_id) = plan_id else {
                return ToolResult::error("action=resume requires a plan_id.".to_string());
            };
            let path = plan_checkpoint_path(&ctx.working_dir, &plan_id);
            let steps = match load_checkpoint(&path) {
                Ok(Some(steps)) => steps,
                Ok(None) => {
                    return ToolResult::error(format!(
                        "No checkpointed plan found for plan_id {plan_id:?} at {}.",
                        path.display()
                    ));
                }
                Err(e) => return ToolResult::error(e),
            };
            if let Err(e) = sync_plan_todos(&ctx.session_id, &steps) {
                return ToolResult::error(e);
            }
            let progress = classify_steps(&steps);
            return ToolResult::success(progress.summary(&plan_id)).with_metadata(json!({
                "transcript_display": {
                    "kind": "updated_plan",
                    "plan": steps_display(&steps),
                },
                "plan_id": plan_id,
                "ready_steps": progress.ready,
                "blocked_steps": progress.blocked,
                "completed": progress.done,
                "total": progress.total,
                "resumed": true,
            }));
        }

        // ----- update -----
        debug!(count = params.plan.len(), "Updating session plan");
        if params.plan.is_empty() {
            return ToolResult::error(
                "Invalid plan: action=update requires a non-empty plan.".to_string(),
            );
        }

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

        let steps = resolve_steps(&params.plan);
        if let Err(e) = validate_dependencies(&steps) {
            return ToolResult::error(e);
        }

        let explanation = params
            .explanation
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);

        // Keep the existing todo/progress surface in sync without requiring the
        // model to use TodoWrite's separate id/priority schema.
        if let Err(e) = sync_plan_todos(&ctx.session_id, &steps) {
            return ToolResult::error(e);
        }

        // Durably checkpoint the plan when a plan_id is provided so it can be
        // resumed in a later session.
        let mut checkpoint_path = None;
        if let Some(ref plan_id) = plan_id {
            let path = plan_checkpoint_path(&ctx.working_dir, plan_id);
            if let Err(e) = save_checkpoint(&path, plan_id, explanation.as_deref(), &steps) {
                return ToolResult::error(e);
            }
            checkpoint_path = Some(path.display().to_string());
        }

        let progress = classify_steps(&steps);
        ToolResult::success("Plan updated").with_metadata(json!({
            "transcript_display": {
                "kind": "updated_plan",
                "explanation": explanation,
                "plan": steps_display(&steps),
            },
            "plan_id": plan_id,
            "checkpoint_path": checkpoint_path,
            "ready_steps": progress.ready,
            "blocked_steps": progress.blocked,
            "completed": progress.done,
            "total": progress.total,
        }))
    }
}

/// A plan step with its dependency edges resolved to stable ids.
struct ResolvedStep {
    id: String,
    step: String,
    status: PlanStatus,
    depends_on: Vec<String>,
}

/// Ready/blocked/done classification of a dependency-aware plan.
struct PlanProgress {
    ready: Vec<Value>,
    blocked: Vec<Value>,
    done: usize,
    total: usize,
}

impl PlanProgress {
    fn summary(&self, plan_id: &str) -> String {
        let ids = |items: &[Value]| -> String {
            let v: Vec<&str> = items
                .iter()
                .filter_map(|item| item["id"].as_str())
                .collect();
            if v.is_empty() {
                "none".to_string()
            } else {
                v.join(", ")
            }
        };
        format!(
            "Plan {:?}: {}/{} completed. Ready: {}. Blocked: {}.",
            plan_id,
            self.done,
            self.total,
            ids(&self.ready),
            ids(&self.blocked),
        )
    }
}

/// Assign each step a stable id (defaulting to its 1-based position) and
/// normalize the dependency edges.
fn resolve_steps(items: &[PlanItem]) -> Vec<ResolvedStep> {
    items
        .iter()
        .enumerate()
        .map(|(idx, item)| ResolvedStep {
            id: item
                .id
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| (idx + 1).to_string()),
            step: item.step.trim().to_string(),
            status: item.status,
            depends_on: item
                .depends_on
                .iter()
                .map(|d| d.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
        })
        .collect()
}

/// Reject duplicate ids, self-edges, dangling dependency ids, and cycles.
fn validate_dependencies(steps: &[ResolvedStep]) -> Result<(), String> {
    let ids: std::collections::HashSet<&str> = steps.iter().map(|s| s.id.as_str()).collect();
    if ids.len() != steps.len() {
        return Err("Invalid plan: step ids must be unique.".to_string());
    }
    for s in steps {
        for dep in &s.depends_on {
            if dep == &s.id {
                return Err(format!("Invalid plan: step {:?} depends on itself.", s.id));
            }
            if !ids.contains(dep.as_str()) {
                return Err(format!(
                    "Invalid plan: step {:?} depends on unknown id {:?}.",
                    s.id, dep
                ));
            }
        }
    }
    detect_cycle(steps)
}

fn detect_cycle(steps: &[ResolvedStep]) -> Result<(), String> {
    use std::collections::HashMap;
    let deps: HashMap<&str, &[String]> = steps
        .iter()
        .map(|s| (s.id.as_str(), s.depends_on.as_slice()))
        .collect();
    // 0 = unvisited, 1 = on the current DFS stack, 2 = fully explored.
    let mut state: HashMap<&str, u8> = HashMap::new();
    for s in steps {
        visit_for_cycle(s.id.as_str(), &deps, &mut state)?;
    }
    Ok(())
}

fn visit_for_cycle<'a>(
    id: &'a str,
    deps: &std::collections::HashMap<&'a str, &'a [String]>,
    state: &mut std::collections::HashMap<&'a str, u8>,
) -> Result<(), String> {
    match state.get(id) {
        Some(2) => return Ok(()),
        Some(1) => {
            return Err(format!(
                "Invalid plan: dependency cycle through step {:?}.",
                id
            ))
        }
        _ => {}
    }
    state.insert(id, 1);
    if let Some(edges) = deps.get(id) {
        for dep in edges.iter() {
            visit_for_cycle(dep.as_str(), deps, state)?;
        }
    }
    state.insert(id, 2);
    Ok(())
}

/// A pending step is `ready` when every dependency is `completed`, otherwise
/// `blocked`.
fn classify_steps(steps: &[ResolvedStep]) -> PlanProgress {
    use std::collections::HashMap;
    let status_by_id: HashMap<&str, PlanStatus> =
        steps.iter().map(|s| (s.id.as_str(), s.status)).collect();
    let done = steps
        .iter()
        .filter(|s| s.status == PlanStatus::Completed)
        .count();
    let mut ready = Vec::new();
    let mut blocked = Vec::new();
    for s in steps {
        if s.status != PlanStatus::Pending {
            continue;
        }
        let deps_satisfied = s.depends_on.iter().all(|dep| {
            matches!(status_by_id.get(dep.as_str()), Some(PlanStatus::Completed))
        });
        let entry = json!({ "id": s.id, "step": s.step });
        if deps_satisfied {
            ready.push(entry);
        } else {
            blocked.push(entry);
        }
    }
    PlanProgress {
        ready,
        blocked,
        done,
        total: steps.len(),
    }
}

fn steps_display(steps: &[ResolvedStep]) -> Vec<Value> {
    steps
        .iter()
        .map(|s| {
            json!({
                "id": s.id,
                "step": s.step,
                "status": s.status.to_string(),
                "depends_on": s.depends_on,
            })
        })
        .collect()
}

/// Mirror the plan onto the session todo surface (id `plan-N`, content = step)
/// so the existing progress display keeps working.
fn sync_plan_todos(session_id: &str, steps: &[ResolvedStep]) -> Result<(), String> {
    let todos: Vec<Value> = steps
        .iter()
        .enumerate()
        .map(|(idx, s)| {
            json!({
                "id": format!("plan-{}", idx + 1),
                "content": s.step,
                "status": s.status.to_string(),
            })
        })
        .collect();
    crate::todo_write::save_todos(session_id, &todos)
}

fn sanitize_plan_id(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

fn plan_checkpoint_path(working_dir: &std::path::Path, plan_id: &str) -> std::path::PathBuf {
    working_dir
        .join(".mangocode")
        .join("plans")
        .join(format!("{}.json", sanitize_plan_id(plan_id)))
}

fn save_checkpoint(
    path: &std::path::Path,
    plan_id: &str,
    explanation: Option<&str>,
    steps: &[ResolvedStep],
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create plan checkpoint directory: {e}"))?;
    }
    let doc = json!({
        "plan_id": plan_id,
        "explanation": explanation,
        "steps": steps.iter().map(|s| json!({
            "id": s.id,
            "step": s.step,
            "status": s.status.to_string(),
            "depends_on": s.depends_on,
        })).collect::<Vec<_>>(),
    });
    let text = serde_json::to_string_pretty(&doc)
        .map_err(|e| format!("Failed to serialize plan checkpoint: {e}"))?;
    // Write atomically (temp file + rename) so a crash mid-write can't leave a
    // truncated checkpoint that fails to parse on resume.
    crate::fs_atomic::write_atomic_sync(path, text.as_bytes())
        .map_err(|e| format!("Failed to write plan checkpoint: {e}"))
}

fn load_checkpoint(path: &std::path::Path) -> Result<Option<Vec<ResolvedStep>>, String> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("Failed to read plan checkpoint: {e}")),
    };
    let doc: Value =
        serde_json::from_str(&text).map_err(|e| format!("Corrupt plan checkpoint: {e}"))?;
    let arr = doc
        .get("steps")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "Plan checkpoint is missing a steps array.".to_string())?;
    let mut steps = Vec::with_capacity(arr.len());
    for (idx, v) in arr.iter().enumerate() {
        let id = v
            .get("id")
            .and_then(|x| x.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| (idx + 1).to_string());
        let step = v
            .get("step")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let status = v
            .get("status")
            .and_then(|x| x.as_str())
            .and_then(|s| PlanStatus::from_str_ci(s).ok())
            .unwrap_or(PlanStatus::Pending);
        let depends_on = v
            .get("depends_on")
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|d| d.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        steps.push(ResolvedStep {
            id,
            step,
            status,
            depends_on,
        });
    }
    Ok(Some(steps))
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
            coordination_actor_id: None,
            coordination_parent_actor_id: None,
            inject_coordination_inbox: true,
            file_history: Arc::new(parking_lot::Mutex::new(
                mangocode_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(AtomicUsize::new(1)),
            non_interactive: true,
            mcp_manager: None,
            config: mangocode_core::config::Config::default(),
            question_prompt_tx: None,
            cancel_token: None,
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
            coordination_actor_id: None,
            coordination_parent_actor_id: None,
            inject_coordination_inbox: true,
            file_history: Arc::new(parking_lot::Mutex::new(
                mangocode_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(AtomicUsize::new(1)),
            non_interactive: true,
            mcp_manager: None,
            config: mangocode_core::config::Config::default(),
            question_prompt_tx: None,
            cancel_token: None,
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

    fn rs(id: &str, status: PlanStatus, deps: &[&str]) -> ResolvedStep {
        ResolvedStep {
            id: id.to_string(),
            step: format!("step {id}"),
            status,
            depends_on: deps.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn classify_marks_blocked_until_deps_complete() {
        let steps = vec![
            rs("a", PlanStatus::Completed, &[]),
            rs("b", PlanStatus::Pending, &["a"]),
            rs("c", PlanStatus::Pending, &["b"]),
        ];
        let p = classify_steps(&steps);
        assert_eq!(p.done, 1);
        let ready: Vec<&str> = p.ready.iter().filter_map(|v| v["id"].as_str()).collect();
        let blocked: Vec<&str> = p.blocked.iter().filter_map(|v| v["id"].as_str()).collect();
        assert_eq!(ready, vec!["b"], "b is ready once a completes");
        assert_eq!(blocked, vec!["c"], "c is blocked until b completes");
    }

    #[test]
    fn validate_rejects_cycles_and_unknown_deps() {
        let cyclic = vec![
            rs("a", PlanStatus::Pending, &["b"]),
            rs("b", PlanStatus::Pending, &["a"]),
        ];
        assert!(validate_dependencies(&cyclic).is_err(), "cycle must be rejected");

        let dangling = vec![rs("a", PlanStatus::Pending, &["ghost"])];
        assert!(
            validate_dependencies(&dangling).is_err(),
            "unknown dep id must be rejected"
        );

        let ok = vec![
            rs("a", PlanStatus::Completed, &[]),
            rs("b", PlanStatus::Pending, &["a"]),
        ];
        assert!(validate_dependencies(&ok).is_ok());
    }

    #[cfg(feature = "tool-update-plan")]
    #[tokio::test]
    async fn checkpoint_and_resume_round_trip() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::Arc;

        let dir = tempfile::tempdir().unwrap();
        let session_id = format!(
            "plan-dag-test-{}",
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
            coordination_actor_id: None,
            coordination_parent_actor_id: None,
            inject_coordination_inbox: true,
            file_history: Arc::new(parking_lot::Mutex::new(
                mangocode_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(AtomicUsize::new(1)),
            non_interactive: true,
            mcp_manager: None,
            config: mangocode_core::config::Config::default(),
            question_prompt_tx: None,
            cancel_token: None,
        };

        // Checkpoint a dependency DAG.
        let res = UpdatePlanTool
            .execute(
                json!({
                    "plan_id": "feature-x",
                    "plan": [
                        { "id": "a", "step": "design", "status": "completed" },
                        { "id": "b", "step": "build", "status": "in_progress", "depends_on": ["a"] },
                        { "id": "c", "step": "test", "status": "pending", "depends_on": ["b"] }
                    ]
                }),
                &ctx,
            )
            .await;
        assert!(!res.is_error, "{}", res.content);
        let meta = res.metadata.as_ref().unwrap();
        assert_eq!(meta["completed"], json!(1));
        assert_eq!(meta["total"], json!(3));
        // c is blocked because its dep b is not completed.
        assert_eq!(meta["blocked_steps"].as_array().unwrap().len(), 1);
        let checkpoint = dir
            .path()
            .join(".mangocode")
            .join("plans")
            .join("feature-x.json");
        assert!(checkpoint.exists(), "checkpoint file must be written");

        // Resume from disk and confirm status is preserved.
        let res2 = UpdatePlanTool
            .execute(json!({ "action": "resume", "plan_id": "feature-x" }), &ctx)
            .await;
        assert!(!res2.is_error, "{}", res2.content);
        let meta2 = res2.metadata.as_ref().unwrap();
        assert_eq!(meta2["resumed"], json!(true));
        assert_eq!(meta2["total"], json!(3));
        assert_eq!(meta2["completed"], json!(1));

        // Resuming an unknown plan id errors clearly.
        let missing = UpdatePlanTool
            .execute(json!({ "action": "resume", "plan_id": "does-not-exist" }), &ctx)
            .await;
        assert!(missing.is_error);
        assert!(missing.content.contains("No checkpointed plan"));

        let _ = std::fs::remove_file(crate::todo_write::todos_path(&session_id));
    }
}
