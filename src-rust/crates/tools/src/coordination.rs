use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use mangocode_core::coordination::{ConflictWarning, CoordinationStore};
use mangocode_core::WorkClaim;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::PathBuf;

fn open_store() -> anyhow::Result<CoordinationStore> {
    CoordinationStore::open_default()
}

fn process_id(session_id: &str) -> String {
    mangocode_core::coordination::process_session_id(session_id)
}

pub fn register_current_session(ctx: &ToolContext) {
    if let Ok(store) = open_store() {
        let _ = store.register_session(&process_id(&ctx.session_id), &ctx.working_dir, "", None);
    }
}

pub fn preflight_write_conflicts(
    ctx: &ToolContext,
    tool_name: &str,
    paths: &[PathBuf],
    confirm_conflicts: bool,
) -> Result<Option<Vec<ConflictWarning>>, ToolResult> {
    let Ok(store) = open_store() else {
        return Ok(None);
    };
    let session_id = process_id(&ctx.session_id);
    let _ = store.register_session(&session_id, &ctx.working_dir, "", None);
    let conflicts = match store.find_conflicts(&session_id, &ctx.working_dir, paths) {
        Ok(conflicts) => conflicts,
        Err(_) => return Ok(None),
    };
    if conflicts.is_empty() {
        return Ok(None);
    }

    let summary = format_conflicts(&conflicts);
    if !confirm_conflicts {
        return Err(ToolResult::error(format!(
            "{tool_name} overlaps with active MangoCode work claims.\n\n{summary}\n\nCoordinate with the other session or retry this tool with confirm_conflicts=true to acknowledge and proceed."
        ))
        .with_metadata(json!({
            "coordination_conflicts": conflicts,
            "requires_confirm_conflicts": true,
        })));
    }

    Ok(Some(conflicts))
}

pub fn append_confirmed_conflict_note(
    content: String,
    conflicts: Option<&[ConflictWarning]>,
) -> String {
    match conflicts {
        Some(conflicts) if !conflicts.is_empty() => format!(
            "{}\n\nCoordination warning acknowledged:\n{}",
            content,
            format_conflicts(conflicts)
        ),
        _ => content,
    }
}

pub fn execution_claim_notice(ctx: &ToolContext, command: &str) -> Option<String> {
    if !looks_like_verification_command(command) {
        return None;
    }
    let store = open_store().ok()?;
    let session_id = process_id(&ctx.session_id);
    let _ = store.register_session(&session_id, &ctx.working_dir, "", None);
    let claims = store.list_claims(Some(&ctx.working_dir)).ok()?;
    let other_claims: Vec<WorkClaim> = claims
        .into_iter()
        .filter(|claim| claim.session_id != session_id)
        .take(8)
        .collect();
    if other_claims.is_empty() {
        return None;
    }

    let mut lines = vec![
        "Coordination notice: another MangoCode session has active write/work claims in this repo, so test/build failures may be transient.".to_string(),
    ];
    for claim in other_claims {
        lines.push(format!(
            "- session {} claims `{}` ({})",
            short_id(&claim.session_id),
            claim.scope,
            claim.summary.as_deref().unwrap_or(&claim.claim_type)
        ));
    }
    Some(lines.join("\n"))
}

pub fn append_execution_notice(mut result: ToolResult, notice: Option<&str>) -> ToolResult {
    if let Some(notice) = notice.filter(|notice| !notice.trim().is_empty()) {
        result.content = format!("{}\n\n{}", result.content, notice);
        let mut metadata = result.metadata.take().unwrap_or_else(|| json!({}));
        if let Some(obj) = metadata.as_object_mut() {
            obj.insert("coordination_notice".to_string(), json!(notice));
        }
        result.metadata = Some(metadata);
    }
    result
}

fn looks_like_verification_command(command: &str) -> bool {
    let cmd = command.to_ascii_lowercase();
    let normalized = cmd.split_whitespace().collect::<Vec<_>>().join(" ");
    let prefixes = [
        "cargo test",
        "cargo nextest",
        "cargo check",
        "cargo clippy",
        "cargo build",
        "npm test",
        "npm run test",
        "pnpm test",
        "pnpm run test",
        "yarn test",
        "yarn run test",
        "pytest",
        "python -m pytest",
        "py -m pytest",
        "go test",
        "dotnet test",
        "mvn test",
        "gradle test",
        "./gradlew test",
        "make test",
    ];
    prefixes.iter().any(|prefix| normalized.starts_with(prefix))
}

fn format_conflicts(conflicts: &[ConflictWarning]) -> String {
    conflicts
        .iter()
        .take(8)
        .map(|c| {
            let summary = c
                .summary
                .as_deref()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or("active work claim");
            format!(
                "- session {} claims `{}` ({summary}); attempted path `{}`",
                short_id(&c.session_id),
                c.scope,
                c.path
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn short_id(session_id: &str) -> &str {
    let end = session_id
        .char_indices()
        .nth(8)
        .map(|(idx, _)| idx)
        .unwrap_or(session_id.len());
    &session_id[..end]
}

#[derive(Debug, Deserialize)]
struct CoordinationStatusInput {
    #[serde(default)]
    all_repos: bool,
    #[serde(default = "default_true")]
    include_inbox: bool,
    #[serde(default)]
    mark_read: bool,
}

#[derive(Debug, Deserialize)]
struct ClaimWorkInput {
    scope: String,
    #[serde(default = "default_claim_type")]
    claim_type: String,
    #[serde(default)]
    summary: Option<String>,
    #[serde(default)]
    confirm_conflicts: bool,
}

#[derive(Debug, Deserialize)]
struct ReleaseWorkInput {
    #[serde(default)]
    claim_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CoordinationMessageInput {
    body: String,
    #[serde(default)]
    to_session_id: Option<String>,
    #[serde(default = "default_true")]
    repo_broadcast: bool,
}

fn default_true() -> bool {
    true
}

fn default_claim_type() -> String {
    "edit".to_string()
}

fn resolve_message_route(
    to_session_id: Option<&str>,
    repo_broadcast: bool,
) -> Result<(Option<&str>, bool), &'static str> {
    let target = to_session_id
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if target.is_none() && !repo_broadcast {
        return Err("Set to_session_id for a direct message, or leave repo_broadcast=true.");
    }
    Ok((target, target.is_none() && repo_broadcast))
}

pub struct CoordinationStatusTool;
pub struct ClaimWorkTool;
pub struct ReleaseWorkTool;
pub struct CoordinationMessageTool;

#[async_trait]
impl Tool for CoordinationStatusTool {
    fn name(&self) -> &str {
        "CoordinationStatus"
    }

    fn description(&self) -> &str {
        "List active local MangoCode sessions, advisory work claims, and unread coordination messages."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "all_repos": {
                    "type": "boolean",
                    "description": "If true, list all local repos; otherwise only the current repo."
                },
                "include_inbox": {
                    "type": "boolean",
                    "description": "Include unread direct and repo-broadcast messages."
                },
                "mark_read": {
                    "type": "boolean",
                    "description": "Mark returned inbox messages as read."
                }
            },
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: CoordinationStatusInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {e}")),
        };
        let store = match open_store() {
            Ok(store) => store,
            Err(e) => return ToolResult::error(format!("Failed to open coordination store: {e}")),
        };
        let session_id = process_id(&ctx.session_id);
        let _ = store.register_session(&session_id, &ctx.working_dir, "", None);
        let repo_filter = (!params.all_repos).then_some(ctx.working_dir.as_path());
        let sessions = match store.list_sessions(repo_filter) {
            Ok(sessions) => sessions,
            Err(e) => return ToolResult::error(format!("Failed to list sessions: {e}")),
        };
        let claims = match store.list_claims(repo_filter) {
            Ok(claims) => claims,
            Err(e) => return ToolResult::error(format!("Failed to list claims: {e}")),
        };
        let inbox = if params.include_inbox {
            match store.inbox(&session_id, &ctx.working_dir, params.mark_read) {
                Ok(messages) => messages,
                Err(e) => return ToolResult::error(format!("Failed to read inbox: {e}")),
            }
        } else {
            Vec::new()
        };

        let mut lines = vec![format!(
            "Active sessions: {}. Active claims: {}. Unread messages: {}.",
            sessions.len(),
            claims.len(),
            inbox.len()
        )];
        for session in sessions.iter().take(8) {
            lines.push(format!(
                "- session {} pid {} repo `{}`{}",
                short_id(&session.session_id),
                session.pid,
                session.repo_root,
                session
                    .title
                    .as_ref()
                    .map(|t| format!(" title `{t}`"))
                    .unwrap_or_default()
            ));
        }
        for claim in claims.iter().take(8) {
            lines.push(format!(
                "- claim {} by {}: `{}` ({})",
                short_id(&claim.claim_id),
                short_id(&claim.session_id),
                claim.scope,
                claim.summary.as_deref().unwrap_or(&claim.claim_type)
            ));
        }
        for message in inbox.iter().take(8) {
            lines.push(format!(
                "- message from {}: {}",
                short_id(&message.from_session_id),
                message.body
            ));
        }

        ToolResult::success(lines.join("\n")).with_metadata(json!({
            "sessions": sessions,
            "claims": claims,
            "inbox": inbox,
        }))
    }
}

#[async_trait]
impl Tool for ClaimWorkTool {
    fn name(&self) -> &str {
        "ClaimWork"
    }

    fn description(&self) -> &str {
        "Create an advisory local work claim for a file, directory, glob, module, or task."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "description": "Repo-relative file, directory, or glob scope, such as src/** or crates/core/src/lib.rs."
                },
                "claim_type": {
                    "type": "string",
                    "description": "Short type such as edit, test, review, build, or refactor."
                },
                "summary": {
                    "type": "string",
                    "description": "Short human-readable summary of the work."
                },
                "confirm_conflicts": {
                    "type": "boolean",
                    "description": "Set true only after acknowledging that this claim overlaps active MangoCode work claims."
                }
            },
            "required": ["scope"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: ClaimWorkInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {e}")),
        };
        if params.scope.trim().is_empty() {
            return ToolResult::error("scope cannot be empty.");
        }
        let store = match open_store() {
            Ok(store) => store,
            Err(e) => return ToolResult::error(format!("Failed to open coordination store: {e}")),
        };
        let session_id = process_id(&ctx.session_id);
        let _ = store.register_session(&session_id, &ctx.working_dir, "", None);
        let conflicts =
            match store.find_scope_conflicts(&session_id, &ctx.working_dir, &params.scope) {
                Ok(conflicts) => conflicts,
                Err(e) => {
                    return ToolResult::error(format!("Failed to check claim conflicts: {e}"))
                }
            };
        if !conflicts.is_empty() && !params.confirm_conflicts {
            return ToolResult::error(format!(
                "ClaimWork overlaps with active MangoCode work claims.\n\n{}\n\nCoordinate with the other session or retry ClaimWork with confirm_conflicts=true to acknowledge and proceed.",
                format_conflicts(&conflicts)
            ))
            .with_metadata(json!({
                "coordination_conflicts": conflicts,
                "requires_confirm_conflicts": true,
            }));
        }
        match store.create_claim(
            &session_id,
            &ctx.working_dir,
            &params.claim_type,
            &params.scope,
            params.summary.as_deref(),
        ) {
            Ok(claim) => {
                let mut message = format!(
                    "Claimed `{}` for {} as {}.",
                    claim.scope,
                    params.summary.as_deref().unwrap_or("coordination"),
                    short_id(&claim.claim_id)
                );
                if !conflicts.is_empty() {
                    message = append_confirmed_conflict_note(message, Some(&conflicts));
                }
                ToolResult::success(message).with_metadata(json!({
                    "claim": claim,
                    "coordination_conflicts": conflicts,
                }))
            }
            Err(e) => ToolResult::error(format!("Failed to create claim: {e}")),
        }
    }
}

#[async_trait]
impl Tool for ReleaseWorkTool {
    fn name(&self) -> &str {
        "ReleaseWork"
    }

    fn description(&self) -> &str {
        "Release this session's advisory work claims, either one claim_id or all claims."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "claim_id": {
                    "type": "string",
                    "description": "Optional claim id to release. Omit to release all claims for this session."
                }
            },
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: ReleaseWorkInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {e}")),
        };
        let store = match open_store() {
            Ok(store) => store,
            Err(e) => return ToolResult::error(format!("Failed to open coordination store: {e}")),
        };
        match store.release_claims(&process_id(&ctx.session_id), params.claim_id.as_deref()) {
            Ok(count) => ToolResult::success(format!("Released {count} work claim(s).")),
            Err(e) => ToolResult::error(format!("Failed to release claims: {e}")),
        }
    }
}

#[async_trait]
impl Tool for CoordinationMessageTool {
    fn name(&self) -> &str {
        "CoordinationMessage"
    }

    fn description(&self) -> &str {
        "Send a local coordination message to one MangoCode session or broadcast to the current repo."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "body": {
                    "type": "string",
                    "description": "Message body to send."
                },
                "to_session_id": {
                    "type": "string",
                    "description": "Optional target session id. Omit for repo broadcast."
                },
                "repo_broadcast": {
                    "type": "boolean",
                    "description": "When no target is set, broadcast to active sessions in this repo."
                }
            },
            "required": ["body"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: CoordinationMessageInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {e}")),
        };
        if params.body.trim().is_empty() {
            return ToolResult::error("body cannot be empty.");
        }
        let (to_session_id, repo_broadcast) =
            match resolve_message_route(params.to_session_id.as_deref(), params.repo_broadcast) {
                Ok(route) => route,
                Err(e) => return ToolResult::error(e.to_string()),
            };
        let store = match open_store() {
            Ok(store) => store,
            Err(e) => return ToolResult::error(format!("Failed to open coordination store: {e}")),
        };
        let session_id = process_id(&ctx.session_id);
        let _ = store.register_session(&session_id, &ctx.working_dir, "", None);
        let repo = if repo_broadcast {
            Some(ctx.working_dir.as_path())
        } else {
            None
        };
        match store.send_message(&session_id, to_session_id, repo, &params.body) {
            Ok(message) => ToolResult::success(format!(
                "Coordination message queued as {}.",
                short_id(&message.message_id)
            ))
            .with_metadata(json!({ "message": message })),
            Err(e) => ToolResult::error(format!("Failed to send message: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verification_command_detection_covers_common_test_and_build_commands() {
        assert!(looks_like_verification_command("cargo test -p core"));
        assert!(looks_like_verification_command("npm test"));
        assert!(looks_like_verification_command("pytest tests"));
        assert!(looks_like_verification_command("cargo check --workspace"));
        assert!(looks_like_verification_command(
            "cargo clippy --all-targets"
        ));
        assert!(looks_like_verification_command("cargo build --release"));
        assert!(!looks_like_verification_command("git status"));
        assert!(!looks_like_verification_command("rg test src"));
    }

    #[test]
    fn message_route_rejects_undeliverable_message() {
        assert!(resolve_message_route(None, false).is_err());
        assert!(resolve_message_route(Some("   "), false).is_err());
        assert_eq!(resolve_message_route(None, true).unwrap(), (None, true));
        assert_eq!(
            resolve_message_route(Some(" session-a "), true).unwrap(),
            (Some("session-a"), false)
        );
    }
}
