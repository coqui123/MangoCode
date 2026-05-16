use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use mangocode_core::coordination::{
    default_db_path, ConflictWarning, CoordinationMessage, CoordinationStore, SessionPresence,
    SessionTargetResolution,
};
use mangocode_core::WorkClaim;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::PathBuf;

fn open_store() -> anyhow::Result<CoordinationStore> {
    CoordinationStore::open_default()
}

pub fn register_current_session(ctx: &ToolContext) {
    if let Ok(store) = open_store() {
        let _ = register_current_session_with_store(&store, ctx);
    }
}

fn register_current_session_with_store(store: &CoordinationStore, ctx: &ToolContext) -> String {
    let session_id = ctx.coordination_actor_id();
    let _ = store.register_session_with_parent(
        &session_id,
        &ctx.working_dir,
        "",
        None,
        ctx.coordination_parent_session_id().as_deref(),
    );
    session_id
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
    let session_id = register_current_session_with_store(&store, ctx);
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
    let session_id = register_current_session_with_store(&store, ctx);
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
        "Coordination notice: another MangoCode actor has active write/work claims in this repo, so test/build failures may be transient.".to_string(),
    ];
    for claim in other_claims {
        lines.push(format!(
            "- actor {} claims `{}` ({})",
            short_id(&claim.session_id),
            claim.scope,
            claim.summary.as_deref().unwrap_or(&claim.claim_type)
        ));
    }
    Some(lines.join("\n"))
}

pub struct TransientWriteClaimGuard {
    session_id: String,
    claim_ids: Vec<String>,
}

impl Drop for TransientWriteClaimGuard {
    fn drop(&mut self) {
        if self.claim_ids.is_empty() {
            return;
        }
        if let Ok(store) = CoordinationStore::open(&default_db_path()) {
            for claim_id in &self.claim_ids {
                let _ = store.release_claims(&self.session_id, Some(claim_id));
            }
        }
    }
}

pub fn begin_transient_write_claim(
    ctx: &ToolContext,
    tool_name: &str,
    paths: &[PathBuf],
    confirm_conflicts: bool,
) -> Result<Option<TransientWriteClaimGuard>, ToolResult> {
    if paths.is_empty() {
        return Ok(None);
    }
    let Ok(store) = open_store() else {
        return Ok(None);
    };
    let session_id = register_current_session_with_store(&store, ctx);
    let mut claim_ids = Vec::new();
    let summary = format!("transient {tool_name} write");
    for path in paths {
        if let Ok(claim) = store.create_claim(
            &session_id,
            &ctx.working_dir,
            "transient-write",
            &path.to_string_lossy(),
            Some(&summary),
        ) {
            claim_ids.push(claim.claim_id);
        }
    }
    if claim_ids.is_empty() {
        return Ok(None);
    }

    let guard = TransientWriteClaimGuard {
        session_id,
        claim_ids,
    };

    let conflicts = match store.find_conflicts(&guard.session_id, &ctx.working_dir, paths) {
        Ok(conflicts) => conflicts,
        Err(_) => return Ok(Some(guard)),
    };
    if !conflicts.is_empty() && !confirm_conflicts {
        let summary = format_conflicts(&conflicts);
        return Err(ToolResult::error(format!(
            "{tool_name} overlaps with active MangoCode work claims that appeared after preflight.\n\n{summary}\n\nCoordinate with the other session or retry this tool with confirm_conflicts=true to acknowledge and proceed."
        ))
        .with_metadata(json!({
            "coordination_conflicts": conflicts,
            "requires_confirm_conflicts": true,
        })));
    }

    Ok(Some(guard))
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
                "- actor {} claims `{}` ({summary}); attempted path `{}`",
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

fn message_actor_metadata(message: &CoordinationMessage) -> Value {
    json!({
        "message_id": message.message_id,
        "from_actor_id": message.from_session_id,
        "to_actor_id": message.to_session_id,
        "repo_root": message.repo_root,
        "body": message.body,
        "created_at": message.created_at,
        "read_at": message.read_at,
    })
}

fn messages_actor_metadata(messages: &[CoordinationMessage]) -> Vec<Value> {
    messages.iter().map(message_actor_metadata).collect()
}

fn session_actor_metadata(session: &SessionPresence) -> Value {
    json!({
        "actor_id": session.session_id,
        "parent_actor_id": session.parent_session_id,
        "pid": session.pid,
        "cwd": session.cwd,
        "repo_root": session.repo_root,
        "model": session.model,
        "title": session.title,
        "started_at": session.started_at,
        "heartbeat_at": session.heartbeat_at,
    })
}

fn sessions_actor_metadata(sessions: &[SessionPresence]) -> Vec<Value> {
    sessions.iter().map(session_actor_metadata).collect()
}

fn claim_actor_metadata(claim: &WorkClaim) -> Value {
    json!({
        "claim_id": claim.claim_id,
        "actor_id": claim.session_id,
        "repo_root": claim.repo_root,
        "claim_type": claim.claim_type,
        "scope": claim.scope,
        "summary": claim.summary,
        "created_at": claim.created_at,
        "updated_at": claim.updated_at,
    })
}

fn claims_actor_metadata(claims: &[WorkClaim]) -> Vec<Value> {
    claims.iter().map(claim_actor_metadata).collect()
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
struct CoordinationInboxInput {
    #[serde(default = "default_inbox_limit")]
    limit: usize,
    #[serde(default = "default_true")]
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
    to_actor_id: Option<String>,
    #[serde(default)]
    to_session_id: Option<String>,
    #[serde(default = "default_true")]
    repo_broadcast: bool,
}

impl CoordinationMessageInput {
    fn target_actor(&self) -> Result<Option<&str>, &'static str> {
        let actor = normalize_optional_target(self.to_actor_id.as_deref());
        let legacy = normalize_optional_target(self.to_session_id.as_deref());
        match (actor, legacy) {
            (Some(actor), Some(legacy)) if actor != legacy => {
                Err("Set only one of to_actor_id or to_session_id for a direct message.")
            }
            (Some(actor), _) => Ok(Some(actor)),
            (_, Some(legacy)) => Ok(Some(legacy)),
            _ => Ok(None),
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_inbox_limit() -> usize {
    20
}

fn default_claim_type() -> String {
    "edit".to_string()
}

fn normalize_optional_target(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn resolve_message_route(
    to_actor_id: Option<&str>,
    repo_broadcast: bool,
) -> Result<(Option<&str>, bool), &'static str> {
    let target = normalize_optional_target(to_actor_id);
    if target.is_none() && !repo_broadcast {
        return Err("Set to_actor_id for a direct message, or leave repo_broadcast=true.");
    }
    Ok((target, target.is_none() && repo_broadcast))
}

fn resolve_direct_target(
    store: &CoordinationStore,
    ctx: &ToolContext,
    target: &str,
) -> Result<String, ToolResult> {
    match store.resolve_session_target(&ctx.working_dir, target) {
        Ok(SessionTargetResolution::Found(session)) => Ok(session.session_id),
        Ok(SessionTargetResolution::NotFound) => Err(ToolResult::error(format!(
            "No active MangoCode actor matched `{}` in this repo. Run CoordinationStatus to see active actor ids.",
            target
        ))),
        Ok(SessionTargetResolution::Stale(candidates)) => {
            let candidate_lines = candidates
                .iter()
                .take(8)
                .map(|session| {
                    format!(
                        "- {} heartbeat {}{}",
                        session.session_id,
                        session.heartbeat_at,
                        session
                            .title
                            .as_ref()
                            .map(|title| format!(" title `{title}`"))
                            .unwrap_or_default()
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            Err(ToolResult::error(format!(
                "`{}` only matches stale MangoCode actor(s). Ask the target to run CoordinationStatus or use a currently active actor id:\n{}",
                target, candidate_lines
            ))
            .with_metadata(json!({ "stale_candidates": candidates })))
        }
        Ok(SessionTargetResolution::Ambiguous(candidates)) => {
            let candidate_lines = candidates
                .iter()
                .take(8)
                .map(|session| {
                    format!(
                        "- {}{}",
                        session.session_id,
                        session
                            .title
                            .as_ref()
                            .map(|title| format!(" title `{title}`"))
                            .unwrap_or_default()
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            Err(ToolResult::error(format!(
                "`{}` matches multiple active MangoCode actors. Use one full actor id:\n{}",
                target, candidate_lines
            ))
            .with_metadata(json!({ "candidates": candidates })))
        }
        Err(e) => Err(ToolResult::error(format!(
            "Failed to resolve target actor: {e}"
        ))),
    }
}

pub fn send_coordination_message(
    ctx: &ToolContext,
    body: &str,
    to_actor_id: Option<&str>,
    repo_broadcast: bool,
) -> Result<CoordinationMessage, ToolResult> {
    let (target, repo_broadcast) = resolve_message_route(to_actor_id, repo_broadcast)
        .map_err(|e| ToolResult::error(e.to_string()))?;
    let store = open_store()
        .map_err(|e| ToolResult::error(format!("Failed to open coordination store: {e}")))?;
    let session_id = register_current_session_with_store(&store, ctx);
    let resolved_target = match target {
        Some(target) => Some(resolve_direct_target(&store, ctx, target)?),
        None => None,
    };
    let repo = if repo_broadcast {
        Some(ctx.working_dir.as_path())
    } else {
        None
    };
    store
        .send_message(&session_id, resolved_target.as_deref(), repo, body)
        .map_err(|e| ToolResult::error(format!("Failed to send message: {e}")))
}

pub struct CoordinationStatusTool;
pub struct CoordinationInboxTool;
pub struct ClaimWorkTool;
pub struct ReleaseWorkTool;
pub struct CoordinationMessageTool;

#[async_trait]
impl Tool for CoordinationStatusTool {
    fn name(&self) -> &str {
        "CoordinationStatus"
    }

    fn description(&self) -> &str {
        "List active local MangoCode actors, advisory work claims, and unread coordination messages."
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
        let session_id = register_current_session_with_store(&store, ctx);
        let repo_filter = (!params.all_repos).then_some(ctx.working_dir.as_path());
        let sessions = match store.list_sessions(repo_filter) {
            Ok(sessions) => sessions,
            Err(e) => return ToolResult::error(format!("Failed to list actors: {e}")),
        };
        let claims = match store.list_claims(repo_filter) {
            Ok(claims) => claims,
            Err(e) => return ToolResult::error(format!("Failed to list claims: {e}")),
        };
        let unread_count = match store.unread_count(&session_id, &ctx.working_dir) {
            Ok(count) => count,
            Err(e) => return ToolResult::error(format!("Failed to count inbox: {e}")),
        };
        let inbox = if params.include_inbox {
            match store.inbox_with_limit(&session_id, &ctx.working_dir, params.mark_read, 50) {
                Ok(messages) => messages,
                Err(e) => return ToolResult::error(format!("Failed to read inbox: {e}")),
            }
        } else {
            Vec::new()
        };
        let remaining_unread_count =
            if params.include_inbox && params.mark_read && !inbox.is_empty() {
                match store.unread_count(&session_id, &ctx.working_dir) {
                    Ok(count) => count,
                    Err(e) => return ToolResult::error(format!("Failed to count inbox: {e}")),
                }
            } else {
                unread_count
            };

        let unread_label = if params.include_inbox && params.mark_read {
            format!("{unread_count} before read, {remaining_unread_count} remaining")
        } else {
            unread_count.to_string()
        };
        let mut lines = vec![format!(
            "Active actors: {}. Active claims: {}. Unread messages: {}.",
            sessions.len(),
            claims.len(),
            unread_label
        )];
        for session in sessions.iter().take(8) {
            lines.push(format!(
                "- actor {} pid {} repo `{}`{}{}",
                session.session_id,
                session.pid,
                session.repo_root,
                session
                    .parent_session_id
                    .as_ref()
                    .map(|parent| format!(" parent {parent}"))
                    .unwrap_or_default(),
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
            let target = message
                .to_session_id
                .as_deref()
                .map(|target| format!("direct to {target}"))
                .unwrap_or_else(|| {
                    format!(
                        "repo broadcast{}",
                        message
                            .repo_root
                            .as_ref()
                            .map(|repo| format!(" for `{repo}`"))
                            .unwrap_or_default()
                    )
                });
            let read_state = message.read_at.as_deref().unwrap_or("unread");
            lines.push(format!(
                "- message {target} from actor {} at {} ({read_state}): {}",
                message.from_session_id, message.created_at, message.body
            ));
        }
        if unread_count > inbox.len() {
            if params.include_inbox {
                lines.push(format!(
                    "Showing {} of {} unread message(s).",
                    inbox.len(),
                    unread_count
                ));
            } else {
                lines.push(format!(
                    "Inbox hidden; {} unread message(s) available.",
                    unread_count
                ));
            }
        }

        let actors = sessions_actor_metadata(&sessions);
        let actor_claims = claims_actor_metadata(&claims);
        let actor_inbox = messages_actor_metadata(&inbox);
        ToolResult::success(lines.join("\n")).with_metadata(json!({
            "actors": actors,
            "sessions": sessions,
            "actor_claims": actor_claims,
            "claims": claims,
            "unread_count": unread_count,
            "remaining_unread_count": remaining_unread_count,
            "actor_inbox": actor_inbox,
            "inbox": inbox,
        }))
    }
}

#[async_trait]
impl Tool for CoordinationInboxTool {
    fn name(&self) -> &str {
        "CoordinationInbox"
    }

    fn aliases(&self) -> Vec<String> {
        vec![
            "ReceiveMessages".to_string(),
            "coordination_inbox".to_string(),
        ]
    }

    fn description(&self) -> &str {
        "Read unread local coordination messages for this MangoCode actor, optionally marking them read."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Maximum unread messages to return. Defaults to 20."
                },
                "mark_read": {
                    "type": "boolean",
                    "description": "Mark returned messages as read. Defaults to true."
                }
            },
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: CoordinationInboxInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {e}")),
        };
        let store = match open_store() {
            Ok(store) => store,
            Err(e) => return ToolResult::error(format!("Failed to open coordination store: {e}")),
        };
        let session_id = register_current_session_with_store(&store, ctx);
        let unread_count = match store.unread_count(&session_id, &ctx.working_dir) {
            Ok(count) => count,
            Err(e) => return ToolResult::error(format!("Failed to count inbox: {e}")),
        };
        let limit = params.limit.clamp(1, 200);
        let inbox =
            match store.inbox_with_limit(&session_id, &ctx.working_dir, params.mark_read, limit) {
                Ok(messages) => messages,
                Err(e) => return ToolResult::error(format!("Failed to read inbox: {e}")),
            };
        let remaining_unread_count = if params.mark_read && !inbox.is_empty() {
            match store.unread_count(&session_id, &ctx.working_dir) {
                Ok(count) => count,
                Err(e) => return ToolResult::error(format!("Failed to count inbox: {e}")),
            }
        } else {
            unread_count
        };

        let mut lines = if params.mark_read {
            vec![format!(
                "Coordination inbox for {}: {} unread message(s) before read, returning {}; {} remaining.",
                short_id(&session_id),
                unread_count,
                inbox.len(),
                remaining_unread_count
            )]
        } else {
            vec![format!(
                "Coordination inbox for {}: {} unread message(s), returning {}.",
                short_id(&session_id),
                unread_count,
                inbox.len()
            )]
        };
        for message in inbox.iter().take(limit) {
            let target = message
                .to_session_id
                .as_deref()
                .map(|target| format!("direct to {target}"))
                .unwrap_or_else(|| {
                    format!(
                        "repo broadcast{}",
                        message
                            .repo_root
                            .as_ref()
                            .map(|repo| format!(" for `{repo}`"))
                            .unwrap_or_default()
                    )
                });
            let read_state = message.read_at.as_deref().unwrap_or("unread");
            lines.push(format!(
                "- [{target}] from actor {} at {} ({read_state}): {}",
                message.from_session_id, message.created_at, message.body
            ));
        }
        if params.mark_read && !inbox.is_empty() {
            lines.push("Marked returned messages as read.".to_string());
        }

        ToolResult::success(lines.join("\n")).with_metadata(json!({
            "actor_id": session_id,
            "unread_count": unread_count,
            "remaining_unread_count": remaining_unread_count,
            "actor_messages": messages_actor_metadata(&inbox),
            "messages": inbox,
            "mark_read": params.mark_read,
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
        let session_id = register_current_session_with_store(&store, ctx);
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
        let session_id = register_current_session_with_store(&store, ctx);
        match store.release_claims(&session_id, params.claim_id.as_deref()) {
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
        "Send a local coordination message to one MangoCode actor or broadcast to the current repo."
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
                "to_actor_id": {
                    "type": "string",
                    "description": "Optional target actor id, unique actor-id prefix, or title. Omit for repo broadcast."
                },
                "to_session_id": {
                    "type": "string",
                    "description": "Deprecated alias for to_actor_id. Kept for compatibility with older callers."
                },
                "repo_broadcast": {
                    "type": "boolean",
                    "description": "When no target is set, broadcast to active actors in this repo."
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
        match send_coordination_message(
            ctx,
            &params.body,
            match params.target_actor() {
                Ok(target) => target,
                Err(e) => return ToolResult::error(e.to_string()),
            },
            params.repo_broadcast,
        ) {
            Ok(message) => ToolResult::success(format!(
                "Coordination message queued as {}.",
                short_id(&message.message_id)
            ))
            .with_metadata(json!({
                "actor_message": message_actor_metadata(&message),
                "message": message,
            })),
            Err(result) => result,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn test_context(root: &std::path::Path) -> ToolContext {
        ToolContext {
            working_dir: root.to_path_buf(),
            permission_mode: mangocode_core::config::PermissionMode::BypassPermissions,
            permission_handler: Arc::new(mangocode_core::permissions::AutoPermissionHandler {
                mode: mangocode_core::config::PermissionMode::BypassPermissions,
            }),
            cost_tracker: mangocode_core::cost::CostTracker::new(),
            session_metrics: None,
            session_id: "coordination-tool-test".to_string(),
            coordination_actor_id: None,
            coordination_parent_actor_id: None,
            inject_coordination_inbox: true,
            file_history: Arc::new(parking_lot::Mutex::new(
                mangocode_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(std::sync::atomic::AtomicUsize::new(1)),
            non_interactive: true,
            mcp_manager: None,
            config: mangocode_core::config::Config::default(),
        }
    }

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

    #[test]
    fn message_actor_metadata_exposes_actor_named_fields() {
        let message = CoordinationMessage {
            message_id: "message-1".to_string(),
            from_session_id: "actor-a".to_string(),
            to_session_id: Some("actor-b".to_string()),
            repo_root: None,
            body: "hello".to_string(),
            created_at: "2026-05-16T00:00:00Z".to_string(),
            read_at: Some("2026-05-16T00:00:01Z".to_string()),
        };

        let metadata = message_actor_metadata(&message);
        assert_eq!(metadata["message_id"], "message-1");
        assert_eq!(metadata["from_actor_id"], "actor-a");
        assert_eq!(metadata["to_actor_id"], "actor-b");
        assert_eq!(metadata["body"], "hello");
        assert_eq!(metadata["read_at"], "2026-05-16T00:00:01Z");
    }

    #[test]
    fn status_metadata_exposes_actor_named_fields() {
        let session = SessionPresence {
            session_id: "actor-a".to_string(),
            parent_session_id: Some("parent-a".to_string()),
            pid: 42,
            cwd: "/repo".to_string(),
            repo_root: "/repo".to_string(),
            model: "model-a".to_string(),
            title: Some("review".to_string()),
            started_at: "2026-05-16T00:00:00Z".to_string(),
            heartbeat_at: "2026-05-16T00:00:01Z".to_string(),
        };
        let actor = session_actor_metadata(&session);
        assert_eq!(actor["actor_id"], "actor-a");
        assert_eq!(actor["parent_actor_id"], "parent-a");
        assert_eq!(actor["title"], "review");

        let claim = WorkClaim {
            claim_id: "claim-a".to_string(),
            session_id: "actor-a".to_string(),
            repo_root: "/repo".to_string(),
            claim_type: "edit".to_string(),
            scope: "src/lib.rs".to_string(),
            summary: Some("editing".to_string()),
            created_at: "2026-05-16T00:00:00Z".to_string(),
            updated_at: "2026-05-16T00:00:01Z".to_string(),
        };
        let actor_claim = claim_actor_metadata(&claim);
        assert_eq!(actor_claim["claim_id"], "claim-a");
        assert_eq!(actor_claim["actor_id"], "actor-a");
        assert_eq!(actor_claim["scope"], "src/lib.rs");
    }

    #[test]
    fn coordination_message_input_prefers_actor_id_and_keeps_legacy_alias() {
        let preferred = CoordinationMessageInput {
            body: "hello".to_string(),
            to_actor_id: Some(" actor-a ".to_string()),
            to_session_id: None,
            repo_broadcast: false,
        };
        assert_eq!(preferred.target_actor().unwrap(), Some("actor-a"));

        let legacy = CoordinationMessageInput {
            body: "hello".to_string(),
            to_actor_id: None,
            to_session_id: Some(" legacy-a ".to_string()),
            repo_broadcast: false,
        };
        assert_eq!(legacy.target_actor().unwrap(), Some("legacy-a"));

        let matching = CoordinationMessageInput {
            body: "hello".to_string(),
            to_actor_id: Some("same-a".to_string()),
            to_session_id: Some(" same-a ".to_string()),
            repo_broadcast: false,
        };
        assert_eq!(matching.target_actor().unwrap(), Some("same-a"));

        let conflicting = CoordinationMessageInput {
            body: "hello".to_string(),
            to_actor_id: Some("actor-a".to_string()),
            to_session_id: Some("legacy-a".to_string()),
            repo_broadcast: false,
        };
        assert!(conflicting.target_actor().is_err());
    }

    #[test]
    fn coordination_message_schema_exposes_actor_targeting_contract() {
        let tool = CoordinationMessageTool;
        assert!(tool.description().contains("MangoCode actor"));
        let schema = tool.input_schema();
        let target_description = schema["properties"]["to_actor_id"]["description"]
            .as_str()
            .unwrap_or_default();
        assert!(target_description.contains("actor id"));
        assert!(target_description.contains("title"));
        let legacy_description = schema["properties"]["to_session_id"]["description"]
            .as_str()
            .unwrap_or_default();
        assert!(legacy_description.contains("Deprecated alias"));
    }

    #[test]
    fn direct_target_resolution_returns_candidates_for_ambiguity() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        let store = CoordinationStore::open(&dir.path().join("coordination.db")).unwrap();
        store
            .register_session("actor-alpha", dir.path(), "model", Some("review"))
            .unwrap();
        store
            .register_session("actor-beta", dir.path(), "model", Some("reviewer"))
            .unwrap();
        let ctx = test_context(dir.path());

        assert_eq!(
            resolve_direct_target(&store, &ctx, "actor-beta").unwrap(),
            "actor-beta"
        );
        assert_eq!(
            resolve_direct_target(&store, &ctx, "actor-a").unwrap(),
            "actor-alpha"
        );

        let ambiguous = resolve_direct_target(&store, &ctx, "review").unwrap_err();
        assert!(ambiguous.content.contains("matches multiple"));
        assert!(ambiguous.metadata.is_some());

        let missing = resolve_direct_target(&store, &ctx, "missing").unwrap_err();
        assert!(missing.content.contains("No active MangoCode actor"));
    }
}
