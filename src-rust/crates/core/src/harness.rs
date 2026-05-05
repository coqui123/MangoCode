//! Durable local harness events, checkpoints, and trace files.
//!
//! This module is intentionally local-first: events are appended to the
//! existing `~/.mangocode/sessions.db` SQLite database and mirrored to JSONL
//! trace files under `~/.mangocode/traces/`. Network export is handled only by
//! the optional OpenTelemetry layer elsewhere.

use crate::config::Settings;
use crate::git_utils;
use crate::SqliteSessionStore;
use crate::{PermissionDecision, PermissionRequest};
use chrono::{DateTime, Utc};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use tokio::sync::broadcast;
use uuid::Uuid;

const EVENT_BUFFER_SIZE: usize = 1024;

static EVENT_BUS: Lazy<broadcast::Sender<HarnessEvent>> = Lazy::new(|| {
    let (tx, _rx) = broadcast::channel(EVENT_BUFFER_SIZE);
    tx
});

static ACTIVE_TURNS: Lazy<Mutex<HashMap<String, Vec<ActiveTurn>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Debug, Clone)]
struct ActiveTurn {
    turn_id: String,
    recorder: HarnessRecorder,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessEvent {
    pub event_id: String,
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_id: Option<String>,
    pub event_type: String,
    pub timestamp: DateTime<Utc>,
    pub payload: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessCheckpoint {
    pub checkpoint_id: String,
    pub session_id: String,
    pub turn_id: String,
    pub kind: CheckpointKind,
    pub backend: CheckpointBackend,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repo_root: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_ref: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<Value>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointKind {
    Before,
    After,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointBackend {
    GitRef,
    FileSnapshot,
    MetadataOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HarnessTurnStatus {
    Started,
    Completed,
    Cancelled,
    Failed,
}

#[derive(Clone)]
pub struct HarnessRecorder {
    session_id: String,
    store: Arc<Mutex<Option<SqliteSessionStore>>>,
}

impl std::fmt::Debug for HarnessRecorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HarnessRecorder")
            .field("session_id", &self.session_id)
            .finish_non_exhaustive()
    }
}

impl HarnessRecorder {
    pub fn new(session_id: impl Into<String>) -> Self {
        let session_id = session_id.into();
        let db_path = default_db_path();
        if let Some(parent) = db_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let store = SqliteSessionStore::open(&db_path).ok();
        Self {
            session_id,
            store: Arc::new(Mutex::new(store)),
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn record(
        &self,
        event_type: impl Into<String>,
        turn_id: Option<String>,
        tool_call_id: Option<String>,
        checkpoint_id: Option<String>,
        payload: Value,
    ) -> HarnessEvent {
        let event = HarnessEvent {
            event_id: Uuid::new_v4().to_string(),
            session_id: self.session_id.clone(),
            turn_id,
            tool_call_id,
            checkpoint_id,
            event_type: event_type.into(),
            timestamp: Utc::now(),
            payload,
        };

        if let Some(store) = self.store.lock().as_ref() {
            let _ = store.append_harness_event(&event);
        }
        append_trace_jsonl(&event);
        let _ = EVENT_BUS.send(event.clone());
        event
    }

    pub fn record_turn_status(
        &self,
        turn_id: &str,
        status: HarnessTurnStatus,
        model: Option<&str>,
        cwd: Option<&Path>,
        detail: Option<Value>,
    ) {
        let now = Utc::now();
        if let Some(store) = self.store.lock().as_ref() {
            let _ = store.upsert_harness_turn(&self.session_id, turn_id, status, model, cwd, now);
        }
        self.record(
            format!(
                "turn.{}",
                match status {
                    HarnessTurnStatus::Started => "started",
                    HarnessTurnStatus::Completed => "completed",
                    HarnessTurnStatus::Cancelled => "cancelled",
                    HarnessTurnStatus::Failed => "failed",
                }
            ),
            Some(turn_id.to_string()),
            None,
            None,
            detail.unwrap_or_else(|| json!({})),
        );
    }

    pub fn capture_checkpoint(
        &self,
        turn_id: &str,
        kind: CheckpointKind,
        cwd: &Path,
        fallback_snapshot: Option<Value>,
    ) -> anyhow::Result<HarnessCheckpoint> {
        let span = crate::session_tracing::start_checkpoint_span(
            &self.session_id,
            turn_id,
            checkpoint_kind_string(kind),
        );
        let checkpoint =
            match capture_checkpoint(&self.session_id, turn_id, kind, cwd, fallback_snapshot) {
                Ok(checkpoint) => checkpoint,
                Err(err) => {
                    span.record_exception(&err.to_string());
                    span.end();
                    return Err(err);
                }
            };
        crate::session_tracing::end_checkpoint_span(
            span,
            &checkpoint.checkpoint_id,
            checkpoint_backend_string(checkpoint.backend),
        );
        if let Some(store) = self.store.lock().as_ref() {
            let _ = store.save_harness_checkpoint(&checkpoint);
        }
        self.record(
            match kind {
                CheckpointKind::Before => "checkpoint.before",
                CheckpointKind::After => "checkpoint.after",
            },
            Some(turn_id.to_string()),
            None,
            Some(checkpoint.checkpoint_id.clone()),
            json!({
                "backend": checkpoint.backend,
                "repo_root": checkpoint.repo_root,
                "git_ref": checkpoint.git_ref,
            }),
        );
        Ok(checkpoint)
    }

    pub fn update_checkpoint_snapshot(
        &self,
        checkpoint: &HarnessCheckpoint,
        snapshot: Value,
    ) -> anyhow::Result<HarnessCheckpoint> {
        let mut updated = checkpoint.clone();
        updated.backend = CheckpointBackend::FileSnapshot;
        updated.snapshot = Some(snapshot);
        if let Some(store) = self.store.lock().as_ref() {
            store.save_harness_checkpoint(&updated)?;
        }
        self.record(
            "checkpoint.updated",
            Some(updated.turn_id.clone()),
            None,
            Some(updated.checkpoint_id.clone()),
            json!({
                "backend": updated.backend,
                "repo_root": updated.repo_root,
                "git_ref": updated.git_ref,
            }),
        );
        Ok(updated)
    }

    pub fn events_for_turn(&self, turn_id: &str) -> Vec<HarnessEvent> {
        let store_guard = self.store.lock();
        let Some(store) = store_guard.as_ref() else {
            return Vec::new();
        };
        store
            .list_harness_events(&self.session_id, None, 10_000)
            .unwrap_or_default()
            .into_iter()
            .map(|(_, event)| event)
            .filter(|event| event.turn_id.as_deref() == Some(turn_id))
            .collect()
    }

    pub fn record_file_change_for_turn(
        &self,
        turn_id: &str,
        tool_call_id: Option<&str>,
        path: &Path,
        before_content: Option<&[u8]>,
        after_content: Option<&[u8]>,
        tool_name: &str,
    ) {
        let before_text = before_content.and_then(|bytes| String::from_utf8(bytes.to_vec()).ok());
        let after_text = after_content.and_then(|bytes| String::from_utf8(bytes.to_vec()).ok());
        let binary = before_content.is_some() && before_text.is_none()
            || after_content.is_some() && after_text.is_none();
        self.record(
            "file.changed",
            Some(turn_id.to_string()),
            tool_call_id.map(ToOwned::to_owned),
            None,
            json!({
                "path": path,
                "tool_name": tool_name,
                "existed": before_content.is_some(),
                "after_existed": after_content.is_some(),
                "before_hash": before_content.map(sha256_hex),
                "after_hash": after_content.map(sha256_hex),
                "before_text": before_text,
                "after_text": after_text,
                "binary": binary,
            }),
        );
    }

    pub fn record_tool_snapshot_before_for_turn(
        &self,
        turn_id: &str,
        tool_call_id: &str,
        path: &Path,
        before_content: Option<&[u8]>,
    ) {
        let before_text = before_content.and_then(|bytes| String::from_utf8(bytes.to_vec()).ok());
        let binary = before_content.is_some() && before_text.is_none();
        let before_base64 = if binary {
            before_content.map(|bytes| {
                use base64::Engine as _;
                base64::engine::general_purpose::STANDARD.encode(bytes)
            })
        } else {
            None
        };
        self.record(
            "tool.snapshot_before",
            Some(turn_id.to_string()),
            Some(tool_call_id.to_string()),
            None,
            json!({
                "path": path,
                "existed": before_content.is_some(),
                "binary": binary,
                "before_text": before_text,
                "before_base64": before_base64,
            }),
        );
    }
}

pub fn subscribe_events() -> broadcast::Receiver<HarnessEvent> {
    EVENT_BUS.subscribe()
}

pub fn start_turn(
    session_id: &str,
    model: Option<&str>,
    cwd: Option<&Path>,
    detail: Option<Value>,
) -> (String, HarnessRecorder) {
    let turn_id = Uuid::new_v4().to_string();
    let recorder = HarnessRecorder::new(session_id);
    recorder.record_turn_status(&turn_id, HarnessTurnStatus::Started, model, cwd, detail);
    ACTIVE_TURNS
        .lock()
        .entry(session_id.to_string())
        .or_default()
        .push(ActiveTurn {
            turn_id: turn_id.clone(),
            recorder: recorder.clone(),
        });
    (turn_id, recorder)
}

pub fn finish_turn(session_id: &str, turn_id: &str) {
    let mut active = ACTIVE_TURNS.lock();
    if let Some(stack) = active.get_mut(session_id) {
        if let Some(idx) = stack
            .iter()
            .rposition(|active_turn| active_turn.turn_id == turn_id)
        {
            stack.remove(idx);
        }
        if stack.is_empty() {
            active.remove(session_id);
        }
    }
}

pub fn active_turn_id(session_id: &str) -> Option<String> {
    ACTIVE_TURNS
        .lock()
        .get(session_id)
        .and_then(|stack| stack.last())
        .map(|active| active.turn_id.clone())
}

pub fn active_recorder(session_id: &str) -> Option<HarnessRecorder> {
    ACTIVE_TURNS
        .lock()
        .get(session_id)
        .and_then(|stack| stack.last())
        .map(|active| active.recorder.clone())
}

pub fn record_file_change(
    session_id: &str,
    path: &Path,
    before_content: &[u8],
    after_content: &[u8],
    tool_name: &str,
) {
    record_file_change_with_tool(
        session_id,
        None,
        path,
        Some(before_content),
        after_content,
        tool_name,
    );
}

pub fn record_file_change_with_tool(
    session_id: &str,
    tool_call_id: Option<&str>,
    path: &Path,
    before_content: Option<&[u8]>,
    after_content: &[u8],
    tool_name: &str,
) {
    let Some(active) = ACTIVE_TURNS.lock().get(session_id).cloned() else {
        return;
    };
    let Some(active) = active.last() else {
        return;
    };
    active.recorder.record_file_change_for_turn(
        &active.turn_id,
        tool_call_id,
        path,
        before_content,
        Some(after_content),
        tool_name,
    );
}

pub fn record_tool_snapshot_before(
    session_id: &str,
    tool_call_id: &str,
    path: &Path,
    before_content: Option<&[u8]>,
) {
    let Some(active) = ACTIVE_TURNS.lock().get(session_id).cloned() else {
        return;
    };
    let Some(active) = active.last() else {
        return;
    };
    active.recorder.record_tool_snapshot_before_for_turn(
        &active.turn_id,
        tool_call_id,
        path,
        before_content,
    );
}

pub fn record_permission_request(session_id: &str, request: &PermissionRequest) {
    let Some(active) = ACTIVE_TURNS.lock().get(session_id).cloned() else {
        return;
    };
    let Some(active) = active.last() else {
        return;
    };
    active.recorder.record(
        "permission.requested",
        Some(active.turn_id.clone()),
        None,
        None,
        json!({
            "tool_name": request.tool_name,
            "description": request.description,
            "details": request.details,
            "is_read_only": request.is_read_only,
            "context_description": request.context_description,
        }),
    );
}

pub fn record_permission_decision(
    session_id: &str,
    request: &PermissionRequest,
    decision: &PermissionDecision,
) {
    let Some(active) = ACTIVE_TURNS.lock().get(session_id).cloned() else {
        return;
    };
    let Some(active) = active.last() else {
        return;
    };
    active.recorder.record(
        "permission.decision",
        Some(active.turn_id.clone()),
        None,
        None,
        json!({
            "tool_name": request.tool_name,
            "description": request.description,
            "details": request.details,
            "is_read_only": request.is_read_only,
            "context_description": request.context_description,
            "decision": permission_decision_string(decision),
            "reason": permission_decision_reason(decision),
        }),
    );
}

pub fn restore_tool_snapshot(
    session_id: &str,
    tool_call_id: &str,
) -> anyhow::Result<(Vec<String>, Vec<String>)> {
    let store = SqliteSessionStore::open(&default_db_path())?;
    let events = store.list_harness_events(session_id, None, 10_000)?;
    let mut reverted = Vec::new();
    let mut errors = Vec::new();

    for (_, event) in events
        .into_iter()
        .filter(|(_, event)| {
            event.event_type == "tool.snapshot_before"
                && event.tool_call_id.as_deref() == Some(tool_call_id)
        })
        .rev()
    {
        let Some(path_str) = event.payload.get("path").and_then(Value::as_str) else {
            continue;
        };
        let path = Path::new(path_str);
        if let Some(content) = event.payload.get("before_text").and_then(Value::as_str) {
            if let Some(parent) = path.parent() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    errors.push(format!("Failed to create {}: {}", parent.display(), e));
                    continue;
                }
            }
            match std::fs::write(path, content) {
                Ok(_) => reverted.push(path.display().to_string()),
                Err(e) => errors.push(format!("Failed to restore {}: {}", path.display(), e)),
            }
            continue;
        }

        if let Some(encoded) = event.payload.get("before_base64").and_then(Value::as_str) {
            use base64::Engine as _;
            match base64::engine::general_purpose::STANDARD.decode(encoded) {
                Ok(bytes) => {
                    if let Some(parent) = path.parent() {
                        if let Err(e) = std::fs::create_dir_all(parent) {
                            errors.push(format!("Failed to create {}: {}", parent.display(), e));
                            continue;
                        }
                    }
                    match std::fs::write(path, bytes) {
                        Ok(_) => reverted.push(path.display().to_string()),
                        Err(e) => {
                            errors.push(format!("Failed to restore {}: {}", path.display(), e))
                        }
                    }
                }
                Err(e) => errors.push(format!("Failed to decode {}: {}", path.display(), e)),
            }
            continue;
        }

        match event.payload.get("existed").and_then(Value::as_bool) {
            Some(true) => errors.push(format!(
                "No restorable content recorded for existing file {}",
                path.display()
            )),
            _ => {
                if path.exists() {
                    match std::fs::remove_file(path) {
                        Ok(_) => reverted.push(path.display().to_string()),
                        Err(e) => {
                            errors.push(format!("Failed to delete {}: {}", path.display(), e))
                        }
                    }
                } else {
                    reverted.push(path.display().to_string());
                }
            }
        }
    }

    Ok((reverted, errors))
}

pub fn list_checkpoints(session_id: &str) -> anyhow::Result<Vec<HarnessCheckpoint>> {
    let store = SqliteSessionStore::open(&default_db_path())?;
    store.list_harness_checkpoints(session_id)
}

pub fn diff_checkpoints(before_id: &str, after_id: &str) -> anyhow::Result<String> {
    let store = SqliteSessionStore::open(&default_db_path())?;
    let before = store.get_harness_checkpoint(before_id)?;
    let after = store.get_harness_checkpoint(after_id)?;

    if before.backend == CheckpointBackend::GitRef
        && after.backend == CheckpointBackend::GitRef
        && before.repo_root == after.repo_root
    {
        let repo_root = before
            .repo_root
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Git checkpoint is missing repo root"))?;
        let before_ref = before
            .git_ref
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Git checkpoint is missing ref"))?;
        let after_ref = after
            .git_ref
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Git checkpoint is missing ref"))?;
        return git_output(repo_root, &["diff", before_ref, after_ref]);
    }

    Ok(serde_json::to_string_pretty(&json!({
        "before": before,
        "after": after,
        "note": "Non-Git checkpoint diff is represented as persisted snapshot metadata."
    }))?)
}

pub fn restore_checkpoint(checkpoint_id: &str) -> anyhow::Result<()> {
    let store = SqliteSessionStore::open(&default_db_path())?;
    let checkpoint = store.get_harness_checkpoint(checkpoint_id)?;
    match checkpoint.backend {
        CheckpointBackend::GitRef => {
            let repo_root = checkpoint
                .repo_root
                .ok_or_else(|| anyhow::anyhow!("Git checkpoint is missing repo root"))?;
            let git_ref = checkpoint
                .git_ref
                .ok_or_else(|| anyhow::anyhow!("Git checkpoint is missing ref"))?;
            restore_git_checkpoint(&repo_root, &git_ref)
        }
        CheckpointBackend::FileSnapshot => {
            restore_file_snapshot(checkpoint.kind, checkpoint.snapshot)
        }
        CheckpointBackend::MetadataOnly => Err(anyhow::anyhow!(
            "Checkpoint {} has no restorable filesystem snapshot",
            checkpoint_id
        )),
    }
}

fn capture_checkpoint(
    session_id: &str,
    turn_id: &str,
    kind: CheckpointKind,
    cwd: &Path,
    fallback_snapshot: Option<Value>,
) -> anyhow::Result<HarnessCheckpoint> {
    if let Some(repo_root) = git_utils::get_repo_root(cwd) {
        match capture_git_checkpoint(session_id, turn_id, kind, &repo_root) {
            Ok((git_ref, checkpoint_id)) => {
                return Ok(HarnessCheckpoint {
                    checkpoint_id,
                    session_id: session_id.to_string(),
                    turn_id: turn_id.to_string(),
                    kind,
                    backend: CheckpointBackend::GitRef,
                    repo_root: Some(repo_root),
                    git_ref: Some(git_ref),
                    snapshot: None,
                    created_at: Utc::now(),
                });
            }
            Err(err) => {
                tracing::debug!(error = %err, "Git checkpoint capture failed; using fallback");
            }
        }
    }

    let backend = if fallback_snapshot.is_some() {
        CheckpointBackend::FileSnapshot
    } else {
        CheckpointBackend::MetadataOnly
    };
    Ok(HarnessCheckpoint {
        checkpoint_id: Uuid::new_v4().to_string(),
        session_id: session_id.to_string(),
        turn_id: turn_id.to_string(),
        kind,
        backend,
        repo_root: None,
        git_ref: None,
        snapshot: fallback_snapshot,
        created_at: Utc::now(),
    })
}

fn capture_git_checkpoint(
    session_id: &str,
    _turn_id: &str,
    _kind: CheckpointKind,
    repo_root: &Path,
) -> anyhow::Result<(String, String)> {
    let checkpoint_id = Uuid::new_v4().to_string();
    let safe_session = sanitize_ref_component(session_id);
    let git_ref = format!(
        "refs/mangocode/checkpoints/{}/{}",
        safe_session, checkpoint_id
    );
    let index_path =
        std::env::temp_dir().join(format!("mangocode-checkpoint-{}.index", checkpoint_id));

    let tree = capture_git_tree(repo_root, &index_path)?;
    run_git(repo_root, &["update-ref", &git_ref, &tree])?;
    let _ = std::fs::remove_file(index_path);
    Ok((git_ref, checkpoint_id))
}

fn capture_git_tree(repo_root: &Path, index_path: &Path) -> anyhow::Result<String> {
    let has_head = git_status(repo_root, &["rev-parse", "--verify", "HEAD"]);
    if has_head {
        run_git_with_index(repo_root, &index_path, &["read-tree", "HEAD"])?;
    } else {
        run_git_with_index(repo_root, &index_path, &["read-tree", "--empty"])?;
    }
    run_git_with_index(repo_root, &index_path, &["add", "-A", "--", "."])?;
    git_output_with_index(repo_root, &index_path, &["write-tree"])
}

fn restore_git_checkpoint(repo_root: &Path, git_ref: &str) -> anyhow::Result<()> {
    let checkpoint_id = Uuid::new_v4().to_string();
    let index_path =
        std::env::temp_dir().join(format!("mangocode-restore-{}.index", checkpoint_id));
    let current_tree = capture_git_tree(repo_root, &index_path)?;
    let _ = std::fs::remove_file(index_path);

    let added = git_output_bytes(
        repo_root,
        &[
            "diff",
            "--name-only",
            "-z",
            "--no-renames",
            "--diff-filter=A",
            git_ref,
            &current_tree,
        ],
    )?;
    for rel_path in nul_paths(&added) {
        let path = safe_repo_path(repo_root, &rel_path)?;
        if let Ok(metadata) = std::fs::symlink_metadata(&path) {
            let file_type = metadata.file_type();
            if file_type.is_dir() && !file_type.is_symlink() {
                std::fs::remove_dir_all(&path)?;
            } else {
                std::fs::remove_file(&path)?;
            }
        }
    }

    run_git(
        repo_root,
        &["restore", "--source", git_ref, "--worktree", "--", "."],
    )
}

fn restore_file_snapshot(kind: CheckpointKind, snapshot: Option<Value>) -> anyhow::Result<()> {
    let Some(Value::Array(entries)) = snapshot else {
        return Err(anyhow::anyhow!("File snapshot checkpoint has no entries"));
    };

    for entry in entries {
        let path = entry
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("Snapshot entry missing path"))?;
        let (exists_key, text_key, base64_key) = match kind {
            CheckpointKind::Before => ("existed", "before_text", "before_base64"),
            CheckpointKind::After => ("after_existed", "after_text", "after_base64"),
        };
        let should_exist = entry
            .get(exists_key)
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let content = entry.get(text_key).and_then(Value::as_str);
        let encoded = entry.get(base64_key).and_then(Value::as_str);
        let path = Path::new(path);
        if !should_exist {
            if path.exists() {
                std::fs::remove_file(path)?;
            }
            continue;
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if let Some(content) = content {
            std::fs::write(path, content)?;
        } else if let Some(encoded) = encoded {
            use base64::Engine as _;
            let bytes = base64::engine::general_purpose::STANDARD.decode(encoded)?;
            std::fs::write(path, bytes)?;
        }
    }
    Ok(())
}

fn append_trace_jsonl(event: &HarnessEvent) {
    let Some(dir) = trace_dir() else {
        return;
    };
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join(format!("{}.jsonl", event.session_id));
    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) else {
        return;
    };
    if let Ok(line) = serde_json::to_string(event) {
        let _ = writeln!(file, "{}", line);
    }
}

fn default_db_path() -> PathBuf {
    Settings::config_dir().join("sessions.db")
}

fn trace_dir() -> Option<PathBuf> {
    Some(Settings::config_dir().join("traces"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

fn sanitize_ref_component(input: &str) -> String {
    input
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn checkpoint_kind_string(kind: CheckpointKind) -> &'static str {
    match kind {
        CheckpointKind::Before => "before",
        CheckpointKind::After => "after",
    }
}

fn checkpoint_backend_string(backend: CheckpointBackend) -> &'static str {
    match backend {
        CheckpointBackend::GitRef => "git_ref",
        CheckpointBackend::FileSnapshot => "file_snapshot",
        CheckpointBackend::MetadataOnly => "metadata_only",
    }
}

fn permission_decision_string(decision: &PermissionDecision) -> &'static str {
    match decision {
        PermissionDecision::Allow => "allow",
        PermissionDecision::AllowPermanently => "allow_permanently",
        PermissionDecision::Deny => "deny",
        PermissionDecision::DenyPermanently => "deny_permanently",
        PermissionDecision::Ask { .. } => "ask",
    }
}

fn permission_decision_reason(decision: &PermissionDecision) -> Option<&str> {
    match decision {
        PermissionDecision::Ask { reason } => Some(reason.as_str()),
        _ => None,
    }
}

fn git_status(repo_root: &Path, args: &[&str]) -> bool {
    Command::new("git")
        .current_dir(repo_root)
        .args(args)
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn run_git(repo_root: &Path, args: &[&str]) -> anyhow::Result<()> {
    let output = Command::new("git")
        .current_dir(repo_root)
        .args(args)
        .output()?;
    if output.status.success() {
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

fn git_output(repo_root: &Path, args: &[&str]) -> anyhow::Result<String> {
    let output = Command::new("git")
        .current_dir(repo_root)
        .args(args)
        .output()?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        Err(anyhow::anyhow!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

fn git_output_bytes(repo_root: &Path, args: &[&str]) -> anyhow::Result<Vec<u8>> {
    let output = Command::new("git")
        .current_dir(repo_root)
        .args(args)
        .output()?;
    if output.status.success() {
        Ok(output.stdout)
    } else {
        Err(anyhow::anyhow!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

fn nul_paths(bytes: &[u8]) -> Vec<String> {
    bytes
        .split(|byte| *byte == 0)
        .filter(|part| !part.is_empty())
        .map(|part| String::from_utf8_lossy(part).to_string())
        .collect()
}

fn safe_repo_path(repo_root: &Path, rel_path: &str) -> anyhow::Result<PathBuf> {
    let path = Path::new(rel_path);
    if path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::Prefix(_)))
    {
        return Err(anyhow::anyhow!(
            "Refusing to restore path outside repo: {}",
            rel_path
        ));
    }
    Ok(repo_root.join(path))
}

fn run_git_with_index(repo_root: &Path, index_path: &Path, args: &[&str]) -> anyhow::Result<()> {
    let output = Command::new("git")
        .current_dir(repo_root)
        .env("GIT_INDEX_FILE", index_path)
        .args(args)
        .output()?;
    if output.status.success() {
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

fn git_output_with_index(
    repo_root: &Path,
    index_path: &Path,
    args: &[&str],
) -> anyhow::Result<String> {
    let output = Command::new("git")
        .current_dir(repo_root)
        .env("GIT_INDEX_FILE", index_path)
        .args(args)
        .output()?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        Err(anyhow::anyhow!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_ids_are_populated() {
        let event = HarnessEvent {
            event_id: Uuid::new_v4().to_string(),
            session_id: "test-session".to_string(),
            turn_id: Some("turn".to_string()),
            tool_call_id: None,
            checkpoint_id: None,
            event_type: "test.event".to_string(),
            timestamp: Utc::now(),
            payload: json!({}),
        };
        assert!(!event.event_id.is_empty());
        assert_eq!(event.session_id, "test-session");
        assert_eq!(event.turn_id.as_deref(), Some("turn"));
    }

    #[test]
    fn ref_component_is_sanitized() {
        assert_eq!(sanitize_ref_component("a/b c"), "a_b_c");
    }

    #[test]
    fn file_snapshot_restore_respects_checkpoint_kind() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("created.txt");
        std::fs::write(&path, "after").unwrap();
        let snapshot = json!([{
            "path": path,
            "existed": false,
            "after_existed": true,
            "before_text": null,
            "after_text": "after"
        }]);

        restore_file_snapshot(CheckpointKind::Before, Some(snapshot.clone())).unwrap();
        assert!(!path.exists());

        restore_file_snapshot(CheckpointKind::After, Some(snapshot)).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "after");
    }

    #[test]
    fn git_checkpoint_restore_removes_files_created_after_checkpoint() {
        if !Command::new("git")
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
        {
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        run_git(dir.path(), &["init", "-q"]).unwrap();
        let existing = dir.path().join("existing.txt");
        let created = dir.path().join("created.txt");
        std::fs::write(&existing, "before").unwrap();

        let (git_ref, _) =
            capture_git_checkpoint("session", "turn", CheckpointKind::Before, dir.path()).unwrap();

        std::fs::write(&existing, "after").unwrap();
        std::fs::write(&created, "created").unwrap();

        restore_git_checkpoint(dir.path(), &git_ref).unwrap();

        assert_eq!(std::fs::read_to_string(existing).unwrap(), "before");
        assert!(!created.exists());
    }

    #[test]
    fn finish_turn_removes_matching_turn_out_of_order() {
        let session_id = format!("test-session-{}", Uuid::new_v4());
        let recorder = HarnessRecorder {
            session_id: session_id.clone(),
            store: Arc::new(Mutex::new(None)),
        };
        let first_turn = Uuid::new_v4().to_string();
        let second_turn = Uuid::new_v4().to_string();
        ACTIVE_TURNS.lock().insert(
            session_id.clone(),
            vec![
                ActiveTurn {
                    turn_id: first_turn.clone(),
                    recorder: recorder.clone(),
                },
                ActiveTurn {
                    turn_id: second_turn.clone(),
                    recorder,
                },
            ],
        );

        finish_turn(&session_id, &first_turn);
        assert_eq!(
            active_turn_id(&session_id).as_deref(),
            Some(second_turn.as_str())
        );

        finish_turn(&session_id, &second_turn);
        assert!(active_turn_id(&session_id).is_none());
    }
}
