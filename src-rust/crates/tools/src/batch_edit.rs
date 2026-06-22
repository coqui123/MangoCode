// BatchEdit tool: apply multiple file edits atomically.
//
// All edits are validated before any change is written.  If any pre-check
// fails the tool returns an error and leaves every file untouched.  If a write
// fails after some files have already been written, the tool attempts to
// restore those files from in-memory backups.

use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tracing::debug;

pub struct BatchEditTool;

#[derive(Debug, Deserialize)]
struct SingleEdit {
    file_path: String,
    old_string: String,
    new_string: String,
}

#[derive(Debug, Deserialize)]
struct BatchEditInput {
    edits: Vec<SingleEdit>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    confirm_conflicts: bool,
}

struct PreparedBatchFile {
    path: std::path::PathBuf,
    grouping_path: std::path::PathBuf,
    display_path: String,
    original_content: String,
    read_hash: sha2::digest::Output<Sha256>,
    new_content: String,
    edits_applied: usize,
}

#[async_trait]
impl Tool for BatchEditTool {
    fn name(&self) -> &str {
        mangocode_core::constants::TOOL_NAME_BATCH_EDIT
    }

    fn description(&self) -> &str {
        "Apply multiple file edits atomically. All edits are validated before any \
         file is modified. If any edit would fail (old_string not found or not \
         unique) the entire batch is rejected with no changes made. If a write \
         fails mid-batch, already-written files are rolled back."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Write
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "description": "List of edits to apply atomically",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Absolute path to the file to modify"
                            },
                            "old_string": {
                                "type": "string",
                                "description": "Text to replace (must occur exactly once in the file)"
                            },
                            "new_string": {
                                "type": "string",
                                "description": "Replacement text"
                            }
                        },
                        "required": ["file_path", "old_string", "new_string"]
                    }
                },
                "description": {
                    "type": "string",
                    "description": "Optional human-readable description of what this batch edit does"
                },
                "confirm_conflicts": {
                    "type": "boolean",
                    "description": "Set true only after acknowledging active MangoCode coordination conflicts for these paths."
                }
            },
            "required": ["edits"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: BatchEditInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        if params.edits.is_empty() {
            return ToolResult::error("edits array must not be empty".to_string());
        }

        // Permission check (one check covers the whole batch).
        let description = params.description.as_deref().unwrap_or("batch file edits");
        if let Err(e) =
            ctx.check_permission(self.name(), &format!("BatchEdit: {}", description), false)
        {
            return ToolResult::error(e.to_string());
        }

        let conflict_paths: Vec<std::path::PathBuf> = params
            .edits
            .iter()
            .map(|edit| ctx.resolve_path(&edit.file_path))
            .collect();
        let coordination_conflicts = match crate::coordination::preflight_write_conflicts(
            ctx,
            self.name(),
            &conflict_paths,
            params.confirm_conflicts,
        ) {
            Ok(conflicts) => conflicts,
            Err(result) => return result,
        };
        let _coordination_write_claim = match crate::coordination::begin_transient_write_claim(
            ctx,
            self.name(),
            &conflict_paths,
            params.confirm_conflicts,
        ) {
            Ok(guard) => guard,
            Err(result) => return result,
        };

        // ----------------------------------------------------------------
        // Phase 1: read all files and validate every edit before writing
        // ----------------------------------------------------------------

        let mut prepared: Vec<PreparedBatchFile> = Vec::with_capacity(params.edits.len());
        let mut pre_check_errors: Vec<String> = Vec::new();

        for (i, edit) in params.edits.iter().enumerate() {
            let path = ctx.resolve_path(&edit.file_path);
            debug!(path = %path.display(), index = i, "BatchEdit pre-check");

            if edit.old_string == edit.new_string {
                pre_check_errors.push(format!(
                    "Edit {}: old_string and new_string must be different",
                    i
                ));
                continue;
            }

            let display_path = path.display().to_string();
            let grouping_path = std::fs::canonicalize(&path).unwrap_or_else(|e| {
                tracing::debug!(
                    path = %path.display(),
                    error = %e,
                    "canonicalize failed; using raw path for batch-edit grouping"
                );
                path.clone()
            });
            let existing_idx = prepared.iter().position(|prepared| {
                prepared.grouping_path == grouping_path || prepared.path == path
            });

            let prepared_idx = match existing_idx {
                Some(idx) => idx,
                None => {
                    let content = match tokio::fs::read_to_string(&path).await {
                        Ok(c) => c,
                        Err(e) => {
                            pre_check_errors.push(format!(
                                "Edit {}: cannot read {}: {}",
                                i,
                                path.display(),
                                e
                            ));
                            continue;
                        }
                    };

                    let read_hash = Sha256::digest(content.as_bytes());
                    prepared.push(PreparedBatchFile {
                        path,
                        grouping_path,
                        display_path,
                        original_content: content.clone(),
                        read_hash,
                        new_content: content,
                        edits_applied: 0,
                    });
                    prepared.len() - 1
                }
            };
            let prepared_file = &mut prepared[prepared_idx];

            // Same CRLF tolerance as FileEdit, via the shared helper, so a
            // bare-LF old_string matches a CRLF file consistently across tools.
            let (old_string, new_string) = crate::edit_hints::align_line_endings(
                &prepared_file.new_content,
                &edit.old_string,
                &edit.new_string,
            );
            let count = prepared_file.new_content.matches(&old_string).count();
            if count == 0 {
                let mut msg = format!(
                    "Edit {}: old_string not found in {}",
                    i, prepared_file.display_path
                );
                if let Some(hint) =
                    crate::edit_hints::not_found_hint(&prepared_file.new_content, &old_string)
                {
                    msg.push_str(" — ");
                    msg.push_str(&hint);
                }
                pre_check_errors.push(msg);
                continue;
            }
            if count > 1 {
                pre_check_errors.push(format!(
                    "Edit {}: old_string appears {} times in {} (must be unique)",
                    i, count, prepared_file.display_path
                ));
                continue;
            }

            prepared_file.new_content =
                prepared_file
                    .new_content
                    .replacen(&old_string, &new_string, 1);
            prepared_file.edits_applied += 1;
        }

        if !pre_check_errors.is_empty() {
            return ToolResult::error(format!(
                "BatchEdit aborted — {} validation error(s):\n{}",
                pre_check_errors.len(),
                pre_check_errors.join("\n")
            ));
        }

        // ----------------------------------------------------------------
        // Phase 2: write all files; roll back on any failure
        // ----------------------------------------------------------------

        if ctx.config.dry_run {
            let file_count = prepared.len();
            let edit_count: usize = prepared.iter().map(|f| f.edits_applied).sum();
            return ToolResult::success(format!(
                "[DRY RUN] Would edit {} file(s) with {} replacement(s). No changes written.",
                file_count, edit_count
            ));
        }

        // TOCTOU: verify no file was modified externally since we read it.
        for prepared_file in &prepared {
            if let Ok(current) = tokio::fs::read_to_string(&prepared_file.path).await {
                if Sha256::digest(current.as_bytes()) != prepared_file.read_hash {
                    return ToolResult::error(format!(
                        "BatchEdit aborted — {} was modified externally since it was read. \
                         Re-read and try again.",
                        prepared_file.display_path
                    ));
                }
            }
        }

        // ----------------------------------------------------------------
        // Checkpoint: no harness-level checkpoint / snapshot store is
        // available to individual tools (it lives in the query driver).
        // As a safety measure we record every file's original state in
        // file_history BEFORE any write begins, so that if the process is
        // interrupted mid-batch the before-content is already persisted
        // and the written files are recoverable.  The in-memory `written`
        // vec still handles graceful rollback when a single write_atomic
        // call fails within this invocation.
        // ----------------------------------------------------------------
        for prepared_file in &prepared {
            ctx.record_file_change(
                prepared_file.path.clone(),
                prepared_file.original_content.as_bytes(),
                prepared_file.new_content.as_bytes(),
                self.name(),
            );
        }

        let mut written: Vec<(std::path::PathBuf, String)> = Vec::new(); // (path, original) for rollback

        for prepared_file in &prepared {
            match crate::fs_atomic::write_atomic(
                &prepared_file.path,
                prepared_file.new_content.as_bytes(),
            )
            .await
            {
                Ok(()) => {
                    written.push((
                        prepared_file.path.clone(),
                        prepared_file.original_content.clone(),
                    ));
                }
                Err(e) => {
                    // Attempt rollback of already-written files.
                    let mut rollback_errors: Vec<String> = Vec::new();
                    for (rb_path, rb_original) in &written {
                        if let Err(re) =
                            crate::fs_atomic::write_atomic_sync(rb_path, rb_original.as_bytes())
                        {
                            rollback_errors.push(format!(
                                "  rollback {}: {}",
                                rb_path.display(),
                                re
                            ));
                        }
                    }

                    let mut msg = format!(
                        "BatchEdit failed while writing {} ({}). Rolled back {} file(s).",
                        prepared_file.display_path,
                        e,
                        written.len()
                    );
                    if !rollback_errors.is_empty() {
                        msg.push_str(&format!(
                            "\nRollback errors:\n{}",
                            rollback_errors.join("\n")
                        ));
                    }
                    return ToolResult::error(msg);
                }
            }
        }

        for prepared_file in &prepared {
            // Run any configured formatter on each written file, then re-record
            // the change with the actual on-disk content so the entry reflects
            // any formatting differences (matches FileEdit/FileWrite behaviour).
            // The pre-write record above ensures recoverability even if we never
            // reach this point.
            crate::try_format_file(&prepared_file.path.to_string_lossy(), ctx).await;
            let after = match tokio::fs::read(&prepared_file.path).await {
                Ok(bytes) => bytes,
                Err(_) => prepared_file.new_content.as_bytes().to_vec(),
            };
            ctx.record_file_change(
                prepared_file.path.clone(),
                prepared_file.original_content.as_bytes(),
                &after,
                self.name(),
            );
        }

        // ----------------------------------------------------------------
        // Build success response
        // ----------------------------------------------------------------

        let file_count = prepared.len();
        let edit_count: usize = prepared.iter().map(|file| file.edits_applied).sum();

        let summary = format!(
            "BatchEdit applied {} edit{} across {} file{}.",
            edit_count,
            if edit_count != 1 { "s" } else { "" },
            file_count,
            if file_count != 1 { "s" } else { "" },
        );

        let summary = crate::coordination::append_confirmed_conflict_note(
            summary,
            coordination_conflicts.as_deref(),
        );

        ToolResult::success(summary).with_metadata(json!({
            "edits_applied": edit_count,
            "files_modified": file_count,
            "files": prepared.iter().map(|file| file.display_path.as_str()).collect::<Vec<_>>(),
            "coordination_conflicts": coordination_conflicts,
        }))
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
            session_id: "batch-edit-test".to_string(),
            coordination_actor_id: None,
            coordination_parent_actor_id: None,
            inject_coordination_inbox: true,
            file_history: Arc::new(parking_lot::Mutex::new(
                mangocode_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(std::sync::atomic::AtomicUsize::new(14)),
            non_interactive: true,
            mcp_manager: None,
            config: mangocode_core::config::Config::default(),
            question_prompt_tx: None,
            cancel_token: None,
        }
    }

    #[tokio::test]
    async fn failed_partial_write_rolls_back_without_file_history() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("first.txt");
        let read_only = dir.path().join("read_only.txt");
        tokio::fs::write(&first, "old first").await.unwrap();
        tokio::fs::write(&read_only, "old second").await.unwrap();

        let mut permissions = std::fs::metadata(&read_only).unwrap().permissions();
        permissions.set_readonly(true);
        std::fs::set_permissions(&read_only, permissions).unwrap();

        let ctx = test_context(dir.path());
        let result = BatchEditTool
            .execute(
                json!({
                    "edits": [
                        {
                            "file_path": "first.txt",
                            "old_string": "old first",
                            "new_string": "new first"
                        },
                        {
                            "file_path": "read_only.txt",
                            "old_string": "old second",
                            "new_string": "new second"
                        }
                    ]
                }),
                &ctx,
            )
            .await;

        let mut permissions = std::fs::metadata(&read_only).unwrap().permissions();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            permissions.set_mode(permissions.mode() | 0o200);
        }
        #[cfg(windows)]
        {
            #[allow(clippy::permissions_set_readonly_false)]
            permissions.set_readonly(false);
        }
        std::fs::set_permissions(&read_only, permissions).unwrap();

        assert!(result.is_error);
        assert_eq!(
            tokio::fs::read_to_string(&first).await.unwrap(),
            "old first"
        );
        assert_eq!(
            tokio::fs::read_to_string(&read_only).await.unwrap(),
            "old second"
        );
        // Pre-write record_file_change ensures original states are saved
        // for recoverability, even when the batch fails and rolls back.
        assert_eq!(ctx.file_history.lock().get_entries_for_turn(14).len(), 2);
    }

    #[tokio::test]
    async fn same_file_edits_are_applied_cumulatively() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("story.txt");
        tokio::fs::write(&file, "alpha beta gamma\n").await.unwrap();

        let ctx = test_context(dir.path());
        let result = BatchEditTool
            .execute(
                json!({
                    "edits": [
                        {
                            "file_path": "story.txt",
                            "old_string": "alpha",
                            "new_string": "ALPHA"
                        },
                        {
                            "file_path": "story.txt",
                            "old_string": "gamma",
                            "new_string": "GAMMA"
                        }
                    ]
                }),
                &ctx,
            )
            .await;

        assert!(!result.is_error, "{}", result.content);
        assert_eq!(
            tokio::fs::read_to_string(&file).await.unwrap(),
            "ALPHA beta GAMMA\n"
        );
        assert_eq!(result.metadata.as_ref().unwrap()["edits_applied"], 2);
        assert_eq!(result.metadata.as_ref().unwrap()["files_modified"], 1);

        let history = ctx.file_history.lock();
        let entries = history.get_entries_for_turn(14);
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries[0].before_text.as_deref(),
            Some("alpha beta gamma\n")
        );
        assert_eq!(entries[0].after_text.as_deref(), Some("ALPHA beta GAMMA\n"));
    }

    #[tokio::test]
    async fn noop_edit_is_rejected_without_writing() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("same.txt");
        tokio::fs::write(&file, "same\n").await.unwrap();

        let ctx = test_context(dir.path());
        let result = BatchEditTool
            .execute(
                json!({
                    "edits": [
                        {
                            "file_path": "same.txt",
                            "old_string": "same",
                            "new_string": "same"
                        }
                    ]
                }),
                &ctx,
            )
            .await;

        assert!(result.is_error);
        assert!(result
            .content
            .contains("old_string and new_string must be different"));
        assert_eq!(tokio::fs::read_to_string(&file).await.unwrap(), "same\n");
        assert!(ctx.file_history.lock().get_entries_for_turn(14).is_empty());
    }
}
