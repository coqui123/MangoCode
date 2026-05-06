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

        // ----------------------------------------------------------------
        // Phase 1: read all files and validate every edit before writing
        // ----------------------------------------------------------------

        // (resolved_path_string, original_content, new_content)
        let mut prepared: Vec<(String, String, String)> = Vec::with_capacity(params.edits.len());
        let mut pre_check_errors: Vec<String> = Vec::new();

        for (i, edit) in params.edits.iter().enumerate() {
            let path = ctx.resolve_path(&edit.file_path);
            debug!(path = %path.display(), index = i, "BatchEdit pre-check");

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

            let count = content.matches(&edit.old_string).count();
            if count == 0 {
                pre_check_errors.push(format!(
                    "Edit {}: old_string not found in {}",
                    i,
                    path.display()
                ));
                continue;
            }
            if count > 1 {
                pre_check_errors.push(format!(
                    "Edit {}: old_string appears {} times in {} (must be unique)",
                    i,
                    count,
                    path.display()
                ));
                continue;
            }

            let new_content = content.replacen(&edit.old_string, &edit.new_string, 1);
            prepared.push((path.display().to_string(), content, new_content));
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

        let mut written: Vec<(String, String)> = Vec::new(); // (path, original) for rollback

        for (path_str, original, new_content) in &prepared {
            let path = std::path::Path::new(path_str);
            match tokio::fs::write(path, new_content).await {
                Ok(()) => {
                    written.push((path_str.clone(), original.clone()));
                }
                Err(e) => {
                    // Attempt rollback of already-written files.
                    let mut rollback_errors: Vec<String> = Vec::new();
                    for (rb_path, rb_original) in &written {
                        if let Err(re) = std::fs::write(rb_path, rb_original) {
                            rollback_errors.push(format!("  rollback {}: {}", rb_path, re));
                        }
                    }

                    let mut msg = format!(
                        "BatchEdit failed while writing {} ({}). Rolled back {} file(s).",
                        path_str,
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

        for (path_str, original, new_content) in &prepared {
            ctx.record_file_change(
                std::path::PathBuf::from(path_str),
                original.as_bytes(),
                new_content.as_bytes(),
                self.name(),
            );
        }

        // ----------------------------------------------------------------
        // Build success response
        // ----------------------------------------------------------------

        let unique_files: std::collections::HashSet<&str> =
            prepared.iter().map(|(p, _, _)| p.as_str()).collect();
        let file_count = unique_files.len();
        let edit_count = prepared.len();

        let summary = format!(
            "BatchEdit applied {} edit{} across {} file{}.",
            edit_count,
            if edit_count != 1 { "s" } else { "" },
            file_count,
            if file_count != 1 { "s" } else { "" },
        );

        ToolResult::success(summary).with_metadata(json!({
            "edits_applied": edit_count,
            "files_modified": file_count,
            "files": prepared.iter().map(|(p, _, _)| p).collect::<Vec<_>>(),
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
            file_history: Arc::new(parking_lot::Mutex::new(
                mangocode_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(std::sync::atomic::AtomicUsize::new(14)),
            non_interactive: true,
            mcp_manager: None,
            config: mangocode_core::config::Config::default(),
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
        assert!(ctx.file_history.lock().get_entries_for_turn(14).is_empty());
    }
}
