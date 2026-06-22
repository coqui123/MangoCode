// FileEdit tool: exact string replacement with old/new strings (like sed but
// deterministic).

use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tracing::debug;

pub struct FileEditTool;

#[derive(Debug, Deserialize)]
struct FileEditInput {
    file_path: String,
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
    #[serde(default)]
    confirm_conflicts: bool,
}

#[async_trait]
impl Tool for FileEditTool {
    fn name(&self) -> &str {
        mangocode_core::constants::TOOL_NAME_FILE_EDIT
    }

    fn description(&self) -> &str {
        "Performs exact string replacements in files. The edit will FAIL if \
         `old_string` is not unique in the file (unless `replace_all` is true). \
         You MUST read the file first before editing. Preserve the exact \
         indentation as it appears in the file."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Write
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to modify"
                },
                "old_string": {
                    "type": "string",
                    "description": "The text to replace (must be unique in the file unless replace_all is true)"
                },
                "new_string": {
                    "type": "string",
                    "description": "The text to replace it with (must be different from old_string)"
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences of old_string (default false)"
                },
                "confirm_conflicts": {
                    "type": "boolean",
                    "description": "Set true only after acknowledging active MangoCode coordination conflicts for this path."
                }
            },
            "required": ["file_path", "old_string", "new_string"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: FileEditInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        // Validate old != new
        if params.old_string == params.new_string {
            return ToolResult::error("old_string and new_string must be different".to_string());
        }

        let path = ctx.resolve_path(&params.file_path);
        if !ctx.is_path_allowed(&path) {
            return ToolResult::error(format!(
                "Path {} is outside the allowed working directory",
                path.display()
            ));
        }
        debug!(path = %path.display(), "Editing file");

        // Permission check
        if let Err(e) =
            ctx.check_permission(self.name(), &format!("Edit {}", path.display()), false)
        {
            return ToolResult::error(e.to_string());
        }

        let coordination_conflicts = match crate::coordination::preflight_write_conflicts(
            ctx,
            self.name(),
            std::slice::from_ref(&path),
            params.confirm_conflicts,
        ) {
            Ok(conflicts) => conflicts,
            Err(result) => return result,
        };
        let _coordination_write_claim = match crate::coordination::begin_transient_write_claim(
            ctx,
            self.name(),
            std::slice::from_ref(&path),
            params.confirm_conflicts,
        ) {
            Ok(guard) => guard,
            Err(result) => return result,
        };

        // Read current content and hash it for TOCTOU protection.
        let content = match tokio::fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => {
                return ToolResult::error(format!("Failed to read file {}: {}", path.display(), e));
            }
        };
        let read_hash = Sha256::digest(content.as_bytes());

        // Tolerate line-ending mismatches without corrupting exact matches; see
        // edit_hints::align_line_endings.
        let (old_string, new_string) =
            crate::edit_hints::align_line_endings(&content, &params.old_string, &params.new_string);

        // Count occurrences
        let count = content.matches(&old_string).count();

        if count == 0 {
            let mut msg = format!(
                "old_string not found in {}. Make sure the string matches exactly, \
                 including whitespace and indentation.",
                path.display()
            );
            if let Some(hint) = crate::edit_hints::not_found_hint(&content, &old_string) {
                msg.push(' ');
                msg.push_str(&hint);
            }
            return ToolResult::error(msg);
        }

        if count > 1 && !params.replace_all {
            return ToolResult::error(format!(
                "old_string appears {} times in {}. Either provide a larger string \
                 with more surrounding context to make it unique, or set replace_all \
                 to true to replace every occurrence.",
                count,
                path.display()
            ));
        }

        // Perform replacement
        let new_content = if params.replace_all {
            content.replace(&old_string, &new_string)
        } else {
            // Replace only the first occurrence
            content.replacen(&old_string, &new_string, 1)
        };

        // Verify the file hasn't been modified externally since we read it.
        if let Ok(current) = tokio::fs::read_to_string(&path).await {
            if Sha256::digest(current.as_bytes()) != read_hash {
                return ToolResult::error(format!(
                    "File {} was modified externally since it was read. \
                     Re-read the file and try again.",
                    path.display()
                ));
            }
        }

        if ctx.config.dry_run {
            let replacements = if params.replace_all { count } else { 1 };
            return ToolResult::success(format!(
                "[DRY RUN] Would edit {} ({} replacement{}). No changes written.",
                path.display(),
                replacements,
                if replacements != 1 { "s" } else { "" }
            ));
        }

        // Write back
        if let Err(e) = crate::fs_atomic::write_atomic(&path, new_content.as_bytes()).await {
            return ToolResult::error(format!("Failed to write file {}: {}", path.display(), e));
        }

        // Run any configured formatter for this file type.
        crate::try_format_file(&path.to_string_lossy(), ctx).await;

        let (final_content, after_exists) = match tokio::fs::read(&path).await {
            Ok(bytes) => (bytes, true),
            Err(_) => (new_content.as_bytes().to_vec(), path.exists()),
        };
        ctx.record_file_change_with_existence(
            path.clone(),
            content.as_bytes(),
            &final_content,
            (true, after_exists),
            self.name(),
        );

        // Build a diff snippet for the response
        let replacements = if params.replace_all { count } else { 1 };
        let msg = format!(
            "Successfully edited {} ({} replacement{}).",
            path.display(),
            replacements,
            if replacements != 1 { "s" } else { "" }
        );

        let msg = crate::coordination::append_confirmed_conflict_note(
            msg,
            coordination_conflicts.as_deref(),
        );
        ToolResult::success(msg).with_metadata(json!({
            "file_path": path.display().to_string(),
            "replacements": replacements,
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
            session_id: "file-edit-test".to_string(),
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
            question_prompt_tx: None,
            cancel_token: None,
        }
    }

    // A CRLF file edited with a bare-LF old_string should still match, and the
    // written file must keep its CRLF endings.
    #[tokio::test]
    async fn edit_matches_crlf_file_with_lf_old_string() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("crlf.rs");
        tokio::fs::write(&file, "fn main() {\r\n    let x = 1;\r\n}\r\n")
            .await
            .unwrap();

        let ctx = test_context(dir.path());
        let result = FileEditTool
            .execute(
                json!({
                    "file_path": "crlf.rs",
                    "old_string": "    let x = 1;\n",
                    "new_string": "    let x = 2;\n"
                }),
                &ctx,
            )
            .await;

        assert!(!result.is_error, "edit should succeed: {}", result.content);
        let written = tokio::fs::read_to_string(&file).await.unwrap();
        assert_eq!(written, "fn main() {\r\n    let x = 2;\r\n}\r\n");
    }

    // A plain LF file must be unaffected by the CRLF tolerance path.
    #[tokio::test]
    async fn edit_lf_file_unchanged_endings() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("lf.rs");
        tokio::fs::write(&file, "a\nb\nc\n").await.unwrap();

        let ctx = test_context(dir.path());
        let result = FileEditTool
            .execute(
                json!({
                    "file_path": "lf.rs",
                    "old_string": "b\n",
                    "new_string": "B\n"
                }),
                &ctx,
            )
            .await;

        assert!(!result.is_error, "edit should succeed: {}", result.content);
        let written = tokio::fs::read_to_string(&file).await.unwrap();
        assert_eq!(written, "a\nB\nc\n");
    }
}
