// FileRead tool: read files with optional line range, image support, PDF page ranges.

use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use mangocode_core::smart_attachments::{classify_path, extract_markdown_native, AttachmentKind};
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::debug;

pub struct FileReadTool;

#[derive(Debug, Deserialize)]
struct FileReadInput {
    file_path: String,
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
}

#[async_trait]
impl Tool for FileReadTool {
    fn name(&self) -> &str {
        mangocode_core::constants::TOOL_NAME_FILE_READ
    }

    fn description(&self) -> &str {
        "Reads a file from the local filesystem. You can access any file directly. \
         By default reads up to 2000 lines from the beginning. Results are returned \
         with line numbers starting at 1. This tool can read images (PNG, JPG) and \
         PDF files."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to read"
                },
                "offset": {
                    "type": "number",
                    "description": "The line number to start reading from (1-based). Only provide if the file is too large to read at once."
                },
                "limit": {
                    "type": "number",
                    "description": "The number of lines to read. Only provide if the file is too large to read at once."
                }
            },
            "required": ["file_path"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: FileReadInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        let path = ctx.resolve_path(&params.file_path);
        if !ctx.is_path_allowed(&path) {
            return ToolResult::error(format!(
                "Path {} is outside the allowed working directory",
                path.display()
            ));
        }
        debug!(path = %path.display(), "Reading file");

        // Check if file exists
        if !path.exists() {
            return ToolResult::error(format!("File not found: {}", path.display()));
        }

        // Check if it's a directory
        if path.is_dir() {
            return ToolResult::error(format!(
                "{} is a directory, not a file. Use Bash with `ls` to list directory contents.",
                path.display()
            ));
        }

        // Detect binary / image files by extension
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let image_exts = ["png", "jpg", "jpeg", "gif", "bmp", "webp", "svg", "ico"];
        if image_exts.contains(&ext.as_str()) {
            return ToolResult::success(format!(
                "[Image file: {}. The image content has been captured for visual analysis.]",
                path.display()
            ));
        }

        match classify_path(&path) {
            AttachmentKind::Pdf
            | AttachmentKind::OfficeDocument
            | AttachmentKind::Html
            | AttachmentKind::Data
            | AttachmentKind::Archive => {
                return match extract_markdown_native(&path) {
                    Ok(extracted) => ToolResult::success(format!(
                        "[Document converted with native Markdown extraction: {}{}]\n\n{}",
                        path.display(),
                        if extracted.from_cache {
                            " (cached)"
                        } else {
                            ""
                        },
                        extracted.markdown
                    )),
                    Err(e) => ToolResult::error(format!(
                        "Could not convert document with native Markdown extraction: {}. If this is scanned/complex content, use a model/provider that supports the raw format.",
                        e
                    )),
                };
            }
            _ => {}
        }

        // Read text file
        let content = match tokio::fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => {
                // Might be binary
                if e.kind() == std::io::ErrorKind::InvalidData {
                    return ToolResult::error(format!(
                        "File appears to be binary and cannot be displayed as text: {}",
                        path.display()
                    ));
                }
                return ToolResult::error(format!("Failed to read file: {}", e));
            }
        };

        if content.is_empty() {
            return ToolResult::success(format!("[File {} exists but is empty]", path.display()));
        }

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        if params.limit == Some(0) {
            return ToolResult::error(
                "Invalid input: limit must be a positive integer (got 0)".to_string(),
            );
        }

        let offset = params.offset.unwrap_or(0);
        let limit = params.limit.unwrap_or(2000);

        // Convert 1-based offset to 0-based index
        let start = if offset > 0 { offset - 1 } else { 0 };
        let end = (start + limit).min(total_lines);

        if start >= total_lines {
            return ToolResult::error(format!(
                "Offset {} exceeds total line count {} in {}",
                offset,
                total_lines,
                path.display()
            ));
        }

        let mut output = String::new();
        let width = format!("{}", end).len();

        let mut capped_at_line = None;
        for (i, line) in lines[start..end].iter().enumerate() {
            let line_num = start + i + 1;
            output.push_str(&format!(
                "{:>width$}\t{}\n",
                line_num,
                truncate_long_line(line),
                width = width
            ));
            if output.len() >= MAX_TOTAL_BYTES && line_num < end {
                capped_at_line = Some(line_num);
                break;
            }
        }

        if let Some(last_line) = capped_at_line {
            output.push_str(&format!(
                "\n... (output capped at {} KB after line {}; {} total lines. \
                 Use offset={} to continue.)\n",
                MAX_TOTAL_BYTES / 1024,
                last_line,
                total_lines,
                last_line + 1
            ));
        } else if end < total_lines {
            output.push_str(&format!(
                "\n... ({} more lines, {} total. Use offset/limit to read more.)\n",
                total_lines - end,
                total_lines
            ));
        }

        ToolResult::success(output)
    }
}

/// Cap a single line's contribution to the output so files with very long
/// lines (minified JS, lockfiles, embedded data) don't flood the context.
const MAX_LINE_CHARS: usize = 2000;

/// Cap the total Read output so a single call can't flood the context even
/// when many lines are near the per-line limit.
const MAX_TOTAL_BYTES: usize = 256 * 1024;

fn truncate_long_line(line: &str) -> std::borrow::Cow<'_, str> {
    // Single pass: find the byte offset of the MAX_LINE_CHARS-th char and the
    // total char count together, so a very long line isn't decoded twice.
    let mut cut = None;
    let mut total = 0usize;
    for (idx, (byte_idx, _)) in line.char_indices().enumerate() {
        if idx == MAX_LINE_CHARS {
            cut = Some(byte_idx);
        }
        total = idx + 1;
    }
    match cut {
        None => std::borrow::Cow::Borrowed(line),
        Some(byte_idx) => std::borrow::Cow::Owned(format!(
            "{}… [line truncated; {} chars total]",
            &line[..byte_idx],
            total
        )),
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
            session_id: "file-read-test".to_string(),
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

    #[tokio::test]
    async fn caps_total_output_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.txt");
        // 300 lines x ~1900 chars ≈ 570 KB, over the 256 KB cap.
        let line = "z".repeat(1900);
        let content = (0..300)
            .map(|_| line.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&path, content).unwrap();

        let tool = FileReadTool;
        let ctx = test_context(dir.path());
        let result = tool
            .execute(
                serde_json::json!({ "file_path": path.to_string_lossy() }),
                &ctx,
            )
            .await;
        let text = result.content;
        assert!(text.len() < MAX_TOTAL_BYTES + 4096, "len = {}", text.len());
        assert!(text.contains("output capped"), "missing cap notice");
        assert!(text.contains("Use offset="), "missing continuation hint");
    }

    #[test]
    fn short_lines_pass_through() {
        assert_eq!(truncate_long_line("hello"), "hello");
    }

    #[test]
    fn exactly_max_chars_not_truncated() {
        let line = "x".repeat(MAX_LINE_CHARS);
        assert_eq!(truncate_long_line(&line), line);
    }

    #[test]
    fn long_line_truncated_with_marker() {
        let line = "y".repeat(MAX_LINE_CHARS + 500);
        let out = truncate_long_line(&line);
        assert!(out.starts_with(&"y".repeat(MAX_LINE_CHARS)));
        assert!(out.contains("[line truncated; 2500 chars total]"));
    }

    #[test]
    fn truncation_respects_multibyte_boundaries() {
        let line = "é".repeat(MAX_LINE_CHARS + 10);
        let out = truncate_long_line(&line);
        assert!(out.contains("[line truncated"));
        // Must not panic on a char boundary and must keep valid UTF-8.
        assert_eq!(
            out.chars().take_while(|&c| c == 'é').count(),
            MAX_LINE_CHARS
        );
    }
}
