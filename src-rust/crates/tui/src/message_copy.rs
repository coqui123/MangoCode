//! Message copy utilities for different formatting options.
//!
//! Provides functions to copy messages in various formats:
//! - Markdown: Preserves markdown formatting
//! - Plaintext: Removes all markdown formatting
//! - Code blocks only: Extracts code blocks
//! - JSON: Serialized message data
//! - Selection only: Selected text only

use crate::messages::render_structured_tool_result;
use mangocode_core::Message;
use serde_json::json;
use std::io::Write;

fn json_pretty_or_error(value: &serde_json::Value) -> String {
    serde_json::to_string_pretty(value)
        .unwrap_or_else(|err| format!("<failed to serialize JSON: {err}>"))
}

/// Copy message as markdown (preserving formatting)
pub fn copy_as_markdown(message: &Message) -> String {
    let content = match &message.content {
        mangocode_core::MessageContent::Text(text) => text.clone(),
        mangocode_core::MessageContent::Blocks(blocks) => {
            blocks
                .iter()
                .filter_map(|block| match block {
                    mangocode_core::ContentBlock::Text { text } => Some(text.clone()),
                    mangocode_core::ContentBlock::Thinking {
                        thinking,
                        signature,
                    } => {
                        // Include thinking blocks as collapsible sections
                        Some(format!(
                            "<details>\n<summary>Thinking ({})</summary>\n\n{}\n</details>",
                            signature, thinking
                        ))
                    }
                    mangocode_core::ContentBlock::ToolUse { id, name, input } => {
                        // Format tool use as code block
                        Some(format!(
                            "```json\n// Tool: {}\n// ID: {}\n{}\n```",
                            name,
                            id,
                            json_pretty_or_error(input)
                        ))
                    }
                    mangocode_core::ContentBlock::ToolResult {
                        tool_use_id: _,
                        content,
                        is_error,
                        metadata,
                    } => {
                        let error_marker = if is_error.unwrap_or(false) {
                            "ERROR: "
                        } else {
                            ""
                        };
                        if !is_error.unwrap_or(false) {
                            if let Some(text) = structured_tool_result_text(metadata.as_ref()) {
                                return Some(format!("```\n{}\n```", text));
                            }
                        }
                        let result_text = match content {
                            mangocode_core::ToolResultContent::Text(text) => text.clone(),
                            mangocode_core::ToolResultContent::Blocks(blocks) => blocks
                                .iter()
                                .filter_map(|b| match b {
                                    mangocode_core::ContentBlock::Text { text } => {
                                        Some(text.clone())
                                    }
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("\n"),
                        };
                        Some(format!("```\n{}{}\n```", error_marker, result_text))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n\n")
        }
    };

    format_markdown_message(&message.role, &content)
}

/// Copy message as plaintext (no markdown formatting)
pub fn copy_as_plaintext(message: &Message) -> String {
    let content = match &message.content {
        mangocode_core::MessageContent::Text(text) => strip_markdown(text),
        mangocode_core::MessageContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| match block {
                mangocode_core::ContentBlock::Text { text } => Some(strip_markdown(text)),
                mangocode_core::ContentBlock::Thinking { thinking, .. } => {
                    Some(format!("[Thinking]\n{}", thinking))
                }
                mangocode_core::ContentBlock::ToolUse { name, input, .. } => {
                    Some(format!("[Tool: {}]\n{}", name, json_pretty_or_error(input)))
                }
                mangocode_core::ContentBlock::ToolResult {
                    tool_use_id: _,
                    content,
                    is_error,
                    metadata,
                } => {
                    let error_marker = if is_error.unwrap_or(false) {
                        "[ERROR] "
                    } else {
                        ""
                    };
                    if !is_error.unwrap_or(false) {
                        if let Some(text) = structured_tool_result_text(metadata.as_ref()) {
                            return Some(text);
                        }
                    }
                    let result_text = match content {
                        mangocode_core::ToolResultContent::Text(text) => text.clone(),
                        mangocode_core::ToolResultContent::Blocks(blocks) => blocks
                            .iter()
                            .filter_map(|b| match b {
                                mangocode_core::ContentBlock::Text { text } => Some(text.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n"),
                    };
                    Some(format!("{}{}", error_marker, result_text))
                }
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n\n"),
    };

    let role_str = match message.role {
        mangocode_core::Role::User => "User",
        mangocode_core::Role::Assistant => "Assistant",
    };
    format!("{}:\n\n{}", role_str, content)
}

/// Extract and copy only code blocks from message
pub fn copy_code_blocks(message: &Message) -> String {
    let mut code_blocks = Vec::new();

    match &message.content {
        mangocode_core::MessageContent::Text(text) => {
            extract_code_blocks_from_text(text, &mut code_blocks);
        }
        mangocode_core::MessageContent::Blocks(blocks) => {
            for block in blocks {
                if let mangocode_core::ContentBlock::Text { text } = block {
                    extract_code_blocks_from_text(text, &mut code_blocks);
                }
            }
        }
    }

    if code_blocks.is_empty() {
        "[No code blocks found in message]".to_string()
    } else {
        code_blocks.join("\n\n---\n\n")
    }
}

/// Copy message as JSON
pub fn copy_as_json(message: &Message) -> String {
    let role_str = match message.role {
        mangocode_core::Role::User => "user",
        mangocode_core::Role::Assistant => "assistant",
    };

    let json_value = json!({
        "role": role_str,
        "content": match &message.content {
            mangocode_core::MessageContent::Text(text) => text.clone(),
            mangocode_core::MessageContent::Blocks(blocks) => {
                blocks.iter().map(format_block_for_json).collect::<Vec<_>>().join("\n")
            }
        },
        "uuid": message.uuid,
        "cost": message.cost.as_ref().map(|c| json!({
            "input_tokens": c.input_tokens,
            "output_tokens": c.output_tokens,
            "cache_creation_input_tokens": c.cache_creation_input_tokens,
            "cache_read_input_tokens": c.cache_read_input_tokens,
            "cost_usd": c.cost_usd,
        }))
    });

    serde_json::to_string_pretty(&json_value).unwrap_or_else(|_| "{}".to_string())
}

/// Extract plaintext from selected text (identity function, for consistency)
pub fn copy_selection(selected_text: &str) -> String {
    selected_text.to_string()
}

// ============================================================================
// Private helpers
// ============================================================================

/// Format a message with role prefix as markdown
fn format_markdown_message(role: &mangocode_core::Role, content: &str) -> String {
    let role_str = match role {
        mangocode_core::Role::User => "**User**",
        mangocode_core::Role::Assistant => "**Assistant**",
    };
    format!("{}\n\n{}", role_str, content)
}

/// Strip markdown formatting from text
fn strip_markdown(text: &str) -> String {
    let mut result = String::new();
    let mut chars = text.chars().peekable();
    let mut _in_code_block = false;
    let mut in_inline_code = false;

    while let Some(ch) = chars.next() {
        match ch {
            // Skip markdown syntax
            '*' | '_' => {
                // Check for bold/italic markers (skip them)
                if chars.peek() == Some(&ch) {
                    chars.next(); // consume second marker
                }
            }
            '`' => {
                in_inline_code = !in_inline_code;
                // Don't add backticks to plaintext
            }
            '[' => {
                // Handle markdown links: [text](url) -> text
                let mut link_text = String::new();
                let mut found_close = false;
                for ch in chars.by_ref() {
                    if ch == ']' {
                        found_close = true;
                        break;
                    }
                    link_text.push(ch);
                }
                result.push_str(&link_text);
                // Skip URL part
                if found_close && chars.peek() == Some(&'(') {
                    chars.next(); // consume '('
                    for ch in chars.by_ref() {
                        if ch == ')' {
                            break;
                        }
                    }
                }
            }
            '#' => {
                // Skip markdown headers, but keep content
                // Skip the hash marks
                while chars.peek() == Some(&'#') {
                    chars.next();
                }
                // Skip space after hashes
                if chars.peek() == Some(&' ') {
                    chars.next();
                }
            }
            '!' => {
                // Skip markdown image syntax ![alt](url)
                if chars.peek() == Some(&'[') {
                    chars.next();
                    for c in chars.by_ref() {
                        if c == ']' {
                            break;
                        }
                    }
                    if chars.peek() == Some(&'(') {
                        chars.next();
                        for c in chars.by_ref() {
                            if c == ')' {
                                break;
                            }
                        }
                    }
                }
            }
            '>' if text.starts_with('>') => {
                // Skip blockquote markers, but keep content
                if result.is_empty() || result.ends_with('\n') {
                    while chars.peek() == Some(&'>') {
                        chars.next();
                    }
                    if chars.peek() == Some(&' ') {
                        chars.next();
                    }
                }
            }
            _ => result.push(ch),
        }
    }

    result.trim().to_string()
}

/// Extract code blocks from markdown text
fn extract_code_blocks_from_text(text: &str, blocks: &mut Vec<String>) {
    let mut in_block = false;
    let mut current_block = String::new();
    let mut language = String::new();
    let lines = text.lines().peekable();

    for line in lines {
        if let Some(stripped) = line.strip_prefix("```") {
            if in_block {
                // End of code block
                if !current_block.trim().is_empty() {
                    blocks.push(current_block.clone());
                }
                current_block.clear();
                language.clear();
                in_block = false;
            } else {
                // Start of code block
                in_block = true;
                language = stripped.trim().to_string();
            }
        } else if in_block {
            current_block.push_str(line);
            current_block.push('\n');
        }
    }

    // Handle unclosed block
    if in_block && !current_block.trim().is_empty() {
        blocks.push(current_block);
    }
}

/// Format a content block as JSON-compatible string
fn format_block_for_json(block: &mangocode_core::ContentBlock) -> String {
    match block {
        mangocode_core::ContentBlock::Text { text } => text.clone(),
        mangocode_core::ContentBlock::Image { .. } => "[Image content]".to_string(),
        mangocode_core::ContentBlock::ToolUse { id, name, input } => {
            format!(
                "[Tool: {} (ID: {})]\n{}",
                name,
                id,
                json_pretty_or_error(input)
            )
        }
        mangocode_core::ContentBlock::ToolResult {
            tool_use_id: _,
            content,
            is_error,
            metadata,
        } => {
            let error_marker = if is_error.unwrap_or(false) {
                "[ERROR] "
            } else {
                ""
            };
            if !is_error.unwrap_or(false) {
                if let Some(text) = structured_tool_result_text(metadata.as_ref()) {
                    return text;
                }
            }
            let result_text = match content {
                mangocode_core::ToolResultContent::Text(text) => text.clone(),
                mangocode_core::ToolResultContent::Blocks(blocks) => blocks
                    .iter()
                    .filter_map(|b| match b {
                        mangocode_core::ContentBlock::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            };
            format!("{}{}", error_marker, result_text)
        }
        mangocode_core::ContentBlock::Thinking { thinking, .. } => thinking.clone(),
        _ => "[Unsupported content type]".to_string(),
    }
}

fn structured_tool_result_text(metadata: Option<&serde_json::Value>) -> Option<String> {
    let lines = render_structured_tool_result(metadata, 100)?;
    let text = lines
        .iter()
        .map(|line| {
            line.spans
                .iter()
                .map(|span| span.content.to_string())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");
    (!text.trim().is_empty()).then_some(text)
}

// ============================================================================
// Clipboard integration
// ============================================================================

/// Attempt to copy text to clipboard using platform CLI tools
pub fn copy_to_clipboard(text: &str) -> bool {
    // Windows
    #[cfg(target_os = "windows")]
    {
        if let Ok(child) = std::process::Command::new("cmd")
            .args(["/C", "clip"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
        {
            return finish_clipboard_writer(child, "cmd /C clip", text);
        }
    }

    // macOS
    #[cfg(target_os = "macos")]
    {
        if let Ok(child) = std::process::Command::new("pbcopy")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
        {
            return finish_clipboard_writer(child, "pbcopy", text);
        }
    }

    // Linux
    #[cfg(target_os = "linux")]
    {
        for cmd in &["xclip -selection clipboard", "xsel --clipboard --input"] {
            let parts: Vec<&str> = cmd.split_whitespace().collect();
            if let Ok(child) = std::process::Command::new(parts[0])
                .args(&parts[1..])
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
            {
                if finish_clipboard_writer(child, parts[0], text) {
                    return true;
                }
            }
        }
    }

    false
}

fn finish_clipboard_writer(mut child: std::process::Child, tool: &str, text: &str) -> bool {
    let write_ok = match child.stdin.take() {
        Some(mut stdin) => match stdin.write_all(text.as_bytes()) {
            Ok(()) => true,
            Err(err) => {
                tracing::warn!(error = %err, %tool, "failed to write clipboard text");
                false
            }
        },
        None => {
            tracing::warn!(%tool, "clipboard helper did not expose stdin");
            false
        }
    };

    let exit_ok = match child.wait() {
        Ok(status) => status.success(),
        Err(err) => {
            tracing::warn!(error = %err, %tool, "failed to wait for clipboard helper");
            false
        }
    };

    write_ok && exit_ok
}

#[cfg(test)]
mod tests {
    use super::*;
    use mangocode_core::{ContentBlock, Message, ToolResultContent};

    #[test]
    fn markdown_copy_uses_structured_plan_metadata() {
        let msg = Message::user_blocks(vec![ContentBlock::ToolResult {
            tool_use_id: "plan-1".to_string(),
            content: ToolResultContent::Text("Plan updated".to_string()),
            is_error: Some(false),
            metadata: Some(serde_json::json!({
                "transcript_display": {
                    "kind": "updated_plan",
                    "plan": [{ "step": "Check copy path", "status": "pending" }]
                }
            })),
        }]);

        let copied = copy_as_markdown(&msg);

        assert!(copied.contains("Updated Plan"));
        assert!(copied.contains("[ ] Check copy path"));
        assert!(!copied.contains("Plan updated"));
    }

    #[test]
    fn plaintext_copy_uses_structured_file_change_metadata() {
        let msg = Message::user_blocks(vec![ContentBlock::ToolResult {
            tool_use_id: "edit-1".to_string(),
            content: ToolResultContent::Text("Successfully edited file".to_string()),
            is_error: Some(false),
            metadata: Some(serde_json::json!({
                "transcript_display": {
                    "kind": "file_changes",
                    "files": [{
                        "path": "src/lib.rs",
                        "change_type": "update",
                        "lines_added": 1,
                        "lines_removed": 0,
                        "binary": false,
                        "unified_diff": "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1,2 @@\n same\n+new\n"
                    }]
                }
            })),
        }]);

        let copied = copy_as_plaintext(&msg);

        assert!(copied.contains("Edited src/lib.rs (+1 -0)"));
        assert!(copied.contains("+new"));
        assert!(!copied.contains("Successfully edited file"));
    }

    #[test]
    fn json_copy_uses_structured_tool_result_text() {
        let msg = Message::user_blocks(vec![ContentBlock::ToolResult {
            tool_use_id: "plan-1".to_string(),
            content: ToolResultContent::Text("Plan updated".to_string()),
            is_error: Some(false),
            metadata: Some(serde_json::json!({
                "transcript_display": {
                    "kind": "updated_plan",
                    "plan": [{ "step": "Export structured text", "status": "completed" }]
                }
            })),
        }]);

        let copied = copy_as_json(&msg);

        assert!(copied.contains("Updated Plan"));
        assert!(copied.contains("[x] Export structured text"));
        assert!(!copied.contains("Plan updated"));
    }
    #[test]
    fn test_strip_markdown() {
        assert_eq!(strip_markdown("**bold**"), "bold");
        assert_eq!(strip_markdown("*italic*"), "italic");
        assert_eq!(strip_markdown("[link](url)"), "link");
        assert_eq!(strip_markdown("# Header"), "Header");
    }

    #[test]
    fn test_extract_code_blocks() {
        let text = r#"Some text
```rust
fn main() {}
```
More text
```python
print("hello")
```"#;
        let mut blocks = Vec::new();
        extract_code_blocks_from_text(text, &mut blocks);
        assert_eq!(blocks.len(), 2);
    }
}
