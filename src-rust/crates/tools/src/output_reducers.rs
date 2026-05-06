//! RTK-style tool output reduction.
//!
//! Reducers keep the model-facing output small while preserving raw command
//! logs on disk for follow-up inspection.

#![cfg_attr(
    not(feature = "tool-tool-log-read"),
    allow(dead_code, unused_imports)
)]

use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::PathBuf;

const MAX_RAW_OUTPUT_CHARS: usize = 100_000;
const MAX_SUMMARY_LINES: usize = 160;

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum OutputMode {
    #[default]
    Auto,
    Raw,
    Summary,
}

impl OutputMode {
    pub fn from_config(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "raw" => Self::Raw,
            "summary" => Self::Summary,
            _ => Self::Auto,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReducedOutput {
    pub content: String,
    pub reducer: String,
    pub original_bytes: usize,
    pub reduced_bytes: usize,
    pub raw_log_path: Option<PathBuf>,
}

impl ReducedOutput {
    pub fn metadata(&self) -> Value {
        json!({
            "reducer": self.reducer,
            "original_bytes": self.original_bytes,
            "reduced_bytes": self.reduced_bytes,
            "raw_log_path": self.raw_log_path.as_ref().map(|path| path.display().to_string()),
        })
    }

    pub fn into_tool_result(self, exit_code: i32, error_prefix: &str) -> ToolResult {
        let metadata = self.metadata();
        if exit_code != 0 {
            ToolResult::error(format!("{} {}\n{}", error_prefix, exit_code, self.content))
                .with_metadata(metadata)
        } else {
            ToolResult::success(self.content).with_metadata(metadata)
        }
    }
}

#[derive(Debug, Deserialize)]
struct ToolLogReadInput {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    latest: bool,
    #[serde(default = "default_max_log_chars")]
    max_chars: usize,
}

fn default_max_log_chars() -> usize {
    MAX_RAW_OUTPUT_CHARS
}

#[cfg(feature = "tool-tool-log-read")]
pub struct ToolLogReadTool;

#[cfg(feature = "tool-tool-log-read")]
#[async_trait]
impl Tool for ToolLogReadTool {
    fn name(&self) -> &str {
        "ToolLogRead"
    }

    fn description(&self) -> &str {
        "Read exact raw command output cached by MangoCode output reducers. Use when compressed tool output omitted details."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Raw log path from a reducer header. Must be under ~/.mangocode/tool-logs." },
                "latest": { "type": "boolean", "description": "Read the newest raw tool log when path is omitted." },
                "max_chars": { "type": "number", "description": "Maximum characters to return, default 100000." }
            }
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: ToolLogReadInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid ToolLogRead input: {}", e)),
        };
        if let Err(e) = ctx.check_permission(self.name(), "Read cached raw tool log", true) {
            return ToolResult::error(e.to_string());
        }
        match read_tool_log(params.path.as_deref(), params.latest, params.max_chars) {
            Ok((path, content)) => ToolResult::success(format!(
                "[raw-tool-log path={} chars={}]\n{}",
                path.display(),
                content.len(),
                content
            )),
            Err(e) => ToolResult::error(e),
        }
    }
}

pub fn reduce_command_output(
    command: &str,
    output: &str,
    exit_code: i32,
    mode: OutputMode,
) -> ReducedOutput {
    let original_bytes = output.len();
    if mode == OutputMode::Raw {
        let content = truncate_raw(output);
        return ReducedOutput {
            reduced_bytes: content.len(),
            content,
            reducer: "raw".to_string(),
            original_bytes,
            raw_log_path: None,
        };
    }

    let raw_log_path = save_raw_log(command, output).ok();
    let reducer = detect_reducer(command);
    let reduced = match reducer.as_str() {
        "git-diff" => reduce_git_diff(output),
        "git-status" => reduce_git_status(output),
        "directory-listing" => reduce_directory_listing(output),
        "cargo" | "pytest" | "javascript" | "typescript" => reduce_test_or_build(output),
        _ => reduce_generic(output),
    };

    let should_keep_raw = mode == OutputMode::Auto
        && reduced.len() >= output.len()
        && output.len() <= MAX_RAW_OUTPUT_CHARS;
    let body = if should_keep_raw {
        output.to_string()
    } else {
        reduced
    };
    let body = truncate_raw(&body);
    let raw_path = raw_log_path
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "(not written)".to_string());
    let content = format!(
        "[tool-output reducer={} exit_code={} original={}B reduced={}B raw_log={} recall=\"ToolLogRead path='<raw_log>'\"]\n{}",
        reducer,
        exit_code,
        original_bytes,
        body.len(),
        raw_path,
        body
    );

    ReducedOutput {
        reduced_bytes: content.len(),
        content,
        reducer,
        original_bytes,
        raw_log_path,
    }
}

fn read_tool_log(
    path: Option<&str>,
    latest: bool,
    max_chars: usize,
) -> Result<(PathBuf, String), String> {
    let selected = if let Some(path) = path.filter(|p| !p.trim().is_empty()) {
        let path = PathBuf::from(path);
        let canonical = path
            .canonicalize()
            .map_err(|e| format!("Failed to resolve raw log path: {}", e))?;
        let allowed = tool_log_dirs()
            .into_iter()
            .filter_map(|dir| dir.canonicalize().ok())
            .any(|dir| canonical.starts_with(dir));
        if !allowed {
            return Err(
                "ToolLogRead only reads files under MangoCode tool-log directories".to_string(),
            );
        }
        canonical
    } else if latest {
        newest_tool_log_anywhere()?
    } else {
        return Err("Provide a raw log path or set latest=true.".to_string());
    };
    let content =
        std::fs::read_to_string(&selected).map_err(|e| format!("Failed to read log: {}", e))?;
    Ok((
        selected,
        mangocode_core::truncate::truncate_bytes_prefix(&content, max_chars.max(1)).to_string(),
    ))
}

fn newest_tool_log_anywhere() -> Result<PathBuf, String> {
    let mut newest = None;
    for dir in tool_log_dirs() {
        if let Ok(path) = newest_tool_log(&dir) {
            let modified = path.metadata().and_then(|m| m.modified()).ok();
            if newest
                .as_ref()
                .map(|(existing, _): &(Option<std::time::SystemTime>, PathBuf)| {
                    modified > *existing
                })
                .unwrap_or(true)
            {
                newest = Some((modified, path));
            }
        }
    }
    newest
        .map(|(_, path)| path)
        .ok_or_else(|| "No raw tool logs found.".to_string())
}

fn newest_tool_log(base: &std::path::Path) -> Result<PathBuf, String> {
    let mut newest = None;
    let entries =
        std::fs::read_dir(base).map_err(|e| format!("Failed to list tool logs: {}", e))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("log") {
            continue;
        }
        let modified = entry.metadata().and_then(|m| m.modified()).ok();
        if newest
            .as_ref()
            .map(|(existing, _): &(Option<std::time::SystemTime>, PathBuf)| modified > *existing)
            .unwrap_or(true)
        {
            newest = Some((modified, path));
        }
    }
    newest
        .map(|(_, path)| path)
        .ok_or_else(|| "No raw tool logs found.".to_string())
}

fn detect_reducer(command: &str) -> String {
    let lower = command.to_ascii_lowercase();
    if lower.contains("git diff") {
        "git-diff"
    } else if lower.contains("git status") {
        "git-status"
    } else if lower.contains("cargo test") || lower.contains("cargo build") {
        "cargo"
    } else if lower.contains("pytest") {
        "pytest"
    } else if lower.contains("tsc") || lower.contains("typescript") {
        "typescript"
    } else if lower.contains("npm ")
        || lower.contains("pnpm ")
        || lower.contains("yarn ")
        || lower.contains("vitest")
        || lower.contains("jest")
    {
        "javascript"
    } else if lower.trim_start().starts_with("ls")
        || lower.trim_start().starts_with("tree")
        || lower.trim_start().starts_with("dir")
        || lower.contains("get-childitem")
    {
        "directory-listing"
    } else {
        "generic"
    }
    .to_string()
}

fn reduce_test_or_build(output: &str) -> String {
    let mut selected = Vec::new();
    for line in collapse_repeated_lines(output).lines() {
        let lower = line.to_ascii_lowercase();
        if lower.contains("error")
            || lower.contains("failed")
            || lower.contains("failure")
            || lower.contains("panic")
            || lower.contains("assert")
            || lower.contains("warning")
            || lower.contains("test result")
            || lower.contains("running ")
            || lower.contains("expected")
            || lower.contains("actual")
            || lower.contains("caused by")
            || lower.contains("traceback")
        {
            selected.push(line.to_string());
        }
        if selected.len() >= MAX_SUMMARY_LINES {
            break;
        }
    }

    if selected.is_empty() {
        take_head_tail(output, 80, 40)
    } else {
        selected.join("\n")
    }
}

fn reduce_git_diff(output: &str) -> String {
    let mut lines = Vec::new();
    let mut file_count = 0usize;
    for line in output.lines() {
        if line.starts_with("diff --git ") {
            file_count += 1;
            lines.push(line.to_string());
        } else if line.starts_with("+++ ")
            || line.starts_with("--- ")
            || line.starts_with("@@")
            || line.starts_with("+")
            || line.starts_with("-")
        {
            lines.push(line.to_string());
        }
        if lines.len() >= MAX_SUMMARY_LINES {
            break;
        }
    }
    if file_count > 0 {
        lines.insert(0, format!("diff files touched: {}", file_count));
    }
    if lines.is_empty() {
        take_head_tail(output, 80, 40)
    } else {
        lines.join("\n")
    }
}

fn reduce_git_status(output: &str) -> String {
    output
        .lines()
        .filter(|line| {
            let l = line.trim_start();
            l.starts_with("modified:")
                || l.starts_with("new file:")
                || l.starts_with("deleted:")
                || l.starts_with("renamed:")
                || l.starts_with("both ")
                || l.starts_with("M ")
                || l.starts_with("A ")
                || l.starts_with("D ")
                || l.starts_with("?? ")
                || l.starts_with("## ")
        })
        .take(MAX_SUMMARY_LINES)
        .collect::<Vec<_>>()
        .join("\n")
}

fn reduce_directory_listing(output: &str) -> String {
    let lines: Vec<&str> = output.lines().collect();
    if lines.len() <= MAX_SUMMARY_LINES {
        return output.to_string();
    }
    let mut selected = lines.iter().take(120).copied().collect::<Vec<_>>();
    selected.push("...");
    selected.extend(lines.iter().rev().take(40).rev().copied());
    selected.join("\n")
}

fn reduce_generic(output: &str) -> String {
    let collapsed = collapse_repeated_lines(output);
    if collapsed.lines().count() > MAX_SUMMARY_LINES {
        take_head_tail(&collapsed, 100, 60)
    } else {
        collapsed
    }
}

fn collapse_repeated_lines(output: &str) -> String {
    let mut out = Vec::new();
    let mut last: Option<&str> = None;
    let mut count = 0usize;

    for line in output.lines() {
        if Some(line) == last {
            count += 1;
            continue;
        }
        if let Some(prev) = last {
            if count > 1 {
                out.push(format!("{} [repeated {}x]", prev, count));
            } else {
                out.push(prev.to_string());
            }
        }
        last = Some(line);
        count = 1;
    }
    if let Some(prev) = last {
        if count > 1 {
            out.push(format!("{} [repeated {}x]", prev, count));
        } else {
            out.push(prev.to_string());
        }
    }
    out.join("\n")
}

fn take_head_tail(output: &str, head: usize, tail: usize) -> String {
    let lines: Vec<&str> = output.lines().collect();
    if lines.len() <= head + tail {
        return output.to_string();
    }
    let omitted = lines.len() - head - tail;
    let mut selected = lines.iter().take(head).copied().collect::<Vec<_>>();
    selected.push("...");
    let omitted_line = format!("[{} lines omitted]", omitted);
    selected.push(&omitted_line);
    selected.push("...");
    selected.extend(lines.iter().rev().take(tail).rev().copied());
    selected.join("\n")
}

fn truncate_raw(output: &str) -> String {
    if output.len() <= MAX_RAW_OUTPUT_CHARS {
        return output.to_string();
    }
    let prefix = mangocode_core::truncate::truncate_bytes_prefix(output, MAX_RAW_OUTPUT_CHARS / 2);
    let suffix_start = output.len().saturating_sub(MAX_RAW_OUTPUT_CHARS / 2);
    let suffix = &output[suffix_start..];
    format!(
        "{}\n\n... ({} characters omitted) ...\n\n{}",
        prefix,
        output.len().saturating_sub(prefix.len() + suffix.len()),
        suffix
    )
}

fn save_raw_log(command: &str, output: &str) -> anyhow::Result<PathBuf> {
    let mut last_err = None;
    for dir in tool_log_dirs() {
        match save_raw_log_in_dir(&dir, command, output) {
            Ok(path) => return Ok(path),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no tool-log directory was available")))
}

fn save_raw_log_in_dir(
    dir: &std::path::Path,
    command: &str,
    output: &str,
) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;

    let stamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ");
    let digest = {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        command.hash(&mut h);
        output.hash(&mut h);
        format!("{:x}", h.finish())
    };
    let path = dir.join(format!("{}-{}.log", stamp, &digest[..12]));
    std::fs::write(&path, output)?;
    Ok(path)
}

fn tool_log_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let mut dir = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    dir.push(".mangocode");
    dir.push("tool-logs");
    dirs.push(dir);
    if let Ok(cwd) = std::env::current_dir() {
        dirs.push(cwd.join(".mangocode").join("tool-logs"));
    }
    dirs.push(std::env::temp_dir().join("mangocode").join("tool-logs"));
    dirs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapses_repeated_lines() {
        let reduced = collapse_repeated_lines("a\na\na\nb");
        assert!(reduced.contains("a [repeated 3x]"));
        assert!(reduced.contains("b"));
    }

    #[test]
    fn cargo_reducer_keeps_errors() {
        let out = "Compiling x\nerror[E000]: bad\nnote\nfailures:\n  test_a\n";
        let reduced = reduce_command_output("cargo test", out, 101, OutputMode::Summary);
        assert!(reduced.content.contains("error[E000]"));
        assert!(reduced.content.contains("reducer=cargo"));
    }

    #[test]
    fn raw_mode_does_not_add_header() {
        let reduced = reduce_command_output("cargo test", "hello", 0, OutputMode::Raw);
        assert_eq!(reduced.content, "hello");
        assert_eq!(reduced.reducer, "raw");
    }

    #[test]
    fn reduced_output_points_to_tool_log_read() {
        let reduced = reduce_command_output("cargo test", "error: bad", 101, OutputMode::Summary);
        assert!(reduced.content.contains("ToolLogRead"));
        assert!(reduced.raw_log_path.is_some());
    }

    #[test]
    fn reduced_output_tool_result_carries_raw_log_metadata() {
        let reduced = reduce_command_output("cargo test", "error: bad", 101, OutputMode::Summary);
        let result = reduced.into_tool_result(101, "Command exited with code");
        let raw_log_path = result
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("raw_log_path"))
            .and_then(Value::as_str);
        assert!(raw_log_path.is_some());
    }
}
