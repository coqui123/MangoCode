//! RTK-style tool output reduction.
//!
//! Reducers keep the model-facing output small while preserving raw command
//! logs on disk for follow-up inspection.

#![cfg_attr(not(feature = "tool-tool-log-read"), allow(dead_code, unused_imports))]

use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::borrow::Cow;
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
    let original_tokens = estimate_tokens(output);
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

    let raw_log_path = match save_raw_log(command, output) {
        Ok(path) => Some(path),
        Err(err) => {
            tracing::warn!(error = %err, "failed to save raw tool output log");
            None
        }
    };
    let cleaned = strip_ansi(output);
    // Redact known secret patterns (API keys, tokens) before the output reaches
    // the model or the reduced result. High-confidence formats only, so normal
    // output is untouched (borrowed fast-path when nothing matches).
    let redacted = crate::redact::redact_secrets(cleaned.as_ref());
    let output_for_reducer = redacted.as_ref();
    let reducer = detect_reducer(command, output_for_reducer);
    let reduced = match reducer.as_str() {
        "git-diff" => reduce_git_diff(output_for_reducer),
        "git-status" => reduce_git_status(output_for_reducer),
        "git-log" => reduce_git_log(output_for_reducer),
        "grep" => reduce_grep_matches(output_for_reducer),
        "find" => reduce_find_paths(output_for_reducer),
        "directory-listing" => reduce_directory_listing_cow(output_for_reducer).into_owned(),
        "cargo" | "pytest" | "javascript" | "typescript" => {
            reduce_test_or_build(output_for_reducer)
        }
        _ => reduce_generic(output_for_reducer),
    };

    let should_keep_raw = mode == OutputMode::Auto
        && reduced.len() >= output.len()
        && output.len() <= MAX_RAW_OUTPUT_CHARS;
    let body: Cow<'_, str> = if should_keep_raw {
        Cow::Borrowed(output)
    } else {
        Cow::Owned(reduced)
    };
    let body = truncate_owned_or_borrowed(body);
    let reduced_tokens = estimate_tokens(&body);
    let saved_tokens = original_tokens.saturating_sub(reduced_tokens);
    let saved_pct = if original_tokens == 0 {
        0
    } else {
        (saved_tokens * 100) / original_tokens
    };
    let raw_path = raw_log_path
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "(not written)".to_string());
    let content = format!(
        "[tool-output reducer={} exit_code={} original={}B/{}tok reduced={}B/{}tok saved={}tok/{}% raw_log={} recall=\"ToolLogRead path='<raw_log>'\"]\n{}",
        reducer,
        exit_code,
        original_bytes,
        original_tokens,
        body.len(),
        reduced_tokens,
        saved_tokens,
        saved_pct,
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
        let mut allowed_dirs = Vec::new();
        for dir in tool_log_dirs() {
            match dir.canonicalize() {
                Ok(dir) => allowed_dirs.push(dir),
                // A candidate dir that simply hasn't been created yet is the
                // normal case (e.g. the per-cwd and temp dirs); only the
                // home dir is created on first write. Don't warn for that.
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                    tracing::debug!(
                        path = %dir.display(),
                        "tool-log directory does not exist yet; skipping"
                    );
                }
                Err(err) => {
                    tracing::warn!(
                        path = %dir.display(),
                        error = %err,
                        "failed to resolve tool-log directory"
                    );
                }
            }
        }
        let allowed = allowed_dirs.iter().any(|dir| canonical.starts_with(dir));
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
    let mut errors = Vec::new();
    for dir in tool_log_dirs() {
        if !dir.exists() {
            continue;
        }
        match newest_tool_log(&dir) {
            Ok(path) => {
                let modified = match path.metadata().and_then(|m| m.modified()) {
                    Ok(modified) => modified,
                    Err(err) => {
                        tracing::warn!(
                            path = %path.display(),
                            error = %err,
                            "failed to inspect latest tool log"
                        );
                        std::time::SystemTime::UNIX_EPOCH
                    }
                };
                if newest
                    .as_ref()
                    .map(|(existing, _): &(std::time::SystemTime, PathBuf)| modified > *existing)
                    .unwrap_or(true)
                {
                    newest = Some((modified, path));
                }
            }
            Err(err) => {
                tracing::warn!(
                    path = %dir.display(),
                    error = %err,
                    "failed to inspect tool-log directory"
                );
                errors.push(format!("{}: {}", dir.display(), err));
            }
        }
    }
    if let Some((_, path)) = newest {
        Ok(path)
    } else if errors.is_empty() {
        Err("No raw tool logs found.".to_string())
    } else {
        Err(format!(
            "No raw tool logs found; failed to inspect: {}",
            errors.join("; ")
        ))
    }
}

fn newest_tool_log(base: &std::path::Path) -> Result<PathBuf, String> {
    let mut newest = None;
    let entries =
        std::fs::read_dir(base).map_err(|e| format!("Failed to list tool logs: {}", e))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read tool log entry: {}", e))?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("log") {
            continue;
        }
        let modified = entry
            .metadata()
            .and_then(|m| m.modified())
            .map_err(|e| format!("Failed to inspect tool log '{}': {}", path.display(), e))?;
        if newest
            .as_ref()
            .map(|(existing, _): &(std::time::SystemTime, PathBuf)| modified > *existing)
            .unwrap_or(true)
        {
            newest = Some((modified, path));
        }
    }
    newest
        .map(|(_, path)| path)
        .ok_or_else(|| "No raw tool logs found.".to_string())
}

fn detect_reducer(command: &str, output: &str) -> String {
    let lower = command.to_ascii_lowercase();
    if lower.contains("git diff") {
        "git-diff"
    } else if lower.contains("git status") {
        "git-status"
    } else if lower.contains("git log") || lower.contains("git show") {
        "git-log"
    } else if lower.contains("rg --files") || lower.contains("fd ") {
        "find"
    } else if lower.contains("cargo test")
        || lower.contains("cargo build")
        || lower.contains("cargo check")
        || lower.contains("cargo clippy")
        || lower.contains("cargo install")
        || lower.contains("cargo nextest")
        || lower.contains("go test")
        || lower.contains("go build")
        || lower.contains("mypy")
        || lower.contains("ruff ")
    {
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
    } else if command_starts_with(&lower, &["rg", "grep", "select-string"]) {
        "grep"
    } else if command_starts_with(&lower, &["find", "fd"]) {
        "find"
    } else if lower.trim_start().starts_with("ls")
        || lower.trim_start().starts_with("tree")
        || lower.trim_start().starts_with("dir")
        || lower.contains("get-childitem")
    {
        "directory-listing"
    } else if looks_like_grep_output(output) {
        "grep"
    } else if looks_like_find_output(output) {
        "find"
    } else {
        "generic"
    }
    .to_string()
}

fn command_starts_with(command: &str, names: &[&str]) -> bool {
    let command = command.trim_start();
    names.iter().any(|name| {
        command == *name
            || command
                .strip_prefix(name)
                .map(|rest| rest.starts_with(char::is_whitespace))
                .unwrap_or(false)
    })
}

fn reduce_test_or_build(output: &str) -> String {
    let compact = reduce_cargo_like_summary(output);
    if !compact.trim().is_empty() {
        return compact;
    }

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

fn reduce_cargo_like_summary(output: &str) -> String {
    let mut compiled = 0usize;
    let mut warnings = 0usize;
    let mut errors = 0usize;
    let mut finished = None;
    let mut test_results = Vec::new();
    let mut failure_blocks = Vec::new();
    let mut in_failure = false;
    let mut current_failure = Vec::new();

    for line in output.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("Compiling ")
            || trimmed.starts_with("Checking ")
            || trimmed.starts_with("Building ")
        {
            compiled += 1;
            continue;
        }
        if trimmed.starts_with("Finished ") {
            finished = Some(trimmed.to_string());
            continue;
        }
        if trimmed.starts_with("warning:") || trimmed.starts_with("warning[") {
            warnings += 1;
        }
        if trimmed.starts_with("error:") || trimmed.starts_with("error[") {
            errors += 1;
        }
        if line.starts_with("test result:") {
            test_results.push(line.trim().to_string());
            in_failure = false;
            if !current_failure.is_empty() {
                failure_blocks.push(current_failure.join("\n"));
                current_failure.clear();
            }
            continue;
        }
        if line.starts_with("---- ") {
            if !current_failure.is_empty() {
                failure_blocks.push(current_failure.join("\n"));
                current_failure.clear();
            }
            in_failure = true;
        }
        if in_failure {
            current_failure.push(line.to_string());
            if current_failure.len() >= 30 {
                failure_blocks.push(current_failure.join("\n"));
                current_failure.clear();
                in_failure = false;
            }
        }
    }
    if !current_failure.is_empty() {
        failure_blocks.push(current_failure.join("\n"));
    }

    if compiled == 0 && warnings == 0 && errors == 0 && test_results.is_empty() {
        return String::new();
    }

    let mut out = Vec::new();
    if !test_results.is_empty() {
        if let Some(aggregate) = aggregate_rust_test_results(&test_results) {
            out.push(aggregate);
        } else {
            out.extend(test_results);
        }
    } else if errors == 0 && warnings == 0 {
        out.push(format!("build ok: {compiled} crates compiled"));
        if let Some(finished) = finished {
            out.push(finished);
        }
    } else {
        out.push(format!(
            "build summary: {errors} errors, {warnings} warnings, {compiled} crates compiled"
        ));
    }

    for block in failure_blocks.into_iter().take(5) {
        out.push(block);
    }
    if errors > 0 || warnings > 0 {
        out.extend(
            output
                .lines()
                .filter(|line| {
                    let trimmed = line.trim_start();
                    trimmed.starts_with("error:")
                        || trimmed.starts_with("error[")
                        || trimmed.starts_with("warning:")
                        || trimmed.starts_with("warning[")
                        || trimmed.starts_with("-->")
                })
                .take(80)
                .map(str::to_string),
        );
    }
    out.join("\n")
}

fn aggregate_rust_test_results(lines: &[String]) -> Option<String> {
    let mut suites = 0usize;
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut ignored = 0usize;
    let mut measured = 0usize;
    let mut filtered = 0usize;
    let mut all_parsed = true;

    for line in lines {
        let Some(after) = line.strip_prefix("test result:") else {
            all_parsed = false;
            break;
        };
        suites += 1;
        let lower = after.to_ascii_lowercase();
        passed += metric_before(&lower, "passed").unwrap_or(0);
        failed += metric_before(&lower, "failed").unwrap_or(0);
        ignored += metric_before(&lower, "ignored").unwrap_or(0);
        measured += metric_before(&lower, "measured").unwrap_or(0);
        filtered += metric_before(&lower, "filtered").unwrap_or(0);
    }

    all_parsed.then(|| {
        format!(
            "cargo test: {passed} passed, {failed} failed ({suites} suites, {ignored} ignored, {measured} measured, {filtered} filtered)"
        )
    })
}

fn metric_before(text: &str, label: &str) -> Option<usize> {
    let index = text.find(label)?;
    text[..index].split_whitespace().last()?.parse().ok()
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

fn reduce_git_log(output: &str) -> String {
    output
        .lines()
        .filter(|line| !line.trim().is_empty())
        .take(80)
        .map(|line| {
            let mut fields = line.split_whitespace();
            let hash = fields.next().unwrap_or("");
            if hash.len() >= 7 {
                line.to_string()
            } else {
                line.trim().to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
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

fn reduce_grep_matches(output: &str) -> String {
    use std::collections::BTreeMap;

    let mut by_file: BTreeMap<&str, Vec<(&str, &str)>> = BTreeMap::new();
    let mut total = 0usize;
    for line in output.lines() {
        let mut parts = line.splitn(3, ':');
        let file = parts.next().unwrap_or("");
        let line_number = parts.next().unwrap_or("");
        let content = parts.next().unwrap_or("");
        if file.is_empty() || line_number.parse::<usize>().is_err() {
            continue;
        }
        total += 1;
        by_file
            .entry(file)
            .or_default()
            .push((line_number, content));
    }

    if total == 0 {
        return reduce_generic(output);
    }

    let mut out = vec![format!("{total} matches in {} files", by_file.len())];
    for (file, matches) in by_file.iter().take(40) {
        out.push(format!("{file} ({})", matches.len()));
        for (line_number, content) in matches.iter().take(8) {
            out.push(format!("  {line_number}: {}", content.trim()));
        }
        if matches.len() > 8 {
            out.push(format!("  +{} more", matches.len() - 8));
        }
    }
    if by_file.len() > 40 {
        out.push(format!("+{} more files", by_file.len() - 40));
    }
    out.join("\n")
}

fn reduce_find_paths(output: &str) -> String {
    use std::collections::BTreeMap;

    let paths: Vec<&str> = output
        .lines()
        .map(str::trim)
        .filter(|line| is_path_like_line(line))
        .collect();
    if paths.len() < 3 {
        return reduce_generic(output);
    }

    let mut by_dir: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for path in &paths {
        let normalized = path.trim_end_matches(std::path::MAIN_SEPARATOR);
        let split = normalized
            .rfind(['/', '\\'])
            .map(|index| (&normalized[..index], &normalized[index + 1..]))
            .unwrap_or((".", normalized));
        by_dir.entry(split.0).or_default().push(split.1);
    }

    let mut out = vec![format!("{} paths in {} dirs", paths.len(), by_dir.len())];
    for (dir, files) in by_dir.iter().take(30) {
        out.push(format!("{dir}/ ({})", files.len()));
        for file in files.iter().take(8) {
            out.push(format!("  {file}"));
        }
        if files.len() > 8 {
            out.push(format!("  +{} more", files.len() - 8));
        }
    }
    if by_dir.len() > 30 {
        out.push(format!("+{} more dirs", by_dir.len() - 30));
    }
    out.join("\n")
}

fn reduce_directory_listing_cow(output: &str) -> Cow<'_, str> {
    let lines: Vec<&str> = output.lines().collect();
    if lines.len() <= MAX_SUMMARY_LINES {
        return Cow::Borrowed(output);
    }
    let mut selected: Vec<String> = lines.iter().take(120).map(|s| (*s).to_string()).collect();
    selected.push("...".to_string());
    selected.extend(lines.iter().rev().take(40).rev().map(|s| (*s).to_string()));
    Cow::Owned(selected.join("\n"))
}

fn reduce_generic(output: &str) -> String {
    let collapsed = collapse_repeated_lines(output);
    if collapsed.lines().count() > MAX_SUMMARY_LINES {
        take_head_tail(&collapsed, 100, 60)
    } else {
        collapsed
    }
}

fn looks_like_grep_output(output: &str) -> bool {
    let mut checked = 0usize;
    let mut matches = 0usize;
    for line in output
        .lines()
        .filter(|line| !line.trim().is_empty())
        .take(8)
    {
        checked += 1;
        let parts: Vec<_> = line.splitn(3, ':').collect();
        if parts.len() == 3 && parts[1].parse::<usize>().is_ok() {
            matches += 1;
        }
    }
    checked >= 2 && matches * 2 >= checked
}

fn looks_like_find_output(output: &str) -> bool {
    let mut checked = 0usize;
    for line in output
        .lines()
        .filter(|line| !line.trim().is_empty())
        .take(8)
    {
        checked += 1;
        let line = line.trim();
        if !is_path_like_line(line) {
            return false;
        }
    }
    checked >= 3
}

fn is_path_like_line(line: &str) -> bool {
    let line = line.trim();
    !line.is_empty()
        && !line.contains(':')
        && !line.starts_with("--- ")
        && (line.starts_with('.')
            || line.starts_with('/')
            || line.contains('/')
            || line.contains('\\'))
}

fn strip_ansi(text: &str) -> Cow<'_, str> {
    // Delegate to the shared, more complete stripper (handles OSC, charset
    // designators, and the full CSI final-byte range — not just `ESC[…alpha`).
    crate::ansi::strip_ansi(text)
}

fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
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
    take_head_tail_cow(output, head, tail).into_owned()
}

fn take_head_tail_cow(output: &str, head: usize, tail: usize) -> Cow<'_, str> {
    let lines: Vec<&str> = output.lines().collect();
    if lines.len() <= head + tail {
        return Cow::Borrowed(output);
    }
    let omitted = lines.len() - head - tail;
    let mut selected: Vec<String> = lines.iter().take(head).map(|s| (*s).to_string()).collect();
    selected.push("...".to_string());
    selected.push(format!("[{} lines omitted]", omitted));
    selected.push("...".to_string());
    selected.extend(
        lines
            .iter()
            .rev()
            .take(tail)
            .rev()
            .map(|s| (*s).to_string()),
    );
    Cow::Owned(selected.join("\n"))
}

pub(crate) fn truncate_middle_bytes(output: &str, max_bytes: usize) -> Cow<'_, str> {
    if output.len() <= max_bytes {
        return Cow::Borrowed(output);
    }

    let half = max_bytes / 2;
    let prefix = mangocode_core::truncate::truncate_bytes_prefix(output, half);
    let mut suffix_start = output.len().saturating_sub(half);
    while suffix_start < output.len() && !output.is_char_boundary(suffix_start) {
        suffix_start += 1;
    }
    let suffix = &output[suffix_start..];
    Cow::Owned(format!(
        "{}\n\n... ({} bytes omitted) ...\n\n{}",
        prefix,
        output.len().saturating_sub(prefix.len() + suffix.len()),
        suffix
    ))
}

fn truncate_raw(output: &str) -> String {
    truncate_middle_bytes(output, MAX_RAW_OUTPUT_CHARS).into_owned()
}

fn truncate_owned_or_borrowed(output: Cow<'_, str>) -> Cow<'_, str> {
    match output {
        Cow::Borrowed(output) => truncate_middle_bytes(output, MAX_RAW_OUTPUT_CHARS),
        Cow::Owned(output) => {
            if output.len() <= MAX_RAW_OUTPUT_CHARS {
                Cow::Owned(output)
            } else {
                Cow::Owned(truncate_raw(&output))
            }
        }
    }
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
    // Redact secrets before persisting, so the on-disk raw log doesn't retain
    // API keys/tokens echoed by a command.
    std::fs::write(&path, crate::redact::redact_secrets(output).as_ref())?;
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
        assert!(reduced.content.contains("saved="));
    }

    #[test]
    fn raw_mode_does_not_add_header() {
        let reduced = reduce_command_output("cargo test", "hello", 0, OutputMode::Raw);
        assert_eq!(reduced.content, "hello");
        assert_eq!(reduced.reducer, "raw");
    }

    #[test]
    fn middle_truncation_handles_multibyte_boundaries() {
        let output = truncate_middle_bytes("abcédefghij", 8);

        assert!(output.starts_with("abc"));
        assert!(!output.starts_with("abcé"));
        assert!(output.ends_with("ghij"));
        assert!(output.contains("bytes omitted"));
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

    #[test]
    fn cargo_success_collapses_compile_noise() {
        let out = "   Compiling a v0.1.0\n   Compiling b v0.1.0\n    Finished dev [unoptimized + debuginfo] target(s) in 1.23s\n";
        let reduced = reduce_command_output("cargo build", out, 0, OutputMode::Summary);
        assert!(reduced.content.contains("build ok: 2 crates compiled"));
        assert!(!reduced.content.contains("Compiling a"));
    }

    #[test]
    fn cargo_test_results_are_aggregated() {
        let out = "running 5 tests\ntest result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\nrunning 2 tests\ntest result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n";
        let reduced = reduce_command_output("cargo test", out, 0, OutputMode::Summary);
        assert!(reduced
            .content
            .contains("cargo test: 7 passed, 0 failed (2 suites"));
    }

    #[test]
    fn grep_output_groups_by_file() {
        let input = "src/a.rs:1:alpha\nsrc/a.rs:2:beta\nsrc/b.rs:9:gamma\n";
        let reduced = reduce_command_output("rg alpha", input, 0, OutputMode::Summary);
        assert!(reduced.content.contains("3 matches in 2 files"));
        assert!(reduced.content.contains("src/a.rs (2)"));
    }

    #[test]
    fn rg_files_output_uses_find_reducer() {
        let input = "./src/main.rs\n./src/lib.rs\n./tests/smoke.rs\n";
        let reduced = reduce_command_output("rg --files", input, 0, OutputMode::Summary);
        assert!(reduced.content.contains("reducer=find"));
        assert!(reduced.content.contains("3 paths"));
    }

    #[test]
    fn ansi_sequences_are_removed_before_reduction() {
        let input = "\u{1b}[31merror: bad\u{1b}[0m\n";
        let reduced = reduce_command_output("cargo check", input, 101, OutputMode::Summary);
        assert!(reduced.content.contains("error: bad"));
        assert!(!reduced.content.contains("\u{1b}[31m"));
    }

    #[test]
    fn secrets_are_redacted_before_reduction() {
        let input = "exporting ANTHROPIC_API_KEY=sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAA done\n";
        let reduced = reduce_command_output("printenv", input, 0, OutputMode::Summary);
        assert!(
            !reduced.content.contains("sk-ant-"),
            "secret leaked into reduced output: {}",
            reduced.content
        );
        assert!(reduced.content.contains("[REDACTED]"));
    }
}
