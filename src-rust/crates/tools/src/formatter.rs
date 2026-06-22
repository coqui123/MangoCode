// formatter.rs — run a configured file formatter after writes/edits.

use crate::ToolContext;
use mangocode_core::config::FormatterConfig;
use std::fmt;
use std::time::Duration;

const FORMATTER_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, PartialEq, Eq)]
enum FormatterRunError {
    Spawn(String),
    Timeout,
    NonZeroExit { code: Option<i32>, stderr: String },
}

impl fmt::Display for FormatterRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn(error) => write!(f, "failed to start formatter: {error}"),
            Self::Timeout => write!(f, "formatter timed out after {:?}", FORMATTER_TIMEOUT),
            Self::NonZeroExit { code, stderr } => {
                write!(
                    f,
                    "formatter exited with status {}",
                    code.map(|value| value.to_string())
                        .unwrap_or_else(|| "unknown".to_string())
                )?;
                let stderr = stderr.trim();
                if !stderr.is_empty() {
                    write!(f, ": {stderr}")?;
                }
                Ok(())
            }
        }
    }
}

fn formatter_args_with_file(command: &[String], path: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut file_injected = false;
    for arg in &command[1..] {
        if arg == "$FILE" || arg == "{file}" {
            args.push(path.to_string());
            file_injected = true;
        } else {
            args.push(arg.clone());
        }
    }
    if !file_injected {
        args.push(path.to_string());
    }
    args
}

async fn run_formatter(fmt: &FormatterConfig, path: &str) -> Result<(), FormatterRunError> {
    let args = formatter_args_with_file(&fmt.command, path);
    let mut cmd = tokio::process::Command::new(&fmt.command[0]);
    cmd.args(&args);

    match tokio::time::timeout(FORMATTER_TIMEOUT, cmd.output()).await {
        Err(_) => Err(FormatterRunError::Timeout),
        Ok(Err(err)) => Err(FormatterRunError::Spawn(err.to_string())),
        Ok(Ok(output)) if !output.status.success() => {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            Err(FormatterRunError::NonZeroExit {
                code: output.status.code(),
                stderr: mangocode_core::truncate::truncate_bytes_prefix(&stderr, 600).to_string(),
            })
        }
        Ok(Ok(_)) => Ok(()),
    }
}

/// Try to format a file using a configured formatter, or a built-in default.
/// Returns silently if nothing handles the file's extension. Formatter failures
/// are logged (not propagated) because formatting is a post-edit convenience,
/// not a reason to discard the already-applied file change.
pub async fn try_format_file(path: &str, ctx: &ToolContext) {
    // Determine the file's extension (with leading dot, e.g. ".ts").
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{}", e))
        .unwrap_or_default();

    // User-configured formatters take precedence. A user entry that names this
    // extension "claims" it (even when disabled), which suppresses the built-in
    // default below so an explicit opt-out is honored.
    let mut user_claims_ext = false;
    for fmt in ctx.config.formatter.values() {
        if !fmt.extensions.iter().any(|e| e == &ext) {
            continue;
        }
        user_claims_ext = true;
        if fmt.disabled || fmt.command.is_empty() {
            continue;
        }
        if let Err(err) = run_formatter(fmt, path).await {
            tracing::warn!(
                error = %err,
                path = %path,
                command = ?fmt.command,
                "configured formatter failed"
            );
        }
        // Only apply the first matching formatter.
        return;
    }

    // Built-in default, used only when no user formatter claims this extension.
    // Best-effort: if the tool isn't installed the run fails quietly (debug log)
    // and never blocks the edit.
    if user_claims_ext {
        return;
    }
    if let Some(builtin) = builtin_formatter(&ext) {
        if let Err(err) = run_formatter(&builtin, path).await {
            tracing::debug!(
                error = %err,
                path = %path,
                "built-in formatter unavailable or failed"
            );
        }
    }
}

/// Built-in default formatters, applied when the user hasn't configured one for
/// the file's extension. Currently: rustfmt for `.rs`. Best-effort only.
fn builtin_formatter(ext: &str) -> Option<FormatterConfig> {
    match ext {
        ".rs" => Some(FormatterConfig {
            command: vec![
                "rustfmt".to_string(),
                "--edition".to_string(),
                "2021".to_string(),
            ],
            extensions: vec![".rs".to_string()],
            disabled: false,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_formatter_covers_rust_only() {
        let rs = builtin_formatter(".rs").expect("rust has a built-in formatter");
        assert_eq!(rs.command.first().map(String::as_str), Some("rustfmt"));
        assert!(rs.command.iter().any(|a| a == "--edition"));
        assert!(builtin_formatter(".ts").is_none());
        assert!(builtin_formatter("").is_none());
    }

    fn formatter(command: Vec<&str>) -> FormatterConfig {
        FormatterConfig {
            command: command.into_iter().map(str::to_string).collect(),
            extensions: vec![".rs".to_string()],
            disabled: false,
        }
    }

    #[test]
    fn formatter_args_append_file_when_no_placeholder_is_present() {
        let fmt = formatter(vec!["rustfmt", "--edition", "2021"]);

        assert_eq!(
            formatter_args_with_file(&fmt.command, "src/main.rs"),
            vec!["--edition", "2021", "src/main.rs"]
        );
    }

    #[test]
    fn formatter_args_replace_supported_file_placeholders() {
        let fmt = formatter(vec!["prettier", "--write", "$FILE", "{file}"]);

        assert_eq!(
            formatter_args_with_file(&fmt.command, "src/main.ts"),
            vec!["--write", "src/main.ts", "src/main.ts"]
        );
    }

    #[tokio::test]
    async fn run_formatter_reports_spawn_failure() {
        let fmt = formatter(vec!["__mangocode_missing_formatter_command__"]);

        let err = run_formatter(&fmt, "src/main.rs").await.unwrap_err();

        assert!(matches!(err, FormatterRunError::Spawn(_)));
    }
}
