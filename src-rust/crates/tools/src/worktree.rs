// Worktree tools: create and exit git worktrees for isolated work sessions.
//
// EnterWorktreeTool – create a new git worktree with an optional branch name,
//                     switching the session's working directory to it.
// ExitWorktreeTool  – exit the current worktree, optionally removing it, and
//                     restore the original working directory.

use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use once_cell::sync::Lazy;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::PathBuf;
#[cfg(any(feature = "tool-enter-worktree", test))]
use std::path::{Component, Path};
use std::sync::Arc;
use tokio::sync::RwLock;
#[cfg(feature = "tool-enter-worktree")]
use tracing::debug;

// ---------------------------------------------------------------------------
// Session-level state: only one active worktree per session.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct WorktreeSession {
    pub original_cwd: PathBuf,
    pub worktree_path: PathBuf,
    pub branch: Option<String>,
    pub original_head: Option<String>,
}

static WORKTREE_SESSION: Lazy<Arc<RwLock<Option<WorktreeSession>>>> =
    Lazy::new(|| Arc::new(RwLock::new(None)));

// ---------------------------------------------------------------------------
// EnterWorktreeTool
// ---------------------------------------------------------------------------

#[cfg(feature = "tool-enter-worktree")]
pub struct EnterWorktreeTool;

#[cfg(feature = "tool-enter-worktree")]
#[derive(Debug, Deserialize)]
struct EnterWorktreeInput {
    /// Optional branch name. If omitted, a timestamped branch is created.
    #[serde(default)]
    branch: Option<String>,
    /// Sub-path under the repo root where the worktree will be created.
    /// Defaults to `.worktrees/<branch>`.
    #[serde(default)]
    path: Option<String>,
    /// Optional shell command to run inside the new worktree directory after creation.
    /// Example: "npm install" or "cargo build".
    #[serde(default)]
    post_create_command: Option<String>,
}

#[cfg(feature = "tool-enter-worktree")]
#[async_trait]
impl Tool for EnterWorktreeTool {
    fn name(&self) -> &str {
        "EnterWorktree"
    }

    fn description(&self) -> &str {
        "Create a new git worktree and switch the session's working directory to it. \
         This gives you an isolated environment to experiment or work on a feature \
         without affecting the main working tree. \
         Use ExitWorktree to return to the original directory."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Write
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "branch": {
                    "type": "string",
                    "description": "Branch name to create. Defaults to a timestamped name like mangocode-20240101-120000."
                },
                "path": {
                    "type": "string",
                    "description": "Optional path for the worktree directory. Defaults to .worktrees/<branch>."
                },
                "post_create_command": {
                    "type": "string",
                    "description": "Optional command to run inside the new worktree after creation (e.g. 'npm install')."
                }
            }
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: EnterWorktreeInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        // Check if already in a worktree session
        {
            let session = WORKTREE_SESSION.read().await;
            if session.is_some() {
                return ToolResult::error(
                    "Already in a worktree session. Call ExitWorktree first.".to_string(),
                );
            }
        }

        if let Err(e) = ctx.check_permission(self.name(), "Create a git worktree", false) {
            return ToolResult::error(e.to_string());
        }

        // Determine branch name — use a human-readable timestamp if none supplied
        let branch = if let Some(branch) = params.branch.as_deref() {
            match normalize_worktree_branch(branch) {
                Ok(branch) => branch,
                Err(e) => return ToolResult::error(e),
            }
        } else {
            // Format: mangocode-YYYYMMDD-HHMMSS
            use std::time::{SystemTime, UNIX_EPOCH};
            let secs = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            // Manual UTC decomposition (no chrono dep in this crate)
            let s = secs % 60;
            let m = (secs / 60) % 60;
            let h = (secs / 3600) % 24;
            let days = secs / 86400;
            // Approximate Gregorian calendar for branch name purposes
            let year = 1970 + days / 365;
            let day_of_year = days % 365;
            let month = day_of_year / 30 + 1;
            let day = day_of_year % 30 + 1;
            format!(
                "mangocode-{:04}{:02}{:02}-{:02}{:02}{:02}",
                year, month, day, h, m, s
            )
        };

        // Verify we are inside a git repository before attempting worktree creation
        let repo_root = match run_git(&ctx.working_dir, &["rev-parse", "--show-toplevel"]).await {
            Ok(root) => {
                let root = root.trim();
                if root.is_empty() {
                    return ToolResult::error(format!(
                        "Cannot create worktree: git did not report a repository root for '{}'.",
                        ctx.working_dir.display()
                    ));
                }
                PathBuf::from(root)
            }
            Err(e) => {
                return ToolResult::error(format!(
                    "Cannot create worktree: the current directory '{}' is not inside a git repository: {}",
                    ctx.working_dir.display(),
                    e.trim()
                ));
            }
        };

        // Determine worktree path
        let worktree_path = match resolve_worktree_path(&repo_root, params.path.as_deref(), &branch)
        {
            Ok(path) => path,
            Err(e) => return ToolResult::error(e),
        };

        let head_result = run_git(&repo_root, &["rev-parse", "--verify", "HEAD"]).await;
        let original_head = match &head_result {
            Ok(h) => Some(h.trim().to_string()),
            Err(e) => {
                let msg = e.to_lowercase();
                if msg.contains("not a git repository") {
                    return ToolResult::error(format!(
                        "Cannot create worktree: the current directory '{}' is not inside a git repository.",
                        ctx.working_dir.display()
                    ));
                }
                None
            }
        };

        // Check if the target path already exists
        if worktree_path.exists() {
            return ToolResult::error(format!(
                "Cannot create worktree: the path '{}' already exists.                  Provide a different 'path' argument or remove the existing directory.",
                worktree_path.display()
            ));
        }

        // Create the worktree
        let worktree_str = worktree_path.to_string_lossy().to_string();
        let result = run_git(
            &repo_root,
            &["worktree", "add", "-b", &branch, "--", &worktree_str],
        )
        .await;

        match result {
            Err(e) => {
                let msg = e.trim().to_string();
                let friendly = if msg.to_lowercase().contains("already exists") {
                    format!(
                        "Failed to create worktree: branch '{}' already exists.                          Use a different branch name or delete the existing branch first.",
                        branch
                    )
                } else if msg.to_lowercase().contains("not a git repository") {
                    format!(
                        "Failed to create worktree: '{}' is not inside a git repository.",
                        ctx.working_dir.display()
                    )
                } else {
                    format!("Failed to create worktree: {}", msg)
                };
                ToolResult::error(friendly)
            }
            Ok(_) => {
                debug!(
                    branch = %branch,
                    path = %worktree_path.display(),
                    "Created worktree"
                );

                // Save session state
                *WORKTREE_SESSION.write().await = Some(WorktreeSession {
                    original_cwd: ctx.working_dir.clone(),
                    worktree_path: worktree_path.clone(),
                    branch: Some(branch.clone()),
                    original_head,
                });

                // Run optional post-create command in the new worktree directory
                let post_create_output = if let Some(cmd) = params.post_create_command {
                    let shell_result = if cfg!(target_os = "windows") {
                        tokio::process::Command::new("cmd")
                            .args(["/C", &cmd])
                            .current_dir(&worktree_path)
                            .output()
                            .await
                    } else {
                        tokio::process::Command::new("sh")
                            .args(["-c", &cmd])
                            .current_dir(&worktree_path)
                            .output()
                            .await
                    };
                    match shell_result {
                        Ok(out) if out.status.success() => {
                            let stdout = String::from_utf8_lossy(&out.stdout);
                            format!(
                                "\nPost-create command '{}' completed successfully.{}",
                                cmd,
                                if stdout.trim().is_empty() {
                                    String::new()
                                } else {
                                    format!("\nOutput: {}", stdout.trim())
                                }
                            )
                        }
                        Ok(out) => {
                            let stderr = String::from_utf8_lossy(&out.stderr);
                            format!(
                                "\nPost-create command '{}' exited with error.\nStderr: {}",
                                cmd,
                                stderr.trim()
                            )
                        }
                        Err(e) => format!("\nCould not run post-create command '{}': {}", cmd, e),
                    }
                } else {
                    String::new()
                };

                ToolResult::success(format!(
                    "Created worktree at {} on branch '{}'.\n\
                     The working directory is now {}.\n\
                     Use ExitWorktree to return to {}.{}",
                    worktree_path.display(),
                    branch,
                    worktree_path.display(),
                    ctx.working_dir.display(),
                    post_create_output,
                ))
                .with_metadata(json!({
                    "worktree_path": worktree_path.to_string_lossy(),
                    "branch": branch,
                    "original_cwd": ctx.working_dir.to_string_lossy(),
                }))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ExitWorktreeTool
// ---------------------------------------------------------------------------

#[cfg(feature = "tool-exit-worktree")]
pub struct ExitWorktreeTool;

#[cfg(feature = "tool-exit-worktree")]
#[derive(Debug, Deserialize)]
struct ExitWorktreeInput {
    /// "keep" = leave the worktree on disk; "remove" = delete it.
    #[serde(default = "default_action")]
    action: String,
    /// Required if action=="remove" and there are uncommitted changes.
    #[serde(default)]
    discard_changes: bool,
}

#[cfg(feature = "tool-exit-worktree")]
fn default_action() -> String {
    "keep".to_string()
}

#[cfg(any(feature = "tool-exit-worktree", test))]
fn normalize_exit_worktree_action(value: &str) -> Result<&'static str, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "keep" => Ok("keep"),
        "remove" => Ok("remove"),
        _ => {
            let value =
                serde_json::to_string(value).unwrap_or_else(|_| "\"<invalid>\"".to_string());
            Err(format!(
                "Invalid action: {value}. Expected one of: keep, remove"
            ))
        }
    }
}

#[cfg(feature = "tool-exit-worktree")]
#[async_trait]
impl Tool for ExitWorktreeTool {
    fn name(&self) -> &str {
        "ExitWorktree"
    }

    fn description(&self) -> &str {
        "Exit the current worktree session created by EnterWorktree and restore the \
         original working directory. Use action='keep' to preserve the worktree on \
         disk, or action='remove' to delete it. Only operates on worktrees created \
         by EnterWorktree in this session."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Write
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["keep", "remove"],
                    "description": "\"keep\" leaves the worktree on disk; \"remove\" deletes it and its branch."
                },
                "discard_changes": {
                    "type": "boolean",
                    "description": "Set true when action=remove and the worktree has uncommitted/unmerged work to discard."
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let mut params: ExitWorktreeInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };
        let action = match normalize_exit_worktree_action(&params.action) {
            Ok(action) => action,
            Err(e) => return ToolResult::error(e),
        };
        params.action = action.to_string();

        let session_guard = WORKTREE_SESSION.read().await;
        let session = match &*session_guard {
            Some(s) => s.clone(),
            None => {
                return ToolResult::error(
                    "No-op: there is no active EnterWorktree session to exit. \
                     This tool only operates on worktrees created by EnterWorktree \
                     in the current session."
                        .to_string(),
                );
            }
        };
        drop(session_guard);

        if let Err(e) = ctx.check_permission(
            self.name(),
            &format!("Exit worktree with action {}", params.action),
            false,
        ) {
            return ToolResult::error(e.to_string());
        }

        let worktree_str = session.worktree_path.to_string_lossy().to_string();

        // If action is "remove", check for uncommitted changes
        if params.action == "remove" && !params.discard_changes {
            let status = match run_git(&session.worktree_path, &["status", "--porcelain"]).await {
                Ok(status) => status,
                Err(e) => {
                    tracing::warn!(
                        path = %session.worktree_path.display(),
                        error = %e,
                        "failed to verify worktree status before removal"
                    );
                    return ToolResult::error(format!(
                        "Could not verify whether worktree {} has uncommitted changes: {}. \
                         The EnterWorktree session is still active; retry, inspect manually, \
                         use action=\"keep\", or set discard_changes=true only after confirming.",
                        session.worktree_path.display(),
                        e.trim()
                    ));
                }
            };
            let changed_files = status.lines().filter(|l| !l.trim().is_empty()).count();

            let commit_count = if let Some(ref head) = session.original_head {
                let rev = match run_git(
                    &session.worktree_path,
                    &["rev-list", "--count", &format!("{}..HEAD", head)],
                )
                .await
                {
                    Ok(rev) => rev,
                    Err(e) => {
                        tracing::warn!(
                            path = %session.worktree_path.display(),
                            original_head = %head,
                            error = %e,
                            "failed to verify worktree commit count before removal"
                        );
                        return ToolResult::error(format!(
                            "Could not verify whether worktree {} has commits after {}: {}. \
                             The EnterWorktree session is still active; retry, inspect manually, \
                             use action=\"keep\", or set discard_changes=true only after confirming.",
                            session.worktree_path.display(),
                            head,
                            e.trim()
                        ));
                    }
                };
                match parse_git_count(&rev) {
                    Ok(count) => count,
                    Err(e) => {
                        tracing::warn!(
                            path = %session.worktree_path.display(),
                            original_head = %head,
                            output = %rev.trim(),
                            error = %e,
                            "failed to parse worktree commit count before removal"
                        );
                        return ToolResult::error(format!(
                            "Could not parse git rev-list output while verifying worktree {}: {}. \
                             The EnterWorktree session is still active; retry or inspect manually.",
                            session.worktree_path.display(),
                            e
                        ));
                    }
                }
            } else {
                0
            };

            if changed_files > 0 || commit_count > 0 {
                let mut parts = Vec::new();
                if changed_files > 0 {
                    parts.push(format!("{} uncommitted file(s)", changed_files));
                }
                if commit_count > 0 {
                    parts.push(format!("{} commit(s) on the worktree branch", commit_count));
                }
                return ToolResult::error(format!(
                    "Worktree has {}. Removing will discard this work permanently. \
                     Confirm with the user, then re-invoke with discard_changes=true — \
                     or use action=\"keep\" to preserve the worktree.",
                    parts.join(" and ")
                ));
            }
        }

        match params.action.as_str() {
            "keep" => {
                // Just remove the worktree from git's tracking list (prune),
                // but keep the directory on disk.
                let lock_warning = match run_git(
                    &session.original_cwd,
                    &[
                        "worktree",
                        "lock",
                        "--reason",
                        "kept by ExitWorktree",
                        &worktree_str,
                    ],
                )
                .await
                {
                    Ok(_) => None,
                    Err(e) => {
                        tracing::warn!(
                            path = %session.worktree_path.display(),
                            error = %e,
                            "failed to lock kept worktree"
                        );
                        Some(format!(
                            "\nWarning: failed to lock kept worktree: {}",
                            e.trim()
                        ))
                    }
                };

                *WORKTREE_SESSION.write().await = None;

                let mut message = format!(
                    "Exited worktree. Work preserved at {} on branch {}. \
                     Session is now back in {}.",
                    session.worktree_path.display(),
                    session.branch.as_deref().unwrap_or("(unknown)"),
                    session.original_cwd.display(),
                );
                if let Some(warning) = lock_warning {
                    message.push_str(&warning);
                }
                ToolResult::success(message)
            }
            "remove" => {
                // Remove the worktree
                if let Err(e) = run_git(
                    &session.original_cwd,
                    &["worktree", "remove", "--force", &worktree_str],
                )
                .await
                {
                    tracing::warn!(
                        path = %session.worktree_path.display(),
                        error = %e,
                        "failed to remove worktree"
                    );
                    return ToolResult::error(format!(
                        "Failed to remove worktree at {}: {}. \
                         The EnterWorktree session is still active; retry removal or use action=\"keep\".",
                        session.worktree_path.display(),
                        e.trim()
                    ));
                }

                // Delete the branch if we created it
                let mut branch_warning = None;
                if let Some(ref branch) = session.branch {
                    if let Err(e) =
                        run_git(&session.original_cwd, &["branch", "-D", "--", branch]).await
                    {
                        tracing::warn!(
                            branch = %branch,
                            error = %e,
                            "failed to delete worktree branch after removing worktree"
                        );
                        branch_warning = Some(format!(
                            "\nWarning: removed the worktree, but failed to delete branch '{}': {}",
                            branch,
                            e.trim()
                        ));
                    }
                }

                *WORKTREE_SESSION.write().await = None;

                let mut message = format!(
                    "Exited and removed worktree at {}. \
                     Session is now back in {}.",
                    session.worktree_path.display(),
                    session.original_cwd.display(),
                );
                if let Some(warning) = branch_warning {
                    message.push_str(&warning);
                }
                ToolResult::success(message)
            }
            other => ToolResult::error(format!(
                "Unknown action '{}'. Use 'keep' or 'remove'.",
                other
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

#[cfg(any(feature = "tool-enter-worktree", test))]
fn normalize_worktree_branch(value: &str) -> Result<String, String> {
    let branch = value.trim();
    if branch.is_empty() {
        return Err("Worktree branch name cannot be empty.".to_string());
    }
    if branch.starts_with('-') {
        return Err(format!(
            "Invalid worktree branch '{}': branch names cannot start with '-'.",
            branch
        ));
    }
    if branch.starts_with('/') || branch.ends_with('/') || branch.contains("//") {
        return Err(format!(
            "Invalid worktree branch '{}': use a normal branch name such as feature/example.",
            branch
        ));
    }
    if branch.ends_with('.') || branch.contains("..") || branch.contains("@{") {
        return Err(format!(
            "Invalid worktree branch '{}': branch name is not a valid git ref.",
            branch
        ));
    }
    if branch
        .split('/')
        .any(|part| part.is_empty() || part.starts_with('.') || part.ends_with(".lock"))
    {
        return Err(format!(
            "Invalid worktree branch '{}': branch name is not a valid git ref.",
            branch
        ));
    }
    if branch.chars().any(|ch| {
        ch.is_control()
            || ch.is_whitespace()
            || matches!(ch, '~' | '^' | ':' | '?' | '*' | '[' | '\\')
    }) {
        return Err(format!(
            "Invalid worktree branch '{}': branch name contains unsupported characters.",
            branch
        ));
    }

    Ok(branch.to_string())
}

#[cfg(any(feature = "tool-enter-worktree", test))]
fn resolve_worktree_path(
    repo_root: &Path,
    requested_path: Option<&str>,
    branch: &str,
) -> Result<PathBuf, String> {
    let default_path;
    let raw = if let Some(path) = requested_path {
        path.trim()
    } else {
        default_path = format!(".worktrees/{branch}");
        default_path.as_str()
    };

    if raw.is_empty() {
        return Err("Worktree path cannot be empty.".to_string());
    }

    let path = Path::new(raw);
    if path.is_absolute() {
        return Err(format!(
            "Worktree path '{}' must be relative to the repository root.",
            raw
        ));
    }

    let mut relative = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => relative.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(format!(
                    "Worktree path '{}' must stay inside the repository root.",
                    raw
                ));
            }
        }
    }

    if relative.as_os_str().is_empty() {
        return Err("Worktree path cannot be empty.".to_string());
    }

    Ok(repo_root.join(relative))
}

async fn run_git(cwd: &std::path::Path, args: &[&str]) -> Result<String, String> {
    let output = tokio::process::Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .await
        .map_err(|e| e.to_string())?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}

fn parse_git_count(output: &str) -> Result<usize, String> {
    let trimmed = output.trim();
    trimmed
        .parse::<usize>()
        .map_err(|e| format!("expected numeric count, got {trimmed:?}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_worktree_action_is_normalized_or_rejected() {
        assert_eq!(normalize_exit_worktree_action(" REMOVE "), Ok("remove"));
        let err = normalize_exit_worktree_action("keep\nremove")
            .expect_err("invalid action must be rejected");
        assert!(err.contains("Invalid action"));
        assert!(err.contains("\"keep\\nremove\""));
    }

    #[test]
    fn worktree_branch_names_are_normalized_or_rejected() {
        assert_eq!(
            normalize_worktree_branch(" feature/example-1 ").unwrap(),
            "feature/example-1"
        );

        for branch in [
            "-bad",
            "feature branch",
            "feature/../bad",
            ".hidden",
            "bad.lock",
        ] {
            assert!(
                normalize_worktree_branch(branch).is_err(),
                "branch should be rejected: {branch}"
            );
        }
    }

    #[test]
    fn worktree_paths_stay_under_repo_root() {
        let repo_root = PathBuf::from(r"C:\repo");

        assert_eq!(
            resolve_worktree_path(&repo_root, None, "feature/demo").unwrap(),
            repo_root.join(".worktrees").join("feature").join("demo")
        );
        assert_eq!(
            resolve_worktree_path(&repo_root, Some("scratch/demo"), "feature/demo").unwrap(),
            repo_root.join("scratch").join("demo")
        );

        for path in ["", "../outside", "scratch/../../outside"] {
            assert!(
                resolve_worktree_path(&repo_root, Some(path), "feature/demo").is_err(),
                "path should be rejected: {path}"
            );
        }
    }

    #[test]
    fn parse_git_count_rejects_unexpected_output() {
        assert_eq!(parse_git_count("42\n"), Ok(42));
        let err = parse_git_count("fatal: bad revision").expect_err("nonnumeric output is unsafe");
        assert!(err.contains("expected numeric count"));
    }
}
