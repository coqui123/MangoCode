// mangocode-tools: All tool implementations for MangoCode.
//
// Each tool maps to a capability the LLM can invoke: running shell commands,
// reading/writing/editing files, searching codebases, fetching web pages, etc.

use async_trait::async_trait;
use mangocode_core::config::PermissionMode;
use mangocode_core::cost::CostTracker;
use mangocode_core::permissions::{PermissionDecision, PermissionHandler, PermissionRequest};
use mangocode_core::types::ToolDefinition;
use mangocode_tool_runtime::{
    ApprovalKey, SandboxPreference, ToolCapabilities, ToolErrorKind, ToolHandlerKind,
    ToolOutputEnvelope, ToolRegistryPlan, ToolSpec,
};
use serde_json::Value;
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// Anti-bot HTML heuristics (shared by research HTTP + browser tooling).
#[cfg(any(
    feature = "tool-web-fetch",
    feature = "tool-doc-search",
    feature = "tool-doc-read",
    feature = "tool-deep-read",
    feature = "tool-rendered-fetch",
    feature = "tool-browser"
))]
pub(crate) mod bot_wall_sniff;

// Sub-modules – each contains a full tool implementation.
#[cfg(feature = "tool-apply-patch")]
pub mod apply_patch;
// HITL question types (QuestionPrompt, QuestionResponse, ...) live in this
// module and are used by ToolContext, so the module must compile even when
// the AskUserQuestionTool itself is not registered. The Tool impl is
// feature-gated inside the file.
pub mod ansi;
pub mod ask_user;
#[cfg(feature = "tool-bash")]
pub mod bash;
#[cfg(feature = "tool-batch-edit")]
pub mod batch_edit;
#[cfg(feature = "tool-brief")]
pub mod brief;
#[cfg(any(feature = "tool-browser", feature = "tool-rendered-fetch"))]
pub mod browser_antibot;
pub mod browser_tool;
#[cfg(feature = "tool-skill")]
pub mod bundled_skills;
#[cfg(feature = "tool-grep")]
pub mod circuit_breaker;
pub mod code_search_tool;
#[cfg(feature = "tool-computer-use")]
pub mod computer_use;
#[cfg(feature = "tool-config")]
pub mod config_tool;
pub mod coordination;
pub mod cron;
pub mod edit_hints;
#[cfg(feature = "tool-enter-plan-mode")]
pub mod enter_plan_mode;
#[cfg(feature = "tool-exit-plan-mode")]
pub mod exit_plan_mode;
#[cfg(feature = "tool-edit")]
pub mod file_edit;
#[cfg(feature = "tool-read")]
pub mod file_read;
#[cfg(feature = "tool-write")]
pub mod file_write;
pub mod formatter;
pub mod fs_atomic;
#[cfg(feature = "tool-glob")]
pub mod glob_tool;
#[cfg(any(
    feature = "tool-get-goal",
    feature = "tool-create-goal",
    feature = "tool-update-goal"
))]
pub mod goal_tool;
#[cfg(feature = "tool-grep")]
pub mod grep_tool;
pub mod humanize;
#[cfg(feature = "tool-lsp")]
pub mod lsp_tool;
#[cfg(feature = "tool-mcp-auth")]
pub mod mcp_auth_tool;
#[cfg(any(
    feature = "tool-list-mcp-resources",
    feature = "tool-read-mcp-resource"
))]
pub mod mcp_resources;
#[cfg(feature = "tool-notebook-edit")]
pub mod notebook_edit;
pub mod output_reducers;
#[cfg(feature = "tool-powershell")]
pub mod powershell;
pub mod pr_watch;
#[cfg(feature = "tool-project-graph")]
pub mod project_graph;
#[cfg(feature = "tool-bash")]
pub mod pty_bash;
pub mod redact;
#[cfg(feature = "tool-remote-trigger")]
pub mod remote_trigger;
#[cfg(feature = "tool-repl")]
pub mod repl_tool;
#[cfg(any(
    feature = "tool-browser",
    feature = "tool-doc-search",
    feature = "tool-doc-read",
    feature = "tool-deep-read",
    feature = "tool-rendered-fetch",
    feature = "tool-web-fetch"
))]
pub mod research;
#[cfg(feature = "tool-send-message")]
pub mod send_message;
#[cfg(feature = "tool-skill")]
pub mod skill_tool;
#[cfg(feature = "tool-sleep")]
pub mod sleep;
#[cfg(feature = "tool-structured-output")]
pub mod synthetic_output;
pub mod tasks;
pub mod team_tool;
pub mod todo_write;
#[cfg(feature = "tool-tool-search")]
pub mod tool_search;
#[cfg(feature = "tool-update-plan")]
pub mod update_plan;
#[cfg(feature = "tool-view-image")]
pub mod view_image;
#[cfg(feature = "tool-web-fetch")]
pub mod web_fetch;
#[cfg(feature = "tool-web-search")]
pub mod web_search;
#[cfg(any(feature = "tool-enter-worktree", feature = "tool-exit-worktree"))]
pub mod worktree;

// Re-exports for convenience.
#[cfg(feature = "tool-apply-patch")]
pub use apply_patch::ApplyPatchTool;
// HITL types are unconditional (ToolContext refers to them).
#[cfg(feature = "tool-ask-user")]
pub use ask_user::AskUserQuestionTool;
pub use ask_user::{
    Question, QuestionAnswer, QuestionOption, QuestionPrompt, QuestionPromptSender,
    QuestionResponse,
};
#[cfg(feature = "tool-bash")]
pub use bash::BashTool;
#[cfg(feature = "tool-batch-edit")]
pub use batch_edit::BatchEditTool;
#[cfg(feature = "tool-brief")]
pub use brief::BriefTool;
#[cfg(feature = "tool-browser")]
pub use browser_tool::BrowserTool;
#[cfg(feature = "tool-grep")]
pub use code_search_tool::CodeSearchTool;
#[cfg(feature = "tool-computer-use")]
pub use computer_use::ComputerUseTool;
#[cfg(feature = "tool-config")]
pub use config_tool::ConfigTool;
#[cfg(feature = "tool-coordination")]
pub use coordination::{
    ClaimWorkTool, CoordinationInboxTool, CoordinationMessageTool, CoordinationStatusTool,
    ReleaseWorkTool,
};
#[cfg(feature = "tool-cron-create")]
pub use cron::CronCreateTool;
#[cfg(feature = "tool-cron-delete")]
pub use cron::CronDeleteTool;
#[cfg(feature = "tool-cron-list")]
pub use cron::CronListTool;
#[cfg(feature = "tool-enter-plan-mode")]
pub use enter_plan_mode::EnterPlanModeTool;
#[cfg(feature = "tool-exit-plan-mode")]
pub use exit_plan_mode::ExitPlanModeTool;
#[cfg(feature = "tool-edit")]
pub use file_edit::FileEditTool;
#[cfg(feature = "tool-read")]
pub use file_read::FileReadTool;
#[cfg(feature = "tool-write")]
pub use file_write::FileWriteTool;
pub use formatter::try_format_file;
#[cfg(feature = "tool-glob")]
pub use glob_tool::GlobTool;
#[cfg(feature = "tool-create-goal")]
pub use goal_tool::CreateGoalTool;
#[cfg(feature = "tool-get-goal")]
pub use goal_tool::GetGoalTool;
#[cfg(feature = "tool-update-goal")]
pub use goal_tool::UpdateGoalTool;
#[cfg(feature = "tool-grep")]
pub use grep_tool::GrepTool;
#[cfg(feature = "tool-lsp")]
pub use lsp_tool::LspTool;
pub use mangocode_tool_runtime as runtime;
#[cfg(feature = "tool-mcp-auth")]
pub use mcp_auth_tool::McpAuthTool;
#[cfg(feature = "tool-list-mcp-resources")]
pub use mcp_resources::ListMcpResourcesTool;
#[cfg(feature = "tool-read-mcp-resource")]
pub use mcp_resources::ReadMcpResourceTool;
#[cfg(feature = "tool-notebook-edit")]
pub use notebook_edit::NotebookEditTool;
#[cfg(feature = "tool-tool-log-read")]
pub use output_reducers::ToolLogReadTool;
#[cfg(feature = "tool-powershell")]
pub use powershell::PowerShellTool;
pub use pr_watch::heartbeat_scan_watched_prs;
#[cfg(feature = "tool-pr-watch")]
pub use pr_watch::PrWatchTool;
#[cfg(feature = "tool-project-graph")]
pub use project_graph::ProjectGraphTool;
#[cfg(feature = "tool-bash")]
pub use pty_bash::PtyBashTool;
#[cfg(feature = "tool-remote-trigger")]
pub use remote_trigger::RemoteTriggerTool;
#[cfg(feature = "tool-repl")]
pub use repl_tool::ReplTool;
#[cfg(feature = "tool-deep-read")]
pub use research::DeepReadTool;
#[cfg(feature = "tool-doc-read")]
pub use research::DocReadTool;
#[cfg(feature = "tool-doc-search")]
pub use research::DocSearchTool;
#[cfg(feature = "tool-rendered-fetch")]
pub use research::RenderedFetchTool;
#[cfg(feature = "tool-send-message")]
pub use send_message::{drain_inbox, peek_inbox, SendMessageTool};
#[cfg(feature = "tool-skill")]
pub use skill_tool::SkillTool;
#[cfg(feature = "tool-sleep")]
pub use sleep::SleepTool;
#[cfg(feature = "tool-structured-output")]
pub use synthetic_output::SyntheticOutputTool;
#[cfg(feature = "tool-task-create")]
pub use tasks::TaskCreateTool;
#[cfg(feature = "tool-task-get")]
pub use tasks::TaskGetTool;
#[cfg(feature = "tool-task-list")]
pub use tasks::TaskListTool;
#[cfg(feature = "tool-task-output")]
pub use tasks::TaskOutputTool;
#[cfg(feature = "tool-task-stop")]
pub use tasks::TaskStopTool;
#[cfg(feature = "tool-task-update")]
pub use tasks::TaskUpdateTool;
pub use tasks::{Task, TaskStatus, TASK_STORE};
#[cfg(feature = "tool-team-create")]
pub use team_tool::TeamCreateTool;
#[cfg(feature = "tool-team-delete")]
pub use team_tool::TeamDeleteTool;
pub use team_tool::{register_agent_runner, AgentRunFn};
#[cfg(feature = "tool-todo-write")]
pub use todo_write::TodoWriteTool;
#[cfg(feature = "tool-tool-search")]
pub use tool_search::ToolSearchTool;
#[cfg(feature = "tool-update-plan")]
pub use update_plan::UpdatePlanTool;
#[cfg(feature = "tool-view-image")]
pub use view_image::ViewImageTool;
#[cfg(feature = "tool-web-fetch")]
pub use web_fetch::WebFetchTool;
#[cfg(feature = "tool-web-search")]
pub use web_search::WebSearchTool;
#[cfg(feature = "tool-enter-worktree")]
pub use worktree::EnterWorktreeTool;
#[cfg(feature = "tool-exit-worktree")]
pub use worktree::ExitWorktreeTool;

// ---------------------------------------------------------------------------
// Core trait & types
// ---------------------------------------------------------------------------

/// The result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// Content to send back to the model as the tool result.
    pub content: String,
    /// Whether this invocation was an error.
    pub is_error: bool,
    /// Optional structured metadata (for the TUI to render diffs, etc.).
    pub metadata: Option<Value>,
}

impl ToolResult {
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            metadata: None,
        }
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
            metadata: None,
        }
    }

    pub fn with_metadata(mut self, meta: Value) -> Self {
        self.metadata = Some(meta);
        self
    }

    pub fn to_envelope(&self) -> ToolOutputEnvelope {
        let raw_log_path = self
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("raw_log_path"))
            .and_then(Value::as_str)
            .map(str::to_string);
        ToolOutputEnvelope {
            success: !self.is_error,
            text: self.content.clone(),
            metadata: self.metadata.clone(),
            duration_ms: None,
            artifacts: Vec::new(),
            affected_paths: Vec::new(),
            raw_log_path,
            error_kind: self.is_error.then_some(ToolErrorKind::ExecutionFailed),
        }
    }
}

/// Permission level required by a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionLevel {
    /// No permission needed (read-only, purely informational).
    None,
    /// Local read-only access.
    ReadOnly,
    /// Outbound network access. Non-mutating, but not equivalent to local reads.
    Network,
    /// Write access to the filesystem.
    Write,
    /// Arbitrary command execution.
    Execute,
    /// Potentially dangerous (e.g., bypass sandbox).
    Dangerous,
    /// Unconditionally forbidden — the action must never be executed regardless
    /// of permission mode.  Used by BashTool when the classifier identifies a
    /// `Critical`-risk command (e.g. `rm -rf /`, fork-bomb, `dd if=…`).
    Forbidden,
}

/// Persistent shell state shared across Bash tool invocations within one session.
///
/// The `BashTool` reads and writes this state on every call so that `cd` and
/// `export` commands persist across separate tool invocations, matching the
/// mental model described in the tool description ("the working directory
/// persists between commands").
#[derive(Debug, Clone, Default)]
pub struct ShellState {
    /// Current working directory as tracked by the shell state.
    /// Starts as the session's `working_dir`; updated after each `cd` command.
    pub cwd: Option<PathBuf>,
    /// Environment variable overrides exported by previous commands.
    pub env_vars: HashMap<String, String>,
}

impl ShellState {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Process-global registry of shell states keyed by session_id.
/// This lets us persist cwd/env across Bash invocations without changing
/// the `ToolContext` struct (which is constructed in places we cannot modify).
static SHELL_STATE_REGISTRY: once_cell::sync::Lazy<
    dashmap::DashMap<String, Arc<parking_lot::Mutex<ShellState>>>,
> = once_cell::sync::Lazy::new(dashmap::DashMap::new);

/// Process-global registry of `SnapshotManager` instances keyed by session_id.
/// Used by tools to record pre-write snapshots and by `/undo` to revert them.
static SNAPSHOT_REGISTRY: once_cell::sync::Lazy<
    dashmap::DashMap<String, Arc<parking_lot::Mutex<mangocode_core::SnapshotManager>>>,
> = once_cell::sync::Lazy::new(dashmap::DashMap::new);

/// Return the persistent `ShellState` for the given session, creating one if needed.
pub fn session_shell_state(session_id: &str) -> Arc<parking_lot::Mutex<ShellState>> {
    SHELL_STATE_REGISTRY
        .entry(session_id.to_string())
        .or_insert_with(|| Arc::new(parking_lot::Mutex::new(ShellState::new())))
        .clone()
}

/// Remove the shell state for a session (e.g. when the session ends).
pub fn clear_session_shell_state(session_id: &str) {
    SHELL_STATE_REGISTRY.remove(session_id);
}

/// Return the persistent `SnapshotManager` for the given session, creating one if needed.
pub fn session_snapshot(
    session_id: &str,
) -> Arc<parking_lot::Mutex<mangocode_core::SnapshotManager>> {
    SNAPSHOT_REGISTRY
        .entry(session_id.to_string())
        .or_insert_with(|| {
            Arc::new(parking_lot::Mutex::new(
                mangocode_core::SnapshotManager::new(),
            ))
        })
        .clone()
}

/// Remove the snapshot manager for a session (e.g. when the session ends).
pub fn clear_session_snapshot(session_id: &str) {
    SNAPSHOT_REGISTRY.remove(session_id);
}

/// Shared context passed to every tool invocation.
#[derive(Clone)]
pub struct ToolContext {
    pub working_dir: PathBuf,
    pub permission_mode: PermissionMode,
    pub permission_handler: Arc<dyn PermissionHandler>,
    pub cost_tracker: Arc<CostTracker>,
    pub session_metrics: Option<Arc<mangocode_core::analytics::SessionMetrics>>,
    pub session_id: String,
    /// Optional local coordination actor id. This is distinct from `session_id`
    /// so sub-agents can be addressable without changing session-scoped state.
    pub coordination_actor_id: Option<String>,
    /// Optional parent coordination actor id for actor-tree display and
    /// targetability. This is separate from `session_id` because session
    /// lifecycle identity is still shared by nested runtime loops.
    pub coordination_parent_actor_id: Option<String>,
    /// Whether tools spawned from this context should allow automatic
    /// coordination inbox injection in nested query loops.
    pub inject_coordination_inbox: bool,
    pub file_history: Arc<parking_lot::Mutex<mangocode_core::file_history::FileHistory>>,
    pub current_turn: Arc<AtomicUsize>,
    /// If true, suppress interactive prompts (batch / CI mode).
    pub non_interactive: bool,
    /// Optional MCP manager for ListMcpResources / ReadMcpResource tools.
    pub mcp_manager: Option<Arc<mangocode_mcp::McpManager>>,
    /// Configured event hooks (PreToolUse, PostToolUse, etc.).
    pub config: mangocode_core::config::Config,
    /// HITL clarification channel for the AskUserQuestion tool. When `Some`,
    /// the tool posts a [`QuestionPrompt`] here and the TUI displays a
    /// dialog; when `None` (sub-agent, ACP, headless, proactive run), the
    /// tool returns an error instead.
    pub question_prompt_tx: Option<crate::ask_user::QuestionPromptSender>,
    /// Parent cancellation token. Sub-agents should derive child tokens from
    /// this so that cancelling a parent also cancels its children.
    pub cancel_token: Option<tokio_util::sync::CancellationToken>,
}

impl ToolContext {
    pub fn coordination_actor_id(&self) -> String {
        let base = self
            .coordination_actor_id
            .as_deref()
            .unwrap_or(&self.session_id);
        mangocode_core::coordination::process_session_id(base)
    }

    pub fn coordination_parent_session_id(&self) -> Option<Cow<'_, str>> {
        if let Some(parent_actor_id) = self.coordination_parent_actor_id.as_deref() {
            return Some(Cow::Borrowed(parent_actor_id));
        }
        if self.coordination_actor_id.is_some() {
            return Some(Cow::Owned(
                mangocode_core::coordination::process_session_id(&self.session_id),
            ));
        }
        None
    }

    /// Resolve a potentially relative path against the working directory.
    pub fn resolve_path(&self, path: &str) -> PathBuf {
        let p = PathBuf::from(path);
        if p.is_absolute() {
            p
        } else {
            self.working_dir.join(p)
        }
    }

    /// Check whether a path is within the allowed working directory or any
    /// configured additional directories.  Canonicalizes the path first; if
    /// the path does not exist yet (e.g. a new file for writes), the parent
    /// directory is canonicalized instead.
    pub fn is_path_allowed(&self, path: &Path) -> bool {
        let canonical = path.canonicalize().or_else(|_| {
            // File may not exist yet — check parent directory instead.
            path.parent().unwrap_or(path).canonicalize()
        });
        let canonical = match canonical {
            Ok(p) => p,
            Err(_) => return false,
        };
        let working_dir = match self.working_dir.canonicalize() {
            Ok(p) => p,
            Err(_) => return false,
        };
        if canonical.starts_with(&working_dir) {
            return true;
        }
        for dir in &self.config.additional_dirs {
            if let Ok(allowed) = dir.canonicalize() {
                if canonical.starts_with(&allowed) {
                    return true;
                }
            }
        }
        false
    }

    /// Resolve a path and validate it is within allowed directories.
    pub fn resolve_and_validate_path(&self, path: &str) -> Result<PathBuf, String> {
        let resolved = self.resolve_path(path);
        if !self.is_path_allowed(&resolved) {
            return Err(format!(
                "Path {} is outside the allowed working directory",
                resolved.display()
            ));
        }
        Ok(resolved)
    }

    /// Check permissions for a tool invocation.
    pub fn check_permission(
        &self,
        tool_name: &str,
        description: &str,
        is_read_only: bool,
    ) -> Result<(), mangocode_core::error::ClaudeError> {
        let request = PermissionRequest {
            tool_name: tool_name.to_string(),
            description: description.to_string(),
            details: None,
            is_read_only,
            context_description: None,
        };
        mangocode_core::harness::record_permission_request(&self.session_id, &request);
        let decision = self.permission_handler.request_permission(&request);
        mangocode_core::harness::record_permission_decision(&self.session_id, &request, &decision);
        match decision {
            PermissionDecision::Allow | PermissionDecision::AllowPermanently => Ok(()),
            _ => Err(mangocode_core::error::ClaudeError::PermissionDenied(
                format!("Permission denied for tool '{}'", tool_name),
            )),
        }
    }

    /// Like `check_permission` but also passes structured `details` text
    /// (e.g. a risk explanation) that the TUI permission dialog can display.
    ///
    /// Used by PowerShellTool (and any future tool) when it needs to show
    /// the user *why* a command is considered risky before they approve it.
    pub fn check_permission_with_details(
        &self,
        tool_name: &str,
        description: &str,
        details: &str,
        is_read_only: bool,
    ) -> Result<(), mangocode_core::error::ClaudeError> {
        let request = PermissionRequest {
            tool_name: tool_name.to_string(),
            description: description.to_string(),
            details: Some(details.to_string()),
            is_read_only,
            context_description: None,
        };
        mangocode_core::harness::record_permission_request(&self.session_id, &request);
        let decision = self.permission_handler.request_permission(&request);
        mangocode_core::harness::record_permission_decision(&self.session_id, &request, &decision);
        match decision {
            PermissionDecision::Allow | PermissionDecision::AllowPermanently => Ok(()),
            _ => Err(mangocode_core::error::ClaudeError::PermissionDenied(
                format!("Permission denied for tool '{}': {}", tool_name, details),
            )),
        }
    }

    pub fn current_turn_index(&self) -> usize {
        self.current_turn.load(Ordering::Relaxed)
    }

    pub fn record_file_change(
        &self,
        path: PathBuf,
        before_content: &[u8],
        after_content: &[u8],
        tool_name: &str,
    ) {
        self.record_file_change_with_existence(
            path,
            before_content,
            after_content,
            (true, true),
            tool_name,
        );
    }

    pub fn record_file_change_with_existence(
        &self,
        path: PathBuf,
        before_content: &[u8],
        after_content: &[u8],
        existence: (bool, bool),
        tool_name: &str,
    ) {
        notify_plugins_file_changed(&path, tool_name, &self.session_id);
        self.file_history.lock().record_modification_with_existence(
            path,
            before_content,
            after_content,
            existence,
            self.current_turn_index(),
            tool_name,
        );
    }
}

/// Fire plugin `FileChanged` lifecycle hooks for a recorded file modification.
/// Detached and best-effort: hook processes run on a separate thread so tool
/// execution is never blocked.
fn notify_plugins_file_changed(path: &std::path::Path, tool_name: &str, session_id: &str) {
    if !mangocode_plugins::has_global_hooks_for_event(mangocode_plugins::HookEventKind::FileChanged)
    {
        return;
    }
    let payload = serde_json::json!({
        "event": "FileChanged",
        "tool_name": tool_name,
        "tool_input": null,
        "tool_output": path.display().to_string(),
        "path": path.display().to_string(),
        "is_error": null,
        "session_id": session_id,
    });
    let _ = file_changed_sender().send(payload);
}

/// Lazily-initialized single worker thread that runs FileChanged lifecycle
/// hooks sequentially. Routing every notification through one channel bounds
/// the number of OS threads to one, so a batch of edits (or a hung hook)
/// cannot spawn an unbounded number of detached threads.
fn file_changed_sender() -> &'static std::sync::mpsc::Sender<serde_json::Value> {
    static TX: std::sync::OnceLock<std::sync::mpsc::Sender<serde_json::Value>> =
        std::sync::OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<serde_json::Value>();
        std::thread::spawn(move || {
            while let Ok(payload) = rx.recv() {
                mangocode_plugins::run_global_lifecycle_hook(
                    mangocode_plugins::HookEventKind::FileChanged,
                    payload,
                );
            }
        });
        tx
    })
}

/// The trait every tool must implement.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Human-readable name (matches the constant in mangocode_core::constants).
    fn name(&self) -> &str;

    /// One-line description shown to the LLM.
    fn description(&self) -> &str;

    /// The permission level the tool requires.
    fn permission_level(&self) -> PermissionLevel;

    /// JSON Schema describing the tool's input parameters.
    fn input_schema(&self) -> Value;

    /// Execute the tool with the given JSON input.
    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult;

    /// Compatibility aliases accepted by the runtime resolver.
    fn aliases(&self) -> Vec<String> {
        default_aliases_for_tool(self.name())
    }

    /// Capability metadata for a concrete invocation.
    fn capabilities(&self, input: &Value) -> ToolCapabilities {
        default_capabilities_for_tool(self.name(), self.permission_level(), input)
    }

    /// Produce a `ToolDefinition` suitable for sending to the API.
    fn to_definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.input_schema(),
        }
    }

    /// Produce a runtime spec for registry planning, discovery, and scheduling.
    fn to_runtime_spec(&self) -> ToolSpec {
        let aliases = self.aliases();
        let mut capabilities = self.capabilities(&Value::Null);
        capabilities.aliases = aliases.clone();
        ToolSpec {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.input_schema(),
            handler_kind: ToolHandlerKind::BuiltIn,
            aliases,
            capabilities,
        }
    }
}

/// Return all built-in tools (excluding AgentTool, which lives in mangocode-query).
pub fn all_tools() -> Vec<Box<dyn Tool>> {
    vec![
        #[cfg(feature = "tool-bash")]
        Box::new(PtyBashTool),
        #[cfg(feature = "tool-read")]
        Box::new(FileReadTool),
        #[cfg(feature = "tool-edit")]
        Box::new(FileEditTool),
        #[cfg(feature = "tool-write")]
        Box::new(FileWriteTool),
        #[cfg(feature = "tool-view-image")]
        Box::new(ViewImageTool),
        #[cfg(feature = "tool-batch-edit")]
        Box::new(BatchEditTool),
        #[cfg(feature = "tool-apply-patch")]
        Box::new(ApplyPatchTool),
        #[cfg(feature = "tool-get-goal")]
        Box::new(GetGoalTool),
        #[cfg(feature = "tool-create-goal")]
        Box::new(CreateGoalTool),
        #[cfg(feature = "tool-update-goal")]
        Box::new(UpdateGoalTool),
        #[cfg(feature = "tool-glob")]
        Box::new(GlobTool),
        #[cfg(feature = "tool-grep")]
        Box::new(GrepTool),
        #[cfg(feature = "tool-grep")]
        Box::new(CodeSearchTool),
        #[cfg(feature = "tool-web-fetch")]
        Box::new(WebFetchTool),
        #[cfg(feature = "tool-web-search")]
        Box::new(WebSearchTool),
        #[cfg(feature = "tool-doc-search")]
        Box::new(DocSearchTool),
        #[cfg(feature = "tool-doc-read")]
        Box::new(DocReadTool),
        #[cfg(feature = "tool-deep-read")]
        Box::new(DeepReadTool),
        #[cfg(feature = "tool-rendered-fetch")]
        Box::new(RenderedFetchTool),
        #[cfg(feature = "tool-tool-log-read")]
        Box::new(ToolLogReadTool),
        #[cfg(feature = "tool-notebook-edit")]
        Box::new(NotebookEditTool),
        #[cfg(feature = "tool-task-create")]
        Box::new(TaskCreateTool),
        #[cfg(feature = "tool-task-get")]
        Box::new(TaskGetTool),
        #[cfg(feature = "tool-task-update")]
        Box::new(TaskUpdateTool),
        #[cfg(feature = "tool-task-list")]
        Box::new(TaskListTool),
        #[cfg(feature = "tool-task-stop")]
        Box::new(TaskStopTool),
        #[cfg(feature = "tool-task-output")]
        Box::new(TaskOutputTool),
        #[cfg(feature = "tool-todo-write")]
        Box::new(TodoWriteTool),
        #[cfg(feature = "tool-update-plan")]
        Box::new(UpdatePlanTool),
        #[cfg(feature = "tool-ask-user")]
        Box::new(AskUserQuestionTool),
        #[cfg(feature = "tool-enter-plan-mode")]
        Box::new(EnterPlanModeTool),
        #[cfg(feature = "tool-exit-plan-mode")]
        Box::new(ExitPlanModeTool),
        #[cfg(feature = "tool-powershell")]
        Box::new(PowerShellTool),
        #[cfg(feature = "tool-sleep")]
        Box::new(SleepTool),
        #[cfg(feature = "tool-pr-watch")]
        Box::new(PrWatchTool),
        #[cfg(feature = "tool-cron-create")]
        Box::new(CronCreateTool),
        #[cfg(feature = "tool-cron-delete")]
        Box::new(CronDeleteTool),
        #[cfg(feature = "tool-cron-list")]
        Box::new(CronListTool),
        #[cfg(feature = "tool-enter-worktree")]
        Box::new(EnterWorktreeTool),
        #[cfg(feature = "tool-exit-worktree")]
        Box::new(ExitWorktreeTool),
        #[cfg(feature = "tool-list-mcp-resources")]
        Box::new(ListMcpResourcesTool),
        #[cfg(feature = "tool-read-mcp-resource")]
        Box::new(ReadMcpResourceTool),
        #[cfg(feature = "tool-tool-search")]
        Box::new(ToolSearchTool),
        #[cfg(feature = "tool-project-graph")]
        Box::new(ProjectGraphTool),
        #[cfg(feature = "tool-brief")]
        Box::new(BriefTool),
        #[cfg(feature = "tool-config")]
        Box::new(ConfigTool),
        #[cfg(feature = "tool-coordination")]
        Box::new(CoordinationStatusTool),
        #[cfg(feature = "tool-coordination")]
        Box::new(CoordinationInboxTool),
        #[cfg(feature = "tool-coordination")]
        Box::new(ClaimWorkTool),
        #[cfg(feature = "tool-coordination")]
        Box::new(ReleaseWorkTool),
        #[cfg(feature = "tool-coordination")]
        Box::new(CoordinationMessageTool),
        #[cfg(feature = "tool-send-message")]
        Box::new(SendMessageTool),
        #[cfg(feature = "tool-skill")]
        Box::new(SkillTool),
        #[cfg(feature = "tool-lsp")]
        Box::new(LspTool),
        #[cfg(feature = "tool-repl")]
        Box::new(ReplTool),
        #[cfg(feature = "tool-team-create")]
        Box::new(TeamCreateTool),
        #[cfg(feature = "tool-team-delete")]
        Box::new(TeamDeleteTool),
        #[cfg(feature = "tool-structured-output")]
        Box::new(SyntheticOutputTool),
        #[cfg(feature = "tool-mcp-auth")]
        Box::new(McpAuthTool),
        #[cfg(feature = "tool-remote-trigger")]
        Box::new(RemoteTriggerTool),
        // Computer Use is only available when compiled with the feature flag.
        #[cfg(feature = "tool-computer-use")]
        Box::new(computer_use::ComputerUseTool),
        // Browser automation is only available when compiled with the feature flag.
        #[cfg(feature = "tool-browser")]
        Box::new(browser_tool::BrowserTool),
    ]
}

/// Find a tool by name (case-sensitive).
pub fn find_tool(name: &str) -> Option<Box<dyn Tool>> {
    all_tools().into_iter().find(|t| t.name() == name)
}

pub fn build_registry_plan(tools: &[Box<dyn Tool>]) -> ToolRegistryPlan {
    ToolRegistryPlan::from_specs(tools.iter().map(|tool| tool.to_runtime_spec()).collect())
}

pub struct McpToolWrapper {
    tool_def: ToolDefinition,
    server_name: String,
    manager_tool_name: String,
    manager: Arc<mangocode_mcp::McpManager>,
}

impl McpToolWrapper {
    fn from_manager_tool(
        server_name: String,
        manager_tool_def: ToolDefinition,
        manager: Arc<mangocode_mcp::McpManager>,
    ) -> Self {
        let manager_tool_name = manager_tool_def.name.clone();
        let (compat_tool_name, _) = mcp_compat_names(&server_name, &manager_tool_name);
        Self {
            tool_def: ToolDefinition {
                name: compat_tool_name,
                description: manager_tool_def.description,
                input_schema: manager_tool_def.input_schema,
            },
            server_name,
            manager_tool_name,
            manager,
        }
    }
}

#[async_trait]
impl Tool for McpToolWrapper {
    fn name(&self) -> &str {
        &self.tool_def.name
    }

    fn description(&self) -> &str {
        &self.tool_def.description
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Execute
    }

    fn input_schema(&self) -> Value {
        self.tool_def.input_schema.clone()
    }

    fn aliases(&self) -> Vec<String> {
        let (_, aliases) = mcp_compat_names(&self.server_name, &self.manager_tool_name);
        aliases
    }

    fn capabilities(&self, input: &Value) -> ToolCapabilities {
        let mut capabilities =
            default_capabilities_for_tool(self.name(), self.permission_level(), input);
        capabilities.aliases = self.aliases();
        capabilities
    }

    fn to_runtime_spec(&self) -> ToolSpec {
        let aliases = self.aliases();
        let mut capabilities = self.capabilities(&Value::Null);
        capabilities.aliases = aliases.clone();
        ToolSpec {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.input_schema(),
            handler_kind: ToolHandlerKind::Mcp,
            aliases,
            capabilities,
        }
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let prefix = format!("mcp__{}__", self.server_name);
        let bare_name = self
            .tool_def
            .name
            .strip_prefix(&prefix)
            .unwrap_or(&self.tool_def.name);

        if let Err(e) = ctx.check_permission(
            self.name(),
            &format!("Call MCP tool {} on {}", bare_name, self.server_name),
            false,
        ) {
            return ToolResult::error(e.to_string());
        }

        let args = if input.is_null() { None } else { Some(input) };

        match self.manager.call_tool(&self.manager_tool_name, args).await {
            Ok(result) => {
                let text = mangocode_core::system_prompt::wrap_untrusted_content(
                    "mcp_tool_result",
                    mangocode_mcp::mcp_result_to_string(&result),
                );
                if result.is_error {
                    ToolResult::error(text)
                } else {
                    ToolResult::success(text)
                }
            }
            Err(e) => ToolResult::error(format!("MCP tool '{}' failed: {}", bare_name, e)),
        }
    }
}

pub fn mcp_tool_wrappers(manager: Arc<mangocode_mcp::McpManager>) -> Vec<Box<dyn Tool>> {
    manager
        .all_tool_definitions()
        .into_iter()
        .map(|(server_name, tool_def)| {
            Box::new(McpToolWrapper::from_manager_tool(
                server_name,
                tool_def,
                manager.clone(),
            )) as Box<dyn Tool>
        })
        .collect()
}

pub fn extend_with_mcp_tools(
    tools: &mut Vec<Box<dyn Tool>>,
    manager: Arc<mangocode_mcp::McpManager>,
) {
    tools.extend(mcp_tool_wrappers(manager));
}

pub fn filter_tools_by_name_config(
    mut tools: Vec<Box<dyn Tool>>,
    allowed_tools: &[String],
    disallowed_tools: &[String],
) -> Vec<Box<dyn Tool>> {
    if !allowed_tools.is_empty() {
        tools.retain(|tool| {
            allowed_tools
                .iter()
                .any(|name| tool_name_matches(tool.as_ref(), name))
        });
    }

    if !disallowed_tools.is_empty() {
        tools.retain(|tool| {
            !disallowed_tools
                .iter()
                .any(|name| tool_name_matches(tool.as_ref(), name))
        });
    }

    tools
}

pub fn resolve_tool<'a>(tools: &'a [Box<dyn Tool>], requested: &str) -> Option<&'a dyn Tool> {
    if let Some(tool) = tools.iter().find(|tool| tool.name() == requested) {
        return Some(tool.as_ref());
    }
    tools
        .iter()
        .find(|tool| tool_name_matches(tool.as_ref(), requested))
        .map(|tool| tool.as_ref())
}

pub fn available_tool_names(tools: &[Box<dyn Tool>]) -> Vec<String> {
    tools.iter().map(|tool| tool.name().to_string()).collect()
}

pub fn restrict_config_to_available_tools(
    config: &mut mangocode_core::config::Config,
    tools: &[Box<dyn Tool>],
) {
    config.allowed_tools = available_tool_names(tools);
    config.disallowed_tools.clear();
}

pub fn sync_tool_context_to_available_tools(ctx: &mut ToolContext, tools: &[Box<dyn Tool>]) {
    restrict_config_to_available_tools(&mut ctx.config, tools);
}

pub fn tool_name_matches(tool: &dyn Tool, requested: &str) -> bool {
    let requested = normalize_tool_name(requested);
    normalize_tool_name(tool.name()) == requested
        || tool
            .aliases()
            .iter()
            .any(|alias| normalize_tool_name(alias) == requested)
}

pub fn tool_supports_parallel(tools: &[Box<dyn Tool>], requested: &str, input: &Value) -> bool {
    resolve_tool(tools, requested)
        .map(|tool| {
            let capabilities = tool.capabilities(input);
            !capabilities.mutating && capabilities.parallel_safe
        })
        .unwrap_or(false)
}

pub fn unknown_tool_message(tools: &[Box<dyn Tool>], requested: &str) -> String {
    let plan = build_registry_plan(tools);
    let suggestions = plan.suggestions_for(requested, 5);
    if suggestions.is_empty() {
        format!(
            "Unknown tool: {requested}. Use ToolSearch to discover runtime-visible tools, or check whether the relevant plugin, MCP server, feature flag, and session visibility config permit it."
        )
    } else {
        format!(
            "Unknown tool: {requested}. Did you mean {}? You can also use ToolSearch for runtime-visible tool discovery. If the expected tool is missing, check plugin, MCP, feature flag, and session visibility config.",
            suggestions.join(", ")
        )
    }
}

pub fn default_aliases_for_tool(name: &str) -> Vec<String> {
    let aliases = match name {
        "Bash" => vec![
            "shell",
            "shell_command",
            "container.exec",
            "local_shell",
            "bash",
            "exec",
        ],
        "PowerShell" => vec!["powershell", "pwsh", "ps"],
        "ApplyPatch" => vec!["apply_patch"],
        "Read" => vec!["read_file", "file_read"],
        "Write" => vec!["write_file", "file_write"],
        "Edit" => vec!["edit_file", "file_edit"],
        "BatchEdit" => vec!["batch_edit"],
        "Glob" => vec!["glob"],
        "Grep" => vec!["grep"],
        "TodoWrite" => vec!["todo_write"],
        "get_goal" => vec!["GetGoal"],
        "create_goal" => vec!["CreateGoal"],
        "update_goal" => vec!["UpdateGoal"],
        "ToolSearch" => vec!["tool_search"],
        "ViewImage" => vec!["view_image"],
        "Browser" => vec!["browser"],
        "computer" => vec!["computer_use"],
        "ListMcpResources" => vec!["list_mcp_resources"],
        "ReadMcpResource" => vec!["read_mcp_resource"],
        "mcp__auth" => vec!["mcp_auth"],
        "LSP" => vec!["lsp"],
        "REPL" => vec!["repl", "node_repl"],
        "Agent" => vec!["agent", "sub_agent", "subagent", "spawn_agent"],
        _ => Vec::new(),
    };
    let mut aliases = aliases.into_iter().map(str::to_string).collect::<Vec<_>>();
    if let Some(alias) = camel_case_tool_alias(name) {
        aliases.push(alias);
    }
    mangocode_tool_runtime::dedupe_strings(aliases)
}

fn camel_case_tool_alias(name: &str) -> Option<String> {
    let name = name.trim();
    if name.is_empty() || name.contains('_') || name.chars().all(|ch| !ch.is_ascii_lowercase()) {
        return None;
    }

    let mut out = String::with_capacity(name.len() + 4);
    let mut prev_was_lower_or_digit = false;
    let mut wrote_separator = false;
    for ch in name.chars() {
        if ch.is_ascii_uppercase() {
            if prev_was_lower_or_digit {
                out.push('_');
                wrote_separator = true;
            }
            out.push(ch.to_ascii_lowercase());
            prev_was_lower_or_digit = false;
        } else if ch == '-' || ch == '.' || ch.is_whitespace() {
            if !out.ends_with('_') && !out.is_empty() {
                out.push('_');
                wrote_separator = true;
            }
            prev_was_lower_or_digit = false;
        } else {
            out.push(ch.to_ascii_lowercase());
            prev_was_lower_or_digit = ch.is_ascii_lowercase() || ch.is_ascii_digit();
        }
    }

    let alias = out.trim_matches('_').to_string();
    (wrote_separator && alias != name.to_ascii_lowercase()).then_some(alias)
}

pub fn mcp_compat_names(server_name: &str, manager_tool_name: &str) -> (String, Vec<String>) {
    let internal_prefix = format!("{server_name}_");
    let bare_tool_name = manager_tool_name
        .strip_prefix(&internal_prefix)
        .unwrap_or(manager_tool_name);
    let compat_name = format!("mcp__{server_name}__{bare_tool_name}");
    let aliases = mangocode_tool_runtime::dedupe_strings([
        manager_tool_name.to_string(),
        bare_tool_name.to_string(),
    ]);
    (compat_name, aliases)
}

pub fn default_capabilities_for_tool(
    name: &str,
    permission_level: PermissionLevel,
    input: &Value,
) -> ToolCapabilities {
    let apply_patch_dry_run = name == "ApplyPatch"
        && input
            .get("dry_run")
            .and_then(Value::as_bool)
            .unwrap_or(false);
    let mut capabilities = match permission_level {
        PermissionLevel::None | PermissionLevel::ReadOnly | PermissionLevel::Network => {
            ToolCapabilities::read_only()
        }
        PermissionLevel::Write
        | PermissionLevel::Execute
        | PermissionLevel::Dangerous
        | PermissionLevel::Forbidden => ToolCapabilities::mutating(),
    };

    capabilities.aliases = default_aliases_for_tool(name);
    capabilities.affected_paths = extract_path_values(input);
    if name == "ApplyPatch" {
        capabilities
            .affected_paths
            .extend(extract_apply_patch_paths(input));
        capabilities.affected_paths =
            mangocode_tool_runtime::dedupe_strings(capabilities.affected_paths.clone());
    }
    if apply_patch_dry_run {
        capabilities.mutating = false;
        capabilities.parallel_safe = true;
        capabilities.sandbox_preference = SandboxPreference::ReadOnly;
    }
    capabilities.network_targets = extract_network_targets(input);
    capabilities.supports_cancellation = matches!(
        name,
        "Bash"
            | "PowerShell"
            | "WebFetch"
            | "WebSearch"
            | "TaskCreate"
            | "TaskStop"
            | "Browser"
            | "computer"
            | "REPL"
    );

    if matches!(
        permission_level,
        PermissionLevel::Execute | PermissionLevel::Dangerous
    ) {
        capabilities.sandbox_preference = SandboxPreference::FullAccess;
    }

    if is_stateful_or_interactive_tool(name) {
        capabilities.mutating = true;
        capabilities.parallel_safe = false;
        if capabilities.sandbox_preference == SandboxPreference::ReadOnly {
            capabilities.sandbox_preference = SandboxPreference::WorkspaceWrite;
        }
    }

    let mut approval_keys = Vec::new();
    if capabilities.mutating || !capabilities.affected_paths.is_empty() {
        for path in &capabilities.affected_paths {
            approval_keys.push(ApprovalKey::new("path", path.clone()));
        }
    }
    for target in &capabilities.network_targets {
        approval_keys.push(ApprovalKey::new("host", target.clone()));
    }
    if let Some(command_prefix) = command_prefix(input) {
        approval_keys.push(ApprovalKey::new("command_prefix", command_prefix));
    }
    if capabilities.mutating || permission_level != PermissionLevel::None {
        approval_keys.push(ApprovalKey::new("tool", name.to_string()));
    }
    capabilities.approval_keys = approval_keys;
    capabilities
}

fn is_stateful_or_interactive_tool(name: &str) -> bool {
    matches!(
        name,
        "AskUserQuestion"
            | "EnterPlanMode"
            | "ExitPlanMode"
            | "update_plan"
            | "TodoWrite"
            | "Sleep"
            | "CronCreate"
            | "CronDelete"
            | "EnterWorktree"
            | "ExitWorktree"
            | "TaskCreate"
            | "TaskUpdate"
            | "TaskStop"
            | "Config"
            | "SendMessage"
            | "CoordinationStatus"
            | "CoordinationInbox"
            | "ClaimWork"
            | "ReleaseWork"
            | "CoordinationMessage"
            | "Skill"
            | "Agent"
            | "REPL"
            | "TeamCreate"
            | "TeamDelete"
            | "StructuredOutput"
            | "mcp__auth"
            | "RemoteTrigger"
            | "Browser"
            | "computer"
    )
}

fn extract_path_values(input: &Value) -> Vec<String> {
    let mut paths = Vec::new();
    collect_string_values_for_keys(
        input,
        &[
            "file_path",
            "filepath",
            "notebook_path",
            "path",
            "paths",
            "old_path",
            "new_path",
            "target_path",
            "source_path",
            "source_dir",
            "output_path",
            "output_file",
            "output_dir",
            "out_dir",
            "destination",
            "dest",
            "graph_path",
            "memory_dir",
            "report_path",
            "manifest_path",
            "html_path",
            "global_dir",
            "worktree_path",
            "team_file_path",
            "global_graph_path",
            "global_manifest_path",
        ],
        &mut paths,
    );
    mangocode_tool_runtime::dedupe_strings(
        paths
            .into_iter()
            .filter(|value| !looks_like_network_target(value)),
    )
}

fn extract_apply_patch_paths(input: &Value) -> Vec<String> {
    let Some(patch) = input.get("patch").and_then(Value::as_str) else {
        return Vec::new();
    };
    mangocode_tool_runtime::dedupe_strings(
        patch
            .lines()
            .filter_map(diff_path_from_header)
            .filter(|path| !looks_like_network_target(path))
            .collect::<Vec<_>>(),
    )
}

fn diff_path_from_header(line: &str) -> Option<String> {
    let raw = line
        .strip_prefix("--- ")
        .or_else(|| line.strip_prefix("+++ "))?;
    let token = raw
        .trim()
        .split_once('\t')
        .map(|(path, _)| path)
        .unwrap_or_else(|| raw.trim())
        .trim();
    let token = unquote_diff_path_token(token);
    if token.is_empty() || token == "/dev/null" {
        return None;
    }
    Some(
        token
            .strip_prefix("a/")
            .or_else(|| token.strip_prefix("b/"))
            .unwrap_or(&token)
            .to_string(),
    )
}

fn unquote_diff_path_token(path: &str) -> String {
    let Some(stripped) = path.strip_prefix('"').and_then(|p| p.strip_suffix('"')) else {
        return path.to_string();
    };

    let mut out = String::with_capacity(stripped.len());
    let mut chars = stripped.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some('t') => out.push('\t'),
                Some('n') => out.push('\n'),
                Some(next) => {
                    out.push('\\');
                    out.push(next);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(ch);
        }
    }
    out
}

fn extract_network_targets(input: &Value) -> Vec<String> {
    let mut values = Vec::new();
    collect_network_target_values(input, &mut values);
    mangocode_tool_runtime::dedupe_strings(
        values
            .into_iter()
            .map(|value| network_approval_target(&value)),
    )
}

fn collect_network_target_values(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, value) in map {
                let is_host_key = key.eq_ignore_ascii_case("host");
                let is_url_key = ["url", "urls", "uri"]
                    .iter()
                    .any(|wanted| key.eq_ignore_ascii_case(wanted));
                match value {
                    Value::String(text) if is_host_key => {
                        if !text.trim().is_empty() {
                            out.push(text.clone());
                        }
                    }
                    Value::String(text) if is_url_key && looks_like_network_target(text) => {
                        out.push(text.clone());
                    }
                    Value::Array(items) if is_host_key || is_url_key => {
                        for item in items {
                            match item {
                                Value::String(text)
                                    if is_host_key || looks_like_network_target(text) =>
                                {
                                    if !text.trim().is_empty() {
                                        out.push(text.clone());
                                    }
                                }
                                _ => collect_network_target_values(item, out),
                            }
                        }
                    }
                    _ => collect_network_target_values(value, out),
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_network_target_values(item, out);
            }
        }
        _ => {}
    }
}

fn collect_string_values_for_keys(value: &Value, keys: &[&str], out: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, value) in map {
                let key_matches = keys.iter().any(|wanted| key.eq_ignore_ascii_case(wanted));
                match value {
                    Value::String(text) if key_matches => out.push(text.clone()),
                    Value::Array(items) if key_matches => {
                        for item in items {
                            if let Value::String(text) = item {
                                out.push(text.clone());
                            } else {
                                collect_string_values_for_keys(item, keys, out);
                            }
                        }
                    }
                    _ => collect_string_values_for_keys(value, keys, out),
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_string_values_for_keys(item, keys, out);
            }
        }
        _ => {}
    }
}

fn command_prefix(input: &Value) -> Option<String> {
    let command = input.get("command").and_then(Value::as_str)?;
    let words = match mangocode_core::split_command_words(command) {
        Ok(words) => words,
        Err(err) => {
            tracing::warn!(
                error = %err,
                "failed to parse command for approval prefix; using malformed-command key"
            );
            return malformed_command_prefix(command);
        }
    };
    let prefix = words
        .into_iter()
        .take(2)
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string();
    if prefix.is_empty() {
        None
    } else {
        Some(prefix)
    }
}

fn malformed_command_prefix(command: &str) -> Option<String> {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut chars = trimmed.chars();
    let mut value: String = chars.by_ref().take(120).collect();
    if chars.next().is_some() {
        value.push_str("...");
    }
    Some(format!("invalid command syntax: {value}"))
}

fn looks_like_network_target(value: &str) -> bool {
    let value = value.trim().to_ascii_lowercase();
    value.starts_with("http://")
        || value.starts_with("https://")
        || value.starts_with("ws://")
        || value.starts_with("wss://")
}

fn network_approval_target(value: &str) -> String {
    let trimmed = value.trim();
    if let Ok(parsed) = reqwest::Url::parse(trimmed) {
        if let Some(host) = parsed.host_str() {
            let mut authority = if host.contains(':') && !host.starts_with('[') {
                format!("[{}]", host.to_ascii_lowercase())
            } else {
                host.to_ascii_lowercase()
            };
            if let Some(port) = parsed.port() {
                authority.push(':');
                authority.push_str(&port.to_string());
            }
            return format!("{}://{}", parsed.scheme().to_ascii_lowercase(), authority);
        }
    }

    let Some((scheme, rest)) = trimmed.split_once("://") else {
        let authority = trimmed
            .split(['/', '?', '#'])
            .next()
            .unwrap_or(trimmed)
            .trim_end_matches('/');
        let authority = authority.rsplit('@').next().unwrap_or(authority);
        return authority.trim_end_matches('/').to_ascii_lowercase();
    };
    let authority = rest
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(rest)
        .trim_end_matches('/');
    let authority = authority.rsplit('@').next().unwrap_or(authority);
    format!(
        "{}://{}",
        scheme.to_ascii_lowercase(),
        authority.to_ascii_lowercase()
    )
}

fn normalize_tool_name(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace(['-', '.'], "_")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Tool registry tests ------------------------------------------------

    fn any_builtin_tool_feature_enabled() -> bool {
        cfg!(feature = "tool-apply-patch")
            || cfg!(feature = "tool-ask-user")
            || cfg!(feature = "tool-bash")
            || cfg!(feature = "tool-batch-edit")
            || cfg!(feature = "tool-brief")
            || cfg!(feature = "tool-browser")
            || cfg!(feature = "tool-computer-use")
            || cfg!(feature = "tool-config")
            || cfg!(feature = "tool-cron-create")
            || cfg!(feature = "tool-cron-delete")
            || cfg!(feature = "tool-cron-list")
            || cfg!(feature = "tool-create-goal")
            || cfg!(feature = "tool-deep-read")
            || cfg!(feature = "tool-doc-read")
            || cfg!(feature = "tool-doc-search")
            || cfg!(feature = "tool-edit")
            || cfg!(feature = "tool-enter-plan-mode")
            || cfg!(feature = "tool-enter-worktree")
            || cfg!(feature = "tool-exit-plan-mode")
            || cfg!(feature = "tool-exit-worktree")
            || cfg!(feature = "tool-get-goal")
            || cfg!(feature = "tool-glob")
            || cfg!(feature = "tool-grep")
            || cfg!(feature = "tool-list-mcp-resources")
            || cfg!(feature = "tool-lsp")
            || cfg!(feature = "tool-mcp-auth")
            || cfg!(feature = "tool-notebook-edit")
            || cfg!(feature = "tool-powershell")
            || cfg!(feature = "tool-pr-watch")
            || cfg!(feature = "tool-read")
            || cfg!(feature = "tool-read-mcp-resource")
            || cfg!(feature = "tool-remote-trigger")
            || cfg!(feature = "tool-rendered-fetch")
            || cfg!(feature = "tool-repl")
            || cfg!(feature = "tool-send-message")
            || cfg!(feature = "tool-skill")
            || cfg!(feature = "tool-sleep")
            || cfg!(feature = "tool-structured-output")
            || cfg!(feature = "tool-task-create")
            || cfg!(feature = "tool-task-get")
            || cfg!(feature = "tool-task-list")
            || cfg!(feature = "tool-task-output")
            || cfg!(feature = "tool-task-stop")
            || cfg!(feature = "tool-task-update")
            || cfg!(feature = "tool-team-create")
            || cfg!(feature = "tool-team-delete")
            || cfg!(feature = "tool-todo-write")
            || cfg!(feature = "tool-tool-log-read")
            || cfg!(feature = "tool-tool-search")
            || cfg!(feature = "tool-project-graph")
            || cfg!(feature = "tool-update-goal")
            || cfg!(feature = "tool-view-image")
            || cfg!(feature = "tool-web-fetch")
            || cfg!(feature = "tool-web-search")
            || cfg!(feature = "tool-write")
    }

    #[test]
    fn test_all_tools_non_empty() {
        let tools = all_tools();
        if any_builtin_tool_feature_enabled() {
            assert!(
                !tools.is_empty(),
                "all_tools() must return at least one tool when tool features are enabled"
            );
        } else {
            assert!(
                tools.is_empty(),
                "all_tools() should be empty when no built-in tool features are enabled"
            );
        }
    }

    #[test]
    fn test_all_tools_have_unique_names() {
        let tools = all_tools();
        let mut names = std::collections::HashSet::new();
        for tool in &tools {
            assert!(
                names.insert(tool.name().to_string()),
                "Duplicate tool name: {}",
                tool.name()
            );
        }
    }

    #[test]
    fn test_all_tools_have_non_empty_descriptions() {
        for tool in all_tools() {
            assert!(
                !tool.description().is_empty(),
                "Tool '{}' has empty description",
                tool.name()
            );
        }
    }

    #[test]
    fn test_all_tools_have_valid_input_schema() {
        for tool in all_tools() {
            let schema = tool.input_schema();
            assert!(
                schema.is_object(),
                "Tool '{}' input_schema must be a JSON object",
                tool.name()
            );
            assert!(
                schema.get("type").is_some() || schema.get("properties").is_some(),
                "Tool '{}' schema missing type or properties",
                tool.name()
            );
        }
    }

    #[test]
    fn runtime_specs_preserve_tool_metadata_contract() {
        for tool in all_tools() {
            let spec = tool.to_runtime_spec();
            assert_eq!(spec.name, tool.name(), "runtime spec name drifted");
            assert_eq!(
                spec.description,
                tool.description(),
                "runtime spec description drifted for {}",
                tool.name()
            );
            assert_eq!(
                spec.handler_kind,
                ToolHandlerKind::BuiltIn,
                "built-in tool {} must advertise built-in handler kind",
                tool.name()
            );
            assert_eq!(
                spec.aliases,
                spec.capabilities.aliases,
                "runtime aliases and capability aliases must match for {}",
                tool.name()
            );
            if spec.capabilities.mutating {
                assert!(
                    !spec.capabilities.parallel_safe,
                    "mutating tool {} must not advertise parallel safety",
                    tool.name()
                );
            }
            if spec.capabilities.mutating || tool.permission_level() != PermissionLevel::None {
                assert!(
                    spec.capabilities
                        .approval_keys
                        .iter()
                        .any(|key| key.kind == "tool" && key.value == spec.name),
                    "tool {} must include a tool approval key when approval metadata is required",
                    tool.name()
                );
            }
            if !spec.capabilities.affected_paths.is_empty() {
                for path in &spec.capabilities.affected_paths {
                    assert!(
                        spec.capabilities
                            .approval_keys
                            .iter()
                            .any(|key| key.kind == "path" && key.value == *path),
                        "tool {} affected path {} missing approval key",
                        tool.name(),
                        path
                    );
                }
            }
            if !spec.capabilities.network_targets.is_empty() {
                for target in &spec.capabilities.network_targets {
                    assert!(
                        spec.capabilities
                            .approval_keys
                            .iter()
                            .any(|key| key.kind == "host" && key.value == *target),
                        "tool {} network target {} missing approval key",
                        tool.name(),
                        target
                    );
                }
            }
        }
    }

    #[test]
    fn runtime_capabilities_preserve_input_derived_approval_keys() {
        let fixtures = vec![
            (
                "Bash",
                serde_json::json!({ "command": "cargo test -p mangocode-query --locked" }),
                Some(("command_prefix", "cargo test")),
                None,
                None,
            ),
            (
                "PowerShell",
                serde_json::json!({ "command": "Get-Content crates/query/src/lib.rs" }),
                Some(("command_prefix", "Get-Content crates/query/src/lib.rs")),
                None,
                None,
            ),
            (
                "Read",
                serde_json::json!({ "file_path": "crates/query/src/lib.rs" }),
                None,
                Some("crates/query/src/lib.rs"),
                None,
            ),
            (
                "Write",
                serde_json::json!({ "file_path": "src/generated.rs", "content": "fn main() {}" }),
                None,
                Some("src/generated.rs"),
                None,
            ),
            (
                "Edit",
                serde_json::json!({ "file_path": "src/lib.rs", "old_string": "a", "new_string": "b" }),
                None,
                Some("src/lib.rs"),
                None,
            ),
            (
                "WebFetch",
                serde_json::json!({ "url": "https://example.com/docs" }),
                None,
                None,
                Some("https://example.com"),
            ),
            (
                "Browser",
                serde_json::json!({ "url": "https://example.com/app" }),
                None,
                None,
                Some("https://example.com"),
            ),
        ];

        for (name, input, expected_key, expected_path, expected_host) in fixtures {
            let Some(tool) = find_tool(name) else {
                continue;
            };
            let capabilities = tool.capabilities(&input);
            if let Some((kind, value)) = expected_key {
                assert!(
                    capabilities
                        .approval_keys
                        .iter()
                        .any(|key| key.kind == kind && key.value == value),
                    "{name} missing expected {kind} approval key for {value}: {:?}",
                    capabilities.approval_keys
                );
            }
            if let Some(path) = expected_path {
                assert!(
                    capabilities.affected_paths.iter().any(|item| item == path),
                    "{name} missing affected path {path}: {:?}",
                    capabilities.affected_paths
                );
                assert!(
                    capabilities
                        .approval_keys
                        .iter()
                        .any(|key| key.kind == "path" && key.value == path),
                    "{name} missing path approval key for {path}: {:?}",
                    capabilities.approval_keys
                );
            }
            if let Some(host) = expected_host {
                assert!(
                    capabilities.network_targets.iter().any(|item| item == host),
                    "{name} missing network target {host}: {:?}",
                    capabilities.network_targets
                );
                assert!(
                    capabilities
                        .approval_keys
                        .iter()
                        .any(|key| key.kind == "host" && key.value == host),
                    "{name} missing host approval key for {host}: {:?}",
                    capabilities.approval_keys
                );
            }
        }
    }

    #[test]
    fn runtime_capabilities_extract_extended_output_path_keys() {
        let capabilities = default_capabilities_for_tool(
            "SyntheticOutput",
            PermissionLevel::Write,
            &serde_json::json!({
                "output_path": "out/report.md",
                "graph_path": "graphify-out/graph.json",
                "global_dir": ".mangocode/project-graph"
            }),
        );

        for path in [
            "out/report.md",
            "graphify-out/graph.json",
            ".mangocode/project-graph",
        ] {
            assert!(
                capabilities.affected_paths.iter().any(|item| item == path),
                "missing affected path {path}: {:?}",
                capabilities.affected_paths
            );
            assert!(
                capabilities
                    .approval_keys
                    .iter()
                    .any(|key| key.kind == "path" && key.value == path),
                "missing path approval key for {path}: {:?}",
                capabilities.approval_keys
            );
        }
    }

    #[test]
    fn runtime_capabilities_extract_apply_patch_paths_from_unified_diff() {
        let capabilities = default_capabilities_for_tool(
            "ApplyPatch",
            PermissionLevel::Write,
            &serde_json::json!({
                "patch": "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-old\n+new\n--- /dev/null\n+++ \"b/src/new file.rs\"\n@@ -0,0 +1 @@\n+new\n"
            }),
        );

        for path in ["src/lib.rs", "src/new file.rs"] {
            assert!(
                capabilities.affected_paths.iter().any(|item| item == path),
                "missing affected path {path}: {:?}",
                capabilities.affected_paths
            );
            assert!(
                capabilities
                    .approval_keys
                    .iter()
                    .any(|key| key.kind == "path" && key.value == path),
                "missing path approval key for {path}: {:?}",
                capabilities.approval_keys
            );
        }
        assert!(
            !capabilities
                .affected_paths
                .iter()
                .any(|path| path == "/dev/null"),
            "diff null path should not be treated as affected: {:?}",
            capabilities.affected_paths
        );
    }

    #[test]
    fn runtime_capabilities_treat_apply_patch_dry_run_as_read_only() {
        let capabilities = default_capabilities_for_tool(
            "ApplyPatch",
            PermissionLevel::Write,
            &serde_json::json!({
                "dry_run": true,
                "patch": "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-old\n+new\n"
            }),
        );

        assert!(!capabilities.mutating);
        assert!(capabilities.parallel_safe);
        assert_eq!(capabilities.sandbox_preference, SandboxPreference::ReadOnly);
        assert!(capabilities
            .affected_paths
            .iter()
            .any(|path| path == "src/lib.rs"));
    }

    #[test]
    fn test_find_tool_found() {
        #[cfg(feature = "tool-bash")]
        {
            let tool = find_tool("Bash");
            assert!(tool.is_some(), "Should find the Bash tool");
            assert_eq!(tool.unwrap().name(), "Bash");
        }

        #[cfg(not(feature = "tool-bash"))]
        if let Some(first_tool) = all_tools().first() {
            let tool = find_tool(first_tool.name());
            assert!(tool.is_some(), "Should find an enabled built-in tool");
            assert_eq!(tool.unwrap().name(), first_tool.name());
        }
    }

    #[test]
    fn test_find_tool_not_found() {
        assert!(find_tool("NonExistentTool12345").is_none());
    }

    #[test]
    fn test_find_tool_case_sensitive() {
        // Tool names are case-sensitive — "bash" should not match "Bash"
        assert!(find_tool("bash").is_none());
        #[cfg(feature = "tool-bash")]
        assert!(find_tool("Bash").is_some());
    }

    #[test]
    fn test_runtime_aliases_resolve_without_renaming_tools() {
        #[allow(unused_variables)]
        let tools = all_tools();
        #[cfg(feature = "tool-bash")]
        {
            let shell = resolve_tool(&tools, "shell_command").expect("shell alias should resolve");
            assert_eq!(shell.name(), "Bash");
            assert!(tool_name_matches(shell, "container.exec"));
        }
        #[cfg(feature = "tool-view-image")]
        {
            let image = resolve_tool(&tools, "view_image").expect("image alias should resolve");
            assert_eq!(image.name(), "ViewImage");
        }
        #[cfg(feature = "tool-lsp")]
        {
            let lsp = resolve_tool(&tools, "lsp").expect("lsp alias should resolve");
            assert_eq!(lsp.name(), "LSP");
        }
        #[cfg(feature = "tool-repl")]
        {
            let repl = resolve_tool(&tools, "node_repl").expect("repl alias should resolve");
            assert_eq!(repl.name(), "REPL");
        }
        #[cfg(feature = "tool-mcp-auth")]
        {
            let mcp_auth = resolve_tool(&tools, "mcp_auth").expect("mcp auth alias should resolve");
            assert_eq!(mcp_auth.name(), "mcp__auth");
        }
        #[cfg(feature = "tool-get-goal")]
        {
            let get_goal = resolve_tool(&tools, "GetGoal").expect("goal alias should resolve");
            assert_eq!(get_goal.name(), "get_goal");
        }
        #[cfg(feature = "tool-create-goal")]
        {
            let create_goal =
                resolve_tool(&tools, "CreateGoal").expect("goal alias should resolve");
            assert_eq!(create_goal.name(), "create_goal");
        }
        #[cfg(feature = "tool-update-goal")]
        {
            let update_goal =
                resolve_tool(&tools, "UpdateGoal").expect("goal alias should resolve");
            assert_eq!(update_goal.name(), "update_goal");
        }
        #[cfg(any(
            feature = "tool-apply-patch",
            feature = "tool-get-goal",
            feature = "tool-todo-write",
            feature = "tool-update-plan",
            feature = "tool-web-fetch",
            feature = "tool-web-search"
        ))]
        let plan = build_registry_plan(&tools);
        #[cfg(feature = "tool-apply-patch")]
        assert_eq!(plan.canonical_name("apply_patch"), Some("ApplyPatch"));
        #[cfg(feature = "tool-update-plan")]
        assert_eq!(plan.canonical_name("update_plan"), Some("update_plan"));
        #[cfg(feature = "tool-todo-write")]
        assert_eq!(plan.canonical_name("TodoWrite"), Some("TodoWrite"));
        #[cfg(feature = "tool-todo-write")]
        assert_eq!(plan.canonical_name("todo_write"), Some("TodoWrite"));
        #[cfg(feature = "tool-get-goal")]
        assert_eq!(plan.canonical_name("GetGoal"), Some("get_goal"));
        #[cfg(feature = "tool-web-search")]
        assert_eq!(plan.canonical_name("web_search"), Some("WebSearch"));
        #[cfg(feature = "tool-web-fetch")]
        assert_eq!(plan.canonical_name("web_fetch"), Some("WebFetch"));
    }

    #[test]
    fn default_aliases_include_generated_snake_case_for_camel_tools() {
        assert!(default_aliases_for_tool("WebSearch")
            .iter()
            .any(|alias| alias == "web_search"));
        assert!(default_aliases_for_tool("WebFetch")
            .iter()
            .any(|alias| alias == "web_fetch"));
        assert!(default_aliases_for_tool("AskUserQuestion")
            .iter()
            .any(|alias| alias == "ask_user_question"));
        assert!(default_aliases_for_tool("ListMcpResources")
            .iter()
            .any(|alias| alias == "list_mcp_resources"));
        assert!(!default_aliases_for_tool("REPL")
            .iter()
            .any(|alias| alias == "r_e_p_l"));
    }

    #[cfg(all(feature = "tool-bash", feature = "tool-read"))]
    #[test]
    fn test_filter_tools_by_name_config_accepts_aliases() {
        let tools = filter_tools_by_name_config(
            all_tools(),
            &["shell_command".to_string(), "read".to_string()],
            &["container.exec".to_string()],
        );

        assert!(tools.iter().any(|tool| tool.name() == "Read"));
        assert!(!tools.iter().any(|tool| tool.name() == "Bash"));
    }

    #[cfg(feature = "tool-web-search")]
    #[test]
    fn test_filter_tools_by_name_config_accepts_generated_snake_case_aliases() {
        let tools = filter_tools_by_name_config(all_tools(), &["web_search".to_string()], &[]);

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "WebSearch");
    }

    #[cfg(all(feature = "tool-todo-write", feature = "tool-update-plan"))]
    #[test]
    fn test_update_plan_and_todo_write_visibility_are_distinct() {
        let todo_only = filter_tools_by_name_config(all_tools(), &["TodoWrite".to_string()], &[]);
        assert!(todo_only.iter().any(|tool| tool.name() == "TodoWrite"));
        assert!(!todo_only.iter().any(|tool| tool.name() == "update_plan"));

        let plan_only = filter_tools_by_name_config(all_tools(), &["update_plan".to_string()], &[]);
        assert!(plan_only.iter().any(|tool| tool.name() == "update_plan"));
        assert!(!plan_only.iter().any(|tool| tool.name() == "TodoWrite"));

        let without_todo =
            filter_tools_by_name_config(all_tools(), &[], &["TodoWrite".to_string()]);
        assert!(!without_todo.iter().any(|tool| tool.name() == "TodoWrite"));
        assert!(without_todo.iter().any(|tool| tool.name() == "update_plan"));
    }

    #[cfg(all(feature = "tool-bash", feature = "tool-tool-search"))]
    #[test]
    fn restrict_config_to_available_tools_uses_runtime_names() {
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(BashTool), Box::new(ToolSearchTool)];
        let mut config = mangocode_core::config::Config {
            allowed_tools: vec!["shell_command".to_string()],
            disallowed_tools: vec!["read".to_string()],
            ..Default::default()
        };

        restrict_config_to_available_tools(&mut config, &tools);

        assert_eq!(config.allowed_tools, vec!["Bash", "ToolSearch"]);
        assert!(config.disallowed_tools.is_empty());
    }

    #[test]
    fn unknown_tool_message_mentions_runtime_visibility_config() {
        let tools: Vec<Box<dyn Tool>> = Vec::new();
        let msg = unknown_tool_message(&tools, "DefinitelyMissing");

        assert!(msg.contains("runtime-visible tools"));
        assert!(msg.contains("session visibility config"));
    }

    struct SuggestionOnlyTool;

    #[async_trait::async_trait]
    impl Tool for SuggestionOnlyTool {
        fn name(&self) -> &str {
            "VisibleTool"
        }

        fn description(&self) -> &str {
            "A visible test tool"
        }

        fn permission_level(&self) -> PermissionLevel {
            PermissionLevel::ReadOnly
        }

        fn input_schema(&self) -> Value {
            serde_json::json!({ "type": "object" })
        }

        async fn execute(&self, _input: Value, _ctx: &ToolContext) -> ToolResult {
            ToolResult::success("ok")
        }
    }

    #[test]
    fn unknown_tool_suggestion_message_mentions_visibility_config() {
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(SuggestionOnlyTool)];
        let msg = unknown_tool_message(&tools, "VisibleToo");

        assert!(msg.contains("Did you mean VisibleTool"));
        assert!(msg.contains("runtime-visible tool discovery"));
        assert!(msg.contains("session visibility config"));
    }

    #[test]
    fn test_mcp_compat_names_keep_only_executable_aliases() {
        let (name, aliases) = mcp_compat_names("github", "github_create_issue");
        assert_eq!(name, "mcp__github__create_issue");
        assert_eq!(aliases, vec!["github_create_issue", "create_issue"]);
    }

    #[test]
    fn test_agent_defaults_to_serialized_runtime_capabilities() {
        let capabilities = default_capabilities_for_tool(
            mangocode_core::constants::TOOL_NAME_AGENT,
            PermissionLevel::None,
            &serde_json::Value::Null,
        );
        assert!(capabilities.mutating);
        assert!(!capabilities.parallel_safe);
        assert!(capabilities
            .aliases
            .iter()
            .any(|alias| alias == "spawn_agent"));
    }

    #[test]
    fn test_notebook_path_is_reported_as_affected_path() {
        let capabilities = default_capabilities_for_tool(
            mangocode_core::constants::TOOL_NAME_NOTEBOOK_EDIT,
            PermissionLevel::Write,
            &serde_json::json!({ "notebook_path": "analysis.ipynb" }),
        );

        assert_eq!(capabilities.affected_paths, vec!["analysis.ipynb"]);
        assert!(capabilities
            .approval_keys
            .iter()
            .any(|key| { key.kind == "path" && key.value == "analysis.ipynb" }));
    }

    #[test]
    fn command_prefix_preserves_quoted_executable_paths() {
        let capabilities = default_capabilities_for_tool(
            "Bash",
            PermissionLevel::Execute,
            &serde_json::json!({
                "command": r#""C:\Program Files\Mango Tool\tool.exe" --flag value"#
            }),
        );

        assert!(capabilities.approval_keys.iter().any(|key| {
            key.kind == "command_prefix"
                && key.value == r"C:\Program Files\Mango Tool\tool.exe --flag"
        }));
    }

    #[test]
    fn command_prefix_uses_specific_key_for_malformed_quoted_commands() {
        let capabilities = default_capabilities_for_tool(
            "Bash",
            PermissionLevel::Execute,
            &serde_json::json!({ "command": r#"git commit "unterminated"# }),
        );

        assert!(capabilities.approval_keys.iter().any(|key| {
            key.kind == "command_prefix"
                && key.value == r#"invalid command syntax: git commit "unterminated"#
        }));
    }

    #[test]
    fn network_targets_include_secure_websockets_and_strip_userinfo() {
        let capabilities = default_capabilities_for_tool(
            "Browser",
            PermissionLevel::Network,
            &serde_json::json!({
                "url": "wss://user:secret@Example.COM:9443/devtools/page?id=1",
            }),
        );

        assert_eq!(capabilities.network_targets, vec!["wss://example.com:9443"]);
        assert!(capabilities
            .approval_keys
            .iter()
            .any(|key| { key.kind == "host" && key.value == "wss://example.com:9443" }));
    }

    #[test]
    fn network_targets_include_schemeless_host_fields() {
        let capabilities = default_capabilities_for_tool(
            "RemoteTrigger",
            PermissionLevel::Network,
            &serde_json::json!({
                "host": "User:Secret@API.Example.COM:8443/api?token=redacted",
                "url": "api.example.com/not-a-url",
                "urls": ["https://Docs.Example.COM/path", "also-not-a-url"],
            }),
        );

        assert_eq!(
            capabilities.network_targets,
            vec!["api.example.com:8443", "https://docs.example.com"]
        );
        assert!(capabilities
            .approval_keys
            .iter()
            .any(|key| { key.kind == "host" && key.value == "api.example.com:8443" }));
        assert!(capabilities
            .approval_keys
            .iter()
            .any(|key| { key.kind == "host" && key.value == "https://docs.example.com" }));
    }

    #[test]
    fn test_mutating_tools_are_not_parallel_safe() {
        #[allow(unused_variables)]
        let tools = all_tools();
        #[cfg(feature = "tool-apply-patch")]
        assert!(!tool_supports_parallel(
            &tools,
            "apply_patch",
            &serde_json::json!({ "patch": "" })
        ));
        #[cfg(feature = "tool-grep")]
        assert!(tool_supports_parallel(
            &tools,
            "grep",
            &serde_json::json!({ "pattern": "fn", "path": "." })
        ));
        #[cfg(feature = "tool-get-goal")]
        assert!(!tool_supports_parallel(
            &tools,
            "get_goal",
            &serde_json::json!({})
        ));
        #[cfg(feature = "tool-create-goal")]
        assert!(!tool_supports_parallel(
            &tools,
            "create_goal",
            &serde_json::json!({ "objective": "test" })
        ));
        #[cfg(feature = "tool-update-goal")]
        assert!(!tool_supports_parallel(
            &tools,
            "update_goal",
            &serde_json::json!({ "status": "complete" })
        ));
        #[cfg(feature = "tool-update-plan")]
        assert!(!tool_supports_parallel(
            &tools,
            "update_plan",
            &serde_json::json!({ "plan": [] })
        ));
        #[cfg(feature = "tool-coordination")]
        {
            assert!(!tool_supports_parallel(
                &tools,
                "CoordinationStatus",
                &serde_json::json!({ "include_inbox": true, "mark_read": true })
            ));
            assert!(!tool_supports_parallel(
                &tools,
                "CoordinationInbox",
                &serde_json::json!({ "mark_read": true })
            ));
            assert!(!tool_supports_parallel(
                &tools,
                "CoordinationMessage",
                &serde_json::json!({ "body": "hi", "repo_broadcast": true })
            ));
            assert!(!tool_supports_parallel(
                &tools,
                "ClaimWork",
                &serde_json::json!({ "scope": "src" })
            ));
            assert!(!tool_supports_parallel(
                &tools,
                "ReleaseWork",
                &serde_json::json!({})
            ));
        }
    }

    #[test]
    fn test_core_tools_present() {
        let expected = [
            ("Bash", cfg!(feature = "tool-bash")),
            ("Read", cfg!(feature = "tool-read")),
            ("Edit", cfg!(feature = "tool-edit")),
            ("Write", cfg!(feature = "tool-write")),
            ("Glob", cfg!(feature = "tool-glob")),
            ("Grep", cfg!(feature = "tool-grep")),
            ("CodeSearch", cfg!(feature = "tool-grep")),
            ("TodoWrite", cfg!(feature = "tool-todo-write")),
            ("update_plan", cfg!(feature = "tool-update-plan")),
            ("get_goal", cfg!(feature = "tool-get-goal")),
            ("create_goal", cfg!(feature = "tool-create-goal")),
            ("update_goal", cfg!(feature = "tool-update-goal")),
            ("Skill", cfg!(feature = "tool-skill")),
        ];
        for (name, should_exist) in expected {
            assert_eq!(
                find_tool(name).is_some(),
                should_exist,
                "tool presence should match feature gate for {name}"
            );
        }
    }

    #[test]
    fn test_feature_gated_web_research_tools_follow_cfg() {
        let expected = [
            ("WebFetch", cfg!(feature = "tool-web-fetch")),
            ("WebSearch", cfg!(feature = "tool-web-search")),
            ("DocSearch", cfg!(feature = "tool-doc-search")),
            ("DocRead", cfg!(feature = "tool-doc-read")),
            ("DeepRead", cfg!(feature = "tool-deep-read")),
            ("RenderedFetch", cfg!(feature = "tool-rendered-fetch")),
            (
                "ListMcpResources",
                cfg!(feature = "tool-list-mcp-resources"),
            ),
            ("ReadMcpResource", cfg!(feature = "tool-read-mcp-resource")),
            ("mcp__auth", cfg!(feature = "tool-mcp-auth")),
        ];

        for (name, should_exist) in expected {
            assert_eq!(
                find_tool(name).is_some(),
                should_exist,
                "tool presence should match feature gate for {name}"
            );
        }
    }

    #[test]
    fn rendered_fetch_feature_does_not_register_interactive_browser_tool_by_itself() {
        if cfg!(feature = "tool-rendered-fetch") && !cfg!(feature = "tool-browser") {
            assert!(find_tool("RenderedFetch").is_some());
            assert!(find_tool("Browser").is_none());
        }
    }

    #[test]
    fn browser_feature_does_not_register_rendered_fetch_tool_by_itself() {
        if cfg!(feature = "tool-browser") && !cfg!(feature = "tool-rendered-fetch") {
            assert!(find_tool("Browser").is_some());
            assert!(find_tool("RenderedFetch").is_none());
        }
    }

    // ---- ToolResult tests ---------------------------------------------------

    #[test]
    fn test_tool_result_success() {
        let r = ToolResult::success("done");
        assert!(!r.is_error);
        assert_eq!(r.content, "done");
        assert!(r.metadata.is_none());
    }

    #[test]
    fn test_tool_result_error() {
        let r = ToolResult::error("something went wrong");
        assert!(r.is_error);
        assert_eq!(r.content, "something went wrong");
    }

    #[test]
    fn test_tool_result_with_metadata() {
        let r = ToolResult::success("ok")
            .with_metadata(serde_json::json!({"file": "foo.rs", "lines": 10}));
        assert!(r.metadata.is_some());
        let meta = r.metadata.unwrap();
        assert_eq!(meta["file"], "foo.rs");
    }

    #[test]
    fn test_tool_result_envelope_preserves_raw_log_path() {
        let r = ToolResult::success("ok")
            .with_metadata(serde_json::json!({"raw_log_path": "logs/out.txt"}));
        let envelope = r.to_envelope();
        assert_eq!(envelope.raw_log_path.as_deref(), Some("logs/out.txt"));
    }

    // ---- ToolContext::resolve_path tests ------------------------------------

    #[test]
    fn test_resolve_path_absolute() {
        use mangocode_core::config::Config;
        use mangocode_core::permissions::AutoPermissionHandler;

        let handler = Arc::new(AutoPermissionHandler {
            mode: mangocode_core::config::PermissionMode::Default,
        });
        let ctx = ToolContext {
            working_dir: PathBuf::from("/workspace"),
            permission_mode: mangocode_core::config::PermissionMode::Default,
            permission_handler: handler,
            cost_tracker: mangocode_core::cost::CostTracker::new(),
            session_metrics: None,
            session_id: "test".to_string(),
            coordination_actor_id: None,
            coordination_parent_actor_id: None,
            inject_coordination_inbox: true,
            file_history: Arc::new(parking_lot::Mutex::new(
                mangocode_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(AtomicUsize::new(0)),
            non_interactive: true,
            mcp_manager: None,
            config: Config::default(),
            question_prompt_tx: None,
            cancel_token: None,
        };

        // Absolute paths pass through unchanged
        let resolved = ctx.resolve_path("/absolute/path/file.rs");
        assert_eq!(resolved, PathBuf::from("/absolute/path/file.rs"));
    }

    #[test]
    fn test_resolve_path_relative() {
        use mangocode_core::config::Config;
        use mangocode_core::permissions::AutoPermissionHandler;

        let handler = Arc::new(AutoPermissionHandler {
            mode: mangocode_core::config::PermissionMode::Default,
        });
        let ctx = ToolContext {
            working_dir: PathBuf::from("/workspace"),
            permission_mode: mangocode_core::config::PermissionMode::Default,
            permission_handler: handler,
            cost_tracker: mangocode_core::cost::CostTracker::new(),
            session_metrics: None,
            session_id: "test".to_string(),
            coordination_actor_id: None,
            coordination_parent_actor_id: None,
            inject_coordination_inbox: true,
            file_history: Arc::new(parking_lot::Mutex::new(
                mangocode_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(AtomicUsize::new(0)),
            non_interactive: true,
            mcp_manager: None,
            config: Config::default(),
            question_prompt_tx: None,
            cancel_token: None,
        };

        // Relative paths get joined with working_dir
        let resolved = ctx.resolve_path("src/main.rs");
        assert_eq!(resolved, PathBuf::from("/workspace/src/main.rs"));
    }

    #[test]
    fn test_coordination_parent_fallback_uses_registered_actor_identity() {
        use mangocode_core::config::Config;
        use mangocode_core::permissions::AutoPermissionHandler;

        let handler = Arc::new(AutoPermissionHandler {
            mode: mangocode_core::config::PermissionMode::Default,
        });
        let mut ctx = ToolContext {
            working_dir: PathBuf::from("/workspace"),
            permission_mode: mangocode_core::config::PermissionMode::Default,
            permission_handler: handler,
            cost_tracker: mangocode_core::cost::CostTracker::new(),
            session_metrics: None,
            session_id: "parent-session".to_string(),
            coordination_actor_id: Some("child-actor".to_string()),
            coordination_parent_actor_id: None,
            inject_coordination_inbox: true,
            file_history: Arc::new(parking_lot::Mutex::new(
                mangocode_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(AtomicUsize::new(0)),
            non_interactive: true,
            mcp_manager: None,
            config: Config::default(),
            question_prompt_tx: None,
            cancel_token: None,
        };

        let expected_parent = mangocode_core::coordination::process_session_id("parent-session");
        assert_eq!(
            ctx.coordination_parent_session_id().as_deref(),
            Some(expected_parent.as_str())
        );

        ctx.coordination_parent_actor_id = Some("explicit-parent".to_string());
        assert_eq!(
            ctx.coordination_parent_session_id().as_deref(),
            Some("explicit-parent")
        );
    }

    // ---- PermissionLevel tests ---------------------------------------------

    #[test]
    fn test_permission_level_order() {
        // Just verify the variants exist and are distinct
        assert_ne!(PermissionLevel::None, PermissionLevel::ReadOnly);
        assert_ne!(PermissionLevel::ReadOnly, PermissionLevel::Network);
        assert_ne!(PermissionLevel::Write, PermissionLevel::Execute);
        assert_ne!(PermissionLevel::Execute, PermissionLevel::Dangerous);
    }

    #[test]
    #[cfg(feature = "tool-bash")]
    fn test_bash_tool_permission_level() {
        assert_eq!(PtyBashTool.permission_level(), PermissionLevel::Execute);
    }

    #[test]
    #[cfg(feature = "tool-read")]
    fn test_file_read_permission_level() {
        assert_eq!(FileReadTool.permission_level(), PermissionLevel::ReadOnly);
    }

    #[test]
    #[cfg(feature = "tool-edit")]
    fn test_file_edit_permission_level() {
        assert_eq!(FileEditTool.permission_level(), PermissionLevel::Write);
    }

    #[test]
    #[cfg(feature = "tool-write")]
    fn test_file_write_permission_level() {
        assert_eq!(FileWriteTool.permission_level(), PermissionLevel::Write);
    }

    #[test]
    #[cfg(feature = "tool-cron-create")]
    fn test_cron_create_permission_level() {
        assert_eq!(CronCreateTool.permission_level(), PermissionLevel::Write);
    }

    #[test]
    #[cfg(feature = "tool-cron-delete")]
    fn test_cron_delete_permission_level() {
        assert_eq!(CronDeleteTool.permission_level(), PermissionLevel::Write);
    }

    #[test]
    #[cfg(feature = "tool-web-search")]
    fn test_web_search_permission_level() {
        assert_eq!(WebSearchTool.permission_level(), PermissionLevel::Network);
    }

    #[test]
    #[cfg(feature = "tool-web-fetch")]
    fn test_web_fetch_permission_level() {
        assert_eq!(WebFetchTool.permission_level(), PermissionLevel::Network);
    }

    #[test]
    #[cfg(feature = "tool-browser")]
    fn test_browser_permission_level() {
        assert_eq!(BrowserTool.permission_level(), PermissionLevel::Dangerous);
    }

    #[test]
    #[cfg(feature = "tool-remote-trigger")]
    fn test_remote_trigger_permission_level() {
        assert_eq!(
            RemoteTriggerTool.permission_level(),
            PermissionLevel::Network
        );
    }

    #[test]
    #[cfg(feature = "tool-list-mcp-resources")]
    fn test_list_mcp_resources_permission_level() {
        assert_eq!(
            ListMcpResourcesTool.permission_level(),
            PermissionLevel::Network
        );
    }

    #[test]
    #[cfg(feature = "tool-read-mcp-resource")]
    fn test_read_mcp_resource_permission_level() {
        assert_eq!(
            ReadMcpResourceTool.permission_level(),
            PermissionLevel::Network
        );
    }

    #[test]
    #[cfg(feature = "tool-mcp-auth")]
    fn test_mcp_auth_permission_level() {
        assert_eq!(McpAuthTool.permission_level(), PermissionLevel::Network);
    }

    #[test]
    fn runtime_tools_match_core_permission_classifier() {
        use mangocode_core::permissions::PermissionLevel as CorePermissionLevel;

        for tool in all_tools() {
            let core_level = CorePermissionLevel::for_tool(tool.name());
            let expected = match tool.permission_level() {
                PermissionLevel::None | PermissionLevel::ReadOnly => CorePermissionLevel::Read,
                PermissionLevel::Network => CorePermissionLevel::Network,
                PermissionLevel::Write => CorePermissionLevel::Write,
                PermissionLevel::Execute
                | PermissionLevel::Dangerous
                | PermissionLevel::Forbidden => CorePermissionLevel::Execute,
            };
            assert_eq!(
                core_level,
                expected,
                "core permission classifier mismatch for {}",
                tool.name()
            );
        }
    }

    // ---- Tool to_definition tests ------------------------------------------

    #[test]
    #[cfg(feature = "tool-bash")]
    fn test_tool_to_definition() {
        let def = PtyBashTool.to_definition();
        assert_eq!(def.name, "Bash");
        assert!(!def.description.is_empty());
        assert!(def.input_schema.is_object());
    }
}
