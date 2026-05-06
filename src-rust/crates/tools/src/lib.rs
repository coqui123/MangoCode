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
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// Sub-modules – each contains a full tool implementation.
#[cfg(feature = "tool-apply-patch")]
pub mod apply_patch;
#[cfg(feature = "tool-ask-user")]
pub mod ask_user;
#[cfg(feature = "tool-bash")]
pub mod bash;
#[cfg(feature = "tool-batch-edit")]
pub mod batch_edit;
#[cfg(feature = "tool-brief")]
pub mod brief;
pub mod browser_tool;
#[cfg(feature = "tool-skill")]
pub mod bundled_skills;
#[cfg(feature = "tool-computer-use")]
pub mod computer_use;
#[cfg(feature = "tool-config")]
pub mod config_tool;
pub mod cron;
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
#[cfg(feature = "tool-bash")]
pub mod pty_bash;
#[cfg(feature = "tool-remote-trigger")]
pub mod remote_trigger;
#[cfg(feature = "tool-repl")]
pub mod repl_tool;
#[cfg(any(
    feature = "tool-doc-search",
    feature = "tool-doc-read",
    feature = "tool-deep-read",
    feature = "tool-rendered-fetch"
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
#[cfg(feature = "tool-ask-user")]
pub use ask_user::AskUserQuestionTool;
#[cfg(feature = "tool-bash")]
pub use bash::BashTool;
#[cfg(feature = "tool-batch-edit")]
pub use batch_edit::BatchEditTool;
#[cfg(feature = "tool-brief")]
pub use brief::BriefTool;
#[cfg(feature = "tool-browser")]
pub use browser_tool::BrowserTool;
#[cfg(feature = "tool-computer-use")]
pub use computer_use::ComputerUseTool;
#[cfg(feature = "tool-config")]
pub use config_tool::ConfigTool;
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
pub use tasks::{Task, TaskStatus, TASK_STORE};
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
pub use team_tool::{register_agent_runner, AgentRunFn};
#[cfg(feature = "tool-team-create")]
pub use team_tool::TeamCreateTool;
#[cfg(feature = "tool-team-delete")]
pub use team_tool::TeamDeleteTool;
#[cfg(feature = "tool-todo-write")]
pub use todo_write::TodoWriteTool;
#[cfg(feature = "tool-tool-search")]
pub use tool_search::ToolSearchTool;
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
    /// Read-only access to the filesystem or network.
    ReadOnly,
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
    pub file_history: Arc<parking_lot::Mutex<mangocode_core::file_history::FileHistory>>,
    pub current_turn: Arc<AtomicUsize>,
    /// If true, suppress interactive prompts (batch / CI mode).
    pub non_interactive: bool,
    /// Optional MCP manager for ListMcpResources / ReadMcpResource tools.
    pub mcp_manager: Option<Arc<mangocode_mcp::McpManager>>,
    /// Configured event hooks (PreToolUse, PostToolUse, etc.).
    pub config: mangocode_core::config::Config,
}

impl ToolContext {
    /// Resolve a potentially relative path against the working directory.
    pub fn resolve_path(&self, path: &str) -> PathBuf {
        let p = PathBuf::from(path);
        if p.is_absolute() {
            p
        } else {
            self.working_dir.join(p)
        }
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

/// Return all built-in tools (excluding AgentTool, which lives in cc-query).
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
        #[cfg(feature = "tool-brief")]
        Box::new(BriefTool),
        #[cfg(feature = "tool-config")]
        Box::new(ConfigTool),
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

    async fn execute(&self, input: Value, _ctx: &ToolContext) -> ToolResult {
        let prefix = format!("mcp__{}__", self.server_name);
        let bare_name = self
            .tool_def
            .name
            .strip_prefix(&prefix)
            .unwrap_or(&self.tool_def.name);

        let args = if input.is_null() { None } else { Some(input) };

        match self.manager.call_tool(&self.manager_tool_name, args).await {
            Ok(result) => {
                let text = mangocode_mcp::mcp_result_to_string(&result);
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
            "Unknown tool: {requested}. Use ToolSearch to discover available tools, or check whether the relevant plugin, MCP server, or feature flag is enabled."
        )
    } else {
        format!(
            "Unknown tool: {requested}. Did you mean {}? You can also use ToolSearch for live tool discovery.",
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
        "TodoWrite" => vec!["update_plan", "todo_write"],
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
    mangocode_tool_runtime::dedupe_strings(aliases.into_iter().map(str::to_string))
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
    let mut capabilities = match permission_level {
        PermissionLevel::None | PermissionLevel::ReadOnly => ToolCapabilities::read_only(),
        PermissionLevel::Write
        | PermissionLevel::Execute
        | PermissionLevel::Dangerous
        | PermissionLevel::Forbidden => ToolCapabilities::mutating(),
    };

    capabilities.aliases = default_aliases_for_tool(name);
    capabilities.affected_paths = extract_path_values(input);
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
            "path",
            "paths",
            "old_path",
            "new_path",
            "target_path",
        ],
        &mut paths,
    );
    mangocode_tool_runtime::dedupe_strings(
        paths
            .into_iter()
            .filter(|value| !looks_like_network_target(value)),
    )
}

fn extract_network_targets(input: &Value) -> Vec<String> {
    let mut values = Vec::new();
    collect_string_values_for_keys(input, &["url", "urls", "uri", "host"], &mut values);
    mangocode_tool_runtime::dedupe_strings(
        values
            .into_iter()
            .filter(|value| looks_like_network_target(value))
            .map(|value| network_approval_target(&value)),
    )
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
    let prefix = command
        .split_whitespace()
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

fn looks_like_network_target(value: &str) -> bool {
    let value = value.trim().to_ascii_lowercase();
    value.starts_with("http://") || value.starts_with("https://") || value.starts_with("ws://")
}

fn network_approval_target(value: &str) -> String {
    let trimmed = value.trim();
    let Some((scheme, rest)) = trimmed.split_once("://") else {
        return trimmed.to_string();
    };
    let authority = rest
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(rest)
        .trim_end_matches('/');
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
        cfg!(feature = "tool-agent")
            || cfg!(feature = "tool-apply-patch")
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
            feature = "tool-todo-write"
        ))]
        let plan = build_registry_plan(&tools);
        #[cfg(feature = "tool-apply-patch")]
        assert_eq!(plan.canonical_name("apply_patch"), Some("ApplyPatch"));
        #[cfg(feature = "tool-todo-write")]
        assert_eq!(plan.canonical_name("update_plan"), Some("TodoWrite"));
        #[cfg(feature = "tool-get-goal")]
        assert_eq!(plan.canonical_name("GetGoal"), Some("get_goal"));
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
            ("TodoWrite", cfg!(feature = "tool-todo-write")),
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
            file_history: Arc::new(parking_lot::Mutex::new(
                mangocode_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(AtomicUsize::new(0)),
            non_interactive: true,
            mcp_manager: None,
            config: Config::default(),
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
            file_history: Arc::new(parking_lot::Mutex::new(
                mangocode_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(AtomicUsize::new(0)),
            non_interactive: true,
            mcp_manager: None,
            config: Config::default(),
        };

        // Relative paths get joined with working_dir
        let resolved = ctx.resolve_path("src/main.rs");
        assert_eq!(resolved, PathBuf::from("/workspace/src/main.rs"));
    }

    // ---- PermissionLevel tests ---------------------------------------------

    #[test]
    fn test_permission_level_order() {
        // Just verify the variants exist and are distinct
        assert_ne!(PermissionLevel::None, PermissionLevel::ReadOnly);
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
