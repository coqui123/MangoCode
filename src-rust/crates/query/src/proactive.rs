// proactive: Background proactive monitor loop.
//
// The proactive agent runs as a detached task and periodically gathers project
// state, then asks the model whether anything is worth surfacing right now.
// It is observe-only and explicitly tool-restricted.

use crate::{run_query_loop, QueryConfig, QueryOutcome};
use async_trait::async_trait;
use mangocode_core::lsp::{global_lsp_manager, LspDiagnostic};
use mangocode_core::types::Message;
use mangocode_tools::todo_write::load_todos;
use mangocode_tools::{PermissionLevel, Tool, ToolContext, ToolResult};
use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher};
use parking_lot::Mutex;
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Allowed tool names for the proactive agent (observe-only)
// ---------------------------------------------------------------------------

const ALLOWED_TOOLS: &[&str] = &[
    "Read",
    "Grep",
    "Glob",
    "Bash",
    "LSP",
    "Brief",
];

const DENIED_TOOLS: &[&str] = &[
    "Write",
    "Edit",
    "NotebookEdit",
    "Agent",
    "SendMessage",
    "TodoWrite",
    "CronCreate",
    "CronDelete",
];

// ---------------------------------------------------------------------------
// Context gathered each heartbeat
// ---------------------------------------------------------------------------

/// Snapshot of project state collected before each heartbeat.
#[derive(Debug, Clone, Default)]
pub struct ProactiveContext {
    /// Last tool execution error from session log.
    pub last_error: Option<String>,
    /// Files changed since last heartbeat (from git or file watcher).
    pub file_changes: Vec<String>,
    /// Latest CI run status (if `gh` is available).
    pub ci_status: Option<CiStatus>,
    /// Alerts from watched PR heartbeat checks.
    pub pr_alerts: Vec<String>,
    /// Incomplete task items.
    pub pending_tasks: Vec<String>,
    /// New LSP diagnostics (errors/warnings).
    pub lsp_diagnostics: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum CiStatus {
    Success,
    Failure { summary: String },
    InProgress,
    Unknown,
}

impl std::fmt::Display for CiStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CiStatus::Success => write!(f, "CI: ✓ passing"),
            CiStatus::Failure { summary } => write!(f, "CI: ✗ failing — {}", summary),
            CiStatus::InProgress => write!(f, "CI: ⏳ in progress"),
            CiStatus::Unknown => write!(f, "CI: unknown"),
        }
    }
}

impl ProactiveContext {
    /// Returns true if there is nothing interesting to report.
    pub fn is_empty(&self) -> bool {
        self.last_error.is_none()
            && self.file_changes.is_empty()
            && self.ci_status.is_none()
            && self.pr_alerts.is_empty()
            && self.pending_tasks.is_empty()
            && self.lsp_diagnostics.is_empty()
    }

    /// Format the context into a human-readable block for the heartbeat prompt.
    pub fn format(&self) -> String {
        let mut parts = Vec::new();

        if let Some(ref err) = self.last_error {
            parts.push(format!("## Last Error\n{}", err));
        }

        if !self.file_changes.is_empty() {
            let files = self
                .file_changes
                .iter()
                .take(20)
                .map(|f| format!("  - {}", f))
                .collect::<Vec<_>>()
                .join("\n");
            let suffix = if self.file_changes.len() > 20 {
                format!("\n  ... and {} more", self.file_changes.len() - 20)
            } else {
                String::new()
            };
            parts.push(format!("## File Changes\n{}{}", files, suffix));
        }

        if let Some(ref ci) = self.ci_status {
            parts.push(format!("## CI Status\n{}", ci));
        }

        if !self.pr_alerts.is_empty() {
            let alerts = self
                .pr_alerts
                .iter()
                .map(|a| format!("  - {}", a.replace('\n', " ")))
                .collect::<Vec<_>>()
                .join("\n");
            parts.push(format!("## Watched PR Alerts\n{}", alerts));
        }

        if !self.pending_tasks.is_empty() {
            let tasks = self
                .pending_tasks
                .iter()
                .map(|t| format!("  - {}", t))
                .collect::<Vec<_>>()
                .join("\n");
            parts.push(format!("## Pending Tasks\n{}", tasks));
        }

        if !self.lsp_diagnostics.is_empty() {
            let diags = self
                .lsp_diagnostics
                .iter()
                .take(10)
                .map(|d| format!("  - {}", d))
                .collect::<Vec<_>>()
                .join("\n");
            parts.push(format!("## LSP Diagnostics\n{}", diags));
        }

        if parts.is_empty() {
            "No notable changes.".to_string()
        } else {
            parts.join("\n\n")
        }
    }
}

// ---------------------------------------------------------------------------
// ProactiveAgent
// ---------------------------------------------------------------------------

/// Global flag: whether the proactive agent is enabled for the current session.
static PROACTIVE_ENABLED: AtomicBool = AtomicBool::new(false);

/// Shared state for the proactive agent visible to slash commands.
static PROACTIVE_STATE: once_cell::sync::Lazy<Mutex<ProactiveState>> =
    once_cell::sync::Lazy::new(|| Mutex::new(ProactiveState::default()));

#[derive(Debug, Default)]
pub struct ProactiveState {
    pub heartbeats: u64,
    pub actions_taken: u64,
    pub last_summary: Option<String>,
    pub interval_secs: u64,
}

/// Check if the proactive agent is currently enabled.
pub fn is_enabled() -> bool {
    PROACTIVE_ENABLED.load(Ordering::Relaxed)
}

/// Returns true when proactive support is compiled in.
pub fn is_supported() -> bool {
    cfg!(feature = "proactive")
}

/// Enable or disable the proactive agent.
pub fn set_enabled(enabled: bool) {
    PROACTIVE_ENABLED.store(enabled, Ordering::Relaxed);
    info!(enabled, "Proactive agent toggled");
}

/// Get a snapshot of the proactive agent's state for /proactive status.
pub fn get_state() -> (bool, u64, u64, Option<String>, u64) {
    let state = PROACTIVE_STATE.lock();
    (
        is_enabled(),
        state.heartbeats,
        state.actions_taken,
        state.last_summary.clone(),
        state.interval_secs,
    )
}

pub struct ProactiveAgent {
    pub enabled: bool,
    pub interval: Duration,
    pub heartbeat_prompt: String,
    pub working_dir: PathBuf,
    pub session_id: String,
    pub last_context: ProactiveContext,
    file_changes: Arc<Mutex<BTreeSet<String>>>,
}

impl Default for ProactiveAgent {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(30),
            heartbeat_prompt: DEFAULT_HEARTBEAT_PROMPT.to_string(),
            working_dir: PathBuf::from("."),
            session_id: String::new(),
            last_context: ProactiveContext::default(),
            file_changes: Arc::new(Mutex::new(BTreeSet::new())),
        }
    }
}

const DEFAULT_HEARTBEAT_PROMPT: &str = "\
You are a proactive monitoring agent running in the background. \
Your job is to observe the current project state and report anything \
the user should know about — build failures, new errors, stale tasks, \
or suspicious file changes. \
\n\n\
You have READ-ONLY access. Do NOT attempt to write, edit, or create files. \
Do NOT spawn sub-agents. \
\n\n\
If there is something worth reporting, provide a concise summary (1-3 sentences). \
If everything looks fine, respond with exactly: IDLE";

impl ProactiveAgent {
    pub fn new(working_dir: PathBuf, session_id: String) -> Self {
        Self {
            working_dir,
            session_id,
            ..Default::default()
        }
    }

    /// Start the proactive monitoring loop as a background tokio task.
    ///
    /// Returns immediately. Cancel via the CancellationToken.
    pub fn start(
        mut self,
        client: Arc<mangocode_api::AnthropicClient>,
        tools: Arc<Vec<Box<dyn Tool>>>,
        tool_ctx: ToolContext,
        query_config: QueryConfig,
        cancel: CancellationToken,
    ) -> JoinHandle<()> {
        // Store interval in shared state.
        {
            let mut state = PROACTIVE_STATE.lock();
            state.interval_secs = self.interval.as_secs();
        }

        if self.session_id.is_empty() {
            self.session_id = tool_ctx.session_id.clone();
        }

        tokio::spawn(async move {
            self.run_loop(client, tools, tool_ctx, query_config, cancel)
                .await;
        })
    }

    async fn run_loop(
        mut self,
        client: Arc<mangocode_api::AnthropicClient>,
        _all_tools: Arc<Vec<Box<dyn Tool>>>,
        tool_ctx: ToolContext,
        query_config: QueryConfig,
        cancel: CancellationToken,
    ) {
        if !cfg!(feature = "proactive") {
            info!("Proactive agent feature disabled at build time");
            return;
        }

        info!(
            interval_secs = self.interval.as_secs(),
            "Proactive agent started"
        );

        let _watcher = start_file_watcher(&self.working_dir, self.file_changes.clone());

        // Build a fresh read-only tool set from the built-in registry,
        // with Bash wrapped to enforce read-only execution.
        let allowed_tools: Arc<Vec<Box<dyn Tool>>> = Arc::new(build_proactive_tools());

        loop {
            // Sleep for the configured interval.
            tokio::select! {
                _ = tokio::time::sleep(self.interval) => {}
                _ = cancel.cancelled() => {
                    info!("Proactive agent stopped");
                    return;
                }
            }

            // User opt-in is required every session.
            if !self.enabled || !is_enabled() {
                debug!("Proactive agent tick — disabled, skipping");
                continue;
            }

            debug!("Proactive agent heartbeat");

            // Gather context.
            let context = self.gather_context(&tool_ctx.config).await;
            if context.is_empty() {
                debug!("Proactive agent: no notable changes, skipping heartbeat");
                {
                    let mut state = PROACTIVE_STATE.lock();
                    state.heartbeats += 1;
                }
                self.last_context = context;
                continue;
            }

            // Build the heartbeat prompt.
            let prompt = format!(
                "{}\n\nCurrent project state:\n{}\n\n\
                 If there's something worth reporting to the user, provide a concise summary. \
                 If not, respond with IDLE.",
                self.heartbeat_prompt,
                context.format()
            );

            let mut messages = vec![Message::user(prompt)];
            let cost_tracker = tool_ctx.cost_tracker.clone();
            let cancel_child = cancel.child_token();

            // Run a mini query loop with restricted tools.
            let outcome = run_query_loop(
                client.as_ref(),
                &mut messages,
                &allowed_tools,
                &tool_ctx,
                &query_config,
                cost_tracker,
                None,  // no UI event channel
                cancel_child,
                None,  // no pending message queue
            )
            .await;

            // Update state.
            {
                let mut state = PROACTIVE_STATE.lock();
                state.heartbeats += 1;
            }

            match outcome {
                QueryOutcome::EndTurn { message, .. } => {
                    let text = message.get_all_text();
                    if text.contains("IDLE") {
                        debug!("Proactive agent: IDLE");
                        continue;
                    }

                    info!("Proactive agent report: {}", &text);

                    // Update state with the summary.
                    {
                        let mut state = PROACTIVE_STATE.lock();
                        state.actions_taken += 1;
                        state.last_summary = Some(text.clone());
                    }

                    // Fire a desktop notification if available.
                    send_notification("MangoCode — Proactive Agent", &text);
                    self.last_context = context;
                }
                QueryOutcome::Error(e) => {
                    warn!("Proactive agent heartbeat error: {}", e);
                }
                QueryOutcome::Cancelled => {
                    info!("Proactive agent heartbeat cancelled");
                    return;
                }
                _ => {
                    debug!("Proactive agent heartbeat ended with non-EndTurn outcome");
                }
            }
        }
    }

    /// Gather context about the current project state.
    async fn gather_context(&self, config: &mangocode_core::config::Config) -> ProactiveContext {
        ProactiveContext {
            file_changes: self.take_file_changes(),
            ci_status: gather_ci_status(&self.working_dir).await,
            pending_tasks: gather_pending_tasks(&self.session_id),
            pr_alerts: mangocode_tools::heartbeat_scan_watched_prs(&self.working_dir, config)
                .await,
            lsp_diagnostics: gather_new_lsp_diagnostics(&self.last_context).await,
            last_error: gather_last_error(&self.session_id),
        }
    }

    fn take_file_changes(&self) -> Vec<String> {
        let mut guard = self.file_changes.lock();
        if guard.is_empty() {
            return Vec::new();
        }
        let items = guard.iter().cloned().collect::<Vec<_>>();
        guard.clear();
        items
    }
}

// ---------------------------------------------------------------------------
// Context gathering helpers
// ---------------------------------------------------------------------------

fn start_file_watcher(
    working_dir: &Path,
    file_changes: Arc<Mutex<BTreeSet<String>>>,
) -> Option<RecommendedWatcher> {
    let root = working_dir.to_path_buf();
    let mut watcher = match notify::recommended_watcher(move |event: notify::Result<Event>| {
        if let Ok(evt) = event {
            let mut set = file_changes.lock();
            for path in evt.paths {
                if let Ok(rel) = path.strip_prefix(&root) {
                    set.insert(rel.to_string_lossy().replace('\\', "/"));
                } else {
                    set.insert(path.to_string_lossy().replace('\\', "/"));
                }
            }
        }
    }) {
        Ok(w) => w,
        Err(e) => {
            warn!(error = %e, "Proactive watcher unavailable");
            return None;
        }
    };

    if let Err(e) = watcher.watch(working_dir, RecursiveMode::Recursive) {
        warn!(error = %e, "Failed to watch project directory");
        return None;
    }

    Some(watcher)
}

async fn gather_ci_status(working_dir: &PathBuf) -> Option<CiStatus> {
    let output = tokio::process::Command::new("gh")
        .args(["run", "list", "--limit", "1", "--json", "status,conclusion,name"])
        .current_dir(working_dir)
        .output()
        .await;

    match output {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            let parsed: Result<Vec<serde_json::Value>, _> = serde_json::from_str(&text);
            match parsed {
                Ok(runs) if !runs.is_empty() => {
                    let run = &runs[0];
                    let status = run["status"].as_str().unwrap_or("");
                    let conclusion = run["conclusion"].as_str().unwrap_or("");
                    let name = run["name"].as_str().unwrap_or("CI");

                    match (status, conclusion) {
                        ("completed", "success") => Some(CiStatus::Success),
                        ("completed", "failure") => Some(CiStatus::Failure {
                            summary: format!("{} failed", name),
                        }),
                        ("in_progress", _) | ("queued", _) | ("waiting", _) => {
                            Some(CiStatus::InProgress)
                        }
                        _ => Some(CiStatus::Unknown),
                    }
                }
                _ => None,
            }
        }
        _ => None, // `gh` not available — skip CI checking
    }
}

fn gather_pending_tasks(session_id: &str) -> Vec<String> {
    load_todos(session_id)
        .into_iter()
        .filter_map(|item| {
            let status = item.get("status").and_then(|s| s.as_str())?;
            if status == "completed" {
                return None;
            }
            let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("?");
            let content = item.get("content").and_then(|v| v.as_str()).unwrap_or("(untitled)");
            Some(format!("{} ({})", content, id))
        })
        .collect()
}

async fn gather_new_lsp_diagnostics(last_context: &ProactiveContext) -> Vec<String> {
    let mgr = global_lsp_manager();
    let guard = mgr.lock().await;
    let all = guard.all_diagnostics();

    let current: BTreeSet<String> = all.iter().map(format_diag).collect();
    let previous: BTreeSet<String> = last_context.lsp_diagnostics.iter().cloned().collect();

    current.difference(&previous).cloned().collect()
}

fn format_diag(diag: &LspDiagnostic) -> String {
    let severity = format!("{:?}", diag.severity);
    let code = diag.code.clone().unwrap_or_default();
    if code.is_empty() {
        format!(
            "{}:{}:{} [{}] {}",
            diag.file, diag.line, diag.column, severity, diag.message
        )
    } else {
        format!(
            "{}:{}:{} [{}:{}] {}",
            diag.file, diag.line, diag.column, severity, code, diag.message
        )
    }
}

fn gather_last_error(session_id: &str) -> Option<String> {
    let candidates = [
        dirs::home_dir()?
            .join(".mangocode")
            .join("sessions")
            .join(format!("{}.log", session_id)),
        dirs::home_dir()?
            .join(".mangocode")
            .join("logs")
            .join("session.log"),
    ];

    for candidate in candidates {
        if let Ok(text) = std::fs::read_to_string(&candidate) {
            if let Some(line) = text
                .lines()
                .rev()
                .find(|l| l.to_ascii_lowercase().contains("error"))
            {
                return Some(line.trim().to_string());
            }
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Desktop notification helper
// ---------------------------------------------------------------------------

/// Send a desktop notification. Best-effort — silently ignores failures.
fn send_notification(title: &str, body: &str) {
    // Truncate body for notification readability.
    let body_truncated = mangocode_core::truncate::truncate_bytes_with_ellipsis(body, 200);

    #[cfg(target_os = "windows")]
    {
        // Use PowerShell toast notification on Windows.
        let ps_script = format!(
            "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null; \
             $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02); \
             $textNodes = $template.GetElementsByTagName('text'); \
             $textNodes.Item(0).AppendChild($template.CreateTextNode('{}')) > $null; \
             $textNodes.Item(1).AppendChild($template.CreateTextNode('{}')) > $null; \
             $toast = [Windows.UI.Notifications.ToastNotification]::new($template); \
             [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('MangoCode').Show($toast)",
            title.replace('\'', "''"),
            body_truncated.replace('\'', "''").replace('\n', " "),
        );
        let _ = std::process::Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", &ps_script])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
    }

    #[cfg(target_os = "macos")]
    {
        let escaped_title = title.replace('"', r#"\""#);
        let escaped_body = body_truncated.replace('"', r#"\""#).replace('\n', " ");
        let script = format!(
            "display notification \"{}\" with title \"{}\"",
            escaped_body, escaped_title
        );
        let _ = std::process::Command::new("osascript")
            .args(["-e", &script])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
    }

    #[cfg(target_os = "linux")]
    {
        let _ = std::process::Command::new("notify-send")
            .args([title, &body_truncated])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
    }
}

// ---------------------------------------------------------------------------
// Tool filtering helper
// ---------------------------------------------------------------------------

/// Build a fresh read-only tool set for the proactive agent.
///
/// Calls `mangocode_tools::all_tools()` and filters to the allowed set.
pub fn build_proactive_tools() -> Vec<Box<dyn Tool>> {
    mangocode_tools::all_tools()
        .into_iter()
        .filter_map(|t| {
            let name = t.name();
            if !ALLOWED_TOOLS.contains(&name) || DENIED_TOOLS.contains(&name) {
                return None;
            }

            if name == "Bash" {
                return Some(Box::new(ReadOnlyBashTool { inner: t }) as Box<dyn Tool>);
            }

            Some(t)
        })
        .collect()
}

struct ReadOnlyBashTool {
    inner: Box<dyn Tool>,
}

impl ReadOnlyBashTool {
    fn args_contain_any(args: &[&str], needles: &[&str]) -> bool {
        args.iter().any(|arg| needles.iter().any(|needle| arg == needle))
    }

    fn args_start_with_any(args: &[&str], prefixes: &[&str]) -> bool {
        args.iter()
            .any(|arg| prefixes.iter().any(|prefix| arg.starts_with(prefix)))
    }

    fn command_is_allowed(segment: &str) -> bool {
        let mut parts = segment.split_whitespace();
        let Some(cmd) = parts.next() else {
            return false;
        };

        let args: Vec<&str> = parts.collect();

        match cmd {
            "ls" | "cat" | "head" | "tail" | "grep" | "rg" | "pwd" | "wc" | "stat"
            | "nl" => true,
            "find" => {
                // `find` can mutate via actions like -delete / -exec / -fprint.
                let forbidden_exact = ["-delete", "-ok", "-okdir", "-fls", "-ls"];
                let forbidden_prefixes = ["-exec", "-fprint", "-fprintf"];
                !Self::args_contain_any(&args, &forbidden_exact)
                    && !Self::args_start_with_any(&args, &forbidden_prefixes)
            }
            "git" => {
                let Some(sub) = args.first().copied() else {
                    return false;
                };

                // Keep to read-only git plumbing/inspection.
                if !matches!(sub, "status" | "diff" | "log" | "show" | "rev-parse") {
                    return false;
                }

                // Some read-ish git subcommands can still write with explicit output flags.
                let tail = &args[1..];
                let forbidden_prefixes = ["--output", "--output="];
                !Self::args_contain_any(tail, &["-o"])
                    && !Self::args_start_with_any(tail, &forbidden_prefixes)
            }
            _ => false,
        }
    }

    fn is_read_only(command: &str) -> bool {
        let lowered = command.to_ascii_lowercase();
        let trimmed = lowered.trim();

        if trimmed.is_empty() {
            return false;
        }

        // Disallow shell control/redirect primitives outright.
        let forbidden_fragments = [
            ";", "&&", "||", "&", "`", "$(", ">", "<", "\n", "\r", "\\\n",
        ];
        if forbidden_fragments.iter().any(|f| trimmed.contains(f)) {
            return false;
        }

        // Allow safe pipelines only when each stage is an approved read command.
        for segment in trimmed.split('|') {
            if !Self::command_is_allowed(segment.trim()) {
                return false;
            }
        }

        true
    }
}

#[async_trait]
impl Tool for ReadOnlyBashTool {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    fn input_schema(&self) -> Value {
        self.inner.input_schema()
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let command = input
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        if !Self::is_read_only(command) {
            return ToolResult::error(
                "Proactive Bash access is read-only. Write or mutating commands are blocked.",
            );
        }

        self.inner.execute(input, ctx).await
    }
}

#[cfg(test)]
mod tests {
    use super::ReadOnlyBashTool;

    #[test]
    fn read_only_blocks_leading_mutations() {
        assert!(!ReadOnlyBashTool::is_read_only("mv a b"));
        assert!(!ReadOnlyBashTool::is_read_only("cp a b"));
        assert!(!ReadOnlyBashTool::is_read_only("mkdir out"));
    }

    #[test]
    fn read_only_allows_safe_reads() {
        assert!(ReadOnlyBashTool::is_read_only("ls -la"));
        assert!(ReadOnlyBashTool::is_read_only("cat README.md"));
        assert!(ReadOnlyBashTool::is_read_only("git status"));
        assert!(ReadOnlyBashTool::is_read_only("find . -name '*.rs'"));
        assert!(ReadOnlyBashTool::is_read_only("rg todo src | head -n 5"));
    }

    #[test]
    fn read_only_blocks_common_bypasses() {
        assert!(!ReadOnlyBashTool::is_read_only("python -c \"open('x','w').write('1')\""));
        assert!(!ReadOnlyBashTool::is_read_only("git commit -m test"));
        assert!(!ReadOnlyBashTool::is_read_only("git remote add origin https://example.invalid/repo.git"));
        assert!(!ReadOnlyBashTool::is_read_only("git diff --output=/tmp/patch.diff"));
        assert!(!ReadOnlyBashTool::is_read_only("git branch -D old-branch"));
        assert!(!ReadOnlyBashTool::is_read_only("find . -delete"));
        assert!(!ReadOnlyBashTool::is_read_only("find . -exec rm {} +"));
        assert!(!ReadOnlyBashTool::is_read_only("ls; rm -rf /tmp/x"));
        assert!(!ReadOnlyBashTool::is_read_only("ls & rm -rf /tmp/x"));
        assert!(!ReadOnlyBashTool::is_read_only("cat a.txt > b.txt"));
    }
}
