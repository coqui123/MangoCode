// mangocode CLI entry point
//
// This is the main binary for MangoCode. It:
// 1. Parses CLI arguments with clap (mirrors cli.tsx + main.tsx flags)
// 2. Loads configuration from settings.json + env vars
// 3. Builds system/user context (git status, AGENTS.md)
// 4. Runs in either:
//    - Headless (--print / -p) mode: single query, output to stdout
//    - Interactive REPL mode: full TUI with ratatui

mod codex_oauth_flow;
mod oauth_flow;

// ---------------------------------------------------------------------------
// Build-time metadata (embedded via build.rs)
// ---------------------------------------------------------------------------

/// Build timestamp in RFC 3339 format
pub const BUILD_TIME: &str = env!("BUILD_TIME");

/// Short git commit hash (or "unknown" if not a git repo)
pub const GIT_COMMIT: &str = env!("GIT_COMMIT");

/// Full version string: "0.0.7 (abc1234)"
const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (", env!("GIT_COMMIT"), ")");

/// Package/distribution identifier
pub const PACKAGE_URL: &str = env!("PACKAGE_URL");

/// Feedback/issue reporting channel
pub const FEEDBACK_CHANNEL: &str = env!("FEEDBACK_CHANNEL");

/// Explanation of issue routing in this build
pub const ISSUES_EXPLAINER: &str = env!("ISSUES_EXPLAINER");

use anyhow::Context;
use async_trait::async_trait;
use clap::{ArgAction, Parser, ValueEnum};
use mangocode_core::types::ToolDefinition;
use mangocode_core::{
    config::{Config, HookEntry, HookEvent, McpServerConfig, PermissionMode, Settings},
    context::ContextBuilder,
    cost::CostTracker,
    permissions::{AutoPermissionHandler, InteractivePermissionHandler},
};
use mangocode_tools::{PermissionLevel, Tool, ToolContext, ToolResult};
use parking_lot::Mutex as ParkingMutex;
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::Arc,
};
use tracing::{debug, info, warn};
use tracing_subscriber::EnvFilter;

// ---------------------------------------------------------------------------
// MCP tool wrapper: makes MCP server tools look like native cc-tools.
// ---------------------------------------------------------------------------

struct McpToolWrapper {
    tool_def: ToolDefinition,
    server_name: String,
    manager_tool_name: String,
    manager: Arc<mangocode_mcp::McpManager>,
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
        // MCP tools run external processes – treat as Execute.
        PermissionLevel::Execute
    }

    fn input_schema(&self) -> serde_json::Value {
        self.tool_def.input_schema.clone()
    }

    async fn execute(&self, input: serde_json::Value, _ctx: &ToolContext) -> ToolResult {
        // Strip the server-name prefix to get the bare tool name.
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

// ---------------------------------------------------------------------------
// CLI argument definition (matches TypeScript main.tsx flags)
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "mangocode",
    version = VERSION,
    about = "MangoCode - AI-powered coding assistant",
    long_about = None,
)]
struct Cli {
    /// Initial prompt to send (enables headless/print mode)
    prompt: Option<String>,

    /// Print mode: send prompt and exit (non-interactive)
    #[arg(short = 'p', long = "print", action = ArgAction::SetTrue)]
    print: bool,

    /// Model to use (defaults to provider-appropriate model if not set)
    #[arg(short = 'm', long = "model")]
    model: Option<String>,

    /// Permission mode
    #[arg(long = "permission-mode", value_enum, default_value_t = CliPermissionMode::Default)]
    permission_mode: CliPermissionMode,

    /// Resume a previous session by ID
    #[arg(long = "resume")]
    resume: Option<String>,

    /// Maximum number of agentic turns
    #[arg(long = "max-turns", default_value_t = 10)]
    max_turns: u32,

    /// Custom system prompt
    #[arg(long = "system-prompt", short = 's')]
    system_prompt: Option<String>,

    /// Append to system prompt
    #[arg(long = "append-system-prompt")]
    append_system_prompt: Option<String>,

    /// Disable AGENTS.md memory files
    #[arg(long = "no-claude-md", action = ArgAction::SetTrue)]
    no_claude_md: bool,

    /// Output format
    #[arg(long = "output-format", value_enum, default_value_t = CliOutputFormat::Text)]
    output_format: CliOutputFormat,

    /// Enable verbose logging
    #[arg(long = "verbose", short = 'v', action = ArgAction::SetTrue)]
    verbose: bool,

    /// API key (overrides ANTHROPIC_API_KEY env var)
    #[arg(long = "api-key")]
    api_key: Option<String>,

    /// Maximum tokens per response
    #[arg(long = "max-tokens")]
    max_tokens: Option<u32>,

    /// Working directory
    #[arg(long = "cwd")]
    cwd: Option<PathBuf>,

    /// Bypass all permission checks (danger!)
    #[arg(long = "dangerously-skip-permissions", action = ArgAction::SetTrue)]
    dangerously_skip_permissions: bool,

    /// Dump the system prompt to stdout and exit
    #[arg(long = "dump-system-prompt", action = ArgAction::SetTrue, hide = true)]
    dump_system_prompt: bool,

    /// List available models and exit
    #[arg(long = "list-models", action = ArgAction::SetTrue)]
    list_models: bool,

    /// MCP config JSON string (inline server definitions)
    #[arg(long = "mcp-config")]
    mcp_config: Option<String>,

    /// Claude-compatible settings override (JSON string or path to JSON file)
    #[arg(long = "settings")]
    settings_override: Option<String>,

    /// Disable auto-compaction
    #[arg(long = "no-auto-compact", action = ArgAction::SetTrue)]
    no_auto_compact: bool,

    /// Grant MangoCode access to an additional directory (can be repeated)
    #[arg(long = "add-dir", value_name = "DIR", action = ArgAction::Append)]
    add_dir: Vec<PathBuf>,

    /// Input format for --print mode (text or stream-json)
    #[arg(long = "input-format", value_enum, default_value_t = CliInputFormat::Text)]
    input_format: CliInputFormat,

    /// Session ID to tag this headless run (for tracking in logs/hooks)
    #[arg(long = "session-id")]
    session_id_flag: Option<String>,

    /// Prefill the first assistant turn with this text
    #[arg(long = "prefill")]
    prefill: Option<String>,

    /// Effort level for extended thinking (low, medium, high, max)
    #[arg(long = "effort", value_name = "LEVEL")]
    effort: Option<String>,

    /// Extended thinking budget in tokens (enables extended thinking)
    #[arg(long = "thinking", value_name = "TOKENS")]
    thinking: Option<u32>,

    /// Continue the most recent conversation
    #[arg(short = 'c', long = "continue", action = ArgAction::SetTrue)]
    continue_session: bool,

    /// Override system prompt from a file
    #[arg(long = "system-prompt-file")]
    system_prompt_file: Option<PathBuf>,

    /// Tools to allow (comma- or space-separated, default: all)
    #[arg(
        long = "allowed-tools",
        alias = "allowedTools",
        value_name = "TOOLS",
        value_delimiter = ',',
        num_args = 1..
    )]
    allowed_tools: Vec<String>,

    /// Tools to disallow (comma- or space-separated)
    #[arg(
        long = "disallowed-tools",
        alias = "disallowedTools",
        value_name = "TOOLS",
        value_delimiter = ',',
        num_args = 1..
    )]
    disallowed_tools: Vec<String>,

    /// Extra beta feature headers to send (comma-separated)
    #[arg(long = "betas", value_name = "HEADERS")]
    betas: Option<String>,

    /// Disable all slash commands
    #[arg(long = "disable-slash-commands", action = ArgAction::SetTrue)]
    disable_slash_commands: bool,

    /// Run in bare mode (no hooks, no plugins, no AGENTS.md)
    #[arg(long = "bare", action = ArgAction::SetTrue)]
    bare: bool,

    /// Billing workload tag
    #[arg(long = "workload", value_name = "TAG")]
    workload: Option<String>,

    /// Maximum spend in USD before aborting the query loop
    #[arg(long = "max-budget-usd", value_name = "USD")]
    max_budget_usd: Option<f64>,

    /// Fallback model to use if the primary model is overloaded or unavailable
    #[arg(long = "fallback-model")]
    fallback_model: Option<String>,

    /// LLM provider to use (default: anthropic). Examples: openai, google, ollama
    #[arg(long, env = "MANGOCODE_PROVIDER")]
    provider: Option<String>,

    /// Override the API base URL for the selected provider
    #[arg(long, env = "MANGOCODE_API_BASE")]
    api_base: Option<String>,

    /// Named agent to use (e.g., build, plan, explore)
    #[arg(long, short = 'A')]
    agent: Option<String>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum CliPermissionMode {
    Default,
    AcceptEdits,
    BypassPermissions,
    Plan,
}

impl From<CliPermissionMode> for PermissionMode {
    fn from(m: CliPermissionMode) -> Self {
        match m {
            CliPermissionMode::Default => PermissionMode::Default,
            CliPermissionMode::AcceptEdits => PermissionMode::AcceptEdits,
            CliPermissionMode::BypassPermissions => PermissionMode::BypassPermissions,
            CliPermissionMode::Plan => PermissionMode::Plan,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum CliOutputFormat {
    Text,
    Json,
    #[value(name = "stream-json")]
    StreamJson,
}

impl From<CliOutputFormat> for mangocode_core::config::OutputFormat {
    fn from(f: CliOutputFormat) -> Self {
        match f {
            CliOutputFormat::Text => mangocode_core::config::OutputFormat::Text,
            CliOutputFormat::Json => mangocode_core::config::OutputFormat::Json,
            CliOutputFormat::StreamJson => mangocode_core::config::OutputFormat::StreamJson,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum CliInputFormat {
    /// Plain text prompt (default)
    Text,
    /// Newline-delimited JSON messages — each line is {"role":"user"|"assistant","content":"..."}
    #[value(name = "stream-json")]
    StreamJson,
}

fn resolve_bridge_config(
    settings: &Settings,
    auth_credential: &str,
    use_bearer_auth: bool,
    is_headless: bool,
) -> Option<mangocode_bridge::BridgeConfig> {
    if is_headless {
        return None;
    }

    let mut bridge_config = mangocode_bridge::BridgeConfig::from_env();

    if settings.remote_control_at_startup {
        bridge_config.enabled = true;
    }

    if bridge_config.session_token.is_none() && use_bearer_auth && !auth_credential.is_empty() {
        bridge_config.session_token = Some(auth_credential.to_string());
    }

    bridge_config.is_active().then_some(bridge_config)
}

fn normalize_tool_list(raw_values: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();

    for raw in raw_values {
        for token in raw.split(|c: char| c == ',' || c.is_whitespace()) {
            let t = token.trim();
            if t.is_empty() {
                continue;
            }
            if seen.insert(t.to_string()) {
                out.push(t.to_string());
            }
        }
    }

    out
}

fn parse_hook_event(name: &str) -> Option<HookEvent> {
    let normalized = name.replace(['-', '_'], "").to_ascii_lowercase();
    match normalized.as_str() {
        "pretooluse" => Some(HookEvent::PreToolUse),
        "posttooluse" => Some(HookEvent::PostToolUse),
        "postmodelturn" => Some(HookEvent::PostModelTurn),
        "userpromptsubmit" => Some(HookEvent::UserPromptSubmit),
        "notification" => Some(HookEvent::Notification),
        "stop" => Some(HookEvent::Stop),
        _ => None,
    }
}

fn parse_hook_entries(value: &serde_json::Value) -> Vec<HookEntry> {
    let mut entries = Vec::new();

    match value {
        serde_json::Value::String(command) => {
            if !command.trim().is_empty() {
                entries.push(HookEntry {
                    command: command.trim().to_string(),
                    tool_filter: None,
                    blocking: true,
                });
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                entries.extend(parse_hook_entries(item));
            }
        }
        serde_json::Value::Object(obj) => {
            if let Some(nested) = obj.get("hooks") {
                entries.extend(parse_hook_entries(nested));
            }

            if let Some(cmd) = obj.get("command").and_then(|v| v.as_str()) {
                let command = cmd.trim();
                if !command.is_empty() {
                    entries.push(HookEntry {
                        command: command.to_string(),
                        tool_filter: obj
                            .get("tool_filter")
                            .or_else(|| obj.get("toolFilter"))
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string()),
                        blocking: obj
                            .get("blocking")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(true),
                    });
                }
            }
        }
        _ => {}
    }

    entries
}

fn parse_mcp_server_from_value(name: &str, value: &serde_json::Value) -> Option<McpServerConfig> {
    let obj = value.as_object()?;
    let command = obj
        .get("command")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let args = obj
        .get("args")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let env = obj
        .get("env")
        .and_then(|v| v.as_object())
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect::<HashMap<_, _>>()
        })
        .unwrap_or_default();
    let url = obj
        .get("url")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let server_type = obj
        .get("type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            if url.is_some() {
                "sse".to_string()
            } else {
                "stdio".to_string()
            }
        });

    Some(McpServerConfig {
        name: name.to_string(),
        command,
        args,
        env,
        url,
        server_type,
    })
}

fn merge_mcp_servers(config: &mut Config, servers: Vec<McpServerConfig>) {
    for server in servers {
        if let Some(existing) = config
            .mcp_servers
            .iter_mut()
            .find(|s| s.name == server.name)
        {
            *existing = server;
        } else {
            config.mcp_servers.push(server);
        }
    }
}

fn apply_settings_override(config: &mut Config, raw: &str) -> anyhow::Result<()> {
    let trimmed = raw.trim();
    let value: serde_json::Value = if trimmed.starts_with('{') || trimmed.starts_with('[') {
        serde_json::from_str(trimmed)
            .map_err(|e| anyhow::anyhow!("Invalid --settings JSON: {e}"))?
    } else {
        let content = std::fs::read_to_string(trimmed)
            .map_err(|e| anyhow::anyhow!("Failed to read --settings file '{trimmed}': {e}"))?;
        let content = content.trim_start_matches('\u{feff}');
        serde_json::from_str(content)
            .map_err(|e| anyhow::anyhow!("Invalid JSON in --settings file '{trimmed}': {e}"))?
    };

    let hooks_value = value
        .get("hooks")
        .or_else(|| value.get("config").and_then(|cfg| cfg.get("hooks")));
    if let Some(hooks_obj) = hooks_value.and_then(|v| v.as_object()) {
        for (event_name, entries_value) in hooks_obj {
            if let Some(event) = parse_hook_event(event_name) {
                let entries = parse_hook_entries(entries_value);
                if !entries.is_empty() {
                    config.hooks.insert(event, entries);
                }
            }
        }
    }

    Ok(())
}

fn apply_mcp_config_override(config: &mut Config, raw: &str) -> anyhow::Result<()> {
    let trimmed = raw.trim();
    let value: serde_json::Value = if trimmed.starts_with('{') || trimmed.starts_with('[') {
        serde_json::from_str(trimmed)
            .map_err(|e| anyhow::anyhow!("Invalid --mcp-config JSON: {e}"))?
    } else {
        let content = std::fs::read_to_string(trimmed)
            .map_err(|e| anyhow::anyhow!("Failed to read --mcp-config file '{trimmed}': {e}"))?;
        let content = content.trim_start_matches('\u{feff}');
        serde_json::from_str(content)
            .map_err(|e| anyhow::anyhow!("Invalid JSON in --mcp-config file '{trimmed}': {e}"))?
    };

    let mut parsed = Vec::new();

    if let Some(obj) = value.get("mcpServers").and_then(|v| v.as_object()) {
        for (name, server_value) in obj {
            if let Some(server) = parse_mcp_server_from_value(name, server_value) {
                parsed.push(server);
            }
        }
    } else if let Some(arr) = value.as_array() {
        for item in arr {
            if let Ok(server) = serde_json::from_value::<McpServerConfig>(item.clone()) {
                parsed.push(server);
            }
        }
    }

    if parsed.is_empty() {
        return Err(anyhow::anyhow!(
            "--mcp-config did not contain any valid MCP servers"
        ));
    }

    merge_mcp_servers(config, parsed);
    Ok(())
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Fast-path: handle --version before parsing everything
    let raw_args: Vec<String> = std::env::args().collect();
    if raw_args.iter().any(|a| a == "--version" || a == "-V") {
        println!("MangoCode v{}", VERSION);
        return Ok(());
    }

    // Fast-path: `claude auth <login|logout|status>` — mirrors TypeScript cli.tsx pattern
    if raw_args.get(1).map(|s| s.as_str()) == Some("auth") {
        return handle_auth_command(&raw_args[2..]).await;
    }

    // Fast-path: `claude acp` — start the Agent Client Protocol stdio server.
    if raw_args.get(1).map(|s| s.as_str()) == Some("acp") {
        return mangocode_acp::run_acp_server().await;
    }

    // Fast-path: `claude models` — list all available providers and models.
    if raw_args.get(1).map(|s| s.as_str()) == Some("models") {
        let mut registry = mangocode_api::ModelRegistry::new();
        // Load cached models.dev data if available so the list is comprehensive.
        let cache_path = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("mangocode")
            .join("models.json");
        registry.load_cache(&cache_path);
        let mut entries = registry.list_all();
        // Sort by provider then model id for stable output.
        entries.sort_by(|a, b| {
            (*a.info.provider_id)
                .cmp(&*b.info.provider_id)
                .then_with(|| (*a.info.id).cmp(&*b.info.id))
        });
        for entry in entries {
            println!(
                "{}/{} — {} (ctx: {}K, in: ${:.2}/M, out: ${:.2}/M)",
                entry.info.provider_id,
                entry.info.id,
                entry.info.name,
                entry.info.context_window / 1000,
                entry.cost_input.unwrap_or(0.0),
                entry.cost_output.unwrap_or(0.0),
            );
        }
        return Ok(());
    }

    // Fast-path: named commands (`claude agents`, `claude ide`, `claude branch`, …)
    // Check before Cli::parse() so these names don't conflict with positional prompt arg.
    if let Some(cmd_name) = raw_args.get(1).map(|s| s.as_str()) {
        // Only intercept if it looks like a subcommand (no leading `-` or `/`)
        if !cmd_name.starts_with('-') && !cmd_name.starts_with('/') {
            if let Some(named_cmd) =
                mangocode_commands::named_commands::find_named_command(cmd_name)
            {
                // Build a minimal CommandContext (named commands are pre-session)
                let settings = Settings::load().await.unwrap_or_default();
                let config = settings.effective_config();
                let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
                let cmd_ctx = mangocode_commands::CommandContext {
                    config,
                    cost_tracker: CostTracker::new(),
                    session_metrics: None,
                    messages: vec![],
                    working_dir: cwd,
                    session_id: "pre-session".to_string(),
                    session_title: None,
                    remote_session_url: None,
                    mcp_manager: None,
                    model_registry: None,
                };
                // Collect remaining args after the command name
                let rest: Vec<&str> = raw_args[2..].iter().map(|s| s.as_str()).collect();
                let result = named_cmd.execute_named(&rest, &cmd_ctx);
                match result {
                    mangocode_commands::CommandResult::Message(msg)
                    | mangocode_commands::CommandResult::UserMessage(msg) => {
                        println!("{}", msg);
                        std::process::exit(0);
                    }
                    mangocode_commands::CommandResult::Error(e) => {
                        eprintln!("Error: {}", e);
                        eprintln!("Usage: {}", named_cmd.usage());
                        std::process::exit(1);
                    }
                    _ => {
                        // For any other result variant, fall through to normal startup
                    }
                }
                return Ok(());
            }
        }
    }

    let cli = Cli::parse();

    // Setup logging
    let log_level = if cli.verbose { "debug" } else { "warn" };
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level)),
        )
        .with_target(false)
        .with_writer(std::io::stderr)
        .without_time()
        .init();

    // Determine working directory
    let cwd = cli
        .cwd
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    debug!(cwd = %cwd.display(), "Starting MangoCode");

    // Load settings from disk (hierarchical: global < project)
    let settings = Settings::load_hierarchical(&cwd).await;

    // Build effective config (CLI args override settings)
    let mut config = settings.effective_config();
    if let Some(ref key) = cli.api_key {
        config.api_key = Some(key.clone());
    }
    config.model = cli.model.clone();
    if let Some(mt) = cli.max_tokens {
        config.max_tokens = Some(mt);
    }
    config.verbose = cli.verbose;
    config.output_format = cli.output_format.into();
    config.disable_claude_mds = cli.no_claude_md;
    if let Some(sp) = cli.system_prompt.clone() {
        config.custom_system_prompt = Some(sp);
    }
    if let Some(asp) = cli.append_system_prompt.clone() {
        config.append_system_prompt = Some(asp);
    }
    if cli.dangerously_skip_permissions {
        // Mirror TS setup.ts: block bypass mode when running as root/sudo.
        #[cfg(unix)]
        if nix::unistd::Uid::effective().is_root() {
            anyhow::bail!(
                "--dangerously-skip-permissions cannot be used with root/sudo privileges for security reasons"
            );
        }
        config.permission_mode = PermissionMode::BypassPermissions;
    } else {
        config.permission_mode = cli.permission_mode.into();
    }
    config.additional_dirs = cli.add_dir.clone();
    if cli.no_auto_compact {
        config.auto_compact = false;
    }
    config.project_dir = Some(cwd.clone());
    if let Some(p) = &cli.provider {
        config.provider = Some(p.clone());
    }
    if let Some(base) = &cli.api_base {
        // Store in the provider's config entry
        let provider_id = config
            .provider
            .clone()
            .unwrap_or_else(|| "anthropic".to_string());
        config
            .provider_configs
            .entry(provider_id)
            .or_default()
            .api_base = Some(base.clone());
    }
    if !cli.allowed_tools.is_empty() {
        config.allowed_tools = normalize_tool_list(&cli.allowed_tools);
    }
    if !cli.disallowed_tools.is_empty() {
        config.disallowed_tools = normalize_tool_list(&cli.disallowed_tools);
    }
    if let Some(ref settings_override) = cli.settings_override {
        apply_settings_override(&mut config, settings_override)?;
    }
    if let Some(ref mcp_config) = cli.mcp_config {
        apply_mcp_config_override(&mut config, mcp_config)?;
    }

    // --list-models fast path
    if cli.list_models {
        let mut registry = mangocode_api::ModelRegistry::new();
        let cache_path = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("mangocode")
            .join("models.json");
        registry.load_cache(&cache_path);
        let mut entries = registry.list_all();
        entries.sort_by(|a, b| {
            (*a.info.provider_id)
                .cmp(&*b.info.provider_id)
                .then_with(|| (*a.info.id).cmp(&*b.info.id))
        });

        match cli.output_format {
            CliOutputFormat::Json | CliOutputFormat::StreamJson => {
                let mut grouped: serde_json::Map<String, serde_json::Value> =
                    serde_json::Map::new();
                for entry in &entries {
                    let provider = entry.info.provider_id.to_string();
                    let model_obj = serde_json::json!({
                        "id": &*entry.info.id,
                        "name": &entry.info.name,
                        "provider": &provider,
                        "context_window": entry.info.context_window,
                        "max_output_tokens": entry.info.max_output_tokens,
                        "cost_per_million_input_tokens": entry.cost_input,
                        "cost_per_million_output_tokens": entry.cost_output,
                    });
                    grouped
                        .entry(provider)
                        .or_insert_with(|| serde_json::json!([]))
                        .as_array_mut()
                        .unwrap()
                        .push(model_obj);
                }
                println!("{}", serde_json::to_string_pretty(&grouped).unwrap());
            }
            CliOutputFormat::Text => {
                let mut current_provider = String::new();
                for entry in &entries {
                    let provider = entry.info.provider_id.to_string();
                    if provider != current_provider {
                        if !current_provider.is_empty() {
                            println!();
                        }
                        println!("{}:", provider);
                        current_provider = provider;
                    }
                    print!(
                        "  {:<40} {:>6}K ctx  {:>6} max out",
                        &*entry.info.id,
                        entry.info.context_window / 1000,
                        entry.info.max_output_tokens,
                    );
                    if let (Some(ci), Some(co)) = (entry.cost_input, entry.cost_output) {
                        print!("  ${:.2}/${:.2} per 1M tok", ci, co);
                    }
                    println!();
                }
            }
        }
        return Ok(());
    }

    // --dump-system-prompt fast path
    if cli.dump_system_prompt {
        let ctx = ContextBuilder::new(cwd.clone()).disable_claude_mds(config.disable_claude_mds);
        let sys = ctx.build_system_context().await;
        let user = ctx.build_user_context().await;
        println!("{}\n\n{}", sys, user);
        return Ok(());
    }

    // Build context
    let ctx_builder =
        ContextBuilder::new(cwd.clone()).disable_claude_mds(config.disable_claude_mds);
    let system_ctx = ctx_builder.build_system_context().await;
    let user_ctx = ctx_builder.build_user_context().await;

    // Build system prompt
    let mut system_parts = vec![
        include_str!("system_prompt.txt").to_string(),
        system_ctx,
        user_ctx,
    ];
    if let Some(ref custom) = config.custom_system_prompt {
        // replace base system prompt
        system_parts[0] = custom.clone();
    }
    if let Some(ref append) = config.append_system_prompt {
        system_parts.push(append.clone());
    }
    let system_prompt = system_parts.join("\n\n");

    // Determine mode early (needed for auth error handling and permission handler selection).
    let is_headless = cli.print || cli.prompt.is_some();

    // Initialize API client.
    // Try config/env first; fall back to saved OAuth tokens.
    // If no Anthropic credentials are found, check whether any other provider is
    // configured (OpenAI, Google, Ollama, Groq, etc.) — if so, proceed without
    // requiring Anthropic auth. Only launch the OAuth flow when Anthropic is
    // explicitly the intended provider and no key exists at all.
    let other_provider_configured = {
        let active_provider = config.provider.as_deref().unwrap_or("anthropic");
        let has_non_anthropic_env = std::env::var("OPENAI_API_KEY").is_ok()
            || std::env::var("GOOGLE_API_KEY").is_ok()
            || std::env::var("GOOGLE_GENERATIVE_AI_API_KEY").is_ok()
            || std::env::var("GROQ_API_KEY").is_ok()
            || std::env::var("XAI_API_KEY").is_ok()
            || std::env::var("MISTRAL_API_KEY").is_ok()
            || std::env::var("OPENROUTER_API_KEY").is_ok()
            || std::env::var("DEEPSEEK_API_KEY").is_ok()
            || std::env::var("COHERE_API_KEY").is_ok()
            || std::env::var("TOGETHER_API_KEY").is_ok()
            || std::env::var("PERPLEXITY_API_KEY").is_ok()
            || std::env::var("CEREBRAS_API_KEY").is_ok()
            || std::env::var("DEEPINFRA_API_KEY").is_ok()
            || std::env::var("VENICE_API_KEY").is_ok()
            || std::env::var("DASHSCOPE_API_KEY").is_ok()
            || std::env::var("AZURE_API_KEY").is_ok()
            || std::env::var("GITHUB_TOKEN").is_ok()
            || std::env::var("AWS_BEARER_TOKEN_BEDROCK").is_ok()
            || std::env::var("AWS_ACCESS_KEY_ID").is_ok()
            || std::env::var("VERTEX_PROJECT_ID").is_ok()
            // Local providers are always available
            || true; // Ollama/LM Studio don't require keys
        active_provider != "anthropic" || has_non_anthropic_env
    };

    let mut cached_tokens = mangocode_core::oauth::OAuthTokens::load().await;
    if let Some(ref tokens) = cached_tokens {
        if tokens.is_expired_or_expiring_soon() {
            match crate::oauth_flow::refresh_oauth_token(tokens).await {
                Ok(refreshed) => {
                    tracing::info!("OAuth token refreshed");
                    cached_tokens = Some(refreshed);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Token refresh failed - will re-login if needed");
                }
            }
        }
    }

    // Keep `auth.json` aligned with `oauth_tokens.json` for Claude Max so the
    // provider registry's AnthropicMaxProvider is not stuck on a stale access token.
    if let Some(ref t) = cached_tokens {
        if t.uses_bearer_auth() {
            mangocode_core::AuthStore::sync_anthropic_max_from_oauth_tokens(t);
        }
    }

    let resolved_auth = config.resolve_auth_async().await.or_else(|| {
        cached_tokens.as_ref().and_then(|tokens| {
            tokens
                .effective_credential()
                .map(|cred| (cred.to_string(), tokens.uses_bearer_auth()))
        })
    });

    let (api_key, use_bearer_auth) = match resolved_auth {
        Some(auth) => auth,
        None if other_provider_configured
            && config.provider.as_deref().unwrap_or("anthropic") != "anthropic" =>
        {
            // Non-Anthropic provider selected — no Anthropic key needed.
            (String::new(), false)
        }
        None => {
            // No Anthropic credential found.

            if is_headless {
                anyhow::bail!(
                    "No API key found. Options:\n\
                     - Set ANTHROPIC_API_KEY for Anthropic\n\
                     - Set OPENAI_API_KEY for OpenAI\n\
                     - Set GOOGLE_API_KEY for Google Gemini\n\
                     - Set GROQ_API_KEY for Groq (fast, free tier available)\n\
                     - Run `mangocode --provider ollama` for local models (no key needed)\n\
                     - Run `mangocode auth login` for Anthropic OAuth"
                );
            } else {
                // Interactive mode: start the TUI anyway — the provider setup
                // dialog will be shown inside the TUI, just like OpenCode does.
                (String::new(), false)
            }
        }
    };

    let client_config = mangocode_api::client::ClientConfig {
        api_key: api_key.clone(),
        api_base: config.resolve_api_base(),
        use_bearer_auth,
        ..Default::default()
    };
    let client = Arc::new(
        mangocode_api::AnthropicClient::new(client_config.clone())
            .context("Failed to create API client")?,
    );

    // Build provider registry: auto-registers all env-configured providers
    // AND providers with keys stored in ~/.mangocode/auth.json (from /connect).
    // Anthropic is always the default; additional providers (OpenAI, Google,
    // Bedrock, Azure, Copilot, Cohere, local providers) are registered when
    // their respective environment variables or auth store entries are found.
    let provider_registry =
        mangocode_api::ProviderRegistry::from_environment_with_auth_store(client_config);

    let bridge_config = resolve_bridge_config(&settings, &api_key, use_bearer_auth, is_headless);
    if let Some(cfg) = bridge_config.as_ref() {
        info!(
            server_url = %cfg.server_url,
            startup_enabled = settings.remote_control_at_startup,
            "Remote control bridge configured for interactive startup"
        );
    }

    // Build tools
    // Interactive mode uses InteractivePermissionHandler which allows writes in Default mode
    // (the user is watching the TUI so they can intervene). Headless/print mode uses
    // AutoPermissionHandler which denies writes in Default mode for safety.
    let permission_handler: Arc<dyn mangocode_core::PermissionHandler> = if is_headless {
        Arc::new(AutoPermissionHandler {
            mode: config.permission_mode.clone(),
        })
    } else {
        Arc::new(InteractivePermissionHandler {
            mode: config.permission_mode.clone(),
        })
    };
    let cost_tracker = CostTracker::new();
    let session_metrics = mangocode_core::analytics::SessionMetrics::new();
    // Use --session-id if provided, otherwise generate a fresh UUID.
    let session_id = cli
        .session_id_flag
        .clone()
        .or_else(|| cli.resume.clone())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let file_history = Arc::new(ParkingMutex::new(
        mangocode_core::file_history::FileHistory::new(),
    ));
    let current_turn = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // Initialize MCP servers first (needed for ToolContext.mcp_manager).
    let mcp_manager_arc = connect_mcp_manager_arc(&config).await;

    let mut tool_ctx = ToolContext {
        working_dir: cwd.clone(),
        permission_mode: config.permission_mode.clone(),
        permission_handler: permission_handler.clone(),
        cost_tracker: cost_tracker.clone(),
        session_metrics: Some(session_metrics.clone()),
        session_id: session_id.clone(),
        file_history: file_history.clone(),
        current_turn: current_turn.clone(),
        non_interactive: cli.print || cli.prompt.is_some(),
        mcp_manager: mcp_manager_arc.clone(),
        config: config.clone(),
    };

    // Register the cc-query-backed agent runner so TeamCreateTool can spawn real
    // sub-agents.  Must be called before any tool execution begins.
    // The function is idempotent if already registered (panics only on double-call,
    // but we guard with a std::sync::OnceLock internally).
    {
        static SWARM_INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        SWARM_INIT.get_or_init(mangocode_query::init_team_swarm_runner);
    }

    // Build the full tool list: built-ins from cc-tools plus AgentTool from cc-query
    // (AgentTool lives in cc-query to avoid a circular cc-tools ↔ cc-query dependency).
    // Wrap in Arc so the list can be shared by the main loop AND the cron scheduler.
    let tools = build_tools_with_mcp(
        mcp_manager_arc.clone(),
        &config.allowed_tools,
        &config.disallowed_tools,
    );

    // Load plugins and register any plugin-provided MCP servers into the
    // in-memory config (does not modify the settings file on disk).
    let plugin_registry = mangocode_plugins::load_plugins(&cwd, &[]).await;
    {
        let plugin_cmd_count = plugin_registry.all_command_defs().len();
        let plugin_hook_count = plugin_registry
            .build_hook_registry()
            .values()
            .map(|v| v.len())
            .sum::<usize>();
        info!(
            plugins = plugin_registry.enabled_count(),
            commands = plugin_cmd_count,
            hooks = plugin_hook_count,
            "Plugins loaded"
        );

        // Register plugin MCP servers into the in-memory config so they are
        // picked up by any subsequent MCP manager construction.
        let existing_names: std::collections::HashSet<String> =
            config.mcp_servers.iter().map(|s| s.name.clone()).collect();
        for mcp_server in plugin_registry.all_mcp_servers() {
            if !existing_names.contains(&mcp_server.name) {
                config.mcp_servers.push(mcp_server);
            }
        }
    }

    // Build model registry for dynamic model/provider resolution.
    // The registry is pre-populated with a hardcoded snapshot and enriched
    // from the models.dev cache if available.
    let model_registry = {
        let mut reg = mangocode_api::ModelRegistry::new();
        let cache_path = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("mangocode")
            .join("models.json");
        reg.load_cache(&cache_path);
        Arc::new(reg)
    };

    // Resolve the effective model and write it back to config so all consumers
    // (Config tool, TUI status bar, init event) report the same model that's
    // actually used for API calls.
    if config.model.is_none() {
        config.model = Some(mangocode_api::effective_model_for_config(&config, &model_registry));
    }
    // Update the ToolContext config to match (it was cloned before registry was available)
    tool_ctx.config.model = config.model.clone();

    // Build query config
    let mut query_config =
        mangocode_query::QueryConfig::from_config_with_registry(&config, &model_registry);
    query_config.model_registry = Some(model_registry.clone());
    query_config.max_turns = cli.max_turns;
    query_config.system_prompt = Some(system_prompt);
    query_config.append_system_prompt = None;
    query_config.working_directory = Some(cwd.display().to_string());
    if let Some(tokens) = cli.thinking {
        query_config.thinking_budget = Some(tokens);
    }
    if let Some(ref level_str) = cli.effort {
        if let Some(level) = mangocode_core::effort::EffortLevel::parse(level_str) {
            query_config.effort_level = Some(level);
        } else {
            eprintln!(
                "Warning: unknown effort level '{}' — expected low/medium/high/max",
                level_str
            );
        }
    }
    if let Some(usd) = cli.max_budget_usd {
        query_config.max_budget_usd = Some(usd);
    }
    if let Some(ref fb) = cli.fallback_model {
        query_config.fallback_model = Some(fb.clone());
    }
    // Wire in the provider registry so non-Anthropic providers can be dispatched.
    let provider_registry = std::sync::Arc::new(provider_registry);
    query_config.provider_registry = Some(provider_registry.clone());

    // Wire in the named agent (--agent flag).
    // Merge built-in default agents with user-defined agents (user wins on collision).
    let tools = if let Some(ref agent_name) = cli.agent {
        query_config.agent_name = Some(agent_name.clone());
        let mut all_agents = mangocode_core::default_agents();
        all_agents.extend(config.agents.clone());
        if let Some(def) = all_agents.get(agent_name) {
            let access = def.access.clone();
            query_config.agent_definition = Some(def.clone());
            // Override max_turns from agent definition when specified.
            if let Some(turns) = def.max_turns {
                query_config.max_turns = turns;
            }
            filter_tools_for_agent(tools, &access)
        } else {
            eprintln!(
                "Warning: unknown agent '{}'. Run /agent to see available agents.",
                agent_name
            );
            tools
        }
    } else {
        tools
    };

    // Spawn the background cron scheduler (fires cron tasks at scheduled times).
    // Cancelled automatically when the process exits since we use a shared token.
    let cron_cancel = tokio_util::sync::CancellationToken::new();
    mangocode_query::start_cron_scheduler(
        client.clone(),
        tools.clone(),
        tool_ctx.clone(),
        query_config.clone(),
        cron_cancel.clone(),
    );

    // Spawn proactive monitor loop (feature-gated + opt-in via /proactive on).
    let proactive_cancel = tokio_util::sync::CancellationToken::new();
    let _proactive_handle = mangocode_query::ProactiveAgent::new(
        cwd.clone(),
        tool_ctx.session_id.clone(),
    )
    .start(
        client.clone(),
        tools.clone(),
        tool_ctx.clone(),
        query_config.clone(),
        proactive_cancel.clone(),
    );

    // Spawn background remote settings poller.
    let remote_settings_cancel = tokio_util::sync::CancellationToken::new();
    let remote_config = mangocode_core::remote_settings::RemoteSettingsConfig::from_env();
    if mangocode_core::remote_settings::RemoteSettingsManager::is_eligible(
        remote_config.api_key.as_deref(),
    ) {
        let manager = std::sync::Arc::new(
            mangocode_core::remote_settings::RemoteSettingsManager::new(remote_config),
        );
        let poll_cancel = remote_settings_cancel.child_token();
        tokio::spawn(async move {
            manager.start_polling(poll_cancel).await;
        });
    }

    // --print mode (headless)
    let result = if is_headless {
        run_headless(&cli, client, tools, tool_ctx, query_config, cost_tracker).await
    } else {
        // Capture provider before `config` is moved into run_interactive.
        let is_non_anthropic_provider = config
            .provider
            .as_deref()
            .map(|p| p != "anthropic")
            .unwrap_or(false);
        run_interactive(InteractiveRunArgs {
            config,
            settings,
            client,
            tools,
            tool_ctx,
            query_config,
            cost_tracker,
            resume_id: cli.resume,
            bridge_config,
            // has_credentials: true if we have an Anthropic key, OR if a
            // non-Anthropic provider is selected (its own auth is checked at
            // request time - we don't want to block TUI startup here).
            has_credentials: !api_key.is_empty() || is_non_anthropic_provider,
            model_registry,
        })
        .await
    };

    cron_cancel.cancel();
    proactive_cancel.cancel();
    remote_settings_cancel.cancel();
    result
}

async fn connect_mcp_manager_arc(config: &Config) -> Option<Arc<mangocode_mcp::McpManager>> {
    if config.mcp_servers.is_empty() {
        return None;
    }

    info!(
        count = config.mcp_servers.len(),
        "Connecting to MCP servers"
    );
    let mcp_manager = mangocode_mcp::McpManager::connect_all(&config.mcp_servers).await;
    let mcp_manager = Arc::new(mcp_manager);
    mcp_manager.clone().spawn_notification_poll_loop();
    Some(mcp_manager)
}

fn build_tools_with_mcp(
    mcp_manager: Option<Arc<mangocode_mcp::McpManager>>,
    allowed_tools: &[String],
    disallowed_tools: &[String],
) -> Arc<Vec<Box<dyn mangocode_tools::Tool>>> {
    let mut v: Vec<Box<dyn mangocode_tools::Tool>> = mangocode_tools::all_tools();
    v.push(Box::new(mangocode_query::AgentTool));

    if let Some(ref manager_arc) = mcp_manager {
        for (server_name, original_tool_def) in manager_arc.all_tool_definitions() {
            let internal_prefix = format!("{}_", server_name);
            let bare_tool_name = original_tool_def
                .name
                .strip_prefix(&internal_prefix)
                .unwrap_or(&original_tool_def.name)
                .to_string();

            let compat_tool_name = format!("mcp__{}__{}", server_name, bare_tool_name);
            let manager_tool_name = original_tool_def.name.clone();
            let tool_def = ToolDefinition {
                name: compat_tool_name,
                description: original_tool_def.description,
                input_schema: original_tool_def.input_schema,
            };

            let wrapper = McpToolWrapper {
                tool_def,
                server_name,
                manager_tool_name,
                manager: manager_arc.clone(),
            };
            v.push(Box::new(wrapper));
        }
        debug!(total_tools = v.len(), "MCP tools registered");
    }

    if !allowed_tools.is_empty() {
        let allowed: HashSet<&str> = allowed_tools.iter().map(|s| s.as_str()).collect();
        v.retain(|t| allowed.contains(t.name()));
    }

    if !disallowed_tools.is_empty() {
        let denied: HashSet<&str> = disallowed_tools.iter().map(|s| s.as_str()).collect();
        v.retain(|t| !denied.contains(t.name()));
    }

    Arc::new(v)
}

/// Filter the tool list based on the agent's access level.
/// - "full"        → all tools allowed (no filtering)
/// - "read-only"   → only ReadOnly/None permission tools and AskUserQuestion
/// - "search-only" → only Grep, Glob, Read, WebSearch, WebFetch tools
fn filter_tools_for_agent(
    tools: Arc<Vec<Box<dyn mangocode_tools::Tool>>>,
    access: &str,
) -> Arc<Vec<Box<dyn mangocode_tools::Tool>>> {
    use mangocode_tools::PermissionLevel as PL;
    match access {
        "read-only" => {
            // Collect names of tools that are read-only, then rebuild from all_tools
            // (Box<dyn Tool> is not Clone so we can't directly filter-and-keep).
            let allowed_names: Vec<String> = tools
                .iter()
                .filter(|t| {
                    matches!(t.permission_level(), PL::ReadOnly | PL::None)
                        || t.name() == "AskUserQuestion"
                })
                .map(|t| t.name().to_string())
                .collect();
            let filtered: Vec<Box<dyn mangocode_tools::Tool>> = mangocode_tools::all_tools()
                .into_iter()
                .filter(|t| allowed_names.iter().any(|n| n == t.name()))
                .collect();
            Arc::new(filtered)
        }
        "search-only" => {
            const SEARCH_TOOLS: &[&str] = &["Grep", "Glob", "Read", "WebSearch", "WebFetch"];
            let filtered: Vec<Box<dyn mangocode_tools::Tool>> = mangocode_tools::all_tools()
                .into_iter()
                .filter(|t| SEARCH_TOOLS.contains(&t.name()))
                .collect();
            Arc::new(filtered)
        }
        _ => tools, // "full" — allow all tools unchanged
    }
}

// ---------------------------------------------------------------------------
// Headless mode: read prompt from arg/stdin, run, print response
// ---------------------------------------------------------------------------

async fn run_headless(
    cli: &Cli,
    client: Arc<mangocode_api::AnthropicClient>,
    tools: Arc<Vec<Box<dyn mangocode_tools::Tool>>>,
    tool_ctx: ToolContext,
    query_config: mangocode_query::QueryConfig,
    cost_tracker: Arc<CostTracker>,
) -> anyhow::Result<()> {
    use mangocode_query::{QueryEvent, QueryOutcome};
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    fn permission_mode_name(mode: &PermissionMode) -> &'static str {
        match mode {
            PermissionMode::Default => "default",
            PermissionMode::AcceptEdits => "acceptEdits",
            PermissionMode::BypassPermissions => "bypassPermissions",
            PermissionMode::Plan => "plan",
        }
    }

    fn emit_ndjson(value: serde_json::Value) {
        println!("{}", value);
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }

    fn model_usage_json(
        model: &str,
        usage: &mangocode_core::types::UsageInfo,
        cost_usd: f64,
    ) -> serde_json::Value {
        serde_json::json!({
            model: {
                "inputTokens": usage.input_tokens,
                "outputTokens": usage.output_tokens,
                "costUSD": cost_usd,
            }
        })
    }

    let start_time = std::time::Instant::now();

    // Load prior conversation when --resume is provided; headless mode keeps
    // using the active tool-context session ID for event/session linkage.
    let mut session = if let Some(ref id) = cli.resume {
        match mangocode_core::history::load_session(id).await {
            Ok(mut s) => {
                if s.id != tool_ctx.session_id {
                    s.id = tool_ctx.session_id.clone();
                }
                s
            }
            Err(e) => {
                eprintln!("Warning: could not load session {}: {}", id, e);
                let mut fresh =
                    mangocode_core::history::ConversationSession::new(query_config.model.clone());
                fresh.id = tool_ctx.session_id.clone();
                fresh
            }
        }
    } else {
        let mut fresh =
            mangocode_core::history::ConversationSession::new(query_config.model.clone());
        fresh.id = tool_ctx.session_id.clone();
        fresh
    };

    session.working_dir = Some(tool_ctx.working_dir.display().to_string());
    if session.model.is_empty() {
        session.model = query_config.model.clone();
    }

    // Build new input messages for this invocation.
    // --input-format stream-json: stdin is newline-delimited JSON, each line is
    //   {"role":"user"|"assistant","content":"..."} (mirrors TS --input-format stream-json).
    // --input-format text (default): read prompt from positional arg or entire stdin as text.
    let mut incoming_messages: Vec<mangocode_core::types::Message> =
        if cli.input_format == CliInputFormat::StreamJson {
            use tokio::io::{self, AsyncBufReadExt, BufReader};
            let stdin = io::stdin();
            let mut reader = BufReader::new(stdin);
            let mut line = String::new();
            let mut parsed: Vec<mangocode_core::types::Message> = Vec::new();
            loop {
                line.clear();
                let n = reader.read_line(&mut line).await?;
                if n == 0 {
                    break;
                }
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                match serde_json::from_str::<serde_json::Value>(trimmed) {
                    Ok(v) => {
                        let role = v.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                        let content = v
                            .get("content")
                            .and_then(|c| c.as_str())
                            .unwrap_or("")
                            .to_string();
                        if role == "assistant" {
                            parsed.push(mangocode_core::types::Message::assistant(content));
                        } else {
                            parsed.push(mangocode_core::types::Message::user(content));
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: skipping malformed JSON line: {} ({:?})",
                            trimmed, e
                        );
                    }
                }
            }
            if parsed.is_empty() {
                // Also check positional arg as fallback
                if let Some(ref p) = cli.prompt {
                    parsed.push(mangocode_core::types::Message::user(p.clone()));
                }
            }
            parsed
        } else {
            // Plain text mode
            let prompt = if let Some(ref p) = cli.prompt {
                p.clone()
            } else {
                use tokio::io::{self, AsyncReadExt};
                let mut stdin = io::stdin();
                let mut buf = String::new();
                stdin.read_to_string(&mut buf).await?;
                buf.trim().to_string()
            };

            if prompt.is_empty() {
                eprintln!("Error: No prompt provided. Use --print <prompt> or pipe text to stdin.");
                std::process::exit(1);
            }

            vec![mangocode_core::types::Message::user(prompt)]
        };

    // --prefill: inject a partial assistant turn before the query so the model
    // continues from that text (mirrors TS --prefill flag).
    if let Some(ref prefill_text) = cli.prefill {
        incoming_messages.push(mangocode_core::types::Message::assistant(
            prefill_text.clone(),
        ));
    }

    let mut messages = session.messages.clone();
    messages.extend(incoming_messages);

    if messages.is_empty() {
        eprintln!("Error: No messages provided.");
        std::process::exit(1);
    }

    let is_json_output = matches!(
        cli.output_format,
        CliOutputFormat::Json | CliOutputFormat::StreamJson
    );
    let is_stream_json = matches!(cli.output_format, CliOutputFormat::StreamJson);

    if is_stream_json {
        let mut tool_names: Vec<String> = tools.iter().map(|t| t.name().to_string()).collect();
        tool_names.sort();

        let mut mcp_servers = Vec::new();
        if let Some(ref manager) = tool_ctx.mcp_manager {
            let mut statuses: Vec<(String, mangocode_mcp::McpServerStatus)> =
                manager.all_statuses().into_iter().collect();
            statuses.sort_by(|a, b| a.0.cmp(&b.0));
            for (name, status) in statuses {
                let status_name = match status {
                    mangocode_mcp::McpServerStatus::Connected { .. } => "connected",
                    mangocode_mcp::McpServerStatus::Connecting => "connecting",
                    mangocode_mcp::McpServerStatus::Disconnected { .. } => "disconnected",
                    mangocode_mcp::McpServerStatus::Failed { .. } => "failed",
                };
                mcp_servers.push(serde_json::json!({
                    "name": name,
                    "status": status_name,
                }));
            }
        }

        emit_ndjson(serde_json::json!({
            "type": "system",
            "subtype": "init",
            "session_id": tool_ctx.session_id,
            "cwd": tool_ctx.working_dir.display().to_string(),
            "model": query_config.model,
            "tools": tool_names,
            "mcp_servers": mcp_servers,
            "permissionMode": permission_mode_name(&tool_ctx.permission_mode),
        }));
    }

    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<QueryEvent>();
    let cancel = CancellationToken::new();
    let stream_session_id = tool_ctx.session_id.clone();
    let mut brief_tool_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    let event_task = tokio::spawn(async move {
        let mut full_text = String::new();
        while let Some(event) = event_rx.recv().await {
            match event {
                QueryEvent::Stream(mangocode_api::AnthropicStreamEvent::ContentBlockDelta {
                    delta: mangocode_api::streaming::ContentDelta::TextDelta { text },
                    ..
                }) => {
                    full_text.push_str(&text);
                    if !is_json_output {
                        print!("{}", text);
                        use std::io::Write;
                        let _ = std::io::stdout().flush();
                    } else if is_stream_json {
                        // Emit CC-compatible content_block_delta + content_block_stop
                        // instead of wrapping in an assistant message, so orchestration
                        // UIs can use the standard Claude streaming adapter.
                        emit_ndjson(serde_json::json!({
                            "type": "content_block_delta",
                            "delta": { "text": text },
                            "session_id": stream_session_id,
                        }));
                        emit_ndjson(serde_json::json!({
                            "type": "content_block_stop",
                            "session_id": stream_session_id,
                        }));
                    }
                }
                QueryEvent::StreamWithParent {
                    event:
                        mangocode_api::AnthropicStreamEvent::ContentBlockDelta {
                            delta: mangocode_api::streaming::ContentDelta::TextDelta { text },
                            ..
                        },
                    parent_tool_use_id,
                } => {
                    full_text.push_str(&text);
                    if !is_json_output {
                        print!("{}", text);
                        use std::io::Write;
                        let _ = std::io::stdout().flush();
                    } else if is_stream_json {
                        emit_ndjson(serde_json::json!({
                            "type": "content_block_delta",
                            "delta": { "text": text },
                            "parent_tool_use_id": parent_tool_use_id,
                            "session_id": stream_session_id,
                        }));
                        emit_ndjson(serde_json::json!({
                            "type": "content_block_stop",
                            "session_id": stream_session_id,
                        }));
                    }
                }
                QueryEvent::Stream(mangocode_api::AnthropicStreamEvent::ContentBlockDelta {
                    delta: mangocode_api::streaming::ContentDelta::ThinkingDelta { thinking },
                    ..
                }) => {
                    if is_stream_json {
                        emit_ndjson(serde_json::json!({
                            "type": "assistant",
                            "message": {
                                "role": "assistant",
                                "content": [{ "type": "thinking", "thinking": thinking }]
                            },
                            "parent_tool_use_id": serde_json::Value::Null,
                            "session_id": stream_session_id,
                        }));
                    }
                }
                QueryEvent::StreamWithParent {
                    event:
                        mangocode_api::AnthropicStreamEvent::ContentBlockDelta {
                            delta:
                                mangocode_api::streaming::ContentDelta::ThinkingDelta { thinking },
                            ..
                        },
                    parent_tool_use_id,
                } => {
                    if is_stream_json {
                        emit_ndjson(serde_json::json!({
                            "type": "assistant",
                            "message": {
                                "role": "assistant",
                                "content": [{ "type": "thinking", "thinking": thinking }]
                            },
                            "parent_tool_use_id": parent_tool_use_id,
                            "session_id": stream_session_id,
                        }));
                    }
                }
                QueryEvent::Stream(mangocode_api::AnthropicStreamEvent::ContentBlockDelta {
                    delta: mangocode_api::streaming::ContentDelta::SignatureDelta { signature },
                    ..
                }) => {
                    if is_stream_json {
                        emit_ndjson(serde_json::json!({
                            "type": "assistant",
                            "message": {
                                "role": "assistant",
                                "content": [{ "type": "thinking", "thinking": "", "signature": signature }]
                            },
                            "parent_tool_use_id": serde_json::Value::Null,
                            "session_id": stream_session_id,
                        }));
                    }
                }
                QueryEvent::StreamWithParent {
                    event:
                        mangocode_api::AnthropicStreamEvent::ContentBlockDelta {
                            delta:
                                mangocode_api::streaming::ContentDelta::SignatureDelta { signature },
                            ..
                        },
                    parent_tool_use_id,
                } => {
                    if is_stream_json {
                        emit_ndjson(serde_json::json!({
                            "type": "assistant",
                            "message": {
                                "role": "assistant",
                                "content": [{ "type": "thinking", "thinking": "", "signature": signature }]
                            },
                            "parent_tool_use_id": parent_tool_use_id,
                            "session_id": stream_session_id,
                        }));
                    }
                }
                QueryEvent::ToolStart {
                    tool_name,
                    tool_id,
                    input_json,
                    parent_tool_use_id,
                } => {
                    if !is_json_output {
                        eprintln!("\n[{}...]", tool_name);
                    } else if is_stream_json {
                        let parsed_input = serde_json::from_str::<serde_json::Value>(&input_json)
                            .unwrap_or_else(|_| serde_json::json!({ "raw": input_json }));
                        // In stream-json mode, suppress Brief tool_use events entirely.
                        // Brief is an internal display wrapper — the model also emits
                        // a separate assistant text event with the same content, so
                        // emitting Brief as tool_use would cause duplicate messages
                        // in orchestration UIs like Conducctor.
                        if tool_name == "Brief" {
                            brief_tool_ids.insert(tool_id.clone());
                        } else {
                            emit_ndjson(serde_json::json!({
                                "type": "assistant",
                                "message": {
                                    "role": "assistant",
                                    "content": [{
                                        "type": "tool_use",
                                        "id": tool_id,
                                        "name": tool_name,
                                        "input": parsed_input,
                                    }]
                                },
                                "parent_tool_use_id": parent_tool_use_id
                                    .map(serde_json::Value::String)
                                    .unwrap_or(serde_json::Value::Null),
                                "session_id": stream_session_id,
                            }));
                        }
                    } else {
                        emit_ndjson(serde_json::json!({ "type": "tool_start", "tool": tool_name }));
                    }
                }
                QueryEvent::ToolEnd {
                    tool_id,
                    result,
                    is_error,
                    ..
                } => {
                    if is_stream_json {
                        // Skip tool_result for Brief (already emitted as text)
                        if brief_tool_ids.remove(&tool_id) {
                            continue;
                        }
                        emit_ndjson(serde_json::json!({
                            "type": "user",
                            "message": {
                                "role": "user",
                                "content": [{
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": result,
                                    "is_error": is_error,
                                }]
                            },
                            "session_id": stream_session_id,
                        }));
                    }
                }
                QueryEvent::Error(msg) => {
                    if is_stream_json {
                        let lower = msg.to_ascii_lowercase();
                        if lower.contains("rate limit") || lower.contains("ratelimit") {
                            emit_ndjson(serde_json::json!({
                                "type": "rate_limit_event",
                                "session_id": stream_session_id,
                                "rate_limit_info": {
                                    "message": msg,
                                }
                            }));
                        }
                        // Stream-json compatibility mode reserves terminal errors
                        // for the final `result/subtype=error` event.
                    } else if is_json_output {
                        eprintln!("{}", serde_json::json!({ "type": "error", "error": msg }));
                    } else {
                        eprintln!("\nError: {}", msg);
                    }
                }
                _ => {}
            }
        }

        full_text
    });

    let outcome = mangocode_query::run_query_loop(
        client.as_ref(),
        &mut messages,
        tools.as_slice(),
        &tool_ctx,
        &query_config,
        cost_tracker.clone(),
        Some(event_tx.clone()),
        cancel,
        None,
    )
    .await;

    drop(event_tx);
    let full_text = event_task.await.unwrap_or_default();

    session.messages = messages;
    session.model = query_config.model.clone();
    session.working_dir = Some(tool_ctx.working_dir.display().to_string());
    session.total_cost = cost_tracker.total_cost_usd();
    session.total_tokens = cost_tracker.input_tokens() + cost_tracker.output_tokens();
    session.updated_at = chrono::Utc::now();
    let duration_ms = start_time.elapsed().as_millis() as u64;

    if let Err(e) = mangocode_core::history::save_session(&session).await {
        eprintln!("Warning: failed to save session {}: {}", session.id, e);
    }

    persist_session_usage(&cost_tracker, &session, duration_ms);

    // Final output
    match cli.output_format {
        CliOutputFormat::Json => match outcome {
            QueryOutcome::EndTurn { message, usage } => {
                let result_text = if full_text.is_empty() {
                    message.get_all_text()
                } else {
                    full_text
                };
                let out = serde_json::json!({
                    "type": "result",
                    "result": result_text,
                    "session_id": tool_ctx.session_id,
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
                        "cache_read_input_tokens": usage.cache_read_input_tokens,
                    },
                    "cost_usd": cost_tracker.total_cost_usd(),
                });
                println!("{}", out);
            }
            QueryOutcome::MaxTokens {
                partial_message,
                usage,
            } => {
                let result_text = if full_text.is_empty() {
                    partial_message.get_all_text()
                } else {
                    full_text
                };
                let out = serde_json::json!({
                    "type": "result",
                    "result": result_text,
                    "session_id": tool_ctx.session_id,
                    "stop_reason": "max_tokens",
                    "usage": {
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
                        "cache_read_input_tokens": usage.cache_read_input_tokens,
                    },
                    "cost_usd": cost_tracker.total_cost_usd(),
                });
                println!("{}", out);
            }
            QueryOutcome::Error(e) => {
                let out = serde_json::json!({ "type": "error", "error": e.to_string() });
                eprintln!("{}", out);
                std::process::exit(1);
            }
            QueryOutcome::BudgetExceeded {
                cost_usd,
                limit_usd,
            } => {
                let out = serde_json::json!({
                    "type": "error",
                    "error": format!("Budget limit ${:.4} reached (spent ${:.4})", limit_usd, cost_usd),
                });
                eprintln!("{}", out);
                std::process::exit(2);
            }
            QueryOutcome::Cancelled => {
                let out = serde_json::json!({ "type": "error", "error": "Cancelled" });
                eprintln!("{}", out);
                std::process::exit(1);
            }
        },
        CliOutputFormat::StreamJson => {
            // Already streamed above; emit final Claude-compatible result event.
            match outcome {
                QueryOutcome::EndTurn { usage, .. } => {
                    emit_ndjson(serde_json::json!({
                        "type": "result",
                        "subtype": "success",
                        "is_error": false,
                        "session_id": tool_ctx.session_id,
                        "stop_reason": "end_turn",
                        "duration_ms": duration_ms,
                        "cost_usd": cost_tracker.total_cost_usd(),
                        "usage": {
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "cache_creation_input_tokens": usage.cache_creation_input_tokens,
                            "cache_read_input_tokens": usage.cache_read_input_tokens,
                        },
                        "modelUsage": model_usage_json(
                            &query_config.model,
                            &usage,
                            cost_tracker.total_cost_usd(),
                        ),
                    }));
                }
                QueryOutcome::MaxTokens { usage, .. } => {
                    emit_ndjson(serde_json::json!({
                        "type": "result",
                        "subtype": "success",
                        "is_error": false,
                        "session_id": tool_ctx.session_id,
                        "stop_reason": "max_tokens",
                        "duration_ms": duration_ms,
                        "cost_usd": cost_tracker.total_cost_usd(),
                        "usage": {
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "cache_creation_input_tokens": usage.cache_creation_input_tokens,
                            "cache_read_input_tokens": usage.cache_read_input_tokens,
                        },
                        "modelUsage": model_usage_json(
                            &query_config.model,
                            &usage,
                            cost_tracker.total_cost_usd(),
                        ),
                    }));
                }
                QueryOutcome::Error(e) => {
                    emit_ndjson(serde_json::json!({
                        "type": "result",
                        "subtype": "error",
                        "is_error": true,
                        "session_id": tool_ctx.session_id,
                        "error": {
                            "type": "Error",
                            "message": e.to_string(),
                        }
                    }));
                    std::process::exit(1);
                }
                QueryOutcome::BudgetExceeded {
                    cost_usd,
                    limit_usd,
                } => {
                    emit_ndjson(serde_json::json!({
                        "type": "result",
                        "subtype": "error",
                        "is_error": true,
                        "session_id": tool_ctx.session_id,
                        "error": {
                            "type": "BudgetExceeded",
                            "message": format!("Budget limit ${:.4} reached (spent ${:.4})", limit_usd, cost_usd),
                        }
                    }));
                    std::process::exit(2);
                }
                QueryOutcome::Cancelled => {
                    emit_ndjson(serde_json::json!({
                        "type": "result",
                        "subtype": "error",
                        "is_error": true,
                        "session_id": tool_ctx.session_id,
                        "error": {
                            "type": "Cancelled",
                            "message": "Cancelled",
                        }
                    }));
                    std::process::exit(1);
                }
            }
        }
        CliOutputFormat::Text => {
            // Streaming text was already printed; add newline
            println!();
            if cli.verbose {
                eprintln!(
                    "\nTokens: {} in / {} out | Cost: ${:.4}",
                    cost_tracker.input_tokens(),
                    cost_tracker.output_tokens(),
                    cost_tracker.total_cost_usd(),
                );
            }
            match outcome {
                QueryOutcome::Error(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
                QueryOutcome::BudgetExceeded {
                    cost_usd,
                    limit_usd,
                } => {
                    eprintln!(
                        "Budget limit ${:.4} reached (spent ${:.4}). Stopping.",
                        limit_usd, cost_usd
                    );
                    std::process::exit(2);
                }
                _ => {}
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Interactive REPL mode
// ---------------------------------------------------------------------------

struct InteractiveRunArgs {
    config: Config,
    settings: mangocode_core::config::Settings,
    client: Arc<mangocode_api::AnthropicClient>,
    tools: Arc<Vec<Box<dyn mangocode_tools::Tool>>>,
    tool_ctx: ToolContext,
    query_config: mangocode_query::QueryConfig,
    cost_tracker: Arc<CostTracker>,
    resume_id: Option<String>,
    bridge_config: Option<mangocode_bridge::BridgeConfig>,
    has_credentials: bool,
    model_registry: Arc<mangocode_api::ModelRegistry>,
}

fn persist_session_usage(
    cost_tracker: &Arc<CostTracker>,
    session: &mangocode_core::history::ConversationSession,
    duration_ms: u64,
) {
    let mut ledger = mangocode_core::usage_ledger::UsageLedger::load();
    ledger.record_session(mangocode_core::usage_ledger::SessionCostRecord {
        session_id: session.id.clone(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        model: session.model.clone(),
        cost_usd: cost_tracker.total_cost_usd(),
        input_tokens: cost_tracker.input_tokens(),
        output_tokens: cost_tracker.output_tokens(),
        cache_creation_tokens: cost_tracker.cache_creation_tokens(),
        cache_read_tokens: cost_tracker.cache_read_tokens(),
        duration_ms,
        working_dir: session.working_dir.clone().unwrap_or_default(),
    });
}

async fn run_interactive(args: InteractiveRunArgs) -> anyhow::Result<()> {
    let InteractiveRunArgs {
        config,
        settings,
        client,
        tools,
        tool_ctx,
        query_config,
        cost_tracker,
        resume_id,
        bridge_config,
        has_credentials,
        model_registry,
    } = args;

    use crossterm::event::{self, Event, KeyCode};
    use mangocode_bridge::{BridgeOutbound, TuiBridgeEvent};
    use mangocode_commands::{execute_command, CommandContext, CommandResult};
    use mangocode_query::{QueryEvent, QueryOutcome};
    use mangocode_tui::{
        bridge_state::BridgeConnectionState,
        device_auth_dialog::DeviceAuthEvent,
        init_mascot,
        notifications::NotificationKind,
        render::{flush_sixel_blit, render_app, reset_sixel_blit_state},
        restore_terminal, setup_terminal, App,
    };
    use std::time::Duration;
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    let mut tool_ctx = tool_ctx;
    let session_start = std::time::Instant::now();
    let mut session = if let Some(ref id) = resume_id {
        match mangocode_core::history::load_session(id).await {
            Ok(session) => {
                println!("Resumed session: {}", id);
                if let Some(saved_dir) = session.working_dir.as_ref() {
                    let saved_path = std::path::PathBuf::from(saved_dir);
                    if saved_path.exists() {
                        tool_ctx.working_dir = saved_path;
                    }
                }
                tool_ctx.session_id = session.id.clone();
                session
            }
            Err(e) => {
                eprintln!("Warning: could not load session {}: {}", id, e);
                let mut session = mangocode_core::history::ConversationSession::new(
                    mangocode_api::effective_model_for_config(&config, &model_registry),
                );
                session.id = tool_ctx.session_id.clone();
                session.working_dir = Some(tool_ctx.working_dir.display().to_string());
                session
            }
        }
    } else {
        let mut session = mangocode_core::history::ConversationSession::new(
            mangocode_api::effective_model_for_config(&config, &model_registry),
        );
        session.id = tool_ctx.session_id.clone();
        session.working_dir = Some(tool_ctx.working_dir.display().to_string());
        session
    };
    let initial_messages = session.messages.clone();
    let base_query_config = query_config;
    let mut live_config = config.clone();
    if !session.model.is_empty() {
        live_config.model = Some(session.model.clone());
    }
    tool_ctx.config = live_config.clone();

    // Set up terminal
    let mut terminal = setup_terminal()?;
    let mut app = App::new(live_config.clone(), cost_tracker.clone());
    init_mascot(&mut app);
    // Sync initial effort level (from --effort flag or /effort command) to TUI indicator.
    if let Some(level) = base_query_config.effort_level {
        use mangocode_tui::EffortLevel as TuiEL;
        app.effort_level = match level {
            mangocode_core::effort::EffortLevel::Low => TuiEL::Low,
            mangocode_core::effort::EffortLevel::Medium => TuiEL::Normal,
            mangocode_core::effort::EffortLevel::High => TuiEL::High,
            mangocode_core::effort::EffortLevel::Max => TuiEL::Max,
        };
    }
    app.provider_registry = base_query_config.provider_registry.clone();

    // Background: refresh the model registry from models.dev.
    // The fetched JSON is saved as a cache file; the App will reload it from
    // disk whenever the /model picker opens.
    {
        let cache_path = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("mangocode")
            .join("models.json");
        tokio::spawn(async move {
            let client = match reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
            {
                Ok(c) => c,
                Err(_) => return,
            };
            let url = std::env::var("MODELS_DEV_URL")
                .unwrap_or_else(|_| "https://models.dev/api.json".to_string());
            if let Ok(resp) = client
                .get(&url)
                .header("User-Agent", "MangoCode/0.0.7")
                .send()
                .await
            {
                if resp.status().is_success() {
                    if let Ok(text) = resp.text().await {
                        let _ = std::fs::create_dir_all(
                            cache_path.parent().unwrap_or(std::path::Path::new(".")),
                        );
                        let _ = std::fs::write(&cache_path, &text);
                        tracing::info!("Models cache refreshed from models.dev");
                    }
                }
            }
        });
    }

    app.config.project_dir = Some(tool_ctx.working_dir.clone());
    app.attach_turn_diff_state(tool_ctx.file_history.clone(), tool_ctx.current_turn.clone());
    if let Some(manager) = tool_ctx.mcp_manager.clone() {
        app.attach_mcp_manager(manager);
    }
    app.replace_messages(initial_messages.clone());

    // Home directory warning: mirror TS feedConfigs.tsx warningText
    let home_dir = dirs::home_dir();
    if home_dir.as_deref() == Some(tool_ctx.working_dir.as_path()) {
        app.home_dir_warning = true;
    }

    // Bypass permissions confirmation dialog: must be accepted before any work
    // Mark whether valid credentials exist so the TUI can show a provider
    // setup dialog instead of failing silently on the first message.
    app.has_credentials = has_credentials;

    // If a non-Anthropic provider is active, prefix model_name with "provider/model"
    // so the status bar can show the provider name.
    if let Some(ref provider) = live_config.provider {
        if provider != "anthropic" && !app.model_name.contains('/') {
            app.model_name = format!("{}/{}", provider, app.model_name);
        }
    }

    // Set agent mode from the --agent flag (carried on query_config).
    if let Some(ref agent_name) = base_query_config.agent_name {
        app.agent_mode = Some(agent_name.clone());
    }

    // Show onboarding: status hint if no credentials, welcome tour if first run.
    if !has_credentials {
        app.status_message =
            Some("No provider configured. Run /connect to set one up.".to_string());
    } else if !settings.has_completed_onboarding {
        app.onboarding_dialog.show();
    }

    // Mirror TS BypassPermissionsModeDialog.tsx startup gate
    use mangocode_core::config::PermissionMode;
    if live_config.permission_mode == PermissionMode::BypassPermissions {
        app.bypass_permissions_dialog.show();
    }

    // Version-upgrade notice: record the current version for future comparisons.
    // (Actual upgrade notice UI is handled by the release-notes slash command.)
    {
        let current_version = mangocode_core::constants::APP_VERSION.to_string();
        if settings.last_seen_version.as_deref() != Some(&current_version) {
            // Persist asynchronously to avoid blocking startup.
            let version_clone = current_version.clone();
            tokio::spawn(async move {
                if let Ok(mut s) = mangocode_core::config::Settings::load().await {
                    s.last_seen_version = Some(version_clone);
                    let _ = s.save().await;
                }
            });
        }
    }

    // CLAUDE_STATUS_COMMAND: optional external command whose stdout replaces the
    // left-side status bar text. Polled every 500ms (debounced) in the main loop.
    // The command is run in a background task; results flow through a channel.
    let status_cmd_str = std::env::var("CLAUDE_STATUS_COMMAND").ok();
    let (status_cmd_tx, mut status_cmd_rx) = mpsc::channel::<String>(4);
    if let Some(ref cmd_str) = status_cmd_str {
        let cmd_str = cmd_str.clone();
        let tx = status_cmd_tx.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(500)).await;
                // Run via shell so pipes/redirects in the command string work.
                let output = if cfg!(target_os = "windows") {
                    tokio::process::Command::new("cmd")
                        .args(["/C", &cmd_str])
                        .output()
                        .await
                } else {
                    tokio::process::Command::new("sh")
                        .args(["-c", &cmd_str])
                        .output()
                        .await
                };
                if let Ok(out) = output {
                    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
                    let _ = tx.try_send(text);
                }
            }
        });
    }

    // Bridge runtime channels — Some when bridge is configured and started.
    //
    // tui_rx:       TUI-facing events from the bridge worker (connect/disconnect/prompts)
    // outbound_tx:  Forward query events to the bridge worker for upload to server
    // bridge_cancel: CancellationToken to stop the bridge worker task
    struct BridgeRuntime {
        tui_rx: mpsc::Receiver<TuiBridgeEvent>,
        outbound_tx: mpsc::Sender<BridgeOutbound>,
        cancel: CancellationToken,
    }

    // Preserve the bridge token before consuming bridge_config so we can reconstruct
    // a BridgeSessionInfo once the bridge worker reports it has connected.
    let bridge_token: Option<String> = bridge_config.as_ref().and_then(|c| c.session_token.clone());

    let mut bridge_runtime: Option<BridgeRuntime> = if let Some(cfg) = bridge_config {
        let bridge_cancel = CancellationToken::new();
        let (tui_tx, tui_rx) = mpsc::channel::<TuiBridgeEvent>(64);
        let (outbound_tx, outbound_rx) = mpsc::channel::<BridgeOutbound>(256);

        // Update TUI state to "connecting" before the task starts.
        app.bridge_state = BridgeConnectionState::Connecting;

        let cancel_clone = bridge_cancel.clone();
        tokio::spawn(async move {
            if let Err(e) =
                mangocode_bridge::run_bridge_loop(cfg, tui_tx, outbound_rx, cancel_clone).await
            {
                warn!("Bridge loop exited with error: {}", e);
            }
        });

        Some(BridgeRuntime {
            tui_rx,
            outbound_tx,
            cancel: bridge_cancel,
        })
    } else {
        None
    };

    // Relay channels for the BridgeSessionInfo-based event path.
    //
    // relay_ev_tx:    receives serialised JSON event payloads from the query-event
    //                 drain loop; a background task consumes them and calls
    //                 post_bridge_event so the web UI sees live streaming events.
    // relay_ev_rx_opt: Option wrapper so we can move the Receiver into the relay
    //                 task exactly once when the bridge session comes online.
    // remote_prompt_tx/rx: inbound user messages polled from poll_bridge_messages
    //                 are delivered here; the main loop injects them as query turns.
    let (relay_ev_tx, relay_ev_rx) = mpsc::channel::<String>(256);
    let mut relay_ev_rx_opt: Option<mpsc::Receiver<String>> = Some(relay_ev_rx);
    let (remote_prompt_tx, mut remote_prompt_rx) = mpsc::channel::<String>(32);

    // Once the bridge worker reports Connected we build this from the session
    // credentials so both relay tasks can POST/poll the /api/bridge/sessions API.
    let mut bridge_session_info: Option<std::sync::Arc<mangocode_bridge::BridgeSessionInfo>> = None;

    let mut messages = initial_messages;
    let mut cmd_ctx = CommandContext {
        config: live_config,
        cost_tracker: cost_tracker.clone(),
        session_metrics: tool_ctx.session_metrics.clone(),
        messages: messages.clone(),
        working_dir: tool_ctx.working_dir.clone(),
        session_id: session.id.clone(),
        session_title: session.title.clone(),
        remote_session_url: session.remote_session_url.clone(),
        mcp_manager: tool_ctx.mcp_manager.clone(),
        model_registry: Some(model_registry.clone()),
    };

    // tools is already Arc<Vec<...>> — share it across spawned tasks without copying.
    let mut tools_arc = tools;

    // Current cancel token (replaced each turn)
    let mut cancel: Option<CancellationToken> = None;
    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<QueryEvent>();
    type MessagesArc = Arc<tokio::sync::Mutex<Vec<mangocode_core::types::Message>>>;
    let mut current_query: Option<(tokio::task::JoinHandle<QueryOutcome>, MessagesArc)> = None;
    // Active effort level (None = use model default / High).
    // Tracks the user's /effort selection; flows into qcfg each turn.
    let mut current_effort: Option<mangocode_core::effort::EffortLevel> = None;

    // Background update check: spawned once at startup; result delivered via channel.
    let (update_tx, mut update_rx) = tokio::sync::mpsc::channel::<Option<String>>(1);
    tokio::spawn(async move {
        let info = mangocode_core::check_for_updates().await;
        let version = info.map(|i| i.latest_version);
        let _ = update_tx.send(version).await;
    });

    // Device code / OAuth auth channel — background tasks send events here
    // so the main loop can update the device_auth_dialog state.
    let (device_auth_tx, mut device_auth_rx) = mpsc::channel::<DeviceAuthEvent>(8);

    'main: loop {
        app.frame_count = app.frame_count.wrapping_add(1);
        app.notifications.tick();

        // Draw the UI
        terminal.draw(|f| render_app(f, &app))?;
        // Flush any pending Sixel mascot blit after ratatui finishes drawing.
        flush_sixel_blit(&app);

        // Poll for crossterm events (keyboard/mouse) with short timeout
        if crossterm::event::poll(Duration::from_millis(16))? {
            let evt = event::read()?;
            match evt {
                Event::Key(key) => {
                    // On Windows crossterm emits Press + Release for a single key.
                    // Only process Press to avoid double-registering input.
                    if key.kind != crossterm::event::KeyEventKind::Press {
                        continue;
                    }

                    // Ctrl+C: copy selected text if there's a selection, otherwise cancel/quit
                    if key.code == KeyCode::Char('c')
                        && key
                            .modifiers
                            .contains(crossterm::event::KeyModifiers::CONTROL)
                    {
                        // Check if there's an active text selection - copy instead of cancel/quit
                        let has_selection = app.selection_anchor.is_some()
                            && !app.selection_text.borrow().is_empty();
                        if has_selection {
                            // Let the app handle the copy via its normal key handler
                            app.handle_key_event(key);
                            continue;
                        }

                        // No selection - handle as cancel (if streaming) or quit
                        if app.is_streaming {
                            if let Some(ref ct) = cancel {
                                ct.cancel();
                            }
                            app.is_streaming = false;
                            app.status_message = Some("Cancelled.".to_string());
                            continue;
                        } else {
                            break 'main;
                        }
                    }

                    // Ctrl+D on empty input => quit
                    if key.code == KeyCode::Char('d')
                        && key
                            .modifiers
                            .contains(crossterm::event::KeyModifiers::CONTROL)
                        && app.prompt_input.is_empty()
                    {
                        break 'main;
                    }

                    // Enter => submit input (but NOT when ANY dialog/overlay is open —
                    // dialogs handle their own Enter in handle_key_event).
                    let any_dialog_open = app.connect_dialog.visible
                        || app.key_input_dialog.visible
                        || app.device_auth_dialog.visible
                        || app.command_palette.visible
                        || app.model_picker.visible
                        || app.onboarding_dialog.visible
                        || app.bypass_permissions_dialog.visible
                        || app.settings_screen.visible
                        || app.export_dialog.visible
                        || app.theme_screen.visible
                        || app.privacy_screen.visible
                        || app.stats_dialog.open
                        || app.invalid_config_dialog.visible
                        || app.context_viz.visible
                        || app.mcp_approval.visible
                        || app.session_browser.visible
                        || app.session_branching.visible
                        || app.tasks_overlay.visible
                        || app.mcp_view.open
                        || app.agents_menu.open
                        || app.diff_viewer.open
                        || app.help_overlay.visible
                        || app.history_search_overlay.visible
                        || app.rewind_flow.visible
                        || app.show_help
                        || app.context_menu_state.is_some()
                        || app.permission_request.is_some()
                        || app.global_search.open;
                    // Esc during streaming cancels the active query turn.
                    if key.code == KeyCode::Esc && app.is_streaming {
                        if let Some(ref ct) = cancel {
                            ct.cancel();
                        }
                        app.is_streaming = false;
                        app.status_message = Some("Cancelled.".to_string());
                        continue;
                    }
                    if key.code == KeyCode::Enter && !app.is_streaming && !any_dialog_open {
                        // If a slash-command suggestion is active, accept and execute immediately.
                        if !app.prompt_input.suggestions.is_empty()
                            && app.prompt_input.suggestion_index.is_some()
                            && app.prompt_input.text.starts_with('/')
                        {
                            app.prompt_input.accept_suggestion();
                            // Fall through to submit — no second Enter needed
                        }

                        let input = app.take_input();
                        if input.is_empty() {
                            continue;
                        }

                        // Check for slash command
                        if input.starts_with('/') {
                            let (cmd_name, cmd_args) =
                                mangocode_tui::input::parse_slash_command(&input);
                            let cmd_name = cmd_name.to_string();
                            let cmd_args = cmd_args.to_string();

                            // ── Step 1: TUI-layer intercept (overlays, toggles) ────────
                            // Run first so we know whether a UI overlay opened, which
                            // lets us suppress redundant CLI text output below.
                            //
                            // Skip TUI overlay for arg-bearing commands where the user
                            // wants to SET state, not browse a picker:
                            //   /model claude-haiku  → set model, don't open picker
                            //   /theme dark          → set theme, don't open picker
                            //   /resume <id>         → load session, don't open browser
                            // Also skip TUI for /vim, /voice, /fast with explicit
                            // on|off args so the blind-toggle doesn't misfire.
                            let skip_tui_for_args = !cmd_args.is_empty()
                                && matches!(
                                    cmd_name.as_str(),
                                    "model"
                                        | "theme"
                                        | "resume"
                                        | "session"
                                        | "vim"
                                        | "vi"
                                        | "voice"
                                        | "fast"
                                        | "speed"
                                );
                            let handled_by_tui = if skip_tui_for_args {
                                false
                            } else {
                                app.intercept_slash_command(&cmd_name)
                            };

                            // Sync effort level when TUI cycled the visual indicator
                            // (no-args /effort → cycle Low→Med→High→Max→Low).
                            if handled_by_tui && cmd_name == "effort" && cmd_args.is_empty() {
                                current_effort = Some(match app.effort_level {
                                    mangocode_tui::EffortLevel::Low => {
                                        mangocode_core::effort::EffortLevel::Low
                                    }
                                    mangocode_tui::EffortLevel::Normal => {
                                        mangocode_core::effort::EffortLevel::Medium
                                    }
                                    mangocode_tui::EffortLevel::High => {
                                        mangocode_core::effort::EffortLevel::High
                                    }
                                    mangocode_tui::EffortLevel::Max => {
                                        mangocode_core::effort::EffortLevel::Max
                                    }
                                });
                            }

                            // Honour exit/quit triggered by TUI intercept immediately.
                            if app.should_quit {
                                break 'main;
                            }

                            // ── Step 2: CLI-layer (real side effects) ──────────────────
                            // Handles: config changes, session ops, file I/O, OAuth, etc.
                            // Always runs — some commands need BOTH (e.g. /clear clears
                            // app state via TUI AND the messages vec via CLI).
                            cmd_ctx.messages = messages.clone();
                            let cli_result = execute_command(&input, &mut cmd_ctx).await;
                            // Start optimistically true; set false for Silent/None below.
                            let mut handled_by_cli = cli_result.is_some();

                            // Whether we need to fall through and submit a user message.
                            let mut submit_user_msg: Option<String> = None;

                            match cli_result {
                                Some(CommandResult::Exit) => break 'main,
                                Some(CommandResult::ClearConversation) => {
                                    messages.clear();
                                    app.replace_messages(Vec::new());
                                    session.messages.clear();
                                    session.updated_at = chrono::Utc::now();
                                    app.status_message = Some("Conversation cleared.".to_string());
                                }
                                Some(CommandResult::SetMessages(new_msgs)) => {
                                    let removed = messages.len().saturating_sub(new_msgs.len());
                                    messages = new_msgs.clone();
                                    app.replace_messages(new_msgs);
                                    session.messages = messages.clone();
                                    session.updated_at = chrono::Utc::now();
                                    app.status_message = Some(format!(
                                        "Rewound {} message{}.",
                                        removed,
                                        if removed == 1 { "" } else { "s" }
                                    ));
                                }
                                Some(CommandResult::OpenRewindOverlay) => {
                                    app.replace_messages(messages.clone());
                                    app.open_rewind_flow();
                                    app.status_message =
                                        Some("Select a message to rewind to.".to_string());
                                }
                                Some(CommandResult::OpenHooksOverlay) => {
                                    // Open the 4-screen hooks configuration browser.
                                    // intercept_slash_command("hooks") already does this
                                    // when the user types /hooks in the TUI prompt, so
                                    // this branch only triggers when the command returns
                                    // the variant explicitly (e.g. from a non-prompt context).
                                    app.hooks_config_menu.open();
                                    app.status_message =
                                        Some("Hooks configuration browser".to_string());
                                }
                                Some(CommandResult::ResumeSession(resumed_session)) => {
                                    session = resumed_session;
                                    messages = session.messages.clone();
                                    app.replace_messages(messages.clone());
                                    cmd_ctx.config.model = Some(session.model.clone());
                                    app.config.model = Some(session.model.clone());
                                    tool_ctx.config.model = Some(session.model.clone());
                                    app.model_name = session.model.clone();
                                    tool_ctx.session_id = session.id.clone();
                                    tool_ctx.file_history = Arc::new(ParkingMutex::new(
                                        mangocode_core::file_history::FileHistory::new(),
                                    ));
                                    tool_ctx.current_turn =
                                        Arc::new(std::sync::atomic::AtomicUsize::new(0));
                                    cmd_ctx.session_id = session.id.clone();
                                    cmd_ctx.session_title = session.title.clone();
                                    if let Some(saved_dir) = session.working_dir.as_ref() {
                                        let saved_path = std::path::PathBuf::from(saved_dir);
                                        if saved_path.exists() {
                                            tool_ctx.working_dir = saved_path.clone();
                                            cmd_ctx.working_dir = saved_path;
                                        }
                                    }
                                    app.config.project_dir = Some(tool_ctx.working_dir.clone());
                                    app.attach_turn_diff_state(
                                        tool_ctx.file_history.clone(),
                                        tool_ctx.current_turn.clone(),
                                    );
                                    app.status_message =
                                        Some(format!("Resumed session {}.", &session.id[..8]));
                                }
                                Some(CommandResult::RenameSession(title)) => {
                                    session.title = Some(title.clone());
                                    session.updated_at = chrono::Utc::now();
                                    cmd_ctx.session_title = session.title.clone();
                                    let _ = mangocode_core::history::save_session(&session).await;
                                    app.status_message =
                                        Some(format!("Session renamed to \"{}\".", title));
                                }
                                Some(CommandResult::Message(msg)) => {
                                    // Suppress text output when TUI already opened an
                                    // overlay for this command (e.g. /stats opens dialog
                                    // AND would push a text message — drop the text).
                                    if !handled_by_tui {
                                        app.push_message(
                                            mangocode_core::types::Message::assistant(msg),
                                        );
                                    }
                                }
                                Some(CommandResult::ConfigChange(new_cfg)) => {
                                    cmd_ctx.config = new_cfg.clone();
                                    tool_ctx.config = new_cfg.clone();
                                    app.config = new_cfg.clone();
                                    // Sync model name shown in the TUI header.
                                    if let Some(ref model) = new_cfg.model {
                                        app.model_name = model.clone();
                                    }
                                    // Sync fast_mode visual indicator.
                                    app.fast_mode = new_cfg
                                        .model
                                        .as_deref()
                                        .map(|m| m.contains("haiku"))
                                        .unwrap_or(false);
                                    // Sync plan_mode visual indicator.
                                    app.plan_mode = matches!(
                                        new_cfg.permission_mode,
                                        mangocode_core::config::PermissionMode::Plan
                                    );
                                    app.status_message = Some("Configuration updated.".to_string());
                                }
                                Some(CommandResult::ConfigChangeMessage(new_cfg, msg)) => {
                                    cmd_ctx.config = new_cfg.clone();
                                    tool_ctx.config = new_cfg.clone();
                                    // Sync model name + fast_mode visual indicator.
                                    if let Some(ref model) = new_cfg.model {
                                        app.model_name = model.clone();
                                        app.fast_mode = model.contains("haiku");
                                    } else {
                                        // model reset to None means fast mode off.
                                        app.fast_mode = false;
                                    }
                                    app.config = new_cfg;
                                    app.status_message = Some(msg);
                                }
                                Some(CommandResult::UserMessage(msg)) => {
                                    // Queue a user-visible turn for the model.
                                    submit_user_msg = Some(msg);
                                }
                                Some(CommandResult::StartOAuthFlow(with_claude_ai)) => {
                                    mangocode_tui::restore_terminal(&mut terminal).ok();
                                    match oauth_flow::run_oauth_login_flow(with_claude_ai).await {
                                        Ok(result) => {
                                            debug!(
                                                credential_len = result.credential.len(),
                                                bearer = result.use_bearer_auth,
                                                "OAuth login complete"
                                            );
                                            app.status_message =
                                                Some("Login successful!".to_string());
                                            eprintln!(
                                                "\nLogin successful! Please restart \
                                                 claude to use the new credentials."
                                            );
                                            break 'main;
                                        }
                                        Err(e) => {
                                            eprintln!("\nLogin failed: {}", e);
                                        }
                                    }
                                    terminal = mangocode_tui::setup_terminal()?;
                                }
                                Some(CommandResult::Error(e)) => {
                                    app.status_message = Some(format!("Error: {}", e));
                                }
                                Some(CommandResult::Silent) | None => {
                                    handled_by_cli = false;
                                }
                            }

                            // Sync effort visual + API level when CLI handled
                            // /effort with explicit args (/effort high).
                            if handled_by_cli && cmd_name == "effort" && !cmd_args.is_empty() {
                                if let Some(level) =
                                    mangocode_core::effort::EffortLevel::parse(&cmd_args)
                                {
                                    current_effort = Some(level);
                                    app.effort_level = match level {
                                        mangocode_core::effort::EffortLevel::Low => {
                                            mangocode_tui::EffortLevel::Low
                                        }
                                        mangocode_core::effort::EffortLevel::Medium => {
                                            mangocode_tui::EffortLevel::Normal
                                        }
                                        mangocode_core::effort::EffortLevel::High => {
                                            mangocode_tui::EffortLevel::High
                                        }
                                        mangocode_core::effort::EffortLevel::Max => {
                                            mangocode_tui::EffortLevel::Max
                                        }
                                    };
                                    app.status_message = Some(format!(
                                        "Effort: {} {}",
                                        app.effort_level.symbol(),
                                        app.effort_level.label(),
                                    ));
                                }
                            }

                            // Sync vim mode when CLI handled /vim with explicit args.
                            if handled_by_cli
                                && matches!(cmd_name.as_str(), "vim" | "vi")
                                && !cmd_args.is_empty()
                            {
                                app.prompt_input.vim_enabled =
                                    matches!(cmd_args.trim(), "on" | "vim");
                            }

                            if !handled_by_cli && !handled_by_tui {
                                app.status_message =
                                    Some(format!("Unknown command: /{}", cmd_name));
                            }

                            // If a UserMessage was queued (e.g. /compact), submit it.
                            if let Some(msg) = submit_user_msg {
                                messages.push(mangocode_core::types::Message::user(msg.clone()));
                                app.push_message(mangocode_core::types::Message::user(msg));
                                // Fall through to the send path below.
                            } else {
                                continue;
                            }
                        }

                        // Fire UserPromptSubmit hook without blocking query launch.
                        let hook_handle = if !config.hooks.is_empty() {
                            let hooks = config.hooks.clone();
                            let working_dir = tool_ctx.working_dir.clone();
                            let hook_ctx = mangocode_core::hooks::HookContext {
                                event: "UserPromptSubmit".to_string(),
                                tool_name: None,
                                tool_input: None,
                                tool_output: Some(input.clone()),
                                is_error: None,
                                session_id: Some(tool_ctx.session_id.clone()),
                            };
                            Some(tokio::spawn(async move {
                                tokio::time::timeout(
                                    Duration::from_secs(5),
                                    mangocode_core::hooks::run_hooks(
                                        &hooks,
                                        mangocode_core::config::HookEvent::UserPromptSubmit,
                                        &hook_ctx,
                                        &working_dir,
                                    ),
                                )
                                .await
                                .unwrap_or_else(|_| {
                                    eprintln!("Hooks timed out after 5s, continuing query");
                                    mangocode_core::hooks::HookOutcome::Allowed
                                })
                            }))
                        } else {
                            None
                        };

                        // Regular user message (with optional image attachments)
                        let pending_imgs = app.prompt_input.clear_images();
                        let user_msg = if pending_imgs.is_empty() {
                            mangocode_core::types::Message::user(input.clone())
                        } else {
                            let mut blocks: Vec<mangocode_core::types::ContentBlock> = pending_imgs
                                .iter()
                                .filter_map(|img| {
                                    mangocode_tui::image_paste::encode_image_base64(&img.path).map(
                                        |b64| mangocode_core::types::ContentBlock::Image {
                                            source: mangocode_core::types::ImageSource {
                                                source_type: "base64".to_string(),
                                                media_type: Some("image/png".to_string()),
                                                data: Some(b64),
                                                url: None,
                                            },
                                        },
                                    )
                                })
                                .collect();
                            blocks.push(mangocode_core::types::ContentBlock::Text {
                                text: input.clone(),
                            });
                            mangocode_core::types::Message::user_blocks(blocks)
                        };
                        messages.push(user_msg.clone());
                        app.push_message(user_msg);
                        session.messages = messages.clone();
                        session.updated_at = chrono::Utc::now();

                        // Start async query
                        app.is_streaming = true;
                        app.streaming_text.clear();

                        let ct = CancellationToken::new();
                        cancel = Some(ct.clone());

                        // Use Arc<Mutex> so the task can write updated messages back
                        let msgs_arc = Arc::new(tokio::sync::Mutex::new(messages.clone()));
                        let msgs_arc_clone = msgs_arc.clone();

                        // Share the Arc so the spawned task can access all tools (incl. MCP).
                        let tools_arc_clone = tools_arc.clone();
                        let ctx_clone = tool_ctx.clone();
                        let mut qcfg = base_query_config.clone();
                        qcfg.model = mangocode_api::effective_model_for_config(
                            &cmd_ctx.config,
                            &model_registry,
                        );
                        qcfg.max_tokens = cmd_ctx.config.effective_max_tokens();
                        qcfg.append_system_prompt = cmd_ctx.config.append_system_prompt.clone();
                        qcfg.system_prompt = base_query_config.system_prompt.clone();
                        qcfg.output_style = cmd_ctx.config.effective_output_style();
                        qcfg.output_style_prompt = cmd_ctx.config.resolve_output_style_prompt();
                        qcfg.working_directory = Some(tool_ctx.working_dir.display().to_string());
                        // Propagate active OAuth provider so system-prompt identity
                        // text reflects the correct product branding (e.g. Claude Max).
                        qcfg.oauth_provider =
                            mangocode_core::system_prompt::OAuthProvider::from_provider_id(
                                cmd_ctx.config.provider.as_deref().unwrap_or(""),
                            );
                        // Apply active effort level (set via /effort command).
                        if let Some(level) = current_effort {
                            qcfg.effort_level = Some(level);
                        }
                        let tracker = cost_tracker.clone();
                        let tx = event_tx.clone();
                        let client_clone = client.clone();

                        let handle = tokio::spawn(async move {
                            let mut msgs = msgs_arc_clone.lock().await.clone();
                            let outcome = mangocode_query::run_query_loop(
                                client_clone.as_ref(),
                                &mut msgs,
                                tools_arc_clone.as_slice(),
                                &ctx_clone,
                                &qcfg,
                                tracker,
                                Some(tx),
                                ct,
                                None,
                            )
                            .await;
                            if let Some(hook_handle) = hook_handle {
                                let _ = hook_handle.await;
                            }
                            // Write updated messages (with tool calls + assistant response) back
                            *msgs_arc_clone.lock().await = msgs;
                            outcome
                        });

                        // Store the Arc so we can read messages after task completes
                        current_query = Some((handle, msgs_arc));
                        continue;
                    }

                    app.handle_key_event(key);
                    cmd_ctx.config = app.config.clone();
                    tool_ctx.config = app.config.clone();
                    if !app.model_name.is_empty() {
                        session.model = app.model_name.clone();
                    }
                    if !app.is_streaming && app.messages.len() < messages.len() {
                        messages = app.messages.clone();
                        session.messages = messages.clone();
                        session.updated_at = chrono::Utc::now();
                    }
                }
                Event::Paste(data) => {
                    if !app.is_streaming && app.permission_request.is_none() {
                        app.prompt_input.paste(&data);
                    }
                }
                Event::Mouse(mouse) => {
                    app.handle_mouse_event(mouse);
                }
                Event::Resize(_, _) => {
                    let _ = terminal.clear();
                    reset_sixel_blit_state();
                    // Regenerate the mascot image at a bounded size for the new viewport.
                    init_mascot(&mut app);
                }
                _ => {}
            }
        }

        // Drain query events in bounded batches so a misbehaving provider
        // cannot starve input/render by flooding the queue.
        const MAX_QUERY_EVENTS_PER_FRAME: usize = 512;
        let mut drained_query_events = 0usize;
        while drained_query_events < MAX_QUERY_EVENTS_PER_FRAME {
            let Ok(evt) = event_rx.try_recv() else { break };
            drained_query_events += 1;

            // Forward to bridge before consuming (clone only what we need).
            if let Some(ref runtime) = bridge_runtime {
                let outbound: Option<BridgeOutbound> = match &evt {
                    QueryEvent::Stream(
                        mangocode_api::AnthropicStreamEvent::ContentBlockDelta {
                            delta: mangocode_api::streaming::ContentDelta::TextDelta { text },
                            index,
                            ..
                        },
                    ) => Some(BridgeOutbound::TextDelta {
                        delta: text.clone(),
                        message_id: format!("msg-{}", index),
                    }),
                    QueryEvent::StreamWithParent {
                        event:
                            mangocode_api::AnthropicStreamEvent::ContentBlockDelta {
                                delta: mangocode_api::streaming::ContentDelta::TextDelta { text },
                                index,
                                ..
                            },
                        ..
                    } => Some(BridgeOutbound::TextDelta {
                        delta: text.clone(),
                        message_id: format!("msg-{}", index),
                    }),
                    QueryEvent::ToolStart {
                        tool_name,
                        tool_id,
                        input_json,
                        ..
                    } => Some(BridgeOutbound::ToolStart {
                        id: tool_id.clone(),
                        name: tool_name.clone(),
                        input_preview: Some(input_json.clone()),
                    }),
                    QueryEvent::ToolEnd {
                        tool_id,
                        result,
                        is_error,
                        ..
                    } => Some(BridgeOutbound::ToolEnd {
                        id: tool_id.clone(),
                        output: result.clone(),
                        is_error: *is_error,
                    }),
                    QueryEvent::TurnComplete {
                        stop_reason, turn, ..
                    } => Some(BridgeOutbound::TurnComplete {
                        message_id: format!("turn-{}", turn),
                        stop_reason: stop_reason.clone(),
                    }),
                    QueryEvent::Error(msg) => Some(BridgeOutbound::Error {
                        message: msg.clone(),
                    }),
                    _ => None,
                };
                if let Some(ob) = outbound {
                    let _ = runtime.outbound_tx.try_send(ob);
                }
            }
            // Also forward to the BridgeSessionInfo relay channel (best-effort).
            // This drives the post_bridge_event relay task spawned on Connected.
            if bridge_session_info.is_some() {
                let relay_payload: Option<String> = match &evt {
                    QueryEvent::Stream(
                        mangocode_api::AnthropicStreamEvent::ContentBlockDelta {
                            delta: mangocode_api::streaming::ContentDelta::TextDelta { text },
                            ..
                        },
                    ) => Some(
                        serde_json::json!({
                            "type": "text_chunk",
                            "text": text,
                        })
                        .to_string(),
                    ),
                    QueryEvent::StreamWithParent {
                        event:
                            mangocode_api::AnthropicStreamEvent::ContentBlockDelta {
                                delta: mangocode_api::streaming::ContentDelta::TextDelta { text },
                                ..
                            },
                        ..
                    } => Some(
                        serde_json::json!({
                            "type": "text_chunk",
                            "text": text,
                        })
                        .to_string(),
                    ),
                    QueryEvent::ToolStart {
                        tool_name,
                        tool_id,
                        input_json,
                        ..
                    } => Some(
                        serde_json::json!({
                            "type": "tool_start",
                            "tool_name": tool_name,
                            "tool_id": tool_id,
                            "input": input_json,
                        })
                        .to_string(),
                    ),
                    QueryEvent::ToolEnd {
                        tool_name,
                        tool_id,
                        result,
                        is_error,
                        ..
                    } => Some(
                        serde_json::json!({
                            "type": "tool_end",
                            "tool_name": tool_name,
                            "tool_id": tool_id,
                            "result": result,
                            "is_error": is_error,
                        })
                        .to_string(),
                    ),
                    _ => None,
                };
                if let Some(payload) = relay_payload {
                    let _ = relay_ev_tx.try_send(payload);
                }
            }
            app.handle_query_event(evt);
        }
        if drained_query_events == MAX_QUERY_EVENTS_PER_FRAME {
            // Keep responsive and continue draining on the next frame.
            tracing::debug!(
                limit = MAX_QUERY_EVENTS_PER_FRAME,
                "Query event drain reached per-frame cap"
            );
        }

        // Drain TUI-facing bridge events.
        let mut disconnect_bridge = false;
        if let Some(runtime) = bridge_runtime.as_mut() {
            loop {
                match runtime.tui_rx.try_recv() {
                    Ok(TuiBridgeEvent::Connected {
                        session_url,
                        session_id: conn_sid,
                    }) => {
                        let short = if session_url.len() > 60 {
                            format!("{}…", &session_url[..60])
                        } else {
                            session_url.clone()
                        };
                        app.bridge_state = BridgeConnectionState::Connected {
                            session_url: session_url.clone(),
                            peer_count: 0,
                        };
                        app.remote_session_url = Some(session_url.clone());
                        cmd_ctx.remote_session_url = Some(session_url.clone());
                        app.notifications.push(
                            NotificationKind::Success,
                            format!("Remote control active: {}", short),
                            Some(5),
                        );
                        // Persist the session URL into the saved session record.
                        session.remote_session_url = Some(session_url.clone());
                        session.updated_at = chrono::Utc::now();
                        let _ = mangocode_core::history::save_session(&session).await;

                        // Wire the BridgeSessionInfo relay so live tool/text events reach
                        // the web UI via /api/bridge/sessions. This runs alongside
                        // run_bridge_loop as a best-effort supplementary delivery path.
                        if let Some(ref token) = bridge_token {
                            let info = std::sync::Arc::new(mangocode_bridge::BridgeSessionInfo {
                                session_id: conn_sid.clone(),
                                session_url: session_url.clone(),
                                token: token.clone(),
                            });
                            bridge_session_info = Some(info.clone());

                            // Relay consumer: moves relay_ev_rx (taken from the Option)
                            // into a background task that calls post_bridge_event per item.
                            if let Some(rx) = relay_ev_rx_opt.take() {
                                let info_relay = info.clone();
                                tokio::spawn(async move {
                                    let mut rx = rx;
                                    while let Some(payload) = rx.recv().await {
                                        let _ = mangocode_bridge::post_bridge_event(
                                            &info_relay,
                                            payload,
                                        )
                                        .await;
                                    }
                                });
                            }

                            // Poll task: periodically calls poll_bridge_messages and
                            // forwards inbound user messages to remote_prompt_tx.
                            let info_poll = info.clone();
                            let poll_tx = remote_prompt_tx.clone();
                            tokio::spawn(async move {
                                let mut since_id: Option<String> = None;
                                loop {
                                    match mangocode_bridge::poll_bridge_messages(
                                        &info_poll,
                                        since_id.as_deref(),
                                    )
                                    .await
                                    {
                                        Ok(msgs) if !msgs.is_empty() => {
                                            for msg in &msgs {
                                                since_id = Some(msg.id.clone());
                                                if msg.role == "user"
                                                    && poll_tx
                                                        .send(msg.content.clone())
                                                        .await
                                                        .is_err()
                                                {
                                                    return;
                                                }
                                            }
                                        }
                                        _ => {}
                                    }
                                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                                }
                            });
                        }
                    }
                    Ok(TuiBridgeEvent::Disconnected { reason }) => {
                        app.bridge_state = BridgeConnectionState::Disconnected;
                        app.remote_session_url = None;
                        cmd_ctx.remote_session_url = None;
                        if let Some(r) = reason {
                            app.notifications.push(
                                NotificationKind::Warning,
                                format!("Bridge disconnected: {}", r),
                                Some(5),
                            );
                        }
                        disconnect_bridge = true;
                        break;
                    }
                    Ok(TuiBridgeEvent::Reconnecting { attempt }) => {
                        app.bridge_state = BridgeConnectionState::Reconnecting { attempt };
                    }
                    Ok(TuiBridgeEvent::InboundPrompt { content, .. }) => {
                        // Inject the remote prompt as if the user typed it, then
                        // trigger submission automatically.
                        app.set_prompt_text(content.clone());
                        // Push as a user message and fire a query immediately.
                        messages.push(mangocode_core::types::Message::user(content.clone()));
                        app.push_message(mangocode_core::types::Message::user(content.clone()));
                        session.messages = messages.clone();
                        session.updated_at = chrono::Utc::now();
                        app.is_streaming = true;
                        app.streaming_text.clear();
                        let ct = CancellationToken::new();
                        cancel = Some(ct.clone());
                        let msgs_arc = Arc::new(tokio::sync::Mutex::new(messages.clone()));
                        let msgs_arc_clone = msgs_arc.clone();
                        let tools_arc_clone = tools_arc.clone();
                        let ctx_clone = tool_ctx.clone();
                        let mut qcfg = base_query_config.clone();
                        qcfg.model = mangocode_api::effective_model_for_config(
                            &cmd_ctx.config,
                            &model_registry,
                        );
                        qcfg.max_tokens = cmd_ctx.config.effective_max_tokens();
                        qcfg.oauth_provider =
                            mangocode_core::system_prompt::OAuthProvider::from_provider_id(
                                cmd_ctx.config.provider.as_deref().unwrap_or(""),
                            );
                        let tracker = cost_tracker.clone();
                        let tx = event_tx.clone();
                        let client_clone = client.clone();
                        let handle = tokio::spawn(async move {
                            let mut msgs = msgs_arc_clone.lock().await.clone();
                            let outcome = mangocode_query::run_query_loop(
                                client_clone.as_ref(),
                                &mut msgs,
                                tools_arc_clone.as_slice(),
                                &ctx_clone,
                                &qcfg,
                                tracker,
                                Some(tx),
                                ct,
                                None,
                            )
                            .await;
                            *msgs_arc_clone.lock().await = msgs;
                            outcome
                        });
                        current_query = Some((handle, msgs_arc));
                    }
                    Ok(TuiBridgeEvent::Cancelled) => {
                        if app.is_streaming {
                            if let Some(ref ct) = cancel {
                                ct.cancel();
                            }
                            app.is_streaming = false;
                            app.status_message = Some("Cancelled by remote control.".to_string());
                        }
                    }
                    Ok(TuiBridgeEvent::PermissionResponse {
                        tool_use_id,
                        response,
                    }) => {
                        // Resolve a pending permission dialog if IDs match.
                        if let Some(ref pr) = app.permission_request {
                            if pr.tool_use_id == tool_use_id {
                                use mangocode_bridge::PermissionResponseKind;
                                let _allow = matches!(
                                    response,
                                    PermissionResponseKind::Allow
                                        | PermissionResponseKind::AllowSession
                                );
                                app.permission_request = None;
                            }
                        }
                    }
                    Ok(TuiBridgeEvent::SessionNameUpdate { title }) => {
                        session.title = Some(title.clone());
                        session.updated_at = chrono::Utc::now();
                        cmd_ctx.session_title = Some(title.clone());
                        app.session_title = Some(title);
                        let _ = mangocode_core::history::save_session(&session).await;
                    }
                    Ok(TuiBridgeEvent::Error(msg)) => {
                        app.bridge_state = BridgeConnectionState::Failed {
                            reason: msg.clone(),
                        };
                        app.notifications.push(
                            NotificationKind::Warning,
                            format!("Bridge error: {}", msg),
                            Some(5),
                        );
                        disconnect_bridge = true;
                        break;
                    }
                    Ok(TuiBridgeEvent::Ping) => {
                        // No TUI action needed; pong is handled inside run_bridge_loop.
                    }
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                        app.bridge_state = BridgeConnectionState::Disconnected;
                        app.remote_session_url = None;
                        cmd_ctx.remote_session_url = None;
                        app.notifications.push(
                            NotificationKind::Warning,
                            "Remote control connection lost.".to_string(),
                            Some(5),
                        );
                        disconnect_bridge = true;
                        break;
                    }
                }
            }
        }
        if disconnect_bridge {
            bridge_runtime = None;
        }

        // Drain inbound prompts from the BridgeSessionInfo poll task.
        // These are user messages received from the web UI via poll_bridge_messages
        // and injected here just like TuiBridgeEvent::InboundPrompt.
        while let Ok(content) = remote_prompt_rx.try_recv() {
            if !app.is_streaming {
                app.set_prompt_text(content.clone());
                messages.push(mangocode_core::types::Message::user(content.clone()));
                app.push_message(mangocode_core::types::Message::user(content.clone()));
                session.messages = messages.clone();
                session.updated_at = chrono::Utc::now();
                app.is_streaming = true;
                app.streaming_text.clear();
                let ct = CancellationToken::new();
                cancel = Some(ct.clone());
                let msgs_arc = Arc::new(tokio::sync::Mutex::new(messages.clone()));
                let msgs_arc_clone = msgs_arc.clone();
                let tools_arc_clone = tools_arc.clone();
                let ctx_clone = tool_ctx.clone();
                let mut qcfg = base_query_config.clone();
                qcfg.model =
                    mangocode_api::effective_model_for_config(&cmd_ctx.config, &model_registry);
                qcfg.max_tokens = cmd_ctx.config.effective_max_tokens();
                qcfg.oauth_provider =
                    mangocode_core::system_prompt::OAuthProvider::from_provider_id(
                        cmd_ctx.config.provider.as_deref().unwrap_or(""),
                    );
                let tracker = cost_tracker.clone();
                let tx = event_tx.clone();
                let client_clone = client.clone();
                let handle = tokio::spawn(async move {
                    let mut msgs = msgs_arc_clone.lock().await.clone();
                    let outcome = mangocode_query::run_query_loop(
                        client_clone.as_ref(),
                        &mut msgs,
                        tools_arc_clone.as_slice(),
                        &ctx_clone,
                        &qcfg,
                        tracker,
                        Some(tx),
                        ct,
                        None,
                    )
                    .await;
                    *msgs_arc_clone.lock().await = msgs;
                    outcome
                });
                current_query = Some((handle, msgs_arc));
                break; // process one prompt per frame
            }
        }

        // Drain CLAUDE_STATUS_COMMAND results (most recent wins)
        if status_cmd_str.is_some() {
            while let Ok(text) = status_cmd_rx.try_recv() {
                app.status_line_override = if text.is_empty() { None } else { Some(text) };
            }
        }

        // Check if the background update task has reported a result.
        if app.update_available.is_none() {
            if let Ok(Some(version)) = update_rx.try_recv() {
                app.update_available = Some(version);
            }
        }

        // ---- Device code / OAuth auth: spawn background task when pending ----
        if let Some(provider_id) = app.device_auth_pending.take() {
            let _tx = device_auth_tx.clone();
            match provider_id.as_str() {
                "github-copilot" => {
                    let tx2 = device_auth_tx.clone();
                    // Use MangoCode's GitHub Copilot device flow app so the returned
                    // token stays bound to MangoCode's own OAuth registration.
                    const COPILOT_CLIENT_ID: &str = "Iv23li4E44oPZR1huPU9";
                    tokio::spawn(async move {
                        // Step 1: Request device code
                        match mangocode_core::device_code::request_device_code(
                            COPILOT_CLIENT_ID,
                            "read:user",
                            "https://github.com/login/device/code",
                        )
                        .await
                        {
                            Ok(resp) => {
                                let _ = tx2
                                    .send(DeviceAuthEvent::GotCode {
                                        user_code: resp.user_code,
                                        verification_uri: resp.verification_uri,
                                        device_code: resp.device_code.clone(),
                                        interval: resp.interval,
                                    })
                                    .await;
                                // Step 2: Poll for access token
                                match mangocode_core::device_code::poll_for_token(
                                    COPILOT_CLIENT_ID,
                                    &resp.device_code,
                                    "https://github.com/login/oauth/access_token",
                                    resp.interval,
                                    300,
                                )
                                .await
                                {
                                    Ok(token) => {
                                        let _ =
                                            tx2.send(DeviceAuthEvent::TokenReceived(token)).await;
                                    }
                                    Err(e) => {
                                        let _ = tx2.send(DeviceAuthEvent::Error(e)).await;
                                    }
                                }
                            }
                            Err(e) => {
                                let _ = tx2.send(DeviceAuthEvent::Error(e)).await;
                            }
                        }
                    });
                }
                "anthropic-max" => {
                    let tx2 = device_auth_tx.clone();
                    // Claude Max (OAuth) — PKCE flow using Claude Code's registered client ID.
                    // run_oauth_login_flow(true) → claude.ai Bearer-token path (Max subscription).
                    tokio::spawn(async move {
                        // Signal the dialog to enter browser-waiting state
                        let placeholder_url = "Opening browser for Claude authentication…".to_string();
                        let _ = tx2
                            .send(DeviceAuthEvent::GotBrowserUrl {
                                url: placeholder_url,
                            })
                            .await;

                        match crate::oauth_flow::run_oauth_login_flow(true).await {
                            Ok(result) => {
                                // Persist into AuthStore under "anthropic-max"
                                let mut store = mangocode_core::AuthStore::load();
                                // Unwrap tokens: we need refresh + expiry for full storage.
                                // access_token is in result.credential when use_bearer_auth=true.
                                let (refresh_tok, expires_u64) = (
                                    result
                                        .tokens
                                        .refresh_token
                                        .clone()
                                        .unwrap_or_default(),
                                    result
                                        .tokens
                                        .expires_at_ms
                                        .map(|ms| ms as u64)
                                        .unwrap_or(0),
                                );
                                store.set(
                                    mangocode_core::ProviderId::ANTHROPIC_MAX,
                                    mangocode_core::auth_store::StoredCredential::OAuthToken {
                                        access: result.credential.clone(),
                                        refresh: refresh_tok,
                                        expires: expires_u64,
                                    },
                                );
                                let _ = tx2
                                    .send(DeviceAuthEvent::TokenReceived(result.credential))
                                    .await;
                            }
                            Err(e) => {
                                let _ = tx2.send(DeviceAuthEvent::Error(e.to_string())).await;
                            }
                        }
                    });
                }
                "anthropic" => {
                    let tx2 = device_auth_tx.clone();
                    // Anthropic OAuth requires a registered application.
                    // MangoCode does not have its own registered OAuth app with Anthropic.
                    // Users should use an API key from console.anthropic.com instead.
                    tokio::spawn(async move {
                        let _ = tx2
                            .send(DeviceAuthEvent::Error(
                                "Anthropic OAuth requires a registered application.\n\
                             Use an API key instead: console.anthropic.com/settings/keys"
                                    .to_string(),
                            ))
                            .await;
                    });
                }
                _ => {
                    // Unknown provider for device auth — should not happen
                    app.device_auth_dialog
                        .set_error(format!("Unsupported auth flow for {}", provider_id));
                }
            }
        }

        // ---- Drain device auth events from the background task ----
        while let Ok(evt) = device_auth_rx.try_recv() {
            match evt {
                DeviceAuthEvent::GotCode {
                    user_code,
                    verification_uri,
                    device_code,
                    interval,
                } => {
                    // Auto-copy the user code to clipboard
                    let _ = mangocode_tui::try_copy_to_clipboard(&user_code);

                    // Auto-open the verification URL in the browser
                    let _ = open::that(&verification_uri);

                    app.device_auth_dialog.set_code(
                        user_code,
                        verification_uri,
                        device_code,
                        interval,
                    );

                    app.notifications.push(
                        mangocode_tui::NotificationKind::Info,
                        "Code copied to clipboard & browser opened.".to_string(),
                        Some(4),
                    );
                }
                DeviceAuthEvent::GotBrowserUrl { url } => {
                    let _ = mangocode_tui::try_copy_to_clipboard(&url);
                    app.device_auth_dialog.set_browser_url(url);
                    app.notifications.push(
                        mangocode_tui::NotificationKind::Info,
                        "Login URL copied to clipboard.".to_string(),
                        Some(5),
                    );
                }
                DeviceAuthEvent::TokenReceived(token) => {
                    app.device_auth_dialog.set_success(token);
                }
                DeviceAuthEvent::Error(msg) => {
                    app.device_auth_dialog.set_error(msg);
                }
            }
        }

        // Check if query task is done; sync messages from the task
        let task_finished = current_query
            .as_ref()
            .map(|(h, _)| h.is_finished())
            .unwrap_or(false);

        if task_finished {
            let synced_messages = current_query.as_ref().and_then(|(_, msgs_arc)| {
                if let Ok(msgs) = msgs_arc.try_lock() {
                    Some(msgs.clone())
                } else {
                    None
                }
            });

            if let Some(synced_messages) = synced_messages {
                if let Some((handle, _msgs_arc)) = current_query.take() {
                    // Get the outcome (ignore errors for now)
                    let _ = handle.await;
                    // Sync the updated conversation back to our local vector
                    messages = synced_messages;
                    session.messages = messages.clone();
                    session.updated_at = chrono::Utc::now();
                    session.model =
                        mangocode_api::effective_model_for_config(&cmd_ctx.config, &model_registry);
                    session.working_dir = Some(tool_ctx.working_dir.display().to_string());
                    app.is_streaming = false;
                    // Sync the authoritative message list (with ToolUse / ToolResult
                    // content blocks) from the query task into the TUI so they render
                    // inline in chronological order. The transient `tool_use_blocks`
                    // staging area is cleared since the canonical history now owns them.
                    app.replace_messages(messages.clone());
                    app.tool_use_blocks.clear();
                    app.status_message = None;

                    // Persist session and search index in background so UI loop stays responsive.
                    let session_clone = session.clone();
                    tokio::spawn(async move {
                        if let Err(e) = mangocode_core::history::save_session(&session_clone).await {
                            eprintln!("Session save failed: {e}");
                        }

                        if let Err(e) = tokio::task::spawn_blocking(move || {
                            let db_path =
                                mangocode_core::config::Settings::config_dir().join("sessions.db");
                            match mangocode_core::SqliteSessionStore::open(&db_path) {
                                Ok(store) => {
                                    if let Err(e) = store.save_session(
                                        &session_clone.id,
                                        session_clone.title.as_deref(),
                                        &session_clone.model,
                                    ) {
                                        eprintln!("SQLite session index failed: {e}");
                                    }

                                    for msg in &session_clone.messages {
                                        let content_str = match &msg.content {
                                            mangocode_core::types::MessageContent::Text(t) => t.clone(),
                                            mangocode_core::types::MessageContent::Blocks(blocks) => {
                                                blocks
                                                    .iter()
                                                    .filter_map(|b| {
                                                        if let mangocode_core::types::ContentBlock::Text {
                                                            text,
                                                        } = b
                                                        {
                                                            Some(text.as_str())
                                                        } else {
                                                            None
                                                        }
                                                    })
                                                    .collect::<Vec<_>>()
                                                    .join(" ")
                                            }
                                        };
                                        let role = match msg.role {
                                            mangocode_core::types::Role::User => "user",
                                            mangocode_core::types::Role::Assistant => "assistant",
                                        };
                                        let msg_id = msg.uuid.as_deref().unwrap_or("unknown");
                                        if let Err(e) = store.save_message(
                                            &session_clone.id,
                                            msg_id,
                                            role,
                                            &content_str,
                                            None,
                                        ) {
                                            eprintln!("SQLite message index failed: {e}");
                                        }
                                    }
                                }
                                Err(e) => eprintln!("SQLite open failed: {e}"),
                            }
                        })
                        .await
                        {
                            eprintln!("SQLite indexing task join failed: {e}");
                        }
                    });
                }
            }
        }

        if !app.is_streaming && current_query.is_none() && app.take_pending_mcp_reconnect() {
            let new_mcp_manager = connect_mcp_manager_arc(&cmd_ctx.config).await;
            tool_ctx.mcp_manager = new_mcp_manager.clone();
            app.mcp_manager = new_mcp_manager.clone();
            tools_arc = build_tools_with_mcp(
                new_mcp_manager.clone(),
                &cmd_ctx.config.allowed_tools,
                &cmd_ctx.config.disallowed_tools,
            );
            if app.mcp_view.open {
                app.refresh_mcp_view();
            }

            let connected = new_mcp_manager
                .as_ref()
                .map(|manager| manager.server_count())
                .unwrap_or(0);
            app.status_message = Some(if cmd_ctx.config.mcp_servers.is_empty() {
                "No MCP servers configured.".to_string()
            } else {
                format!(
                    "Reconnected MCP runtime ({} connected server{}).",
                    connected,
                    if connected == 1 { "" } else { "s" }
                )
            });
        }

        if app.should_quit {
            break 'main;
        }
    }

    if let Some(runtime) = bridge_runtime.take() {
        runtime.cancel.cancel();
    }

    session.total_cost = cost_tracker.total_cost_usd();
    session.total_tokens = cost_tracker.input_tokens() + cost_tracker.output_tokens();
    session.updated_at = chrono::Utc::now();
    let _ = mangocode_core::history::save_session(&session).await;
    let duration_ms = session_start.elapsed().as_millis() as u64;
    persist_session_usage(&cost_tracker, &session, duration_ms);

    restore_terminal(&mut terminal)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// `claude auth` subcommand handler
// ---------------------------------------------------------------------------
// Mirrors TypeScript cli.tsx `if (args[0] === 'auth') { ... }` fast-path.
// Called before Cli::parse() so it doesn't conflict with positional `prompt`.
//
// Usage:
//   claude auth login [--console]   — OAuth PKCE login (claude.ai by default)
//   claude auth logout              — Clear stored credentials
//   claude auth status [--json]     — Show authentication status

async fn handle_auth_command(args: &[String]) -> anyhow::Result<()> {
    match args.first().map(|s| s.as_str()) {
        Some("login") => {
            // --console flag selects the Console OAuth flow (creates an API key)
            // Default (no flag) uses the Claude.ai flow (Bearer token)
            let login_with_claude_ai = !args.iter().any(|a| a == "--console");
            println!("Starting authentication...");
            match oauth_flow::run_oauth_login_flow(login_with_claude_ai).await {
                Ok(result) => {
                    debug!(
                        credential_len = result.credential.len(),
                        bearer = result.use_bearer_auth,
                        "OAuth login complete"
                    );
                    println!("Successfully logged in!");
                    if let Some(email) = &result.tokens.email {
                        println!("  Account: {}", email);
                    }
                    if result.use_bearer_auth {
                        println!("  Auth method: claude.ai");
                    } else {
                        println!("  Auth method: console (API key)");
                    }
                    std::process::exit(0);
                }
                Err(e) => {
                    eprintln!("Login failed: {}", e);
                    std::process::exit(1);
                }
            }
        }

        Some("logout") => {
            auth_logout().await;
        }

        Some("status") => {
            let json_output = args.iter().any(|a| a == "--json");
            auth_status(json_output).await;
        }

        Some(unknown) => {
            eprintln!("Unknown auth subcommand: '{}'", unknown);
            eprintln!();
            eprintln!("Usage: claude auth <subcommand>");
            eprintln!(
                "  login [--console]   Authenticate (claude.ai by default; --console for API key)"
            );
            eprintln!("  logout              Remove stored credentials");
            eprintln!("  status [--json]     Show authentication status");
            std::process::exit(1);
        }

        None => {
            eprintln!("Usage: claude auth <login|logout|status>");
            eprintln!("  login [--console]   Authenticate with Anthropic");
            eprintln!("  logout              Remove stored credentials");
            eprintln!("  status [--json]     Show authentication status");
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Print current auth status, then exit with code 0 (logged in) or 1 (not logged in).
async fn auth_status(json_output: bool) {
    // Gather auth state
    let env_api_key = std::env::var("ANTHROPIC_API_KEY")
        .ok()
        .filter(|k| !k.is_empty());
    let settings = Settings::load().await.unwrap_or_default();
    let settings_api_key = settings.config.api_key.clone().filter(|k| !k.is_empty());
    let oauth_tokens = mangocode_core::oauth::OAuthTokens::load().await;
    let api_provider = "Anthropic";
    let api_key_source = if env_api_key.is_some() {
        Some("ANTHROPIC_API_KEY".to_string())
    } else if settings_api_key.is_some() {
        Some("settings".to_string())
    } else if oauth_tokens
        .as_ref()
        .is_some_and(|tokens| !tokens.uses_bearer_auth() && tokens.api_key.is_some())
    {
        Some("/login managed key".to_string())
    } else {
        None
    };
    let token_source = oauth_tokens.as_ref().map(|tokens| {
        if tokens.uses_bearer_auth() {
            "claude.ai".to_string()
        } else {
            "console_oauth".to_string()
        }
    });
    let login_method = oauth_tokens
        .as_ref()
        .and_then(|tokens| subscription_label(tokens.subscription_type.as_deref()))
        .or_else(|| {
            oauth_tokens.as_ref().map(|tokens| {
                if tokens.uses_bearer_auth() {
                    "MangoCode Account".to_string()
                } else {
                    "Console Account".to_string()
                }
            })
        })
        .or_else(|| api_key_source.as_ref().map(|_| "API Key".to_string()));
    let billing_mode = oauth_tokens.as_ref().map_or_else(
        || {
            if api_key_source.is_some() {
                "API".to_string()
            } else {
                "None".to_string()
            }
        },
        |tokens| {
            if tokens.uses_bearer_auth() {
                "Subscription".to_string()
            } else {
                "API".to_string()
            }
        },
    );

    // Determine auth method (mirrors TypeScript authStatus())
    let (auth_method, logged_in) = if let Some(ref tokens) = oauth_tokens {
        let uses_bearer = tokens.uses_bearer_auth();
        let method = if uses_bearer {
            "claude.ai"
        } else {
            "oauth_token"
        };
        (method.to_string(), true)
    } else if env_api_key.is_some() || settings_api_key.is_some() {
        ("api_key".to_string(), true)
    } else {
        ("none".to_string(), false)
    };

    if json_output {
        // JSON output (used by SDK + scripts)
        let mut obj = serde_json::json!({
            "loggedIn": logged_in,
            "authMethod": auth_method,
            "apiProvider": api_provider,
            "billing": billing_mode,
        });

        // Include API key source if known
        if let Some(ref source) = api_key_source {
            obj["apiKeySource"] = serde_json::Value::String(source.clone());
        }
        if let Some(ref source) = token_source {
            obj["tokenSource"] = serde_json::Value::String(source.clone());
        }
        if let Some(ref method) = login_method {
            obj["loginMethod"] = serde_json::Value::String(method.clone());
        }

        if let Some(ref tokens) = oauth_tokens {
            obj["email"] = json_null_or_string(&tokens.email);
            obj["orgId"] = json_null_or_string(&tokens.organization_uuid);
            obj["subscriptionType"] = json_null_or_string(&tokens.subscription_type);
        }

        println!("{}", serde_json::to_string_pretty(&obj).unwrap_or_default());
    } else {
        // Human-readable text output
        if !logged_in {
            println!("Not logged in. Run `claude auth login` to authenticate.");
        } else {
            println!("Logged in.");
            println!("  API provider: {}", api_provider);
            println!("  Billing: {}", billing_mode);
            if let Some(ref method) = login_method {
                println!("  Login method: {}", method);
            }
            if let Some(ref source) = token_source {
                println!("  Auth token: {}", source);
            }
            if let Some(ref source) = api_key_source {
                println!("  API key: {}", source);
            }
            match auth_method.as_str() {
                "claude.ai" | "oauth_token" => {
                    if let Some(ref tokens) = oauth_tokens {
                        if let Some(ref email) = tokens.email {
                            println!("  Email: {}", email);
                        }
                        if let Some(ref org) = tokens.organization_uuid {
                            println!("  Organization ID: {}", org);
                        } else {
                            println!("  Organization ID: unavailable");
                        }
                        if let Some(ref sub) = tokens.subscription_type {
                            println!("  Subscription: {}", sub);
                        }
                    }
                }
                "api_key" => {
                    println!("  Organization ID: unavailable for direct API key auth");
                }
                _ => {}
            }
        }
    }

    std::process::exit(if logged_in { 0 } else { 1 });
}

/// Clear all stored credentials and exit.
async fn auth_logout() {
    let mut had_error = false;

    // Clear OAuth tokens
    if let Err(e) = mangocode_core::oauth::OAuthTokens::clear().await {
        eprintln!("Warning: failed to clear OAuth tokens: {}", e);
        had_error = true;
    }

    // Also clear any API key stored in settings.json
    match Settings::load().await {
        Ok(mut settings) => {
            if settings.config.api_key.is_some() {
                settings.config.api_key = None;
                if let Err(e) = settings.save().await {
                    eprintln!("Warning: failed to update settings.json: {}", e);
                    had_error = true;
                }
            }
        }
        Err(e) => {
            eprintln!("Warning: failed to load settings.json: {}", e);
        }
    }

    if had_error {
        eprintln!("Logout completed with warnings.");
        std::process::exit(1);
    } else {
        println!("Successfully logged out from your Anthropic account.");
        std::process::exit(0);
    }
}

/// Helper: convert `Option<String>` to a JSON string or null.
fn subscription_label(subscription_type: Option<&str>) -> Option<String> {
    match subscription_type? {
        "enterprise" => Some("MangoCode Enterprise Account".to_string()),
        "team" => Some("MangoCode Team Account".to_string()),
        "max" => Some("MangoCode Max Account".to_string()),
        "pro" => Some("MangoCode Pro Account".to_string()),
        other if !other.is_empty() => Some(format!("{} Account", other)),
        _ => None,
    }
}

/// Helper: convert `Option<String>` to a JSON string or null.
fn json_null_or_string(opt: &Option<String>) -> serde_json::Value {
    match opt {
        Some(s) => serde_json::Value::String(s.clone()),
        None => serde_json::Value::Null,
    }
}
