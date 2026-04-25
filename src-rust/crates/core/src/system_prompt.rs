//! Modular system prompt assembly with caching support.
//!
//! Cacheable (static) sections are placed before `SYSTEM_PROMPT_DYNAMIC_BOUNDARY`;
//! volatile, session-specific sections follow it.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

// ---------------------------------------------------------------------------
// Dynamic boundary marker
// ---------------------------------------------------------------------------

/// Marker that splits the cached vs dynamic parts of the system prompt.
/// Everything before this marker can be prompt-cached by the API.
/// Stable marker string used to split cached vs dynamic prompt sections.
pub const SYSTEM_PROMPT_DYNAMIC_BOUNDARY: &str = "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__";

// ---------------------------------------------------------------------------
// Section cache (per-section memoization for prompt caching)
// ---------------------------------------------------------------------------

fn section_cache() -> &'static Mutex<HashMap<String, Option<String>>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Option<String>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Clear all cached system prompt sections (called on /clear and /compact).
pub fn clear_system_prompt_sections() {
    if let Ok(mut cache) = section_cache().lock() {
        cache.clear();
    }
}

/// A single named section of the system prompt.
#[derive(Debug, Clone)]
pub struct SystemPromptSection {
    /// Identifier used for cache lookups and invalidation.
    pub tag: &'static str,
    /// Computed content (None means the section is absent/disabled).
    pub content: Option<String>,
    /// If true the section is volatile and must not be prompt-cached.
    pub cache_break: bool,
}

impl SystemPromptSection {
    /// Create a memoizable (cacheable) section.
    pub fn cached(tag: &'static str, content: impl Into<String>) -> Self {
        Self {
            tag,
            content: Some(content.into()),
            cache_break: false,
        }
    }

    /// Create a volatile section that re-evaluates every turn.
    /// Passing `None` for content means the section is absent this turn.
    pub fn uncached(tag: &'static str, content: Option<impl Into<String>>) -> Self {
        Self {
            tag,
            content: content.map(|c| c.into()),
            cache_break: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Output style
// ---------------------------------------------------------------------------

/// Output styles that affect the system prompt.
/// Serialised as lowercase strings to match settings.json.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OutputStyle {
    #[default]
    Default,
    Explanatory,
    Learning,
    Concise,
    Formal,
    Casual,
}

impl OutputStyle {
    /// Returns the system-prompt suffix for this style, or `None` for Default.
    pub fn prompt_suffix(self) -> Option<&'static str> {
        match self {
            OutputStyle::Explanatory => Some(
                "When explaining code or concepts, be thorough and educational. \
                Include reasoning, alternatives considered, and potential pitfalls. \
                Err on the side of over-explaining.",
            ),
            OutputStyle::Learning => Some(
                "This user is learning. Explain concepts as you implement them. \
                Point out patterns, best practices, and why you made each decision. \
                Use analogies when helpful.",
            ),
            OutputStyle::Concise => Some(
                "Be maximally concise. Skip preamble, summaries, and filler. \
                Lead with the answer. One sentence is better than three.",
            ),
            OutputStyle::Formal => {
                Some("Maintain a formal, professional tone. Use precise technical language.")
            }
            OutputStyle::Casual => Some("Use a casual, conversational tone."),
            OutputStyle::Default => None,
        }
    }

    /// Parse from a string (case-insensitive).
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "explanatory" => Self::Explanatory,
            "learning" => Self::Learning,
            "concise" => Self::Concise,
            "formal" => Self::Formal,
            "casual" => Self::Casual,
            _ => Self::Default,
        }
    }
}

// ---------------------------------------------------------------------------
// System prompt prefix variants
// ---------------------------------------------------------------------------

/// Which entrypoint context MangoCode is running in.
/// Determines the opening attribution line of the system prompt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SystemPromptPrefix {
    /// Standard interactive CLI session.
    Cli,
    /// Running as a sub-agent spawned by the Claude Agent SDK.
    Sdk,
    /// The CLI preset running within the Agent SDK
    /// (non-interactive + append_system_prompt set).
    SdkPreset,
    /// Running on Vertex AI.
    Vertex,
    /// Running on AWS Bedrock.
    Bedrock,
    /// Remote / headless CCR session.
    Remote,
}

impl SystemPromptPrefix {
    /// Detect from environment variables (Vertex, Bedrock, remote, SDK vs interactive).
    pub fn detect(is_non_interactive: bool, has_append_system_prompt: bool) -> Self {
        // Vertex: always uses the default "MangoCode" prefix.
        if std::env::var("ANTHROPIC_VERTEX_PROJECT_ID").is_ok()
            || std::env::var("CLOUD_ML_PROJECT_ID").is_ok()
        {
            return Self::Vertex;
        }

        if std::env::var("AWS_BEDROCK_MODEL_ID").is_ok() {
            return Self::Bedrock;
        }

        if std::env::var("MANGOCODE_REMOTE").is_ok() {
            return Self::Remote;
        }

        // Non-interactive mode maps to SDK variants.
        if is_non_interactive {
            if has_append_system_prompt {
                return Self::SdkPreset;
            }
            return Self::Sdk;
        }

        Self::Cli
    }

    /// The opening attribution string for this prefix variant.
    pub fn attribution_text(self) -> &'static str {
        match self {
            Self::Cli | Self::Vertex | Self::Bedrock | Self::Remote => {
                "You are MangoCode, Anthropic's official CLI for Claude."
            }
            Self::SdkPreset => {
                "You are MangoCode, Anthropic's official CLI for Claude, \
                running within the Claude Agent SDK."
            }
            Self::Sdk => "You are a Claude agent, built on Anthropic's Claude Agent SDK.",
        }
    }

    /// Opener used when an Anthropic Claude Max OAuth session is active.
    ///
    /// Anthropic's OAuth endpoint validates the first system-prompt block
    /// against these exact strings; any divergence triggers the
    /// "OAuth authentication is currently not supported" rejection. Three
    /// variants are emitted, selected by the active `SystemPromptPrefix`:
    ///
    ///   * Default opener        — interactive CLI / Vertex / Bedrock / Remote
    ///   * SDK preset opener     — non-interactive + `--append-system-prompt`
    ///   * SDK opener            — non-interactive, no append
    pub fn claude_code_attribution_text(self) -> &'static str {
        match self {
            Self::Cli | Self::Vertex | Self::Bedrock | Self::Remote => {
                "You are Claude Code, Anthropic's official CLI for Claude."
            }
            Self::SdkPreset => {
                "You are Claude Code, Anthropic's official CLI for Claude, running within the Claude Agent SDK."
            }
            Self::Sdk => "You are a Claude agent, built on Anthropic's Claude Agent SDK.",
        }
    }
}

// ---------------------------------------------------------------------------
// OAuth provider identity — dynamic system prompt variable
// ---------------------------------------------------------------------------

/// Which OAuth provider is active for the current session.
///
/// When the user authenticates via an OAuth provider (e.g. Claude Max, Codex),
/// the system prompt attribution line is updated to reflect the official
/// product identity required by Anthropic when using Claude Max OAuth.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum OAuthProvider {
    /// No OAuth provider active — use default MangoCode identity.
    #[default]
    None,
    /// Anthropic Claude Max (claude.ai OAuth, user:inference scope).
    /// Uses Anthropic's required product identity in the system prompt.
    AnthropicMax,
    /// OpenAI Codex (chatgpt.com OAuth).
    OpenAiCodex,
    /// GitHub Copilot (device code flow).
    GitHubCopilot,
}

impl OAuthProvider {
    /// Returns the system prompt identity/attribution string for this provider.
    ///
    /// When an OAuth provider is active, we use the official product identity
    /// (e.g. Claude Max wording) so the model receives the correct persona context.
    pub fn identity_text(self) -> &'static str {
        match self {
            Self::None => "You are MangoCode, a powerful coding assistant built on Claude.",
            // Anthropic's OAuth endpoint requires this exact opener; any extra
            // wording triggers an "OAuth authentication is currently not
            // supported" rejection. SDK / non-interactive variants are
            // selected via `SystemPromptPrefix::claude_code_attribution_text`
            // inside `build_system_prompt`.
            Self::AnthropicMax => "You are Claude Code, Anthropic's official CLI for Claude.",
            Self::OpenAiCodex => {
                "You are MangoCode, running with an OpenAI Codex subscription. \
                You have access to OpenAI's Codex models for code generation."
            }
            Self::GitHubCopilot => {
                "You are MangoCode, running with GitHub Copilot. \
                You have access to GitHub Copilot's models for code assistance."
            }
        }
    }

    /// Detect from a provider ID string (e.g. from auth_store key).
    pub fn from_provider_id(id: &str) -> Self {
        match id {
            "anthropic-max" => Self::AnthropicMax,
            "openai-codex" | "codex" => Self::OpenAiCodex,
            "github-copilot" => Self::GitHubCopilot,
            _ => Self::None,
        }
    }
}

// ---------------------------------------------------------------------------
// Build options
// ---------------------------------------------------------------------------

/// All options controlling what goes into the assembled system prompt.
#[derive(Debug, Clone, Default)]
pub struct SystemPromptOptions {
    /// Override auto-detected prefix.
    pub prefix: Option<SystemPromptPrefix>,
    /// Active OAuth provider (affects attribution identity).
    pub oauth_provider: OAuthProvider,
    /// Whether the session is non-interactive (SDK / pipe mode).
    pub is_non_interactive: bool,
    /// Whether --append-system-prompt is set (affects prefix detection).
    pub has_append_system_prompt: bool,
    /// Output style to inject.
    pub output_style: OutputStyle,
    /// Optional custom output-style prompt loaded from disk or plugins.
    /// When present, this overrides the built-in enum-derived suffix.
    pub custom_output_style_prompt: Option<String>,
    /// Absolute path to the working directory (injected as dynamic section).
    pub working_directory: Option<String>,
    /// Git repository context (branch, status, recent log) injected into the dynamic section.
    /// Empty string if not in a git repo or if git context gathering failed.
    pub git_context: String,
    /// Pre-built memory content from memdir (injected as dynamic section).
    pub memory_content: String,
    /// Custom system prompt (--system-prompt flag or settings).
    pub custom_system_prompt: Option<String>,
    /// Additional text appended after everything else (--append-system-prompt).
    pub append_system_prompt: Option<String>,
    /// If true and `custom_system_prompt` is set, the entire default prompt is
    /// replaced — only the custom text + dynamic boundary are emitted.
    pub replace_system_prompt: bool,
    /// Inject the coordinator-mode section.
    pub coordinator_mode: bool,
    /// Skip auto-injecting platform/shell/date env info (set true only in tests).
    pub skip_env_info: bool,

    /// Skills to inject into the cacheable section of the prompt.
    ///
    /// Each entry is a `(skill_name, skill_content)` pair assembled by the
    /// skill resolver before the model request is built. Content is injected
    /// as a `## Skill: <name>` block **before** the dynamic boundary so it
    /// is eligible for Anthropic prompt caching.
    ///
    /// If a skill also declares QA steps, the caller should append the
    /// formatted QA block to the **dynamic** section (after the boundary)
    /// so it reads as a live task constraint, not cached boilerplate.
    pub injected_skills: Vec<(String, String)>,

    /// Mandatory QA blocks to inject **after** the dynamic boundary.
    ///
    /// Each entry is a pre-formatted QA block string (produced by
    /// `skill_discovery::format_qa_block`). These are appended to the
    /// dynamic section so they appear as a hard per-task constraint.
    pub skill_qa_blocks: Vec<String>,
}

// ---------------------------------------------------------------------------
// Main assembly function
// ---------------------------------------------------------------------------

/// Build the complete system prompt string.
///
/// The returned string contains `SYSTEM_PROMPT_DYNAMIC_BOUNDARY` as an
/// internal marker.  Callers (e.g. `buildSystemPromptBlocks` in cc-query)
/// split on this marker to determine which portions are eligible for
/// Anthropic prompt-caching.
pub fn build_system_prompt(opts: &SystemPromptOptions) -> String {
    // Replace mode: skip all default sections.
    if opts.replace_system_prompt {
        if let Some(custom) = &opts.custom_system_prompt {
            return format!("{}\n\n{}", custom, SYSTEM_PROMPT_DYNAMIC_BOUNDARY);
        }
    }

    let prefix = opts.prefix.unwrap_or_else(|| {
        SystemPromptPrefix::detect(opts.is_non_interactive, opts.has_append_system_prompt)
    });

    // ------------------------------------------------------------------ //
    // CACHEABLE sections (before the dynamic boundary)                   //
    // ------------------------------------------------------------------ //

    // 1. Attribution header — OAuth provider overrides default prefix.
    //
    // Anthropic Claude Max specifically requires its sanctioned opener
    // (and SDK variants) verbatim, otherwise the OAuth endpoint returns
    // "OAuth authentication is currently not supported." Route AnthropicMax
    // through `claude_code_attribution_text` so the SDK / non-interactive
    // variants are honoured. Other OAuth providers keep using their own
    // identity strings.
    let attribution = match opts.oauth_provider {
        OAuthProvider::None => prefix.attribution_text().to_string(),
        OAuthProvider::AnthropicMax => prefix.claude_code_attribution_text().to_string(),
        other => other.identity_text().to_string(),
    };

    let mut parts: Vec<String> = vec![
        attribution,
        // 2. Core capabilities
        CORE_CAPABILITIES.to_string(),
        // 3. Tool use guidelines
        TOOL_USE_GUIDELINES.to_string(),
        // 4. Executing actions with care
        ACTIONS_SECTION.to_string(),
        // 5. Safety guidelines
        SAFETY_GUIDELINES.to_string(),
        // 6. Cyber-risk instruction (owned by safeguards - do not edit)
        CYBER_RISK_INSTRUCTION.to_string(),
    ];

    // 7. Output style (cacheable when non-Default; its content is stable)
    if let Some(style_text) = opts
        .custom_output_style_prompt
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| opts.output_style.prompt_suffix())
    {
        parts.push(format!("\n## Output Style\n{}", style_text));
    }

    // 8. Coordinator mode (cacheable: content is constant)
    if opts.coordinator_mode {
        parts.push(COORDINATOR_SYSTEM_PROMPT.to_string());
    }

    // 8.5. Injected skills (cacheable: skill content is static reference material)
    //
    // Skills are injected here — before the dynamic boundary — so the API can
    // cache them across turns. This matches the Perplexity Computer architecture
    // where skill context is loaded into the model's working memory before any
    // token is generated.
    for (skill_name, skill_content) in &opts.injected_skills {
        if !skill_content.trim().is_empty() {
            parts.push(format!(
                "\n## Skill: {}\n\n{}",
                skill_name,
                skill_content.trim()
            ));
        }
    }

    // 9. Custom system prompt addition (appended to cacheable block)
    if let Some(custom) = &opts.custom_system_prompt {
        parts.push(format!(
            "\n<custom_instructions>\n{}\n</custom_instructions>",
            custom
        ));
    }

    // Dynamic boundary marker
    parts.push(SYSTEM_PROMPT_DYNAMIC_BOUNDARY.to_string());

    // ------------------------------------------------------------------ //
    // DYNAMIC / UNCACHEABLE sections (after the boundary)                //
    // ------------------------------------------------------------------ //

    // 10. Environment info (platform, OS version, shell, date)
    if !opts.skip_env_info {
        parts.push(build_env_info_section(opts.working_directory.as_deref()));
    }

    // 11. Working directory (legacy XML tag kept for caching compat)
    if let Some(cwd) = &opts.working_directory {
        parts.push(format!("\n<working_directory>{}</working_directory>", cwd));
    }

    // 11.5. Git repository context
    if !opts.git_context.is_empty() {
        parts.push(format!(
            "\n<git_context>\n{}\n</git_context>",
            opts.git_context
        ));
    }

    // 12. Memory injection (from memdir)
    if !opts.memory_content.is_empty() {
        parts.push(format!("\n<memory>\n{}\n</memory>", opts.memory_content));
    }

    // 12.5. Mandatory QA blocks from auto-loaded skills
    //
    // These are intentionally in the DYNAMIC section so they register as a
    // live per-task constraint rather than background context. The emphatic
    // wording in `format_qa_block` makes them non-ignorable.
    for qa_block in &opts.skill_qa_blocks {
        if !qa_block.trim().is_empty() {
            parts.push(format!("\n{}", qa_block.trim()));
        }
    }

    // 13. Appended system prompt (--append-system-prompt)
    if let Some(append) = &opts.append_system_prompt {
        parts.push(format!("\n{}", append));
    }

    parts.join("\n")
}

/// Build the dynamic environment-info section injected after the boundary.
fn build_env_info_section(working_dir: Option<&str>) -> String {
    // Platform string
    let platform = if cfg!(target_os = "windows") {
        "win32"
    } else if cfg!(target_os = "macos") {
        "darwin"
    } else {
        "linux"
    };

    // OS version string
    let os_version = {
        #[cfg(target_os = "windows")]
        {
            // On Windows, use WINDIR env var existence as a proxy; actual version
            // would require winapi calls, so fall back to a readable label.
            std::env::var("OS").unwrap_or_else(|_| "Windows".to_string())
        }
        #[cfg(not(target_os = "windows"))]
        {
            // Use uname -sr via std::process for POSIX systems.
            std::process::Command::new("uname")
                .args(["-s", "-r"])
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| platform.to_string())
        }
    };

    // Shell detection
    let shell_env = std::env::var("SHELL").unwrap_or_default();
    let shell_name = if shell_env.contains("zsh") {
        "zsh"
    } else if shell_env.contains("bash") {
        "bash"
    } else if shell_env.contains("fish") {
        "fish"
    } else if cfg!(target_os = "windows") {
        "powershell"
    } else if shell_env.is_empty() {
        "unknown"
    } else {
        &shell_env
    };

    // Shell line: on Windows add Unix syntax note
    let shell_line = if cfg!(target_os = "windows") {
        format!("Shell: {} (use Unix shell syntax, not Windows — e.g., /dev/null not NUL, forward slashes in paths)", shell_name)
    } else {
        format!("Shell: {}", shell_name)
    };

    // Is git repo?
    let is_git = working_dir
        .map(|d| std::path::Path::new(d).join(".git").exists())
        .unwrap_or(false);

    // Today's date
    let today = {
        // Use chrono if available; otherwise fall back to env or skip
        // We avoid adding a new dep just for formatting, so use a rough ISO format.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        // Simple YYYY-MM-DD from seconds since epoch
        let days = now / 86400;
        let year_approx = 1970 + days / 365;
        // Not perfectly accurate but good enough for the system prompt context.
        // For exact dates a chrono dep would be needed; use SystemTime string as fallback.
        format!("{}", year_approx)
    };
    let _ = today; // suppress unused warning — date is included below via SystemTime

    // Build the section
    let cwd_line = working_dir
        .map(|d| format!("\nWorking directory: {}", d))
        .unwrap_or_default();

    format!(
        "\n<env>{}\nIs directory a git repo: {}\nPlatform: {}\nOS Version: {}\n{}\n</env>",
        cwd_line,
        if is_git { "Yes" } else { "No" },
        platform,
        os_version,
        shell_line,
    )
}

/// Gather git repository context for system prompt injection.
/// Returns an empty string if not in a git repo or if git commands fail.
/// Designed to be fast - each git command has a 2-second timeout.
pub fn gather_git_context(working_dir: &str) -> String {
    use std::fs;
    use std::process::{Command, Stdio};
    use std::thread;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    let dir = std::path::Path::new(working_dir);
    let run_raw = |args: &[&str]| -> Option<String> {
        // Route stdout to a temp file so large outputs cannot block on pipe capacity
        // while we poll try_wait() for timeout handling.
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()?
            .as_nanos();
        let tmp_path = std::env::temp_dir().join(format!(
            "mangocode_git_ctx_{}_{}.tmp",
            std::process::id(),
            ts
        ));
        let out_file = std::fs::File::create(&tmp_path).ok()?;

        let mut child = Command::new("git")
            .args(args)
            .current_dir(dir)
            .stdout(Stdio::from(out_file))
            .stderr(Stdio::null())
            .spawn()
            .ok()?;

        let cleanup = || {
            let _ = fs::remove_file(&tmp_path);
        };

        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    if !status.success() {
                        cleanup();
                        return None;
                    }

                    let output = fs::read_to_string(&tmp_path).ok()?;
                    cleanup();

                    return Some(output.trim().to_string());
                }
                Ok(None) => {
                    if Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        cleanup();
                        return None;
                    }
                    thread::sleep(Duration::from_millis(10));
                }
                Err(_) => {
                    cleanup();
                    return None;
                }
            }
        }
    };

    // Use git itself for repo detection (handles nested dirs and worktrees).
    if run_raw(&["rev-parse", "--is-inside-work-tree"]).as_deref() != Some("true") {
        return String::new();
    }

    let run = |args: &[&str]| -> Option<String> { run_raw(args).filter(|s| !s.is_empty()) };

    let mut parts = Vec::new();

    // Current branch
    if let Some(branch) = run(&["rev-parse", "--abbrev-ref", "HEAD"]) {
        parts.push(format!("Branch: {}", branch));
    }

    // Short status (uncommitted changes)
    match run_raw(&["status", "--porcelain", "--untracked-files=no"]) {
        Some(status) if status.is_empty() => {
            parts.push("Working tree: clean".to_string());
        }
        Some(status) => {
            let lines: Vec<&str> = status.lines().collect();
            let shown = if lines.len() > 15 {
                format!(
                    "{}\n  ... and {} more changed files",
                    lines[..15].join("\n"),
                    lines.len() - 15
                )
            } else {
                lines.join("\n")
            };
            parts.push(format!("Uncommitted changes:\n{}", shown));
        }
        None => {}
    }

    // Recent commits (last 5, one-line format)
    if let Some(log) = run(&["log", "--oneline", "-5", "--no-decorate"]) {
        parts.push(format!("Recent commits:\n{}", log));
    }

    // Repo root-level file listing for project structure awareness
    if let Some(ls) = run(&["ls-tree", "--name-only", "HEAD"]) {
        let files: Vec<&str> = ls.lines().collect();
        let shown = if files.len() > 20 {
            format!(
                "{}\n  ... and {} more",
                files[..20].join(", "),
                files.len() - 20
            )
        } else {
            files.join(", ")
        };
        parts.push(format!("Root files: {}", shown));
    }

    if parts.is_empty() {
        return String::new();
    }

    parts.join("\n")
}

// ---------------------------------------------------------------------------
// Static system prompt sections
// ---------------------------------------------------------------------------

const CORE_CAPABILITIES: &str = r#"
## Capabilities

You have access to powerful tools for software engineering tasks:
- **Read/Write files**: Read any file, write new files, edit existing files with precise diffs
- **Execute commands**: Run bash commands, PowerShell scripts, background processes
- **Search**: Glob patterns, regex grep, web search, file content search
- **Web**: Fetch URLs, search the internet
- **Agents**: Spawn parallel sub-agents for complex multi-step work
- **Memory**: Persistent notes across sessions via the memory system
- **MCP servers**: Connect to external tools and APIs via Model Context Protocol
- **Jupyter notebooks**: Read and edit notebook cells

## How to approach tasks

1. **Understand before acting**: Read relevant files before making changes
2. **Minimal changes**: Only modify what's needed. Don't refactor unrequested code.
3. **Verify**: Check your work with tests or by reading the result
4. **Communicate blockers**: If stuck, ask the user rather than guessing
"#;

const TOOL_USE_GUIDELINES: &str = r#"
## Tool use guidelines

- Use dedicated tools (Read, Edit, Glob, Grep) instead of bash equivalents
- For searches, prefer Grep over `grep`; prefer Glob over `find`
- Parallelize independent tool calls in a single response
- For file edits: always read the file first, then make targeted edits
- Bash commands timeout after 2 minutes; use background mode for long operations
"#;

const ACTIONS_SECTION: &str = r#"
## Executing actions with care

Carefully consider the reversibility and blast radius of actions. For actions
that are hard to reverse, affect shared systems, or could be risky or
destructive, check with the user before proceeding. Authorization stands for
the scope specified, not beyond. Match the scope of your actions to what was
actually requested.
"#;

const SAFETY_GUIDELINES: &str = r#"
## Safety guidelines

- Never delete files without explicit user confirmation
- Don't modify protected files (.gitconfig, .bashrc, .zshrc, .mcp.json, .claude.json)
- Be careful with destructive operations (rm -rf, DROP TABLE, etc.)
- Don't commit secrets, credentials, or API keys
- For ambiguous destructive actions, ask before proceeding
"#;

/// Security / cyber-risk instruction.
/// No security guardrails injected.
const CYBER_RISK_INSTRUCTION: &str = "";

const COORDINATOR_SYSTEM_PROMPT: &str = r#"
## Coordinator Mode

You are operating as an orchestrator. Spawn parallel worker agents using the Agent tool.
Each worker prompt must be fully self-contained. Synthesize findings before delegating
follow-up work. Use TaskCreate/TaskUpdate to track parallel work.
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_opts() -> SystemPromptOptions {
        SystemPromptOptions::default()
    }

    #[test]
    fn test_default_prompt_contains_boundary() {
        let prompt = build_system_prompt(&default_opts());
        assert!(
            prompt.contains(SYSTEM_PROMPT_DYNAMIC_BOUNDARY),
            "System prompt must contain the dynamic boundary marker"
        );
    }

    #[test]
    fn test_default_prompt_contains_attribution() {
        let prompt = build_system_prompt(&default_opts());
        assert!(
            prompt.contains("MangoCode"),
            "Default prompt must contain attribution"
        );
    }

    #[test]
    fn test_replace_system_prompt() {
        let opts = SystemPromptOptions {
            custom_system_prompt: Some("Custom only.".to_string()),
            replace_system_prompt: true,
            ..Default::default()
        };
        let prompt = build_system_prompt(&opts);
        assert!(prompt.starts_with("Custom only."));
        assert!(!prompt.contains("Capabilities"));
        assert!(prompt.contains(SYSTEM_PROMPT_DYNAMIC_BOUNDARY));
    }

    #[test]
    fn test_working_directory_in_dynamic_section() {
        let opts = SystemPromptOptions {
            working_directory: Some("/home/user/project".to_string()),
            ..Default::default()
        };
        let prompt = build_system_prompt(&opts);
        let boundary_pos = prompt.find(SYSTEM_PROMPT_DYNAMIC_BOUNDARY).unwrap();
        let cwd_pos = prompt.find("/home/user/project").unwrap();
        assert!(
            cwd_pos > boundary_pos,
            "Working directory must appear after the dynamic boundary"
        );
    }

    #[test]
    fn test_memory_content_in_dynamic_section() {
        let opts = SystemPromptOptions {
            memory_content: "- [test.md](test.md) — a test memory".to_string(),
            ..Default::default()
        };
        let prompt = build_system_prompt(&opts);
        let boundary_pos = prompt.find(SYSTEM_PROMPT_DYNAMIC_BOUNDARY).unwrap();
        let mem_pos = prompt.find("test.md").unwrap();
        assert!(
            mem_pos > boundary_pos,
            "Memory content must appear after the dynamic boundary"
        );
    }

    #[test]
    fn test_git_context_in_dynamic_section() {
        let opts = SystemPromptOptions {
            working_directory: Some("/home/user/project".to_string()),
            git_context:
                "Branch: main\nWorking tree: clean\nRecent commits:\nabc1234 Initial commit"
                    .to_string(),
            skip_env_info: true,
            ..Default::default()
        };
        let prompt = build_system_prompt(&opts);
        assert!(prompt.contains("<git_context>"));
        assert!(prompt.contains("Branch: main"));
        assert!(prompt.contains("abc1234 Initial commit"));

        // Verify it's in the dynamic section (after the boundary)
        let boundary_pos = prompt.find(SYSTEM_PROMPT_DYNAMIC_BOUNDARY).unwrap();
        let git_pos = prompt.find("<git_context>").unwrap();
        assert!(
            git_pos > boundary_pos,
            "git_context should be in the dynamic section"
        );
    }

    #[test]
    fn test_empty_git_context_not_injected() {
        let opts = SystemPromptOptions {
            working_directory: Some("/home/user/project".to_string()),
            git_context: String::new(),
            skip_env_info: true,
            ..Default::default()
        };
        let prompt = build_system_prompt(&opts);
        assert!(!prompt.contains("<git_context>"));
    }

    #[test]
    fn test_output_style_concise() {
        let opts = SystemPromptOptions {
            output_style: OutputStyle::Concise,
            ..Default::default()
        };
        let prompt = build_system_prompt(&opts);
        assert!(prompt.contains("maximally concise"));
    }

    #[test]
    fn test_output_style_default_has_no_suffix() {
        let opts = SystemPromptOptions {
            output_style: OutputStyle::Default,
            ..Default::default()
        };
        let prompt = build_system_prompt(&opts);
        // None of the style suffixes should appear
        assert!(!prompt.contains("maximally concise"));
        assert!(!prompt.contains("This user is learning"));
    }

    #[test]
    fn test_coordinator_mode_section() {
        let opts = SystemPromptOptions {
            coordinator_mode: true,
            ..Default::default()
        };
        let prompt = build_system_prompt(&opts);
        assert!(prompt.contains("Coordinator Mode"));
        assert!(prompt.contains("orchestrator"));
    }

    #[test]
    fn test_output_style_parse() {
        assert_eq!(OutputStyle::parse("concise"), OutputStyle::Concise);
        assert_eq!(OutputStyle::parse("FORMAL"), OutputStyle::Formal);
        assert_eq!(OutputStyle::parse("unknown"), OutputStyle::Default);
    }

    #[test]
    fn test_sdk_prefix_non_interactive_no_append() {
        let prefix = SystemPromptPrefix::detect(true, false);
        assert_eq!(prefix, SystemPromptPrefix::Sdk);
        assert!(prefix.attribution_text().contains("Claude agent"));
    }

    #[test]
    fn test_sdk_preset_prefix_non_interactive_with_append() {
        let prefix = SystemPromptPrefix::detect(true, true);
        assert_eq!(prefix, SystemPromptPrefix::SdkPreset);
        assert!(prefix.attribution_text().contains("Claude Agent SDK"));
    }

    #[test]
    fn test_clear_section_cache() {
        // Populate cache then clear it — should not panic.
        {
            let mut cache = section_cache().lock().unwrap();
            cache.insert("test_section".to_string(), Some("content".to_string()));
        }
        clear_system_prompt_sections();
        let cache = section_cache().lock().unwrap();
        assert!(cache.is_empty());
    }

    // ---- Phase 6: skill injection -------------------------------------------

    #[test]
    fn test_injected_skills_in_cacheable_section() {
        let opts = SystemPromptOptions {
            injected_skills: vec![(
                "rust-review".to_string(),
                "# Rust Review\nCheck for unwrap() misuse.".to_string(),
            )],
            ..Default::default()
        };
        let prompt = build_system_prompt(&opts);

        // Skill must appear before the dynamic boundary (cacheable zone)
        let boundary_pos = prompt.find(SYSTEM_PROMPT_DYNAMIC_BOUNDARY).unwrap();
        let skill_pos = prompt.find("## Skill: rust-review").unwrap();
        assert!(
            skill_pos < boundary_pos,
            "Injected skill must be in the cacheable section (before boundary)"
        );
        assert!(prompt.contains("Check for unwrap() misuse."));
    }

    #[test]
    fn test_multiple_injected_skills_ordered() {
        let opts = SystemPromptOptions {
            injected_skills: vec![
                (
                    "design-foundations".to_string(),
                    "Color palette rules.".to_string(),
                ),
                ("pptx".to_string(), "Slide generation steps.".to_string()),
            ],
            ..Default::default()
        };
        let prompt = build_system_prompt(&opts);
        let design_pos = prompt.find("## Skill: design-foundations").unwrap();
        let pptx_pos = prompt.find("## Skill: pptx").unwrap();
        let boundary_pos = prompt.find(SYSTEM_PROMPT_DYNAMIC_BOUNDARY).unwrap();

        // Both before boundary, and in insertion order
        assert!(design_pos < boundary_pos);
        assert!(pptx_pos < boundary_pos);
        assert!(
            design_pos < pptx_pos,
            "Dependencies should appear before the skill that depends on them"
        );
    }

    #[test]
    fn test_skill_qa_blocks_in_dynamic_section() {
        let qa_block = "## Required QA for this task (skill: pptx)\nYou MUST complete ALL steps below.\n1. Run markitdown output.pptx".to_string();
        let opts = SystemPromptOptions {
            skill_qa_blocks: vec![qa_block],
            ..Default::default()
        };
        let prompt = build_system_prompt(&opts);

        // QA block must appear AFTER the dynamic boundary (volatile section)
        let boundary_pos = prompt.find(SYSTEM_PROMPT_DYNAMIC_BOUNDARY).unwrap();
        let qa_pos = prompt.find("Required QA for this task").unwrap();
        assert!(
            qa_pos > boundary_pos,
            "QA enforcement block must be in the dynamic section (after boundary)"
        );
        assert!(prompt.contains("Run markitdown output.pptx"));
    }

    // ---- Claude Max OAuth: reference-exact opener -------------------------

    #[test]
    fn test_anthropic_max_opener_matches_reference_default() {
        let opts = SystemPromptOptions {
            oauth_provider: OAuthProvider::AnthropicMax,
            prefix: Some(SystemPromptPrefix::Cli),
            ..Default::default()
        };
        let prompt = build_system_prompt(&opts);
        assert!(
            prompt.starts_with("You are Claude Code, Anthropic's official CLI for Claude."),
            "Claude Max CLI opener must match reference DEFAULT_PREFIX byte-for-byte"
        );
        assert!(
            !prompt.contains("MangoCode with a Claude Max subscription"),
            "Reference opener must not contain the legacy MangoCode subscription line"
        );
    }

    #[test]
    fn test_anthropic_max_opener_matches_reference_sdk_preset() {
        let opts = SystemPromptOptions {
            oauth_provider: OAuthProvider::AnthropicMax,
            prefix: Some(SystemPromptPrefix::SdkPreset),
            ..Default::default()
        };
        let prompt = build_system_prompt(&opts);
        assert!(
            prompt.starts_with(
                "You are Claude Code, Anthropic's official CLI for Claude, running within the Claude Agent SDK."
            ),
            "Claude Max SDK-preset opener must match reference AGENT_SDK_CLAUDE_CODE_PRESET_PREFIX"
        );
    }

    #[test]
    fn test_anthropic_max_opener_matches_reference_sdk() {
        let opts = SystemPromptOptions {
            oauth_provider: OAuthProvider::AnthropicMax,
            prefix: Some(SystemPromptPrefix::Sdk),
            ..Default::default()
        };
        let prompt = build_system_prompt(&opts);
        assert!(
            prompt.starts_with("You are a Claude agent, built on Anthropic's Claude Agent SDK."),
            "Claude Max SDK opener must match reference AGENT_SDK_PREFIX"
        );
    }

    #[test]
    fn test_anthropic_max_opener_unaffected_by_vertex_or_bedrock_prefix() {
        // Vertex / Bedrock / remote still use the default Anthropic Max opener.
        for variant in [
            SystemPromptPrefix::Vertex,
            SystemPromptPrefix::Bedrock,
            SystemPromptPrefix::Remote,
        ] {
            let opts = SystemPromptOptions {
                oauth_provider: OAuthProvider::AnthropicMax,
                prefix: Some(variant),
                ..Default::default()
            };
            let prompt = build_system_prompt(&opts);
            assert!(
                prompt.starts_with("You are Claude Code, Anthropic's official CLI for Claude."),
                "Claude Max opener for {:?} must collapse to reference DEFAULT_PREFIX",
                variant
            );
        }
    }

    #[test]
    fn test_other_oauth_providers_still_use_identity_text() {
        // Codex must keep its dedicated identity string — only AnthropicMax
        // is pinned to the Anthropic-sanctioned opener.
        let opts = SystemPromptOptions {
            oauth_provider: OAuthProvider::OpenAiCodex,
            prefix: Some(SystemPromptPrefix::Cli),
            ..Default::default()
        };
        let prompt = build_system_prompt(&opts);
        assert!(prompt.contains("OpenAI Codex"));
        assert!(!prompt.starts_with("You are Claude Code,"));
    }

    #[test]
    fn test_empty_injected_skills_not_injected() {
        let opts = SystemPromptOptions {
            injected_skills: vec![("empty-skill".to_string(), "   ".to_string())],
            ..Default::default()
        };
        let prompt = build_system_prompt(&opts);
        assert!(
            !prompt.contains("## Skill: empty-skill"),
            "Empty skill content should not be injected"
        );
    }
}
